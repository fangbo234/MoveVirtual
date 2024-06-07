class GrandParentClass:
    def grandparent_method(self):
        print("This is a method from GrandParentClass.")

class ParentClass(GrandParentClass):
    def parent_method(self):
        print("This is a method from ParentClass.")

class ChildClass(ParentClass):
    def child_method(self):
        print("This is a method from ChildClass.")

c = ChildClass()
c.grandparent_method()  # This is a method from GrandParentClass.
c.parent_method()       # This is a method from ParentClass.
c.child_method()        # This is a method from ChildClass.

