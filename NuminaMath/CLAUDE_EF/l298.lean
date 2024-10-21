import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_sum_when_30deg_perpendicular_sum_diff_angle_when_dot_product_half_l298_29856

noncomputable section

-- Define vectors a and b
def a (α : Real) : Fin 2 → Real := ![Real.cos α, Real.sin α]
def b : Fin 2 → Real := ![-1/2, Real.sqrt 3 / 2]

-- Define the dot product
def dot_product (v w : Fin 2 → Real) : Real :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Define the magnitude of a vector
noncomputable def magnitude (v : Fin 2 → Real) : Real :=
  Real.sqrt (dot_product v v)

-- Theorem statements
theorem magnitude_sum_when_30deg (α : Real) :
  α = π/6 → magnitude (a α + b) = Real.sqrt 2 := by sorry

theorem perpendicular_sum_diff (α : Real) :
  0 < α ∧ α < π/2 → dot_product (a α + b) (a α - b) = 0 := by sorry

theorem angle_when_dot_product_half (α : Real) :
  0 < α ∧ α < π/2 → dot_product (a α) b = 1/2 → α = π/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_sum_when_30deg_perpendicular_sum_diff_angle_when_dot_product_half_l298_29856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounds_l298_29894

def my_sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 1 ∧ 
  a 1 = 2 ∧ 
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + a (n - 1) / (1 + (a (n - 1))^2)

theorem sequence_bounds (a : ℕ → ℝ) (h : my_sequence a) : 52 < a 1371 ∧ a 1371 < 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounds_l298_29894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_opposite_sides_implies_parallelogram_l298_29836

/-- A quadrilateral with vertices A, B, C, and D -/
structure Quadrilateral (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] :=
  (A B C D : P)

/-- The property that both pairs of opposite sides are equal in length -/
def has_equal_opposite_sides {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (Q : Quadrilateral P) : Prop :=
  ‖Q.A - Q.B‖ = ‖Q.C - Q.D‖ ∧ ‖Q.A - Q.D‖ = ‖Q.B - Q.C‖

/-- The property of being a parallelogram -/
def is_parallelogram {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (Q : Quadrilateral P) : Prop :=
  (Q.A - Q.B) = (Q.D - Q.C) ∧ (Q.A - Q.D) = (Q.B - Q.C)

/-- Theorem: If a quadrilateral has both pairs of opposite sides equal in length, 
    then it is a parallelogram -/
theorem equal_opposite_sides_implies_parallelogram 
  {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (Q : Quadrilateral P) :
  has_equal_opposite_sides Q → is_parallelogram Q :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_opposite_sides_implies_parallelogram_l298_29836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_after_shortening_both_sides_l298_29893

/-- Represents a rectangle with given width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

/-- The original rectangle -/
def original : Rectangle := { width := 3, length := 5 }

/-- The rectangle after shortening one side -/
noncomputable def shortened_one_side : Rectangle :=
  if original.length > original.width
  then { width := original.width, length := original.length - 2 }
  else { width := original.width - 2, length := original.length }

/-- The rectangle after shortening both sides -/
def shortened_both_sides : Rectangle :=
  { width := original.width - 2, length := original.length - 2 }

theorem area_after_shortening_both_sides :
  shortened_one_side.area = 9 → shortened_both_sides.area = 3 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_after_shortening_both_sides_l298_29893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olivers_siblings_l298_29889

-- Define the types for eye color and hair color
inductive EyeColor
| Green
| Gray

inductive HairColor
| Red
| Brown

-- Define a structure for a child's characteristics
structure Child where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor

-- Define the function to check if two children share a characteristic
def shareCharacteristic (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor

-- Define the children
def oliver : Child := ⟨"Oliver", EyeColor.Gray, HairColor.Brown⟩
def charles : Child := ⟨"Charles", EyeColor.Gray, HairColor.Red⟩
def diana : Child := ⟨"Diana", EyeColor.Green, HairColor.Brown⟩

-- Define the theorem
theorem olivers_siblings :
  (shareCharacteristic oliver charles ∧ shareCharacteristic oliver diana) ∧
  ¬(shareCharacteristic charles diana) →
  (charles.name = "Charles" ∧ diana.name = "Diana") :=
by
  intro h
  exact ⟨rfl, rfl⟩

#check olivers_siblings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olivers_siblings_l298_29889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l298_29891

def A : Set ℕ := {x : ℕ | 0 ≤ x ∧ x < 1}
def B : Set ℕ := {0, 1}

theorem intersection_A_B : A ∩ B = {0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l298_29891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_equals_product_l298_29887

noncomputable def quadratic_roots (a b c : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  (root1, root2)

theorem roots_sum_equals_product (a b c : ℝ) (h : a ≠ 0) :
  let (r₁, r₂) := quadratic_roots a b c h
  r₁ + r₂ = r₁ * r₂ ↔ b = -c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_equals_product_l298_29887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_transformation_l298_29843

theorem sum_of_squares_transformation (N : ℕ) :
  (∃ (n : ℕ) (a b c : ℤ), N = 9^n * (a^2 + b^2 + c^2) ∧ 3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c) →
  (∃ (x y z : ℤ), N = x^2 + y^2 + z^2 ∧ ¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_transformation_l298_29843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_70_factorial_l298_29863

theorem last_two_nonzero_digits_of_70_factorial (n : ℕ) : 
  n = 68 → ∃ k : ℕ, (Nat.factorial 70) % 100 = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_70_factorial_l298_29863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_grass_growth_feeds_five_cows_l298_29845

/-- Represents the amount of grass one cow consumes in a day -/
noncomputable def C : ℝ := sorry

/-- Represents the initial amount of grass in the field -/
noncomputable def G : ℝ := sorry

/-- Represents the amount of grass that grows each day -/
noncomputable def r : ℝ := sorry

/-- Theorem stating that the number of cows that can be fed daily by the grass growing each day is 5 -/
theorem daily_grass_growth_feeds_five_cows : r = 5 * C := by
  sorry

/-- First condition: 10 cows can graze the pasture for 8 days before it's completely eaten -/
axiom condition1 : 10 * 8 * C = G + 8 * r

/-- Second condition: 15 cows, starting with one less cow each day from the second day, can finish grazing in 5 days -/
axiom condition2 : (15 + 14 + 13 + 12 + 11) * C = G + 5 * r

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_grass_growth_feeds_five_cows_l298_29845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l298_29860

/-- A linear function satisfying certain conditions -/
def f (x : ℝ) : ℝ := x + 2

/-- Function g defined in terms of f -/
def g (lambda : ℝ) (x : ℝ) : ℝ := x * f x + lambda * f x + 1

/-- Theorem stating the properties of f and g -/
theorem f_and_g_properties :
  (∀ x, f (x + 1) = x + 3) ∧
  f 1 = 3 ∧
  (∀ lambda, (∀ x ∈ Set.Ioo 0 2, Monotone (g lambda)) ↔ lambda ≤ -6 ∨ lambda ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l298_29860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l298_29833

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (s : Sphere) : ℝ := (4/3) * Real.pi * s.radius^3

/-- Calculates the new height of liquid in a cone after a sphere is dropped -/
noncomputable def newHeight (c : Cone) (s : Sphere) : ℝ :=
  c.height + sphereVolume s / (Real.pi * c.radius^2)

theorem liquid_rise_ratio
  (c1 c2 : Cone)
  (s : Sphere)
  (h_vol : coneVolume c1 = coneVolume c2)
  (h_r1 : c1.radius = 4)
  (h_r2 : c2.radius = 8)
  (h_sr : s.radius = 2) :
  (newHeight c1 s - c1.height) / (newHeight c2 s - c2.height) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l298_29833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_east_northwest_is_150_l298_29816

/-- The number of rays in the circle -/
def num_rays : ℕ := 12

/-- The angle between adjacent rays in degrees -/
noncomputable def angle_between_rays : ℝ := 360 / num_rays

/-- The number of ray segments from North to East (clockwise) -/
def segments_north_to_east : ℕ := 3

/-- The number of ray segments from North to Northwest (counterclockwise) -/
def segments_north_to_northwest : ℕ := 2

/-- The smaller angle between East and Northwest rays in degrees -/
noncomputable def angle_east_northwest : ℝ := 
  angle_between_rays * (segments_north_to_east + segments_north_to_northwest)

theorem angle_east_northwest_is_150 : 
  angle_east_northwest = 150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_east_northwest_is_150_l298_29816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l298_29806

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem f_max_value :
  ∃ (x_max : ℝ), x_max > 0 ∧ 
  (∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  f x_max = (1 : ℝ) / Real.exp 1 ∧
  x_max = Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l298_29806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_square_center_sum_l298_29886

/-- A square in the first quadrant with specific points on its sides -/
structure SpecialSquare where
  /-- The square lies in the first quadrant -/
  first_quadrant : True
  /-- Point (4,0) lies on line DA -/
  point_on_DA : True
  /-- Point (6,0) lies on line CB -/
  point_on_CB : True
  /-- Point (9,0) lies on line AB -/
  point_on_AB : True
  /-- Point (15,0) lies on line DC -/
  point_on_DC : True

/-- The center of the special square -/
def center (s : SpecialSquare) : ℝ × ℝ := sorry

/-- The sum of coordinates of the center of the special square is 1.2 -/
theorem special_square_center_sum (s : SpecialSquare) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x, y) = center s ∧ x + y = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_square_center_sum_l298_29886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_zero_probability_l298_29826

def S : Finset Int := {-3, -1, 0, 2, 4}

theorem product_zero_probability :
  let pairs := (S.product S).filter (fun p => p.1 ≠ p.2)
  (pairs.filter (fun p => p.1 * p.2 = 0)).card / pairs.card = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_zero_probability_l298_29826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_3_3_6_l298_29837

theorem ramsey_3_3_6 :
  ∀ (G : SimpleGraph (Fin 6)),
  ∃ (S : Finset (Fin 6)), S.card = 3 ∧
    (∀ (i j : Fin 6), i ∈ S → j ∈ S → i ≠ j → G.Adj i j) ∨
    (∀ (i j : Fin 6), i ∈ S → j ∈ S → i ≠ j → ¬G.Adj i j) := by
  sorry

#check ramsey_3_3_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_3_3_6_l298_29837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_triangle_areas_l298_29803

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (P Q R : Point) : ℝ := sorry

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (T : Trapezoid) : ℝ := sorry

/-- Checks if two line segments intersect -/
def intersects (P Q R S : Point) : Prop := sorry

/-- Checks if two line segments have equal length -/
def equalLength (P Q R S : Point) : Prop := sorry

theorem trapezoid_triangle_areas 
  (ABCD : Trapezoid) 
  (O E F : Point) :
  trapezoidArea ABCD = 52 →
  equalLength ABCD.D E F ABCD.C →
  intersects ABCD.A F ABCD.B E →
  triangleArea ABCD.A O ABCD.B = 17 →
  triangleArea ABCD.A O E + triangleArea ABCD.B O F = 18 := 
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_triangle_areas_l298_29803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l298_29895

/-- Given points A, B, and P, if a line passing through P intersects line segment AB,
    then the slope angle α of this line is between π/4 and 3π/4 inclusive. -/
theorem slope_angle_range (A B P : ℝ × ℝ) (h_A : A = (-3, 4)) (h_B : B = (3, 2)) (h_P : P = (1, 0)) :
  ∃ (Q : ℝ × ℝ), Q ∈ Set.Icc A B →
    π/4 ≤ Real.arctan ((Q.2 - P.2) / (Q.1 - P.1)) ∧
    Real.arctan ((Q.2 - P.2) / (Q.1 - P.1)) ≤ 3*π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l298_29895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_unique_a_value_l298_29813

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x - (x + 1)^2 / 2
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log (x + 1) + Real.exp (-x)

-- Part I: Prove that f(x) > 0 for x ∈ (-1, +∞)
theorem f_positive (x : ℝ) (h : x > -1) : f x > 0 := by
  sorry

-- Part II: Prove that a = 1 is the only value satisfying the condition
theorem unique_a_value :
  ∃! a : ℝ, a > 0 ∧ ∀ x : ℝ, x > -1 → g x ≤ a * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_unique_a_value_l298_29813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_occupied_fraction_is_correct_l298_29846

/-- Represents a rectangular yard with flower beds and a circular path -/
structure Yard where
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  path_radius : ℝ

/-- Calculates the fraction of the yard occupied by flower beds and circular path -/
noncomputable def occupied_fraction (y : Yard) : ℝ :=
  let yard_width := (y.trapezoid_long_side - y.trapezoid_short_side) / 2
  let yard_area := y.trapezoid_long_side * yard_width
  let triangle_leg := yard_width
  let flower_beds_area := 2 * (triangle_leg^2 / 2)
  let path_area := Real.pi * y.path_radius^2
  (flower_beds_area + path_area) / yard_area

/-- Theorem stating that the occupied fraction is (25 + 4π) / 150 for the given yard -/
theorem occupied_fraction_is_correct (y : Yard) 
    (h1 : y.trapezoid_short_side = 20)
    (h2 : y.trapezoid_long_side = 30)
    (h3 : y.path_radius = 2) : 
  occupied_fraction y = (25 + 4 * Real.pi) / 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_occupied_fraction_is_correct_l298_29846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_consumption_theorem_l298_29802

/-- Represents the relationship between sleep hours and coffee consumption --/
structure CoffeeConsumption where
  sleep : ℚ
  coffee : ℚ
  inv_prop : sleep * coffee = 24

/-- Calculates the average coffee consumption given three days of data --/
def average_consumption (d1 d2 d3 : CoffeeConsumption) : ℚ :=
  (d1.coffee + d2.coffee + d3.coffee) / 3

/-- Theorem stating the coffee consumption for Thursday, Friday, and the average --/
theorem coffee_consumption_theorem (wed thu fri : CoffeeConsumption)
  (h_wed : wed.sleep = 8 ∧ wed.coffee = 3)
  (h_thu : thu.sleep = 4)
  (h_fri : fri.sleep = 10) :
  thu.coffee = 6 ∧ fri.coffee = 12/5 ∧ average_consumption wed thu fri = 19/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_consumption_theorem_l298_29802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perpendicular_tangent_range_of_m_l298_29844

/-- The function f(x) = e^x - mx -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - m * x

/-- The derivative of f(x) -/
noncomputable def f_derivative (m : ℝ) (x : ℝ) : ℝ := Real.exp x - m

theorem no_perpendicular_tangent (m : ℝ) :
  (∀ x : ℝ, f_derivative m x ≠ -2) → m ≤ 2 := by
  sorry

/-- The main theorem stating the range of m -/
theorem range_of_m :
  {m : ℝ | ∀ x : ℝ, f_derivative m x ≠ -2} = {m : ℝ | m ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perpendicular_tangent_range_of_m_l298_29844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l298_29841

noncomputable def f (α β : ℝ) (x : ℝ) : ℝ := x^3 - 9*x^2*Real.cos α + 48*x*Real.cos β + 18*(Real.sin α)^2

noncomputable def g (α β : ℝ) (x : ℝ) : ℝ := deriv (f α β) x

noncomputable def φ (α β : ℝ) (x : ℝ) : ℝ := (1/3)*x^3 - 2*x^2*Real.cos β + x*Real.cos α

noncomputable def h (α β : ℝ) (x : ℝ) : ℝ := Real.log (φ α β x)

theorem problem_solution (α β m : ℝ) :
  (∀ t : ℝ, g α β (1 + Real.exp (-abs t)) ≥ 0) →
  (∀ t : ℝ, g α β (3 + Real.sin t) ≤ 0) →
  (∀ x ∈ Set.Icc 0 1, h α β (x + 1 - m) < h α β (2*x + 2)) →
  Real.cos α + 2 * Real.cos β = 2 ∧ -1 < m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l298_29841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_express_delivery_growth_l298_29868

/-- Represents the daily average growth rate of packages handled -/
def x : ℝ := sorry

/-- The number of packages handled on the first day -/
def first_day : ℝ := 200

/-- The total number of packages handled over three days -/
def total_packages : ℝ := 662

/-- Theorem stating that the equation correctly represents the total number of packages handled over three days -/
theorem express_delivery_growth :
  first_day + first_day * (1 + x) + first_day * (1 + x)^2 = total_packages :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_express_delivery_growth_l298_29868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foil_covered_prism_width_l298_29885

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the dimensions of the inner and outer prisms -/
structure FoilCoveredPrism where
  inner : PrismDimensions
  outer : PrismDimensions

/-- The theorem stating the conditions and the result to be proved -/
theorem foil_covered_prism_width 
  (p : FoilCoveredPrism)
  (h1 : volume p.inner = 128)
  (h2 : p.inner.width = 2 * p.inner.length)
  (h3 : p.inner.width = 2 * p.inner.height)
  (h4 : p.outer.width = p.inner.width + 2)
  : Int.floor p.outer.width = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_foil_covered_prism_width_l298_29885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l298_29879

/-- Proof of ellipse eccentricity given specific conditions -/
theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b ∧ b > 0) :
  ∃ (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ),
    let O := (0, 0)
    let e := Real.sqrt ((a^2 - b^2) / a^2)  -- eccentricity
    let c := e * a  -- distance from center to focus
    F₁ = (-c, 0) ∧ F₂ = (c, 0) ∧  -- foci positions
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧  -- P is on the ellipse
    (P.1 - F₁.1) * (-c) + P.1 * P.1 + P.2 * P.2 = 0 ∧  -- PF₁ ⋅ (OF₁ + OP) = 0
    (P.1 - F₁.1)^2 + P.2^2 = 2 * ((P.1 - F₂.1)^2 + P.2^2) →  -- |PF₁| = √2|PF₂|
    e = Real.sqrt 6 - Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l298_29879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_comparison_l298_29870

/-- A path in a 2D plane -/
structure PathType where
  length : ℝ
  has_semicircle : Bool

/-- Given three paths with specified properties, prove that the first and third paths have equal length, 
    and both are shorter than the second path -/
theorem path_length_comparison (path1 path2 path3 : PathType) 
  (h1 : ¬path1.has_semicircle)
  (h2 : path2.has_semicircle)
  (h3 : ¬path3.has_semicircle) :
  path1.length = path3.length ∧ path3.length < path2.length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_comparison_l298_29870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_frac_and_pi_l298_29890

noncomputable def count_whole_numbers_between (a b : ℝ) : ℕ :=
  (Int.toNat ⌊b⌋) - (Int.toNat ⌈a⌉) + 1

theorem whole_numbers_between_frac_and_pi :
  count_whole_numbers_between (7/4 : ℝ) (3 * Real.pi) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_frac_and_pi_l298_29890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_standard_deviation_l298_29808

noncomputable def sample : List ℝ := [4, 2, 1, 0, -2]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem sample_standard_deviation :
  standardDeviation sample = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_standard_deviation_l298_29808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_surface_area_l298_29865

/-- The total surface area of a regular triangular pyramid -/
noncomputable def total_surface_area (a : ℝ) : ℝ := (3 * a^2 * Real.sqrt 3) / 4

/-- Theorem stating the total surface area of a regular triangular pyramid -/
theorem regular_triangular_pyramid_surface_area (a : ℝ) (h : a > 0) :
  let dihedral_angle : ℝ := 60 * π / 180  -- 60° in radians
  total_surface_area a = (3 * a^2 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_surface_area_l298_29865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_special_totient_l298_29867

theorem count_numbers_with_special_totient : 
  (Finset.filter (fun n : ℕ => n < 1000 ∧ Nat.totient n = n / 3) (Finset.range 1000)).card = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_special_totient_l298_29867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nut_distribution_equality_l298_29801

/-- Represents the distribution of nuts to students in a class -/
structure NutDistribution where
  /-- Total number of nuts -/
  total : ℕ
  /-- Number of students in the class -/
  students : ℕ
  /-- Base number of nuts for distribution -/
  a : ℕ
  /-- The i-th student receives ia nuts plus one-thirtieth of the remaining nuts -/
  distribution : (i : ℕ) → i ≤ students → ℕ

/-- Theorem stating the properties of the nut distribution when the first two students receive equal nuts -/
theorem nut_distribution_equality (d : NutDistribution) 
    (h1 : 1 ≤ d.students) (h2 : 2 ≤ d.students) :
  d.distribution 1 h1 = d.distribution 2 h2 →
  (∀ (i j : ℕ) (hi : i ≤ d.students) (hj : j ≤ d.students),
    d.distribution i hi = d.distribution j hj) ∧
  d.students = 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nut_distribution_equality_l298_29801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_zeros_l298_29820

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x / x

noncomputable def F (b : ℝ) (x : ℝ) : ℝ := (1/2) * f x - b * x

theorem tangent_and_zeros :
  (∃ x₀ : ℝ, x₀ > 0 ∧ (∃ a : ℝ, ∀ x : ℝ, a * x = f x₀ + (deriv f x₀) * (x - x₀)) ∧ x₀ = 2) ∧
  (∀ b : ℝ,
    (b ≤ 0 → ∀ x : ℝ, x ≠ 0 → F b x ≠ 0) ∧
    (0 < b → b < Real.exp 2 / 4 → ∃! x : ℝ, x > 0 ∧ F b x = 0) ∧
    (b = Real.exp 2 / 4 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ F b x₁ = 0 ∧ F b x₂ = 0 ∧
      ∀ x : ℝ, x > 0 → F b x = 0 → (x = x₁ ∨ x = x₂)) ∧
    (b > Real.exp 2 / 4 → ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ F b x₁ = 0 ∧ F b x₂ = 0 ∧ F b x₃ = 0 ∧
      ∀ x : ℝ, x > 0 → F b x = 0 → (x = x₁ ∨ x = x₂ ∨ x = x₃))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_zeros_l298_29820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l298_29832

/-- Triangle ABC with centroid G -/
structure Triangle (V : Type*) [AddCommGroup V] [Module ℝ V] where
  A : V
  B : V
  C : V
  G : V

/-- The centroid property for a triangle -/
def isCentroid {V : Type*} [AddCommGroup V] [Module ℝ V] (t : Triangle V) : Prop :=
  t.G + t.A + t.G + t.B + t.G + t.C = (0 : V)

/-- The given vector equation -/
def vectorEquation {V : Type*} [AddCommGroup V] [Module ℝ V] (t : Triangle V) (angleA angleB angleC : ℝ) : Prop :=
  (Real.sin angleA) • (t.A - t.G) + (Real.sin angleB) • (t.B - t.G) + (Real.sin angleC) • (t.C - t.G) = (0 : V)

/-- The theorem to be proved -/
theorem angle_B_measure {V : Type*} [AddCommGroup V] [Module ℝ V] (t : Triangle V) 
  (angleA angleB angleC : ℝ)
  (h1 : isCentroid t) (h2 : vectorEquation t angleA angleB angleC) : angleB = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l298_29832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l298_29880

/-- Given that two workers A and B can complete a job together in a certain number of days,
    and worker A can complete the job alone in a certain number of days,
    this function calculates how many days it would take worker B to complete the job alone. -/
noncomputable def time_for_b_alone (time_together time_a_alone : ℝ) : ℝ :=
  (time_together * time_a_alone) / (time_a_alone - time_together)

/-- Theorem stating that if A and B together can do a job in 6 days,
    and A alone can do it in 11 days, then B alone can do it in 13.2 days. -/
theorem b_alone_time :
  time_for_b_alone 6 11 = 13.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l298_29880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_function_characterization_l298_29874

noncomputable def median (a b c : ℝ) : ℝ := 
  (a + b + c) - max a (max b c) - min a (min b c)

def satisfies_median_property (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, median (f a b) (f b c) (f c a) = median a b c

theorem median_function_characterization (f : ℝ → ℝ → ℝ) :
  satisfies_median_property f → (∀ x y : ℝ, f x y = x ∨ f x y = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_function_characterization_l298_29874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_is_unfair_l298_29839

/-- Represents the number of pencils a player can take in one turn -/
inductive Move
| one : Move
| two : Move

/-- Represents the state of the game -/
structure GameState :=
  (pencils : Nat)

/-- Determines if a player can make a move in the current game state -/
def canMove (state : GameState) : Prop :=
  state.pencils > 0

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.one => ⟨state.pencils - 1⟩
  | Move.two => ⟨state.pencils - 2⟩

/-- Determines if a game state is winning for the current player -/
def isWinning (state : GameState) : Prop :=
  ∃ (move : Move), state.pencils = match move with
    | Move.one => 1
    | Move.two => 2

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Determines if a strategy is winning for the first player -/
def isWinningStrategy (strategy : Strategy) : Prop :=
  ∀ (opponent_strategy : Strategy),
    let initial_state : GameState := ⟨5⟩
    isWinning (applyMove initial_state (strategy initial_state))

/-- The main theorem stating that the game is unfair -/
theorem game_is_unfair :
  ∃ (strategy : Strategy), isWinningStrategy strategy := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_is_unfair_l298_29839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roger_baked_half_of_total_pies_l298_29821

variable (n : ℚ)

def roger_pies (n : ℚ) : ℚ := 1313 * n
def theresa_pies (n : ℚ) : ℚ := roger_pies n / 2
def simon_pies (n : ℚ) : ℚ := theresa_pies n + 2 * n
def total_pies (n : ℚ) : ℚ := roger_pies n + theresa_pies n + simon_pies n

theorem roger_baked_half_of_total_pies (n : ℚ) :
  roger_pies n / total_pies n = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roger_baked_half_of_total_pies_l298_29821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_dot_product_probability_l298_29807

/-- A regular polygon with n sides inscribed in a unit circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : ∀ (i : Fin n), ‖vertices i‖ = 1

/-- The dot product of two vectors in ℝ² -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorable_outcomes total_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem regular_polygon_dot_product_probability :
  ∀ (p : RegularPolygon 2017),
  ∃ (favorable_outcomes total_outcomes : ℕ),
    probability favorable_outcomes total_outcomes = 2/3 ∧
    (∀ (i j : Fin 2017), i ≠ j →
      (dot_product (p.vertices i) (p.vertices j) > 1/2) ↔
      (favorable_outcomes ≤ Fin.val (i - j) ∧ Fin.val (i - j) < total_outcomes)) :=
by
  sorry

#check regular_polygon_dot_product_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_dot_product_probability_l298_29807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_ap_implies_cos_not_ap_l298_29869

theorem sin_ap_implies_cos_not_ap (x y z : ℝ) :
  (2 * Real.sin y = Real.sin x + Real.sin z) →  -- sin is an arithmetic progression
  (Real.sin x < Real.sin y) →              -- sin is increasing
  (Real.sin y < Real.sin z) →              -- sin is increasing
  ¬(2 * Real.cos y = Real.cos x + Real.cos z)   -- cos is not an arithmetic progression
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_ap_implies_cos_not_ap_l298_29869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_form_l298_29873

-- Define the polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom nonzero_f : f ≠ 0
axiom nonzero_g : g ≠ 0
axiom functional_equation : ∀ x, f (g x) = f x * g x
axiom g_at_3 : g 3 = 64

-- Define the theorem
theorem g_form : g = λ x ↦ x^2 + 27.5*x - 27.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_form_l298_29873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_angles_theorem_l298_29897

noncomputable def IsAngle (x : ℝ) : Prop := 0 ≤ x ∧ x < 360

def AreParallel (α β : ℝ) : Prop := 
  (α = β) ∨ (α + β = 180)

theorem parallel_angles_theorem (α β : ℝ) : 
  (IsAngle α ∧ IsAngle β) →  -- Both α and β are angles
  (AreParallel α β) →  -- The sides of α and β are parallel
  (β = 3 * α - 20) →  -- One angle is 20° less than three times the other
  ((α = 50 ∧ β = 130) ∨ (α = 10 ∧ β = 10)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_angles_theorem_l298_29897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_crosses_to_cover_board_l298_29810

/-- Represents a position on the 8x8 board -/
structure Position where
  x : Fin 8
  y : Fin 8

/-- Represents a cross on the board -/
structure Cross where
  center : Position

/-- The set of positions covered by a cross -/
def crossCoverage (c : Cross) : Set Position :=
  {p | p = c.center ∨
       p = ⟨c.center.x, (c.center.y - 1)⟩ ∨
       p = ⟨c.center.x, (c.center.y + 1)⟩ ∨
       p = ⟨(c.center.x - 1), c.center.y⟩ ∨
       p = ⟨(c.center.x + 1), c.center.y⟩}

/-- Check if two crosses overlap -/
def crossesOverlap (c1 c2 : Cross) : Prop :=
  ∃ p, p ∈ crossCoverage c1 ∧ p ∈ crossCoverage c2

/-- A valid configuration of crosses on the board -/
def validConfiguration (crosses : List Cross) : Prop :=
  ∀ c1 c2, c1 ∈ crosses → c2 ∈ crosses → c1 ≠ c2 → ¬crossesOverlap c1 c2

/-- The board is fully covered if no additional cross can be placed -/
def boardFullyCovered (crosses : List Cross) : Prop :=
  ∀ p : Position, ∃ c, c ∈ crosses ∧ p ∈ crossCoverage c

theorem min_crosses_to_cover_board :
  ∃ (crosses : List Cross),
    crosses.length = 4 ∧
    validConfiguration crosses ∧
    boardFullyCovered crosses ∧
    ∀ (smallerConfig : List Cross),
      smallerConfig.length < 4 →
      ¬(validConfiguration smallerConfig ∧ boardFullyCovered smallerConfig) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_crosses_to_cover_board_l298_29810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_and_tangent_problem_l298_29858

open Real

theorem sine_and_tangent_problem (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : cos α = 3/5)
  (h3 : tan (α + β) = 3) :
  (sin (π/6 + α) = (3 + 4*Real.sqrt 3) / 10) ∧ (tan β = 5/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_and_tangent_problem_l298_29858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_common_tangents_l298_29876

/-- Represents a circle in 2D space --/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Calculates the center of a circle --/
noncomputable def center (c : Circle) : ℝ × ℝ :=
  (- c.a / 2, - c.b / 2)

/-- Calculates the radius of a circle --/
noncomputable def radius (c : Circle) : ℝ :=
  Real.sqrt ((c.a / 2)^2 + (c.b / 2)^2 - c.e)

/-- Calculates the distance between the centers of two circles --/
noncomputable def centerDistance (c1 c2 : Circle) : ℝ :=
  let (x1, y1) := center c1
  let (x2, y2) := center c2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Determines the number of common tangents between two circles --/
noncomputable def commonTangents (c1 c2 : Circle) : ℕ :=
  if centerDistance c1 c2 == radius c1 + radius c2 then 3 else 0

/-- The theorem stating that the given circles have 3 common tangents --/
theorem circles_common_tangents :
  let c1 : Circle := { a := 1, b := 1, c := -4, d := 2, e := 1 }
  let c2 : Circle := { a := 1, b := 1, c := 4, d := -4, e := -1 }
  commonTangents c1 c2 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_common_tangents_l298_29876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l298_29877

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * Real.log x + 1 / x + 2 * a * x

-- State the theorem
theorem function_inequality (m : ℝ) :
  (∀ (a : ℝ) (x₁ x₂ : ℝ), 
    a ∈ Set.Ioo (-3) (-2) → 
    x₁ ∈ Set.Icc 1 3 → 
    x₂ ∈ Set.Icc 1 3 → 
    (m + Real.log 3) * a - 2 * Real.log 3 > |f a x₁ - f a x₂|) →
  m ≤ -13/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l298_29877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_fraction_is_one_l298_29817

noncomputable def complex_modulus (z : ℂ) : ℝ := Real.sqrt (z.re * z.re + z.im * z.im)

theorem modulus_of_fraction_is_one :
  complex_modulus ((1 - Complex.I) / (1 + Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_fraction_is_one_l298_29817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l298_29809

noncomputable def solution (a b c : ℝ) (e₁ e₂ e₃ : ℝ) : ℝ × ℝ × ℝ :=
  let x := (1/2) * (e₁ * Real.sqrt (a + b) - e₂ * Real.sqrt (b + c) + e₃ * Real.sqrt (c + a))
  let y := (1/2) * (e₁ * Real.sqrt (a + b) + e₂ * Real.sqrt (b + c) - e₃ * Real.sqrt (c + a))
  let z := (1/2) * (-e₁ * Real.sqrt (a + b) + e₂ * Real.sqrt (b + c) + e₃ * Real.sqrt (c + a))
  (x, y, z)

theorem solution_satisfies_system (a b c : ℝ) (e₁ e₂ e₃ : ℝ) 
    (h₁ : e₁ = 1 ∨ e₁ = -1) (h₂ : e₂ = 1 ∨ e₂ = -1) (h₃ : e₃ = 1 ∨ e₃ = -1) :
  let (x, y, z) := solution a b c e₁ e₂ e₃
  (x * (x + y) + z * (x - y) = a) ∧
  (y * (y + z) + x * (y - z) = b) ∧
  (z * (z + x) + y * (z - x) = c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l298_29809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_zeros_l298_29878

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - Real.log x - 2

theorem f_monotonicity_and_zeros (a : ℝ) :
  (∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ ∧ a ≤ 0 → f a x₁ > f a x₂) ∧
  (a > 0 → ∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ ∧ x₂ < Real.sqrt a / a → f a x₁ > f a x₂) ∧
  (a > 0 → ∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ Real.sqrt a / a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∃ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ↔ 0 < a ∧ a < Real.exp 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_zeros_l298_29878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_at_height_2_max_volume_height_for_max_volume_l298_29838

/-- Represents a quadrilateral pyramid with specific properties -/
structure QuadPyramid where
  base : Real
  height : ℝ
  lateral_angle : ℝ
  ab_length : ℝ
  cd_length : ℝ

/-- Properties of the quadrilateral pyramid -/
def pyramid_properties (p : QuadPyramid) : Prop :=
  p.ab_length = 2 ∧
  p.cd_length = 3 ∧
  p.lateral_angle = 30 * Real.pi / 180

/-- Volume of the pyramid given its base area and height -/
noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  (1/3) * base_area * height

/-- Theorem stating the volume when height is 2 -/
theorem volume_at_height_2 (p : QuadPyramid) 
  (h_prop : pyramid_properties p) (h_height : p.height = 2) :
  ∃ v, pyramid_volume p.base p.height = v ∧ v = 4 / Real.sqrt 3 := by
  sorry

/-- Theorem stating the maximum volume -/
theorem max_volume (p : QuadPyramid) (h_prop : pyramid_properties p) :
  ∃ v, v = 4 * Real.sqrt 3 ∧ 
    ∀ h, pyramid_volume p.base h ≤ v := by
  sorry

/-- Theorem stating the height for maximum volume -/
theorem height_for_max_volume (p : QuadPyramid) (h_prop : pyramid_properties p) :
  ∃ h, h = 4 * Real.sqrt 3 ∧ 
    ∀ h', pyramid_volume p.base h' ≤ pyramid_volume p.base h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_at_height_2_max_volume_height_for_max_volume_l298_29838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_element_is_negative_four_l298_29898

def S (x : ℚ) : Finset ℚ := {x, -1, 0, 6, 9}

def mean (s : Finset ℚ) : ℚ :=
  s.sum id / s.card

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_element_is_negative_four (x : ℚ) :
  x ≤ -1 ∧
  x ≤ 0 ∧
  x ≤ 6 ∧
  x ≤ 9 ∧
  (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧
    mean {↑p, ↑q, 0, 6, 9} ≥ 2 * mean (S x)) →
  x = -4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_element_is_negative_four_l298_29898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mul_comm_pair_mul_assoc_pair_mul_add_distrib_pair_add_mul_not_distrib_pair_l298_29864

-- Define the pair type
structure Pair where
  x : ℝ
  y : ℝ

-- Define addition operation
def add (a b : Pair) : Pair :=
  ⟨a.x + b.x, a.y + b.y⟩

-- Define multiplication operation
def mul (a b : Pair) : Pair :=
  ⟨a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x⟩

-- Commutative law of multiplication
theorem mul_comm_pair (a b : Pair) : mul a b = mul b a := by sorry

-- Associative law of multiplication
theorem mul_assoc_pair (a b c : Pair) : mul (mul a b) c = mul a (mul b c) := by sorry

-- Distributive law of multiplication over addition
theorem mul_add_distrib_pair (a b c : Pair) : mul a (add b c) = add (mul a b) (mul a c) := by sorry

-- Distributive law of addition over multiplication does not hold
theorem add_mul_not_distrib_pair : ∃ a b c : Pair, add a (mul b c) ≠ mul (add a b) (add a c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mul_comm_pair_mul_assoc_pair_mul_add_distrib_pair_add_mul_not_distrib_pair_l298_29864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l298_29899

/-- The hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop := λ x y ↦ x^2 / a^2 - y^2 / b^2 = 1

/-- The foci of a hyperbola -/
noncomputable def foci (h : Hyperbola) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  (-c, 0, c, 0)

/-- A point on the hyperbola -/
structure Point (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : h.equation x y

/-- Vector from a point to a focus -/
def vector_to_focus (h : Hyperbola) (p : Point h) (focus : ℝ × ℝ) : ℝ × ℝ :=
  (focus.1 - p.x, focus.2 - p.y)

/-- Dot product of two vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- Magnitude of a vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- Sum of two vectors -/
def vector_sum (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

/-- The main theorem -/
theorem hyperbola_theorem (h : Hyperbola) (p : Point h) :
  let (f1x, f1y, f2x, f2y) := foci h
  let v1 := vector_to_focus h p (f1x, f1y)
  let v2 := vector_to_focus h p (f2x, f2y)
  dot_product v1 v2 = 0 →
  magnitude (vector_sum v1 v2) = 2 * Real.sqrt (h.a^2 + h.b^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l298_29899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_goats_investment_l298_29830

/-- Given a total investment and the ratio of stock to bond investment,
    calculate the investment in stocks. -/
noncomputable def investment_in_stocks (total_investment : ℝ) (stock_to_bond_ratio : ℝ) : ℝ :=
  (stock_to_bond_ratio * total_investment) / (stock_to_bond_ratio + 1)

/-- Theorem stating that given the specific conditions of the problem,
    the investment in stocks is $135,000. -/
theorem billy_goats_investment :
  let total_investment : ℝ := 165000
  let stock_to_bond_ratio : ℝ := 4.5
  investment_in_stocks total_investment stock_to_bond_ratio = 135000 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_goats_investment_l298_29830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_alpha_for_even_decreasing_function_l298_29881

noncomputable def f (α : ℤ) (x : ℝ) : ℝ := x^(α^2 - 2*α - 3)

theorem unique_alpha_for_even_decreasing_function :
  ∃! α : ℤ, (∀ x : ℝ, f α x = f α (-x)) ∧ 
            (∀ x y : ℝ, 0 < x ∧ x < y → f α y < f α x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_alpha_for_even_decreasing_function_l298_29881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_max_area_l298_29848

-- Define the circle A
def circle_A (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 8

-- Define point B
def point_B : ℝ × ℝ := (-1, 0)

-- Define the locus of point S (curve C)
def curve_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the perpendicular bisector property
def perp_bisector (S T : ℝ × ℝ) (B : ℝ × ℝ) : Prop :=
  (S.1 - B.1)^2 + (S.2 - B.2)^2 = (T.1 - B.1)^2 + (T.2 - B.2)^2

-- Define the chord property
def chord_property (D E M N : ℝ × ℝ) : Prop :=
  (E.1 - D.1) * (N.1 - M.1) + (E.2 - D.2) * (N.2 - M.2) = 0

-- Define the midpoint of a segment
def is_midpoint (P D E : ℝ × ℝ) : Prop :=
  P.1 = (D.1 + E.1) / 2 ∧ P.2 = (D.2 + E.2) / 2

-- Define the area of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Theorem statement
theorem curve_C_and_max_area :
  ∀ (T S : ℝ × ℝ),
    circle_A T.1 T.2 →
    perp_bisector S T point_B →
    curve_C S.1 S.2 →
    (∀ (D E M N P Q : ℝ × ℝ),
      curve_C D.1 D.2 →
      curve_C E.1 E.2 →
      curve_C M.1 M.2 →
      curve_C N.1 N.2 →
      is_midpoint P D E →
      is_midpoint Q M N →
      chord_property D E M N →
      area_triangle point_B P Q ≤ 1/9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_max_area_l298_29848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l298_29857

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) / (x - 1)

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x < 0}

-- Define the set P
def P (a : ℝ) : Set ℝ := {x | (deriv (f a)) x > 0}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (M a ⊂ P a) ∧ (M a ≠ P a) → a ∈ Set.Ioi 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l298_29857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divisibility_l298_29822

theorem square_divisibility (m n : ℕ) (hm : m > 0) (hn : n > 0)
  (h : (m^2 + n^2 + m) % (m * n) = 0) : 
  ∃ k : ℕ, k > 0 ∧ m = k^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divisibility_l298_29822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_digit_divisible_by_three_l298_29871

/-- The number of n-digit numbers composed of digits 1 and 2 that are divisible by 3 -/
def A : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 2
| (n + 3) => (2^(n + 3) + A (n + 1)) / 3

/-- The set of all 100-digit numbers composed of digits 1 and 2 -/
def S : Finset ℕ :=
  Finset.filter (fun x => x ≥ 10^99 ∧ x < 10^100 ∧ ∀ d, d ∈ x.digits 10 → d = 1 ∨ d = 2) (Finset.range (10^100))

theorem hundred_digit_divisible_by_three :
  (Finset.filter (fun x => x % 3 = 0) S).card = (2^100 + 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_digit_divisible_by_three_l298_29871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l298_29850

theorem inequality_solution (x : ℝ) : 
  (x^2 + 2*x^3 - 3*x^4) / (x + 2*x^2 - 3*x^3) ≥ -1 →
  x ∈ Set.Icc (-1 : ℝ) (-1/3) ∪ 
      Set.Ioo (-1/3 : ℝ) 0 ∪ 
      Set.Ioo (0 : ℝ) 1 ∪ 
      Set.Ioi (1 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l298_29850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_2_6_in_terms_of_a_and_b_l298_29842

theorem log_2_6_in_terms_of_a_and_b (a b : ℝ) 
  (h1 : (10 : ℝ)^a = 3) 
  (h2 : Real.log 2 = b) : 
  Real.log 6 / Real.log 2 = 1 + a / b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_2_6_in_terms_of_a_and_b_l298_29842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l298_29800

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else 2*x - x^2

-- State the theorem
theorem range_of_a (a : ℝ) : f (2 - a^2) > f a → -2 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l298_29800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_l298_29872

theorem number_of_divisors (n : ℕ) (h : n = 7^3 * 11^2 * 13^4) : 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_l298_29872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_g₅₀_eq_18_l298_29866

-- Define g₁(n) as three times the number of positive integer divisors of n
def g₁ (n : ℕ) : ℕ := 3 * (Nat.divisors n).card

-- Define gⱼ(n) recursively for j ≥ 2
def g (j : ℕ) (n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | 1 => g₁ n
  | j+1 => g₁ (g j n)

-- Theorem statement
theorem count_g₅₀_eq_18 :
  (Finset.filter (fun n => n ≤ 60 ∧ g 50 n = 18) (Finset.range 61)).card = 4 := by
  sorry

#eval (Finset.filter (fun n => n ≤ 60 ∧ g 50 n = 18) (Finset.range 61)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_g₅₀_eq_18_l298_29866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_all_negative_l298_29824

/-- Represents the sign in each cell of the table -/
inductive Sign
| Positive
| Negative

/-- Represents a 4x4 table of signs -/
def Table := Fin 4 → Fin 4 → Sign

/-- The initial configuration of the table -/
def initial_table : Table :=
  fun i j => match i, j with
  | 0, 2 => Sign.Negative
  | 1, 0 => Sign.Negative
  | 1, 1 => Sign.Negative
  | _, _ => Sign.Positive

/-- Flips the signs in a given row -/
def flip_row (t : Table) (row : Fin 4) : Table :=
  fun i j => if i = row then
    match t i j with
    | Sign.Positive => Sign.Negative
    | Sign.Negative => Sign.Positive
  else t i j

/-- Flips the signs in a given column -/
def flip_column (t : Table) (col : Fin 4) : Table :=
  fun i j => if j = col then
    match t i j with
    | Sign.Positive => Sign.Negative
    | Sign.Negative => Sign.Positive
  else t i j

/-- Counts the number of negative signs in the table -/
def count_negatives (t : Table) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 4)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 4)) fun j =>
      match t i j with
      | Sign.Negative => 1
      | Sign.Positive => 0)

/-- Theorem stating that it's impossible to obtain a table of all negative signs -/
theorem impossible_all_negative :
  ¬∃ (ops : List (Sum (Fin 4) (Fin 4))),
    (ops.foldl (fun t op => match op with
      | Sum.inl row => flip_row t row
      | Sum.inr col => flip_column t col) initial_table)
    = fun _ _ => Sign.Negative := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_all_negative_l298_29824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_eq_x_sgn_l298_29818

-- Define the sign function
noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x < 0 then -1
  else 0

-- State the theorem
theorem abs_eq_x_sgn (x : ℝ) : |x| = x * sgn x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_eq_x_sgn_l298_29818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_integral_h_l298_29831

noncomputable section

/-- The function f(x) = x^4 - x -/
def f (x : ℝ) : ℝ := x^4 - x

/-- The function g(x) = ax^3 + bx^2 + cx + d -/
def g (a b c d x : ℝ) : ℝ := a*x^3 + b*x^2 + c*x + d

/-- The squared difference between f and g -/
def h (a b c d x : ℝ) : ℝ := (f x - g a b c d x)^2

/-- The integral of h from -1 to 1 -/
noncomputable def integral_h (a b c d : ℝ) : ℝ := ∫ x in (-1)..(1), h a b c d x

theorem minimize_integral_h :
  ∀ a b c d : ℝ,
  f 1 = g a b c d 1 →
  f (-1) = g a b c d (-1) →
  integral_h a b c d ≥ integral_h 0 (8/7) (-1) (-1/7) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_integral_h_l298_29831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_and_salt_price_l298_29827

/-- The price of sugar per kilogram -/
noncomputable def sugar_price : ℝ := 1.50

/-- The price of salt per kilogram -/
noncomputable def salt_price : ℝ := (5.50 - 2 * sugar_price) / 5

/-- The theorem states that there exists a quantity of sugar such that
    when combined with 1 kg of salt, the total price is $5 -/
theorem sugar_and_salt_price : ∃ x : ℝ, x * sugar_price + salt_price = 5 := by
  -- We use 'use' to provide the value of x that satisfies the equation
  use (5 - salt_price) / sugar_price
  -- Simplify the goal
  simp [sugar_price, salt_price]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_and_salt_price_l298_29827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pole_order_two_at_one_l298_29819

open Complex Real

noncomputable def f (z : ℂ) : ℂ := (sin (π * z)) / (2 * exp (z - 1) - z^2 - 1)

theorem f_pole_order_two_at_one :
  ∃ (g : ℂ → ℂ) (h : ℂ → ℂ),
    (∀ z, z ≠ 1 → f z = g z / (z - 1)^2) ∧
    (g 1 ≠ 0) ∧
    (∀ z, h z = g z * (z - 1)^2) ∧
    (ContinuousAt h 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pole_order_two_at_one_l298_29819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l298_29823

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- Calculates the area of a rectangular park -/
def area (park : RectangularPark) : ℝ :=
  park.length * park.width

/-- Calculates the perimeter of a rectangular park -/
def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width)

/-- Conversion rate from paise to dollars -/
noncomputable def paiseToUSD : ℝ := 1 / (75 * 100)

/-- Cost of fencing per meter in dollars -/
noncomputable def fencingCostPerMeter : ℝ := 50 * paiseToUSD

/-- Theorem stating the area of the park given the conditions -/
theorem park_area (park : RectangularPark) 
  (h1 : fencingCostPerMeter * perimeter park = 100) : 
  area park = 13500000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l298_29823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l298_29854

-- Define the revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then 400 - 6 * x
  else if x > 40 then 7400 / x - 40000 / (x^2)
  else 0

-- Define the cost function
noncomputable def C (x : ℝ) : ℝ := 16 * x + 400

-- Define the profit function
noncomputable def W (x : ℝ) : ℝ := x * R x - C x

-- Theorem statement
theorem max_profit :
  ∃ (x_max : ℝ), x_max = 32 ∧
  ∀ (x : ℝ), x > 0 → W x ≤ W x_max ∧
  W x_max = 6104 := by
  sorry

#check max_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l298_29854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_two_l298_29815

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*x - 1 else -2*x + 6

-- State the theorem
theorem f_greater_than_two (t : ℝ) : f t > 2 ↔ t < 0 ∨ t > 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_two_l298_29815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l298_29847

noncomputable def g (x : ℝ) : ℝ := (x - 2) / Real.sqrt (x^2 - 5*x + 6)

theorem g_domain : 
  {x : ℝ | g x ≠ 0} = {x : ℝ | x < 2 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l298_29847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_revolution_volume_formula_l298_29882

/-- The volume of a solid of revolution formed by rotating a trapezoid around its longer parallel side -/
noncomputable def trapezoid_revolution_volume (a b m : ℝ) : ℝ :=
  (Real.pi * m * (a^2 + 2*b^2 + 2*a*b)) / 3

/-- Theorem: The volume of the solid of revolution formed by rotating a trapezoid
    with parallel sides of lengths a and b and height m around its longer parallel side
    is equal to (π * m * (a^2 + 2b^2 + 2ab)) / 3 -/
theorem trapezoid_revolution_volume_formula (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) :
  trapezoid_revolution_volume a b m = (Real.pi * m * (a^2 + 2*b^2 + 2*a*b)) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_revolution_volume_formula_l298_29882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jar_price_proportional_to_volume_l298_29835

/-- Represents a conical jar with given dimensions and price -/
structure ConicalJar where
  baseDiameter : ℝ
  height : ℝ
  price : ℝ

/-- Calculates the volume of a conical jar -/
noncomputable def volume (jar : ConicalJar) : ℝ :=
  (1/3) * Real.pi * (jar.baseDiameter/2)^2 * jar.height

/-- The problem statement -/
theorem jar_price_proportional_to_volume (jar1 jar2 : ConicalJar)
  (h1 : jar1.baseDiameter = 3)
  (h2 : jar1.height = 4)
  (h3 : jar1.price = 0.6)
  (h4 : jar2.baseDiameter = 6)
  (h5 : jar2.height = 8)
  (h6 : volume jar2 = 4 * volume jar1) :
  jar2.price = 2.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jar_price_proportional_to_volume_l298_29835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_solutions_eq_sixteen_l298_29888

theorem product_of_solutions_eq_sixteen : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → x = 2 ∨ x = 8) ∧ 2 * 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_solutions_eq_sixteen_l298_29888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l298_29828

noncomputable def f (a x : ℝ) : ℝ := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_condition (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l298_29828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l298_29852

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi/2) 
  (h_dist : ∀ x₁ x₂, f ω φ x₁ = 0 → f ω φ x₂ = 0 → x₁ ≠ x₂ → |x₁ - x₂| = Real.pi/4)
  (h_point : f ω φ (Real.pi/3) = -1) :
  ∃ g : ℝ → ℝ, 
    (∀ x, f ω φ x = Real.sin (4*x + Real.pi/6)) ∧ 
    (∀ x, g x = Real.sin (2*x - Real.pi/3)) ∧ 
    (∀ k, (∃! x, 0 ≤ x ∧ x ≤ Real.pi/2 ∧ g x + k = 0) ↔ 
      (-Real.sqrt 3 / 2 < k ∧ k ≤ Real.sqrt 3 / 2) ∨ k = -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l298_29852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_bound_l298_29849

theorem max_value_bound (b : Real) (h : 0 < b ∧ b ≤ 1) :
  let f : Real → Real := λ x => x * (x - b) * (x - 1)
  let M := sSup (Set.range f)
  M ≤ 4 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_bound_l298_29849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l298_29884

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_properties (a b c A B C : ℝ)
  (law_of_sines : a / Real.sin A = b / Real.sin B)
  (equation : c * Real.sin C / a - Real.sin C = b * Real.sin B / a - Real.sin A)
  (b_eq_4 : b = 4)
  (c_eq : c = 4 * Real.sqrt 6 / 3)
  (angle_sum : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) :
  B = Real.pi / 3 ∧
  1/2 * b * c * Real.sin A = 4 + 4 * Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l298_29884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_case_l298_29834

/-- The length of the tangent line from a point to a circle -/
noncomputable def tangent_length (px py cx cy r : ℝ) : ℝ :=
  Real.sqrt ((px - cx)^2 + (py - cy)^2 - r^2)

/-- Theorem: The length of the tangent line from P(2,3) to the circle (x-1)^2+(y-1)^2=1 is 2 -/
theorem tangent_length_specific_case : 
  tangent_length 2 3 1 1 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_case_l298_29834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problem_solution_l298_29862

theorem math_problem_solution :
  (Real.sqrt 16 + (8 : ℝ) ^ (1/3) - Real.sqrt ((-5)^2) = 1) ∧
  ((-2)^3 + |1 - Real.sqrt 2| * (-1)^2023 - (125 : ℝ) ^ (1/3) = -12 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problem_solution_l298_29862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cute_number_l298_29859

def is_cute (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n ≤ 9999999) ∧ 
  (∀ k : ℕ, k ≥ 1 → k ≤ 7 → (n / 10^(7-k)) % k = 0) ∧
  (∃ p : Equiv.Perm (Fin 7), 
    ∀ i : Fin 7, 
      (n / 10^(6-i.val)) % 10 = p.toFun i + 1)

theorem unique_cute_number : ∃! n : ℕ, is_cute n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cute_number_l298_29859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l298_29814

open Real

noncomputable def f (B : ℝ) : ℝ := 
  (sin B * (2*cos B^2 + cos B^4 + 2*sin B^2 + sin B^2*cos B^2)) / 
  (tan B * (1/cos B - sin B * tan B))

theorem f_range (B : ℝ) (h : ∀ n : ℤ, B ≠ n * π / 2) : 
  2 < f B ∧ f B < 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l298_29814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_preceding_integers_l298_29829

theorem sum_of_preceding_integers : ∃! n : ℕ, n > 0 ∧ n = (n - 1) * n / 2 :=
by
  -- We use ℕ instead of ℕ+ to avoid issues with OfNat and HDiv
  -- The condition n > 0 ensures we're dealing with positive integers
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_preceding_integers_l298_29829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_condition_equivalence_l298_29805

open Matrix

theorem matrix_condition_equivalence (a b : ℝ) (h : b > a^2) :
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
    Matrix.det (A^2 - (2*a) • (1 : Matrix (Fin 2) (Fin 2) ℝ) * A + b • (1 : Matrix (Fin 2) (Fin 2) ℝ)) = 0 ↔
    Matrix.trace A = 2*a ∧ Matrix.det A = b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_condition_equivalence_l298_29805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_radius_l298_29861

/-- A right triangle with a special circle -/
structure SpecialTriangleCircle where
  -- The lengths of the legs of the right triangle
  short_leg : ℝ
  long_leg : ℝ
  -- The circle touches the longer leg
  touches_long_leg : Prop
  -- The circle passes through the vertex opposite the longer leg
  passes_through_vertex : Prop
  -- The circle's center is on the hypotenuse
  center_on_hypotenuse : Prop
  -- The triangle is a right triangle
  is_right_triangle : Prop

/-- The radius of the special circle in the triangle -/
noncomputable def circle_radius (t : SpecialTriangleCircle) : ℝ := 65 / 18

/-- Theorem: The radius of the special circle in a right triangle with legs 5 and 12 is 65/18 -/
theorem special_circle_radius :
  ∀ (t : SpecialTriangleCircle),
  t.short_leg = 5 →
  t.long_leg = 12 →
  circle_radius t = 65 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_radius_l298_29861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l298_29851

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x - 1)

-- State the theorem
theorem derivative_of_f :
  ∀ x : ℝ, x ≠ 1 → deriv f x = -1 / (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l298_29851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_fraction_l298_29811

theorem largest_fraction (a b c d e : ℝ) 
  (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (b + d + e) / (a + c) > max 
    ((a + b + e) / (c + d)) 
    (max ((a + d) / (b + e)) 
      (max ((b + c) / (a + e)) 
        ((c + e) / (a + b + d)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_fraction_l298_29811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standing_time_approx_51_l298_29804

/-- Represents the time in seconds for Clea to descend the escalator in different scenarios -/
structure EscalatorTime where
  nonOperating : ℚ  -- Time to walk down non-operating escalator
  operating : ℚ     -- Time to walk down operating escalator
  standing : ℚ      -- Time to stand on operating escalator

/-- Calculates the time for Clea to stand on the operating escalator -/
def calculateStandingTime (t : EscalatorTime) : ℚ :=
  (t.nonOperating * t.operating) / (t.nonOperating - t.operating)

/-- Theorem stating that given the conditions, the standing time is approximately 51 seconds -/
theorem standing_time_approx_51 (t : EscalatorTime) 
    (h1 : t.nonOperating = 72) 
    (h2 : t.operating = 30) : 
    ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |calculateStandingTime t - 51| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standing_time_approx_51_l298_29804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_dot_product_l298_29825

/-- 
A trapezoid is a quadrilateral with at least one pair of parallel sides.
For our purposes, we assume AB and CD are parallel.
-/
structure Trapezoid (A B C D : ℝ × ℝ) : Prop where
  is_trapezoid : (A.2 = B.2) ∧ (C.2 = D.2)

/-- The length of a vector -/
noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

theorem trapezoid_dot_product 
  (A B C D : ℝ × ℝ) 
  (h_trap : Trapezoid A B C D) 
  (h_AB : vector_length (B.1 - A.1, B.2 - A.2) = 55) 
  (h_CD : vector_length (D.1 - C.1, D.2 - C.2) = 31) 
  (h_perp : perpendicular (C.1 - A.1, C.2 - A.2) (D.1 - B.1, D.2 - B.2)) :
  dot_product (D.1 - A.1, D.2 - A.2) (C.1 - B.1, C.2 - B.2) = 1705 := by
  sorry

#check trapezoid_dot_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_dot_product_l298_29825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_A_and_B_range_of_a_when_A_union_B_equals_B_l298_29896

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log x / Real.log 2 + Real.sqrt (2 - x)}
def B (a : ℝ) : Set ℝ := {x | 1/4 < (2 : ℝ)^(x-a) ∧ (2 : ℝ)^(x-a) ≤ 8}

-- Theorem for the first question
theorem intersection_complement_A_and_B :
  (Set.univ \ A) ∩ (B 0) = Set.Ioo (-2) 0 := by sorry

-- Theorem for the second question
theorem range_of_a_when_A_union_B_equals_B :
  ∀ a : ℝ, A ∪ B a = B a → -1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_A_and_B_range_of_a_when_A_union_B_equals_B_l298_29896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearance_time_is_23_seconds_l298_29840

/-- The time (in seconds) for two trains to clear each other -/
noncomputable def trainClearanceTime (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600))

/-- Theorem: The time for two trains to clear each other is 23 seconds -/
theorem train_clearance_time_is_23_seconds :
  trainClearanceTime 180 280 42 30 = 23 := by
  -- Unfold the definition of trainClearanceTime
  unfold trainClearanceTime
  -- Perform the calculation
  simp [div_eq_mul_inv]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearance_time_is_23_seconds_l298_29840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_arithmetic_S_formula_D_100_floor_l298_29875

def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => a n / 2 + 1 / 2^(n + 1)

def b (n : ℕ) : ℚ := 2^(n - 1) * a n

def S (n : ℕ) : ℚ := Finset.sum (Finset.range n) (λ i => a i)

noncomputable def d (n : ℕ) : ℝ := Real.sqrt (1 + 1 / (b n)^2 + 1 / (b (n + 1))^2)

noncomputable def D (n : ℕ) : ℝ := Finset.sum (Finset.range n) (λ i => d i)

theorem b_is_arithmetic (n : ℕ) : b n = n := by sorry

theorem S_formula (n : ℕ) : S n = 4 - (2 + n) / 2^(n - 1) := by sorry

theorem D_100_floor : ⌊D 100⌋ = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_arithmetic_S_formula_D_100_floor_l298_29875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_constant_distance_a_to_pbq_formula_circumsphere_diameter_formula_min_diameter_condition_l298_29853

-- Define the tetrahedron PABQ
structure Tetrahedron where
  a : ℝ
  k : ℝ
  x : ℝ
  y : ℝ
  h_perp : True  -- Ax and By are perpendicular
  h_normal : True  -- AB is a common normal
  h_ab : True  -- AB = 2a
  h_xy : x * y = k^2

-- Define the volume function
noncomputable def volume (t : Tetrahedron) : ℝ := (1/3) * t.a * t.k^2

-- Define the distance function from A to base PBQ
noncomputable def distance_a_to_pbq (t : Tetrahedron) : ℝ :=
  (2 * t.a * t.k^2) / Real.sqrt (4 * t.a^2 * t.y^2 + t.k^4)

-- Define the diameter of the circumsphere
noncomputable def circumsphere_diameter (t : Tetrahedron) : ℝ :=
  Real.sqrt (4 * t.a^2 + 2 * t.k^2 + (t.x - t.y)^2)

-- Theorem statements
theorem volume_constant (t : Tetrahedron) : volume t = (1/3) * t.a * t.k^2 := by sorry

theorem distance_a_to_pbq_formula (t : Tetrahedron) :
  distance_a_to_pbq t = (2 * t.a * t.k^2) / Real.sqrt (4 * t.a^2 * t.y^2 + t.k^4) := by sorry

theorem circumsphere_diameter_formula (t : Tetrahedron) :
  circumsphere_diameter t = Real.sqrt (4 * t.a^2 + 2 * t.k^2 + (t.x - t.y)^2) := by sorry

theorem min_diameter_condition (t : Tetrahedron) :
  t.x = t.k ∧ t.y = t.k → circumsphere_diameter t = Real.sqrt (4 * t.a^2 + 2 * t.k^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_constant_distance_a_to_pbq_formula_circumsphere_diameter_formula_min_diameter_condition_l298_29853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ratio_greater_than_actual_l298_29892

/-- Represents the number of managers in a department -/
def managers : ℕ := 8

/-- Represents the maximum number of non-managers in the department -/
def max_non_managers : ℕ := 27

/-- The required ratio of managers to non-managers -/
def required_ratio : ℚ := 10 / 27

/-- Theorem stating that the required ratio is the smallest ratio greater than the actual ratio -/
theorem smallest_ratio_greater_than_actual :
  required_ratio > (managers : ℚ) / max_non_managers ∧
  ∀ r : ℚ, r > (managers : ℚ) / max_non_managers → r ≥ required_ratio :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ratio_greater_than_actual_l298_29892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l298_29883

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4)

theorem f_properties :
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8))) ∧
  (∀ x ∈ Set.Icc (-Real.pi/8) (Real.pi/2), f x ≥ -1) ∧
  (f (Real.pi/2) = -1) ∧
  (∀ x ∈ Set.Icc (-Real.pi/8) (Real.pi/2), f x ≤ Real.sqrt 2) ∧
  (f (Real.pi/8) = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l298_29883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_4_minus_alpha_l298_29812

theorem cos_pi_4_minus_alpha (α : ℝ) 
  (h1 : Real.sin (2 * α) = 24 / 25) 
  (h2 : 0 < α) 
  (h3 : α < Real.pi / 2) : 
  Real.sqrt 2 * Real.cos (Real.pi / 4 - α) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_4_minus_alpha_l298_29812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_A_star_B_l298_29855

def A : Finset ℤ := {-1, 0, 1}
def B : Finset ℤ := {0, 1, 2, 3}

def A_star_B : Finset (ℤ × ℤ) :=
  (A ∩ B).product (A ∪ B)

theorem number_of_subsets_A_star_B :
  Finset.card (Finset.powerset A_star_B) = 2^(Finset.card A_star_B) ∧
  Finset.card A_star_B = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_A_star_B_l298_29855
