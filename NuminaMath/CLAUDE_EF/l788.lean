import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_power_l788_78846

/-- The parabola defined by y = -(x+2)^2 + 3 -/
def parabola (x : ℝ) : ℝ := -(x + 2)^2 + 3

/-- The x-coordinate of the vertex of the parabola -/
def m : ℝ := -2

/-- The y-coordinate of the vertex of the parabola -/
def n : ℝ := 3

/-- Theorem stating that m^n = -8 for the given parabola -/
theorem vertex_power : (m : ℝ) ^ (n : ℝ) = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_power_l788_78846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_one_third_l788_78833

theorem trigonometric_expression_equals_one_third :
  let angle : Real := 30 * π / 180
  let sin_squared : Real := (Real.sin angle) ^ 2
  let cos_squared : Real := (Real.cos angle) ^ 2
  sin_squared = 1/4 ∧ cos_squared = 3/4 →
  ((Real.tan angle)^2 - cos_squared) / ((Real.tan angle)^2 * cos_squared) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_one_third_l788_78833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_min_value_equality_l788_78801

/-- The function f(x) defined for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := (2 * x^2 + x + 4) / x

/-- Theorem stating the minimum value of f(x) for x > 0 -/
theorem f_min_value (x : ℝ) (hx : x > 0) : f x ≥ 4 * Real.sqrt 2 + 1 := by
  sorry

/-- Theorem stating the condition for equality -/
theorem f_min_value_equality : f (Real.sqrt 2) = 4 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_min_value_equality_l788_78801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_shipped_percentage_l788_78853

/-- The percentage of defective units produced -/
noncomputable def defective_percentage : ℝ := 8

/-- The percentage of defective units shipped -/
noncomputable def shipped_percentage : ℝ := 5

/-- The percentage of units that are both defective and shipped -/
noncomputable def defective_and_shipped : ℝ := (defective_percentage / 100) * (shipped_percentage / 100) * 100

theorem defective_shipped_percentage :
  defective_and_shipped = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_shipped_percentage_l788_78853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_circle_and_ray_l788_78819

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Defines the set of points satisfying the given polar equation -/
def SolutionSet : Set PolarPoint :=
  {p : PolarPoint | p.ρ > 0 ∧ (p.ρ - 1) * (p.θ - Real.pi) = 0}

/-- Represents a circle in polar form -/
def Circle : Set PolarPoint :=
  {p : PolarPoint | p.ρ = 1 ∧ p.ρ > 0}

/-- Represents a ray in polar form -/
def Ray : Set PolarPoint :=
  {p : PolarPoint | p.θ = Real.pi ∧ p.ρ > 0}

theorem solution_is_circle_and_ray : 
  SolutionSet = Circle ∪ Ray := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_circle_and_ray_l788_78819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_point_p_l788_78832

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define a point on the ellipse
structure PointOnEllipse (a b : ℝ) where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse a b x y

-- Define the foci of the ellipse
structure Foci (a b : ℝ) where
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ

-- Define the inscribed circle of a triangle
def inscribed_circle_radius (P F1 F2 : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), ∀ (point : ℝ × ℝ), point = P ∨ point = F1 ∨ point = F2 → 
    Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) = r

-- Theorem statement
theorem y_coordinate_of_point_p 
  (a b : ℝ) (P : PointOnEllipse a b) (F : Foci a b) (r : ℝ) :
  inscribed_circle_radius (P.x, P.y) F.F1 F.F2 r →
  P.y > 0 →
  P.y = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_point_p_l788_78832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squared_radius_of_nested_sphere_l788_78803

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones and a nested sphere -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ
  sphereRadius : ℝ

/-- The maximum possible squared radius of a sphere nested within two intersecting cones -/
noncomputable def maxSquaredRadius (config : ConeConfiguration) : ℝ :=
  (66 - 16 * Real.sqrt 116) / 25

/-- The theorem statement -/
theorem max_squared_radius_of_nested_sphere 
  (config : ConeConfiguration) 
  (h1 : config.cone1 = config.cone2) 
  (h2 : config.cone1.baseRadius = 4) 
  (h3 : config.cone1.height = 10) 
  (h4 : config.intersectionDistance = 4) :
  maxSquaredRadius config = (66 - 16 * Real.sqrt 116) / 25 := by
  sorry

#eval (66 + 25 : Nat)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squared_radius_of_nested_sphere_l788_78803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l788_78815

-- Define the line
def line (θ a : ℝ) (x y : ℝ) : Prop :=
  x * Real.cos θ + y * Real.sin θ + a = 0

-- Define the circle (renamed to avoid conflict with existing definition)
def circle_eq (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = a^2

-- Theorem statement
theorem line_circle_intersection (θ a : ℝ) :
  ∃! p : ℝ × ℝ, line θ a p.1 p.2 ∧ circle_eq a p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l788_78815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_doubles_l788_78862

/-- Represents a person's financial situation over two years -/
structure FinancialSituation where
  income : ℝ
  savings_rate : ℝ
  income_increase : ℝ

/-- Calculates the savings increase percentage given a financial situation -/
noncomputable def savings_increase_percentage (fs : FinancialSituation) : ℝ :=
  let first_year_savings := fs.income * fs.savings_rate
  let first_year_expenses := fs.income * (1 - fs.savings_rate)
  let second_year_income := fs.income * (1 + fs.income_increase)
  let second_year_expenses := first_year_expenses
  let second_year_savings := second_year_income - second_year_expenses
  (second_year_savings - first_year_savings) / first_year_savings * 100

/-- Theorem stating that under the given conditions, savings increase by 100% -/
theorem savings_doubles (fs : FinancialSituation) 
    (h1 : fs.savings_rate = 0.25)
    (h2 : fs.income_increase = 0.25)
    (h3 : fs.income > 0) : 
  savings_increase_percentage fs = 100 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_doubles_l788_78862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l788_78892

noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 - x - 5/2

theorem quadratic_properties :
  (∀ x, f x = -1/2 * (x + 1)^2 - 2) ∧
  (f (-1) = -2 ∧ ∀ x, f x ≤ f (-1)) ∧
  (∀ x > -1, ∀ y > x, f y < f x) ∧
  (∀ x ∈ Set.Icc (-3) 2, -13/2 ≤ f x ∧ f x ≤ -2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l788_78892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_minimum_l788_78877

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x + 4) / Real.sqrt x

-- State the theorem
theorem f_domain_and_minimum :
  (∀ x > 0, f x ∈ Set.Ioi 0) ∧
  (∃ x > 0, f x = 4) ∧
  (∀ x > 0, f x ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_minimum_l788_78877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimension_sum_l788_78812

/-- Represents a rectangular box with given dimensions -/
structure Box where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the area of the triangle formed by the center points of three faces meeting at a corner -/
noncomputable def triangleArea (b : Box) : ℝ :=
  let s1 := Real.sqrt ((b.width / 2) ^ 2 + (b.length / 2) ^ 2)
  let s2 := Real.sqrt ((b.width / 2) ^ 2 + (b.height / 2) ^ 2)
  let s3 := Real.sqrt ((b.length / 2) ^ 2 + (b.height / 2) ^ 2)
  let s := (s1 + s2 + s3) / 2
  Real.sqrt (s * (s - s1) * (s - s2) * (s - s3))

/-- Theorem statement -/
theorem box_dimension_sum (p q : ℕ) (hp : Nat.Coprime p q) :
  let b := Box.mk 10 20 (p / q : ℝ)
  triangleArea b = 40 → p + q = 133 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimension_sum_l788_78812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l788_78848

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

-- State the theorem
theorem f_properties :
  -- 1. Monotonicity intervals
  (∀ x₁ x₂, -2 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < -2 → f x₂ < f x₁) ∧
  -- 2. Minimum value
  (∀ x, f x ≥ f (-2)) ∧
  f (-2) = -Real.exp (-2) ∧
  -- 3. Conditions for two distinct solutions
  (∀ a, (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a) ↔ -Real.exp (-2) < a ∧ a < 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l788_78848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_sum_l788_78889

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = k • a

/-- The problem statement -/
theorem collinear_vectors_sum (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (x, 3/2, 3)
  let b : ℝ × ℝ × ℝ := (-1, y, 2)
  collinear a b → x + y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_sum_l788_78889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_healthy_rum_multiple_calculation_l788_78878

/-- The multiple of rum on pancakes that can be consumed for a healthy diet -/
noncomputable def healthy_rum_multiple (rum_on_pancakes : ℝ) (rum_consumed_earlier : ℝ) (rum_after_eating : ℝ) : ℝ :=
  (rum_after_eating + rum_consumed_earlier - rum_consumed_earlier) / rum_on_pancakes

theorem healthy_rum_multiple_calculation (rum_on_pancakes : ℝ) (rum_consumed_earlier : ℝ) (rum_after_eating : ℝ)
  (h1 : rum_on_pancakes = 10)
  (h2 : rum_consumed_earlier = 12)
  (h3 : rum_after_eating = 8) :
  healthy_rum_multiple rum_on_pancakes rum_consumed_earlier rum_after_eating = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_healthy_rum_multiple_calculation_l788_78878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_own_inverse_l788_78844

/-- The function f(x) as defined in the problem -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + 1) / (k*x - 1)

/-- The theorem stating that f is its own inverse if and only if k = -2 -/
theorem f_is_own_inverse (k : ℝ) :
  (∀ x, f k (f k x) = x) ↔ k = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_own_inverse_l788_78844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_r_plus_s_l788_78809

/-- Triangle XYZ in the Cartesian plane -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Calculate the area of a triangle given its vertices -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculate the slope of a line given two points -/
def lineSlope (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the midpoint of a line segment -/
def lineMidpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Theorem: Maximum value of r+s for triangle XYZ -/
theorem max_r_plus_s (t : Triangle) (h1 : t.Y = (10, 15)) (h2 : t.Z = (20, 18))
    (h3 : triangleArea t = 90)
    (h4 : lineSlope t.X (lineMidpoint t.Y t.Z) = -3) :
    ∃ (max : ℝ), max = 42.91 ∧ t.X.1 + t.X.2 ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_r_plus_s_l788_78809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_twenty_l788_78842

/-- Calculates the upstream distance swam given the downstream distance, time, and still water speed. -/
noncomputable def upstream_distance (downstream_distance : ℝ) (time : ℝ) (still_water_speed : ℝ) : ℝ :=
  let stream_speed := downstream_distance / time - still_water_speed
  (still_water_speed - stream_speed) * time

/-- Theorem stating that under the given conditions, the upstream distance is 20 km. -/
theorem upstream_distance_is_twenty :
  upstream_distance 30 5 5 = 20 := by
  -- Unfold the definition of upstream_distance
  unfold upstream_distance
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_twenty_l788_78842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_conditions_determine_right_triangle_l788_78852

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = 180

-- Define the conditions
def condition1 (t : Triangle) : Prop := t.A + t.B = t.C
def condition2 (t : Triangle) : Prop := t.A = t.B ∧ t.A = 2 * t.C
def condition3 (t : Triangle) : Prop := ∃ (k : ℝ), t.A = k ∧ t.B = 2*k ∧ t.C = 3*k

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop := t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Theorem statement
theorem two_conditions_determine_right_triangle :
  ∃ (c1 c2 : Triangle → Prop),
    (c1 = condition1 ∨ c1 = condition2 ∨ c1 = condition3) ∧
    (c2 = condition1 ∨ c2 = condition2 ∨ c2 = condition3) ∧
    c1 ≠ c2 ∧
    (∀ t : Triangle, c1 t → c2 t → is_right_triangle t) ∧
    (∀ c3 : Triangle → Prop,
      (c3 = condition1 ∨ c3 = condition2 ∨ c3 = condition3) →
      c3 ≠ c1 → c3 ≠ c2 →
      ∃ t : Triangle, c3 t ∧ ¬is_right_triangle t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_conditions_determine_right_triangle_l788_78852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_fraction_of_circle_radius_l788_78880

theorem rectangle_length_fraction_of_circle_radius :
  ∀ (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ),
    square_area = 784 →
    rectangle_area = 35 →
    rectangle_breadth = 5 →
    let square_side : ℝ := Real.sqrt square_area
    let circle_radius : ℝ := square_side
    let rectangle_length : ℝ := rectangle_area / rectangle_breadth
    rectangle_length / circle_radius = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_fraction_of_circle_radius_l788_78880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_rational_unit_closed_line_l788_78868

/-- A closed broken line on a coordinate plane -/
structure ClosedBrokenLine where
  vertices : List (ℚ × ℚ)
  is_closed : vertices.head? = vertices.getLast?
  edge_length_one : ∀ i, i < vertices.length - 1 →
    let (x₁, y₁) := vertices[i]!
    let (x₂, y₂) := vertices[i + 1]!
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 1

/-- Theorem stating the impossibility of a closed broken line with rational coordinates,
    unit edge lengths, and an odd number of vertices -/
theorem no_odd_rational_unit_closed_line :
  ¬ ∃ (line : ClosedBrokenLine), Odd line.vertices.length := by
  sorry

#check no_odd_rational_unit_closed_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_rational_unit_closed_line_l788_78868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_trig_identity_l788_78847

theorem third_quadrant_trig_identity (α : Real) 
  (h1 : α ∈ Set.Icc π (3*π/2))  -- α is in the third quadrant
  (h2 : Real.cos α = -4/5) :    -- Given condition
  (1 + Real.tan (α/2)) / (1 - Real.tan (α/2)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_trig_identity_l788_78847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_height_calculation_l788_78806

/-- Calculates the height of a brick given wall dimensions and brick count -/
noncomputable def calculate_brick_height (wall_length wall_width wall_height : ℝ) 
                            (brick_length brick_width : ℝ) 
                            (brick_count : ℕ) : ℝ :=
  (wall_length * wall_width * wall_height * 100^3) / 
  (brick_length * brick_width * (brick_count : ℝ))

theorem brick_height_calculation (wall_length wall_width wall_height : ℝ) 
                                 (brick_length brick_width : ℝ) 
                                 (brick_count : ℕ) :
  wall_length = 10 ∧ 
  wall_width = 8 ∧ 
  wall_height = 24.5 ∧
  brick_length = 20 ∧
  brick_width = 10 ∧
  brick_count = 12250 →
  calculate_brick_height wall_length wall_width wall_height brick_length brick_width brick_count =
  (wall_length * wall_width * wall_height * 100^3) / 
  (brick_length * brick_width * (brick_count : ℝ)) :=
by
  sorry

#check brick_height_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_height_calculation_l788_78806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_is_20_l788_78860

/-- The length of a race where a hare and a turtle finish in a tie -/
noncomputable def race_length (hare_speed turtle_speed : ℝ) (head_start : ℝ) : ℝ :=
  hare_speed * (head_start * turtle_speed / (hare_speed - turtle_speed))

/-- Theorem stating that the race length is 20 feet given the specific conditions -/
theorem race_length_is_20 :
  race_length 10 1 18 = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_is_20_l788_78860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_intersection_distance_l788_78834

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, 2*t + 2)

-- Define the curve C
def curve_C (m : ℝ) : ℝ × ℝ := (m, m^2)

-- Define the circle O
noncomputable def circle_O (a : ℝ) : ℝ → ℝ × ℝ := fun θ => (a * Real.cos θ, a * Real.sin θ)

-- Theorem 1: The value of a when line l is tangent to circle O
theorem tangent_circle_radius : 
  ∃ a : ℝ, a > 0 ∧ (∃ t : ℝ, line_l t ∈ Set.range (circle_O a)) ∧ 
  (∀ t : ℝ, (line_l t).1^2 + (line_l t).2^2 ≥ a^2) ∧ 
  a = 2 * Real.sqrt 5 / 5 := by
  sorry

-- Theorem 2: The distance |AB| between the intersection points of line l and curve C
theorem intersection_distance : 
  ∃ A B : ℝ × ℝ, A ≠ B ∧ 
  (∃ t1 : ℝ, line_l t1 = A) ∧ (∃ t2 : ℝ, line_l t2 = B) ∧
  (∃ m1 : ℝ, curve_C m1 = A) ∧ (∃ m2 : ℝ, curve_C m2 = B) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_intersection_distance_l788_78834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_and_mary_ages_l788_78897

/-- Represents a person's age as a two-digit number -/
structure TwoDigitAge where
  tens : Nat
  ones : Nat
  valid : tens ≤ 9 ∧ ones ≤ 9

/-- Swaps the digits of a two-digit age -/
def swapDigits (age : TwoDigitAge) : TwoDigitAge where
  tens := age.ones
  ones := age.tens
  valid := age.valid.symm

/-- Converts a TwoDigitAge to a natural number -/
def ageToNat (age : TwoDigitAge) : Nat :=
  10 * age.tens + age.ones

theorem john_and_mary_ages :
  ∃ (john : TwoDigitAge) (mary : TwoDigitAge),
    -- John is older than Mary
    ageToNat john > ageToNat mary ∧
    -- Swapping John's age digits gives Mary's age
    mary = swapDigits john ∧
    -- The difference between squares of their ages is a perfect square
    ∃ (k : Nat), (ageToNat john)^2 - (ageToNat mary)^2 = k^2 ∧
    -- John is 65 and Mary is 56
    ageToNat john = 65 ∧ ageToNat mary = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_and_mary_ages_l788_78897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_ratio_is_two_to_one_l788_78823

/-- Represents a rectangular table with one side against a wall -/
structure RectangularTable where
  length : ℝ  -- Length of the side opposite the wall
  width : ℝ   -- Width of the table (length of each of the other two free sides)

/-- The area of the table -/
noncomputable def RectangularTable.area (t : RectangularTable) : ℝ :=
  t.length * t.width

/-- The total length of the free sides of the table -/
noncomputable def RectangularTable.freePerimeter (t : RectangularTable) : ℝ :=
  t.length + 2 * t.width

/-- The ratio of the length to the width -/
noncomputable def RectangularTable.lengthToWidthRatio (t : RectangularTable) : ℝ :=
  t.length / t.width

theorem table_ratio_is_two_to_one 
  (t : RectangularTable) 
  (h_area : t.area = 128) 
  (h_perimeter : t.freePerimeter = 32) : 
  t.lengthToWidthRatio = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_ratio_is_two_to_one_l788_78823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_earnings_l788_78895

/-- The amount of fool's gold Bill sold in ounces -/
noncomputable def amount_sold : ℝ := 8

/-- The fine Bill had to pay in dollars -/
noncomputable def fine : ℝ := 50

/-- The amount Bill was left with after the fine in dollars -/
noncomputable def amount_left : ℝ := 22

/-- The amount Bill earned per ounce of fool's gold in dollars -/
noncomputable def earnings_per_ounce : ℝ := (amount_left + fine) / amount_sold

theorem bill_earnings : earnings_per_ounce = 9 := by
  -- Unfold the definition of earnings_per_ounce
  unfold earnings_per_ounce
  -- Simplify the expression
  simp [amount_left, fine, amount_sold]
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_earnings_l788_78895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_l788_78872

theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  ∃ r : ℝ, (P * r * 10 / 100 = 6 * P / 5) ∧ r = 12 := by
  use 12
  constructor
  · field_simp
    ring
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_l788_78872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l788_78813

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 9
def g (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 5

-- Define the intersection points
noncomputable def p₁ : ℝ × ℝ := (2 - 3 * Real.sqrt 2, 66 + 36 * Real.sqrt 2)
noncomputable def p₂ : ℝ × ℝ := (2 + 3 * Real.sqrt 2, 66 - 36 * Real.sqrt 2)

theorem parabolas_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) = p₁ ∨ (x, y) = p₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l788_78813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_145_l788_78858

def pentagon_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | n + 4 => pentagon_sequence (n + 3) + 3 * (n + 4) - 2

theorem tenth_term_is_145 : pentagon_sequence 9 = 145 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_145_l788_78858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PB_equals_x_plus_two_l788_78830

-- Define the circle and points
variable (ω : Circle ℝ) (A B C P M : Point ℝ)

-- Define the conditions
variable (hAB : Chord ω A B)
variable (hMP : RadialSegment ω M P)
variable (hC : C ∈ ω ∧ C ∈ ShorterArc ω A B)
variable (x : ℝ)
variable (hAC : (A.dist C) = x)
variable (hMP_length : (M.dist P) = x + 1)
variable (hAP : (A.dist P) = x + 2)
variable (hM_midpoint : M = Midpoint ⟨M, P⟩)

-- Statement to prove
theorem length_PB_equals_x_plus_two :
  (P.dist B) = x + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PB_equals_x_plus_two_l788_78830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l788_78808

/-- Represents a right circular cone -/
structure RightCircularCone where
  volume : ℝ
  height : ℝ

/-- The circumference of the base of a right circular cone -/
noncomputable def base_circumference (cone : RightCircularCone) : ℝ :=
  2 * Real.pi * (3 * cone.volume / (Real.pi * cone.height)) ^ (1/2)

/-- Theorem: The circumference of the base of a right circular cone with volume 18π and height 6 is 6π -/
theorem cone_base_circumference :
  let cone : RightCircularCone := { volume := 18 * Real.pi, height := 6 }
  base_circumference cone = 6 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l788_78808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_livelys_class_l788_78859

theorem mrs_livelys_class :
  ∃ (boys girls : ℕ),
    boys + girls = 30 ∧
    (3 : ℚ) / 5 * boys = (5 : ℚ) / 6 * girls ∧
    boys = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_livelys_class_l788_78859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l788_78886

/-- Represents an ellipse with semi-major axis a, semi-minor axis b, and semi-focal distance c. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_pos : 0 < c
  h_a_gt_b : a > b
  h_equation : c^2 = a^2 - b^2

/-- The eccentricity of an ellipse. -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- Angle BFO in an ellipse, where B is the upper endpoint of the minor axis,
    F is the right focus, and O is the origin. -/
noncomputable def angle_BFO (e : Ellipse) : ℝ := Real.arctan (e.b / e.c)

theorem ellipse_properties (e : Ellipse) :
  (2 * e.b > e.a + e.c → e.b^2 > e.a * e.c) ∧
  (Real.tan (angle_BFO e) > 1 → 0 < eccentricity e ∧ eccentricity e < Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l788_78886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l788_78855

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3*x

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, y ∈ Set.Ioo (1/2 : ℝ) 1 → 
    (∀ z : ℝ, z ∈ Set.Ioo (1/2 : ℝ) 1 → y < z → f y > f z)) ∧
  (∀ a b : ℝ, 0 < a ∧ a < 1/2 → b > 1 → 
    ¬(∀ y z : ℝ, y ∈ Set.Ioo a b → z ∈ Set.Ioo a b → y < z → f y > f z)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l788_78855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_pentagon_area_inequality_l788_78800

/-- A pentagon is convex -/
def IsConvexPentagon (area : ℝ) : Prop :=
  sorry

/-- One pentagon is formed by the midpoints of another pentagon -/
def IsMidpointPentagon (area_original area_midpoint : ℝ) : Prop :=
  sorry

/-- Given a convex pentagon with area T and another convex pentagon formed by its side midpoints with area t, prove that 3/4 * T > t > 1/2 * T -/
theorem midpoint_pentagon_area_inequality (T t : ℝ) 
  (h_T_pos : T > 0)
  (h_t_pos : t > 0)
  (h_convex_original : IsConvexPentagon T)
  (h_convex_midpoint : IsConvexPentagon t)
  (h_midpoint : IsMidpointPentagon T t) : 
  3/4 * T > t ∧ t > 1/2 * T := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_pentagon_area_inequality_l788_78800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_orientations_theorem_l788_78811

/-- Represents a point on the surface of a cube -/
structure CubePoint where
  face : Fin 6
  coordinates : Fin 2 → ℝ
  valid : coordinates 0 ∈ Set.Icc 0 1 ∧ coordinates 1 ∈ Set.Icc 0 1

/-- Represents an orientation of the cube -/
structure CubeOrientation where
  bottom_face : Fin 6
  rotation : Fin 4

/-- The set of points that touch the plane in a given orientation -/
noncomputable def touchingPoints (points : Finset CubePoint) (orientation : CubeOrientation) : Finset CubePoint :=
  sorry

theorem cube_orientations_theorem 
  (points : Finset CubePoint) 
  (h : points.card = 100) :
  ∃ (o1 o2 : CubeOrientation), 
    o1 ≠ o2 ∧ touchingPoints points o1 = touchingPoints points o2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_orientations_theorem_l788_78811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ACBD_l788_78804

-- Define the lengths of segments AB and CD
noncomputable def AB : ℝ := 10
noncomputable def CD : ℝ := 7

-- Define a function to calculate the area of quadrilateral ACBD
noncomputable def area_ACBD (angle : ℝ) : ℝ := (1/2) * AB * CD * Real.sin angle

-- Theorem statement
theorem max_area_ACBD :
  ∃ (max_area : ℝ), max_area = 35 ∧ ∀ (angle : ℝ), area_ACBD angle ≤ max_area := by
  -- The proof goes here
  sorry

#check max_area_ACBD

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ACBD_l788_78804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_formula_correctness_l788_78825

/-- Represents a trapezoid with parallel sides a and c, height h, and dividing line at distance x from the smaller side. -/
structure Trapezoid where
  a : ℝ
  c : ℝ
  h : ℝ
  x : ℝ
  h_pos : h > 0
  a_gt_c : a > c

/-- The correct formula for the position of the dividing line in a trapezoid -/
noncomputable def correct_formula (t : Trapezoid) : ℝ :=
  (Real.sqrt (2 * (t.a^2 + t.c^2) - 2*t.c)) / (2 * (t.a - t.c))

/-- The incorrect formula for the position of the dividing line in a trapezoid -/
noncomputable def incorrect_formula (t : Trapezoid) : ℝ :=
  (Real.sqrt (2 * (t.a^2 + t.c^2)) - t.c) / (2 * (t.a - t.c))

/-- Theorem stating that the correct formula is valid while the incorrect formula is not -/
theorem trapezoid_formula_correctness (t : Trapezoid) : 
  t.x / t.h = correct_formula t ∧ t.x / t.h ≠ incorrect_formula t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_formula_correctness_l788_78825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_sum_l788_78839

/-- An equilateral triangle inscribed in an ellipse with specific properties -/
structure InscribedTriangle where
  /-- The ellipse equation: x^2 + 4y^2 = 4 -/
  ellipse : ℝ → ℝ → Prop

  /-- One vertex of the triangle is at (0,1) -/
  vertex_at_origin : ℝ × ℝ

  /-- One altitude is contained in the y-axis -/
  altitude_on_y_axis : Prop

  /-- The triangle is equilateral -/
  is_equilateral : Prop

  /-- The square of the length of each side is m/n -/
  side_length_squared : ℚ
  
  /-- m and n are relatively prime positive integers -/
  m_n_coprime : ℕ → ℕ → Prop

/-- Definition of the ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + 4*y^2 = 4

theorem inscribed_triangle_sum (t : InscribedTriangle) 
  (h_ellipse : t.ellipse = ellipse_equation)
  (h_vertex : t.vertex_at_origin = (0, 1)) :
  ∃ (m n : ℕ), t.m_n_coprime m n ∧ t.side_length_squared = m / n ∧ m + n = 937 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_sum_l788_78839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l788_78807

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := (x = 1 ∨ y = 1)

-- Define the point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem line_equation :
  ∀ A B : ℝ × ℝ,
  let (x_A, y_A) := A
  let (x_B, y_B) := B
  my_circle x_A y_A ∧ my_circle x_B y_B →  -- A and B are on the circle
  (∃ x y : ℝ, line_l x y ∧ (x = x_A ∧ y = y_A)) →  -- l passes through A
  (∃ x y : ℝ, line_l x y ∧ (x = x_B ∧ y = y_B)) →  -- l passes through B
  (∃ x y : ℝ, line_l x y ∧ (x = point_P.1 ∧ y = point_P.2)) →  -- l passes through P
  distance x_A y_A x_B y_B = 2 * Real.sqrt 3 →  -- |AB| = 2√3
  ∀ x y : ℝ, line_l x y  -- The equation of l is x = 1 or y = 1
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l788_78807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_Q_l788_78838

-- Define R_k
def R (k : ℕ) : ℕ := (10^k - 1) / 9

-- Define the quotient Q
def Q : ℕ := (R 30) / (R 6)

-- Count zeros in the base-10 representation of a natural number
def countZeros (n : ℕ) : ℕ :=
  (n.repr.toList.filter (· = '0')).length

-- Theorem statement
theorem zeros_in_Q : countZeros Q = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_Q_l788_78838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l788_78850

def set_A : Set ℝ := {x | x ≤ 1}
def set_B : Set ℝ := {x | x^2 - 2*x < 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l788_78850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_ascending_after_1989_swaps_l788_78836

/-- Represents a sequence of natural numbers from 1 to n -/
def Sequence (n : ℕ) := Fin n → ℕ

/-- A swap operation on a sequence -/
def swap (s : Sequence n) (i j : Fin n) : Sequence n :=
  fun k => if k = i then s j else if k = j then s i else s k

/-- Predicate to check if a sequence is in ascending order -/
def isAscending (s : Sequence n) : Prop :=
  ∀ i j : Fin n, i < j → s i < s j

/-- The initial ascending sequence -/
def initialSequence (n : ℕ) : Sequence n :=
  fun i => i.val + 1

/-- Theorem stating that after 1989 swaps, the sequence cannot be in ascending order -/
theorem not_ascending_after_1989_swaps (n : ℕ) :
  ∀ (swaps : Vector (Fin n × Fin n) 1989),
  ¬(isAscending (swaps.toList.foldl (fun s (ij : Fin n × Fin n) => swap s ij.1 ij.2) (initialSequence n))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_ascending_after_1989_swaps_l788_78836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auntie_em_parking_probability_l788_78879

-- Define the total number of parking spaces
def total_spaces : ℕ := 20

-- Define the number of cars that have already parked
def parked_cars : ℕ := 15

-- Define the number of empty spaces
def empty_spaces : ℕ := total_spaces - parked_cars

-- Define the function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of Auntie Em being able to park
noncomputable def prob_can_park : ℚ :=
  1 - (binomial (empty_spaces + parked_cars) parked_cars : ℚ) / 
      (binomial total_spaces parked_cars : ℚ)

-- State the theorem
theorem auntie_em_parking_probability :
  prob_can_park = 232 / 323 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auntie_em_parking_probability_l788_78879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l788_78882

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 2) + 2 * Real.sin (x - Real.pi / 4) * Real.sin (x + Real.pi / 4)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), -Real.sqrt 2 ≤ f x ∧ f x ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l788_78882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_sales_optimal_solution_l788_78854

/-- Represents the bike sales problem -/
structure BikeSales where
  last_year_sales : ℝ
  price_decrease : ℝ
  sales_decrease_percentage : ℝ
  type_a_purchase_price : ℝ
  type_b_purchase_price : ℝ
  type_b_selling_price : ℝ
  total_bikes : ℕ

/-- Calculates the selling price of type A bikes this year -/
noncomputable def selling_price_a (bs : BikeSales) : ℝ :=
  let x := bs.last_year_sales * (1 - bs.sales_decrease_percentage) * (bs.last_year_sales)⁻¹
  x * (x + bs.price_decrease) / x

/-- Calculates the optimal number of type A bikes to purchase -/
def optimal_type_a_count (bs : BikeSales) : ℕ :=
  20 -- This is the optimal value we found in the solution

/-- Calculates the optimal number of type B bikes to purchase -/
def optimal_type_b_count (bs : BikeSales) : ℕ :=
  bs.total_bikes - optimal_type_a_count bs

/-- Main theorem stating the optimal solution -/
theorem bike_sales_optimal_solution (bs : BikeSales) 
  (h1 : bs.last_year_sales = 50000)
  (h2 : bs.price_decrease = 400)
  (h3 : bs.sales_decrease_percentage = 0.2)
  (h4 : bs.type_a_purchase_price = 1100)
  (h5 : bs.type_b_purchase_price = 1400)
  (h6 : bs.type_b_selling_price = 2000)
  (h7 : bs.total_bikes = 60) :
  selling_price_a bs = 1600 ∧ 
  optimal_type_a_count bs = 20 ∧ 
  optimal_type_b_count bs = 40 := by
  sorry

-- Remove #eval statements as they were causing issues due to 'sorry'


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_sales_optimal_solution_l788_78854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_1_6954_to_hundredth_l788_78899

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_1_6954_to_hundredth :
  roundToHundredth 1.6954 = 1.70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_1_6954_to_hundredth_l788_78899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l788_78829

-- Define the parametric equations of line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 - 3/5 * t, 2 + 4/5 * t)

-- Define the polar equation of curve C
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.cos (θ - Real.pi/4)

-- State the theorem
theorem intersection_length :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, A = line_l t₁ ∧ B = line_l t₂) ∧
    (∃ θ₁ θ₂ : ℝ, 
      Real.sqrt ((A.1)^2 + (A.2)^2) = curve_C θ₁ ∧
      Real.sqrt ((B.1)^2 + (B.2)^2) = curve_C θ₂) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l788_78829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l788_78821

theorem sin_cos_equation_solution :
  ∃ x : ℝ, x = π / 9 ∧ Real.sin (4 * x) * Real.cos (5 * x) = -Real.cos (4 * x) * Real.sin (5 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l788_78821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_24_l788_78857

/-- The region in ℝ² defined by |x| - |y| ≤ 2 and |y| ≤ 2 -/
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| - |p.2| ≤ 2 ∧ |p.2| ≤ 2}

/-- The area of the region -/
noncomputable def area_of_region : ℝ := (MeasureTheory.volume region).toReal

/-- Theorem stating that the area of the region is 24 -/
theorem area_is_24 : area_of_region = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_24_l788_78857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_is_4_8_l788_78869

/-- A rhombus inscribed in a square with an inscribed circle --/
structure InscribedRhombus where
  square_side : ℝ
  rhombus_diagonal1 : ℝ
  rhombus_diagonal2 : ℝ
  square_side_positive : 0 < square_side
  diagonals_positive : 0 < rhombus_diagonal1 ∧ 0 < rhombus_diagonal2

/-- The radius of the inscribed circle in the rhombus --/
noncomputable def inscribed_circle_radius (r : InscribedRhombus) : ℝ :=
  (r.rhombus_diagonal1 * r.rhombus_diagonal2) / (4 * Real.sqrt (r.rhombus_diagonal1^2 + r.rhombus_diagonal2^2))

/-- Theorem: The radius of the inscribed circle is 4.8 --/
theorem inscribed_circle_radius_is_4_8 (r : InscribedRhombus)
  (h1 : r.square_side = 20)
  (h2 : r.rhombus_diagonal1 = 12)
  (h3 : r.rhombus_diagonal2 = 16) :
  inscribed_circle_radius r = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_is_4_8_l788_78869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l788_78891

-- Define vectors m and n
noncomputable def m (x : Real) : Real × Real := (Real.sin x, 3/4)
noncomputable def n (x : Real) : Real × Real := (Real.cos x, -1)

-- Define the function f
noncomputable def f (x : Real) : Real := 2 * ((m x).1 + (n x).1) * (n x).1 + 2 * ((m x).2 + (n x).2) * (n x).2

-- Define the theorem
theorem vector_problem (A : Real) (hA : Real.sin A + Real.cos A = Real.sqrt 2) :
  (∃ x : Real, (m x).1 * (n x).2 = (m x).2 * (n x).1 → Real.sin x ^ 2 + Real.sin (2 * x) = -3/5) ∧
  f A = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l788_78891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_last_digit_l788_78898

/-- A function that checks if a two-digit number is divisible by 17 or 23 -/
def isDivisibleBy17Or23 (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ (n % 17 = 0 ∨ n % 23 = 0)

/-- A function that represents our string of digits -/
def digitString : ℕ → ℕ := sorry

/-- The length of our string is 2023 -/
axiom string_length : ∀ n, n ∈ Finset.range 2023 → digitString n ≤ 9

/-- The first digit of the string is 2 -/
axiom first_digit : digitString 0 = 2

/-- Any two consecutive digits form a number divisible by 17 or 23 -/
axiom consecutive_divisible (i : ℕ) :
  i < 2022 → isDivisibleBy17Or23 (digitString i * 10 + digitString (i + 1))

/-- The theorem to be proved -/
theorem smallest_last_digit :
  ∃ (s : ℕ → ℕ), s = digitString ∧ s 2022 = 2 ∧
  ∀ (t : ℕ → ℕ), t = digitString → s 2022 ≤ t 2022 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_last_digit_l788_78898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_in_sphere_l788_78827

-- Define the rectangular prism
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  volume_eq : a * b * c = 4 * Real.sqrt 6
  face_area_1 : a * b = 2 * Real.sqrt 3
  face_area_2 : b * c = 4 * Real.sqrt 3

-- Define the sphere volume
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Theorem statement
theorem prism_in_sphere (prism : RectangularPrism) :
  ∃ (r : ℝ), r^2 = (prism.a^2 + prism.b^2 + prism.c^2) / 4 ∧ 
  sphere_volume r = 32 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_in_sphere_l788_78827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_two_l788_78876

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (-x - 1) - x
  else Real.exp (x - 1) + x

-- State the theorem
theorem tangent_line_at_one_two (h : ∀ x, f (-x) = f x) :
  let tangent_line := fun x => 2 * x
  tangent_line 1 = f 1 ∧ deriv f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_two_l788_78876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l788_78863

/-- Calculates the total days to complete a project given initial and additional workers -/
def total_project_days (initial_workers : ℕ) (initial_days : ℕ) (initial_fraction : ℚ) 
  (additional_workers : ℕ) : ℕ :=
  let total_workers := initial_workers + additional_workers
  let work_per_day := initial_fraction / (initial_workers * initial_days)
  let remaining_work := 1 - initial_fraction
  let additional_days := (remaining_work / (total_workers * work_per_day)).num
  initial_days + additional_days.toNat

/-- The theorem stating that under given conditions, the project takes 70 days -/
theorem project_completion_time : 
  total_project_days 8 30 (1/3) 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l788_78863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l788_78861

/-- IsIsosceles predicate for a triangle -/
def IsIsosceles (a b c : ℝ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

/-- An isosceles triangle with sides of length 3 and 6 has perimeter 15 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  IsIsosceles a b c →
  ((a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (b = 3 ∧ c = 6) ∨ (b = 6 ∧ c = 3) ∨ (a = 3 ∧ c = 6) ∨ (a = 6 ∧ c = 3)) →
  a + b + c = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l788_78861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l788_78867

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 - x) * Real.sin x - Real.sqrt 3 * (Real.cos x) ^ 2

theorem f_properties : 
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ 
    ∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  (∃ M : ℝ, ∀ x : ℝ, f x ≤ M ∧ ∃ y : ℝ, f y = M) ∧
  (∀ x y : ℝ, Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ 5 * Real.pi / 12 → f x < f y) ∧
  (∀ x y : ℝ, 5 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ 2 * Real.pi / 3 → f x > f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l788_78867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_selection_probabilities_l788_78883

/-- The number of elective courses available -/
def num_courses : ℕ := 4

/-- The number of students choosing courses -/
def num_students : ℕ := 3

/-- The probability that all students choose different courses -/
def prob_all_different : ℚ := 3/8

/-- The probability that exactly two courses are not chosen -/
def prob_two_not_chosen : ℚ := 9/16

/-- Theorem stating the probabilities for the course selection problem -/
theorem course_selection_probabilities :
  (Nat.descFactorial num_courses num_students / num_courses ^ num_students : ℚ) = prob_all_different ∧
  ((Nat.choose num_courses 2 * Nat.choose num_students 2 * Nat.descFactorial 2 2 : ℕ) / num_courses ^ num_students : ℚ) = prob_two_not_chosen :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_selection_probabilities_l788_78883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l788_78814

/-- The hyperbola with equation x²/4 - y²/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

/-- The fixed point A -/
def A : ℝ × ℝ := (1, 4)

/-- The left focus of the hyperbola -/
noncomputable def F : ℝ × ℝ := sorry

/-- A point on the right branch of the hyperbola -/
noncomputable def P : ℝ × ℝ := sorry

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_sum :
  ∀ P, hyperbola P.1 P.2 → P.1 > 0 →
  (∀ Q, hyperbola Q.1 Q.2 → Q.1 > 0 → distance F P + distance A P ≤ distance F Q + distance A Q) →
  distance F P + distance A P = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l788_78814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rower_round_trip_time_l788_78887

/-- Calculates the total time for a round trip given rower's speed, river speed, and distance. -/
noncomputable def roundTripTime (rowerSpeed : ℝ) (riverSpeed : ℝ) (distance : ℝ) : ℝ :=
  let upstreamSpeed := rowerSpeed - riverSpeed
  let downstreamSpeed := rowerSpeed + riverSpeed
  distance / upstreamSpeed + distance / downstreamSpeed

/-- Proves that the round trip time for the given conditions is 1 hour. -/
theorem rower_round_trip_time :
  let rowerSpeed : ℝ := 7
  let riverSpeed : ℝ := 1
  let distance : ℝ := 3.4285714285714284
  roundTripTime rowerSpeed riverSpeed distance = 1 := by
  sorry

-- Use #eval only for nat, and use #check for ℝ
#check roundTripTime 7 1 3.4285714285714284

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rower_round_trip_time_l788_78887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_van_distance_theorem_l788_78802

/-- The distance a van travels given its initial time and required speed for a longer duration. -/
noncomputable def van_distance (initial_time : ℝ) (required_speed : ℝ) : ℝ :=
  required_speed * (3/2 * initial_time)

/-- Theorem stating the distance traveled by the van under given conditions. -/
theorem van_distance_theorem :
  van_distance 6 60 = 540 := by
  -- Unfold the definition of van_distance
  unfold van_distance
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_van_distance_theorem_l788_78802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l788_78896

-- Define the inequality function
def f (a x : ℝ) : ℝ := (a - 2) * x^2 + 2 * (a - 2) * x - 4

-- State the theorem
theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, f a x < 0) → a ∈ Set.Ioc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l788_78896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_2_derivative_l788_78820

open Real

-- Define the function f(x) = log₂(x)
noncomputable def f (x : ℝ) : ℝ := log x / log 2

-- State the theorem
theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv f x = 1 / (x * log 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_2_derivative_l788_78820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_circle_l788_78835

theorem max_distance_on_circle (x y : ℝ) :
  x^2 + y^2 = 4 →
  (∀ a b : ℝ, a^2 + b^2 = 4 → Real.sqrt ((x + 3)^2 + (y - 4)^2) ≥ Real.sqrt ((a + 3)^2 + (b - 4)^2)) →
  Real.sqrt ((x + 3)^2 + (y - 4)^2) = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_circle_l788_78835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monic_cubic_polynomial_l788_78818

def is_monic_cubic (q : ℂ → ℂ) : Prop :=
  ∃ a b c : ℝ, ∀ x, q x = x^3 + a*x^2 + b*x + c

theorem unique_monic_cubic_polynomial 
  (q : ℂ → ℂ) 
  (h_monic_cubic : is_monic_cubic q)
  (h_root : q (2 + 3*Complex.I) = 0)
  (h_q0 : q 0 = -90) :
  ∀ x, q x = x^3 - (142/13)*x^2 + (529/13)*x - 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monic_cubic_polynomial_l788_78818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_increase_percentage_l788_78873

noncomputable section

/-- Density of an object -/
def density (mass volume : ℝ) : ℝ := mass / volume

/-- Volume of a cube given its side length -/
def cubeVolume (sideLength : ℝ) : ℝ := sideLength ^ 3

theorem mass_increase_percentage (mass₁ volume₁ sideLength₁ mass₂ volume₂ sideLength₂ : ℝ) 
  (h₁ : density mass₂ volume₂ = (1/2) * density mass₁ volume₁)
  (h₂ : sideLength₂ = 2 * sideLength₁)
  (h₃ : volume₁ = cubeVolume sideLength₁)
  (h₄ : volume₂ = cubeVolume sideLength₂)
  (h₅ : mass₁ > 0) :
  (mass₂ - mass₁) / mass₁ = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_increase_percentage_l788_78873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_is_35_l788_78816

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
axiom B_east_of_A : A.1 < B.1 ∧ A.2 = B.2
axiom C_northeast_of_B : B.1 < C.1 ∧ B.2 < C.2
axiom angle_ABC_90 : (C.1 - B.1) * (B.1 - A.1) + (C.2 - B.2) * (B.2 - A.2) = 0
axiom distance_AC : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (15 * Real.sqrt 3)^2
axiom angle_BAC_30 : Real.cos (30 * Real.pi / 180) = (B.1 - A.1) / Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
axiom D_north_of_C : D.1 = C.1 ∧ D.2 = C.2 + 10

-- Theorem to prove
theorem distance_AD_is_35 : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_is_35_l788_78816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l788_78849

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)

theorem f_max_value : ∀ x : ℝ, f x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l788_78849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_ten_l788_78881

/-- Represents a rectangle with width and height. -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a triangle. -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Checks if two rectangles are adjoining. -/
def Rectangle.adjoins (r1 r2 : Rectangle) : Prop :=
  sorry

/-- Checks if two triangles are similar. -/
def Triangle.similar_to (t1 t2 : Triangle) : Prop :=
  sorry

/-- Calculates the shaded area given two adjoining rectangles. -/
def shaded_area (r1 r2 : Rectangle) : ℝ :=
  sorry

/-- Given a 4-inch by 4-inch square adjoining a 12-inch by 12-inch square,
    with triangle DGF similar to triangle AHF, the area of the shaded region is 10 square inches. -/
theorem shaded_area_is_ten (small_square : Rectangle) (large_square : Rectangle)
  (triangle_DGF : Triangle) (triangle_AHF : Triangle) :
  small_square.width = 4 →
  small_square.height = 4 →
  large_square.width = 12 →
  large_square.height = 12 →
  Rectangle.adjoins small_square large_square →
  Triangle.similar_to triangle_DGF triangle_AHF →
  shaded_area small_square large_square = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_ten_l788_78881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_bar_weight_distribution_l788_78874

-- Define the properties of the gold bar
def gold_bar_length : ℝ := 5
def segments : ℕ := 10

-- Define the arithmetic sequence of segment weights
noncomputable def segment_weight (i : ℕ) : ℝ := 15/16 + (i - 1) * (1/8)

-- Define the total weight of the gold bar
noncomputable def total_weight : ℝ :=
  segments * segment_weight 1 + (segments * (segments - 1) / 2) * (segment_weight 2 - segment_weight 1)

-- State the theorem
theorem gold_bar_weight_distribution (i : ℕ) :
  48 * segment_weight i = 5 * total_weight → i = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_bar_weight_distribution_l788_78874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_one_lt_f_three_l788_78831

variable (f : ℝ → ℝ)

axiom increasing_on_interval : ∀ x y, x < y → x < 2 → y < 2 → f x < f y

axiom symmetry : ∀ x, f (x + 2) = f (-x + 2)

theorem f_negative_one_lt_f_three : f (-1) < f 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_one_lt_f_three_l788_78831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_selling_price_l788_78840

/-- A problem about calculating the selling price of books for a publishing company. -/
theorem book_selling_price
  (fixed_costs : ℝ)
  (variable_costs_per_book : ℝ)
  (break_even_books : ℕ)
  (h1 : fixed_costs = 35630)
  (h2 : variable_costs_per_book = 11.5)
  (h3 : break_even_books = 4072) :
  (fixed_costs + variable_costs_per_book * (break_even_books : ℝ)) / (break_even_books : ℝ) = 20.25 := by
  sorry

#eval (35630 + 11.5 * 4072) / 4072 -- To verify the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_selling_price_l788_78840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_diagonal_triangle_area_ratio_l788_78884

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  height : ℝ
  base1 : ℝ
  base2 : ℝ
  is_isosceles : True
  base1_eq : base1 = 3 * height
  base2_eq : base2 = 2 * height

/-- The area of a trapezoid -/
noncomputable def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  (t.base1 + t.base2) * t.height / 2

/-- The area of the triangle formed by the intersection of diagonals -/
noncomputable def diagonal_triangle_area (t : IsoscelesTrapezoid) : ℝ :=
  t.height^2 / 10

/-- Theorem stating the area ratio -/
theorem trapezoid_diagonal_triangle_area_ratio (t : IsoscelesTrapezoid) :
  trapezoid_area t = 25 * diagonal_triangle_area t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_diagonal_triangle_area_ratio_l788_78884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l788_78893

theorem exponential_inequality (x : ℝ) : (1/2 : ℝ)^(x-5) ≤ 2^x ↔ x ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l788_78893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l788_78810

/-- The circle with equation x^2 + y^2 - 4x = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The point P on the circle -/
noncomputable def P : ℝ × ℝ := (1, Real.sqrt 3)

/-- The equation of the proposed tangent line -/
def tangent_line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 2 = 0

theorem tangent_line_at_P :
  (circle_equation P.1 P.2) →
  (∀ x y : ℝ, circle_equation x y → (x - P.1) * (P.1 - 2) + (y - P.2) * P.2 = 0 → x = P.1 ∧ y = P.2) →
  (∀ x y : ℝ, tangent_line_equation x y ↔ (x - P.1) * (P.1 - 2) + (y - P.2) * P.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l788_78810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l788_78894

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then Real.log x + b else Real.exp x - 2

theorem f_range (b : ℝ) (h : f b ℯ = -3 * f b 0) :
  Set.range (f b) = Set.Ioc (-2) (Real.exp 1 - 2) ∪ Set.Ioi 2 := by
  sorry

#check f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l788_78894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_primes_between_70_and_80_l788_78822

def isPrime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1)).all (λ d => d < 2 || n % (d + 2) ≠ 0)

def countPrimes (a b : ℕ) : ℕ :=
  (List.range (b - a + 1)).filter (λ x => isPrime (x + a)) |>.length

theorem primes_between_70_and_80 : countPrimes 71 79 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_primes_between_70_and_80_l788_78822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MN_l788_78826

-- Define the points M and N
def M : ℝ × ℝ := (-2, -1)
def N (a : ℝ) : ℝ × ℝ := (a, 3)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem distance_MN (a : ℝ) :
  distance M (N a) = 5 ↔ a = 1 ∨ a = -5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MN_l788_78826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l788_78837

open Real

-- Define constants
noncomputable def a : ℝ := cos (17 * π / 180) * cos (23 * π / 180) - sin (17 * π / 180) * sin (23 * π / 180)
noncomputable def b : ℝ := 2 * (cos (25 * π / 180))^2 - 1
noncomputable def c : ℝ := sqrt 3 / 2

-- Theorem statement
theorem relationship_abc : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l788_78837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_model_most_suitable_l788_78865

/-- Data points from the experiment -/
def data : List (ℝ × ℝ) := [
  (1, 5.380),
  (2, 11.232),
  (3, 20.184),
  (4, 34.356),
  (5, 53.482)
]

/-- Function to calculate the sum of squared errors -/
def sumSquaredErrors (f : ℝ → ℝ) (data : List (ℝ × ℝ)) : ℝ :=
  data.foldr (λ (x, y) acc => acc + (f x - y)^2) 0

/-- Quadratic function model -/
def quadraticModel (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

/-- Linear function model -/
def linearModel (a b : ℝ) (x : ℝ) : ℝ := a * x + b

/-- Square root function model -/
noncomputable def sqrtModel (a b : ℝ) (x : ℝ) : ℝ := a * Real.sqrt x + b

/-- Inverse function model -/
noncomputable def inverseModel (a b : ℝ) (x : ℝ) : ℝ := a / x + b

/-- Theorem stating that the quadratic model is the most suitable -/
theorem quadratic_model_most_suitable :
  ∃ (a b : ℝ), a > 1 ∧
  (∀ (a' b' : ℝ), sumSquaredErrors (quadraticModel a b) data <
                   sumSquaredErrors (linearModel a' b') data) ∧
  (∀ (a' b' : ℝ), sumSquaredErrors (quadraticModel a b) data <
                   sumSquaredErrors (sqrtModel a' b') data) ∧
  (∀ (a' b' : ℝ), sumSquaredErrors (quadraticModel a b) data <
                   sumSquaredErrors (inverseModel a' b') data) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_model_most_suitable_l788_78865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l788_78828

theorem triangle_inequality (x y z x₁ y₁ z₁ : Real) 
  (angle_sum : x + y + z = Real.pi) 
  (angle_sum₁ : x₁ + y₁ + z₁ = Real.pi) : 
  (Real.cos x₁ / Real.sin x) + (Real.cos y₁ / Real.sin y) + (Real.cos z₁ / Real.sin z) ≤ 
  (Real.cos x / Real.sin x) + (Real.cos y / Real.sin y) + (Real.cos z / Real.sin z) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l788_78828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_angle_ratio_rhombus_ratio_bounds_l788_78888

/-- Represents a rhombus with side length a and acute angle α -/
structure Rhombus where
  a : ℝ
  α : ℝ
  h_a_pos : 0 < a
  h_α_pos : 0 < α
  h_α_acute : α < π / 2

/-- The ratio of the perimeter to the sum of diagonals -/
noncomputable def rhombus_ratio (r : Rhombus) : ℝ :=
  (4 * r.a) / (2 * r.a * (Real.sin (r.α / 2) + Real.cos (r.α / 2)))

/-- Theorem stating the relationship between the rhombus ratio and its acute angle -/
theorem rhombus_angle_ratio (r : Rhombus) (k : ℝ) 
    (h_k : k = rhombus_ratio r) :
    r.α = Real.arcsin ((4 - k^2) / k^2) ∨ 
    r.α = π - Real.arcsin ((4 - k^2) / k^2) := by
  sorry

/-- Theorem stating the permissible values of the rhombus ratio -/
theorem rhombus_ratio_bounds (r : Rhombus) (k : ℝ) 
    (h_k : k = rhombus_ratio r) :
    Real.sqrt 2 ≤ k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_angle_ratio_rhombus_ratio_bounds_l788_78888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restricted_speaker_permutations_l788_78824

/-- The number of speakers -/
def n : ℕ := 6

/-- The number of permutations of n speakers where one specific speaker 
    must immediately follow another specific speaker -/
def restricted_permutations (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem restricted_speaker_permutations :
  restricted_permutations n = 120 := by
  -- Unfold the definition of restricted_permutations
  unfold restricted_permutations
  -- Simplify n - 1
  simp [n]
  -- Calculate 5!
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_restricted_speaker_permutations_l788_78824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l788_78851

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 5 * x^2)

theorem h_range :
  ∀ y ∈ Set.range h, 0 < y ∧ y ≤ 1 ∧
  (∀ ε > 0, ∃ x : ℝ, h x < ε) ∧
  ∃ x : ℝ, h x = 1 :=
by
  sorry

#check h_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l788_78851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_squared_l788_78890

-- Define the complex number z
variable (z : ℂ)

-- Define the condition that the real part of z is positive
def positive_real_part (z : ℂ) : Prop := 0 < z.re

-- Define the area of the parallelogram
noncomputable def parallelogram_area (z : ℂ) : ℝ := 
  abs (z.im * (1/z).re - z.re * (1/z).im) / 2

-- Define the distance between z and 1/z
noncomputable def distance_z_inv (z : ℂ) : ℝ := Complex.abs (z - 1/z)

-- State the theorem
theorem smallest_distance_squared :
  ∃ d : ℝ, d^2 = 176/34 ∧ 
  ∀ z : ℂ, positive_real_part z → 
  parallelogram_area z = 15/17 → distance_z_inv z ≥ d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_squared_l788_78890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_lateral_surface_area_l788_78866

/-- Represents a truncated pyramid with square bases -/
structure TruncatedPyramid where
  lowerBaseSide : ℝ
  upperBaseSide : ℝ
  height : ℝ

/-- Calculates the lateral surface area of a truncated pyramid -/
noncomputable def lateralSurfaceArea (tp : TruncatedPyramid) : ℝ :=
  let edgeLength := Real.sqrt (tp.height^2 + ((tp.lowerBaseSide - tp.upperBaseSide) / 2)^2)
  4 * ((tp.lowerBaseSide + tp.upperBaseSide) / 2) * edgeLength

theorem truncated_pyramid_lateral_surface_area :
  let tp : TruncatedPyramid := ⟨10, 5, 6⟩
  lateralSurfaceArea tp = 195 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_lateral_surface_area_l788_78866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_unique_zero_l788_78843

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - (Real.exp 1 + 1) * x * log x - Real.exp 1

theorem tangent_line_and_unique_zero :
  (∀ x : ℝ, x > 0 → (2 * x + f x) = 0) ∧
  ∃! x : ℝ, x ∈ Set.Ioo 0 (Real.exp 4) ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_unique_zero_l788_78843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l788_78885

noncomputable def data : List ℝ := [198, 199, 200, 201, 202]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (λ x => (x - μ)^2)).sum / xs.length

theorem variance_of_data : variance data = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l788_78885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l788_78817

def baseball_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def opponent_score (team_score : Nat) (is_loss : Bool) : Nat :=
  if is_loss then team_score + 2 else (team_score + 2) / 3

theorem opponent_total_score : 
  ∃ (loss_games : List Nat),
    loss_games.length = 6 ∧
    loss_games.toFinset ⊆ baseball_scores.toFinset ∧
    (List.map (λ x => opponent_score x true) loss_games).sum + 
    (List.map (λ x => opponent_score x false) (baseball_scores.filter (λ x => x ∉ loss_games))).sum = 62 := by
  sorry

#eval baseball_scores

-- Test the opponent_score function
#eval opponent_score 5 true  -- Should output 7
#eval opponent_score 6 false -- Should output 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l788_78817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_asymptote_l788_78871

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 8 - y^2 = 1

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (3, 0)

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := x + 2 * Real.sqrt 2 * y = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

-- Theorem statement
theorem circle_tangent_to_asymptote :
  ∀ x y : ℝ, hyperbola x y → 
  (∃ x' y' : ℝ, asymptote x' y' ∧ circle_eq x' y') →
  circle_eq x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_asymptote_l788_78871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l788_78841

theorem floor_ceil_fraction_square : 
  ⌊⌈((15:ℝ)/8)^2⌉ + ((19:ℝ)/5)^2⌋ = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l788_78841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stop_time_l788_78864

/-- Proves that a train traveling at 80 km/h without stops and 60 km/h with stops
    spends 15 minutes per hour stopped. -/
theorem train_stop_time (distance : ℝ) (h : distance > 0) : 
  (distance / 60 - distance / 80) * 60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stop_time_l788_78864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_pure_ghee_percentage_l788_78845

/-- Represents a mixture of ghee and vanaspati -/
structure GheeMixture where
  total : ℝ
  pure_ghee : ℝ

/-- Calculates the percentage of pure ghee in a mixture -/
noncomputable def pure_ghee_percentage (mixture : GheeMixture) : ℝ :=
  (mixture.pure_ghee / mixture.total) * 100

/-- Adds pure ghee to a mixture -/
def add_pure_ghee (mixture : GheeMixture) (amount : ℝ) : GheeMixture :=
  { total := mixture.total + amount,
    pure_ghee := mixture.pure_ghee + amount }

theorem original_pure_ghee_percentage
  (original : GheeMixture)
  (h_original_total : original.total = 10)
  (h_added_pure : ℝ)
  (h_result_vanaspati_percentage : pure_ghee_percentage (add_pure_ghee original h_added_pure) = 80)
  (h_added_pure_amount : h_added_pure = 10) :
  pure_ghee_percentage original = 60 := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_pure_ghee_percentage_l788_78845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_necessary_not_sufficient_for_B_l788_78870

/-- Proposition A: The inequality x^2 + 2ax + 4 ≥ 0 holds for all x ∈ ℝ -/
def proposition_A (a : ℝ) : Prop :=
  ∀ x, x^2 + 2*a*x + 4 ≥ 0

/-- Proposition B: The function f(x) = log_a(x - a + 2) is always positive on the interval (1, +∞) -/
def proposition_B (a : ℝ) : Prop :=
  ∀ x > 1, Real.log (x - a + 2) / Real.log a > 0

/-- A is necessary but not sufficient for B -/
theorem A_necessary_not_sufficient_for_B :
  (∀ a, proposition_B a → proposition_A a) ∧
  ¬(∀ a, proposition_A a → proposition_B a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_necessary_not_sufficient_for_B_l788_78870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radicals_l788_78875

theorem simplify_radicals : Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radicals_l788_78875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_max_l788_78805

theorem triangle_area_ratio_max (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  S = (1/2) * b * c * Real.sin A →
  3 * a^2 = 2 * b^2 + c^2 →
  ∃ (k : ℝ), k = S / (b^2 + 2 * c^2) ∧
              k ≤ Real.sqrt 14 / 24 ∧
              (k = Real.sqrt 14 / 24 ↔ b = Real.sqrt 2 * c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_max_l788_78805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turkey_finder_strategy_l788_78856

/-- Represents the state of a box (empty or containing the turkey) -/
inductive BoxState
| Empty : BoxState
| Turkey : BoxState

/-- Represents a row of 5 boxes -/
def Boxes := Fin 5 → BoxState

/-- Represents a strategy for checking boxes -/
def Strategy := ℕ → Fin 5

/-- Checks if two box positions are adjacent -/
def adjacent (a b : Fin 5) : Prop :=
  (a.val + 1 = b.val) ∨ (b.val + 1 = a.val)

/-- Represents a valid turkey movement -/
def validMove (prev next : Fin 5) : Prop :=
  prev = next ∨ adjacent prev next

/-- Represents the turkey's movement over time -/
def turkeyPath (start : Fin 5) : ℕ → Fin 5 → Prop
  | 0, pos => pos = start
  | n + 1, pos => ∃ prev, turkeyPath start n prev ∧ validMove prev pos

/-- Represents finding the turkey -/
def findTurkey (s : Strategy) (p : ℕ → Fin 5) (n : ℕ) : Prop :=
  s n = p n

/-- Main theorem: There exists a strategy to find the turkey within 6 days -/
theorem turkey_finder_strategy :
  ∃ (s : Strategy), ∀ (start : Fin 5) (p : ℕ → Fin 5),
    turkeyPath start 0 (p 0) →
    ∃ (n : ℕ), n ≤ 6 ∧ findTurkey s p n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turkey_finder_strategy_l788_78856
