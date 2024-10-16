import Mathlib

namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l488_48839

/-- Given a line passing through points (3, -2) and (-1, 6), 
    prove that it intersects the x-axis at (2, 0) -/
theorem line_intersects_x_axis : 
  ∀ (f : ℝ → ℝ), 
  (f 3 = -2) → 
  (f (-1) = 6) → 
  (∃ x : ℝ, f x = 0 ∧ x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l488_48839


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l488_48893

theorem algebraic_expression_value (a b : ℝ) (h : a + b = 3) :
  2 * (a + 2 * b) - (3 * a + 5 * b) + 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l488_48893


namespace NUMINAMATH_CALUDE_min_surface_pips_is_58_l488_48800

/-- Represents a standard die -/
structure StandardDie :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)

/-- Represents four dice glued in a 2x2 square configuration -/
structure GluedDice :=
  (dice : Fin 4 → StandardDie)

/-- Calculates the number of pips on the surface of glued dice -/
def surface_pips (gd : GluedDice) : ℕ :=
  sorry

/-- The minimum number of pips on the surface of glued dice -/
def min_surface_pips : ℕ :=
  sorry

theorem min_surface_pips_is_58 : min_surface_pips = 58 :=
  sorry

end NUMINAMATH_CALUDE_min_surface_pips_is_58_l488_48800


namespace NUMINAMATH_CALUDE_locus_of_E_l488_48894

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/48 + y^2/16 = 1

-- Define a point on the ellipse
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : C x y

-- Define symmetric points
def symmetric_y (p : PointOnC) : PointOnC :=
  ⟨-p.x, p.y, by sorry⟩

def symmetric_origin (p : PointOnC) : PointOnC :=
  ⟨-p.x, -p.y, by sorry⟩

def symmetric_x (p : PointOnC) : PointOnC :=
  ⟨p.x, -p.y, by sorry⟩

-- Define perpendicularity
def perpendicular (p q r s : ℝ × ℝ) : Prop :=
  (p.1 - q.1) * (r.1 - s.1) + (p.2 - q.2) * (r.2 - s.2) = 0

-- Define the locus of E
def locus_E (x y : ℝ) : Prop := x^2/12 + y^2/4 = 1

-- Main theorem
theorem locus_of_E (M : PointOnC) (N : PointOnC) 
  (h_diff : (M.x, M.y) ≠ (N.x, N.y))
  (h_perp : perpendicular (M.x, M.y) (N.x, N.y) (M.x, M.y) (-M.x, -M.y)) :
  ∃ (E : ℝ × ℝ), locus_E E.1 E.2 := by sorry

end NUMINAMATH_CALUDE_locus_of_E_l488_48894


namespace NUMINAMATH_CALUDE_job_completion_time_l488_48812

theorem job_completion_time (rateA rateB rateC : ℝ) 
  (h1 : 3 * (rateA + rateB) = 1)
  (h2 : 6 * (rateB + rateC) = 1)
  (h3 : 3.6 * (rateA + rateC) = 1) :
  1 / rateC = 18 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l488_48812


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l488_48878

/-- Given a point P with polar coordinates (1, π), 
    the equation of the line passing through P and perpendicular to the polar axis is ρ = -1 / (cos θ) -/
theorem perpendicular_line_equation (P : ℝ × ℝ) (h : P = (1, π)) :
  ∃ (f : ℝ → ℝ), (∀ θ, f θ = -1 / (Real.cos θ)) ∧ 
  (∀ ρ θ, (ρ * Real.cos θ = -1) ↔ (ρ = f θ)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l488_48878


namespace NUMINAMATH_CALUDE_only_one_correct_proposition_l488_48855

/-- A proposition about the relationship between lines and planes in 3D space -/
inductive GeometryProposition
  | InfinitePointsImpliesParallel
  | ParallelToPlaneImpliesParallelToLines
  | ParallelLineImpliesParallelToPlane
  | ParallelToPlaneImpliesNoIntersection

/-- Predicate to check if a geometry proposition is correct -/
def is_correct_proposition (p : GeometryProposition) : Prop :=
  match p with
  | GeometryProposition.InfinitePointsImpliesParallel => False
  | GeometryProposition.ParallelToPlaneImpliesParallelToLines => False
  | GeometryProposition.ParallelLineImpliesParallelToPlane => False
  | GeometryProposition.ParallelToPlaneImpliesNoIntersection => True

/-- Theorem stating that only one of the geometry propositions is correct -/
theorem only_one_correct_proposition :
  ∃! (p : GeometryProposition), is_correct_proposition p :=
sorry

end NUMINAMATH_CALUDE_only_one_correct_proposition_l488_48855


namespace NUMINAMATH_CALUDE_largest_number_below_threshold_l488_48816

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

theorem largest_number_below_threshold :
  (numbers.filter (λ x => x ≤ threshold)).maximum? = some (9/10) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_below_threshold_l488_48816


namespace NUMINAMATH_CALUDE_undefined_power_implies_m_equals_two_l488_48862

-- Define a proposition that (m-2)^0 is undefined
def is_undefined (m : ℝ) : Prop := (m - 2)^0 ≠ 1

-- Theorem statement
theorem undefined_power_implies_m_equals_two (m : ℝ) :
  is_undefined m → m = 2 := by sorry

end NUMINAMATH_CALUDE_undefined_power_implies_m_equals_two_l488_48862


namespace NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l488_48804

theorem largest_value_when_x_is_quarter (x : ℝ) (h : x = 1/4) :
  (1/x > x) ∧ (1/x > x^2) ∧ (1/x > (1/2)*x) ∧ (1/x > Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l488_48804


namespace NUMINAMATH_CALUDE_arcsin_cos_eq_x_div_3_solutions_l488_48841

theorem arcsin_cos_eq_x_div_3_solutions (x : Real) :
  -3 * Real.pi / 2 ≤ x ∧ x ≤ 3 * Real.pi / 2 →
  (Real.arcsin (Real.cos x) = x / 3 ↔ (x = 3 * Real.pi / 10 ∨ x = 3 * Real.pi / 8)) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_cos_eq_x_div_3_solutions_l488_48841


namespace NUMINAMATH_CALUDE_unique_four_digit_power_sum_l488_48879

theorem unique_four_digit_power_sum : ∃! (peru : ℕ), 
  1000 ≤ peru ∧ peru < 10000 ∧
  ∃ (p e r u : ℕ), 
    p > 0 ∧ p < 10 ∧ e < 10 ∧ r < 10 ∧ u < 10 ∧
    peru = 1000 * p + 100 * e + 10 * r + u ∧
    peru = (p + e + r + u) ^ u ∧
    peru = 4913 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_power_sum_l488_48879


namespace NUMINAMATH_CALUDE_light_switch_correspondence_l488_48814

/-- Represents a room in the house -/
structure Room (n : ℕ) where
  id : Fin (2^n)

/-- Represents a light switch in the house -/
structure Switch (n : ℕ) where
  id : Fin (2^n)

/-- A function that represents a check of switches -/
def Check (n : ℕ) := Fin (2^n) → Bool

/-- A sequence of checks -/
def CheckSequence (n : ℕ) (m : ℕ) := Fin m → Check n

/-- A bijection between rooms and switches -/
def Correspondence (n : ℕ) := {f : Room n → Switch n // Function.Bijective f}

/-- The main theorem stating that 2n checks are sufficient and 2n-1 checks are not -/
theorem light_switch_correspondence (n : ℕ) :
  (∃ (cs : CheckSequence n (2*n)), ∃ (c : Correspondence n), 
    ∀ (r : Room n), ∃ (s : Switch n), c.val r = s ∧ 
      ∀ (i : Fin (2*n)), cs i (r.id) = cs i (s.id)) ∧
  (∀ (cs : CheckSequence n (2*n - 1)), ¬∃ (c : Correspondence n), 
    ∀ (r : Room n), ∃ (s : Switch n), c.val r = s ∧ 
      ∀ (i : Fin (2*n - 1)), cs i (r.id) = cs i (s.id)) :=
sorry

end NUMINAMATH_CALUDE_light_switch_correspondence_l488_48814


namespace NUMINAMATH_CALUDE_age_ratio_l488_48803

def arun_future_age : ℕ := 30
def years_to_future : ℕ := 10
def deepak_age : ℕ := 50

theorem age_ratio :
  (arun_future_age - years_to_future) / deepak_age = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l488_48803


namespace NUMINAMATH_CALUDE_triangle_properties_area_condition1_area_condition2_l488_48825

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)

-- Define the main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.B ≠ π/2)
  (h2 : Real.cos (2 * t.B) = Real.sqrt 3 * Real.cos t.B - 1) :
  t.B = π/6 ∧ 
  ((Real.sin t.A = Real.sqrt 3 * Real.sin t.C ∧ t.b = 2 → t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3) ∨
   (2 * t.b = 3 * t.a ∧ t.b * Real.sin t.A = 1 → t.a * t.c * Real.sin t.B / 2 = (Real.sqrt 3 + 2 * Real.sqrt 2) / 2)) :=
by sorry

-- Define additional theorems for each condition
theorem area_condition1 (t : Triangle) 
  (h1 : t.B = π/6)
  (h2 : Real.sin t.A = Real.sqrt 3 * Real.sin t.C)
  (h3 : t.b = 2) :
  t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3 :=
by sorry

theorem area_condition2 (t : Triangle)
  (h1 : t.B = π/6)
  (h2 : 2 * t.b = 3 * t.a)
  (h3 : t.b * Real.sin t.A = 1) :
  t.a * t.c * Real.sin t.B / 2 = (Real.sqrt 3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_area_condition1_area_condition2_l488_48825


namespace NUMINAMATH_CALUDE_solution_properties_l488_48887

-- Define the equation (1) as a function
def equation_one (a b : ℝ) : Prop := a^2 - 5*b^2 = 1

-- Theorem statement
theorem solution_properties (a b : ℝ) (h : equation_one a b) :
  (equation_one a (-b)) ∧ (1 / (a + b * Real.sqrt 5) = a - b * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_properties_l488_48887


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l488_48832

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1027 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l488_48832


namespace NUMINAMATH_CALUDE_performance_orders_count_l488_48853

/-- The number of programs available --/
def total_programs : ℕ := 8

/-- The number of programs to be selected --/
def selected_programs : ℕ := 4

/-- The number of special programs (A and B) --/
def special_programs : ℕ := 2

/-- Calculate the number of different performance orders --/
def calculate_orders : ℕ :=
  -- First category: only one of A or B is selected
  (special_programs.choose 1) * ((total_programs - special_programs).choose (selected_programs - 1)) * (selected_programs.factorial) +
  -- Second category: both A and B are selected
  ((total_programs - special_programs).choose (selected_programs - special_programs)) * (special_programs.factorial) * ((selected_programs - special_programs).factorial)

/-- The theorem to be proved --/
theorem performance_orders_count : calculate_orders = 1140 := by
  sorry

end NUMINAMATH_CALUDE_performance_orders_count_l488_48853


namespace NUMINAMATH_CALUDE_counting_functions_l488_48869

/-- The number of strictly increasing functions from {1,2,...,m} to {1,2,...,n} -/
def strictlyIncreasingFunctions (m n : ℕ) : ℕ :=
  Nat.choose n m

/-- The number of increasing functions from {1,2,...,m} to {1,2,...,n} -/
def increasingFunctions (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

theorem counting_functions (m n : ℕ) (h : m ≠ 0 ∧ n ≠ 0) :
  (m ≤ n → strictlyIncreasingFunctions m n = Nat.choose n m) ∧
  increasingFunctions m n = Nat.choose (m + n) m := by
  sorry

end NUMINAMATH_CALUDE_counting_functions_l488_48869


namespace NUMINAMATH_CALUDE_triangle_dot_product_l488_48896

/-- Given a triangle ABC with |AB| = 4, |AC| = 1, and area = √3,
    prove that the dot product of AB and AC is ±2 -/
theorem triangle_dot_product (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1^2 + AB.2^2 = 16) →  -- |AB| = 4
  (AC.1^2 + AC.2^2 = 1) →   -- |AC| = 1
  (abs (AB.1 * AC.2 - AB.2 * AC.1) = 2 * Real.sqrt 3) →  -- Area = √3
  ((AB.1 * AC.1 + AB.2 * AC.2)^2 = 4) :=  -- Dot product squared = 4
by sorry

end NUMINAMATH_CALUDE_triangle_dot_product_l488_48896


namespace NUMINAMATH_CALUDE_quadratic_root_range_l488_48876

theorem quadratic_root_range (k : ℝ) : 
  (∃ α β : ℝ, 
    (7 * α^2 - (k + 13) * α + k^2 - k - 2 = 0) ∧ 
    (7 * β^2 - (k + 13) * β + k^2 - k - 2 = 0) ∧ 
    (0 < α) ∧ (α < 1) ∧ (1 < β) ∧ (β < 2)) →
  ((3 < k ∧ k < 4) ∨ (-2 < k ∧ k < -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l488_48876


namespace NUMINAMATH_CALUDE_specific_quadrilateral_area_l488_48872

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a quadrilateral given its four vertices -/
def quadrilateralArea (a b c d : Point) : ℝ := sorry

/-- The theorem stating that the area of the specific quadrilateral is 4.5 -/
theorem specific_quadrilateral_area :
  let a : Point := ⟨0, 0⟩
  let b : Point := ⟨0, 2⟩
  let c : Point := ⟨3, 2⟩
  let d : Point := ⟨3, 3⟩
  quadrilateralArea a b c d = 4.5 := by sorry

end NUMINAMATH_CALUDE_specific_quadrilateral_area_l488_48872


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l488_48827

/-- The ellipse with semi-major axis 13 and semi-minor axis 12 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 169) + (p.2^2 / 144) = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_focal_property (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  P ∈ Ellipse → F₁ ∈ Foci → F₂ ∈ Foci → distance P F₁ = 4 →
  distance P F₂ = 22 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l488_48827


namespace NUMINAMATH_CALUDE_graph_reflection_l488_48826

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the reflection across y-axis
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Statement: The graph of y = g(-x) is the reflection of y = g(x) across the y-axis
theorem graph_reflection (x : ℝ) : 
  reflect_y (x, g x) = (-x, g (-x)) := by sorry

end NUMINAMATH_CALUDE_graph_reflection_l488_48826


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l488_48830

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ := sorry

/-- Theorem: For a line with slope -3 and x-intercept (7,0), the y-intercept is (0,21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := (7, 0) }
  y_intercept l = (0, 21) := by sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l488_48830


namespace NUMINAMATH_CALUDE_val_coins_value_l488_48838

/-- Calculates the total value of Val's coins given the initial number of nickels and the number of additional nickels found. -/
def total_value (initial_nickels : ℕ) (found_nickels : ℕ) : ℚ :=
  let total_nickels := initial_nickels + found_nickels
  let dimes := 3 * initial_nickels
  let quarters := 2 * dimes
  let nickel_value := (5 : ℚ) / 100
  let dime_value := (10 : ℚ) / 100
  let quarter_value := (25 : ℚ) / 100
  (total_nickels : ℚ) * nickel_value + (dimes : ℚ) * dime_value + (quarters : ℚ) * quarter_value

theorem val_coins_value :
  total_value 20 40 = 39 := by
  sorry

end NUMINAMATH_CALUDE_val_coins_value_l488_48838


namespace NUMINAMATH_CALUDE_friends_recycled_pounds_l488_48811

/-- The number of pounds of paper recycled to earn one point -/
def pounds_per_point : ℕ := 4

/-- The number of pounds Paige recycled -/
def paige_pounds : ℕ := 14

/-- The total number of points earned by Paige and her friends -/
def total_points : ℕ := 4

/-- Calculate the number of points earned for a given number of pounds -/
def points_earned (pounds : ℕ) : ℕ :=
  pounds / pounds_per_point

/-- The number of pounds recycled by Paige's friends -/
def friends_pounds : ℕ := 4

theorem friends_recycled_pounds :
  friends_pounds = total_points * pounds_per_point - points_earned paige_pounds * pounds_per_point :=
by sorry

end NUMINAMATH_CALUDE_friends_recycled_pounds_l488_48811


namespace NUMINAMATH_CALUDE_skew_diagonal_cube_volume_l488_48858

/-- Represents a cube with skew diagonals on its surface. -/
structure SkewDiagonalCube where
  side_length : ℝ
  has_skew_diagonals : Bool
  skew_diagonal_distance : ℝ

/-- Theorem stating that for a cube with skew diagonals where the distance between two skew lines is 1,
    the volume of the cube is either 1 or 3√3. -/
theorem skew_diagonal_cube_volume 
  (cube : SkewDiagonalCube) 
  (h1 : cube.has_skew_diagonals = true) 
  (h2 : cube.skew_diagonal_distance = 1) : 
  cube.side_length ^ 3 = 1 ∨ cube.side_length ^ 3 = 3 * Real.sqrt 3 := by
  sorry

#check skew_diagonal_cube_volume

end NUMINAMATH_CALUDE_skew_diagonal_cube_volume_l488_48858


namespace NUMINAMATH_CALUDE_sum_of_factors_36_l488_48852

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_36 : sum_of_factors 36 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_36_l488_48852


namespace NUMINAMATH_CALUDE_taller_cylinder_radius_l488_48885

/-- Represents a cylindrical container --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Given two cylinders with the same volume, one with height four times the other,
    and the shorter cylinder having a radius of 10 units,
    prove that the radius of the taller cylinder is 40 units. --/
theorem taller_cylinder_radius
  (c1 c2 : Cylinder) -- Two cylindrical containers
  (h : ℝ) -- Height variable
  (volume_eq : c1.radius ^ 2 * c1.height = c2.radius ^ 2 * c2.height) -- Same volume
  (height_relation : c1.height = 4 * c2.height) -- One height is four times the other
  (shorter_radius : c2.radius = 10) -- Radius of shorter cylinder is 10 units
  : c1.radius = 40 := by
  sorry

end NUMINAMATH_CALUDE_taller_cylinder_radius_l488_48885


namespace NUMINAMATH_CALUDE_associated_number_equality_l488_48856

-- Define the associated number function
def associated_number (x : ℚ) : ℚ :=
  if x ≥ 0 then 2 * x - 1 else -2 * x + 1

-- State the theorem
theorem associated_number_equality (a b : ℚ) (ha : a > 0) (hb : b < 0) 
  (h_eq : associated_number a = associated_number b) : 
  (a + b)^2 - 2*a - 2*b = -1 := by sorry

end NUMINAMATH_CALUDE_associated_number_equality_l488_48856


namespace NUMINAMATH_CALUDE_inequality_proof_l488_48897

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  Real.sqrt (a * b^2 + a^2 * b) + Real.sqrt ((1 - a) * (1 - b)^2 + (1 - a)^2 * (1 - b)) < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l488_48897


namespace NUMINAMATH_CALUDE_guest_author_payment_l488_48891

theorem guest_author_payment (B : ℕ) (h1 : B < 10) (h2 : B > 0) 
  (h3 : (200 + 10 * B) % 14 = 0) : B = 8 := by
  sorry

end NUMINAMATH_CALUDE_guest_author_payment_l488_48891


namespace NUMINAMATH_CALUDE_sugar_price_reduction_l488_48829

/-- Calculates the percentage reduction in sugar price given the original price and the amount that can be bought after reduction. -/
theorem sugar_price_reduction 
  (original_price : ℝ) 
  (budget : ℝ) 
  (extra_amount : ℝ) 
  (h1 : original_price = 8) 
  (h2 : budget = 120) 
  (h3 : extra_amount = 1) 
  (h4 : budget / original_price + extra_amount = budget / (budget / (budget / original_price + extra_amount))) : 
  (original_price - budget / (budget / original_price + extra_amount)) / original_price * 100 = 6.25 := by
sorry

end NUMINAMATH_CALUDE_sugar_price_reduction_l488_48829


namespace NUMINAMATH_CALUDE_f_properties_l488_48851

/-- A function with a local minimum at x = 1 -/
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + 2*b*x

/-- The function has a local minimum of -1 at x = 1 -/
def has_local_min (a b : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - 1| < ε → f a b x ≥ f a b 1 ∧ f a b 1 = -1

/-- The range of f on [0,2] -/
def range_f (a b : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc 0 2, f a b x = y}

theorem f_properties :
  ∃ a b : ℝ, has_local_min a b ∧ a = 1/3 ∧ b = -1/2 ∧ range_f a b = Set.Icc (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l488_48851


namespace NUMINAMATH_CALUDE_victoria_friends_count_l488_48817

theorem victoria_friends_count (total_pairs : Nat) (shoes_per_person : Nat) : 
  total_pairs = 36 → shoes_per_person = 2 → (total_pairs * 2 - shoes_per_person) / shoes_per_person = 35 := by
  sorry

end NUMINAMATH_CALUDE_victoria_friends_count_l488_48817


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l488_48854

theorem gcd_of_squares_sum : Nat.gcd (130^2 + 250^2 + 360^2) (129^2 + 249^2 + 361^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l488_48854


namespace NUMINAMATH_CALUDE_shoes_sold_main_theorem_l488_48809

/-- Represents the inventory of a shoe shop -/
structure ShoeInventory where
  large_boots : Nat
  medium_sandals : Nat
  small_sneakers : Nat
  large_sandals : Nat
  medium_boots : Nat
  small_boots : Nat

/-- Calculates the total number of shoes in the inventory -/
def total_shoes (inventory : ShoeInventory) : Nat :=
  inventory.large_boots + inventory.medium_sandals + inventory.small_sneakers +
  inventory.large_sandals + inventory.medium_boots + inventory.small_boots

/-- Theorem: The shop sold 106 pairs of shoes -/
theorem shoes_sold (initial_inventory : ShoeInventory) (pairs_left : Nat) : Nat :=
  let initial_total := total_shoes initial_inventory
  initial_total - pairs_left

/-- Main theorem: The shop sold 106 pairs of shoes -/
theorem main_theorem : shoes_sold
  { large_boots := 22
    medium_sandals := 32
    small_sneakers := 24
    large_sandals := 45
    medium_boots := 35
    small_boots := 26 }
  78 = 106 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_main_theorem_l488_48809


namespace NUMINAMATH_CALUDE_square_rectangle_equal_area_l488_48865

theorem square_rectangle_equal_area (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  a^2 = b * c → a = Real.sqrt (b * c) := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_equal_area_l488_48865


namespace NUMINAMATH_CALUDE_trig_identity_proof_l488_48842

theorem trig_identity_proof : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 
  4 * Real.sin (10 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l488_48842


namespace NUMINAMATH_CALUDE_inequality_equivalence_l488_48822

theorem inequality_equivalence (x : ℝ) : (x - 2) / (x - 4) ≥ 3 ↔ x ∈ Set.Ioo 4 5 ∪ {5} := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l488_48822


namespace NUMINAMATH_CALUDE_probability_theorem_l488_48880

structure StudyGroup where
  total_members : ℕ
  women_percentage : ℚ
  men_percentage : ℚ
  women_lawyer_percentage : ℚ
  women_doctor_percentage : ℚ
  women_engineer_percentage : ℚ
  women_architect_percentage : ℚ
  women_finance_percentage : ℚ
  men_lawyer_percentage : ℚ
  men_doctor_percentage : ℚ
  men_engineer_percentage : ℚ
  men_architect_percentage : ℚ
  men_finance_percentage : ℚ

def probability_female_engineer_male_doctor_male_lawyer (group : StudyGroup) : ℚ :=
  group.women_percentage * group.women_engineer_percentage +
  group.men_percentage * group.men_doctor_percentage +
  group.men_percentage * group.men_lawyer_percentage

theorem probability_theorem (group : StudyGroup) 
  (h1 : group.women_percentage = 3/5)
  (h2 : group.men_percentage = 2/5)
  (h3 : group.women_engineer_percentage = 1/5)
  (h4 : group.men_doctor_percentage = 1/4)
  (h5 : group.men_lawyer_percentage = 3/10) :
  probability_female_engineer_male_doctor_male_lawyer group = 17/50 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l488_48880


namespace NUMINAMATH_CALUDE_sum_x_coordinates_above_line_l488_48895

def points : List (ℚ × ℚ) := [(2, 8), (5, 15), (10, 25), (15, 36), (19, 45), (22, 52), (25, 66)]

def isAboveLine (p : ℚ × ℚ) : Bool :=
  p.2 > 2 * p.1 + 5

def pointsAboveLine : List (ℚ × ℚ) :=
  points.filter isAboveLine

theorem sum_x_coordinates_above_line :
  (pointsAboveLine.map (·.1)).sum = 81 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_above_line_l488_48895


namespace NUMINAMATH_CALUDE_rectangle_grid_40_squares_l488_48837

/-- Represents a rectangle divided into squares -/
structure RectangleGrid where
  rows : ℕ
  cols : ℕ
  total_squares : ℕ
  h_total : rows * cols = total_squares
  h_more_than_one_row : rows > 1
  h_odd_rows : Odd rows

/-- The number of squares not in the middle row of a rectangle grid -/
def squares_not_in_middle_row (r : RectangleGrid) : ℕ :=
  r.total_squares - r.cols

theorem rectangle_grid_40_squares (r : RectangleGrid) 
  (h_40_squares : r.total_squares = 40) :
  squares_not_in_middle_row r = 32 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_grid_40_squares_l488_48837


namespace NUMINAMATH_CALUDE_horner_method_operation_count_l488_48847

/-- Polynomial coefficients in descending order of degree -/
def coefficients : List ℝ := [3, 4, -5, -6, 7, -8, 1]

/-- The point at which to evaluate the polynomial -/
def x : ℝ := 0.4

/-- Count of operations in Horner's method -/
structure OperationCount where
  multiplications : ℕ
  additions : ℕ

/-- Horner's method for polynomial evaluation -/
def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ := sorry

/-- Count operations in Horner's method -/
def count_operations (coeffs : List ℝ) : OperationCount := sorry

/-- Theorem: Horner's method for the given polynomial requires 6 multiplications and 6 additions -/
theorem horner_method_operation_count :
  let count := count_operations coefficients
  count.multiplications = 6 ∧ count.additions = 6 := by sorry

end NUMINAMATH_CALUDE_horner_method_operation_count_l488_48847


namespace NUMINAMATH_CALUDE_sum_equality_l488_48835

theorem sum_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d)
  (sum_nonzero : a + b ≠ 0)
  (product_eq : a * c = b * d) : 
  a + c = b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_equality_l488_48835


namespace NUMINAMATH_CALUDE_credit_card_problem_l488_48807

/-- Calculates the amount added to a credit card in the second month given the initial balance,
    interest rate, and final balance after two months. -/
def amount_added (initial_balance : ℚ) (interest_rate : ℚ) (final_balance : ℚ) : ℚ :=
  let first_month_balance := initial_balance * (1 + interest_rate)
  let x := (final_balance - first_month_balance * (1 + interest_rate)) / (1 + interest_rate)
  x

theorem credit_card_problem :
  let initial_balance : ℚ := 50
  let interest_rate : ℚ := 1/5
  let final_balance : ℚ := 96
  amount_added initial_balance interest_rate final_balance = 20 := by
  sorry

end NUMINAMATH_CALUDE_credit_card_problem_l488_48807


namespace NUMINAMATH_CALUDE_students_who_didnt_pass_l488_48883

def total_students : ℕ := 804
def pass_percentage : ℚ := 75 / 100

theorem students_who_didnt_pass (total : ℕ) (pass_rate : ℚ) : 
  total - (total * pass_rate).floor = 201 :=
by sorry

end NUMINAMATH_CALUDE_students_who_didnt_pass_l488_48883


namespace NUMINAMATH_CALUDE_antoinette_weight_l488_48850

theorem antoinette_weight (rupert_weight antoinette_weight : ℝ) : 
  antoinette_weight = 2 * rupert_weight - 7 →
  antoinette_weight + rupert_weight = 98 →
  antoinette_weight = 63 := by
  sorry

end NUMINAMATH_CALUDE_antoinette_weight_l488_48850


namespace NUMINAMATH_CALUDE_unique_positive_zero_implies_a_negative_l488_48866

/-- The function f(x) = ax³ + 3x² + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

/-- Theorem: If f(x) has a unique zero point x₀ > 0, then a ∈ (-∞, 0) -/
theorem unique_positive_zero_implies_a_negative
  (a : ℝ)
  (h_unique : ∃! x₀ : ℝ, f a x₀ = 0)
  (h_positive : ∃ x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) :
  a < 0 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_zero_implies_a_negative_l488_48866


namespace NUMINAMATH_CALUDE_left_handed_to_non_throwers_ratio_l488_48833

/- Define the football team -/
def total_players : ℕ := 70
def throwers : ℕ := 37
def right_handed : ℕ := 59

/- Theorem to prove the ratio -/
theorem left_handed_to_non_throwers_ratio :
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed := non_throwers - right_handed_non_throwers
  (left_handed : ℚ) / non_throwers = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_to_non_throwers_ratio_l488_48833


namespace NUMINAMATH_CALUDE_area_of_region_t_l488_48802

/-- Represents a rhombus PQRS -/
structure Rhombus where
  side_length : ℝ
  angle_q : ℝ

/-- Represents the region T inside the rhombus -/
def region_t (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- Calculates the area of a given set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem stating the area of region T in the given rhombus -/
theorem area_of_region_t (r : Rhombus) 
  (h1 : r.side_length = 4) 
  (h2 : r.angle_q = 150 * π / 180) : 
  abs (area (region_t r) - 1.034) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_t_l488_48802


namespace NUMINAMATH_CALUDE_basketball_team_selection_l488_48870

def total_players : ℕ := 15
def quadruplets : ℕ := 4
def team_size : ℕ := 7
def remaining_players : ℕ := total_players - quadruplets
def players_to_choose : ℕ := team_size - quadruplets

theorem basketball_team_selection :
  Nat.choose remaining_players players_to_choose = 165 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l488_48870


namespace NUMINAMATH_CALUDE_min_value_expression_l488_48848

theorem min_value_expression (a b : ℝ) (h : a - b^2 = 4) :
  ∃ (m : ℝ), m = 5 ∧ ∀ (x y : ℝ), x - y^2 = 4 → x^2 - 3*y^2 + x - 15 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l488_48848


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l488_48819

/-- Given that M(4,7) is the midpoint of line segment AB and A(5,3) is one endpoint,
    the product of the coordinates of point B is 33. -/
theorem midpoint_coordinate_product : 
  let A : ℝ × ℝ := (5, 3)
  let M : ℝ × ℝ := (4, 7)
  ∃ B : ℝ × ℝ, 
    (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) → 
    B.1 * B.2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l488_48819


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l488_48818

def solution_set (a b c : ℝ) : Set ℝ :=
  {x : ℝ | x ≤ -3 ∨ x ≥ 4}

theorem quadratic_inequality_theorem (a b c : ℝ) 
  (h : solution_set a b c = {x : ℝ | a * x^2 + b * x + c ≥ 0}) :
  (a > 0) ∧ 
  ({x : ℝ | c * x^2 - b * x + a < 0} = {x : ℝ | x < -1/4 ∨ x > 1/3}) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l488_48818


namespace NUMINAMATH_CALUDE_polynomial_simplification_l488_48868

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 6*x^2 + 9 + 11*x - 12*x^2 - 15 + 17*x + 18*x^2 - 2*x^3 = -2*x^3 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l488_48868


namespace NUMINAMATH_CALUDE_even_function_max_symmetry_l488_48808

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f has a maximum value on an interval [a, b] if there exists
    a point c in [a, b] such that f(c) ≥ f(x) for all x in [a, b] -/
def HasMaximumOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c

/-- If f is an even function and has a maximum value on [1, 7],
    then it also has a maximum value on [-7, -1] -/
theorem even_function_max_symmetry (f : ℝ → ℝ) :
  EvenFunction f → HasMaximumOn f 1 7 → HasMaximumOn f (-7) (-1) :=
by
  sorry

end NUMINAMATH_CALUDE_even_function_max_symmetry_l488_48808


namespace NUMINAMATH_CALUDE_january_first_is_tuesday_l488_48840

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Counts the occurrences of a specific day in a month -/
def countDayInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Main theorem: If January has 31 days, and there are exactly four Fridays and four Mondays, then January 1st is a Tuesday -/
theorem january_first_is_tuesday (jan : Month) :
  jan.days = 31 →
  countDayInMonth jan DayOfWeek.Friday = 4 →
  countDayInMonth jan DayOfWeek.Monday = 4 →
  jan.firstDay = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_january_first_is_tuesday_l488_48840


namespace NUMINAMATH_CALUDE_problem_solution_l488_48871

theorem problem_solution :
  (∀ x y : ℝ, 3 * (x - y)^2 - 6 * (x - y)^2 + 2 * (x - y)^2 = -(x - y)^2) ∧
  (∀ a b : ℝ, a^2 - 2*b = 2 → 4*a^2 - 8*b - 9 = -1) ∧
  (∀ a b c d : ℝ, a - 2*b = 4 → b - c = -5 → 3*c + d = 10 → (a + 3*c) - (2*b + c) + (b + d) = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l488_48871


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l488_48824

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l488_48824


namespace NUMINAMATH_CALUDE_parallel_lines_in_plane_l488_48875

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem parallel_lines_in_plane 
  (α β : Plane) (a b c : Line) :
  parallel a α →
  parallel b α →
  intersect β α c →
  contained_in a β →
  contained_in b β →
  parallel_lines a b :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_in_plane_l488_48875


namespace NUMINAMATH_CALUDE_bacon_students_count_l488_48874

-- Define the total number of students
def total_students : ℕ := 310

-- Define the number of students who suggested mashed potatoes
def mashed_potatoes_students : ℕ := 185

-- Theorem to prove
theorem bacon_students_count : total_students - mashed_potatoes_students = 125 := by
  sorry

end NUMINAMATH_CALUDE_bacon_students_count_l488_48874


namespace NUMINAMATH_CALUDE_power_ratio_simplification_l488_48886

theorem power_ratio_simplification : (10^2003 + 10^2001) / (10^2002 + 10^2002) = 101/20 := by
  sorry

end NUMINAMATH_CALUDE_power_ratio_simplification_l488_48886


namespace NUMINAMATH_CALUDE_function_is_zero_l488_48864

/-- A function f: ℝ → ℝ is bounded on (0,1) -/
def BoundedOn01 (f : ℝ → ℝ) : Prop :=
  ∃ M > 0, ∀ x ∈ Set.Ioo 0 1, |f x| ≤ M

/-- The functional equation that f satisfies -/
def SatisfiesFunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x^2 * f x - y^2 * f y = (x^2 - y^2) * f (x + y) - x * y * f (x - y)

theorem function_is_zero
  (f : ℝ → ℝ)
  (hb : BoundedOn01 f)
  (hf : SatisfiesFunctionalEq f) :
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_is_zero_l488_48864


namespace NUMINAMATH_CALUDE_concurrency_condition_l488_48846

/-- An isosceles triangle ABC with geometric progressions on its sides -/
structure GeometricTriangle where
  -- Triangle side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Isosceles condition
  isosceles : BC = CA
  -- Specific side lengths
  AB_length : AB = 4
  BC_length : BC = 6
  -- Geometric progressions on sides
  X : ℕ → ℝ  -- Points on AB
  Y : ℕ → ℝ  -- Points on CB
  Z : ℕ → ℝ  -- Points on AC
  -- Geometric progression conditions
  X_gp : ∀ n : ℕ, X (n + 1) - X n = 3 * (1/4)^n
  Y_gp : ∀ n : ℕ, Y (n + 1) - Y n = 3 * (1/2)^n
  Z_gp : ∀ n : ℕ, Z (n + 1) - Z n = 3 * (1/2)^n
  -- Initial conditions
  X_init : X 0 = 0
  Y_init : Y 0 = 0
  Z_init : Z 0 = 0

/-- The concurrency condition for the triangle -/
def concurrent (T : GeometricTriangle) (a b c : ℕ+) : Prop :=
  4^c.val - 1 = (2^a.val - 1) * (2^b.val - 1)

/-- The main theorem stating the conditions for concurrency -/
theorem concurrency_condition (T : GeometricTriangle) :
  ∀ a b c : ℕ+, concurrent T a b c ↔ (a = 1 ∧ b = 2 * c) ∨ (a = 2 * c ∧ b = 1) :=
sorry

end NUMINAMATH_CALUDE_concurrency_condition_l488_48846


namespace NUMINAMATH_CALUDE_min_columns_for_formation_l488_48806

theorem min_columns_for_formation (n : ℕ) : n ≥ 141 → ∃ k : ℕ, 8 * n = 225 * k + 3 :=
sorry

end NUMINAMATH_CALUDE_min_columns_for_formation_l488_48806


namespace NUMINAMATH_CALUDE_power_product_simplification_l488_48889

theorem power_product_simplification (x : ℝ) : (3 * x)^2 * x^2 = 9 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l488_48889


namespace NUMINAMATH_CALUDE_at_least_one_wrong_probability_l488_48882

/-- The probability of getting a single multiple-choice question wrong -/
def p_wrong : ℝ := 0.1

/-- The number of multiple-choice questions -/
def n : ℕ := 3

/-- The probability of getting at least one question wrong out of n questions -/
def p_at_least_one_wrong : ℝ := 1 - (1 - p_wrong) ^ n

theorem at_least_one_wrong_probability :
  p_at_least_one_wrong = 0.271 := by sorry

end NUMINAMATH_CALUDE_at_least_one_wrong_probability_l488_48882


namespace NUMINAMATH_CALUDE_elroy_extra_miles_l488_48823

/-- Proves that Elroy walks 5 more miles than last year's winner to collect the same amount -/
theorem elroy_extra_miles
  (last_year_rate : ℝ)
  (this_year_rate : ℝ)
  (last_year_amount : ℝ)
  (h1 : last_year_rate = 4)
  (h2 : this_year_rate = 2.75)
  (h3 : last_year_amount = 44) :
  (last_year_amount / this_year_rate) - (last_year_amount / last_year_rate) = 5 := by
sorry

end NUMINAMATH_CALUDE_elroy_extra_miles_l488_48823


namespace NUMINAMATH_CALUDE_company_x_employees_l488_48867

theorem company_x_employees (full_time : ℕ) (worked_year : ℕ) (neither : ℕ) (both : ℕ) :
  full_time = 80 →
  worked_year = 100 →
  neither = 20 →
  both = 30 →
  full_time + worked_year - both + neither = 170 := by
  sorry

end NUMINAMATH_CALUDE_company_x_employees_l488_48867


namespace NUMINAMATH_CALUDE_expression_evaluation_l488_48859

theorem expression_evaluation :
  let x : ℤ := -1
  let y : ℤ := -2
  let z : ℤ := 3
  3*x - 2*y - (2*x + 2*y - (2*x*y*z + x + 2*z) - 4*x + 2*z) - x*y*z = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l488_48859


namespace NUMINAMATH_CALUDE_facebook_bonus_calculation_l488_48815

/-- Calculates the bonus amount for each female mother employee at Facebook --/
theorem facebook_bonus_calculation (total_employees : ℕ) (non_mother_females : ℕ) 
  (annual_earnings : ℚ) (bonus_percentage : ℚ) :
  total_employees = 3300 →
  non_mother_females = 1200 →
  annual_earnings = 5000000 →
  bonus_percentage = 1/4 →
  ∃ (bonus_per_employee : ℚ),
    bonus_per_employee = 1250 ∧
    bonus_per_employee = (annual_earnings * bonus_percentage) / 
      (total_employees - (total_employees / 3) - non_mother_females) :=
by
  sorry


end NUMINAMATH_CALUDE_facebook_bonus_calculation_l488_48815


namespace NUMINAMATH_CALUDE_second_number_value_l488_48877

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 550 →
  a = 2 * b →
  c = (1 / 3) * a →
  b = 150 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l488_48877


namespace NUMINAMATH_CALUDE_complex_number_problem_l488_48857

def z (b : ℝ) : ℂ := 3 + b * Complex.I

theorem complex_number_problem (b : ℝ) 
  (h : ∃ k : ℝ, (1 + 3 * Complex.I) * z b = k * Complex.I) :
  z b = 3 + Complex.I ∧ Complex.abs ((z b) / (2 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l488_48857


namespace NUMINAMATH_CALUDE_addition_problem_l488_48860

theorem addition_problem (m n p q : ℕ) : 
  (1 ≤ m ∧ m ≤ 9) →
  (1 ≤ n ∧ n ≤ 9) →
  (1 ≤ p ∧ p ≤ 9) →
  (1 ≤ q ∧ q ≤ 9) →
  3 + 2 + q = 12 →
  1 + 6 + p + 8 = 24 →
  2 + n + 7 + 5 = 20 →
  m = 2 →
  m + n + p + q = 24 := by
sorry

end NUMINAMATH_CALUDE_addition_problem_l488_48860


namespace NUMINAMATH_CALUDE_g_of_4_equals_26_l488_48801

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 6

-- Theorem statement
theorem g_of_4_equals_26 : g 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_g_of_4_equals_26_l488_48801


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l488_48828

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x ≤ x + 1) ↔ (∀ x : ℝ, x > 0 → Real.log x > x + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l488_48828


namespace NUMINAMATH_CALUDE_special_line_equation_l488_48888

/-- A line passing through (-3, 4) with intercepts summing to 12 -/
def special_line (a b : ℝ) : Prop :=
  a + b = 12 ∧ -3 / a + 4 / b = 1

/-- The equation of the special line -/
def line_equation (x y : ℝ) : Prop :=
  x + 3 * y - 9 = 0 ∨ 4 * x - y + 16 = 0

/-- Theorem stating that the special line satisfies one of the two equations -/
theorem special_line_equation :
  ∀ a b : ℝ, special_line a b → ∃ x y : ℝ, line_equation x y :=
by
  sorry

end NUMINAMATH_CALUDE_special_line_equation_l488_48888


namespace NUMINAMATH_CALUDE_inverse_tangent_identity_l488_48881

/-- For all real x, if g(x) = arctan(x), then g((5x - x^5) / (1 + 5x^4)) = 5g(x) - g(x)^5 -/
theorem inverse_tangent_identity (g : ℝ → ℝ) (h : ∀ x, g x = Real.arctan x) :
  ∀ x, g ((5 * x - x^5) / (1 + 5 * x^4)) = 5 * g x - (g x)^5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_tangent_identity_l488_48881


namespace NUMINAMATH_CALUDE_second_discount_percentage_l488_48834

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount_percent : ℝ)
  (final_sale_price : ℝ)
  (h1 : original_price = 495)
  (h2 : first_discount_percent = 15)
  (h3 : final_sale_price = 378.675) :
  ∃ (second_discount_percent : ℝ),
    second_discount_percent = 10 ∧
    final_sale_price = original_price * (1 - first_discount_percent / 100) * (1 - second_discount_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l488_48834


namespace NUMINAMATH_CALUDE_locus_is_circle_l488_48810

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Calculates the sum of squares of distances from a point to the vertices of an isosceles triangle -/
def sumOfSquaredDistances (p : Point) (t : IsoscelesTriangle) : ℝ :=
  3 * p.x^2 + 4 * p.y^2 - 2 * t.height * p.y + t.height^2 + t.base^2

/-- Theorem: The locus of points with constant sum of squared distances to the vertices of an isosceles triangle is a circle iff the sum exceeds h^2 + b^2 -/
theorem locus_is_circle (t : IsoscelesTriangle) (a : ℝ) :
  (∃ (center : Point) (radius : ℝ), ∀ (p : Point), 
    sumOfSquaredDistances p t = a ↔ (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2) ↔ 
  a > t.height^2 + t.base^2 := by
  sorry

end NUMINAMATH_CALUDE_locus_is_circle_l488_48810


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l488_48884

-- Define the repeating decimal 2.36̄
def repeating_decimal : ℚ := 2 + 36 / 99

-- Theorem statement
theorem repeating_decimal_as_fraction :
  repeating_decimal = 26 / 11 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l488_48884


namespace NUMINAMATH_CALUDE_company_workforce_company_workforce_proof_l488_48831

theorem company_workforce (initial_female_percentage : ℚ) 
                          (additional_male_workers : ℕ) 
                          (final_female_percentage : ℚ) : Prop :=
  initial_female_percentage = 60 / 100 →
  additional_male_workers = 22 →
  final_female_percentage = 55 / 100 →
  ∃ (initial_employees final_employees : ℕ),
    (initial_employees : ℚ) * initial_female_percentage = 
      (final_employees : ℚ) * final_female_percentage ∧
    final_employees = initial_employees + additional_male_workers ∧
    final_employees = 264

-- The proof of the theorem
theorem company_workforce_proof : 
  company_workforce (60 / 100) 22 (55 / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_company_workforce_company_workforce_proof_l488_48831


namespace NUMINAMATH_CALUDE_part_one_part_two_l488_48821

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x - 3) + m / (3 - x) = 3

-- Part 1
theorem part_one (m : ℝ) :
  equation 2 m → m = 5 := by
  sorry

-- Part 2
theorem part_two (x m : ℝ) :
  equation x m → x > 0 → m < 9 ∧ m ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l488_48821


namespace NUMINAMATH_CALUDE_candy_distribution_l488_48845

theorem candy_distribution (total_candies : ℕ) (sour_percentage : ℚ) (num_people : ℕ) :
  total_candies = 300 →
  sour_percentage = 40 / 100 →
  num_people = 3 →
  (total_candies - (sour_percentage * total_candies).floor) / num_people = 60 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l488_48845


namespace NUMINAMATH_CALUDE_paddyfieldWarblersKingfishersRatio_l488_48890

/-- Represents the bird population in Goshawk-Eurasian nature reserve -/
structure BirdPopulation where
  total : ℝ
  hawks : ℝ
  paddyfieldWarblers : ℝ
  kingfishers : ℝ
  others : ℝ

/-- The conditions of the bird population in the nature reserve -/
def reserveConditions (pop : BirdPopulation) : Prop :=
  pop.total > 0 ∧
  pop.hawks = 0.3 * pop.total ∧
  pop.paddyfieldWarblers = 0.4 * (pop.total - pop.hawks) ∧
  pop.others = 0.35 * pop.total ∧
  pop.total = pop.hawks + pop.paddyfieldWarblers + pop.kingfishers + pop.others

/-- The theorem stating that 25% of paddyfield-warblers are kingfishers -/
theorem paddyfieldWarblersKingfishersRatio (pop : BirdPopulation) 
  (h : reserveConditions pop) : 
  pop.kingfishers / pop.paddyfieldWarblers = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_paddyfieldWarblersKingfishersRatio_l488_48890


namespace NUMINAMATH_CALUDE_cake_division_l488_48805

theorem cake_division (num_cakes : ℕ) (num_children : ℕ) (max_cuts : ℕ) :
  num_cakes = 9 →
  num_children = 4 →
  max_cuts = 2 →
  ∃ (whole_cakes : ℕ) (fractional_cake : ℚ),
    whole_cakes + fractional_cake = num_cakes / num_children ∧
    whole_cakes = 2 ∧
    fractional_cake = 1/4 ∧
    (∀ cake, cake ≤ max_cuts) :=
by sorry

end NUMINAMATH_CALUDE_cake_division_l488_48805


namespace NUMINAMATH_CALUDE_three_solutions_implies_b_equals_three_l488_48863

theorem three_solutions_implies_b_equals_three (a b : ℚ) (ha : |a| > 0) :
  (∃! (s : Finset ℚ), s.card = 3 ∧ ∀ x ∈ s, ‖|x - a| - b‖ = 3) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_implies_b_equals_three_l488_48863


namespace NUMINAMATH_CALUDE_l_shaped_playground_area_l488_48843

def large_rectangle_length : ℕ := 10
def large_rectangle_width : ℕ := 7
def small_rectangle_length : ℕ := 3
def small_rectangle_width : ℕ := 2
def num_small_rectangles : ℕ := 2

theorem l_shaped_playground_area :
  (large_rectangle_length * large_rectangle_width) -
  (num_small_rectangles * small_rectangle_length * small_rectangle_width) = 58 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_playground_area_l488_48843


namespace NUMINAMATH_CALUDE_scarves_per_box_l488_48899

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_items : ℕ) : 
  num_boxes = 7 → 
  mittens_per_box = 4 → 
  total_items = 49 → 
  (total_items - num_boxes * mittens_per_box) / num_boxes = 3 := by
sorry

end NUMINAMATH_CALUDE_scarves_per_box_l488_48899


namespace NUMINAMATH_CALUDE_jacks_hair_length_l488_48844

/-- Given the relative lengths of Kate's, Emily's, Logan's, and Jack's hair, prove that Jack's hair is 39 inches long. -/
theorem jacks_hair_length (logan_hair emily_hair kate_hair jack_hair : ℝ) : 
  logan_hair = 20 →
  emily_hair = logan_hair + 6 →
  kate_hair = emily_hair / 2 →
  jack_hair = 3 * kate_hair →
  jack_hair = 39 :=
by
  sorry

#check jacks_hair_length

end NUMINAMATH_CALUDE_jacks_hair_length_l488_48844


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l488_48873

theorem division_multiplication_equality : (120 / 4 / 2 * 3 : ℚ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l488_48873


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l488_48861

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 5 / 4 →  -- ratio of angles is 5:4
  |a - b| = 10 :=  -- positive difference is 10°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l488_48861


namespace NUMINAMATH_CALUDE_dot_product_theorem_l488_48898

def vector_dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

def vector_perpendicular (v w : ℝ × ℝ) : Prop :=
  vector_dot_product v w = 0

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem dot_product_theorem (x y : ℝ) : 
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (1, y)
  let c : ℝ × ℝ := (3, -6)
  vector_perpendicular a c → vector_parallel b c →
  vector_dot_product (a.1 + b.1, a.2 + b.2) c = 15 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l488_48898


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l488_48813

theorem smallest_m_for_integral_solutions :
  ∀ m : ℕ+,
  (∃ x : ℤ, 12 * x^2 - m * x + 504 = 0) →
  m ≥ 156 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l488_48813


namespace NUMINAMATH_CALUDE_max_a_value_l488_48849

/-- The function f as defined in the problem -/
def f (x k a : ℝ) : ℝ := x^2 - (k^2 - 5*a*k + 3)*x + 7

/-- The theorem stating the maximum value of a -/
theorem max_a_value :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (k : ℝ), k ∈ Set.Icc 0 2 → 
    ∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc k (k+a) → x₂ ∈ Set.Icc (k+2*a) (k+4*a) → 
      f x₁ k a ≥ f x₂ k a) ∧
  (∀ (a' : ℝ), a' > a → 
    ∃ (k : ℝ), k ∈ Set.Icc 0 2 ∧ 
      ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc k (k+a') ∧ x₂ ∈ Set.Icc (k+2*a') (k+4*a') ∧ 
        f x₁ k a' < f x₂ k a') ∧
  a = (2 * Real.sqrt 6 - 4) / 5 := by
sorry

end NUMINAMATH_CALUDE_max_a_value_l488_48849


namespace NUMINAMATH_CALUDE_bowling_team_weight_l488_48820

theorem bowling_team_weight (original_players : ℕ) (original_avg_weight : ℝ)
  (new_players : ℕ) (second_player_weight : ℝ) (new_avg_weight : ℝ)
  (h1 : original_players = 7)
  (h2 : original_avg_weight = 112)
  (h3 : new_players = 2)
  (h4 : second_player_weight = 60)
  (h5 : new_avg_weight = 106) :
  let total_players := original_players + new_players
  let original_total_weight := original_players * original_avg_weight
  let new_total_weight := total_players * new_avg_weight
  let first_player_weight := new_total_weight - original_total_weight - second_player_weight
  first_player_weight = 110 := by
sorry

end NUMINAMATH_CALUDE_bowling_team_weight_l488_48820


namespace NUMINAMATH_CALUDE_proportional_relationship_and_point_value_l488_48892

/-- Given that y is directly proportional to x-1 and y = 4 when x = 3,
    prove that the relationship between y and x is y = 2x - 2,
    and when the point (-1,m) lies on this graph, m = -4. -/
theorem proportional_relationship_and_point_value 
  (y : ℝ → ℝ) 
  (h1 : ∃ k : ℝ, ∀ x, y x = k * (x - 1)) 
  (h2 : y 3 = 4) :
  (∀ x, y x = 2*x - 2) ∧ 
  y (-1) = -4 := by
sorry

end NUMINAMATH_CALUDE_proportional_relationship_and_point_value_l488_48892


namespace NUMINAMATH_CALUDE_largest_sum_l488_48836

/-- A digit is a natural number between 0 and 9 inclusive. -/
def Digit : Type := {n : ℕ // n ≤ 9}

/-- The sum function for the given problem. -/
def sum (A B C : Digit) : ℕ := 111 * A.val + 10 * C.val + 2 * B.val

/-- The theorem stating that 976 is the largest possible 3-digit sum. -/
theorem largest_sum :
  ∀ A B C : Digit,
    A ≠ B → A ≠ C → B ≠ C →
    sum A B C ≤ 976 ∧
    (∃ A' B' C' : Digit, A' ≠ B' ∧ A' ≠ C' ∧ B' ≠ C' ∧ sum A' B' C' = 976) :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_l488_48836
