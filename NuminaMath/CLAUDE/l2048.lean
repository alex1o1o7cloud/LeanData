import Mathlib

namespace NUMINAMATH_CALUDE_exists_real_cube_less_than_one_no_rational_square_root_of_two_not_all_natural_cube_greater_than_square_all_real_square_plus_one_positive_l2048_204890

-- Statement 1
theorem exists_real_cube_less_than_one : ∃ x : ℝ, x^3 < 1 := by sorry

-- Statement 2
theorem no_rational_square_root_of_two : ¬ ∃ x : ℚ, x^2 = 2 := by sorry

-- Statement 3
theorem not_all_natural_cube_greater_than_square : 
  ¬ ∀ x : ℕ, x^3 > x^2 := by sorry

-- Statement 4
theorem all_real_square_plus_one_positive : 
  ∀ x : ℝ, x^2 + 1 > 0 := by sorry

end NUMINAMATH_CALUDE_exists_real_cube_less_than_one_no_rational_square_root_of_two_not_all_natural_cube_greater_than_square_all_real_square_plus_one_positive_l2048_204890


namespace NUMINAMATH_CALUDE_range_of_a_l2048_204805

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |2*x + 2| - |2*x - 2| ≤ a) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2048_204805


namespace NUMINAMATH_CALUDE_endpoint_is_200_l2048_204800

/-- The endpoint of a range of even integers starting from 20, given that its average
    is 35 greater than the average of even integers from 10 to 140 inclusive. -/
def endpoint : ℕ :=
  let start1 := 20
  let start2 := 10
  let end2 := 140
  let diff := 35
  let avg2 := (start2 + end2) / 2
  let endpoint := 2 * (avg2 + diff) - start1
  endpoint

theorem endpoint_is_200 : endpoint = 200 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_is_200_l2048_204800


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2048_204850

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 12/7
  let r : ℚ := a₂ / a₁
  r = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2048_204850


namespace NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l2048_204855

/-- The number of days in a week -/
def daysInWeek : ℕ := 7

/-- The daily water consumption of the first sibling -/
def sibling1DailyConsumption : ℕ := 8

/-- The daily water consumption of the second sibling -/
def sibling2DailyConsumption : ℕ := 7

/-- The daily water consumption of the third sibling -/
def sibling3DailyConsumption : ℕ := 9

/-- The total weekly water consumption of all siblings -/
def totalWeeklyConsumption : ℕ :=
  (sibling1DailyConsumption + sibling2DailyConsumption + sibling3DailyConsumption) * daysInWeek

theorem siblings_weekly_water_consumption :
  totalWeeklyConsumption = 168 := by
  sorry

end NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l2048_204855


namespace NUMINAMATH_CALUDE_congruence_sufficient_not_necessary_for_equal_area_l2048_204899

-- Define the property of two triangles being congruent
def are_congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the property of two triangles having equal area
def have_equal_area (t1 t2 : Triangle) : Prop := sorry

-- Theorem stating that congruence is sufficient but not necessary for equal area
theorem congruence_sufficient_not_necessary_for_equal_area :
  (∀ t1 t2 : Triangle, are_congruent t1 t2 → have_equal_area t1 t2) ∧
  (∃ t1 t2 : Triangle, have_equal_area t1 t2 ∧ ¬are_congruent t1 t2) := by sorry

end NUMINAMATH_CALUDE_congruence_sufficient_not_necessary_for_equal_area_l2048_204899


namespace NUMINAMATH_CALUDE_power_sum_geq_product_l2048_204881

theorem power_sum_geq_product (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_one : a + b + c = 1) : 
  a^4 + b^4 + c^4 ≥ a*b*c :=
by sorry

end NUMINAMATH_CALUDE_power_sum_geq_product_l2048_204881


namespace NUMINAMATH_CALUDE_surface_area_is_39_l2048_204893

/-- Represents the structure made of unit cubes -/
structure CubeStructure where
  total_cubes : Nat
  pyramid_base : Nat
  extension_height : Nat

/-- Calculates the exposed surface area of the cube structure -/
def exposed_surface_area (s : CubeStructure) : Nat :=
  sorry

/-- The theorem stating that the exposed surface area of the given structure is 39 square meters -/
theorem surface_area_is_39 (s : CubeStructure) 
  (h1 : s.total_cubes = 18)
  (h2 : s.pyramid_base = 3)
  (h3 : s.extension_height = 4) : 
  exposed_surface_area s = 39 :=
sorry

end NUMINAMATH_CALUDE_surface_area_is_39_l2048_204893


namespace NUMINAMATH_CALUDE_division_property_l2048_204846

theorem division_property (n : ℕ) : 
  (n / 5 = 248) ∧ (n % 5 = 4) → (n / 9 + n % 9 = 140) := by
  sorry

end NUMINAMATH_CALUDE_division_property_l2048_204846


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l2048_204863

theorem polynomial_product_sum (g h k : ℤ) : 
  (∀ d : ℤ, (5*d^2 + 4*d + g) * (4*d^2 + h*d - 5) = 20*d^4 + 11*d^3 - 9*d^2 + k*d - 20) →
  g + h + k = -16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l2048_204863


namespace NUMINAMATH_CALUDE_speed_limit_exceeders_l2048_204895

/-- Represents the percentage of motorists who exceed the speed limit -/
def exceed_limit_percent : ℝ := sorry

/-- Represents the percentage of all motorists who receive speeding tickets -/
def receive_ticket_percent : ℝ := 10

/-- Represents the percentage of speed limit exceeders who do not receive tickets -/
def no_ticket_percent : ℝ := 30

theorem speed_limit_exceeders :
  exceed_limit_percent = 14 :=
by
  have h1 : receive_ticket_percent = exceed_limit_percent * (100 - no_ticket_percent) / 100 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_speed_limit_exceeders_l2048_204895


namespace NUMINAMATH_CALUDE_slope_characterization_l2048_204816

/-- The set of all possible slopes for a line with y-intercept (0,3) that intersects
    the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -2/5 ∨ m ≥ 2/5}

/-- The equation of the line with slope m and y-intercept (0,3) -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem slope_characterization :
  ∀ m : ℝ, m ∈ possible_slopes ↔
    ∃ x : ℝ, ellipse_equation x (line_equation m x) := by sorry

end NUMINAMATH_CALUDE_slope_characterization_l2048_204816


namespace NUMINAMATH_CALUDE_complex_abs_ratio_bounds_l2048_204815

theorem complex_abs_ratio_bounds (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m M : ℝ),
    (∀ z w : ℂ, z ≠ 0 → w ≠ 0 → m ≤ Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ∧
                                 Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ≤ M) ∧
    m = 0 ∧
    M = 1 ∧
    M - m = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_abs_ratio_bounds_l2048_204815


namespace NUMINAMATH_CALUDE_square_area_increase_l2048_204874

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.35 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.8225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l2048_204874


namespace NUMINAMATH_CALUDE_jorkins_christmas_spending_l2048_204843

-- Define the type for British currency
structure BritishCurrency where
  pounds : ℕ
  shillings : ℕ

def BritishCurrency.toShillings (bc : BritishCurrency) : ℕ :=
  20 * bc.pounds + bc.shillings

def BritishCurrency.halfValue (bc : BritishCurrency) : ℕ :=
  bc.toShillings / 2

theorem jorkins_christmas_spending (initial : BritishCurrency) 
  (h1 : initial.halfValue = 20 * (initial.shillings / 2) + initial.pounds)
  (h2 : initial.shillings / 2 = initial.pounds)
  (h3 : initial.pounds = initial.shillings / 2) :
  initial = BritishCurrency.mk 19 18 := by
  sorry

#check jorkins_christmas_spending

end NUMINAMATH_CALUDE_jorkins_christmas_spending_l2048_204843


namespace NUMINAMATH_CALUDE_subsets_containing_neither_A_nor_B_l2048_204847

variable (X : Finset ℕ)
variable (A B : Finset ℕ)

theorem subsets_containing_neither_A_nor_B :
  X.card = 10 →
  A ⊆ X →
  B ⊆ X →
  A.card = 3 →
  B.card = 4 →
  Disjoint A B →
  (X.powerset.filter (λ S => ¬(A ⊆ S) ∧ ¬(B ⊆ S))).card = 840 :=
by sorry

end NUMINAMATH_CALUDE_subsets_containing_neither_A_nor_B_l2048_204847


namespace NUMINAMATH_CALUDE_g_sum_property_l2048_204836

def g (x : ℝ) : ℝ := 2 * x^8 + 3 * x^6 - 5 * x^4 + 7

theorem g_sum_property : g 10 = 15 → g 10 + g (-10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_property_l2048_204836


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_equality_l2048_204813

theorem binomial_expansion_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_equality_l2048_204813


namespace NUMINAMATH_CALUDE_system_solution_l2048_204866

theorem system_solution (x y a : ℝ) : 
  (4 * x + y = a ∧ 2 * x + 5 * y = 3 * a ∧ x = 2) → a = 18 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2048_204866


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2048_204812

theorem simplify_fraction_product : 5 * (14 / 3) * (27 / (-35)) * (9 / 7) = -6 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2048_204812


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2048_204880

/-- Given a hyperbola with the following properties:
    - Equation: x²/a² - y²/b² = 1
    - a > 0, b > 0
    - Focal distance is 8
    - Left vertex A is at (-a, 0)
    - Point B is at (0, b)
    - Right focus F is at (4, 0)
    - Dot product of BA and BF equals 2a
    The eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let A : ℝ × ℝ := (-a, 0)
  let B : ℝ × ℝ := (0, b)
  let F : ℝ × ℝ := (4, 0)
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    (x - (-a))^2 + y^2 = (x - 4)^2 + y^2) →
  (B.1 - A.1) * (F.1 - B.1) + (B.2 - A.2) * (F.2 - B.2) = 2 * a →
  4 / a = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2048_204880


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_1000_l2048_204803

theorem greatest_multiple_of_5_and_7_under_1000 : ∃ n : ℕ, 
  (n % 5 = 0) ∧ 
  (n % 7 = 0) ∧ 
  (n < 1000) ∧ 
  (∀ m : ℕ, (m % 5 = 0) ∧ (m % 7 = 0) ∧ (m < 1000) → m ≤ n) ∧
  (n = 980) := by
sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_1000_l2048_204803


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2048_204860

/-- A point is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The point P has coordinates (2, -3) -/
def P : ℝ × ℝ := (2, -3)

theorem point_in_fourth_quadrant :
  fourth_quadrant P.1 P.2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2048_204860


namespace NUMINAMATH_CALUDE_number_equation_l2048_204811

theorem number_equation : ∃ n : ℝ, 2 * 2 + n = 6 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2048_204811


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l2048_204810

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that if the difference between the third terms is 5 times the
    difference between the second terms, then the sum of the common ratios is 5. -/
theorem geometric_sequence_ratio_sum (k p r : ℝ) (hk : k ≠ 0) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  k * p^2 - k * r^2 = 5 * (k * p - k * r) → p + r = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l2048_204810


namespace NUMINAMATH_CALUDE_P_Q_disjoint_l2048_204854

def P : Set ℕ := {n | ∃ k, n = 2 * k^2 - 2 * k + 1}

def Q : Set ℕ := {n | n > 0 ∧ (Complex.I + 1)^(2*n) = 2^n * Complex.I}

theorem P_Q_disjoint : P ∩ Q = ∅ := by sorry

end NUMINAMATH_CALUDE_P_Q_disjoint_l2048_204854


namespace NUMINAMATH_CALUDE_line_perp_plane_transitive_l2048_204898

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_transitive 
  (m n : Line) (α : Plane) :
  para m n → perp n α → perp m α :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_transitive_l2048_204898


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2048_204825

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2048_204825


namespace NUMINAMATH_CALUDE_triangle_problem_l2048_204848

open Real

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : cos t.B = 4/5) : 
  (t.a = 5/3 → t.A = π/6) ∧ 
  (t.a + t.c = 2 * Real.sqrt 10 → 
    1/2 * t.a * t.c * sin t.B = 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2048_204848


namespace NUMINAMATH_CALUDE_f_is_convex_f_range_a_l2048_204888

/-- Definition of a convex function -/
def IsConvex (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f ((x₁ + x₂) / 2) ≤ (f x₁ + f x₂) / 2

/-- The quadratic function f(x) = ax^2 + x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

/-- Theorem: f is convex when a > 0 -/
theorem f_is_convex (a : ℝ) (ha : a > 0) : IsConvex (f a) := by sorry

/-- Theorem: Range of a when |f(x)| ≤ 1 for x ∈ [0,1] -/
theorem f_range_a (a : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1) ↔ a ∈ Set.Icc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_f_is_convex_f_range_a_l2048_204888


namespace NUMINAMATH_CALUDE_parabola_no_intersection_l2048_204857

/-- A parabola is defined by the equation y = -x^2 - 6x + m -/
def parabola (x m : ℝ) : ℝ := -x^2 - 6*x + m

/-- The parabola does not intersect the x-axis if it has no real roots -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x : ℝ, parabola x m ≠ 0

/-- If the parabola does not intersect the x-axis, then m < -9 -/
theorem parabola_no_intersection (m : ℝ) :
  no_intersection m → m < -9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_no_intersection_l2048_204857


namespace NUMINAMATH_CALUDE_football_inventory_solution_l2048_204878

/-- Represents the football inventory problem -/
structure FootballInventory where
  total_footballs : ℕ
  total_cost : ℕ
  football_a_purchase : ℕ
  football_a_marked : ℕ
  football_b_purchase : ℕ
  football_b_marked : ℕ
  football_a_discount : ℚ
  football_b_discount : ℚ

/-- The specific football inventory problem instance -/
def problem : FootballInventory :=
  { total_footballs := 200
  , total_cost := 14400
  , football_a_purchase := 80
  , football_a_marked := 120
  , football_b_purchase := 60
  , football_b_marked := 90
  , football_a_discount := 1/5
  , football_b_discount := 1/10
  }

/-- Theorem stating the solution to the football inventory problem -/
theorem football_inventory_solution (p : FootballInventory) 
  (h1 : p = problem) : 
  ∃ (a b profit : ℕ), 
    a + b = p.total_footballs ∧ 
    a * p.football_a_purchase + b * p.football_b_purchase = p.total_cost ∧
    a = 120 ∧ 
    b = 80 ∧
    profit = a * (p.football_a_marked * (1 - p.football_a_discount) - p.football_a_purchase) + 
             b * (p.football_b_marked * (1 - p.football_b_discount) - p.football_b_purchase) ∧
    profit = 3600 :=
by
  sorry

end NUMINAMATH_CALUDE_football_inventory_solution_l2048_204878


namespace NUMINAMATH_CALUDE_cricketer_average_last_four_matches_l2048_204814

/-- Calculates the average score for the last 4 matches of a cricketer given the average score for all 10 matches and the average score for the first 6 matches. -/
def average_last_four_matches (total_average : ℚ) (first_six_average : ℚ) : ℚ :=
  let total_runs := total_average * 10
  let first_six_runs := first_six_average * 6
  let last_four_runs := total_runs - first_six_runs
  last_four_runs / 4

/-- Theorem stating that given a cricketer with an average score of 38.9 runs for 10 matches
    and an average of 42 runs for the first 6 matches, the average for the last 4 matches is 34.25 runs. -/
theorem cricketer_average_last_four_matches :
  average_last_four_matches (389 / 10) 42 = 34.25 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_last_four_matches_l2048_204814


namespace NUMINAMATH_CALUDE_symmetric_point_coords_l2048_204856

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the y-axis. -/
def symmetricYAxis (a b : Point2D) : Prop :=
  b.x = -a.x ∧ b.y = a.y

/-- Theorem: If point B is symmetric to point A(2, -1) with respect to the y-axis,
    then the coordinates of point B are (-2, -1). -/
theorem symmetric_point_coords :
  let a : Point2D := ⟨2, -1⟩
  let b : Point2D := ⟨-2, -1⟩
  symmetricYAxis a b → b = ⟨-2, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coords_l2048_204856


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2048_204876

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The line y = x -/
def bisector_line : Line := { a := 1, b := -1, c := 0 }

/-- Checks if a line is the angle bisector of two other lines -/
def is_angle_bisector (bisector : Line) (l1 : Line) (l2 : Line) : Prop := sorry

/-- Theorem: If the bisector of the angle between lines l₁ and l₂ is y = x,
    and the equation of l₁ is ax + by + c = 0 (ab > 0),
    then the equation of l₂ is bx + ay + c = 0 -/
theorem symmetric_line_equation (l1 : Line) (l2 : Line) 
    (h1 : is_angle_bisector bisector_line l1 l2)
    (h2 : l1.a * l1.b > 0) : 
  l2.a = l1.b ∧ l2.b = l1.a ∧ l2.c = l1.c := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2048_204876


namespace NUMINAMATH_CALUDE_correct_mean_after_errors_l2048_204820

theorem correct_mean_after_errors (n : ℕ) (initial_mean : ℚ) 
  (error1_actual error1_copied error2_actual error2_copied error3_actual error3_copied : ℚ) :
  n = 50 ∧ 
  initial_mean = 325 ∧
  error1_actual = 200 ∧ error1_copied = 150 ∧
  error2_actual = 175 ∧ error2_copied = 220 ∧
  error3_actual = 592 ∧ error3_copied = 530 →
  let incorrect_sum := n * initial_mean
  let correction := (error1_actual - error1_copied) + (error3_actual - error3_copied) - (error2_actual - error2_copied)
  let corrected_sum := incorrect_sum + correction
  let correct_mean := corrected_sum / n
  correct_mean = 326.34 := by
sorry


end NUMINAMATH_CALUDE_correct_mean_after_errors_l2048_204820


namespace NUMINAMATH_CALUDE_area_under_curve_l2048_204891

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the bounds
def a : ℝ := 0
def b : ℝ := 2

-- State the theorem
theorem area_under_curve : 
  (∫ x in a..b, f x) = 4 := by sorry

end NUMINAMATH_CALUDE_area_under_curve_l2048_204891


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2048_204852

/-- A rhombus with diagonals in ratio 1:2 and shorter diagonal 4 cm has side length 2√5 cm -/
theorem rhombus_side_length (d1 d2 side : ℝ) : 
  d1 > 0 → -- shorter diagonal is positive
  d2 = 2 * d1 → -- ratio of diagonals is 1:2
  d1 = 4 → -- shorter diagonal is 4 cm
  side^2 = (d1/2)^2 + (d2/2)^2 → -- Pythagorean theorem for half-diagonals
  side = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2048_204852


namespace NUMINAMATH_CALUDE_max_elevation_l2048_204873

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 200 * t - 20 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (t' : ℝ), s t' ≤ s t ∧ s t = 500 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_l2048_204873


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2048_204828

theorem quadratic_inequality_solution (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < a ∧ a < 1 → (a * x^2 - (a + 1) * x + 1 < 0 ↔ 1 < x ∧ x < 1/a)) ∧
  (∀ x : ℝ, a > 1 → (a * x^2 - (a + 1) * x + 1 < 0 ↔ 1/a < x ∧ x < 1)) ∧
  (a = 1 → ¬∃ x : ℝ, a * x^2 - (a + 1) * x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2048_204828


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2048_204892

theorem inverse_sum_reciprocal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (a * b + a * c + b * c) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2048_204892


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l2048_204885

theorem sqrt_sum_equals_seven (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l2048_204885


namespace NUMINAMATH_CALUDE_cougar_sleep_duration_l2048_204821

/-- Given a cougar's nightly sleep duration C and a zebra's nightly sleep duration Z,
    where Z = C + 2 and C + Z = 70, prove that C = 34. -/
theorem cougar_sleep_duration (C Z : ℕ) (h1 : Z = C + 2) (h2 : C + Z = 70) : C = 34 := by
  sorry

end NUMINAMATH_CALUDE_cougar_sleep_duration_l2048_204821


namespace NUMINAMATH_CALUDE_cuboid_height_theorem_l2048_204802

/-- Represents a cuboid (rectangular box) -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- Theorem: A cuboid with volume 315, width 9, and length 7 has height 5 -/
theorem cuboid_height_theorem (c : Cuboid) 
  (h_volume : volume c = 315)
  (h_width : c.width = 9)
  (h_length : c.length = 7) : 
  c.height = 5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_theorem_l2048_204802


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l2048_204879

theorem cupboard_cost_price (selling_price selling_price_increased : ℝ) 
  (h1 : selling_price = 0.84 * 5625)
  (h2 : selling_price_increased = 1.16 * 5625)
  (h3 : selling_price_increased - selling_price = 1800) : 
  5625 = 5625 := by sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l2048_204879


namespace NUMINAMATH_CALUDE_pure_imaginary_product_imaginary_part_quotient_l2048_204824

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m + Complex.I
def z₂ (m : ℝ) : ℂ := 2 + m * Complex.I

-- Theorem 1
theorem pure_imaginary_product (m : ℝ) :
  (z₁ m * z₂ m).re = 0 → m = 0 := by sorry

-- Theorem 2
theorem imaginary_part_quotient (m : ℝ) :
  z₁ m ^ 2 - 2 * z₁ m + 2 = 0 →
  (z₂ m / z₁ m).im = -1/2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_imaginary_part_quotient_l2048_204824


namespace NUMINAMATH_CALUDE_quarters_found_l2048_204877

def dime_value : ℚ := 0.1
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01
def quarter_value : ℚ := 0.25

def num_dimes : ℕ := 3
def num_nickels : ℕ := 4
def num_pennies : ℕ := 200
def total_amount : ℚ := 5

theorem quarters_found :
  ∃ (num_quarters : ℕ),
    (num_quarters : ℚ) * quarter_value +
    (num_dimes : ℚ) * dime_value +
    (num_nickels : ℚ) * nickel_value +
    (num_pennies : ℚ) * penny_value = total_amount ∧
    num_quarters = 10 :=
by sorry

end NUMINAMATH_CALUDE_quarters_found_l2048_204877


namespace NUMINAMATH_CALUDE_onion_chop_time_is_four_l2048_204871

/-- Represents the time in minutes for Bill's omelet preparation tasks -/
structure OmeletPrep where
  pepper_chop_time : ℕ
  cheese_grate_time : ℕ
  assemble_cook_time : ℕ
  total_peppers : ℕ
  total_onions : ℕ
  total_omelets : ℕ
  total_prep_time : ℕ

/-- Calculates the time to chop an onion given the omelet preparation details -/
def time_to_chop_onion (prep : OmeletPrep) : ℕ :=
  let pepper_time := prep.pepper_chop_time * prep.total_peppers
  let cheese_time := prep.cheese_grate_time * prep.total_omelets
  let cook_time := prep.assemble_cook_time * prep.total_omelets
  let remaining_time := prep.total_prep_time - (pepper_time + cheese_time + cook_time)
  remaining_time / prep.total_onions

/-- Theorem stating that it takes 4 minutes to chop an onion given the specific conditions -/
theorem onion_chop_time_is_four : 
  let prep : OmeletPrep := {
    pepper_chop_time := 3,
    cheese_grate_time := 1,
    assemble_cook_time := 5,
    total_peppers := 4,
    total_onions := 2,
    total_omelets := 5,
    total_prep_time := 50
  }
  time_to_chop_onion prep = 4 := by
  sorry

end NUMINAMATH_CALUDE_onion_chop_time_is_four_l2048_204871


namespace NUMINAMATH_CALUDE_cuboid_probabilities_l2048_204838

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of unit cubes in a cuboid -/
def Cuboid.totalUnitCubes (c : Cuboid) : ℕ := c.length * c.width * c.height

/-- Calculates the number of unit cubes with no faces painted -/
def Cuboid.noPaintedFaces (c : Cuboid) : ℕ := (c.length - 2) * (c.width - 2) * (c.height - 2)

/-- Calculates the number of unit cubes with two faces painted -/
def Cuboid.twoFacesPainted (c : Cuboid) : ℕ :=
  (c.length - 2) * c.width + (c.width - 2) * c.height + (c.height - 2) * c.length

/-- Calculates the number of unit cubes with three faces painted -/
def Cuboid.threeFacesPainted (c : Cuboid) : ℕ := 8

theorem cuboid_probabilities (c : Cuboid) (h1 : c.length = 3) (h2 : c.width = 4) (h3 : c.height = 5) :
  (c.noPaintedFaces : ℚ) / c.totalUnitCubes = 1 / 10 ∧
  ((c.twoFacesPainted + c.threeFacesPainted : ℚ) / c.totalUnitCubes = 8 / 15) := by
  sorry

end NUMINAMATH_CALUDE_cuboid_probabilities_l2048_204838


namespace NUMINAMATH_CALUDE_problem_solution_l2048_204807

theorem problem_solution (x y : ℝ) : 
  x / y = 15 / 5 → y = 25 → x = 75 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2048_204807


namespace NUMINAMATH_CALUDE_total_lost_or_given_equals_sum_l2048_204884

/-- Represents the number of crayons in various states --/
structure CrayonCounts where
  given_to_friends : ℕ
  lost : ℕ
  total_lost_or_given : ℕ

/-- Theorem stating that the total number of crayons lost or given away
    is equal to the sum of crayons given to friends and crayons lost --/
theorem total_lost_or_given_equals_sum (c : CrayonCounts)
  (h1 : c.given_to_friends = 52)
  (h2 : c.lost = 535)
  (h3 : c.total_lost_or_given = 587) :
  c.total_lost_or_given = c.given_to_friends + c.lost := by
  sorry

#check total_lost_or_given_equals_sum

end NUMINAMATH_CALUDE_total_lost_or_given_equals_sum_l2048_204884


namespace NUMINAMATH_CALUDE_five_digit_with_eight_count_l2048_204819

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def contains_eight (n : ℕ) : Prop := ∃ (d : ℕ), d < 5 ∧ (n / 10^d) % 10 = 8

def count_five_digit : ℕ := 90000

def count_without_eight : ℕ := 52488

theorem five_digit_with_eight_count :
  (count_five_digit - count_without_eight) = 37512 :=
sorry

end NUMINAMATH_CALUDE_five_digit_with_eight_count_l2048_204819


namespace NUMINAMATH_CALUDE_max_x_value_l2048_204897

theorem max_x_value (x : ℝ) : 
  ((6*x - 15)/(4*x - 5))^2 - 3*((6*x - 15)/(4*x - 5)) - 10 = 0 → x ≤ 25/14 :=
by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l2048_204897


namespace NUMINAMATH_CALUDE_boat_round_trip_time_l2048_204817

theorem boat_round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 2)
  (h3 : distance = 7560)
  : (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = 960 :=
by sorry

end NUMINAMATH_CALUDE_boat_round_trip_time_l2048_204817


namespace NUMINAMATH_CALUDE_seven_solutions_condition_l2048_204869

-- Define the function f
def f (x : ℝ) : ℝ := |x^2 - 1| - 1

-- State the theorem
theorem seven_solutions_condition (b c : ℝ) :
  (∃! (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, f x ^ 2 - b * f x + c = 0) ↔ 
  (c ≤ 0 ∧ 0 < b ∧ b < 1) :=
sorry

end NUMINAMATH_CALUDE_seven_solutions_condition_l2048_204869


namespace NUMINAMATH_CALUDE_coefficient_x4_in_q_squared_l2048_204837

/-- Given q(x) = x^5 - 4x^2 + 3, prove that the coefficient of x^4 in (q(x))^2 is 16 -/
theorem coefficient_x4_in_q_squared (x : ℝ) : 
  let q : ℝ → ℝ := λ x => x^5 - 4*x^2 + 3
  (q x)^2 = x^10 - 8*x^7 + 16*x^4 + 6*x^5 - 24*x^2 + 9 := by
  sorry


end NUMINAMATH_CALUDE_coefficient_x4_in_q_squared_l2048_204837


namespace NUMINAMATH_CALUDE_acute_angle_inequalities_l2048_204808

theorem acute_angle_inequalities (α β : Real) 
  (h_α : 0 < α ∧ α < Real.pi / 2) 
  (h_β : 0 < β ∧ β < Real.pi / 2) : 
  (Real.sin (α + β) < Real.cos α + Real.cos β) ∧ 
  (Real.sin (α - β) < Real.cos α + Real.cos β) := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_inequalities_l2048_204808


namespace NUMINAMATH_CALUDE_triangular_fence_perimeter_l2048_204865

/-- Calculates the perimeter of a triangular fence with evenly spaced posts -/
theorem triangular_fence_perimeter
  (num_posts : ℕ)
  (post_width : ℝ)
  (post_spacing : ℝ)
  (h_num_posts : num_posts = 18)
  (h_post_width : post_width = 0.5)
  (h_post_spacing : post_spacing = 4)
  (h_divisible : num_posts % 3 = 0) :
  let posts_per_side := num_posts / 3
  let side_length := (posts_per_side - 1) * post_spacing + posts_per_side * post_width
  3 * side_length = 69 := by sorry

end NUMINAMATH_CALUDE_triangular_fence_perimeter_l2048_204865


namespace NUMINAMATH_CALUDE_roger_spent_calculation_l2048_204844

/-- Calculates the amount of money Roger spent given his initial amount,
    the amount received from his mom, and his current amount. -/
def money_spent (initial : ℕ) (received : ℕ) (current : ℕ) : ℕ :=
  initial + received - current

theorem roger_spent_calculation :
  money_spent 45 46 71 = 20 := by
  sorry

end NUMINAMATH_CALUDE_roger_spent_calculation_l2048_204844


namespace NUMINAMATH_CALUDE_large_square_area_l2048_204835

-- Define the squares
structure Square where
  side : ℕ

-- Define the problem setup
structure SquareProblem where
  small : Square
  medium : Square
  large : Square
  small_perimeter_lt_medium_side : 4 * small.side < medium.side
  exposed_area : (large.side ^ 2 - (small.side ^ 2 + medium.side ^ 2)) = 10

-- Theorem statement
theorem large_square_area (problem : SquareProblem) : problem.large.side ^ 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_large_square_area_l2048_204835


namespace NUMINAMATH_CALUDE_trailing_zeros_80_factorial_l2048_204870

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

theorem trailing_zeros_80_factorial :
  trailingZeros 73 = 16 → trailingZeros 80 = 18 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_80_factorial_l2048_204870


namespace NUMINAMATH_CALUDE_animath_interns_pigeonhole_l2048_204883

theorem animath_interns_pigeonhole (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x < n - 1) ∧ f i = k ∧ f j = k) :=
sorry

end NUMINAMATH_CALUDE_animath_interns_pigeonhole_l2048_204883


namespace NUMINAMATH_CALUDE_candy_problem_l2048_204809

/-- Represents a set of candies with three types: hard, chocolate, and gummy -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The total number of candies in a set -/
def total (s : CandySet) : ℕ := s.hard + s.chocolate + s.gummy

theorem candy_problem (s1 s2 s3 : CandySet) 
  (h1 : s1.hard + s2.hard + s3.hard = s1.chocolate + s2.chocolate + s3.chocolate)
  (h2 : s1.hard + s2.hard + s3.hard = s1.gummy + s2.gummy + s3.gummy)
  (h3 : s1.chocolate = s1.gummy)
  (h4 : s1.hard = s1.chocolate + 7)
  (h5 : s2.hard = s2.chocolate)
  (h6 : s2.gummy = s2.hard - 15)
  (h7 : s3.hard = 0) : 
  total s3 = 29 := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l2048_204809


namespace NUMINAMATH_CALUDE_final_plant_count_l2048_204849

def total_seeds : ℕ := 23
def marigold_seeds : ℕ := 10
def sunflower_seeds : ℕ := 8
def lavender_seeds : ℕ := 5
def seeds_not_grown : ℕ := 5

def marigold_growth_rate : ℚ := 2/5
def sunflower_growth_rate : ℚ := 3/5
def lavender_growth_rate : ℚ := 7/10

def squirrel_eat_rate : ℚ := 1/2
def rabbit_eat_rate : ℚ := 1/4

def pest_control_success_rate : ℚ := 3/4
def pest_control_reduction_rate : ℚ := 1/10

def weed_strangle_rate : ℚ := 1/3

def weeds_pulled : ℕ := 2
def weeds_kept : ℕ := 1

theorem final_plant_count :
  ∃ (grown_marigolds grown_sunflowers grown_lavenders : ℕ),
    grown_marigolds ≤ (marigold_seeds : ℚ) * marigold_growth_rate ∧
    grown_sunflowers ≤ (sunflower_seeds : ℚ) * sunflower_growth_rate ∧
    grown_lavenders ≤ (lavender_seeds : ℚ) * lavender_growth_rate ∧
    ∃ (eaten_marigolds eaten_sunflowers : ℕ),
      eaten_marigolds = ⌊(grown_marigolds : ℚ) * squirrel_eat_rate⌋ ∧
      eaten_sunflowers = ⌊(grown_sunflowers : ℚ) * rabbit_eat_rate⌋ ∧
      ∃ (protected_lavenders : ℕ),
        protected_lavenders ≤ ⌊(grown_lavenders : ℚ) * pest_control_success_rate⌋ ∧
        ∃ (final_marigolds final_sunflowers : ℕ),
          final_marigolds ≤ ⌊(grown_marigolds - eaten_marigolds : ℚ) * (1 - pest_control_reduction_rate)⌋ ∧
          final_sunflowers ≤ ⌊(grown_sunflowers - eaten_sunflowers : ℚ) * (1 - pest_control_reduction_rate)⌋ ∧
          ∃ (total_plants : ℕ),
            total_plants = final_marigolds + final_sunflowers + protected_lavenders ∧
            ∃ (strangled_plants : ℕ),
              strangled_plants = ⌊(total_plants : ℚ) * weed_strangle_rate⌋ ∧
              total_plants - strangled_plants + weeds_kept = 6 :=
by sorry

end NUMINAMATH_CALUDE_final_plant_count_l2048_204849


namespace NUMINAMATH_CALUDE_blue_pencil_length_l2048_204840

theorem blue_pencil_length (total : ℝ) (purple : ℝ) (black : ℝ) (blue : ℝ)
  (h_total : total = 4)
  (h_purple : purple = 1.5)
  (h_black : black = 0.5)
  (h_sum : total = purple + black + blue) :
  blue = 2 := by
sorry

end NUMINAMATH_CALUDE_blue_pencil_length_l2048_204840


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2048_204806

/-- Given that the solution set of ax^2 + bx + c ≤ 0 is {x | x ≤ -1/3 ∨ x ≥ 2},
    prove that the solution set of cx^2 + bx + a > 0 is {x | x < -3 ∨ x > 1/2} -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c ≤ 0 ↔ x ≤ -1/3 ∨ x ≥ 2) :
  ∀ x, c*x^2 + b*x + a > 0 ↔ x < -3 ∨ x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2048_204806


namespace NUMINAMATH_CALUDE_periodic_even_function_extension_l2048_204872

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem periodic_even_function_extension
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_defined : ∀ x ∈ Set.Icc 2 3, f x = -2 * (x - 3)^2 + 4) :
  ∀ x ∈ Set.Icc 0 2, f x = -2 * (x - 1)^2 + 4 :=
sorry

end NUMINAMATH_CALUDE_periodic_even_function_extension_l2048_204872


namespace NUMINAMATH_CALUDE_first_part_to_total_ratio_l2048_204804

theorem first_part_to_total_ratio (total : ℚ) (first_part : ℚ) : 
  total = 782 →
  first_part = 204 →
  ∃ (x : ℚ), (x + 2/3 + 3/4) * first_part = total →
  first_part / total = 102 / 391 := by
  sorry

end NUMINAMATH_CALUDE_first_part_to_total_ratio_l2048_204804


namespace NUMINAMATH_CALUDE_fraction_problem_l2048_204842

theorem fraction_problem (N : ℚ) (h : (1/4) * (1/3) * (2/5) * N = 30) :
  ∃ F : ℚ, F * N = 120 ∧ F = 2/15 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l2048_204842


namespace NUMINAMATH_CALUDE_record_breaking_time_l2048_204818

/-- The number of jumps in the record -/
def record : ℕ := 54000

/-- The number of jumps Mark can do per second -/
def jumps_per_second : ℕ := 3

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The time required to break the record in hours -/
def time_to_break_record : ℚ :=
  (record / jumps_per_second) / seconds_per_hour

theorem record_breaking_time :
  time_to_break_record = 5 := by sorry

end NUMINAMATH_CALUDE_record_breaking_time_l2048_204818


namespace NUMINAMATH_CALUDE_theta_max_ratio_l2048_204830

/-- Represents a participant's scores in the competition -/
structure Participant where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ
  day3_score : ℕ
  day3_total : ℕ

/-- The competition setup and conditions -/
def Competition (omega theta : Participant) : Prop :=
  omega.day1_score = 200 ∧
  omega.day1_total = 400 ∧
  omega.day2_score + omega.day3_score = 150 ∧
  omega.day2_total + omega.day3_total = 200 ∧
  omega.day1_total + omega.day2_total + omega.day3_total = 600 ∧
  theta.day1_total + theta.day2_total + theta.day3_total = 600 ∧
  theta.day1_score > 0 ∧ theta.day2_score > 0 ∧ theta.day3_score > 0 ∧
  (theta.day1_score : ℚ) / theta.day1_total < (omega.day1_score : ℚ) / omega.day1_total ∧
  (theta.day2_score : ℚ) / theta.day2_total < (omega.day2_score : ℚ) / omega.day2_total ∧
  (theta.day3_score : ℚ) / theta.day3_total < (omega.day3_score : ℚ) / omega.day3_total

/-- Theta's overall success ratio -/
def ThetaRatio (theta : Participant) : ℚ :=
  (theta.day1_score + theta.day2_score + theta.day3_score : ℚ) /
  (theta.day1_total + theta.day2_total + theta.day3_total)

/-- The main theorem stating Theta's maximum possible success ratio -/
theorem theta_max_ratio (omega theta : Participant) 
  (h : Competition omega theta) : ThetaRatio theta ≤ 56 / 75 := by
  sorry


end NUMINAMATH_CALUDE_theta_max_ratio_l2048_204830


namespace NUMINAMATH_CALUDE_dodecagon_enclosed_by_dodecagons_l2048_204833

/-- The number of sides of the inner regular polygon -/
def inner_sides : ℕ := 12

/-- The number of outer regular polygons -/
def num_outer_polygons : ℕ := 12

/-- The number of sides of each outer regular polygon -/
def outer_sides : ℕ := 12

/-- The interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ :=
  (n - 2 : ℚ) * 180 / n

/-- The exterior angle of a regular polygon with n sides -/
def exterior_angle (n : ℕ) : ℚ :=
  360 / n

/-- Theorem stating that a regular 12-sided polygon can be exactly enclosed
    by 12 regular 12-sided polygons -/
theorem dodecagon_enclosed_by_dodecagons :
  2 * (exterior_angle outer_sides / 2) = 
  180 - interior_angle inner_sides :=
sorry

end NUMINAMATH_CALUDE_dodecagon_enclosed_by_dodecagons_l2048_204833


namespace NUMINAMATH_CALUDE_probability_arrives_before_l2048_204862

/-- Represents a student -/
structure Student :=
  (name : String)

/-- Represents the arrival order of students -/
def ArrivalOrder := List Student

/-- Given a list of students, generates all possible arrival orders -/
def allPossibleArrivals (students : List Student) : List ArrivalOrder :=
  sorry

/-- Checks if student1 arrives before student2 in a given arrival order -/
def arrivesBeforeIn (student1 student2 : Student) (order : ArrivalOrder) : Bool :=
  sorry

/-- Counts the number of arrival orders where student1 arrives before student2 -/
def countArrivesBeforeOrders (student1 student2 : Student) (orders : List ArrivalOrder) : Nat :=
  sorry

theorem probability_arrives_before (student1 student2 student3 : Student) :
  let students := [student1, student2, student3]
  let allOrders := allPossibleArrivals students
  let favorableOrders := countArrivesBeforeOrders student1 student2 allOrders
  favorableOrders = allOrders.length / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_arrives_before_l2048_204862


namespace NUMINAMATH_CALUDE_sum_of_digits_of_special_number_l2048_204826

/-- A number consisting of n digits all equal to 1 -/
def allOnes (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sumOfDigits (n / 10)

/-- The main theorem -/
theorem sum_of_digits_of_special_number :
  let L := allOnes 2022
  sumOfDigits (9 * L^2 + 2 * L) = 4044 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_special_number_l2048_204826


namespace NUMINAMATH_CALUDE_square_triangle_perimeter_equality_l2048_204882

theorem square_triangle_perimeter_equality (x : ℝ) :
  x = 4 →
  4 * (x + 2) = 3 * (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_perimeter_equality_l2048_204882


namespace NUMINAMATH_CALUDE_stick_cutting_l2048_204887

theorem stick_cutting (short_length long_length : ℝ) : 
  short_length > 0 →
  long_length = short_length + 12 →
  short_length + long_length = 20 →
  (long_length / short_length : ℝ) = 4 := by
sorry

end NUMINAMATH_CALUDE_stick_cutting_l2048_204887


namespace NUMINAMATH_CALUDE_polygon_with_135_degree_angles_is_octagon_l2048_204851

theorem polygon_with_135_degree_angles_is_octagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 135 →
    (n - 2) * 180 / n = interior_angle →
    n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_135_degree_angles_is_octagon_l2048_204851


namespace NUMINAMATH_CALUDE_initial_fliers_count_l2048_204889

theorem initial_fliers_count (morning_fraction : ℚ) (afternoon_fraction : ℚ) (remaining_fliers : ℕ) : 
  morning_fraction = 1/5 →
  afternoon_fraction = 1/4 →
  remaining_fliers = 1500 →
  ∃ initial_fliers : ℕ, 
    initial_fliers = 2500 ∧
    (1 - morning_fraction) * (1 - afternoon_fraction) * initial_fliers = remaining_fliers :=
by
  sorry

end NUMINAMATH_CALUDE_initial_fliers_count_l2048_204889


namespace NUMINAMATH_CALUDE_six_people_arrangement_l2048_204834

theorem six_people_arrangement (n : ℕ) (h : n = 6) : 
  n.factorial - 2 * (n-1).factorial - 2 * 2 * (n-1).factorial + 2 * 2 * (n-2).factorial = 96 :=
by sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l2048_204834


namespace NUMINAMATH_CALUDE_equation_infinite_solutions_l2048_204858

theorem equation_infinite_solutions (a b : ℝ) : 
  b = 1 → 
  (∀ x : ℝ, a * (3 * x - 2) + b * (2 * x - 3) = 8 * x - 7) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_infinite_solutions_l2048_204858


namespace NUMINAMATH_CALUDE_toys_production_time_l2048_204896

theorem toys_production_time (goal : ℕ) (rate : ℕ) (days_worked : ℕ) (days_left : ℕ) : 
  goal = 1000 → 
  rate = 100 → 
  days_worked = 6 → 
  rate * days_worked + rate * days_left = goal → 
  days_left = 4 := by
sorry

end NUMINAMATH_CALUDE_toys_production_time_l2048_204896


namespace NUMINAMATH_CALUDE_banana_pies_count_l2048_204868

def total_pies : ℕ := 30
def ratio_sum : ℕ := 2 + 5 + 3

theorem banana_pies_count :
  let banana_ratio : ℕ := 3
  (banana_ratio * total_pies) / ratio_sum = 9 :=
by sorry

end NUMINAMATH_CALUDE_banana_pies_count_l2048_204868


namespace NUMINAMATH_CALUDE_sum_simplification_l2048_204801

theorem sum_simplification (n : ℕ) : 
  (Finset.range n).sum (λ i => (n - i) * 2^i) = 2^n + 1 - n - 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l2048_204801


namespace NUMINAMATH_CALUDE_sam_total_spending_l2048_204867

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1 / 10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- Calculate the total value of coins -/
def coin_value (pennies nickels dimes quarters : ℕ) : ℚ :=
  (pennies : ℚ) * penny_value + (nickels : ℚ) * nickel_value +
  (dimes : ℚ) * dime_value + (quarters : ℚ) * quarter_value

/-- Sam's spending for each day of the week -/
def monday_spending : ℚ := coin_value 5 3 0 0
def tuesday_spending : ℚ := coin_value 0 0 8 4
def wednesday_spending : ℚ := coin_value 0 7 10 2
def thursday_spending : ℚ := coin_value 20 15 12 6
def friday_spending : ℚ := coin_value 45 20 25 10

/-- The total amount Sam spent during the week -/
def total_spending : ℚ :=
  monday_spending + tuesday_spending + wednesday_spending + thursday_spending + friday_spending

/-- Theorem: Sam spent $14.05 in total during the week -/
theorem sam_total_spending : total_spending = 1405 / 100 := by
  sorry


end NUMINAMATH_CALUDE_sam_total_spending_l2048_204867


namespace NUMINAMATH_CALUDE_rowing_speed_contradiction_l2048_204861

theorem rowing_speed_contradiction (man_rate : ℝ) (with_stream : ℝ) (against_stream : ℝ) :
  man_rate = 6 →
  with_stream = 20 →
  with_stream = man_rate + (with_stream - man_rate) →
  against_stream = man_rate - (with_stream - man_rate) →
  against_stream < 0 :=
by sorry

#check rowing_speed_contradiction

end NUMINAMATH_CALUDE_rowing_speed_contradiction_l2048_204861


namespace NUMINAMATH_CALUDE_price_change_theorem_l2048_204853

theorem price_change_theorem (p : ℝ) : 
  (1 + p / 100) * (1 - p / 200) = 1 + p / 300 → p = 100 / 3 :=
by sorry

end NUMINAMATH_CALUDE_price_change_theorem_l2048_204853


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_x_plus_2y_equals_5_l2048_204822

theorem positive_integer_solutions_of_x_plus_2y_equals_5 :
  {(x, y) : ℕ × ℕ | x + 2 * y = 5 ∧ x > 0 ∧ y > 0} = {(1, 2), (3, 1)} := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_x_plus_2y_equals_5_l2048_204822


namespace NUMINAMATH_CALUDE_derivative_sin_cos_plus_one_l2048_204886

theorem derivative_sin_cos_plus_one (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin x * (Real.cos x + 1)
  (deriv f) x = Real.cos (2 * x) + Real.cos x := by
sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_plus_one_l2048_204886


namespace NUMINAMATH_CALUDE_min_value_expression_l2048_204832

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 12) / Real.sqrt (x - 4) ≥ 8 ∧ ∃ y : ℝ, y > 4 ∧ (y + 12) / Real.sqrt (y - 4) = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2048_204832


namespace NUMINAMATH_CALUDE_girls_percentage_in_class_l2048_204831

theorem girls_percentage_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (girls_ratio : ℚ) / (boys_ratio + girls_ratio) * total_students / total_students * 100 = 57.14 := by
  sorry

end NUMINAMATH_CALUDE_girls_percentage_in_class_l2048_204831


namespace NUMINAMATH_CALUDE_log_equation_holds_l2048_204875

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) + Real.log 7 / Real.log 10 = Real.log 7 / Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_holds_l2048_204875


namespace NUMINAMATH_CALUDE_person_age_l2048_204827

/-- The age of a person satisfying a specific equation is 32 years old. -/
theorem person_age : ∃ (age : ℕ), 4 * (age + 4) - 4 * (age - 4) = age ∧ age = 32 := by
  sorry

end NUMINAMATH_CALUDE_person_age_l2048_204827


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l2048_204859

theorem divisibility_of_sum_of_powers : 
  let n : ℕ := 3^105 + 4^105
  ∃ (a b c d : ℕ), n = 13 * a ∧ n = 49 * b ∧ n = 181 * c ∧ n = 379 * d ∧
  ¬(∃ (e : ℕ), n = 5 * e) ∧ ¬(∃ (f : ℕ), n = 11 * f) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l2048_204859


namespace NUMINAMATH_CALUDE_sine_cosine_equation_l2048_204823

theorem sine_cosine_equation (x y : ℝ) 
  (h : (Real.sin x ^ 2 - Real.cos x ^ 2 + Real.cos x ^ 2 * Real.cos y ^ 2 - Real.sin x ^ 2 * Real.sin y ^ 2) / Real.sin (x + y) = 1) :
  ∃ k : ℤ, x - y = 2 * k * Real.pi + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_equation_l2048_204823


namespace NUMINAMATH_CALUDE_common_chord_length_l2048_204839

theorem common_chord_length (r : ℝ) (d : ℝ) (h1 : r = 12) (h2 : d = 8) :
  let chord_length := 2 * Real.sqrt (r^2 - (d/2)^2)
  chord_length = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_l2048_204839


namespace NUMINAMATH_CALUDE_johns_weight_l2048_204829

/-- Given that Roy weighs 4 pounds and John is 77 pounds heavier than Roy,
    prove that John weighs 81 pounds. -/
theorem johns_weight (roy_weight : ℕ) (weight_difference : ℕ) :
  roy_weight = 4 →
  weight_difference = 77 →
  roy_weight + weight_difference = 81 :=
by sorry

end NUMINAMATH_CALUDE_johns_weight_l2048_204829


namespace NUMINAMATH_CALUDE_inverse_g_sum_l2048_204894

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x^2

theorem inverse_g_sum : ∃ (f : ℝ → ℝ), Function.LeftInverse f g ∧ Function.RightInverse f g ∧ f (-4) + f 0 + f 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l2048_204894


namespace NUMINAMATH_CALUDE_cost_of_sneakers_l2048_204841

/-- Given the costs of items John bought, prove the cost of sneakers. -/
theorem cost_of_sneakers
  (total_cost : ℕ)
  (racket_cost : ℕ)
  (outfit_cost : ℕ)
  (h1 : total_cost = 750)
  (h2 : racket_cost = 300)
  (h3 : outfit_cost = 250) :
  total_cost - racket_cost - outfit_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_sneakers_l2048_204841


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l2048_204845

theorem stratified_sampling_size (total_population : ℕ) (stratum_size : ℕ) (stratum_sample : ℕ) (h1 : total_population = 3600) (h2 : stratum_size = 1000) (h3 : stratum_sample = 25) : 
  (stratum_size : ℚ) / total_population * (total_sample : ℚ) = stratum_sample → total_sample = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l2048_204845


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_l2048_204864

-- Define the parabola
def parabola (x m : ℝ) : ℝ := x^2 + 2*x + m - 1

-- Theorem statement
theorem parabola_intersects_x_axis (m : ℝ) :
  (∃ x : ℝ, parabola x m = 0) ↔ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_l2048_204864
