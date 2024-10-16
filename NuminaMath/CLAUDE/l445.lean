import Mathlib

namespace NUMINAMATH_CALUDE_salary_restoration_l445_44527

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) : 
  let reduced_salary := original_salary * (1 - 0.2)
  reduced_salary * (1 + 0.25) = original_salary := by
sorry

end NUMINAMATH_CALUDE_salary_restoration_l445_44527


namespace NUMINAMATH_CALUDE_cos_range_from_inequality_l445_44552

theorem cos_range_from_inequality (α : ℝ) :
  12 * (Real.sin α)^2 + Real.cos α > 11 →
  -1/4 < Real.cos (-α) ∧ Real.cos (-α) < 1/3 :=
by sorry

end NUMINAMATH_CALUDE_cos_range_from_inequality_l445_44552


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l445_44506

theorem unique_positive_integer_solution : 
  ∃! (x : ℕ), x > 0 ∧ 12 * x = x^2 + 36 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l445_44506


namespace NUMINAMATH_CALUDE_sin_cos_105_degrees_l445_44581

theorem sin_cos_105_degrees : Real.sin (105 * π / 180) * Real.cos (105 * π / 180) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_105_degrees_l445_44581


namespace NUMINAMATH_CALUDE_function_inequality_l445_44539

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : ∀ x, 3 * f x - f' x > 0) :
  f 1 < Real.exp 3 * f 0 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l445_44539


namespace NUMINAMATH_CALUDE_existence_of_square_with_no_visible_points_l445_44534

/-- A point is visible from the origin if the greatest common divisor of its coordinates is 1 -/
def visible_from_origin (x y : ℤ) : Prop := Int.gcd x y = 1

/-- A point (x, y) is inside a square with bottom-left corner (a, b) and side length n if
    a < x < a + n and b < y < b + n -/
def inside_square (x y a b n : ℤ) : Prop :=
  a < x ∧ x < a + n ∧ b < y ∧ y < b + n

theorem existence_of_square_with_no_visible_points :
  ∀ n : ℕ, n > 0 → ∃ a b : ℤ,
    ∀ x y : ℤ, inside_square x y a b n → ¬(visible_from_origin x y) :=
sorry

end NUMINAMATH_CALUDE_existence_of_square_with_no_visible_points_l445_44534


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l445_44523

theorem nested_fraction_evaluation :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l445_44523


namespace NUMINAMATH_CALUDE_largest_reciprocal_l445_44518

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 5/6 → b = 1/2 → c = 3 → d = 8/3 → e = 240 →
  (1/b ≥ 1/a) ∧ (1/b ≥ 1/c) ∧ (1/b ≥ 1/d) ∧ (1/b ≥ 1/e) :=
by sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l445_44518


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l445_44568

theorem divisibility_by_eleven (n : ℕ) (a b c d e : ℕ) 
  (h1 : n = a * 10000 + b * 1000 + c * 100 + d * 10 + e)
  (h2 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) : 
  n ≡ (a + c + e) - (b + d) [MOD 11] := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l445_44568


namespace NUMINAMATH_CALUDE_smallest_y_for_inequality_l445_44505

theorem smallest_y_for_inequality : ∃ y : ℕ, (∀ z : ℕ, 27^z > 3^24 → y ≤ z) ∧ 27^y > 3^24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_inequality_l445_44505


namespace NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l445_44544

/-- Given a complex number z that corresponds to a point in the fourth quadrant,
    prove that the real parameter a in z = (a + 2i³) / (2 - i) is in the range (-1, 4) -/
theorem range_of_a_in_fourth_quadrant (a : ℝ) :
  let z : ℂ := (a + 2 * Complex.I ^ 3) / (2 - Complex.I)
  (z.re > 0 ∧ z.im < 0) → -1 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l445_44544


namespace NUMINAMATH_CALUDE_room_length_proof_l445_44593

theorem room_length_proof (width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) :
  width = 3.75 →
  total_cost = 28875 →
  cost_per_sqm = 1400 →
  (total_cost / cost_per_sqm) / width = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l445_44593


namespace NUMINAMATH_CALUDE_fish_per_bowl_l445_44567

theorem fish_per_bowl (total_bowls : ℕ) (total_fish : ℕ) (h1 : total_bowls = 261) (h2 : total_fish = 6003) :
  total_fish / total_bowls = 23 :=
by sorry

end NUMINAMATH_CALUDE_fish_per_bowl_l445_44567


namespace NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l445_44510

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l445_44510


namespace NUMINAMATH_CALUDE_point_alignment_implies_m_value_l445_44554

/-- Three points lie on the same straight line if and only if 
    the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) = (y₃ - y₁) / (x₃ - x₁)

theorem point_alignment_implies_m_value :
  ∀ m : ℝ, collinear 1 (-2) 3 4 6 (m/3) → m = 39 := by
  sorry


end NUMINAMATH_CALUDE_point_alignment_implies_m_value_l445_44554


namespace NUMINAMATH_CALUDE_more_solutions_without_plus_one_l445_44572

/-- The upper bound for x, y, z, and t -/
def upperBound : ℕ := 10^6

/-- The number of integral solutions for x^2 - y^2 = z^3 - t^3 -/
def N : ℕ := sorry

/-- The number of integral solutions for x^2 - y^2 = z^3 - t^3 + 1 -/
def M : ℕ := sorry

/-- Theorem stating that N > M -/
theorem more_solutions_without_plus_one : N > M := by
  sorry

end NUMINAMATH_CALUDE_more_solutions_without_plus_one_l445_44572


namespace NUMINAMATH_CALUDE_waiter_customers_l445_44590

theorem waiter_customers (initial : ℕ) (left : ℕ) (new : ℕ) : 
  initial = 14 → left = 3 → new = 39 → initial - left + new = 50 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l445_44590


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l445_44519

/-- Given a geometric sequence {a_n} where 3a_1, (1/2)a_5, and 2a_3 form an arithmetic sequence,
    prove that (a_9 + a_10) / (a_7 + a_8) = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = q * a n) 
    (h2 : (1/2) * a 5 = (3 * a 1 + 2 * a 3) / 2) :
  (a 9 + a 10) / (a 7 + a 8) = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l445_44519


namespace NUMINAMATH_CALUDE_course_passing_logic_l445_44565

variable (Student : Type)
variable (answered_correctly : Student → Prop)
variable (passed_course : Student → Prop)

theorem course_passing_logic :
  (∀ s : Student, answered_correctly s → passed_course s) →
  (∀ s : Student, ¬passed_course s → ¬answered_correctly s) :=
by sorry

end NUMINAMATH_CALUDE_course_passing_logic_l445_44565


namespace NUMINAMATH_CALUDE_geometric_series_sum_is_four_thirds_l445_44559

/-- The sum of the infinite geometric series with first term 1 and common ratio 1/4 -/
def geometric_series_sum : ℚ := 4/3

/-- The first term of the geometric series -/
def a : ℚ := 1

/-- The common ratio of the geometric series -/
def r : ℚ := 1/4

/-- Theorem stating that the sum of the infinite geometric series
    1 + (1/4) + (1/4)² + (1/4)³ + ... is equal to 4/3 -/
theorem geometric_series_sum_is_four_thirds :
  geometric_series_sum = (a / (1 - r)) := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_is_four_thirds_l445_44559


namespace NUMINAMATH_CALUDE_earnings_proof_l445_44553

/-- Calculates the total earnings for a worker given their hourly wage, widget bonus, number of widgets produced, and work hours per week. -/
def total_earnings (hourly_wage : ℚ) (widget_bonus : ℚ) (widgets_produced : ℕ) (work_hours : ℕ) : ℚ :=
  hourly_wage * work_hours + widget_bonus * widgets_produced

/-- Proves that given the specified conditions, the total earnings are $700. -/
theorem earnings_proof :
  let hourly_wage : ℚ := 25/2
  let widget_bonus : ℚ := 4/25
  let widgets_produced : ℕ := 1250
  let work_hours : ℕ := 40
  total_earnings hourly_wage widget_bonus widgets_produced work_hours = 700 := by
sorry


end NUMINAMATH_CALUDE_earnings_proof_l445_44553


namespace NUMINAMATH_CALUDE_negation_of_conditional_l445_44561

theorem negation_of_conditional (x y : ℝ) :
  ¬(((x - 1) * (y + 2) = 0) → (x = 1 ∨ y = -2)) ↔
  (((x - 1) * (y + 2) ≠ 0) → (x ≠ 1 ∧ y ≠ -2)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l445_44561


namespace NUMINAMATH_CALUDE_find_c_l445_44575

theorem find_c (a b c d e : ℝ) : 
  (a + b + c) / 3 = 16 →
  (c + d + e) / 3 = 26 →
  (a + b + c + d + e) / 5 = 20 →
  c = 26 := by
sorry

end NUMINAMATH_CALUDE_find_c_l445_44575


namespace NUMINAMATH_CALUDE_friendly_point_properties_l445_44501

def is_friendly_point (x y : ℝ) : Prop :=
  ∃ m n : ℝ, m - n = 6 ∧ m - 1 = x ∧ 3*n + 1 = y

theorem friendly_point_properties :
  (¬ is_friendly_point 7 1) ∧ 
  (is_friendly_point 6 4) ∧
  (∀ x y t : ℝ, x + y = 2 → 2*x - y = t → is_friendly_point x y → t = 10) := by
  sorry

end NUMINAMATH_CALUDE_friendly_point_properties_l445_44501


namespace NUMINAMATH_CALUDE_expression_simplification_l445_44531

theorem expression_simplification (x : ℝ) (h : x^2 + 2*x - 6 = 0) :
  ((x - 1) / (x - 3) - (x + 1) / x) / ((x^2 + 3*x) / (x^2 - 6*x + 9)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l445_44531


namespace NUMINAMATH_CALUDE_xyz_max_value_l445_44589

theorem xyz_max_value (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≤ 3/125 := by
sorry

end NUMINAMATH_CALUDE_xyz_max_value_l445_44589


namespace NUMINAMATH_CALUDE_P_on_xoz_plane_l445_44515

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- The given point P -/
def P : Point3D := ⟨-2, 0, 3⟩

/-- Theorem: Point P lies on the xoz plane -/
theorem P_on_xoz_plane : P ∈ xoz_plane := by
  sorry


end NUMINAMATH_CALUDE_P_on_xoz_plane_l445_44515


namespace NUMINAMATH_CALUDE_power_four_2024_mod_11_l445_44538

theorem power_four_2024_mod_11 : 4^2024 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_four_2024_mod_11_l445_44538


namespace NUMINAMATH_CALUDE_curve_is_circle_l445_44550

/-- The curve represented by the equation |x-1| = √(1-(y+1)²) -/
def curve_equation (x y : ℝ) : Prop := |x - 1| = Real.sqrt (1 - (y + 1)^2)

/-- The equation of a circle with center (1, -1) and radius 1 -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

/-- Theorem stating that the curve equation represents a circle -/
theorem curve_is_circle :
  ∀ x y : ℝ, curve_equation x y ↔ circle_equation x y :=
by sorry

end NUMINAMATH_CALUDE_curve_is_circle_l445_44550


namespace NUMINAMATH_CALUDE_tangent_line_at_1_l445_44594

-- Define the function f
def f (x : ℝ) : ℝ := -(x^3) + x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 2*x

-- Theorem statement
theorem tangent_line_at_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -x + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_1_l445_44594


namespace NUMINAMATH_CALUDE_simplify_fraction_l445_44597

theorem simplify_fraction : (72 : ℚ) / 108 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l445_44597


namespace NUMINAMATH_CALUDE_correct_mark_calculation_l445_44511

theorem correct_mark_calculation (n : ℕ) (initial_avg final_avg wrong_mark : ℚ) :
  n = 30 →
  initial_avg = 60 →
  wrong_mark = 90 →
  final_avg = 57.5 →
  (n : ℚ) * initial_avg - wrong_mark + ((n : ℚ) * final_avg - (n : ℚ) * initial_avg + wrong_mark) = 15 :=
by sorry

end NUMINAMATH_CALUDE_correct_mark_calculation_l445_44511


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l445_44543

theorem logarithm_sum_simplification :
  let expr := (1 / (Real.log 3 / Real.log 12 + 1)) + 
              (1 / (Real.log 2 / Real.log 8 + 1)) + 
              (1 / (Real.log 3 / Real.log 9 + 1))
  expr = (5 * Real.log 2 + 2 * Real.log 3) / (4 * Real.log 2 + 3 * Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l445_44543


namespace NUMINAMATH_CALUDE_gcd_102_238_l445_44520

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l445_44520


namespace NUMINAMATH_CALUDE_six_cube_forming_configurations_l445_44579

/-- Represents a position where an additional square can be attached -/
inductive AttachmentPosition
| TopLeft | TopCenter | TopRight
| MiddleLeft | MiddleRight
| BottomLeft | BottomCenter | BottomRight
| LeftCenter | RightCenter

/-- Represents the cross-shaped arrangement of squares -/
structure CrossArrangement :=
  (center : Square)
  (top : Square)
  (right : Square)
  (bottom : Square)
  (left : Square)

/-- Represents a configuration with an additional square attached -/
structure Configuration :=
  (base : CrossArrangement)
  (attachment : AttachmentPosition)

/-- Predicate to check if a configuration can form a cube with one face missing -/
def can_form_cube (config : Configuration) : Prop :=
  sorry

/-- The main theorem stating that exactly 6 configurations can form a cube -/
theorem six_cube_forming_configurations :
  ∃ (valid_configs : Finset Configuration),
    (∀ c ∈ valid_configs, can_form_cube c) ∧
    (∀ c : Configuration, can_form_cube c → c ∈ valid_configs) ∧
    valid_configs.card = 6 :=
  sorry

end NUMINAMATH_CALUDE_six_cube_forming_configurations_l445_44579


namespace NUMINAMATH_CALUDE_square_root_problem_l445_44599

theorem square_root_problem (a b : ℝ) : 
  ((2 * a - 1)^2 = 4) → (b = 1) → (2 * a - b = 2 ∨ 2 * a - b = -2) := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l445_44599


namespace NUMINAMATH_CALUDE_complex_number_problem_l445_44578

/-- Given a complex number z = bi (b ∈ ℝ) such that (z-2)/(1+i) is real,
    prove that z = -2i and (m+z)^2 is in the first quadrant iff m < -2 -/
theorem complex_number_problem (b : ℝ) (z : ℂ) (h1 : z = Complex.I * b) 
    (h2 : ∃ (r : ℝ), (z - 2) / (1 + Complex.I) = r) :
  z = -2 * Complex.I ∧ 
  ∀ m : ℝ, (Complex.re ((m + z)^2) > 0 ∧ Complex.im ((m + z)^2) > 0) ↔ m < -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l445_44578


namespace NUMINAMATH_CALUDE_ellipse_m_range_l445_44571

theorem ellipse_m_range (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (2 + m)) - (y^2 / (m + 1)) = 1 ∧ 
   ((2 + m > 0 ∧ -(m + 1) > 0) ∨ (-(m + 1) > 0 ∧ 2 + m > 0))) ↔ 
  (m ∈ Set.Ioo (-2 : ℝ) (-3/2) ∪ Set.Ioo (-3/2 : ℝ) (-1)) := by
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l445_44571


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l445_44545

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.000136 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.36 ∧ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l445_44545


namespace NUMINAMATH_CALUDE_investment_problem_l445_44563

/-- The investment problem -/
theorem investment_problem (a b total_profit a_profit : ℕ)
  (h1 : a = 6300)
  (h2 : b = 4200)
  (h3 : total_profit = 12600)
  (h4 : a_profit = 3780)
  (h5 : ∀ x : ℕ, a / (a + b + x) = a_profit / total_profit) :
  ∃ c : ℕ, c = 10500 ∧ a / (a + b + c) = a_profit / total_profit :=
sorry

end NUMINAMATH_CALUDE_investment_problem_l445_44563


namespace NUMINAMATH_CALUDE_equation_has_real_root_l445_44560

-- Define the polynomial function
def f (K x : ℝ) : ℝ := K^2 * (x - 1) * (x - 2) * (x - 3) - x

-- Theorem statement
theorem equation_has_real_root :
  ∀ K : ℝ, ∃ x : ℝ, f K x = 0 :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l445_44560


namespace NUMINAMATH_CALUDE_prime_cube_equation_solutions_l445_44573

theorem prime_cube_equation_solutions :
  ∀ m n p : ℕ+,
    Nat.Prime p.val →
    (m.val^3 + n.val) * (n.val^3 + m.val) = p.val^3 →
    ((m = 2 ∧ n = 1 ∧ p = 3) ∨ (m = 1 ∧ n = 2 ∧ p = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_cube_equation_solutions_l445_44573


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l445_44587

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 48 → 
  Even heart → 
  heart + club ≤ 26 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l445_44587


namespace NUMINAMATH_CALUDE_square_mirror_side_length_l445_44517

theorem square_mirror_side_length 
  (wall_width : ℝ) 
  (wall_length : ℝ) 
  (mirror_area_ratio : ℝ) :
  wall_width = 42 →
  wall_length = 27.428571428571427 →
  mirror_area_ratio = 1 / 2 →
  ∃ (mirror_side : ℝ), 
    mirror_side = 24 ∧ 
    mirror_side^2 = mirror_area_ratio * wall_width * wall_length :=
by sorry

end NUMINAMATH_CALUDE_square_mirror_side_length_l445_44517


namespace NUMINAMATH_CALUDE_order_of_constants_l445_44574

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def b : ℝ := Real.exp (1/2)
noncomputable def c : ℝ := Real.log 2 / Real.log 10

-- Theorem statement
theorem order_of_constants : a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_order_of_constants_l445_44574


namespace NUMINAMATH_CALUDE_sin_585_degrees_l445_44512

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l445_44512


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l445_44596

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 2| = |x - 4| + |x - 6| + |x - 8| :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l445_44596


namespace NUMINAMATH_CALUDE_smallest_prime_for_divisibility_l445_44509

theorem smallest_prime_for_divisibility : ∃ (p : ℕ), 
  Nat.Prime p ∧ 
  (11002 + p) % 11 = 0 ∧ 
  (11002 + p) % 7 = 0 ∧
  ∀ (q : ℕ), Nat.Prime q → (11002 + q) % 11 = 0 → (11002 + q) % 7 = 0 → p ≤ q :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_prime_for_divisibility_l445_44509


namespace NUMINAMATH_CALUDE_subtraction_from_percentage_l445_44591

theorem subtraction_from_percentage (n : ℝ) : n = 70 → (n * 0.5 - 10 = 25) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_from_percentage_l445_44591


namespace NUMINAMATH_CALUDE_find_number_l445_44516

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 5) = 105 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l445_44516


namespace NUMINAMATH_CALUDE_concentric_circles_radius_l445_44558

theorem concentric_circles_radius (r₁ r₂ AB : ℝ) : 
  r₁ > 0 → r₂ > 0 →
  r₂ / r₁ = 7 / 3 →
  AB = 20 →
  ∃ (AC BC : ℝ), 
    AC = 2 * r₂ ∧
    BC^2 + AB^2 = AC^2 ∧
    BC^2 = r₂^2 - r₁^2 →
  r₂ = 70 / 3 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radius_l445_44558


namespace NUMINAMATH_CALUDE_solve_equation_l445_44502

theorem solve_equation (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 7 → y = 29 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l445_44502


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l445_44522

theorem regular_polygon_diagonals (n : ℕ) (h1 : n > 2) (h2 : (n - 2) * 180 / n = 120) :
  n - 3 = 3 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l445_44522


namespace NUMINAMATH_CALUDE_positive_sum_one_inequality_l445_44549

theorem positive_sum_one_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_one_inequality_l445_44549


namespace NUMINAMATH_CALUDE_sarahs_deleted_folder_size_l445_44562

theorem sarahs_deleted_folder_size 
  (initial_free : ℝ) 
  (initial_used : ℝ) 
  (new_files_size : ℝ) 
  (new_drive_size : ℝ) 
  (new_drive_free : ℝ)
  (h1 : initial_free = 2.4)
  (h2 : initial_used = 12.6)
  (h3 : new_files_size = 2)
  (h4 : new_drive_size = 20)
  (h5 : new_drive_free = 10) : 
  ∃ (deleted_folder_size : ℝ), 
    deleted_folder_size = 4.6 ∧ 
    initial_used - deleted_folder_size + new_files_size = new_drive_size - new_drive_free :=
by sorry

end NUMINAMATH_CALUDE_sarahs_deleted_folder_size_l445_44562


namespace NUMINAMATH_CALUDE_car_dealership_problem_l445_44532

theorem car_dealership_problem (initial_cars : ℕ) (initial_silver_percent : ℚ)
  (new_cars : ℕ) (total_silver_percent : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percent = 15 / 100)
  (h3 : new_cars = 80)
  (h4 : total_silver_percent = 25 / 100) :
  (new_cars - (total_silver_percent * (initial_cars + new_cars) - initial_silver_percent * initial_cars)) / new_cars = 70 / 100 := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_problem_l445_44532


namespace NUMINAMATH_CALUDE_cos_rational_angle_irrational_l445_44580

open Real

theorem cos_rational_angle_irrational (p q : ℤ) (h : q ≠ 0) :
  let x := cos (p / q * π)
  x ≠ 0 ∧ x ≠ 1/2 ∧ x ≠ -1/2 ∧ x ≠ 1 ∧ x ≠ -1 → Irrational x :=
by sorry

end NUMINAMATH_CALUDE_cos_rational_angle_irrational_l445_44580


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l445_44541

theorem fraction_zero_implies_x_equals_two (x : ℝ) :
  (x^2 - x - 2) / (x + 1) = 0 → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l445_44541


namespace NUMINAMATH_CALUDE_trig_simplification_l445_44548

theorem trig_simplification (α : ℝ) :
  Real.cos (π / 3 + α) + Real.sin (π / 6 + α) = Real.cos α := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l445_44548


namespace NUMINAMATH_CALUDE_cloud_height_above_lake_l445_44588

/-- The height of a cloud above a lake surface, given observation conditions --/
theorem cloud_height_above_lake (h : ℝ) (elevation_angle depression_angle : ℝ) : 
  h = 10 → 
  elevation_angle = 30 * π / 180 →
  depression_angle = 45 * π / 180 →
  ∃ (cloud_height : ℝ), abs (cloud_height - 37.3) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_cloud_height_above_lake_l445_44588


namespace NUMINAMATH_CALUDE_percentage_problem_l445_44547

theorem percentage_problem (p : ℝ) (h1 : 0.25 * 680 = p * 1000 - 30) : p = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l445_44547


namespace NUMINAMATH_CALUDE_abs_min_value_min_value_at_two_unique_min_value_l445_44555

theorem abs_min_value (x : ℝ) : |x - 2| + 3 ≥ 3 := by sorry

theorem min_value_at_two : ∃ (x : ℝ), |x - 2| + 3 = 3 := by sorry

theorem unique_min_value (x : ℝ) : |x - 2| + 3 = 3 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_abs_min_value_min_value_at_two_unique_min_value_l445_44555


namespace NUMINAMATH_CALUDE_race_result_l445_44535

/-- Represents a runner in the race -/
structure Runner where
  position : ℝ
  speed : ℝ

/-- The race setup and result -/
theorem race_result 
  (race_length : ℝ) 
  (a b : Runner) 
  (h1 : race_length = 3000)
  (h2 : a.position = race_length - 500)
  (h3 : b.position = race_length - 600)
  (h4 : a.speed > 0)
  (h5 : b.speed > 0) :
  let time_to_finish_a := (race_length - a.position) / a.speed
  let b_final_position := b.position + b.speed * time_to_finish_a
  race_length - b_final_position = 120 := by
sorry

end NUMINAMATH_CALUDE_race_result_l445_44535


namespace NUMINAMATH_CALUDE_sum_x_y_is_12_l445_44536

/-- An equilateral triangle with side lengths x + 5, y + 11, and 14 -/
structure EquilateralTriangle (x y : ℝ) : Prop where
  side1 : x + 5 = 14
  side2 : y + 11 = 14
  side3 : (14 : ℝ) = 14

/-- The sum of x and y in the equilateral triangle is 12 -/
theorem sum_x_y_is_12 {x y : ℝ} (t : EquilateralTriangle x y) : x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_is_12_l445_44536


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_odd_integers_l445_44595

theorem greatest_sum_consecutive_odd_integers (n : ℕ) : 
  (n % 2 = 1) →  -- n is odd
  (n * (n + 2) < 500) →  -- product is less than 500
  (∀ m : ℕ, m % 2 = 1 → m * (m + 2) < 500 → m ≤ n) →  -- n is the greatest such odd number
  n + (n + 2) = 44 :=  -- the sum is 44
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_odd_integers_l445_44595


namespace NUMINAMATH_CALUDE_triangle_theorem_l445_44556

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.c * Real.sin (t.A - t.B) = t.b * Real.sin (t.C - t.A)) :
  (t.a ^ 2 = t.b * t.c → t.A = π / 3) ∧
  (t.a = 2 ∧ Real.cos t.A = 4 / 5 → t.a + t.b + t.c = 2 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l445_44556


namespace NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l445_44540

theorem arcsin_arccos_equation_solution (x : ℝ) :
  Real.arcsin (3 * x) + Real.arccos (2 * x) = π / 4 →
  x = 1 / Real.sqrt (11 - 2 * Real.sqrt 2) ∨
  x = -1 / Real.sqrt (11 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l445_44540


namespace NUMINAMATH_CALUDE_rectangular_solid_properties_l445_44537

theorem rectangular_solid_properties (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 6)
  (h2 : a * c = Real.sqrt 3)
  (h3 : b * c = Real.sqrt 2) :
  a * b * c = 6 ∧ Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_solid_properties_l445_44537


namespace NUMINAMATH_CALUDE_range_of_f_l445_44546

noncomputable def odot (a b : ℝ) : ℝ := if a ≤ b then a else b

noncomputable def f (x : ℝ) : ℝ := odot (2^x) (2^(-x))

theorem range_of_f :
  (∀ y, y ∈ Set.range f → 0 < y ∧ y ≤ 1) ∧
  (∀ y, 0 < y ∧ y ≤ 1 → ∃ x, f x = y) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l445_44546


namespace NUMINAMATH_CALUDE_magpie_call_not_correlation_l445_44521

-- Define a type for statements
inductive Statement
| HeavySnow : Statement
| GreatTeachers : Statement
| Smoking : Statement
| MagpieCall : Statement

-- Define a predicate for correlation
def IsCorrelation (s : Statement) : Prop :=
  match s with
  | Statement.HeavySnow => True
  | Statement.GreatTeachers => True
  | Statement.Smoking => True
  | Statement.MagpieCall => False

-- Theorem statement
theorem magpie_call_not_correlation :
  ∀ s : Statement, 
    (s = Statement.HeavySnow ∨ s = Statement.GreatTeachers ∨ s = Statement.Smoking → IsCorrelation s) ∧
    (s = Statement.MagpieCall → ¬IsCorrelation s) :=
by sorry

end NUMINAMATH_CALUDE_magpie_call_not_correlation_l445_44521


namespace NUMINAMATH_CALUDE_raisin_count_proof_l445_44557

/-- Given 5 boxes of raisins with a total of 437 raisins, where one box has 72 raisins,
    another has 74 raisins, and the remaining three boxes have an equal number of raisins,
    prove that each of these three boxes contains 97 raisins. -/
theorem raisin_count_proof (total_raisins : ℕ) (total_boxes : ℕ) 
  (box1_raisins : ℕ) (box2_raisins : ℕ) (other_boxes_raisins : ℕ) :
  total_raisins = 437 →
  total_boxes = 5 →
  box1_raisins = 72 →
  box2_raisins = 74 →
  total_raisins = box1_raisins + box2_raisins + 3 * other_boxes_raisins →
  other_boxes_raisins = 97 := by
  sorry

end NUMINAMATH_CALUDE_raisin_count_proof_l445_44557


namespace NUMINAMATH_CALUDE_special_sequence_2023_l445_44551

/-- A sequence of positive terms with a special property -/
structure SpecialSequence where
  a : ℕ → ℕ+
  S : ℕ → ℕ
  property : ∀ n, 2 * S n = (a n).val * ((a n).val + 1)

/-- The 2023rd term of a special sequence is 2023 -/
theorem special_sequence_2023 (seq : SpecialSequence) : seq.a 2023 = ⟨2023, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_2023_l445_44551


namespace NUMINAMATH_CALUDE_license_plate_count_l445_44508

/-- The number of vowels available for the license plate. -/
def num_vowels : ℕ := 6

/-- The number of consonants available for the license plate. -/
def num_consonants : ℕ := 20

/-- The number of digits available for the license plate. -/
def num_digits : ℕ := 10

/-- The number of special characters available for the license plate. -/
def num_special_chars : ℕ := 2

/-- The total number of possible license plates. -/
def total_license_plates : ℕ := num_vowels * num_consonants * num_digits * num_consonants * num_special_chars

theorem license_plate_count : total_license_plates = 48000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l445_44508


namespace NUMINAMATH_CALUDE_inverse_mod_53_l445_44514

theorem inverse_mod_53 (h : (15⁻¹ : ZMod 53) = 31) : (38⁻¹ : ZMod 53) = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l445_44514


namespace NUMINAMATH_CALUDE_two_visit_days_365_l445_44513

def alice_visits (d : ℕ) : Bool := d % 4 = 0
def bianca_visits (d : ℕ) : Bool := d % 6 = 0
def carmen_visits (d : ℕ) : Bool := d % 8 = 0

def exactly_two_visit (d : ℕ) : Bool :=
  let visit_count := (alice_visits d).toNat + (bianca_visits d).toNat + (carmen_visits d).toNat
  visit_count = 2

def count_two_visit_days (n : ℕ) : ℕ :=
  (List.range n).filter exactly_two_visit |>.length

theorem two_visit_days_365 :
  count_two_visit_days 365 = 45 := by
  sorry

end NUMINAMATH_CALUDE_two_visit_days_365_l445_44513


namespace NUMINAMATH_CALUDE_two_times_two_thousand_fifteen_minus_two_thousand_fifteen_l445_44592

theorem two_times_two_thousand_fifteen_minus_two_thousand_fifteen : 2 * 2015 - 2015 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_two_times_two_thousand_fifteen_minus_two_thousand_fifteen_l445_44592


namespace NUMINAMATH_CALUDE_league_games_l445_44500

theorem league_games (num_teams : ℕ) (total_games : ℕ) (games_per_matchup : ℕ) : 
  num_teams = 50 → 
  total_games = 4900 → 
  games_per_matchup * (num_teams - 1) * num_teams = 2 * total_games → 
  games_per_matchup = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_league_games_l445_44500


namespace NUMINAMATH_CALUDE_total_accidents_l445_44529

/-- Represents the number of vehicles involved in accidents per 100 million vehicles -/
def A (k : ℝ) (x : ℝ) : ℝ := 96 + k * x

/-- The constant k for morning hours -/
def k_morning : ℝ := 1

/-- The constant k for evening hours -/
def k_evening : ℝ := 3

/-- The number of vehicles (in billions) during morning hours -/
def x_morning : ℝ := 2

/-- The number of vehicles (in billions) during evening hours -/
def x_evening : ℝ := 1

/-- Theorem stating the total number of vehicles involved in accidents -/
theorem total_accidents : 
  A k_morning (100 * x_morning) + A k_evening (100 * x_evening) = 5192 := by
  sorry

end NUMINAMATH_CALUDE_total_accidents_l445_44529


namespace NUMINAMATH_CALUDE_divisible_by_seven_l445_44564

theorem divisible_by_seven (n : ℕ) : 7 ∣ (6^(2*n+1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l445_44564


namespace NUMINAMATH_CALUDE_cookie_jar_spending_l445_44542

theorem cookie_jar_spending (initial_amount : ℝ) (amount_left : ℝ) (doris_spent : ℝ) : 
  initial_amount = 21 →
  amount_left = 12 →
  initial_amount - (doris_spent + doris_spent / 2) = amount_left →
  doris_spent = 6 := by
sorry

end NUMINAMATH_CALUDE_cookie_jar_spending_l445_44542


namespace NUMINAMATH_CALUDE_printer_z_time_l445_44503

/-- Given printers X, Y, and Z with the following properties:
  - The ratio of time for X alone to Y and Z together is 2.25
  - X can do the job in 15 hours
  - Y can do the job in 10 hours
Prove that Z takes 20 hours to do the job alone. -/
theorem printer_z_time (tx ty tz : ℝ) : 
  tx = 15 → 
  ty = 10 → 
  tx = 2.25 * (1 / (1 / ty + 1 / tz)) → 
  tz = 20 := by
sorry

end NUMINAMATH_CALUDE_printer_z_time_l445_44503


namespace NUMINAMATH_CALUDE_quadratic_solution_l445_44569

theorem quadratic_solution (c : ℝ) : 
  (18^2 + 12*18 + c = 0) → 
  (∃ x : ℝ, x^2 + 12*x + c = 0 ∧ x ≠ 18) → 
  ((-30)^2 + 12*(-30) + c = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l445_44569


namespace NUMINAMATH_CALUDE_percentage_six_years_or_more_l445_44528

def employee_distribution (x : ℕ) : List ℕ :=
  [4*x, 7*x, 5*x, 4*x, 3*x, 3*x, 2*x, 2*x, 2*x, 2*x]

def total_employees (x : ℕ) : ℕ :=
  List.sum (employee_distribution x)

def employees_six_years_or_more (x : ℕ) : ℕ :=
  List.sum (List.drop 6 (employee_distribution x))

theorem percentage_six_years_or_more (x : ℕ) :
  (employees_six_years_or_more x : ℚ) / (total_employees x : ℚ) * 100 = 2222 / 100 :=
sorry

end NUMINAMATH_CALUDE_percentage_six_years_or_more_l445_44528


namespace NUMINAMATH_CALUDE_hotdog_distribution_l445_44525

theorem hotdog_distribution (E : ℚ) 
  (total_hotdogs : E + E + 2*E + 3*E = 14) : E = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotdog_distribution_l445_44525


namespace NUMINAMATH_CALUDE_min_value_at_three_l445_44582

/-- The quadratic function we're minimizing -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The statement that x = 3 minimizes the function f -/
theorem min_value_at_three :
  ∀ x : ℝ, f 3 ≤ f x :=
by
  sorry

#check min_value_at_three

end NUMINAMATH_CALUDE_min_value_at_three_l445_44582


namespace NUMINAMATH_CALUDE_correct_num_technicians_l445_44566

/-- The number of technicians in a workshop -/
def num_technicians : ℕ := 5

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 15

/-- The average salary of all workers -/
def avg_salary_all : ℕ := 700

/-- The average salary of technicians -/
def avg_salary_technicians : ℕ := 800

/-- The average salary of non-technicians -/
def avg_salary_others : ℕ := 650

/-- Theorem stating that the number of technicians is correct given the conditions -/
theorem correct_num_technicians :
  num_technicians = 5 ∧
  num_technicians ≤ total_workers ∧
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_others) / total_workers = avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_correct_num_technicians_l445_44566


namespace NUMINAMATH_CALUDE_equation_solution_l445_44576

theorem equation_solution (a : ℝ) : 
  (∀ x, 3*x + |a - 2| = -3 ↔ 3*x + 4 = 0) → 
  ((a - 2)^2010 - 2*a + 1 = -4 ∨ (a - 2)^2010 - 2*a + 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l445_44576


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l445_44586

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → m ≤ n :=
by
  use 62
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l445_44586


namespace NUMINAMATH_CALUDE_dispersion_measures_l445_44524

-- Define a sample as a list of real numbers
def Sample := List Real

-- Define the concept of a statistic as a function from a sample to a real number
def Statistic := Sample → Real

-- Define the concept of measuring dispersion
def MeasuresDispersion (s : Statistic) : Prop := sorry

-- Define standard deviation
def StandardDeviation : Statistic := sorry

-- Define median
def Median : Statistic := sorry

-- Define range
def Range : Statistic := sorry

-- Define mean
def Mean : Statistic := sorry

-- Theorem stating that only standard deviation and range measure dispersion
theorem dispersion_measures (sample : Sample) :
  MeasuresDispersion StandardDeviation ∧
  MeasuresDispersion Range ∧
  ¬MeasuresDispersion Median ∧
  ¬MeasuresDispersion Mean :=
sorry

end NUMINAMATH_CALUDE_dispersion_measures_l445_44524


namespace NUMINAMATH_CALUDE_cooler_contents_l445_44577

/-- The number of cherry sodas in the cooler -/
def cherry_sodas : ℕ := 8

/-- The number of orange pops in the cooler -/
def orange_pops : ℕ := 2 * cherry_sodas

/-- The total number of cans in the cooler -/
def total_cans : ℕ := 24

theorem cooler_contents : 
  cherry_sodas + orange_pops = total_cans ∧ cherry_sodas = 8 := by
  sorry

end NUMINAMATH_CALUDE_cooler_contents_l445_44577


namespace NUMINAMATH_CALUDE_factor_x6_minus_64_l445_44504

theorem factor_x6_minus_64 (x : ℝ) : 
  x^6 - 64 = (x - 2) * (x + 2) * (x^4 + 4*x^2 + 16) := by
  sorry

end NUMINAMATH_CALUDE_factor_x6_minus_64_l445_44504


namespace NUMINAMATH_CALUDE_range_of_m_l445_44570

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m :
  (∀ m : ℝ, ¬(p m ∧ q m)) →
  (∀ m : ℝ, q m) →
  ∀ m : ℝ, 1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l445_44570


namespace NUMINAMATH_CALUDE_a_is_four_l445_44585

def rounds_to_9430 (a b : ℕ) : Prop :=
  9000 + 100 * a + 30 + b ≥ 9425 ∧ 9000 + 100 * a + 30 + b < 9435

theorem a_is_four (a b : ℕ) (h : rounds_to_9430 a b) : a = 4 :=
sorry

end NUMINAMATH_CALUDE_a_is_four_l445_44585


namespace NUMINAMATH_CALUDE_min_board_sum_with_hundred_ones_l445_44598

/-- Represents the state of the board --/
structure BoardState where
  ones : ℕ
  tens : ℕ
  twentyFives : ℕ

/-- Defines the allowed operations on the board --/
inductive Operation
  | replaceOneWithTen
  | replaceTenWithOneAndTwentyFive
  | replaceTwentyFiveWithTwoTens

/-- Applies an operation to the board state --/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.replaceOneWithTen => 
    { ones := state.ones - 1, tens := state.tens + 1, twentyFives := state.twentyFives }
  | Operation.replaceTenWithOneAndTwentyFive => 
    { ones := state.ones + 1, tens := state.tens - 1, twentyFives := state.twentyFives + 1 }
  | Operation.replaceTwentyFiveWithTwoTens => 
    { ones := state.ones, tens := state.tens + 2, twentyFives := state.twentyFives - 1 }

/-- Calculates the sum of all numbers on the board --/
def boardSum (state : BoardState) : ℕ :=
  state.ones + 10 * state.tens + 25 * state.twentyFives

/-- The main theorem to prove --/
theorem min_board_sum_with_hundred_ones : 
  ∃ (final : BoardState) (ops : List Operation),
    final.ones = 100 ∧
    (∀ (state : BoardState), 
      state.ones = 100 → boardSum state ≥ boardSum final) ∧
    boardSum final = 1370 := by
  sorry


end NUMINAMATH_CALUDE_min_board_sum_with_hundred_ones_l445_44598


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_20_l445_44507

theorem smallest_k_for_64_power_gt_4_power_20 : ∃ k : ℕ, k = 7 ∧ (∀ m : ℕ, 64^m > 4^20 → m ≥ k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_20_l445_44507


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l445_44584

theorem distinct_prime_factors_count : 
  let n : ℕ := 101 * 103 * 107 * 109
  ∀ (is_prime_101 : Nat.Prime 101) 
    (is_prime_103 : Nat.Prime 103) 
    (is_prime_107 : Nat.Prime 107) 
    (is_prime_109 : Nat.Prime 109),
  Finset.card (Nat.factors n).toFinset = 4 := by
sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l445_44584


namespace NUMINAMATH_CALUDE_negative_inequality_l445_44526

theorem negative_inequality (m n : ℝ) (h : m > n) : -m < -n := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l445_44526


namespace NUMINAMATH_CALUDE_fraction_multiplication_l445_44533

theorem fraction_multiplication (x y : ℝ) (h : x + y ≠ 0) :
  (3*x * 3*y) / (3*x + 3*y) = 3 * (x*y / (x+y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l445_44533


namespace NUMINAMATH_CALUDE_first_cat_blue_eyed_count_l445_44583

/-- The number of blue-eyed kittens in the first cat's litter -/
def blue_eyed_first_cat : ℕ := sorry

/-- The number of brown-eyed kittens in the first cat's litter -/
def brown_eyed_first_cat : ℕ := 7

/-- The number of blue-eyed kittens in the second cat's litter -/
def blue_eyed_second_cat : ℕ := 4

/-- The number of brown-eyed kittens in the second cat's litter -/
def brown_eyed_second_cat : ℕ := 6

/-- The percentage of blue-eyed kittens among all kittens -/
def blue_eyed_percentage : ℚ := 35 / 100

theorem first_cat_blue_eyed_count :
  blue_eyed_first_cat = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_first_cat_blue_eyed_count_l445_44583


namespace NUMINAMATH_CALUDE_dance_partners_l445_44530

theorem dance_partners (total_participants : ℕ) (n : ℕ) : 
  total_participants = 42 →
  (∀ k : ℕ, k ≥ 1 ∧ k ≤ n → k + 6 ≤ total_participants - n) →
  n + 6 = total_participants - n →
  n = 18 ∧ total_participants - n = 24 :=
by sorry

end NUMINAMATH_CALUDE_dance_partners_l445_44530
