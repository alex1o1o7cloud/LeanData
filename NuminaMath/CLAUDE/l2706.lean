import Mathlib

namespace NUMINAMATH_CALUDE_marks_team_three_pointers_l2706_270609

/-- Represents the number of 3-pointers scored by Mark's team -/
def marks_three_pointers : ℕ := sorry

/-- The total points scored by both teams -/
def total_points : ℕ := 201

/-- The number of 2-pointers scored by Mark's team -/
def marks_two_pointers : ℕ := 25

/-- The number of free throws scored by Mark's team -/
def marks_free_throws : ℕ := 10

theorem marks_team_three_pointers :
  marks_three_pointers = 8 ∧
  (2 * marks_two_pointers + 3 * marks_three_pointers + marks_free_throws) +
  (2 * (2 * marks_two_pointers) + 3 * (marks_three_pointers / 2) + (marks_free_throws / 2)) = total_points :=
sorry

end NUMINAMATH_CALUDE_marks_team_three_pointers_l2706_270609


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2706_270663

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 4) * (x^2 + 11*x + 30) + (x^2 + 8*x - 10) =
  (x^2 + 8*x + 7) * (x^2 + 8*x + 19) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2706_270663


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_12_l2706_270652

theorem no_linear_term_implies_m_equals_12 (m : ℝ) : 
  (∃ a b c : ℝ, (mx + 8) * (2 - 3*x) = a*x^2 + b*x + c ∧ b = 0) → m = 12 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_12_l2706_270652


namespace NUMINAMATH_CALUDE_upward_translation_4_units_l2706_270695

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation2D where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (p : Point2D) (t : Translation2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem upward_translation_4_units 
  (M : Point2D)
  (N : Point2D)
  (h1 : M.x = -1 ∧ M.y = -1)
  (h2 : N.x = -1 ∧ N.y = 3) :
  ∃ (t : Translation2D), t.dx = 0 ∧ t.dy = 4 ∧ applyTranslation M t = N :=
sorry

end NUMINAMATH_CALUDE_upward_translation_4_units_l2706_270695


namespace NUMINAMATH_CALUDE_father_reaches_mom_age_in_three_years_l2706_270683

/-- Represents the ages and time in the problem -/
structure AgesProblem where
  talia_future_age : ℕ      -- Talia's age in 7 years
  talia_future_years : ℕ    -- Years until Talia reaches future_age
  father_current_age : ℕ    -- Talia's father's current age
  mom_age_ratio : ℕ         -- Ratio of mom's age to Talia's current age

/-- Calculates the years until Talia's father reaches Talia's mom's current age -/
def years_until_father_reaches_mom_age (p : AgesProblem) : ℕ :=
  let talia_current_age := p.talia_future_age - p.talia_future_years
  let mom_current_age := talia_current_age * p.mom_age_ratio
  mom_current_age - p.father_current_age

/-- Theorem stating the solution to the problem -/
theorem father_reaches_mom_age_in_three_years (p : AgesProblem) 
    (h1 : p.talia_future_age = 20)
    (h2 : p.talia_future_years = 7)
    (h3 : p.father_current_age = 36)
    (h4 : p.mom_age_ratio = 3) :
  years_until_father_reaches_mom_age p = 3 := by
  sorry


end NUMINAMATH_CALUDE_father_reaches_mom_age_in_three_years_l2706_270683


namespace NUMINAMATH_CALUDE_det_E_l2706_270621

/-- A 3x3 matrix representing a dilation centered at the origin with scale factor 4 -/
def E : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ ↦ 4)

/-- Theorem: The determinant of E is 64 -/
theorem det_E : Matrix.det E = 64 := by
  sorry

end NUMINAMATH_CALUDE_det_E_l2706_270621


namespace NUMINAMATH_CALUDE_reflection_matrix_correct_l2706_270660

/-- Reflection matrix over the line y = x -/
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1; 1, 0]

/-- A point in 2D space -/
def Point := Fin 2 → ℝ

/-- Reflect a point over the line y = x -/
def reflect (p : Point) : Point :=
  λ i => p (if i = 0 then 1 else 0)

theorem reflection_matrix_correct :
  ∀ (p : Point), reflection_matrix.mulVec p = reflect p :=
by sorry

end NUMINAMATH_CALUDE_reflection_matrix_correct_l2706_270660


namespace NUMINAMATH_CALUDE_equidistant_function_property_l2706_270607

open Complex

theorem equidistant_function_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ z : ℂ, abs ((a + b * I) * z - z) = abs ((a + b * I) * z)) →
  abs (a + b * I) = 5 →
  b^2 = 99/4 := by sorry

end NUMINAMATH_CALUDE_equidistant_function_property_l2706_270607


namespace NUMINAMATH_CALUDE_all_nonnegative_possible_l2706_270681

theorem all_nonnegative_possible (nums : List ℝ) (h1 : nums.length = 10) 
  (h2 : nums.sum / nums.length = 0) : 
  ∃ (nonneg_nums : List ℝ), nonneg_nums.length = 10 ∧ 
    nonneg_nums.sum / nonneg_nums.length = 0 ∧
    ∀ x ∈ nonneg_nums, x ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_all_nonnegative_possible_l2706_270681


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2706_270612

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- The point P -/
def P : ℝ × ℝ := (4, 0)

/-- The circle ⊙P -/
def circleP : Circle := { center := P, radius := 5 }

/-- The line y = kx + 2 -/
def line (k : ℝ) : Line := { k := k, b := 2 }

/-- Theorem: The line y = kx + 2 (k ≠ 0) always intersects the circle ⊙P -/
theorem line_intersects_circle (k : ℝ) (h : k ≠ 0) : 
  ∃ (x y : ℝ), (y = k * x + 2) ∧ 
  ((x - circleP.center.1)^2 + (y - circleP.center.2)^2 = circleP.radius^2) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2706_270612


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2706_270657

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  c = 2 * Real.sqrt 2 →
  Real.cos A = -(Real.sqrt 2) / 4 →
  b = 2 ∧
  Real.sin C = (Real.sqrt 7) / 4 ∧
  Real.cos (2 * A + π / 6) = (Real.sqrt 7 - 3 * Real.sqrt 3) / 8 :=
by sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2706_270657


namespace NUMINAMATH_CALUDE_lines_no_common_points_parallel_or_skew_l2706_270676

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate to check if two lines have no common points -/
def NoCommonPoints (l1 l2 : Line3D) : Prop :=
  ∀ t s : ℝ, l1.point + t • l1.direction ≠ l2.point + s • l2.direction

/-- Predicate to check if two lines are parallel -/
def Parallel (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, l1.direction = k • l2.direction

/-- Predicate to check if two lines are skew -/
def Skew (l1 l2 : Line3D) : Prop :=
  ¬ Parallel l1 l2 ∧ NoCommonPoints l1 l2

/-- Theorem stating that if two lines have no common points, they are either parallel or skew -/
theorem lines_no_common_points_parallel_or_skew (l1 l2 : Line3D) :
  NoCommonPoints l1 l2 → Parallel l1 l2 ∨ Skew l1 l2 :=
by
  sorry


end NUMINAMATH_CALUDE_lines_no_common_points_parallel_or_skew_l2706_270676


namespace NUMINAMATH_CALUDE_hula_hoop_problem_l2706_270646

/-- Hula hoop problem -/
theorem hula_hoop_problem (nancy casey morgan alex : ℕ) : 
  nancy = 10 →
  casey = nancy - 3 →
  morgan = 3 * casey →
  alex = (nancy + casey + morgan) / 2 →
  alex = 19 := by sorry

end NUMINAMATH_CALUDE_hula_hoop_problem_l2706_270646


namespace NUMINAMATH_CALUDE_joan_pencils_l2706_270684

theorem joan_pencils (initial_pencils final_pencils : ℕ) 
  (h1 : initial_pencils = 33)
  (h2 : final_pencils = 60) :
  final_pencils - initial_pencils = 27 := by
  sorry

end NUMINAMATH_CALUDE_joan_pencils_l2706_270684


namespace NUMINAMATH_CALUDE_polynomial_coefficient_b_l2706_270685

theorem polynomial_coefficient_b (a b c : ℚ) :
  (∀ x : ℚ, (3 * x^2 - 2 * x + 5/4) * (a * x^2 + b * x + c) = 
    9 * x^4 - 5 * x^3 + 31/4 * x^2 - 10/3 * x + 5/12) →
  b = 1/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_b_l2706_270685


namespace NUMINAMATH_CALUDE_max_distance_ellipse_to_line_l2706_270686

/-- An ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop := λ x y ↦ x^2 / (a^2) + y^2 / (b^2) = 1

/-- A line in the xy-plane represented by parametric equations -/
structure ParametricLine where
  fx : ℝ → ℝ
  fy : ℝ → ℝ

/-- The distance between a point and a line -/
def distance_point_to_line (x y : ℝ) (l : ParametricLine) : ℝ := sorry

/-- The maximum distance from a point on an ellipse to a line -/
def max_distance (e : Ellipse) (l : ParametricLine) : ℝ := sorry

theorem max_distance_ellipse_to_line :
  let e : Ellipse := { a := 4, b := 2, equation := λ x y ↦ x^2 / 16 + y^2 / 4 = 1 }
  let l : ParametricLine := { fx := λ t ↦ Real.sqrt 2 - t, fy := λ t ↦ t / 2 }
  max_distance e l = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_to_line_l2706_270686


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2706_270641

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h1 : a 4 + a 10 + a 16 = 30) : a 18 - 2 * a 14 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2706_270641


namespace NUMINAMATH_CALUDE_ab_equals_zero_l2706_270677

theorem ab_equals_zero (a b : ℝ) 
  (h1 : (2 : ℝ) ^ a = (2 : ℝ) ^ (2 * (b + 1)))
  (h2 : (7 : ℝ) ^ b = (7 : ℝ) ^ (a - 2)) : 
  a * b = 0 := by
sorry

end NUMINAMATH_CALUDE_ab_equals_zero_l2706_270677


namespace NUMINAMATH_CALUDE_walking_speed_solution_l2706_270610

/-- Represents the problem of finding A's walking speed -/
def walking_speed_problem (v : ℝ) : Prop :=
  let b_speed : ℝ := 20
  let time_diff : ℝ := 3
  let catch_up_distance : ℝ := 60
  let catch_up_time : ℝ := catch_up_distance / b_speed
  v * (time_diff + catch_up_time) = catch_up_distance ∧ v = 10

/-- Theorem stating that the solution to the walking speed problem is 10 kmph -/
theorem walking_speed_solution :
  ∃ v : ℝ, walking_speed_problem v :=
sorry

end NUMINAMATH_CALUDE_walking_speed_solution_l2706_270610


namespace NUMINAMATH_CALUDE_min_boxes_to_eliminate_l2706_270628

/-- Represents the game setup with total boxes and valuable boxes -/
structure GameSetup :=
  (total_boxes : ℕ)
  (valuable_boxes : ℕ)

/-- Calculates the probability of holding a valuable box -/
def probability (setup : GameSetup) (eliminated : ℕ) : ℚ :=
  setup.valuable_boxes / (setup.total_boxes - eliminated)

/-- Theorem stating the minimum number of boxes to eliminate -/
theorem min_boxes_to_eliminate (setup : GameSetup) 
  (h1 : setup.total_boxes = 30)
  (h2 : setup.valuable_boxes = 9) :
  ∃ (n : ℕ), 
    (n = 3) ∧ 
    (probability setup n ≥ 1/3) ∧ 
    (∀ m : ℕ, m < n → probability setup m < 1/3) :=
sorry

end NUMINAMATH_CALUDE_min_boxes_to_eliminate_l2706_270628


namespace NUMINAMATH_CALUDE_coffee_brew_efficiency_l2706_270659

theorem coffee_brew_efficiency (total_lbs : ℕ) (cups_per_day : ℕ) (total_days : ℕ) 
  (h1 : total_lbs = 3)
  (h2 : cups_per_day = 3)
  (h3 : total_days = 40) :
  (cups_per_day * total_days) / total_lbs = 40 := by
  sorry

#check coffee_brew_efficiency

end NUMINAMATH_CALUDE_coffee_brew_efficiency_l2706_270659


namespace NUMINAMATH_CALUDE_square_area_after_cut_l2706_270664

theorem square_area_after_cut (x : ℝ) : 
  x > 0 → 
  x^2 - 2*x = 80 → 
  x^2 = 100 := by
sorry

end NUMINAMATH_CALUDE_square_area_after_cut_l2706_270664


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l2706_270661

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
by sorry

theorem lower_bound_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) = -17 + 12 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l2706_270661


namespace NUMINAMATH_CALUDE_sqrt3_div3_sufficient_sqrt3_div3_not_necessary_sqrt3_div3_sufficient_not_necessary_l2706_270620

/-- The condition for a line to be tangent to a circle --/
def is_tangent (k : ℝ) : Prop :=
  let line := fun x => k * (x + 2)
  let circle := fun x y => x^2 + y^2 = 1
  ∃ x y, circle x y ∧ y = line x ∧
  ∀ x' y', circle x' y' → (y' - line x')^2 ≥ 0

/-- k = √3/3 is sufficient for tangency --/
theorem sqrt3_div3_sufficient :
  is_tangent (Real.sqrt 3 / 3) := by sorry

/-- k = √3/3 is not necessary for tangency --/
theorem sqrt3_div3_not_necessary :
  ∃ k, k ≠ Real.sqrt 3 / 3 ∧ is_tangent k := by sorry

/-- k = √3/3 is a sufficient but not necessary condition for tangency --/
theorem sqrt3_div3_sufficient_not_necessary :
  (is_tangent (Real.sqrt 3 / 3)) ∧
  (∃ k, k ≠ Real.sqrt 3 / 3 ∧ is_tangent k) := by sorry

end NUMINAMATH_CALUDE_sqrt3_div3_sufficient_sqrt3_div3_not_necessary_sqrt3_div3_sufficient_not_necessary_l2706_270620


namespace NUMINAMATH_CALUDE_smallest_coprime_to_210_l2706_270692

theorem smallest_coprime_to_210 :
  ∀ y : ℕ, y > 1 → y < 11 → Nat.gcd y 210 ≠ 1 ∧ Nat.gcd 11 210 = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_coprime_to_210_l2706_270692


namespace NUMINAMATH_CALUDE_m_range_when_p_false_l2706_270655

theorem m_range_when_p_false :
  (¬∀ x : ℝ, ∃ m : ℝ, 4*x - 2*x + 1 + m = 0) →
  {m : ℝ | ∃ x : ℝ, 4*x - 2*x + 1 + m ≠ 0} = Set.Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_when_p_false_l2706_270655


namespace NUMINAMATH_CALUDE_group_size_theorem_l2706_270631

theorem group_size_theorem (n : ℕ) (k : ℕ) : 
  (k * (n - 1) * n = 440 ∧ n > 0 ∧ k > 0) → (n = 5 ∨ n = 11) :=
sorry

end NUMINAMATH_CALUDE_group_size_theorem_l2706_270631


namespace NUMINAMATH_CALUDE_simplify_fourth_root_l2706_270687

theorem simplify_fourth_root (a : ℝ) (h : a < 1/2) : 
  (2*a - 1)^2^(1/4) = Real.sqrt (1 - 2*a) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_l2706_270687


namespace NUMINAMATH_CALUDE_ratio_of_x_intercepts_l2706_270667

/-- Given two lines with the same non-zero y-intercept, where the first line has
    a slope of 8 and an x-intercept of (u, 0), and the second line has a slope
    of 4 and an x-intercept of (v, 0), prove that the ratio of u to v is 1/2. -/
theorem ratio_of_x_intercepts (b : ℝ) (u v : ℝ) 
    (h1 : b ≠ 0)
    (h2 : 0 = 8 * u + b)
    (h3 : 0 = 4 * v + b) :
    u / v = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_intercepts_l2706_270667


namespace NUMINAMATH_CALUDE_third_number_value_l2706_270629

theorem third_number_value (a b c : ℝ) : 
  a + b + c = 500 →
  a = 200 →
  b = 2 * c →
  c = 100 := by
sorry

end NUMINAMATH_CALUDE_third_number_value_l2706_270629


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_2000_l2706_270648

theorem greatest_multiple_of_four_under_cube_root_2000 :
  ∃ (x : ℕ), x > 0 ∧ 4 ∣ x ∧ x^3 < 2000 ∧
  ∀ (y : ℕ), y > 0 → 4 ∣ y → y^3 < 2000 → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_2000_l2706_270648


namespace NUMINAMATH_CALUDE_q_polynomial_expression_l2706_270603

theorem q_polynomial_expression (q : ℝ → ℝ) : 
  (∀ x, q x + (2*x^6 + 4*x^4 + 6*x^2 + 2) = 8*x^4 + 27*x^3 + 30*x^2 + 10*x + 3) →
  (∀ x, q x = -2*x^6 + 4*x^4 + 27*x^3 + 24*x^2 + 10*x + 1) := by
sorry

end NUMINAMATH_CALUDE_q_polynomial_expression_l2706_270603


namespace NUMINAMATH_CALUDE_A_intersect_B_l2706_270644

def A : Set ℕ := {0, 2, 4}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2^x}

theorem A_intersect_B : A ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2706_270644


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2706_270653

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 3| - |x + 1| ≤ a) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2706_270653


namespace NUMINAMATH_CALUDE_unique_valid_statement_l2706_270627

theorem unique_valid_statement : ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧
  (Real.sqrt (a^2 + b^2) = |a - b|) ∧
  ¬(Real.sqrt (a^2 + b^2) = a^2 - b^2) ∧
  ¬(Real.sqrt (a^2 + b^2) = a + b) ∧
  ¬(Real.sqrt (a^2 + b^2) = |a| + |b|) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_valid_statement_l2706_270627


namespace NUMINAMATH_CALUDE_jeff_swimming_laps_l2706_270622

/-- The number of laps Jeff swam on Saturday -/
def saturday_laps : ℕ := 27

/-- The number of laps Jeff swam on Sunday morning -/
def sunday_morning_laps : ℕ := 15

/-- The number of laps Jeff had remaining after the break -/
def remaining_laps : ℕ := 56

/-- The total number of laps Jeff's coach required him to swim over the weekend -/
def total_required_laps : ℕ := saturday_laps + sunday_morning_laps + remaining_laps

theorem jeff_swimming_laps : total_required_laps = 98 := by
  sorry

end NUMINAMATH_CALUDE_jeff_swimming_laps_l2706_270622


namespace NUMINAMATH_CALUDE_angle_calculation_l2706_270601

-- Define the triangles and angles
def Triangle (a b c : ℝ) := a + b + c = 180

-- Theorem statement
theorem angle_calculation (T1_angle1 T1_angle2 T2_angle1 T2_angle2 α β : ℝ) 
  (h1 : Triangle T1_angle1 T1_angle2 (180 - α))
  (h2 : Triangle T2_angle1 T2_angle2 β)
  (h3 : T1_angle1 = 70)
  (h4 : T1_angle2 = 50)
  (h5 : T2_angle1 = 45)
  (h6 : T2_angle2 = 50) :
  α = 120 ∧ β = 85 := by
  sorry

end NUMINAMATH_CALUDE_angle_calculation_l2706_270601


namespace NUMINAMATH_CALUDE_rectangle_to_square_cut_l2706_270691

theorem rectangle_to_square_cut (length width : ℝ) (h1 : length = 16) (h2 : width = 9) :
  ∃ (side : ℝ), side = 12 ∧ 
  2 * (side * side) = length * width ∧
  side ≤ length ∧ side ≤ width + (length - side) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_cut_l2706_270691


namespace NUMINAMATH_CALUDE_find_M_and_N_convex_polygon_diagonals_calculate_y_l2706_270671

-- Part 1 and 2
theorem find_M_and_N :
  ∃ (M N : ℕ),
    M < 10 ∧ N < 10 ∧
    258024 * 10 + M * 10 + 8 * 9 = 2111110 * N * 11 ∧
    M = 9 ∧ N = 2 := by sorry

-- Part 3
theorem convex_polygon_diagonals (n : ℕ) (h : n = 20) :
  (n * (n - 3)) / 2 = 170 := by sorry

-- Part 4
theorem calculate_y (a b : ℕ) (h1 : a = 99) (h2 : b = 49) :
  a * b + a + b + 1 = 4999 := by sorry

end NUMINAMATH_CALUDE_find_M_and_N_convex_polygon_diagonals_calculate_y_l2706_270671


namespace NUMINAMATH_CALUDE_fishing_theorem_l2706_270638

def fishing_problem (jordan_catch perry_catch alex_catch bird_steal release_fraction : ℕ) : ℕ :=
  let total_catch := jordan_catch + perry_catch + alex_catch
  let after_bird := total_catch - bird_steal
  let to_release := (after_bird * release_fraction) / 3
  after_bird - to_release

theorem fishing_theorem :
  fishing_problem 4 8 36 2 1 = 31 :=
by sorry

end NUMINAMATH_CALUDE_fishing_theorem_l2706_270638


namespace NUMINAMATH_CALUDE_diana_apollo_dice_probability_l2706_270602

theorem diana_apollo_dice_probability :
  let diana_die := Finset.range 10
  let apollo_die := Finset.range 6
  let total_outcomes := diana_die.card * apollo_die.card
  let favorable_outcomes := (apollo_die.sum fun a => 
    (diana_die.filter (fun d => d > a)).card)
  (favorable_outcomes : ℚ) / total_outcomes = 13 / 20 := by
sorry

end NUMINAMATH_CALUDE_diana_apollo_dice_probability_l2706_270602


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2706_270630

def U : Set ℝ := {x | x ≥ 0}
def A : Set ℝ := {x | x ≥ 1}

theorem complement_of_A_in_U : 
  (U \ A) = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2706_270630


namespace NUMINAMATH_CALUDE_smallest_dividend_l2706_270632

theorem smallest_dividend (A B : ℕ) (h1 : A = B * 28 + 4) (h2 : B > 0) : A ≥ 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dividend_l2706_270632


namespace NUMINAMATH_CALUDE_smallest_quadratic_nonresidue_bound_l2706_270689

theorem smallest_quadratic_nonresidue_bound (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ x : ℕ, x < Int.floor (Real.sqrt p + 1) ∧ x > 0 ∧ ¬ ∃ y : ℤ, (y * y) % p = x % p := by
  sorry

end NUMINAMATH_CALUDE_smallest_quadratic_nonresidue_bound_l2706_270689


namespace NUMINAMATH_CALUDE_composite_sum_of_product_equal_l2706_270651

theorem composite_sum_of_product_equal (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ a^1984 + b^1984 + c^1984 + d^1984 = m * n :=
sorry

end NUMINAMATH_CALUDE_composite_sum_of_product_equal_l2706_270651


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2706_270625

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 - 4*i) / (1 - i)
  (z.im : ℝ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2706_270625


namespace NUMINAMATH_CALUDE_parallel_lines_minimum_value_l2706_270697

theorem parallel_lines_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 2 * a + 3 * b ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_minimum_value_l2706_270697


namespace NUMINAMATH_CALUDE_incomplete_factor_multiple_statement_l2706_270666

theorem incomplete_factor_multiple_statement : ¬(56 / 7 = 8 → (∃n : ℕ, 56 = n * 7) ∧ (∃m : ℕ, 7 * m = 56)) := by
  sorry

end NUMINAMATH_CALUDE_incomplete_factor_multiple_statement_l2706_270666


namespace NUMINAMATH_CALUDE_rectangle_length_l2706_270694

theorem rectangle_length (square_side : ℝ) (rectangle_area : ℝ) : 
  square_side = 15 →
  rectangle_area = 216 →
  ∃ (rectangle_length rectangle_width : ℝ),
    4 * square_side = 2 * (rectangle_length + rectangle_width) ∧
    rectangle_length * rectangle_width = rectangle_area ∧
    rectangle_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l2706_270694


namespace NUMINAMATH_CALUDE_division_equation_problem_l2706_270606

theorem division_equation_problem (A B C : ℕ) : 
  (∃ (q : ℕ), A = B * q + 8) → -- A ÷ B = C with remainder 8
  (A + B + C = 2994) →         -- Sum condition
  (A = 8 ∨ A = 2864) :=        -- Conclusion
by
  sorry

end NUMINAMATH_CALUDE_division_equation_problem_l2706_270606


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1987_l2706_270635

theorem rightmost_three_digits_of_7_to_1987 :
  7^1987 ≡ 643 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1987_l2706_270635


namespace NUMINAMATH_CALUDE_point_P_in_fourth_quadrant_l2706_270605

def point_P : ℝ × ℝ := (8, -3)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_P_in_fourth_quadrant :
  in_fourth_quadrant point_P := by
  sorry

end NUMINAMATH_CALUDE_point_P_in_fourth_quadrant_l2706_270605


namespace NUMINAMATH_CALUDE_orchestra_seat_price_l2706_270672

/-- Represents the theater ticket sales scenario --/
structure TheaterSales where
  orchestra_price : ℕ
  balcony_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ
  balcony_orchestra_diff : ℕ

/-- Theorem stating the orchestra seat price given the conditions --/
theorem orchestra_seat_price (ts : TheaterSales)
  (h1 : ts.balcony_price = 8)
  (h2 : ts.total_tickets = 340)
  (h3 : ts.total_revenue = 3320)
  (h4 : ts.balcony_orchestra_diff = 40) :
  ts.orchestra_price = 12 := by
  sorry


end NUMINAMATH_CALUDE_orchestra_seat_price_l2706_270672


namespace NUMINAMATH_CALUDE_expression_evaluation_l2706_270614

theorem expression_evaluation : 7^2 - 4^2 + 2*5 - 3^3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2706_270614


namespace NUMINAMATH_CALUDE_negative_one_power_equality_l2706_270633

theorem negative_one_power_equality : (-1 : ℤ)^3 = (-1 : ℤ)^2023 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_power_equality_l2706_270633


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2706_270615

theorem quadratic_root_difference (a b c : ℝ) (h : a ≠ 0) :
  let eq := fun x => a * x^2 + b * x + c
  let r1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  eq 1 + 40 + 300 = -64 →
  |r1 - r2| = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2706_270615


namespace NUMINAMATH_CALUDE_students_taking_courses_l2706_270673

theorem students_taking_courses (total : ℕ) (history : ℕ) (statistics : ℕ) (history_only : ℕ)
  (h_total : total = 89)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_only : history_only = 27) :
  ∃ (both : ℕ) (statistics_only : ℕ),
    history_only + statistics_only + both = 59 ∧
    both = history - history_only ∧
    statistics_only = statistics - both :=
by sorry

end NUMINAMATH_CALUDE_students_taking_courses_l2706_270673


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l2706_270668

/-- Given a purchase in country B with a tax-free threshold, calculate the tax rate -/
theorem tax_rate_calculation (total_value tax_free_threshold tax_paid : ℝ) : 
  total_value = 1720 →
  tax_free_threshold = 600 →
  tax_paid = 123.2 →
  (tax_paid / (total_value - tax_free_threshold)) * 100 = 11 := by
sorry

end NUMINAMATH_CALUDE_tax_rate_calculation_l2706_270668


namespace NUMINAMATH_CALUDE_blue_sky_project_expo_course_l2706_270642

theorem blue_sky_project_expo_course (n m : ℕ) (hn : n = 6) (hm : m = 6) :
  (Nat.choose n 2) * (m - 1) ^ (n - 2) = 
    (Nat.choose 6 2) * 5^4 :=
sorry

end NUMINAMATH_CALUDE_blue_sky_project_expo_course_l2706_270642


namespace NUMINAMATH_CALUDE_largest_n_for_product_l2706_270637

/-- An arithmetic sequence with integer terms -/
def ArithmeticSequence (a₁ : ℤ) (d : ℤ) : ℕ → ℤ := fun n => a₁ + (n - 1) * d

theorem largest_n_for_product (x y : ℤ) (hxy : x < y) :
  let a := ArithmeticSequence 2 x
  let b := ArithmeticSequence 3 y
  (∃ n : ℕ, a n * b n = 1638) →
  (∀ m : ℕ, a m * b m = 1638 → m ≤ 35) ∧
  (a 35 * b 35 = 1638) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_l2706_270637


namespace NUMINAMATH_CALUDE_prob_jack_queen_king_ace_value_l2706_270674

-- Define the total number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of cards for each face value
def cards_per_face : ℕ := 4

-- Define the probability of drawing the specific sequence
def prob_jack_queen_king_ace : ℚ :=
  (cards_per_face : ℚ) / total_cards *
  (cards_per_face : ℚ) / (total_cards - 1) *
  (cards_per_face : ℚ) / (total_cards - 2) *
  (cards_per_face : ℚ) / (total_cards - 3)

-- Theorem statement
theorem prob_jack_queen_king_ace_value :
  prob_jack_queen_king_ace = 16 / 4048375 := by
  sorry

end NUMINAMATH_CALUDE_prob_jack_queen_king_ace_value_l2706_270674


namespace NUMINAMATH_CALUDE_faye_earnings_l2706_270650

/-- Calculates the total amount earned from selling necklaces -/
def total_earned (bead_count gemstone_count pearl_count crystal_count : ℕ) 
                 (bead_price gemstone_price pearl_price crystal_price : ℕ) : ℕ :=
  bead_count * bead_price + 
  gemstone_count * gemstone_price + 
  pearl_count * pearl_price + 
  crystal_count * crystal_price

/-- Theorem: The total amount Faye earned is $190 -/
theorem faye_earnings : 
  total_earned 3 7 2 5 7 10 12 15 = 190 := by
  sorry

end NUMINAMATH_CALUDE_faye_earnings_l2706_270650


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l2706_270624

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x + y = -4/5 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l2706_270624


namespace NUMINAMATH_CALUDE_correct_equation_l2706_270675

theorem correct_equation (x y : ℝ) : x * y - 2 * (x * y) = -(x * y) := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l2706_270675


namespace NUMINAMATH_CALUDE_problem_solution_l2706_270647

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 3 = y^2) 
  (h3 : x / 5 = 5*y) : 
  x = 625 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2706_270647


namespace NUMINAMATH_CALUDE_divide_3x8_rectangle_into_trominoes_l2706_270611

/-- Represents an L-shaped tromino -/
structure LTromino :=
  (cells : Nat)

/-- Represents a rectangle -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- Number of ways to divide a rectangle into L-shaped trominoes -/
def divideRectangle (r : Rectangle) (t : LTromino) : Nat :=
  sorry

/-- Theorem: The number of ways to divide a 3 × 8 rectangle into L-shaped trominoes is 16 -/
theorem divide_3x8_rectangle_into_trominoes :
  let r := Rectangle.mk 8 3
  let t := LTromino.mk 3
  divideRectangle r t = 16 := by
  sorry

end NUMINAMATH_CALUDE_divide_3x8_rectangle_into_trominoes_l2706_270611


namespace NUMINAMATH_CALUDE_xy_value_l2706_270698

theorem xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h1 : x^2 + y^2 = 3) (h2 : x^4 + y^4 = 15/8) : x * y = Real.sqrt 57 / 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2706_270698


namespace NUMINAMATH_CALUDE_maria_students_l2706_270669

/-- The number of students in Maria's high school -/
def M : ℕ := sorry

/-- The number of students in Jackson's high school -/
def J : ℕ := sorry

/-- Maria's high school has 4 times as many students as Jackson's high school -/
axiom maria_jackson_ratio : M = 4 * J

/-- The total number of students in both high schools is 3600 -/
axiom total_students : M + J = 3600

/-- Theorem: Maria's high school has 2880 students -/
theorem maria_students : M = 2880 := by sorry

end NUMINAMATH_CALUDE_maria_students_l2706_270669


namespace NUMINAMATH_CALUDE_car_trade_profit_l2706_270678

theorem car_trade_profit (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let buying_price := original_price * (1 - 0.05)
  let selling_price := buying_price * (1 + 0.60)
  let profit := selling_price - original_price
  let profit_percentage := (profit / original_price) * 100
  profit_percentage = 52 := by sorry

end NUMINAMATH_CALUDE_car_trade_profit_l2706_270678


namespace NUMINAMATH_CALUDE_function_roots_bound_l2706_270618

/-- The function f(x) defined with given parameters has no more than 14 positive roots -/
theorem function_roots_bound 
  (a b c d : ℝ) 
  (k l m p q r : ℕ) 
  (h1 : k ≥ l ∧ l ≥ m) 
  (h2 : p ≥ q ∧ q ≥ r) :
  let f : ℝ → ℝ := λ x => a*(x+1)^k * (x+2)^p + b*(x+1)^l * (x+2)^q + c*(x+1)^m * (x+2)^r - d
  ∃ (S : Finset ℝ), (∀ x ∈ S, x > 0 ∧ f x = 0) ∧ Finset.card S ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_function_roots_bound_l2706_270618


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l2706_270634

/-- Given that 40 smaller pie crusts each use 1/8 cup of flour,
    prove that 25 larger pie crusts using the same total amount of flour
    will each require 1/5 cup of flour. -/
theorem pie_crust_flour_calculation (small_crusts : ℕ) (large_crusts : ℕ)
  (small_flour : ℚ) (large_flour : ℚ) :
  small_crusts = 40 →
  large_crusts = 25 →
  small_flour = 1/8 →
  small_crusts * small_flour = large_crusts * large_flour →
  large_flour = 1/5 := by
sorry

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l2706_270634


namespace NUMINAMATH_CALUDE_unique_solutions_l2706_270604

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_sequence (seq : List ℕ) : Prop :=
  seq.length = 16 ∧
  (∀ n, n ∈ seq → 1 ≤ n ∧ n ≤ 16) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 16 → n ∈ seq) ∧
  (∀ i, i < 15 → is_perfect_square (seq[i]! + seq[i+1]!))

def solution1 : List ℕ := [16, 9, 7, 2, 14, 11, 5, 4, 12, 13, 3, 6, 10, 15, 1, 8]
def solution2 : List ℕ := [8, 1, 15, 10, 6, 3, 13, 12, 4, 5, 11, 14, 2, 7, 9, 16]

theorem unique_solutions :
  (∀ seq : List ℕ, valid_sequence seq → seq = solution1 ∨ seq = solution2) ∧
  valid_sequence solution1 ∧
  valid_sequence solution2 :=
sorry

end NUMINAMATH_CALUDE_unique_solutions_l2706_270604


namespace NUMINAMATH_CALUDE_well_depth_l2706_270636

/-- The depth of a well given specific conditions -/
theorem well_depth : 
  -- Define the distance function
  let distance (t : ℝ) : ℝ := 16 * t^2
  -- Define the speed of sound
  let sound_speed : ℝ := 1120
  -- Define the total time
  let total_time : ℝ := 7.7
  -- Define the depth of the well
  let depth : ℝ := distance (total_time - depth / sound_speed)
  -- Prove that the depth is 784 feet
  depth = 784 := by sorry

end NUMINAMATH_CALUDE_well_depth_l2706_270636


namespace NUMINAMATH_CALUDE_chicken_problem_model_l2706_270649

/-- Represents the system of equations for the chicken buying problem -/
def chicken_equations (x y : ℕ) : Prop :=
  (8 * x - y = 3) ∧ (y - 7 * x = 4)

/-- Proves that the system of equations correctly models the given conditions -/
theorem chicken_problem_model (x y : ℕ) :
  (x > 0 ∧ y > 0) →
  (chicken_equations x y ↔
    (8 * x = y + 3 ∧ 7 * x + 4 = y)) :=
by sorry

end NUMINAMATH_CALUDE_chicken_problem_model_l2706_270649


namespace NUMINAMATH_CALUDE_line_equation_for_triangle_l2706_270626

/-- Given a line passing through (-a, 0) and cutting a triangle with area T in the second quadrant,
    prove that the equation of the line is 2Tx - a²y + 2aT = 0 --/
theorem line_equation_for_triangle (a T : ℝ) (h_a : a > 0) (h_T : T > 0) :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b → (x = -a ∧ y = 0) ∨ (x ≥ 0 ∧ y ≥ 0)) ∧ 
    (1/2 * a * (b : ℝ) = T) ∧
    (∀ x y : ℝ, y = m * x + b ↔ 2 * T * x - a^2 * y + 2 * a * T = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_for_triangle_l2706_270626


namespace NUMINAMATH_CALUDE_square_difference_evaluation_l2706_270639

theorem square_difference_evaluation (c d : ℕ) (h1 : c = 5) (h2 : d = 3) :
  (c^2 + d)^2 - (c^2 - d)^2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_evaluation_l2706_270639


namespace NUMINAMATH_CALUDE_adam_books_before_shopping_l2706_270662

/-- Calculates the number of books Adam had before his shopping trip -/
def books_before_shopping (shelves : ℕ) (avg_books_per_shelf : ℕ) (new_books : ℕ) (leftover : ℕ) : ℕ :=
  shelves * avg_books_per_shelf - (new_books - leftover)

/-- Theorem stating that Adam had 56 books before his shopping trip -/
theorem adam_books_before_shopping :
  books_before_shopping 4 20 26 2 = 56 := by
  sorry

#eval books_before_shopping 4 20 26 2

end NUMINAMATH_CALUDE_adam_books_before_shopping_l2706_270662


namespace NUMINAMATH_CALUDE_count_nonincreasing_7digit_integers_l2706_270665

/-- The number of 7-digit positive integers with nonincreasing digits -/
def nonincreasing_7digit_integers : ℕ :=
  Nat.choose 16 7 - 1

/-- Proposition: The number of 7-digit positive integers with nonincreasing digits is 11439 -/
theorem count_nonincreasing_7digit_integers :
  nonincreasing_7digit_integers = 11439 := by
  sorry

end NUMINAMATH_CALUDE_count_nonincreasing_7digit_integers_l2706_270665


namespace NUMINAMATH_CALUDE_shari_walking_distance_l2706_270690

-- Define Shari's walking speed
def walking_speed : ℝ := 4

-- Define the duration of the first walking segment
def first_segment_duration : ℝ := 2

-- Define the duration of the rest period (not used in calculation)
def rest_duration : ℝ := 0.5

-- Define the duration of the second walking segment
def second_segment_duration : ℝ := 1

-- Define the function to calculate distance
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem shari_walking_distance :
  distance walking_speed first_segment_duration +
  distance walking_speed second_segment_duration = 12 := by
  sorry

end NUMINAMATH_CALUDE_shari_walking_distance_l2706_270690


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2706_270613

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_relation : ℝ → ℝ) 
  (base : ℝ) :
  area = 128 ∧ 
  altitude_base_relation = (λ x => 2 * x) ∧ 
  area = base * (altitude_base_relation base) →
  base = 8 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2706_270613


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2706_270679

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (((a > 2 ∧ b > 2) → (a + b > 4)) ∧ 
   (∃ x y : ℝ, x + y > 4 ∧ ¬(x > 2 ∧ y > 2))) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2706_270679


namespace NUMINAMATH_CALUDE_abs_greater_than_negative_l2706_270608

theorem abs_greater_than_negative (a b : ℝ) (h : a < b ∧ b < 0) : |a| > -b := by
  sorry

end NUMINAMATH_CALUDE_abs_greater_than_negative_l2706_270608


namespace NUMINAMATH_CALUDE_R_equals_eleven_l2706_270617

def F : ℝ := 2^121 - 1

def Q : ℕ := 120

theorem R_equals_eleven :
  Real.sqrt (Real.log (1 + F) / Real.log 2) = 11 := by sorry

end NUMINAMATH_CALUDE_R_equals_eleven_l2706_270617


namespace NUMINAMATH_CALUDE_factorization_problems_l2706_270699

theorem factorization_problems (x y : ℝ) : 
  (7 * x^2 - 63 = 7 * (x + 3) * (x - 3)) ∧ 
  (x^3 + 6 * x^2 * y + 9 * x * y^2 = x * (x + 3 * y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2706_270699


namespace NUMINAMATH_CALUDE_v_2023_equals_1_l2706_270623

-- Define the function g
def g : ℕ → ℕ
| 1 => 3
| 2 => 4
| 3 => 2
| 4 => 1
| 5 => 5
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n)

-- Theorem statement
theorem v_2023_equals_1 : v 2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_v_2023_equals_1_l2706_270623


namespace NUMINAMATH_CALUDE_equation_solution_l2706_270654

theorem equation_solution :
  ∃! x : ℝ, x - 5 ≥ 0 ∧
  (7 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 3) +
   8 / (Real.sqrt (x - 5) + 3) + 13 / (Real.sqrt (x - 5) + 10) = 0) ∧
  x = 1486 / 225 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2706_270654


namespace NUMINAMATH_CALUDE_h_at_two_l2706_270682

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the tangent line function g
def g (x : ℝ) : ℝ := (3*x^2 - 3)*x - 2*x^3

-- Define the function h
def h (x : ℝ) : ℝ := f x - g x

-- Theorem statement
theorem h_at_two : h 2 = 2^3 - 12*2 + 16 := by sorry

end NUMINAMATH_CALUDE_h_at_two_l2706_270682


namespace NUMINAMATH_CALUDE_equalizeTable_l2706_270696

-- Define the table as a matrix
def Table (n : ℕ) := Matrix (Fin n) (Fin n) ℕ

-- Initial configuration of the table
def initialTable (n : ℕ) : Table n :=
  Matrix.diagonal (λ _ => 1)

-- Define a rook path as a list of positions
def RookPath (n : ℕ) := List (Fin n × Fin n)

-- Predicate to check if a path is valid (closed and non-self-intersecting)
def isValidPath (n : ℕ) (path : RookPath n) : Prop := sorry

-- Function to apply a rook transformation
def applyRookTransformation (t : Table n) (path : RookPath n) : Table n := sorry

-- Predicate to check if all numbers in the table are equal
def allEqual (t : Table n) : Prop := sorry

-- The main theorem
theorem equalizeTable (n : ℕ) :
  (∃ (transformations : List (RookPath n)), 
    allEqual (transformations.foldl applyRookTransformation (initialTable n))) ↔ 
  Odd n := by sorry

end NUMINAMATH_CALUDE_equalizeTable_l2706_270696


namespace NUMINAMATH_CALUDE_commission_percentage_is_4_percent_l2706_270688

/-- Represents the commission rate as a real number between 0 and 1 -/
def CommissionRate : Type := { r : ℝ // 0 ≤ r ∧ r ≤ 1 }

/-- The first salary option -/
def salary1 : ℝ := 1800

/-- The base salary for the second option -/
def baseSalary : ℝ := 1600

/-- The sales amount at which both options are equal -/
def equalSalesAmount : ℝ := 5000

/-- The commission rate that makes both options equal at the given sales amount -/
def commissionRate : CommissionRate :=
  sorry

theorem commission_percentage_is_4_percent :
  (commissionRate.val * 100 : ℝ) = 4 :=
sorry

end NUMINAMATH_CALUDE_commission_percentage_is_4_percent_l2706_270688


namespace NUMINAMATH_CALUDE_partnership_profit_l2706_270616

/-- Calculates the total profit of a partnership given the investments, time periods, and one partner's profit. -/
def total_profit (a_investment : ℕ) (b_investment : ℕ) (a_period : ℕ) (b_period : ℕ) (b_profit : ℕ) : ℕ :=
  let profit_ratio := (a_investment * a_period) / (b_investment * b_period)
  let total_parts := profit_ratio + 1
  total_parts * b_profit

/-- Theorem stating that under the given conditions, the total profit is 42000. -/
theorem partnership_profit : 
  ∀ (b_investment : ℕ) (b_period : ℕ),
    b_investment > 0 → b_period > 0 →
    total_profit (3 * b_investment) b_investment (2 * b_period) b_period 6000 = 42000 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l2706_270616


namespace NUMINAMATH_CALUDE_subtracted_number_l2706_270619

theorem subtracted_number (x N : ℤ) (h1 : 3 * x = (N - x) + 16) (h2 : x = 13) : N = 36 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2706_270619


namespace NUMINAMATH_CALUDE_expected_socks_theorem_l2706_270645

/-- The expected number of socks picked to retrieve both favorite socks -/
def expected_socks_picked (n : ℕ) : ℚ :=
  2 * (n + 1) / 3

/-- Theorem: The expected number of socks picked to retrieve both favorite socks is 2(n+1)/3 -/
theorem expected_socks_theorem (n : ℕ) (h : n ≥ 2) :
  expected_socks_picked n = 2 * (n + 1) / 3 := by
  sorry

#check expected_socks_theorem

end NUMINAMATH_CALUDE_expected_socks_theorem_l2706_270645


namespace NUMINAMATH_CALUDE_line_direction_vector_l2706_270656

-- Define the two points on the line
def point1 : ℝ × ℝ := (-3, 0)
def point2 : ℝ × ℝ := (0, 3)

-- Define the direction vector
def direction_vector : ℝ × ℝ := (3, 3)

-- Theorem statement
theorem line_direction_vector :
  (point2.1 - point1.1, point2.2 - point1.2) = direction_vector :=
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2706_270656


namespace NUMINAMATH_CALUDE_intersection_point_distance_to_line_l2706_270680

-- Define the lines
def l1 (x y : ℝ) : Prop := x - y + 2 = 0
def l2 (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l (x y : ℝ) : Prop := 3*x + 4*y - 10 = 0

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Theorem for the intersection point
theorem intersection_point : ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧ x = -1 ∧ y = 1 := by sorry

-- Theorem for the distance
theorem distance_to_line : 
  let d := |3 * P.1 + 4 * P.2 - 10| / Real.sqrt (3^2 + 4^2)
  d = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_point_distance_to_line_l2706_270680


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_range_l2706_270658

/-- An isosceles triangle with perimeter 16 -/
structure IsoscelesTriangle where
  x : ℝ  -- base length
  y : ℝ  -- leg length
  perimeter_eq : x + 2*y = 16
  leg_eq : y = -1/2 * x + 8

/-- The range of the base length x in an isosceles triangle -/
theorem isosceles_triangle_base_range (t : IsoscelesTriangle) : 0 < t.x ∧ t.x < 8 := by
  sorry

#check isosceles_triangle_base_range

end NUMINAMATH_CALUDE_isosceles_triangle_base_range_l2706_270658


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l2706_270640

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^1200 - 1) (2^1230 - 1) = 2^30 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l2706_270640


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l2706_270693

theorem polynomial_product_expansion (x : ℝ) : 
  (7 * x^2 + 5 * x - 3) * (3 * x^3 + 2 * x^2 - x + 4) = 
  21 * x^5 + 29 * x^4 - 6 * x^3 + 17 * x^2 + 23 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l2706_270693


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2706_270600

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 + i) / (3 - i) = (1 + 2*i) / 5 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2706_270600


namespace NUMINAMATH_CALUDE_tangent_circle_circumference_l2706_270670

-- Define the geometric configuration
structure GeometricConfig where
  -- Centers of the arcs
  A : Point
  B : Point
  -- Points on the arcs
  C : Point
  -- Radii of the arcs
  r1 : ℝ
  r2 : ℝ
  -- Angle subtended by arc AC at center B
  angle_ACB : ℝ
  -- Length of arc BC
  length_BC : ℝ
  -- Radius of the tangent circle
  r : ℝ

-- State the theorem
theorem tangent_circle_circumference (config : GeometricConfig) 
  (h1 : config.angle_ACB = 75 * π / 180)
  (h2 : config.length_BC = 18)
  (h3 : config.r1 = 54 / π)
  (h4 : config.r2 = 216 / (5 * π))
  (h5 : config.r = 30 / π) : 
  2 * π * config.r = 60 := by
  sorry


end NUMINAMATH_CALUDE_tangent_circle_circumference_l2706_270670


namespace NUMINAMATH_CALUDE_range_of_f_l2706_270643

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the domain
def domain : Set ℝ := Set.Icc 1 5

-- Theorem statement
theorem range_of_f :
  Set.range (fun x => f x) ∩ (Set.image f domain) = Set.Ico 2 11 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2706_270643
