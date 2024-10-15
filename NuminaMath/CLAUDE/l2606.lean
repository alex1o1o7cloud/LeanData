import Mathlib

namespace NUMINAMATH_CALUDE_hens_and_cows_l2606_260641

/-- Given a total number of animals and feet, calculates the number of hens -/
def number_of_hens (total_animals : ℕ) (total_feet : ℕ) : ℕ :=
  total_animals - (total_feet - 2 * total_animals) / 2

/-- Theorem stating that given 50 animals with 140 feet, where hens have 2 feet
    and cows have 4 feet, the number of hens is 30 -/
theorem hens_and_cows (total_animals : ℕ) (total_feet : ℕ) 
  (h1 : total_animals = 50)
  (h2 : total_feet = 140) :
  number_of_hens total_animals total_feet = 30 := by
  sorry

#eval number_of_hens 50 140  -- Should output 30

end NUMINAMATH_CALUDE_hens_and_cows_l2606_260641


namespace NUMINAMATH_CALUDE_expected_marbles_theorem_l2606_260631

/-- The expected number of marbles drawn until the special marble is picked, given that no ugly marbles were drawn -/
def expected_marbles_drawn (blue_marbles : ℕ) (ugly_marbles : ℕ) (special_marbles : ℕ) : ℚ :=
  let total_marbles := blue_marbles + ugly_marbles + special_marbles
  let prob_blue := blue_marbles / total_marbles
  let prob_special := special_marbles / total_marbles
  let expected_draws := (prob_blue / (1 - prob_blue)) / prob_special
  expected_draws

/-- Theorem stating that the expected number of marbles drawn is 20/11 -/
theorem expected_marbles_theorem :
  expected_marbles_drawn 9 10 1 = 20 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expected_marbles_theorem_l2606_260631


namespace NUMINAMATH_CALUDE_common_chord_length_l2606_260690

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 3 = 0

-- Define the common chord line
def common_chord_line (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    common_chord_line A.1 A.2 ∧ common_chord_line B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_common_chord_length_l2606_260690


namespace NUMINAMATH_CALUDE_jack_plates_problem_l2606_260648

theorem jack_plates_problem (flower_initial : ℕ) (checked : ℕ) (total_final : ℕ) :
  flower_initial = 4 →
  total_final = 27 →
  total_final = (flower_initial - 1) + checked + 2 * checked →
  checked = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_plates_problem_l2606_260648


namespace NUMINAMATH_CALUDE_solve_iterated_f_equation_l2606_260659

def f (x : ℝ) : ℝ := x^2 + 6*x + 6

def iterate_f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem solve_iterated_f_equation :
  ∃ x : ℝ, iterate_f 2017 x = 2017 ∧
  x = -3 + (2020 : ℝ)^(1/(2^2017)) ∨
  x = -3 - (2020 : ℝ)^(1/(2^2017)) :=
sorry

end NUMINAMATH_CALUDE_solve_iterated_f_equation_l2606_260659


namespace NUMINAMATH_CALUDE_reciprocal_sum_equals_two_l2606_260681

theorem reciprocal_sum_equals_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_sum : x + y = 2) (h_prod : x * y = 1) : 
  1 / x + 1 / y = 2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equals_two_l2606_260681


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2606_260602

def A : Set ℕ := {2, 3, 5, 7}
def B : Set ℕ := {1, 2, 3, 5, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2606_260602


namespace NUMINAMATH_CALUDE_range_of_m_given_one_root_l2606_260629

/-- The function f(x) defined in terms of x and m -/
def f (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

/-- The property that f has exactly one root in [0, 1] -/
def has_one_root_in_unit_interval (m : ℝ) : Prop :=
  ∃! x, x ∈ Set.Icc 0 1 ∧ f x m = 0

/-- The theorem stating the range of m given the condition -/
theorem range_of_m_given_one_root :
  ∀ m, has_one_root_in_unit_interval m → m ∈ Set.Icc (-1) 0 ∪ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_given_one_root_l2606_260629


namespace NUMINAMATH_CALUDE_sweet_cookies_eaten_l2606_260624

theorem sweet_cookies_eaten (initial_sweet : ℕ) (final_sweet : ℕ) (eaten_sweet : ℕ) :
  initial_sweet = final_sweet + eaten_sweet →
  eaten_sweet = initial_sweet - final_sweet :=
by sorry

end NUMINAMATH_CALUDE_sweet_cookies_eaten_l2606_260624


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l2606_260658

/-- Given a triangle XYZ where the measure of ∠X is 78 degrees and 
    the measure of ∠Y is 14 degrees less than four times the measure of ∠Z,
    prove that the measure of ∠Z is 23.2 degrees. -/
theorem angle_measure_in_triangle (X Y Z : ℝ) 
  (h1 : X = 78)
  (h2 : Y = 4 * Z - 14)
  (h3 : X + Y + Z = 180) :
  Z = 23.2 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l2606_260658


namespace NUMINAMATH_CALUDE_correct_average_calculation_l2606_260685

theorem correct_average_calculation (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 20 →
  incorrect_num = 26 →
  correct_num = 86 →
  (n : ℚ) * incorrect_avg - incorrect_num + correct_num = n * 26 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l2606_260685


namespace NUMINAMATH_CALUDE_cannon_probability_l2606_260630

theorem cannon_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.5) (h2 : p2 = 0.8) (h3 : p3 = 0.7) : 
  p1 * p2 * p3 = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_cannon_probability_l2606_260630


namespace NUMINAMATH_CALUDE_smallest_b_for_g_nested_equals_g_l2606_260600

def g (x : ℤ) : ℤ :=
  if x % 15 = 0 then x / 15
  else if x % 3 = 0 then 5 * x
  else if x % 5 = 0 then 3 * x
  else x + 5

def g_nested (b : ℕ) (x : ℤ) : ℤ :=
  match b with
  | 0 => x
  | n + 1 => g (g_nested n x)

theorem smallest_b_for_g_nested_equals_g :
  ∀ b : ℕ, b > 1 → g_nested b 2 = g 2 → b ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_g_nested_equals_g_l2606_260600


namespace NUMINAMATH_CALUDE_floor_sum_example_l2606_260645

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l2606_260645


namespace NUMINAMATH_CALUDE_work_ratio_is_three_to_eight_l2606_260655

/-- Represents a worker's weekly schedule and earnings -/
structure WorkerSchedule where
  days_per_week : ℕ
  weekly_earnings : ℚ
  daily_salary : ℚ

/-- Calculates the ratio of time worked on the last three days to the first four days -/
def work_ratio (w : WorkerSchedule) : ℚ × ℚ :=
  let last_three_days_earnings := w.weekly_earnings - (4 * w.daily_salary)
  let last_three_days_time := last_three_days_earnings / w.daily_salary
  let first_four_days_time := 4
  (last_three_days_time * 2, first_four_days_time * 2)

/-- Theorem stating the work ratio for a specific worker schedule -/
theorem work_ratio_is_three_to_eight (w : WorkerSchedule) 
  (h1 : w.days_per_week = 7)
  (h2 : w.weekly_earnings = 55)
  (h3 : w.daily_salary = 10) :
  work_ratio w = (3, 8) := by
  sorry

#eval work_ratio ⟨7, 55, 10⟩

end NUMINAMATH_CALUDE_work_ratio_is_three_to_eight_l2606_260655


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_factors_of_125_l2606_260683

theorem smallest_sum_of_three_factors_of_125 :
  ∀ a b c : ℕ+,
  a * b * c = 125 →
  ∀ x y z : ℕ+,
  x * y * z = 125 →
  a + b + c ≤ x + y + z →
  a + b + c = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_factors_of_125_l2606_260683


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2606_260640

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 6) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧ 
  (∀ m : ℕ, m ≥ 100 ∧ m ≤ 999 ∧ (m + 6) % 9 = 0 ∧ (m - 9) % 6 = 0 → m ≥ n) ∧
  n = 111 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2606_260640


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2606_260633

/-- Given an arithmetic sequence with the first four terms as specified,
    prove that the fifth term is x^2 - (21x)/5 -/
theorem arithmetic_sequence_fifth_term (x y : ℝ) :
  let a₁ := x^2 + 3*y
  let a₂ := (x - 2) * y
  let a₃ := x^2 - y
  let a₄ := x / (y + 1)
  -- The sequence is arithmetic
  (a₂ - a₁ = a₃ - a₂) ∧ (a₃ - a₂ = a₄ - a₃) →
  -- The fifth term
  ∃ (a₅ : ℝ), a₅ = x^2 - (21 * x) / 5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2606_260633


namespace NUMINAMATH_CALUDE_negation_of_implication_l2606_260657

theorem negation_of_implication (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2606_260657


namespace NUMINAMATH_CALUDE_sequence_limit_existence_l2606_260643

theorem sequence_limit_existence (a : ℕ → ℝ) (h : ∀ n, 0 ≤ a n ∧ a n ≤ 1) :
  ∃ (n : ℕ → ℕ) (A : ℝ),
    (∀ i j, i < j → n i < n j) ∧
    (∀ ε > 0, ∃ N, ∀ i j, i ≠ j → i > N → j > N → |a (n i + n j) - A| < ε) := by
  sorry

end NUMINAMATH_CALUDE_sequence_limit_existence_l2606_260643


namespace NUMINAMATH_CALUDE_percentage_problem_l2606_260689

theorem percentage_problem (x : ℝ) : 
  (0.15 * 25) + (x / 100 * 45) = 9.15 ↔ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2606_260689


namespace NUMINAMATH_CALUDE_sqrt_negative_product_equals_two_sqrt_two_l2606_260634

theorem sqrt_negative_product_equals_two_sqrt_two :
  Real.sqrt ((-4) * (-2)) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_negative_product_equals_two_sqrt_two_l2606_260634


namespace NUMINAMATH_CALUDE_line_tangent_to_curve_l2606_260601

/-- Tangency condition for a line to a curve -/
theorem line_tangent_to_curve
  {m n u v : ℝ}
  (hm : m > 1)
  (hn : m⁻¹ + n⁻¹ = 1) :
  (∀ x y : ℝ, u * x + v * y = 1 →
    (∃ a : ℝ, x^m + y^m = a) →
    (∀ δ ε : ℝ, δ ≠ 0 ∨ ε ≠ 0 →
      (x + δ)^m + (y + ε)^m > a)) ↔
  u^n + v^n = 1 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_curve_l2606_260601


namespace NUMINAMATH_CALUDE_even_integer_sequence_sum_l2606_260660

theorem even_integer_sequence_sum : 
  ∀ (a b c d : ℤ),
  (∃ (k₁ k₂ k₃ k₄ : ℤ), a = 2 * k₁ ∧ b = 2 * k₂ ∧ c = 2 * k₃ ∧ d = 2 * k₄) →
  (0 < a) →
  (a < b) →
  (b < c) →
  (c < d) →
  (d - a = 90) →
  (∃ (r : ℤ), c - b = b - a ∧ b - a = r) →
  (∃ (q : ℚ), c / b = d / c ∧ c / b = q) →
  (a + b + c + d = 194) :=
by sorry

end NUMINAMATH_CALUDE_even_integer_sequence_sum_l2606_260660


namespace NUMINAMATH_CALUDE_concert_attendance_l2606_260607

/-- The number of buses used for the concert -/
def num_buses : ℕ := 12

/-- The number of students each bus can carry -/
def students_per_bus : ℕ := 57

/-- The total number of students who went to the concert -/
def total_students : ℕ := num_buses * students_per_bus

theorem concert_attendance : total_students = 684 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l2606_260607


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2606_260682

theorem complex_product_theorem : 
  let z : ℂ := Complex.exp (Complex.I * (3 * Real.pi / 11))
  (3 * z + z^3) * (3 * z^3 + z^9) * (3 * z^5 + z^15) * 
  (3 * z^7 + z^21) * (3 * z^9 + z^27) * (3 * z^11 + z^33) = 2197 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2606_260682


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l2606_260638

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l2606_260638


namespace NUMINAMATH_CALUDE_transformation_result_l2606_260677

/-- Rotates a point (x, y) 180° counterclockwise around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2*h - x, 2*k - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqualX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem transformation_result (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectAboutYEqualX rotated.1 rotated.2
  final = (1, -4) → b - a = -3 := by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l2606_260677


namespace NUMINAMATH_CALUDE_house_price_proof_l2606_260623

theorem house_price_proof (price_first : ℝ) (price_second : ℝ) : 
  price_second = 2 * price_first →
  price_first + price_second = 600000 →
  price_first = 200000 := by
sorry

end NUMINAMATH_CALUDE_house_price_proof_l2606_260623


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2606_260696

theorem min_value_of_expression (c d : ℤ) (h : c^2 > d^2) :
  (((c^2 + d^2) / (c^2 - d^2)) + ((c^2 - d^2) / (c^2 + d^2)) : ℚ) ≥ 2 ∧
  ∃ (c d : ℤ), c^2 > d^2 ∧ ((c^2 + d^2) / (c^2 - d^2)) + ((c^2 - d^2) / (c^2 + d^2)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2606_260696


namespace NUMINAMATH_CALUDE_inequality_proof_l2606_260680

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2606_260680


namespace NUMINAMATH_CALUDE_quadratic_positive_combination_l2606_260615

/-- A quadratic polynomial -/
def QuadraticPolynomial := ℝ → ℝ

/-- Predicate to check if a function is negative on an interval -/
def NegativeOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → f x < 0

/-- Predicate to check if two intervals are non-overlapping -/
def NonOverlappingIntervals (a b c d : ℝ) : Prop :=
  b < c ∨ d < a

/-- Main theorem statement -/
theorem quadratic_positive_combination
  (f g : QuadraticPolynomial)
  (a b c d : ℝ)
  (hf : NegativeOnInterval f a b)
  (hg : NegativeOnInterval g c d)
  (h_non_overlap : NonOverlappingIntervals a b c d) :
  ∃ (α β : ℝ), α > 0 ∧ β > 0 ∧ ∀ x, α * f x + β * g x > 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_positive_combination_l2606_260615


namespace NUMINAMATH_CALUDE_range_of_a_l2606_260687

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / (a * Real.log (x + 1))

/-- The function g(x) defined in the problem -/
def g (x : ℝ) : ℝ := x^2 * (x - 1)^2

/-- The helper function h(x) used in the proof -/
noncomputable def h (x : ℝ) : ℝ := x^2 / Real.log x

theorem range_of_a :
  ∃ (a : ℝ), ∀ (x₁ x₂ : ℝ),
    (Real.exp (1/4) - 1 < x₁ ∧ x₁ < Real.exp 1 - 1) →
    x₂ < 0 →
    x₂ = -x₁ →
    (f a x₁) * (-x₁) + (g x₂) * x₁ = 0 →
    2 * Real.exp 1 ≤ a ∧ a < Real.exp 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2606_260687


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2606_260639

theorem complex_equation_solution (a : ℝ) (i : ℂ) 
  (hi : i * i = -1) 
  (h : (1 + a * i) * i = 3 + i) : 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2606_260639


namespace NUMINAMATH_CALUDE_percentage_calculation_l2606_260699

theorem percentage_calculation (P : ℝ) : 
  0.15 * 0.30 * P * 5600 = 126 → P = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2606_260699


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2606_260620

/-- The total surface area of a right cylinder with height 8 and radius 3 is 66π -/
theorem cylinder_surface_area :
  let h : ℝ := 8
  let r : ℝ := 3
  let lateral_area := 2 * π * r * h
  let base_area := π * r^2
  let total_area := lateral_area + 2 * base_area
  total_area = 66 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2606_260620


namespace NUMINAMATH_CALUDE_empty_set_implies_a_zero_l2606_260654

theorem empty_set_implies_a_zero (a : ℝ) : (∀ x : ℝ, ax + 2 ≠ 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_set_implies_a_zero_l2606_260654


namespace NUMINAMATH_CALUDE_class_size_proof_l2606_260649

/-- Represents the number of pupils in a class. -/
def num_pupils : ℕ := 56

/-- Represents the wrongly entered mark. -/
def wrong_mark : ℕ := 73

/-- Represents the correct mark. -/
def correct_mark : ℕ := 45

/-- Represents the increase in average marks due to the error. -/
def avg_increase : ℚ := 1/2

theorem class_size_proof :
  (wrong_mark - correct_mark : ℚ) / num_pupils = avg_increase :=
sorry

end NUMINAMATH_CALUDE_class_size_proof_l2606_260649


namespace NUMINAMATH_CALUDE_placement_count_no_restriction_placement_count_with_restriction_l2606_260698

/-- The number of booths in the exhibition room -/
def total_booths : ℕ := 9

/-- The number of exhibits to be displayed -/
def num_exhibits : ℕ := 3

/-- Calculates the number of ways to place exhibits under the given conditions -/
def calculate_placements (max_distance : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of placement options without distance restriction -/
theorem placement_count_no_restriction : calculate_placements total_booths = 60 :=
  sorry

/-- Theorem stating the number of placement options with distance restriction -/
theorem placement_count_with_restriction : calculate_placements 2 = 48 :=
  sorry

end NUMINAMATH_CALUDE_placement_count_no_restriction_placement_count_with_restriction_l2606_260698


namespace NUMINAMATH_CALUDE_wood_sawing_problem_l2606_260692

theorem wood_sawing_problem (original_length final_length : ℝ) 
  (h1 : original_length = 8.9)
  (h2 : final_length = 6.6) :
  original_length - final_length = 2.3 := by
  sorry

end NUMINAMATH_CALUDE_wood_sawing_problem_l2606_260692


namespace NUMINAMATH_CALUDE_three_five_power_sum_l2606_260647

theorem three_five_power_sum (x y : ℕ+) (h : 3^(x.val) * 5^(y.val) = 225) : x.val + y.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_three_five_power_sum_l2606_260647


namespace NUMINAMATH_CALUDE_snow_probability_first_week_l2606_260622

def probability_of_snow (days : ℕ) (daily_prob : ℚ) : ℚ :=
  1 - (1 - daily_prob) ^ days

theorem snow_probability_first_week :
  let prob_first_four := probability_of_snow 4 (1/4)
  let prob_next_three := probability_of_snow 3 (1/3)
  let total_prob := 1 - (1 - prob_first_four) * (1 - prob_next_three)
  total_prob = 29/32 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_first_week_l2606_260622


namespace NUMINAMATH_CALUDE_sqrt_equation_sum_l2606_260671

theorem sqrt_equation_sum (y : ℝ) (d e f : ℕ+) : 
  y = Real.sqrt ((Real.sqrt 73 / 3) + (5 / 3)) →
  y^52 = 3*y^50 + 10*y^48 + 25*y^46 - y^26 + d*y^22 + e*y^20 + f*y^18 →
  d + e + f = 184 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_sum_l2606_260671


namespace NUMINAMATH_CALUDE_difference_of_squares_l2606_260618

theorem difference_of_squares : 303^2 - 297^2 = 3600 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2606_260618


namespace NUMINAMATH_CALUDE_sandra_current_age_l2606_260661

/-- Sandra's current age -/
def sandra_age : ℕ := 36

/-- Sandra's son's current age -/
def son_age : ℕ := 14

/-- Theorem stating Sandra's current age based on the given conditions -/
theorem sandra_current_age : 
  (sandra_age - 3 = 3 * (son_age - 3)) → sandra_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_sandra_current_age_l2606_260661


namespace NUMINAMATH_CALUDE_last_digit_of_N_l2606_260674

theorem last_digit_of_N (total_coins : ℕ) (h : total_coins = 3080) : 
  ∃ N : ℕ, (N * (N + 1)) / 2 = total_coins ∧ N % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_N_l2606_260674


namespace NUMINAMATH_CALUDE_positive_decreasing_function_l2606_260625

/-- A function f: ℝ → ℝ is decreasing if for all x < y, f(x) > f(y) -/
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The second derivative of a function f: ℝ → ℝ -/
def SecondDerivative (f : ℝ → ℝ) : ℝ → ℝ := sorry

theorem positive_decreasing_function
  (f : ℝ → ℝ)
  (h_decreasing : Decreasing f)
  (h_second_derivative : ∀ x, f x / SecondDerivative f x < 1 - x) :
  ∀ x, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_decreasing_function_l2606_260625


namespace NUMINAMATH_CALUDE_inequality_condition_neither_sufficient_nor_necessary_l2606_260679

theorem inequality_condition_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, (a > b → 1/a < 1/b) → a > b) ∧
  ¬(∀ a b : ℝ, a > b → (1/a < 1/b)) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_neither_sufficient_nor_necessary_l2606_260679


namespace NUMINAMATH_CALUDE_number_of_rooms_l2606_260611

/-- Calculates the number of equal-sized rooms given the original dimensions,
    increase in dimensions, and total area. --/
def calculate_rooms (original_length original_width increase_dim total_area : ℕ) : ℕ :=
  let new_length := original_length + increase_dim
  let new_width := original_width + increase_dim
  let room_area := new_length * new_width
  let double_room_area := 2 * room_area
  let equal_rooms_area := total_area - double_room_area
  equal_rooms_area / room_area

/-- Theorem stating that the number of equal-sized rooms is 4 --/
theorem number_of_rooms : calculate_rooms 13 18 2 1800 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_rooms_l2606_260611


namespace NUMINAMATH_CALUDE_unique_cube_difference_l2606_260695

theorem unique_cube_difference (n : ℕ+) : 
  (∃ x y : ℕ+, (837 + n : ℕ) = y^3 ∧ (837 - n : ℕ) = x^3) ↔ n = 494 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_difference_l2606_260695


namespace NUMINAMATH_CALUDE_unique_valid_number_l2606_260669

def shares_one_digit (n m : Nat) : Bool :=
  let n_digits := n.digits 10
  let m_digits := m.digits 10
  (n_digits.filter (fun d => m_digits.contains d)).length = 1

def is_valid_number (n : Nat) : Bool :=
  n ≥ 100 ∧ n < 1000 ∧
  shares_one_digit n 543 ∧
  shares_one_digit n 142 ∧
  shares_one_digit n 562

theorem unique_valid_number : 
  ∀ n : Nat, is_valid_number n ↔ n = 163 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2606_260669


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2606_260637

/-- The length of the real axis of the hyperbola 2x^2 - y^2 = 8 is 4 -/
theorem hyperbola_real_axis_length :
  ∃ (a : ℝ), a > 0 ∧ (∀ x y : ℝ, 2 * x^2 - y^2 = 8 → x^2 / (2 * a^2) - y^2 / (2 * a^2) = 1) ∧ 2 * a = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2606_260637


namespace NUMINAMATH_CALUDE_ernesto_age_proof_l2606_260688

/-- Ernesto's current age -/
def ernesto_age : ℕ := 11

/-- Jayden's current age -/
def jayden_age : ℕ := 4

/-- The number of years in the future when the age comparison is made -/
def years_future : ℕ := 3

theorem ernesto_age_proof :
  ernesto_age = 11 ∧
  jayden_age = 4 ∧
  jayden_age + years_future = (ernesto_age + years_future) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ernesto_age_proof_l2606_260688


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2606_260663

/-- The length of the minor axis of the ellipse 9x^2 + y^2 = 36 is 4 -/
theorem ellipse_minor_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 36}
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | (x^2 / a^2) + (y^2 / b^2) = 1} ∧
    2 * min a b = 4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2606_260663


namespace NUMINAMATH_CALUDE_withdraw_300_from_two_banks_in_20_bills_l2606_260636

/-- Calculates the number of bills received when withdrawing from two banks -/
def number_of_bills (withdrawal_per_bank : ℕ) (bill_denomination : ℕ) : ℕ :=
  (2 * withdrawal_per_bank) / bill_denomination

/-- Theorem: Withdrawing $300 from each of two banks in $20 bills results in 30 bills -/
theorem withdraw_300_from_two_banks_in_20_bills : 
  number_of_bills 300 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_withdraw_300_from_two_banks_in_20_bills_l2606_260636


namespace NUMINAMATH_CALUDE_donation_in_scientific_notation_l2606_260651

/-- Definition of a billion in the context of this problem -/
def billion : ℕ := 10^8

/-- The donation amount in yuan -/
def donation : ℚ := 2.94 * billion

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℚ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that 2.94 billion yuan is equal to 2.94 × 10^8 in scientific notation -/
theorem donation_in_scientific_notation :
  ∃ (sn : ScientificNotation), (sn.coefficient * (10 : ℚ)^sn.exponent) = donation ∧
    sn.coefficient = 2.94 ∧ sn.exponent = 8 := by sorry

end NUMINAMATH_CALUDE_donation_in_scientific_notation_l2606_260651


namespace NUMINAMATH_CALUDE_sinusoidal_function_omega_l2606_260613

/-- Given a sinusoidal function y = 2sin(ωx + π/6) with ω > 0,
    if the distance between adjacent symmetry axes is π/2,
    then ω = 2. -/
theorem sinusoidal_function_omega (ω : ℝ) (h1 : ω > 0) :
  (∀ x : ℝ, 2 * Real.sin (ω * x + π / 6) = 2 * Real.sin (ω * (x + π / (2 * ω)) + π / 6)) →
  ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_omega_l2606_260613


namespace NUMINAMATH_CALUDE_tyler_meal_choices_l2606_260650

def meat_options : ℕ := 3
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 4
def drink_options : ℕ := 4
def vegetables_to_choose : ℕ := 3

def number_of_meals : ℕ := meat_options * Nat.choose vegetable_options vegetables_to_choose * dessert_options * drink_options

theorem tyler_meal_choices : number_of_meals = 480 := by
  sorry

end NUMINAMATH_CALUDE_tyler_meal_choices_l2606_260650


namespace NUMINAMATH_CALUDE_product_digits_sum_l2606_260667

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a number in base-7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := 
  toBase7 (toBase10 a * toBase10 b)

theorem product_digits_sum :
  sumOfDigitsBase7 (multiplyBase7 24 30) = 6 := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_l2606_260667


namespace NUMINAMATH_CALUDE_symmetry_of_functions_l2606_260691

theorem symmetry_of_functions (f : ℝ → ℝ) : 
  ∀ x y : ℝ, f (1 - x) = y ↔ f (x - 1) = y :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_functions_l2606_260691


namespace NUMINAMATH_CALUDE_inequality_proof_l2606_260666

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_eq_four : a + b + c + d = 4) :
  a * Real.sqrt (3*a + b + c) + b * Real.sqrt (3*b + c + d) + 
  c * Real.sqrt (3*c + d + a) + d * Real.sqrt (3*d + a + b) ≥ 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2606_260666


namespace NUMINAMATH_CALUDE_sector_central_angle_l2606_260603

/-- Given a sector with radius r and perimeter 3r, its central angle is 1. -/
theorem sector_central_angle (r : ℝ) (h : r > 0) :
  (∃ (l : ℝ), l > 0 ∧ 2 * r + l = 3 * r) →
  (∃ (α : ℝ), α = l / r ∧ α = 1) :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2606_260603


namespace NUMINAMATH_CALUDE_prob_unit_apart_value_l2606_260610

/-- A rectangle with 10 points spaced at unit intervals on corners and edge midpoints -/
structure UnitRectangle :=
  (width : ℕ)
  (height : ℕ)
  (total_points : ℕ)
  (h_width : width = 5)
  (h_height : height = 2)
  (h_total_points : total_points = 10)

/-- The number of pairs of points that are exactly one unit apart -/
def unit_apart_pairs (r : UnitRectangle) : ℕ := 13

/-- The total number of ways to choose two points from the rectangle -/
def total_pairs (r : UnitRectangle) : ℕ := r.total_points.choose 2

/-- The probability of selecting two points that are exactly one unit apart -/
def prob_unit_apart (r : UnitRectangle) : ℚ :=
  (unit_apart_pairs r : ℚ) / (total_pairs r : ℚ)

theorem prob_unit_apart_value (r : UnitRectangle) :
  prob_unit_apart r = 13 / 45 := by sorry

end NUMINAMATH_CALUDE_prob_unit_apart_value_l2606_260610


namespace NUMINAMATH_CALUDE_escalator_length_proof_l2606_260617

/-- The length of an escalator in feet. -/
def escalator_length : ℝ := 160

/-- The speed of the escalator in feet per second. -/
def escalator_speed : ℝ := 8

/-- The walking speed of a person on the escalator in feet per second. -/
def person_speed : ℝ := 2

/-- The time taken by the person to cover the entire length of the escalator in seconds. -/
def time_taken : ℝ := 16

/-- Theorem stating that the length of the escalator is 160 feet, given the conditions. -/
theorem escalator_length_proof :
  escalator_length = (escalator_speed + person_speed) * time_taken :=
by sorry

end NUMINAMATH_CALUDE_escalator_length_proof_l2606_260617


namespace NUMINAMATH_CALUDE_herrings_caught_l2606_260676

/-- Given the total number of fish caught and the number of pikes and sturgeons,
    calculate the number of herrings caught. -/
theorem herrings_caught (total : ℕ) (pikes : ℕ) (sturgeons : ℕ) 
  (h1 : total = 145) (h2 : pikes = 30) (h3 : sturgeons = 40) :
  total - pikes - sturgeons = 75 := by
  sorry

end NUMINAMATH_CALUDE_herrings_caught_l2606_260676


namespace NUMINAMATH_CALUDE_cube_geometric_shapes_l2606_260672

structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

def isRectangle (points : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  sorry

def isParallelogramNotRectangle (points : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  sorry

def isRightAngledTetrahedron (points : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  sorry

def fourVertices (c : Cube) : Type :=
  {v : Fin 4 → Fin 8 // ∀ i j, i ≠ j → v i ≠ v j}

theorem cube_geometric_shapes (c : Cube) :
  ∃ (v : fourVertices c),
    isRectangle (fun i => c.vertices (v.val i)) ∧
    isParallelogramNotRectangle (fun i => c.vertices (v.val i)) ∧
    isRightAngledTetrahedron (fun i => c.vertices (v.val i)) ∧
    ¬∃ (w : fourVertices c),
      (isRectangle (fun i => c.vertices (w.val i)) ∧
       isParallelogramNotRectangle (fun i => c.vertices (w.val i)) ∧
       isRightAngledTetrahedron (fun i => c.vertices (w.val i)) ∧
       (isRectangle (fun i => c.vertices (w.val i)) ≠
        isRectangle (fun i => c.vertices (v.val i)) ∨
        isParallelogramNotRectangle (fun i => c.vertices (w.val i)) ≠
        isParallelogramNotRectangle (fun i => c.vertices (v.val i)) ∨
        isRightAngledTetrahedron (fun i => c.vertices (w.val i)) ≠
        isRightAngledTetrahedron (fun i => c.vertices (v.val i)))) :=
  sorry

end NUMINAMATH_CALUDE_cube_geometric_shapes_l2606_260672


namespace NUMINAMATH_CALUDE_event_occurrence_limit_l2606_260652

theorem event_occurrence_limit (ε : ℝ) (hε : 0 < ε) :
  ∀ δ > 0, ∃ N : ℕ, ∀ n ≥ N, 1 - (1 - ε)^n > 1 - δ :=
sorry

end NUMINAMATH_CALUDE_event_occurrence_limit_l2606_260652


namespace NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l2606_260662

theorem largest_number_from_hcf_lcm_factors (a b : ℕ+) 
  (hcf_ab : Nat.gcd a b = 50)
  (lcm_factor1 : ∃ k : ℕ+, Nat.lcm a b = 50 * 11 * k)
  (lcm_factor2 : ∃ k : ℕ+, Nat.lcm a b = 50 * 12 * k) :
  max a b = 600 := by
sorry

end NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l2606_260662


namespace NUMINAMATH_CALUDE_second_round_votes_l2606_260670

/-- Represents the total number of votes in the second round of an election. -/
def total_votes : ℕ := sorry

/-- Represents the percentage of votes received by Candidate A in the second round. -/
def candidate_a_percentage : ℚ := 50 / 100

/-- Represents the percentage of votes received by Candidate B in the second round. -/
def candidate_b_percentage : ℚ := 30 / 100

/-- Represents the percentage of votes received by Candidate C in the second round. -/
def candidate_c_percentage : ℚ := 20 / 100

/-- Represents the majority of votes by which Candidate A won over Candidate B. -/
def majority : ℕ := 1350

theorem second_round_votes : 
  (candidate_a_percentage - candidate_b_percentage) * total_votes = majority ∧
  total_votes = 6750 := by sorry

end NUMINAMATH_CALUDE_second_round_votes_l2606_260670


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l2606_260697

theorem polynomial_factor_implies_coefficients 
  (a b : ℚ) 
  (h : ∃ (c d : ℚ), ax^4 + bx^3 + 40*x^2 - 20*x + 10 = (5*x^2 - 3*x + 2)*(c*x^2 + d*x + 5)) :
  a = 25/4 ∧ b = -65/4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l2606_260697


namespace NUMINAMATH_CALUDE_number_subtracted_from_x_l2606_260632

-- Define the problem conditions
def problem_conditions (x y z a : ℤ) : Prop :=
  ((x - a) * (y - 5) * (z - 2) = 1000) ∧
  (∀ (x' y' z' : ℤ), ((x' - a) * (y' - 5) * (z' - 2) = 1000) → (x + y + z ≤ x' + y' + z')) ∧
  (x + y + z = 7)

-- State the theorem
theorem number_subtracted_from_x :
  ∃ (x y z a : ℤ), problem_conditions x y z a ∧ a = -30 :=
sorry

end NUMINAMATH_CALUDE_number_subtracted_from_x_l2606_260632


namespace NUMINAMATH_CALUDE_complement_of_A_l2606_260621

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 2}

-- State the theorem
theorem complement_of_A : Set.compl A = {x : ℝ | x < 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l2606_260621


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l2606_260614

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- State the theorem
theorem opposite_of_negative_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l2606_260614


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2606_260619

/-- An arithmetic sequence with positive common ratio -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : q > 0
  h_arithmetic : ∀ n : ℕ, a (n + 1) = a n * q

/-- The problem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 2 * seq.a 6 = 8 * seq.a 4)
  (h2 : seq.a 2 = 2) :
  seq.a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2606_260619


namespace NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l2606_260693

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 7

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x + y = 4

-- State the theorem
theorem shortest_distance_circle_to_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - Real.sqrt 7 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ → C₂ x₂ y₂ →
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l2606_260693


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2606_260673

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2606_260673


namespace NUMINAMATH_CALUDE_percentage_comparison_l2606_260616

theorem percentage_comparison : 
  (0.85 * 250 - 0.75 * 180) < 0.90 * 320 := by
  sorry

end NUMINAMATH_CALUDE_percentage_comparison_l2606_260616


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_l2606_260609

/-- The coefficient of a monomial is the numerical factor multiplied by the variables. -/
def coefficient (m : ℝ) (x y : ℝ) : ℝ := m

/-- For the monomial -2π * x^2 * y, prove that its coefficient is -2π. -/
theorem coefficient_of_monomial :
  coefficient (-2 * Real.pi) (x ^ 2) y = -2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_l2606_260609


namespace NUMINAMATH_CALUDE_line_through_point_l2606_260642

theorem line_through_point (k : ℚ) :
  (1 - 3 * k * (1/2) = 10 * 3) ↔ (k = -58/3) := by sorry

end NUMINAMATH_CALUDE_line_through_point_l2606_260642


namespace NUMINAMATH_CALUDE_angle_ABH_in_regular_octagon_l2606_260675

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The measure of an angle in degrees -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

theorem angle_ABH_in_regular_octagon (ABCDEFGH : RegularOctagon) :
  let vertices := ABCDEFGH.vertices
  angle_measure (vertices 0) (vertices 1) (vertices 7) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABH_in_regular_octagon_l2606_260675


namespace NUMINAMATH_CALUDE_value_of_q_l2606_260608

theorem value_of_q (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : (1 / p) + (1 / q) = 1) 
  (h4 : p * q = 16 / 3) : 
  q = 4 := by
sorry

end NUMINAMATH_CALUDE_value_of_q_l2606_260608


namespace NUMINAMATH_CALUDE_rectangular_field_width_l2606_260646

theorem rectangular_field_width (width length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 384 →
  width = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l2606_260646


namespace NUMINAMATH_CALUDE_other_focus_coordinates_l2606_260664

/-- An ellipse with specific properties -/
structure Ellipse where
  /-- The ellipse is tangent to the y-axis at the origin -/
  tangent_at_origin : True
  /-- The length of the major axis -/
  major_axis_length : ℝ
  /-- The coordinates of one focus -/
  focus1 : ℝ × ℝ

/-- Theorem: Given an ellipse with specific properties, the other focus has coordinates (-3, -4) -/
theorem other_focus_coordinates (e : Ellipse) 
  (h1 : e.major_axis_length = 20)
  (h2 : e.focus1 = (3, 4)) :
  ∃ (other_focus : ℝ × ℝ), other_focus = (-3, -4) := by
  sorry

end NUMINAMATH_CALUDE_other_focus_coordinates_l2606_260664


namespace NUMINAMATH_CALUDE_perimeter_difference_zero_l2606_260644

/-- Perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ := 2 * (length + width)

/-- Perimeter of Figure 1 --/
def figure1Perimeter : ℕ := rectanglePerimeter 6 1 + 4

/-- Perimeter of Figure 2 --/
def figure2Perimeter : ℕ := rectanglePerimeter 7 2

theorem perimeter_difference_zero : figure1Perimeter = figure2Perimeter := by
  sorry

#eval figure1Perimeter
#eval figure2Perimeter

end NUMINAMATH_CALUDE_perimeter_difference_zero_l2606_260644


namespace NUMINAMATH_CALUDE_tax_deduction_proof_l2606_260686

/-- Represents the hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℝ := 0.025

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℝ := 100

/-- Calculates the tax deduction in cents -/
def tax_deduction_cents : ℝ := hourly_wage * cents_per_dollar * tax_rate

theorem tax_deduction_proof : tax_deduction_cents = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_tax_deduction_proof_l2606_260686


namespace NUMINAMATH_CALUDE_tangent_slope_when_chord_maximized_l2606_260612

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 2
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 1

-- Define a point on circle M
def point_on_M (P : ℝ × ℝ) : Prop := circle_M P.1 P.2

-- Define a tangent line from a point on M to O
def is_tangent_line (P : ℝ × ℝ) (m : ℝ) : Prop :=
  point_on_M P ∧ ∃ A : ℝ × ℝ, circle_O A.1 A.2 ∧ (A.2 - P.2) = m * (A.1 - P.1)

-- Define the other intersection point Q
def other_intersection (P : ℝ × ℝ) (m : ℝ) (Q : ℝ × ℝ) : Prop :=
  point_on_M Q ∧ (Q.2 - P.2) = m * (Q.1 - P.1) ∧ P ≠ Q

-- Theorem statement
theorem tangent_slope_when_chord_maximized :
  ∃ P : ℝ × ℝ, ∃ m : ℝ, is_tangent_line P m ∧
  (∀ Q : ℝ × ℝ, other_intersection P m Q →
    ∀ P' : ℝ × ℝ, ∀ m' : ℝ, ∀ Q' : ℝ × ℝ,
      is_tangent_line P' m' ∧ other_intersection P' m' Q' →
      (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ≥ (P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) →
  m = -7 ∨ m = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_when_chord_maximized_l2606_260612


namespace NUMINAMATH_CALUDE_average_age_of_three_students_l2606_260668

theorem average_age_of_three_students
  (total_students : ℕ)
  (total_average : ℝ)
  (eleven_students : ℕ)
  (eleven_average : ℝ)
  (fifteenth_student_age : ℝ)
  (h1 : total_students = 15)
  (h2 : total_average = 15)
  (h3 : eleven_students = 11)
  (h4 : eleven_average = 16)
  (h5 : fifteenth_student_age = 7)
  : (total_students * total_average - eleven_students * eleven_average - fifteenth_student_age) / (total_students - eleven_students - 1) = 14 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_three_students_l2606_260668


namespace NUMINAMATH_CALUDE_clothing_division_l2606_260627

theorem clothing_division (total : ℕ) (first_load : ℕ) (h1 : total = 36) (h2 : first_load = 18) :
  (total - first_load) / 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_clothing_division_l2606_260627


namespace NUMINAMATH_CALUDE_probability_three_or_more_smile_l2606_260656

def probability_single_baby_smile : ℚ := 1 / 3

def number_of_babies : ℕ := 6

def probability_at_least_three_smile (p : ℚ) (n : ℕ) : ℚ :=
  1 - (Finset.sum (Finset.range 3) (λ k => (n.choose k : ℚ) * p^k * (1 - p)^(n - k)))

theorem probability_three_or_more_smile :
  probability_at_least_three_smile probability_single_baby_smile number_of_babies = 353 / 729 :=
sorry

end NUMINAMATH_CALUDE_probability_three_or_more_smile_l2606_260656


namespace NUMINAMATH_CALUDE_problem_statement_l2606_260606

theorem problem_statement (a b c d e : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : e < 0)      -- e is negative
  (h4 : |e| = 1)    -- absolute value of e is 1
  : (-a*b)^2009 - (c+d)^2010 - e^2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2606_260606


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2606_260626

theorem fraction_subtraction : (16 : ℚ) / 40 - (3 : ℚ) / 9 = (1 : ℚ) / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2606_260626


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_in_sequence_l2606_260605

theorem infinitely_many_perfect_squares_in_sequence :
  ∀ k : ℕ, ∃ x y : ℕ+, x > k ∧ y^2 = ⌊x * Real.sqrt 2⌋ := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_in_sequence_l2606_260605


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l2606_260628

/-- A parabola y = ax^2 + 6 is tangent to the line y = x if and only if a = 1/24 -/
theorem parabola_tangent_to_line (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 6 = x ∧ ∀ y : ℝ, y ≠ x → a * y^2 + 6 ≠ y) ↔ a = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l2606_260628


namespace NUMINAMATH_CALUDE_correct_substitution_l2606_260694

theorem correct_substitution (x y : ℝ) : 
  y = 1 - x ∧ x - 2*y = 4 → x - 2 + 2*x = 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_substitution_l2606_260694


namespace NUMINAMATH_CALUDE_system_solutions_l2606_260653

def is_solution (x y z w : ℝ) : Prop :=
  x^2 + 2*y^2 + 2*z^2 + w^2 = 43 ∧
  y^2 + z^2 + w^2 = 29 ∧
  5*z^2 - 3*w^2 + 4*x*y + 12*y*z + 6*z*x = 95

theorem system_solutions :
  {(x, y, z, w) : ℝ × ℝ × ℝ × ℝ | is_solution x y z w} =
  {(1, 2, 3, 4), (1, 2, 3, -4), (-1, -2, -3, 4), (-1, -2, -3, -4)} :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2606_260653


namespace NUMINAMATH_CALUDE_quadratic_properties_l2606_260635

/-- A quadratic function f(x) = mx² + (m-2)x + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

/-- The function is symmetric about the y-axis -/
def is_symmetric (m : ℝ) : Prop := ∀ x, f m x = f m (-x)

theorem quadratic_properties (m : ℝ) (h : is_symmetric m) :
  m = 2 ∧ 
  (∀ x y, x < y → f m x < f m y) ∧ 
  (∀ x, x > 0 → f m x > f m 0) ∧
  (∀ x, f m x ≥ 2) ∧ 
  f m 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2606_260635


namespace NUMINAMATH_CALUDE_modulus_of_z_l2606_260678

-- Define the complex number z
def z : ℂ := Complex.I * (2 - Complex.I)

-- State the theorem
theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2606_260678


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_nine_l2606_260665

theorem cubic_fraction_equals_nine (x y : ℝ) (hx : x = 7) (hy : y = 2) :
  (x^3 + y^3) / (x^2 - x*y + y^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_nine_l2606_260665


namespace NUMINAMATH_CALUDE_plot_width_calculation_l2606_260684

/-- Calculates the width of a rectangular plot given its length and fence specifications. -/
theorem plot_width_calculation (length width : ℝ) (num_poles : ℕ) (pole_distance : ℝ) : 
  length = 90 ∧ 
  num_poles = 28 ∧ 
  pole_distance = 10 ∧ 
  (num_poles - 1) * pole_distance = 2 * (length + width) →
  width = 45 := by
sorry

end NUMINAMATH_CALUDE_plot_width_calculation_l2606_260684


namespace NUMINAMATH_CALUDE_max_candies_for_one_student_l2606_260604

theorem max_candies_for_one_student
  (num_students : ℕ)
  (mean_candies : ℕ)
  (h_num_students : num_students = 40)
  (h_mean_candies : mean_candies = 6)
  (h_at_least_one : ∀ student, student ≥ 1) :
  ∃ (max_candies : ℕ), max_candies = 201 ∧
    ∀ (student_candies : ℕ),
      student_candies ≤ max_candies ∧
      (num_students - 1) * 1 + student_candies ≤ num_students * mean_candies :=
by sorry

end NUMINAMATH_CALUDE_max_candies_for_one_student_l2606_260604
