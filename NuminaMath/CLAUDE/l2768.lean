import Mathlib

namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l2768_276838

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the concept of axis of symmetry
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop := 
  ∀ x : ℝ, f (a + x) = f (a - x)

theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) (h : EvenFunction f) :
  AxisOfSymmetry (fun x ↦ f (x + 1)) (-1) := by
sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l2768_276838


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2768_276846

def quadratic_function (a m b x : ℝ) := a * x * (x - m) + b

theorem quadratic_function_properties
  (a m b : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_y_1_at_0 : quadratic_function a m b 0 = 1)
  (h_y_1_at_2 : quadratic_function a m b 2 = 1)
  (h_y_gt_4_at_3 : quadratic_function a m b 3 > 4)
  (k : ℝ)
  (h_passes_1_k : quadratic_function a m b 1 = k)
  (h_k_over_a : 0 < k / a ∧ k / a < 1) :
  m = 2 ∧ b = 1 ∧ a > 1 ∧ 1/2 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2768_276846


namespace NUMINAMATH_CALUDE_pyramid_sum_l2768_276800

/-- Given a pyramid of numbers where each number is the sum of the two above it,
    prove that the top number is 381. -/
theorem pyramid_sum (y z : ℕ) (h1 : y + 600 = 1119) (h2 : z + 1119 = 2019) (h3 : 381 + y = z) :
  ∃ x : ℕ, x = 381 ∧ x + y = z :=
by sorry

end NUMINAMATH_CALUDE_pyramid_sum_l2768_276800


namespace NUMINAMATH_CALUDE_unique_reverse_double_minus_one_l2768_276878

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Theorem stating that 37 is the unique two-digit number that satisfies the given condition -/
theorem unique_reverse_double_minus_one :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 2 * n - 1 = reverse_digits n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_reverse_double_minus_one_l2768_276878


namespace NUMINAMATH_CALUDE_scientific_notation_570_million_l2768_276824

theorem scientific_notation_570_million :
  (570000000 : ℝ) = 5.7 * (10 : ℝ)^8 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_570_million_l2768_276824


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2768_276856

theorem sqrt_product_equality : Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2768_276856


namespace NUMINAMATH_CALUDE_solution_set_transformation_l2768_276887

theorem solution_set_transformation (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c < 0 ↔ x < -2 ∨ x > 1/3) →
  (∀ x, c*x^2 - b*x + a ≥ 0 ↔ x ≤ -3 ∨ x ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_transformation_l2768_276887


namespace NUMINAMATH_CALUDE_courtyard_width_l2768_276861

theorem courtyard_width (length : ℝ) (num_bricks : ℕ) (brick_length brick_width : ℝ) :
  length = 25 ∧ 
  num_bricks = 20000 ∧ 
  brick_length = 0.2 ∧ 
  brick_width = 0.1 →
  (num_bricks : ℝ) * brick_length * brick_width / length = 16 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_width_l2768_276861


namespace NUMINAMATH_CALUDE_min_fraction_sum_l2768_276873

/-- The set of digits to choose from -/
def digits : Finset ℕ := {0, 1, 5, 6, 7, 8, 9}

/-- The proposition that four natural numbers are distinct digits from our set -/
def are_distinct_digits (w x y z : ℕ) : Prop :=
  w ∈ digits ∧ x ∈ digits ∧ y ∈ digits ∧ z ∈ digits ∧
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z

/-- The theorem stating the minimum value of the sum -/
theorem min_fraction_sum :
  ∃ (w x y z : ℕ), are_distinct_digits w x y z ∧ x ≠ 0 ∧ z ≠ 0 ∧
  (∀ (w' x' y' z' : ℕ), are_distinct_digits w' x' y' z' ∧ x' ≠ 0 ∧ z' ≠ 0 →
    (w : ℚ) / x + (y : ℚ) / z ≤ (w' : ℚ) / x' + (y' : ℚ) / z') ∧
  (w : ℚ) / x + (y : ℚ) / z = 1 / 8 :=
sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l2768_276873


namespace NUMINAMATH_CALUDE_distance_origin_to_point_l2768_276888

/-- The distance from the origin (0, 0) to the point (12, -5) in a rectangular coordinate system is 13 units. -/
theorem distance_origin_to_point :
  Real.sqrt (12^2 + (-5)^2) = 13 := by sorry

end NUMINAMATH_CALUDE_distance_origin_to_point_l2768_276888


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2768_276851

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := 4 * x^2 + b * x + c

-- Theorem statement
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (f b c (-1) = -1 ∧ f b c 0 = 0) →
  (∃ x₁ x₂ : ℝ, f b c x₁ = 20 ∧ f b c x₂ = 20 ∧ f b c (x₁ + x₂) = 0) →
  (∀ x : ℝ, (x < -5/4 ∨ x > 0) → f b c x > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2768_276851


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_l2768_276839

/-- Represents the daily rental business for canoes and kayaks. -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_count : ℕ
  kayak_count : ℕ

/-- The conditions of the rental business problem. -/
def rental_problem : RentalBusiness where
  canoe_price := 9
  kayak_price := 12
  canoe_count := 24  -- We know this from the solution, but it's derived from the conditions
  kayak_count := 18  -- We know this from the solution, but it's derived from the conditions

/-- The theorem stating the difference between canoes and kayaks rented. -/
theorem canoe_kayak_difference (b : RentalBusiness) 
  (h1 : b.canoe_price = 9)
  (h2 : b.kayak_price = 12)
  (h3 : 4 * b.kayak_count = 3 * b.canoe_count)
  (h4 : b.canoe_price * b.canoe_count + b.kayak_price * b.kayak_count = 432) :
  b.canoe_count - b.kayak_count = 6 := by
  sorry

#eval rental_problem.canoe_count - rental_problem.kayak_count

end NUMINAMATH_CALUDE_canoe_kayak_difference_l2768_276839


namespace NUMINAMATH_CALUDE_simplify_and_sum_exponents_l2768_276870

theorem simplify_and_sum_exponents (a b d : ℝ) : 
  ∃ (k : ℝ), (54 * a^5 * b^9 * d^14)^(1/3) = 3 * a * b^3 * d^4 * k ∧ 1 + 3 + 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_exponents_l2768_276870


namespace NUMINAMATH_CALUDE_debate_committee_combinations_l2768_276867

/-- The number of teams in the debate club -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the organizing team -/
def organizing_team_selection : ℕ := 4

/-- The number of members selected from each non-organizing team -/
def other_team_selection : ℕ := 3

/-- The total number of members in the debate organizing committee -/
def committee_size : ℕ := 16

/-- The number of possible debate organizing committees -/
def num_committees : ℕ := 3442073600

theorem debate_committee_combinations :
  (num_teams * Nat.choose team_size organizing_team_selection * 
   (Nat.choose team_size other_team_selection ^ (num_teams - 1))) = num_committees :=
sorry

end NUMINAMATH_CALUDE_debate_committee_combinations_l2768_276867


namespace NUMINAMATH_CALUDE_crow_percentage_among_non_pigeons_l2768_276885

theorem crow_percentage_among_non_pigeons (total_birds : ℝ) (crow_percentage : ℝ) (pigeon_percentage : ℝ)
  (h1 : crow_percentage = 40)
  (h2 : pigeon_percentage = 20)
  (h3 : 0 < total_birds) :
  (crow_percentage / (100 - pigeon_percentage)) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_crow_percentage_among_non_pigeons_l2768_276885


namespace NUMINAMATH_CALUDE_max_value_fraction_sum_l2768_276817

theorem max_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2 / 3 ∧
  ∃ x y, x > 0 ∧ y > 0 ∧ (x / (2 * x + y)) + (y / (x + 2 * y)) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_sum_l2768_276817


namespace NUMINAMATH_CALUDE_min_max_sum_l2768_276840

theorem min_max_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) (h_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 300) : 
  (max (x₁ + x₂) (max (x₂ + x₃) (max (x₃ + x₄) (x₄ + x₅)))) ≥ 100 ∧ 
  ∃ (y₁ y₂ y₃ y₄ y₅ : ℝ), y₁ ≥ 0 ∧ y₂ ≥ 0 ∧ y₃ ≥ 0 ∧ y₄ ≥ 0 ∧ y₅ ≥ 0 ∧ 
  y₁ + y₂ + y₃ + y₄ + y₅ = 300 ∧ 
  max (y₁ + y₂) (max (y₂ + y₃) (max (y₃ + y₄) (y₄ + y₅))) = 100 :=
by sorry

end NUMINAMATH_CALUDE_min_max_sum_l2768_276840


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l2768_276825

/-- Given two vectors a and b in ℝ³, we define c₁ and c₂ as linear combinations of a and b.
    This theorem states that c₁ and c₂ are not collinear. -/
theorem vectors_not_collinear :
  let a : Fin 3 → ℝ := ![1, 0, 1]
  let b : Fin 3 → ℝ := ![-2, 3, 5]
  let c₁ : Fin 3 → ℝ := a + 2 • b
  let c₂ : Fin 3 → ℝ := 3 • a - b
  ¬ ∃ (k : ℝ), c₁ = k • c₂ := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_collinear_l2768_276825


namespace NUMINAMATH_CALUDE_total_sharks_l2768_276816

/-- The total number of sharks on three beaches given specific ratios -/
theorem total_sharks (newport : ℕ) (dana_point : ℕ) (huntington : ℕ) 
  (h1 : newport = 22)
  (h2 : dana_point = 4 * newport)
  (h3 : huntington = dana_point / 2) :
  newport + dana_point + huntington = 154 := by
  sorry

end NUMINAMATH_CALUDE_total_sharks_l2768_276816


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2768_276865

def solution_set : Set ℝ := {x | x < -1 ∨ (-1 < x ∧ x < 0) ∨ x > 2}

def inequality (x : ℝ) : Prop := x^2 * (x^2 + 2*x + 1) > 2*x * (x^2 + 2*x + 1)

theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2768_276865


namespace NUMINAMATH_CALUDE_minute_hand_angle_2h40m_l2768_276889

/-- The angle turned by the minute hand when the hour hand moves for a given time -/
def minute_hand_angle (hours : ℝ) (minutes : ℝ) : ℝ :=
  -(hours * 360 + minutes * 6)

/-- Theorem: When the hour hand moves for 2 hours and 40 minutes, 
    the angle turned by the minute hand is -960° -/
theorem minute_hand_angle_2h40m :
  minute_hand_angle 2 40 = -960 := by sorry

end NUMINAMATH_CALUDE_minute_hand_angle_2h40m_l2768_276889


namespace NUMINAMATH_CALUDE_equation_one_solution_l2768_276869

theorem equation_one_solution :
  ∃ x₁ x₂ : ℝ, (3 * x₁^2 - 9 = 0) ∧ (3 * x₂^2 - 9 = 0) ∧ (x₁ = Real.sqrt 3) ∧ (x₂ = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_one_solution_l2768_276869


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2768_276845

theorem sqrt_equation_solution (x y : ℝ) :
  Real.sqrt (x^2 + y^2 - 1) = 1 - x - y ↔ (x = 1 ∧ y ≤ 0) ∨ (y = 1 ∧ x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2768_276845


namespace NUMINAMATH_CALUDE_recurrence_sequence_properties_l2768_276852

/-- A sequence that satisfies the given recurrence relation -/
def RecurrenceSequence (x : ℕ → ℝ) (a : ℝ) : Prop :=
  ∀ n, x (n + 2) = 3 * x (n + 1) - 2 * x n + a

/-- An arithmetic progression -/
def ArithmeticProgression (x : ℕ → ℝ) (b c : ℝ) : Prop :=
  ∀ n, x n = b + (c - b) * (n - 1)

/-- A geometric progression -/
def GeometricProgression (x : ℕ → ℝ) (b q : ℝ) : Prop :=
  ∀ n, x n = b * q^(n - 1)

theorem recurrence_sequence_properties
  (x : ℕ → ℝ) (a b c : ℝ) (h : a < 0) :
  (RecurrenceSequence x a ∧ ArithmeticProgression x b c) →
    (a = c - b ∧ c < b) ∧
  (RecurrenceSequence x a ∧ GeometricProgression x b 2) →
    (a = 0 ∧ c = 2*b ∧ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_recurrence_sequence_properties_l2768_276852


namespace NUMINAMATH_CALUDE_expression_evaluation_l2768_276884

theorem expression_evaluation : -20 + 8 * (10 / 2) - 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2768_276884


namespace NUMINAMATH_CALUDE_unique_element_implies_a_value_l2768_276807

def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

theorem unique_element_implies_a_value (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_element_implies_a_value_l2768_276807


namespace NUMINAMATH_CALUDE_smallest_multiple_square_l2768_276858

theorem smallest_multiple_square (a : ℕ) : 
  (∃ k : ℕ, a = 6 * k) ∧ 
  (∃ m : ℕ, a = 15 * m) ∧ 
  (∃ n : ℕ, a = n * n) ∧ 
  (∀ b : ℕ, b > 0 ∧ 
    (∃ k : ℕ, b = 6 * k) ∧ 
    (∃ m : ℕ, b = 15 * m) ∧ 
    (∃ n : ℕ, b = n * n) → 
    a ≤ b) → 
  a = 900 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_square_l2768_276858


namespace NUMINAMATH_CALUDE_range_of_sum_l2768_276833

theorem range_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 1) :
  a + b ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l2768_276833


namespace NUMINAMATH_CALUDE_five_letter_word_count_l2768_276899

/-- The number of letters in the alphabet -/
def alphabet_size : Nat := 26

/-- The number of vowels -/
def vowel_count : Nat := 5

/-- The number of five-letter words that begin and end with the same letter, 
    with the second letter always being a vowel -/
def word_count : Nat := alphabet_size * vowel_count * alphabet_size * alphabet_size

theorem five_letter_word_count : word_count = 87700 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_word_count_l2768_276899


namespace NUMINAMATH_CALUDE_college_entrance_exam_score_l2768_276801

theorem college_entrance_exam_score (total_questions unanswered_questions answered_questions correct_answers incorrect_answers : ℕ)
  (raw_score : ℚ) :
  total_questions = 85 →
  unanswered_questions = 3 →
  answered_questions = 82 →
  answered_questions = correct_answers + incorrect_answers →
  raw_score = 67 →
  raw_score = correct_answers - 0.25 * incorrect_answers →
  correct_answers = 70 := by
sorry

end NUMINAMATH_CALUDE_college_entrance_exam_score_l2768_276801


namespace NUMINAMATH_CALUDE_min_value_x_plus_3y_l2768_276809

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 3/b = 1 → x + 3*y ≤ a + 3*b ∧ ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ 1/c + 3/d = 1 ∧ c + 3*d = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_3y_l2768_276809


namespace NUMINAMATH_CALUDE_staircase_ratio_proof_l2768_276892

theorem staircase_ratio_proof (steps_first : ℕ) (step_height : ℚ) (total_height : ℚ) 
  (h1 : steps_first = 20)
  (h2 : step_height = 1/2)
  (h3 : total_height = 45) :
  ∃ (r : ℚ), 
    r * steps_first = (total_height / step_height - steps_first - (r * steps_first - 10)) ∧ 
    r = 2 := by
  sorry

end NUMINAMATH_CALUDE_staircase_ratio_proof_l2768_276892


namespace NUMINAMATH_CALUDE_solve_system_l2768_276811

theorem solve_system (a b : ℚ) 
  (eq1 : 2020 * a + 2024 * b = 2028)
  (eq2 : 2022 * a + 2026 * b = 2030) : 
  a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2768_276811


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l2768_276820

theorem hyperbola_line_intersection (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ P Q : ℝ × ℝ,
    (P.1^2 / a - P.2^2 / b = 1) ∧
    (Q.1^2 / a - Q.2^2 / b = 1) ∧
    (P.1 + P.2 = 1) ∧
    (Q.1 + Q.2 = 1) ∧
    (P.1 * Q.1 + P.2 * Q.2 = 0) →
    1 / a - 1 / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l2768_276820


namespace NUMINAMATH_CALUDE_teacherStudentArrangements_eq_144_l2768_276826

/-- The number of ways to arrange 2 teachers and 4 students in a row
    with exactly 2 students between the teachers -/
def teacherStudentArrangements : ℕ :=
  3 * 2 * 24

/-- Proof that the number of arrangements is 144 -/
theorem teacherStudentArrangements_eq_144 :
  teacherStudentArrangements = 144 := by
  sorry

end NUMINAMATH_CALUDE_teacherStudentArrangements_eq_144_l2768_276826


namespace NUMINAMATH_CALUDE_square_expression_l2768_276863

theorem square_expression (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  (1 / (1 / x - 1 / (x + 1)) - x = x^2) ∧
  (1 / (1 / (x - 1) - 1 / x) + x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_square_expression_l2768_276863


namespace NUMINAMATH_CALUDE_min_value_expression_l2768_276864

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 128) :
  x^2 + 8*x*y + 4*y^2 + 8*z^2 ≥ 384 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 128 ∧ x₀^2 + 8*x₀*y₀ + 4*y₀^2 + 8*z₀^2 = 384 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2768_276864


namespace NUMINAMATH_CALUDE_max_product_for_maximized_fraction_l2768_276844

def Digits := Fin 8

def validDigit (d : Digits) : ℕ := d.val + 2

theorem max_product_for_maximized_fraction :
  ∃ (A B C D : Digits),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (∀ (A' B' C' D' : Digits),
      A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' →
      (validDigit A' * validDigit B') / (validDigit C' * validDigit D' : ℚ) ≤
      (validDigit A * validDigit B) / (validDigit C * validDigit D : ℚ)) ∧
    validDigit A * validDigit B = 72 :=
by sorry

end NUMINAMATH_CALUDE_max_product_for_maximized_fraction_l2768_276844


namespace NUMINAMATH_CALUDE_consecutive_sum_product_l2768_276857

theorem consecutive_sum_product (n : ℕ) (h : n > 100) :
  ∃ (a b c : ℕ), (a > 1 ∧ b > 1 ∧ c > 1) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  ((n + (n + 1) + (n + 2) = a * b * c) ∨
   ((n + 1) + (n + 2) + (n + 3) = a * b * c)) := by
sorry


end NUMINAMATH_CALUDE_consecutive_sum_product_l2768_276857


namespace NUMINAMATH_CALUDE_train_length_calculation_l2768_276886

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : Real) (crossing_time : Real) (bridge_length : Real) :
  train_speed = 54 * (1000 / 3600) →
  crossing_time = 16.13204276991174 →
  bridge_length = 132 →
  train_speed * crossing_time - bridge_length = 109.9806415486761 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2768_276886


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l2768_276872

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 :=
sorry

theorem min_value_reciprocal_sum_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 4 / y = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l2768_276872


namespace NUMINAMATH_CALUDE_total_turnips_l2768_276874

theorem total_turnips (keith_turnips alyssa_turnips : ℕ) 
  (h1 : keith_turnips = 6) 
  (h2 : alyssa_turnips = 9) : 
  keith_turnips + alyssa_turnips = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l2768_276874


namespace NUMINAMATH_CALUDE_smallest_n_is_correct_l2768_276832

/-- The smallest positive integer n such that all roots of z^5 - z^3 + z = 0 are n^th roots of unity -/
def smallest_n : ℕ := 12

/-- The complex polynomial z^5 - z^3 + z -/
def f (z : ℂ) : ℂ := z^5 - z^3 + z

theorem smallest_n_is_correct :
  ∀ z : ℂ, f z = 0 → ∃ k : ℕ, z^smallest_n = 1 ∧
  ∀ m : ℕ, (∀ w : ℂ, f w = 0 → w^m = 1) → smallest_n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_is_correct_l2768_276832


namespace NUMINAMATH_CALUDE_race_speed_ratio_l2768_276860

/-- Given two runners a and b, where a's speed is some multiple of b's speed,
    and a gives b a 0.05 part of the race length as a head start to finish at the same time,
    prove that the ratio of a's speed to b's speed is 1/0.95 -/
theorem race_speed_ratio (v_a v_b : ℝ) (h1 : v_a > 0) (h2 : v_b > 0) 
    (h3 : ∃ k : ℝ, v_a = k * v_b) 
    (h4 : ∀ L : ℝ, L > 0 → L / v_a = (L - 0.05 * L) / v_b) : 
  v_a / v_b = 1 / 0.95 := by
sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l2768_276860


namespace NUMINAMATH_CALUDE_ice_pop_probability_l2768_276871

def total_ice_pops : ℕ := 17
def cherry_ice_pops : ℕ := 5
def children : ℕ := 5

theorem ice_pop_probability :
  1 - (Nat.factorial cherry_ice_pops : ℚ) / (Nat.factorial total_ice_pops / Nat.factorial (total_ice_pops - children)) = 1 - 1 / 4762 := by
  sorry

end NUMINAMATH_CALUDE_ice_pop_probability_l2768_276871


namespace NUMINAMATH_CALUDE_cos_alpha_plus_five_sixths_pi_l2768_276805

theorem cos_alpha_plus_five_sixths_pi (α : Real) 
  (h : Real.sin (α + π / 3) = 1 / 4) : 
  Real.cos (α + 5 * π / 6) = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_five_sixths_pi_l2768_276805


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2768_276841

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a + b)^2 + (b + c)^2 + (c + a)^2 = 2*(a + b + c) + 6*a*b*c) :
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ |2*(a + b + c) - 6*a*b*c| := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2768_276841


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_two_l2768_276879

-- Define the polynomial
def P (a b x : ℝ) : ℝ := a * (x^3 - x^2 + 3*x) + b * (2*x^2 + x) + x^3 - 5

-- State the theorem
theorem polynomial_value_at_negative_two 
  (a b : ℝ) 
  (h : P a b 2 = -17) : 
  P a b (-2) = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_two_l2768_276879


namespace NUMINAMATH_CALUDE_tan_pi_4_minus_theta_l2768_276862

theorem tan_pi_4_minus_theta (θ : Real) 
  (h1 : θ > -π/2 ∧ θ < 0) 
  (h2 : Real.cos (2*θ) - 3*Real.sin (θ - π/2) = 1) : 
  Real.tan (π/4 - θ) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_4_minus_theta_l2768_276862


namespace NUMINAMATH_CALUDE_tom_has_nine_balloons_l2768_276815

/-- The number of yellow balloons Sara has -/
def sara_balloons : ℕ := 8

/-- The total number of yellow balloons Tom and Sara have together -/
def total_balloons : ℕ := 17

/-- The number of yellow balloons Tom has -/
def tom_balloons : ℕ := total_balloons - sara_balloons

theorem tom_has_nine_balloons : tom_balloons = 9 := by
  sorry

end NUMINAMATH_CALUDE_tom_has_nine_balloons_l2768_276815


namespace NUMINAMATH_CALUDE_average_weight_of_class_class_average_weight_l2768_276875

theorem average_weight_of_class (group1_count : ℕ) (group1_avg : ℚ) 
                                (group2_count : ℕ) (group2_avg : ℚ) : ℚ :=
  let total_count := group1_count + group2_count
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  total_weight / total_count

theorem class_average_weight :
  average_weight_of_class 24 (50.25 : ℚ) 8 (45.15 : ℚ) = 49 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_class_class_average_weight_l2768_276875


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2768_276882

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_zero : 
  deriv f 0 = -120 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2768_276882


namespace NUMINAMATH_CALUDE_total_fish_caught_l2768_276806

def leo_fish : ℕ := 40
def agrey_fish : ℕ := leo_fish + 20

theorem total_fish_caught : leo_fish + agrey_fish = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_caught_l2768_276806


namespace NUMINAMATH_CALUDE_average_value_of_sequence_l2768_276847

theorem average_value_of_sequence (z : ℝ) : 
  (0 + 3*z + 6*z + 12*z + 24*z) / 5 = 9*z := by sorry

end NUMINAMATH_CALUDE_average_value_of_sequence_l2768_276847


namespace NUMINAMATH_CALUDE_expression_simplification_l2768_276853

theorem expression_simplification (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (a - 3*a/(a+1)) / ((a^2 - 4*a + 4)/(a+1)) = a / (a-2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2768_276853


namespace NUMINAMATH_CALUDE_lindas_hourly_rate_l2768_276849

/-- Proves that Linda's hourly rate for babysitting is $10.00 -/
theorem lindas_hourly_rate (application_fee : ℝ) (num_colleges : ℕ) (hours_worked : ℝ) :
  application_fee = 25 →
  num_colleges = 6 →
  hours_worked = 15 →
  (application_fee * num_colleges) / hours_worked = 10 := by
  sorry

end NUMINAMATH_CALUDE_lindas_hourly_rate_l2768_276849


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2768_276818

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def condition (a : ℕ → ℝ) : Prop :=
  ∀ n, |a (n + 1)| > a n

theorem condition_necessary_not_sufficient (a : ℕ → ℝ) :
  (is_increasing a → condition a) ∧ ¬(condition a → is_increasing a) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2768_276818


namespace NUMINAMATH_CALUDE_smallest_add_to_multiple_of_five_l2768_276804

theorem smallest_add_to_multiple_of_five : ∃ (n : ℕ), n > 0 ∧ (729 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (729 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_add_to_multiple_of_five_l2768_276804


namespace NUMINAMATH_CALUDE_clothing_company_wage_promise_l2768_276848

/-- Represents the wage calculation and constraints for skilled workers in a clothing company. -/
theorem clothing_company_wage_promise (base_salary : ℝ) (wage_a : ℝ) (wage_b : ℝ) 
  (hours_per_day : ℝ) (days_per_month : ℝ) (time_a : ℝ) (time_b : ℝ) :
  base_salary = 800 →
  wage_a = 16 →
  wage_b = 12 →
  hours_per_day = 8 →
  days_per_month = 25 →
  time_a = 2 →
  time_b = 1 →
  ∀ a : ℝ, 
    a ≥ (hours_per_day * days_per_month - 2 * a) / 2 →
    a ≥ 0 →
    a ≤ hours_per_day * days_per_month / (2 * time_a) →
    base_salary + wage_a * a + wage_b * (hours_per_day * days_per_month / time_b - 2 * a / time_b) < 3000 :=
by sorry

end NUMINAMATH_CALUDE_clothing_company_wage_promise_l2768_276848


namespace NUMINAMATH_CALUDE_car_ownership_theorem_l2768_276894

/-- The number of cars owned by Cathy, Lindsey, Carol, and Susan -/
def total_cars (cathy lindsey carol susan : ℕ) : ℕ :=
  cathy + lindsey + carol + susan

/-- Theorem stating the total number of cars owned by the four people -/
theorem car_ownership_theorem (cathy lindsey carol susan : ℕ) 
  (h1 : cathy = 5)
  (h2 : lindsey = cathy + 4)
  (h3 : carol = 2 * cathy)
  (h4 : susan = carol - 2) :
  total_cars cathy lindsey carol susan = 32 := by
  sorry

#check car_ownership_theorem

end NUMINAMATH_CALUDE_car_ownership_theorem_l2768_276894


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2768_276823

/-- Geometric sequence with given properties -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 3/2) ∧
  (∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q) ∧
  (a 1 + a 2 + a 3 = 9/2)

/-- The general term of the geometric sequence -/
theorem geometric_sequence_general_term (a : ℕ → ℚ) (h : geometric_sequence a) :
  (∀ n : ℕ, a n = 3/2 * (-2)^(n - 1)) ∨ (∀ n : ℕ, a n = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2768_276823


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2768_276803

theorem complex_power_magnitude : 
  Complex.abs ((2/5 : ℂ) + (7/5 : ℂ) * Complex.I) ^ 8 = 7890481/390625 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2768_276803


namespace NUMINAMATH_CALUDE_ellipse_m_values_l2768_276890

/-- Definition of the ellipse equation -/
def is_ellipse (x y m : ℝ) : Prop :=
  x^2 / (10 - m) + y^2 / (m - 2) = 1

/-- Definition of focal length -/
def focal_length (m : ℝ) : ℝ := 4

/-- Theorem stating the possible values of m -/
theorem ellipse_m_values :
  ∀ m : ℝ, (∃ x y : ℝ, is_ellipse x y m) → focal_length m = 4 → m = 4 ∨ m = 8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_values_l2768_276890


namespace NUMINAMATH_CALUDE_village_birth_probability_l2768_276827

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- A village with a custom of having children until a boy is born -/
structure Village where
  /-- The probability of having a boy in a single birth -/
  prob_boy : ℝ
  /-- The probability of having a girl in a single birth -/
  prob_girl : ℝ
  /-- The proportion of boys to girls in the village after some time -/
  boy_girl_ratio : ℝ
  /-- The probabilities sum to 1 -/
  prob_sum_one : prob_boy + prob_girl = 1
  /-- The proportion of boys to girls is 1:1 -/
  equal_ratio : boy_girl_ratio = 1

/-- Theorem: In a village with the given custom, the probability of having a boy or a girl is 1/2 -/
theorem village_birth_probability (v : Village) : v.prob_boy = 1/2 ∧ v.prob_girl = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_village_birth_probability_l2768_276827


namespace NUMINAMATH_CALUDE_pin_combinations_l2768_276812

/-- The number of unique permutations of a multiset with elements {5, 3, 3, 7} -/
def pinPermutations : ℕ :=
  Nat.factorial 4 / (Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 1)

theorem pin_combinations : pinPermutations = 12 := by
  sorry

end NUMINAMATH_CALUDE_pin_combinations_l2768_276812


namespace NUMINAMATH_CALUDE_fractional_parts_inequality_l2768_276836

theorem fractional_parts_inequality (q : ℕ+) (hq : ¬ ∃ (m : ℕ), q = m^3) :
  ∃ (c : ℝ), c > 0 ∧
  ∀ (n : ℕ+), 
    (n : ℝ) * q.val ^ (1/3 : ℝ) - ⌊(n : ℝ) * q.val ^ (1/3 : ℝ)⌋ +
    (n : ℝ) * q.val ^ (2/3 : ℝ) - ⌊(n : ℝ) * q.val ^ (2/3 : ℝ)⌋ ≥
    c * (n : ℝ) ^ (-1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_fractional_parts_inequality_l2768_276836


namespace NUMINAMATH_CALUDE_complex_vector_sum_l2768_276850

theorem complex_vector_sum (z₁ z₂ z₃ : ℂ) (x y : ℝ) 
  (h₁ : z₁ = -1 + I)
  (h₂ : z₂ = 1 + I)
  (h₃ : z₃ = 1 + 4*I)
  (h₄ : z₃ = x • z₁ + y • z₂) :
  x + y = 4 := by sorry

end NUMINAMATH_CALUDE_complex_vector_sum_l2768_276850


namespace NUMINAMATH_CALUDE_bobs_roommates_l2768_276821

theorem bobs_roommates (john_roommates : ℕ) (h1 : john_roommates = 25) :
  ∃ (bob_roommates : ℕ), john_roommates = 2 * bob_roommates + 5 → bob_roommates = 10 := by
  sorry

end NUMINAMATH_CALUDE_bobs_roommates_l2768_276821


namespace NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l2768_276810

theorem sum_of_quotient_dividend_divisor : 
  ∀ (N D : ℕ), 
  N = 50 → 
  D = 5 → 
  N + D + (N / D) = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l2768_276810


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_l2768_276866

def is_valid_equation (a b c : ℕ) : Prop :=
  a + b = c ∧ 
  a ≥ 1000 ∧ a < 10000 ∧
  b ≥ 10 ∧ b < 100 ∧
  c ≥ 1000 ∧ c < 10000

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  (∀ i j, i ≠ j → digits.nthLe i sorry ≠ digits.nthLe j sorry) ∧
  digits.length ≤ 10

theorem smallest_four_digit_number (a b c : ℕ) :
  is_valid_equation a b c →
  has_distinct_digits a →
  has_distinct_digits b →
  has_distinct_digits c →
  (∀ x, has_distinct_digits x → x ≥ 1000 → x < c → False) →
  c = 2034 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_l2768_276866


namespace NUMINAMATH_CALUDE_spinster_cat_problem_l2768_276834

theorem spinster_cat_problem (S C : ℕ) : 
  S * 9 = C * 2 →  -- Ratio of spinsters to cats is 2:9
  C = S + 42 →     -- There are 42 more cats than spinsters
  S = 12           -- The number of spinsters is 12
:= by sorry

end NUMINAMATH_CALUDE_spinster_cat_problem_l2768_276834


namespace NUMINAMATH_CALUDE_tyler_aquariums_l2768_276835

-- Define the given conditions
def animals_per_aquarium : ℕ := 64
def total_animals : ℕ := 512

-- State the theorem
theorem tyler_aquariums : 
  total_animals / animals_per_aquarium = 8 := by
  sorry

end NUMINAMATH_CALUDE_tyler_aquariums_l2768_276835


namespace NUMINAMATH_CALUDE_evaluate_expression_l2768_276876

theorem evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = 2) : y^2 * (y - 2*x) = 16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2768_276876


namespace NUMINAMATH_CALUDE_product_over_sum_equals_180_l2768_276808

theorem product_over_sum_equals_180 : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7) / (1 + 2 + 3 + 4 + 5 + 6 + 7) = 180 := by
  sorry

end NUMINAMATH_CALUDE_product_over_sum_equals_180_l2768_276808


namespace NUMINAMATH_CALUDE_sphere_radii_ratio_l2768_276819

/-- The ratio of radii of two spheres given their volumes -/
theorem sphere_radii_ratio (V_large V_small : ℝ) (h1 : V_large = 432 * Real.pi) 
  (h2 : V_small = 0.275 * V_large) : 
  (V_small / V_large)^(1/3 : ℝ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radii_ratio_l2768_276819


namespace NUMINAMATH_CALUDE_min_value_problem_l2768_276831

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  ∃ (m : ℝ), m = (1 : ℝ)/5184 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 
    1/x + 1/y + 1/z = 9 → x^4 * y^3 * z^2 ≥ m ∧
    a^4 * b^3 * c^2 = m :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2768_276831


namespace NUMINAMATH_CALUDE_range_of_a_inequality_l2768_276893

theorem range_of_a_inequality (a : ℝ) : 
  (∃ x : ℝ, |a| ≥ |x + 1| + |x - 2|) ↔ a ∈ Set.Iic 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_inequality_l2768_276893


namespace NUMINAMATH_CALUDE_prism_with_27_edges_has_11_faces_l2768_276898

/-- A prism is a polyhedron with two congruent and parallel faces (called bases) 
    and whose other faces (called lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ
  lateral_faces : ℕ
  base_edges : ℕ

/-- The number of edges in a prism is equal to 3 times the number of lateral faces. -/
axiom prism_edge_count (p : Prism) : p.edges = 3 * p.lateral_faces

/-- The number of edges in each base of a prism is equal to the number of lateral faces. -/
axiom prism_base_edge_count (p : Prism) : p.base_edges = p.lateral_faces

/-- The total number of faces in a prism is equal to the number of lateral faces plus 2 (for the bases). -/
def total_faces (p : Prism) : ℕ := p.lateral_faces + 2

/-- Theorem: A prism with 27 edges has 11 faces. -/
theorem prism_with_27_edges_has_11_faces (p : Prism) (h : p.edges = 27) : total_faces p = 11 := by
  sorry


end NUMINAMATH_CALUDE_prism_with_27_edges_has_11_faces_l2768_276898


namespace NUMINAMATH_CALUDE_addition_to_reach_81_l2768_276868

theorem addition_to_reach_81 : 5 * 12 / (180 / 3) + 80 = 81 := by
  sorry

end NUMINAMATH_CALUDE_addition_to_reach_81_l2768_276868


namespace NUMINAMATH_CALUDE_inequality_relation_l2768_276855

theorem inequality_relation (x y : ℝ) : 2*x - 5 < 2*y - 5 → x < y := by
  sorry

end NUMINAMATH_CALUDE_inequality_relation_l2768_276855


namespace NUMINAMATH_CALUDE_statement_1_statement_2_statement_3_l2768_276896

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (5*a + 1) * x + 4*a + 4

-- Statement 1
theorem statement_1 (a : ℝ) (h : a < -1) : f a 0 < 0 := by sorry

-- Statement 2
theorem statement_2 (a : ℝ) (h : a > 0) : 
  ∃ (y : ℝ), y = 3 ∧ ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → f a x ≤ y := by sorry

-- Statement 3
theorem statement_3 (a : ℝ) (h : a < 0) : 
  f a 2 > f a 3 ∧ f a 3 > f a 4 := by sorry

end NUMINAMATH_CALUDE_statement_1_statement_2_statement_3_l2768_276896


namespace NUMINAMATH_CALUDE_craftsman_production_theorem_l2768_276822

/-- The number of parts manufactured by a master craftsman during a shift -/
def parts_manufactured : ℕ → ℕ → ℕ → ℕ
  | initial_rate, rate_increase, additional_parts =>
    initial_rate + additional_parts

/-- The time needed to manufacture parts at a given rate -/
def time_needed : ℕ → ℕ → ℚ
  | parts, rate => (parts : ℚ) / (rate : ℚ)

theorem craftsman_production_theorem 
  (initial_rate : ℕ) 
  (rate_increase : ℕ) 
  (additional_parts : ℕ) :
  initial_rate = 35 →
  rate_increase = 15 →
  time_needed additional_parts initial_rate - 
    time_needed additional_parts (initial_rate + rate_increase) = (3 : ℚ) / 2 →
  parts_manufactured initial_rate rate_increase additional_parts = 210 :=
by sorry

end NUMINAMATH_CALUDE_craftsman_production_theorem_l2768_276822


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2768_276829

def f (x : ℝ) := x^2 - 2*x - 3

theorem quadratic_inequality (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-3) = y₁) 
  (h₂ : f (-2) = y₂) 
  (h₃ : f 2 = y₃) : 
  y₃ < y₂ ∧ y₂ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2768_276829


namespace NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l2768_276830

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let d₂ := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let d₃ := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  d₁ = d₂ ∧ d₂ = d₃

-- Define the branches of the hyperbola
def on_branch_1 (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ hyperbola x y
def on_branch_2 (x y : ℝ) : Prop := x < 0 ∧ y < 0 ∧ hyperbola x y

-- Theorem statement
theorem hyperbola_equilateral_triangle :
  ∀ (P Q R : ℝ × ℝ),
  hyperbola P.1 P.2 → hyperbola Q.1 Q.2 → hyperbola R.1 R.2 →
  is_equilateral_triangle P Q R →
  P = (-1, -1) →
  on_branch_2 P.1 P.2 →
  on_branch_1 Q.1 Q.2 →
  on_branch_1 R.1 R.2 →
  (¬(on_branch_1 P.1 P.2 ∧ on_branch_1 Q.1 Q.2 ∧ on_branch_1 R.1 R.2) ∧
   ¬(on_branch_2 P.1 P.2 ∧ on_branch_2 Q.1 Q.2 ∧ on_branch_2 R.1 R.2)) ∧
  ((Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) ∨
   (Q = (2 + Real.sqrt 3, 2 - Real.sqrt 3) ∧ R = (2 - Real.sqrt 3, 2 + Real.sqrt 3))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l2768_276830


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l2768_276891

theorem opposite_of_negative_three : -(-(3 : ℤ)) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l2768_276891


namespace NUMINAMATH_CALUDE_even_increasing_negative_inequality_l2768_276814

/-- A function that is even and increasing on (-∞, -1] -/
def EvenIncreasingNegative (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x ≤ y → y ≤ -1 → f x ≤ f y)

/-- Theorem stating the inequality for functions that are even and increasing on (-∞, -1] -/
theorem even_increasing_negative_inequality (f : ℝ → ℝ) 
  (h : EvenIncreasingNegative f) : 
  f 2 < f (-3/2) ∧ f (-3/2) < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_negative_inequality_l2768_276814


namespace NUMINAMATH_CALUDE_dereks_savings_l2768_276877

theorem dereks_savings (n : ℕ) (a : ℝ) (r : ℝ) : 
  n = 12 → a = 2 → r = 2 → 
  a * (1 - r^n) / (1 - r) = 8190 := by
  sorry

end NUMINAMATH_CALUDE_dereks_savings_l2768_276877


namespace NUMINAMATH_CALUDE_xy_equals_ten_l2768_276802

theorem xy_equals_ten (x y : ℝ) (h : x * (x + y) = x^2 + 10) : x * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_ten_l2768_276802


namespace NUMINAMATH_CALUDE_square_of_binomial_c_value_l2768_276813

theorem square_of_binomial_c_value (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + c = (a * x + b)^2) → c = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_c_value_l2768_276813


namespace NUMINAMATH_CALUDE_extra_bananas_per_child_l2768_276843

/-- Given the total number of children, number of absent children, and original banana allocation,
    calculate the number of extra bananas each present child received. -/
theorem extra_bananas_per_child 
  (total_children : ℕ) 
  (absent_children : ℕ) 
  (original_allocation : ℕ) 
  (h1 : total_children = 780)
  (h2 : absent_children = 390)
  (h3 : original_allocation = 2)
  (h4 : absent_children < total_children) :
  (total_children * original_allocation) / (total_children - absent_children) - original_allocation = 2 :=
by sorry

end NUMINAMATH_CALUDE_extra_bananas_per_child_l2768_276843


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l2768_276897

theorem max_digits_product_5_4 : 
  ∀ (a b : ℕ), 
  10000 ≤ a ∧ a ≤ 99999 → 
  1000 ≤ b ∧ b ≤ 9999 → 
  a * b < 1000000000 := by
sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l2768_276897


namespace NUMINAMATH_CALUDE_coin_difference_l2768_276895

/-- Represents the number of coins of each denomination in Tom's collection -/
structure CoinCollection where
  fiveCent : ℚ
  tenCent : ℚ
  twentyCent : ℚ

/-- Conditions for Tom's coin collection -/
def validCollection (c : CoinCollection) : Prop :=
  c.fiveCent + c.tenCent + c.twentyCent = 30 ∧
  c.tenCent = 2 * c.fiveCent ∧
  5 * c.fiveCent + 10 * c.tenCent + 20 * c.twentyCent = 340

/-- The main theorem to prove -/
theorem coin_difference (c : CoinCollection) 
  (h : validCollection c) : c.twentyCent - c.fiveCent = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_coin_difference_l2768_276895


namespace NUMINAMATH_CALUDE_average_monthly_sales_l2768_276883

def monthly_sales : List ℝ := [80, 100, 75, 95, 110, 180, 90, 115, 130, 200, 160, 140]

theorem average_monthly_sales :
  (monthly_sales.sum / monthly_sales.length : ℝ) = 122.92 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l2768_276883


namespace NUMINAMATH_CALUDE_equations_not_equivalent_l2768_276854

/-- The solution set of the equation 2√(x+5) = x+2 -/
def SolutionSet1 : Set ℝ :=
  {x : ℝ | 2 * Real.sqrt (x + 5) = x + 2}

/-- The solution set of the equation 4(x+5) = (x+2)² -/
def SolutionSet2 : Set ℝ :=
  {x : ℝ | 4 * (x + 5) = (x + 2)^2}

/-- Theorem stating that the equations are not equivalent -/
theorem equations_not_equivalent : SolutionSet1 ≠ SolutionSet2 := by
  sorry

#check equations_not_equivalent

end NUMINAMATH_CALUDE_equations_not_equivalent_l2768_276854


namespace NUMINAMATH_CALUDE_xy_yz_zx_bounds_l2768_276837

theorem xy_yz_zx_bounds (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2 + 1) : 
  ∃ (N n : ℝ), (∀ t : ℝ, t = x*y + y*z + z*x → t ≤ N ∧ n ≤ t) ∧ 11 < N + 6*n ∧ N + 6*n < 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_yz_zx_bounds_l2768_276837


namespace NUMINAMATH_CALUDE_stuffed_animals_problem_l2768_276842

theorem stuffed_animals_problem (M : ℕ) : 
  (M + 2*M + (2*M + 5) = 175) → M = 34 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_problem_l2768_276842


namespace NUMINAMATH_CALUDE_parabola_above_line_l2768_276880

/-- Given non-zero real numbers a, b, and c, if the parabola y = ax^2 + bx + c is positioned
    above the line y = cx, then the parabola y = cx^2 - bx + a is positioned above
    the line y = cx - b. -/
theorem parabola_above_line (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_above : ∀ x, a * x^2 + b * x + c > c * x) :
  ∀ x, c * x^2 - b * x + a > c * x - b :=
by sorry

end NUMINAMATH_CALUDE_parabola_above_line_l2768_276880


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2768_276859

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 → (x + y : ℕ) ≤ (a + b : ℕ)) → 
  (x + y : ℕ) = 50 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2768_276859


namespace NUMINAMATH_CALUDE_h_of_3_eq_3_l2768_276828

/-- The function h(x) is defined implicitly by this equation -/
def h_equation (x : ℝ) (h : ℝ → ℝ) : Prop :=
  (x^(2^2007 - 1) - 1) * h x = (x + 1) * (x^2 + 1) * (x^4 + 1) * (x^(2^2006) + 1) - 1

/-- The theorem states that h(3) = 3 for the function h defined by h_equation -/
theorem h_of_3_eq_3 :
  ∃ h : ℝ → ℝ, h_equation 3 h ∧ h 3 = 3 := by sorry

end NUMINAMATH_CALUDE_h_of_3_eq_3_l2768_276828


namespace NUMINAMATH_CALUDE_percentage_equation_solution_l2768_276881

/-- The solution to the equation (47% of 1442 - x% of 1412) + 63 = 252 is approximately 34.63% -/
theorem percentage_equation_solution : 
  ∃ x : ℝ, abs (x - 34.63) < 0.01 ∧ 
  ((47 / 100) * 1442 - (x / 100) * 1412) + 63 = 252 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_solution_l2768_276881
