import Mathlib

namespace NUMINAMATH_CALUDE_probability_different_tens_digits_value_l2465_246516

def range_start : ℕ := 10
def range_end : ℕ := 59
def num_chosen : ℕ := 5

def probability_different_tens_digits : ℚ :=
  (10 ^ num_chosen : ℚ) / (Nat.choose (range_end - range_start + 1) num_chosen)

theorem probability_different_tens_digits_value :
  probability_different_tens_digits = 2500 / 52969 := by sorry

end NUMINAMATH_CALUDE_probability_different_tens_digits_value_l2465_246516


namespace NUMINAMATH_CALUDE_race_earnings_theorem_l2465_246572

/-- Represents the race parameters and results -/
structure RaceData where
  duration : ℕ         -- Race duration in minutes
  lap_distance : ℕ     -- Distance of one lap in meters
  certificate_rate : ℚ -- Gift certificate rate in dollars per 100 meters
  winner_laps : ℕ      -- Number of laps run by the winner

/-- Calculates the average earnings per minute for the race winner -/
def average_earnings_per_minute (data : RaceData) : ℚ :=
  (data.winner_laps * data.lap_distance * data.certificate_rate) / (100 * data.duration)

/-- Theorem stating that for the given race conditions, the average earnings per minute is $7 -/
theorem race_earnings_theorem (data : RaceData) 
  (h1 : data.duration = 12)
  (h2 : data.lap_distance = 100)
  (h3 : data.certificate_rate = 7/2)
  (h4 : data.winner_laps = 24) :
  average_earnings_per_minute data = 7 := by
  sorry

end NUMINAMATH_CALUDE_race_earnings_theorem_l2465_246572


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2465_246584

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2465_246584


namespace NUMINAMATH_CALUDE_sequence_formulas_l2465_246542

/-- Given an arithmetic sequence a_n with first term 19 and common difference -2,
    and a geometric sequence b_n - a_n with first term 1 and common ratio 3,
    prove the formulas for a_n, S_n, b_n, and T_n. -/
theorem sequence_formulas (n : ℕ) :
  let a : ℕ → ℝ := λ k => 19 - 2 * (k - 1)
  let S : ℕ → ℝ := λ k => (k * (a 1 + a k)) / 2
  let b : ℕ → ℝ := λ k => a k + 3^(k - 1)
  let T : ℕ → ℝ := λ k => S k + (3^k - 1) / 2
  (a n = 21 - 2 * n) ∧
  (S n = 20 * n - n^2) ∧
  (b n = 21 - 2 * n + 3^(n - 1)) ∧
  (T n = 20 * n - n^2 + (3^n - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formulas_l2465_246542


namespace NUMINAMATH_CALUDE_wednesday_is_valid_start_day_l2465_246567

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

def isValidRedemptionDay (startDay : DayOfWeek) : Prop :=
  ∀ i : Fin 6, advanceDays startDay (i.val * 10) ≠ DayOfWeek.Sunday

theorem wednesday_is_valid_start_day :
  isValidRedemptionDay DayOfWeek.Wednesday ∧
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Wednesday → ¬isValidRedemptionDay d :=
sorry

end NUMINAMATH_CALUDE_wednesday_is_valid_start_day_l2465_246567


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2465_246550

/-- The quadratic function f(x) = -x^2 + 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

/-- y₁ is the value of f at x = -2 -/
def y₁ : ℝ := f (-2)

/-- y₂ is the value of f at x = 2 -/
def y₂ : ℝ := f 2

/-- y₃ is the value of f at x = -4 -/
def y₃ : ℝ := f (-4)

/-- Theorem: For the quadratic function f(x) = -x^2 + 2x + 3,
    if f(-2) = y₁, f(2) = y₂, and f(-4) = y₃, then y₂ > y₁ > y₃ -/
theorem quadratic_inequality : y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2465_246550


namespace NUMINAMATH_CALUDE_peter_class_size_l2465_246535

/-- The number of hands in Peter's class, excluding Peter's hands -/
def hands_excluding_peter : ℕ := 20

/-- The number of hands each student has -/
def hands_per_student : ℕ := 2

/-- The total number of hands in the class, including Peter's -/
def total_hands : ℕ := hands_excluding_peter + hands_per_student

/-- The number of students in Peter's class, including Peter -/
def students_in_class : ℕ := total_hands / hands_per_student

theorem peter_class_size :
  students_in_class = 11 :=
sorry

end NUMINAMATH_CALUDE_peter_class_size_l2465_246535


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2465_246585

/-- A line intersects a circle at two points with a specific distance between them -/
theorem line_circle_intersection (a : ℝ) (A B : ℝ × ℝ) :
  a > 0 →
  (∀ x y, y = x + 2*a → x^2 + y^2 - 2*a*y - 2 = 0 → (x, y) = A ∨ (x, y) = B) →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 →
  a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2465_246585


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2465_246579

theorem polynomial_divisibility (a : ℤ) : 
  ∃ k : ℤ, (3*a + 5)^2 - 4 = (a + 1) * k := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2465_246579


namespace NUMINAMATH_CALUDE_weight_comparison_l2465_246544

def weights : List ℝ := [4, 4, 5, 7, 9, 120]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem weight_comparison (h : weights = [4, 4, 5, 7, 9, 120]) : 
  mean weights - median weights = 19 := by sorry

end NUMINAMATH_CALUDE_weight_comparison_l2465_246544


namespace NUMINAMATH_CALUDE_system_solution_relation_l2465_246533

theorem system_solution_relation (a₁ a₂ c₁ c₂ : ℝ) :
  (2 * a₁ + 3 = c₁ ∧ 2 * a₂ + 3 = c₂) →
  (∃! (x y : ℝ), a₁ * x + y = a₁ - c₁ ∧ a₂ * x + y = a₂ - c₂ ∧ x = -1 ∧ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_relation_l2465_246533


namespace NUMINAMATH_CALUDE_unique_function_exists_l2465_246503

-- Define the positive rationals
def PositiveRationals := {q : ℚ // q > 0}

-- Define the function type
def FunctionType := PositiveRationals → PositiveRationals

-- Define the conditions
def Condition1 (f : FunctionType) : Prop :=
  ∀ q : PositiveRationals, 0 < q.val ∧ q.val < 1/2 →
    f q = ⟨1 + (f ⟨q.val / (1 - 2*q.val), sorry⟩).val, sorry⟩

def Condition2 (f : FunctionType) : Prop :=
  ∀ q : PositiveRationals, 1 < q.val ∧ q.val ≤ 2 →
    f q = ⟨1 + (f ⟨q.val + 1, sorry⟩).val, sorry⟩

def Condition3 (f : FunctionType) : Prop :=
  ∀ q : PositiveRationals, (f q).val * (f ⟨1/q.val, sorry⟩).val = 1

-- State the theorem
theorem unique_function_exists :
  ∃! f : FunctionType, Condition1 f ∧ Condition2 f ∧ Condition3 f :=
sorry

end NUMINAMATH_CALUDE_unique_function_exists_l2465_246503


namespace NUMINAMATH_CALUDE_green_ball_probability_l2465_246577

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers A, B, and C -/
def containerA : Container := ⟨5, 5⟩
def containerB : Container := ⟨7, 3⟩
def containerC : Container := ⟨6, 4⟩

/-- The list of all containers -/
def containers : List Container := [containerA, containerB, containerC]

/-- The theorem stating the probability of selecting a green ball -/
theorem green_ball_probability : 
  (List.sum (containers.map greenProbability)) / containers.length = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2465_246577


namespace NUMINAMATH_CALUDE_problem_solution_l2465_246518

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -15)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 6) :
  b / (a + b) + c / (b + c) + a / (c + a) = 12 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2465_246518


namespace NUMINAMATH_CALUDE_third_term_base_l2465_246570

theorem third_term_base (h a b c : ℕ+) (base : ℕ+) : 
  (225 ∣ h) → 
  (216 ∣ h) → 
  h = 2^(a.val) * 3^(b.val) * base^(c.val) →
  a.val + b.val + c.val = 8 →
  base = 5 := by sorry

end NUMINAMATH_CALUDE_third_term_base_l2465_246570


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l2465_246540

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y = 0

/-- The radius of the circle defined by circle_equation -/
def circle_radius : ℝ := 5

/-- Theorem stating that the radius of the circle defined by circle_equation is 5 -/
theorem circle_radius_is_five : 
  ∀ x y : ℝ, circle_equation x y → 
  ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l2465_246540


namespace NUMINAMATH_CALUDE_biology_marks_calculation_l2465_246543

def english_marks : ℕ := 73
def math_marks : ℕ := 69
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 64
def average_marks : ℕ := 76
def total_subjects : ℕ := 5

theorem biology_marks_calculation : 
  (english_marks + math_marks + physics_marks + chemistry_marks + 
   (average_marks * total_subjects - (english_marks + math_marks + physics_marks + chemistry_marks))) 
  / total_subjects = average_marks :=
by sorry

end NUMINAMATH_CALUDE_biology_marks_calculation_l2465_246543


namespace NUMINAMATH_CALUDE_square_sum_theorem_l2465_246529

theorem square_sum_theorem (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 8) 
  (h2 : x * y = -6) : 
  9 * x^2 + 16 * y^2 = 208 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l2465_246529


namespace NUMINAMATH_CALUDE_log_equation_solution_l2465_246524

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log (x^3) / Real.log 9 = 9 →
  x = 3^(18/5) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2465_246524


namespace NUMINAMATH_CALUDE_triangle_properties_l2465_246521

/-- Given a triangle ABC where b = 2√3 and 2a - c = 2b cos C, prove that B = π/3 and the maximum value of 3a + 2c is 4√19 -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b = 2 * Real.sqrt 3 →
  2 * a - c = 2 * b * Real.cos C →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (B = π / 3 ∧ ∃ (x : ℝ), 3 * a + 2 * c ≤ 4 * Real.sqrt 19 ∧ 
    ∃ (A' B' C' a' b' c' : ℝ), 
      b' = 2 * Real.sqrt 3 ∧
      2 * a' - c' = 2 * b' * Real.cos C' ∧
      0 < A' ∧ A' < π ∧
      0 < B' ∧ B' < π ∧
      0 < C' ∧ C' < π ∧
      A' + B' + C' = π ∧
      3 * a' + 2 * c' = x) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2465_246521


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2465_246596

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 99) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2465_246596


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2465_246537

theorem equilateral_triangle_perimeter (R : ℝ) (chord_length : ℝ) (chord_distance : ℝ) :
  chord_length = 2 →
  chord_distance = 3 →
  R^2 = chord_distance^2 + (chord_length/2)^2 →
  ∃ (perimeter : ℝ), perimeter = 3 * R * Real.sqrt 3 ∧ perimeter = 3 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2465_246537


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2465_246594

-- Define the function f(x) = x^2 + px + q
def f (p q x : ℝ) : ℝ := x^2 + p*x + q

-- Theorem statement
theorem min_value_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f p q x_min ≤ f p q x ∧ x_min = -p/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2465_246594


namespace NUMINAMATH_CALUDE_custom_op_zero_l2465_246561

def custom_op (a b c : ℝ) : ℝ := 3 * (a - b - c)^2

theorem custom_op_zero (x y z : ℝ) : 
  custom_op ((x - y - z)^2) ((y - x - z)^2) ((z - x - y)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_zero_l2465_246561


namespace NUMINAMATH_CALUDE_complement_of_union_l2465_246515

def I : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {0, 3, 5, 7}

theorem complement_of_union : (I \ (M ∪ N)) = {6, 8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2465_246515


namespace NUMINAMATH_CALUDE_emily_coloring_books_l2465_246581

/-- The number of coloring books Emily gave away -/
def books_given_away : ℕ := 2

/-- The initial number of coloring books Emily had -/
def initial_books : ℕ := 7

/-- The number of coloring books Emily bought -/
def books_bought : ℕ := 14

/-- The final number of coloring books Emily has -/
def final_books : ℕ := 19

theorem emily_coloring_books : 
  initial_books - books_given_away + books_bought = final_books :=
sorry

end NUMINAMATH_CALUDE_emily_coloring_books_l2465_246581


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l2465_246511

/-- The arithmetic square root of a non-negative real number -/
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

/-- The arithmetic square root is non-negative -/
axiom arithmetic_sqrt_nonneg (x : ℝ) : x ≥ 0 → arithmetic_sqrt x ≥ 0

/-- The arithmetic square root of 9 is 3 -/
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l2465_246511


namespace NUMINAMATH_CALUDE_coins_player1_l2465_246578

/-- Represents the number of sectors and players -/
def n : ℕ := 9

/-- Represents the number of rotations -/
def rotations : ℕ := 11

/-- Represents the coins received by player 4 -/
def coins_player4 : ℕ := 90

/-- Represents the coins received by player 8 -/
def coins_player8 : ℕ := 35

/-- Theorem stating the number of coins received by player 1 -/
theorem coins_player1 (h1 : n = 9) (h2 : rotations = 11) 
  (h3 : coins_player4 = 90) (h4 : coins_player8 = 35) : 
  ∃ (coins_player1 : ℕ), coins_player1 = 57 :=
sorry


end NUMINAMATH_CALUDE_coins_player1_l2465_246578


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2465_246575

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : 
  1 / x + 1 / y = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2465_246575


namespace NUMINAMATH_CALUDE_only_solutions_l2465_246559

/-- A four-digit number is composed of two two-digit numbers x and y -/
def is_valid_four_digit (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≥ 10 ∧ x < 100 ∧ y ≥ 10 ∧ y < 100 ∧ n = 100 * x + y

/-- The condition that the square of the sum of x and y equals the four-digit number -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ (x y : ℕ), is_valid_four_digit n ∧ (x + y)^2 = n

/-- The theorem stating that 3025 and 2025 are the only solutions -/
theorem only_solutions : ∀ (n : ℕ), satisfies_condition n ↔ (n = 3025 ∨ n = 2025) :=
sorry

end NUMINAMATH_CALUDE_only_solutions_l2465_246559


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2465_246509

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, x > 0 → x > -1) ∧ (∃ x, x > -1 ∧ ¬(x > 0)) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2465_246509


namespace NUMINAMATH_CALUDE_opposite_points_theorem_l2465_246557

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Given two points on a number line, proves that if they represent opposite numbers, 
    their distance is 8, and the first point is to the left of the second, 
    then they represent -4 and 4 respectively -/
theorem opposite_points_theorem (A B : Point) : 
  A.value + B.value = 0 →  -- A and B represent opposite numbers
  |A.value - B.value| = 8 →  -- Distance between A and B is 8
  A.value < B.value →  -- A is to the left of B
  A.value = -4 ∧ B.value = 4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_points_theorem_l2465_246557


namespace NUMINAMATH_CALUDE_q_must_be_true_l2465_246536

theorem q_must_be_true (h1 : ¬p) (h2 : p ∨ q) : q :=
sorry

end NUMINAMATH_CALUDE_q_must_be_true_l2465_246536


namespace NUMINAMATH_CALUDE_ratio_not_always_constant_l2465_246547

theorem ratio_not_always_constant : ∃ (f g : ℝ → ℝ), ¬(∀ x : ℝ, ∃ c : ℝ, f x = c * g x) :=
sorry

end NUMINAMATH_CALUDE_ratio_not_always_constant_l2465_246547


namespace NUMINAMATH_CALUDE_exists_n_with_uniform_200th_digit_distribution_l2465_246517

def digit_at_position (x : ℝ) (pos : ℕ) : ℕ := sorry

def count_occurrences (digit : ℕ) (numbers : List ℝ) (pos : ℕ) : ℕ := sorry

theorem exists_n_with_uniform_200th_digit_distribution :
  ∃ (n : ℕ+),
    ∀ (digit : Fin 10),
      count_occurrences digit.val
        (List.map (λ k => Real.sqrt (n.val + k)) (List.range 1000))
        200 = 100 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_with_uniform_200th_digit_distribution_l2465_246517


namespace NUMINAMATH_CALUDE_peanuts_in_box_l2465_246588

/-- The number of peanuts in a box after adding more -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of peanuts is 10 when starting with 4 and adding 6 -/
theorem peanuts_in_box : total_peanuts 4 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l2465_246588


namespace NUMINAMATH_CALUDE_investment_ratio_is_seven_to_five_l2465_246546

/-- Represents the investment and profit information for two partners -/
structure PartnerInvestment where
  profit_ratio : Rat
  p_investment_time : ℕ
  q_investment_time : ℕ

/-- Calculates the investment ratio given the profit ratio and investment times -/
def investment_ratio (info : PartnerInvestment) : Rat :=
  (info.profit_ratio * info.q_investment_time) / info.p_investment_time

/-- Theorem stating that given the specified conditions, the investment ratio is 7:5 -/
theorem investment_ratio_is_seven_to_five (info : PartnerInvestment) 
  (h1 : info.profit_ratio = 7 / 10)
  (h2 : info.p_investment_time = 2)
  (h3 : info.q_investment_time = 4) :
  investment_ratio info = 7 / 5 := by
  sorry

#eval investment_ratio { profit_ratio := 7 / 10, p_investment_time := 2, q_investment_time := 4 }

end NUMINAMATH_CALUDE_investment_ratio_is_seven_to_five_l2465_246546


namespace NUMINAMATH_CALUDE_fortieth_number_in_sampling_l2465_246580

/-- Represents the systematic sampling process in a math competition. -/
def systematicSampling (totalStudents : Nat) (sampleSize : Nat) (firstSelected : Nat) : Nat → Nat :=
  fun n => firstSelected + (totalStudents / sampleSize) * (n - 1)

/-- Theorem stating the 40th number in the systematic sampling. -/
theorem fortieth_number_in_sampling :
  systematicSampling 1000 50 15 40 = 795 := by
  sorry

end NUMINAMATH_CALUDE_fortieth_number_in_sampling_l2465_246580


namespace NUMINAMATH_CALUDE_cos_2theta_minus_7pi_over_2_l2465_246541

theorem cos_2theta_minus_7pi_over_2 (θ : ℝ) (h : Real.sin θ + Real.cos θ = -Real.sqrt 5 / 3) :
  Real.cos (2 * θ - 7 * Real.pi / 2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_minus_7pi_over_2_l2465_246541


namespace NUMINAMATH_CALUDE_total_stocking_stuffers_l2465_246595

def num_kids : ℕ := 3
def candy_canes_per_stocking : ℕ := 4
def beanie_babies_per_stocking : ℕ := 2
def books_per_stocking : ℕ := 1
def small_toys_per_stocking : ℕ := 3
def gift_cards_per_stocking : ℕ := 1

def items_per_stocking : ℕ := 
  candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking + 
  small_toys_per_stocking + gift_cards_per_stocking

theorem total_stocking_stuffers : 
  num_kids * items_per_stocking = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_stocking_stuffers_l2465_246595


namespace NUMINAMATH_CALUDE_max_value_z_l2465_246520

theorem max_value_z (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y ≤ 2) (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3*x - y ∧ z ≤ 6 ∧ ∃ (x' y' : ℝ), x' - y' ≥ 0 ∧ x' + y' ≤ 2 ∧ y' ≥ 0 ∧ 3*x' - y' = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l2465_246520


namespace NUMINAMATH_CALUDE_evaluate_expression_l2465_246522

theorem evaluate_expression (a b : ℝ) (ha : a = 3) (hb : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2465_246522


namespace NUMINAMATH_CALUDE_angle_D_value_l2465_246525

theorem angle_D_value (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 40)
  (h4 : B + C = 130) :
  D = 40 := by sorry

end NUMINAMATH_CALUDE_angle_D_value_l2465_246525


namespace NUMINAMATH_CALUDE_last_ten_shots_made_l2465_246582

/-- Represents the number of shots made in a sequence of basketball shots -/
structure BasketballShots where
  total : ℕ
  made : ℕ
  percentage : ℚ
  inv_percentage_def : percentage = made / total

/-- The problem statement -/
theorem last_ten_shots_made 
  (initial : BasketballShots)
  (final : BasketballShots)
  (h1 : initial.total = 30)
  (h2 : initial.percentage = 3/5)
  (h3 : final.total = initial.total + 10)
  (h4 : final.percentage = 29/50)
  : final.made - initial.made = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_ten_shots_made_l2465_246582


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l2465_246576

/-- The number of trailing zeros in a positive integer -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 125 and 360 -/
def product : ℕ := 125 * 360

/-- Theorem: The number of trailing zeros in the product of 125 and 360 is 3 -/
theorem product_trailing_zeros : trailingZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l2465_246576


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2465_246526

-- Define the function f(x) = x^2 + 2x - 3
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | f x < 0} = {x : ℝ | -3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2465_246526


namespace NUMINAMATH_CALUDE_park_fencing_cost_l2465_246532

/-- The cost of fencing a rectangular park -/
theorem park_fencing_cost 
  (length width : ℝ) 
  (area : ℝ) 
  (fencing_cost_paise : ℝ) : 
  length / width = 3 / 2 →
  area = length * width →
  area = 2400 →
  fencing_cost_paise = 50 →
  2 * (length + width) * (fencing_cost_paise / 100) = 100 := by
  sorry


end NUMINAMATH_CALUDE_park_fencing_cost_l2465_246532


namespace NUMINAMATH_CALUDE_distinct_roots_iff_m_less_than_one_zero_root_implies_m_values_and_other_root_l2465_246597

/-- The quadratic equation x^2 + 2(m-1)x + m^2 - 1 = 0 -/
def quadratic_equation (m x : ℝ) : Prop :=
  x^2 + 2*(m-1)*x + m^2 - 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  4*(m-1)^2 - 4*(m^2-1)

theorem distinct_roots_iff_m_less_than_one (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y) ↔ m < 1 :=
sorry

theorem zero_root_implies_m_values_and_other_root (m : ℝ) :
  (quadratic_equation m 0) →
  ((m = 1 ∧ ∀ x : ℝ, quadratic_equation m x → x = 0) ∨
   (m = -1 ∧ ∃ x : ℝ, x = 4 ∧ quadratic_equation m x)) :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_iff_m_less_than_one_zero_root_implies_m_values_and_other_root_l2465_246597


namespace NUMINAMATH_CALUDE_chair_cost_l2465_246592

theorem chair_cost (total_cost : ℝ) (table_cost : ℝ) (num_chairs : ℕ) 
  (h1 : total_cost = 135)
  (h2 : table_cost = 55)
  (h3 : num_chairs = 4) :
  (total_cost - table_cost) / num_chairs = 20 := by
  sorry

end NUMINAMATH_CALUDE_chair_cost_l2465_246592


namespace NUMINAMATH_CALUDE_derivative_f_at_3_l2465_246513

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem derivative_f_at_3 : 
  deriv f 3 = 1 / (3 * Real.log 3) := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_3_l2465_246513


namespace NUMINAMATH_CALUDE_percent_difference_l2465_246568

theorem percent_difference (y q w z : ℝ) 
  (hw : w = 0.6 * q)
  (hq : q = 0.6 * y)
  (hz : z = 0.54 * y) :
  z = w * 1.5 := by
sorry

end NUMINAMATH_CALUDE_percent_difference_l2465_246568


namespace NUMINAMATH_CALUDE_contrapositive_exponential_l2465_246590

theorem contrapositive_exponential (a b : ℝ) :
  (∀ a b, a > b → 2^a > 2^b) ↔ (∀ a b, 2^a ≤ 2^b → a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_exponential_l2465_246590


namespace NUMINAMATH_CALUDE_max_value_inequality_l2465_246505

theorem max_value_inequality (x y : ℝ) (hx : |x - 1| ≤ 1) (hy : |y - 2| ≤ 1) :
  |x - y + 1| ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), |x₀ - 1| ≤ 1 ∧ |y₀ - 2| ≤ 1 ∧ |x₀ - y₀ + 1| = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2465_246505


namespace NUMINAMATH_CALUDE_insulation_cost_example_l2465_246500

/-- Calculates the total cost of insulating a rectangular tank with two layers -/
def insulation_cost (length width height : ℝ) (cost1 cost2 : ℝ) : ℝ :=
  let surface_area := 2 * (length * width + length * height + width * height)
  surface_area * (cost1 + cost2)

/-- Theorem: The cost of insulating a 4x5x2 tank with $20 and $15 per sq ft layers is $2660 -/
theorem insulation_cost_example : insulation_cost 4 5 2 20 15 = 2660 := by
  sorry

end NUMINAMATH_CALUDE_insulation_cost_example_l2465_246500


namespace NUMINAMATH_CALUDE_square_fold_angle_l2465_246564

/-- A square with side length 1 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1))

/-- The angle formed by two lines after folding a square along its diagonal -/
def dihedral_angle (s : Square) : ℝ := sorry

/-- Theorem: The dihedral angle formed by folding a square along its diagonal is 60° -/
theorem square_fold_angle (s : Square) : dihedral_angle s = 60 * π / 180 := by sorry

end NUMINAMATH_CALUDE_square_fold_angle_l2465_246564


namespace NUMINAMATH_CALUDE_initial_roses_count_l2465_246566

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses added to the vase -/
def added_roses : ℕ := 13

/-- The total number of roses in the vase after adding -/
def total_roses : ℕ := 20

/-- Theorem stating that the initial number of roses is 7 -/
theorem initial_roses_count : initial_roses = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_roses_count_l2465_246566


namespace NUMINAMATH_CALUDE_two_thirds_bucket_fill_time_l2465_246599

/-- Given a bucket that takes 3 minutes to fill completely, 
    prove that it takes 2 minutes to fill two-thirds of the bucket. -/
theorem two_thirds_bucket_fill_time :
  let total_time : ℝ := 3  -- Time to fill the entire bucket
  let fraction_to_fill : ℝ := 2/3  -- Fraction of the bucket we want to fill
  (fraction_to_fill * total_time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_bucket_fill_time_l2465_246599


namespace NUMINAMATH_CALUDE_reciprocal_sum_equals_33_16_l2465_246586

/-- The decimal representation of 0.151515... -/
def recurring_15 : ℚ := 5 / 33

/-- The decimal representation of 0.333... -/
def recurring_3 : ℚ := 1 / 3

/-- The reciprocal of the sum of recurring_15 and recurring_3 -/
def reciprocal_sum : ℚ := (recurring_15 + recurring_3)⁻¹

theorem reciprocal_sum_equals_33_16 : reciprocal_sum = 33 / 16 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equals_33_16_l2465_246586


namespace NUMINAMATH_CALUDE_inequality_solution_l2465_246571

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 3)) < 1 / 4) ↔
  (x < -3 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2465_246571


namespace NUMINAMATH_CALUDE_translated_quadratic_vertex_l2465_246562

/-- The vertex of a quadratic function translated to the right by 3 units -/
theorem translated_quadratic_vertex (f g : ℝ → ℝ) (h : ℝ) :
  (∀ x, f x = 2 * (x - 1)^2 - 3) →
  (∀ x, g x = 2 * (x - 4)^2 - 3) →
  (∀ x, g x = f (x - 3)) →
  h = 4 →
  (∀ x, g x ≥ g h) →
  g h = -3 :=
by sorry

end NUMINAMATH_CALUDE_translated_quadratic_vertex_l2465_246562


namespace NUMINAMATH_CALUDE_coin_toss_probability_l2465_246555

theorem coin_toss_probability : 
  let n : ℕ := 5  -- Total number of coins
  let k : ℕ := 3  -- Number of heads we want
  let p : ℚ := 1/2  -- Probability of getting heads on a single toss
  Nat.choose n k * p^n = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l2465_246555


namespace NUMINAMATH_CALUDE_small_branches_count_l2465_246553

/-- Represents the structure of a plant with branches and small branches. -/
structure Plant where
  small_branches_per_branch : ℕ
  total_count : ℕ

/-- The plant satisfies the given conditions. -/
def valid_plant (p : Plant) : Prop :=
  p.total_count = 1 + p.small_branches_per_branch + p.small_branches_per_branch^2

/-- Theorem: Given the conditions, the number of small branches per branch is 9. -/
theorem small_branches_count (p : Plant) 
    (h : valid_plant p) 
    (h_total : p.total_count = 91) : 
  p.small_branches_per_branch = 9 := by
  sorry

#check small_branches_count

end NUMINAMATH_CALUDE_small_branches_count_l2465_246553


namespace NUMINAMATH_CALUDE_absolute_value_equation_l2465_246538

theorem absolute_value_equation (x : ℝ) : 
  |3990 * x + 1995| = 1995 → x = 0 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l2465_246538


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2465_246556

/-- Two lines are parallel if their slopes are equal and they are not the same line -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1 ∧ a1 * c2 ≠ a2 * c1

/-- The theorem stating that if two lines (3+m)x+4y=5-3m and 2x+(5+m)y=8 are parallel, then m = -7 -/
theorem parallel_lines_m_value (m : ℝ) :
  are_parallel (3 + m) 4 (3*m - 5) 2 (5 + m) (-8) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l2465_246556


namespace NUMINAMATH_CALUDE_root_sum_product_reciprocal_sum_l2465_246545

theorem root_sum_product_reciprocal_sum (α β : ℝ) (x₁ x₂ x₃ : ℝ) :
  α ≠ 0 →
  β ≠ 0 →
  (α * x₁^3 - α * x₁^2 + β * x₁ + β = 0) →
  (α * x₂^3 - α * x₂^2 + β * x₂ + β = 0) →
  (α * x₃^3 - α * x₃^2 + β * x₃ + β = 0) →
  (x₁ + x₂ + x₃) * (1/x₁ + 1/x₂ + 1/x₃) = -1 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_product_reciprocal_sum_l2465_246545


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2465_246501

theorem ratio_x_to_y (x y : ℝ) (h : (8*x - 5*y) / (11*x - 3*y) = 4/7) : 
  x/y = 23/12 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2465_246501


namespace NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_10_l2465_246563

-- Define a prime number with 2009 digits
def largest_prime_2009_digits : Nat :=
  sorry

-- Define the property of being the largest prime with 2009 digits
def is_largest_prime_2009_digits (p : Nat) : Prop :=
  Nat.Prime p ∧ 
  (Nat.digits 10 p).length = 2009 ∧
  ∀ q, Nat.Prime q → (Nat.digits 10 q).length = 2009 → q ≤ p

-- Theorem statement
theorem smallest_k_for_divisibility_by_10 (p : Nat) 
  (h_p : is_largest_prime_2009_digits p) : 
  (∃ k : Nat, k > 0 ∧ (p^2 - k) % 10 = 0) ∧
  (∀ k : Nat, k > 0 → (p^2 - k) % 10 = 0 → k ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_10_l2465_246563


namespace NUMINAMATH_CALUDE_associate_professor_pencils_l2465_246539

theorem associate_professor_pencils 
  (total_people : ℕ) 
  (total_pencils : ℕ) 
  (total_charts : ℕ) 
  (associate_pencils : ℕ) 
  (h1 : total_people = 6) 
  (h2 : total_pencils = 10) 
  (h3 : total_charts = 8) :
  let associate_count := total_people - (total_charts - total_people) / 2
  associate_pencils = (total_pencils - (total_people - associate_count)) / associate_count →
  associate_pencils = 2 := by
sorry

end NUMINAMATH_CALUDE_associate_professor_pencils_l2465_246539


namespace NUMINAMATH_CALUDE_prob_not_six_four_dice_value_l2465_246569

/-- The probability that (a-6)(b-6)(c-6)(d-6) ≠ 0 when four standard dice are tossed -/
def prob_not_six_four_dice : ℚ :=
  625 / 1296

/-- Theorem stating that the probability of (a-6)(b-6)(c-6)(d-6) ≠ 0 
    when four standard dice are tossed is equal to 625/1296 -/
theorem prob_not_six_four_dice_value : 
  prob_not_six_four_dice = 625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_six_four_dice_value_l2465_246569


namespace NUMINAMATH_CALUDE_tank_problem_l2465_246502

theorem tank_problem (tank1_capacity tank2_capacity tank3_capacity : ℚ)
  (tank1_fill_ratio tank2_fill_ratio : ℚ) (total_water : ℚ) :
  tank1_capacity = 7000 →
  tank2_capacity = 5000 →
  tank3_capacity = 3000 →
  tank1_fill_ratio = 3/4 →
  tank2_fill_ratio = 4/5 →
  total_water = 10850 →
  (total_water - (tank1_capacity * tank1_fill_ratio + tank2_capacity * tank2_fill_ratio)) / tank3_capacity = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_tank_problem_l2465_246502


namespace NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_alpha_l2465_246560

theorem cos_five_pi_sixth_minus_alpha (α : ℝ) (h : Real.cos (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 - α) = -(Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_alpha_l2465_246560


namespace NUMINAMATH_CALUDE_seconds_in_day_scientific_notation_l2465_246551

/-- The number of seconds in a day -/
def seconds_in_day : ℕ := 86400

/-- Scientific notation representation of seconds in a day -/
def scientific_notation : ℝ := 8.64 * (10 ^ 4)

theorem seconds_in_day_scientific_notation :
  (seconds_in_day : ℝ) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_seconds_in_day_scientific_notation_l2465_246551


namespace NUMINAMATH_CALUDE_accurate_number_range_l2465_246531

/-- The approximate number obtained by rounding -/
def approximate_number : ℝ := 0.270

/-- A number rounds to the given approximate number if it's within 0.0005 of it -/
def rounds_to (x : ℝ) : Prop :=
  x ≥ approximate_number - 0.0005 ∧ x < approximate_number + 0.0005

/-- The theorem stating the range of the accurate number -/
theorem accurate_number_range (a : ℝ) (h : rounds_to a) :
  a ≥ 0.2695 ∧ a < 0.2705 := by
  sorry

end NUMINAMATH_CALUDE_accurate_number_range_l2465_246531


namespace NUMINAMATH_CALUDE_olivia_change_olivia_change_proof_l2465_246514

/-- Calculates the change Olivia received after buying basketball and baseball cards -/
theorem olivia_change (basketball_packs : ℕ) (basketball_price : ℕ) 
  (baseball_decks : ℕ) (baseball_price : ℕ) (bill : ℕ) : ℕ :=
  let total_cost := basketball_packs * basketball_price + baseball_decks * baseball_price
  bill - total_cost

/-- Proves that Olivia received $24 in change -/
theorem olivia_change_proof :
  olivia_change 2 3 5 4 50 = 24 := by
  sorry

end NUMINAMATH_CALUDE_olivia_change_olivia_change_proof_l2465_246514


namespace NUMINAMATH_CALUDE_non_attacking_knights_count_l2465_246573

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)

/-- Represents a position on the chessboard --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Checks if two positions are distinct --/
def are_distinct (p1 p2 : Position) : Prop :=
  p1 ≠ p2

/-- Calculates the square of the distance between two positions --/
def distance_squared (p1 p2 : Position) : Nat :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Checks if two knights attack each other --/
def knights_attack (p1 p2 : Position) : Prop :=
  distance_squared p1 p2 = 5

/-- Counts the number of ways to place two knights that do not attack each other --/
def count_non_attacking_placements (board : Chessboard) : Nat :=
  sorry

theorem non_attacking_knights_count :
  ∀ (board : Chessboard),
    board.size = 8 →
    count_non_attacking_placements board = 1848 :=
by sorry

end NUMINAMATH_CALUDE_non_attacking_knights_count_l2465_246573


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2465_246565

/-- The area of the shaded region formed by two intersecting rectangles minus a circular cut-out -/
theorem shaded_area_calculation (rect1_width rect1_length rect2_width rect2_length : ℝ)
  (circle_radius : ℝ) (h1 : rect1_width = 3) (h2 : rect1_length = 12)
  (h3 : rect2_width = 4) (h4 : rect2_length = 7) (h5 : circle_radius = 1) :
  let rect1_area := rect1_width * rect1_length
  let rect2_area := rect2_width * rect2_length
  let overlap_area := min rect1_width rect2_width * min rect1_length rect2_length
  let circle_area := Real.pi * circle_radius^2
  rect1_area + rect2_area - overlap_area - circle_area = 64 - Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2465_246565


namespace NUMINAMATH_CALUDE_pencil_pen_problem_l2465_246554

theorem pencil_pen_problem (S : Finset Nat) (A B : Finset Nat) :
  S.card = 400 →
  A ⊆ S →
  B ⊆ S →
  A.card = 375 →
  B.card = 80 →
  S = A ∪ B →
  (A \ B).card = 320 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_problem_l2465_246554


namespace NUMINAMATH_CALUDE_min_cubes_for_box_l2465_246552

-- Define the box dimensions
def box_length : ℝ := 10
def box_width : ℝ := 18
def box_height : ℝ := 4

-- Define the volume of a single cube
def cube_volume : ℝ := 12

-- Theorem statement
theorem min_cubes_for_box :
  ⌈(box_length * box_width * box_height) / cube_volume⌉ = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_box_l2465_246552


namespace NUMINAMATH_CALUDE_simplify_expression_l2465_246583

theorem simplify_expression (a : ℝ) (h : a ≠ 1) :
  1 - (1 / (1 + (a + 1) / (1 - a))) = (1 + a) / 2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2465_246583


namespace NUMINAMATH_CALUDE_hundred_to_fifty_zeros_l2465_246574

theorem hundred_to_fifty_zeros (n : ℕ) : 100^50 = 10^100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_to_fifty_zeros_l2465_246574


namespace NUMINAMATH_CALUDE_replacement_process_terminates_l2465_246549

/-- Represents a finite sequence of binary digits (0 or 1) -/
def BinarySequence := List Nat

/-- The operation that replaces "01" with "1000" in a binary sequence -/
def replace_operation (seq : BinarySequence) : BinarySequence :=
  sorry

/-- Predicate to check if a sequence contains the subsequence "01" -/
def has_replaceable_subsequence (seq : BinarySequence) : Prop :=
  sorry

/-- The number of ones in a binary sequence -/
def count_ones (seq : BinarySequence) : Nat :=
  sorry

theorem replacement_process_terminates (initial_seq : BinarySequence) :
  ∃ (n : Nat), ∀ (m : Nat), m ≥ n →
    ¬(has_replaceable_subsequence ((replace_operation^[m]) initial_seq)) :=
  sorry

end NUMINAMATH_CALUDE_replacement_process_terminates_l2465_246549


namespace NUMINAMATH_CALUDE_billy_sleep_theorem_l2465_246534

def night1_sleep : ℕ := 6

def night2_sleep (n1 : ℕ) : ℕ := n1 + 2

def night3_sleep (n2 : ℕ) : ℕ := n2 / 2

def night4_sleep (n3 : ℕ) : ℕ := n3 * 3

def total_sleep (n1 n2 n3 n4 : ℕ) : ℕ := n1 + n2 + n3 + n4

theorem billy_sleep_theorem :
  let n1 := night1_sleep
  let n2 := night2_sleep n1
  let n3 := night3_sleep n2
  let n4 := night4_sleep n3
  total_sleep n1 n2 n3 n4 = 30 := by sorry

end NUMINAMATH_CALUDE_billy_sleep_theorem_l2465_246534


namespace NUMINAMATH_CALUDE_experiment_duration_in_seconds_l2465_246504

/-- Converts hours to seconds -/
def hoursToSeconds (hours : ℕ) : ℕ := hours * 3600

/-- Converts minutes to seconds -/
def minutesToSeconds (minutes : ℕ) : ℕ := minutes * 60

/-- Represents the duration of an experiment -/
structure ExperimentDuration where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the total seconds of an experiment duration -/
def totalSeconds (duration : ExperimentDuration) : ℕ :=
  hoursToSeconds duration.hours + minutesToSeconds duration.minutes + duration.seconds

/-- Theorem stating that the experiment lasting 2 hours, 45 minutes, and 30 seconds is equivalent to 9930 seconds -/
theorem experiment_duration_in_seconds :
  totalSeconds { hours := 2, minutes := 45, seconds := 30 } = 9930 := by
  sorry


end NUMINAMATH_CALUDE_experiment_duration_in_seconds_l2465_246504


namespace NUMINAMATH_CALUDE_calculator_key_functions_l2465_246558

/-- Represents the keys on a calculator --/
inductive CalculatorKey
  | ON_C
  | OFF
  | Other

/-- Represents the functions of calculator keys --/
inductive KeyFunction
  | ClearScreen
  | PowerOff
  | Other

/-- Maps calculator keys to their functions --/
def key_function : CalculatorKey → KeyFunction
  | CalculatorKey.ON_C => KeyFunction.ClearScreen
  | CalculatorKey.OFF => KeyFunction.PowerOff
  | CalculatorKey.Other => KeyFunction.Other

theorem calculator_key_functions :
  (key_function CalculatorKey.ON_C = KeyFunction.ClearScreen) ∧
  (key_function CalculatorKey.OFF = KeyFunction.PowerOff) :=
by sorry

end NUMINAMATH_CALUDE_calculator_key_functions_l2465_246558


namespace NUMINAMATH_CALUDE_solve_assignment_problem_l2465_246587

def assignment_problem (t : ℚ) : Prop :=
  let part1 := t
  let part2 := 2 * t
  let part3 := 3 / 4
  part1 + part2 + part3 = 2

theorem solve_assignment_problem :
  ∃ t : ℚ, assignment_problem t ∧ t = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_assignment_problem_l2465_246587


namespace NUMINAMATH_CALUDE_expression_factorization_l2465_246548

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 90 * x - 6) - (-3 * x^3 + 5 * x - 6) = 5 * x * (3 * x^2 + 17) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l2465_246548


namespace NUMINAMATH_CALUDE_binomial_problem_l2465_246507

def binomial_expansion (m n : ℕ) (x : ℝ) : ℝ :=
  (1 + m * x) ^ n

theorem binomial_problem (m n : ℕ) (h1 : m ≠ 0) (h2 : n ≥ 2) :
  (∃ k, k = 5 ∧ ∀ j, j ≠ k → Nat.choose n j ≤ Nat.choose n k) →
  (Nat.choose n 2 * m^2 = 9 * Nat.choose n 1 * m) →
  (m = 2 ∧ n = 10 ∧ (binomial_expansion m n (-9)) % 6 = 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_problem_l2465_246507


namespace NUMINAMATH_CALUDE_misha_dog_savings_l2465_246519

theorem misha_dog_savings (current_amount desired_amount : ℕ) 
  (h1 : current_amount = 34)
  (h2 : desired_amount = 47) :
  desired_amount - current_amount = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_misha_dog_savings_l2465_246519


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2465_246591

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) (final : ℕ) : 
  left = 6 → new = 42 → final = 47 → initial + new - left = final → initial = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2465_246591


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2465_246589

/-- Given an arithmetic sequence {a_n} where a_2 + a_8 = 16, prove that a_5 = 8 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : a 2 + a 8 = 16) : 
  a 5 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2465_246589


namespace NUMINAMATH_CALUDE_circle_division_relationship_l2465_246593

theorem circle_division_relationship (a k : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → (x = a ∨ x = -a ∨ y = k * x)) →
  a^2 * (k^2 + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_division_relationship_l2465_246593


namespace NUMINAMATH_CALUDE_inspector_examination_l2465_246510

/-- Given an inspector who rejects 0.02% of meters as defective and examined 10,000 meters to reject 2 meters,
    prove that to reject x meters, the inspector needs to examine 5000x meters. -/
theorem inspector_examination (x : ℝ) : 
  (2 / 10000 = x / (5000 * x)) → 5000 * x = (x * 10000) / 2 := by
sorry

end NUMINAMATH_CALUDE_inspector_examination_l2465_246510


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2465_246527

/-- 
For a two-digit number n where the unit's digit exceeds the 10's digit by 2, 
and n = 24, the product of n and the sum of its digits is 144.
-/
theorem two_digit_number_property : 
  ∀ (a b : ℕ), 
    (10 * a + b = 24) → 
    (b = a + 2) → 
    24 * (a + b) = 144 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2465_246527


namespace NUMINAMATH_CALUDE_green_ball_probability_l2465_246528

structure Container where
  red : ℕ
  green : ℕ

def X : Container := { red := 5, green := 7 }
def Y : Container := { red := 7, green := 5 }
def Z : Container := { red := 7, green := 5 }

def total_containers : ℕ := 3

def prob_select_container : ℚ := 1 / total_containers

def prob_green (c : Container) : ℚ := c.green / (c.red + c.green)

def total_prob_green : ℚ := 
  prob_select_container * prob_green X + 
  prob_select_container * prob_green Y + 
  prob_select_container * prob_green Z

theorem green_ball_probability : total_prob_green = 17 / 36 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2465_246528


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2465_246530

/-- Given a quadratic equation x^2 - 4x + m = 0 with one root x₁ = 1, 
    prove that the other root x₂ = 3 -/
theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 3 ∧ 
   ∀ x : ℝ, x^2 - 4*x + m = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2465_246530


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2465_246598

/-- A rectangular field with width one-third of its length and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_field_area (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  w = l / 3 → 2 * (w + l) = 72 → w * l = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2465_246598


namespace NUMINAMATH_CALUDE_jason_lost_three_balloons_l2465_246512

/-- The number of violet balloons Jason lost -/
def lost_balloons (initial current : ℕ) : ℕ := initial - current

/-- Proof that Jason lost 3 violet balloons -/
theorem jason_lost_three_balloons :
  let initial_violet : ℕ := 7
  let current_violet : ℕ := 4
  lost_balloons initial_violet current_violet = 3 := by
  sorry

end NUMINAMATH_CALUDE_jason_lost_three_balloons_l2465_246512


namespace NUMINAMATH_CALUDE_total_amount_after_three_years_l2465_246523

/-- Calculates the compound interest for a given principal, rate, and time --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The original bill amount --/
def initial_amount : ℝ := 350

/-- The interest rate for the first year --/
def first_year_rate : ℝ := 0.03

/-- The interest rate for the second and third years --/
def later_years_rate : ℝ := 0.05

/-- The total time period in years --/
def total_years : ℕ := 3

theorem total_amount_after_three_years :
  let amount_after_first_year := compound_interest initial_amount first_year_rate 1
  let final_amount := compound_interest amount_after_first_year later_years_rate 2
  ∃ ε > 0, |final_amount - 397.45| < ε :=
sorry

end NUMINAMATH_CALUDE_total_amount_after_three_years_l2465_246523


namespace NUMINAMATH_CALUDE_workbook_problems_l2465_246508

theorem workbook_problems (T : ℕ) : 
  (T : ℚ) / 2 + T / 4 + T / 6 + 20 = T → T = 240 := by
  sorry

end NUMINAMATH_CALUDE_workbook_problems_l2465_246508


namespace NUMINAMATH_CALUDE_resulting_solution_percentage_l2465_246506

/-- Calculates the percentage of chemicals in the resulting solution when a portion of a 90% solution is replaced with an equal amount of 20% solution. -/
theorem resulting_solution_percentage 
  (original_concentration : Real) 
  (replacement_concentration : Real)
  (replaced_portion : Real) :
  original_concentration = 0.9 →
  replacement_concentration = 0.2 →
  replaced_portion = 0.7142857142857143 →
  let remaining_portion := 1 - replaced_portion
  let chemicals_in_remaining := remaining_portion * original_concentration
  let chemicals_in_added := replaced_portion * replacement_concentration
  let total_chemicals := chemicals_in_remaining + chemicals_in_added
  let resulting_concentration := total_chemicals / 1
  resulting_concentration = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_resulting_solution_percentage_l2465_246506
