import Mathlib

namespace NUMINAMATH_CALUDE_median_invariant_after_remove_min_max_l247_24788

/-- A function that returns the median of a list of real numbers -/
def median (l : List ℝ) : ℝ := sorry

/-- A function that removes the minimum and maximum elements from a list -/
def removeMinMax (l : List ℝ) : List ℝ := sorry

theorem median_invariant_after_remove_min_max (data : List ℝ) :
  data.length > 2 →
  data.Nodup →
  median data = median (removeMinMax data) :=
sorry

end NUMINAMATH_CALUDE_median_invariant_after_remove_min_max_l247_24788


namespace NUMINAMATH_CALUDE_solution_exists_l247_24755

/-- The system of equations has at least one solution if and only if a is in the specified set -/
theorem solution_exists (a : ℝ) : 
  (∃ x y : ℝ, x - 1 = a * (y^3 - 1) ∧ 
              2 * x / (|y^3| + y^3) = Real.sqrt x ∧ 
              y > 0 ∧ x ≥ 0) ↔ 
  a < 0 ∨ (0 ≤ a ∧ a < 1) ∨ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_exists_l247_24755


namespace NUMINAMATH_CALUDE_calculate_female_students_l247_24758

/-- Given a population of students and a sample, calculate the number of female students in the population -/
theorem calculate_female_students 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (male_in_sample : ℕ) 
  (h1 : total_population = 2000) 
  (h2 : sample_size = 200) 
  (h3 : male_in_sample = 103) :
  (total_population - (male_in_sample * (total_population / sample_size))) = 970 := by
  sorry

#check calculate_female_students

end NUMINAMATH_CALUDE_calculate_female_students_l247_24758


namespace NUMINAMATH_CALUDE_complex_simplification_l247_24709

theorem complex_simplification :
  ((2 + Complex.I) ^ 200) / ((2 - Complex.I) ^ 200) = Complex.exp (200 * Complex.I * Complex.arctan (4 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l247_24709


namespace NUMINAMATH_CALUDE_count_nines_in_subtraction_l247_24794

/-- The number of occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The result of subtracting 101011 from 10000000000 -/
def subtraction_result : ℕ := 10000000000 - 101011

/-- Theorem stating that the number of 9's in the subtraction result is 8 -/
theorem count_nines_in_subtraction : countDigit subtraction_result 9 = 8 := by sorry

end NUMINAMATH_CALUDE_count_nines_in_subtraction_l247_24794


namespace NUMINAMATH_CALUDE_sum_of_digits_product_76_eights_76_fives_l247_24798

/-- Represents a number consisting of n repetitions of a single digit -/
def repeatedDigitNumber (digit : Nat) (n : Nat) : Nat :=
  digit * (10^n - 1) / 9

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved -/
theorem sum_of_digits_product_76_eights_76_fives : 
  sumOfDigits (repeatedDigitNumber 8 76 * repeatedDigitNumber 5 76) = 304 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_digits_product_76_eights_76_fives_l247_24798


namespace NUMINAMATH_CALUDE_equation_solution_l247_24714

-- Define the functions
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4
def g (x : ℝ) : ℝ := x^4 + 2*x^3 + x^2 + 11*x + 11
def h (x : ℝ) : ℝ := x + 1

-- Define the set of solutions
def solution_set : Set ℝ := {x | x = 1 ∨ x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2}

-- State the theorem
theorem equation_solution :
  ∀ x ∈ solution_set, ∃ y, f y = g x ∧ y = h x :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l247_24714


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l247_24737

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), C = Real.sqrt 2 ∧ 
  (∀ (x y z : ℝ), x^2 + y^2 + z^3 + 1 ≥ C * (x + y + z)) ∧
  (∀ (C' : ℝ), C' > C → ∃ (x y z : ℝ), x^2 + y^2 + z^3 + 1 < C' * (x + y + z)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l247_24737


namespace NUMINAMATH_CALUDE_base_value_l247_24743

/-- A triangle with specific side length properties -/
structure SpecificTriangle where
  left : ℝ
  right : ℝ
  base : ℝ
  sum_of_sides : left + right + base = 50
  right_longer : right = left + 2
  left_value : left = 12

theorem base_value (t : SpecificTriangle) : t.base = 24 := by
  sorry

end NUMINAMATH_CALUDE_base_value_l247_24743


namespace NUMINAMATH_CALUDE_office_age_problem_l247_24780

theorem office_age_problem (total_people : Nat) (group1_people : Nat) (group2_people : Nat)
  (total_avg_age : ℝ) (group1_avg_age : ℝ) (group2_avg_age : ℝ)
  (h1 : total_people = 16)
  (h2 : group1_people = 5)
  (h3 : group2_people = 9)
  (h4 : total_avg_age = 15)
  (h5 : group1_avg_age = 14)
  (h6 : group2_avg_age = 16) :
  (total_people : ℝ) * total_avg_age - 
  (group1_people : ℝ) * group1_avg_age - 
  (group2_people : ℝ) * group2_avg_age = 52 := by
sorry

end NUMINAMATH_CALUDE_office_age_problem_l247_24780


namespace NUMINAMATH_CALUDE_existence_of_special_number_l247_24787

/-- A function that computes the sum of digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number contains only digits 2-9. -/
def contains_only_2_to_9 (n : ℕ) : Prop := sorry

/-- The number of digits in a natural number. -/
def num_digits (n : ℕ) : ℕ := sorry

theorem existence_of_special_number :
  ∃ N : ℕ, 
    num_digits N = 2020 ∧ 
    contains_only_2_to_9 N ∧ 
    N % sum_of_digits N = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l247_24787


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l247_24701

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Check if a point lies on a line -/
def pointOnLine (x y : ℝ) (l : Line) : Prop :=
  y = l.slope * x + l.intercept

/-- The given line y = -2x + 1 -/
def givenLine : Line :=
  { slope := -2, intercept := 1 }

theorem parallel_line_through_point :
  ∃ (l : Line), parallel l givenLine ∧ pointOnLine (-1) 2 l ∧ l.slope * x + l.intercept = -2 * x :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l247_24701


namespace NUMINAMATH_CALUDE_inequality_solution_set_l247_24781

theorem inequality_solution_set (x : ℝ) :
  (∀ x, -x^2 - 3*x + 4 > 0 ↔ -4 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l247_24781


namespace NUMINAMATH_CALUDE_domain_sqrt_one_minus_x_squared_l247_24793

theorem domain_sqrt_one_minus_x_squared (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 ↔ 1 - x^2 ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_domain_sqrt_one_minus_x_squared_l247_24793


namespace NUMINAMATH_CALUDE_triangle_side_length_l247_24771

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 4 → b = 2 → Real.cos A = 1/4 → c^2 = a^2 + b^2 - 2*a*b*(Real.cos A) → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l247_24771


namespace NUMINAMATH_CALUDE_total_fish_cost_l247_24747

def fish_cost : ℕ := 4
def dog_fish : ℕ := 40
def cat_fish : ℕ := dog_fish / 2

theorem total_fish_cost : dog_fish * fish_cost + cat_fish * fish_cost = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_cost_l247_24747


namespace NUMINAMATH_CALUDE_no_nontrivial_solutions_l247_24731

theorem no_nontrivial_solutions (x y z t : ℤ) :
  x^2 = 2*y^2 ∧ x^4 + 3*y^4 + 27*z^4 = 9*t^4 → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nontrivial_solutions_l247_24731


namespace NUMINAMATH_CALUDE_joan_gave_25_marbles_l247_24738

/-- The number of yellow marbles Joan gave Sam -/
def marbles_from_joan (initial_yellow : ℝ) (final_yellow : ℕ) : ℝ :=
  final_yellow - initial_yellow

theorem joan_gave_25_marbles :
  let initial_yellow : ℝ := 86.0
  let final_yellow : ℕ := 111
  marbles_from_joan initial_yellow final_yellow = 25 := by
  sorry

end NUMINAMATH_CALUDE_joan_gave_25_marbles_l247_24738


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l247_24770

theorem factorization_of_quadratic (a : ℝ) : a^2 + 5*a = a*(a+5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l247_24770


namespace NUMINAMATH_CALUDE_total_weight_theorem_l247_24769

/-- The total weight of three balls -/
def total_weight (blue_weight brown_weight green_weight : ℝ) : ℝ :=
  blue_weight + brown_weight + green_weight

/-- Theorem: The total weight of the three balls is 9.12 + x -/
theorem total_weight_theorem (x : ℝ) :
  total_weight 6 3.12 x = 9.12 + x := by
  sorry

end NUMINAMATH_CALUDE_total_weight_theorem_l247_24769


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l247_24768

theorem complex_power_magnitude (z : ℂ) : z = (1 / 2 : ℂ) + (Complex.I * (Real.sqrt 3 / 2)) → Complex.abs (z ^ 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l247_24768


namespace NUMINAMATH_CALUDE_obtuse_angle_equation_l247_24708

theorem obtuse_angle_equation (α : Real) : 
  α > π / 2 ∧ α < π →
  Real.sin α * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 →
  α = 140 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_obtuse_angle_equation_l247_24708


namespace NUMINAMATH_CALUDE_base_8_4532_equals_2394_l247_24711

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4532_equals_2394 :
  base_8_to_10 [2, 3, 5, 4] = 2394 := by
  sorry

end NUMINAMATH_CALUDE_base_8_4532_equals_2394_l247_24711


namespace NUMINAMATH_CALUDE_orange_ring_weight_l247_24724

/-- The weight of the orange ring in an experiment, given the weights of other rings and the total weight -/
theorem orange_ring_weight 
  (total_weight : Float) 
  (purple_weight : Float) 
  (white_weight : Float) 
  (h1 : total_weight = 0.8333333333) 
  (h2 : purple_weight = 0.3333333333333333) 
  (h3 : white_weight = 0.4166666666666667) : 
  total_weight - purple_weight - white_weight = 0.0833333333 := by
  sorry

end NUMINAMATH_CALUDE_orange_ring_weight_l247_24724


namespace NUMINAMATH_CALUDE_greater_number_sum_and_difference_l247_24741

theorem greater_number_sum_and_difference (x y : ℝ) : 
  x + y = 30 → x - y = 6 → x > y → x = 18 := by sorry

end NUMINAMATH_CALUDE_greater_number_sum_and_difference_l247_24741


namespace NUMINAMATH_CALUDE_chess_tournament_results_l247_24799

/-- Represents a chess tournament with given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players
  total_score : ℕ
  one_player_score : ℕ
  h1 : total_score = 210
  h2 : one_player_score = 12
  h3 : n * (n - 1) = total_score

/-- Theorem stating the main results of the tournament analysis -/
theorem chess_tournament_results (t : ChessTournament) :
  (t.n = 15) ∧ 
  (∃ (max_squares : ℕ), max_squares = 33 ∧ 
    ∀ (squares : ℕ), (squares = number_of_squares_knight_can_reach_in_two_moves) → 
      squares ≤ max_squares) ∧
  (∃ (winner_score : ℕ), winner_score > t.one_player_score) :=
sorry

/-- Helper function to calculate the number of squares a knight can reach in two moves -/
def number_of_squares_knight_can_reach_in_two_moves : ℕ :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_results_l247_24799


namespace NUMINAMATH_CALUDE_salary_restoration_l247_24765

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) :
  let reduced_salary := 0.8 * original_salary
  let restored_salary := reduced_salary * 1.25
  restored_salary = original_salary := by
sorry

end NUMINAMATH_CALUDE_salary_restoration_l247_24765


namespace NUMINAMATH_CALUDE_find_number_l247_24790

theorem find_number : ∃ N : ℕ, N = (555 + 445) * (2 * (555 - 445)) + 70 ∧ N = 220070 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l247_24790


namespace NUMINAMATH_CALUDE_units_digit_of_n_l247_24735

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 31^4) (h2 : units_digit m = 6) : 
  units_digit n = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l247_24735


namespace NUMINAMATH_CALUDE_student_addition_mistake_l247_24764

theorem student_addition_mistake (a b : ℤ) :
  (a + 10 * b = 7182) ∧ (a + b = 3132) → (a = 2682 ∧ b = 450) := by
  sorry

end NUMINAMATH_CALUDE_student_addition_mistake_l247_24764


namespace NUMINAMATH_CALUDE_inequality_solution_l247_24739

theorem inequality_solution (a : ℝ) : 
  (∃ x : ℝ, 2 * x - (1/3) * a ≤ 0 ∧ x ≤ 2) → a = 12 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l247_24739


namespace NUMINAMATH_CALUDE_greatest_x_value_l247_24761

theorem greatest_x_value (x : ℤ) : 
  (2.134 * (10 : ℝ) ^ (x : ℝ) < 21000) ↔ x ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l247_24761


namespace NUMINAMATH_CALUDE_expected_value_biased_die_l247_24742

/-- A biased die with six faces and specified winning conditions -/
structure BiasedDie where
  /-- The probability of rolling each number is 1/6 -/
  prob : Fin 6 → ℚ
  prob_eq : ∀ i, prob i = 1/6
  /-- The winnings for each roll -/
  winnings : Fin 6 → ℚ
  /-- Rolling 1 or 2 wins $5 -/
  win_12 : winnings 0 = 5 ∧ winnings 1 = 5
  /-- Rolling 3 or 4 wins $0 -/
  win_34 : winnings 2 = 0 ∧ winnings 3 = 0
  /-- Rolling 5 or 6 loses $4 -/
  lose_56 : winnings 4 = -4 ∧ winnings 5 = -4

/-- The expected value of winnings after one roll of the biased die is 1/3 -/
theorem expected_value_biased_die (d : BiasedDie) : 
  (Finset.univ.sum fun i => d.prob i * d.winnings i) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_biased_die_l247_24742


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l247_24784

universe u

def U : Set Nat := {1, 2, 3, 4}
def P : Set Nat := {2, 3, 4}
def Q : Set Nat := {1, 2}

theorem complement_P_intersect_Q :
  (Set.univ \ P) ∩ Q = {1} := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l247_24784


namespace NUMINAMATH_CALUDE_prob_select_boy_is_correct_l247_24728

/-- Represents the number of boys in the calligraphy group -/
def calligraphy_boys : ℕ := 6

/-- Represents the number of girls in the calligraphy group -/
def calligraphy_girls : ℕ := 4

/-- Represents the number of boys in the original art group -/
def art_boys : ℕ := 5

/-- Represents the number of girls in the original art group -/
def art_girls : ℕ := 5

/-- Represents the number of people selected from the calligraphy group -/
def selected_from_calligraphy : ℕ := 2

/-- Calculates the probability of selecting a boy from the new art group -/
def prob_select_boy : ℚ := 31/60

theorem prob_select_boy_is_correct :
  prob_select_boy = 31/60 := by sorry

end NUMINAMATH_CALUDE_prob_select_boy_is_correct_l247_24728


namespace NUMINAMATH_CALUDE_adam_figurines_count_l247_24726

/-- The number of figurines that can be made from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be made from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be made from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of basswood blocks Adam owns -/
def basswood_blocks : ℕ := 15

/-- The number of butternut wood blocks Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ :=
  basswood_blocks * basswood_figurines +
  butternut_blocks * butternut_figurines +
  aspen_blocks * aspen_figurines

theorem adam_figurines_count :
  total_figurines = 245 := by
  sorry

end NUMINAMATH_CALUDE_adam_figurines_count_l247_24726


namespace NUMINAMATH_CALUDE_balance_theorem_l247_24734

/-- Represents the balance between shapes -/
structure Balance where
  triangle : ℚ
  diamond : ℚ
  circle : ℚ

/-- First balance equation: 5 triangles + 2 diamonds = 12 circles -/
def balance1 : Balance := { triangle := 5, diamond := 2, circle := 12 }

/-- Second balance equation: 1 triangle = 1 diamond + 3 circles -/
def balance2 : Balance := { triangle := 1, diamond := 1, circle := 3 }

/-- The balance we want to prove: 4 diamonds = 12/7 circles -/
def target_balance : Balance := { triangle := 0, diamond := 4, circle := 12/7 }

/-- Checks if two balances are equivalent -/
def is_equivalent (b1 b2 : Balance) : Prop :=
  b1.triangle / b2.triangle = b1.diamond / b2.diamond ∧
  b1.triangle / b2.triangle = b1.circle / b2.circle

/-- The main theorem to prove -/
theorem balance_theorem (b1 b2 : Balance) (h1 : is_equivalent b1 balance1) 
    (h2 : is_equivalent b2 balance2) : 
  is_equivalent target_balance { triangle := 0, diamond := 1, circle := 3/7 } := by
  sorry

end NUMINAMATH_CALUDE_balance_theorem_l247_24734


namespace NUMINAMATH_CALUDE_blue_balls_count_l247_24733

theorem blue_balls_count (total : ℕ) 
  (h_green : (1 : ℚ) / 4 * total = (total / 4 : ℕ))
  (h_blue : (1 : ℚ) / 8 * total = (total / 8 : ℕ))
  (h_yellow : (1 : ℚ) / 12 * total = (total / 12 : ℕ))
  (h_white : total - (total / 4 + total / 8 + total / 12) = 26) :
  total / 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l247_24733


namespace NUMINAMATH_CALUDE_exam_time_ratio_l247_24783

theorem exam_time_ratio :
  let total_time : ℕ := 3 * 60  -- 3 hours in minutes
  let time_type_a : ℕ := 120    -- Time spent on type A problems in minutes
  let time_type_b : ℕ := total_time - time_type_a  -- Time spent on type B problems
  (time_type_a : ℚ) / time_type_b = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_ratio_l247_24783


namespace NUMINAMATH_CALUDE_faster_train_speed_l247_24746

theorem faster_train_speed 
  (train_length : ℝ) 
  (speed_difference : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 75) 
  (h2 : speed_difference = 36) 
  (h3 : passing_time = 54) : 
  ∃ (faster_speed : ℝ), faster_speed = 46 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l247_24746


namespace NUMINAMATH_CALUDE_sqrt_three_properties_l247_24704

theorem sqrt_three_properties : ∃ x : ℝ, Irrational x ∧ 0 < x ∧ x < 3 :=
  by
  use Real.sqrt 3
  sorry

end NUMINAMATH_CALUDE_sqrt_three_properties_l247_24704


namespace NUMINAMATH_CALUDE_georges_initial_money_l247_24706

theorem georges_initial_money (shirt_cost sock_cost money_left : ℕ) :
  shirt_cost = 24 →
  sock_cost = 11 →
  money_left = 65 →
  shirt_cost + sock_cost + money_left = 100 :=
by sorry

end NUMINAMATH_CALUDE_georges_initial_money_l247_24706


namespace NUMINAMATH_CALUDE_pastor_prayer_ratio_l247_24776

/-- Represents the number of prayers for a pastor on a given day --/
structure DailyPrayers where
  weekday : ℕ
  sunday : ℕ

/-- Represents the total prayers for a pastor in a week --/
def WeeklyPrayers (d : DailyPrayers) : ℕ := 6 * d.weekday + d.sunday

/-- Pastor Paul's prayer schedule --/
def paul : DailyPrayers :=
  { weekday := 20
    sunday := 40 }

/-- Pastor Bruce's prayer schedule --/
def bruce : DailyPrayers :=
  { weekday := paul.weekday / 2
    sunday := WeeklyPrayers paul - WeeklyPrayers { weekday := paul.weekday / 2, sunday := 0 } - 20 }

theorem pastor_prayer_ratio :
  bruce.sunday / paul.sunday = 2 := by sorry

end NUMINAMATH_CALUDE_pastor_prayer_ratio_l247_24776


namespace NUMINAMATH_CALUDE_greatest_common_multiple_8_12_under_90_l247_24702

theorem greatest_common_multiple_8_12_under_90 : 
  ∃ (n : ℕ), n = 72 ∧ 
  (∀ m : ℕ, m < 90 → m % 8 = 0 → m % 12 = 0 → m ≤ n) ∧
  72 % 8 = 0 ∧ 72 % 12 = 0 ∧ 72 < 90 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_8_12_under_90_l247_24702


namespace NUMINAMATH_CALUDE_coin_distribution_l247_24791

theorem coin_distribution (x y k : ℕ) (hxy : x + y = 81) (hne : x ≠ y) 
  (hsq : x^2 - y^2 = k * (x - y)) : k = 81 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_l247_24791


namespace NUMINAMATH_CALUDE_distance_AB_is_130_l247_24712

-- Define the speeds of the three people
def speed_A : ℝ := 3
def speed_B : ℝ := 2
def speed_C : ℝ := 1

-- Define the initial distance traveled by A
def initial_distance_A : ℝ := 50

-- Define the distance between C and D
def distance_CD : ℝ := 12

-- Theorem statement
theorem distance_AB_is_130 :
  let total_distance := 4 * (speed_A + speed_B + speed_C) * distance_CD + initial_distance_A
  total_distance = 130 := by sorry

end NUMINAMATH_CALUDE_distance_AB_is_130_l247_24712


namespace NUMINAMATH_CALUDE_cubic_polynomial_evaluation_l247_24750

theorem cubic_polynomial_evaluation : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_evaluation_l247_24750


namespace NUMINAMATH_CALUDE_congruence_problem_l247_24754

theorem congruence_problem : ∃! n : ℕ, n ≤ 14 ∧ n ≡ 8657 [ZMOD 15] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l247_24754


namespace NUMINAMATH_CALUDE_sine_inequality_l247_24725

theorem sine_inequality (x : Real) (h : 0 < x ∧ x < Real.pi / 4) :
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) :=
by sorry

end NUMINAMATH_CALUDE_sine_inequality_l247_24725


namespace NUMINAMATH_CALUDE_two_digit_S_equals_50_l247_24762

/-- R(n) is the sum of remainders when n is divided by 2, 3, 4, 5, and 6 -/
def R (n : ℕ) : ℕ :=
  n % 2 + n % 3 + n % 4 + n % 5 + n % 6

/-- S(n) is defined as R(n) + R(n+2) -/
def S (n : ℕ) : ℕ :=
  R n + R (n + 2)

/-- A two-digit number is between 10 and 99, inclusive -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- There are exactly 2 two-digit integers n such that S(n) = 50 -/
theorem two_digit_S_equals_50 :
  ∃! (count : ℕ), count = (Finset.filter (fun n => S n = 50) (Finset.range 90)).card ∧ count = 2 :=
sorry

end NUMINAMATH_CALUDE_two_digit_S_equals_50_l247_24762


namespace NUMINAMATH_CALUDE_x_value_l247_24759

theorem x_value (x : Real) : 
  Real.sin (π / 2 - x) = -Real.sqrt 3 / 2 → 
  π < x → 
  x < 2 * π → 
  x = 7 * π / 6 := by
sorry

end NUMINAMATH_CALUDE_x_value_l247_24759


namespace NUMINAMATH_CALUDE_find_B_l247_24717

-- Define the polynomial g(x)
def g (A B C D x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

-- State the theorem
theorem find_B :
  ∀ (A B C D : ℝ),
  (∀ x : ℝ, g A B C D x = 0 ↔ x = -2 ∨ x = 1 ∨ x = 2) →
  g A B C D 0 = -8 →
  B = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_find_B_l247_24717


namespace NUMINAMATH_CALUDE_cubic_identity_for_fifty_l247_24722

theorem cubic_identity_for_fifty : 50^3 + 3*(50^2) + 3*50 + 1 = 261051 := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_for_fifty_l247_24722


namespace NUMINAMATH_CALUDE_triangle_side_length_l247_24740

theorem triangle_side_length (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let angleB := Real.arccos ((BC^2 + (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))^2 - (Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2))^2) / (2 * BC * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)))
  let area := (1/2) * BC * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sin angleB
  BC = 1 → angleB = π/3 → area = Real.sqrt 3 → 
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l247_24740


namespace NUMINAMATH_CALUDE_cosine_equation_solutions_l247_24705

theorem cosine_equation_solutions :
  ∃! (n : ℕ), ∃ (S : Finset ℝ),
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
    (∀ x ∈ S, 3 * (Real.cos x)^3 - 7 * (Real.cos x)^2 + 3 * Real.cos x = 0) ∧
    Finset.card S = n ∧
    n = 4 :=
by sorry

end NUMINAMATH_CALUDE_cosine_equation_solutions_l247_24705


namespace NUMINAMATH_CALUDE_quadratic_roots_implications_l247_24713

theorem quadratic_roots_implications (a b c : ℝ) 
  (h_roots : ∃ (α β : ℝ), α > 0 ∧ β ≠ 0 ∧ 
    (∀ x : ℂ, x^2 - (a + b + c) * x + (a * b + b * c + c * a) = 0 ↔ x = α + β * I ∨ x = α - β * I)) :
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (Real.sqrt a + Real.sqrt b > Real.sqrt c ∧
   Real.sqrt b + Real.sqrt c > Real.sqrt a ∧
   Real.sqrt c + Real.sqrt a > Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_implications_l247_24713


namespace NUMINAMATH_CALUDE_system_solvable_l247_24773

/-- The system of equations has a real solution if and only if m ≠ 3/2 -/
theorem system_solvable (m : ℝ) : 
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 3/2 := by
sorry

end NUMINAMATH_CALUDE_system_solvable_l247_24773


namespace NUMINAMATH_CALUDE_right_triangular_pyramid_property_l247_24707

/-- A right-angled triangular pyramid -/
structure RightTriangularPyramid where
  /-- Area of the first right-angle face -/
  S₁ : ℝ
  /-- Area of the second right-angle face -/
  S₂ : ℝ
  /-- Area of the third right-angle face -/
  S₃ : ℝ
  /-- Area of the oblique face -/
  S : ℝ
  /-- All areas are positive -/
  S₁_pos : S₁ > 0
  S₂_pos : S₂ > 0
  S₃_pos : S₃ > 0
  S_pos : S > 0
  /-- Lateral edges are perpendicular to each other -/
  lateral_edges_perpendicular : True

/-- The property of a right-angled triangular pyramid -/
theorem right_triangular_pyramid_property (p : RightTriangularPyramid) :
  p.S₁^2 + p.S₂^2 + p.S₃^2 = p.S^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_pyramid_property_l247_24707


namespace NUMINAMATH_CALUDE_total_crayons_for_six_children_l247_24723

/-- Calculates the total number of crayons given the number of children and crayons per child -/
def total_crayons (num_children : ℕ) (crayons_per_child : ℕ) : ℕ :=
  num_children * crayons_per_child

/-- Theorem: Given 6 children with 3 crayons each, the total number of crayons is 18 -/
theorem total_crayons_for_six_children :
  total_crayons 6 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_for_six_children_l247_24723


namespace NUMINAMATH_CALUDE_point_plane_configuration_exists_l247_24719

-- Define a type for points in space
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in space
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if a point lies on a plane
def pointOnPlane (p : Point) (pl : Plane) : Prop :=
  pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d = 0

-- Define a function to check if a set of points is collinear
def collinear (points : Set Point) : Prop :=
  ∃ (a b c : ℝ), ∀ p ∈ points, a * p.x + b * p.y + c * p.z = 0

-- State the theorem
theorem point_plane_configuration_exists :
  ∃ (points : Set Point) (planes : Set Plane),
    -- There are several points and planes
    (points.Nonempty ∧ planes.Nonempty) ∧
    -- Through any two points, exactly two planes pass
    (∀ p q : Point, p ∈ points → q ∈ points → p ≠ q →
      ∃! (pl1 pl2 : Plane), pl1 ∈ planes ∧ pl2 ∈ planes ∧ pl1 ≠ pl2 ∧
        pointOnPlane p pl1 ∧ pointOnPlane q pl1 ∧
        pointOnPlane p pl2 ∧ pointOnPlane q pl2) ∧
    -- Each plane contains at least four points
    (∀ pl : Plane, pl ∈ planes →
      ∃ (p1 p2 p3 p4 : Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p4 ∈ points ∧
        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
        pointOnPlane p1 pl ∧ pointOnPlane p2 pl ∧ pointOnPlane p3 pl ∧ pointOnPlane p4 pl) ∧
    -- Not all points lie on a single line
    ¬collinear points :=
by
  sorry

end NUMINAMATH_CALUDE_point_plane_configuration_exists_l247_24719


namespace NUMINAMATH_CALUDE_magazine_purchase_combinations_l247_24778

-- Define the number of magazines and their prices
def total_magazines : ℕ := 11
def magazines_2yuan : ℕ := 8
def magazines_1yuan : ℕ := 3
def total_money : ℕ := 10

-- Define a function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the theorem
theorem magazine_purchase_combinations : 
  (combination magazines_1yuan 2 * combination magazines_2yuan 4) + 
  (combination magazines_2yuan 5) = 266 := by
  sorry

#check magazine_purchase_combinations

end NUMINAMATH_CALUDE_magazine_purchase_combinations_l247_24778


namespace NUMINAMATH_CALUDE_cube_equation_solution_l247_24757

theorem cube_equation_solution :
  ∃! x : ℝ, (8 - x)^3 = x^3 ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l247_24757


namespace NUMINAMATH_CALUDE_solution_set_f_geq_neg_two_max_a_for_f_leq_x_minus_a_l247_24732

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

-- Theorem for the solution set of f(x) ≥ -2
theorem solution_set_f_geq_neg_two :
  {x : ℝ | f x ≥ -2} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 6} := by sorry

-- Theorem for the maximum value of a
theorem max_a_for_f_leq_x_minus_a :
  ∀ a : ℝ, (∀ x : ℝ, f x ≤ x - a) ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_neg_two_max_a_for_f_leq_x_minus_a_l247_24732


namespace NUMINAMATH_CALUDE_xy_equals_three_l247_24797

theorem xy_equals_three (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x ≠ y) 
  (h4 : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_three_l247_24797


namespace NUMINAMATH_CALUDE_ln_cube_inequality_l247_24700

theorem ln_cube_inequality (a b : ℝ) : 
  (∃ a b, a^3 < b^3 ∧ ¬(Real.log a < Real.log b)) ∧ 
  (∀ a b, Real.log a < Real.log b → a^3 < b^3) :=
sorry

end NUMINAMATH_CALUDE_ln_cube_inequality_l247_24700


namespace NUMINAMATH_CALUDE_total_clothing_cost_l247_24775

def shorts_cost : ℚ := 14.28
def jacket_cost : ℚ := 4.74

theorem total_clothing_cost : shorts_cost + jacket_cost = 19.02 := by
  sorry

end NUMINAMATH_CALUDE_total_clothing_cost_l247_24775


namespace NUMINAMATH_CALUDE_new_figure_length_is_32_l247_24720

/-- Represents the dimensions of the original polygon --/
structure PolygonDimensions where
  vertical_side : ℝ
  top_first_horizontal : ℝ
  top_second_horizontal : ℝ
  remaining_horizontal : ℝ
  last_vertical_drop : ℝ

/-- Calculates the total length of segments in the new figure after removing four sides --/
def newFigureLength (d : PolygonDimensions) : ℝ :=
  d.vertical_side + (d.top_first_horizontal + d.top_second_horizontal + d.remaining_horizontal) +
  (d.vertical_side - d.last_vertical_drop) + d.last_vertical_drop

/-- Theorem stating that for the given dimensions, the new figure length is 32 units --/
theorem new_figure_length_is_32 (d : PolygonDimensions)
    (h1 : d.vertical_side = 10)
    (h2 : d.top_first_horizontal = 3)
    (h3 : d.top_second_horizontal = 4)
    (h4 : d.remaining_horizontal = 5)
    (h5 : d.last_vertical_drop = 2) :
    newFigureLength d = 32 := by
  sorry

end NUMINAMATH_CALUDE_new_figure_length_is_32_l247_24720


namespace NUMINAMATH_CALUDE_potato_distribution_l247_24721

theorem potato_distribution (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) :
  total_potatoes = 24 →
  num_people = 3 →
  total_potatoes = num_people * potatoes_per_person →
  potatoes_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_potato_distribution_l247_24721


namespace NUMINAMATH_CALUDE_simplify_expression_l247_24785

theorem simplify_expression : (81 ^ (1/4) - (33/4) ^ (1/2)) ^ 2 = (69 - 12 * 33 ^ (1/2)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l247_24785


namespace NUMINAMATH_CALUDE_laundry_time_l247_24760

def total_time : ℕ := 120
def bathroom_time : ℕ := 15
def room_time : ℕ := 35
def homework_time : ℕ := 40

theorem laundry_time : 
  ∃ (laundry_time : ℕ), 
    laundry_time + bathroom_time + room_time + homework_time = total_time ∧ 
    laundry_time = 30 :=
by sorry

end NUMINAMATH_CALUDE_laundry_time_l247_24760


namespace NUMINAMATH_CALUDE_simplest_common_denominator_example_l247_24789

/-- The simplest common denominator of two fractions -/
def simplestCommonDenominator (f1 f2 : ℚ) : ℤ :=
  sorry

/-- Theorem: The simplest common denominator of 1/(m^2-9) and 1/(2m+6) is 2(m+3)(m-3) -/
theorem simplest_common_denominator_example (m : ℚ) :
  simplestCommonDenominator (1 / (m^2 - 9)) (1 / (2*m + 6)) = 2 * (m + 3) * (m - 3) :=
by sorry

end NUMINAMATH_CALUDE_simplest_common_denominator_example_l247_24789


namespace NUMINAMATH_CALUDE_valid_lineup_count_l247_24772

def team_size : ℕ := 15
def lineup_size : ℕ := 6

def cannot_play_together (p1 p2 : ℕ) : Prop := p1 ≠ p2

def excludes_player (p1 p2 : ℕ) : Prop := p1 ≠ p2

def valid_lineup (lineup : Finset ℕ) : Prop :=
  lineup.card = lineup_size ∧
  (∀ p ∈ lineup, p ≤ team_size) ∧
  ¬(1 ∈ lineup ∧ 2 ∈ lineup) ∧
  (1 ∈ lineup → 3 ∉ lineup)

def count_valid_lineups : ℕ := sorry

theorem valid_lineup_count :
  count_valid_lineups = 3795 := by sorry

end NUMINAMATH_CALUDE_valid_lineup_count_l247_24772


namespace NUMINAMATH_CALUDE_water_bottle_calculation_l247_24786

/-- Given an initial number of bottles, calculate the final number after removing some and adding more. -/
def final_bottles (initial remove add : ℕ) : ℕ :=
  initial - remove + add

/-- Theorem: Given 14 initial bottles, removing 8 and adding 45 results in 51 bottles. -/
theorem water_bottle_calculation :
  final_bottles 14 8 45 = 51 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_calculation_l247_24786


namespace NUMINAMATH_CALUDE_actual_average_height_l247_24753

/-- Calculates the actual average height of students given initial incorrect data and correction --/
theorem actual_average_height
  (num_students : ℕ)
  (initial_average : ℝ)
  (incorrect_height : ℝ)
  (actual_height : ℝ)
  (h_num_students : num_students = 20)
  (h_initial_average : initial_average = 175)
  (h_incorrect_height : incorrect_height = 151)
  (h_actual_height : actual_height = 136) :
  (num_students * initial_average - (incorrect_height - actual_height)) / num_students = 174.25 := by
  sorry

end NUMINAMATH_CALUDE_actual_average_height_l247_24753


namespace NUMINAMATH_CALUDE_order_of_x_l247_24782

theorem order_of_x (x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (eq1 : x₁ + x₂ + x₃ = a₁)
  (eq2 : x₂ + x₃ + x₁ = a₂)
  (eq3 : x₃ + x₄ + x₅ = a₃)
  (eq4 : x₄ + x₅ + x₁ = a₄)
  (eq5 : x₅ + x₁ + x₂ = a₅)
  (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅) :
  x₃ > x₁ ∧ x₁ > x₄ ∧ x₄ > x₂ ∧ x₂ > x₅ := by
  sorry

end NUMINAMATH_CALUDE_order_of_x_l247_24782


namespace NUMINAMATH_CALUDE_distinct_integers_product_sum_l247_24774

theorem distinct_integers_product_sum (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 80 →
  p + q + r + s + t = 36 := by
sorry

end NUMINAMATH_CALUDE_distinct_integers_product_sum_l247_24774


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l247_24792

/-- The total surface area of a cube with holes -/
def total_surface_area (cube_edge : ℝ) (hole_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_side^2
  let new_exposed_area := 6 * 4 * hole_side^2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem: The total surface area of a cube with edge length 3 and square holes of side 1 is 72 -/
theorem cube_with_holes_surface_area :
  total_surface_area 3 1 = 72 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l247_24792


namespace NUMINAMATH_CALUDE_complex_coordinate_l247_24703

theorem complex_coordinate (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = (2 + 4*i) / i → (z.re = 4 ∧ z.im = -2) :=
by sorry

end NUMINAMATH_CALUDE_complex_coordinate_l247_24703


namespace NUMINAMATH_CALUDE_music_school_tuition_cost_l247_24796

/-- The cost calculation for music school tuition with sibling discounts -/
theorem music_school_tuition_cost : 
  let base_tuition : ℕ := 45
  let first_sibling_discount : ℕ := 15
  let additional_sibling_discount : ℕ := 10
  let num_children : ℕ := 4
  
  base_tuition + 
  (base_tuition - first_sibling_discount) + 
  (base_tuition - additional_sibling_discount) + 
  (base_tuition - additional_sibling_discount) = 145 :=
by sorry

end NUMINAMATH_CALUDE_music_school_tuition_cost_l247_24796


namespace NUMINAMATH_CALUDE_distance_sum_is_48_l247_24715

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13)
  (bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 14)
  (ca_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 15)

-- Define points Q and R
def Q (t : Triangle) : ℝ × ℝ := sorry
def R (t : Triangle) : ℝ × ℝ := sorry

-- Define the right angle condition
def is_right_angle (A B C : ℝ × ℝ) : Prop := sorry

-- Define similarity of triangles
def are_similar (t1 t2 t3 : Triangle) : Prop := sorry

-- Define the distance from a point to a line
def distance_to_line (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem distance_sum_is_48 (t : Triangle) 
  (h1 : is_right_angle (Q t) (t.C) (t.B))
  (h2 : is_right_angle (R t) (t.B) (t.C))
  (P1 P2 : ℝ × ℝ)
  (h3 : are_similar 
    ⟨P1, Q t, R t, sorry, sorry, sorry⟩ 
    ⟨P2, Q t, R t, sorry, sorry, sorry⟩ 
    t) :
  distance_to_line P1 t.B t.C + distance_to_line P2 t.B t.C = 48 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_is_48_l247_24715


namespace NUMINAMATH_CALUDE_danny_found_58_new_caps_l247_24744

/-- Represents the number of bottle caps Danny has at different stages -/
structure BottleCaps where
  initial : ℕ
  thrown_away : ℕ
  final : ℕ

/-- Calculates the number of new bottle caps Danny found -/
def new_bottle_caps (bc : BottleCaps) : ℕ :=
  bc.final - (bc.initial - bc.thrown_away)

/-- Theorem stating that Danny found 58 new bottle caps -/
theorem danny_found_58_new_caps : 
  ∀ (bc : BottleCaps), 
  bc.initial = 69 → bc.thrown_away = 60 → bc.final = 67 → 
  new_bottle_caps bc = 58 := by
  sorry

end NUMINAMATH_CALUDE_danny_found_58_new_caps_l247_24744


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l247_24748

-- Define set A
def A : Set ℝ := {0, 1, 2}

-- Define set B
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l247_24748


namespace NUMINAMATH_CALUDE_absolute_value_sum_l247_24727

theorem absolute_value_sum (a b : ℝ) : 
  (abs a = 3) → (abs b = 4) → (a * b < 0) → abs (a + b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l247_24727


namespace NUMINAMATH_CALUDE_naoh_equals_agoh_l247_24718

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- The reaction between AgNO3 and NaOH to form AgOH -/
structure Reaction where
  agno3_initial : Moles
  agoh_formed : Moles
  naoh_combined : Moles

/-- The conditions of the reaction -/
class ReactionConditions (r : Reaction) where
  agno3_agoh_equal : r.agno3_initial = r.agoh_formed
  one_to_one_ratio : r.agoh_formed = r.naoh_combined

/-- Theorem stating that the number of moles of NaOH combined equals the number of moles of AgOH formed -/
theorem naoh_equals_agoh (r : Reaction) [ReactionConditions r] : r.naoh_combined = r.agoh_formed := by
  sorry

end NUMINAMATH_CALUDE_naoh_equals_agoh_l247_24718


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_progression_l247_24795

theorem smallest_b_in_arithmetic_progression (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  (∃ d : ℝ, a = b - d ∧ c = b + d) →
  a * b * c = 125 →
  b ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_progression_l247_24795


namespace NUMINAMATH_CALUDE_sin_A_value_l247_24710

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem sin_A_value (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.b = Real.sqrt 3) 
  (h3 : t.A + t.C = 2 * t.B) 
  (h4 : t.A + t.B + t.C = Real.pi) 
  (h5 : t.a / Real.sin t.A = t.b / Real.sin t.B) : 
  Real.sin t.A = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_A_value_l247_24710


namespace NUMINAMATH_CALUDE_min_people_for_tests_l247_24767

/-- The minimum number of people required to achieve the given score ranges -/
def min_people (ranges : List ℕ) (min_range : ℕ) : ℕ :=
  if ranges.maximum = some min_range then 2 else 1

/-- Theorem: Given the conditions, at least 2 people took the tests -/
theorem min_people_for_tests : min_people [17, 28, 35, 45] 45 = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_people_for_tests_l247_24767


namespace NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l247_24763

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 300! has 74 trailing zeros -/
theorem factorial_300_trailing_zeros :
  trailingZeros 300 = 74 := by
  sorry

end NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l247_24763


namespace NUMINAMATH_CALUDE_equal_tabletops_and_legs_l247_24766

/-- Represents the amount of wood used for tabletops -/
def wood_for_tabletops : ℝ := 3

/-- Represents the amount of wood used for legs -/
def wood_for_legs : ℝ := 5 - wood_for_tabletops

/-- Represents the number of tabletops that can be made from 1 cubic meter of wood -/
def tabletops_per_cubic_meter : ℝ := 50

/-- Represents the number of legs that can be made from 1 cubic meter of wood -/
def legs_per_cubic_meter : ℝ := 300

/-- Represents the number of legs per table -/
def legs_per_table : ℝ := 4

theorem equal_tabletops_and_legs :
  wood_for_tabletops * tabletops_per_cubic_meter = 
  wood_for_legs * legs_per_cubic_meter / legs_per_table := by
  sorry

end NUMINAMATH_CALUDE_equal_tabletops_and_legs_l247_24766


namespace NUMINAMATH_CALUDE_equality_of_products_l247_24779

theorem equality_of_products (a b c d x y z q : ℝ) 
  (h1 : a ^ x = c ^ q) (h2 : c ^ q = b) 
  (h3 : c ^ y = a ^ z) (h4 : a ^ z = d) : 
  x * y = q * z := by
  sorry

end NUMINAMATH_CALUDE_equality_of_products_l247_24779


namespace NUMINAMATH_CALUDE_mrs_petersons_change_l247_24716

def change_calculation (num_tumblers : ℕ) (cost_per_tumbler : ℚ) (discount_rate : ℚ) (num_bills : ℕ) (bill_value : ℚ) : ℚ :=
  let total_cost_before_discount := num_tumblers * cost_per_tumbler
  let discount_amount := total_cost_before_discount * discount_rate
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let total_amount_paid := num_bills * bill_value
  total_amount_paid - total_cost_after_discount

theorem mrs_petersons_change :
  change_calculation 10 45 (1/10) 5 100 = 95 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_petersons_change_l247_24716


namespace NUMINAMATH_CALUDE_smallest_root_of_unity_order_l247_24756

theorem smallest_root_of_unity_order (z : ℂ) : 
  (∃ (n : ℕ), n > 0 ∧ (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^n = 1) ∧ 
   (∀ m : ℕ, m > 0 → (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^m = 1) → m ≥ n)) → 
  (∃ (n : ℕ), n = 18 ∧ n > 0 ∧ (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^n = 1) ∧ 
   (∀ m : ℕ, m > 0 → (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^m = 1) → m ≥ n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_root_of_unity_order_l247_24756


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l247_24752

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b : ℝ, |a - b^2| + |b - a^2| ≤ 1 → (a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧
  (∃ a b : ℝ, (a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2 ∧ |a - b^2| + |b - a^2| > 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l247_24752


namespace NUMINAMATH_CALUDE_table_covering_l247_24745

/-- A tile type used to cover the table -/
inductive Tile
  | Square  -- 2×2 square tile
  | LShaped -- L-shaped tile with 5 cells

/-- Represents a covering of the table -/
def Covering (m n : ℕ) := List (ℕ × ℕ × Tile)

/-- Checks if a covering is valid for the given table dimensions -/
def IsValidCovering (m n : ℕ) (c : Covering m n) : Prop := sorry

/-- The main theorem stating the condition for possible covering -/
theorem table_covering (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (∃ c : Covering m n, IsValidCovering m n c) ↔ (6 ∣ m ∨ 6 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_table_covering_l247_24745


namespace NUMINAMATH_CALUDE_good_sets_exist_l247_24729

-- Define a "good" subset of natural numbers
def is_good (A : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (∃! p : ℕ, Prime p ∧ p ∣ n ∧ (n - p) ∈ A)

-- Define the set of perfect squares
def perfect_squares : Set ℕ := {n : ℕ | ∃ k : ℕ, n = k^2}

-- Define the set of prime numbers
def prime_set : Set ℕ := {p : ℕ | Prime p}

theorem good_sets_exist :
  (is_good perfect_squares) ∧ 
  (is_good prime_set) ∧ 
  (Set.Infinite prime_set) ∧ 
  (perfect_squares ∩ prime_set = ∅) := by
  sorry

end NUMINAMATH_CALUDE_good_sets_exist_l247_24729


namespace NUMINAMATH_CALUDE_tommys_tomato_profit_l247_24736

/-- Represents the problem of calculating Tommy's profit from selling tomatoes -/
theorem tommys_tomato_profit :
  let crate_capacity : ℕ := 20 -- kg
  let num_crates : ℕ := 3
  let crates_cost : ℕ := 330 -- $
  let selling_price : ℕ := 6 -- $ per kg
  let rotten_tomatoes : ℕ := 3 -- kg
  
  let total_capacity : ℕ := crate_capacity * num_crates
  let sellable_tomatoes : ℕ := total_capacity - rotten_tomatoes
  let revenue : ℕ := sellable_tomatoes * selling_price
  let profit : ℤ := revenue - crates_cost

  profit = 12 := by
  sorry

/- Note: We use ℕ (natural numbers) for non-negative integers and ℤ (integers) for the final profit calculation to allow for the possibility of negative profit. -/

end NUMINAMATH_CALUDE_tommys_tomato_profit_l247_24736


namespace NUMINAMATH_CALUDE_rhinoceros_grazing_area_l247_24730

theorem rhinoceros_grazing_area 
  (initial_population : ℕ) 
  (watering_area : ℕ) 
  (population_increase_rate : ℚ) 
  (total_preserve_area : ℕ) 
  (h1 : initial_population = 8000)
  (h2 : watering_area = 10000)
  (h3 : population_increase_rate = 1/10)
  (h4 : total_preserve_area = 890000) :
  let final_population := initial_population + initial_population * population_increase_rate
  let grazing_area := total_preserve_area - watering_area
  grazing_area / final_population = 100 := by
sorry

end NUMINAMATH_CALUDE_rhinoceros_grazing_area_l247_24730


namespace NUMINAMATH_CALUDE_water_bottle_cost_l247_24749

theorem water_bottle_cost (cola_price : ℝ) (juice_price : ℝ) (water_price : ℝ)
  (cola_sold : ℕ) (juice_sold : ℕ) (water_sold : ℕ) (total_revenue : ℝ)
  (h1 : cola_price = 3)
  (h2 : juice_price = 1.5)
  (h3 : cola_sold = 15)
  (h4 : juice_sold = 12)
  (h5 : water_sold = 25)
  (h6 : total_revenue = 88)
  (h7 : total_revenue = cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold) :
  water_price = 1 := by
sorry

end NUMINAMATH_CALUDE_water_bottle_cost_l247_24749


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_5_l247_24751

theorem smallest_common_multiple_of_6_and_5 : ∃ n : ℕ, 
  n > 0 ∧ 
  6 ∣ n ∧ 
  5 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 6 ∣ m → 5 ∣ m → n ≤ m :=
by
  use 30
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_5_l247_24751


namespace NUMINAMATH_CALUDE_exponent_division_l247_24777

theorem exponent_division (a : ℝ) : a^4 / a^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l247_24777
