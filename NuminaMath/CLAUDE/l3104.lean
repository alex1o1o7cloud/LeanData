import Mathlib

namespace y1_value_l3104_310446

theorem y1_value (y1 y2 y3 : ℝ) 
  (h1 : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1) 
  (h2 : (1 - y1)^2 + (y1 - y2)^2 + (y2 - y3)^2 + y3^2 = 1/9) : 
  y1 = 1/2 := by
sorry

end y1_value_l3104_310446


namespace all_twentynine_l3104_310476

/-- A function that represents a circular arrangement of 2017 integers. -/
def CircularArrangement := Fin 2017 → ℤ

/-- Predicate to check if five consecutive elements in the arrangement are "arrangeable". -/
def IsArrangeable (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 2017, arr i - arr (i + 1) + arr (i + 2) - arr (i + 3) + arr (i + 4) = 29

/-- Theorem stating that if all consecutive five-tuples in a circular arrangement of 2017 integers
    are arrangeable, then all integers in the arrangement must be 29. -/
theorem all_twentynine (arr : CircularArrangement) (h : IsArrangeable arr) :
    ∀ i : Fin 2017, arr i = 29 := by
  sorry

end all_twentynine_l3104_310476


namespace arithmetic_sequence_sum_l3104_310430

/-- Given an arithmetic sequence {a_n} where a_4 = 4, prove that S_7 = 28 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n : ℝ) / 2 * (a 1 + a n)) →  -- Definition of S_n
  (∀ k m, a (k + m) - a k = m * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  a 4 = 4 →  -- Given condition
  S 7 = 28 :=
by sorry

end arithmetic_sequence_sum_l3104_310430


namespace cubic_roots_sum_l3104_310471

theorem cubic_roots_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) → 
  (0 < b ∧ b < 1) → 
  (0 < c ∧ c < 1) → 
  a ≠ b → b ≠ c → a ≠ c →
  40 * a^3 - 70 * a^2 + 32 * a - 3 = 0 →
  40 * b^3 - 70 * b^2 + 32 * b - 3 = 0 →
  40 * c^3 - 70 * c^2 + 32 * c - 3 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 :=
by
  sorry

end cubic_roots_sum_l3104_310471


namespace three_digit_permutation_sum_l3104_310424

/-- A three-digit number with no zeros -/
def ThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∀ d, d ∣ n → d ≠ 0

/-- Sum of all distinct permutations of the digits of a number -/
def SumOfPermutations (n : ℕ) : ℕ := sorry

theorem three_digit_permutation_sum (n : ℕ) :
  ThreeDigitNumber n → SumOfPermutations n = 2775 → n = 889 ∨ n = 997 := by
  sorry

end three_digit_permutation_sum_l3104_310424


namespace card_sequence_return_l3104_310428

theorem card_sequence_return (n : ℕ) (hn : n > 0) : 
  Nat.totient (2 * n - 1) ≤ 2 * n - 2 := by
  sorry

end card_sequence_return_l3104_310428


namespace solve_equation_l3104_310456

theorem solve_equation : ∃ r : ℤ, 19 - 3 = 2 + r ∧ r = 14 := by sorry

end solve_equation_l3104_310456


namespace pencils_per_child_l3104_310492

theorem pencils_per_child (num_children : ℕ) (total_pencils : ℕ) 
  (h1 : num_children = 11) 
  (h2 : total_pencils = 22) : 
  total_pencils / num_children = 2 := by
  sorry

end pencils_per_child_l3104_310492


namespace polynomial_root_coefficients_l3104_310473

theorem polynomial_root_coefficients :
  ∀ (a b c : ℝ),
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - Complex.I : ℂ) ^ 4 + a * (2 - Complex.I : ℂ) ^ 3 + b * (2 - Complex.I : ℂ) ^ 2 - 2 * (2 - Complex.I : ℂ) + c = 0 →
  a = 2 + 2 * Real.sqrt 1.5 ∧
  b = 10 + 2 * Real.sqrt 1.5 ∧
  c = 10 - 8 * Real.sqrt 1.5 :=
by sorry

end polynomial_root_coefficients_l3104_310473


namespace bug_return_probability_l3104_310409

/-- Probability of the bug being at the starting corner after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The probability of the bug returning to its starting corner on its eighth move -/
theorem bug_return_probability : Q 8 = 547 / 2187 := by
  sorry

end bug_return_probability_l3104_310409


namespace train_passing_jogger_train_passes_jogger_in_35_seconds_l3104_310489

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger 
  (jogger_speed : Real) 
  (train_speed : Real) 
  (train_length : Real) 
  (initial_lead : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_lead + train_length
  total_distance / relative_speed

/-- Proof that the train passes the jogger in 35 seconds -/
theorem train_passes_jogger_in_35_seconds : 
  train_passing_jogger 9 45 110 240 = 35 := by
  sorry

end train_passing_jogger_train_passes_jogger_in_35_seconds_l3104_310489


namespace sum_of_factors_of_125_l3104_310431

theorem sum_of_factors_of_125 :
  ∃ (a b c : ℕ+),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a.val * b.val * c.val = 125) ∧
    (a.val + b.val + c.val = 31) := by
  sorry

end sum_of_factors_of_125_l3104_310431


namespace solve_system_l3104_310488

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 14) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -1/11 := by
sorry

end solve_system_l3104_310488


namespace sprint_tournament_races_l3104_310451

/-- Calculates the minimum number of races needed to determine a champion -/
def minimumRaces (totalSprinters : Nat) (lanesPerRace : Nat) : Nat :=
  let eliminationsNeeded := totalSprinters - 1
  let eliminationsPerRace := lanesPerRace - 1
  (eliminationsNeeded + eliminationsPerRace - 1) / eliminationsPerRace

theorem sprint_tournament_races : 
  minimumRaces 256 8 = 37 := by
  sorry

#eval minimumRaces 256 8

end sprint_tournament_races_l3104_310451


namespace base_conversion_sum_approx_l3104_310452

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [3, 6, 2]  -- 263 in base 8
def num2 : List Nat := [3, 1]     -- 13 in base 3
def num3 : List Nat := [3, 4, 2]  -- 243 in base 7
def num4 : List Nat := [5, 3]     -- 35 in base 6

-- State the theorem
theorem base_conversion_sum_approx :
  let x1 := baseToDecimal num1 8
  let x2 := baseToDecimal num2 3
  let x3 := baseToDecimal num3 7
  let x4 := baseToDecimal num4 6
  abs ((x1 / x2 + x3 / x4 : ℚ) - 35.442) < 0.001 := by
  sorry

end base_conversion_sum_approx_l3104_310452


namespace special_polynomial_inequality_l3104_310499

/-- A polynomial with real coefficients that has three positive real roots and a negative value at x = 0 -/
structure SpecialPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  has_three_positive_roots : ∃ (x₁ x₂ x₃ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
    ∀ x, a * x^3 + b * x^2 + c * x + d = a * (x - x₁) * (x - x₂) * (x - x₃)
  negative_at_zero : d < 0

/-- The inequality holds for special polynomials -/
theorem special_polynomial_inequality (φ : SpecialPolynomial) :
  2 * φ.b^3 + 9 * φ.a^2 * φ.d - 7 * φ.a * φ.b * φ.c ≤ 0 := by
  sorry

end special_polynomial_inequality_l3104_310499


namespace paolo_sevilla_birthday_friends_l3104_310498

theorem paolo_sevilla_birthday_friends :
  ∀ (n : ℕ) (total_bill : ℝ),
    (total_bill / (n + 2 : ℝ) = 12) →
    (total_bill / n = 16) →
    n = 6 := by
  sorry

end paolo_sevilla_birthday_friends_l3104_310498


namespace sqrt_meaningful_range_l3104_310440

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 3) → x ≥ 3 := by
  sorry

end sqrt_meaningful_range_l3104_310440


namespace stratified_sampling_correctness_problem_case_proof_l3104_310408

/-- Represents the number of students in each year and the total sample size. -/
structure SchoolData where
  totalStudents : ℕ
  freshmanStudents : ℕ
  sophomoreStudents : ℕ
  juniorStudents : ℕ
  sampleSize : ℕ

/-- Calculates the number of students to be sampled from a specific year. -/
def sampledStudents (data : SchoolData) (yearStudents : ℕ) : ℕ :=
  (yearStudents * data.sampleSize) / data.totalStudents

/-- Theorem stating that the sum of sampled students from each year equals the total sample size. -/
theorem stratified_sampling_correctness (data : SchoolData) 
    (h1 : data.totalStudents = data.freshmanStudents + data.sophomoreStudents + data.juniorStudents)
    (h2 : data.sampleSize ≤ data.totalStudents) :
  sampledStudents data data.freshmanStudents +
  sampledStudents data data.sophomoreStudents +
  sampledStudents data data.juniorStudents = data.sampleSize := by
  sorry

/-- Verifies the specific case given in the problem. -/
def verifyProblemCase : Prop :=
  let data : SchoolData := {
    totalStudents := 1200,
    freshmanStudents := 300,
    sophomoreStudents := 400,
    juniorStudents := 500,
    sampleSize := 60
  }
  sampledStudents data data.freshmanStudents = 15 ∧
  sampledStudents data data.sophomoreStudents = 20 ∧
  sampledStudents data data.juniorStudents = 25

/-- Proves the specific case given in the problem. -/
theorem problem_case_proof : verifyProblemCase := by
  sorry

end stratified_sampling_correctness_problem_case_proof_l3104_310408


namespace smallest_common_multiple_of_9_and_6_l3104_310486

theorem smallest_common_multiple_of_9_and_6 :
  ∃ n : ℕ, n > 0 ∧ 9 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m > 0 → 9 ∣ m → 6 ∣ m → n ≤ m :=
by
  -- Proof goes here
  sorry

end smallest_common_multiple_of_9_and_6_l3104_310486


namespace interest_problem_l3104_310454

/-- Given a sum P put at simple interest for 10 years, if increasing the interest rate
    by 5% results in Rs. 150 more interest, then P = 300. -/
theorem interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 150 → P = 300 := by
  sorry


end interest_problem_l3104_310454


namespace base_27_to_3_conversion_l3104_310463

/-- Converts a single digit from base 27 to its three-digit representation in base 3 -/
def convert_digit_27_to_3 (d : Nat) : Nat × Nat × Nat :=
  (d / 9, (d % 9) / 3, d % 3)

/-- Converts a number from base 27 to base 3 -/
def convert_27_to_3 (n : Nat) : List Nat :=
  let digits := n.digits 27
  List.join (digits.map (fun d => let (a, b, c) := convert_digit_27_to_3 d; [a, b, c]))

theorem base_27_to_3_conversion :
  convert_27_to_3 652 = [0, 2, 0, 0, 1, 2, 0, 0, 2] := by
  sorry

end base_27_to_3_conversion_l3104_310463


namespace unique_root_condition_l3104_310469

/-- The equation √(ax² + ax + 2) = ax + 2 has a unique real root if and only if a = -8 or a ≥ 1 -/
theorem unique_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.sqrt (a * x^2 + a * x + 2) = a * x + 2) ↔ (a = -8 ∨ a ≥ 1) :=
by sorry

end unique_root_condition_l3104_310469


namespace card_game_remainder_l3104_310437

def deck_size : ℕ := 60
def hand_size : ℕ := 12

def possible_remainders : List ℕ := [20, 40, 60, 80, 0]

theorem card_game_remainder :
  ∃ (r : ℕ), r ∈ possible_remainders ∧ 
  (Nat.choose deck_size hand_size) % 100 = r :=
sorry

end card_game_remainder_l3104_310437


namespace greatest_two_digit_multiple_of_three_l3104_310401

theorem greatest_two_digit_multiple_of_three : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 3 = 0 → n ≤ 99 :=
by sorry

end greatest_two_digit_multiple_of_three_l3104_310401


namespace max_leftover_candy_exists_max_leftover_candy_l3104_310432

theorem max_leftover_candy (x : ℕ) : x % 11 ≤ 10 := by sorry

theorem exists_max_leftover_candy : ∃ x : ℕ, x % 11 = 10 := by sorry

end max_leftover_candy_exists_max_leftover_candy_l3104_310432


namespace smallest_k_proof_l3104_310461

/-- The smallest integer k for which x^2 - x + 2 - k = 0 has two distinct real roots -/
def smallest_k : ℕ := 2

/-- The quadratic equation x^2 - x + 2 - k = 0 -/
def quadratic (x k : ℝ) : Prop := x^2 - x + 2 - k = 0

theorem smallest_k_proof :
  (∀ k < smallest_k, ¬∃ x y : ℝ, x ≠ y ∧ quadratic x k ∧ quadratic y k) ∧
  (∃ x y : ℝ, x ≠ y ∧ quadratic x smallest_k ∧ quadratic y smallest_k) :=
sorry

end smallest_k_proof_l3104_310461


namespace point_b_coordinate_l3104_310474

theorem point_b_coordinate (b : ℝ) : 
  (|(-2) - b| = 3) ↔ (b = -5 ∨ b = 1) := by sorry

end point_b_coordinate_l3104_310474


namespace fib_divisibility_implies_fib_number_l3104_310462

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Condition: For every positive integer m, there exists a positive integer n such that m | Fₙ - k -/
def condition (k : ℕ) : Prop :=
  ∀ m : ℕ, m > 0 → ∃ n : ℕ, n > 0 ∧ (fib n - k) % m = 0

/-- Main theorem: If the condition holds, then k is a Fibonacci number -/
theorem fib_divisibility_implies_fib_number (k : ℕ) (h : condition k) :
  ∃ n : ℕ, fib n = k :=
sorry

end fib_divisibility_implies_fib_number_l3104_310462


namespace work_left_after_nine_days_l3104_310415

/-- The fraction of work left after 9 days given the work rates of A, B, and C -/
theorem work_left_after_nine_days (a_rate b_rate c_rate : ℚ) : 
  a_rate = 1 / 15 →
  b_rate = 1 / 20 →
  c_rate = 1 / 25 →
  let combined_rate := a_rate + b_rate + c_rate
  let work_done_first_four_days := 4 * combined_rate
  let ac_rate := a_rate + c_rate
  let work_done_next_five_days := 5 * ac_rate
  let total_work_done := work_done_first_four_days + work_done_next_five_days
  total_work_done ≥ 1 := by sorry

#check work_left_after_nine_days

end work_left_after_nine_days_l3104_310415


namespace parabola_focus_l3104_310458

/-- A parabola is defined by the equation y² = -16x + 64. -/
def parabola (x y : ℝ) : Prop := y^2 = -16*x + 64

/-- The focus of a parabola is a point on its axis of symmetry. -/
def is_focus (x y : ℝ) : Prop := sorry

/-- The focus of the parabola y² = -16x + 64 is at (0, 0). -/
theorem parabola_focus :
  is_focus 0 0 ∧ ∀ x y, parabola x y → is_focus x y → x = 0 ∧ y = 0 := by sorry

end parabola_focus_l3104_310458


namespace expression_simplification_l3104_310484

theorem expression_simplification :
  ∀ x : ℝ, ((3*x^2 + 2*x - 1) + x^2*2)*4 + (5 - 2/2)*(3*x^2 + 6*x - 8) = 32*x^2 + 32*x - 36 := by
  sorry

end expression_simplification_l3104_310484


namespace robert_can_finish_both_books_l3104_310411

/-- Represents the number of pages Robert can read per hour -/
def reading_speed : ℕ := 120

/-- Represents the number of pages in the first book -/
def book1_pages : ℕ := 360

/-- Represents the number of pages in the second book -/
def book2_pages : ℕ := 180

/-- Represents the number of hours Robert has available for reading -/
def available_time : ℕ := 7

/-- Theorem stating that Robert can finish both books within the available time -/
theorem robert_can_finish_both_books :
  (book1_pages / reading_speed + book2_pages / reading_speed : ℚ) ≤ available_time :=
sorry

end robert_can_finish_both_books_l3104_310411


namespace puzzle_min_cost_l3104_310403

/-- Represents the cost structure and purchase requirement for puzzles -/
structure PuzzlePurchase where
  single_cost : ℕ  -- Cost of a single puzzle
  box_cost : ℕ    -- Cost of a box of puzzles
  box_size : ℕ    -- Number of puzzles in a box
  required : ℕ    -- Number of puzzles required

/-- Calculates the minimum cost for purchasing the required number of puzzles -/
def minCost (p : PuzzlePurchase) : ℕ :=
  let boxes := p.required / p.box_size
  let singles := p.required % p.box_size
  boxes * p.box_cost + singles * p.single_cost

/-- Theorem stating that the minimum cost for 25 puzzles is $210 -/
theorem puzzle_min_cost :
  let p : PuzzlePurchase := {
    single_cost := 10,
    box_cost := 50,
    box_size := 6,
    required := 25
  }
  minCost p = 210 := by
  sorry


end puzzle_min_cost_l3104_310403


namespace car_average_speed_l3104_310425

/-- The average speed of a car given its speeds for two hours -/
theorem car_average_speed (speed1 speed2 : ℝ) (h : speed1 = 145 ∧ speed2 = 60) :
  (speed1 + speed2) / 2 = 102.5 := by
  sorry

end car_average_speed_l3104_310425


namespace number_problem_l3104_310485

theorem number_problem (x : ℚ) : x^2 + 105 = (x - 19)^2 → x = 128/19 := by
  sorry

end number_problem_l3104_310485


namespace quotient_calculation_l3104_310413

theorem quotient_calculation (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) 
  (h1 : dividend = 149)
  (h2 : divisor = 16)
  (h3 : remainder = 5)
  (h4 : dividend = divisor * 9 + remainder) :
  9 = dividend / divisor := by
sorry

end quotient_calculation_l3104_310413


namespace max_guests_left_l3104_310410

/-- Represents a guest with their galoshes size -/
structure Guest where
  size : ℕ

/-- Represents the state of galoshes in the hallway -/
structure GaloshesState where
  sizes : Finset ℕ

/-- Defines when a guest can wear a pair of galoshes -/
def canWear (g : Guest) (s : ℕ) : Prop := g.size ≤ s

/-- Defines the initial state with 10 guests and their galoshes -/
def initialState : Finset Guest × GaloshesState :=
  sorry

/-- Simulates guests leaving and wearing galoshes -/
def guestsLeave (state : Finset Guest × GaloshesState) : Finset Guest × GaloshesState :=
  sorry

/-- Checks if any remaining guest can wear any remaining galoshes -/
def canAnyGuestLeave (state : Finset Guest × GaloshesState) : Prop :=
  sorry

theorem max_guests_left (final_state : Finset Guest × GaloshesState) :
  final_state = guestsLeave initialState →
  ¬canAnyGuestLeave final_state →
  Finset.card final_state.1 ≤ 5 :=
sorry

end max_guests_left_l3104_310410


namespace star_difference_l3104_310475

def star (x y : ℤ) : ℤ := x * y + 3 * x - y

theorem star_difference : (star 7 4) - (star 4 7) = 12 := by sorry

end star_difference_l3104_310475


namespace smallest_n_square_cube_l3104_310477

theorem smallest_n_square_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 3 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 3 * x = y^2) → (∃ (z : ℕ), 5 * x = z^3) → x ≥ n) ∧
  n = 675 := by
sorry

end smallest_n_square_cube_l3104_310477


namespace number_equals_five_times_difference_l3104_310414

theorem number_equals_five_times_difference : ∃! x : ℝ, x = 5 * (x - 4) := by
  sorry

end number_equals_five_times_difference_l3104_310414


namespace sqrt_one_implies_one_l3104_310404

theorem sqrt_one_implies_one (a : ℝ) : Real.sqrt a = 1 → a = 1 := by
  sorry

end sqrt_one_implies_one_l3104_310404


namespace mother_twice_age_2040_l3104_310445

/-- The year when Tina's mother's age is twice Tina's age -/
def year_mother_twice_age (tina_birth_year : ℕ) (tina_age_2010 : ℕ) (mother_age_multiplier_2010 : ℕ) : ℕ :=
  tina_birth_year + (mother_age_multiplier_2010 - 2) * tina_age_2010

theorem mother_twice_age_2040 :
  year_mother_twice_age 2000 10 5 = 2040 := by
  sorry

#eval year_mother_twice_age 2000 10 5

end mother_twice_age_2040_l3104_310445


namespace range_of_a_l3104_310435

def P (a : ℝ) : Set ℝ := {x | a - 4 < x ∧ x < a + 4}

def Q : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Q → x ∈ P a) ↔ -1 ≤ a ∧ a ≤ 5 :=
by sorry

end range_of_a_l3104_310435


namespace number_of_hens_l3104_310465

theorem number_of_hens (total_heads total_feet num_hens num_cows : ℕ) 
  (total_heads_eq : total_heads = 48)
  (total_feet_eq : total_feet = 144)
  (min_hens : num_hens ≥ 10)
  (min_cows : num_cows ≥ 5)
  (total_animals_eq : num_hens + num_cows = total_heads)
  (total_feet_calc : 2 * num_hens + 4 * num_cows = total_feet) :
  num_hens = 24 := by
sorry

end number_of_hens_l3104_310465


namespace exists_arrangement_for_23_l3104_310495

/-- Fibonacci-like sequence defined by F_0 = 0, F_1 = 1, F_i = 3F_{i-1} - F_{i-2} for i ≥ 2 -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required conditions for P = 23 -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), F 0 = 0 ∧ F 1 = 1 ∧ 
  (∀ n : ℕ, n ≥ 2 → F n = 3 * F (n - 1) - F (n - 2)) ∧ F 12 % 23 = 0 :=
sorry

end exists_arrangement_for_23_l3104_310495


namespace factor_w6_minus_81_l3104_310460

theorem factor_w6_minus_81 (w : ℝ) : 
  w^6 - 81 = (w - 3) * (w^2 + 3*w + 9) * (w^3 + 9) := by sorry

end factor_w6_minus_81_l3104_310460


namespace sum_of_digits_8_pow_2004_l3104_310497

theorem sum_of_digits_8_pow_2004 : ∃ (n : ℕ), 
  8^2004 % 100 = n ∧ (n / 10 + n % 10 = 7) := by sorry

end sum_of_digits_8_pow_2004_l3104_310497


namespace problem_solution_l3104_310448

theorem problem_solution : 
  ((2023 - Real.sqrt 5) ^ 0 - 2 + abs (Real.sqrt 3 - 1) = Real.sqrt 3 - 2) ∧
  ((Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) + (Real.sqrt 15 * Real.sqrt 3) / Real.sqrt 5 = 2) := by
sorry

end problem_solution_l3104_310448


namespace not_difference_of_squares_l3104_310421

/-- The difference of squares formula cannot be directly applied to (-x+y)(x-y) -/
theorem not_difference_of_squares (x y : ℝ) : 
  ¬ ∃ (a b : ℝ), (-x + y) * (x - y) = a^2 - b^2 :=
sorry

end not_difference_of_squares_l3104_310421


namespace plane_sphere_sum_l3104_310438

-- Define the origin
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Define the fixed point (2a, 2b, 2c)
def fixed_point (a b c : ℝ) : ℝ × ℝ × ℝ := (2*a, 2*b, 2*c)

-- Define the points A, B, C on the axes
def A (α : ℝ) : ℝ × ℝ × ℝ := (α, 0, 0)
def B (β : ℝ) : ℝ × ℝ × ℝ := (0, β, 0)
def C (γ : ℝ) : ℝ × ℝ × ℝ := (0, 0, γ)

-- Define the center of the sphere
def sphere_center (p q r : ℝ) : ℝ × ℝ × ℝ := (p, q, r)

-- State the theorem
theorem plane_sphere_sum (a b c p q r α β γ : ℝ) 
  (h1 : A α ≠ O) (h2 : B β ≠ O) (h3 : C γ ≠ O)
  (h4 : sphere_center p q r ≠ O)
  (h5 : ∃ (t : ℝ), t * (2*a) / α + t * (2*b) / β + t * (2*c) / γ = t) 
  (h6 : ∀ (x y z : ℝ), (x - p)^2 + (y - q)^2 + (z - r)^2 = p^2 + q^2 + r^2 → 
    (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = α ∧ y = 0 ∧ z = 0) ∨ 
    (x = 0 ∧ y = β ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = γ)) :
  (2*a)/p + (2*b)/q + (2*c)/r = 2 := by
sorry

end plane_sphere_sum_l3104_310438


namespace teacher_total_score_l3104_310493

/-- Calculates the total score of a teacher based on their written test and interview scores -/
def calculate_total_score (written_score : ℝ) (interview_score : ℝ) 
  (written_weight : ℝ) (interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

/-- Theorem: The teacher's total score is 72 points -/
theorem teacher_total_score : 
  let written_score : ℝ := 80
  let interview_score : ℝ := 60
  let written_weight : ℝ := 0.6
  let interview_weight : ℝ := 0.4
  calculate_total_score written_score interview_score written_weight interview_weight = 72 := by
sorry

end teacher_total_score_l3104_310493


namespace paper_folding_ratio_l3104_310494

theorem paper_folding_ratio : 
  let square_side : ℝ := 8
  let folded_height : ℝ := square_side / 2
  let folded_width : ℝ := square_side
  let cut_height : ℝ := folded_height / 3
  let small_rect_height : ℝ := cut_height
  let small_rect_width : ℝ := folded_width
  let large_rect_height : ℝ := folded_height - cut_height
  let large_rect_width : ℝ := folded_width
  let small_rect_perimeter : ℝ := 2 * (small_rect_height + small_rect_width)
  let large_rect_perimeter : ℝ := 2 * (large_rect_height + large_rect_width)
  small_rect_perimeter / large_rect_perimeter = 7 / 11 := by
sorry

end paper_folding_ratio_l3104_310494


namespace no_distinct_naturals_satisfying_equation_l3104_310447

theorem no_distinct_naturals_satisfying_equation :
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + 1 : ℚ) / a = ((b + 1 : ℚ) / b + (c + 1 : ℚ) / c) / 2 := by
  sorry

end no_distinct_naturals_satisfying_equation_l3104_310447


namespace distance_O_to_J_l3104_310416

/-- A right triangle with its circumcircle and incircle -/
structure RightTriangleWithCircles where
  /-- The center of the circumcircle -/
  O : ℝ × ℝ
  /-- The center of the incircle -/
  I : ℝ × ℝ
  /-- The radius of the circumcircle -/
  R : ℝ
  /-- The radius of the incircle -/
  r : ℝ
  /-- The vertex of the right angle -/
  C : ℝ × ℝ
  /-- The point symmetric to C with respect to I -/
  J : ℝ × ℝ
  /-- Ensure that C is the right angle vertex -/
  right_angle : (C.1 - O.1)^2 + (C.2 - O.2)^2 = R^2
  /-- Ensure that J is symmetric to C with respect to I -/
  symmetry : J.1 - I.1 = I.1 - C.1 ∧ J.2 - I.2 = I.2 - C.2

/-- The theorem to be proved -/
theorem distance_O_to_J (t : RightTriangleWithCircles) : 
  ((t.O.1 - t.J.1)^2 + (t.O.2 - t.J.2)^2)^(1/2) = t.R - 2 * t.r := by
  sorry

end distance_O_to_J_l3104_310416


namespace regular_polygon_sides_l3104_310417

theorem regular_polygon_sides (D : ℕ) : D = 20 → ∃ (n : ℕ), n > 2 ∧ D = n * (n - 3) / 2 ∧ n = 8 := by
  sorry

end regular_polygon_sides_l3104_310417


namespace combination_problem_l3104_310480

theorem combination_problem (n : ℕ) : 
  n * (n - 1) = 42 → n.choose 3 = 35 := by
sorry

end combination_problem_l3104_310480


namespace cody_game_count_l3104_310444

def final_game_count (initial_games : ℕ) (games_to_jake : ℕ) (games_to_sarah : ℕ) (new_games : ℕ) : ℕ :=
  initial_games - (games_to_jake + games_to_sarah) + new_games

theorem cody_game_count :
  final_game_count 9 4 2 3 = 6 := by
  sorry

end cody_game_count_l3104_310444


namespace fishing_and_camping_l3104_310491

/-- Represents the fishing and camping problem -/
theorem fishing_and_camping
  (total_fish_weight : ℝ)
  (wastage_percentage : ℝ)
  (adult_consumption : ℝ)
  (child_consumption : ℝ)
  (adult_child_ratio : ℚ)
  (max_campers : ℕ)
  (h1 : total_fish_weight = 44)
  (h2 : wastage_percentage = 0.2)
  (h3 : adult_consumption = 3)
  (h4 : child_consumption = 1)
  (h5 : adult_child_ratio = 2 / 5)
  (h6 : max_campers = 12) :
  ∃ (adult_campers child_campers : ℕ),
    adult_campers = 2 ∧
    child_campers = 5 ∧
    adult_campers + child_campers ≤ max_campers ∧
    (adult_campers : ℚ) / (child_campers : ℚ) = adult_child_ratio ∧
    (adult_campers : ℝ) * adult_consumption + (child_campers : ℝ) * child_consumption ≤
      total_fish_weight * (1 - wastage_percentage) :=
by sorry

end fishing_and_camping_l3104_310491


namespace triangle_inequality_l3104_310449

/-- Given a triangle with side lengths a, b, c and area T, 
    prove that a^2 + b^2 + c^2 ≥ 4√3 T, 
    with equality if and only if the triangle is equilateral -/
theorem triangle_inequality (a b c T : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : T > 0)
  (h_T : T = Real.sqrt ((a + b + c) * (a + b - c) * (b + c - a) * (c + a - b)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * T ∧ 
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * T ↔ a = b ∧ b = c) :=
by sorry

end triangle_inequality_l3104_310449


namespace square_circle_overlap_ratio_l3104_310470

theorem square_circle_overlap_ratio (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let square_side := 2 * r
  let overlap_area := square_side^2
  overlap_area / circle_area = 4 / π :=
by sorry

end square_circle_overlap_ratio_l3104_310470


namespace interest_rate_calculation_l3104_310453

/-- Given a principal amount and an interest rate, proves that if the simple interest
    for 2 years is $600 and the compound interest for 2 years is $609,
    then the interest rate is 3% per annum. -/
theorem interest_rate_calculation (P r : ℝ) : 
  P * r * 2 = 600 →
  P * ((1 + r)^2 - 1) = 609 →
  r = 0.03 := by
sorry

end interest_rate_calculation_l3104_310453


namespace sum_proper_divisors_243_l3104_310487

theorem sum_proper_divisors_243 : 
  (Finset.filter (fun x => x ≠ 243 ∧ 243 % x = 0) (Finset.range 244)).sum id = 121 := by
  sorry

end sum_proper_divisors_243_l3104_310487


namespace angle_sum_theorem_l3104_310400

theorem angle_sum_theorem (x₁ x₂ : Real) (h₁ : 0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi) 
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi) 
  (eq₁ : Real.sin x₁ ^ 3 - Real.cos x₁ ^ 3 = (1 / Real.cos x₁) - (1 / Real.sin x₁))
  (eq₂ : Real.sin x₂ ^ 3 - Real.cos x₂ ^ 3 = (1 / Real.cos x₂) - (1 / Real.sin x₂)) :
  x₁ + x₂ = 3 * Real.pi / 2 := by
sorry

end angle_sum_theorem_l3104_310400


namespace oprah_band_total_weight_l3104_310466

/-- Represents the Oprah Winfrey High School marching band -/
structure MarchingBand where
  trumpet_count : ℕ
  clarinet_count : ℕ
  trombone_count : ℕ
  tuba_count : ℕ
  drum_count : ℕ
  trumpet_weight : ℕ
  clarinet_weight : ℕ
  trombone_weight : ℕ
  tuba_weight : ℕ
  drum_weight : ℕ

/-- Calculates the total weight carried by the marching band -/
def total_weight (band : MarchingBand) : ℕ :=
  band.trumpet_count * band.trumpet_weight +
  band.clarinet_count * band.clarinet_weight +
  band.trombone_count * band.trombone_weight +
  band.tuba_count * band.tuba_weight +
  band.drum_count * band.drum_weight

/-- The Oprah Winfrey High School marching band configuration -/
def oprah_band : MarchingBand := {
  trumpet_count := 6
  clarinet_count := 9
  trombone_count := 8
  tuba_count := 3
  drum_count := 2
  trumpet_weight := 5
  clarinet_weight := 5
  trombone_weight := 10
  tuba_weight := 20
  drum_weight := 15
}

/-- Theorem stating that the total weight carried by the Oprah Winfrey High School marching band is 245 pounds -/
theorem oprah_band_total_weight :
  total_weight oprah_band = 245 := by
  sorry

end oprah_band_total_weight_l3104_310466


namespace exists_empty_subsquare_l3104_310467

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- Checks if a point is inside a square -/
def isInside (p : Point) (s : Square) : Prop :=
  s.bottomLeft.x ≤ p.x ∧ p.x < s.bottomLeft.x + s.sideLength ∧
  s.bottomLeft.y ≤ p.y ∧ p.y < s.bottomLeft.y + s.sideLength

theorem exists_empty_subsquare (bigSquare : Square) (points : Finset Point) :
  bigSquare.sideLength = 4 →
  points.card = 15 →
  (∀ p ∈ points, isInside p bigSquare) →
  ∃ (smallSquare : Square),
    smallSquare.sideLength = 1 ∧
    isInside smallSquare.bottomLeft bigSquare ∧
    (∀ p ∈ points, ¬isInside p smallSquare) :=
by sorry

end exists_empty_subsquare_l3104_310467


namespace circle_area_after_radius_multiplication_area_of_new_circle_l3104_310442

/-- Theorem: Area of a circle after radius multiplication -/
theorem circle_area_after_radius_multiplication (A : ℝ) (k : ℝ) :
  A > 0 → k > 0 → (k * (A / Real.pi).sqrt)^2 * Real.pi = k^2 * A := by
  sorry

/-- The area of a circle with radius multiplied by 5 -/
theorem area_of_new_circle (original_area : ℝ) (new_area : ℝ) :
  original_area = 30 →
  new_area = (5 * (original_area / Real.pi).sqrt)^2 * Real.pi →
  new_area = 750 := by
  sorry

end circle_area_after_radius_multiplication_area_of_new_circle_l3104_310442


namespace ellipse1_properties_ellipse2_properties_l3104_310436

-- Part 1
def ellipse1 (x y : ℝ) : Prop := x^2/15 + y^2/10 = 1

def given_ellipse (x y : ℝ) : Prop := 4*x^2 + 9*y^2 = 36

theorem ellipse1_properties :
  (∀ x y, ellipse1 x y → (x = 3 ∧ y = -2)) ∧
  (∀ x y, ellipse1 x y → ∃ c, c^2 = 5 ∧ 
    ∃ a b, a^2 = 15 ∧ b^2 = 10 ∧ c^2 = a^2 - b^2) :=
sorry

-- Part 2
def ellipse2 (x y : ℝ) : Prop := x^2/16 + y^2/8 = 1

def origin : ℝ × ℝ := (0, 0)
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0
def eccentricity (e : ℝ) : Prop := e = Real.sqrt 2 / 2
def triangle_perimeter (p : ℝ) : Prop := p = 16

theorem ellipse2_properties :
  (∀ x y, ellipse2 x y → ∃ c, c > 0 ∧ 
    (∃ f1 f2 : ℝ × ℝ, f1 = (-c, 0) ∧ f2 = (c, 0) ∧
      on_x_axis f1 ∧ on_x_axis f2)) ∧
  eccentricity (Real.sqrt 2 / 2) ∧
  (∃ p, triangle_perimeter p) :=
sorry

end ellipse1_properties_ellipse2_properties_l3104_310436


namespace malt_shop_shakes_l3104_310427

/-- Given a malt shop scenario where:
  * Each shake uses 4 ounces of chocolate syrup
  * Each cone uses 6 ounces of chocolate syrup
  * 1 cone was sold
  * A total of 14 ounces of chocolate syrup was used
  Prove that 2 shakes were sold. -/
theorem malt_shop_shakes : 
  ∀ (shakes : ℕ), 
    (4 * shakes + 6 * 1 = 14) → shakes = 2 := by
  sorry

end malt_shop_shakes_l3104_310427


namespace remainder_equality_l3104_310429

theorem remainder_equality (A B D S S' s s' : ℕ) : 
  A > B →
  (A + 3) % D = S →
  (B - 2) % D = S' →
  ((A + 3) * (B - 2)) % D = s →
  (S * S') % D = s' →
  s = s' := by sorry

end remainder_equality_l3104_310429


namespace orange_purchase_calculation_l3104_310434

/-- The amount of oranges initially planned to be purchased -/
def initial_purchase : ℝ := sorry

/-- The total amount of oranges purchased over three weeks -/
def total_purchase : ℝ := 75

theorem orange_purchase_calculation :
  initial_purchase = 14 :=
by
  have week1 : ℝ := initial_purchase + 5
  have week2 : ℝ := 2 * initial_purchase
  have week3 : ℝ := 2 * initial_purchase
  have total_equation : week1 + week2 + week3 = total_purchase := by sorry
  sorry

end orange_purchase_calculation_l3104_310434


namespace units_digit_problem_l3104_310402

theorem units_digit_problem : (8 * 25 * 983 - 8^3) % 10 = 8 := by
  sorry

end units_digit_problem_l3104_310402


namespace sqrt_equation_solution_l3104_310450

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (5 + 11 - 7) = Real.sqrt (5 + 11) - Real.sqrt x → x = 1 := by
  sorry

end sqrt_equation_solution_l3104_310450


namespace f_and_g_properties_l3104_310405

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define g as the derivative of f
def g : ℝ → ℝ := f'

-- Axioms based on the problem conditions
axiom f_diff : ∀ x, HasDerivAt f (f' x) x
axiom f_even : ∀ x, f (3/2 - 2*x) = f (3/2 + 2*x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- Theorem to prove
theorem f_and_g_properties :
  f (-1) = f 4 ∧ g (-1/2) = 0 :=
sorry

end f_and_g_properties_l3104_310405


namespace count_ordered_pairs_eq_six_l3104_310419

/-- The number of ordered pairs of positive integers (M, N) satisfying M/8 = 4/N -/
def count_ordered_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 32) (Finset.product (Finset.range 33) (Finset.range 33))).card

theorem count_ordered_pairs_eq_six : count_ordered_pairs = 6 := by
  sorry

end count_ordered_pairs_eq_six_l3104_310419


namespace carrie_harvest_money_l3104_310443

/-- Calculates the total money earned from selling tomatoes and carrots -/
def totalMoney (numTomatoes : ℕ) (numCarrots : ℕ) (priceTomato : ℚ) (priceCarrot : ℚ) : ℚ :=
  numTomatoes * priceTomato + numCarrots * priceCarrot

/-- Proves that the total money earned is correct for Carrie's harvest -/
theorem carrie_harvest_money :
  totalMoney 200 350 1 (3/2) = 725 := by
  sorry

end carrie_harvest_money_l3104_310443


namespace two_equidistant_points_l3104_310490

/-- A line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Three distinct lines in a plane -/
structure ThreeLines :=
  (l₁ : Line)
  (l₂ : Line)
  (l₃ : Line)
  (distinct : l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃)

/-- l₂ intersects l₁ -/
def intersects (l₁ l₂ : Line) : Prop :=
  l₁.slope ≠ l₂.slope

/-- l₃ is parallel to l₁ -/
def parallel (l₁ l₃ : Line) : Prop :=
  l₁.slope = l₃.slope

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A point is equidistant from three lines -/
def equidistant (p : Point) (lines : ThreeLines) : Prop := sorry

/-- The main theorem -/
theorem two_equidistant_points (lines : ThreeLines) 
  (h₁ : intersects lines.l₁ lines.l₂)
  (h₂ : parallel lines.l₁ lines.l₃) :
  ∃! (p₁ p₂ : Point), p₁ ≠ p₂ ∧ 
    equidistant p₁ lines ∧ 
    equidistant p₂ lines ∧
    ∀ (p : Point), equidistant p lines → p = p₁ ∨ p = p₂ :=
sorry

end two_equidistant_points_l3104_310490


namespace three_to_six_minus_one_prime_factors_l3104_310496

theorem three_to_six_minus_one_prime_factors :
  let n := 3^6 - 1
  ∃ (p q r : Nat), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    n % p = 0 ∧ n % q = 0 ∧ n % r = 0 ∧
    (∀ (s : Nat), Nat.Prime s → n % s = 0 → s = p ∨ s = q ∨ s = r) ∧
    p + q + r = 22 :=
by sorry

end three_to_six_minus_one_prime_factors_l3104_310496


namespace unique_c_value_l3104_310418

theorem unique_c_value (c : ℝ) : c + ⌊c⌋ = 23.2 → c = 11.7 := by
  sorry

end unique_c_value_l3104_310418


namespace largest_quotient_is_30_l3104_310422

def S : Set Int := {-30, -5, -1, 0, 3, 9}

theorem largest_quotient_is_30 : 
  ∀ a b : Int, a ∈ S → b ∈ S → b ≠ 0 → a / b ≤ 30 :=
by
  sorry

end largest_quotient_is_30_l3104_310422


namespace max_F_value_l3104_310479

def is_eternal_number (M : ℕ) : Prop :=
  M ≥ 1000 ∧ M < 10000 ∧
  (M / 100 % 10 + M / 10 % 10 + M % 10 = 12)

def N (M : ℕ) : ℕ :=
  (M / 1000) * 100 + (M / 100 % 10) * 1000 + (M / 10 % 10) + (M % 10) * 10

def F (M : ℕ) : ℚ :=
  (M - N M) / 9

theorem max_F_value (M : ℕ) :
  is_eternal_number M →
  (M / 100 % 10 - M % 10 = M / 1000) →
  (F M / 9).isInt →
  F M ≤ 9 :=
sorry

end max_F_value_l3104_310479


namespace max_path_length_l3104_310420

/-- A rectangular prism with dimensions 1, 2, and 3 -/
structure RectangularPrism where
  length : ℝ := 1
  width : ℝ := 2
  height : ℝ := 3

/-- A path in the rectangular prism -/
structure PrismPath (p : RectangularPrism) where
  -- The path starts and ends at the same corner
  start_end_same : Bool
  -- The path visits each corner exactly once
  visits_all_corners_once : Bool
  -- The path consists of straight lines between corners
  straight_lines : Bool
  -- The length of the path
  length : ℝ

/-- The theorem stating the maximum path length in the rectangular prism -/
theorem max_path_length (p : RectangularPrism) :
  ∃ (path : PrismPath p), 
    path.start_end_same ∧ 
    path.visits_all_corners_once ∧ 
    path.straight_lines ∧
    path.length = 2 * Real.sqrt 14 + 4 * Real.sqrt 13 ∧
    ∀ (other_path : PrismPath p), 
      other_path.start_end_same ∧ 
      other_path.visits_all_corners_once ∧ 
      other_path.straight_lines → 
      other_path.length ≤ path.length :=
sorry

end max_path_length_l3104_310420


namespace cube_root_equation_solution_l3104_310406

theorem cube_root_equation_solution :
  ∀ x : ℝ, (10 - 6 * x)^(1/3 : ℝ) = -2 → x = 3 := by
  sorry

end cube_root_equation_solution_l3104_310406


namespace jian_has_second_most_l3104_310439

-- Define the number of notebooks for each person
def jian_notebooks : ℕ := 3
def doyun_notebooks : ℕ := 5
def siu_notebooks : ℕ := 2

-- Define a function to determine if a person has the second most notebooks
def has_second_most (x y z : ℕ) : Prop :=
  (x > y ∧ x < z) ∨ (x > z ∧ x < y)

-- Theorem statement
theorem jian_has_second_most :
  has_second_most jian_notebooks siu_notebooks doyun_notebooks :=
sorry

end jian_has_second_most_l3104_310439


namespace triangle_perimeter_l3104_310472

/-- Proves that a triangle with inradius 2.5 cm and area 45 cm² has a perimeter of 36 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 45 → A = r * (p / 2) → p = 36 := by
  sorry

end triangle_perimeter_l3104_310472


namespace smallest_divisible_by_79_and_83_l3104_310459

theorem smallest_divisible_by_79_and_83 :
  ∃ (m : ℕ), 
    m > 0 ∧
    79 ∣ (m^3 - 3*m^2 + 2*m) ∧
    83 ∣ (m^3 - 3*m^2 + 2*m) ∧
    (∀ (k : ℕ), k > 0 ∧ k < m → ¬(79 ∣ (k^3 - 3*k^2 + 2*k) ∧ 83 ∣ (k^3 - 3*k^2 + 2*k))) ∧
    m = 3715 :=
by sorry

end smallest_divisible_by_79_and_83_l3104_310459


namespace fifteen_sided_figure_area_l3104_310464

/-- A fifteen-sided figure on a 1 cm × 1 cm graph paper -/
structure FifteenSidedFigure where
  full_squares : ℕ
  small_triangles : ℕ
  h_full_squares : full_squares = 10
  h_small_triangles : small_triangles = 10

/-- The area of the fifteen-sided figure is 15 cm² -/
theorem fifteen_sided_figure_area (fig : FifteenSidedFigure) : 
  (fig.full_squares : ℝ) + (fig.small_triangles : ℝ) / 2 = 15 := by
  sorry

end fifteen_sided_figure_area_l3104_310464


namespace exponent_product_simplification_l3104_310478

theorem exponent_product_simplification :
  (10 ^ 0.5) * (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.05) * (10 ^ 1.05) = 100 := by
  sorry

end exponent_product_simplification_l3104_310478


namespace quadratic_square_of_binomial_l3104_310426

theorem quadratic_square_of_binomial (a : ℚ) :
  (∃ r s : ℚ, ∀ x, a * x^2 + 22 * x + 9 = (r * x + s)^2) →
  a = 121 / 9 := by
sorry

end quadratic_square_of_binomial_l3104_310426


namespace quadratic_function_range_l3104_310455

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- State the theorem
theorem quadratic_function_range (a : ℝ) : 
  (∀ x ∈ Set.Ioc 0 1, |f a x| ≤ 1) → a ∈ Set.Icc (-2) 0 \ {0} :=
sorry

end quadratic_function_range_l3104_310455


namespace cubic_roots_sum_l3104_310457

theorem cubic_roots_sum (u v w : ℝ) : 
  (u - Real.rpow 17 (1/3 : ℝ)) * (u - Real.rpow 67 (1/3 : ℝ)) * (u - Real.rpow 137 (1/3 : ℝ)) = 2/5 ∧
  (v - Real.rpow 17 (1/3 : ℝ)) * (v - Real.rpow 67 (1/3 : ℝ)) * (v - Real.rpow 137 (1/3 : ℝ)) = 2/5 ∧
  (w - Real.rpow 17 (1/3 : ℝ)) * (w - Real.rpow 67 (1/3 : ℝ)) * (w - Real.rpow 137 (1/3 : ℝ)) = 2/5 ∧
  u ≠ v ∧ u ≠ w ∧ v ≠ w →
  u^3 + v^3 + w^3 = 221 + 6/5 - 3 * 1549 :=
by sorry


end cubic_roots_sum_l3104_310457


namespace initial_speed_proof_l3104_310468

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 10

/-- The height of the building in meters -/
def h : ℝ := 180

/-- The time taken to fall the last 60 meters in seconds -/
def t : ℝ := 1

/-- The distance fallen in the last second in meters -/
def d : ℝ := 60

/-- The initial downward speed of the object in m/s -/
def v₀ : ℝ := 25

theorem initial_speed_proof : 
  ∃ (v : ℝ), v = v₀ ∧ 
  d = (v + v₀) / 2 * t ∧ 
  v^2 = v₀^2 + 2 * g * (h - d) :=
sorry

end initial_speed_proof_l3104_310468


namespace part_i_part_ii_l3104_310441

-- Define propositions P and Q
def P (m : ℝ) : Prop := ∀ x ∈ Set.Icc (-1 : ℝ) 1, -x^2 + 3*m - 1 ≤ 0

def Q (m a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, m - a*x ≤ 0

-- Part (i)
theorem part_i (m : ℝ) : 
  (¬(P m) ∧ ¬(Q m 1) ∧ (P m ∨ Q m 1)) → (1/3 < m ∧ m ≤ 1) :=
sorry

-- Part (ii)
theorem part_ii (m a : ℝ) :
  ((P m → Q m a) ∧ ¬(Q m a → P m)) → (a ≥ 1/3 ∨ a ≤ -1/3) :=
sorry

end part_i_part_ii_l3104_310441


namespace trundic_word_count_l3104_310423

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 15

/-- The maximum word length -/
def max_word_length : ℕ := 5

/-- The number of required letters (A and B) -/
def required_letters : ℕ := 2

/-- Calculates the number of valid words in the Trundic language -/
def count_valid_words (alphabet_size : ℕ) (max_word_length : ℕ) (required_letters : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid words in the Trundic language -/
theorem trundic_word_count :
  count_valid_words alphabet_size max_word_length required_letters = 35180 :=
sorry

end trundic_word_count_l3104_310423


namespace comparison_proofs_l3104_310482

theorem comparison_proofs :
  (-2.3 < 2.4) ∧ (-3/4 > -5/6) ∧ (0 > -Real.pi) := by
  sorry

end comparison_proofs_l3104_310482


namespace all_hungarian_teams_face_foreign_l3104_310412

-- Define the total number of teams
def total_teams : ℕ := 8

-- Define the number of Hungarian teams
def hungarian_teams : ℕ := 3

-- Define the number of foreign teams
def foreign_teams : ℕ := total_teams - hungarian_teams

-- Define the probability of all Hungarian teams facing foreign opponents
def prob_all_hungarian_foreign : ℚ := 4/7

-- Theorem statement
theorem all_hungarian_teams_face_foreign :
  (foreign_teams.choose hungarian_teams * hungarian_teams.factorial) / 
  (total_teams.choose 2 * (total_teams / 2).factorial) = prob_all_hungarian_foreign := by
  sorry

end all_hungarian_teams_face_foreign_l3104_310412


namespace angle_bisector_length_right_triangle_l3104_310433

theorem angle_bisector_length_right_triangle (a b c : ℝ) (h1 : a = 15) (h2 : b = 20) (h3 : c = 25) 
  (h4 : a^2 + b^2 = c^2) : ∃ (AA₁ : ℝ), AA₁ = (20 * Real.sqrt 10) / 3 :=
by
  sorry

end angle_bisector_length_right_triangle_l3104_310433


namespace logarithm_identity_l3104_310483

theorem logarithm_identity (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  Real.log x / Real.log (a * b) = (Real.log x / Real.log a * Real.log x / Real.log b) /
    (Real.log x / Real.log a + Real.log x / Real.log b) := by
  sorry

end logarithm_identity_l3104_310483


namespace area_ratio_lateral_angle_relation_area_ratio_bounds_l3104_310481

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  -- Add necessary fields here
  mk ::

/-- The ratio of cross-section area to lateral surface area -/
def area_ratio (p : RegularQuadPyramid) : ℝ := sorry

/-- The angle between two adjacent lateral faces -/
def lateral_face_angle (p : RegularQuadPyramid) : ℝ := sorry

/-- Theorem about the relationship between area ratio and lateral face angle -/
theorem area_ratio_lateral_angle_relation (p : RegularQuadPyramid) :
  lateral_face_angle p = Real.arccos (8 * (area_ratio p)^2 - 1) :=
sorry

/-- Theorem about the permissible values of the area ratio -/
theorem area_ratio_bounds (p : RegularQuadPyramid) :
  0 < area_ratio p ∧ area_ratio p < Real.sqrt 2 / 4 :=
sorry

end area_ratio_lateral_angle_relation_area_ratio_bounds_l3104_310481


namespace real_number_line_bijection_l3104_310407

-- Define the set of points on a number line
def NumberLine : Type := ℝ

-- State the theorem
theorem real_number_line_bijection : 
  ∃ f : ℝ → NumberLine, Function.Bijective f := by sorry

end real_number_line_bijection_l3104_310407
