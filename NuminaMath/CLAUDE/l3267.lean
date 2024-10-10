import Mathlib

namespace exists_student_with_eight_sessions_l3267_326797

/-- A structure representing a club with students and sessions. -/
structure Club where
  students : Finset Nat
  sessions : Finset Nat
  attended : Nat → Finset Nat
  meet_once : ∀ s₁ s₂, s₁ ∈ students → s₂ ∈ students → s₁ ≠ s₂ →
    ∃! session, session ∈ sessions ∧ s₁ ∈ attended session ∧ s₂ ∈ attended session
  not_all_in_one : ∀ session, session ∈ sessions → ∃ s, s ∈ students ∧ s ∉ attended session

/-- Theorem stating that in a club satisfying the given conditions,
    there exists a student who attended at least 8 sessions. -/
theorem exists_student_with_eight_sessions (c : Club) (h : c.students.card = 50) :
  ∃ s, s ∈ c.students ∧ (c.sessions.filter (fun session => s ∈ c.attended session)).card ≥ 8 :=
sorry

end exists_student_with_eight_sessions_l3267_326797


namespace awards_assignment_count_l3267_326796

/-- The number of different types of awards -/
def num_awards : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 8

/-- The total number of ways to assign awards -/
def total_assignments : ℕ := num_awards ^ num_students

/-- Theorem stating that the total number of assignments is 65536 -/
theorem awards_assignment_count :
  total_assignments = 65536 := by
  sorry

end awards_assignment_count_l3267_326796


namespace geometric_sequence_sum_l3267_326707

-- Define a geometric sequence with common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

-- Theorem statement
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a → a 1 + a 2 = 3 → a 4 + a 5 = 24 := by
  sorry

end geometric_sequence_sum_l3267_326707


namespace estate_distribution_l3267_326704

/-- Represents the estate distribution problem --/
theorem estate_distribution (total : ℚ) 
  (daughter_share : ℚ) (son_share : ℚ) (husband_share : ℚ) (gardener_share : ℚ) : 
  daughter_share + son_share = (3 : ℚ) / 5 * total →
  daughter_share = (3 : ℚ) / 5 * (daughter_share + son_share) →
  husband_share = 3 * son_share →
  gardener_share = 600 →
  total = daughter_share + son_share + husband_share + gardener_share →
  total = 1875 := by
  sorry

end estate_distribution_l3267_326704


namespace dice_sum_possibilities_l3267_326746

/-- The number of dice being rolled -/
def num_dice : ℕ := 4

/-- The minimum value on a die face -/
def min_face : ℕ := 1

/-- The maximum value on a die face -/
def max_face : ℕ := 6

/-- The minimum possible sum when rolling the dice -/
def min_sum : ℕ := num_dice * min_face

/-- The maximum possible sum when rolling the dice -/
def max_sum : ℕ := num_dice * max_face

/-- The number of distinct possible sums when rolling the dice -/
def num_distinct_sums : ℕ := max_sum - min_sum + 1

theorem dice_sum_possibilities : num_distinct_sums = 21 := by
  sorry

end dice_sum_possibilities_l3267_326746


namespace smallest_integer_inequality_l3267_326726

theorem smallest_integer_inequality : ∀ x : ℤ, x + 5 < 3*x - 9 → x ≥ 8 ∧ 8 + 5 < 3*8 - 9 := by sorry

end smallest_integer_inequality_l3267_326726


namespace division_of_mixed_number_by_fraction_l3267_326700

theorem division_of_mixed_number_by_fraction :
  (2 + 1 / 4 : ℚ) / (2 / 3 : ℚ) = 27 / 8 := by sorry

end division_of_mixed_number_by_fraction_l3267_326700


namespace seventh_observation_l3267_326753

theorem seventh_observation (n : Nat) (initial_avg : ℝ) (new_avg : ℝ) : 
  n = 6 →
  initial_avg = 15 →
  new_avg = initial_avg - 1 →
  (n * initial_avg + (n + 1) * new_avg) / (n + 1) = new_avg →
  (n + 1) * new_avg - n * initial_avg = 8 :=
by sorry

end seventh_observation_l3267_326753


namespace not_right_triangle_4_6_11_l3267_326747

/-- Checks if three line segments can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem: The line segments 4, 6, and 11 cannot form a right triangle -/
theorem not_right_triangle_4_6_11 : ¬ is_right_triangle 4 6 11 := by
  sorry

#check not_right_triangle_4_6_11

end not_right_triangle_4_6_11_l3267_326747


namespace units_digit_of_13_power_2003_l3267_326794

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to compute the units digit of 3^n
def unitsDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

-- State the theorem
theorem units_digit_of_13_power_2003 :
  unitsDigit (13^2003) = unitsDigitOf3Power 2003 :=
sorry

end units_digit_of_13_power_2003_l3267_326794


namespace binary_sum_equals_638_l3267_326783

/-- The sum of the binary numbers 111111111₂ and 1111111₂ is equal to 638 in base 10. -/
theorem binary_sum_equals_638 : 
  (2^9 - 1) + (2^7 - 1) = 638 := by
  sorry

#check binary_sum_equals_638

end binary_sum_equals_638_l3267_326783


namespace tangent_y_intercept_l3267_326719

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_y_intercept :
  let m := (3 : ℝ) * P.1^2  -- Slope of the tangent line
  let b := P.2 - m * P.1    -- y-intercept of the tangent line
  b = 9 := by sorry

end tangent_y_intercept_l3267_326719


namespace alice_win_probability_l3267_326709

-- Define the game types
inductive Move
| Rock
| Paper
| Scissors

-- Define the player types
inductive Player
| Alice
| Bob
| Other

-- Define the tournament structure
def TournamentSize : Nat := 8
def NumRounds : Nat := 3

-- Define the rules of the game
def beats (m1 m2 : Move) : Bool :=
  match m1, m2 with
  | Move.Rock, Move.Scissors => true
  | Move.Scissors, Move.Paper => true
  | Move.Paper, Move.Rock => true
  | _, _ => false

-- Define the strategy for each player
def playerMove (p : Player) : Move :=
  match p with
  | Player.Alice => Move.Rock
  | Player.Bob => Move.Paper
  | Player.Other => Move.Scissors

-- Define the probability of Alice winning
def aliceWinProbability : Rat := 6/7

-- Theorem statement
theorem alice_win_probability :
  (TournamentSize = 8) →
  (NumRounds = 3) →
  (∀ p, playerMove p = match p with
    | Player.Alice => Move.Rock
    | Player.Bob => Move.Paper
    | Player.Other => Move.Scissors) →
  (∀ m1 m2, beats m1 m2 = match m1, m2 with
    | Move.Rock, Move.Scissors => true
    | Move.Scissors, Move.Paper => true
    | Move.Paper, Move.Rock => true
    | _, _ => false) →
  aliceWinProbability = 6/7 := by
  sorry

end alice_win_probability_l3267_326709


namespace middle_speed_calculation_l3267_326737

/-- Represents the speed and duration of a part of the journey -/
structure JourneyPart where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (part : JourneyPart) : ℝ := part.speed * part.duration

theorem middle_speed_calculation (total_distance : ℝ) (first_part last_part middle_part : JourneyPart) 
  (h1 : total_distance = 800)
  (h2 : first_part.speed = 80 ∧ first_part.duration = 6)
  (h3 : last_part.speed = 40 ∧ last_part.duration = 2)
  (h4 : middle_part.duration = 4)
  (h5 : total_distance = distance first_part + distance middle_part + distance last_part) :
  middle_part.speed = 60 := by
sorry

end middle_speed_calculation_l3267_326737


namespace smallest_n_for_integer_T_l3267_326742

/-- Sum of reciprocals of prime digits -/
def P : ℚ := 1/2 + 1/3 + 1/5 + 1/7

/-- T_n is the sum of the reciprocals of the prime digits of integers from 1 to 5^n inclusive -/
def T (n : ℕ) : ℚ := n * (5^(n-1) : ℚ) * P

/-- 42 is the smallest positive integer n for which T_n is an integer -/
theorem smallest_n_for_integer_T : ∀ k : ℕ, k > 0 → (∃ m : ℤ, T k = m) → k ≥ 42 :=
sorry

end smallest_n_for_integer_T_l3267_326742


namespace company_employees_l3267_326703

theorem company_employees (december_employees : ℕ) (january_employees : ℕ) 
  (h1 : december_employees = 460) 
  (h2 : december_employees = january_employees + (january_employees * 15 / 100)) : 
  january_employees = 400 := by
sorry

end company_employees_l3267_326703


namespace binomial_12_choose_10_l3267_326750

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_choose_10_l3267_326750


namespace james_pages_per_year_l3267_326759

/-- Calculates the number of pages James writes in a year -/
def pages_per_year (pages_per_letter : ℕ) (friends : ℕ) (times_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  pages_per_letter * friends * times_per_week * weeks_per_year

/-- Proves that James writes 624 pages in a year -/
theorem james_pages_per_year :
  pages_per_year 3 2 2 52 = 624 := by
  sorry

end james_pages_per_year_l3267_326759


namespace T_recursive_relation_l3267_326791

/-- The number of binary strings of length n such that any 4 adjacent digits sum to at least 1 -/
def T (n : ℕ) : ℕ :=
  if n < 4 then
    match n with
    | 0 => 1  -- Convention: empty string is valid
    | 1 => 2  -- "0" and "1" are valid
    | 2 => 3  -- "00", "01", "10", "11" are valid except "00"
    | 3 => 6  -- All combinations except "0000"
    | _ => 0  -- Should never reach here
  else
    T (n - 1) + T (n - 2) + T (n - 3) + T (n - 4)

/-- The main theorem stating the recursive relation for T(n) when n ≥ 4 -/
theorem T_recursive_relation (n : ℕ) (h : n ≥ 4) :
  T n = T (n - 1) + T (n - 2) + T (n - 3) + T (n - 4) := by sorry

end T_recursive_relation_l3267_326791


namespace square_sum_given_diff_and_product_l3267_326733

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a * b = -15) : 
  a^2 + b^2 = 34 := by
  sorry

end square_sum_given_diff_and_product_l3267_326733


namespace ages_solution_l3267_326705

/-- Represents the current ages of Justin, Angelina, and Larry -/
structure Ages where
  justin : ℝ
  angelina : ℝ
  larry : ℝ

/-- The conditions given in the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.angelina = ages.justin + 4 ∧
  ages.angelina + 5 = 40 ∧
  ages.larry = ages.justin + 0.5 * ages.justin

/-- The theorem to be proved -/
theorem ages_solution (ages : Ages) :
  problem_conditions ages → ages.justin = 31 ∧ ages.larry = 46.5 := by
  sorry


end ages_solution_l3267_326705


namespace large_mean_small_variance_reflects_common_prosperity_l3267_326739

/-- Represents a personal income distribution --/
structure IncomeDistribution where
  mean : ℝ
  variance : ℝ
  mean_nonneg : 0 ≤ mean
  variance_nonneg : 0 ≤ variance

/-- Defines the concept of common prosperity --/
def common_prosperity (id : IncomeDistribution) : Prop :=
  id.mean > 0 ∧ id.variance < 1 -- Arbitrary thresholds for illustration

/-- Defines universal prosperity --/
def universal_prosperity (id : IncomeDistribution) : Prop :=
  id.mean > 0

/-- Defines elimination of polarization and poverty --/
def no_polarization_poverty (id : IncomeDistribution) : Prop :=
  id.variance < 1 -- Arbitrary threshold for illustration

/-- Theorem stating that large mean and small variance best reflect common prosperity --/
theorem large_mean_small_variance_reflects_common_prosperity
  (id : IncomeDistribution)
  (h1 : universal_prosperity id → common_prosperity id)
  (h2 : no_polarization_poverty id → common_prosperity id) :
  common_prosperity id ↔ (id.mean > 0 ∧ id.variance < 1) := by
  sorry

#check large_mean_small_variance_reflects_common_prosperity

end large_mean_small_variance_reflects_common_prosperity_l3267_326739


namespace product_inequality_l3267_326774

theorem product_inequality (a a' b b' c c' : ℝ) 
  (h1 : a * a' > 0) 
  (h2 : a * c ≥ b^2) 
  (h3 : a' * c' ≥ b'^2) : 
  (a + a') * (c + c') ≥ (b + b')^2 := by
  sorry

end product_inequality_l3267_326774


namespace divisible_by_25_l3267_326799

theorem divisible_by_25 (n : ℕ) : 25 ∣ (2^(n+2) * 3^n + 5*n - 4) := by
  sorry

end divisible_by_25_l3267_326799


namespace fraction_to_decimal_l3267_326715

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l3267_326715


namespace train_speed_l3267_326761

theorem train_speed (train_length : Real) (man_speed : Real) (passing_time : Real) :
  train_length = 220 →
  man_speed = 6 →
  passing_time = 12 →
  (train_length / 1000) / (passing_time / 3600) - man_speed = 60 :=
by
  sorry

end train_speed_l3267_326761


namespace distinct_quotients_exist_l3267_326714

/-- A function that checks if a number is composed of five twos and three ones -/
def is_valid_number (n : ℕ) : Prop :=
  (n.digits 10).count 2 = 5 ∧ (n.digits 10).count 1 = 3 ∧ (n.digits 10).length = 8

/-- The theorem statement -/
theorem distinct_quotients_exist : ∃ (a b c d e : ℕ),
  is_valid_number a ∧
  is_valid_number b ∧
  is_valid_number c ∧
  is_valid_number d ∧
  is_valid_number e ∧
  a % 7 = 0 ∧
  b % 7 = 0 ∧
  c % 7 = 0 ∧
  d % 7 = 0 ∧
  e % 7 = 0 ∧
  a / 7 ≠ b / 7 ∧
  a / 7 ≠ c / 7 ∧
  a / 7 ≠ d / 7 ∧
  a / 7 ≠ e / 7 ∧
  b / 7 ≠ c / 7 ∧
  b / 7 ≠ d / 7 ∧
  b / 7 ≠ e / 7 ∧
  c / 7 ≠ d / 7 ∧
  c / 7 ≠ e / 7 ∧
  d / 7 ≠ e / 7 :=
sorry

end distinct_quotients_exist_l3267_326714


namespace polynomial_composite_l3267_326728

def P (x : ℕ) : ℕ := 4*x^3 + 6*x^2 + 4*x + 1

theorem polynomial_composite : ∀ x : ℕ, ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ P x = a * b :=
sorry

end polynomial_composite_l3267_326728


namespace exactly_two_sunny_days_probability_l3267_326708

theorem exactly_two_sunny_days_probability :
  let days : ℕ := 3
  let rain_probability : ℚ := 60 / 100
  let sunny_probability : ℚ := 1 - rain_probability
  let ways_to_choose_two_days : ℕ := (days.choose 2)
  let probability_two_sunny_one_rainy : ℚ := sunny_probability^2 * rain_probability
  ways_to_choose_two_days * probability_two_sunny_one_rainy = 36 / 125 := by
  sorry

end exactly_two_sunny_days_probability_l3267_326708


namespace g_3_equals_9_l3267_326772

-- Define the function g
def g (x : ℝ) : ℝ := 3*x^6 - 2*x^4 + 5*x^2 - 7

-- Theorem statement
theorem g_3_equals_9 (h : g (-3) = 9) : g 3 = 9 := by
  sorry

end g_3_equals_9_l3267_326772


namespace rectangle_width_l3267_326780

theorem rectangle_width (perimeter : ℝ) (length_difference : ℝ) : perimeter = 46 → length_difference = 7 → 
  let length := (perimeter / 2 - length_difference) / 2
  let width := length + length_difference
  width = 15 := by sorry

end rectangle_width_l3267_326780


namespace sampling_methods_classification_l3267_326792

-- Define the characteristics of sampling methods
def is_systematic_sampling (method : String) : Prop :=
  method = "Samples at equal time intervals"

def is_simple_random_sampling (method : String) : Prop :=
  method = "Selects individuals from a small population with little difference among them"

-- Define the two sampling methods
def sampling_method_1 : String :=
  "Samples a bag for inspection every 30 minutes in a milk production line"

def sampling_method_2 : String :=
  "Selects 3 students from a group of 30 math enthusiasts in a middle school"

-- Theorem to prove
theorem sampling_methods_classification :
  is_systematic_sampling sampling_method_1 ∧
  is_simple_random_sampling sampling_method_2 := by
  sorry


end sampling_methods_classification_l3267_326792


namespace solve_equation_l3267_326773

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
  sorry

end solve_equation_l3267_326773


namespace sum_first_ten_primes_with_units_digit_three_l3267_326798

/-- A function that returns true if a number has a units digit of 3 -/
def hasUnitsDigitThree (n : ℕ) : Bool :=
  n % 10 = 3

/-- The sequence of prime numbers with a units digit of 3 -/
def primesWithUnitsDigitThree : List ℕ :=
  (List.range 200).filter (λ n => n.Prime && hasUnitsDigitThree n)

/-- The sum of the first ten prime numbers with a units digit of 3 -/
def sumFirstTenPrimesWithUnitsDigitThree : ℕ :=
  (primesWithUnitsDigitThree.take 10).sum

theorem sum_first_ten_primes_with_units_digit_three :
  sumFirstTenPrimesWithUnitsDigitThree = 639 := by
  sorry


end sum_first_ten_primes_with_units_digit_three_l3267_326798


namespace purely_imaginary_complex_number_l3267_326735

theorem purely_imaginary_complex_number (m : ℝ) :
  (((m * (m + 2)) / (m - 1) : ℂ) + (m^2 + m - 2) * I).re = 0 ∧
  (((m * (m + 2)) / (m - 1) : ℂ) + (m^2 + m - 2) * I).im ≠ 0 →
  m = 0 := by sorry

end purely_imaginary_complex_number_l3267_326735


namespace triangle_function_properties_l3267_326752

open Real

theorem triangle_function_properties (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sine_law : sin A / a = sin B / b ∧ sin B / b = sin C / c)
  (f : ℝ → ℝ)
  (h_f_def : ∀ x, f x = sin (2*x + B) + sqrt 3 * cos (2*x + B))
  (h_f_odd : ∀ x, f (x - π/3) = -f (-(x - π/3))) :
  B = π/3 ∧
  (∀ x, f x = 2 * sin (2*x + 2*π/3)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k*π - 7*π/12) (k*π - π/12), 
    ∀ y ∈ Set.Icc (k*π - 7*π/12) (k*π - π/12), 
    x < y → f x < f y) ∧
  (a = 1 → b = f 0 → (1/2) * a * b * sin C = sqrt 3 / 4) :=
by sorry

end triangle_function_properties_l3267_326752


namespace max_marks_calculation_l3267_326769

/-- Given a passing threshold, a student's score, and the shortfall to pass,
    calculate the maximum possible marks. -/
theorem max_marks_calculation (passing_threshold : ℚ) (score : ℕ) (shortfall : ℕ) :
  passing_threshold = 30 / 100 →
  score = 212 →
  shortfall = 28 →
  (score + shortfall) / passing_threshold = 800 :=
by sorry

end max_marks_calculation_l3267_326769


namespace trigonometric_problem_l3267_326734

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.sin α = 4 / 5)
  (h4 : Real.cos (α + β) = 5 / 13) :
  (Real.cos β = 63 / 65) ∧ 
  ((Real.sin α)^2 + Real.sin (2 * α)) / (Real.cos (2 * α) - 1) = -5 / 4 := by
sorry

end trigonometric_problem_l3267_326734


namespace sin_graph_symmetry_l3267_326776

theorem sin_graph_symmetry (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x)
  let g (x : ℝ) := f (x + π / 6)
  ∀ y : ℝ, g (π / 6 - x) = g (π / 6 + x) := by
sorry

end sin_graph_symmetry_l3267_326776


namespace sqrt_inequality_l3267_326713

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end sqrt_inequality_l3267_326713


namespace arithmetic_mean_problem_l3267_326795

/-- Given that 15 is the arithmetic mean of the set {7, 12, 19, 8, 10, y}, prove that y = 34 -/
theorem arithmetic_mean_problem (y : ℝ) : 
  (7 + 12 + 19 + 8 + 10 + y) / 6 = 15 → y = 34 := by
  sorry

end arithmetic_mean_problem_l3267_326795


namespace simple_interest_principal_calculation_l3267_326716

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (simple_interest : ℝ)
  (time : ℝ)
  (rate : ℝ)
  (h1 : simple_interest = 176)
  (h2 : time = 4)
  (h3 : rate = 5.5 / 100) :
  simple_interest = (800 : ℝ) * rate * time := by
sorry

end simple_interest_principal_calculation_l3267_326716


namespace largest_n_for_product_l3267_326785

/-- Arithmetic sequence (a_n) -/
def a (n : ℕ) (d : ℤ) : ℤ := 2 + (n - 1 : ℤ) * d

/-- Arithmetic sequence (b_n) -/
def b (n : ℕ) (e : ℤ) : ℤ := 3 + (n - 1 : ℤ) * e

theorem largest_n_for_product (d e : ℤ) (h1 : a 2 d ≤ b 2 e) :
  (∃ n : ℕ, a n d * b n e = 2728) →
  (∀ m : ℕ, a m d * b m e = 2728 → m ≤ 52) :=
by sorry

end largest_n_for_product_l3267_326785


namespace prob_two_tails_after_HHT_is_correct_l3267_326721

/-- A fair coin flip sequence that stops when two consecutive heads or tails are obtained -/
def CoinFlipSequence : Type := List Bool

/-- The probability of getting a specific sequence of coin flips -/
def prob_sequence (s : CoinFlipSequence) : ℚ :=
  (1 / 2) ^ s.length

/-- The probability of getting two tails after HHT -/
def prob_two_tails_after_HHT : ℚ :=
  1 / 24

/-- The theorem stating that the probability of getting two tails after HHT is 1/24 -/
theorem prob_two_tails_after_HHT_is_correct :
  prob_two_tails_after_HHT = 1 / 24 := by
  sorry

#check prob_two_tails_after_HHT_is_correct

end prob_two_tails_after_HHT_is_correct_l3267_326721


namespace cubic_equation_solution_l3267_326763

theorem cubic_equation_solution (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 24*x^3) 
  (h3 : a - b = x) : 
  a = (x*(3 + Real.sqrt 92))/6 ∨ a = (x*(3 - Real.sqrt 92))/6 := by
sorry

end cubic_equation_solution_l3267_326763


namespace amusement_park_running_cost_l3267_326755

/-- The amusement park problem -/
theorem amusement_park_running_cost 
  (initial_cost : ℝ) 
  (daily_tickets : ℕ) 
  (ticket_price : ℝ) 
  (days_to_breakeven : ℕ) 
  (h1 : initial_cost = 100000)
  (h2 : daily_tickets = 150)
  (h3 : ticket_price = 10)
  (h4 : days_to_breakeven = 200) :
  let daily_revenue := daily_tickets * ticket_price
  let total_revenue := daily_revenue * days_to_breakeven
  let daily_running_cost_percentage := 
    (total_revenue - initial_cost) / (initial_cost * days_to_breakeven) * 100
  daily_running_cost_percentage = 10 := by sorry

end amusement_park_running_cost_l3267_326755


namespace fraction_meaningful_l3267_326724

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 2) / (x - 3)) ↔ x ≠ 3 :=
by sorry

end fraction_meaningful_l3267_326724


namespace tangent_line_equation_range_of_a_l3267_326770

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 10

-- Theorem for the tangent line equation
theorem tangent_line_equation (a : ℝ) :
  a = 1 →
  ∃ (m b : ℝ), m = 8 ∧ b = -2 ∧
  ∀ (x y : ℝ), y = f a x → (x = 2 → m*x - y + b = 0) :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ (a : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f a x < 0) →
  a > 9/2 :=
sorry

end tangent_line_equation_range_of_a_l3267_326770


namespace script_lines_proof_l3267_326777

theorem script_lines_proof :
  ∀ (lines1 lines2 lines3 : ℕ),
  lines1 = 20 →
  lines1 = lines2 + 8 →
  lines2 = 3 * lines3 + 6 →
  lines3 = 2 :=
by
  sorry

end script_lines_proof_l3267_326777


namespace first_term_values_l3267_326712

def fibonacci_like_sequence (a b : ℕ) : ℕ → ℕ
  | 0 => a
  | 1 => b
  | (n + 2) => fibonacci_like_sequence a b n + fibonacci_like_sequence a b (n + 1)

theorem first_term_values (a b : ℕ) :
  fibonacci_like_sequence a b 2 = 7 ∧
  fibonacci_like_sequence a b 2013 % 4 = 1 →
  a = 1 ∨ a = 5 := by
sorry

end first_term_values_l3267_326712


namespace gary_chicken_multiple_l3267_326731

/-- The multiple of chickens Gary has now compared to the start -/
def chicken_multiple (initial_chickens : ℕ) (eggs_per_day : ℕ) (total_eggs_per_week : ℕ) : ℕ :=
  (total_eggs_per_week / (eggs_per_day * 7)) / initial_chickens

/-- Proof that Gary's chicken multiple is 8 -/
theorem gary_chicken_multiple :
  chicken_multiple 4 6 1344 = 8 := by
  sorry

end gary_chicken_multiple_l3267_326731


namespace evaluate_M_l3267_326757

theorem evaluate_M : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) + Real.sqrt (4 - 2 * Real.sqrt 3)
  M = 7/4 := by
sorry

end evaluate_M_l3267_326757


namespace f_monotonicity_and_inequality_l3267_326706

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / (Real.exp 1 * x)

theorem f_monotonicity_and_inequality (a : ℝ) :
  (∀ x y, 0 < x ∧ 0 < y ∧ x < y ∧ a ≤ 0 → f a x < f a y) ∧
  (a > 0 →
    (∀ x y, 0 < x ∧ x < y ∧ y < a / Real.exp 1 → f a y < f a x) ∧
    (∀ x y, a / Real.exp 1 < x ∧ x < y → f a x < f a y)) ∧
  (a = 2 → ∀ x, x > 0 → f a x > Real.exp (-x)) := by
  sorry

end f_monotonicity_and_inequality_l3267_326706


namespace intersection_M_N_l3267_326784

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x < 2}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end intersection_M_N_l3267_326784


namespace no_solution_exists_l3267_326720

/-- Represents the number of books for each subject -/
structure BookCounts where
  math : ℕ
  history : ℕ
  science : ℕ
  literature : ℕ

/-- The problem constraints -/
def satisfiesConstraints (books : BookCounts) : Prop :=
  books.math + books.history + books.science + books.literature = 80 ∧
  4 * books.math + 5 * books.history + 6 * books.science + 7 * books.literature = 520 ∧
  3 * books.history = 2 * books.math ∧
  2 * books.science = books.math ∧
  4 * books.literature = books.math

theorem no_solution_exists : ¬∃ (books : BookCounts), satisfiesConstraints books := by
  sorry

end no_solution_exists_l3267_326720


namespace sector_area_l3267_326766

/-- A sector with perimeter 12 cm and central angle 2 rad has an area of 9 cm² -/
theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 12 → central_angle = 2 → area = 9 := by
  sorry

#check sector_area

end sector_area_l3267_326766


namespace zoo_bus_seats_l3267_326751

theorem zoo_bus_seats (total_children : ℕ) (children_per_seat : ℕ) (seats_needed : ℕ) : 
  total_children = 58 → children_per_seat = 2 → seats_needed = total_children / children_per_seat → 
  seats_needed = 29 := by
sorry

end zoo_bus_seats_l3267_326751


namespace bandwidth_calculation_correct_l3267_326765

/-- Represents the parameters for an audio channel --/
structure AudioChannelParams where
  sessionDurationMinutes : ℕ
  samplingRate : ℕ
  samplingDepth : ℕ
  metadataBytes : ℕ
  metadataPerAudioKilobits : ℕ

/-- Calculates the required bandwidth for a stereo audio channel --/
def calculateBandwidth (params : AudioChannelParams) : ℚ :=
  let sessionDurationSeconds := params.sessionDurationMinutes * 60
  let dataVolume := params.samplingRate * params.samplingDepth * sessionDurationSeconds
  let metadataVolume := params.metadataBytes * 8 * dataVolume / (params.metadataPerAudioKilobits * 1024)
  let totalDataVolume := (dataVolume + metadataVolume) * 2
  totalDataVolume / (sessionDurationSeconds * 1024)

/-- Theorem stating that the calculated bandwidth matches the expected result --/
theorem bandwidth_calculation_correct (params : AudioChannelParams) 
  (h1 : params.sessionDurationMinutes = 51)
  (h2 : params.samplingRate = 63)
  (h3 : params.samplingDepth = 17)
  (h4 : params.metadataBytes = 47)
  (h5 : params.metadataPerAudioKilobits = 5) :
  calculateBandwidth params = 2.25 := by
  sorry

#eval calculateBandwidth {
  sessionDurationMinutes := 51,
  samplingRate := 63,
  samplingDepth := 17,
  metadataBytes := 47,
  metadataPerAudioKilobits := 5
}

end bandwidth_calculation_correct_l3267_326765


namespace scaled_vector_is_monomial_l3267_326762

/-- A vector in ℝ² -/
def vector : ℝ × ℝ := (1, 5)

/-- The scalar multiple of the vector -/
def scaled_vector : ℝ × ℝ := (-3 * vector.1, -3 * vector.2)

/-- Definition of a monomial in this context -/
def is_monomial (v : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ) (n : ℕ × ℕ), v = (c * n.1, c * n.2)

theorem scaled_vector_is_monomial : is_monomial scaled_vector := by
  sorry

end scaled_vector_is_monomial_l3267_326762


namespace time_to_cut_one_piece_l3267_326730

-- Define the total number of pieces
def total_pieces : ℕ := 146

-- Define the total time taken in seconds
def total_time : ℕ := 580

-- Define the time taken to cut one piece
def time_per_piece : ℚ := total_time / total_pieces

-- Theorem to prove
theorem time_to_cut_one_piece : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) ∧ |time_per_piece - 4| < ε :=
sorry

end time_to_cut_one_piece_l3267_326730


namespace f_f_7_equals_0_l3267_326779

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_4 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

theorem f_f_7_equals_0 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_periodic : is_periodic_4 f)
  (h_f_1 : f 1 = 4) :
  f (f 7) = 0 := by
  sorry

end f_f_7_equals_0_l3267_326779


namespace log_equation_solution_l3267_326732

theorem log_equation_solution (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x - 2*y > 0) :
  2 * Real.log (x - 2*y) = Real.log x + Real.log y → x / y = 4 := by
sorry

end log_equation_solution_l3267_326732


namespace tetrahedron_max_volume_edge_ratio_l3267_326781

/-- Given a tetrahedron with volume V and edge lengths a, b, c, d where no three edges are coplanar,
    and L = a + b + c + d, the maximum value of V/L^3 is √2/2592 -/
theorem tetrahedron_max_volume_edge_ratio :
  ∀ (V a b c d L : ℝ),
  V > 0 → a > 0 → b > 0 → c > 0 → d > 0 →
  (∀ (x y z : ℝ), x + y + z ≠ a + b + c + d) →  -- No three edges are coplanar
  L = a + b + c + d →
  (∃ (V' : ℝ), V' = V ∧ V' / L^3 ≤ Real.sqrt 2 / 2592) :=
by sorry

end tetrahedron_max_volume_edge_ratio_l3267_326781


namespace notebook_duration_is_seven_l3267_326722

/-- Represents the number of weeks John's notebooks last -/
def notebook_duration (
  num_notebooks : ℕ
  ) (pages_per_notebook : ℕ
  ) (math_pages_per_day : ℕ
  ) (math_days_per_week : ℕ
  ) (science_pages_per_day : ℕ
  ) (science_days_per_week : ℕ
  ) (history_pages_per_day : ℕ
  ) (history_days_per_week : ℕ
  ) : ℕ :=
  let total_pages := num_notebooks * pages_per_notebook
  let pages_per_week := 
    math_pages_per_day * math_days_per_week +
    science_pages_per_day * science_days_per_week +
    history_pages_per_day * history_days_per_week
  total_pages / pages_per_week

theorem notebook_duration_is_seven :
  notebook_duration 5 40 4 3 5 2 6 1 = 7 := by
  sorry

end notebook_duration_is_seven_l3267_326722


namespace club_membership_l3267_326764

theorem club_membership (total : Nat) (left_handed : Nat) (jazz_lovers : Nat) (right_handed_jazz_dislikers : Nat) :
  total = 25 →
  left_handed = 12 →
  jazz_lovers = 18 →
  right_handed_jazz_dislikers = 3 →
  (∃ (left_handed_jazz_lovers : Nat),
    left_handed_jazz_lovers +
    (left_handed - left_handed_jazz_lovers) +
    (jazz_lovers - left_handed_jazz_lovers) +
    right_handed_jazz_dislikers = total ∧
    left_handed_jazz_lovers = 8) := by
  sorry

#check club_membership

end club_membership_l3267_326764


namespace power_of_power_l3267_326758

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l3267_326758


namespace shaded_area_is_75_l3267_326778

-- Define the side lengths of the squares
def larger_side : ℝ := 10
def smaller_side : ℝ := 5

-- Define the areas of the squares
def larger_area : ℝ := larger_side ^ 2
def smaller_area : ℝ := smaller_side ^ 2

-- Define the shaded area
def shaded_area : ℝ := larger_area - smaller_area

-- Theorem to prove
theorem shaded_area_is_75 : shaded_area = 75 := by
  sorry

end shaded_area_is_75_l3267_326778


namespace expression_value_l3267_326749

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (3 * x^4 + 4 * y^2) / 12 = 25.5833 := by
  sorry

end expression_value_l3267_326749


namespace max_intersected_edges_is_twelve_l3267_326738

/-- A regular 10-sided prism -/
structure RegularDecagonalPrism where
  -- We don't need to define the internal structure,
  -- as the problem doesn't require specific properties beyond it being a regular 10-sided prism

/-- A plane in 3D space -/
structure Plane where
  -- We don't need to define the internal structure of a plane

/-- The number of edges a plane intersects with a prism -/
def intersected_edges (prism : RegularDecagonalPrism) (plane : Plane) : ℕ :=
  sorry -- Definition not provided, as it's not explicitly given in the problem

/-- The maximum number of edges that can be intersected by any plane -/
def max_intersected_edges (prism : RegularDecagonalPrism) : ℕ :=
  sorry -- Definition not provided, as it's not explicitly given in the problem

/-- Theorem: The maximum number of edges of a regular 10-sided prism 
    that can be intersected by a plane is 12 -/
theorem max_intersected_edges_is_twelve (prism : RegularDecagonalPrism) :
  max_intersected_edges prism = 12 := by
  sorry

-- The proof is omitted as per instructions

end max_intersected_edges_is_twelve_l3267_326738


namespace special_rectangle_sides_l3267_326786

/-- A rectangle with special properties -/
structure SpecialRectangle where
  -- The length of the rectangle
  l : ℝ
  -- The width of the rectangle
  w : ℝ
  -- The perimeter of the rectangle is 24
  perimeter : l + w = 12
  -- M is the midpoint of BC
  midpoint : w / 2 = w / 2
  -- MA is perpendicular to MD
  perpendicular : l ^ 2 + (w / 2) ^ 2 = l ^ 2 + (w / 2) ^ 2

/-- The sides of a special rectangle are 4 and 8 -/
theorem special_rectangle_sides (r : SpecialRectangle) : r.l = 4 ∧ r.w = 8 := by
  sorry

#check special_rectangle_sides

end special_rectangle_sides_l3267_326786


namespace triangle_property_l3267_326767

theorem triangle_property (a b c A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b * Real.cos A + Real.sqrt 3 * b * Real.sin A - c - a = 0 →
  b = Real.sqrt 3 →
  B = π / 3 ∧ ∀ (a' c' : ℝ), a' + c' ≤ 2 * Real.sqrt 3 := by
  sorry

end triangle_property_l3267_326767


namespace opposite_of_2023_l3267_326745

theorem opposite_of_2023 : 
  ∀ y : ℤ, (2023 + y = 0) ↔ (y = -2023) := by sorry

end opposite_of_2023_l3267_326745


namespace equation_proof_l3267_326736

theorem equation_proof : 300 * 2 + (12 + 4) * (1 / 8) = 602 := by
  sorry

end equation_proof_l3267_326736


namespace min_value_of_function_l3267_326790

theorem min_value_of_function (x : ℝ) (h : x ∈ Set.Ico 1 2) : 
  (1 / x + 1 / (2 - x)) ≥ 2 ∧ 
  (1 / x + 1 / (2 - x) = 2 ↔ x = 1) :=
by sorry

end min_value_of_function_l3267_326790


namespace expenditure_difference_l3267_326718

theorem expenditure_difference
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percent : ℝ)
  (purchased_quantity_percent : ℝ)
  (h1 : price_increase_percent = 25)
  (h2 : purchased_quantity_percent = 72)
  : (1 + price_increase_percent / 100) * (purchased_quantity_percent / 100) - 1 = -0.1 := by
  sorry

end expenditure_difference_l3267_326718


namespace statement_b_statement_c_not_statement_a_not_statement_d_l3267_326727

-- Statement B
theorem statement_b (a b : ℝ) : a > b → a - 1 > b - 2 := by sorry

-- Statement C
theorem statement_c (a b c : ℝ) (h : c ≠ 0) : a / c^2 > b / c^2 → a > b := by sorry

-- Disproof of Statement A
theorem not_statement_a : ¬ (∀ a b c : ℝ, a > b → a * c^2 > b * c^2) := by sorry

-- Disproof of Statement D
theorem not_statement_d : ¬ (∀ a b : ℝ, a > b → a^2 > b^2) := by sorry

end statement_b_statement_c_not_statement_a_not_statement_d_l3267_326727


namespace fraction_equivalence_l3267_326775

theorem fraction_equivalence : 
  ∀ (n : ℚ), n = 1/2 → (4 - n) / (7 - n) = 3/5 := by
  sorry

end fraction_equivalence_l3267_326775


namespace state_selection_difference_l3267_326756

theorem state_selection_difference (total_candidates : ℕ) 
  (selection_rate_A selection_rate_B : ℚ) : 
  total_candidates = 8000 →
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B * total_candidates : ℚ) - (selection_rate_A * total_candidates : ℚ) = 80 := by
  sorry

end state_selection_difference_l3267_326756


namespace remainder_19_power_1999_mod_25_l3267_326729

theorem remainder_19_power_1999_mod_25 : 19^1999 % 25 = 4 := by
  sorry

end remainder_19_power_1999_mod_25_l3267_326729


namespace complex_trajectory_line_l3267_326711

theorem complex_trajectory_line (z : ℂ) :
  Complex.abs (z + 1) = Complex.abs (1 + Complex.I * z) →
  (z.re : ℝ) + z.im = 0 := by
sorry

end complex_trajectory_line_l3267_326711


namespace tangent_circle_equation_l3267_326740

/-- A circle with center on the line y = x and tangent to lines x + y = 0 and x + y + 4 = 0 -/
structure TangentCircle where
  a : ℝ
  center_on_diagonal : a = a
  tangent_to_first_line : |2 * a| / Real.sqrt 2 = |0 - 0| / Real.sqrt 2
  tangent_to_second_line : |2 * a| / Real.sqrt 2 = |4| / Real.sqrt 2

/-- The equation of the circle described by TangentCircle is (x+1)² + (y+1)² = 2 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ x y : ℝ, (x + 1)^2 + (y + 1)^2 = 2 ↔ 
  (x - (-1))^2 + (y - (-1))^2 = (Real.sqrt 2)^2 :=
by sorry

end tangent_circle_equation_l3267_326740


namespace compound_interest_multiple_l3267_326741

theorem compound_interest_multiple (P r m : ℝ) 
  (h1 : P * r^2 = 40)
  (h2 : P * (m * r)^2 = 360) :
  m = 3 := by
sorry

end compound_interest_multiple_l3267_326741


namespace replaced_sailor_weight_l3267_326787

/-- Given 8 sailors, if replacing one sailor with a new 64 kg sailor increases
    the average weight by 1 kg, then the replaced sailor's weight was 56 kg. -/
theorem replaced_sailor_weight
  (num_sailors : ℕ)
  (new_sailor_weight : ℕ)
  (avg_weight_increase : ℚ)
  (h1 : num_sailors = 8)
  (h2 : new_sailor_weight = 64)
  (h3 : avg_weight_increase = 1)
  : ℕ :=
by
  sorry

#check replaced_sailor_weight

end replaced_sailor_weight_l3267_326787


namespace cross_pollinated_percentage_l3267_326754

/-- Represents the apple orchard with Fuji and Gala trees -/
structure Orchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The percentage of cross-pollinated trees in the orchard is 2/3 -/
theorem cross_pollinated_percentage (o : Orchard) : 
  o.totalTrees = o.pureFuji + o.pureGala + o.crossPollinated →
  o.pureFuji + o.crossPollinated = 170 →
  o.pureFuji = 3 * o.totalTrees / 4 →
  o.pureGala = 30 →
  o.crossPollinated * 3 = o.totalTrees * 2 := by
  sorry

#check cross_pollinated_percentage

end cross_pollinated_percentage_l3267_326754


namespace monomial_combination_l3267_326723

theorem monomial_combination (m n : ℤ) : 
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 3 * a^4 * b^(n+2) = 5 * a^(m-1) * b^(2*n+3)) → 
  m + n = 4 := by
sorry

end monomial_combination_l3267_326723


namespace segment_length_l3267_326701

-- Define the line segment AB and points P and Q
structure Segment where
  length : ℝ

structure Point where
  position : ℝ

-- Define the ratios for P and Q
def ratio_P : ℚ := 3 / 7
def ratio_Q : ℚ := 4 / 9

-- State the theorem
theorem segment_length 
  (AB : Segment) 
  (P Q : Point) 
  (h1 : P.position ≤ Q.position) -- P and Q are on the same side of the midpoint
  (h2 : P.position = ratio_P * AB.length) -- P divides AB in ratio 3:4
  (h3 : Q.position = ratio_Q * AB.length) -- Q divides AB in ratio 4:5
  (h4 : Q.position - P.position = 3) -- PQ = 3
  : AB.length = 189 := by
  sorry


end segment_length_l3267_326701


namespace cannot_form_triangle_l3267_326744

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the third side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that line segments 4, 5, and 9 cannot form a triangle -/
theorem cannot_form_triangle : ¬(can_form_triangle 4 5 9) := by
  sorry


end cannot_form_triangle_l3267_326744


namespace trajectory_is_straight_line_l3267_326793

/-- The set of points (x, y) in ℝ² where x + y = 0 forms a straight line -/
theorem trajectory_is_straight_line :
  {p : ℝ × ℝ | p.1 + p.2 = 0} = {p : ℝ × ℝ | ∃ (t : ℝ), p = (t, -t)} := by
  sorry

end trajectory_is_straight_line_l3267_326793


namespace sum_of_factorials_1_to_10_l3267_326743

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_of_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => sum_of_factorials n + factorial (n + 1)

theorem sum_of_factorials_1_to_10 : sum_of_factorials 10 = 4037913 := by
  sorry

end sum_of_factorials_1_to_10_l3267_326743


namespace pens_per_student_is_five_l3267_326760

-- Define the given constants
def num_students : ℕ := 30
def notebooks_per_student : ℕ := 3
def binders_per_student : ℕ := 1
def highlighters_per_student : ℕ := 2
def pen_cost : ℚ := 0.5
def notebook_cost : ℚ := 1.25
def binder_cost : ℚ := 4.25
def highlighter_cost : ℚ := 0.75
def teacher_discount : ℚ := 100
def total_spent : ℚ := 260

-- Define the theorem
theorem pens_per_student_is_five :
  let cost_per_student_excl_pens := notebooks_per_student * notebook_cost + 
                                    binders_per_student * binder_cost + 
                                    highlighters_per_student * highlighter_cost
  let total_cost_excl_pens := num_students * cost_per_student_excl_pens
  let total_spent_before_discount := total_spent + teacher_discount
  let total_spent_on_pens := total_spent_before_discount - total_cost_excl_pens
  let total_pens := total_spent_on_pens / pen_cost
  let pens_per_student := total_pens / num_students
  pens_per_student = 5 := by sorry

end pens_per_student_is_five_l3267_326760


namespace cosh_inequality_l3267_326771

theorem cosh_inequality (c : ℝ) :
  (∀ x : ℝ, (Real.exp x + Real.exp (-x)) / 2 ≤ Real.exp (c * x^2)) ↔ c ≥ (1/2 : ℝ) := by
  sorry

end cosh_inequality_l3267_326771


namespace positive_number_square_sum_l3267_326782

theorem positive_number_square_sum : ∃ n : ℕ+, (n : ℝ)^2 + 2*(n : ℝ) = 170 ∧ n = 12 := by
  sorry

end positive_number_square_sum_l3267_326782


namespace geometric_sequence_common_ratio_l3267_326710

/-- A geometric sequence with a₂ = 8 and a₅ = 64 has a common ratio of 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence definition
  a 2 = 8 →                              -- Given condition
  a 5 = 64 →                             -- Given condition
  a 2 / a 1 = 2 :=                       -- Common ratio q = 2
by sorry

end geometric_sequence_common_ratio_l3267_326710


namespace squirrel_survey_l3267_326789

theorem squirrel_survey (total : ℕ) 
  (harmful_belief_rate : ℚ) 
  (attack_belief_rate : ℚ) 
  (wrong_believers : ℕ) 
  (h1 : harmful_belief_rate = 883 / 1000) 
  (h2 : attack_belief_rate = 538 / 1000) 
  (h3 : wrong_believers = 28) :
  (↑wrong_believers / (harmful_belief_rate * attack_belief_rate) : ℚ).ceil = total → 
  total = 59 := by
sorry

end squirrel_survey_l3267_326789


namespace bottom_row_is_2143_l3267_326748

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Fin 4

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  -- Each number appears exactly once per row and column
  (∀ i j k, i ≠ k → g i j ≠ g k j) ∧
  (∀ i j k, j ≠ k → g i j ≠ g i k) ∧
  -- L-shaped sum constraints
  g 0 0 + g 0 1 = 3 ∧
  g 0 3 + g 1 3 = 6 ∧
  g 2 1 + g 3 1 = 5

-- Define the bottom row
def bottom_row (g : Grid) : Fin 4 → Fin 4 := g 3

-- Theorem stating the bottom row forms 2143
theorem bottom_row_is_2143 (g : Grid) (h : is_valid_grid g) :
  (bottom_row g 0, bottom_row g 1, bottom_row g 2, bottom_row g 3) = (2, 1, 4, 3) :=
sorry

end bottom_row_is_2143_l3267_326748


namespace kitten_weight_l3267_326717

theorem kitten_weight (kitten lighter_dog heavier_dog : ℝ) 
  (h1 : kitten + lighter_dog + heavier_dog = 36)
  (h2 : kitten + heavier_dog = 3 * lighter_dog)
  (h3 : kitten + lighter_dog = (1/2) * heavier_dog) :
  kitten = 3 := by
  sorry

end kitten_weight_l3267_326717


namespace sqrt_sum_equals_sqrt_of_sum_sqrt_l3267_326768

theorem sqrt_sum_equals_sqrt_of_sum_sqrt (a b : ℚ) :
  (Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3)) ↔
  ((a = 1/2 ∧ b = 3/2) ∨ (a = 3/2 ∧ b = 1/2)) :=
sorry

end sqrt_sum_equals_sqrt_of_sum_sqrt_l3267_326768


namespace min_value_and_monotonicity_l3267_326788

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x ^ 2 - a * x

theorem min_value_and_monotonicity (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ f (-2 * Real.exp 1) x = 3 ∧ 
    ∀ (y : ℝ), y > 0 → f (-2 * Real.exp 1) y ≥ f (-2 * Real.exp 1) x) ∧
  (∀ (x y : ℝ), 0 < x ∧ x < y → (f a x ≥ f a y ↔ a ≥ 2 / Real.exp 1)) :=
sorry

end min_value_and_monotonicity_l3267_326788


namespace cube_plus_reciprocal_cube_l3267_326725

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
  sorry

end cube_plus_reciprocal_cube_l3267_326725


namespace some_number_equation_l3267_326702

theorem some_number_equation (y : ℝ) : 
  ∃ (n : ℝ), n * (1 + y) + 17 = n * (-1 + y) - 21 ∧ n = -19 := by
  sorry

end some_number_equation_l3267_326702
