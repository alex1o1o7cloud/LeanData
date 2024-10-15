import Mathlib

namespace NUMINAMATH_CALUDE_work_problem_l2065_206564

/-- Proves that 8 men became absent given the conditions of the work problem -/
theorem work_problem (total_men : ℕ) (original_days : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 42)
  (h2 : original_days = 17)
  (h3 : actual_days = 21)
  (h4 : total_men * original_days = (total_men - absent_men) * actual_days) :
  absent_men = 8 := by
  sorry

#check work_problem

end NUMINAMATH_CALUDE_work_problem_l2065_206564


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_97_l2065_206507

theorem gcd_of_powers_of_97 :
  Nat.gcd (97^10 + 1) (97^10 + 97^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_97_l2065_206507


namespace NUMINAMATH_CALUDE_bowling_ball_weight_calculation_l2065_206560

-- Define the weight of a canoe
def canoe_weight : ℚ := 32

-- Define the number of canoes and bowling balls
def num_canoes : ℕ := 4
def num_bowling_balls : ℕ := 5

-- Define the total weight of canoes
def total_canoe_weight : ℚ := num_canoes * canoe_weight

-- Define the weight of one bowling ball
def bowling_ball_weight : ℚ := total_canoe_weight / num_bowling_balls

-- Theorem statement
theorem bowling_ball_weight_calculation :
  bowling_ball_weight = 128 / 5 := by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_calculation_l2065_206560


namespace NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l2065_206570

open Complex

theorem complex_exp_13pi_div_2 : exp (13 * π / 2 * I) = I := by sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l2065_206570


namespace NUMINAMATH_CALUDE_sqrt_6_equality_l2065_206532

theorem sqrt_6_equality : (3 : ℝ) / Real.sqrt 6 = Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_6_equality_l2065_206532


namespace NUMINAMATH_CALUDE_ratio_of_segments_l2065_206506

/-- Given points A, B, C, D, and E on a line in that order, prove the ratio of AC to BD -/
theorem ratio_of_segments (A B C D E : ℝ) : 
  A < B → B < C → C < D → D < E →  -- Points lie on a line in order
  B - A = 3 →                      -- AB = 3
  C - B = 7 →                      -- BC = 7
  E - D = 4 →                      -- DE = 4
  D - A = 17 →                     -- AD = 17
  (C - A) / (D - B) = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l2065_206506


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2065_206579

/-- Given a geometric sequence {a_n} with sum S_n of first n terms, prove the general formula. -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 3 = 3/2 →                   -- given a_3
  S 3 = 9/2 →                   -- given S_3
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula for geometric sequence
  (q = 1 ∨ q = -1/2) ∧
  (∀ n, (q = 1 → a n = 3/2) ∧ 
        (q = -1/2 → a n = 6 * (-1/2)^(n-1))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2065_206579


namespace NUMINAMATH_CALUDE_no_consecutive_heads_probability_sum_of_numerator_and_denominator_l2065_206582

/-- Number of valid sequences of length n ending in Tails -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n+2) => a (n+1) + a n

/-- Number of valid sequences of length n ending in Heads -/
def b : ℕ → ℕ
| 0 => 0
| (n+1) => a n

/-- Total number of valid sequences of length n -/
def valid_sequences (n : ℕ) : ℕ := a n + b n

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

theorem no_consecutive_heads_probability :
  (valid_sequences 10 : ℚ) / (total_sequences 10 : ℚ) = 9 / 64 :=
sorry

theorem sum_of_numerator_and_denominator : 9 + 64 = 73 :=
sorry

end NUMINAMATH_CALUDE_no_consecutive_heads_probability_sum_of_numerator_and_denominator_l2065_206582


namespace NUMINAMATH_CALUDE_sqrt_164_between_12_and_13_l2065_206591

theorem sqrt_164_between_12_and_13 : 12 < Real.sqrt 164 ∧ Real.sqrt 164 < 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_164_between_12_and_13_l2065_206591


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2065_206537

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ) (X : ℕ),
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt 11 + E * Real.sqrt X) / F ∧
    F > 0 ∧
    A + B + C + D + E + F = 20 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2065_206537


namespace NUMINAMATH_CALUDE_john_scores_42_points_l2065_206522

/-- Calculates the total points scored by John given the specified conditions -/
def total_points_scored (shots_per_interval : ℕ) (points_per_shot : ℕ) (three_point_shots : ℕ) 
                        (interval_duration : ℕ) (num_periods : ℕ) (period_duration : ℕ) : ℕ :=
  let total_time := num_periods * period_duration
  let num_intervals := total_time / interval_duration
  let points_per_interval := shots_per_interval * points_per_shot + three_point_shots * 3
  num_intervals * points_per_interval

/-- Theorem stating that John scores 42 points under the given conditions -/
theorem john_scores_42_points : 
  total_points_scored 2 2 1 4 2 12 = 42 := by
  sorry


end NUMINAMATH_CALUDE_john_scores_42_points_l2065_206522


namespace NUMINAMATH_CALUDE_sun_city_population_relation_l2065_206515

/-- The population of Willowdale city -/
def willowdale_population : ℕ := 2000

/-- The population of Roseville city -/
def roseville_population : ℕ := 3 * willowdale_population - 500

/-- The population of Sun City -/
def sun_city_population : ℕ := 12000

/-- The multiple of Roseville City's population that Sun City has 1000 more than -/
def multiple : ℚ := (sun_city_population - 1000) / roseville_population

theorem sun_city_population_relation :
  sun_city_population = multiple * roseville_population + 1000 ∧ multiple = 2 := by sorry

end NUMINAMATH_CALUDE_sun_city_population_relation_l2065_206515


namespace NUMINAMATH_CALUDE_original_fraction_value_l2065_206544

theorem original_fraction_value (n : ℚ) : 
  (n + 1) / (n + 6) = 7 / 12 → n / (n + 5) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_value_l2065_206544


namespace NUMINAMATH_CALUDE_division_equality_l2065_206555

theorem division_equality (h : 29.94 / 1.45 = 17.1) : 2994 / 14.5 = 171 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l2065_206555


namespace NUMINAMATH_CALUDE_candies_given_away_l2065_206593

/-- Given a girl who initially had 60 candies and now has 20 left, 
    prove that she gave away 40 candies. -/
theorem candies_given_away (initial : ℕ) (remaining : ℕ) (given_away : ℕ) : 
  initial = 60 → remaining = 20 → given_away = initial - remaining → given_away = 40 := by
  sorry

end NUMINAMATH_CALUDE_candies_given_away_l2065_206593


namespace NUMINAMATH_CALUDE_swimming_pool_count_l2065_206530

theorem swimming_pool_count (total : ℕ) (garage : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 90 → garage = 50 → both = 35 → neither = 35 → 
  ∃ (pool : ℕ), pool = 40 ∧ 
    total = garage + pool - both + neither :=
by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_count_l2065_206530


namespace NUMINAMATH_CALUDE_limit_special_function_l2065_206538

/-- The limit of (4^(5x) - 9^(-2x)) / (sin(x) - tan(x^3)) as x approaches 0 is ln(1024 * 81) -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ →
    |(4^(5*x) - 9^(-2*x)) / (Real.sin x - Real.tan (x^3)) - Real.log (1024 * 81)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_special_function_l2065_206538


namespace NUMINAMATH_CALUDE_function_value_proof_l2065_206541

theorem function_value_proof (f : ℝ → ℝ) (h : ∀ x, f (1 - 2*x) = 1 / x^2) :
  f (1/2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_function_value_proof_l2065_206541


namespace NUMINAMATH_CALUDE_percentage_on_rent_is_14_l2065_206566

/-- Calculates the percentage of remaining income spent on house rent --/
def percentage_on_rent (total_income : ℚ) (petrol_expense : ℚ) (rent_expense : ℚ) : ℚ :=
  let remaining_income := total_income - petrol_expense
  (rent_expense / remaining_income) * 100

/-- Theorem: Given the conditions, the percentage spent on house rent is 14% --/
theorem percentage_on_rent_is_14 :
  ∀ (total_income : ℚ),
  total_income > 0 →
  total_income * (30 / 100) = 300 →
  percentage_on_rent total_income 300 98 = 14 := by
  sorry

#eval percentage_on_rent 1000 300 98

end NUMINAMATH_CALUDE_percentage_on_rent_is_14_l2065_206566


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l2065_206584

/-- The distance between Town A and Town B in miles -/
def distance_AB : ℝ := 80

/-- The speed difference between Cyclist Y and Cyclist X in mph -/
def speed_difference : ℝ := 6

/-- The distance from Town B where the cyclists meet after Cyclist Y turns back, in miles -/
def meeting_distance : ℝ := 20

/-- The speed of Cyclist X in mph -/
def speed_X : ℝ := 9

/-- The speed of Cyclist Y in mph -/
def speed_Y : ℝ := speed_X + speed_difference

theorem cyclist_speed_proof :
  speed_X * ((distance_AB + meeting_distance) / speed_Y) = distance_AB - meeting_distance :=
sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l2065_206584


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l2065_206589

def arithmetic_sequence (a₁ a₂ a₃ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

theorem tenth_term_of_sequence (h₁ : arithmetic_sequence (1/2) (5/6) (7/6) 2 = 5/6) 
                               (h₂ : arithmetic_sequence (1/2) (5/6) (7/6) 3 = 7/6) :
  arithmetic_sequence (1/2) (5/6) (7/6) 10 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l2065_206589


namespace NUMINAMATH_CALUDE_evaluate_expression_l2065_206592

theorem evaluate_expression : 5^2 - 5 + (6^2 - 6) - (7^2 - 7) + (8^2 - 8) = 64 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2065_206592


namespace NUMINAMATH_CALUDE_balloon_count_l2065_206569

/-- Calculates the total number of balloons given the number of gold, silver, and black balloons -/
def total_balloons (gold : ℕ) (silver : ℕ) (black : ℕ) : ℕ :=
  gold + silver + black

/-- Proves that the total number of balloons is 573 given the specified conditions -/
theorem balloon_count : 
  let gold : ℕ := 141
  let silver : ℕ := 2 * gold
  let black : ℕ := 150
  total_balloons gold silver black = 573 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l2065_206569


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2065_206568

def num_questions : ℕ := 7
def num_positive : ℕ := 3
def prob_positive : ℚ := 3/7

theorem magic_8_ball_probability :
  (Nat.choose num_questions num_positive : ℚ) *
  (prob_positive ^ num_positive) *
  ((1 - prob_positive) ^ (num_questions - num_positive)) =
  242112/823543 := by sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2065_206568


namespace NUMINAMATH_CALUDE_modulus_of_z_squared_l2065_206575

theorem modulus_of_z_squared (i : ℂ) (h : i^2 = -1) : 
  let z := (2 - i)^2
  Complex.abs z = 5 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_z_squared_l2065_206575


namespace NUMINAMATH_CALUDE_binary_51_l2065_206578

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Convert a list of booleans to a natural number, interpreting it as binary -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem binary_51 :
  toBinary 51 = [true, true, false, false, true, true] :=
by sorry

end NUMINAMATH_CALUDE_binary_51_l2065_206578


namespace NUMINAMATH_CALUDE_cube_order_l2065_206511

theorem cube_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_order_l2065_206511


namespace NUMINAMATH_CALUDE_meatballs_stolen_l2065_206577

/-- The number of meatballs Hayley initially had -/
def initial_meatballs : ℕ := 25

/-- The number of meatballs Hayley has now -/
def current_meatballs : ℕ := 11

/-- The number of meatballs Kirsten stole -/
def stolen_meatballs : ℕ := initial_meatballs - current_meatballs

theorem meatballs_stolen : stolen_meatballs = 14 := by
  sorry

end NUMINAMATH_CALUDE_meatballs_stolen_l2065_206577


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2065_206572

theorem chess_tournament_participants : ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2065_206572


namespace NUMINAMATH_CALUDE_sum_of_six_consecutive_odd_iff_l2065_206521

/-- Predicate to check if an integer is odd -/
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

/-- Predicate to check if an integer can be written as the sum of six consecutive odd integers -/
def IsSumOfSixConsecutiveOdd (S : ℤ) : Prop :=
  IsOdd ((S - 30) / 6)

theorem sum_of_six_consecutive_odd_iff (S : ℤ) :
  IsSumOfSixConsecutiveOdd S ↔
  ∃ n : ℤ, S = n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) ∧ IsOdd n :=
sorry

end NUMINAMATH_CALUDE_sum_of_six_consecutive_odd_iff_l2065_206521


namespace NUMINAMATH_CALUDE_tax_reduction_percentage_l2065_206540

/-- Proves that if a tax rate is reduced by X%, consumption increases by 15%,
    and revenue decreases by 3.4%, then X = 16%. -/
theorem tax_reduction_percentage
  (T : ℝ)  -- Original tax rate (in percentage)
  (X : ℝ)  -- Percentage by which tax is reduced
  (h1 : T > 0)  -- Assumption that original tax rate is positive
  (h2 : X > 0)  -- Assumption that tax reduction is positive
  (h3 : X < T)  -- Assumption that tax reduction is less than original tax
  : ((T - X) / 100 * 115 = T / 100 * 96.6) → X = 16 :=
by sorry

end NUMINAMATH_CALUDE_tax_reduction_percentage_l2065_206540


namespace NUMINAMATH_CALUDE_multiple_valid_scenarios_exist_l2065_206563

/-- Represents the ticket sales scenario for the Red Rose Theatre -/
structure TicketSales where
  total_tickets : ℕ
  total_sales : ℚ
  tickets_at_price1 : ℕ
  price1 : ℚ
  price2 : ℚ

/-- Checks if the given ticket sales scenario is valid -/
def is_valid_scenario (s : TicketSales) : Prop :=
  s.total_tickets = s.tickets_at_price1 + (s.total_tickets - s.tickets_at_price1) ∧
  s.total_sales = s.tickets_at_price1 * s.price1 + (s.total_tickets - s.tickets_at_price1) * s.price2

/-- States that multiple valid scenarios can exist for the same input data -/
theorem multiple_valid_scenarios_exist (total_tickets : ℕ) (total_sales : ℚ) (tickets_at_price1 : ℕ) :
  ∃ (s1 s2 : TicketSales),
    s1.total_tickets = total_tickets ∧
    s1.total_sales = total_sales ∧
    s1.tickets_at_price1 = tickets_at_price1 ∧
    s2.total_tickets = total_tickets ∧
    s2.total_sales = total_sales ∧
    s2.tickets_at_price1 = tickets_at_price1 ∧
    is_valid_scenario s1 ∧
    is_valid_scenario s2 ∧
    s1.price1 ≠ s2.price1 :=
  sorry

#check multiple_valid_scenarios_exist 380 1972.5 205

end NUMINAMATH_CALUDE_multiple_valid_scenarios_exist_l2065_206563


namespace NUMINAMATH_CALUDE_preferred_groups_2000_l2065_206550

/-- The number of non-empty subsets of {1, 2, ..., n} whose sum is divisible by 5 -/
def preferred_groups (n : ℕ) : ℕ := sorry

/-- The formula for the number of preferred groups when n = 2000 -/
def preferred_groups_formula : ℕ := 
  2^400 * ((1 / 5) * (2^1600 - 1) + 1) - 1

theorem preferred_groups_2000 : 
  preferred_groups 2000 = preferred_groups_formula := by sorry

end NUMINAMATH_CALUDE_preferred_groups_2000_l2065_206550


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2065_206567

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to seat 8 people in a row with restrictions. -/
def seatingArrangements : ℕ :=
  let totalArrangements := factorial 8
  let wilmaAndPaulTogether := factorial 7 * factorial 2
  let adamAndEveTogether := factorial 7 * factorial 2
  let bothPairsTogether := factorial 6 * factorial 2 * factorial 2
  totalArrangements - (wilmaAndPaulTogether + adamAndEveTogether - bothPairsTogether)

/-- Theorem stating that the number of seating arrangements is 23040. -/
theorem seating_arrangements_count :
  seatingArrangements = 23040 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2065_206567


namespace NUMINAMATH_CALUDE_f_one_less_than_f_two_necessary_not_sufficient_l2065_206535

def increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem f_one_less_than_f_two_necessary_not_sufficient :
  ∃ f : ℝ → ℝ, (increasing f → f 1 < f 2) ∧
  ¬(f 1 < f 2 → increasing f) :=
by sorry

end NUMINAMATH_CALUDE_f_one_less_than_f_two_necessary_not_sufficient_l2065_206535


namespace NUMINAMATH_CALUDE_square_root_of_nine_l2065_206545

-- Define the square root operation
def square_root (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem square_root_of_nine : square_root 9 = {-3, 3} := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l2065_206545


namespace NUMINAMATH_CALUDE_even_sum_from_even_expression_l2065_206529

theorem even_sum_from_even_expression (n m : ℤ) : 
  Even (n^2 + m^2 + n*m) → Even (n + m) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_from_even_expression_l2065_206529


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_l2065_206520

theorem mean_equality_implies_z (z : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + z) / 2 → z = 25 / 3 :=
by sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_l2065_206520


namespace NUMINAMATH_CALUDE_max_value_theorem_l2065_206556

theorem max_value_theorem (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) :
  ∃ (max : ℝ), (∀ (x' y' : ℝ), 4 * x'^2 + y'^2 + x' * y' = 1 → 2 * x' + y' ≤ max) ∧
                max = 2 * Real.sqrt 10 / 5 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2065_206556


namespace NUMINAMATH_CALUDE_max_value_implies_a_l2065_206553

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + a

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc (-1) 1, f a x = 2) →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_max_value_implies_a_l2065_206553


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l2065_206573

/-- The bowtie operation defined as a ⋈ b = a + √(b + √(b + √(b + ...))) -/
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem: If 3 ⋈ z = 9, then z = 30 -/
theorem bowtie_equation_solution :
  ∃ z : ℝ, bowtie 3 z = 9 ∧ z = 30 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l2065_206573


namespace NUMINAMATH_CALUDE_calculate_expression_l2065_206548

theorem calculate_expression : 3 * 3^3 + 4^7 / 4^5 = 97 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2065_206548


namespace NUMINAMATH_CALUDE_missing_number_is_four_l2065_206561

theorem missing_number_is_four : 
  ∃ x : ℤ, (x + 3) + (8 - 3 - 1) = 11 ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_missing_number_is_four_l2065_206561


namespace NUMINAMATH_CALUDE_factorization_x4_minus_9_factorization_quadratic_in_a_and_b_l2065_206562

-- Problem 1
theorem factorization_x4_minus_9 (x : ℝ) : 
  x^4 - 9 = (x^2 + 3) * (x + Real.sqrt 3) * (x - Real.sqrt 3) := by sorry

-- Problem 2
theorem factorization_quadratic_in_a_and_b (a b : ℝ) :
  -a^2*b + 2*a*b - b = -b*(a-1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_9_factorization_quadratic_in_a_and_b_l2065_206562


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l2065_206557

/-- Represents the daily sales and profit of a clothing store --/
structure ClothingStore where
  initialSales : ℕ
  initialProfit : ℝ
  priceReductionEffect : ℝ → ℕ
  targetProfit : ℝ

/-- Calculates the daily profit based on price reduction --/
def dailyProfit (store : ClothingStore) (priceReduction : ℝ) : ℝ :=
  let newSales := store.initialSales + store.priceReductionEffect priceReduction
  let newProfit := store.initialProfit - priceReduction
  newSales * newProfit

/-- Theorem stating that a $20 price reduction achieves the target profit --/
theorem price_reduction_achieves_target_profit (store : ClothingStore) 
  (h1 : store.initialSales = 20)
  (h2 : store.initialProfit = 40)
  (h3 : ∀ x, store.priceReductionEffect x = (8 / 4 : ℝ) * x)
  (h4 : store.targetProfit = 1200) :
  dailyProfit store 20 = store.targetProfit := by
  sorry


end NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l2065_206557


namespace NUMINAMATH_CALUDE_company_capital_growth_l2065_206587

/-- Calculates the final capital after n years given initial capital, growth rate, and yearly consumption --/
def finalCapital (initialCapital : ℝ) (growthRate : ℝ) (yearlyConsumption : ℝ) (years : ℕ) : ℝ :=
  match years with
  | 0 => initialCapital
  | n + 1 => (finalCapital initialCapital growthRate yearlyConsumption n * (1 + growthRate)) - yearlyConsumption

/-- The problem statement --/
theorem company_capital_growth (x : ℝ) : 
  finalCapital 1 0.5 x 3 = 2.9 ↔ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_company_capital_growth_l2065_206587


namespace NUMINAMATH_CALUDE_remainder_of_12345678910_mod_101_l2065_206513

theorem remainder_of_12345678910_mod_101 : 12345678910 % 101 = 31 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_12345678910_mod_101_l2065_206513


namespace NUMINAMATH_CALUDE_smallest_X_value_l2065_206534

/-- A function that checks if a natural number is composed only of 0s and 1s -/
def isComposedOf0sAnd1s (n : ℕ) : Prop := sorry

/-- The smallest positive integer composed of 0s and 1s that is divisible by 6 -/
def smallestValidT : ℕ := 1110

theorem smallest_X_value :
  ∀ T : ℕ,
  T > 0 →
  isComposedOf0sAnd1s T →
  T % 6 = 0 →
  T / 6 ≥ 185 :=
sorry

end NUMINAMATH_CALUDE_smallest_X_value_l2065_206534


namespace NUMINAMATH_CALUDE_bench_wood_length_l2065_206543

theorem bench_wood_length (num_long_pieces : ℕ) (long_piece_length : ℝ) (total_wood : ℝ) :
  num_long_pieces = 6 →
  long_piece_length = 4 →
  total_wood = 28 →
  total_wood - (num_long_pieces : ℝ) * long_piece_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_bench_wood_length_l2065_206543


namespace NUMINAMATH_CALUDE_cos_equality_proof_l2065_206516

theorem cos_equality_proof (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (1534 * π / 180) → n = 154 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_proof_l2065_206516


namespace NUMINAMATH_CALUDE_curve_properties_l2065_206594

/-- The curve y = ax³ + bx passing through point (2,2) with tangent slope 9 -/
def Curve (a b : ℝ) : Prop :=
  2 * a * 8 + 2 * b = 2 ∧ 3 * a * 4 + b = 9

/-- The function f(x) = ax³ + bx -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x

theorem curve_properties :
  ∀ a b : ℝ, Curve a b →
  (a * b = -3) ∧
  (∀ x : ℝ, -3/2 ≤ x ∧ x ≤ 3 → -2 ≤ f a b x ∧ f a b x ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l2065_206594


namespace NUMINAMATH_CALUDE_range_of_c_l2065_206585

theorem range_of_c (a b c : ℝ) :
  (∀ x : ℝ, (Real.sqrt (2 * x^2 + a * x + b) > x - c) ↔ (x ≤ 0 ∨ x > 1)) →
  c ∈ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_c_l2065_206585


namespace NUMINAMATH_CALUDE_sin_600_degrees_l2065_206533

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l2065_206533


namespace NUMINAMATH_CALUDE_amritsar_bombay_encounters_l2065_206527

/-- Represents a train journey from Amritsar to Bombay -/
structure TrainJourney where
  startTime : Nat  -- Start time in minutes after midnight
  duration : Nat   -- Duration of journey in minutes
  dailyDepartures : Nat  -- Number of trains departing each day

/-- Calculates the number of trains encountered during the journey -/
def encountersCount (journey : TrainJourney) : Nat :=
  sorry

/-- Theorem stating that a train journey with given conditions encounters 5 other trains -/
theorem amritsar_bombay_encounters :
  ∀ (journey : TrainJourney),
    journey.startTime = 9 * 60 →  -- 9 am start time
    journey.duration = 3 * 24 * 60 + 30 →  -- 3 days and 30 minutes duration
    journey.dailyDepartures = 1 →  -- One train departs each day
    encountersCount journey = 5 :=
  sorry

end NUMINAMATH_CALUDE_amritsar_bombay_encounters_l2065_206527


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2065_206536

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1 ∧ a > 0 ∧ b > 0

-- Define the focus
def focus (x y : ℝ) : Prop := x = 2 ∧ y = 0

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) :
  (∃ x y : ℝ, hyperbola a b x y ∧ focus x y) →
  eccentricity 2 →
  (∀ x y : ℝ, hyperbola a b x y ↔ x^2 - y^2/3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2065_206536


namespace NUMINAMATH_CALUDE_harry_sister_stamps_l2065_206565

theorem harry_sister_stamps (total : ℕ) (harry_ratio : ℕ) (sister_stamps : ℕ) : 
  total = 240 → 
  harry_ratio = 3 → 
  sister_stamps + harry_ratio * sister_stamps = total → 
  sister_stamps = 60 := by
sorry

end NUMINAMATH_CALUDE_harry_sister_stamps_l2065_206565


namespace NUMINAMATH_CALUDE_donation_distribution_l2065_206518

theorem donation_distribution (total : ℝ) (community_ratio : ℝ) (crisis_ratio : ℝ) (livelihood_ratio : ℝ)
  (h_total : total = 240)
  (h_community : community_ratio = 1/3)
  (h_crisis : crisis_ratio = 1/2)
  (h_livelihood : livelihood_ratio = 1/4) :
  let community := total * community_ratio
  let crisis := total * crisis_ratio
  let remaining := total - community - crisis
  let livelihood := remaining * livelihood_ratio
  total - community - crisis - livelihood = 30 := by
sorry

end NUMINAMATH_CALUDE_donation_distribution_l2065_206518


namespace NUMINAMATH_CALUDE_max_employees_l2065_206508

theorem max_employees (x : ℝ) (h : x > 4) : 
  ∃ (n : ℕ), n = ⌊2 * (x / (2 * x - 8))⌋ ∧ 
  ∀ (m : ℕ), (∀ (i j : ℕ), i < m → j < m → i ≠ j → 
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 8 ∧ t + x / 60 ≤ 8 ∧
    ∃ (ti tj : ℝ), 0 ≤ ti ∧ ti ≤ 8 ∧ 0 ≤ tj ∧ tj ≤ 8 ∧
    (t ≤ ti ∧ ti < t + x / 60) ∧ (t ≤ tj ∧ tj < t + x / 60)) →
  m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_employees_l2065_206508


namespace NUMINAMATH_CALUDE_total_score_is_219_l2065_206549

/-- Represents a player's score in a basketball game -/
structure PlayerScore where
  twoPointers : Nat
  threePointers : Nat
  freeThrows : Nat

/-- Calculates the total score for a player -/
def calculatePlayerScore (score : PlayerScore) : Nat :=
  2 * score.twoPointers + 3 * score.threePointers + score.freeThrows

/-- Theorem: The total points scored by all players is 219 -/
theorem total_score_is_219 
  (sam : PlayerScore)
  (alex : PlayerScore)
  (jake : PlayerScore)
  (lily : PlayerScore)
  (h_sam : sam = { twoPointers := 20, threePointers := 5, freeThrows := 10 })
  (h_alex : alex = { twoPointers := 15, threePointers := 6, freeThrows := 8 })
  (h_jake : jake = { twoPointers := 10, threePointers := 8, freeThrows := 5 })
  (h_lily : lily = { twoPointers := 12, threePointers := 3, freeThrows := 16 }) :
  calculatePlayerScore sam + calculatePlayerScore alex + 
  calculatePlayerScore jake + calculatePlayerScore lily = 219 := by
  sorry

end NUMINAMATH_CALUDE_total_score_is_219_l2065_206549


namespace NUMINAMATH_CALUDE_smallest_d_inequality_l2065_206509

theorem smallest_d_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  Real.sqrt (x * y * z) + (1/3) * |x^2 - y^2 + z^2| ≥ (x + y + z) / 3 ∧
  ∀ d : ℝ, d > 0 → d < 1/3 → ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
    Real.sqrt (a * b * c) + d * |a^2 - b^2 + c^2| < (a + b + c) / 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_inequality_l2065_206509


namespace NUMINAMATH_CALUDE_average_weight_abc_l2065_206571

theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 42 →
  (b + c) / 2 = 43 →
  b = 35 →
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_abc_l2065_206571


namespace NUMINAMATH_CALUDE_paint_time_per_room_l2065_206581

theorem paint_time_per_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_rooms = 12) 
  (h2 : painted_rooms = 5) 
  (h3 : remaining_time = 49) : 
  remaining_time / (total_rooms - painted_rooms) = 7 := by
  sorry

end NUMINAMATH_CALUDE_paint_time_per_room_l2065_206581


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l2065_206546

/-- Represents a normally distributed random variable -/
structure NormalRV (μ : ℝ) (σ : ℝ) where
  (μ_pos : μ > 0)
  (σ_pos : σ > 0)

/-- The probability of a random variable being in an interval -/
noncomputable def prob (X : Type) (a b : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry 
  (a σ : ℝ) (ξ : NormalRV a σ) 
  (h : prob (NormalRV a σ) 0 a = 0.3) : 
  prob (NormalRV a σ) 0 (2 * a) = 0.6 :=
sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l2065_206546


namespace NUMINAMATH_CALUDE_study_group_size_l2065_206500

theorem study_group_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n * (n - 1) = 90 ∧ 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_study_group_size_l2065_206500


namespace NUMINAMATH_CALUDE_two_fifths_in_three_fourths_l2065_206525

theorem two_fifths_in_three_fourths : (3 : ℚ) / 4 / ((2 : ℚ) / 5) = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_in_three_fourths_l2065_206525


namespace NUMINAMATH_CALUDE_unique_sequence_exists_l2065_206523

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 > 1 ∧
  ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2))

theorem unique_sequence_exists : ∃! a : ℕ → ℤ, is_valid_sequence a := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_exists_l2065_206523


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l2065_206504

theorem power_mod_thirteen : 7^137 % 13 = 11 := by sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l2065_206504


namespace NUMINAMATH_CALUDE_sum_set_size_bounds_l2065_206512

theorem sum_set_size_bounds (A : Finset ℕ) (S : Finset ℕ) : 
  A.card = 100 → 
  S = Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (A.product A) → 
  199 ≤ S.card ∧ S.card ≤ 5050 := by
  sorry

end NUMINAMATH_CALUDE_sum_set_size_bounds_l2065_206512


namespace NUMINAMATH_CALUDE_bus_passengers_l2065_206539

theorem bus_passengers (total : ℕ) 
  (h1 : 3 * total = 5 * (total / 5 * 3))  -- 3/5 of total are Dutch
  (h2 : (total / 5 * 3) / 2 * 2 = total / 5 * 3)  -- 1/2 of Dutch are American
  (h3 : ((total / 5 * 3) / 2) / 3 * 3 = (total / 5 * 3) / 2)  -- 1/3 of Dutch Americans got window seats
  (h4 : ((total / 5 * 3) / 2) / 3 = 9)  -- Number of Dutch Americans at windows is 9
  : total = 90 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l2065_206539


namespace NUMINAMATH_CALUDE_ellipse_equation_fixed_point_l2065_206501

/-- Ellipse C with center at origin, foci on x-axis, and eccentricity 1/2 -/
structure EllipseC where
  equation : ℝ → ℝ → Prop
  center_origin : equation 0 0
  foci_on_x_axis : ∀ x y, equation x y → y = 0 → x ≠ 0
  eccentricity : (∀ x y, equation x y → x^2 + y^2 = 1) → 
                 (∃ c, c > 0 ∧ ∀ x y, equation x y → x^2 + y^2 = (1 - c^2) * x^2 + y^2)

/-- Parabola with equation x = 1/4 * y^2 -/
def parabola (x y : ℝ) : Prop := x = 1/4 * y^2

/-- One vertex of ellipse C coincides with the focus of the parabola -/
axiom vertex_coincides_focus (C : EllipseC) : 
  ∃ x y, C.equation x y ∧ x^2 + y^2 = 1 ∧ x = 1 ∧ y = 0

/-- Theorem: Standard equation of ellipse C -/
theorem ellipse_equation (C : EllipseC) : 
  ∀ x y, C.equation x y ↔ x^2 + 4/3 * y^2 = 1 :=
sorry

/-- Chord AB of ellipse C passing through (1, 0) -/
def chord (C : EllipseC) (m : ℝ) (x y : ℝ) : Prop :=
  C.equation x y ∧ y = m * (x - 1)

/-- A' is the reflection of A over the x-axis -/
def reflect_over_x (x y : ℝ) : ℝ × ℝ := (x, -y)

/-- Theorem: Line A'B passes through (1, 0) -/
theorem fixed_point (C : EllipseC) (m : ℝ) (h : m ≠ 0) :
  ∃ x₁ y₁ x₂ y₂, 
    chord C m x₁ y₁ ∧ 
    chord C m x₂ y₂ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    let (x₁', y₁') := reflect_over_x x₁ y₁
    (y₁' - y₂) / (x₁' - x₂) = (0 - y₂) / (1 - x₂) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_fixed_point_l2065_206501


namespace NUMINAMATH_CALUDE_cubic_roots_difference_l2065_206552

-- Define the cubic polynomial
def cubic_poly (x : ℝ) : ℝ := 27 * x^3 - 81 * x^2 + 63 * x - 14

-- Define a predicate for roots in geometric progression
def roots_in_geometric_progression (r₁ r₂ r₃ : ℝ) : Prop :=
  ∃ (a r : ℝ), r₁ = a ∧ r₂ = a * r ∧ r₃ = a * r^2

-- Theorem statement
theorem cubic_roots_difference (r₁ r₂ r₃ : ℝ) :
  cubic_poly r₁ = 0 ∧ cubic_poly r₂ = 0 ∧ cubic_poly r₃ = 0 →
  roots_in_geometric_progression r₁ r₂ r₃ →
  (max r₁ (max r₂ r₃))^2 - (min r₁ (min r₂ r₃))^2 = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_difference_l2065_206552


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_l2065_206514

theorem roots_sum_reciprocal (a b : ℝ) : 
  (a^2 + 10*a + 5 = 0) → 
  (b^2 + 10*b + 5 = 0) → 
  (a/b + b/a = 18) := by sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_l2065_206514


namespace NUMINAMATH_CALUDE_road_repair_theorem_l2065_206596

/-- The number of persons in the first group -/
def first_group : ℕ := 42

/-- The number of days the first group works -/
def first_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_hours : ℕ := 5

/-- The number of days the second group works -/
def second_days : ℕ := 14

/-- The number of hours per day the second group works -/
def second_hours : ℕ := 6

/-- The total man-hours required to complete the work -/
def total_man_hours : ℕ := first_group * first_days * first_hours

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (second_days * second_hours)

theorem road_repair_theorem : second_group = 30 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_theorem_l2065_206596


namespace NUMINAMATH_CALUDE_last_score_must_be_86_l2065_206554

def scores : List ℕ := [73, 78, 84, 86, 97]

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

def average_is_integer (entered_scores : List ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → k ≤ entered_scores.length →
    is_integer ((entered_scores.take k).sum / k)

theorem last_score_must_be_86 :
  ∀ perm : List ℕ, perm.length = 5 →
  perm.toFinset = scores.toFinset →
  average_is_integer perm →
  perm.getLast? = some 86 :=
sorry

end NUMINAMATH_CALUDE_last_score_must_be_86_l2065_206554


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l2065_206588

theorem negation_of_existence_inequality (p : Prop) :
  (¬ (∃ x : ℝ, Real.exp x - x - 1 ≤ 0)) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l2065_206588


namespace NUMINAMATH_CALUDE_matrix_product_equality_l2065_206542

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; 1, 2]
def B : Matrix (Fin 2) (Fin 1) ℝ := !![4; -6]
def result : Matrix (Fin 2) (Fin 1) ℝ := !![26; -8]

theorem matrix_product_equality : A * B = result := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l2065_206542


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l2065_206559

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (onion_weight carrot_weight : ℕ),
    onion_weight + carrot_weight = (bag_weights.sum - t) ∧
    carrot_weight = 2 * onion_weight ∧
    ∃ (onion_bags carrot_bags : List ℕ),
      onion_bags ++ carrot_bags = bag_weights.filter (λ w => w ≠ t) ∧
      onion_bags.sum = onion_weight ∧
      carrot_bags.sum = carrot_weight

theorem turnip_bag_weights :
  ∀ t : ℕ, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 :=
by sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l2065_206559


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2065_206598

theorem complex_number_quadrant (a b : ℝ) (h : (1 : ℂ) + a * I = (b + I) * (1 + I)) :
  a > 0 ∧ b > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2065_206598


namespace NUMINAMATH_CALUDE_binomial_2024_1_l2065_206597

theorem binomial_2024_1 : Nat.choose 2024 1 = 2024 := by sorry

end NUMINAMATH_CALUDE_binomial_2024_1_l2065_206597


namespace NUMINAMATH_CALUDE_smallest_five_digit_cube_sum_l2065_206528

theorem smallest_five_digit_cube_sum (x : ℕ) : 
  x ≥ 10000 ∧ x < 100000 ∧ 
  (∃ k : ℕ, (343 * x) / 90 = k^3) ∧
  (∀ y : ℕ, y ≥ 10000 ∧ y < x → ¬(∃ k : ℕ, (343 * y) / 90 = k^3)) →
  x = 11250 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_cube_sum_l2065_206528


namespace NUMINAMATH_CALUDE_smallest_valid_sequence_length_l2065_206583

def is_valid_sequence (a : List Int) : Prop :=
  a.sum = 2005 ∧ a.prod = 2005

theorem smallest_valid_sequence_length :
  (∃ (n : Nat) (a : List Int), n > 1 ∧ a.length = n ∧ is_valid_sequence a) ∧
  (∀ (m : Nat) (b : List Int), m > 1 ∧ m < 5 ∧ b.length = m → ¬is_valid_sequence b) ∧
  (∃ (c : List Int), c.length = 5 ∧ is_valid_sequence c) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_sequence_length_l2065_206583


namespace NUMINAMATH_CALUDE_train_length_l2065_206524

/-- The length of a train given its speed and the time it takes to cross a bridge of known length. -/
theorem train_length (v : ℝ) (t : ℝ) (bridge_length : ℝ) (h1 : v = 36 * (1000 / 3600)) (h2 : t = 26.997840172786177) (h3 : bridge_length = 150) :
  v * t - bridge_length = 119.97840172786177 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2065_206524


namespace NUMINAMATH_CALUDE_taxi_driver_problem_l2065_206590

def distances : List Int := [8, -6, 3, -4, 8, -4, 4, -3]

def total_time : Rat := 4/3

theorem taxi_driver_problem (distances : List Int) (total_time : Rat) :
  (distances.sum = 6) ∧
  (((distances.map abs).sum : Rat) / total_time = 30) :=
by sorry

end NUMINAMATH_CALUDE_taxi_driver_problem_l2065_206590


namespace NUMINAMATH_CALUDE_money_division_l2065_206502

theorem money_division (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 3 * (ben / 5) →
  carlos = 9 * (ben / 5) →
  ben = 50 →
  total = 170 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l2065_206502


namespace NUMINAMATH_CALUDE_four_term_expression_l2065_206505

theorem four_term_expression (x : ℝ) : 
  ∃ (a b c d : ℝ), (x^4 - 3)^2 + (x^3 + 3*x)^2 = a*x^8 + b*x^6 + c*x^2 + d ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_four_term_expression_l2065_206505


namespace NUMINAMATH_CALUDE_soccer_team_uniform_numbers_l2065_206574

/-- A predicate to check if a number is a two-digit prime -/
def isTwoDigitPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

/-- The uniform numbers of Emily, Fiona, and Grace -/
structure UniformNumbers where
  emily : ℕ
  fiona : ℕ
  grace : ℕ

/-- The conditions of the soccer team uniform numbers problem -/
structure SoccerTeamConditions (u : UniformNumbers) : Prop where
  emily_prime : isTwoDigitPrime u.emily
  fiona_prime : isTwoDigitPrime u.fiona
  grace_prime : isTwoDigitPrime u.grace
  emily_fiona_sum : u.emily + u.fiona = 23
  emily_grace_sum : u.emily + u.grace = 31

theorem soccer_team_uniform_numbers (u : UniformNumbers) 
  (h : SoccerTeamConditions u) : u.grace = 19 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_uniform_numbers_l2065_206574


namespace NUMINAMATH_CALUDE_sum_of_differences_equals_6999993_l2065_206580

/-- Calculates the local value of a digit in a number based on its position --/
def localValue (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (10 ^ position)

/-- Calculates the difference between local value and face value for a digit --/
def valueDifference (digit : ℕ) (position : ℕ) : ℕ :=
  localValue digit position - digit

/-- The numeral we're working with --/
def numeral : ℕ := 657932657

/-- Positions of 7 in the numeral (0-indexed from right) --/
def sevenPositions : List ℕ := [0, 6]

/-- Sum of differences between local and face values for all 7s in the numeral --/
def sumOfDifferences : ℕ :=
  (sevenPositions.map (valueDifference 7)).sum

theorem sum_of_differences_equals_6999993 :
  sumOfDifferences = 6999993 := by sorry

end NUMINAMATH_CALUDE_sum_of_differences_equals_6999993_l2065_206580


namespace NUMINAMATH_CALUDE_power_function_value_l2065_206531

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x^α

-- Define the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f 4 = 1/2 → f (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l2065_206531


namespace NUMINAMATH_CALUDE_line_equation_k_value_l2065_206526

/-- Given a line passing through points (m, n) and (m + 2, n + 0.4), 
    with equation x = ky + 5, prove that k = 5 -/
theorem line_equation_k_value (m n : ℝ) : 
  let p : ℝ := 0.4
  let point1 : ℝ × ℝ := (m, n)
  let point2 : ℝ × ℝ := (m + 2, n + p)
  let k : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)
  ∀ x y : ℝ, x = k * y + 5 → k = 5 := by
sorry

end NUMINAMATH_CALUDE_line_equation_k_value_l2065_206526


namespace NUMINAMATH_CALUDE_lawn_mowing_total_l2065_206510

theorem lawn_mowing_total (spring_mows summer_mows : ℕ) 
  (h1 : spring_mows = 6) 
  (h2 : summer_mows = 5) : 
  spring_mows + summer_mows = 11 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_total_l2065_206510


namespace NUMINAMATH_CALUDE_integer_solution_range_l2065_206576

theorem integer_solution_range (b : ℝ) : 
  (∀ x : ℤ, |3 * (x : ℝ) - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → 
  (5 < b ∧ b < 7) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_range_l2065_206576


namespace NUMINAMATH_CALUDE_parabola_line_intersection_minimum_l2065_206551

/-- A parabola with equation y^2 = 16x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- A line passing through a point --/
structure Line where
  passingPoint : ℝ × ℝ

/-- Intersection points of a line and a parabola --/
structure Intersection where
  M : ℝ × ℝ
  N : ℝ × ℝ

/-- Distance between two points --/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The main theorem --/
theorem parabola_line_intersection_minimum (p : Parabola) (l : Line) (i : Intersection) :
  p.equation = (fun x y => y^2 = 16*x) →
  p.focus = (4, 0) →
  l.passingPoint = p.focus →
  (∃ (x y : ℝ), p.equation x y ∧ (x, y) = i.M) →
  (∃ (x y : ℝ), p.equation x y ∧ (x, y) = i.N) →
  (∀ NF MF : ℝ, NF = distance i.N p.focus → MF = distance i.M p.focus →
    NF / 9 - 4 / MF ≥ 1 / 3) ∧
  (∃ NF MF : ℝ, NF = distance i.N p.focus ∧ MF = distance i.M p.focus ∧
    NF / 9 - 4 / MF = 1 / 3) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_minimum_l2065_206551


namespace NUMINAMATH_CALUDE_fibonacci_factorial_sum_last_two_digits_l2065_206547

def fibonacci_factorial_series := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem fibonacci_factorial_sum_last_two_digits : 
  (fibonacci_factorial_series.map (λ x => last_two_digits (factorial x))).sum = 50 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_factorial_sum_last_two_digits_l2065_206547


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2065_206595

theorem complex_equation_solution (a b : ℝ) : 
  (a - 2 * Complex.I = (b + Complex.I) * Complex.I) → (a = -1 ∧ b = -2) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2065_206595


namespace NUMINAMATH_CALUDE_sequence_ratio_l2065_206599

/-- Given a sequence {a_n} where the sum of the first n terms S_n satisfies S_n = 2a_n - 2,
    prove that the ratio a_8 / a_6 = 4. -/
theorem sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = 2 * a n - 2) : 
  a 8 / a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_l2065_206599


namespace NUMINAMATH_CALUDE_worker_a_completion_time_l2065_206517

/-- The number of days it takes for two workers to complete a job together -/
def combined_days : ℝ := 18

/-- The ratio of worker a's speed to worker b's speed -/
def speed_ratio : ℝ := 1.5

/-- The number of days it takes for worker a to complete the job alone -/
def days_a : ℝ := 30

theorem worker_a_completion_time : 
  1 / combined_days = 1 / days_a + 1 / (speed_ratio * days_a) :=
by sorry

end NUMINAMATH_CALUDE_worker_a_completion_time_l2065_206517


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l2065_206503

/-- Calculates the average speed of a car trip given specific conditions -/
theorem car_trip_average_speed
  (total_time : ℝ)
  (initial_time : ℝ)
  (initial_speed : ℝ)
  (remaining_speed : ℝ)
  (h_total_time : total_time = 6)
  (h_initial_time : initial_time = 4)
  (h_initial_speed : initial_speed = 55)
  (h_remaining_speed : remaining_speed = 70) :
  (initial_speed * initial_time + remaining_speed * (total_time - initial_time)) / total_time = 60 :=
by sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l2065_206503


namespace NUMINAMATH_CALUDE_parabola_vertex_l2065_206519

/-- The parabola defined by the equation y = x^2 + 2x + 5 -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 5

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := -1

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y : ℝ := 4

/-- Theorem stating that (vertex_x, vertex_y) is the vertex of the parabola -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≥ parabola vertex_x) ∧
  parabola vertex_x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2065_206519


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2065_206558

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 ∧ 
    (∃ (y : ℕ), 5 * x = y^2) ∧ 
    (∃ (z : ℕ), 3 * x = z^3) → 
    x ≥ n) ∧
  n = 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2065_206558


namespace NUMINAMATH_CALUDE_calculate_expression_l2065_206586

theorem calculate_expression (a b : ℝ) (hb : b ≠ 0) :
  4 * a * (3 * a^2 * b) / (2 * a * b) = 6 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2065_206586
