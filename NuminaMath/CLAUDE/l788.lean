import Mathlib

namespace NUMINAMATH_CALUDE_complex_norm_product_l788_78881

theorem complex_norm_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_norm_product_l788_78881


namespace NUMINAMATH_CALUDE_prob_two_red_crayons_l788_78877

/-- The probability of selecting 2 red crayons from a jar containing 6 crayons (3 red, 2 blue, 1 green) -/
theorem prob_two_red_crayons (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) :
  total = 6 →
  red = 3 →
  blue = 2 →
  green = 1 →
  (Nat.choose red 2 : ℚ) / (Nat.choose total 2) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_crayons_l788_78877


namespace NUMINAMATH_CALUDE_xy_value_l788_78873

theorem xy_value (x y : ℝ) (h : x * (x + 2*y) = x^2 + 10) : x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l788_78873


namespace NUMINAMATH_CALUDE_expand_and_simplify_l788_78816

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l788_78816


namespace NUMINAMATH_CALUDE_anna_age_proof_l788_78888

/-- Anna's current age -/
def anna_age : ℕ := 54

/-- Clara's current age -/
def clara_age : ℕ := 80

/-- Years in the past -/
def years_ago : ℕ := 41

theorem anna_age_proof :
  anna_age = 54 ∧
  clara_age = 80 ∧
  clara_age - years_ago = 3 * (anna_age - years_ago) :=
sorry

end NUMINAMATH_CALUDE_anna_age_proof_l788_78888


namespace NUMINAMATH_CALUDE_soup_distribution_l788_78898

-- Define the total amount of soup
def total_soup : ℚ := 1

-- Define the number of grandchildren
def num_children : ℕ := 5

-- Define the amount taken by Ângela and Daniela
def angela_daniela_portion : ℚ := 2 / 5

-- Define Laura's division
def laura_division : ℕ := 5

-- Define João's division
def joao_division : ℕ := 4

-- Define the container size in ml
def container_size : ℕ := 100

-- Theorem statement
theorem soup_distribution (
  laura_portion : ℚ)
  (toni_portion : ℚ)
  (min_soup_amount : ℚ) :
  laura_portion = 3 / 25 ∧
  toni_portion = 9 / 25 ∧
  min_soup_amount = 5 / 2 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_soup_distribution_l788_78898


namespace NUMINAMATH_CALUDE_remaining_three_digit_numbers_l788_78886

/-- The count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers where the first and last digits are the same
    and the middle digit is different -/
def excluded_numbers : ℕ := 81

/-- Theorem: The count of three-digit numbers excluding those where the first and last digits
    are the same and the middle digit is different is equal to 819 -/
theorem remaining_three_digit_numbers :
  total_three_digit_numbers - excluded_numbers = 819 := by
  sorry

end NUMINAMATH_CALUDE_remaining_three_digit_numbers_l788_78886


namespace NUMINAMATH_CALUDE_unique_solution_system_l788_78882

theorem unique_solution_system (m : ℝ) : ∃! (x y : ℝ), 
  ((m + 1) * x - y - 3 * m = 0) ∧ (4 * x + (m - 1) * y + 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l788_78882


namespace NUMINAMATH_CALUDE_choose_one_friend_from_ten_l788_78802

def number_of_friends : ℕ := 10
def friends_to_choose : ℕ := 1

theorem choose_one_friend_from_ten :
  Nat.choose number_of_friends friends_to_choose = 10 := by
  sorry

end NUMINAMATH_CALUDE_choose_one_friend_from_ten_l788_78802


namespace NUMINAMATH_CALUDE_max_f_value_1997_l788_78808

def f : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => f (n / 2) + (n + 2) - 2 * (n / 2)

theorem max_f_value_1997 :
  (∃ (n : ℕ), n ≤ 1997 ∧ f n = 10) ∧
  (∀ (n : ℕ), n ≤ 1997 → f n ≤ 10) :=
sorry

end NUMINAMATH_CALUDE_max_f_value_1997_l788_78808


namespace NUMINAMATH_CALUDE_coin_pile_impossibility_l788_78835

/-- Represents a pile of coins -/
structure CoinPile :=
  (coins : ℕ)

/-- Represents the state of all coin piles -/
structure CoinState :=
  (piles : List CoinPile)

/-- Allowed operations on coin piles -/
inductive CoinOperation
  | Combine : CoinPile → CoinPile → CoinOperation
  | Split : CoinPile → CoinOperation

/-- Applies a coin operation to a coin state -/
def applyOperation (state : CoinState) (op : CoinOperation) : CoinState :=
  sorry

/-- Checks if a coin state matches the target configuration -/
def isTargetState (state : CoinState) : Prop :=
  ∃ (p1 p2 p3 : CoinPile),
    state.piles = [p1, p2, p3] ∧
    p1.coins = 52 ∧ p2.coins = 48 ∧ p3.coins = 5

/-- The main theorem stating the impossibility of reaching the target state -/
theorem coin_pile_impossibility :
  ∀ (initial : CoinState) (ops : List CoinOperation),
    initial.piles = [CoinPile.mk 51, CoinPile.mk 49, CoinPile.mk 5] →
    ¬(isTargetState (ops.foldl applyOperation initial)) :=
  sorry

end NUMINAMATH_CALUDE_coin_pile_impossibility_l788_78835


namespace NUMINAMATH_CALUDE_geometric_sum_7_terms_l788_78838

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_7_terms :
  let a : ℚ := 1/2
  let r : ℚ := -1/3
  let n : ℕ := 7
  geometric_sum a r n = 547/1458 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_7_terms_l788_78838


namespace NUMINAMATH_CALUDE_olivia_weekly_earnings_l788_78801

/-- Calculates the total earnings for a week given an hourly rate and hours worked on specific days -/
def weeklyEarnings (hourlyRate : ℕ) (mondayHours wednesdayHours fridayHours : ℕ) : ℕ :=
  hourlyRate * (mondayHours + wednesdayHours + fridayHours)

/-- Proves that Olivia's earnings for the week equal $117 -/
theorem olivia_weekly_earnings :
  weeklyEarnings 9 4 3 6 = 117 := by
  sorry

end NUMINAMATH_CALUDE_olivia_weekly_earnings_l788_78801


namespace NUMINAMATH_CALUDE_product_of_first_1001_primes_factors_product_of_first_1001_primes_not_factor_l788_78850

def first_n_primes (n : ℕ) : List ℕ :=
  sorry

def product_of_list (l : List ℕ) : ℕ :=
  sorry

def is_factor (a b : ℕ) : Prop :=
  ∃ k : ℕ, b = a * k

theorem product_of_first_1001_primes_factors (n : ℕ) :
  let P := product_of_list (first_n_primes 1001)
  (n = 2002 ∨ n = 3003 ∨ n = 5005 ∨ n = 6006) →
  is_factor n P :=
sorry

theorem product_of_first_1001_primes_not_factor :
  let P := product_of_list (first_n_primes 1001)
  ¬ is_factor 7007 P :=
sorry

end NUMINAMATH_CALUDE_product_of_first_1001_primes_factors_product_of_first_1001_primes_not_factor_l788_78850


namespace NUMINAMATH_CALUDE_erica_safari_elephants_l788_78819

/-- The number of elephants Erica saw on her safari --/
def elephants_seen (total_animals : ℕ) (lions_saturday : ℕ) (animals_sunday_monday : ℕ) : ℕ :=
  total_animals - lions_saturday - animals_sunday_monday

/-- Theorem stating the number of elephants Erica saw on Saturday --/
theorem erica_safari_elephants :
  elephants_seen 20 3 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_erica_safari_elephants_l788_78819


namespace NUMINAMATH_CALUDE_cara_age_l788_78812

/-- Given the ages of three generations in a family, prove Cara's age. -/
theorem cara_age (cara_age mom_age grandma_age : ℕ) 
  (h1 : cara_age + 20 = mom_age)
  (h2 : mom_age + 15 = grandma_age)
  (h3 : grandma_age = 75) :
  cara_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_cara_age_l788_78812


namespace NUMINAMATH_CALUDE_nancy_carrots_l788_78884

/-- Calculates the total number of carrots Nancy has -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  initial - thrown_out + new_picked

/-- Proves that Nancy's total carrots is 31 given the problem conditions -/
theorem nancy_carrots : total_carrots 12 2 21 = 31 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrots_l788_78884


namespace NUMINAMATH_CALUDE_caravan_camel_count_l788_78894

/-- Represents the number of camels in the caravan -/
def num_camels : ℕ := 6

/-- Represents the number of hens in the caravan -/
def num_hens : ℕ := 60

/-- Represents the number of goats in the caravan -/
def num_goats : ℕ := 35

/-- Represents the number of keepers in the caravan -/
def num_keepers : ℕ := 10

/-- Represents the difference between the total number of feet and heads -/
def feet_head_difference : ℕ := 193

theorem caravan_camel_count : 
  (2 * num_hens + 4 * num_goats + 4 * num_camels + 2 * num_keepers) = 
  (num_hens + num_goats + num_camels + num_keepers + feet_head_difference) := by
  sorry

end NUMINAMATH_CALUDE_caravan_camel_count_l788_78894


namespace NUMINAMATH_CALUDE_pants_price_problem_l788_78806

theorem pants_price_problem (total_cost belt_price pants_price : ℝ) : 
  total_cost = 70.93 →
  pants_price = belt_price - 2.93 →
  total_cost = belt_price + pants_price →
  pants_price = 34.00 := by
  sorry

end NUMINAMATH_CALUDE_pants_price_problem_l788_78806


namespace NUMINAMATH_CALUDE_problem_statement_l788_78863

theorem problem_statement (a b : ℝ) : 
  (a + b + 1 = -2) → (a + b - 1) * (1 - a - b) = -16 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l788_78863


namespace NUMINAMATH_CALUDE_cost_of_type_B_theorem_l788_78851

/-- The cost of purchasing type B books given the total number of books,
    the number of type A books purchased, and the unit price of type B books. -/
def cost_of_type_B (total_books : ℕ) (type_A_books : ℕ) (unit_price_B : ℕ) : ℕ :=
  unit_price_B * (total_books - type_A_books)

/-- Theorem stating that the cost of purchasing type B books
    is equal to 8(100-x) given the specified conditions. -/
theorem cost_of_type_B_theorem (x : ℕ) (h : x ≤ 100) :
  cost_of_type_B 100 x 8 = 8 * (100 - x) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_type_B_theorem_l788_78851


namespace NUMINAMATH_CALUDE_stating_repeating_decimal_equals_fraction_l788_78891

/-- Represents a repeating decimal where the fractional part is 0.325325325... -/
def repeating_decimal : ℚ := 3/10 + 25/990

/-- The fraction 161/495 in its lowest terms -/
def target_fraction : ℚ := 161/495

/-- 
Theorem stating that the repeating decimal 0.3̅25̅ is equal to the fraction 161/495
-/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_stating_repeating_decimal_equals_fraction_l788_78891


namespace NUMINAMATH_CALUDE_quadratic_coefficient_bound_l788_78820

theorem quadratic_coefficient_bound (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc (-1) 1) →
  |f 1| ≤ 1 →
  |f (1/2)| ≤ 1 →
  |f 0| ≤ 1 →
  |a| + |b| + |c| ≤ 17 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_bound_l788_78820


namespace NUMINAMATH_CALUDE_five_mondays_in_november_l788_78848

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Given that a month starts on a certain day, 
    returns the number of occurrences of a specific day in that month -/
def countDayOccurrences (month : Month) (day : DayOfWeek) : Nat :=
  sorry

/-- October of year M -/
def october : Month :=
  { days := 31, firstDay := sorry }

/-- November of year M -/
def november : Month :=
  { days := 30, firstDay := sorry }

theorem five_mondays_in_november 
  (h : countDayOccurrences october DayOfWeek.Friday = 5) :
  countDayOccurrences november DayOfWeek.Monday = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_mondays_in_november_l788_78848


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l788_78807

theorem merchant_profit_percentage (C S : ℝ) (h : 24 * C = 16 * S) : 
  (S - C) / C * 100 = 50 :=
sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l788_78807


namespace NUMINAMATH_CALUDE_inequality_proof_l788_78896

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π/2) : 
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l788_78896


namespace NUMINAMATH_CALUDE_arcsin_plus_arccos_eq_pi_half_l788_78827

theorem arcsin_plus_arccos_eq_pi_half (x : ℝ) :
  Real.arcsin x + Real.arccos (1 - x) = π / 2 → x = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_plus_arccos_eq_pi_half_l788_78827


namespace NUMINAMATH_CALUDE_female_percentage_l788_78813

/-- Represents a classroom with double desks -/
structure Classroom where
  male_students : ℕ
  female_students : ℕ
  male_with_male : ℕ
  female_with_female : ℕ

/-- All seats are occupied and the percentages of same-gender pairings are as given -/
def valid_classroom (c : Classroom) : Prop :=
  c.male_with_male = (6 * c.male_students) / 10 ∧
  c.female_with_female = (2 * c.female_students) / 10 ∧
  c.male_students - c.male_with_male = c.female_students - c.female_with_female

theorem female_percentage (c : Classroom) (h : valid_classroom c) :
  (c.female_students : ℚ) / (c.male_students + c.female_students) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_female_percentage_l788_78813


namespace NUMINAMATH_CALUDE_reimbursement_difference_l788_78845

/-- The problem of reimbursement in a group activity --/
theorem reimbursement_difference (tom emma harry : ℝ) : 
  tom = 95 →
  emma = 140 →
  harry = 165 →
  let total := tom + emma + harry
  let share := total / 3
  let t := share - tom
  let e := share - emma
  e - t = -45 := by
  sorry

end NUMINAMATH_CALUDE_reimbursement_difference_l788_78845


namespace NUMINAMATH_CALUDE_percentage_no_conditions_is_13_33_l788_78864

/-- Represents the survey results of teachers' health conditions -/
structure TeacherSurvey where
  total : ℕ
  highBP : ℕ
  heartTrouble : ℕ
  diabetes : ℕ
  highBP_heartTrouble : ℕ
  heartTrouble_diabetes : ℕ
  highBP_diabetes : ℕ
  all_three : ℕ

/-- Calculates the percentage of teachers with no health conditions -/
def percentageWithNoConditions (survey : TeacherSurvey) : ℚ :=
  let withConditions := 
    survey.highBP + survey.heartTrouble + survey.diabetes -
    survey.highBP_heartTrouble - survey.heartTrouble_diabetes - survey.highBP_diabetes +
    survey.all_three
  let withoutConditions := survey.total - withConditions
  (withoutConditions : ℚ) / survey.total * 100

/-- The main theorem stating that the percentage of teachers with no health conditions is 13.33% -/
theorem percentage_no_conditions_is_13_33 (survey : TeacherSurvey) 
  (h1 : survey.total = 150)
  (h2 : survey.highBP = 80)
  (h3 : survey.heartTrouble = 60)
  (h4 : survey.diabetes = 30)
  (h5 : survey.highBP_heartTrouble = 20)
  (h6 : survey.heartTrouble_diabetes = 10)
  (h7 : survey.highBP_diabetes = 15)
  (h8 : survey.all_three = 5) :
  percentageWithNoConditions survey = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_no_conditions_is_13_33_l788_78864


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l788_78887

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, 
    (Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4)) → 
    t = 37/10 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l788_78887


namespace NUMINAMATH_CALUDE_income_comparison_l788_78839

/-- Given that Mart's income is 30% more than Tim's income and 78% of Juan's income,
    prove that Tim's income is 40% less than Juan's income. -/
theorem income_comparison (tim mart juan : ℝ) 
  (h1 : mart = 1.3 * tim) 
  (h2 : mart = 0.78 * juan) : 
  tim = 0.6 * juan := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l788_78839


namespace NUMINAMATH_CALUDE_eating_time_theorem_l788_78859

/-- Represents the eating rate of a character in jars per minute -/
structure EatingRate :=
  (condensed_milk : ℚ)
  (honey : ℚ)

/-- Calculates the time taken to eat a certain amount of food given the eating rate -/
def time_to_eat (rate : EatingRate) (condensed_milk : ℚ) (honey : ℚ) : ℚ :=
  (condensed_milk / rate.condensed_milk) + (honey / rate.honey)

/-- Calculates the combined eating rate of two characters -/
def combined_rate (rate1 rate2 : EatingRate) : EatingRate :=
  { condensed_milk := rate1.condensed_milk + rate2.condensed_milk,
    honey := rate1.honey + rate2.honey }

theorem eating_time_theorem (pooh_rate piglet_rate : EatingRate) : 
  (time_to_eat pooh_rate 3 1 = 25) →
  (time_to_eat piglet_rate 3 1 = 55) →
  (time_to_eat pooh_rate 1 3 = 35) →
  (time_to_eat piglet_rate 1 3 = 85) →
  (time_to_eat (combined_rate pooh_rate piglet_rate) 6 0 = 20) := by
  sorry

end NUMINAMATH_CALUDE_eating_time_theorem_l788_78859


namespace NUMINAMATH_CALUDE_triangle_angle_c_l788_78815

theorem triangle_angle_c (A B C : Real) (perimeter area : Real) :
  perimeter = Real.sqrt 2 + 1 →
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →
  area = (1 / 6) * Real.sin C →
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l788_78815


namespace NUMINAMATH_CALUDE_cube_sum_symmetric_polynomials_l788_78868

theorem cube_sum_symmetric_polynomials (x y z : ℝ) :
  let σ₁ : ℝ := x + y + z
  let σ₂ : ℝ := x * y + y * z + z * x
  let σ₃ : ℝ := x * y * z
  x^3 + y^3 + z^3 = σ₁^3 - 3 * σ₁ * σ₂ + 3 * σ₃ := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_symmetric_polynomials_l788_78868


namespace NUMINAMATH_CALUDE_two_integers_problem_l788_78879

theorem two_integers_problem :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x - y = 8 ∧ x * y = 180 ∧ x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_problem_l788_78879


namespace NUMINAMATH_CALUDE_min_lcm_ac_is_90_l788_78843

def min_lcm_ac (a b c : ℕ) : Prop :=
  (Nat.lcm a b = 20) ∧ (Nat.lcm b c = 18) → Nat.lcm a c ≥ 90

theorem min_lcm_ac_is_90 :
  ∃ (a b c : ℕ), min_lcm_ac a b c ∧ Nat.lcm a c = 90 :=
sorry

end NUMINAMATH_CALUDE_min_lcm_ac_is_90_l788_78843


namespace NUMINAMATH_CALUDE_sock_pairs_count_l788_78836

/-- Given a number of sock pairs, calculate the number of ways to select two socks
    from different pairs. -/
def nonMatchingSelections (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The problem statement -/
theorem sock_pairs_count : ∃ (n : ℕ), n > 0 ∧ nonMatchingSelections n = 112 :=
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l788_78836


namespace NUMINAMATH_CALUDE_basketball_game_second_half_score_l788_78865

/-- Represents the points scored by a team in each quarter -/
structure QuarterlyPoints where
  q1 : ℝ
  q2 : ℝ
  q3 : ℝ
  q4 : ℝ

/-- The game between Raiders and Wildcats -/
structure BasketballGame where
  raiders : QuarterlyPoints
  wildcats : QuarterlyPoints

def BasketballGame.total_score (game : BasketballGame) : ℝ :=
  game.raiders.q1 + game.raiders.q2 + game.raiders.q3 + game.raiders.q4 +
  game.wildcats.q1 + game.wildcats.q2 + game.wildcats.q3 + game.wildcats.q4

def BasketballGame.second_half_score (game : BasketballGame) : ℝ :=
  game.raiders.q3 + game.raiders.q4 + game.wildcats.q3 + game.wildcats.q4

theorem basketball_game_second_half_score
  (a b d r : ℝ)
  (hr : r ≥ 1)
  (game : BasketballGame)
  (h1 : game.raiders = ⟨a, a*r, a*r^2, a*r^3⟩)
  (h2 : game.wildcats = ⟨b, b+d, b+2*d, b+3*d⟩)
  (h3 : game.raiders.q1 = game.wildcats.q1)
  (h4 : game.total_score = 2 * game.raiders.q1 + game.raiders.q2 + game.raiders.q3 + game.raiders.q4 +
                           game.wildcats.q2 + game.wildcats.q3 + game.wildcats.q4)
  (h5 : game.total_score = 2 * (4*b + 6*d + 3))
  (h6 : ∀ q, q ∈ [game.raiders.q1, game.raiders.q2, game.raiders.q3, game.raiders.q4,
                  game.wildcats.q1, game.wildcats.q2, game.wildcats.q3, game.wildcats.q4] → q ≤ 100) :
  game.second_half_score = 112 :=
sorry

end NUMINAMATH_CALUDE_basketball_game_second_half_score_l788_78865


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l788_78824

theorem triangle_side_lengths (x : ℕ+) : 
  (9 + 12 > x^2 ∧ x^2 + 9 > 12 ∧ x^2 + 12 > 9) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l788_78824


namespace NUMINAMATH_CALUDE_x_power_n_plus_inverse_l788_78826

theorem x_power_n_plus_inverse (θ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : x + 1 / x = -2 * Real.sin θ) (h4 : n > 0) :
  x^n + 1 / x^n = -2 * Real.sin (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_x_power_n_plus_inverse_l788_78826


namespace NUMINAMATH_CALUDE_representatives_count_l788_78817

/-- The number of ways to select 3 representatives from 3 boys and 2 girls, 
    such that at least one girl is included. -/
def select_representatives (num_boys num_girls num_representatives : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of ways to select 3 representatives from 3 boys and 2 girls, 
    such that at least one girl is included, is equal to 9. -/
theorem representatives_count : select_representatives 3 2 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_representatives_count_l788_78817


namespace NUMINAMATH_CALUDE_least_value_of_quadratic_l788_78828

theorem least_value_of_quadratic (y : ℝ) : 
  (5 * y^2 + 7 * y + 3 = 5) → y ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_least_value_of_quadratic_l788_78828


namespace NUMINAMATH_CALUDE_correct_prediction_probability_l788_78890

def n_monday : ℕ := 5
def n_tuesday : ℕ := 6
def n_total : ℕ := n_monday + n_tuesday
def n_correct : ℕ := 7
def n_correct_monday : ℕ := 3

theorem correct_prediction_probability :
  let p : ℝ := 1 / 2
  (Nat.choose n_monday n_correct_monday * p ^ n_monday * (1 - p) ^ (n_monday - n_correct_monday)) *
  (Nat.choose n_tuesday (n_correct - n_correct_monday) * p ^ (n_correct - n_correct_monday) * (1 - p) ^ (n_tuesday - (n_correct - n_correct_monday))) /
  (Nat.choose n_total n_correct * p ^ n_correct * (1 - p) ^ (n_total - n_correct)) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_correct_prediction_probability_l788_78890


namespace NUMINAMATH_CALUDE_sum_of_100th_row_general_row_sum_formula_l788_78858

/-- Represents the sum of numbers in the nth row of the triangular array -/
def rowSum (n : ℕ) : ℕ :=
  2^n - 3 * (n - 1)

/-- The triangular array is defined with 0, 1, 2, 3, ... along the sides,
    and interior numbers are obtained by adding the two adjacent numbers
    in the previous row and adding 1 to each sum. -/
axiom array_definition : True

theorem sum_of_100th_row :
  rowSum 100 = 2^100 - 297 :=
by sorry

theorem general_row_sum_formula (n : ℕ) (h : n > 0) :
  rowSum n = 2^n - 3 * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_100th_row_general_row_sum_formula_l788_78858


namespace NUMINAMATH_CALUDE_total_trip_time_l788_78857

def driving_time : ℝ := 5

theorem total_trip_time :
  let traffic_time := 2 * driving_time
  driving_time + traffic_time = 15 := by sorry

end NUMINAMATH_CALUDE_total_trip_time_l788_78857


namespace NUMINAMATH_CALUDE_coin_value_is_70_rupees_l788_78892

/-- Calculates the total value in rupees given the number of coins and their values -/
def total_value_in_rupees (total_coins : ℕ) (coins_20_paise : ℕ) : ℚ :=
  let coins_25_paise := total_coins - coins_20_paise
  let value_20_paise := coins_20_paise * 20
  let value_25_paise := coins_25_paise * 25
  let total_paise := value_20_paise + value_25_paise
  total_paise / 100

/-- Proves that the total value of the given coins is 70 rupees -/
theorem coin_value_is_70_rupees :
  total_value_in_rupees 324 220 = 70 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_is_70_rupees_l788_78892


namespace NUMINAMATH_CALUDE_car_speed_l788_78889

/-- The speed of a car in km/h given the tire's rotation rate and circumference -/
theorem car_speed (revolutions_per_minute : ℝ) (tire_circumference : ℝ) : 
  revolutions_per_minute = 400 → 
  tire_circumference = 1 → 
  (revolutions_per_minute * tire_circumference * 60) / 1000 = 24 := by
sorry

end NUMINAMATH_CALUDE_car_speed_l788_78889


namespace NUMINAMATH_CALUDE_fraction_calculation_l788_78856

theorem fraction_calculation : (16 : ℚ) / 42 * 18 / 27 - 4 / 21 = 4 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l788_78856


namespace NUMINAMATH_CALUDE_max_f_value_l788_78830

theorem max_f_value (a b c d e f : ℝ) 
  (sum_condition : a + b + c + d + e + f = 10)
  (square_sum_condition : (a - 1)^2 + (b - 1)^2 + (c - 1)^2 + (d - 1)^2 + (e - 1)^2 + (f - 1)^2 = 6) :
  f ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_f_value_l788_78830


namespace NUMINAMATH_CALUDE_article_original_price_l788_78842

/-- Given an article with a 25% profit margin, where the profit is 775 rupees, 
    prove that the original price of the article is 3100 rupees. -/
theorem article_original_price (profit_percentage : ℝ) (profit : ℝ) (original_price : ℝ) : 
  profit_percentage = 0.25 →
  profit = 775 →
  profit = profit_percentage * original_price →
  original_price = 3100 :=
by
  sorry

end NUMINAMATH_CALUDE_article_original_price_l788_78842


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l788_78897

theorem wire_ratio_proof (total_length shorter_length : ℝ) 
  (h1 : total_length = 35)
  (h2 : shorter_length = 10)
  (h3 : shorter_length < total_length) :
  shorter_length / (total_length - shorter_length) = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l788_78897


namespace NUMINAMATH_CALUDE_arrangements_for_six_people_l788_78818

/-- The number of people in the line -/
def n : ℕ := 6

/-- The number of arrangements of n people in a line where two specific people
    must stand next to each other and two other specific people must not stand
    next to each other -/
def arrangements (n : ℕ) : ℕ := 
  2 * (n - 2).factorial * ((n - 2) * (n - 3))

/-- Theorem stating that the number of arrangements for 6 people
    under the given conditions is 144 -/
theorem arrangements_for_six_people : arrangements n = 144 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_for_six_people_l788_78818


namespace NUMINAMATH_CALUDE_max_ab_squared_l788_78855

theorem max_ab_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∃ (m : ℝ), m = (4 * Real.sqrt 6) / 9 ∧ ∀ x y : ℝ, 0 < x → 0 < y → x + y = 2 → x * y^2 ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_ab_squared_l788_78855


namespace NUMINAMATH_CALUDE_camel_cost_l788_78811

/-- The cost relationship between animals and the cost of a camel --/
theorem camel_cost (camel horse ox elephant : ℝ) 
  (h1 : 10 * camel = 24 * horse)
  (h2 : 16 * horse = 4 * ox)
  (h3 : 6 * ox = 4 * elephant)
  (h4 : 10 * elephant = 120000) :
  camel = 4800 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l788_78811


namespace NUMINAMATH_CALUDE_angle_around_point_l788_78804

theorem angle_around_point (a b : ℝ) (h1 : a + b + 200 = 360) (h2 : a = b) : a = 80 := by
  sorry

end NUMINAMATH_CALUDE_angle_around_point_l788_78804


namespace NUMINAMATH_CALUDE_chessboard_repaint_theorem_l788_78895

/-- Represents a chessboard of size n × n -/
structure Chessboard (n : ℕ) where
  size : n ≥ 3

/-- Represents an L-shaped tetromino and its rotations -/
inductive Tetromino
  | L
  | RotatedL90
  | RotatedL180
  | RotatedL270

/-- Represents a move that repaints a tetromino on the chessboard -/
def Move (n : ℕ) := Fin n → Fin n → Tetromino

/-- Predicate to check if a series of moves can repaint the entire chessboard -/
def CanRepaintEntireBoard (n : ℕ) (moves : List (Move n)) : Prop :=
  sorry

/-- Main theorem: The chessboard can be entirely repainted if and only if n is even and n ≥ 4 -/
theorem chessboard_repaint_theorem (n : ℕ) (b : Chessboard n) :
  (∃ (moves : List (Move n)), CanRepaintEntireBoard n moves) ↔ (Even n ∧ n ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_chessboard_repaint_theorem_l788_78895


namespace NUMINAMATH_CALUDE_total_rooms_is_260_l788_78846

/-- Represents the hotel booking scenario -/
structure HotelBooking where
  singleRooms : ℕ
  doubleRooms : ℕ
  singleRoomCost : ℕ
  doubleRoomCost : ℕ
  totalIncome : ℕ

/-- Calculates the total number of rooms booked -/
def totalRooms (booking : HotelBooking) : ℕ :=
  booking.singleRooms + booking.doubleRooms

/-- Theorem stating that the total number of rooms booked is 260 -/
theorem total_rooms_is_260 (booking : HotelBooking) 
  (h1 : booking.singleRooms = 64)
  (h2 : booking.singleRoomCost = 35)
  (h3 : booking.doubleRoomCost = 60)
  (h4 : booking.totalIncome = 14000) :
  totalRooms booking = 260 := by
  sorry

#eval totalRooms { singleRooms := 64, doubleRooms := 196, singleRoomCost := 35, doubleRoomCost := 60, totalIncome := 14000 }

end NUMINAMATH_CALUDE_total_rooms_is_260_l788_78846


namespace NUMINAMATH_CALUDE_power_inequality_l788_78876

theorem power_inequality (x y : ℝ) 
  (h1 : x^5 > y^4) 
  (h2 : y^5 > x^4) : 
  x^3 > y^2 := by
sorry

end NUMINAMATH_CALUDE_power_inequality_l788_78876


namespace NUMINAMATH_CALUDE_probability_of_two_queens_or_at_least_one_king_l788_78885

-- Define a standard deck
def standard_deck : ℕ := 52

-- Define the number of queens in a deck
def queens_in_deck : ℕ := 4

-- Define the number of kings in a deck
def kings_in_deck : ℕ := 4

-- Define the probability of the event
def prob_two_queens_or_at_least_one_king : ℚ := 2 / 13

-- State the theorem
theorem probability_of_two_queens_or_at_least_one_king :
  let p := (queens_in_deck * (queens_in_deck - 1) / 2 +
            kings_in_deck * (standard_deck - kings_in_deck) +
            kings_in_deck * (kings_in_deck - 1) / 2) /
           (standard_deck * (standard_deck - 1) / 2)
  p = prob_two_queens_or_at_least_one_king :=
by sorry

end NUMINAMATH_CALUDE_probability_of_two_queens_or_at_least_one_king_l788_78885


namespace NUMINAMATH_CALUDE_no_alpha_sequence_exists_l788_78833

theorem no_alpha_sequence_exists :
  ¬ ∃ (α : ℝ) (a : ℕ → ℝ),
    (0 < α ∧ α < 1) ∧
    (∀ n, 0 < a n) ∧
    (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end NUMINAMATH_CALUDE_no_alpha_sequence_exists_l788_78833


namespace NUMINAMATH_CALUDE_g_evaluation_l788_78853

def g (x : ℝ) : ℝ := 3 * x^3 + 5 * x^2 - 6 * x + 4

theorem g_evaluation : 3 * g 2 - 2 * g (-1) = 84 := by
  sorry

end NUMINAMATH_CALUDE_g_evaluation_l788_78853


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l788_78854

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -1 + Real.sqrt 6 ∧ x₂ = -1 - Real.sqrt 6 ∧ 
    x₁^2 + 2*x₁ = 5 ∧ x₂^2 + 2*x₂ = 5) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ 
    x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -5/2 ∧ x₂ = 1 ∧ 
    2*x₁^2 + 3*x₁ - 5 = 0 ∧ 2*x₂^2 + 3*x₂ - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l788_78854


namespace NUMINAMATH_CALUDE_strawberry_weight_sum_l788_78867

/-- The total weight of Marco's and his dad's strawberries is 40 pounds. -/
theorem strawberry_weight_sum : 
  let marco_weight : ℕ := 8
  let dad_weight : ℕ := 32
  marco_weight + dad_weight = 40 := by sorry

end NUMINAMATH_CALUDE_strawberry_weight_sum_l788_78867


namespace NUMINAMATH_CALUDE_rotation_result_l788_78825

/-- Given a point A(3, -4) rotated counterclockwise by π/2 around the origin,
    the resulting point B has a y-coordinate of 3. -/
theorem rotation_result : ∃ (B : ℝ × ℝ), 
  let A : ℝ × ℝ := (3, -4)
  let angle : ℝ := π / 2
  B.1 = A.1 * Real.cos angle - A.2 * Real.sin angle ∧
  B.2 = A.1 * Real.sin angle + A.2 * Real.cos angle ∧
  B.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_rotation_result_l788_78825


namespace NUMINAMATH_CALUDE_cards_given_to_miguel_l788_78872

/-- Represents the card distribution problem --/
def card_distribution (total_cards : ℕ) (kept_cards : ℕ) (friends : ℕ) (cards_per_friend : ℕ) 
  (sisters : ℕ) (cards_per_sister : ℕ) : ℕ :=
  let remaining_after_keeping := total_cards - kept_cards
  let given_to_friends := friends * cards_per_friend
  let remaining_after_friends := remaining_after_keeping - given_to_friends
  let given_to_sisters := sisters * cards_per_sister
  remaining_after_friends - given_to_sisters

/-- Theorem stating the number of cards Rick gave to Miguel --/
theorem cards_given_to_miguel : 
  card_distribution 250 25 12 15 4 7 = 17 := by
  sorry


end NUMINAMATH_CALUDE_cards_given_to_miguel_l788_78872


namespace NUMINAMATH_CALUDE_max_product_constraint_l788_78840

theorem max_product_constraint (a b : ℝ) (h : a^2 + b^2 = 6) : a * b ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l788_78840


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l788_78841

-- Define a complex number
def complex_number (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Define condition p
def condition_p (a b : ℝ) : Prop := is_purely_imaginary (complex_number a b)

-- Define condition q
def condition_q (a b : ℝ) : Prop := a = 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary :
  (∀ a b : ℝ, condition_p a b → condition_q a b) ∧
  (∃ a b : ℝ, condition_q a b ∧ ¬condition_p a b) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l788_78841


namespace NUMINAMATH_CALUDE_f_properties_l788_78875

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^2 + 12 * x - 15

-- Theorem statement
theorem f_properties :
  -- 1. Zeros of f(x)
  (∃ x : ℝ, f x = 0 ↔ x = -5 ∨ x = 1) ∧
  -- 2. Minimum and maximum values on [-3, 3]
  (∀ x : ℝ, x ∈ Set.Icc (-3) 3 → f x ≥ -27) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ f x = -27) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-3) 3 → f x ≤ 48) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ f x = 48) ∧
  -- 3. f(x) is increasing on [-2, +∞)
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ici (-2) → x₂ ∈ Set.Ici (-2) → x₁ < x₂ → f x₁ < f x₂) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l788_78875


namespace NUMINAMATH_CALUDE_exists_unsolvable_grid_l788_78823

/-- Represents a 9x9 grid with values either 1 or -1 -/
def Grid := Fin 9 → Fin 9 → Int

/-- Defines a valid grid where all values are either 1 or -1 -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j, g i j = 1 ∨ g i j = -1

/-- Defines the neighbors of a cell in the grid -/
def neighbors (i j : Fin 9) : List (Fin 9 × Fin 9) :=
  [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]

/-- Computes the new value of a cell after a move -/
def move (g : Grid) (i j : Fin 9) : Int :=
  (neighbors i j).foldl (λ acc (ni, nj) => acc * g ni nj) 1

/-- Applies a move to the entire grid -/
def apply_move (g : Grid) : Grid :=
  λ i j => move g i j

/-- Checks if all cells in the grid are 1 -/
def all_ones (g : Grid) : Prop :=
  ∀ i j, g i j = 1

/-- The main theorem: there exists a valid grid that cannot be transformed to all ones -/
theorem exists_unsolvable_grid :
  ∃ (g : Grid), valid_grid g ∧ 
    ∀ (n : ℕ), ¬(all_ones ((apply_move^[n]) g)) :=
  sorry

end NUMINAMATH_CALUDE_exists_unsolvable_grid_l788_78823


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l788_78860

theorem quadratic_inequality_no_solution (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - m * x + (m - 1) ≤ 0) ↔ m ≤ -2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l788_78860


namespace NUMINAMATH_CALUDE_product_real_implies_a_equals_two_l788_78831

theorem product_real_implies_a_equals_two (a : ℝ) : 
  let z₁ : ℂ := 2 + Complex.I
  let z₂ : ℂ := a - Complex.I
  (z₁ * z₂).im = 0 → a = 2 := by
sorry

end NUMINAMATH_CALUDE_product_real_implies_a_equals_two_l788_78831


namespace NUMINAMATH_CALUDE_product_of_roots_l788_78893

theorem product_of_roots (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1008 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l788_78893


namespace NUMINAMATH_CALUDE_tangent_line_properties_l788_78822

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the tangent line condition
def tangent_condition (x₁ : ℝ) (a : ℝ) : Prop :=
  ∃ x₂ : ℝ, f' x₁ * (x₂ - x₁) + f x₁ = g a x₂ ∧ f' x₁ = 2 * x₂

-- State the theorem
theorem tangent_line_properties :
  (∀ x₁ a : ℝ, tangent_condition x₁ a → (x₁ = -1 → a = 3)) ∧
  (∀ a : ℝ, (∃ x₁ : ℝ, tangent_condition x₁ a) → a ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l788_78822


namespace NUMINAMATH_CALUDE_grocery_bill_calculation_l788_78803

/-- Calculates the new total bill for a grocery delivery order with item substitutions -/
theorem grocery_bill_calculation
  (original_order : ℝ)
  (tomatoes_old tomatoes_new : ℝ)
  (lettuce_old lettuce_new : ℝ)
  (celery_old celery_new : ℝ)
  (delivery_and_tip : ℝ)
  (h1 : original_order = 25)
  (h2 : tomatoes_old = 0.99)
  (h3 : tomatoes_new = 2.20)
  (h4 : lettuce_old = 1.00)
  (h5 : lettuce_new = 1.75)
  (h6 : celery_old = 1.96)
  (h7 : celery_new = 2.00)
  (h8 : delivery_and_tip = 8.00) :
  original_order + (tomatoes_new - tomatoes_old) + (lettuce_new - lettuce_old) +
  (celery_new - celery_old) + delivery_and_tip = 35 :=
by sorry

end NUMINAMATH_CALUDE_grocery_bill_calculation_l788_78803


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l788_78880

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x ≤ 1}
def Q : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l788_78880


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_prime_factor_constraint_l788_78861

theorem arithmetic_progression_with_prime_factor_constraint :
  ∀ (a b c : ℕ), 
    0 < a → a < b → b < c →
    b - a = c - b →
    (∀ p : ℕ, Prime p → p > 3 → (p ∣ a ∨ p ∣ b ∨ p ∣ c) → False) →
    ∃ (k m n : ℕ), 
      (a = k ∧ b = 2*k ∧ c = 3*k) ∨
      (a = 2*k ∧ b = 3*k ∧ c = 4*k) ∨
      (a = 2*k ∧ b = 9*k ∧ c = 16*k) ∧
      k = 2^m * 3^n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_prime_factor_constraint_l788_78861


namespace NUMINAMATH_CALUDE_parabola_vertex_l788_78844

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 18

/-- The x-coordinate of the vertex -/
def p : ℝ := -2

/-- The y-coordinate of the vertex -/
def q : ℝ := 10

/-- Theorem: The vertex of the parabola y = 2x^2 + 8x + 18 is at (-2, 10) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola p) ∧ parabola p = q := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l788_78844


namespace NUMINAMATH_CALUDE_point_B_coordinates_l788_78805

/-- Given points A and C, and the relation between vectors AB and BC, 
    prove that the coordinates of point B are (-2, 5/3) -/
theorem point_B_coordinates 
  (A B C : ℝ × ℝ) 
  (hA : A = (2, 3)) 
  (hC : C = (0, 1)) 
  (h_vec : B - A = -2 • (C - B)) : 
  B = (-2, 5/3) := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l788_78805


namespace NUMINAMATH_CALUDE_solution_implies_value_l788_78810

theorem solution_implies_value (a b : ℝ) : 
  (-a * 3 - b = 5 - 2 * 3) → (3 - 6 * a - 2 * b = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_value_l788_78810


namespace NUMINAMATH_CALUDE_product_of_roots_l788_78874

theorem product_of_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 15*x^2 + 75*x - 50 = (x - r₁) * (x - r₂) * (x - r₃)) → 
  ∃ r₁ r₂ r₃ : ℝ, x^3 - 15*x^2 + 75*x - 50 = (x - r₁) * (x - r₂) * (x - r₃) ∧ r₁ * r₂ * r₃ = 50 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l788_78874


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l788_78809

/-- Proves that if a farmer has 479 tomatoes and picks 364 of them, he will have 115 tomatoes left. -/
theorem farmer_tomatoes (initial : ℕ) (picked : ℕ) (remaining : ℕ) : 
  initial = 479 → picked = 364 → remaining = initial - picked → remaining = 115 :=
by sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l788_78809


namespace NUMINAMATH_CALUDE_complex_sum_eighth_power_l788_78862

open Complex

theorem complex_sum_eighth_power :
  ((-1 + I) / 2) ^ 8 + ((-1 - I) / 2) ^ 8 = (1 : ℂ) / 8 := by sorry

end NUMINAMATH_CALUDE_complex_sum_eighth_power_l788_78862


namespace NUMINAMATH_CALUDE_C_share_of_profit_l788_78834

def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 2000
def total_profit : ℕ := 252000

theorem C_share_of_profit :
  (investment_C : ℚ) / (investment_A + investment_B + investment_C) * total_profit = 36000 :=
by sorry

end NUMINAMATH_CALUDE_C_share_of_profit_l788_78834


namespace NUMINAMATH_CALUDE_remainder_problem_l788_78870

theorem remainder_problem : 123456789012 % 200 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l788_78870


namespace NUMINAMATH_CALUDE_nicky_running_time_l788_78866

/-- The time Nicky runs before Cristina catches up to him in a race with given conditions -/
theorem nicky_running_time (race_length : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_length = 500)
  (h2 : cristina_speed > nicky_speed)
  (h3 : head_start = 12)
  (h4 : cristina_speed = 5)
  (h5 : nicky_speed = 3) :
  head_start + (head_start * nicky_speed) / (cristina_speed - nicky_speed) = 30 := by
  sorry

#check nicky_running_time

end NUMINAMATH_CALUDE_nicky_running_time_l788_78866


namespace NUMINAMATH_CALUDE_parallelogram_area_in_regular_hexagon_l788_78878

/-- The area of the parallelogram formed by connecting every second vertex of a regular hexagon --/
theorem parallelogram_area_in_regular_hexagon (side_length : ℝ) (h : side_length = 12) :
  let large_triangle_area := Real.sqrt 3 / 4 * (2 * side_length) ^ 2
  let small_triangle_area := Real.sqrt 3 / 4 * side_length ^ 2
  large_triangle_area - 3 * small_triangle_area = 36 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_in_regular_hexagon_l788_78878


namespace NUMINAMATH_CALUDE_simplify_square_roots_l788_78814

theorem simplify_square_roots : 
  (Real.sqrt 448 / Real.sqrt 128) + (Real.sqrt 98 / Real.sqrt 49) = 
  (Real.sqrt 14 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l788_78814


namespace NUMINAMATH_CALUDE_solve_inequality_m_neg_four_solve_inequality_x_greater_than_one_l788_78883

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| - |2*x + m|

-- Theorem for part (1)
theorem solve_inequality_m_neg_four :
  ∀ x : ℝ, f x (-4) < 0 ↔ x < 5/3 ∨ x > 3 := by sorry

-- Theorem for part (2)
theorem solve_inequality_x_greater_than_one :
  ∀ m : ℝ, (∀ x : ℝ, x > 1 → f x m < 0) ↔ m ≥ -2 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_m_neg_four_solve_inequality_x_greater_than_one_l788_78883


namespace NUMINAMATH_CALUDE_existence_of_m_n_l788_78832

theorem existence_of_m_n (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n ≤ (p + 1) / 2 ∧ p ∣ 2^n * 3^m - 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l788_78832


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l788_78847

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallelism relation between planes
variable (parallel_plane : Plane → Plane → Prop)

-- Define the intersection relation between lines
variable (intersect : Line → Line → Prop)

-- Define the relation for a line being outside a plane
variable (outside : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_parallelism 
  (m n : Line) (α β : Plane) :
  intersect m n ∧ 
  outside m α ∧ outside m β ∧ 
  outside n α ∧ outside n β ∧
  parallel_line_plane m α ∧ parallel_line_plane m β ∧ 
  parallel_line_plane n α ∧ parallel_line_plane n β →
  parallel_plane α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l788_78847


namespace NUMINAMATH_CALUDE_extreme_value_and_maximum_l788_78821

-- Define the function f and its derivative
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x) + a * Real.cos x + x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (2 * x) - a * Real.sin x + 1

theorem extreme_value_and_maximum (a : ℝ) :
  f' a (π / 6) = 0 →
  a = 4 ∧
  ∀ x ∈ Set.Icc (-π / 6) (7 * π / 6), f 4 x ≤ (5 * Real.sqrt 3) / 2 + π / 6 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_maximum_l788_78821


namespace NUMINAMATH_CALUDE_k_range_for_two_distinct_roots_l788_78852

/-- A quadratic equation ax^2 + bx + c = 0 has two distinct real roots if and only if its discriminant is positive -/
axiom two_distinct_roots_iff_positive_discriminant (a b c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b^2 - 4*a*c > 0

/-- The range of k for which kx^2 - 6x + 9 = 0 has two distinct real roots -/
theorem k_range_for_two_distinct_roots :
  ∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6 * x + 9 = 0 ∧ k * y^2 - 6 * y + 9 = 0) ↔ k < 1 ∧ k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_two_distinct_roots_l788_78852


namespace NUMINAMATH_CALUDE_compare_b_and_d_l788_78849

theorem compare_b_and_d (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a = b * 1.02)
  (hac : c = a * 0.99)
  (hcd : d = c * 0.99) : 
  b > d := by
sorry

end NUMINAMATH_CALUDE_compare_b_and_d_l788_78849


namespace NUMINAMATH_CALUDE_dog_length_calculation_l788_78869

/-- Represents the length of a dog's body parts in inches -/
structure DogMeasurements where
  tail_length : ℝ
  body_length : ℝ
  head_length : ℝ

/-- Calculates the overall length of a dog given its measurements -/
def overall_length (d : DogMeasurements) : ℝ :=
  d.body_length + d.head_length

/-- Theorem stating the overall length of a dog with specific proportions -/
theorem dog_length_calculation (d : DogMeasurements) 
  (h1 : d.tail_length = d.body_length / 2)
  (h2 : d.head_length = d.body_length / 6)
  (h3 : d.tail_length = 9) :
  overall_length d = 21 := by
  sorry

#check dog_length_calculation

end NUMINAMATH_CALUDE_dog_length_calculation_l788_78869


namespace NUMINAMATH_CALUDE_subset_iff_range_l788_78837

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x - 4 ≤ 0}

-- State the theorem
theorem subset_iff_range (a : ℝ) : B a ⊆ A ↔ 0 ≤ a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_subset_iff_range_l788_78837


namespace NUMINAMATH_CALUDE_max_k_value_l788_78829

theorem max_k_value (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : k * a * b * c / (a + b + c) ≤ (a + b)^2 + (a + b + 4*c)^2) : 
  k ≤ 100 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l788_78829


namespace NUMINAMATH_CALUDE_prob_ace_of_spades_l788_78871

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (rank_count : (cards.image Prod.fst).card = 13)
  (suit_count : (cards.image Prod.snd).card = 4)

/-- The probability of drawing a specific card from a shuffled deck -/
def prob_draw_specific_card (d : Deck) : ℚ :=
  1 / 52

/-- Theorem: The probability of drawing the Ace of Spades from a shuffled standard deck is 1/52 -/
theorem prob_ace_of_spades (d : Deck) :
  prob_draw_specific_card d = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_of_spades_l788_78871


namespace NUMINAMATH_CALUDE_total_interest_received_l788_78800

/-- Simple interest calculation function -/
def simple_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time / 100

theorem total_interest_received (loan_b_principal loan_c_principal : ℕ) 
  (loan_b_time loan_c_time : ℕ) (interest_rate : ℚ) : 
  loan_b_principal = 5000 →
  loan_c_principal = 3000 →
  loan_b_time = 2 →
  loan_c_time = 4 →
  interest_rate = 15 →
  simple_interest loan_b_principal interest_rate loan_b_time + 
  simple_interest loan_c_principal interest_rate loan_c_time = 3300 := by
sorry

end NUMINAMATH_CALUDE_total_interest_received_l788_78800


namespace NUMINAMATH_CALUDE_emily_marbles_l788_78899

/-- Emily's marble problem -/
theorem emily_marbles :
  let initial_marbles : ℕ := 6
  let megan_gives := 2 * initial_marbles
  let emily_new_total := initial_marbles + megan_gives
  let emily_gives_back := emily_new_total / 2 + 1
  let emily_final := emily_new_total - emily_gives_back
  emily_final = 8 := by sorry

end NUMINAMATH_CALUDE_emily_marbles_l788_78899
