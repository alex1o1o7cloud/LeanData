import Mathlib

namespace minimum_rows_required_l435_43576

theorem minimum_rows_required (n : ℕ) : (3 * n * (n + 1)) / 2 ≥ 150 ↔ n ≥ 10 := 
by
  sorry

end minimum_rows_required_l435_43576


namespace grunters_win_all_five_l435_43580

theorem grunters_win_all_five (p : ℚ) (games : ℕ) (win_prob : ℚ) :
  games = 5 ∧ win_prob = 3 / 5 → 
  p = (win_prob) ^ games ∧ p = 243 / 3125 := 
by
  intros h
  cases h
  sorry

end grunters_win_all_five_l435_43580


namespace sum_distances_between_l435_43507

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2).sqrt

theorem sum_distances_between (A B D : ℝ × ℝ)
  (hB : B = (0, 5))
  (hD : D = (8, 0))
  (hA : A = (20, 0)) :
  21 < distance A D + distance B D ∧ distance A D + distance B D < 22 :=
by
  sorry

end sum_distances_between_l435_43507


namespace average_weight_of_three_l435_43528

theorem average_weight_of_three :
  ∀ A B C : ℝ,
  (A + B) / 2 = 40 →
  (B + C) / 2 = 43 →
  B = 31 →
  (A + B + C) / 3 = 45 :=
by
  intros A B C h1 h2 h3
  sorry

end average_weight_of_three_l435_43528


namespace length_of_bridge_l435_43553

theorem length_of_bridge
  (T : ℕ) (t : ℕ) (s : ℕ)
  (hT : T = 250)
  (ht : t = 20)
  (hs : s = 20) :
  ∃ L : ℕ, L = 150 :=
by
  sorry

end length_of_bridge_l435_43553


namespace sum_of_possible_n_values_l435_43597

theorem sum_of_possible_n_values (m n : ℕ) 
  (h : 0 < m ∧ 0 < n)
  (eq1 : 1/m + 1/n = 1/5) : 
  n = 6 ∨ n = 10 ∨ n = 30 → 
  m = 30 ∨ m = 10 ∨ m = 6 ∨ m = 5 ∨ m = 25 ∨ m = 1 →
  (6 + 10 + 30 = 46) := 
by 
  sorry

end sum_of_possible_n_values_l435_43597


namespace problem_1_problem_2_l435_43500

-- Define the function f(x) = |x + a| + |x|
def f (x : ℝ) (a : ℝ) : ℝ := abs (x + a) + abs x

-- (Ⅰ) Prove that for a = 1, the solution set for f(x) ≥ 2 is (-∞, -1/2] ∪ [3/2, +∞)
theorem problem_1 : 
  ∀ (x : ℝ), f x 1 ≥ 2 ↔ (x ≤ -1/2 ∨ x ≥ 3/2) :=
by
  intro x
  sorry

-- (Ⅱ) Prove that if there exists x ∈ ℝ such that f(x) < 2, then -2 < a < 2
theorem problem_2 :
  (∃ (x : ℝ), f x a < 2) → -2 < a ∧ a < 2 :=
by
  intro h
  sorry

end problem_1_problem_2_l435_43500


namespace base_number_of_equation_l435_43501

theorem base_number_of_equation (y : ℕ) (b : ℕ) (h1 : 16 ^ y = b ^ 14) (h2 : y = 7) : b = 4 := 
by 
  sorry

end base_number_of_equation_l435_43501


namespace interest_earned_l435_43581

-- Define the principal, interest rate, and number of years
def principal : ℝ := 1200
def annualInterestRate : ℝ := 0.12
def numberOfYears : ℕ := 4

-- Define the compound interest formula
def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Define the total interest earned
def totalInterest (P A : ℝ) : ℝ :=
  A - P

-- State the theorem
theorem interest_earned :
  totalInterest principal (compoundInterest principal annualInterestRate numberOfYears) = 688.224 :=
by
  sorry

end interest_earned_l435_43581


namespace tims_total_earnings_l435_43596

theorem tims_total_earnings (days_of_week : ℕ) (tasks_per_day : ℕ) (tasks_40_rate : ℕ) (tasks_30_rate1 : ℕ) (tasks_30_rate2 : ℕ)
    (rate_40 : ℝ) (rate_30_1 : ℝ) (rate_30_2 : ℝ) (bonus_per_50 : ℝ) (performance_bonus : ℝ)
    (total_earnings : ℝ) :
  days_of_week = 6 →
  tasks_per_day = 100 →
  tasks_40_rate = 40 →
  tasks_30_rate1 = 30 →
  tasks_30_rate2 = 30 →
  rate_40 = 1.2 →
  rate_30_1 = 1.5 →
  rate_30_2 = 2.0 →
  bonus_per_50 = 10 →
  performance_bonus = 20 →
  total_earnings = 1058 :=
by
  intros
  sorry

end tims_total_earnings_l435_43596


namespace even_not_divisible_by_4_not_sum_of_two_consecutive_odds_l435_43552

theorem even_not_divisible_by_4_not_sum_of_two_consecutive_odds (x n : ℕ) (h₁ : Even x) (h₂ : ¬ ∃ k, x = 4 * k) : x ≠ (2 * n + 1) + (2 * n + 3) := by
  sorry

end even_not_divisible_by_4_not_sum_of_two_consecutive_odds_l435_43552


namespace solve_equation_l435_43578

theorem solve_equation (x : ℝ) (h : (x - 3) / 2 - (2 * x) / 3 = 1) : x = -15 := 
by 
  sorry

end solve_equation_l435_43578


namespace range_of_a_zeros_of_g_l435_43592

-- Definitions for the original functions f and g and their corresponding conditions
noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2

noncomputable def g (x x2 a : ℝ) : ℝ := f x a - (x2 / 2)

-- Proving the range of a
theorem range_of_a (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ x1 * Real.log x1 - (a / 2) * x1^2 = 0 ∧ x2 * Real.log x2 - (a / 2) * x2^2 = 0) :
  0 < a ∧ a < 1 := 
sorry

-- Proving the number of zeros of g based on the value of a
theorem zeros_of_g (a : ℝ) (x1 x2 : ℝ) (h : x1 < x2 ∧ x1 * Real.log x1 - (a / 2) * x1^2 = 0 ∧ x2 * Real.log x2 - (a / 2) * x2^2 = 0) :
  (0 < a ∧ a < 3 / Real.exp 2 → ∃ x3 x4, x3 ≠ x4 ∧ g x3 x2 a = 0 ∧ g x4 x2 a = 0) ∧
  (a = 3 / Real.exp 2 → ∃ x3, g x3 x2 a = 0) ∧
  (3 / Real.exp 2 < a ∧ a < 1 → ∀ x, g x x2 a ≠ 0) :=
sorry

end range_of_a_zeros_of_g_l435_43592


namespace scientific_notation_correct_l435_43566

theorem scientific_notation_correct :
  27600 = 2.76 * 10^4 :=
sorry

end scientific_notation_correct_l435_43566


namespace time_period_is_12_hours_l435_43599

-- Define the conditions in the problem
def birth_rate := 8 / 2 -- people per second
def death_rate := 6 / 2 -- people per second
def net_increase := 86400 -- people

-- Define the net increase per second
def net_increase_per_second := birth_rate - death_rate

-- Total time period in seconds
def time_period_seconds := net_increase / net_increase_per_second

-- Convert the time period to hours
def time_period_hours := time_period_seconds / 3600

-- The theorem we want to state and prove
theorem time_period_is_12_hours : time_period_hours = 12 :=
by
  -- Proof goes here
  sorry

end time_period_is_12_hours_l435_43599


namespace fraction_value_l435_43513

theorem fraction_value (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (1 / (y : ℚ) / (1 / (x : ℚ))) = 3 / 4 :=
by
  rw [hx, hy]
  norm_num

end fraction_value_l435_43513


namespace barrels_are_1360_l435_43509

-- Defining the top layer dimensions and properties
def a : ℕ := 2
def b : ℕ := 1
def n : ℕ := 15

-- Defining the dimensions of the bottom layer based on given properties
def c : ℕ := a + n
def d : ℕ := b + n

-- Formula for the total number of barrels
def total_barrels : ℕ := n * ((2 * a + c) * b + (2 * c + a) * d + (d - b)) / 6

-- Theorem to prove
theorem barrels_are_1360 : total_barrels = 1360 :=
by
  sorry

end barrels_are_1360_l435_43509


namespace probability_at_least_one_of_each_color_l435_43575

theorem probability_at_least_one_of_each_color
  (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ)
  (h_total : total_balls = 16)
  (h_black : black_balls = 8)
  (h_white : white_balls = 5)
  (h_red : red_balls = 3) :
  ((black_balls.choose 1) * (white_balls.choose 1) * (red_balls.choose 1) : ℚ) / total_balls.choose 3 = 3 / 14 :=
by
  sorry

end probability_at_least_one_of_each_color_l435_43575


namespace power_of_power_l435_43586

theorem power_of_power (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := 
  sorry

end power_of_power_l435_43586


namespace find_f_value_l435_43585

noncomputable def f (x y z : ℝ) : ℝ := 2 * x^3 * Real.sin y + Real.log (z^2)

theorem find_f_value :
  f 1 (Real.pi / 2) (Real.exp 2) = 8 →
  f 2 Real.pi (Real.exp 3) = 6 :=
by
  intro h
  unfold f
  sorry

end find_f_value_l435_43585


namespace vova_gave_pavlik_three_nuts_l435_43589

variable {V P k : ℕ}
variable (h1 : V > P)
variable (h2 : V - P = 2 * P)
variable (h3 : k ≤ 5)
variable (h4 : ∃ m : ℕ, V - k = 3 * m)

theorem vova_gave_pavlik_three_nuts (h1 : V > P) (h2 : V - P = 2 * P) (h3 : k ≤ 5) (h4 : ∃ m : ℕ, V - k = 3 * m) : k = 3 := by
  sorry

end vova_gave_pavlik_three_nuts_l435_43589


namespace sufficient_but_not_necessary_condition_l435_43546

def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3
def q (x : ℝ) : Prop := x ≠ 0

theorem sufficient_but_not_necessary_condition (h: ∀ x : ℝ, p x → q x) : (∀ x : ℝ, q x → p x) → false := sorry

end sufficient_but_not_necessary_condition_l435_43546


namespace Keiko_speed_l435_43568

theorem Keiko_speed (a b s : ℝ) (h1 : 8 = 8) 
  (h2 : (2 * a + 2 * π * (b + 8)) / s = (2 * a + 2 * π * b) / s + 48) : 
  s = π / 3 :=
by
  sorry

end Keiko_speed_l435_43568


namespace katie_spending_l435_43583

theorem katie_spending :
  let price_per_flower : ℕ := 6
  let number_of_roses : ℕ := 5
  let number_of_daisies : ℕ := 5
  let total_number_of_flowers := number_of_roses + number_of_daisies
  let total_spending := total_number_of_flowers * price_per_flower
  total_spending = 60 :=
by
  sorry

end katie_spending_l435_43583


namespace hyperbola_eccentricity_l435_43504

theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (h4 : c = 3 * b) 
  (h5 : c * c = a * a + b * b)
  (h6 : e = c / a) :
  e = 3 * Real.sqrt 2 / 4 :=
by
  sorry

end hyperbola_eccentricity_l435_43504


namespace sampling_probability_equal_l435_43524

theorem sampling_probability_equal :
  let total_people := 2014
  let first_sample := 14
  let remaining_people := total_people - first_sample
  let sample_size := 50
  let probability := sample_size / total_people
  50 / 2014 = 25 / 1007 :=
by
  sorry

end sampling_probability_equal_l435_43524


namespace alex_ate_more_pears_than_sam_l435_43506

namespace PearEatingContest

def number_of_pears_eaten (Alex Sam : ℕ) : ℕ :=
  Alex - Sam

theorem alex_ate_more_pears_than_sam :
  number_of_pears_eaten 8 2 = 6 := by
  -- proof
  sorry

end PearEatingContest

end alex_ate_more_pears_than_sam_l435_43506


namespace distance_from_point_to_asymptote_l435_43520

theorem distance_from_point_to_asymptote :
  ∃ (d : ℝ), ∀ (x₀ y₀ : ℝ), (x₀, y₀) = (3, 0) ∧ 3 * x₀ - 4 * y₀ = 0 →
  d = 9 / 5 :=
by
  sorry

end distance_from_point_to_asymptote_l435_43520


namespace find_sum_l435_43534

variables (x y : ℝ)

def condition1 : Prop := x^3 - 3 * x^2 + 5 * x = 1
def condition2 : Prop := y^3 - 3 * y^2 + 5 * y = 5

theorem find_sum : condition1 x → condition2 y → x + y = 2 := 
by 
  sorry -- The proof goes here

end find_sum_l435_43534


namespace divisor_of_1058_l435_43557

theorem divisor_of_1058 :
  ∃ (d : ℕ), (∃ (k : ℕ), 1058 = d * k) ∧ (¬ ∃ (d : ℕ), (∃ (l : ℕ), 1 < d ∧ d < 1058 ∧ 1058 = d * l)) :=
by {
  sorry
}

end divisor_of_1058_l435_43557


namespace vertex_of_parabola_is_max_and_correct_l435_43514

theorem vertex_of_parabola_is_max_and_correct (x y : ℝ) (h : y = -3 * x^2 + 6 * x + 1) :
  (x, y) = (1, 4) ∧ ∃ ε > 0, ∀ z : ℝ, abs (z - x) < ε → y ≥ -3 * z^2 + 6 * z + 1 :=
by
  sorry

end vertex_of_parabola_is_max_and_correct_l435_43514


namespace least_number_of_stamps_l435_43526

def min_stamps (x y : ℕ) : ℕ := x + y

theorem least_number_of_stamps {x y : ℕ} (h : 5 * x + 7 * y = 50) 
  : min_stamps x y = 8 :=
sorry

end least_number_of_stamps_l435_43526


namespace black_white_difference_l435_43508

theorem black_white_difference (m n : ℕ) (h_dim : m = 7 ∧ n = 9) (h_first_black : m % 2 = 1 ∧ n % 2 = 1) :
  let black_count := (5 * 4 + 4 * 3)
  let white_count := (4 * 4 + 5 * 3)
  black_count - white_count = 1 := 
by
  -- We start with known dimensions and conditions
  let ⟨hm, hn⟩ := h_dim
  have : m = 7 := by rw [hm]
  have : n = 9 := by rw [hn]
  
  -- Calculate the number of black and white squares 
  let black_count := (5 * 4 + 4 * 3)
  let white_count := (4 * 4 + 5 * 3)
  
  -- Use given formulas to calculate the difference
  have diff : black_count - white_count = 1 := by
    sorry -- proof to be provided
  
  exact diff

end black_white_difference_l435_43508


namespace inequality_transformation_l435_43521

variable (x y : ℝ)

theorem inequality_transformation (h : x > y) : x - 2 > y - 2 :=
by
  sorry

end inequality_transformation_l435_43521


namespace sum_of_angles_l435_43562

variables (A B C D E F : ℝ)

theorem sum_of_angles 
  (h : E = 30) :
  A + B + C + D + E + F = 420 :=
sorry

end sum_of_angles_l435_43562


namespace angles_of_triangle_l435_43560

theorem angles_of_triangle 
  (α β γ : ℝ)
  (triangle_ABC : α + β + γ = 180)
  (median_bisector_height : (γ / 4) * 4 = 90) :
  α = 22.5 ∧ β = 67.5 ∧ γ = 90 :=
by
  sorry

end angles_of_triangle_l435_43560


namespace isosceles_trapezoid_ratio_l435_43587

theorem isosceles_trapezoid_ratio (a b h : ℝ) 
  (h1: h = b / 2)
  (h2: a = 1 - ((1 - b) / 2))
  (h3 : 1 = ((a + 1) / 2)^2 + (b / 2)^2) :
  b / a = (-1 + Real.sqrt 7) / 2 := 
sorry

end isosceles_trapezoid_ratio_l435_43587


namespace total_hotdogs_sold_l435_43563

theorem total_hotdogs_sold : 
  let small := 58.3
  let medium := 21.7
  let large := 35.9
  let extra_large := 15.4
  small + medium + large + extra_large = 131.3 :=
by 
  sorry

end total_hotdogs_sold_l435_43563


namespace eggs_per_day_second_store_l435_43564

-- Define the number of eggs in a dozen
def eggs_in_a_dozen : ℕ := 12

-- Define the number of dozen eggs supplied to the first store each day
def dozen_per_day_first_store : ℕ := 5

-- Define the number of eggs supplied to the first store each day
def eggs_per_day_first_store : ℕ := dozen_per_day_first_store * eggs_in_a_dozen

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Calculate the weekly supply to the first store
def weekly_supply_first_store : ℕ := eggs_per_day_first_store * days_in_week

-- Define the total weekly supply to both stores
def total_weekly_supply : ℕ := 630

-- Calculate the weekly supply to the second store
def weekly_supply_second_store : ℕ := total_weekly_supply - weekly_supply_first_store

-- Define the theorem to prove the number of eggs supplied to the second store each day
theorem eggs_per_day_second_store : weekly_supply_second_store / days_in_week = 30 := by
  sorry

end eggs_per_day_second_store_l435_43564


namespace henry_classical_cds_l435_43561

variable (R C : ℕ)

theorem henry_classical_cds :
  (23 - 3 = R) →
  (R = 2 * C) →
  C = 10 :=
by
  intros h1 h2
  sorry

end henry_classical_cds_l435_43561


namespace sequence_inequality_l435_43570

theorem sequence_inequality (a : ℕ → ℤ) (h₀ : a 1 > a 0) 
  (h₁ : ∀ n : ℕ, n ≥ 1 → a (n+1) = 3 * a n - 2 * a (n-1)) : 
  a 100 > 2^99 := 
sorry

end sequence_inequality_l435_43570


namespace one_third_sugar_amount_l435_43523

-- Define the original amount of sugar as a mixed number
def original_sugar_mixed : ℚ := 6 + 1 / 3

-- Define the fraction representing one-third of the recipe
def one_third : ℚ := 1 / 3

-- Define the expected amount of sugar for one-third of the recipe
def expected_sugar_mixed : ℚ := 2 + 1 / 9

-- The theorem stating the proof problem
theorem one_third_sugar_amount : (one_third * original_sugar_mixed) = expected_sugar_mixed :=
sorry

end one_third_sugar_amount_l435_43523


namespace evaluate_f_at_3_l435_43550

-- Function definition
def f (x : ℚ) : ℚ := (x - 2) / (4 * x + 5)

-- Problem statement
theorem evaluate_f_at_3 : f 3 = 1 / 17 := by
  sorry

end evaluate_f_at_3_l435_43550


namespace average_runs_l435_43525

theorem average_runs (games : ℕ) (runs1 matches1 runs2 matches2 runs3 matches3 : ℕ)
  (h1 : runs1 = 1) 
  (h2 : matches1 = 1) 
  (h3 : runs2 = 4) 
  (h4 : matches2 = 2)
  (h5 : runs3 = 5) 
  (h6 : matches3 = 3) 
  (h_games : games = matches1 + matches2 + matches3) :
  (runs1 * matches1 + runs2 * matches2 + runs3 * matches3) / games = 4 :=
by
  sorry

end average_runs_l435_43525


namespace tile_count_difference_l435_43518

theorem tile_count_difference :
  let red_initial := 15
  let yellow_initial := 10
  let yellow_added := 18
  let yellow_total := yellow_initial + yellow_added
  let red_total := red_initial
  yellow_total - red_total = 13 :=
by
  sorry

end tile_count_difference_l435_43518


namespace smallest_number_divisible_l435_43537

/-- The smallest number which, when diminished by 20, is divisible by 15, 30, 45, and 60 --/
theorem smallest_number_divisible (n : ℕ) (h : ∀ k : ℕ, n - 20 = k * Int.lcm 15 (Int.lcm 30 (Int.lcm 45 60))) : n = 200 :=
sorry

end smallest_number_divisible_l435_43537


namespace sufficient_not_necessary_condition_l435_43516

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ (a : ℝ), a = 2 → (-(a) * (a / 4) = -1)) ∧ ∀ (a : ℝ), (-(a) * (a / 4) = -1 → a = 2 ∨ a = -2) :=
by
  sorry

end sufficient_not_necessary_condition_l435_43516


namespace evaluate_at_minus_three_l435_43531

def g (x : ℝ) : ℝ := 3 * x^5 - 5 * x^4 + 9 * x^3 - 6 * x^2 + 15 * x - 210

theorem evaluate_at_minus_three : g (-3) = -1686 :=
by
  sorry

end evaluate_at_minus_three_l435_43531


namespace twenty_one_less_than_sixty_thousand_l435_43543

theorem twenty_one_less_than_sixty_thousand : 60000 - 21 = 59979 :=
by
  sorry

end twenty_one_less_than_sixty_thousand_l435_43543


namespace intersection_of_A_and_B_l435_43593

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | 1 ≤ x ∧ x < 4}

-- The theorem stating the problem
theorem intersection_of_A_and_B : A ∩ B = {1} :=
by
  sorry

end intersection_of_A_and_B_l435_43593


namespace someone_received_grade_D_or_F_l435_43511

theorem someone_received_grade_D_or_F (m x : ℕ) (hboys : ∃ n : ℕ, n = m + 3) 
  (hgrades_B : ∃ k : ℕ, k = x + 2) (hgrades_C : ∃ l : ℕ, l = 2 * (x + 2)) :
  ∃ p : ℕ, p = 1 ∨ p = 2 :=
by
  sorry

end someone_received_grade_D_or_F_l435_43511


namespace daisy_lunch_vs_breakfast_spending_l435_43529

noncomputable def breakfast_cost : ℝ := 2 + 3 + 4 + 3.5 + 1.5
noncomputable def lunch_base_cost : ℝ := 3.5 + 4 + 5.25 + 6 + 1 + 3
noncomputable def service_charge : ℝ := 0.10 * lunch_base_cost
noncomputable def lunch_cost_with_service_charge : ℝ := lunch_base_cost + service_charge
noncomputable def food_tax : ℝ := 0.05 * lunch_cost_with_service_charge
noncomputable def total_lunch_cost : ℝ := lunch_cost_with_service_charge + food_tax
noncomputable def difference : ℝ := total_lunch_cost - breakfast_cost

theorem daisy_lunch_vs_breakfast_spending :
  difference = 12.28 :=
by 
  sorry

end daisy_lunch_vs_breakfast_spending_l435_43529


namespace part1_general_formula_part2_sum_S_l435_43595

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => a n + 1

theorem part1_general_formula (n : ℕ) : a n = n + 1 := by
  sorry

noncomputable def b (n : ℕ) : ℝ := 1 / (↑n * ↑(n + 2))

noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i => b (i + 1))

theorem part2_sum_S (n : ℕ) : 
  S n = (1/2) * ((3/2) - (1 / (n + 1)) - (1 / (n + 2))) := by
  sorry

end part1_general_formula_part2_sum_S_l435_43595


namespace least_possible_value_of_smallest_integer_l435_43582

theorem least_possible_value_of_smallest_integer 
  (A B C D : ℤ) 
  (H_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (H_avg : (A + B + C + D) / 4 = 74)
  (H_max : D = 90) :
  A ≥ 31 :=
by sorry

end least_possible_value_of_smallest_integer_l435_43582


namespace solve_x_eq_40_l435_43510

theorem solve_x_eq_40 : ∀ (x : ℝ), x + 2 * x = 400 - (3 * x + 4 * x) → x = 40 :=
by
  intro x
  intro h
  sorry

end solve_x_eq_40_l435_43510


namespace calculation_l435_43536

theorem calculation (a b c d e : ℤ)
  (h1 : a = (-4)^6)
  (h2 : b = 4^4)
  (h3 : c = 2^5)
  (h4 : d = 7^2)
  (h5 : e = (a / b) + c - d) :
  e = -1 := by
  sorry

end calculation_l435_43536


namespace mass_percentage_Al_in_AlBr₃_l435_43594

theorem mass_percentage_Al_in_AlBr₃ :
  let Al_mass := 26.98
  let Br_mass := 79.90
  let M_AlBr₃ := Al_mass + 3 * Br_mass
  (Al_mass / M_AlBr₃ * 100) = 10.11 :=
by 
  let Al_mass := 26.98
  let Br_mass := 79.90
  let M_AlBr₃ := Al_mass + 3 * Br_mass
  have : (Al_mass / M_AlBr₃ * 100) = 10.11 := sorry
  assumption

end mass_percentage_Al_in_AlBr₃_l435_43594


namespace determine_speeds_l435_43545

structure Particle :=
  (speed : ℝ)

def distance : ℝ := 3.01 -- meters

def initial_distance (m1_speed : ℝ) : ℝ :=
  301 - 11 * m1_speed -- converted to cm

theorem determine_speeds :
  ∃ (m1 m2 : Particle), 
  m1.speed = 11 ∧ m2.speed = 7 ∧ 
  ∀ t : ℝ, (t = 10 ∨ t = 45) →
  (initial_distance m1.speed) = t * (m1.speed + m2.speed) ∧
  20 * m2.speed = 35 * (m1.speed - m2.speed) :=
by {
  sorry 
}

end determine_speeds_l435_43545


namespace angle_A_range_l435_43522

theorem angle_A_range (a b : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) :
  ∃ A : ℝ, 0 < A ∧ A ≤ Real.pi / 4 :=
sorry

end angle_A_range_l435_43522


namespace distance_on_dirt_section_distance_on_mud_section_l435_43556

noncomputable def v_highway : ℝ := 120 -- km/h
noncomputable def v_dirt : ℝ := 40 -- km/h
noncomputable def v_mud : ℝ := 10 -- km/h
noncomputable def initial_distance : ℝ := 0.6 -- km

theorem distance_on_dirt_section : 
  ∃ s_1 : ℝ, 
  (s_1 = 0.2 * 1000 ∧ -- converting km to meters
  v_highway = 120 ∧ 
  v_dirt = 40 ∧ 
  v_mud = 10 ∧ 
  initial_distance = 0.6 ) :=
sorry

theorem distance_on_mud_section : 
  ∃ s_2 : ℝ, 
  (s_2 = 50 ∧
  v_highway = 120 ∧ 
  v_dirt = 40 ∧ 
  v_mud = 10 ∧ 
  initial_distance = 0.6 ) :=
sorry

end distance_on_dirt_section_distance_on_mud_section_l435_43556


namespace combined_score_210_l435_43598

-- Define the constants and variables
def total_questions : ℕ := 50
def marks_per_question : ℕ := 2
def jose_wrong_questions : ℕ := 5
def jose_extra_marks (alisson_score : ℕ) : ℕ := 40
def meghan_less_marks (jose_score : ℕ) : ℕ := 20

-- Define the total possible marks
def total_possible_marks : ℕ := total_questions * marks_per_question

-- Given the conditions, we need to prove the total combined score is 210
theorem combined_score_210 : 
  ∃ (jose_score meghan_score alisson_score combined_score : ℕ), 
  jose_score = total_possible_marks - (jose_wrong_questions * marks_per_question) ∧
  meghan_score = jose_score - meghan_less_marks jose_score ∧
  alisson_score = jose_score - jose_extra_marks alisson_score ∧
  combined_score = jose_score + meghan_score + alisson_score ∧
  combined_score = 210 := by
  sorry

end combined_score_210_l435_43598


namespace paint_amount_third_day_l435_43555

theorem paint_amount_third_day : 
  let initial_paint := 80
  let first_day_usage := initial_paint / 2
  let paint_after_first_day := initial_paint - first_day_usage
  let added_paint := 20
  let new_total_paint := paint_after_first_day + added_paint
  let second_day_usage := new_total_paint / 2
  let paint_after_second_day := new_total_paint - second_day_usage
  paint_after_second_day = 30 :=
by
  sorry

end paint_amount_third_day_l435_43555


namespace part1_part2_l435_43567

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

theorem part1 (x : ℝ) : f x 1 >= f x 1 := sorry

theorem part2 (a b : ℝ) (h : ∀ x > 0, f x a ≤ b - a) : b / a ≥ 0 := sorry

end part1_part2_l435_43567


namespace wrapping_paper_each_present_l435_43527

theorem wrapping_paper_each_present (total_paper : ℚ) (num_presents : ℕ)
  (h1 : total_paper = 1 / 2) (h2 : num_presents = 5) :
  (total_paper / num_presents = 1 / 10) :=
by
  sorry

end wrapping_paper_each_present_l435_43527


namespace sum_primes_upto_20_l435_43540

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l435_43540


namespace evaluate_polynomial_at_3_l435_43551

def f (x : ℕ) : ℕ := 3 * x^7 + 2 * x^5 + 4 * x^3 + x

theorem evaluate_polynomial_at_3 : f 3 = 7158 := by
  sorry

end evaluate_polynomial_at_3_l435_43551


namespace deposits_exceed_10_on_second_Tuesday_l435_43590

noncomputable def deposits_exceed_10 (n : ℕ) : ℕ :=
2 * (2^n - 1)

theorem deposits_exceed_10_on_second_Tuesday :
  ∃ n, deposits_exceed_10 n > 1000 ∧ 1 + (n - 1) % 7 = 2 ∧ n < 21 :=
sorry

end deposits_exceed_10_on_second_Tuesday_l435_43590


namespace proof_expression_C_equals_negative_one_l435_43549

def A : ℤ := abs (-1)
def B : ℤ := -(-1)
def C : ℤ := -(1^2)
def D : ℤ := (-1)^2

theorem proof_expression_C_equals_negative_one : C = -1 :=
by 
  sorry

end proof_expression_C_equals_negative_one_l435_43549


namespace twice_joan_more_than_karl_l435_43559

-- Define the conditions
def J : ℕ := 158
def total : ℕ := 400
def K : ℕ := total - J

-- Define the theorem to be proven
theorem twice_joan_more_than_karl :
  2 * J - K = 74 := by
    -- Skip the proof steps using 'sorry'
    sorry

end twice_joan_more_than_karl_l435_43559


namespace proportion_equation_l435_43505

theorem proportion_equation (x y : ℝ) (h : 3 * x = 4 * y) (hy : y ≠ 0) : (x / 4 = y / 3) :=
by
  sorry

end proportion_equation_l435_43505


namespace barbara_needs_more_weeks_l435_43512

/-
  Problem Statement:
  Barbara wants to save up for a new wristwatch that costs $100. Her parents give her an allowance
  of $5 a week and she can either save it all up or spend it as she wishes. 10 weeks pass and
  due to spending some of her money, Barbara currently only has $20. How many more weeks does she need
  to save for a watch if she stops spending on other things right now?
-/

def wristwatch_cost : ℕ := 100
def allowance_per_week : ℕ := 5
def current_savings : ℕ := 20
def amount_needed : ℕ := wristwatch_cost - current_savings
def weeks_needed : ℕ := amount_needed / allowance_per_week

theorem barbara_needs_more_weeks :
  weeks_needed = 16 :=
by
  -- proof goes here
  sorry

end barbara_needs_more_weeks_l435_43512


namespace tunnel_length_is_correct_l435_43515

-- Define the conditions given in the problem
def length_of_train : ℕ := 90
def speed_of_train : ℕ := 160
def time_to_pass_tunnel : ℕ := 3

-- Define the length of the tunnel to be proven
def length_of_tunnel : ℕ := 480 - length_of_train

-- Define the statement to be proven
theorem tunnel_length_is_correct : length_of_tunnel = 390 := by
  sorry

end tunnel_length_is_correct_l435_43515


namespace measure_angle_F_l435_43591

theorem measure_angle_F :
  ∃ (F : ℝ), F = 18 ∧
  ∃ (D E : ℝ),
  D = 75 ∧
  E = 15 + 4 * F ∧
  D + E + F = 180 :=
by
  sorry

end measure_angle_F_l435_43591


namespace pet_store_total_birds_l435_43548

def total_birds_in_pet_store (bird_cages parrots_per_cage parakeets_per_cage : ℕ) : ℕ :=
  bird_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_total_birds :
  total_birds_in_pet_store 4 8 2 = 40 :=
by
  sorry

end pet_store_total_birds_l435_43548


namespace contrapositive_example_l435_43573

theorem contrapositive_example (a b : ℕ) (h : a = 0 → ab = 0) : ab ≠ 0 → a ≠ 0 :=
by sorry

end contrapositive_example_l435_43573


namespace cube_sum_divisible_by_six_l435_43519

theorem cube_sum_divisible_by_six
  (a b c : ℤ)
  (h1 : 6 ∣ (a^2 + b^2 + c^2))
  (h2 : 3 ∣ (a * b + b * c + c * a))
  : 6 ∣ (a^3 + b^3 + c^3) := 
sorry

end cube_sum_divisible_by_six_l435_43519


namespace construct_triangle_l435_43544

variable (h_a h_b h_c : ℝ)

noncomputable def triangle_exists_and_similar :=
  ∃ (a b c : ℝ), (a = h_b) ∧ (b = h_a) ∧ (c = h_a * h_b / h_c) ∧
  (∃ (area : ℝ), area = 1/2 * a * (h_a * h_c / h_b) ∧ area = 1/2 * b * (h_b * h_c / h_a) ∧ area = 1/2 * c * h_c)

theorem construct_triangle (h_a h_b h_c : ℝ) :
  ∃ a b c, a = h_b ∧ b = h_a ∧ c = h_a * h_b / h_c ∧
  ∃ area, area = 1/2 * a * (h_a * h_c / h_b) ∧ area = 1/2 * b * (h_b * h_c / h_a) ∧ area = 1/2 * c * h_c := 
  sorry

end construct_triangle_l435_43544


namespace complex_division_l435_43535

theorem complex_division (i : ℂ) (hi : i^2 = -1) : (2 * i) / (1 - i) = -1 + i :=
by sorry

end complex_division_l435_43535


namespace inequality_ge_one_l435_43579

open Nat

variable (p q : ℝ) (m n : ℕ)

def conditions := p ≥ 0 ∧ q ≥ 0 ∧ p + q = 1 ∧ m > 0 ∧ n > 0

theorem inequality_ge_one (h : conditions p q m n) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 := 
by sorry

end inequality_ge_one_l435_43579


namespace measure_exterior_angle_BAC_l435_43530

-- Define the interior angle of a regular nonagon
def nonagon_interior_angle := (180 * (9 - 2)) / 9

-- Define the exterior angle of the nonagon
def nonagon_exterior_angle := 360 - nonagon_interior_angle

-- The square's interior angle
def square_interior_angle := 90

-- The question to be proven
theorem measure_exterior_angle_BAC :
  nonagon_exterior_angle - square_interior_angle = 130 :=
  by
  sorry

end measure_exterior_angle_BAC_l435_43530


namespace second_tap_fills_in_15_hours_l435_43533

theorem second_tap_fills_in_15_hours 
  (r1 r3 : ℝ) 
  (x : ℝ) 
  (H1 : r1 = 1 / 10) 
  (H2 : r3 = 1 / 6) 
  (H3 : r1 + 1 / x + r3 = 1 / 3) : 
  x = 15 :=
sorry

end second_tap_fills_in_15_hours_l435_43533


namespace daily_sales_volume_relationship_maximize_daily_sales_profit_l435_43547

variables (x : ℝ) (y : ℝ) (P : ℝ)

-- Conditions
def cost_per_box : ℝ := 40
def min_selling_price : ℝ := 45
def initial_selling_price : ℝ := 45
def initial_sales_volume : ℝ := 700
def decrease_in_sales_volume_per_dollar : ℝ := 20

-- The functional relationship between y and x
theorem daily_sales_volume_relationship (hx : min_selling_price ≤ x ∧ x < 80) : y = -20 * x + 1600 := by
  sorry

-- The profit function
def profit_function (x : ℝ) := (x - cost_per_box) * (initial_sales_volume - decrease_in_sales_volume_per_dollar * (x - initial_selling_price))

-- Maximizing the profit
theorem maximize_daily_sales_profit : ∃ x_max, x_max = 60 ∧ P = profit_function 60 ∧ P = 8000 := by
  sorry

end daily_sales_volume_relationship_maximize_daily_sales_profit_l435_43547


namespace age_of_17th_student_is_75_l435_43569

variables (T A : ℕ)

def avg_17_students := 17
def avg_5_students := 14
def avg_9_students := 16
def total_17_students := 17 * avg_17_students
def total_5_students := 5 * avg_5_students
def total_9_students := 9 * avg_9_students
def age_17th_student : ℕ := total_17_students - (total_5_students + total_9_students)

theorem age_of_17th_student_is_75 :
  age_17th_student = 75 := by sorry

end age_of_17th_student_is_75_l435_43569


namespace remainder_7_pow_137_mod_11_l435_43577

theorem remainder_7_pow_137_mod_11 :
    (137 = 13 * 10 + 7) →
    (7^10 ≡ 1 [MOD 11]) →
    (7^137 ≡ 6 [MOD 11]) :=
by
  intros h1 h2
  sorry

end remainder_7_pow_137_mod_11_l435_43577


namespace trigonometric_identities_l435_43554

open Real

theorem trigonometric_identities :
  (cos 75 * cos 75 = (2 - sqrt 3) / 4) ∧
  ((1 + tan 105) / (1 - tan 105) ≠ sqrt 3 / 3) ∧
  (tan 1 + tan 44 + tan 1 * tan 44 = 1) ∧
  (sin 70 * (sqrt 3 / tan 40 - 1) ≠ 2) :=
by
  sorry

end trigonometric_identities_l435_43554


namespace apples_in_basket_l435_43541

theorem apples_in_basket
  (total_rotten : ℝ := 12 / 100)
  (total_spots : ℝ := 7 / 100)
  (total_insects : ℝ := 5 / 100)
  (total_varying_rot : ℝ := 3 / 100)
  (perfect_apples : ℝ := 66) :
  (perfect_apples / ((1 - (total_rotten + total_spots + total_insects + total_varying_rot))) = 90) :=
by
  sorry

end apples_in_basket_l435_43541


namespace value_of_x_for_real_y_l435_43532

theorem value_of_x_for_real_y (x y : ℝ) (h : 4 * y^2 + 2 * x * y + |x| + 8 = 0) :
  (x ≤ -10) ∨ (x ≥ 10) :=
sorry

end value_of_x_for_real_y_l435_43532


namespace no_real_roots_eq_xsq_abs_x_plus_1_eq_0_l435_43502

theorem no_real_roots_eq_xsq_abs_x_plus_1_eq_0 :
  ¬ ∃ x : ℝ, x^2 + abs x + 1 = 0 :=
by
  sorry

end no_real_roots_eq_xsq_abs_x_plus_1_eq_0_l435_43502


namespace range_of_k_l435_43584

theorem range_of_k (k : ℝ) : 
  (∀ x, x ∈ {x | -3 ≤ x ∧ x ≤ 2} ∩ {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1} ↔ x ∈ {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1}) →
   -1 ≤ k ∧ k ≤ 1 / 2 :=
by sorry

end range_of_k_l435_43584


namespace cube_side_length_l435_43517

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 / (6 * n^3) = 1 / 3) : n = 3 :=
by
  sorry

end cube_side_length_l435_43517


namespace acute_angle_probability_l435_43572

/-- 
  Given a clock with two hands (the hour and the minute hand) and assuming:
  1. The hour hand is always pointing at 12 o'clock.
  2. The angle between the hands is acute if the minute hand is either in the first quadrant 
     (between 12 and 3 o'clock) or in the fourth quadrant (between 9 and 12 o'clock).

  Prove that the probability that the angle between the hands is acute is 1/2.
-/
theorem acute_angle_probability : 
  let total_intervals := 12
  let favorable_intervals := 6
  (favorable_intervals / total_intervals : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end acute_angle_probability_l435_43572


namespace largest_divisor_l435_43558

theorem largest_divisor (x : ℤ) (hx : x % 2 = 1) : 180 ∣ (15 * x + 3) * (15 * x + 9) * (10 * x + 5) := 
by
  sorry

end largest_divisor_l435_43558


namespace find_a4_b4_l435_43503

theorem find_a4_b4 :
  ∃ (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ),
    a₁ * b₁ + a₂ * b₃ = 1 ∧
    a₁ * b₂ + a₂ * b₄ = 0 ∧
    a₃ * b₁ + a₄ * b₃ = 0 ∧
    a₃ * b₂ + a₄ * b₄ = 1 ∧
    a₂ * b₃ = 7 ∧
    a₄ * b₄ = -6 :=
by
  sorry

end find_a4_b4_l435_43503


namespace scientific_notation_100000_l435_43571

theorem scientific_notation_100000 : ∃ a n, (1 ≤ a) ∧ (a < 10) ∧ (100000 = a * 10 ^ n) :=
by
  use 1, 5
  repeat { split }
  repeat { sorry }

end scientific_notation_100000_l435_43571


namespace bobs_share_l435_43588

theorem bobs_share 
  (r : ℕ → ℕ → ℕ → Prop) (s : ℕ) 
  (h_ratio : r 1 2 3) 
  (bill_share : s = 300) 
  (hr : ∃ p, s = 2 * p) :
  ∃ b, b = 3 * (s / 2) ∧ b = 450 := 
by
  sorry

end bobs_share_l435_43588


namespace deluxe_stereo_time_fraction_l435_43574

theorem deluxe_stereo_time_fraction (S : ℕ) (B : ℝ)
  (H1 : 2 / 3 > 0)
  (H2 : 1.6 > 0) :
  (1.6 / 3 * S * B) / (1.2 * S * B) = 4 / 9 :=
by
  sorry

end deluxe_stereo_time_fraction_l435_43574


namespace charlie_older_than_bobby_by_three_l435_43542

variable (J C B x : ℕ)

def jenny_older_charlie_by_five (J C : ℕ) := J = C + 5
def charlie_age_when_jenny_twice_bobby_age (C x : ℕ) := C + x = 11
def jenny_twice_bobby (J B x : ℕ) := J + x = 2 * (B + x)

theorem charlie_older_than_bobby_by_three
  (h1 : jenny_older_charlie_by_five J C)
  (h2 : charlie_age_when_jenny_twice_bobby_age C x)
  (h3 : jenny_twice_bobby J B x) :
  (C = B + 3) :=
by
  sorry

end charlie_older_than_bobby_by_three_l435_43542


namespace rational_t_l435_43539

variable (A B t : ℚ)

theorem rational_t (A B : ℚ) (hA : A = 2 * t / (1 + t^2)) (hB : B = (1 - t^2) / (1 + t^2)) : ∃ t' : ℚ, t = t' :=
by
  sorry

end rational_t_l435_43539


namespace trig_expression_value_l435_43565

theorem trig_expression_value (θ : ℝ) (h : Real.tan θ = 2) : 
  (2 * Real.cos θ) / (Real.sin (Real.pi / 2 + θ) + Real.sin (Real.pi + θ)) = -2 := 
by 
  sorry

end trig_expression_value_l435_43565


namespace molecular_weight_correct_l435_43538

-- Define atomic weights
def atomic_weight_aluminium : Float := 26.98
def atomic_weight_oxygen : Float := 16.00
def atomic_weight_hydrogen : Float := 1.01
def atomic_weight_silicon : Float := 28.09
def atomic_weight_nitrogen : Float := 14.01

-- Define the number of each atom in the compound
def num_aluminium : Nat := 2
def num_oxygen : Nat := 6
def num_hydrogen : Nat := 3
def num_silicon : Nat := 2
def num_nitrogen : Nat := 4

-- Calculate the expected molecular weight
def expected_molecular_weight : Float :=
  (2 * atomic_weight_aluminium) + 
  (6 * atomic_weight_oxygen) + 
  (3 * atomic_weight_hydrogen) + 
  (2 * atomic_weight_silicon) + 
  (4 * atomic_weight_nitrogen)

-- Prove that the expected molecular weight is 265.21 amu
theorem molecular_weight_correct : expected_molecular_weight = 265.21 :=
by
  sorry

end molecular_weight_correct_l435_43538
