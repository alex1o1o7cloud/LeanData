import Mathlib

namespace prime_quadruple_solution_l100_100044

-- Define the problem statement in Lean
theorem prime_quadruple_solution :
  ∀ (p q r : ℕ) (n : ℕ),
    Prime p → Prime q → Prime r → n > 0 →
    p^2 = q^2 + r^n →
    (p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4) :=
by
  sorry -- Proof omitted

end prime_quadruple_solution_l100_100044


namespace widgets_per_week_l100_100840

theorem widgets_per_week 
  (widgets_per_hour : ℕ) 
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (h1 : widgets_per_hour = 20) 
  (h2 : hours_per_day = 8) 
  (h3 : days_per_week = 5) :
  widgets_per_hour * hours_per_day * days_per_week = 800 :=
by
  rw [h1, h2, h3]
  exact rfl

end widgets_per_week_l100_100840


namespace plane_arrival_time_l100_100899

-- Define the conditions
def departure_time := 11 -- common departure time in hours (11:00)
def bus_speed := 100 -- bus speed in km/h
def train_speed := 300 -- train speed in km/h
def plane_speed := 900 -- plane speed in km/h
def bus_arrival := 20 -- bus arrival time in hours (20:00)
def train_arrival := 14 -- train arrival time in hours (14:00)

-- Given these conditions, we need to prove the plane arrival time
theorem plane_arrival_time : (departure_time + (900 / plane_speed)) = 12 := by
  sorry

end plane_arrival_time_l100_100899


namespace cube_fit_count_cube_volume_percentage_l100_100032

-- Definitions based on the conditions in the problem.
def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_side : ℕ := 4

-- Definitions for the calculated values.
def num_cubes_length : ℕ := box_length / cube_side
def num_cubes_width : ℕ := box_width / cube_side
def num_cubes_height : ℕ := box_height / cube_side

def total_cubes : ℕ := num_cubes_length * num_cubes_width * num_cubes_height

def volume_cube : ℕ := cube_side^3
def volume_cubes_total : ℕ := total_cubes * volume_cube
def volume_box : ℕ := box_length * box_width * box_height

def percentage_volume : ℕ := (volume_cubes_total * 100) / volume_box

-- The proof statements.
theorem cube_fit_count : total_cubes = 6 := by
  sorry

theorem cube_volume_percentage : percentage_volume = 100 := by
  sorry

end cube_fit_count_cube_volume_percentage_l100_100032


namespace M_geq_N_l100_100048

variable (a b : ℝ)

def M : ℝ := a^2 + 12 * a - 4 * b
def N : ℝ := 4 * a - 20 - b^2

theorem M_geq_N : M a b ≥ N a b := by
  sorry

end M_geq_N_l100_100048


namespace a100_gt_two_pow_99_l100_100217

theorem a100_gt_two_pow_99 (a : ℕ → ℤ) (h_pos : ∀ n, 0 < a n) 
  (h1 : a 1 > a 0) (h_rec : ∀ n ≥ 2, a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2 ^ 99 :=
sorry

end a100_gt_two_pow_99_l100_100217


namespace line_equation_l100_100864

-- Given conditions
def param_x (t : ℝ) : ℝ := 3 * t + 6
def param_y (t : ℝ) : ℝ := 5 * t - 7

-- Proof problem: for any real t, the parameterized line can be described by the equation y = 5x/3 - 17.
theorem line_equation (t : ℝ) : ∃ (m b : ℝ), (∃ t : ℝ, param_y t = m * (param_x t) + b) ∧ m = 5 / 3 ∧ b = -17 :=
by
  exists 5 / 3
  exists -17
  sorry

end line_equation_l100_100864


namespace find_m_l100_100663

variables (m : ℝ)
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-1, m)
def c : ℝ × ℝ := (-1, 2)

-- Define the property of vector parallelism in ℝ.
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Statement to be proven
theorem find_m :
    parallel (1, m - 1) c →
    m = -1 :=
by
  sorry

end find_m_l100_100663


namespace integers_less_than_2019_divisible_by_18_or_21_but_not_both_l100_100667

theorem integers_less_than_2019_divisible_by_18_or_21_but_not_both :
  ∃ (N : ℕ), (∀ (n : ℕ), (n < 2019 → (n % 18 = 0 ∨ n % 21 = 0) → n % (18 * 21 / gcd 18 21) ≠ 0) ↔ (∀ (m : ℕ), m < N)) ∧ N = 176 :=
by
  sorry

end integers_less_than_2019_divisible_by_18_or_21_but_not_both_l100_100667


namespace pizza_toppings_l100_100775

theorem pizza_toppings (n : ℕ) (h : n = 7) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 63 :=
by
  rw h
  simp
  sorry -- Placeholder to complete the proof

end pizza_toppings_l100_100775


namespace a2b_etc_ge_9a2b2c2_l100_100360

theorem a2b_etc_ge_9a2b2c2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 :=
by
  sorry

end a2b_etc_ge_9a2b2c2_l100_100360


namespace sum_of_solutions_l100_100070

theorem sum_of_solutions (y : ℝ) (h : y^2 = 25) : ∃ (a b : ℝ), (a = 5 ∨ a = -5) ∧ (b = 5 ∨ b = -5) ∧ a + b = 0 :=
sorry

end sum_of_solutions_l100_100070


namespace sara_gave_dan_pears_l100_100119

theorem sara_gave_dan_pears :
  ∀ (original_pears left_pears given_to_dan : ℕ),
    original_pears = 35 →
    left_pears = 7 →
    given_to_dan = original_pears - left_pears →
    given_to_dan = 28 :=
by
  intros original_pears left_pears given_to_dan h_original h_left h_given
  rw [h_original, h_left] at h_given
  exact h_given

end sara_gave_dan_pears_l100_100119


namespace factor_81_minus_36x4_l100_100523

theorem factor_81_minus_36x4 (x : ℝ) : 
    81 - 36 * x^4 = 9 * (Real.sqrt 3 - Real.sqrt 2 * x) * (Real.sqrt 3 + Real.sqrt 2 * x) * (3 + 2 * x^2) :=
sorry

end factor_81_minus_36x4_l100_100523


namespace recurring_decimal_sum_l100_100345

theorem recurring_decimal_sum (x y : ℚ) (hx : x = 4/9) (hy : y = 7/9) :
  x + y = 11/9 :=
by
  rw [hx, hy]
  exact sorry

end recurring_decimal_sum_l100_100345


namespace base_10_uniqueness_l100_100962

theorem base_10_uniqueness : 
  (∀ a : ℕ, 12 = 3 * 4 ∧ 56 = 7 * 8 ↔ (a * b + (a + 1) = (a + 2) * (a + 3))) → b = 10 :=
sorry

end base_10_uniqueness_l100_100962


namespace younger_person_age_l100_100127

/-- Let E be the present age of the elder person and Y be the present age of the younger person.
Given the conditions :
1) E - Y = 20
2) E - 15 = 2 * (Y - 15)
Prove that Y = 35. -/
theorem younger_person_age (E Y : ℕ) 
  (h1 : E - Y = 20) 
  (h2 : E - 15 = 2 * (Y - 15)) : 
  Y = 35 :=
sorry

end younger_person_age_l100_100127


namespace repeating_decimal_sum_l100_100346

noncomputable def repeating_decimal_four : ℚ := 0.44444 -- 0.\overline{4}
noncomputable def repeating_decimal_seven : ℚ := 0.77777 -- 0.\overline{7}

-- Proving that the sum of these repeating decimals is equivalent to the fraction 11/9.
theorem repeating_decimal_sum : repeating_decimal_four + repeating_decimal_seven = 11/9 := by
  -- Placeholder to skip the actual proof
  sorry

end repeating_decimal_sum_l100_100346


namespace decreasing_function_l100_100942

noncomputable def f (a x : ℝ) : ℝ := a^(1 - x)

theorem decreasing_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : ∀ x > 1, f a x < 1) :
  ∀ x y : ℝ, x < y → f a x > f a y :=
sorry

end decreasing_function_l100_100942


namespace adjusted_volume_bowling_ball_l100_100897

noncomputable def bowling_ball_diameter : ℝ := 40
noncomputable def hole1_diameter : ℝ := 5
noncomputable def hole1_depth : ℝ := 10
noncomputable def hole2_diameter : ℝ := 4
noncomputable def hole2_depth : ℝ := 12
noncomputable def expected_adjusted_volume : ℝ := 10556.17 * Real.pi

theorem adjusted_volume_bowling_ball :
  let radius := bowling_ball_diameter / 2
  let volume_ball := (4 / 3) * Real.pi * radius^3
  let hole1_radius := hole1_diameter / 2
  let hole1_volume := Real.pi * hole1_radius^2 * hole1_depth
  let hole2_radius := hole2_diameter / 2
  let hole2_volume := Real.pi * hole2_radius^2 * hole2_depth
  let adjusted_volume := volume_ball - hole1_volume - hole2_volume
  adjusted_volume = expected_adjusted_volume :=
by
  sorry

end adjusted_volume_bowling_ball_l100_100897


namespace fraction_equivalence_l100_100332

theorem fraction_equivalence (x y : ℝ) (h : x ≠ y) :
  (x - y)^2 / (x^2 - y^2) = (x - y) / (x + y) :=
by
  sorry

end fraction_equivalence_l100_100332


namespace tan_alpha_eq_2_l100_100213

theorem tan_alpha_eq_2 (α : Real) (h : Real.tan α = 2) : 
  1 / (Real.sin (2 * α) + Real.cos (α) ^ 2) = 1 := 
by 
  sorry

end tan_alpha_eq_2_l100_100213


namespace tunnel_build_equation_l100_100766

theorem tunnel_build_equation (x : ℝ) (h1 : 1280 > 0) (h2 : x > 0) : 
  (1280 - x) / x = (1280 - x) / (1.4 * x) + 2 := 
by
  sorry

end tunnel_build_equation_l100_100766


namespace max_value_expression_l100_100596

theorem max_value_expression (s : ℝ) : 
  ∃ M, M = -3 * s^2 + 36 * s + 7 ∧ (∀ t : ℝ, -3 * t^2 + 36 * t + 7 ≤ M) :=
by
  use 115
  sorry

end max_value_expression_l100_100596


namespace executive_board_elections_l100_100624

noncomputable def num_candidates : ℕ := 18
noncomputable def num_positions : ℕ := 6
noncomputable def num_former_board_members : ℕ := 8

noncomputable def total_selections := Nat.choose num_candidates num_positions
noncomputable def no_former_board_members_selections := Nat.choose (num_candidates - num_former_board_members) num_positions

noncomputable def valid_selections := total_selections - no_former_board_members_selections

theorem executive_board_elections : valid_selections = 18354 :=
by sorry

end executive_board_elections_l100_100624


namespace geometric_sequence_sum_l100_100723

theorem geometric_sequence_sum :
  ∀ (a : ℕ → ℝ) (q : ℝ),
    a 1 * q + a 1 * q ^ 3 = 20 →
    a 1 * q ^ 2 + a 1 * q ^ 4 = 40 →
    a 1 * q ^ 4 + a 1 * q ^ 6 = 160 :=
by
  sorry

end geometric_sequence_sum_l100_100723


namespace calories_per_person_l100_100705

theorem calories_per_person 
  (oranges : ℕ)
  (pieces_per_orange : ℕ)
  (people : ℕ)
  (calories_per_orange : ℝ)
  (h_oranges : oranges = 7)
  (h_pieces_per_orange : pieces_per_orange = 12)
  (h_people : people = 6)
  (h_calories_per_orange : calories_per_orange = 80.0) :
  (oranges * pieces_per_orange / people) * (calories_per_orange / pieces_per_orange) = 93.3338 :=
by
  sorry

end calories_per_person_l100_100705


namespace simplify_polynomial_l100_100421

theorem simplify_polynomial :
  (2 * x * (4 * x ^ 3 - 3 * x + 1) - 4 * (2 * x ^ 3 - x ^ 2 + 3 * x - 5)) =
  8 * x ^ 4 - 8 * x ^ 3 - 2 * x ^ 2 - 10 * x + 20 :=
by
  sorry

end simplify_polynomial_l100_100421


namespace compute_expression_l100_100030

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 :=
by
  sorry

end compute_expression_l100_100030


namespace expand_expression_l100_100202

theorem expand_expression (x : ℝ) : (7 * x + 5) * (3 * x^2) = 21 * x^3 + 15 * x^2 :=
by
  sorry

end expand_expression_l100_100202


namespace MitchWorks25Hours_l100_100114

noncomputable def MitchWorksHours : Prop :=
  let weekday_earnings_rate := 3
  let weekend_earnings_rate := 6
  let weekly_earnings := 111
  let weekend_hours := 6
  let weekday_hours (x : ℕ) := 5 * x
  let weekend_earnings := weekend_hours * weekend_earnings_rate
  let weekday_earnings (x : ℕ) := x * weekday_earnings_rate
  let total_weekday_earnings (x : ℕ) := weekly_earnings - weekend_earnings
  ∀ (x : ℕ), weekday_earnings x = total_weekday_earnings x → x = 25

theorem MitchWorks25Hours : MitchWorksHours := by
  sorry

end MitchWorks25Hours_l100_100114


namespace bus_speed_excluding_stoppages_l100_100927

-- Define the conditions
def speed_including_stoppages : ℝ := 48 -- km/hr
def stoppage_time_per_hour : ℝ := 15 / 60 -- 15 minutes is 15/60 hours

-- The main theorem stating what we need to prove
theorem bus_speed_excluding_stoppages : ∃ v : ℝ, (v * (1 - stoppage_time_per_hour) = speed_including_stoppages) ∧ v = 64 :=
begin
  sorry,
end

end bus_speed_excluding_stoppages_l100_100927


namespace prime_pairs_summing_to_50_count_l100_100670

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of primes less than 25
def primes_less_than_25 : Set ℕ := { p | is_prime p ∧ p < 25 }

-- Define the condition that a pair of primes sums to 50
def sums_to_50 (a b : ℕ) : Prop := a + b = 50 ∧ a ≠ b

-- Define the set of pairs of primes that sum to 50
def prime_pairs_summing_to_50 : Set (ℕ × ℕ) :=
  { (a, b) | a ∈ primes_less_than_25 ∧ b ∈ primes_less_than_25 ∧ sums_to_50 a b }

-- Prove that there are exactly 4 unordered pairs of primes that sum to 50
theorem prime_pairs_summing_to_50_count : 
  (Set.card prime_pairs_summing_to_50) / 2 = 4 :=
by
  sorry

end prime_pairs_summing_to_50_count_l100_100670


namespace geese_population_1996_l100_100999

theorem geese_population_1996 (k x : ℝ) 
  (h1 : x - 39 = k * 60) 
  (h2 : 123 - 60 = k * x) : 
  x = 84 := 
by
  sorry

end geese_population_1996_l100_100999


namespace price_difference_l100_100909

noncomputable def originalPriceStrawberries (s : ℝ) (sale_revenue_s : ℝ) := sale_revenue_s / (0.70 * s)
noncomputable def originalPriceBlueberries (b : ℝ) (sale_revenue_b : ℝ) := sale_revenue_b / (0.80 * b)

theorem price_difference
    (s : ℝ) (sale_revenue_s : ℝ)
    (b : ℝ) (sale_revenue_b : ℝ)
    (h1 : sale_revenue_s = 70 * (0.70 * s))
    (h2 : sale_revenue_b = 50 * (0.80 * b)) :
    originalPriceStrawberries (sale_revenue_s / 49) sale_revenue_s - originalPriceBlueberries (sale_revenue_b / 40) sale_revenue_b = 0.71 :=
by
  sorry

end price_difference_l100_100909


namespace intersection_of_A_and_B_l100_100539

-- Define the set A
def A : Set ℝ := {-1, 0, 1}

-- Define the set B based on the given conditions
def B : Set ℝ := {y | ∃ x ∈ A, y = Real.cos (Real.pi * x)}

-- The main theorem to prove that A ∩ B is {-1, 1}
theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by
  sorry

end intersection_of_A_and_B_l100_100539


namespace option_A_correct_l100_100750

theorem option_A_correct (y x : ℝ) : y * x - 2 * (x * y) = - (x * y) :=
by
  sorry

end option_A_correct_l100_100750


namespace lucy_last_10_shots_l100_100329

variable (shots_30 : ℕ) (percentage_30 : ℚ) (total_shots : ℕ) (percentage_40 : ℚ)
variable (shots_made_30 : ℕ) (shots_made_40 : ℕ) (shots_made_last_10 : ℕ)

theorem lucy_last_10_shots 
    (h1 : shots_30 = 30) 
    (h2 : percentage_30 = 0.60) 
    (h3 : total_shots = 40) 
    (h4 : percentage_40 = 0.62 )
    (h5 : shots_made_30 = Nat.floor (percentage_30 * shots_30)) 
    (h6 : shots_made_40 = Nat.floor (percentage_40 * total_shots))
    (h7 : shots_made_last_10 = shots_made_40 - shots_made_30) 
    : shots_made_last_10 = 7 := sorry

end lucy_last_10_shots_l100_100329


namespace Luke_trips_l100_100716

variable (carries : Nat) (table1 : Nat) (table2 : Nat)

theorem Luke_trips (h1 : carries = 4) (h2 : table1 = 20) (h3 : table2 = 16) : 
  (table1 / carries + table2 / carries) = 9 :=
by
  sorry

end Luke_trips_l100_100716


namespace borrowing_period_l100_100772

theorem borrowing_period 
  (principal : ℕ) (rate_1 : ℕ) (rate_2 : ℕ) (gain : ℕ)
  (h1 : principal = 5000)
  (h2 : rate_1 = 4)
  (h3 : rate_2 = 8)
  (h4 : gain = 200)
  : ∃ n : ℕ, n = 1 :=
by
  sorry

end borrowing_period_l100_100772


namespace pencils_per_row_l100_100203

-- Definitions of conditions.
def num_pencils : ℕ := 35
def num_rows : ℕ := 7

-- Hypothesis: given the conditions, prove the number of pencils per row.
theorem pencils_per_row : num_pencils / num_rows = 5 := 
  by 
  -- Proof steps go here, but are replaced by sorry.
  sorry

end pencils_per_row_l100_100203


namespace exists_ints_a_b_l100_100047

theorem exists_ints_a_b (n : ℤ) (h : n % 4 ≠ 2) : ∃ a b : ℤ, n + a^2 = b^2 :=
by
  sorry

end exists_ints_a_b_l100_100047


namespace money_last_weeks_l100_100095

-- Conditions
def money_from_mowing : ℕ := 14
def money_from_weeding : ℕ := 31
def weekly_spending : ℕ := 5

-- Total money made
def total_money : ℕ := money_from_mowing + money_from_weeding

-- Expected result
def expected_weeks : ℕ := 9

-- Prove the number of weeks the money will last Jerry
theorem money_last_weeks : (total_money / weekly_spending) = expected_weeks :=
by
  sorry

end money_last_weeks_l100_100095


namespace top_card_is_club_probability_l100_100017

-- Definitions based on the conditions
def deck_size := 52
def suit_count := 4
def cards_per_suit := deck_size / suit_count

-- The question we want to prove
theorem top_card_is_club_probability :
  (13 : ℝ) / (52 : ℝ) = 1 / 4 :=
by 
  sorry

end top_card_is_club_probability_l100_100017


namespace solution_system_of_equations_l100_100738

theorem solution_system_of_equations : 
  ∃ (x y : ℝ), (2 * x - y = 3 ∧ x + y = 3) ∧ (x = 2 ∧ y = 1) := 
by
  sorry

end solution_system_of_equations_l100_100738


namespace fractional_exponent_calculation_l100_100025

variables (a b : ℝ) -- Define a and b as real numbers
variable (ha : a > 0) -- Condition a > 0
variable (hb : b > 0) -- Condition b > 0

theorem fractional_exponent_calculation :
  (a^(2 * b^(1/4)) / (a * b^(1/2))^(1/2)) = a^(1/2) :=
by
  sorry -- Proof is not required, skip with sorry

end fractional_exponent_calculation_l100_100025


namespace ratio_pow_eq_l100_100485

variable (a b c d e f p q r : ℝ)
variable (n : ℕ)
variable (h : a / b = c / d)
variable (h1 : a / b = e / f)
variable (h2 : p ≠ 0 ∨ q ≠ 0 ∨ r ≠ 0)

theorem ratio_pow_eq
  (h : a / b = c / d)
  (h1 : a / b = e / f)
  (h2 : p ≠ 0 ∨ q ≠ 0 ∨ r ≠ 0)
  (n_ne_zero : n ≠ 0):
  (a / b) ^ n = (p * a ^ n + q * c ^ n + r * e ^ n) / (p * b ^ n + q * d ^ n + r * f ^ n) :=
by
  sorry

end ratio_pow_eq_l100_100485


namespace total_selection_ways_l100_100947

-- Defining the conditions
def groupA_male_students : ℕ := 5
def groupA_female_students : ℕ := 3
def groupB_male_students : ℕ := 6
def groupB_female_students : ℕ := 2

-- Define combinations (choose function)
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- The required theorem statement
theorem total_selection_ways :
  C groupA_female_students 1 * C groupA_male_students 1 * C groupB_male_students 2 +
  C groupB_female_students 1 * C groupB_male_students 1 * C groupA_male_students 2 = 345 :=
by
  sorry

end total_selection_ways_l100_100947


namespace cardboard_box_height_l100_100110

theorem cardboard_box_height :
  ∃ (x : ℕ), x ≥ 0 ∧ 10 * x^2 + 4 * x ≥ 130 ∧ (2 * x + 1) = 9 :=
sorry

end cardboard_box_height_l100_100110


namespace traffic_lights_states_l100_100960

theorem traffic_lights_states (n k : ℕ) : 
  (k ≤ n) → 
  (∃ (ways : ℕ), ways = 3^k * 2^(n - k)) :=
by
  sorry

end traffic_lights_states_l100_100960


namespace complex_division_example_l100_100226

theorem complex_division_example (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + i) = (1/2 : ℂ) - (3/2 : ℂ) * i :=
by
  -- proof would go here
  sorry

end complex_division_example_l100_100226


namespace geometric_sequence_sum_inequality_l100_100057

theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h_geom : ∀ k, a (k + 1) = a k * q)
  (h_pos : ∀ k ≤ 7, a k > 0)
  (h_q_ne_one : q ≠ 1) :
  a 0 + a 7 > a 3 + a 4 :=
sorry

end geometric_sequence_sum_inequality_l100_100057


namespace number_of_valid_numbers_l100_100998

-- Define a function that checks if a number is composed of digits from the set {1, 2, 3}
def composed_of_123 (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 1 ∨ d = 2 ∨ d = 3

-- Define a predicate for a number being less than 200,000
def less_than_200000 (n : ℕ) : Prop := n < 200000

-- Define a predicate for a number being divisible by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- The main theorem statement
theorem number_of_valid_numbers : ∃ (count : ℕ), count = 202 ∧ 
  (∀ (n : ℕ), less_than_200000 n → composed_of_123 n → divisible_by_3 n → n < count) :=
sorry

end number_of_valid_numbers_l100_100998


namespace number_of_boys_l100_100986

-- Definitions from the problem conditions
def trees : ℕ := 29
def trees_per_boy : ℕ := 3

-- Prove the number of boys is 10
theorem number_of_boys : (trees / trees_per_boy) + 1 = 10 :=
by sorry

end number_of_boys_l100_100986


namespace parallelogram_area_l100_100786

open Matrix

noncomputable def vector3 := (ℝ × ℝ × ℝ)
noncomputable def vec_sub (a b : vector3) : vector3 :=
  (a.1 - b.1, a.2 - b.2, a.3 - b.3)

noncomputable def cross_product (a b : vector3) : vector3 :=
  ((a.2 * b.3 - a.3 * b.2), (a.3 * b.1 - a.1 * b.3), (a.1 * b.2 - a.2 * b.1))

noncomputable def magnitude (v : vector3) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def P : vector3 := (2, -5, 3)
noncomputable def Q : vector3 := (4, -9, 6)
noncomputable def R : vector3 := (1, -4, 1)
noncomputable def S : vector3 := (3, -8, 4)

theorem parallelogram_area :
  let pq := vec_sub Q P,
      sr := vec_sub S R,
      rp := vec_sub R P,
      cross := cross_product pq rp in
  (pq = sr) ∧ (magnitude cross = real.sqrt 62) :=
by
  sorry

end parallelogram_area_l100_100786


namespace length_RS_14_l100_100873

-- Definitions of conditions
def edges : List ℕ := [8, 14, 19, 28, 37, 42]
def PQ_length : ℕ := 42

-- Problem statement
theorem length_RS_14 (edges : List ℕ) (PQ_length : ℕ) (h : PQ_length = 42) (h_edges : edges = [8, 14, 19, 28, 37, 42]) :
  ∃ RS_length : ℕ, RS_length ∈ edges ∧ RS_length = 14 :=
by
  sorry

end length_RS_14_l100_100873


namespace equation_of_W_rectangle_perimeter_greater_than_3sqrt3_l100_100086

theorem equation_of_W (P : ℝ × ℝ) :
  let x := P.1 in let y := P.2 in
  |y| = real.sqrt (x^2 + (y - 1/2)^2) ↔ y = x^2 + 1/4 :=
by sorry

theorem rectangle_perimeter_greater_than_3sqrt3 {A B C D : ℝ × ℝ}
  (hA : A.2 = A.1^2 + 1/4) (hB : B.2 = B.1^2 + 1/4) (hC : C.2 = C.1^2 + 1/4)
  (hAB_perp_BC : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) :
  2 * ((real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) + (real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2))) > 3 * real.sqrt 3 :=
by sorry

end equation_of_W_rectangle_perimeter_greater_than_3sqrt3_l100_100086


namespace factor_expression_l100_100348

theorem factor_expression (x : ℕ) : 63 * x + 54 = 9 * (7 * x + 6) :=
by
  sorry

end factor_expression_l100_100348


namespace min_value_on_top_layer_l100_100607

-- Definitions reflecting conditions
def bottom_layer : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def block_value (layer : List ℕ) (i : ℕ) : ℕ :=
  layer.getD (i-1) 0 -- assuming 1-based indexing

def second_layer_values : List ℕ :=
  [block_value bottom_layer 1 + block_value bottom_layer 2 + block_value bottom_layer 3,
   block_value bottom_layer 2 + block_value bottom_layer 3 + block_value bottom_layer 4,
   block_value bottom_layer 4 + block_value bottom_layer 5 + block_value bottom_layer 6,
   block_value bottom_layer 5 + block_value bottom_layer 6 + block_value bottom_layer 7,
   block_value bottom_layer 7 + block_value bottom_layer 8 + block_value bottom_layer 9,
   block_value bottom_layer 8 + block_value bottom_layer 9 + block_value bottom_layer 10]

def third_layer_values : List ℕ :=
  [second_layer_values.getD 0 0 + second_layer_values.getD 1 0 + second_layer_values.getD 2 0,
   second_layer_values.getD 1 0 + second_layer_values.getD 2 0 + second_layer_values.getD 3 0,
   second_layer_values.getD 3 0 + second_layer_values.getD 4 0 + second_layer_values.getD 5 0]

def top_layer_value : ℕ :=
  third_layer_values.getD 0 0 + third_layer_values.getD 1 0 + third_layer_values.getD 2 0

theorem min_value_on_top_layer : top_layer_value = 114 :=
by
  have h0 := block_value bottom_layer 1 -- intentionally leaving this incomplete as we're skipping the actual proof
  sorry

end min_value_on_top_layer_l100_100607


namespace volume_of_water_flowing_per_minute_l100_100603

variable (d w r : ℝ) (V : ℝ)

theorem volume_of_water_flowing_per_minute (h1 : d = 3) 
                                           (h2 : w = 32) 
                                           (h3 : r = 33.33) : 
  V = 3199.68 :=
by
  sorry

end volume_of_water_flowing_per_minute_l100_100603


namespace probability_denis_oleg_play_l100_100022

theorem probability_denis_oleg_play (n : ℕ) (h_n : n = 26) :
  (1 : ℚ) / 13 = 
  let total_matches : ℕ := n - 1 in
  let num_pairs := n * (n - 1) / 2 in
  (total_matches : ℚ) / num_pairs :=
by
  -- You can provide a proof here if necessary
  sorry

end probability_denis_oleg_play_l100_100022


namespace find_some_value_l100_100316

theorem find_some_value (m n k : ℝ)
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + k = (n + 18) / 6 - 2 / 5) : 
  k = 3 :=
sorry

end find_some_value_l100_100316


namespace complex_number_problem_l100_100053

theorem complex_number_problem (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i :=
by {
  -- provide proof here
  sorry
}

end complex_number_problem_l100_100053


namespace units_digit_27_64_l100_100358

/-- 
  Given that the units digit of 27 is 7, 
  and the units digit of 64 is 4, 
  prove that the units digit of 27 * 64 is 8.
-/
theorem units_digit_27_64 : 
  ∀ (n m : ℕ), 
  (n % 10 = 7) → 
  (m % 10 = 4) → 
  ((n * m) % 10 = 8) :=
by
  intros n m h1 h2
  -- Utilize modular arithmetic properties
  sorry

end units_digit_27_64_l100_100358


namespace find_a_l100_100799

variable (x y a : ℝ)

theorem find_a (h1 : (a * x + 8 * y) / (x - 2 * y) = 29) (h2 : x / (2 * y) = 3 / 2) : a = 7 :=
sorry

end find_a_l100_100799


namespace repeating_decimal_eq_l100_100469

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l100_100469


namespace remainders_identical_l100_100720

theorem remainders_identical (a b : ℕ) (h1 : a > b) :
  ∃ r₁ r₂ q₁ q₂ : ℕ, 
  a = (a - b) * q₁ + r₁ ∧ 
  b = (a - b) * q₂ + r₂ ∧ 
  r₁ = r₂ := by 
sorry

end remainders_identical_l100_100720


namespace tank_full_capacity_l100_100545

variable (T : ℝ) -- Define T as a real number representing the total capacity of the tank.

-- The main condition: (3 / 4) * T + 5 = (7 / 8) * T
axiom condition : (3 / 4) * T + 5 = (7 / 8) * T

-- Proof statement: Prove that T = 40
theorem tank_full_capacity : T = 40 :=
by
  -- Using the given condition to derive that T = 40.
  sorry

end tank_full_capacity_l100_100545


namespace abs_neg_three_l100_100274

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l100_100274


namespace probability_of_events_l100_100450

-- Define the sets of tiles in each box
def boxA : Set ℕ := {n | 1 ≤ n ∧ n ≤ 25}
def boxB : Set ℕ := {n | 15 ≤ n ∧ n ≤ 40}

-- Define the specific conditions
def eventA (tile : ℕ) : Prop := tile ≤ 20
def eventB (tile : ℕ) : Prop := (Odd tile ∨ tile > 35)

-- Define the probabilities as calculations
def prob_eventA : ℚ := 20 / 25
def prob_eventB : ℚ := 15 / 26

-- The final probability given independence
def combined_prob : ℚ := prob_eventA * prob_eventB

-- The theorem statement we want to prove
theorem probability_of_events :
  combined_prob = 6 / 13 := 
by 
  -- proof details would go here
  sorry

end probability_of_events_l100_100450


namespace sphere_surface_area_l100_100877

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (S : ℝ)
  (hV : V = 36 * π)
  (hvol : V = (4 / 3) * π * r^3) :
  S = 4 * π * r^2 :=
by
  sorry

end sphere_surface_area_l100_100877


namespace odds_of_picking_blue_marble_l100_100340

theorem odds_of_picking_blue_marble :
  ∀ (total_marbles yellow_marbles : ℕ)
  (h1 : total_marbles = 60)
  (h2 : yellow_marbles = 20)
  (green_marbles : ℕ)
  (h3 : green_marbles = yellow_marbles / 2)
  (remaining_marbles : ℕ)
  (h4 : remaining_marbles = total_marbles - yellow_marbles - green_marbles)
  (blue_marbles : ℕ)
  (h5 : blue_marbles = remaining_marbles / 2),
  (blue_marbles / total_marbles : ℚ) * 100 = 25 :=
by
  intros total_marbles yellow_marbles h1 h2 green_marbles h3 remaining_marbles h4 blue_marbles h5
  sorry

end odds_of_picking_blue_marble_l100_100340


namespace calculation_is_correct_l100_100780

theorem calculation_is_correct : 450 / (6 * 5 - 10 / 2) = 18 :=
by {
  -- Let me provide an outline for solving this problem
  -- (6 * 5 - 10 / 2) must be determined first
  -- After that substituted into the fraction
  sorry
}

end calculation_is_correct_l100_100780


namespace value_set_of_t_l100_100229

theorem value_set_of_t (t : ℝ) :
  (1 > 2 * (1) + 1 - t) ∧ (∀ x : ℝ, x^2 + (2*t-4)*x + 4 > 0) → 3 < t ∧ t < 4 :=
by
  intros h
  sorry

end value_set_of_t_l100_100229


namespace geometric_mean_condition_l100_100550

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

theorem geometric_mean_condition
  (h_arith : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) / 6 = (a 3 + a 4) / 2)
  (h_geom_pos : ∀ n, 0 < b n) :
  Real.sqrt (b 1 * b 2 * b 3 * b 4 * b 5 * b 6) = Real.sqrt (b 3 * b 4) :=
sorry

end geometric_mean_condition_l100_100550


namespace compute_fraction_l100_100513

theorem compute_fraction : 
  (2045^2 - 2030^2) / (2050^2 - 2025^2) = 3 / 5 :=
by
  sorry

end compute_fraction_l100_100513


namespace cindy_marbles_l100_100846

-- Define the initial constants and their values
def initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def number_of_friends : ℕ := 4

-- Define the problem statement in Lean 4
theorem cindy_marbles :
  4 * (initial_marbles - (marbles_per_friend * number_of_friends)) = 720 := by
  sorry

end cindy_marbles_l100_100846


namespace S6_values_l100_100532

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := sorry

axiom geo_seq (q : ℝ) :
  ∀ n : ℕ, a n = a 0 * q ^ n

variable (a3_eq_4 : a 2 = 4) 
variable (S3_eq_7 : S 3 = 7)

theorem S6_values : S 6 = 63 ∨ S 6 = 133 / 27 := sorry

end S6_values_l100_100532


namespace probability_matching_shoes_l100_100483

theorem probability_matching_shoes :
  let total_shoes := 24;
  let total_pairs := 12;
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2;
  let matching_pairs := total_pairs;
  let probability := matching_pairs / total_combinations;
  probability = 1 / 23 :=
by
  let total_shoes := 24
  let total_pairs := 12
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := total_pairs
  let probability := matching_pairs / total_combinations
  have : total_combinations = 276 := by norm_num
  have : matching_pairs = 12 := by norm_num
  have : probability = 1 / 23 := by norm_num
  exact this

end probability_matching_shoes_l100_100483


namespace number_of_prime_pairs_sum_50_l100_100677

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem number_of_prime_pairs_sum_50 : 
  ∃ (P : list (ℕ × ℕ)), 
  (∀ ab ∈ P, (is_prime ab.1) ∧ (is_prime ab.2) ∧ (ab.1 + ab.2 = 50)) 
  ∧ (P.length = 4) := 
by {
  sorry
}

end number_of_prime_pairs_sum_50_l100_100677


namespace bug_twelfth_move_l100_100609

theorem bug_twelfth_move (Q : ℕ → ℚ)
  (hQ0 : Q 0 = 1)
  (hQ1 : Q 1 = 0)
  (hQ2 : Q 2 = 1/2)
  (h_recursive : ∀ n, Q (n + 1) = 1/2 * (1 - Q n)) :
  let m := 683
  let n := 2048
  (Nat.gcd m n = 1) ∧ (m + n = 2731) :=
by
  sorry

end bug_twelfth_move_l100_100609


namespace negation_of_exists_is_forall_l100_100869

theorem negation_of_exists_is_forall :
  (¬ ∃ x : ℝ, x^3 + 1 = 0) ↔ ∀ x : ℝ, x^3 + 1 ≠ 0 :=
by 
  sorry

end negation_of_exists_is_forall_l100_100869


namespace betty_age_l100_100178

theorem betty_age : ∀ (A M B : ℕ), A = 2 * M → A = 4 * B → M = A - 10 → B = 5 :=
by
  intros A M B h1 h2 h3
  sorry

end betty_age_l100_100178


namespace wilted_flowers_correct_l100_100311

-- Definitions based on the given conditions
def total_flowers := 45
def flowers_per_bouquet := 5
def bouquets_made := 2

-- Calculating the number of flowers used for bouquets
def used_flowers : ℕ := bouquets_made * flowers_per_bouquet

-- Question: How many flowers wilted before the wedding?
-- Statement: Prove the number of wilted flowers is 35.
theorem wilted_flowers_correct : total_flowers - used_flowers = 35 := by
  sorry

end wilted_flowers_correct_l100_100311


namespace problem_l100_100830

-- Definitions according to the conditions
def red_balls : ℕ := 1
def black_balls (n : ℕ) : ℕ := n
def total_balls (n : ℕ) : ℕ := red_balls + black_balls n
noncomputable def probability_red (n : ℕ) : ℚ := (red_balls : ℚ) / (total_balls n : ℚ)
noncomputable def variance (n : ℕ) : ℚ := (black_balls n : ℚ) / (total_balls n : ℚ)^2

-- The theorem we want to prove
theorem problem (n : ℕ) (h : 0 < n) : 
  (∀ m : ℕ, n < m → probability_red m < probability_red n) ∧ 
  (∀ m : ℕ, n < m → variance m < variance n) :=
sorry

end problem_l100_100830


namespace necessary_but_not_sufficient_l100_100823

theorem necessary_but_not_sufficient (A B : Prop) (h : A → B) : ¬ (B → A) :=
sorry

end necessary_but_not_sufficient_l100_100823


namespace total_spent_l100_100322

theorem total_spent (bracelet_price keychain_price coloring_book_price : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ)
  (total : ℕ) :
  bracelet_price = 4 →
  keychain_price = 5 →
  coloring_book_price = 3 →
  paula_bracelets = 2 →
  paula_keychains = 1 →
  olive_coloring_books = 1 →
  olive_bracelets = 1 →
  total = paula_bracelets * bracelet_price + paula_keychains * keychain_price +
          olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price →
  total = 20 :=
by sorry

end total_spent_l100_100322


namespace point_on_x_axis_l100_100412

theorem point_on_x_axis (m : ℝ) (h : m - 2 = 0) :
  (m + 3, m - 2) = (5, 0) :=
by
  sorry

end point_on_x_axis_l100_100412


namespace distance_between_foci_l100_100046

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 36 = 0

-- Define the distance between the foci of the ellipse
theorem distance_between_foci (x y : ℝ) (h : ellipse_eq x y) : 2 * Real.sqrt 14.28 = 2 * Real.sqrt 14.28 :=
by sorry

end distance_between_foci_l100_100046


namespace infinitely_many_colorings_l100_100574

def colorings_exist (clr : ℕ → Prop) : Prop :=
  ∀ a b : ℕ, (clr a = clr b) ∧ (0 < a - 10 * b) → clr (a - 10 * b) = clr a

theorem infinitely_many_colorings : ∃ (clr : ℕ → Prop), colorings_exist clr :=
sorry

end infinitely_many_colorings_l100_100574


namespace lcm_18_28_45_65_eq_16380_l100_100354

theorem lcm_18_28_45_65_eq_16380 : Nat.lcm 18 (Nat.lcm 28 (Nat.lcm 45 65)) = 16380 :=
sorry

end lcm_18_28_45_65_eq_16380_l100_100354


namespace joe_speed_l100_100707

theorem joe_speed (pete_speed : ℝ) (joe_speed : ℝ) (time_run : ℝ) (distance : ℝ) 
  (h1 : joe_speed = 2 * pete_speed)
  (h2 : time_run = 2 / 3)
  (h3 : distance = 16)
  (h4 : distance = 3 * pete_speed * time_run) :
  joe_speed = 16 :=
by sorry

end joe_speed_l100_100707


namespace num_prime_pairs_summing_to_50_l100_100672

open Nat

def primes_below_25 : List Nat :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_valid_pair (p q : Nat) : Bool :=
  p + q = 50 ∧ p ∈ primes_below_25 ∧ q ∈ primes_below_25

def pairs_of_primes_that_sum_to_50 : List (Nat × Nat) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem num_prime_pairs_summing_to_50 : pairs_of_primes_that_sum_to_50.length = 4 := 
by 
  -- proof to be provided 
  sorry

end num_prime_pairs_summing_to_50_l100_100672


namespace ferris_wheel_seat_calculation_l100_100317

theorem ferris_wheel_seat_calculation (n k : ℕ) (h1 : n = 4) (h2 : k = 2) : n / k = 2 := 
by
  sorry

end ferris_wheel_seat_calculation_l100_100317


namespace total_distance_between_alice_bob_l100_100910

-- Define the constants for Alice's and Bob's speeds and the time duration in terms of conditions.
def alice_speed := 1 / 12  -- miles per minute
def bob_speed := 3 / 20    -- miles per minute
def time_duration := 120   -- minutes

-- Statement: Prove that the total distance between Alice and Bob after 2 hours is 28 miles.
theorem total_distance_between_alice_bob : (alice_speed * time_duration) + (bob_speed * time_duration) = 28 :=
by
  sorry

end total_distance_between_alice_bob_l100_100910


namespace impossible_tiling_l_tromino_l100_100092

theorem impossible_tiling_l_tromino (k : ℕ) : ¬ ∃ (tiling : (ℕ × ℕ) → ℕ), 
  (∀ x y, 0 ≤ x ∧ x < 5 → 0 ≤ y ∧ y < 7 → ∃ n, tiling (x, y) = k * n) ∧ 
  (∀ t, ttro t ∧ covers_tiling t tiling → 
    ∀ (x y), (x, y) ∈ cells_covered_by t → tiling (x, y) = k) :=
sorry

end impossible_tiling_l_tromino_l100_100092


namespace employee_salary_proof_l100_100881

variable (x : ℝ) (M : ℝ) (P : ℝ)

theorem employee_salary_proof (h1 : x + 1.2 * x + 1.8 * x = 1500)
(h2 : M = 1.2 * x)
(h3 : P = 1.8 * x)
: x = 375 ∧ M = 450 ∧ P = 675 :=
sorry

end employee_salary_proof_l100_100881


namespace partition_weights_l100_100446

theorem partition_weights :
  ∃ A B C : Finset ℕ,
    (∀ x ∈ A, x ≤ 552) ∧
    (∀ x ∈ B, x ≤ 552) ∧
    (∀ x ∈ C, x ≤ 552) ∧
    ∀ x, (x ∈ A ∨ x ∈ B ∨ x ∈ C) ↔ 1 ≤ x ∧ x ≤ 552 ∧
    A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
    A.sum id = 50876 ∧ B.sum id = 50876 ∧ C.sum id = 50876 :=
by
  sorry

end partition_weights_l100_100446


namespace limit_fx_x_squared_equals_f_prime_0_l100_100537

open Filter

noncomputable def f (x : ℝ) : ℝ := x^2

theorem limit_fx_x_squared_equals_f_prime_0 :
  tendsto (λ (Δx : ℝ), (f Δx - f 0) / Δx) (𝓝 0) (𝓝 0) :=
sorry

end limit_fx_x_squared_equals_f_prime_0_l100_100537


namespace function_range_y_eq_1_div_x_minus_2_l100_100555

theorem function_range_y_eq_1_div_x_minus_2 (x : ℝ) : (∀ y : ℝ, y = 1 / (x - 2) ↔ x ∈ {x : ℝ | x ≠ 2}) :=
sorry

end function_range_y_eq_1_div_x_minus_2_l100_100555


namespace compare_f_values_l100_100230

noncomputable def f (x : Real) : Real := 
  Real.cos x + 2 * x * (1 / 2)  -- given f''(pi/6) = 1/2

theorem compare_f_values :
  f (-Real.pi / 3) < f (Real.pi / 3) :=
by
  sorry

end compare_f_values_l100_100230


namespace least_number_divisible_by_11_and_remainder_2_l100_100745

theorem least_number_divisible_by_11_and_remainder_2 :
  ∃ n, (∀ k : ℕ, 3 ≤ k ∧ k ≤ 7 → n % k = 2) ∧ n % 11 = 0 ∧ n = 1262 :=
by
  sorry

end least_number_divisible_by_11_and_remainder_2_l100_100745


namespace problem_statement_l100_100361

def y_and (y : ℤ) : ℤ := 9 - y
def and_y (y : ℤ) : ℤ := y - 9

theorem problem_statement : and_y (y_and 15) = -15 := 
by
  sorry

end problem_statement_l100_100361


namespace abs_neg_three_eq_three_l100_100288

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l100_100288


namespace laura_park_time_percentage_l100_100564

theorem laura_park_time_percentage (num_trips: ℕ) (time_in_park: ℝ) (walking_time: ℝ) 
    (total_percentage_in_park: ℝ) 
    (h1: num_trips = 6) 
    (h2: time_in_park = 2) 
    (h3: walking_time = 0.5) 
    (h4: total_percentage_in_park = 80) : 
    (time_in_park * num_trips) / ((time_in_park + walking_time) * num_trips) * 100 = total_percentage_in_park :=
by
  sorry

end laura_park_time_percentage_l100_100564


namespace repeating_decimal_as_fraction_l100_100464

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l100_100464


namespace functional_equation_satisfied_for_continuous_f_l100_100568

variables {H : Type*} [inner_product_space ℝ H] [complete_space H]
variables {a : ℝ} (b : H → ℝ) (c : H →ₗ[ℝ] H)
variables [is_self_adjoint c]

noncomputable def f (z : H) : ℝ := a + b z + ⟪c z, z⟫

theorem functional_equation_satisfied_for_continuous_f :
  (∀ x y z : H, f (x + y + real.pi • z) + f (x + real.sqrt 2 • z) + 
    f (y + real.sqrt 2 • z) + f (real.pi • z)
    = f (x + y + real.sqrt 2 • z) + f (x + real.pi • z) + 
      f (y + real.pi • z) + f (real.sqrt 2 • z)) :=
sorry

end functional_equation_satisfied_for_continuous_f_l100_100568


namespace abs_neg_three_eq_three_l100_100293

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l100_100293


namespace turnip_weight_possible_l100_100001

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l100_100001


namespace number_of_dice_l100_100610

theorem number_of_dice (n : ℕ) (h : (1 / 6 : ℝ) ^ (n - 1) = 0.0007716049382716049) : n = 5 :=
sorry

end number_of_dice_l100_100610


namespace symmetric_trapezoid_construction_possible_l100_100033

-- Define lengths of legs and distance from intersection point
variables (a b : ℝ)

-- Symmetric trapezoid feasibility condition
theorem symmetric_trapezoid_construction_possible : 3 * b > 2 * a := sorry

end symmetric_trapezoid_construction_possible_l100_100033


namespace rectangle_width_decrease_percent_l100_100132

theorem rectangle_width_decrease_percent (L W : ℝ) (h : L * W = L * W) :
  let L_new := 1.3 * L
  let W_new := W / 1.3 
  let percent_decrease := (1 - (W_new / W)) * 100
  percent_decrease = 23.08 :=
sorry

end rectangle_width_decrease_percent_l100_100132


namespace cos_triple_angle_l100_100692

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l100_100692


namespace train_length_l100_100506

theorem train_length
    (V : ℝ) -- train speed in m/s
    (L : ℝ) -- length of the train in meters
    (H1 : L = V * 18) -- condition: train crosses signal pole in 18 sec
    (H2 : L + 333.33 = V * 38) -- condition: train crosses platform in 38 sec
    (V_pos : 0 < V) -- additional condition: speed must be positive
    : L = 300 :=
by
-- here goes the proof which is not required for our task
sorry

end train_length_l100_100506


namespace cindy_marbles_l100_100844

-- Define the initial constants and their values
def initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def number_of_friends : ℕ := 4

-- Define the problem statement in Lean 4
theorem cindy_marbles :
  4 * (initial_marbles - (marbles_per_friend * number_of_friends)) = 720 := by
  sorry

end cindy_marbles_l100_100844


namespace inequality_solution_set_l100_100302

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono_dec : is_monotonically_decreasing_on_nonneg f) :
  { x : ℝ | f 1 - f (1 / x) < 0 } = { x : ℝ | x < -1 ∨ x > 1 } :=
by
  sorry

end inequality_solution_set_l100_100302


namespace find_dimensions_l100_100806

def is_solution (m n r : ℕ) : Prop :=
  ∃ k0 k1 k2 : ℕ, 
    k0 = (m - 2) * (n - 2) * (r - 2) ∧
    k1 = 2 * ((m - 2) * (n - 2) + (n - 2) * (r - 2) + (r - 2) * (m - 2)) ∧
    k2 = 4 * ((m - 2) + (n - 2) + (r - 2)) ∧
    k0 + k2 - k1 = 1985

theorem find_dimensions (m n r : ℕ) (h : m ≤ n ∧ n ≤ r) (hp : 0 < m ∧ 0 < n ∧ 0 < r) : 
  is_solution m n r :=
sorry

end find_dimensions_l100_100806


namespace simplify_expression_l100_100918

variable (x y : ℝ)

theorem simplify_expression : (x^2 + x * y) / (x * y) * (y^2 / (x + y)) = y := by
  sorry

end simplify_expression_l100_100918


namespace cars_with_neither_features_l100_100893

-- Define the given conditions
def total_cars : ℕ := 65
def cars_with_power_steering : ℕ := 45
def cars_with_power_windows : ℕ := 25
def cars_with_both_features : ℕ := 17

-- Define the statement to be proved
theorem cars_with_neither_features : total_cars - (cars_with_power_steering + cars_with_power_windows - cars_with_both_features) = 12 :=
by
  sorry

end cars_with_neither_features_l100_100893


namespace find_m_l100_100944

open Real

noncomputable def curve_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

noncomputable def line_equation (m t x y : ℝ) : Prop :=
  x = (sqrt 3 / 2) * t + m ∧ y = (1 / 2) * t

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_m (m : ℝ) (h_nonneg : 0 ≤ m) :
  (∀ (t1 t2 : ℝ), (∀ x y, line_equation m t1 x y → curve_equation x y) → 
                   (∀ x y, line_equation m t2 x y → curve_equation x y) →
                   (dist m 0 x1 y1) * (dist m 0 x2 y2) = 1) →
  m = 1 ∨ m = 1 + sqrt 2 :=
sorry

end find_m_l100_100944


namespace area_of_triangle_OAB_is_4_l100_100810

variable (a : ℝ) (h : a ≠ 0)

def center_of_circle_C := (a, 2 / a)
def passes_through_origin := ((0 - a)^2 + (0 - 2 / a)^2 = a^2 + (2 / a)^2)
def x_axis_intersection := (2 * a, 0)
def y_axis_intersection := (0, 4 / a)

noncomputable def area_of_triangle_OAB :=
  1 / 2 * abs (4 / a) * abs (2 * a)

theorem area_of_triangle_OAB_is_4 :
  passes_through_origin a h →
  area_of_triangle_OAB a h = 4 :=
by
  intros
  sorry

end area_of_triangle_OAB_is_4_l100_100810


namespace min_equilateral_triangles_l100_100888

theorem min_equilateral_triangles (s : ℝ) (S : ℝ) :
  s = 1 → S = 15 → 
  225 = (S / s) ^ 2 :=
by
  intros hs hS
  rw [hs, hS]
  simp
  sorry

end min_equilateral_triangles_l100_100888


namespace find_amplitude_l100_100913

noncomputable def amplitude (a b c d x : ℝ) := a * Real.sin (b * x + c) + d

theorem find_amplitude (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_range : ∀ x, -1 ≤ amplitude a b c d x ∧ amplitude a b c d x ≤ 7) :
  a = 4 :=
by
  sorry

end find_amplitude_l100_100913


namespace simplified_evaluation_eq_half_l100_100266

theorem simplified_evaluation_eq_half :
  ∃ x y : ℝ, (|x - 2| + (y + 1)^2 = 0) → 
             (3 * x - 2 * (x^2 - (1/2) * y^2) + (x - (1/2) * y^2) = 1/2) :=
by
  sorry

end simplified_evaluation_eq_half_l100_100266


namespace find_p_l100_100871

theorem find_p :
  ∀ r s : ℝ, (3 * r^2 + 4 * r + 2 = 0) → (3 * s^2 + 4 * s + 2 = 0) →
  (∀ p q : ℝ, (p = - (1/(r^2)) - (1/(s^2))) → (p = -1)) :=
by 
  intros r s hr hs p q hp
  sorry

end find_p_l100_100871


namespace negative_integers_abs_le_4_l100_100620

theorem negative_integers_abs_le_4 (x : Int) (h1 : x < 0) (h2 : abs x ≤ 4) : 
  x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4 :=
by
  sorry

end negative_integers_abs_le_4_l100_100620


namespace beijing_olympics_problem_l100_100778

theorem beijing_olympics_problem
  (M T J D: Type)
  (sports: M → Type)
  (swimming gymnastics athletics volleyball: M → Prop)
  (athlete_sits: M → M → Prop)
  (Maria Tania Juan David: M)
  (woman: M → Prop)
  (left right front next_to: M → M → Prop)
  (h1: ∀ x, swimming x → left x Maria)
  (h2: ∀ x, gymnastics x → front x Juan)
  (h3: next_to Tania David)
  (h4: ∀ x, volleyball x → ∃ y, woman y ∧ next_to y x) :
  athletics David := 
sorry

end beijing_olympics_problem_l100_100778


namespace hot_dogs_remainder_l100_100825

theorem hot_dogs_remainder :
  25197625 % 4 = 1 :=
by
  sorry

end hot_dogs_remainder_l100_100825


namespace find_a_plus_b_l100_100972

noncomputable def f (a b x : ℝ) := a * x + b
noncomputable def g (x : ℝ) := 3 * x - 4

theorem find_a_plus_b (a b : ℝ) (h : ∀ (x : ℝ), g (f a b x) = 4 * x + 5) : a + b = 13 / 3 := 
  sorry

end find_a_plus_b_l100_100972


namespace point_on_x_axis_coordinates_l100_100072

-- Define the conditions
def lies_on_x_axis (M : ℝ × ℝ) : Prop := M.snd = 0

-- State the problem
theorem point_on_x_axis_coordinates (a : ℝ) :
  lies_on_x_axis (a + 3, a + 1) → (a = -1) ∧ ((a + 3, 0) = (2, 0)) :=
by
  intro h
  rw [lies_on_x_axis] at h
  sorry

end point_on_x_axis_coordinates_l100_100072


namespace reach_any_natural_number_l100_100336

theorem reach_any_natural_number (n : ℕ) : ∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = 3 * f k + 1 ∨ f (k + 1) = f k / 2) ∧ (∃ m, f m = n) := by
  sorry

end reach_any_natural_number_l100_100336


namespace arithmetic_mean_of_q_and_r_l100_100126

theorem arithmetic_mean_of_q_and_r (p q r : ℝ) 
  (h₁: (p + q) / 2 = 10) 
  (h₂: r - p = 20) : 
  (q + r) / 2 = 20 :=
sorry

end arithmetic_mean_of_q_and_r_l100_100126


namespace quadrilateral_is_parallelogram_l100_100170

theorem quadrilateral_is_parallelogram 
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 - 2*a*c - 2*b*d = 0) 
  : (a = c ∧ b = d) → parallelogram :=
by
  sorry

end quadrilateral_is_parallelogram_l100_100170


namespace angle_C_in_triangle_l100_100090

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 90) (h2 : A + B + C = 180) : C = 90 :=
sorry

end angle_C_in_triangle_l100_100090


namespace sum_partition_36_l100_100710

theorem sum_partition_36 : 
  ∃ (S : Finset ℕ), S.card = 36 ∧ S.sum id = ((Finset.range 72).sum id) / 2 :=
by
  sorry

end sum_partition_36_l100_100710


namespace value_of_e_l100_100155

theorem value_of_e
  (a b c d e : ℤ)
  (h1 : b = a + 2)
  (h2 : c = a + 4)
  (h3 : d = a + 6)
  (h4 : e = a + 8)
  (h5 : a + c = 146) :
  e = 79 :=
  by sorry

end value_of_e_l100_100155


namespace systematic_sampling_first_group_l100_100593

/-- 
    In a systematic sampling of size 20 from 160 students,
    where students are divided into 20 groups evenly,
    if the number drawn from the 15th group is 116,
    then the number drawn from the first group is 4.
-/
theorem systematic_sampling_first_group (groups : ℕ) (students : ℕ) (interval : ℕ)
  (number_from_15th : ℕ) (number_from_first : ℕ) :
  groups = 20 →
  students = 160 →
  interval = 8 →
  number_from_15th = 116 →
  number_from_first = number_from_15th - interval * 14 →
  number_from_first = 4 :=
by
  intros hgroups hstudents hinterval hnumber_from_15th hequation
  sorry

end systematic_sampling_first_group_l100_100593


namespace circle_sum_condition_l100_100115

theorem circle_sum_condition (n : ℕ) (n_ge_1 : n ≥ 1)
  (x : Fin n → ℝ) (sum_x : (Finset.univ.sum x) = n - 1) :
  ∃ j : Fin n, ∀ k : ℕ, k ≥ 1 → k ≤ n → (Finset.range k).sum (fun i => x ⟨(j + i) % n, sorry⟩) ≥ k - 1 :=
sorry

end circle_sum_condition_l100_100115


namespace remainder_when_a_plus_b_div_40_is_28_l100_100977

theorem remainder_when_a_plus_b_div_40_is_28 :
  ∃ k j : ℤ, (a = 80 * k + 74 ∧ b = 120 * j + 114) → (a + b) % 40 = 28 := by
  sorry

end remainder_when_a_plus_b_div_40_is_28_l100_100977


namespace abs_neg_three_l100_100275

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l100_100275


namespace find_c_in_triangle_l100_100078

theorem find_c_in_triangle
  (angle_B : ℝ)
  (a : ℝ)
  (S : ℝ)
  (h1 : angle_B = 45)
  (h2 : a = 4)
  (h3 : S = 16 * Real.sqrt 2) :
  ∃ c : ℝ, c = 16 :=
by
  sorry

end find_c_in_triangle_l100_100078


namespace denis_oleg_probability_l100_100021

theorem denis_oleg_probability :
  let n := 26 in
  let players := {denis, oleg} in
  let total_matches := n - 1 in
  let total_pairs := n * (n - 1) / 2 in
  ∃ (P : ℚ), P = 1 / 13 ∧
  ∀ (i : ℕ), i ∈ fin total_matches →
  let match_pairs := (n - i) * (n - i - 1) / 2 in
  players ⊆ fin match_pairs → P = 1 / 13 := 
sorry

end denis_oleg_probability_l100_100021


namespace find_cows_l100_100548

theorem find_cows :
  ∃ (D C : ℕ), (2 * D + 4 * C = 2 * (D + C) + 30) → C = 15 := 
sorry

end find_cows_l100_100548


namespace smallest_gcd_value_l100_100953

theorem smallest_gcd_value (m n : ℕ) (hmn : Nat.gcd m n = 15) (hm : m > 0) (hn : n > 0) : Nat.gcd (14 * m) (20 * n) = 30 := 
sorry

end smallest_gcd_value_l100_100953


namespace find_extrema_on_interval_l100_100355

noncomputable def y (x : ℝ) := (10 * x + 10) / (x^2 + 2 * x + 2)

theorem find_extrema_on_interval :
  ∃ (min_val max_val : ℝ) (min_x max_x : ℝ), 
    min_val = 0 ∧ min_x = -1 ∧ max_val = 5 ∧ max_x = 0 ∧ 
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, y x ≥ min_val) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, y x ≤ max_val) :=
by
  sorry

end find_extrema_on_interval_l100_100355


namespace necessary_and_sufficient_condition_l100_100804

variable (a b : ℝ)

theorem necessary_and_sufficient_condition:
  (ab + 1 ≠ a + b) ↔ (a ≠ 1 ∧ b ≠ 1) :=
sorry

end necessary_and_sufficient_condition_l100_100804


namespace abs_neg_three_l100_100284

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l100_100284


namespace emily_page_production_difference_l100_100116

variables (p h : ℕ)

def first_day_pages (p h : ℕ) : ℕ := p * h
def second_day_pages (p h : ℕ) : ℕ := (p - 3) * (h + 3)
def page_difference (p h : ℕ) : ℕ := second_day_pages p h - first_day_pages p h

theorem emily_page_production_difference (h : ℕ) (p_eq_3h : p = 3 * h) :
  page_difference p h = 6 * h - 9 :=
by sorry

end emily_page_production_difference_l100_100116


namespace hannahs_son_cuts_three_strands_per_minute_l100_100068

variable (x : ℕ)

theorem hannahs_son_cuts_three_strands_per_minute
  (total_strands : ℕ)
  (hannah_rate : ℕ)
  (total_time : ℕ)
  (total_strands_cut : ℕ := hannah_rate * total_time)
  (son_rate := (total_strands - total_strands_cut) / total_time)
  (hannah_rate := 8)
  (total_time := 2)
  (total_strands := 22) :
  son_rate = 3 := 
by
  sorry

end hannahs_son_cuts_three_strands_per_minute_l100_100068


namespace range_of_a_l100_100392

theorem range_of_a (a : ℝ) : 
  (2 * (-1) + 0 + a) * (2 * 2 + (-1) + a) < 0 ↔ -3 < a ∧ a < 2 := 
by 
  sorry

end range_of_a_l100_100392


namespace cos_triple_angle_l100_100688

theorem cos_triple_angle :
  (cos θ = 1 / 3) → cos (3 * θ) = -23 / 27 :=
by
  intro h
  have h1 : cos θ = 1 / 3 := h
  sorry

end cos_triple_angle_l100_100688


namespace binomial_expansion_example_l100_100782

theorem binomial_expansion_example :
  57^3 + 3 * (57^2) * 4 + 3 * 57 * (4^2) + 4^3 = 226981 :=
by
  -- The proof would go here, using the steps outlined.
  sorry

end binomial_expansion_example_l100_100782


namespace line_is_tangent_to_circle_l100_100957

theorem line_is_tangent_to_circle
  (θ : Real)
  (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop)
  (h_l : ∀ x y, l x y ↔ x * Real.sin θ + 2 * y * Real.cos θ = 1)
  (h_C : ∀ x y, C x y ↔ x^2 + y^2 = 1) :
  (∀ x y, l x y ↔ x = 1 ∨ x = -1) ↔
  (∃ x y, C x y ∧ ∀ x y, l x y → Real.sqrt ((x * Real.sin θ + 2 * y * Real.cos θ - 1)^2 / (Real.sin θ^2 + 4 * Real.cos θ^2)) = 1) :=
sorry

end line_is_tangent_to_circle_l100_100957


namespace integer_values_in_interval_l100_100949

theorem integer_values_in_interval : (∃ n : ℕ, n = 25 ∧ ∀ x : ℤ, abs x < 4 * π ↔ -12 ≤ x ∧ x ≤ 12) :=
by
  sorry

end integer_values_in_interval_l100_100949


namespace area_ACD_l100_100444

def base_ABD : ℝ := 8
def height_ABD : ℝ := 4
def base_ABC : ℝ := 4
def height_ABC : ℝ := 4

theorem area_ACD : (1/2 * base_ABD * height_ABD) - (1/2 * base_ABC * height_ABC) = 8 := by
  sorry

end area_ACD_l100_100444


namespace train_crosses_signal_pole_in_12_seconds_l100_100761

noncomputable def time_to_cross_signal_pole (length_train : ℕ) (time_to_cross_platform : ℕ) (length_platform : ℕ) : ℕ :=
  let distance_train_platform := length_train + length_platform
  let speed_train := distance_train_platform / time_to_cross_platform
  let time_to_cross_pole := length_train / speed_train
  time_to_cross_pole

theorem train_crosses_signal_pole_in_12_seconds :
  time_to_cross_signal_pole 300 39 675 = 12 :=
by
  -- expected proof in the interactive mode
  sorry

end train_crosses_signal_pole_in_12_seconds_l100_100761


namespace sum_abs_a1_to_a10_l100_100553

def S (n : ℕ) : ℤ := n^2 - 4 * n + 2
def a (n : ℕ) : ℤ := if n = 1 then S 1 else S n - S (n - 1)

theorem sum_abs_a1_to_a10 : (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 66) := 
by
  sorry

end sum_abs_a1_to_a10_l100_100553


namespace jebb_expense_l100_100561

-- Define the costs
def seafood_platter := 45.0
def rib_eye_steak := 38.0
def vintage_wine_glass := 18.0
def chocolate_dessert := 12.0

-- Define the rules and discounts
def discount_percentage := 0.10
def service_fee_12 := 0.12
def service_fee_15 := 0.15
def tip_percentage := 0.20

-- Total food and wine cost
def total_food_and_wine_cost := 
  seafood_platter + rib_eye_steak + (2 * vintage_wine_glass) + chocolate_dessert

-- Total food cost excluding wine
def food_cost_excluding_wine := 
  seafood_platter + rib_eye_steak + chocolate_dessert

-- 10% discount on food cost excluding wine
def discount_amount := discount_percentage * food_cost_excluding_wine
def reduced_food_cost := food_cost_excluding_wine - discount_amount

-- New total cost before applying the service fee
def total_cost_before_service_fee := reduced_food_cost + (2 * vintage_wine_glass)

-- Determine the service fee based on cost
def service_fee := 
  if total_cost_before_service_fee > 80.0 then 
    service_fee_15 * total_cost_before_service_fee 
  else if total_cost_before_service_fee >= 50.0 then 
    service_fee_12 * total_cost_before_service_fee 
  else 
    0.0

-- Total cost after discount and service fee
def total_cost_after_service_fee := total_cost_before_service_fee + service_fee

-- Tip amount (20% of total cost after discount and service fee)
def tip_amount := tip_percentage * total_cost_after_service_fee

-- Total amount Jebb spent
def total_amount_spent := total_cost_after_service_fee + tip_amount

-- Lean theorem statement
theorem jebb_expense :
  total_amount_spent = 167.67 :=
by
  -- prove the theorem here
  sorry

end jebb_expense_l100_100561


namespace factor_theorem_solution_l100_100645

theorem factor_theorem_solution (t : ℝ) :
  (x - t ∣ 3 * x^2 + 10 * x - 8) ↔ (t = 2 / 3 ∨ t = -4) :=
by
  sorry

end factor_theorem_solution_l100_100645


namespace biff_break_even_hours_l100_100626

theorem biff_break_even_hours :
  let ticket := 11
  let drinks_snacks := 3
  let headphones := 16
  let expenses := ticket + drinks_snacks + headphones
  let hourly_income := 12
  let hourly_wifi_cost := 2
  let net_income_per_hour := hourly_income - hourly_wifi_cost
  expenses / net_income_per_hour = 3 :=
by
  sorry

end biff_break_even_hours_l100_100626


namespace peter_horses_food_requirement_l100_100409

theorem peter_horses_food_requirement :
  let daily_oats_per_horse := 4 * 2 in
  let daily_grain_per_horse := 3 in
  let daily_food_per_horse := daily_oats_per_horse + daily_grain_per_horse in
  let number_of_horses := 4 in
  let daily_food_all_horses := daily_food_per_horse * number_of_horses in
  let days := 3 in
  daily_food_all_horses * days = 132 :=
by
  sorry

end peter_horses_food_requirement_l100_100409


namespace time_addition_sum_l100_100094

theorem time_addition_sum (A B C : ℕ) (h1 : A = 7) (h2 : B = 59) (h3 : C = 59) : A + B + C = 125 :=
sorry

end time_addition_sum_l100_100094


namespace triangle_area_ratio_l100_100096

theorem triangle_area_ratio :
  let base_jihye := 3
  let height_jihye := 2
  let base_donggeon := 3
  let height_donggeon := 6.02
  let area_jihye := (base_jihye * height_jihye) / 2
  let area_donggeon := (base_donggeon * height_donggeon) / 2
  (area_donggeon / area_jihye) = 3.01 :=
by
  sorry

end triangle_area_ratio_l100_100096


namespace max_of_a_l100_100442

theorem max_of_a (a b c d : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0)
  (h5 : a + b + c + d = 4) (h6 : a^2 + b^2 + c^2 + d^2 = 8) : a ≤ 1 + Real.sqrt 3 :=
sorry

end max_of_a_l100_100442


namespace rectangle_dimensions_l100_100039

theorem rectangle_dimensions (a b : ℝ) 
  (h_area : a * b = 12) 
  (h_perimeter : 2 * (a + b) = 26) : 
  (a = 1 ∧ b = 12) ∨ (a = 12 ∧ b = 1) :=
sorry

end rectangle_dimensions_l100_100039


namespace find_red_cards_l100_100501

-- We use noncomputable here as we are dealing with real numbers in a theoretical proof context.
noncomputable def red_cards (r b : ℕ) (_initial_prob : r / (r + b) = 1 / 5) 
                            (_added_prob : r / (r + b + 6) = 1 / 7) : ℕ := 
r

theorem find_red_cards 
  {r b : ℕ}
  (h1 : r / (r + b) = 1 / 5)
  (h2 : r / (r + b + 6) = 1 / 7) : 
  red_cards r b h1 h2 = 3 :=
sorry  -- Proof not required

end find_red_cards_l100_100501


namespace count_solutions_inequalities_l100_100950

theorem count_solutions_inequalities :
  {x : ℤ | -5 * x ≥ 2 * x + 10} ∩ {x : ℤ | -3 * x ≤ 15} ∩ {x : ℤ | -6 * x ≥ 3 * x + 21} = {x : ℤ | x = -5 ∨ x = -4 ∨ x = -3} :=
by 
  sorry

end count_solutions_inequalities_l100_100950


namespace percent_of_z_equals_120_percent_of_y_l100_100391

variable {x y z : ℝ}
variable {p : ℝ}

theorem percent_of_z_equals_120_percent_of_y
  (h1 : (p / 100) * z = 1.2 * y)
  (h2 : y = 0.75 * x)
  (h3 : z = 2 * x) :
  p = 45 :=
by sorry

end percent_of_z_equals_120_percent_of_y_l100_100391


namespace area_triangle_QCA_l100_100193

/--
  Given:
  - θ (θ is acute) is the angle at Q between QA and QC
  - Q is at the coordinates (0, 12)
  - A is at the coordinates (3, 12)
  - C is at the coordinates (0, p)

  Prove that the area of triangle QCA is (3/2) * (12 - p) * sin(θ).
-/
theorem area_triangle_QCA (p θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  let Q := (0, 12)
  let A := (3, 12)
  let C := (0, p)
  let base := 3
  let height := (12 - p) * Real.sin θ
  let area := (1 / 2) * base * height
  area = (3 / 2) * (12 - p) * Real.sin θ := by
  sorry

end area_triangle_QCA_l100_100193


namespace find_e_l100_100487

variables (j p t b a : ℝ) (e : ℝ)

theorem find_e
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (e / 100) * p)
  (h4 : b = 1.40 * j)
  (h5 : a = 0.85 * b)
  (h6 : e = 2 * ((p - a) / p) * 100) :
  e = 21.5 := by
  sorry

end find_e_l100_100487


namespace sum_of_first_twelve_terms_l100_100930

section ArithmeticSequence

variables (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ)

-- General definition of the nth term in arithmetic progression
def arithmetic_term (n : ℕ) : ℚ := a₁ + (n - 1) * d

-- Given conditions in the problem
axiom fifth_term : arithmetic_term a₁ d 5 = 1
axiom seventeenth_term : arithmetic_term a₁ d 17 = 18

-- Define the sum of the first n terms in arithmetic sequence
def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Statement of the proof problem
theorem sum_of_first_twelve_terms : 
  sum_arithmetic_sequence a₁ d 12 = 37.5 := 
sorry

end ArithmeticSequence

end sum_of_first_twelve_terms_l100_100930


namespace Henry_age_l100_100243

-- Define the main proof statement
theorem Henry_age (h s : ℕ) 
(h1 : h + 8 = 3 * (s - 1))
(h2 : (h - 25) + (s - 25) = 83) : h = 97 :=
by
  sorry

end Henry_age_l100_100243


namespace expression_calculation_l100_100189

theorem expression_calculation : 
  (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = 28 * 21^1005 :=
by
  sorry

end expression_calculation_l100_100189


namespace cord_lengths_l100_100588

noncomputable def cordLengthFirstDog (distance : ℝ) : ℝ :=
  distance / 2

noncomputable def cordLengthSecondDog (distance : ℝ) : ℝ :=
  distance / 2

noncomputable def cordLengthThirdDog (radius : ℝ) : ℝ :=
  radius

theorem cord_lengths (d1 d2 r : ℝ) (h1 : d1 = 30) (h2 : d2 = 40) (h3 : r = 20) :
  cordLengthFirstDog d1 = 15 ∧ cordLengthSecondDog d2 = 20 ∧ cordLengthThirdDog r = 20 := by
  sorry

end cord_lengths_l100_100588


namespace least_b_not_in_range_l100_100743

theorem least_b_not_in_range : ∃ b : ℤ, -10 = b ∧ ∀ x : ℝ, x^2 + b * x + 20 ≠ -10 :=
sorry

end least_b_not_in_range_l100_100743


namespace problem_l100_100861

variable (a b : ℝ)

theorem problem (h : a = 1.25 * b) : (4 * b) / a = 3.2 :=
by
  sorry

end problem_l100_100861


namespace part1_part2_l100_100661

variable (m x : ℝ)

def y (m x : ℝ) := (m + 1) * x ^ 2 - m * x + m - 1

theorem part1 (m : ℝ) (h_empty : ∀ x : ℝ, y m x < 0 → false) : 
  m ∈ Ici (2 * Real.sqrt 3 / 3) :=
sorry

theorem part2 (m : ℝ) (h_subset : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → y m x ≥ 0) : 
  m ∈ Ici (2 * Real.sqrt 3 / 3) :=
sorry

end part1_part2_l100_100661


namespace room_length_l100_100134

theorem room_length (w : ℝ) (cost_rate : ℝ) (total_cost : ℝ) (h : w = 4) (h1 : cost_rate = 800) (h2 : total_cost = 17600) : 
  let L := total_cost / (w * cost_rate)
  L = 5.5 :=
by
  sorry

end room_length_l100_100134


namespace length_of_other_leg_l100_100868

theorem length_of_other_leg (c a b : ℕ) (h1 : c = 10) (h2 : a = 6) (h3 : c^2 = a^2 + b^2) : b = 8 :=
by
  sorry

end length_of_other_leg_l100_100868


namespace discarded_number_l100_100990

theorem discarded_number (S x : ℕ) (h1 : S / 50 = 50) (h2 : (S - x - 55) / 48 = 50) : x = 45 :=
by
  sorry

end discarded_number_l100_100990


namespace percentage_of_original_solution_l100_100504

-- Define the problem and conditions
variable (P : ℝ)
variable (h1 : (0.5 * P + 0.5 * 60) = 55)

-- The theorem to prove
theorem percentage_of_original_solution : P = 50 :=
by
  -- Proof will go here
  sorry

end percentage_of_original_solution_l100_100504


namespace vanya_meets_mother_opposite_dir_every_4_minutes_l100_100325

-- Define the parameters
def lake_perimeter : ℝ := sorry  -- Length of the lake's perimeter, denoted as l
def mother_time_lap : ℝ := 12    -- Time taken by the mother to complete one lap (in minutes)
def vanya_time_overtake : ℝ := 12 -- Time taken by Vanya to overtake the mother (in minutes)

-- Define speeds
noncomputable def mother_speed : ℝ := lake_perimeter / mother_time_lap
noncomputable def vanya_speed : ℝ := 2 * lake_perimeter / vanya_time_overtake

-- Define their relative speed when moving in opposite directions
noncomputable def relative_speed : ℝ := mother_speed + vanya_speed

-- Prove that the meeting interval is 4 minutes
theorem vanya_meets_mother_opposite_dir_every_4_minutes :
  (lake_perimeter / relative_speed) = 4 := by
  sorry

end vanya_meets_mother_opposite_dir_every_4_minutes_l100_100325


namespace revenue_equation_l100_100081

theorem revenue_equation (x : ℝ) (r_j r_t : ℝ) (h1 : r_j = 90) (h2 : r_t = 144) :
  r_j + r_j * (1 + x) + r_j * (1 + x)^2 = r_t :=
by
  rw [h1, h2]
  sorry

end revenue_equation_l100_100081


namespace cards_received_at_home_l100_100406

-- Definitions based on the conditions
def initial_cards := 403
def total_cards := 690

-- The theorem to prove the number of cards received at home
theorem cards_received_at_home : total_cards - initial_cards = 287 :=
by
  -- Proof goes here, but we use sorry as a placeholder.
  sorry

end cards_received_at_home_l100_100406


namespace min_value_quadratic_l100_100917

theorem min_value_quadratic : 
  ∀ x : ℝ, (4 * x^2 - 12 * x + 9) ≥ 0 :=
by
  sorry

end min_value_quadratic_l100_100917


namespace calc_pow_product_l100_100028

theorem calc_pow_product : (0.25 ^ 2023) * (4 ^ 2023) = 1 := 
  by 
  sorry

end calc_pow_product_l100_100028


namespace tangent_line_solution_l100_100526

variables (x y : ℝ)

noncomputable def circle_equation (m : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + m * y = 0

def point_on_circle (m : ℝ) : Prop :=
  circle_equation 1 1 m

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

theorem tangent_line_solution (m : ℝ) :
  point_on_circle m →
  m = 2 →
  tangent_line_equation 1 1 :=
by
  sorry

end tangent_line_solution_l100_100526


namespace point_on_parabola_l100_100455

def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem point_on_parabola : parabola (1/2) = 0 := 
by sorry

end point_on_parabola_l100_100455


namespace repeating_decimal_as_fraction_l100_100463

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l100_100463


namespace sqrt49_times_sqrt25_eq_5sqrt7_l100_100187

noncomputable def sqrt49_times_sqrt25 : ℝ :=
  Real.sqrt (49 * Real.sqrt 25)

theorem sqrt49_times_sqrt25_eq_5sqrt7 :
  sqrt49_times_sqrt25 = 5 * Real.sqrt 7 :=
by
sorry

end sqrt49_times_sqrt25_eq_5sqrt7_l100_100187


namespace fishermen_total_catch_l100_100492

noncomputable def m : ℕ := 30  -- Mike can catch 30 fish per hour
noncomputable def j : ℕ := 2 * m  -- Jim can catch twice as much as Mike
noncomputable def b : ℕ := j + (j / 2)  -- Bob can catch 50% more than Jim

noncomputable def fish_caught_in_40_minutes : ℕ := (2 * m) / 3 -- Fishermen fish together for 40 minutes (2/3 hour)
noncomputable def fish_caught_by_jim_in_remaining_time : ℕ := j / 3 -- Jim fishes alone for the remaining 20 minutes (1/3 hour)

noncomputable def total_fish_caught : ℕ :=
  fish_caught_in_40_minutes * 3 + fish_caught_by_jim_in_remaining_time

theorem fishermen_total_catch : total_fish_caught = 140 := by
  sorry

end fishermen_total_catch_l100_100492


namespace intersection_one_point_l100_100896

def quadratic_function (x : ℝ) : ℝ := -x^2 + 5 * x
def linear_function (x : ℝ) (t : ℝ) : ℝ := -3 * x + t
def quadratic_combined_function (x : ℝ) (t : ℝ) : ℝ := x^2 - 8 * x + t

theorem intersection_one_point (t : ℝ) : 
  (64 - 4 * t = 0) → t = 16 :=
by
  intro h
  sorry

end intersection_one_point_l100_100896


namespace sum_divisible_by_11_l100_100521

theorem sum_divisible_by_11 (n : ℕ) : (6^(2*n) + 3^n + 3^(n+2)) % 11 = 0 := by
  sorry

end sum_divisible_by_11_l100_100521


namespace lark_lock_combination_count_l100_100401

-- Definitions for the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def lark_lock_combination (a b c : ℕ) : Prop := 
  is_odd a ∧ is_even b ∧ is_multiple_of_5 c ∧ 1 ≤ a ∧ a ≤ 30 ∧ 1 ≤ b ∧ b ≤ 30 ∧ 1 ≤ c ∧ c ≤ 30

-- The core theorem
theorem lark_lock_combination_count : 
  (∃ a b c : ℕ, lark_lock_combination a b c) ↔ (15 * 15 * 6 = 1350) :=
by
  sorry

end lark_lock_combination_count_l100_100401


namespace arithmetic_sequence_properties_summation_inequality_l100_100551

noncomputable def a_n (n : ℕ) : ℕ := 3 + 2 * (n - 1)
noncomputable def b_n (n : ℕ) : ℕ := 8 ^ (n - 1)
noncomputable def S_n (n : ℕ) : ℕ := n * (n + 2)

theorem arithmetic_sequence_properties (d q : ℕ) (h₁ : a_1 = 3) (h₂ : b_1 = 1)
    (h₃ : b_2 * S_2 = 64) (h₄ : b_3 * S_3 = 960) :
    a_n = (λ n, 2 * n + 1) ∧ b_n = (λ n, 8 ^ (n - 1)) :=
by
  sorry

theorem summation_inequality (n : ℕ) :
    ∑ k in Finset.range n, (1 / S_n k) < 3 / 4 :=
by
  sorry

end arithmetic_sequence_properties_summation_inequality_l100_100551


namespace complement_union_A_B_is_correct_l100_100534

-- Define the set of real numbers R
def R : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | ∃ (y : ℝ), y = Real.log (x + 3) }

-- Simplified definition for A to reflect x > -3
def A_simplified : Set ℝ := { x | x > -3 }

-- Define set B
def B : Set ℝ := { x | x ≥ 2 }

-- Define the union of A and B
def union_A_B : Set ℝ := A_simplified ∪ B

-- Define the complement of the union in R
def complement_R_union_A_B : Set ℝ := R \ union_A_B

-- State the theorem
theorem complement_union_A_B_is_correct :
  complement_R_union_A_B = { x | x ≤ -3 } := by
  sorry

end complement_union_A_B_is_correct_l100_100534


namespace chennai_to_hyderabad_distance_l100_100633

-- Definitions of the conditions
def david_speed := 50 -- mph
def lewis_speed := 70 -- mph
def meet_point := 250 -- miles from Chennai

-- Theorem statement
theorem chennai_to_hyderabad_distance :
  ∃ D T : ℝ, lewis_speed * T = D + (D - meet_point) ∧ david_speed * T = meet_point ∧ D = 300 :=
by
  sorry

end chennai_to_hyderabad_distance_l100_100633


namespace largest_whole_number_l100_100194

theorem largest_whole_number (x : ℕ) (h : 6 * x + 3 < 150) : x ≤ 24 :=
sorry

end largest_whole_number_l100_100194


namespace problem1_problem2_l100_100818

-- Definitions based on the given conditions
def p (a : ℝ) (x : ℝ) : Prop := a < x ∧ x < 3 * a
def q (x : ℝ) : Prop := x^2 - 5 * x + 6 < 0

-- Problem (1)
theorem problem1 (a x : ℝ) (h : a = 1) (hp : p a x) (hq : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Problem (2)
theorem problem2 (a : ℝ) (h : ∀ x, q x → p a x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end problem1_problem2_l100_100818


namespace num_colors_l100_100571

def total_balls := 350
def balls_per_color := 35

theorem num_colors :
  total_balls / balls_per_color = 10 := 
by
  sorry

end num_colors_l100_100571


namespace circle_area_is_162_pi_l100_100038

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def circle_area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

def R : ℝ × ℝ := (5, -2)
def S : ℝ × ℝ := (-4, 7)

theorem circle_area_is_162_pi :
  circle_area (distance R S) = 162 * Real.pi :=
by
  sorry

end circle_area_is_162_pi_l100_100038


namespace percentage_increase_on_bought_price_l100_100175

-- Define the conditions as Lean definitions
def original_price (P : ℝ) : ℝ := P
def bought_price (P : ℝ) : ℝ := 0.90 * P
def selling_price (P : ℝ) : ℝ := 1.62000000000000014 * P

-- Lean statement to prove the required result
theorem percentage_increase_on_bought_price (P : ℝ) :
  (selling_price P - bought_price P) / bought_price P * 100 = 80.00000000000002 := by
  sorry

end percentage_increase_on_bought_price_l100_100175


namespace BP_value_l100_100496

-- Define the problem conditions and statement.
theorem BP_value
  (A B C D P : Point)
  (on_circle : ∀ point ∈ {A, B, C, D}, is_on_circle point)
  (intersect : P ∈ (line_through A C) ∧ P ∈ (line_through B D))
  (AP : Real := 10)
  (PC : Real := 2)
  (BD : Real := 9)
  (BP_lt_DP : ∃ x y : Real, BP = x ∧ DP = y ∧ x + y = BD ∧ x < y) :
  BP = 4 :=
by
  sorry -- Proof is omitted

end BP_value_l100_100496


namespace volume_conversion_l100_100748

theorem volume_conversion (a : Nat) (b : Nat) (c : Nat) (d : Nat) (e : Nat) (f : Nat)
  (h1 : a = 1) (h2 : b = 3) (h3 : c = a^3) (h4 : d = b^3) (h5 : c = 1) (h6 : d = 27) 
  (h7 : 1 = 1) (h8 : 27 = 27) (h9 : e = 5) 
  (h10 : f = e * d) : 
  f = 135 := 
sorry

end volume_conversion_l100_100748


namespace discount_offered_is_5_percent_l100_100173

noncomputable def cost_price : ℝ := 100

noncomputable def selling_price_with_discount : ℝ := cost_price * 1.216

noncomputable def selling_price_without_discount : ℝ := cost_price * 1.28

noncomputable def discount : ℝ := selling_price_without_discount - selling_price_with_discount

noncomputable def discount_percentage : ℝ := (discount / selling_price_without_discount) * 100

theorem discount_offered_is_5_percent : discount_percentage = 5 :=
by 
  sorry

end discount_offered_is_5_percent_l100_100173


namespace existence_of_intersection_l100_100220

def setA (m : ℝ) : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x^2 + m * x - y + 2 = 0) }
def setB : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x - y + 1 = 0) ∧ (0 ≤ x ∧ x ≤ 2) }

theorem existence_of_intersection (m : ℝ) : (∃ (p : ℝ × ℝ), p ∈ (setA m ∩ setB)) ↔ m ≤ -1 := 
sorry

end existence_of_intersection_l100_100220


namespace num_divisors_of_m_cubed_l100_100770

theorem num_divisors_of_m_cubed (m : ℕ) (h : ∃ p : ℕ, Nat.Prime p ∧ m = p ^ 4) :
    Nat.totient (m ^ 3) = 13 := 
sorry

end num_divisors_of_m_cubed_l100_100770


namespace problem1_problem2_l100_100560

variables (A B C D : Type) [Real.uniform_space A]
variables {a b c BD AD DC : Real}
variables (angle_ABC angle_ACB : Real)

def proof1 (h1 : b^2 = a * c) (h2 : BD * Real.sin angle_ABC = a * Real.sin angle_ACB) : Prop :=
  BD = b

def proof2 (h3 : AD = 2 * DC) (h1 : b^2 = a * c) : Prop :=
  Real.cos angle_ABC = 7 / 12

theorem problem1 {A B C D : Type} [Real.uniform_space A]
  {a b c BD : Real}
  {angle_ABC angle_ACB : Real}
  (h1 : b^2 = a * c)
  (h2 : BD * Real.sin angle_ABC = a * Real.sin angle_ACB) :
  proof1 a b c BD angle_ABC angle_ACB h1 h2 :=
sorry

theorem problem2 {A B C D : Type} [Real.uniform_space A]
  {a b c BD AD DC : Real}
  {angle_ABC angle_ACB : Real}
  (h1 : b^2 = a * c)
  (h3 : AD = 2 * DC) :
  proof2 a b c BD angle_ABC h1 h3 :=
sorry

end problem1_problem2_l100_100560


namespace Mr_Kishore_Savings_l100_100616

noncomputable def total_expenses := 
  5000 + 1500 + 4500 + 2500 + 2000 + 6100 + 3500 + 2700

noncomputable def monthly_salary (S : ℝ) := 
  total_expenses + 0.10 * S = S

noncomputable def savings (S : ℝ) := 
  0.10 * S

theorem Mr_Kishore_Savings : 
  ∃ S : ℝ, monthly_salary S ∧ savings S = 3422.22 :=
by
  sorry

end Mr_Kishore_Savings_l100_100616


namespace least_number_divisible_by_11_and_leaves_remainder_2_l100_100747

theorem least_number_divisible_by_11_and_leaves_remainder_2 : 
  ∃ n : ℕ, (n % 11 = 0) ∧ (∀ m ∈ {3, 4, 5, 6, 7}, n % m = 2) ∧ n = 3782 :=
by
  sorry

end least_number_divisible_by_11_and_leaves_remainder_2_l100_100747


namespace repeating_decimal_to_fraction_l100_100479

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l100_100479


namespace approx_cube_of_331_l100_100638

noncomputable def cube (x : ℝ) : ℝ := x * x * x

theorem approx_cube_of_331 : 
  ∃ ε > 0, abs (cube 0.331 - 0.037) < ε :=
by
  sorry

end approx_cube_of_331_l100_100638


namespace quadratic_factorization_l100_100730

theorem quadratic_factorization (C D : ℤ) (h : (15 * y^2 - 74 * y + 48) = (C * y - 16) * (D * y - 3)) :
  C * D + C = 20 :=
sorry

end quadratic_factorization_l100_100730


namespace equation_of_line_with_x_intercept_and_slope_l100_100993

theorem equation_of_line_with_x_intercept_and_slope :
  ∃ (a b c : ℝ), a * x - b * y + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
sorry

end equation_of_line_with_x_intercept_and_slope_l100_100993


namespace two_sin_cos_75_eq_half_l100_100185

noncomputable def two_sin_cos_of_75_deg : ℝ :=
  2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180)

theorem two_sin_cos_75_eq_half : two_sin_cos_of_75_deg = 1 / 2 :=
by
  -- The steps to prove this theorem are omitted deliberately
  sorry

end two_sin_cos_75_eq_half_l100_100185


namespace same_terminal_side_angle_l100_100988

theorem same_terminal_side_angle (k : ℤ) : 
  ∃ (θ : ℤ), θ = k * 360 + 257 ∧ (θ % 360 = (-463) % 360) :=
by
  sorry

end same_terminal_side_angle_l100_100988


namespace evaluate_fraction_l100_100892

theorem evaluate_fraction : (35 / 0.07) = 500 := 
by
  sorry

end evaluate_fraction_l100_100892


namespace find_denomination_of_oliver_bills_l100_100260

-- Definitions based on conditions
def denomination (x : ℕ) : Prop :=
  let oliver_total := 10 * x + 3 * 5
  let william_total := 15 * 10 + 4 * 5
  oliver_total = william_total + 45

-- The Lean theorem statement
theorem find_denomination_of_oliver_bills (x : ℕ) : denomination x → x = 20 := by
  sorry

end find_denomination_of_oliver_bills_l100_100260


namespace min_value_of_x2_plus_y2_min_value_of_reciprocal_sum_l100_100225

namespace MathProof

-- Definitions and conditions
variables {x y : ℝ}
axiom x_pos : x > 0
axiom y_pos : y > 0
axiom sum_eq_one : x + y = 1

-- Problem Statement 1: Prove the minimum value of x^2 + y^2 is 1/2
theorem min_value_of_x2_plus_y2 : ∃ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) ∧ (x^2 + y^2 = 1/2) :=
by
  sorry

-- Problem Statement 2: Prove the minimum value of 1/x + 1/y + 1/(xy) is 6
theorem min_value_of_reciprocal_sum : ∃ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) ∧ ((1/x + 1/y + 1/(x*y)) = 6) :=
by
  sorry

end MathProof

end min_value_of_x2_plus_y2_min_value_of_reciprocal_sum_l100_100225


namespace shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder_l100_100529

theorem shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder
  (c : ℝ)
  (r : ℝ)
  (θ : ℝ)
  (hr : r ≥ 0)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  ∃ (x y z : ℝ), (z = c) ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ :=
by
  sorry

end shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder_l100_100529


namespace differentiable_inequality_l100_100104

theorem differentiable_inequality 
  {a b : ℝ} 
  {f g : ℝ → ℝ} 
  (hdiff_f : DifferentiableOn ℝ f (Set.Icc a b))
  (hdiff_g : DifferentiableOn ℝ g (Set.Icc a b))
  (hderiv_ineq : ∀ x ∈ Set.Ioo a b, (deriv f x > deriv g x)) :
  ∀ x ∈ Set.Ioo a b, f x + g a > g x + f a :=
by 
  sorry

end differentiable_inequality_l100_100104


namespace common_root_exists_l100_100520

theorem common_root_exists :
  ∃ x, (3 * x^4 + 13 * x^3 + 20 * x^2 + 17 * x + 7 = 0) ∧ (3 * x^4 + x^3 - 8 * x^2 + 11 * x - 7 = 0) → x = -7 / 3 := 
by
  sorry

end common_root_exists_l100_100520


namespace product_of_solutions_abs_eq_40_l100_100788

theorem product_of_solutions_abs_eq_40 :
  (∃ x1 x2 : ℝ, (|3 * x1 - 5| = 40) ∧ (|3 * x2 - 5| = 40) ∧ ((x1 * x2) = -175)) :=
by
  sorry

end product_of_solutions_abs_eq_40_l100_100788


namespace nearest_integer_ratio_l100_100956

variable (a b : ℝ)

-- Given condition and constraints
def condition : Prop := (a > b) ∧ (b > 0) ∧ (a + b) / 2 = 3 * Real.sqrt (a * b)

-- Main statement to prove
theorem nearest_integer_ratio (h : condition a b) : Int.floor (a / b) = 34 ∨ Int.floor (a / b) = 33 := sorry

end nearest_integer_ratio_l100_100956


namespace sheila_monthly_savings_l100_100857

-- Define the conditions and the question in Lean
def initial_savings : ℕ := 3000
def family_contribution : ℕ := 7000
def years : ℕ := 4
def final_amount : ℕ := 23248

-- Function to calculate the monthly saving given the conditions
def monthly_savings (initial_savings family_contribution years final_amount : ℕ) : ℕ :=
  (final_amount - (initial_savings + family_contribution)) / (years * 12)

-- The theorem we need to prove in Lean
theorem sheila_monthly_savings :
  monthly_savings initial_savings family_contribution years final_amount = 276 :=
by
  sorry

end sheila_monthly_savings_l100_100857


namespace multiples_of_5_with_units_digit_0_l100_100951

theorem multiples_of_5_with_units_digit_0 (h1 : ∀ n : ℕ, n % 5 = 0 → (n % 10 = 0 ∨ n % 10 = 5))
  (h2 : ∀ m : ℕ, m < 200 → m % 5 = 0) :
  ∃ k : ℕ, k = 19 ∧ (∀ x : ℕ, (x < 200) ∧ (x % 5 = 0) → (x % 10 = 0) → k = (k - 1) + 1) := sorry

end multiples_of_5_with_units_digit_0_l100_100951


namespace solve_inequality_l100_100984

theorem solve_inequality :
  {x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4} = {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3)} :=
sorry

end solve_inequality_l100_100984


namespace calculate_E_l100_100489

theorem calculate_E (P J T B A E : ℝ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : B = 1.40 * J)
  (h4 : A = 0.85 * B)
  (h5 : T = P - (E / 100) * P)
  (h6 : E = 2 * ((P - A) / P) * 100) : 
  E = 21.5 := 
sorry

end calculate_E_l100_100489


namespace olivia_total_cost_l100_100261

-- Definitions based on conditions given in the problem.
def daily_rate : ℕ := 30 -- daily rate in dollars per day
def mileage_rate : ℕ := 25 -- mileage rate in cents per mile (converted to cents to avoid fractions)
def rental_days : ℕ := 3 -- number of days the car is rented
def miles_driven : ℕ := 500 -- number of miles driven

-- Calculate costs in cents to avoid fractions in the Lean theorem statement.
def daily_rental_cost : ℕ := daily_rate * rental_days * 100
def mileage_cost : ℕ := mileage_rate * miles_driven
def total_cost : ℕ := daily_rental_cost + mileage_cost

-- Final statement to be proved, converting total cost back to dollars.
theorem olivia_total_cost : (total_cost / 100) = 215 := by
  sorry

end olivia_total_cost_l100_100261


namespace anthony_initial_pencils_l100_100180

def initial_pencils (given_pencils : ℝ) (remaining_pencils : ℝ) : ℝ :=
  given_pencils + remaining_pencils

theorem anthony_initial_pencils :
  initial_pencils 9.0 47.0 = 56.0 :=
by
  sorry

end anthony_initial_pencils_l100_100180


namespace directrix_of_parabola_l100_100432

theorem directrix_of_parabola (x y : ℝ) (h : y = (1/4) * x^2) : y = -1 :=
sorry

end directrix_of_parabola_l100_100432


namespace taylor_pets_count_l100_100724

noncomputable def totalPetsTaylorFriends (T : ℕ) (x1 : ℕ) (x2 : ℕ) : ℕ :=
  T + 3 * x1 + 2 * x2

theorem taylor_pets_count (T : ℕ) (x1 x2 : ℕ) (h1 : x1 = 2 * T) (h2 : x2 = 2) (h3 : totalPetsTaylorFriends T x1 x2 = 32) :
  T = 4 :=
by
  sorry

end taylor_pets_count_l100_100724


namespace cindy_marbles_l100_100845

-- Define the initial constants and their values
def initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def number_of_friends : ℕ := 4

-- Define the problem statement in Lean 4
theorem cindy_marbles :
  4 * (initial_marbles - (marbles_per_friend * number_of_friends)) = 720 := by
  sorry

end cindy_marbles_l100_100845


namespace seating_arrangement_l100_100790

theorem seating_arrangement (x y : ℕ) (h1 : 9 * x + 7 * y = 61) : x = 6 :=
by 
  sorry

end seating_arrangement_l100_100790


namespace find_BP_l100_100495

theorem find_BP
    (A B C D P : Type) [Point A] [Point B] [Point C] [Point D] [Point P]
    (h_circle : Circle A B C D) 
    (h_intersect : Intersect AC BD P)
    (h_AP : AP = 10) 
    (h_PC : PC = 2) 
    (h_BD : BD = 9) 
    (h_BP_DP : BP < DP) : 
    BP = 4 := 
sorry

end find_BP_l100_100495


namespace maxwell_distance_traveled_l100_100158

theorem maxwell_distance_traveled
  (distance_between_homes : ℕ)
  (maxwell_speed : ℕ)
  (brad_speed : ℕ)
  (meeting_time : ℕ)
  (h1 : distance_between_homes = 72)
  (h2 : maxwell_speed = 6)
  (h3 : brad_speed = 12)
  (h4 : meeting_time = distance_between_homes / (maxwell_speed + brad_speed)) :
  maxwell_speed * meeting_time = 24 :=
by
  sorry

end maxwell_distance_traveled_l100_100158


namespace unique_solution_of_function_eq_l100_100646

theorem unique_solution_of_function_eq (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y) : f = id := 
sorry

end unique_solution_of_function_eq_l100_100646


namespace leadership_ways_l100_100767

open Fintype

/--
In a village with 16 members, prove that the total number
of ways to choose a leadership consisting of 1 mayor,
3 deputy mayors, each with 3 council members, is 154828800.
-/
theorem leadership_ways (n : ℕ) (hn : n = 16) :
  (∃ m d1 d2 d3 c1 c2 c3 : Finₓ n,
    ∀ (h : m ≠ d1) (h : m ≠ d2) (h : m ≠ d3) (h : d1 ≠ d2)
    (h : d1 ≠ d3) (h : d2 ≠ d3),
      16 * 15 * 14 * 13 * choose 12 3 * choose 9 3 * choose 6 3 = 154828800) :=
by -- proof to be added
  sorry

end leadership_ways_l100_100767


namespace sequence_max_length_l100_100792

theorem sequence_max_length (x : ℕ) :
  (2000 - 2 * x > 0) ∧ (3 * x - 2000 > 0) ∧ (4000 - 5 * x > 0) ∧ 
  (8 * x - 6000 > 0) ∧ (10000 - 13 * x > 0) ∧ (21 * x - 16000 > 0) → x = 762 :=
by
  sorry

end sequence_max_length_l100_100792


namespace ratio_of_larger_to_smaller_is_sqrt_six_l100_100142

def sum_of_squares_eq_seven_times_difference (a b : ℝ) : Prop := 
  a^2 + b^2 = 7 * (a - b)

theorem ratio_of_larger_to_smaller_is_sqrt_six {a b : ℝ} (h : sum_of_squares_eq_seven_times_difference a b) (h1 : a > b) : 
  a / b = Real.sqrt 6 :=
sorry

end ratio_of_larger_to_smaller_is_sqrt_six_l100_100142


namespace abs_neg_three_l100_100281

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l100_100281


namespace coefficient_of_x_squared_l100_100963

noncomputable theory

open_locale big_operators

theorem coefficient_of_x_squared :
  let term (r : ℕ) := (-(1/2))^r * (Nat.choose 5 r : ℚ) * x^((10 - 3 * r) / 2) in
  (∑ r in Finset.range 6, term r).coeff 2 = (5 / 2 : ℚ) := 
sorry

end coefficient_of_x_squared_l100_100963


namespace g_at_neg2_eq_8_l100_100108

-- Define the functions f and g
def f (x : ℤ) : ℤ := 4 * x - 6
def g (y : ℤ) : ℤ := 3 * (y + 6/4)^2 + 4 * (y + 6/4) + 1

-- Statement of the math proof problem:
theorem g_at_neg2_eq_8 : g (-2) = 8 := 
by 
  sorry

end g_at_neg2_eq_8_l100_100108


namespace abs_neg_three_l100_100294

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l100_100294


namespace non_congruent_triangles_l100_100395

-- Definition of points and isosceles property
variable (A B C P Q R : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited R]

-- Conditions of the problem
def is_isosceles (A B C : Type) : Prop := (A = B) ∧ (A = C)
def is_midpoint (P Q R : Type) (A B C : Type) : Prop := sorry -- precise formal definition omitted for brevity

-- Theorem stating the final result
theorem non_congruent_triangles (A B C P Q R : Type)
  (h_iso : is_isosceles A B C)
  (h_midpoints : is_midpoint P Q R A B C) :
  ∃ (n : ℕ), n = 4 := 
  by 
    -- proof abbreviated
    sorry

end non_congruent_triangles_l100_100395


namespace minimize_f_l100_100037

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l100_100037


namespace waiter_earning_correct_l100_100182

-- Definitions based on the conditions
def tip1 : ℝ := 25 * 0.15
def tip2 : ℝ := 22 * 0.18
def tip3 : ℝ := 35 * 0.20
def tip4 : ℝ := 30 * 0.10

def total_tips : ℝ := tip1 + tip2 + tip3 + tip4
def commission : ℝ := total_tips * 0.05
def net_tips : ℝ := total_tips - commission

-- Theorem statement
theorem waiter_earning_correct : net_tips = 16.82 := by
  sorry

end waiter_earning_correct_l100_100182


namespace part_one_part_two_l100_100943

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem part_one (a : ℝ) (h_pos : 0 < a) :
  (∀ x > 0, (Real.log x + 1/x + 1 - a) ≥ 0) ↔ (0 < a ∧ a ≤ 2) :=
sorry

theorem part_two (a : ℝ) (h_pos : 0 < a) :
  (∀ x, (x - 1) * (f x a) ≥ 0) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end part_one_part_two_l100_100943


namespace sin_of_right_angle_l100_100961

theorem sin_of_right_angle (A B C : Type)
  (angle_A : Real) (AB BC : Real)
  (h_angleA : angle_A = 90)
  (h_AB : AB = 16)
  (h_BC : BC = 24) :
  Real.sin (angle_A) = 1 :=
by
  sorry

end sin_of_right_angle_l100_100961


namespace proof_statements_BCD_l100_100387

variable (a b : ℝ)

theorem proof_statements_BCD (h1 : a > b) (h2 : b > 0) :
  (-1 / b < -1 / a) ∧ (a^2 * b > a * b^2) ∧ (a / b > b / a) :=
by
  sorry

end proof_statements_BCD_l100_100387


namespace solution_set_of_inequality_l100_100874

-- Definitions for the problem
def inequality (x : ℝ) : Prop := (1 + x) * (2 - x) * (3 + x^2) > 0

-- Statement of the theorem
theorem solution_set_of_inequality :
  {x : ℝ | inequality x} = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_of_inequality_l100_100874


namespace negation_universal_prop_l100_100735

theorem negation_universal_prop:
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
  sorry

end negation_universal_prop_l100_100735


namespace distinct_real_roots_range_l100_100651

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem distinct_real_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 2 * x + a = 0) ∧ (y^2 - 2 * y + a = 0))
  ↔ a < 1 := 
by
  sorry

end distinct_real_roots_range_l100_100651


namespace clownfish_in_display_tank_l100_100911

theorem clownfish_in_display_tank (C B : ℕ) (h1 : C = B) (h2 : C + B = 100) : 
  (B - 26 - (B - 26) / 3) = 16 := by
  sorry

end clownfish_in_display_tank_l100_100911


namespace problem_statement_l100_100973

theorem problem_statement :
  ∃ (w x y z : ℕ), (2^w * 3^x * 5^y * 7^z = 588) ∧ (2 * w + 3 * x + 5 * y + 7 * z = 21) :=
by
  sorry

end problem_statement_l100_100973


namespace quadratic_graph_nature_l100_100702

theorem quadratic_graph_nature (a b : Real) (h : a ≠ 0) :
  ∀ (x : Real), (a * x^2 + b * x + (b^2 / (2 * a)) > 0) ∨ (a * x^2 + b * x + (b^2 / (2 * a)) < 0) :=
by
  sorry

end quadratic_graph_nature_l100_100702


namespace correct_quotient_and_remainder_l100_100752

theorem correct_quotient_and_remainder:
  let incorrect_divisor := 47
  let incorrect_quotient := 5
  let incorrect_remainder := 8
  let incorrect_dividend := incorrect_divisor * incorrect_quotient + incorrect_remainder
  let correct_dividend := 243
  let correct_divisor := 74
  (correct_dividend / correct_divisor = 3 ∧ correct_dividend % correct_divisor = 21) :=
by sorry

end correct_quotient_and_remainder_l100_100752


namespace geometric_seq_sum_l100_100056

theorem geometric_seq_sum :
  ∀ (a : ℕ → ℤ) (q : ℤ), 
    (∀ n, a (n + 1) = a n * q) ∧ 
    (a 4 + a 7 = 2) ∧ 
    (a 5 * a 6 = -8) → 
    a 1 + a 10 = -7 := 
by sorry

end geometric_seq_sum_l100_100056


namespace num_unordered_prime_pairs_summing_to_50_l100_100680

-- Define what it means for an unordered pair of numbers to sum to 50
def sum_to_50 (a b : ℕ) : Prop :=
  a + b = 50

-- Define the condition that the numbers must be prime
def is_unordered_prime_pair (a b : ℕ) : Prop :=
  a ≤ b ∧ Nat.Prime a ∧ Nat.Prime b ∧ sum_to_50 a b

-- Define the set of prime pairs summing to 50
def pairs_summing_to_50 :=
  { p : (ℕ × ℕ) // is_unordered_prime_pair p.1 p.2 }

-- Define the theorem to prove the number of such pairs is 4
theorem num_unordered_prime_pairs_summing_to_50 : 
  finset.card (finset.image (λ (p : (ℕ × ℕ)), (p.1, p.2)) (finset.filter (λ (p : ℕ × ℕ), is_unordered_prime_pair p.1 p.2) ((finset.range 50).product (finset.range 50)))) = 4 :=
  sorry

end num_unordered_prime_pairs_summing_to_50_l100_100680


namespace count_three_digit_integers_with_product_thirty_l100_100948

theorem count_three_digit_integers_with_product_thirty :
  (∃ S : Finset (ℕ × ℕ × ℕ),
      (∀ (a b c : ℕ), (a, b, c) ∈ S → a * b * c = 30 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9) 
    ∧ S.card = 12) :=
by
  sorry

end count_three_digit_integers_with_product_thirty_l100_100948


namespace total_expenditure_l100_100323

-- Definitions of costs and purchases
def bracelet_cost : ℕ := 4
def keychain_cost : ℕ := 5
def coloring_book_cost : ℕ := 3

def paula_bracelets : ℕ := 2
def paula_keychains : ℕ := 1

def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 1

-- Hypothesis stating the total expenditure for Paula and Olive
theorem total_expenditure
  (bracelet_cost keychain_cost coloring_book_cost : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ) :
  paula_bracelets * bracelet_cost + paula_keychains * keychain_cost + olive_coloring_books * coloring_book_cost + olive_bracelets * bracelet_cost = 20 := 
  by
  -- Applying the given costs
  let bracelet_cost := 4
  let keychain_cost := 5
  let coloring_book_cost := 3 

  -- Applying the purchases made by Paula and Olive
  let paula_bracelets := 2
  let paula_keychains := 1
  let olive_coloring_books := 1
  let olive_bracelets := 1

  sorry

end total_expenditure_l100_100323


namespace part1_proof_part2_proof_l100_100557

-- Definitions corresponding to the conditions in a)
variables (a b c BD : ℝ) (A B C : RealAngle)
variables (D : Point) (AD DC : ℝ)

-- Replace the conditions with the necessary hypotheses
hypothesis h1 : b^2 = a * c
hypothesis h2 : BD * sin B = a * sin C
hypothesis h3 : AD = 2 * DC

noncomputable def Part1 : Prop := BD = b

theorem part1_proof : Part1 a b c BD A B C do
  sorry

noncomputable def Part2 : Prop := cos B = 7 / 12

theorem part2_proof (hADDC : AD = 2 * DC) : Part2 A B C.
  sorry

end part1_proof_part2_proof_l100_100557


namespace abs_neg_three_eq_three_l100_100291

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l100_100291


namespace economic_model_l100_100902

theorem economic_model :
  ∃ (Q_s : ℝ → ℝ) (T t_max T_max : ℝ),
  (∀ P : ℝ, Q_d P = 688 - 4 * P) ∧
  (∀ P_e Q_e : ℝ, 1.5 * (4 * P_e / Q_e) = (Q_s'.eval P_e / Q_e)) ∧
  (Q_s 64 = 72) ∧
  (∀ P : ℝ, Q_s P = 6 * P - 312) ∧
  (T = 6480) ∧
  (t_max = 60) ∧
  (T_max = 8640)
where 
  Q_d: ℝ → ℝ := λ P, 688 - 4 * P
  Q_s'.eval : ℝ → ℝ := sorry

end economic_model_l100_100902


namespace allowable_rectangular_formations_count_l100_100901

theorem allowable_rectangular_formations_count (s t f : ℕ) 
  (h1 : s * t = 240)
  (h2 : Nat.Prime s)
  (h3 : 8 ≤ t ∧ t ≤ 30)
  (h4 : f ≤ 8)
  : f = 0 :=
sorry

end allowable_rectangular_formations_count_l100_100901


namespace anglet_angle_measurement_l100_100333

-- Definitions based on conditions
def anglet_measurement := 1
def sixth_circle_degrees := 360 / 6
def anglets_in_sixth_circle := 6000

-- Lean theorem statement proving the implied angle measurement
theorem anglet_angle_measurement (one_percent : Real := 0.01) :
  (anglets_in_sixth_circle * one_percent * sixth_circle_degrees) = anglet_measurement * 60 := 
  sorry

end anglet_angle_measurement_l100_100333


namespace biff_break_even_hours_l100_100627

theorem biff_break_even_hours :
  let ticket := 11
  let drinks_snacks := 3
  let headphones := 16
  let expenses := ticket + drinks_snacks + headphones
  let hourly_income := 12
  let hourly_wifi_cost := 2
  let net_income_per_hour := hourly_income - hourly_wifi_cost
  expenses / net_income_per_hour = 3 :=
by
  sorry

end biff_break_even_hours_l100_100627


namespace negation_of_proposition_p_is_false_l100_100811

variable (p : Prop)

theorem negation_of_proposition_p_is_false
  (h : ¬p) : ¬(¬p) :=
by
  sorry

end negation_of_proposition_p_is_false_l100_100811


namespace temperature_relationship_l100_100313

def temperature (t : ℕ) (T : ℕ) :=
  ∀ t < 10, T = 7 * t + 30

-- Proof not required, hence added sorry.
theorem temperature_relationship (t : ℕ) (T : ℕ) (h : t < 10) :
  temperature t T :=
by {
  sorry
}

end temperature_relationship_l100_100313


namespace value_of_a_l100_100066

theorem value_of_a (M : Set ℝ) (N : Set ℝ) (a : ℝ) 
  (hM : M = {-1, 0, 1, 2}) (hN : N = {x | x^2 - a * x < 0}) 
  (hIntersect : M ∩ N = {1, 2}) : 
  a = 3 := 
sorry

end value_of_a_l100_100066


namespace rose_bushes_in_park_l100_100878

theorem rose_bushes_in_park (current_bushes : ℕ) (newly_planted : ℕ) (h1 : current_bushes = 2) (h2 : newly_planted = 4) : current_bushes + newly_planted = 6 :=
by
  sorry

end rose_bushes_in_park_l100_100878


namespace find_abs_3h_minus_4k_l100_100076

theorem find_abs_3h_minus_4k
  (h k : ℤ)
  (factor1_eq_zero : 3 * (-3)^3 - h * (-3) - 3 * k = 0)
  (factor2_eq_zero : 3 * 2^3 - h * 2 - 3 * k = 0) :
  |3 * h - 4 * k| = 615 :=
by
  sorry

end find_abs_3h_minus_4k_l100_100076


namespace find_k_l100_100403

theorem find_k
  (k x1 x2 : ℝ)
  (h1 : x1^2 - 3*x1 + k = 0)
  (h2 : x2^2 - 3*x2 + k = 0)
  (h3 : x1 = 2 * x2) :
  k = 2 :=
sorry

end find_k_l100_100403


namespace total_germs_calculation_l100_100554

def number_of_dishes : ℕ := 10800
def germs_per_dish : ℕ := 500
def total_germs : ℕ := 5400000

theorem total_germs_calculation : germs_per_ddish * number_of_idshessh = total_germs :=
by sorry

end total_germs_calculation_l100_100554


namespace part_1_part_2_l100_100559

variables {A B C D : Type}
variables {a b c : ℝ} -- Side lengths of the triangle
variables {A_angle B_angle C_angle : ℝ} -- Angles of the triangle
variables {R : ℝ} -- Circumradius of the triangle

-- Assuming the given conditions:
axiom b_squared_eq_ac : b^2 = a * c
axiom bd_sin_eq_a_sin_C : ∀ {BD : ℝ}, BD * sin B_angle = a * sin C_angle
axiom ad_eq_2dc : ∀ {AD DC : ℝ}, AD = 2 * DC

-- Theorems to prove:
theorem part_1 (BD : ℝ) : BD * sin B_angle = a * sin C_angle → BD = b := by
  intros h
  sorry

theorem part_2 (AD DC : ℝ) (H : AD = 2 * DC) : cos B_angle = 7 / 12 := by
  intros h
  sorry

end part_1_part_2_l100_100559


namespace no_solution_exists_l100_100798

open Nat

theorem no_solution_exists : ¬ ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 2 ^ x + 3 ^ y - 5 ^ z = 2 * 11 :=
by
  sorry

end no_solution_exists_l100_100798


namespace blue_eyed_kitten_percentage_is_correct_l100_100978

def total_blue_eyed_kittens : ℕ := 5 + 6 + 4 + 7 + 3

def total_kittens : ℕ := 12 + 16 + 11 + 19 + 12

def percentage_blue_eyed_kittens (blue : ℕ) (total : ℕ) : ℚ := (blue : ℚ) / (total : ℚ) * 100

theorem blue_eyed_kitten_percentage_is_correct :
  percentage_blue_eyed_kittens total_blue_eyed_kittens total_kittens = 35.71 := sorry

end blue_eyed_kitten_percentage_is_correct_l100_100978


namespace repeating_decimal_to_fraction_l100_100478

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l100_100478


namespace sherman_drives_nine_hours_a_week_l100_100418

-- Define the daily commute time in minutes.
def daily_commute_time := 30 + 30

-- Define the number of weekdays Sherman commutes.
def weekdays := 5

-- Define the weekly commute time in minutes.
def weekly_commute_time := weekdays * daily_commute_time

-- Define the conversion from minutes to hours.
def minutes_to_hours (m : ℕ) : ℕ := m / 60

-- Define the weekend driving time in hours.
def weekend_driving_time := 2 * 2

-- Define the total weekly driving time in hours.
def total_weekly_driving_time := minutes_to_hours weekly_commute_time + weekend_driving_time

-- The theorem we need to prove
theorem sherman_drives_nine_hours_a_week :
  total_weekly_driving_time = 9 :=
by
  sorry

end sherman_drives_nine_hours_a_week_l100_100418


namespace four_digit_palindromic_squares_with_different_middle_digits_are_zero_l100_100235

theorem four_digit_palindromic_squares_with_different_middle_digits_are_zero :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (∃ k, k * k = n) ∧ (∃ a b, n = 1001 * a + 110 * b) → a ≠ b → false :=
by sorry

end four_digit_palindromic_squares_with_different_middle_digits_are_zero_l100_100235


namespace ball_placement_count_l100_100933

-- Definitions for the balls and their numbering
inductive Ball
| b1
| b2
| b3
| b4

-- Definitions for the boxes and their numbering
inductive Box
| box1
| box2
| box3

-- Function that checks if an assignment is valid given the conditions.
def isValidAssignment (assignment : Ball → Box) : Prop :=
  assignment Ball.b1 ≠ Box.box1 ∧ assignment Ball.b3 ≠ Box.box3

-- Main statement to prove
theorem ball_placement_count : 
  ∃ (assignments : Finset (Ball → Box)), 
    (∀ f ∈ assignments, isValidAssignment f) ∧ assignments.card = 14 := 
sorry

end ball_placement_count_l100_100933


namespace odd_function_iff_a2_b2_zero_l100_100975

noncomputable def f (x a b : ℝ) : ℝ := x * |x - a| + b

theorem odd_function_iff_a2_b2_zero {a b : ℝ} :
  (∀ x, f x a b = - f (-x) a b) ↔ a^2 + b^2 = 0 := by
  sorry

end odd_function_iff_a2_b2_zero_l100_100975


namespace smallest_n_divisible_by_13_l100_100802

theorem smallest_n_divisible_by_13 : ∃ (n : ℕ), 5^n + n^5 ≡ 0 [MOD 13] ∧ ∀ (m : ℕ), m < n → ¬(5^m + m^5 ≡ 0 [MOD 13]) :=
sorry

end smallest_n_divisible_by_13_l100_100802


namespace interest_rate_per_annum_l100_100527

theorem interest_rate_per_annum (P A : ℝ) (T : ℝ)
  (principal_eq : P = 973.913043478261)
  (amount_eq : A = 1120)
  (time_eq : T = 3):
  (A - P) / (T * P) * 100 = 5 := 
by 
  sorry

end interest_rate_per_annum_l100_100527


namespace repeating_decimal_fraction_l100_100461

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l100_100461


namespace system1_solution_l100_100422

theorem system1_solution (x y : ℝ) (h₁ : x = 2 * y) (h₂ : 3 * x - 2 * y = 8) : x = 4 ∧ y = 2 := 
by admit

end system1_solution_l100_100422


namespace inequality_proof_l100_100954

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b + c) / (2 * a) + (c + a) / (2 * b) + (a + b) / (2 * c) ≥ (2 * a) / (b + c) + (2 * b) / (c + a) + (2 * c) / (a + b) :=
by
  sorry

end inequality_proof_l100_100954


namespace fewest_number_of_gymnasts_l100_100330

theorem fewest_number_of_gymnasts (n : ℕ) (h : n % 2 = 0)
  (handshakes : ∀ (n : ℕ), (n * (n - 1) / 2) + n = 465) : 
  n = 30 :=
by
  sorry

end fewest_number_of_gymnasts_l100_100330


namespace cubic_root_expression_l100_100106

theorem cubic_root_expression (u v w : ℂ) (huvwx : u * v * w ≠ 0)
  (h1 : u^3 - 6 * u^2 + 11 * u - 6 = 0)
  (h2 : v^3 - 6 * v^2 + 11 * v - 6 = 0)
  (h3 : w^3 - 6 * w^2 + 11 * w - 6 = 0) :
  (u * v / w) + (v * w / u) + (w * u / v) = 49 / 6 :=
sorry

end cubic_root_expression_l100_100106


namespace all_three_items_fans_l100_100801

theorem all_three_items_fans 
  (h1 : ∀ n, n = 4800 % 80 → n = 0)
  (h2 : ∀ n, n = 4800 % 40 → n = 0)
  (h3 : ∀ n, n = 4800 % 60 → n = 0)
  (h4 : ∀ n, n = 4800):
  (∃ k, k = 20):=
by
  sorry

end all_three_items_fans_l100_100801


namespace prime_squares_between_5000_and_10000_l100_100668

open Nat

theorem prime_squares_between_5000_and_10000 :
  (finset.filter prime (finset.Ico 71 100)).card = 6 :=
by
  sorry

end prime_squares_between_5000_and_10000_l100_100668


namespace sequence_difference_l100_100737

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

theorem sequence_difference (hS : ∀ n, S n = n^2 - 5 * n)
                            (hna : ∀ n, a n = S n - S (n - 1))
                            (hpq : p - q = 4) :
                            a p - a q = 8 := by
    sorry

end sequence_difference_l100_100737


namespace both_false_of_not_or_l100_100214

-- Define propositions p and q
variables (p q : Prop)

-- The condition given: ¬(p ∨ q)
theorem both_false_of_not_or (h : ¬(p ∨ q)) : ¬ p ∧ ¬ q :=
by {
  sorry
}

end both_false_of_not_or_l100_100214


namespace cuboid_breadth_l100_100351

theorem cuboid_breadth (l h A : ℝ) (w : ℝ) :
  l = 8 ∧ h = 12 ∧ A = 960 → 2 * (l * w + l * h + w * h) = A → w = 19.2 :=
by
  intros h1 h2
  sorry

end cuboid_breadth_l100_100351


namespace arrangement_count_27_arrangement_count_26_l100_100385

open Int

def valid_arrangement_count (n : ℕ) : ℕ :=
  if n = 27 then 14 else if n = 26 then 105 else 0

theorem arrangement_count_27 : valid_arrangement_count 27 = 14 :=
  by
    sorry

theorem arrangement_count_26 : valid_arrangement_count 26 = 105 :=
  by
    sorry

end arrangement_count_27_arrangement_count_26_l100_100385


namespace original_volume_l100_100144

variable {π : Real} (r h : Real)

theorem original_volume (hπ : π ≠ 0) (hr : r ≠ 0) (hh : h ≠ 0) (condition : 3 * π * r^2 * h = 180) : π * r^2 * h = 60 := by
  sorry

end original_volume_l100_100144


namespace yearly_profit_l100_100098

variable (num_subletters : ℕ) (rent_per_subletter_per_month rent_per_month : ℕ)

theorem yearly_profit (h1 : num_subletters = 3)
                     (h2 : rent_per_subletter_per_month = 400)
                     (h3 : rent_per_month = 900) :
  12 * (num_subletters * rent_per_subletter_per_month - rent_per_month) = 3600 :=
by
  sorry

end yearly_profit_l100_100098


namespace find_m_interval_l100_100516

-- Define the sequence recursively
def sequence_recursive (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x 0 = 5 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 5 * x n + 4) / (x n + 6)

-- The left-hand side of the inequality
noncomputable def target_value : ℝ := 4 + 1 / (2 ^ 20)

-- The condition that the sequence element must satisfy
def condition (x : ℕ → ℝ) (m : ℕ) : Prop :=
  x m ≤ target_value

-- The proof problem statement, m lies within the given interval
theorem find_m_interval (x : ℕ → ℝ) (m : ℕ) :
  sequence_recursive x n →
  condition x m →
  81 ≤ m ∧ m ≤ 242 :=
sorry

end find_m_interval_l100_100516


namespace mrs_hilt_money_l100_100843

-- Definitions and given conditions
def cost_of_pencil := 5  -- in cents
def number_of_pencils := 10

-- The theorem we need to prove
theorem mrs_hilt_money : cost_of_pencil * number_of_pencils = 50 := by
  sorry

end mrs_hilt_money_l100_100843


namespace abs_neg_three_l100_100295

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l100_100295


namespace four_times_remaining_marbles_l100_100850

theorem four_times_remaining_marbles (initial total_given : ℕ) (remaining : ℕ := initial - total_given) :
  initial = 500 → total_given = 4 * 80 → 4 * remaining = 720 := by sorry

end four_times_remaining_marbles_l100_100850


namespace total_duration_of_running_l100_100898

-- Definition of conditions
def constant_speed_1 : ℝ := 18
def constant_time_1 : ℝ := 3
def next_distance : ℝ := 70
def average_speed_2 : ℝ := 14

-- Proof statement
theorem total_duration_of_running : 
    let distance_1 := constant_speed_1 * constant_time_1
    let time_2 := next_distance / average_speed_2
    distance_1 = 54 ∧ time_2 = 5 → (constant_time_1 + time_2 = 8) :=
sorry

end total_duration_of_running_l100_100898


namespace abs_neg_three_l100_100272

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l100_100272


namespace inequality_proof_l100_100935

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  (1 + 1/x) * (1 + 1/y) ≥ 4 :=
sorry

end inequality_proof_l100_100935


namespace collinear_vectors_m_n_sum_l100_100371

theorem collinear_vectors_m_n_sum (m n : ℕ)
  (h1 : (2, 3, m) = (2 * n, 6, 8)) :
  m + n = 6 :=
sorry

end collinear_vectors_m_n_sum_l100_100371


namespace sin_2x_plus_one_equals_9_over_5_l100_100212

theorem sin_2x_plus_one_equals_9_over_5 (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin (2 * x) + 1 = 9 / 5 :=
sorry

end sin_2x_plus_one_equals_9_over_5_l100_100212


namespace problem_statement_l100_100062

noncomputable def find_sum (x y : ℝ) : ℝ := x + y

theorem problem_statement (x y : ℝ)
  (hx : |x| + x + y = 12)
  (hy : x + |y| - y = 14) :
  find_sum x y = 22 / 5 :=
sorry

end problem_statement_l100_100062


namespace shadow_projection_height_l100_100613

theorem shadow_projection_height :
  ∃ (x : ℝ), (∃ (shadow_area : ℝ), shadow_area = 192) ∧ 1000 * x = 25780 :=
by
  sorry

end shadow_projection_height_l100_100613


namespace average_annual_growth_rate_l100_100583

-- Define the conditions
def revenue_current_year : ℝ := 280
def revenue_planned_two_years : ℝ := 403.2

-- Define the growth equation
def growth_equation (x : ℝ) : Prop :=
  revenue_current_year * (1 + x)^2 = revenue_planned_two_years

-- State the theorem
theorem average_annual_growth_rate : ∃ x : ℝ, growth_equation x ∧ x = 0.2 := by
  sorry

end average_annual_growth_rate_l100_100583


namespace inversely_proportional_x_y_l100_100425

-- Statement of the problem
theorem inversely_proportional_x_y :
  ∃ k : ℝ, (∀ (x y : ℝ), (x * y = k) ∧ (x = 4) ∧ (y = 2) → x * (-5) = -8 / 5) :=
by
  sorry

end inversely_proportional_x_y_l100_100425


namespace number_of_factors_l100_100342

theorem number_of_factors : 
  ∃ (count : ℕ), count = 45 ∧
    (∀ n : ℕ, (1 ≤ n ∧ n ≤ 500) → 
      ∃ a b : ℤ, (x - a) * (x - b) = x^2 + 2 * x - n) :=
by
  sorry

end number_of_factors_l100_100342


namespace mixed_number_expression_l100_100204

open Real

-- Definitions of the given mixed numbers
def mixed_number1 : ℚ := (37 / 7)
def mixed_number2 : ℚ := (18 / 5)
def mixed_number3 : ℚ := (19 / 6)
def mixed_number4 : ℚ := (9 / 4)

-- Main theorem statement
theorem mixed_number_expression :
  25 * (mixed_number1 - mixed_number2) / (mixed_number3 + mixed_number4) = 7 + 49 / 91 :=
by
  sorry

end mixed_number_expression_l100_100204


namespace total_goals_l100_100835

-- Definitions
def louie_goals_last_match := 4
def louie_previous_goals := 40
def brother_multiplier := 2
def seasons := 3
def games_per_season := 50

-- Total number of goals scored by Louie and his brother
theorem total_goals : (louie_previous_goals + louie_goals_last_match) 
                      + (brother_multiplier * louie_goals_last_match * seasons * games_per_season) 
                      = 1244 :=
by sorry

end total_goals_l100_100835


namespace toothpaste_runs_out_in_two_days_l100_100443

noncomputable def toothpaste_capacity := 90
noncomputable def dad_usage_per_brushing := 4
noncomputable def mom_usage_per_brushing := 3
noncomputable def anne_usage_per_brushing := 2
noncomputable def brother_usage_per_brushing := 1
noncomputable def sister_usage_per_brushing := 1

noncomputable def dad_brushes_per_day := 4
noncomputable def mom_brushes_per_day := 4
noncomputable def anne_brushes_per_day := 4
noncomputable def brother_brushes_per_day := 4
noncomputable def sister_brushes_per_day := 2

noncomputable def total_daily_usage :=
  dad_usage_per_brushing * dad_brushes_per_day + 
  mom_usage_per_brushing * mom_brushes_per_day + 
  anne_usage_per_brushing * anne_brushes_per_day + 
  brother_usage_per_brushing * brother_brushes_per_day + 
  sister_usage_per_brushing * sister_brushes_per_day

theorem toothpaste_runs_out_in_two_days :
  toothpaste_capacity / total_daily_usage = 2 := by
  -- Proof omitted
  sorry

end toothpaste_runs_out_in_two_days_l100_100443


namespace bus_stop_time_l100_100642

noncomputable def time_stopped_per_hour (excl_speed incl_speed : ℕ) : ℕ :=
  60 * (excl_speed - incl_speed) / excl_speed

theorem bus_stop_time:
  time_stopped_per_hour 54 36 = 20 :=
by
  sorry

end bus_stop_time_l100_100642


namespace sum_of_powers_l100_100259

theorem sum_of_powers (m n : ℤ)
  (h1 : m + n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 4)
  (h4 : m^4 + n^4 = 7)
  (h5 : m^5 + n^5 = 11) :
  m^9 + n^9 = 76 :=
sorry

end sum_of_powers_l100_100259


namespace perpendicular_vectors_l100_100364

/-- Given vectors a and b, prove that m = 6 if a is perpendicular to b -/
theorem perpendicular_vectors {m : ℝ} (h₁ : (1, 5, -2) = (1, 5, -2)) (h₂ : ∃ m : ℝ, (m, 2, m+2) = (m, 2, m+2)) (h₃ : (1 * m + 5 * 2 + (-2) * (m + 2) = 0)) :
  m = 6 :=
sorry

end perpendicular_vectors_l100_100364


namespace tim_same_age_tina_l100_100590

-- Define the ages of Tim and Tina
variables (x y : ℤ)

-- Given conditions
def condition_tim := x + 2 = 2 * (x - 2)
def condition_tina := y + 3 = 3 * (y - 3)

-- The goal is to prove that Tim is the same age as Tina
theorem tim_same_age_tina (htim : condition_tim x) (htina : condition_tina y) : x = y :=
by 
  sorry

end tim_same_age_tina_l100_100590


namespace rate_is_900_l100_100435

noncomputable def rate_per_square_meter (L W : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (L * W)

theorem rate_is_900 :
  rate_per_square_meter 5 4.75 21375 = 900 := by
  sorry

end rate_is_900_l100_100435


namespace area_of_fourth_rectangle_l100_100166

theorem area_of_fourth_rectangle
  (A1 A2 A3 A_total : ℕ)
  (h1 : A1 = 24)
  (h2 : A2 = 30)
  (h3 : A3 = 18)
  (h_total : A_total = 100) :
  ∃ A4 : ℕ, A1 + A2 + A3 + A4 = A_total ∧ A4 = 28 :=
by
  sorry

end area_of_fourth_rectangle_l100_100166


namespace pastries_made_initially_l100_100912

theorem pastries_made_initially 
  (sold : ℕ) (remaining : ℕ) (initial : ℕ) 
  (h1 : sold = 103) (h2 : remaining = 45) : 
  initial = 148 :=
by
  have h := h1
  have r := h2
  sorry

end pastries_made_initially_l100_100912


namespace find_k_l100_100662

theorem find_k (x y k : ℝ) (h1 : x + 2 * y = k + 1) (h2 : 2 * x + y = 1) (h3 : x + y = 3) : k = 7 :=
by
  sorry

end find_k_l100_100662


namespace percentage_of_50_l100_100024

theorem percentage_of_50 (P : ℝ) :
  (0.10 * 30) + (P / 100 * 50) = 10.5 → P = 15 := by
  sorry

end percentage_of_50_l100_100024


namespace range_x_when_p_and_q_m_eq_1_range_m_for_not_p_necessary_not_sufficient_q_l100_100219

-- Define the propositions p and q in terms of x and m
def p (x m : ℝ) : Prop := |2 * x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3 * x) / (x + 2) > 0

-- The range of x for p ∧ q when m = 1
theorem range_x_when_p_and_q_m_eq_1 : {x : ℝ | p x 1 ∧ q x} = {x : ℝ | -2 < x ∧ x ≤ 0} :=
by sorry

-- The range of m where ¬p is a necessary but not sufficient condition for q
theorem range_m_for_not_p_necessary_not_sufficient_q : {m : ℝ | ∀ x, ¬p x m → q x} ∩ {m : ℝ | ∃ x, ¬p x m ∧ q x} = {m : ℝ | -3 ≤ m ∧ m ≤ -1/3} :=
by sorry

end range_x_when_p_and_q_m_eq_1_range_m_for_not_p_necessary_not_sufficient_q_l100_100219


namespace quadratic_sum_roots_l100_100699

theorem quadratic_sum_roots {a b : ℝ}
  (h1 : ∀ x, x^2 - a * x + b < 0 ↔ -1 < x ∧ x < 3) :
  a + b = -1 :=
sorry

end quadratic_sum_roots_l100_100699


namespace find_x_squared_plus_y_squared_l100_100644

theorem find_x_squared_plus_y_squared (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y + x + y = 17) (h4 : x^2 * y + x * y^2 = 72) : x^2 + y^2 = 65 := 
  sorry

end find_x_squared_plus_y_squared_l100_100644


namespace no_blonde_girls_added_l100_100111

-- Initial number of girls
def total_girls : Nat := 80
def initial_blonde_girls : Nat := 30
def black_haired_girls : Nat := 50

-- Number of blonde girls added
def blonde_girls_added : Nat := total_girls - black_haired_girls - initial_blonde_girls

theorem no_blonde_girls_added : blonde_girls_added = 0 :=
by
  sorry

end no_blonde_girls_added_l100_100111


namespace f_const_one_l100_100530

-- Mathematical Translation of the Definitions
variable (f g h : ℕ → ℕ)

-- Given conditions
axiom h_injective : Function.Injective h
axiom g_surjective : Function.Surjective g
axiom f_eq : ∀ n, f n = g n - h n + 1

-- Theorem to Prove
theorem f_const_one : ∀ n, f n = 1 :=
by
  sorry

end f_const_one_l100_100530


namespace initial_time_to_cover_distance_l100_100162

theorem initial_time_to_cover_distance (s t : ℝ) (h1 : 540 = s * t) (h2 : 540 = 60 * (3/4) * t) : t = 12 :=
sorry

end initial_time_to_cover_distance_l100_100162


namespace ages_of_patients_l100_100160

theorem ages_of_patients (x y : ℕ) 
  (h1 : x - y = 44) 
  (h2 : x * y = 1280) : 
  (x = 64 ∧ y = 20) ∨ (x = 20 ∧ y = 64) := by
  sorry

end ages_of_patients_l100_100160


namespace completing_square_transformation_l100_100885

theorem completing_square_transformation : ∀ x : ℝ, x^2 - 4 * x - 7 = 0 → (x - 2)^2 = 11 :=
by
  intros x h
  sorry

end completing_square_transformation_l100_100885


namespace find_number_l100_100904

theorem find_number :
  let s := 2615 + 3895
  let d := 3895 - 2615
  let q := 3 * d
  let x := s * q + 65
  x = 24998465 :=
by
  let s := 2615 + 3895
  let d := 3895 - 2615
  let q := 3 * d
  let x := s * q + 65
  sorry

end find_number_l100_100904


namespace zero_point_interval_l100_100654

variable (f : ℝ → ℝ)
variable (f_deriv : ℝ → ℝ)
variable (e : ℝ)
variable (monotonic_f : MonotoneOn f (Set.Ioi 0))

noncomputable def condition1_property (x : ℝ) (h : 0 < x) : f (f x - Real.log x) = Real.exp 1 + 1 := sorry
noncomputable def derivative_property (x : ℝ) (h : 0 < x) : f_deriv x = (deriv f) x := sorry

theorem zero_point_interval :
  ∃ x ∈ Set.Ioo 1 2, f x - f_deriv x - e = 0 := sorry

end zero_point_interval_l100_100654


namespace count_prime_pairs_sum_50_l100_100673

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a set of prime numbers below 50
noncomputable def primes_below_50 : set ℕ := { n | is_prime n ∧ n < 50 }

-- Define the main theorem statement
theorem count_prime_pairs_sum_50 : 
  (∃! (pairs : finset (ℕ × ℕ)), 
    ∀ pair ∈ pairs, 
      let (p, q) := pair in 
      p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q) := sorry

end count_prime_pairs_sum_50_l100_100673


namespace solve_expression_l100_100934

theorem solve_expression (x y : ℝ) (h : (x + y - 2020) * (2023 - x - y) = 2) :
  (x + y - 2020)^2 * (2023 - x - y)^2 = 4 := by
  sorry

end solve_expression_l100_100934


namespace sum_of_numbers_l100_100159

/-- Given three numbers in the ratio 1:2:5, with the sum of their squares being 4320,
prove that the sum of the numbers is 96. -/

theorem sum_of_numbers (x : ℝ) (h1 : (x:ℝ) = x) (h2 : 2 * x = 2 * x) (h3 : 5 * x = 5 * x) 
  (h4 : x^2 + (2 * x)^2 + (5 * x)^2 = 4320) :
  x + 2 * x + 5 * x = 96 := 
sorry

end sum_of_numbers_l100_100159


namespace toys_produced_each_day_l100_100156

theorem toys_produced_each_day (total_weekly_production : ℕ) (days_worked_per_week : ℕ)
  (same_number_toys_each_day : Prop) : 
  total_weekly_production = 4340 → days_worked_per_week = 2 → 
  same_number_toys_each_day →
  (total_weekly_production / days_worked_per_week = 2170) :=
by
  intros h_production h_days h_same_toys
  -- proof skipped
  sorry

end toys_produced_each_day_l100_100156


namespace M_intersect_N_l100_100234

def M : Set ℝ := {x | 1 + x > 0}
def N : Set ℝ := {x | x < 1}

theorem M_intersect_N : M ∩ N = {x | -1 < x ∧ x < 1} := 
by
  sorry

end M_intersect_N_l100_100234


namespace part1_simplified_part2_value_part3_independent_l100_100363

-- Definitions of A and B
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Proof statement for part 1
theorem part1_simplified (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y :=
by sorry

-- Proof statement for part 2
theorem part2_value (x y : ℝ) (hxy : x + y = 6/7) (hprod : x * y = -1) :
  2 * A x y - 3 * B x y = 17 :=
by sorry

-- Proof statement for part 3
theorem part3_independent (y : ℝ) :
  2 * A (7/11) y - 3 * B (7/11) y = 49/11 :=
by sorry

end part1_simplified_part2_value_part3_independent_l100_100363


namespace abs_neg_three_l100_100285

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l100_100285


namespace abs_neg_three_l100_100297

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l100_100297


namespace perpendicular_lines_l100_100813

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, (a * x - y + 2 * a = 0) → ((2 * a - 1) * x + a * y + a = 0) -> 
  (a ≠ 0 → ∃ k : ℝ, k = (a * ((1 - 2 * a) / a)) ∧ k = -1) -> a * ((1 - 2 * a) / a) = -1) →
  a = 0 ∨ a = 1 := by sorry

end perpendicular_lines_l100_100813


namespace max_consecutive_interesting_numbers_l100_100255

def is_interesting (n : ℕ) : Prop :=
  (n / 100 % 3 = 0) ∨ (n / 10 % 10 % 3 = 0) ∨ (n % 10 % 3 = 0)

theorem max_consecutive_interesting_numbers :
  ∃ l r, 100 ≤ l ∧ r ≤ 999 ∧ r - l + 1 = 122 ∧ (∀ n, l ≤ n ∧ n ≤ r → is_interesting n) ∧ 
  ∀ l' r', 100 ≤ l' ∧ r' ≤ 999 ∧ r' - l' + 1 > 122 → ∃ n, l' ≤ n ∧ n ≤ r' ∧ ¬ is_interesting n := 
sorry

end max_consecutive_interesting_numbers_l100_100255


namespace fraction_of_remaining_supplies_used_l100_100907

theorem fraction_of_remaining_supplies_used 
  (initial_food : ℕ)
  (food_used_first_day_fraction : ℚ)
  (food_remaining_after_three_days : ℕ) 
  (food_used_second_period_fraction : ℚ) :
  initial_food = 400 →
  food_used_first_day_fraction = 2 / 5 →
  food_remaining_after_three_days = 96 →
  (initial_food - initial_food * food_used_first_day_fraction) * (1 - food_used_second_period_fraction) = food_remaining_after_three_days →
  food_used_second_period_fraction = 3 / 5 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_of_remaining_supplies_used_l100_100907


namespace full_tank_cost_l100_100758

-- Definitions from the conditions
def total_liters_given := 36
def total_cost_given := 18
def tank_capacity := 64

-- Hypothesis based on the conditions
def price_per_liter := total_cost_given / total_liters_given

-- Conclusion we need to prove
theorem full_tank_cost: price_per_liter * tank_capacity = 32 :=
  sorry

end full_tank_cost_l100_100758


namespace proof_A_union_B_eq_R_l100_100405

def A : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - 5) < a }

theorem proof_A_union_B_eq_R (a : ℝ) (h : a > 6) : 
  A ∪ B a = Set.univ :=
by {
  sorry
}

end proof_A_union_B_eq_R_l100_100405


namespace percentage_y_less_than_x_l100_100754

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 11 * y) : 
  ((x - y) / x) * 100 = 90.91 := 
by 
  sorry -- proof to be provided separately

end percentage_y_less_than_x_l100_100754


namespace eq_zero_l100_100859

variable {x y z : ℤ}

theorem eq_zero (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end eq_zero_l100_100859


namespace abs_neg_three_eq_three_l100_100292

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l100_100292


namespace sequence_is_increasing_l100_100939

variable (a_n : ℕ → ℝ)

def sequence_positive_numbers (a_n : ℕ → ℝ) : Prop :=
∀ n, 0 < a_n n

def sequence_condition (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n (n + 1) = 2 * a_n n

theorem sequence_is_increasing 
  (h1 : sequence_positive_numbers a_n) 
  (h2 : sequence_condition a_n) : 
  ∀ n, a_n (n + 1) > a_n n :=
by
  sorry

end sequence_is_increasing_l100_100939


namespace James_leftover_money_l100_100706

variable (W : ℝ)
variable (M : ℝ)

theorem James_leftover_money 
  (h1 : M = (W / 2 - 2))
  (h2 : M + 114 = W) : 
  M = 110 := sorry

end James_leftover_money_l100_100706


namespace monotonicity_and_inequality_l100_100538

noncomputable def f (x : ℝ) := 2 * Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) := a * x + 2
noncomputable def F (a : ℝ) (x : ℝ) := f x - g a x

theorem monotonicity_and_inequality (a : ℝ) (x₁ x₂ : ℝ) (hF_nonneg : ∀ x, F a x ≥ 0) (h_lt : x₁ < x₂) :
  (F a x₂ - F a x₁) / (x₂ - x₁) > 2 * (Real.exp x₁ - 1) :=
sorry

end monotonicity_and_inequality_l100_100538


namespace find_angle_B_in_right_triangle_l100_100552

theorem find_angle_B_in_right_triangle (A B C : ℝ) (hC : C = 90) (hA : A = 35) :
  B = 55 :=
by
  -- Assuming A, B, and C represent the three angles of a triangle ABC
  -- where C = 90 degrees and A = 35 degrees, we need to prove B = 55 degrees.
  sorry

end find_angle_B_in_right_triangle_l100_100552


namespace abs_neg_three_l100_100278

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l100_100278


namespace part1_part2_l100_100945

def y (x : ℝ) : ℝ := -x^2 + 8*x - 7

-- Part (1) Lean statement
theorem part1 : ∀ x : ℝ, x < 4 → y x < y (x + 1) := sorry

-- Part (2) Lean statement
theorem part2 : ∀ x : ℝ, (x < 1 ∨ x > 7) → y x < 0 := sorry

end part1_part2_l100_100945


namespace mints_ratio_l100_100762

theorem mints_ratio (n : ℕ) (green_mints red_mints : ℕ) (h1 : green_mints + red_mints = n) (h2 : green_mints = 3 * (n / 4)) : green_mints / red_mints = 3 :=
by
  sorry

end mints_ratio_l100_100762


namespace find_a_plus_b_l100_100388

theorem find_a_plus_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a^2 - b^4 = 2009) : a + b = 47 := 
by 
  sorry

end find_a_plus_b_l100_100388


namespace sqrt49_times_sqrt25_eq_5sqrt7_l100_100188

noncomputable def sqrt49_times_sqrt25 : ℝ :=
  Real.sqrt (49 * Real.sqrt 25)

theorem sqrt49_times_sqrt25_eq_5sqrt7 :
  sqrt49_times_sqrt25 = 5 * Real.sqrt 7 :=
by
sorry

end sqrt49_times_sqrt25_eq_5sqrt7_l100_100188


namespace number_of_cows_l100_100717

/-- 
The number of cows Mr. Reyansh has on his dairy farm 
given the conditions of water consumption and total water used in a week. 
-/
theorem number_of_cows (C : ℕ) 
  (h1 : ∀ (c : ℕ), (c = 80 * 7))
  (h2 : ∀ (s : ℕ), (s = 10 * C))
  (h3 : ∀ (d : ℕ), (d = 20 * 7))
  (h4 : 1960 * C = 78400) : 
  C = 40 :=
sorry

end number_of_cows_l100_100717


namespace degree_product_l100_100268

-- Define the degrees of the polynomials p and q
def degree_p : ℕ := 3
def degree_q : ℕ := 4

-- Define the functions p(x) and q(x) as polynomials and their respective degrees
axiom degree_p_definition (p : Polynomial ℝ) : p.degree = degree_p
axiom degree_q_definition (q : Polynomial ℝ) : q.degree = degree_q

-- Define the degree of the product p(x^2) * q(x^4)
noncomputable def degree_p_x2_q_x4 (p q : Polynomial ℝ) : ℕ :=
  2 * degree_p + 4 * degree_q

-- Prove that the degree of p(x^2) * q(x^4) is 22
theorem degree_product (p q : Polynomial ℝ) (hp : p.degree = degree_p) (hq : q.degree = degree_q) :
  degree_p_x2_q_x4 p q = 22 :=
by
  sorry

end degree_product_l100_100268


namespace find_f_of_4_l100_100058

noncomputable def power_function (x : ℝ) (α : ℝ) : ℝ := x^α

theorem find_f_of_4 :
  (∃ α : ℝ, power_function 3 α = Real.sqrt 3) →
  power_function 4 (1/2) = 2 :=
by
  sorry

end find_f_of_4_l100_100058


namespace perfect_number_divisibility_l100_100168

theorem perfect_number_divisibility (P : ℕ) (h1 : P > 28) (h2 : Nat.Perfect P) (h3 : 7 ∣ P) : 49 ∣ P := 
sorry

end perfect_number_divisibility_l100_100168


namespace count_prime_pairs_summing_to_50_l100_100681

-- Definition: List of primes less than 25
def primes_below_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Condition: Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Property: List of valid prime pairs summing to 50
def prime_pairs_summing_to (n : ℕ) (ps : List ℕ) : List (ℕ × ℕ) :=
  (ps.product ps).filter (λ (p, q), p + q = n ∧ is_prime q)

-- Theorem: There are exactly 4 pairs of primes summing up to 50
theorem count_prime_pairs_summing_to_50 :
  (prime_pairs_summing_to 50 primes_below_25).length = 4 :=
by {
  -- Proof is skipped as per instructions
  sorry
}

end count_prime_pairs_summing_to_50_l100_100681


namespace number_of_teams_l100_100080

-- Define the statement representing the problem and conditions
theorem number_of_teams (n : ℕ) (h : 2 * n * (n - 1) = 9800) : n = 50 :=
sorry

end number_of_teams_l100_100080


namespace given_problem_l100_100069

theorem given_problem (x y : ℝ) (hx : x ≠ 0) (hx4 : x ≠ 4) (hy : y ≠ 0) (hy6 : y ≠ 6) :
  (2 / x + 3 / y = 1 / 2) ↔ (4 * y / (y - 6) = x) :=
sorry

end given_problem_l100_100069


namespace percentage_books_not_sold_l100_100490

theorem percentage_books_not_sold :
    let initial_stock := 700
    let books_sold_mon := 50
    let books_sold_tue := 82
    let books_sold_wed := 60
    let books_sold_thu := 48
    let books_sold_fri := 40
    let total_books_sold := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri 
    let books_not_sold := initial_stock - total_books_sold
    let percentage_not_sold := (books_not_sold * 100) / initial_stock
    percentage_not_sold = 60 :=
by
  -- definitions
  let initial_stock := 700
  let books_sold_mon := 50
  let books_sold_tue := 82
  let books_sold_wed := 60
  let books_sold_thu := 48
  let books_sold_fri := 40
  let total_books_sold := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri
  let books_not_sold := initial_stock - total_books_sold
  let percentage_not_sold := (books_not_sold * 100) / initial_stock
  have : percentage_not_sold = 60 := sorry
  exact this

end percentage_books_not_sold_l100_100490


namespace bus_speed_excluding_stoppages_l100_100926

theorem bus_speed_excluding_stoppages (v : ℝ) (stoppage_time : ℝ) (speed_incl_stoppages : ℝ) :
  stoppage_time = 15 / 60 ∧ speed_incl_stoppages = 48 → v = 64 :=
by
  intro h
  sorry

end bus_speed_excluding_stoppages_l100_100926


namespace problem_part1_problem_part2_l100_100252

-- Define the set A and the property it satisfies
variable (A : Set ℝ)
variable (H : ∀ a ∈ A, (1 + a) / (1 - a) ∈ A)

-- Suppose 2 is in A
theorem problem_part1 (h : 2 ∈ A) : A = {2, -3, -1 / 2, 1 / 3} :=
sorry

-- Prove the conjecture based on the elements of A found in part 1
theorem problem_part2 (h : 2 ∈ A) (hA : A = {2, -3, -1 / 2, 1 / 3}) :
  ¬ (0 ∈ A ∨ 1 ∈ A ∨ -1 ∈ A) ∧
  (2 * (-1 / 2) = -1 ∧ -3 * (1 / 3) = -1) :=
sorry

end problem_part1_problem_part2_l100_100252


namespace cindy_marbles_problem_l100_100849

theorem cindy_marbles_problem
  (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 500) (h2 : friends = 4) (h3 : marbles_per_friend = 80) :
  4 * (initial_marbles - (marbles_per_friend * friends)) = 720 :=
by
  sorry

end cindy_marbles_problem_l100_100849


namespace coordinates_C_l100_100014

theorem coordinates_C 
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) 
  (hA : A = (-1, 3)) 
  (hB : B = (11, 7))
  (hBC_AB : (C.1 - B.1, C.2 - B.2) = (2 / 3) • (B.1 - A.1, B.2 - A.2)) :
  C = (19, 29 / 3) :=
sorry

end coordinates_C_l100_100014


namespace average_of_scores_with_average_twice_l100_100331

variable (scores: List ℝ) (A: ℝ) (A': ℝ)
variable (h1: scores.length = 50)
variable (h2: A = (scores.sum) / 50)
variable (h3: A' = ((scores.sum + 2 * A) / 52))

theorem average_of_scores_with_average_twice (h1: scores.length = 50) (h2: A = (scores.sum) / 50) (h3: A' = ((scores.sum + 2 * A) / 52)) :
  A' = A :=
by
  sorry

end average_of_scores_with_average_twice_l100_100331


namespace sherman_drives_nine_hours_a_week_l100_100420

-- Define the daily commute time in minutes.
def daily_commute_time := 30 + 30

-- Define the number of weekdays Sherman commutes.
def weekdays := 5

-- Define the weekly commute time in minutes.
def weekly_commute_time := weekdays * daily_commute_time

-- Define the conversion from minutes to hours.
def minutes_to_hours (m : ℕ) : ℕ := m / 60

-- Define the weekend driving time in hours.
def weekend_driving_time := 2 * 2

-- Define the total weekly driving time in hours.
def total_weekly_driving_time := minutes_to_hours weekly_commute_time + weekend_driving_time

-- The theorem we need to prove
theorem sherman_drives_nine_hours_a_week :
  total_weekly_driving_time = 9 :=
by
  sorry

end sherman_drives_nine_hours_a_week_l100_100420


namespace cos_triple_angle_l100_100691

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l100_100691


namespace product_floor_ceil_sequence_l100_100925

theorem product_floor_ceil_sequence :
  (Int.floor (-6 - 0.5) * Int.ceil (6 + 0.5) *
   Int.floor (-5 - 0.5) * Int.ceil (5 + 0.5) *
   Int.floor (-4 - 0.5) * Int.ceil (4 + 0.5) *
   Int.floor (-3 - 0.5) * Int.ceil (3 + 0.5) *
   Int.floor (-2 - 0.5) * Int.ceil (2 + 0.5) *
   Int.floor (-1 - 0.5) * Int.ceil (1 + 0.5) *
   Int.floor (-0.5) * Int.ceil (0.5)) = -25401600 := sorry

end product_floor_ceil_sequence_l100_100925


namespace jean_total_cost_l100_100398

theorem jean_total_cost 
  (num_pants : ℕ)
  (original_price_per_pant : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (num_pants_eq : num_pants = 10)
  (original_price_per_pant_eq : original_price_per_pant = 45)
  (discount_rate_eq : discount_rate = 0.2)
  (tax_rate_eq : tax_rate = 0.1) : 
  ∃ total_cost : ℝ, total_cost = 396 :=
by
  sorry

end jean_total_cost_l100_100398


namespace length_of_XY_l100_100101

theorem length_of_XY (A B C D P Q X Y : ℝ) (h₁ : A = B) (h₂ : C = D) 
  (h₃ : A + B = 13) (h₄ : C + D = 21) (h₅ : A + P = 7) 
  (h₆ : C + Q = 8) (h₇ : P ≠ Q) (h₈ : P + Q = 30) :
  ∃ k : ℝ, XY = 2 * k + 30 + 31 / 15 :=
by sorry

end length_of_XY_l100_100101


namespace total_weight_of_rings_l100_100709

-- Define the weights of the rings
def weight_orange : Real := 0.08
def weight_purple : Real := 0.33
def weight_white : Real := 0.42
def weight_blue : Real := 0.59
def weight_red : Real := 0.24
def weight_green : Real := 0.16

-- Define the total weight of the rings
def total_weight : Real :=
  weight_orange + weight_purple + weight_white + weight_blue + weight_red + weight_green

-- The task is to prove that the total weight equals 1.82
theorem total_weight_of_rings : total_weight = 1.82 := 
  by
    sorry

end total_weight_of_rings_l100_100709


namespace football_daily_practice_hours_l100_100319

-- Define the total practice hours and the days missed.
def total_hours := 30
def days_missed := 1
def days_in_week := 7

-- Calculate the number of days practiced.
def days_practiced := days_in_week - days_missed

-- Define the daily practice hours.
def daily_practice_hours := total_hours / days_practiced

-- State the proposition.
theorem football_daily_practice_hours :
  daily_practice_hours = 5 := sorry

end football_daily_practice_hours_l100_100319


namespace largest_base_b_digits_not_18_l100_100152

-- Definition of the problem:
-- Let n = 12^3 in base 10
def n : ℕ := 12 ^ 3

-- Definition of the conditions:
-- In base 8, 1728 (12^3 in base 10) has its digits sum to 17
def sum_of_digits_base_8 (x : ℕ) : ℕ :=
  let digits := x.digits (8)
  digits.sum

-- Proof statement
theorem largest_base_b_digits_not_18 : ∃ b : ℕ, (max b) = 8 ∧ sum_of_digits_base_8 n ≠ 18 := by
  sorry

end largest_base_b_digits_not_18_l100_100152


namespace find_m_n_l100_100796

theorem find_m_n (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) 
  (h1 : m * n ∣ 3 ^ m + 1) (h2 : m * n ∣ 3 ^ n + 1) : 
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) :=
by
  sorry

end find_m_n_l100_100796


namespace complement_of_A_in_U_l100_100379

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}
def complementA : Set ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = complementA :=
by
  sorry

end complement_of_A_in_U_l100_100379


namespace mens_wages_l100_100161

-- Definitions from the conditions.
variables (men women boys total_earnings : ℕ) (wage : ℚ)
variable (equivalence : 5 * men = 8 * boys)
variable (totalEarnings : total_earnings = 120)

-- The final statement to prove the men's wages.
theorem mens_wages (h_eq : 5 = 5) : wage = 46.15 :=
by
  sorry

end mens_wages_l100_100161


namespace non_honda_red_percentage_l100_100079

-- Define the conditions
def total_cars : ℕ := 900
def honda_percentage_red : ℝ := 0.90
def total_percentage_red : ℝ := 0.60
def honda_cars : ℕ := 500

-- The statement to prove
theorem non_honda_red_percentage : 
  (0.60 * 900 - 0.90 * 500) / (900 - 500) * 100 = 22.5 := 
  by sorry

end non_honda_red_percentage_l100_100079


namespace ratio_of_speeds_l100_100263

theorem ratio_of_speeds (a b v1 v2 S : ℝ)
  (h1 : S = a * (v1 + v2))
  (h2 : S = b * (v1 - v2)) :
  v2 / v1 = (a + b) / (b - a) :=
by
  sorry

end ratio_of_speeds_l100_100263


namespace probability_Denis_Oleg_play_l100_100019

theorem probability_Denis_Oleg_play (n : ℕ) (h : n = 26) :
  let C := λ (n : ℕ), n * (n - 1) / 2 in
  (n - 1 : ℚ) / C n = 1 / 13 :=
by 
  sorry

end probability_Denis_Oleg_play_l100_100019


namespace increasing_interval_l100_100437

noncomputable def f (x k : ℝ) : ℝ := (x^2 / 2) - k * (Real.log x)

theorem increasing_interval (k : ℝ) (h₀ : 0 < k) : 
  ∃ (a : ℝ), (a = Real.sqrt k) ∧ 
  ∀ (x : ℝ), (x > a) → (∃ ε > 0, ∀ y, (x < y) → (f y k > f x k)) :=
sorry

end increasing_interval_l100_100437


namespace calculate_heartsuit_ratio_l100_100238

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem calculate_heartsuit_ratio :
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 :=
by sorry

end calculate_heartsuit_ratio_l100_100238


namespace who_received_q_first_round_l100_100176

-- Define the variables and conditions
variables (p q r : ℕ) (A B C : ℕ → ℕ) (n : ℕ)

-- Conditions
axiom h1 : 0 < p
axiom h2 : p < q
axiom h3 : q < r
axiom h4 : n ≥ 3
axiom h5 : A n = 20
axiom h6 : B n = 10
axiom h7 : C n = 9
axiom h8 : ∀ k, k > 0 → (B k = r → B (k-1) ≠ r)
axiom h9 : p + q + r = 13

-- Theorem to prove
theorem who_received_q_first_round : C 1 = q :=
sorry

end who_received_q_first_round_l100_100176


namespace total_length_XYZ_l100_100031

theorem total_length_XYZ :
  let straight_segments := 7
  let slanted_segments := 7 * Real.sqrt 2
  straight_segments + slanted_segments = 7 + 7 * Real.sqrt 2 :=
by
  sorry

end total_length_XYZ_l100_100031


namespace abs_neg_three_l100_100298

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l100_100298


namespace maximum_profit_and_price_range_l100_100908

-- Definitions
def cost_per_item : ℝ := 60
def max_profit_percentage : ℝ := 0.45
def sales_volume (x : ℝ) : ℝ := -x + 120
def profit (x : ℝ) : ℝ := sales_volume x * (x - cost_per_item)

-- The main theorem
theorem maximum_profit_and_price_range :
  (∃ x : ℝ, x = 87 ∧ profit x = 891) ∧
  (∀ x : ℝ, profit x ≥ 500 ↔ (70 ≤ x ∧ x ≤ 110)) :=
by
  sorry

end maximum_profit_and_price_range_l100_100908


namespace symmetric_line_eq_x_axis_l100_100301

theorem symmetric_line_eq_x_axis (x y : ℝ) :
  (3 * x - 4 * y + 5 = 0) → (3 * x + 4 * y + 5 = 0) :=
sorry

end symmetric_line_eq_x_axis_l100_100301


namespace prime_pairs_sum_50_l100_100671

open Nat

/-- There are 4 unordered pairs of prime numbers whose sum is 50. -/
theorem prime_pairs_sum_50 : 
  {p : ℕ // prime p} × {q : ℕ // prime q} → p + q = 50 → p ≤ q → ∃ (a b : ℕ), a + b = 50 ∧ prime a ∧ prime b ∧ (a, b) = (3, 47) ∨ (a, b) = (7, 43) ∨ (a, b) = (13, 37) ∨ (a, b) = (19, 31):= 
by
  sorry

end prime_pairs_sum_50_l100_100671


namespace how_much_does_c_have_l100_100753

theorem how_much_does_c_have (A B C : ℝ) (h1 : A + B + C = 400) (h2 : A + C = 300) (h3 : B + C = 150) : C = 50 :=
by
  sorry

end how_much_does_c_have_l100_100753


namespace cos_triple_angle_l100_100694

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l100_100694


namespace total_food_needed_l100_100410

-- Definitions for the conditions
def horses : ℕ := 4
def oats_per_meal : ℕ := 4
def oats_meals_per_day : ℕ := 2
def grain_per_day : ℕ := 3
def days : ℕ := 3

-- Theorem stating the problem
theorem total_food_needed :
  (horses * (days * (oats_per_meal * oats_meals_per_day) + days * grain_per_day)) = 132 :=
by sorry

end total_food_needed_l100_100410


namespace problem_divisible_by_480_l100_100826

theorem problem_divisible_by_480 (a : ℕ) (h1 : a % 10 = 4) (h2 : ¬ (a % 4 = 0)) : ∃ k : ℕ, a * (a^2 - 1) * (a^2 - 4) = 480 * k :=
by
  sorry

end problem_divisible_by_480_l100_100826


namespace johns_current_income_l100_100768

theorem johns_current_income
  (prev_income : ℝ := 1000000)
  (prev_tax_rate : ℝ := 0.20)
  (new_tax_rate : ℝ := 0.30)
  (extra_taxes_paid : ℝ := 250000) :
  ∃ (X : ℝ), 0.30 * X - 0.20 * prev_income = extra_taxes_paid ∧ X = 1500000 :=
by
  use 1500000
  -- Proof would come here
  sorry

end johns_current_income_l100_100768


namespace sales_tax_difference_l100_100634

theorem sales_tax_difference :
  let price : ℝ := 50
  let tax_rate1 : ℝ := 0.075
  let tax_rate2 : ℝ := 0.0625
  let tax1 := price * tax_rate1
  let tax2 := price * tax_rate2
  let difference := tax1 - tax2
  difference = 0.625 :=
by
  sorry

end sales_tax_difference_l100_100634


namespace polynomial_roots_to_determinant_l100_100404

noncomputable def determinant_eq (a b c m p q : ℂ) : Prop :=
  (Matrix.det ![
    ![a, 1, 1],
    ![1, b, 1],
    ![1, 1, c]
  ] = 2 - m - q)

theorem polynomial_roots_to_determinant (a b c m p q : ℂ) 
  (h1 : Polynomial.eval a (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  (h2 : Polynomial.eval b (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  (h3 : Polynomial.eval c (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  : determinant_eq a b c m p q :=
by sorry

end polynomial_roots_to_determinant_l100_100404


namespace binary_sum_to_decimal_l100_100919

theorem binary_sum_to_decimal :
  let bin1 := "1101011"
  let bin2 := "1010110"
  let dec1 := 64 + 32 + 0 + 8 + 0 + 2 + 1 -- decimal value of "1101011"
  let dec2 := 64 + 0 + 16 + 0 + 4 + 2 + 0 -- decimal value of "1010110"
  dec1 + dec2 = 193 := by
  sorry

end binary_sum_to_decimal_l100_100919


namespace find_ab_l100_100518

theorem find_ab (a b : ℕ) (h : (Real.sqrt 30 - Real.sqrt 18) * (3 * Real.sqrt a + Real.sqrt b) = 12) : a = 2 ∧ b = 30 :=
sorry

end find_ab_l100_100518


namespace intersection_S_T_l100_100377

def S := {x : ℝ | abs x < 5}
def T := {x : ℝ | (x + 7) * (x - 3) < 0}

theorem intersection_S_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} :=
by
  sorry

end intersection_S_T_l100_100377


namespace largest_n_polynomials_l100_100635

theorem largest_n_polynomials :
  ∃ (P : ℕ → (ℝ → ℝ)), (∀ i j, i ≠ j → ∀ x, P i x + P j x ≠ 0) ∧ (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ∃ x, P i x + P j x + P k x = 0) ↔ n = 3 := 
sorry

end largest_n_polynomials_l100_100635


namespace turnip_weights_are_13_or_16_l100_100003

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l100_100003


namespace restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l100_100903

-- Defining the given conditions
noncomputable def market_demand (P : ℝ) : ℝ := 688 - 4 * P
noncomputable def post_tax_producer_price : ℝ := 64
noncomputable def per_unit_tax : ℝ := 90
noncomputable def elasticity_supply_no_tax (P_e : ℝ) (Q_e : ℝ) : ℝ :=
  1.5 * (-(4 * P_e / Q_e))

-- Supply function to be proven
noncomputable def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Total tax revenue to be proven
noncomputable def total_tax_revenue : ℝ := 6480

-- Optimal tax rate to be proven
noncomputable def optimal_tax_rate : ℝ := 60

-- Maximum tax revenue to be proven
noncomputable def maximum_tax_revenue : ℝ := 8640

-- Theorem statements that need to be proven
theorem restore_supply_function (P : ℝ) : 
  supply_function P = 6 * P - 312 := sorry

theorem determine_tax_revenue : 
  total_tax_revenue = 6480 := sorry

theorem determine_optimal_tax_rate : 
  optimal_tax_rate = 60 := sorry

theorem determine_maximum_tax_revenue : 
  maximum_tax_revenue = 8640 := sorry

end restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l100_100903


namespace negative_integers_abs_le_4_l100_100617

theorem negative_integers_abs_le_4 :
  ∀ x : ℤ, x < 0 ∧ |x| ≤ 4 ↔ (x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4) :=
by
  sorry

end negative_integers_abs_le_4_l100_100617


namespace nail_insertion_l100_100011

theorem nail_insertion (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  (4/7) + (4/7) * k + (4/7) * k^2 = 1 :=
by sorry

end nail_insertion_l100_100011


namespace ratio_fraction_l100_100353

theorem ratio_fraction (x : ℚ) : x = 2 / 9 ↔ (2 / 6) / x = (3 / 4) / (1 / 2) := by
  sorry

end ratio_fraction_l100_100353


namespace recurring_decimal_to_fraction_correct_l100_100470

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l100_100470


namespace units_digit_of_large_powers_l100_100207

theorem units_digit_of_large_powers : 
  (2^1007 * 6^1008 * 14^1009) % 10 = 2 := 
  sorry

end units_digit_of_large_powers_l100_100207


namespace cube_surface_area_l100_100373

theorem cube_surface_area (V : ℝ) (hV : V = 64) : ∃ S : ℝ, S = 96 := 
by
  sorry

end cube_surface_area_l100_100373


namespace cab_speed_ratio_l100_100764

variable (S_u S_c : ℝ)

theorem cab_speed_ratio (h1 : ∃ S_u S_c : ℝ, S_u * 25 = S_c * 30) : S_c / S_u = 5 / 6 :=
by
  sorry

end cab_speed_ratio_l100_100764


namespace total_spent_l100_100321

theorem total_spent (bracelet_price keychain_price coloring_book_price : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ)
  (total : ℕ) :
  bracelet_price = 4 →
  keychain_price = 5 →
  coloring_book_price = 3 →
  paula_bracelets = 2 →
  paula_keychains = 1 →
  olive_coloring_books = 1 →
  olive_bracelets = 1 →
  total = paula_bracelets * bracelet_price + paula_keychains * keychain_price +
          olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price →
  total = 20 :=
by sorry

end total_spent_l100_100321


namespace num_prime_pairs_sum_50_l100_100669

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a set of prime numbers less than 25
def primes_less_than_25 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define a set of pairs (p, q) where the sum is 50, p and q are primes and p ≤ q
def pairs_sum_50 : set (ℕ × ℕ) := {(p, q) | p + q = 50 ∧ is_prime p ∧ is_prime q ∧ p ≤ q}

-- Theorem stating that there are 4 pairs of primes that sum to 50
theorem num_prime_pairs_sum_50 : (pairs_sum_50 primes_less_than_25).card = 4 :=
sorry

end num_prime_pairs_sum_50_l100_100669


namespace turnip_weights_are_13_or_16_l100_100005

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l100_100005


namespace Olivia_paint_area_l100_100789

theorem Olivia_paint_area
  (length width height : ℕ) (door_window_area : ℕ) (bedrooms : ℕ)
  (h_length : length = 14) 
  (h_width : width = 11) 
  (h_height : height = 9) 
  (h_door_window_area : door_window_area = 70) 
  (h_bedrooms : bedrooms = 4) :
  (2 * (length * height) + 2 * (width * height) - door_window_area) * bedrooms = 1520 :=
by
  sorry

end Olivia_paint_area_l100_100789


namespace total_weight_lifted_l100_100507

-- Given definitions from the conditions
def weight_left_hand : ℕ := 10
def weight_right_hand : ℕ := 10

-- The proof problem statement
theorem total_weight_lifted : weight_left_hand + weight_right_hand = 20 := 
by 
  -- Proof goes here
  sorry

end total_weight_lifted_l100_100507


namespace expression_always_positive_l100_100117

theorem expression_always_positive (x : ℝ) : x^2 + |x| + 1 > 0 :=
by 
  sorry

end expression_always_positive_l100_100117


namespace abs_neg_three_l100_100282

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l100_100282


namespace remainder_calculation_l100_100597

theorem remainder_calculation :
  ((2367 * 1023) % 500) = 41 := by
  sorry

end remainder_calculation_l100_100597


namespace max_cities_l100_100959

theorem max_cities (n : ℕ) (h1 : ∀ (c : Fin n), ∃ (neighbors : Finset (Fin n)), neighbors.card ≤ 3 ∧ c ∈ neighbors) (h2 : ∀ (c1 c2 : Fin n), c1 ≠ c2 → ∃ c : Fin n, c1 ≠ c ∧ c2 ≠ c) : n ≤ 10 := 
sorry

end max_cities_l100_100959


namespace negative_integers_abs_le_4_l100_100619

theorem negative_integers_abs_le_4 (x : Int) (h1 : x < 0) (h2 : abs x ≤ 4) : 
  x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4 :=
by
  sorry

end negative_integers_abs_le_4_l100_100619


namespace total_books_l100_100721

-- Definitions for the conditions
def SandyBooks : Nat := 10
def BennyBooks : Nat := 24
def TimBooks : Nat := 33

-- Stating the theorem we need to prove
theorem total_books : SandyBooks + BennyBooks + TimBooks = 67 := by
  sorry

end total_books_l100_100721


namespace D_coin_count_l100_100749

def A_coin_count : ℕ := 21
def B_coin_count := A_coin_count - 9
def C_coin_count := B_coin_count + 17
def sum_A_B := A_coin_count + B_coin_count
def sum_C_D := sum_A_B + 5

theorem D_coin_count :
  ∃ D : ℕ, sum_C_D - C_coin_count = D :=
sorry

end D_coin_count_l100_100749


namespace outlined_square_digit_l100_100306

theorem outlined_square_digit :
  ∀ (digit : ℕ), (digit ∈ {n | ∃ (m : ℕ), 10 ≤ 3^m ∧ 3^m < 1000 ∧ digit = (3^m / 10) % 10 }) →
  (digit ∈ {n | ∃ (n : ℕ), 10 ≤ 7^n ∧ 7^n < 1000 ∧ digit = (7^n / 10) % 10 }) →
  digit = 4 :=
by sorry

end outlined_square_digit_l100_100306


namespace simplify_expression_l100_100858

theorem simplify_expression :
  64^(1/4) - 144^(1/4) = 2 * Real.sqrt 2 - 2 * Real.sqrt 3 := 
by
  sorry

end simplify_expression_l100_100858


namespace mass_percentage_I_in_CaI2_l100_100206

theorem mass_percentage_I_in_CaI2 :
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_I : ℝ := 126.90
  let molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
  let mass_percentage_I : ℝ := (2 * molar_mass_I / molar_mass_CaI2) * 100
  mass_percentage_I = 86.36 := by
  sorry

end mass_percentage_I_in_CaI2_l100_100206


namespace geometric_seq_arithmetic_example_l100_100248

noncomputable def a_n (n : ℕ) (q : ℝ) : ℝ :=
if n = 0 then 1 else q ^ n

theorem geometric_seq_arithmetic_example {q : ℝ} (h₀ : q ≠ 0)
    (h₁ : ∀ n : ℕ, a_n 0 q = 1)
    (h₂ : 2 * (2 * (q ^ 2)) = 3 * q) :
    (q + q^2 + (q^3)) = 14 :=
by sorry

end geometric_seq_arithmetic_example_l100_100248


namespace pencils_per_row_l100_100794

-- Define the conditions
def total_pencils := 25
def number_of_rows := 5

-- Theorem statement: The number of pencils per row is 5 given the conditions
theorem pencils_per_row : total_pencils / number_of_rows = 5 :=
by
  -- The proof should go here
  sorry

end pencils_per_row_l100_100794


namespace vacation_animals_total_l100_100573

noncomputable def lisa := 40
noncomputable def alex := lisa / 2
noncomputable def jane := alex + 10
noncomputable def rick := 3 * jane
noncomputable def tim := 2 * rick
noncomputable def you := 5 * tim
noncomputable def total_animals := lisa + alex + jane + rick + tim + you

theorem vacation_animals_total : total_animals = 1260 := by
  sorry

end vacation_animals_total_l100_100573


namespace abs_neg_three_l100_100296

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l100_100296


namespace tina_made_more_140_dollars_l100_100842

def candy_bars_cost : ℕ := 2
def marvin_candy_bars : ℕ := 35
def tina_candy_bars : ℕ := 3 * marvin_candy_bars
def marvin_money : ℕ := marvin_candy_bars * candy_bars_cost
def tina_money : ℕ := tina_candy_bars * candy_bars_cost
def tina_extra_money : ℕ := tina_money - marvin_money

theorem tina_made_more_140_dollars :
  tina_extra_money = 140 := by
  sorry

end tina_made_more_140_dollars_l100_100842


namespace range_of_ab_l100_100936

def circle_eq (x y : ℝ) := x^2 + y^2 + 2 * x - 4 * y + 1 = 0
def line_eq (a b x y : ℝ) := 2 * a * x - b * y + 2 = 0

theorem range_of_ab (a b : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq a b x y) ∧ (∃ x y : ℝ, x = -1 ∧ y = 2) →
  ab <= 1/4 := 
by
  sorry

end range_of_ab_l100_100936


namespace sequence_bound_exists_l100_100517

def sequence (n : ℕ) : ℕ → ℝ
| 0 := 5
| (n + 1) := (sequence n ^ 2 + 5 * sequence n + 4) / (sequence n + 6)

theorem sequence_bound_exists :
  ∃ m : ℕ, 81 ≤ m ∧ m ≤ 242 ∧ sequence m ≤ 4 + 1 / 2^20 :=
sorry

end sequence_bound_exists_l100_100517


namespace at_least_one_inequality_holds_l100_100107

theorem at_least_one_inequality_holds (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y > 2) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l100_100107


namespace rainfall_second_week_january_l100_100522

-- Define the conditions
def total_rainfall_2_weeks (rainfall_first_week rainfall_second_week : ℝ) : Prop :=
  rainfall_first_week + rainfall_second_week = 20

def rainfall_second_week_is_1_5_times_first (rainfall_first_week rainfall_second_week : ℝ) : Prop :=
  rainfall_second_week = 1.5 * rainfall_first_week

-- Define the statement to prove
theorem rainfall_second_week_january (rainfall_first_week rainfall_second_week : ℝ) :
  total_rainfall_2_weeks rainfall_first_week rainfall_second_week →
  rainfall_second_week_is_1_5_times_first rainfall_first_week rainfall_second_week →
  rainfall_second_week = 12 :=
by
  sorry

end rainfall_second_week_january_l100_100522


namespace problem_l100_100367

theorem problem (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) : 
  x^3 + 3 * y^2 + 3 * z^2 + 3 * x * y * z = 20 := by
sorry

end problem_l100_100367


namespace trigonometric_expression_l100_100049

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 3) : 
  ((Real.cos (α - π / 2) + Real.cos (α + π)) / (2 * Real.sin α) = 1 / 3) :=
by
  sorry

end trigonometric_expression_l100_100049


namespace certain_fraction_ratio_l100_100352

theorem certain_fraction_ratio :
  ∃ x : ℚ,
    (2 / 5 : ℚ) / x = (0.46666666666666673 : ℚ) / (1 / 2) ∧ x = 3 / 7 :=
by sorry

end certain_fraction_ratio_l100_100352


namespace cross_section_area_l100_100870

-- Definitions representing the conditions
variables (AK KD BP PC DM DC : ℝ)
variable (h : ℝ)
variable (Volume : ℝ)

-- Conditions
axiom hyp1 : AK = KD
axiom hyp2 : BP = PC
axiom hyp3 : DM = 0.4 * DC
axiom hyp4 : h = 1
axiom hyp5 : Volume = 5

-- Proof problem: Prove that the area S of the cross-section of the pyramid is 3
theorem cross_section_area (S : ℝ) : S = 3 :=
by sorry

end cross_section_area_l100_100870


namespace solve_for_x_l100_100599

theorem solve_for_x : ∃ x : ℝ, (2010 + x)^3 = -x^3 ∧ x = -1005 := 
by
  use -1005
  sorry

end solve_for_x_l100_100599


namespace cost_of_peaches_eq_2_per_pound_l100_100703

def initial_money : ℕ := 20
def after_buying_peaches : ℕ := 14
def pounds_of_peaches : ℕ := 3
def cost_per_pound : ℕ := 2

theorem cost_of_peaches_eq_2_per_pound (h: initial_money - after_buying_peaches = pounds_of_peaches * cost_per_pound) :
  cost_per_pound = 2 := by
  sorry

end cost_of_peaches_eq_2_per_pound_l100_100703


namespace a_7_is_4_l100_100045

-- Define the geometric sequence and its properties
variable {a : ℕ → ℝ}

-- Given conditions
axiom pos_seq : ∀ n, a n > 0
axiom geom_seq : ∀ n m, a (n + m) = a n * a m
axiom specific_condition : a 3 * a 11 = 16

theorem a_7_is_4 : a 7 = 4 :=
by
  sorry

end a_7_is_4_l100_100045


namespace right_triangle_fab_eccentricity_l100_100129

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  (classical.some (quadratic_eq_exists (a * a - b * b) (-b * b) 1 rfl))

theorem right_triangle_fab_eccentricity :
  ∀ {a b : ℝ} (h : a > b ∧ b > 0),
  eccentricity_of_ellipse a b h = (-1 + Real.sqrt 5) / 2 := by
  sorry

end right_triangle_fab_eccentricity_l100_100129


namespace solution_inequality_l100_100875

-- Define the condition as a predicate
def inequality_condition (x : ℝ) : Prop :=
  (x - 1) * (x + 1) < 0

-- State the theorem that we need to prove
theorem solution_inequality : ∀ x : ℝ, inequality_condition x → (-1 < x ∧ x < 1) :=
by
  intro x hx
  sorry

end solution_inequality_l100_100875


namespace JohnsonsYield_l100_100251

def JohnsonYieldPerTwoMonths (J : ℕ) : Prop :=
  ∀ (neighbor_hectares neighbor_yield_per_hectare total_yield_six_months : ℕ),
    neighbor_hectares = 2 →
    neighbor_yield_per_hectare = 2 * J →
    total_yield_six_months = 1200 →
    3 * J + 3 * (neighbor_hectares * neighbor_yield_per_hectare) = total_yield_six_months →
    J = 80

theorem JohnsonsYield
  (J : ℕ)
  (neighbor_hectares neighbor_yield_per_hectare total_yield_six_months : ℕ)
  (h1 : neighbor_hectares = 2)
  (h2 : neighbor_yield_per_hectare = 2 * J)
  (h3 : total_yield_six_months = 1200)
  (h4 : 3 * J + 3 * (neighbor_hectares * neighbor_yield_per_hectare) = total_yield_six_months) :
  J = 80 :=
by
  sorry

end JohnsonsYield_l100_100251


namespace stamps_problem_l100_100839

def largest_common_divisor (a b c : ℕ) : ℕ :=
  gcd (gcd a b) c

theorem stamps_problem :
  largest_common_divisor 1020 1275 1350 = 15 :=
by
  sorry

end stamps_problem_l100_100839


namespace prime_pairs_sum_to_50_l100_100676

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_less_than (n : ℕ) : list ℕ :=
  (list.range n).filter is_prime

def prime_pairs_sum_50 : list (ℕ × ℕ) :=
  [(3, 47), (7, 43), (13, 37), (19, 31)]

theorem prime_pairs_sum_to_50 : 
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs = prime_pairs_sum_50 ∧ 
    (∀ p ∈ pairs, is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50) ∧ 
    pairs.length = 4 :=
begin
  use prime_pairs_sum_50,
  split,
  { refl },
  split,
  { intros p hp,
    cases p,
    dsimp at hp,
    rcases hp with rfl | rfl | rfl | rfl;
    simp [is_prime] },
  { refl }
end

end prime_pairs_sum_to_50_l100_100676


namespace next_divisor_after_391_l100_100250

theorem next_divisor_after_391 (m : ℕ) (h1 : m % 2 = 0) (h2 : m ≥ 1000 ∧ m < 10000) (h3 : 391 ∣ m) : 
  ∃ n, n > 391 ∧ n ∣ m ∧ (∀ k, k > 391 ∧ k < n → ¬ k ∣ m) ∧ n = 782 :=
sorry

end next_divisor_after_391_l100_100250


namespace abs_neg_three_eq_three_l100_100290

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l100_100290


namespace min_value_ge_8_min_value_8_at_20_l100_100211

noncomputable def min_value (x : ℝ) (h : x > 4) : ℝ := (x + 12) / Real.sqrt (x - 4)

theorem min_value_ge_8 (x : ℝ) (h : x > 4) : min_value x h ≥ 8 := sorry

theorem min_value_8_at_20 : min_value 20 (by norm_num) = 8 := sorry

end min_value_ge_8_min_value_8_at_20_l100_100211


namespace proof_l100_100528

noncomputable def problem_statement : Prop :=
  ( ( (Real.sqrt 1.21 * Real.sqrt 1.44) / (Real.sqrt 0.81 * Real.sqrt 0.64)
    + (Real.sqrt 1.0 * Real.sqrt 3.24) / (Real.sqrt 0.49 * Real.sqrt 2.25) ) ^ 3 
  = 44.6877470366 )

theorem proof : problem_statement := 
  by
  sorry

end proof_l100_100528


namespace relatively_prime_perfect_squares_l100_100855

theorem relatively_prime_perfect_squares (a b c : ℤ) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
    (h_eq : (1:ℚ) / a + (1:ℚ) / b = (1:ℚ) / c) :
    ∃ x y z : ℤ, (a + b = x^2 ∧ a - c = y^2 ∧ b - c = z^2) :=
  sorry

end relatively_prime_perfect_squares_l100_100855


namespace num_bags_in_range_l100_100614

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

noncomputable def stddev (l : List ℝ) : ℝ :=
  real.sqrt (variance l)

def weights : List ℝ := [495, 500, 503, 508, 498, 500, 493, 500, 503, 500]

def bags_in_range (l : List ℝ) (μ σ : ℝ) : ℕ :=
  (l.filter (λ x => μ - σ ≤ x ∧ x ≤ μ + σ)).length

theorem num_bags_in_range : bags_in_range weights (mean weights) (stddev weights) = 7 := by
  let μ := mean weights
  let σ := stddev weights
  have h_mean : μ = 500 := by sorry
  have h_stddev : σ = 4 := by sorry
  rw [h_mean, h_stddev]
  show bags_in_range weights 500 4 = 7
  exact sorry

end num_bags_in_range_l100_100614


namespace part2_x_values_part3_no_real_x_for_2000_l100_100765

noncomputable def average_daily_sales (x : ℝ) : ℝ :=
  24 + 4 * x

noncomputable def profit_per_unit (x : ℝ) : ℝ :=
  60 - 5 * x

noncomputable def daily_sales_profit (x : ℝ) : ℝ :=
  (60 - 5 * x) * (24 + 4 * x)

theorem part2_x_values : 
  {x : ℝ | daily_sales_profit x = 1540} = {1, 5} := sorry

theorem part3_no_real_x_for_2000 : 
  ∀ x : ℝ, daily_sales_profit x ≠ 2000 := sorry

end part2_x_values_part3_no_real_x_for_2000_l100_100765


namespace locus_equation_rectangle_perimeter_greater_l100_100087

open Real

theorem locus_equation (P : ℝ × ℝ) : 
  (abs P.2 = sqrt (P.1 ^ 2 + (P.2 - 1 / 2) ^ 2)) → (P.2 = P.1 ^ 2 + 1 / 4) :=
by
  intro h
  sorry

theorem rectangle_perimeter_greater (A B C D : ℝ × ℝ) :
  (A.2 = A.1 ^ 2 + 1 / 4) ∧ 
  (B.2 = B.1 ^ 2 + 1 / 4) ∧ 
  (C.2 = C.1 ^ 2 + 1 / 4) ∧ 
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) → 
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
by
  intro h
  sorry

end locus_equation_rectangle_perimeter_greater_l100_100087


namespace cos_triple_angle_l100_100693

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l100_100693


namespace product_floor_ceil_sequence_l100_100923

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem product_floor_ceil_sequence :
    (floor (-6 - 0.5) * ceil (6 + 0.5)) *
    (floor (-5 - 0.5) * ceil (5 + 0.5)) *
    (floor (-4 - 0.5) * ceil (4 + 0.5)) *
    (floor (-3 - 0.5) * ceil (3 + 0.5)) *
    (floor (-2 - 0.5) * ceil (2 + 0.5)) *
    (floor (-1 - 0.5) * ceil (1 + 0.5)) *
    (floor (-0.5) * ceil (0.5)) = -25401600 :=
by
  sorry

end product_floor_ceil_sequence_l100_100923


namespace turnip_bag_weight_l100_100009

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l100_100009


namespace least_number_divisible_by_11_and_remainder_2_l100_100744

theorem least_number_divisible_by_11_and_remainder_2 :
  ∃ n, (∀ k : ℕ, 3 ≤ k ∧ k ≤ 7 → n % k = 2) ∧ n % 11 = 0 ∧ n = 1262 :=
by
  sorry

end least_number_divisible_by_11_and_remainder_2_l100_100744


namespace problem1_l100_100895

theorem problem1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) : 
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
sorry

end problem1_l100_100895


namespace derek_february_savings_l100_100035

theorem derek_february_savings :
  ∀ (savings : ℕ → ℕ),
  (savings 1 = 2) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 12 → savings (n + 1) = 2 * savings n) ∧
  (savings 12 = 4096) →
  savings 2 = 4 :=
by
  sorry

end derek_february_savings_l100_100035


namespace perimeter_of_square_is_160_cm_l100_100580

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_of_square (area_of_rectangle : ℝ) : ℝ := 5 * area_of_rectangle

noncomputable def side_length_of_square (area_of_square : ℝ) : ℝ := Real.sqrt area_of_square

noncomputable def perimeter_of_square (side_length : ℝ) : ℝ := 4 * side_length

theorem perimeter_of_square_is_160_cm :
  perimeter_of_square (side_length_of_square (area_of_square (area_of_rectangle 32 10))) = 160 :=
by
  sorry

end perimeter_of_square_is_160_cm_l100_100580


namespace rectangle_width_decrease_l100_100130

theorem rectangle_width_decrease (L W : ℝ) (h1 : 0 < L) (h2 : 0 < W) 
(h3 : ∀ W' : ℝ, 0 < W' → (1.3 * L * W' = L * W) → W' = (100 - 23.077) / 100 * W) : 
  ∃ W' : ℝ, 0 < W' ∧ (1.3 * L * W' = L * W) ∧ ((W - W') / W = 23.077 / 100) :=
by
  sorry

end rectangle_width_decrease_l100_100130


namespace range_of_m_l100_100064

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2) * x^2 - 2 * x + 5

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Icc (-1 : ℝ) 2, f x < m) ↔ 7 < m :=
begin
  sorry
end

end range_of_m_l100_100064


namespace fraction_exists_l100_100264

theorem fraction_exists (n d k : ℕ) (h₁ : n = k * d) (h₂ : d > 0) (h₃ : k > 0) : 
  ∃ (i j : ℕ), i < n ∧ j < n ∧ i + j = n ∧ i/j = d-1 :=
by
  sorry

end fraction_exists_l100_100264


namespace train_crossing_time_l100_100760

theorem train_crossing_time
  (length_of_train : ℝ)
  (speed_in_kmh : ℝ)
  (speed_in_mps : ℝ)
  (conversion_factor : ℝ)
  (time : ℝ)
  (h1 : length_of_train = 160)
  (h2 : speed_in_kmh = 36)
  (h3 : conversion_factor = 1 / 3.6)
  (h4 : speed_in_mps = speed_in_kmh * conversion_factor)
  (h5 : time = length_of_train / speed_in_mps) : time = 16 :=
by
  sorry

end train_crossing_time_l100_100760


namespace find_values_of_m_and_n_l100_100370

theorem find_values_of_m_and_n (m n : ℝ) (h : m / (1 + I) = 1 - n * I) : 
  m = 2 ∧ n = 1 :=
sorry

end find_values_of_m_and_n_l100_100370


namespace average_problem_l100_100863

theorem average_problem
  (h : (20 + 40 + 60) / 3 = (x + 50 + 45) / 3 + 5) :
  x = 10 :=
by
  sorry

end average_problem_l100_100863


namespace periodic_sequence_criteria_l100_100808

theorem periodic_sequence_criteria (K : ℕ) (hK_pos : K > 0) :
  ( ∀ p, ∃ m, ∀ n ≥ m, (Nat.choose (2 * n) n) % K = (Nat.choose (2 * (n + p)) (n + p)) % K ) →
  (K = 1 ∨ K = 2) :=
begin
  sorry,
end

end periodic_sequence_criteria_l100_100808


namespace biff_break_even_hours_l100_100631

-- Definitions based on conditions
def ticket_expense : ℕ := 11
def snacks_expense : ℕ := 3
def headphones_expense : ℕ := 16
def total_expenses : ℕ := ticket_expense + snacks_expense + headphones_expense
def gross_income_per_hour : ℕ := 12
def wifi_cost_per_hour : ℕ := 2
def net_income_per_hour : ℕ := gross_income_per_hour - wifi_cost_per_hour

-- The proof statement
theorem biff_break_even_hours : ∃ h : ℕ, h * net_income_per_hour = total_expenses ∧ h = 3 :=
by 
  have h_value : ℕ := 3
  exists h_value
  split
  · show h_value * net_income_per_hour = total_expenses
    sorry
  · show h_value = 3
    rfl

end biff_break_even_hours_l100_100631


namespace percent_increase_l100_100915

/-- Problem statement: Given (1/2)x = 1, prove that the percentage increase from 1/2 to x is 300%. -/
theorem percent_increase (x : ℝ) (h : (1/2) * x = 1) : 
  ((x - (1/2)) / (1/2)) * 100 = 300 := 
by
  sorry

end percent_increase_l100_100915


namespace find_BP_l100_100494

-- Define points
variables {A B C D P : Type}  

-- Define lengths
variables (AP PC BP DP BD : ℝ)

-- Provided conditions
axiom h1 : AP = 10
axiom h2 : PC = 2
axiom h3 : BD = 9

-- Assume intersect and lengths relations setup
axiom intersect : BP < DP
axiom power_of_point : AP * PC = BP * DP

-- Target statement
theorem find_BP (h1 : AP = 10) (h2 : PC = 2) (h3 : BD = 9)
  (intersect : BP < DP) (power_of_point : AP * PC = BP * DP) : BP = 4 :=
  sorry

end find_BP_l100_100494


namespace abs_neg_three_eq_three_l100_100289

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l100_100289


namespace peter_food_necessity_l100_100411

/-- Discuss the conditions  -/
def peter_horses (num_horses num_days : ℕ) (oats_per_meal grain_per_day : ℕ) (meals_per_day : ℕ) : ℕ :=
  let daily_oats := oats_per_meal * meals_per_day in
  let total_oats := daily_oats * num_days * num_horses in
  let total_grain := grain_per_day * num_days * num_horses in
  total_oats + total_grain

/-- Prove that Peter needs 132 pounds of food to feed his horses for 3 days -/
theorem peter_food_necessity : peter_horses 4 3 4 3 2 = 132 :=
  sorry

end peter_food_necessity_l100_100411


namespace recurring_to_fraction_l100_100474

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l100_100474


namespace mall_b_better_for_fewer_than_6_mall_equal_for_6_mall_a_better_for_more_than_6_l100_100497

-- Definitions
def original_price : ℕ := 80
def discount_mallA (n : ℕ) : ℕ := min ((4 * n) * n) (80 * n / 2)
def discount_mallB (n : ℕ) : ℕ := (80 * n * 3) / 10

def total_cost_mallA (n : ℕ) : ℕ := (original_price * n) - discount_mallA n
def total_cost_mallB (n : ℕ) : ℕ := (original_price * n) - discount_mallB n

-- Theorem statements
theorem mall_b_better_for_fewer_than_6 (n : ℕ) (h : n < 6) : total_cost_mallA n > total_cost_mallB n := sorry
theorem mall_equal_for_6 (n : ℕ) (h : n = 6) : total_cost_mallA n = total_cost_mallB n := sorry
theorem mall_a_better_for_more_than_6 (n : ℕ) (h : n > 6) : total_cost_mallA n < total_cost_mallB n := sorry

end mall_b_better_for_fewer_than_6_mall_equal_for_6_mall_a_better_for_more_than_6_l100_100497


namespace shaded_area_of_hexagon_with_quarter_circles_l100_100171

noncomputable def area_inside_hexagon_outside_circles
  (s : ℝ) (h : s = 4) : ℝ :=
  let hex_area := (3 * Real.sqrt 3) / 2 * s^2
  let quarter_circle_area := (1 / 4) * Real.pi * s^2
  let total_quarter_circles_area := 6 * quarter_circle_area
  hex_area - total_quarter_circles_area

theorem shaded_area_of_hexagon_with_quarter_circles :
  area_inside_hexagon_outside_circles 4 rfl = 48 * Real.sqrt 3 - 24 * Real.pi := by
  sorry

end shaded_area_of_hexagon_with_quarter_circles_l100_100171


namespace evaluate_polynomial_at_minus_two_l100_100640

noncomputable def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - x^2 + 2 * x + 5

theorem evaluate_polynomial_at_minus_two : polynomial (-2) = 5 := by
  sorry

end evaluate_polynomial_at_minus_two_l100_100640


namespace repeating_decimal_to_fraction_l100_100480

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l100_100480


namespace convert_110110001_to_base4_l100_100034

def binary_to_base4_conversion (b : ℕ) : ℕ :=
  -- assuming b is the binary representation of the number to be converted
  1 * 4^4 + 3 * 4^3 + 2 * 4^2 + 0 * 4^1 + 1 * 4^0

theorem convert_110110001_to_base4 : binary_to_base4_conversion 110110001 = 13201 :=
  sorry

end convert_110110001_to_base4_l100_100034


namespace arithmetic_evaluation_l100_100740

theorem arithmetic_evaluation : 6 * 2 - 3 = 9 := by
  sorry

end arithmetic_evaluation_l100_100740


namespace proof_problem_l100_100987

theorem proof_problem
  (x y : ℤ)
  (hx : ∃ m : ℤ, x = 6 * m)
  (hy : ∃ n : ℤ, y = 12 * n) :
  (x + y) % 2 = 0 ∧ (x + y) % 6 = 0 ∧ ¬ (x + y) % 12 = 0 → ¬ (x + y) % 12 = 0 :=
  sorry

end proof_problem_l100_100987


namespace base6_to_base10_l100_100125

theorem base6_to_base10 (c d : ℕ) (h1 : 524 = 2 * (10 * c + d)) (hc : c < 10) (hd : d < 10) :
  (c * d : ℚ) / 12 = 3 / 4 := by
  sorry

end base6_to_base10_l100_100125


namespace negation_of_P_l100_100218

def P : Prop := ∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 ≤ 0

theorem negation_of_P : ¬ P ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by sorry

end negation_of_P_l100_100218


namespace x_is_perfect_square_l100_100118

theorem x_is_perfect_square (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (hdiv : 2 * x * y ∣ x^2 + y^2 - x) : ∃ (n : ℕ), x = n^2 :=
by
  sorry

end x_is_perfect_square_l100_100118


namespace max_sum_x_y_l100_100656

theorem max_sum_x_y (x y : ℝ) (h : (2015 + x^2) * (2015 + y^2) = 2 ^ 22) : 
  x + y ≤ 2 * Real.sqrt 33 :=
sorry

end max_sum_x_y_l100_100656


namespace solve_for_x_l100_100586

theorem solve_for_x (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (4 * x + 28)) : 
  x = -17 / 5 := 
by 
  sorry

end solve_for_x_l100_100586


namespace cheryl_mms_l100_100512

/-- Cheryl's m&m problem -/
theorem cheryl_mms (c l g d : ℕ) (h1 : c = 25) (h2 : l = 7) (h3 : g = 13) :
  (c - l - g) = d → d = 5 :=
by
  sorry

end cheryl_mms_l100_100512


namespace factorize_expression_l100_100349

theorem factorize_expression (a : ℝ) : 3 * a^2 + 6 * a + 3 = 3 * (a + 1)^2 := 
by sorry

end factorize_expression_l100_100349


namespace inequality_N_value_l100_100343

theorem inequality_N_value (a c : ℝ) (ha : 0 < a) (hc : 0 < c) (b : ℝ) (hb : b = 2 * a) : 
  (a^2 + b^2) / c^2 > 5 / 9 := 
by sorry

end inequality_N_value_l100_100343


namespace line_through_intersection_and_origin_l100_100866

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := 2023 * x - 2022 * y - 1 = 0
def line2 (x y : ℝ) : Prop := 2022 * x + 2023 * y + 1 = 0

-- Define the line passing through the origin
def line_pass_origin (x y : ℝ) : Prop := 4045 * x + y = 0

-- Define the intersection point of the two lines
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define the theorem stating the desired property
theorem line_through_intersection_and_origin (x y : ℝ)
    (h1 : intersection x y)
    (h2 : x = 0 ∧ y = 0) :
    line_pass_origin x y :=
by
    sorry

end line_through_intersection_and_origin_l100_100866


namespace find_common_difference_max_sum_first_n_terms_max_n_Sn_positive_l100_100440

-- Define the arithmetic sequence
def a (n : ℕ) (d : ℤ) := 23 + n * d

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) (d : ℤ) := n * 23 + (n * (n - 1) / 2) * d

-- Prove the common difference is -4
theorem find_common_difference (d : ℤ) :
  a 5 d > 0 ∧ a 6 d < 0 → d = -4 := sorry

-- Prove the maximum value of the sum S_n of the first n terms
theorem max_sum_first_n_terms (S_n : ℕ) :
  S 6 -4 = 78 := sorry

-- Prove the maximum value of n such that S_n > 0
theorem max_n_Sn_positive (n : ℕ) :
  S n -4 > 0 → n ≤ 12 := sorry

end find_common_difference_max_sum_first_n_terms_max_n_Sn_positive_l100_100440


namespace total_students_l100_100546

-- Given definitions
def basketball_count : ℕ := 7
def cricket_count : ℕ := 5
def both_count : ℕ := 3

-- The goal to prove
theorem total_students : basketball_count + cricket_count - both_count = 9 :=
by
  sorry

end total_students_l100_100546


namespace largest_consecutive_odd_integer_sum_l100_100880

theorem largest_consecutive_odd_integer_sum :
  ∃ N : ℤ, N + (N + 2) + (N + 4) = -147 ∧ (N + 4) = -47 :=
begin
  sorry
end

end largest_consecutive_odd_integer_sum_l100_100880


namespace abs_neg_three_l100_100299

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l100_100299


namespace common_root_quadratic_l100_100803

theorem common_root_quadratic (a x1: ℝ) :
  (x1^2 + a * x1 + 1 = 0) ∧ (x1^2 + x1 + a = 0) ↔ a = -2 :=
sorry

end common_root_quadratic_l100_100803


namespace Sherman_weekly_driving_time_l100_100417

theorem Sherman_weekly_driving_time (daily_commute : ℕ := 30) (weekend_drive : ℕ := 2) : 
  (5 * (2 * daily_commute) / 60 + 2 * weekend_drive) = 9 := 
by
  sorry

end Sherman_weekly_driving_time_l100_100417


namespace shirt_cost_is_ten_l100_100611

theorem shirt_cost_is_ten (S J : ℝ) (h1 : J = 2 * S) 
    (h2 : 20 * S + 10 * J = 400) : S = 10 :=
by
  -- proof skipped
  sorry

end shirt_cost_is_ten_l100_100611


namespace weekly_earnings_proof_l100_100269

def minutes_in_hour : ℕ := 60
def hourly_rate : ℕ := 4

def monday_minutes : ℕ := 150
def tuesday_minutes : ℕ := 40
def wednesday_minutes : ℕ := 155
def thursday_minutes : ℕ := 45

def weekly_minutes : ℕ := monday_minutes + tuesday_minutes + wednesday_minutes + thursday_minutes
def weekly_hours : ℕ := weekly_minutes / minutes_in_hour

def sylvia_earnings : ℕ := weekly_hours * hourly_rate

theorem weekly_earnings_proof :
  sylvia_earnings = 26 := by
  sorry

end weekly_earnings_proof_l100_100269


namespace inequality_range_l100_100817

theorem inequality_range (a : ℝ) : 
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 2 ≤ y ∧ y ≤ 3 → x * y ≤ a * x^2 + 2 * y^2) ↔ a ≥ -1 := by 
  sorry

end inequality_range_l100_100817


namespace max_c_l100_100535

theorem max_c (c : ℝ) : 
  (∀ x y : ℝ, x > y ∧ y > 0 → x^2 - 2 * y^2 ≤ c * x * (y - x)) 
  → c ≤ 2 * Real.sqrt 2 - 4 := 
by
  sorry

end max_c_l100_100535


namespace marketing_firm_l100_100769

variable (Total_households : ℕ) (A_only : ℕ) (A_and_B : ℕ) (B_to_A_and_B_ratio : ℕ)

def neither_soap_households : ℕ :=
  Total_households - (A_only + (B_to_A_and_B_ratio * A_and_B) + A_and_B)

theorem marketing_firm (h1 : Total_households = 300)
                       (h2 : A_only = 60)
                       (h3 : A_and_B = 40)
                       (h4 : B_to_A_and_B_ratio = 3)
                       : neither_soap_households 300 60 40 3 = 80 :=
by {
  sorry
}

end marketing_firm_l100_100769


namespace lopez_family_seating_arrangement_count_l100_100257

def lopez_family_seating_arrangements : Nat := 2 * 4 * 6

theorem lopez_family_seating_arrangement_count : lopez_family_seating_arrangements = 48 :=
by 
    sorry

end lopez_family_seating_arrangement_count_l100_100257


namespace arithmetic_sequence_common_difference_l100_100245

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 2 = 1) (h2 : a 6 = 13) 
  (arithmetic : ∀ n : ℕ, n ≥ 2 → a n = a 1 + (n - 1) * d) : 
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l100_100245


namespace find_x_when_y_neg_five_l100_100424

-- Definitions based on the conditions provided
variable (x y : ℝ)
def inversely_proportional (x y : ℝ) := ∃ (k : ℝ), x * y = k

-- Proving the main result
theorem find_x_when_y_neg_five (h_prop : inversely_proportional x y) (hx4 : x = 4) (hy2 : y = 2) :
    (y = -5) → x = - 8 / 5 := by
  sorry

end find_x_when_y_neg_five_l100_100424


namespace determine_k_coplanar_l100_100711

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable {A B C D : V}
variable (k : ℝ)

theorem determine_k_coplanar (h : 4 • A - 3 • B + 6 • C + k • D = 0) : k = -13 :=
sorry

end determine_k_coplanar_l100_100711


namespace sum_of_first_twelve_terms_l100_100931

-- Define the arithmetic sequence with given conditions
variable {a d : ℚ}

-- The fifth term of the sequence
def a5 : ℚ := a + 4 * d

-- The seventeenth term of the sequence
def a17 : ℚ := a + 16 * d

-- Sum of the first twelve terms of the arithmetic sequence
def S12 (a d : ℚ) : ℚ := 6 * (2 * a + 11 * d)

theorem sum_of_first_twelve_terms (a : ℚ) (d : ℚ) (h₁ : a5 = 1) (h₂ : a17 = 18) :
  S12 a d = 37.5 := by
  sorry

end sum_of_first_twelve_terms_l100_100931


namespace find_g_neg3_l100_100402

def f (x : ℚ) : ℚ := 4 * x - 6
def g (u : ℚ) : ℚ := 3 * (f u)^2 + 4 * (f u) - 2

theorem find_g_neg3 : g (-3) = 43 / 16 := by
  sorry

end find_g_neg3_l100_100402


namespace cos_double_angle_l100_100055

open Real

theorem cos_double_angle (α β : ℝ) 
    (h1 : sin α = 2 * sin β) 
    (h2 : tan α = 3 * tan β) :
  cos (2 * α) = -1 / 4 ∨ cos (2 * α) = 1 := 
sorry

end cos_double_angle_l100_100055


namespace common_rational_root_l100_100996

-- Definitions for the given conditions
def polynomial1 (a b c : ℤ) (x : ℚ) := 50 * x^4 + a * x^3 + b * x^2 + c * x + 16 = 0
def polynomial2 (d e f g : ℤ) (x : ℚ) := 16 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 50 = 0

-- The proof problem statement: Given the conditions, proving that -1/2 is a common rational root
theorem common_rational_root (a b c d e f g : ℤ) (k : ℚ) 
  (h1 : polynomial1 a b c k)
  (h2 : polynomial2 d e f g k) 
  (h3 : ∃ m n : ℤ, k = -((m : ℚ) / n) ∧ Int.gcd m n = 1) :
  k = -1/2 :=
sorry

end common_rational_root_l100_100996


namespace f_half_l100_100051

theorem f_half (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - 2 * x) = 1 / (x ^ 2)) :
  f (1 / 2) = 16 :=
sorry

end f_half_l100_100051


namespace daily_shoppers_correct_l100_100759

noncomputable def daily_shoppers (P : ℝ) : Prop :=
  let weekly_taxes : ℝ := 6580
  let daily_taxes := weekly_taxes / 7
  let percent_taxes := 0.94
  percent_taxes * P = daily_taxes

theorem daily_shoppers_correct : ∃ P : ℝ, daily_shoppers P ∧ P = 1000 :=
by
  sorry

end daily_shoppers_correct_l100_100759


namespace x_lt_1_nec_not_suff_l100_100531

theorem x_lt_1_nec_not_suff (x : ℝ) : (x < 1 → x^2 < 1) ∧ (¬(x < 1) → x^2 < 1) := 
by {
  sorry
}

end x_lt_1_nec_not_suff_l100_100531


namespace GCF_LCM_15_21_14_20_l100_100567

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCF_LCM_15_21_14_20 :
  GCF (LCM 15 21) (LCM 14 20) = 35 :=
by
  sorry

end GCF_LCM_15_21_14_20_l100_100567


namespace dry_mixed_fruits_weight_l100_100167

theorem dry_mixed_fruits_weight :
  ∀ (fresh_grapes_weight fresh_apples_weight : ℕ)
    (grapes_water_content fresh_grapes_dry_matter_perc : ℕ)
    (apples_water_content fresh_apples_dry_matter_perc : ℕ),
    fresh_grapes_weight = 400 →
    fresh_apples_weight = 300 →
    grapes_water_content = 65 →
    fresh_grapes_dry_matter_perc = 35 →
    apples_water_content = 84 →
    fresh_apples_dry_matter_perc = 16 →
    (fresh_grapes_weight * fresh_grapes_dry_matter_perc / 100) +
    (fresh_apples_weight * fresh_apples_dry_matter_perc / 100) = 188 := by
  sorry

end dry_mixed_fruits_weight_l100_100167


namespace total_money_collected_l100_100742

def number_of_people := 610
def price_adult := 2
def price_child := 1
def number_of_adults := 350

theorem total_money_collected :
  (number_of_people - number_of_adults) * price_child + number_of_adults * price_adult = 960 := by
  sorry

end total_money_collected_l100_100742


namespace ellipse_equation_constants_l100_100622

noncomputable def ellipse_parametric_eq (t : ℝ) : ℝ × ℝ :=
  ((3 * (Real.sin t - 2)) / (3 - Real.cos t),
  (4 * (Real.cos t - 4)) / (3 - Real.cos t))

theorem ellipse_equation_constants :
  ∃ (A B C D E F : ℤ), ∀ (x y : ℝ),
  ((∃ t : ℝ, (x, y) = ellipse_parametric_eq t) → (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) ∧
  (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd A B) C) D) E) F = 1) ∧
  (|A| + |B| + |C| + |D| + |E| + |F| = 2502) :=
sorry

end ellipse_equation_constants_l100_100622


namespace cos_3theta_value_l100_100690

open Real

noncomputable def cos_3theta (theta : ℝ) : ℝ := 4 * (cos theta)^3 - 3 * (cos theta)

theorem cos_3theta_value (theta : ℝ) (h : cos theta = 1 / 3) : cos_3theta theta = - 23 / 27 :=
by
  sorry

end cos_3theta_value_l100_100690


namespace floor_ceil_product_l100_100924

theorem floor_ceil_product :
  let f : ℕ → ℤ := λ n, (Int.floor (- (n : ℤ) - 0.5)) * (Int.ceil ((n : ℤ) + 0.5))
  let product : ℤ := ∏ i in Finset.range 7, -(i + 1)^2
  ∑ n in Finset.range 7, f n = product :=
by
  sorry

end floor_ceil_product_l100_100924


namespace all_positive_integers_are_clever_l100_100452

theorem all_positive_integers_are_clever : ∀ n : ℕ, 0 < n → ∃ a b c d : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ n = (a^2 - b^2) / (c^2 + d^2) := 
by
  intros n h_pos
  sorry

end all_positive_integers_are_clever_l100_100452


namespace box_upper_surface_area_l100_100428

theorem box_upper_surface_area (L W H : ℕ) 
    (h1 : L * W = 120) 
    (h2 : L * H = 72) 
    (h3 : L * W * H = 720) : 
    L * W = 120 := 
by 
  sorry

end box_upper_surface_area_l100_100428


namespace symmetric_line_l100_100729

theorem symmetric_line (x y : ℝ) : (2 * x + y - 4 = 0) → (2 * x - y + 4 = 0) :=
by
  sorry

end symmetric_line_l100_100729


namespace time_boarding_in_London_l100_100029

open Nat

def time_in_ET_to_London_time (time_et: ℕ) : ℕ :=
  (time_et + 5) % 24

def subtract_hours (time: ℕ) (hours: ℕ) : ℕ :=
  (time + 24 * (hours / 24) - (hours % 24)) % 24

theorem time_boarding_in_London :
  let cape_town_arrival_time_et := 10
  let flight_duration_ny_to_cape := 10
  let ny_departure_time := subtract_hours cape_town_arrival_time_et flight_duration_ny_to_cape
  let flight_duration_london_to_ny := 18
  let ny_arrival_time := subtract_hours ny_departure_time flight_duration_london_to_ny
  let london_time := time_in_ET_to_London_time ny_arrival_time
  let london_departure_time := subtract_hours london_time flight_duration_london_to_ny
  london_departure_time = 17 :=
by
  -- Proof omitted
  sorry

end time_boarding_in_London_l100_100029


namespace solid_produces_quadrilateral_l100_100776

-- Define the solids and their properties
inductive Solid 
| cone 
| cylinder 
| sphere

-- Define the condition for a plane cut resulting in a quadrilateral cross-section
def can_produce_quadrilateral_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.cone => False
  | Solid.cylinder => True
  | Solid.sphere => False

-- Theorem to prove that only a cylinder can produce a quadrilateral cross-section
theorem solid_produces_quadrilateral : 
  ∃ s : Solid, can_produce_quadrilateral_cross_section s :=
by
  existsi Solid.cylinder
  trivial

end solid_produces_quadrilateral_l100_100776


namespace find_Y_length_l100_100246

theorem find_Y_length (Y : ℝ) : 
  (3 + 2 + 3 + 4 + Y = 7 + 4 + 2) → Y = 1 :=
by
  intro h
  sorry

end find_Y_length_l100_100246


namespace range_of_a_l100_100195

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) := 
by 
  sorry

end range_of_a_l100_100195


namespace midpoint_trajectory_of_moving_point_l100_100937

/-- Given a fixed point A (4, -3) and a moving point B on the circle (x+1)^2 + y^2 = 4, prove that 
    the equation of the trajectory of the midpoint M of the line segment AB is 
    (x - 3/2)^2 + (y + 3/2)^2 = 1. -/
theorem midpoint_trajectory_of_moving_point {x y : ℝ} :
  (∃ (B : ℝ × ℝ), (B.1 + 1)^2 + B.2^2 = 4 ∧ 
    (x, y) = ((B.1 + 4) / 2, (B.2 - 3) / 2)) →
  (x - 3/2)^2 + (y + 3/2)^2 = 1 :=
by sorry

end midpoint_trajectory_of_moving_point_l100_100937


namespace car_not_sold_probability_l100_100829

theorem car_not_sold_probability (a b : ℕ) (h : a = 5) (k : b = 6) : (b : ℚ) / (a + b : ℚ) = 6 / 11 :=
  by
    rw [h, k]
    norm_num

end car_not_sold_probability_l100_100829


namespace y_value_l100_100544

theorem y_value (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -6) : y = 29 :=
by
  sorry

end y_value_l100_100544


namespace centroid_distance_l100_100971

-- Define the given conditions and final goal
theorem centroid_distance (a b c p q r : ℝ) 
  (ha : a ≠ 0)  (hb : b ≠ 0)  (hc : c ≠ 0)
  (centroid : p = a / 3 ∧ q = b / 3 ∧ r = c / 3) 
  (plane_distance : (1 / (1 / a^2 + 1 / b^2 + 1 / c^2).sqrt) = 2) :
  (1 / p^2 + 1 / q^2 + 1 / r^2) = 2.25 := 
by 
  -- Start proof here
  sorry

end centroid_distance_l100_100971


namespace choose_positions_from_8_people_l100_100836

theorem choose_positions_from_8_people : 
  ∃ (ways : ℕ), ways = 8 * 7 * 6 := 
sorry

end choose_positions_from_8_people_l100_100836


namespace exist_a_b_for_every_n_l100_100719

theorem exist_a_b_for_every_n (n : ℕ) (hn : 0 < n) : 
  ∃ (a b : ℤ), 1 < a ∧ 1 < b ∧ a^2 + 1 = 2 * b^2 ∧ (a - b) % n = 0 := 
sorry

end exist_a_b_for_every_n_l100_100719


namespace area_of_triangle_QCA_l100_100036

noncomputable def triangle_area (x p : ℝ) (hx : x > 0) (hp : p < 12) : ℝ :=
  1 / 2 * x * (12 - p)

theorem area_of_triangle_QCA (x p : ℝ) (hx : x > 0) (hp : p < 12) :
  triangle_area x p hx hp = x * (12 - p) / 2 := by
  sorry

end area_of_triangle_QCA_l100_100036


namespace stamps_initial_count_l100_100121

theorem stamps_initial_count (total_stamps stamps_received initial_stamps : ℕ) 
  (h1 : total_stamps = 61)
  (h2 : stamps_received = 27)
  (h3 : initial_stamps = total_stamps - stamps_received) :
  initial_stamps = 34 :=
sorry

end stamps_initial_count_l100_100121


namespace cannot_form_right_triangle_l100_100456

theorem cannot_form_right_triangle :
  ¬ (6^2 + 7^2 = 8^2) :=
by
  sorry

end cannot_form_right_triangle_l100_100456


namespace parabola_constant_term_l100_100905

theorem parabola_constant_term (b c : ℝ)
  (h1 : 2 * b + c = 8)
  (h2 : -2 * b + c = -4)
  (h3 : 4 * b + c = 24) :
  c = 2 :=
sorry

end parabola_constant_term_l100_100905


namespace points_coincide_l100_100591

open EuclideanGeometry

theorem points_coincide 
  (A B C : Point) 
  (h : Triangle ABC)
  (circ : circumcircle h)
  (A1 B1 C1 : Point)
  (A1_diam : diametrically_opposite circ A A1)
  (B1_diam : diametrically_opposite circ B B1)
  (C1_diam : diametrically_opposite circ C C1)
  (A0 B0 C0 : Point)
  (A0_mid : midpoint A0 B C)
  (B0_mid : midpoint B0 A C)
  (C0_mid : midpoint C0 A B)
  (A2 B2 C2 : Point)
  (A2_symm : symmetric A1 A0 A2)
  (B2_symm : symmetric B1 B0 B2)
  (C2_symm : symmetric C1 C0 C2) 
: A2 = B2 ∧ B2 = C2 :=
sorry

end points_coincide_l100_100591


namespace least_number_divisible_by_11_and_leaves_remainder_2_l100_100746

theorem least_number_divisible_by_11_and_leaves_remainder_2 : 
  ∃ n : ℕ, (n % 11 = 0) ∧ (∀ m ∈ {3, 4, 5, 6, 7}, n % m = 2) ∧ n = 3782 :=
by
  sorry

end least_number_divisible_by_11_and_leaves_remainder_2_l100_100746


namespace Bryan_has_more_skittles_l100_100493

-- Definitions for conditions
def Bryan_skittles : ℕ := 50
def Ben_mms : ℕ := 20

-- Main statement to be proven
theorem Bryan_has_more_skittles : Bryan_skittles > Ben_mms ∧ Bryan_skittles - Ben_mms = 30 :=
by
  sorry

end Bryan_has_more_skittles_l100_100493


namespace triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero_l100_100854

theorem triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero
  (A B C : ℝ) (h : A + B + C = 180): 
    (A = 60 ∨ B = 60 ∨ C = 60) ↔ (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 0) := 
by
  sorry

end triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero_l100_100854


namespace average_income_BC_l100_100989

theorem average_income_BC {A_income B_income C_income : ℝ}
  (hAB : (A_income + B_income) / 2 = 4050)
  (hAC : (A_income + C_income) / 2 = 4200)
  (hA : A_income = 3000) :
  (B_income + C_income) / 2 = 5250 :=
by sorry

end average_income_BC_l100_100989


namespace number_of_girls_is_eleven_l100_100183

-- Conditions transformation
def boys_wear_red_hats : Prop := true
def girls_wear_yellow_hats : Prop := true
def teachers_wear_blue_hats : Prop := true
def cannot_see_own_hat : Prop := true
def little_qiang_sees_hats (x k : ℕ) : Prop := (x + 2) = (x + 2)
def little_hua_sees_hats (x k : ℕ) : Prop := x = 2 * k
def teacher_sees_hats (x k : ℕ) : Prop := k + 2 = (x + 2) + k - 11

-- Proof Statement
theorem number_of_girls_is_eleven (x k : ℕ) (h1 : boys_wear_red_hats)
  (h2 : girls_wear_yellow_hats) (h3 : teachers_wear_blue_hats)
  (h4 : cannot_see_own_hat) (hq : little_qiang_sees_hats x k)
  (hh : little_hua_sees_hats x k) (ht : teacher_sees_hats x k) : x = 11 :=
sorry

end number_of_girls_is_eleven_l100_100183


namespace max_complexity_51_l100_100976

-- Define the complexity of a number 
def complexity (x : ℚ) : ℕ := sorry -- Placeholder for the actual complexity function definition

-- Define the sequence for m values
def m_sequence (k : ℕ) : List ℕ :=
  List.range' 1 (2^(k-1)) |>.filter (λ n => n % 2 = 1)

-- Define the candidate number
def candidate_number (k : ℕ) : ℚ :=
  (2^(k + 1) + (-1)^k) / (3 * 2^k)

theorem max_complexity_51 : 
  ∃ m, m ∈ m_sequence 50 ∧ 
  (∀ n, n ∈ m_sequence 50 → complexity (n / 2^50) ≤ complexity (candidate_number 50 / 2^50)) :=
sorry

end max_complexity_51_l100_100976


namespace find_x_when_y_neg_five_l100_100423

-- Definitions based on the conditions provided
variable (x y : ℝ)
def inversely_proportional (x y : ℝ) := ∃ (k : ℝ), x * y = k

-- Proving the main result
theorem find_x_when_y_neg_five (h_prop : inversely_proportional x y) (hx4 : x = 4) (hy2 : y = 2) :
    (y = -5) → x = - 8 / 5 := by
  sorry

end find_x_when_y_neg_five_l100_100423


namespace tan_theta_solution_l100_100920

theorem tan_theta_solution (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < 15) 
  (h_tan_eq : Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) = 0) :
  Real.tan θ = 1 / Real.sqrt 2 :=
sorry

end tan_theta_solution_l100_100920


namespace widgets_made_per_week_l100_100841

theorem widgets_made_per_week
  (widgets_per_hour : Nat)
  (hours_per_day : Nat)
  (days_per_week : Nat)
  (total_widgets : Nat) :
  widgets_per_hour = 20 →
  hours_per_day = 8 →
  days_per_week = 5 →
  total_widgets = widgets_per_hour * hours_per_day * days_per_week →
  total_widgets = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end widgets_made_per_week_l100_100841


namespace log_function_domain_l100_100991

theorem log_function_domain {x : ℝ} (h : 1 / x - 1 > 0) : 0 < x ∧ x < 1 :=
sorry

end log_function_domain_l100_100991


namespace numValidSeqs_solution_l100_100655

def isValidSeq (A : Fin 10 → Fin 10) : Prop :=
  (Perm.isPerm A) ∧
  (A 0 < A 1 ∧ A 2 < A 3 ∧ A 4 < A 5 ∧ A 6 < A 7 ∧ A 8 < A 9) ∧ 
  (A 1 > A 2 ∧ A 3 > A 4 ∧ A 5 > A 6 ∧ A 7 > A 8) ∧
  (∀ i j k : Fin 10, i < j ∧ j < k → ¬ (A i < A k ∧ A k < A j))

theorem numValidSeqs : (Fin 10 → Fin 10) → ℕ
  := sorry

theorem solution : numValidSeqs = 42 :=
by
  sorry

end numValidSeqs_solution_l100_100655


namespace toothpicks_in_300th_stage_l100_100303

/-- 
Prove that the number of toothpicks needed for the 300th stage is 1201, given:
1. The first stage has 5 toothpicks.
2. Each subsequent stage adds 4 toothpicks to the previous stage.
-/
theorem toothpicks_in_300th_stage :
  let a_1 := 5
  let d := 4
  let a_n (n : ℕ) := a_1 + (n - 1) * d
  a_n 300 = 1201 := by
  sorry

end toothpicks_in_300th_stage_l100_100303


namespace blue_polygons_more_than_red_polygons_l100_100718

open_locale big_operators

theorem blue_polygons_more_than_red_polygons (red_points : Finset ℕ) (H : red_points.card = 40) :
  let blue := 1,
      red_polygons := ∑ k in range (40 + 1), if 3 ≤ k then finset.card (finset.powerset_len k.red_points) else 0,
      blue_polygons := ∑ k in finset.range 40, if 2 ≤ k then finset.card (finset.powerset_len k (red_points ∪ {blue})) else 0
  in blue_polygons - red_polygons = 780 := sorry

end blue_polygons_more_than_red_polygons_l100_100718


namespace age_difference_l100_100413

theorem age_difference (p f : ℕ) (hp : p = 11) (hf : f = 42) : f - p = 31 :=
by
  sorry

end age_difference_l100_100413


namespace ratio_of_speeds_l100_100200

/-- Define the conditions -/
def distance_AB : ℝ := 540 -- Distance between city A and city B is 540 km
def time_Eddy : ℝ := 3     -- Eddy takes 3 hours to travel to city B
def distance_AC : ℝ := 300 -- Distance between city A and city C is 300 km
def time_Freddy : ℝ := 4   -- Freddy takes 4 hours to travel to city C

/-- Define the average speeds -/
noncomputable def avg_speed_Eddy : ℝ := distance_AB / time_Eddy
noncomputable def avg_speed_Freddy : ℝ := distance_AC / time_Freddy

/-- The statement to prove -/
theorem ratio_of_speeds : avg_speed_Eddy / avg_speed_Freddy = 12 / 5 :=
by sorry

end ratio_of_speeds_l100_100200


namespace xy_condition_l100_100824

variable (x y : ℝ) -- This depends on the problem context specifying real numbers.

theorem xy_condition (h : x ≠ 0 ∧ y ≠ 0) : (x + y = 0 ↔ y / x + x / y = -2) :=
  sorry

end xy_condition_l100_100824


namespace find_sum_of_perimeters_l100_100429

variables (x y : ℝ)
noncomputable def sum_of_perimeters := 4 * x + 4 * y

theorem find_sum_of_perimeters (h1 : x^2 + y^2 = 65) (h2 : x^2 - y^2 = 33) :
  sum_of_perimeters x y = 44 :=
sorry

end find_sum_of_perimeters_l100_100429


namespace find_AV_length_l100_100163

noncomputable def circle_A_center : ℝ × ℝ := (0, 0)
noncomputable def circle_A_radius : ℝ := 10
noncomputable def circle_B_center : ℝ × ℝ := (13, 0)
noncomputable def circle_B_radius : ℝ := 3

theorem find_AV_length :
  let A := circle_A_center,
  let B := circle_B_center,
  let U : ℝ × ℝ := (10, 0),
  let V : ℝ × ℝ := (13, 3),
  let AV := dist A V in
  AV = 2 * Real.sqrt 55 :=
by
  -- Without proof
  sorry

end find_AV_length_l100_100163


namespace range_of_m_l100_100061

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4^x + m * 2^x + m^2 - 1 = 0) ↔ - (2 * Real.sqrt 3) / 3 ≤ m ∧ m < 1 :=
sorry

end range_of_m_l100_100061


namespace total_time_to_clean_and_complete_l100_100307

def time_to_complete_assignment : Nat := 10
def num_remaining_keys : Nat := 14
def time_per_key : Nat := 3

theorem total_time_to_clean_and_complete :
  time_to_complete_assignment + num_remaining_keys * time_per_key = 52 :=
by
  sorry

end total_time_to_clean_and_complete_l100_100307


namespace football_team_throwers_l100_100262

theorem football_team_throwers {T N : ℕ} (h1 : 70 - T = N)
                                (h2 : 62 = T + (2 / 3 * N)) : 
                                T = 46 := 
by
  sorry

end football_team_throwers_l100_100262


namespace trajectory_of_Q_is_parabola_l100_100368

/--
Given a point P (x, y) moves on a unit circle centered at the origin,
prove that the trajectory of point Q (u, v) defined by u = x + y and v = xy 
satisfies u^2 - 2v = 1 and is thus a parabola.
-/
theorem trajectory_of_Q_is_parabola 
  (x y u v : ℝ) 
  (h1 : x^2 + y^2 = 1) 
  (h2 : u = x + y) 
  (h3 : v = x * y) :
  u^2 - 2 * v = 1 :=
sorry

end trajectory_of_Q_is_parabola_l100_100368


namespace find_e_l100_100486

variables (j p t b a : ℝ) (e : ℝ)

theorem find_e
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (e / 100) * p)
  (h4 : b = 1.40 * j)
  (h5 : a = 0.85 * b)
  (h6 : e = 2 * ((p - a) / p) * 100) :
  e = 21.5 := by
  sorry

end find_e_l100_100486


namespace proof_equivalence_l100_100059

variables {a b c d e f : Prop}

theorem proof_equivalence (h₁ : (a ≥ b) → (c > d)) 
                        (h₂ : (c > d) → (a ≥ b)) 
                        (h₃ : (a < b) ↔ (e ≤ f)) :
  (c ≤ d) ↔ (e ≤ f) :=
sorry

end proof_equivalence_l100_100059


namespace calculate_E_l100_100488

theorem calculate_E (P J T B A E : ℝ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : B = 1.40 * J)
  (h4 : A = 0.85 * B)
  (h5 : T = P - (E / 100) * P)
  (h6 : E = 2 * ((P - A) / P) * 100) : 
  E = 21.5 := 
sorry

end calculate_E_l100_100488


namespace math_problem_l100_100027

theorem math_problem :
  (-1:ℤ) ^ 2023 - |(-3:ℤ)| + ((-1/3:ℚ) ^ (-2:ℤ)) + ((Real.pi - 3.14)^0) = 6 := 
by 
  sorry

end math_problem_l100_100027


namespace average_test_score_first_25_percent_l100_100239

theorem average_test_score_first_25_percent (x : ℝ) :
  (0.25 * x) + (0.50 * 65) + (0.25 * 90) = 1 * 75 → x = 80 :=
by
  sorry

end average_test_score_first_25_percent_l100_100239


namespace sum_of_roots_l100_100932

theorem sum_of_roots :
  let a := (6 : ℝ) + 3 * Real.sqrt 3
  let b := (3 : ℝ) + Real.sqrt 3
  let c := -(3 : ℝ)
  let root_sum := -b / a
  root_sum = -1 + Real.sqrt 3 / 3 := sorry

end sum_of_roots_l100_100932


namespace value_of_f_at_6_l100_100369

-- The condition that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- The condition that f(x + 2) = -f(x)
def periodic_sign_flip (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = -f (x)

-- The theorem statement
theorem value_of_f_at_6 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : periodic_sign_flip f) : f 6 = 0 :=
sorry

end value_of_f_at_6_l100_100369


namespace line_through_chord_with_midpoint_l100_100940

theorem line_through_chord_with_midpoint (x y : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (x = x1 ∧ y = y1 ∨ x = x2 ∧ y = y2) ∧
    x = -1 ∧ y = 1 ∧
    x1^2 / 4 + y1^2 / 3 = 1 ∧
    x2^2 / 4 + y2^2 / 3 = 1) →
  3 * x - 4 * y + 7 = 0 :=
by
  sorry

end line_through_chord_with_midpoint_l100_100940


namespace males_in_sample_l100_100900

theorem males_in_sample (total_employees female_employees sample_size : ℕ) 
  (h1 : total_employees = 300)
  (h2 : female_employees = 160)
  (h3 : sample_size = 15)
  (h4 : (female_employees * sample_size) / total_employees = 8) :
  sample_size - ((female_employees * sample_size) / total_employees) = 7 :=
by
  sorry

end males_in_sample_l100_100900


namespace croissant_to_orange_ratio_l100_100201

-- Define the conditions as given in the problem
variables (c o : ℝ)
variable (emily_expenditure : ℝ)
variable (lucas_expenditure : ℝ)

-- Given conditions of expenditures
axiom emily_expenditure_is : emily_expenditure = 5 * c + 4 * o
axiom lucas_expenditure_is : lucas_expenditure = 3 * emily_expenditure
axiom lucas_expenditure_as_purchased : lucas_expenditure = 4 * c + 10 * o

-- Prove the ratio of the cost of a croissant to an orange
theorem croissant_to_orange_ratio : (c / o) = 2 / 11 :=
by sorry

end croissant_to_orange_ratio_l100_100201


namespace smallest_positive_integer_l100_100196

theorem smallest_positive_integer (N : ℕ) :
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 7 = 4) ∧ (N % 11 = 9) → N = 207 :=
by
  sorry

end smallest_positive_integer_l100_100196


namespace ratio_of_wilted_roses_to_total_l100_100708

-- Defining the conditions
def initial_roses := 24
def traded_roses := 12
def total_roses := initial_roses + traded_roses
def remaining_roses_after_second_night := 9
def roses_before_second_night := remaining_roses_after_second_night * 2
def wilted_roses_after_first_night := total_roses - roses_before_second_night
def ratio_wilted_to_total := wilted_roses_after_first_night / total_roses

-- Proving the ratio of wilted flowers to the total number of flowers after the first night is 1:2
theorem ratio_of_wilted_roses_to_total :
  ratio_wilted_to_total = (1/2) := by
  sorry

end ratio_of_wilted_roses_to_total_l100_100708


namespace largest_sum_is_three_fourths_l100_100784

-- Definitions of sums
def sum1 := (1 / 4) + (1 / 2)
def sum2 := (1 / 4) + (1 / 9)
def sum3 := (1 / 4) + (1 / 3)
def sum4 := (1 / 4) + (1 / 10)
def sum5 := (1 / 4) + (1 / 6)

-- The theorem stating that sum1 is the maximum of the sums
theorem largest_sum_is_three_fourths : max (max (max (max sum1 sum2) sum3) sum4) sum5 = 3 / 4 := 
sorry

end largest_sum_is_three_fourths_l100_100784


namespace find_dividend_l100_100547

theorem find_dividend (divisor quotient remainder : ℕ) (h_divisor : divisor = 38) (h_quotient : quotient = 19) (h_remainder : remainder = 7) :
  divisor * quotient + remainder = 729 := by
  sorry

end find_dividend_l100_100547


namespace turnip_weights_are_13_or_16_l100_100002

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l100_100002


namespace no_integer_pairs_satisfy_equation_l100_100357

theorem no_integer_pairs_satisfy_equation :
  ¬ ∃ m n : ℤ, m^3 + 8 * m^2 + 17 * m = 8 * n^3 + 12 * n^2 + 6 * n + 1 :=
sorry

end no_integer_pairs_satisfy_equation_l100_100357


namespace volume_relation_l100_100632

theorem volume_relation 
  (r h : ℝ) 
  (heightC_eq_three_times_radiusD : h = 3 * r)
  (radiusC_eq_heightD : r = h)
  (volumeD_eq_three_times_volumeC : ∀ (π : ℝ), 3 * (π * h^2 * r) = π * r^2 * h) :
  3 = (3 : ℝ) := 
by
  sorry

end volume_relation_l100_100632


namespace gcd_840_1764_l100_100997

theorem gcd_840_1764 : gcd 840 1764 = 84 := 
by
  -- proof steps will go here
  sorry

end gcd_840_1764_l100_100997


namespace prime_pairs_sum_50_l100_100678

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of relevant primes under 25
def primes_under_25 : set ℕ := { p | is_prime p ∧ p < 25 }

-- A helper definition to count the pairs of primes that sum to 50
def prime_pairs_sum_50_count : ℕ :=
  (primes_under_25.to_finset.filter (λ p, is_prime (50 - p) ∧ (50 - p) < p)).card

-- Statement of the problem
theorem prime_pairs_sum_50 : prime_pairs_sum_50_count = 4 := sorry

end prime_pairs_sum_50_l100_100678


namespace odd_integer_95th_l100_100453

theorem odd_integer_95th : (2 * 95 - 1) = 189 := 
by
  -- The proof would go here
  sorry

end odd_integer_95th_l100_100453


namespace quadratic_function_properties_l100_100362

def quadratic_function (x : ℝ) : ℝ :=
  -6 * x^2 + 36 * x - 48

theorem quadratic_function_properties :
  quadratic_function 2 = 0 ∧ quadratic_function 4 = 0 ∧ quadratic_function 3 = 6 :=
by
  -- The proof is omitted
  -- Placeholder for the proof
  sorry

end quadratic_function_properties_l100_100362


namespace verify_polynomial_relationship_l100_100067

theorem verify_polynomial_relationship :
  (∀ x : ℕ, f x = x^3 + 2 * x + 1) ∧
  f 1 = 4 ∧
  f 2 = 15 ∧
  f 3 = 40 ∧
  f 4 = 85 ∧
  f 5 = 156 :=
by
  let f := λ x : ℕ, x^3 + 2 * x + 1
  split; intros;
  { sorry }

end verify_polynomial_relationship_l100_100067


namespace change_is_13_82_l100_100704

def sandwich_cost : ℝ := 5
def num_sandwiches : ℕ := 3
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05
def payment : ℝ := 20 + 5 + 3

def total_cost_before_discount : ℝ := num_sandwiches * sandwich_cost
def discount_amount : ℝ := total_cost_before_discount * discount_rate
def discounted_cost : ℝ := total_cost_before_discount - discount_amount
def tax_amount : ℝ := discounted_cost * tax_rate
def total_cost_after_tax : ℝ := discounted_cost + tax_amount

def change (payment total_cost : ℝ) : ℝ := payment - total_cost

theorem change_is_13_82 : change payment total_cost_after_tax = 13.82 := 
by
  -- Proof will be provided here
  sorry

end change_is_13_82_l100_100704


namespace arithmetic_geometric_sequences_l100_100814

theorem arithmetic_geometric_sequences :
  ∀ (a₁ a₂ b₁ b₂ b₃ : ℤ), 
  (a₂ = a₁ + (a₁ - (-1))) ∧ 
  (-4 = -1 + 3 * (a₂ - a₁)) ∧ 
  (-4 = -1 * (b₃/b₁)^4) ∧ 
  (b₂ = b₁ * (b₂/b₁)^2) →
  (a₂ - a₁) / b₂ = 1 / 2 := 
by
  intros a₁ a₂ b₁ b₂ b₃ h
  sorry

end arithmetic_geometric_sequences_l100_100814


namespace sally_weekend_reading_l100_100856

theorem sally_weekend_reading (pages_on_weekdays : ℕ) (total_pages : ℕ) (weeks : ℕ) (weekdays_per_week : ℕ) (total_days : ℕ) 
  (finishing_time : ℕ) (weekend_days : ℕ) (pages_weekdays_total : ℕ) :
  pages_on_weekdays = 10 →
  total_pages = 180 →
  weeks = 2 →
  weekdays_per_week = 5 →
  weekend_days = (total_days - weekdays_per_week * weeks) →
  total_days = 7 * weeks →
  finishing_time = weeks →
  pages_weekdays_total = pages_on_weekdays * weekdays_per_week * weeks →
  (total_pages - pages_weekdays_total) / weekend_days = 20 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end sally_weekend_reading_l100_100856


namespace max_value_of_function_l100_100136

noncomputable def max_value (x : ℝ) : ℝ := 3 * Real.sin x + 2

theorem max_value_of_function : 
  ∀ x : ℝ, (- (Real.pi / 2)) ≤ x ∧ x ≤ 0 → max_value x ≤ 2 :=
sorry

end max_value_of_function_l100_100136


namespace tank_leak_time_l100_100484

/--
The rate at which the tank is filled without a leak is R = 1/5 tank per hour.
The effective rate with the leak is 1/6 tank per hour.
Prove that the time it takes for the leak to empty the full tank is 30 hours.
-/
theorem tank_leak_time (R : ℝ) (L : ℝ) (h1 : R = 1 / 5) (h2 : R - L = 1 / 6) :
  1 / L = 30 :=
by
  sorry

end tank_leak_time_l100_100484


namespace find_shirts_yesterday_l100_100334

def shirts_per_minute : ℕ := 8
def total_minutes : ℕ := 2
def shirts_today : ℕ := 3

def total_shirts : ℕ := shirts_per_minute * total_minutes
def shirts_yesterday : ℕ := total_shirts - shirts_today

theorem find_shirts_yesterday : shirts_yesterday = 13 := by
  sorry

end find_shirts_yesterday_l100_100334


namespace no_nat_nums_x4_minus_y4_eq_x3_plus_y3_l100_100198

theorem no_nat_nums_x4_minus_y4_eq_x3_plus_y3 : ∀ (x y : ℕ), x^4 - y^4 ≠ x^3 + y^3 :=
by
  intro x y
  sorry

end no_nat_nums_x4_minus_y4_eq_x3_plus_y3_l100_100198


namespace part1_prove_BD_eq_b_part2_prove_cos_ABC_l100_100558

-- Definition of the problem setup
variables {a b c : ℝ}
variables {A B C : ℝ}    -- angles
variables {D : ℝ}        -- point on side AC

-- Conditions
axiom b_squared_eq_ac : b^2 = a * c
axiom BD_sin_ABC_eq_a_sin_C : D * Real.sin B = a * Real.sin C
axiom AD_eq_2DC : A = 2 * C

-- Part (1)
theorem part1_prove_BD_eq_b : D = b :=
by
  sorry

-- Part (2)
theorem part2_prove_cos_ABC :
  Real.cos B = 7 / 12 :=
by
  sorry

end part1_prove_BD_eq_b_part2_prove_cos_ABC_l100_100558


namespace paper_pieces_l100_100508

theorem paper_pieces (n : ℕ) (h1 : 20 = 2 * n - 8) : n^2 + 20 = 216 := 
by
  sorry

end paper_pieces_l100_100508


namespace students_neither_play_l100_100831

theorem students_neither_play (total_students football_players tennis_players both_players neither_players : ℕ)
  (h1 : total_students = 40)
  (h2 : football_players = 26)
  (h3 : tennis_players = 20)
  (h4 : both_players = 17)
  (h5 : neither_players = total_students - (football_players + tennis_players - both_players)) :
  neither_players = 11 :=
by
  sorry

end students_neither_play_l100_100831


namespace total_cups_of_liquid_drunk_l100_100791

-- Definitions for the problem conditions
def elijah_pints : ℝ := 8.5
def emilio_pints : ℝ := 9.5
def cups_per_pint : ℝ := 2
def elijah_cups : ℝ := elijah_pints * cups_per_pint
def emilio_cups : ℝ := emilio_pints * cups_per_pint
def total_cups : ℝ := elijah_cups + emilio_cups

-- Theorem to prove the required equality
theorem total_cups_of_liquid_drunk : total_cups = 36 :=
by
  sorry

end total_cups_of_liquid_drunk_l100_100791


namespace Sherman_weekly_driving_time_l100_100415

theorem Sherman_weekly_driving_time (daily_commute : ℕ := 30) (weekend_drive : ℕ := 2) : 
  (5 * (2 * daily_commute) / 60 + 2 * weekend_drive) = 9 := 
by
  sorry

end Sherman_weekly_driving_time_l100_100415


namespace sqrt_interval_l100_100184

theorem sqrt_interval :
  let expr := (Real.sqrt 18) / 3 - (Real.sqrt 2) * (Real.sqrt (1 / 2))
  0 < expr ∧ expr < 1 :=
by
  let expr := (Real.sqrt 18) / 3 - (Real.sqrt 2) * (Real.sqrt (1 / 2))
  sorry

end sqrt_interval_l100_100184


namespace shaded_region_occupies_32_percent_of_total_area_l100_100247

-- Conditions
def angle_sector := 90
def r_small := 1
def r_large := 3
def r_sector := 4

-- Question: Prove the shaded region occupies 32% of the total area given the conditions
theorem shaded_region_occupies_32_percent_of_total_area :
  let area_large_sector := (1 / 4) * Real.pi * (r_sector ^ 2)
  let area_small_sector := (1 / 4) * Real.pi * (r_large ^ 2)
  let total_area := area_large_sector + area_small_sector
  let shaded_area := (1 / 4) * Real.pi * (r_large ^ 2) - (1 / 4) * Real.pi * (r_small ^ 2)
  let shaded_percent := (shaded_area / total_area) * 100
  shaded_percent = 32 := by
  sorry

end shaded_region_occupies_32_percent_of_total_area_l100_100247


namespace avg_of_8_numbers_l100_100447

theorem avg_of_8_numbers
  (n : ℕ)
  (h₁ : n = 8)
  (sum_first_half : ℝ)
  (h₂ : sum_first_half = 158.4)
  (avg_second_half : ℝ)
  (h₃ : avg_second_half = 46.6) :
  ((sum_first_half + avg_second_half * (n / 2)) / n) = 43.1 :=
by
  sorry

end avg_of_8_numbers_l100_100447


namespace average_mark_of_all_three_boys_is_432_l100_100834

noncomputable def max_score : ℝ := 900
noncomputable def get_score (percent : ℝ) : ℝ := (percent / 100) * max_score

noncomputable def amar_score : ℝ := get_score 64
noncomputable def bhavan_score : ℝ := get_score 36
noncomputable def chetan_score : ℝ := get_score 44

noncomputable def total_score : ℝ := amar_score + bhavan_score + chetan_score
noncomputable def average_score : ℝ := total_score / 3

theorem average_mark_of_all_three_boys_is_432 : average_score = 432 := 
by
  sorry

end average_mark_of_all_three_boys_is_432_l100_100834


namespace triangle_angle_contradiction_l100_100886

theorem triangle_angle_contradiction (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h4 : A + B + C = 180) :
  A > 60 → B > 60 → C > 60 → false :=
by
  sorry

end triangle_angle_contradiction_l100_100886


namespace maximum_k_inequality_l100_100356

open Real

noncomputable def inequality_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : Prop :=
  (x / sqrt (y + z)) + (y / sqrt (z + x)) + (z / sqrt (x + y)) ≥ sqrt (3 / 2) * sqrt (x + y + z)
 
-- This is the theorem statement
theorem maximum_k_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  inequality_problem x y z h1 h2 h3 :=
  sorry

end maximum_k_inequality_l100_100356


namespace rectangle_width_decrease_l100_100131

theorem rectangle_width_decrease (L W : ℝ) (h1 : 0 < L) (h2 : 0 < W) 
(h3 : ∀ W' : ℝ, 0 < W' → (1.3 * L * W' = L * W) → W' = (100 - 23.077) / 100 * W) : 
  ∃ W' : ℝ, 0 < W' ∧ (1.3 * L * W' = L * W) ∧ ((W - W') / W = 23.077 / 100) :=
by
  sorry

end rectangle_width_decrease_l100_100131


namespace central_angle_unit_circle_l100_100083

theorem central_angle_unit_circle :
  ∀ (θ : ℝ), (∃ (A : ℝ), A = 1 ∧ (A = 1 / 2 * θ)) → θ = 2 :=
by
  intro θ
  rintro ⟨A, hA1, hA2⟩
  sorry

end central_angle_unit_circle_l100_100083


namespace turnip_bag_weight_l100_100008

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l100_100008


namespace right_triangle_area_l100_100733

theorem right_triangle_area (a b c : ℝ) (h1 : c = 13) (h2 : a = 5) (h3 : a^2 + b^2 = c^2) : 0.5 * a * b = 30 := by
  sorry

end right_triangle_area_l100_100733


namespace custom_op_seven_three_l100_100236

def custom_op (a b : ℕ) : ℕ := 4 * a + 5 * b - a * b + 1

theorem custom_op_seven_three : custom_op 7 3 = 23 := by
  -- proof steps would go here
  sorry

end custom_op_seven_three_l100_100236


namespace distance_to_place_is_24_l100_100773

-- Definitions of the problem's conditions
def rowing_speed_still_water := 10    -- kmph
def current_velocity := 2             -- kmph
def round_trip_time := 5              -- hours

-- Effective speeds
def effective_speed_with_current := rowing_speed_still_water + current_velocity
def effective_speed_against_current := rowing_speed_still_water - current_velocity

-- Define the unknown distance D
variable (D : ℕ)

-- Define the times for each leg of the trip
def time_with_current := D / effective_speed_with_current
def time_against_current := D / effective_speed_against_current

-- The final theorem stating the round trip distance
theorem distance_to_place_is_24 :
  time_with_current + time_against_current = round_trip_time → D = 24 :=
by sorry

end distance_to_place_is_24_l100_100773


namespace negation_of_exists_gt_one_l100_100734

theorem negation_of_exists_gt_one : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by 
  sorry

end negation_of_exists_gt_one_l100_100734


namespace engineering_student_max_marks_l100_100315

/-- 
If an engineering student has to secure 36% marks to pass, and he gets 130 marks but fails by 14 marks, 
then the maximum number of marks is 400.
-/
theorem engineering_student_max_marks (M : ℝ) (passing_percentage : ℝ) (marks_obtained : ℝ) (marks_failed_by : ℝ) (pass_marks : ℝ) :
  passing_percentage = 0.36 →
  marks_obtained = 130 →
  marks_failed_by = 14 →
  pass_marks = marks_obtained + marks_failed_by →
  pass_marks = passing_percentage * M →
  M = 400 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end engineering_student_max_marks_l100_100315


namespace combined_loading_time_l100_100589

theorem combined_loading_time (rA rB rC : ℝ) (hA : rA = 1 / 6) (hB : rB = 1 / 8) (hC : rC = 1 / 10) :
  1 / (rA + rB + rC) = 120 / 47 := by
  sorry

end combined_loading_time_l100_100589


namespace XiaoYing_minimum_water_usage_l100_100727

-- Definitions based on the problem's conditions
def first_charge_rate : ℝ := 2.8
def excess_charge_rate : ℝ := 3
def initial_threshold : ℝ := 5
def minimum_bill : ℝ := 29

-- Main statement for the proof based on the derived inequality
theorem XiaoYing_minimum_water_usage (x : ℝ) (h1 : 2.8 * initial_threshold + 3 * (x - initial_threshold) ≥ 29) : x ≥ 10 := by
  sorry

end XiaoYing_minimum_water_usage_l100_100727


namespace diesel_fuel_cost_l100_100757

def cost_per_liter (total_cost : ℝ) (num_liters : ℝ) : ℝ := total_cost / num_liters

def full_tank_cost (cost_per_l : ℝ) (tank_capacity : ℝ) : ℝ := cost_per_l * tank_capacity

theorem diesel_fuel_cost (total_cost : ℝ) (num_liters : ℝ) (tank_capacity : ℝ) :
  total_cost = 18 → num_liters = 36 → tank_capacity = 64 → full_tank_cost (cost_per_liter total_cost num_liters) tank_capacity = 32 :=
by
  intros h_total h_num h_tank
  rw [h_total, h_num, h_tank]
  norm_num
  sorry -- Full proof can be completed with detailed steps.

end diesel_fuel_cost_l100_100757


namespace angle_in_third_quadrant_l100_100386

theorem angle_in_third_quadrant (α : ℝ) (k : ℤ) :
  (2 * ↑k * Real.pi + Real.pi < α ∧ α < 2 * ↑k * Real.pi + 3 * Real.pi / 2) →
  (∃ (m : ℤ), (0 < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < Real.pi ∨
                π < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < 3 * Real.pi / 2 ∨ 
                -π < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < 0)) :=
by
  sorry

end angle_in_third_quadrant_l100_100386


namespace find_b_plus_m_l100_100565

def matrix_C (b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 3, b],
    ![0, 1, 5],
    ![0, 0, 1]
  ]

def matrix_RHS : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 27, 3003],
    ![0, 1, 45],
    ![0, 0, 1]
  ]

theorem find_b_plus_m (b m : ℕ) (h : matrix_C b ^ m = matrix_RHS) : b + m = 306 := 
  sorry

end find_b_plus_m_l100_100565


namespace probability_neither_defective_l100_100755

noncomputable def n := 9
noncomputable def k := 2
noncomputable def total_pens := 9
noncomputable def defective_pens := 3
noncomputable def non_defective_pens := total_pens - defective_pens

noncomputable def total_combinations := Nat.choose total_pens k
noncomputable def non_defective_combinations := Nat.choose non_defective_pens k

theorem probability_neither_defective :
  (non_defective_combinations : ℚ) / total_combinations = 5 / 12 := by
sorry

end probability_neither_defective_l100_100755


namespace max_min_value_of_fg_l100_100210

noncomputable def f (x : ℝ) : ℝ := 4 - x^2
noncomputable def g (x : ℝ) : ℝ := 3 * x
noncomputable def min' (a b : ℝ) : ℝ := if a < b then a else b

theorem max_min_value_of_fg : ∃ x : ℝ, min' (f x) (g x) = 3 :=
by
  sorry

end max_min_value_of_fg_l100_100210


namespace tank_capacity_l100_100164

theorem tank_capacity (c w : ℝ) 
  (h1 : w / c = 1 / 7) 
  (h2 : (w + 5) / c = 1 / 5) : 
  c = 87.5 := 
by
  sorry

end tank_capacity_l100_100164


namespace polynomial_evaluation_l100_100041

theorem polynomial_evaluation :
  ∀ x : ℤ, x = -2 → (x^3 + x^2 + x + 1 = -5) :=
by
  intros x hx
  rw [hx]
  norm_num

end polynomial_evaluation_l100_100041


namespace sin_tan_correct_value_l100_100241

noncomputable def sin_tan_value (x y : ℝ) (h : x^2 + y^2 = 1) : ℝ :=
  let sin_alpha := y
  let tan_alpha := y / x
  sin_alpha * tan_alpha

theorem sin_tan_correct_value :
  sin_tan_value (3/5) (-4/5) (by norm_num) = 16/15 := 
by
  sorry

end sin_tan_correct_value_l100_100241


namespace divide_24kg_into_parts_l100_100381

theorem divide_24kg_into_parts (W : ℕ) (part1 part2 : ℕ) (h_sum : part1 + part2 = 24) :
  (part1 = 9 ∧ part2 = 15) ∨ (part1 = 15 ∧ part2 = 9) :=
by
  sorry

end divide_24kg_into_parts_l100_100381


namespace cindy_marbles_problem_l100_100847

theorem cindy_marbles_problem
  (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 500) (h2 : friends = 4) (h3 : marbles_per_friend = 80) :
  4 * (initial_marbles - (marbles_per_friend * friends)) = 720 :=
by
  sorry

end cindy_marbles_problem_l100_100847


namespace prime_pairs_sum_to_50_l100_100679

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of prime numbers less than 25
def primes_less_than_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Define the target value
def target_sum : ℕ := 50

-- Define a function to check if a pair of numbers are primes that sum to 50
def primes_sum_to_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = target_sum

-- Define a function to get all unordered pairs of primes that sum to 50
def count_primes_sum_to_50 : ℕ :=
  (primes_less_than_25.filter (λ p, (50 - p) ∈ primes_less_than_25)).length / 2

-- The main theorem to prove
theorem prime_pairs_sum_to_50 : count_primes_sum_to_50 = 4 :=
by
  sorry

end prime_pairs_sum_to_50_l100_100679


namespace another_representation_l100_100350

def positive_int_set : Set ℕ := {x | x > 0}

theorem another_representation :
  {x ∈ positive_int_set | x - 3 < 2} = {1, 2, 3, 4} :=
by
  sorry

end another_representation_l100_100350


namespace complex_fraction_expression_equals_half_l100_100482

theorem complex_fraction_expression_equals_half :
  ((2 / (3 + 1/5)) + (((3 + 1/4) / 13) / (2 / 3)) + (((2 + 5/18) - (17/36)) * (18 / 65))) * (1 / 3) = 0.5 :=
by
  sorry

end complex_fraction_expression_equals_half_l100_100482


namespace cos_triple_angle_l100_100696

variable (θ : ℝ)

theorem cos_triple_angle (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l100_100696


namespace sales_tax_paid_l100_100515

variable (total_cost : ℝ)
variable (tax_rate : ℝ)
variable (tax_free_cost : ℝ)

theorem sales_tax_paid (h_total : total_cost = 25) (h_rate : tax_rate = 0.10) (h_free : tax_free_cost = 21.7) :
  ∃ (X : ℝ), 21.7 + X + (0.10 * X) = 25 ∧ (0.10 * X = 0.3) := 
by
  sorry

end sales_tax_paid_l100_100515


namespace line_equation_l100_100994

theorem line_equation (x y : ℝ) (h : ∀ x : ℝ, (x - 2) * 1 = y) : x - y - 2 = 0 :=
sorry

end line_equation_l100_100994


namespace abs_eq_condition_l100_100215

theorem abs_eq_condition (x : ℝ) (h : |x - 1| + |x - 5| = 4) : 1 ≤ x ∧ x ≤ 5 :=
by 
  sorry

end abs_eq_condition_l100_100215


namespace cos_triple_angle_l100_100687

theorem cos_triple_angle :
  (cos θ = 1 / 3) → cos (3 * θ) = -23 / 27 :=
by
  intro h
  have h1 : cos θ = 1 / 3 := h
  sorry

end cos_triple_angle_l100_100687


namespace arithmetic_sequence_sum_l100_100585

noncomputable def Sn (a d n : ℕ) : ℕ :=
n * a + (n * (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a d : ℕ) (h1 : a = 3 * d) (h2 : Sn a d 5 = 50) : Sn a d 8 = 104 :=
by
/-
  From the given conditions:
  - \(a_4\) is the geometric mean of \(a_2\) and \(a_7\) implies \(a = 3d\)
  - Sum of first 5 terms is 50 implies \(S_5 = 50\)
  We need to prove \(S_8 = 104\)
-/
  sorry

end arithmetic_sequence_sum_l100_100585


namespace point_divides_segment_l100_100797

theorem point_divides_segment (x₁ y₁ x₂ y₂ m n : ℝ) (h₁ : (x₁, y₁) = (3, 7)) (h₂ : (x₂, y₂) = (5, 1)) (h₃ : m = 1) (h₄ : n = 3) :
  ( (m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n) ) = (3.5, 5.5) :=
by
  sorry

end point_divides_segment_l100_100797


namespace find_M_l100_100146

theorem find_M (x y z M : ℝ) 
  (h1 : x + y + z = 120) 
  (h2 : x - 10 = M) 
  (h3 : y + 10 = M) 
  (h4 : z / 10 = M) : 
  M = 10 := 
by
  sorry

end find_M_l100_100146


namespace abs_neg_three_l100_100271

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l100_100271


namespace max_drinks_amount_l100_100199

noncomputable def initial_milk : ℚ := 3 / 4
noncomputable def rachel_fraction : ℚ := 1 / 2
noncomputable def max_fraction : ℚ := 1 / 3

def amount_rachel_drinks (initial: ℚ) (fraction: ℚ) : ℚ := initial * fraction
def remaining_milk_after_rachel (initial: ℚ) (amount_rachel: ℚ) : ℚ := initial - amount_rachel
def amount_max_drinks (remaining: ℚ) (fraction: ℚ) : ℚ := remaining * fraction

theorem max_drinks_amount :
  amount_max_drinks (remaining_milk_after_rachel initial_milk (amount_rachel_drinks initial_milk rachel_fraction)) max_fraction = 1 / 8 := 
sorry

end max_drinks_amount_l100_100199


namespace minimum_value_of_expression_l100_100231

theorem minimum_value_of_expression (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : a * b > 0) : 
  (1 / a) + (2 / b) = 5 :=
sorry

end minimum_value_of_expression_l100_100231


namespace required_bike_speed_l100_100922

theorem required_bike_speed (swim_distance run_distance bike_distance swim_speed run_speed total_time : ℝ)
  (h_swim_dist : swim_distance = 0.5)
  (h_run_dist : run_distance = 4)
  (h_bike_dist : bike_distance = 12)
  (h_swim_speed : swim_speed = 1)
  (h_run_speed : run_speed = 8)
  (h_total_time : total_time = 1.5) :
  (bike_distance / ((total_time - (swim_distance / swim_speed + run_distance / run_speed)))) = 24 :=
by
  sorry

end required_bike_speed_l100_100922


namespace steven_apples_peaches_difference_l100_100966

def steven_apples := 19
def jake_apples (steven_apples : ℕ) := steven_apples + 4
def jake_peaches (steven_peaches : ℕ) := steven_peaches - 3

theorem steven_apples_peaches_difference (P : ℕ) :
  19 - P = steven_apples - P :=
by
  sorry

end steven_apples_peaches_difference_l100_100966


namespace product_of_three_consecutive_integers_is_square_l100_100787

theorem product_of_three_consecutive_integers_is_square (x : ℤ) : 
  ∃ n : ℤ, x * (x + 1) * (x + 2) = n^2 → x = 0 ∨ x = -1 ∨ x = -2 :=
by
  sorry

end product_of_three_consecutive_integers_is_square_l100_100787


namespace proof_l100_100378

open Set

variable (U M P : Set ℕ)

noncomputable def prob_statement : Prop :=
  let C_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}
  U = {1,2,3,4,5,6,7,8} ∧ M = {2,3,4} ∧ P = {1,3,6} ∧ C_U (M ∪ P) = {5,7,8}

theorem proof : prob_statement {1,2,3,4,5,6,7,8} {2,3,4} {1,3,6} :=
by
  sorry

end proof_l100_100378


namespace oxen_b_is_12_l100_100605

variable (oxen_b : ℕ)

def share (oxen months : ℕ) : ℕ := oxen * months

def total_share (oxen_a oxen_b oxen_c months_a months_b months_c : ℕ) : ℕ :=
  share oxen_a months_a + share oxen_b months_b + share oxen_c months_c

def proportion (rent_c rent total_share_c total_share : ℕ) : Prop :=
  rent_c * total_share = rent * total_share_c

theorem oxen_b_is_12 : oxen_b = 12 := by
  let oxen_a := 10
  let oxen_c := 15
  let months_a := 7
  let months_b := 5
  let months_c := 3
  let rent := 210
  let rent_c := 54
  let share_a := share oxen_a months_a
  let share_c := share oxen_c months_c
  let total_share_val := total_share oxen_a oxen_b oxen_c months_a months_b months_c
  let total_rent := share_a + 5 * oxen_b + share_c
  have h1 : proportion rent_c rent share_c total_rent := by sorry
  rw [proportion] at h1
  sorry

end oxen_b_is_12_l100_100605


namespace y_coord_of_third_vertex_of_equilateral_l100_100623

/-- Given two vertices of an equilateral triangle at (0, 6) and (10, 6), and the third vertex in the first quadrant,
    prove that the y-coordinate of the third vertex is 6 + 5 * sqrt 3. -/
theorem y_coord_of_third_vertex_of_equilateral (A B C : ℝ × ℝ)
  (hA : A = (0, 6)) (hB : B = (10, 6)) (hAB : dist A B = 10) (hC : C.2 > 6):
  C.2 = 6 + 5 * Real.sqrt 3 :=
sorry

end y_coord_of_third_vertex_of_equilateral_l100_100623


namespace tennis_tournament_cycle_l100_100339

noncomputable def exists_cycle_of_three_players (P : Type) [Fintype P] (G : P → P → Bool) : Prop :=
  (∃ (a b c : P), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ G a b ∧ G b c ∧ G c a)

theorem tennis_tournament_cycle (P : Type) [Fintype P] (n : ℕ) (hp : 3 ≤ n) 
  (G : P → P → Bool) (H : ∀ a b : P, a ≠ b → (G a b ∨ G b a))
  (Hw : ∀ a : P, ∃ b : P, a ≠ b ∧ G a b) : exists_cycle_of_three_players P G :=
by 
  sorry

end tennis_tournament_cycle_l100_100339


namespace sum_consecutive_integers_product_1080_l100_100582

theorem sum_consecutive_integers_product_1080 :
  ∃ n : ℕ, n * (n + 1) = 1080 ∧ n + (n + 1) = 65 :=
by
  sorry

end sum_consecutive_integers_product_1080_l100_100582


namespace sector_central_angle_l100_100658

theorem sector_central_angle (r l : ℝ) (α : ℝ) 
  (h1 : l + 2 * r = 12) 
  (h2 : 1 / 2 * l * r = 8) : 
  α = 1 ∨ α = 4 :=
by
  sorry

end sector_central_angle_l100_100658


namespace negation_of_proposition_l100_100224

theorem negation_of_proposition (a b c : ℝ) :
  ¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) :=
sorry

end negation_of_proposition_l100_100224


namespace calculate_expression_l100_100511

theorem calculate_expression :
  ((16^10 / 16^8) ^ 3 * 8 ^ 3) / 2 ^ 9 = 16777216 := by
  sorry

end calculate_expression_l100_100511


namespace find_f_neg_2_l100_100533

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

variable (a b : ℝ)

theorem find_f_neg_2 (h1 : f a b 2 = 6) : f a b (-2) = -14 :=
by
  sorry

end find_f_neg_2_l100_100533


namespace elapsed_time_l100_100190

theorem elapsed_time (x : ℕ) (h1 : 99 > 0) (h2 : (2 : ℚ) / (3 : ℚ) * x = (4 : ℚ) / (5 : ℚ) * (99 - x)) : x = 54 := by
  sorry

end elapsed_time_l100_100190


namespace cos_pi_over_3_plus_2alpha_l100_100657

variable (α : ℝ)

theorem cos_pi_over_3_plus_2alpha (h : Real.sin (π / 3 - α) = 1 / 3) :
  Real.cos (π / 3 + 2 * α) = 7 / 9 :=
  sorry

end cos_pi_over_3_plus_2alpha_l100_100657


namespace no_integer_root_l100_100120

theorem no_integer_root (q : ℤ) : ¬ ∃ x : ℤ, x^2 + 7 * x - 14 * (q^2 + 1) = 0 := sorry

end no_integer_root_l100_100120


namespace megan_final_balance_same_as_starting_balance_l100_100113

theorem megan_final_balance_same_as_starting_balance :
  let starting_balance : ℝ := 125
  let increased_balance := starting_balance * (1 + 0.25)
  let final_balance := increased_balance * (1 - 0.20)
  final_balance = starting_balance :=
by
  sorry

end megan_final_balance_same_as_starting_balance_l100_100113


namespace biff_break_even_hours_l100_100630

-- Definitions based on conditions
def ticket_expense : ℕ := 11
def snacks_expense : ℕ := 3
def headphones_expense : ℕ := 16
def total_expenses : ℕ := ticket_expense + snacks_expense + headphones_expense
def gross_income_per_hour : ℕ := 12
def wifi_cost_per_hour : ℕ := 2
def net_income_per_hour : ℕ := gross_income_per_hour - wifi_cost_per_hour

-- The proof statement
theorem biff_break_even_hours : ∃ h : ℕ, h * net_income_per_hour = total_expenses ∧ h = 3 :=
by 
  have h_value : ℕ := 3
  exists h_value
  split
  · show h_value * net_income_per_hour = total_expenses
    sorry
  · show h_value = 3
    rfl

end biff_break_even_hours_l100_100630


namespace sum_of_first_twelve_terms_arithmetic_sequence_l100_100929

-- Definitions
def a (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

def Sn (a1 d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

-- Main Statement
theorem sum_of_first_twelve_terms_arithmetic_sequence  (a1 d : ℝ) 
  (h1 : a a1 d 5 = 1) (h2 : a a1 d 17 = 18) :
  Sn a1 d 12 = 37.5 :=
sorry

end sum_of_first_twelve_terms_arithmetic_sequence_l100_100929


namespace fibonacci_periodicity_l100_100157

-- Definitions for p-arithmetic and Fibonacci sequence
def is_prime (p : ℕ) := Nat.Prime p
def sqrt_5_extractable (p : ℕ) : Prop := ∃ k : ℕ, p = 5 * k + 1 ∨ p = 5 * k - 1

-- Definitions of Fibonacci sequences and properties
def fibonacci : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fibonacci n + fibonacci (n + 1)

-- Main theorem
theorem fibonacci_periodicity (p : ℕ) (r : ℕ) (h_prime : is_prime p) (h_not_2_or_5 : p ≠ 2 ∧ p ≠ 5)
    (h_period : r = (p+1) ∨ r = (p-1)) (h_div : (sqrt_5_extractable p → r ∣ (p - 1)) ∧ (¬ sqrt_5_extractable p → r ∣ (p + 1)))
    : (fibonacci (p+1) % p = 0 ∨ fibonacci (p-1) % p = 0) := by
          sorry

end fibonacci_periodicity_l100_100157


namespace add_multiply_round_l100_100328

theorem add_multiply_round :
  let a := 73.5891
  let b := 24.376
  let c := (a + b) * 2
  (Float.round (c * 100) / 100) = 195.93 :=
by
  sorry

end add_multiply_round_l100_100328


namespace turnip_bag_weight_l100_100006

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l100_100006


namespace function_range_y_eq_1_div_x_minus_2_l100_100556

theorem function_range_y_eq_1_div_x_minus_2 (x : ℝ) : (∀ y : ℝ, y = 1 / (x - 2) ↔ x ∈ {x : ℝ | x ≠ 2}) :=
sorry

end function_range_y_eq_1_div_x_minus_2_l100_100556


namespace first_step_of_testing_circuit_broken_l100_100890

-- Definitions based on the problem
def circuit_broken : Prop := true
def binary_search_method : Prop := true
def test_first_step_at_midpoint : Prop := true

-- The theorem stating the first step in testing a broken circuit using the binary search method
theorem first_step_of_testing_circuit_broken (h1 : circuit_broken) (h2 : binary_search_method) :
  test_first_step_at_midpoint :=
sorry

end first_step_of_testing_circuit_broken_l100_100890


namespace count_four_digit_integers_with_1_or_7_l100_100382

/-- 
The total number of four-digit integers with at least one digit being 1 or 7 is 5416.
-/
theorem count_four_digit_integers_with_1_or_7 : 
  let all_four_digit_integers := 9000
  let without_1_or_7 := 7 * 8 * 8 * 8
  let with_1_or_7 := all_four_digit_integers - without_1_or_7
  with_1_or_7 = 5416
:= by
  let all_four_digit_integers := 9000
  let without_1_or_7 := 7 * 8 * 8 * 8
  let with_1_or_7 := all_four_digit_integers - without_1_or_7
  show with_1_or_7 = 5416
  sorry

end count_four_digit_integers_with_1_or_7_l100_100382


namespace length_YW_l100_100838

-- Definitions of the sides of the triangle
def XY := 6
def YZ := 8
def XZ := 10

-- The total perimeter of triangle XYZ
def perimeter : ℕ := XY + YZ + XZ

-- Each ant travels half the perimeter
def halfPerimeter : ℕ := perimeter / 2

-- Distance one ant travels from X to W through Y
def distanceXtoW : ℕ := XY + 6

-- Prove that the distance segment YW is 6
theorem length_YW : distanceXtoW = halfPerimeter := by sorry

end length_YW_l100_100838


namespace total_candies_is_90_l100_100979

-- Defining the conditions
def boxes_chocolate := 6
def boxes_caramel := 4
def pieces_per_box := 9

-- Defining the total number of boxes
def total_boxes := boxes_chocolate + boxes_caramel

-- Defining the total number of candies
def total_candies := total_boxes * pieces_per_box

-- Theorem stating the proof problem
theorem total_candies_is_90 : total_candies = 90 := by
  -- Provide a placeholder for the proof
  sorry

end total_candies_is_90_l100_100979


namespace correct_option_is_B_l100_100454

theorem correct_option_is_B (a : ℝ) : 
  (¬ (-2 * a^2 * b)^3 = -6 * a^6 * b^3) ∧
  (a^7 / a = a^6) ∧
  (¬ (a + 1)^2 = a^2 + 1) ∧
  (¬ 2 * a + 3 * b = 5 * a * b) :=
by
  sorry

end correct_option_is_B_l100_100454


namespace athlete_total_heartbeats_l100_100179

theorem athlete_total_heartbeats (h : ℕ) (p : ℕ) (d : ℕ) (r : ℕ) : (h = 150) ∧ (p = 6) ∧ (d = 30) ∧ (r = 15) → (p * d + r) * h = 29250 :=
by
  sorry

end athlete_total_heartbeats_l100_100179


namespace empty_solution_set_implies_a_range_l100_100536

def f (a x: ℝ) := x^2 + (1 - a) * x - a

theorem empty_solution_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬ (f a (f a x) < 0)) → -3 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 3 :=
by
  sorry

end empty_solution_set_implies_a_range_l100_100536


namespace simplify_div_expression_evaluate_at_2_l100_100267

variable (a : ℝ)

theorem simplify_div_expression (h0 : a ≠ 0) (h1 : a ≠ 1) :
  (1 - 1 / a) / ((a^2 - 2 * a + 1) / a) = 1 / (a - 1) :=
by
  sorry

theorem evaluate_at_2 : (1 - 1 / 2) / ((2^2 - 2 * 2 + 1) / 2) = 1 :=
by 
  sorry

end simplify_div_expression_evaluate_at_2_l100_100267


namespace num_prime_pairs_sum_50_l100_100685

def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def is_prime (n : ℕ) : Prop := ∀ k ∈ (List.range n).drop 1, k ∣ n → k = n

theorem num_prime_pairs_sum_50 :
  let pairs := (prime_numbers.toFinset.product prime_numbers.toFinset).filter (λ p, p.1 < p.2 ∧ p.1 + p.2 = 50)
  pairs.card = 4 := by
  sorry

end num_prime_pairs_sum_50_l100_100685


namespace repeating_decimal_eq_l100_100468

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l100_100468


namespace abs_neg_three_l100_100276

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l100_100276


namespace remainder_when_sum_div_by_3_l100_100827

theorem remainder_when_sum_div_by_3 
  (m n p q : ℕ)
  (a : ℕ := 6 * m + 4)
  (b : ℕ := 6 * n + 4)
  (c : ℕ := 6 * p + 4)
  (d : ℕ := 6 * q + 4)
  : (a + b + c + d) % 3 = 1 :=
by
  sorry

end remainder_when_sum_div_by_3_l100_100827


namespace eight_people_lineup_ways_l100_100436

theorem eight_people_lineup_ways : (Nat.factorial 8 = 40320) :=
by
  sorry

end eight_people_lineup_ways_l100_100436


namespace xy_power_l100_100366

def x : ℚ := 3/4
def y : ℚ := 4/3

theorem xy_power : x^7 * y^8 = 4/3 := by
  sorry

end xy_power_l100_100366


namespace arithmetic_sequence_geometric_condition_l100_100608

noncomputable def S (n : ℕ) : ℕ := n * (2 + n - 1) / 2

noncomputable def a (n : ℕ) : ℕ := 1 + (n - 1) * 1

theorem arithmetic_sequence_geometric_condition :
  S 5 = 15 → 
  (a 3) * (a 12) = (a 6) * (a 6) →
  (S 2023) / (a 2023) = 1012 := 
by
  sorry

end arithmetic_sequence_geometric_condition_l100_100608


namespace solve_2x2_minus1_eq_3x_l100_100577
noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let root1 := (-b + sqrt_discriminant) / (2 * a)
  let root2 := (-b - sqrt_discriminant) / (2 * a)
  (root1, root2)

theorem solve_2x2_minus1_eq_3x :
  solve_quadratic 2 (-3) (-1) = ( (3 + Real.sqrt 17) / 4, (3 - Real.sqrt 17) / 4 ) :=
by
  let roots := solve_quadratic 2 (-3) (-1)
  have : roots = ( (3 + Real.sqrt 17) / 4, (3 - Real.sqrt 17) / 4) := by sorry
  exact this

end solve_2x2_minus1_eq_3x_l100_100577


namespace DollOutfit_l100_100147

variables (VeraDress OlyaCoat VeraCoat NinaCoat : Prop)
axiom FirstAnswer : (VeraDress ∧ ¬OlyaCoat) ∨ (¬VeraDress ∧ OlyaCoat)
axiom SecondAnswer : (VeraCoat ∧ ¬NinaCoat) ∨ (¬VeraCoat ∧ NinaCoat)
axiom OnlyOneTrueFirstAnswer : (VeraDress ∨ OlyaCoat) ∧ ¬(VeraDress ∧ OlyaCoat)
axiom OnlyOneTrueSecondAnswer : (VeraCoat ∨ NinaCoat) ∧ ¬(VeraCoat ∧ NinaCoat)

theorem DollOutfit :
  VeraDress ∧ NinaCoat ∧ ¬OlyaCoat ∧ ¬VeraCoat ∧ ¬NinaCoat :=
sorry

end DollOutfit_l100_100147


namespace tile_5x7_rectangle_with_L_trominos_l100_100093

theorem tile_5x7_rectangle_with_L_trominos :
  ∀ k : ℕ, ¬ (∃ (tile : ℕ → ℕ → ℕ), (∀ i j, tile (i+1) (j+1) = tile (i+3) (j+3)) ∧
    ∀ i j, (i < 5 ∧ j < 7) → (tile i j = k)) :=
by sorry

end tile_5x7_rectangle_with_L_trominos_l100_100093


namespace smallest_prime_with_digit_sum_25_l100_100889

-- Definitions used in Lean statement:
-- 1. Prime predicate based on primality check.
-- 2. Digit sum function.

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Lean 4 statement to prove that the smallest prime whose digits sum to 25 is 1699.

theorem smallest_prime_with_digit_sum_25 : ∃ n : ℕ, is_prime n ∧ digit_sum n = 25 ∧ n = 1699 :=
by
  sorry

end smallest_prime_with_digit_sum_25_l100_100889


namespace at_least_one_not_greater_than_minus_four_l100_100713

theorem at_least_one_not_greater_than_minus_four {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4 :=
sorry

end at_least_one_not_greater_than_minus_four_l100_100713


namespace line_quadrant_conditions_l100_100812

theorem line_quadrant_conditions (k b : ℝ) 
  (H1 : ∃ x : ℝ, x > 0 ∧ k * x + b > 0)
  (H3 : ∃ x : ℝ, x < 0 ∧ k * x + b < 0)
  (H4 : ∃ x : ℝ, x > 0 ∧ k * x + b < 0) : k > 0 ∧ b < 0 :=
sorry

end line_quadrant_conditions_l100_100812


namespace certain_number_divisible_by_9_l100_100666

theorem certain_number_divisible_by_9 : ∃ N : ℕ, (∀ k : ℕ, (0 ≤ k ∧ k < 1110 → N + 9 * k ≤ 10000 ∧ (N + 9 * k) % 9 = 0)) ∧ N = 27 :=
by
  -- Given conditions:
  -- Numbers are in an arithmetic sequence with common difference 9.
  -- Total count of such numbers is 1110.
  -- The last number ≤ 10000 that is divisible by 9 is 9999.
  let L := 9999
  let n := 1110
  let d := 9
  -- First term in the sequence:
  let a := L - (n - 1) * d
  exists 27
  -- Proof of the conditions would follow here ...
  sorry

end certain_number_divisible_by_9_l100_100666


namespace average_weight_increase_l100_100242

theorem average_weight_increase 
  (n : ℕ) (A : ℕ → ℝ)
  (h_total : n = 10)
  (h_replace : A 65 = 137) : 
  (137 - 65) / 10 = 7.2 := 
by 
  sorry

end average_weight_increase_l100_100242


namespace abs_opposite_of_three_eq_5_l100_100237

theorem abs_opposite_of_three_eq_5 : ∀ (a : ℤ), a = -3 → |a - 2| = 5 := by
  sorry

end abs_opposite_of_three_eq_5_l100_100237


namespace line_through_point_parallel_to_given_line_line_through_point_sum_intercepts_is_minus_four_l100_100375

theorem line_through_point_parallel_to_given_line :
  ∃ c : ℤ, (∀ x y : ℤ, 2 * x + 3 * y + c = 0 ↔ (x, y) = (2, 1)) ∧ c = -7 :=
sorry

theorem line_through_point_sum_intercepts_is_minus_four :
  ∃ (a b : ℤ), (∀ x y : ℤ, (x / a) + (y / b) = 1 ↔ (x, y) = (-3, 1)) ∧ (a + b = -4) ∧ 
  ((a = -6 ∧ b = 2) ∨ (a = -2 ∧ b = -2)) ∧ 
  ((∀ x y : ℤ, x - 3 * y + 6 = 0 ↔ (x, y) = (-3, 1)) ∨ 
  (∀ x y : ℤ, x + y + 2 = 0 ↔ (x, y) = (-3, 1))) :=
sorry

end line_through_point_parallel_to_given_line_line_through_point_sum_intercepts_is_minus_four_l100_100375


namespace semicircle_radius_in_trapezoid_l100_100394

theorem semicircle_radius_in_trapezoid 
  (AB CD : ℝ) (AD BC : ℝ) (r : ℝ)
  (h1 : AB = 27) 
  (h2 : CD = 45) 
  (h3 : AD = 13) 
  (h4 : BC = 15) 
  (h5 : r = 13.5) :
  r = 13.5 :=
by
  sorry  -- Detailed proof steps will go here

end semicircle_radius_in_trapezoid_l100_100394


namespace henry_apple_weeks_l100_100541

theorem henry_apple_weeks (apples_per_box : ℕ) (boxes : ℕ) (people : ℕ) (apples_per_day : ℕ) (days_per_week : ℕ) :
  apples_per_box = 14 → boxes = 3 → people = 2 → apples_per_day = 1 → days_per_week = 7 →
  (apples_per_box * boxes) / (people * apples_per_day * days_per_week) = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end henry_apple_weeks_l100_100541


namespace rate_of_stream_is_5_l100_100763

-- Define the conditions
def boat_speed : ℝ := 16  -- Boat speed in still water
def time_downstream : ℝ := 3  -- Time taken downstream
def distance_downstream : ℝ := 63  -- Distance covered downstream

-- Define the rate of the stream as an unknown variable
def rate_of_stream (v : ℝ) : Prop := 
  distance_downstream = (boat_speed + v) * time_downstream

-- Statement to prove
theorem rate_of_stream_is_5 : 
  ∃ (v : ℝ), rate_of_stream v ∧ v = 5 :=
by
  use 5
  simp [boat_speed, time_downstream, distance_downstream, rate_of_stream]
  sorry

end rate_of_stream_is_5_l100_100763


namespace solve_system_of_equations_l100_100985

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : 6.751 * x + 3.249 * y = 26.751) 
  (h2 : 3.249 * x + 6.751 * y = 23.249) : 
  x = 3 ∧ y = 2 := 
sorry

end solve_system_of_equations_l100_100985


namespace problem_l100_100542

theorem problem (x : ℝ) (h : x + 2 / x = 4) : - (5 * x) / (x^2 + 2) = -5 / 4 := 
sorry

end problem_l100_100542


namespace lcm_of_9_12_15_l100_100595

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end lcm_of_9_12_15_l100_100595


namespace fixed_point_of_function_l100_100209

-- Definition: The function passes through a fixed point (a, b) for all real numbers k.
def passes_through_fixed_point (f : ℝ → ℝ) (a b : ℝ) := ∀ k : ℝ, f a = b

-- Given the function y = 9x^2 + 3kx - 6k, we aim to prove the fixed point is (2, 36).
theorem fixed_point_of_function : passes_through_fixed_point (fun x => 9 * x^2 + 3 * k * x - 6 * k) 2 36 := by
  sorry

end fixed_point_of_function_l100_100209


namespace turnip_weights_are_13_or_16_l100_100004

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l100_100004


namespace pentagon_area_l100_100821

-- Definitions of the side lengths of the pentagon
def side1 : ℕ := 12
def side2 : ℕ := 17
def side3 : ℕ := 25
def side4 : ℕ := 18
def side5 : ℕ := 17

-- Definitions for the rectangle and triangle dimensions
def rectangle_width : ℕ := side4
def rectangle_height : ℕ := side1
def triangle_base : ℕ := side4
def triangle_height : ℕ := side3 - side1

-- The area of the pentagon proof statement
theorem pentagon_area : rectangle_width * rectangle_height +
    (triangle_base * triangle_height) / 2 = 333 := by
  sorry

end pentagon_area_l100_100821


namespace probability_consecutive_computer_scientists_l100_100449

theorem probability_consecutive_computer_scientists :
  let n := 12
  let k := 5
  let total_permutations := Nat.factorial (n - 1)
  let consecutive_permutations := Nat.factorial (7) * Nat.factorial (5)
  let probability := consecutive_permutations / total_permutations
  probability = (1 / 66) :=
by
  sorry

end probability_consecutive_computer_scientists_l100_100449


namespace locus_equation_perimeter_greater_l100_100088

-- Define the conditions under which the problem is stated
def distance_to_x_axis (P : ℝ × ℝ) : ℝ := abs P.2
def distance_to_point (P : ℝ × ℝ) (Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- P is a point on the locus W if the distance to the x-axis is equal to the distance to (0, 1/2)
def on_locus (P : ℝ × ℝ) : Prop := 
  distance_to_x_axis P = distance_to_point P (0, 1/2)

-- Prove that the equation of W is y = x^2 + 1/4 given the conditions
theorem locus_equation (P : ℝ × ℝ) (h : on_locus P) : 
  P.2 = P.1^2 + 1/4 := 
sorry

-- Assume rectangle ABCD with three points on W
def point_on_w (P : ℝ × ℝ) : Prop := 
  P.2 = P.1^2 + 1/4

def points_form_rectangle (A B C D : ℝ × ℝ) : Prop := 
  A.1 ≠ B.1 ∧ B.1 ≠ C.1 ∧ C.1 ≠ D.1 ∧ D.1 ≠ A.1 ∧
  A.2 ≠ B.2 ∧ B.2 ≠ C.2 ∧ C.2 ≠ D.2 ∧ D.2 ≠ A.2

-- P1, P2, and P3 are three points on the locus W
def points_on_locus (A B C : ℝ × ℝ) : Prop := 
  point_on_w A ∧ point_on_w B ∧ point_on_w C

-- Prove the perimeter of rectangle ABCD with three points on W is greater than 3sqrt(3)
theorem perimeter_greater (A B C D : ℝ × ℝ) 
  (h1 : points_on_locus A B C) 
  (h2 : points_form_rectangle A B C D) : 
  2 * (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
       real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) > 
  3 * real.sqrt 3 := 
sorry

end locus_equation_perimeter_greater_l100_100088


namespace calc_g_x_plus_3_l100_100105

def g (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem calc_g_x_plus_3 (x : ℝ) : g (x + 3) = x^2 + 3*x + 2 :=
by
  sorry

end calc_g_x_plus_3_l100_100105


namespace mac_runs_faster_by_120_minutes_l100_100181

theorem mac_runs_faster_by_120_minutes :
  ∀ (D : ℝ), (D / 3 - D / 4 = 2) → 2 * 60 = 120 := by
  -- Definitions matching the conditions
  intro D
  intro h

  -- The proof is not required, hence using sorry
  sorry

end mac_runs_faster_by_120_minutes_l100_100181


namespace first_discount_percentage_l100_100872

-- Definitions based on the conditions provided
def listed_price : ℝ := 400
def final_price : ℝ := 334.4
def additional_discount : ℝ := 5

-- The equation relating these quantities
theorem first_discount_percentage (D : ℝ) (h : listed_price * (1 - D / 100) * (1 - additional_discount / 100) = final_price) : D = 12 :=
sorry

end first_discount_percentage_l100_100872


namespace infinite_fixpoints_l100_100715

variable {f : ℕ+ → ℕ+}
variable (H : ∀ (m n : ℕ+), (∃ k : ℕ+ , k ≤ f n ∧ n ∣ f (m + k)) ∧ (∀ j : ℕ+ , j ≤ f n → j ≠ k → ¬ n ∣ f (m + j)))

theorem infinite_fixpoints : ∃ᶠ n in at_top, f n = n :=
sorry

end infinite_fixpoints_l100_100715


namespace find_angle_C_find_max_perimeter_l100_100964

-- Define the first part of the problem
theorem find_angle_C 
  (a b c A B C : ℝ) (h1 : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) :
  C = (2 * Real.pi) / 3 :=
sorry

-- Define the second part of the problem
theorem find_max_perimeter 
  (a b A B : ℝ)
  (C : ℝ := (2 * Real.pi) / 3)
  (c : ℝ := Real.sqrt 3)
  (h1 : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) :
  (2 * Real.sqrt 3 < a + b + c) ∧ (a + b + c <= 2 + Real.sqrt 3) :=
sorry

end find_angle_C_find_max_perimeter_l100_100964


namespace prob_ge_neg2_l100_100060

noncomputable def normal_dist (mean variance : ℝ) : Type := sorry -- This is a placeholder for the normal distribution type.

variable (ξ : normal_dist 0 4)

theorem prob_ge_neg2 (h : P(ξ ≥ 2) = 0.3) : P(ξ ≥ -2) = 0.7 :=
begin
  sorry
end

end prob_ge_neg2_l100_100060


namespace kim_hard_correct_l100_100701

-- Definitions
def points_per_easy := 2
def points_per_average := 3
def points_per_hard := 5
def easy_correct := 6
def average_correct := 2
def total_points := 38

-- Kim's correct answers in the hard round is 4
theorem kim_hard_correct : (total_points - (easy_correct * points_per_easy + average_correct * points_per_average)) / points_per_hard = 4 :=
by
  sorry

end kim_hard_correct_l100_100701


namespace total_annual_salary_excluding_turban_l100_100946

-- Let X be the total amount of money Gopi gives as salary for one year, excluding the turban.
variable (X : ℝ)

-- Condition: The servant leaves after 9 months and receives Rs. 60 plus the turban.
variable (received_money : ℝ)
variable (turban_price : ℝ)

-- Condition values:
axiom received_money_condition : received_money = 60
axiom turban_price_condition : turban_price = 30

-- Question: Prove that X equals 90.
theorem total_annual_salary_excluding_turban :
  3/4 * (X + turban_price) = 90 :=
sorry

end total_annual_salary_excluding_turban_l100_100946


namespace sequences_properties_l100_100653

-- Definitions based on the problem conditions
def geom_sequence (a : ℕ → ℕ) := ∃ q : ℕ, a 1 = 2 ∧ a 3 = 18 ∧ ∀ n, a (n + 1) = a n * q
def arith_sequence (b : ℕ → ℕ) := b 1 = 2 ∧ ∃ d : ℕ, ∀ n, b (n + 1) = b n + d
def condition (a : ℕ → ℕ) (b : ℕ → ℕ) := a 1 + a 2 + a 3 > 20 ∧ a 1 + a 2 + a 3 = b 1 + b 2 + b 3 + b 4

-- Proof statement: proving the general term of the geometric sequence and the sum of the arithmetic sequence
theorem sequences_properties (a : ℕ → ℕ) (b : ℕ → ℕ) :
  geom_sequence a → arith_sequence b → condition a b →
  (∀ n, a n = 2 * 3^(n - 1)) ∧ (∀ n, S_n = 3 / 2 * n^2 + 1 / 2 * n) :=
by
  sorry

end sequences_properties_l100_100653


namespace sum_of_ages_is_nineteen_l100_100625

-- Definitions representing the conditions
def Bella_age : ℕ := 5
def Brother_is_older : ℕ := 9
def Brother_age : ℕ := Bella_age + Brother_is_older
def Sum_of_ages : ℕ := Bella_age + Brother_age

-- Mathematical statement (theorem) to be proved
theorem sum_of_ages_is_nineteen : Sum_of_ages = 19 := by
  sorry

end sum_of_ages_is_nineteen_l100_100625


namespace find_x_parallel_vectors_l100_100540

theorem find_x_parallel_vectors :
  ∀ x : ℝ, (∃ k : ℝ, (1, 2) = (k * (2 * x), k * (-3))) → x = -3 / 4 :=
by
  sorry

end find_x_parallel_vectors_l100_100540


namespace recurring_to_fraction_l100_100477

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l100_100477


namespace four_times_remaining_marbles_l100_100851

theorem four_times_remaining_marbles (initial total_given : ℕ) (remaining : ℕ := initial - total_given) :
  initial = 500 → total_given = 4 * 80 → 4 * remaining = 720 := by sorry

end four_times_remaining_marbles_l100_100851


namespace trig_identity_l100_100781

theorem trig_identity :
  (Real.cos (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (48 * Real.pi / 180) * Real.sin (18 * Real.pi / 180)) = 1 / 2 := 
by sorry

end trig_identity_l100_100781


namespace average_cost_across_all_products_sold_is_670_l100_100337

-- Definitions based on conditions
def iphones_sold : ℕ := 100
def ipad_sold : ℕ := 20
def appletv_sold : ℕ := 80

def cost_iphone : ℕ := 1000
def cost_ipad : ℕ := 900
def cost_appletv : ℕ := 200

-- Calculations based on conditions
def revenue_iphone : ℕ := iphones_sold * cost_iphone
def revenue_ipad : ℕ := ipad_sold * cost_ipad
def revenue_appletv : ℕ := appletv_sold * cost_appletv

def total_revenue : ℕ := revenue_iphone + revenue_ipad + revenue_appletv
def total_products_sold : ℕ := iphones_sold + ipad_sold + appletv_sold

def average_cost := total_revenue / total_products_sold

-- Theorem to be proved
theorem average_cost_across_all_products_sold_is_670 :
  average_cost = 670 :=
by
  sorry

end average_cost_across_all_products_sold_is_670_l100_100337


namespace area_triangle_BRS_l100_100725

def point := ℝ × ℝ
def x_intercept (p : point) : ℝ := p.1
def y_intercept (p : point) : ℝ := p.2

noncomputable def distance_from_y_axis (p : point) : ℝ := abs p.1

theorem area_triangle_BRS (B R S : point)
  (hB : B = (4, 10))
  (h_perp : ∃ m₁ m₂, m₁ * m₂ = -1)
  (h_sum_zero : x_intercept R + x_intercept S = 0)
  (h_dist : distance_from_y_axis B = 10) :
  ∃ area : ℝ, area = 60 := 
sorry

end area_triangle_BRS_l100_100725


namespace find_other_number_l100_100434

theorem find_other_number (LCM : ℕ) (HCF : ℕ) (n1 : ℕ) (n2 : ℕ) 
  (h_lcm : LCM = 2310) (h_hcf : HCF = 26) (h_n1 : n1 = 210) :
  n2 = 286 :=
by
  sorry

end find_other_number_l100_100434


namespace determine_right_triangle_l100_100601

-- Definitions based on conditions
def condition_A (A B C : ℝ) : Prop := A^2 + B^2 = C^2
def condition_B (A B C : ℝ) : Prop := A^2 - B^2 = C^2
def condition_C (A B C : ℝ) : Prop := A + B = C
def condition_D (A B C : ℝ) : Prop := A / B = 3 / 4 ∧ B / C = 4 / 5

-- Problem statement: D cannot determine that triangle ABC is a right triangle
theorem determine_right_triangle (A B C : ℝ) : ¬ condition_D A B C :=
by sorry

end determine_right_triangle_l100_100601


namespace count_prime_pairs_sum_50_exactly_4_l100_100684

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p < q

theorem count_prime_pairs_sum_50_exactly_4 :
  {pq : ℕ × ℕ // prime_pairs_50 pq.1 pq.2}.to_finset.card = 4 :=
by sorry

end count_prime_pairs_sum_50_exactly_4_l100_100684


namespace monkeys_bananas_minimum_l100_100882

theorem monkeys_bananas_minimum (b1 b2 b3 : ℕ) (x y z : ℕ) : 
  (x = 2 * y) ∧ (z = (2 * y) / 3) ∧ 
  (x = (2 * b1) / 3 + (b2 / 3) + (5 * b3) / 12) ∧ 
  (y = (b1 / 6) + (b2 / 3) + (5 * b3) / 12) ∧ 
  (z = (b1 / 6) + (b2 / 3) + (b3 / 6)) →
  b1 = 324 ∧ b2 = 162 ∧ b3 = 72 ∧ (b1 + b2 + b3 = 558) :=
sorry

end monkeys_bananas_minimum_l100_100882


namespace additional_people_needed_l100_100637

-- Definitions corresponding to the given conditions
def person_hours (n : ℕ) (t : ℕ) : ℕ := n * t
def initial_people : ℕ := 8
def initial_time : ℕ := 10
def total_person_hours := person_hours initial_people initial_time

-- Lean statement of the problem
theorem additional_people_needed (new_time : ℕ) (new_people : ℕ) : 
  new_time = 5 → person_hours new_people new_time = total_person_hours → new_people - initial_people = 8 :=
by
  intro h1 h2
  sorry

end additional_people_needed_l100_100637


namespace cash_price_of_tablet_l100_100112

-- Define the conditions
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 4 * 40
def next_4_months_payment : ℕ := 4 * 35
def last_4_months_payment : ℕ := 4 * 30
def savings : ℕ := 70

-- Define the total installment payments
def total_installment_payments : ℕ := down_payment + first_4_months_payment + next_4_months_payment + last_4_months_payment

-- The statement to prove
theorem cash_price_of_tablet : total_installment_payments - savings = 450 := by
  -- proof goes here
  sorry

end cash_price_of_tablet_l100_100112


namespace A_finish_time_l100_100177

theorem A_finish_time {A_work B_work C_work : ℝ} 
  (h1 : A_work + B_work + C_work = 1/4)
  (h2 : B_work = 1/24)
  (h3 : C_work = 1/8) :
  1 / A_work = 12 := by
  sorry

end A_finish_time_l100_100177


namespace net_sag_calculation_l100_100621

open Real

noncomputable def sag_of_net (m1 m2 h1 h2 x1 : ℝ) : ℝ :=
  let g := 9.81
  let a := 28
  let b := -1.75
  let c := -50.75
  let D := b^2 - 4*a*c
  let sqrtD := sqrt D
  (1.75 + sqrtD) / (2 * a)

theorem net_sag_calculation :
  let m1 := 78.75
  let x1 := 1
  let h1 := 15
  let m2 := 45
  let h2 := 29
  sag_of_net m1 m2 h1 h2 x1 = 1.38 := 
by
  sorry

end net_sag_calculation_l100_100621


namespace repeating_decimal_as_fraction_l100_100465

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l100_100465


namespace value_of_x_l100_100587

theorem value_of_x (x : ℕ) (h : x + (10 * x + x) = 12) : x = 1 := by
  sorry

end value_of_x_l100_100587


namespace minimum_value_of_expression_l100_100649

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) :
  2 * x + 1 / x^6 ≥ 3 :=
sorry

end minimum_value_of_expression_l100_100649


namespace cargo_total_ship_l100_100016

-- Define the initial cargo and the additional cargo loaded
def initial_cargo := 5973
def additional_cargo := 8723

-- Define the total cargo the ship holds after loading additional cargo
def total_cargo := initial_cargo + additional_cargo

-- Statement of the problem
theorem cargo_total_ship (h1 : initial_cargo = 5973) (h2 : additional_cargo = 8723) : 
  total_cargo = 14696 := 
by
  sorry

end cargo_total_ship_l100_100016


namespace base7_digits_l100_100102

theorem base7_digits (D E F : ℕ) (h1 : D ≠ 0) (h2 : E ≠ 0) (h3 : F ≠ 0) (h4 : D < 7) (h5 : E < 7) (h6 : F < 7)
  (h_diff1 : D ≠ E) (h_diff2 : D ≠ F) (h_diff3 : E ≠ F)
  (h_eq : (49 * D + 7 * E + F) + (49 * E + 7 * F + D) + (49 * F + 7 * D + E) = 400 * D) :
  E + F = 6 :=
by
  sorry

end base7_digits_l100_100102


namespace paint_canvas_cost_ratio_l100_100509

theorem paint_canvas_cost_ratio (C P : ℝ) (hc : 0.6 * C = C - 0.4 * C) (hp : 0.4 * P = P - 0.6 * P)
 (total_cost_reduction : 0.4 * P + 0.6 * C = 0.44 * (P + C)) :
  P / C = 4 :=
by
  sorry

end paint_canvas_cost_ratio_l100_100509


namespace values_of_a_l100_100384

theorem values_of_a (a : ℝ) : 
  ∃a1 a2 : ℝ, 
  (∀ x y : ℝ, (y = 3 * x + a) ∧ (y = x^3 + 3 * a^2) → (x = 0) → (y = 3 * a^2)) →
  ((a = 0) ∨ (a = 1/3)) ∧ 
  ((a1 = 0) ∨ (a1 = 1/3)) ∧
  ((a2 = 0) ∨ (a2 = 1/3)) ∧ 
  (a ≠ a1 ∨ a ≠ a2) ∧ 
  (∃ n : ℤ, n = 2) :=
by sorry

end values_of_a_l100_100384


namespace mrs_oaklyn_rugs_l100_100258

theorem mrs_oaklyn_rugs (buying_price selling_price total_profit : ℕ) (h1 : buying_price = 40) (h2 : selling_price = 60) (h3 : total_profit = 400) : 
  ∃ (num_rugs : ℕ), num_rugs = 20 :=
by
  sorry

end mrs_oaklyn_rugs_l100_100258


namespace max_value_sin_function_l100_100135

theorem max_value_sin_function : 
  ∀ x, (-(π)/2 ≤ x ∧ x ≤ 0) → (3 * sin x + 2 ≤ 2) :=
by
  assume x h,
  sorry

end max_value_sin_function_l100_100135


namespace negative_integers_abs_le_4_l100_100618

theorem negative_integers_abs_le_4 :
  ∀ x : ℤ, x < 0 ∧ |x| ≤ 4 ↔ (x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4) :=
by
  sorry

end negative_integers_abs_le_4_l100_100618


namespace inversely_proportional_x_y_l100_100426

-- Statement of the problem
theorem inversely_proportional_x_y :
  ∃ k : ℝ, (∀ (x y : ℝ), (x * y = k) ∧ (x = 4) ∧ (y = 2) → x * (-5) = -8 / 5) :=
by
  sorry

end inversely_proportional_x_y_l100_100426


namespace polygons_intersection_l100_100393

/-- In a square with an area of 5, nine polygons, each with an area of 1, are placed. 
    Prove that some two of them must have an intersection area of at least 1 / 9. -/
theorem polygons_intersection 
  (S : ℝ) (hS : S = 5)
  (n : ℕ) (hn : n = 9)
  (polygons : Fin n → ℝ) (hpolys : ∀ i, polygons i = 1)
  (intersection : Fin n → Fin n → ℝ) : 
  ∃ i j : Fin n, i ≠ j ∧ intersection i j ≥ 1 / 9 := 
sorry

end polygons_intersection_l100_100393


namespace P_is_subtract_0_set_P_is_not_subtract_1_set_no_subtract_2_set_exists_all_subtract_1_sets_l100_100216

def is_subtract_set (T : Set ℕ) (i : ℕ) := T ⊆ Set.univ ∧ T ≠ {1} ∧ (∀ {x y : ℕ}, x ∈ Set.univ → y ∈ Set.univ → x + y ∈ T → x * y - i ∈ T)

theorem P_is_subtract_0_set : is_subtract_set {1, 2} 0 := sorry

theorem P_is_not_subtract_1_set : ¬ is_subtract_set {1, 2} 1 := sorry

theorem no_subtract_2_set_exists : ¬∃ T : Set ℕ, is_subtract_set T 2 := sorry

theorem all_subtract_1_sets : ∀ T : Set ℕ, is_subtract_set T 1 ↔ T = {1, 3} ∨ T = {1, 3, 5} := sorry

end P_is_subtract_0_set_P_is_not_subtract_1_set_no_subtract_2_set_exists_all_subtract_1_sets_l100_100216


namespace least_positive_integer_n_l100_100222

theorem least_positive_integer_n (n : ℕ) (h : (n > 0)) :
  (∃ m : ℕ, m > 0 ∧ (1 / (m : ℝ) - 1 / (m + 1 : ℝ) < 1 / 8) ∧ (∀ k : ℕ, k > 0 ∧ (1 / (k : ℝ) - 1 / (k + 1 : ℝ) < 1 / 8) → m ≤ k)) →
  n = 3 := by
  sorry

end least_positive_integer_n_l100_100222


namespace evaluate_polynomial_at_minus_two_l100_100639

noncomputable def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - x^2 + 2 * x + 5

theorem evaluate_polynomial_at_minus_two : polynomial (-2) = 5 := by
  sorry

end evaluate_polynomial_at_minus_two_l100_100639


namespace cyclic_quadrilateral_count_l100_100665

theorem cyclic_quadrilateral_count : 
  (∃ a b c d : ℕ, 
     a + b + c + d = 36 ∧ 
     a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ 
     a * b * c * d ≠ 0 ∧ 
     is_convex_cyclic_quadrilateral a b c d) 
  → (count_valid_quadrilaterals (36) = 823) :=
sorry

def is_convex_cyclic_quadrilateral (a b c d : ℕ) : Prop := 
  -- definition goes here, which checks whether the given a, b, c, d form a convex cyclic quadrilateral
  
noncomputable def count_valid_quadrilaterals (n : ℕ) : ℕ :=
  -- definition that counts the number of valid convex cyclic quadrilaterals with perimeter n goes here

end cyclic_quadrilateral_count_l100_100665


namespace solution_set_l100_100648

variable (x : ℝ)

noncomputable def expr := (x - 1)^2 / (x - 5)^2

theorem solution_set :
  { x : ℝ | expr x ≥ 0 } = { x | x < 5 } ∪ { x | x > 5 } :=
by
  sorry

end solution_set_l100_100648


namespace abs_neg_three_l100_100273

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l100_100273


namespace pirate_treasure_probability_l100_100013

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem pirate_treasure_probability :
  let p_treasure := 1 / 5
  let p_traps := 1 / 10
  let p_neither := 7 / 10
  let num_islands := 8
  let num_treasure := 4
  binomial num_islands num_treasure * p_treasure^num_treasure * p_neither^(num_islands - num_treasure) = 673 / 25000 :=
by
  sorry

end pirate_treasure_probability_l100_100013


namespace probability_defective_units_l100_100256

theorem probability_defective_units (X : ℝ) (hX : X > 0) :
  let defectA := (14 / 2000) * (0.40 * X)
  let defectB := (9 / 1500) * (0.35 * X)
  let defectC := (7 / 1000) * (0.25 * X)
  let total_defects := defectA + defectB + defectC
  let total_units := X
  let probability := total_defects / total_units
  probability = 0.00665 :=
by
  sorry

end probability_defective_units_l100_100256


namespace gain_percent_is_25_l100_100828

theorem gain_percent_is_25 (C S : ℝ) (h : 50 * C = 40 * S) : (S - C) / C * 100 = 25 :=
  sorry

end gain_percent_is_25_l100_100828


namespace solve_for_y_l100_100123

theorem solve_for_y (y : ℝ) (h : (8 * y^2 + 50 * y + 3) / (4 * y + 21) = 2 * y + 1) : y = 4.5 :=
by
  -- Proof goes here
  sorry

end solve_for_y_l100_100123


namespace min_value_on_neg_infinite_l100_100955

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def max_value_on_interval (F : ℝ → ℝ) (a b : ℝ) (max_val : ℝ) : Prop :=
∀ x, (0 < x → F x ≤ max_val) ∧ (∃ y, 0 < y ∧ F y = max_val)

theorem min_value_on_neg_infinite (f g : ℝ → ℝ) (a b : ℝ) (F : ℝ → ℝ)
  (h_odd_f : odd_function f) (h_odd_g : odd_function g)
  (h_def_F : ∀ x, F x = a * f x + b * g x + 2)
  (h_max_F_on_0_inf : max_value_on_interval F a b 8) :
  ∃ x, x < 0 ∧ F x = -4 :=
sorry

end min_value_on_neg_infinite_l100_100955


namespace solve_for_x_l100_100122

theorem solve_for_x (x : ℝ) (h : (x - 75) / 3 = (8 - 3 * x) / 4) : 
  x = 324 / 13 :=
sorry

end solve_for_x_l100_100122


namespace second_job_pay_rate_l100_100399

-- Definitions of the conditions
def h1 : ℕ := 3 -- hours for the first job
def r1 : ℕ := 7 -- rate for the first job
def h2 : ℕ := 2 -- hours for the second job
def h3 : ℕ := 4 -- hours for the third job
def r3 : ℕ := 12 -- rate for the third job
def d : ℕ := 5   -- number of days
def T : ℕ := 445 -- total earnings

-- The proof statement
theorem second_job_pay_rate (x : ℕ) : 
  d * (h1 * r1 + 2 * x + h3 * r3) = T ↔ x = 10 := 
by 
  -- Implement the necessary proof steps here
  sorry

end second_job_pay_rate_l100_100399


namespace recurring_to_fraction_l100_100476

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l100_100476


namespace resort_total_cost_l100_100615

noncomputable def first_cabin_cost (P : ℝ) := P
noncomputable def second_cabin_cost (P : ℝ) := (1/2) * P
noncomputable def third_cabin_cost (P : ℝ) := (1/6) * P
noncomputable def land_cost (P : ℝ) := 4 * P
noncomputable def pool_cost (P : ℝ) := P

theorem resort_total_cost (P : ℝ) (h : P = 22500) :
  first_cabin_cost P + pool_cost P + second_cabin_cost P + third_cabin_cost P + land_cost P = 150000 :=
by
  sorry

end resort_total_cost_l100_100615


namespace subset_implies_range_of_a_l100_100566

theorem subset_implies_range_of_a (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 5 → x > a) → a < -2 :=
by
  intro h
  sorry

end subset_implies_range_of_a_l100_100566


namespace min_value_of_3x_plus_4y_is_5_l100_100073

theorem min_value_of_3x_plus_4y_is_5 :
  ∀ (x y : ℝ), 0 < x → 0 < y → (3 / x + 1 / y = 5) → (∃ (b : ℝ), b = 3 * x + 4 * y ∧ ∀ (x y : ℝ), 0 < x → 0 < y → (3 / x + 1 / y = 5) → 3 * x + 4 * y ≥ b) :=
by
  intro x y x_pos y_pos h_eq
  let b := 5
  use b
  simp [b]
  sorry

end min_value_of_3x_plus_4y_is_5_l100_100073


namespace equation_of_line_with_x_intercept_and_slope_l100_100992

theorem equation_of_line_with_x_intercept_and_slope :
  ∃ (a b c : ℝ), a * x - b * y + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
sorry

end equation_of_line_with_x_intercept_and_slope_l100_100992


namespace GCD_is_six_l100_100312

-- Define the numbers
def a : ℕ := 36
def b : ℕ := 60
def c : ℕ := 90

-- Define the GCD using Lean's gcd function
def GCD_abc : ℕ := Nat.gcd (Nat.gcd a b) c

-- State the theorem that GCD of 36, 60, and 90 is 6
theorem GCD_is_six : GCD_abc = 6 := by
  sorry -- Proof skipped

end GCD_is_six_l100_100312


namespace find_number_l100_100894

theorem find_number (x : ℝ) : ((1.5 * x) / 7 = 271.07142857142856) → x = 1265 :=
by
  sorry

end find_number_l100_100894


namespace journey_divided_into_portions_l100_100731

theorem journey_divided_into_portions
  (total_distance : ℕ)
  (speed : ℕ)
  (time : ℝ)
  (portion_distance : ℕ)
  (portions_covered : ℕ)
  (h1 : total_distance = 35)
  (h2 : speed = 40)
  (h3 : time = 0.7)
  (h4 : portions_covered = 4)
  (distance_covered := speed * time)
  (one_portion_distance := distance_covered / portions_covered)
  (total_portions := total_distance / one_portion_distance) :
  total_portions = 5 := 
sorry

end journey_divided_into_portions_l100_100731


namespace combined_weight_of_boxes_l100_100970

def first_box_weight := 2
def second_box_weight := 11
def last_box_weight := 5

theorem combined_weight_of_boxes :
  first_box_weight + second_box_weight + last_box_weight = 18 := by
  sorry

end combined_weight_of_boxes_l100_100970


namespace find_a_l100_100232

noncomputable def A (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }
def B : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }

theorem find_a (a : ℝ) :
  A a ∪ B = B ↔ a = 0 ∨ a = 1 ∨ a = 1 / 2 :=
sorry

end find_a_l100_100232


namespace man_has_2_nickels_l100_100502

theorem man_has_2_nickels
  (d n : ℕ)
  (h1 : 10 * d + 5 * n = 70)
  (h2 : d + n = 8) :
  n = 2 := 
by
  -- omit the proof
  sorry

end man_has_2_nickels_l100_100502


namespace juan_original_number_l100_100563

theorem juan_original_number (x : ℝ) (h : (3 * (x + 3) - 4) / 2 = 10) : x = 5 :=
by
  sorry

end juan_original_number_l100_100563


namespace perimeter_T2_l100_100712

def Triangle (a b c : ℝ) :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem perimeter_T2 (a b c : ℝ) (h : Triangle a b c) (ha : a = 10) (hb : b = 15) (hc : c = 20) : 
  let AM := a / 2
  let BN := b / 2
  let CP := c / 2
  0 < AM ∧ 0 < BN ∧ 0 < CP →
  AM + BN + CP = 22.5 :=
by
  sorry

end perimeter_T2_l100_100712


namespace washes_per_bottle_l100_100249

def bottle_cost : ℝ := 4.0
def total_weeks : ℕ := 20
def total_cost : ℝ := 20.0

theorem washes_per_bottle : (total_weeks / (total_cost / bottle_cost)) = 4 := by
  sorry

end washes_per_bottle_l100_100249


namespace solve_equation_l100_100983

theorem solve_equation (x : ℝ) : (x + 2) * (x + 1) = 3 * (x + 1) ↔ (x = -1 ∨ x = 1) :=
by sorry

end solve_equation_l100_100983


namespace claire_photos_l100_100491

theorem claire_photos (L R C : ℕ) (h1 : L = R) (h2 : L = 3 * C) (h3 : R = C + 28) : C = 14 := by
  sorry

end claire_photos_l100_100491


namespace find_a1_l100_100570

-- Given an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Arithmetic sequence is monotonically increasing
def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

-- First condition: sum of first three terms
def sum_first_three_terms (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 12

-- Second condition: product of first three terms
def product_first_three_terms (a : ℕ → ℝ) : Prop :=
  a 0 * a 1 * a 2 = 48

-- Proving that a_1 = 2 given the conditions
theorem find_a1 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : is_monotonically_increasing a)
  (h3 : sum_first_three_terms a) (h4 : product_first_three_terms a) : a 0 = 2 :=
by
  -- Proof will be filled in here
  sorry

end find_a1_l100_100570


namespace mixed_number_sum_l100_100445

theorem mixed_number_sum : (2 + (1 / 10 : ℝ)) + (3 + (11 / 100 : ℝ)) = 5.21 := by
  sorry

end mixed_number_sum_l100_100445


namespace locus_equation_perimeter_greater_than_3sqrt3_l100_100085

-- The locus W and the conditions 
def locus_eq (P : ℝ × ℝ) : Prop :=
  P.snd = P.fst ^ 2 + 1/4

def point_on_x_axis (P : ℝ × ℝ) : Prop :=
  |P.snd| = sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2)

-- Prove part (1): The locus W is y = x^2 + 1/4
theorem locus_equation (x y : ℝ) : 
  point_on_x_axis (x, y) → locus_eq (x, y) :=
sorry

-- Prove part (2): Perimeter of rectangle ABCD is greater than 3sqrt(3) if three vertices are on W
def rectangle_on_w (A B C : ℝ × ℝ) (D : ℝ × ℝ) : Prop :=
  locus_eq A ∧ locus_eq B ∧ locus_eq C ∧ locus_eq D ∧ 
  (∃x₁ x₂ x₃ x₄ : ℝ, A = (x₁, x₁ ^ 2 + 1/4) ∧ B = (x₂, x₂ ^ 2 + 1/4) ∧ 
  C = (x₃, x₃ ^ 2 + 1/4) ∧ D = (x₄, x₄ ^ 2 + 1/4))

theorem perimeter_greater_than_3sqrt3 (A B C D : ℝ × ℝ) : 
  rectangle_on_w A B C D → 
  2 * (abs (B.fst - A.fst) + abs (C.fst - B.fst)) > 3 * sqrt 3 :=
sorry

end locus_equation_perimeter_greater_than_3sqrt3_l100_100085


namespace inequality_proof_l100_100805

variable {x : ℝ}
variable {n : ℕ}
variable {a : ℝ}

theorem inequality_proof (h1 : x > 0) (h2 : n > 0) (h3 : x + a / x^n ≥ n + 1) : a = n^n := 
sorry

end inequality_proof_l100_100805


namespace g_of_zero_l100_100365

theorem g_of_zero (f g : ℤ → ℤ) (h₁ : ∀ x, f x = 2 * x + 3) (h₂ : ∀ x, g (x + 2) = f x) : 
  g 0 = -1 :=
by
  sorry

end g_of_zero_l100_100365


namespace find_m_l100_100075

theorem find_m {x : ℝ} (m : ℝ) (h : ∀ x, (0 < x ∧ x < 2) ↔ (-1/2 * x^2 + 2 * x > m * x)) : m = 1 :=
sorry

end find_m_l100_100075


namespace abs_neg_three_l100_100287

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l100_100287


namespace primes_sum_50_l100_100674

-- Given primes less than 25
def primes_less_25 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Definition of prime to identify prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The problem statement to prove the number of pairs
theorem primes_sum_50 : 
  (List.filter (λ p, p < 50 ∧ is_prime p ∧ is_prime (50 - p) ∧ p ≤ (50 - p)) 
    ((primes_less_25).bind (λ x, [(x, 50 - x)]))).length = 4 :=
sorry

end primes_sum_50_l100_100674


namespace real_solutions_count_l100_100383

theorem real_solutions_count :
  (∃ x : ℝ, |x - 2| - 4 = 1 / |x - 3|) ∧
  (∃ y : ℝ, |y - 2| - 4 = 1 / |y - 3| ∧ x ≠ y) :=
sorry

end real_solutions_count_l100_100383


namespace prime_pairs_sum_50_l100_100675

/-- 
We define a function to check if a number is prime.
Then, we define a function to count the pairs of prime numbers whose sum is 50 
and check that there are exactly 4 such pairs.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_pairs_with_sum_50 : ℕ :=
  (Finset.filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = 50)
     ((Finset.Ico 2 25).product (Finset.Ico 2 50))).card

theorem prime_pairs_sum_50 : count_prime_pairs_with_sum_50 = 4 :=
  sorry

end prime_pairs_sum_50_l100_100675


namespace yanna_kept_apples_l100_100153

-- Define the given conditions
def initial_apples : ℕ := 60
def percentage_given_to_zenny : ℝ := 0.40
def percentage_given_to_andrea : ℝ := 0.25

-- Prove the main statement
theorem yanna_kept_apples : 
  let apples_given_to_zenny := (percentage_given_to_zenny * initial_apples)
  let apples_remaining_after_zenny := (initial_apples - apples_given_to_zenny)
  let apples_given_to_andrea := (percentage_given_to_andrea * apples_remaining_after_zenny)
  let apples_kept := (apples_remaining_after_zenny - apples_given_to_andrea)
  apples_kept = 27 :=
by
  sorry

end yanna_kept_apples_l100_100153


namespace motorcycle_tire_max_distance_l100_100310

theorem motorcycle_tire_max_distance :
  let wear_front := (1 : ℝ) / 25000
  let wear_rear := (1 : ℝ) / 15000
  let s := 18750
  wear_front * (s / 2) + wear_rear * (s / 2) = 1 :=
by 
  let wear_front := (1 : ℝ) / 25000
  let wear_rear := (1 : ℝ) / 15000
  sorry

end motorcycle_tire_max_distance_l100_100310


namespace minimum_value_of_f_l100_100052

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem minimum_value_of_f : ∃ x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end minimum_value_of_f_l100_100052


namespace solve_problem_l100_100240

def is_solution (a : ℕ) : Prop :=
  a % 3 = 1 ∧ ∃ k : ℕ, a = 5 * k

theorem solve_problem : ∃ a : ℕ, is_solution a ∧ ∀ b : ℕ, is_solution b → a ≤ b := 
  sorry

end solve_problem_l100_100240


namespace correlation_comparison_l100_100865

def data_X_Y : list (ℝ × ℝ) := [(10, 1), (11.3, 2), (11.8, 3), (12.5, 4), (13, 5)]
def data_U_V : list (ℝ × ℝ) := [(10, 5), (11.3, 4), (11.8, 3), (12.5, 2), (13, 1)]

def mean (lst : list ℝ) : ℝ :=
  lst.sum / lst.length

-- Calculate means
def mean_X : ℝ := mean (data_X_Y.map Prod.fst)
def mean_Y : ℝ := mean (data_X_Y.map Prod.snd)
def mean_U : ℝ := mean (data_U_V.map Prod.fst)
def mean_V : ℝ := mean (data_U_V.map Prod.snd)

-- Calculate the correlation coefficients (placeholders to insert actual computation below)
def correlation_coefficient (data : list (ℝ × ℝ)) : ℝ :=
  sorry

def r1 : ℝ := correlation_coefficient data_X_Y
def r2 : ℝ := correlation_coefficient data_U_V

theorem correlation_comparison :
  r2 < 0 ∧ 0 < r1 :=
sorry

end correlation_comparison_l100_100865


namespace meter_to_leap_l100_100578

theorem meter_to_leap
  (strides leaps bounds meters : ℝ)
  (h1 : 3 * strides = 4 * leaps)
  (h2 : 5 * bounds = 7 * strides)
  (h3 : 2 * bounds = 9 * meters) :
  1 * meters = (56 / 135) * leaps :=
by
  sorry

end meter_to_leap_l100_100578


namespace jenna_water_cups_l100_100969

theorem jenna_water_cups (O S W : ℕ) (h1 : S = 3 * O) (h2 : W = 3 * S) (h3 : O = 4) : W = 36 :=
by
  sorry

end jenna_water_cups_l100_100969


namespace max_integer_k_l100_100660

noncomputable theory

open Real

def f (x : ℝ) : ℝ := (x * (1 + log x)) / (x - 1)

theorem max_integer_k {k : ℤ} :
  (∀ x > 1, f x > (k:ℝ)) ↔ k ≤ 3 :=
by
  sorry

end max_integer_k_l100_100660


namespace product_of_numbers_l100_100739

theorem product_of_numbers (x y : ℕ) (h1 : x + y = 15) (h2 : x - y = 11) : x * y = 26 :=
by
  sorry

end product_of_numbers_l100_100739


namespace best_discount_sequence_l100_100438

/-- 
The initial price of the book is 30.
Stay focused on two sequences of discounts.
Sequence 1: $5 off, then 10% off, then $2 off if applicable.
Sequence 2: 10% off, then $5 off, then $2 off if applicable.
Compare the final prices obtained from applying these sequences.
-/
noncomputable def initial_price : ℝ := 30
noncomputable def five_off (price : ℝ) : ℝ := price - 5
noncomputable def ten_percent_off (price : ℝ) : ℝ := 0.9 * price
noncomputable def additional_two_off_if_applicable (price : ℝ) : ℝ := 
  if price > 20 then price - 2 else price

noncomputable def sequence1_final_price : ℝ := 
  additional_two_off_if_applicable (ten_percent_off (five_off initial_price))

noncomputable def sequence2_final_price : ℝ := 
  additional_two_off_if_applicable (five_off (ten_percent_off initial_price))

theorem best_discount_sequence : 
  sequence2_final_price = 20 ∧ 
  sequence2_final_price < sequence1_final_price ∧ 
  sequence1_final_price - sequence2_final_price = 0.5 :=
by
  sorry

end best_discount_sequence_l100_100438


namespace eval_expression_l100_100641

theorem eval_expression (a b c : ℕ) (h₀ : a = 3) (h₁ : b = 2) (h₂ : c = 1) : 
  (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 :=
by
  sorry

end eval_expression_l100_100641


namespace circumscribed_sphere_radius_l100_100305

theorem circumscribed_sphere_radius (a b R : ℝ) (ha : a > 0) (hb : b > 0) :
  R = b^2 / (2 * (Real.sqrt (b^2 - a^2))) :=
sorry

end circumscribed_sphere_radius_l100_100305


namespace range_of_a_l100_100376

noncomputable def f (a : ℝ) (x : ℝ) := 2 * a * x^2 + 4 * (a - 3) * x + 5

theorem range_of_a (a : ℝ) :
  (∀ x < 3, f a x ≤ f a (3 : ℝ)) ↔ 0 ≤ a ∧ a ≤ 3 / 4 :=
by
  sorry

end range_of_a_l100_100376


namespace expression_bounds_l100_100100

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) (hw : 0 ≤ w) (hw1 : w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by
  sorry

end expression_bounds_l100_100100


namespace choose_lines_intersect_l100_100091

-- We need to define the proof problem
theorem choose_lines_intersect : 
  ∃ (lines : ℕ → ℝ × ℝ → ℝ), 
    (∀ i j, i < 100 ∧ j < 100 ∧ i ≠ j → (lines i = lines j) → ∃ (p : ℕ), p = 2022) :=
sorry

end choose_lines_intersect_l100_100091


namespace eval_abc_l100_100040

theorem eval_abc (a b c : ℚ) (h1 : a = 1 / 2) (h2 : b = 3 / 4) (h3 : c = 8) :
  a^3 * b^2 * c = 9 / 16 :=
by
  sorry

end eval_abc_l100_100040


namespace fraction_meaningful_l100_100074

theorem fraction_meaningful (x : ℝ) : (x-5) ≠ 0 ↔ (1 / (x - 5)) = (1 / (x - 5)) := 
by 
  sorry

end fraction_meaningful_l100_100074


namespace find_m_plus_n_l100_100714

def operation (m n : ℕ) : ℕ := m^n + m * n

theorem find_m_plus_n :
  ∃ (m n : ℕ), (2 ≤ m) ∧ (2 ≤ n) ∧ (operation m n = 64) ∧ (m + n = 6) :=
by {
  -- Begin the proof context
  sorry
}

end find_m_plus_n_l100_100714


namespace pencils_in_drawer_l100_100145

theorem pencils_in_drawer (P : ℕ) 
  (h1 : 19 + 16 = 35)
  (h2 : P + 35 = 78) : 
  P = 43 := 
by
  sorry

end pencils_in_drawer_l100_100145


namespace problem_l100_100569

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.exp x

theorem problem (a : ℝ) (x : ℝ) (hx : x ∈ Set.Ici (-5)) (ha : a = 1) : 
  f x a + x + 5 ≥ -6 / Real.exp 5 := 
sorry

end problem_l100_100569


namespace tan_alpha_implication_l100_100050

theorem tan_alpha_implication (α : ℝ) (h : Real.tan α = 2) :
    (2 * Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 3 / 5 := 
by 
  sorry

end tan_alpha_implication_l100_100050


namespace tan_alpha_plus_405_deg_l100_100223

theorem tan_alpha_plus_405_deg (α : ℝ) (h : Real.tan (180 - α) = -4 / 3) : Real.tan (α + 405) = -7 := 
sorry

end tan_alpha_plus_405_deg_l100_100223


namespace compute_f_g_at_2_l100_100697

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 4 * x - 1

theorem compute_f_g_at_2 :
  f (g 2) = 49 :=
by
  sorry

end compute_f_g_at_2_l100_100697


namespace obtain_any_natural_l100_100335

-- Define the operations
def op1 (x : ℕ) : ℕ := 3 * x + 1
def op2 (x : ℤ) : ℕ := (x / 2).to_nat

-- Theorem statement: Starting from 1, any natural number n can be obtained using op1 and op2
theorem obtain_any_natural (n : ℕ) : ∃ (f : ℕ → ℕ), f 1 = n ∧ ∀ i, f (i + 1) = op1 (f i) ∨ f (i + 1) = op2 (f i) := sorry

end obtain_any_natural_l100_100335


namespace savings_difference_correct_l100_100519

noncomputable def savings_1989_dick : ℝ := 5000
noncomputable def savings_1989_jane : ℝ := 5000

noncomputable def savings_1990_dick : ℝ := savings_1989_dick + 0.10 * savings_1989_dick
noncomputable def savings_1990_jane : ℝ := savings_1989_jane - 0.05 * savings_1989_jane

noncomputable def savings_1991_dick : ℝ := savings_1990_dick + 0.07 * savings_1990_dick
noncomputable def savings_1991_jane : ℝ := savings_1990_jane + 0.08 * savings_1990_jane

noncomputable def savings_1992_dick : ℝ := savings_1991_dick - 0.12 * savings_1991_dick
noncomputable def savings_1992_jane : ℝ := savings_1991_jane + 0.15 * savings_1991_jane

noncomputable def total_savings_dick : ℝ :=
savings_1989_dick + savings_1990_dick + savings_1991_dick + savings_1992_dick

noncomputable def total_savings_jane : ℝ :=
savings_1989_jane + savings_1990_jane + savings_1991_jane + savings_1992_jane

noncomputable def difference_of_savings : ℝ :=
total_savings_dick - total_savings_jane

theorem savings_difference_correct :
  difference_of_savings = 784.30 :=
by sorry

end savings_difference_correct_l100_100519


namespace initial_men_count_l100_100860

theorem initial_men_count 
  (M : ℕ)
  (h1 : 8 * M * 30 = (M + 77) * 6 * 50) :
  M = 63 :=
by
  sorry

end initial_men_count_l100_100860


namespace sherman_drives_nine_hours_a_week_l100_100419

-- Define the daily commute time in minutes.
def daily_commute_time := 30 + 30

-- Define the number of weekdays Sherman commutes.
def weekdays := 5

-- Define the weekly commute time in minutes.
def weekly_commute_time := weekdays * daily_commute_time

-- Define the conversion from minutes to hours.
def minutes_to_hours (m : ℕ) : ℕ := m / 60

-- Define the weekend driving time in hours.
def weekend_driving_time := 2 * 2

-- Define the total weekly driving time in hours.
def total_weekly_driving_time := minutes_to_hours weekly_commute_time + weekend_driving_time

-- The theorem we need to prove
theorem sherman_drives_nine_hours_a_week :
  total_weekly_driving_time = 9 :=
by
  sorry

end sherman_drives_nine_hours_a_week_l100_100419


namespace sample_size_correct_l100_100884

def population_size : Nat := 8000
def sampled_students : List Nat := List.replicate 400 1 -- We use 1 as a placeholder for the heights

theorem sample_size_correct : sampled_students.length = 400 := by
  sorry

end sample_size_correct_l100_100884


namespace chocolate_cost_l100_100192

theorem chocolate_cost (Ccb Cc : ℝ) (h1 : Ccb = 6) (h2 : Ccb = Cc + 3) : Cc = 3 :=
by
  sorry

end chocolate_cost_l100_100192


namespace birds_percentage_hawks_l100_100549

-- Define the conditions and the main proof problem
theorem birds_percentage_hawks (H : ℝ) :
  (0.4 * (1 - H) + 0.25 * 0.4 * (1 - H) + H = 0.65) → (H = 0.3) :=
by
  intro h
  sorry

end birds_percentage_hawks_l100_100549


namespace total_vases_l100_100397

theorem total_vases (vases_per_day : ℕ) (days : ℕ) (total_vases : ℕ) 
  (h1 : vases_per_day = 16) 
  (h2 : days = 16) 
  (h3 : total_vases = vases_per_day * days) : 
  total_vases = 256 := 
by 
  sorry

end total_vases_l100_100397


namespace product_of_numbers_eq_120_l100_100139

theorem product_of_numbers_eq_120 (x y P : ℝ) (h1 : x + y = 23) (h2 : x^2 + y^2 = 289) (h3 : x * y = P) : P = 120 := 
sorry

end product_of_numbers_eq_120_l100_100139


namespace sum_of_consecutive_integers_between_ln20_l100_100197

theorem sum_of_consecutive_integers_between_ln20 : ∃ a b : ℤ, a < b ∧ b = a + 1 ∧ 1 ≤ a ∧ a + 1 ≤ 3 ∧ (a + b = 4) :=
by
  sorry

end sum_of_consecutive_integers_between_ln20_l100_100197


namespace arithmetic_geometric_sequence_l100_100054

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_common_diff : d = 2) (h_geom : a 2 ^ 2 = a 1 * a 5) : 
  a 2 = 3 :=
by
  sorry

end arithmetic_geometric_sequence_l100_100054


namespace largest_k_consecutive_sum_l100_100205

theorem largest_k_consecutive_sum (k n : ℕ) :
  (5^7 = (k * (2 * n + k + 1)) / 2) → 1 ≤ k → k * (2 * n + k + 1) = 2 * 5^7 → k = 250 :=
sorry

end largest_k_consecutive_sum_l100_100205


namespace largest_exponent_l100_100457

theorem largest_exponent : 
  ∀ (a b c d e : ℕ), a = 2^5000 → b = 3^4000 → c = 4^3000 → d = 5^2000 → e = 6^1000 → b > a ∧ b > c ∧ b > d ∧ b > e :=
by
  sorry

end largest_exponent_l100_100457


namespace range_of_a_l100_100227

theorem range_of_a 
  (a : ℝ) 
  (h₀ : ∀ x : ℝ, (3 ≤ x ∧ x ≤ 4) ↔ (y = 2 * x + (3 - a))) : 
  9 ≤ a ∧ a ≤ 11 := 
sorry

end range_of_a_l100_100227


namespace recurring_decimal_sum_l100_100344

theorem recurring_decimal_sum (x y : ℚ) (hx : x = 4/9) (hy : y = 7/9) :
  x + y = 11/9 :=
by
  rw [hx, hy]
  exact sorry

end recurring_decimal_sum_l100_100344


namespace solve_for_x_l100_100822

theorem solve_for_x (x : ℝ) (h₁ : 3 * x^2 - 9 * x = 0) (h₂ : x ≠ 0) : x = 3 := 
by {
  sorry
}

end solve_for_x_l100_100822


namespace expected_value_of_coin_flip_l100_100010

open ProbabilityTheory

noncomputable def coinFlipWinnings : pmf ℤ :=
  pmf.of_multiset { 5, -3 }

theorem expected_value_of_coin_flip :
  expected_value coinFlipWinnings = 1 := by
  sorry

end expected_value_of_coin_flip_l100_100010


namespace interest_difference_l100_100774

theorem interest_difference (P R T : ℝ) (SI : ℝ) (Diff : ℝ) :
  P = 250 ∧ R = 4 ∧ T = 8 ∧ SI = (P * R * T) / 100 ∧ Diff = P - SI → Diff = 170 :=
by sorry

end interest_difference_l100_100774


namespace biff_break_even_hours_l100_100629

def totalSpent (ticket drinks snacks headphones : ℕ) : ℕ :=
  ticket + drinks + snacks + headphones

def netEarningsPerHour (earningsCost wifiCost : ℕ) : ℕ :=
  earningsCost - wifiCost

def hoursToBreakEven (totalSpent netEarnings : ℕ) : ℕ :=
  totalSpent / netEarnings

-- given conditions
def given_ticket : ℕ := 11
def given_drinks : ℕ := 3
def given_snacks : ℕ := 16
def given_headphones : ℕ := 16
def given_earningsPerHour : ℕ := 12
def given_wifiCostPerHour : ℕ := 2

theorem biff_break_even_hours :
  hoursToBreakEven (totalSpent given_ticket given_drinks given_snacks given_headphones) 
                   (netEarningsPerHour given_earningsPerHour given_wifiCostPerHour) = 3 :=
by
  sorry

end biff_break_even_hours_l100_100629


namespace cuberoot_eight_is_512_l100_100138

-- Define the condition on x
def cuberoot_is_eight (x : ℕ) : Prop := 
  x^(1 / 3) = 8

-- The statement to be proved
theorem cuberoot_eight_is_512 : ∃ x : ℕ, cuberoot_is_eight x ∧ x = 512 := 
by 
  -- Proof is omitted
  sorry

end cuberoot_eight_is_512_l100_100138


namespace coffee_amount_l100_100314

theorem coffee_amount (total_mass : ℕ) (coffee_ratio : ℕ) (milk_ratio : ℕ) (h_total_mass : total_mass = 4400) (h_coffee_ratio : coffee_ratio = 2) (h_milk_ratio : milk_ratio = 9) : 
  total_mass * coffee_ratio / (coffee_ratio + milk_ratio) = 800 :=
by
  -- Placeholder for the proof
  sorry

end coffee_amount_l100_100314


namespace cookies_prepared_l100_100914

theorem cookies_prepared (n_people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) 
  (h1 : n_people = 25) (h2 : cookies_per_person = 45) : total_cookies = 1125 :=
by
  sorry

end cookies_prepared_l100_100914


namespace initial_puppies_correct_l100_100169

def initial_puppies (total_puppies_after: ℝ) (bought_puppies: ℝ) : ℝ :=
  total_puppies_after - bought_puppies

theorem initial_puppies_correct : initial_puppies (4.2 * 5.0) 3.0 = 18.0 := by
  sorry

end initial_puppies_correct_l100_100169


namespace turnip_bag_weight_l100_100007

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l100_100007


namespace length_of_OP_is_sqrt_200_div_3_l100_100084

open Real

def square (a : ℝ) := a * a

theorem length_of_OP_is_sqrt_200_div_3 (KL MO MP OP : ℝ) (h₁ : KL = 10)
  (h₂: MO = MP) (h₃: square (10) = 100)
  (h₄ : 1 / 6 * 100 = 1 / 2 * (MO * MP)) : OP = sqrt (200/3) :=
by
  sorry

end length_of_OP_is_sqrt_200_div_3_l100_100084


namespace sixth_number_is_811_l100_100777

noncomputable def sixth_number_in_21st_row : ℕ := 
  let n := 21 
  let k := 6
  let total_numbers_up_to_previous_row := n * n
  let position_in_row := total_numbers_up_to_previous_row + k
  2 * position_in_row - 1

theorem sixth_number_is_811 : sixth_number_in_21st_row = 811 := by
  sorry

end sixth_number_is_811_l100_100777


namespace abs_h_eq_1_div_2_l100_100141

theorem abs_h_eq_1_div_2 {h : ℝ} 
  (h_sum_sq_roots : ∀ (r s : ℝ), (r + s) = 4 * h ∧ (r * s) = -8 → (r ^ 2 + s ^ 2) = 20) : 
  |h| = 1 / 2 :=
sorry

end abs_h_eq_1_div_2_l100_100141


namespace second_hose_correct_l100_100380

/-- Define the problem parameters -/
def first_hose_rate : ℕ := 50
def initial_hours : ℕ := 3
def additional_hours : ℕ := 2
def total_capacity : ℕ := 390

/-- Define the total hours the first hose was used -/
def total_hours (initial_hours additional_hours : ℕ) : ℕ := initial_hours + additional_hours

/-- Define the amount of water sprayed by the first hose -/
def first_hose_total (first_hose_rate initial_hours additional_hours : ℕ) : ℕ :=
  first_hose_rate * (initial_hours + additional_hours)

/-- Define the remaining water needed to fill the pool -/
def remaining_water (total_capacity first_hose_total : ℕ) : ℕ :=
  total_capacity - first_hose_total

/-- Define the additional water sprayed by the first hose during the last 2 hours -/
def additional_first_hose (first_hose_rate additional_hours : ℕ) : ℕ :=
  first_hose_rate * additional_hours

/-- Define the water sprayed by the second hose -/
def second_hose_total (remaining_water additional_first_hose : ℕ) : ℕ :=
  remaining_water - additional_first_hose

/-- Define the rate of the second hose (output) -/
def second_hose_rate (second_hose_total additional_hours : ℕ) : ℕ :=
  second_hose_total / additional_hours

/-- Define the theorem we want to prove -/
theorem second_hose_correct :
  second_hose_rate
    (second_hose_total
        (remaining_water total_capacity (first_hose_total first_hose_rate initial_hours additional_hours))
        (additional_first_hose first_hose_rate additional_hours))
    additional_hours = 20 := by
  sorry

end second_hose_correct_l100_100380


namespace eval_expression_l100_100887

theorem eval_expression : 6 + 15 / 3 - 4^2 + 1 = -4 := by
  sorry

end eval_expression_l100_100887


namespace toys_produced_each_day_l100_100165

theorem toys_produced_each_day (weekly_production : ℕ) (days_worked : ℕ) (h₁ : weekly_production = 4340) (h₂ : days_worked = 2) : weekly_production / days_worked = 2170 :=
by {
  -- Proof can be filled in here
  sorry
}

end toys_produced_each_day_l100_100165


namespace cost_of_each_card_is_2_l100_100097

-- Define the conditions
def christmas_cards : ℕ := 20
def birthday_cards : ℕ := 15
def total_spent : ℝ := 70

-- Define the total number of cards
def total_cards : ℕ := christmas_cards + birthday_cards

-- Define the cost per card
noncomputable def cost_per_card : ℝ := total_spent / total_cards

-- The theorem
theorem cost_of_each_card_is_2 : cost_per_card = 2 := by
  sorry

end cost_of_each_card_is_2_l100_100097


namespace find_p_l100_100837

theorem find_p (m n p : ℝ) 
  (h₁ : m = 5 * n + 5) 
  (h₂ : m + 2 = 5 * (n + p) + 5) :
  p = 2 / 5 :=
by sorry

end find_p_l100_100837


namespace repeating_decimal_eq_l100_100466

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l100_100466


namespace almond_butter_servings_l100_100500

def servings_of_almond_butter (tbsp_in_container : ℚ) (tbsp_per_serving : ℚ) : ℚ :=
  tbsp_in_container / tbsp_per_serving

def container_holds : ℚ := 37 + 2/3

def serving_size : ℚ := 3

theorem almond_butter_servings :
  servings_of_almond_butter container_holds serving_size = 12 + 5/9 := 
by
  sorry

end almond_butter_servings_l100_100500


namespace initial_speeds_l100_100408

/-- Motorcyclists Vasya and Petya ride at constant speeds around a circular track 1 km long.
    Petya overtakes Vasya every 2 minutes. Then Vasya doubles his speed and now he himself 
    overtakes Petya every 2 minutes. What were the initial speeds of Vasya and Petya? 
    Answer: 1000 and 1500 meters per minute.
-/

theorem initial_speeds (V_v V_p : ℕ) (track_length : ℕ) (time_interval : ℕ) 
  (h1 : track_length = 1000)
  (h2 : time_interval = 2)
  (h3 : V_p - V_v = track_length / time_interval)
  (h4 : 2 * V_v - V_p = track_length / time_interval):
  V_v = 1000 ∧ V_p = 1500 :=
by
  sorry

end initial_speeds_l100_100408


namespace find_sum_of_perimeters_l100_100430

variables (x y : ℝ)
noncomputable def sum_of_perimeters := 4 * x + 4 * y

theorem find_sum_of_perimeters (h1 : x^2 + y^2 = 65) (h2 : x^2 - y^2 = 33) :
  sum_of_perimeters x y = 44 :=
sorry

end find_sum_of_perimeters_l100_100430


namespace problem_statement_l100_100514

theorem problem_statement : 20 * (256 / 4 + 64 / 16 + 16 / 64 + 2) = 1405 := by
  sorry

end problem_statement_l100_100514


namespace committee_count_l100_100499

noncomputable def num_acceptable_committees (total_people : ℕ) (committee_size : ℕ) (conditions : List (Set ℕ)) : ℕ := sorry

theorem committee_count :
  num_acceptable_committees 9 5 [ {1, 2}, {3, 4} ] = 41 := sorry

end committee_count_l100_100499


namespace find_integer_solutions_l100_100795

theorem find_integer_solutions :
  {n : ℤ | n + 2 ∣ n^2 + 3} = {-9, -3, -1, 5} :=
  sorry

end find_integer_solutions_l100_100795


namespace angle_sum_eq_pi_div_2_l100_100606

open Real

theorem angle_sum_eq_pi_div_2 (θ1 θ2 : ℝ) (h1 : 0 < θ1 ∧ θ1 < π / 2) (h2 : 0 < θ2 ∧ θ2 < π / 2)
  (h : (sin θ1)^2020 / (cos θ2)^2018 + (cos θ1)^2020 / (sin θ2)^2018 = 1) :
  θ1 + θ2 = π / 2 :=
sorry

end angle_sum_eq_pi_div_2_l100_100606


namespace arithmetic_sequence_50th_term_l100_100148

-- Definitions based on the conditions stated
def first_term := 3
def common_difference := 5
def n := 50

-- Function to calculate the n-th term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- The theorem that needs to be proven
theorem arithmetic_sequence_50th_term : nth_term first_term common_difference n = 248 := 
by
  sorry

end arithmetic_sequence_50th_term_l100_100148


namespace recurring_decimal_to_fraction_correct_l100_100472

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l100_100472


namespace factorize_expression_l100_100643

theorem factorize_expression
  (x : ℝ) :
  ( (x^2-1)*(x^4+x^2+1)-(x^3+1)^2 ) = -2*(x + 1)*(x^2 - x + 1) :=
by
  sorry

end factorize_expression_l100_100643


namespace repeating_decimal_as_fraction_l100_100462

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l100_100462


namespace least_number_to_add_l100_100151

theorem least_number_to_add (n : ℕ) :
  (exists n, 1202 + n % 4 = 0 ∧ (∀ m, (1202 + m) % 4 = 0 → n ≤ m)) → n = 2 :=
by
  sorry

end least_number_to_add_l100_100151


namespace total_expenditure_l100_100324

-- Definitions of costs and purchases
def bracelet_cost : ℕ := 4
def keychain_cost : ℕ := 5
def coloring_book_cost : ℕ := 3

def paula_bracelets : ℕ := 2
def paula_keychains : ℕ := 1

def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 1

-- Hypothesis stating the total expenditure for Paula and Olive
theorem total_expenditure
  (bracelet_cost keychain_cost coloring_book_cost : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ) :
  paula_bracelets * bracelet_cost + paula_keychains * keychain_cost + olive_coloring_books * coloring_book_cost + olive_bracelets * bracelet_cost = 20 := 
  by
  -- Applying the given costs
  let bracelet_cost := 4
  let keychain_cost := 5
  let coloring_book_cost := 3 

  -- Applying the purchases made by Paula and Olive
  let paula_bracelets := 2
  let paula_keychains := 1
  let olive_coloring_books := 1
  let olive_bracelets := 1

  sorry

end total_expenditure_l100_100324


namespace solve_for_p_l100_100576

-- Conditions
def C1 (n : ℕ) : Prop := (3 : ℚ) / 4 = n / 48
def C2 (m n : ℕ) : Prop := (3 : ℚ) / 4 = (m + n) / 96
def C3 (p m : ℕ) : Prop := (3 : ℚ) / 4 = (p - m) / 160

-- Theorem to prove
theorem solve_for_p (n m p : ℕ) (h1 : C1 n) (h2 : C2 m n) (h3 : C3 p m) : p = 156 := 
by 
    sorry

end solve_for_p_l100_100576


namespace find_a_l100_100543

theorem find_a (a : ℝ) : (∃ k : ℝ, (x - 2) * (x + k) = x^2 + a * x - 5) ↔ a = 1 / 2 :=
by
  sorry

end find_a_l100_100543


namespace focus_of_hyperbola_l100_100208

theorem focus_of_hyperbola (m : ℝ) :
  let focus_parabola := (0, 4)
  let focus_hyperbola_upper := (0, 4)
  ∃ focus_parabola, ∃ focus_hyperbola_upper, 
    (focus_parabola = (0, 4)) ∧ (focus_hyperbola_upper = (0, 4)) ∧ 
    (3 + m = 16) → m = 13 :=
by
  sorry

end focus_of_hyperbola_l100_100208


namespace multiplication_problem_l100_100026

theorem multiplication_problem :
  250 * 24.98 * 2.498 * 1250 = 19484012.5 := by
  sorry

end multiplication_problem_l100_100026


namespace sector_perimeter_l100_100172

theorem sector_perimeter (r : ℝ) (c : ℝ) (angle_deg : ℝ) (angle_rad := angle_deg * Real.pi / 180) 
  (arc_length := r * angle_rad) (P := arc_length + c)
  (h1 : r = 10) (h2 : c = 10) (h3 : angle_deg = 120) :
  P = 20 * Real.pi / 3 + 10 :=
by
  sorry

end sector_perimeter_l100_100172


namespace repeating_decimal_sum_l100_100347

noncomputable def repeating_decimal_four : ℚ := 0.44444 -- 0.\overline{4}
noncomputable def repeating_decimal_seven : ℚ := 0.77777 -- 0.\overline{7}

-- Proving that the sum of these repeating decimals is equivalent to the fraction 11/9.
theorem repeating_decimal_sum : repeating_decimal_four + repeating_decimal_seven = 11/9 := by
  -- Placeholder to skip the actual proof
  sorry

end repeating_decimal_sum_l100_100347


namespace contradiction_divisible_by_2_l100_100592

open Nat

theorem contradiction_divisible_by_2 (a b : ℕ) (h : (a * b) % 2 = 0) : a % 2 = 0 ∨ b % 2 = 0 :=
by
  sorry

end contradiction_divisible_by_2_l100_100592


namespace triangle_inequality_inequality_l100_100974

-- Define a helper function to describe the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

-- Define the main statement
theorem triangle_inequality_inequality (a b c : ℝ) (h_triangle : triangle_inequality a b c):
  a * (b - c) ^ 2 + b * (c - a) ^ 2 + c * (a - b) ^ 2 + 4 * a * b * c > a ^ 3 + b ^ 3 + c ^ 3 :=
sorry

end triangle_inequality_inequality_l100_100974


namespace emus_count_l100_100524

theorem emus_count (E : ℕ) (heads : ℕ) (legs : ℕ) 
  (h_heads : ∀ e : ℕ, heads = e) 
  (h_legs : ∀ e : ℕ, legs = 2 * e)
  (h_total : heads + legs = 60) : 
  E = 20 :=
by sorry

end emus_count_l100_100524


namespace min_value_of_f_l100_100938

noncomputable def f (x y z : ℝ) : ℝ :=
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

theorem min_value_of_f (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : c * y + b * z = a) (h2 : a * z + c * x = b) (h3 : b * x + a * y = c) :
  ∃ x y z : ℝ, f x y z = 1 / 2 := sorry

end min_value_of_f_l100_100938


namespace expand_expression_l100_100793

theorem expand_expression (x : ℝ) : 
  (x - 3) * (x + 3) * (x^2 + 5) = x^4 - 4 * x^2 - 45 := 
by
  sorry

end expand_expression_l100_100793


namespace exist_triangle_l100_100341

-- Definitions of points and properties required in the conditions
structure Point :=
(x : ℝ) (y : ℝ)

def orthocenter (M : Point) := M 
def centroid (S : Point) := S 
def vertex (C : Point) := C 

-- The problem statement that needs to be proven
theorem exist_triangle (M S C : Point) 
    (h_orthocenter : orthocenter M = M)
    (h_centroid : centroid S = S)
    (h_vertex : vertex C = C) : 
    ∃ (A B : Point), 
        -- A, B, and C form a triangle ABC
        -- S is the centroid of this triangle
        -- M is the orthocenter of this triangle
        -- C is one of the vertices
        true := 
sorry

end exist_triangle_l100_100341


namespace Luca_milk_water_needed_l100_100883

def LucaMilk (flour : ℕ) : ℕ := (flour / 250) * 50
def LucaWater (flour : ℕ) : ℕ := (flour / 250) * 30

theorem Luca_milk_water_needed (flour : ℕ) (h : flour = 1250) : LucaMilk flour = 250 ∧ LucaWater flour = 150 := by
  rw [h]
  sorry

end Luca_milk_water_needed_l100_100883


namespace find_constants_cd_l100_100254

noncomputable def N : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![-2, 4]
]

theorem find_constants_cd :
  ∃ (c d : ℚ), (inverse N = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
               (c = -1/14) ∧ (d = 3/7) :=
by
  sorry

end find_constants_cd_l100_100254


namespace recurring_decimal_to_fraction_correct_l100_100471

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l100_100471


namespace paul_spent_374_43_l100_100980

noncomputable def paul_total_cost_after_discounts : ℝ :=
  let dress_shirts := 4 * 15.00
  let discount_dress_shirts := dress_shirts * 0.20
  let cost_dress_shirts := dress_shirts - discount_dress_shirts
  
  let pants := 2 * 40.00
  let discount_pants := pants * 0.30
  let cost_pants := pants - discount_pants
  
  let suit := 150.00
  
  let sweaters := 2 * 30.00
  
  let ties := 3 * 20.00
  let discount_tie := 20.00 * 0.50
  let cost_ties := 20.00 + (20.00 - discount_tie) + 20.00

  let shoes := 80.00
  let discount_shoes := shoes * 0.25
  let cost_shoes := shoes - discount_shoes

  let total_after_discounts := cost_dress_shirts + cost_pants + suit + sweaters + cost_ties + cost_shoes
  
  let total_after_coupon := total_after_discounts * 0.90
  
  let total_after_rewards := total_after_coupon - (500 * 0.05)
  
  let total_after_tax := total_after_rewards * 1.05
  
  total_after_tax

theorem paul_spent_374_43 :
  paul_total_cost_after_discounts = 374.43 :=
by
  sorry

end paul_spent_374_43_l100_100980


namespace find_principal_l100_100604

theorem find_principal (SI : ℝ) (R : ℝ) (T : ℝ) (hSI : SI = 4025.25) (hR : R = 9) (hT : T = 5) :
    let P := SI / (R * T / 100)
    P = 8950 :=
by
  -- we will put proof steps here
  sorry

end find_principal_l100_100604


namespace student_scores_l100_100018

def weighted_average (math history science geography : ℝ) : ℝ :=
  (math * 0.30) + (history * 0.30) + (science * 0.20) + (geography * 0.20)

theorem student_scores :
  ∀ (math history science geography : ℝ),
    math = 74 →
    history = 81 →
    science = geography + 5 →
    science ≥ 75 →
    weighted_average math history science geography = 80 →
    science = 86.25 ∧ geography = 81.25 :=
by
  intros math history science geography h_math h_history h_science h_min_sci h_avg
  sorry

end student_scores_l100_100018


namespace negation_prop_l100_100304

theorem negation_prop (p : Prop) : 
  (∀ (x : ℝ), x > 2 → x^2 - 1 > 0) → (¬(∀ (x : ℝ), x > 2 → x^2 - 1 > 0) ↔ (∃ (x : ℝ), x > 2 ∧ x^2 - 1 ≤ 0)) :=
by 
  sorry

end negation_prop_l100_100304


namespace number_of_valid_polynomials_l100_100928

noncomputable def count_polynomials_meeting_conditions : ℕ := sorry

theorem number_of_valid_polynomials :
  count_polynomials_meeting_conditions = 7200 :=
sorry

end number_of_valid_polynomials_l100_100928


namespace maximum_area_of_right_triangle_l100_100228

theorem maximum_area_of_right_triangle
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b)
  (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = 2) : 
  ∃ S, S ≤ (3 - 2 * Real.sqrt 2) ∧ S = (1/2) * a * b :=
by
  sorry

end maximum_area_of_right_triangle_l100_100228


namespace cos_triple_angle_l100_100695

variable (θ : ℝ)

theorem cos_triple_angle (h : cos θ = 1 / 3) : cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l100_100695


namespace b_income_percentage_increase_l100_100581

theorem b_income_percentage_increase (A_m B_m C_m : ℕ) (annual_income_A : ℕ)
  (C_income : C_m = 15000)
  (annual_income_A_cond : annual_income_A = 504000)
  (ratio_cond : A_m / B_m = 5 / 2)
  (A_m_cond : A_m = annual_income_A / 12) :
  ((B_m - C_m) * 100 / C_m) = 12 :=
by
  sorry

end b_income_percentage_increase_l100_100581


namespace village_population_rate_l100_100594

theorem village_population_rate (R : ℕ) :
  (76000 - 17 * R = 42000 + 17 * 800) → R = 1200 :=
by
  intro h
  -- The actual proof is omitted.
  sorry

end village_population_rate_l100_100594


namespace determine_B_l100_100819

open Set

-- Define the universal set U and the sets A and B
variable (U A B : Set ℕ)

-- Definitions based on the problem conditions
def U_def : U = A ∪ B := 
  by sorry

def cond1 : (U = {1, 2, 3, 4, 5, 6, 7}) := 
  by sorry

def cond2 : (A ∩ (U \ B) = {2, 4, 6}) := 
  by sorry

-- The main statement
theorem determine_B (h1 : U = {1, 2, 3, 4, 5, 6, 7}) (h2 : A ∩ (U \ B) = {2, 4, 6}) : B = {1, 3, 5, 7} :=
  by sorry

end determine_B_l100_100819


namespace repeating_decimal_eq_l100_100467

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l100_100467


namespace discount_calculation_l100_100441

noncomputable def cost_price : ℝ := 180
noncomputable def markup_percentage : ℝ := 0.4778
noncomputable def profit_percentage : ℝ := 0.20

noncomputable def marked_price (CP : ℝ) (MP_percent : ℝ) : ℝ := CP + (MP_percent * CP)
noncomputable def selling_price (CP : ℝ) (PP_percent : ℝ) : ℝ := CP + (PP_percent * CP)
noncomputable def discount (MP : ℝ) (SP : ℝ) : ℝ := MP - SP

theorem discount_calculation :
  discount (marked_price cost_price markup_percentage) (selling_price cost_price profit_percentage) = 50.004 :=
by
  sorry

end discount_calculation_l100_100441


namespace find_a_l100_100815

theorem find_a (f : ℝ → ℝ) (h1 : ∀ x, f (x + 1) = 3 * x + 2) (h2 : f a = 5) : a = 2 :=
sorry

end find_a_l100_100815


namespace european_math_school_gathering_l100_100579

theorem european_math_school_gathering :
  ∃ n : ℕ, n < 400 ∧ n % 17 = 16 ∧ n % 19 = 12 ∧ n = 288 :=
by
  sorry

end european_math_school_gathering_l100_100579


namespace trajectory_of_moving_point_l100_100807

noncomputable def point (x y : ℝ) := (x, y)

theorem trajectory_of_moving_point :
  ∀ (P : ℝ × ℝ),
  let M := point 2 0,
      N := point (-2) 0
  in
    (Euclidean.dist P M - Euclidean.dist P N = 2) →
    (P.1^2 - P.2^2 / 3 = 1 ∧ P.1 ≤ -1) :=
begin
  sorry
end

end trajectory_of_moving_point_l100_100807


namespace turnip_weight_possible_l100_100000

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l100_100000


namespace cyclic_quadrilaterals_count_l100_100664

theorem cyclic_quadrilaterals_count :
  let is_cyclic_quadrilateral (a b c d : ℕ) : Prop :=
    a + b + c + d = 36 ∧
    a * c + b * d <= (a + c) * (b + d) ∧ -- cyclic quadrilateral inequality
    a + b > c ∧ a + c > b ∧ a + d > b ∧ b + c > d ∧ -- convex quadilateral inequality

  (finset.univ.filter (λ (s : finset ℕ), 
    s.card = 4 ∧ fact (multiset.card s.to_multiset = 36) ∧ is_cyclic_quadrilateral s.to_multiset.sum)).card = 1440 :=
sorry

end cyclic_quadrilaterals_count_l100_100664


namespace g_at_1_l100_100952

variable (g : ℝ → ℝ)

theorem g_at_1 (h : ∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) : g 1 = 18 := by
  sorry

end g_at_1_l100_100952


namespace polynomial_solution_l100_100043

variable (P : ℝ → ℝ → ℝ)

theorem polynomial_solution :
  (∀ x y : ℝ, P (x + y) (x - y) = 2 * P x y) →
  (∃ b c d : ℝ, ∀ x y : ℝ, P x y = b * x^2 + c * x * y + d * y^2) :=
by
  sorry

end polynomial_solution_l100_100043


namespace percentage_difference_l100_100012

theorem percentage_difference (y : ℝ) (h : y ≠ 0) (x z : ℝ) (hx : x = 5 * y) (hz : z = 1.20 * y) :
  ((z - y) / x * 100) = 4 :=
by
  rw [hz, hx]
  simp
  sorry

end percentage_difference_l100_100012


namespace functions_from_M_to_N_l100_100233

def M : Set ℤ := { -1, 1, 2, 3 }
def N : Set ℤ := { 0, 1, 2, 3, 4 }
def f2 (x : ℤ) := x + 1
def f4 (x : ℤ) := (x - 1)^2

theorem functions_from_M_to_N :
  (∀ x ∈ M, f2 x ∈ N) ∧ (∀ x ∈ M, f4 x ∈ N) :=
by
  sorry

end functions_from_M_to_N_l100_100233


namespace common_elements_count_l100_100253

open Finset

def S : Finset ℕ := Finset.range (3005 + 1) |>.image (λ n, 5 * n)
def T : Finset ℕ := Finset.range (3005 + 1) |>.image (λ n, 8 * n)

theorem common_elements_count : (S ∩ T).card = 375 := by
  sorry

end common_elements_count_l100_100253


namespace smallest_natural_number_B_l100_100650

theorem smallest_natural_number_B (A : ℕ) (h : A % 2 = 0 ∧ A % 3 = 0) :
    ∃ B : ℕ, (360 / (A^3 / B) = 5) ∧ B = 3 :=
by
  sorry

end smallest_natural_number_B_l100_100650


namespace gemstones_needed_for_sets_l100_100414

-- Define the number of magnets per earring
def magnets_per_earring : ℕ := 2

-- Define the number of buttons per earring as half the number of magnets
def buttons_per_earring (magnets : ℕ) : ℕ := magnets / 2

-- Define the number of gemstones per earring as three times the number of buttons
def gemstones_per_earring (buttons : ℕ) : ℕ := 3 * buttons

-- Define the number of earrings per set
def earrings_per_set : ℕ := 2

-- Define the number of sets
def sets : ℕ := 4

-- Prove that Rebecca needs 24 gemstones for 4 sets of earrings given the conditions
theorem gemstones_needed_for_sets :
  gemstones_per_earring (buttons_per_earring magnets_per_earring) * earrings_per_set * sets = 24 :=
by
  sorry

end gemstones_needed_for_sets_l100_100414


namespace stuffed_animal_cost_l100_100338

variables 
  (M S A A_single C : ℝ)
  (Coupon_discount : ℝ)
  (Maximum_budget : ℝ)

noncomputable def conditions : Prop :=
  M = 6 ∧
  M = 3 * S ∧
  M = A / 4 ∧
  A_single = A / 2 ∧
  C = A_single / 2 ∧
  C = 2 * S ∧
  Coupon_discount = 0.10 ∧
  Maximum_budget = 30

theorem stuffed_animal_cost (h : conditions M S A A_single C Coupon_discount Maximum_budget) :
  A_single = 12 :=
sorry

end stuffed_animal_cost_l100_100338


namespace find_x_l100_100600

theorem find_x (x : ℝ) (h : 70 + 60 / (x / 3) = 71) : x = 180 :=
sorry

end find_x_l100_100600


namespace circle_equation_bisects_l100_100244

-- Define the given conditions
def circle1_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 8)^2 = 1
def circle2_eq (x y : ℝ) : Prop := (x - 6)^2 + (y + 6)^2 = 9

-- Define the goal equation
def circleC_eq (x y : ℝ) : Prop := x^2 + y^2 = 81

-- The statement of the problem
theorem circle_equation_bisects (a r : ℝ) (h1 : ∀ x y, circle1_eq x y → circleC_eq x y) (h2 : ∀ x y, circle2_eq x y → circleC_eq x y):
  circleC_eq (a * r) 0 := sorry

end circle_equation_bisects_l100_100244


namespace find_principal_l100_100174

theorem find_principal
  (SI : ℝ)
  (R : ℝ)
  (T : ℝ)
  (h_SI : SI = 4025.25)
  (h_R : R = 0.09)
  (h_T : T = 5) : 
  (SI / (R * T / 100)) = 8950 :=
by
  rw [h_SI, h_R, h_T]
  sorry

end find_principal_l100_100174


namespace volume_of_cylinder_cut_l100_100359

open Real

noncomputable def cylinder_cut_volume (R α : ℝ) : ℝ :=
  (2 / 3) * R^3 * tan α

theorem volume_of_cylinder_cut (R α : ℝ) :
  cylinder_cut_volume R α = (2 / 3) * R^3 * tan α :=
by
  sorry

end volume_of_cylinder_cut_l100_100359


namespace product_of_integers_l100_100143

theorem product_of_integers (x y : ℕ) (h1 : x + y = 72) (h2 : x - y = 18) : x * y = 1215 := 
sorry

end product_of_integers_l100_100143


namespace recurring_to_fraction_l100_100475

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l100_100475


namespace smallest_number_divisible_by_conditions_l100_100150

theorem smallest_number_divisible_by_conditions (N : ℕ) (X : ℕ) (H1 : (N - 12) % 8 = 0) (H2 : (N - 12) % 12 = 0)
(H3 : (N - 12) % X = 0) (H4 : (N - 12) % 24 = 0) (H5 : (N - 12) / 24 = 276) : N = 6636 :=
by
  sorry

end smallest_number_divisible_by_conditions_l100_100150


namespace solve_for_x_l100_100077

theorem solve_for_x (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) : x = 37 := by
  sorry

end solve_for_x_l100_100077


namespace solve_inequality_l100_100941

def f (a x : ℝ) : ℝ := a * x * (x + 1) + 1

theorem solve_inequality (a x : ℝ) (h : f a x < 0) : x < (1 / a) ∨ (x > 1 ∧ a ≠ 0) := by
  sorry

end solve_inequality_l100_100941


namespace arithmetic_sequence_sum_zero_l100_100374

theorem arithmetic_sequence_sum_zero {a1 d n : ℤ} 
(h1 : a1 = 35) 
(h2 : d = -2) 
(h3 : (n * (2 * a1 + (n - 1) * d)) / 2 = 0) : 
n = 36 :=
by sorry

end arithmetic_sequence_sum_zero_l100_100374


namespace right_triangle_set_D_l100_100751

theorem right_triangle_set_D : (5^2 + 12^2 = 13^2) ∧ 
  ((3^2 + 3^2 ≠ 5^2) ∧ (6^2 + 8^2 ≠ 9^2) ∧ (4^2 + 5^2 ≠ 6^2)) :=
by
  sorry

end right_triangle_set_D_l100_100751


namespace password_probability_l100_100779

theorem password_probability : 
  (5/10) * (51/52) * (9/10) = 459 / 1040 := by
  sorry

end password_probability_l100_100779


namespace find_n_l100_100659

def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Given conditions
variable (n : ℕ)
variable (coef : ℕ)
variable (h : coef = binomial_coeff n 2 * 9)

-- Proof target
theorem find_n (h : coef = 54) : n = 4 :=
  sorry

end find_n_l100_100659


namespace contradiction_example_l100_100451

theorem contradiction_example (a b c d : ℝ) 
  (h1 : a + b = 1) 
  (h2 : c + d = 1) 
  (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) → 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 :=
by
  sorry

end contradiction_example_l100_100451


namespace problem_part_1_problem_part_2_l100_100063

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x - 1

theorem problem_part_1 (m n : ℝ) :
  (∀ x, f x m < 0 ↔ -2 < x ∧ x < n) → m = 5 / 2 ∧ n = 1 / 2 :=
sorry

theorem problem_part_2 (m : ℝ) :
  (∀ x, m ≤ x ∧ x ≤ m + 1 → f x m < 0) → m ∈ Set.Ioo (-Real.sqrt (2) / 2) 0 :=
sorry

end problem_part_1_problem_part_2_l100_100063


namespace pentagon_area_l100_100771

theorem pentagon_area (a b c d e : ℝ) (r s : ℝ) 
    (h1 : a = 14 ∨ a = 21 ∨ a = 22 ∨ a = 28 ∨ a = 35)
    (h2 : b = 14 ∨ b = 21 ∨ b = 22 ∨ b = 28 ∨ b = 35)
    (h3 : c = 14 ∨ c = 21 ∨ c = 22 ∨ c = 28 ∨ c = 35)
    (h4 : d = 14 ∨ d = 21 ∨ d = 22 ∨ d = 28 ∨ d = 35)
    (h5 : e = 14 ∨ e = 21 ∨ e = 22 ∨ e = 28 ∨ e = 35)
    (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
          b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
          c ≠ d ∧ c ≠ e ∧ 
          d ≠ e)
    (h6 : r^2 + s^2 = e^2)
    (h7 : r = b - d)
    (h8 : s = c - a) 
    : (a + b + c + d + e) * 1 / 2 + (b * c - 1 / 2 * r * s) = 1421 := 
begin
  sorry
end

end pentagon_area_l100_100771


namespace tan_eq_one_over_three_l100_100652

theorem tan_eq_one_over_three (x : ℝ) (h1 : x ∈ Set.Ioo 0 Real.pi)
  (h2 : Real.cos (2 * x - (Real.pi / 2)) = Real.sin x ^ 2) :
  Real.tan (x - Real.pi / 4) = 1 / 3 := by
  sorry

end tan_eq_one_over_three_l100_100652


namespace cone_new_height_eq_sqrt_85_l100_100612

/-- A cone has a uniform circular base of radius 6 feet and a slant height of 13 feet.
    After the side breaks, the slant height reduces by 2 feet, making the new slant height 11 feet.
    We need to determine the new height from the base to the tip of the cone, and prove it is sqrt(85). -/
theorem cone_new_height_eq_sqrt_85 :
  let r : ℝ := 6
  let l : ℝ := 13
  let l' : ℝ := 11
  let h : ℝ := Real.sqrt (13^2 - 6^2)
  let H : ℝ := Real.sqrt (11^2 - 6^2)
  H = Real.sqrt 85 :=
by
  sorry


end cone_new_height_eq_sqrt_85_l100_100612


namespace angle_in_second_quadrant_l100_100736

def inSecondQuadrant (θ : ℤ) : Prop :=
  90 < θ ∧ θ < 180

theorem angle_in_second_quadrant :
  ∃ k : ℤ, inSecondQuadrant (-2015 + 360 * k) :=
by {
  sorry
}

end angle_in_second_quadrant_l100_100736


namespace largest_of_three_consecutive_odds_l100_100879

theorem largest_of_three_consecutive_odds (n : ℤ) (h_sum : n + (n + 2) + (n + 4) = -147) : n + 4 = -47 :=
by {
  -- Proof steps here, but we're skipping for this exercise
  sorry
}

end largest_of_three_consecutive_odds_l100_100879


namespace transformation_result_l100_100191

theorem transformation_result (a b : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (a, b))
  (h2 : ∃ Q : ℝ × ℝ, Q = (b, a))
  (h3 : ∃ R : ℝ × ℝ, R = (2 - b, 10 - a))
  (h4 : (2 - b, 10 - a) = (-8, 2)) : 
  a - b = -2 := 
by 
  sorry

end transformation_result_l100_100191


namespace g_neg501_l100_100862

noncomputable def g : ℝ → ℝ := sorry

axiom g_eq (x y : ℝ) : g (x * y) + 2 * x = x * g y + g x

axiom g_neg1 : g (-1) = 7

theorem g_neg501 : g (-501) = 507 :=
by
  sorry

end g_neg501_l100_100862


namespace exponent_problem_proof_l100_100785

theorem exponent_problem_proof :
  3 * 3^4 - 27^60 / 27^58 = -486 :=
by
  sorry

end exponent_problem_proof_l100_100785


namespace quadratic_inequality_condition_l100_100503

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) → 0 ≤ a ∧ a < 4 :=
sorry

end quadratic_inequality_condition_l100_100503


namespace factorization_check_l100_100602

theorem factorization_check 
  (A : 4 - x^2 + 3 * x ≠ (2 - x) * (2 + x) + 3)
  (B : -x^2 + 3 * x + 4 ≠ -(x + 4) * (x - 1))
  (D : x^2 * y - x * y + x^3 * y ≠ x * (x * y - y + x^2 * y)) :
  1 - 2 * x + x^2 = (1 - x) ^ 2 :=
by
  sorry

end factorization_check_l100_100602


namespace power_calc_l100_100071

noncomputable def n := 2 ^ 0.3
noncomputable def b := 13.333333333333332

theorem power_calc : n ^ b = 16 := by
  sorry

end power_calc_l100_100071


namespace abs_neg_three_l100_100277

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l100_100277


namespace marble_color_197th_l100_100505

theorem marble_color_197th (n : ℕ) (total_marbles : ℕ) (marble_color : ℕ → ℕ)
                          (h_total : total_marbles = 240)
                          (h_pattern : ∀ k, marble_color (k + 15) = marble_color k)
                          (h_colors : ∀ i, (0 ≤ i ∧ i < 15) →
                                   (marble_color i = if i < 6 then 1
                                   else if i < 11 then 2
                                   else if i < 15 then 3
                                   else 0)) :
  marble_color 197 = 1 := sorry

end marble_color_197th_l100_100505


namespace abs_neg_three_l100_100270

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l100_100270


namespace total_amount_spent_l100_100921

def price_of_brand_X_pen : ℝ := 4.00
def price_of_brand_Y_pen : ℝ := 2.20
def total_pens_purchased : ℝ := 12
def brand_X_pens_purchased : ℝ := 6

theorem total_amount_spent :
  let brand_X_cost := brand_X_pens_purchased * price_of_brand_X_pen
  let brand_Y_pens_purchased := total_pens_purchased - brand_X_pens_purchased
  let brand_Y_cost := brand_Y_pens_purchased * price_of_brand_Y_pen
  brand_X_cost + brand_Y_cost = 37.20 :=
by
  sorry

end total_amount_spent_l100_100921


namespace ratio_equality_l100_100390

variable (a b : ℝ)

theorem ratio_equality (h : a / b = 4 / 3) : (3 * a + 2 * b) / (3 * a - 2 * b) = 3 :=
by
sorry

end ratio_equality_l100_100390


namespace total_amount_paid_l100_100968

theorem total_amount_paid (monthly_payment_1 monthly_payment_2 : ℕ) (years_1 years_2 : ℕ)
  (monthly_payment_1_eq : monthly_payment_1 = 300)
  (monthly_payment_2_eq : monthly_payment_2 = 350)
  (years_1_eq : years_1 = 3)
  (years_2_eq : years_2 = 2) :
  let annual_payment_1 := monthly_payment_1 * 12
  let annual_payment_2 := monthly_payment_2 * 12
  let total_1 := annual_payment_1 * years_1
  let total_2 := annual_payment_2 * years_2
  total_1 + total_2 = 19200 :=
by
  sorry

end total_amount_paid_l100_100968


namespace solve_x_in_equation_l100_100982

theorem solve_x_in_equation : ∃ (x : ℤ), 24 - 4 * 2 = 3 + x ∧ x = 13 :=
by
  use 13
  sorry

end solve_x_in_equation_l100_100982


namespace expand_binomial_square_l100_100783

variables (x : ℝ)

theorem expand_binomial_square (x : ℝ) : (2 - x) ^ 2 = 4 - 4 * x + x ^ 2 := 
sorry

end expand_binomial_square_l100_100783


namespace standard_deviation_upper_bound_l100_100876

theorem standard_deviation_upper_bound (Mean StdDev : ℝ) (h : Mean = 54) (h2 : 54 - 3 * StdDev > 47) : StdDev < 2.33 :=
by
  sorry

end standard_deviation_upper_bound_l100_100876


namespace total_cats_and_kittens_received_l100_100448

theorem total_cats_and_kittens_received (total_adult_cats : ℕ) (percentage_female : ℕ) (fraction_with_kittens : ℚ) (kittens_per_litter : ℕ) 
  (h1 : total_adult_cats = 100) (h2 : percentage_female = 40) (h3 : fraction_with_kittens = 2 / 3) (h4 : kittens_per_litter = 3) :
  total_adult_cats + ((percentage_female * total_adult_cats / 100) * (fraction_with_kittens * total_adult_cats * kittens_per_litter) / 100) = 181 := by
  sorry

end total_cats_and_kittens_received_l100_100448


namespace repeating_decimal_to_fraction_l100_100481

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l100_100481


namespace painted_cells_l100_100700

open Int

theorem painted_cells : ∀ (m n : ℕ), (m = 20210) → (n = 1505) →
  let sub_rectangles := 215
  let cells_per_diagonal := 100
  let total_cells := sub_rectangles * cells_per_diagonal
  let total_painted_cells := 2 * total_cells
  let overlap_cells := sub_rectangles
  let unique_painted_cells := total_painted_cells - overlap_cells
  unique_painted_cells = 42785 := sorry

end painted_cells_l100_100700


namespace fraction_cows_sold_is_one_fourth_l100_100741

def num_cows : ℕ := 184
def num_dogs (C : ℕ) : ℕ := C / 2
def remaining_animals : ℕ := 161
def fraction_dogs_sold : ℚ := 3 / 4
def fraction_cows_sold (C remaining_cows : ℕ) : ℚ := (C - remaining_cows) / C

theorem fraction_cows_sold_is_one_fourth :
  ∀ (C remaining_dogs remaining_cows: ℕ),
    C = 184 →
    remaining_animals = 161 →
    remaining_dogs = (1 - fraction_dogs_sold) * num_dogs C →
    remaining_cows = remaining_animals - remaining_dogs →
    fraction_cows_sold C remaining_cows = 1 / 4 :=
by sorry

end fraction_cows_sold_is_one_fourth_l100_100741


namespace cos_fourth_minus_sin_fourth_l100_100221

theorem cos_fourth_minus_sin_fourth (α : ℝ) (h : Real.sin α = (Real.sqrt 5) / 5) :
  Real.cos α ^ 4 - Real.sin α ^ 4 = 3 / 5 := 
sorry

end cos_fourth_minus_sin_fourth_l100_100221


namespace angle_XCY_less_than_60_degrees_l100_100082

open EuclideanGeometry

/-- In a triangle ABC, AB is the shortest side.
Points X and Y are given on the circumcircle of △ABC
such that CX = AX + BX and CY = AY + BY.
Prove that ∠XCY < 60° . -/
theorem angle_XCY_less_than_60_degrees
  {A B C X Y : Point}
  (hABC: Triangle ABC)
  (h_short: dist A B < dist A C ∧ dist A B < dist B C)
  (hX_circum: OnCircumcircle X A B C)
  (hY_circum: OnCircumcircle Y A B C)
  (hCX_eq: dist C X = dist A X + dist B X)
  (hCY_eq: dist C Y = dist A Y + dist B Y) :
  ∠ X C Y < 60 :=
sorry

end angle_XCY_less_than_60_degrees_l100_100082


namespace round_trip_time_correct_l100_100728

variables (river_current_speed boat_speed_still_water distance_upstream_distance : ℕ)

def upstream_speed := boat_speed_still_water - river_current_speed
def downstream_speed := boat_speed_still_water + river_current_speed

def time_upstream := distance_upstream_distance / upstream_speed
def time_downstream := distance_upstream_distance / downstream_speed

def round_trip_time := time_upstream + time_downstream

theorem round_trip_time_correct :
  river_current_speed = 10 →
  boat_speed_still_water = 50 →
  distance_upstream_distance = 120 →
  round_trip_time river_current_speed boat_speed_still_water distance_upstream_distance = 5 :=
by
  intros rc bs d
  sorry

end round_trip_time_correct_l100_100728


namespace range_of_a_l100_100867

variable (a : ℝ)
def f (x : ℝ) := x^2 + 2 * (a - 1) * x + 2
def f_deriv (x : ℝ) := 2 * x + 2 * (a - 1)

theorem range_of_a (h : ∀ x ≥ -4, f_deriv a x ≥ 0) : a ≥ 5 :=
sorry

end range_of_a_l100_100867


namespace cindy_marbles_problem_l100_100848

theorem cindy_marbles_problem
  (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 500) (h2 : friends = 4) (h3 : marbles_per_friend = 80) :
  4 * (initial_marbles - (marbles_per_friend * friends)) = 720 :=
by
  sorry

end cindy_marbles_problem_l100_100848


namespace biff_break_even_hours_l100_100628

def totalSpent (ticket drinks snacks headphones : ℕ) : ℕ :=
  ticket + drinks + snacks + headphones

def netEarningsPerHour (earningsCost wifiCost : ℕ) : ℕ :=
  earningsCost - wifiCost

def hoursToBreakEven (totalSpent netEarnings : ℕ) : ℕ :=
  totalSpent / netEarnings

-- given conditions
def given_ticket : ℕ := 11
def given_drinks : ℕ := 3
def given_snacks : ℕ := 16
def given_headphones : ℕ := 16
def given_earningsPerHour : ℕ := 12
def given_wifiCostPerHour : ℕ := 2

theorem biff_break_even_hours :
  hoursToBreakEven (totalSpent given_ticket given_drinks given_snacks given_headphones) 
                   (netEarningsPerHour given_earningsPerHour given_wifiCostPerHour) = 3 :=
by
  sorry

end biff_break_even_hours_l100_100628


namespace diff_of_squares_l100_100958

theorem diff_of_squares (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 10) : x^2 - y^2 = 50 := by
  sorry

end diff_of_squares_l100_100958


namespace mistaken_divisor_l100_100832

theorem mistaken_divisor (x : ℕ) (h : 49 * x = 28 * 21) : x = 12 :=
sorry

end mistaken_divisor_l100_100832


namespace faster_train_speed_l100_100308

theorem faster_train_speed (V_s : ℝ) (t : ℝ) (l : ℝ) (V_f : ℝ) : 
  V_s = 36 → t = 20 → l = 200 → V_f = V_s + (l / t) * 3.6 → V_f = 72 
  := by
    intros _ _ _ _
    sorry

end faster_train_speed_l100_100308


namespace four_times_remaining_marbles_l100_100852

theorem four_times_remaining_marbles (initial total_given : ℕ) (remaining : ℕ := initial - total_given) :
  initial = 500 → total_given = 4 * 80 → 4 * remaining = 720 := by sorry

end four_times_remaining_marbles_l100_100852


namespace cos_3theta_value_l100_100689

open Real

noncomputable def cos_3theta (theta : ℝ) : ℝ := 4 * (cos theta)^3 - 3 * (cos theta)

theorem cos_3theta_value (theta : ℝ) (h : cos theta = 1 / 3) : cos_3theta theta = - 23 / 27 :=
by
  sorry

end cos_3theta_value_l100_100689


namespace mutually_exclusive_not_opposite_l100_100320

namespace event_theory

-- Definition to represent the student group
structure Group where
  boys : ℕ
  girls : ℕ

def student_group : Group := {boys := 3, girls := 2}

-- Definition of events
inductive Event
| AtLeastOneBoyAndOneGirl
| ExactlyOneBoyExactlyTwoBoys
| AtLeastOneBoyAllGirls
| AtMostOneBoyAllGirls

open Event

-- Conditions provided in the problem
def condition (grp : Group) : Prop :=
  grp.boys = 3 ∧ grp.girls = 2

-- The main statement to prove in Lean
theorem mutually_exclusive_not_opposite :
  condition student_group →
  ∃ e₁ e₂ : Event, e₁ = ExactlyOneBoyExactlyTwoBoys ∧ e₂ = ExactlyOneBoyExactlyTwoBoys ∧ (
    (e₁ ≠ e₂) ∧ (¬ (e₁ = e₂ ∧ e₁ = ExactlyOneBoyExactlyTwoBoys))
  ) :=
by
  sorry

end event_theory

end mutually_exclusive_not_opposite_l100_100320


namespace max_abs_sum_on_ellipse_l100_100686

theorem max_abs_sum_on_ellipse :
  ∀ (x y : ℝ), 4 * x^2 + y^2 = 4 -> |x| + |y| ≤ (3 * Real.sqrt 2) / Real.sqrt 5 :=
by
  intro x y h
  sorry

end max_abs_sum_on_ellipse_l100_100686


namespace probability_factor_120_less_9_l100_100149

theorem probability_factor_120_less_9 : 
  ∀ n : ℕ, n = 120 → (∃ p : ℚ, p = 7 / 16 ∧ (∃ factors_less_9 : ℕ, factors_less_9 < 16 ∧ factors_less_9 = 7)) := 
by 
  sorry

end probability_factor_120_less_9_l100_100149


namespace rectangle_original_length_doubles_area_l100_100809

-- Let L and W denote the length and width of a rectangle respectively
-- Given the condition: (L + 2)W = 2LW
-- We need to prove that L = 2

theorem rectangle_original_length_doubles_area (L W : ℝ) (h : (L + 2) * W = 2 * L * W) : L = 2 :=
by 
  sorry

end rectangle_original_length_doubles_area_l100_100809


namespace max_area_cross_section_rect_prism_l100_100906

/-- The maximum area of the cross-sectional cut of a rectangular prism 
having its vertical edges parallel to the z-axis, with cross-section 
rectangle of sides 8 and 12, whose bottom side lies in the xy-plane 
centered at the origin (0,0,0), cut by the plane 3x + 5y - 2z = 30 
is approximately 118.34. --/
theorem max_area_cross_section_rect_prism :
  ∃ A : ℝ, abs (A - 118.34) < 0.01 :=
sorry

end max_area_cross_section_rect_prism_l100_100906


namespace div_poly_iff_l100_100981

-- Definitions from conditions
def P (x : ℂ) (n : ℕ) := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℂ) := x^4 + x^3 + x^2 + x + 1

-- The main theorem stating the problem
theorem div_poly_iff (n : ℕ) : 
  ∀ x : ℂ, (P x n) ∣ (Q x) ↔ n % 5 ≠ 0 :=
by sorry

end div_poly_iff_l100_100981


namespace max_min_values_f_l100_100800

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem max_min_values_f :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 2) ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2) ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ Real.sqrt 3) ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = Real.sqrt 3) :=
by
  sorry

end max_min_values_f_l100_100800


namespace sum_of_sequence_l100_100015

def sequence_t (n : ℕ) : ℚ :=
  if n % 2 = 1 then 1 / 7^n else 2 / 7^n

theorem sum_of_sequence :
  (∑' n:ℕ, sequence_t (n + 1)) = 3 / 16 :=
by
  sorry

end sum_of_sequence_l100_100015


namespace number_of_ordered_pairs_l100_100636

-- Define the predicate that defines the condition for the ordered pairs (m, n)
def satisfies_condition (m n : ℕ) : Prop :=
  6 % m = 0 ∧ 3 % n = 0 ∧ 6 / m + 3 / n = 1

-- Define the main theorem for the problem statement
theorem number_of_ordered_pairs : 
  (∃ count : ℕ, count = 6 ∧ 
  (∀ m n : ℕ, satisfies_condition m n → m > 0 ∧ n > 0)) :=
by {
 sorry -- Placeholder for the actual proof
}

end number_of_ordered_pairs_l100_100636


namespace nina_money_l100_100756

theorem nina_money :
  ∃ (m C : ℝ), 
    m = 6 * C ∧ 
    m = 8 * (C - 1) ∧ 
    m = 24 :=
by
  sorry

end nina_money_l100_100756


namespace boiling_point_C_l100_100309

-- Water boils at 212 °F
def water_boiling_point_F : ℝ := 212
-- Ice melts at 32 °F
def ice_melting_point_F : ℝ := 32
-- Ice melts at 0 °C
def ice_melting_point_C : ℝ := 0
-- The temperature of a pot of water in °C
def pot_water_temp_C : ℝ := 40
-- The temperature of the pot of water in °F
def pot_water_temp_F : ℝ := 104

-- The boiling point of water in Celsius is 100 °C.
theorem boiling_point_C : water_boiling_point_F = 212 ∧ ice_melting_point_F = 32 ∧ ice_melting_point_C = 0 ∧ pot_water_temp_C = 40 ∧ pot_water_temp_F = 104 → exists bp_C : ℝ, bp_C = 100 :=
by
  sorry

end boiling_point_C_l100_100309


namespace michael_peach_pies_l100_100407

/--
Michael ran a bakeshop and had to fill an order for some peach pies, 4 apple pies and 3 blueberry pies.
Each pie recipe called for 3 pounds of fruit each. At the market, produce was on sale for $1.00 per pound for both blueberries and apples.
The peaches each cost $2.00 per pound. Michael spent $51 at the market buying the fruit for his pie order.
Prove that Michael had to make 5 peach pies.
-/
theorem michael_peach_pies :
  let apple_pies := 4
  let blueberry_pies := 3
  let peach_pie_cost_per_pound := 2
  let apple_blueberry_cost_per_pound := 1
  let pounds_per_pie := 3
  let total_spent := 51
  (total_spent - ((apple_pies + blueberry_pies) * pounds_per_pie * apple_blueberry_cost_per_pound)) 
  / (pounds_per_pie * peach_pie_cost_per_pound) = 5 :=
by
  let apple_pies := 4
  let blueberry_pies := 3
  let peach_pie_cost_per_pound := 2
  let apple_blueberry_cost_per_pound := 1
  let pounds_per_pie := 3
  let total_spent := 51
  have H1 : (total_spent - ((apple_pies + blueberry_pies) * pounds_per_pie * apple_blueberry_cost_per_pound)) 
             / (pounds_per_pie * peach_pie_cost_per_pound) = 5 := sorry
  exact H1

end michael_peach_pies_l100_100407


namespace sufficient_but_not_necessary_l100_100584

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1) → (x < -1 ∨ x > 1) ∧ ¬(∀ y : ℝ, (x < -1 ∨ y > 1) → (y < -1)) :=
by
  -- This means we would prove that if x < -1, then x < -1 ∨ x > 1 holds (sufficient),
  -- and show that there is a case (x > 1) where x < -1 is not necessary for x < -1 ∨ x > 1. 
  sorry

end sufficient_but_not_necessary_l100_100584


namespace rectangle_width_decrease_percent_l100_100133

theorem rectangle_width_decrease_percent (L W : ℝ) (h : L * W = L * W) :
  let L_new := 1.3 * L
  let W_new := W / 1.3 
  let percent_decrease := (1 - (W_new / W)) * 100
  percent_decrease = 23.08 :=
sorry

end rectangle_width_decrease_percent_l100_100133


namespace john_new_salary_after_raise_l100_100562

theorem john_new_salary_after_raise (original_salary : ℝ) (percentage_increase : ℝ) (h1 : original_salary = 60) (h2 : percentage_increase = 0.8333333333333334) : 
  original_salary * (1 + percentage_increase) = 110 := 
sorry

end john_new_salary_after_raise_l100_100562


namespace line_equation_l100_100995

theorem line_equation (x y : ℝ) (h : ∀ x : ℝ, (x - 2) * 1 = y) : x - y - 2 = 0 :=
sorry

end line_equation_l100_100995


namespace max_value_y_l100_100389

variable (x : ℝ)
def y : ℝ := -3 * x^2 + 6

theorem max_value_y : ∃ M, ∀ x : ℝ, y x ≤ M ∧ (∀ x : ℝ, y x = M → x = 0) :=
by
  use 6
  sorry

end max_value_y_l100_100389


namespace repeating_decimal_fraction_l100_100460

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l100_100460


namespace Ben_ate_25_percent_of_cake_l100_100265

theorem Ben_ate_25_percent_of_cake (R B : ℕ) (h_ratio : R / B = 3 / 1) : B / (R + B) * 100 = 25 := by
  sorry

end Ben_ate_25_percent_of_cake_l100_100265


namespace find_x_such_that_l100_100042

theorem find_x_such_that {x : ℝ} (h : ⌈x⌉ * x + 15 = 210) : x = 195 / 14 :=
by
  sorry

end find_x_such_that_l100_100042


namespace find_a_plus_b_l100_100103

def star (a b : ℕ) : ℕ := a^b + a + b

theorem find_a_plus_b (a b : ℕ) (h2a : 2 ≤ a) (h2b : 2 ≤ b) (h_ab : star a b = 20) :
  a + b = 6 :=
sorry

end find_a_plus_b_l100_100103


namespace linear_dependence_condition_l100_100140

theorem linear_dependence_condition (k : ℝ) :
  (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * 1 + b * 4 = 0) ∧ (a * 2 + b * k = 0) ∧ (a * 1 + b * 2 = 0)) ↔ k = 8 := 
by sorry

end linear_dependence_condition_l100_100140


namespace correct_total_cost_correct_remaining_donuts_l100_100124

-- Conditions
def budget : ℝ := 50
def cost_per_box : ℝ := 12
def discount_percentage : ℝ := 0.10
def number_of_boxes_bought : ℕ := 4
def donuts_per_box : ℕ := 12
def boxes_given_away : ℕ := 1
def additional_donuts_given_away : ℕ := 6

-- Calculations based on conditions
def total_cost_before_discount : ℝ := number_of_boxes_bought * cost_per_box
def discount_amount : ℝ := discount_percentage * total_cost_before_discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

def total_donuts : ℕ := number_of_boxes_bought * donuts_per_box
def total_donuts_given_away : ℕ := (boxes_given_away * donuts_per_box) + additional_donuts_given_away
def remaining_donuts : ℕ := total_donuts - total_donuts_given_away

-- Theorems to prove
theorem correct_total_cost : total_cost_after_discount = 43.20 := by
  -- proof here
  sorry

theorem correct_remaining_donuts : remaining_donuts = 30 := by
  -- proof here
  sorry

end correct_total_cost_correct_remaining_donuts_l100_100124


namespace smallest_square_length_proof_l100_100598

-- Define square side length required properties
noncomputable def smallest_square_side_length (rect_w rect_h min_side : ℝ) : ℝ :=
  if h : min_side^2 % (rect_w * rect_h) = 0 then min_side 
  else if h : (min_side + 1)^2 % (rect_w * rect_h) = 0 then min_side + 1
  else if h : (min_side + 2)^2 % (rect_w * rect_h) = 0 then min_side + 2
  else if h : (min_side + 3)^2 % (rect_w * rect_h) = 0 then min_side + 3
  else if h : (min_side + 4)^2 % (rect_w * rect_h) = 0 then min_side + 4
  else if h : (min_side + 5)^2 % (rect_w * rect_h) = 0 then min_side + 5
  else if h : (min_side + 6)^2 % (rect_w * rect_h) = 0 then min_side + 6
  else if h : (min_side + 7)^2 % (rect_w * rect_h) = 0 then min_side + 7
  else if h : (min_side + 8)^2 % (rect_w * rect_h) = 0 then min_side + 8
  else if h : (min_side + 9)^2 % (rect_w * rect_h) = 0 then min_side + 9
  else min_side + 2 -- ensuring it can't be less than min_side

-- State the theorem
theorem smallest_square_length_proof : smallest_square_side_length 2 3 10 = 12 :=
by 
  unfold smallest_square_side_length
  norm_num
  sorry

end smallest_square_length_proof_l100_100598


namespace repeating_decimal_fraction_l100_100459

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l100_100459


namespace Apollonius_circle_symmetry_l100_100089

theorem Apollonius_circle_symmetry (a : ℝ) (h : a > 1): 
  let F1 := (-1, 0)
  let F2 := (1, 0)
  let locus_C := {P : ℝ × ℝ | ∃ x y, P = (x, y) ∧ (Real.sqrt ((x + 1)^2 + y^2) = a * Real.sqrt ((x - 1)^2 + y^2))}
  let symmetric_y := ∀ (P : ℝ × ℝ), P ∈ locus_C → (P.1, -P.2) ∈ locus_C
  symmetric_y := sorry

end Apollonius_circle_symmetry_l100_100089


namespace correct_subsidy_equation_l100_100427

-- Define the necessary variables and conditions
def sales_price (x : ℝ) := x  -- sales price of the mobile phone in yuan
def subsidy_rate : ℝ := 0.13  -- 13% subsidy rate
def number_of_phones : ℝ := 20  -- 20 units sold
def total_subsidy : ℝ := 2340  -- total subsidy provided

-- Lean theorem statement to prove the correct equation
theorem correct_subsidy_equation (x : ℝ) :
  number_of_phones * x * subsidy_rate = total_subsidy :=
by
  sorry -- proof to be completed

end correct_subsidy_equation_l100_100427


namespace abs_neg_three_l100_100280

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l100_100280


namespace neg_prop_l100_100137

-- Definition of the proposition to be negated
def prop (x : ℝ) : Prop := x^2 + 2 * x + 5 = 0

-- Negation of the proposition
theorem neg_prop : ¬ (∃ x : ℝ, prop x) ↔ ∀ x : ℝ, ¬ prop x :=
by
  sorry

end neg_prop_l100_100137


namespace repeating_decimal_fraction_l100_100458

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l100_100458


namespace num_prime_pairs_sum_50_l100_100683

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def pairs_sum_50 (pair : ℕ × ℕ) : Prop :=
  pair.1 + pair.2 = 50 ∧ is_prime pair.1 ∧ is_prime pair.2 ∧ pair.1 ≤ pair.2

theorem num_prime_pairs_sum_50 : finset.card (finset.filter pairs_sum_50 ({(a, b) | ∃ (a b : ℕ), a ≤ 25 ∧ b ≤ 25}.to_finset)) = 4 :=
  sorry

end num_prime_pairs_sum_50_l100_100683


namespace probability_uniform_same_color_l100_100510

noncomputable def probability_same_color (choices : List String) (athleteA: ℕ) (athleteB: ℕ) : ℚ :=
  if choices.length = 3 ∧ athleteA ∈ [0,1,2] ∧ athleteB ∈ [0,1,2] then
    1 / 3
  else
    0

theorem probability_uniform_same_color :
  probability_same_color ["red", "white", "blue"] 0 1 = 1 / 3 :=
by
  sorry

end probability_uniform_same_color_l100_100510


namespace purchasing_methods_count_l100_100326

def material_cost : ℕ := 40
def instrument_cost : ℕ := 60
def budget : ℕ := 400
def min_materials : ℕ := 4
def min_instruments : ℕ := 2

theorem purchasing_methods_count : 
  (∃ (n_m m : ℕ), 
    n_m ≥ min_materials ∧ m ≥ min_instruments ∧ 
    n_m * material_cost + m * instrument_cost ≤ budget) → 
  (∃ (count : ℕ), count = 7) :=
by 
  sorry

end purchasing_methods_count_l100_100326


namespace jordan_time_to_run_7_miles_l100_100400

def time_taken (distance time_per_unit : ℝ) : ℝ :=
  distance * time_per_unit

theorem jordan_time_to_run_7_miles :
  ∀ (t_S d_S d_J : ℝ), t_S = 36 → d_S = 6 → d_J = 4 → time_taken 7 ((t_S / 2) / d_J) = 31.5 :=
by
  intros t_S d_S d_J h_t_S h_d_S h_d_J
  -- skipping the proof
  sorry

end jordan_time_to_run_7_miles_l100_100400


namespace vertex_of_parabola_l100_100300

-- Define the statement of the problem
theorem vertex_of_parabola :
  ∀ (a h k : ℝ), (∀ x : ℝ, 3 * (x - 5) ^ 2 + 4 = a * (x - h) ^ 2 + k) → (h, k) = (5, 4) :=
by
  sorry

end vertex_of_parabola_l100_100300


namespace simplify_expr1_simplify_expr2_simplify_expr3_l100_100722

theorem simplify_expr1 (y : ℤ) (hy : y = 2) : -3 * y^2 - 6 * y + 2 * y^2 + 5 * y = -6 := 
by sorry

theorem simplify_expr2 (a : ℤ) (ha : a = -2) : 15 * a^2 * (-4 * a^2 + (6 * a - a^2) - 3 * a) = -1560 :=
by sorry

theorem simplify_expr3 (x y : ℤ) (h1 : x * y = 2) (h2 : x + y = 3) : (3 * x * y + 10 * y) + (5 * x - (2 * x * y + 2 * y - 3 * x)) = 26 :=
by sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l100_100722


namespace length_of_bridge_l100_100732

def length_of_train : ℝ := 135  -- Length of the train in meters
def speed_of_train_km_per_hr : ℝ := 45  -- Speed of the train in km/hr
def speed_of_train_m_per_s : ℝ := 12.5  -- Speed of the train in m/s
def time_to_cross_bridge : ℝ := 30  -- Time to cross the bridge in seconds
def distance_covered : ℝ := speed_of_train_m_per_s * time_to_cross_bridge  -- Total distance covered

theorem length_of_bridge :
  distance_covered - length_of_train = 240 :=
by
  sorry

end length_of_bridge_l100_100732


namespace find_real_numbers_l100_100647

theorem find_real_numbers (x : ℝ) :
  (x^3 - x^2 = (x^2 - x)^2) ↔ (x = 0 ∨ x = 1 ∨ x = 2) :=
by
  sorry

end find_real_numbers_l100_100647


namespace scatter_plot_correlation_l100_100698

noncomputable def correlation_coefficient (points : List (ℝ × ℝ)) : ℝ := sorry

theorem scatter_plot_correlation {points : List (ℝ × ℝ)} 
  (h : ∃ (m : ℝ) (b : ℝ), m ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ points → y = m * x + b) :
  correlation_coefficient points = 1 := 
sorry

end scatter_plot_correlation_l100_100698


namespace samantha_spends_36_dollars_l100_100128

def cost_per_toy : ℝ := 12.00
def discount_factor : ℝ := 0.5
def num_toys_bought : ℕ := 4

def total_spent (cost_per_toy : ℝ) (discount_factor : ℝ) (num_toys_bought : ℕ) : ℝ :=
  let pair_cost := cost_per_toy + (cost_per_toy * discount_factor)
  let num_pairs := num_toys_bought / 2
  num_pairs * pair_cost

theorem samantha_spends_36_dollars :
  total_spent cost_per_toy discount_factor num_toys_bought = 36.00 :=
sorry

end samantha_spends_36_dollars_l100_100128


namespace minimum_value_2a_plus_3b_is_25_l100_100372

noncomputable def minimum_value_2a_plus_3b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (2 / a) + (3 / b) = 1) : ℝ :=
2 * a + 3 * b

theorem minimum_value_2a_plus_3b_is_25
  (a b : ℝ)
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : (2 / a) + (3 / b) = 1) :
  minimum_value_2a_plus_3b a b h₁ h₂ h₃ = 25 :=
sorry

end minimum_value_2a_plus_3b_is_25_l100_100372


namespace group_left_to_clean_is_third_group_l100_100318

-- Definition of group sizes
def group1 := 7
def group2 := 10
def group3 := 16
def group4 := 18

-- Definitions and conditions
def total_students := group1 + group2 + group3 + group4
def lecture_factor := 4
def english_students := 7  -- From solution: must be 7 students attending the English lecture
def math_students := lecture_factor * english_students

-- Hypothesis of the students allocating to lectures
def students_attending_lectures := english_students + math_students
def students_left_to_clean := total_students - students_attending_lectures

-- The statement to be proved in Lean
theorem group_left_to_clean_is_third_group
  (h : students_left_to_clean = group3) :
  students_left_to_clean = 16 :=
sorry

end group_left_to_clean_is_third_group_l100_100318


namespace Sherman_weekly_driving_time_l100_100416

theorem Sherman_weekly_driving_time (daily_commute : ℕ := 30) (weekend_drive : ℕ := 2) : 
  (5 * (2 * daily_commute) / 60 + 2 * weekend_drive) = 9 := 
by
  sorry

end Sherman_weekly_driving_time_l100_100416


namespace ice_cream_flavors_l100_100820

theorem ice_cream_flavors (F : ℕ) (h1 : F / 4 + F / 2 + 25 = F) : F = 100 :=
by
  sorry

end ice_cream_flavors_l100_100820


namespace initial_seashells_l100_100967

-- Definitions for the conditions
def seashells_given_to_Tim : ℕ := 13
def seashells_now : ℕ := 36

-- Proving the number of initially found seashells
theorem initial_seashells : seashells_now + seashells_given_to_Tim = 49 :=
by
  -- we omit the proof steps with sorry
  sorry

end initial_seashells_l100_100967


namespace number_of_emus_l100_100525

theorem number_of_emus (total_heads_and_legs : ℕ) (heads_per_emu legs_per_emu : ℕ) (total_emu : ℕ) :
  total_heads_and_legs = 60 → 
  heads_per_emu = 1 → 
  legs_per_emu = 2 → 
  total_emu = total_heads_and_legs / (heads_per_emu + legs_per_emu) → 
  total_emu = 20 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  refine h4
  sorry

end number_of_emus_l100_100525


namespace area_of_rectangular_field_l100_100726

theorem area_of_rectangular_field (length width perimeter : ℕ) 
  (h_perimeter : perimeter = 2 * (length + width)) 
  (h_length : length = 15) 
  (h_perimeter_value : perimeter = 70) : 
  (length * width = 300) :=
by
  sorry

end area_of_rectangular_field_l100_100726


namespace exists_A_for_sqrt_d_l100_100853

def is_not_perfect_square (d : ℕ) : Prop := ∀ m : ℕ, m * m ≠ d

def s (d n : ℕ) : ℕ := 
  -- count number of 1's in the first n digits of binary representation of √d
  sorry 

theorem exists_A_for_sqrt_d (d : ℕ) (h : is_not_perfect_square d) :
  ∃ A : ℕ, ∀ n ≥ A, s d n > Int.sqrt (2 * n) - 2 :=
sorry

end exists_A_for_sqrt_d_l100_100853


namespace value_of_y_l100_100891

theorem value_of_y (y : ℝ) (h : (3 * y - 9) / 3 = 18) : y = 21 :=
sorry

end value_of_y_l100_100891


namespace simplify_expression_l100_100186

theorem simplify_expression : 8^5 + 8^5 + 8^5 + 8^5 = 8^(17/3) :=
by
  -- Proof will be completed here
  sorry

end simplify_expression_l100_100186


namespace total_cans_collected_l100_100433

theorem total_cans_collected (students_perez : ℕ) (half_perez_collected_20 : ℕ) (two_perez_collected_0 : ℕ) (remaining_perez_collected_8 : ℕ)
                             (students_johnson : ℕ) (third_johnson_collected_25 : ℕ) (three_johnson_collected_0 : ℕ) (remaining_johnson_collected_10 : ℕ)
                             (hp : students_perez = 28) (hc1 : half_perez_collected_20 = 28 / 2) (hc2 : two_perez_collected_0 = 2) (hc3 : remaining_perez_collected_8 = 12)
                             (hj : students_johnson = 30) (jc1 : third_johnson_collected_25 = 30 / 3) (jc2 : three_johnson_collected_0 = 3) (jc3 : remaining_johnson_collected_10 = 18) :
    (half_perez_collected_20 * 20 + two_perez_collected_0 * 0 + remaining_perez_collected_8 * 8
    + third_johnson_collected_25 * 25 + three_johnson_collected_0 * 0 + remaining_johnson_collected_10 * 10) = 806 :=
by
  sorry

end total_cans_collected_l100_100433


namespace max_min_value_l100_100065

def f (x t : ℝ) : ℝ := x^2 - 2 * t * x + t

theorem max_min_value : 
  ∀ t : ℝ, (-1 ≤ t ∧ t ≤ 1) →
  (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → f x t ≥ -t^2 + t) →
  (∃ t : ℝ, (-1 ≤ t ∧ t ≤ 1) ∧ ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → f x t ≥ -t^2 + t ∧ -t^2 + t = 1/4) :=
sorry

end max_min_value_l100_100065


namespace abs_neg_three_l100_100286

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l100_100286


namespace problem_l100_100431

theorem problem
  (x y : ℝ)
  (h1 : x - y = 12)
  (h2 : x^2 + y^2 = 320) :
  x * y = 64 ∧ x^3 + y^3 = 4160 :=
by
  sorry

end problem_l100_100431


namespace recurring_decimal_to_fraction_correct_l100_100473

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l100_100473


namespace stars_substitution_correct_l100_100575

-- Define x and y with given conditions
def ends_in_5 (n : ℕ) : Prop := n % 10 = 5
def product_ends_in_25 (x y : ℕ) : Prop := (x * y) % 100 = 25
def tens_digit_even (n : ℕ) : Prop := (n / 10) % 2 = 0
def valid_tens_digit (n : ℕ) : Prop := (n / 10) % 10 ≤ 3

theorem stars_substitution_correct :
  ∃ (x y : ℕ), ends_in_5 x ∧ ends_in_5 y ∧ product_ends_in_25 x y ∧ tens_digit_even x ∧ valid_tens_digit y ∧ x * y = 9125 :=
sorry

end stars_substitution_correct_l100_100575


namespace abs_neg_three_l100_100279

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l100_100279


namespace denis_and_oleg_probability_l100_100020

noncomputable def probability_denisolga_play_each_other (n : ℕ) (i j : ℕ) (h1 : n = 26) (h2 : i ≠ j) : ℚ :=
  let number_of_pairs := (n * (n - 1)) / 2
  in (n - 1) / number_of_pairs

theorem denis_and_oleg_probability :
  probability_denisolga_play_each_other 26 1 2 rfl dec_trivial = 1 / 13 :=
sorry

end denis_and_oleg_probability_l100_100020


namespace total_students_in_class_l100_100833

theorem total_students_in_class
  (S : ℕ)
  (H1 : 5/8 * S = S - 60)
  (H2 : 60 = 3/8 * S) :
  S = 160 :=
by
  sorry

end total_students_in_class_l100_100833


namespace prime_pairs_sum_50_l100_100682

/-- There are exactly 4 unordered pairs of prime numbers (p1, p2) such that p1 + p2 = 50. -/
theorem prime_pairs_sum_50 : 
  { pair : ℕ × ℕ // pair.fst.prime ∧ pair.snd.prime ∧ (pair.fst + pair.snd = 50) } = 4 := 
sorry

end prime_pairs_sum_50_l100_100682


namespace abs_neg_three_l100_100283

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l100_100283


namespace least_number_of_marbles_divisible_l100_100327

theorem least_number_of_marbles_divisible (n : ℕ) : 
  (∀ k ∈ [2, 3, 4, 5, 6, 7, 8], n % k = 0) -> n >= 840 :=
by sorry

end least_number_of_marbles_divisible_l100_100327


namespace time_to_Lake_Park_restaurant_l100_100965

def time_to_Hidden_Lake := 15
def time_back_to_Park_Office := 7
def total_time_gone := 32

theorem time_to_Lake_Park_restaurant : 
  (total_time_gone = time_to_Hidden_Lake + time_back_to_Park_Office +
  (32 - (time_to_Hidden_Lake + time_back_to_Park_Office))) -> 
  (32 - (time_to_Hidden_Lake + time_back_to_Park_Office) = 10) := by
  intros 
  sorry

end time_to_Lake_Park_restaurant_l100_100965


namespace find_principal_l100_100498

/-- Given that the simple interest SI is Rs. 90, the rate R is 3.5 percent, and the time T is 4 years,
prove that the principal P is approximately Rs. 642.86 using the simple interest formula. -/
theorem find_principal
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 90) (h2 : R = 3.5) (h3 : T = 4) 
  : P = 90 * 100 / (3.5 * 4) :=
by
  sorry

end find_principal_l100_100498


namespace perimeter_of_square_l100_100439

theorem perimeter_of_square (a : Real) (h_a : a ^ 2 = 144) : 4 * a = 48 :=
by
  sorry

end perimeter_of_square_l100_100439


namespace triangle_angle_measure_l100_100396

theorem triangle_angle_measure {D E F : ℝ} (hD : D = 90) (hE : E = 2 * F + 15) : 
  D + E + F = 180 → F = 25 :=
by
  intro h_sum
  sorry

end triangle_angle_measure_l100_100396


namespace jackie_apples_l100_100023

variable (A J : ℕ)

-- Condition: Adam has 3 more apples than Jackie.
axiom h1 : A = J + 3

-- Condition: Adam has 9 apples.
axiom h2 : A = 9

-- Question: How many apples does Jackie have?
theorem jackie_apples : J = 6 :=
by
  -- We would normally the proof steps here, but we'll skip to the answer
  sorry

end jackie_apples_l100_100023


namespace mike_pens_l100_100154

-- Definitions based on the conditions
def initial_pens : ℕ := 25
def pens_after_mike (M : ℕ) : ℕ := initial_pens + M
def pens_after_cindy (M : ℕ) : ℕ := 2 * pens_after_mike M
def pens_after_sharon (M : ℕ) : ℕ := pens_after_cindy M - 19
def final_pens : ℕ := 75

-- The theorem we need to prove
theorem mike_pens (M : ℕ) (h : pens_after_sharon M = final_pens) : M = 22 := by
  have h1 : pens_after_sharon M = 2 * (25 + M) - 19 := rfl
  rw [h1] at h
  sorry

end mike_pens_l100_100154


namespace sum_of_common_ratios_l100_100109

variable {k p r : ℝ}

-- Condition 1: geometric sequences with distinct common ratios
-- Condition 2: a_3 - b_3 = 3(a_2 - b_2)
def geometric_sequences (k p r : ℝ) : Prop :=
  (k ≠ 0) ∧ (p ≠ r) ∧ (k * p^2 - k * r^2 = 3 * (k * p - k * r))

theorem sum_of_common_ratios (k p r : ℝ) (h : geometric_sequences k p r) : p + r = 3 :=
by
  sorry

end sum_of_common_ratios_l100_100109


namespace inverse_g_of_neg_92_l100_100816

noncomputable def g (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 1

theorem inverse_g_of_neg_92 : g (-3) = -92 :=
by 
-- This would be the proof but we are skipping it as requested
sorry

end inverse_g_of_neg_92_l100_100816


namespace price_of_skateboard_l100_100572

-- Given condition (0.20 * p = 300)
variable (p : ℝ)
axiom upfront_payment : 0.20 * p = 300

-- Theorem statement to prove the price of the skateboard
theorem price_of_skateboard : p = 1500 := by
  sorry

end price_of_skateboard_l100_100572


namespace calculation_correct_l100_100916

theorem calculation_correct : (18 / (3 + 9 - 6)) * 4 = 12 :=
by
  sorry

end calculation_correct_l100_100916


namespace juanita_loss_l100_100099

theorem juanita_loss
  (entry_fee : ℝ) (hit_threshold : ℕ) (drum_payment_per_hit : ℝ) (drums_hit : ℕ) :
  entry_fee = 10 →
  hit_threshold = 200 →
  drum_payment_per_hit = 0.025 →
  drums_hit = 300 →
  - (entry_fee - ((drums_hit - hit_threshold) * drum_payment_per_hit)) = 7.50 :=
by
  intros h1 h2 h3 h4
  sorry

end juanita_loss_l100_100099
