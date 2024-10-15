import Mathlib

namespace NUMINAMATH_GPT_new_percentage_of_managers_is_98_l2098_209820

def percentage_of_managers (initial_employees : ℕ) (initial_percentage_managers : ℕ) (managers_leaving : ℕ) : ℕ :=
  let initial_managers := initial_percentage_managers * initial_employees / 100
  let remaining_managers := initial_managers - managers_leaving
  let remaining_employees := initial_employees - managers_leaving
  (remaining_managers * 100) / remaining_employees

theorem new_percentage_of_managers_is_98 :
  percentage_of_managers 500 99 250 = 98 :=
by
  sorry

end NUMINAMATH_GPT_new_percentage_of_managers_is_98_l2098_209820


namespace NUMINAMATH_GPT_part1_part2_l2098_209836

def f (x : ℝ) : ℝ := abs (x + 2) - 2 * abs (x - 1)

theorem part1 : { x : ℝ | f x ≥ -2 } = { x : ℝ | -2/3 ≤ x ∧ x ≤ 6 } :=
by
  sorry

theorem part2 (a : ℝ) :
  (∀ x ≥ a, f x ≤ x - a) ↔ a ≤ -2 ∨ a ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2098_209836


namespace NUMINAMATH_GPT_john_gets_30_cans_l2098_209846

def normal_price : ℝ := 0.60
def total_paid : ℝ := 9.00

theorem john_gets_30_cans :
  (total_paid / normal_price) * 2 = 30 :=
by
  sorry

end NUMINAMATH_GPT_john_gets_30_cans_l2098_209846


namespace NUMINAMATH_GPT_ineq_power_sum_lt_pow_two_l2098_209864

theorem ineq_power_sum_lt_pow_two (x : ℝ) (n : ℕ) (hx : |x| < 1) (hn : 2 ≤ n) : 
  (1 + x)^n + (1 - x)^n < 2^n :=
by
  sorry

end NUMINAMATH_GPT_ineq_power_sum_lt_pow_two_l2098_209864


namespace NUMINAMATH_GPT_least_tablets_l2098_209872

theorem least_tablets (num_A num_B : ℕ) (hA : num_A = 10) (hB : num_B = 14) :
  ∃ n, n = 12 ∧
  ∀ extracted_tablets, extracted_tablets > 0 →
    (∃ (a b : ℕ), a + b = extracted_tablets ∧ a ≥ 2 ∧ b ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_least_tablets_l2098_209872


namespace NUMINAMATH_GPT_sandwich_cost_l2098_209827

theorem sandwich_cost (S : ℝ) (h : 2 * S + 4 * 0.87 = 8.36) : S = 2.44 :=
by sorry

end NUMINAMATH_GPT_sandwich_cost_l2098_209827


namespace NUMINAMATH_GPT_fair_total_revenue_l2098_209845

noncomputable def price_per_ticket : ℝ := 8
noncomputable def total_ticket_revenue : ℝ := 8000
noncomputable def total_tickets_sold : ℝ := total_ticket_revenue / price_per_ticket

noncomputable def food_revenue : ℝ := (3/5) * total_tickets_sold * 10
noncomputable def rounded_ride_revenue : ℝ := (333 : ℝ) * 6
noncomputable def ride_revenue : ℝ := rounded_ride_revenue
noncomputable def rounded_souvenir_revenue : ℝ := (166 : ℝ) * 18
noncomputable def souvenir_revenue : ℝ := rounded_souvenir_revenue
noncomputable def game_revenue : ℝ := (1/10) * total_tickets_sold * 5

noncomputable def total_additional_revenue : ℝ := food_revenue + ride_revenue + souvenir_revenue + game_revenue
noncomputable def total_revenue : ℝ := total_ticket_revenue + total_additional_revenue

theorem fair_total_revenue : total_revenue = 19486 := by
  sorry

end NUMINAMATH_GPT_fair_total_revenue_l2098_209845


namespace NUMINAMATH_GPT_max_value_of_function_l2098_209882

theorem max_value_of_function : 
  ∀ (x : ℝ), 0 ≤ x → x ≤ 1 → (3 * x - 4 * x^3) ≤ 1 :=
by
  intro x hx0 hx1
  -- proof goes here
  sorry

end NUMINAMATH_GPT_max_value_of_function_l2098_209882


namespace NUMINAMATH_GPT_negation_of_implication_l2098_209895

-- Definitions based on the conditions from part (a)
def original_prop (x : ℝ) : Prop := x > 5 → x > 0
def negation_candidate_A (x : ℝ) : Prop := x ≤ 5 → x ≤ 0

-- The goal is to prove that the negation of the original proposition
-- is equivalent to option A, that is:
theorem negation_of_implication (x : ℝ) : (¬ (x > 5 → x > 0)) = (x ≤ 5 → x ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_implication_l2098_209895


namespace NUMINAMATH_GPT_race_winner_laps_l2098_209866

/-- Given:
  * A lap equals 100 meters.
  * Award per hundred meters is $3.5.
  * The winner earned $7 per minute.
  * The race lasted 12 minutes.
  Prove that the number of laps run by the winner is 24.
-/ 
theorem race_winner_laps :
  let lap_distance := 100 -- meters
  let award_per_100meters := 3.5 -- dollars per 100 meters
  let earnings_per_minute := 7 -- dollars per minute
  let race_duration := 12 -- minutes
  let total_earnings := earnings_per_minute * race_duration
  let total_100meters := total_earnings / award_per_100meters
  let laps := total_100meters
  laps = 24 := by
  sorry

end NUMINAMATH_GPT_race_winner_laps_l2098_209866


namespace NUMINAMATH_GPT_direct_proportion_function_decrease_no_first_quadrant_l2098_209890

-- Part (1)
theorem direct_proportion_function (a b : ℝ) (h : y = (2*a-4)*x + (3-b)) : a ≠ 2 ∧ b = 3 :=
sorry

-- Part (2)
theorem decrease_no_first_quadrant (a b : ℝ) (h : y = (2*a-4)*x + (3-b)) : a < 2 ∧ b ≥ 3 :=
sorry

end NUMINAMATH_GPT_direct_proportion_function_decrease_no_first_quadrant_l2098_209890


namespace NUMINAMATH_GPT_angle_B_value_value_of_k_l2098_209832

variable {A B C a b c : ℝ}
variable {k : ℝ}
variable {m n : ℝ × ℝ}

theorem angle_B_value
  (h1 : (2 * a - c) * Real.cos B = b * Real.cos C) :
  B = Real.pi / 3 :=
by sorry

theorem value_of_k
  (hA : 0 < A ∧ A < 2 * Real.pi / 3)
  (hm : m = (Real.sin A, Real.cos (2 * A)))
  (hn : n = (4 * k, 1))
  (hM : 4 * k * Real.sin A + Real.cos (2 * A) = 7) :
  k = 2 :=
by sorry

end NUMINAMATH_GPT_angle_B_value_value_of_k_l2098_209832


namespace NUMINAMATH_GPT_ratio_of_areas_of_squares_l2098_209870

theorem ratio_of_areas_of_squares (sideC sideD : ℕ) (hC : sideC = 45) (hD : sideD = 60) : 
  (sideC ^ 2) / (sideD ^ 2) = 9 / 16 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_squares_l2098_209870


namespace NUMINAMATH_GPT_marks_in_mathematics_l2098_209894

-- Define the marks obtained in each subject and the average
def marks_in_english : ℕ := 86
def marks_in_physics : ℕ := 82
def marks_in_chemistry : ℕ := 87
def marks_in_biology : ℕ := 85
def average_marks : ℕ := 85
def number_of_subjects : ℕ := 5

-- The theorem to prove the marks in Mathematics
theorem marks_in_mathematics : ℕ :=
  let sum_of_marks := average_marks * number_of_subjects
  let sum_of_known_marks := marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology
  sum_of_marks - sum_of_known_marks

-- The expected result that we need to prove
example : marks_in_mathematics = 85 := by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_marks_in_mathematics_l2098_209894


namespace NUMINAMATH_GPT_certain_number_z_l2098_209897

theorem certain_number_z (x y z : ℝ) (h1 : 0.5 * x = y + z) (h2 : x - 2 * y = 40) : z = 20 :=
by 
  sorry

end NUMINAMATH_GPT_certain_number_z_l2098_209897


namespace NUMINAMATH_GPT_common_ratio_l2098_209858

variable {G : Type} [LinearOrderedField G]

-- Definitions based on conditions
def geometric_seq (a₁ q : G) (n : ℕ) : G := a₁ * q^(n-1)
def sum_geometric_seq (a₁ q : G) (n : ℕ) : G :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from conditions
variable {a₁ q : G}
variable (h1 : sum_geometric_seq a₁ q 3 = 7)
variable (h2 : sum_geometric_seq a₁ q 6 = 63)

theorem common_ratio (a₁ q : G) (h1 : sum_geometric_seq a₁ q 3 = 7)
  (h2 : sum_geometric_seq a₁ q 6 = 63) : q = 2 :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_common_ratio_l2098_209858


namespace NUMINAMATH_GPT_melanie_total_payment_l2098_209892

noncomputable def totalCost (rentalCostPerDay : ℝ) (insuranceCostPerDay : ℝ) (mileageCostPerMile : ℝ) (days : ℕ) (miles : ℕ) : ℝ :=
  (rentalCostPerDay * days) + (insuranceCostPerDay * days) + (mileageCostPerMile * miles)

theorem melanie_total_payment :
  totalCost 30 5 0.25 3 350 = 192.5 :=
by
  sorry

end NUMINAMATH_GPT_melanie_total_payment_l2098_209892


namespace NUMINAMATH_GPT_find_m_for_perfect_square_trinomial_l2098_209829

theorem find_m_for_perfect_square_trinomial :
  ∃ m : ℤ, (∀ (x y : ℝ), (9 * x^2 + m * x * y + 16 * y^2 = (3 * x + 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (3 * x - 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (-3 * x + 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (-3 * x - 4 * y)^2)) ↔ 
          (m = 24 ∨ m = -24) := 
by
  sorry

end NUMINAMATH_GPT_find_m_for_perfect_square_trinomial_l2098_209829


namespace NUMINAMATH_GPT_dandelion_seeds_percentage_approx_29_27_l2098_209822

/-
Mathematical conditions:
- Carla has the following set of plants and seeds per plant:
  - 6 sunflowers with 9 seeds each
  - 8 dandelions with 12 seeds each
  - 4 roses with 7 seeds each
  - 10 tulips with 15 seeds each.
- Calculate:
  - total seeds
  - percentage of seeds from dandelions
-/ 

def num_sunflowers : ℕ := 6
def num_dandelions : ℕ := 8
def num_roses : ℕ := 4
def num_tulips : ℕ := 10

def seeds_per_sunflower : ℕ := 9
def seeds_per_dandelion : ℕ := 12
def seeds_per_rose : ℕ := 7
def seeds_per_tulip : ℕ := 15

def total_sunflower_seeds : ℕ := num_sunflowers * seeds_per_sunflower
def total_dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion
def total_rose_seeds : ℕ := num_roses * seeds_per_rose
def total_tulip_seeds : ℕ := num_tulips * seeds_per_tulip

def total_seeds : ℕ := total_sunflower_seeds + total_dandelion_seeds + total_rose_seeds + total_tulip_seeds

def percentage_dandelion_seeds : ℚ := (total_dandelion_seeds : ℚ) / total_seeds * 100

theorem dandelion_seeds_percentage_approx_29_27 : abs (percentage_dandelion_seeds - 29.27) < 0.01 :=
sorry

end NUMINAMATH_GPT_dandelion_seeds_percentage_approx_29_27_l2098_209822


namespace NUMINAMATH_GPT_number_of_siblings_l2098_209859

-- Definitions for the given conditions
def total_height : ℕ := 330
def sibling1_height : ℕ := 66
def sibling2_height : ℕ := 66
def sibling3_height : ℕ := 60
def last_sibling_height : ℕ := 70  -- Derived from the solution steps
def eliza_height : ℕ := last_sibling_height - 2

-- The final question to validate
theorem number_of_siblings (h : 2 * sibling1_height + sibling3_height + last_sibling_height + eliza_height = total_height) :
  4 = 4 :=
by {
  -- Condition h states that the total height is satisfied
  -- Therefore, it directly justifies our claim without further computation here.
  sorry
}

end NUMINAMATH_GPT_number_of_siblings_l2098_209859


namespace NUMINAMATH_GPT_amplitude_of_cosine_wave_l2098_209886

theorem amplitude_of_cosine_wave 
  (a b c d : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) 
  (h_max_min : ∀ x : ℝ, d + a = 5 ∧ d - a = 1) 
  : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_amplitude_of_cosine_wave_l2098_209886


namespace NUMINAMATH_GPT_proof_problem_l2098_209804

variable {a b c : ℝ}

-- Condition: a < 0
variable (ha : a < 0)
-- Condition: b > 0
variable (hb : b > 0)
-- Condition: c > 0
variable (hc : c > 0)
-- Condition: a < b < c
variable (hab : a < b) (hbc : b < c)

-- Proof statement
theorem proof_problem :
  (ab * b < b * c) ∧
  (a * c < b * c) ∧
  (a + c < b + c) ∧
  (c / a < 1) :=
  by
    sorry

end NUMINAMATH_GPT_proof_problem_l2098_209804


namespace NUMINAMATH_GPT_leak_empties_tank_in_24_hours_l2098_209849

theorem leak_empties_tank_in_24_hours (A L : ℝ) (hA : A = 1 / 8) (h_comb : A - L = 1 / 12) : 1 / L = 24 :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_leak_empties_tank_in_24_hours_l2098_209849


namespace NUMINAMATH_GPT_congruent_triangles_solve_x_l2098_209816

theorem congruent_triangles_solve_x (x : ℝ) (h1 : x > 0)
    (h2 : x^2 - 1 = 3) (h3 : x^2 + 1 = 5) (h4 : x^2 + 3 = 7) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_congruent_triangles_solve_x_l2098_209816


namespace NUMINAMATH_GPT_sandy_age_when_record_l2098_209819

noncomputable def calc_age (record_length current_length monthly_growth_rate age : ℕ) : ℕ :=
  let yearly_growth_rate := monthly_growth_rate * 12
  let needed_length := record_length - current_length
  let years_needed := needed_length / yearly_growth_rate
  age + years_needed

theorem sandy_age_when_record (record_length current_length monthly_growth_rate age : ℕ) :
  record_length = 26 →
  current_length = 2 →
  monthly_growth_rate = 1 →
  age = 12 →
  calc_age record_length current_length monthly_growth_rate age = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold calc_age
  simp
  sorry

end NUMINAMATH_GPT_sandy_age_when_record_l2098_209819


namespace NUMINAMATH_GPT_solve_system_l2098_209853

theorem solve_system (X Y Z : ℝ)
  (h1 : 0.15 * 40 = 0.25 * X + 2)
  (h2 : 0.30 * 60 = 0.20 * Y + 3)
  (h3 : 0.10 * Z = X - Y) :
  X = 16 ∧ Y = 75 ∧ Z = -590 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2098_209853


namespace NUMINAMATH_GPT_find_cost_of_chocolate_l2098_209884

theorem find_cost_of_chocolate
  (C : ℕ)
  (h1 : 5 * C + 10 = 90 - 55)
  (h2 : 5 * 2 = 10)
  (h3 : 55 = 90 - (5 * C + 10)):
  C = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_of_chocolate_l2098_209884


namespace NUMINAMATH_GPT_Katie_has_more_games_than_friends_l2098_209863

def katie_new_games : ℕ := 57
def katie_old_games : ℕ := 39
def friends_new_games : ℕ := 34

theorem Katie_has_more_games_than_friends :
  (katie_new_games + katie_old_games) - friends_new_games = 62 := by
  sorry

end NUMINAMATH_GPT_Katie_has_more_games_than_friends_l2098_209863


namespace NUMINAMATH_GPT_min_value_2x_plus_y_l2098_209887

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 2 / (y + 1) = 2) :
  2 * x + y = 3 :=
sorry

end NUMINAMATH_GPT_min_value_2x_plus_y_l2098_209887


namespace NUMINAMATH_GPT_proof_line_eq_l2098_209855

variable (a T : ℝ) (line : ℝ × ℝ → Prop)

def line_eq (point : ℝ × ℝ) : Prop := 
  point.2 = (-2 * T / a^2) * point.1 + (2 * T / a)

def correct_line_eq (point : ℝ × ℝ) : Prop :=
  -2 * T * point.1 + a^2 * point.2 + 2 * a * T = 0

theorem proof_line_eq :
  ∀ point : ℝ × ℝ, line_eq a T point ↔ correct_line_eq a T point :=
by
  sorry

end NUMINAMATH_GPT_proof_line_eq_l2098_209855


namespace NUMINAMATH_GPT_amoeba_growth_one_week_l2098_209878

theorem amoeba_growth_one_week :
  (3 ^ 7 = 2187) :=
by
  sorry

end NUMINAMATH_GPT_amoeba_growth_one_week_l2098_209878


namespace NUMINAMATH_GPT_bus_trip_speed_l2098_209811

theorem bus_trip_speed :
  ∃ v : ℝ, v > 0 ∧ (660 / v - 1 = 660 / (v + 5)) ∧ v = 55 :=
by
  sorry

end NUMINAMATH_GPT_bus_trip_speed_l2098_209811


namespace NUMINAMATH_GPT_anna_original_money_l2098_209815

theorem anna_original_money (x : ℝ) (h : (3 / 4) * x = 24) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_anna_original_money_l2098_209815


namespace NUMINAMATH_GPT_total_animals_to_spay_l2098_209862

theorem total_animals_to_spay : 
  ∀ (c d : ℕ), c = 7 → d = 2 * c → c + d = 21 :=
by
  intros c d h1 h2
  sorry

end NUMINAMATH_GPT_total_animals_to_spay_l2098_209862


namespace NUMINAMATH_GPT_left_handed_rock_lovers_l2098_209842

def total_people := 30
def left_handed := 12
def like_rock_music := 20
def right_handed_dislike_rock := 3

theorem left_handed_rock_lovers : ∃ x, x + (left_handed - x) + (like_rock_music - x) + right_handed_dislike_rock = total_people ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_left_handed_rock_lovers_l2098_209842


namespace NUMINAMATH_GPT_max_square_test_plots_l2098_209840

theorem max_square_test_plots (length width fence : ℕ)
  (h_length : length = 36)
  (h_width : width = 66)
  (h_fence : fence = 2200) :
  ∃ (n : ℕ), n * (11 / 6) * n = 264 ∧
      (36 * n + (11 * n - 6) * 66) ≤ 2200 := sorry

end NUMINAMATH_GPT_max_square_test_plots_l2098_209840


namespace NUMINAMATH_GPT_solve_f_l2098_209847

open Nat

theorem solve_f (f : ℕ → ℕ) (h : ∀ n : ℕ, f (f n) + f n = 2 * n + 3) : f 1993 = 1994 := by
  -- assumptions and required proof
  sorry

end NUMINAMATH_GPT_solve_f_l2098_209847


namespace NUMINAMATH_GPT_ribbon_tape_length_l2098_209843

theorem ribbon_tape_length
  (one_ribbon: ℝ)
  (remaining_cm: ℝ)
  (num_ribbons: ℕ)
  (total_used: ℝ)
  (remaining_meters: remaining_cm = 0.50)
  (ribbon_meter: one_ribbon = 0.84)
  (ribbons_made: num_ribbons = 10)
  (used_len: total_used = one_ribbon * num_ribbons):
  total_used + 0.50 = 8.9 :=
by
  sorry

end NUMINAMATH_GPT_ribbon_tape_length_l2098_209843


namespace NUMINAMATH_GPT_vertical_asymptotes_sum_l2098_209876

theorem vertical_asymptotes_sum : 
  (∀ x : ℝ, 4 * x^2 + 7 * x + 3 = 0 → x = -3 / 4 ∨ x = -1) →
  (-3 / 4) + (-1) = -7 / 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_vertical_asymptotes_sum_l2098_209876


namespace NUMINAMATH_GPT_pen_average_price_l2098_209893

theorem pen_average_price (pens_purchased pencils_purchased : ℕ) (total_cost pencil_avg_price : ℝ)
  (H0 : pens_purchased = 30) (H1 : pencils_purchased = 75) 
  (H2 : total_cost = 690) (H3 : pencil_avg_price = 2) :
  (total_cost - (pencils_purchased * pencil_avg_price)) / pens_purchased = 18 :=
by
  rw [H0, H1, H2, H3]
  sorry

end NUMINAMATH_GPT_pen_average_price_l2098_209893


namespace NUMINAMATH_GPT_track_circumference_is_180_l2098_209821

noncomputable def track_circumference : ℕ :=
  let brenda_first_meeting_dist := 120
  let sally_second_meeting_dist := 180
  let brenda_speed_factor : ℕ := 2
  -- circumference of the track
  let circumference := 3 * brenda_first_meeting_dist / brenda_speed_factor
  circumference

theorem track_circumference_is_180 :
  track_circumference = 180 :=
by 
  sorry

end NUMINAMATH_GPT_track_circumference_is_180_l2098_209821


namespace NUMINAMATH_GPT_coin_problem_l2098_209830

theorem coin_problem :
  ∃ n : ℕ, (n % 8 = 5) ∧ (n % 7 = 2) ∧ (n % 9 = 1) := 
sorry

end NUMINAMATH_GPT_coin_problem_l2098_209830


namespace NUMINAMATH_GPT_log_sqrt_pi_simplification_l2098_209818

theorem log_sqrt_pi_simplification:
  2 * Real.log 4 + Real.log (5 / 8) + Real.sqrt ((Real.sqrt 3 - Real.pi) ^ 2) = 1 + Real.pi - Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_log_sqrt_pi_simplification_l2098_209818


namespace NUMINAMATH_GPT_max_area_of_garden_l2098_209835

theorem max_area_of_garden (l w : ℝ) 
  (h : 2 * l + w = 400) : 
  l * w ≤ 20000 :=
sorry

end NUMINAMATH_GPT_max_area_of_garden_l2098_209835


namespace NUMINAMATH_GPT_root_of_unity_product_l2098_209833

theorem root_of_unity_product (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (1 - ω + ω^2) * (1 + ω - ω^2) = 1 :=
  sorry

end NUMINAMATH_GPT_root_of_unity_product_l2098_209833


namespace NUMINAMATH_GPT_expression_comparison_l2098_209801

theorem expression_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) :
  let exprI := (a + (1 / a)) * (b + (1 / b))
  let exprII := (Real.sqrt (a * b) + (1 / Real.sqrt (a * b))) ^ 2
  let exprIII := (((a + b) / 2) + (2 / (a + b))) ^ 2
  (exprI = exprII ∨ exprI = exprIII ∨ exprII = exprIII ∨ 
   (exprI > exprII ∧ exprI > exprIII) ∨
   (exprII > exprI ∧ exprII > exprIII) ∨
   (exprIII > exprI ∧ exprIII > exprII)) ∧
  ¬((exprI > exprII ∧ exprI > exprIII) ∨
    (exprII > exprI ∧ exprII > exprIII) ∨
    (exprIII > exprI ∧ exprIII > exprII)) :=
by
  let exprI := (a + (1 / a)) * (b + (1 / b))
  let exprII := (Real.sqrt (a * b) + (1 / Real.sqrt (a * b))) ^ 2
  let exprIII := (((a + b) / 2) + (2 / (a + b))) ^ 2
  sorry

end NUMINAMATH_GPT_expression_comparison_l2098_209801


namespace NUMINAMATH_GPT_impossible_seed_germinate_without_water_l2098_209861

-- Definitions for the conditions
def heats_up_when_conducting (conducts : Bool) : Prop := conducts
def determines_plane (non_collinear : Bool) : Prop := non_collinear
def germinates_without_water (germinates : Bool) : Prop := germinates
def wins_lottery_consecutively (wins_twice : Bool) : Prop := wins_twice

-- The fact that a seed germinates without water is impossible
theorem impossible_seed_germinate_without_water 
  (conducts : Bool) 
  (non_collinear : Bool) 
  (germinates : Bool) 
  (wins_twice : Bool) 
  (h1 : heats_up_when_conducting conducts) 
  (h2 : determines_plane non_collinear) 
  (h3 : ¬germinates_without_water germinates) 
  (h4 : wins_lottery_consecutively wins_twice) :
  ¬germinates_without_water true :=
sorry

end NUMINAMATH_GPT_impossible_seed_germinate_without_water_l2098_209861


namespace NUMINAMATH_GPT_age_difference_is_58_l2098_209806

def Milena_age : ℕ := 7
def Grandmother_age : ℕ := 9 * Milena_age
def Grandfather_age : ℕ := Grandmother_age + 2
def Age_difference : ℕ := Grandfather_age - Milena_age

theorem age_difference_is_58 : Age_difference = 58 := by
  sorry

end NUMINAMATH_GPT_age_difference_is_58_l2098_209806


namespace NUMINAMATH_GPT_negative_solution_exists_l2098_209898

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end NUMINAMATH_GPT_negative_solution_exists_l2098_209898


namespace NUMINAMATH_GPT_rectangle_length_increase_l2098_209879

variable (L B : ℝ) -- Original length and breadth
variable (A : ℝ) -- Original area
variable (p : ℝ) -- Percentage increase in length
variable (A' : ℝ) -- New area

theorem rectangle_length_increase (hA : A = L * B) 
  (hp : L' = L + (p / 100) * L) 
  (hB' : B' = B * 0.9) 
  (hA' : A' = 1.035 * A)
  (hl' : L' = (1 + (p / 100)) * L)
  (hb_length : L' * B' = A') :
  p = 15 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_increase_l2098_209879


namespace NUMINAMATH_GPT_solvable_consecutive_integers_solvable_consecutive_even_integers_not_solvable_consecutive_odd_integers_l2098_209885

-- Definitions only directly appearing in the conditions problem
def consecutive_integers (x y z : ℤ) : Prop := x = y - 1 ∧ z = y + 1
def consecutive_even_integers (x y z : ℤ) : Prop := x = y - 2 ∧ z = y + 2 ∧ y % 2 = 0
def consecutive_odd_integers (x y z : ℤ) : Prop := x = y - 2 ∧ z = y + 2 ∧ y % 2 = 1

-- Problem Statements
theorem solvable_consecutive_integers : ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_integers x y z :=
sorry

theorem solvable_consecutive_even_integers : ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_even_integers x y z :=
sorry

theorem not_solvable_consecutive_odd_integers : ¬ ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_odd_integers x y z :=
sorry

end NUMINAMATH_GPT_solvable_consecutive_integers_solvable_consecutive_even_integers_not_solvable_consecutive_odd_integers_l2098_209885


namespace NUMINAMATH_GPT_negation_of_universal_to_existential_l2098_209865

theorem negation_of_universal_to_existential :
  (¬(∀ x : ℝ, x^2 > 0)) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_universal_to_existential_l2098_209865


namespace NUMINAMATH_GPT_maximum_value_of_f_l2098_209813

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem maximum_value_of_f : ∃ x : ℝ, f x = 13 :=
by 
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_l2098_209813


namespace NUMINAMATH_GPT_mary_characters_initial_D_l2098_209867

theorem mary_characters_initial_D (total_characters initial_A initial_C initial_D initial_E : ℕ)
  (h1 : total_characters = 60)
  (h2 : initial_A = total_characters / 2)
  (h3 : initial_C = initial_A / 2)
  (remaining := total_characters - initial_A - initial_C)
  (h4 : remaining = initial_D + initial_E)
  (h5 : initial_D = 2 * initial_E) : initial_D = 10 := by
  sorry

end NUMINAMATH_GPT_mary_characters_initial_D_l2098_209867


namespace NUMINAMATH_GPT_second_group_work_days_l2098_209854

theorem second_group_work_days (M B : ℕ) (d1 d2 : ℕ) (H1 : M = 2 * B) 
  (H2 : (12 * M + 16 * B) * 5 = d1) (H3 : (13 * M + 24 * B) * d2 = d1) : 
  d2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_second_group_work_days_l2098_209854


namespace NUMINAMATH_GPT_Lisa_photos_l2098_209826

variable (a f s : ℕ)

theorem Lisa_photos (h1: a = 10) (h2: f = 3 * a) (h3: s = f - 10) : a + f + s = 60 := by
  sorry

end NUMINAMATH_GPT_Lisa_photos_l2098_209826


namespace NUMINAMATH_GPT_find_A_and_B_l2098_209856

theorem find_A_and_B (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ -6 → (5 * x - 3) / (x^2 + 3 * x - 18) = A / (x - 3) + B / (x + 6)) →
  A = 4 / 3 ∧ B = 11 / 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_A_and_B_l2098_209856


namespace NUMINAMATH_GPT_product_divisibility_l2098_209848

theorem product_divisibility (a b c : ℤ)
  (h₁ : (a + b + c) ^ 2 = -(a * b + a * c + b * c))
  (h₂ : a + b ≠ 0)
  (h₃ : b + c ≠ 0)
  (h₄ : a + c ≠ 0) :
  (a + b) * (a + c) % (b + c) = 0 ∧
  (a + b) * (b + c) % (a + c) = 0 ∧
  (a + c) * (b + c) % (a + b) = 0 := by
  sorry

end NUMINAMATH_GPT_product_divisibility_l2098_209848


namespace NUMINAMATH_GPT_additional_fertilizer_on_final_day_l2098_209834

noncomputable def normal_usage_per_day : ℕ := 2
noncomputable def total_days : ℕ := 9
noncomputable def total_fertilizer_used : ℕ := 22

theorem additional_fertilizer_on_final_day :
  total_fertilizer_used - (normal_usage_per_day * total_days) = 4 := by
  sorry

end NUMINAMATH_GPT_additional_fertilizer_on_final_day_l2098_209834


namespace NUMINAMATH_GPT_total_rooms_booked_l2098_209875

variable (S D : ℕ)

theorem total_rooms_booked (h1 : 35 * S + 60 * D = 14000) (h2 : D = 196) : S + D = 260 :=
by
  sorry

end NUMINAMATH_GPT_total_rooms_booked_l2098_209875


namespace NUMINAMATH_GPT_inequality_proof_l2098_209808

open scoped BigOperators

theorem inequality_proof {n : ℕ} (a : Fin n → ℝ) 
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 1 / 2) :
  (∑ i, (a i)^2 / (∑ i, a i)^2) ≥ (∑ i, (1 - a i)^2 / (∑ i, (1 - a i))^2) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l2098_209808


namespace NUMINAMATH_GPT_number_of_squares_with_prime_condition_l2098_209810

theorem number_of_squares_with_prime_condition : 
  ∃! (n : ℕ), ∃ (p : ℕ), Prime p ∧ n^2 = p + 4 := 
sorry

end NUMINAMATH_GPT_number_of_squares_with_prime_condition_l2098_209810


namespace NUMINAMATH_GPT_geometric_sequence_a8_l2098_209841

theorem geometric_sequence_a8 (a : ℕ → ℝ) (q : ℝ) 
  (h₁ : a 3 = 3)
  (h₂ : a 6 = 24)
  (h₃ : ∀ n, a (n + 1) = a n * q) : 
  a 8 = 96 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a8_l2098_209841


namespace NUMINAMATH_GPT_circle_diameter_l2098_209807
open Real

theorem circle_diameter (A : ℝ) (hA : A = 50.26548245743669) : ∃ d : ℝ, d = 8 :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_l2098_209807


namespace NUMINAMATH_GPT_man_reaches_home_at_11_pm_l2098_209880

theorem man_reaches_home_at_11_pm :
  let start_time := 15 -- represents 3 pm in 24-hour format
  let level_speed := 4 -- km/hr
  let uphill_speed := 3 -- km/hr
  let downhill_speed := 6 -- km/hr
  let total_distance := 12 -- km
  let level_distance := 4 -- km
  let uphill_distance := 4 -- km
  let downhill_distance := 4 -- km
  let level_time := level_distance / level_speed -- time for 4 km on level ground
  let uphill_time := uphill_distance / uphill_speed -- time for 4 km uphill
  let downhill_time := downhill_distance / downhill_speed -- time for 4 km downhill
  let total_time_one_way := level_time + uphill_time + downhill_time + level_time
  let destination_time := start_time + total_time_one_way
  let return_time := destination_time + total_time_one_way
  return_time = 23 := -- represents 11 pm in 24-hour format
by
  sorry

end NUMINAMATH_GPT_man_reaches_home_at_11_pm_l2098_209880


namespace NUMINAMATH_GPT_find_largest_angle_l2098_209852

noncomputable def largest_angle_in_convex_pentagon (x : ℝ) : Prop :=
  let angle1 := 2 * x + 2
  let angle2 := 3 * x - 3
  let angle3 := 4 * x + 4
  let angle4 := 6 * x - 6
  let angle5 := x + 5
  angle1 + angle2 + angle3 + angle4 + angle5 = 540 ∧
  max (max angle1 (max angle2 (max angle3 angle4))) angle5 = angle4 ∧
  angle4 = 195.75

theorem find_largest_angle (x : ℝ) : largest_angle_in_convex_pentagon x := by
  sorry

end NUMINAMATH_GPT_find_largest_angle_l2098_209852


namespace NUMINAMATH_GPT_inequality_not_satisfied_integer_values_count_l2098_209888

theorem inequality_not_satisfied_integer_values_count :
  ∃ (n : ℕ), n = 5 ∧ ∀ (x : ℤ), 3 * x^2 + 17 * x + 20 ≤ 25 → x ∈ [-4, -3, -2, -1, 0] :=
  sorry

end NUMINAMATH_GPT_inequality_not_satisfied_integer_values_count_l2098_209888


namespace NUMINAMATH_GPT_rate_of_mixed_oil_per_litre_l2098_209817

theorem rate_of_mixed_oil_per_litre :
  let oil1_litres := 10
  let oil1_rate := 55
  let oil2_litres := 5
  let oil2_rate := 66
  let total_cost := oil1_litres * oil1_rate + oil2_litres * oil2_rate
  let total_volume := oil1_litres + oil2_litres
  let rate_per_litre := total_cost / total_volume
  rate_per_litre = 58.67 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_mixed_oil_per_litre_l2098_209817


namespace NUMINAMATH_GPT_george_money_left_after_donations_and_groceries_l2098_209828

def monthly_income : ℕ := 240
def donation (income : ℕ) : ℕ := income / 2
def post_donation_money (income : ℕ) : ℕ := income - donation income
def groceries_cost : ℕ := 20
def money_left (income : ℕ) : ℕ := post_donation_money income - groceries_cost

theorem george_money_left_after_donations_and_groceries :
  money_left monthly_income = 100 :=
by
  sorry

end NUMINAMATH_GPT_george_money_left_after_donations_and_groceries_l2098_209828


namespace NUMINAMATH_GPT_notepad_duration_l2098_209896

theorem notepad_duration (a8_papers_per_a4 : ℕ)
  (a4_papers : ℕ)
  (notes_per_day : ℕ)
  (notes_per_side : ℕ) :
  a8_papers_per_a4 = 16 →
  a4_papers = 8 →
  notes_per_day = 15 →
  notes_per_side = 2 →
  (a4_papers * a8_papers_per_a4 * notes_per_side) / notes_per_day = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_notepad_duration_l2098_209896


namespace NUMINAMATH_GPT_MrBensonPaidCorrectAmount_l2098_209800

-- Definitions based on the conditions
def generalAdmissionTicketPrice : ℤ := 40
def VIPTicketPrice : ℤ := 60
def premiumTicketPrice : ℤ := 80

def generalAdmissionTicketsBought : ℤ := 10
def VIPTicketsBought : ℤ := 3
def premiumTicketsBought : ℤ := 2

def generalAdmissionExcessThreshold : ℤ := 8
def VIPExcessThreshold : ℤ := 2
def premiumExcessThreshold : ℤ := 1

def generalAdmissionDiscountPercentage : ℤ := 3
def VIPDiscountPercentage : ℤ := 7
def premiumDiscountPercentage : ℤ := 10

-- Function to calculate the cost without discounts
def costWithoutDiscount : ℤ :=
  (generalAdmissionTicketsBought * generalAdmissionTicketPrice) +
  (VIPTicketsBought * VIPTicketPrice) +
  (premiumTicketsBought * premiumTicketPrice)

-- Function to calculate the total discount
def totalDiscount : ℤ :=
  let generalAdmissionDiscount := if generalAdmissionTicketsBought > generalAdmissionExcessThreshold then 
    (generalAdmissionTicketsBought - generalAdmissionExcessThreshold) * generalAdmissionTicketPrice * generalAdmissionDiscountPercentage / 100 else 0
  let VIPDiscount := if VIPTicketsBought > VIPExcessThreshold then 
    (VIPTicketsBought - VIPExcessThreshold) * VIPTicketPrice * VIPDiscountPercentage / 100 else 0
  let premiumDiscount := if premiumTicketsBought > premiumExcessThreshold then 
    (premiumTicketsBought - premiumExcessThreshold) * premiumTicketPrice * premiumDiscountPercentage / 100 else 0
  generalAdmissionDiscount + VIPDiscount + premiumDiscount

-- Function to calculate the total cost after discounts
def totalCostAfterDiscount : ℤ := costWithoutDiscount - totalDiscount

-- Proof statement
theorem MrBensonPaidCorrectAmount :
  totalCostAfterDiscount = 723 :=
by
  sorry

end NUMINAMATH_GPT_MrBensonPaidCorrectAmount_l2098_209800


namespace NUMINAMATH_GPT_range_of_2alpha_minus_beta_over_3_l2098_209871

theorem range_of_2alpha_minus_beta_over_3 (α β : ℝ) (hα : 0 < α) (hα' : α < π / 2) (hβ : 0 < β) (hβ' : β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := 
sorry

end NUMINAMATH_GPT_range_of_2alpha_minus_beta_over_3_l2098_209871


namespace NUMINAMATH_GPT_part_one_part_two_l2098_209803

def f (x : ℝ) : ℝ := |x| + |x - 1|

theorem part_one (m : ℝ) (h : ∀ x, f x ≥ |m - 1|) : m ≤ 2 := by
  sorry

theorem part_two (a b : ℝ) (M : ℝ) (ha : 0 < a) (hb : 0 < b) (hM : a^2 + b^2 = M) (hM_value : M = 2) : a + b ≥ 2 * a * b := by
  sorry

end NUMINAMATH_GPT_part_one_part_two_l2098_209803


namespace NUMINAMATH_GPT_solve_inequality_system_l2098_209823

-- Define the conditions and the correct answer
def system_of_inequalities (x : ℝ) : Prop :=
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x)

def solution_set (x : ℝ) : Prop :=
  2 < x ∧ x ≤ 4

-- State that solving the system of inequalities is equivalent to the solution set
theorem solve_inequality_system (x : ℝ) : system_of_inequalities x ↔ solution_set x :=
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l2098_209823


namespace NUMINAMATH_GPT_evaluate_x_squared_plus_y_squared_l2098_209874

theorem evaluate_x_squared_plus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 20) :
  x^2 + y^2 = 80 := by
  sorry

end NUMINAMATH_GPT_evaluate_x_squared_plus_y_squared_l2098_209874


namespace NUMINAMATH_GPT_range_of_b_l2098_209873

variable (a b c : ℝ)

theorem range_of_b (h1 : a + b + c = 9) (h2 : a * b + b * c + c * a = 24) : 1 ≤ b ∧ b ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l2098_209873


namespace NUMINAMATH_GPT_find_a_given_solution_set_l2098_209851

theorem find_a_given_solution_set :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 ↔ x^2 + a * x + 6 ≤ 0) → a = -5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_given_solution_set_l2098_209851


namespace NUMINAMATH_GPT_sin_double_angle_shift_l2098_209869

variable (θ : Real)

theorem sin_double_angle_shift (h : Real.cos (θ + Real.pi) = -1 / 3) :
  Real.sin (2 * θ + Real.pi / 2) = -7 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_shift_l2098_209869


namespace NUMINAMATH_GPT_division_result_l2098_209850

-- Definitions for the values used in the problem
def numerator := 0.0048 * 3.5
def denominator := 0.05 * 0.1 * 0.004

-- Theorem statement
theorem division_result : numerator / denominator = 840 := by 
  sorry

end NUMINAMATH_GPT_division_result_l2098_209850


namespace NUMINAMATH_GPT_max_value_of_y_l2098_209814

open Classical

noncomputable def satisfies_equation (x y : ℝ) : Prop := y * x * (x + y) = x - y

theorem max_value_of_y : 
  ∀ (y : ℝ), (∃ (x : ℝ), x > 0 ∧ satisfies_equation x y) → y ≤ 1 / 3 := 
sorry

end NUMINAMATH_GPT_max_value_of_y_l2098_209814


namespace NUMINAMATH_GPT_students_drawn_from_class_A_l2098_209891

-- Given conditions
def classA_students : Nat := 40
def classB_students : Nat := 50
def total_sample : Nat := 18

-- Predicate that checks if the number of students drawn from Class A is correct
theorem students_drawn_from_class_A (students_from_A : Nat) : students_from_A = 9 :=
by
  sorry

end NUMINAMATH_GPT_students_drawn_from_class_A_l2098_209891


namespace NUMINAMATH_GPT_rectangle_to_square_area_ratio_is_24_25_l2098_209881

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_to_square_area_ratio_is_24_25_l2098_209881


namespace NUMINAMATH_GPT_factor_of_polynomial_l2098_209877

theorem factor_of_polynomial (t : ℚ) : (8 * t^2 + 17 * t - 10 = 0) ↔ (t = 5/8 ∨ t = -2) :=
by sorry

end NUMINAMATH_GPT_factor_of_polynomial_l2098_209877


namespace NUMINAMATH_GPT_champion_is_C_l2098_209809

-- Definitions of statements made by Zhang, Wang, and Li
def zhang_statement (winner : String) : Bool := winner = "A" ∨ winner = "B"
def wang_statement (winner : String) : Bool := winner ≠ "C"
def li_statement (winner : String) : Bool := winner ≠ "A" ∧ winner ≠ "B"

-- Predicate that indicates exactly one of the statements is correct
def exactly_one_correct (winner : String) : Prop :=
  (zhang_statement winner ∧ ¬wang_statement winner ∧ ¬li_statement winner) ∨
  (¬zhang_statement winner ∧ wang_statement winner ∧ ¬li_statement winner) ∨
  (¬zhang_statement winner ∧ ¬wang_statement winner ∧ li_statement winner)

-- The theorem stating the correct answer to the problem
theorem champion_is_C : (exactly_one_correct "C") :=
  by
    sorry  -- Proof goes here

-- Note: The import statement and sorry definition are included to ensure the code builds.

end NUMINAMATH_GPT_champion_is_C_l2098_209809


namespace NUMINAMATH_GPT_pencils_bought_at_cost_price_l2098_209802

variable (C S : ℝ)
variable (n : ℕ)

theorem pencils_bought_at_cost_price (h1 : n * C = 8 * S) (h2 : S = 1.5 * C) : n = 12 := 
by sorry

end NUMINAMATH_GPT_pencils_bought_at_cost_price_l2098_209802


namespace NUMINAMATH_GPT_compare_neg_rational_decimal_l2098_209889

theorem compare_neg_rational_decimal : 
  -3 / 4 > -0.8 := 
by 
  sorry

end NUMINAMATH_GPT_compare_neg_rational_decimal_l2098_209889


namespace NUMINAMATH_GPT_smallest_k_base_representation_l2098_209838

theorem smallest_k_base_representation :
  ∃ k : ℕ, (k > 0) ∧ (∀ n k, 0 = (42 * (1 - k^(n+1))/(1 - k))) ∧ (0 = (4 * (53 * (1 - k^(n+1))/(1 - k)))) →
  (k = 11) := sorry

end NUMINAMATH_GPT_smallest_k_base_representation_l2098_209838


namespace NUMINAMATH_GPT_a_perp_a_add_b_l2098_209812

def vector (α : Type*) := α × α

def a : vector ℤ := (2, -1)
def b : vector ℤ := (1, 7)

def dot_product (v1 v2 : vector ℤ) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

def add_vector (v1 v2 : vector ℤ) : vector ℤ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def perpendicular (v1 v2 : vector ℤ) : Prop :=
  dot_product v1 v2 = 0

theorem a_perp_a_add_b :
  perpendicular a (add_vector a b) :=
by {
  sorry
}

end NUMINAMATH_GPT_a_perp_a_add_b_l2098_209812


namespace NUMINAMATH_GPT_joels_age_when_dad_twice_l2098_209857

theorem joels_age_when_dad_twice
  (joel_age_now : ℕ)
  (dad_age_now : ℕ)
  (years : ℕ)
  (H1 : joel_age_now = 5)
  (H2 : dad_age_now = 32)
  (H3 : years = 22)
  (H4 : dad_age_now + years = 2 * (joel_age_now + years))
  : joel_age_now + years = 27 := 
by sorry

end NUMINAMATH_GPT_joels_age_when_dad_twice_l2098_209857


namespace NUMINAMATH_GPT_ratio_of_speeds_l2098_209837

theorem ratio_of_speeds (D H : ℕ) (h1 : D = 2 * H) (h2 : 10 * D = 20 * H) :
  10 * D = 2 * (10 * H) :=
by
  sorry

example (D H : ℕ) (h1 : D = 2 * H) (h2 : 10 * D = 20 * H) :
  10 = 10 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l2098_209837


namespace NUMINAMATH_GPT_pipe_empty_cistern_l2098_209860

theorem pipe_empty_cistern (h : 1 / 3 * t = 6) : 2 / 3 * t = 12 :=
sorry

end NUMINAMATH_GPT_pipe_empty_cistern_l2098_209860


namespace NUMINAMATH_GPT_original_population_multiple_of_5_l2098_209805

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem original_population_multiple_of_5 (x y z : ℕ) 
  (H1 : is_perfect_square (x * x)) 
  (H2 : x * x + 200 = y * y) 
  (H3 : y * y + 180 = z * z) : 
  ∃ k : ℕ, x * x = 5 * k := 
sorry

end NUMINAMATH_GPT_original_population_multiple_of_5_l2098_209805


namespace NUMINAMATH_GPT_rice_in_each_container_l2098_209883

theorem rice_in_each_container 
  (total_weight : ℚ) 
  (num_containers : ℕ)
  (conversion_factor : ℚ) 
  (equal_division : total_weight = 29 / 4 ∧ num_containers = 4 ∧ conversion_factor = 16) : 
  (total_weight / num_containers) * conversion_factor = 29 := 
by 
  sorry

end NUMINAMATH_GPT_rice_in_each_container_l2098_209883


namespace NUMINAMATH_GPT_max_product_not_less_than_993_squared_l2098_209839

theorem max_product_not_less_than_993_squared :
  ∀ (a : Fin 1985 → ℕ), 
    (∀ i, ∃ j, a j = i + 1) →  -- representation of permutation
    (∃ i : Fin 1985, i * (a i) ≥ 993 * 993) :=
by
  intros a h
  sorry

end NUMINAMATH_GPT_max_product_not_less_than_993_squared_l2098_209839


namespace NUMINAMATH_GPT_toy_cars_in_third_box_l2098_209831

theorem toy_cars_in_third_box (total_cars first_box second_box : ℕ) (H1 : total_cars = 71) 
    (H2 : first_box = 21) (H3 : second_box = 31) : total_cars - (first_box + second_box) = 19 :=
by
  sorry

end NUMINAMATH_GPT_toy_cars_in_third_box_l2098_209831


namespace NUMINAMATH_GPT_young_employees_l2098_209824

theorem young_employees (ratio_young : ℕ)
                        (ratio_middle : ℕ)
                        (ratio_elderly : ℕ)
                        (sample_selected : ℕ)
                        (prob_selection : ℚ)
                        (h_ratio : ratio_young = 10 ∧ ratio_middle = 8 ∧ ratio_elderly = 7)
                        (h_sample : sample_selected = 200)
                        (h_prob : prob_selection = 0.2) :
                        10 * (sample_selected / prob_selection) / 25 = 400 :=
by {
  sorry
}

end NUMINAMATH_GPT_young_employees_l2098_209824


namespace NUMINAMATH_GPT_phillip_remaining_amount_l2098_209868

-- Define the initial amount of money
def initial_amount : ℕ := 95

-- Define the amounts spent on various items
def amount_spent_on_oranges : ℕ := 14
def amount_spent_on_apples : ℕ := 25
def amount_spent_on_candy : ℕ := 6

-- Calculate the total amount spent
def total_spent : ℕ := amount_spent_on_oranges + amount_spent_on_apples + amount_spent_on_candy

-- Calculate the remaining amount of money
def remaining_amount : ℕ := initial_amount - total_spent

-- Statement to be proved
theorem phillip_remaining_amount : remaining_amount = 50 :=
by
  sorry

end NUMINAMATH_GPT_phillip_remaining_amount_l2098_209868


namespace NUMINAMATH_GPT_length_of_crease_l2098_209899

theorem length_of_crease (θ : ℝ) : 
  let B := 5
  let DM := 5 * (Real.tan θ)
  DM = 5 * (Real.tan θ) := 
by 
  sorry

end NUMINAMATH_GPT_length_of_crease_l2098_209899


namespace NUMINAMATH_GPT_competition_results_correct_l2098_209844

theorem competition_results_correct :
  ∃ (first second third fourth : String), 
    (first = "Oleg" ∧ second = "Olya" ∧ third = "Polya" ∧ fourth = "Pasha") ∧
    ∀ (claims : String → String → Prop),
      (claims "Olya" "all_odd_places_boys") ∧ 
      (claims "Oleg" "consecutive_places_with_olya") ∧
      (claims "Pasha" "all_odd_places_names_start_O") ∧
      ∃ (truth_teller : String), 
        truth_teller = "Oleg" ∧ 
        (claims "Oleg" "first_place") ∧ 
        ¬ (claims "Olya" "first_place") ∧ 
        ¬ (claims "Pasha" "first_place") ∧ 
        ¬ (claims "Polya" "first_place") :=
sorry

end NUMINAMATH_GPT_competition_results_correct_l2098_209844


namespace NUMINAMATH_GPT_wang_hua_withdrawal_correct_l2098_209825

noncomputable def wang_hua_withdrawal : ℤ :=
  let d : ℤ := 14
  let c : ℤ := 32
  -- The amount Wang Hua was supposed to withdraw in yuan
  (d * 100 + c)

theorem wang_hua_withdrawal_correct (d c : ℤ) :
  let initial_amount := (100 * d + c)
  let incorrect_amount := (100 * c + d)
  let amount_spent := 350
  let remaining_amount := incorrect_amount - amount_spent
  let expected_remaining := 2 * initial_amount
  remaining_amount = expected_remaining ∧ 
  d = 14 ∧ 
  c = 32 :=
by
  sorry

end NUMINAMATH_GPT_wang_hua_withdrawal_correct_l2098_209825
