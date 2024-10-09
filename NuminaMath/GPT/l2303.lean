import Mathlib

namespace nate_cooking_for_people_l2303_230362

/-- Given that 8 jumbo scallops weigh one pound, scallops cost $24.00 per pound, Nate is pairing 2 scallops with a corn bisque per person, and he spends $48 on scallops. We want to prove that Nate is cooking for 8 people. -/
theorem nate_cooking_for_people :
  (8 : ℕ) = 8 →
  (24 : ℕ) = 24 →
  (2 : ℕ) = 2 →
  (48 : ℕ) = 48 →
  let scallops_per_pound := 8
  let cost_per_pound := 24
  let scallops_per_person := 2
  let money_spent := 48
  let pounds_of_scallops := money_spent / cost_per_pound
  let total_scallops := scallops_per_pound * pounds_of_scallops
  let people := total_scallops / scallops_per_person
  people = 8 :=
by
  sorry

end nate_cooking_for_people_l2303_230362


namespace intersection_with_x_axis_intersection_with_y_axis_l2303_230358

theorem intersection_with_x_axis (x y : ℝ) : y = -2 * x + 4 ∧ y = 0 ↔ x = 2 ∧ y = 0 := by
  sorry

theorem intersection_with_y_axis (x y : ℝ) : y = -2 * x + 4 ∧ x = 0 ↔ x = 0 ∧ y = 4 := by
  sorry

end intersection_with_x_axis_intersection_with_y_axis_l2303_230358


namespace distance_between_intersections_l2303_230399

theorem distance_between_intersections (a : ℝ) (a_pos : 0 < a) : 
  |(Real.log a / Real.log 2) - (Real.log (a / 3) / Real.log 2)| = Real.log 3 / Real.log 2 :=
by
  sorry

end distance_between_intersections_l2303_230399


namespace min_value_reciprocals_l2303_230306

open Real

theorem min_value_reciprocals (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + 3 * b = 1) :
  ∃ m : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 3 * y = 1 → (1 / x + 1 / y) ≥ m) ∧ m = 4 + 2 * sqrt 3 :=
sorry

end min_value_reciprocals_l2303_230306


namespace all_values_are_equal_l2303_230317

theorem all_values_are_equal
  (f : ℤ × ℤ → ℕ)
  (h : ∀ x y : ℤ, f (x, y) * 4 = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1))
  (hf_pos : ∀ x y : ℤ, 0 < f (x, y)) : 
  ∀ x y x' y' : ℤ, f (x, y) = f (x', y') :=
by
  sorry

end all_values_are_equal_l2303_230317


namespace distinct_even_numbers_between_100_and_999_l2303_230357

def count_distinct_even_numbers_between_100_and_999 : ℕ :=
  let possible_units_digits := 5 -- {0, 2, 4, 6, 8}
  let possible_hundreds_digits := 8 -- {1, 2, ..., 9} excluding the chosen units digit
  let possible_tens_digits := 8 -- {0, 1, 2, ..., 9} excluding the chosen units and hundreds digits
  possible_units_digits * possible_hundreds_digits * possible_tens_digits

theorem distinct_even_numbers_between_100_and_999 : count_distinct_even_numbers_between_100_and_999 = 320 :=
  by sorry

end distinct_even_numbers_between_100_and_999_l2303_230357


namespace manny_remaining_money_l2303_230378

def cost_chair (cost_total_chairs : ℕ) (number_of_chairs : ℕ) : ℕ :=
  cost_total_chairs / number_of_chairs

def cost_table (cost_chair : ℕ) (chairs_for_table : ℕ) : ℕ :=
  cost_chair * chairs_for_table

def total_cost (cost_table : ℕ) (cost_chairs : ℕ) : ℕ :=
  cost_table + cost_chairs

def remaining_money (initial_amount : ℕ) (spent_amount : ℕ) : ℕ :=
  initial_amount - spent_amount

theorem manny_remaining_money : remaining_money 100 (total_cost (cost_table (cost_chair 55 5) 3) ((cost_chair 55 5) * 2)) = 45 :=
by
  sorry

end manny_remaining_money_l2303_230378


namespace estimate_expr_l2303_230392

theorem estimate_expr : 1 < (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 ∧ (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 < 2 := by
  sorry

end estimate_expr_l2303_230392


namespace third_median_length_l2303_230342

theorem third_median_length (a b: ℝ) (h_a: a = 5) (h_b: b = 8)
  (area: ℝ) (h_area: area = 6 * Real.sqrt 15) (m: ℝ):
  m = 3 * Real.sqrt 6 :=
sorry

end third_median_length_l2303_230342


namespace evaluate_expression_l2303_230387

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 :=
by
  sorry

end evaluate_expression_l2303_230387


namespace ratio_of_cards_lost_l2303_230354

-- Definitions based on the conditions
def purchases_per_week : ℕ := 20
def weeks_per_year : ℕ := 52
def cards_left : ℕ := 520

-- Main statement to be proved
theorem ratio_of_cards_lost (total_cards : ℕ := purchases_per_week * weeks_per_year)
                            (cards_lost : ℕ := total_cards - cards_left) :
                            (cards_lost : ℚ) / total_cards = 1 / 2 :=
by
  sorry

end ratio_of_cards_lost_l2303_230354


namespace man_climbs_out_of_well_in_65_days_l2303_230373

theorem man_climbs_out_of_well_in_65_days (depth climb slip net_days last_climb : ℕ) 
  (h_depth : depth = 70)
  (h_climb : climb = 6)
  (h_slip : slip = 5)
  (h_net_days : net_days = 64)
  (h_last_climb : last_climb = 1) :
  ∃ days : ℕ, days = net_days + last_climb ∧ days = 65 := by
  sorry

end man_climbs_out_of_well_in_65_days_l2303_230373


namespace eval_exp_l2303_230353

theorem eval_exp {a b : ℝ} (h : a = 3^4) : a^(5/4) = 243 :=
by
  sorry

end eval_exp_l2303_230353


namespace magician_guarantee_success_l2303_230316

-- Definitions based on the conditions in part a).
def deck_size : ℕ := 52

def is_edge_position (position : ℕ) : Prop :=
  position = 0 ∨ position = deck_size - 1

-- Statement of the proof problem in part c).
theorem magician_guarantee_success (position : ℕ) : is_edge_position position ↔ 
  forall spectator_strategy : ℕ → ℕ, 
  exists magician_strategy : (ℕ → ℕ → ℕ), 
  forall t : ℕ, t = position →
  (∃ k : ℕ, t = magician_strategy k (spectator_strategy k)) :=
sorry

end magician_guarantee_success_l2303_230316


namespace factorial_sum_mod_30_l2303_230367

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map factorial |>.sum

def remainder_when_divided_by (m k : Nat) : Nat :=
  m % k

theorem factorial_sum_mod_30 : remainder_when_divided_by (sum_of_factorials 100) 30 = 3 :=
by
  sorry

end factorial_sum_mod_30_l2303_230367


namespace assorted_candies_count_l2303_230351

theorem assorted_candies_count
  (total_candies : ℕ)
  (chewing_gums : ℕ)
  (chocolate_bars : ℕ)
  (assorted_candies : ℕ) :
  total_candies = 50 →
  chewing_gums = 15 →
  chocolate_bars = 20 →
  assorted_candies = total_candies - (chewing_gums + chocolate_bars) →
  assorted_candies = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end assorted_candies_count_l2303_230351


namespace find_n_l2303_230359

theorem find_n (n : ℕ) : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ n → n = 29 := by
  intros h
  sorry

end find_n_l2303_230359


namespace smallest_integer_remainder_l2303_230350

theorem smallest_integer_remainder (b : ℕ) :
  (b ≡ 2 [MOD 3]) ∧ (b ≡ 3 [MOD 5]) → b = 8 := 
by
  sorry

end smallest_integer_remainder_l2303_230350


namespace fraction_addition_l2303_230375

theorem fraction_addition :
  (1 / 6) + (1 / 3) + (5 / 9) = 19 / 18 :=
by
  sorry

end fraction_addition_l2303_230375


namespace find_amount_l2303_230365

-- Given conditions
variables (x A : ℝ)

theorem find_amount :
  (0.65 * x = 0.20 * A) → (x = 190) → (A = 617.5) :=
by
  intros h1 h2
  sorry

end find_amount_l2303_230365


namespace find_uncommon_cards_l2303_230369

def numRare : ℕ := 19
def numCommon : ℕ := 30
def costRare : ℝ := 1
def costUncommon : ℝ := 0.50
def costCommon : ℝ := 0.25
def totalCostDeck : ℝ := 32

theorem find_uncommon_cards (U : ℕ) (h : U * costUncommon + numRare * costRare + numCommon * costCommon = totalCostDeck) : U = 11 := by
  sorry

end find_uncommon_cards_l2303_230369


namespace expected_value_of_fair_6_sided_die_l2303_230370

noncomputable def fair_die_expected_value : ℝ :=
  (1/6) * 1 + (1/6) * 2 + (1/6) * 3 + (1/6) * 4 + (1/6) * 5 + (1/6) * 6

theorem expected_value_of_fair_6_sided_die : fair_die_expected_value = 3.5 := by
  sorry

end expected_value_of_fair_6_sided_die_l2303_230370


namespace elastic_collision_ball_speed_l2303_230327

open Real

noncomputable def final_ball_speed (v_car v_ball : ℝ) : ℝ :=
  let relative_speed := v_ball + v_car
  relative_speed + v_car

theorem elastic_collision_ball_speed :
  let v_car := 5
  let v_ball := 6
  final_ball_speed v_car v_ball = 16 := 
by
  sorry

end elastic_collision_ball_speed_l2303_230327


namespace tan_pink_violet_probability_l2303_230319

noncomputable def probability_tan_pink_violet_consecutive_order : ℚ :=
  let num_ways := (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 5)
  let total_ways := Nat.factorial 12
  num_ways / total_ways

theorem tan_pink_violet_probability :
  probability_tan_pink_violet_consecutive_order = 1 / 27720 := by
  sorry

end tan_pink_violet_probability_l2303_230319


namespace vampire_daily_blood_suction_l2303_230323

-- Conditions from the problem
def vampire_bl_need_per_week : ℕ := 7  -- gallons of blood per week
def blood_per_person_in_pints : ℕ := 2  -- pints of blood per person
def pints_per_gallon : ℕ := 8            -- pints in 1 gallon

-- Theorem statement to prove
theorem vampire_daily_blood_suction :
  let daily_requirement_in_gallons : ℕ := vampire_bl_need_per_week / 7   -- gallons per day
  let daily_requirement_in_pints : ℕ := daily_requirement_in_gallons * pints_per_gallon
  let num_people_needed_per_day : ℕ := daily_requirement_in_pints / blood_per_person_in_pints
  num_people_needed_per_day = 4 :=
by
  sorry

end vampire_daily_blood_suction_l2303_230323


namespace total_CDs_in_stores_l2303_230302

def shelvesA := 5
def racksPerShelfA := 7
def cdsPerRackA := 8

def shelvesB := 4
def racksPerShelfB := 6
def cdsPerRackB := 7

def totalCDsA := shelvesA * racksPerShelfA * cdsPerRackA
def totalCDsB := shelvesB * racksPerShelfB * cdsPerRackB

def totalCDs := totalCDsA + totalCDsB

theorem total_CDs_in_stores :
  totalCDs = 448 := 
by 
  sorry

end total_CDs_in_stores_l2303_230302


namespace lines_intersect_at_l2303_230377

def Line1 (t : ℝ) : ℝ × ℝ :=
  let x := 1 + 3 * t
  let y := 2 - t
  (x, y)

def Line2 (u : ℝ) : ℝ × ℝ :=
  let x := -1 + 4 * u
  let y := 4 + 3 * u
  (x, y)

theorem lines_intersect_at :
  ∃ t u : ℝ, Line1 t = Line2 u ∧
             Line1 t = (-53 / 17, 56 / 17) :=
by
  sorry

end lines_intersect_at_l2303_230377


namespace range_of_k_l2303_230391

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → k * (Real.exp (k * x) + 1) - ((1 / x) + 1) * Real.log x > 0) ↔ k > 1 / Real.exp 1 := 
  sorry

end range_of_k_l2303_230391


namespace find_y_l2303_230345

theorem find_y (y : ℝ) (h : (8 + 15 + 22 + 5 + y) / 5 = 12) : y = 10 :=
by
  -- the proof is skipped
  sorry

end find_y_l2303_230345


namespace faith_change_l2303_230356

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def two_twenty_bills : ℕ := 2 * 20
def loose_coins : ℕ := 3
def total_cost : ℕ := flour_cost + cake_stand_cost
def total_given : ℕ := two_twenty_bills + loose_coins
def change : ℕ := total_given - total_cost

theorem faith_change : change = 10 := by
  sorry

end faith_change_l2303_230356


namespace min_value_x_2y_l2303_230385

theorem min_value_x_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2*y + 2*x*y = 8) : x + 2*y ≥ 4 :=
sorry

end min_value_x_2y_l2303_230385


namespace exists_infinitely_many_m_l2303_230330

theorem exists_infinitely_many_m (k : ℕ) (hk : 0 < k) : 
  ∃ᶠ m in at_top, 3 ^ k ∣ m ^ 3 + 10 :=
sorry

end exists_infinitely_many_m_l2303_230330


namespace system_of_equations_solution_l2303_230388

theorem system_of_equations_solution (x y z : ℝ) (h1 : x + y = 1) (h2 : x + z = 0) (h3 : y + z = -1) : 
    x = 1 ∧ y = 0 ∧ z = -1 := 
by 
  sorry

end system_of_equations_solution_l2303_230388


namespace line_does_not_pass_through_second_quadrant_l2303_230329

theorem line_does_not_pass_through_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ x - y - a^2 = 0 := 
by
  sorry

end line_does_not_pass_through_second_quadrant_l2303_230329


namespace find_ABC_l2303_230304

variables (A B C D : ℕ)

-- Conditions
def non_zero_distinct_digits_less_than_7 : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A < 7 ∧ B < 7 ∧ C < 7 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

def ab_c_seven : Prop := 
  (A * 7 + B) + C = C * 7

def ab_ba_dc_seven : Prop :=
  (A * 7 + B) + (B * 7 + A) = D * 7 + C

-- Theorem to prove
theorem find_ABC 
  (h1 : non_zero_distinct_digits_less_than_7 A B C) 
  (h2 : ab_c_seven A B C) 
  (h3 : ab_ba_dc_seven A B C D) : 
  A * 100 + B * 10 + C = 516 :=
sorry

end find_ABC_l2303_230304


namespace num_pos_four_digit_integers_l2303_230314

theorem num_pos_four_digit_integers : 
  ∃ (n : ℕ), n = (Nat.factorial 4) / ((Nat.factorial 3) * (Nat.factorial 1)) ∧ n = 4 := 
by
  sorry

end num_pos_four_digit_integers_l2303_230314


namespace discs_angular_velocity_relation_l2303_230389

variables {r1 r2 ω1 ω2 : ℝ} -- Radii and angular velocities

-- Conditions:
-- Discs have radii r1 and r2, and angular velocities ω1 and ω2, respectively.
-- Discs come to a halt after being brought into contact via friction.
-- Discs have identical thickness and are made of the same material.
-- Prove the required relation.

theorem discs_angular_velocity_relation
  (h1 : r1 > 0)
  (h2 : r2 > 0)
  (halt_contact : ω1 * r1^3 = ω2 * r2^3) :
  ω1 * r1^3 = ω2 * r2^3 :=
sorry

end discs_angular_velocity_relation_l2303_230389


namespace ratio_friday_to_monday_l2303_230326

-- Definitions from conditions
def rabbits : ℕ := 16
def monday_toys : ℕ := 6
def wednesday_toys : ℕ := 2 * monday_toys
def saturday_toys : ℕ := wednesday_toys / 2
def total_toys : ℕ := 3 * rabbits

-- Definition to represent the number of toys bought on Friday
def friday_toys : ℕ := total_toys - (monday_toys + wednesday_toys + saturday_toys)

-- Theorem to prove the ratio is 4:1
theorem ratio_friday_to_monday : friday_toys / monday_toys = 4 := by
  -- Placeholder for the proof
  sorry

end ratio_friday_to_monday_l2303_230326


namespace equation_of_rotated_translated_line_l2303_230318

theorem equation_of_rotated_translated_line (x y : ℝ) :
  (∀ x, y = 3 * x → y = x / -3 + 1 / -3) →
  (∀ x, y = -1/3 * (x - 1)) →
  y = -1/3 * x + 1/3 :=
sorry

end equation_of_rotated_translated_line_l2303_230318


namespace petes_average_speed_l2303_230333

theorem petes_average_speed
    (map_distance : ℝ := 5) 
    (time_taken : ℝ := 1.5) 
    (map_scale : ℝ := 0.05555555555555555) :
    (map_distance / map_scale) / time_taken = 60 := 
by
    sorry

end petes_average_speed_l2303_230333


namespace trigonometric_identity_l2303_230331

open Real

theorem trigonometric_identity :
  (sin (20 * π / 180) * sin (80 * π / 180) - cos (160 * π / 180) * sin (10 * π / 180) = 1 / 2) :=
by
  -- Trigonometric calculations
  sorry

end trigonometric_identity_l2303_230331


namespace substitutions_made_in_first_half_l2303_230334

-- Definitions based on given problem conditions
def total_players : ℕ := 24
def starters : ℕ := 11
def non_players : ℕ := 7
def first_half_substitutions (S : ℕ) : ℕ := S
def second_half_substitutions (S : ℕ) : ℕ := 2 * S
def total_players_played (S : ℕ) := starters + first_half_substitutions S + second_half_substitutions S
def remaining_players : ℕ := total_players - non_players

-- Proof problem statement
theorem substitutions_made_in_first_half (S : ℕ) (h : total_players_played S = remaining_players) : S = 2 :=
by
  sorry

end substitutions_made_in_first_half_l2303_230334


namespace solve_for_x_l2303_230308

theorem solve_for_x : 
  (∃ x : ℝ, (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 4 * x - 21) 
  ∧ x = 4.5) := by
{
  sorry
}

end solve_for_x_l2303_230308


namespace find_k_intersection_on_line_l2303_230384

theorem find_k_intersection_on_line (k : ℝ) :
  (∃ (x y : ℝ), x - 2 * y - 2 * k = 0 ∧ 2 * x - 3 * y - k = 0 ∧ 3 * x - y = 0) → k = 0 :=
by
  sorry

end find_k_intersection_on_line_l2303_230384


namespace emily_lives_total_l2303_230376

variable (x : ℤ)

def total_lives_after_stages (x : ℤ) : ℤ :=
  let lives_after_stage1 := x + 25
  let lives_after_stage2 := lives_after_stage1 + 24
  let lives_after_stage3 := lives_after_stage2 + 15
  lives_after_stage3

theorem emily_lives_total : total_lives_after_stages x = x + 64 := by
  -- The proof will go here
  sorry

end emily_lives_total_l2303_230376


namespace total_amount_in_bank_l2303_230397

-- Definition of the checks and their values
def checks_1mil : Nat := 25
def checks_100k : Nat := 8
def value_1mil : Nat := 1000000
def value_100k : Nat := 100000

-- The proof statement
theorem total_amount_in_bank 
  (total : Nat) 
  (h1 : checks_1mil * value_1mil = 25000000)
  (h2 : checks_100k * value_100k = 800000):
  total = 25000000 + 800000 :=
sorry

end total_amount_in_bank_l2303_230397


namespace william_time_on_road_l2303_230310

-- Define departure and arrival times
def departure_time := 7 -- 7:00 AM
def arrival_time := 20 -- 8:00 PM in 24-hour format

-- Define stop times in minutes
def stop1 := 25
def stop2 := 10
def stop3 := 25

-- Define total journey time in hours
def total_travel_time := arrival_time - departure_time

-- Define total stop time in hours
def total_stop_time := (stop1 + stop2 + stop3) / 60

-- Define time spent on the road
def time_on_road := total_travel_time - total_stop_time

-- The theorem to prove
theorem william_time_on_road : time_on_road = 12 := by
  sorry

end william_time_on_road_l2303_230310


namespace fraction_black_part_l2303_230360

theorem fraction_black_part (L : ℝ) (blue_part : ℝ) (white_part_fraction : ℝ) 
  (h1 : L = 8) (h2 : blue_part = 3.5) (h3 : white_part_fraction = 0.5) : 
  (8 - (3.5 + 0.5 * (8 - 3.5))) / 8 = 9 / 32 :=
by
  sorry

end fraction_black_part_l2303_230360


namespace total_accessories_correct_l2303_230311

-- Definitions
def dresses_first_period := 10 * 4
def dresses_second_period := 3 * 5
def total_dresses := dresses_first_period + dresses_second_period
def accessories_per_dress := 3 + 2 + 1
def total_accessories := total_dresses * accessories_per_dress

-- Theorem statement
theorem total_accessories_correct : total_accessories = 330 := by
  sorry

end total_accessories_correct_l2303_230311


namespace asher_speed_l2303_230340

theorem asher_speed :
  (5 * 60 ≠ 0) → (6600 / (5 * 60) = 22) :=
by
  intros h
  sorry

end asher_speed_l2303_230340


namespace intersection_correct_l2303_230393

variable (x : ℝ)

def M : Set ℝ := { x | x^2 > 4 }
def N : Set ℝ := { x | x^2 - 3 * x ≤ 0 }
def NM_intersection : Set ℝ := { x | 2 < x ∧ x ≤ 3 }

theorem intersection_correct :
  {x | (M x) ∧ (N x)} = NM_intersection :=
sorry

end intersection_correct_l2303_230393


namespace set_intersection_l2303_230349

def setM : Set ℝ := {x | x^2 - 1 < 0}
def setN : Set ℝ := {y | ∃ x ∈ setM, y = Real.log (x + 2)}

theorem set_intersection : setM ∩ setN = {y | 0 < y ∧ y < Real.log 3} :=
by
  sorry

end set_intersection_l2303_230349


namespace harkamal_grapes_purchase_l2303_230382

-- Define the conditions as parameters and constants
def cost_per_kg_grapes := 70
def kg_mangoes := 9
def cost_per_kg_mangoes := 45
def total_payment := 965

-- The theorem stating Harkamal purchased 8 kg of grapes
theorem harkamal_grapes_purchase : 
  ∃ G : ℕ, (cost_per_kg_grapes * G + cost_per_kg_mangoes * kg_mangoes = total_payment) ∧ G = 8 :=
by
  use 8
  unfold cost_per_kg_grapes cost_per_kg_mangoes kg_mangoes total_payment
  show 70 * 8 + 45 * 9 = 965 ∧ 8 = 8
  sorry

end harkamal_grapes_purchase_l2303_230382


namespace verify_chebyshev_polynomials_l2303_230395

-- Define the Chebyshev polynomials of the first kind Tₙ(x)
def T : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => x
| (n+1), x => 2 * x * T n x - T (n-1) x

-- Define the Chebyshev polynomials of the second kind Uₙ(x)
def U : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => 2 * x
| (n+1), x => 2 * x * U n x - U (n-1) x

-- State the theorem to verify the Chebyshev polynomials initial conditions and recurrence relations
theorem verify_chebyshev_polynomials (n : ℕ) (x : ℝ) :
  T 0 x = 1 ∧ T 1 x = x ∧
  U 0 x = 1 ∧ U 1 x = 2 * x ∧
  (T (n+1) x = 2 * x * T n x - T (n-1) x) ∧
  (U (n+1) x = 2 * x * U n x - U (n-1) x) := sorry

end verify_chebyshev_polynomials_l2303_230395


namespace solve_for_x_l2303_230372

theorem solve_for_x (x : ℝ) (h : (x / 4) / 2 = 4 / (x / 2)) : x = 8 ∨ x = -8 :=
by
  sorry

end solve_for_x_l2303_230372


namespace factorize_polynomial_l2303_230361

theorem factorize_polynomial (x y : ℝ) : 2 * x^3 - 8 * x^2 * y + 8 * x * y^2 = 2 * x * (x - 2 * y) ^ 2 := 
by sorry

end factorize_polynomial_l2303_230361


namespace min_area_circle_tangent_l2303_230386

theorem min_area_circle_tangent (h : ∀ (x : ℝ), x > 0 → y = 2 / x) : 
  ∃ (a b r : ℝ), (∀ (x : ℝ), x > 0 → 2 * a + b = 2 + 2 / x) ∧
  (∀ (x : ℝ), x > 0 → (x - 1)^2 + (y - 2)^2 = 5) :=
sorry

end min_area_circle_tangent_l2303_230386


namespace unique_hexagon_angles_sides_identity_1_identity_2_l2303_230366

noncomputable def lengths_angles_determined 
  (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₀ : α + β + γ < 180) : Prop :=
  -- Assuming this is the expression we need to handle:
  ∀ (δ ε ζ : ℝ),
    δ = 180 - α ∧
    ε = 180 - β ∧
    ζ = 180 - γ →
  ∃ (angles_determined : Prop),
    angles_determined

theorem unique_hexagon_angles_sides 
  (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₀ : α + β + γ < 180) : 
  lengths_angles_determined a b c d e f α β γ h₀ :=
sorry

theorem identity_1 
  (a b c d : ℝ) 
  (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) : 
  (1 / a + 1 / c = 1 / b + 1 / d) :=
sorry

theorem identity_2 
  (a b c d e f : ℝ) 
  (h₀ : true) : 
  ((a + f) * (b + d) * (c + e) = (a + e) * (b + f) * (c + d)) :=
sorry

end unique_hexagon_angles_sides_identity_1_identity_2_l2303_230366


namespace members_not_in_A_nor_B_l2303_230398

variable (U A B : Finset ℕ) -- We define the sets as finite sets of natural numbers.
variable (hU_size : U.card = 190) -- Size of set U is 190.
variable (hB_size : (U ∩ B).card = 49) -- 49 items are in set B.
variable (hAB_size : (A ∩ U ∩ B).card = 23) -- 23 items are in both A and B.
variable (hA_size : (U ∩ A).card = 105) -- 105 items are in set A.

theorem members_not_in_A_nor_B :
  (U \ (A ∪ B)).card = 59 := sorry

end members_not_in_A_nor_B_l2303_230398


namespace largest_digit_B_divisible_by_3_l2303_230312

-- Define the six-digit number form and the known digits sum.
def isIntegerDivisibleBy3 (b : ℕ) : Prop :=
  b < 10 ∧ (b + 30) % 3 = 0

-- The main theorem: Find the largest digit B such that the number 4B5,894 is divisible by 3.
theorem largest_digit_B_divisible_by_3 : ∃ (B : ℕ), isIntegerDivisibleBy3 B ∧ ∀ (b' : ℕ), isIntegerDivisibleBy3 b' → b' ≤ B := by
  -- Notice the existential and universal quantifiers involved in finding the largest B.
  sorry

end largest_digit_B_divisible_by_3_l2303_230312


namespace range_of_m_if_p_and_q_true_range_of_t_if_q_necessary_for_s_l2303_230300

/-- There exists a real number x such that 2x^2 + (m-1)x + 1/2 ≤ 0 -/
def proposition_p (m : ℝ) : Prop :=
  ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1 / 2 ≤ 0

/-- The curve C1: x^2/m^2 + y^2/(2m+8) = 1 represents an ellipse with foci on the x-axis -/
def proposition_q (m : ℝ) : Prop :=
  m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0

/-- The curve C2: x^2/(m-t) + y^2/(m-t-1) = 1 represents a hyperbola -/
def proposition_s (m t : ℝ) : Prop :=
  (m - t) * (m - t - 1) < 0

/-- Find the range of values for m if p and q are true -/
theorem range_of_m_if_p_and_q_true (m : ℝ) :
  proposition_p m ∧ proposition_q m ↔ (-4 < m ∧ m < -2) ∨ m > 4 :=
  sorry

/-- Find the range of values for t if q is a necessary but not sufficient condition for s -/
theorem range_of_t_if_q_necessary_for_s (m t : ℝ) :
  (∀ m, proposition_q m → proposition_s m t) ∧ ¬(proposition_s m t → proposition_q m) ↔ 
  (-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4 :=
  sorry

end range_of_m_if_p_and_q_true_range_of_t_if_q_necessary_for_s_l2303_230300


namespace proof_problem_l2303_230313

def otimes (a b : ℕ) : ℕ := (a^2 - b) / (a - b)

theorem proof_problem : otimes (otimes 7 5) 2 = 24 := by
  sorry

end proof_problem_l2303_230313


namespace total_weight_of_fish_l2303_230390

-- Define the weights of fish caught by Peter, Ali, and Joey.
variables (P A J : ℕ)

-- Ali caught twice as much fish as Peter.
def condition1 := A = 2 * P

-- Joey caught 1 kg more fish than Peter.
def condition2 := J = P + 1

-- Ali caught 12 kg of fish.
def condition3 := A = 12

-- Prove the total weight of the fish caught by all three is 25 kg.
theorem total_weight_of_fish :
  condition1 P A → condition2 P J → condition3 A → P + A + J = 25 :=
by
  intros h1 h2 h3
  sorry

end total_weight_of_fish_l2303_230390


namespace pump_no_leak_fill_time_l2303_230341

noncomputable def pump_fill_time (P t l : ℝ) :=
  1 / P - 1 / l = 1 / t

theorem pump_no_leak_fill_time :
  ∃ P : ℝ, pump_fill_time P (13 / 6) 26 ∧ P = 2 :=
by
  sorry

end pump_no_leak_fill_time_l2303_230341


namespace minimum_value_F_l2303_230355

noncomputable def minimum_value_condition (x y : ℝ) : Prop :=
  x^2 + y^2 + 25 = 10 * (x + y)

noncomputable def F (x y : ℝ) : ℝ :=
  6 * y + 8 * x - 9

theorem minimum_value_F :
  (∃ x y : ℝ, minimum_value_condition x y) → ∃ x y : ℝ, minimum_value_condition x y ∧ F x y = 11 :=
sorry

end minimum_value_F_l2303_230355


namespace scientific_notation_142000_l2303_230320

theorem scientific_notation_142000 : (142000 : ℝ) = 1.42 * 10^5 := sorry

end scientific_notation_142000_l2303_230320


namespace ThreePowerTowerIsLarger_l2303_230324

-- original power tower definitions
def A : ℕ := 3^(3^(3^3))
def B : ℕ := 2^(2^(2^(2^2)))

-- reduced forms given from the conditions
def reducedA : ℕ := 3^(3^27)
def reducedB : ℕ := 2^(2^16)

theorem ThreePowerTowerIsLarger : reducedA > reducedB := by
  sorry

end ThreePowerTowerIsLarger_l2303_230324


namespace max_min_of_f_in_M_l2303_230363

noncomputable def domain (x : ℝ) : Prop := 3 - 4*x + x^2 > 0

def M : Set ℝ := { x | domain x }

noncomputable def f (x : ℝ) : ℝ := 2^(x+2) - 3 * 4^x

theorem max_min_of_f_in_M :
  ∃ (xₘ xₘₐₓ : ℝ), xₘ ∈ M ∧ xₘₐₓ ∈ M ∧ 
  (∀ x ∈ M, f xₘₐₓ ≥ f x) ∧ 
  (∀ x ∈ M, f xₘ ≠ f xₓₐₓ) :=
by
  sorry

end max_min_of_f_in_M_l2303_230363


namespace number_of_hockey_players_l2303_230371

theorem number_of_hockey_players 
  (cricket_players : ℕ) 
  (football_players : ℕ) 
  (softball_players : ℕ) 
  (total_players : ℕ) 
  (hockey_players : ℕ) 
  (h1 : cricket_players = 10) 
  (h2 : football_players = 16) 
  (h3 : softball_players = 13) 
  (h4 : total_players = 51) 
  (calculation : hockey_players = total_players - (cricket_players + football_players + softball_players)) : 
  hockey_players = 12 :=
by 
  rw [h1, h2, h3, h4] at calculation
  exact calculation

end number_of_hockey_players_l2303_230371


namespace abs_diff_61st_terms_l2303_230368

noncomputable def seq_C (n : ℕ) : ℤ := 20 + 15 * (n - 1)
noncomputable def seq_D (n : ℕ) : ℤ := 20 - 15 * (n - 1)

theorem abs_diff_61st_terms :
  |seq_C 61 - seq_D 61| = 1800 := sorry

end abs_diff_61st_terms_l2303_230368


namespace find_unit_prices_minimal_cost_l2303_230381

-- Definitions for part 1
def unitPrices (x y : ℕ) : Prop :=
  20 * x + 30 * y = 2920 ∧ x - y = 11 

-- Definitions for part 2
def costFunction (m : ℕ) : ℕ :=
  52 * m + 48 * (40 - m)

def additionalPurchase (m : ℕ) : Prop :=
  m ≥ 40 / 3

-- Statement for unit prices proof
theorem find_unit_prices (x y : ℕ) (h1 : 20 * x + 30 * y = 2920) (h2 : x - y = 11) : x = 65 ∧ y = 54 := 
  sorry

-- Statement for minimal cost proof
theorem minimal_cost (m : ℕ) (x y : ℕ) 
  (hx : 20 * x + 30 * y = 2920) 
  (hy : x - y = 11)
  (hx_65 : x = 65)
  (hy_54 : y = 54)
  (hm : m ≥ 40 / 3) : 
  costFunction m = 1976 ∧ m = 14 :=
  sorry

end find_unit_prices_minimal_cost_l2303_230381


namespace sum_of_b_values_l2303_230325

theorem sum_of_b_values :
  let discriminant (b : ℝ) := (b + 6) ^ 2 - 4 * 3 * 12
  ∃ b1 b2 : ℝ, discriminant b1 = 0 ∧ discriminant b2 = 0 ∧ b1 + b2 = -12 :=
by sorry

end sum_of_b_values_l2303_230325


namespace solve_system_of_equations_simplify_expression_l2303_230352

-- Statement for system of equations
theorem solve_system_of_equations (s t : ℚ) 
  (h1 : 2 * s + 3 * t = 2) 
  (h2 : 2 * s - 6 * t = -1) :
  s = 1 / 2 ∧ t = 1 / 3 :=
sorry

-- Statement for simplifying the expression
theorem simplify_expression (x y : ℚ) :
  ((x - y)^2 + (x + y) * (x - y)) / (2 * x) = x - y :=
sorry

end solve_system_of_equations_simplify_expression_l2303_230352


namespace sum_of_possible_b_values_l2303_230339

noncomputable def g (x b : ℝ) : ℝ := x^2 - b * x + 3 * b

theorem sum_of_possible_b_values :
  (∀ (x₀ x₁ : ℝ), g x₀ x₁ = 0 → g x₀ x₁ = (x₀ - x₁) * (x₀ - 3)) → ∃ b : ℝ, b = 12 ∨ b = 16 :=
sorry

end sum_of_possible_b_values_l2303_230339


namespace cubical_tank_water_volume_l2303_230322

theorem cubical_tank_water_volume 
    (s : ℝ) -- side length of the cube in feet
    (h_fill : 1 / 4 * s = 1) -- tank is filled to 0.25 of its capacity, water level is 1 foot
    (h_volume_water : 0.25 * (s ^ 3) = 16) -- 0.25 of the tank's total volume is the volume of water
    : s ^ 3 = 64 := 
by
  sorry

end cubical_tank_water_volume_l2303_230322


namespace minimum_value_1_minimum_value_2_l2303_230379

noncomputable section

open Real -- Use the real numbers

theorem minimum_value_1 (x y z : ℝ) (h : x - 2 * y + z = 4) : x^2 + y^2 + z^2 >= 8 / 3 :=
by
  sorry  -- Proof omitted
 
theorem minimum_value_2 (x y z : ℝ) (h : x - 2 * y + z = 4) : x^2 + (y - 1)^2 + z^2 >= 6 :=
by
  sorry  -- Proof omitted

end minimum_value_1_minimum_value_2_l2303_230379


namespace tangent_line_at_e_l2303_230344

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_at_e : ∀ x y : ℝ, (x = Real.exp 1) → (y = f x) → (y = 2 * x - Real.exp 1) :=
by
  intros x y hx hy
  sorry

end tangent_line_at_e_l2303_230344


namespace four_consecutive_numbers_l2303_230309

theorem four_consecutive_numbers (numbers : List ℝ) (h_distinct : numbers.Nodup) (h_length : numbers.length = 100) :
  ∃ (a b c d : ℝ) (h_seq : ([a, b, c, d] ∈ numbers.cyclicPermutations)), b + c < a + d :=
by
  sorry

end four_consecutive_numbers_l2303_230309


namespace carrie_pays_199_27_l2303_230383

noncomputable def carrie_payment : ℝ :=
  let shirts := 8 * 12
  let pants := 4 * 25
  let jackets := 4 * 75
  let skirts := 3 * 30
  let shoes := 2 * 50
  let shirts_discount := 0.20 * shirts
  let jackets_discount := 0.20 * jackets
  let skirts_discount := 0.10 * skirts
  let total_cost := shirts + pants + jackets + skirts + shoes
  let discounted_cost := (shirts - shirts_discount) + (pants) + (jackets - jackets_discount) + (skirts - skirts_discount) + shoes
  let mom_payment := 2 / 3 * discounted_cost
  let carrie_payment := discounted_cost - mom_payment
  carrie_payment

theorem carrie_pays_199_27 : carrie_payment = 199.27 :=
by
  sorry

end carrie_pays_199_27_l2303_230383


namespace find_tangent_line_equation_l2303_230364

noncomputable def tangent_line_equation (f : ℝ → ℝ) (perp_line : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  let y₀ := f x₀
  let slope_perp_to_tangent := -2
  let slope_tangent := -1 / 2
  slope_perp_to_tangent = -1 / (deriv f x₀) ∧
  x₀ = 1 ∧ y₀ = 1 ∧
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3

theorem find_tangent_line_equation :
  tangent_line_equation (fun (x : ℝ) => Real.sqrt x) (fun (x : ℝ) => -2 * x - 4) 1 := by
  sorry

end find_tangent_line_equation_l2303_230364


namespace friend_jogging_time_l2303_230374

theorem friend_jogging_time (D : ℝ) (my_time : ℝ) (friend_speed : ℝ) :
  my_time = 3 * 60 →
  friend_speed = 2 * (D / my_time) →
  (D / friend_speed) = 90 :=
by
  sorry

end friend_jogging_time_l2303_230374


namespace geometric_sequence_common_ratio_l2303_230394

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : S 2 = 2 * a 2 + 3)
  (h2 : S 3 = 2 * a 3 + 3)
  (h3 : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) : q = 2 := 
by
  sorry

end geometric_sequence_common_ratio_l2303_230394


namespace angles_sum_correct_l2303_230332

-- Definitions from the problem conditions
def identicalSquares (n : Nat) := n = 13

variable (α β γ δ ε ζ η θ : ℝ) -- Angles of interest

def anglesSum :=
  (α + β + γ + δ) + (ε + ζ + η + θ)

-- Lean 4 statement
theorem angles_sum_correct
  (h₁ : identicalSquares 13)
  (h₂ : α = 90) (h₃ : β = 90) (h₄ : γ = 90) (h₅ : δ = 90)
  (h₆ : ε = 90) (h₇ : ζ = 90) (h₈ : η = 45) (h₉ : θ = 45) :
  anglesSum α β γ δ ε ζ η θ = 405 :=
by
  simp [anglesSum]
  sorry

end angles_sum_correct_l2303_230332


namespace cube_splitting_height_l2303_230337

/-- If we split a cube with an edge of 1 meter into small cubes with an edge of 1 millimeter,
what will be the height of a column formed by stacking all the small cubes one on top of another? -/
theorem cube_splitting_height :
  let edge_meter := 1
  let edge_mm := 1000
  let num_cubes := (edge_meter * edge_mm) ^ 3
  let height_mm := num_cubes * edge_mm
  let height_km := height_mm / (1000 * 1000 * 1000)
  height_km = 1000 :=
by
  sorry

end cube_splitting_height_l2303_230337


namespace find_a_and_b_find_monotonic_intervals_and_extreme_values_l2303_230305

-- Definitions and conditions
def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

def takes_extreme_values (f : ℝ → ℝ) (a b c : ℝ) : Prop := 
  ∃ x₁ x₂, x₁ = 1 ∧ x₂ = -2/3 ∧ 3*x₁^2 + 2*a*x₁ + b = 0 ∧ 3*x₂^2 + 2*a*x₂ + b = 0

def f_at_specific_point (f : ℝ → ℝ) (x v : ℝ) : Prop :=
  f x = v

theorem find_a_and_b (a b c : ℝ) :
  takes_extreme_values (f a b c) a b c →
  a = -1/2 ∧ b = -2 :=
sorry

theorem find_monotonic_intervals_and_extreme_values (a b c : ℝ) :
  takes_extreme_values (f a b c) a b c →
  f_at_specific_point (f a b c) (-1) (3/2) →
  c = 1 ∧ 
  (∀ x, x < -2/3 ∨ x > 1 → deriv (f a b c) x > 0) ∧
  (∀ x, -2/3 < x ∧ x < 1 → deriv (f a b c) x < 0) ∧
  f a b c (-2/3) = 49/27 ∧ 
  f a b c 1 = -1/2 :=
sorry

end find_a_and_b_find_monotonic_intervals_and_extreme_values_l2303_230305


namespace value_of_abcg_defh_l2303_230307

theorem value_of_abcg_defh
  (a b c d e f g h: ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6)
  (h6 : f / g = 5 / 2)
  (h7 : g / h = 3 / 4) :
  abcg / defh = 5 / 48 :=
by
  sorry

end value_of_abcg_defh_l2303_230307


namespace M_inter_N_eq_l2303_230336

def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_inter_N_eq : (M ∩ N) = Set.Ico 0 (1/2) := 
by
  sorry

end M_inter_N_eq_l2303_230336


namespace non_congruent_non_square_rectangles_count_l2303_230303

theorem non_congruent_non_square_rectangles_count :
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ), x ∈ S → 2 * (x.1 + x.2) = 80) ∧
    S.card = 19 ∧
    (∀ (x : ℕ × ℕ), x ∈ S → x.1 ≠ x.2) ∧
    (∀ (x y : ℕ × ℕ), x ∈ S → y ∈ S → x ≠ y → x.1 = y.2 → x.2 = y.1) :=
sorry

end non_congruent_non_square_rectangles_count_l2303_230303


namespace smallest_n_exists_l2303_230347

theorem smallest_n_exists (G : Type) [Fintype G] [DecidableEq G] (connected : G → G → Prop)
  (distinct_naturals : G → ℕ) :
  (∀ a b : G, ¬ connected a b → gcd (distinct_naturals a + distinct_naturals b) 15 = 1) ∧
  (∀ a b : G, connected a b → gcd (distinct_naturals a + distinct_naturals b) 15 > 1) →
  (∀ n : ℕ, 
    (∀ a b : G, ¬ connected a b → gcd (distinct_naturals a + distinct_naturals b) n = 1) ∧
    (∀ a b : G, connected a b → gcd (distinct_naturals a + distinct_naturals b) n > 1) →
    15 ≤ n) :=
sorry

end smallest_n_exists_l2303_230347


namespace translation_coordinates_l2303_230335

theorem translation_coordinates (A : ℝ × ℝ) (T : ℝ × ℝ) (A' : ℝ × ℝ) 
  (hA : A = (-4, 3)) (hT : T = (2, 0)) (hA' : A' = (A.1 + T.1, A.2 + T.2)) : 
  A' = (-2, 3) := sorry

end translation_coordinates_l2303_230335


namespace vanessa_points_l2303_230338

theorem vanessa_points (total_points : ℕ) (num_other_players : ℕ) (avg_points_other : ℕ) 
  (h1 : total_points = 65) (h2 : num_other_players = 7) (h3 : avg_points_other = 5) :
  ∃ vp : ℕ, vp = 30 :=
by
  sorry

end vanessa_points_l2303_230338


namespace route_y_saves_time_l2303_230343

theorem route_y_saves_time (distance_X speed_X : ℕ)
                           (distance_Y_WOCZ distance_Y_CZ speed_Y speed_Y_CZ : ℕ)
                           (time_saved_in_minutes : ℚ) :
  distance_X = 8 → 
  speed_X = 40 → 
  distance_Y_WOCZ = 6 → 
  distance_Y_CZ = 1 → 
  speed_Y = 50 → 
  speed_Y_CZ = 25 → 
  time_saved_in_minutes = 2.4 →
  (distance_X / speed_X : ℚ) * 60 - 
  ((distance_Y_WOCZ / speed_Y + distance_Y_CZ / speed_Y_CZ) * 60) = time_saved_in_minutes :=
by
  intros
  sorry

end route_y_saves_time_l2303_230343


namespace reciprocals_expression_value_l2303_230301

theorem reciprocals_expression_value (a b : ℝ) (h : a * b = 1) : a^2 * b - (a - 2023) = 2023 := 
by 
  sorry

end reciprocals_expression_value_l2303_230301


namespace even_function_zeros_l2303_230315

noncomputable def f (x m : ℝ) : ℝ := (x - 1) * (x + m)

theorem even_function_zeros (m : ℝ) (h : ∀ x : ℝ, f x m = f (-x) m ) : 
  m = 1 ∧ (∀ x : ℝ, f x m = 0 → (x = 1 ∨ x = -1)) := by
  sorry

end even_function_zeros_l2303_230315


namespace case_one_case_two_l2303_230380

theorem case_one (n : ℝ) (h : n > -1) : n^3 + 1 > n^2 + n :=
sorry

theorem case_two (n : ℝ) (h : n < -1) : n^3 + 1 < n^2 + n :=
sorry

end case_one_case_two_l2303_230380


namespace gnollish_valid_sentence_count_is_48_l2303_230321

-- Define the problem parameters
def gnollish_words : List String := ["word1", "word2", "splargh", "glumph", "kreeg"]

def valid_sentence_count : Nat :=
  let total_sentences := 4 * 4 * 4
  let invalid_sentences :=
    4 +         -- (word) splargh glumph
    4 +         -- splargh glumph (word)
    4 +         -- (word) splargh kreeg
    4           -- splargh kreeg (word)
  total_sentences - invalid_sentences

-- Prove that the number of valid 3-word sentences is 48
theorem gnollish_valid_sentence_count_is_48 : valid_sentence_count = 48 := by
  sorry

end gnollish_valid_sentence_count_is_48_l2303_230321


namespace problem_statement_l2303_230346

theorem problem_statement :
  (-2010)^2011 = - (2010 ^ 2011) :=
by
  -- proof to be filled in
  sorry

end problem_statement_l2303_230346


namespace complement_A_correct_l2303_230396

def A : Set ℝ := {x | 1 - (8 / (x - 2)) < 0}

def complement_A : Set ℝ := {x | x ≤ 2 ∨ x ≥ 10}

theorem complement_A_correct : (Aᶜ = complement_A) :=
by {
  -- Placeholder for the necessary proof
  sorry
}

end complement_A_correct_l2303_230396


namespace total_eggs_needed_l2303_230328

-- Define the conditions
def eggsFromAndrew : ℕ := 155
def eggsToBuy : ℕ := 67

-- Define the total number of eggs
def totalEggs : ℕ := eggsFromAndrew + eggsToBuy

-- The theorem to be proven
theorem total_eggs_needed : totalEggs = 222 := by
  sorry

end total_eggs_needed_l2303_230328


namespace solution_set_l2303_230348

noncomputable def f (x : ℝ) : ℝ :=
  x * Real.sin x + Real.cos x + x^2

theorem solution_set (x : ℝ) :
  f (Real.log x) + f (Real.log (1 / x)) < 2 * f 1 ↔ (1 / Real.exp 1 < x ∧ x < Real.exp 1) :=
by {
  sorry
}

end solution_set_l2303_230348
