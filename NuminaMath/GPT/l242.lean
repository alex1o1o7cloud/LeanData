import Mathlib

namespace NUMINAMATH_GPT_probability_is_seven_fifteenths_l242_24220

-- Define the problem conditions
def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def choose_3_from_10 : ℕ := Nat.choose 10 3
def choose_3_red : ℕ := Nat.choose 5 3
def choose_3_green : ℕ := Nat.choose 3 3
def choose_2_red_1_green : ℕ := Nat.choose 5 2 * Nat.choose 3 1
def choose_2_green_1_red : ℕ := Nat.choose 3 2 * Nat.choose 5 1

-- Calculate favorable outcomes
def favorable_outcomes : ℕ :=
  choose_3_red + choose_3_green + choose_2_red_1_green + choose_2_green_1_red

-- Calculate the required probability
def probability_all_red_or_green : ℚ := favorable_outcomes / choose_3_from_10

-- Prove that probability_all_red_or_green is 7/15
theorem probability_is_seven_fifteenths :
  probability_all_red_or_green = 7 / 15 :=
by 
  -- Leaving the proof as a sorry for now
  sorry

end NUMINAMATH_GPT_probability_is_seven_fifteenths_l242_24220


namespace NUMINAMATH_GPT_small_slices_sold_l242_24218

theorem small_slices_sold (S L : ℕ) 
  (h1 : S + L = 5000) 
  (h2 : 150 * S + 250 * L = 1050000) : 
  S = 2000 :=
by
  sorry

end NUMINAMATH_GPT_small_slices_sold_l242_24218


namespace NUMINAMATH_GPT_max_red_socks_l242_24244

theorem max_red_socks (r b g t : ℕ) (h1 : t ≤ 2500) (h2 : r + b + g = t) 
  (h3 : (r * (r - 1) + b * (b - 1) + g * (g - 1)) = (2 / 3) * t * (t - 1)) : 
  r ≤ 1625 :=
by 
  sorry

end NUMINAMATH_GPT_max_red_socks_l242_24244


namespace NUMINAMATH_GPT_range_of_k_l242_24213

theorem range_of_k (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l242_24213


namespace NUMINAMATH_GPT_probability_route_X_is_8_over_11_l242_24282

-- Definitions for the graph paths and probabilities
def routes_from_A_to_B (X Y : Nat) : Nat := 2 + 6 + 3

def routes_passing_through_X (X Y : Nat) : Nat := 2 + 6

def probability_passing_through_X (total_routes passing_routes : Nat) : Rat :=
  (passing_routes : Rat) / (total_routes : Rat)

theorem probability_route_X_is_8_over_11 :
  let total_routes := routes_from_A_to_B 2 3
  let passing_routes := routes_passing_through_X 2 3
  probability_passing_through_X total_routes passing_routes = 8 / 11 :=
by
  -- Assumes correct route calculations from the conditions and aims to prove the probability value
  sorry

end NUMINAMATH_GPT_probability_route_X_is_8_over_11_l242_24282


namespace NUMINAMATH_GPT_find_value_of_a_l242_24236

def pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem find_value_of_a (a : ℝ) :
  pure_imaginary ((a^3 - a) + (a / (1 - a)) * Complex.I) ↔ a = -1 := 
sorry

end NUMINAMATH_GPT_find_value_of_a_l242_24236


namespace NUMINAMATH_GPT_sum_of_squares_l242_24228

theorem sum_of_squares (x : ℕ) (h : 2 * x = 14) : (3 * x)^2 + (2 * x)^2 + (5 * x)^2 = 1862 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_l242_24228


namespace NUMINAMATH_GPT_range_of_a_l242_24259

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (2 * a - 3) * x - 1 else x ^ 2 + 1

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ (3 / 2 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l242_24259


namespace NUMINAMATH_GPT_pyramid_volume_is_232_l242_24291

noncomputable def pyramid_volume (length : ℝ) (width : ℝ) (slant_height : ℝ) : ℝ :=
  (1 / 3) * (length * width) * (Real.sqrt ((slant_height)^2 - ((length / 2)^2 + (width / 2)^2)))

theorem pyramid_volume_is_232 :
  pyramid_volume 5 10 15 = 232 := 
by
  sorry

end NUMINAMATH_GPT_pyramid_volume_is_232_l242_24291


namespace NUMINAMATH_GPT_solution_set_inequality_l242_24234

theorem solution_set_inequality {a b : ℝ} 
  (h₁ : {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | ax^2 - bx + 2 < 0}) : a + b = -2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l242_24234


namespace NUMINAMATH_GPT_anne_cleaning_time_l242_24265

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end NUMINAMATH_GPT_anne_cleaning_time_l242_24265


namespace NUMINAMATH_GPT_cheese_fries_cost_l242_24298

def jim_money : ℝ := 20
def cousin_money : ℝ := 10
def combined_money : ℝ := jim_money + cousin_money
def expenditure : ℝ := 0.80 * combined_money
def cheeseburger_cost : ℝ := 3
def milkshake_cost : ℝ := 5
def cheeseburgers_cost : ℝ := 2 * cheeseburger_cost
def milkshakes_cost : ℝ := 2 * milkshake_cost
def meal_cost : ℝ := cheeseburgers_cost + milkshakes_cost

theorem cheese_fries_cost :
  let cheese_fries_cost := expenditure - meal_cost 
  cheese_fries_cost = 8 := 
by
  sorry

end NUMINAMATH_GPT_cheese_fries_cost_l242_24298


namespace NUMINAMATH_GPT_sean_total_cost_l242_24214

noncomputable def total_cost (soda_cost soup_cost sandwich_cost : ℕ) (num_soda num_soup num_sandwich : ℕ) : ℕ :=
  num_soda * soda_cost + num_soup * soup_cost + num_sandwich * sandwich_cost

theorem sean_total_cost :
  let soda_cost := 1
  let soup_cost := 3 * soda_cost
  let sandwich_cost := 3 * soup_cost
  let num_soda := 3
  let num_soup := 2
  let num_sandwich := 1
  total_cost soda_cost soup_cost sandwich_cost num_soda num_soup num_sandwich = 18 :=
by
  sorry

end NUMINAMATH_GPT_sean_total_cost_l242_24214


namespace NUMINAMATH_GPT_sqrt_of_sum_eq_l242_24271

noncomputable def cube_term : ℝ := 2 ^ 3
noncomputable def sum_cubes : ℝ := cube_term + cube_term + cube_term + cube_term
noncomputable def sqrt_sum : ℝ := Real.sqrt sum_cubes

theorem sqrt_of_sum_eq :
  sqrt_sum = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_sum_eq_l242_24271


namespace NUMINAMATH_GPT_part1_part2_l242_24292

variables (a x : ℝ)

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (|x - 1| ≤ 2) ∧ ((x + 3) / (x - 2) ≥ 0)

-- Part 1
theorem part1 (h_a : a = 1) (h_p : p a x) (h_q : q x) : 2 < x ∧ x < 3 := sorry

-- Part 2
theorem part2 (h_suff : ∀ x, q x → p a x) : 1 < a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_part1_part2_l242_24292


namespace NUMINAMATH_GPT_translate_parabola_l242_24247

theorem translate_parabola :
  (∀ x : ℝ, (y : ℝ) = 6 * x^2 -> y = 6 * (x + 2)^2 + 3) :=
by
  sorry

end NUMINAMATH_GPT_translate_parabola_l242_24247


namespace NUMINAMATH_GPT_average_weight_calculation_l242_24225

noncomputable def new_average_weight (initial_people : ℕ) (initial_avg_weight : ℝ) 
                                     (new_person_weight : ℝ) (total_people : ℕ) : ℝ :=
  (initial_people * initial_avg_weight + new_person_weight) / total_people

theorem average_weight_calculation :
  new_average_weight 6 160 97 7 = 151 := by
  sorry

end NUMINAMATH_GPT_average_weight_calculation_l242_24225


namespace NUMINAMATH_GPT_slowest_time_l242_24219

open Real

def time_lola (stories : ℕ) (run_time : ℝ) : ℝ := stories * run_time

def time_sam (stories_run stories_elevator : ℕ) (run_time elevate_time stop_time : ℝ) (wait_time : ℝ) : ℝ :=
  let run_part  := stories_run * run_time
  let wait_part := wait_time
  let elevator_part := stories_elevator * elevate_time + (stories_elevator - 1) * stop_time
  run_part + wait_part + elevator_part

def time_tara (stories : ℕ) (elevate_time stop_time : ℝ) : ℝ :=
  stories * elevate_time + (stories - 1) * stop_time

theorem slowest_time 
  (build_stories : ℕ) (lola_run_time sam_run_time elevate_time stop_time wait_time : ℝ)
  (h_build : build_stories = 50)
  (h_lola_run : lola_run_time = 12) (h_sam_run : sam_run_time = 15)
  (h_elevate : elevate_time = 10) (h_stop : stop_time = 4) (h_wait : wait_time = 20) :
  max (time_lola build_stories lola_run_time) 
    (max (time_sam 25 25 sam_run_time elevate_time stop_time wait_time) 
         (time_tara build_stories elevate_time stop_time)) = 741 := by
  sorry

end NUMINAMATH_GPT_slowest_time_l242_24219


namespace NUMINAMATH_GPT_smallest_number_of_packs_l242_24211

theorem smallest_number_of_packs (n b w : ℕ) (Hn : n = 13) (Hb : b = 8) (Hw : w = 17) :
  Nat.lcm (Nat.lcm n b) w = 1768 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_packs_l242_24211


namespace NUMINAMATH_GPT_three_lines_intersect_single_point_l242_24200

theorem three_lines_intersect_single_point (a : ℝ) :
  (∀ x y : ℝ, (x + 2*y + a) * (x^2 - y^2) = 0) ↔ a = 0 := by
  sorry

end NUMINAMATH_GPT_three_lines_intersect_single_point_l242_24200


namespace NUMINAMATH_GPT_find_f_neg1_l242_24276

-- Definitions based on conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable {b : ℝ} (f : ℝ → ℝ)

axiom odd_f : odd_function f
axiom f_form : ∀ x, 0 ≤ x → f x = 2^(x + 1) + 2 * x + b
axiom b_value : b = -2

theorem find_f_neg1 : f (-1) = -4 :=
sorry

end NUMINAMATH_GPT_find_f_neg1_l242_24276


namespace NUMINAMATH_GPT_number_of_pupils_l242_24226

theorem number_of_pupils (n : ℕ) (M : ℕ)
  (avg_all : 39 * n = M)
  (pupil_marks : 25 + 12 + 15 + 19 = 71)
  (new_avg : (M - 71) / (n - 4) = 44) :
  n = 21 := sorry

end NUMINAMATH_GPT_number_of_pupils_l242_24226


namespace NUMINAMATH_GPT_a_cubed_value_l242_24254

theorem a_cubed_value (a b : ℝ) (k : ℝ) (h1 : a^3 * b^2 = k) (h2 : a = 5) (h3 : b = 2) : 
  ∃ (a : ℝ), (64 * a^3 = 500) → (a^3 = 125 / 16) :=
by
  sorry

end NUMINAMATH_GPT_a_cubed_value_l242_24254


namespace NUMINAMATH_GPT_smallest_number_is_20_l242_24293

theorem smallest_number_is_20 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ≤ b) (h5 : b ≤ c)
  (mean_condition : (a + b + c) / 3 = 30)
  (median_condition : b = 31)
  (largest_condition : b = c - 8) :
  a = 20 :=
sorry

end NUMINAMATH_GPT_smallest_number_is_20_l242_24293


namespace NUMINAMATH_GPT_house_orderings_l242_24231

/-- Ralph walks past five houses each painted in a different color: 
orange, red, blue, yellow, and green.
Conditions:
1. Ralph passed the orange house before the red house.
2. Ralph passed the blue house before the yellow house.
3. The blue house was not next to the yellow house.
4. Ralph passed the green house before the red house and after the blue house.
Given these conditions, prove that there are exactly 3 valid orderings of the houses.
-/
theorem house_orderings : 
  ∃ (orderings : Finset (List String)), 
  orderings.card = 3 ∧
  (∀ (o : List String), 
   o ∈ orderings ↔ 
    ∃ (idx_o idx_r idx_b idx_y idx_g : ℕ), 
    o = ["orange", "red", "blue", "yellow", "green"] ∧
    idx_o < idx_r ∧ 
    idx_b < idx_y ∧ 
    (idx_b + 1 < idx_y ∨ idx_y + 1 < idx_b) ∧ 
    idx_b < idx_g ∧ idx_g < idx_r) := sorry

end NUMINAMATH_GPT_house_orderings_l242_24231


namespace NUMINAMATH_GPT_intersection_M_N_l242_24264

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 10 }
def N : Set ℝ := { x | x > 7 ∨ x < 1 }
def MN_intersection : Set ℝ := { x | (-1 ≤ x ∧ x < 1) ∨ (7 < x ∧ x ≤ 10) }

theorem intersection_M_N :
  M ∩ N = MN_intersection :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l242_24264


namespace NUMINAMATH_GPT_solve_for_vee_l242_24285

theorem solve_for_vee (vee : ℝ) (h : 4 * vee ^ 2 = 144) : vee = 6 ∨ vee = -6 :=
by
  -- We state that this theorem should be true for all vee and given the condition h
  sorry

end NUMINAMATH_GPT_solve_for_vee_l242_24285


namespace NUMINAMATH_GPT_total_distance_covered_l242_24263

noncomputable def speed_train_a : ℚ := 80          -- Speed of Train A in kmph
noncomputable def speed_train_b : ℚ := 110         -- Speed of Train B in kmph
noncomputable def duration : ℚ := 15               -- Duration in minutes
noncomputable def conversion_factor : ℚ := 60      -- Conversion factor from hours to minutes

theorem total_distance_covered : 
    (speed_train_a / conversion_factor) * duration + 
    (speed_train_b / conversion_factor) * duration = 47.5 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_covered_l242_24263


namespace NUMINAMATH_GPT_jacob_total_bill_l242_24215

def base_cost : ℝ := 25
def included_hours : ℕ := 25
def cost_per_text : ℝ := 0.08
def cost_per_extra_minute : ℝ := 0.13
def jacob_texts : ℕ := 150
def jacob_hours : ℕ := 31

theorem jacob_total_bill : 
  let extra_minutes := (jacob_hours - included_hours) * 60
  let total_cost := base_cost + jacob_texts * cost_per_text + extra_minutes * cost_per_extra_minute
  total_cost = 83.80 := 
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_jacob_total_bill_l242_24215


namespace NUMINAMATH_GPT_exponent_property_l242_24249

theorem exponent_property (a x y : ℝ) (hx : a ^ x = 2) (hy : a ^ y = 3) : a ^ (x + y) = 6 := by
  sorry

end NUMINAMATH_GPT_exponent_property_l242_24249


namespace NUMINAMATH_GPT_probability_sum_of_three_dice_is_9_l242_24204

def sum_of_three_dice_is_9 : Prop :=
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 9)

theorem probability_sum_of_three_dice_is_9 : 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 → a + b + c = 9 → sum_of_three_dice_is_9) ∧ 
  (1 / 216 = 25 / 216) := 
by
  sorry

end NUMINAMATH_GPT_probability_sum_of_three_dice_is_9_l242_24204


namespace NUMINAMATH_GPT_trajectory_parabola_l242_24246

noncomputable def otimes (x1 x2 : ℝ) : ℝ := (x1 + x2)^2 - (x1 - x2)^2

theorem trajectory_parabola (x : ℝ) (h : 0 ≤ x) : 
  ∃ (y : ℝ), y^2 = 8 * x ∧ (∀ P : ℝ × ℝ, P = (x, y) → (P.snd^2 = 8 * P.fst)) :=
by
  sorry

end NUMINAMATH_GPT_trajectory_parabola_l242_24246


namespace NUMINAMATH_GPT_max_value_8a_3b_5c_l242_24241

theorem max_value_8a_3b_5c (a b c : ℝ) (h_condition : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ (Real.sqrt 373) / 6 :=
by
  sorry

end NUMINAMATH_GPT_max_value_8a_3b_5c_l242_24241


namespace NUMINAMATH_GPT_set_listing_l242_24267

open Set

def A : Set (ℤ × ℤ) := {p | ∃ x y : ℤ, p = (x, y) ∧ x^2 = y + 1 ∧ |x| < 2}

theorem set_listing :
  A = {(-1, 0), (0, -1), (1, 0)} :=
by {
  sorry
}

end NUMINAMATH_GPT_set_listing_l242_24267


namespace NUMINAMATH_GPT_Chloe_final_points_l242_24261

-- Define the points scored (or lost) in each round
def round1_points : ℤ := 40
def round2_points : ℤ := 50
def round3_points : ℤ := 60
def round4_points : ℤ := 70
def round5_points : ℤ := -4
def round6_points : ℤ := 80
def round7_points : ℤ := -6

-- Statement to prove: Chloe's total points at the end of the game
theorem Chloe_final_points : 
  round1_points + round2_points + round3_points + round4_points + round5_points + round6_points + round7_points = 290 :=
by
  sorry

end NUMINAMATH_GPT_Chloe_final_points_l242_24261


namespace NUMINAMATH_GPT_system_of_equations_solution_l242_24251

theorem system_of_equations_solution :
  ∃ (X Y: ℝ), 
    (X^2 * Y^2 + X * Y^2 + X^2 * Y + X * Y + X + Y + 3 = 0) ∧ 
    (X^2 * Y + X * Y + 1 = 0) ∧ 
    (X = -2) ∧ (Y = -1/2) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l242_24251


namespace NUMINAMATH_GPT_intersection_empty_l242_24274

noncomputable def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
noncomputable def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem intersection_empty : A ∩ B = ∅ :=
by
  sorry

end NUMINAMATH_GPT_intersection_empty_l242_24274


namespace NUMINAMATH_GPT_sin_cos_value_l242_24297

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : (Real.sin x) * (Real.cos x) = 4 / 17 := by
  sorry

end NUMINAMATH_GPT_sin_cos_value_l242_24297


namespace NUMINAMATH_GPT_trig_expression_l242_24255

theorem trig_expression (θ : ℝ) (h : Real.tan (Real.pi / 4 + θ) = 3) : 
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
by sorry

end NUMINAMATH_GPT_trig_expression_l242_24255


namespace NUMINAMATH_GPT_kevin_expected_away_time_l242_24238

theorem kevin_expected_away_time
  (leak_rate : ℝ)
  (bucket_capacity : ℝ)
  (bucket_factor : ℝ)
  (leak_rate_eq : leak_rate = 1.5)
  (bucket_capacity_eq : bucket_capacity = 36)
  (bucket_factor_eq : bucket_factor = 2)
  : ((bucket_capacity / bucket_factor) / leak_rate) = 12 :=
by
  rw [bucket_capacity_eq, leak_rate_eq, bucket_factor_eq]
  sorry

end NUMINAMATH_GPT_kevin_expected_away_time_l242_24238


namespace NUMINAMATH_GPT_total_crayons_lost_or_given_away_l242_24203

def crayons_given_away : ℕ := 52
def crayons_lost : ℕ := 535

theorem total_crayons_lost_or_given_away :
  crayons_given_away + crayons_lost = 587 :=
by
  sorry

end NUMINAMATH_GPT_total_crayons_lost_or_given_away_l242_24203


namespace NUMINAMATH_GPT_difference_of_bases_l242_24212

def base8_to_base10 (n : ℕ) : ℕ :=
  5 * (8^5) + 4 * (8^4) + 3 * (8^3) + 2 * (8^2) + 1 * (8^1) + 0 * (8^0)

def base5_to_base10 (n : ℕ) : ℕ :=
  4 * (5^4) + 3 * (5^3) + 2 * (5^2) + 1 * (5^1) + 0 * (5^0)

theorem difference_of_bases : 
  base8_to_base10 543210 - base5_to_base10 43210 = 177966 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_bases_l242_24212


namespace NUMINAMATH_GPT_number_of_tangent_and_parallel_lines_l242_24283

theorem number_of_tangent_and_parallel_lines (p : ℝ × ℝ) (a : ℝ) (h : p = (2, 4)) (hp_on_parabola : (p.1)^2 = 8 * p.2) :
  ∃ l1 l2 : (ℝ × ℝ) → Prop, 
    (l1 (2, 4) ∧ l2 (2, 4)) ∧ 
    (∀ l, (l = l1 ∨ l = l2) ↔ (∃ q, q ≠ p ∧ q ∈ {p' | (p'.1)^2 = 8 * p'.2})) ∧ 
    (∀ p' ∈ {p' | (p'.1)^2 = 8 * p'.2}, (l1 p' ∨ l2 p') → False) :=
sorry

end NUMINAMATH_GPT_number_of_tangent_and_parallel_lines_l242_24283


namespace NUMINAMATH_GPT_vector_parallel_solution_l242_24210

theorem vector_parallel_solution (x : ℝ) :
  let a := (1, x)
  let b := (x - 1, 2)
  (a.1 * b.2 = a.2 * b.1) → (x = 2 ∨ x = -1) :=
by
  intros a b h
  let a := (1, x)
  let b := (x - 1, 2)
  sorry

end NUMINAMATH_GPT_vector_parallel_solution_l242_24210


namespace NUMINAMATH_GPT_find_x_l242_24248

-- Definitions of the conditions
def eq1 (x y z : ℕ) : Prop := x + y + z = 25
def eq2 (y z : ℕ) : Prop := y + z = 14

-- Statement of the mathematically equivalent proof problem
theorem find_x (x y z : ℕ) (h1 : eq1 x y z) (h2 : eq2 y z) : x = 11 :=
by {
  -- This is where the proof would go, but we can omit it for now:
  sorry
}

end NUMINAMATH_GPT_find_x_l242_24248


namespace NUMINAMATH_GPT_bike_distance_difference_l242_24262

-- Defining constants for Alex's and Bella's rates and the time duration
def Alex_rate : ℕ := 12
def Bella_rate : ℕ := 10
def time : ℕ := 6

-- The goal is to prove the difference in distance is 12 miles
theorem bike_distance_difference : (Alex_rate * time) - (Bella_rate * time) = 12 := by
  sorry

end NUMINAMATH_GPT_bike_distance_difference_l242_24262


namespace NUMINAMATH_GPT_ring_display_capacity_l242_24273

def necklace_capacity : ℕ := 12
def current_necklaces : ℕ := 5
def ring_capacity : ℕ := 18
def bracelet_capacity : ℕ := 15
def current_bracelets : ℕ := 8
def necklace_cost : ℕ := 4
def ring_cost : ℕ := 10
def bracelet_cost : ℕ := 5
def total_cost : ℕ := 183

theorem ring_display_capacity : ring_capacity + (total_cost - ((necklace_capacity - current_necklaces) * necklace_cost + (bracelet_capacity - current_bracelets) * bracelet_cost)) / ring_cost = 30 := by
  sorry

end NUMINAMATH_GPT_ring_display_capacity_l242_24273


namespace NUMINAMATH_GPT_cos_beta_l242_24287

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.cos α = 3/5)
  (h2 : Real.cos (α + β) = -5/13) : Real.cos β = 33/65 := 
sorry

end NUMINAMATH_GPT_cos_beta_l242_24287


namespace NUMINAMATH_GPT_simplify_fraction_product_l242_24269

theorem simplify_fraction_product : 
  (21 / 28) * (14 / 33) * (99 / 42) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_product_l242_24269


namespace NUMINAMATH_GPT_sum_due_is_42_l242_24253

-- Define the conditions
def BD : ℝ := 42
def TD : ℝ := 36

-- Statement to prove
theorem sum_due_is_42 (H1 : BD = 42) (H2 : TD = 36) : ∃ (FV : ℝ), FV = 42 := by
  -- Proof Placeholder
  sorry

end NUMINAMATH_GPT_sum_due_is_42_l242_24253


namespace NUMINAMATH_GPT_min_sum_of_positive_real_solution_l242_24252

theorem min_sum_of_positive_real_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_min_sum_of_positive_real_solution_l242_24252


namespace NUMINAMATH_GPT_gcd_36_48_72_l242_24295

theorem gcd_36_48_72 : Int.gcd (Int.gcd 36 48) 72 = 12 := by
  have h1 : 36 = 2^2 * 3^2 := by norm_num
  have h2 : 48 = 2^4 * 3 := by norm_num
  have h3 : 72 = 2^3 * 3^2 := by norm_num
  sorry

end NUMINAMATH_GPT_gcd_36_48_72_l242_24295


namespace NUMINAMATH_GPT_local_maximum_no_global_maximum_equation_root_condition_l242_24286

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x^2 + 2*x - 3) * Real.exp x

theorem local_maximum_no_global_maximum : (∃ x0 : ℝ, f' x0 = 0 ∧ (∀ x < x0, f x < f x0) ∧ (∀ x > x0, f x < f x0))
∧ (f 1 = -2 * Real.exp 1) 
∧ (∀ x : ℝ, ∃ b : ℝ, f x = b ∧ b > 6 * Real.exp (-3) → ¬(f x = f 1))
:= sorry

theorem equation_root_condition (b : ℝ) : (∃ x1 x2 x3 : ℝ, f x1 = b ∧ f x2 = b ∧ f x3 = b ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) 
→ (0 < b ∧ b < 6 * Real.exp (-3))
:= sorry

end NUMINAMATH_GPT_local_maximum_no_global_maximum_equation_root_condition_l242_24286


namespace NUMINAMATH_GPT_number_of_positive_integers_l242_24224

theorem number_of_positive_integers (n : ℕ) : 
  (0 < n ∧ n < 36 ∧ (∃ k : ℕ, n = k * (36 - k))) → 
  n = 18 ∨ n = 24 ∨ n = 30 ∨ n = 32 ∨ n = 34 ∨ n = 35 :=
sorry

end NUMINAMATH_GPT_number_of_positive_integers_l242_24224


namespace NUMINAMATH_GPT_sequence_sum_problem_l242_24202

theorem sequence_sum_problem (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * a n - n) :
  (2 / (a 1 * a 2) + 4 / (a 2 * a 3) + 8 / (a 3 * a 4) + 16 / (a 4 * a 5) : ℚ) = 30 / 31 := 
sorry

end NUMINAMATH_GPT_sequence_sum_problem_l242_24202


namespace NUMINAMATH_GPT_find_y_l242_24289

theorem find_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3 * y) = 8) : y = 1 := by
  sorry

end NUMINAMATH_GPT_find_y_l242_24289


namespace NUMINAMATH_GPT_number_of_distinct_d_l242_24275

noncomputable def calculateDistinctValuesOfD (u v w x : ℂ) (h_distinct : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x) : ℕ := 
by
  sorry

theorem number_of_distinct_d (u v w x : ℂ) (h : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x)
    (h_eqs : ∀ z : ℂ, (z - u) * (z - v) * (z - w) * (z - x) = 
             (z - (d * u)) * (z - (d * v)) * (z - (d * w)) * (z - (d * x))) : 
    calculateDistinctValuesOfD u v w x h = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_distinct_d_l242_24275


namespace NUMINAMATH_GPT_find_positive_integers_l242_24294

theorem find_positive_integers (a b : ℕ) (h1 : a > 1) (h2 : b ∣ (a - 1)) (h3 : (2 * a + 1) ∣ (5 * b - 3)) : a = 10 ∧ b = 9 :=
sorry

end NUMINAMATH_GPT_find_positive_integers_l242_24294


namespace NUMINAMATH_GPT_cost_of_50_roses_l242_24230

def cost_of_dozen_roses : ℝ := 24

def is_proportional (n : ℕ) (cost : ℝ) : Prop :=
  cost = (cost_of_dozen_roses / 12) * n

def has_discount (n : ℕ) : Prop :=
  n ≥ 45

theorem cost_of_50_roses :
  ∃ (cost : ℝ), is_proportional 50 cost ∧ has_discount 50 ∧ cost * 0.9 = 90 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_50_roses_l242_24230


namespace NUMINAMATH_GPT_find_first_number_l242_24256

theorem find_first_number (x y : ℝ) (h1 : x + y = 50) (h2 : 2 * (x - y) = 20) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l242_24256


namespace NUMINAMATH_GPT_find_m_given_a3_eq_40_l242_24216

theorem find_m_given_a3_eq_40 (m : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 - m * x) ^ 5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_3 = 40 →
  m = -1 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_given_a3_eq_40_l242_24216


namespace NUMINAMATH_GPT_usual_time_proof_l242_24279

noncomputable 
def usual_time (P T : ℝ) := (P * T) / (100 - P)

theorem usual_time_proof (P T U : ℝ) (h1 : P > 0) (h2 : P < 100) (h3 : T > 0) (h4 : U = usual_time P T) : U = (P * T) / (100 - P) :=
by
    sorry

end NUMINAMATH_GPT_usual_time_proof_l242_24279


namespace NUMINAMATH_GPT_fill_pool_time_l242_24272

theorem fill_pool_time (pool_volume : ℕ := 32000) 
                       (num_hoses : ℕ := 5) 
                       (flow_rate_per_hose : ℕ := 4) 
                       (operation_minutes : ℕ := 45) 
                       (maintenance_minutes : ℕ := 15) 
                       : ℕ :=
by
  -- Calculation steps will go here in the actual proof
  sorry

example : fill_pool_time = 47 := by
  -- Proof of the theorem fill_pool_time here
  sorry

end NUMINAMATH_GPT_fill_pool_time_l242_24272


namespace NUMINAMATH_GPT_total_ways_is_13_l242_24227

-- Define the problem conditions
def num_bus_services : ℕ := 8
def num_train_services : ℕ := 3
def num_ferry_services : ℕ := 2

-- Define the total number of ways a person can travel from A to B
def total_ways : ℕ := num_bus_services + num_train_services + num_ferry_services

-- State the theorem that the total number of ways is 13
theorem total_ways_is_13 : total_ways = 13 :=
by
  -- Add a sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_ways_is_13_l242_24227


namespace NUMINAMATH_GPT_proof_of_p_and_not_q_l242_24288

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x
def q : Prop := ∀ x : ℝ, exp x > 1

theorem proof_of_p_and_not_q : p ∧ ¬q :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_of_p_and_not_q_l242_24288


namespace NUMINAMATH_GPT_jill_runs_more_than_jack_l242_24260

noncomputable def streetWidth : ℝ := 15 -- Street width in feet
noncomputable def blockSide : ℝ := 300 -- Side length of the block in feet

noncomputable def jacksPerimeter : ℝ := 4 * blockSide -- Perimeter of Jack's running path
noncomputable def jillsPerimeter : ℝ := 4 * (blockSide + 2 * streetWidth) -- Perimeter of Jill's running path on the opposite side of the street

theorem jill_runs_more_than_jack :
  jillsPerimeter - jacksPerimeter = 120 :=
by
  sorry

end NUMINAMATH_GPT_jill_runs_more_than_jack_l242_24260


namespace NUMINAMATH_GPT_total_students_l242_24201

-- Definitions based on problem conditions
def H := 36
def S := 32
def union_H_S := 59
def history_not_statistics := 27

-- The proof statement
theorem total_students : H + S - (H - history_not_statistics) = union_H_S :=
by sorry

end NUMINAMATH_GPT_total_students_l242_24201


namespace NUMINAMATH_GPT_inequality_one_solution_inequality_two_solution_l242_24209

-- The statement for the first inequality
theorem inequality_one_solution (x : ℝ) :
  |1 - ((2 * x - 1) / 3)| ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5 := sorry

-- The statement for the second inequality
theorem inequality_two_solution (x : ℝ) :
  (2 - x) * (x + 3) < 2 - x ↔ x < -2 ∨ x > 2 := sorry

end NUMINAMATH_GPT_inequality_one_solution_inequality_two_solution_l242_24209


namespace NUMINAMATH_GPT_rate_of_drawing_barbed_wire_is_correct_l242_24268

noncomputable def rate_of_drawing_barbed_wire (area cost: ℝ) (gate_width barbed_wire_extension: ℝ) : ℝ :=
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_barbed_wire := (perimeter - 2 * gate_width) + 4 * barbed_wire_extension
  cost / total_barbed_wire

theorem rate_of_drawing_barbed_wire_is_correct :
  rate_of_drawing_barbed_wire 3136 666 1 3 = 2.85 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_drawing_barbed_wire_is_correct_l242_24268


namespace NUMINAMATH_GPT_johns_chore_homework_time_l242_24208

-- Definitions based on problem conditions
def cartoons_time : ℕ := 150  -- John's cartoon watching time in minutes
def chores_homework_per_10 : ℕ := 13  -- 13 minutes combined chores and homework per 10 minutes of cartoons
def cartoon_period : ℕ := 10  -- Per 10 minutes period

-- Theorem statement
theorem johns_chore_homework_time :
  cartoons_time / cartoon_period * chores_homework_per_10 = 195 :=
by sorry

end NUMINAMATH_GPT_johns_chore_homework_time_l242_24208


namespace NUMINAMATH_GPT_length_DC_of_ABCD_l242_24235

open Real

structure Trapezoid (ABCD : Type) :=
  (AB DC : ℝ)
  (BC : ℝ := 0)
  (angleBCD angleCDA : ℝ)

noncomputable def given_trapezoid : Trapezoid ℝ :=
{ AB := 5,
  DC := 8 + sqrt 3, -- this is from the answer
  BC := 3 * sqrt 2,
  angleBCD := π / 4,   -- 45 degrees in radians
  angleCDA := π / 3 }  -- 60 degrees in radians

variable (ABCD : Trapezoid ℝ)

theorem length_DC_of_ABCD :
  ABCD.AB = 5 ∧
  ABCD.BC = 3 * sqrt 2 ∧
  ABCD.angleBCD = π / 4 ∧
  ABCD.angleCDA = π / 3 →
  ABCD.DC = 8 + sqrt 3 :=
sorry

end NUMINAMATH_GPT_length_DC_of_ABCD_l242_24235


namespace NUMINAMATH_GPT_necessarily_positive_l242_24278

theorem necessarily_positive (x y z : ℝ) (h1 : 0 < x ∧ x < 2) (h2 : -2 < y ∧ y < 0) (h3 : 0 < z ∧ z < 3) : 
  y + 2 * z > 0 := 
sorry

end NUMINAMATH_GPT_necessarily_positive_l242_24278


namespace NUMINAMATH_GPT_boat_problem_l242_24280

theorem boat_problem (x y : ℕ) (h : 12 * x + 5 * y = 99) :
  (x = 2 ∧ y = 15) ∨ (x = 7 ∧ y = 3) :=
sorry

end NUMINAMATH_GPT_boat_problem_l242_24280


namespace NUMINAMATH_GPT_range_of_a_l242_24217

theorem range_of_a (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, y = f x) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, y = f (x - a) + f (x + a)) ↔ -1/2 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l242_24217


namespace NUMINAMATH_GPT_det_transformed_matrix_l242_24239

variables {p q r s : ℝ} -- Defining the variables over the real numbers

-- Defining the first determinant condition as an axiom
axiom det_initial_matrix : (p * s - q * r) = 10

-- Stating the theorem to be proved
theorem det_transformed_matrix : 
  (p + 2 * r) * s - (q + 2 * s) * r = 10 :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_GPT_det_transformed_matrix_l242_24239


namespace NUMINAMATH_GPT_one_third_of_product_l242_24221

theorem one_third_of_product (a b c : ℕ) (h1 : a = 7) (h2 : b = 9) (h3 : c = 4) : (1 / 3 : ℚ) * (a * b * c : ℕ) = 84 := by
  sorry

end NUMINAMATH_GPT_one_third_of_product_l242_24221


namespace NUMINAMATH_GPT_sheep_to_cow_water_ratio_l242_24205

-- Set up the initial conditions
def number_of_cows := 40
def water_per_cow_per_day := 80
def number_of_sheep := 10 * number_of_cows
def total_water_per_week := 78400

-- Calculate total water consumption of cows per week
def water_cows_per_week := number_of_cows * water_per_cow_per_day * 7

-- Calculate total water consumption of sheep per week
def water_sheep_per_week := total_water_per_week - water_cows_per_week

-- Calculate daily water consumption per sheep
def water_sheep_per_day := water_sheep_per_week / 7
def daily_water_per_sheep := water_sheep_per_day / number_of_sheep

-- Define the target ratio
def target_ratio := 1 / 4

-- Statement to prove
theorem sheep_to_cow_water_ratio :
  (daily_water_per_sheep / water_per_cow_per_day) = target_ratio :=
sorry

end NUMINAMATH_GPT_sheep_to_cow_water_ratio_l242_24205


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_parallel_lines_l242_24222

theorem necessary_but_not_sufficient_for_parallel_lines (m : ℝ) : 
  (m = -1/2 ∨ m = 0) ↔ (∀ x y : ℝ, (x + 2*m*y - 1 = 0 ∧ (3*m + 1)*x - m*y - 1 = 0) → false) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_parallel_lines_l242_24222


namespace NUMINAMATH_GPT_Mitya_age_l242_24281

/--
Assume Mitya's current age is M and Shura's current age is S. If Mitya is 11 years older than Shura,
and when Mitya was as old as Shura is now, he was twice as old as Shura,
then prove that M = 27.5.
-/
theorem Mitya_age (S M : ℝ) (h1 : M = S + 11) (h2 : M - S = 2 * (S - (M - S))) : M = 27.5 :=
by
  sorry

end NUMINAMATH_GPT_Mitya_age_l242_24281


namespace NUMINAMATH_GPT_ticket_price_divisors_count_l242_24290

theorem ticket_price_divisors_count :
  ∃ (x : ℕ), (36 % x = 0) ∧ (60 % x = 0) ∧ (Nat.divisors (Nat.gcd 36 60)).card = 6 := 
by
  sorry

end NUMINAMATH_GPT_ticket_price_divisors_count_l242_24290


namespace NUMINAMATH_GPT_at_least_one_ge_one_l242_24284

theorem at_least_one_ge_one (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 2) : 
  max (|a - b|) (max (|b - c|) (|c - a|)) ≥ 1 :=
by 
  sorry

end NUMINAMATH_GPT_at_least_one_ge_one_l242_24284


namespace NUMINAMATH_GPT_jill_second_bus_time_l242_24277

-- Define constants representing the times
def wait_time_first_bus : ℕ := 12
def ride_time_first_bus : ℕ := 30

-- Define a function to calculate the total time for the first bus
def total_time_first_bus (wait : ℕ) (ride : ℕ) : ℕ :=
  wait + ride

-- Define a function to calculate the time for the second bus
def time_second_bus (total_first_bus_time : ℕ) : ℕ :=
  total_first_bus_time / 2

-- The theorem to prove
theorem jill_second_bus_time : 
  time_second_bus (total_time_first_bus wait_time_first_bus ride_time_first_bus) = 21 := by
  sorry

end NUMINAMATH_GPT_jill_second_bus_time_l242_24277


namespace NUMINAMATH_GPT_sum_of_roots_eq_l242_24250

theorem sum_of_roots_eq (k : ℝ) : ∃ x1 x2 : ℝ, (2 * x1 ^ 2 - 3 * x1 + k = 7) ∧ (2 * x2 ^ 2 - 3 * x2 + k = 7) ∧ (x1 + x2 = 3 / 2) :=
by sorry

end NUMINAMATH_GPT_sum_of_roots_eq_l242_24250


namespace NUMINAMATH_GPT_linear_function_quadrant_l242_24257

theorem linear_function_quadrant (x y : ℝ) : 
  y = 2 * x - 3 → ¬ ((x < 0 ∧ y > 0)) := 
sorry

end NUMINAMATH_GPT_linear_function_quadrant_l242_24257


namespace NUMINAMATH_GPT_product_of_two_numbers_l242_24232

theorem product_of_two_numbers (x y : ℕ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 221) : x * y = 60 := sorry

end NUMINAMATH_GPT_product_of_two_numbers_l242_24232


namespace NUMINAMATH_GPT_quadratic_completion_l242_24229

theorem quadratic_completion (x : ℝ) :
  (x^2 + 6 * x - 2) = ((x + 3)^2 - 11) := sorry

end NUMINAMATH_GPT_quadratic_completion_l242_24229


namespace NUMINAMATH_GPT_friend_spending_l242_24240

-- Definitions based on conditions
def total_spent (you friend : ℝ) : Prop := you + friend = 15
def friend_spent (you friend : ℝ) : Prop := friend = you + 1

-- Prove that the friend's spending equals $8 given the conditions
theorem friend_spending (you friend : ℝ) (htotal : total_spent you friend) (hfriend : friend_spent you friend) : friend = 8 :=
by
  sorry

end NUMINAMATH_GPT_friend_spending_l242_24240


namespace NUMINAMATH_GPT_winning_ticket_probability_l242_24223

theorem winning_ticket_probability (eligible_numbers : List ℕ) (length_eligible_numbers : eligible_numbers.length = 12)
(pick_6 : Π(t : List ℕ), List ℕ) (valid_ticket : List ℕ → Bool) (probability : ℚ) : 
(probability = (1 : ℚ) / (4 : ℚ)) :=
  sorry

end NUMINAMATH_GPT_winning_ticket_probability_l242_24223


namespace NUMINAMATH_GPT_sum_of_denominators_of_fractions_l242_24245

theorem sum_of_denominators_of_fractions {a b : ℕ} (ha : 3 * a / 5 * b + 2 * a / 9 * b + 4 * a / 15 * b = 28 / 45) (gcd_ab : Nat.gcd a b = 1) :
  5 * b + 9 * b + 15 * b = 203 := sorry

end NUMINAMATH_GPT_sum_of_denominators_of_fractions_l242_24245


namespace NUMINAMATH_GPT_surface_area_eighth_block_l242_24242

theorem surface_area_eighth_block {A B C D E F G H : ℕ} 
  (blockA : A = 148) 
  (blockB : B = 46) 
  (blockC : C = 72) 
  (blockD : D = 28) 
  (blockE : E = 88) 
  (blockF : F = 126) 
  (blockG : G = 58) 
  : H = 22 :=
by 
  sorry

end NUMINAMATH_GPT_surface_area_eighth_block_l242_24242


namespace NUMINAMATH_GPT_proof_problem_l242_24299

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, (y = 2^x - 1) ∧ (x ≤ 2)}

-- Define the complement of set A in U
def complement_A : Set ℝ := Set.compl A

-- Define the intersection of complement_A and B
def complement_A_inter_B : Set ℝ := complement_A ∩ B

-- State the theorem
theorem proof_problem : complement_A_inter_B = {x | (-1 < x) ∧ (x ≤ 2)} :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l242_24299


namespace NUMINAMATH_GPT_pentagon_angle_E_l242_24206

theorem pentagon_angle_E 
    (A B C D E : Type)
    [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
    (AB BC CD DE : ℝ)
    (angle_B angle_C angle_D : ℝ)
    (h1 : AB = BC)
    (h2 : BC = CD)
    (h3 : CD = DE)
    (h4 : angle_B = 96)
    (h5 : angle_C = 108)
    (h6 : angle_D = 108) :
    ∃ angle_E : ℝ, angle_E = 102 := 
by
  sorry

end NUMINAMATH_GPT_pentagon_angle_E_l242_24206


namespace NUMINAMATH_GPT_forgotten_angles_correct_l242_24266

theorem forgotten_angles_correct (n : ℕ) (h1 : (n - 2) * 180 = 2520) (h2 : 2345 + 175 = 2520) : 
  ∃ a b : ℕ, a + b = 175 :=
by
  sorry

end NUMINAMATH_GPT_forgotten_angles_correct_l242_24266


namespace NUMINAMATH_GPT_solve_for_x_l242_24243

-- Define the quadratic equation condition
def quadratic_eq (x : ℝ) : Prop := 3 * x^2 - 7 * x - 6 = 0

-- The main theorem to prove
theorem solve_for_x (x : ℝ) : x > 0 ∧ quadratic_eq x → x = 3 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l242_24243


namespace NUMINAMATH_GPT_girls_try_out_l242_24296

-- Given conditions
variables (boys callBacks didNotMakeCut : ℕ)
variable (G : ℕ)

-- Define the conditions
def conditions : Prop := 
  boys = 14 ∧ 
  callBacks = 2 ∧ 
  didNotMakeCut = 21 ∧ 
  G + boys = callBacks + didNotMakeCut

-- The statement of the proof
theorem girls_try_out (h : conditions boys callBacks didNotMakeCut G) : G = 9 :=
by
  sorry

end NUMINAMATH_GPT_girls_try_out_l242_24296


namespace NUMINAMATH_GPT_smallest_root_of_quadratic_l242_24237

theorem smallest_root_of_quadratic (y : ℝ) (h : 4 * y^2 - 7 * y + 3 = 0) : y = 3 / 4 :=
sorry

end NUMINAMATH_GPT_smallest_root_of_quadratic_l242_24237


namespace NUMINAMATH_GPT_vector_dot_product_l242_24270

-- Define the vectors
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)

-- Vector addition and scalar multiplication
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The mathematical statement to prove
theorem vector_dot_product : dot_product (vec_add (scalar_mul 2 vec_a) vec_b) vec_a = 6 :=
by
  -- Sorry is used to skip the proof; it's just a placeholder.
  sorry

end NUMINAMATH_GPT_vector_dot_product_l242_24270


namespace NUMINAMATH_GPT_solve_equation_l242_24207

theorem solve_equation (x : ℕ) (h : x = 88320) : x + 1315 + 9211 - 1569 = 97277 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l242_24207


namespace NUMINAMATH_GPT_no_integers_exist_l242_24233

theorem no_integers_exist :
  ¬ ∃ a b : ℤ, ∃ x y : ℤ, a^5 * b + 3 = x^3 ∧ a * b^5 + 3 = y^3 :=
by
  sorry

end NUMINAMATH_GPT_no_integers_exist_l242_24233


namespace NUMINAMATH_GPT_set_inter_complement_l242_24258

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 3}

-- Define set B
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem set_inter_complement :
  B ∩ (U \ A) = {2} :=
by
  sorry

end NUMINAMATH_GPT_set_inter_complement_l242_24258
