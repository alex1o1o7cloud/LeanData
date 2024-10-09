import Mathlib

namespace triangle_side_lengths_count_l1206_120664

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l1206_120664


namespace parking_lot_cars_l1206_120676

theorem parking_lot_cars (total_capacity : ℕ) (num_levels : ℕ) (parked_cars : ℕ) 
  (h1 : total_capacity = 425) (h2 : num_levels = 5) (h3 : parked_cars = 23) : 
  (total_capacity / num_levels) - parked_cars = 62 :=
by
  sorry

end parking_lot_cars_l1206_120676


namespace avg_of_14_23_y_is_21_l1206_120648

theorem avg_of_14_23_y_is_21 (y : ℝ) (h : (14 + 23 + y) / 3 = 21) : y = 26 :=
by
  sorry

end avg_of_14_23_y_is_21_l1206_120648


namespace original_price_is_1611_11_l1206_120633

theorem original_price_is_1611_11 (profit: ℝ) (rate: ℝ) (original_price: ℝ) (selling_price: ℝ) 
(h1: profit = 725) (h2: rate = 0.45) (h3: profit = rate * original_price) : 
original_price = 725 / 0.45 := 
sorry

end original_price_is_1611_11_l1206_120633


namespace passed_both_tests_l1206_120678

theorem passed_both_tests :
  ∀ (total_students passed_long_jump passed_shot_put failed_both passed_both: ℕ),
  total_students = 50 →
  passed_long_jump = 40 →
  passed_shot_put = 31 →
  failed_both = 4 →
  passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both = total_students →
  passed_both = 25 :=
by
  intros total_students passed_long_jump passed_shot_put failed_both passed_both h1 h2 h3 h4 h5
  -- proof can be skipped using sorry
  sorry

end passed_both_tests_l1206_120678


namespace shaded_region_area_l1206_120640

theorem shaded_region_area (RS : ℝ) (n_shaded : ℕ)
  (h1 : RS = 10) (h2 : n_shaded = 20) :
  (20 * (RS / (2 * Real.sqrt 2))^2) = 250 :=
by
  sorry

end shaded_region_area_l1206_120640


namespace sequence_form_l1206_120635

-- Defining the sequence a_n as a function f
def seq (f : ℕ → ℕ) : Prop :=
  ∃ c : ℝ, (0 < c) ∧ ∀ m n : ℕ, Nat.gcd (f m + n) (f n + m) > (c * (m + n))

-- Proving that if there exists such a sequence, then it is of the form n + c
theorem sequence_form (f : ℕ → ℕ) (h : seq f) :
  ∃ c : ℤ, ∀ n : ℕ, f n = n + c :=
sorry

end sequence_form_l1206_120635


namespace simplify_expression_correct_l1206_120606

noncomputable def simplify_expression : ℝ :=
  2 - 2 / (2 + Real.sqrt 5) - 2 / (2 - Real.sqrt 5)

theorem simplify_expression_correct : simplify_expression = 10 := by
  sorry

end simplify_expression_correct_l1206_120606


namespace avg_one_sixth_one_fourth_l1206_120691

theorem avg_one_sixth_one_fourth : (1 / 6 + 1 / 4) / 2 = 5 / 24 := by
  sorry

end avg_one_sixth_one_fourth_l1206_120691


namespace sum_composite_l1206_120656

theorem sum_composite (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 34 * a = 43 * b) : ∃ d : ℕ, d > 1 ∧ d < a + b ∧ d ∣ (a + b) :=
by
  sorry

end sum_composite_l1206_120656


namespace hallie_net_earnings_correct_l1206_120642

noncomputable def hallieNetEarnings : ℚ :=
  let monday_hours := 7
  let monday_rate := 10
  let monday_tips := 18
  let tuesday_hours := 5
  let tuesday_rate := 12
  let tuesday_tips := 12
  let wednesday_hours := 7
  let wednesday_rate := 10
  let wednesday_tips := 20
  let thursday_hours := 8
  let thursday_rate := 11
  let thursday_tips := 25
  let thursday_discount := 0.10
  let friday_hours := 6
  let friday_rate := 9
  let friday_tips := 15
  let income_tax := 0.05

  let monday_earnings := monday_hours * monday_rate
  let tuesday_earnings := tuesday_hours * tuesday_rate
  let wednesday_earnings := wednesday_hours * wednesday_rate
  let thursday_earnings := thursday_hours * thursday_rate
  let thursday_earnings_after_discount := thursday_earnings * (1 - thursday_discount)
  let friday_earnings := friday_hours * friday_rate

  let total_hourly_earnings := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings + friday_earnings
  let total_tips := monday_tips + tuesday_tips + wednesday_tips + thursday_tips + friday_tips

  let total_tax := total_hourly_earnings * income_tax
  
  let net_earnings := (total_hourly_earnings - total_tax) - (thursday_earnings - thursday_earnings_after_discount) + total_tips
  net_earnings

theorem hallie_net_earnings_correct : hallieNetEarnings = 406.10 := by
  sorry

end hallie_net_earnings_correct_l1206_120642


namespace pedro_furniture_area_l1206_120652

theorem pedro_furniture_area :
  let width : ℝ := 2
  let length : ℝ := 2.5
  let door_arc_area := (1 / 4) * Real.pi * (0.5 ^ 2)
  let window_arc_area := 2 * (1 / 2) * Real.pi * (0.5 ^ 2)
  let room_area := width * length
  room_area - door_arc_area - window_arc_area = (80 - 9 * Real.pi) / 16 := 
by
  sorry

end pedro_furniture_area_l1206_120652


namespace number_of_children_in_group_l1206_120651

-- Definitions based on the conditions
def num_adults : ℕ := 55
def meal_for_adults : ℕ := 70
def meal_for_children : ℕ := 90
def remaining_children_after_adults : ℕ := 81
def num_adults_eaten : ℕ := 7
def ratio_adult_to_child : ℚ := (70 : ℚ) / 90

-- Statement of the problem to prove number of children in the group
theorem number_of_children_in_group : 
  ∃ C : ℕ, 
    (meal_for_adults - num_adults_eaten) * (ratio_adult_to_child) = (remaining_children_after_adults) ∧
    C = remaining_children_after_adults := 
sorry

end number_of_children_in_group_l1206_120651


namespace aquarium_water_ratio_l1206_120693

theorem aquarium_water_ratio :
  let length := 4
  let width := 6
  let height := 3
  let volume := length * width * height
  let halfway_volume := volume / 2
  let water_after_cat := halfway_volume / 2
  let final_water := 54
  (final_water / water_after_cat) = 3 := by
  sorry

end aquarium_water_ratio_l1206_120693


namespace sec_150_eq_neg_two_div_sqrt_three_l1206_120674

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l1206_120674


namespace sufficient_condition_for_reciprocal_inequality_l1206_120657

theorem sufficient_condition_for_reciprocal_inequality (a b : ℝ) (h : b < a ∧ a < 0) : (1 / a) < (1 / b) :=
sorry

end sufficient_condition_for_reciprocal_inequality_l1206_120657


namespace simplify_and_calculate_expression_l1206_120623

theorem simplify_and_calculate_expression (a b : ℤ) (ha : a = -1) (hb : b = -2) :
  (2 * a + b) * (b - 2 * a) - (a - 3 * b) ^ 2 = -25 :=
by 
  -- We can use 'by' to start the proof and 'sorry' to skip it
  sorry

end simplify_and_calculate_expression_l1206_120623


namespace dyed_pink_correct_l1206_120638

def silk_dyed_green := 61921
def total_yards_dyed := 111421
def yards_dyed_pink := total_yards_dyed - silk_dyed_green

theorem dyed_pink_correct : yards_dyed_pink = 49500 := by 
  sorry

end dyed_pink_correct_l1206_120638


namespace passengers_landed_in_newberg_last_year_l1206_120663

theorem passengers_landed_in_newberg_last_year :
  let airport_a_on_time : ℕ := 16507
  let airport_a_late : ℕ := 256
  let airport_b_on_time : ℕ := 11792
  let airport_b_late : ℕ := 135
  airport_a_on_time + airport_a_late + airport_b_on_time + airport_b_late = 28690 :=
by
  let airport_a_on_time := 16507
  let airport_a_late := 256
  let airport_b_on_time := 11792
  let airport_b_late := 135
  show airport_a_on_time + airport_a_late + airport_b_on_time + airport_b_late = 28690
  sorry

end passengers_landed_in_newberg_last_year_l1206_120663


namespace proj_v_onto_w_l1206_120602

open Real

noncomputable def v : ℝ × ℝ := (8, -4)
noncomputable def w : ℝ × ℝ := (2, 3)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let coeff := dot_product v w / dot_product w w
  (coeff * w.1, coeff * w.2)

theorem proj_v_onto_w :
  projection v w = (8 / 13, 12 / 13) :=
by
  sorry

end proj_v_onto_w_l1206_120602


namespace percentage_wax_left_eq_10_l1206_120688

def total_amount_wax : ℕ := 
  let wax20 := 5 * 20
  let wax5 := 5 * 5
  let wax1 := 25 * 1
  wax20 + wax5 + wax1

def wax_used_for_new_candles : ℕ := 
  3 * 5

def percentage_wax_used (total_wax : ℕ) (wax_used : ℕ) : ℕ := 
  (wax_used * 100) / total_wax

theorem percentage_wax_left_eq_10 :
  percentage_wax_used total_amount_wax wax_used_for_new_candles = 10 :=
by
  sorry

end percentage_wax_left_eq_10_l1206_120688


namespace segments_can_form_triangle_l1206_120689

noncomputable def can_form_triangle (a b c : ℝ) : Prop :=
  a + b + c = 2 ∧ a + b > 1 ∧ a + c > b ∧ b + c > a

theorem segments_can_form_triangle (a b c : ℝ) (h : a + b + c = 2) : (a + b > 1) ↔ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end segments_can_form_triangle_l1206_120689


namespace find_beta_l1206_120619

open Real

theorem find_beta 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h1 : sin α = sqrt 5 / 5)
  (h2 : sin (α - β) = - sqrt 10 / 10):
  β = π / 4 :=
sorry

end find_beta_l1206_120619


namespace quad_form_b_c_sum_l1206_120665

theorem quad_form_b_c_sum :
  ∃ (b c : ℝ), (b + c = -10) ∧ (∀ x : ℝ, x^2 - 20 * x + 100 = (x + b)^2 + c) :=
by
  sorry

end quad_form_b_c_sum_l1206_120665


namespace percentage_sum_of_v_and_w_l1206_120629

variable {x y z v w : ℝ} 

theorem percentage_sum_of_v_and_w (h1 : 0.45 * z = 0.39 * y) (h2 : y = 0.75 * x) 
                                  (h3 : v = 0.80 * z) (h4 : w = 0.60 * y) :
                                  v + w = 0.97 * x :=
by 
  sorry

end percentage_sum_of_v_and_w_l1206_120629


namespace Sarah_correct_responses_l1206_120672

theorem Sarah_correct_responses : ∃ x : ℕ, x ≥ 22 ∧ (7 * x - (26 - x) + 4 ≥ 150) :=
by
  sorry

end Sarah_correct_responses_l1206_120672


namespace range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3_l1206_120607

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3 :
  {x : ℝ | f (2 * x) > f (x + 3)} = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3_l1206_120607


namespace spherical_to_rectangular_conversion_l1206_120621

noncomputable def convert_spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  convert_spherical_to_rectangular 8 (5 * Real.pi / 4) (Real.pi / 4) = (-4, -4, 4 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l1206_120621


namespace justin_run_time_l1206_120634

theorem justin_run_time : 
  let flat_ground_rate := 2 / 2 -- Justin runs 2 blocks in 2 minutes on flat ground
  let uphill_rate := 2 / 3 -- Justin runs 2 blocks in 3 minutes uphill
  let total_blocks := 10 -- Justin is 10 blocks from home
  let uphill_blocks := 6 -- 6 of those blocks are uphill
  let flat_ground_blocks := total_blocks - uphill_blocks -- Remainder are flat ground
  let flat_ground_time := flat_ground_blocks * flat_ground_rate
  let uphill_time := uphill_blocks * uphill_rate
  let total_time := flat_ground_time + uphill_time
  total_time = 13 := 
by 
  sorry

end justin_run_time_l1206_120634


namespace sum_of_numbers_in_third_column_is_96_l1206_120653

theorem sum_of_numbers_in_third_column_is_96 :
  ∃ (a : ℕ), (136 = a + 16 * a) ∧ (272 = 2 * a + 32 * a) ∧ (12 * a = 96) :=
by
  let a := 8
  have h1 : 136 = a + 16 * a := by sorry  -- Proof here that 136 = 8 + 16 * 8
  have h2 : 272 = 2 * a + 32 * a := by sorry  -- Proof here that 272 = 2 * 8 + 32 * 8
  have h3 : 12 * a = 96 := by sorry  -- Proof here that 12 * 8 = 96
  existsi a
  exact ⟨h1, h2, h3⟩

end sum_of_numbers_in_third_column_is_96_l1206_120653


namespace frozenFruitSold_l1206_120671

variable (totalFruit : ℕ) (freshFruit : ℕ)

-- Define the condition that the total fruit sold is 9792 pounds
def totalFruitSold := totalFruit = 9792

-- Define the condition that the fresh fruit sold is 6279 pounds
def freshFruitSold := freshFruit = 6279

-- Define the question as a Lean statement
theorem frozenFruitSold
  (h1 : totalFruitSold totalFruit)
  (h2 : freshFruitSold freshFruit) :
  totalFruit - freshFruit = 3513 := by
  sorry

end frozenFruitSold_l1206_120671


namespace probability_black_ball_BoxB_higher_l1206_120615

def boxA_red_balls : ℕ := 40
def boxA_black_balls : ℕ := 10
def boxB_red_balls : ℕ := 60
def boxB_black_balls : ℕ := 40
def boxB_white_balls : ℕ := 50

theorem probability_black_ball_BoxB_higher :
  (boxA_black_balls : ℚ) / (boxA_red_balls + boxA_black_balls) <
  (boxB_black_balls : ℚ) / (boxB_red_balls + boxB_black_balls + boxB_white_balls) :=
by
  sorry

end probability_black_ball_BoxB_higher_l1206_120615


namespace find_x_values_l1206_120661

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem find_x_values (x : ℝ) :
  (f (f x) = f x) ↔ (x = 0 ∨ x = 2 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end find_x_values_l1206_120661


namespace distance_C_to_C_l1206_120641

noncomputable def C : ℝ × ℝ := (-3, 2)
noncomputable def C' : ℝ × ℝ := (3, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_C_to_C' : distance C C' = 2 * Real.sqrt 13 := by
  sorry

end distance_C_to_C_l1206_120641


namespace greatest_b_value_l1206_120603

theorem greatest_b_value (b : ℝ) : 
  (-b^3 + b^2 + 7 * b - 10 ≥ 0) ↔ b ≤ 4 + Real.sqrt 6 :=
sorry

end greatest_b_value_l1206_120603


namespace betty_total_cost_l1206_120609

theorem betty_total_cost :
    (6 * 2.5) + (4 * 1.25) + (8 * 3) = 44 :=
by
    sorry

end betty_total_cost_l1206_120609


namespace tan_half_alpha_third_quadrant_sine_cos_expression_l1206_120617

-- Problem (1): Proof for tan(α/2) = -5 given the conditions
theorem tan_half_alpha_third_quadrant (α : ℝ) (h1 : α ∈ Set.Ioo π (3 * π / 2))
  (h2 : Real.sin α = -5/13) :
  Real.tan (α / 2) = -5 := by
  sorry

-- Problem (2): Proof for sin²(π - α) + 2sin(3π/2 + α)cos(π/2 + α) = 8/5 given the condition
theorem sine_cos_expression (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (π - α) ^ 2 + 2 * Real.sin (3 * π / 2 + α) * Real.cos (π / 2 + α) = 8 / 5 := by
  sorry

end tan_half_alpha_third_quadrant_sine_cos_expression_l1206_120617


namespace votes_cast_is_330_l1206_120622

variable (T A F : ℝ)

theorem votes_cast_is_330
  (h1 : A = 0.40 * T)
  (h2 : F = A + 66)
  (h3 : T = F + A) :
  T = 330 :=
by
  sorry

end votes_cast_is_330_l1206_120622


namespace factors_of_1320_l1206_120694

theorem factors_of_1320 : ∃ n : ℕ, n = 24 ∧ ∃ (a b c d : ℕ),
  1320 = 2^a * 3^b * 5^c * 11^d ∧ (a = 0 ∨ a = 1 ∨ a = 2) ∧ (b = 0 ∨ b = 1) ∧ (c = 0 ∨ c = 1) ∧ (d = 0 ∨ d = 1) :=
by {
  sorry
}

end factors_of_1320_l1206_120694


namespace determine_m_n_l1206_120662

theorem determine_m_n 
  {a b c d m n : ℕ} 
  (h₁ : a + b + c + d = m^2)
  (h₂ : a^2 + b^2 + c^2 + d^2 = 1989)
  (h₃ : max (max a b) (max c d) = n^2) 
  : m = 9 ∧ n = 6 := by 
  sorry

end determine_m_n_l1206_120662


namespace barium_oxide_amount_l1206_120677

theorem barium_oxide_amount (BaO H2O BaOH₂ : ℕ) 
  (reaction : BaO + H2O = BaOH₂) 
  (molar_ratio : BaOH₂ = BaO) 
  (required_BaOH₂ : BaOH₂ = 2) :
  BaO = 2 :=
by 
  sorry

end barium_oxide_amount_l1206_120677


namespace value_of_a_plus_b_l1206_120698

theorem value_of_a_plus_b (a b : ℝ) (h : |a - 2| = -(b + 5)^2) : a + b = -3 :=
sorry

end value_of_a_plus_b_l1206_120698


namespace perpendicular_lines_sum_l1206_120636

theorem perpendicular_lines_sum (a b : ℝ) :
  (∃ (x y : ℝ), 2 * x - 5 * y + b = 0 ∧ a * x + 4 * y - 2 = 0 ∧ x = 1 ∧ y = -2) ∧
  (-a / 4) * (2 / 5) = -1 →
  a + b = -2 :=
by
  sorry

end perpendicular_lines_sum_l1206_120636


namespace bicycle_distance_l1206_120695

theorem bicycle_distance (b t : ℝ) (h : t ≠ 0) :
  let rate := (b / 2) / t / 3
  let total_seconds := 5 * 60
  rate * total_seconds = 50 * b / t := by
    sorry

end bicycle_distance_l1206_120695


namespace fraction_geq_81_l1206_120601

theorem fraction_geq_81 {p q r s : ℝ} (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) :
  ((p^2 + p + 1) * (q^2 + q + 1) * (r^2 + r + 1) * (s^2 + s + 1)) / (p * q * r * s) ≥ 81 :=
by
  sorry

end fraction_geq_81_l1206_120601


namespace divisibility_of_product_l1206_120686

def three_consecutive_integers (a1 a2 a3 : ℤ) : Prop :=
  a1 = a2 - 1 ∧ a3 = a2 + 1

theorem divisibility_of_product (a1 a2 a3 : ℤ) (h : three_consecutive_integers a1 a2 a3) : 
  a2^3 ∣ (a1 * a2 * a3 + a2) :=
by
  cases h with
  | intro ha1 ha3 =>
    sorry

end divisibility_of_product_l1206_120686


namespace find_x_plus_y_l1206_120682

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y + 3 * Real.sin y = 2005) (h3 : 0 ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2009 + Real.pi / 2 := 
sorry

end find_x_plus_y_l1206_120682


namespace multiple_choice_options_l1206_120659

-- Define the problem conditions
def num_true_false_combinations : ℕ := 14
def num_possible_keys (n : ℕ) : ℕ := num_true_false_combinations * n^2
def total_keys : ℕ := 224

-- The theorem problem
theorem multiple_choice_options : ∃ n : ℕ, num_possible_keys n = total_keys ∧ n = 4 := by
  -- We don't need to provide the proof, so we use sorry. 
  sorry

end multiple_choice_options_l1206_120659


namespace janet_saves_time_l1206_120645

theorem janet_saves_time (looking_time_per_day : ℕ := 8) (complaining_time_per_day : ℕ := 3) (days_per_week : ℕ := 7) :
  (looking_time_per_day + complaining_time_per_day) * days_per_week = 77 := 
sorry

end janet_saves_time_l1206_120645


namespace train_length_proper_l1206_120618

noncomputable def train_length (speed distance_time pass_time : ℝ) : ℝ :=
  speed * pass_time

axiom speed_of_train : ∀ (distance_time : ℝ), 
  (10 * 1000 / (15 * 60)) = 11.11

theorem train_length_proper :
  train_length 11.11 900 10 = 111.1 := by
  sorry

end train_length_proper_l1206_120618


namespace book_collection_example_l1206_120660

theorem book_collection_example :
  ∃ (P C B : ℕ), 
    (P : ℚ) / C = 3 / 2 ∧ 
    (C : ℚ) / B = 4 / 3 ∧ 
    P + C + B = 3002 ∧ 
    P + C + B > 3000 :=
by
  sorry

end book_collection_example_l1206_120660


namespace order_of_x_y_z_l1206_120612

theorem order_of_x_y_z (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  let x : ℝ := (a + b) * (c + d)
  let y : ℝ := (a + c) * (b + d)
  let z : ℝ := (a + d) * (b + c)
  x < y ∧ y < z :=
by
  let x : ℝ := (a + b) * (c + d)
  let y : ℝ := (a + c) * (b + d)
  let z : ℝ := (a + d) * (b + c)
  sorry

end order_of_x_y_z_l1206_120612


namespace constant_c_for_local_maximum_l1206_120658

theorem constant_c_for_local_maximum (c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x * (x - c) ^ 2) (h2 : ∃ δ > 0, ∀ x, |x - 2| < δ → f x ≤ f 2) : c = 6 :=
sorry

end constant_c_for_local_maximum_l1206_120658


namespace geometric_sequence_general_term_geometric_sequence_sum_n_l1206_120628

theorem geometric_sequence_general_term (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) : 
  a n = 48 * (1 / 2) ^ (n - 1) := 
by {
  sorry
}

theorem geometric_sequence_sum_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) 
  (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) 
  (h4 : ∀ (n : ℕ), S n = 48 * (1 - (1 / 2) ^ n) / (1 - 1 / 2))
  (h5 : S 5 = 93) : 
  5 = 5 := 
by {
  sorry
}

end geometric_sequence_general_term_geometric_sequence_sum_n_l1206_120628


namespace ellipse_range_of_k_l1206_120626

theorem ellipse_range_of_k (k : ℝ) :
  (1 - k > 0) ∧ (1 + k > 0) ∧ (1 - k ≠ 1 + k) ↔ (-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1) :=
by
  sorry

end ellipse_range_of_k_l1206_120626


namespace train_speed_l1206_120696

/-- Proof that calculates the speed of a train given the times to pass a man and a platform,
and the length of the platform, and shows it equals 54.00432 km/hr. -/
theorem train_speed (L V : ℝ) 
  (platform_length : ℝ := 360.0288)
  (time_to_pass_man : ℝ := 20)
  (time_to_pass_platform : ℝ := 44)
  (equation1 : L = V * time_to_pass_man)
  (equation2 : L + platform_length = V * time_to_pass_platform) :
  V = 15.0012 → V * 3.6 = 54.00432 :=
by sorry

end train_speed_l1206_120696


namespace log_conversion_l1206_120679

theorem log_conversion (a b : ℝ) (h1 : a = Real.log 225 / Real.log 8) (h2 : b = Real.log 15 / Real.log 2) : a = (2 * b) / 3 := 
sorry

end log_conversion_l1206_120679


namespace geometric_seq_sum_S40_l1206_120632

noncomputable def geometric_seq_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q ≠ 1 then a1 * (1 - q^n) / (1 - q) else a1 * n

theorem geometric_seq_sum_S40 :
  ∃ (a1 q : ℝ), (0 < q ∧ q ≠ 1) ∧ 
                geometric_seq_sum a1 q 10 = 10 ∧
                geometric_seq_sum a1 q 30 = 70 ∧
                geometric_seq_sum a1 q 40 = 150 :=
by
  sorry

end geometric_seq_sum_S40_l1206_120632


namespace add_to_make_divisible_l1206_120655

theorem add_to_make_divisible :
  ∃ n, n = 34 ∧ ∃ k : ℕ, 758492136547 + n = 51 * k := by
  sorry

end add_to_make_divisible_l1206_120655


namespace find_f2_l1206_120646

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f2 (h : ∀ x : ℝ, f x + 3 * f (8 - x) = x) : f 2 = 2 :=
by
  sorry

end find_f2_l1206_120646


namespace number_of_girls_in_class_l1206_120610

variable (B S G : ℕ)

theorem number_of_girls_in_class
  (h1 : (3 / 4 : ℚ) * B = 18)
  (h2 : B = (2 / 3 : ℚ) * S) :
  G = S - B → G = 12 := by
  intro hg
  sorry

end number_of_girls_in_class_l1206_120610


namespace find_greatest_and_second_greatest_problem_solution_l1206_120675

theorem find_greatest_and_second_greatest
  (a b c d : ℝ)
  (ha : a = 4 ^ (1 / 4))
  (hb : b = 5 ^ (1 / 5))
  (hc : c = 16 ^ (1 / 16))
  (hd : d = 25 ^ (1 / 25))
  : (a > b) ∧ (b > c) ∧ (c > d) :=
by 
  sorry

def greatest_and_second_greatest_eq (x1 x2 : ℝ) : Prop :=
  x1 = 4 ^ (1 / 4) ∧ x2 = 5 ^ (1 / 5)

theorem problem_solution (a b c d : ℝ)
  (ha : a = 4 ^ (1 / 4))
  (hb : b = 5 ^ (1 / 5))
  (hc : c = 16 ^ (1 / 16))
  (hd : d = 25 ^ (1 / 25))
  : greatest_and_second_greatest_eq a b :=
by 
  sorry

end find_greatest_and_second_greatest_problem_solution_l1206_120675


namespace tom_average_speed_l1206_120680

theorem tom_average_speed 
  (karen_speed : ℕ) (tom_distance : ℕ) (karen_advantage : ℕ) (delay : ℚ)
  (h1 : karen_speed = 60)
  (h2 : tom_distance = 24)
  (h3 : karen_advantage = 4)
  (h4 : delay = 4/60) :
  ∃ (v : ℚ), v = 45 := by
  sorry

end tom_average_speed_l1206_120680


namespace max_distance_difference_l1206_120684

-- Given definitions and conditions
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 15 = 1
def circle1 (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1

-- Main theorem to prove the maximum value of |PM| - |PN|
theorem max_distance_difference (P M N : ℝ × ℝ) :
  hyperbola P.1 P.2 →
  circle1 M.1 M.2 →
  circle2 N.1 N.2 →
  ∃ max_val : ℝ, max_val = 5 :=
by
  -- Proof skipped, only statement is required
  sorry

end max_distance_difference_l1206_120684


namespace find_a2023_l1206_120673

noncomputable def a_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n, 2 * S n = a n * (a n + 1))

theorem find_a2023 (a S : ℕ → ℕ) 
  (h_seq : a_sequence a S)
  (h_pos : ∀ n, 0 < a n) 
  : a 2023 = 2023 :=
sorry

end find_a2023_l1206_120673


namespace dice_probability_l1206_120605

def first_die_prob : ℚ := 3 / 8
def second_die_prob : ℚ := 3 / 4
def combined_prob : ℚ := first_die_prob * second_die_prob

theorem dice_probability :
  combined_prob = 9 / 32 :=
by
  -- Here we write the proof steps.
  sorry

end dice_probability_l1206_120605


namespace find_m_value_l1206_120654

theorem find_m_value (m x y : ℝ) (hx : x = 2) (hy : y = -1) (h_eq : m * x - y = 3) : m = 1 :=
by
  sorry

end find_m_value_l1206_120654


namespace small_seat_capacity_indeterminate_l1206_120667

-- Conditions
def small_seats : ℕ := 3
def large_seats : ℕ := 7
def capacity_per_large_seat : ℕ := 12
def total_large_capacity : ℕ := 84

theorem small_seat_capacity_indeterminate
  (h1 : large_seats * capacity_per_large_seat = total_large_capacity)
  (h2 : ∀ s : ℕ, ∃ p : ℕ, p ≠ s * capacity_per_large_seat) :
  ¬ ∃ n : ℕ, ∀ m : ℕ, small_seats * m = n * small_seats :=
by {
  sorry
}

end small_seat_capacity_indeterminate_l1206_120667


namespace total_books_l1206_120668

def school_books : ℕ := 19
def sports_books : ℕ := 39

theorem total_books : school_books + sports_books = 58 := by
  sorry

end total_books_l1206_120668


namespace triangle_area_l1206_120614

noncomputable def area_triangle_ACD (t p : ℝ) : ℝ :=
  1 / 2 * p * (t - 2)

theorem triangle_area (t p : ℝ) (ht : 0 < t ∧ t < 12) (hp : 0 < p ∧ p < 12) :
  area_triangle_ACD t p = 1 / 2 * p * (t - 2) :=
sorry

end triangle_area_l1206_120614


namespace quadratic_one_solution_set_l1206_120604

theorem quadratic_one_solution_set (a : ℝ) :
  (∃ x : ℝ, ax^2 + x + 1 = 0 ∧ (∀ y : ℝ, ax^2 + x + 1 = 0 → y = x)) ↔ (a = 0 ∨ a = 1 / 4) :=
by sorry

end quadratic_one_solution_set_l1206_120604


namespace total_cost_correct_l1206_120608

noncomputable def cost_4_canvases : ℕ := 40
noncomputable def cost_paints : ℕ := cost_4_canvases / 2
noncomputable def cost_easel : ℕ := 15
noncomputable def cost_paintbrushes : ℕ := 15
noncomputable def total_cost : ℕ := cost_4_canvases + cost_paints + cost_easel + cost_paintbrushes

theorem total_cost_correct : total_cost = 90 :=
by
  unfold total_cost
  unfold cost_4_canvases
  unfold cost_paints
  unfold cost_easel
  unfold cost_paintbrushes
  simp
  sorry

end total_cost_correct_l1206_120608


namespace arithmetic_sequence_S30_l1206_120649

theorem arithmetic_sequence_S30
  (S : ℕ → ℕ)
  (h_arith_seq: ∀ m : ℕ, 2 * (S (2 * m) - S m) = S m + S (3 * m) - S (2 * m))
  (h_S10: S 10 = 4)
  (h_S20: S 20 = 20) :
  S 30 = 48 := 
by
  sorry

end arithmetic_sequence_S30_l1206_120649


namespace pet_store_cages_l1206_120637

def initial_puppies : ℕ := 18
def puppies_sold : ℕ := 3
def puppies_per_cage : ℕ := 5

def remaining_puppies : ℕ := initial_puppies - puppies_sold
def number_of_cages : ℕ := remaining_puppies / puppies_per_cage

theorem pet_store_cages : number_of_cages = 3 :=
by sorry

end pet_store_cages_l1206_120637


namespace binom_divisibility_l1206_120685

theorem binom_divisibility (k n : ℕ) (p : ℕ) (h1 : k > 1) (h2 : n > 1) 
  (h3 : p = 2 * k - 1) (h4 : Nat.Prime p) (h5 : p ∣ (Nat.choose n 2 - Nat.choose k 2)) : 
  p^2 ∣ (Nat.choose n 2 - Nat.choose k 2) := 
sorry

end binom_divisibility_l1206_120685


namespace Jim_runs_total_distance_l1206_120692

-- Definitions based on the conditions
def miles_day_1 := 5
def miles_day_31 := 10
def miles_day_61 := 20

def days_period := 30

-- Mathematical statement to prove
theorem Jim_runs_total_distance :
  let total_distance := 
    (miles_day_1 * days_period) + 
    (miles_day_31 * days_period) + 
    (miles_day_61 * days_period)
  total_distance = 1050 := by
  sorry

end Jim_runs_total_distance_l1206_120692


namespace maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism_l1206_120687

open Real

theorem maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism 
  (a b : ℝ)
  (ha : a^2 + b^2 = 25) 
  (AC_eq_5 : AC = 5) :
  ∃ (r : ℝ), 4 * π * r^2 = 25 * (3 - 3 * sqrt 2) * π :=
sorry

end maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism_l1206_120687


namespace soccer_team_percentage_l1206_120690

theorem soccer_team_percentage (total_games won_games : ℕ) (h1 : total_games = 140) (h2 : won_games = 70) :
  (won_games / total_games : ℚ) * 100 = 50 := by
  sorry

end soccer_team_percentage_l1206_120690


namespace delta_value_l1206_120613

theorem delta_value (Delta : ℤ) (h : 5 * (-3) = Delta - 3) : Delta = -12 := 
by 
  sorry

end delta_value_l1206_120613


namespace part_one_part_one_equality_part_two_l1206_120620

-- Given constants and their properties
variables (a b c d : ℝ)

-- Statement for the first problem
theorem part_one : a^6 + b^6 + c^6 + d^6 - 6 * a * b * c * d ≥ -2 :=
sorry

-- Statement for the equality condition in the first problem
theorem part_one_equality (h : |a| = 1 ∧ |b| = 1 ∧ |c| = 1 ∧ |d| = 1) : 
  a^6 + b^6 + c^6 + d^6 - 6 * a * b * c * d = -2 :=
sorry

-- Statement for the second problem (existence of Mk for k >= 4 and odd)
theorem part_two (k : ℕ) (hk1 : 4 ≤ k) (hk2 : k % 2 = 1) : ∃ Mk : ℝ, ∀ a b c d : ℝ, a^k + b^k + c^k + d^k - k * a * b * c * d ≥ Mk :=
sorry

end part_one_part_one_equality_part_two_l1206_120620


namespace num_square_tiles_is_zero_l1206_120683

def triangular_tiles : ℕ := sorry
def square_tiles : ℕ := sorry
def hexagonal_tiles : ℕ := sorry

axiom tile_count_eq : triangular_tiles + square_tiles + hexagonal_tiles = 30
axiom edge_count_eq : 3 * triangular_tiles + 4 * square_tiles + 6 * hexagonal_tiles = 120

theorem num_square_tiles_is_zero : square_tiles = 0 :=
by
  sorry

end num_square_tiles_is_zero_l1206_120683


namespace intersection_complement_l1206_120630

def U : Set ℤ := Set.univ
def M : Set ℤ := {1, 2}
def P : Set ℤ := {-2, -1, 0, 1, 2}
def CUM : Set ℤ := {x : ℤ | x ∉ M}

theorem intersection_complement :
  P ∩ CUM = {-2, -1, 0} :=
by
  sorry

end intersection_complement_l1206_120630


namespace angle_bisector_relation_l1206_120650

theorem angle_bisector_relation (a b : ℝ) (h : a = -b ∨ a = -b) : a = -b :=
sorry

end angle_bisector_relation_l1206_120650


namespace train_length_l1206_120647

-- Definitions of the conditions as Lean terms/functions
def V (L : ℕ) := (L + 170) / 15
def U (L : ℕ) := (L + 250) / 20

-- The theorem to prove that the length of the train is 70 meters.
theorem train_length : ∃ L : ℕ, (V L = U L) → L = 70 := by
  sorry

end train_length_l1206_120647


namespace find_sum_of_coefficients_l1206_120697

theorem find_sum_of_coefficients : 
  (∃ m n p : ℕ, 
    (n.gcd p = 1) ∧ 
    m + 36 = 72 ∧
    n + 33*3 = 103 ∧ 
    p = 3 ∧ 
    (72 + 33 * ℼ + (8 * (1/8 * (4 * π / 3))) + 36) = m + n * π / p) → 
  m + n + p = 430 :=
by {
  sorry
}

end find_sum_of_coefficients_l1206_120697


namespace intersection_points_hyperbola_l1206_120666

theorem intersection_points_hyperbola (t : ℝ) :
  ∃ x y : ℝ, (2 * t * x - 3 * y - 4 * t = 0) ∧ (2 * x - 3 * t * y + 4 = 0) ∧ 
  (x^2 / 4 - y^2 / (9 / 16) = 1) :=
sorry

end intersection_points_hyperbola_l1206_120666


namespace max_min_z_l1206_120639

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 + 4*y^2 = 4*x

-- Define the function z
def z (x y : ℝ) : ℝ :=
  x^2 - y^2

-- Define the required points
def P1 (x y : ℝ) :=
  x = 4 ∧ y = 0

def P2 (x y : ℝ) :=
  x = 2/5 ∧ (y = 3/5 ∨ y = -3/5)

-- Theorem stating the required conditions
theorem max_min_z (x y : ℝ) (h : on_ellipse x y) :
  (P1 x y → z x y = 16) ∧ (P2 x y → z x y = -1/5) :=
by
  sorry

end max_min_z_l1206_120639


namespace equalize_costs_l1206_120627

theorem equalize_costs (A B : ℝ) (h_lt : A < B) :
  (B - A) / 2 = (A + B) / 2 - A :=
by sorry

end equalize_costs_l1206_120627


namespace div_d_a_value_l1206_120611

variable {a b c d : ℚ}

theorem div_d_a_value (h1 : a / b = 3) (h2 : b / c = 5 / 3) (h3 : c / d = 2) : d / a = 1 / 10 := by
  sorry

end div_d_a_value_l1206_120611


namespace parabola_focus_distance_l1206_120600

noncomputable def parabolic_distance (x y : ℝ) : ℝ :=
  x + x / 2

theorem parabola_focus_distance : 
  (∃ y : ℝ, (1 : ℝ) = (1 / 2) * y^2) → 
  parabolic_distance 1 y = 3 / 2 :=
by 
  intros hy
  obtain ⟨y, hy⟩ := hy
  unfold parabolic_distance
  have hx : 1 = (1 / 2) * y^2 := hy
  sorry

end parabola_focus_distance_l1206_120600


namespace allowance_is_14_l1206_120644

def initial := 11
def spent := 3
def final := 22

def allowance := final - (initial - spent)

theorem allowance_is_14 : allowance = 14 := by
  -- proof goes here
  sorry

end allowance_is_14_l1206_120644


namespace moli_bought_7_clips_l1206_120624

theorem moli_bought_7_clips (R C S x : ℝ) 
  (h1 : 3*R + x*C + S = 120) 
  (h2 : 4*R + 10*C + S = 164) 
  (h3 : R + C + S = 32) : 
  x = 7 := 
by
  sorry

end moli_bought_7_clips_l1206_120624


namespace train_speed_l1206_120681

def train_length : ℝ := 1000  -- train length in meters
def time_to_cross_pole : ℝ := 200  -- time to cross the pole in seconds

theorem train_speed : train_length / time_to_cross_pole = 5 := by
  sorry

end train_speed_l1206_120681


namespace total_people_transport_l1206_120670

-- Define the conditions
def boatA_trips_day1 := 7
def boatB_trips_day1 := 5
def boatA_capacity := 20
def boatB_capacity := 15
def boatA_trips_day2 := 5
def boatB_trips_day2 := 6

-- Define the theorem statement
theorem total_people_transport :
  (boatA_trips_day1 * boatA_capacity + boatB_trips_day1 * boatB_capacity) +
  (boatA_trips_day2 * boatA_capacity + boatB_trips_day2 * boatB_capacity)
  = 405 := 
  by
  sorry

end total_people_transport_l1206_120670


namespace max_sum_x_y_l1206_120625

theorem max_sum_x_y 
  (x y : ℝ)
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
sorry

end max_sum_x_y_l1206_120625


namespace total_birds_on_fence_l1206_120643

-- Definitions based on conditions.
def initial_birds : ℕ := 12
def additional_birds : ℕ := 8

-- Theorem corresponding to the problem statement.
theorem total_birds_on_fence : initial_birds + additional_birds = 20 := by 
  sorry

end total_birds_on_fence_l1206_120643


namespace good_horse_catchup_l1206_120669

theorem good_horse_catchup (x : ℕ) : 240 * x = 150 * (x + 12) :=
by sorry

end good_horse_catchup_l1206_120669


namespace average_age_of_women_l1206_120699

theorem average_age_of_women (A : ℕ) :
  (6 * (A + 2) = 6 * A - 22 + W) → (W / 2 = 17) :=
by
  intro h
  sorry

end average_age_of_women_l1206_120699


namespace magnitude_a_minus_2b_l1206_120616

noncomputable def magnitude_of_vector_difference : ℝ :=
  let a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
  let b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2)

theorem magnitude_a_minus_2b :
  let a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
  let b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 3 :=
by
  sorry

end magnitude_a_minus_2b_l1206_120616


namespace slope_transformation_l1206_120631

theorem slope_transformation :
  ∀ (b : ℝ), ∃ k : ℝ, 
  (∀ x : ℝ, k * x + b = k * (x + 4) + b + 1) → k = -1/4 :=
by
  intros b
  use -1/4
  intros h
  sorry

end slope_transformation_l1206_120631
