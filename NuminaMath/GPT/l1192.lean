import Mathlib

namespace stickers_per_page_l1192_119290

theorem stickers_per_page (n_pages total_stickers : ℕ) (h_n_pages : n_pages = 22) (h_total_stickers : total_stickers = 220) : total_stickers / n_pages = 10 :=
by
  sorry

end stickers_per_page_l1192_119290


namespace max_gcd_bn_bnp1_l1192_119211

def b_n (n : ℕ) : ℤ := (7 ^ n - 4) / 3
def b_n_plus_1 (n : ℕ) : ℤ := (7 ^ (n + 1) - 4) / 3

theorem max_gcd_bn_bnp1 (n : ℕ) : ∃ d_max : ℕ, (∀ d : ℕ, (gcd (b_n n) (b_n_plus_1 n) ≤ d) → d ≤ d_max) ∧ d_max = 3 :=
sorry

end max_gcd_bn_bnp1_l1192_119211


namespace find_k_l1192_119219

-- Define the conditions
variables (a b : Real) (x y : Real)

-- The problem's conditions
def tan_x : Prop := Real.tan x = a / b
def tan_2x : Prop := Real.tan (x + x) = b / (a + b)
def y_eq_x : Prop := y = x

-- The goal to prove
theorem find_k (ha : tan_x a b x) (hb : tan_2x a b x) (hy : y_eq_x x y) :
  ∃ k, x = Real.arctan k ∧ k = 1 / (a + 2) :=
sorry

end find_k_l1192_119219


namespace betty_afternoon_catch_l1192_119279

def flies_eaten_per_day := 2
def days_in_week := 7
def flies_needed_for_week := days_in_week * flies_eaten_per_day
def flies_caught_morning := 5
def additional_flies_needed := 4
def flies_currently_have := flies_needed_for_week - additional_flies_needed
def flies_caught_afternoon := flies_currently_have - flies_caught_morning
def flies_escaped := 1

theorem betty_afternoon_catch :
  flies_caught_afternoon + flies_escaped = 6 :=
by
  sorry

end betty_afternoon_catch_l1192_119279


namespace find_sum_l1192_119222

theorem find_sum {x y : ℝ} (h1 : x = 13.0) (h2 : x + y = 24) : 7 * x + 5 * y = 146 := 
by
  sorry

end find_sum_l1192_119222


namespace option_D_is_empty_l1192_119259

theorem option_D_is_empty :
  {x : ℝ | x^2 + x + 1 = 0} = ∅ :=
by
  sorry

end option_D_is_empty_l1192_119259


namespace expected_winnings_is_0_25_l1192_119292

def prob_heads : ℚ := 3 / 8
def prob_tails : ℚ := 1 / 4
def prob_edge  : ℚ := 1 / 8
def prob_disappear : ℚ := 1 / 4

def winnings_heads : ℚ := 2
def winnings_tails : ℚ := 5
def winnings_edge  : ℚ := -2
def winnings_disappear : ℚ := -6

def expected_winnings : ℚ := 
  prob_heads * winnings_heads +
  prob_tails * winnings_tails +
  prob_edge  * winnings_edge +
  prob_disappear * winnings_disappear

theorem expected_winnings_is_0_25 : expected_winnings = 0.25 := by
  sorry

end expected_winnings_is_0_25_l1192_119292


namespace volume_of_new_cube_is_2744_l1192_119294

-- Define the volume function for a cube given side length
def volume_of_cube (side : ℝ) : ℝ := side ^ 3

-- Given the original cube with a specific volume
def original_volume : ℝ := 343

-- Find the side length of the original cube by taking the cube root of the volume
def original_side_length := (original_volume : ℝ)^(1/3)

-- The side length of the new cube is twice the side length of the original cube
def new_side_length := 2 * original_side_length

-- The volume of the new cube should be calculated
def new_volume := volume_of_cube new_side_length

-- Theorem stating that the new volume is 2744 cubic feet
theorem volume_of_new_cube_is_2744 : new_volume = 2744 := sorry

end volume_of_new_cube_is_2744_l1192_119294


namespace simplify_expression_l1192_119297

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem simplify_expression :
  (1/2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b) = 6 • b - 3 • a :=
by sorry

end simplify_expression_l1192_119297


namespace circle_radius_zero_l1192_119248

theorem circle_radius_zero : ∀ (x y : ℝ), x^2 + 10 * x + y^2 - 4 * y + 29 = 0 → 0 = 0 :=
by intro x y h
   sorry

end circle_radius_zero_l1192_119248


namespace smallest_possible_value_l1192_119288

theorem smallest_possible_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a^2 + b^2) / (a * b) + (a * b) / (a^2 + b^2) ≥ 2 :=
sorry

end smallest_possible_value_l1192_119288


namespace stanley_walk_distance_l1192_119235

variable (run_distance walk_distance : ℝ)

theorem stanley_walk_distance : 
  run_distance = 0.4 ∧ run_distance = walk_distance + 0.2 → walk_distance = 0.2 :=
by
  sorry

end stanley_walk_distance_l1192_119235


namespace polynomial_has_root_l1192_119231

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end polynomial_has_root_l1192_119231


namespace algebra_expr_solution_l1192_119268

theorem algebra_expr_solution (a b : ℝ) (h : 2 * a - b = 5) : 2 * b - 4 * a + 8 = -2 :=
by
  sorry

end algebra_expr_solution_l1192_119268


namespace major_axis_length_l1192_119236

noncomputable def length_of_major_axis (f1 f2 : ℝ × ℝ) (tangent_y_axis : Bool) (tangent_line_y : ℝ) : ℝ :=
  if f1 = (-Real.sqrt 5, 2) ∧ f2 = (Real.sqrt 5, 2) ∧ tangent_y_axis ∧ tangent_line_y = 1 then 2
  else 0

theorem major_axis_length :
  length_of_major_axis (-Real.sqrt 5, 2) (Real.sqrt 5, 2) true 1 = 2 :=
by
  sorry

end major_axis_length_l1192_119236


namespace truncated_pyramid_smaller_base_area_l1192_119243

noncomputable def smaller_base_area (a : ℝ) (α β : ℝ) : ℝ :=
  (a^2 * (Real.sin (α - β))^2) / (Real.sin (α + β))^2

theorem truncated_pyramid_smaller_base_area (a α β : ℝ) :
  smaller_base_area a α β = (a^2 * (Real.sin (α - β))^2) / (Real.sin (α + β))^2 :=
by
  unfold smaller_base_area
  sorry

end truncated_pyramid_smaller_base_area_l1192_119243


namespace manicure_cost_before_tip_l1192_119206

theorem manicure_cost_before_tip (total_paid : ℝ) (tip_percentage : ℝ) (cost_before_tip : ℝ) : 
  total_paid = 39 → tip_percentage = 0.30 → total_paid = cost_before_tip + tip_percentage * cost_before_tip → cost_before_tip = 30 :=
by
  intro h1 h2 h3
  sorry

end manicure_cost_before_tip_l1192_119206


namespace original_intensity_45_percent_l1192_119237

variable (I : ℝ) -- Intensity of the original red paint in percentage.

-- Conditions
variable (h1 : 25 * 0.25 + 0.75 * I = 40) -- Given conditions about the intensities and the new solution.
variable (h2 : ∀ I : ℝ, 0.75 * I + 25 * 0.25 = 40) -- Rewriting the given condition to look specifically for I.

theorem original_intensity_45_percent (I : ℝ) (h1 : 25 * 0.25 + 0.75 * I = 40) : I = 45 := by
  -- We only need the statement. Proof is not required.
  sorry

end original_intensity_45_percent_l1192_119237


namespace major_axis_length_l1192_119282

theorem major_axis_length (r : ℝ) (minor_axis major_axis : ℝ) 
  (h1 : r = 2) 
  (h2 : minor_axis = 2 * r) 
  (h3 : major_axis = minor_axis + 0.8 * minor_axis) :
  major_axis = 7.2 :=
sorry

end major_axis_length_l1192_119282


namespace smallest_perfect_cube_divisor_l1192_119280

theorem smallest_perfect_cube_divisor (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (h : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  ∃ (a b c : ℕ), a = 6 ∧ b = 6 ∧ c = 6 ∧ (p^a * q^b * r^c) = (p^2 * q^2 * r^2)^3 ∧ 
  (p^a * q^b * r^c) % (p^2 * q^3 * r^4) = 0 := 
by
  sorry

end smallest_perfect_cube_divisor_l1192_119280


namespace inequality_ge_five_halves_l1192_119271

open Real

noncomputable def xy_yz_zx_eq_one (x y z : ℝ) := x * y + y * z + z * x = 1
noncomputable def non_neg (x y z : ℝ) := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

theorem inequality_ge_five_halves (x y z : ℝ) (h1 : xy_yz_zx_eq_one x y z) (h2 : non_neg x y z) :
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 5 / 2 := 
sorry

end inequality_ge_five_halves_l1192_119271


namespace length_of_platform_l1192_119216

noncomputable def train_length : ℝ := 300
noncomputable def time_to_cross_platform : ℝ := 39
noncomputable def time_to_cross_pole : ℝ := 9

theorem length_of_platform : ∃ P : ℝ, P = 1000 :=
by
  let train_speed := train_length / time_to_cross_pole
  let total_distance_cross_platform := train_length + 1000
  let platform_length := total_distance_cross_platform - train_length
  existsi platform_length
  sorry

end length_of_platform_l1192_119216


namespace conservation_center_total_turtles_l1192_119239

-- Define the green turtles and the relationship between green and hawksbill turtles.
def green_turtles : ℕ := 800
def hawksbill_turtles : ℕ := 2 * green_turtles

-- Statement we need to prove, which is the total number of turtles equals 3200.
theorem conservation_center_total_turtles : green_turtles + hawksbill_turtles = 3200 := by
  sorry

end conservation_center_total_turtles_l1192_119239


namespace find_integer_pairs_l1192_119246

theorem find_integer_pairs :
  ∃ (x y : ℤ),
    (x, y) = (-7, -99) ∨ (x, y) = (-1, -9) ∨ (x, y) = (1, 5) ∨ (x, y) = (7, -97) ∧
    2 * x^3 + x * y - 7 = 0 :=
by
  sorry

end find_integer_pairs_l1192_119246


namespace solve_for_s_l1192_119209

noncomputable def compute_s : Set ℝ :=
  { s | ∀ (x : ℝ), (x ≠ -1) → ((s * x - 3) / (x + 1) = x ↔ x^2 + (1 - s) * x + 3 = 0) ∧
    ((1 - s) ^ 2 - 4 * 3 = 0) }

theorem solve_for_s (h : ∀ s ∈ compute_s, s = 1 + 2 * Real.sqrt 3 ∨ s = 1 - 2 * Real.sqrt 3) :
  compute_s = {1 + 2 * Real.sqrt 3, 1 - 2 * Real.sqrt 3} :=
by
  sorry

end solve_for_s_l1192_119209


namespace sum_arithmetic_sequence_l1192_119212

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n + 1) / 2 * (2 * a 0 + n * (a 1 - a 0))

theorem sum_arithmetic_sequence (h_arith : arithmetic_sequence a) (h_condition : a 3 + a 4 + a 5 + a 6 = 18) :
  S a 9 = 45 :=
sorry

end sum_arithmetic_sequence_l1192_119212


namespace new_person_weight_is_75_l1192_119245

noncomputable def new_person_weight (previous_person_weight: ℝ) (average_increase: ℝ) (total_people: ℕ): ℝ :=
  previous_person_weight + total_people * average_increase

theorem new_person_weight_is_75 :
  new_person_weight 55 2.5 8 = 75 := 
by
  sorry

end new_person_weight_is_75_l1192_119245


namespace custom_op_seven_three_l1192_119262

def custom_op (a b : ℕ) : ℕ := 4 * a + 5 * b - a * b + 1

theorem custom_op_seven_three : custom_op 7 3 = 23 := by
  -- proof steps would go here
  sorry

end custom_op_seven_three_l1192_119262


namespace quadratic_completing_square_t_l1192_119225

theorem quadratic_completing_square_t : 
  ∀ (x k t : ℝ), (4 * x^2 + 16 * x - 400 = 0) →
  ((x + k)^2 = t) →
  t = 104 :=
by
  intros x k t h1 h2
  sorry

end quadratic_completing_square_t_l1192_119225


namespace kim_easy_round_correct_answers_l1192_119270

variable (E : ℕ)

theorem kim_easy_round_correct_answers 
    (h1 : 2 * E + 3 * 2 + 5 * 4 = 38) : 
    E = 6 := 
sorry

end kim_easy_round_correct_answers_l1192_119270


namespace people_who_like_both_l1192_119213

-- Conditions
variables (total : ℕ) (a : ℕ) (b : ℕ) (none : ℕ)
-- Express the problem
theorem people_who_like_both : total = 50 → a = 23 → b = 20 → none = 14 → (a + b - (total - none) = 7) :=
by
  intros
  sorry

end people_who_like_both_l1192_119213


namespace problem_i_l1192_119266

theorem problem_i (n : ℕ) (h : n ≥ 1) : n ∣ 2^n - 1 ↔ n = 1 := by
  sorry

end problem_i_l1192_119266


namespace find_x_for_divisibility_18_l1192_119202

theorem find_x_for_divisibility_18 (x : ℕ) (h_digits : x < 10) :
  (1001 * x + 150) % 18 = 0 ↔ x = 6 :=
by
  sorry

end find_x_for_divisibility_18_l1192_119202


namespace height_of_room_is_twelve_l1192_119278

-- Defining the dimensions of the room
def length : ℝ := 25
def width : ℝ := 15

-- Defining the dimensions of the door and windows
def door_area : ℝ := 6 * 3
def window_area : ℝ := 3 * (4 * 3)

-- Total cost of whitewashing
def total_cost : ℝ := 5436

-- Cost per square foot for whitewashing
def cost_per_sqft : ℝ := 6

-- The equation to solve for height
def height_equation (h : ℝ) : Prop :=
  cost_per_sqft * (2 * (length + width) * h - (door_area + window_area)) = total_cost

theorem height_of_room_is_twelve : ∃ h : ℝ, height_equation h ∧ h = 12 := by
  -- Proof would go here
  sorry

end height_of_room_is_twelve_l1192_119278


namespace number_of_possible_values_for_a_l1192_119226

theorem number_of_possible_values_for_a :
  ∀ (a b c d : ℕ), 
  a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 3010 ∧ a^2 - b^2 + c^2 - d^2 = 3010 →
  ∃ n, n = 751 :=
by {
  sorry
}

end number_of_possible_values_for_a_l1192_119226


namespace c_alone_finishes_in_6_days_l1192_119260

theorem c_alone_finishes_in_6_days (a b c : ℝ) (W : ℝ) :
  (1 / 36) * W + (1 / 18) * W + (1 / c) * W = (1 / 4) * W → c = 6 :=
by
  intros h
  simp at h
  sorry

end c_alone_finishes_in_6_days_l1192_119260


namespace ahmed_total_distance_l1192_119228

theorem ahmed_total_distance (d : ℝ) (h : (3 / 4) * d = 12) : d = 16 := 
by 
  sorry

end ahmed_total_distance_l1192_119228


namespace basketball_team_points_l1192_119208

variable (a b x : ℕ)

theorem basketball_team_points (h1 : 2 * a = 3 * b) 
                             (h2 : x = a + 1)
                             (h3 : 2 * a + 3 * b + x = 61) : 
    x = 13 :=
by {
  sorry
}

end basketball_team_points_l1192_119208


namespace work_completion_days_l1192_119251

theorem work_completion_days (A B C : ℕ) (work_rate_A : A = 4) (work_rate_B : B = 10) (work_rate_C : C = 20 / 3) :
  (1 / A) + (1 / B) + (3 / C) = 1 / 2 :=
by
  sorry

end work_completion_days_l1192_119251


namespace ratio_of_radii_l1192_119224

theorem ratio_of_radii (a b c : ℝ) (h1 : π * c^2 - π * a^2 = 4 * π * a^2) (h2 : π * b^2 = (π * a^2 + π * c^2) / 2) :
  a / c = 1 / Real.sqrt 5 := by
  sorry

end ratio_of_radii_l1192_119224


namespace multiplication_verification_l1192_119210

theorem multiplication_verification (x : ℕ) (h : 23 - x = 4) : 23 * x = 437 := by
  sorry

end multiplication_verification_l1192_119210


namespace find_b_of_triangle_ABC_l1192_119286

theorem find_b_of_triangle_ABC (a b c : ℝ) (cos_A : ℝ) 
  (h1 : a = 2) 
  (h2 : c = 2 * Real.sqrt 3) 
  (h3 : cos_A = Real.sqrt 3 / 2) 
  (h4 : b < c) : 
  b = 2 := 
by
  sorry

end find_b_of_triangle_ABC_l1192_119286


namespace min_buses_needed_l1192_119205

theorem min_buses_needed (x y : ℕ) (h1 : 45 * x + 35 * y ≥ 530) (h2 : y ≥ 3) : x + y = 13 :=
by
  sorry

end min_buses_needed_l1192_119205


namespace amount_of_money_around_circumference_l1192_119233

-- Define the given conditions
def horizontal_coins : ℕ := 6
def vertical_coins : ℕ := 4
def coin_value_won : ℕ := 100

-- The goal is to prove the total amount of money around the circumference
theorem amount_of_money_around_circumference : 
  (2 * (horizontal_coins - 2) + 2 * (vertical_coins - 2) + 4) * coin_value_won = 1600 :=
by
  sorry

end amount_of_money_around_circumference_l1192_119233


namespace range_of_a_l1192_119242

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - (a - 1)*x + (a - 1) > 0) ↔ (1 < a ∧ a < 5) := by
  sorry

end range_of_a_l1192_119242


namespace simplify_div_expr_l1192_119254

theorem simplify_div_expr (x : ℝ) (h : x = Real.sqrt 3) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x - 1) / (x^2 + 2 * x + 1)) = 1 + Real.sqrt 3 / 3 := by
sorry

end simplify_div_expr_l1192_119254


namespace correct_operation_l1192_119253

theorem correct_operation (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by sorry

end correct_operation_l1192_119253


namespace train_crossing_time_l1192_119241

/--
A train requires 8 seconds to pass a pole while it requires some seconds to cross a stationary train which is 400 meters long. 
The speed of the train is 144 km/h. Prove that it takes 18 seconds for the train to cross the stationary train.
-/
theorem train_crossing_time
  (train_speed_kmh : ℕ)
  (time_to_pass_pole : ℕ)
  (length_stationary_train : ℕ)
  (speed_mps : ℕ)
  (length_moving_train : ℕ)
  (total_length : ℕ)
  (crossing_time : ℕ) :
  train_speed_kmh = 144 →
  time_to_pass_pole = 8 →
  length_stationary_train = 400 →
  speed_mps = (train_speed_kmh * 1000) / 3600 →
  length_moving_train = speed_mps * time_to_pass_pole →
  total_length = length_moving_train + length_stationary_train →
  crossing_time = total_length / speed_mps →
  crossing_time = 18 :=
by
  intros;
  sorry

end train_crossing_time_l1192_119241


namespace yellow_block_weight_proof_l1192_119247

-- Define the weights and the relationship between them
def green_block_weight : ℝ := 0.4
def additional_weight : ℝ := 0.2
def yellow_block_weight : ℝ := green_block_weight + additional_weight

-- The theorem to prove
theorem yellow_block_weight_proof : yellow_block_weight = 0.6 :=
by
  -- Proof will be supplied here
  sorry

end yellow_block_weight_proof_l1192_119247


namespace condition_iff_inequality_l1192_119244

theorem condition_iff_inequality (a b : ℝ) (h : a * b ≠ 0) : (0 < a ∧ 0 < b) ↔ ((a + b) / 2 ≥ Real.sqrt (a * b)) :=
by
  -- Proof goes here
  sorry 

end condition_iff_inequality_l1192_119244


namespace f_at_2_is_neg_1_l1192_119272

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 - x + 2

-- Given condition: f(-2) = 5
axiom h : ∀ (a b : ℝ), f a b (-2) = 5

-- Prove that f(2) = -1 given the above conditions
theorem f_at_2_is_neg_1 (a b : ℝ) (h_ab : f a b (-2) = 5) : f a b 2 = -1 := by
  sorry

end f_at_2_is_neg_1_l1192_119272


namespace percentage_increase_in_gross_revenue_l1192_119284

theorem percentage_increase_in_gross_revenue 
  (P R : ℝ) 
  (hP : P > 0) 
  (hR : R > 0) 
  (new_price : ℝ := 0.80 * P) 
  (new_quantity : ℝ := 1.60 * R) : 
  (new_price * new_quantity - P * R) / (P * R) * 100 = 28 := 
by
  sorry

end percentage_increase_in_gross_revenue_l1192_119284


namespace scientific_notation_correct_l1192_119204

theorem scientific_notation_correct :
  ∃! (n : ℝ) (a : ℝ), 0.000000012 = a * 10 ^ n ∧ a = 1.2 ∧ n = -8 :=
by
  sorry

end scientific_notation_correct_l1192_119204


namespace carol_is_inviting_friends_l1192_119296

theorem carol_is_inviting_friends :
  ∀ (invitations_per_pack packs_needed friends_invited : ℕ), 
  invitations_per_pack = 2 → 
  packs_needed = 5 → 
  friends_invited = invitations_per_pack * packs_needed → 
  friends_invited = 10 :=
by
  intros invitations_per_pack packs_needed friends_invited h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end carol_is_inviting_friends_l1192_119296


namespace jenny_chocolate_milk_probability_l1192_119256

-- Define the binomial probability function.
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  ( Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

-- Given conditions: probability each day and total number of days.
def probability_each_day : ℚ := 2 / 3
def num_days : ℕ := 7
def successful_days : ℕ := 3

-- The problem statement to prove.
theorem jenny_chocolate_milk_probability :
  binomial_probability num_days successful_days probability_each_day = 280 / 2187 :=
by
  sorry

end jenny_chocolate_milk_probability_l1192_119256


namespace arithmetic_sequence_common_difference_l1192_119274

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 = 2)
  (h3 : ∃ r, a 2 = r * a 1 ∧ a 5 = r * a 2) :
  d = 4 :=
sorry

end arithmetic_sequence_common_difference_l1192_119274


namespace a_minus_b_is_30_l1192_119255

-- Definition of the sum of the arithmetic series
def sum_arithmetic_series (first last : ℕ) (n : ℕ) : ℕ :=
  (n * (first + last)) / 2

-- Definitions based on problem conditions
def a : ℕ := sum_arithmetic_series 2 60 30
def b : ℕ := sum_arithmetic_series 1 59 30

theorem a_minus_b_is_30 : a - b = 30 :=
  by sorry

end a_minus_b_is_30_l1192_119255


namespace solution_l1192_119265

-- Definitions for vectors a and b with given conditions for orthogonality and equal magnitudes
def a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

-- Orthogonality condition
def orthogonal (p q : ℝ) : Prop := 4 * 3 + p * 2 + (-2) * q = 0

-- Equal magnitude condition
def equal_magnitudes (p q : ℝ) : Prop :=
  4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2

-- Proof problem
theorem solution (p q : ℝ) (h_orthogonal : orthogonal p q) (h_equal_magnitudes : equal_magnitudes p q) :
  p = -29 / 12 ∧ q = 43 / 12 := 
by 
  sorry

end solution_l1192_119265


namespace avg_speed_l1192_119298

noncomputable def jane_total_distance : ℝ := 120
noncomputable def time_period_hours : ℝ := 7

theorem avg_speed :
  jane_total_distance / time_period_hours = (120 / 7 : ℝ):=
by
  sorry

end avg_speed_l1192_119298


namespace problem_I_problem_II_l1192_119238

namespace ProofProblems

def f (x a : ℝ) : ℝ := |x - a| + |x + 5|

theorem problem_I (x : ℝ) : (f x 1) ≥ 2 * |x + 5| ↔ x ≤ -2 := 
by sorry

theorem problem_II (a : ℝ) : 
  (∀ x : ℝ, (f x a) ≥ 8) ↔ (a ≥ 3 ∨ a ≤ -13) := 
by sorry

end ProofProblems

end problem_I_problem_II_l1192_119238


namespace arithmetic_sequence_general_term_sum_sequence_proof_l1192_119263

theorem arithmetic_sequence_general_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ) (a1 : ℝ)
  (h1 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h2 : d > 0)
  (h3 : a1 * (a1 + 3 * d) = 22)
  (h4 : 4 * a1 + 6 * d = 26) :
  ∀ n, a_n n = 3 * n - 1 := sorry

theorem sum_sequence_proof (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (h1 : ∀ n, a_n n = 3 * n - 1)
  (h2 : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1)))
  (h3 : ∀ n, T_n n = (Finset.range n).sum b_n)
  (n : ℕ) :
  T_n n < 1 / 6 := sorry

end arithmetic_sequence_general_term_sum_sequence_proof_l1192_119263


namespace mutually_exclusive_not_complementary_l1192_119218

-- Definitions of events
def EventA (n : ℕ) : Prop := n % 2 = 1
def EventB (n : ℕ) : Prop := n % 2 = 0
def EventC (n : ℕ) : Prop := n % 2 = 0
def EventD (n : ℕ) : Prop := n = 2 ∨ n = 4

-- Mutual exclusivity and complementarity
def mutually_exclusive {α : Type} (A B : α → Prop) : Prop :=
∀ x, ¬ (A x ∧ B x)

def complementary {α : Type} (A B : α → Prop) : Prop :=
∀ x, A x ∨ B x

-- The statement to be proved
theorem mutually_exclusive_not_complementary :
  mutually_exclusive EventA EventD ∧ ¬ complementary EventA EventD :=
by sorry

end mutually_exclusive_not_complementary_l1192_119218


namespace task2_X_alone_l1192_119276

namespace TaskWork

variables (r_X r_Y r_Z : ℝ)

-- Task 1 conditions
axiom task1_XY : r_X + r_Y = 1 / 4
axiom task1_YZ : r_Y + r_Z = 1 / 6
axiom task1_XZ : r_X + r_Z = 1 / 3

-- Task 2 condition
axiom task2_XYZ : r_X + r_Y + r_Z = 1 / 2

-- Theorem to be proven
theorem task2_X_alone : 1 / r_X = 4.8 :=
sorry

end TaskWork

end task2_X_alone_l1192_119276


namespace smallest_ratio_l1192_119281

-- Define the system of equations as conditions
def eq1 (x y : ℝ) := x^3 + 3 * y^3 = 11
def eq2 (x y : ℝ) := (x^2 * y) + (x * y^2) = 6

-- Define the goal: proving the smallest value of x/y for the solutions (x, y) is -1.31
theorem smallest_ratio (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) :
  ∃ t : ℝ, t = x / y ∧ ∀ t', t' = x / y → t' ≥ -1.31 :=
sorry

end smallest_ratio_l1192_119281


namespace stadium_length_in_yards_l1192_119299

def length_in_feet := 183
def feet_per_yard := 3

theorem stadium_length_in_yards : length_in_feet / feet_per_yard = 61 := by
  sorry

end stadium_length_in_yards_l1192_119299


namespace find_f_2n_l1192_119234

variable (f : ℤ → ℤ)
variable (n : ℕ)

axiom axiom1 {x y : ℤ} : f (x + y) = f x + f y + 2 * x * y + 1
axiom axiom2 : f (-2) = 1

theorem find_f_2n (n : ℕ) (h : n > 0) : f (2 * n) = 4 * n^2 + 2 * n - 1 := sorry

end find_f_2n_l1192_119234


namespace area_of_circle_l1192_119252

theorem area_of_circle 
  (r : ℝ → ℝ)
  (h : ∀ θ : ℝ, r θ = 3 * Real.cos θ - 4 * Real.sin θ) :
  ∃ A : ℝ, A = (25 / 4) * Real.pi :=
by
  sorry

end area_of_circle_l1192_119252


namespace find_a1_an_l1192_119232

noncomputable def arith_geo_seq (a : ℕ → ℝ) : Prop :=
  (∃ d ≠ 0, (a 2 + a 4 = 10) ∧ (a 2 ^ 2 = a 1 * a 5))

theorem find_a1_an (a : ℕ → ℝ)
  (h_arith_geo_seq : arith_geo_seq a) :
  a 1 = 1 ∧ (∀ n, a n = 2 * n - 1) :=
sorry

end find_a1_an_l1192_119232


namespace only_one_real_solution_l1192_119291

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

theorem only_one_real_solution (a : ℝ) (h : ∀ x : ℝ, abs (f x) = g a x → x = 1) : a < 0 := 
by
  sorry

end only_one_real_solution_l1192_119291


namespace intersection_point_l1192_119221

def L1 (x y : ℚ) : Prop := y = -3 * x
def L2 (x y : ℚ) : Prop := y + 4 = 9 * x

theorem intersection_point : ∃ x y : ℚ, L1 x y ∧ L2 x y ∧ x = 1/3 ∧ y = -1 := sorry

end intersection_point_l1192_119221


namespace cubic_poly_l1192_119261

noncomputable def q (x : ℝ) : ℝ := - (2 / 3) * x ^ 3 + 2 * x ^ 2 - (8 / 3) * x - (16 / 3)

theorem cubic_poly:
  ( ∃ (a b c d : ℝ), 
    (∀ x : ℝ, q x = a * x ^ 3 + b * x ^ 2 + c * x + d)
    ∧ q 1 = -6
    ∧ q 2 = -8
    ∧ q 3 = -14
    ∧ q 4 = -28
  ) → 
  q x = - (2 / 3) * x ^ 3 + 2 * x ^ 2 - (8 / 3) * x - (16 / 3) := 
sorry

end cubic_poly_l1192_119261


namespace cos_reflected_value_l1192_119215

theorem cos_reflected_value (x : ℝ) (h : Real.cos (π / 6 + x) = 1 / 3) :
  Real.cos (5 * π / 6 - x) = -1 / 3 := 
by {
  sorry
}

end cos_reflected_value_l1192_119215


namespace possible_values_of_m_l1192_119214

open Complex

theorem possible_values_of_m (p q r s m : ℂ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0)
  (h5 : p * m^3 + q * m^2 + r * m + s = 0)
  (h6 : q * m^3 + r * m^2 + s * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I :=
sorry

end possible_values_of_m_l1192_119214


namespace solve_quadratic_eq_l1192_119229

theorem solve_quadratic_eq (a : ℝ) (x : ℝ) 
  (h : a ∈ ({-1, 1, a^2} : Set ℝ)) : 
  (x^2 - (1 - a) * x - 2 = 0) → (x = 2 ∨ x = -1) := by
  sorry

end solve_quadratic_eq_l1192_119229


namespace cut_piece_ratio_l1192_119258

noncomputable def original_log_length : ℕ := 20
noncomputable def weight_per_foot : ℕ := 150
noncomputable def cut_piece_weight : ℕ := 1500

theorem cut_piece_ratio :
  (cut_piece_weight / weight_per_foot / original_log_length) = (1 / 2) := by
  sorry

end cut_piece_ratio_l1192_119258


namespace circles_through_two_points_in_4x4_grid_l1192_119287

noncomputable def number_of_circles (n : ℕ) : ℕ :=
  if n = 4 then
    52
  else
    sorry

theorem circles_through_two_points_in_4x4_grid :
  number_of_circles 4 = 52 :=
by
  exact rfl  -- Reflexivity of equality shows the predefined value of 52

end circles_through_two_points_in_4x4_grid_l1192_119287


namespace younger_person_age_l1192_119293

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

end younger_person_age_l1192_119293


namespace bricks_required_for_courtyard_l1192_119223

/-- 
A courtyard is 45 meters long and 25 meters broad needs to be paved with bricks of 
dimensions 15 cm by 7 cm. What will be the total number of bricks required?
-/
theorem bricks_required_for_courtyard 
  (courtyard_length : ℕ) (courtyard_width : ℕ)
  (brick_length : ℕ) (brick_width : ℕ)
  (H1 : courtyard_length = 4500) (H2 : courtyard_width = 2500)
  (H3 : brick_length = 15) (H4 : brick_width = 7) :
  let courtyard_area_cm : ℕ := courtyard_length * courtyard_width
  let brick_area_cm : ℕ := brick_length * brick_width
  let total_bricks : ℕ := (courtyard_area_cm + brick_area_cm - 1) / brick_area_cm
  total_bricks = 107143 := by
  sorry

end bricks_required_for_courtyard_l1192_119223


namespace tangent_parallel_l1192_119207

-- Define the curve function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the slope of the line 4x - y - 1 = 0, which is 4
def line_slope : ℝ := 4

-- The main theorem statement
theorem tangent_parallel (a b : ℝ) (h1 : f a = b) (h2 : f' a = line_slope) :
  (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = -4) :=
sorry

end tangent_parallel_l1192_119207


namespace geometric_sequence_r_value_l1192_119273

theorem geometric_sequence_r_value (S : ℕ → ℚ) (r : ℚ) (n : ℕ) (h : n ≥ 2) (h1 : ∀ n, S n = 3^n + r) :
    r = -1 :=
sorry

end geometric_sequence_r_value_l1192_119273


namespace infinite_geometric_series_sum_l1192_119200

theorem infinite_geometric_series_sum : 
  ∑' n : ℕ, (1 / 3) ^ n = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l1192_119200


namespace find_a_l1192_119269

theorem find_a (a : ℝ) (h : 0.005 * a = 65) : a = 13000 / 100 :=
by
  sorry

end find_a_l1192_119269


namespace maximum_value_is_l1192_119250

noncomputable def maximum_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) : ℝ :=
  x^2 + 2 * x * y + 3 * y^2

theorem maximum_value_is (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) :
  maximum_value x y h₁ h₂ h₃ ≤ 18 + 12 * Real.sqrt 3 :=
sorry

end maximum_value_is_l1192_119250


namespace molecular_weight_one_mole_l1192_119285

theorem molecular_weight_one_mole (mw_three_moles : ℕ) (h : mw_three_moles = 882) : mw_three_moles / 3 = 294 :=
by
  -- proof is omitted
  sorry

end molecular_weight_one_mole_l1192_119285


namespace find_added_value_l1192_119227

theorem find_added_value (N : ℕ) (V : ℕ) (H : N = 1280) :
  ((N + V) / 125 = 7392 / 462) → V = 720 :=
by 
  sorry

end find_added_value_l1192_119227


namespace pyramid_volume_l1192_119230

noncomputable def volume_of_pyramid (S α β : ℝ) : ℝ :=
  (1 / 6) * S * (Real.sqrt (2 * S * (Real.tan α) * (Real.tan β)))

theorem pyramid_volume 
  (S α β : ℝ)
  (base_area : S > 0)
  (equal_lateral_edges : true)
  (dihedral_angles : α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2) :
  volume_of_pyramid S α β = (1 / 6) * S * (Real.sqrt (2 * S * (Real.tan α) * (Real.tan β))) :=
by
  sorry

end pyramid_volume_l1192_119230


namespace three_digit_with_five_is_divisible_by_five_l1192_119295

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

def divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem three_digit_with_five_is_divisible_by_five (M : ℕ) :
  is_three_digit M ∧ ends_in_five M → divisible_by_five M :=
by
  sorry

end three_digit_with_five_is_divisible_by_five_l1192_119295


namespace sum_of_powers_pattern_l1192_119264

theorem sum_of_powers_pattern :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 5^32 + 7^32 :=
  sorry

end sum_of_powers_pattern_l1192_119264


namespace cars_in_group_l1192_119203

open Nat

theorem cars_in_group (C : ℕ) : 
  (47 ≤ C) →                  -- At least 47 cars in the group
  (53 ≤ C) →                  -- At least 53 cars in the group
  C ≥ 100 :=                  -- Conclusion: total cars is at least 100
by
  -- Begin the proof
  sorry                       -- Skip proof for now

end cars_in_group_l1192_119203


namespace find_n_l1192_119275

theorem find_n (n : ℕ) (S : ℕ) (h1 : S = n * (n + 1) / 2)
  (h2 : ∃ a : ℕ, a > 0 ∧ a < 10 ∧ S = 111 * a) : n = 36 :=
sorry

end find_n_l1192_119275


namespace perfect_squares_factors_360_l1192_119257

theorem perfect_squares_factors_360 : 
  let n := 360
  let prime_factors := (2, 3, 5)
  let exponents := (3, 2, 1)
  ∃ (count : ℕ), count = 4 :=
by
  let n := 360
  let prime_factors := (2, 3, 5)
  let exponents := (3, 2, 1)
  -- Calculation by hand has shown us that there are 4 perfect square factors
  exact ⟨4, rfl⟩

end perfect_squares_factors_360_l1192_119257


namespace Lizzy_total_after_loan_returns_l1192_119217

theorem Lizzy_total_after_loan_returns : 
  let initial_amount := 50
  let alice_loan := 25 
  let alice_interest_rate := 0.15
  let bob_loan := 20
  let bob_interest_rate := 0.20
  let alice_interest := alice_loan * alice_interest_rate
  let bob_interest := bob_loan * bob_interest_rate
  let total_alice := alice_loan + alice_interest
  let total_bob := bob_loan + bob_interest
  let total_amount := total_alice + total_bob
  total_amount = 52.75 :=
by
  sorry

end Lizzy_total_after_loan_returns_l1192_119217


namespace sugar_and_granulated_sugar_delivered_l1192_119240

theorem sugar_and_granulated_sugar_delivered (total_bags : ℕ) (percentage_more : ℚ) (mass_ratio : ℚ) (total_weight : ℚ)
    (h_total_bags : total_bags = 63)
    (h_percentage_more : percentage_more = 1.25)
    (h_mass_ratio : mass_ratio = 3 / 4)
    (h_total_weight : total_weight = 4.8) :
    ∃ (sugar_weight granulated_sugar_weight : ℚ),
        (granulated_sugar_weight = 1.8) ∧ (sugar_weight = 3) ∧
        ((sugar_weight + granulated_sugar_weight = total_weight) ∧
        (sugar_weight / 28 = (granulated_sugar_weight / 35) * mass_ratio)) :=
by
    sorry

end sugar_and_granulated_sugar_delivered_l1192_119240


namespace probability_even_sum_includes_ball_15_l1192_119249

-- Definition of the conditions in Lean
def balls : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

def odd_balls : Set ℕ := {n ∈ balls | n % 2 = 1}
def even_balls : Set ℕ := {n ∈ balls | n % 2 = 0}
def ball_15 : ℕ := 15

-- The number of ways to choose k elements from a set of n elements
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Number of ways to draw 7 balls ensuring the sum is even and ball 15 is included
def favorable_outcomes : ℕ :=
  choose 6 5 * choose 8 1 +   -- 5 other odd and 1 even
  choose 6 3 * choose 8 3 +   -- 3 other odd and 3 even
  choose 6 1 * choose 8 5     -- 1 other odd and 5 even

-- Total number of ways to choose 7 balls including ball 15:
def total_outcomes : ℕ := choose 14 6

-- Probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- The proof we require
theorem probability_even_sum_includes_ball_15 :
  probability = 1504 / 3003 :=
by
  -- proof omitted for brevity
  sorry

end probability_even_sum_includes_ball_15_l1192_119249


namespace cookies_taken_in_four_days_l1192_119277

-- Define the initial conditions
def initial_cookies : ℕ := 70
def remaining_cookies : ℕ := 28
def days_in_week : ℕ := 7
def days_of_interest : ℕ := 4

-- Define the total cookies taken out in a week
def cookies_taken_week := initial_cookies - remaining_cookies

-- Define the cookies taken out each day
def cookies_taken_per_day := cookies_taken_week / days_in_week

-- Final statement to show the number of cookies taken out in four days
theorem cookies_taken_in_four_days : cookies_taken_per_day * days_of_interest = 24 := by
  sorry -- The proof steps will be here.

end cookies_taken_in_four_days_l1192_119277


namespace speed_in_still_water_l1192_119220

-- Define the conditions: upstream and downstream speeds.
def upstream_speed : ℝ := 10
def downstream_speed : ℝ := 20

-- Define the still water speed theorem.
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 15 := by
  sorry

end speed_in_still_water_l1192_119220


namespace positive_difference_between_two_numbers_l1192_119283

theorem positive_difference_between_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 60) 
  (h2 : y^2 - 4 * x^2 = 80) : 
  |y - x| = 179.33 := 
by sorry

end positive_difference_between_two_numbers_l1192_119283


namespace age_of_first_man_replaced_l1192_119267

theorem age_of_first_man_replaced (x : ℕ) (avg_before : ℝ) : avg_before * 15 + 30 = avg_before * 15 + 74 - (x + 23) → (37 * 2 - (x + 23) = 30) → x = 21 :=
sorry

end age_of_first_man_replaced_l1192_119267


namespace tooth_fairy_left_amount_l1192_119289

-- Define the values of the different types of coins
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50
def dime_value : ℝ := 0.10

-- Define the number of each type of coins Joan received
def num_quarters : ℕ := 14
def num_half_dollars : ℕ := 14
def num_dimes : ℕ := 14

-- Calculate the total values for each type of coin
def total_quarters_value : ℝ := num_quarters * quarter_value
def total_half_dollars_value : ℝ := num_half_dollars * half_dollar_value
def total_dimes_value : ℝ := num_dimes * dime_value

-- The total amount of money left by the tooth fairy
def total_amount_left := total_quarters_value + total_half_dollars_value + total_dimes_value

-- The theorem stating that the total amount is $11.90
theorem tooth_fairy_left_amount : total_amount_left = 11.90 := by 
  sorry

end tooth_fairy_left_amount_l1192_119289


namespace percentage_to_pass_is_correct_l1192_119201

-- Define the conditions
def marks_obtained : ℕ := 130
def marks_failed_by : ℕ := 14
def max_marks : ℕ := 400

-- Define the function to calculate the passing percentage
def passing_percentage (obtained : ℕ) (failed_by : ℕ) (max : ℕ) : ℚ :=
  ((obtained + failed_by : ℕ) / (max : ℚ)) * 100

-- Statement of the problem
theorem percentage_to_pass_is_correct :
  passing_percentage marks_obtained marks_failed_by max_marks = 36 := 
sorry

end percentage_to_pass_is_correct_l1192_119201
