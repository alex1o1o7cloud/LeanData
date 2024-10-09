import Mathlib

namespace max_pieces_with_3_cuts_l892_89207

theorem max_pieces_with_3_cuts (cake : Type) : 
  (∀ (cuts : ℕ), cuts = 3 → (∃ (max_pieces : ℕ), max_pieces = 8)) := by
  sorry

end max_pieces_with_3_cuts_l892_89207


namespace volume_of_cuboid_l892_89237

theorem volume_of_cuboid (l w h : ℝ) (hlw: l * w = 120) (hwh: w * h = 72) (hhl: h * l = 60) : l * w * h = 720 :=
  sorry

end volume_of_cuboid_l892_89237


namespace area_of_right_triangle_integers_l892_89234

theorem area_of_right_triangle_integers (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ (A : ℤ), A = (a * b) / 2 := 
sorry

end area_of_right_triangle_integers_l892_89234


namespace percentage_calculation_l892_89250

def part : ℝ := 12.356
def whole : ℝ := 12356
def expected_percentage : ℝ := 0.1

theorem percentage_calculation (p w : ℝ) (h_p : p = part) (h_w : w = whole) : 
  (p / w) * 100 = expected_percentage :=
sorry

end percentage_calculation_l892_89250


namespace min_value_expr_ge_52_l892_89232

open Real

theorem min_value_expr_ge_52 (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  (sin x + 3 * (1 / sin x)) ^ 2 + (cos x + 3 * (1 / cos x)) ^ 2 ≥ 52 := 
by
  sorry

end min_value_expr_ge_52_l892_89232


namespace simplify_eval_expr_l892_89241

noncomputable def a : ℝ := (Real.sqrt 2) + 1
noncomputable def b : ℝ := (Real.sqrt 2) - 1

theorem simplify_eval_expr (a b : ℝ) (ha : a = (Real.sqrt 2) + 1) (hb : b = (Real.sqrt 2) - 1) : 
  (a^2 - b^2) / a / (a + (2 * a * b + b^2) / a) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_eval_expr_l892_89241


namespace distinct_digit_S_problem_l892_89217

theorem distinct_digit_S_problem :
  ∃! (S : ℕ), S < 10 ∧ 
  ∃ (P Q R : ℕ), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ S ∧ R ≠ S ∧ 
  P < 10 ∧ Q < 10 ∧ R < 10 ∧
  ((P + Q = S) ∨ (P + Q = S + 10)) ∧
  (R = 0) :=
sorry

end distinct_digit_S_problem_l892_89217


namespace circle_area_eq_25pi_l892_89299

theorem circle_area_eq_25pi :
  (∃ (x y : ℝ), x^2 + y^2 - 4 * x + 6 * y - 12 = 0) →
  (∃ (area : ℝ), area = 25 * Real.pi) :=
by
  sorry

end circle_area_eq_25pi_l892_89299


namespace layla_goldfish_count_l892_89288

def goldfish_count (total_food : ℕ) (swordtails_count : ℕ) (swordtails_food : ℕ) (guppies_count : ℕ) (guppies_food : ℕ) (goldfish_food : ℕ) : ℕ :=
  total_food - (swordtails_count * swordtails_food + guppies_count * guppies_food) / goldfish_food

theorem layla_goldfish_count : goldfish_count 12 3 2 8 1 1 = 2 := by
  sorry

end layla_goldfish_count_l892_89288


namespace count_FourDigitNumsWithThousandsDigitFive_is_1000_l892_89244

def count_FourDigitNumsWithThousandsDigitFive : Nat :=
  let minNum := 5000
  let maxNum := 5999
  maxNum - minNum + 1

theorem count_FourDigitNumsWithThousandsDigitFive_is_1000 :
  count_FourDigitNumsWithThousandsDigitFive = 1000 :=
by
  sorry

end count_FourDigitNumsWithThousandsDigitFive_is_1000_l892_89244


namespace find_value_of_2a_plus_c_l892_89264

theorem find_value_of_2a_plus_c (a b c : ℝ) (h1 : 3 * a + b + 2 * c = 3) (h2 : a + 3 * b + 2 * c = 1) :
  2 * a + c = 2 :=
sorry

end find_value_of_2a_plus_c_l892_89264


namespace reduce_to_one_piece_l892_89216

-- Definitions representing the conditions:
def plane_divided_into_unit_triangles : Prop := sorry
def initial_configuration (n : ℕ) : Prop := sorry
def possible_moves : Prop := sorry

-- Main theorem statement:
theorem reduce_to_one_piece (n : ℕ) 
  (H1 : plane_divided_into_unit_triangles) 
  (H2 : initial_configuration n) 
  (H3 : possible_moves) : 
  ∃ k : ℕ, k * 3 = n :=
sorry

end reduce_to_one_piece_l892_89216


namespace part1_part2_l892_89218

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) - abs (x + 2)

theorem part1 {x : ℝ} : f x > 0 ↔ (x < -1 / 3 ∨ x > 3) := sorry

theorem part2 {m : ℝ} (h : ∃ x₀ : ℝ, f x₀ + 2 * m^2 < 4 * m) : -1 / 2 < m ∧ m < 5 / 2 := sorry

end part1_part2_l892_89218


namespace man_speed_l892_89223

theorem man_speed (train_length : ℝ) (time_to_cross : ℝ) (train_speed_kmph : ℝ) 
  (h1 : train_length = 100) (h2 : time_to_cross = 6) (h3 : train_speed_kmph = 54.99520038396929) : 
  ∃ man_speed : ℝ, man_speed = 16.66666666666667 - 15.27644455165814 :=
by sorry

end man_speed_l892_89223


namespace abigail_savings_l892_89259

-- Define the parameters for monthly savings and number of months in a year.
def monthlySavings : ℕ := 4000
def numberOfMonthsInYear : ℕ := 12

-- Define the total savings calculation.
def totalSavings (monthlySavings : ℕ) (numberOfMonths : ℕ) : ℕ :=
  monthlySavings * numberOfMonths

-- State the theorem that we need to prove.
theorem abigail_savings : totalSavings monthlySavings numberOfMonthsInYear = 48000 := by
  sorry

end abigail_savings_l892_89259


namespace plum_balances_pear_l892_89212

variable (A G S : ℕ)

-- Definitions as per the problem conditions
axiom condition1 : 3 * A + G = 10 * S
axiom condition2 : A + 6 * S = G

-- The goal is to prove the following statement
theorem plum_balances_pear : G = 7 * S :=
by
  -- Skipping the proof as only statement is needed
  sorry

end plum_balances_pear_l892_89212


namespace geometric_sequence_and_sum_l892_89276

theorem geometric_sequence_and_sum (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ)
  (h_a1 : a 1 = 3/2)
  (h_a_recur : ∀ n : ℕ, a (n + 1) = 3 * a n - 1)
  (h_b_def : ∀ n : ℕ, b n = a n - 1/2) :
  (∀ n : ℕ, b (n + 1) = 3 * b n ∧ b 1 = 1) ∧ 
  (∀ n : ℕ, S n = (3^n + n - 1) / 2) :=
sorry

end geometric_sequence_and_sum_l892_89276


namespace modular_arithmetic_proof_l892_89200

open Nat

theorem modular_arithmetic_proof (m : ℕ) (h0 : 0 ≤ m ∧ m < 37) (h1 : 4 * m ≡ 1 [MOD 37]) :
  (3^m)^4 ≡ 27 + 3 [MOD 37] :=
by
  -- Although some parts like modular inverse calculation or finding specific m are skipped,
  -- the conclusion directly should reflect (3^m)^4 ≡ 27 + 3 [MOD 37]
  -- Considering (3^m)^4 - 3 ≡ 24 [MOD 37] translates to the above statement
  sorry

end modular_arithmetic_proof_l892_89200


namespace line_intersection_l892_89248

-- Parameters for the first line
def line1_param (s : ℝ) : ℝ × ℝ := (1 - 2 * s, 4 + 3 * s)

-- Parameters for the second line
def line2_param (v : ℝ) : ℝ × ℝ := (-v, 5 + 6 * v)

-- Statement of the intersection point
theorem line_intersection :
  ∃ (s v : ℝ), line1_param s = (-1 / 9, 17 / 3) ∧ line2_param v = (-1 / 9, 17 / 3) :=
by
  -- Placeholder for the proof, which we are not providing as per instructions
  sorry

end line_intersection_l892_89248


namespace four_is_square_root_of_sixteen_l892_89225

theorem four_is_square_root_of_sixteen : (4 : ℝ) * (4 : ℝ) = 16 :=
by
  sorry

end four_is_square_root_of_sixteen_l892_89225


namespace compare_a_x_l892_89282

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem compare_a_x (x a b : ℝ) (h1 : a = log_base 5 (3^x + 4^x))
                    (h2 : b = log_base 4 (5^x - 3^x)) (h3 : a ≥ b) : x ≤ a :=
by
  sorry

end compare_a_x_l892_89282


namespace first_scenario_machines_l892_89235

theorem first_scenario_machines (M : ℕ) (h1 : 20 = 10 * 2 * M) (h2 : 140 = 20 * 17.5 * 2) : M = 5 :=
by sorry

end first_scenario_machines_l892_89235


namespace find_other_integer_l892_89274

theorem find_other_integer (x y : ℤ) (h1 : 3*x + 4*y = 103) (h2 : x = 19 ∨ y = 19) : x = 9 ∨ y = 9 :=
by sorry

end find_other_integer_l892_89274


namespace range_of_m_l892_89270

theorem range_of_m (m : ℝ) (x : ℝ) (h₁ : x^2 - 8*x - 20 ≤ 0) 
  (h₂ : (x - 1 - m) * (x - 1 + m) ≤ 0) (h₃ : 0 < m) : 
  m ≤ 3 := sorry

end range_of_m_l892_89270


namespace solve_for_x_l892_89255

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 ↔ x = 2 / 9 := by
  sorry

end solve_for_x_l892_89255


namespace monkey_distance_l892_89268

-- Define the initial speeds and percentage adjustments
def swing_speed : ℝ := 10
def run_speed : ℝ := 15
def wind_resistance_percentage : ℝ := 0.10
def branch_assistance_percentage : ℝ := 0.05

-- Conditions
def adjusted_swing_speed : ℝ := swing_speed * (1 - wind_resistance_percentage)
def adjusted_run_speed : ℝ := run_speed * (1 + branch_assistance_percentage)
def run_time : ℝ := 5
def swing_time : ℝ := 10

-- Define the distance formulas based on the conditions
def run_distance : ℝ := adjusted_run_speed * run_time
def swing_distance : ℝ := adjusted_swing_speed * swing_time

-- Total distance calculation
def total_distance : ℝ := run_distance + swing_distance

-- Statement for the proof
theorem monkey_distance : total_distance = 168.75 := by
  sorry

end monkey_distance_l892_89268


namespace max_donation_amount_l892_89275

theorem max_donation_amount (x : ℝ) : 
  (500 * x + 1500 * (x / 2) = 0.4 * 3750000) → x = 1200 :=
by 
  sorry

end max_donation_amount_l892_89275


namespace my_cousin_reading_time_l892_89293

-- Define the conditions
def reading_time_me_hours : ℕ := 3
def reading_speed_ratio : ℕ := 5
def reading_time_me_min : ℕ := reading_time_me_hours * 60

-- Define the statement to be proved
theorem my_cousin_reading_time : (reading_time_me_min / reading_speed_ratio) = 36 := by
  sorry

end my_cousin_reading_time_l892_89293


namespace distinct_real_solutions_l892_89273

open Real Nat

noncomputable def p_n : ℕ → ℝ → ℝ 
| 0, x => x
| (n+1), x => (p_n n (x^2 - 2))

theorem distinct_real_solutions (n : ℕ) : 
  ∃ S : Finset ℝ, S.card = 2^n ∧ ∀ x ∈ S, p_n n x = x ∧ (∀ y ∈ S, x ≠ y → x ≠ y) := 
sorry

end distinct_real_solutions_l892_89273


namespace dogwood_tree_cut_count_l892_89254

theorem dogwood_tree_cut_count
  (trees_part1 : ℝ) (trees_part2 : ℝ) (trees_left : ℝ)
  (h1 : trees_part1 = 5.0) (h2 : trees_part2 = 4.0)
  (h3 : trees_left = 2.0) :
  trees_part1 + trees_part2 - trees_left = 7.0 :=
by
  sorry

end dogwood_tree_cut_count_l892_89254


namespace num_times_teams_face_each_other_l892_89295

-- Conditions
variable (teams games total_games : ℕ)
variable (k : ℕ)
variable (h1 : teams = 17)
variable (h2 : games = teams * (teams - 1) * k / 2)
variable (h3 : total_games = 1360)

-- Proof problem
theorem num_times_teams_face_each_other : k = 5 := 
by 
  sorry

end num_times_teams_face_each_other_l892_89295


namespace train_crossing_time_l892_89238

-- Conditions
def length_train1 : ℕ := 200 -- Train 1 length in meters
def length_train2 : ℕ := 160 -- Train 2 length in meters
def speed_train1 : ℕ := 68 -- Train 1 speed in kmph
def speed_train2 : ℕ := 40 -- Train 2 speed in kmph

-- Conversion factors and formulas
def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600
def total_distance (l1 l2 : ℕ) := l1 + l2
def relative_speed (s1 s2 : ℕ) := kmph_to_mps (s1 + s2)
def crossing_time (dist speed : ℕ) := dist / speed

-- Proof statement
theorem train_crossing_time : 
  crossing_time (total_distance length_train1 length_train2) (relative_speed speed_train1 speed_train2) = 12 := by sorry

end train_crossing_time_l892_89238


namespace sequence_an_formula_l892_89209

theorem sequence_an_formula (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, a (n + 1) = a n^2 - n * a n + 1) :
  ∀ n : ℕ, a n = n + 1 :=
sorry

end sequence_an_formula_l892_89209


namespace mark_and_carolyn_total_l892_89292

theorem mark_and_carolyn_total (m c : ℝ) (hm : m = 3 / 4) (hc : c = 3 / 10) :
    m + c = 1.05 :=
by
  sorry

end mark_and_carolyn_total_l892_89292


namespace expected_value_of_winnings_after_one_flip_l892_89245

-- Definitions based on conditions from part a)
def prob_heads : ℚ := 1 / 3
def prob_tails : ℚ := 2 / 3
def win_heads : ℚ := 3
def lose_tails : ℚ := -2

-- The statement to prove:
theorem expected_value_of_winnings_after_one_flip :
  prob_heads * win_heads + prob_tails * lose_tails = -1 / 3 :=
by
  sorry

end expected_value_of_winnings_after_one_flip_l892_89245


namespace quadrilateral_is_parallelogram_l892_89266

theorem quadrilateral_is_parallelogram
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 - 2 * a * c - 2 * b * d = 0) :
  (a = c) ∧ (b = d) :=
by {
  sorry
}

end quadrilateral_is_parallelogram_l892_89266


namespace max_value_of_sum_l892_89228

open Real

theorem max_value_of_sum (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : a + b + c = 3) :
  (ab / (a + b) + bc / (b + c) + ca / (c + a)) ≤ 3 / 2 :=
sorry

end max_value_of_sum_l892_89228


namespace distance_CD_l892_89269

-- Conditions
variable (width_small : ℝ) 
variable (length_small : ℝ := 2 * width_small) 
variable (perimeter_small : ℝ := 2 * (width_small + length_small))
variable (width_large : ℝ := 3 * width_small)
variable (length_large : ℝ := 2 * length_small)
variable (area_large : ℝ := width_large * length_large)

-- Condition assertions
axiom smaller_rectangle_perimeter : perimeter_small = 6
axiom larger_rectangle_area : area_large = 12

-- Calculating distance hypothesis
theorem distance_CD (CD_x CD_y : ℝ) (width_small length_small width_large length_large : ℝ) 
  (smaller_rectangle_perimeter : 2 * (width_small + length_small) = 6)
  (larger_rectangle_area : (3 * width_small) * (2 * length_small) = 12)
  (CD_x_def : CD_x = 2 * length_small)
  (CD_y_def : CD_y = 2 * width_large - width_small)
  : Real.sqrt ((CD_x) ^ 2 + (CD_y) ^ 2) = Real.sqrt 45 := 
sorry

end distance_CD_l892_89269


namespace conversion_200_meters_to_kilometers_l892_89247

noncomputable def meters_to_kilometers (meters : ℕ) : ℝ :=
  meters / 1000

theorem conversion_200_meters_to_kilometers :
  meters_to_kilometers 200 = 0.2 :=
by
  sorry

end conversion_200_meters_to_kilometers_l892_89247


namespace geometric_series_sum_l892_89294

theorem geometric_series_sum : 
  let a : ℕ := 2
  let r : ℕ := 3
  let n : ℕ := 6
  let S_n := (a * (r^n - 1)) / (r - 1)
  S_n = 728 :=
by
  sorry

end geometric_series_sum_l892_89294


namespace probability_non_defective_pens_l892_89240

theorem probability_non_defective_pens :
  let total_pens := 12
  let defective_pens := 6
  let non_defective_pens := total_pens - defective_pens
  let probability_first_non_defective := non_defective_pens / total_pens
  let probability_second_non_defective := (non_defective_pens - 1) / (total_pens - 1)
  (probability_first_non_defective * probability_second_non_defective = 5 / 22) :=
by
  rfl

end probability_non_defective_pens_l892_89240


namespace smallest_y_in_arithmetic_series_l892_89222

theorem smallest_y_in_arithmetic_series (x y z : ℝ) (h1 : x < y) (h2 : y < z) (h3 : (x * y * z) = 216) : y = 6 :=
by 
  sorry

end smallest_y_in_arithmetic_series_l892_89222


namespace positive_root_in_range_l892_89279

theorem positive_root_in_range : ∃ x > 0, (x^2 - 2 * x - 1 = 0) ∧ (2 < x ∧ x < 3) :=
by
  sorry

end positive_root_in_range_l892_89279


namespace race_time_A_l892_89285

theorem race_time_A (v_A v_B : ℝ) (t_A t_B : ℝ) (hA_time_eq : v_A = 1000 / t_A)
  (hB_time_eq : v_B = 960 / t_B) (hA_beats_B_40m : 1000 / v_A = 960 / v_B)
  (hA_beats_B_8s : t_B = t_A + 8) : t_A = 200 := 
  sorry

end race_time_A_l892_89285


namespace elizabeth_husband_weight_l892_89226

-- Defining the variables for weights of the three wives
variable (s : ℝ) -- Weight of Simona
def elizabeta_weight : ℝ := s + 5
def georgetta_weight : ℝ := s + 10

-- Condition: The total weight of all wives
def total_wives_weight : ℝ := s + elizabeta_weight s + georgetta_weight s

-- Given: The total weight of all wives is 171 kg
def total_wives_weight_cond : Prop := total_wives_weight s = 171

-- Given:
-- Leon weighs the same as his wife.
-- Victor weighs one and a half times more than his wife.
-- Maurice weighs twice as much as his wife.

-- Given: Elizabeth's weight relationship
def elizabeth_weight_cond : Prop := (s + 5 * 1.5) = 85.5

-- Main proof problem:
theorem elizabeth_husband_weight (s : ℝ) (h1: total_wives_weight_cond s) : elizabeth_weight_cond s :=
by
  sorry

end elizabeth_husband_weight_l892_89226


namespace value_of_f_at_2_l892_89203

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem value_of_f_at_2 : f 2 = 62 :=
by
  -- The proof will be inserted here, it follows Horner's method steps shown in the solution
  sorry

end value_of_f_at_2_l892_89203


namespace red_light_probability_l892_89260

theorem red_light_probability (n : ℕ) (p_r : ℚ) (waiting_time_for_two_red : ℚ) 
    (prob_two_red : ℚ) :
    n = 4 →
    p_r = (1/3 : ℚ) →
    waiting_time_for_two_red = 4 →
    prob_two_red = (8/27 : ℚ) :=
by
  intros hn hp hw
  sorry

end red_light_probability_l892_89260


namespace increase_in_lines_l892_89231

variable (L : ℝ)
variable (h1 : L + (1 / 3) * L = 240)

theorem increase_in_lines : (240 - L) = 60 := by
  sorry

end increase_in_lines_l892_89231


namespace smallest_y_not_defined_l892_89263

theorem smallest_y_not_defined : 
  ∃ y : ℝ, (6 * y^2 - 37 * y + 6 = 0) ∧ (∀ z : ℝ, (6 * z^2 - 37 * z + 6 = 0) → y ≤ z) ∧ y = 1 / 6 :=
by
  sorry

end smallest_y_not_defined_l892_89263


namespace eliza_total_clothes_l892_89233

def time_per_blouse : ℕ := 15
def time_per_dress : ℕ := 20
def blouse_time : ℕ := 2 * 60   -- 2 hours in minutes
def dress_time : ℕ := 3 * 60    -- 3 hours in minutes

theorem eliza_total_clothes :
  (blouse_time / time_per_blouse) + (dress_time / time_per_dress) = 17 :=
by
  sorry

end eliza_total_clothes_l892_89233


namespace total_number_of_people_l892_89252

theorem total_number_of_people
  (cannoneers : ℕ) 
  (women : ℕ) 
  (men : ℕ) 
  (hc : cannoneers = 63)
  (hw : women = 2 * cannoneers)
  (hm : men = 2 * women) :
  cannoneers + women + men = 378 := 
sorry

end total_number_of_people_l892_89252


namespace trig_identity_proof_l892_89229

noncomputable def trig_identity (α β : ℝ) (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 2) : ℝ :=
  (Real.sin (2 * α)) / (Real.cos (2 * β))

theorem trig_identity_proof (α β : ℝ) (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 2) :
  trig_identity α β h1 h2 = 1 :=
sorry

end trig_identity_proof_l892_89229


namespace sum_of_products_lt_zero_l892_89257

theorem sum_of_products_lt_zero (a b c d e f : ℤ) (h : ∃ (i : ℕ), i ≤ 6 ∧ i ≠ 6 ∧ (∀ i ∈ [a, b, c, d, e, f], i < 0 → i ≤ i)) :
  ab + cdef < 0 :=
sorry

end sum_of_products_lt_zero_l892_89257


namespace catalyst_second_addition_is_882_l892_89271

-- Constants for the problem
def lower_bound : ℝ := 500
def upper_bound : ℝ := 1500
def golden_ratio_method : ℝ := 0.618

-- Calculated values
def first_addition : ℝ := lower_bound + golden_ratio_method * (upper_bound - lower_bound)
def second_bound : ℝ := first_addition - lower_bound
def second_addition : ℝ := lower_bound + golden_ratio_method * second_bound

theorem catalyst_second_addition_is_882 :
  lower_bound = 500 → upper_bound = 1500 → golden_ratio_method = 0.618 → second_addition = 882 := by
  -- Proof goes here
  sorry

end catalyst_second_addition_is_882_l892_89271


namespace cat_food_more_than_dog_food_l892_89224

-- Define the number of packages and cans per package for cat food
def cat_food_packages : ℕ := 9
def cat_food_cans_per_package : ℕ := 10

-- Define the number of packages and cans per package for dog food
def dog_food_packages : ℕ := 7
def dog_food_cans_per_package : ℕ := 5

-- Total number of cans of cat food
def total_cat_food_cans : ℕ := cat_food_packages * cat_food_cans_per_package

-- Total number of cans of dog food
def total_dog_food_cans : ℕ := dog_food_packages * dog_food_cans_per_package

-- Prove the difference between the total cans of cat food and total cans of dog food
theorem cat_food_more_than_dog_food : total_cat_food_cans - total_dog_food_cans = 55 := by
  -- Provide the calculation results directly
  have h_cat : total_cat_food_cans = 90 := by rfl
  have h_dog : total_dog_food_cans = 35 := by rfl
  calc
    total_cat_food_cans - total_dog_food_cans = 90 - 35 := by rw [h_cat, h_dog]
    _ = 55 := rfl

end cat_food_more_than_dog_food_l892_89224


namespace find_x_value_l892_89297

theorem find_x_value : ∃ x : ℝ, 35 - (23 - (15 - x)) = 12 * 2 / 1 / 2 → x = -21 :=
by
  sorry

end find_x_value_l892_89297


namespace mother_daughter_age_equality_l892_89208

theorem mother_daughter_age_equality :
  ∀ (x : ℕ), (24 * 12 + 3) + x = 12 * ((-5 : ℤ) + x) → x = 32 := 
by
  intros x h
  sorry

end mother_daughter_age_equality_l892_89208


namespace valid_pairs_l892_89277

def valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def valid_number (n : ℕ) : Prop :=
  let digits := [5, 3, 2, 9, n / 10 % 10, n % 10]
  (n % 2 = 0) ∧ (digits.sum % 3 = 0)

theorem valid_pairs (d₀ d₁ : ℕ) :
  valid_digit d₀ →
  valid_digit d₁ →
  (d₀ % 2 = 0) →
  valid_number (53290 * 10 + d₀ * 10 + d₁) →
  (d₀, d₁) = (0, 3) ∨ (d₀, d₁) = (2, 0) ∨ (d₀, d₁) = (2, 3) ∨ (d₀, d₁) = (2, 6) ∨
  (d₀, d₁) = (2, 9) ∨ (d₀, d₁) = (4, 1) ∨ (d₀, d₁) = (4, 4) ∨ (d₀, d₁) = (4, 7) ∨
  (d₀, d₁) = (6, 2) ∨ (d₀, d₁) = (6, 5) ∨ (d₀, d₁) = (6, 8) ∨ (d₀, d₁) = (8, 0) :=
by sorry

end valid_pairs_l892_89277


namespace find_b_l892_89284

theorem find_b
  (a b c : ℤ)
  (h1 : a + 5 = b)
  (h2 : 5 + b = c)
  (h3 : b + c = a) : b = -10 :=
by
  sorry

end find_b_l892_89284


namespace point_on_line_l892_89236

theorem point_on_line (A B C x₀ y₀ : ℝ) :
  (A * x₀ + B * y₀ + C = 0) ↔ (A * (x₀ - x₀) + B * (y₀ - y₀) = 0) :=
by 
  sorry

end point_on_line_l892_89236


namespace inequality_solution_l892_89201

def solutionSetInequality (x : ℝ) : Prop :=
  (x > 1 ∨ x < -2)

theorem inequality_solution (x : ℝ) : 
  (x+2)/(x-1) > 0 ↔ solutionSetInequality x := 
  sorry

end inequality_solution_l892_89201


namespace find_m_of_quadratic_root_zero_l892_89287

theorem find_m_of_quadratic_root_zero (m : ℝ) (h : ∃ x, (m * x^2 + 5 * x + m^2 - 2 * m = 0) ∧ x = 0) : m = 2 :=
sorry

end find_m_of_quadratic_root_zero_l892_89287


namespace volume_box_constraint_l892_89249

theorem volume_box_constraint : ∀ x : ℕ, ((2 * x + 6) * (x^3 - 8) * (x^2 + 4) < 1200) → x = 2 :=
by
  intros x h
  -- Proof is skipped
  sorry

end volume_box_constraint_l892_89249


namespace problem_1_problem_2_l892_89296

-- Definition of the operation ⊕
def my_oplus (a b : ℚ) : ℚ := (a + 3 * b) / 2

-- Prove that 4(2 ⊕ 5) = 34
theorem problem_1 : 4 * my_oplus 2 5 = 34 := 
by sorry

-- Definitions of A and B
def A (x y : ℚ) : ℚ := x^2 + 2 * x * y + y^2
def B (x y : ℚ) : ℚ := -2 * x * y + y^2

-- Prove that (A ⊕ B) + (B ⊕ A) = 2x^2 + 4y^2
theorem problem_2 (x y : ℚ) : 
  my_oplus (A x y) (B x y) + my_oplus (B x y) (A x y) = 2 * x^2 + 4 * y^2 := 
by sorry

end problem_1_problem_2_l892_89296


namespace even_n_equals_identical_numbers_l892_89278

theorem even_n_equals_identical_numbers (n : ℕ) (h1 : n ≥ 2) : 
  (∃ f : ℕ → ℕ, (∀ a b, f a = f b + f b) ∧ n % 2 = 0) :=
sorry


end even_n_equals_identical_numbers_l892_89278


namespace find_abs_x_l892_89289

-- Given conditions
def A (x : ℝ) : ℝ := 3 + x
def B (x : ℝ) : ℝ := 3 - x
def distance (a b : ℝ) : ℝ := abs (a - b)

-- Problem statement: Prove |x| = 4 given the conditions
theorem find_abs_x (x : ℝ) (h : distance (A x) (B x) = 8) : abs x = 4 := 
  sorry

end find_abs_x_l892_89289


namespace correct_conclusions_l892_89227

noncomputable def f1 (x : ℝ) : ℝ := 2^x - 1
noncomputable def f2 (x : ℝ) : ℝ := x^3
noncomputable def f3 (x : ℝ) : ℝ := x
noncomputable def f4 (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2

theorem correct_conclusions :
  ((∀ x, 0 < x ∧ x < 1 → f4 x > f1 x ∧ f4 x > f2 x ∧ f4 x > f3 x) ∧
  (∀ x, x > 1 → f4 x < f1 x ∧ f4 x < f2 x ∧ f4 x < f3 x)) ∧
  (∀ x, ¬(f3 x > f1 x ∧ f3 x > f2 x ∧ f3 x > f4 x) ∧
        ¬(f3 x < f1 x ∧ f3 x < f2 x ∧ f3 x < f4 x)) ∧
  (∃ x, x > 0 ∧ ∀ y, y > x → f1 y > f2 y ∧ f1 y > f3 y ∧ f1 y > f4 y) := by
  sorry

end correct_conclusions_l892_89227


namespace geometric_sequence_arithmetic_Sn_l892_89205

theorem geometric_sequence_arithmetic_Sn (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (a1 : ℝ) (n : ℕ) :
  (∀ n, a n = a1 * q ^ (n - 1)) →
  (∀ n, S n = a1 * (1 - q ^ n) / (1 - q)) →
  (∀ n, S (n + 1) - S n = S n - S (n - 1)) →
  q = 1 :=
by
  sorry

end geometric_sequence_arithmetic_Sn_l892_89205


namespace tina_days_to_use_pink_pens_tina_total_pens_l892_89210

-- Definitions based on the problem conditions.
def pink_pens : ℕ := 15
def green_pens : ℕ := pink_pens - 9
def blue_pens : ℕ := green_pens + 3
def total_pink_green := pink_pens + green_pens
def yellow_pens : ℕ := total_pink_green - 5
def pink_pens_per_day := 4

-- Prove the two statements based on the definitions.
theorem tina_days_to_use_pink_pens 
  (h1 : pink_pens = 15)
  (h2 : pink_pens_per_day = 4) :
  4 = 4 :=
by sorry

theorem tina_total_pens 
  (h1 : pink_pens = 15)
  (h2 : green_pens = pink_pens - 9)
  (h3 : blue_pens = green_pens + 3)
  (h4 : yellow_pens = total_pink_green - 5) :
  pink_pens + green_pens + blue_pens + yellow_pens = 46 :=
by sorry

end tina_days_to_use_pink_pens_tina_total_pens_l892_89210


namespace total_miles_run_correct_l892_89283

-- Define the number of people on the sprint team and the miles each person runs.
def number_of_people : Float := 150.0
def miles_per_person : Float := 5.0

-- Define the total miles run by the sprint team.
def total_miles_run : Float := number_of_people * miles_per_person

-- State the theorem to prove that the total miles run is equal to 750.0 miles.
theorem total_miles_run_correct : total_miles_run = 750.0 := sorry

end total_miles_run_correct_l892_89283


namespace complete_the_square_result_l892_89213

-- Define the equation
def initial_eq (x : ℝ) : Prop := x^2 + 4 * x + 3 = 0

-- State the theorem based on the condition and required to prove the question equals the answer
theorem complete_the_square_result (x : ℝ) : initial_eq x → (x + 2) ^ 2 = 1 := 
by
  intro h
  -- Proof is to be skipped
  sorry

end complete_the_square_result_l892_89213


namespace negate_proposition_l892_89272

-- Definitions of sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

-- The original proposition p
def p : Prop := ∀ x, is_odd x → is_even (2 * x)

-- The negation of the proposition p
def neg_p : Prop := ∃ x, is_odd x ∧ ¬ is_even (2 * x)

-- Proof problem statement: Prove that the negation of proposition p is as defined in neg_p
theorem negate_proposition :
  (∀ x, is_odd x → is_even (2 * x)) ↔ (∃ x, is_odd x ∧ ¬ is_even (2 * x)) :=
sorry

end negate_proposition_l892_89272


namespace alexa_pages_left_l892_89267

theorem alexa_pages_left 
  (total_pages : ℕ) 
  (first_day_read : ℕ) 
  (next_day_read : ℕ) 
  (total_pages_val : total_pages = 95) 
  (first_day_read_val : first_day_read = 18) 
  (next_day_read_val : next_day_read = 58) : 
  total_pages - (first_day_read + next_day_read) = 19 := by
  sorry

end alexa_pages_left_l892_89267


namespace molecular_weight_N2O3_correct_l892_89286

/-- Conditions -/
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

/-- Proof statement -/
theorem molecular_weight_N2O3_correct :
  (2 * atomic_weight_N + 3 * atomic_weight_O) = 76.02 ∧
  name_of_N2O3 = "dinitrogen trioxide" := sorry

/-- Definition of the compound name based on formula -/
def name_of_N2O3 : String := "dinitrogen trioxide"

end molecular_weight_N2O3_correct_l892_89286


namespace older_brother_has_17_stamps_l892_89206

def stamps_problem (y : ℕ) : Prop := y + (2 * y + 1) = 25

theorem older_brother_has_17_stamps (y : ℕ) (h : stamps_problem y) : 2 * y + 1 = 17 :=
by
  sorry

end older_brother_has_17_stamps_l892_89206


namespace find_b_find_area_l892_89265

open Real

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := A + π / 2
noncomputable def a : ℝ := 3
noncomputable def cos_A : ℝ := sqrt 6 / 3
noncomputable def b : ℝ := 3 * sqrt 2
noncomputable def area : ℝ := 3 * sqrt 2 / 2

theorem find_b (A : ℝ) (H1 : a = 3) (H2 : cos A = sqrt 6 / 3) (H3 : B = A + π / 2) : 
  b = 3 * sqrt 2 := 
  sorry

theorem find_area (A : ℝ) (H1 : a = 3) (H2 : cos A = sqrt 6 / 3) (H3 : B = A + π / 2) : 
  area = 3 * sqrt 2 / 2 := 
  sorry

end find_b_find_area_l892_89265


namespace percentage_increase_l892_89262

theorem percentage_increase (x : ℝ) (h1 : x = 99.9) : 
  ((x - 90) / 90) * 100 = 11 :=
by 
  -- Add the required proof steps here
  sorry

end percentage_increase_l892_89262


namespace divisibility_theorem_l892_89258

theorem divisibility_theorem {a m x n : ℕ} : (m ∣ n) ↔ (x^m - a^m ∣ x^n - a^n) :=
by
  sorry

end divisibility_theorem_l892_89258


namespace five_digit_number_divisibility_l892_89261

theorem five_digit_number_divisibility (a : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) : 11 ∣ 100001 * a :=
by
  sorry

end five_digit_number_divisibility_l892_89261


namespace dog_bones_remaining_l892_89239

noncomputable def initial_bones : ℕ := 350
noncomputable def factor : ℕ := 9
noncomputable def found_bones : ℕ := factor * initial_bones
noncomputable def total_bones : ℕ := initial_bones + found_bones
noncomputable def bones_given_away : ℕ := 120
noncomputable def bones_remaining : ℕ := total_bones - bones_given_away

theorem dog_bones_remaining : bones_remaining = 3380 :=
by
  sorry

end dog_bones_remaining_l892_89239


namespace years_to_earn_house_l892_89251

-- Defining the variables
variables (E S H : ℝ)

-- Defining the assumptions
def annual_expenses_savings_relation (E S : ℝ) : Prop :=
  8 * E = 12 * S

def annual_income_relation (H E S : ℝ) : Prop :=
  H / 24 = E + S

-- Theorem stating that it takes 60 years to earn the amount needed to buy the house
theorem years_to_earn_house (E S H : ℝ) 
  (h1 : annual_expenses_savings_relation E S) 
  (h2 : annual_income_relation H E S) : 
  H / S = 60 :=
by
  sorry

end years_to_earn_house_l892_89251


namespace perfect_square_of_division_l892_89221

theorem perfect_square_of_division (a b : ℤ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a * b + 1) ∣ (a^2 + b^2)) : ∃ k : ℤ, 0 < k ∧ k^2 = (a^2 + b^2) / (a * b + 1) :=
by
  sorry

end perfect_square_of_division_l892_89221


namespace angle_of_inclination_range_l892_89256

noncomputable def curve (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

noncomputable def tangent_slope (x : ℝ) : ℝ := 
  -4 * Real.exp x / (Real.exp x + 1) ^ 2

theorem angle_of_inclination_range (x : ℝ) (a : ℝ) 
  (hx : tangent_slope x = Real.tan a) : 
  (3 * Real.pi / 4 ≤ a ∧ a < Real.pi) :=
by 
  sorry

end angle_of_inclination_range_l892_89256


namespace problem_statement_l892_89243

theorem problem_statement (x : ℝ) (h : (2024 - x)^2 + (2022 - x)^2 = 4038) : 
  (2024 - x) * (2022 - x) = 2017 :=
sorry

end problem_statement_l892_89243


namespace distance_to_store_l892_89215

noncomputable def D : ℝ := 4

theorem distance_to_store :
  (1/3) * (D/2 + D/10 + D/10) = 56/60 :=
by
  sorry

end distance_to_store_l892_89215


namespace roots_of_quadratic_l892_89220

theorem roots_of_quadratic (a b : ℝ) (h : ab ≠ 0) : 
  (a + b = -2 * b) ∧ (a * b = a) → (a = -3 ∧ b = 1) :=
by
  sorry

end roots_of_quadratic_l892_89220


namespace stocks_closed_higher_l892_89298

-- Definition of the conditions:
def stocks : Nat := 1980
def increased (H L : Nat) : Prop := H = (1.20 : ℝ) * L
def total_stocks (H L : Nat) : Prop := H + L = stocks

-- Claim to prove
theorem stocks_closed_higher (H L : Nat) (h1 : increased H L) (h2 : total_stocks H L) : H = 1080 :=
by
  sorry

end stocks_closed_higher_l892_89298


namespace division_4073_by_38_l892_89290

theorem division_4073_by_38 :
  ∃ q r, 4073 = 38 * q + r ∧ 0 ≤ r ∧ r < 38 ∧ q = 107 ∧ r = 7 := by
  sorry

end division_4073_by_38_l892_89290


namespace david_english_marks_l892_89219

def david_marks (math physics chemistry biology avg : ℕ) : ℕ :=
  avg * 5 - (math + physics + chemistry + biology)

theorem david_english_marks :
  let math := 95
  let physics := 82
  let chemistry := 97
  let biology := 95
  let avg := 93
  david_marks math physics chemistry biology avg = 96 :=
by
  -- Proof is skipped
  sorry

end david_english_marks_l892_89219


namespace total_cases_l892_89202

-- Define the number of boys' high schools and girls' high schools
def boys_high_schools : Nat := 4
def girls_high_schools : Nat := 3

-- Theorem to be proven
theorem total_cases (B G : Nat) (hB : B = boys_high_schools) (hG : G = girls_high_schools) : 
  B + G = 7 :=
by
  rw [hB, hG]
  exact rfl

end total_cases_l892_89202


namespace simplify_fraction_144_12672_l892_89281

theorem simplify_fraction_144_12672 : (144 / 12672 : ℚ) = 1 / 88 :=
by
  sorry

end simplify_fraction_144_12672_l892_89281


namespace find_circle_center_l892_89214

-- The statement to prove that the center of the given circle equation is (1, -2)
theorem find_circle_center : 
  ∃ (h k : ℝ), 3 * x^2 - 6 * x + 3 * y^2 + 12 * y - 75 = 0 → (h, k) = (1, -2) := 
by
  sorry

end find_circle_center_l892_89214


namespace star_number_of_intersections_2018_25_l892_89211

-- Definitions for the conditions
def rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def star_intersections (n k : ℕ) : ℕ := 
  n * (k - 1)

-- The main theorem
theorem star_number_of_intersections_2018_25 :
  2018 ≥ 5 ∧ 25 < 2018 / 2 ∧ rel_prime 2018 25 → 
  star_intersections 2018 25 = 48432 :=
by
  intros h
  sorry

end star_number_of_intersections_2018_25_l892_89211


namespace tankard_one_quarter_full_l892_89246

theorem tankard_one_quarter_full
  (C : ℝ) 
  (h : (3 / 4) * C = 480) : 
  (1 / 4) * C = 160 := 
by
  sorry

end tankard_one_quarter_full_l892_89246


namespace positive_difference_of_y_l892_89280

theorem positive_difference_of_y (y : ℝ) (h : (50 + y) / 2 = 35) : |50 - y| = 30 :=
by
  sorry

end positive_difference_of_y_l892_89280


namespace slower_train_pass_time_l892_89253

noncomputable def time_to_pass (length_train : ℕ) (speed_faster_kmh : ℕ) (speed_slower_kmh : ℕ) : ℕ :=
  let speed_faster_mps := speed_faster_kmh * 5 / 18
  let speed_slower_mps := speed_slower_kmh * 5 / 18
  let relative_speed := speed_faster_mps + speed_slower_mps
  let distance := length_train
  distance * 18 / (relative_speed * 5)

theorem slower_train_pass_time :
  time_to_pass 500 45 15 = 300 :=
by
  sorry

end slower_train_pass_time_l892_89253


namespace no_real_solution_exists_l892_89291

theorem no_real_solution_exists:
  ¬ ∃ (x y z : ℝ), (x ^ 2 + 4 * y * z + 2 * z = 0) ∧
                   (x + 2 * x * y + 2 * z ^ 2 = 0) ∧
                   (2 * x * z + y ^ 2 + y + 1 = 0) :=
by
  sorry

end no_real_solution_exists_l892_89291


namespace last_digit_to_appear_is_6_l892_89242

def modified_fib (n : ℕ) : ℕ :=
match n with
| 1 => 2
| 2 => 3
| n + 3 => modified_fib (n + 2) + modified_fib (n + 1)
| _ => 0 -- To silence the "missing cases" warning; won't be hit.

theorem last_digit_to_appear_is_6 :
  ∃ N : ℕ, ∀ n : ℕ, (n < N → ∃ d, d < 10 ∧ 
    (∀ m < n, (modified_fib m) % 10 ≠ d) ∧ d = 6) := sorry

end last_digit_to_appear_is_6_l892_89242


namespace intersection_of_solution_sets_solution_set_of_modified_inequality_l892_89230

open Set Real

theorem intersection_of_solution_sets :
  let A := {x | x ^ 2 - 2 * x - 3 < 0}
  let B := {x | x ^ 2 + x - 6 < 0}
  A ∩ B = {x | -1 < x ∧ x < 2} := by {
  sorry
}

theorem solution_set_of_modified_inequality :
  let A := {x | x ^ 2 + (-1) * x + (-2) < 0}
  A = {x | true} := by {
  sorry
}

end intersection_of_solution_sets_solution_set_of_modified_inequality_l892_89230


namespace smallest_positive_integer_x_l892_89204

theorem smallest_positive_integer_x :
  ∃ (x : ℕ), 0 < x ∧ (45 * x + 13) % 17 = 5 % 17 ∧ ∀ y : ℕ, 0 < y ∧ (45 * y + 13) % 17 = 5 % 17 → y ≥ x := 
sorry

end smallest_positive_integer_x_l892_89204
