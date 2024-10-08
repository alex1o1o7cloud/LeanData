import Mathlib

namespace margo_total_distance_l74_74480

-- Definitions based on the conditions
def time_to_friends_house_min : ℕ := 15
def time_to_return_home_min : ℕ := 25
def total_walking_time_min : ℕ := time_to_friends_house_min + time_to_return_home_min
def total_walking_time_hours : ℚ := total_walking_time_min / 60
def average_walking_rate_mph : ℚ := 3
def total_distance_miles : ℚ := average_walking_rate_mph * total_walking_time_hours

-- The statement of the proof problem
theorem margo_total_distance : total_distance_miles = 2 := by
  sorry

end margo_total_distance_l74_74480


namespace margie_drive_distance_l74_74557

theorem margie_drive_distance
  (miles_per_gallon : ℕ)
  (cost_per_gallon : ℕ)
  (dollar_amount : ℕ)
  (h₁ : miles_per_gallon = 32)
  (h₂ : cost_per_gallon = 4)
  (h₃ : dollar_amount = 20) :
  (dollar_amount / cost_per_gallon) * miles_per_gallon = 160 :=
by
  sorry

end margie_drive_distance_l74_74557


namespace blown_out_sand_dunes_l74_74410

theorem blown_out_sand_dunes (p_remain p_lucky p_both : ℝ) (h_rem: p_remain = 1 / 3) (h_luck: p_lucky = 2 / 3)
(h_both: p_both = 0.08888888888888889) : 
  ∃ N : ℕ, N = 8 :=
by
  sorry

end blown_out_sand_dunes_l74_74410


namespace domain_of_f_l74_74268

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 6*x + 9)

theorem domain_of_f :
  {x : ℝ | f x ≠ f (-3)} = Iio (-3) ∪ Ioi (-3) :=
by
  sorry

end domain_of_f_l74_74268


namespace perimeter_of_rhombus_l74_74730

theorem perimeter_of_rhombus (d1 d2 : ℝ) (hd1 : d1 = 8) (hd2 : d2 = 30) :
  (perimeter : ℝ) = 4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) :=
by
  simp [hd1, hd2]
  sorry

end perimeter_of_rhombus_l74_74730


namespace students_above_90_l74_74708

theorem students_above_90 (total_students : ℕ) (above_90_chinese : ℕ) (above_90_math : ℕ)
  (all_above_90_at_least_one_subject : total_students = 50 ∧ above_90_chinese = 33 ∧ above_90_math = 38 ∧ 
    ∀ (n : ℕ), n < total_students → (n < above_90_chinese ∨ n < above_90_math)) :
  (above_90_chinese + above_90_math - total_students) = 21 :=
by
  sorry

end students_above_90_l74_74708


namespace operation_on_original_number_l74_74808

theorem operation_on_original_number (f : ℕ → ℕ) (x : ℕ) (h : 3 * (f x + 9) = 51) (hx : x = 4) :
  f x = 2 * x :=
by
  sorry

end operation_on_original_number_l74_74808


namespace problem_a_b_l74_74011

theorem problem_a_b (a b : ℝ) (h₁ : a + b = 10) (h₂ : a - b = 4) : a^2 - b^2 = 40 :=
by
  sorry

end problem_a_b_l74_74011


namespace complete_square_example_l74_74457

theorem complete_square_example :
  ∃ c : ℝ, ∃ d : ℝ, (∀ x : ℝ, x^2 + 12 * x + 4 = (x + c)^2 - d) ∧ d = 32 := by
  sorry

end complete_square_example_l74_74457


namespace george_money_left_after_donations_and_groceries_l74_74033

def monthly_income : ℕ := 240
def donation (income : ℕ) : ℕ := income / 2
def post_donation_money (income : ℕ) : ℕ := income - donation income
def groceries_cost : ℕ := 20
def money_left (income : ℕ) : ℕ := post_donation_money income - groceries_cost

theorem george_money_left_after_donations_and_groceries :
  money_left monthly_income = 100 :=
by
  sorry

end george_money_left_after_donations_and_groceries_l74_74033


namespace part_I_solution_part_II_solution_l74_74509

-- Defining f(x) given parameters a and b
def f (x a b : ℝ) := |x - a| + |x + b|

-- Part (I): Given a = 1 and b = 2, solve the inequality f(x) ≤ 5
theorem part_I_solution (x : ℝ) : 
  (f x 1 2) ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
by
  sorry

-- Part (II): Given the minimum value of f(x) is 3, find min (a^2 / b + b^2 / a)
theorem part_II_solution (a b : ℝ) (h : 3 = |a| + |b|) (ha : a > 0) (hb : b > 0) : 
  (min (a^2 / b + b^2 / a)) = 3 := 
by
  sorry

end part_I_solution_part_II_solution_l74_74509


namespace silver_value_percentage_l74_74672

theorem silver_value_percentage
  (side_length : ℝ) (weight_per_cubic_inch : ℝ) (price_per_ounce : ℝ) 
  (selling_price : ℝ) (volume : ℝ) (weight : ℝ) (silver_value : ℝ) 
  (percentage_sold : ℝ ) 
  (h1 : side_length = 3) 
  (h2 : weight_per_cubic_inch = 6) 
  (h3 : price_per_ounce = 25)
  (h4 : selling_price = 4455)
  (h5 : volume = side_length^3)
  (h6 : weight = volume * weight_per_cubic_inch)
  (h7 : silver_value = weight * price_per_ounce)
  (h8 : percentage_sold = (selling_price / silver_value) * 100) :
  percentage_sold = 110 :=
by
  sorry

end silver_value_percentage_l74_74672


namespace charlies_age_22_l74_74356

variable (A : ℕ) (C : ℕ)

theorem charlies_age_22 (h1 : C = 2 * A + 8) (h2 : C = 22) : A = 7 := by
  sorry

end charlies_age_22_l74_74356


namespace nathan_has_83_bananas_l74_74706

def nathan_bananas (bunches_eight bananas_eight bunches_seven bananas_seven: Nat) : Nat :=
  bunches_eight * bananas_eight + bunches_seven * bananas_seven

theorem nathan_has_83_bananas (h1 : bunches_eight = 6) (h2 : bananas_eight = 8) (h3 : bunches_seven = 5) (h4 : bananas_seven = 7) : 
  nathan_bananas bunches_eight bananas_eight bunches_seven bananas_seven = 83 := by
  sorry

end nathan_has_83_bananas_l74_74706


namespace usual_time_to_cover_distance_l74_74486

theorem usual_time_to_cover_distance (S T : ℝ) (h1 : 0.75 * S = S / (T + 24)) (h2 : S * T = 0.75 * S * (T + 24)) : T = 72 :=
by
  sorry

end usual_time_to_cover_distance_l74_74486


namespace given_problem_l74_74438

noncomputable def improper_fraction_5_2_7 : ℚ := 37 / 7
noncomputable def improper_fraction_6_1_3 : ℚ := 19 / 3
noncomputable def improper_fraction_3_1_2 : ℚ := 7 / 2
noncomputable def improper_fraction_2_1_5 : ℚ := 11 / 5

theorem given_problem :
  71 * (improper_fraction_5_2_7 - improper_fraction_6_1_3) / (improper_fraction_3_1_2 + improper_fraction_2_1_5) = -13 - 37 / 1197 := 
  sorry

end given_problem_l74_74438


namespace no_real_solution_l74_74767

theorem no_real_solution (x y : ℝ) : x^3 + y^2 = 2 → x^2 + x * y + y^2 - y = 0 → false := 
by 
  intro h1 h2
  sorry

end no_real_solution_l74_74767


namespace find_x_l74_74084

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 182) : x = 13 :=
sorry

end find_x_l74_74084


namespace inner_tetrahedron_volume_l74_74678

def volume_of_inner_tetrahedron(cube_side : ℕ) : ℚ :=
  let base_area := (cube_side * cube_side) / 2
  let height := cube_side
  let original_tetra_volume := (1 / 3) * base_area * height
  let inner_tetra_volume := original_tetra_volume / 8
  inner_tetra_volume

theorem inner_tetrahedron_volume {cube_side : ℕ} (h : cube_side = 2) : 
  volume_of_inner_tetrahedron cube_side = 1 / 6 := 
by
  rw [h]
  unfold volume_of_inner_tetrahedron 
  norm_num
  sorry

end inner_tetrahedron_volume_l74_74678


namespace transformed_mean_stddev_l74_74710

variables (n : ℕ) (x : Fin n → ℝ)

-- Given conditions
def mean_is_4 (mean : ℝ) : Prop :=
  mean = 4

def stddev_is_7 (stddev : ℝ) : Prop :=
  stddev = 7

-- Definitions for transformations and the results
def transformed_mean (mean : ℝ) : ℝ :=
  3 * mean + 2

def transformed_stddev (stddev : ℝ) : ℝ :=
  3 * stddev

-- The proof problem
theorem transformed_mean_stddev (mean stddev : ℝ) 
  (h_mean : mean_is_4 mean) 
  (h_stddev : stddev_is_7 stddev) :
  transformed_mean mean = 14 ∧ transformed_stddev stddev = 21 :=
by
  rw [h_mean, h_stddev]
  unfold transformed_mean transformed_stddev
  rw [← h_mean, ← h_stddev]
  sorry

end transformed_mean_stddev_l74_74710


namespace domain_of_function_l74_74156

theorem domain_of_function :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x ≠ 0} = {x : ℝ | -2 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_function_l74_74156


namespace angle_SRT_l74_74039

-- Define angles in degrees
def angle_P : ℝ := 50
def angle_Q : ℝ := 60
def angle_R : ℝ := 40

-- Define the problem: Prove that angle SRT is 30 degrees given the above conditions
theorem angle_SRT : 
  (angle_P = 50 ∧ angle_Q = 60 ∧ angle_R = 40) → (∃ angle_SRT : ℝ, angle_SRT = 30) :=
by
  intros h
  sorry

end angle_SRT_l74_74039


namespace cost_of_baseball_cards_l74_74226

variables (cost_football cost_pokemon total_spent cost_baseball : ℝ)
variable (h1 : cost_football = 2 * 2.73)
variable (h2 : cost_pokemon = 4.01)
variable (h3 : total_spent = 18.42)
variable (total_cost_football_pokemon : ℝ)
variable (h4 : total_cost_football_pokemon = cost_football + cost_pokemon)

theorem cost_of_baseball_cards
  (h : cost_baseball = total_spent - total_cost_football_pokemon) : 
  cost_baseball = 8.95 :=
by
  sorry

end cost_of_baseball_cards_l74_74226


namespace seventh_oblong_is_56_l74_74181

def oblong (n : ℕ) : ℕ := n * (n + 1)

theorem seventh_oblong_is_56 : oblong 7 = 56 := by
  sorry

end seventh_oblong_is_56_l74_74181


namespace seventh_triangular_number_is_28_l74_74422

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem seventh_triangular_number_is_28 : triangular_number 7 = 28 :=
by
  /- proof goes here -/
  sorry

end seventh_triangular_number_is_28_l74_74422


namespace problem_statements_l74_74248

noncomputable def f (x : ℕ) : ℕ := x % 2
noncomputable def g (x : ℕ) : ℕ := x % 3

theorem problem_statements (x : ℕ) : (f (2 * x) = 0) ∧ (f x + f (x + 3) = 1) :=
by
  sorry

end problem_statements_l74_74248


namespace rectangle_width_l74_74872

theorem rectangle_width (w : ℝ)
    (h₁ : 5 > 0) (h₂ : 6 > 0) (h₃ : 3 > 0) 
    (area_relation : w * 5 = 3 * 6 + 2) : w = 4 :=
by
  sorry

end rectangle_width_l74_74872


namespace sum_abc_equals_16_l74_74883

theorem sum_abc_equals_16 (a b c : ℝ) (h : (a - 2)^2 + (b - 6)^2 + (c - 8)^2 = 0) : 
  a + b + c = 16 :=
by
  sorry

end sum_abc_equals_16_l74_74883


namespace sin_double_angle_of_tan_l74_74765

-- Given condition: tan(alpha) = 2
-- To prove: sin(2 * alpha) = 4/5
theorem sin_double_angle_of_tan (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 :=
  sorry

end sin_double_angle_of_tan_l74_74765


namespace thomas_monthly_earnings_l74_74300

def weekly_earnings : ℕ := 4550
def weeks_in_month : ℕ := 4
def monthly_earnings : ℕ := weekly_earnings * weeks_in_month

theorem thomas_monthly_earnings : monthly_earnings = 18200 := by
  sorry

end thomas_monthly_earnings_l74_74300


namespace certain_number_z_l74_74044

theorem certain_number_z (x y z : ℝ) (h1 : 0.5 * x = y + z) (h2 : x - 2 * y = 40) : z = 20 :=
by 
  sorry

end certain_number_z_l74_74044


namespace maria_purse_value_l74_74846

def value_of_nickels (num_nickels : ℕ) : ℕ := num_nickels * 5
def value_of_dimes (num_dimes : ℕ) : ℕ := num_dimes * 10
def value_of_quarters (num_quarters : ℕ) : ℕ := num_quarters * 25
def total_value (num_nickels num_dimes num_quarters : ℕ) : ℕ := 
  value_of_nickels num_nickels + value_of_dimes num_dimes + value_of_quarters num_quarters
def percentage_of_dollar (value_cents : ℕ) : ℕ := value_cents * 100 / 100

theorem maria_purse_value : percentage_of_dollar (total_value 2 3 2) = 90 := by
  sorry

end maria_purse_value_l74_74846


namespace tutors_next_together_l74_74449

-- Define the conditions given in the problem
def Elisa_work_days := 5
def Frank_work_days := 6
def Giselle_work_days := 8
def Hector_work_days := 9

-- Theorem statement to prove the number of days until they all work together again
theorem tutors_next_together (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = Elisa_work_days) 
  (h2 : d2 = Frank_work_days) 
  (h3 : d3 = Giselle_work_days) 
  (h4 : d4 = Hector_work_days) : 
  Nat.lcm (Nat.lcm (Nat.lcm d1 d2) d3) d4 = 360 := 
by
  -- Translate the problem statement into Lean terms and structure
  sorry

end tutors_next_together_l74_74449


namespace anthony_more_than_mabel_l74_74311

noncomputable def transactions := 
  let M := 90  -- Mabel's transactions
  let J := 82  -- Jade's transactions
  let C := J - 16  -- Cal's transactions
  let A := (3 / 2) * C  -- Anthony's transactions
  let P := ((A - M) / M) * 100 -- Percentage more transactions Anthony handled than Mabel
  P

theorem anthony_more_than_mabel : transactions = 10 := by
  sorry

end anthony_more_than_mabel_l74_74311


namespace x_power_2023_zero_or_neg_two_l74_74184

variable {x : ℂ} -- Assuming x is a complex number to handle general roots of unity.

theorem x_power_2023_zero_or_neg_two 
  (h1 : (x - 1) * (x + 1) = x^2 - 1)
  (h2 : (x - 1) * (x^2 + x + 1) = x^3 - 1)
  (h3 : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1)
  (pattern : (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0) :
  x^2023 - 1 = 0 ∨ x^2023 - 1 = -2 :=
by
  sorry

end x_power_2023_zero_or_neg_two_l74_74184


namespace not_equal_fractions_l74_74174

theorem not_equal_fractions :
  ¬ ((14 / 12 = 7 / 6) ∧
     (1 + 1 / 6 = 7 / 6) ∧
     (21 / 18 = 7 / 6) ∧
     (1 + 2 / 12 = 7 / 6) ∧
     (1 + 1 / 3 = 7 / 6)) :=
by 
  sorry

end not_equal_fractions_l74_74174


namespace bracelet_arrangements_l74_74550

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem bracelet_arrangements : 
  (factorial 8) / (8 * 2) = 2520 := by
    sorry

end bracelet_arrangements_l74_74550


namespace cherry_ratio_l74_74933

theorem cherry_ratio (total_lollipops cherry_lollipops watermelon_lollipops sour_apple_lollipops grape_lollipops : ℕ) 
  (h_total : total_lollipops = 42) 
  (h_rest_equally_distributed : watermelon_lollipops = sour_apple_lollipops ∧ sour_apple_lollipops = grape_lollipops) 
  (h_grape : grape_lollipops = 7) 
  (h_total_sum : cherry_lollipops + watermelon_lollipops + sour_apple_lollipops + grape_lollipops = total_lollipops) : 
  cherry_lollipops = 21 ∧ (cherry_lollipops : ℚ) / total_lollipops = 1 / 2 :=
by
  sorry

end cherry_ratio_l74_74933


namespace f_zero_f_odd_f_not_decreasing_f_increasing_l74_74858

noncomputable def f (x : ℝ) : ℝ := sorry -- The function definition is abstract.

-- Functional equation condition
axiom functional_eq (x y : ℝ) (h1 : -1 < x) (h2 : x < 1) (h3 : -1 < y) (h4 : y < 1) : 
  f x + f y = f ((x + y) / (1 + x * y))

-- Condition for negative interval
axiom neg_interval (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : f x < 0

-- Statements to prove

-- a): f(0) = 0
theorem f_zero : f 0 = 0 := 
by
  sorry

-- b): f(x) is an odd function
theorem f_odd (x : ℝ) (h1 : -1 < x) (h2 : x < 1) : f (-x) = -f x := 
by
  sorry

-- c): f(x) is not a decreasing function
theorem f_not_decreasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : ¬(f x1 > f x2) :=
by
  sorry

-- d): f(x) is an increasing function
theorem f_increasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : f x1 < f x2 :=
by
  sorry

end f_zero_f_odd_f_not_decreasing_f_increasing_l74_74858


namespace geometric_sequence_product_proof_l74_74688

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geometric_sequence_product_proof (a : ℕ → ℝ) (q : ℝ)
  (h_geo : geometric_sequence a q) 
  (h1 : a 2010 * a 2011 * a 2012 = 3)
  (h2 : a 2013 * a 2014 * a 2015 = 24) :
  a 2016 * a 2017 * a 2018 = 192 :=
sorry

end geometric_sequence_product_proof_l74_74688


namespace parallel_lines_a_value_l74_74211

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ((a + 1) * x + 3 * y + 3 = 0) → (x + (a - 1) * y + 1 = 0)) → a = -2 :=
by
  sorry

end parallel_lines_a_value_l74_74211


namespace contest_score_difference_l74_74376

theorem contest_score_difference :
  let percent_50 := 0.05
  let percent_60 := 0.20
  let percent_70 := 0.25
  let percent_80 := 0.30
  let percent_90 := 1 - (percent_50 + percent_60 + percent_70 + percent_80)
  let mean := (percent_50 * 50) + (percent_60 * 60) + (percent_70 * 70) + (percent_80 * 80) + (percent_90 * 90)
  let median := 70
  median - mean = -4 :=
by
  sorry

end contest_score_difference_l74_74376


namespace other_root_l74_74111

theorem other_root (m : ℝ) (x : ℝ) (hx : 3 * x ^ 2 + m * x - 7 = 0) (root1 : x = 1) :
  ∃ y : ℝ, 3 * y ^ 2 + m * y - 7 = 0 ∧ y = -7 / 3 :=
by
  sorry

end other_root_l74_74111


namespace toy_cars_in_third_box_l74_74002

theorem toy_cars_in_third_box (total_cars first_box second_box : ℕ) (H1 : total_cars = 71) 
    (H2 : first_box = 21) (H3 : second_box = 31) : total_cars - (first_box + second_box) = 19 :=
by
  sorry

end toy_cars_in_third_box_l74_74002


namespace second_trial_amount_691g_l74_74313

theorem second_trial_amount_691g (low high : ℝ) (h_range : low = 500) (h_high : high = 1000) (h_method : ∃ x, x = 0.618) : 
  high - 0.618 * (high - low) = 691 :=
by
  sorry

end second_trial_amount_691g_l74_74313


namespace complement_of_A_in_U_l74_74975

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}

theorem complement_of_A_in_U : (U \ A) = {1, 3, 5} := by
  sorry

end complement_of_A_in_U_l74_74975


namespace rate_of_mixed_oil_per_litre_l74_74035

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

end rate_of_mixed_oil_per_litre_l74_74035


namespace unique_integer_cube_triple_l74_74995

theorem unique_integer_cube_triple (x : ℤ) (h : x^3 < 3 * x) : x = 1 := 
sorry

end unique_integer_cube_triple_l74_74995


namespace geometric_fraction_l74_74143

noncomputable def a_n : ℕ → ℝ := sorry
axiom a1_a5_condition : a_n 1 + a_n 5 = a_n 1 * a_n 5
axiom geometric_sequence : ∀ n, a_n (n + 1) = a_n n * a_n (n + 1) / a_n (n - 1) 

theorem geometric_fraction (a_n : ℕ → ℝ) (a1_a5_condition : a_n 1 + a_n 5 = a_n 1 * a_n 5) :
  (a_n 13) / (a_n 9) = 9 :=
sorry

end geometric_fraction_l74_74143


namespace multiply_expression_l74_74500

theorem multiply_expression (x : ℝ) : (x^4 + 12 * x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end multiply_expression_l74_74500


namespace base_area_of_cylinder_l74_74199

variables (S : ℝ) (cylinder : Type)
variables (square_cross_section : cylinder → Prop) (area_square : cylinder → ℝ)
variables (base_area : cylinder → ℝ)

-- Assume that the cylinder has a square cross-section with a given area
axiom cross_section_square : ∀ c : cylinder, square_cross_section c → area_square c = 4 * S

-- Theorem stating the area of the base of the cylinder
theorem base_area_of_cylinder (c : cylinder) (h : square_cross_section c) : base_area c = π * S :=
by
  -- Proof omitted
  sorry

end base_area_of_cylinder_l74_74199


namespace negation_of_implication_l74_74058

-- Definitions based on the conditions from part (a)
def original_prop (x : ℝ) : Prop := x > 5 → x > 0
def negation_candidate_A (x : ℝ) : Prop := x ≤ 5 → x ≤ 0

-- The goal is to prove that the negation of the original proposition
-- is equivalent to option A, that is:
theorem negation_of_implication (x : ℝ) : (¬ (x > 5 → x > 0)) = (x ≤ 5 → x ≤ 0) :=
by
  sorry

end negation_of_implication_l74_74058


namespace cost_of_fixing_clothes_l74_74041

def num_shirts : ℕ := 10
def num_pants : ℕ := 12
def time_per_shirt : ℝ := 1.5
def time_per_pant : ℝ := 3.0
def rate_per_hour : ℝ := 30.0

theorem cost_of_fixing_clothes : 
  let total_time := (num_shirts * time_per_shirt) + (num_pants * time_per_pant)
  let total_cost := total_time * rate_per_hour
  total_cost = 1530 :=
by 
  sorry

end cost_of_fixing_clothes_l74_74041


namespace sam_initial_puppies_l74_74729

theorem sam_initial_puppies (gave_away : ℝ) (now_has : ℝ) (initially : ℝ) 
    (h1 : gave_away = 2.0) (h2 : now_has = 4.0) : initially = 6.0 :=
by
  sorry

end sam_initial_puppies_l74_74729


namespace ratio_son_grandson_l74_74260

-- Define the conditions
variables (Markus_age Son_age Grandson_age : ℕ)
axiom Markus_twice_son : Markus_age = 2 * Son_age
axiom sum_ages : Markus_age + Son_age + Grandson_age = 140
axiom Grandson_age_20 : Grandson_age = 20

-- Define the goal to prove
theorem ratio_son_grandson : (Son_age : ℚ) / Grandson_age = 2 :=
by
  sorry

end ratio_son_grandson_l74_74260


namespace problem_solution_l74_74997

variables {a b c : ℝ}

theorem problem_solution (h : a + b + c = 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^3 * b^3 / ((a^3 - b^2 * c) * (b^3 - a^2 * c)) +
  a^3 * c^3 / ((a^3 - b^2 * c) * (c^3 - a^2 * b)) +
  b^3 * c^3 / ((b^3 - a^2 * c) * (c^3 - a^2 * b))) = 1 :=
sorry

end problem_solution_l74_74997


namespace remainder_of_p_div_x_minus_3_l74_74483

def p (x : ℝ) : ℝ := x^4 - x^3 - 4 * x + 7

theorem remainder_of_p_div_x_minus_3 : 
  let remainder := p 3 
  remainder = 49 := 
by
  sorry

end remainder_of_p_div_x_minus_3_l74_74483


namespace Robert_can_read_one_book_l74_74517

def reading_speed : ℕ := 100 -- pages per hour
def book_length : ℕ := 350 -- pages
def available_time : ℕ := 5 -- hours

theorem Robert_can_read_one_book :
  (available_time * reading_speed) >= book_length ∧ 
  (available_time * reading_speed) < 2 * book_length :=
by {
  -- The proof steps are omitted as instructed.
  sorry
}

end Robert_can_read_one_book_l74_74517


namespace time_difference_l74_74812

-- Definitions for the conditions
def blocks_to_office : Nat := 12
def walk_time_per_block : Nat := 1 -- time in minutes
def bike_time_per_block : Nat := 20 / 60 -- time in minutes, converted from seconds

-- Definitions for the total times
def walk_time : Nat := blocks_to_office * walk_time_per_block
def bike_time : Nat := blocks_to_office * bike_time_per_block

-- Theorem statement
theorem time_difference : walk_time - bike_time = 8 := by
  -- Proof omitted
  sorry

end time_difference_l74_74812


namespace notepad_duration_l74_74043

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

end notepad_duration_l74_74043


namespace gcd_lcm_product_l74_74531

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 150) : 
  Nat.gcd a b * Nat.lcm a b = 13500 := 
by 
  sorry

end gcd_lcm_product_l74_74531


namespace garden_area_maximal_l74_74240

/-- Given a garden with sides 20 meters, 16 meters, 12 meters, and 10 meters, 
    prove that the area is approximately 194.4 square meters. -/
theorem garden_area_maximal (a b c d : ℝ) (h1 : a = 20) (h2 : b = 16) (h3 : c = 12) (h4 : d = 10) :
    ∃ A : ℝ, abs (A - 194.4) < 0.1 :=
by
  sorry

end garden_area_maximal_l74_74240


namespace sum_of_prime_factors_1729728_l74_74632

def prime_factors_sum (n : ℕ) : ℕ := 
  -- Suppose that a function defined to calculate the sum of distinct prime factors
  -- In a practical setting, you would define this function or use an existing library
  sorry 

theorem sum_of_prime_factors_1729728 : prime_factors_sum 1729728 = 36 :=
by {
  -- Proof would go here
  sorry
}

end sum_of_prime_factors_1729728_l74_74632


namespace diagonals_in_octagon_l74_74683

/-- The formula to calculate the number of diagonals in a polygon -/
def number_of_diagonals (n : Nat) : Nat :=
  (n * (n - 3)) / 2

/-- The number of sides in an octagon -/
def sides_of_octagon : Nat := 8

/-- The number of diagonals in an octagon is 20. -/
theorem diagonals_in_octagon : number_of_diagonals sides_of_octagon = 20 :=
by
  sorry

end diagonals_in_octagon_l74_74683


namespace friends_travelled_distance_l74_74766

theorem friends_travelled_distance :
  let lionel_distance : ℝ := 4 * 5280
  let esther_distance : ℝ := 975 * 3
  let niklaus_distance : ℝ := 1287
  let isabella_distance : ℝ := 18 * 1000 * 3.28084
  let sebastian_distance : ℝ := 2400 * 3.28084
  let total_distance := lionel_distance + esther_distance + niklaus_distance + isabella_distance + sebastian_distance
  total_distance = 91261.136 := 
by
  sorry

end friends_travelled_distance_l74_74766


namespace LCM_is_4199_l74_74873

theorem LCM_is_4199 :
  let beats_of_cymbals := 13
  let beats_of_triangle := 17
  let beats_of_tambourine := 19
  Nat.lcm (Nat.lcm beats_of_cymbals beats_of_triangle) beats_of_tambourine = 4199 := 
by 
  sorry 

end LCM_is_4199_l74_74873


namespace sin_cos_15_deg_l74_74951

noncomputable def sin_deg (deg : ℝ) : ℝ := Real.sin (deg * Real.pi / 180)
noncomputable def cos_deg (deg : ℝ) : ℝ := Real.cos (deg * Real.pi / 180)

theorem sin_cos_15_deg :
  (sin_deg 15 + cos_deg 15) * (sin_deg 15 - cos_deg 15) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_cos_15_deg_l74_74951


namespace log_sqrt_pi_simplification_l74_74036

theorem log_sqrt_pi_simplification:
  2 * Real.log 4 + Real.log (5 / 8) + Real.sqrt ((Real.sqrt 3 - Real.pi) ^ 2) = 1 + Real.pi - Real.sqrt 3 :=
sorry

end log_sqrt_pi_simplification_l74_74036


namespace mandy_bike_time_l74_74757

-- Definitions of the ratios and time spent on yoga
def ratio_gym_bike : ℕ × ℕ := (2, 3)
def ratio_yoga_exercise : ℕ × ℕ := (2, 3)
def time_yoga : ℕ := 20

-- Theorem stating that Mandy will spend 18 minutes riding her bike
theorem mandy_bike_time (r_gb : ℕ × ℕ) (r_ye : ℕ × ℕ) (t_y : ℕ) 
  (h_rgb : r_gb = (2, 3)) (h_rye : r_ye = (2, 3)) (h_ty : t_y = 20) : 
  let t_e := (r_ye.snd * t_y) / r_ye.fst
  let t_part := t_e / (r_gb.fst + r_gb.snd)
  t_part * r_gb.snd = 18 := sorry

end mandy_bike_time_l74_74757


namespace pencils_bought_at_cost_price_l74_74008

variable (C S : ℝ)
variable (n : ℕ)

theorem pencils_bought_at_cost_price (h1 : n * C = 8 * S) (h2 : S = 1.5 * C) : n = 12 := 
by sorry

end pencils_bought_at_cost_price_l74_74008


namespace largest_num_pencils_in_package_l74_74479

theorem largest_num_pencils_in_package (Ming_pencils Catherine_pencils : ℕ) 
  (Ming_pencils := 40) 
  (Catherine_pencils := 24) 
  (H : ∃ k, Ming_pencils = k * a ∧ Catherine_pencils = k * b) :
  gcd Ming_pencils Catherine_pencils = 8 :=
by
  sorry

end largest_num_pencils_in_package_l74_74479


namespace mr_green_yield_l74_74585

noncomputable def steps_to_feet (steps : ℕ) : ℝ :=
  steps * 2.5

noncomputable def total_yield (steps_x : ℕ) (steps_y : ℕ) (yield_potato_per_sqft : ℝ) (yield_carrot_per_sqft : ℝ) : ℝ :=
  let width := steps_to_feet steps_x
  let height := steps_to_feet steps_y
  let area := width * height
  (area * yield_potato_per_sqft) + (area * yield_carrot_per_sqft)

theorem mr_green_yield :
  total_yield 20 25 0.5 0.25 = 2343.75 :=
by
  sorry

end mr_green_yield_l74_74585


namespace chosen_number_is_5_l74_74216

theorem chosen_number_is_5 (x : ℕ) (h_pos : x > 0)
  (h_eq : ((10 * x + 5 - x^2) / x) - x = 1) : x = 5 :=
by
  sorry

end chosen_number_is_5_l74_74216


namespace compare_expressions_l74_74066

theorem compare_expressions (a b : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) :=
by {
  sorry
}

end compare_expressions_l74_74066


namespace iggy_pace_l74_74627

theorem iggy_pace 
  (monday_miles : ℕ) (tuesday_miles : ℕ) (wednesday_miles : ℕ)
  (thursday_miles : ℕ) (friday_miles : ℕ) (total_hours : ℕ) 
  (h1 : monday_miles = 3) (h2 : tuesday_miles = 4) 
  (h3 : wednesday_miles = 6) (h4 : thursday_miles = 8) 
  (h5 : friday_miles = 3) (h6 : total_hours = 4) :
  (total_hours * 60) / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles) = 10 :=
sorry

end iggy_pace_l74_74627


namespace num_perfect_cubes_between_bounds_l74_74083

   noncomputable def lower_bound := 2^8 + 1
   noncomputable def upper_bound := 2^18 + 1

   theorem num_perfect_cubes_between_bounds : 
     ∃ (k : ℕ), k = 58 ∧ (∀ (n : ℕ), (lower_bound ≤ n^3 ∧ n^3 ≤ upper_bound) ↔ (7 ≤ n ∧ n ≤ 64)) :=
   sorry
   
end num_perfect_cubes_between_bounds_l74_74083


namespace pure_imaginary_complex_number_l74_74485

theorem pure_imaginary_complex_number (m : ℝ) (h : (m^2 - 3*m) = 0) :
  (m^2 - 5*m + 6) ≠ 0 → m = 0 :=
by
  intro h_im
  have h_fact : (m = 0) ∨ (m = 3) := by
    sorry -- This is where the factorization steps would go
  cases h_fact with
  | inl h0 =>
    assumption
  | inr h3 =>
    exfalso
    have : (3^2 - 5*3 + 6) = 0 := by
      sorry -- Simplify to check that m = 3 is not a valid solution
    contradiction

end pure_imaginary_complex_number_l74_74485


namespace integer_satisfies_mod_and_range_l74_74332

theorem integer_satisfies_mod_and_range :
  ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ (-150 ≡ n [ZMOD 25]) → n = 0 :=
by
  sorry

end integer_satisfies_mod_and_range_l74_74332


namespace total_cards_needed_l74_74202

def red_card_credits := 3
def blue_card_credits := 5
def total_credits := 84
def red_cards := 8

theorem total_cards_needed :
  red_card_credits * red_cards + blue_card_credits * (total_credits - red_card_credits * red_cards) / blue_card_credits = 20 := by
  sorry

end total_cards_needed_l74_74202


namespace greatest_non_fiction_books_l74_74976

def is_prime (p : ℕ) := p > 1 ∧ (∀ d : ℕ, d ∣ p → d = 1 ∨ d = p)

theorem greatest_non_fiction_books (n f k : ℕ) :
  (n + f = 100 ∧ f = n + k ∧ is_prime k) → n ≤ 49 :=
by
  sorry

end greatest_non_fiction_books_l74_74976


namespace am_gm_inequality_l74_74817

theorem am_gm_inequality (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - (Real.sqrt (a * b)) ∧ 
  (a + b) / 2 - (Real.sqrt (a * b)) < (a - b)^2 / (8 * b) := 
sorry

end am_gm_inequality_l74_74817


namespace children_less_than_adults_l74_74985

theorem children_less_than_adults (total_members : ℕ)
  (percent_adults : ℝ) (percent_teenagers : ℝ) (percent_children : ℝ) :
  total_members = 500 →
  percent_adults = 0.45 →
  percent_teenagers = 0.25 →
  percent_children = 1 - percent_adults - percent_teenagers →
  (percent_children * total_members) - (percent_adults * total_members) = -75 := 
by
  intros h_total h_adults h_teenagers h_children
  sorry

end children_less_than_adults_l74_74985


namespace log_expression_evaluation_l74_74610

open Real

theorem log_expression_evaluation : log 5 * log 20 + (log 2) ^ 2 = 1 := 
sorry

end log_expression_evaluation_l74_74610


namespace total_cows_l74_74465

variable (D C : ℕ)

-- The conditions of the problem translated to Lean definitions
def total_heads := D + C
def total_legs := 2 * D + 4 * C 

-- The main theorem based on the conditions and the result to prove
theorem total_cows (h1 : total_legs D C = 2 * total_heads D C + 40) : C = 20 :=
by
  sorry


end total_cows_l74_74465


namespace solve_inequality_l74_74576

theorem solve_inequality (x : ℝ) : 6 - x - 2 * x^2 < 0 ↔ x < -2 ∨ x > 3 / 2 := sorry

end solve_inequality_l74_74576


namespace fraction_of_yard_occupied_l74_74395

noncomputable def area_triangle_flower_bed : ℝ := 
  2 * (0.5 * (10:ℝ) * (10:ℝ))

noncomputable def area_circular_flower_bed : ℝ := 
  Real.pi * (2:ℝ)^2

noncomputable def total_area_flower_beds : ℝ := 
  area_triangle_flower_bed + area_circular_flower_bed

noncomputable def area_yard : ℝ := 
  (40:ℝ) * (10:ℝ)

noncomputable def fraction_occupied := 
  total_area_flower_beds / area_yard

theorem fraction_of_yard_occupied : 
  fraction_occupied = 0.2814 := 
sorry

end fraction_of_yard_occupied_l74_74395


namespace quadratic_zeros_l74_74369

theorem quadratic_zeros (a b : ℝ) (h1 : (4 - 2 * a + b = 0)) (h2 : (9 + 3 * a + b = 0)) : a + b = -7 := 
by
  sorry

end quadratic_zeros_l74_74369


namespace vacation_fund_percentage_l74_74003

variable (s : ℝ) (vs : ℝ)
variable (d : ℝ)
variable (v : ℝ)

-- conditions:
-- 1. Jill's net monthly salary
#check (s = 3700)
-- 2. Jill's discretionary income is one fifth of her salary
#check (d = s / 5)
-- 3. Savings percentage
#check (0.20 * d)
-- 4. Eating out and socializing percentage
#check (0.35 * d)
-- 5. Gifts and charitable causes
#check (111)

-- Prove: 
theorem vacation_fund_percentage : 
  s = 3700 -> d = s / 5 -> 
  (v * d + 0.20 * d + 0.35 * d + 111 = d) -> 
  v = 222 / 740 :=
by
  sorry -- proof skipped

end vacation_fund_percentage_l74_74003


namespace ratio_length_to_width_l74_74571

def garden_length := 80
def garden_perimeter := 240

theorem ratio_length_to_width : ∃ W, 2 * garden_length + 2 * W = garden_perimeter ∧ garden_length / W = 2 := by
  sorry

end ratio_length_to_width_l74_74571


namespace pool_ratio_l74_74238

theorem pool_ratio 
  (total_pools : ℕ)
  (ark_athletic_wear_pools : ℕ)
  (total_pools_eq : total_pools = 800)
  (ark_athletic_wear_pools_eq : ark_athletic_wear_pools = 200)
  : ((total_pools - ark_athletic_wear_pools) / ark_athletic_wear_pools) = 3 :=
by
  sorry

end pool_ratio_l74_74238


namespace weight_of_3_moles_HBrO3_is_386_73_l74_74172

noncomputable def H_weight : ℝ := 1.01
noncomputable def Br_weight : ℝ := 79.90
noncomputable def O_weight : ℝ := 16.00
noncomputable def HBrO3_weight : ℝ := H_weight + Br_weight + 3 * O_weight
noncomputable def weight_of_3_moles_of_HBrO3 : ℝ := 3 * HBrO3_weight

theorem weight_of_3_moles_HBrO3_is_386_73 : weight_of_3_moles_of_HBrO3 = 386.73 := by
  sorry

end weight_of_3_moles_HBrO3_is_386_73_l74_74172


namespace jovana_bucket_shells_l74_74468

theorem jovana_bucket_shells :
  let a0 := 5.2
  let a1 := a0 + 15.7
  let a2 := a1 + 17.5
  let a3 := a2 - 4.3
  let a4 := 3 * a3
  a4 = 102.3 := 
by
  sorry

end jovana_bucket_shells_l74_74468


namespace intersection_nonempty_condition_l74_74609

theorem intersection_nonempty_condition (m n : ℝ) :
  (∃ x : ℝ, (m - 1 < x ∧ x < m + 1) ∧ (3 - n < x ∧ x < 4 - n)) ↔ (2 < m + n ∧ m + n < 5) := 
by
  sorry

end intersection_nonempty_condition_l74_74609


namespace negation_of_square_positivity_l74_74420

theorem negation_of_square_positivity :
  (¬ ∀ n : ℕ, n * n > 0) ↔ (∃ n : ℕ, n * n ≤ 0) :=
  sorry

end negation_of_square_positivity_l74_74420


namespace find_m_values_l74_74612

theorem find_m_values (α : Real) (m : Real) (h1 : α ∈ Set.Ioo π (3 * π / 2)) 
  (h2 : Real.sin α = (3 * m - 2) / (m + 3)) 
  (h3 : Real.cos α = (m - 5) / (m + 3)) : m = (10 / 9) ∨ m = 2 := by 
  sorry

end find_m_values_l74_74612


namespace length_of_crease_l74_74064

theorem length_of_crease (θ : ℝ) : 
  let B := 5
  let DM := 5 * (Real.tan θ)
  DM = 5 * (Real.tan θ) := 
by 
  sorry

end length_of_crease_l74_74064


namespace value_of_expression_l74_74305

theorem value_of_expression : (3 + 2) - (2 + 1) = 2 :=
by
  sorry

end value_of_expression_l74_74305


namespace find_y_l74_74589

noncomputable def x : ℝ := 3.3333333333333335

theorem find_y (y x: ℝ) (h1: x = 3.3333333333333335) (h2: x * 10 / y = x^2) :
  y = 3 :=
by
  sorry

end find_y_l74_74589


namespace relationship_xy_qz_l74_74054

theorem relationship_xy_qz
  (a c b d : ℝ)
  (x y q z : ℝ)
  (h1 : a^(2 * x) = c^(2 * q) ∧ c^(2 * q) = b^2)
  (h2 : c^(3 * y) = a^(3 * z) ∧ a^(3 * z) = d^2) :
  x * y = q * z :=
by
  sorry

end relationship_xy_qz_l74_74054


namespace num_pairs_eq_seven_l74_74427

theorem num_pairs_eq_seven :
  ∃ S : Finset (Nat × Nat), 
    (∀ (a b : Nat), (a, b) ∈ S ↔ (0 < a ∧ 0 < b ∧ a + b ≤ 100 ∧ (a + 1 / b) / (1 / a + b) = 13)) ∧
    S.card = 7 :=
sorry

end num_pairs_eq_seven_l74_74427


namespace sector_radius_cone_l74_74882

theorem sector_radius_cone {θ R r : ℝ} (sector_angle : θ = 120) (cone_base_radius : r = 2) :
  (R * θ / 360) * 2 * π = 2 * π * r → R = 6 :=
by
  intros h
  sorry

end sector_radius_cone_l74_74882


namespace solve_problem_l74_74690

theorem solve_problem (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  7^m - 3 * 2^n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) := sorry

end solve_problem_l74_74690


namespace probability_zero_after_2017_days_l74_74029

-- Define the people involved
inductive Person
| Lunasa | Merlin | Lyrica
deriving DecidableEq, Inhabited

open Person

-- Define the initial state with each person having their own distinct hat
def initial_state : Person → Person
| Lunasa => Lunasa
| Merlin => Merlin
| Lyrica => Lyrica

-- Define a function that represents switching hats between two people
def switch_hats (p1 p2 : Person) (state : Person → Person) : Person → Person :=
  λ p => if p = p1 then state p2 else if p = p2 then state p1 else state p

-- Define a function to represent the state after n days (iterations)
def iter_switch_hats (n : ℕ) : Person → Person :=
  sorry -- This would involve implementing the iterative random switching

-- Proposition: The probability that after 2017 days, every person has their own hat back is 0
theorem probability_zero_after_2017_days :
  iter_switch_hats 2017 = initial_state → false :=
by
  sorry

end probability_zero_after_2017_days_l74_74029


namespace common_root_poly_identity_l74_74884

theorem common_root_poly_identity
  (α p p' q q' : ℝ)
  (h1 : α^3 + p*α + q = 0)
  (h2 : α^3 + p'*α + q' = 0) : 
  (p * q' - q * p') * (p - p')^2 = (q - q')^3 := 
by
  sorry

end common_root_poly_identity_l74_74884


namespace smallest_number_is_D_l74_74302

-- Define the given numbers in Lean
def A := 25
def B := 111
def C := 16 + 4 + 2  -- since 10110_{(2)} equals 22 in base 10
def D := 16 + 2 + 1  -- since 10011_{(2)} equals 19 in base 10

-- The Lean statement for the proof problem
theorem smallest_number_is_D : min (min A B) (min C D) = D := by
  sorry

end smallest_number_is_D_l74_74302


namespace repeating_decimal_fraction_difference_l74_74256

theorem repeating_decimal_fraction_difference :
  ∀ (F : ℚ),
  F = 817 / 999 → (999 - 817 = 182) :=
by
  sorry

end repeating_decimal_fraction_difference_l74_74256


namespace valid_numbers_eq_l74_74412

-- Definition of the number representation
def is_valid_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999 ∧
  ∃ (a b c : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    x = 100 * a + 10 * b + c ∧
    x = a^3 + b^3 + c^3

-- The theorem to prove
theorem valid_numbers_eq : 
  {x : ℕ | is_valid_number x} = {153, 407} :=
by
  sorry

end valid_numbers_eq_l74_74412


namespace probability_scoring_less_than_8_l74_74594

theorem probability_scoring_less_than_8 
  (P10 P9 P8 : ℝ) 
  (hP10 : P10 = 0.3) 
  (hP9 : P9 = 0.3) 
  (hP8 : P8 = 0.2) : 
  1 - (P10 + P9 + P8) = 0.2 := 
by 
  sorry

end probability_scoring_less_than_8_l74_74594


namespace find_missing_number_l74_74095

theorem find_missing_number (x : ℤ) (h : 10010 - 12 * x * 2 = 9938) : x = 3 :=
by
  sorry

end find_missing_number_l74_74095


namespace power_mod_eight_l74_74620

theorem power_mod_eight (n : ℕ) : (3^101 + 5) % 8 = 0 :=
by
  sorry

end power_mod_eight_l74_74620


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l74_74693

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l74_74693


namespace cube_sum_div_by_9_implies_prod_div_by_3_l74_74542

theorem cube_sum_div_by_9_implies_prod_div_by_3 
  {a1 a2 a3 a4 a5 : ℤ} 
  (h : 9 ∣ a1^3 + a2^3 + a3^3 + a4^3 + a5^3) : 
  3 ∣ a1 * a2 * a3 * a4 * a5 := by
  sorry

end cube_sum_div_by_9_implies_prod_div_by_3_l74_74542


namespace smallest_nat_satisfying_conditions_l74_74404

theorem smallest_nat_satisfying_conditions : 
  ∃ x : ℕ, 
  (x % 4 = 2) ∧ 
  (x % 5 = 2) ∧ 
  (x % 6 = 2) ∧ 
  (x % 12 = 2) ∧ 
  (∀ y : ℕ, (y % 4 = 2) ∧ (y % 5 = 2) ∧ (y % 6 = 2) ∧ (y % 12 = 2) → x ≤ y) :=
  sorry

end smallest_nat_satisfying_conditions_l74_74404


namespace rope_length_l74_74655

theorem rope_length (x : ℝ) 
  (h : 10^2 + (x - 4)^2 = x^2) : 
  x = 14.5 :=
sorry

end rope_length_l74_74655


namespace bucket_capacity_l74_74839

theorem bucket_capacity (x : ℕ) (h₁ : 12 * x = 132 * 5) : x = 55 := by
  sorry

end bucket_capacity_l74_74839


namespace positional_relationship_l74_74521

theorem positional_relationship 
  (m n : ℝ) 
  (h_points_on_ellipse : (m^2 / 4) + (n^2 / 3) = 1)
  (h_relation : n^2 = 3 - (3/4) * m^2) : 
  (∃ x y : ℝ, (x^2 + y^2 = 1/3) ∧ (m * x + n * y + 1 = 0)) ∨ 
  (∀ x y : ℝ, (x^2 + y^2 = 1/3) → (m * x + n * y + 1 ≠ 0)) :=
sorry

end positional_relationship_l74_74521


namespace probability_more_than_70_l74_74566

-- Definitions based on problem conditions
def P_A : ℝ := 0.15
def P_B : ℝ := 0.45
def P_C : ℝ := 0.25

-- Theorem to state that the probability of scoring more than 70 points is 0.85
theorem probability_more_than_70 (hA : P_A = 0.15) (hB : P_B = 0.45) (hC : P_C = 0.25):
  P_A + P_B + P_C = 0.85 :=
by
  rw [hA, hB, hC]
  sorry

end probability_more_than_70_l74_74566


namespace connie_tickets_l74_74098

variable (T : ℕ)

theorem connie_tickets (h : T = T / 2 + 10 + 15) : T = 50 :=
by 
sorry

end connie_tickets_l74_74098


namespace find_n_after_folding_l74_74917

theorem find_n_after_folding (n : ℕ) (h : 2 ^ n = 128) : n = 7 := by
  sorry

end find_n_after_folding_l74_74917


namespace allison_rolls_greater_probability_l74_74744

theorem allison_rolls_greater_probability :
  let allison_roll : ℕ := 6
  let charlie_prob_less_6 := 5 / 6
  let mia_prob_rolls_3 := 4 / 6
  let combined_prob := charlie_prob_less_6 * (mia_prob_rolls_3)
  combined_prob = 5 / 9 := by
  sorry

end allison_rolls_greater_probability_l74_74744


namespace impossibility_of_sum_sixteen_l74_74792

open Nat

def max_roll_value : ℕ := 6
def sum_of_two_rolls (a b : ℕ) : ℕ := a + b

theorem impossibility_of_sum_sixteen :
  ∀ a b : ℕ, (1 ≤ a ∧ a ≤ max_roll_value) ∧ (1 ≤ b ∧ b ≤ max_roll_value) → sum_of_two_rolls a b ≠ 16 :=
by
  intros a b h
  sorry

end impossibility_of_sum_sixteen_l74_74792


namespace fraction_irreducible_gcd_2_power_l74_74266

-- Proof problem (a)
theorem fraction_irreducible (n : ℕ) : gcd (12 * n + 1) (30 * n + 2) = 1 :=
sorry

-- Proof problem (b)
theorem gcd_2_power (n m : ℕ) : gcd (2^100 - 1) (2^120 - 1) = 2^20 - 1 :=
sorry

end fraction_irreducible_gcd_2_power_l74_74266


namespace tangent_line_eq_l74_74749

/-- The equation of the tangent line to the curve y = 2x * tan x at the point x = π/4 is 
    (2 + π/2) * x - y - π^2/4 = 0. -/
theorem tangent_line_eq : ∀ x y : ℝ, 
  (y = 2 * x * Real.tan x) →
  (x = Real.pi / 4) →
  ((2 + Real.pi / 2) * x - y - Real.pi^2 / 4 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_eq_l74_74749


namespace six_hundred_billion_in_scientific_notation_l74_74946

theorem six_hundred_billion_in_scientific_notation (billion : ℕ) (h_billion : billion = 10^9) : 
  600 * billion = 6 * 10^11 :=
by
  rw [h_billion]
  sorry

end six_hundred_billion_in_scientific_notation_l74_74946


namespace sale_in_fifth_month_l74_74813

-- Define the sale amounts and average sale required.
def sale_first_month : ℕ := 7435
def sale_second_month : ℕ := 7920
def sale_third_month : ℕ := 7855
def sale_fourth_month : ℕ := 8230
def sale_sixth_month : ℕ := 6000
def average_sale_required : ℕ := 7500

-- State the theorem to determine the sale in the fifth month.
theorem sale_in_fifth_month
  (s1 s2 s3 s4 s6 avg : ℕ)
  (h1 : s1 = sale_first_month)
  (h2 : s2 = sale_second_month)
  (h3 : s3 = sale_third_month)
  (h4 : s4 = sale_fourth_month)
  (h6 : s6 = sale_sixth_month)
  (havg : avg = average_sale_required) :
  s1 + s2 + s3 + s4 + s6 + x = 6 * avg →
  x = 7560 :=
by
  sorry

end sale_in_fifth_month_l74_74813


namespace sum_of_squares_of_biking_jogging_swimming_rates_l74_74563

theorem sum_of_squares_of_biking_jogging_swimming_rates (b j s : ℕ) 
  (h1 : 2 * b + 3 * j + 4 * s = 74) 
  (h2 : 4 * b + 2 * j + 3 * s = 91) : 
  (b^2 + j^2 + s^2 = 314) :=
sorry

end sum_of_squares_of_biking_jogging_swimming_rates_l74_74563


namespace optionD_is_quadratic_l74_74383

variable (x : ℝ)

-- Original equation in Option D
def optionDOriginal := (x^2 + 2 * x = 2 * x^2 - 1)

-- Rearranged form of Option D's equation
def optionDRearranged := (-x^2 + 2 * x + 1 = 0)

theorem optionD_is_quadratic : optionDOriginal x → optionDRearranged x :=
by
  intro h
  -- The proof steps would go here, but we use sorry to skip it
  sorry

end optionD_is_quadratic_l74_74383


namespace sandy_change_correct_l74_74961

def football_cost : ℚ := 914 / 100
def baseball_cost : ℚ := 681 / 100
def payment : ℚ := 20

def total_cost : ℚ := football_cost + baseball_cost
def change_received : ℚ := payment - total_cost

theorem sandy_change_correct :
  change_received = 405 / 100 :=
by
  -- The proof should go here
  sorry

end sandy_change_correct_l74_74961


namespace city_phone_number_remainder_l74_74019

theorem city_phone_number_remainder :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧
  (312837 % n = 96) ∧ (310650 % n = 96) := sorry

end city_phone_number_remainder_l74_74019


namespace prime_remainder_30_l74_74193

theorem prime_remainder_30 (p : ℕ) (hp : Nat.Prime p) (hgt : p > 30) (hmod2 : p % 2 ≠ 0) 
(hmod3 : p % 3 ≠ 0) (hmod5 : p % 5 ≠ 0) : 
  ∃ (r : ℕ), r < 30 ∧ (p % 30 = r) ∧ (r = 1 ∨ Nat.Prime r) := 
by
  sorry

end prime_remainder_30_l74_74193


namespace projection_problem_l74_74105

noncomputable def vector_proj (w v : ℝ × ℝ) : ℝ × ℝ := sorry -- assume this definition

variables (v w : ℝ × ℝ)

-- Given condition
axiom proj_v : vector_proj w v = ⟨4, 3⟩

-- Proof Statement
theorem projection_problem :
  vector_proj w (7 • v + 2 • w) = ⟨28, 21⟩ + 2 • w :=
sorry

end projection_problem_l74_74105


namespace system1_solution_system2_solution_system3_solution_l74_74625

theorem system1_solution (x y : ℝ) : 
  (x = 3/2) → (y = 1/2) → (x + 3 * y = 3) ∧ (x - y = 1) :=
by intros; sorry

theorem system2_solution (x y : ℝ) : 
  (x = 0) → (y = 2/5) → ((x + 3 * y) / 2 = 3 / 5) ∧ (5 * (x - 2 * y) = -4) :=
by intros; sorry

theorem system3_solution (x y z : ℝ) : 
  (x = 1) → (y = 2) → (z = 3) → 
  (3 * x + 4 * y + z = 14) ∧ (x + 5 * y + 2 * z = 17) ∧ (2 * x + 2 * y - z = 3) :=
by intros; sorry

end system1_solution_system2_solution_system3_solution_l74_74625


namespace machines_finish_job_in_24_over_11_hours_l74_74575

theorem machines_finish_job_in_24_over_11_hours :
    let work_rate_A := 1 / 4
    let work_rate_B := 1 / 12
    let work_rate_C := 1 / 8
    let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
    (1 : ℝ) / combined_work_rate = 24 / 11 :=
by
  sorry

end machines_finish_job_in_24_over_11_hours_l74_74575


namespace original_amount_of_water_l74_74717

variable {W : ℝ} -- Assume W is a real number representing the original amount of water

theorem original_amount_of_water (h1 : 30 * 0.02 = 0.6) (h2 : 0.6 = 0.06 * W) : W = 10 :=
by
  sorry

end original_amount_of_water_l74_74717


namespace train_length_l74_74096

theorem train_length (L : ℝ) 
    (cross_bridge : ∀ (t_bridge : ℝ), t_bridge = 10 → L + 200 = t_bridge * (L / 5))
    (cross_lamp_post : ∀ (t_lamp_post : ℝ), t_lamp_post = 5 → L = t_lamp_post * (L / 5)) :
  L = 200 := 
by 
  -- sorry is used to skip the proof part
  sorry

end train_length_l74_74096


namespace total_red_stripes_l74_74555

theorem total_red_stripes 
  (flagA_stripes : ℕ := 30) 
  (flagB_stripes : ℕ := 45) 
  (flagC_stripes : ℕ := 60)
  (flagA_count : ℕ := 20) 
  (flagB_count : ℕ := 30) 
  (flagC_count : ℕ := 40)
  (flagA_red : ℕ := 15)
  (flagB_red : ℕ := 15)
  (flagC_red : ℕ := 14) : 
  300 + 450 + 560 = 1310 := 
by
  have flagA_red_stripes : 15 = 15 := by rfl
  have flagB_red_stripes : 15 = 15 := by rfl
  have flagC_red_stripes : 14 = 14 := by rfl
  have total_A_red_stripes : 15 * 20 = 300 := by norm_num
  have total_B_red_stripes : 15 * 30 = 450 := by norm_num
  have total_C_red_stripes : 14 * 40 = 560 := by norm_num
  exact add_assoc 300 450 560 ▸ rfl

end total_red_stripes_l74_74555


namespace tower_count_l74_74344

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binom (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def multinomialCoeff (n : Nat) (ks : List Nat) : Nat :=
  factorial n / List.foldr (fun k acc => acc * factorial k) 1 ks

theorem tower_count :
  let totalCubes := 9
  let usedCubes := 8
  let redCubes := 2
  let blueCubes := 3
  let greenCubes := 4
  multinomialCoeff totalCubes [redCubes, blueCubes, greenCubes] = 1260 :=
by
  sorry

end tower_count_l74_74344


namespace value_of_m_l74_74276

theorem value_of_m (m : ℝ) : (∀ x : ℝ, x^2 + m * x + 9 = (x + 3)^2) → m = 6 :=
by
  intro h
  sorry

end value_of_m_l74_74276


namespace more_larger_boxes_l74_74491

theorem more_larger_boxes (S L : ℕ) 
  (h1 : 12 * S + 16 * L = 480)
  (h2 : S + L = 32)
  (h3 : L > S) : L - S = 16 := 
sorry

end more_larger_boxes_l74_74491


namespace gcd_4536_8721_l74_74652

theorem gcd_4536_8721 : Nat.gcd 4536 8721 = 3 := by
  sorry

end gcd_4536_8721_l74_74652


namespace alpha_beta_purchase_ways_l74_74577

-- Definitions for the problem
def number_of_flavors : ℕ := 7
def number_of_milk_types : ℕ := 4
def total_products_to_purchase : ℕ := 5

-- Conditions
def alpha_max_per_flavor : ℕ := 2
def beta_only_cookies (x : ℕ) : Prop := x = number_of_flavors

-- Main theorem (statement only)
theorem alpha_beta_purchase_ways : 
  ∃ (ways : ℕ), 
    ways = 17922 ∧
    ∀ (alpha beta : ℕ), 
      alpha + beta = total_products_to_purchase →
      (alpha <= alpha_max_per_flavor * number_of_flavors ∧ beta <= total_products_to_purchase - alpha) :=
sorry

end alpha_beta_purchase_ways_l74_74577


namespace expression_always_positive_l74_74953

theorem expression_always_positive (x : ℝ) : x^2 + |x| + 1 > 0 :=
by 
  sorry

end expression_always_positive_l74_74953


namespace circumscribed_triangle_area_relation_l74_74705

theorem circumscribed_triangle_area_relation
  (a b c D E F : ℝ)
  (h₁ : a = 18) (h₂ : b = 24) (h₃ : c = 30)
  (triangle_right : a^2 + b^2 = c^2)
  (triangle_area : (1/2) * a * b = 216)
  (circle_area : π * (c / 2)^2 = 225 * π)
  (non_triangle_areas : D + E + 216 = F) :
  D + E + 216 = F :=
by
  sorry

end circumscribed_triangle_area_relation_l74_74705


namespace mass_percentage_of_Cl_in_NaOCl_l74_74220

theorem mass_percentage_of_Cl_in_NaOCl :
  let Na_mass := 22.99
  let O_mass := 16.00
  let Cl_mass := 35.45
  let NaOCl_mass := Na_mass + O_mass + Cl_mass
  100 * (Cl_mass / NaOCl_mass) = 47.6 := 
by
  let Na_mass := 22.99
  let O_mass := 16.00
  let Cl_mass := 35.45
  let NaOCl_mass := Na_mass + O_mass + Cl_mass
  sorry

end mass_percentage_of_Cl_in_NaOCl_l74_74220


namespace evaluate_expression_l74_74233

theorem evaluate_expression (x : ℝ) : x * (x * (x * (x - 3) - 5) + 12) + 2 = x^4 - 3 * x^3 - 5 * x^2 + 12 * x + 2 :=
by
  sorry

end evaluate_expression_l74_74233


namespace daily_shampoo_usage_l74_74292

theorem daily_shampoo_usage
  (S : ℝ)
  (h1 : ∀ t : ℝ, t = 14 → 14 * S + 14 * (S / 2) = 21) :
  S = 1 := by
  sorry

end daily_shampoo_usage_l74_74292


namespace denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry_l74_74888

variable (DenyMotion : Prop) (AcknowledgeStillness : Prop) (LeadsToRelativism : Prop)
variable (LeadsToSophistry : Prop)

theorem denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry
  (h1 : DenyMotion)
  (h2 : AcknowledgeStillness)
  (h3 : DenyMotion ∧ AcknowledgeStillness → ¬LeadsToRelativism)
  (h4 : DenyMotion ∧ AcknowledgeStillness → ¬LeadsToSophistry):
  ¬ (DenyMotion ∧ AcknowledgeStillness → LeadsToRelativism ∧ LeadsToSophistry) :=
by sorry

end denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry_l74_74888


namespace min_y_value_l74_74818

noncomputable def y (x : ℝ) : ℝ := x^2 + 16 * x + 20

theorem min_y_value : ∀ (x : ℝ), x ≥ -3 → y x ≥ -19 :=
by
  intro x hx
  sorry

end min_y_value_l74_74818


namespace common_ratio_arith_geo_sequence_l74_74152

theorem common_ratio_arith_geo_sequence (a : ℕ → ℝ) (d : ℝ) (q : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geo : (a 1 + 2) * q = a 5 + 5) 
  (h_geo' : (a 5 + 5) * q = a 9 + 8) :
  q = 1 :=
by
  sorry

end common_ratio_arith_geo_sequence_l74_74152


namespace part_one_part_two_l74_74009

def f (x : ℝ) : ℝ := |x| + |x - 1|

theorem part_one (m : ℝ) (h : ∀ x, f x ≥ |m - 1|) : m ≤ 2 := by
  sorry

theorem part_two (a b : ℝ) (M : ℝ) (ha : 0 < a) (hb : 0 < b) (hM : a^2 + b^2 = M) (hM_value : M = 2) : a + b ≥ 2 * a * b := by
  sorry

end part_one_part_two_l74_74009


namespace perfect_square_and_solutions_exist_l74_74573

theorem perfect_square_and_solutions_exist (m n t : ℕ)
  (h1 : t > 0) (h2 : m > 0) (h3 : n > 0)
  (h4 : t * (m^2 - n^2) + m - n^2 - n = 0) :
  ∃ (k : ℕ), m - n = k * k ∧ (∀ t > 0, ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (t * (m^2 - n^2) + m - n^2 - n = 0)) :=
by
  sorry

end perfect_square_and_solutions_exist_l74_74573


namespace inequality_proof_l74_74912

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x^2 + y^2 + z^2 = 2 * (x * y + y * z + z * x)) :
  (x + y + z) / 3 ≥ (2 * x * y * z)^(1/3 : ℝ) :=
by
  sorry

end inequality_proof_l74_74912


namespace total_number_of_students_l74_74789

theorem total_number_of_students 
    (group1 : Nat) (group2 : Nat) (group3 : Nat) (group4 : Nat) 
    (h1 : group1 = 5) (h2 : group2 = 8) (h3 : group3 = 7) (h4 : group4 = 4) : 
    group1 + group2 + group3 + group4 = 24 := 
by
  sorry

end total_number_of_students_l74_74789


namespace find_other_person_weight_l74_74458

theorem find_other_person_weight
    (initial_avg_weight : ℕ)
    (final_avg_weight : ℕ)
    (initial_group_size : ℕ)
    (new_person_weight : ℕ)
    (final_group_size : ℕ)
    (initial_total_weight : ℕ)
    (final_total_weight : ℕ)
    (new_total_weight : ℕ)
    (other_person_weight : ℕ) :
  initial_avg_weight = 48 →
  final_avg_weight = 51 →
  initial_group_size = 23 →
  final_group_size = 25 →
  new_person_weight = 93 →
  initial_total_weight = initial_group_size * initial_avg_weight →
  final_total_weight = final_group_size * final_avg_weight →
  new_total_weight = initial_total_weight + new_person_weight + other_person_weight →
  final_total_weight = new_total_weight →
  other_person_weight = 78 :=
by
  sorry

end find_other_person_weight_l74_74458


namespace S_11_l74_74534

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
-- Define that {a_n} is an arithmetic sequence.
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) := ∀ n, a (n + 1) = a n + d

-- Sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) := ∀ n, S n = n * (a 1 + a n) / 2

-- Given condition: a_5 + a_7 = 14
def sum_condition (a : ℕ → ℕ) := a 5 + a 7 = 14

-- Prove S_{11} = 77
theorem S_11 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (d : ℕ)
  (h1 : arithmetic_sequence a d)
  (h2 : sum_arithmetic_sequence S a)
  (h3 : sum_condition a) :
  S 11 = 77 := by
  -- The proof steps would follow here.
  sorry

end S_11_l74_74534


namespace expression_equals_384_l74_74528

noncomputable def problem_expression : ℤ :=
  2021^4 - 4 * 2023^4 + 6 * 2025^4 - 4 * 2027^4 + 2029^4

theorem expression_equals_384 : problem_expression = 384 := by
  sorry

end expression_equals_384_l74_74528


namespace complement_of_A_l74_74541

/-
Given:
1. Universal set U = {0, 1, 2, 3, 4}
2. Set A = {1, 2}

Prove:
C_U A = {0, 3, 4}
-/

section
  variable (U : Set ℕ) (A : Set ℕ)
  variable (hU : U = {0, 1, 2, 3, 4})
  variable (hA : A = {1, 2})

  theorem complement_of_A (C_UA : Set ℕ) (hCUA : C_UA = {0, 3, 4}) : 
    {x ∈ U | x ∉ A} = C_UA :=
  by
    sorry
end

end complement_of_A_l74_74541


namespace calculate_first_year_sample_l74_74562

noncomputable def stratified_sampling : ℕ :=
  let total_sample_size := 300
  let first_grade_ratio := 4
  let second_grade_ratio := 5
  let third_grade_ratio := 5
  let fourth_grade_ratio := 6
  let total_ratio := first_grade_ratio + second_grade_ratio + third_grade_ratio + fourth_grade_ratio
  let first_grade_proportion := first_grade_ratio / total_ratio
  300 * first_grade_proportion

theorem calculate_first_year_sample :
  stratified_sampling = 60 :=
by sorry

end calculate_first_year_sample_l74_74562


namespace ratio_equivalence_l74_74740

theorem ratio_equivalence (x : ℕ) (h1 : 3 / 12 = x / 16) : x = 4 :=
by sorry

end ratio_equivalence_l74_74740


namespace solution_set_m5_range_m_sufficient_condition_l74_74988

theorem solution_set_m5 (x : ℝ) : 
  (|x + 1| + |x - 2| > 5) ↔ (x < -2 ∨ x > 3) := 
sorry

theorem range_m_sufficient_condition (x m : ℝ) (h : ∀ x : ℝ, |x + 1| + |x - 2| - m ≥ 2) : 
  m ≤ 1 := 
sorry

end solution_set_m5_range_m_sufficient_condition_l74_74988


namespace Ned_earning_money_l74_74747

def total_games : Nat := 15
def non_working_games : Nat := 6
def price_per_game : Nat := 7
def working_games : Nat := total_games - non_working_games
def total_money : Nat := working_games * price_per_game

theorem Ned_earning_money : total_money = 63 := by
  sorry

end Ned_earning_money_l74_74747


namespace range_m_l74_74467

namespace MathProof

noncomputable def f (x m : ℝ) : ℝ := x^3 - 3 * x + 2 + m

theorem range_m
  (m : ℝ)
  (h : m > 0)
  (a b c : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 2)
  (hb : 0 ≤ b ∧ b ≤ 2)
  (hc : 0 ≤ c ∧ c ≤ 2)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_triangle : f a m ^ 2 + f b m ^ 2 = f c m ^ 2 ∨
                f a m ^ 2 + f c m ^ 2 = f b m ^ 2 ∨
                f b m ^ 2 + f c m ^ 2 = f a m ^ 2) :
  0 < m ∧ m < 3 + 4 * Real.sqrt 2 :=
by
  sorry

end MathProof

end range_m_l74_74467


namespace solution_set_f_x_minus_1_lt_0_l74_74151

noncomputable def f (x : ℝ) : ℝ :=
if h : x ≥ 0 then x - 1 else -x - 1

theorem solution_set_f_x_minus_1_lt_0 :
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_f_x_minus_1_lt_0_l74_74151


namespace value_of_N_l74_74773

theorem value_of_N (a b c N : ℚ) (h1 : a + b + c = 120) (h2 : a - 10 = N) (h3 : 10 * b = N) (h4 : c - 10 = N) : N = 1100 / 21 := 
sorry

end value_of_N_l74_74773


namespace cat_clothing_probability_l74_74519

-- Define the conditions as Lean definitions
def n_items : ℕ := 3
def total_legs : ℕ := 4
def favorable_outcomes_per_leg : ℕ := 1
def possible_outcomes_per_leg : ℕ := (n_items.factorial : ℕ)
def probability_per_leg : ℚ := favorable_outcomes_per_leg / possible_outcomes_per_leg

-- Theorem statement to show the combined probability for all legs
theorem cat_clothing_probability
    (n_items_eq : n_items = 3)
    (total_legs_eq : total_legs = 4)
    (fact_n_items : (n_items.factorial) = 6)
    (prob_leg_eq : probability_per_leg = 1 / 6) :
    (probability_per_leg ^ total_legs = 1 / 1296) := by
    sorry

end cat_clothing_probability_l74_74519


namespace contrapositive_proof_l74_74407

theorem contrapositive_proof (a b : ℕ) : (a = 1 ∧ b = 2) → (a + b = 3) :=
by {
  sorry
}

end contrapositive_proof_l74_74407


namespace smallest_add_to_2002_l74_74834

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def next_palindrome_after (n : ℕ) : ℕ :=
  -- a placeholder function for the next palindrome calculation
  -- implementation logic is skipped
  2112

def smallest_add_to_palindrome (n target : ℕ) : ℕ :=
  target - n

theorem smallest_add_to_2002 :
  let target := next_palindrome_after 2002
  ∃ k, is_palindrome (2002 + k) ∧ (2002 < 2002 + k) ∧ target = 2002 + k ∧ k = 110 := 
by
  use 110
  sorry

end smallest_add_to_2002_l74_74834


namespace rock_paper_scissors_score_divisible_by_3_l74_74403

theorem rock_paper_scissors_score_divisible_by_3 
  (R : ℕ) 
  (rock_shown : ℕ) 
  (scissors_shown : ℕ) 
  (paper_shown : ℕ)
  (points : ℕ)
  (h_equal_shows : 3 * ((rock_shown + scissors_shown + paper_shown) / 3) = rock_shown + scissors_shown + paper_shown)
  (h_points_awarded : ∀ (r s p : ℕ), r + s + p = 3 → (r = 2 ∧ s = 1 ∧ p = 0) ∨ (r = 0 ∧ s = 2 ∧ p = 1) ∨ (r = 1 ∧ s = 0 ∧ p = 2) → points % 3 = 0) :
  points % 3 = 0 := 
sorry

end rock_paper_scissors_score_divisible_by_3_l74_74403


namespace set_inter_complement_l74_74340

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem set_inter_complement :
  A ∩ (U \ B) = {1, 3} :=
by
  sorry

end set_inter_complement_l74_74340


namespace solve_equation_l74_74925

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) (h1 : x ≠ 1) : 
  x = 5 / 3 :=
sorry

end solve_equation_l74_74925


namespace k_not_possible_l74_74434

theorem k_not_possible (S : ℕ → ℚ) (a b : ℕ → ℚ) (n k : ℕ) (k_gt_2 : k > 2) :
  (S n = (n^2 + n) / 2) →
  (a n = S n - S (n - 1)) →
  (b n = 1 / a n) →
  (2 * b (n + 2) = b n + b (n + k)) →
  k ≠ 4 ∧ k ≠ 10 :=
by
  -- Proof goes here (skipped)
  sorry

end k_not_possible_l74_74434


namespace hiking_committee_selection_l74_74935

def comb (n k : ℕ) : ℕ := n.choose k

theorem hiking_committee_selection :
  comb 10 3 = 120 :=
by
  sorry

end hiking_committee_selection_l74_74935


namespace prob_selected_first_eq_third_l74_74361

noncomputable def total_students_first := 800
noncomputable def total_students_second := 600
noncomputable def total_students_third := 500
noncomputable def selected_students_third := 25
noncomputable def prob_selected_third := selected_students_third / total_students_third

theorem prob_selected_first_eq_third :
  (selected_students_third / total_students_third = 1 / 20) →
  (prob_selected_third = 1 / 20) :=
by
  intros h
  sorry

end prob_selected_first_eq_third_l74_74361


namespace no_convex_27gon_with_distinct_integer_angles_l74_74346

noncomputable def sum_of_interior_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

def is_convex (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  ∀ i, angles i < 180

def all_distinct (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  ∀ i j, i ≠ j → angles i ≠ angles j

def sum_is_correct (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  Finset.sum (Finset.univ : Finset (Fin n)) angles = sum_of_interior_angles n

theorem no_convex_27gon_with_distinct_integer_angles :
  ¬ ∃ (angles : Fin 27 → ℕ), is_convex 27 angles ∧ all_distinct 27 angles ∧ sum_is_correct 27 angles :=
by
  sorry

end no_convex_27gon_with_distinct_integer_angles_l74_74346


namespace two_card_draw_probability_l74_74809

open ProbabilityTheory

def card_values (card : ℕ) : ℕ :=
  if card = 1 ∨ card = 11 ∨ card = 12 ∨ card = 13 then 10 else card

def deck_size := 52

def total_prob : ℚ :=
  let cards := (1, deck_size)
  let case_1 := (card_values 6 * card_values 9 / (deck_size * (deck_size - 1))) + 
                (card_values 7 * card_values 8 / (deck_size * (deck_size - 1)))
  let case_2 := (3 * 4 / (deck_size * (deck_size - 1))) + 
                (4 * 3 / (deck_size * (deck_size - 1)))
  case_1 + case_2

theorem two_card_draw_probability :
  total_prob = 16 / 331 :=
by
  sorry

end two_card_draw_probability_l74_74809


namespace number_of_photographs_is_twice_the_number_of_paintings_l74_74831

theorem number_of_photographs_is_twice_the_number_of_paintings (P Q : ℕ) :
  (Q * (Q - 1) * P) = 2 * (P * (Q * (Q - 1)) / 2) := by
  sorry

end number_of_photographs_is_twice_the_number_of_paintings_l74_74831


namespace sqrt_of_sixteen_l74_74461

theorem sqrt_of_sixteen : ∃ x : ℤ, x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end sqrt_of_sixteen_l74_74461


namespace hilt_miles_traveled_l74_74596

theorem hilt_miles_traveled (initial_miles lunch_additional_miles : Real) (h_initial : initial_miles = 212.3) (h_lunch : lunch_additional_miles = 372.0) :
  initial_miles + lunch_additional_miles = 584.3 :=
by
  sorry

end hilt_miles_traveled_l74_74596


namespace speed_in_still_water_l74_74085

theorem speed_in_still_water (upstream downstream : ℝ) (h_upstream : upstream = 37) (h_downstream : downstream = 53) : 
  (upstream + downstream) / 2 = 45 := 
by
  sorry

end speed_in_still_water_l74_74085


namespace real_inequality_l74_74246

theorem real_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + a * c + b * c := by
  sorry

end real_inequality_l74_74246


namespace arthur_reading_pages_l74_74527

theorem arthur_reading_pages :
  let total_goal : ℕ := 800
  let pages_read_from_500_book : ℕ := 500 * 80 / 100 -- 80% of 500 pages
  let pages_read_from_1000_book : ℕ := 1000 / 5 -- 1/5 of 1000 pages
  let total_pages_read : ℕ := pages_read_from_500_book + pages_read_from_1000_book
  let remaining_pages : ℕ := total_goal - total_pages_read
  remaining_pages = 200 :=
by
  -- placeholder for actual proof
  sorry

end arthur_reading_pages_l74_74527


namespace range_of_a_l74_74363

theorem range_of_a (x a : ℝ) (h₁ : x > 1) (h₂ : a ≤ x + 1 / (x - 1)) : 
  a < 3 :=
sorry

end range_of_a_l74_74363


namespace calculate_surface_area_of_modified_cube_l74_74739

-- Definitions of the conditions
def edge_length_of_cube : ℕ := 5
def side_length_of_hole : ℕ := 2

-- The main theorem statement to be proven
theorem calculate_surface_area_of_modified_cube :
  let original_surface_area := 6 * (edge_length_of_cube * edge_length_of_cube)
  let area_removed_by_holes := 6 * (side_length_of_hole * side_length_of_hole)
  let area_exposed_by_holes := 6 * 6 * (side_length_of_hole * side_length_of_hole)
  original_surface_area - area_removed_by_holes + area_exposed_by_holes = 270 :=
by
  sorry

end calculate_surface_area_of_modified_cube_l74_74739


namespace smallest_m_for_integral_solutions_l74_74102

theorem smallest_m_for_integral_solutions :
  ∃ (m : ℕ), (∀ (x : ℤ), (12 * x^2 - m * x + 504 = 0 → ∃ (p q : ℤ), p + q = m / 12 ∧ p * q = 42)) ∧
  m = 156 := by
sorry

end smallest_m_for_integral_solutions_l74_74102


namespace find_original_number_l74_74038

theorem find_original_number (x y : ℕ) (h1 : x + y = 8) (h2 : 10 * y + x = 10 * x + y + 18) : 10 * x + y = 35 := 
sorry

end find_original_number_l74_74038


namespace line_symmetric_to_itself_l74_74867

theorem line_symmetric_to_itself :
  ∀ x y : ℝ, y = 3 * x + 3 ↔ ∃ (m b : ℝ), y = m * x + b ∧ m = 3 ∧ b = 3 :=
by
  sorry

end line_symmetric_to_itself_l74_74867


namespace proof_f_values_l74_74330

def f (x : ℤ) : ℤ :=
  if x < 0 then
    2 * x + 7
  else
    x^2 - 2

theorem proof_f_values :
  f (-2) = 3 ∧ f (3) = 7 :=
by
  sorry

end proof_f_values_l74_74330


namespace complex_number_expression_l74_74738

noncomputable def compute_expression (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1)

theorem complex_number_expression (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  compute_expression r h1 h2 = 5 :=
sorry

end complex_number_expression_l74_74738


namespace expected_disease_count_l74_74861

/-- Define the probability of an American suffering from the disease. -/
def probability_of_disease := 1 / 3

/-- Define the sample size of Americans surveyed. -/
def sample_size := 450

/-- Calculate the expected number of individuals suffering from the disease in the sample. -/
noncomputable def expected_number := probability_of_disease * sample_size

/-- State the theorem: the expected number of individuals suffering from the disease is 150. -/
theorem expected_disease_count : expected_number = 150 :=
by
  -- Proof is required but skipped using sorry.
  sorry

end expected_disease_count_l74_74861


namespace video_total_votes_l74_74167

theorem video_total_votes (x : ℕ) (L D : ℕ)
  (h1 : L + D = x)
  (h2 : L - D = 130)
  (h3 : 70 * x = 100 * L) :
  x = 325 :=
by
  sorry

end video_total_votes_l74_74167


namespace base_conversion_min_sum_l74_74648

theorem base_conversion_min_sum (c d : ℕ) (h : 5 * c + 8 = 8 * d + 5) : c + d = 15 := by
  sorry

end base_conversion_min_sum_l74_74648


namespace molly_gift_cost_l74_74876

noncomputable def cost_per_package : ℕ := 5
noncomputable def num_parents : ℕ := 2
noncomputable def num_brothers : ℕ := 3
noncomputable def num_sisters_in_law : ℕ := num_brothers -- each brother is married
noncomputable def num_children_per_brother : ℕ := 2
noncomputable def num_nieces_nephews : ℕ := num_brothers * num_children_per_brother
noncomputable def total_relatives : ℕ := num_parents + num_brothers + num_sisters_in_law + num_nieces_nephews

theorem molly_gift_cost : (total_relatives * cost_per_package) = 70 := by
  sorry

end molly_gift_cost_l74_74876


namespace price_reduction_percentage_price_increase_amount_l74_74463

theorem price_reduction_percentage (x : ℝ) (hx : 50 * (1 - x)^2 = 32) : x = 0.2 := 
sorry

theorem price_increase_amount (y : ℝ) 
  (hy1 : 0 < y ∧ y ≤ 8) 
  (hy2 : 6000 = (10 + y) * (500 - 20 * y)) : y = 5 := 
sorry

end price_reduction_percentage_price_increase_amount_l74_74463


namespace number_of_knights_l74_74455

def traveler := Type
def is_knight (t : traveler) : Prop := sorry
def is_liar (t : traveler) : Prop := sorry

axiom total_travelers : Finset traveler
axiom vasily : traveler
axiom  h_total : total_travelers.card = 16

axiom kn_lie (t : traveler) : is_knight t ∨ is_liar t

axiom vasily_liar : is_liar vasily
axiom contradictory_statements_in_room (rooms: Finset (Finset traveler)):
  (∀ room ∈ rooms, ∃ t ∈ room, (is_liar t ∧ is_knight t))
  ∧
  (∀ room ∈ rooms, ∃ t ∈ room, (is_knight t ∧ is_liar t))

theorem number_of_knights : 
  ∃ k, k = 9 ∧ (∃ l, l = 7 ∧ ∀ t ∈ total_travelers, (is_knight t ∨ is_liar t)) :=
sorry

end number_of_knights_l74_74455


namespace cylinder_lateral_area_l74_74647

-- Define the cylindrical lateral area calculation
noncomputable def lateral_area_of_cylinder (d h : ℝ) : ℝ := (2 * Real.pi * (d / 2)) * h

-- The statement of the problem in Lean 4.
theorem cylinder_lateral_area : lateral_area_of_cylinder 4 4 = 16 * Real.pi := by
  sorry

end cylinder_lateral_area_l74_74647


namespace y_relationship_range_of_x_l74_74783

-- Definitions based on conditions
variable (x : ℝ) (y : ℝ)

-- Condition: Perimeter of the isosceles triangle is 6 cm
def perimeter_is_6 (x : ℝ) (y : ℝ) : Prop :=
  2 * x + y = 6

-- Condition: Function relationship of y in terms of x
def y_function (x : ℝ) : ℝ :=
  6 - 2 * x

-- Prove the functional relationship y = 6 - 2x
theorem y_relationship (x : ℝ) : y = y_function x ↔ perimeter_is_6 x y := by
  sorry

-- Prove the range of values for x
theorem range_of_x (x : ℝ) : 3 / 2 < x ∧ x < 3 ↔ (0 < y_function x ∧ perimeter_is_6 x (y_function x)) := by
  sorry

end y_relationship_range_of_x_l74_74783


namespace hash_difference_l74_74956

def hash (x y : ℕ) : ℤ := x * y - 3 * x + y

theorem hash_difference : (hash 8 5) - (hash 5 8) = -12 := by
  sorry

end hash_difference_l74_74956


namespace james_jump_height_is_16_l74_74641

-- Define given conditions
def mark_jump_height : ℕ := 6
def lisa_jump_height : ℕ := 2 * mark_jump_height
def jacob_jump_height : ℕ := 2 * lisa_jump_height
def james_jump_height : ℕ := (2 * jacob_jump_height) / 3

-- Problem Statement to prove
theorem james_jump_height_is_16 : james_jump_height = 16 :=
by
  sorry

end james_jump_height_is_16_l74_74641


namespace TripleApplicationOfF_l74_74175

def f (N : ℝ) : ℝ := 0.7 * N + 2

theorem TripleApplicationOfF :
  f (f (f 40)) = 18.1 :=
  sorry

end TripleApplicationOfF_l74_74175


namespace currency_notes_total_l74_74553

theorem currency_notes_total (num_50_notes total_amount remaining_amount num_100_notes : ℕ) 
  (h1 : remaining_amount = total_amount - (num_50_notes * 50))
  (h2 : num_50_notes = 3500 / 50)
  (h3 : total_amount = 5000)
  (h4 : remaining_amount = 1500)
  (h5 : num_100_notes = remaining_amount / 100) : 
  num_50_notes + num_100_notes = 85 :=
by sorry

end currency_notes_total_l74_74553


namespace same_graphs_at_x_eq_1_l74_74262

theorem same_graphs_at_x_eq_1 :
  let y1 := 2 - 1
  let y2 := (1^3 - 1) / (1 - 1)
  let y3 := (1^3 - 1) / (1 - 1)
  y2 = 3 ∧ y3 = 3 ∧ y1 ≠ y2 := 
by
  let y1 := 2 - 1
  let y2 := (1^3 - 1) / (1 - 1)
  let y3 := (1^3 - 1) / (1 - 1)
  sorry

end same_graphs_at_x_eq_1_l74_74262


namespace distance_from_origin_to_line_l74_74913

theorem distance_from_origin_to_line : 
  let a := 1
  let b := 2
  let c := -5
  let x0 := 0
  let y0 := 0
  let distance := (|a * x0 + b * y0 + c|) / (Real.sqrt (a^2 + b^2))
  distance = Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_line_l74_74913


namespace two_digit_numbers_condition_l74_74464

theorem two_digit_numbers_condition (a b : ℕ) (h1 : a ≠ 0) (h2 : 1 ≤ a ∧ a ≤ 9) (h3 : 0 ≤ b ∧ b ≤ 9) :
  (a + 1) * (b + 1) = 10 * a + b + 1 ↔ b = 9 := 
sorry

end two_digit_numbers_condition_l74_74464


namespace solve_system_l74_74684

theorem solve_system :
  ∃ x y : ℝ, (x + 2*y = 1 ∧ 3*x - 2*y = 7) → (x = 2 ∧ y = -1/2) :=
by
  sorry

end solve_system_l74_74684


namespace find_room_width_l74_74776

theorem find_room_width
  (length : ℝ)
  (cost_per_sqm : ℝ)
  (total_cost : ℝ)
  (h_length : length = 10)
  (h_cost_per_sqm : cost_per_sqm = 900)
  (h_total_cost : total_cost = 42750) :
  ∃ width : ℝ, width = 4.75 :=
by
  sorry

end find_room_width_l74_74776


namespace chef_cooked_additional_wings_l74_74742

def total_chicken_wings_needed (friends : ℕ) (wings_per_friend : ℕ) : ℕ :=
  friends * wings_per_friend

def additional_chicken_wings (total_needed : ℕ) (already_cooked : ℕ) : ℕ :=
  total_needed - already_cooked

theorem chef_cooked_additional_wings :
  let friends := 4
  let wings_per_friend := 4
  let already_cooked := 9
  additional_chicken_wings (total_chicken_wings_needed friends wings_per_friend) already_cooked = 7 := by
  sorry

end chef_cooked_additional_wings_l74_74742


namespace valid_votes_l74_74722

theorem valid_votes (V : ℝ) 
  (h1 : 0.70 * V - 0.30 * V = 176): V = 440 :=
  sorry

end valid_votes_l74_74722


namespace triangle_angle_and_area_l74_74117

theorem triangle_angle_and_area (a b c A B C : ℝ)
  (h₁ : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A))
  (h₂ : 0 < C ∧ C < Real.pi)
  (h₃ : c = 2 * Real.sqrt 3) :
  C = Real.pi / 3 ∧ 0 ≤ (1 / 2) * a * b * Real.sin C ∧ (1 / 2) * a * b * Real.sin C ≤ 3 * Real.sqrt 3 :=
by
  sorry

end triangle_angle_and_area_l74_74117


namespace four_digit_integer_transformation_l74_74857

theorem four_digit_integer_transformation (a b c d n : ℕ) (A : ℕ)
  (hA : A = 1000 * a + 100 * b + 10 * c + d)
  (ha : a + 2 < 10)
  (hc : c + 2 < 10)
  (hb : b ≥ 2)
  (hd : d ≥ 2)
  (hA4 : 1000 ≤ A ∧ A < 10000) :
  (1000 * (a + n) + 100 * (b - n) + 10 * (c + n) + (d - n)) = n * A → n = 2 → A = 1818 :=
by sorry

end four_digit_integer_transformation_l74_74857


namespace termite_ridden_fraction_l74_74469

theorem termite_ridden_fraction (T : ℝ) 
    (h1 : 5/8 * T > 0)
    (h2 : 3/8 * T = 0.125) : T = 1/8 :=
by
  sorry

end termite_ridden_fraction_l74_74469


namespace hexagon_shaded_area_correct_l74_74466

theorem hexagon_shaded_area_correct :
  let side_length := 3
  let semicircle_radius := side_length / 2
  let central_circle_radius := 1
  let hexagon_area := (3 * Real.sqrt 3 / 2) * side_length ^ 2
  let semicircle_area := (π * (semicircle_radius ^ 2)) / 2
  let total_semicircle_area := 6 * semicircle_area
  let central_circle_area := π * (central_circle_radius ^ 2)
  let shaded_area := hexagon_area - (total_semicircle_area + central_circle_area)
  shaded_area = 13.5 * Real.sqrt 3 - 7.75 * π := by
  sorry

end hexagon_shaded_area_correct_l74_74466


namespace jane_spent_75_days_reading_l74_74826

def pages : ℕ := 500
def speed_first_half : ℕ := 10
def speed_second_half : ℕ := 5

def book_reading_days (p s1 s2 : ℕ) : ℕ :=
  let half_pages := p / 2
  let days_first_half := half_pages / s1
  let days_second_half := half_pages / s2
  days_first_half + days_second_half

theorem jane_spent_75_days_reading :
  book_reading_days pages speed_first_half speed_second_half = 75 :=
by
  sorry

end jane_spent_75_days_reading_l74_74826


namespace intersection_A_B_l74_74114

def A : Set ℝ := { x | (x + 1) / (x - 1) ≤ 0 }
def B : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l74_74114


namespace floor_ceil_sum_l74_74180

theorem floor_ceil_sum (x : ℝ) (h : Int.floor x + Int.ceil x = 7) : x ∈ { x : ℝ | 3 < x ∧ x < 4 } ∪ {3.5} :=
sorry

end floor_ceil_sum_l74_74180


namespace find_n_l74_74993

-- Definitions based on conditions
def a := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7
def b (n : ℕ) := 2 * n

-- Theorem stating the problem
theorem find_n (n : ℕ) (h : a^2 - (b n)^2 = 0) : n = 10 :=
by sorry

end find_n_l74_74993


namespace problem_1_problem_2_l74_74890

open Real

theorem problem_1 : sqrt 3 * cos (π / 12) - sin (π / 12) = sqrt 2 := 
sorry

theorem problem_2 : ∀ θ : ℝ, sqrt 3 * cos θ - sin θ ≤ 2 := 
sorry

end problem_1_problem_2_l74_74890


namespace sum_of_squares_eq_power_l74_74869

theorem sum_of_squares_eq_power (n : ℕ) : ∃ x y z : ℕ, x^2 + y^2 = z^n :=
sorry

end sum_of_squares_eq_power_l74_74869


namespace pencils_given_out_l74_74646

theorem pencils_given_out
  (num_children : ℕ)
  (pencils_per_student : ℕ)
  (dozen : ℕ)
  (children : num_children = 46)
  (dozen_def : dozen = 12)
  (pencils_def : pencils_per_student = 4 * dozen) :
  num_children * pencils_per_student = 2208 :=
by {
  sorry
}

end pencils_given_out_l74_74646


namespace fraction_absent_l74_74664

theorem fraction_absent (p : ℕ) (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 1) (h2 : p * 1 = (1 - x) * p * 1.5) : x = 1 / 3 :=
by
  sorry

end fraction_absent_l74_74664


namespace find_m_l74_74321

noncomputable def first_series_sum : ℝ := 
  let a1 : ℝ := 18
  let a2 : ℝ := 6
  let r : ℝ := a2 / a1
  a1 / (1 - r)

noncomputable def second_series_sum (m : ℝ) : ℝ := 
  let b1 : ℝ := 18
  let b2 : ℝ := 6 + m
  let s : ℝ := b2 / b1
  b1 / (1 - s)

theorem find_m : 
  (3 : ℝ) * first_series_sum = second_series_sum m → m = 8 := 
by 
  sorry

end find_m_l74_74321


namespace find_OQ_l74_74312
-- Import the required math libarary

-- Define points on a line with the given distances
def O := 0
def A (a : ℝ) := 2 * a
def B (b : ℝ) := 4 * b
def C (c : ℝ) := 5 * c
def D (d : ℝ) := 7 * d

-- Given P between B and C such that ratio condition holds
def P (a b c d x : ℝ) := 
  B b ≤ x ∧ x ≤ C c ∧ 
  (A a - x) * (x - C c) = (B b - x) * (x - D d)

-- Calculate Q based on given ratio condition
def Q (b c d y : ℝ) := 
  C c ≤ y ∧ y ≤ D d ∧ 
  (C c - y) * (y - D d) = (B b - C c) * (C c - D d)

-- Main Proof Statement to prove OQ
theorem find_OQ (a b c d y : ℝ) 
  (hP : ∃ x, P a b c d x)
  (hQ : ∃ y, Q b c d y) :
  y = (14 * c * d - 10 * b * c) / (5 * c - 7 * d) := by
  sorry

end find_OQ_l74_74312


namespace determine_phi_l74_74588

theorem determine_phi
  (A ω : ℝ) (φ : ℝ) (x : ℝ)
  (hA : 0 < A)
  (hω : 0 < ω)
  (hφ : abs φ < Real.pi / 2)
  (h_symm : ∃ f : ℝ → ℝ, f (-Real.pi / 4) = A ∨ f (-Real.pi / 4) = -A)
  (h_zero : ∃ x₀ : ℝ, A * Real.sin (ω * x₀ + φ) = 0 ∧ abs (x₀ + Real.pi / 4) = Real.pi / 2) :
  φ = -Real.pi / 4 :=
sorry

end determine_phi_l74_74588


namespace magnitude_BC_eq_sqrt29_l74_74162

noncomputable def A : (ℝ × ℝ) := (2, -1)
noncomputable def C : (ℝ × ℝ) := (0, 2)
noncomputable def AB : (ℝ × ℝ) := (3, 5)

theorem magnitude_BC_eq_sqrt29
    (A : ℝ × ℝ := (2, -1))
    (C : ℝ × ℝ := (0, 2))
    (AB : ℝ × ℝ := (3, 5)) :
    ∃ B : ℝ × ℝ, (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = 29 := 
by
  sorry

end magnitude_BC_eq_sqrt29_l74_74162


namespace distance_between_towns_l74_74756

-- Define the custom scale for conversion
def scale_in_km := 1.05  -- 1 km + 50 meters as 1.05 km

-- Input distances on the map and their conversion
def map_distance_in_inches := 6 + 11/16

noncomputable def actual_distance_in_km : ℝ :=
  let distance_in_inches := (6 * 8 + 11) / 16
  distance_in_inches * (8 / 3)

theorem distance_between_towns :
  actual_distance_in_km = 17.85 := by
  -- Equivalent mathematical steps and tests here
  sorry

end distance_between_towns_l74_74756


namespace negation_proof_l74_74244

theorem negation_proof :
  ¬ (∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  -- Proof to be filled
  sorry

end negation_proof_l74_74244


namespace part1_part2_l74_74230
noncomputable def equation1 (x k : ℝ) := 3 * (2 * x - 1) = k + 2 * x
noncomputable def equation2 (x k : ℝ) := (x - k) / 2 = x + 2 * k

theorem part1 (x k : ℝ) (h1 : equation1 4 k) : equation2 x k ↔ x = -65 := sorry

theorem part2 (x k : ℝ) (h1 : equation1 x k) (h2 : equation2 x k) : k = -1 / 7 := sorry

end part1_part2_l74_74230


namespace population_after_panic_l74_74239

noncomputable def original_population : ℕ := 7200
def first_event_loss (population : ℕ) : ℕ := population * 10 / 100
def after_first_event (population : ℕ) : ℕ := population - first_event_loss population
def second_event_loss (population : ℕ) : ℕ := population * 25 / 100
def after_second_event (population : ℕ) : ℕ := population - second_event_loss population

theorem population_after_panic : after_second_event (after_first_event original_population) = 4860 := sorry

end population_after_panic_l74_74239


namespace common_chord_of_circles_l74_74502

theorem common_chord_of_circles :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (x^2 + y^2 - 4 * y = 0) → (x = y) :=
by
  intros x y h
  sorry

end common_chord_of_circles_l74_74502


namespace center_of_circle_l74_74968

-- Let's define the circle as a set of points satisfying the given condition.
def circle (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 4

-- Prove that the point (2, -1) is the center of this circle in ℝ².
theorem center_of_circle : ∀ (x y : ℝ), circle (x - 2) (y + 1) ↔ (x, y) = (2, -1) :=
by
  intros x y
  sorry

end center_of_circle_l74_74968


namespace total_coins_Zain_l74_74089

variable (quartersEmerie dimesEmerie nickelsEmerie : Nat)
variable (additionalCoins : Nat)

theorem total_coins_Zain (h_q : quartersEmerie = 6)
                         (h_d : dimesEmerie = 7)
                         (h_n : nickelsEmerie = 5)
                         (h_add : additionalCoins = 10) :
    let quartersZain := quartersEmerie + additionalCoins
    let dimesZain := dimesEmerie + additionalCoins
    let nickelsZain := nickelsEmerie + additionalCoins
    quartersZain + dimesZain + nickelsZain = 48 := by
  sorry

end total_coins_Zain_l74_74089


namespace jim_anne_mary_paul_report_time_l74_74905

def typing_rate_jim := 1 / 12
def typing_rate_anne := 1 / 20
def combined_typing_rate := typing_rate_jim + typing_rate_anne
def typing_time := 1 / combined_typing_rate

def editing_rate_mary := 1 / 30
def editing_rate_paul := 1 / 10
def combined_editing_rate := editing_rate_mary + editing_rate_paul
def editing_time := 1 / combined_editing_rate

theorem jim_anne_mary_paul_report_time : 
  typing_time + editing_time = 15 := by
  sorry

end jim_anne_mary_paul_report_time_l74_74905


namespace annual_interest_rate_is_approx_14_87_percent_l74_74778

-- Let P be the principal amount, r the annual interest rate, and n the number of years
-- Given: A = P(1 + r)^n, where A is the amount of money after n years
-- In this problem: A = 2P, n = 5

theorem annual_interest_rate_is_approx_14_87_percent
    (P : Real) (r : Real) (n : Real) (A : Real) (condition1 : n = 5)
    (condition2 : A = 2 * P)
    (condition3 : A = P * (1 + r)^n) :
  r = 2^(1/5) - 1 := 
  sorry

end annual_interest_rate_is_approx_14_87_percent_l74_74778


namespace g_is_odd_function_l74_74901

noncomputable def g (x : ℝ) := 5 / (3 * x^5 - 7 * x)

theorem g_is_odd_function : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  unfold g
  sorry

end g_is_odd_function_l74_74901


namespace unit_prices_purchasing_schemes_maximize_profit_l74_74138

-- Define the conditions and variables
def purchase_price_system (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 240) ∧ (3 * x + 4 * y = 340)

def possible_schemes (a b : ℕ) : Prop :=
  (a + b = 200) ∧ (60 * a + 40 * b ≤ 10440) ∧ (a ≥ 3 * b / 2)

def max_profit (x y : ℝ) (a b : ℕ) : ℝ :=
  (x - 60) * a + (y - 40) * b

-- Prove the unit prices are $60 and $40
theorem unit_prices : ∃ x y, purchase_price_system x y ∧ x = 60 ∧ y = 40 :=
by
  sorry

-- Prove the possible purchasing schemes
theorem purchasing_schemes : ∀ a b, possible_schemes a b → 
  (a = 120 ∧ b = 80 ∨ a = 121 ∧ b = 79 ∨ a = 122 ∧ b = 78) :=
by
  sorry

-- Prove the maximum profit is 3610 with the purchase amounts (122, 78)
theorem maximize_profit :
  ∃ (a b : ℕ), max_profit 80 55 a b = 3610 ∧ purchase_price_system 60 40 ∧ possible_schemes a b ∧ a = 122 ∧ b = 78 :=
by
  sorry

end unit_prices_purchasing_schemes_maximize_profit_l74_74138


namespace raisin_cost_fraction_l74_74650

theorem raisin_cost_fraction
  (R : ℝ)                -- cost of a pound of raisins
  (cost_nuts : ℝ := 2 * R)  -- cost of a pound of nuts
  (cost_raisins : ℝ := 3 * R)  -- cost of 3 pounds of raisins
  (cost_nuts_total : ℝ := 4 * cost_nuts)  -- cost of 4 pounds of nuts
  (total_cost : ℝ := cost_raisins + cost_nuts_total)  -- total cost of the mixture
  (fraction_of_raisins : ℝ := cost_raisins / total_cost)  -- fraction of cost of raisins
  : fraction_of_raisins = 3 / 11 := 
by
  sorry

end raisin_cost_fraction_l74_74650


namespace Jane_age_l74_74445

theorem Jane_age (x : ℕ) 
  (h1 : ∃ n1 : ℕ, x - 1 = n1 ^ 2) 
  (h2 : ∃ n2 : ℕ, x + 1 = n2 ^ 3) : 
  x = 26 :=
sorry

end Jane_age_l74_74445


namespace remainder_of_2519_div_8_l74_74080

theorem remainder_of_2519_div_8 : 2519 % 8 = 7 := 
by 
  sorry

end remainder_of_2519_div_8_l74_74080


namespace fourth_boy_payment_l74_74334

theorem fourth_boy_payment (a b c d : ℝ) 
  (h₁ : a = (1 / 2) * (b + c + d)) 
  (h₂ : b = (1 / 3) * (a + c + d)) 
  (h₃ : c = (1 / 4) * (a + b + d)) 
  (h₄ : a + b + c + d = 60) : 
  d = 13 := 
sorry

end fourth_boy_payment_l74_74334


namespace C3PO_Optimal_Play_Wins_l74_74488

def initial_number : List ℕ := [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]

-- Conditions for the game
structure GameConditions where
  number : List ℕ
  robots : List String
  cannot_swap : List (ℕ × ℕ) -- Pair of digits that cannot be swapped again
  cannot_start_with_zero : Bool
  c3po_starts : Bool

-- Define the initial conditions
def initial_conditions : GameConditions :=
{
  number := initial_number,
  robots := ["C3PO", "R2D2"],
  cannot_swap := [],
  cannot_start_with_zero := true,
  c3po_starts := true
}

-- Define the winning condition for C3PO
def C3PO_wins : Prop :=
  ∀ game : GameConditions, game = initial_conditions → ∃ is_c3po_winner : Bool, is_c3po_winner = true

-- The theorem statement
theorem C3PO_Optimal_Play_Wins : C3PO_wins :=
by
  sorry

end C3PO_Optimal_Play_Wins_l74_74488


namespace certain_number_eq_l74_74296

theorem certain_number_eq :
  ∃ y : ℝ, y + (y * 4) = 48 ∧ y = 9.6 :=
by
  sorry

end certain_number_eq_l74_74296


namespace problem_l74_74477

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ) (a b c : ℝ)
  (h1 : ∀ x, f (x + 1) = f (-x + 1))
  (h2 : ∀ x, 1 < x → f x ≤ f (x - 1))
  (ha : a = f 2)
  (hb : b = f (Real.log 2 / Real.log 3))
  (hc : c = f (1 / 2))

theorem problem (h : a = f 2 ∧ b = f (Real.log 2 / Real.log 3) ∧ c = f (1 / 2)) : 
  a < c ∧ c < b := sorry

end problem_l74_74477


namespace quadratic_root_property_l74_74338

theorem quadratic_root_property (m n : ℝ)
  (hmn : m^2 + m - 2021 = 0)
  (hn : n^2 + n - 2021 = 0) :
  m^2 + 2 * m + n = 2020 :=
by sorry

end quadratic_root_property_l74_74338


namespace problem1_problem2_problem3_problem4_l74_74865

theorem problem1 : (5 / 16) - (3 / 16) + (7 / 16) = 9 / 16 := by
  sorry

theorem problem2 : (3 / 12) - (4 / 12) + (6 / 12) = 5 / 12 := by
  sorry

theorem problem3 : 64 + 27 + 81 + 36 + 173 + 219 + 136 = 736 := by
  sorry

theorem problem4 : (2 : ℚ) - (8 / 9) - (1 / 9) + (1 + 98 / 99) = 2 + 98 / 99 := by
  sorry

end problem1_problem2_problem3_problem4_l74_74865


namespace intersection_point_exists_l74_74421

theorem intersection_point_exists
  (m n a b : ℝ)
  (h1 : m * a + 2 * m * b = 5)
  (h2 : n * a - 2 * n * b = 7)
  : (∃ x y : ℝ, 
    (y = (5 / (2 * m)) - (1 / 2) * x) ∧ 
    (y = (1 / 2) * x - (7 / (2 * n))) ∧
    (x = a) ∧ (y = b)) :=
sorry

end intersection_point_exists_l74_74421


namespace carbonic_acid_formation_l74_74712

-- Definition of amounts of substances involved
def moles_CO2 : ℕ := 3
def moles_H2O : ℕ := 3

-- Stoichiometric condition derived from the equation CO2 + H2O → H2CO3
def stoichiometric_ratio (a b c : ℕ) : Prop := (a = b) ∧ (a = c)

-- The main statement to prove
theorem carbonic_acid_formation : 
  stoichiometric_ratio moles_CO2 moles_H2O 3 :=
by
  sorry

end carbonic_acid_formation_l74_74712


namespace rectangle_area_coefficient_l74_74645

theorem rectangle_area_coefficient (length width d k : ℝ) 
(h1 : length / width = 5 / 2) 
(h2 : d^2 = length^2 + width^2) 
(h3 : k = 10 / 29) :
  (length * width = k * d^2) :=
by
  sorry

end rectangle_area_coefficient_l74_74645


namespace sin_neg_pi_div_two_l74_74815

theorem sin_neg_pi_div_two : Real.sin (-π / 2) = -1 := by
  -- Define the necessary conditions
  let π_in_deg : ℝ := 180 -- π radians equals 180 degrees
  have sin_neg_angle : ∀ θ : ℝ, Real.sin (-θ) = -Real.sin θ := sorry -- sin(-θ) = -sin(θ) for any θ
  have sin_90_deg : Real.sin (π_in_deg / 2) = 1 := sorry -- sin(90 degrees) = 1

  -- The main statement to prove
  sorry

end sin_neg_pi_div_two_l74_74815


namespace part_I_part_II_l74_74819

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

theorem part_I (α : ℝ) (hα : ∃ (P : ℝ × ℝ), P = (Real.sqrt 3, -1) ∧
  (Real.tan α = -1 / Real.sqrt 3 ∨ Real.tan α = - (Real.sqrt 3) / 3)) :
  f α = -3 := by
  sorry

theorem part_II (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  -2 ≤ f x ∧ f x ≤ 1 := by
  sorry

end part_I_part_II_l74_74819


namespace calculate_expression_l74_74456

theorem calculate_expression : (Real.pi - 2023)^0 - |1 - Real.sqrt 2| + 2 * Real.cos (Real.pi / 4) - (1 / 2)⁻¹ = 0 :=
by
  sorry

end calculate_expression_l74_74456


namespace price_of_each_shirt_is_15_30_l74_74229

theorem price_of_each_shirt_is_15_30:
  ∀ (shorts_price : ℝ) (num_shorts : ℕ) (shirt_num : ℕ) (total_paid : ℝ) (discount : ℝ),
  shorts_price = 15 →
  num_shorts = 3 →
  shirt_num = 5 →
  total_paid = 117 →
  discount = 0.10 →
  (total_paid - (num_shorts * shorts_price - discount * (num_shorts * shorts_price))) / shirt_num = 15.30 :=
by 
  sorry

end price_of_each_shirt_is_15_30_l74_74229


namespace min_value_of_derivative_l74_74429

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2 * a * x^2 + (1 / a) * x

noncomputable def f' (a : ℝ) : ℝ := 3 * 2^2 + 4 * a * 2 + (1 / a)

theorem min_value_of_derivative (a : ℝ) (h : a > 0) : 
  f' a ≥ 12 + 8 * Real.sqrt 2 :=
sorry

end min_value_of_derivative_l74_74429


namespace initial_momentum_eq_2Fx_div_v_l74_74164

variable (m v F x t : ℝ)
variable (H_initial_conditions : v ≠ 0)
variable (H_force : F > 0)
variable (H_distance : x > 0)
variable (H_time : t > 0)
variable (H_stopping_distance : x = (m * v^2) / (2 * F))
variable (H_stopping_time : t = (m * v) / F)

theorem initial_momentum_eq_2Fx_div_v :
  m * v = (2 * F * x) / v :=
sorry

end initial_momentum_eq_2Fx_div_v_l74_74164


namespace original_volume_l74_74378

variable {π : Real} (r h : Real)

theorem original_volume (hπ : π ≠ 0) (hr : r ≠ 0) (hh : h ≠ 0) (condition : 3 * π * r^2 * h = 180) : π * r^2 * h = 60 := by
  sorry

end original_volume_l74_74378


namespace range_of_a_l74_74811

open Real

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem range_of_a (a : ℝ) (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) :
  (f (a * sin θ) + f (1 - a) > 0) → a ≤ 1 :=
sorry

end range_of_a_l74_74811


namespace range_of_a_l74_74616

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x ∈ (Set.Iio (-1) ∪ Set.Ioi 3)) → ((x + a) * (x + 1) > 0)) ∧ 
  (∃ x : ℝ, ¬(x ∈ (Set.Iio (-1) ∪ Set.Ioi 3)) ∧ ((x + a) * (x + 1) > 0)) → 
  a ∈ Set.Iio (-3) := 
  sorry

end range_of_a_l74_74616


namespace second_percentage_increase_l74_74132

theorem second_percentage_increase (P : ℝ) (x : ℝ) :
  1.25 * P * (1 + x / 100) = 1.625 * P ↔ x = 30 :=
by
  sorry

end second_percentage_increase_l74_74132


namespace workEfficiencyRatioProof_is_2_1_l74_74885

noncomputable def workEfficiencyRatioProof : Prop :=
  ∃ (A B : ℝ), 
  (1 / B = 21) ∧ 
  (1 / (A + B) = 7) ∧
  (A / B = 2)

theorem workEfficiencyRatioProof_is_2_1 : workEfficiencyRatioProof :=
  sorry

end workEfficiencyRatioProof_is_2_1_l74_74885


namespace tan_ratio_l74_74752

theorem tan_ratio (α β : ℝ) (h : Real.sin (2 * α) = 3 * Real.sin (2 * β)) :
  (Real.tan (α - β) / Real.tan (α + β)) = 1 / 2 :=
sorry

end tan_ratio_l74_74752


namespace max_gcd_b_n_b_n_plus_1_max_possible_value_of_e_n_l74_74860

def b_n (n : ℕ) : ℤ := (10 ^ n - 9) / 3
def e_n (n : ℕ) : ℤ := Int.gcd (b_n n) (b_n (n + 1))

theorem max_gcd_b_n_b_n_plus_1 : ∀ n : ℕ, e_n n ≤ 3 :=
by
  -- Provide the proof here
  sorry

theorem max_possible_value_of_e_n : ∃ n : ℕ, e_n n = 3 :=
by
  -- Provide the proof here
  sorry

end max_gcd_b_n_b_n_plus_1_max_possible_value_of_e_n_l74_74860


namespace fastest_pipe_is_4_l74_74613

/-- There are five pipes with flow rates Q_1, Q_2, Q_3, Q_4, and Q_5.
    The ordering of their flow rates is given by:
    (1) Q_1 > Q_3
    (2) Q_2 < Q_4
    (3) Q_3 < Q_5
    (4) Q_4 > Q_1
    (5) Q_5 < Q_2
    We need to prove that single pipe Q_4 will fill the pool the fastest.
 -/
theorem fastest_pipe_is_4 
  (Q1 Q2 Q3 Q4 Q5 : ℝ)
  (h1 : Q1 > Q3)
  (h2 : Q2 < Q4)
  (h3 : Q3 < Q5)
  (h4 : Q4 > Q1)
  (h5 : Q5 < Q2) :
  Q4 > Q1 ∧ Q4 > Q2 ∧ Q4 > Q3 ∧ Q4 > Q5 :=
by
  sorry

end fastest_pipe_is_4_l74_74613


namespace evaluate_special_operation_l74_74125

-- Define the operation @
def special_operation (a b : ℕ) : ℚ := (a * b) / (a - b)

-- State the theorem
theorem evaluate_special_operation : special_operation 6 3 = 6 := by
  sorry

end evaluate_special_operation_l74_74125


namespace incorrect_statement_l74_74724

-- Conditions
variable (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
variables (triangleABC : Triangle A B C) (triangleDEF : Triangle D E F)

-- Congruence of triangles
axiom congruent_triangles : triangleABC ≌ triangleDEF

-- Proving incorrect statement
theorem incorrect_statement : ¬ (AB = EF) := by
  sorry

end incorrect_statement_l74_74724


namespace basketball_free_throws_l74_74075

theorem basketball_free_throws:
  ∀ (a b x : ℕ),
    3 * b = 4 * a →
    x = 2 * a →
    2 * a + 3 * b + x = 65 →
    x = 18 := 
by
  intros a b x h1 h2 h3
  sorry

end basketball_free_throws_l74_74075


namespace solve_inequality_l74_74213

theorem solve_inequality (x : ℝ) : -3 * x^2 + 8 * x + 1 < 0 ↔ x ∈ Set.Ioo (-1 / 3 : ℝ) 1 :=
sorry

end solve_inequality_l74_74213


namespace simplify_product_of_fractions_l74_74253

theorem simplify_product_of_fractions :
  (252 / 21) * (7 / 168) * (12 / 4) = 3 / 2 :=
by
  sorry

end simplify_product_of_fractions_l74_74253


namespace ratio_d_s_l74_74614

theorem ratio_d_s (s d : ℝ) 
  (h : (25 * 25 * s^2) / (25 * s + 50 * d)^2 = 0.81) :
  d / s = 1 / 18 :=
by
  sorry

end ratio_d_s_l74_74614


namespace event_B_more_likely_l74_74093

theorem event_B_more_likely (A B : Set (ℕ → ℕ)) 
  (hA : ∀ ω, ω ∈ A ↔ ∃ i j, i ≠ j ∧ ω i = ω j)
  (hB : ∀ ω, ω ∈ B ↔ ∀ i j, i ≠ j → ω i ≠ ω j) :
  ∃ prob_A prob_B : ℚ, prob_A = 4 / 9 ∧ prob_B = 5 / 9 ∧ prob_B > prob_A :=
by
  sorry

end event_B_more_likely_l74_74093


namespace sum_of_cubes_l74_74725

theorem sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) (h3 : abc = 5) : a^3 + b^3 + c^3 = 15 :=
by
  sorry

end sum_of_cubes_l74_74725


namespace hyperbola_eccentricity_l74_74689

theorem hyperbola_eccentricity (a b c : ℝ) (h : (c^2 - a^2 = 5 * a^2)) (hb : a / b = 2) :
  (c / a = Real.sqrt 5) :=
by
  sorry

end hyperbola_eccentricity_l74_74689


namespace A_days_to_complete_job_l74_74297

noncomputable def time_for_A (x : ℝ) (work_left : ℝ) : ℝ :=
  let work_rate_A := 1 / x
  let work_rate_B := 1 / 30
  let combined_work_rate := work_rate_A + work_rate_B
  let completed_work := 4 * combined_work_rate
  let fraction_work_left := 1 - completed_work
  fraction_work_left

theorem A_days_to_complete_job : ∃ x : ℝ, time_for_A x 0.6 = 0.6 ∧ x = 15 :=
by {
  use 15,
  sorry
}

end A_days_to_complete_job_l74_74297


namespace smallest_n_for_terminating_fraction_l74_74245

-- Define what it means for a number to be a product of only prime factors of 2 and 5
def isTerminatingDenominator (d : ℕ) : Prop := ∃ (a b : ℕ), d = 2^a * 5^b

-- The main statement to prove
theorem smallest_n_for_terminating_fraction : 
  ∃ (n : ℕ), 0 < n ∧ isTerminatingDenominator (n + 150) ∧ 
  ∀ (m : ℕ), (0 < m → isTerminatingDenominator (m + 150) → n ≤ m)
:= sorry

end smallest_n_for_terminating_fraction_l74_74245


namespace visiting_plans_correct_l74_74218

-- Define the number of students
def num_students : ℕ := 4

-- Define the number of places to visit
def num_places : ℕ := 3

-- Define the total number of visiting plans without any restrictions
def total_visiting_plans : ℕ := num_places ^ num_students

-- Define the number of visiting plans where no one visits Haxi Station
def no_haxi_visiting_plans : ℕ := (num_places - 1) ^ num_students

-- Define the number of visiting plans where Haxi Station has at least one visitor
def visiting_plans_with_haxi : ℕ := total_visiting_plans - no_haxi_visiting_plans

-- Prove that the number of different visiting plans with at least one student visiting Haxi Station is 65
theorem visiting_plans_correct : visiting_plans_with_haxi = 65 := by
  -- Omitted proof
  sorry

end visiting_plans_correct_l74_74218


namespace Dima_broke_more_l74_74497

theorem Dima_broke_more (D F : ℕ) (h : 2 * D + 7 * F = 3 * (D + F)) : D = 4 * F :=
sorry

end Dima_broke_more_l74_74497


namespace percent_increase_expenditure_l74_74980

theorem percent_increase_expenditure (cost_per_minute_2005 minutes_2005 minutes_2020 total_expenditure_2005 total_expenditure_2020 : ℕ)
  (h1 : cost_per_minute_2005 = 10)
  (h2 : minutes_2005 = 200)
  (h3 : minutes_2020 = 2 * minutes_2005)
  (h4 : total_expenditure_2005 = minutes_2005 * cost_per_minute_2005)
  (h5 : total_expenditure_2020 = minutes_2020 * cost_per_minute_2005) :
  ((total_expenditure_2020 - total_expenditure_2005) * 100 / total_expenditure_2005) = 100 :=
by
  sorry

end percent_increase_expenditure_l74_74980


namespace purchasing_power_increase_l74_74851

theorem purchasing_power_increase (P M : ℝ) (h : 0 < P ∧ 0 < M) :
  let new_price := 0.80 * P
  let original_quantity := M / P
  let new_quantity := M / new_price
  new_quantity = 1.25 * original_quantity :=
by
  sorry

end purchasing_power_increase_l74_74851


namespace NoahMealsCount_l74_74736

-- Definition of all the choices available to Noah
def MainCourses := ["Pizza", "Burger", "Pasta"]
def Beverages := ["Soda", "Juice"]
def Snacks := ["Apple", "Banana", "Cookie"]

-- Condition that Noah avoids soda with pizza
def isValidMeal (main : String) (beverage : String) : Bool :=
  not (main = "Pizza" ∧ beverage = "Soda")

-- Total number of valid meal combinations
def totalValidMeals : Nat :=
  (if isValidMeal "Pizza" "Juice" then 1 else 0) * Snacks.length +
  (Beverages.length - 1) * Snacks.length * (MainCourses.length - 1) + -- for Pizza
  Beverages.length * Snacks.length * 2 -- for Burger and Pasta

-- The theorem that Noah can buy 15 distinct meals
theorem NoahMealsCount : totalValidMeals = 15 := by
  sorry

end NoahMealsCount_l74_74736


namespace second_offset_length_l74_74160

-- Definitions based on the given conditions.
def diagonal : ℝ := 24
def offset1 : ℝ := 9
def area_quad : ℝ := 180

-- Statement to prove the length of the second offset.
theorem second_offset_length :
  ∃ h : ℝ, (1 / 2) * diagonal * offset1 + (1 / 2) * diagonal * h = area_quad ∧ h = 6 :=
by
  sorry

end second_offset_length_l74_74160


namespace equation_one_equation_two_l74_74082

-- Equation (1): Show that for the equation ⟦ ∀ x, (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3 ↔ x = 1 / 5) ⟧
theorem equation_one (x : ℝ) : (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3) ↔ (x = 1 / 5) :=
sorry

-- Equation (2): Show that for the equation ⟦ ∀ x, ((4 / (x^2 - 4) - 1 / (x - 2) = 0) ↔ false) ⟧
theorem equation_two (x : ℝ) : (4 / (x^2 - 4) - 1 / (x - 2) = 0) ↔ false :=
sorry

end equation_one_equation_two_l74_74082


namespace problem1_problem2_l74_74100

-- Problem 1: Prove that (x + y + z)² - (x + y - z)² = 4z(x + y) for x, y, z ∈ ℝ
theorem problem1 (x y z : ℝ) : (x + y + z)^2 - (x + y - z)^2 = 4 * z * (x + y) := 
sorry

-- Problem 2: Prove that (a + 2b)² - 2(a + 2b)(a - 2b) + (a - 2b)² = 16b² for a, b ∈ ℝ
theorem problem2 (a b : ℝ) : (a + 2 * b)^2 - 2 * (a + 2 * b) * (a - 2 * b) + (a - 2 * b)^2 = 16 * b^2 := 
sorry

end problem1_problem2_l74_74100


namespace least_number_to_make_divisible_l74_74185

def least_common_multiple (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem least_number_to_make_divisible (n : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) : 
  least_common_multiple a b = 77 → 
  (n % least_common_multiple a b) = 40 →
  c = (least_common_multiple a b - (n % least_common_multiple a b)) →
  c = 37 :=
by
sorry

end least_number_to_make_divisible_l74_74185


namespace range_of_x_l74_74841

variable {f : ℝ → ℝ}

-- Define the function is_increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem range_of_x (h_inc : is_increasing f) (h_ineq : ∀ x : ℝ, f x < f (2 * x - 3)) :
  ∀ x : ℝ, 3 < x → f x < f (2 * x - 3) := 
sorry

end range_of_x_l74_74841


namespace remainder_77_pow_77_minus_15_mod_19_l74_74694

theorem remainder_77_pow_77_minus_15_mod_19 : (77^77 - 15) % 19 = 5 := by
  sorry

end remainder_77_pow_77_minus_15_mod_19_l74_74694


namespace part1_part2_l74_74448

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^3 + k * Real.log x
noncomputable def f' (x : ℝ) (k : ℝ) : ℝ := 3 * x^2 + k / x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x k - f' x k + 9 / x

-- Part (1): Prove the monotonic intervals and extreme values for k = 6:
theorem part1 :
  (∀ x : ℝ, 0 < x ∧ x < 1 → g x 6 < g 1 6) ∧
  (∀ x : ℝ, 1 < x → g x 6 > g 1 6) ∧
  (g 1 6 = 1) := sorry

-- Part (2): Prove the given inequality for k ≥ -3:
theorem part2 (k : ℝ) (hk : k ≥ -3) (x1 x2 : ℝ) (hx1 : x1 ≥ 1) (hx2 : x2 ≥ 1) (h : x1 > x2) :
  (f' x1 k + f' x2 k) / 2 > (f x1 k - f x2 k) / (x1 - x2) := sorry

end part1_part2_l74_74448


namespace rectangle_area_error_percentage_l74_74433

theorem rectangle_area_error_percentage (L W : ℝ) :
  let L' := 1.10 * L
  let W' := 0.95 * W
  let A := L * W 
  let A' := L' * W'
  let error := A' - A
  let error_percentage := (error / A) * 100
  error_percentage = 4.5 := by
  sorry

end rectangle_area_error_percentage_l74_74433


namespace scientific_notation_11580000_l74_74208

theorem scientific_notation_11580000 :
  (11580000 : ℝ) = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l74_74208


namespace sufficient_condition_for_inequality_l74_74472

theorem sufficient_condition_for_inequality (m : ℝ) : (m ≥ 2) → (∀ x : ℝ, x^2 - 2 * x + m ≥ 0) :=
by
  sorry

end sufficient_condition_for_inequality_l74_74472


namespace interval_between_births_l74_74430

variables {A1 A2 A3 A4 A5 : ℝ}
variable {x : ℝ}

def ages (A1 A2 A3 A4 A5 : ℝ) := A1 + A2 + A3 + A4 + A5 = 50
def youngest (A1 : ℝ) := A1 = 4
def interval (x : ℝ) := x = 3.4

theorem interval_between_births
  (h_age_sum: ages A1 A2 A3 A4 A5)
  (h_youngest: youngest A1)
  (h_ages: A2 = A1 + x ∧ A3 = A1 + 2 * x ∧ A4 = A1 + 3 * x ∧ A5 = A1 + 4 * x) :
  interval x :=
by {
  sorry
}

end interval_between_births_l74_74430


namespace rice_difference_on_15th_and_first_10_squares_l74_74207

-- Definitions
def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_first_n_squares (n : ℕ) : ℕ := 
  (3 * (3^n - 1)) / (3 - 1)

-- Theorem statement
theorem rice_difference_on_15th_and_first_10_squares :
  grains_on_square 15 - sum_first_n_squares 10 = 14260335 :=
by
  sorry

end rice_difference_on_15th_and_first_10_squares_l74_74207


namespace solve_problem_l74_74287

-- Definitions from the conditions
def is_divisible_by (n k : ℕ) : Prop :=
  ∃ m, k * m = n

def count_divisors (limit k : ℕ) : ℕ :=
  Nat.div limit k

def count_numbers_divisible_by_neither_5_nor_7 (limit : ℕ) : ℕ :=
  let total := limit - 1
  let divisible_by_5 := count_divisors limit 5
  let divisible_by_7 := count_divisors limit 7
  let divisible_by_35 := count_divisors limit 35
  total - (divisible_by_5 + divisible_by_7 - divisible_by_35)

-- The statement to be proved
theorem solve_problem : count_numbers_divisible_by_neither_5_nor_7 1000 = 686 :=
by
  sorry

end solve_problem_l74_74287


namespace total_bill_correct_l74_74973

def scoop_cost : ℕ := 2
def pierre_scoops : ℕ := 3
def mom_scoops : ℕ := 4

def pierre_total : ℕ := pierre_scoops * scoop_cost
def mom_total : ℕ := mom_scoops * scoop_cost
def total_bill : ℕ := pierre_total + mom_total

theorem total_bill_correct : total_bill = 14 :=
by
  sorry

end total_bill_correct_l74_74973


namespace original_price_l74_74503

noncomputable def original_selling_price (CP : ℝ) : ℝ := CP * 1.25
noncomputable def selling_price_at_loss (CP : ℝ) : ℝ := CP * 0.5

theorem original_price (CP : ℝ) (h : selling_price_at_loss CP = 320) : original_selling_price CP = 800 :=
by
  sorry

end original_price_l74_74503


namespace lattice_point_count_l74_74630

theorem lattice_point_count :
  (∃ (S : Finset (ℤ × ℤ)), S.card = 16 ∧ ∀ (p : ℤ × ℤ), p ∈ S → (|p.1| - 1) ^ 2 + (|p.2| - 1) ^ 2 < 2) :=
sorry

end lattice_point_count_l74_74630


namespace simplify_expression_l74_74069

theorem simplify_expression : 
  (20 * (9 / 14) * (1 / 18) : ℚ) = (5 / 7) := 
by 
  sorry

end simplify_expression_l74_74069


namespace expand_and_simplify_l74_74451

theorem expand_and_simplify :
  (x : ℝ) → (x^2 - 3 * x + 3) * (x^2 + 3 * x + 3) = x^4 - 3 * x^2 + 9 :=
by 
  sorry

end expand_and_simplify_l74_74451


namespace carol_sold_cupcakes_l74_74428

variable (initial_cupcakes := 30) (additional_cupcakes := 28) (final_cupcakes := 49)

theorem carol_sold_cupcakes : (initial_cupcakes + additional_cupcakes - final_cupcakes = 9) :=
by sorry

end carol_sold_cupcakes_l74_74428


namespace pencils_count_l74_74638

theorem pencils_count (pens pencils : ℕ) 
  (h_ratio : 6 * pens = 5 * pencils) 
  (h_difference : pencils = pens + 6) : 
  pencils = 36 := 
by 
  sorry

end pencils_count_l74_74638


namespace maximum_M_value_l74_74081

theorem maximum_M_value (x y z u M : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < u)
  (h5 : x - 2 * y = z - 2 * u) (h6 : 2 * y * z = u * x) (h7 : z ≥ y) 
  : ∃ M, M ≤ z / y ∧ M ≤ 6 + 4 * Real.sqrt 2 :=
sorry

end maximum_M_value_l74_74081


namespace greatest_k_inequality_l74_74097

theorem greatest_k_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ( ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
    (a / b + b / c + c / a - 3) ≥ k * (a / (b + c) + b / (c + a) + c / (a + b) - 3 / 2) ) ↔ k = 1 := 
sorry

end greatest_k_inequality_l74_74097


namespace sin_double_angle_l74_74954

theorem sin_double_angle (x : ℝ)
  (h : Real.sin (x + Real.pi / 4) = 4 / 5) :
  Real.sin (2 * x) = 7 / 25 :=
by
  sorry

end sin_double_angle_l74_74954


namespace radius_of_garden_outer_boundary_l74_74022

-- Definitions based on the conditions from the problem statement
def fountain_diameter : ℝ := 12
def garden_width : ℝ := 10

-- Question translated to a proof statement
theorem radius_of_garden_outer_boundary :
  (fountain_diameter / 2 + garden_width) = 16 := 
by 
  sorry

end radius_of_garden_outer_boundary_l74_74022


namespace quadratic_properties_l74_74658

noncomputable def quadratic (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ)
  (root_neg1 : quadratic a b c (-1) = 0)
  (ineq_condition : ∀ x : ℝ, (quadratic a b c x - x) * (quadratic a b c x - (x^2 + 1) / 2) ≤ 0) :
  quadratic a b c 1 = 1 ∧ ∀ x : ℝ, quadratic a b c x = (1 / 4) * x^2 + (1 / 2) * x + (1 / 4) :=
by
  sorry

end quadratic_properties_l74_74658


namespace candice_spending_l74_74868

variable (total_budget : ℕ) (remaining_money : ℕ) (mildred_spending : ℕ)

theorem candice_spending 
  (h1 : total_budget = 100)
  (h2 : remaining_money = 40)
  (h3 : mildred_spending = 25) :
  (total_budget - remaining_money) - mildred_spending = 35 := 
by
  sorry

end candice_spending_l74_74868


namespace radius_of_semi_circle_l74_74514

variable (r w l : ℝ)

def rectangle_inscribed_semi_circle (w l : ℝ) := 
  l = 3*w ∧ 
  2*l + 2*w = 126 ∧ 
  (∃ r, l = 2*r)

theorem radius_of_semi_circle :
  (∃ w l r, rectangle_inscribed_semi_circle w l ∧ l = 2*r) → r = 23.625 :=
by
  sorry

end radius_of_semi_circle_l74_74514


namespace red_sequence_57_eq_103_l74_74631

-- Definitions based on conditions described in the problem
def red_sequence : Nat → Nat
| 0 => 1  -- First number is 1
| 1 => 2  -- Next even number
| 2 => 4  -- Next even number
-- Continue defining based on patterns from problem
| (n+3) => -- Each element recursively following the pattern
 sorry  -- Detailed pattern definition is skipped

-- Main theorem: the 57th number in the red subsequence is 103
theorem red_sequence_57_eq_103 : red_sequence 56 = 103 :=
 sorry

end red_sequence_57_eq_103_l74_74631


namespace honda_day_shift_production_l74_74398

theorem honda_day_shift_production (S : ℕ) (day_shift_production : ℕ)
  (h1 : day_shift_production = 4 * S)
  (h2 : day_shift_production + S = 5500) :
  day_shift_production = 4400 :=
sorry

end honda_day_shift_production_l74_74398


namespace rice_in_each_container_l74_74372

variable (weight_in_pounds : ℚ := 35 / 2)
variable (num_containers : ℕ := 4)
variable (pound_to_oz : ℕ := 16)

theorem rice_in_each_container :
  (weight_in_pounds * pound_to_oz) / num_containers = 70 :=
by
  sorry

end rice_in_each_container_l74_74372


namespace area_of_triangle_l74_74939

-- Define the function to calculate the area of a right isosceles triangle given the side lengths of squares
theorem area_of_triangle (a b c : ℝ) (h1 : a = 10) (h2 : b = 8) (h3 : c = 10) (right_isosceles : true) :
  (1 / 2) * a * c = 50 :=
by
  -- We state the theorem but leave the proof as sorry.
  sorry

end area_of_triangle_l74_74939


namespace total_young_fish_l74_74791

-- Define conditions
def tanks : ℕ := 3
def fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20

-- Define the main proof statement
theorem total_young_fish : tanks * fish_per_tank * young_per_fish = 240 := by
  sorry

end total_young_fish_l74_74791


namespace train_speed_in_m_per_s_l74_74680

-- Define the given train speed in kmph
def train_speed_kmph : ℕ := 72

-- Define the conversion factor from kmph to m/s
def km_per_hour_to_m_per_second (speed_in_kmph : ℕ) : ℕ := (speed_in_kmph * 1000) / 3600

-- State the theorem
theorem train_speed_in_m_per_s (h : train_speed_kmph = 72) : km_per_hour_to_m_per_second train_speed_kmph = 20 := by
  sorry

end train_speed_in_m_per_s_l74_74680


namespace intersection_of_A_and_B_l74_74289

-- Definitions of the sets A and B
def A : Set ℝ := { x | x^2 + 2*x - 3 < 0 }
def B : Set ℝ := { x | |x - 1| < 2 }

-- The statement to prove their intersection
theorem intersection_of_A_and_B : A ∩ B = { x | -1 < x ∧ x < 1 } :=
by 
  sorry

end intersection_of_A_and_B_l74_74289


namespace stella_annual_income_after_tax_l74_74144

-- Definitions of the conditions
def base_salary_per_month : ℝ := 3500
def bonuses : List ℝ := [1200, 600, 1500, 900, 1200]
def months_paid : ℝ := 10
def tax_rate : ℝ := 0.05

-- Calculations derived from the conditions
def total_base_salary : ℝ := base_salary_per_month * months_paid
def total_bonuses : ℝ := bonuses.sum
def total_income_before_tax : ℝ := total_base_salary + total_bonuses
def tax_deduction : ℝ := total_income_before_tax * tax_rate
def annual_income_after_tax : ℝ := total_income_before_tax - tax_deduction

-- The theorem to prove
theorem stella_annual_income_after_tax :
  annual_income_after_tax = 38380 := by
  sorry

end stella_annual_income_after_tax_l74_74144


namespace largest_possible_red_socks_l74_74804

theorem largest_possible_red_socks (r b : ℕ) (h1 : 0 < r) (h2 : 0 < b)
  (h3 : r + b ≤ 2500) (h4 : r > b) :
  r * (r - 1) + b * (b - 1) = 3/5 * (r + b) * (r + b - 1) → r ≤ 1164 :=
by sorry

end largest_possible_red_socks_l74_74804


namespace find_side_lengths_l74_74955

variable (a b : ℝ)

-- Conditions
def diff_side_lengths := a - b = 2
def diff_areas := a^2 - b^2 = 40

-- Theorem to prove
theorem find_side_lengths (h1 : diff_side_lengths a b) (h2 : diff_areas a b) :
  a = 11 ∧ b = 9 := by
  -- Proof skipped
  sorry

end find_side_lengths_l74_74955


namespace num_valid_k_l74_74889

/--
The number of natural numbers \( k \), not exceeding 485000, 
such that \( k^2 - 1 \) is divisible by 485 is 4000.
-/
theorem num_valid_k (n : ℕ) (h₁ : n ≤ 485000) (h₂ : 485 ∣ (n^2 - 1)) : 
  (∃ k : ℕ, k = 4000) :=
sorry

end num_valid_k_l74_74889


namespace sqrt_div_sqrt_eq_sqrt_fraction_l74_74529

theorem sqrt_div_sqrt_eq_sqrt_fraction
  (x y : ℝ)
  (h : ((1 / 2) ^ 2 + (1 / 3) ^ 2) / ((1 / 3) ^ 2 + (1 / 6) ^ 2) = 13 * x / (47 * y)) :
  (Real.sqrt x / Real.sqrt y) = (Real.sqrt 47 / Real.sqrt 5) :=
by
  sorry

end sqrt_div_sqrt_eq_sqrt_fraction_l74_74529


namespace find_cheesecake_price_l74_74169

def price_of_cheesecake (C : ℝ) (coffee_price : ℝ) (discount_rate : ℝ) (final_price : ℝ) : Prop :=
  let original_price := coffee_price + C
  let discounted_price := discount_rate * original_price
  discounted_price = final_price

theorem find_cheesecake_price : ∃ C : ℝ,
  price_of_cheesecake C 6 0.75 12 ∧ C = 10 :=
by
  sorry

end find_cheesecake_price_l74_74169


namespace miranda_savings_l74_74662

theorem miranda_savings:
  ∀ (months : ℕ) (sister_contribution price shipping total paid_per_month : ℝ),
    months = 3 →
    sister_contribution = 50 →
    price = 210 →
    shipping = 20 →
    total = 230 →
    total - sister_contribution = price + shipping →
    paid_per_month = (total - sister_contribution) / months →
    paid_per_month = 60 :=
by
  intros months sister_contribution price shipping total paid_per_month h1 h2 h3 h4 h5 h6 h7
  sorry

end miranda_savings_l74_74662


namespace fraction_minimum_decimal_digits_l74_74526

def minimum_decimal_digits (n d : ℕ) : ℕ := sorry

theorem fraction_minimum_decimal_digits :
  minimum_decimal_digits 987654321 (2^28 * 5^3) = 28 :=
sorry

end fraction_minimum_decimal_digits_l74_74526


namespace possible_triplets_l74_74695

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem possible_triplets (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (is_power_of_two (a * b - c) ∧ is_power_of_two (b * c - a) ∧ is_power_of_two (c * a - b)) ↔ 
  (a = 2 ∧ b = 2 ∧ c = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 6) ∨
  (a = 3 ∧ b = 5 ∧ c = 7) :=
by
  sorry

end possible_triplets_l74_74695


namespace triangle_inequality_l74_74090

-- Define the side lengths of a triangle
variables {a b c : ℝ}

-- State the main theorem
theorem triangle_inequality :
  (a + b) * (b + c) * (c + a) ≥ 8 * (a + b - c) * (b + c - a) * (c + a - b) :=
sorry

end triangle_inequality_l74_74090


namespace triangle_median_inequality_l74_74692

-- Defining the parameters and the inequality theorem.
theorem triangle_median_inequality
  (a b c : ℝ)
  (ma mb mc : ℝ)
  (Δ : ℝ)
  (median_medians : ∀ {a b c : ℝ}, ma ≤ mb ∧ mb ≤ mc ∧ a ≥ b ∧ b ≥ c)  :
  a * (-ma + mb + mc) + b * (ma - mb + mc) + c * (ma + mb - mc) ≥ 6 * Δ := 
sorry

end triangle_median_inequality_l74_74692


namespace isosceles_trapezoid_legs_squared_l74_74304

theorem isosceles_trapezoid_legs_squared
  (A B C D : Type)
  (AB CD AD BC : ℝ)
  (isosceles_trapezoid : AB = 50 ∧ CD = 14 ∧ AD = BC)
  (circle_tangent : ∃ M : ℝ, M = 25 ∧ ∀ x : ℝ, MD = 7 ↔ AD = x ∧ BC = x) :
  AD^2 = 800 := 
by
  sorry

end isosceles_trapezoid_legs_squared_l74_74304


namespace exists_n_good_but_not_succ_good_l74_74323

def S (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

def n_good (n : ℕ) (a : ℕ) : Prop :=
  ∃ (a_seq : Fin (n + 1) → ℕ), 
    a_seq n = a ∧ (∀ i : Fin n, a_seq (Fin.succ i) = a_seq i - S (a_seq i))

theorem exists_n_good_but_not_succ_good (n : ℕ) : 
  ∃ a, n_good n a ∧ ¬ n_good (n + 1) a := 
sorry

end exists_n_good_but_not_succ_good_l74_74323


namespace lcm_multiplied_by_2_is_72x_l74_74675

-- Define the denominators
def denom1 (x : ℕ) := 4 * x
def denom2 (x : ℕ) := 6 * x
def denom3 (x : ℕ) := 9 * x

-- Define the least common multiple of three natural numbers
def lcm_three (a b c : ℕ) := Nat.lcm a (Nat.lcm b c)

-- Define the multiplication by 2
def multiply_by_2 (n : ℕ) := 2 * n

-- Define the final result
def final_result (x : ℕ) := 72 * x

-- The proof statement
theorem lcm_multiplied_by_2_is_72x (x : ℕ): 
  multiply_by_2 (lcm_three (denom1 x) (denom2 x) (denom3 x)) = final_result x := 
by
  sorry

end lcm_multiplied_by_2_is_72x_l74_74675


namespace problem_statement_l74_74934

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
  else -- define elsewhere based on periodicity and oddness properties
    sorry 

theorem problem_statement : 
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x) → f 2015.5 = -0.5 :=
by
  intros
  sorry

end problem_statement_l74_74934


namespace find_number_of_values_l74_74342

theorem find_number_of_values (n S : ℕ) (h1 : S / n = 250) (h2 : S + 30 = 251 * n) : n = 30 :=
sorry

end find_number_of_values_l74_74342


namespace sqrt_and_cbrt_eq_self_l74_74707

theorem sqrt_and_cbrt_eq_self (x : ℝ) (h1 : x = Real.sqrt x) (h2 : x = x^(1/3)) : x = 0 := by
  sorry

end sqrt_and_cbrt_eq_self_l74_74707


namespace product_gcd_lcm_l74_74983

-- Conditions.
def num1 : ℕ := 12
def num2 : ℕ := 9

-- Theorem to prove.
theorem product_gcd_lcm (a b : ℕ) (h1 : a = num1) (h2 : b = num2) :
  (Nat.gcd a b) * (Nat.lcm a b) = 108 :=
by
  sorry

end product_gcd_lcm_l74_74983


namespace problem_part1_problem_part2_l74_74133

open Real

-- Part (1)
theorem problem_part1 : ∀ x > 0, log x ≤ x - 1 := 
by 
  sorry -- proof goes here


-- Part (2)
theorem problem_part2 : (∀ x > 0, log x ≤ a * x + (a - 1) / x - 1) → 1 ≤ a := 
by 
  sorry -- proof goes here

end problem_part1_problem_part2_l74_74133


namespace nested_fraction_evaluation_l74_74806

theorem nested_fraction_evaluation : 
  (1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - 1 / 3)))))))) = (21 / 55) :=
by
  sorry

end nested_fraction_evaluation_l74_74806


namespace remainder_of_polynomial_l74_74020

theorem remainder_of_polynomial (x : ℤ) : 
  (x^4 - 1) * (x^2 - 1) % (x^2 + x + 1) = 3 := 
sorry

end remainder_of_polynomial_l74_74020


namespace michael_completes_in_50_days_l74_74686

theorem michael_completes_in_50_days :
  ∀ {M A W : ℝ},
    (W / M + W / A = W / 20) →
    (14 * W / 20 + 10 * W / A = W) →
    M = 50 :=
by
  sorry

end michael_completes_in_50_days_l74_74686


namespace square_side_length_l74_74595

theorem square_side_length (x S : ℕ) (h1 : S > 0) (h2 : x = 4) (h3 : 4 * S = 6 * x) : S = 6 := by
  subst h2
  sorry

end square_side_length_l74_74595


namespace problem_statement_l74_74687

theorem problem_statement (n k : ℕ) (h1 : n = 2^2007 * k + 1) (h2 : k % 2 = 1) : ¬ n ∣ 2^(n-1) + 1 := by
  sorry

end problem_statement_l74_74687


namespace sum_expression_l74_74052

theorem sum_expression (x k : ℝ) (h1 : y = 3 * x) (h2 : z = k * y) : x + y + z = (4 + 3 * k) * x :=
by
  sorry

end sum_expression_l74_74052


namespace valentino_farm_birds_total_l74_74121

theorem valentino_farm_birds_total :
  let chickens := 200
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof here
  sorry

end valentino_farm_birds_total_l74_74121


namespace rabbit_travel_time_l74_74322

theorem rabbit_travel_time (distance : ℕ) (speed : ℕ) (time_in_minutes : ℕ) 
  (h_distance : distance = 3) 
  (h_speed : speed = 6) 
  (h_time_eqn : time_in_minutes = (distance * 60) / speed) : 
  time_in_minutes = 30 := 
by 
  sorry

end rabbit_travel_time_l74_74322


namespace corner_cells_different_colors_l74_74339

theorem corner_cells_different_colors 
  (colors : Fin 4 → Prop)
  (painted : (Fin 100 × Fin 100) → Fin 4)
  (h : ∀ (i j : Fin 99), 
    ∃ f g h k, 
      f ≠ g ∧ f ≠ h ∧ f ≠ k ∧
      g ≠ h ∧ g ≠ k ∧ 
      h ≠ k ∧ 
      painted (i, j) = f ∧ 
      painted (i.succ, j) = g ∧ 
      painted (i, j.succ) = h ∧ 
      painted (i.succ, j.succ) = k) :
  painted (0, 0) ≠ painted (99, 0) ∧
  painted (0, 0) ≠ painted (0, 99) ∧
  painted (0, 0) ≠ painted (99, 99) ∧
  painted (99, 0) ≠ painted (0, 99) ∧
  painted (99, 0) ≠ painted (99, 99) ∧
  painted (0, 99) ≠ painted (99, 99) :=
  sorry

end corner_cells_different_colors_l74_74339


namespace problem_statement_l74_74702

noncomputable def a := Real.sqrt 3 + Real.sqrt 2
noncomputable def b := Real.sqrt 3 - Real.sqrt 2
noncomputable def expression := a^(2 * Real.log (Real.sqrt 5) / Real.log b)

theorem problem_statement : expression = 1 / 5 := by
  sorry

end problem_statement_l74_74702


namespace gallons_10_percent_milk_needed_l74_74088

-- Definitions based on conditions
def amount_of_butterfat (x : ℝ) : ℝ := 0.10 * x
def total_butterfat_in_existing_milk : ℝ := 4
def final_butterfat (x : ℝ) : ℝ := amount_of_butterfat x + total_butterfat_in_existing_milk
def total_milk (x : ℝ) : ℝ := x + 8
def desired_butterfat (x : ℝ) : ℝ := 0.20 * total_milk x

-- Lean proof statement
theorem gallons_10_percent_milk_needed (x : ℝ) : final_butterfat x = desired_butterfat x → x = 24 :=
by
  intros h
  sorry

end gallons_10_percent_milk_needed_l74_74088


namespace largest_integer_solution_l74_74830

theorem largest_integer_solution (x : ℤ) : 
  (x - 3 * (x - 2) ≥ 4) → (2 * x + 1 < x - 1) → (x = -3) :=
by
  sorry

end largest_integer_solution_l74_74830


namespace squido_oysters_l74_74802

theorem squido_oysters (S C : ℕ) (h1 : C ≥ 2 * S) (h2 : S + C = 600) : S = 200 :=
sorry

end squido_oysters_l74_74802


namespace ordered_pairs_count_l74_74452

theorem ordered_pairs_count : 
  (∀ (b c : ℕ), b > 0 ∧ b ≤ 6 ∧ c > 0 ∧ c ≤ 6 ∧ b^2 - 4 * c < 0 ∧ c^2 - 4 * b < 0 → 
  ((b = 1 ∧ (c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 2 ∧ (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 3 ∧ (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 4 ∧ (c = 5 ∨ c = 6)))) ∧
  (∃ (n : ℕ), n = 15) := sorry

end ordered_pairs_count_l74_74452


namespace divisor_of_subtracted_number_l74_74272

theorem divisor_of_subtracted_number (n : ℕ) (m : ℕ) (h : n = 5264 - 11) : Nat.gcd n 5264 = 5253 :=
by
  sorry

end divisor_of_subtracted_number_l74_74272


namespace largest_possible_value_l74_74040

theorem largest_possible_value (X Y Z m: ℕ) 
  (hX_range: 0 ≤ X ∧ X ≤ 4) 
  (hY_range: 0 ≤ Y ∧ Y ≤ 4) 
  (hZ_range: 0 ≤ Z ∧ Z ≤ 4) 
  (h1: m = 25 * X + 5 * Y + Z)
  (h2: m = 81 * Z + 9 * Y + X):
  m = 121 :=
by
  -- The proof goes here
  sorry

end largest_possible_value_l74_74040


namespace points_on_octagon_boundary_l74_74565

def is_on_octagon_boundary (x y : ℝ) : Prop :=
  |x| + |y| + |x - 1| + |y - 1| = 4

theorem points_on_octagon_boundary :
  ∀ (x y : ℝ), is_on_octagon_boundary x y ↔ ((0 ≤ x ∧ x ≤ 1 ∧ (y = 2 ∨ y = -1)) ∨
                                             (0 ≤ y ∧ y ≤ 1 ∧ (x = 2 ∨ x = -1)) ∨
                                             (x ≥ 1 ∧ y ≥ 1 ∧ x + y = 3) ∨
                                             (x ≤ 1 ∧ y ≤ 1 ∧ x + y = 1) ∨
                                             (x ≥ 1 ∧ y ≤ -1 ∧ x + y = 1) ∨
                                             (x ≤ -1 ∧ y ≥ 1 ∧ x + y = 1) ∨
                                             (x ≤ -1 ∧ y ≤ 1 ∧ x + y = -1) ∨
                                             (x ≤ 1 ∧ y ≤ -1 ∧ x + y = -1)) :=
by
  sorry

end points_on_octagon_boundary_l74_74565


namespace candidate_failed_by_45_marks_l74_74584

-- Define the main parameters
def passing_percentage : ℚ := 45 / 100
def candidate_marks : ℝ := 180
def maximum_marks : ℝ := 500
def passing_marks : ℝ := passing_percentage * maximum_marks
def failing_marks : ℝ := passing_marks - candidate_marks

-- State the theorem to be proved
theorem candidate_failed_by_45_marks : failing_marks = 45 := by
  sorry

end candidate_failed_by_45_marks_l74_74584


namespace value_in_half_dollars_percentage_l74_74919

theorem value_in_half_dollars_percentage (n h q : ℕ) (hn : n = 75) (hh : h = 40) (hq : q = 30) : 
  (h * 50 : ℕ) / (n * 5 + h * 50 + q * 25 : ℕ) * 100 = 64 := by
  sorry

end value_in_half_dollars_percentage_l74_74919


namespace claudia_coins_l74_74265

variable (x y : ℕ)

theorem claudia_coins :
  (x + y = 15 ∧ ((145 - 5 * x) / 5) + 1 = 23) → y = 9 :=
by
  intro h
  -- The proof steps would go here, but we'll leave it as sorry for now.
  sorry

end claudia_coins_l74_74265


namespace leak_drain_time_l74_74405

theorem leak_drain_time :
  ∀ (P L : ℝ),
  P = 1/6 →
  P - L = 1/12 →
  (1/L) = 12 :=
by
  intros P L hP hPL
  sorry

end leak_drain_time_l74_74405


namespace total_books_l74_74720

theorem total_books (joan_books : ℕ) (tom_books : ℕ) (h1 : joan_books = 10) (h2 : tom_books = 38) : joan_books + tom_books = 48 :=
by
  -- insert proof here
  sorry

end total_books_l74_74720


namespace tenth_term_is_26_l74_74224

-- Definitions used from the conditions
def first_term : ℤ := 8
def common_difference : ℤ := 2
def term_number : ℕ := 10

-- Define the formula for the nth term of an arithmetic progression
def nth_term (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Proving that the 10th term is 26 given the conditions
theorem tenth_term_is_26 : nth_term first_term common_difference term_number = 26 := by
  sorry

end tenth_term_is_26_l74_74224


namespace distinct_stone_arrangements_l74_74761

-- Define the set of 12 unique stones
def stones := Finset.range 12

-- Define the number of unique placements without considering symmetries
def placements : ℕ := stones.card.factorial

-- Define the number of symmetries (6 rotations and 6 reflections)
def symmetries : ℕ := 12

-- The total number of distinct configurations accounting for symmetries
def distinct_arrangements : ℕ := placements / symmetries

-- The main theorem stating the number of distinct arrangements
theorem distinct_stone_arrangements : distinct_arrangements = 39916800 := by 
  sorry

end distinct_stone_arrangements_l74_74761


namespace money_collected_is_correct_l74_74649

-- Define the conditions as constants and definitions in Lean
def ticket_price_adult : ℝ := 0.60
def ticket_price_child : ℝ := 0.25
def total_persons : ℕ := 280
def children_attended : ℕ := 80

-- Define the number of adults
def adults_attended : ℕ := total_persons - children_attended

-- Define the total money collected
def total_money_collected : ℝ :=
  (adults_attended * ticket_price_adult) + (children_attended * ticket_price_child)

-- Statement to prove
theorem money_collected_is_correct :
  total_money_collected = 140 := by
  sorry

end money_collected_is_correct_l74_74649


namespace axis_of_symmetry_parabola_eq_l74_74341

theorem axis_of_symmetry_parabola_eq : ∀ (x y p : ℝ), 
  y = -2 * x^2 → 
  (x^2 = -2 * p * y) → 
  (p = 1/4) →  
  (y = p / 2) → 
  y = 1 / 8 := by 
  intros x y p h1 h2 h3 h4
  sorry

end axis_of_symmetry_parabola_eq_l74_74341


namespace slope_angle_135_l74_74718

theorem slope_angle_135 (x y : ℝ) : 
  (∃ (m b : ℝ), 3 * x + 3 * y + 1 = 0 ∧ y = m * x + b ∧ m = -1) ↔ 
  (∃ α : ℝ, 0 ≤ α ∧ α < 180 ∧ Real.tan α = -1 ∧ α = 135) :=
sorry

end slope_angle_135_l74_74718


namespace simplify_expression_l74_74600

variable (a : ℚ)
def expression := ((a + 3) / (a - 1) - 1 / (a - 1)) / ((a^2 + 4 * a + 4) / (a^2 - a))

theorem simplify_expression (h : a = 3) : expression a = 3 / 5 :=
by
  rw [h]
  -- additional simplifications would typically go here if the steps were spelled out
  sorry

end simplify_expression_l74_74600


namespace sin_alpha_in_second_quadrant_l74_74618

theorem sin_alpha_in_second_quadrant 
  (α : ℝ) 
  (h1 : π / 2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.tan α = -1 / 2)  -- tan α = -1/2
  : Real.sin α = Real.sqrt 5 / 5 :=
sorry

end sin_alpha_in_second_quadrant_l74_74618


namespace mixed_tea_sale_price_l74_74107

noncomputable def sale_price_of_mixed_tea (weight1 weight2 weight3 price1 price2 price3 profit1 profit2 profit3 : ℝ) : ℝ :=
  let total_cost1 := weight1 * price1
  let total_cost2 := weight2 * price2
  let total_cost3 := weight3 * price3
  let total_profit1 := profit1 * total_cost1
  let total_profit2 := profit2 * total_cost2
  let total_profit3 := profit3 * total_cost3
  let selling_price1 := total_cost1 + total_profit1
  let selling_price2 := total_cost2 + total_profit2
  let selling_price3 := total_cost3 + total_profit3
  let total_selling_price := selling_price1 + selling_price2 + selling_price3
  let total_weight := weight1 + weight2 + weight3
  total_selling_price / total_weight

theorem mixed_tea_sale_price :
  sale_price_of_mixed_tea 120 45 35 30 40 60 0.50 0.30 0.25 = 51.825 :=
by
  sorry

end mixed_tea_sale_price_l74_74107


namespace not_divisible_by_n_l74_74772

theorem not_divisible_by_n (n : ℕ) (h : n > 1) : ¬ (n ∣ (2^n - 1)) :=
by
  sorry

end not_divisible_by_n_l74_74772


namespace hydrochloric_acid_required_l74_74903

-- Define the quantities for the balanced reaction equation
def molesOfAgNO3 : ℕ := 2
def molesOfHNO3 : ℕ := 2
def molesOfHCl : ℕ := 2

-- Define the condition for the reaction (balances the equation)
def balanced_reaction (x y z w : ℕ) : Prop :=
  x = y ∧ x = z ∧ y = w

-- The goal is to prove that the number of moles of HCl needed is 2
theorem hydrochloric_acid_required :
  balanced_reaction molesOfAgNO3 molesOfHCl molesOfHNO3 2 →
  molesOfHCl = 2 :=
by sorry

end hydrochloric_acid_required_l74_74903


namespace solve_system_l74_74580

theorem solve_system :
  ∃ x y : ℝ, (x^2 + 3 * x * y = 18 ∧ x * y + 3 * y^2 = 6) ∧ ((x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = -1)) :=
by
  sorry

end solve_system_l74_74580


namespace xy_sum_l74_74225

variable (x y : ℚ)

theorem xy_sum : (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 := by
  intros h1 h2
  sorry

end xy_sum_l74_74225


namespace function_increment_l74_74750

theorem function_increment (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2 / x) : f 1.5 - f 2 = 1 / 3 := 
by {
  sorry
}

end function_increment_l74_74750


namespace travelers_on_liner_l74_74877

theorem travelers_on_liner (a : ℤ) :
  250 ≤ a ∧ a ≤ 400 ∧ 
  a % 15 = 7 ∧
  a % 25 = 17 →
  a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l74_74877


namespace sarah_daily_candy_consumption_l74_74317

def neighbors_candy : ℕ := 66
def sister_candy : ℕ := 15
def days : ℕ := 9

def total_candy : ℕ := neighbors_candy + sister_candy
def average_daily_consumption : ℕ := total_candy / days

theorem sarah_daily_candy_consumption : average_daily_consumption = 9 := by
  sorry

end sarah_daily_candy_consumption_l74_74317


namespace license_plate_possibilities_count_l74_74355

def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

def digits : Finset Char := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

theorem license_plate_possibilities_count : 
  (vowels.card * digits.card * 2 = 100) := 
by {
  -- vowels.card = 5 because there are 5 vowels.
  -- digits.card = 10 because there are 10 digits.
  -- 2 because the middle character must match either the first vowel or the last digit.
  sorry
}

end license_plate_possibilities_count_l74_74355


namespace card_probability_multiple_l74_74608

def is_multiple_of (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

def count_multiples (n k : ℕ) : ℕ :=
  if k = 0 then 0 else n / k

def inclusion_exclusion (a b c : ℕ) (n : ℕ) : ℕ :=
  (count_multiples n a) + (count_multiples n b) + (count_multiples n c) - 
  (count_multiples n (Nat.lcm a b)) - (count_multiples n (Nat.lcm a c)) - 
  (count_multiples n (Nat.lcm b c)) + 
  count_multiples n (Nat.lcm a (Nat.lcm b c))

theorem card_probability_multiple (n : ℕ) 
  (a b c : ℕ) (hne : n ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (inclusion_exclusion a b c n) / n = 47 / 100 := by
  sorry

end card_probability_multiple_l74_74608


namespace problem1_problem2_problem3_problem4_l74_74994

theorem problem1 : -15 + (-23) - 26 - (-15) = -49 := 
by sorry

theorem problem2 : (- (1 / 2) + (2 / 3) - (1 / 4)) * (-24) = 2 := 
by sorry

theorem problem3 : -24 / (-6) * (- (1 / 4)) = -1 := 
by sorry

theorem problem4 : -1 ^ 2024 - (-2) ^ 3 - 3 ^ 2 + 2 / (2 / 3 * (3 / 2)) = 5 / 2 := 
by sorry

end problem1_problem2_problem3_problem4_l74_74994


namespace sum_fraction_nonnegative_le_one_l74_74079

theorem sum_fraction_nonnegative_le_one 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b + c = 2) :
  a * b / (c^2 + 1) + b * c / (a^2 + 1) + c * a / (b^2 + 1) ≤ 1 :=
sorry

end sum_fraction_nonnegative_le_one_l74_74079


namespace weight_comparison_l74_74016

theorem weight_comparison :
  let weights := [10, 20, 30, 120]
  let average := (10 + 20 + 30 + 120) / 4
  let median := (20 + 30) / 2
  average = 45 ∧ median = 25 ∧ average - median = 20 :=
by
  let weights := [10, 20, 30, 120]
  let average := (10 + 20 + 30 + 120) / 4
  let median := (20 + 30) / 2
  have h1 : average = 45 := sorry
  have h2 : median = 25 := sorry
  have h3 : average - median = 20 := sorry
  exact ⟨h1, h2, h3⟩

end weight_comparison_l74_74016


namespace pool_cleaning_l74_74852

theorem pool_cleaning (full_capacity_liters : ℕ) (percent_full : ℕ) (loss_per_jump_ml : ℕ) 
    (full_capacity : full_capacity_liters = 2000) (trigger_clean : percent_full = 80) 
    (loss_per_jump : loss_per_jump_ml = 400) : 
    let trigger_capacity_liters := (full_capacity_liters * percent_full) / 100
    let splash_out_capacity_liters := full_capacity_liters - trigger_capacity_liters
    let splash_out_capacity_ml := splash_out_capacity_liters * 1000
    (splash_out_capacity_ml / loss_per_jump_ml) = 1000 :=
by {
    sorry
}

end pool_cleaning_l74_74852


namespace sum_of_digits_d_l74_74754

theorem sum_of_digits_d (d : ℕ) (exchange_rate : 10 * d / 7 - 60 = d) : 
  (d = 140) -> (Nat.digits 10 140).sum = 5 :=
by
  sorry

end sum_of_digits_d_l74_74754


namespace molecular_weight_compound_l74_74210

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def num_H : ℝ := 1
def num_Br : ℝ := 1
def num_O : ℝ := 3

def molecular_weight (num_H num_Br num_O atomic_weight_H atomic_weight_Br atomic_weight_O : ℝ) : ℝ :=
  (num_H * atomic_weight_H) + (num_Br * atomic_weight_Br) + (num_O * atomic_weight_O)

theorem molecular_weight_compound : molecular_weight num_H num_Br num_O atomic_weight_H atomic_weight_Br atomic_weight_O = 128.91 :=
by
  sorry

end molecular_weight_compound_l74_74210


namespace count_3_digit_numbers_divisible_by_13_l74_74855

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l74_74855


namespace number_of_terms_in_expanded_polynomial_l74_74992

theorem number_of_terms_in_expanded_polynomial : 
  ∀ (a : Fin 4 → Type) (b : Fin 2 → Type) (c : Fin 3 → Type), 
  (4 * 2 * 3 = 24) := 
by
  intros a b c
  sorry

end number_of_terms_in_expanded_polynomial_l74_74992


namespace determine_c_l74_74437

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem determine_c (a b : ℝ) (m c : ℝ) 
  (h1 : ∀ x, 0 ≤ x → f x a b = x^2 + a * x + b)
  (h2 : ∃ m : ℝ, ∀ x : ℝ, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
sorry

end determine_c_l74_74437


namespace selling_price_correct_l74_74926

noncomputable def cost_price : ℝ := 100
noncomputable def gain_percent : ℝ := 0.15
noncomputable def profit : ℝ := gain_percent * cost_price
noncomputable def selling_price : ℝ := cost_price + profit

theorem selling_price_correct : selling_price = 115 := by
  sorry

end selling_price_correct_l74_74926


namespace brick_wall_completion_time_l74_74423

def rate (hours : ℚ) : ℚ := 1 / hours

/-- Avery can build a brick wall in 3 hours. -/
def avery_rate : ℚ := rate 3
/-- Tom can build a brick wall in 2.5 hours. -/
def tom_rate : ℚ := rate 2.5
/-- Catherine can build a brick wall in 4 hours. -/
def catherine_rate : ℚ := rate 4
/-- Derek can build a brick wall in 5 hours. -/
def derek_rate : ℚ := rate 5

/-- Combined rate for Avery, Tom, and Catherine working together. -/
def combined_rate_1 : ℚ := avery_rate + tom_rate + catherine_rate
/-- Combined rate for Tom and Catherine working together. -/
def combined_rate_2 : ℚ := tom_rate + catherine_rate
/-- Combined rate for Tom, Catherine, and Derek working together. -/
def combined_rate_3 : ℚ := tom_rate + catherine_rate + derek_rate

/-- Total time taken to complete the wall. -/
def total_time (t : ℚ) : Prop :=
  t = 2

theorem brick_wall_completion_time (t : ℚ) : total_time t :=
by
  sorry

end brick_wall_completion_time_l74_74423


namespace remaining_pencils_total_l74_74377

-- Definitions corresponding to the conditions:
def J : ℝ := 300
def J_d : ℝ := 0.30 * J
def J_r : ℝ := J - J_d

def V : ℝ := 2 * J
def V_d : ℝ := 125
def V_r : ℝ := V - V_d

def S : ℝ := 450
def S_d : ℝ := 0.60 * S
def S_r : ℝ := S - S_d

-- Proving the remaining pencils add up to the required amount:
theorem remaining_pencils_total : J_r + V_r + S_r = 865 := by
  sorry

end remaining_pencils_total_l74_74377


namespace difference_in_cans_l74_74591

-- Definitions of the conditions
def total_cans_collected : ℕ := 9
def cans_in_bag : ℕ := 7

-- Statement of the proof problem
theorem difference_in_cans :
  total_cans_collected - cans_in_bag = 2 := by
  sorry

end difference_in_cans_l74_74591


namespace solve_for_y_l74_74821

theorem solve_for_y (x y : ℝ) (h : 3 * x - 5 * y = 7) : y = (3 * x - 7) / 5 :=
sorry

end solve_for_y_l74_74821


namespace duration_of_resulting_video_l74_74697

theorem duration_of_resulting_video 
    (vasya_walk_time : ℕ) (petya_walk_time : ℕ) 
    (sync_meet_point : ℕ) :
    vasya_walk_time = 8 → petya_walk_time = 5 → sync_meet_point = sync_meet_point → 
    (vasya_walk_time - sync_meet_point + petya_walk_time) = 5 :=
by
  intros
  sorry

end duration_of_resulting_video_l74_74697


namespace original_population_multiple_of_5_l74_74018

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem original_population_multiple_of_5 (x y z : ℕ) 
  (H1 : is_perfect_square (x * x)) 
  (H2 : x * x + 200 = y * y) 
  (H3 : y * y + 180 = z * z) : 
  ∃ k : ℕ, x * x = 5 * k := 
sorry

end original_population_multiple_of_5_l74_74018


namespace relationship_among_sets_l74_74524

-- Definitions of the integer sets E, F, and G
def E := {e : ℝ | ∃ m : ℤ, e = m + 1 / 6}
def F := {f : ℝ | ∃ n : ℤ, f = n / 2 - 1 / 3}
def G := {g : ℝ | ∃ p : ℤ, g = p / 2 + 1 / 6}

-- The theorem statement capturing the relationship among E, F, and G
theorem relationship_among_sets : E ⊆ F ∧ F = G := by
  sorry

end relationship_among_sets_l74_74524


namespace intersection_of_sets_l74_74764

theorem intersection_of_sets :
  let A := {1, 2}
  let B := {x : ℝ | x^2 - 3 * x + 2 = 0}
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_sets_l74_74764


namespace sheets_in_a_bundle_l74_74969

variable (B : ℕ) -- Denotes the number of sheets in a bundle

-- Conditions
variable (NumBundles NumBunches NumHeaps : ℕ)
variable (SheetsPerBunch SheetsPerHeap TotalSheets : ℕ)

-- Definitions of given conditions
def numBundles := 3
def numBunches := 2
def numHeaps := 5
def sheetsPerBunch := 4
def sheetsPerHeap := 20
def totalSheets := 114

-- Theorem to prove
theorem sheets_in_a_bundle :
  3 * B + 2 * sheetsPerBunch + 5 * sheetsPerHeap = totalSheets → B = 2 := by
  intro h
  sorry

end sheets_in_a_bundle_l74_74969


namespace problem_statement_l74_74886

-- Define the set of numbers
def num_set := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

-- Conditions
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_multiple (a b : ℕ) : Prop := b ∣ a

-- Problem statement
theorem problem_statement (al bill cal : ℕ) (h_al : al ∈ num_set) (h_bill : bill ∈ num_set) (h_cal : cal ∈ num_set) (h_distinct: distinct al bill cal) : 
  (is_multiple al bill) ∧ (is_multiple bill cal) →
  ∃ (p : ℚ), p = 1 / 190 :=
sorry

end problem_statement_l74_74886


namespace helga_shoe_pairs_l74_74094

theorem helga_shoe_pairs
  (first_store_pairs: ℕ) 
  (second_store_pairs: ℕ) 
  (third_store_pairs: ℕ)
  (fourth_store_pairs: ℕ)
  (h1: first_store_pairs = 7)
  (h2: second_store_pairs = first_store_pairs + 2)
  (h3: third_store_pairs = 0)
  (h4: fourth_store_pairs = 2 * (first_store_pairs + second_store_pairs + third_store_pairs))
  : first_store_pairs + second_store_pairs + third_store_pairs + fourth_store_pairs = 48 := 
by
  sorry

end helga_shoe_pairs_l74_74094


namespace pizza_area_percentage_increase_l74_74368

theorem pizza_area_percentage_increase :
  let r1 := 6
  let r2 := 4
  let A1 := Real.pi * r1^2
  let A2 := Real.pi * r2^2
  let deltaA := A1 - A2
  let N := (deltaA / A2) * 100
  N = 125 := by
  sorry

end pizza_area_percentage_increase_l74_74368


namespace cost_of_hiring_actors_l74_74668

theorem cost_of_hiring_actors
  (A : ℕ)
  (CostOfFood : ℕ := 150)
  (EquipmentRental : ℕ := 300 + 2 * A)
  (TotalCost : ℕ := 3 * A + 450)
  (SellingPrice : ℕ := 10000)
  (Profit : ℕ := 5950) :
  TotalCost = SellingPrice - Profit → A = 1200 :=
by
  intro h
  sorry

end cost_of_hiring_actors_l74_74668


namespace candy_in_each_box_l74_74578

theorem candy_in_each_box (C K : ℕ) (h1 : 6 * C + 4 * K = 90) (h2 : C = K) : C = 9 :=
by
  -- Proof will go here
  sorry

end candy_in_each_box_l74_74578


namespace zhen_zhen_test_score_l74_74178

theorem zhen_zhen_test_score
  (avg1 avg2 : ℝ) (n m : ℝ)
  (h1 : avg1 = 88)
  (h2 : avg2 = 90)
  (h3 : n = 4)
  (h4 : m = 5) :
  avg2 * m - avg1 * n = 98 :=
by
  -- Given the hypotheses h1, h2, h3, and h4,
  -- we need to show that avg2 * m - avg1 * n = 98.
  sorry

end zhen_zhen_test_score_l74_74178


namespace sandwiches_lunch_monday_l74_74123

-- Define the conditions
variables (L : ℕ) 
variables (sandwiches_monday sandwiches_tuesday : ℕ)
variables (h1 : sandwiches_monday = L + 2 * L)
variables (h2 : sandwiches_tuesday = 1)

-- Define the fact that he ate 8 more sandwiches on Monday compared to Tuesday.
variables (h3 : sandwiches_monday = sandwiches_tuesday + 8)

theorem sandwiches_lunch_monday : L = 3 := 
by
  -- We need to prove L = 3 given the conditions (h1, h2, h3)
  -- Here is where the necessary proof would be constructed
  -- This placeholder indicates a proof needs to be inserted here
  sorry

end sandwiches_lunch_monday_l74_74123


namespace rectangle_problem_l74_74615

def rectangle_perimeter (L B : ℕ) : ℕ :=
  2 * (L + B)

theorem rectangle_problem (L B : ℕ) (h1 : L - B = 23) (h2 : L * B = 2520) : rectangle_perimeter L B = 206 := by
  sorry

end rectangle_problem_l74_74615


namespace problem1_problem2_l74_74957

theorem problem1 (x : ℝ) : (x + 3) * (x - 1) ≤ 0 ↔ -3 ≤ x ∧ x ≤ 1 :=
sorry

theorem problem2 (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 :=
sorry

end problem1_problem2_l74_74957


namespace subtract_value_is_34_l74_74512

theorem subtract_value_is_34 
    (x y : ℤ) 
    (h1 : (x - 5) / 7 = 7) 
    (h2 : (x - y) / 10 = 2) : 
    y = 34 := 
sorry

end subtract_value_is_34_l74_74512


namespace investment_interest_min_l74_74895

theorem investment_interest_min (x y : ℝ) (hx : x + y = 25000) (hmax : x ≤ 11000) : 
  0.07 * x + 0.12 * y ≥ 2450 :=
by
  sorry

end investment_interest_min_l74_74895


namespace distinct_m_count_l74_74700

noncomputable def countDistinctMValues : Nat :=
  let pairs := [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6), 
                (-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)]
  let ms := pairs.map (λ p => p.1 + p.2)
  ms.eraseDups.length

theorem distinct_m_count :
  countDistinctMValues = 10 := sorry

end distinct_m_count_l74_74700


namespace calculation_correct_l74_74352

theorem calculation_correct (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b :=
by sorry

end calculation_correct_l74_74352


namespace factorization_eq_l74_74234

theorem factorization_eq (x : ℝ) : 
  -3 * x^3 + 12 * x^2 - 12 * x = -3 * x * (x - 2)^2 :=
by
  sorry

end factorization_eq_l74_74234


namespace sum_a1_to_a5_l74_74325

-- Define the conditions
def equation_holds (x a0 a1 a2 a3 a4 a5 : ℝ) : Prop :=
  x^5 + 2 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5

-- State the theorem
theorem sum_a1_to_a5 (a0 a1 a2 a3 a4 a5 : ℝ) (h : ∀ x : ℝ, equation_holds x a0 a1 a2 a3 a4 a5) :
  a1 + a2 + a3 + a4 + a5 = 31 :=
by
  sorry

end sum_a1_to_a5_l74_74325


namespace solution_set_of_quadratic_l74_74396

theorem solution_set_of_quadratic (a b x : ℝ) (h1 : a = 5) (h2 : b = -6) :
  (2 ≤ x ∧ x ≤ 3) → (bx^2 - ax - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end solution_set_of_quadratic_l74_74396


namespace probability_blue_ball_l74_74391

-- Define the probabilities of drawing a red and yellow ball
def P_red : ℝ := 0.48
def P_yellow : ℝ := 0.35

-- Define the total probability formula in this sample space
def total_probability (P_red P_yellow P_blue : ℝ) : Prop :=
  P_red + P_yellow + P_blue = 1

-- The theorem we need to prove
theorem probability_blue_ball :
  ∃ P_blue : ℝ, total_probability P_red P_yellow P_blue ∧ P_blue = 0.17 :=
sorry

end probability_blue_ball_l74_74391


namespace segments_form_pentagon_l74_74515

theorem segments_form_pentagon (a b c d e : ℝ) 
  (h_sum : a + b + c + d + e = 2)
  (h_a : a > 1/10)
  (h_b : b > 1/10)
  (h_c : c > 1/10)
  (h_d : d > 1/10)
  (h_e : e > 1/10) :
  a + b + c + d > e ∧ a + b + c + e > d ∧ a + b + d + e > c ∧ a + c + d + e > b ∧ b + c + d + e > a := 
sorry

end segments_form_pentagon_l74_74515


namespace buckets_needed_to_fill_tank_l74_74056

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem buckets_needed_to_fill_tank :
  let radius_tank := 8
  let height_tank := 32
  let radius_bucket := 8
  let volume_bucket := volume_of_sphere radius_bucket
  let volume_tank := volume_of_cylinder radius_tank height_tank
  volume_tank / volume_bucket = 3 :=
by sorry

end buckets_needed_to_fill_tank_l74_74056


namespace final_amount_correct_l74_74435

def wallet_cost : ℝ := 22
def purse_cost : ℝ := 4 * wallet_cost - 3
def shoes_cost : ℝ := wallet_cost + purse_cost + 7
def total_cost_before_discount : ℝ := wallet_cost + purse_cost + shoes_cost
def discount_rate : ℝ := 0.10
def discounted_amount : ℝ := total_cost_before_discount * discount_rate
def final_amount : ℝ := total_cost_before_discount - discounted_amount

theorem final_amount_correct :
  final_amount = 198.90 := by
  -- Here we would provide the proof of the theorem
  sorry

end final_amount_correct_l74_74435


namespace jenny_spent_625_dollars_l74_74832

def adoption_fee := 50
def vet_visits_cost := 500
def monthly_food_cost := 25
def toys_cost := 200
def year_months := 12

def jenny_adoption_vet_share := (adoption_fee + vet_visits_cost) / 2
def jenny_food_share := (monthly_food_cost * year_months) / 2
def jenny_total_cost := jenny_adoption_vet_share + jenny_food_share + toys_cost

theorem jenny_spent_625_dollars :
  jenny_total_cost = 625 := by
  sorry

end jenny_spent_625_dollars_l74_74832


namespace sum_of_integers_990_l74_74015

theorem sum_of_integers_990 :
  ∃ (n m : ℕ), (n * (n + 1) = 990 ∧ (m - 1) * m * (m + 1) = 990 ∧ (n + n + 1 + m - 1 + m + m + 1 = 90)) :=
sorry

end sum_of_integers_990_l74_74015


namespace furniture_definition_based_on_vocabulary_study_l74_74949

theorem furniture_definition_based_on_vocabulary_study (term : String) (h : term = "furniture") :
  term = "furniture" :=
by
  sorry

end furniture_definition_based_on_vocabulary_study_l74_74949


namespace quadratic_no_real_solution_l74_74704

theorem quadratic_no_real_solution (a : ℝ) : (∀ x : ℝ, x^2 - x + a ≠ 0) → a > 1 / 4 :=
by
  intro h
  sorry

end quadratic_no_real_solution_l74_74704


namespace triangle_at_most_one_obtuse_l74_74943

theorem triangle_at_most_one_obtuse (A B C : ℝ) (h1 : 0 < A ∧ A < 180) (h2 : 0 < B ∧ B < 180) (h3 : 0 < C ∧ C < 180) (h4 : A + B + C = 180) : A ≤ 90 ∨ B ≤ 90 ∨ C ≤ 90 :=
by
  sorry

end triangle_at_most_one_obtuse_l74_74943


namespace find_number_of_terms_l74_74952

theorem find_number_of_terms (n : ℕ) (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a n = (2^n - 1) / (2^n)) → S n = 321 / 64 → n = 6 :=
by
  sorry

end find_number_of_terms_l74_74952


namespace simplify_abs_expression_l74_74168

theorem simplify_abs_expression (a b c : ℝ) (h1 : a + c > b) (h2 : b + c > a) (h3 : a + b > c) :
  |a - b + c| - |a - b - c| = 2 * a - 2 * b :=
by
  sorry

end simplify_abs_expression_l74_74168


namespace tickets_spent_dunk_a_clown_booth_l74_74109

/-
The conditions given:
1. Tom bought 40 tickets.
2. Tom went on 3 rides.
3. Each ride costs 4 tickets.
-/
def total_tickets : ℕ := 40
def rides_count : ℕ := 3
def tickets_per_ride : ℕ := 4

/-
We aim to prove that Tom spent 28 tickets at the 'dunk a clown' booth.
-/
theorem tickets_spent_dunk_a_clown_booth :
  (total_tickets - rides_count * tickets_per_ride) = 28 :=
by
  sorry

end tickets_spent_dunk_a_clown_booth_l74_74109


namespace tangent_line_at_five_l74_74257

variable {f : ℝ → ℝ}

theorem tangent_line_at_five 
  (h_tangent : ∀ x, f x = -x + 8)
  (h_tangent_deriv : deriv f 5 = -1) :
  f 5 = 3 ∧ deriv f 5 = -1 :=
by sorry

end tangent_line_at_five_l74_74257


namespace factorization_of_x10_minus_1024_l74_74413

theorem factorization_of_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2 * x^3 + 4 * x^2 + 8 * x + 16) :=
by sorry

end factorization_of_x10_minus_1024_l74_74413


namespace area_excluding_hole_l74_74770

open Polynomial

theorem area_excluding_hole (x : ℝ) : 
  ((x^2 + 7) * (x^2 + 5)) - ((2 * x^2 - 3) * (x^2 - 2)) = -x^4 + 19 * x^2 + 29 :=
by
  sorry

end area_excluding_hole_l74_74770


namespace hyperbola_eccentricity_l74_74263

theorem hyperbola_eccentricity :
  let a := 2
  let b := 2 * Real.sqrt 2
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (e = Real.sqrt 3) :=
by {
  sorry
}

end hyperbola_eccentricity_l74_74263


namespace total_seeds_grace_can_plant_l74_74874

theorem total_seeds_grace_can_plant :
  let lettuce_seeds_per_row := 25
  let carrot_seeds_per_row := 20
  let radish_seeds_per_row := 30
  let large_bed_rows_limit := 5
  let medium_bed_rows_limit := 3
  let small_bed_rows_limit := 2
  let large_beds := 2
  let medium_beds := 2
  let small_bed := 1
  let large_bed_planting := 
    [(3, lettuce_seeds_per_row), (2, carrot_seeds_per_row)]  -- 3 rows of lettuce, 2 rows of carrots in large beds
  let medium_bed_planting := 
    [(1, lettuce_seeds_per_row), (1, carrot_seeds_per_row), (1, radish_seeds_per_row)] --in medium beds
  let small_bed_planting := 
    [(1, carrot_seeds_per_row), (1, radish_seeds_per_row)] --in small beds
  (3 * lettuce_seeds_per_row + 2 * carrot_seeds_per_row) * large_beds +
  (1 * lettuce_seeds_per_row + 1 * carrot_seeds_per_row + 1 * radish_seeds_per_row) * medium_beds +
  (1 * carrot_seeds_per_row + 1 * radish_seeds_per_row) * small_bed = 430 :=
by
  sorry

end total_seeds_grace_can_plant_l74_74874


namespace age_of_replaced_person_l74_74947

theorem age_of_replaced_person (avg_age x : ℕ) (h1 : 10 * avg_age - 10 * (avg_age - 3) = x - 18) : x = 48 := 
by
  -- The proof goes here, but we are omitting it as per instruction.
  sorry

end age_of_replaced_person_l74_74947


namespace find_square_sum_l74_74126

theorem find_square_sum :
  ∃ a b c : ℕ, a = 2494651 ∧ b = 1385287 ∧ c = 9406087 ∧ (a + b + c = 3645^2) :=
by
  have h1 : 2494651 + 1385287 + 9406087 = 13286025 := by norm_num
  have h2 : 3645^2 = 13286025 := by norm_num
  exact ⟨2494651, 1385287, 9406087, rfl, rfl, rfl, h2⟩

end find_square_sum_l74_74126


namespace sandy_jacket_price_l74_74337

noncomputable def discounted_shirt_price (initial_shirt_price discount_percentage : ℝ) : ℝ :=
  initial_shirt_price - (initial_shirt_price * discount_percentage / 100)

noncomputable def money_left (initial_money additional_money discounted_price : ℝ) : ℝ :=
  initial_money + additional_money - discounted_price

noncomputable def jacket_price_before_tax (remaining_money tax_percentage : ℝ) : ℝ :=
  remaining_money / (1 + tax_percentage / 100)

theorem sandy_jacket_price :
  let initial_money := 13.99
  let initial_shirt_price := 12.14
  let discount_percentage := 5.0
  let additional_money := 7.43
  let tax_percentage := 10.0
  
  let discounted_price := discounted_shirt_price initial_shirt_price discount_percentage
  let remaining_money := money_left initial_money additional_money discounted_price
  
  jacket_price_before_tax remaining_money tax_percentage = 8.99 := sorry

end sandy_jacket_price_l74_74337


namespace distance_center_is_12_l74_74217

-- Define the side length of the square and the radius of the circle
def side_length_square : ℝ := 5
def radius_circle : ℝ := 1

-- The center path forms a smaller square inside the original square
-- with side length 3 units
def side_length_smaller_square : ℝ := side_length_square - 2 * radius_circle

-- The perimeter of the smaller square, which is the path length that
-- the center of the circle travels
def distance_center_travel : ℝ := 4 * side_length_smaller_square

-- Prove that the distance traveled by the center of the circle is 12 units
theorem distance_center_is_12 : distance_center_travel = 12 := by
  -- the proof is skipped
  sorry

end distance_center_is_12_l74_74217


namespace perimeter_of_square_C_l74_74604

theorem perimeter_of_square_C (s_A s_B s_C : ℝ)
  (h1 : 4 * s_A = 16)
  (h2 : 4 * s_B = 32)
  (h3 : s_C = s_B - s_A) :
  4 * s_C = 16 :=
by
  sorry

end perimeter_of_square_C_l74_74604


namespace probability_of_x_greater_than_3y_l74_74944

noncomputable def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height
noncomputable def area_rectangle (width height : ℝ) : ℝ := width * height
noncomputable def probability (area_triangle area_rectangle : ℝ) : ℝ := area_triangle / area_rectangle

theorem probability_of_x_greater_than_3y :
  let base := 2010
  let height_triangle := base / 3
  let height_rectangle := 2011
  let area_triangle := area_triangle base height_triangle
  let area_rectangle := area_rectangle base height_rectangle
  let prob := probability area_triangle area_rectangle
  prob = (335 : ℝ) / 2011 :=
by
  sorry

end probability_of_x_greater_than_3y_l74_74944


namespace Anne_weight_l74_74751

-- Define variables
def Douglas_weight : ℕ := 52
def weight_difference : ℕ := 15

-- Theorem to prove
theorem Anne_weight : Douglas_weight + weight_difference = 67 :=
by sorry

end Anne_weight_l74_74751


namespace frog_climbs_out_l74_74179

theorem frog_climbs_out (d climb slip : ℕ) (h : d = 20) (h_climb : climb = 3) (h_slip : slip = 2) :
  ∃ n : ℕ, n = 20 ∧ d ≤ n * (climb - slip) + climb :=
sorry

end frog_climbs_out_l74_74179


namespace number_of_teachers_l74_74746

-- Definitions from the problem conditions
def num_students : Nat := 1500
def classes_per_student : Nat := 6
def classes_per_teacher : Nat := 5
def students_per_class : Nat := 25

-- The proof problem statement
theorem number_of_teachers : 
  (num_students * classes_per_student / students_per_class) / classes_per_teacher = 72 := by
  sorry

end number_of_teachers_l74_74746


namespace problem_solution_l74_74735

theorem problem_solution (x y z : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) (h3 : 0.6 * y = z) : 
  z = 60 := by
  sorry

end problem_solution_l74_74735


namespace find_m_for_perfect_square_trinomial_l74_74025

theorem find_m_for_perfect_square_trinomial :
  ∃ m : ℤ, (∀ (x y : ℝ), (9 * x^2 + m * x * y + 16 * y^2 = (3 * x + 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (3 * x - 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (-3 * x + 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (-3 * x - 4 * y)^2)) ↔ 
          (m = 24 ∨ m = -24) := 
by
  sorry

end find_m_for_perfect_square_trinomial_l74_74025


namespace more_trees_died_than_survived_l74_74828

def haley_trees : ℕ := 14
def died_in_typhoon : ℕ := 9
def survived_trees := haley_trees - died_in_typhoon

theorem more_trees_died_than_survived : (died_in_typhoon - survived_trees) = 4 := by
  -- proof goes here
  sorry

end more_trees_died_than_survived_l74_74828


namespace units_digit_of_j_squared_plus_3_power_j_l74_74636

def j : ℕ := 2023^3 + 3^2023 + 2023

theorem units_digit_of_j_squared_plus_3_power_j (j : ℕ) (h : j = 2023^3 + 3^2023 + 2023) : 
  ((j^2 + 3^j) % 10) = 6 := 
  sorry

end units_digit_of_j_squared_plus_3_power_j_l74_74636


namespace jeffrey_walks_to_mailbox_l74_74010

theorem jeffrey_walks_to_mailbox :
  ∀ (D total_steps net_gain_per_set steps_per_set sets net_gain : ℕ),
    steps_per_set = 3 ∧ 
    net_gain = 1 ∧ 
    total_steps = 330 ∧ 
    net_gain_per_set = net_gain ∧ 
    sets = total_steps / steps_per_set ∧ 
    D = sets * net_gain →
    D = 110 :=
by
  intro D total_steps net_gain_per_set steps_per_set sets net_gain
  intro h
  sorry

end jeffrey_walks_to_mailbox_l74_74010


namespace benny_missed_games_l74_74204

theorem benny_missed_games (total_games attended_games missed_games : ℕ)
  (H1 : total_games = 39)
  (H2 : attended_games = 14)
  (H3 : missed_games = total_games - attended_games) :
  missed_games = 25 :=
by
  sorry

end benny_missed_games_l74_74204


namespace greatest_common_length_cords_l74_74601

theorem greatest_common_length_cords (l1 l2 l3 l4 : ℝ) (h1 : l1 = Real.sqrt 20) (h2 : l2 = Real.pi) (h3 : l3 = Real.exp 1) (h4 : l4 = Real.sqrt 98) : 
  ∃ d : ℝ, d = 1 ∧ (∀ k1 k2 k3 k4 : ℝ, k1 * d = l1 → k2 * d = l2 → k3 * d = l3 → k4 * d = l4 → ∀i : ℝ, i = d) :=
by
  sorry

end greatest_common_length_cords_l74_74601


namespace students_in_both_band_and_chorus_l74_74677

-- Definitions of conditions
def total_students := 250
def band_students := 90
def chorus_students := 120
def band_or_chorus_students := 180

-- Theorem statement to prove the number of students in both band and chorus
theorem students_in_both_band_and_chorus : 
  (band_students + chorus_students - band_or_chorus_students) = 30 := 
by sorry

end students_in_both_band_and_chorus_l74_74677


namespace sqrt_expression_l74_74582

theorem sqrt_expression : 2 * Real.sqrt 3 - (3 * Real.sqrt 2 + Real.sqrt 3) = Real.sqrt 3 - 3 * Real.sqrt 2 :=
by
  sorry

end sqrt_expression_l74_74582


namespace conic_section_is_hyperbola_l74_74113

theorem conic_section_is_hyperbola :
  ∀ (x y : ℝ), x^2 - 16 * y^2 - 8 * x + 16 * y + 32 = 0 → 
               (∃ h k a b : ℝ, h = 4 ∧ k = 0.5 ∧ a = b ∧ a^2 = 2 ∧ b^2 = 2) :=
by
  sorry

end conic_section_is_hyperbola_l74_74113


namespace k_value_l74_74118

theorem k_value {x y k : ℝ} (h : ∃ c : ℝ, (x ^ 2 + k * x * y + 49 * y ^ 2) = c ^ 2) : k = 14 ∨ k = -14 :=
by sorry

end k_value_l74_74118


namespace equal_numbers_in_sequence_l74_74205

theorem equal_numbers_in_sequence (a : ℕ → ℚ)
  (h : ∀ m n : ℕ, a m + a n = a (m * n)) : 
  ∃ i j : ℕ, i ≠ j ∧ a i = a j :=
sorry

end equal_numbers_in_sequence_l74_74205


namespace pen_average_price_l74_74077

theorem pen_average_price (pens_purchased pencils_purchased : ℕ) (total_cost pencil_avg_price : ℝ)
  (H0 : pens_purchased = 30) (H1 : pencils_purchased = 75) 
  (H2 : total_cost = 690) (H3 : pencil_avg_price = 2) :
  (total_cost - (pencils_purchased * pencil_avg_price)) / pens_purchased = 18 :=
by
  rw [H0, H1, H2, H3]
  sorry

end pen_average_price_l74_74077


namespace solve_equation_l74_74660

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_equation:
    (7.331 * ((log_base 3 x - 1) / (log_base 3 (x / 3))) - 
    2 * (log_base 3 (Real.sqrt x)) + (log_base 3 x)^2 = 3) → 
    (x = 1 / 3 ∨ x = 9) := by
  sorry

end solve_equation_l74_74660


namespace max_sum_e3_f3_g3_h3_i3_l74_74273

theorem max_sum_e3_f3_g3_h3_i3 (e f g h i : ℝ) (h_cond : e^4 + f^4 + g^4 + h^4 + i^4 = 5) :
  e^3 + f^3 + g^3 + h^3 + i^3 ≤ 5^(3/4) :=
sorry

end max_sum_e3_f3_g3_h3_i3_l74_74273


namespace second_closest_location_l74_74127
-- Import all necessary modules from the math library

-- Define the given distances (conditions)
def distance_library : ℝ := 1.912 * 1000  -- distance in meters
def distance_park : ℝ := 876              -- distance in meters
def distance_clothing_store : ℝ := 1.054 * 1000  -- distance in meters

-- State the proof problem
theorem second_closest_location :
  (distance_library = 1912) →
  (distance_park = 876) →
  (distance_clothing_store = 1054) →
  (distance_clothing_store = 1054) :=
by
  intros h1 h2 h3
  -- sorry to skip the proof
  sorry

end second_closest_location_l74_74127


namespace ferry_tourist_total_l74_74940

theorem ferry_tourist_total :
  let number_of_trips := 8
  let a := 120 -- initial number of tourists
  let d := -2  -- common difference
  let total_tourists := (number_of_trips * (2 * a + (number_of_trips - 1) * d)) / 2
  total_tourists = 904 := 
by {
  sorry
}

end ferry_tourist_total_l74_74940


namespace remove_remaining_wallpaper_time_l74_74893

noncomputable def time_per_wall : ℕ := 2
noncomputable def walls_dining_room : ℕ := 4
noncomputable def walls_living_room : ℕ := 4
noncomputable def walls_completed : ℕ := 1

theorem remove_remaining_wallpaper_time : 
    time_per_wall * (walls_dining_room - walls_completed) + time_per_wall * walls_living_room = 14 :=
by
  sorry

end remove_remaining_wallpaper_time_l74_74893


namespace general_term_a_l74_74028

noncomputable def S (n : ℕ) : ℤ := 3^n - 2

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2 * 3^(n - 1)

theorem general_term_a (n : ℕ) (hn : n > 0) : a n = if n = 1 then 1 else 2 * 3^(n - 1) := by
  -- Proof goes here
  sorry

end general_term_a_l74_74028


namespace two_digit_integer_divides_491_remainder_59_l74_74201

theorem two_digit_integer_divides_491_remainder_59 :
  ∃ n Q : ℕ, (n = 10 * x + y) ∧ (0 < x) ∧ (x ≤ 9) ∧ (0 ≤ y) ∧ (y ≤ 9) ∧ (491 = n * Q + 59) ∧ (n = 72) :=
by
  sorry

end two_digit_integer_divides_491_remainder_59_l74_74201


namespace simplify_and_evaluate_l74_74131

theorem simplify_and_evaluate :
  ∀ (a : ℚ), a = 3 → ((a - 1) / (a + 2) / ((a ^ 2 - 2 * a) / (a ^ 2 - 4)) - (a + 1) / a) = -2 / 3 :=
by
  intros a ha
  have : a = 3 := ha
  sorry

end simplify_and_evaluate_l74_74131


namespace xuzhou_test_2014_l74_74753

variables (A B C D : ℝ) -- Assume A, B, C, D are real numbers.

theorem xuzhou_test_2014 :
  (C < D) → (A > B) :=
sorry

end xuzhou_test_2014_l74_74753


namespace minimum_pawns_remaining_l74_74540

-- Define the initial placement and movement conditions
structure Chessboard :=
  (white_pawns : ℕ)
  (black_pawns : ℕ)
  (on_board : ℕ)

def valid_placement (cb : Chessboard) : Prop :=
  cb.white_pawns = 32 ∧ cb.black_pawns = 32 ∧ cb.on_board = 64

def can_capture (player_pawn : ℕ → ℕ → Prop) := 
  ∀ (wp bp : ℕ), 
  wp ≥ 0 ∧ bp ≥ 0 ∧ wp + bp = 64 →
  ∀ (p_wp p_bp : ℕ), 
  player_pawn wp p_wp ∧ player_pawn bp p_bp →
  p_wp + p_bp ≥ 2
  
-- Our theorem to prove
theorem minimum_pawns_remaining (cb : Chessboard) (player_pawn : ℕ → ℕ → Prop) :
  valid_placement cb →
  can_capture player_pawn →
  ∃ min_pawns : ℕ, min_pawns = 2 :=
by
  sorry

end minimum_pawns_remaining_l74_74540


namespace abs_a_gt_neg_b_l74_74099

variable {a b : ℝ}

theorem abs_a_gt_neg_b (h : a < b ∧ b < 0) : |a| > -b :=
by
  sorry

end abs_a_gt_neg_b_l74_74099


namespace set_C_cannot_form_right_triangle_l74_74670

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem set_C_cannot_form_right_triangle :
  ¬ is_right_triangle 7 8 9 :=
by
  sorry

end set_C_cannot_form_right_triangle_l74_74670


namespace find_two_digit_integers_l74_74030

theorem find_two_digit_integers :
  ∃ (m n : ℕ), 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 ∧
    (∃ (a b : ℚ), a = m ∧ b = n ∧ (a + b) / 2 = b + a / 100) ∧ (m + n < 150) ∧ m = 50 ∧ n = 49 := 
by
  sorry

end find_two_digit_integers_l74_74030


namespace richard_older_than_david_by_l74_74602

-- Definitions based on given conditions

def richard : ℕ := sorry
def david : ℕ := 14 -- David is 14 years old.
def scott : ℕ := david - 8 -- Scott is 8 years younger than David.

-- In 8 years, Richard will be twice as old as Scott
axiom richard_in_8_years : richard + 8 = 2 * (scott + 8)

-- To prove: How many years older is Richard than David?
theorem richard_older_than_david_by : richard - david = 6 := sorry

end richard_older_than_david_by_l74_74602


namespace negation_proposition_l74_74071

theorem negation_proposition (x : ℝ) :
  ¬(∀ x : ℝ, x^2 - x + 3 > 0) ↔ ∃ x : ℝ, x^2 - x + 3 ≤ 0 := 
by { sorry }

end negation_proposition_l74_74071


namespace roots_of_cubic_l74_74758

/-- Let p, q, and r be the roots of the polynomial x^3 - 15x^2 + 10x + 24 = 0. 
   The value of (1 + p)(1 + q)(1 + r) is equal to 2. -/
theorem roots_of_cubic (p q r : ℝ)
  (h1 : p + q + r = 15)
  (h2 : p * q + q * r + r * p = 10)
  (h3 : p * q * r = -24) :
  (1 + p) * (1 + q) * (1 + r) = 2 := 
by 
  sorry

end roots_of_cubic_l74_74758


namespace quadratic_real_roots_l74_74547

theorem quadratic_real_roots (a b c : ℝ) (h : a * c < 0) : 
  ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y :=
by
  sorry

end quadratic_real_roots_l74_74547


namespace algebraic_expression_evaluates_to_2_l74_74737

theorem algebraic_expression_evaluates_to_2 (x : ℝ) (h : x^2 + x - 5 = 0) : 
(x - 1)^2 - x * (x - 3) + (x + 2) * (x - 2) = 2 := 
by 
  sorry

end algebraic_expression_evaluates_to_2_l74_74737


namespace angle_sum_triangle_l74_74357

theorem angle_sum_triangle (x : ℝ) 
  (h1 : 70 + 70 + x = 180) : 
  x = 40 :=
by
  sorry

end angle_sum_triangle_l74_74357


namespace circle_equation_l74_74124

theorem circle_equation 
  (circle_eq : ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = (x - 3)^2 + (y - 2)^2) 
  (tangent_to_line : ∀ (x y : ℝ), (2*x - y + 5) = 0 → 
    (x = -2 ∧ y = 1))
  (passes_through_N : ∀ (x y : ℝ), (x = 3 ∧ y = 2)) :
  ∀ (x y : ℝ), x^2 + y^2 - 9*x + (9/2)*y - (55/2) = 0 := 
sorry

end circle_equation_l74_74124


namespace largest_multiple_negation_greater_than_neg150_l74_74401

theorem largest_multiple_negation_greater_than_neg150 (n : ℤ) (h₁ : n % 6 = 0) (h₂ : -n > -150) : n = 144 := 
sorry

end largest_multiple_negation_greater_than_neg150_l74_74401


namespace quadratic_other_x_intercept_l74_74894

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → a * x^2 + b * x + c = -3)
  (h_intercept : a * 1^2 + b * 1 + c = 0) : 
  ∃ x0 : ℝ, x0 = 9 ∧ a * x0^2 + b * x0 + c = 0 :=
by
  sorry

end quadratic_other_x_intercept_l74_74894


namespace num_of_lists_is_correct_l74_74254

theorem num_of_lists_is_correct :
  let num_balls := 15
  let num_selections := 4
  let total_lists := num_balls ^ num_selections
  total_lists = 50625 :=
by
  let num_balls := 15
  let num_selections := 4
  let total_lists := num_balls ^ num_selections
  show total_lists = 50625
  sorry

end num_of_lists_is_correct_l74_74254


namespace find_product_of_roots_plus_one_l74_74012

-- Define the problem conditions
variables (x1 x2 : ℝ)
axiom sum_roots : x1 + x2 = 3
axiom prod_roots : x1 * x2 = 2

-- State the theorem corresponding to the proof problem
theorem find_product_of_roots_plus_one : (x1 + 1) * (x2 + 1) = 6 :=
by 
  sorry

end find_product_of_roots_plus_one_l74_74012


namespace infinitely_many_sum_form_l74_74425

theorem infinitely_many_sum_form {a : ℕ → ℕ} (h : ∀ n, a n < a (n + 1)) :
  ∀ i, ∃ᶠ n in at_top, ∃ r s j, r > 0 ∧ s > 0 ∧ i < j ∧ a n = r * a i + s * a j := 
by
  sorry

end infinitely_many_sum_form_l74_74425


namespace range_of_k_for_one_solution_l74_74561

-- Definitions
def angle_B : ℝ := 60 -- Angle B in degrees
def side_b : ℝ := 12 -- Length of side b
def side_a (k : ℝ) : ℝ := k -- Length of side a (parameterized by k)

-- Theorem stating the range of k that makes the side_a have exactly one solution
theorem range_of_k_for_one_solution (k : ℝ) : (0 < k ∧ k <= 12) ∨ k = 8 * Real.sqrt 3 := 
sorry

end range_of_k_for_one_solution_l74_74561


namespace total_pieces_of_bread_correct_l74_74991

-- Define the constants for the number of bread pieces needed per type of sandwich
def pieces_per_regular_sandwich : ℕ := 2
def pieces_per_double_meat_sandwich : ℕ := 3

-- Define the quantities of each type of sandwich
def regular_sandwiches : ℕ := 14
def double_meat_sandwiches : ℕ := 12

-- Define the total pieces of bread calculation
def total_pieces_of_bread : ℕ := pieces_per_regular_sandwich * regular_sandwiches + pieces_per_double_meat_sandwich * double_meat_sandwiches

-- State the theorem
theorem total_pieces_of_bread_correct : total_pieces_of_bread = 64 :=
by
  -- Proof goes here (using sorry for now)
  sorry

end total_pieces_of_bread_correct_l74_74991


namespace find_d_minus_r_l74_74621

theorem find_d_minus_r :
  ∃ (d r : ℕ), d > 1 ∧ 1083 % d = r ∧ 1455 % d = r ∧ 2345 % d = r ∧ d - r = 1 := by
  sorry

end find_d_minus_r_l74_74621


namespace find_a_l74_74140

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x >= 0 then a^x else a^(-x)

theorem find_a (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a)
(h_ge : ∀ x : ℝ, x >= 0 → f x a = a ^ x)
(h_a_gt_1 : a > 1)
(h_sol : ∀ x : ℝ, f x a ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2) :
a = 2 :=
sorry

end find_a_l74_74140


namespace trading_cards_initial_total_l74_74255

theorem trading_cards_initial_total (x : ℕ) 
  (h1 : ∃ d : ℕ, d = (1 / 3 : ℕ) * x)
  (h2 : ∃ n1 : ℕ, n1 = (1 / 5 : ℕ) * (1 / 3 : ℕ) * x)
  (h3 : ∃ n2 : ℕ, n2 = (1 / 3 : ℕ) * ((1 / 5 : ℕ) * (1 / 3 : ℕ) * x))
  (h4 : ∃ n3 : ℕ, n3 = (1 / 2 : ℕ) * (2 / 45 : ℕ) * x)
  (h5 : (1 / 15 : ℕ) * x + (2 / 45 : ℕ) * x + (1 / 45 : ℕ) * x = 850) :
  x = 6375 := 
sorry

end trading_cards_initial_total_l74_74255


namespace son_l74_74087

theorem son's_age (F S : ℕ) (h1 : F + S = 75) (h2 : F = 8 * (S - (F - S))) : S = 27 :=
sorry

end son_l74_74087


namespace math_problem_l74_74902

-- Conditions
variables {f g : ℝ → ℝ}
axiom f_zero : f 0 = 0
axiom inequality : ∀ x y : ℝ, g (x - y) ≥ f x * f y + g x * g y

-- Problem Statement
theorem math_problem : ∀ x : ℝ, f x ^ 2008 + g x ^ 2008 ≤ 1 :=
by
  sorry

end math_problem_l74_74902


namespace problem1_problem2_l74_74187

theorem problem1 : (-5 : ℝ) ^ 0 - (1 / 3) ^ (-2 : ℝ) + (-2 : ℝ) ^ 2 = -4 := 
by
  sorry

variable (a : ℝ)

theorem problem2 : (-3 * a ^ 3) ^ 2 * 2 * a ^ 3 - 8 * a ^ 12 / (2 * a ^ 3) = 14 * a ^ 9 :=
by
  sorry

end problem1_problem2_l74_74187


namespace find_side_b_in_triangle_l74_74667

noncomputable def triangle_side_b (a A : ℝ) (cosB : ℝ) : ℝ :=
  let sinB := Real.sqrt (1 - cosB^2)
  let sinA := Real.sin A
  (a * sinB) / sinA

theorem find_side_b_in_triangle :
  triangle_side_b 5 (Real.pi / 4) (3 / 5) = 4 * Real.sqrt 2 :=
by
  sorry

end find_side_b_in_triangle_l74_74667


namespace max_cards_with_digit_three_l74_74348

/-- There are ten cards each of the digits "3", "4", and "5". Choose any 8 cards such that their sum is 27. 
Prove that the maximum number of these cards that can be "3" is 6. -/
theorem max_cards_with_digit_three (c3 c4 c5 : ℕ) (hc3 : c3 + c4 + c5 = 8) (h_sum : 3 * c3 + 4 * c4 + 5 * c5 = 27) :
  c3 ≤ 6 :=
sorry

end max_cards_with_digit_three_l74_74348


namespace solve_system_l74_74000

theorem solve_system (x y : ℝ) :
  (2 * y = (abs (2 * x + 3)) - (abs (2 * x - 3))) ∧ 
  (4 * x = (abs (y + 2)) - (abs (y - 2))) → 
  (-1 ≤ x ∧ x ≤ 1 ∧ y = 2 * x) := 
by
  sorry

end solve_system_l74_74000


namespace find_X_l74_74505

-- Defining the given conditions and what we need to prove
theorem find_X (X : ℝ) (h : (X + 43 / 151) * 151 = 2912) : X = 19 :=
sorry

end find_X_l74_74505


namespace even_func_monotonic_on_negative_interval_l74_74237

variable {α : Type*} [LinearOrderedField α]
variable {f : α → α}

theorem even_func_monotonic_on_negative_interval 
  (h_even : ∀ x : α, f (-x) = f x)
  (h_mon_incr : ∀ x y : α, x < y → (x < 0 ∧ y ≤ 0) → f x < f y) :
  f 2 < f (-3 / 2) :=
sorry

end even_func_monotonic_on_negative_interval_l74_74237


namespace no_rational_x_y_m_n_with_conditions_l74_74057

noncomputable def f (t : ℚ) : ℚ := t^3 + t

theorem no_rational_x_y_m_n_with_conditions :
  ¬ ∃ (x y : ℚ) (m n : ℕ), xy = 3 ∧ m > 0 ∧ n > 0 ∧
    (f^[m] x = f^[n] y) := 
sorry

end no_rational_x_y_m_n_with_conditions_l74_74057


namespace wallpaper_expenditure_l74_74307

structure Room :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

def cost_per_square_meter : ℕ := 75

def total_expenditure (room : Room) : ℕ :=
  let perimeter := 2 * (room.length + room.width)
  let area_of_walls := perimeter * room.height
  let area_of_ceiling := room.length * room.width
  let total_area := area_of_walls + area_of_ceiling
  total_area * cost_per_square_meter

theorem wallpaper_expenditure (room : Room) : 
  room = Room.mk 30 25 10 →
  total_expenditure room = 138750 :=
by 
  intros h
  rw [h]
  sorry

end wallpaper_expenditure_l74_74307


namespace acute_triangle_side_range_l74_74999

theorem acute_triangle_side_range {x : ℝ} (h : ∀ a b c : ℝ, a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2) :
  2 < 4 ∧ 4 < x → (2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 5) :=
  sorry

end acute_triangle_side_range_l74_74999


namespace gifts_needed_l74_74418

def num_teams : ℕ := 7
def num_gifts_per_team : ℕ := 2

theorem gifts_needed (h1 : num_teams = 7) (h2 : num_gifts_per_team = 2) : num_teams * num_gifts_per_team = 14 := 
by
  -- proof skipped
  sorry

end gifts_needed_l74_74418


namespace cone_base_radius_l74_74798

noncomputable def sector_radius : ℝ := 9
noncomputable def central_angle_deg : ℝ := 240

theorem cone_base_radius :
  let arc_length := (central_angle_deg * Real.pi * sector_radius) / 180
  let base_circumference := arc_length
  let base_radius := base_circumference / (2 * Real.pi)
  base_radius = 6 :=
by
  sorry

end cone_base_radius_l74_74798


namespace compare_abc_l74_74911

theorem compare_abc 
  (a : ℝ := 1 / 11) 
  (b : ℝ := Real.sqrt (1 / 10)) 
  (c : ℝ := Real.log (11 / 10)) : 
  b > c ∧ c > a := 
by
  sorry

end compare_abc_l74_74911


namespace value_of_ratio_l74_74001

theorem value_of_ratio (x y : ℝ)
    (hx : x > 0)
    (hy : y > 0)
    (h : 2 * x + 3 * y = 8) :
    (2 / x + 3 / y) = 25 / 8 := 
by
  sorry

end value_of_ratio_l74_74001


namespace sum_of_powers_of_i_l74_74288

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 + i^4 + i^5 = i := by
  sorry

end sum_of_powers_of_i_l74_74288


namespace distinct_arrangements_on_3x3_grid_l74_74190

def is_valid_position (pos : ℤ × ℤ) : Prop :=
  0 ≤ pos.1 ∧ pos.1 < 3 ∧ 0 ≤ pos.2 ∧ pos.2 < 3

def rotations_equiv (pos1 pos2 : ℤ × ℤ) : Prop :=
  pos1 = pos2 ∨ pos1 = (2 - pos2.2, pos2.1) ∨ pos1 = (2 - pos2.1, 2 - pos2.2) ∨ pos1 = (pos2.2, 2 - pos2.1)

def distinct_positions_count (grid_size : ℕ) : ℕ :=
  10  -- given from the problem solution

theorem distinct_arrangements_on_3x3_grid : distinct_positions_count 3 = 10 := sorry

end distinct_arrangements_on_3x3_grid_l74_74190


namespace roots_cubic_inv_sum_l74_74053

theorem roots_cubic_inv_sum (a b c r s : ℝ) (h_eq : ∃ (r s : ℝ), r^2 * a + b * r - c = 0 ∧ s^2 * a + b * s - c = 0) :
  (1 / r^3) + (1 / s^3) = (b^3 + 3 * a * b * c) / c^3 :=
by
  sorry

end roots_cubic_inv_sum_l74_74053


namespace range_of_m_l74_74941

noncomputable def quadratic_inequality_solution_set_is_R (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0

theorem range_of_m :
  { m : ℝ | quadratic_inequality_solution_set_is_R m } = { m : ℝ | 1 ≤ m ∧ m < 9 } :=
by
  sorry

end range_of_m_l74_74941


namespace height_difference_l74_74923

variables (H1 H2 H3 : ℕ)
variable (x : ℕ)
variable (h_ratio : H1 = 4 * x ∧ H2 = 5 * x ∧ H3 = 6 * x)
variable (h_lightest : H1 = 120)

theorem height_difference :
  (H1 + H3) - H2 = 150 :=
by
  -- Proof will go here
  sorry

end height_difference_l74_74923


namespace shift_sine_graph_l74_74998

theorem shift_sine_graph (x : ℝ) : 
  (∃ θ : ℝ, θ = (5 * Real.pi) / 4 ∧ 
  y = Real.sin (x - Real.pi / 4) → y = Real.sin (x + θ) 
  ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) := sorry

end shift_sine_graph_l74_74998


namespace min_value_of_function_l74_74814

theorem min_value_of_function (x : ℝ) (h : x > -1) : 
  ∃ x, (x > -1) ∧ (x = 0) ∧ ∀ y, (y = x + (1 / (x + 1))) → y ≥ 1 := 
sorry

end min_value_of_function_l74_74814


namespace triangle_inequality_l74_74362

theorem triangle_inequality
  (α β γ a b c : ℝ)
  (h_angles_sum : α + β + γ = Real.pi)
  (h_pos_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_pos_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  a * (1 / β + 1 / γ) + b * (1 / γ + 1 / α) + c * (1 / α + 1 / β) ≥ 2 * (a / α + b / β + c / γ) := by
  sorry

end triangle_inequality_l74_74362


namespace BillCookingTime_l74_74471

-- Definitions corresponding to the conditions
def chopTimePepper : Nat := 3  -- minutes to chop one pepper
def chopTimeOnion : Nat := 4   -- minutes to chop one onion
def grateTimeCheese : Nat := 1 -- minutes to grate cheese for one omelet
def cookTimeOmelet : Nat := 5  -- minutes to assemble and cook one omelet

def numberOfPeppers : Nat := 4  -- number of peppers Bill needs to chop
def numberOfOnions : Nat := 2   -- number of onions Bill needs to chop
def numberOfOmelets : Nat := 5  -- number of omelets Bill prepares

-- Calculations based on conditions
def totalChopTimePepper : Nat := numberOfPeppers * chopTimePepper
def totalChopTimeOnion : Nat := numberOfOnions * chopTimeOnion
def totalGrateTimeCheese : Nat := numberOfOmelets * grateTimeCheese
def totalCookTimeOmelet : Nat := numberOfOmelets * cookTimeOmelet

-- Total preparation and cooking time
def totalTime : Nat := totalChopTimePepper + totalChopTimeOnion + totalGrateTimeCheese + totalCookTimeOmelet

-- Theorem statement
theorem BillCookingTime :
  totalTime = 50 := by
  sorry

end BillCookingTime_l74_74471


namespace count_students_neither_math_physics_chemistry_l74_74964

def total_students := 150

def students_math := 90
def students_physics := 70
def students_chemistry := 40

def students_math_and_physics := 20
def students_math_and_chemistry := 15
def students_physics_and_chemistry := 10
def students_all_three := 5

theorem count_students_neither_math_physics_chemistry :
  (total_students - 
   (students_math + students_physics + students_chemistry - 
    students_math_and_physics - students_math_and_chemistry - 
    students_physics_and_chemistry + students_all_three)) = 5 := by
  sorry

end count_students_neither_math_physics_chemistry_l74_74964


namespace a_n_formula_b_n_geometric_sequence_l74_74431

noncomputable def a_n (n : ℕ) : ℝ := 3 * n - 1

def S_n (n : ℕ) : ℝ := sorry -- Sum of the first n terms of b_n

def b_n (n : ℕ) : ℝ := 2 - 2 * S_n n

theorem a_n_formula (n : ℕ) : a_n n = 3 * n - 1 :=
by { sorry }

theorem b_n_geometric_sequence : ∀ n ≥ 2, b_n n / b_n (n - 1) = 1 / 3 :=
by { sorry }

end a_n_formula_b_n_geometric_sequence_l74_74431


namespace middle_number_in_consecutive_nat_sum_squares_equals_2030_l74_74119

theorem middle_number_in_consecutive_nat_sum_squares_equals_2030 
  (n : ℕ)
  (h1 : (n - 1)^2 + n^2 + (n + 1)^2 = 2030)
  (h2 : (n^3 - n^2) % 7 = 0)
  : n = 26 := 
sorry

end middle_number_in_consecutive_nat_sum_squares_equals_2030_l74_74119


namespace countTwoLeggedBirds_l74_74748

def countAnimals (x y : ℕ) : Prop :=
  x + y = 200 ∧ 2 * x + 4 * y = 522

theorem countTwoLeggedBirds (x y : ℕ) (h : countAnimals x y) : x = 139 :=
by
  sorry

end countTwoLeggedBirds_l74_74748


namespace ellipse_problem_l74_74970

theorem ellipse_problem
  (a b : ℝ)
  (h₀ : 0 < a)
  (h₁ : 0 < b)
  (h₂ : a > b)
  (P Q : ℝ × ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1})
  (A : ℝ × ℝ)
  (hA : A = (a, 0))
  (R : ℝ × ℝ)
  (O : ℝ × ℝ)
  (hO : O = (0, 0))
  (AQ_OP_parallels : ∀ (x y : ℝ) (Qx Qy Px Py : ℝ), 
    x = a ∧ y = 0  ∧ (Qx, Qy) = (x, y) ↔ (O.1, O.2) = (Px, Py)
    ) :
  ∀ (AQ AR OP : ℝ), 
  AQ = dist (a, 0) Q → 
  AR = dist A R → 
  OP = dist O P → 
  |AQ * AR| / (OP ^ 2) = 2 :=
  sorry

end ellipse_problem_l74_74970


namespace smallest_base_for_62_three_digits_l74_74279

theorem smallest_base_for_62_three_digits: 
  ∃ b : ℕ, (b^2 ≤ 62 ∧ 62 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 62 ∧ 62 < n^3) → n ≥ b :=
by
  sorry

end smallest_base_for_62_three_digits_l74_74279


namespace base_conversion_min_sum_l74_74513

theorem base_conversion_min_sum : ∃ a b : ℕ, a > 6 ∧ b > 6 ∧ (6 * a + 3 = 3 * b + 6) ∧ (a + b = 20) :=
by
  sorry

end base_conversion_min_sum_l74_74513


namespace prime_p_p_plus_15_l74_74624

theorem prime_p_p_plus_15 (p : ℕ) (hp : Nat.Prime p) (hp15 : Nat.Prime (p + 15)) : p = 2 :=
sorry

end prime_p_p_plus_15_l74_74624


namespace find_three_fifths_of_neg_twelve_sevenths_l74_74446

def a : ℚ := -12 / 7
def b : ℚ := 3 / 5
def c : ℚ := -36 / 35

theorem find_three_fifths_of_neg_twelve_sevenths : b * a = c := by 
  -- sorry is a placeholder for the actual proof
  sorry

end find_three_fifths_of_neg_twelve_sevenths_l74_74446


namespace find_X_plus_Y_l74_74554

-- Statement of the problem translated from the given problem-solution pair.
theorem find_X_plus_Y (X Y : ℚ) :
  (∀ x : ℚ, x ≠ 5 → x ≠ 6 →
    (Y * x + 8) / (x^2 - 11 * x + 30) = X / (x - 5) + 7 / (x - 6)) →
  X + Y = -22 / 3 :=
by
  sorry

end find_X_plus_Y_l74_74554


namespace min_T_tiles_needed_l74_74380

variable {a b c d : Nat}
variable (total_blocks : Nat := a + b + c + d)
variable (board_size : Nat := 8 * 10)
variable (block_size : Nat := 4)
variable (tile_types := ["T_horizontal", "T_vertical", "S_horizontal", "S_vertical"])
variable (conditions : Prop := total_blocks = 20 ∧ a + c ≥ 5)

theorem min_T_tiles_needed
    (h : conditions)
    (covering : total_blocks * block_size = board_size)
    (T_tiles : a ≥ 6) :
    a = 6 := sorry

end min_T_tiles_needed_l74_74380


namespace total_amount_is_correct_l74_74250

-- Definitions based on the conditions
def share_a (x : ℕ) : ℕ := 2 * x
def share_b (x : ℕ) : ℕ := 4 * x
def share_c (x : ℕ) : ℕ := 5 * x
def share_d (x : ℕ) : ℕ := 4 * x

-- Condition: combined share of a and b is 1800
def combined_share_of_ab (x : ℕ) : Prop := share_a x + share_b x = 1800

-- Theorem we want to prove: Total amount given to all children is $4500
theorem total_amount_is_correct (x : ℕ) (h : combined_share_of_ab x) : 
  share_a x + share_b x + share_c x + share_d x = 4500 := sorry

end total_amount_is_correct_l74_74250


namespace closest_integer_to_cube_root_of_1728_l74_74990

theorem closest_integer_to_cube_root_of_1728: 
  ∃ n : ℕ, n^3 = 1728 ∧ (∀ m : ℤ, m^3 < 1728 → m < n) ∧ (∀ p : ℤ, p^3 > 1728 → p > n) :=
by
  sorry

end closest_integer_to_cube_root_of_1728_l74_74990


namespace petunia_fertilizer_problem_l74_74965

theorem petunia_fertilizer_problem
  (P : ℕ)
  (h1 : 4 * P * 8 + 3 * 6 * 3 + 2 * 2 = 314) :
  P = 8 :=
by
  sorry

end petunia_fertilizer_problem_l74_74965


namespace expression_divisible_by_9_for_any_int_l74_74492

theorem expression_divisible_by_9_for_any_int (a b : ℤ) : 9 ∣ ((3 * a + 2)^2 - (3 * b + 2)^2) := 
by 
  sorry

end expression_divisible_by_9_for_any_int_l74_74492


namespace width_of_barrier_l74_74533

theorem width_of_barrier (r1 r2 : ℝ) (h : 2 * π * r1 - 2 * π * r2 = 16 * π) : r1 - r2 = 8 :=
by
  -- The proof would be inserted here, but is not required as per instructions.
  sorry

end width_of_barrier_l74_74533


namespace CodgerNeedsTenPairs_l74_74333

def CodgerHasThreeFeet : Prop := true

def ShoesSoldInPairs : Prop := true

def ShoesSoldInEvenNumberedPairs : Prop := true

def CodgerOwnsOneThreePieceSet : Prop := true

-- Main theorem stating Codger needs 10 pairs of shoes to have 7 complete 3-piece sets
theorem CodgerNeedsTenPairs (h1 : CodgerHasThreeFeet) (h2 : ShoesSoldInPairs)
  (h3 : ShoesSoldInEvenNumberedPairs) (h4 : CodgerOwnsOneThreePieceSet) : 
  ∃ pairsToBuy : ℕ, pairsToBuy = 10 := 
by {
  -- We have to prove codger needs 10 pairs of shoes to have 7 complete 3-piece sets
  sorry
}

end CodgerNeedsTenPairs_l74_74333


namespace track_circumference_is_180_l74_74031

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

end track_circumference_is_180_l74_74031


namespace cos_beta_value_l74_74568

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
    (h1 : Real.sin α = 3/5) (h2 : Real.cos (α + β) = 5/13) : 
    Real.cos β = 56/65 := 
by
  sorry

end cos_beta_value_l74_74568


namespace sandwich_cost_l74_74032

theorem sandwich_cost (S : ℝ) (h : 2 * S + 4 * 0.87 = 8.36) : S = 2.44 :=
by sorry

end sandwich_cost_l74_74032


namespace isosceles_triangle_perimeter_l74_74037

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2))
  (h2 : ∃ x y z : ℕ, (x = a ∧ y = a ∧ z = b) ∨ (x = b ∧ y = b ∧ z = a) ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
  a + a + b = 12 :=
by
  sorry

end isosceles_triangle_perimeter_l74_74037


namespace moneySpentOnPaintbrushes_l74_74709

def totalExpenditure := 90
def costOfCanvases := 40
def costOfPaints := costOfCanvases / 2
def costOfEasel := 15
def costOfOthers := costOfCanvases + costOfPaints + costOfEasel

theorem moneySpentOnPaintbrushes : totalExpenditure - costOfOthers = 15 := by
  sorry

end moneySpentOnPaintbrushes_l74_74709


namespace regular_polygon_exterior_angle_l74_74987

theorem regular_polygon_exterior_angle (n : ℕ) (h : 60 * n = 360) : n = 6 :=
sorry

end regular_polygon_exterior_angle_l74_74987


namespace sequence_a_n_l74_74924

theorem sequence_a_n {a : ℕ → ℤ}
  (h1 : a 2 = 5)
  (h2 : a 1 = 1)
  (h3 : ∀ n ≥ 2, a (n+1) - 2 * a n + a (n-1) = 7) :
  a 17 = 905 :=
  sorry

end sequence_a_n_l74_74924


namespace eq_value_of_2a_plus_b_l74_74047

theorem eq_value_of_2a_plus_b (a b : ℝ) (h : abs (a + 2) + (b - 5)^2 = 0) : 2 * a + b = 1 := by
  sorry

end eq_value_of_2a_plus_b_l74_74047


namespace david_wins_2011th_even_l74_74713

theorem david_wins_2011th_even :
  ∃ n : ℕ, (∃ k : ℕ, k = 2011 ∧ n = 2 * k) ∧ (∀ a b : ℕ, a < b → a + b < b * a) ∧ (n % 2 = 0) := 
sorry

end david_wins_2011th_even_l74_74713


namespace drawing_red_ball_random_drawing_yellow_ball_impossible_probability_black_ball_number_of_additional_black_balls_l74_74286

-- Definitions for the initial conditions
def initial_white_balls := 2
def initial_black_balls := 3
def initial_red_balls := 5
def total_balls := initial_white_balls + initial_black_balls + initial_red_balls

-- Statement for part 1: Drawing a red ball is a random event
theorem drawing_red_ball_random : (initial_red_balls > 0) := by
  sorry

-- Statement for part 1: Drawing a yellow ball is impossible
theorem drawing_yellow_ball_impossible : (0 = 0) := by
  sorry

-- Statement for part 2: Probability of drawing a black ball
theorem probability_black_ball : (initial_black_balls : ℚ) / total_balls = 3 / 10 := by
  sorry

-- Definitions for the conditions in part 3
def additional_black_balls (x : ℕ) := initial_black_balls + x
def new_total_balls (x : ℕ) := total_balls + x

-- Statement for part 3: Finding the number of additional black balls
theorem number_of_additional_black_balls (x : ℕ)
  (h : (additional_black_balls x : ℚ) / new_total_balls x = 3 / 4) : x = 18 := by
  sorry

end drawing_red_ball_random_drawing_yellow_ball_impossible_probability_black_ball_number_of_additional_black_balls_l74_74286


namespace translation_symmetric_graphs_l74_74499

/-- The graph of the function f(x)=sin(x/π + φ) is translated to the right by θ (θ>0) units to obtain the graph of the function g(x).
    On the graph of f(x), point A is translated to point B, let x_A and x_B be the abscissas of points A and B respectively.
    If the axes of symmetry of the graphs of f(x) and g(x) coincide, then the real values that can be taken as x_A - x_B are -2π² or -π². -/
theorem translation_symmetric_graphs (θ : ℝ) (hθ : θ > 0) (x_A x_B : ℝ) (φ : ℝ) :
  ((x_A - x_B = -2 * π^2) ∨ (x_A - x_B = -π^2)) :=
sorry

end translation_symmetric_graphs_l74_74499


namespace find_t_l74_74462

theorem find_t (t : ℕ) : 
  t > 3 ∧ (3 * t - 10) * (4 * t - 9) = (t + 12) * (2 * t + 1) → t = 6 := 
by
  intro h
  have h1 : t > 3 := h.1
  have h2 : (3 * t - 10) * (4 * t - 9) = (t + 12) * (2 * t + 1) := h.2
  sorry

end find_t_l74_74462


namespace perfect_square_n_l74_74516

theorem perfect_square_n (n : ℕ) (hn_pos : n > 0) :
  (∃ (m : ℕ), m * m = (n^2 + 11 * n - 4) * n.factorial + 33 * 13^n + 4) ↔ n = 1 ∨ n = 2 :=
by sorry

end perfect_square_n_l74_74516


namespace integer_solutions_exist_l74_74459

theorem integer_solutions_exist (k : ℤ) :
  (∃ x : ℤ, 9 * x - 3 = k * x + 14) ↔ (k = 8 ∨ k = 10 ∨ k = -8 ∨ k = 26) :=
by
  sorry

end integer_solutions_exist_l74_74459


namespace bake_sale_money_made_l74_74982

theorem bake_sale_money_made :
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  total_money = 648 :=
by
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  sorry

end bake_sale_money_made_l74_74982


namespace grapes_purchased_l74_74145

variable (G : ℕ)
variable (rate_grapes : ℕ) (qty_mangoes : ℕ) (rate_mangoes : ℕ) (total_paid : ℕ)

theorem grapes_purchased (h1 : rate_grapes = 70)
                        (h2 : qty_mangoes = 9)
                        (h3 : rate_mangoes = 55)
                        (h4 : total_paid = 1055) :
                        70 * G + 9 * 55 = 1055 → G = 8 :=
by
  sorry

end grapes_purchased_l74_74145


namespace actual_number_of_children_l74_74501

-- Define the conditions of the problem
def condition1 (C B : ℕ) : Prop := B = 2 * C
def condition2 : ℕ := 320
def condition3 (C B : ℕ) : Prop := B = 4 * (C - condition2)

-- Define the statement to be proved
theorem actual_number_of_children (C B : ℕ) 
  (h1 : condition1 C B) (h2 : condition3 C B) : C = 640 :=
by 
  -- Proof will be added here
  sorry

end actual_number_of_children_l74_74501


namespace red_paint_intensity_l74_74611

theorem red_paint_intensity (x : ℝ) (h1 : 0.5 * 10 + 0.5 * x = 15) : x = 20 :=
sorry

end red_paint_intensity_l74_74611


namespace Cally_colored_shirts_l74_74194

theorem Cally_colored_shirts (C : ℕ) (hcally : 10 + 7 + 6 = 23) (hdanny : 6 + 8 + 10 + 6 = 30) (htotal : 23 + 30 + C = 58) : 
  C = 5 := 
by
  sorry

end Cally_colored_shirts_l74_74194


namespace Ming_initial_ladybugs_l74_74820

-- Define the conditions
def Sami_spiders : Nat := 3
def Hunter_ants : Nat := 12
def insects_remaining : Nat := 21
def ladybugs_flew_away : Nat := 2

-- Formalize the proof problem
theorem Ming_initial_ladybugs : Sami_spiders + Hunter_ants + (insects_remaining + ladybugs_flew_away) - (Sami_spiders + Hunter_ants) = 8 := by
  sorry

end Ming_initial_ladybugs_l74_74820


namespace union_sets_l74_74136

open Set

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem union_sets :
  A ∪ B = {1, 2, 3, 4, 5} :=
by
  sorry

end union_sets_l74_74136


namespace petya_square_larger_l74_74574

noncomputable def dimension_petya_square (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

noncomputable def dimension_vasya_square (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2)

theorem petya_square_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  dimension_vasya_square a b < dimension_petya_square a b :=
by
  sorry

end petya_square_larger_l74_74574


namespace items_purchased_total_profit_l74_74535

-- Definitions based on conditions given in part (a)
def total_cost := 6000
def cost_A := 22
def cost_B := 30
def sell_A := 29
def sell_B := 40

-- Proven answers from the solution (part (b))
def items_A := 150
def items_B := 90
def profit := 1950

-- Lean theorem statements (problems to be proved)
theorem items_purchased : (22 * items_A + 30 * (items_A / 2 + 15) = total_cost) → 
                          (items_A = 150) ∧ (items_B = 90) := sorry

theorem total_profit : (items_A = 150) → (items_B = 90) → 
                       ((items_A * (sell_A - cost_A) + items_B * (sell_B - cost_B)) = profit) := sorry

end items_purchased_total_profit_l74_74535


namespace find_fourth_student_number_l74_74838

theorem find_fourth_student_number 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (student1_num : ℕ) 
  (student2_num : ℕ) 
  (student3_num : ℕ) 
  (student4_num : ℕ)
  ( H1 : total_students = 52 )
  ( H2 : sample_size = 4 )
  ( H3 : student1_num = 6 )
  ( H4 : student2_num = 32 )
  ( H5 : student3_num = 45 ) :
  student4_num = 19 :=
sorry

end find_fourth_student_number_l74_74838


namespace infinite_series_eq_15_l74_74794

theorem infinite_series_eq_15 (x : ℝ) :
  (∑' (n : ℕ), (5 + n * x) / 3^n) = 15 ↔ x = 10 :=
by
  sorry

end infinite_series_eq_15_l74_74794


namespace sum_of_pairs_l74_74402

theorem sum_of_pairs (a : ℕ → ℝ) (h1 : ∀ n, a n ≠ 0)
  (h2 : ∀ n, a n * a (n + 3) = a (n + 2) * a (n + 5))
  (h3 : a 1 * a 2 + a 3 * a 4 + a 5 * a 6 = 6) :
  a 1 * a 2 + a 3 * a 4 + a 5 * a 6 + a 7 * a 8 + a 9 * a 10 + a 11 * a 12 + 
  a 13 * a 14 + a 15 * a 16 + a 17 * a 18 + a 19 * a 20 + a 21 * a 22 + 
  a 23 * a 24 + a 25 * a 26 + a 27 * a 28 + a 29 * a 30 + a 31 * a 32 + 
  a 33 * a 34 + a 35 * a 36 + a 37 * a 38 + a 39 * a 40 + a 41 * a 42 = 42 := 
sorry

end sum_of_pairs_l74_74402


namespace probability_three_heads_l74_74353

noncomputable def fair_coin_flip: ℝ := 1 / 2

theorem probability_three_heads :
  (fair_coin_flip * fair_coin_flip * fair_coin_flip) = 1 / 8 :=
by
  -- proof would go here
  sorry

end probability_three_heads_l74_74353


namespace molecular_weight_is_122_l74_74291

noncomputable def molecular_weight_of_compound := 
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.008
  let atomic_weight_O := 16.00
  7 * atomic_weight_C + 6 * atomic_weight_H + 2 * atomic_weight_O

theorem molecular_weight_is_122 :
  molecular_weight_of_compound = 122 := by
  sorry

end molecular_weight_is_122_l74_74291


namespace find_sum_l74_74454

variable (a b c d : ℝ)

theorem find_sum (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end find_sum_l74_74454


namespace triangle_PQ_length_l74_74510

theorem triangle_PQ_length (RP PQ : ℝ) (n : ℕ) (h_rp : RP = 2.4) (h_n : n = 25) : RP = 2.4 → PQ = 3 := by
  sorry

end triangle_PQ_length_l74_74510


namespace common_root_solutions_l74_74247

theorem common_root_solutions (a : ℝ) (b : ℝ) :
  (a^2 * b^2 + a * b - 1 = 0) ∧ (b^2 - a * b - a^2 = 0) →
  a = (-1 + Real.sqrt 5) / 2 ∨ a = (-1 - Real.sqrt 5) / 2 ∨
  a = (1 + Real.sqrt 5) / 2 ∨ a = (1 - Real.sqrt 5) / 2 :=
by
  intro h
  sorry

end common_root_solutions_l74_74247


namespace geometric_sequence_common_ratio_l74_74948

noncomputable def common_ratio_q (a1 a5 a : ℕ) (q : ℕ) : Prop :=
  a1 * a5 = 16 ∧ a1 > 0 ∧ a5 > 0 ∧ a = 2 ∧ q = 2

theorem geometric_sequence_common_ratio : ∀ (a1 a5 a q : ℕ), 
  common_ratio_q a1 a5 a q → q = 2 :=
by
  intros a1 a5 a q h
  have h1 : a1 * a5 = 16 := h.1
  have h2 : a1 > 0 := h.2.1
  have h3 : a5 > 0 := h.2.2.1
  have h4 : a = 2 := h.2.2.2.1
  have h5 : q = 2 := h.2.2.2.2
  exact h5

end geometric_sequence_common_ratio_l74_74948


namespace ellipse_slope_condition_l74_74394

theorem ellipse_slope_condition (a b x y x₀ y₀ : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h_ellipse1 : x^2 / a^2 + y^2 / b^2 = 1) 
  (h_ellipse2 : x₀^2 / a^2 + y₀^2 / b^2 = 1) 
  (hA : x ≠ x₀ ∨ y ≠ y₀) 
  (hB : x ≠ -x₀ ∨ y ≠ -y₀) :
  ((y - y₀) / (x - x₀)) * ((y + y₀) / (x + x₀)) = -b^2 / a^2 := 
sorry

end ellipse_slope_condition_l74_74394


namespace students_drawn_from_class_A_l74_74061

-- Given conditions
def classA_students : Nat := 40
def classB_students : Nat := 50
def total_sample : Nat := 18

-- Predicate that checks if the number of students drawn from Class A is correct
theorem students_drawn_from_class_A (students_from_A : Nat) : students_from_A = 9 :=
by
  sorry

end students_drawn_from_class_A_l74_74061


namespace area_of_side_face_l74_74622

theorem area_of_side_face (L W H : ℝ) 
  (h1 : W * H = (1/2) * (L * W))
  (h2 : L * W = 1.5 * (H * L))
  (h3 : L * W * H = 648) :
  H * L = 72 := 
by
  sorry

end area_of_side_face_l74_74622


namespace total_spent_is_140_l74_74177

-- Define the original prices and discounts
def original_price_shoes : ℕ := 50
def original_price_dress : ℕ := 100
def discount_shoes : ℕ := 40
def discount_dress : ℕ := 20

-- Define the number of items purchased
def number_of_shoes : ℕ := 2
def number_of_dresses : ℕ := 1

-- Define the calculation of discounted prices
def discounted_price_shoes (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

def discounted_price_dress (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

-- Define the total cost calculation
def total_cost : ℕ :=
  discounted_price_shoes original_price_shoes discount_shoes number_of_shoes +
  discounted_price_dress original_price_dress discount_dress number_of_dresses

-- The theorem to prove
theorem total_spent_is_140 : total_cost = 140 := by
  sorry

end total_spent_is_140_l74_74177


namespace find_b_l74_74301

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 :=
by
  sorry

end find_b_l74_74301


namespace find_r_l74_74719

def cubic_function (p q r x : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_r (p q r : ℝ) (h1 : cubic_function p q r (-1) = 0) :
  r = p - 2 :=
sorry

end find_r_l74_74719


namespace y_is_multiple_of_3_and_6_l74_74859

-- Define y as a sum of given numbers
def y : ℕ := 48 + 72 + 144 + 216 + 432 + 648 + 2592

theorem y_is_multiple_of_3_and_6 :
  (y % 3 = 0) ∧ (y % 6 = 0) :=
by
  -- Proof would go here, but we will end with sorry
  sorry

end y_is_multiple_of_3_and_6_l74_74859


namespace solve_puzzle_l74_74400

theorem solve_puzzle (x1 x2 x3 x4 x5 x6 x7 x8 : ℕ) : 
  (8 + x1 + x2 = 20) →
  (x1 + x2 + x3 = 20) →
  (x2 + x3 + x4 = 20) →
  (x3 + x4 + x5 = 20) →
  (x4 + x5 + 5 = 20) →
  (x5 + 5 + x6 = 20) →
  (5 + x6 + x7 = 20) →
  (x6 + x7 + x8 = 20) →
  (x1 = 7 ∧ x2 = 5 ∧ x3 = 8 ∧ x4 = 7 ∧ x5 = 5 ∧ x6 = 8 ∧ x7 = 7 ∧ x8 = 5) :=
by {
  sorry
}

end solve_puzzle_l74_74400


namespace max_height_of_ball_l74_74848

def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 20

theorem max_height_of_ball : ∃ t₀, h t₀ = 81.25 ∧ ∀ t, h t ≤ 81.25 :=
by
  sorry

end max_height_of_ball_l74_74848


namespace evaluate_absolute_value_l74_74481

theorem evaluate_absolute_value (π : ℝ) (h : π < 5.5) : |5.5 - π| = 5.5 - π :=
by
  sorry

end evaluate_absolute_value_l74_74481


namespace quadratic_inequality_solution_l74_74441

theorem quadratic_inequality_solution (x : ℝ) : 2 * x^2 - 5 * x - 3 ≥ 0 ↔ x ≤ -1/2 ∨ x ≥ 3 := 
by
  sorry

end quadratic_inequality_solution_l74_74441


namespace fuel_tank_capacity_l74_74293

def ethanol_content_fuel_A (fuel_A : ℝ) : ℝ := 0.12 * fuel_A
def ethanol_content_fuel_B (fuel_B : ℝ) : ℝ := 0.16 * fuel_B

theorem fuel_tank_capacity (C : ℝ) :
  ethanol_content_fuel_A 122 + ethanol_content_fuel_B (C - 122) = 30 → C = 218 :=
by
  sorry

end fuel_tank_capacity_l74_74293


namespace polarBearDailyFish_l74_74546

-- Define the conditions
def polarBearDailyTrout : ℝ := 0.2
def polarBearDailySalmon : ℝ := 0.4

-- Define the statement to be proven
theorem polarBearDailyFish : polarBearDailyTrout + polarBearDailySalmon = 0.6 :=
by
  sorry

end polarBearDailyFish_l74_74546


namespace sum_factors_of_18_l74_74147

theorem sum_factors_of_18 : (1 + 18 + 2 + 9 + 3 + 6) = 39 := by
  sorry

end sum_factors_of_18_l74_74147


namespace interest_difference_l74_74654

def principal : ℝ := 3600
def rate : ℝ := 0.25
def time : ℕ := 2

def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

theorem interest_difference :
  let SI := simple_interest principal rate time;
  let CI := compound_interest principal rate time;
  CI - SI = 225 :=
by
  sorry

end interest_difference_l74_74654


namespace number_of_boys_l74_74384

theorem number_of_boys
  (x y : ℕ) 
  (h1 : x + y = 43)
  (h2 : 24 * x + 27 * y = 1101) : 
  x = 20 := by
  sorry

end number_of_boys_l74_74384


namespace find_radius_l74_74723

noncomputable def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Prop := sorry

theorem find_radius (C1 : ℝ × ℝ × ℝ) (r1 : ℝ) (C2 : ℝ × ℝ × ℝ) (r : ℝ) :
  C1 = (3, 5, 0) →
  r1 = 2 →
  C2 = (0, 5, -8) →
  (sphere ((3, 5, -8) : ℝ × ℝ × ℝ) (2 * Real.sqrt 17)) →
  r = Real.sqrt 59 :=
by
  intros h1 h2 h3 h4
  sorry

end find_radius_l74_74723


namespace tax_rate_correct_l74_74800

/-- The tax rate in dollars per $100.00 is $82.00, given that the tax rate as a percent is 82%. -/
theorem tax_rate_correct (x : ℝ) (h : x = 82) : (x / 100) * 100 = 82 :=
by
  rw [h]
  sorry

end tax_rate_correct_l74_74800


namespace sphere_volume_given_surface_area_l74_74231

theorem sphere_volume_given_surface_area (r : ℝ) (V : ℝ) (S : ℝ)
  (hS : S = 36 * Real.pi)
  (h_surface_area : 4 * Real.pi * r^2 = S)
  (h_volume : V = (4/3) * Real.pi * r^3) : V = 36 * Real.pi := by
  sorry

end sphere_volume_given_surface_area_l74_74231


namespace fraction_to_decimal_l74_74424

theorem fraction_to_decimal (n d : ℕ) (hn : n = 53) (hd : d = 160) (gcd_nd : Nat.gcd n d = 1)
  (prime_factorization_d : ∃ k l : ℕ, d = 2^k * 5^l) : ∃ dec : ℚ, (n:ℚ) / (d:ℚ) = dec ∧ dec = 0.33125 :=
by sorry

end fraction_to_decimal_l74_74424


namespace part1_part2_l74_74159

namespace Problem

open Set

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

-- Part (1)
theorem part1 : A ∩ (B ∩ C) = {3} := by 
  sorry

-- Part (2)
theorem part2 : A ∩ (A \ (B ∪ C)) = {-6, -5, -4, -3, -2, -1, 0} := by 
  sorry

end Problem

end part1_part2_l74_74159


namespace base_k_for_repeating_series_equals_fraction_l74_74590

-- Define the fraction 5/29
def fraction := 5 / 29

-- Define the repeating series in base k
def repeating_series (k : ℕ) : ℚ :=
  (1 / k) / (1 - 1 / k^2) + (3 / k^2) / (1 - 1 / k^2)

-- State the problem
theorem base_k_for_repeating_series_equals_fraction (k : ℕ) (hk1 : 0 < k) (hk2 : k ≠ 1):
  repeating_series k = fraction ↔ k = 8 := sorry

end base_k_for_repeating_series_equals_fraction_l74_74590


namespace minimum_value_of_quadratic_polynomial_l74_74419

-- Define the quadratic polynomial
def quadratic_polynomial (x : ℝ) : ℝ := x^2 + 14 * x + 3

-- Statement to prove
theorem minimum_value_of_quadratic_polynomial : ∃ x : ℝ, quadratic_polynomial x = quadratic_polynomial (-7) :=
sorry

end minimum_value_of_quadratic_polynomial_l74_74419


namespace bromine_is_liquid_at_25C_1atm_l74_74447

-- Definitions for the melting and boiling points
def melting_point (element : String) : Float :=
  match element with
  | "Br" => -7.2
  | "Kr" => -157.4 -- Not directly used, but included for completeness
  | "P" => 44.1 -- Not directly used, but included for completeness
  | "Xe" => -111.8 -- Not directly used, but included for completeness
  | _ => 0.0 -- default case; not used

def boiling_point (element : String) : Float :=
  match element with
  | "Br" => 58.8
  | "Kr" => -153.4
  | "P" => 280.5 -- Not directly used, but included for completeness
  | "Xe" => -108.1
  | _ => 0.0 -- default case; not used

-- Define the condition of the problem
def is_liquid_at (element : String) (temperature : Float) (pressure : Float) : Bool :=
  melting_point element < temperature ∧ temperature < boiling_point element

-- Goal statement
theorem bromine_is_liquid_at_25C_1atm : is_liquid_at "Br" 25 1 = true :=
by
  sorry

end bromine_is_liquid_at_25C_1atm_l74_74447


namespace eliza_tom_difference_l74_74388

theorem eliza_tom_difference (q : ℕ) : 
  let eliza_quarters := 7 * q + 3
  let tom_quarters := 2 * q + 8
  let quarter_difference := (7 * q + 3) - (2 * q + 8)
  let nickel_value := 5
  let groups_of_5 := quarter_difference / 5
  let difference_in_cents := nickel_value * groups_of_5
  difference_in_cents = 5 * (q - 1) := by
  sorry

end eliza_tom_difference_l74_74388


namespace converse_prop_inverse_prop_contrapositive_prop_l74_74696

-- Given condition: the original proposition is true
axiom original_prop : ∀ (x y : ℝ), x * y = 0 → x = 0 ∨ y = 0

-- Converse: If x=0 or y=0, then xy=0 - prove this is true
theorem converse_prop (x y : ℝ) : (x = 0 ∨ y = 0) → x * y = 0 :=
by
  sorry

-- Inverse: If xy ≠ 0, then x ≠ 0 and y ≠ 0 - prove this is true
theorem inverse_prop (x y : ℝ) : x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0 :=
by
  sorry

-- Contrapositive: If x ≠ 0 and y ≠ 0, then xy ≠ 0 - prove this is true
theorem contrapositive_prop (x y : ℝ) : (x ≠ 0 ∧ y ≠ 0) → x * y ≠ 0 :=
by
  sorry

end converse_prop_inverse_prop_contrapositive_prop_l74_74696


namespace number_of_men_in_first_group_l74_74731

-- Definitions for the conditions
def rate_of_work (men : ℕ) (length : ℕ) (days : ℕ) : ℕ :=
  length / days / men

def work_rate_first_group (M : ℕ) : ℕ :=
  rate_of_work M 48 2

def work_rate_second_group : ℕ :=
  rate_of_work 2 36 3

theorem number_of_men_in_first_group (M : ℕ) 
  (h₁ : work_rate_first_group M = 24)
  (h₂ : work_rate_second_group = 12) :
  M = 4 :=
  sorry

end number_of_men_in_first_group_l74_74731


namespace find_dividend_l74_74370

theorem find_dividend :
  ∀ (Divisor Quotient Remainder : ℕ), Divisor = 15 → Quotient = 9 → Remainder = 5 → (Divisor * Quotient + Remainder) = 140 :=
by
  intros Divisor Quotient Remainder hDiv hQuot hRem
  subst hDiv
  subst hQuot
  subst hRem
  sorry

end find_dividend_l74_74370


namespace total_annual_donation_l74_74146

-- Defining the conditions provided in the problem
def monthly_donation : ℕ := 1707
def months_in_year : ℕ := 12

-- Stating the theorem that answers the question
theorem total_annual_donation : monthly_donation * months_in_year = 20484 := 
by
  -- The proof is omitted for brevity
  sorry

end total_annual_donation_l74_74146


namespace min_value_sqrt_expression_l74_74875

open Real

theorem min_value_sqrt_expression : ∃ x : ℝ, ∀ y : ℝ, 
  sqrt (y^2 + (2 - y)^2) + sqrt ((y - 1)^2 + (y + 2)^2) ≥ sqrt 17 :=
by
  sorry

end min_value_sqrt_expression_l74_74875


namespace horner_v3_value_correct_l74_74183

def f (x : ℕ) : ℕ :=
  x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_eval (x : ℕ) : ℕ :=
  ((((x + 0) * x + 2) * x + 3) * x + 1) * x + 1

theorem horner_v3_value_correct :
  horner_eval 3 = 36 :=
sorry

end horner_v3_value_correct_l74_74183


namespace jars_water_fraction_l74_74350

theorem jars_water_fraction (S L W : ℝ) (h1 : W = 1/6 * S) (h2 : W = 1/5 * L) : 
  (2 * W / L) = 2 / 5 :=
by
  -- We are only stating the theorem here, not proving it.
  sorry

end jars_water_fraction_l74_74350


namespace sufficient_but_not_necessary_condition_l74_74249

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 > 1) ∧ ¬(x^2 > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l74_74249


namespace range_of_k_l74_74285

theorem range_of_k (k : ℝ) (x y : ℝ) : 
  (y = 2 * x - 5 * k + 7) → 
  (y = - (1 / 2) * x + 2) → 
  (x > 0) → 
  (y > 0) → 
  (1 < k ∧ k < 3) :=
by
  sorry

end range_of_k_l74_74285


namespace find_a_cubed_l74_74034

-- Definitions based on conditions
def varies_inversely (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^4 = k

-- Theorem statement with given conditions
theorem find_a_cubed (a b : ℝ) (k : ℝ) (h1 : varies_inversely a b)
    (h2 : a = 2) (h3 : b = 4) (k_val : k = 2048) (b_new : b = 8) : a^3 = 1 / 2 :=
sorry

end find_a_cubed_l74_74034


namespace domain_of_h_l74_74436

open Real

theorem domain_of_h : ∀ x : ℝ, |x - 5| + |x + 2| ≠ 0 := by
  intro x
  sorry

end domain_of_h_l74_74436


namespace fraction_home_l74_74206

-- Defining the conditions
def fractionFun := 5 / 13
def fractionYouth := 4 / 13

-- Stating the theorem to be proven
theorem fraction_home : 1 - (fractionFun + fractionYouth) = 4 / 13 := by
  sorry

end fraction_home_l74_74206


namespace question_1_question_2_question_3_l74_74603
-- Importing the Mathlib library for necessary functions

-- Definitions and assumptions based on the problem conditions
def z0 (m : ℝ) : ℂ := 1 - m * Complex.I
def z (x y : ℝ) : ℂ := x + y * Complex.I
def w (x' y' : ℝ) : ℂ := x' + y' * Complex.I

/-- The proof problem in Lean 4 to find necessary values and relationships -/
theorem question_1 (m : ℝ) (hm : m > 0) :
  (Complex.abs (z0 m) = 2 → m = Real.sqrt 3) ∧
  (∀ (x y : ℝ), ∃ (x' y' : ℝ), x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y) :=
by
  sorry

theorem question_2 (x y : ℝ) (hx : y = x + 1) :
  ∃ x' y', x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y ∧ 
  y' = (2 - Real.sqrt 3) * x' - 2 * Real.sqrt 3 + 2 :=
by
  sorry

theorem question_3 (x y : ℝ) :
  (∃ (k b : ℝ), y = k * x + b ∧ 
  (∀ (x y x' y' : ℝ), x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y ∧ y' = k * x' + b → 
  y = Real.sqrt 3 / 3 * x ∨ y = - Real.sqrt 3 * x)) :=
by
  sorry

end question_1_question_2_question_3_l74_74603


namespace percentage_less_than_l74_74439

namespace PercentProblem

noncomputable def A (C : ℝ) : ℝ := 0.65 * C
noncomputable def B (C : ℝ) : ℝ := 0.8923076923076923 * A C

theorem percentage_less_than (C : ℝ) (hC : C ≠ 0) : (C - B C) / C = 0.42 :=
by
  sorry

end PercentProblem

end percentage_less_than_l74_74439


namespace sum_of_cubes_l74_74067

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 :=
sorry

end sum_of_cubes_l74_74067


namespace cylinder_volume_ratio_l74_74525

noncomputable def volume_ratio (h1 h2 : ℝ) (c1 c2 : ℝ) : ℝ :=
  let r1 := c1 / (2 * Real.pi)
  let r2 := c2 / (2 * Real.pi)
  let V1 := Real.pi * r1^2 * h1
  let V2 := Real.pi * r2^2 * h2
  if V1 > V2 then V1 / V2 else V2 / V1

theorem cylinder_volume_ratio :
  volume_ratio 7 6 6 7 = 7 / 4 :=
by
  sorry

end cylinder_volume_ratio_l74_74525


namespace slip_3_5_in_F_l74_74108

def slips := [1.5, 2, 2, 2.5, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5]

def cup_sum (x : List ℝ) := List.sum x

def slips_dist (A B C D E F : List ℝ) : Prop :=
  cup_sum A + cup_sum B + cup_sum C + cup_sum D + cup_sum E + cup_sum F = 50 ∧ 
  cup_sum A = 6 ∧ cup_sum B = 8 ∧ cup_sum C = 10 ∧ cup_sum D = 12 ∧ cup_sum E = 14 ∧ cup_sum F = 16 ∧
  2.5 ∈ B ∧ 2.5 ∈ D ∧ 4 ∈ C

def contains_slip (c : List ℝ) (v : ℝ) : Prop := v ∈ c

theorem slip_3_5_in_F (A B C D E F : List ℝ) (h : slips_dist A B C D E F) : 
  contains_slip F 3.5 :=
sorry

end slip_3_5_in_F_l74_74108


namespace ab_value_l74_74981

noncomputable def func (x : ℝ) (a b : ℝ) : ℝ := 4 * x ^ 3 - a * x ^ 2 - 2 * b * x + 2

theorem ab_value 
  (a b : ℝ)
  (h_max : func 1 a b = -3)
  (h_deriv : (12 - 2 * a - 2 * b) = 0) :
  a * b = 9 :=
by
  sorry

end ab_value_l74_74981


namespace Marcus_walking_speed_l74_74414

def bath_time : ℕ := 20  -- in minutes
def blow_dry_time : ℕ := bath_time / 2  -- in minutes
def trail_distance : ℝ := 3  -- in miles
def total_dog_time : ℕ := 60  -- in minutes

theorem Marcus_walking_speed :
  let walking_time := total_dog_time - (bath_time + blow_dry_time)
  let walking_time_hours := (walking_time:ℝ) / 60
  (trail_distance / walking_time_hours) = 6 := by
  sorry

end Marcus_walking_speed_l74_74414


namespace total_males_below_50_is_2638_l74_74762

def branchA_total_employees := 4500
def branchA_percentage_males := 60 / 100
def branchA_percentage_males_at_least_50 := 40 / 100

def branchB_total_employees := 3500
def branchB_percentage_males := 50 / 100
def branchB_percentage_males_at_least_50 := 55 / 100

def branchC_total_employees := 2200
def branchC_percentage_males := 35 / 100
def branchC_percentage_males_at_least_50 := 70 / 100

def males_below_50_branchA := (1 - branchA_percentage_males_at_least_50) * (branchA_percentage_males * branchA_total_employees)
def males_below_50_branchB := (1 - branchB_percentage_males_at_least_50) * (branchB_percentage_males * branchB_total_employees)
def males_below_50_branchC := (1 - branchC_percentage_males_at_least_50) * (branchC_percentage_males * branchC_total_employees)

def total_males_below_50 := males_below_50_branchA + males_below_50_branchB + males_below_50_branchC

theorem total_males_below_50_is_2638 : total_males_below_50 = 2638 := 
by
  -- Numerical evaluation and equality verification here
  sorry

end total_males_below_50_is_2638_l74_74762


namespace wolf_nobel_laureates_l74_74163

theorem wolf_nobel_laureates (W N total W_prize N_prize N_noW N_W : ℕ)
  (h1 : W_prize = 31)
  (h2 : total = 50)
  (h3 : N_prize = 27)
  (h4 : N_noW + N_W = total - W_prize)
  (h5 : N_W = N_noW + 3)
  (h6 : N_prize = W + N_W) :
  W = 16 :=
by {
  sorry
}

end wolf_nobel_laureates_l74_74163


namespace problem_solution_l74_74771

theorem problem_solution (k x1 x2 y1 y2 : ℝ) 
  (h₁ : k ≠ 0) 
  (h₂ : y1 = k * x1) 
  (h₃ : y1 = -5 / x1) 
  (h₄ : y2 = k * x2) 
  (h₅ : y2 = -5 / x2) 
  (h₆ : x1 = -x2) 
  (h₇ : y1 = -y2) : 
  x1 * y2 - 3 * x2 * y1 = 10 := 
sorry

end problem_solution_l74_74771


namespace count_non_decreasing_digits_of_12022_l74_74892

/-- Proof that the number of digits left in the number 12022 that form a non-decreasing sequence is 3. -/
theorem count_non_decreasing_digits_of_12022 : 
  let num := [1, 2, 0, 2, 2]
  let remaining := [1, 2, 2] -- non-decreasing sequence from 12022
  List.length remaining = 3 :=
by
  let num := [1, 2, 0, 2, 2]
  let remaining := [1, 2, 2]
  have h : List.length remaining = 3 := rfl
  exact h

end count_non_decreasing_digits_of_12022_l74_74892


namespace rectangle_total_area_l74_74308

-- Let s be the side length of the smaller squares
variable (s : ℕ)

-- Define the areas of the squares
def smaller_square_area := s ^ 2
def larger_square_area := (3 * s) ^ 2

-- Define the total_area
def total_area : ℕ := 2 * smaller_square_area s + larger_square_area s

-- Assert the total area of the rectangle ABCD is 11s^2
theorem rectangle_total_area (s : ℕ) : total_area s = 11 * s ^ 2 := 
by 
  -- the proof is skipped
  sorry

end rectangle_total_area_l74_74308


namespace point_in_fourth_quadrant_l74_74360

def point : ℝ × ℝ := (3, -4)

def isFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def isSecondQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def isThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def isFourthQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : isFourthQuadrant point :=
by
  sorry

end point_in_fourth_quadrant_l74_74360


namespace coris_aunt_age_today_l74_74653

variable (Cori_age_now : ℕ) (age_diff : ℕ)

theorem coris_aunt_age_today (H1 : Cori_age_now = 3) (H2 : ∀ (Cori_age5 Aunt_age5 : ℕ), Cori_age5 = Cori_age_now + 5 → Aunt_age5 = 3 * Cori_age5 → Aunt_age5 - 5 = age_diff) :
  age_diff = 19 := 
by
  intros
  sorry

end coris_aunt_age_today_l74_74653


namespace smallest_number_increased_by_nine_divisible_by_8_11_24_l74_74074

theorem smallest_number_increased_by_nine_divisible_by_8_11_24 :
  ∃ x : ℕ, (x + 9) % 8 = 0 ∧ (x + 9) % 11 = 0 ∧ (x + 9) % 24 = 0 ∧ x = 255 :=
by
  sorry

end smallest_number_increased_by_nine_divisible_by_8_11_24_l74_74074


namespace smallest_cube_with_divisor_l74_74945

theorem smallest_cube_with_divisor (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ (m : ℕ), m = (p * q * r^2) ^ 3 ∧ (p * q^3 * r^5 ∣ m) :=
by
  sorry

end smallest_cube_with_divisor_l74_74945


namespace ajay_saves_each_month_l74_74059

def monthly_income : ℝ := 90000
def spend_household : ℝ := 0.50 * monthly_income
def spend_clothes : ℝ := 0.25 * monthly_income
def spend_medicines : ℝ := 0.15 * monthly_income
def total_spent : ℝ := spend_household + spend_clothes + spend_medicines
def amount_saved : ℝ := monthly_income - total_spent

theorem ajay_saves_each_month : amount_saved = 9000 :=
by sorry

end ajay_saves_each_month_l74_74059


namespace arcs_intersection_l74_74188

theorem arcs_intersection (k : ℕ) : (1 ≤ k ∧ k ≤ 99) ∧ ¬(∃ m : ℕ, k + 1 = 8 * m) ↔ ∃ n l : ℕ, (2 * l + 1) * 100 = (k + 1) * n ∧ n = 100 ∧ k < 100 := by
  sorry

end arcs_intersection_l74_74188


namespace line_through_point_parallel_l74_74134

theorem line_through_point_parallel (x y : ℝ) (h₁ : 2 * 2 + 4 * 3 + x = 0) (h₂ : x = -16) (h₃ : y = 8) :
  2 * x + 4 * y - 3 = 0 → x + 2 * y - 8 = 0 :=
by
  intro h₄
  sorry

end line_through_point_parallel_l74_74134


namespace unemployment_percentage_next_year_l74_74478

theorem unemployment_percentage_next_year (E U : ℝ) (h1 : E > 0) :
  ( (0.91 * (0.056 * E)) / (1.04 * E) ) * 100 = 4.9 := by
  sorry

end unemployment_percentage_next_year_l74_74478


namespace solve_equation_l74_74784

theorem solve_equation :
  ∀ x : ℝ, (101 * x ^ 2 - 18 * x + 1) ^ 2 - 121 * x ^ 2 * (101 * x ^ 2 - 18 * x + 1) + 2020 * x ^ 4 = 0 ↔ 
    x = 1 / 18 ∨ x = 1 / 9 :=
by
  intro x
  sorry

end solve_equation_l74_74784


namespace people_happy_correct_l74_74198

-- Define the size and happiness percentage of an institution.
variables (size : ℕ) (happiness_percentage : ℚ)

-- Assume the size is between 100 and 200.
axiom size_range : 100 ≤ size ∧ size ≤ 200

-- Assume the happiness percentage is between 0.6 and 0.95.
axiom happiness_percentage_range : 0.6 ≤ happiness_percentage ∧ happiness_percentage ≤ 0.95

-- Define the number of people made happy at an institution.
def people_made_happy (size : ℕ) (happiness_percentage : ℚ) : ℚ := 
  size * happiness_percentage

-- Theorem stating that the number of people made happy is as expected.
theorem people_happy_correct : 
  ∀ (size : ℕ) (happiness_percentage : ℚ), 
  100 ≤ size → size ≤ 200 → 
  0.6 ≤ happiness_percentage → happiness_percentage ≤ 0.95 → 
  people_made_happy size happiness_percentage = size * happiness_percentage := 
by 
  intros size happiness_percentage hsize1 hsize2 hperc1 hperc2
  unfold people_made_happy
  sorry

end people_happy_correct_l74_74198


namespace arithmetic_sequence_a10_l74_74070

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 1) (h3 : a 3 = 5) 
  (h_diff : d = (a 3 - a 1) / (3 - 1)) :
  a 10 = 19 := 
by 
  sorry

end arithmetic_sequence_a10_l74_74070


namespace product_to_difference_l74_74959

def x := 88 * 1.25
def y := 150 * 0.60
def z := 60 * 1.15

def product := x * y * z
def difference := x - y

theorem product_to_difference :
  product ^ difference = 683100 ^ 20 := 
sorry

end product_to_difference_l74_74959


namespace evaluate_expression_right_to_left_l74_74354

variable (a b c d : ℝ)

theorem evaluate_expression_right_to_left:
  (a * b + c - d) = (a * (b + c - d)) :=
by {
  -- Group operations from right to left according to the given condition
  sorry
}

end evaluate_expression_right_to_left_l74_74354


namespace laptop_price_l74_74721

theorem laptop_price (upfront_percent : ℝ) (upfront_payment full_price : ℝ)
  (h1 : upfront_percent = 0.20)
  (h2 : upfront_payment = 240)
  (h3 : upfront_payment = upfront_percent * full_price) :
  full_price = 1200 := 
sorry

end laptop_price_l74_74721


namespace volume_increase_factor_l74_74942

variable (π : ℝ) (r h : ℝ)

def original_volume : ℝ := π * r^2 * h

def new_volume : ℝ := π * (2 * r)^2 * (3 * h)

theorem volume_increase_factor : new_volume π r h = 12 * original_volume π r h :=
by
  -- Here we would include the proof that new_volume = 12 * original_volume
  sorry

end volume_increase_factor_l74_74942


namespace cake_eaten_fraction_l74_74989

noncomputable def cake_eaten_after_four_trips : ℚ :=
  let consumption_ratio := (1/3 : ℚ)
  let first_trip := consumption_ratio
  let second_trip := consumption_ratio * consumption_ratio
  let third_trip := second_trip * consumption_ratio
  let fourth_trip := third_trip * consumption_ratio
  first_trip + second_trip + third_trip + fourth_trip

theorem cake_eaten_fraction : cake_eaten_after_four_trips = (40 / 81 : ℚ) :=
by
  sorry

end cake_eaten_fraction_l74_74989


namespace diagonals_perpendicular_l74_74850

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 3 }
def B : Point := { x := 2, y := 6 }
def C : Point := { x := 6, y := -1 }
def D : Point := { x := -3, y := -4 }

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

theorem diagonals_perpendicular :
  let AC := vector A C
  let BD := vector B D
  dot_product AC BD = 0 :=
by
  let AC := vector A C
  let BD := vector B D
  sorry

end diagonals_perpendicular_l74_74850


namespace average_of_consecutive_sequences_l74_74551

theorem average_of_consecutive_sequences (a b : ℕ) (h : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) : 
    ((b + (b+1) + (b+2) + (b+3) + (b+4)) / 5) = a + 4 :=
by
  sorry

end average_of_consecutive_sequences_l74_74551


namespace sin_and_tan_inequality_l74_74977

theorem sin_and_tan_inequality (n : ℕ) (hn : 0 < n) :
  2 * Real.sin (1 / n) + Real.tan (1 / n) > 3 / n :=
sorry

end sin_and_tan_inequality_l74_74977


namespace sum_of_r_s_l74_74698

theorem sum_of_r_s (m : ℝ) (x : ℝ) (y : ℝ) (r s : ℝ) 
  (parabola_eqn : y = x^2 + 4) 
  (point_Q : (x, y) = (10, 5)) 
  (roots_rs : ∀ (m : ℝ), m^2 - 40*m + 4 = 0 → r < m → m < s)
  : r + s = 40 := 
sorry

end sum_of_r_s_l74_74698


namespace determine_phi_l74_74170

theorem determine_phi (phi : ℝ) (h : 0 < phi ∧ phi < π) :
  (∃ k : ℤ, phi = 2*k*π + (3*π/4)) :=
by
  sorry

end determine_phi_l74_74170


namespace circle_symmetric_line_l74_74298

theorem circle_symmetric_line (a b : ℝ) 
  (h1 : ∃ x y, x^2 + y^2 - 4 * x + 2 * y + 1 = 0)
  (h2 : ∀ x y, (x, y) = (2, -1))
  (h3 : 2 * a + 2 * b - 1 = 0) :
  ab ≤ 1 / 16 := sorry

end circle_symmetric_line_l74_74298


namespace Q1_Intersection_Q1_Union_Q2_l74_74928

namespace Example

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}

def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

-- Question 1: 
theorem Q1_Intersection (a : ℝ) (ha : a = -1) : 
  A ∩ B a = {x | -2 ≤ x ∧ x ≤ -1} :=
sorry

theorem Q1_Union (a : ℝ) (ha : a = -1) :
  A ∪ B a = {x | x ≤ 1 ∨ x ≥ 5} :=
sorry

-- Question 2:
theorem Q2 (a : ℝ) :
  (A ∩ B a = B a) ↔ (a ≤ -3 ∨ a > 2) :=
sorry

end Example

end Q1_Intersection_Q1_Union_Q2_l74_74928


namespace find_n_l74_74824

theorem find_n (n : ℚ) : 1 / 2 + 2 / 3 + 3 / 4 + n / 12 = 2 ↔ n = 1 := by
  -- proof here
  sorry

end find_n_l74_74824


namespace compare_pow_value_l74_74426

theorem compare_pow_value : 
  ∀ (x : ℝ) (n : ℕ), x = 0.01 → n = 1000 → (1 + x)^n > 1000 := 
by 
  intros x n hx hn
  rw [hx, hn]
  sorry

end compare_pow_value_l74_74426


namespace function_satisfies_condition_l74_74833

noncomputable def f : ℕ → ℕ := sorry

theorem function_satisfies_condition (f : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → f (n + 1) > (f n + f (f n)) / 2) :
  (∃ b : ℕ, ∀ n : ℕ, (n < b → f n = n) ∧ (n ≥ b → f n = n + 1)) :=
sorry

end function_satisfies_condition_l74_74833


namespace percentage_reduction_in_production_l74_74558

theorem percentage_reduction_in_production :
  let daily_production_rate := 10
  let days_in_year := 365
  let total_production_first_year := daily_production_rate * days_in_year
  let total_production_second_year := 3285
  let reduction_in_production := total_production_first_year - total_production_second_year
  let percentage_reduction := (reduction_in_production * 100) / total_production_first_year
  percentage_reduction = 10 :=
by
  let daily_production_rate := 10
  let days_in_year := 365
  let total_production_first_year := daily_production_rate * days_in_year
  let total_production_second_year := 3285
  let reduction_in_production := total_production_first_year - total_production_second_year
  let percentage_reduction := (reduction_in_production * 100) / total_production_first_year
  sorry

end percentage_reduction_in_production_l74_74558


namespace boys_passed_percentage_l74_74258

theorem boys_passed_percentage
  (total_candidates : ℝ)
  (total_girls : ℝ)
  (failed_percentage : ℝ)
  (girls_passed_percentage : ℝ)
  (boys_passed_percentage : ℝ) :
  total_candidates = 2000 →
  total_girls = 900 →
  failed_percentage = 70.2 →
  girls_passed_percentage = 32 →
  boys_passed_percentage = 28 :=
by
  sorry

end boys_passed_percentage_l74_74258


namespace box_triple_count_l74_74679

theorem box_triple_count (a b c : ℕ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a * b * c = 2 * (a * b + b * c + c * a)) :
  (a = 2 ∧ b = 8 ∧ c = 8) ∨ (a = 3 ∧ b = 6 ∧ c = 6) ∨ (a = 4 ∧ b = 4 ∧ c = 4) ∨ (a = 5 ∧ b = 5 ∧ c = 5) ∨ (a = 6 ∧ b = 6 ∧ c = 6) :=
sorry

end box_triple_count_l74_74679


namespace maximum_candies_after_20_hours_l74_74406

-- Define a function to compute the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Define the recursive function to model the candy process
def candies_after_hours (n : ℕ) (hours : ℕ) : ℕ :=
  if hours = 0 then n 
  else candies_after_hours (n + sum_of_digits n) (hours - 1)

theorem maximum_candies_after_20_hours :
  candies_after_hours 1 20 = 148 :=
sorry

end maximum_candies_after_20_hours_l74_74406


namespace total_time_l74_74241

theorem total_time {minutes seconds : ℕ} (hmin : minutes = 3450) (hsec : seconds = 7523) :
  ∃ h m s : ℕ, h = 59 ∧ m = 35 ∧ s = 23 :=
by
  sorry

end total_time_l74_74241


namespace monthly_interest_payment_l74_74879

theorem monthly_interest_payment (principal : ℝ) (annual_rate : ℝ) (months_in_year : ℝ) : 
  principal = 31200 → 
  annual_rate = 0.09 → 
  months_in_year = 12 → 
  (principal * annual_rate) / months_in_year = 234 := 
by 
  intros h_principal h_rate h_months
  rw [h_principal, h_rate, h_months]
  sorry

end monthly_interest_payment_l74_74879


namespace percentage_of_total_l74_74656

theorem percentage_of_total (total part : ℕ) (h₁ : total = 100) (h₂ : part = 30):
  (part / total) * 100 = 30 := by
  sorry

end percentage_of_total_l74_74656


namespace milk_volume_in_ounces_l74_74269

theorem milk_volume_in_ounces
  (packets : ℕ)
  (volume_per_packet_ml : ℕ)
  (ml_per_oz : ℕ)
  (total_volume_ml : ℕ)
  (total_volume_oz : ℕ)
  (h1 : packets = 150)
  (h2 : volume_per_packet_ml = 250)
  (h3 : ml_per_oz = 30)
  (h4 : total_volume_ml = packets * volume_per_packet_ml)
  (h5 : total_volume_oz = total_volume_ml / ml_per_oz) :
  total_volume_oz = 1250 :=
by
  sorry

end milk_volume_in_ounces_l74_74269


namespace quarters_total_l74_74840

variable (q1 q2 S: Nat)

def original_quarters := 760
def additional_quarters := 418

theorem quarters_total : S = original_quarters + additional_quarters :=
sorry

end quarters_total_l74_74840


namespace log_product_evaluation_l74_74460

noncomputable def evaluate_log_product : ℝ :=
  Real.log 9 / Real.log 2 * Real.log 16 / Real.log 3 * Real.log 27 / Real.log 7

theorem log_product_evaluation : evaluate_log_product = 24 := 
  sorry

end log_product_evaluation_l74_74460


namespace chloe_cherries_l74_74251

noncomputable def cherries_received (x y : ℝ) : Prop :=
  x = y + 8 ∧ y = x / 3

theorem chloe_cherries : ∃ (x : ℝ), ∀ (y : ℝ), cherries_received x y → x = 12 := 
by
  sorry

end chloe_cherries_l74_74251


namespace multiply_101_self_l74_74807

theorem multiply_101_self : 101 * 101 = 10201 := 
by
  -- Proof omitted
  sorry

end multiply_101_self_l74_74807


namespace four_corresponds_to_364_l74_74375

noncomputable def number_pattern (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 36
  | 3 => 363
  | 5 => 365
  | 36 => 2
  | _ => 0 -- Assume 0 as the default case

theorem four_corresponds_to_364 : number_pattern 4 = 364 :=
sorry

end four_corresponds_to_364_l74_74375


namespace each_child_play_time_l74_74045

theorem each_child_play_time (n_children : ℕ) (game_time : ℕ) (children_per_game : ℕ)
  (h1 : n_children = 8) (h2 : game_time = 120) (h3 : children_per_game = 2) :
  ((children_per_game * game_time) / n_children) = 30 :=
  sorry

end each_child_play_time_l74_74045


namespace no_such_function_exists_l74_74173

theorem no_such_function_exists (f : ℤ → ℤ) (h : ∀ m n : ℤ, f (m + f n) = f m - n) : false :=
sorry

end no_such_function_exists_l74_74173


namespace distance_covered_by_wheel_l74_74165

noncomputable def pi_num : ℝ := 3.14159

noncomputable def wheel_diameter : ℝ := 14

noncomputable def number_of_revolutions : ℝ := 33.03002729754322

noncomputable def circumference : ℝ := pi_num * wheel_diameter

noncomputable def calculated_distance : ℝ := circumference * number_of_revolutions

theorem distance_covered_by_wheel : 
  calculated_distance = 1452.996 :=
sorry

end distance_covered_by_wheel_l74_74165


namespace V3_is_correct_l74_74316

-- Definitions of the polynomial and Horner's method applied at x = -4
def f (x : ℤ) : ℤ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def V_3_value : ℤ := 
  let v0 := -4
  let v1 := v0 * 3 + 5
  let v2 := v0 * v1 + 6
  v0 * v2 + 79

theorem V3_is_correct : V_3_value = -57 := 
  by sorry

end V3_is_correct_l74_74316


namespace find_xyz_sum_l74_74135

theorem find_xyz_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 12)
  (h2 : y^2 + y * z + z^2 = 25)
  (h3 : z^2 + z * x + x^2 = 37) :
  x * y + y * z + z * x = 20 :=
sorry

end find_xyz_sum_l74_74135


namespace complex_division_l74_74235

def i_units := Complex.I

def numerator := (3 : ℂ) + i_units
def denominator := (1 : ℂ) + i_units
def expected_result := (2 : ℂ) - i_units

theorem complex_division :
  numerator / denominator = expected_result :=
by sorry

end complex_division_l74_74235


namespace total_sales_is_10400_l74_74209

-- Define the conditions
def tough_week_sales : ℝ := 800
def good_week_sales : ℝ := 2 * tough_week_sales
def good_weeks : ℕ := 5
def tough_weeks : ℕ := 3

-- Define the total sales function
def total_sales (good_sales : ℝ) (tough_sales : ℝ) (good_weeks : ℕ) (tough_weeks : ℕ) : ℝ :=
  good_weeks * good_sales + tough_weeks * tough_sales

-- Prove that the total sales is $10400
theorem total_sales_is_10400 : total_sales good_week_sales tough_week_sales good_weeks tough_weeks = 10400 := 
by
  sorry

end total_sales_is_10400_l74_74209


namespace trebled_resultant_is_correct_l74_74628

-- Definitions based on the conditions provided in step a)
def initial_number : ℕ := 5
def doubled_result : ℕ := initial_number * 2
def added_15_result : ℕ := doubled_result + 15
def trebled_resultant : ℕ := added_15_result * 3

-- We need to prove that the trebled resultant is equal to 75
theorem trebled_resultant_is_correct : trebled_resultant = 75 :=
by
  sorry

end trebled_resultant_is_correct_l74_74628


namespace problem_16_l74_74887

-- Definitions of the problem conditions
def trapezoid_inscribed_in_circle (r : ℝ) (a b : ℝ) : Prop :=
  r = 25 ∧ a = 14 ∧ b = 30 

def average_leg_length_of_trapezoid (a b : ℝ) (m : ℝ) : Prop :=
  a = 14 ∧ b = 30 ∧ m = 2000 

-- Using Lean to state the problem
theorem problem_16 (r a b m : ℝ) 
  (h1 : trapezoid_inscribed_in_circle r a b) 
  (h2 : average_leg_length_of_trapezoid a b m) : 
  m = 2000 := by
  sorry

end problem_16_l74_74887


namespace arithmetic_sum_example_l74_74494

def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sum_example (a1 d : ℤ) 
  (S20_eq_340 : S 20 a1 d = 340) :
  a 6 a1 d + a 9 a1 d + a 11 a1 d + a 16 a1 d = 68 :=
by
  sorry

end arithmetic_sum_example_l74_74494


namespace new_percentage_of_managers_is_98_l74_74006

def percentage_of_managers (initial_employees : ℕ) (initial_percentage_managers : ℕ) (managers_leaving : ℕ) : ℕ :=
  let initial_managers := initial_percentage_managers * initial_employees / 100
  let remaining_managers := initial_managers - managers_leaving
  let remaining_employees := initial_employees - managers_leaving
  (remaining_managers * 100) / remaining_employees

theorem new_percentage_of_managers_is_98 :
  percentage_of_managers 500 99 250 = 98 :=
by
  sorry

end new_percentage_of_managers_is_98_l74_74006


namespace servings_of_peanut_butter_l74_74192

-- Definitions from conditions
def total_peanut_butter : ℚ := 35 + 4/5
def serving_size : ℚ := 2 + 1/3

-- Theorem to be proved
theorem servings_of_peanut_butter :
  total_peanut_butter / serving_size = 15 + 17/35 := by
  sorry

end servings_of_peanut_butter_l74_74192


namespace race_head_start_l74_74496

theorem race_head_start
  (v_A v_B L x : ℝ)
  (h1 : v_A = (4 / 3) * v_B)
  (h2 : L / v_A = (L - x * L) / v_B) :
  x = 1 / 4 :=
sorry

end race_head_start_l74_74496


namespace value_of_n_l74_74556

theorem value_of_n (n : ℕ) (k : ℕ) (h : k = 11) (eqn : (1/2)^n * (1/81)^k = 1/18^22) : n = 22 :=
by
  sorry

end value_of_n_l74_74556


namespace matrix_pow_C_50_l74_74051

def C : Matrix (Fin 2) (Fin 2) ℤ := 
  !![3, 1; -4, -1]

theorem matrix_pow_C_50 : C^50 = !![101, 50; -200, -99] := 
  sorry

end matrix_pow_C_50_l74_74051


namespace shares_correct_l74_74635

open Real

-- Problem setup
def original_problem (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = 1020 ∧
  a = (3 / 4) * b ∧
  b = (2 / 3) * c ∧
  c = (1 / 4) * d ∧
  d = (5 / 6) * e

-- Goal
theorem shares_correct : ∃ (a b c d e : ℝ),
  original_problem a b c d e ∧
  abs (a - 58.17) < 0.01 ∧
  abs (b - 77.56) < 0.01 ∧
  abs (c - 116.34) < 0.01 ∧
  abs (d - 349.02) < 0.01 ∧
  abs (e - 419.42) < 0.01 := by
  sorry

end shares_correct_l74_74635


namespace positive_difference_l74_74617

theorem positive_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 3 * x = 27) : y - x = 9 :=
sorry

end positive_difference_l74_74617


namespace investment_period_l74_74790

theorem investment_period (x t : ℕ) (p_investment q_investment q_time : ℕ) (profit_ratio : ℚ):
  q_investment = 5 * x →
  p_investment = 7 * x →
  q_time = 16 →
  profit_ratio = 7 / 10 →
  7 * x * t = profit_ratio * 5 * x * q_time →
  t = 8 := sorry

end investment_period_l74_74790


namespace fraction_of_25_l74_74062

theorem fraction_of_25 (x : ℝ) (h1 : 0.65 * 40 = 26) (h2 : 26 = x * 25 + 6) : x = 4 / 5 :=
sorry

end fraction_of_25_l74_74062


namespace geometric_progressions_common_ratio_l74_74854

theorem geometric_progressions_common_ratio (a b p q : ℝ) :
  (∀ n : ℕ, (a * p^n + b * q^n) = (a * b) * ((p^n + q^n)/a)) →
  p = q := by
  sorry

end geometric_progressions_common_ratio_l74_74854


namespace sam_pennies_l74_74586

def pennies_from_washing_clothes (total_money_cents : ℤ) (quarters : ℤ) : ℤ :=
  total_money_cents - (quarters * 25)

theorem sam_pennies :
  pennies_from_washing_clothes 184 7 = 9 :=
by
  sorry

end sam_pennies_l74_74586


namespace find_central_cell_l74_74880

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l74_74880


namespace range_of_x_plus_y_l74_74444

theorem range_of_x_plus_y (x y : ℝ) (hx1 : y = 3 * ⌊x⌋ + 4) (hx2 : y = 4 * ⌊x - 3⌋ + 7) (hxnint : ¬ ∃ z : ℤ, x = z): 
  40 < x + y ∧ x + y < 41 :=
by
  sorry

end range_of_x_plus_y_l74_74444


namespace marks_in_mathematics_l74_74042

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

end marks_in_mathematics_l74_74042


namespace min_value_function_l74_74932

theorem min_value_function (x y: ℝ) (hx: x > 2) (hy: y > 2) : 
  (∃c: ℝ, c = (x^3/(y - 2) + y^3/(x - 2)) ∧ ∀x y: ℝ, x > 2 → y > 2 → (x^3/(y - 2) + y^3/(x - 2)) ≥ c) ∧ c = 96 :=
sorry

end min_value_function_l74_74932


namespace min_max_value_sum_l74_74593

variable (a b c d e : ℝ)

theorem min_max_value_sum :
  a + b + c + d + e = 10 ∧ a^2 + b^2 + c^2 + d^2 + e^2 = 30 →
  let expr := 5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4)
  let m := 42
  let M := 52
  m + M = 94 := sorry

end min_max_value_sum_l74_74593


namespace all_have_perp_property_l74_74345

def M₁ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, x^3 - 2 * x^2 + 3)}
def M₂ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, Real.log (2 - x) / Real.log 2)}
def M₃ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 2 - 2^x)}
def M₄ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 1 - Real.sin x)}

def perp_property (M : Set (ℝ × ℝ)) : Prop :=
∀ p ∈ M, ∃ q ∈ M, p.1 * q.1 + p.2 * q.2 = 0

-- Theorem statement
theorem all_have_perp_property :
  perp_property M₁ ∧ perp_property M₂ ∧ perp_property M₃ ∧ perp_property M₄ :=
sorry

end all_have_perp_property_l74_74345


namespace calvin_buys_chips_days_per_week_l74_74450

-- Define the constants based on the problem conditions
def cost_per_pack : ℝ := 0.50
def total_amount_spent : ℝ := 10
def number_of_weeks : ℕ := 4

-- Define the proof statement
theorem calvin_buys_chips_days_per_week : 
  (total_amount_spent / cost_per_pack) / number_of_weeks = 5 := 
by
  -- Placeholder proof
  sorry

end calvin_buys_chips_days_per_week_l74_74450


namespace original_savings_l74_74200

/-- Linda spent 3/4 of her savings on furniture and the rest on a TV costing $210. 
    What were her original savings? -/
theorem original_savings (S : ℝ) (h1 : S * (1/4) = 210) : S = 840 :=
by
  sorry

end original_savings_l74_74200


namespace product_equation_l74_74408

theorem product_equation (a b : ℝ) (h1 : ∀ (a b : ℝ), 0.2 * b = 0.9 * a - b) : 
  0.9 * a - b = 0.2 * b :=
by
  sorry

end product_equation_l74_74408


namespace number_of_women_l74_74157

-- Definitions for the given conditions
variables (m w : ℝ)
variable (x : ℝ)

-- Conditions
def cond1 : Prop := 3 * m + 8 * w = 6 * m + 2 * w
def cond2 : Prop := 4 * m + x * w = 0.9285714285714286 * (3 * m + 8 * w)

-- Theorem to prove the number of women in the third group (x)
theorem number_of_women (h1 : cond1 m w) (h2 : cond2 m w x) : x = 5 :=
sorry

end number_of_women_l74_74157


namespace cuboid_height_l74_74743

theorem cuboid_height
  (volume : ℝ)
  (width : ℝ)
  (length : ℝ)
  (height : ℝ)
  (h_volume : volume = 315)
  (h_width : width = 9)
  (h_length : length = 7)
  (h_volume_eq : volume = length * width * height) :
  height = 5 :=
by
  sorry

end cuboid_height_l74_74743


namespace percentage_difference_l74_74960

theorem percentage_difference (x : ℝ) : 
  (62 / 100) * 150 - (x / 100) * 250 = 43 → x = 20 :=
by
  intro h
  sorry

end percentage_difference_l74_74960


namespace johns_commute_distance_l74_74331

theorem johns_commute_distance
  (y : ℝ)  -- distance in miles
  (h1 : 200 * (y / 200) = y)  -- John usually takes 200 minutes, so usual speed is y/200 miles per minute
  (h2 : 320 = (y / (2 * (y / 200))) + (y / (2 * ((y / 200) - 15/60)))) -- Total journey time on the foggy day
  : y = 92 :=
sorry

end johns_commute_distance_l74_74331


namespace calculator_change_problem_l74_74065

theorem calculator_change_problem :
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  change_received = 28 := by
{
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  have h1 : scientific_cost = 16 := sorry
  have h2 : graphing_cost = 48 := sorry
  have h3 : total_cost = 72 := sorry
  have h4 : change_received = 28 := sorry
  exact h4
}

end calculator_change_problem_l74_74065


namespace range_of_b_l74_74149

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem range_of_b (b : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (f x1 = b) ∧ (f x2 = b) ∧ (f x3 = b))
  ↔ (-4 / 3 < b ∧ b < 28 / 3) :=
by
  sorry

end range_of_b_l74_74149


namespace assume_proof_by_contradiction_l74_74900

theorem assume_proof_by_contradiction (a b : ℤ) (hab : ∃ k : ℤ, ab = 3 * k) :
  (¬ (∃ k : ℤ, a = 3 * k) ∧ ¬ (∃ k : ℤ, b = 3 * k)) :=
sorry

end assume_proof_by_contradiction_l74_74900


namespace length_of_rectangle_l74_74068

theorem length_of_rectangle (L : ℝ) (W : ℝ) (A_triangle : ℝ) (hW : W = 4) (hA_triangle : A_triangle = 60)
  (hRatio : (L * W) / A_triangle = 2 / 5) : L = 6 :=
by
  sorry

end length_of_rectangle_l74_74068


namespace endpoint_coordinates_l74_74891

theorem endpoint_coordinates (x y : ℝ) (h : y > 0) :
  let slope_condition := (y - 2) / (x - 2) = 3 / 4
  let distance_condition := (x - 2) ^ 2 + (y - 2) ^ 2 = 64
  slope_condition → distance_condition → 
    (x = 2 + (4 * Real.sqrt 5475) / 25 ∧ y = (3 / 4) * (2 + (4 * Real.sqrt 5475) / 25) + 1 / 2) ∨
    (x = 2 - (4 * Real.sqrt 5475) / 25 ∧ y = (3 / 4) * (2 - (4 * Real.sqrt 5475) / 25) + 1 / 2) :=
by
  intros slope_condition distance_condition
  sorry

end endpoint_coordinates_l74_74891


namespace longest_side_of_triangle_l74_74545

theorem longest_side_of_triangle (x : ℝ) (a b c : ℝ)
  (h1 : a = 5)
  (h2 : b = 2 * x + 3)
  (h3 : c = 3 * x - 2)
  (h4 : a + b + c = 41) :
  c = 19 :=
by
  sorry

end longest_side_of_triangle_l74_74545


namespace tan_product_pi_nine_l74_74836

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l74_74836


namespace distance_to_destination_l74_74629

theorem distance_to_destination :
  ∀ (D : ℝ) (T : ℝ),
    (15:ℝ) = T →
    (30:ℝ) = T / 2 →
    T - (T / 2) = 3 →
    D = 15 * T → D = 90 :=
by
  intros D T Theon_speed Yara_speed time_difference distance_calc
  sorry

end distance_to_destination_l74_74629


namespace january_31_is_friday_l74_74853

theorem january_31_is_friday (h : ∀ (d : ℕ), (d % 7 = 0 → d = 1)) : ∀ d, (d = 31) → (d % 7 = 3) :=
by
  sorry

end january_31_is_friday_l74_74853


namespace inequality_solution_range_of_a_l74_74336

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

def range_y := Set.Icc (-2 : ℝ) 2

def subset_property (a : ℝ) : Prop := 
  Set.Icc a (2 * a - 1) ⊆ range_y

theorem inequality_solution (x : ℝ) :
  f x ≤ x^2 - 3 * x + 1 ↔ x ≤ 1 ∨ x ≥ 3 := sorry

theorem range_of_a (a : ℝ) :
  subset_property a ↔ 1 ≤ a ∧ a ≤ 3 / 2 := sorry

end inequality_solution_range_of_a_l74_74336


namespace find_ab_l74_74570

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 :=
sorry

end find_ab_l74_74570


namespace no_perfect_squares_in_seq_l74_74242

def seq (x : ℕ → ℤ) : Prop :=
  x 0 = 1 ∧ x 1 = 3 ∧ ∀ n : ℕ, 0 < n → x (n + 1) = 6 * x n - x (n - 1)

theorem no_perfect_squares_in_seq (x : ℕ → ℤ) (n : ℕ) (h_seq : seq x) :
  ¬ ∃ k : ℤ, k * k = x (n + 1) :=
by
  sorry

end no_perfect_squares_in_seq_l74_74242


namespace area_of_triangle_ABC_l74_74781

noncomputable def distance (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem area_of_triangle_ABC (A B C O : ℝ × ℝ)
  (h_isosceles_right : ∃ d: ℝ, distance A B = d ∧ distance A C = d ∧ distance B C = Real.sqrt (2 * d^2))
  (h_A_right : A = (0, 0))
  (h_OA : distance O A = 5)
  (h_OB : distance O B = 7)
  (h_OC : distance O C = 3) :
  ∃ S : ℝ, S = (29 / 2) + (5 / 2) * Real.sqrt 17 :=
sorry

end area_of_triangle_ABC_l74_74781


namespace inequality_holds_for_minimal_a_l74_74295

theorem inequality_holds_for_minimal_a :
  ∀ (x : ℝ), (1 ≤ x) → (x ≤ 4) → (1 + x) * Real.log x + x ≤ x * 1.725 :=
by
  intros x h1 h2
  sorry

end inequality_holds_for_minimal_a_l74_74295


namespace check_prime_large_number_l74_74870

def large_number := 23021^377 - 1

theorem check_prime_large_number : ¬ Prime large_number := by
  sorry

end check_prime_large_number_l74_74870


namespace no_solution_for_s_l74_74523

theorem no_solution_for_s : ∀ s : ℝ,
  (s^2 - 6 * s + 8) / (s^2 - 9 * s + 20) ≠ (s^2 - 3 * s - 18) / (s^2 - 2 * s - 15) :=
by
  intros s
  sorry

end no_solution_for_s_l74_74523


namespace sin_alpha_cos_half_beta_minus_alpha_l74_74666

open Real

noncomputable def problem_condition (α β : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  sin (π / 3 - α) = 3 / 5 ∧
  cos (β / 2 - π / 3) = 2 * sqrt 5 / 5

theorem sin_alpha (α β : ℝ) (h : problem_condition α β) : 
  sin α = (4 * sqrt 3 - 3) / 10 := sorry

theorem cos_half_beta_minus_alpha (α β : ℝ) (h : problem_condition α β) :
  cos (β / 2 - α) = 11 * sqrt 5 / 25 := sorry

end sin_alpha_cos_half_beta_minus_alpha_l74_74666


namespace arithmetic_sequence_problem_l74_74703

-- Define the arithmetic sequence and related sum functions
def a_n (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

def S (a1 d : ℤ) (n : ℕ) : ℤ :=
  (a1 + a_n a1 d n) * n / 2

-- Problem statement: proving a_5 = -1 given the conditions
theorem arithmetic_sequence_problem :
  (∃ (a1 d : ℕ), S a1 d 2 = S a1 d 6 ∧ a_n a1 d 4 = 1) → a_n a1 d 5 = -1 :=
by
  -- Assume the statement and then skip the proof
  sorry

end arithmetic_sequence_problem_l74_74703


namespace min_L_pieces_correct_l74_74487

noncomputable def min_L_pieces : ℕ :=
  have pieces : Nat := 11
  pieces

theorem min_L_pieces_correct :
  min_L_pieces = 11 := 
by
  sorry

end min_L_pieces_correct_l74_74487


namespace sector_area_l74_74908

theorem sector_area (α r : ℝ) (hα : α = 2) (h_r : r = 1 / Real.sin 1) : 
  (1 / 2) * r^2 * α = 1 / (Real.sin 1)^2 :=
by
  sorry

end sector_area_l74_74908


namespace system_of_inequalities_solution_set_quadratic_equation_when_m_is_2_l74_74392

theorem system_of_inequalities_solution_set : 
  (∀ x : ℝ, (2 * x - 1 < 7) → (x + 1 > 2) ↔ (1 < x ∧ x < 4)) := 
by 
  sorry

theorem quadratic_equation_when_m_is_2 : 
  (∀ x : ℝ, x^2 - 2 * x - 2 = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3)) := 
by 
  sorry

end system_of_inequalities_solution_set_quadratic_equation_when_m_is_2_l74_74392


namespace central_angle_unit_circle_l74_74715

theorem central_angle_unit_circle :
  ∀ (θ : ℝ), (∃ (A : ℝ), A = 1 ∧ (A = 1 / 2 * θ)) → θ = 2 :=
by
  intro θ
  rintro ⟨A, hA1, hA2⟩
  sorry

end central_angle_unit_circle_l74_74715


namespace john_tran_probability_2_9_l74_74640

def johnArrivalProbability (train_start train_end john_min john_max: ℕ) : ℚ := 
  let overlap_area := ((train_end - train_start - 15) * 15) / 2 
  let total_area := (john_max - john_min) * (train_end - train_start)
  overlap_area / total_area

theorem john_tran_probability_2_9 :
  johnArrivalProbability 30 90 0 90 = 2 / 9 := by
  sorry

end john_tran_probability_2_9_l74_74640


namespace remi_water_consumption_proof_l74_74150

-- Definitions for the conditions
def daily_consumption (bottle_volume : ℕ) (refills_per_day : ℕ) : ℕ :=
  bottle_volume * refills_per_day

def total_spillage (spill1 : ℕ) (spill2 : ℕ) : ℕ :=
  spill1 + spill2

def total_consumption (daily : ℕ) (days : ℕ) (spill : ℕ) : ℕ :=
  (daily * days) - spill

-- Theorem proving the number of days d
theorem remi_water_consumption_proof (bottle_volume : ℕ) (refills_per_day : ℕ)
  (spill1 spill2 total_water : ℕ) (d : ℕ)
  (h1 : bottle_volume = 20) (h2 : refills_per_day = 3)
  (h3 : spill1 = 5) (h4 : spill2 = 8)
  (h5 : total_water = 407) :
  total_consumption (daily_consumption bottle_volume refills_per_day) d
    (total_spillage spill1 spill2) = total_water → d = 7 := 
by
  -- Assuming the hypotheses to show the equality
  intro h
  have daily := h1 ▸ h2 ▸ 20 * 3 -- ⇒ daily = 60
  have spillage := h3 ▸ h4 ▸ 5 + 8 -- ⇒ spillage = 13
  rw [daily_consumption, total_spillage, h5] at h
  rw [h1, h2, h3, h4] at h -- Substitute conditions in the hypothesis
  sorry -- place a placeholder for the actual proof

end remi_water_consumption_proof_l74_74150


namespace amy_owes_thirty_l74_74189

variable (A D : ℝ)

theorem amy_owes_thirty
  (total_pledged remaining_owed sally_carl_owe derek_half_amys_owes : ℝ)
  (h1 : total_pledged = 285)
  (h2 : remaining_owed = 400 - total_pledged)
  (h3 : sally_carl_owe = 35 + 35)
  (h4 : derek_half_amys_owes = A / 2)
  (h5 : remaining_owed - sally_carl_owe = 45)
  (h6 : 45 = A + (A / 2)) :
  A = 30 :=
by
  -- Proof steps skipped
  sorry

end amy_owes_thirty_l74_74189


namespace studentsInBandOrSports_l74_74306

-- conditions definitions
def totalStudents : ℕ := 320
def studentsInBand : ℕ := 85
def studentsInSports : ℕ := 200
def studentsInBoth : ℕ := 60

-- theorem statement
theorem studentsInBandOrSports : studentsInBand + studentsInSports - studentsInBoth = 225 :=
by
  sorry

end studentsInBandOrSports_l74_74306


namespace find_m3_minus_2mn_plus_n3_l74_74898

theorem find_m3_minus_2mn_plus_n3 (m n : ℝ) (h1 : m^2 = n + 2) (h2 : n^2 = m + 2) (h3 : m ≠ n) : m^3 - 2 * m * n + n^3 = -2 := by
  sorry

end find_m3_minus_2mn_plus_n3_l74_74898


namespace length_of_living_room_l74_74078

theorem length_of_living_room
  (l : ℝ) -- length of the living room
  (w : ℝ) -- width of the living room
  (boxes_coverage : ℝ) -- area covered by one box
  (initial_area : ℝ) -- area already covered
  (additional_boxes : ℕ) -- additional boxes required
  (total_area : ℝ) -- total area required
  (w_condition : w = 20)
  (boxes_coverage_condition : boxes_coverage = 10)
  (initial_area_condition : initial_area = 250)
  (additional_boxes_condition : additional_boxes = 7)
  (total_area_condition : total_area = l * w)
  (full_coverage_condition : additional_boxes * boxes_coverage + initial_area = total_area) :
  l = 16 := by
  sorry

end length_of_living_room_l74_74078


namespace expand_binomials_l74_74822

theorem expand_binomials (x : ℝ) : (x - 3) * (4 * x + 8) = 4 * x^2 - 4 * x - 24 :=
by
  sorry

end expand_binomials_l74_74822


namespace statement_B_is_algorithm_l74_74825

def is_algorithm (statement : String) : Prop := 
  statement = "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."

def condition_A : String := "At home, it is generally the mother who cooks."
def condition_B : String := "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."
def condition_C : String := "Cooking outdoors is called a picnic."
def condition_D : String := "Rice is necessary for cooking."

theorem statement_B_is_algorithm : is_algorithm condition_B :=
by
  sorry

end statement_B_is_algorithm_l74_74825


namespace greatest_decimal_is_7391_l74_74490

noncomputable def decimal_conversion (n d : ℕ) : ℝ :=
  n / d

noncomputable def forty_two_percent_of (r : ℝ) : ℝ :=
  0.42 * r

theorem greatest_decimal_is_7391 :
  let a := forty_two_percent_of (decimal_conversion 7 11)
  let b := decimal_conversion 17 23
  let c := 0.7391
  let d := decimal_conversion 29 47
  a < b ∧ a < c ∧ a < d ∧ b = c ∧ d < b :=
by
  have dec1 := forty_two_percent_of (decimal_conversion 7 11)
  have dec2 := decimal_conversion 17 23
  have dec3 := 0.7391
  have dec4 := decimal_conversion 29 47
  sorry

end greatest_decimal_is_7391_l74_74490


namespace add_base_3_l74_74619

def base3_addition : Prop :=
  2 + (1 * 3^2 + 2 * 3^1 + 0 * 3^0) + 
  (2 * 3^2 + 0 * 3^1 + 1 * 3^0) + 
  (1 * 3^3 + 2 * 3^1 + 0 * 3^0) = 
  (1 * 3^3) + (1 * 3^2) + (0 * 3^1) + (2 * 3^0)

theorem add_base_3 : base3_addition :=
by 
  -- We will skip the proof as per instructions
  sorry

end add_base_3_l74_74619


namespace eugene_swim_time_l74_74786

-- Define the conditions
variable (S : ℕ) -- Swim time on Sunday
variable (swim_time_mon : ℕ := 30) -- Swim time on Monday
variable (swim_time_tue : ℕ := 45) -- Swim time on Tuesday
variable (average_swim_time : ℕ := 34) -- Average swim time over three days

-- The total swim time over three days
def total_swim_time := S + swim_time_mon + swim_time_tue

-- The problem statement: Prove that given the conditions, Eugene swam for 27 minutes on Sunday.
theorem eugene_swim_time : total_swim_time S = 3 * average_swim_time → S = 27 := by
  -- Proof process will follow here
  sorry

end eugene_swim_time_l74_74786


namespace evaluate_expression_l74_74294

theorem evaluate_expression : 2 + 0 - 2 * 0 = 2 :=
by
  sorry

end evaluate_expression_l74_74294


namespace find_plaid_shirts_l74_74986

def total_shirts : ℕ := 5
def total_pants : ℕ := 24
def total_items : ℕ := total_shirts + total_pants
def neither_plaid_nor_purple : ℕ := 21
def total_plaid_or_purple : ℕ := total_items - neither_plaid_nor_purple
def purple_pants : ℕ := 5
def plaid_shirts (p : ℕ) : Prop := total_plaid_or_purple - purple_pants = p

theorem find_plaid_shirts : plaid_shirts 3 := by
  unfold plaid_shirts
  repeat { sorry }

end find_plaid_shirts_l74_74986


namespace maximum_value_of_w_l74_74644

variables (x y : ℝ)

def condition : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

def w (x y : ℝ) := 4 * x + 3 * y

theorem maximum_value_of_w : ∃ x y, condition x y ∧ w x y = 74 :=
sorry

end maximum_value_of_w_l74_74644


namespace number_of_valid_arrangements_l74_74091

def total_permutations (n : ℕ) : ℕ := n.factorial

def valid_permutations (total : ℕ) (block : ℕ) (specific_restriction : ℕ) : ℕ :=
  total - specific_restriction

theorem number_of_valid_arrangements : valid_permutations (total_permutations 5) 48 24 = 96 :=
by
  sorry

end number_of_valid_arrangements_l74_74091


namespace range_of_h_l74_74122

def f (x : ℝ) : ℝ := 4 * x - 3
def h (x : ℝ) : ℝ := f (f (f x))

theorem range_of_h : 
  (∀ x, -1 ≤ x ∧ x ≤ 3 → -127 ≤ h x ∧ h x ≤ 129) :=
by
  sorry

end range_of_h_l74_74122


namespace prime_saturated_96_l74_74897

def is_prime_saturated (d : ℕ) : Prop :=
  let prime_factors := [2, 3]  -- list of the different positive prime factors of 96
  prime_factors.prod < d       -- the product of prime factors should be less than d

theorem prime_saturated_96 : is_prime_saturated 96 :=
by
  sorry

end prime_saturated_96_l74_74897


namespace skipping_rates_l74_74374

theorem skipping_rates (x y : ℕ) (h₀ : 300 / (x + 19) = 270 / x) (h₁ : y = x + 19) :
  x = 171 ∧ y = 190 := by
  sorry

end skipping_rates_l74_74374


namespace intersection_eq_l74_74665

open Set

-- Define the sets A and B according to the given conditions
def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | (x - 1) * (x + 2) < 0}

-- Define the intended intersection result
def C : Set ℤ := {-1, 0}

-- The theorem to prove
theorem intersection_eq : A ∩ {x | (x - 1) * (x + 2) < 0} = C := by
  sorry

end intersection_eq_l74_74665


namespace perfect_square_difference_l74_74633

theorem perfect_square_difference (m n : ℕ) (h : 2001 * m^2 + m = 2002 * n^2 + n) : ∃ k : ℕ, k^2 = m - n :=
sorry

end perfect_square_difference_l74_74633


namespace yellow_side_probability_correct_l74_74581

-- Define the problem scenario
structure CardBox where
  total_cards : ℕ := 8
  green_green_cards : ℕ := 4
  green_yellow_cards : ℕ := 2
  yellow_yellow_cards : ℕ := 2

noncomputable def yellow_side_probability 
  (box : CardBox)
  (picked_is_yellow : Bool) : ℚ :=
  if picked_is_yellow then
    let total_yellow_sides := 2 * box.green_yellow_cards + 2 * box.yellow_yellow_cards
    let yellow_yellow_sides := 2 * box.yellow_yellow_cards
    yellow_yellow_sides / total_yellow_sides
  else 0

theorem yellow_side_probability_correct :
  yellow_side_probability {total_cards := 8, green_green_cards := 4, green_yellow_cards := 2, yellow_yellow_cards := 2} true = 2 / 3 :=
by 
  sorry

end yellow_side_probability_correct_l74_74581


namespace measure_of_angle_R_l74_74112

variable (S T A R : ℝ) -- Represent the angles as real numbers.

-- The conditions given in the problem.
axiom angles_congruent : S = T ∧ T = A ∧ A = R
axiom angle_A_equals_angle_S : A = S

-- Statement: Prove that the measure of angle R is 108 degrees.
theorem measure_of_angle_R : R = 108 :=
by
  sorry

end measure_of_angle_R_l74_74112


namespace andrew_correct_answer_l74_74476

variable {x : ℕ}

theorem andrew_correct_answer (h : (x - 8) / 7 = 15) : (x - 5) / 11 = 10 :=
by
  sorry

end andrew_correct_answer_l74_74476


namespace compute_difference_a_b_l74_74274

-- Define the initial amounts paid by Alex, Bob, and Carol
def alex_paid := 120
def bob_paid := 150
def carol_paid := 210

-- Define the total amount and equal share
def total_costs := alex_paid + bob_paid + carol_paid
def equal_share := total_costs / 3

-- Define the amounts Alex and Carol gave to Bob, satisfying their balances
def a := equal_share - alex_paid
def b := carol_paid - equal_share

-- Lean 4 statement to prove a - b = 30
theorem compute_difference_a_b : a - b = 30 := by
  sorry

end compute_difference_a_b_l74_74274


namespace find_speed_of_second_boy_l74_74910

theorem find_speed_of_second_boy
  (v : ℝ)
  (speed_first_boy : ℝ)
  (distance_apart : ℝ)
  (time_taken : ℝ)
  (h1 : speed_first_boy = 5.3)
  (h2 : distance_apart = 10.5)
  (h3 : time_taken = 35) :
  v = 5.6 :=
by {
  -- translation of the steps to work on the proof
  -- sorry is used to indicate that the proof is not provided here
  sorry
}

end find_speed_of_second_boy_l74_74910


namespace problem_I_problem_II_l74_74320

def setA : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≤ 0}

theorem problem_I (a : ℝ) : (setB a ⊆ setA) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

theorem problem_II (a : ℝ) : (setA ∩ setB a = {1}) ↔ a ≤ 1 := by
  sorry

end problem_I_problem_II_l74_74320


namespace centroid_path_is_ellipse_l74_74303

theorem centroid_path_is_ellipse
  (b r : ℝ)
  (C : ℝ → ℝ × ℝ)
  (H1 : ∃ t θ, C t = (r * Real.cos θ, r * Real.sin θ))
  (G : ℝ → ℝ × ℝ)
  (H2 : ∀ t, G t = (1 / 3 * (b + (C t).fst), 1 / 3 * ((C t).snd))) :
  ∃ a c : ℝ, ∀ t, (G t).fst^2 / a^2 + (G t).snd^2 / c^2 = 1 :=
sorry

end centroid_path_is_ellipse_l74_74303


namespace mean_age_is_10_l74_74130

def ages : List ℤ := [7, 7, 7, 14, 15]

theorem mean_age_is_10 : (List.sum ages : ℤ) / (ages.length : ℤ) = 10 := by
-- sorry placeholder for the actual proof
sorry

end mean_age_is_10_l74_74130


namespace find_k_and_angle_l74_74643

def vector := ℝ × ℝ

def dot_product (u v: vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def orthogonal (u v: vector) : Prop :=
  dot_product u v = 0

theorem find_k_and_angle (k : ℝ) :
  let a : vector := (3, -1)
  let b : vector := (1, k)
  orthogonal a b →
  (k = 3 ∧ dot_product (3+1, -1+3) (3-1, -1-3) = 0) :=
by
  intros
  sorry

end find_k_and_angle_l74_74643


namespace lcm_180_616_l74_74106

theorem lcm_180_616 : Nat.lcm 180 616 = 27720 := 
by
  sorry

end lcm_180_616_l74_74106


namespace chocolates_difference_l74_74203

-- Conditions
def Robert_chocolates : Nat := 13
def Nickel_chocolates : Nat := 4

-- Statement
theorem chocolates_difference : (Robert_chocolates - Nickel_chocolates) = 9 := by
  sorry

end chocolates_difference_l74_74203


namespace range_of_a_l74_74579

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + (1 / 2) * a

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) ↔ (0 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l74_74579


namespace percentage_of_b_l74_74277

variable (a b c p : ℝ)

theorem percentage_of_b (h1 : 0.06 * a = 12) (h2 : p * b = 6) (h3 : c = b / a) : 
  p = 6 / (200 * c) := by
  sorry

end percentage_of_b_l74_74277


namespace actual_distance_traveled_l74_74642

theorem actual_distance_traveled (D : ℕ) (h : (D:ℚ) / 12 = (D + 20) / 16) : D = 60 :=
sorry

end actual_distance_traveled_l74_74642


namespace find_B_l74_74373

theorem find_B (A B : Nat) (hA : A ≤ 9) (hB : B ≤ 9) (h_eq : 6 * A + 10 * B + 2 = 77) : B = 1 :=
by
-- proof steps would go here
sorry

end find_B_l74_74373


namespace ellipse_foci_distance_l74_74639

noncomputable def distance_between_foci : ℝ := 2 * Real.sqrt 29

theorem ellipse_foci_distance : 
  ∀ (x y : ℝ), 
  (Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25) → 
  distance_between_foci = 2 * Real.sqrt 29 := 
by
  intros x y h
  -- proof goes here (skipped)
  sorry

end ellipse_foci_distance_l74_74639


namespace initial_scissors_l74_74572

-- Define conditions as per the problem
def Keith_placed (added : ℕ) : Prop := added = 22
def total_now (total : ℕ) : Prop := total = 76

-- Define the problem statement as a theorem
theorem initial_scissors (added total initial : ℕ) (h1 : Keith_placed added) (h2 : total_now total) 
  (h3 : total = initial + added) : initial = 54 := by
  -- This is where the proof would go
  sorry

end initial_scissors_l74_74572


namespace cake_slices_l74_74728

theorem cake_slices (S : ℕ) (h : 347 * S = 6 * 375 + 526) : S = 8 :=
sorry

end cake_slices_l74_74728


namespace find_prices_max_basketballs_l74_74182

-- Define price of basketballs and soccer balls
def basketball_price : ℕ := 80
def soccer_ball_price : ℕ := 50

-- Define the equations given in the problem
theorem find_prices (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 310)
  (h2 : 5 * x + 2 * y = 500) : 
  x = basketball_price ∧ y = soccer_ball_price :=
sorry

-- Define the maximum number of basketballs given the cost constraints
theorem max_basketballs (m : ℕ)
  (htotal : m + (60 - m) = 60)
  (hcost : 80 * m + 50 * (60 - m) ≤ 4000) : 
  m ≤ 33 :=
sorry

end find_prices_max_basketballs_l74_74182


namespace calculate_expression_l74_74440

theorem calculate_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) * (3 + 2) = X :=
by
  sorry

end calculate_expression_l74_74440


namespace arnold_danny_age_l74_74827

theorem arnold_danny_age (x : ℕ) : (x + 1) * (x + 1) = x * x + 17 → x = 8 :=
by
  sorry

end arnold_danny_age_l74_74827


namespace bus_trip_speed_l74_74023

theorem bus_trip_speed :
  ∃ v : ℝ, v > 0 ∧ (660 / v - 1 = 660 / (v + 5)) ∧ v = 55 :=
by
  sorry

end bus_trip_speed_l74_74023


namespace david_remaining_money_l74_74844

noncomputable def initial_funds : ℝ := 1500
noncomputable def spent_on_accommodations : ℝ := 400
noncomputable def spent_on_food_eur : ℝ := 300
noncomputable def eur_to_usd : ℝ := 1.10
noncomputable def spent_on_souvenirs_yen : ℝ := 5000
noncomputable def yen_to_usd : ℝ := 0.009
noncomputable def loan_to_friend : ℝ := 200
noncomputable def difference : ℝ := 500

noncomputable def spent_on_food_usd : ℝ := spent_on_food_eur * eur_to_usd
noncomputable def spent_on_souvenirs_usd : ℝ := spent_on_souvenirs_yen * yen_to_usd
noncomputable def total_spent_excluding_loan : ℝ := spent_on_accommodations + spent_on_food_usd + spent_on_souvenirs_usd

theorem david_remaining_money : 
  initial_funds - total_spent_excluding_loan - difference = 275 :=
by
  sorry

end david_remaining_money_l74_74844


namespace complement_union_l74_74060

-- Definition of the universal set U
def U : Set ℤ := {x | x^2 - 5 * x - 6 ≤ 0}

-- Definition of set A
def A : Set ℤ := {x | x * (2 - x) ≥ 0}

-- Definition of set B
def B : Set ℤ := {1, 2, 3}

-- The proof statement
theorem complement_union (h : U = {x | x^2 - 5 * x - 6 ≤ 0} ∧ 
                           A = {x | x * (2 - x) ≥ 0} ∧ 
                           B = {1, 2, 3}) : 
  U \ (A ∪ B) = {-1, 4, 5, 6} :=
by {
  sorry
}

end complement_union_l74_74060


namespace find_four_numbers_l74_74191

theorem find_four_numbers (a b c d : ℕ) 
  (h1 : a + b = 2024) 
  (h2 : a + c = 2026) 
  (h3 : a + d = 2030) 
  (h4 : b + c = 2028) 
  (h5 : b + d = 2032) 
  (h6 : c + d = 2036) : 
  (a = 1011 ∧ b = 1012 ∧ c = 1013 ∧ d = 1015) := 
sorry

end find_four_numbers_l74_74191


namespace solve_equation_l74_74324

theorem solve_equation (x : ℝ) (h : (x - 1) / 2 = 1 - (x + 2) / 3) : x = 1 :=
sorry

end solve_equation_l74_74324


namespace pencils_ratio_l74_74397

theorem pencils_ratio (C J : ℕ) (hJ : J = 18) 
    (hJ_to_A : J_to_A = J / 3) (hJ_left : J_left = J - J_to_A)
    (hJ_left_eq : J_left = C + 3) :
    (C : ℚ) / (J : ℚ) = 1 / 2 :=
by
  sorry

end pencils_ratio_l74_74397


namespace trigonometric_identity_l74_74212

-- Define the problem conditions and formulas
variables (α : Real) (h : Real.cos (Real.pi / 6 + α) = Real.sqrt 3 / 3)

-- State the theorem
theorem trigonometric_identity : Real.cos (5 * Real.pi / 6 - α) = - (Real.sqrt 3 / 3) :=
by
  -- Placeholder for the proof
  sorry

end trigonometric_identity_l74_74212


namespace divisible_by_two_of_square_l74_74381

theorem divisible_by_two_of_square {a : ℤ} (h : 2 ∣ a^2) : 2 ∣ a :=
sorry

end divisible_by_two_of_square_l74_74381


namespace find_d_l74_74343

noncomputable def polynomial_d (a b c d : ℤ) (p q r s : ℤ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧
  1 + a + b + c + d = 2024 ∧
  (1 + p) * (1 + q) * (1 + r) * (1 + s) = 2024 ∧
  d = p * q * r * s

theorem find_d (a b c d : ℤ) (h : polynomial_d a b c d 7 10 22 11) : d = 17020 :=
  sorry

end find_d_l74_74343


namespace outfits_count_l74_74793

def num_outfits (n : Nat) (total_colors : Nat) : Nat :=
  let total_combinations := n * n * n
  let undesirable_combinations := total_colors
  total_combinations - undesirable_combinations

theorem outfits_count : num_outfits 5 5 = 120 :=
  by
  sorry

end outfits_count_l74_74793


namespace equation_of_perpendicular_line_through_point_l74_74930

theorem equation_of_perpendicular_line_through_point :
  ∃ (a : ℝ) (b : ℝ) (c : ℝ), (a = 3) ∧ (b = 1) ∧ (x - 2 * y - 3 = 0 → y = (-(1/2)) * x + 3/2) ∧ (2 * a + b - 7 = 0) := sorry

end equation_of_perpendicular_line_through_point_l74_74930


namespace average_value_continuous_l74_74092

noncomputable def average_value (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (1 / (b - a)) * ∫ x in a..b, f x

theorem average_value_continuous (f : ℝ → ℝ) (a b : ℝ) (h : ContinuousOn f (Set.Icc a b)) :
  (average_value f a b) = (1 / (b - a)) * (∫ x in a..b, f x) :=
by
  sorry

end average_value_continuous_l74_74092


namespace find_B_sin_squared_sum_range_l74_74155

-- Define the angles and vectors
variables {A B C : ℝ}
variables (m n : ℝ × ℝ)
variables (α : ℝ)

-- Basic triangle angle sum condition
axiom angle_sum : A + B + C = Real.pi

-- Define vectors as per the problem statement
axiom vector_m : m = (Real.sin B, 1 - Real.cos B)
axiom vector_n : n = (2, 0)

-- The angle between vectors m and n is π/3
axiom angle_between_vectors : α = Real.pi / 3
axiom angle_condition : Real.cos α = (2 * Real.sin B + 0 * (1 - Real.cos B)) / 
                                     (Real.sqrt (Real.sin B ^ 2 + (1 - Real.cos B) ^ 2) * 2)

theorem find_B : B = 2 * Real.pi / 3 := 
sorry

-- Conditions for range of sin^2 A + sin^2 C
axiom range_condition : (0 < A ∧ A < Real.pi / 3) 
                     ∧ (0 < C ∧ C < Real.pi / 3)
                     ∧ (A + C = Real.pi / 3)

theorem sin_squared_sum_range : (Real.sin A) ^ 2 + (Real.sin C) ^ 2 ∈ Set.Ico (1 / 2) 1 := 
sorry

end find_B_sin_squared_sum_range_l74_74155


namespace buddy_cards_on_thursday_is_32_l74_74760

def buddy_cards_on_monday := 30
def buddy_cards_on_tuesday := buddy_cards_on_monday / 2
def buddy_cards_on_wednesday := buddy_cards_on_tuesday + 12
def buddy_cards_bought_on_thursday := buddy_cards_on_tuesday / 3
def buddy_cards_on_thursday := buddy_cards_on_wednesday + buddy_cards_bought_on_thursday

theorem buddy_cards_on_thursday_is_32 : buddy_cards_on_thursday = 32 :=
by sorry

end buddy_cards_on_thursday_is_32_l74_74760


namespace roots_of_polynomial_l74_74915

theorem roots_of_polynomial :
  (∃ (r : List ℤ), r = [1, 3, 4] ∧ 
    (∀ x : ℤ, x ∈ r → x^3 - 8*x^2 + 19*x - 12 = 0)) ∧ 
  (∀ x, x^3 - 8*x^2 + 19*x - 12 = 0 → x ∈ [1, 3, 4]) := 
sorry

end roots_of_polynomial_l74_74915


namespace function_neither_even_nor_odd_l74_74785

noncomputable def f (x : ℝ) : ℝ := (4 * x ^ 3 - 3) / (x ^ 6 + 2)

theorem function_neither_even_nor_odd : 
  (∀ x : ℝ, f (-x) ≠ f x) ∧ (∀ x : ℝ, f (-x) ≠ -f x) :=
by
  sorry

end function_neither_even_nor_odd_l74_74785


namespace nature_of_roots_of_quadratic_l74_74849

theorem nature_of_roots_of_quadratic (k : ℝ) (h1 : k > 0) (h2 : 3 * k^2 - 2 = 10) :
  let a := 1
  let b := -(4 * k - 3)
  let c := 3 * k^2 - 2
  let Δ := b^2 - 4 * a * c
  Δ < 0 :=
by
  sorry

end nature_of_roots_of_quadratic_l74_74849


namespace intersection_complement_l74_74137

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

theorem intersection_complement :
  A ∩ (U \ B) = {2} :=
by {
  sorry
}

end intersection_complement_l74_74137


namespace find_m_plus_c_l74_74676

-- We need to define the conditions first
variable {A : ℝ × ℝ} {B : ℝ × ℝ} {c : ℝ} {m : ℝ}

-- Given conditions from part a)
def A_def : Prop := A = (1, 3)
def B_def : Prop := B = (m, -1)
def centers_line : Prop := ∀ C : ℝ × ℝ, (C.1 - C.2 + c = 0)

-- Define the theorem for the proof problem
theorem find_m_plus_c (A_def : A = (1, 3)) (B_def : B = (m, -1)) (centers_line : ∀ C : ℝ × ℝ, (C.1 - C.2 + c = 0)) : m + c = 3 :=
sorry

end find_m_plus_c_l74_74676


namespace students_behind_yoongi_l74_74385

theorem students_behind_yoongi (total_students jungkoo_position students_between_jungkook_yoongi : ℕ) 
    (h1 : total_students = 20)
    (h2 : jungkoo_position = 3)
    (h3 : students_between_jungkook_yoongi = 5) : 
    (total_students - (jungkoo_position + students_between_jungkook_yoongi + 1)) = 11 :=
by
  sorry

end students_behind_yoongi_l74_74385


namespace rect_plot_length_more_than_breadth_l74_74937

theorem rect_plot_length_more_than_breadth (b x : ℕ) (cost_per_m : ℚ)
  (length_eq : b + x = 56)
  (fencing_cost : (4 * b + 2 * x) * cost_per_m = 5300)
  (cost_rate : cost_per_m = 26.50) : x = 12 :=
by
  sorry

end rect_plot_length_more_than_breadth_l74_74937


namespace sum_a_b_l74_74086

theorem sum_a_b (a b : ℚ) (h1 : 3 * a + 7 * b = 12) (h2 : 9 * a + 2 * b = 23) : a + b = 176 / 57 :=
by
  sorry

end sum_a_b_l74_74086


namespace moon_iron_percentage_l74_74921

variables (x : ℝ) -- percentage of iron in the moon

-- Given conditions
def carbon_percentage_of_moon : ℝ := 0.20
def mass_of_moon : ℝ := 250
def mass_of_mars : ℝ := 2 * mass_of_moon
def mass_of_other_elements_on_mars : ℝ := 150
def composition_same (m : ℝ) (x : ℝ) := 
  (x / 100 * m + carbon_percentage_of_moon * m + (100 - x - 20) / 100 * m) = m

-- Theorem statement
theorem moon_iron_percentage : x = 50 :=
by
  sorry

end moon_iron_percentage_l74_74921


namespace units_digit_3_pow_2005_l74_74856

theorem units_digit_3_pow_2005 : 
  let units_digit (n : ℕ) : ℕ := n % 10
  units_digit (3^2005) = 3 :=
by
  sorry

end units_digit_3_pow_2005_l74_74856


namespace sum_difference_20_l74_74489

def sum_of_even_integers (n : ℕ) : ℕ := (n / 2) * (2 + 2 * (n - 1))

def sum_of_odd_integers (n : ℕ) : ℕ := (n / 2) * (1 + 2 * (n - 1))

theorem sum_difference_20 : sum_of_even_integers (20) - sum_of_odd_integers (20) = 20 := by
  sorry

end sum_difference_20_l74_74489


namespace area_within_fence_l74_74607

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end area_within_fence_l74_74607


namespace coin_problem_l74_74026

theorem coin_problem :
  ∃ n : ℕ, (n % 8 = 5) ∧ (n % 7 = 2) ∧ (n % 9 = 1) := 
sorry

end coin_problem_l74_74026


namespace find_fraction_l74_74196

theorem find_fraction
  (F : ℚ) (m : ℕ) 
  (h1 : F^m * (1 / 4)^2 = 1 / 10^4)
  (h2 : m = 4) : 
  F = 1 / 5 :=
by
  sorry

end find_fraction_l74_74196


namespace problem1_problem2_l74_74318

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x * f a x - x) / Real.exp x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 1 - (f a x - 1) / Real.exp x

theorem problem1 (x : ℝ) (h₁ : x ≥ 5) : g 1 x < 1 :=
sorry

theorem problem2 (a : ℝ) (h₂ : a > Real.exp 2 / 4) : 
∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 ∧ h a x1 = 0 ∧ h a x2 = 0 :=
sorry

end problem1_problem2_l74_74318


namespace remainder_zero_l74_74282

theorem remainder_zero (x : ℤ) :
  (x^5 - 1) * (x^3 - 1) % (x^2 + x + 1) = 0 := by
sorry

end remainder_zero_l74_74282


namespace opposite_numbers_A_l74_74484

theorem opposite_numbers_A :
  let A1 := -((-1)^2)
  let A2 := abs (-1)

  let B1 := (-2)^3
  let B2 := -(2^3)
  
  let C1 := 2
  let C2 := 1 / 2
  
  let D1 := -(-1)
  let D2 := 1
  
  (A1 = -A2 ∧ A2 = 1) ∧ ¬(B1 = -B2) ∧ ¬(C1 = -C2) ∧ ¬(D1 = -D2)
:= by
  let A1 := -((-1)^2)
  let A2 := abs (-1)

  let B1 := (-2)^3
  let B2 := -(2^3)
  
  let C1 := 2
  let C2 := 1 / 2
  
  let D1 := -(-1)
  let D2 := 1

  sorry

end opposite_numbers_A_l74_74484


namespace rita_hours_per_month_l74_74936

theorem rita_hours_per_month :
  let t := 1500
  let h_backstroke := 50
  let h_breaststroke := 9
  let h_butterfly := 121
  let m := 6
  let h_completed := h_backstroke + h_breaststroke + h_butterfly
  let h_remaining := t - h_completed
  let h := h_remaining / m
  h = 220
:= by 
  let t := 1500
  let h_backstroke := 50
  let h_breaststroke := 9
  let h_butterfly := 121
  let m := 6
  let h_completed := h_backstroke + h_breaststroke + h_butterfly
  have h_remaining := t - h_completed
  have h := h_remaining / m
  sorry

end rita_hours_per_month_l74_74936


namespace find_N_l74_74673

-- Definition of the conditions
def is_largest_divisor_smaller_than (m N : ℕ) : Prop := m < N ∧ Nat.gcd m N = m

def produces_power_of_ten (N m : ℕ) : Prop := ∃ k : ℕ, k > 0 ∧ N + m = 10^k

-- Final statement to prove
theorem find_N (N : ℕ) : (∃ m : ℕ, is_largest_divisor_smaller_than m N ∧ produces_power_of_ten N m) → N = 75 :=
by
  sorry

end find_N_l74_74673


namespace last_integer_in_sequence_l74_74548

theorem last_integer_in_sequence : ∀ (n : ℕ), n = 1000000 → (∀ k : ℕ, n = k * 3 → k * 3 < n) → n = 1000000 :=
by
  intro n hn hseq
  have h := hseq 333333 sorry
  exact hn

end last_integer_in_sequence_l74_74548


namespace possible_values_of_a_l74_74148

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a + b * Real.cos x + c * Real.sin x

theorem possible_values_of_a 
    (a b c : ℝ) 
    (h1 : f a b c (Real.pi / 2) = 1) 
    (h2 : f a b c Real.pi = 1) 
    (h3 : ∀ x : ℝ, |f a b c x| ≤ 2) :
    4 - 3 * Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2 :=
by
  sorry

end possible_values_of_a_l74_74148


namespace longest_badminton_match_duration_l74_74863

theorem longest_badminton_match_duration :
  let hours := 12
  let minutes := 25
  (hours * 60 + minutes = 745) :=
by
  sorry

end longest_badminton_match_duration_l74_74863


namespace melanie_total_payment_l74_74072

noncomputable def totalCost (rentalCostPerDay : ℝ) (insuranceCostPerDay : ℝ) (mileageCostPerMile : ℝ) (days : ℕ) (miles : ℕ) : ℝ :=
  (rentalCostPerDay * days) + (insuranceCostPerDay * days) + (mileageCostPerMile * miles)

theorem melanie_total_payment :
  totalCost 30 5 0.25 3 350 = 192.5 :=
by
  sorry

end melanie_total_payment_l74_74072


namespace find_abscissas_l74_74349

theorem find_abscissas (x_A x_B : ℝ) (y_A y_B : ℝ) : 
  ((y_A = x_A^2) ∧ (y_B = x_B^2) ∧ (0, 15) = (0,  (5 * y_B + 3 * y_A) / 8) ∧ (5 * x_B + 3 * x_A = 0)) → 
  ((x_A = -5 ∧ x_B = 3) ∨ (x_A = 5 ∧ x_B = -3)) :=
by
  sorry

end find_abscissas_l74_74349


namespace volume_of_soil_removal_l74_74774

theorem volume_of_soil_removal {a b m c d : ℝ} :
  (∃ (K : ℝ), K = (m / 6) * (2 * a * c + 2 * b * d + a * d + b * c)) :=
sorry

end volume_of_soil_removal_l74_74774


namespace count_integers_in_interval_l74_74261

theorem count_integers_in_interval : 
  ∃ (n : ℕ), (∀ (x : ℤ), (-2 ≤ x ∧ x ≤ 8 → ∃ (k : ℕ), k < n ∧ x = -2 + k)) ∧ n = 11 := 
by
  sorry

end count_integers_in_interval_l74_74261


namespace area_of_quadrilateral_l74_74927

def Quadrilateral (A B C D : Type) :=
  ∃ (ABC_deg : ℝ) (ADC_deg : ℝ) (AD : ℝ) (DC : ℝ) (AB : ℝ) (BC : ℝ),
  (ABC_deg = 90) ∧ (ADC_deg = 90) ∧ (AD = DC) ∧ (AB + BC = 20)

theorem area_of_quadrilateral (A B C D : Type) (h : Quadrilateral A B C D) : 
  ∃ (area : ℝ), area = 100 := 
sorry

end area_of_quadrilateral_l74_74927


namespace number_of_days_woman_weaves_l74_74795

theorem number_of_days_woman_weaves
  (a_1 : ℝ) (a_n : ℝ) (S_n : ℝ) (n : ℝ)
  (h1 : a_1 = 5)
  (h2 : a_n = 1)
  (h3 : S_n = 90)
  (h4 : S_n = n * (a_1 + a_n) / 2) :
  n = 30 :=
by
  rw [h1, h2, h3] at h4
  sorry

end number_of_days_woman_weaves_l74_74795


namespace find_x_for_fx_neg_half_l74_74837

open Function 

theorem find_x_for_fx_neg_half (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 2) = -f x)
  (h_interval : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 1/2 * x) :
  {x : ℝ | f x = -1/2} = {x : ℝ | ∃ n : ℤ, x = 4 * n - 1} :=
by
  sorry

end find_x_for_fx_neg_half_l74_74837


namespace problem_statement_l74_74796

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2 - 2 * a * x

theorem problem_statement (a : ℝ) (x1 x2 : ℝ) (h_a : a > 1) (h1 : x1 < x2) (h_extreme : f a x1 = 0 ∧ f a x2 = 0) : 
  f a x2 < -3/2 :=
sorry

end problem_statement_l74_74796


namespace trains_cross_time_l74_74186

noncomputable def time_to_cross_trains : ℝ :=
  let l1 := 220 -- length of the first train in meters
  let s1 := 120 * (5 / 18) -- speed of the first train in meters per second
  let l2 := 280.04 -- length of the second train in meters
  let s2 := 80 * (5 / 18) -- speed of the second train in meters per second
  let relative_speed := s1 + s2 -- relative speed in meters per second
  let total_length := l1 + l2 -- total length to be crossed in meters
  total_length / relative_speed -- time in seconds

theorem trains_cross_time :
  abs (time_to_cross_trains - 9) < 0.01 := -- Allowing a small error to account for approximation
by
  sorry

end trains_cross_time_l74_74186


namespace sum_first_last_l74_74314

theorem sum_first_last (A B C D : ℕ) (h1 : (A + B + C) / 3 = 6) (h2 : (B + C + D) / 3 = 5) (h3 : D = 4) : A + D = 11 :=
by
  sorry

end sum_first_last_l74_74314


namespace symmetric_point_correct_l74_74522

-- Define the point P in a three-dimensional Cartesian coordinate system.
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the function to find the symmetric point with respect to the x-axis.
def symmetricWithRespectToXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Given point P(1, -2, 3).
def P : Point3D := { x := 1, y := -2, z := 3 }

-- The expected symmetric point
def symmetricP : Point3D := { x := 1, y := 2, z := -3 }

-- The proposition we need to prove
theorem symmetric_point_correct :
  symmetricWithRespectToXAxis P = symmetricP :=
by
  sorry

end symmetric_point_correct_l74_74522


namespace coprime_integers_lt_15_l74_74055

theorem coprime_integers_lt_15 : ∃ (S : Finset ℕ), S.card = 8 ∧ (∀ a ∈ S, a < 15 ∧ Nat.gcd a 15 = 1) :=
by
  sorry

end coprime_integers_lt_15_l74_74055


namespace profit_ratio_l74_74699

theorem profit_ratio (I_P I_Q : ℝ) (t_P t_Q : ℕ) 
  (h1 : I_P / I_Q = 7 / 5)
  (h2 : t_P = 5)
  (h3 : t_Q = 14) : 
  (I_P * t_P) / (I_Q * t_Q) = 1 / 2 :=
by
  sorry

end profit_ratio_l74_74699


namespace total_workers_calculation_l74_74227

theorem total_workers_calculation :
  ∀ (N : ℕ), 
  (∀ (total_avg_salary : ℕ) (techs_salary : ℕ) (nontech_avg_salary : ℕ),
    total_avg_salary = 8000 → 
    techs_salary = 7 * 20000 → 
    nontech_avg_salary = 6000 →
    8000 * (7 + N) = 7 * 20000 + N * 6000 →
    (7 + N) = 49) :=
by
  intros
  sorry

end total_workers_calculation_l74_74227


namespace negative_solution_exists_l74_74049

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l74_74049


namespace solve_inequality_system_l74_74005

-- Define the conditions and the correct answer
def system_of_inequalities (x : ℝ) : Prop :=
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x)

def solution_set (x : ℝ) : Prop :=
  2 < x ∧ x ≤ 4

-- State that solving the system of inequalities is equivalent to the solution set
theorem solve_inequality_system (x : ℝ) : system_of_inequalities x ↔ solution_set x :=
  sorry

end solve_inequality_system_l74_74005


namespace remainder_when_4x_div_7_l74_74950

theorem remainder_when_4x_div_7 (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 :=
by
  sorry

end remainder_when_4x_div_7_l74_74950


namespace score_in_first_round_l74_74962

theorem score_in_first_round (cards : List ℕ) (scores : List ℕ) 
  (total_rounds : ℕ) (last_round_score : ℕ) (total_score : ℕ) : 
  cards = [2, 4, 7, 13] ∧ scores = [16, 17, 21, 24] ∧ total_rounds = 3 ∧ last_round_score = 2 ∧ total_score = 16 →
  ∃ first_round_score, first_round_score = 7 := by
  sorry

end score_in_first_round_l74_74962


namespace xyz_value_l74_74958

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (xy + xz + yz) = 40) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) 
  : x * y * z = 10 :=
sorry

end xyz_value_l74_74958


namespace abc_value_l74_74929

theorem abc_value {a b c : ℂ} 
  (h1 : a * b + 5 * b + 20 = 0) 
  (h2 : b * c + 5 * c + 20 = 0) 
  (h3 : c * a + 5 * a + 20 = 0) : 
  a * b * c = 100 := 
by 
  sorry

end abc_value_l74_74929


namespace intersection_of_M_and_N_l74_74922

-- Define sets M and N as given in the conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- The theorem statement to prove the intersection of M and N is {2, 3}
theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by sorry  -- The proof is skipped with 'sorry'

end intersection_of_M_and_N_l74_74922


namespace Margie_distance_on_25_dollars_l74_74359

theorem Margie_distance_on_25_dollars
  (miles_per_gallon : ℝ)
  (cost_per_gallon : ℝ)
  (amount_spent : ℝ) :
  miles_per_gallon = 40 →
  cost_per_gallon = 5 →
  amount_spent = 25 →
  (amount_spent / cost_per_gallon) * miles_per_gallon = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end Margie_distance_on_25_dollars_l74_74359


namespace candy_mixture_solution_l74_74797

theorem candy_mixture_solution :
  ∃ x y : ℝ, 18 * x + 10 * y = 1500 ∧ x + y = 100 ∧ x = 62.5 ∧ y = 37.5 := by
  sorry

end candy_mixture_solution_l74_74797


namespace standard_equation_of_ellipse_l74_74727

theorem standard_equation_of_ellipse
  (a b c : ℝ)
  (h_major_minor : 2 * a = 6 * b)
  (h_focal_distance : 2 * c = 8)
  (h_ellipse_relation : a^2 = b^2 + c^2) :
  (∀ x y : ℝ, (x^2 / 18 + y^2 / 2 = 1) ∨ (y^2 / 18 + x^2 / 2 = 1)) :=
by {
  sorry
}

end standard_equation_of_ellipse_l74_74727


namespace find_b_l74_74599

variable (a b c : ℕ)
variable (h1 : (a + b + c) / 3 = 45)
variable (h2 : (a + b) / 2 = 40)
variable (h3 : (b + c) / 2 = 43)

theorem find_b : b = 31 := sorry

end find_b_l74_74599


namespace tiles_required_for_floor_l74_74775

def tileDimensionsInFeet (width_in_inches : ℚ) (length_in_inches : ℚ) : ℚ × ℚ :=
  (width_in_inches / 12, length_in_inches / 12)

def area (length : ℚ) (width : ℚ) : ℚ :=
  length * width

noncomputable def numberOfTiles (floor_length : ℚ) (floor_width : ℚ) (tile_length : ℚ) (tile_width : ℚ) : ℚ :=
  (area floor_length floor_width) / (area tile_length tile_width)

theorem tiles_required_for_floor : numberOfTiles 10 15 (5/12) (2/3) = 540 := by
  sorry

end tiles_required_for_floor_l74_74775


namespace sqrt_112_consecutive_integers_product_l74_74232

theorem sqrt_112_consecutive_integers_product : 
  (∃ (a b : ℕ), a * a < 112 ∧ 112 < b * b ∧ b = a + 1 ∧ a * b = 110) :=
by 
  use 10, 11
  repeat { sorry }

end sqrt_112_consecutive_integers_product_l74_74232


namespace length_of_train_is_125_l74_74506

noncomputable def speed_kmph : ℝ := 90
noncomputable def time_sec : ℝ := 5
noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)
noncomputable def length_train : ℝ := speed_mps * time_sec

theorem length_of_train_is_125 :
  length_train = 125 := 
by
  sorry

end length_of_train_is_125_l74_74506


namespace boxes_of_bolts_purchased_l74_74358

theorem boxes_of_bolts_purchased 
  (bolts_per_box : ℕ) 
  (nuts_per_box : ℕ) 
  (num_nut_boxes : ℕ) 
  (leftover_bolts : ℕ) 
  (leftover_nuts : ℕ) 
  (total_bolts_nuts_used : ℕ)
  (B : ℕ) :
  bolts_per_box = 11 →
  nuts_per_box = 15 →
  num_nut_boxes = 3 →
  leftover_bolts = 3 →
  leftover_nuts = 6 →
  total_bolts_nuts_used = 113 →
  B = 7 :=
by
  intros
  sorry

end boxes_of_bolts_purchased_l74_74358


namespace problem_I_problem_II_problem_III_l74_74139

-- Problem (I)
noncomputable def f (x a : ℝ) := Real.log x - a * (x - 1)
noncomputable def tangent_line (x a : ℝ) := (1 - a) * (x - 1)

theorem problem_I (a : ℝ) :
  ∃ y, tangent_line y a = f 1 a / (1 : ℝ) :=
sorry

-- Problem (II)
theorem problem_II (a : ℝ) (h : a ≥ 1 / 2) :
  ∀ x ≥ 1, f x a ≤ Real.log x / (x + 1) :=
sorry

-- Problem (III)
theorem problem_III (a : ℝ) :
  ∀ x ≥ 1, Real.exp (x - 1) - a * (x ^ 2 - x) ≥ x * f x a + 1 :=
sorry

end problem_I_problem_II_problem_III_l74_74139


namespace quadratic_roots_property_l74_74351

theorem quadratic_roots_property (a b : ℝ)
  (h1 : a^2 - 2 * a - 1 = 0)
  (h2 : b^2 - 2 * b - 1 = 0)
  (ha_b_sum : a + b = 2)
  (ha_b_product : a * b = -1) :
  a^2 + 2 * b - a * b = 6 :=
sorry

end quadratic_roots_property_l74_74351


namespace minimum_soldiers_to_add_l74_74780

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l74_74780


namespace rectangle_area_problem_l74_74470

theorem rectangle_area_problem (l w l1 l2 w1 w2 : ℝ) (h1 : l = l1 + l2) (h2 : w = w1 + w2) 
  (h3 : l1 * w1 = 12) (h4 : l2 * w1 = 15) (h5 : l1 * w2 = 12) 
  (h6 : l2 * w2 = 8) (h7 : w1 * l2 = 18) (h8 : l1 * w2 = 20) :
  l2 * w1 = 18 :=
sorry

end rectangle_area_problem_l74_74470


namespace harry_walks_9_dogs_on_thursday_l74_74816

-- Define the number of dogs Harry walks on specific days
def dogs_monday : Nat := 7
def dogs_wednesday : Nat := 7
def dogs_friday : Nat := 7
def dogs_tuesday : Nat := 12

-- Define the payment per dog
def payment_per_dog : Nat := 5

-- Define total weekly earnings
def total_weekly_earnings : Nat := 210

-- Define the number of dogs Harry walks on Thursday
def dogs_thursday : Nat := 9

-- Define the total earnings for Monday, Wednesday, Friday, and Tuesday
def earnings_first_four_days : Nat := (dogs_monday + dogs_wednesday + dogs_friday + dogs_tuesday) * payment_per_dog

-- Now we state the theorem that we need to prove
theorem harry_walks_9_dogs_on_thursday :
  (total_weekly_earnings - earnings_first_four_days) / payment_per_dog = dogs_thursday :=
by
  -- Proof omitted
  sorry

end harry_walks_9_dogs_on_thursday_l74_74816


namespace customers_at_start_l74_74734

def initial_customers (X : ℕ) : Prop :=
  let first_hour := X + 3
  let second_hour := first_hour - 6
  second_hour = 12

theorem customers_at_start {X : ℕ} : initial_customers X → X = 15 :=
by
  sorry

end customers_at_start_l74_74734


namespace geometric_seq_problem_l74_74971

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r

theorem geometric_seq_problem (h_geom : geometric_sequence a) 
  (h_cond : a 8 * a 9 * a 10 = -a 13 ^ 2 ∧ -a 13 ^ 2 = -1000) :
  a 10 * a 12 = 100 * Real.sqrt 10 :=
by
  sorry

end geometric_seq_problem_l74_74971


namespace area_difference_zero_l74_74335

theorem area_difference_zero
  (AG CE : ℝ)
  (s : ℝ)
  (area_square area_rectangle : ℝ)
  (h1 : AG = 2)
  (h2 : CE = 2)
  (h3 : s = 2)
  (h4 : area_square = s^2)
  (h5 : area_rectangle = 2 * 2) :
  (area_square - area_rectangle = 0) :=
by sorry

end area_difference_zero_l74_74335


namespace books_per_week_l74_74453

-- Define the conditions
def total_books_read : ℕ := 20
def weeks : ℕ := 5

-- Define the statement to be proved
theorem books_per_week : (total_books_read / weeks) = 4 := by
  -- Proof omitted
  sorry

end books_per_week_l74_74453


namespace arithmetic_sequence_solution_geometric_sequence_solution_l74_74626

-- Problem 1: Arithmetic sequence
noncomputable def arithmetic_general_term (n : ℕ) : ℕ := 30 - 3 * n
noncomputable def arithmetic_sum_terms (n : ℕ) : ℝ := -1.5 * n^2 + 28.5 * n

theorem arithmetic_sequence_solution (n : ℕ) (a8 a10 : ℕ) (sequence : ℕ → ℝ) :
  a8 = 6 → a10 = 0 → (sequence n = arithmetic_general_term n) ∧ (sequence n = arithmetic_sum_terms n) ∧ (n = 9 ∨ n = 10) := 
sorry

-- Problem 2: Geometric sequence
noncomputable def geometric_general_term (n : ℕ) : ℝ := 2^(n-2)
noncomputable def geometric_sum_terms (n : ℕ) : ℝ := 2^(n-1) - 0.5

theorem geometric_sequence_solution (n : ℕ) (a1 a4 : ℝ) (sequence : ℕ → ℝ):
  a1 = 0.5 → a4 = 4 → (sequence n = geometric_general_term n) ∧ (sequence n = geometric_sum_terms n) := 
sorry

end arithmetic_sequence_solution_geometric_sequence_solution_l74_74626


namespace trajectory_equation_l74_74623

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem trajectory_equation (P : ℝ × ℝ) (h : |distance P (1, 0) - P.1| = 1) :
  (P.1 ≥ 0 → P.2 ^ 2 = 4 * P.1) ∧ (P.1 < 0 → P.2 = 0) :=
by
  sorry

end trajectory_equation_l74_74623


namespace oranges_equivalency_l74_74495

theorem oranges_equivalency :
  ∀ (w_orange w_apple w_pear : ℕ), 
  (9 * w_orange = 6 * w_apple + w_pear) →
  (36 * w_orange = 24 * w_apple + 4 * w_pear) :=
by
  -- The proof will go here; for now, we'll use sorry to skip it
  sorry

end oranges_equivalency_l74_74495


namespace inequality_correct_l74_74195

theorem inequality_correct (a b : ℝ) (h : a - |b| > 0) : a + b > 0 :=
sorry

end inequality_correct_l74_74195


namespace max_product_condition_l74_74280

theorem max_product_condition (x y : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 12) (h3 : 0 ≤ y) (h4 : y ≤ 12) (h_eq : x * y = (12 - x) ^ 2 * (12 - y) ^ 2) : x * y ≤ 81 :=
sorry

end max_product_condition_l74_74280


namespace area_of_triangle_ABC_l74_74365

def Point : Type := (ℝ × ℝ)

def A : Point := (0, 0)
def B : Point := (2, 2)
def C : Point := (2, 0)

def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem area_of_triangle_ABC :
  triangle_area A B C = 2 :=
by
  sorry

end area_of_triangle_ABC_l74_74365


namespace amount_each_girl_receives_l74_74663

theorem amount_each_girl_receives (total_amount : ℕ) (total_children : ℕ) (amount_per_boy : ℕ) (num_boys : ℕ) (remaining_amount : ℕ) (num_girls : ℕ) (amount_per_girl : ℕ) 
  (h1 : total_amount = 460) 
  (h2 : total_children = 41)
  (h3 : amount_per_boy = 12)
  (h4 : num_boys = 33)
  (h5 : remaining_amount = total_amount - num_boys * amount_per_boy)
  (h6 : num_girls = total_children - num_boys)
  (h7 : amount_per_girl = remaining_amount / num_girls) :
  amount_per_girl = 8 := 
sorry

end amount_each_girl_receives_l74_74663


namespace tv_cost_l74_74158

-- Definitions from the problem conditions
def fraction_on_furniture : ℚ := 3 / 4
def total_savings : ℚ := 1800
def fraction_on_tv : ℚ := 1 - fraction_on_furniture  -- Fraction of savings on TV

-- The proof problem statement
theorem tv_cost : total_savings * fraction_on_tv = 450 := by
  sorry

end tv_cost_l74_74158


namespace inverse_of_problem_matrix_is_zero_matrix_l74_74711

def det (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

def zero_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 0], ![0, 0]]

noncomputable def problem_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, -6], ![-2, 3]]

theorem inverse_of_problem_matrix_is_zero_matrix :
  det problem_matrix = 0 → problem_matrix⁻¹ = zero_matrix :=
by
  intro h
  -- Proof steps will be written here
  sorry

end inverse_of_problem_matrix_is_zero_matrix_l74_74711


namespace seventh_term_value_l74_74714

theorem seventh_term_value (a d : ℤ) (h1 : a = 12) (h2 : a + 3 * d = 18) : a + 6 * d = 24 := 
by
  sorry

end seventh_term_value_l74_74714


namespace find_a_l74_74315

theorem find_a (a : ℝ) (p : ℕ → ℝ) (h : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 → p k = a * (1 / 2) ^ k)
  (prob_sum : a * (1 / 2 + (1 / 2) ^ 2 + (1 / 2) ^ 3) = 1) : a = 8 / 7 :=
sorry

end find_a_l74_74315


namespace find_k_l74_74382

noncomputable def curve_C (x y : ℝ) : Prop :=
  x^2 + (y^2 / 4) = 1

noncomputable def line_eq (k x y : ℝ) : Prop :=
  y = k * x + 1

theorem find_k (k : ℝ) :
  (∃ A B : ℝ × ℝ, (curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧ line_eq k A.1 A.2 ∧ line_eq k B.1 B.2 ∧ 
   (A.1 * B.1 + A.2 * B.2 = 0))) ↔ (k = 1/2 ∨ k = -1/2) :=
sorry

end find_k_l74_74382


namespace maximum_negative_roots_l74_74104

theorem maximum_negative_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
    (discriminant1 : b^2 - 4 * a * c ≥ 0)
    (discriminant2 : c^2 - 4 * b * a ≥ 0)
    (discriminant3 : a^2 - 4 * c * b ≥ 0) :
    ∃ n : ℕ, n ≤ 2 ∧ ∀ x ∈ {x | a * x^2 + b * x + c = 0 ∨ b * x^2 + c * x + a = 0 ∨ c * x^2 + a * x + b = 0}, x < 0 ↔ n = 2 := 
sorry

end maximum_negative_roots_l74_74104


namespace elem_of_M_l74_74017

variable (U M : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : U \ M = {1, 3})

theorem elem_of_M : 2 ∈ M :=
by {
  sorry
}

end elem_of_M_l74_74017


namespace other_diagonal_length_l74_74587

theorem other_diagonal_length (d2 : ℝ) (A : ℝ) (d1 : ℝ) 
  (h1 : d2 = 120) 
  (h2 : A = 4800) 
  (h3 : A = (d1 * d2) / 2) : d1 = 80 :=
by
  sorry

end other_diagonal_length_l74_74587


namespace fingers_game_conditions_l74_74914

noncomputable def minNForWinningSubset (N : ℕ) : Prop :=
  N ≥ 220

-- To state the probability condition, we need to express it in terms of actual probabilities
noncomputable def probLeaderWins (N : ℕ) : ℝ := 
  1 / N

noncomputable def leaderWinProbabilityTendsToZero : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, probLeaderWins n < ε

theorem fingers_game_conditions (N : ℕ) (probLeaderWins : ℕ → ℝ) :
  (minNForWinningSubset N) ∧ leaderWinProbabilityTendsToZero :=
by
  sorry

end fingers_game_conditions_l74_74914


namespace marbles_per_boy_l74_74881

theorem marbles_per_boy (boys marbles : ℕ) (h1 : boys = 5) (h2 : marbles = 35) : marbles / boys = 7 := by
  sorry

end marbles_per_boy_l74_74881


namespace root_of_f_l74_74866

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

theorem root_of_f (h_inv : f_inv 0 = 2) (h_interval : 1 ≤ (f_inv 0) ∧ (f_inv 0) ≤ 4) : f 2 = 0 := 
sorry

end root_of_f_l74_74866


namespace find_smallest_natural_number_l74_74701

theorem find_smallest_natural_number :
  ∃ x : ℕ, (2 * x = b^2 ∧ 3 * x = c^3) ∧ (∀ y : ℕ, (2 * y = d^2 ∧ 3 * y = e^3) → x ≤ y) := by
  sorry

end find_smallest_natural_number_l74_74701


namespace age_difference_l74_74101

theorem age_difference (A B : ℕ) (h1 : B = 38) (h2 : A + 10 = 2 * (B - 10)) : A - B = 8 :=
by
  sorry

end age_difference_l74_74101


namespace remainder_4_power_100_div_9_l74_74674

theorem remainder_4_power_100_div_9 : (4^100) % 9 = 4 :=
by
  sorry

end remainder_4_power_100_div_9_l74_74674


namespace ratio_is_one_to_two_l74_74685

def valentina_share_to_whole_ratio (valentina_share : ℕ) (whole_burger : ℕ) : ℕ × ℕ :=
  (valentina_share / (Nat.gcd valentina_share whole_burger), 
   whole_burger / (Nat.gcd valentina_share whole_burger))

theorem ratio_is_one_to_two : valentina_share_to_whole_ratio 6 12 = (1, 2) := 
  by
  sorry

end ratio_is_one_to_two_l74_74685


namespace form_square_from_trapezoid_l74_74021

noncomputable def trapezoid_area (a b h : ℝ) : ℝ :=
  (a + b) * h / 2

theorem form_square_from_trapezoid (a b h : ℝ) (trapezoid_area_eq_five : trapezoid_area a b h = 5) :
  ∃ s : ℝ, s^2 = 5 :=
by
  use (Real.sqrt 5)
  sorry

end form_square_from_trapezoid_l74_74021


namespace chose_number_l74_74275

theorem chose_number (x : ℝ) (h : (x / 12)^2 - 240 = 8) : x = 24 * Real.sqrt 62 :=
sorry

end chose_number_l74_74275


namespace event_B_is_certain_l74_74801

-- Define the event that the sum of two sides of a triangle is greater than the third side
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the term 'certain event'
def certain_event (E : Prop) : Prop := E

/-- Prove that the event "the sum of two sides of a triangle is greater than the third side" is a certain event -/
theorem event_B_is_certain (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  certain_event (triangle_inequality a b c) :=
sorry

end event_B_is_certain_l74_74801


namespace determine_K_class_comparison_l74_74153

variables (a b : ℕ) -- number of students in classes A and B respectively
variable (K : ℕ) -- amount that each A student would pay if they covered all cost

-- Conditions from the problem statement
def first_event_total (a b : ℕ) := 5 * a + 3 * b
def second_event_total (a b : ℕ) := 4 * a + 6 * b
def total_balance (a b K : ℕ) := 9 * (a + b) = K * (a + b)

-- Questions to be answered
theorem determine_K : total_balance a b K → K = 9 :=
by
  sorry

theorem class_comparison (a b : ℕ) : 5 * a + 3 * b = 4 * a + 6 * b → b > a :=
by
  sorry

end determine_K_class_comparison_l74_74153


namespace jessa_cupcakes_l74_74219

-- Define the number of classes and students
def fourth_grade_classes : ℕ := 3
def students_per_fourth_grade_class : ℕ := 30
def pe_classes : ℕ := 1
def students_per_pe_class : ℕ := 50

-- Calculate the total number of cupcakes needed
def total_cupcakes_needed : ℕ :=
  (fourth_grade_classes * students_per_fourth_grade_class) +
  (pe_classes * students_per_pe_class)

-- Statement to prove
theorem jessa_cupcakes : total_cupcakes_needed = 140 :=
by
  sorry

end jessa_cupcakes_l74_74219


namespace greatest_fourth_term_arith_seq_sum_90_l74_74651

theorem greatest_fourth_term_arith_seq_sum_90 :
  ∃ a d : ℕ, 6 * a + 15 * d = 90 ∧ (∀ n : ℕ, n < 6 → a + n * d > 0) ∧ (a + 3 * d = 17) :=
by
  sorry

end greatest_fourth_term_arith_seq_sum_90_l74_74651


namespace peanut_butter_candy_pieces_l74_74110

theorem peanut_butter_candy_pieces :
  ∀ (pb_candy grape_candy banana_candy : ℕ),
  pb_candy = 4 * grape_candy →
  grape_candy = banana_candy + 5 →
  banana_candy = 43 →
  pb_candy = 192 :=
by
  sorry

end peanut_butter_candy_pieces_l74_74110


namespace friends_who_dont_eat_meat_l74_74310

-- Definitions based on conditions
def number_of_friends : Nat := 10
def burgers_per_friend : Nat := 3
def buns_per_pack : Nat := 8
def packs_of_buns : Nat := 3
def friends_dont_eat_meat : Nat := 1
def friends_dont_eat_bread : Nat := 1

-- Total number of buns Alex plans to buy
def total_buns : Nat := buns_per_pack * packs_of_buns

-- Calculation of friends needing buns
def friends_needing_buns : Nat := number_of_friends - friends_dont_eat_meat - friends_dont_eat_bread

-- Total buns needed
def buns_needed : Nat := friends_needing_buns * burgers_per_friend

theorem friends_who_dont_eat_meat :
  buns_needed = total_buns -> friends_dont_eat_meat = 1 := by
  sorry

end friends_who_dont_eat_meat_l74_74310


namespace range_of_m_l74_74493

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem range_of_m (m : ℝ) (h : ∀ x > 0, f x > m * x) : m ≤ 2 := sorry

end range_of_m_l74_74493


namespace range_of_a_l74_74281

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 1 then a^x else (4 - a/2)*x + 2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, x1 < x2 → f a x1 ≤ f a x2) ↔ (4 ≤ a ∧ a < 8) :=
by
  sorry

end range_of_a_l74_74281


namespace triangle_equilateral_from_midpoint_circles_l74_74583

theorem triangle_equilateral_from_midpoint_circles (a b c : ℝ)
  (h1 : ∃ E F G : ℝ → ℝ, ∀ x, (|E x| = a/4 ∨ |F x| = b/4 ∨ |G x| = c/4))
  (h2 : (|a/2| ≤ a/4 + b/4) ∧ (|b/2| ≤ b/4 + c/4) ∧ (|c/2| ≤ c/4 + a/4)) :
  a = b ∧ b = c :=
sorry

end triangle_equilateral_from_midpoint_circles_l74_74583


namespace fraction_product_l74_74538

theorem fraction_product : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l74_74538


namespace marbles_before_purchase_l74_74063

-- Lean 4 statement for the problem
theorem marbles_before_purchase (bought : ℝ) (total_now : ℝ) (initial : ℝ) 
    (h1 : bought = 134.0) 
    (h2 : total_now = 321) 
    (h3 : total_now = initial + bought) : 
    initial = 187 :=
by 
    sorry

end marbles_before_purchase_l74_74063


namespace front_view_correct_l74_74443

-- Define the number of blocks in each column
def Blocks_Column_A : Nat := 3
def Blocks_Column_B : Nat := 5
def Blocks_Column_C : Nat := 2
def Blocks_Column_D : Nat := 4

-- Define the front view representation
def front_view : List Nat := [3, 5, 2, 4]

-- Statement to be proved
theorem front_view_correct :
  [Blocks_Column_A, Blocks_Column_B, Blocks_Column_C, Blocks_Column_D] = front_view :=
by
  sorry

end front_view_correct_l74_74443


namespace license_plates_count_l74_74326

def number_of_license_plates : ℕ :=
  let digit_choices := 10^5
  let letter_block_choices := 3 * 26^2
  let block_positions := 6
  digit_choices * letter_block_choices * block_positions

theorem license_plates_count : number_of_license_plates = 1216800000 := by
  -- proof steps here
  sorry

end license_plates_count_l74_74326


namespace anusha_share_l74_74552

theorem anusha_share (A B E D G X : ℝ) 
  (h1: 20 * A = X)
  (h2: 15 * B = X)
  (h3: 8 * E = X)
  (h4: 12 * D = X)
  (h5: 10 * G = X)
  (h6: A + B + E + D + G = 950) : 
  A = 112 := 
by 
  sorry

end anusha_share_l74_74552


namespace perfect_square_quotient_l74_74399

theorem perfect_square_quotient {a b : ℕ} (hpos: 0 < a ∧ 0 < b) (hdiv : (ab + 1) ∣ (a^2 + b^2)) : ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end perfect_square_quotient_l74_74399


namespace max_daily_sales_revenue_l74_74972

noncomputable def P (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 24 then t + 2 else if 25 ≤ t ∧ t ≤ 30 then 100 - t else 0

noncomputable def Q (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 30 then 40 - t else 0

noncomputable def y (t : ℕ) : ℕ :=
  P t * Q t

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 115 :=
sorry

end max_daily_sales_revenue_l74_74972


namespace last_number_written_on_sheet_l74_74142

/-- The given problem is to find the last number written on a sheet with specific rules. 
Given:
- The sheet has dimensions of 100 characters in width and 100 characters in height.
- Numbers are written successively with a space between each number.
- If the end of a line is reached, the next number continues at the beginning of the next line.

We need to prove that the last number written on the sheet is 2220.
-/
theorem last_number_written_on_sheet :
  ∃ (n : ℕ), n = 2220 ∧ 
    let width := 100
    let height := 100
    let sheet_size := width * height
    let write_number size occupied_space := occupied_space + size + 1 
    ∃ (numbers : ℕ → ℕ) (space_per_number : ℕ → ℕ),
      ( ∀ i, space_per_number i = if numbers i < 10 then 2 else if numbers i < 100 then 3 else if numbers i < 1000 then 4 else 5 ) ∧
      ∃ (current_space : ℕ), 
        (current_space ≤ sheet_size) ∧
        (∀ i, current_space = write_number (space_per_number i) current_space ) :=
sorry

end last_number_written_on_sheet_l74_74142


namespace vacationers_city_correctness_l74_74904

noncomputable def vacationer_cities : Prop :=
  ∃ (city : String → String),
    (city "Amelie" = "Acapulco" ∨ city "Amelie" = "Brest" ∨ city "Amelie" = "Madrid") ∧
    (city "Benoit" = "Acapulco" ∨ city "Benoit" = "Brest" ∨ city "Benoit" = "Madrid") ∧
    (city "Pierre" = "Paris" ∨ city "Pierre" = "Brest" ∨ city "Pierre" = "Madrid") ∧
    (city "Melanie" = "Acapulco" ∨ city "Melanie" = "Brest" ∨ city "Melanie" = "Madrid") ∧
    (city "Charles" = "Acapulco" ∨ city "Charles" = "Brest" ∨ city "Charles" = "Madrid") ∧
    -- Conditions stated by participants
    ((city "Amelie" = "Acapulco") ∨ (city "Amelie" ≠ "Acapulco" ∧ city "Benoit" = "Acapulco" ∧ city "Pierre" = "Paris")) ∧
    ((city "Benoit" = "Brest") ∨ (city "Benoit" ≠ "Brest" ∧ city "Charles" = "Brest" ∧ city "Pierre" = "Paris")) ∧
    ((city "Pierre" ≠ "France") ∨ (city "Pierre" = "Paris" ∧ city "Amelie" ≠ "France" ∧ city "Melanie" = "Madrid")) ∧
    ((city "Melanie" = "Clermont-Ferrand") ∨ (city "Melanie" ≠ "Clermont-Ferrand" ∧ city "Amelie" = "Acapulco" ∧ city "Pierre" = "Paris")) ∧
    ((city "Charles" = "Clermont-Ferrand") ∨ (city "Charles" ≠ "Clermont-Ferrand" ∧ city "Amelie" = "Acapulco" ∧ city "Benoit" = "Acapulco"))

theorem vacationers_city_correctness : vacationer_cities :=
  sorry

end vacationers_city_correctness_l74_74904


namespace jacob_younger_than_michael_l74_74606

-- Definitions based on the conditions.
def jacob_current_age : ℕ := 9
def michael_current_age : ℕ := 2 * (jacob_current_age + 3) - 3

-- Theorem to prove that Jacob is 12 years younger than Michael.
theorem jacob_younger_than_michael : michael_current_age - jacob_current_age = 12 :=
by
  -- Placeholder for proof
  sorry

end jacob_younger_than_michael_l74_74606


namespace inner_tetrahedron_volume_ratio_l74_74763

noncomputable def volume_ratio_of_tetrahedrons (s : ℝ) : ℝ :=
  let V_original := (s^3 * Real.sqrt 2) / 12
  let a := (Real.sqrt 6 / 9) * s
  let V_inner := (a^3 * Real.sqrt 2) / 12
  V_inner / V_original

theorem inner_tetrahedron_volume_ratio {s : ℝ} (hs : s > 0) : volume_ratio_of_tetrahedrons s = 1 / 243 :=
by
  sorry

end inner_tetrahedron_volume_ratio_l74_74763


namespace scale_length_l74_74409

theorem scale_length (length_of_part : ℕ) (number_of_parts : ℕ) (h1 : number_of_parts = 2) (h2 : length_of_part = 40) :
  number_of_parts * length_of_part = 80 := 
by
  sorry

end scale_length_l74_74409


namespace x_fifth_power_sum_l74_74259

theorem x_fifth_power_sum (x : ℝ) (h : x + 1 / x = -5) : x^5 + 1 / x^5 = -2525 := by
  sorry

end x_fifth_power_sum_l74_74259


namespace square_perimeter_l74_74974

theorem square_perimeter (s : ℕ) (h : 5 * s / 2 = 40) : 4 * s = 64 := by
  sorry

end square_perimeter_l74_74974


namespace imaginary_part_of_quotient_l74_74907

noncomputable def imaginary_part_of_complex (z : ℂ) : ℂ := z.im

theorem imaginary_part_of_quotient :
  imaginary_part_of_complex (i / (1 - i)) = 1 / 2 :=
by sorry

end imaginary_part_of_quotient_l74_74907


namespace eggs_needed_for_recipe_l74_74779

noncomputable section

theorem eggs_needed_for_recipe 
  (total_eggs : ℕ) 
  (rotten_eggs : ℕ) 
  (prob_all_rotten : ℝ)
  (h_total : total_eggs = 36)
  (h_rotten : rotten_eggs = 3)
  (h_prob : prob_all_rotten = 0.0047619047619047615) 
  : (2 : ℕ) = 2 :=
by
  sorry

end eggs_needed_for_recipe_l74_74779


namespace percentage_increase_l74_74543

theorem percentage_increase (x : ℝ) (h : 2 * x = 540) (new_price : ℝ) (h_new_price : new_price = 351) :
  ((new_price - x) / x) * 100 = 30 := by
  sorry

end percentage_increase_l74_74543


namespace min_value_fraction_l74_74221

theorem min_value_fraction (a b : ℝ) (h : x^2 - 3*x + a*b < 0 ∧ 1 < x ∧ x < 2) (h1 : a > b) : 
  (∃ minValue : ℝ, minValue = 4 ∧ ∀ a b : ℝ, a > b → minValue ≤ (a^2 + b^2) / (a - b)) := 
sorry

end min_value_fraction_l74_74221


namespace binom_12_10_eq_66_l74_74909

theorem binom_12_10_eq_66 : Nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l74_74909


namespace max_min_values_l74_74847

-- Define the function f(x) = x^2 - 2ax + 1
def f (x a : ℝ) : ℝ := x ^ 2 - 2 * a * x + 1

-- Define the interval [0, 2]
def interval : Set ℝ := Set.Icc 0 2

theorem max_min_values (a : ℝ) : 
  (a > 2 → (∀ x ∈ interval, f x a ≤ 1) ∧ (∃ x ∈ interval, f x a = 5 - 4 * a))
  ∧ (1 ≤ a ∧ a ≤ 2 → (∀ x ∈ interval, f x a ≤ 5 - 4 * a) ∧ (∃ x ∈ interval, f x a = -a^2 + 1))
  ∧ (0 ≤ a ∧ a < 1 → (∀ x ∈ interval, f x a ≤ 1) ∧ (∃ x ∈ interval, f x a = -a^2 + 1))
  ∧ (a < 0 → (∀ x ∈ interval, f x a ≤ 5 - 4 * a) ∧ (∃ x ∈ interval, f x a = 1)) := by
  sorry

end max_min_values_l74_74847


namespace one_greater_others_less_l74_74805

theorem one_greater_others_less {a b c : ℝ} (h1 : a > 0 ∧ b > 0 ∧ c > 0) (h2 : a * b * c = 1) (h3 : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ (b > 1 ∧ a < 1 ∧ c < 1) ∨ (c > 1 ∧ a < 1 ∧ b < 1) :=
by
  sorry

end one_greater_others_less_l74_74805


namespace cubes_with_odd_neighbors_in_5x5x5_l74_74823

theorem cubes_with_odd_neighbors_in_5x5x5 (unit_cubes : Fin 125 → ℕ) 
  (neighbors : ∀ (i : Fin 125), Fin 125 → Prop) : ∃ n, n = 62 := 
by
  sorry

end cubes_with_odd_neighbors_in_5x5x5_l74_74823


namespace new_average_age_after_person_leaves_l74_74176

theorem new_average_age_after_person_leaves (avg_age : ℕ) (n : ℕ) (leaving_age : ℕ) (remaining_count : ℕ) :
  ((n * avg_age - leaving_age) / remaining_count) = 33 :=
by
  -- Given conditions
  let avg_age := 30
  let n := 5
  let leaving_age := 18
  let remaining_count := n - 1
  -- Conclusion
  sorry

end new_average_age_after_person_leaves_l74_74176


namespace average_sweater_less_by_21_after_discount_l74_74659

theorem average_sweater_less_by_21_after_discount
  (shirt_count sweater_count jeans_count : ℕ)
  (total_shirt_price total_sweater_price total_jeans_price : ℕ)
  (shirt_discount sweater_discount jeans_discount : ℕ)
  (shirt_avg_before_discount sweater_avg_before_discount jeans_avg_before_discount 
   shirt_avg_after_discount sweater_avg_after_discount jeans_avg_after_discount : ℕ) :
  shirt_count = 20 →
  sweater_count = 45 →
  jeans_count = 30 →
  total_shirt_price = 360 →
  total_sweater_price = 900 →
  total_jeans_price = 1200 →
  shirt_discount = 2 →
  sweater_discount = 4 →
  jeans_discount = 3 →
  shirt_avg_before_discount = total_shirt_price / shirt_count →
  sweater_avg_before_discount = total_sweater_price / sweater_count →
  jeans_avg_before_discount = total_jeans_price / jeans_count →
  shirt_avg_after_discount = shirt_avg_before_discount - shirt_discount →
  sweater_avg_after_discount = sweater_avg_before_discount - sweater_discount →
  jeans_avg_after_discount = jeans_avg_before_discount - jeans_discount →
  sweater_avg_after_discount = shirt_avg_after_discount →
  jeans_avg_after_discount - sweater_avg_after_discount = 21 :=
by
  intros
  sorry

end average_sweater_less_by_21_after_discount_l74_74659


namespace expr_eval_l74_74759

theorem expr_eval : 3^3 - 3^2 + 3^1 - 3^0 = 20 := by
  sorry

end expr_eval_l74_74759


namespace quadratic_passes_through_l74_74799

def quadratic_value_at_point (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_passes_through (a b c : ℝ) :
  quadratic_value_at_point a b c 1 = 5 ∧ 
  quadratic_value_at_point a b c 3 = n ∧ 
  a * (-2)^2 + b * (-2) + c = -8 ∧ 
  (-4*a + b = 0) → 
  n = 253/9 := 
sorry

end quadratic_passes_through_l74_74799


namespace polynomial_roots_l74_74319

theorem polynomial_roots :
  (∀ x, x^3 - 3 * x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := 
by
  sorry

end polynomial_roots_l74_74319


namespace number_of_permutations_l74_74569

def total_letters : ℕ := 10
def freq_s : ℕ := 3
def freq_t : ℕ := 2
def freq_i : ℕ := 2
def freq_a : ℕ := 1
def freq_c : ℕ := 1

theorem number_of_permutations : 
  (total_letters.factorial / (freq_s.factorial * freq_t.factorial * freq_i.factorial * freq_a.factorial * freq_c.factorial)) = 75600 :=
by
  sorry

end number_of_permutations_l74_74569


namespace bake_sale_total_money_l74_74899

def dozens_to_pieces (dozens : Nat) : Nat :=
  dozens * 12

def total_money_raised
  (betty_chocolate_chip_dozen : Nat)
  (betty_oatmeal_raisin_dozen : Nat)
  (betty_brownies_dozen : Nat)
  (paige_sugar_cookies_dozen : Nat)
  (paige_blondies_dozen : Nat)
  (paige_cream_cheese_brownies_dozen : Nat)
  (price_per_cookie : Rat)
  (price_per_brownie_blondie : Rat) : Rat :=
let betty_cookies := dozens_to_pieces betty_chocolate_chip_dozen + dozens_to_pieces betty_oatmeal_raisin_dozen
let paige_cookies := dozens_to_pieces paige_sugar_cookies_dozen
let total_cookies := betty_cookies + paige_cookies
let betty_brownies := dozens_to_pieces betty_brownies_dozen
let paige_brownies_blondies := dozens_to_pieces paige_blondies_dozen + dozens_to_pieces paige_cream_cheese_brownies_dozen
let total_brownies_blondies := betty_brownies + paige_brownies_blondies
(total_cookies * price_per_cookie) + (total_brownies_blondies * price_per_brownie_blondie)

theorem bake_sale_total_money :
  total_money_raised 4 6 2 6 3 5 1 2 = 432 :=
by
  sorry

end bake_sale_total_money_l74_74899


namespace total_bill_correct_l74_74560

def first_family_adults := 2
def first_family_children := 3
def second_family_adults := 4
def second_family_children := 2
def third_family_adults := 3
def third_family_children := 4

def adult_meal_cost := 8
def child_meal_cost := 5
def drink_cost_per_person := 2

def calculate_total_cost 
  (adults1 : ℕ) (children1 : ℕ) 
  (adults2 : ℕ) (children2 : ℕ) 
  (adults3 : ℕ) (children3 : ℕ)
  (adult_cost : ℕ) (child_cost : ℕ)
  (drink_cost : ℕ) : ℕ := 
  let meal_cost1 := (adults1 * adult_cost) + (children1 * child_cost)
  let meal_cost2 := (adults2 * adult_cost) + (children2 * child_cost)
  let meal_cost3 := (adults3 * adult_cost) + (children3 * child_cost)
  let drink_cost1 := (adults1 + children1) * drink_cost
  let drink_cost2 := (adults2 + children2) * drink_cost
  let drink_cost3 := (adults3 + children3) * drink_cost
  meal_cost1 + drink_cost1 + meal_cost2 + drink_cost2 + meal_cost3 + drink_cost3
   
theorem total_bill_correct :
  calculate_total_cost
    first_family_adults first_family_children
    second_family_adults second_family_children
    third_family_adults third_family_children
    adult_meal_cost child_meal_cost drink_cost_per_person = 153 :=
  sorry

end total_bill_correct_l74_74560


namespace interest_rate_first_part_l74_74475

theorem interest_rate_first_part 
  (total_amount : ℤ) 
  (amount_at_first_rate : ℤ) 
  (amount_at_second_rate : ℤ) 
  (rate_second_part : ℤ) 
  (total_annual_interest : ℤ) 
  (r : ℤ) 
  (h_split : total_amount = amount_at_first_rate + amount_at_second_rate) 
  (h_second : rate_second_part = 5)
  (h_interest : (amount_at_first_rate * r) / 100 + (amount_at_second_rate * rate_second_part) / 100 = total_annual_interest) :
  r = 3 := 
by 
  sorry

end interest_rate_first_part_l74_74475


namespace rectangle_other_side_l74_74154

theorem rectangle_other_side
  (a b : ℝ)
  (Area : ℝ := 12 * a ^ 2 - 6 * a * b)
  (side1 : ℝ := 3 * a)
  (side2 : ℝ := Area / side1) :
  side2 = 4 * a - 2 * b :=
by
  sorry

end rectangle_other_side_l74_74154


namespace days_to_complete_work_l74_74367

variable {P W D : ℕ}

axiom condition_1 : 2 * P * 3 = W / 2
axiom condition_2 : P * D = W

theorem days_to_complete_work : D = 12 :=
by
  -- As an axiom or sorry is used, the proof is omitted.
  sorry

end days_to_complete_work_l74_74367


namespace maximum_area_rectangular_backyard_l74_74236

theorem maximum_area_rectangular_backyard (x : ℕ) (y : ℕ) (h_perimeter : 2 * (x + y) = 100) : 
  x * y ≤ 625 :=
by
  sorry

end maximum_area_rectangular_backyard_l74_74236


namespace r_daily_earnings_l74_74116

def earnings_problem (P Q R : ℝ) : Prop :=
  (9 * (P + Q + R) = 1890) ∧ 
  (5 * (P + R) = 600) ∧ 
  (7 * (Q + R) = 910)

theorem r_daily_earnings :
  ∃ P Q R : ℝ, earnings_problem P Q R ∧ R = 40 := sorry

end r_daily_earnings_l74_74116


namespace janet_earned_1390_in_interest_l74_74128

def janets_total_interest (total_investment investment_at_10_rate investment_at_10_interest investment_at_1_rate remaining_investment remaining_investment_interest : ℝ) : ℝ :=
    investment_at_10_interest + remaining_investment_interest

theorem janet_earned_1390_in_interest :
  janets_total_interest 31000 12000 0.10 (12000 * 0.10) 0.01 (19000 * 0.01) = 1390 :=
by
  sorry

end janet_earned_1390_in_interest_l74_74128


namespace find_C_l74_74864

theorem find_C (A B C : ℕ) :
  (8 + 5 + 6 + 3 + 2 + A + B) % 3 = 0 →
  (4 + 3 + 7 + 5 + A + B + C) % 3 = 0 →
  C = 2 :=
by
  intros h1 h2
  sorry

end find_C_l74_74864


namespace half_plus_five_l74_74920

theorem half_plus_five (n : ℕ) (h : n = 16) : n / 2 + 5 = 13 := by
  sorry

end half_plus_five_l74_74920


namespace hyperbola_standard_equation_correct_l74_74592

-- Define the initial values given in conditions
def a : ℝ := 12
def b : ℝ := 5
def c : ℝ := 4

-- Define the hyperbola equation form based on conditions and focal properties
noncomputable def hyperbola_standard_equation : Prop :=
  let a2 := (8 / 5)
  let b2 := (72 / 5)
  (∀ x y : ℝ, y^2 / a2 - x^2 / b2 = 1)

-- State the final problem as a theorem
theorem hyperbola_standard_equation_correct :
  ∀ x y : ℝ, y^2 / (8 / 5) - x^2 / (72 / 5) = 1 :=
by
  sorry

end hyperbola_standard_equation_correct_l74_74592


namespace inequality_on_abc_l74_74862

theorem inequality_on_abc (α β γ : ℝ) (h : α^2 + β^2 + γ^2 = 1) :
  -1/2 ≤ α * β + β * γ + γ * α ∧ α * β + β * γ + γ * α ≤ 1 :=
by {
  sorry -- Proof to be added
}

end inequality_on_abc_l74_74862


namespace number_of_true_propositions_l74_74050

variable {a b c : ℝ}

theorem number_of_true_propositions :
  (2 = (if (a > b → a * c ^ 2 > b * c ^ 2) then 1 else 0) +
       (if (a * c ^ 2 > b * c ^ 2 → a > b) then 1 else 0) +
       (if (¬(a * c ^ 2 > b * c ^ 2) → ¬(a > b)) then 1 else 0) +
       (if (¬(a > b) → ¬(a * c ^ 2 > b * c ^ 2)) then 1 else 0)) :=
sorry

end number_of_true_propositions_l74_74050


namespace original_price_of_movie_ticket_l74_74264

theorem original_price_of_movie_ticket
    (P : ℝ)
    (new_price : ℝ)
    (h1 : new_price = 80)
    (h2 : new_price = 0.80 * P) :
    P = 100 :=
by
  sorry

end original_price_of_movie_ticket_l74_74264


namespace number_100_in_row_15_l74_74788

theorem number_100_in_row_15 (A : ℕ) (H1 : 1 ≤ A)
  (H2 : ∀ n : ℕ, n > 0 → n ≤ 100 * A)
  (H3 : ∃ k : ℕ, 4 * A + 1 ≤ 31 ∧ 31 ≤ 5 * A ∧ k = 5):
  ∃ r : ℕ, (14 * A + 1 ≤ 100 ∧ 100 ≤ 15 * A ∧ r = 15) :=
by {
  sorry
}

end number_100_in_row_15_l74_74788


namespace altitude_length_l74_74284

theorem altitude_length {s t : ℝ} 
  (A B C : ℝ × ℝ) 
  (hA : A = (-s, s^2))
  (hB : B = (s, s^2))
  (hC : C = (t, t^2))
  (h_parabola_A : A.snd = (A.fst)^2)
  (h_parabola_B : B.snd = (B.fst)^2)
  (h_parabola_C : C.snd = (C.fst)^2)
  (hyp_parallel : A.snd = B.snd)
  (right_triangle : (t + s) * (t - s) + (t^2 - s^2)^2 = 0) :
  (s^2 - (t^2)) = 1 :=
by
  sorry

end altitude_length_l74_74284


namespace stepa_and_petya_are_wrong_l74_74283

-- Define the six-digit number where all digits are the same.
def six_digit_same (a : ℕ) : ℕ := a * 111111

-- Define the sum of distinct prime divisors of 1001 and 111.
def prime_divisor_sum : ℕ := 7 + 11 + 13 + 3 + 37

-- Define the sum of prime divisors when a is considered.
def additional_sum (a : ℕ) : ℕ :=
  if (a = 2) || (a = 6) || (a = 8) then 2
  else if (a = 5) then 5
  else 0

-- Summarize the possible correct sums
def correct_sums (a : ℕ) : ℕ := prime_divisor_sum + additional_sum a

-- The proof statement
theorem stepa_and_petya_are_wrong (a : ℕ) :
  correct_sums a ≠ 70 ∧ correct_sums a ≠ 80 := 
by {
  sorry
}

end stepa_and_petya_are_wrong_l74_74283


namespace solution_set_of_inequality_l74_74411

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_inequality_l74_74411


namespace cos_seven_theta_l74_74755

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l74_74755


namespace cyclic_points_exist_l74_74432

noncomputable def f (x : ℝ) : ℝ := 
if x < (1 / 3) then 
  2 * x + (1 / 3) 
else 
  (3 / 2) * (1 - x)

theorem cyclic_points_exist :
  ∃ (x0 x1 x2 x3 x4 : ℝ), 
  0 ≤ x0 ∧ x0 ≤ 1 ∧
  0 ≤ x1 ∧ x1 ≤ 1 ∧
  0 ≤ x2 ∧ x2 ≤ 1 ∧
  0 ≤ x3 ∧ x3 ≤ 1 ∧
  0 ≤ x4 ∧ x4 ≤ 1 ∧
  x0 ≠ x1 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ x4 ≠ x0 ∧
  f x0 = x1 ∧ f x1 = x2 ∧ f x2 = x3 ∧ f x3 = x4 ∧ f x4 = x0 :=
sorry

end cyclic_points_exist_l74_74432


namespace new_average_production_l74_74243

theorem new_average_production (n : ℕ) (average_past today : ℕ) (h₁ : average_past = 70) (h₂ : today = 90) (h₃ : n = 3) : 
  (average_past * n + today) / (n + 1) = 75 := by
  sorry

end new_average_production_l74_74243


namespace cost_per_topping_is_2_l74_74871

theorem cost_per_topping_is_2 : 
  ∃ (x : ℝ), 
    let large_pizza_cost := 14 
    let num_large_pizzas := 2 
    let num_toppings_per_pizza := 3 
    let tip_rate := 0.25 
    let total_cost := 50 
    let cost_pizzas := num_large_pizzas * large_pizza_cost 
    let num_toppings := num_large_pizzas * num_toppings_per_pizza 
    let cost_toppings := num_toppings * x 
    let before_tip_cost := cost_pizzas + cost_toppings 
    let tip := tip_rate * before_tip_cost 
    let final_cost := before_tip_cost + tip 
    final_cost = total_cost ∧ x = 2 := 
by
  simp
  sorry

end cost_per_topping_is_2_l74_74871


namespace xiao_qian_has_been_to_great_wall_l74_74504

-- Define the four students
inductive Student
| XiaoZhao
| XiaoQian
| XiaoSun
| XiaoLi

open Student

-- Define the relations for their statements
def has_been (s : Student) : Prop :=
  match s with
  | XiaoZhao => false
  | XiaoQian => true
  | XiaoSun => true
  | XiaoLi => false

def said (s : Student) : Prop :=
  match s with
  | XiaoZhao => ¬has_been XiaoZhao
  | XiaoQian => has_been XiaoLi
  | XiaoSun => has_been XiaoQian
  | XiaoLi => ¬has_been XiaoLi

axiom only_one_lying : ∃ l : Student, ∀ s : Student, said s → (s ≠ l)

theorem xiao_qian_has_been_to_great_wall : has_been XiaoQian :=
by {
  sorry -- Proof elided
}

end xiao_qian_has_been_to_great_wall_l74_74504


namespace young_employees_l74_74004

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

end young_employees_l74_74004


namespace part_a_part_b_l74_74835

theorem part_a (n : ℕ) (h_n : 1 < n) (d : ℝ) (h_d : d = 1) (μ : ℝ) (h_μ : 0 < μ ∧ μ < (2 * (Real.sqrt n + 1) / (n - 1))) :
  μ < (2 * (Real.sqrt n + 1) / (n - 1)) :=
by 
  exact h_μ.2

theorem part_b (n : ℕ) (h_n : 1 < n) (d : ℝ) (h_d : d = 1) (μ : ℝ) (h_μ : 0 < μ ∧ μ < (2 * Real.sqrt 3 * (Real.sqrt n + 1) / (3 * (n - 1)))) :
  μ < (2 * Real.sqrt 3 * (Real.sqrt n + 1) / (3 * (n - 1))) :=
by
  exact h_μ.2

end part_a_part_b_l74_74835


namespace roots_of_polynomial_l74_74532

noncomputable def p (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial : {x : ℝ | p x = 0} = {1, -1, 3} :=
by
  sorry

end roots_of_polynomial_l74_74532


namespace combined_perimeter_two_right_triangles_l74_74416

theorem combined_perimeter_two_right_triangles :
  ∀ (h1 h2 : ℝ),
    (h1^2 = 15^2 + 20^2) ∧
    (h2^2 = 9^2 + 12^2) ∧
    (h1 = h2) →
    (15 + 20 + h1) + (9 + 12 + h2) = 106 := by
  sorry

end combined_perimeter_two_right_triangles_l74_74416


namespace no_such_integers_exist_l74_74896

theorem no_such_integers_exist : ¬ ∃ (n k : ℕ), n > 0 ∧ k > 0 ∧ (n ∣ (k ^ n - 1)) ∧ (n.gcd (k - 1) = 1) :=
by
  sorry

end no_such_integers_exist_l74_74896


namespace basketball_scores_l74_74270

theorem basketball_scores :
  (∃ P : Finset ℕ, P = { P | ∃ x : ℕ, x ∈ (Finset.range 8) ∧ P = x + 14 } ∧ P.card = 8) :=
by
  sorry

end basketball_scores_l74_74270


namespace largest_n_satisfying_conditions_l74_74671

theorem largest_n_satisfying_conditions :
  ∃ n : ℤ, n = 181 ∧
    (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧
    ∃ k : ℤ, 2 * n + 79 = k^2 :=
by
  sorry

end largest_n_satisfying_conditions_l74_74671


namespace min_value_2a_plus_b_value_of_t_l74_74598

theorem min_value_2a_plus_b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 2/b = 2) :
  2 * a + b = 4 :=
sorry

theorem value_of_t (a b t : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 2/b = 2) (h₄ : 4^a = t) (h₅ : 3^b = t) :
  t = 6 :=
sorry

end min_value_2a_plus_b_value_of_t_l74_74598


namespace lines_perpendicular_l74_74559

variable (b : ℝ)

/-- Proof that if the given lines are perpendicular, then b must be 3 -/
theorem lines_perpendicular (h : b ≠ 0) :
    let l₁_slope := -3
    let l₂_slope := b / 9
    l₁_slope * l₂_slope = -1 → b = 3 :=
by
  intros slope_prod
  simp only [h]
  sorry

end lines_perpendicular_l74_74559


namespace f_at_2_f_pos_solution_set_l74_74214

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - (3 - a) * x + 2 * (1 - a)

-- Question (I)
theorem f_at_2 : f a 2 = 0 := by sorry

-- Question (II)
theorem f_pos_solution_set :
  (∀ x, (a < -1 → (f a x > 0 ↔ (x < 2 ∨ 1 - a < x))) ∧
       (a = -1 → ¬(f a x > 0)) ∧
       (a > -1 → (f a x > 0 ↔ (1 - a < x ∧ x < 2)))) := 
by sorry

end f_at_2_f_pos_solution_set_l74_74214


namespace real_part_of_complex_l74_74978

theorem real_part_of_complex (a : ℝ) (h : a^2 + 2 * a - 15 = 0 ∧ a + 5 ≠ 0) : a = 3 :=
by sorry

end real_part_of_complex_l74_74978


namespace solve_equation_l74_74161

-- Define the equation as a function of y
def equation (y : ℝ) : ℝ :=
  y^4 - 20 * y + 1

-- State the theorem that y = -1 satisfies the equation.
theorem solve_equation : equation (-1) = 22 := 
  sorry

end solve_equation_l74_74161


namespace pages_difference_l74_74347

def second_chapter_pages : ℕ := 18
def third_chapter_pages : ℕ := 3

theorem pages_difference : second_chapter_pages - third_chapter_pages = 15 := by 
  sorry

end pages_difference_l74_74347


namespace no_integral_solutions_l74_74716

theorem no_integral_solutions : ∀ (x : ℤ), x^5 - 31 * x + 2015 ≠ 0 :=
by
  sorry

end no_integral_solutions_l74_74716


namespace current_at_resistance_12_l74_74115

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l74_74115


namespace alchemists_less_than_half_l74_74530

variable (k c a : ℕ)

theorem alchemists_less_than_half (h1 : k = c + a) (h2 : c > a) : a < k / 2 := by
  sorry

end alchemists_less_than_half_l74_74530


namespace average_age_population_l74_74564

theorem average_age_population 
  (k : ℕ) 
  (hwomen : ℕ := 7 * k)
  (hmen : ℕ := 5 * k)
  (avg_women_age : ℕ := 40)
  (avg_men_age : ℕ := 30)
  (h_age_women : ℕ := avg_women_age * hwomen)
  (h_age_men : ℕ := avg_men_age * hmen) : 
  (h_age_women + h_age_men) / (hwomen + hmen) = 35 + 5/6 :=
by
  sorry -- proof will fill in here

end average_age_population_l74_74564


namespace num_boys_l74_74878

theorem num_boys (total_students : ℕ) (girls_ratio boys_ratio others_ratio : ℕ) (r : girls_ratio = 4) (b : boys_ratio = 3) (o : others_ratio = 2) (total_eq : girls_ratio * k + boys_ratio * k + others_ratio * k = total_students) (total_given : total_students = 63) : 
  boys_ratio * k = 21 :=
by
  sorry

end num_boys_l74_74878


namespace correct_option_C_l74_74963

theorem correct_option_C (a b c : ℝ) : 2 * a^2 * b * c - a^2 * b * c = a^2 * b * c := 
sorry

end correct_option_C_l74_74963


namespace five_alpha_plus_two_beta_is_45_l74_74768

theorem five_alpha_plus_two_beta_is_45
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 1 / 7) 
  (tan_β : Real.tan β = 3 / 79) :
  5 * α + 2 * β = π / 4 :=
by
  sorry

end five_alpha_plus_two_beta_is_45_l74_74768


namespace find_k_l74_74328

open Real

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_k
  (a b : vector)
  (h_a : a = (2, -1))
  (h_b : b = (-1, 4))
  (h_perpendicular : dot_product (a.1 - k * b.1, a.2 + 4 * k) (3, -5) = 0) :
  k = -11/17 := sorry

end find_k_l74_74328


namespace anna_original_money_l74_74013

theorem anna_original_money (x : ℝ) (h : (3 / 4) * x = 24) : x = 32 :=
by
  sorry

end anna_original_money_l74_74013


namespace h_h_of_2_l74_74309

def h (x : ℝ) : ℝ := 4 * x^2 - 8

theorem h_h_of_2 : h (h 2) = 248 := by
  -- Proof goes here
  sorry

end h_h_of_2_l74_74309


namespace tenth_term_is_correct_l74_74842

-- Conditions and calculation
variable (a l : ℚ)
variable (d : ℚ)
variable (a10 : ℚ)

-- Setting the given values:
noncomputable def first_term : ℚ := 2 / 3
noncomputable def seventeenth_term : ℚ := 3 / 2
noncomputable def common_difference : ℚ := (seventeenth_term - first_term) / 16

-- Calculate the tenth term using the common difference
noncomputable def tenth_term : ℚ := first_term + 9 * common_difference

-- Statement to prove
theorem tenth_term_is_correct : 
  first_term = 2 / 3 →
  seventeenth_term = 3 / 2 →
  common_difference = (3 / 2 - 2 / 3) / 16 →
  tenth_term = 2 / 3 + 9 * ((3 / 2 - 2 / 3) / 16) →
  tenth_term = 109 / 96 :=
  by
    sorry

end tenth_term_is_correct_l74_74842


namespace smallest_x_abs_eq_18_l74_74387

theorem smallest_x_abs_eq_18 : 
  ∃ x : ℝ, (|2 * x + 5| = 18) ∧ (∀ y : ℝ, (|2 * y + 5| = 18) → x ≤ y) :=
sorry

end smallest_x_abs_eq_18_l74_74387


namespace geometric_sequence_expression_l74_74389

theorem geometric_sequence_expression (a : ℕ → ℝ) (q : ℝ) (h_q : q = 4)
  (h_geom : ∀ n, a (n + 1) = q * a n) (h_sum : a 0 + a 1 + a 2 = 21) :
  ∀ n, a n = 4 ^ n :=
by sorry

end geometric_sequence_expression_l74_74389


namespace proof_problem_l74_74938

-- Definitions for the given conditions in the problem
def equations (a x y : ℝ) : Prop :=
(x + 5 * y = 4 - a) ∧ (x - y = 3 * a)

-- The conclusions from the problem
def conclusion1 (a x y : ℝ) : Prop :=
a = 1 → x + y = 4 - a

def conclusion2 (a x y : ℝ) : Prop :=
a = -2 → x = -y

def conclusion3 (a x y : ℝ) : Prop :=
2 * x + 7 * y = 6

def conclusion4 (a x y : ℝ) : Prop :=
x ≤ 1 → y > 4 / 7

-- The main theorem to be proven
theorem proof_problem (a x y : ℝ) :
  equations a x y →
  (¬ conclusion1 a x y ∨ ¬ conclusion2 a x y ∨ ¬ conclusion3 a x y ∨ ¬ conclusion4 a x y) →
  (∃ n : ℕ, n = 2 ∧ ((conclusion1 a x y ∨ conclusion2 a x y ∨ conclusion3 a x y ∨ conclusion4 a x y) → false)) :=
by {
  sorry
}

end proof_problem_l74_74938


namespace complex_equality_l74_74386

theorem complex_equality (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : a - b * i = (1 + i) * i^3) : a = 1 ∧ b = -1 :=
by sorry

end complex_equality_l74_74386


namespace R_and_D_per_increase_l74_74299

def R_and_D_t : ℝ := 3013.94
def Delta_APL_t2 : ℝ := 3.29

theorem R_and_D_per_increase :
  R_and_D_t / Delta_APL_t2 = 916 := by
  sorry

end R_and_D_per_increase_l74_74299


namespace printing_time_345_l74_74966

def printing_time (total_pages : ℕ) (rate : ℕ) : ℕ :=
  total_pages / rate

theorem printing_time_345 :
  printing_time 345 23 = 15 :=
by
  sorry

end printing_time_345_l74_74966


namespace arithmetic_evaluation_l74_74252

theorem arithmetic_evaluation : (10 - 9^2 + 8 * 7 + 6^2 - 5 * 4 + 3 - 2^3) = -4 :=
by
  sorry

end arithmetic_evaluation_l74_74252


namespace quadruples_characterization_l74_74417

/-- Proving the characterization of quadruples (a, b, c, d) of non-negative integers 
such that ab = 2(1 + cd) and there exists a non-degenerate triangle with sides (a - c), 
(b - d), and (c + d). -/
theorem quadruples_characterization :
  ∀ (a b c d : ℕ), 
    ab = 2 * (1 + cd) ∧ 
    (a - c) + (b - d) > c + d ∧ 
    (a - c) + (c + d) > b - d ∧ 
    (b - d) + (c + d) > a - c ∧
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    (a = 1 ∧ b = 2 ∧ c = 0 ∧ d = 1) ∨ 
    (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 0) :=
by sorry

end quadruples_characterization_l74_74417


namespace problem_c_l74_74278

theorem problem_c (x y : ℝ) (h : x - 3 = y - 3): x - y = 0 :=
by
  sorry

end problem_c_l74_74278


namespace hexagon_internal_angle_A_l74_74371

theorem hexagon_internal_angle_A
  (B C D E F : ℝ) 
  (hB : B = 134) 
  (hC : C = 98) 
  (hD : D = 120) 
  (hE : E = 139) 
  (hF : F = 109) 
  (H : B + C + D + E + F + A = 720) : A = 120 := 
sorry

end hexagon_internal_angle_A_l74_74371


namespace range_of_m_for_false_proposition_l74_74549

theorem range_of_m_for_false_proposition :
  (∀ x ∈ (Set.Icc 0 (Real.pi / 4)), Real.tan x < m) → False ↔ m ≤ 1 :=
by
  sorry

end range_of_m_for_false_proposition_l74_74549


namespace min_additional_trains_needed_l74_74726

-- Definitions
def current_trains : ℕ := 31
def trains_per_row : ℕ := 8
def smallest_num_additional_trains (current : ℕ) (per_row : ℕ) : ℕ :=
  let next_multiple := ((current + per_row - 1) / per_row) * per_row
  next_multiple - current

-- Theorem
theorem min_additional_trains_needed :
  smallest_num_additional_trains current_trains trains_per_row = 1 :=
by
  sorry

end min_additional_trains_needed_l74_74726


namespace find_constant_t_l74_74329

theorem find_constant_t (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = t + 5^n) ∧ (∀ n ≥ 2, a n = S n - S (n - 1)) ∧ (a 1 = S 1) ∧ 
  (∃ q, ∀ n ≥ 1, a (n + 1) = q * a n) → 
  t = -1 := by
  sorry

end find_constant_t_l74_74329


namespace expand_polynomial_l74_74379

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l74_74379


namespace speed_ratio_l74_74733

theorem speed_ratio (v_A v_B : ℝ) (t : ℝ) (h1 : v_A = 200 / t) (h2 : v_B = 120 / t) : 
  v_A / v_B = 5 / 3 :=
by
  sorry

end speed_ratio_l74_74733


namespace problem_1_problem_2_l74_74803

noncomputable def f (x : ℝ) : ℝ :=
  (Real.logb 3 (x / 27)) * (Real.logb 3 (3 * x))

theorem problem_1 (h₁ : 1 / 27 ≤ x)
(h₂ : x ≤ 1 / 9) :
    (∀ x, f x ≤ 12) ∧ (∃ x, f x = 5) := 
sorry

theorem problem_2
(m α β : ℝ)
(h₁ : f α + m = 0)
(h₂ : f β + m = 0) :
    α * β = 9 :=
sorry

end problem_1_problem_2_l74_74803


namespace part1_a_value_part2_solution_part3_incorrect_solution_l74_74271

-- Part 1: Given solution {x = 1, y = 1}, prove a = 3
theorem part1_a_value (a : ℤ) (h1 : 1 + 2 * 1 = a) : a = 3 := 
by 
  sorry

-- Part 2: Given a = -2, prove the solution is {x = 0, y = -1}
theorem part2_solution (x y : ℤ) (h1 : x + 2 * y = -2) (h2 : 2 * x - y = 1) : x = 0 ∧ y = -1 := 
by 
  sorry

-- Part 3: Given {x = -2, y = -2}, prove that it is not a solution
theorem part3_incorrect_solution (a : ℤ) (h1 : -2 + 2 * (-2) = a) (h2 : 2 * (-2) - (-2) = 1) : False := 
by 
  sorry

end part1_a_value_part2_solution_part3_incorrect_solution_l74_74271


namespace simplify_product_of_fractions_l74_74223

theorem simplify_product_of_fractions :
  8 * (15 / 4) * (-28 / 45) = -56 / 3 := by
  sorry

end simplify_product_of_fractions_l74_74223


namespace range_of_m_l74_74637

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → (7 / 4) ≤ (x^2 - 3 * x + 4) ∧ (x^2 - 3 * x + 4) ≤ 4) ↔ (3 / 2 ≤ m ∧ m ≤ 3) := 
sorry

end range_of_m_l74_74637


namespace groupB_is_basis_l74_74657

section
variables (eA1 eA2 : ℝ × ℝ) (eB1 eB2 : ℝ × ℝ) (eC1 eC2 : ℝ × ℝ) (eD1 eD2 : ℝ × ℝ)

def is_collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k • w) ∨ w = (k • v)

-- Define each vector group
def groupA := eA1 = (0, 0) ∧ eA2 = (1, -2)
def groupB := eB1 = (-1, 2) ∧ eB2 = (5, 7)
def groupC := eC1 = (3, 5) ∧ eC2 = (6, 10)
def groupD := eD1 = (2, -3) ∧ eD2 = (1/2, -3/4)

-- The goal is to prove that group B vectors can serve as a basis
theorem groupB_is_basis : ¬ is_collinear eB1 eB2 :=
sorry
end

end groupB_is_basis_l74_74657


namespace not_beautiful_739_and_741_l74_74366

-- Define the function g and its properties
variable (g : ℤ → ℤ)

-- Condition: g(x) ≠ x
axiom g_neq_x (x : ℤ) : g x ≠ x

-- Definition of "beautiful"
def beautiful (a : ℤ) : Prop :=
  ∀ x : ℤ, g x = g (a - x)

-- The theorem to prove
theorem not_beautiful_739_and_741 :
  ¬ (beautiful g 739 ∧ beautiful g 741) :=
sorry

end not_beautiful_739_and_741_l74_74366


namespace meat_needed_for_40_hamburgers_l74_74267

theorem meat_needed_for_40_hamburgers (meat_per_10_hamburgers : ℕ) (hamburgers_needed : ℕ) (meat_per_hamburger : ℚ) (total_meat_needed : ℚ) :
  meat_per_10_hamburgers = 5 ∧ hamburgers_needed = 40 ∧
  meat_per_hamburger = meat_per_10_hamburgers / 10 ∧
  total_meat_needed = meat_per_hamburger * hamburgers_needed → 
  total_meat_needed = 20 := by
  sorry

end meat_needed_for_40_hamburgers_l74_74267


namespace ratio_of_speeds_l74_74393

variable (a b : ℝ)

theorem ratio_of_speeds (h1 : b = 1 / 60) (h2 : a + b = 1 / 12) : a / b = 4 := 
sorry

end ratio_of_speeds_l74_74393


namespace least_possible_value_of_b_l74_74171

theorem least_possible_value_of_b (a b : ℕ) 
  (ha : ∃ p, (∀ q, p ∣ q ↔ q = 1 ∨ q = p ∨ q = p*p ∨ q = a))
  (hb : ∃ k, (∀ l, k ∣ l ↔ (l = 1 ∨ l = b)))
  (hdiv : a ∣ b) : 
  b = 12 :=
sorry

end least_possible_value_of_b_l74_74171


namespace other_x_intercept_vertex_symmetric_l74_74141

theorem other_x_intercept_vertex_symmetric (a b c : ℝ)
  (h_vertex : ∀ x y : ℝ, (4, 10) = (x, y) → y = a * x^2 + b * x + c)
  (h_intercept : ∀ x : ℝ, (-1, 0) = (x, 0) → a * x^2 + b * x + c = 0) :
  a * 9^2 + b * 9 + c = 0 :=
sorry

end other_x_intercept_vertex_symmetric_l74_74141


namespace cos_double_angle_l74_74048

theorem cos_double_angle (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 3 / 5) : Real.cos (2 * α) = -7 / 25 :=
sorry

end cos_double_angle_l74_74048


namespace daily_salmon_l74_74508

-- Definitions of the daily consumption of trout and total fish
def daily_trout : ℝ := 0.2
def daily_total_fish : ℝ := 0.6

-- Theorem statement that the daily consumption of salmon is 0.4 buckets
theorem daily_salmon : daily_total_fish - daily_trout = 0.4 := 
by
  -- Skipping the proof, as required
  sorry

end daily_salmon_l74_74508


namespace william_washed_2_normal_cars_l74_74007

def time_spent_on_one_normal_car : Nat := 4 + 7 + 4 + 9

def time_spent_on_suv : Nat := 2 * time_spent_on_one_normal_car

def total_time_spent : Nat := 96

def time_spent_on_normal_cars : Nat := total_time_spent - time_spent_on_suv

def number_of_normal_cars : Nat := time_spent_on_normal_cars / time_spent_on_one_normal_car

theorem william_washed_2_normal_cars : number_of_normal_cars = 2 := by
  sorry

end william_washed_2_normal_cars_l74_74007


namespace no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5_l74_74536

theorem no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5 :
  ¬ ∃ n : ℕ, (∀ d ∈ (Nat.digits 10 n), 5 < d) ∧ (∀ d ∈ (Nat.digits 10 (n^2)), d < 5) :=
by
  sorry

end no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5_l74_74536


namespace multiples_of_15_between_12_and_152_l74_74537

theorem multiples_of_15_between_12_and_152 : 
  ∃ n : ℕ, n = 10 ∧ ∀ m : ℕ, (m * 15 > 12 ∧ m * 15 < 152) ↔ (1 ≤ m ∧ m ≤ 10) :=
by
  sorry

end multiples_of_15_between_12_and_152_l74_74537


namespace remainder_of_3045_div_32_l74_74979

theorem remainder_of_3045_div_32 : 3045 % 32 = 5 :=
by sorry

end remainder_of_3045_div_32_l74_74979


namespace passengers_remaining_after_fourth_stop_l74_74567

theorem passengers_remaining_after_fourth_stop :
  let initial_passengers := 64
  let remaining_fraction := 2 / 3
  (initial_passengers * remaining_fraction * remaining_fraction * remaining_fraction * remaining_fraction = 1024 / 81) :=
by
  let initial_passengers := 64
  let remaining_fraction := 2 / 3
  have H1 : initial_passengers * remaining_fraction = 128 / 3 := sorry
  have H2 : (128 / 3) * remaining_fraction = 256 / 9 := sorry
  have H3 : (256 / 9) * remaining_fraction = 512 / 27 := sorry
  have H4 : (512 / 27) * remaining_fraction = 1024 / 81 := sorry
  exact H4

end passengers_remaining_after_fourth_stop_l74_74567


namespace card_draw_count_l74_74327

theorem card_draw_count : 
  let total_cards := 12
  let red_cards := 3
  let yellow_cards := 3
  let blue_cards := 3
  let green_cards := 3
  let total_ways := Nat.choose total_cards 3
  let invalid_same_color := 4 * Nat.choose 3 3
  let invalid_two_red := Nat.choose red_cards 2 * Nat.choose (total_cards - red_cards) 1
  total_ways - invalid_same_color - invalid_two_red = 189 :=
by
  sorry

end card_draw_count_l74_74327


namespace base8_problem_l74_74741

/--
Let A, B, and C be non-zero and distinct digits in base 8 such that
ABC_8 + BCA_8 + CAB_8 = AAA0_8 and A + B = 2C.
Prove that B + C = 14 in base 8.
-/
theorem base8_problem (A B C : ℕ) 
    (h1 : A > 0 ∧ B > 0 ∧ C > 0)
    (h2 : A < 8 ∧ B < 8 ∧ C < 8)
    (h3 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
    (bcd_sum : (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) 
        = 8^3 * A + 8^2 * A + 8 * A)
    (sum_condition : A + B = 2 * C) :
    B + C = A + B := by {
  sorry
}

end base8_problem_l74_74741


namespace birth_date_16_Jan_1993_l74_74390

noncomputable def year_of_birth (current_date : Nat) (age_years : Nat) :=
  current_date - age_years * 365

noncomputable def month_of_birth (current_date : Nat) (age_years : Nat) (age_months : Nat) :=
  current_date - (age_years * 12 + age_months) * 30

theorem birth_date_16_Jan_1993 :
  let boy_age_years := 10
  let boy_age_months := 1
  let current_date := 16 + 31 * 12 * 2003 -- 16th February 2003 represented in days
  let full_months_lived := boy_age_years * 12 + boy_age_months
  full_months_lived - boy_age_years = 111 → 
  year_of_birth current_date boy_age_years = 1993 ∧ month_of_birth current_date boy_age_years boy_age_months = 31 * 1 * 1993 := 
sorry

end birth_date_16_Jan_1993_l74_74390


namespace problem_statement_l74_74782

theorem problem_statement 
  (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 2 := 
    sorry

end problem_statement_l74_74782


namespace tickets_used_correct_l74_74769

def ferris_wheel_rides : ℕ := 7
def bumper_car_rides : ℕ := 3
def cost_per_ride : ℕ := 5

def total_rides : ℕ := ferris_wheel_rides + bumper_car_rides
def total_tickets_used : ℕ := total_rides * cost_per_ride

theorem tickets_used_correct : total_tickets_used = 50 := by
  sorry

end tickets_used_correct_l74_74769


namespace sufficient_not_necessary_l74_74442

theorem sufficient_not_necessary (b c: ℝ) : (c < 0) → ∃ x y : ℝ, x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0 :=
by
  sorry

end sufficient_not_necessary_l74_74442


namespace problem_statement_l74_74197

theorem problem_statement (a b c : ℕ) (h1 : a < 12) (h2 : b < 12) (h3 : c < 12) (h4 : b + c = 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b + c :=
by
  sorry

end problem_statement_l74_74197


namespace mixed_fruit_juice_litres_opened_l74_74996

theorem mixed_fruit_juice_litres_opened (cocktail_cost_per_litre : ℝ)
  (mixed_juice_cost_per_litre : ℝ) (acai_cost_per_litre : ℝ)
  (acai_litres_added : ℝ) (total_mixed_juice_opened : ℝ) :
  cocktail_cost_per_litre = 1399.45 ∧
  mixed_juice_cost_per_litre = 262.85 ∧
  acai_cost_per_litre = 3104.35 ∧
  acai_litres_added = 23.333333333333336 ∧
  (mixed_juice_cost_per_litre * total_mixed_juice_opened + 
  acai_cost_per_litre * acai_litres_added = 
  cocktail_cost_per_litre * (total_mixed_juice_opened + acai_litres_added)) →
  total_mixed_juice_opened = 35 :=
sorry

end mixed_fruit_juice_litres_opened_l74_74996


namespace number_of_different_ways_is_18_l74_74845

-- Define the problem conditions
def number_of_ways_to_place_balls : ℕ :=
  let total_balls := 9
  let boxes := 3
  -- Placeholder function to compute the requirement
  -- The actual function would involve combinatorial logic
  -- Let us define it as an axiom for now.
  sorry

-- The theorem to be proven
theorem number_of_different_ways_is_18 :
  number_of_ways_to_place_balls = 18 :=
sorry

end number_of_different_ways_is_18_l74_74845


namespace Mildred_final_oranges_l74_74916

def initial_oranges : ℕ := 215
def father_oranges : ℕ := 3 * initial_oranges
def total_after_father : ℕ := initial_oranges + father_oranges
def sister_takes_away : ℕ := 174
def after_sister : ℕ := total_after_father - sister_takes_away
def final_oranges : ℕ := 2 * after_sister

theorem Mildred_final_oranges : final_oranges = 1372 := by
  sorry

end Mildred_final_oranges_l74_74916


namespace expression_evaluates_at_1_l74_74745

variable (x : ℚ)

def original_expr (x : ℚ) : ℚ := (x + 2) / (x - 3)

def substituted_expr (x : ℚ) : ℚ :=
  (original_expr (original_expr x) + 2) / (original_expr (original_expr x) - 3)

theorem expression_evaluates_at_1 :
  substituted_expr 1 = -1 / 9 :=
by
  sorry

end expression_evaluates_at_1_l74_74745


namespace smallest_pos_int_for_congruence_l74_74073

theorem smallest_pos_int_for_congruence :
  ∃ (n : ℕ), 5 * n % 33 = 980 % 33 ∧ n > 0 ∧ n = 19 := 
by {
  sorry
}

end smallest_pos_int_for_congruence_l74_74073


namespace diana_can_paint_statues_l74_74076

theorem diana_can_paint_statues : (3 / 6) / (1 / 6) = 3 := 
by 
  sorry

end diana_can_paint_statues_l74_74076


namespace problem_statement_l74_74691

theorem problem_statement (p : ℕ) (hprime : Prime p) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ p = m^2 + n^2 ∧ p ∣ (m^3 + n^3 + 8 * m * n)) → p = 5 :=
by
  sorry

end problem_statement_l74_74691


namespace problem_1_problem_2_problem_3_problem_4_l74_74215

theorem problem_1 : 42.67 - (12.67 - 2.87) = 32.87 :=
by sorry

theorem problem_2 : (4.8 - 4.8 * (3.2 - 2.7)) / 0.24 = 10 :=
by sorry

theorem problem_3 : 4.31 * 0.57 + 0.43 * 4.31 - 4.31 = 0 :=
by sorry

theorem problem_4 : 9.99 * 222 + 3.33 * 334 = 3330 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l74_74215


namespace three_integers_desc_order_l74_74843

theorem three_integers_desc_order (a b c : ℤ) : ∃ a' b' c' : ℤ, 
  (a = a' ∨ a = b' ∨ a = c') ∧
  (b = a' ∨ b = b' ∨ b = c') ∧
  (c = a' ∨ c = b' ∨ c = c') ∧ 
  (a' ≠ b' ∨ a' ≠ c' ∨ b' ≠ c') ∧
  a' ≥ b' ∧ b' ≥ c' :=
sorry

end three_integers_desc_order_l74_74843


namespace inequality_solution_l74_74518

theorem inequality_solution (x : ℝ) (h : x > -4/3) : 2 - 1 / (3 * x + 4) < 5 :=
sorry

end inequality_solution_l74_74518


namespace find_fraction_l74_74103

theorem find_fraction (N : ℕ) (hN : N = 90) (f : ℚ)
  (h : 3 + (1/2) * f * (1/5) * N = (1/15) * N) :
  f = 1/3 :=
by {
  sorry
}

end find_fraction_l74_74103


namespace payment_relationship_l74_74290

noncomputable def payment_amount (x : ℕ) (price_per_book : ℕ) (discount_percent : ℕ) : ℕ :=
  if x > 20 then ((x - 20) * (price_per_book * (100 - discount_percent) / 100) + 20 * price_per_book) else x * price_per_book

theorem payment_relationship (x : ℕ) (h : x > 20) : payment_amount x 25 20 = 20 * x + 100 := 
by
  sorry

end payment_relationship_l74_74290


namespace kabadi_kho_kho_players_l74_74027

theorem kabadi_kho_kho_players (total_players kabadi_only kho_kho_only both_games : ℕ)
  (h1 : kabadi_only = 10)
  (h2 : kho_kho_only = 40)
  (h3 : total_players = 50)
  (h4 : kabadi_only + kho_kho_only - both_games = total_players) :
  both_games = 0 := by
  sorry

end kabadi_kho_kho_players_l74_74027


namespace find_a3_l74_74511

theorem find_a3 (a : ℝ) (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) 
    (h1 : (1 + x) * (a - x)^6 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7)
    (h2 : a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = 0) :
  a = 1 → a3 = -5 := 
by 
  sorry

end find_a3_l74_74511


namespace ratio_of_areas_l74_74597

theorem ratio_of_areas (h a b R : ℝ) (h_triangle : a^2 + b^2 = h^2) (h_circumradius : R = h / 2) :
  (π * R^2) / (1/2 * a * b) = π * h / (4 * R) :=
by sorry

end ratio_of_areas_l74_74597


namespace carlos_initial_blocks_l74_74810

theorem carlos_initial_blocks (g : ℕ) (l : ℕ) (total : ℕ) : g = 21 → l = 37 → total = g + l → total = 58 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end carlos_initial_blocks_l74_74810


namespace quadratic_intersects_x_axis_l74_74732

theorem quadratic_intersects_x_axis (a : ℝ) (h : a ≠ 0) :
  ∃ x : ℝ, a * x^2 - (3 * a + 1) * x + 3 = 0 := 
by {
  -- The proof will go here
  sorry
}

end quadratic_intersects_x_axis_l74_74732


namespace tiffany_initial_lives_l74_74906

variable (x : ℝ) -- Define the variable x representing the initial number of lives

-- Define the conditions
def condition1 : Prop := x + 14.0 + 27.0 = 84.0

-- Prove the initial number of lives
theorem tiffany_initial_lives (h : condition1 x) : x = 43.0 := by
  sorry

end tiffany_initial_lives_l74_74906


namespace novel_to_history_ratio_l74_74931

-- Define the conditions
def history_book_pages : ℕ := 300
def science_book_pages : ℕ := 600
def novel_pages := science_book_pages / 4

-- Define the target ratio to prove
def target_ratio := (novel_pages : ℚ) / (history_book_pages : ℚ)

theorem novel_to_history_ratio :
  target_ratio = (1 : ℚ) / (2 : ℚ) :=
by
  sorry

end novel_to_history_ratio_l74_74931


namespace angle_C_measurement_l74_74498

variables (A B C : ℝ)

theorem angle_C_measurement
  (h1 : A + C = 2 * B)
  (h2 : C - A = 80)
  (h3 : A + B + C = 180) :
  C = 100 :=
by sorry

end angle_C_measurement_l74_74498


namespace locus_of_P_is_parabola_slopes_form_arithmetic_sequence_l74_74681

/-- Given a circle with center at point P passes through point A (1,0) 
    and is tangent to the line x = -1, the locus of point P is the parabola C. -/
theorem locus_of_P_is_parabola (P A : ℝ × ℝ) (x y : ℝ):
  (A = (1, 0)) → (P.1 + 1)^2 + P.2^2 = 0 → y^2 = 4 * x := 
sorry

/-- If the line passing through point H(4, 0) intersects the parabola 
    C (denoted by y^2 = 4x) at points M and N, and T is any point on 
    the line x = -4, then the slopes of lines TM, TH, and TN form an 
    arithmetic sequence. -/
theorem slopes_form_arithmetic_sequence (H M N T : ℝ × ℝ) (m n k : ℝ): 
  (H = (4, 0)) → (T.1 = -4) → 
  (M.1, M.2) = (k^2, 4*k) ∧ (N.1, N.2) = (m^2, 4*m) → 
  ((T.2 - M.2) / (T.1 - M.1) + (T.2 - N.2) / (T.1 - N.1)) = 
  2 * (T.2 / -8) := 
sorry

end locus_of_P_is_parabola_slopes_form_arithmetic_sequence_l74_74681


namespace value_of_x_l74_74415

theorem value_of_x :
  ∀ (x : ℕ), 
    x = 225 + 2 * 15 * 9 + 81 → 
    x = 576 := 
by
  intro x h
  sorry

end value_of_x_l74_74415


namespace f_is_odd_f_is_decreasing_range_of_m_l74_74228

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

-- Prove that f(x) is an odd function
theorem f_is_odd (x : ℝ) : f (-x) = - f x := by
  sorry

-- Prove that f(x) is decreasing on ℝ
theorem f_is_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := by
  sorry

-- Prove the range of m if f(m-1) + f(2m-1) > 0
theorem range_of_m (m : ℝ) (h : f (m - 1) + f (2 * m - 1) > 0) : m < 2 / 3 := by
  sorry

end f_is_odd_f_is_decreasing_range_of_m_l74_74228


namespace cost_of_items_l74_74539

theorem cost_of_items (x : ℝ) (cost_caramel_apple cost_ice_cream_cone : ℝ) :
  3 * cost_caramel_apple + 4 * cost_ice_cream_cone = 2 ∧
  cost_caramel_apple = cost_ice_cream_cone + 0.25 →
  cost_ice_cream_cone = 0.17857 ∧ cost_caramel_apple = 0.42857 :=
sorry

end cost_of_items_l74_74539


namespace probability_different_colors_l74_74918

/-- There are 5 blue chips and 3 yellow chips in a bag. One chip is drawn from the bag and placed
back into the bag. A second chip is then drawn. Prove that the probability of the two selected chips
being of different colors is 15/32. -/
theorem probability_different_colors : 
  let total_chips := 8
  let blue_chips := 5
  let yellow_chips := 3
  let prob_blue_then_yellow := (blue_chips/total_chips) * (yellow_chips/total_chips)
  let prob_yellow_then_blue := (yellow_chips/total_chips) * (blue_chips/total_chips)
  prob_blue_then_yellow + prob_yellow_then_blue = 15/32 := by
  sorry

end probability_different_colors_l74_74918


namespace three_a_greater_three_b_l74_74829

variable (a b : ℝ)

theorem three_a_greater_three_b (h : a > b) : 3 * a > 3 * b :=
  sorry

end three_a_greater_three_b_l74_74829


namespace max_distance_without_fuel_depots_l74_74014

def exploration_max_distance : ℕ :=
  360

-- Define the conditions
def cars_count : ℕ :=
  9

def full_tank_distance : ℕ :=
  40

def additional_gal_capacity : ℕ :=
  9

def total_gallons_per_car : ℕ :=
  1 + additional_gal_capacity

-- Define the distance calculation under the given constraints
theorem max_distance_without_fuel_depots (n : ℕ) (d_tank : ℕ) (d_add : ℕ) :
  ∀ (cars : ℕ), (cars = cars_count) →
  (d_tank = full_tank_distance) →
  (d_add = additional_gal_capacity) →
  ((cars * (1 + d_add)) * d_tank) / (2 * cars - 1) = exploration_max_distance :=
by
  intros _ hc ht ha
  rw [hc, ht, ha]
  -- Proof skipped
  sorry

end max_distance_without_fuel_depots_l74_74014


namespace trigonometric_problem_l74_74967

theorem trigonometric_problem (θ : ℝ) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 := by
  sorry

end trigonometric_problem_l74_74967


namespace sqrt_defined_range_l74_74682

theorem sqrt_defined_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 2)) → (x ≥ 2) := by
  sorry

end sqrt_defined_range_l74_74682


namespace race_length_l74_74669

theorem race_length
  (B_s : ℕ := 50) -- Biff's speed in yards per minute
  (K_s : ℕ := 51) -- Kenneth's speed in yards per minute
  (D_above_finish : ℕ := 10) -- distance Kenneth is past the finish line when Biff finishes
  : {L : ℕ // L = 500} := -- the length of the race is 500 yards.
  sorry

end race_length_l74_74669


namespace divisor_greater_than_2_l74_74120

theorem divisor_greater_than_2 (w n d : ℕ) (h1 : ∃ q1 : ℕ, w = d * q1 + 2)
                                       (h2 : n % 8 = 5)
                                       (h3 : n < 180) : 2 < d :=
sorry

end divisor_greater_than_2_l74_74120


namespace tv_price_reduction_percentage_l74_74482

noncomputable def price_reduction (x : ℝ) : Prop :=
  (1 - x / 100) * 1.80 = 1.44000000000000014

theorem tv_price_reduction_percentage : price_reduction 20 :=
by
  sorry

end tv_price_reduction_percentage_l74_74482


namespace sequence_a_10_l74_74046

theorem sequence_a_10 : ∀ {a : ℕ → ℕ}, (a 1 = 1) → (∀ n, a (n+1) = a n + 2^n) → (a 10 = 1023) :=
by
  intros a h1 h_rec
  sorry

end sequence_a_10_l74_74046


namespace bus_ride_cost_l74_74222

/-- The cost of a bus ride from town P to town Q, given that the cost of a train ride is $2.35 more 
    than a bus ride, and the combined cost of one train ride and one bus ride is $9.85. -/
theorem bus_ride_cost (B : ℝ) (h1 : ∃T, T = B + 2.35) (h2 : ∃T, T + B = 9.85) : B = 3.75 :=
by
  obtain ⟨T1, hT1⟩ := h1
  obtain ⟨T2, hT2⟩ := h2
  simp only [hT1, add_right_inj] at hT2
  sorry

end bus_ride_cost_l74_74222


namespace johns_weight_l74_74507

theorem johns_weight (j m : ℝ) (h1 : j + m = 240) (h2 : j - m = j / 3) : j = 144 :=
by
  sorry

end johns_weight_l74_74507


namespace ratio_right_to_left_l74_74129

theorem ratio_right_to_left (L C R : ℕ) (hL : L = 12) (hC : C = L + 2) (hTotal : L + C + R = 50) :
  R / L = 2 :=
by
  sorry

end ratio_right_to_left_l74_74129


namespace same_terminal_side_l74_74661

theorem same_terminal_side : ∃ k : ℤ, k * 360 - 60 = 300 := by
  sorry

end same_terminal_side_l74_74661


namespace front_wheel_revolutions_l74_74473

theorem front_wheel_revolutions (P_front P_back : ℕ) (R_back : ℕ) (H1 : P_front = 30) (H2 : P_back = 20) (H3 : R_back = 360) :
  ∃ F : ℕ, F = 240 := by
  sorry

end front_wheel_revolutions_l74_74473


namespace find_n_l74_74544

theorem find_n (n : ℕ) (h : 4 ^ 6 = 8 ^ n) : n = 4 :=
by
  sorry

end find_n_l74_74544


namespace converse_proposition_converse_proposition_true_l74_74634

theorem converse_proposition (x : ℝ) (h : x > 0) : x^2 - 1 > 0 :=
by sorry

theorem converse_proposition_true (x : ℝ) (h : x^2 - 1 > 0) : x > 0 :=
by sorry

end converse_proposition_converse_proposition_true_l74_74634


namespace solve_equation_l74_74984

theorem solve_equation (x : ℝ) (h : x ≠ 1) : 
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) → 
  x = -4 ∨ x = -2 :=
by 
  sorry

end solve_equation_l74_74984


namespace janet_total_pills_l74_74605

-- Define number of days per week
def days_per_week : ℕ := 7

-- Define pills per day for each week
def pills_first_2_weeks :=
  let multivitamins := 2 * days_per_week * 2
  let calcium := 3 * days_per_week * 2
  let magnesium := 5 * days_per_week * 2
  multivitamins + calcium + magnesium

def pills_third_week :=
  let multivitamins := 2 * days_per_week
  let calcium := 1 * days_per_week
  let magnesium := 0 * days_per_week
  multivitamins + calcium + magnesium

def pills_fourth_week :=
  let multivitamins := 3 * days_per_week
  let calcium := 2 * days_per_week
  let magnesium := 3 * days_per_week
  multivitamins + calcium + magnesium

def total_pills := pills_first_2_weeks + pills_third_week + pills_fourth_week

theorem janet_total_pills : total_pills = 245 := by
  -- Lean will generate a proof goal here with the left-hand side of the equation
  -- equal to an evaluated term, and we say that this equals 245 based on the problem's solution.
  sorry

end janet_total_pills_l74_74605


namespace total_acres_cleaned_l74_74474

theorem total_acres_cleaned (A D : ℕ) (h1 : (D - 1) * 90 + 30 = A) (h2 : D * 80 = A) : A = 480 :=
sorry

end total_acres_cleaned_l74_74474


namespace work_completion_l74_74364

theorem work_completion (A B C : ℝ) (h₁ : A + B = 1 / 18) (h₂ : B + C = 1 / 24) (h₃ : A + C = 1 / 36) : 
  1 / (A + B + C) = 16 := 
by
  sorry

end work_completion_l74_74364


namespace find_c_l74_74520

theorem find_c (a b c : ℝ) : 
  (a * x^2 + b * x - 5) * (a * x^2 + b * x + 25) + c = (a * x^2 + b * x + 10)^2 → 
  c = 225 :=
by sorry

end find_c_l74_74520


namespace jihye_wallet_total_l74_74166

-- Declare the amounts
def notes_amount : Nat := 2 * 1000
def coins_amount : Nat := 560

-- Theorem statement asserting the total amount
theorem jihye_wallet_total : notes_amount + coins_amount = 2560 := by
  sorry

end jihye_wallet_total_l74_74166


namespace find_a_l74_74777

theorem find_a (a : ℝ) (h : a ≠ 0) :
  (∀ x, -1 ≤ x ∧ x ≤ 4 → ax - a + 2 ≤ 7) →
  (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ ax - a + 2 = 7) →
  (a = 5/3 ∨ a = -5/2) :=
by
  sorry

end find_a_l74_74777


namespace digit_sum_equality_l74_74787

-- Definitions for the conditions
def is_permutation_of_digits (a b : ℕ) : Prop :=
  -- Assume implementation that checks if b is a permutation of the digits of a
  sorry

def sum_of_digits (n : ℕ) : ℕ :=
  -- Assume implementation that computes the sum of digits of n
  sorry

-- The theorem statement
theorem digit_sum_equality (a b : ℕ)
  (h : is_permutation_of_digits a b) :
  sum_of_digits (5 * a) = sum_of_digits (5 * b) :=
sorry

end digit_sum_equality_l74_74787


namespace a_perp_a_add_b_l74_74024

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

end a_perp_a_add_b_l74_74024
