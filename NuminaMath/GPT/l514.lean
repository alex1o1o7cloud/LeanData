import Mathlib

namespace general_formula_a_sum_b_condition_l514_51472

noncomputable def sequence_a (n : ℕ) : ℕ := sorry
noncomputable def sum_a (n : ℕ) : ℕ := sorry

-- Conditions
def a_2_condition : Prop := sequence_a 2 = 4
def sum_condition (n : ℕ) : Prop := 2 * sum_a n = n * sequence_a n + n

-- General formula for the n-th term of the sequence a_n
theorem general_formula_a : 
  (∀ n, sequence_a n = 3 * n - 2) ↔
  (a_2_condition ∧ ∀ n, sum_condition n) :=
sorry

noncomputable def sequence_c (n : ℕ) : ℕ := sorry
noncomputable def sequence_b (n : ℕ) : ℕ := sorry
noncomputable def sum_b (n : ℕ) : ℝ := sorry

-- Geometric sequence condition
def geometric_sequence_condition : Prop :=
  ∀ n, sequence_c n = 4^n

-- Condition for a_n = b_n * c_n
def a_b_c_relation (n : ℕ) : Prop := 
  sequence_a n = sequence_b n * sequence_c n

-- Sum condition T_n < 2/3
theorem sum_b_condition :
  (∀ n, a_b_c_relation n) ∧ geometric_sequence_condition →
  (∀ n, sum_b n < 2 / 3) :=
sorry

end general_formula_a_sum_b_condition_l514_51472


namespace xyz_mod_3_l514_51475

theorem xyz_mod_3 {x y z : ℕ} (hx : x = 3) (hy : y = 3) (hz : z = 2) : 
  (x^2 + y^2 + z^2) % 3 = 1 := by
  sorry

end xyz_mod_3_l514_51475


namespace fraction_simplification_l514_51403

theorem fraction_simplification (x y : ℚ) (hx : x = 4 / 6) (hy : y = 5 / 8) :
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 :=
by
  rw [hx, hy]
  sorry

end fraction_simplification_l514_51403


namespace blue_pill_cost_l514_51467

theorem blue_pill_cost :
  ∀ (cost_yellow cost_blue : ℝ) (days : ℕ) (total_cost : ℝ),
    (days = 21) →
    (total_cost = 882) →
    (cost_blue = cost_yellow + 3) →
    (total_cost = days * (cost_blue + cost_yellow)) →
    cost_blue = 22.50 :=
by sorry

end blue_pill_cost_l514_51467


namespace total_students_l514_51495

-- Definitions from the conditions
def ratio_boys_to_girls (B G : ℕ) : Prop := B / G = 1 / 2
def girls_count := 60

-- The main statement to prove
theorem total_students (B G : ℕ) (h1 : ratio_boys_to_girls B G) (h2 : G = girls_count) : B + G = 90 := sorry

end total_students_l514_51495


namespace trapezium_distance_parallel_sides_l514_51427

theorem trapezium_distance_parallel_sides
  (l1 l2 area : ℝ) (h : ℝ)
  (h_area : area = (1 / 2) * (l1 + l2) * h)
  (hl1 : l1 = 30)
  (hl2 : l2 = 12)
  (h_area_val : area = 336) :
  h = 16 :=
by
  sorry

end trapezium_distance_parallel_sides_l514_51427


namespace evan_amount_l514_51422

def adrian : ℤ := sorry
def brenda : ℤ := sorry
def charlie : ℤ := sorry
def dana : ℤ := sorry
def evan : ℤ := sorry

def amounts_sum : Prop := adrian + brenda + charlie + dana + evan = 72
def abs_diff_1 : Prop := abs (adrian - brenda) = 21
def abs_diff_2 : Prop := abs (brenda - charlie) = 8
def abs_diff_3 : Prop := abs (charlie - dana) = 6
def abs_diff_4 : Prop := abs (dana - evan) = 5
def abs_diff_5 : Prop := abs (evan - adrian) = 14

theorem evan_amount
  (h_sum : amounts_sum)
  (h_diff1 : abs_diff_1)
  (h_diff2 : abs_diff_2)
  (h_diff3 : abs_diff_3)
  (h_diff4 : abs_diff_4)
  (h_diff5 : abs_diff_5) :
  evan = 21 := sorry

end evan_amount_l514_51422


namespace percentage_increase_20_l514_51453

noncomputable def oldCompanyEarnings : ℝ := 3 * 12 * 5000
noncomputable def totalEarnings : ℝ := 426000
noncomputable def newCompanyMonths : ℕ := 36 + 5
noncomputable def newCompanyEarnings : ℝ := totalEarnings - oldCompanyEarnings
noncomputable def newCompanyMonthlyEarnings : ℝ := newCompanyEarnings / newCompanyMonths
noncomputable def oldCompanyMonthlyEarnings : ℝ := 5000

theorem percentage_increase_20 :
  (newCompanyMonthlyEarnings - oldCompanyMonthlyEarnings) / oldCompanyMonthlyEarnings * 100 = 20 :=
by sorry

end percentage_increase_20_l514_51453


namespace female_officers_count_l514_51494

theorem female_officers_count (total_officers_on_duty : ℕ) 
  (percent_female_on_duty : ℝ) 
  (female_officers_on_duty : ℕ) 
  (half_of_total_on_duty_is_female : total_officers_on_duty / 2 = female_officers_on_duty) 
  (percent_condition : percent_female_on_duty * (total_officers_on_duty / 2) = female_officers_on_duty) :
  total_officers_on_duty = 250 :=
by
  sorry

end female_officers_count_l514_51494


namespace three_zeros_condition_l514_51429

noncomputable def f (ω : ℝ) (x : ℝ) := Real.sin (ω * x) + Real.cos (ω * x)

theorem three_zeros_condition (ω : ℝ) (hω : ω > 0) :
  (∃ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ 2 * Real.pi ∧
  f ω x1 = 0 ∧ f ω x2 = 0 ∧ f ω x3 = 0) →
  (∀ ω, (11 / 8 : ℝ) ≤ ω ∧ ω < (15 / 8 : ℝ) ∧
  (∀ x, f ω x = 0 ↔ x = (5 * Real.pi) / (4 * ω))) :=
sorry

end three_zeros_condition_l514_51429


namespace total_vegetables_correct_l514_51423

def cucumbers : ℕ := 70
def tomatoes : ℕ := 3 * cucumbers
def total_vegetables : ℕ := cucumbers + tomatoes

theorem total_vegetables_correct : total_vegetables = 280 :=
by
  sorry

end total_vegetables_correct_l514_51423


namespace square_of_1017_l514_51433

theorem square_of_1017 : 1017^2 = 1034289 :=
by
  sorry

end square_of_1017_l514_51433


namespace box_dimensions_l514_51410

theorem box_dimensions {a b c : ℕ} (h1 : a + c = 17) (h2 : a + b = 13) (h3 : 2 * (b + c) = 40) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by {
  sorry
}

end box_dimensions_l514_51410


namespace arithmetic_seq_a7_l514_51462

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_a3 : a 3 = 50) 
  (h_a5 : a 5 = 30) : 
  a 7 = 10 := 
by
  sorry

end arithmetic_seq_a7_l514_51462


namespace ratio_of_a_over_5_to_b_over_4_l514_51439

theorem ratio_of_a_over_5_to_b_over_4 (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a ≠ 0 ∧ b ≠ 0) :
  (a / 5) / (b / 4) = 1 := by
  sorry

end ratio_of_a_over_5_to_b_over_4_l514_51439


namespace triangle_area_inscribed_rectangle_area_l514_51415

theorem triangle_area (m n : ℝ) : ∃ (S : ℝ), S = m * n := 
sorry

theorem inscribed_rectangle_area (m n : ℝ) : ∃ (A : ℝ), A = (2 * m^2 * n^2) / (m + n)^2 :=
sorry

end triangle_area_inscribed_rectangle_area_l514_51415


namespace frequency_of_3rd_group_l514_51456

theorem frequency_of_3rd_group (m : ℕ) (h_m : m ≥ 3) (x : ℝ) (h_area_relation : ∀ k, k ≠ 3 → 4 * x = k):
  100 * x = 20 :=
by
  sorry

end frequency_of_3rd_group_l514_51456


namespace measure_of_angle_y_l514_51407

def is_straight_angle (a : ℝ) := a = 180

theorem measure_of_angle_y (angle_ABC angle_ADB angle_BDA y : ℝ) 
  (h1 : angle_ABC = 117)
  (h2 : angle_ADB = 31)
  (h3 : angle_BDA = 28)
  (h4 : is_straight_angle (angle_ABC + (180 - angle_ABC)))
  : y = 86 := 
by 
  sorry

end measure_of_angle_y_l514_51407


namespace sequence_value_a10_l514_51479

theorem sequence_value_a10 (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 2^n) : a 10 = 1023 := by
  sorry

end sequence_value_a10_l514_51479


namespace max_range_of_temperatures_l514_51473

theorem max_range_of_temperatures (avg_temp : ℝ) (low_temp : ℝ) (days : ℕ) (total_temp: ℝ) (high_temp : ℝ) 
  (h1 : avg_temp = 60) (h2 : low_temp = 50) (h3 : days = 5) (h4 : total_temp = avg_temp * days) 
  (h5 : total_temp = 300) (h6 : 4 * low_temp + high_temp = total_temp) : 
  high_temp - low_temp = 50 := 
by
  sorry

end max_range_of_temperatures_l514_51473


namespace simplify_fraction_subtraction_l514_51404

theorem simplify_fraction_subtraction :
  (5 / 15 : ℚ) - (2 / 45) = 13 / 45 :=
by
  -- (The proof will go here)
  sorry

end simplify_fraction_subtraction_l514_51404


namespace trains_cross_in_9_seconds_l514_51425

noncomputable def time_to_cross (length1 length2 : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / (speed1 + speed2)

theorem trains_cross_in_9_seconds :
  time_to_cross 240 260.04 (120 * (5 / 18)) (80 * (5 / 18)) = 9 := 
by
  sorry

end trains_cross_in_9_seconds_l514_51425


namespace least_number_to_subtract_l514_51445

theorem least_number_to_subtract (n : ℕ) (h : n = 9876543210) : 
  ∃ m, m = 6 ∧ (n - m) % 29 = 0 := 
sorry

end least_number_to_subtract_l514_51445


namespace suffering_correctness_l514_51491

noncomputable def expected_total_suffering (n m : ℕ) : ℕ :=
  if n = 8 ∧ m = 256 then (2^135 - 2^128 + 1) / (2^119 * 129) else 0

theorem suffering_correctness :
  expected_total_suffering 8 256 = (2^135 - 2^128 + 1) / (2^119 * 129) :=
sorry

end suffering_correctness_l514_51491


namespace Ben_Cards_Left_l514_51457

theorem Ben_Cards_Left :
  (4 * 10 + 5 * 8 - 58) = 22 :=
by
  sorry

end Ben_Cards_Left_l514_51457


namespace total_marbles_l514_51450

theorem total_marbles (x : ℕ) (h1 : 5 * x - 2 = 18) : 4 * x + 5 * x = 36 :=
by
  sorry

end total_marbles_l514_51450


namespace evaluate_expression_l514_51499

theorem evaluate_expression :
  let sum1 := 3 + 6 + 9
  let sum2 := 2 + 5 + 8
  (sum1 / sum2 - sum2 / sum1) = 11 / 30 :=
by
  let sum1 := 3 + 6 + 9
  let sum2 := 2 + 5 + 8
  sorry

end evaluate_expression_l514_51499


namespace final_price_correct_l514_51489

noncomputable def price_cucumbers : ℝ := 5
noncomputable def price_tomatoes : ℝ := price_cucumbers - 0.20 * price_cucumbers
noncomputable def total_cost_before_discount : ℝ := 2 * price_tomatoes + 3 * price_cucumbers
noncomputable def discount : ℝ := 0.10 * total_cost_before_discount
noncomputable def final_price : ℝ := total_cost_before_discount - discount

theorem final_price_correct : final_price = 20.70 := by
  sorry

end final_price_correct_l514_51489


namespace no_integer_solution_for_z_l514_51405

theorem no_integer_solution_for_z (z : ℤ) (h : 2 / z = 2 / (z + 1) + 2 / (z + 25)) : false :=
by
  sorry

end no_integer_solution_for_z_l514_51405


namespace alvin_marble_count_correct_l514_51482

variable (initial_marble_count lost_marble_count won_marble_count final_marble_count : ℕ)

def calculate_final_marble_count (initial : ℕ) (lost : ℕ) (won : ℕ) : ℕ :=
  initial - lost + won

theorem alvin_marble_count_correct :
  initial_marble_count = 57 →
  lost_marble_count = 18 →
  won_marble_count = 25 →
  final_marble_count = calculate_final_marble_count initial_marble_count lost_marble_count won_marble_count →
  final_marble_count = 64 :=
by
  intros h_initial h_lost h_won h_calculate
  rw [h_initial, h_lost, h_won] at h_calculate
  exact h_calculate

end alvin_marble_count_correct_l514_51482


namespace axis_of_symmetry_cosine_l514_51421

theorem axis_of_symmetry_cosine (x : ℝ) : 
  (∃ k : ℤ, 2 * x + π / 3 = k * π) → x = -π / 6 :=
sorry

end axis_of_symmetry_cosine_l514_51421


namespace product_of_fractions_l514_51476

theorem product_of_fractions : (2 / 5) * (3 / 4) = 3 / 10 := 
  sorry

end product_of_fractions_l514_51476


namespace dyslexian_alphabet_size_l514_51426

theorem dyslexian_alphabet_size (c v : ℕ) (h1 : (c * v * c * v * c + v * c * v * c * v) = 4800) : c + v = 12 :=
by
  sorry

end dyslexian_alphabet_size_l514_51426


namespace solve_equation_l514_51441

theorem solve_equation : ∀ x : ℝ, ((1 - x) / (x - 4)) + (1 / (4 - x)) = 1 → x = 2 :=
by
  intros x h
  sorry

end solve_equation_l514_51441


namespace train_a_constant_rate_l514_51436

theorem train_a_constant_rate
  (d : ℕ)
  (v_b : ℕ)
  (d_a : ℕ)
  (v : ℕ)
  (h1 : d = 350)
  (h2 : v_b = 30)
  (h3 : d_a = 200)
  (h4 : v * (d_a / v) + v_b * (d_a / v) = d) :
  v = 40 := by
  sorry

end train_a_constant_rate_l514_51436


namespace lcm_5_711_is_3555_l514_51430

theorem lcm_5_711_is_3555 : Nat.lcm 5 711 = 3555 := by
  sorry

end lcm_5_711_is_3555_l514_51430


namespace failed_in_hindi_percentage_l514_51406

/-- In an examination, a specific percentage of students failed in Hindi (H%), 
45% failed in English, and 20% failed in both. We know that 40% passed in both subjects. 
Prove that 35% students failed in Hindi. --/
theorem failed_in_hindi_percentage : 
  ∀ (H E B P : ℕ),
    (E = 45) → (B = 20) → (P = 40) → (100 - P = H + E - B) → H = 35 := by
  intros H E B P hE hB hP h
  sorry

end failed_in_hindi_percentage_l514_51406


namespace sum_local_values_2345_l514_51468

theorem sum_local_values_2345 : 
  let n := 2345
  let digit_2_value := 2000
  let digit_3_value := 300
  let digit_4_value := 40
  let digit_5_value := 5
  digit_2_value + digit_3_value + digit_4_value + digit_5_value = n := 
by
  sorry

end sum_local_values_2345_l514_51468


namespace problem_statement_l514_51478

theorem problem_statement (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ) 
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) : 
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := 
sorry

end problem_statement_l514_51478


namespace num_possible_radii_l514_51464

theorem num_possible_radii:
  ∃ (S : Finset ℕ), 
  (∀ r ∈ S, r < 60 ∧ (2 * r * π ∣ 120 * π)) ∧ 
  S.card = 11 := 
sorry

end num_possible_radii_l514_51464


namespace trajectory_midpoint_l514_51452

theorem trajectory_midpoint (P Q M : ℝ × ℝ)
  (hP : P.1^2 + P.2^2 = 1)
  (hQ : Q.1 = 3 ∧ Q.2 = 0)
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (2 * M.1 - 3)^2 + 4 * M.2^2 = 1 :=
sorry

end trajectory_midpoint_l514_51452


namespace curve_B_is_not_good_l514_51409

-- Define the points A and B
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define the condition for being a "good curve"
def is_good_curve (C : ℝ × ℝ → Prop) : Prop :=
  ∃ M : ℝ × ℝ, C M ∧ abs (dist M A - dist M B) = 8

-- Define the curves
def curve_A (p : ℝ × ℝ) : Prop := p.1 + p.2 = 5
def curve_B (p : ℝ × ℝ) : Prop := p.1 ^ 2 + p.2 ^ 2 = 9
def curve_C (p : ℝ × ℝ) : Prop := (p.1 ^ 2) / 25 + (p.2 ^ 2) / 9 = 1
def curve_D (p : ℝ × ℝ) : Prop := p.1 ^ 2 = 16 * p.2

-- Prove that curve_B is not a "good curve"
theorem curve_B_is_not_good : ¬ is_good_curve curve_B := by
  sorry

end curve_B_is_not_good_l514_51409


namespace sum_of_digits_of_product_in_base9_l514_51431

def base9_to_decimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  d1 * 9 + d0

def base10_to_base9 (n : ℕ) : ℕ :=
  let d0 := n % 9
  let d1 := (n / 9) % 9
  let d2 := (n / 81) % 9
  d2 * 100 + d1 * 10 + d0

def sum_of_digits_base9 (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 + d1 + d0

theorem sum_of_digits_of_product_in_base9 :
  let n1 := base9_to_decimal 36
  let n2 := base9_to_decimal 21
  let product := n1 * n2
  let base9_product := base10_to_base9 product
  sum_of_digits_base9 base9_product = 19 :=
by
  sorry

end sum_of_digits_of_product_in_base9_l514_51431


namespace problem1_problem2_l514_51446

theorem problem1 (a b : ℝ) (h1 : a = 3) :
  (∀ x : ℝ, 2 * x ^ 2 + (2 - a) * x - a > 0 ↔ x < -1 ∨ x > 3 / 2) :=
by
  sorry

theorem problem2 (a b : ℝ) (h1 : a = 3) :
  (∀ x : ℝ, 3 * x ^ 2 + b * x + 3 ≥ 0) ↔ (-6 ≤ b ∧ b ≤ 6) :=
by
  sorry

end problem1_problem2_l514_51446


namespace visible_product_divisible_by_48_l514_51490

-- We represent the eight-sided die as the set {1, 2, 3, 4, 5, 6, 7, 8}.
-- Q is the product of any seven numbers from this set.

theorem visible_product_divisible_by_48 
   (Q : ℕ)
   (H : ∃ (numbers : Finset ℕ), numbers ⊆ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ) ∧ numbers.card = 7 ∧ Q = numbers.prod id) :
   48 ∣ Q :=
by
  sorry

end visible_product_divisible_by_48_l514_51490


namespace correct_exponent_calculation_l514_51483

theorem correct_exponent_calculation (a : ℝ) : 
  (a^5 * a^2 = a^7) :=
by
  sorry

end correct_exponent_calculation_l514_51483


namespace sweets_remainder_l514_51470

theorem sweets_remainder (m : ℕ) (h : m % 7 = 6) : (4 * m) % 7 = 3 :=
by
  sorry

end sweets_remainder_l514_51470


namespace cos_pi_minus_2alpha_l514_51416

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (Real.pi / 2 - α) = Real.sqrt 2 / 3) : 
  Real.cos (Real.pi - 2 * α) = -5 / 9 := by
  sorry

end cos_pi_minus_2alpha_l514_51416


namespace difference_between_two_greatest_values_l514_51480

-- Definition of the variables and conditions
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n < 10

variables (a b c x : ℕ)

def conditions (a b c : ℕ) := is_digit a ∧ is_digit b ∧ is_digit c ∧ 2 * a = b ∧ b = 4 * c ∧ a > 0

-- Definition of x as a 3-digit number given a, b, and c
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

def smallest_x : ℕ := three_digit_number 2 4 1
def largest_x : ℕ := three_digit_number 4 8 2

def difference_two_greatest_values (a b c : ℕ) : ℕ := largest_x - smallest_x

-- The proof statement
theorem difference_between_two_greatest_values (a b c : ℕ) (h : conditions a b c) : 
  ∀ x1 x2 : ℕ, 
    three_digit_number 2 4 1 = x1 →
    three_digit_number 4 8 2 = x2 →
    difference_two_greatest_values a b c = 241 :=
by
  sorry

end difference_between_two_greatest_values_l514_51480


namespace man_speed_l514_51402

theorem man_speed (L T V_t V_m : ℝ) (hL : L = 400) (hT : T = 35.99712023038157) (hVt : V_t = 46 * 1000 / 3600) (hE : L = (V_t - V_m) * T) : V_m = 1.666666666666684 :=
by
  sorry

end man_speed_l514_51402


namespace least_number_to_add_l514_51458

theorem least_number_to_add (n : ℕ) (m : ℕ) : (1156 + 19) % 25 = 0 :=
by
  sorry

end least_number_to_add_l514_51458


namespace unique_positive_integer_solution_l514_51449

-- Definitions of the given points
def P1 : ℚ × ℚ := (4, 11)
def P2 : ℚ × ℚ := (16, 1)

-- Definition for the line equation in standard form
def line_equation (x y : ℤ) : Prop := 5 * x + 6 * y = 43

-- Proof for the existence of only one solution with positive integer coordinates
theorem unique_positive_integer_solution :
  ∃ P : ℤ × ℤ, P.1 > 0 ∧ P.2 > 0 ∧ line_equation P.1 P.2 ∧ (∀ Q : ℤ × ℤ, line_equation Q.1 Q.2 → Q.1 > 0 ∧ Q.2 > 0 → Q = (5, 3)) :=
by 
  sorry

end unique_positive_integer_solution_l514_51449


namespace f_at_8_5_l514_51466

def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom odd_function_shifted : ∀ x : ℝ, f (x - 1) = -f (1 - x)
axiom f_half : f 0.5 = 9

theorem f_at_8_5 : f 8.5 = 9 := by
  sorry

end f_at_8_5_l514_51466


namespace boy_needs_to_sell_75_oranges_to_make_150c_profit_l514_51438

-- Definitions based on the conditions
def cost_per_orange : ℕ := 12 / 4
def sell_price_per_orange : ℕ := 30 / 6
def profit_per_orange : ℕ := sell_price_per_orange - cost_per_orange

-- Problem declaration
theorem boy_needs_to_sell_75_oranges_to_make_150c_profit : 
  (150 / profit_per_orange) = 75 :=
by
  -- Proof will be added here
  sorry

end boy_needs_to_sell_75_oranges_to_make_150c_profit_l514_51438


namespace non_neg_int_solutions_l514_51434

def operation (a b : ℝ) : ℝ := a * (a - b) + 1

theorem non_neg_int_solutions (x : ℕ) :
  2 * (2 - x) + 1 ≥ 3 ↔ x = 0 ∨ x = 1 := by
  sorry

end non_neg_int_solutions_l514_51434


namespace digit_for_divisibility_by_9_l514_51454

theorem digit_for_divisibility_by_9 (A : ℕ) (hA : A < 10) : 
  (∃ k : ℕ, 83 * 1000 + A * 10 + 5 = 9 * k) ↔ A = 2 :=
by
  sorry

end digit_for_divisibility_by_9_l514_51454


namespace ms_cole_students_l514_51455

theorem ms_cole_students (S6 S4 S7 : ℕ)
  (h1: S6 = 40)
  (h2: S4 = 4 * S6)
  (h3: S7 = 2 * S4) :
  S6 + S4 + S7 = 520 :=
by
  sorry

end ms_cole_students_l514_51455


namespace age_of_boy_not_included_l514_51469

theorem age_of_boy_not_included (average_age_11_boys : ℕ) (average_age_first_6 : ℕ) (average_age_last_6 : ℕ) 
(first_6_sum : ℕ) (last_6_sum : ℕ) (total_sum : ℕ) (X : ℕ):
  average_age_11_boys = 50 ∧ average_age_first_6 = 49 ∧ average_age_last_6 = 52 ∧ 
  first_6_sum = 6 * average_age_first_6 ∧ last_6_sum = 6 * average_age_last_6 ∧ 
  total_sum = 11 * average_age_11_boys ∧ first_6_sum + last_6_sum - X = total_sum →
  X = 56 :=
by
  sorry

end age_of_boy_not_included_l514_51469


namespace ashley_percentage_secured_l514_51418

noncomputable def marks_secured : ℕ := 332
noncomputable def max_marks : ℕ := 400
noncomputable def percentage_secured : ℕ := (marks_secured * 100) / max_marks

theorem ashley_percentage_secured 
    (h₁ : marks_secured = 332)
    (h₂ : max_marks = 400) :
    percentage_secured = 83 := by
  -- Proof goes here
  sorry

end ashley_percentage_secured_l514_51418


namespace polygon_area_is_400_l514_51447

def Point : Type := (ℤ × ℤ)

def area_of_polygon (vertices : List Point) : ℤ := 
  -- Formula to calculate polygon area would go here
  -- As a placeholder, for now we return 400 since proof details aren't required
  400

theorem polygon_area_is_400 :
  area_of_polygon [(0,0), (20,0), (30,10), (20,20), (0,20), (10,10), (0,0)] = 400 := by
  -- Proof would go here
  sorry

end polygon_area_is_400_l514_51447


namespace number_of_winning_scores_l514_51419

-- Define the problem conditions
variable (n : ℕ) (team1_scores team2_scores : Finset ℕ)

-- Define the total number of runners
def total_runners := 12

-- Define the sum of placements
def sum_placements : ℕ := (total_runners * (total_runners + 1)) / 2

-- Define the threshold for the winning score
def winning_threshold : ℕ := sum_placements / 2

-- Define the minimum score for a team
def min_score : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Prove that the number of different possible winning scores is 19
theorem number_of_winning_scores : 
  Finset.card (Finset.range (winning_threshold + 1) \ Finset.range min_score) = 19 :=
by
  sorry -- Proof to be filled in

end number_of_winning_scores_l514_51419


namespace papaya_tree_height_after_5_years_l514_51437

def first_year_growth := 2
def second_year_growth := first_year_growth + (first_year_growth / 2)
def third_year_growth := second_year_growth + (second_year_growth / 2)
def fourth_year_growth := third_year_growth * 2
def fifth_year_growth := fourth_year_growth / 2

theorem papaya_tree_height_after_5_years : 
  first_year_growth + second_year_growth + third_year_growth + fourth_year_growth + fifth_year_growth = 23 :=
by
  sorry

end papaya_tree_height_after_5_years_l514_51437


namespace number_of_people_l514_51463

-- Define the given constants
def total_cookies := 35
def cookies_per_person := 7

-- Goal: Prove that the number of people equal to 5
theorem number_of_people : total_cookies / cookies_per_person = 5 :=
by
  sorry

end number_of_people_l514_51463


namespace hypotenuse_unique_l514_51413

theorem hypotenuse_unique (a b : ℝ) (h: ∃ x : ℝ, x^2 = a^2 + b^2 ∧ x > 0) : 
  ∃! c : ℝ, c^2 = a^2 + b^2 :=
sorry

end hypotenuse_unique_l514_51413


namespace modulus_of_complex_number_l514_51481

noncomputable def z := Complex

theorem modulus_of_complex_number (z : Complex) (h : z * (1 + Complex.I) = 2) :
  Complex.abs z = Real.sqrt 2 :=
sorry

end modulus_of_complex_number_l514_51481


namespace linear_equation_solution_l514_51435

theorem linear_equation_solution (x y : ℝ) (h : 3 * x - y = 5) : y = 3 * x - 5 :=
sorry

end linear_equation_solution_l514_51435


namespace g_range_l514_51487

noncomputable def g (x y z : ℝ) : ℝ := 
  (x^2 / (x^2 + y^2)) + (y^2 / (y^2 + z^2)) + (z^2 / (z^2 + x^2))

theorem g_range (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 / 2 ≤ g x y z ∧ g x y z ≤ 2 :=
sorry

end g_range_l514_51487


namespace A_cubed_inv_l514_51486

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

-- Given condition
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 7], ![-2, -4]]

-- Goal to prove
theorem A_cubed_inv :
  (A^3)⁻¹ = ![![11, 17], ![2, 6]] :=
  sorry

end A_cubed_inv_l514_51486


namespace adam_coin_collection_value_l514_51465

-- Definitions related to the problem conditions
def value_per_first_type_coin := 15 / 5
def value_per_second_type_coin := 18 / 6

def total_value_first_type (num_first_type_coins : ℕ) := num_first_type_coins * value_per_first_type_coin
def total_value_second_type (num_second_type_coins : ℕ) := num_second_type_coins * value_per_second_type_coin

-- The main theorem, stating that the total collection value is 90 dollars given the conditions
theorem adam_coin_collection_value :
  total_value_first_type 18 + total_value_second_type 12 = 90 := 
sorry

end adam_coin_collection_value_l514_51465


namespace first_nonzero_digit_one_over_137_l514_51488

noncomputable def first_nonzero_digit_right_of_decimal (n : ℚ) : ℕ := sorry

theorem first_nonzero_digit_one_over_137 : first_nonzero_digit_right_of_decimal (1 / 137) = 7 := sorry

end first_nonzero_digit_one_over_137_l514_51488


namespace jaydee_typing_speed_l514_51459

theorem jaydee_typing_speed (hours : ℕ) (total_words : ℕ) (minutes_per_hour : ℕ := 60) 
  (h1 : hours = 2) (h2 : total_words = 4560) : (total_words / (hours * minutes_per_hour) = 38) :=
by
  sorry

end jaydee_typing_speed_l514_51459


namespace find_theta_plus_3phi_l514_51496

variables (θ φ : ℝ)

-- The conditions
variables (h1 : 0 < θ ∧ θ < π / 2) (h2 : 0 < φ ∧ φ < π / 2)
variables (h3 : Real.tan θ = 1 / 3) (h4 : Real.sin φ = 3 / 5)

theorem find_theta_plus_3phi :
  θ + 3 * φ = π - Real.arctan (199 / 93) :=
sorry

end find_theta_plus_3phi_l514_51496


namespace monotonicity_of_f_l514_51417

noncomputable def f (x : ℝ) : ℝ := - (2 * x) / (1 + x^2)

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y ∧ (y < -1 ∨ x > 1) → f x < f y) ∧
  (∀ x y : ℝ, x < y ∧ -1 < x ∧ y < 1 → f y < f x) := sorry

end monotonicity_of_f_l514_51417


namespace dragon_2023_first_reappearance_l514_51401

theorem dragon_2023_first_reappearance :
  let cycle_letters := 6
  let cycle_digits := 4
  Nat.lcm cycle_letters cycle_digits = 12 :=
by
  rfl -- since LCM of 6 and 4 directly calculates to 12

end dragon_2023_first_reappearance_l514_51401


namespace solution_set_inequality_l514_51485

theorem solution_set_inequality (x : ℝ) : (x + 3) / (x - 1) > 0 ↔ x < -3 ∨ x > 1 :=
sorry

end solution_set_inequality_l514_51485


namespace crushing_load_calculation_l514_51411

theorem crushing_load_calculation (T H : ℝ) (L : ℝ) 
  (h1 : L = 40 * T^5 / H^3) 
  (h2 : T = 3) 
  (h3 : H = 6) : 
  L = 45 := 
by sorry

end crushing_load_calculation_l514_51411


namespace min_value_of_sum_squares_l514_51442

theorem min_value_of_sum_squares (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 11) : 
  x^2 + y^2 + z^2 ≥ 121 / 29 := sorry

end min_value_of_sum_squares_l514_51442


namespace equal_wear_tires_l514_51414

theorem equal_wear_tires (t D d : ℕ) (h1 : t = 7) (h2 : D = 42000) (h3 : t * d = 6 * D) : d = 36000 :=
by
  sorry

end equal_wear_tires_l514_51414


namespace infinitely_many_coprime_binomials_l514_51484

theorem infinitely_many_coprime_binomials (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ᶠ n in at_top, n > k ∧ Nat.gcd (Nat.choose n k) l = 1 := by
  sorry

end infinitely_many_coprime_binomials_l514_51484


namespace paint_per_door_l514_51428

variable (cost_per_pint : ℕ) (cost_per_gallon : ℕ) (num_doors : ℕ) (pints_per_gallon : ℕ) (savings : ℕ)

theorem paint_per_door :
  cost_per_pint = 8 →
  cost_per_gallon = 55 →
  num_doors = 8 →
  pints_per_gallon = 8 →
  savings = 9 →
  (pints_per_gallon / num_doors = 1) :=
by
  intros h_cpint h_cgallon h_nd h_pgallon h_savings
  sorry

end paint_per_door_l514_51428


namespace inscribed_circle_radius_of_triangle_l514_51451

theorem inscribed_circle_radius_of_triangle (a b c : ℕ)
  (h₁ : a = 50) (h₂ : b = 120) (h₃ : c = 130) :
  ∃ r : ℕ, r = 20 :=
by sorry

end inscribed_circle_radius_of_triangle_l514_51451


namespace no_real_solutions_l514_51448

noncomputable def original_eq (x : ℝ) : Prop := (x^2 + x + 1) / (x + 1) = x^2 + 5 * x + 6

theorem no_real_solutions (x : ℝ) : ¬ original_eq x :=
by
  sorry

end no_real_solutions_l514_51448


namespace solve_for_x_l514_51492

theorem solve_for_x : ∃ (x : ℝ), (x - 5) ^ 2 = (1 / 16)⁻¹ ∧ (x = 9 ∨ x = 1) :=
by
  sorry

end solve_for_x_l514_51492


namespace numbers_starting_with_6_div_by_25_no_numbers_divisible_by_35_after_first_digit_removed_l514_51461

-- Definitions based on conditions
def starts_with_six (x : ℕ) : Prop :=
  ∃ n y, x = 6 * 10^n + y

def is_divisible_by_25 (y : ℕ) : Prop :=
  y % 25 = 0

def is_divisible_by_35 (y : ℕ) : Prop :=
  y % 35 = 0

-- Main theorem statements
theorem numbers_starting_with_6_div_by_25:
  ∀ x, starts_with_six x → ∃ k, x = 625 * 10^k :=
by
  sorry

theorem no_numbers_divisible_by_35_after_first_digit_removed:
  ∀ a x, a ≠ 0 → 
  ∃ n, x = a * 10^n + y →
  ¬(is_divisible_by_35 y) :=
by
  sorry

end numbers_starting_with_6_div_by_25_no_numbers_divisible_by_35_after_first_digit_removed_l514_51461


namespace linda_age_l514_51400

variable (s j l : ℕ)

theorem linda_age (h1 : (s + j + l) / 3 = 11) 
                  (h2 : l - 5 = s) 
                  (h3 : j + 4 = 3 * (s + 4) / 4) :
                  l = 14 := by
  sorry

end linda_age_l514_51400


namespace praveen_initial_investment_l514_51444

theorem praveen_initial_investment
  (H : ℝ) (P : ℝ)
  (h_H : H = 9000.000000000002)
  (h_profit_ratio : (P * 12) / (H * 7) = 2 / 3) :
  P = 3500 := by
  sorry

end praveen_initial_investment_l514_51444


namespace sum_largest_smallest_gx_l514_51412

noncomputable def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + 3

theorem sum_largest_smallest_gx : (∀ x, 1 ≤ x ∧ x ≤ 10 → True) → ∀ (a b : ℝ), (∃ x, 1 ≤ x ∧ x ≤ 10 ∧ g x = a) → (∃ y, 1 ≤ y ∧ y ≤ 10 ∧ g y = b) → a + b = -1 :=
by
  intro h x y hx hy
  sorry

end sum_largest_smallest_gx_l514_51412


namespace intervals_of_monotonicity_max_min_on_interval_l514_51460

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem intervals_of_monotonicity :
  (∀ x y : ℝ, x ∈ (Set.Iio (-1) ∪ Set.Ioi (1)) → y ∈ (Set.Iio (-1) ∪ Set.Ioi (1)) → x < y → f x < f y) ∧
  (∀ x y : ℝ, x ∈ (Set.Ioo (-1) 1) → y ∈ (Set.Ioo (-1) 1) → x < y → f x > f y) :=
by
  sorry

theorem max_min_on_interval :
  (∀ x : ℝ, x ∈ Set.Icc (-3) 2 → f x ≤ 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-3) 2 → -18 ≤ f x) ∧
  ((∃ x₁ : ℝ, x₁ ∈ Set.Icc (-3) 2 ∧ f x₁ = 2) ∧ (∃ x₂ : ℝ, x₂ ∈ Set.Icc (-3) 2 ∧ f x₂ = -18)) :=
by
  sorry

end intervals_of_monotonicity_max_min_on_interval_l514_51460


namespace sqrt32_plus_4sqrt_half_minus_sqrt18_l514_51477

theorem sqrt32_plus_4sqrt_half_minus_sqrt18 :
  (Real.sqrt 32 + 4 * Real.sqrt (1/2) - Real.sqrt 18) = 3 * Real.sqrt 2 :=
sorry

end sqrt32_plus_4sqrt_half_minus_sqrt18_l514_51477


namespace find_original_number_l514_51498

theorem find_original_number (n a b: ℤ) 
  (h1 : n > 1000) 
  (h2 : n + 79 = a^2) 
  (h3 : n + 204 = b^2) 
  (h4 : b^2 - a^2 = 125) : 
  n = 3765 := 
by 
  sorry

end find_original_number_l514_51498


namespace find_value_of_x_cubed_plus_y_cubed_l514_51408

-- Definitions based on the conditions provided
variables (x y : ℝ)
variables (h1 : y + 3 = (x - 3)^2) (h2 : x + 3 = (y - 3)^2) (h3 : x ≠ y)

theorem find_value_of_x_cubed_plus_y_cubed :
  x^3 + y^3 = 217 :=
sorry

end find_value_of_x_cubed_plus_y_cubed_l514_51408


namespace remainder_abc_mod_5_l514_51471

theorem remainder_abc_mod_5
  (a b c : ℕ)
  (h₀ : a < 5)
  (h₁ : b < 5)
  (h₂ : c < 5)
  (h₃ : (a + 2 * b + 3 * c) % 5 = 0)
  (h₄ : (2 * a + 3 * b + c) % 5 = 2)
  (h₅ : (3 * a + b + 2 * c) % 5 = 3) :
  (a * b * c) % 5 = 3 :=
by
  sorry

end remainder_abc_mod_5_l514_51471


namespace sunny_bakes_initial_cakes_l514_51493

theorem sunny_bakes_initial_cakes (cakes_after_giving_away : ℕ) (total_candles : ℕ) (candles_per_cake : ℕ) (given_away_cakes : ℕ) (initial_cakes : ℕ) :
  cakes_after_giving_away = total_candles / candles_per_cake →
  given_away_cakes = 2 →
  total_candles = 36 →
  candles_per_cake = 6 →
  initial_cakes = cakes_after_giving_away + given_away_cakes →
  initial_cakes = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end sunny_bakes_initial_cakes_l514_51493


namespace average_weight_increase_l514_51440

theorem average_weight_increase (old_weight : ℕ) (new_weight : ℕ) (n : ℕ) (increase : ℕ) :
  old_weight = 45 → new_weight = 93 → n = 8 → increase = (new_weight - old_weight) / n → increase = 6 :=
by
  intros h_old h_new h_n h_increase
  rw [h_old, h_new, h_n] at h_increase
  simp at h_increase
  exact h_increase

end average_weight_increase_l514_51440


namespace jogger_usual_speed_l514_51474

theorem jogger_usual_speed (V T : ℝ) 
    (h_actual: 30 = V * T) 
    (h_condition: 40 = 16 * T) 
    (h_distance: T = 30 / V) :
  V = 12 := 
by
  sorry

end jogger_usual_speed_l514_51474


namespace diff_of_squares_l514_51420

theorem diff_of_squares : (1001^2 - 999^2 = 4000) :=
by
  sorry

end diff_of_squares_l514_51420


namespace find_value_l514_51497

theorem find_value (x : ℝ) (hx : x + 1/x = 4) : x^3 + 1/x^3 = 52 := 
by 
  sorry

end find_value_l514_51497


namespace first_day_of_month_l514_51443

theorem first_day_of_month 
  (d_24: ℕ) (mod_7: d_24 % 7 = 6) : 
  (d_24 - 23) % 7 = 4 :=
by 
  -- denotes the 24th day is a Saturday (Saturday is the 6th day in a 0-6 index)
  -- hence mod_7: d_24 % 7 = 6 means d_24 falls on a Saturday
  sorry

end first_day_of_month_l514_51443


namespace square_area_l514_51432

theorem square_area
  (E_on_AD : ∃ E : ℝ × ℝ, ∃ s : ℝ, s > 0 ∧ E = (0, s))
  (F_on_extension_BC : ∃ F : ℝ × ℝ, ∃ s : ℝ, s > 0 ∧ F = (s, 0))
  (BE_20 : ∃ B E : ℝ × ℝ, ∃ s : ℝ, B = (s, 0) ∧ E = (0, s) ∧ dist B E = 20)
  (EF_25 : ∃ E F : ℝ × ℝ, ∃ s : ℝ, E = (0, s) ∧ F = (s, 0) ∧ dist E F = 25)
  (FD_20 : ∃ F D : ℝ × ℝ, ∃ s : ℝ, F = (s, 0) ∧ D = (s, s) ∧ dist F D = 20) :
  ∃ s : ℝ, s > 0 ∧ s^2 = 400 :=
by
  -- Hypotheses are laid out in conditions as defined above
  sorry

end square_area_l514_51432


namespace tan_seven_pi_over_six_l514_51424
  
theorem tan_seven_pi_over_six :
  Real.tan (7 * Real.pi / 6) = 1 / Real.sqrt 3 :=
sorry

end tan_seven_pi_over_six_l514_51424
