import Mathlib

namespace NUMINAMATH_GPT_min_days_is_9_l1081_108132

theorem min_days_is_9 (n : ℕ) (rain_morning rain_afternoon sunny_morning sunny_afternoon : ℕ)
  (h1 : rain_morning + rain_afternoon = 7)
  (h2 : rain_afternoon ≤ sunny_morning)
  (h3 : sunny_afternoon = 5)
  (h4 : sunny_morning = 6) :
  n ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_days_is_9_l1081_108132


namespace NUMINAMATH_GPT_tan_expression_val_l1081_108107

theorem tan_expression_val (A B : ℝ) (hA : A = 30) (hB : B = 15) :
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_expression_val_l1081_108107


namespace NUMINAMATH_GPT_net_income_after_tax_l1081_108129

theorem net_income_after_tax (gross_income : ℝ) (tax_rate : ℝ) : 
  (gross_income = 45000) → (tax_rate = 0.13) → 
  (gross_income - gross_income * tax_rate = 39150) :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_net_income_after_tax_l1081_108129


namespace NUMINAMATH_GPT_sum_eq_zero_l1081_108120

variable {R : Type} [Field R]

-- Define the conditions
def cond1 (a b c : R) : Prop := (a + b) / c = (b + c) / a
def cond2 (a b c : R) : Prop := (b + c) / a = (a + c) / b
def neq (b c : R) : Prop := b ≠ c

-- State the theorem
theorem sum_eq_zero (a b c : R) (h1 : cond1 a b c) (h2 : cond2 a b c) (h3 : neq b c) : a + b + c = 0 := 
by sorry

end NUMINAMATH_GPT_sum_eq_zero_l1081_108120


namespace NUMINAMATH_GPT_student_count_l1081_108169

theorem student_count 
  (initial_avg_height : ℚ)
  (incorrect_height : ℚ)
  (actual_height : ℚ)
  (actual_avg_height : ℚ)
  (n : ℕ)
  (h1 : initial_avg_height = 175)
  (h2 : incorrect_height = 151)
  (h3 : actual_height = 136)
  (h4 : actual_avg_height = 174.5)
  (h5 : n > 0) : n = 30 :=
by
  sorry

end NUMINAMATH_GPT_student_count_l1081_108169


namespace NUMINAMATH_GPT_evaluate_expression_l1081_108160

theorem evaluate_expression : -1 ^ 2010 + (-1) ^ 2011 + 1 ^ 2012 - 1 ^ 2013 = -2 :=
by
  -- sorry is added as a placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1081_108160


namespace NUMINAMATH_GPT_A_greater_than_B_l1081_108114

theorem A_greater_than_B (A B : ℝ) (h₁ : A * 4 = B * 5) (h₂ : A ≠ 0) (h₃ : B ≠ 0) : A > B :=
by
  sorry

end NUMINAMATH_GPT_A_greater_than_B_l1081_108114


namespace NUMINAMATH_GPT_necessary_condition_l1081_108135

theorem necessary_condition :
  (∀ x : ℝ, (1 / x < 3) → (x > 1 / 3)) → (∀ x : ℝ, (1 / x < 3) ↔ (x > 1 / 3)) → False :=
by
  sorry

end NUMINAMATH_GPT_necessary_condition_l1081_108135


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l1081_108193

open BigOperators

theorem sum_of_reciprocals_of_squares (n : ℕ) (h : n ≥ 2) :
   (∑ k in Finset.range n, 1 / (k + 1)^2) < (2 * n - 1) / n :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l1081_108193


namespace NUMINAMATH_GPT_first_installment_amount_l1081_108121

-- Define the conditions stated in the problem
def original_price : ℝ := 480
def discount_rate : ℝ := 0.05
def monthly_installment : ℝ := 102
def number_of_installments : ℕ := 3

-- The final price after discount
def final_price : ℝ := original_price * (1 - discount_rate)

-- The total amount of the 3 monthly installments
def total_of_3_installments : ℝ := monthly_installment * number_of_installments

-- The first installment paid
def first_installment : ℝ := final_price - total_of_3_installments

-- The main theorem to prove the first installment amount
theorem first_installment_amount : first_installment = 150 := by
  unfold first_installment
  unfold final_price
  unfold total_of_3_installments
  unfold original_price
  unfold discount_rate
  unfold monthly_installment
  unfold number_of_installments
  sorry

end NUMINAMATH_GPT_first_installment_amount_l1081_108121


namespace NUMINAMATH_GPT_find_number_l1081_108112

def number_condition (N : ℝ) : Prop := 
  0.20 * 0.15 * 0.40 * 0.30 * 0.50 * N = 180

theorem find_number (N : ℝ) (h : number_condition N) : N = 1000000 :=
sorry

end NUMINAMATH_GPT_find_number_l1081_108112


namespace NUMINAMATH_GPT_range_of_x_for_valid_sqrt_l1081_108166

theorem range_of_x_for_valid_sqrt (x : ℝ) (h : 2 * x - 4 ≥ 0) : x ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_for_valid_sqrt_l1081_108166


namespace NUMINAMATH_GPT_solve_for_y_l1081_108106

theorem solve_for_y (y : ℕ) (h : 9^y = 3^12) : y = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_y_l1081_108106


namespace NUMINAMATH_GPT_candy_game_solution_l1081_108145

open Nat

theorem candy_game_solution 
  (total_candies : ℕ) 
  (nick_candies : ℕ) 
  (tim_candies : ℕ)
  (tim_wins : ℕ)
  (m n : ℕ)
  (htotal : total_candies = 55) 
  (hnick : nick_candies = 30) 
  (htim : tim_candies = 25)
  (htim_wins : tim_wins = 2)
  (hrounds_total : total_candies = nick_candies + tim_candies)
  (hwinner_condition1 : m > n) 
  (hwinner_condition2 : n > 0) 
  (hwinner_candies_total : total_candies = tim_wins * m + (total_candies / (m + n) - tim_wins) * n)
: m = 8 := 
sorry

end NUMINAMATH_GPT_candy_game_solution_l1081_108145


namespace NUMINAMATH_GPT_shelly_thread_length_l1081_108191

theorem shelly_thread_length 
  (threads_per_keychain : ℕ := 12) 
  (friends_in_class : ℕ := 6) 
  (friends_from_clubs := friends_in_class / 2)
  (total_friends := friends_in_class + friends_from_clubs) 
  (total_threads_needed := total_friends * threads_per_keychain) : 
  total_threads_needed = 108 := 
by 
  -- proof skipped
  sorry

end NUMINAMATH_GPT_shelly_thread_length_l1081_108191


namespace NUMINAMATH_GPT_infinite_nested_radical_l1081_108104

theorem infinite_nested_radical : ∀ (x : ℝ), (x > 0) → (x = Real.sqrt (12 + x)) → x = 4 :=
by
  intro x
  intro hx_pos
  intro hx_eq
  sorry

end NUMINAMATH_GPT_infinite_nested_radical_l1081_108104


namespace NUMINAMATH_GPT_joy_pencils_count_l1081_108117

theorem joy_pencils_count :
  ∃ J, J = 30 ∧ (∃ (pencils_cost_J pencils_cost_C : ℕ), 
  pencils_cost_C = 50 * 4 ∧ pencils_cost_J = pencils_cost_C - 80 ∧ J = pencils_cost_J / 4) := sorry

end NUMINAMATH_GPT_joy_pencils_count_l1081_108117


namespace NUMINAMATH_GPT_new_light_wattage_l1081_108127

theorem new_light_wattage (w_old : ℕ) (p : ℕ) (w_new : ℕ) (h1 : w_old = 110) (h2 : p = 30) (h3 : w_new = w_old + (p * w_old / 100)) : w_new = 143 :=
by
  -- Using the conditions provided
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_new_light_wattage_l1081_108127


namespace NUMINAMATH_GPT_problem_solution_l1081_108181

theorem problem_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2):
    (x ∈ Set.Iio (-2) ∪ Set.Ioo (-2) ((1 - Real.sqrt 129)/8) ∪ Set.Ioo 2 3 ∪ Set.Ioi ((1 + (Real.sqrt 129))/8)) ↔
    (2 * x^2 / (x + 2) ≥ 3 / (x - 2) + 6 / 4) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1081_108181


namespace NUMINAMATH_GPT_not_divisible_by_n_plus_4_l1081_108147

theorem not_divisible_by_n_plus_4 (n : ℕ) (h : n > 0) : ¬ ∃ k : ℕ, n^2 + 8 * n + 15 = k * (n + 4) := by
  sorry

end NUMINAMATH_GPT_not_divisible_by_n_plus_4_l1081_108147


namespace NUMINAMATH_GPT_determine_x_l1081_108151

theorem determine_x (x : ℝ) (h : (1 / (Real.log x / Real.log 3) + 1 / (Real.log x / Real.log 5) + 1 / (Real.log x / Real.log 6) = 1)) : 
    x = 90 := 
by 
  sorry

end NUMINAMATH_GPT_determine_x_l1081_108151


namespace NUMINAMATH_GPT_total_amount_paid_l1081_108190

theorem total_amount_paid :
  let chapati_cost := 6
  let rice_cost := 45
  let mixed_vegetable_cost := 70
  let ice_cream_cost := 40
  let chapati_quantity := 16
  let rice_quantity := 5
  let mixed_vegetable_quantity := 7
  let ice_cream_quantity := 6
  let total_cost := chapati_quantity * chapati_cost +
                    rice_quantity * rice_cost +
                    mixed_vegetable_quantity * mixed_vegetable_cost +
                    ice_cream_quantity * ice_cream_cost
  total_cost = 1051 := by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1081_108190


namespace NUMINAMATH_GPT_remainder_13_pow_51_mod_5_l1081_108139

theorem remainder_13_pow_51_mod_5 : 13^51 % 5 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_13_pow_51_mod_5_l1081_108139


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_l1081_108142

variable (p q r : ℝ)

theorem radius_of_inscribed_circle (hp : p > 0) (hq : q > 0) (area_eq : q^2 = r * p) : r = q^2 / p :=
by
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_l1081_108142


namespace NUMINAMATH_GPT_find_k_l1081_108153

theorem find_k (x y z k : ℝ) (h1 : 8 / (x + y + 1) = k / (x + z + 2)) (h2 : k / (x + z + 2) = 12 / (z - y + 3)) : k = 20 := by
  sorry

end NUMINAMATH_GPT_find_k_l1081_108153


namespace NUMINAMATH_GPT_mod_2_pow_1000_by_13_l1081_108198

theorem mod_2_pow_1000_by_13 :
  (2 ^ 1000) % 13 = 3 := by
  sorry

end NUMINAMATH_GPT_mod_2_pow_1000_by_13_l1081_108198


namespace NUMINAMATH_GPT_problem1_problem2_l1081_108176

-- Step 1
theorem problem1 (a b c A B C : ℝ) (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) :
  2 * a^2 = b^2 + c^2 := sorry

-- Step 2
theorem problem2 (a b c : ℝ) (h_a : a = 5) (h_cosA : Real.cos A = 25 / 31) 
  (h_conditions : 2 * a^2 = b^2 + c^2 ∧ 2 * b * c = a^2 / Real.cos A) :
  a + b + c = 14 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1081_108176


namespace NUMINAMATH_GPT_Panthers_total_games_l1081_108109

/-
Given:
1) The Panthers had won 60% of their basketball games before the district play.
2) During district play, they won four more games and lost four.
3) They finished the season having won half of their total games.
Prove that the total number of games they played in all is 48.
-/

theorem Panthers_total_games
  (y : ℕ) -- total games before district play
  (x : ℕ) -- games won before district play
  (h1 : x = 60 * y / 100) -- they won 60% of the games before district play
  (h2 : (x + 4) = 50 * (y + 8) / 100) -- they won half of the total games including district play
  : (y + 8) = 48 := -- total games they played in all
sorry

end NUMINAMATH_GPT_Panthers_total_games_l1081_108109


namespace NUMINAMATH_GPT_bakery_regular_price_l1081_108116

theorem bakery_regular_price (y : ℝ) (h₁ : y / 4 * 0.4 = 2) : y = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_bakery_regular_price_l1081_108116


namespace NUMINAMATH_GPT_triangular_prism_distance_sum_l1081_108102

theorem triangular_prism_distance_sum (V K H1 H2 H3 H4 S1 S2 S3 S4 : ℝ)
  (h1 : S1 = K)
  (h2 : S2 = 2 * K)
  (h3 : S3 = 3 * K)
  (h4 : S4 = 4 * K)
  (hV : (S1 * H1 + S2 * H2 + S3 * H3 + S4 * H4) / 3 = V) :
  H1 + 2 * H2 + 3 * H3 + 4 * H4 = 3 * V / K :=
by sorry

end NUMINAMATH_GPT_triangular_prism_distance_sum_l1081_108102


namespace NUMINAMATH_GPT_max_range_walk_min_range_walk_count_max_range_sequences_l1081_108178

variable {a b : ℕ}

-- Condition: a > b
def valid_walk (a b : ℕ) : Prop := a > b

-- Proof that the maximum possible range of the walk is a
theorem max_range_walk (h : valid_walk a b) : 
  (a + b) = a + b := sorry

-- Proof that the minimum possible range of the walk is a - b
theorem min_range_walk (h : valid_walk a b) : 
  (a - b) = a - b := sorry

-- Proof that the number of different sequences with the maximum possible range is b + 1
theorem count_max_range_sequences (h : valid_walk a b) : 
  b + 1 = b + 1 := sorry

end NUMINAMATH_GPT_max_range_walk_min_range_walk_count_max_range_sequences_l1081_108178


namespace NUMINAMATH_GPT_student_correct_answers_l1081_108124

theorem student_correct_answers 
  (c w : ℕ) 
  (h1 : c + w = 60) 
  (h2 : 4 * c - w = 130) : 
  c = 38 :=
by
  sorry

end NUMINAMATH_GPT_student_correct_answers_l1081_108124


namespace NUMINAMATH_GPT_num_children_in_family_l1081_108134

def regular_ticket_cost := 15
def elderly_ticket_cost := 10
def adult_ticket_cost := 12
def child_ticket_cost := adult_ticket_cost - 5
def total_money_handled := 3 * 50
def change_received := 3
def num_adults := 4
def num_elderly := 2
def total_cost_for_adults := num_adults * adult_ticket_cost
def total_cost_for_elderly := num_elderly * elderly_ticket_cost
def total_cost_of_tickets := total_money_handled - change_received

theorem num_children_in_family : ∃ (num_children : ℕ), 
  total_cost_of_tickets = total_cost_for_adults + total_cost_for_elderly + num_children * child_ticket_cost ∧ 
  num_children = 11 := 
by
  sorry

end NUMINAMATH_GPT_num_children_in_family_l1081_108134


namespace NUMINAMATH_GPT_smallest_value_of_3a_plus_2_l1081_108130

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) : 
  ∃ (x : ℝ), x = 3 * a + 2 ∧ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_3a_plus_2_l1081_108130


namespace NUMINAMATH_GPT_parallel_vectors_l1081_108177

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2, 1)) (h₂ : b = (1, m))
  (h₃ : ∃ k : ℝ, b = k • a) : m = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_parallel_vectors_l1081_108177


namespace NUMINAMATH_GPT_profit_percentage_is_correct_l1081_108171

noncomputable def CP : ℝ := 47.50
noncomputable def SP : ℝ := 74.21875
noncomputable def MP : ℝ := SP / 0.8
noncomputable def Profit : ℝ := SP - CP
noncomputable def ProfitPercentage : ℝ := (Profit / CP) * 100

theorem profit_percentage_is_correct : ProfitPercentage = 56.25 := by
  -- Proof steps to be filled in
  sorry

end NUMINAMATH_GPT_profit_percentage_is_correct_l1081_108171


namespace NUMINAMATH_GPT_probability_abs_x_le_one_l1081_108179

noncomputable def geometric_probability (a b c d : ℝ) : ℝ := (b - a) / (d - c)

theorem probability_abs_x_le_one : 
  ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 3 →  
  geometric_probability (-1) 1 (-1) 3 = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_probability_abs_x_le_one_l1081_108179


namespace NUMINAMATH_GPT_S6_geometric_sum_l1081_108140

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem S6_geometric_sum (a r : ℝ)
    (sum_n : ℕ → ℝ)
    (geo_seq : ∀ n, sum_n n = geometric_sequence_sum a r n)
    (S2 : sum_n 2 = 6)
    (S4 : sum_n 4 = 30) :
    sum_n 6 = 126 := 
by
  sorry

end NUMINAMATH_GPT_S6_geometric_sum_l1081_108140


namespace NUMINAMATH_GPT_max_C_trees_l1081_108194

theorem max_C_trees 
  (price_A : ℕ) (price_B : ℕ) (price_C : ℕ) (total_price : ℕ) (total_trees : ℕ)
  (h_price_ratio : 2 * price_B = 2 * price_A ∧ 3 * price_A = 2 * price_C)
  (h_price_A : price_A = 200)
  (h_total_price : total_price = 220120)
  (h_total_trees : total_trees = 1000) :
  ∃ (num_C : ℕ), num_C = 201 ∧ ∀ num_C', num_C' > num_C → 
  total_price < price_A * (total_trees - num_C') + price_C * num_C' :=
by
  sorry

end NUMINAMATH_GPT_max_C_trees_l1081_108194


namespace NUMINAMATH_GPT_distinct_real_roots_find_p_l1081_108195

theorem distinct_real_roots (p : ℝ) : 
  let f := (fun x => (x - 3) * (x - 2) - p^2)
  let Δ := 1 + 4 * p ^ 2 
  0 < Δ :=
by sorry

theorem find_p (x1 x2 p : ℝ) : 
  (x1 + x2 = 5) → 
  (x1 * x2 = 6 - p^2) → 
  (x1^2 + x2^2 = 3 * x1 * x2) → 
  (p = 1 ∨ p = -1) :=
by sorry

end NUMINAMATH_GPT_distinct_real_roots_find_p_l1081_108195


namespace NUMINAMATH_GPT_polygon_sides_l1081_108137

theorem polygon_sides (n : ℕ) (h₁ : ∀ (m : ℕ), m = n → n > 2) (h₂ : 180 * (n - 2) = 156 * n) : n = 15 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1081_108137


namespace NUMINAMATH_GPT_common_ratio_common_difference_l1081_108103

noncomputable def common_ratio_q {a b : ℕ → ℝ} (d : ℝ) (q : ℝ) :=
  (∀ n, b (n+1) = q * b n) ∧ (a 2 = -1) ∧ (a 1 < a 2) ∧ 
  (b 1 = (a 1)^2) ∧ (b 2 = (a 2)^2) ∧ (b 3 = (a 3)^2) ∧ 
  (∀ n, a (n+1) = a n + d)

theorem common_ratio
  {a b : ℕ → ℝ} {d : ℝ}
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_nonzero : d ≠ 0)
  (h_geom : ∀ n, b (n + 1) = (b 1^(1/2)) ^ (2 ^ n))
  (h_b1 : b 1 = (a 1) ^ 2) (h_b2 : b 2 = (a 2) ^ 2)
  (h_b3 : b 3 = (a 3) ^ 2) (h_a2 : a 2 = -1) (h_a1a2 : a 1 < a 2) :
  q = 3 - 2 * (2:ℝ).sqrt :=
sorry

theorem common_difference
  {a b : ℕ → ℝ} {d : ℝ}
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_nonzero : d ≠ 0)
  (h_geom : ∀ n, b (n + 1) = (b 1^(1/2)) ^ (2 ^ n))
  (h_b1 : b 1 = (a 1) ^ 2) (h_b2 : b 2 = (a 2) ^ 2)
  (h_b3 : b 3 = (a 3) ^ 2) (h_a2 : a 2 = -1) (h_a1a2 : a 1 < a 2) :
  d = (2 : ℝ).sqrt :=
sorry

end NUMINAMATH_GPT_common_ratio_common_difference_l1081_108103


namespace NUMINAMATH_GPT_flattest_ellipse_is_B_l1081_108156

-- Definitions for the given ellipses
def ellipseA : Prop := ∀ (x y : ℝ), (x^2 / 16 + y^2 / 12 = 1)
def ellipseB : Prop := ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1)
def ellipseC : Prop := ∀ (x y : ℝ), (x^2 / 6 + y^2 / 3 = 1)
def ellipseD : Prop := ∀ (x y : ℝ), (x^2 / 9 + y^2 / 8 = 1)

-- The proof to show that ellipseB is the flattest
theorem flattest_ellipse_is_B : ellipseB := by
  sorry

end NUMINAMATH_GPT_flattest_ellipse_is_B_l1081_108156


namespace NUMINAMATH_GPT_line_segment_value_of_x_l1081_108125

theorem line_segment_value_of_x (x : ℝ) (h1 : (1 - 4)^2 + (3 - x)^2 = 25) (h2 : x > 0) : x = 7 :=
sorry

end NUMINAMATH_GPT_line_segment_value_of_x_l1081_108125


namespace NUMINAMATH_GPT_calculate_sequence_sum_l1081_108197

noncomputable def sum_arithmetic_sequence (a l d: Int) : Int :=
  let n := ((l - a) / d) + 1
  (n * (a + l)) / 2

theorem calculate_sequence_sum :
  3 * (sum_arithmetic_sequence 45 93 2) + 2 * (sum_arithmetic_sequence (-4) 38 2) = 5923 := by
  sorry

end NUMINAMATH_GPT_calculate_sequence_sum_l1081_108197


namespace NUMINAMATH_GPT_propositions_correctness_l1081_108172

variable {a b c d : ℝ}

theorem propositions_correctness (h0 : a > b) (h1 : c > d) (h2 : c > 0) :
  (a > b ∧ c > d → a + c > b + d) ∧ 
  (a > b ∧ c > d → ¬(a - c > b - d)) ∧ 
  (a > b ∧ c > d → ¬(a * c > b * d)) ∧ 
  (a > b ∧ c > 0 → a * c > b * c) :=
by
  sorry

end NUMINAMATH_GPT_propositions_correctness_l1081_108172


namespace NUMINAMATH_GPT_compute_fg_l1081_108143

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 3 * x + 4

theorem compute_fg : f (g (-3)) = 25 := by
  sorry

end NUMINAMATH_GPT_compute_fg_l1081_108143


namespace NUMINAMATH_GPT_nylon_cord_length_l1081_108149

-- Let the length of cord be w
-- Dog runs 30 feet forming a semicircle, that is pi * w = 30
-- Prove that w is approximately 9.55

theorem nylon_cord_length (pi_approx : Real := 3.14) : Real :=
  let w := 30 / pi_approx
  w

end NUMINAMATH_GPT_nylon_cord_length_l1081_108149


namespace NUMINAMATH_GPT_trajectory_of_circle_center_l1081_108101

open Real

noncomputable def circle_trajectory_equation (x y : ℝ) : Prop :=
  (y ^ 2 = 8 * x - 16)

theorem trajectory_of_circle_center (x y : ℝ) :
  (∃ C : ℝ × ℝ, (C.1 = 4 ∧ C.2 = 0) ∧
    (∃ MN : ℝ × ℝ, (MN.1 = 0 ∧ MN.2 ^ 2 = 64) ∧
    (x = C.1 ∧ y = C.2)) ∧
    circle_trajectory_equation x y) :=
sorry

end NUMINAMATH_GPT_trajectory_of_circle_center_l1081_108101


namespace NUMINAMATH_GPT_charge_per_mile_l1081_108183

def rental_fee : ℝ := 20.99
def total_amount_paid : ℝ := 95.74
def miles_driven : ℝ := 299

theorem charge_per_mile :
  (total_amount_paid - rental_fee) / miles_driven = 0.25 := 
sorry

end NUMINAMATH_GPT_charge_per_mile_l1081_108183


namespace NUMINAMATH_GPT_kenneth_money_left_l1081_108122

noncomputable def baguettes : ℝ := 2 * 2
noncomputable def water : ℝ := 2 * 1

noncomputable def chocolate_bars_cost_before_discount : ℝ := 2 * 1.5
noncomputable def chocolate_bars_cost_after_discount : ℝ := chocolate_bars_cost_before_discount * (1 - 0.20)
noncomputable def chocolate_bars_final_cost : ℝ := chocolate_bars_cost_after_discount * 1.08

noncomputable def milk_cost_after_discount : ℝ := 3.5 * (1 - 0.10)

noncomputable def chips_cost_before_tax : ℝ := 2.5 + (2.5 * 0.50)
noncomputable def chips_final_cost : ℝ := chips_cost_before_tax * 1.08

noncomputable def total_cost : ℝ :=
  baguettes + water + chocolate_bars_final_cost + milk_cost_after_discount + chips_final_cost

noncomputable def initial_amount : ℝ := 50
noncomputable def amount_left : ℝ := initial_amount - total_cost

theorem kenneth_money_left : amount_left = 50 - 15.792 := by
  sorry

end NUMINAMATH_GPT_kenneth_money_left_l1081_108122


namespace NUMINAMATH_GPT_largest_even_integer_product_l1081_108144

theorem largest_even_integer_product (n : ℕ) (h : 2 * n * (2 * n + 2) * (2 * n + 4) * (2 * n + 6) = 5040) :
  2 * n + 6 = 20 :=
by
  sorry

end NUMINAMATH_GPT_largest_even_integer_product_l1081_108144


namespace NUMINAMATH_GPT_best_choice_to_calculate_89_8_sq_l1081_108170

theorem best_choice_to_calculate_89_8_sq 
  (a b c d : ℚ) 
  (h1 : (89 + 0.8)^2 = a) 
  (h2 : (80 + 9.8)^2 = b) 
  (h3 : (90 - 0.2)^2 = c) 
  (h4 : (100 - 10.2)^2 = d) : 
  c = 89.8^2 := by
  sorry

end NUMINAMATH_GPT_best_choice_to_calculate_89_8_sq_l1081_108170


namespace NUMINAMATH_GPT_max_sum_of_factors_l1081_108159

theorem max_sum_of_factors (x y : ℕ) (h1 : x * y = 48) (h2 : x ≠ y) : x + y ≤ 49 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_of_factors_l1081_108159


namespace NUMINAMATH_GPT_games_required_for_champion_l1081_108118

-- Define the number of players in the tournament
def players : ℕ := 512

-- Define the tournament conditions
def single_elimination_tournament (n : ℕ) : Prop :=
  ∀ (g : ℕ), g = n - 1

-- State the theorem that needs to be proven
theorem games_required_for_champion : single_elimination_tournament players :=
by
  sorry

end NUMINAMATH_GPT_games_required_for_champion_l1081_108118


namespace NUMINAMATH_GPT_parabola_y_intercepts_zero_l1081_108115

-- Define the quadratic equation
def quadratic (a b c y: ℝ) : ℝ := a * y^2 + b * y + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Condition: equation of the parabola and discriminant calculation
def parabola_equation : Prop := 
  let a := 3
  let b := -4
  let c := 5
  discriminant a b c < 0

-- Statement to prove
theorem parabola_y_intercepts_zero : 
  (parabola_equation) → (∀ y : ℝ, quadratic 3 (-4) 5 y ≠ 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_y_intercepts_zero_l1081_108115


namespace NUMINAMATH_GPT_nancy_soap_bars_l1081_108163

def packs : ℕ := 6
def bars_per_pack : ℕ := 5

theorem nancy_soap_bars : packs * bars_per_pack = 30 := by
  sorry

end NUMINAMATH_GPT_nancy_soap_bars_l1081_108163


namespace NUMINAMATH_GPT_count_multiples_less_than_300_l1081_108108

theorem count_multiples_less_than_300 : ∀ n : ℕ, n < 300 → (2 * 3 * 5 * 7 ∣ n) ↔ n = 210 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_less_than_300_l1081_108108


namespace NUMINAMATH_GPT_exists_colored_right_triangle_l1081_108146

theorem exists_colored_right_triangle (color : ℝ × ℝ → ℕ) 
  (h_nonempty_blue  : ∃ p, color p = 0)
  (h_nonempty_green : ∃ p, color p = 1)
  (h_nonempty_red   : ∃ p, color p = 2) :
  ∃ p1 p2 p3 : ℝ × ℝ, 
    (p1 ≠ p2) ∧ (p2 ≠ p3) ∧ (p1 ≠ p3) ∧ 
    ((color p1 = 0) ∧ (color p2 = 1) ∧ (color p3 = 2) ∨ 
     (color p1 = 0) ∧ (color p2 = 2) ∧ (color p3 = 1) ∨ 
     (color p1 = 1) ∧ (color p2 = 0) ∧ (color p3 = 2) ∨ 
     (color p1 = 1) ∧ (color p2 = 2) ∧ (color p3 = 0) ∨ 
     (color p1 = 2) ∧ (color p2 = 0) ∧ (color p3 = 1) ∨ 
     (color p1 = 2) ∧ (color p2 = 1) ∧ (color p3 = 0))
  ∧ ((p1.1 = p2.1 ∧ p2.2 = p3.2) ∨ (p1.2 = p2.2 ∧ p2.1 = p3.1)) :=
sorry

end NUMINAMATH_GPT_exists_colored_right_triangle_l1081_108146


namespace NUMINAMATH_GPT_find_b_l1081_108199

theorem find_b (a b : ℝ) (h_inv_var : a^2 * Real.sqrt b = k) (h_ab : a * b = 72) (ha3 : a = 3) (hb64 : b = 64) : b = 18 :=
sorry

end NUMINAMATH_GPT_find_b_l1081_108199


namespace NUMINAMATH_GPT_min_value_f_range_m_l1081_108148

-- Part I: Prove that the minimum value of f(a) = a^2 + 2/a for a > 0 is 3
theorem min_value_f (a : ℝ) (h : a > 0) : a^2 + 2 / a ≥ 3 :=
sorry

-- Part II: Prove the range of m given the inequality for any positive real number a
theorem range_m (m : ℝ) : (∀ (a : ℝ), a > 0 → a^3 + 2 ≥ 3 * a * (|m - 1| - |2 * m + 3|)) → (m ≤ -3 ∨ m ≥ -1) :=
sorry

end NUMINAMATH_GPT_min_value_f_range_m_l1081_108148


namespace NUMINAMATH_GPT_complete_square_form_l1081_108187

theorem complete_square_form :
  ∀ x : ℝ, (3 * x^2 - 6 * x + 2 = 0) → (x - 1)^2 = (1 / 3) :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_complete_square_form_l1081_108187


namespace NUMINAMATH_GPT_percentage_with_diploma_l1081_108136

-- Define the percentages as variables for clarity
def low_income_perc := 0.25
def lower_middle_income_perc := 0.35
def upper_middle_income_perc := 0.25
def high_income_perc := 0.15

def low_income_diploma := 0.05
def lower_middle_income_diploma := 0.35
def upper_middle_income_diploma := 0.60
def high_income_diploma := 0.80

theorem percentage_with_diploma :
  (low_income_perc * low_income_diploma +
   lower_middle_income_perc * lower_middle_income_diploma +
   upper_middle_income_perc * upper_middle_income_diploma +
   high_income_perc * high_income_diploma) = 0.405 :=
by sorry

end NUMINAMATH_GPT_percentage_with_diploma_l1081_108136


namespace NUMINAMATH_GPT_image_of_neg2_3_preimages_2_neg3_l1081_108110

variables {A B : Type}
def f (x y : ℤ) : ℤ × ℤ := (x + y, x * y)

-- Prove that the image of (-2, 3) under f is (1, -6)
theorem image_of_neg2_3 : f (-2) 3 = (1, -6) := sorry

-- Find the preimages of (2, -3) under f
def preimages_of_2_neg3 (p : ℤ × ℤ) : Prop := f p.1 p.2 = (2, -3)

theorem preimages_2_neg3 : preimages_of_2_neg3 (-1, 3) ∧ preimages_of_2_neg3 (3, -1) := sorry

end NUMINAMATH_GPT_image_of_neg2_3_preimages_2_neg3_l1081_108110


namespace NUMINAMATH_GPT_division_expression_evaluation_l1081_108111

theorem division_expression_evaluation : 120 / (6 / 2) = 40 := by
  sorry

end NUMINAMATH_GPT_division_expression_evaluation_l1081_108111


namespace NUMINAMATH_GPT_product_of_slopes_l1081_108184

theorem product_of_slopes (m n : ℝ) (φ₁ φ₂ : ℝ) 
  (h1 : ∀ x, y = m * x)
  (h2 : ∀ x, y = n * x)
  (h3 : φ₁ = 2 * φ₂) 
  (h4 : m = 3 * n)
  (h5 : m ≠ 0 ∧ n ≠ 0)
  : m * n = 3 / 5 :=
sorry

end NUMINAMATH_GPT_product_of_slopes_l1081_108184


namespace NUMINAMATH_GPT_circle_center_l1081_108188

theorem circle_center (x y : ℝ) : (x - 2)^2 + (y + 1)^2 = 3 → (2, -1) = (2, -1) :=
by
  intro h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_circle_center_l1081_108188


namespace NUMINAMATH_GPT_sales_volume_maximum_profit_l1081_108123

noncomputable def profit (x : ℝ) : ℝ := (x - 34) * (-2 * x + 296)

theorem sales_volume (x : ℝ) : 200 - 2 * (x - 48) = -2 * x + 296 := by
  sorry

theorem maximum_profit :
  (∀ x : ℝ, profit x ≤ profit 91) ∧ profit 91 = 6498 := by
  sorry

end NUMINAMATH_GPT_sales_volume_maximum_profit_l1081_108123


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1081_108175

theorem sufficient_not_necessary (x : ℝ) : (x - 1 > 0) → (x^2 - 1 > 0) ∧ ¬((x^2 - 1 > 0) → (x - 1 > 0)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1081_108175


namespace NUMINAMATH_GPT_pages_per_side_is_4_l1081_108113

-- Define the conditions
def num_books := 2
def pages_per_book := 600
def sheets_used := 150
def sides_per_sheet := 2

-- Define the total number of pages and sides
def total_pages := num_books * pages_per_book
def total_sides := sheets_used * sides_per_sheet

-- Prove the number of pages per side is 4
theorem pages_per_side_is_4 : total_pages / total_sides = 4 := by
  sorry

end NUMINAMATH_GPT_pages_per_side_is_4_l1081_108113


namespace NUMINAMATH_GPT_fred_green_balloons_l1081_108133

theorem fred_green_balloons (initial : ℕ) (given : ℕ) (final : ℕ) (h1 : initial = 709) (h2 : given = 221) (h3 : final = initial - given) : final = 488 :=
by
  sorry

end NUMINAMATH_GPT_fred_green_balloons_l1081_108133


namespace NUMINAMATH_GPT_intimate_interval_proof_l1081_108141

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := 2 * x - 3

-- Define the concept of intimate functions over an interval
def are_intimate_functions (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- Prove that the interval [2, 3] is a subset of [a, b]
theorem intimate_interval_proof (a b : ℝ) (h : are_intimate_functions a b) :
  2 ≤ b ∧ a ≤ 3 :=
sorry

end NUMINAMATH_GPT_intimate_interval_proof_l1081_108141


namespace NUMINAMATH_GPT_quadrant_of_points_l1081_108158

theorem quadrant_of_points (x y : ℝ) (h : |3 * x + 2| + |2 * y - 1| = 0) : 
  ((x < 0) ∧ (y > 0) ∧ (x + 1 > 0) ∧ (y - 2 < 0)) :=
by
  sorry

end NUMINAMATH_GPT_quadrant_of_points_l1081_108158


namespace NUMINAMATH_GPT_greatest_value_l1081_108182

theorem greatest_value (y : ℝ) (h : 4 * y^2 + 4 * y + 3 = 1) : (y + 1)^2 = 1/4 :=
sorry

end NUMINAMATH_GPT_greatest_value_l1081_108182


namespace NUMINAMATH_GPT_g_10_44_l1081_108173

def g (x y : ℕ) : ℕ := sorry

axiom g_cond1 (x : ℕ) : g x x = x ^ 2
axiom g_cond2 (x y : ℕ) : g x y = g y x
axiom g_cond3 (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_10_44 : g 10 44 = 440 := sorry

end NUMINAMATH_GPT_g_10_44_l1081_108173


namespace NUMINAMATH_GPT_sum_coordinates_of_k_l1081_108152

theorem sum_coordinates_of_k :
  ∀ (f k : ℕ → ℕ), (f 4 = 8) → (∀ x, k x = (f x) ^ 3) → (4 + k 4) = 516 :=
by
  intros f k h1 h2
  sorry

end NUMINAMATH_GPT_sum_coordinates_of_k_l1081_108152


namespace NUMINAMATH_GPT_game_show_prizes_l1081_108128

theorem game_show_prizes :
  let digits := [1, 1, 2, 2, 3, 3, 3, 3]
  let permutations := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 2)
  let partitions := Nat.choose 7 3
  permutations * partitions = 14700 :=
by
  let digits := [1, 1, 2, 2, 3, 3, 3, 3]
  let permutations := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 2)
  let partitions := Nat.choose 7 3
  exact sorry

end NUMINAMATH_GPT_game_show_prizes_l1081_108128


namespace NUMINAMATH_GPT_tree_age_when_23_feet_l1081_108162

theorem tree_age_when_23_feet (initial_age initial_height growth_rate final_height : ℕ) 
(h_initial_age : initial_age = 1)
(h_initial_height : initial_height = 5) 
(h_growth_rate : growth_rate = 3) 
(h_final_height : final_height = 23) : 
initial_age + (final_height - initial_height) / growth_rate = 7 := 
by sorry

end NUMINAMATH_GPT_tree_age_when_23_feet_l1081_108162


namespace NUMINAMATH_GPT_calculate_avg_l1081_108126

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem calculate_avg :
  avg3 (avg3 1 2 0) (avg2 0 2) 0 = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_avg_l1081_108126


namespace NUMINAMATH_GPT_division_expression_result_l1081_108164

theorem division_expression_result :
  -1 / (-5) / (-1 / 5) = -1 :=
by sorry

end NUMINAMATH_GPT_division_expression_result_l1081_108164


namespace NUMINAMATH_GPT_polygon_sides_l1081_108174

theorem polygon_sides (n : ℕ) (c : ℕ) 
  (h₁ : c = n * (n - 3) / 2)
  (h₂ : c = 2 * n) : n = 7 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l1081_108174


namespace NUMINAMATH_GPT_smallest_prime_fifth_term_of_arithmetic_sequence_l1081_108157

theorem smallest_prime_fifth_term_of_arithmetic_sequence :
  ∃ (a d : ℕ) (seq : ℕ → ℕ), 
    (∀ n, seq n = a + n * d) ∧ 
    (∀ n < 5, Prime (seq n)) ∧ 
    d = 6 ∧ 
    a = 5 ∧ 
    seq 4 = 29 := by
  sorry

end NUMINAMATH_GPT_smallest_prime_fifth_term_of_arithmetic_sequence_l1081_108157


namespace NUMINAMATH_GPT_sum_first_9_terms_arithmetic_sequence_l1081_108167

noncomputable def sum_of_first_n_terms (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

def arithmetic_sequence_term (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

theorem sum_first_9_terms_arithmetic_sequence :
  ∃ a_1 d : ℤ, (a_1 + arithmetic_sequence_term a_1 d 4 + arithmetic_sequence_term a_1 d 7 = 39) ∧
               (arithmetic_sequence_term a_1 d 3 + arithmetic_sequence_term a_1 d 6 + arithmetic_sequence_term a_1 d 9 = 27) ∧
               (sum_of_first_n_terms a_1 d 9 = 99) :=
by
  sorry

end NUMINAMATH_GPT_sum_first_9_terms_arithmetic_sequence_l1081_108167


namespace NUMINAMATH_GPT_sum_of_coords_of_four_points_l1081_108131

noncomputable def four_points_sum_coords : ℤ :=
  let y1 := 13 + 5
  let y2 := 13 - 5
  let x1 := 7 + 12
  let x2 := 7 - 12
  ((x2 + y2) + (x2 + y1) + (x1 + y2) + (x1 + y1))

theorem sum_of_coords_of_four_points : four_points_sum_coords = 80 :=
  by
    sorry

end NUMINAMATH_GPT_sum_of_coords_of_four_points_l1081_108131


namespace NUMINAMATH_GPT_find_a_l1081_108180

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a / ((Real.exp (2 * x)) - 1)

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, f a x = -f a (-x)) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1081_108180


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l1081_108138

theorem algebraic_expression_evaluation (a b : ℤ) (h : a - 3 * b = -3) : 5 - a + 3 * b = 8 :=
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l1081_108138


namespace NUMINAMATH_GPT_total_chairs_in_canteen_l1081_108150

theorem total_chairs_in_canteen (numRoundTables : ℕ) (numRectangularTables : ℕ) 
                                (chairsPerRoundTable : ℕ) (chairsPerRectangularTable : ℕ)
                                (h1 : numRoundTables = 2)
                                (h2 : numRectangularTables = 2)
                                (h3 : chairsPerRoundTable = 6)
                                (h4 : chairsPerRectangularTable = 7) : 
                                (numRoundTables * chairsPerRoundTable + numRectangularTables * chairsPerRectangularTable = 26) :=
by
  sorry

end NUMINAMATH_GPT_total_chairs_in_canteen_l1081_108150


namespace NUMINAMATH_GPT_range_of_function_l1081_108100

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_function : 
  (∀ x : ℝ, x ≠ -2 → f x ≠ 1) ∧
  (∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, f x = y) :=
sorry

end NUMINAMATH_GPT_range_of_function_l1081_108100


namespace NUMINAMATH_GPT_integer_solutions_l1081_108186

theorem integer_solutions (x y z : ℤ) : 
  x + y + z = 3 ∧ x^3 + y^3 + z^3 = 3 ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 4 ∧ y = 4 ∧ z = -5) ∨
  (x = 4 ∧ y = -5 ∧ z = 4) ∨
  (x = -5 ∧ y = 4 ∧ z = 4) := 
sorry

end NUMINAMATH_GPT_integer_solutions_l1081_108186


namespace NUMINAMATH_GPT_common_tangent_theorem_l1081_108105

-- Define the first circle with given equation (x+2)^2 + (y-2)^2 = 1
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 1

-- Define the second circle with given equation (x-2)^2 + (y-5)^2 = 16
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 16

-- Define a predicate that expresses the concept of common tangents between two circles
def common_tangents_count (circle1 circle2 : ℝ → ℝ → Prop) : ℕ := sorry

-- The statement to prove that the number of common tangents is 3
theorem common_tangent_theorem : common_tangents_count circle1 circle2 = 3 :=
by
  -- We would proceed with the proof if required, but we end with sorry as requested.
  sorry

end NUMINAMATH_GPT_common_tangent_theorem_l1081_108105


namespace NUMINAMATH_GPT_quadratic_roots_proof_l1081_108119

noncomputable def quadratic_roots_statement : Prop :=
  ∃ (x1 x2 : ℝ), 
    (x1 ≠ x2 ∨ x1 = x2) ∧ 
    (x1 = -20 ∧ x2 = -20) ∧ 
    (x1^2 + 40 * x1 + 300 = -100) ∧ 
    (x1 - x2 = 0 ∧ x1 * x2 = 400)  

theorem quadratic_roots_proof : quadratic_roots_statement :=
sorry

end NUMINAMATH_GPT_quadratic_roots_proof_l1081_108119


namespace NUMINAMATH_GPT_henri_total_time_l1081_108196

variable (m1 m2 : ℝ) (r w : ℝ)

theorem henri_total_time (H1 : m1 = 3.5) (H2 : m2 = 1.5) (H3 : r = 10) (H4 : w = 1800) :
    m1 + m2 + w / r / 60 = 8 := by
  sorry

end NUMINAMATH_GPT_henri_total_time_l1081_108196


namespace NUMINAMATH_GPT_lines_parallel_condition_l1081_108154

theorem lines_parallel_condition (a : ℝ) : 
  (a = 1) ↔ (∀ x y : ℝ, (a * x + 2 * y - 1 = 0 → x + (a + 1) * y + 4 = 0)) :=
sorry

end NUMINAMATH_GPT_lines_parallel_condition_l1081_108154


namespace NUMINAMATH_GPT_probability_third_smallest_is_five_l1081_108185

theorem probability_third_smallest_is_five :
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4)
  let probability := favorable_ways / total_ways
  probability = Rat.ofInt 35 / 132 :=
by
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4)
  let probability := favorable_ways / total_ways
  show probability = Rat.ofInt 35 / 132
  sorry

end NUMINAMATH_GPT_probability_third_smallest_is_five_l1081_108185


namespace NUMINAMATH_GPT_quadratic_complex_roots_condition_l1081_108165

theorem quadratic_complex_roots_condition (a : ℝ) :
  (∀ a, -2 ≤ a ∧ a ≤ 2 → (a^2 < 4)) ∧ 
  ¬(∀ a, (a^2 < 4) → -2 ≤ a ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_complex_roots_condition_l1081_108165


namespace NUMINAMATH_GPT_combined_weight_l1081_108189

theorem combined_weight (y z : ℝ) 
  (h_avg : (25 + 31 + 35 + 33) / 4 = (25 + 31 + 35 + 33 + y + z) / 6) :
  y + z = 62 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_l1081_108189


namespace NUMINAMATH_GPT_odd_nat_composite_iff_exists_a_l1081_108168

theorem odd_nat_composite_iff_exists_a (c : ℕ) (h_odd : c % 2 = 1) :
  (∃ a : ℕ, a ≤ c / 3 - 1 ∧ ∃ k : ℕ, (2*a - 1)^2 + 8*c = k^2) ↔
  ∃ d : ℕ, ∃ e : ℕ, d > 1 ∧ e > 1 ∧ d * e = c := 
sorry

end NUMINAMATH_GPT_odd_nat_composite_iff_exists_a_l1081_108168


namespace NUMINAMATH_GPT_factorize_expr_l1081_108192

theorem factorize_expr (x y : ℝ) : 3 * x^2 + 6 * x * y + 3 * y^2 = 3 * (x + y)^2 := 
  sorry

end NUMINAMATH_GPT_factorize_expr_l1081_108192


namespace NUMINAMATH_GPT_min_workers_to_make_profit_l1081_108155

theorem min_workers_to_make_profit :
  ∃ n : ℕ, 500 + 8 * 15 * n < 124 * n ∧ n = 126 :=
by
  sorry

end NUMINAMATH_GPT_min_workers_to_make_profit_l1081_108155


namespace NUMINAMATH_GPT_quadratic_inequality_empty_solution_set_l1081_108161

theorem quadratic_inequality_empty_solution_set (a b c : ℝ) (hₐ : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0 → False) ↔ (a > 0 ∧ (b^2 - 4 * a * c) ≤ 0) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_inequality_empty_solution_set_l1081_108161
