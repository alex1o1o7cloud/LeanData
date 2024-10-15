import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l2111_211180

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)

theorem problem_statement : ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l2111_211180


namespace NUMINAMATH_GPT_find_x_and_C_l2111_211107

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2 - x}

theorem find_x_and_C (x : ℝ) (C : Set ℝ) :
  B x ⊆ A x → B (-2) ∪ C = A (-2) → x = -2 ∧ C = {3} :=
by
  sorry

end NUMINAMATH_GPT_find_x_and_C_l2111_211107


namespace NUMINAMATH_GPT_new_average_is_10_5_l2111_211154

-- define the conditions
def average_of_eight_numbers (numbers : List ℝ) : Prop :=
  numbers.length = 8 ∧ (numbers.sum / 8) = 8

def add_four_to_five_numbers (numbers : List ℝ) (new_numbers : List ℝ) : Prop :=
  new_numbers = (numbers.take 5).map (λ x => x + 4) ++ numbers.drop 5

-- state the theorem
theorem new_average_is_10_5 (numbers new_numbers : List ℝ) 
  (h1 : average_of_eight_numbers numbers)
  (h2 : add_four_to_five_numbers numbers new_numbers) :
  (new_numbers.sum / 8) = 10.5 := 
by 
  sorry

end NUMINAMATH_GPT_new_average_is_10_5_l2111_211154


namespace NUMINAMATH_GPT_child_admission_charge_l2111_211197

-- Given conditions
variables (A C : ℝ) (T : ℝ := 3.25) (n : ℕ := 3)

-- Admission charge for an adult
def admission_charge_adult : ℝ := 1

-- Admission charge for a child
def admission_charge_child (C : ℝ) : ℝ := C

-- Total cost paid by adult with 3 children
def total_cost (A C : ℝ) (n : ℕ) : ℝ := A + n * C

-- The proof statement
theorem child_admission_charge (C : ℝ) : total_cost 1 C 3 = 3.25 -> C = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_child_admission_charge_l2111_211197


namespace NUMINAMATH_GPT_cookies_per_child_l2111_211120

def num_adults : ℕ := 4
def num_children : ℕ := 6
def cookies_jar1 : ℕ := 240
def cookies_jar2 : ℕ := 360
def cookies_jar3 : ℕ := 480

def fraction_eaten_jar1 : ℚ := 1 / 4
def fraction_eaten_jar2 : ℚ := 1 / 3
def fraction_eaten_jar3 : ℚ := 1 / 5

theorem cookies_per_child :
  let eaten_jar1 := fraction_eaten_jar1 * cookies_jar1
  let eaten_jar2 := fraction_eaten_jar2 * cookies_jar2
  let eaten_jar3 := fraction_eaten_jar3 * cookies_jar3
  let remaining_jar1 := cookies_jar1 - eaten_jar1
  let remaining_jar2 := cookies_jar2 - eaten_jar2
  let remaining_jar3 := cookies_jar3 - eaten_jar3
  let total_remaining_cookies := remaining_jar1 + remaining_jar2 + remaining_jar3
  let cookies_each_child := total_remaining_cookies / num_children
  cookies_each_child = 134 := by
  sorry

end NUMINAMATH_GPT_cookies_per_child_l2111_211120


namespace NUMINAMATH_GPT_two_digit_integers_satisfy_R_n_eq_R_n_plus_2_l2111_211149

def R (n : ℕ) : ℕ := 
  let remainders := List.range' 2 11 |>.map (λ k => n % k)
  remainders.sum

theorem two_digit_integers_satisfy_R_n_eq_R_n_plus_2 :
  let two_digit_numbers := List.range' 10 89
  (two_digit_numbers.filter (λ n => R n = R (n + 2))).length = 2 := 
by
  sorry

end NUMINAMATH_GPT_two_digit_integers_satisfy_R_n_eq_R_n_plus_2_l2111_211149


namespace NUMINAMATH_GPT_bowling_ball_weight_l2111_211114

noncomputable def weight_of_one_bowling_ball : ℕ := 20

theorem bowling_ball_weight (b c : ℕ) (h1 : 10 * b = 5 * c) (h2 : 3 * c = 120) : b = weight_of_one_bowling_ball := by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l2111_211114


namespace NUMINAMATH_GPT_digit_B_divisible_by_9_l2111_211189

theorem digit_B_divisible_by_9 (B : ℕ) (h1 : B ≤ 9) (h2 : (4 + B + B + 1 + 3) % 9 = 0) : B = 5 := 
by {
  /- Proof omitted -/
  sorry
}

end NUMINAMATH_GPT_digit_B_divisible_by_9_l2111_211189


namespace NUMINAMATH_GPT_average_price_of_remaining_packets_l2111_211104

variables (initial_avg_price : ℕ) (initial_packets : ℕ) (returned_packets : ℕ) (returned_avg_price : ℕ)

def total_initial_cost := initial_avg_price * initial_packets
def total_returned_cost := returned_avg_price * returned_packets
def remaining_packets := initial_packets - returned_packets
def total_remaining_cost := total_initial_cost initial_avg_price initial_packets - total_returned_cost returned_avg_price returned_packets
def remaining_avg_price := total_remaining_cost initial_avg_price initial_packets returned_avg_price returned_packets / remaining_packets initial_packets returned_packets

theorem average_price_of_remaining_packets :
  initial_avg_price = 20 →
  initial_packets = 5 →
  returned_packets = 2 →
  returned_avg_price = 32 →
  remaining_avg_price initial_avg_price initial_packets returned_avg_price returned_packets = 12
:=
by
  intros h1 h2 h3 h4
  rw [remaining_avg_price, total_remaining_cost, total_initial_cost, total_returned_cost]
  norm_num [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_average_price_of_remaining_packets_l2111_211104


namespace NUMINAMATH_GPT_distinct_natural_primes_l2111_211125

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem distinct_natural_primes :
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 5 ∧
  is_prime (a * b + c * d) ∧
  is_prime (a * c + b * d) ∧
  is_prime (a * d + b * c) := by
  sorry

end NUMINAMATH_GPT_distinct_natural_primes_l2111_211125


namespace NUMINAMATH_GPT_calen_pencils_loss_l2111_211146

theorem calen_pencils_loss
  (P_Candy : ℕ)
  (P_Caleb : ℕ)
  (P_Calen_original : ℕ)
  (P_Calen_after_loss : ℕ)
  (h1 : P_Candy = 9)
  (h2 : P_Caleb = 2 * P_Candy - 3)
  (h3 : P_Calen_original = P_Caleb + 5)
  (h4 : P_Calen_after_loss = 10) :
  P_Calen_original - P_Calen_after_loss = 10 := 
sorry

end NUMINAMATH_GPT_calen_pencils_loss_l2111_211146


namespace NUMINAMATH_GPT_complement_U_A_l2111_211169

-- Define the sets U and A
def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

-- State the theorem
theorem complement_U_A :
  U \ A = {0} :=
sorry

end NUMINAMATH_GPT_complement_U_A_l2111_211169


namespace NUMINAMATH_GPT_inequality_of_f_on_angles_l2111_211116

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function

-- Stating the properties of the function f
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, f (x + 1) = -f x
axiom decreasing_interval : ∀ x y : ℝ, (-3 ≤ x ∧ x < y ∧ y ≤ -2) → f x > f y

-- Stating the properties of the angles α and β
variables (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) (hαβ : α ≠ β)

-- The proof statement we want to prove
theorem inequality_of_f_on_angles : f (Real.sin α) > f (Real.cos β) :=
sorry -- The proof is omitted

end NUMINAMATH_GPT_inequality_of_f_on_angles_l2111_211116


namespace NUMINAMATH_GPT_correct_system_of_equations_l2111_211101

theorem correct_system_of_equations (x y : ℕ) (h1 : x + y = 145) (h2 : 10 * x + 12 * y = 1580) :
  (x + y = 145) ∧ (10 * x + 12 * y = 1580) :=
by
  sorry

end NUMINAMATH_GPT_correct_system_of_equations_l2111_211101


namespace NUMINAMATH_GPT_averagePricePerBook_l2111_211102

-- Define the prices and quantities from the first store
def firstStoreFictionBooks : ℕ := 25
def firstStoreFictionPrice : ℝ := 20
def firstStoreNonFictionBooks : ℕ := 15
def firstStoreNonFictionPrice : ℝ := 30
def firstStoreChildrenBooks : ℕ := 20
def firstStoreChildrenPrice : ℝ := 8

-- Define the prices and quantities from the second store
def secondStoreFictionBooks : ℕ := 10
def secondStoreFictionPrice : ℝ := 18
def secondStoreNonFictionBooks : ℕ := 20
def secondStoreNonFictionPrice : ℝ := 25
def secondStoreChildrenBooks : ℕ := 30
def secondStoreChildrenPrice : ℝ := 5

-- Definition of total books from first and second store
def totalBooks : ℕ :=
  firstStoreFictionBooks + firstStoreNonFictionBooks + firstStoreChildrenBooks +
  secondStoreFictionBooks + secondStoreNonFictionBooks + secondStoreChildrenBooks

-- Definition of the total cost from first and second store
def totalCost : ℝ :=
  (firstStoreFictionBooks * firstStoreFictionPrice) +
  (firstStoreNonFictionBooks * firstStoreNonFictionPrice) +
  (firstStoreChildrenBooks * firstStoreChildrenPrice) +
  (secondStoreFictionBooks * secondStoreFictionPrice) +
  (secondStoreNonFictionBooks * secondStoreNonFictionPrice) +
  (secondStoreChildrenBooks * secondStoreChildrenPrice)

-- Theorem: average price per book
theorem averagePricePerBook : (totalCost / totalBooks : ℝ) = 16.17 := by
  sorry

end NUMINAMATH_GPT_averagePricePerBook_l2111_211102


namespace NUMINAMATH_GPT_slope_of_line_determined_by_solutions_l2111_211147

theorem slope_of_line_determined_by_solutions :
  ∀ (x1 x2 y1 y2 : ℝ), 
  (4 / x1 + 6 / y1 = 0) → (4 / x2 + 6 / y2 = 0) →
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by
  intros x1 x2 y1 y2 h1 h2
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_slope_of_line_determined_by_solutions_l2111_211147


namespace NUMINAMATH_GPT_electronics_weight_l2111_211152

-- Define the initial conditions and the solution we want to prove.
theorem electronics_weight (B C E : ℕ) (k : ℕ) 
  (h1 : B = 7 * k) 
  (h2 : C = 4 * k) 
  (h3 : E = 3 * k) 
  (h4 : (B : ℚ) / (C - 8 : ℚ) = 2 * (B : ℚ) / (C : ℚ)) :
  E = 12 := 
sorry

end NUMINAMATH_GPT_electronics_weight_l2111_211152


namespace NUMINAMATH_GPT_suitable_graph_for_air_composition_is_pie_chart_l2111_211100

/-- The most suitable type of graph to visually represent the percentage 
of each component in the air is a pie chart, based on the given conditions. -/
theorem suitable_graph_for_air_composition_is_pie_chart 
  (bar_graph : Prop)
  (line_graph : Prop)
  (pie_chart : Prop)
  (histogram : Prop)
  (H1 : bar_graph → comparing_quantities)
  (H2 : line_graph → display_data_over_time)
  (H3 : pie_chart → show_proportions_of_whole)
  (H4 : histogram → show_distribution_of_dataset) 
  : suitable_graph_to_represent_percentage = pie_chart :=
sorry

end NUMINAMATH_GPT_suitable_graph_for_air_composition_is_pie_chart_l2111_211100


namespace NUMINAMATH_GPT_solution_set_non_empty_implies_a_gt_1_l2111_211122

theorem solution_set_non_empty_implies_a_gt_1 (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := 
  sorry

end NUMINAMATH_GPT_solution_set_non_empty_implies_a_gt_1_l2111_211122


namespace NUMINAMATH_GPT_moving_circle_fixed_point_coordinates_l2111_211193

theorem moving_circle_fixed_point_coordinates (m x y : Real) :
    (∀ m : ℝ, x^2 + y^2 - 2 * m * x - 4 * m * y + 6 * m - 2 = 0) →
    (x = 1 ∧ y = 1 ∨ x = 1 / 5 ∧ y = 7 / 5) :=
  by
    sorry

end NUMINAMATH_GPT_moving_circle_fixed_point_coordinates_l2111_211193


namespace NUMINAMATH_GPT_find_unsuitable_activity_l2111_211166

-- Definitions based on the conditions
def suitable_for_questionnaire (activity : String) : Prop :=
  activity = "D: The radiation produced by various mobile phones during use"

-- Question transformed into a statement to prove in Lean
theorem find_unsuitable_activity :
  suitable_for_questionnaire "D: The radiation produced by various mobile phones during use" :=
by
  sorry

end NUMINAMATH_GPT_find_unsuitable_activity_l2111_211166


namespace NUMINAMATH_GPT_cos_beta_value_l2111_211138

open Real

theorem cos_beta_value (α β : ℝ) (h1 : sin α = sqrt 5 / 5) (h2 : sin (α - β) = - sqrt 10 / 10) (h3 : 0 < α ∧ α < π / 2) (h4 : 0 < β ∧ β < π / 2) : cos β = sqrt 2 / 2 :=
by
sorry

end NUMINAMATH_GPT_cos_beta_value_l2111_211138


namespace NUMINAMATH_GPT_trig_identity_nec_but_not_suff_l2111_211155

open Real

theorem trig_identity_nec_but_not_suff (α β : ℝ) (k : ℤ) :
  (α + β = 2 * k * π + π / 6) → (sin α * cos β + cos α * sin β = 1 / 2) := by
  sorry

end NUMINAMATH_GPT_trig_identity_nec_but_not_suff_l2111_211155


namespace NUMINAMATH_GPT_final_price_after_discounts_l2111_211194

theorem final_price_after_discounts (original_price : ℝ)
  (first_discount_pct : ℝ) (second_discount_pct : ℝ) (third_discount_pct : ℝ) :
  original_price = 200 → 
  first_discount_pct = 0.40 → 
  second_discount_pct = 0.20 → 
  third_discount_pct = 0.10 → 
  (original_price * (1 - first_discount_pct) * (1 - second_discount_pct) * (1 - third_discount_pct) = 86.40) := 
by
  intros
  sorry

end NUMINAMATH_GPT_final_price_after_discounts_l2111_211194


namespace NUMINAMATH_GPT_point_on_graph_l2111_211153

def f (x : ℝ) : ℝ := -2 * x + 3

theorem point_on_graph (x y : ℝ) : 
  ( (x = 1 ∧ y = 1) ↔ y = f x ) :=
by 
  sorry

end NUMINAMATH_GPT_point_on_graph_l2111_211153


namespace NUMINAMATH_GPT_number_of_yellow_balloons_l2111_211184

-- Define the problem
theorem number_of_yellow_balloons :
  ∃ (Y B : ℕ), 
  B = Y + 1762 ∧ 
  Y + B = 10 * 859 ∧ 
  Y = 3414 :=
by
  -- Proof is skipped, so we use sorry
  sorry

end NUMINAMATH_GPT_number_of_yellow_balloons_l2111_211184


namespace NUMINAMATH_GPT_cara_total_bread_l2111_211139

variable (L B : ℕ)  -- Let L and B be the amount of bread for lunch and breakfast, respectively

theorem cara_total_bread :
  (dinner = 240) → 
  (dinner = 8 * L) → 
  (dinner = 6 * B) → 
  (total_bread = dinner + L + B) → 
  total_bread = 310 :=
by
  intros
  -- Here you'd begin your proof, implementing each given condition
  sorry

end NUMINAMATH_GPT_cara_total_bread_l2111_211139


namespace NUMINAMATH_GPT_digit_sum_divisible_by_9_l2111_211157

theorem digit_sum_divisible_by_9 (n : ℕ) (h : n < 10) : 
  (8 + 6 + 5 + n + 7 + 4 + 3 + 2) % 9 = 0 ↔ n = 1 := 
by sorry 

end NUMINAMATH_GPT_digit_sum_divisible_by_9_l2111_211157


namespace NUMINAMATH_GPT_roses_in_december_l2111_211195

theorem roses_in_december (rOct rNov rJan rFeb : ℕ) 
  (hOct : rOct = 108)
  (hNov : rNov = 120)
  (hJan : rJan = 144)
  (hFeb : rFeb = 156)
  (pattern : (rNov - rOct = 12 ∨ rNov - rOct = 24) ∧ 
             (rJan - rNov = 12 ∨ rJan - rNov = 24) ∧
             (rFeb - rJan = 12 ∨ rFeb - rJan = 24) ∧ 
             (∀ m n, (m - n = 12 ∨ m - n = 24) → 
               ((rNov - rOct) ≠ (rJan - rNov) ↔ 
               (rJan - rNov) ≠ (rFeb - rJan)))) : 
  ∃ rDec : ℕ, rDec = 132 := 
by {
  sorry
}

end NUMINAMATH_GPT_roses_in_december_l2111_211195


namespace NUMINAMATH_GPT_abc_inequality_l2111_211106

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a * b + b * c + c * a ≥ a * Real.sqrt (b * c) + b * Real.sqrt (a * c) + c * Real.sqrt (a * b) :=
sorry

end NUMINAMATH_GPT_abc_inequality_l2111_211106


namespace NUMINAMATH_GPT_compare_y1_y2_l2111_211164

theorem compare_y1_y2 (m y1 y2 : ℝ) 
  (h1 : y1 = (-1)^2 - 2*(-1) + m) 
  (h2 : y2 = 2^2 - 2*2 + m) : 
  y1 > y2 := 
sorry

end NUMINAMATH_GPT_compare_y1_y2_l2111_211164


namespace NUMINAMATH_GPT_ellipse_sum_l2111_211112

theorem ellipse_sum (h k a b : ℝ) (h_val : h = 3) (k_val : k = -5) (a_val : a = 6) (b_val : b = 2) : h + k + a + b = 6 :=
by
  rw [h_val, k_val, a_val, b_val]
  norm_num

end NUMINAMATH_GPT_ellipse_sum_l2111_211112


namespace NUMINAMATH_GPT_solution_set_abs_inequality_l2111_211126

theorem solution_set_abs_inequality :
  { x : ℝ | |x - 2| - |2 * x - 1| > 0 } = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_abs_inequality_l2111_211126


namespace NUMINAMATH_GPT_largest_circle_area_215_l2111_211141

theorem largest_circle_area_215
  (length width : ℝ)
  (h1 : length = 16)
  (h2 : width = 10)
  (P : ℝ := 2 * (length + width))
  (C : ℝ := P)
  (r : ℝ := C / (2 * Real.pi))
  (A : ℝ := Real.pi * r^2) :
  round A = 215 := by sorry

end NUMINAMATH_GPT_largest_circle_area_215_l2111_211141


namespace NUMINAMATH_GPT_balloon_total_l2111_211186

def total_balloons (joan_balloons melanie_balloons : ℕ) : ℕ :=
  joan_balloons + melanie_balloons

theorem balloon_total :
  total_balloons 40 41 = 81 :=
by
  sorry

end NUMINAMATH_GPT_balloon_total_l2111_211186


namespace NUMINAMATH_GPT_valid_outfit_combinations_l2111_211129

theorem valid_outfit_combinations (shirts pants hats shoes : ℕ) (colors : ℕ) 
  (h₁ : shirts = 6) (h₂ : pants = 6) (h₃ : hats = 6) (h₄ : shoes = 6) (h₅ : colors = 6) :
  ∀ (valid_combinations : ℕ),
  (valid_combinations = colors * (colors - 1) * (colors - 2) * (colors - 3)) → valid_combinations = 360 := 
by
  intros valid_combinations h_valid_combinations
  sorry

end NUMINAMATH_GPT_valid_outfit_combinations_l2111_211129


namespace NUMINAMATH_GPT_farmer_sowed_buckets_l2111_211150

-- Define the initial and final buckets of seeds
def initial_buckets : ℝ := 8.75
def final_buckets : ℝ := 6.00

-- The goal: prove the number of buckets sowed is 2.75
theorem farmer_sowed_buckets : initial_buckets - final_buckets = 2.75 := by
  sorry

end NUMINAMATH_GPT_farmer_sowed_buckets_l2111_211150


namespace NUMINAMATH_GPT_overall_average_score_l2111_211113

theorem overall_average_score 
  (M : ℝ) (E : ℝ) (m e : ℝ)
  (hM : M = 82)
  (hE : E = 75)
  (hRatio : m / e = 5 / 3) :
  (M * m + E * e) / (m + e) = 79.375 := 
by
  sorry

end NUMINAMATH_GPT_overall_average_score_l2111_211113


namespace NUMINAMATH_GPT_number_of_integer_values_of_a_l2111_211148

theorem number_of_integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^2 + a * x + 9 * a = 0) ↔ 
  (∃ (a_values : Finset ℤ), a_values.card = 6 ∧ ∀ a ∈ a_values, ∃ x : ℤ, x^2 + a * x + 9 * a = 0) :=
by
  sorry

end NUMINAMATH_GPT_number_of_integer_values_of_a_l2111_211148


namespace NUMINAMATH_GPT_apples_needed_per_month_l2111_211124

theorem apples_needed_per_month (chandler_apples_per_week : ℕ) (lucy_apples_per_week : ℕ) (weeks_per_month : ℕ)
  (h1 : chandler_apples_per_week = 23)
  (h2 : lucy_apples_per_week = 19)
  (h3 : weeks_per_month = 4) :
  (chandler_apples_per_week + lucy_apples_per_week) * weeks_per_month = 168 :=
by
  sorry

end NUMINAMATH_GPT_apples_needed_per_month_l2111_211124


namespace NUMINAMATH_GPT_tan_tan_lt_half_l2111_211119

noncomputable def tan_half (x : ℝ) : ℝ := Real.tan (x / 2)

theorem tan_tan_lt_half (a b c α β : ℝ) (h1: a + b < 3 * c) (h2: tan_half α * tan_half β = (a + b - c) / (a + b + c)) :
  tan_half α * tan_half β < 1 / 2 := 
sorry

end NUMINAMATH_GPT_tan_tan_lt_half_l2111_211119


namespace NUMINAMATH_GPT_milton_zoology_books_l2111_211161

variable (Z : ℕ)
variable (total_books botany_books : ℕ)

theorem milton_zoology_books (h1 : total_books = 960)
    (h2 : botany_books = 7 * Z)
    (h3 : total_books = Z + botany_books) :
    Z = 120 := by
  sorry

end NUMINAMATH_GPT_milton_zoology_books_l2111_211161


namespace NUMINAMATH_GPT_product_of_four_consecutive_integers_divisible_by_twelve_l2111_211158

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end NUMINAMATH_GPT_product_of_four_consecutive_integers_divisible_by_twelve_l2111_211158


namespace NUMINAMATH_GPT_harry_blue_weights_l2111_211130

theorem harry_blue_weights (B : ℕ) 
  (h1 : 2 * B + 17 = 25) : B = 4 :=
by {
  -- proof code here
  sorry
}

end NUMINAMATH_GPT_harry_blue_weights_l2111_211130


namespace NUMINAMATH_GPT_distance_between_C_and_A_l2111_211192

theorem distance_between_C_and_A 
    (A B C : Type)
    (d_AB : ℝ) (d_BC : ℝ)
    (h1 : d_AB = 8)
    (h2 : d_BC = 10) :
    ∃ x : ℝ, 2 ≤ x ∧ x ≤ 18 ∧ ¬ (∃ y : ℝ, y = x) :=
sorry

end NUMINAMATH_GPT_distance_between_C_and_A_l2111_211192


namespace NUMINAMATH_GPT_remainder_of_N_l2111_211159

-- Definition of the sequence constraints
def valid_sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧ (∀ i, a i < 512) ∧ (∀ k, 1 ≤ k → k ≤ 9 → ∃ m, 0 ≤ m ∧ m ≤ k - 1 ∧ ((a k - 2 * a m) * (a k - 2 * a m - 1) = 0))

-- Defining N as the number of sequences that are valid.
noncomputable def N : ℕ :=
  Nat.factorial 10 - 2^9

-- The goal is to prove that N mod 1000 is 288
theorem remainder_of_N : N % 1000 = 288 :=
  sorry

end NUMINAMATH_GPT_remainder_of_N_l2111_211159


namespace NUMINAMATH_GPT_total_birds_on_fence_l2111_211170

-- Definitions for the problem conditions
def initial_birds : ℕ := 12
def new_birds : ℕ := 8

-- Theorem to state that the total number of birds on the fence is 20
theorem total_birds_on_fence : initial_birds + new_birds = 20 :=
by
  -- Skip the proof as required
  sorry

end NUMINAMATH_GPT_total_birds_on_fence_l2111_211170


namespace NUMINAMATH_GPT_correct_A_correct_B_intersection_A_B_complement_B_l2111_211142

noncomputable def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem correct_A : A = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem correct_B : B = {x : ℝ | 1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem intersection_A_B : (A ∩ B) = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem complement_B : (Bᶜ) = {x : ℝ | x < 1 ∨ x > 4} :=
by
  sorry

end NUMINAMATH_GPT_correct_A_correct_B_intersection_A_B_complement_B_l2111_211142


namespace NUMINAMATH_GPT_vector_t_perpendicular_l2111_211117

theorem vector_t_perpendicular (t : ℝ) :
  let a := (2, 4)
  let b := (-1, 1)
  let c := (2 + t, 4 - t)
  b.1 * c.1 + b.2 * c.2 = 0 → t = 1 := by
  sorry

end NUMINAMATH_GPT_vector_t_perpendicular_l2111_211117


namespace NUMINAMATH_GPT_S_4n_l2111_211143

variable {a : ℕ → ℕ}
variable (S : ℕ → ℝ)
variable (n : ℕ)
variable (r : ℝ)
variable (a1 : ℝ)

-- Conditions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * r
axiom positive_terms : ∀ n, 0 < a n
axiom sum_n : S n = a1 * (1 - r^n) / (1 - r)
axiom sum_3n : S (3 * n) = 14
axiom sum_n_value : S n = 2

-- Theorem
theorem S_4n : S (4 * n) = 30 :=
sorry

end NUMINAMATH_GPT_S_4n_l2111_211143


namespace NUMINAMATH_GPT_general_term_formaula_sum_of_seq_b_l2111_211110

noncomputable def seq_a (n : ℕ) := 2 * n + 1

noncomputable def seq_b (n : ℕ) := 1 / ((seq_a n)^2 - 1)

noncomputable def sum_seq_a (n : ℕ) := (Finset.range n).sum seq_a

noncomputable def sum_seq_b (n : ℕ) := (Finset.range n).sum seq_b

theorem general_term_formaula (n : ℕ) (h : n > 0) :
  (∀ n, (seq_a n) > 0) ∧ (∀ n, (seq_a n)^2 + 2 * (seq_a n) = 4 * (sum_seq_a n) + 3) →
  seq_a n = 2 * n + 1 :=
by
  intros
  sorry

theorem sum_of_seq_b (n : ℕ) (h : n > 0) :
  (∀ n, (seq_a n) > 0) ∧ (∀ n, (seq_a n)^2 + 2 * (seq_a n) = 4 * (sum_seq_a n) + 3) →
  sum_seq_b n = n / (4 * (n + 1)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_general_term_formaula_sum_of_seq_b_l2111_211110


namespace NUMINAMATH_GPT_exactly_one_even_contradiction_assumption_l2111_211160

variable (a b c : ℕ)

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
  (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0)

def conclusion (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (c % 2 = 0 ∧ a % 2 = 0)

theorem exactly_one_even_contradiction_assumption :
    exactly_one_even a b c ↔ ¬ conclusion a b c :=
by
  sorry

end NUMINAMATH_GPT_exactly_one_even_contradiction_assumption_l2111_211160


namespace NUMINAMATH_GPT_find_g_at_4_l2111_211156

def g (x : ℝ) : ℝ := sorry

theorem find_g_at_4 (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x ^ 2) : g 4 = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_find_g_at_4_l2111_211156


namespace NUMINAMATH_GPT_degree_of_monomial_l2111_211190

def degree (m : String) : Nat :=  -- Placeholder type, replace with appropriate type that represents a monomial
  sorry  -- Logic to compute the degree would go here, if required for full implementation

theorem degree_of_monomial : degree "-(3/5) * a * b^2" = 3 := by
  sorry

end NUMINAMATH_GPT_degree_of_monomial_l2111_211190


namespace NUMINAMATH_GPT_total_distance_covered_l2111_211177

theorem total_distance_covered :
  let speed_fox := 50       -- km/h
  let speed_rabbit := 60    -- km/h
  let speed_deer := 80      -- km/h
  let time_hours := 2       -- hours
  let distance_fox := speed_fox * time_hours
  let distance_rabbit := speed_rabbit * time_hours
  let distance_deer := speed_deer * time_hours
  distance_fox + distance_rabbit + distance_deer = 380 := by
sorry

end NUMINAMATH_GPT_total_distance_covered_l2111_211177


namespace NUMINAMATH_GPT_central_angle_of_sector_l2111_211183

noncomputable def central_angle (radius perimeter: ℝ) : ℝ :=
  ((perimeter - 2 * radius) / (2 * Real.pi * radius)) * 360

theorem central_angle_of_sector :
  central_angle 28 144 = 180.21 :=
by
  simp [central_angle]
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l2111_211183


namespace NUMINAMATH_GPT_calculate_exponent_product_l2111_211108

theorem calculate_exponent_product :
  (2^0.5) * (2^0.3) * (2^0.2) * (2^0.1) * (2^0.9) = 4 :=
by
  sorry

end NUMINAMATH_GPT_calculate_exponent_product_l2111_211108


namespace NUMINAMATH_GPT_total_votes_l2111_211121

theorem total_votes (votes_brenda : ℕ) (total_votes : ℕ) 
  (h1 : votes_brenda = 50) 
  (h2 : votes_brenda = (1/4 : ℚ) * total_votes) : 
  total_votes = 200 :=
by 
  sorry

end NUMINAMATH_GPT_total_votes_l2111_211121


namespace NUMINAMATH_GPT_largest_three_digit_multiple_of_17_l2111_211131

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_three_digit_multiple_of_17_l2111_211131


namespace NUMINAMATH_GPT_prob_no_decrease_white_in_A_is_correct_l2111_211178

-- Define the conditions of the problem
def bagA_white : ℕ := 3
def bagA_black : ℕ := 5
def bagB_white : ℕ := 4
def bagB_black : ℕ := 6

-- Define the probabilities involved
def prob_draw_black_from_A : ℚ := 5 / 8
def prob_draw_white_from_A : ℚ := 3 / 8
def prob_put_white_back_into_A_conditioned_on_white_drawn : ℚ := 5 / 11

-- Calculate the combined probability
def prob_no_decrease_white_in_A : ℚ := prob_draw_black_from_A + prob_draw_white_from_A * prob_put_white_back_into_A_conditioned_on_white_drawn

-- Prove the probability is as expected
theorem prob_no_decrease_white_in_A_is_correct : prob_no_decrease_white_in_A = 35 / 44 := by
  sorry

end NUMINAMATH_GPT_prob_no_decrease_white_in_A_is_correct_l2111_211178


namespace NUMINAMATH_GPT_point_on_x_axis_l2111_211182

theorem point_on_x_axis (m : ℤ) (hx : 2 + m = 0) : (m - 3, 2 + m) = (-5, 0) :=
by sorry

end NUMINAMATH_GPT_point_on_x_axis_l2111_211182


namespace NUMINAMATH_GPT_matthew_crackers_left_l2111_211109

-- Definition of the conditions:
def initial_crackers := 23
def friends := 2
def crackers_eaten_per_friend := 6

-- Calculate the number of crackers Matthew has left:
def crackers_left (total_crackers : ℕ) (num_friends : ℕ) (eaten_per_friend : ℕ) : ℕ :=
  let crackers_given := (total_crackers - total_crackers % num_friends)
  let kept_by_matthew := total_crackers % num_friends
  let remaining_with_friends := (crackers_given / num_friends - eaten_per_friend) * num_friends
  kept_by_matthew + remaining_with_friends
  
-- Theorem to prove:
theorem matthew_crackers_left : crackers_left initial_crackers friends crackers_eaten_per_friend = 11 := by
  sorry

end NUMINAMATH_GPT_matthew_crackers_left_l2111_211109


namespace NUMINAMATH_GPT_largest_divisor_of_Pn_for_even_n_l2111_211127

def P (n : ℕ) : ℕ := 
  (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9)

theorem largest_divisor_of_Pn_for_even_n : 
  ∀ (n : ℕ), (0 < n ∧ n % 2 = 0) → ∃ d, d = 15 ∧ d ∣ P n :=
by
  intro n h
  sorry

end NUMINAMATH_GPT_largest_divisor_of_Pn_for_even_n_l2111_211127


namespace NUMINAMATH_GPT_full_time_score_l2111_211167

variables (x : ℕ)

def half_time_score_visitors := 14
def half_time_score_home := 9
def visitors_full_time_score := half_time_score_visitors + x
def home_full_time_score := half_time_score_home + 2 * x
def home_team_win_by_one := home_full_time_score = visitors_full_time_score + 1

theorem full_time_score 
  (h : home_team_win_by_one) : 
  visitors_full_time_score = 20 ∧ home_full_time_score = 21 :=
by
  sorry

end NUMINAMATH_GPT_full_time_score_l2111_211167


namespace NUMINAMATH_GPT_smallest_six_digit_negative_integer_congruent_to_five_mod_17_l2111_211162

theorem smallest_six_digit_negative_integer_congruent_to_five_mod_17 :
  ∃ x : ℤ, x < -100000 ∧ x ≥ -999999 ∧ x % 17 = 5 ∧ x = -100011 :=
by
  sorry

end NUMINAMATH_GPT_smallest_six_digit_negative_integer_congruent_to_five_mod_17_l2111_211162


namespace NUMINAMATH_GPT_sum_of_coefficients_l2111_211145

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) 
  (h : (1 + 2*x)^7 = a + a₁*(1 - x) + a₂*(1 - x)^2 + a₃*(1 - x)^3 + a₄*(1 - x)^4 + a₅*(1 - x)^5 + a₆*(1 - x)^6 + a₇*(1 - x)^7) :
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 1 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2111_211145


namespace NUMINAMATH_GPT_find_a_b_f_inequality_l2111_211172

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

-- a == 1 and b == 1 from the given conditions
theorem find_a_b (e : ℝ) (h_e : e = Real.exp 1) (b : ℝ) (a : ℝ) 
  (h_tangent : ∀ x, f x a = (e - 2) * x + b → a = 1 ∧ b = 1) : a = 1 ∧ b = 1 :=
sorry

-- prove f(x) > x^2 + 4x - 14 for x >= 0
theorem f_inequality (e : ℝ) (h_e : e = Real.exp 1) :
  ∀ x : ℝ, 0 ≤ x → f x 1 > x^2 + 4 * x - 14 :=
sorry

end NUMINAMATH_GPT_find_a_b_f_inequality_l2111_211172


namespace NUMINAMATH_GPT_nat_triple_solution_l2111_211188

theorem nat_triple_solution (x y n : ℕ) :
  (x! + y!) / n! = 3^n ↔ (x = 1 ∧ y = 2 ∧ n = 1) ∨ (x = 2 ∧ y = 1 ∧ n = 1) := 
by
  sorry

end NUMINAMATH_GPT_nat_triple_solution_l2111_211188


namespace NUMINAMATH_GPT_total_candy_l2111_211151

/-- Bobby ate 26 pieces of candy initially. -/
def initial_candy : ℕ := 26

/-- Bobby ate 17 more pieces of candy thereafter. -/
def more_candy : ℕ := 17

/-- Prove that the total number of pieces of candy Bobby ate is 43. -/
theorem total_candy : initial_candy + more_candy = 43 := by
  -- The total number of candies should be 26 + 17 which is 43
  sorry

end NUMINAMATH_GPT_total_candy_l2111_211151


namespace NUMINAMATH_GPT_inequality_proof_l2111_211165

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (3 / (a^3 + b^3 + c^3)) ≤ 
  ((1 / (a^3 + b^3 + abc)) + (1 / (b^3 + c^3 + abc)) + (1 / (c^3 + a^3 + abc))) ∧ 
  ((1 / (a^3 + b^3 + abc)) + (1 / (b^3 + c^3 + abc)) + (1 / (c^3 + a^3 + abc)) ≤ (1 / (abc))) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l2111_211165


namespace NUMINAMATH_GPT_distance_between_Sasha_and_Koyla_is_19m_l2111_211175

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end NUMINAMATH_GPT_distance_between_Sasha_and_Koyla_is_19m_l2111_211175


namespace NUMINAMATH_GPT_intersection_complement_l2111_211135

open Set

theorem intersection_complement (U A B : Set ℕ) (hU : U = {x | x ≤ 6}) (hA : A = {1, 3, 5}) (hB : B = {4, 5, 6}) :
  (U \ A) ∩ B = {4, 6} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l2111_211135


namespace NUMINAMATH_GPT_problem_l2111_211199

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem problem (f : ℝ → ℝ) (h : isOddFunction f) : 
  f (-2012) + f (-2011) + f 0 + f 2011 + f 2012 = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2111_211199


namespace NUMINAMATH_GPT_tino_jellybeans_l2111_211136

variable (Tino Lee Arnold : ℕ)

-- Conditions
variable (h1 : Tino = Lee + 24)
variable (h2 : Arnold = Lee / 2)
variable (h3 : Arnold = 5)

-- Prove Tino has 34 jellybeans
theorem tino_jellybeans : Tino = 34 :=
by
  sorry

end NUMINAMATH_GPT_tino_jellybeans_l2111_211136


namespace NUMINAMATH_GPT_cost_of_children_ticket_l2111_211168

theorem cost_of_children_ticket (total_cost : ℝ) (cost_adult_ticket : ℝ) (num_total_tickets : ℕ) (num_adult_tickets : ℕ) (cost_children_ticket : ℝ) :
  total_cost = 119 ∧ cost_adult_ticket = 21 ∧ num_total_tickets = 7 ∧ num_adult_tickets = 4 -> cost_children_ticket = 11.67 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_cost_of_children_ticket_l2111_211168


namespace NUMINAMATH_GPT_possible_values_of_k_l2111_211128

theorem possible_values_of_k (k : ℕ) (N : ℕ) (h₁ : (k * (k + 1)) / 2 = N^2) (h₂ : N < 100) :
  k = 1 ∨ k = 8 ∨ k = 49 :=
sorry

end NUMINAMATH_GPT_possible_values_of_k_l2111_211128


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l2111_211171

theorem simplify_and_evaluate_expr (a b : ℕ) (h₁ : a = 2) (h₂ : b = 2023) : 
  (a + b)^2 + b * (a - b) - 3 * a * b = 4 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l2111_211171


namespace NUMINAMATH_GPT_lucy_current_fish_l2111_211118

-- Definitions based on conditions in the problem
def total_fish : ℕ := 280
def fish_needed_to_buy : ℕ := 68

-- Proving the number of fish Lucy currently has
theorem lucy_current_fish : total_fish - fish_needed_to_buy = 212 :=
by
  sorry

end NUMINAMATH_GPT_lucy_current_fish_l2111_211118


namespace NUMINAMATH_GPT_problem_statement_l2111_211185

theorem problem_statement (x : ℝ) (h₀ : x > 0) (n : ℕ) (hn : n > 0) :
  (x + (n^n : ℝ) / x^n) ≥ (n + 1) :=
sorry

end NUMINAMATH_GPT_problem_statement_l2111_211185


namespace NUMINAMATH_GPT_functional_equation_solution_l2111_211123

noncomputable def func_form (f : ℝ → ℝ) : Prop :=
  ∃ α β : ℝ, (α = 1 ∨ α = -1 ∨ α = 0) ∧ (∀ x, f x = α * x + β ∨ f x = α * x ^ 3 + β)

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, (f a - f b) * (f b - f c) * (f c - f a) = f (a * b ^ 2 + b * c ^ 2 + c * a ^ 2) - f (a ^ 2 * b + b ^ 2 * c + c ^ 2 * a)) →
  func_form f :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l2111_211123


namespace NUMINAMATH_GPT_galaxy_destruction_probability_l2111_211111

theorem galaxy_destruction_probability :
  let m := 45853
  let n := 65536
  m + n = 111389 :=
by
  sorry

end NUMINAMATH_GPT_galaxy_destruction_probability_l2111_211111


namespace NUMINAMATH_GPT_minimum_distance_sum_squared_l2111_211173

variable (P : ℝ × ℝ)
variable (F₁ F₂ : ℝ × ℝ)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / 4) + y^2 = 1

def distance_squared (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2

theorem minimum_distance_sum_squared
  (hP : on_ellipse P)
  (hF1 : F₁ = (2, 0) ∨ F₁ = (-2, 0)) -- Assuming standard position of foci
  (hF2 : F₂ = (2, 0) ∨ F₂ = (-2, 0)) :
  ∃ P : ℝ × ℝ, on_ellipse P ∧ F₁ ≠ F₂ → distance_squared P F₁ + distance_squared P F₂ = 8 :=
by
  sorry

end NUMINAMATH_GPT_minimum_distance_sum_squared_l2111_211173


namespace NUMINAMATH_GPT_average_of_r_s_t_l2111_211176

theorem average_of_r_s_t (r s t : ℝ) (h : (5/4) * (r + s + t) = 20) : (r + s + t) / 3 = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_of_r_s_t_l2111_211176


namespace NUMINAMATH_GPT_problem_statement_l2111_211137

open Real

variable (a b c : ℝ)

theorem problem_statement
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_cond : a + b + c + a * b * c = 4) :
  (1 + a / b + c * a) * (1 + b / c + a * b) * (1 + c / a + b * c) ≥ 27 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2111_211137


namespace NUMINAMATH_GPT_binom_1000_1000_and_999_l2111_211163

theorem binom_1000_1000_and_999 :
  (Nat.choose 1000 1000 = 1) ∧ (Nat.choose 1000 999 = 1000) :=
by
  sorry

end NUMINAMATH_GPT_binom_1000_1000_and_999_l2111_211163


namespace NUMINAMATH_GPT_factorize_expression_l2111_211140

theorem factorize_expression (a b : ℝ) : 
  a^2 * b + 2 * a * b^2 + b^3 = b * (a + b)^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_factorize_expression_l2111_211140


namespace NUMINAMATH_GPT_exist_positive_real_x_l2111_211103

theorem exist_positive_real_x (x : ℝ) (hx1 : 0 < x) (hx2 : Nat.floor x * x = 90) : x = 10 := 
sorry

end NUMINAMATH_GPT_exist_positive_real_x_l2111_211103


namespace NUMINAMATH_GPT_solve_diff_l2111_211191

-- Definitions based on conditions
def equation (e y : ℝ) : Prop := y^2 + e^2 = 3 * e * y + 1

theorem solve_diff (e a b : ℝ) (h1 : equation e a) (h2 : equation e b) (h3 : a ≠ b) : 
  |a - b| = Real.sqrt (5 * e^2 + 4) := 
sorry

end NUMINAMATH_GPT_solve_diff_l2111_211191


namespace NUMINAMATH_GPT_sum_of_roots_eq_14_l2111_211105

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end NUMINAMATH_GPT_sum_of_roots_eq_14_l2111_211105


namespace NUMINAMATH_GPT_least_positive_integer_divisible_by_5_to_15_l2111_211134

def is_divisible_by_all (n : ℕ) (l : List ℕ) : Prop :=
  ∀ m ∈ l, m ∣ n

theorem least_positive_integer_divisible_by_5_to_15 :
  ∃ n : ℕ, n > 0 ∧ is_divisible_by_all n [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] ∧
  ∀ m : ℕ, m > 0 ∧ is_divisible_by_all m [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] → n ≤ m ∧ n = 360360 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_divisible_by_5_to_15_l2111_211134


namespace NUMINAMATH_GPT_intersect_A_B_l2111_211115

def A : Set ℝ := {x | 1/x < 1}
def B : Set ℝ := {-1, 0, 1, 2}
def intersection_result : Set ℝ := {-1, 2}

theorem intersect_A_B : A ∩ B = intersection_result :=
by
  sorry

end NUMINAMATH_GPT_intersect_A_B_l2111_211115


namespace NUMINAMATH_GPT_inequality_problem_l2111_211198

variable {a b c : ℝ}

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end NUMINAMATH_GPT_inequality_problem_l2111_211198


namespace NUMINAMATH_GPT_slower_train_speed_l2111_211181

theorem slower_train_speed (length_train : ℕ) (speed_fast : ℕ) (time_seconds : ℕ) (distance_meters : ℕ): 
  (length_train = 150) → 
  (speed_fast = 46) → 
  (time_seconds = 108) → 
  (distance_meters = 300) → 
  (distance_meters = (speed_fast - speed_slow) * 5 / 18 * time_seconds) → 
  speed_slow = 36 := 
by
    intros h1 h2 h3 h4 h5
    sorry

end NUMINAMATH_GPT_slower_train_speed_l2111_211181


namespace NUMINAMATH_GPT_remainder_of_largest_divided_by_next_largest_l2111_211179

/-
  Conditions:
  Let a = 10, b = 11, c = 12, d = 13.
  The largest number is d (13) and the next largest number is c (12).

  Question:
  What is the remainder when the largest number is divided by the next largest number?

  Answer:
  The remainder is 1.
-/

theorem remainder_of_largest_divided_by_next_largest :
  let a := 10 
  let b := 11
  let c := 12
  let d := 13
  d % c = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_largest_divided_by_next_largest_l2111_211179


namespace NUMINAMATH_GPT_reciprocal_of_neg_2023_l2111_211132

-- Define the number and its proposed reciprocal
def x : ℤ := -2023
def r : ℚ := -1 / 2023

-- State the theorem that the reciprocal of x is r
theorem reciprocal_of_neg_2023: x * r = 1 := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_2023_l2111_211132


namespace NUMINAMATH_GPT_min_questions_to_determine_number_l2111_211174

theorem min_questions_to_determine_number : 
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 50) → 
  ∃ (q : ℕ), q = 15 ∧ 
  ∀ (primes : ℕ → Prop), 
  (∀ p, primes p → Nat.Prime p ∧ p ≤ 50) → 
  (∀ p, primes p → (n % p = 0 ↔ p ∣ n)) → 
  (∃ m, (∀ k, k < m → primes k → k ∣ n)) :=
sorry

end NUMINAMATH_GPT_min_questions_to_determine_number_l2111_211174


namespace NUMINAMATH_GPT_simplify_eval_l2111_211187

theorem simplify_eval (a : ℝ) (h : a = Real.sqrt 3 / 3) : (a + 1) ^ 2 + a * (1 - a) = Real.sqrt 3 + 1 := 
by
  sorry

end NUMINAMATH_GPT_simplify_eval_l2111_211187


namespace NUMINAMATH_GPT_James_age_l2111_211133

-- Defining variables
variables (James John Tim : ℕ)
variables (h1 : James + 12 = John)
variables (h2 : Tim + 5 = 2 * John)
variables (h3 : Tim = 79)

-- Statement to prove James' age
theorem James_age : James = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_James_age_l2111_211133


namespace NUMINAMATH_GPT_second_number_is_22_l2111_211196

theorem second_number_is_22 (x second_number : ℕ) : 
  (x + second_number = 33) → 
  (second_number = 2 * x) → 
  second_number = 22 :=
by
  intros h_sum h_double
  sorry

end NUMINAMATH_GPT_second_number_is_22_l2111_211196


namespace NUMINAMATH_GPT_tilted_rectangle_l2111_211144

theorem tilted_rectangle (VWYZ : Type) (YW ZV : ℝ) (ZY VW : ℝ) (W_above_horizontal : ℝ) (Z_height : ℝ) (x : ℝ) :
  YW = 100 → ZV = 100 → ZY = 150 → VW = 150 → W_above_horizontal = 20 → Z_height = (100 + x) →
  x = 67 :=
by
  sorry

end NUMINAMATH_GPT_tilted_rectangle_l2111_211144
