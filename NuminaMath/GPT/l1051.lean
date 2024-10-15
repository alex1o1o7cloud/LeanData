import Mathlib

namespace NUMINAMATH_GPT_repeating_block_length_7_div_13_l1051_105196

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end NUMINAMATH_GPT_repeating_block_length_7_div_13_l1051_105196


namespace NUMINAMATH_GPT_definite_integral_cos_exp_l1051_105189

open Real

theorem definite_integral_cos_exp :
  ∫ x in -π..0, (cos x + exp x) = 1 - (1 / exp π) :=
by
  sorry

end NUMINAMATH_GPT_definite_integral_cos_exp_l1051_105189


namespace NUMINAMATH_GPT_maximum_daily_sales_revenue_l1051_105179

noncomputable def P (t : ℕ) : ℤ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

noncomputable def Q (t : ℕ) : ℤ :=
  if 0 < t ∧ t ≤ 30 then -t + 40 else 0

noncomputable def y (t : ℕ) : ℤ := P t * Q t

theorem maximum_daily_sales_revenue : 
  ∃ (t : ℕ), 0 < t ∧ t ≤ 30 ∧ y t = 1125 :=
by
  sorry

end NUMINAMATH_GPT_maximum_daily_sales_revenue_l1051_105179


namespace NUMINAMATH_GPT_remaining_amount_is_9_l1051_105145

-- Define the original prices of the books
def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00

-- Define the discount rate for the first two books
def discount_rate : ℝ := 0.25

-- Define the total cost without discount
def total_cost_without_discount := book1_price + book2_price + book3_price + book4_price

-- Calculate the discounts for the first two books
def book1_discount := book1_price * discount_rate
def book2_discount := book2_price * discount_rate

-- Calculate the discounted prices for the first two books
def discounted_book1_price := book1_price - book1_discount
def discounted_book2_price := book2_price - book2_discount

-- Calculate the total cost of the books with discounts applied
def total_cost_with_discount := discounted_book1_price + discounted_book2_price + book3_price + book4_price

-- Define the free shipping threshold
def free_shipping_threshold : ℝ := 50.00

-- Calculate the remaining amount Connor needs to spend
def remaining_amount_to_spend := free_shipping_threshold - total_cost_with_discount

-- State the theorem
theorem remaining_amount_is_9 : remaining_amount_to_spend = 9.00 := by
  -- we would provide the proof here
  sorry

end NUMINAMATH_GPT_remaining_amount_is_9_l1051_105145


namespace NUMINAMATH_GPT_tan_identity_proof_l1051_105143

noncomputable def tan_add_pi_over_3 (α β : ℝ) : ℝ :=
  Real.tan (α + Real.pi / 3)

theorem tan_identity_proof 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan (β - Real.pi / 3) = 1 / 4) :
  tan_add_pi_over_3 α β = 7 / 23 := 
sorry

end NUMINAMATH_GPT_tan_identity_proof_l1051_105143


namespace NUMINAMATH_GPT_find_extreme_value_number_of_zeros_l1051_105173

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 + (a - 2) * x - Real.log x

-- Math proof problem I
theorem find_extreme_value (a : ℝ) (h : (∀ x : ℝ, x ≠ 0 → x ≠ 1 → f a x > f a 1)) : a = 1 := 
sorry

-- Math proof problem II
theorem number_of_zeros (a : ℝ) (h : 0 < a ∧ a < 1) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 := 
sorry

end NUMINAMATH_GPT_find_extreme_value_number_of_zeros_l1051_105173


namespace NUMINAMATH_GPT_real_roots_if_and_only_if_m_leq_5_l1051_105111

theorem real_roots_if_and_only_if_m_leq_5 (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x + 1 = 0) ↔ m ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_real_roots_if_and_only_if_m_leq_5_l1051_105111


namespace NUMINAMATH_GPT_graph_of_equation_is_two_lines_l1051_105155

theorem graph_of_equation_is_two_lines : 
  ∀ (x y : ℝ), (x - y)^2 = x^2 - y^2 ↔ (x = 0 ∨ y = 0) := 
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_is_two_lines_l1051_105155


namespace NUMINAMATH_GPT_complex_fraction_expression_equals_half_l1051_105182

theorem complex_fraction_expression_equals_half :
  ((2 / (3 + 1/5)) + (((3 + 1/4) / 13) / (2 / 3)) + (((2 + 5/18) - (17/36)) * (18 / 65))) * (1 / 3) = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_complex_fraction_expression_equals_half_l1051_105182


namespace NUMINAMATH_GPT_smallest_number_of_people_l1051_105159

open Nat

theorem smallest_number_of_people (x : ℕ) :
  (∃ x, x % 18 = 0 ∧ x % 50 = 0 ∧
  (∀ y, y % 18 = 0 ∧ y % 50 = 0 → x ≤ y)) → x = 450 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_people_l1051_105159


namespace NUMINAMATH_GPT_complement_intersection_l1051_105130

open Set

-- Define the universal set I, and sets M and N
def I : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {3, 4, 5}

-- Lean statement to prove the desired result
theorem complement_intersection : (I \ N) ∩ M = {1, 2} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1051_105130


namespace NUMINAMATH_GPT_union_A_B_inter_A_B_inter_compA_B_l1051_105152

-- Extend the universal set U to be the set of all real numbers ℝ
def U : Set ℝ := Set.univ

-- Define set A as the set of all real numbers x such that -3 ≤ x ≤ 4
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}

-- Define set B as the set of all real numbers x such that -1 < x < 5
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}

-- Prove that A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 5}
theorem union_A_B : A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 5} := by
  sorry

-- Prove that A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 4}
theorem inter_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by
  sorry

-- Define the complement of A in U
def comp_A : Set ℝ := {x : ℝ | x < -3 ∨ x > 4}

-- Prove that (complement_U A) ∩ B = {x : ℝ | 4 < x ∧ x < 5}
theorem inter_compA_B : comp_A ∩ B = {x : ℝ | 4 < x ∧ x < 5} := by
  sorry

end NUMINAMATH_GPT_union_A_B_inter_A_B_inter_compA_B_l1051_105152


namespace NUMINAMATH_GPT_joshua_needs_more_cents_l1051_105117

-- Definitions of inputs
def cost_of_pen_dollars : ℕ := 6
def joshua_money_dollars : ℕ := 5
def borrowed_cents : ℕ := 68

-- Convert dollar amounts to cents
def dollar_to_cents (d : ℕ) : ℕ := d * 100

def cost_of_pen_cents := dollar_to_cents cost_of_pen_dollars
def joshua_money_cents := dollar_to_cents joshua_money_dollars

-- Total amount Joshua has in cents
def total_cents := joshua_money_cents + borrowed_cents

-- Calculation of the required amount
def needed_cents := cost_of_pen_cents - total_cents

theorem joshua_needs_more_cents : needed_cents = 32 := by 
  sorry

end NUMINAMATH_GPT_joshua_needs_more_cents_l1051_105117


namespace NUMINAMATH_GPT_add_ten_to_certain_number_l1051_105137

theorem add_ten_to_certain_number (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 :=
by
  sorry

end NUMINAMATH_GPT_add_ten_to_certain_number_l1051_105137


namespace NUMINAMATH_GPT_wizard_achievable_for_odd_n_l1051_105178

-- Define what it means for the wizard to achieve his goal
def wizard_goal_achievable (n : ℕ) : Prop :=
  ∃ (pairs : Finset (ℕ × ℕ)), 
    pairs.card = 2 * n ∧ 
    ∀ (sorcerer_breaks : Finset (ℕ × ℕ)), sorcerer_breaks.card = n → 
      ∃ (dwarves : Finset ℕ), dwarves.card = 2 * n ∧
      ∀ k ∈ dwarves, ((k, (k + 1) % n) ∈ pairs ∨ ((k + 1) % n, k) ∈ pairs) ∧
                     (∀ i j, (i, j) ∈ sorcerer_breaks → ¬((i, j) ∈ pairs ∨ (j, i) ∈ pairs))

theorem wizard_achievable_for_odd_n (n : ℕ) (h : Odd n) : wizard_goal_achievable n := sorry

end NUMINAMATH_GPT_wizard_achievable_for_odd_n_l1051_105178


namespace NUMINAMATH_GPT_solve_equation_l1051_105190

theorem solve_equation (x : ℝ) (h1 : x + 1 ≠ 0) (h2 : 2 * x - 1 ≠ 0) :
  (2 / (x + 1) = 3 / (2 * x - 1)) ↔ (x = 5) := 
sorry

end NUMINAMATH_GPT_solve_equation_l1051_105190


namespace NUMINAMATH_GPT_gain_is_rs_150_l1051_105113

noncomputable def P : ℝ := 5000
noncomputable def R_borrow : ℝ := 4
noncomputable def R_lend : ℝ := 7
noncomputable def T : ℝ := 2

noncomputable def SI (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (P * R * T) / 100

noncomputable def interest_paid := SI P R_borrow T
noncomputable def interest_earned := SI P R_lend T

noncomputable def gain_per_year : ℝ :=
  (interest_earned / T) - (interest_paid / T)

theorem gain_is_rs_150 : gain_per_year = 150 :=
by
  sorry

end NUMINAMATH_GPT_gain_is_rs_150_l1051_105113


namespace NUMINAMATH_GPT_johnny_years_ago_l1051_105126

theorem johnny_years_ago 
  (J : ℕ) (hJ : J = 8) (X : ℕ) 
  (h : J + 2 = 2 * (J - X)) : 
  X = 3 := by
  sorry

end NUMINAMATH_GPT_johnny_years_ago_l1051_105126


namespace NUMINAMATH_GPT_prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29_l1051_105147

theorem prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29 
  (n : ℕ) (h1 : Prime n) (h2 : 20 < n) (h3 : n < 30) (h4 : n % 8 = 5) : n = 29 := 
by
  sorry

end NUMINAMATH_GPT_prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29_l1051_105147


namespace NUMINAMATH_GPT_new_person_weight_l1051_105107

theorem new_person_weight (avg_increase : ℝ) (num_people : ℕ) (weight_replaced : ℝ) (new_weight : ℝ) : 
    num_people = 8 → avg_increase = 1.5 → weight_replaced = 65 → 
    new_weight = weight_replaced + num_people * avg_increase → 
    new_weight = 77 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_new_person_weight_l1051_105107


namespace NUMINAMATH_GPT_sufficient_condition_l1051_105168

variable (x : ℝ) (a : ℝ)

theorem sufficient_condition (h : ∀ x : ℝ, |x| + |x - 1| ≥ 1) : a < 1 → ∀ x : ℝ, a ≤ |x| + |x - 1| :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_l1051_105168


namespace NUMINAMATH_GPT_girls_more_than_boys_l1051_105165

theorem girls_more_than_boys (total_students boys : ℕ) (h1 : total_students = 650) (h2 : boys = 272) :
  (total_students - boys) - boys = 106 :=
by
  sorry

end NUMINAMATH_GPT_girls_more_than_boys_l1051_105165


namespace NUMINAMATH_GPT_find_a_l1051_105174

theorem find_a (a : ℝ) : (∃ (p : ℝ × ℝ), p = (3, -9) ∧ (3 * a * p.1 + (2 * a + 1) * p.2 = 3 * a + 3)) → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1051_105174


namespace NUMINAMATH_GPT_number_of_integers_between_cubed_values_l1051_105129

theorem number_of_integers_between_cubed_values :
  ∃ n : ℕ, n = (1278 - 1122 + 1) ∧ 
  ∀ x : ℤ, (1122 < x ∧ x < 1278) → (1123 ≤ x ∧ x ≤ 1277) := 
by
  sorry

end NUMINAMATH_GPT_number_of_integers_between_cubed_values_l1051_105129


namespace NUMINAMATH_GPT_inequality_not_true_l1051_105109

variable (a b c : ℝ)

theorem inequality_not_true (h : a < b) : ¬ (-3 * a < -3 * b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_not_true_l1051_105109


namespace NUMINAMATH_GPT_find_original_number_l1051_105122

theorem find_original_number (r : ℝ) (h : 1.15 * r - 0.7 * r = 40) : r = 88.88888888888889 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l1051_105122


namespace NUMINAMATH_GPT_soccer_field_solution_l1051_105183

noncomputable def soccer_field_problem : Prop :=
  ∃ (a b c d : ℝ), 
    (abs (a - b) = 1 ∨ abs (a - b) = 2 ∨ abs (a - b) = 3 ∨ abs (a - b) = 4 ∨ abs (a - b) = 5 ∨ abs (a - b) = 6) ∧
    (abs (a - c) = 1 ∨ abs (a - c) = 2 ∨ abs (a - c) = 3 ∨ abs (a - c) = 4 ∨ abs (a - c) = 5 ∨ abs (a - c) = 6) ∧
    (abs (a - d) = 1 ∨ abs (a - d) = 2 ∨ abs (a - d) = 3 ∨ abs (a - d) = 4 ∨ abs (a - d) = 5 ∨ abs (a - d) = 6) ∧
    (abs (b - c) = 1 ∨ abs (b - c) = 2 ∨ abs (b - c) = 3 ∨ abs (b - c) = 4 ∨ abs (b - c) = 5 ∨ abs (b - c) = 6) ∧
    (abs (b - d) = 1 ∨ abs (b - d) = 2 ∨ abs (b - d) = 3 ∨ abs (b - d) = 4 ∨ abs (b - d) = 5 ∨ abs (b - d) = 6) ∧
    (abs (c - d) = 1 ∨ abs (c - d) = 2 ∨ abs (c - d) = 3 ∨ abs (c - d) = 4 ∨ abs (c - d) = 5 ∨ abs (c - d) = 6)

theorem soccer_field_solution : soccer_field_problem :=
  sorry

end NUMINAMATH_GPT_soccer_field_solution_l1051_105183


namespace NUMINAMATH_GPT_total_pieces_of_gum_l1051_105193

theorem total_pieces_of_gum (packages pieces_per_package : ℕ) 
  (h_packages : packages = 9)
  (h_pieces_per_package : pieces_per_package = 15) : 
  packages * pieces_per_package = 135 := by
  subst h_packages
  subst h_pieces_per_package
  exact Nat.mul_comm 9 15 ▸ rfl

end NUMINAMATH_GPT_total_pieces_of_gum_l1051_105193


namespace NUMINAMATH_GPT_remainder_of_sum_div_11_is_9_l1051_105136

def seven_times_ten_pow_twenty : ℕ := 7 * 10 ^ 20
def two_pow_twenty : ℕ := 2 ^ 20
def sum : ℕ := seven_times_ten_pow_twenty + two_pow_twenty

theorem remainder_of_sum_div_11_is_9 : sum % 11 = 9 := by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_div_11_is_9_l1051_105136


namespace NUMINAMATH_GPT_quadratic_value_at_5_l1051_105177

-- Define the conditions provided in the problem
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Create a theorem that states that if a quadratic with given conditions has its vertex at (2, 7) and passes through (0, -7), then passing through (5, n) means n = -24.5
theorem quadratic_value_at_5 (a b c n : ℝ)
  (h1 : quadratic a b c 2 = 7)
  (h2 : quadratic a b c 0 = -7)
  (h3 : quadratic a b c 5 = n) :
  n = -24.5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_value_at_5_l1051_105177


namespace NUMINAMATH_GPT_binomial_sum_of_coefficients_l1051_105169

theorem binomial_sum_of_coefficients (n : ℕ) (h₀ : (1 - 2)^n = 8) :
  (1 - 2)^n = -1 :=
sorry

end NUMINAMATH_GPT_binomial_sum_of_coefficients_l1051_105169


namespace NUMINAMATH_GPT_polyhedron_volume_l1051_105102

-- Define the polyhedron and its properties
def polyhedron (P : Type) : Prop :=
∃ (C : Type), 
  (∀ (p : P) (e : ℝ), e = 2) ∧ 
  (∃ (octFaces triFaces : ℕ), octFaces = 6 ∧ triFaces = 8) ∧
  (∀ (vol : ℝ), vol = (56 + (112 * Real.sqrt 2) / 3))
  
-- A theorem stating the volume of the polyhedron
theorem polyhedron_volume : ∀ (P : Type), polyhedron P → ∃ (vol : ℝ), vol = 56 + (112 * Real.sqrt 2) / 3 :=
by
  intros P hP
  sorry

end NUMINAMATH_GPT_polyhedron_volume_l1051_105102


namespace NUMINAMATH_GPT_clock_malfunction_fraction_correct_l1051_105104

theorem clock_malfunction_fraction_correct : 
  let hours_total := 24
  let hours_incorrect := 6
  let minutes_total := 60
  let minutes_incorrect := 6
  let fraction_correct_hours := (hours_total - hours_incorrect) / hours_total
  let fraction_correct_minutes := (minutes_total - minutes_incorrect) / minutes_total
  (fraction_correct_hours * fraction_correct_minutes) = 27 / 40
:= 
by
  sorry

end NUMINAMATH_GPT_clock_malfunction_fraction_correct_l1051_105104


namespace NUMINAMATH_GPT_decreasing_geometric_sequence_l1051_105142

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * q ^ n

theorem decreasing_geometric_sequence (a₁ q : ℝ) (aₙ : ℕ → ℝ) (hₙ : ∀ n, aₙ n = geometric_sequence a₁ q n) 
  (h_condition : 0 < q ∧ q < 1) : ¬(0 < q ∧ q < 1 ↔ ∀ n, aₙ n > aₙ (n + 1)) :=
sorry

end NUMINAMATH_GPT_decreasing_geometric_sequence_l1051_105142


namespace NUMINAMATH_GPT_find_second_projection_l1051_105162

noncomputable def second_projection (plane : Prop) (first_proj : Prop) (distance : ℝ) : Prop :=
∃ second_proj : Prop, true

theorem find_second_projection 
  (plane : Prop) 
  (first_proj : Prop) 
  (distance : ℝ) :
  ∃ second_proj : Prop, true :=
sorry

end NUMINAMATH_GPT_find_second_projection_l1051_105162


namespace NUMINAMATH_GPT_pool_water_left_l1051_105156

theorem pool_water_left 
  (h1_rate: ℝ) (h1_time: ℝ)
  (h2_rate: ℝ) (h2_time: ℝ)
  (h4_rate: ℝ) (h4_time: ℝ)
  (leak_loss: ℝ)
  (h1_rate_eq: h1_rate = 8)
  (h1_time_eq: h1_time = 1)
  (h2_rate_eq: h2_rate = 10)
  (h2_time_eq: h2_time = 2)
  (h4_rate_eq: h4_rate = 14)
  (h4_time_eq: h4_time = 1)
  (leak_loss_eq: leak_loss = 8) :
  (h1_rate * h1_time) + (h2_rate * h2_time) + (h2_rate * h2_time) + (h4_rate * h4_time) - leak_loss = 34 :=
by
  rw [h1_rate_eq, h1_time_eq, h2_rate_eq, h2_time_eq, h4_rate_eq, h4_time_eq, leak_loss_eq]
  norm_num
  sorry

end NUMINAMATH_GPT_pool_water_left_l1051_105156


namespace NUMINAMATH_GPT_probability_black_ball_l1051_105175

variable (total_balls : ℕ)
variable (red_balls : ℕ)
variable (white_probability : ℝ)

def number_of_balls : Prop := total_balls = 100
def red_ball_count : Prop := red_balls = 45
def white_ball_probability : Prop := white_probability = 0.23

theorem probability_black_ball 
  (h1 : number_of_balls total_balls)
  (h2 : red_ball_count red_balls)
  (h3 : white_ball_probability white_probability) :
  let white_balls := white_probability * total_balls 
  let black_balls := total_balls - red_balls - white_balls
  let black_ball_prob := black_balls / total_balls
  black_ball_prob = 0.32 :=
sorry

end NUMINAMATH_GPT_probability_black_ball_l1051_105175


namespace NUMINAMATH_GPT_largest_int_mod_6_less_than_100_l1051_105148

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end NUMINAMATH_GPT_largest_int_mod_6_less_than_100_l1051_105148


namespace NUMINAMATH_GPT_bob_age_l1051_105131

theorem bob_age (a b : ℝ) 
    (h1 : b = 3 * a - 20)
    (h2 : b + a = 70) : 
    b = 47.5 := by
    sorry

end NUMINAMATH_GPT_bob_age_l1051_105131


namespace NUMINAMATH_GPT_positivity_of_xyz_l1051_105199

variable {x y z : ℝ}

theorem positivity_of_xyz
  (h1 : x + y + z > 0)
  (h2 : xy + yz + zx > 0)
  (h3 : xyz > 0) :
  x > 0 ∧ y > 0 ∧ z > 0 := 
sorry

end NUMINAMATH_GPT_positivity_of_xyz_l1051_105199


namespace NUMINAMATH_GPT_total_birds_correct_l1051_105167

def numPairs : Nat := 3
def birdsPerPair : Nat := 2
def totalBirds : Nat := numPairs * birdsPerPair

theorem total_birds_correct : totalBirds = 6 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_birds_correct_l1051_105167


namespace NUMINAMATH_GPT_minimum_value_of_f_l1051_105186

noncomputable def f (x y z : ℝ) : ℝ := (1 / (x + y)) + (1 / (x + z)) + (1 / (y + z)) - (x * y * z)

theorem minimum_value_of_f :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 3 → f x y z = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1051_105186


namespace NUMINAMATH_GPT_correct_operation_l1051_105140

theorem correct_operation (a b : ℝ) : 
  (a+2)*(a-2) = a^2 - 4 :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1051_105140


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1051_105146

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < -2) : 
  { x : ℝ | ax^2 + (a - 2)*x - 2 ≥ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 2/a } := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1051_105146


namespace NUMINAMATH_GPT_total_monthly_sales_l1051_105127

-- Definitions and conditions
def num_customers_per_month : ℕ := 500
def lettuce_per_customer : ℕ := 2
def price_per_lettuce : ℕ := 1
def tomatoes_per_customer : ℕ := 4
def price_per_tomato : ℕ := 1 / 2

-- Statement to prove
theorem total_monthly_sales : num_customers_per_month * (lettuce_per_customer * price_per_lettuce + tomatoes_per_customer * price_per_tomato) = 2000 := 
by 
  sorry

end NUMINAMATH_GPT_total_monthly_sales_l1051_105127


namespace NUMINAMATH_GPT_annie_journey_time_l1051_105150

noncomputable def total_time_journey (walk_speed1 bus_speed train_speed walk_speed2 blocks_walk1 blocks_bus blocks_train blocks_walk2 : ℝ) : ℝ :=
  let time_walk1 := blocks_walk1 / walk_speed1
  let time_bus := blocks_bus / bus_speed
  let time_train := blocks_train / train_speed
  let time_walk2 := blocks_walk2 / walk_speed2
  let time_back := time_walk2
  time_walk1 + time_bus + time_train + time_walk2 + time_back + time_train + time_bus + time_walk1

theorem annie_journey_time :
  total_time_journey 2 4 5 2 5 7 10 4 = 16.5 := by 
  sorry

end NUMINAMATH_GPT_annie_journey_time_l1051_105150


namespace NUMINAMATH_GPT_fireworks_display_l1051_105192

def year_fireworks : Nat := 4 * 6
def letters_fireworks : Nat := 12 * 5
def boxes_fireworks : Nat := 50 * 8

theorem fireworks_display : year_fireworks + letters_fireworks + boxes_fireworks = 484 := by
  have h1 : year_fireworks = 24 := rfl
  have h2 : letters_fireworks = 60 := rfl
  have h3 : boxes_fireworks = 400 := rfl
  calc
    year_fireworks + letters_fireworks + boxes_fireworks 
        = 24 + 60 + 400 := by rw [h1, h2, h3]
    _ = 484 := rfl

end NUMINAMATH_GPT_fireworks_display_l1051_105192


namespace NUMINAMATH_GPT_fixed_cost_to_break_even_l1051_105161

def cost_per_handle : ℝ := 0.6
def selling_price_per_handle : ℝ := 4.6
def num_handles_to_break_even : ℕ := 1910

theorem fixed_cost_to_break_even (F : ℝ) (h : F = num_handles_to_break_even * (selling_price_per_handle - cost_per_handle)) :
  F = 7640 := by
  sorry

end NUMINAMATH_GPT_fixed_cost_to_break_even_l1051_105161


namespace NUMINAMATH_GPT_lizard_eye_difference_l1051_105135

def jan_eye : ℕ := 3
def jan_wrinkle : ℕ := 3 * jan_eye
def jan_spot : ℕ := 7 * jan_wrinkle

def cousin_eye : ℕ := 3
def cousin_wrinkle : ℕ := 2 * cousin_eye
def cousin_spot : ℕ := 5 * cousin_wrinkle

def total_eyes : ℕ := jan_eye + cousin_eye
def total_wrinkles : ℕ := jan_wrinkle + cousin_wrinkle
def total_spots : ℕ := jan_spot + cousin_spot
def total_spots_and_wrinkles : ℕ := total_wrinkles + total_spots

theorem lizard_eye_difference : total_spots_and_wrinkles - total_eyes = 102 := by
  sorry

end NUMINAMATH_GPT_lizard_eye_difference_l1051_105135


namespace NUMINAMATH_GPT_area_of_rectangle_l1051_105100

-- Definitions and conditions
def side_of_square : ℕ := 50
def radius_of_circle : ℕ := side_of_square
def length_of_rectangle : ℕ := (2 * radius_of_circle) / 5
def breadth_of_rectangle : ℕ := 10

-- Theorem statement
theorem area_of_rectangle :
  (length_of_rectangle * breadth_of_rectangle = 200) := by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1051_105100


namespace NUMINAMATH_GPT_total_decorations_l1051_105116

theorem total_decorations 
  (skulls : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) (pumpkins : ℕ) 
  (cauldron : ℕ) (budget_decorations : ℕ) (left_decorations : ℕ)
  (h_skulls : skulls = 12)
  (h_broomsticks : broomsticks = 4)
  (h_spiderwebs : spiderwebs = 12)
  (h_pumpkins : pumpkins = 2 * spiderwebs)
  (h_cauldron : cauldron = 1)
  (h_budget_decorations : budget_decorations = 20)
  (h_left_decorations : left_decorations = 10) : 
  skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget_decorations + left_decorations = 83 := 
by 
  sorry

end NUMINAMATH_GPT_total_decorations_l1051_105116


namespace NUMINAMATH_GPT_ratio_evaluation_l1051_105166

theorem ratio_evaluation : (5^3003 * 2^3005) / (10^3004) = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_evaluation_l1051_105166


namespace NUMINAMATH_GPT_tree_ratio_l1051_105112

theorem tree_ratio (A P C : ℕ) 
  (hA : A = 58)
  (hP : P = 3 * A)
  (hC : C = 5 * P) : (A, P, C) = (1, 3 * 58, 15 * 58) :=
by
  sorry

end NUMINAMATH_GPT_tree_ratio_l1051_105112


namespace NUMINAMATH_GPT_range_of_a_l1051_105154

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x >= 2 then (a - 1 / 2) * x 
  else a^x - 4

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ (1 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1051_105154


namespace NUMINAMATH_GPT_quadratic_eq_l1051_105184

noncomputable def roots (r s : ℝ): Prop := r + s = 12 ∧ r * s = 27 ∧ (r = 2 * s ∨ s = 2 * r)

theorem quadratic_eq (r s : ℝ) (h : roots r s) : 
   Polynomial.C 1 * (X^2 - Polynomial.C (r + s) * X + Polynomial.C (r * s)) = X ^ 2 - 12 * X + 27 := 
sorry

end NUMINAMATH_GPT_quadratic_eq_l1051_105184


namespace NUMINAMATH_GPT_find_alpha_l1051_105164

theorem find_alpha (α : ℝ) (h_cos : Real.cos α = - (Real.sqrt 3 / 2)) (h_range : 0 < α ∧ α < Real.pi) : α = 5 * Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_find_alpha_l1051_105164


namespace NUMINAMATH_GPT_math_proof_problem_l1051_105158

noncomputable def proof_problem : Prop :=
  ∃ (p : ℝ) (k m : ℝ), 
    (∀ (x y : ℝ), y^2 = 2 * p * x) ∧
    (p > 0) ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      (y1 * y2 = -8) ∧
      (x1 = 4 ∧ y1 = 0 ∨ x2 = 4 ∧ y2 = 0)) ∧
    (p = 1) ∧ 
    (∀ x0 : ℝ, 
      (2 * k * m = 1) ∧
      (∀ (x y : ℝ), y = k * x + m) ∧ 
      (∃ (r : ℝ), 
        ((x0 - r + 1 = 0) ∧
         (x0 - r * x0 + r^2 = 0))) ∧ 
       x0 = -1 / 2 )

theorem math_proof_problem : proof_problem := 
  sorry

end NUMINAMATH_GPT_math_proof_problem_l1051_105158


namespace NUMINAMATH_GPT_num_students_third_section_l1051_105108

-- Define the conditions
def num_students_first_section : ℕ := 65
def num_students_second_section : ℕ := 35
def num_students_fourth_section : ℕ := 42
def mean_marks_first_section : ℝ := 50
def mean_marks_second_section : ℝ := 60
def mean_marks_third_section : ℝ := 55
def mean_marks_fourth_section : ℝ := 45
def overall_average_marks : ℝ := 51.95

-- Theorem stating the number of students in the third section
theorem num_students_third_section
  (x : ℝ)
  (h : (num_students_first_section * mean_marks_first_section
       + num_students_second_section * mean_marks_second_section
       + x * mean_marks_third_section
       + num_students_fourth_section * mean_marks_fourth_section)
       = overall_average_marks * (num_students_first_section + num_students_second_section + x + num_students_fourth_section)) :
  x = 45 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_num_students_third_section_l1051_105108


namespace NUMINAMATH_GPT_find_integers_l1051_105157

theorem find_integers (x : ℤ) : x^2 < 3 * x → x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_GPT_find_integers_l1051_105157


namespace NUMINAMATH_GPT_value_of_a10_l1051_105153

/-- Define arithmetic sequence and properties -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n) / 2)

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
axiom arith_seq : arithmetic_sequence a d
axiom sum_formula : sum_of_first_n_terms a 5 S
axiom sum_condition : S 5 = 60
axiom term_condition : a 1 + a 2 + a 3 = a 4 + a 5

theorem value_of_a10 : a 10 = 26 :=
sorry

end NUMINAMATH_GPT_value_of_a10_l1051_105153


namespace NUMINAMATH_GPT_housewife_spent_on_oil_l1051_105198

-- Define the conditions
variables (P A : ℝ)
variables (h_price_reduced : 0.7 * P = 70)
variables (h_more_oil : A / 70 = A / P + 3)

-- Define the theorem to be proven
theorem housewife_spent_on_oil : A = 700 :=
by
  sorry

end NUMINAMATH_GPT_housewife_spent_on_oil_l1051_105198


namespace NUMINAMATH_GPT_shinyoung_initial_candies_l1051_105151

theorem shinyoung_initial_candies : 
  ∀ (C : ℕ), 
    (C / 2) - ((C / 6) + 5) = 5 → 
    C = 30 := by
  intros C h
  sorry

end NUMINAMATH_GPT_shinyoung_initial_candies_l1051_105151


namespace NUMINAMATH_GPT_find_m_from_decomposition_l1051_105123

theorem find_m_from_decomposition (m : ℕ) (h : m > 0) : (m^2 - m + 1 = 73) → (m = 9) :=
by
  sorry

end NUMINAMATH_GPT_find_m_from_decomposition_l1051_105123


namespace NUMINAMATH_GPT_sector_central_angle_l1051_105110

theorem sector_central_angle (r θ : ℝ) (h1 : 2 * r + r * θ = 6) (h2 : 0.5 * r * r * θ = 2) : θ = 1 ∨ θ = 4 :=
sorry

end NUMINAMATH_GPT_sector_central_angle_l1051_105110


namespace NUMINAMATH_GPT_select_4_blocks_no_same_row_column_l1051_105172

theorem select_4_blocks_no_same_row_column :
  ∃ (n : ℕ), n = (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4) ∧ n = 5400 :=
by
  sorry

end NUMINAMATH_GPT_select_4_blocks_no_same_row_column_l1051_105172


namespace NUMINAMATH_GPT_ninety_times_ninety_l1051_105195

theorem ninety_times_ninety : (90 * 90) = 8100 := by
  let a := 100
  let b := 10
  have h1 : (90 * 90) = (a - b) * (a - b) := by decide
  have h2 : (a - b) * (a - b) = a^2 - 2 * a * b + b^2 := by decide
  have h3 : a = 100 := rfl
  have h4 : b = 10 := rfl
  have h5 : 100^2 - 2 * 100 * 10 + 10^2 = 8100 := by decide
  sorry

end NUMINAMATH_GPT_ninety_times_ninety_l1051_105195


namespace NUMINAMATH_GPT_simplified_expression_is_one_l1051_105171

-- Define the specific mathematical expressions
def expr1 := -1 ^ 2023
def expr2 := (-2) ^ 3
def expr3 := (-2) * (-3)

-- Construct the full expression
def full_expr := expr1 - expr2 - expr3

-- State the theorem that this full expression equals 1
theorem simplified_expression_is_one : full_expr = 1 := by
  sorry

end NUMINAMATH_GPT_simplified_expression_is_one_l1051_105171


namespace NUMINAMATH_GPT_exists_x_y_mod_p_l1051_105194

theorem exists_x_y_mod_p (p : ℕ) (hp : Nat.Prime p) (a : ℤ) : ∃ x y : ℤ, (x^2 + y^3) % p = a % p :=
by
  sorry

end NUMINAMATH_GPT_exists_x_y_mod_p_l1051_105194


namespace NUMINAMATH_GPT_total_stops_traveled_l1051_105120

-- Definitions based on the conditions provided
def yoojeong_stops : ℕ := 3
def namjoon_stops : ℕ := 2

-- Theorem statement to prove the total number of stops
theorem total_stops_traveled : yoojeong_stops + namjoon_stops = 5 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_stops_traveled_l1051_105120


namespace NUMINAMATH_GPT_difference_in_pages_l1051_105118

def purple_pages_per_book : ℕ := 230
def orange_pages_per_book : ℕ := 510
def purple_books_read : ℕ := 5
def orange_books_read : ℕ := 4

theorem difference_in_pages : 
  orange_books_read * orange_pages_per_book - purple_books_read * purple_pages_per_book = 890 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_pages_l1051_105118


namespace NUMINAMATH_GPT_initial_people_count_l1051_105103

theorem initial_people_count (x : ℕ) 
  (h1 : (x + 15) % 5 = 0)
  (h2 : (x + 15) / 5 = 12) : 
  x = 45 := 
by
  sorry

end NUMINAMATH_GPT_initial_people_count_l1051_105103


namespace NUMINAMATH_GPT_polynomial_division_l1051_105139

theorem polynomial_division (a b c : ℤ) :
  (∀ x : ℝ, (17 * x^2 - 3 * x + 4) - (a * x^2 + b * x + c) = (5 * x + 6) * (2 * x + 1)) →
  a - b - c = 29 := by
  sorry

end NUMINAMATH_GPT_polynomial_division_l1051_105139


namespace NUMINAMATH_GPT_division_quotient_less_dividend_l1051_105121

theorem division_quotient_less_dividend
  (a1 : (6 : ℝ) > 0)
  (a2 : (5 / 7 : ℝ) > 0)
  (a3 : (3 / 8 : ℝ) > 0)
  (h1 : (3 / 5 : ℝ) < 1)
  (h2 : (5 / 4 : ℝ) > 1)
  (h3 : (5 / 12 : ℝ) < 1):
  (6 / (3 / 5) > 6) ∧ (5 / 7 / (5 / 4) < 5 / 7) ∧ (3 / 8 / (5 / 12) > 3 / 8) :=
by
  sorry

end NUMINAMATH_GPT_division_quotient_less_dividend_l1051_105121


namespace NUMINAMATH_GPT_bottles_of_regular_soda_l1051_105160

theorem bottles_of_regular_soda
  (diet_soda : ℕ)
  (apples : ℕ)
  (more_bottles_than_apples : ℕ)
  (R : ℕ)
  (h1 : diet_soda = 32)
  (h2 : apples = 78)
  (h3 : more_bottles_than_apples = 26)
  (h4 : R + diet_soda = apples + more_bottles_than_apples) :
  R = 72 := 
by sorry

end NUMINAMATH_GPT_bottles_of_regular_soda_l1051_105160


namespace NUMINAMATH_GPT_find_speed_of_stream_l1051_105176

-- Definitions of the conditions:
def downstream_equation (b s : ℝ) : Prop := b + s = 60
def upstream_equation (b s : ℝ) : Prop := b - s = 30

-- Theorem stating the speed of the stream given the conditions:
theorem find_speed_of_stream (b s : ℝ) (h1 : downstream_equation b s) (h2 : upstream_equation b s) : s = 15 := 
sorry

end NUMINAMATH_GPT_find_speed_of_stream_l1051_105176


namespace NUMINAMATH_GPT_wendy_furniture_time_l1051_105181

theorem wendy_furniture_time (chairs tables minutes_per_piece : ℕ) 
    (h_chairs : chairs = 4) 
    (h_tables : tables = 4) 
    (h_minutes_per_piece : minutes_per_piece = 6) : 
    chairs + tables * minutes_per_piece = 48 := 
by 
    sorry

end NUMINAMATH_GPT_wendy_furniture_time_l1051_105181


namespace NUMINAMATH_GPT_b_contribution_is_correct_l1051_105105

-- Definitions based on the conditions
def A_investment : ℕ := 35000
def B_join_after_months : ℕ := 5
def profit_ratio_A_B : ℕ := 2
def profit_ratio_B_A : ℕ := 3
def A_total_months : ℕ := 12
def B_total_months : ℕ := 7
def profit_ratio := (profit_ratio_A_B, profit_ratio_B_A)
def total_investment_time_ratio : ℕ := 12 * 35000 / 7

-- The property to be proven
theorem b_contribution_is_correct (X : ℕ) (h : 35000 * 12 / (X * 7) = 2 / 3) : X = 90000 :=
by
  sorry

end NUMINAMATH_GPT_b_contribution_is_correct_l1051_105105


namespace NUMINAMATH_GPT_lemons_per_glass_l1051_105197

theorem lemons_per_glass (lemons glasses : ℕ) (h : lemons = 18 ∧ glasses = 9) : lemons / glasses = 2 :=
by
  sorry

end NUMINAMATH_GPT_lemons_per_glass_l1051_105197


namespace NUMINAMATH_GPT_triangle_right_angle_l1051_105188

theorem triangle_right_angle
  (a b m : ℝ)
  (h1 : 0 < b)
  (h2 : b < m)
  (h3 : a^2 + b^2 = m^2) :
  a^2 + b^2 = m^2 :=
by sorry

end NUMINAMATH_GPT_triangle_right_angle_l1051_105188


namespace NUMINAMATH_GPT_sequence_a1_l1051_105149

variable (S : ℕ → ℤ) (a : ℕ → ℤ)

def Sn_formula (n : ℕ) (a₁ : ℤ) : ℤ := (a₁ * (4^n - 1)) / 3

theorem sequence_a1 (h1 : ∀ n : ℕ, S n = Sn_formula n (a 1))
                    (h2 : a 4 = 32) :
  a 1 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a1_l1051_105149


namespace NUMINAMATH_GPT_number_of_true_statements_l1051_105124

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m
def is_odd (n : ℕ) : Prop := ∃ m : ℕ, n = 2 * m + 1
def is_even (n : ℕ) : Prop := ∃ m : ℕ, n = 2 * m

theorem number_of_true_statements : 3 = (ite ((∀ p q : ℕ, is_prime p → is_prime q → is_prime (p * q)) = false) 0 1) +
                                     (ite ((∀ a b : ℕ, is_square a → is_square b → is_square (a * b)) = true) 1 0) +
                                     (ite ((∀ x y : ℕ, is_odd x → is_odd y → is_odd (x * y)) = true) 1 0) +
                                     (ite ((∀ u v : ℕ, is_even u → is_even v → is_even (u * v)) = true) 1 0) :=
by
  sorry

end NUMINAMATH_GPT_number_of_true_statements_l1051_105124


namespace NUMINAMATH_GPT_volume_of_wedge_l1051_105128

theorem volume_of_wedge (d : ℝ) (angle : ℝ) (V : ℝ) (n : ℕ) 
  (h_d : d = 18) 
  (h_angle : angle = 60)
  (h_radius_height : ∀ r h, r = d / 2 ∧ h = d) 
  (h_volume_cylinder : V = π * (d / 2) ^ 2 * d) 
  : n = 729 ↔ V / 2 = n * π :=
by
  sorry

end NUMINAMATH_GPT_volume_of_wedge_l1051_105128


namespace NUMINAMATH_GPT_apples_needed_l1051_105163

-- Define a simple equivalence relation between the weights of oranges and apples.
def weight_equivalent (oranges apples : ℕ) : Prop :=
  8 * apples = 6 * oranges
  
-- State the main theorem based on the given conditions
theorem apples_needed (oranges_count : ℕ) (h : weight_equivalent 1 1) : oranges_count = 32 → ∃ apples_count, apples_count = 24 :=
by
  sorry

end NUMINAMATH_GPT_apples_needed_l1051_105163


namespace NUMINAMATH_GPT_city_map_distance_example_l1051_105144

variable (distance_on_map : ℝ)
variable (scale : ℝ)
variable (actual_distance : ℝ)

theorem city_map_distance_example
  (h1 : distance_on_map = 16)
  (h2 : scale = 1 / 10000)
  (h3 : actual_distance = distance_on_map / scale) :
  actual_distance = 1.6 * 10^3 :=
by
  sorry

end NUMINAMATH_GPT_city_map_distance_example_l1051_105144


namespace NUMINAMATH_GPT_determine_a_l1051_105134

-- Define the sets A and B
def A : Set ℝ := { -1, 0, 2 }
def B (a : ℝ) : Set ℝ := { 2^a }

-- State the main theorem
theorem determine_a (a : ℝ) (h : B a ⊆ A) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l1051_105134


namespace NUMINAMATH_GPT_dot_product_calculation_l1051_105133

def vector := (ℤ × ℤ)

def dot_product (v1 v2 : vector) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : vector := (1, 3)
def b : vector := (-1, 2)

def scalar_mult (c : ℤ) (v : vector) : vector :=
  (c * v.1, c * v.2)

def vector_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem dot_product_calculation :
  dot_product (vector_add (scalar_mult 2 a) b) b = 15 := by
  sorry

end NUMINAMATH_GPT_dot_product_calculation_l1051_105133


namespace NUMINAMATH_GPT_charcoal_amount_l1051_105187

theorem charcoal_amount (water_per_charcoal : ℕ) (charcoal_ratio : ℕ) (water_added : ℕ) (charcoal_needed : ℕ) 
  (h1 : water_per_charcoal = 30) (h2 : charcoal_ratio = 2) (h3 : water_added = 900) : charcoal_needed = 60 :=
by
  sorry

end NUMINAMATH_GPT_charcoal_amount_l1051_105187


namespace NUMINAMATH_GPT_james_muffins_correct_l1051_105114

-- Arthur baked 115 muffins
def arthur_muffins : ℕ := 115

-- James baked 12 times as many muffins as Arthur
def james_multiplier : ℕ := 12

-- The number of muffins James baked
def james_muffins : ℕ := arthur_muffins * james_multiplier

-- The expected result
def expected_james_muffins : ℕ := 1380

-- The statement we want to prove
theorem james_muffins_correct : james_muffins = expected_james_muffins := by
  sorry

end NUMINAMATH_GPT_james_muffins_correct_l1051_105114


namespace NUMINAMATH_GPT_total_books_count_l1051_105191

theorem total_books_count (books_read : ℕ) (books_unread : ℕ) (h1 : books_read = 13) (h2 : books_unread = 8) : books_read + books_unread = 21 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_books_count_l1051_105191


namespace NUMINAMATH_GPT_find_b_in_geometric_sequence_l1051_105138

theorem find_b_in_geometric_sequence (a_1 : ℤ) :
  ∀ (n : ℕ), ∃ (b : ℤ), (3^n - b = (a_1 * (3^n - 1)) / 2) :=
by
  sorry

example (a_1 : ℤ) :
  ∃ (b : ℤ), ∀ (n : ℕ), 3^n - b = (a_1 * (3^n - 1)) / 2 :=
by
  use 1
  sorry

end NUMINAMATH_GPT_find_b_in_geometric_sequence_l1051_105138


namespace NUMINAMATH_GPT_checkerboard_contains_5_black_squares_l1051_105180

def is_checkerboard (x y : ℕ) : Prop := 
  x < 8 ∧ y < 8 ∧ (x + y) % 2 = 0

def contains_5_black_squares (x y n : ℕ) : Prop :=
  ∃ k l : ℕ, k ≤ n ∧ l ≤ n ∧ (x + k + y + l) % 2 = 0 ∧ k * l >= 5

theorem checkerboard_contains_5_black_squares :
  ∃ num, num = 73 ∧
  (∀ x y n, contains_5_black_squares x y n → num = 73) :=
by
  sorry

end NUMINAMATH_GPT_checkerboard_contains_5_black_squares_l1051_105180


namespace NUMINAMATH_GPT_gcf_lcm_60_72_l1051_105141

def gcf_lcm_problem (a b : ℕ) : Prop :=
  gcd a b = 12 ∧ lcm a b = 360

theorem gcf_lcm_60_72 : gcf_lcm_problem 60 72 :=
by {
  sorry
}

end NUMINAMATH_GPT_gcf_lcm_60_72_l1051_105141


namespace NUMINAMATH_GPT_original_pencil_count_l1051_105106

-- Defining relevant constants and assumptions based on the problem conditions
def pencilsRemoved : ℕ := 4
def pencilsLeft : ℕ := 83

-- Theorem to prove the original number of pencils is 87
theorem original_pencil_count : pencilsLeft + pencilsRemoved = 87 := by
  sorry

end NUMINAMATH_GPT_original_pencil_count_l1051_105106


namespace NUMINAMATH_GPT_functional_equation_solution_l1051_105119

-- Define the conditions of the problem.
variable (f : ℝ → ℝ) 
variable (h : ∀ x y u v : ℝ, (f x + f y) * (f u + f v) = f (x * u - y * v) + f (x * v + y * u))

-- Formalize the statement that no other functions satisfy the conditions except f(x) = x^2.
theorem functional_equation_solution : (∀ x : ℝ, f x = x^2) :=
by
  -- The proof goes here, but since the proof is not required, we skip it.
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1051_105119


namespace NUMINAMATH_GPT_soja_book_page_count_l1051_105115

theorem soja_book_page_count (P : ℕ) (h1 : P > 0) (h2 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 100) : P = 300 :=
by
  -- The Lean proof is not required, so we just add sorry to skip the proof
  sorry

end NUMINAMATH_GPT_soja_book_page_count_l1051_105115


namespace NUMINAMATH_GPT_ad_lt_bc_l1051_105185

theorem ad_lt_bc (a b c d : ℝ ) (h1a : a > 0) (h1b : b > 0) (h1c : c > 0) (h1d : d > 0)
  (h2 : a + d = b + c) (h3 : |a - d| < |b - c|) : a * d < b * c :=
  sorry

end NUMINAMATH_GPT_ad_lt_bc_l1051_105185


namespace NUMINAMATH_GPT_parabola_distance_to_focus_l1051_105101

theorem parabola_distance_to_focus :
  ∀ (P : ℝ × ℝ), P.1 = 2 ∧ P.2^2 = 4 * P.1 → dist P (1, 0) = 3 :=
by
  intro P h
  have h₁ : P.1 = 2 := h.1
  have h₂ : P.2^2 = 4 * P.1 := h.2
  sorry

end NUMINAMATH_GPT_parabola_distance_to_focus_l1051_105101


namespace NUMINAMATH_GPT_compute_value_l1051_105170

noncomputable def repeating_decimal_31 : ℝ := 31 / 100000
noncomputable def repeating_decimal_47 : ℝ := 47 / 100000
def term : ℝ := 10^5 - 10^3

theorem compute_value : (term * repeating_decimal_31 + term * repeating_decimal_47) = 77.22 := 
by
  sorry

end NUMINAMATH_GPT_compute_value_l1051_105170


namespace NUMINAMATH_GPT_neg_exists_le_zero_iff_forall_gt_zero_l1051_105132

variable (m : ℝ)

theorem neg_exists_le_zero_iff_forall_gt_zero :
  (¬ ∃ x : ℤ, (x:ℝ)^2 + 2 * x + m ≤ 0) ↔ ∀ x : ℤ, (x:ℝ)^2 + 2 * x + m > 0 :=
by
  sorry

end NUMINAMATH_GPT_neg_exists_le_zero_iff_forall_gt_zero_l1051_105132


namespace NUMINAMATH_GPT_delacroix_band_max_members_l1051_105125

theorem delacroix_band_max_members :
  ∃ n : ℕ, 30 * n % 28 = 6 ∧ 30 * n < 1200 ∧ 30 * n = 930 :=
by
  sorry

end NUMINAMATH_GPT_delacroix_band_max_members_l1051_105125
