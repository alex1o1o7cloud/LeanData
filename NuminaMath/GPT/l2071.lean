import Mathlib

namespace students_play_neither_l2071_207108

def total_students : ℕ := 35
def play_football : ℕ := 26
def play_tennis : ℕ := 20
def play_both : ℕ := 17

theorem students_play_neither : (total_students - (play_football + play_tennis - play_both)) = 6 := by
  sorry

end students_play_neither_l2071_207108


namespace will_money_left_l2071_207131

def initial_money : ℝ := 74
def sweater_cost : ℝ := 9
def tshirt_cost : ℝ := 11
def shoes_cost : ℝ := 30
def hat_cost : ℝ := 5
def socks_cost : ℝ := 4
def refund_percentage : ℝ := 0.85
def discount_percentage : ℝ := 0.1
def tax_percentage : ℝ := 0.05

-- Total cost before returns and discounts
def total_cost_before : ℝ := 
  sweater_cost + tshirt_cost + shoes_cost + hat_cost + socks_cost

-- Refund for shoes
def shoes_refund : ℝ := refund_percentage * shoes_cost

-- New total cost after refund
def total_cost_after_refund : ℝ := total_cost_before - shoes_refund

-- Total cost of remaining items (excluding shoes)
def remaining_items_cost : ℝ := total_cost_before - shoes_cost

-- Discount on remaining items
def discount : ℝ := discount_percentage * remaining_items_cost

-- New total cost after discount
def total_cost_after_discount : ℝ := total_cost_after_refund - discount

-- Sales tax on the final purchase amount
def sales_tax : ℝ := tax_percentage * total_cost_after_discount

-- Final purchase amount with tax
def final_purchase_amount : ℝ := total_cost_after_discount + sales_tax

-- Money left after the final purchase
def money_left : ℝ := initial_money - final_purchase_amount

theorem will_money_left : money_left = 41.87 := by 
  sorry

end will_money_left_l2071_207131


namespace students_like_apple_and_chocolate_not_carrot_l2071_207147

-- Definitions based on the conditions
def total_students : ℕ := 50
def apple_likers : ℕ := 23
def chocolate_likers : ℕ := 20
def carrot_likers : ℕ := 10
def non_likers : ℕ := 15

-- The main statement we need to prove: 
-- the number of students who liked both apple pie and chocolate cake but not carrot cake
theorem students_like_apple_and_chocolate_not_carrot : 
  ∃ (a b c d : ℕ), a + b + d = apple_likers ∧
                    a + c + d = chocolate_likers ∧
                    b + c + d = carrot_likers ∧
                    a + b + c + (50 - (35) - 15) = 35 ∧ 
                    a = 7 :=
by 
  sorry

end students_like_apple_and_chocolate_not_carrot_l2071_207147


namespace solve_xy_l2071_207173

theorem solve_xy : ∃ x y : ℝ, (x - y = 10 ∧ x^2 + y^2 = 100) ↔ ((x = 0 ∧ y = -10) ∨ (x = 10 ∧ y = 0)) := 
by {
  sorry
}

end solve_xy_l2071_207173


namespace range_of_function_l2071_207178

theorem range_of_function : ∀ y : ℝ, ∃ x : ℝ, y = (x^2 + 3*x + 2)/(x^2 + x + 1) :=
by
  sorry

end range_of_function_l2071_207178


namespace total_import_value_l2071_207192

-- Define the given conditions
def export_value : ℝ := 8.07
def additional_amount : ℝ := 1.11
def factor : ℝ := 1.5

-- Define the import value to be proven
def import_value : ℝ := 46.4

-- Main theorem statement
theorem total_import_value :
  export_value = factor * import_value + additional_amount → import_value = 46.4 :=
by sorry

end total_import_value_l2071_207192


namespace probability_odd_product_sum_divisible_by_5_l2071_207102

theorem probability_odd_product_sum_divisible_by_5 :
  (∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧ (a * b % 2 = 1 ∧ (a + b) % 5 = 0)) →
  ∃ (p : ℚ), p = 3 / 95 :=
by
  sorry

end probability_odd_product_sum_divisible_by_5_l2071_207102


namespace find_number_l2071_207151

theorem find_number (x : ℕ) (h : 112 * x = 70000) : x = 625 :=
by
  sorry

end find_number_l2071_207151


namespace minimum_cost_for_13_bottles_l2071_207170

def cost_per_bottle_shop_A := 200 -- in cents
def discount_shop_B := 15 / 100 -- discount
def promotion_B_threshold := 4
def promotion_A_threshold := 4

-- Function to calculate the cost in Shop A for given number of bottles
def shop_A_cost (bottles : ℕ) : ℕ :=
  let batches := bottles / 5
  let remainder := bottles % 5
  (batches * 4 + remainder) * cost_per_bottle_shop_A

-- Function to calculate the cost in Shop B for given number of bottles
def shop_B_cost (bottles : ℕ) : ℕ :=
  if bottles >= promotion_B_threshold then
    (bottles * cost_per_bottle_shop_A) * (1 - discount_shop_B)
  else
    bottles * cost_per_bottle_shop_A

-- Function to calculate combined cost for given numbers of bottles from Shops A and B
def combined_cost (bottles_A bottles_B : ℕ) : ℕ :=
  shop_A_cost bottles_A + shop_B_cost bottles_B

theorem minimum_cost_for_13_bottles : ∃ a b, a + b = 13 ∧ combined_cost a b = 2000 := 
sorry

end minimum_cost_for_13_bottles_l2071_207170


namespace original_price_of_coffee_l2071_207189

/-- 
  Define the prices of the cups of coffee as per the conditions.
  Let x be the original price of one cup of coffee.
  Assert the conditions and find the original price.
-/
theorem original_price_of_coffee (x : ℝ) 
  (h1 : x + x / 2 + 3 = 57) 
  (h2 : (x + x / 2 + 3)/3 = 19) : 
  x = 36 := 
by
  sorry

end original_price_of_coffee_l2071_207189


namespace vector_equality_l2071_207134

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_equality {a x : V} (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a :=
by
  sorry

end vector_equality_l2071_207134


namespace ratio_sum_product_is_constant_l2071_207164

variables {p a : ℝ} (h_a : 0 < a)
theorem ratio_sum_product_is_constant
    (k : ℝ) (h_k : k ≠ 0)
    (x₁ x₂ : ℝ) (h_intersection : x₁ * (2 * p * (x₂ - a)) = 2 * p * (x₁ - a) ∧ x₂ * (2 * p * (x₁ - a)) = 2 * p * (x₂ - a)) :
  (x₁ + x₂) / (x₁ * x₂) = 1 / a := by
  sorry

end ratio_sum_product_is_constant_l2071_207164


namespace quadratic_has_real_roots_l2071_207148

-- Define the condition that a quadratic equation has real roots given ac < 0

variable {a b c : ℝ}

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_real_roots (h : a * c < 0) : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by
  sorry

end quadratic_has_real_roots_l2071_207148


namespace mr_a_loss_l2071_207122

noncomputable def house_initial_value := 12000
noncomputable def first_transaction_loss := 15 / 100
noncomputable def second_transaction_gain := 20 / 100

def house_value_after_first_transaction (initial_value loss : ℝ) : ℝ :=
  initial_value * (1 - loss)

def house_value_after_second_transaction (value_after_first gain : ℝ) : ℝ :=
  value_after_first * (1 + gain)

theorem mr_a_loss :
  let initial_value := house_initial_value
  let loss := first_transaction_loss
  let gain := second_transaction_gain
  let value_after_first := house_value_after_first_transaction initial_value loss
  let value_after_second := house_value_after_second_transaction value_after_first gain
  value_after_second - initial_value = 240 :=
by
  sorry

end mr_a_loss_l2071_207122


namespace arithmetic_sequence_problem_l2071_207120

variables {a : ℕ → ℕ} (d a1 : ℕ)

def arithmetic_sequence (n : ℕ) : ℕ := a1 + (n - 1) * d

theorem arithmetic_sequence_problem
  (h1 : arithmetic_sequence 1 + arithmetic_sequence 3 + arithmetic_sequence 9 = 20) :
  4 * arithmetic_sequence 5 - arithmetic_sequence 7 = 20 :=
by
  sorry

end arithmetic_sequence_problem_l2071_207120


namespace inverse_true_l2071_207136

theorem inverse_true : 
  (∀ (angles : Type) (l1 l2 : Type) (supplementary : angles → angles → Prop)
    (parallel : l1 → l2 → Prop), 
    (∀ a b, supplementary a b → a = b) ∧ (∀ l1 l2, parallel l1 l2)) ↔ 
    (∀ (angles : Type) (l1 l2 : Type) (supplementary : angles → angles → Prop)
    (parallel : l1 → l2 → Prop),
    (∀ l1 l2, parallel l1 l2) ∧ (∀ a b, supplementary a b → a = b)) :=
sorry

end inverse_true_l2071_207136


namespace four_points_nonexistent_l2071_207138

theorem four_points_nonexistent :
  ¬ (∃ (A B C D : ℝ × ℝ × ℝ), 
    dist A B = 8 ∧ 
    dist C D = 8 ∧ 
    dist A C = 10 ∧ 
    dist B D = 10 ∧ 
    dist A D = 13 ∧ 
    dist B C = 13) :=
by
  sorry

end four_points_nonexistent_l2071_207138


namespace distinguishable_large_triangles_l2071_207156

def num_of_distinguishable_large_eq_triangles : Nat :=
  let colors := 8
  let pairs := 7 + Nat.choose 7 2
  colors * pairs

theorem distinguishable_large_triangles : num_of_distinguishable_large_eq_triangles = 224 := by
  sorry

end distinguishable_large_triangles_l2071_207156


namespace quiz_minimum_correct_l2071_207180

theorem quiz_minimum_correct (x : ℕ) (hx : 7 * x + 14 ≥ 120) : x ≥ 16 := 
by sorry

end quiz_minimum_correct_l2071_207180


namespace inequality_solution_set_impossible_l2071_207146

theorem inequality_solution_set_impossible (a b : ℝ) (h_b : b ≠ 0) : ¬ (a = 0 ∧ ∀ x, ax + b > 0 ∧ x > (b / a)) :=
by {
  sorry
}

end inequality_solution_set_impossible_l2071_207146


namespace time_difference_between_shoes_l2071_207116

-- Define the conditions
def time_per_mile_regular := 10
def time_per_mile_new := 13
def distance_miles := 5

-- Define the theorem to be proven
theorem time_difference_between_shoes :
  (distance_miles * time_per_mile_new) - (distance_miles * time_per_mile_regular) = 15 :=
by
  sorry

end time_difference_between_shoes_l2071_207116


namespace arithmetic_geometric_ratio_l2071_207188

noncomputable def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
1 + 3 = a1 + a2

noncomputable def geometric_sequence (b2 : ℝ) : Prop :=
b2 ^ 2 = 4

theorem arithmetic_geometric_ratio (a1 a2 b2 : ℝ) 
  (h1 : arithmetic_sequence a1 a2) 
  (h2 : geometric_sequence b2) : 
  (a1 + a2) / b2 = 5 / 2 :=
by sorry

end arithmetic_geometric_ratio_l2071_207188


namespace n_divisible_by_100_l2071_207163

theorem n_divisible_by_100 
    (n : ℕ) 
    (h_pos : 0 < n) 
    (h_div : 100 ∣ n^3) : 
    100 ∣ n := 
sorry

end n_divisible_by_100_l2071_207163


namespace find_m_l2071_207142

noncomputable def f (m : ℝ) (x : ℝ) := (x^2 + m * x) * Real.exp x

def is_monotonically_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem find_m 
  (a b : ℝ) 
  (h_interval : a = -3/2 ∧ b = 1)
  (h_decreasing : is_monotonically_decreasing_on_interval (f m) a b) :
  m = -3/2 := 
sorry

end find_m_l2071_207142


namespace fraction_simplification_l2071_207118

theorem fraction_simplification :
  (20 + 16 * 20) / (20 * 16) = 17 / 16 :=
by
  sorry

end fraction_simplification_l2071_207118


namespace problem_statement_l2071_207112

theorem problem_statement (x y : ℝ) (h1 : 1/x + 1/y = 5) (h2 : x * y + x + y = 7) : 
  x^2 * y + x * y^2 = 245 / 36 := 
by
  sorry

end problem_statement_l2071_207112


namespace problem_1_solution_problem_2_solution_l2071_207194

noncomputable def problem_1 : Real :=
  (-3) + (2 - Real.pi)^0 - (1 / 2)⁻¹

theorem problem_1_solution :
  problem_1 = -4 :=
by
  sorry

noncomputable def problem_2 (a : Real) : Real :=
  (2 * a)^3 - a * a^2 + 3 * a^6 / a^3

theorem problem_2_solution (a : Real) :
  problem_2 a = 10 * a^3 :=
by
  sorry

end problem_1_solution_problem_2_solution_l2071_207194


namespace find_t_l2071_207117

-- Given conditions 
variables (p j t : ℝ)

-- Condition 1: j is 25% less than p
def condition1 : Prop := j = 0.75 * p

-- Condition 2: j is 20% less than t
def condition2 : Prop := j = 0.80 * t

-- Condition 3: t is t% less than p
def condition3 : Prop := t = p * (1 - t / 100)

-- Final proof statement
theorem find_t (h1 : condition1 p j) (h2 : condition2 j t) (h3 : condition3 p t) : t = 6.25 :=
sorry

end find_t_l2071_207117


namespace increase_in_sides_of_polygon_l2071_207167

theorem increase_in_sides_of_polygon (n n' : ℕ) (h : (n' - 2) * 180 - (n - 2) * 180 = 180) : n' = n + 1 :=
by
  sorry

end increase_in_sides_of_polygon_l2071_207167


namespace sin_721_eq_sin_1_l2071_207119

theorem sin_721_eq_sin_1 : Real.sin (721 * Real.pi / 180) = Real.sin (1 * Real.pi / 180) := 
by
  sorry

end sin_721_eq_sin_1_l2071_207119


namespace largest_divisor_of_five_even_numbers_l2071_207187

theorem largest_divisor_of_five_even_numbers (n : ℕ) (h₁ : n % 2 = 1) : 
  ∃ d, (∀ n, n % 2 = 1 → d ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) ∧ 
       (∀ d', (∀ n, n % 2 = 1 → d' ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) → d' ≤ d) ∧ 
       d = 480 := sorry

end largest_divisor_of_five_even_numbers_l2071_207187


namespace rectangle_not_sum_110_l2071_207154

noncomputable def not_sum_110 : Prop :=
  ∀ (w : ℕ), (w > 0) → (2 * w^2 + 6 * w ≠ 110)

theorem rectangle_not_sum_110 : not_sum_110 := 
  sorry

end rectangle_not_sum_110_l2071_207154


namespace ordered_triples_count_l2071_207135

theorem ordered_triples_count :
  ∃ (count : ℕ), count = 4 ∧
  (∃ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.lcm a b = 90 ∧
    Nat.lcm a c = 980 ∧
    Nat.lcm b c = 630) :=
by
  sorry

end ordered_triples_count_l2071_207135


namespace classical_prob_exp_is_exp1_l2071_207141

-- Define the conditions under which an experiment is a classical probability model
def classical_probability_model (experiment : String) : Prop :=
  match experiment with
  | "exp1" => true  -- experiment ①: finite outcomes and equal likelihood
  | "exp2" => false -- experiment ②: infinite outcomes
  | "exp3" => false -- experiment ③: unequal likelihood
  | "exp4" => false -- experiment ④: infinite outcomes
  | _ => false

theorem classical_prob_exp_is_exp1 : classical_probability_model "exp1" = true ∧
                                      classical_probability_model "exp2" = false ∧
                                      classical_probability_model "exp3" = false ∧
                                      classical_probability_model "exp4" = false :=
by
  sorry

end classical_prob_exp_is_exp1_l2071_207141


namespace problem_statement_l2071_207196

noncomputable def a : ℝ := Real.sqrt 3 - Real.sqrt 11
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 11

theorem problem_statement : (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 3 :=
by
  -- conditions
  let a := Real.sqrt 3 - Real.sqrt 11
  let b := Real.sqrt 3 + Real.sqrt 11
  have h1 : a = Real.sqrt 3 - Real.sqrt 11 := rfl
  have h2 : b = Real.sqrt 3 + Real.sqrt 11 := rfl
  -- question statement
  sorry

end problem_statement_l2071_207196


namespace Mary_sleep_hours_for_avg_score_l2071_207159

def sleep_score_inverse_relation (sleep1 score1 sleep2 score2 : ℝ) : Prop :=
  sleep1 * score1 = sleep2 * score2

theorem Mary_sleep_hours_for_avg_score (h1 s1 s2 : ℝ) (h_eq : h1 = 6) (s1_eq : s1 = 60)
  (avg_score_cond : (s1 + s2) / 2 = 75) :
  ∃ h2 : ℝ, sleep_score_inverse_relation h1 s1 h2 s2 ∧ h2 = 4 := 
by
  sorry

end Mary_sleep_hours_for_avg_score_l2071_207159


namespace expand_square_binomial_l2071_207195

variable (m n : ℝ)

theorem expand_square_binomial : (3 * m - n) ^ 2 = 9 * m ^ 2 - 6 * m * n + n ^ 2 :=
by
  sorry

end expand_square_binomial_l2071_207195


namespace middle_angle_range_l2071_207114

theorem middle_angle_range (α β γ : ℝ) (h₀: α + β + γ = 180) (h₁: 0 < α) (h₂: 0 < β) (h₃: 0 < γ) (h₄: α ≤ β) (h₅: β ≤ γ) : 
  0 < β ∧ β < 90 :=
by
  sorry

end middle_angle_range_l2071_207114


namespace axis_of_symmetry_l2071_207166

-- Define points and the parabola equation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 2 5
def B := Point.mk 4 5

def parabola (b c : ℝ) (p : Point) : Prop :=
  p.y = 2 * p.x^2 + b * p.x + c

theorem axis_of_symmetry (b c : ℝ) (hA : parabola b c A) (hB : parabola b c B) : ∃ x_axis : ℝ, x_axis = 3 :=
by
  -- Proof to be provided
  sorry

end axis_of_symmetry_l2071_207166


namespace roots_cubic_polynomial_l2071_207149

theorem roots_cubic_polynomial (r s t : ℝ)
  (h₁ : 8 * r^3 + 1001 * r + 2008 = 0)
  (h₂ : 8 * s^3 + 1001 * s + 2008 = 0)
  (h₃ : 8 * t^3 + 1001 * t + 2008 = 0)
  (h₄ : r + s + t = 0) :
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 753 := 
sorry

end roots_cubic_polynomial_l2071_207149


namespace three_digit_cubes_divisible_by_8_and_9_l2071_207184

theorem three_digit_cubes_divisible_by_8_and_9 : 
  ∃! n : ℕ, (216 ≤ n^3 ∧ n^3 ≤ 999) ∧ (n % 6 = 0) :=
sorry

end three_digit_cubes_divisible_by_8_and_9_l2071_207184


namespace maximize_profit_l2071_207174

noncomputable def profit_function (x : ℝ) : ℝ := -3 * x^2 + 252 * x - 4860

theorem maximize_profit :
  (∀ x : ℝ, 30 ≤ x ∧ x ≤ 54 → profit_function x ≤ 432) ∧ profit_function 42 = 432 := sorry

end maximize_profit_l2071_207174


namespace cost_of_cheese_without_coupon_l2071_207106

theorem cost_of_cheese_without_coupon
    (cost_bread : ℝ := 4.00)
    (cost_meat : ℝ := 5.00)
    (coupon_cheese : ℝ := 1.00)
    (coupon_meat : ℝ := 1.00)
    (cost_sandwich : ℝ := 2.00)
    (num_sandwiches : ℝ := 10)
    (C : ℝ) : 
    (num_sandwiches * cost_sandwich = (cost_bread + (cost_meat - coupon_meat) + cost_meat + (C - coupon_cheese) + C)) → (C = 4.50) :=
by {
    sorry
}

end cost_of_cheese_without_coupon_l2071_207106


namespace man_older_than_son_l2071_207103

variables (M S : ℕ)

theorem man_older_than_son
  (h_son_age : S = 26)
  (h_future_age : M + 2 = 2 * (S + 2)) :
  M - S = 28 :=
by sorry

end man_older_than_son_l2071_207103


namespace isosceles_base_l2071_207145

theorem isosceles_base (s b : ℕ) 
  (h1 : 3 * s = 45) 
  (h2 : 2 * s + b = 40) 
  (h3 : s = 15): 
  b = 10 :=
  sorry

end isosceles_base_l2071_207145


namespace wickets_before_last_match_l2071_207198

-- Define the conditions
variable (W : ℕ)

-- Initial average
def initial_avg : ℝ := 12.4

-- Runs given in the last match
def runs_last_match : ℝ := 26

-- Wickets taken in the last match
def wickets_last_match : ℕ := 4

-- The new average after the last match
def new_avg : ℝ := initial_avg - 0.4

-- Prove the theorem
theorem wickets_before_last_match :
  (12.4 * W + runs_last_match) / (W + wickets_last_match) = new_avg → W = 55 :=
by
  sorry

end wickets_before_last_match_l2071_207198


namespace inequality_proof_l2071_207172

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) (h1 : 0 < n) (h2 : (Finset.univ.sum a) ≥ 0) :
  (Finset.univ.sum (λ i => Real.sqrt (a i ^ 2 + 1))) ≥
  Real.sqrt (2 * n * (Finset.univ.sum a)) :=
by
  sorry

end inequality_proof_l2071_207172


namespace unique_solution_quadratic_eq_l2071_207150

theorem unique_solution_quadratic_eq (q : ℚ) (hq : q ≠ 0) :
  (∀ x : ℚ, q * x^2 - 10 * x + 2 = 0) ↔ q = 12.5 :=
by
  sorry

end unique_solution_quadratic_eq_l2071_207150


namespace no_such_function_exists_l2071_207113

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 1 :=
by
  sorry

end no_such_function_exists_l2071_207113


namespace Micah_words_per_minute_l2071_207190

-- Defining the conditions
def Isaiah_words_per_minute : ℕ := 40
def extra_words : ℕ := 1200

-- Proving the statement that Micah can type 20 words per minute
theorem Micah_words_per_minute (Isaiah_wpm : ℕ) (extra_w : ℕ) : Isaiah_wpm = 40 → extra_w = 1200 → (Isaiah_wpm * 60 - extra_w) / 60 = 20 :=
by
  -- Sorry is used to skip the proof
  sorry

end Micah_words_per_minute_l2071_207190


namespace range_of_a_l2071_207197

theorem range_of_a (x : ℝ) (a : ℝ) (hx : 0 < x ∧ x < 4) : |x - 1| < a → a ≥ 3 := sorry

end range_of_a_l2071_207197


namespace sofiya_wins_l2071_207101

/-- Define the initial configuration and game rules -/
def initial_configuration : Type := { n : Nat // n = 2025 }

/--
  Define the game such that Sofiya starts and follows the strategy of always
  removing a neighbor from the arc with an even number of people.
-/
def winning_strategy (n : initial_configuration) : Prop :=
  n.1 % 2 = 1 ∧ 
  (∀ turn : Nat, turn % 2 = 0 → 
    (∃ arc : initial_configuration, arc.1 % 2 = 0 ∧ arc.1 < n.1) ∧
    (∀ marquis_turn : Nat, marquis_turn % 2 = 1 → 
      (∃ arc : initial_configuration, arc.1 % 2 = 1)))

/-- Sofiya has the winning strategy given the conditions of the game -/
theorem sofiya_wins : winning_strategy ⟨2025, rfl⟩ :=
sorry

end sofiya_wins_l2071_207101


namespace sum_reciprocal_transformation_l2071_207152

theorem sum_reciprocal_transformation 
  (a b c d S : ℝ) 
  (h1 : a + b + c + d = S)
  (h2 : 1 / a + 1 / b + 1 / c + 1 / d = S)
  (h3 : a ≠ 0 ∧ a ≠ 1)
  (h4 : b ≠ 0 ∧ b ≠ 1)
  (h5 : c ≠ 0 ∧ c ≠ 1)
  (h6 : d ≠ 0 ∧ d ≠ 1) :
  S = -2 :=
by
  sorry

end sum_reciprocal_transformation_l2071_207152


namespace circle_symmetric_line_l2071_207162

theorem circle_symmetric_line (a b : ℝ) (h : a < 2) (hb : b = -2) : a + b < 0 := by
  sorry

end circle_symmetric_line_l2071_207162


namespace range_of_x_l2071_207107

variable (a b x : ℝ)

def conditions : Prop := (a > 0) ∧ (b > 0)

theorem range_of_x (h : conditions a b) : (x^2 + 2*x < 8) -> (-4 < x) ∧ (x < 2) := 
by
  sorry

end range_of_x_l2071_207107


namespace sara_has_8_balloons_l2071_207157

-- Define the number of yellow balloons Tom has.
def tom_balloons : ℕ := 9 

-- Define the total number of yellow balloons.
def total_balloons : ℕ := 17

-- Define the number of yellow balloons Sara has.
def sara_balloons : ℕ := total_balloons - tom_balloons

-- Theorem stating that Sara has 8 yellow balloons.
theorem sara_has_8_balloons : sara_balloons = 8 := by
  -- Proof goes here. Adding sorry for now to skip the proof.
  sorry

end sara_has_8_balloons_l2071_207157


namespace intersection_points_count_l2071_207155

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2 ∧ (∀ x, f x = g x → x = x1 ∨ x = x2) :=
by
  sorry

end intersection_points_count_l2071_207155


namespace group_size_is_eight_l2071_207177

/-- Theorem: The number of people in the group is 8 if the 
average weight increases by 6 kg when a new person replaces 
one weighing 45 kg, and the weight of the new person is 93 kg. -/
theorem group_size_is_eight
    (n : ℕ)
    (H₁ : 6 * n = 48)
    (H₂ : 93 - 45 = 48) :
    n = 8 :=
by
  sorry

end group_size_is_eight_l2071_207177


namespace function_satisfies_condition_l2071_207100

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - x + 4)

theorem function_satisfies_condition :
  ∀ (x : ℝ), 2 * f (1 - x) + 1 = x * f x :=
by
  intro x
  unfold f
  sorry

end function_satisfies_condition_l2071_207100


namespace multiplication_problem_l2071_207186

noncomputable def problem_statement (x : ℂ) : Prop :=
  (x^4 + 30 * x^2 + 225) * (x^2 - 15) = x^6 - 3375

theorem multiplication_problem (x : ℂ) : 
  problem_statement x :=
sorry

end multiplication_problem_l2071_207186


namespace jordan_rectangle_width_l2071_207140

theorem jordan_rectangle_width
  (length_carol : ℕ) (width_carol : ℕ) (length_jordan : ℕ) (width_jordan : ℕ)
  (h1 : length_carol = 5) (h2 : width_carol = 24) (h3 : length_jordan = 2)
  (h4 : length_carol * width_carol = length_jordan * width_jordan) :
  width_jordan = 60 := by
  sorry

end jordan_rectangle_width_l2071_207140


namespace edward_money_proof_l2071_207132

def edward_total_money (earned_per_lawn : ℕ) (number_of_lawns : ℕ) (saved_up : ℕ) : ℕ :=
  earned_per_lawn * number_of_lawns + saved_up

theorem edward_money_proof :
  edward_total_money 8 5 7 = 47 :=
by
  sorry

end edward_money_proof_l2071_207132


namespace range_of_x_minus_cos_y_l2071_207165

theorem range_of_x_minus_cos_y {x y : ℝ} (h : x^2 + 2 * Real.cos y = 1) :
  ∃ (a b : ℝ), ∀ z, z = x - Real.cos y → a ≤ z ∧ z ≤ b ∧ a = -1 ∧ b = 1 + Real.sqrt 3 :=
by
  sorry

end range_of_x_minus_cos_y_l2071_207165


namespace expansion_contains_no_x2_l2071_207125

theorem expansion_contains_no_x2 (n : ℕ) (h1 : 5 ≤ n ∧ n ≤ 8) :
  ¬ (∃ k, (x + 1)^2 * (x + 1 / x^3)^n = k * x^2) → n = 7 :=
sorry

end expansion_contains_no_x2_l2071_207125


namespace birds_flew_up_l2071_207185

theorem birds_flew_up (original_birds total_birds birds_flew_up : ℕ) 
  (h1 : original_birds = 14)
  (h2 : total_birds = 35)
  (h3 : total_birds = original_birds + birds_flew_up) :
  birds_flew_up = 21 :=
by
  rw [h1, h2] at h3
  linarith

end birds_flew_up_l2071_207185


namespace painted_cubes_l2071_207160

theorem painted_cubes (n : ℕ) (h1 : 3 < n)
  (h2 : 6 * (n - 2)^2 = 12 * (n - 2)) :
  n = 4 := by
  sorry

end painted_cubes_l2071_207160


namespace percentage_of_water_in_mixture_l2071_207175

-- Conditions
def percentage_water_LiquidA : ℝ := 0.10
def percentage_water_LiquidB : ℝ := 0.15
def percentage_water_LiquidC : ℝ := 0.25

def volume_LiquidA (v : ℝ) : ℝ := 4 * v
def volume_LiquidB (v : ℝ) : ℝ := 3 * v
def volume_LiquidC (v : ℝ) : ℝ := 2 * v

-- Proof
theorem percentage_of_water_in_mixture (v : ℝ) :
  (percentage_water_LiquidA * volume_LiquidA v + percentage_water_LiquidB * volume_LiquidB v + percentage_water_LiquidC * volume_LiquidC v) / (volume_LiquidA v + volume_LiquidB v + volume_LiquidC v) * 100 = 15 :=
by
  sorry

end percentage_of_water_in_mixture_l2071_207175


namespace sum_of_distinct_integers_l2071_207158

theorem sum_of_distinct_integers (a b c d e : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_prod : (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120) : a + b + c + d + e = 39 :=
by
  sorry

end sum_of_distinct_integers_l2071_207158


namespace arithmetic_sequence_common_difference_l2071_207121

noncomputable def common_difference (a b : ℝ) : ℝ := a - 1

theorem arithmetic_sequence_common_difference :
  ∀ (a b : ℝ), 
    (a - 1 = b - a) → 
    ((a + 2) ^ 2 = 3 * (b + 5)) → 
    common_difference a b = 3 := by
  intros a b h1 h2
  sorry

end arithmetic_sequence_common_difference_l2071_207121


namespace find_physics_marks_l2071_207105

variable (P C M : ℕ)

theorem find_physics_marks
  (h1 : P + C + M = 225)
  (h2 : P + M = 180)
  (h3 : P + C = 140) : 
  P = 95 :=
by
  sorry

end find_physics_marks_l2071_207105


namespace equal_functions_A_l2071_207124

-- Define the functions
def f₁ (x : ℝ) : ℝ := x^2 - 2*x - 1
def f₂ (t : ℝ) : ℝ := t^2 - 2*t - 1

-- Theorem stating that f₁ is equal to f₂
theorem equal_functions_A : ∀ x : ℝ, f₁ x = f₂ x :=
by
  intros x
  sorry

end equal_functions_A_l2071_207124


namespace max_g_value_on_interval_l2071_207193

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_value_on_interval : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ ∀ y,  0 ≤ y ∧ y ≤ 2 → g x ≥ g y ∧ g x = 3 :=
-- Proof goes here
sorry

end max_g_value_on_interval_l2071_207193


namespace variance_uniform_l2071_207168

noncomputable def variance_of_uniform (α β : ℝ) (h : α < β) : ℝ :=
  let E := (α + β) / 2
  (β - α)^2 / 12

theorem variance_uniform (α β : ℝ) (h : α < β) :
  variance_of_uniform α β h = (β - α)^2 / 12 :=
by
  -- statement of proof only, actual proof here is sorry
  sorry

end variance_uniform_l2071_207168


namespace car_distance_calculation_l2071_207143

noncomputable def total_distance (u a v t1 t2: ℝ) : ℝ :=
  let d1 := (u * t1) + (1 / 2) * a * t1^2
  let d2 := v * t2
  d1 + d2

theorem car_distance_calculation :
  total_distance 30 5 60 2 3 = 250 :=
by
  unfold total_distance
  -- next steps include simplifying the math, but we'll defer details to proof
  sorry

end car_distance_calculation_l2071_207143


namespace largest_term_l2071_207111

-- Given conditions
def U : ℕ := 2 * (2010 ^ 2011)
def V : ℕ := 2010 ^ 2011
def W : ℕ := 2009 * (2010 ^ 2010)
def X : ℕ := 2 * (2010 ^ 2010)
def Y : ℕ := 2010 ^ 2010
def Z : ℕ := 2010 ^ 2009

-- Proposition to prove
theorem largest_term : 
  (U - V) > (V - W) ∧ 
  (U - V) > (W - X + 100) ∧ 
  (U - V) > (X - Y) ∧ 
  (U - V) > (Y - Z) := 
by 
  sorry

end largest_term_l2071_207111


namespace line_curve_intersection_l2071_207176

theorem line_curve_intersection (a : ℝ) : 
  (∃! (x y : ℝ), (y = a * (x + 2)) ∧ (x ^ 2 - y * |y| = 1)) ↔ a ∈ Set.Ioo (-Real.sqrt 3 / 3) 1 :=
by
  sorry

end line_curve_intersection_l2071_207176


namespace focus_of_parabola_l2071_207127

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the coordinates of the focus
def is_focus (x y : ℝ) : Prop := (x = 0) ∧ (y = 1)

-- The theorem statement
theorem focus_of_parabola : 
  (∃ x y : ℝ, parabola x y ∧ is_focus x y) :=
sorry

end focus_of_parabola_l2071_207127


namespace seven_power_product_prime_count_l2071_207137

theorem seven_power_product_prime_count (n : ℕ) :
  ∃ primes: List ℕ, (∀ p ∈ primes, Prime p) ∧ primes.prod = 7^(7^n) + 1 ∧ primes.length ≥ 2*n + 3 :=
by
  sorry

end seven_power_product_prime_count_l2071_207137


namespace jack_kids_solution_l2071_207123

def jack_kids (k : ℕ) : Prop :=
  7 * 3 * k = 63

theorem jack_kids_solution : jack_kids 3 :=
by
  sorry

end jack_kids_solution_l2071_207123


namespace length_of_arc_l2071_207182

variable {O A B : Type}
variable (angle_OAB : Real) (radius_OA : Real)

theorem length_of_arc (h1 : angle_OAB = 45) (h2 : radius_OA = 5) :
  (length_of_arc_AB = 5 * π / 4) :=
sorry

end length_of_arc_l2071_207182


namespace x_lt_y_l2071_207153

variable {a b c d x y : ℝ}

theorem x_lt_y 
  (ha : a > 1) 
  (hb : b > 1) 
  (hc : c > 1) 
  (hd : d > 1)
  (h1 : a^x + b^y = (a^2 + b^2)^x)
  (h2 : c^x + d^y = 2^y * (cd)^(y/2)) : 
  x < y :=
by 
  sorry

end x_lt_y_l2071_207153


namespace contrapositive_proposition_l2071_207110

-- Define the necessary elements in the context of real numbers
variables {a b c d : ℝ}

-- The statement of the contrapositive
theorem contrapositive_proposition : (a + c ≠ b + d) → (a ≠ b ∨ c ≠ d) :=
sorry

end contrapositive_proposition_l2071_207110


namespace man_speed_with_current_l2071_207104

theorem man_speed_with_current
  (v : ℝ)  -- man's speed in still water
  (current_speed : ℝ) (against_current_speed : ℝ)
  (h1 : against_current_speed = v - 3.2)
  (h2 : current_speed = 3.2) :
  v = 12.8 → (v + current_speed = 16.0) :=
by
  sorry

end man_speed_with_current_l2071_207104


namespace total_cans_from_recycling_l2071_207133

noncomputable def recycleCans (n : ℕ) : ℕ :=
  if n < 6 then 0 else n / 6 + recycleCans (n / 6 + n % 6)

theorem total_cans_from_recycling:
  recycleCans 486 = 96 :=
by
  sorry

end total_cans_from_recycling_l2071_207133


namespace find_m_l2071_207161

-- Define the functions f and g
def f (x m : ℝ) := x^2 - 2 * x + m
def g (x m : ℝ) := x^2 - 3 * x + 5 * m

-- The condition to be proved
theorem find_m (m : ℝ) : 3 * f 4 m = g 4 m → m = 10 :=
by
  sorry

end find_m_l2071_207161


namespace julia_total_watches_l2071_207171

-- Definitions based on conditions.
def silver_watches : Nat := 20
def bronze_watches : Nat := 3 * silver_watches
def total_silver_bronze_watches : Nat := silver_watches + bronze_watches
def gold_watches : Nat := total_silver_bronze_watches / 10

-- The final proof statement without providing the proof.
theorem julia_total_watches : (silver_watches + bronze_watches + gold_watches) = 88 :=
by 
  -- Since we don't need to provide the actual proof, we use sorry
  sorry

end julia_total_watches_l2071_207171


namespace smallest_N_proof_l2071_207191

theorem smallest_N_proof (N c1 c2 c3 c4 : ℕ)
  (h1 : N + c1 = 4 * c3 - 2)
  (h2 : N + c2 = 4 * c1 - 3)
  (h3 : 2 * N + c3 = 4 * c4 - 1)
  (h4 : 3 * N + c4 = 4 * c2) :
  N = 12 :=
sorry

end smallest_N_proof_l2071_207191


namespace area_ratio_trapezoid_triangle_l2071_207169

-- Define the geometric elements and given conditions.
variable (AB CD EAB ABCD : ℝ)
variable (trapezoid_ABCD : AB = 10)
variable (trapezoid_ABCD_CD : CD = 25)
variable (ratio_areas_EDC_EAB : (CD / AB)^2 = 25 / 4)
variable (trapezoid_relation : (ABCD + EAB) / EAB = 25 / 4)

-- The goal is to prove the ratio of the areas of triangle EAB to trapezoid ABCD.
theorem area_ratio_trapezoid_triangle :
  (EAB / ABCD) = 4 / 21 :=
by
  sorry

end area_ratio_trapezoid_triangle_l2071_207169


namespace trigonometric_problem_l2071_207109

open Real

noncomputable def problem1 (α : ℝ) : Prop :=
  2 * sin α = 2 * (sin (α / 2))^2 - 1

noncomputable def problem2 (β : ℝ) : Prop :=
  3 * (tan β)^2 - 2 * tan β = 1

theorem trigonometric_problem (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : π / 2 < β ∧ β < π)
  (h1 : problem1 α) (h2 : problem2 β) :
  sin (2 * α) + cos (2 * α) = -1 / 5 ∧ α + β = 7 * π / 4 :=
  sorry

end trigonometric_problem_l2071_207109


namespace tan_family_total_cost_l2071_207130

-- Define the number of people in each age group and respective discounts
def num_children : ℕ := 2
def num_adults : ℕ := 2
def num_seniors : ℕ := 2

def price_adult_ticket : ℝ := 10
def discount_senior : ℝ := 0.30
def discount_child : ℝ := 0.20
def group_discount : ℝ := 0.10

-- Calculate the cost for each group with discounts applied
def price_senior_ticket := price_adult_ticket * (1 - discount_senior)
def price_child_ticket := price_adult_ticket * (1 - discount_child)

-- Calculate the total cost of tickets before group discount
def total_cost_before_group_discount :=
  (price_senior_ticket * num_seniors) +
  (price_child_ticket * num_children) +
  (price_adult_ticket * num_adults)

-- Check if the family qualifies for group discount and apply if necessary
def total_cost_after_group_discount :=
  if (num_children + num_adults + num_seniors > 5)
  then total_cost_before_group_discount * (1 - group_discount)
  else total_cost_before_group_discount

-- Main theorem statement
theorem tan_family_total_cost : total_cost_after_group_discount = 45 := by
  sorry

end tan_family_total_cost_l2071_207130


namespace roses_given_to_mother_is_6_l2071_207128

-- Define the initial conditions
def initial_roses : ℕ := 20
def roses_to_grandmother : ℕ := 9
def roses_to_sister : ℕ := 4
def roses_kept : ℕ := 1

-- Define the expected number of roses given to mother
def roses_given_to_mother : ℕ := initial_roses - (roses_to_grandmother + roses_to_sister + roses_kept)

-- The theorem stating the number of roses given to the mother
theorem roses_given_to_mother_is_6 : roses_given_to_mother = 6 := by
  sorry

end roses_given_to_mother_is_6_l2071_207128


namespace find_a_and_x_range_l2071_207144

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem find_a_and_x_range :
  (∃ a, (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3)) →
  (∀ x, ∃ a, f x a ≤ 5 → 
    ((a = 1 → (0 ≤ x ∧ x ≤ 5)) ∧
     (a = 7 → (3 ≤ x ∧ x ≤ 8)))) :=
by sorry

end find_a_and_x_range_l2071_207144


namespace division_example_l2071_207181

theorem division_example : 0.45 / 0.005 = 90 := by
  sorry

end division_example_l2071_207181


namespace slope_tangent_at_pi_div_six_l2071_207129

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x - 2 * Real.cos x

theorem slope_tangent_at_pi_div_six : (deriv f π / 6) = 3 / 2 := 
by 
  sorry

end slope_tangent_at_pi_div_six_l2071_207129


namespace ratio_of_enclosed_area_l2071_207126

theorem ratio_of_enclosed_area
  (R : ℝ)
  (h_chords_eq : ∀ (A B C : ℝ), A = B → A = C)
  (h_inscribed_angle : ∀ (A B C O : ℝ), AOC = 30 * π / 180)
  : ((π * R^2 / 6) + (R^2 / 2)) / (π * R^2) = (π + 3) / (6 * π) :=
by
  sorry

end ratio_of_enclosed_area_l2071_207126


namespace book_pages_l2071_207115

theorem book_pages (P : ℕ) 
  (h1 : P / 2 + 11 + (P - (P / 2 + 11)) / 2 = 19)
  (h2 : P - (P / 2 + 11) = 2 * 19) : 
  P = 98 :=
by
  sorry

end book_pages_l2071_207115


namespace arithmetic_progression_20th_term_and_sum_l2071_207183

theorem arithmetic_progression_20th_term_and_sum :
  let a := 3
  let d := 4
  let n := 20
  let a_20 := a + (n - 1) * d
  let S_20 := n / 2 * (a + a_20)
  a_20 = 79 ∧ S_20 = 820 := by
    let a := 3
    let d := 4
    let n := 20
    let a_20 := a + (n - 1) * d
    let S_20 := n / 2 * (a + a_20)
    sorry

end arithmetic_progression_20th_term_and_sum_l2071_207183


namespace bags_of_sugar_bought_l2071_207139

-- Define the conditions as constants
def cups_at_home : ℕ := 3
def cups_per_bag : ℕ := 6
def cups_per_batter_dozen : ℕ := 1
def cups_per_frosting_dozen : ℕ := 2
def dozens_of_cupcakes : ℕ := 5

-- Prove that the number of bags of sugar Lillian bought is 2
theorem bags_of_sugar_bought : ∃ bags : ℕ, bags = 2 :=
by
  let total_cups_batter := dozens_of_cupcakes * cups_per_batter_dozen
  let total_cups_frosting := dozens_of_cupcakes * cups_per_frosting_dozen
  let total_cups_needed := total_cups_batter + total_cups_frosting
  let cups_to_buy := total_cups_needed - cups_at_home
  let bags := cups_to_buy / cups_per_bag
  have h : bags = 2 := sorry
  exact ⟨bags, h⟩

end bags_of_sugar_bought_l2071_207139


namespace smallest_prime_divisor_of_sum_l2071_207199

theorem smallest_prime_divisor_of_sum : ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ (3 ^ 15 + 11 ^ 21) :=
by
  sorry

end smallest_prime_divisor_of_sum_l2071_207199


namespace rectangle_perimeter_l2071_207179

-- Defining the given conditions
def rectangleArea := 4032
noncomputable def ellipseArea := 4032 * Real.pi
noncomputable def b := Real.sqrt 2016
noncomputable def a := 2 * Real.sqrt 2016

-- Problem statement: the perimeter of the rectangle
theorem rectangle_perimeter (x y : ℝ) (h1 : x * y = rectangleArea)
  (h2 : x + y = 2 * a) : 2 * (x + y) = 8 * Real.sqrt 2016 :=
by
  sorry

end rectangle_perimeter_l2071_207179
