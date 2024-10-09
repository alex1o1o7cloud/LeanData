import Mathlib

namespace find_f_ln2_l263_26361

variable (f : ℝ → ℝ)

-- Condition: f is an odd function
axiom odd_fn : ∀ x : ℝ, f (-x) = -f x

-- Condition: f(x) = e^(-x) - 2 for x < 0
axiom def_fn : ∀ x : ℝ, x < 0 → f x = Real.exp (-x) - 2

-- Problem: Find f(ln 2)
theorem find_f_ln2 : f (Real.log 2) = 0 := by
  sorry

end find_f_ln2_l263_26361


namespace ferry_speeds_l263_26314

theorem ferry_speeds (v_P v_Q : ℝ) 
  (h1: v_P = v_Q - 1) 
  (h2: 3 * v_P * 3 = v_Q * (3 + 5))
  : v_P = 8 := 
sorry

end ferry_speeds_l263_26314


namespace valid_integer_pairs_l263_26387

theorem valid_integer_pairs :
  { (x, y) : ℤ × ℤ |
    (∃ α β : ℝ, α^2 + β^2 < 4 ∧ α + β = (-x : ℝ) ∧ α * β = y ∧ x^2 - 4 * y ≥ 0) } =
  {(-2,1), (-1,-1), (-1,0), (0, -1), (0,0), (1,0), (1,-1), (2,1)} :=
sorry

end valid_integer_pairs_l263_26387


namespace distinct_integers_real_roots_l263_26394

theorem distinct_integers_real_roots (a b c : ℤ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > b) (h5 : b > c) :
    (∃ x : ℝ, x^2 + 2 * a * x + 3 * (b + c) = 0) :=
sorry

end distinct_integers_real_roots_l263_26394


namespace green_peaches_more_than_red_l263_26310

theorem green_peaches_more_than_red :
  let red_peaches := 5
  let green_peaches := 11
  (green_peaches - red_peaches) = 6 := by
  sorry

end green_peaches_more_than_red_l263_26310


namespace sum_q_p_eq_zero_l263_26376

def p (x : Int) : Int := x^2 - 4

def q (x : Int) : Int := 
  if x ≥ 0 then -x
  else x

def q_p (x : Int) : Int := q (p x)

#eval List.sum (List.map q_p [-3, -2, -1, 0, 1, 2, 3]) = 0

theorem sum_q_p_eq_zero :
  List.sum (List.map q_p [-3, -2, -1, 0, 1, 2, 3]) = 0 :=
sorry

end sum_q_p_eq_zero_l263_26376


namespace drink_price_half_promotion_l263_26351

theorem drink_price_half_promotion (P : ℝ) (h : P + (1/2) * P = 13.5) : P = 9 := 
by
  sorry

end drink_price_half_promotion_l263_26351


namespace fraction_add_eq_l263_26320

theorem fraction_add_eq (n : ℤ) :
  (3 + n) = 4 * ((4 + n) - 5) → n = 1 := sorry

end fraction_add_eq_l263_26320


namespace cubic_increasing_l263_26359

-- The definition of an increasing function
def increasing_function (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

-- The function y = x^3
def cubic_function (x : ℝ) : ℝ := x^3

-- The statement we want to prove
theorem cubic_increasing : increasing_function cubic_function :=
sorry

end cubic_increasing_l263_26359


namespace range_of_a_l263_26352

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → (a ≤ 1 ∨ a ≥ 3) :=
sorry

end range_of_a_l263_26352


namespace xiaomings_mother_money_l263_26363

-- Definitions for the conditions
def price_A : ℕ := 6
def price_B : ℕ := 9
def units_more_A := 2

-- Main statement to prove
theorem xiaomings_mother_money (x : ℕ) (M : ℕ) :
  M = 6 * x ∧ M = 9 * (x - 2) → M = 36 :=
by
  -- Assuming the conditions are given
  rintro ⟨hA, hB⟩
  -- The proof is omitted
  sorry

end xiaomings_mother_money_l263_26363


namespace number_of_pupils_l263_26309

-- Define the number of total people
def total_people : ℕ := 803

-- Define the number of parents
def parents : ℕ := 105

-- We need to prove the number of pupils is 698
theorem number_of_pupils : (total_people - parents) = 698 := 
by
  -- Skip the proof steps
  sorry

end number_of_pupils_l263_26309


namespace polynomial_coeff_sum_l263_26335

theorem polynomial_coeff_sum (a0 a1 a2 a3 : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a3 * x^3 + a2 * x^2 + a1 * x + a0) →
  a0 + a1 + a2 + a3 = 27 :=
by
  sorry

end polynomial_coeff_sum_l263_26335


namespace total_area_needed_l263_26357

-- Definitions based on conditions
def oak_trees_first_half := 100
def pine_trees_first_half := 100
def oak_trees_second_half := 150
def pine_trees_second_half := 150
def oak_tree_planting_ratio := 4
def pine_tree_planting_ratio := 2
def oak_tree_space := 4
def pine_tree_space := 2

-- Total area needed for tree planting during the entire year
theorem total_area_needed : (oak_trees_first_half * oak_tree_planting_ratio * oak_tree_space) + ((pine_trees_first_half + pine_trees_second_half) * pine_tree_planting_ratio * pine_tree_space) = 2600 :=
by
  sorry

end total_area_needed_l263_26357


namespace flyers_left_to_hand_out_l263_26349

-- Definitions for given conditions
def total_flyers : Nat := 1236
def jack_handout : Nat := 120
def rose_handout : Nat := 320

-- Statement of the problem
theorem flyers_left_to_hand_out : total_flyers - (jack_handout + rose_handout) = 796 :=
by
  -- proof goes here
  sorry

end flyers_left_to_hand_out_l263_26349


namespace total_tickets_sold_l263_26392

/-
Problem: Prove that the total number of tickets sold is 65 given the conditions.
Conditions:
1. Senior citizen tickets cost 10 dollars each.
2. Regular tickets cost 15 dollars each.
3. Total sales were 855 dollars.
4. 24 senior citizen tickets were sold.
-/

def senior_tickets_sold : ℕ := 24
def senior_ticket_cost : ℕ := 10
def regular_ticket_cost : ℕ := 15
def total_sales : ℕ := 855

theorem total_tickets_sold (R : ℕ) (H : total_sales = senior_tickets_sold * senior_ticket_cost + R * regular_ticket_cost) :
  senior_tickets_sold + R = 65 :=
by
  sorry

end total_tickets_sold_l263_26392


namespace find_number_l263_26322

theorem find_number : ∃ x : ℝ, (x / 5 + 7 = x / 4 - 7) ∧ x = 280 :=
by
  -- Here, we state the existence of a real number x
  -- such that the given condition holds and x = 280.
  sorry

end find_number_l263_26322


namespace roots_ratio_quadratic_l263_26303

theorem roots_ratio_quadratic (p : ℤ) (h : (∃ x1 x2 : ℤ, x1*x2 = -16 ∧ x1 + x2 = -p ∧ x2 = -4 * x1)) :
  p = 6 ∨ p = -6 :=
sorry

end roots_ratio_quadratic_l263_26303


namespace odd_number_expression_l263_26325

theorem odd_number_expression (o n : ℤ) (ho : o % 2 = 1) : (o^2 + n * o + 1) % 2 = 1 ↔ n % 2 = 1 := by
  sorry

end odd_number_expression_l263_26325


namespace tenth_term_geometric_sequence_l263_26355

theorem tenth_term_geometric_sequence :
  let a : ℚ := 5
  let r : ℚ := 3 / 4
  let a_n (n : ℕ) : ℚ := a * r ^ (n - 1)
  a_n 10 = 98415 / 262144 :=
by
  sorry

end tenth_term_geometric_sequence_l263_26355


namespace expected_value_is_minus_one_half_l263_26364

def prob_heads := 1 / 4
def prob_tails := 2 / 4
def prob_edge := 1 / 4
def win_heads := 4
def win_tails := -3
def win_edge := 0

theorem expected_value_is_minus_one_half :
  (prob_heads * win_heads + prob_tails * win_tails + prob_edge * win_edge) = -1 / 2 :=
by
  sorry

end expected_value_is_minus_one_half_l263_26364


namespace fermat_little_theorem_l263_26326

theorem fermat_little_theorem (N p : ℕ) (hp : Nat.Prime p) (hNp : ¬ p ∣ N) : p ∣ (N ^ (p - 1) - 1) := 
sorry

end fermat_little_theorem_l263_26326


namespace total_marbles_l263_26329

namespace MarbleBag

def numBlue : ℕ := 5
def numRed : ℕ := 9
def probRedOrWhite : ℚ := 5 / 6

theorem total_marbles (total_mar : ℕ) (numWhite : ℕ) (h1 : probRedOrWhite = (numRed + numWhite) / total_mar)
                      (h2 : total_mar = numBlue + numRed + numWhite) :
  total_mar = 30 :=
by
  sorry

end MarbleBag

end total_marbles_l263_26329


namespace prism_volume_l263_26379

theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 150 * Real.sqrt 5 :=
by
  sorry

end prism_volume_l263_26379


namespace distance_post_office_l263_26390

theorem distance_post_office 
  (D : ℝ)
  (speed_to_post_office : ℝ := 25)
  (speed_back : ℝ := 4)
  (total_time : ℝ := 5 + (48 / 60)) :
  (D / speed_to_post_office + D / speed_back = total_time) → D = 20 :=
by
  sorry

end distance_post_office_l263_26390


namespace points_lie_on_parabola_l263_26301

theorem points_lie_on_parabola (u : ℝ) :
  ∃ (x y : ℝ), x = 3^u - 4 ∧ y = 9^u - 7 * 3^u - 2 ∧ y = x^2 + x - 14 :=
by
  sorry

end points_lie_on_parabola_l263_26301


namespace Noemi_blackjack_loss_l263_26339

-- Define the conditions
def start_amount : ℕ := 1700
def end_amount : ℕ := 800
def roulette_loss : ℕ := 400

-- Define the total loss calculation
def total_loss : ℕ := start_amount - end_amount

-- Main theorem statement
theorem Noemi_blackjack_loss :
  ∃ (blackjack_loss : ℕ), blackjack_loss = total_loss - roulette_loss := 
by
  -- Start by calculating the total_loss
  let total_loss_eq := start_amount - end_amount
  -- The blackjack loss should be 900 - 400, which we claim to be 500
  use total_loss_eq - roulette_loss
  sorry

end Noemi_blackjack_loss_l263_26339


namespace dot_product_is_one_l263_26386

variable (a : ℝ × ℝ := (1, 1))
variable (b : ℝ × ℝ := (-1, 2))

theorem dot_product_is_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_is_one_l263_26386


namespace range_of_a_l263_26370

def p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a < 0 ∨ (1/4 < a ∧ a < 4) := 
sorry

end range_of_a_l263_26370


namespace geometric_progression_theorem_l263_26331

theorem geometric_progression_theorem 
  (a b c d : ℝ) (q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = a * q^2) 
  (h3 : d = a * q^3) 
  : (a - d)^2 = (a - c)^2 + (b - c)^2 + (b - d)^2 := 
by sorry

end geometric_progression_theorem_l263_26331


namespace right_triangle_30_60_90_l263_26398

theorem right_triangle_30_60_90 (a b : ℝ) (h : a = 15) :
  (b = 30) ∧ (b = 15 * Real.sqrt 3) :=
by
  sorry

end right_triangle_30_60_90_l263_26398


namespace rectangle_area_l263_26396

theorem rectangle_area (w l : ℝ) (hw : w = 2) (hl : l = 3) : w * l = 6 := by
  sorry

end rectangle_area_l263_26396


namespace remainder_twice_sum_first_150_mod_10000_eq_2650_l263_26393

theorem remainder_twice_sum_first_150_mod_10000_eq_2650 :
  let n := 150
  let S := n * (n + 1) / 2  -- Sum of first 150 numbers
  let result := 2 * S
  result % 10000 = 2650 :=
by
  sorry -- proof not required

end remainder_twice_sum_first_150_mod_10000_eq_2650_l263_26393


namespace tangent_product_power_l263_26368

noncomputable def tangent_product : ℝ :=
  (1 + Real.tan (1 * Real.pi / 180))
  * (1 + Real.tan (2 * Real.pi / 180))
  * (1 + Real.tan (3 * Real.pi / 180))
  * (1 + Real.tan (4 * Real.pi / 180))
  * (1 + Real.tan (5 * Real.pi / 180))
  * (1 + Real.tan (6 * Real.pi / 180))
  * (1 + Real.tan (7 * Real.pi / 180))
  * (1 + Real.tan (8 * Real.pi / 180))
  * (1 + Real.tan (9 * Real.pi / 180))
  * (1 + Real.tan (10 * Real.pi / 180))
  * (1 + Real.tan (11 * Real.pi / 180))
  * (1 + Real.tan (12 * Real.pi / 180))
  * (1 + Real.tan (13 * Real.pi / 180))
  * (1 + Real.tan (14 * Real.pi / 180))
  * (1 + Real.tan (15 * Real.pi / 180))
  * (1 + Real.tan (16 * Real.pi / 180))
  * (1 + Real.tan (17 * Real.pi / 180))
  * (1 + Real.tan (18 * Real.pi / 180))
  * (1 + Real.tan (19 * Real.pi / 180))
  * (1 + Real.tan (20 * Real.pi / 180))
  * (1 + Real.tan (21 * Real.pi / 180))
  * (1 + Real.tan (22 * Real.pi / 180))
  * (1 + Real.tan (23 * Real.pi / 180))
  * (1 + Real.tan (24 * Real.pi / 180))
  * (1 + Real.tan (25 * Real.pi / 180))
  * (1 + Real.tan (26 * Real.pi / 180))
  * (1 + Real.tan (27 * Real.pi / 180))
  * (1 + Real.tan (28 * Real.pi / 180))
  * (1 + Real.tan (29 * Real.pi / 180))
  * (1 + Real.tan (30 * Real.pi / 180))
  * (1 + Real.tan (31 * Real.pi / 180))
  * (1 + Real.tan (32 * Real.pi / 180))
  * (1 + Real.tan (33 * Real.pi / 180))
  * (1 + Real.tan (34 * Real.pi / 180))
  * (1 + Real.tan (35 * Real.pi / 180))
  * (1 + Real.tan (36 * Real.pi / 180))
  * (1 + Real.tan (37 * Real.pi / 180))
  * (1 + Real.tan (38 * Real.pi / 180))
  * (1 + Real.tan (39 * Real.pi / 180))
  * (1 + Real.tan (40 * Real.pi / 180))
  * (1 + Real.tan (41 * Real.pi / 180))
  * (1 + Real.tan (42 * Real.pi / 180))
  * (1 + Real.tan (43 * Real.pi / 180))
  * (1 + Real.tan (44 * Real.pi / 180))
  * (1 + Real.tan (45 * Real.pi / 180))
  * (1 + Real.tan (46 * Real.pi / 180))
  * (1 + Real.tan (47 * Real.pi / 180))
  * (1 + Real.tan (48 * Real.pi / 180))
  * (1 + Real.tan (49 * Real.pi / 180))
  * (1 + Real.tan (50 * Real.pi / 180))
  * (1 + Real.tan (51 * Real.pi / 180))
  * (1 + Real.tan (52 * Real.pi / 180))
  * (1 + Real.tan (53 * Real.pi / 180))
  * (1 + Real.tan (54 * Real.pi / 180))
  * (1 + Real.tan (55 * Real.pi / 180))
  * (1 + Real.tan (56 * Real.pi / 180))
  * (1 + Real.tan (57 * Real.pi / 180))
  * (1 + Real.tan (58 * Real.pi / 180))
  * (1 + Real.tan (59 * Real.pi / 180))
  * (1 + Real.tan (60 * Real.pi / 180))

theorem tangent_product_power : tangent_product = 2^30 := by
  sorry

end tangent_product_power_l263_26368


namespace find_wheel_diameter_l263_26313

noncomputable def wheel_diameter (revolutions distance : ℝ) (π_approx : ℝ) : ℝ := 
  distance / (π_approx * revolutions)

theorem find_wheel_diameter : wheel_diameter 47.04276615104641 4136 3.14159 = 27.99 :=
by
  sorry

end find_wheel_diameter_l263_26313


namespace coat_price_proof_l263_26389

variable (W : ℝ) -- wholesale price
variable (currentPrice : ℝ) -- current price of the coat

-- Condition 1: The retailer marked up the coat by 90%.
def markup_90 : Prop := currentPrice = 1.9 * W

-- Condition 2: Further $4 increase achieves a 100% markup.
def increase_4 : Prop := 2 * W - currentPrice = 4

-- Theorem: The current price of the coat is $76.
theorem coat_price_proof (h1 : markup_90 W currentPrice) (h2 : increase_4 W currentPrice) : currentPrice = 76 :=
sorry

end coat_price_proof_l263_26389


namespace trigonometric_identity_l263_26385

theorem trigonometric_identity : (1 / 4) * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 16 := by
  sorry

end trigonometric_identity_l263_26385


namespace solution_set_of_inequality_system_l263_26374

theorem solution_set_of_inequality_system :
  (6 - 2 * x ≥ 0) ∧ (2 * x + 4 > 0) ↔ (-2 < x ∧ x ≤ 3) := 
sorry

end solution_set_of_inequality_system_l263_26374


namespace square_root_area_ratio_l263_26340

theorem square_root_area_ratio 
  (side_C : ℝ) (side_D : ℝ)
  (hC : side_C = 45) 
  (hD : side_D = 60) : 
  Real.sqrt ((side_C^2) / (side_D^2)) = 3 / 4 := by
  -- proof goes here
  sorry

end square_root_area_ratio_l263_26340


namespace min_checkout_counters_l263_26344

variable (n : ℕ)
variable (x y : ℝ)

-- Conditions based on problem statement
axiom cond1 : 40 * y = 20 * x + n
axiom cond2 : 36 * y = 12 * x + n

theorem min_checkout_counters (m : ℕ) (h : 6 * m * y > 6 * x + n) : m ≥ 6 :=
  sorry

end min_checkout_counters_l263_26344


namespace part1_part2_l263_26369

theorem part1 (a : ℝ) : (a - 3 ≠ 0) ∧ (16 - 4 * (a-3) * (-1) = 0) → 
  a = -1 ∧ ∀ x : ℝ, (4 * x^2 + 4 * x + 1 = 0 ↔ x = -1/2) :=
sorry

theorem part2 (a : ℝ) : (a - 3 ≠ 0) ∧ (16 - 4 * (a-3) * (-1) > 0) → 
  a > -1 ∧ a ≠ 3 :=
sorry

end part1_part2_l263_26369


namespace find_a_l263_26380

theorem find_a (a : ℤ) (h₀ : 0 ≤ a ∧ a ≤ 13) (h₁ : 13 ∣ (51 ^ 2016 - a)) : a = 1 := sorry

end find_a_l263_26380


namespace molecular_weight_acetic_acid_l263_26345

-- Define atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of each atom in acetic acid
def num_C : ℕ := 2
def num_H : ℕ := 4
def num_O : ℕ := 2

-- Define the molecular formula of acetic acid
def molecular_weight_CH3COOH : ℝ :=
  num_C * atomic_weight_C +
  num_H * atomic_weight_H +
  num_O * atomic_weight_O

-- State the proposition
theorem molecular_weight_acetic_acid :
  molecular_weight_CH3COOH = 60.052 := by
  sorry

end molecular_weight_acetic_acid_l263_26345


namespace max_coefficient_terms_l263_26375

theorem max_coefficient_terms (x : ℝ) :
  let n := 8
  let T_3 := 7 * x^2
  let T_4 := 7 * x
  true := by
  sorry

end max_coefficient_terms_l263_26375


namespace number_of_pencils_l263_26366

theorem number_of_pencils (P L : ℕ) (h1 : (P : ℚ) / L = 5 / 6) (h2 : L = P + 6) : L = 36 :=
sorry

end number_of_pencils_l263_26366


namespace find_dividend_l263_26308

theorem find_dividend
  (R : ℕ)
  (Q : ℕ)
  (D : ℕ)
  (hR : R = 6)
  (hD_eq_5Q : D = 5 * Q)
  (hD_eq_3R_plus_2 : D = 3 * R + 2) :
  D * Q + R = 86 :=
by
  sorry

end find_dividend_l263_26308


namespace brian_shoes_l263_26342

theorem brian_shoes (J E B : ℕ) (h1 : J = E / 2) (h2 : E = 3 * B) (h3 : J + E + B = 121) : B = 22 :=
sorry

end brian_shoes_l263_26342


namespace five_p_squared_plus_two_q_squared_odd_p_squared_plus_pq_plus_q_squared_odd_l263_26319

variable (p q : ℕ)
variable (hp : p % 2 = 1)  -- p is odd
variable (hq : q % 2 = 1)  -- q is odd

theorem five_p_squared_plus_two_q_squared_odd 
    (hp : p % 2 = 1) 
    (hq : q % 2 = 1) : 
    (5 * p^2 + 2 * q^2) % 2 = 1 := 
sorry

theorem p_squared_plus_pq_plus_q_squared_odd 
    (hp : p % 2 = 1) 
    (hq : q % 2 = 1) : 
    (p^2 + p * q + q^2) % 2 = 1 := 
sorry

end five_p_squared_plus_two_q_squared_odd_p_squared_plus_pq_plus_q_squared_odd_l263_26319


namespace min_segments_to_erase_l263_26341

noncomputable def nodes (m n : ℕ) : ℕ := (m - 2) * (n - 2)

noncomputable def segments_to_erase (m n : ℕ) : ℕ := (nodes m n + 1) / 2

theorem min_segments_to_erase (m n : ℕ) (hm : m = 11) (hn : n = 11) :
  segments_to_erase m n = 41 := by
  sorry

end min_segments_to_erase_l263_26341


namespace perpendicular_vectors_l263_26311

def vector (α : Type) := (α × α)
def dot_product {α : Type} [Add α] [Mul α] (a b : vector α) : α :=
  a.1 * b.1 + a.2 * b.2

theorem perpendicular_vectors
    (a : vector ℝ) (b : vector ℝ)
    (h : dot_product a b = 0)
    (ha : a = (2, 4))
    (hb : b = (-1, n)) : 
    n = 1 / 2 := 
  sorry

end perpendicular_vectors_l263_26311


namespace find_first_factor_of_lcm_l263_26378

theorem find_first_factor_of_lcm (hcf : ℕ) (A : ℕ) (X : ℕ) (B : ℕ) (lcm_val : ℕ) 
  (h_hcf : hcf = 59)
  (h_A : A = 944)
  (h_lcm_val : lcm_val = 59 * X * 16)
  (h_A_lcm : A = lcm_val) :
  X = 1 := 
by
  sorry

end find_first_factor_of_lcm_l263_26378


namespace not_right_triangle_condition_C_l263_26365

theorem not_right_triangle_condition_C :
  ∀ (a b c : ℝ), 
    (a^2 = b^2 + c^2) ∨
    (∀ (angleA angleB angleC : ℝ), angleA = angleB + angleC ∧ angleA + angleB + angleC = 180) ∨
    (∀ (angleA angleB angleC : ℝ), angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5) ∨
    (a^2 / b^2 = 1 / 2 ∧ b^2 / c^2 = 2 / 3) ->
    ¬ (∀ (angleA angleB angleC : ℝ), angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5 -> angleA = 90 ∨ angleB = 90 ∨ angleC = 90) :=
by
  intro a b c h
  cases h
  case inl h1 =>
    -- Option A: b^2 = a^2 - c^2
    sorry
  case inr h2 =>
    cases h2
    case inl h3 => 
      -- Option B: angleA = angleB + angleC
      sorry
    case inr h4 =>
      cases h4
      case inl h5 =>
        -- Option C: angleA : angleB : angleC = 3 : 4 : 5
        sorry
      case inr h6 =>
        -- Option D: a^2 : b^2 : c^2 = 1 : 2 : 3
        sorry

end not_right_triangle_condition_C_l263_26365


namespace sum_x_coordinates_common_points_l263_26306

theorem sum_x_coordinates_common_points (x y : ℤ) (h1 : y ≡ 3 * x + 5 [ZMOD 13]) (h2 : y ≡ 9 * x + 1 [ZMOD 13]) : x ≡ 5 [ZMOD 13] :=
sorry

end sum_x_coordinates_common_points_l263_26306


namespace unknown_card_value_l263_26377

theorem unknown_card_value (cards_total : ℕ)
  (p1_hand : ℕ) (p1_hand_extra : ℕ) (table_card1 : ℕ) (total_card_values : ℕ)
  (sum_removed_cards_sets : ℕ)
  (n : ℕ) :
  cards_total = 40 ∧ 
  p1_hand = 5 ∧ 
  p1_hand_extra = 3 ∧ 
  table_card1 = 9 ∧ 
  total_card_values = 220 ∧ 
  sum_removed_cards_sets = 15 * n → 
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 10 ∧ total_card_values = p1_hand + p1_hand_extra + table_card1 + x + sum_removed_cards_sets → 
  x = 8 := 
sorry

end unknown_card_value_l263_26377


namespace total_shaded_area_l263_26323

theorem total_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) :
  1 * S ^ 2 + 8 * (T ^ 2) = 13.5 := by
  sorry

end total_shaded_area_l263_26323


namespace hyperbola_equation_l263_26360

noncomputable def sqrt_cubed := Real.sqrt 3

theorem hyperbola_equation
  (P : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (hP : P = (1, sqrt_cubed))
  (hAsymptote : (1 / a)^2 - (sqrt_cubed / b)^2 = 0)
  (hAngle : ∀ F : ℝ × ℝ, ∀ O : ℝ × ℝ, (F.1 - 1)^2 + (F.2 - sqrt_cubed)^2 + F.1^2 + F.2^2 = 16) :
  (a^2 = 4) ∧ (b^2 = 12) ∧ (c = 4) →
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1 :=
by
  sorry

end hyperbola_equation_l263_26360


namespace sum_diff_9114_l263_26343

def sum_odd_ints (n : ℕ) := (n + 1) / 2 * (1 + n)
def sum_even_ints (n : ℕ) := n / 2 * (2 + n)

theorem sum_diff_9114 : 
  let m := sum_odd_ints 215
  let t := sum_even_ints 100
  m - t = 9114 :=
by
  sorry

end sum_diff_9114_l263_26343


namespace problem_solution_l263_26383

noncomputable def f (x a : ℝ) : ℝ :=
  2 * (Real.cos x)^2 - 2 * a * Real.cos x - (2 * a + 1)

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a < 2 then -a^2 / 2 - 2 * a - 1
  else 1 - 4 * a

theorem problem_solution :
  g a = 1 ∨ g a = (-a^2 / 2 - 2 * a - 1) ∨ g a = 1 - 4 * a →
  (∀ a, g a = 1 / 2 → a = -1) ∧ (f x (-1) ≤ 5) :=
sorry

end problem_solution_l263_26383


namespace binom_21_13_l263_26332

theorem binom_21_13 : (Nat.choose 21 13) = 203490 :=
by
  have h1 : (Nat.choose 20 13) = 77520 := by sorry
  have h2 : (Nat.choose 20 12) = 125970 := by sorry
  have pascal : (Nat.choose 21 13) = (Nat.choose 20 13) + (Nat.choose 20 12) :=
    by rw [Nat.choose_succ_succ, h1, h2]
  exact pascal

end binom_21_13_l263_26332


namespace troy_initial_straws_l263_26397

theorem troy_initial_straws (total_piglets : ℕ) (straws_per_piglet : ℕ)
  (fraction_adult_pigs : ℚ) (fraction_piglets : ℚ) 
  (adult_pigs_straws : ℕ) (piglets_straws : ℕ) 
  (total_straws : ℕ) (initial_straws : ℚ) :
  total_piglets = 20 →
  straws_per_piglet = 6 →
  fraction_adult_pigs = 3 / 5 →
  fraction_piglets = 3 / 5 →
  piglets_straws = total_piglets * straws_per_piglet →
  adult_pigs_straws = piglets_straws →
  total_straws = piglets_straws + adult_pigs_straws →
  (fraction_adult_pigs + fraction_piglets) * initial_straws = total_straws →
  initial_straws = 200 := 
by 
  sorry

end troy_initial_straws_l263_26397


namespace line_equation_l263_26327

theorem line_equation 
    (passes_through_intersection : ∃ (P : ℝ × ℝ), P ∈ { (x, y) | 11 * x + 3 * y - 7 = 0 } ∧ P ∈ { (x, y) | 12 * x + y - 19 = 0 })
    (equidistant_from_A_and_B : ∃ (P : ℝ × ℝ), dist P (3, -2) = dist P (-1, 6)) :
    ∃ (a b c : ℝ), (a = 7 ∧ b = 1 ∧ c = -9) ∨ (a = 2 ∧ b = 1 ∧ c = 1) ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := 
sorry

end line_equation_l263_26327


namespace find_original_selling_price_l263_26333

variable (x : ℝ) (discount_rate : ℝ) (final_price : ℝ)

def original_selling_price_exists (x : ℝ) (discount_rate : ℝ) (final_price : ℝ) : Prop :=
  (x * (1 - discount_rate) = final_price) → (x = 700)

theorem find_original_selling_price
  (discount_rate : ℝ := 0.20)
  (final_price : ℝ := 560) :
  ∃ x : ℝ, original_selling_price_exists x discount_rate final_price :=
by
  use 700
  sorry

end find_original_selling_price_l263_26333


namespace Claire_takes_6_photos_l263_26302

-- Define the number of photos Claire has taken
variable (C : ℕ)

-- Define the conditions as stated in the problem
def Lisa_photos := 3 * C
def Robert_photos := C + 12
def same_number_photos := Lisa_photos C = Robert_photos C

-- The goal is to prove that C = 6
theorem Claire_takes_6_photos (h : same_number_photos C) : C = 6 := by
  sorry

end Claire_takes_6_photos_l263_26302


namespace smallest_marble_count_l263_26358

theorem smallest_marble_count (N : ℕ) (a b c : ℕ) (h1 : N > 1)
  (h2 : N ≡ 2 [MOD 5])
  (h3 : N ≡ 2 [MOD 7])
  (h4 : N ≡ 2 [MOD 9]) : N = 317 :=
sorry

end smallest_marble_count_l263_26358


namespace fraction_simplification_l263_26312

theorem fraction_simplification : (8 : ℝ) / (4 * 25) = 0.08 :=
by
  sorry

end fraction_simplification_l263_26312


namespace david_age_l263_26384

theorem david_age (x : ℕ) (y : ℕ) (h1 : y = x + 7) (h2 : y = 2 * x) : x = 7 :=
by
  sorry

end david_age_l263_26384


namespace algae_free_day_22_l263_26346

def algae_coverage (day : ℕ) : ℝ :=
if day = 25 then 1 else 2 ^ (25 - day)

theorem algae_free_day_22 :
  1 - algae_coverage 22 = 0.875 :=
by
  -- Proof to be filled in
  sorry

end algae_free_day_22_l263_26346


namespace total_items_in_jar_l263_26356

/--
A jar contains 3409.0 pieces of candy and 145.0 secret eggs with a prize.
We aim to prove that the total number of items in the jar is 3554.0.
-/
theorem total_items_in_jar :
  let number_of_pieces_of_candy := 3409.0
  let number_of_secret_eggs := 145.0
  number_of_pieces_of_candy + number_of_secret_eggs = 3554.0 :=
by
  sorry

end total_items_in_jar_l263_26356


namespace lines_intersect_and_sum_l263_26316

theorem lines_intersect_and_sum (a b : ℝ) :
  (∃ x y : ℝ, x = (1 / 3) * y + a ∧ y = (1 / 3) * x + b ∧ x = 3 ∧ y = 3) →
  a + b = 4 :=
by
  sorry

end lines_intersect_and_sum_l263_26316


namespace like_terms_exponent_l263_26338

theorem like_terms_exponent (a : ℝ) : (2 * a = a + 3) → a = 3 := 
by
  intros h
  -- Proof here
  sorry

end like_terms_exponent_l263_26338


namespace commute_solution_l263_26337

noncomputable def commute_problem : Prop :=
  let t : ℝ := 1                -- 1 hour from 7:00 AM to 8:00 AM
  let late_minutes : ℝ := 5 / 60  -- 5 minutes = 5/60 hours
  let early_minutes : ℝ := 4 / 60 -- 4 minutes = 4/60 hours
  let speed1 : ℝ := 30          -- 30 mph
  let speed2 : ℝ := 70          -- 70 mph
  let d1 : ℝ := speed1 * (t + late_minutes)
  let d2 : ℝ := speed2 * (t - early_minutes)

  ∃ (speed : ℝ), d1 = d2 ∧ speed = d1 / t ∧ speed = 32.5

theorem commute_solution : commute_problem :=
by sorry

end commute_solution_l263_26337


namespace average_of_five_quantities_l263_26382

theorem average_of_five_quantities (a b c d e : ℝ) 
  (h1 : (a + b + c) / 3 = 4) 
  (h2 : (d + e) / 2 = 33) : 
  ((a + b + c + d + e) / 5) = 15.6 := 
sorry

end average_of_five_quantities_l263_26382


namespace find_c_l263_26347

theorem find_c (c : ℝ) (h1 : ∃ x : ℝ, (⌊c⌋ : ℝ) = x ∧ 3 * x^2 + 12 * x - 27 = 0)
                      (h2 : ∃ x : ℝ, (c - ⌊c⌋) = x ∧ 4 * x^2 - 12 * x + 5 = 0) :
                      c = -8.5 :=
by
  sorry

end find_c_l263_26347


namespace percentage_invalid_l263_26399

theorem percentage_invalid (total_votes valid_votes_A : ℕ) (percent_A : ℝ) (total_valid_votes : ℝ) (percent_invalid : ℝ) :
  total_votes = 560000 →
  valid_votes_A = 333200 →
  percent_A = 0.70 →
  (1 - percent_invalid / 100) * total_votes = total_valid_votes →
  percent_A * total_valid_votes = valid_votes_A →
  percent_invalid = 15 :=
by
  intros h_total_votes h_valid_votes_A h_percent_A h_total_valid_votes h_valid_poll_A
  sorry

end percentage_invalid_l263_26399


namespace abs_neg_2023_eq_2023_l263_26395

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_eq_2023_l263_26395


namespace unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2_l263_26354

theorem unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2 : ∃! (x : ℤ), x - 9 / (x - 2) = 5 - 9 / (x - 2) := 
by
  sorry

end unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2_l263_26354


namespace stratified_sampling_sum_l263_26381

theorem stratified_sampling_sum :
  let grains := 40
  let vegetable_oils := 10
  let animal_foods := 30
  let fruits_and_vegetables := 20
  let sample_size := 20
  let total_food_types := grains + vegetable_oils + animal_foods + fruits_and_vegetables
  let sampling_fraction := sample_size / total_food_types
  let number_drawn := sampling_fraction * (vegetable_oils + fruits_and_vegetables)
  number_drawn = 6 :=
by
  sorry

end stratified_sampling_sum_l263_26381


namespace certain_number_l263_26350

theorem certain_number (x : ℝ) (h : (2.28 * x) / 6 = 480.7) : x = 1265.0 := 
by 
  sorry

end certain_number_l263_26350


namespace solution_pairs_l263_26321

theorem solution_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x ^ 2 + y ^ 2 - 5 * x * y + 5 = 0 ↔ (x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 9 ∧ y = 2) ∨ (x = 1 ∧ y = 2) := by
  sorry

end solution_pairs_l263_26321


namespace rhombus_diagonals_ratio_l263_26324

theorem rhombus_diagonals_ratio (a b d1 d2 : ℝ) 
  (h1: a > 0) (h2: b > 0)
  (h3: d1 = 2 * (a / Real.cos θ))
  (h4: d2 = 2 * (b / Real.cos θ)) :
  d1 / d2 = a / b := 
sorry

end rhombus_diagonals_ratio_l263_26324


namespace steven_peaches_l263_26315

theorem steven_peaches (jake_peaches : ℕ) (steven_peaches : ℕ) (h1 : jake_peaches = 3) (h2 : jake_peaches + 10 = steven_peaches) : steven_peaches = 13 :=
by
  sorry

end steven_peaches_l263_26315


namespace valves_fill_pool_l263_26334

theorem valves_fill_pool
  (a b c d : ℝ)
  (h1 : 1 / a + 1 / b + 1 / c = 1 / 12)
  (h2 : 1 / b + 1 / c + 1 / d = 1 / 15)
  (h3 : 1 / a + 1 / d = 1 / 20) :
  1 / a + 1 / b + 1 / c + 1 / d = 1 / 10 := 
sorry

end valves_fill_pool_l263_26334


namespace solve_for_a_l263_26372

theorem solve_for_a (a : ℝ) (h : 50 - |a - 2| = |4 - a|) :
  a = -22 ∨ a = 28 :=
sorry

end solve_for_a_l263_26372


namespace number_is_28_l263_26362

-- Definitions from conditions in part a
def inner_expression := 15 - 15
def middle_expression := 37 - inner_expression
def outer_expression (some_number : ℕ) := 45 - (some_number - middle_expression)

-- Lean 4 statement to state the proof problem
theorem number_is_28 (some_number : ℕ) (h : outer_expression some_number = 54) : some_number = 28 := by
  sorry

end number_is_28_l263_26362


namespace bookstore_discount_l263_26300

theorem bookstore_discount (P MP price_paid : ℝ) (h1 : MP = 0.80 * P) (h2 : price_paid = 0.60 * MP) :
  price_paid / P = 0.48 :=
by
  sorry

end bookstore_discount_l263_26300


namespace quadratic_algebraic_expression_l263_26371

theorem quadratic_algebraic_expression (a b : ℝ) (h₁ : a^2 - 3 * a + 1 = 0) (h₂ : b^2 - 3 * b + 1 = 0) :
    a + b - a * b = 2 := by
  sorry

end quadratic_algebraic_expression_l263_26371


namespace jessica_found_seashells_l263_26305

-- Define the given conditions
def mary_seashells : ℕ := 18
def total_seashells : ℕ := 59

-- Define the goal for the number of seashells Jessica found
def jessica_seashells (mary_seashells total_seashells : ℕ) : ℕ := total_seashells - mary_seashells

-- The theorem stating Jessica found 41 seashells
theorem jessica_found_seashells : jessica_seashells mary_seashells total_seashells = 41 := by
  -- We assume the conditions and skip the proof
  sorry

end jessica_found_seashells_l263_26305


namespace solve_system_l263_26330

theorem solve_system :
  ∃ (x y : ℝ), (2 * x - y = 1) ∧ (x + y = 2) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end solve_system_l263_26330


namespace sample_variance_is_two_l263_26304

theorem sample_variance_is_two (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) : (1 / 5) * ((-1 - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end sample_variance_is_two_l263_26304


namespace choose_three_consecutive_circles_l263_26367

theorem choose_three_consecutive_circles (n : ℕ) (hn : n = 33) : 
  ∃ (ways : ℕ), ways = 57 :=
by
  sorry

end choose_three_consecutive_circles_l263_26367


namespace solve_problem_l263_26348

theorem solve_problem
    (x y z : ℝ)
    (h1 : x > 0)
    (h2 : y > 0)
    (h3 : z > 0)
    (h4 : x^2 + x * y + y^2 = 2)
    (h5 : y^2 + y * z + z^2 = 5)
    (h6 : z^2 + z * x + x^2 = 3) :
    x * y + y * z + z * x = 2 * Real.sqrt 2 := 
by
  sorry

end solve_problem_l263_26348


namespace solution_of_inequality_l263_26318

theorem solution_of_inequality (a : ℝ) :
  (a = 0 → ∀ x : ℝ, ax^2 - (a + 1) * x + 1 < 0 ↔ x > 1) ∧
  (a < 0 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ x > 1 ∨ x < 1/a)) ∧
  (0 < a ∧ a < 1 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ 1 < x ∧ x < 1/a)) ∧
  (a > 1 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ 1/a < x ∧ x < 1)) ∧
  (a = 1 → ∀ x : ℝ, ¬(ax^2 - (a + 1) * x + 1 < 0)) :=
by
  sorry

end solution_of_inequality_l263_26318


namespace units_digit_of_product_of_odds_between_10_and_50_l263_26328

def product_of_odds_units_digit : ℕ :=
  let odds := [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
  let product := odds.foldl (· * ·) 1
  product % 10

theorem units_digit_of_product_of_odds_between_10_and_50 : product_of_odds_units_digit = 5 :=
  sorry

end units_digit_of_product_of_odds_between_10_and_50_l263_26328


namespace jumping_contest_l263_26388

theorem jumping_contest (grasshopper_jump frog_jump : ℕ) (h_grasshopper : grasshopper_jump = 9) (h_frog : frog_jump = 12) : frog_jump - grasshopper_jump = 3 := by
  ----- h_grasshopper and h_frog are our conditions -----
  ----- The goal is to prove frog_jump - grasshopper_jump = 3 -----
  sorry

end jumping_contest_l263_26388


namespace clean_room_to_homework_ratio_l263_26391

-- Define the conditions
def timeHomework : ℕ := 30
def timeWalkDog : ℕ := timeHomework + 5
def timeTrash : ℕ := timeHomework / 6
def totalTimeAvailable : ℕ := 120
def remainingTime : ℕ := 35

-- Definition to calculate total time spent on other tasks
def totalTimeOnOtherTasks : ℕ := timeHomework + timeWalkDog + timeTrash

-- Definition to calculate the time to clean the room
def timeCleanRoom : ℕ := totalTimeAvailable - remainingTime - totalTimeOnOtherTasks

-- The theorem to prove the ratio
theorem clean_room_to_homework_ratio : (timeCleanRoom : ℚ) / (timeHomework : ℚ) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end clean_room_to_homework_ratio_l263_26391


namespace line_through_A_and_B_l263_26317

variables (x y x₁ y₁ x₂ y₂ : ℝ)

-- Conditions
def condition1 : Prop := 3 * x₁ - 4 * y₁ - 2 = 0
def condition2 : Prop := 3 * x₂ - 4 * y₂ - 2 = 0

-- Proof that the line passing through A(x₁, y₁) and B(x₂, y₂) is 3x - 4y - 2 = 0
theorem line_through_A_and_B (h1 : condition1 x₁ y₁) (h2 : condition2 x₂ y₂) :
    ∀ (x y : ℝ), (∃ k : ℝ, x = x₁ + k * (x₂ - x₁) ∧ y = y₁ + k * (y₂ - y₁)) → 3 * x - 4 * y - 2 = 0 :=
sorry

end line_through_A_and_B_l263_26317


namespace ticket_price_increase_one_day_later_l263_26307

noncomputable def ticket_price : ℝ := 1050
noncomputable def days_before_departure : ℕ := 14
noncomputable def daily_increase_rate : ℝ := 0.05

theorem ticket_price_increase_one_day_later :
  ∀ (price : ℝ) (days : ℕ) (rate : ℝ), price = ticket_price → days = days_before_departure → rate = daily_increase_rate →
  price * rate = 52.50 :=
by
  intros price days rate hprice hdays hrate
  rw [hprice, hrate]
  exact sorry

end ticket_price_increase_one_day_later_l263_26307


namespace solve_problem_l263_26353
open Complex

noncomputable def problem (a b c d : ℝ) (ω : ℂ) : Prop :=
  (a ≠ -1) ∧ (b ≠ -1) ∧ (c ≠ -1) ∧ (d ≠ -1) ∧ (ω ^ 4 = 1) ∧ (ω ≠ 1) ∧
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 3 / ω ^ 2)
  
theorem solve_problem {a b c d : ℝ} {ω : ℂ} (h : problem a b c d ω) : 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2) :=
sorry

end solve_problem_l263_26353


namespace lower_percentage_increase_l263_26373

theorem lower_percentage_increase (E P : ℝ) (h1 : 1.26 * E = 693) (h2 : (1 + P) * E = 660) : P = 0.2 := by
  sorry

end lower_percentage_increase_l263_26373


namespace pool_water_volume_after_evaporation_l263_26336

theorem pool_water_volume_after_evaporation :
  let initial_volume := 300
  let evaporation_first_15_days := 1 -- in gallons per day
  let evaporation_next_15_days := 2 -- in gallons per day
  initial_volume - (15 * evaporation_first_15_days + 15 * evaporation_next_15_days) = 255 :=
by
  sorry

end pool_water_volume_after_evaporation_l263_26336
