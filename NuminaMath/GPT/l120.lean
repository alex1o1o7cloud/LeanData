import Mathlib

namespace pyramid_area_l120_120605

noncomputable def base_edge : ℝ := 8
noncomputable def lateral_edge : ℝ := 7
noncomputable def height := real.sqrt (lateral_edge^2 - (base_edge / 2)^2)
noncomputable def one_triangle_area := (1 / 2) * base_edge * height
noncomputable def total_area := 4 * one_triangle_area

theorem pyramid_area : total_area = 16 * real.sqrt 33 := by
  sorry

end pyramid_area_l120_120605


namespace op_add_mul_example_l120_120367

def op_add (a b : ℤ) : ℤ := a + b - 1
def op_mul (a b : ℤ) : ℤ := a * b - 1

theorem op_add_mul_example : op_mul (op_add 6 8) (op_add 3 5) = 90 :=
by
  -- Rewriting it briefly without proof steps
  sorry

end op_add_mul_example_l120_120367


namespace B_subsetneq_A_l120_120248

def A : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }
def B : Set ℝ := { x : ℝ | 1 - x^2 > 0 }

theorem B_subsetneq_A : B ⊂ A :=
by
  sorry

end B_subsetneq_A_l120_120248


namespace find_height_on_BC_l120_120728

noncomputable def height_on_BC (a b : ℝ) (A B C : ℝ) : ℝ := b * (Real.sin C)

theorem find_height_on_BC (A B C a b h : ℝ)
  (h_a: a = Real.sqrt 3)
  (h_b: b = Real.sqrt 2)
  (h_cos: 1 + 2 * Real.cos (B + C) = 0)
  (h_A: A = Real.pi / 3)
  (h_B: B = Real.pi / 4)
  (h_C: C = 5 * Real.pi / 12)
  (h_h: h = height_on_BC a b A B C) :
  h = (Real.sqrt 3 + 1) / 2 :=
sorry

end find_height_on_BC_l120_120728


namespace tax_is_one_l120_120069

-- Define costs
def cost_eggs : ℕ := 3
def cost_pancakes : ℕ := 2
def cost_cocoa : ℕ := 2

-- Initial order
def initial_eggs := 1
def initial_pancakes := 1
def initial_mugs_of_cocoa := 2

-- Additional order by Ben
def additional_pancakes := 1
def additional_mugs_of_cocoa := 1

-- Calculate costs
def initial_cost : ℕ := initial_eggs * cost_eggs + initial_pancakes * cost_pancakes + initial_mugs_of_cocoa * cost_cocoa
def additional_cost : ℕ := additional_pancakes * cost_pancakes + additional_mugs_of_cocoa * cost_cocoa
def total_cost_before_tax : ℕ := initial_cost + additional_cost

-- Payment and change
def total_paid : ℕ := 15
def change : ℕ := 1
def actual_payment : ℕ := total_paid - change

-- Calculate tax
def tax : ℕ := actual_payment - total_cost_before_tax

-- Prove that the tax is $1
theorem tax_is_one : tax = 1 :=
by
  sorry

end tax_is_one_l120_120069


namespace pyramid_area_l120_120601

theorem pyramid_area (base_edge lateral_edge : ℝ) (H_base_edge : base_edge = 8) (H_lateral_edge : lateral_edge = 7) :
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_area_l120_120601


namespace complement_of_M_in_U_is_correct_l120_120048

def U : Set ℤ := {1, -2, 3, -4, 5, -6}
def M : Set ℤ := {1, -2, 3, -4}
def complement_M_in_U : Set ℤ := {5, -6}

theorem complement_of_M_in_U_is_correct : (U \ M) = complement_M_in_U := by
  sorry

end complement_of_M_in_U_is_correct_l120_120048


namespace matroskin_milk_amount_l120_120045

theorem matroskin_milk_amount :
  ∃ S M x : ℝ, S + M = 10 ∧ (S - x) = (1 / 3) * S ∧ (M + x) = 3 * M ∧ (M + x) = 7.5 := 
sorry

end matroskin_milk_amount_l120_120045


namespace find_angle_D_l120_120866

variable (A B C D : ℝ)
variable (h1 : A + B = 180)
variable (h2 : C = D)
variable (h3 : C + 50 + 60 = 180)

theorem find_angle_D : D = 70 := by
  sorry

end find_angle_D_l120_120866


namespace det_A_eq_6_l120_120741

open Matrix

variables {R : Type*} [Field R]

def A (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![a, 2], ![-3, d]]

def B (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![2 * a, 1], ![-1, d]]

noncomputable def B_inv (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  let detB := (2 * a * d + 1)
  ![![d / detB, -1 / detB], ![1 / detB, (2 * a) / detB]]

theorem det_A_eq_6 (a d : R) (hB_inv : (A a d) + (B_inv a d) = 0) : det (A a d) = 6 :=
  sorry

end det_A_eq_6_l120_120741


namespace placemat_length_l120_120490

noncomputable def calculate_placemat_length
    (R : ℝ)
    (num_mats : ℕ)
    (mat_width : ℝ)
    (overlap_ratio : ℝ) : ℝ := 
    let circumference := 2 * Real.pi * R
    let arc_length := circumference / num_mats
    let angle := 2 * Real.pi / num_mats
    let chord_length := 2 * R * Real.sin (angle / 2)
    let effective_mat_length := chord_length / (1 - overlap_ratio * 2)
    effective_mat_length

theorem placemat_length (R : ℝ) (num_mats : ℕ) (mat_width : ℝ) (overlap_ratio : ℝ): 
    R = 5 ∧ num_mats = 8 ∧ mat_width = 2 ∧ overlap_ratio = (1 / 4)
    → calculate_placemat_length R num_mats mat_width overlap_ratio = 7.654 :=
by
  sorry

end placemat_length_l120_120490


namespace floor_div_eq_floor_floor_div_l120_120569

theorem floor_div_eq_floor_floor_div (α : ℝ) (d : ℕ) (hα : 0 < α) :
  ⌊α / d⌋ = ⌊⌊α⌋ / d⌋ :=
by sorry

end floor_div_eq_floor_floor_div_l120_120569


namespace mark_ate_fruit_first_four_days_l120_120135

theorem mark_ate_fruit_first_four_days (total_fruit : ℕ) (kept_for_next_week : ℕ) (brought_to_school : ℕ) :
  total_fruit = 10 → kept_for_next_week = 2 → brought_to_school = 3 → 
  (total_fruit - kept_for_next_week - brought_to_school) = 5 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end mark_ate_fruit_first_four_days_l120_120135


namespace find_intended_number_l120_120727

theorem find_intended_number (x : ℕ) 
    (condition : 3 * x = (10 * 3 * x + 2) / 19 + 7) : 
    x = 5 :=
sorry

end find_intended_number_l120_120727


namespace perimeter_of_structure_l120_120150

noncomputable def structure_area : ℝ := 576
noncomputable def num_squares : ℕ := 9
noncomputable def square_area : ℝ := structure_area / num_squares
noncomputable def side_length : ℝ := Real.sqrt square_area
noncomputable def perimeter (side_length : ℝ) : ℝ := 8 * side_length

theorem perimeter_of_structure : perimeter side_length = 64 := by
  -- proof will follow here
  sorry

end perimeter_of_structure_l120_120150


namespace brian_fewer_seashells_l120_120658

-- Define the conditions
def cb_ratio (Craig Brian : ℕ) : Prop := 9 * Brian = 7 * Craig
def craig_seashells (Craig : ℕ) : Prop := Craig = 54

-- Define the main theorem to be proven
theorem brian_fewer_seashells (Craig Brian : ℕ) (h1 : cb_ratio Craig Brian) (h2 : craig_seashells Craig) : Craig - Brian = 12 :=
by
  sorry

end brian_fewer_seashells_l120_120658


namespace maciek_total_cost_l120_120058

-- Define the cost of pretzels without discount
def pretzel_price : ℝ := 4.0

-- Define the discounted price of pretzels when buying 3 or more packs
def pretzel_discount_price : ℝ := 3.5

-- Define the cost of chips without discount
def chips_price : ℝ := 7.0

-- Define the discounted price of chips when buying 2 or more packs
def chips_discount_price : ℝ := 6.0

-- Define the number of pretzels Maciek buys
def pretzels_bought : ℕ := 3

-- Define the number of chips Maciek buys
def chips_bought : ℕ := 4

-- Calculate the total cost of pretzels
def pretzel_cost : ℝ :=
  if pretzels_bought >= 3 then pretzels_bought * pretzel_discount_price else pretzels_bought * pretzel_price

-- Calculate the total cost of chips
def chips_cost : ℝ :=
  if chips_bought >= 2 then chips_bought * chips_discount_price else chips_bought * chips_price

-- Calculate the total amount Maciek needs to pay
def total_cost : ℝ :=
  pretzel_cost + chips_cost

theorem maciek_total_cost :
  total_cost = 34.5 :=
by 
  sorry

end maciek_total_cost_l120_120058


namespace value_of_p_l120_120026

noncomputable def p_value_condition (p q : ℝ) (h1 : p + q = 1) (h2 : p > 0) (h3 : q > 0) : Prop :=
  (9 * p^8 * q = 36 * p^7 * q^2)

theorem value_of_p (p q : ℝ) (h1 : p + q = 1) (h2 : p > 0) (h3 : q > 0) (h4 : p_value_condition p q h1 h2 h3) :
  p = 4 / 5 :=
by
  sorry

end value_of_p_l120_120026


namespace factorial_division_l120_120958

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l120_120958


namespace part1_part2_l120_120420

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem part1 (x : ℝ) : (∀ x, f x 2 ≤ x + 4 → (1 / 2 ≤ x ∧ x ≤ 7 / 2)) :=
by sorry

theorem part2 (x : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -5 ∨ a ≥ 3) :=
by sorry

end part1_part2_l120_120420


namespace find_b_l120_120375

theorem find_b (a b c : ℚ) (h : (3 * x^2 - 4 * x + 2) * (a * x^2 + b * x + c) = 9 * x^4 - 10 * x^3 + 5 * x^2 - 8 * x + 4)
  (ha : a = 3) : b = 2 / 3 :=
by
  sorry

end find_b_l120_120375


namespace crossed_out_number_is_29_l120_120142

theorem crossed_out_number_is_29 : 
  ∀ n : ℕ, (11 * n + 66 - (325 - (12 * n + 66 - 325))) = 29 :=
by sorry

end crossed_out_number_is_29_l120_120142


namespace simplify_polynomial_l120_120580

variable {R : Type*} [CommRing R]

theorem simplify_polynomial (x : R) :
  (12 * x ^ 10 + 9 * x ^ 9 + 5 * x ^ 8) + (2 * x ^ 12 + x ^ 10 + 2 * x ^ 9 + 3 * x ^ 8 + 4 * x ^ 4 + 6 * x ^ 2 + 9) =
  2 * x ^ 12 + 13 * x ^ 10 + 11 * x ^ 9 + 8 * x ^ 8 + 4 * x ^ 4 + 6 * x ^ 2 + 9 :=
  sorry

end simplify_polynomial_l120_120580


namespace third_generation_tail_length_is_25_l120_120280

def first_generation_tail_length : ℝ := 16
def growth_rate : ℝ := 0.25

def second_generation_tail_length : ℝ := first_generation_tail_length * (1 + growth_rate)
def third_generation_tail_length : ℝ := second_generation_tail_length * (1 + growth_rate)

theorem third_generation_tail_length_is_25 :
  third_generation_tail_length = 25 := by
  sorry

end third_generation_tail_length_is_25_l120_120280


namespace problem_l120_120107

noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x + 10

theorem problem (m : ℝ) (h1 : m > 1) (h2 : f m = 1) :
  m = 3 ∧ (∀ x ∈ (Set.Icc 3 5), f x ≤ 5) ∧ (∀ x ∈ (Set.Icc 3 5), f x ≥ 1) :=
by
  sorry

end problem_l120_120107


namespace min_sum_abc_l120_120159

theorem min_sum_abc (a b c : ℕ) (h1 : a * b * c = 3960) : a + b + c ≥ 150 :=
sorry

end min_sum_abc_l120_120159


namespace four_consecutive_none_multiple_of_5_l120_120028

theorem four_consecutive_none_multiple_of_5 (n : ℤ) :
  (∃ k : ℤ, n + (n + 1) + (n + 2) + (n + 3) = 5 * k) →
  ¬ (∃ m : ℤ, (n = 5 * m) ∨ (n + 1 = 5 * m) ∨ (n + 2 = 5 * m) ∨ (n + 3 = 5 * m)) :=
by sorry

end four_consecutive_none_multiple_of_5_l120_120028


namespace minimum_slit_length_l120_120323

theorem minimum_slit_length (circumference : ℝ) (speed_ratio : ℝ) (reliability : ℝ) :
  circumference = 1 → speed_ratio = 2 → (∀ (s : ℝ), (s < 2/3) → (¬ reliable)) → reliability =
    2 / 3 :=
by
  intros hcirc hspeed hrel
  have s := (2 : ℝ) / 3
  sorry

end minimum_slit_length_l120_120323


namespace theater_revenue_l120_120193

theorem theater_revenue 
  (seats : ℕ)
  (capacity_percentage : ℝ)
  (ticket_price : ℝ)
  (days : ℕ)
  (H1 : seats = 400)
  (H2 : capacity_percentage = 0.8)
  (H3 : ticket_price = 30)
  (H4 : days = 3)
  : (seats * capacity_percentage * ticket_price * days = 28800) :=
by
  sorry

end theater_revenue_l120_120193


namespace three_digit_integers_congruent_to_2_mod_4_l120_120713

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l120_120713


namespace correct_calculation_only_A_l120_120329

-- Definitions of the expressions
def exprA (a : ℝ) : Prop := 3 * a + 2 * a = 5 * a
def exprB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def exprC (a : ℝ) : Prop := 3 * a * 2 * a = 6 * a
def exprD (a : ℝ) : Prop := 3 * a / (2 * a) = (3 / 2) * a

-- The theorem stating that only exprA is correct
theorem correct_calculation_only_A (a : ℝ) :
  exprA a ∧ ¬exprB a ∧ ¬exprC a ∧ ¬exprD a :=
by
  sorry

end correct_calculation_only_A_l120_120329


namespace distance_from_origin_l120_120639

noncomputable def point_distance (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem distance_from_origin (x y : ℝ) (h₁ : abs y = 15) (h₂ : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) (h₃ : x > 2) :
  point_distance x y = Real.sqrt (334 + 4 * Real.sqrt 105) :=
by
  sorry

end distance_from_origin_l120_120639


namespace squareable_numbers_l120_120168

def is_squareable (n : ℕ) : Prop :=
  ∃ (perm : ℕ → ℕ), (∀ i, 1 ≤ perm i ∧ perm i ≤ n) ∧ (∀ i, ∃ k, perm i + i = k * k)

theorem squareable_numbers : is_squareable 9 ∧ is_squareable 15 ∧ ¬ is_squareable 7 ∧ ¬ is_squareable 11 :=
by sorry

end squareable_numbers_l120_120168


namespace common_difference_arithmetic_sequence_l120_120996

theorem common_difference_arithmetic_sequence 
    (a : ℕ → ℝ) 
    (S₅ : ℝ)
    (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h₁ : a 4 + a 6 = 6)
    (h₂ : S₅ = (a 1 + a 2 + a 3 + a 4 + a 5))
    (h_S₅_val : S₅ = 10) :
  ∃ d : ℝ, d = (a 5 - a 1) / 4 ∧ d = 1/2 := 
by
  sorry

end common_difference_arithmetic_sequence_l120_120996


namespace complement_of_M_in_U_l120_120002

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5}
noncomputable def M : Set ℕ := {0, 1}

theorem complement_of_M_in_U : (U \ M) = {2, 3, 4, 5} :=
by
  -- The proof is omitted here.
  sorry

end complement_of_M_in_U_l120_120002


namespace three_in_A_even_not_in_A_l120_120535

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

-- (1) Prove that 3 ∈ A
theorem three_in_A : 3 ∈ A :=
sorry

-- (2) Prove that ∀ k ∈ ℤ, 4k - 2 ∉ A
theorem even_not_in_A (k : ℤ) : (4 * k - 2) ∉ A :=
sorry

end three_in_A_even_not_in_A_l120_120535


namespace mark_eats_fruit_l120_120136

-- Question: How many pieces of fruit did Mark eat in the first four days of the week?
theorem mark_eats_fruit (total_fruit : ℕ) (kept_fruit : ℕ) (friday_fruit : ℕ) :
  total_fruit = 10 → kept_fruit = 2 → friday_fruit = 3 → (total_fruit - kept_fruit - friday_fruit) = 5 :=
by
  intros h_total h_kept h_friday
  rw [h_total, h_kept, h_friday]
  simp
  exact rfl

end mark_eats_fruit_l120_120136


namespace distance_between_Sneezy_and_Grumpy_is_8_l120_120775

variables (DS DV SP VP: ℕ) (SV: ℕ)

theorem distance_between_Sneezy_and_Grumpy_is_8
  (hDS : DS = 5)
  (hDV : DV = 4)
  (hSP : SP = 10)
  (hVP : VP = 17)
  (hSV_condition1 : SV + SP > VP)
  (hSV_condition2 : SV < DS + DV)
  (hSV_condition3 : 7 < SV) :
  SV = 8 := 
sorry

end distance_between_Sneezy_and_Grumpy_is_8_l120_120775


namespace subset_bound_l120_120000

open Finset

variables {α : Type*}

theorem subset_bound (n : ℕ) (S : Finset (Finset (Fin (4 * n)))) (hS : ∀ {s t : Finset (Fin (4 * n))}, s ∈ S → t ∈ S → s ≠ t → (s ∩ t).card ≤ n) (h_card : ∀ s ∈ S, s.card = 2 * n) :
  S.card ≤ 6 ^ ((n + 1) / 2) :=
sorry

end subset_bound_l120_120000


namespace revenue_difference_l120_120192

theorem revenue_difference {x z : ℕ} (hx : 10 ≤ x ∧ x ≤ 96) (hz : z = x + 3) :
  1000 * z + 10 * x - (1000 * x + 10 * z) = 2920 :=
by
  sorry

end revenue_difference_l120_120192


namespace analects_deductive_reasoning_l120_120585

theorem analects_deductive_reasoning :
  (∀ (P Q R S T U V : Prop), 
    (P → Q) → 
    (Q → R) → 
    (R → S) → 
    (S → T) → 
    (T → U) → 
    ((P → U) ↔ deductive_reasoning)) :=
sorry

end analects_deductive_reasoning_l120_120585


namespace train_speed_l120_120647

theorem train_speed (L V : ℝ) (h1 : L = V * 20) (h2 : L + 300.024 = V * 50) : V = 10.0008 :=
by
  sorry

end train_speed_l120_120647


namespace inequality_am_gm_l120_120431

theorem inequality_am_gm (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x^4 + y^2) + y / (x^2 + y^4)) ≤ (1 / (x * y)) :=
by
  sorry

end inequality_am_gm_l120_120431


namespace price_of_peaches_is_2_l120_120295

noncomputable def price_per_pound_peaches (total_spent: ℝ) (price_per_pound_other: ℝ) (total_weight_peaches: ℝ) (total_weight_apples: ℝ) (total_weight_blueberries: ℝ) : ℝ :=
  (total_spent - (total_weight_apples + total_weight_blueberries) * price_per_pound_other) / total_weight_peaches

theorem price_of_peaches_is_2 
  (total_spent: ℝ := 51)
  (price_per_pound_other: ℝ := 1)
  (num_peach_pies: ℕ := 5)
  (num_apple_pies: ℕ := 4)
  (num_blueberry_pies: ℕ := 3)
  (weight_per_pie: ℝ := 3):
  price_per_pound_peaches total_spent price_per_pound_other 
                          (num_peach_pies * weight_per_pie) 
                          (num_apple_pies * weight_per_pie) 
                          (num_blueberry_pies * weight_per_pie) = 2 := 
by
  sorry

end price_of_peaches_is_2_l120_120295


namespace least_possible_value_d_l120_120474

theorem least_possible_value_d 
  (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (hxy : x < y)
  (hyz : y < z)
  (hyx_gt_five : y - x > 5) : 
  z - x = 9 :=
sorry

end least_possible_value_d_l120_120474


namespace range_of_x_l120_120108

def f (x a : ℝ) : ℝ := x^3 + 3 * a * x - 1

def g (x a : ℝ) : ℝ := 3 * x^2 - a * x + 3 * a - 5

def condition (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1

theorem range_of_x (x a : ℝ) (h : condition a) : g x a < 0 → -2/3 < x ∧ x < 1 := 
sorry

end range_of_x_l120_120108


namespace exists_integers_x_l120_120130

theorem exists_integers_x (a1 a2 a3 : ℤ) (h : 0 < a1 ∧ a1 < a2 ∧ a2 < a3) :
  ∃ (x1 x2 x3 : ℤ), (|x1| + |x2| + |x3| > 0) ∧ (a1 * x1 + a2 * x2 + a3 * x3 = 0) ∧ (max (max (|x1|) (|x2|)) (|x3|) < (2 / Real.sqrt 3 * Real.sqrt a3) + 1) := 
sorry

end exists_integers_x_l120_120130


namespace all_visitors_can_buy_ticket_l120_120264

-- Define the coin types
inductive Coin
  | Three
  | Five

-- Define a function to calculate the total money from a list of coins
def totalMoney (coins : List Coin) : Int :=
  coins.foldr (fun c acc => acc + (match c with | Coin.Three => 3 | Coin.Five => 5)) 0

-- Define the initial state: each person has 22 tugriks in some combination of 3 and 5 tugrik coins
def initial_money := 22
def ticket_cost := 4

-- Each visitor and the cashier has 22 tugriks initially
axiom visitor_money_all_22 (n : Nat) : n ≤ 200 → totalMoney (List.replicate 2 Coin.Five ++ List.replicate 4 Coin.Three) = initial_money

-- We want to prove that all visitors can buy a ticket
theorem all_visitors_can_buy_ticket :
  ∀ n, n ≤ 200 → ∃ coins: List Coin, totalMoney coins = initial_money ∧ totalMoney coins ≥ ticket_cost := by
    sorry -- Proof goes here

end all_visitors_can_buy_ticket_l120_120264


namespace complex_vector_PQ_l120_120545

theorem complex_vector_PQ (P Q : ℂ) (hP : P = 3 + 1 * I) (hQ : Q = 2 + 3 * I) : 
  (Q - P) = -1 + 2 * I :=
by sorry

end complex_vector_PQ_l120_120545


namespace arithmetic_sequence_a12_l120_120270

theorem arithmetic_sequence_a12 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 7 + a 9 = 16) (h2 : a 4 = 1) 
  (h3 : ∀ n, a (n + 1) = a n + d) : a 12 = 15 := 
by {
  -- Proof steps would go here
  sorry
}

end arithmetic_sequence_a12_l120_120270


namespace part1_part2_l120_120421

variable (x : ℝ)

def A := {x : ℝ | 1 < x ∧ x < 3}
def B := {x : ℝ | x < -3 ∨ 2 < x}

theorem part1 : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

theorem part2 (a b : ℝ) : (∀ x, 2 < x ∧ x < 3 → x^2 + a * x + b < 0) → a = -5 ∧ b = 6 := by
  sorry

end part1_part2_l120_120421


namespace student_walks_fifth_to_first_l120_120729

theorem student_walks_fifth_to_first :
  let floors := 4
  let staircases := 2
  (staircases ^ floors) = 16 := by
  sorry

end student_walks_fifth_to_first_l120_120729


namespace bridge_length_increase_l120_120049

open Real

def elevation_change : ℝ := 800
def original_gradient : ℝ := 0.02
def new_gradient : ℝ := 0.015

theorem bridge_length_increase :
  let original_length := elevation_change / original_gradient
  let new_length := elevation_change / new_gradient
  new_length - original_length = 13333 := by
  sorry

end bridge_length_increase_l120_120049


namespace profit_percent_calc_l120_120863

theorem profit_percent_calc (SP CP : ℝ) (h : CP = 0.25 * SP) : (SP - CP) / CP * 100 = 300 :=
by
  sorry

end profit_percent_calc_l120_120863


namespace enthusiasts_min_max_l120_120358

-- Define the conditions
def total_students : ℕ := 100
def basketball_enthusiasts : ℕ := 63
def football_enthusiasts : ℕ := 75

-- Define the main proof problem
theorem enthusiasts_min_max :
  ∃ (common_enthusiasts : ℕ), 38 ≤ common_enthusiasts ∧ common_enthusiasts ≤ 63 :=
sorry

end enthusiasts_min_max_l120_120358


namespace trigonometric_identity_l120_120680

theorem trigonometric_identity (α : ℝ) 
  (h : Real.tan (π / 4 + α) = 1) : 
  (2 * Real.sin α + Real.cos α) / (3 * Real.cos α - Real.sin α) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l120_120680


namespace min_value_term_l120_120246

def seq (n : ℕ) : ℝ := 2 * n^2 - 10 * n + 3

theorem min_value_term :
  ∃ n : ℕ, (n = 2 ∨ n = 3) ∧ ∀ m : ℕ, m > 0 → seq n ≤ seq m :=
by
  sorry

end min_value_term_l120_120246


namespace value_of_six_inch_cube_l120_120344

-- Defining the conditions
def original_cube_weight : ℝ := 5 -- in pounds
def original_cube_value : ℝ := 600 -- in dollars
def original_cube_side : ℝ := 4 -- in inches

def new_cube_side : ℝ := 6 -- in inches

def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

-- Statement of the theorem
theorem value_of_six_inch_cube :
  cube_volume new_cube_side / cube_volume original_cube_side * original_cube_value = 2025 :=
by
  -- Here goes the proof
  sorry

end value_of_six_inch_cube_l120_120344


namespace trapezoid_distances_l120_120157

-- Define the problem parameters
variables (AB CD AD BC : ℝ)
-- Assume given conditions
axiom h1 : AD > BC
noncomputable def k := AD / BC

-- Formalizing the proof problem in Lean 4
theorem trapezoid_distances (M : Type) (BM AM CM DM : ℝ) :
  BM = AB * BC / (AD - BC) →
  AM = AB * AD / (AD - BC) →
  CM = CD * BC / (AD - BC) →
  DM = CD * AD / (AD - BC) →
  true :=
sorry

end trapezoid_distances_l120_120157


namespace find_max_marks_l120_120755

variable (marks_scored : ℕ) -- 212
variable (shortfall : ℕ) -- 22
variable (pass_percentage : ℝ) -- 0.30

theorem find_max_marks (h_marks : marks_scored = 212) 
                       (h_short : shortfall = 22) 
                       (h_pass : pass_percentage = 0.30) : 
  ∃ M : ℝ, M = 780 :=
by {
  sorry
}

end find_max_marks_l120_120755


namespace cylinder_not_occupied_volume_l120_120343

theorem cylinder_not_occupied_volume :
  let r := 10
  let h_cylinder := 30
  let h_full_cone := 10
  let volume_cylinder := π * r^2 * h_cylinder
  let volume_full_cone := (1 / 3) * π * r^2 * h_full_cone
  let volume_half_cone := (1 / 2) * volume_full_cone
  let volume_unoccupied := volume_cylinder - (volume_full_cone + volume_half_cone)
  volume_unoccupied = 2500 * π := 
by
  sorry

end cylinder_not_occupied_volume_l120_120343


namespace expected_value_variance_defective_items_l120_120410

variable (ξ : ℕ) [distribution ξ]

theorem expected_value_variance_defective_items :
  (distribution.binom 200 0.01) ξ →
  (ξ.mean = 2 ∧ ξ.variance = 1.98) := by
  sorry

end expected_value_variance_defective_items_l120_120410


namespace carla_total_time_l120_120654

def time_sharpening : ℝ := 15
def time_peeling : ℝ := 3 * time_sharpening
def time_chopping : ℝ := 0.5 * time_peeling
def time_breaks : ℝ := 2 * 5

def total_time : ℝ :=
  time_sharpening + time_peeling + time_chopping + time_breaks

theorem carla_total_time : total_time = 92.5 :=
by sorry

end carla_total_time_l120_120654


namespace laura_mowing_time_correct_l120_120887

noncomputable def laura_mowing_time : ℝ := 
  let combined_time := 1.71428571429
  let sammy_time := 3
  let combined_rate := 1 / combined_time
  let sammy_rate := 1 / sammy_time
  let laura_rate := combined_rate - sammy_rate
  1 / laura_rate

theorem laura_mowing_time_correct : laura_mowing_time = 4.2 := 
  by
    sorry

end laura_mowing_time_correct_l120_120887


namespace solve_quadratic_equation_l120_120433

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2 := by
sorry

end solve_quadratic_equation_l120_120433


namespace hyperbola_center_l120_120982

theorem hyperbola_center :
  ∀ (x y : ℝ), 
  (4 * x + 8)^2 / 36 - (3 * y - 6)^2 / 25 = 1 → (x, y) = (-2, 2) :=
by
  intros x y h
  sorry

end hyperbola_center_l120_120982


namespace part1_part2_l120_120106

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x - m|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * f x m - f (x + m) m

theorem part1 (h : ∀ x, g x m ≥ -1) : m = 1 :=
  sorry

theorem part2 {a b m : ℝ} (ha : |a| < m) (hb : |b| < m) (a_ne_zero : a ≠ 0) (hm: m = 1) : 
  f (a * b) m > |a| * f (b / a) m :=
  sorry

end part1_part2_l120_120106


namespace find_price_per_package_l120_120651

theorem find_price_per_package (P : ℝ) :
  (10 * P + 50 * (4/5 * P) = 1340) → (P = 26.80) := by
  intros h
  sorry

end find_price_per_package_l120_120651


namespace time_descend_hill_l120_120883

-- Definitions
def time_to_top : ℝ := 4
def avg_speed_whole_journey : ℝ := 3
def avg_speed_uphill : ℝ := 2.25

-- Theorem statement
theorem time_descend_hill (t : ℝ) 
  (h1 : time_to_top = 4) 
  (h2 : avg_speed_whole_journey = 3) 
  (h3 : avg_speed_uphill = 2.25) : 
  t = 2 := 
sorry

end time_descend_hill_l120_120883


namespace symmetric_circle_equation_l120_120376

theorem symmetric_circle_equation :
  ∀ (x y : ℝ),
    (x^2 + y^2 - 6 * x + 8 * y + 24 = 0) →
    (x - 3 * y - 5 = 0) →
    (∃ x₀ y₀ : ℝ, (x₀ - 1)^2 + (y₀ - 2)^2 = 1) :=
by
  sorry

end symmetric_circle_equation_l120_120376


namespace time_for_each_student_l120_120054

-- Define the conditions as variables
variables (num_students : ℕ) (period_length : ℕ) (num_periods : ℕ)
-- Assume the conditions from the problem
def conditions := num_students = 32 ∧ period_length = 40 ∧ num_periods = 4

-- Define the total time available
def total_time (num_periods period_length : ℕ) := num_periods * period_length

-- Define the time per student
def time_per_student (total_time num_students : ℕ) := total_time / num_students

-- State the theorem to be proven
theorem time_for_each_student : 
  conditions num_students period_length num_periods →
  time_per_student (total_time num_periods period_length) num_students = 5 := sorry

end time_for_each_student_l120_120054


namespace possible_n_values_l120_120526

theorem possible_n_values (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 → n = 1 ∨ n = 3 :=
by 
  sorry

end possible_n_values_l120_120526


namespace price_of_orange_is_60_l120_120204

theorem price_of_orange_is_60
  (x a o : ℕ)
  (h1 : 40 * a + x * o = 540)
  (h2 : a + o = 10)
  (h3 : 40 * a + x * (o - 5) = 240) :
  x = 60 :=
by
  sorry

end price_of_orange_is_60_l120_120204


namespace sum_of_roots_eq_3n_l120_120743

variable {n : ℝ} 

-- Define the conditions
def quadratic_eq (x : ℝ) (m : ℝ) (n : ℝ) : Prop :=
  x^2 - (m + n) * x + m * n = 0

theorem sum_of_roots_eq_3n (m : ℝ) (n : ℝ) 
  (hm : m = 2 * n)
  (hroot_m : quadratic_eq m m n)
  (hroot_n : quadratic_eq n m n) :
  m + n = 3 * n :=
by sorry

end sum_of_roots_eq_3n_l120_120743


namespace polynomial_remainder_l120_120210

theorem polynomial_remainder 
  (y: ℤ) 
  (root_cond: y^3 + y^2 + y + 1 = 0) 
  (beta_is_root: ∃ β: ℚ, β^3 + β^2 + β + 1 = 0) 
  (beta_four: ∀ β: ℚ, β^3 + β^2 + β + 1 = 0 → β^4 = 1) : 
  ∃ q r, (y^20 + y^15 + y^10 + y^5 + 1) = q * (y^3 + y^2 + y + 1) + r ∧ (r = 1) :=
by
  sorry

end polynomial_remainder_l120_120210


namespace ball_fall_time_l120_120787

theorem ball_fall_time (h g : ℝ) (t : ℝ) : 
  h = 20 → g = 10 → h + 20 * (t - 2) - 5 * ((t - 2) ^ 2) = t * (20 - 10 * (t - 2)) → 
  t = Real.sqrt 8 := 
by
  intros h_eq g_eq motion_eq
  sorry

end ball_fall_time_l120_120787


namespace complete_the_square_transforms_l120_120039

theorem complete_the_square_transforms (x : ℝ) :
  (x^2 + 8 * x + 7 = 0) → ((x + 4) ^ 2 = 9) :=
by
  intro h
  have step1 : x^2 + 8 * x = -7 := by sorry
  have step2 : x^2 + 8 * x + 16 = -7 + 16 := by sorry
  have step3 : (x + 4) ^ 2 = 9 := by sorry
  exact step3

end complete_the_square_transforms_l120_120039


namespace find_erased_number_l120_120471

/-- Define the variables used in the conditions -/
def n : ℕ := 69
def erased_number_mean : ℚ := 35 + 7 / 17
def sequence_sum : ℕ := n * (n + 1) / 2

/-- State the condition for the erased number -/
noncomputable def erased_number (x : ℕ) : Prop :=
  (sequence_sum - x) / (n - 1) = erased_number_mean

/-- The main theorem stating that the erased number is 7 -/
theorem find_erased_number : ∃ x : ℕ, erased_number x ∧ x = 7 :=
by
  use 7
  unfold erased_number sequence_sum
  -- Sum of first 69 natural numbers is 69 * (69 + 1) / 2
  -- Hence,
  -- (69 * 70 / 2 - 7) / 68 = 35 + 7 / 17
  -- which simplifies to true under these conditions
  -- Detailed proof skipped here as per instructions
  sorry

end find_erased_number_l120_120471


namespace proof_fraction_problem_l120_120979

def fraction_problem :=
  (1 / 5 + 1 / 3) / (3 / 4 - 1 / 8) = 64 / 75

theorem proof_fraction_problem : fraction_problem :=
by
  sorry

end proof_fraction_problem_l120_120979


namespace expected_value_boy_girl_adjacent_pairs_l120_120896

/-- Considering 10 boys and 15 girls lined up in a row, we need to show that
    the expected number of adjacent positions where a boy and a girl stand next to each other is 12. -/
theorem expected_value_boy_girl_adjacent_pairs :
  let boys := 10
  let girls := 15
  let total_people := boys + girls
  let total_adjacent_pairs := total_people - 1
  let p_boy_then_girl := (boys / total_people) * (girls / (total_people - 1))
  let p_girl_then_boy := (girls / total_people) * (boys / (total_people - 1))
  let expected_T := total_adjacent_pairs * (p_boy_then_girl + p_girl_then_boy)
  expected_T = 12 :=
by
  sorry

end expected_value_boy_girl_adjacent_pairs_l120_120896


namespace no_conditions_satisfy_l120_120071

-- Define the conditions
def condition1 (a b c : ℤ) : Prop := a = 1 ∧ b = 1 ∧ c = 1
def condition2 (a b c : ℤ) : Prop := a = b - 1 ∧ b = c - 1
def condition3 (a b c : ℤ) : Prop := a = b ∧ b = c
def condition4 (a b c : ℤ) : Prop := a > c ∧ c = b - 1 

-- Define the equations
def equation1 (a b c : ℤ) : ℤ := a * (a - b)^3 + b * (b - c)^3 + c * (c - a)^3
def equation2 (a b c : ℤ) : Prop := a + b + c = 3

-- Proof statement for the original problem
theorem no_conditions_satisfy (a b c : ℤ) :
  ¬ (condition1 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition2 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition3 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition4 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) :=
sorry

end no_conditions_satisfy_l120_120071


namespace donuts_purchased_l120_120285

/-- John goes to a bakery every day for a four-day workweek and chooses between a 
    60-cent croissant or a 90-cent donut. At the end of the week, he spent a whole 
    number of dollars. Prove that he must have purchased 2 donuts. -/
theorem donuts_purchased (d c : ℕ) (h1 : d + c = 4) (h2 : 90 * d + 60 * c % 100 = 0) : d = 2 :=
sorry

end donuts_purchased_l120_120285


namespace helen_choc_chip_yesterday_l120_120851

variable (total_cookies morning_cookies : ℕ)

theorem helen_choc_chip_yesterday :
  total_cookies = 1081 →
  morning_cookies = 554 →
  total_cookies - morning_cookies = 527 := by
  sorry

end helen_choc_chip_yesterday_l120_120851


namespace number_of_three_digit_integers_congruent_mod4_l120_120722

def integer_congruent_to_mod (a b n : ℕ) : Prop := ∃ k : ℤ, n = a * k + b

theorem number_of_three_digit_integers_congruent_mod4 :
  (finset.filter (λ n, integer_congruent_to_mod 4 2 (n : ℕ)) 
   (finset.Icc (100 : ℕ) (999 : ℕ))).card = 225 :=
by
  sorry

end number_of_three_digit_integers_congruent_mod4_l120_120722


namespace solutions_in_nat_solutions_in_non_neg_int_l120_120852

-- Definitions for Part A
def nat_sol_count (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem solutions_in_nat (x1 x2 x3 : ℕ) : 
  (x1 > 0) → (x2 > 0) → (x3 > 0) → (x1 + x2 + x3 = 1000) → 
  nat_sol_count 997 3 = Nat.choose 999 2 := sorry

-- Definitions for Part B
theorem solutions_in_non_neg_int (x1 x2 x3 : ℕ) : 
  (x1 + x2 + x3 = 1000) → 
  nat_sol_count 1000 3 = Nat.choose 1002 2 := sorry

end solutions_in_nat_solutions_in_non_neg_int_l120_120852


namespace min_cost_of_packaging_l120_120346

def packaging_problem : Prop :=
  ∃ (x y : ℕ), 35 * x + 24 * y = 106 ∧ 140 * x + 120 * y = 500

theorem min_cost_of_packaging : packaging_problem :=
sorry

end min_cost_of_packaging_l120_120346


namespace no_absolute_winner_probability_l120_120503

-- Define the probabilities of matches
def P_AB : ℝ := 0.6  -- Probability Alyosha wins against Borya
def P_BV : ℝ := 0.4  -- Probability Borya wins against Vasya

-- Define the event C that there is no absolute winner
def event_C (P_AV : ℝ) (P_VB : ℝ) : ℝ :=
  let scenario1 := P_AB * P_BV * P_AV in
  let scenario2 := P_AB * P_VB * (1 - P_AV) in
  scenario1 + scenario2

-- Main theorem to prove
theorem no_absolute_winner_probability : 
  event_C 1 0.6 = 0.24 :=
by
  rw [event_C]
  simp
  norm_num
  sorry

end no_absolute_winner_probability_l120_120503


namespace bread_per_day_baguettes_per_day_croissants_per_day_l120_120873

-- Define the conditions
def loaves_per_hour : ℕ := 10
def hours_per_day : ℕ := 6
def baguettes_per_2hours : ℕ := 30
def croissants_per_75minutes : ℕ := 20

-- Conversion factors
def minutes_per_hour : ℕ := 60
def minutes_per_block : ℕ := 75
def blocks_per_75minutes : ℕ := 360 / 75

-- Proof statements
theorem bread_per_day :
  loaves_per_hour * hours_per_day = 60 := by sorry

theorem baguettes_per_day :
  (hours_per_day / 2) * baguettes_per_2hours = 90 := by sorry

theorem croissants_per_day :
  (blocks_per_75minutes * croissants_per_75minutes) = 80 := by sorry

end bread_per_day_baguettes_per_day_croissants_per_day_l120_120873


namespace amy_biking_miles_l120_120202

theorem amy_biking_miles (x : ℕ) (h1 : ∀ y : ℕ, y = 2 * x - 3) (h2 : ∀ y : ℕ, x + y = 33) : x = 12 :=
by
  sorry

end amy_biking_miles_l120_120202


namespace factorial_div_l120_120960

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l120_120960


namespace part1_part2_l120_120750

open Set

variable {α : Type*} [PartialOrder α]

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem part1 : A ∩ B = {x | 2 < x ∧ x < 3} :=
by
  sorry

theorem part2 : (compl B) = {x | x ≤ 1 ∨ x ≥ 3} :=
by
  sorry

end part1_part2_l120_120750


namespace problem1_problem2_l120_120047

open Real -- Open the Real namespace for trigonometric functions

-- Part 1: Prove cos(5π + α) * tan(α - 7π) = 4/5 given π < α < 2π and cos α = 3/5
theorem problem1 (α : ℝ) (hα1 : π < α) (hα2 : α < 2 * π) (hcos : cos α = 3 / 5) : 
  cos (5 * π + α) * tan (α - 7 * π) = 4 / 5 := sorry

-- Part 2: Prove sin(π/3 + α) = √3/3 given cos (π/6 - α) = √3/3
theorem problem2 (α : ℝ) (hcos : cos (π / 6 - α) = sqrt 3 / 3) : 
  sin (π / 3 + α) = sqrt 3 / 3 := sorry

end problem1_problem2_l120_120047


namespace gazprom_rd_expense_l120_120518

theorem gazprom_rd_expense
  (R_and_D_t : ℝ) (ΔAPL_t_plus_1 : ℝ)
  (h1 : R_and_D_t = 3289.31)
  (h2 : ΔAPL_t_plus_1 = 1.55) :
  R_and_D_t / ΔAPL_t_plus_1 = 2122 := 
by
  sorry

end gazprom_rd_expense_l120_120518


namespace train_length_l120_120645

theorem train_length (L V : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 175 = V * 39) : 
  L = 150 := 
by 
  -- proof omitted 
  sorry

end train_length_l120_120645


namespace min_sum_abc_l120_120158

theorem min_sum_abc (a b c : ℕ) (h1 : a * b * c = 3960) : a + b + c ≥ 150 :=
sorry

end min_sum_abc_l120_120158


namespace factorial_div_sum_l120_120968

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l120_120968


namespace total_number_of_bills_received_l120_120781

open Nat

-- Definitions based on the conditions:
def total_withdrawal_amount : ℕ := 600
def bill_denomination : ℕ := 20

-- Mathematically equivalent proof problem
theorem total_number_of_bills_received : (total_withdrawal_amount / bill_denomination) = 30 := 
by
  sorry

end total_number_of_bills_received_l120_120781


namespace dividend_is_correct_l120_120177

def quotient : ℕ := 36
def divisor : ℕ := 85
def remainder : ℕ := 26

theorem dividend_is_correct : divisor * quotient + remainder = 3086 := by
  sorry

end dividend_is_correct_l120_120177


namespace find_real_triples_l120_120213

theorem find_real_triples :
  ∀ (a b c : ℝ), a^2 + a * b + c = 0 ∧ b^2 + b * c + a = 0 ∧ c^2 + c * a + b = 0
  ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = -1/2 ∧ b = -1/2 ∧ c = -1/2) :=
by
  sorry

end find_real_triples_l120_120213


namespace part_one_part_two_l120_120841

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |1 - 2 * x|

theorem part_one (x : ℝ) : f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
begin
  sorry
end

theorem part_two {a b : ℝ} (h_cond1 : 0 < b) (h_cond2 : b < 1/2) (h_cond3 : 1/2 < a)
  (h_eq : f a = 3 * f b) : ∃ m : ℤ, a^2 + b^2 > m ∧ m = 2 :=
begin
  sorry
end

end part_one_part_two_l120_120841


namespace theater_total_revenue_l120_120195

theorem theater_total_revenue :
  let seats := 400
  let capacity := 0.8
  let ticket_price := 30
  let days := 3
  seats * capacity * ticket_price * days = 28800 := by
  sorry

end theater_total_revenue_l120_120195


namespace drum_wife_leopard_cost_l120_120187

-- Definitions
variables (x y z : ℤ)

def system1 := 2 * x + 3 * y + z = 111
def system2 := 3 * x + 4 * y - 2 * z = -8
def even_condition := z % 2 = 0

theorem drum_wife_leopard_cost:
  system1 x y z ∧ system2 x y z ∧ even_condition z →
  x = 20 ∧ y = 9 ∧ z = 44 :=
by
  intro h
  -- Full proof can be provided here
  sorry

end drum_wife_leopard_cost_l120_120187


namespace log_constant_expression_l120_120455

theorem log_constant_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (hcond : x^2 + y^2 = 18 * x * y) :
  ∃ k : ℝ, (Real.log (x - y) / Real.log (Real.sqrt 2) - (1 / 2) * (Real.log x / Real.log (Real.sqrt 2) + Real.log y / Real.log (Real.sqrt 2))) = k :=
sorry

end log_constant_expression_l120_120455


namespace probability_no_absolute_winner_l120_120506

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end probability_no_absolute_winner_l120_120506


namespace both_sports_l120_120411

-- Definitions based on the given conditions
def total_members := 80
def badminton_players := 48
def tennis_players := 46
def neither_players := 7

-- The theorem to be proved
theorem both_sports : (badminton_players + tennis_players - (total_members - neither_players)) = 21 := by
  sorry

end both_sports_l120_120411


namespace domino_swap_correct_multiplication_l120_120129

theorem domino_swap_correct_multiplication :
  ∃ (a b c d e f : ℕ), 
    a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 3 ∧ e = 12 ∧ f = 3 ∧ 
    a * b = 6 ∧ c * d = 3 ∧ e * f = 36 ∧
    ∃ (x y : ℕ), x * y = 36 := sorry

end domino_swap_correct_multiplication_l120_120129


namespace train_stop_duration_l120_120176

theorem train_stop_duration (speed_without_stoppages speed_with_stoppages : ℕ) (h1 : speed_without_stoppages = 45) (h2 : speed_with_stoppages = 42) :
  ∃ t : ℕ, t = 4 :=
by
  sorry

end train_stop_duration_l120_120176


namespace puppy_weight_l120_120191

variable (a b c : ℝ)

theorem puppy_weight :
  (a + b + c = 30) →
  (a + c = 3 * b) →
  (a + b = c) →
  a = 7.5 := by
  intros h1 h2 h3
  sorry

end puppy_weight_l120_120191


namespace inequality_solution_sets_min_value_exists_l120_120534

-- Define the function f
def f (x : ℝ) (m : ℝ) : ℝ := m * x^2 - 2 * x - 3

-- Existence of roots at -1 and n
def roots_of_quadratic (m : ℝ) (n : ℝ) : Prop :=
  m * (-1)^2 - 2 * (-1) - 3 = 0 ∧ m * n^2 - 2 * n - 3 = 0 ∧ m > 0

-- Main problem statements
theorem inequality_solution_sets (a : ℝ) (m : ℝ) (n : ℝ)
  (h1 : roots_of_quadratic m n) (h2 : m = 1) (h3 : n = 3) (h4 : a > 0) :
  if 0 < a ∧ a ≤ 1 then 
    ∀ x : ℝ, x > 2 / a ∨ x < 2
  else if 1 < a ∧ a < 2 then
    ∀ x : ℝ, x > 2 ∨ x < 2 / a
  else 
    False :=
sorry

theorem min_value_exists (a : ℝ) (m : ℝ)
  (h1 : 0 < a ∧ a < 1) (h2 : m = 1) (h3 : f (a^2) m - 3*a^3 = -5) :
  a = (Real.sqrt 5 - 1) / 2 :=
sorry

end inequality_solution_sets_min_value_exists_l120_120534


namespace total_lobster_pounds_l120_120250

variable (lobster_other_harbor1 : ℕ)
variable (lobster_other_harbor2 : ℕ)
variable (lobster_hooper_bay : ℕ)

-- Conditions
axiom h_eq : lobster_hooper_bay = 2 * (lobster_other_harbor1 + lobster_other_harbor2)
axiom other_harbors_eq : lobster_other_harbor1 = 80 ∧ lobster_other_harbor2 = 80

-- Proof statement
theorem total_lobster_pounds : 
  lobster_other_harbor1 + lobster_other_harbor2 + lobster_hooper_bay = 480 :=
by
  sorry

end total_lobster_pounds_l120_120250


namespace faster_speed_l120_120638

theorem faster_speed (v : ℝ) :
  (∀ t : ℝ, (40 / 10 = t) ∧ (60 / v = t)) → v = 15 :=
by
  sorry

end faster_speed_l120_120638


namespace equivalent_sets_l120_120770

-- Definitions of the condition and expected result
def condition_set : Set ℕ := { x | x - 3 < 2 }
def expected_set : Set ℕ := {0, 1, 2, 3, 4}

-- Theorem statement
theorem equivalent_sets : condition_set = expected_set := 
by
  sorry

end equivalent_sets_l120_120770


namespace solve_for_x_l120_120432

theorem solve_for_x (x : ℝ) (h : (x^2 + 2*x + 3) / (x + 1) = x + 3) : x = 0 :=
by
  sorry

end solve_for_x_l120_120432


namespace isosceles_triangle_base_length_l120_120025

def is_isosceles (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_base_length
  (x y : ℝ)
  (h1 : 2 * x + 2 * y = 16)
  (h2 : 4^2 + y^2 = x^2)
  (h3 : is_isosceles x x (2 * y) ) :
  2 * y = 6 := 
by
  sorry

end isosceles_triangle_base_length_l120_120025


namespace find_polynomials_l120_120666

theorem find_polynomials (W : Polynomial ℤ) : 
  (∀ n : ℕ, 0 < n → (2^n - 1) % W.eval n = 0) → (W = Polynomial.C 1 ∨ W = Polynomial.C (-1)) :=
by sorry

end find_polynomials_l120_120666


namespace find_m_l120_120686

-- Define the vectors and the real number m
variables {Vec : Type*} [AddCommGroup Vec] [Module ℝ Vec]
variables (e1 e2 : Vec) (m : ℝ)

-- Define the collinearity condition and non-collinearity of the basis vectors.
def non_collinear (v1 v2 : Vec) : Prop := ¬(∃ (a : ℝ), v2 = a • v1)

def collinear (v1 v2 : Vec) : Prop := ∃ (a : ℝ), v2 = a • v1

-- Given conditions
axiom e1_e2_non_collinear : non_collinear e1 e2
axiom AB_eq : ∀ (m : ℝ), Vec
axiom CB_eq : Vec

theorem find_m (h : collinear (e1 + m • e2) (e1 - e2)) : m = -1 :=
sorry

end find_m_l120_120686


namespace proof_mn_proof_expr_l120_120685

variables (m n : ℚ)
-- Conditions
def condition1 : Prop := (m + n)^2 = 9
def condition2 : Prop := (m - n)^2 = 1

-- Expected results
def expected_mn : ℚ := 2
def expected_expr : ℚ := 3

-- The theorem to be proved
theorem proof_mn : condition1 m n → condition2 m n → m * n = expected_mn :=
by
  sorry

theorem proof_expr : condition1 m n → condition2 m n → m^2 + n^2 - m * n = expected_expr :=
by
  sorry

end proof_mn_proof_expr_l120_120685


namespace sum_x_y_z_w_l120_120584

-- Define the conditions in Lean
variables {x y z w : ℤ}
axiom h1 : x - y + z = 7
axiom h2 : y - z + w = 8
axiom h3 : z - w + x = 4
axiom h4 : w - x + y = 3

-- Prove the result
theorem sum_x_y_z_w : x + y + z + w = 22 := by
  sorry

end sum_x_y_z_w_l120_120584


namespace parabola_no_intersect_l120_120117

theorem parabola_no_intersect (m : ℝ) : 
  (¬ ∃ x : ℝ, -x^2 - 6*x + m = 0 ) ↔ m < -9 :=
by
  sorry

end parabola_no_intersect_l120_120117


namespace square_perimeter_eq_area_perimeter_16_l120_120308

theorem square_perimeter_eq_area_perimeter_16 (s : ℕ) (h : s^2 = 4 * s) : 4 * s = 16 := by
  sorry

end square_perimeter_eq_area_perimeter_16_l120_120308


namespace sum_gcd_lcm_eq_180195_l120_120173

def gcd_60_45045 := Nat.gcd 60 45045
def lcm_60_45045 := Nat.lcm 60 45045

theorem sum_gcd_lcm_eq_180195 : gcd_60_45045 + lcm_60_45045 = 180195 := by
  sorry

end sum_gcd_lcm_eq_180195_l120_120173


namespace largest_coins_l120_120613

theorem largest_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n = 146 :=
by
  sorry

end largest_coins_l120_120613


namespace max_sum_at_1008_l120_120231

noncomputable def sum_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem max_sum_at_1008 (a : ℕ → ℝ) : 
  sum_sequence a 2015 > 0 → 
  sum_sequence a 2016 < 0 → 
  ∃ n, n = 1008 ∧ ∀ m, sum_sequence a m ≤ sum_sequence a 1008 :=
by
  intros h1 h2
  sorry

end max_sum_at_1008_l120_120231


namespace range_of_2a_plus_b_l120_120529

theorem range_of_2a_plus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 2) :
  0 < 2 * a + b ∧ 2 * a + b < 10 :=
sorry

end range_of_2a_plus_b_l120_120529


namespace smallest_consecutive_even_sum_560_l120_120438

theorem smallest_consecutive_even_sum_560 (n : ℕ) (h : 7 * n + 42 = 560) : n = 74 :=
  by
    sorry

end smallest_consecutive_even_sum_560_l120_120438


namespace tan_neg_seven_pi_sixths_l120_120673

noncomputable def tan_neg_pi_seven_sixths : Real :=
  -Real.sqrt 3 / 3

theorem tan_neg_seven_pi_sixths : Real.tan (-7 * Real.pi / 6) = -Real.sqrt 3 / 3 := by
  sorry

end tan_neg_seven_pi_sixths_l120_120673


namespace maximum_correct_answers_l120_120730

theorem maximum_correct_answers (a b c : ℕ) (h1 : a + b + c = 60)
  (h2 : 5 * a - 2 * c = 150) : a ≤ 38 :=
by
  sorry

end maximum_correct_answers_l120_120730


namespace find_number_l120_120327

-- Define the number 40 and the percentage 90.
def num : ℝ := 40
def percent : ℝ := 0.9

-- Define the condition that 4/5 of x is smaller than 90% of 40 by 16
def condition (x : ℝ) : Prop := (4/5 : ℝ) * x = percent * num - 16

-- Proof statement in Lean 4
theorem find_number : ∃ x : ℝ, condition x ∧ x = 25 :=
by 
  use 25
  unfold condition
  norm_num
  sorry

end find_number_l120_120327


namespace reduced_price_per_dozen_is_3_l120_120935

variable (P : ℝ) -- original price of an apple
variable (R : ℝ) -- reduced price of an apple
variable (A : ℝ) -- number of apples originally bought for Rs. 40
variable (cost_per_dozen_reduced : ℝ) -- reduced price per dozen apples

-- Define the conditions
axiom reduction_condition : R = 0.60 * P
axiom apples_bought_condition : 40 = A * P
axiom more_apples_condition : 40 = (A + 64) * R

-- Define the proof problem
theorem reduced_price_per_dozen_is_3 : cost_per_dozen_reduced = 3 :=
by
  sorry

end reduced_price_per_dozen_is_3_l120_120935


namespace new_plan_cost_correct_l120_120007

-- Define the conditions
def old_plan_cost := 150
def increase_rate := 0.3

-- Define the increased amount
def increase_amount := increase_rate * old_plan_cost

-- Define the cost of the new plan
def new_plan_cost := old_plan_cost + increase_amount

-- Prove the main statement
theorem new_plan_cost_correct : new_plan_cost = 195 :=
by
  sorry

end new_plan_cost_correct_l120_120007


namespace total_area_of_pyramid_faces_l120_120606

-- Define the basic parameters of the pyramid
def base_edges := 8
def lateral_edges := 7

-- Define the Pythagorean theorem components
def altitude_squared := lateral_edges^2 - (base_edges / 2)^2
def altitude := real.sqrt altitude_squared

-- Define the area of one triangular face using half-base and altitude
def one_face_area := (1 / 2) * base_edges * altitude

-- Define the total area of the four triangular faces
def total_area_of_faces := 4 * one_face_area

-- Statement to prove
theorem total_area_of_pyramid_faces : total_area_of_faces = 16 * real.sqrt 33 :=
by
  -- Necessary calculations are assured correct by previous definitions and attributes
  sorry

end total_area_of_pyramid_faces_l120_120606


namespace find_a_l120_120725

theorem find_a (a : ℝ) (h : (Polynomial.monomial 3 (Nat.choose 5 3 * a^3) * Polynomial.C (80 : ℝ)) = 80) : a = 2 :=
by
  sorry

end find_a_l120_120725


namespace range_of_k_l120_120263

noncomputable def quadratic_inequality (k x : ℝ) : ℝ :=
  k * x^2 + 2 * k * x - (k + 2)

theorem range_of_k :
  (∀ x : ℝ, quadratic_inequality k x < 0) ↔ -1 < k ∧ k < 0 :=
by
  sorry

end range_of_k_l120_120263


namespace units_digit_sum_l120_120832

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum
  (h1 : units_digit 13 = 3)
  (h2 : units_digit 41 = 1)
  (h3 : units_digit 27 = 7)
  (h4 : units_digit 34 = 4) :
  units_digit ((13 * 41) + (27 * 34)) = 1 :=
by
  sorry

end units_digit_sum_l120_120832


namespace negation_of_existence_statement_l120_120381

theorem negation_of_existence_statement :
  (¬ (∃ x : ℝ, x^2 + x + 1 < 0)) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existence_statement_l120_120381


namespace volume_of_mixture_l120_120459

section
variable (Va Vb Vtotal : ℝ)

theorem volume_of_mixture :
  (Va / Vb = 3 / 2) →
  (800 * Va + 850 * Vb = 2460) →
  (Vtotal = Va + Vb) →
  Vtotal = 2.998 :=
by
  intros h1 h2 h3
  sorry
end

end volume_of_mixture_l120_120459


namespace actual_distance_traveled_l120_120544

-- Definitions based on conditions
def original_speed : ℕ := 12
def increased_speed : ℕ := 20
def distance_difference : ℕ := 24

-- We need to prove the actual distance traveled by the person.
theorem actual_distance_traveled : 
  ∃ t : ℕ, increased_speed * t = original_speed * t + distance_difference → original_speed * t = 36 :=
by
  sorry

end actual_distance_traveled_l120_120544


namespace parabola_coefficients_l120_120305

theorem parabola_coefficients (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ (y = (x + 2)^2 + 5) ∧ y = 9 ↔ x = 0) →
  (a, b, c) = (1, 4, 9) :=
by
  intros h
  sorry

end parabola_coefficients_l120_120305


namespace isosceles_trapezoid_perimeter_l120_120268

/-- In an isosceles trapezoid ABCD with bases AB = 10 units and CD = 18 units, 
and height from AB to CD is 4 units, the perimeter of ABCD is 28 + 8 * sqrt(2) units. -/
theorem isosceles_trapezoid_perimeter :
  ∃ (A B C D : Type) (AB CD AD BC h : ℝ), 
      AB = 10 ∧ 
      CD = 18 ∧ 
      AD = BC ∧ 
      h = 4 →
      ∀ (P : ℝ), P = AB + BC + CD + DA → 
      P = 28 + 8 * Real.sqrt 2 :=
by
  sorry

end isosceles_trapezoid_perimeter_l120_120268


namespace total_acorns_l120_120144

theorem total_acorns (s_a : ℕ) (s_b : ℕ) (d : ℕ)
  (h1 : s_a = 7)
  (h2 : s_b = 5 * s_a)
  (h3 : s_b + 3 = d) :
  s_a + s_b + d = 80 :=
by
  sorry

end total_acorns_l120_120144


namespace man_speed_in_still_water_l120_120330

theorem man_speed_in_still_water
  (vm vs : ℝ)
  (h1 : vm + vs = 6)  -- effective speed downstream
  (h2 : vm - vs = 4)  -- effective speed upstream
  : vm = 5 := 
by
  sorry

end man_speed_in_still_water_l120_120330


namespace complement_A_B_correct_l120_120630

open Set

-- Given sets A and B
def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

-- Define the complement of B with respect to A
def complement_A_B : Set ℕ := A \ B

-- Statement to prove
theorem complement_A_B_correct : complement_A_B = {0, 2, 6, 10} :=
  by sorry

end complement_A_B_correct_l120_120630


namespace number_of_three_digit_integers_congruent_mod4_l120_120721

def integer_congruent_to_mod (a b n : ℕ) : Prop := ∃ k : ℤ, n = a * k + b

theorem number_of_three_digit_integers_congruent_mod4 :
  (finset.filter (λ n, integer_congruent_to_mod 4 2 (n : ℕ)) 
   (finset.Icc (100 : ℕ) (999 : ℕ))).card = 225 :=
by
  sorry

end number_of_three_digit_integers_congruent_mod4_l120_120721


namespace trajectory_of_point_inside_square_is_conic_or_degenerates_l120_120335

noncomputable def is_conic_section (a : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ (m n l : ℝ) (x y : ℝ), 
    x = P.1 ∧ y = P.2 ∧ 
    (m^2 + n^2) * x^2 - 2 * n * (l + m) * x * y + (l^2 + n^2) * y^2 = (l * m - n^2)^2 ∧
    4 * n^2 * (l + m)^2 - 4 * (m^2 + n^2) * (l^2 + n^2) ≤ 0

theorem trajectory_of_point_inside_square_is_conic_or_degenerates
  (a : ℝ) (P : ℝ × ℝ)
  (h1 : 0 < P.1) (h2 : P.1 < 2 * a)
  (h3 : 0 < P.2) (h4 : P.2 < 2 * a)
  : is_conic_section a P :=
sorry

end trajectory_of_point_inside_square_is_conic_or_degenerates_l120_120335


namespace rectangle_area_perimeter_l120_120390

theorem rectangle_area_perimeter (a b : ℝ) (h₁ : a * b = 6) (h₂ : a + b = 6) : a^2 + b^2 = 24 := 
by
  sorry

end rectangle_area_perimeter_l120_120390


namespace smallest_solution_l120_120828

theorem smallest_solution (x : ℝ) (h₁ : x ≥ 0 → x^2 - 3*x - 2 = 0 → x = (3 + Real.sqrt 17) / 2)
                         (h₂ : x < 0 → x^2 + 3*x + 2 = 0 → (x = -1 ∨ x = -2)) :
  x = -2 :=
by
  sorry

end smallest_solution_l120_120828


namespace required_moles_of_H2O_l120_120215

-- Definition of the balanced chemical reaction
def balanced_reaction_na_to_naoh_and_H2 : Prop :=
  ∀ (NaH H2O NaOH H2 : ℕ), NaH + H2O = NaOH + H2

-- The given moles of NaH
def moles_NaH : ℕ := 2

-- Assertion that we need to prove: amount of H2O required is 2 moles
theorem required_moles_of_H2O (balanced : balanced_reaction_na_to_naoh_and_H2) : 
  (2 * 1) = 2 :=
by
  sorry

end required_moles_of_H2O_l120_120215


namespace average_15_19_x_eq_20_l120_120587

theorem average_15_19_x_eq_20 (x : ℝ) : (15 + 19 + x) / 3 = 20 → x = 26 :=
by
  sorry

end average_15_19_x_eq_20_l120_120587


namespace max_area_rectangle_l120_120061

-- Definition of the problem
def optimalRectangle (p : ℝ) : Prop :=
  let length := p / 2
  let width := p / 4
  length * width = (p^2) / 8

-- Statement of the theorem
theorem max_area_rectangle (p : ℝ) (h : 0 ≤ p) : 
  optimalRectangle p :=
  sorry

end max_area_rectangle_l120_120061


namespace value_of_a_l120_120524

noncomputable def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ :=
  a * b^3 + c

theorem value_of_a :
  F a 2 3 = F a 3 4 → a = -1 / 19 :=
by
  sorry

end value_of_a_l120_120524


namespace bush_height_l120_120051

theorem bush_height (h : ℕ → ℕ) (h0 : h 5 = 81) (h1 : ∀ n, h (n + 1) = 3 * h n) :
  h 2 = 3 := 
sorry

end bush_height_l120_120051


namespace part1_solution_set_l120_120840

def f (x : ℝ) : ℝ := |x + 1| + |1 - 2 * x|

theorem part1_solution_set : {x : ℝ | f x ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end part1_solution_set_l120_120840


namespace abs_a_eq_5_and_a_add_b_eq_0_l120_120401

theorem abs_a_eq_5_and_a_add_b_eq_0 (a b : ℤ) (h1 : |a| = 5) (h2 : a + b = 0) :
  a - b = 10 ∨ a - b = -10 :=
by
  sorry

end abs_a_eq_5_and_a_add_b_eq_0_l120_120401


namespace theater_total_revenue_l120_120196

theorem theater_total_revenue :
  let seats := 400
  let capacity := 0.8
  let ticket_price := 30
  let days := 3
  seats * capacity * ticket_price * days = 28800 := by
  sorry

end theater_total_revenue_l120_120196


namespace kernel_selects_white_probability_l120_120475

open Probability

def kernel_popping_problem 
  (P_white_kernels : ℝ)
  (P_yellow_kernels : ℝ)
  (P_white_pops : ℝ)
  (P_yellow_pops : ℝ)
  (P_popped : ℝ) : Prop :=
P_white_kernels = 2 / 3 ∧
P_yellow_kernels = 1 / 3 ∧
P_white_pops = 1 / 2 ∧
P_yellow_pops = 2 / 3 ∧
P_popped = (P_white_kernels * P_white_pops) + (P_yellow_kernels * P_yellow_pops) ∧
(P_white_kernels * P_white_pops) / P_popped = 3 / 5

theorem kernel_selects_white_probability :
  kernel_popping_problem (2 / 3) (1 / 3) (1 / 2) (2 / 3) ((2 / 3) * (1 / 2) + (1 / 3) * (2 / 3)) :=
begin
  sorry
end

end kernel_selects_white_probability_l120_120475


namespace pigeon_problem_l120_120790

theorem pigeon_problem (x y : ℕ) :
  (1 / 6 : ℝ) * (x + y) = y - 1 ∧ x - 1 = y + 1 → x = 4 ∧ y = 2 :=
by
  sorry

end pigeon_problem_l120_120790


namespace g_five_eq_one_l120_120023

noncomputable def g : ℝ → ℝ := sorry

theorem g_five_eq_one 
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0)
  (h3 : ∀ x : ℝ, g x = g (-x)) : 
  g 5 = 1 :=
sorry

end g_five_eq_one_l120_120023


namespace find_acute_angle_as_pi_over_4_l120_120241
open Real

-- Definitions from the problem's conditions
variables (x : ℝ)
def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2
def trig_eq (x : ℝ) : Prop := (sin x) ^ 3 + (cos x) ^ 3 = sqrt 2 / 2

-- The math proof problem statement
theorem find_acute_angle_as_pi_over_4 (h_acute : is_acute x) (h_trig_eq : trig_eq x) : x = π / 4 := 
sorry

end find_acute_angle_as_pi_over_4_l120_120241


namespace train_length_l120_120354

theorem train_length
  (speed_kmph : ℕ) (time_s : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_s = 12) :
  speed_kmph * (1000 / 3600 : ℕ) * time_s = 240 :=
by
  sorry

end train_length_l120_120354


namespace intersection_M_N_l120_120696

open Set Int

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l120_120696


namespace percentage_per_annum_is_correct_l120_120021

-- Define the conditions of the problem
def banker_gain : ℝ := 24
def present_worth : ℝ := 600
def time : ℕ := 2

-- Define the formula for the amount due
def amount_due (r : ℝ) (t : ℕ) (PW : ℝ) : ℝ := PW * (1 + r * t)

-- Define the given conditions translated from the problem
def given_conditions (r : ℝ) : Prop :=
  amount_due r time present_worth = present_worth + banker_gain

-- Lean statement of the problem to be proved
theorem percentage_per_annum_is_correct :
  ∃ r : ℝ, given_conditions r ∧ r = 0.02 :=
by {
  sorry
}

end percentage_per_annum_is_correct_l120_120021


namespace real_number_value_of_m_pure_imaginary_value_of_m_l120_120388

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem real_number_value_of_m (m : ℝ) : 
  is_real ((m^2 + 2 * m - 8) + (m^2 - 2 * m) * I) ↔ (m = 0 ∨ m = 2) := 
by sorry

theorem pure_imaginary_value_of_m (m : ℝ) : 
  is_pure_imaginary ((m^2 + 2 * m - 8) + (m^2 - 2 * m) * I) ↔ (m = -4) := 
by sorry

end real_number_value_of_m_pure_imaginary_value_of_m_l120_120388


namespace ones_digit_of_6_pow_52_l120_120466

theorem ones_digit_of_6_pow_52 : (6 ^ 52) % 10 = 6 := by
  -- we'll put the proof here
  sorry

end ones_digit_of_6_pow_52_l120_120466


namespace base_conversion_subtraction_l120_120360

def base8_to_base10 : Nat := 5 * 8^5 + 4 * 8^4 + 3 * 8^3 + 2 * 8^2 + 1 * 8^1 + 0 * 8^0
def base9_to_base10 : Nat := 6 * 9^4 + 5 * 9^3 + 4 * 9^2 + 3 * 9^1 + 2 * 9^0

theorem base_conversion_subtraction :
  base8_to_base10 - base9_to_base10 = 136532 :=
by
  -- Proof steps go here
  sorry

end base_conversion_subtraction_l120_120360


namespace weight_of_replaced_person_l120_120151

variable (average_weight_increase : ℝ)
variable (num_persons : ℝ)
variable (weight_new_person : ℝ)

theorem weight_of_replaced_person 
    (h1 : average_weight_increase = 2.5) 
    (h2 : num_persons = 10) 
    (h3 : weight_new_person = 90)
    : ∃ weight_replaced : ℝ, weight_replaced = 65 := 
by
  sorry

end weight_of_replaced_person_l120_120151


namespace fundraiser_successful_l120_120989

-- Defining the conditions
def num_students_bringing_brownies := 30
def brownies_per_student := 12
def num_students_bringing_cookies := 20
def cookies_per_student := 24
def num_students_bringing_donuts := 15
def donuts_per_student := 12
def price_per_treat := 2

-- Calculating the total number of each type of treat
def total_brownies := num_students_bringing_brownies * brownies_per_student
def total_cookies := num_students_bringing_cookies * cookies_per_student
def total_donuts := num_students_bringing_donuts * donuts_per_student

-- Calculating the total number of treats
def total_treats := total_brownies + total_cookies + total_donuts

-- Calculating the total money raised
def total_money_raised := total_treats * price_per_treat

theorem fundraiser_successful : total_money_raised = 2040 := by
    -- We introduce a sorry here because we are not providing the proof steps.
    sorry

end fundraiser_successful_l120_120989


namespace nba_conferences_division_l120_120125

theorem nba_conferences_division (teams : ℕ) (games_per_team : ℕ) (E : ℕ) :
  teams = 30 ∧ games_per_team = 82 ∧
  (teams = E + (teams - E)) ∧
  (games_per_team / 2 * E) + (games_per_team / 2 * (teams - E))  ≠ teams * games_per_team / 2 :=
by
  sorry

end nba_conferences_division_l120_120125


namespace minimum_ab_condition_l120_120098

open Int

theorem minimum_ab_condition 
  (a b : ℕ) 
  (h_pos : 0 < a ∧ 0 < b)
  (h_div7_ab_sum : ab * (a + b) % 7 ≠ 0) 
  (h_div7_expansion : ((a + b) ^ 7 - a ^ 7 - b ^ 7) % 7 = 0) : 
  ab = 18 :=
sorry

end minimum_ab_condition_l120_120098


namespace hyperbola_condition_l120_120162

noncomputable def hyperbola_eccentricity_difference (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let e_2pi_over_3 := Real.sqrt 3 + 1
  let e_pi_over_3 := (Real.sqrt 3) / 3 + 1
  e_2pi_over_3 - e_pi_over_3

theorem hyperbola_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  hyperbola_eccentricity_difference a b h1 h2 = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end hyperbola_condition_l120_120162


namespace lucy_first_round_cookies_l120_120003

theorem lucy_first_round_cookies (x : ℕ) : 
  (x + 27 = 61) → x = 34 :=
by
  intros h
  sorry

end lucy_first_round_cookies_l120_120003


namespace opposite_numbers_l120_120650

theorem opposite_numbers (a b : ℤ) (h1 : -5^2 = a) (h2 : (-5)^2 = b) : a = -b :=
by sorry

end opposite_numbers_l120_120650


namespace inverse_h_l120_120877

-- Definitions from the problem conditions
def f (x : ℝ) : ℝ := 4 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 5
def h (x : ℝ) : ℝ := f (g x)

-- Statement of the theorem for the inverse of h
theorem inverse_h : ∀ x : ℝ, h⁻¹ x = (x + 18) / 12 :=
sorry

end inverse_h_l120_120877


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l120_120707

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l120_120707


namespace john_unanswered_questions_l120_120738

theorem john_unanswered_questions (c w u : ℕ) 
  (h1 : 25 + 5 * c - 2 * w = 95) 
  (h2 : 6 * c - w + 3 * u = 105) 
  (h3 : c + w + u = 30) : 
  u = 2 := 
sorry

end john_unanswered_questions_l120_120738


namespace polynomial_division_quotient_l120_120986

noncomputable def P (x : ℝ) := 8 * x^3 + 5 * x^2 - 4 * x - 7
noncomputable def D (x : ℝ) := x + 3

theorem polynomial_division_quotient :
  ∀ x : ℝ, (P x) / (D x) = 8 * x^2 - 19 * x + 53 := sorry

end polynomial_division_quotient_l120_120986


namespace intersection_of_A_and_B_l120_120560

open Set

def A := {x : ℝ | 2 + x ≥ 4}
def B := {x : ℝ | -1 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 5} := sorry

end intersection_of_A_and_B_l120_120560


namespace locus_of_A_is_hyperbola_l120_120395

-- Define the coordinates of points B and C
def B : (ℝ × ℝ) := (-6, 0)
def C : (ℝ × ℝ) := (6, 0)

-- Define Angles and their conditions
variables (A B C : ℝ) -- assuming B, C are angle values not coordinates here

-- Given condition for sin
def sin_condition : Prop :=
  sin B - sin C = (1/2) * sin A

-- Define the distances |AB| and |AC|
def AB (A : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 + 6)^2 + (A.2)^2)

def AC (A : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - 6)^2 + (A.2)^2)

-- Define the condition for the hyperbola
def hyperbola_condition (A : ℝ × ℝ) : Prop :=
  (A.1 ^ 2) / 9 - (A.2 ^ 2) / 27 = 1

-- The statement we need to prove
theorem locus_of_A_is_hyperbola (A : ℝ × ℝ) (h : sin_condition A B C) : 
  hyperbola_condition A :=
sorry -- Proof omitted

end locus_of_A_is_hyperbola_l120_120395


namespace polynomial_divisibility_l120_120928

theorem polynomial_divisibility (n : ℕ) : (∀ x : ℤ, (x^2 + x + 1 ∣ x^(2*n) + x^n + 1)) ↔ (3 ∣ n) := by
  sorry

end polynomial_divisibility_l120_120928


namespace solution_set_for_inequality_l120_120847

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else x^2 - 2*x - 5

theorem solution_set_for_inequality :
  {x : ℝ | f x >= -2} = {x | -2 <= x ∧ x < 1 ∨ x >= 3} := sorry

end solution_set_for_inequality_l120_120847


namespace rd_expense_necessary_for_increase_l120_120520

theorem rd_expense_necessary_for_increase :
  ∀ (R_and_D_t : ℝ) (delta_APL_t1 : ℝ),
  R_and_D_t = 3289.31 → delta_APL_t1 = 1.55 →
  R_and_D_t / delta_APL_t1 = 2122 := 
by
  intros R_and_D_t delta_APL_t1 hR hD
  rw [hR, hD]
  norm_num
  sorry

end rd_expense_necessary_for_increase_l120_120520


namespace least_number_to_subtract_l120_120783

theorem least_number_to_subtract (n : ℕ) (h : n = 652543) : 
  ∃ x : ℕ, x = 7 ∧ (n - x) % 12 = 0 :=
by
  sorry

end least_number_to_subtract_l120_120783


namespace find_x_l120_120113

theorem find_x (x : ℝ) (h : (3 * x - 7) / 4 = 14) : x = 21 :=
sorry

end find_x_l120_120113


namespace smallest_obtuse_triangles_l120_120810

def obtuseTrianglesInTriangulation (n : Nat) : Nat :=
  if n < 3 then 0 else (n - 2) - 2

theorem smallest_obtuse_triangles (n : Nat) (h : n = 2003) :
  obtuseTrianglesInTriangulation n = 1999 := by
  sorry

end smallest_obtuse_triangles_l120_120810


namespace final_weight_is_correct_l120_120492

-- Define the initial weight of marble
def initial_weight := 300.0

-- Define the percentage reductions each week
def first_week_reduction := 0.3 * initial_weight
def second_week_reduction := 0.3 * (initial_weight - first_week_reduction)
def third_week_reduction := 0.15 * (initial_weight - first_week_reduction - second_week_reduction)

-- Calculate the final weight of the statue
def final_weight := initial_weight - first_week_reduction - second_week_reduction - third_week_reduction

-- The statement to prove
theorem final_weight_is_correct : final_weight = 124.95 := by
  -- Here would be the proof, which we are omitting
  sorry

end final_weight_is_correct_l120_120492


namespace mod_3_power_87_plus_5_l120_120914

theorem mod_3_power_87_plus_5 :
  (3 ^ 87 + 5) % 11 = 3 := 
by
  sorry

end mod_3_power_87_plus_5_l120_120914


namespace arithmetic_seq_num_terms_l120_120842

theorem arithmetic_seq_num_terms (a1 : ℕ := 1) (S_odd S_even : ℕ) (n : ℕ) 
  (h1 : S_odd = 341) (h2 : S_even = 682) : 2 * n = 10 :=
by
  sorry

end arithmetic_seq_num_terms_l120_120842


namespace positive_integers_count_l120_120633

theorem positive_integers_count (n : ℕ) : 
  ∃ m : ℕ, (m ≤ n / 2014 ∧ m ≤ n / 2016 ∧ (m + 1) * 2014 > n ∧ (m + 1) * 2016 > n) ↔
  (n = 1015056) :=
by
  sorry

end positive_integers_count_l120_120633


namespace simplify_expression_l120_120579

noncomputable def i : ℂ := Complex.I

theorem simplify_expression : 7*(4 - 2*i) + 4*i*(3 - 2*i) = 36 - 2*i :=
by
  sorry

end simplify_expression_l120_120579


namespace inscribed_triangle_area_l120_120494

noncomputable def triangle_inscribed_area (r : ℝ) (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem inscribed_triangle_area :
  ∀ (r : ℝ), r = 12 / Real.pi →
  let a := 2 * r * Real.sin (60 * Real.pi / 180) in
  let b := 2 * r * Real.sin (75 * Real.pi / 180) in
  let c := 2 * r * Real.sin (45 * Real.pi / 180) in
  b = c →
  triangle_inscribed_area r a b c = sorry :=
by
  intro r h rfl B_eq_C
  let a := 2 * r * Real.sin (60 * Real.pi / 180)
  let b := 2 * r * Real.sin (75 * Real.pi / 180)
  let c := 2 * r * Real.sin (45 * Real.pi / 180)
  exact sorry

end inscribed_triangle_area_l120_120494


namespace find_A_l120_120146

theorem find_A (A M C : Nat) (h1 : (10 * A^2 + 10 * M + C) * (A + M^2 + C^2) = 1050) (h2 : A < 10) (h3 : M < 10) (h4 : C < 10) : A = 2 := by
  sorry

end find_A_l120_120146


namespace expand_expression_l120_120978

theorem expand_expression (x y : ℝ) : 5 * (4 * x^3 - 3 * x * y + 7) = 20 * x^3 - 15 * x * y + 35 := 
sorry

end expand_expression_l120_120978


namespace part_a_l120_120926

theorem part_a (n : ℕ) : ((x^2 + x + 1) ∣ (x^(2 * n) + x^n + 1)) ↔ (n % 3 = 0) := sorry

end part_a_l120_120926


namespace beth_total_packs_l120_120946

def initial_packs := 4
def number_of_people := 10
def packs_per_person := initial_packs / number_of_people
def packs_found_later := 6

theorem beth_total_packs : packs_per_person + packs_found_later = 6.4 := by
  sorry

end beth_total_packs_l120_120946


namespace problem_1_problem_2_l120_120879

namespace ProofProblems

def U : Set ℝ := {y | true}

def E : Set ℝ := {y | y > 2}

def F : Set ℝ := {y | ∃ (x : ℝ), (-1 < x ∧ x < 2 ∧ y = x^2 - 2*x)}

def complement (A : Set ℝ) : Set ℝ := {y | y ∉ A}

theorem problem_1 : 
  (complement E ∩ F) = {y | -1 ≤ y ∧ y ≤ 2} := 
  sorry

def G (a : ℝ) : Set ℝ := {y | ∃ (x : ℝ), (0 < x ∧ x < a ∧ y = Real.log x / Real.log 2)}

theorem problem_2 (a : ℝ) :
  (∀ y, (y ∈ G a → y < 3)) → a ≥ 8 :=
  sorry

end ProofProblems

end problem_1_problem_2_l120_120879


namespace distance_to_SFL_is_81_l120_120357

variable (Speed : ℝ)
variable (Time : ℝ)

def distance_to_SFL (Speed : ℝ) (Time : ℝ) := Speed * Time

theorem distance_to_SFL_is_81 : distance_to_SFL 27 3 = 81 :=
by
  sorry

end distance_to_SFL_is_81_l120_120357


namespace common_tangents_l120_120902

noncomputable def radius1 := 8
noncomputable def radius2 := 6
noncomputable def distance := 2

theorem common_tangents (r1 r2 d : ℕ) 
  (h1 : r1 = radius1) 
  (h2 : r2 = radius2) 
  (h3 : d = distance) :
  (d = r1 - r2) → 1 = 1 := by 
  sorry

end common_tangents_l120_120902


namespace new_plan_cost_correct_l120_120005

def oldPlanCost : ℝ := 150
def rateIncrease : ℝ := 0.3
def newPlanCost : ℝ := oldPlanCost * (1 + rateIncrease) 

theorem new_plan_cost_correct : newPlanCost = 195 := by
  sorry

end new_plan_cost_correct_l120_120005


namespace triangle_area_PQR_l120_120034

def point := (ℝ × ℝ)

def P : point := (2, 3)
def Q : point := (7, 3)
def R : point := (4, 10)

noncomputable def triangle_area (A B C : point) : ℝ :=
  (1/2) * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_PQR : triangle_area P Q R = 17.5 :=
  sorry

end triangle_area_PQR_l120_120034


namespace marites_saves_120_per_year_l120_120423

def current_internet_speed := 10 -- Mbps
def current_monthly_bill := 20 -- dollars

def monthly_cost_20mbps := current_monthly_bill + 10 -- dollars
def monthly_cost_30mbps := current_monthly_bill * 2 -- dollars

def bundled_cost_20mbps := 80 -- dollars per month
def bundled_cost_30mbps := 90 -- dollars per month

def annual_cost_20mbps := bundled_cost_20mbps * 12 -- dollars per year
def annual_cost_30mbps := bundled_cost_30mbps * 12 -- dollars per year

theorem marites_saves_120_per_year :
  annual_cost_30mbps - annual_cost_20mbps = 120 := 
by
  sorry

end marites_saves_120_per_year_l120_120423


namespace intersection_point_parallel_line_through_intersection_l120_120537

-- Definitions for the problem
def l1 (x y : ℝ) : Prop := x + 8 * y + 7 = 0
def l2 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l3 (x y : ℝ) : Prop := x + y + 1 = 0
def parallel (x y c : ℝ) : Prop := x + y + c = 0
def point (x y : ℝ) : Prop := x = 1 ∧ y = -1

-- (1) Proof that the intersection point of l1 and l2 is (1, -1)
theorem intersection_point : ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧ point x y :=
by 
  sorry

-- (2) Proof that the line passing through the intersection point of l1 and l2
-- which is parallel to l3 is x + y = 0
theorem parallel_line_through_intersection : ∃ (c : ℝ), parallel 1 (-1) c ∧ c = 0 :=
by 
  sorry

end intersection_point_parallel_line_through_intersection_l120_120537


namespace factorial_computation_l120_120964

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l120_120964


namespace trigonometric_identity_l120_120533

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 1) : 
  1 - 2 * Real.sin α * Real.cos α - 3 * (Real.cos α)^2 = -3 / 2 :=
sorry

end trigonometric_identity_l120_120533


namespace aaron_brothers_l120_120199

theorem aaron_brothers (A : ℕ) (h1 : 6 = 2 * A - 2) : A = 4 :=
by
  sorry

end aaron_brothers_l120_120199


namespace find_f_f_2sqrt2_l120_120243

def f : ℝ → ℝ :=
λ x, if x < 1 then 3 * x + 5 else log (1/2) x - 1

theorem find_f_f_2sqrt2 : f (f (2 * real.sqrt 2)) = -5 / 2 := by
  sorry

end find_f_f_2sqrt2_l120_120243


namespace max_value_of_f_on_interval_l120_120848

noncomputable def f (x : ℝ) : ℝ := (Real.sin (4 * x)) / (2 * Real.sin ((Real.pi / 2) - 2 * x))

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 6), f x = (Real.sqrt 3) / 2 := sorry

end max_value_of_f_on_interval_l120_120848


namespace combined_area_of_tracts_l120_120701

theorem combined_area_of_tracts :
  let length1 := 300
  let width1 := 500
  let length2 := 250
  let width2 := 630
  let area1 := length1 * width1
  let area2 := length2 * width2
  let combined_area := area1 + area2
  combined_area = 307500 :=
by
  sorry

end combined_area_of_tracts_l120_120701


namespace number_of_ways_to_adjust_items_l120_120126

theorem number_of_ways_to_adjust_items :
  let items_on_upper_shelf := 4
  let items_on_lower_shelf := 8
  let move_items := 2
  let total_ways := Nat.choose items_on_lower_shelf move_items
  total_ways = 840 :=
by
  sorry

end number_of_ways_to_adjust_items_l120_120126


namespace set_intersection_complement_eq_l120_120850

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 6}
def B : Set ℕ := {2, 4, 5, 6}

noncomputable def complement (U B : Set ℕ) : Set ℕ := { x ∈ U | x ∉ B }

theorem set_intersection_complement_eq : (A ∩ (complement U B)) = {1, 3} := 
by 
  sorry

end set_intersection_complement_eq_l120_120850


namespace cube_painting_l120_120635

-- Let's start with importing Mathlib for natural number operations

theorem cube_painting (n : ℕ) (h : 2 < n)
  (num_one_black_face : ℕ := 3 * (n - 2)^2)
  (num_unpainted : ℕ := (n - 2)^3) :
  num_one_black_face = num_unpainted → n = 5 :=
by
  sorry

end cube_painting_l120_120635


namespace two_digit_number_exists_l120_120943

theorem two_digit_number_exists (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) :
  (9 * x + 8) * (80 - 9 * x) = 1855 → (9 * x + 8 = 35 ∨ 9 * x + 8 = 53) := by
  sorry

end two_digit_number_exists_l120_120943


namespace determine_solution_set_inequality_l120_120697

-- Definitions based on given conditions
def quadratic_inequality_solution (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c > 0
def new_quadratic_inequality_solution (c b a : ℝ) (x : ℝ) := c * x^2 + b * x + a < 0

-- The proof statement
theorem determine_solution_set_inequality (a b c : ℝ):
  (∀ x : ℝ, -1/3 < x ∧ x < 2 → quadratic_inequality_solution a b c x) →
  (∀ x : ℝ, -3 < x ∧ x < 1/2 ↔ new_quadratic_inequality_solution c b a x) := sorry

end determine_solution_set_inequality_l120_120697


namespace reciprocal_of_5_over_7_l120_120768

theorem reciprocal_of_5_over_7 : (5 / 7 : ℚ) * (7 / 5) = 1 := by
  sorry

end reciprocal_of_5_over_7_l120_120768


namespace total_population_l120_120141

theorem total_population (n : ℕ) (avg_population : ℕ) (h1 : n = 20) (h2 : avg_population = 4750) :
  n * avg_population = 95000 := by
  subst_vars
  sorry

end total_population_l120_120141


namespace fish_cost_l120_120114

theorem fish_cost (F P : ℝ) (h1 : 4 * F + 2 * P = 530) (h2 : 7 * F + 3 * P = 875) : F = 80 := 
by
  sorry

end fish_cost_l120_120114


namespace pirate_overtakes_at_8pm_l120_120059

noncomputable def pirate_overtake_trade : Prop :=
  let initial_distance := 15
  let pirate_speed_before_damage := 14
  let trade_speed := 10
  let time_before_damage := 3
  let pirate_distance_before_damage := pirate_speed_before_damage * time_before_damage
  let trade_distance_before_damage := trade_speed * time_before_damage
  let remaining_distance := initial_distance + trade_distance_before_damage - pirate_distance_before_damage
  let pirate_speed_after_damage := (18 / 17) * 10
  let relative_speed_after_damage := pirate_speed_after_damage - trade_speed
  let time_to_overtake_after_damage := remaining_distance / relative_speed_after_damage
  let total_time := time_before_damage + time_to_overtake_after_damage
  total_time = 8

theorem pirate_overtakes_at_8pm : pirate_overtake_trade :=
by
  sorry

end pirate_overtakes_at_8pm_l120_120059


namespace red_tulips_l120_120319

theorem red_tulips (white_tulips : ℕ) (bouquets : ℕ)
  (hw : white_tulips = 21)
  (hb : bouquets = 7)
  (div_prop : ∀ n, white_tulips % n = 0 ↔ bouquets % n = 0) : 
  ∃ red_tulips : ℕ, red_tulips = 7 :=
by
  sorry

end red_tulips_l120_120319


namespace lim_n_to_infinity_fraction_l120_120808

noncomputable def limit_expression : ℝ := 
  real.limit (λ n: ℕ, (n: ℝ + 1) / (3 * n - 1)) sorry

theorem lim_n_to_infinity_fraction :
  limit_expression = 1 / 3 :=
sorry

end lim_n_to_infinity_fraction_l120_120808


namespace miner_distance_when_explosion_heard_l120_120796

-- Distance function for the miner (in feet)
def miner_distance (t : ℕ) : ℕ := 30 * t

-- Distance function for the sound after the explosion (in feet)
def sound_distance (t : ℕ) : ℕ := 1100 * (t - 45)

theorem miner_distance_when_explosion_heard :
  ∃ t : ℕ, miner_distance t / 3 = 463 ∧ miner_distance t = sound_distance t :=
sorry

end miner_distance_when_explosion_heard_l120_120796


namespace employed_males_percent_l120_120414

def percent_employed_population : ℝ := 96
def percent_females_among_employed : ℝ := 75

theorem employed_males_percent :
  percent_employed_population * (1 - percent_females_among_employed / 100) = 24 := by
    sorry

end employed_males_percent_l120_120414


namespace cost_of_items_l120_120186

theorem cost_of_items (x y z : ℕ) 
  (h1 : 2 * x + 3 * y + z = 111) 
  (h2 : 3 * x + 4 * y - 2 * z = -8) 
  (h3 : z % 2 = 0) : 
  (x = 20 ∧ y = 9 ∧ z = 44) :=
sorry

end cost_of_items_l120_120186


namespace value_of_expression_l120_120115

-- Defining the given conditions as Lean definitions
def x : ℚ := 2 / 3
def y : ℚ := 5 / 2

-- The theorem statement to prove that the given expression equals the correct answer
theorem value_of_expression : (1 / 3) * x^7 * y^6 = 125 / 261 :=
by
  sorry

end value_of_expression_l120_120115


namespace largest_gold_coins_l120_120621

noncomputable def max_gold_coins (n : ℕ) : ℕ :=
  if h : ∃ k : ℕ, n = 13 * k + 3 ∧ n < 150 then
    n
  else 0

theorem largest_gold_coins : max_gold_coins 146 = 146 :=
by
  sorry

end largest_gold_coins_l120_120621


namespace determine_avery_height_l120_120428

-- Define Meghan's height
def meghan_height : ℕ := 188

-- Define range of players' heights
def height_range : ℕ := 33

-- Define the predicate to determine Avery's height
def avery_height : ℕ := meghan_height - height_range

-- The theorem we need to prove
theorem determine_avery_height : avery_height = 155 := by
  sorry

end determine_avery_height_l120_120428


namespace replace_digits_and_check_divisibility_l120_120575

theorem replace_digits_and_check_divisibility (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
    (30 * 10^5 + a * 10^4 + b * 10^2 + 3 ≠ 0 ∧ 
     (30 * 10^5 + a * 10^4 + b * 10^2 + 3) % 13 = 0) ↔ 
    (30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3000803 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3020303 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3030703 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3050203 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3060603 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3080103 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3090503) := sorry

end replace_digits_and_check_divisibility_l120_120575


namespace alex_jellybeans_l120_120355

theorem alex_jellybeans (n : ℕ) (h1 : n ≥ 200) (h2 : n % 17 = 15) : n = 202 :=
sorry

end alex_jellybeans_l120_120355


namespace largest_number_of_gold_coins_l120_120618

theorem largest_number_of_gold_coins 
  (num_friends : ℕ)
  (extra_coins : ℕ)
  (total_coins : ℕ) :
  num_friends = 13 →
  extra_coins = 3 →
  total_coins < 150 →
  ∀ k : ℕ, total_coins = num_friends * k + extra_coins →
  total_coins ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l120_120618


namespace find_m_value_l120_120687

theorem find_m_value (m : ℤ) : (∃ a : ℤ, x^2 + 2 * (m + 1) * x + 25 = (x + a)^2) ↔ (m = 4 ∨ m = -6) := 
sorry

end find_m_value_l120_120687


namespace abs_e_pi_minus_six_l120_120948

noncomputable def e : ℝ := 2.718
noncomputable def pi : ℝ := 3.14159

theorem abs_e_pi_minus_six : |e + pi - 6| = 0.14041 := by
  sorry

end abs_e_pi_minus_six_l120_120948


namespace sum_of_inverses_inequality_l120_120095

theorem sum_of_inverses_inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum_eq : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 :=
sorry

end sum_of_inverses_inequality_l120_120095


namespace sequence_is_arithmetic_max_value_a_n_b_n_l120_120287

open Real

theorem sequence_is_arithmetic (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (Sn : ℕ → ℝ) 
  (h_Sn : ∀ n, Sn n = (a n ^ 2 + a n) / 2) :
    ∀ n, a n = n := sorry 

theorem max_value_a_n_b_n (a b : ℕ → ℝ)
  (h_b : ∀ n, b n = - n + 5)
  (h_a : ∀ n, a n = n) :
    ∀ n, n ≥ 2 → n ≤ 3 → 
    ∃ k, a k * b k = 25 / 4 := by 
      sorry

end sequence_is_arithmetic_max_value_a_n_b_n_l120_120287


namespace parallel_lines_condition_l120_120788

theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, ax + y + 1 = 0 ↔ x + ay - 1 = 0) ↔ (a = 1) :=
sorry

end parallel_lines_condition_l120_120788


namespace adult_tickets_sold_l120_120941

theorem adult_tickets_sold (A S : ℕ) (h1 : S = 3 * A) (h2 : A + S = 600) : A = 150 :=
by
  sorry

end adult_tickets_sold_l120_120941


namespace option_B_correct_l120_120175

-- Define the commutativity of multiplication
def commutativity_of_mul (a b : Nat) : Prop :=
  a * b = b * a

-- State the problem, which is to prove that 2ab + 3ba = 5ab given commutativity
theorem option_B_correct (a b : Nat) : commutativity_of_mul a b → 2 * (a * b) + 3 * (b * a) = 5 * (a * b) :=
by
  intro h_comm
  rw [←h_comm]
  sorry

end option_B_correct_l120_120175


namespace graveling_cost_l120_120924

theorem graveling_cost
  (length_lawn : ℝ) (width_lawn : ℝ)
  (width_road : ℝ)
  (cost_per_sq_m : ℝ)
  (h1: length_lawn = 80) (h2: width_lawn = 40) (h3: width_road = 10) (h4: cost_per_sq_m = 3) :
  (length_lawn * width_road + width_lawn * width_road - width_road * width_road) * cost_per_sq_m = 3900 := 
by
  sorry

end graveling_cost_l120_120924


namespace value_of_a8_l120_120683

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → 2 * a n + a (n + 1) = 0

theorem value_of_a8 (a : ℕ → ℝ) (h1 : seq a) (h2 : a 3 = -2) : a 8 = 64 :=
sorry

end value_of_a8_l120_120683


namespace total_tiles_covering_floor_l120_120350

-- Let n be the width of the rectangle (in tiles)
-- The length would then be 2n (in tiles)
-- The total number of tiles that lie on both diagonals is given as 39

theorem total_tiles_covering_floor (n : ℕ) (H : 2 * n + 1 = 39) : 2 * n^2 = 722 :=
by sorry

end total_tiles_covering_floor_l120_120350


namespace comb_sum_C8_2_C8_3_l120_120653

open Nat

theorem comb_sum_C8_2_C8_3 : (Nat.choose 8 2) + (Nat.choose 8 3) = 84 :=
by
  sorry

end comb_sum_C8_2_C8_3_l120_120653


namespace evaluate_64_pow_3_div_2_l120_120976

theorem evaluate_64_pow_3_div_2 : (64 : ℝ)^(3/2) = 512 := by
  -- given 64 = 2^6
  have h : (64 : ℝ) = 2^6 := by norm_num
  -- use this substitution and properties of exponents
  rw [h, ←pow_mul]
  norm_num
  sorry -- completing the proof, not needed based on the guidelines

end evaluate_64_pow_3_div_2_l120_120976


namespace at_least_one_le_one_l120_120745

theorem at_least_one_le_one (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end at_least_one_le_one_l120_120745


namespace part1_part2_part3_l120_120559

noncomputable def A : Set ℝ := { x | x ≥ 1 ∨ x ≤ -3 }
noncomputable def B : Set ℝ := { x | -4 < x ∧ x < 0 }
noncomputable def C : Set ℝ := { x | x ≤ -4 ∨ x ≥ 0 }

theorem part1 : A ∩ B = { x | -4 < x ∧ x ≤ -3 } := 
by { sorry }

theorem part2 : A ∪ B = { x | x < 0 ∨ x ≥ 1 } := 
by { sorry }

theorem part3 : A ∪ C = { x | x ≤ -3 ∨ x ≥ 0 } := 
by { sorry }

end part1_part2_part3_l120_120559


namespace petya_equals_vasya_l120_120568

def petya_word_count (m : ℕ) : ℕ :=
  sorry -- The actual count of m-letter words with equal T's and O's using letters T, O, W, and N.

def vasya_word_count (m : ℕ) : ℕ :=
  sorry -- The actual count of 2m-letter words with equal T's and O's using only letters T and O.

theorem petya_equals_vasya (m : ℕ) : petya_word_count m = vasya_word_count m :=
  sorry

end petya_equals_vasya_l120_120568


namespace allan_balloons_l120_120356

def jak_balloons : ℕ := 11
def diff_balloons : ℕ := 6

theorem allan_balloons (jake_allan_diff : jak_balloons = diff_balloons + 5) : jak_balloons - diff_balloons = 5 :=
by
  sorry

end allan_balloons_l120_120356


namespace rotation_of_unit_circle_l120_120734

open Real

noncomputable def rotated_coordinates (θ : ℝ) : ℝ × ℝ :=
  ( -sin θ, cos θ )

theorem rotation_of_unit_circle (θ : ℝ) (k : ℤ) (h : θ ≠ k * π + π / 2) :
  let A := (cos θ, sin θ)
  let O := (0, 0)
  let B := rotated_coordinates (θ)
  B = (-sin θ, cos θ) :=
sorry

end rotation_of_unit_circle_l120_120734


namespace find_k_value_for_unique_real_solution_l120_120985

noncomputable def cubic_has_exactly_one_real_solution (k : ℝ) : Prop :=
    ∃! x : ℝ, 4*x^3 + 9*x^2 + k*x + 4 = 0

theorem find_k_value_for_unique_real_solution :
  ∃ (k : ℝ), k > 0 ∧ cubic_has_exactly_one_real_solution k ∧ k = 6.75 :=
sorry

end find_k_value_for_unique_real_solution_l120_120985


namespace adult_ticket_cost_l120_120363

-- Definitions based on given conditions.
def children_ticket_cost : ℝ := 7.5
def total_bill : ℝ := 138
def total_tickets : ℕ := 12
def additional_children_tickets : ℕ := 8

-- Proof statement: Prove the cost of each adult ticket.
theorem adult_ticket_cost (x : ℕ) (A : ℝ)
  (h1 : x + (x + additional_children_tickets) = total_tickets)
  (h2 : x * A + (x + additional_children_tickets) * children_ticket_cost = total_bill) :
  A = 31.50 :=
  sorry

end adult_ticket_cost_l120_120363


namespace dogs_eat_times_per_day_l120_120008

theorem dogs_eat_times_per_day (dogs : ℕ) (food_per_dog_per_meal : ℚ) (total_food : ℚ) 
                                (food_left : ℚ) (days : ℕ) 
                                (dogs_eat_times_per_day : ℚ)
                                (h_dogs : dogs = 3)
                                (h_food_per_dog_per_meal : food_per_dog_per_meal = 1 / 2)
                                (h_total_food : total_food = 30)
                                (h_food_left : food_left = 9)
                                (h_days : days = 7) :
                                dogs_eat_times_per_day = 2 :=
by
  -- Proof goes here
  sorry

end dogs_eat_times_per_day_l120_120008


namespace total_lobster_pounds_l120_120252

theorem total_lobster_pounds
  (combined_other_harbors : ℕ)
  (hooper_bay : ℕ)
  (H1 : combined_other_harbors = 160)
  (H2 : hooper_bay = 2 * combined_other_harbors) :
  combined_other_harbors + hooper_bay = 480 :=
by
  -- proof goes here
  sorry

end total_lobster_pounds_l120_120252


namespace find_k_l120_120766

noncomputable def a_squared : ℝ := 9
noncomputable def b_squared (k : ℝ) : ℝ := 4 + k
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

noncomputable def c_squared_1 (k : ℝ) : ℝ := 5 - k
noncomputable def c_squared_2 (k : ℝ) : ℝ := k - 5

theorem find_k (k : ℝ) :
  (eccentricity (Real.sqrt (c_squared_1 k)) (Real.sqrt a_squared) = 4 / 5 →
   k = -19 / 25) ∨ 
  (eccentricity (Real.sqrt (c_squared_2 k)) (Real.sqrt (b_squared k)) = 4 / 5 →
   k = 21) :=
sorry

end find_k_l120_120766


namespace find_a_l120_120669

namespace MathProof

theorem find_a (a : ℕ) (h_pos : a > 0) (h_eq : (a : ℚ) / (a + 18) = 47 / 50) : a = 282 :=
by
  sorry

end MathProof

end find_a_l120_120669


namespace difference_in_roi_l120_120817

theorem difference_in_roi (E_investment : ℝ) (B_investment : ℝ) (E_rate : ℝ) (B_rate : ℝ) (years : ℕ) :
  E_investment = 300 → B_investment = 500 → E_rate = 0.15 → B_rate = 0.10 → years = 2 →
  (B_rate * B_investment * years) - (E_rate * E_investment * years) = 10 :=
by
  intros E_investment_eq B_investment_eq E_rate_eq B_rate_eq years_eq
  sorry

end difference_in_roi_l120_120817


namespace Maria_selling_price_l120_120881

-- Define the constants based on the given conditions
def brush_cost : ℕ := 20
def canvas_cost : ℕ := 3 * brush_cost
def paint_cost_per_liter : ℕ := 8
def paint_needed : ℕ := 5
def earnings : ℕ := 80

-- Calculate the total cost and the selling price
def total_cost : ℕ := brush_cost + canvas_cost + (paint_cost_per_liter * paint_needed)
def selling_price : ℕ := total_cost + earnings

-- Proof statement
theorem Maria_selling_price : selling_price = 200 := by
  sorry

end Maria_selling_price_l120_120881


namespace three_digit_integers_congruent_to_2_mod_4_l120_120717

theorem three_digit_integers_congruent_to_2_mod_4 : 
  let count := (249 - 25 + 1) in
  count = 225 :=
by
  let k_min := 25
  let k_max := 249
  have h_count : count = (k_max - k_min + 1) := rfl
  rw h_count
  norm_num

end three_digit_integers_congruent_to_2_mod_4_l120_120717


namespace tangent_line_intersect_x_l120_120483

noncomputable def tangent_intercept_x : ℚ := 9/2

theorem tangent_line_intersect_x (x : ℚ)
  (h₁ : x > 0)
  (h₂ : ∃ r₁ r₂ d : ℚ, r₁ = 3 ∧ r₂ = 5 ∧ d = 12 ∧ x = (r₂ * d) / (r₁ + r₂)) :
  x = tangent_intercept_x :=
by
  sorry

end tangent_line_intersect_x_l120_120483


namespace min_sum_a_b_l120_120994

theorem min_sum_a_b (a b : ℕ) (h1 : a ≠ b) (h2 : 0 < a ∧ 0 < b) (h3 : (1/a + 1/b) = 1/12) : a + b = 54 :=
sorry

end min_sum_a_b_l120_120994


namespace gcd_f100_f101_l120_120133

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - x + 2010

-- A statement asserting the greatest common divisor of f(100) and f(101) is 10
theorem gcd_f100_f101 : Int.gcd (f 100) (f 101) = 10 := by
  sorry

end gcd_f100_f101_l120_120133


namespace alton_weekly_profit_l120_120495

-- Definitions of the given conditions
def dailyEarnings : ℕ := 8
def daysInWeek : ℕ := 7
def weeklyRent : ℕ := 20

-- The proof problem: Prove that the total profit every week is $36
theorem alton_weekly_profit : (dailyEarnings * daysInWeek) - weeklyRent = 36 := by
  sorry

end alton_weekly_profit_l120_120495


namespace mixed_number_expression_l120_120361

theorem mixed_number_expression :
  (7 + 1/2 - (5 + 3/4)) * (3 + 1/6 + (2 + 1/8)) = 9 + 25/96 :=
by
  -- here we would provide the proof steps
  sorry

end mixed_number_expression_l120_120361


namespace option_D_functions_same_l120_120920

theorem option_D_functions_same (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by 
  sorry

end option_D_functions_same_l120_120920


namespace no_absolute_winner_prob_l120_120509

open_locale probability

-- Define the probability of Alyosha winning against Borya
def P_A_wins_B : ℝ := 0.6

-- Define the probability of Borya winning against Vasya
def P_B_wins_V : ℝ := 0.4

-- There are no ties, and each player plays with each other once
-- Conditions ensure that all pairs have played exactly once

-- Define the event that there will be no absolute winner
def P_no_absolute_winner : ℝ := P_A_wins_B * P_B_wins_V * 1 + P_A_wins_B * (1 - P_B_wins_V) * (1 - 1)

-- Statement of the problem: Prove that the probability of event C is 0.24
theorem no_absolute_winner_prob :
  P_no_absolute_winner = 0.24 :=
  by
    -- Placeholder for proof
    sorry

end no_absolute_winner_prob_l120_120509


namespace reflection_matrix_solution_l120_120245

variable (a b : ℚ)

def matrix_R : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b], ![-(3/4 : ℚ), (4/5 : ℚ)]]

theorem reflection_matrix_solution (h : matrix_R a b ^ 2 = 1) :
    (a, b) = (-4/5, -3/5) := sorry

end reflection_matrix_solution_l120_120245


namespace frequency_of_middle_group_l120_120555

theorem frequency_of_middle_group (sample_size : ℕ) (x : ℝ) (h : sample_size = 160) (h_rel_freq : x = 0.2) 
  (h_relation : x = (1 / 4) * (10 * x)) : 
  sample_size * x = 32 :=
by
  sorry

end frequency_of_middle_group_l120_120555


namespace no_absolute_winner_probability_l120_120504

-- Define the probabilities of matches
def P_AB : ℝ := 0.6  -- Probability Alyosha wins against Borya
def P_BV : ℝ := 0.4  -- Probability Borya wins against Vasya

-- Define the event C that there is no absolute winner
def event_C (P_AV : ℝ) (P_VB : ℝ) : ℝ :=
  let scenario1 := P_AB * P_BV * P_AV in
  let scenario2 := P_AB * P_VB * (1 - P_AV) in
  scenario1 + scenario2

-- Main theorem to prove
theorem no_absolute_winner_probability : 
  event_C 1 0.6 = 0.24 :=
by
  rw [event_C]
  simp
  norm_num
  sorry

end no_absolute_winner_probability_l120_120504


namespace project_completion_time_l120_120488

theorem project_completion_time (initial_workers : ℕ) (initial_days : ℕ) (extra_workers : ℕ) (extra_days : ℕ) : 
  initial_workers = 10 →
  initial_days = 15 →
  extra_workers = 5 →
  extra_days = 5 →
  total_days = 6 := by
  sorry

end project_completion_time_l120_120488


namespace Mason_tables_needed_l120_120293

theorem Mason_tables_needed
  (w_silverware_piece : ℕ := 4) 
  (n_silverware_piece_per_setting : ℕ := 3) 
  (w_plate : ℕ := 12) 
  (n_plates_per_setting : ℕ := 2) 
  (n_settings_per_table : ℕ := 8) 
  (n_backup_settings : ℕ := 20) 
  (total_weight : ℕ := 5040) : 
  ∃ (n_tables : ℕ), n_tables = 15 :=
by
  sorry

end Mason_tables_needed_l120_120293


namespace sum_xy_22_l120_120404

theorem sum_xy_22 (x y : ℕ) (h1 : 0 < x) (h2 : x < 25) (h3 : 0 < y) (h4 : y < 25) 
  (h5 : x + y + x * y = 118) : x + y = 22 :=
sorry

end sum_xy_22_l120_120404


namespace roots_quadratic_inequality_l120_120570

theorem roots_quadratic_inequality (t x1 x2 : ℝ) (h_eqn : x1 ^ 2 - t * x1 + t = 0) 
  (h_eqn2 : x2 ^ 2 - t * x2 + t = 0) (h_real : x1 + x2 = t) (h_prod : x1 * x2 = t) :
  x1 ^ 2 + x2 ^ 2 ≥ 2 * (x1 + x2) := 
sorry

end roots_quadratic_inequality_l120_120570


namespace find_a_for_exponential_function_l120_120546

theorem find_a_for_exponential_function (a : ℝ) :
  a - 2 = 1 ∧ a > 0 ∧ a ≠ 1 → a = 3 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end find_a_for_exponential_function_l120_120546


namespace pick_three_cards_in_order_l120_120938

theorem pick_three_cards_in_order (deck_size : ℕ) (first_card_ways : ℕ) (second_card_ways : ℕ) (third_card_ways : ℕ) 
  (total_combinations : ℕ) (h1 : deck_size = 52) (h2 : first_card_ways = 52) 
  (h3 : second_card_ways = 51) (h4 : third_card_ways = 50) (h5 : total_combinations = first_card_ways * second_card_ways * third_card_ways) : 
  total_combinations = 132600 := 
by 
  sorry

end pick_three_cards_in_order_l120_120938


namespace intersection_eq_l120_120695

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_eq : A ∩ B = {1, 3} :=
by
  sorry

end intersection_eq_l120_120695


namespace solve_system1_solve_system2_l120_120895

-- Definition for System (1)
theorem solve_system1 (x y : ℤ) (h1 : x - 2 * y = 0) (h2 : 3 * x - y = 5) : x = 2 ∧ y = 1 := 
by
  sorry

-- Definition for System (2)
theorem solve_system2 (x y : ℤ) 
  (h1 : 3 * (x - 1) - 4 * (y + 1) = -1) 
  (h2 : (x / 2) + (y / 3) = -2) : x = -2 ∧ y = -3 := 
by
  sorry

end solve_system1_solve_system2_l120_120895


namespace number_of_trees_in_park_l120_120348

def number_of_trees (length width area_per_tree : ℕ) : ℕ :=
  (length * width) / area_per_tree

theorem number_of_trees_in_park :
  number_of_trees 1000 2000 20 = 100000 :=
by
  sorry

end number_of_trees_in_park_l120_120348


namespace platform_length_is_correct_l120_120646

noncomputable def length_of_platform (time_to_pass_man : ℝ) (time_to_cross_platform : ℝ) (length_of_train : ℝ) : ℝ := 
  length_of_train * time_to_cross_platform / time_to_pass_man - length_of_train

theorem platform_length_is_correct : length_of_platform 8 20 178 = 267 := 
  sorry

end platform_length_is_correct_l120_120646


namespace rhombus_properties_l120_120936

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2
noncomputable def side_length_of_rhombus (d1 d2 : ℝ) : ℝ := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)

theorem rhombus_properties (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 16) :
  area_of_rhombus d1 d2 = 144 ∧ side_length_of_rhombus d1 d2 = Real.sqrt 145 := by
  sorry

end rhombus_properties_l120_120936


namespace alpha_values_m_range_l120_120419

noncomputable section

open Real

def f (x : ℝ) (α : ℝ) : ℝ := 2^(x + cos α) - 2^(-x + cos α)

-- Problem 1: Set of values for α
theorem alpha_values (h : f 1 α = 3/4) : ∃ k : ℤ, α = 2 * k * π + π :=
sorry

-- Problem 2: Range of values for real number m
theorem m_range (h0 : 0 ≤ θ ∧ θ ≤ π / 2) 
  (h1 : ∀ (m : ℝ), f (m * cos θ) (-1) + f (1 - m) (-1) > 0) : 
  ∀ (m : ℝ), m < 1 :=
sorry

end alpha_values_m_range_l120_120419


namespace factorial_division_identity_l120_120962

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l120_120962


namespace natasha_average_speed_l120_120010

theorem natasha_average_speed
  (time_up time_down : ℝ)
  (speed_up distance_up total_distance total_time average_speed : ℝ)
  (h1 : time_up = 4)
  (h2 : time_down = 2)
  (h3 : speed_up = 3)
  (h4 : distance_up = speed_up * time_up)
  (h5 : total_distance = distance_up + distance_up)
  (h6 : total_time = time_up + time_down)
  (h7 : average_speed = total_distance / total_time) :
  average_speed = 4 := by
  sorry

end natasha_average_speed_l120_120010


namespace jade_transactions_l120_120042

-- Definitions for each condition
def transactions_mabel : ℕ := 90
def transactions_anthony : ℕ := transactions_mabel + (transactions_mabel / 10)
def transactions_cal : ℕ := 2 * transactions_anthony / 3
def transactions_jade : ℕ := transactions_cal + 17

-- The theorem stating that Jade handled 83 transactions
theorem jade_transactions : transactions_jade = 83 := by
  sorry

end jade_transactions_l120_120042


namespace max_value_fn_l120_120899

theorem max_value_fn : ∀ x : ℝ, y = 1 / (|x| + 2) → 
  ∃ y : ℝ, y = 1 / 2 ∧ ∀ x : ℝ, 1 / (|x| + 2) ≤ y :=
sorry

end max_value_fn_l120_120899


namespace factorial_div_l120_120959

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l120_120959


namespace train_length_l120_120352

theorem train_length (speed_km_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_hr = 72) (h_time : time_sec = 12) : 
  ∃ length_m : ℕ, length_m = 240 := 
by
  sorry

end train_length_l120_120352


namespace find_a_14_l120_120104

variable {α : Type} [LinearOrderedField α]

-- Define the arithmetic sequence sum formula
def arithmetic_seq_sum (a_1 d : α) (n : ℕ) : α :=
  n * a_1 + n * (n - 1) / 2 * d

-- Define the nth term of an arithmetic sequence
def arithmetic_seq_nth (a_1 d : α) (n : ℕ) : α :=
  a_1 + (n - 1 : ℕ) * d

theorem find_a_14
  (a_1 d : α)
  (h1 : arithmetic_seq_sum a_1 d 11 = 55)
  (h2 : arithmetic_seq_nth a_1 d 10 = 9) :
  arithmetic_seq_nth a_1 d 14 = 13 :=
by
  sorry

end find_a_14_l120_120104


namespace broken_marbles_total_l120_120087

theorem broken_marbles_total :
  let broken_set_1 := 0.10 * 50
  let broken_set_2 := 0.20 * 60
  let broken_set_3 := 0.30 * 70
  let broken_set_4 := 0.15 * 80
  let total_broken := broken_set_1 + broken_set_2 + broken_set_3 + broken_set_4
  total_broken = 50 :=
by
  sorry


end broken_marbles_total_l120_120087


namespace solve_system_l120_120070

theorem solve_system (x y : ℚ) 
  (h₁ : 7 * x - 14 * y = 3) 
  (h₂ : 3 * y - x = 5) : 
  x = 79 / 7 ∧ y = 38 / 7 := 
by 
  sorry

end solve_system_l120_120070


namespace count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle_l120_120703

noncomputable def count_equally_spaced_integers : ℕ := 
  sorry

theorem count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle:
  count_equally_spaced_integers = 4 :=
sorry

end count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle_l120_120703


namespace probability_no_absolute_winner_l120_120507

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end probability_no_absolute_winner_l120_120507


namespace complex_number_solution_l120_120227

def i : ℂ := Complex.I

theorem complex_number_solution (z : ℂ) (h : z * (1 - i) = 2 * i) : z = -1 + i :=
by
  sorry

end complex_number_solution_l120_120227


namespace bob_cleaning_time_l120_120557

theorem bob_cleaning_time (alice_time : ℕ) (h1 : alice_time = 25) (bob_ratio : ℚ) (h2 : bob_ratio = 2 / 5) : 
  bob_time = 10 :=
by
  -- Definitions for conditions
  let bob_time := bob_ratio * alice_time
  -- Sorry to represent the skipped proof
  sorry

end bob_cleaning_time_l120_120557


namespace remainder_18_pow_63_mod_5_l120_120467

theorem remainder_18_pow_63_mod_5 :
  (18:ℤ) ^ 63 % 5 = 2 :=
by
  -- Given conditions
  have h1 : (18:ℤ) % 5 = 3 := by norm_num
  have h2 : (3:ℤ) ^ 4 % 5 = 1 := by norm_num
  sorry

end remainder_18_pow_63_mod_5_l120_120467


namespace cells_at_end_of_9th_day_l120_120183

def initial_cells : ℕ := 4
def split_ratio : ℕ := 3
def total_days : ℕ := 9
def days_per_split : ℕ := 3

def num_terms : ℕ := total_days / days_per_split

noncomputable def number_of_cells (initial_cells split_ratio num_terms : ℕ) : ℕ :=
  initial_cells * split_ratio ^ (num_terms - 1)

theorem cells_at_end_of_9th_day :
  number_of_cells initial_cells split_ratio num_terms = 36 :=
by
  sorry

end cells_at_end_of_9th_day_l120_120183


namespace no_absolute_winner_prob_l120_120508

open_locale probability

-- Define the probability of Alyosha winning against Borya
def P_A_wins_B : ℝ := 0.6

-- Define the probability of Borya winning against Vasya
def P_B_wins_V : ℝ := 0.4

-- There are no ties, and each player plays with each other once
-- Conditions ensure that all pairs have played exactly once

-- Define the event that there will be no absolute winner
def P_no_absolute_winner : ℝ := P_A_wins_B * P_B_wins_V * 1 + P_A_wins_B * (1 - P_B_wins_V) * (1 - 1)

-- Statement of the problem: Prove that the probability of event C is 0.24
theorem no_absolute_winner_prob :
  P_no_absolute_winner = 0.24 :=
  by
    -- Placeholder for proof
    sorry

end no_absolute_winner_prob_l120_120508


namespace inequality_f_lt_g_range_of_a_l120_120749

def f (x : ℝ) : ℝ := |x - 4|
def g (x : ℝ) : ℝ := |2 * x + 1|

theorem inequality_f_lt_g :
  ∀ x : ℝ, f x = |x - 4| ∧ g x = |2 * x + 1| →
  (f x < g x ↔ (x < -5 ∨ x > 1)) :=
by
   sorry

theorem range_of_a :
  ∀ x a : ℝ, f x = |x - 4| ∧ g x = |2 * x + 1| →
  (2 * f x + g x > a * x) →
  (-4 ≤ a ∧ a < 9/4) :=
by
   sorry

end inequality_f_lt_g_range_of_a_l120_120749


namespace delores_money_left_l120_120370

theorem delores_money_left (initial_amount spent_computer spent_printer : ℝ) 
    (h1 : initial_amount = 450) 
    (h2 : spent_computer = 400) 
    (h3 : spent_printer = 40) : 
    initial_amount - (spent_computer + spent_printer) = 10 := 
by 
    sorry

end delores_money_left_l120_120370


namespace thabo_paperback_diff_l120_120765

variable (total_books : ℕ) (H_books : ℕ) (P_books : ℕ) (F_books : ℕ)

def thabo_books_conditions :=
  total_books = 160 ∧
  H_books = 25 ∧
  P_books > H_books ∧
  F_books = 2 * P_books ∧
  total_books = F_books + P_books + H_books 

theorem thabo_paperback_diff :
  thabo_books_conditions total_books H_books P_books F_books → 
  (P_books - H_books) = 20 :=
by
  sorry

end thabo_paperback_diff_l120_120765


namespace eight_bags_weight_l120_120405

theorem eight_bags_weight :
  ∀ (total_bags : ℕ) (total_weight : ℚ) (bags_needed: ℕ), 
    total_bags = 12 → 
    total_weight = 24 → 
    bags_needed = 8 → 
    total_weight / total_bags * bags_needed = 16 :=
begin
  intros total_bags total_weight bags_needed hb hw hn,
  rw [hb, hw, hn],
  norm_num,
end

end eight_bags_weight_l120_120405


namespace no_solution_to_inequality_l120_120763

theorem no_solution_to_inequality (x : ℝ) (h : x ≥ -1/4) : ¬(-1 - 1 / (3 * x + 4) < 2) :=
by sorry

end no_solution_to_inequality_l120_120763


namespace smallest_solution_of_abs_eq_l120_120826

theorem smallest_solution_of_abs_eq (x : ℝ) : 
  (x * |x| = 3 * x + 2 → x ≥ 0 → x = (3 + Real.sqrt 17) / 2) ∧
  (x * |x| = 3 * x + 2 → x < 0 → x = -2) ∧
  (x * |x| = 3 * x + 2 → x = -2 → x = -2) :=
by
  sorry

end smallest_solution_of_abs_eq_l120_120826


namespace decompose_expression_l120_120366

-- Define the variables a and b as real numbers
variables (a b : ℝ)

-- State the theorem corresponding to the proof problem
theorem decompose_expression : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) :=
by
  sorry

end decompose_expression_l120_120366


namespace possible_value_of_2n_plus_m_l120_120274

variable (n m : ℤ)

theorem possible_value_of_2n_plus_m : (3 * n - m < 5) → (n + m > 26) → (3 * m - 2 * n < 46) → (2 * n + m = 36) :=
by
  sorry

end possible_value_of_2n_plus_m_l120_120274


namespace rd_expense_necessary_for_increase_l120_120521

theorem rd_expense_necessary_for_increase :
  ∀ (R_and_D_t : ℝ) (delta_APL_t1 : ℝ),
  R_and_D_t = 3289.31 → delta_APL_t1 = 1.55 →
  R_and_D_t / delta_APL_t1 = 2122 := 
by
  intros R_and_D_t delta_APL_t1 hR hD
  rw [hR, hD]
  norm_num
  sorry

end rd_expense_necessary_for_increase_l120_120521


namespace find_x_l120_120833

theorem find_x (x : ℝ) (y : ℝ) : (∀ y, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 := by
  intros h
  -- At this point, you would include the necessary proof steps, but for now we skip it.
  sorry

end find_x_l120_120833


namespace solve_for_n_l120_120528

theorem solve_for_n : ∃ n : ℤ, 3^3 - 5 = 4^2 + n ∧ n = 6 := 
by
  use 6
  sorry

end solve_for_n_l120_120528


namespace transformations_count_l120_120132

def transformations := {rotation := (λ (p : ℝ × ℝ), (-p.2, p.1)),
                        reflection_x := (λ (p : ℝ × ℝ), (p.1, -p.2)),
                        reflection_y := (λ (p : ℝ × ℝ), (-p.1, p.2)),
                        translation := (λ (p : ℝ × ℝ), (p.1 + 6, p.2 + 2))}

def apply_transformations (seq : list (ℝ × ℝ → ℝ × ℝ)) (p : ℝ × ℝ)  :=
seq.foldl (λ acc f, f acc) p 

noncomputable def count_valid_sequences : ℕ :=
(finset.univ : finset (vector (ℝ × ℝ → ℝ × ℝ) 4)).filter (λ seq,
  let transformed_rectangle := set.image (apply_transformations seq.val) (set.of (λ pt, pt ∈ {(0,0), (6,0), (6,2), (0,2)})) in
  transformed_rectangle = {(0,0), (6,0), (6,2), (0,2)}).card

theorem transformations_count : count_valid_sequences = 3 :=
sorry

end transformations_count_l120_120132


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l120_120705

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l120_120705


namespace factorize_expression_l120_120818

-- Define the expression E
def E (x y z : ℝ) : ℝ := x^2 + x*y - x*z - y*z

-- State the theorem to prove \(E = (x + y)(x - z)\)
theorem factorize_expression (x y z : ℝ) : 
  E x y z = (x + y) * (x - z) := 
sorry

end factorize_expression_l120_120818


namespace perpendicular_lines_foot_l120_120688

variables (a b c : ℝ)

theorem perpendicular_lines_foot (h1 : a * -2/20 = -1)
  (h2_foot_l1 : a * 1 + 4 * c - 2 = 0)
  (h3_foot_l2 : 2 * 1 - 5 * c + b = 0) :
  a + b + c = -4 :=
sorry

end perpendicular_lines_foot_l120_120688


namespace delores_money_left_l120_120368

def initial : ℕ := 450
def computer_cost : ℕ := 400
def printer_cost : ℕ := 40
def money_left (initial computer_cost printer_cost : ℕ) : ℕ := initial - (computer_cost + printer_cost)

theorem delores_money_left : money_left initial computer_cost printer_cost = 10 := by
  sorry

end delores_money_left_l120_120368


namespace pyramid_volume_QEFGH_l120_120572

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * EF * FG * QE

theorem pyramid_volume_QEFGH :
  let EF := 10
  let FG := 5
  let QE := 9
  volume_of_pyramid EF FG QE = 150 := by
  sorry

end pyramid_volume_QEFGH_l120_120572


namespace problem_2535_l120_120001

theorem problem_2535 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 1) :
  a + b + (a^3 / b^2) + (b^3 / a^2) = 2535 := sorry

end problem_2535_l120_120001


namespace evaluate_expression_l120_120940

theorem evaluate_expression : (5^2 - 4^2)^3 = 729 :=
by
  sorry

end evaluate_expression_l120_120940


namespace bucket_weight_full_l120_120181

theorem bucket_weight_full (c d : ℝ) (x y : ℝ) 
  (h1 : x + (1 / 3) * y = c) 
  (h2 : x + (3 / 4) * y = d) : 
  x + y = (-3 * c + 8 * d) / 5 :=
sorry

end bucket_weight_full_l120_120181


namespace cumulus_to_cumulonimbus_ratio_l120_120905

theorem cumulus_to_cumulonimbus_ratio (cirrus cumulonimbus cumulus : ℕ) (x : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = x * cumulonimbus)
  (h3 : cumulonimbus = 3)
  (h4 : cirrus = 144) :
  x = 12 := by
  sorry

end cumulus_to_cumulonimbus_ratio_l120_120905


namespace difference_between_neutrons_and_electrons_l120_120201

def proton_number : Nat := 118
def mass_number : Nat := 293

def number_of_neutrons : Nat := mass_number - proton_number
def number_of_electrons : Nat := proton_number

theorem difference_between_neutrons_and_electrons :
  (number_of_neutrons - number_of_electrons) = 57 := by
  sorry

end difference_between_neutrons_and_electrons_l120_120201


namespace soccer_campers_l120_120644

theorem soccer_campers (total_campers : ℕ) (basketball_campers : ℕ) (football_campers : ℕ) (h1 : total_campers = 88) (h2 : basketball_campers = 24) (h3 : football_campers = 32) : 
  total_campers - (basketball_campers + football_campers) = 32 := 
by 
  -- Proof omitted
  sorry

end soccer_campers_l120_120644


namespace carpet_length_l120_120053

-- Define the conditions as hypotheses
def width_of_carpet : ℝ := 4
def area_of_living_room : ℝ := 60

-- Formalize the corresponding proof problem
theorem carpet_length (h : 60 = width_of_carpet * length) : length = 15 :=
sorry

end carpet_length_l120_120053


namespace no_99_percent_confidence_distribution_expectation_variance_l120_120018

open ProbabilityTheory MeasureTheory

-- Data from the conditions
def a : ℕ := 40
def b : ℕ := 10
def c : ℕ := 30
def d : ℕ := 20
def n : ℕ := 100

-- Definitions required for the part (1)
def k_square : ℝ := (n * ((a * d - b * c) ^ 2) : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Statement for part (1)
theorem no_99_percent_confidence : k_square < 6.635 :=
sorry

-- Definitions required for part (2)
noncomputable def p : ℝ := 2 / 5

def X : Type := Fin₃

def P_X (k : ℕ) : ℝ :=
  if k = 0 then (3.choose 0) * ((1 - p) ^ 3)
  else if k = 1 then (3.choose 1) * (p) * ((1 - p) ^ 2)
  else if k = 2 then (3.choose 2) * (p ^ 2) * ((1 - p))
  else if k = 3 then (3.choose 3) * (p ^ 3)
  else 0

-- Discrete table for distribution of X
def distribution_X : List (ℕ × ℝ) :=
  [(0, P_X 0), (1, P_X 1), (2, P_X 2), (3, P_X 3)]

-- Expected value
def E_X : ℝ := 3 * p

-- Variance
def var_X : ℝ := 3 * p * (1 - p)

theorem distribution_expectation_variance :
  distribution_X = [(0, 27/125), (1, 54/125), (2, 36/125), (3, 8/125)] ∧
  E_X = 6/5 ∧
  var_X = 18/25 :=
sorry

end no_99_percent_confidence_distribution_expectation_variance_l120_120018


namespace factorial_division_l120_120957

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l120_120957


namespace special_pair_example_1_special_pair_example_2_special_pair_negation_l120_120760

-- Definition of "special rational number pair"
def is_special_rational_pair (a b : ℚ) : Prop := a + b = a * b - 1

-- Problem (1)
theorem special_pair_example_1 : is_special_rational_pair 5 (3 / 2) :=
  by sorry

-- Problem (2)
theorem special_pair_example_2 (a : ℚ) : is_special_rational_pair a 3 → a = 2 :=
  by sorry

-- Problem (3)
theorem special_pair_negation (m n : ℚ) : is_special_rational_pair m n → ¬ is_special_rational_pair (-n) (-m) :=
  by sorry

end special_pair_example_1_special_pair_example_2_special_pair_negation_l120_120760


namespace range_of_a_l120_120088

-- Define conditions
def setA : Set ℝ := {x | x^2 - x ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0}

-- Problem statement in Lean 4
theorem range_of_a (a : ℝ) (h : setA ⊆ setB a) : a ≤ -2 :=
by
  sorry

end range_of_a_l120_120088


namespace relationship_abc_l120_120090

theorem relationship_abc (a b c : ℝ) 
  (h₁ : a = Real.log 0.5 / Real.log 2) 
  (h₂ : b = Real.sqrt 2) 
  (h₃ : c = 0.5 ^ 2) : 
  a < c ∧ c < b := by
  sorry

end relationship_abc_l120_120090


namespace cost_formula_l120_120309

def cost (P : ℕ) : ℕ :=
  if P ≤ 5 then 5 * P + 10 else 5 * P + 5

theorem cost_formula (P : ℕ) : 
  cost P = (if P ≤ 5 then 5 * P + 10 else 5 * P + 5) :=
by 
  sorry

end cost_formula_l120_120309


namespace product_of_consecutive_natural_numbers_l120_120677

theorem product_of_consecutive_natural_numbers (n : ℕ) : 
  (∃ t : ℕ, n = t * (t + 1) - 1) ↔ ∃ x : ℕ, n^2 - 1 = x * (x + 1) * (x + 2) * (x + 3) := 
sorry

end product_of_consecutive_natural_numbers_l120_120677


namespace average_after_12th_innings_l120_120792

variable (runs_11 score_12 increase_avg : ℕ)
variable (A : ℕ)

theorem average_after_12th_innings
  (h1 : score_12 = 60)
  (h2 : increase_avg = 2)
  (h3 : 11 * A = runs_11)
  (h4 : (runs_11 + score_12) / 12 = A + increase_avg) :
  (A + 2 = 38) :=
by
  sorry

end average_after_12th_innings_l120_120792


namespace symmetry_about_origin_l120_120589

def f (x : ℝ) : ℝ := x^3 - x

theorem symmetry_about_origin : 
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end symmetry_about_origin_l120_120589


namespace perpendicular_lines_l120_120244

def line_l1 (m x y : ℝ) : Prop := m * x - y + 1 = 0
def line_l2 (m x y : ℝ) : Prop := 2 * x - (m - 1) * y + 1 = 0

theorem perpendicular_lines (m : ℝ): (∃ x y : ℝ, line_l1 m x y) ∧ (∃ x y : ℝ, line_l2 m x y) ∧ (∀ x y : ℝ, line_l1 m x y → line_l2 m x y → m * (2 / (m - 1)) = -1) → m = 1 / 3 := by
  sorry

end perpendicular_lines_l120_120244


namespace total_weight_of_balls_l120_120886

theorem total_weight_of_balls :
  let weight_blue := 6
  let weight_brown := 3.12
  weight_blue + weight_brown = 9.12 :=
by
  sorry

end total_weight_of_balls_l120_120886


namespace circumradius_eq_exradius_opposite_BC_l120_120839

-- Definitions of points and triangles
variable {A B C : Point}
variable (O I D : Point)
variable {α β γ : Angle}

-- Definitions of circumcenter, incenter, altitude, and collinearity
def is_circumcenter (O : Point) (A B C : Point) : Prop := sorry
def is_incenter (I : Point) (A B C : Point) : Prop := sorry
def is_altitude (A D B C : Point) : Prop := sorry
def collinear (O D I : Point) : Prop := sorry

-- Definitions of circumradius and exradius
def circumradius (A B C : Point) : ℝ := sorry
def exradius_opposite_BC (A B C : Point) : ℝ := sorry

-- Main theorem statement
theorem circumradius_eq_exradius_opposite_BC
  (h_circ : is_circumcenter O A B C)
  (h_incenter : is_incenter I A B C)
  (h_altitude : is_altitude A D B C)
  (h_collinear : collinear O D I) : 
  circumradius A B C = exradius_opposite_BC A B C :=
sorry

end circumradius_eq_exradius_opposite_BC_l120_120839


namespace roots_poly_eval_l120_120289

theorem roots_poly_eval : ∀ (c d : ℝ), (c + d = 6 ∧ c * d = 8) → c^4 + c^3 * d + d^3 * c + d^4 = 432 :=
by
  intros c d h
  sorry

end roots_poly_eval_l120_120289


namespace log_div_log_inv_of_16_l120_120468

theorem log_div_log_inv_of_16 : (Real.log 16) / (Real.log (1 / 16)) = -1 :=
by
  sorry

end log_div_log_inv_of_16_l120_120468


namespace find_m_l120_120547

/-- 
If the function y=x + m/(x-1) defined for x > 1 attains its minimum value at x = 3,
then the positive number m is 4.
-/
theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 < x -> x + m / (x - 1) ≥ 3 + m / 2):
  m = 4 :=
sorry

end find_m_l120_120547


namespace bus_travel_time_kimovsk_moscow_l120_120314

noncomputable def travel_time_kimovsk_moscow (d1 d2 d3: ℝ) (max_speed: ℝ) (t_kt: ℝ) (t_nm: ℝ) : Prop :=
  35 ≤ d1 ∧ d1 ≤ 35 ∧
  60 ≤ d2 ∧ d2 ≤ 60 ∧
  200 ≤ d3 ∧ d3 ≤ 200 ∧
  max_speed <= 60 ∧
  2 ≤ t_kt ∧ t_kt ≤ 2 ∧
  5 ≤ t_nm ∧ t_nm ≤ 5 ∧
  (5 + 7/12 : ℝ) ≤ t_kt + t_nm ∧ t_kt + t_nm ≤ 6

theorem bus_travel_time_kimovsk_moscow
  (d1 d2 d3 : ℝ) (max_speed : ℝ) (t_kt : ℝ) (t_nm : ℝ) :
  travel_time_kimovsk_moscow d1 d2 d3 max_speed t_kt t_nm := 
by
  sorry

end bus_travel_time_kimovsk_moscow_l120_120314


namespace volleyball_tournament_first_place_score_l120_120382

theorem volleyball_tournament_first_place_score :
  ∃ (a b c d : ℕ), (a + b + c + d = 18) ∧ (a < b ∧ b < c ∧ c < d) ∧ (d = 6) :=
by
  sorry

end volleyball_tournament_first_place_score_l120_120382


namespace problem1_problem2_l120_120949

section Calculations

-- Problem 1
theorem problem1 : sqrt 8 / sqrt 2 + (sqrt 5 + 3) * (sqrt 5 - 3) = -2 := by
  sorry

-- Problem 2
theorem problem2 : sqrt 27 + abs (1 - sqrt 3) + (1 / 3 : ℝ)⁻¹ - (π - 3)^0 = 4 * sqrt 3 + 1 := by
  sorry

end Calculations

end problem1_problem2_l120_120949


namespace find_m_collinear_l120_120400

theorem find_m_collinear (m : ℝ) 
    (a : ℝ × ℝ := (m + 3, 2)) 
    (b : ℝ × ℝ := (m, 1)) 
    (collinear : a.1 * 1 - 2 * b.1 = 0) : 
    m = 3 :=
by {
    sorry
}

end find_m_collinear_l120_120400


namespace total_oranges_over_four_days_l120_120761

def jeremy_oranges_monday := 100
def jeremy_oranges_tuesday (B: ℕ) := 3 * jeremy_oranges_monday
def jeremy_oranges_wednesday (B: ℕ) (C: ℕ) := 2 * (jeremy_oranges_monday + B)
def jeremy_oranges_thursday := 70
def brother_oranges_tuesday := 3 * jeremy_oranges_monday - jeremy_oranges_monday -- This is B from Tuesday
def cousin_oranges_wednesday (B: ℕ) (C: ℕ) := 2 * (jeremy_oranges_monday + B) - (jeremy_oranges_monday + B)

theorem total_oranges_over_four_days (B: ℕ) (C: ℕ)
        (B_equals_tuesday: B = brother_oranges_tuesday)
        (J_plus_B_equals_300 : jeremy_oranges_tuesday B = 300)
        (J_plus_B_plus_C_equals_600 : jeremy_oranges_wednesday B C = 600)
        (J_thursday_is_70 : jeremy_oranges_thursday = 70)
        (B_thursday_is_B : B = brother_oranges_tuesday):
    100 + 300 + 600 + 270 = 1270 := by
        sorry

end total_oranges_over_four_days_l120_120761


namespace total_budget_is_correct_l120_120757

-- Define the costs of TV, fridge, and computer based on the given conditions
def cost_tv : ℕ := 600
def cost_computer : ℕ := 250
def cost_fridge : ℕ := cost_computer + 500

-- Statement to prove the total budget
theorem total_budget_is_correct : cost_tv + cost_computer + cost_fridge = 1600 :=
by
  sorry

end total_budget_is_correct_l120_120757


namespace factorial_expression_l120_120954

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l120_120954


namespace initial_number_of_nurses_l120_120450

theorem initial_number_of_nurses (N : ℕ) (initial_doctors : ℕ) (remaining_staff : ℕ) 
  (h1 : initial_doctors = 11) 
  (h2 : remaining_staff = 22) 
  (h3 : initial_doctors - 5 + N - 2 = remaining_staff) : N = 18 :=
by
  rw [h1, h2] at h3
  sorry

end initial_number_of_nurses_l120_120450


namespace lcm_10_to_30_l120_120984

def list_of_ints := [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

def lcm_of_list (l : List Nat) : Nat :=
  l.foldr Nat.lcm 1

theorem lcm_10_to_30 : lcm_of_list list_of_ints = 232792560 :=
  sorry

end lcm_10_to_30_l120_120984


namespace exist_divisible_number_l120_120013

theorem exist_divisible_number (d : ℕ) (hd : d > 0) :
  ∃ n : ℕ, (n % d = 0) ∧ ∃ k : ℕ, (k > 0) ∧ (k < 10) ∧ 
  ((∃ m : ℕ, m = n - k*(10^k / 10^k) ∧ m % d = 0) ∨ ∃ m : ℕ, m = n - k * (10^(k - 1)) ∧ m % d = 0) :=
sorry

end exist_divisible_number_l120_120013


namespace more_triangles_with_perimeter_2003_than_2000_l120_120657

theorem more_triangles_with_perimeter_2003_than_2000 :
  (∃ (count_2003 count_2000 : ℕ), 
   count_2003 > count_2000 ∧ 
   (∀ (a b c : ℕ), a + b + c = 2000 → a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧ 
   (∀ (a b c : ℕ), a + b + c = 2003 → a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a))
  := 
sorry

end more_triangles_with_perimeter_2003_than_2000_l120_120657


namespace total_number_of_fish_l120_120594

def number_of_tuna : Nat := 5
def number_of_spearfish : Nat := 2

theorem total_number_of_fish : number_of_tuna + number_of_spearfish = 7 := by
  sorry

end total_number_of_fish_l120_120594


namespace area_of_table_l120_120489

-- Definitions of the given conditions
def free_side_conditions (L W : ℝ) : Prop :=
  (L = 2 * W) ∧ (2 * W + L = 32)

-- Statement to prove the area of the rectangular table
theorem area_of_table {L W : ℝ} (h : free_side_conditions L W) : L * W = 128 := by
  sorry

end area_of_table_l120_120489


namespace television_combinations_l120_120678

def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem television_combinations :
  ∃ (combinations : ℕ), 
  ∀ (A B total : ℕ), A = 4 → B = 5 → total = 3 →
  combinations = (combination 4 2 * combination 5 1 + combination 4 1 * combination 5 2) →
  combinations = 70 :=
sorry

end television_combinations_l120_120678


namespace wrapping_paper_l120_120889

theorem wrapping_paper (total_used_per_roll : ℚ) (number_of_presents : ℕ) (fraction_used : ℚ) (fraction_left : ℚ) 
  (h1 : total_used_per_roll = 2 / 5) 
  (h2 : number_of_presents = 5) 
  (h3 : fraction_used = total_used_per_roll / number_of_presents) 
  (h4 : fraction_left = 1 - total_used_per_roll) : 
  fraction_used = 2 / 25 ∧ fraction_left = 3 / 5 := 
by 
  sorry

end wrapping_paper_l120_120889


namespace no_absolute_winner_l120_120511

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end no_absolute_winner_l120_120511


namespace fixed_point_for_line_l120_120974

theorem fixed_point_for_line (m : ℝ) : (m * (1 - 1) + (1 - 1) = 0) :=
by
  sorry

end fixed_point_for_line_l120_120974


namespace smallest_solution_l120_120827

theorem smallest_solution (x : ℝ) (h₁ : x ≥ 0 → x^2 - 3*x - 2 = 0 → x = (3 + Real.sqrt 17) / 2)
                         (h₂ : x < 0 → x^2 + 3*x + 2 = 0 → (x = -1 ∨ x = -2)) :
  x = -2 :=
by
  sorry

end smallest_solution_l120_120827


namespace quadratic_two_distinct_real_roots_l120_120014

theorem quadratic_two_distinct_real_roots
  (a1 a2 a3 a4 : ℝ)
  (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - (a1 + a2 + a3 + a4) * x1 + (a1 * a3 + a2 * a4) = 0)
  ∧ (x2^2 - (a1 + a2 + a3 + a4) * x2 + (a1 * a3 + a2 * a4) = 0) :=
by 
  sorry

end quadratic_two_distinct_real_roots_l120_120014


namespace number_of_red_socks_l120_120004

-- Definitions:
def red_sock_pairs (R : ℕ) := R
def red_sock_cost (R : ℕ) := 3 * R
def blue_socks_pairs : ℕ := 6
def blue_sock_cost : ℕ := 5
def total_amount_spent := 42

-- Proof Statement
theorem number_of_red_socks (R : ℕ) (h : red_sock_cost R + blue_socks_pairs * blue_sock_cost = total_amount_spent) : 
  red_sock_pairs R = 4 :=
by 
  sorry

end number_of_red_socks_l120_120004


namespace matrix_det_is_zero_l120_120655

noncomputable def matrixDetProblem (a b : ℝ) : ℝ :=
  Matrix.det ![
    ![1, Real.cos (a - b), Real.sin a],
    ![Real.cos (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]

theorem matrix_det_is_zero (a b : ℝ) : matrixDetProblem a b = 0 :=
  sorry

end matrix_det_is_zero_l120_120655


namespace find_q_l120_120724

theorem find_q (q : ℕ) (h1 : 32 = 2^5) (h2 : 32^5 = 2^q) : q = 25 := by
  sorry

end find_q_l120_120724


namespace probability_of_winning_pair_is_correct_l120_120867

noncomputable def probability_of_winning_pair : ℚ :=
  let total_cards := 10
  let red_cards := 5
  let blue_cards := 5
  let total_ways := Nat.choose total_cards 2 -- Combination C(10,2)
  let same_color_ways := Nat.choose red_cards 2 + Nat.choose blue_cards 2 -- Combination C(5,2) for each color
  let consecutive_pairs_per_color := 4
  let consecutive_ways := 2 * consecutive_pairs_per_color -- Two colors
  let favorable_ways := same_color_ways + consecutive_ways
  favorable_ways / total_ways

theorem probability_of_winning_pair_is_correct : 
  probability_of_winning_pair = 28 / 45 := sorry

end probability_of_winning_pair_is_correct_l120_120867


namespace inequality_transformation_l120_120541

theorem inequality_transformation (m n : ℝ) (h : -m / 2 < -n / 6) : 3 * m > n := by
  sorry

end inequality_transformation_l120_120541


namespace max_children_l120_120139

theorem max_children (x : ℕ) (h1 : x * (x - 2) + 2 * 5 = 58) : x = 8 :=
by
  sorry

end max_children_l120_120139


namespace simplify_expression_l120_120089

theorem simplify_expression (a : ℝ) (h : a < 1 / 4) : 4 * (4 * a - 1)^2 = (1 - 4 * a)^(2 : ℝ) :=
by sorry

end simplify_expression_l120_120089


namespace determinant_expression_l120_120562

theorem determinant_expression (a b c d p q r : ℝ)
  (h1: (∃ x: ℝ, x^4 + p*x^2 + q*x + r = 0) → (x = a ∨ x = b ∨ x = c ∨ x = d))
  (h2: a*b + a*c + a*d + b*c + b*d + c*d = p)
  (h3: a*b*c + a*b*d + a*c*d + b*c*d = q)
  (h4: a*b*c*d = -r):
  (Matrix.det ![![1 + a, 1, 1, 1], ![1, 1 + b, 1, 1], ![1, 1, 1 + c, 1], ![1, 1, 1, 1 + d]]) 
  = r + q + p := 
sorry

end determinant_expression_l120_120562


namespace theater_ticket_sales_l120_120776

theorem theater_ticket_sales (x y : ℕ) (h1 : x + y = 175) (h2 : 6 * x + 2 * y = 750) : y = 75 :=
sorry

end theater_ticket_sales_l120_120776


namespace find_p_inversely_proportional_l120_120583

theorem find_p_inversely_proportional :
  ∀ (p q r : ℚ), (p * (r * q) = k) → (p = 16) → (q = 8) → (r = 2) →
  (k = 256) → (q' = 10) → (r' = 3) →
  (∃ p' : ℚ, p' = 128 / 15) :=
by
  sorry

end find_p_inversely_proportional_l120_120583


namespace solve_for_y_l120_120112

theorem solve_for_y (x y : ℝ) (h1 : 2 * x - y = 10) (h2 : x + 3 * y = 2) : y = -6 / 7 := 
by
  sorry

end solve_for_y_l120_120112


namespace largest_gold_coins_l120_120620

noncomputable def max_gold_coins (n : ℕ) : ℕ :=
  if h : ∃ k : ℕ, n = 13 * k + 3 ∧ n < 150 then
    n
  else 0

theorem largest_gold_coins : max_gold_coins 146 = 146 :=
by
  sorry

end largest_gold_coins_l120_120620


namespace president_and_committee_l120_120412

def combinatorial (n k : ℕ) : ℕ := Nat.choose n k

theorem president_and_committee :
  let num_people := 10
  let num_president := 1
  let num_committee := 3
  let num_ways_president := 10
  let num_remaining_people := num_people - num_president
  let num_ways_committee := combinatorial num_remaining_people num_committee
  num_ways_president * num_ways_committee = 840 := 
by
  sorry

end president_and_committee_l120_120412


namespace root_sum_reciprocal_l120_120291

theorem root_sum_reciprocal (p q r s : ℂ)
  (h1 : (∀ x : ℂ, x^4 - 6*x^3 + 11*x^2 - 6*x + 3 = 0 → x = p ∨ x = q ∨ x = r ∨ x = s))
  (h2 : p*q*r*s = 3) 
  (h3 : p*q + p*r + p*s + q*r + q*s + r*s = 11) :
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s)) = 11/3 :=
by
  sorry

end root_sum_reciprocal_l120_120291


namespace largest_number_of_gold_coins_l120_120622

theorem largest_number_of_gold_coins (n : ℕ) :
  (∃ k : ℕ, n = 13 * k + 3 ∧ n < 150) → n ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l120_120622


namespace complex_magnitude_l120_120240

theorem complex_magnitude (z : ℂ) (h : z * (2 - 4 * Complex.I) = 1 + 3 * Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 :=
by
  sorry

end complex_magnitude_l120_120240


namespace solve_for_x_l120_120301

theorem solve_for_x (x : ℝ) (hx_pos : x > 0) (h_eq : 3 * x^2 + 13 * x - 10 = 0) : x = 2 / 3 :=
sorry

end solve_for_x_l120_120301


namespace number_of_solutions_l120_120454

-- Defining the conditions for the equation
def isCondition (x : ℝ) : Prop := x ≠ 2 ∧ x ≠ 3

-- Defining the equation
def eqn (x : ℝ) : Prop := (3 * x^2 - 15 * x + 18) / (x^2 - 5 * x + 6) = x - 2

-- Defining the property that we need to prove
def property (x : ℝ) : Prop := eqn x ∧ isCondition x

-- Statement of the proof problem
theorem number_of_solutions : 
  ∃! x : ℝ, property x :=
sorry

end number_of_solutions_l120_120454


namespace find_k_l120_120515

open BigOperators

noncomputable
def hyperbola_property (k : ℝ) (x a b c : ℝ) : Prop :=
  k > 0 ∧
  (a / 2, b / 2) = (a / 2, k / a / 2) ∧ -- midpoint condition
  abs (a * b) / 2 = 3 ∧                -- area condition
  b = k / a                            -- point B on the hyperbola

theorem find_k (k : ℝ) (x a b c : ℝ) : hyperbola_property k x a b c → k = 2 :=
by
  sorry

end find_k_l120_120515


namespace tail_length_third_generation_l120_120281

theorem tail_length_third_generation (initial_length : ℕ) (growth_rate : ℕ) :
  initial_length = 16 ∧ growth_rate = 25 → 
  let sec_len := initial_length * (100 + growth_rate) / 100 in
  let third_len := sec_len * (100 + growth_rate) / 100 in
  third_len = 25 := by
  intros h
  sorry

end tail_length_third_generation_l120_120281


namespace find_n_l120_120223

/-- Given: 
1. The second term in the expansion of (x + a)^n is binom n 1 * x^(n-1) * a = 210.
2. The third term in the expansion of (x + a)^n is binom n 2 * x^(n-2) * a^2 = 840.
3. The fourth term in the expansion of (x + a)^n is binom n 3 * x^(n-3) * a^3 = 2520.
We are to prove that n = 10. -/
theorem find_n (x a : ℕ) (n : ℕ)
  (h1 : Nat.choose n 1 * x^(n-1) * a = 210)
  (h2 : Nat.choose n 2 * x^(n-2) * a^2 = 840)
  (h3 : Nat.choose n 3 * x^(n-3) * a^3 = 2520) : 
  n = 10 := by sorry

end find_n_l120_120223


namespace sum_of_four_numbers_in_ratio_is_correct_l120_120992

variable (A B C D : ℝ)
variable (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 4 ∧ C / D = 4 / 5)
variable (h_biggest : D = 672)

theorem sum_of_four_numbers_in_ratio_is_correct :
  A + B + C + D = 1881.6 :=
by
  sorry

end sum_of_four_numbers_in_ratio_is_correct_l120_120992


namespace problem_1_problem_2_problem_3_l120_120700

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := 8 * x^2 + 16 * x - k
noncomputable def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x
noncomputable def h (x : ℝ) (k : ℝ) : ℝ := g x - f x k

theorem problem_1 (k : ℝ) : (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x k ≤ g x) → 45 ≤ k := by
  sorry

theorem problem_2 (k : ℝ) : (∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ f x k ≤ g x) → -7 ≤ k := by
  sorry

theorem problem_3 (k : ℝ) : (∀ x1 x2 : ℝ, (-3 ≤ x1 ∧ x1 ≤ 3) ∧ (-3 ≤ x2 ∧ x2 ≤ 3) → f x1 k ≤ g x2) → 141 ≤ k := by
  sorry

end problem_1_problem_2_problem_3_l120_120700


namespace joe_total_cars_l120_120284

def initial_cars := 50
def multiplier := 3

theorem joe_total_cars : initial_cars + (multiplier * initial_cars) = 200 := by
  sorry

end joe_total_cars_l120_120284


namespace worker_savings_multiple_l120_120198

variable (P : ℝ)

theorem worker_savings_multiple (h1 : P > 0) (h2 : 0.4 * P + 0.6 * P = P) : 
  (12 * 0.4 * P) / (0.6 * P) = 8 :=
by
  sorry

end worker_savings_multiple_l120_120198


namespace angle_BEA_l120_120556

noncomputable def triangle_ABC (A B C E : Point) : Prop :=
  is_on_segment B C E ∧ 
  angle B A E = 20 ∧ 
  angle E A C = 40

theorem angle_BEA (A B C E : Point) (h : triangle_ABC A B C E) : 
  angle B E A = 80 := 
sorry

end angle_BEA_l120_120556


namespace max_plus_min_eq_four_l120_120564

theorem max_plus_min_eq_four {g : ℝ → ℝ} (h_odd_function : ∀ x, g (-x) = -g x)
  (M m : ℝ) (h_f : ∀ x, 2 + g x ≤ M) (h_f' : ∀ x, m ≤ 2 + g x) :
  M + m = 4 :=
by
  sorry

end max_plus_min_eq_four_l120_120564


namespace surface_area_of_rectangular_solid_is_334_l120_120812

theorem surface_area_of_rectangular_solid_is_334
  (l w h : ℕ)
  (h_l_prime : Prime l)
  (h_w_prime : Prime w)
  (h_h_prime : Prime h)
  (volume_eq_385 : l * w * h = 385) : 
  2 * (l * w + l * h + w * h) = 334 := 
sorry

end surface_area_of_rectangular_solid_is_334_l120_120812


namespace find_particular_number_l120_120302

theorem find_particular_number (x : ℤ) (h : x - 7 = 2) : x = 9 :=
by {
  -- The proof will be written here.
  sorry
}

end find_particular_number_l120_120302


namespace three_digit_integers_congruent_to_2_mod_4_l120_120714

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l120_120714


namespace jelly_bean_ratio_l120_120030

theorem jelly_bean_ratio
  (initial_jelly_beans : ℕ)
  (num_people : ℕ)
  (remaining_jelly_beans : ℕ)
  (amount_taken_by_each_of_last_four : ℕ)
  (total_taken_by_last_four : ℕ)
  (total_jelly_beans_taken : ℕ)
  (X : ℕ)
  (ratio : ℕ)
  (h0 : initial_jelly_beans = 8000)
  (h1 : num_people = 10)
  (h2 : remaining_jelly_beans = 1600)
  (h3 : amount_taken_by_each_of_last_four = 400)
  (h4 : total_taken_by_last_four = 4 * amount_taken_by_each_of_last_four)
  (h5 : total_jelly_beans_taken = initial_jelly_beans - remaining_jelly_beans)
  (h6 : X = total_jelly_beans_taken - total_taken_by_last_four)
  (h7 : ratio = X / total_taken_by_last_four)
  : ratio = 3 :=
by sorry

end jelly_bean_ratio_l120_120030


namespace no_integer_roots_of_quadratic_l120_120531

theorem no_integer_roots_of_quadratic
  (a b c : ℤ) (f : ℤ → ℤ)
  (h_def : ∀ x, f x = a * x * x + b * x + c)
  (h_a_nonzero : a ≠ 0)
  (h_f0_odd : Odd (f 0))
  (h_f1_odd : Odd (f 1)) :
  ∀ x : ℤ, f x ≠ 0 :=
by
  sorry

end no_integer_roots_of_quadratic_l120_120531


namespace factorial_computation_l120_120963

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l120_120963


namespace factorize_expression_l120_120527

variable (a : ℝ) (b : ℝ)

theorem factorize_expression : 2 * a - 8 * a * b^2 = 2 * a * (1 - 2 * b) * (1 + 2 * b) := by
  sorry

end factorize_expression_l120_120527


namespace alberto_spent_more_l120_120944

noncomputable def alberto_total_before_discount : ℝ := 2457 + 374 + 520
noncomputable def alberto_discount : ℝ := 0.05 * alberto_total_before_discount
noncomputable def alberto_total_after_discount : ℝ := alberto_total_before_discount - alberto_discount

noncomputable def samara_total_before_tax : ℝ := 25 + 467 + 79 + 150
noncomputable def samara_tax : ℝ := 0.07 * samara_total_before_tax
noncomputable def samara_total_after_tax : ℝ := samara_total_before_tax + samara_tax

noncomputable def amount_difference : ℝ := alberto_total_after_discount - samara_total_after_tax

theorem alberto_spent_more : amount_difference = 2411.98 :=
by
  sorry

end alberto_spent_more_l120_120944


namespace find_g_720_l120_120744

noncomputable def g (n : ℕ) : ℕ := sorry

axiom g_multiplicative : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g_8 : g 8 = 12
axiom g_12 : g 12 = 16

theorem find_g_720 : g 720 = 44 := by sorry

end find_g_720_l120_120744


namespace additional_grazed_area_correct_l120_120937

noncomputable def additional_grazed_area (r1 r2 : ℝ) : ℝ :=
  π * r2^2 - π * r1^2

theorem additional_grazed_area_correct :
  additional_grazed_area 10 23 = 429 * real.pi :=
by
  unfold additional_grazed_area
  norm_num
  sorry

end additional_grazed_area_correct_l120_120937


namespace valid_starting_lineups_correct_l120_120011

-- Define the parameters from the problem
def volleyball_team : Finset ℕ := Finset.range 18
def quadruplets : Finset ℕ := {0, 1, 2, 3}

-- Define the main computation: total lineups excluding those where all quadruplets are chosen
noncomputable def valid_starting_lineups : ℕ :=
  (volleyball_team.card.choose 7) - ((volleyball_team \ quadruplets).card.choose 3)

-- The theorem states that the number of valid starting lineups is 31460
theorem valid_starting_lineups_correct : valid_starting_lineups = 31460 := by
  sorry

end valid_starting_lineups_correct_l120_120011


namespace possible_value_of_2n_plus_m_l120_120273

variable (n m : ℤ)

theorem possible_value_of_2n_plus_m : (3 * n - m < 5) → (n + m > 26) → (3 * m - 2 * n < 46) → (2 * n + m = 36) :=
by
  sorry

end possible_value_of_2n_plus_m_l120_120273


namespace gretchen_work_hours_l120_120402

noncomputable def walking_ratio (walking: ℤ) (sitting: ℤ) : Prop :=
  walking * 90 = sitting * 10

theorem gretchen_work_hours (walking_time: ℤ) (h: ℤ) (condition1: walking_ratio 40 (60 * h)) :
  h = 6 :=
by sorry

end gretchen_work_hours_l120_120402


namespace min_value_quadratic_function_l120_120437

def f (a b c x : ℝ) : ℝ := a * (x - b) * (x - c)

theorem min_value_quadratic_function :
  ∃ a b c : ℝ, 
    (1 ≤ a ∧ a < 10) ∧
    (1 ≤ b ∧ b < 10) ∧
    (1 ≤ c ∧ c < 10) ∧
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (∀ x : ℝ, f a b c x ≥ -128) :=
sorry

end min_value_quadratic_function_l120_120437


namespace mean_score_of_all_students_l120_120756

-- Define the conditions as given in the problem
variables (M A : ℝ) (m a : ℝ)
  (hM : M = 90)
  (hA : A = 75)
  (hRatio : m / a = 2 / 5)

-- State the theorem which proves that the mean score of all students is 79
theorem mean_score_of_all_students (hM : M = 90) (hA : A = 75) (hRatio : m / a = 2 / 5) : 
  (36 * a + 75 * a) / ((2 / 5) * a + a) = 79 := 
by
  sorry -- Proof is omitted

end mean_score_of_all_students_l120_120756


namespace part1_part2_l120_120399

noncomputable def f (m x : ℝ) : ℝ := exp (x - m) - x * log x - (m - 1) * x
noncomputable def f' (m x : ℝ) : ℝ := deriv (λ x, exp (x - m) - x * log x - (m - 1) * x) x

theorem part1 (x : ℝ) (hx1 : x > 0) : f' 1 x ≥ 0 := sorry

theorem part2 (m : ℝ) (hx2 : ∃ x1 x2, x1 < x2 ∧ f' m x1 = 0 ∧ f' m x2 = 0) : m > 1 := sorry

end part1_part2_l120_120399


namespace min_value_of_a_k_l120_120103

-- Define the conditions for our proof in Lean

-- a_n is a positive arithmetic sequence
def is_positive_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ d, ∀ m, a (m + 1) = a m + d

-- Given inequality condition for the sequence
def inequality_condition (a : ℕ → ℝ) (k : ℕ) : Prop :=
  k ≥ 2 ∧ (1 / a 1 + 4 / a (2 * k - 1) ≤ 1)

-- Prove the minimum value of a_k
theorem min_value_of_a_k (a : ℕ → ℝ) (k : ℕ) (h_arith : is_positive_arithmetic_seq a) (h_ineq : inequality_condition a k) :
  a k = 9 / 2 :=
sorry

end min_value_of_a_k_l120_120103


namespace third_generation_tail_length_l120_120278

theorem third_generation_tail_length (tail_length : ℕ → ℕ) (h0 : tail_length 0 = 16)
    (h_next : ∀ n, tail_length (n + 1) = tail_length n + (25 * tail_length n) / 100) :
    tail_length 2 = 25 :=
by
  sorry

end third_generation_tail_length_l120_120278


namespace solve_problem_l120_120276

-- Declare the variables n and m
variables (n m : ℤ)

-- State the theorem with given conditions and prove that 2n + m = 36
theorem solve_problem
  (h1 : 3 * n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3 * m - 2 * n < 46) :
  2 * n + m = 36 :=
sorry

end solve_problem_l120_120276


namespace condition_iff_absolute_value_l120_120416

theorem condition_iff_absolute_value (a b : ℝ) : (a > b) ↔ (a * |a| > b * |b|) :=
sorry

end condition_iff_absolute_value_l120_120416


namespace calculate_average_fish_caught_l120_120860

-- Definitions based on conditions
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def Toph_fish : ℕ := 12

-- Total fish and average calculation
def total_fish : ℕ := Aang_fish + Sokka_fish + Toph_fish
def number_of_people : ℕ := 3
def average_fish_per_person : ℕ := total_fish / number_of_people

-- Theorem to prove
theorem calculate_average_fish_caught : average_fish_per_person = 8 := 
by 
  -- Proof steps are skipped with 'sorry', but the statement is set up correctly
  sorry

end calculate_average_fish_caught_l120_120860


namespace solution_set_empty_range_a_l120_120118

theorem solution_set_empty_range_a (a : ℝ) :
  (∀ x : ℝ, ¬((a - 1) * x^2 + 2 * (a - 1) * x - 4 ≥ 0)) ↔ -3 < a ∧ a ≤ 1 :=
by
  sorry

end solution_set_empty_range_a_l120_120118


namespace function_zeros_range_l120_120105

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then (1 / 2)^x + 2 / x else x * Real.log x - a

theorem function_zeros_range (a : ℝ) :
  (∀ x : ℝ, f x a = 0 → x < 0) ∧ (∀ x : ℝ, f x a = 0 → x > 0 → (a > -1 / Real.exp 1 ∧ a < 0)) ↔
  (a > -1 / Real.exp 1 ∧ a < 0) :=
sorry

end function_zeros_range_l120_120105


namespace profit_percent_300_l120_120628

theorem profit_percent_300 (SP : ℝ) (CP : ℝ) (h : CP = 0.25 * SP) : ((SP - CP) / CP) * 100 = 300 :=
by
  sorry

end profit_percent_300_l120_120628


namespace find_f_x_l120_120226

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem find_f_x (x : ℝ) : (f (x+1)) = x^2 - 3*x + 2 :=
by
  sorry

end find_f_x_l120_120226


namespace min_sum_of_factors_l120_120161

theorem min_sum_of_factors (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 3960) : 
  a + b + c = 72 :=
sorry

end min_sum_of_factors_l120_120161


namespace white_balls_count_l120_120349

-- Definitions for the conditions
variable (x y : ℕ) 

-- Lean statement representing the problem
theorem white_balls_count : 
  x < y ∧ y < 2 * x ∧ 2 * x + 3 * y = 60 → x = 9 := 
sorry

end white_balls_count_l120_120349


namespace emani_money_l120_120813

def emani_has_30_more (E H : ℝ) : Prop := E = H + 30
def equal_share (E H : ℝ) : Prop := (E + H) / 2 = 135

theorem emani_money (E H : ℝ) (h1: emani_has_30_more E H) (h2: equal_share E H) : E = 150 :=
by
  sorry

end emani_money_l120_120813


namespace minimal_solution_x_eq_neg_two_is_solution_smallest_solution_l120_120829

theorem minimal_solution (x : ℝ) (h : x * |x| = 3 * x + 2) : -2 ≤ x :=
begin
  sorry
end

theorem x_eq_neg_two_is_solution : ( -2 : ℝ ) * |-2| = 3 * -2 + 2 :=
begin
  norm_num,
end

/-- The smallest value of x satisfying x|x| = 3x + 2 is -2 -/
theorem smallest_solution : ∃ x : ℝ, x * |x| = 3 * x + 2 ∧ ∀ y : ℝ, y * |y| = 3 * y + 2 → y ≥ x :=
begin
  use -2,
  split,
  { norm_num },
  { intro y,
    sorry }
end

end minimal_solution_x_eq_neg_two_is_solution_smallest_solution_l120_120829


namespace roots_real_roots_equal_l120_120836

noncomputable def discriminant (a : ℝ) : ℝ :=
  let b := 4 * a
  let c := 2 * a^2 - 1 + 3 * a
  b^2 - 4 * 1 * c

theorem roots_real (a : ℝ) : discriminant a ≥ 0 ↔ a ≤ 1/2 ∨ a ≥ 1 := sorry

theorem roots_equal (a : ℝ) : discriminant a = 0 ↔ a = 1 ∨ a = 1/2 := sorry

end roots_real_roots_equal_l120_120836


namespace part_a_l120_120927

theorem part_a (n : ℕ) : ((x^2 + x + 1) ∣ (x^(2 * n) + x^n + 1)) ↔ (n % 3 = 0) := sorry

end part_a_l120_120927


namespace fraction_ordering_l120_120171

theorem fraction_ordering :
  (6:ℚ)/29 < (8:ℚ)/25 ∧ (8:ℚ)/25 < (10:ℚ)/31 :=
by
  sorry

end fraction_ordering_l120_120171


namespace square_vectors_l120_120094

theorem square_vectors (AB CD AD : ℝ × ℝ)
  (side_length: ℝ)
  (M N : ℝ × ℝ)
  (x y: ℝ)
  (MN : ℝ × ℝ):
  side_length = 2 →
  M = ((AB.1 + CD.1) / 2, (AB.2 + CD.2) / 2) →
  N = ((CD.1 + AD.1) / 2, (CD.2 + AD.2) / 2) →
  MN = (x * AB.1 + y * AD.1, x * AB.2 + y * AD.2) →
  (x = -1/2) ∧ (y = 1/2) →
  (x * y = -1/4) ∧ ((N.1 - M.1) * AD.1 + (N.2 - M.2) * AD.2 - (N.1 - M.1) * AB.1 - (N.2 - M.2) * AB.2 = -1) :=
by
  intros side_length_cond M_cond N_cond MN_cond xy_cond
  sorry

end square_vectors_l120_120094


namespace roots_of_quadratic_discriminant_positive_l120_120611

theorem roots_of_quadratic_discriminant_positive {a b c : ℝ} (h : b^2 - 4 * a * c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by {
  sorry
}

end roots_of_quadratic_discriminant_positive_l120_120611


namespace slope_of_line_l120_120671

theorem slope_of_line : ∀ (x y : ℝ), 4 * x - 7 * y = 28 → y = (4/7) * x - 4 :=
by
  sorry

end slope_of_line_l120_120671


namespace inequality_always_negative_l120_120262

theorem inequality_always_negative (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3 / 4 < 0) ↔ (-3 < k ∧ k ≤ 0) :=
by
  -- Proof omitted
  sorry

end inequality_always_negative_l120_120262


namespace cost_of_dozen_pens_l120_120043

-- Define the costs and conditions as given in the problem.
def cost_of_pen (x : ℝ) : ℝ := 5 * x
def cost_of_pencil (x : ℝ) : ℝ := x

-- The given conditions transformed into Lean definitions.
def condition1 (x : ℝ) : Prop := 3 * cost_of_pen x + 5 * cost_of_pencil x = 100
def condition2 (x : ℝ) : Prop := cost_of_pen x / cost_of_pencil x = 5

-- Prove that the cost of one dozen pens is Rs. 300.
theorem cost_of_dozen_pens : ∃ x : ℝ, condition1 x ∧ condition2 x ∧ 12 * cost_of_pen x = 300 := by
  sorry

end cost_of_dozen_pens_l120_120043


namespace derivative_of_cos_over_x_l120_120216

open Real

noncomputable def f (x : ℝ) : ℝ := (cos x) / x

theorem derivative_of_cos_over_x (x : ℝ) (h : x ≠ 0) : 
  deriv f x = - (x * sin x + cos x) / (x^2) :=
sorry

end derivative_of_cos_over_x_l120_120216


namespace length_of_faster_train_is_correct_l120_120324

def speed_faster_train := 54 -- kmph
def speed_slower_train := 36 -- kmph
def crossing_time := 27 -- seconds

def kmph_to_mps (s : ℕ) : ℕ :=
  s * 1000 / 3600

def relative_speed_faster_train := kmph_to_mps (speed_faster_train - speed_slower_train)

def length_faster_train := relative_speed_faster_train * crossing_time

theorem length_of_faster_train_is_correct : length_faster_train = 135 := 
  by
  sorry

end length_of_faster_train_is_correct_l120_120324


namespace limit_fraction_l120_120299

theorem limit_fraction :
  ∀ ε > 0, ∃ (N : ℕ), ∀ n ≥ N, |((4 * n - 1) / (2 * n + 1) : ℚ) - 2| < ε := 
  by sorry

end limit_fraction_l120_120299


namespace find_set_M_l120_120292

variable (U : Set ℕ) (M : Set ℕ)

def isUniversalSet : Prop := U = {1, 2, 3, 4, 5, 6}
def isComplement : Prop := U \ M = {1, 2, 4}

theorem find_set_M (hU : isUniversalSet U) (hC : isComplement U M) : M = {3, 5, 6} :=
  sorry

end find_set_M_l120_120292


namespace farmer_rectangle_partition_l120_120055

theorem farmer_rectangle_partition (m : ℝ) :
  (3 * m + 8) * (m - 3) = 70 ↔ m = (1 + Real.sqrt 1129) / 6 := 
begin
  sorry,
end

end farmer_rectangle_partition_l120_120055


namespace greater_number_l120_120907

theorem greater_number (x y : ℕ) (h_sum : x + y = 50) (h_diff : x - y = 16) : x = 33 :=
by
  sorry

end greater_number_l120_120907


namespace factorial_division_identity_l120_120961

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l120_120961


namespace total_students_is_88_l120_120318

def orchestra_students : Nat := 20
def band_students : Nat := 2 * orchestra_students
def choir_boys : Nat := 12
def choir_girls : Nat := 16
def choir_students : Nat := choir_boys + choir_girls

def total_students : Nat := orchestra_students + band_students + choir_students

theorem total_students_is_88 : total_students = 88 := by
  sorry

end total_students_is_88_l120_120318


namespace total_arrangements_correct_adjacent_males_correct_descending_heights_correct_l120_120632

-- Total number of different arrangements of 3 male students and 2 female students.
def total_arrangements (males females : ℕ) : ℕ :=
  (males + females).factorial

-- Number of arrangements where exactly two male students are adjacent.
def adjacent_males (males females : ℕ) : ℕ :=
  if males = 3 ∧ females = 2 then 72 else 0

-- Number of arrangements where male students of different heights are arranged from tallest to shortest.
def descending_heights (heights : Nat → ℕ) (males females : ℕ) : ℕ :=
  if males = 3 ∧ females = 2 then 20 else 0

-- Theorem statements corresponding to the questions.
theorem total_arrangements_correct : total_arrangements 3 2 = 120 := sorry

theorem adjacent_males_correct : adjacent_males 3 2 = 72 := sorry

theorem descending_heights_correct (heights : Nat → ℕ) : descending_heights heights 3 2 = 20 := sorry

end total_arrangements_correct_adjacent_males_correct_descending_heights_correct_l120_120632


namespace max_value_of_expression_l120_120219

theorem max_value_of_expression :
  ∃ (x y z : ℝ), 
    let expr := (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) in 
    expr = 4.5 ∧
    ∀ (x y z : ℝ), 
      (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 :=
begin
  sorry,
end

end max_value_of_expression_l120_120219


namespace first_term_to_common_difference_ratio_l120_120918

theorem first_term_to_common_difference_ratio (a d : ℝ) 
  (h : (14 / 2) * (2 * a + 13 * d) = 3 * (7 / 2) * (2 * a + 6 * d)) :
  a / d = 4 :=
by
  sorry

end first_term_to_common_difference_ratio_l120_120918


namespace longest_side_enclosure_l120_120364

variable (l w : ℝ)

theorem longest_side_enclosure (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 1920) : max l w = 101 :=
sorry

end longest_side_enclosure_l120_120364


namespace e_exp_ax1_ax2_gt_two_l120_120849

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) - a * (x + 2)

theorem e_exp_ax1_ax2_gt_two {a x1 x2 : ℝ} (h : a ≠ 0) (h1 : f a x1 = 0) (h2 : f a x2 = 0) (hx : x1 < x2) : 
  Real.exp (a * x1) + Real.exp (a * x2) > 2 :=
sorry

end e_exp_ax1_ax2_gt_two_l120_120849


namespace sequence_remainder_zero_l120_120037

theorem sequence_remainder_zero :
  let a := 3
  let d := 8
  let n := 32
  let aₙ := a + (n - 1) * d
  let Sₙ := n * (a + aₙ) / 2
  aₙ = 251 → Sₙ % 8 = 0 :=
by
  intros
  sorry

end sequence_remainder_zero_l120_120037


namespace binary_sum_eq_669_l120_120517

def binary111111111 : ℕ := 511
def binary1111111 : ℕ := 127
def binary11111 : ℕ := 31

theorem binary_sum_eq_669 :
  binary111111111 + binary1111111 + binary11111 = 669 :=
by
  sorry

end binary_sum_eq_669_l120_120517


namespace fib_inequality_l120_120332

def Fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => Fib n + Fib (n + 1)

theorem fib_inequality {n : ℕ} (h : 2 ≤ n) : Fib (n + 5) > 10 * Fib n :=
  sorry

end fib_inequality_l120_120332


namespace distance_after_time_l120_120801

noncomputable def Adam_speed := 12 -- speed in mph
noncomputable def Simon_speed := 6 -- speed in mph
noncomputable def time_when_100_miles_apart := 100 / 15 -- hours

theorem distance_after_time (x : ℝ) : 
  (Adam_speed * x)^2 + (Simon_speed * x)^2 = 100^2 ->
  x = time_when_100_miles_apart := 
by
  sorry

end distance_after_time_l120_120801


namespace John_profit_is_correct_l120_120739

-- Definitions of conditions as necessary in Lean
variable (initial_puppies : ℕ) (given_away_puppies : ℕ) (kept_puppy : ℕ) (price_per_puppy : ℤ) (payment_to_stud_owner : ℤ)

-- Specific values from the problem
def John_initial_puppies := 8
def John_given_away_puppies := 4
def John_kept_puppy := 1
def John_price_per_puppy := 600
def John_payment_to_stud_owner := 300

-- Calculate the number of puppies left to sell
def John_remaining_puppies := John_initial_puppies - John_given_away_puppies - John_kept_puppy

-- Calculate total earnings from selling puppies
def John_earnings := John_remaining_puppies * John_price_per_puppy

-- Calculate the profit by subtracting payment to the stud owner from earnings
def John_profit := John_earnings - John_payment_to_stud_owner

-- Statement to prove
theorem John_profit_is_correct : 
  John_profit = 1500 := 
by 
  -- The proof will be here but we use sorry to skip it as requested.
  sorry

-- This ensures the definitions match the given problem conditions
#eval (John_initial_puppies, John_given_away_puppies, John_kept_puppy, John_price_per_puppy, John_payment_to_stud_owner)

end John_profit_is_correct_l120_120739


namespace goldfinch_percentage_l120_120753

noncomputable def percentage_of_goldfinches 
  (goldfinches : ℕ) (sparrows : ℕ) (grackles : ℕ) : ℚ :=
  (goldfinches : ℚ) / (goldfinches + sparrows + grackles) * 100

theorem goldfinch_percentage (goldfinches sparrows grackles : ℕ)
  (h_goldfinches : goldfinches = 6)
  (h_sparrows : sparrows = 9)
  (h_grackles : grackles = 5) :
  percentage_of_goldfinches goldfinches sparrows grackles = 30 :=
by
  rw [h_goldfinches, h_sparrows, h_grackles]
  show percentage_of_goldfinches 6 9 5 = 30
  sorry

end goldfinch_percentage_l120_120753


namespace factorial_sum_division_l120_120966

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l120_120966


namespace zero_of_function_l120_120909

theorem zero_of_function : ∃ x : ℝ, (x + 1)^2 = 0 :=
by
  use -1
  sorry

end zero_of_function_l120_120909


namespace digit_solve_l120_120403

theorem digit_solve : ∀ (D : ℕ), D < 10 → (D * 9 + 6 = D * 10 + 3) → D = 3 :=
by
  intros D hD h
  sorry

end digit_solve_l120_120403


namespace total_students_l120_120317

theorem total_students (orchestra band choir_boys choir_girls : ℕ)
  (h_orchestra : orchestra = 20)
  (h_band : band = 2 * orchestra)
  (h_choir_boys : choir_boys = 12)
  (h_choir_girls : choir_girls = 16)
  (h_disjoint : ∀ x, x ∈ orchestra ∨ x ∈ band ∨ x ∈ (choir_boys + choir_girls) → 
                    (x ∈ orchestra → x ∉ band ∧ x ∉ (choir_boys + choir_girls)) ∧
                    (x ∈ band → x ∉ orchestra ∧ x ∉ (choir_boys + choir_girls)) ∧
                    (x ∈ (choir_boys + choir_girls) → x ∉ orchestra ∧ x ∉ band)) :
  orchestra + band + (choir_boys + choir_girls) = 88 := 
by
  rw [h_orchestra, h_band, h_choir_boys, h_choir_girls]
  show 20 + 2 * 20 + (12 + 16) = 88 from
  calc
    20 + 2 * 20 + (12 + 16) = 20 + 40 + 28 : by rfl
    ...                      = 88            : by rfl

end total_students_l120_120317


namespace dave_deleted_apps_l120_120971

theorem dave_deleted_apps :
  ∃ d : ℕ, d = 150 - 65 :=
sorry

end dave_deleted_apps_l120_120971


namespace solve_problem_l120_120275

-- Declare the variables n and m
variables (n m : ℤ)

-- State the theorem with given conditions and prove that 2n + m = 36
theorem solve_problem
  (h1 : 3 * n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3 * m - 2 * n < 46) :
  2 * n + m = 36 :=
sorry

end solve_problem_l120_120275


namespace sin_690_l120_120806

-- Defining the known conditions as hypotheses:
axiom sin_periodic (x : ℝ) : Real.sin (x + 360) = Real.sin x
axiom sin_odd (x : ℝ) : Real.sin (-x) = - Real.sin x
axiom sin_thirty : Real.sin 30 = 1 / 2

theorem sin_690 : Real.sin 690 = -1 / 2 :=
by
  -- Proof would go here, but it is skipped with sorry.
  sorry

end sin_690_l120_120806


namespace sum_of_parallelogram_sides_l120_120549

-- Definitions of the given conditions.
def length_one_side : ℕ := 10
def length_other_side : ℕ := 7

-- Theorem stating the sum of the lengths of the four sides of the parallelogram.
theorem sum_of_parallelogram_sides : 
    (length_one_side + length_one_side + length_other_side + length_other_side) = 34 :=
by
    sorry

end sum_of_parallelogram_sides_l120_120549


namespace willie_cream_l120_120083

theorem willie_cream : ∀ (total_cream needed_cream: ℕ), total_cream = 300 → needed_cream = 149 → (total_cream - needed_cream) = 151 :=
by
  intros total_cream needed_cream h1 h2
  sorry

end willie_cream_l120_120083


namespace aaron_brothers_l120_120200

theorem aaron_brothers (A : ℕ) (h1 : 6 = 2 * A - 2) : A = 4 :=
by
  sorry

end aaron_brothers_l120_120200


namespace three_digit_integers_congruent_to_2_mod_4_l120_120716

theorem three_digit_integers_congruent_to_2_mod_4 : 
  let count := (249 - 25 + 1) in
  count = 225 :=
by
  let k_min := 25
  let k_max := 249
  have h_count : count = (k_max - k_min + 1) := rfl
  rw h_count
  norm_num

end three_digit_integers_congruent_to_2_mod_4_l120_120716


namespace not_integer_fraction_l120_120740

theorem not_integer_fraction (a b : ℤ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (hrelprime : Nat.gcd a.natAbs b.natAbs = 1) : 
  ¬(∃ (k : ℤ), 2 * a * (a^2 + b^2) = k * (a^2 - b^2)) :=
  sorry

end not_integer_fraction_l120_120740


namespace factorial_division_sum_l120_120970

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l120_120970


namespace largest_power_of_2_that_divides_n_l120_120811

def n : ℕ := 15^4 - 9^4

theorem largest_power_of_2_that_divides_n :
  ∃ k : ℕ, 2^k ∣ n ∧ ¬ (2^(k+1) ∣ n) ∧ k = 5 := sorry

end largest_power_of_2_that_divides_n_l120_120811


namespace decreasing_interval_l120_120448

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → deriv f x < 0 :=
by sorry

end decreasing_interval_l120_120448


namespace inequality_solution_set_l120_120904

theorem inequality_solution_set :
  { x : ℝ | -3 < x ∧ x < 2 } = { x : ℝ | abs (x - 1) + abs (x + 2) < 5 } :=
by
  sorry

end inequality_solution_set_l120_120904


namespace simplify_expression_l120_120893

variable (a b : ℝ)

theorem simplify_expression :
  (-2 * a^2 * b^3) * (-a * b^2)^2 + (- (1 / 2) * a^2 * b^3)^2 * 4 * b = -a^4 * b^7 := 
by 
  sorry

end simplify_expression_l120_120893


namespace prob_mc_tf_correct_prob_at_least_one_mc_correct_l120_120429

-- Define the total number of questions and their types
def total_questions : ℕ := 5
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2
def total_outcomes : ℕ := total_questions * (total_questions - 1)

-- Probability calculation for one drawing a multiple-choice and the other drawing a true/false question
def prob_mc_tf : ℚ := (multiple_choice_questions * true_false_questions + true_false_questions * multiple_choice_questions) / total_outcomes

-- Probability calculation for at least one drawing a multiple-choice question
def prob_at_least_one_mc : ℚ := 1 - (true_false_questions * (true_false_questions - 1)) / total_outcomes

theorem prob_mc_tf_correct : prob_mc_tf = 3/5 := by
  sorry

theorem prob_at_least_one_mc_correct : prob_at_least_one_mc = 9/10 := by
  sorry

end prob_mc_tf_correct_prob_at_least_one_mc_correct_l120_120429


namespace max_expr_value_l120_120378

theorem max_expr_value (a b c d : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) (hc : 0 ≤ c) (hc1 : c ≤ 1) (hd : 0 ≤ d) (hd1 : d ≤ 1) : 
  a + b + c + d - a * b - b * c - c * d - d * a ≤ 2 :=
sorry

end max_expr_value_l120_120378


namespace focus_of_parabola_y_eq_8x2_l120_120447

open Real

noncomputable def parabola_focus (a p : ℝ) : ℝ × ℝ :=
  (0, 1 / (4 * p))

theorem focus_of_parabola_y_eq_8x2 :
  parabola_focus 8 (1 / 16) = (0, 1 / 32) :=
by
  sorry

end focus_of_parabola_y_eq_8x2_l120_120447


namespace sum_of_two_cosines_not_equal_to_third_l120_120592

theorem sum_of_two_cosines_not_equal_to_third
  (α β γ : ℝ)
  (h_pos_α : 0 < α)
  (h_pos_β : 0 < β)
  (h_pos_γ : 0 < γ)
  (h_sum_angles : α + β + γ = π / 2) :
  ¬ (cos α + cos β = cos γ) :=
sorry

end sum_of_two_cosines_not_equal_to_third_l120_120592


namespace simplify_expr1_simplify_expr2_l120_120894

-- Define the variables a and b
variables (a b : ℝ)

-- First problem: simplify 2a^2 - 3a^3 + 5a + 2a^3 - a^2 to a^2 - a^3 + 5a
theorem simplify_expr1 : 2*a^2 - 3*a^3 + 5*a + 2*a^3 - a^2 = a^2 - a^3 + 5*a :=
  by sorry

-- Second problem: simplify (2 / 3) (2 * a - b) + 2 (b - 2 * a) - 3 (2 * a - b) - (4 / 3) (b - 2 * a) to -6 * a + 3 * b
theorem simplify_expr2 : 
  (2 / 3) * (2 * a - b) + 2 * (b - 2 * a) - 3 * (2 * a - b) - (4 / 3) * (b - 2 * a) = -6 * a + 3 * b :=
  by sorry

end simplify_expr1_simplify_expr2_l120_120894


namespace probability_all_boys_probability_one_girl_probability_at_least_one_girl_l120_120578

-- Assumptions and Definitions
def total_outcomes := Nat.choose 5 3
def all_boys_outcomes := Nat.choose 3 3
def one_girl_outcomes := Nat.choose 3 2 * Nat.choose 2 1
def at_least_one_girl_outcomes := one_girl_outcomes + Nat.choose 3 1 * Nat.choose 2 2

-- The probability calculation proofs
theorem probability_all_boys : all_boys_outcomes / total_outcomes = 1 / 10 := by 
  sorry

theorem probability_one_girl : one_girl_outcomes / total_outcomes = 6 / 10 := by 
  sorry

theorem probability_at_least_one_girl : at_least_one_girl_outcomes / total_outcomes = 9 / 10 := by 
  sorry

end probability_all_boys_probability_one_girl_probability_at_least_one_girl_l120_120578


namespace find_f_7_l120_120097

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom function_period : ∀ x : ℝ, f (x + 2) = -f x
axiom function_value_range : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7 : f 7 = -1 := by
  sorry

end find_f_7_l120_120097


namespace four_distinct_numbers_are_prime_l120_120980

-- Lean 4 statement proving the conditions
theorem four_distinct_numbers_are_prime : 
  ∃ (a b c d : ℕ), 
    a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 5 ∧ 
    (Prime (a * b + c * d)) ∧ 
    (Prime (a * c + b * d)) ∧ 
    (Prime (a * d + b * c)) := 
sorry

end four_distinct_numbers_are_prime_l120_120980


namespace solve_expression_l120_120080

theorem solve_expression (a b c : ℝ) (ha : a^3 - 2020*a^2 + 1010 = 0) (hb : b^3 - 2020*b^2 + 1010 = 0) (hc : c^3 - 2020*c^2 + 1010 = 0) (habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
    (1 / (a * b) + 1 / (b * c) + 1 / (a * c) = -2) := 
sorry

end solve_expression_l120_120080


namespace infinitely_many_good_numbers_seven_does_not_divide_good_number_l120_120024

-- Define what it means for a number to be good
def is_good_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a + b = n ∧ (a * b) ∣ (n^2 + n + 1)

-- Part (a): Show that there are infinitely many good numbers
theorem infinitely_many_good_numbers : ∃ (f : ℕ → ℕ), ∀ n, is_good_number (f n) :=
sorry

-- Part (b): Show that if n is a good number, then 7 does not divide n
theorem seven_does_not_divide_good_number (n : ℕ) (h : is_good_number n) : ¬ (7 ∣ n) :=
sorry

end infinitely_many_good_numbers_seven_does_not_divide_good_number_l120_120024


namespace ninth_graders_only_math_l120_120590

theorem ninth_graders_only_math 
  (total_students : ℕ)
  (math_students : ℕ)
  (foreign_language_students : ℕ)
  (science_only_students : ℕ)
  (math_and_foreign_language_no_science : ℕ)
  (h1 : total_students = 120)
  (h2 : math_students = 85)
  (h3 : foreign_language_students = 75)
  (h4 : science_only_students = 20)
  (h5 : math_and_foreign_language_no_science = 40) :
  math_students - math_and_foreign_language_no_science = 45 :=
by 
  sorry

end ninth_graders_only_math_l120_120590


namespace initial_milk_quantity_l120_120062

theorem initial_milk_quantity (A B C D : ℝ) (hA : A > 0)
  (hB : B = 0.55 * A)
  (hC : C = 1.125 * A)
  (hD : D = 0.8 * A)
  (hTransferBC : B + 150 = C - 150 + 100)
  (hTransferDC : C - 50 = D - 100)
  (hEqual : B + 150 = D - 100) : 
  A = 1000 :=
by sorry

end initial_milk_quantity_l120_120062


namespace train_distance_proof_l120_120912

-- Definitions
def speed_train1 : ℕ := 40
def speed_train2 : ℕ := 48
def time_hours : ℕ := 8
def initial_distance : ℕ := 892

-- Function to calculate distance after given time
def distance (speed time : ℕ) : ℕ := speed * time

-- Increased/Decreased distance after time
def distance_diff : ℕ := distance speed_train2 time_hours - distance speed_train1 time_hours

-- Final distances
def final_distance_same_direction : ℕ := initial_distance + distance_diff
def final_distance_opposite_direction : ℕ := initial_distance - distance_diff

-- Proof statement
theorem train_distance_proof :
  final_distance_same_direction = 956 ∧ final_distance_opposite_direction = 828 :=
by
  -- The proof is omitted here
  sorry

end train_distance_proof_l120_120912


namespace factorial_sum_division_l120_120965

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l120_120965


namespace factorial_division_sum_l120_120969

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l120_120969


namespace smaller_number_is_270_l120_120310

theorem smaller_number_is_270 (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : S = 270 :=
sorry

end smaller_number_is_270_l120_120310


namespace parallelogram_coordinate_sum_l120_120596

theorem parallelogram_coordinate_sum:
  (A B D : ℝ × ℝ) -- Define the vertices of the parallelogram
  (hA : A = (-1, 1))
  (hB : B = (3, 5))
  (hD : D = (11, -3))
  (area : ℝ)
  (harea : area = 48) :
  ∃ (C : ℝ × ℝ), 
    let x := C.1 in let y := C.2 in x + y = 0 :=
by
  sorry

end parallelogram_coordinate_sum_l120_120596


namespace Georgie_prank_l120_120185

theorem Georgie_prank (w : ℕ) (condition1 : w = 8) : 
  ∃ (ways : ℕ), ways = 336 := 
by
  sorry

end Georgie_prank_l120_120185


namespace sum_of_cubes_is_24680_l120_120461

noncomputable def jake_age := 10
noncomputable def amy_age := 12
noncomputable def ryan_age := 28

theorem sum_of_cubes_is_24680 (j a r : ℕ) (h1 : 2 * j + 3 * a = 4 * r)
  (h2 : j^3 + a^3 = 1 / 2 * r^3) (h3 : j + a + r = 50) : j^3 + a^3 + r^3 = 24680 :=
by
  sorry

end sum_of_cubes_is_24680_l120_120461


namespace negation_proposition_equivalence_l120_120901

theorem negation_proposition_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2 * x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) :=
by
  sorry

end negation_proposition_equivalence_l120_120901


namespace gcd_lcm_product_correct_l120_120221

noncomputable def gcd_lcm_product : ℕ :=
  let a := 90
  let b := 135
  gcd a b * lcm a b

theorem gcd_lcm_product_correct : gcd_lcm_product = 12150 :=
  by
  sorry

end gcd_lcm_product_correct_l120_120221


namespace fifth_stair_area_and_perimeter_stair_for_area_78_stair_for_perimeter_100_l120_120032

-- Conditions
def square_side : ℕ := 1
def area_per_square : ℕ := square_side * square_side
def area_of_stair (n : ℕ) : ℕ := (n * (n + 1)) / 2
def perimeter_of_stair (n : ℕ) : ℕ := 4 * n

-- Part (a)
theorem fifth_stair_area_and_perimeter :
  area_of_stair 5 = 15 ∧ perimeter_of_stair 5 = 20 := by
  sorry

-- Part (b)
theorem stair_for_area_78 :
  ∃ n, area_of_stair n = 78 ∧ n = 12 := by
  sorry

-- Part (c)
theorem stair_for_perimeter_100 :
  ∃ n, perimeter_of_stair n = 100 ∧ n = 25 := by
  sorry

end fifth_stair_area_and_perimeter_stair_for_area_78_stair_for_perimeter_100_l120_120032


namespace solution_for_m_exactly_one_solution_l120_120975

theorem solution_for_m_exactly_one_solution (m : ℚ) : 
  (∀ x : ℚ, (x - 3) / (m * x + 4) = 2 * x → 
            (2 * m * x^2 + 7 * x + 3 = 0)) →
  (49 - 24 * m = 0) → 
  m = 49 / 24 :=
by
  intro h1 h2
  sorry

end solution_for_m_exactly_one_solution_l120_120975


namespace simplify_sum_of_squares_roots_l120_120807

theorem simplify_sum_of_squares_roots :
  Real.sqrt 12 + Real.sqrt 27 + Real.sqrt 48 = 9 * Real.sqrt 3 :=
by
  sorry

end simplify_sum_of_squares_roots_l120_120807


namespace pizza_problem_l120_120931

theorem pizza_problem (diameter : ℝ) (sectors : ℕ) (h1 : diameter = 18) (h2 : sectors = 4) : 
  let R := diameter / 2 
  let θ := (2 * Real.pi / sectors : ℝ)
  let m := 2 * R * Real.sin (θ / 2) 
  (m^2 = 162) := by
  sorry

end pizza_problem_l120_120931


namespace largest_number_is_27_l120_120334

-- Define the condition as a predicate
def three_consecutive_multiples_sum_to (k : ℕ) (sum : ℕ) : Prop :=
  ∃ n : ℕ, (3 * n) + (3 * n + 3) + (3 * n + 6) = sum

-- Define the proof statement
theorem largest_number_is_27 : three_consecutive_multiples_sum_to 3 72 → 3 * 7 + 6 = 27 :=
by
  intro h
  cases' h with n h_eq
  sorry

end largest_number_is_27_l120_120334


namespace minimal_solution_x_eq_neg_two_is_solution_smallest_solution_l120_120830

theorem minimal_solution (x : ℝ) (h : x * |x| = 3 * x + 2) : -2 ≤ x :=
begin
  sorry
end

theorem x_eq_neg_two_is_solution : ( -2 : ℝ ) * |-2| = 3 * -2 + 2 :=
begin
  norm_num,
end

/-- The smallest value of x satisfying x|x| = 3x + 2 is -2 -/
theorem smallest_solution : ∃ x : ℝ, x * |x| = 3 * x + 2 ∧ ∀ y : ℝ, y * |y| = 3 * y + 2 → y ≥ x :=
begin
  use -2,
  split,
  { norm_num },
  { intro y,
    sorry }
end

end minimal_solution_x_eq_neg_two_is_solution_smallest_solution_l120_120830


namespace first_digit_base_9_of_y_l120_120439

def base_3_to_base_10 (n : Nat) : Nat := sorry
def base_10_to_base_9_first_digit (n : Nat) : Nat := sorry

theorem first_digit_base_9_of_y :
  let y := 11220022110022112221
  let base_10_y := base_3_to_base_10 y
  base_10_to_base_9_first_digit base_10_y = 4 :=
by
  let y := 11220022110022112221
  let base_10_y := base_3_to_base_10 y
  show base_10_to_base_9_first_digit base_10_y = 4
  sorry

end first_digit_base_9_of_y_l120_120439


namespace sedrach_divides_each_pie_l120_120890

theorem sedrach_divides_each_pie (P : ℕ) :
  (13 * P * 5 = 130) → P = 2 :=
by
  sorry

end sedrach_divides_each_pie_l120_120890


namespace factor_x_squared_minus_169_l120_120072

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := 
by
  -- Recognize that 169 is a perfect square
  have h : 169 = 13^2 := by norm_num
  -- Use the difference of squares formula
  -- Sorry is used to skip the proof part
  sorry

end factor_x_squared_minus_169_l120_120072


namespace parameterize_line_l120_120451

theorem parameterize_line (f : ℝ → ℝ) (t : ℝ) (x y : ℝ)
  (h1 : y = 2 * x - 30)
  (h2 : (x, y) = (f t, 20 * t - 10)) :
  f t = 10 * t + 10 :=
sorry

end parameterize_line_l120_120451


namespace problem1_problem2_l120_120237

variable (α : ℝ) (tan_alpha_eq_three : Real.tan α = 3)

theorem problem1 : (4 * Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 11 / 14 :=
by sorry

theorem problem2 : Real.sin α * Real.cos α = 3 / 10 :=
by sorry

end problem1_problem2_l120_120237


namespace eight_bags_weight_l120_120406

theorem eight_bags_weight
  (bags_weight : ℕ → ℕ)
  (h1 : bags_weight 12 = 24) :
  bags_weight 8 = 16 :=
  sorry

end eight_bags_weight_l120_120406


namespace radiator_water_fraction_l120_120339

noncomputable def fraction_of_water_after_replacements (initial_water : ℚ) (initial_antifreeze : ℚ) (removal_fraction : ℚ)
  (num_replacements : ℕ) : ℚ :=
  initial_water * (removal_fraction ^ num_replacements)

theorem radiator_water_fraction :
  let initial_water := 10
  let initial_antifreeze := 10
  let total_volume := 20
  let removal_volume := 5
  let removal_fraction := 3 / 4
  let num_replacements := 4
  fraction_of_water_after_replacements initial_water initial_antifreeze removal_fraction num_replacements / total_volume = 0.158 := 
sorry

end radiator_water_fraction_l120_120339


namespace part1_part2_l120_120682

noncomputable def f (x : ℝ) := Real.exp x

theorem part1 (x : ℝ) (h : x ≥ 0) (m : ℝ) : 
  (x - 1) * f x ≥ m * x^2 - 1 ↔ m ≤ 1 / 2 :=
sorry

theorem part2 (x : ℝ) (h : x > 0) : 
  f x > 4 * Real.log x + 8 - 8 * Real.log 2 :=
sorry

end part1_part2_l120_120682


namespace who_is_werewolf_choose_companion_l120_120921

-- Define inhabitants with their respective statements
inductive Inhabitant
| A | B | C

-- Assume each inhabitant can be either a knight (truth-teller) or a liar
def is_knight (i : Inhabitant) : Prop := sorry

-- Define statements made by each inhabitant
def A_statement : Prop := ∃ werewolf : Inhabitant, werewolf = Inhabitant.C
def B_statement : Prop := ¬(∃ werewolf : Inhabitant, werewolf = Inhabitant.B)
def C_statement : Prop := ∃ liar1 liar2 : Inhabitant, liar1 ≠ liar2 ∧ liar1 ≠ Inhabitant.C ∧ liar2 ≠ Inhabitant.C

-- Define who is the werewolf (liar)
def is_werewolf (i : Inhabitant) : Prop := ¬is_knight i

-- The given conditions from statements
axiom A_is_knight : is_knight Inhabitant.A ↔ A_statement
axiom B_is_knight : is_knight Inhabitant.B ↔ B_statement
axiom C_is_knight : is_knight Inhabitant.C ↔ C_statement

-- The conclusion: C is the werewolf and thus a liar.
theorem who_is_werewolf : is_werewolf Inhabitant.C :=
by sorry

-- Choosing a companion: 
-- If C is a werewolf, we prefer to pick A as a companion over B or C.
theorem choose_companion (worry_about_werewolf : Bool) : Inhabitant :=
if worry_about_werewolf then Inhabitant.A else sorry

end who_is_werewolf_choose_companion_l120_120921


namespace number_of_honey_bees_l120_120726

theorem number_of_honey_bees (total_honey : ℕ) (honey_one_bee : ℕ) (days : ℕ) (h1 : total_honey = 30) (h2 : honey_one_bee = 1) (h3 : days = 30) : 
  (total_honey / honey_one_bee) = 30 :=
by
  -- Given total_honey = 30 grams in 30 days
  -- Given honey_one_bee = 1 gram in 30 days
  -- We need to prove (total_honey / honey_one_bee) = 30
  sorry

end number_of_honey_bees_l120_120726


namespace range_m_l120_120229

def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

noncomputable def problem :=
  ∀ (m : ℝ), (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

theorem range_m (m : ℝ) : problem := 
  sorry

end range_m_l120_120229


namespace positive_difference_eq_six_l120_120908

theorem positive_difference_eq_six (x y : ℝ) (h1 : x + y = 8) (h2 : x ^ 2 - y ^ 2 = 48) : |x - y| = 6 := by
  sorry

end positive_difference_eq_six_l120_120908


namespace value_of_a_l120_120846

noncomputable def f (a : ℝ) (x : ℝ) := (x-1)*(x^2 - 3*x + a)

-- Define the condition that 1 is not a critical point
def not_critical (a : ℝ) : Prop := f a 1 ≠ 0

theorem value_of_a (a : ℝ) (h : not_critical a) : a = 2 := 
sorry

end value_of_a_l120_120846


namespace each_cow_gives_5_liters_per_day_l120_120184

-- Define conditions
def cows : ℕ := 52
def weekly_milk : ℕ := 1820
def days_in_week : ℕ := 7

-- Define daily_milk as the daily milk production
def daily_milk := weekly_milk / days_in_week

-- Define milk_per_cow as the amount of milk each cow produces per day
def milk_per_cow := daily_milk / cows

-- Statement to prove
theorem each_cow_gives_5_liters_per_day : milk_per_cow = 5 :=
by
  -- This is where you would normally fill in the proof steps
  sorry

end each_cow_gives_5_liters_per_day_l120_120184


namespace three_digit_integers_congruent_to_2_mod_4_l120_120719

theorem three_digit_integers_congruent_to_2_mod_4 : 
  let count := (249 - 25 + 1) in
  count = 225 :=
by
  let k_min := 25
  let k_max := 249
  have h_count : count = (k_max - k_min + 1) := rfl
  rw h_count
  norm_num

end three_digit_integers_congruent_to_2_mod_4_l120_120719


namespace inversely_proportional_example_l120_120304

theorem inversely_proportional_example (x y k : ℝ) (h₁ : x * y = k) (h₂ : x = 30) (h₃ : y = 8) :
  y = 24 → x = 10 :=
by
  sorry

end inversely_proportional_example_l120_120304


namespace three_digit_integers_congruent_to_2_mod_4_l120_120715

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l120_120715


namespace sin_sq_sub_sin_double_l120_120844

open Real

theorem sin_sq_sub_sin_double (alpha : ℝ) (h : tan alpha = 1 / 2) : sin alpha ^ 2 - sin (2 * alpha) = -3 / 5 := 
by 
  sorry

end sin_sq_sub_sin_double_l120_120844


namespace expression_positive_for_all_integers_l120_120885

theorem expression_positive_for_all_integers (n : ℤ) : 6 * n^2 - 7 * n + 2 > 0 :=
by
  sorry

end expression_positive_for_all_integers_l120_120885


namespace quadratic_function_expression_l120_120394

-- Definitions based on conditions
def quadratic (f : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
def condition1 (f : ℝ → ℝ) : Prop := (f 0 = 1)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) - f x = 4 * x

-- The theorem we want to prove
theorem quadratic_function_expression (f : ℝ → ℝ) 
  (hf_quad : quadratic f)
  (hf_cond1 : condition1 f)
  (hf_cond2 : condition2 f) : 
  ∃ (a b c : ℝ), a = 2 ∧ b = -2 ∧ c = 1 ∧ ∀ x, f x = a * x^2 + b * x + c :=
sorry

end quadratic_function_expression_l120_120394


namespace p_hyperbola_implies_m_range_p_necessary_not_sufficient_for_q_l120_120843

def p (m : ℝ) (x y : ℝ) : Prop := (x^2) / (m - 1) + (y^2) / (m - 4) = 1
def q (m : ℝ) (x y : ℝ) : Prop := (x^2) / (m - 2) + (y^2) / (4 - m) = 1

theorem p_hyperbola_implies_m_range (m : ℝ) (x y : ℝ) :
  p m x y → 1 < m ∧ m < 4 :=
sorry

theorem p_necessary_not_sufficient_for_q (m : ℝ) (x y : ℝ) :
  (1 < m ∧ m < 4) ∧ p m x y →
  (q m x y → (2 < m ∧ m < 3) ∨ (3 < m ∧ m < 4)) :=
sorry

end p_hyperbola_implies_m_range_p_necessary_not_sufficient_for_q_l120_120843


namespace card_probability_l120_120031

-- Definitions to capture the problem's conditions in Lean
def total_cards : ℕ := 52
def remaining_after_first : ℕ := total_cards - 1
def remaining_after_second : ℕ := total_cards - 2

def kings : ℕ := 4
def non_heart_kings : ℕ := 3
def non_kings_in_hearts : ℕ := 12
def spades_and_diamonds : ℕ := 26

-- Define probabilities for each step
def prob_first_king : ℚ := non_heart_kings / total_cards
def prob_second_heart : ℚ := non_kings_in_hearts / remaining_after_first
def prob_third_spade_or_diamond : ℚ := spades_and_diamonds / remaining_after_second

-- Calculate total probability
def total_probability : ℚ := prob_first_king * prob_second_heart * prob_third_spade_or_diamond

-- Theorem statement that encapsulates the problem
theorem card_probability : total_probability = 26 / 3675 :=
by sorry

end card_probability_l120_120031


namespace dvd_cost_packs_l120_120473

theorem dvd_cost_packs (cost_per_pack : ℕ) (number_of_packs : ℕ) (total_money : ℕ) :
  cost_per_pack = 12 → number_of_packs = 11 → total_money = (cost_per_pack * number_of_packs) → total_money = 132 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dvd_cost_packs_l120_120473


namespace find_E_l120_120180

variables (E F G H : ℕ)

noncomputable def conditions := 
  (E * F = 120) ∧ 
  (G * H = 120) ∧ 
  (E - F = G + H - 2) ∧ 
  (E ≠ F) ∧
  (E ≠ G) ∧ 
  (E ≠ H) ∧
  (F ≠ G) ∧
  (F ≠ H) ∧
  (G ≠ H)

theorem find_E (E F G H : ℕ) (h : conditions E F G H) : E = 30 :=
sorry

end find_E_l120_120180


namespace correctness_of_statements_l120_120155

theorem correctness_of_statements :
  (statement1 ∧ statement4 ∧ statement5) :=
by sorry

end correctness_of_statements_l120_120155


namespace cost_of_a_pen_l120_120081

theorem cost_of_a_pen:
  ∃ x y : ℕ, 5 * x + 4 * y = 345 ∧ 3 * x + 6 * y = 285 ∧ x = 52 :=
by
  sorry

end cost_of_a_pen_l120_120081


namespace tangent_line_at_point_l120_120398

noncomputable def func (x : ℝ) : ℝ := x + Real.log x

-- Given that f(x) is differentiable in (0, +∞) and f(e^x) = x + e^x, prove the tangent line equation at x = 1
theorem tangent_line_at_point (f : ℝ → ℝ) (hf : ∀ x > 0, DifferentiableAt ℝ f x) (h : ∀ x : ℝ, f (Real.exp x) = x + Real.exp x) :
  f 1 = 1 ∧ deriv f 1 = 2 ∧ (∀ (y : ℝ), y = f 1 → (λ x, 2 * x - y - 1) = 0) :=
by
  sorry

end tangent_line_at_point_l120_120398


namespace correct_option_l120_120612

theorem correct_option :
  (3 * a^2 + 5 * a^2 ≠ 8 * a^4) ∧
  (5 * a^2 * b - 6 * a * b^2 ≠ -a * b^2) ∧
  (2 * x + 3 * y ≠ 5 * x * y) ∧
  (9 * x * y - 6 * x * y = 3 * x * y) :=
by
  sorry

end correct_option_l120_120612


namespace parabola_focus_directrix_distance_l120_120588

theorem parabola_focus_directrix_distance :
  ∀ {x y : ℝ}, y^2 = (1/4) * x → dist (1/16, 0) (-1/16, 0) = 1/8 := by
sorry

end parabola_focus_directrix_distance_l120_120588


namespace platform_length_l120_120331

theorem platform_length (train_speed_kmph : ℕ) (train_time_man_seconds : ℕ) (train_time_platform_seconds : ℕ) (train_speed_mps : ℕ) : 
  train_speed_kmph = 54 →
  train_time_man_seconds = 20 →
  train_time_platform_seconds = 30 →
  train_speed_mps = (54 * 1000 / 3600) →
  (54 * 5 / 18) = 15 →
  ∃ (P : ℕ), (train_speed_mps * train_time_platform_seconds) = (train_speed_mps * train_time_man_seconds) + P ∧ P = 150 :=
by
  sorry

end platform_length_l120_120331


namespace problem_statement_l120_120228

-- Defining the condition x^3 = 8
def condition1 (x : ℝ) : Prop := x^3 = 8

-- Defining the function f(x) = (x-1)(x+1)(x^2 + x + 1)
def f (x : ℝ) : ℝ := (x - 1) * (x + 1) * (x^2 + x + 1)

-- The theorem we want to prove: For any x satisfying the condition, the function value is 21
theorem problem_statement (x : ℝ) (h : condition1 x) : f x = 21 := 
by
  sorry

end problem_statement_l120_120228


namespace three_digit_integers_congruent_to_2_mod_4_l120_120718

theorem three_digit_integers_congruent_to_2_mod_4 : 
  let count := (249 - 25 + 1) in
  count = 225 :=
by
  let k_min := 25
  let k_max := 249
  have h_count : count = (k_max - k_min + 1) := rfl
  rw h_count
  norm_num

end three_digit_integers_congruent_to_2_mod_4_l120_120718


namespace average_age_after_leaves_is_27_l120_120586

def average_age_of_remaining_people (initial_avg_age : ℕ) (initial_people_count : ℕ) 
    (age_leave1 : ℕ) (age_leave2 : ℕ) (remaining_people_count : ℕ) : ℕ :=
  let initial_total_age := initial_avg_age * initial_people_count
  let new_total_age := initial_total_age - (age_leave1 + age_leave2)
  new_total_age / remaining_people_count

theorem average_age_after_leaves_is_27 :
  average_age_of_remaining_people 25 6 20 22 4 = 27 :=
by
  -- Proof is skipped
  sorry

end average_age_after_leaves_is_27_l120_120586


namespace poles_inside_base_l120_120491

theorem poles_inside_base :
  ∃ n : ℕ, 2015 + n ≡ 0 [MOD 36] ∧ n = 1 :=
sorry

end poles_inside_base_l120_120491


namespace solution_set_non_empty_implies_a_gt_1_l120_120408

theorem solution_set_non_empty_implies_a_gt_1 (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := 
  sorry

end solution_set_non_empty_implies_a_gt_1_l120_120408


namespace squareable_natural_numbers_l120_120169

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def can_be_arranged (n : ℕ) (arrangement : list ℕ) : Prop :=
  (arrangement.length = n) ∧
  (∀ (k : ℕ), k < n → is_perfect_square ((arrangement.nth k).getD 0 + (k + 1)))

def is_squareable (n : ℕ) : Prop :=
  ∃ (arrangement : list ℕ), can_be_arranged n arrangement

theorem squareable_natural_numbers : (is_squareable 7 → False) ∧
                                     (is_squareable 9) ∧
                                     (is_squareable 11 → False) ∧
                                     (is_squareable 15) :=
by {
  sorry
}

end squareable_natural_numbers_l120_120169


namespace wall_length_proof_l120_120777

-- Define the conditions from the problem
def wall_height : ℝ := 100 -- Height in cm
def wall_thickness : ℝ := 5 -- Thickness in cm
def brick_length : ℝ := 25 -- Brick length in cm
def brick_width : ℝ := 11 -- Brick width in cm
def brick_height : ℝ := 6 -- Brick height in cm
def number_of_bricks : ℝ := 242.42424242424244

-- Calculate the volume of one brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Calculate the total volume of the bricks
def total_brick_volume : ℝ := brick_volume * number_of_bricks

-- Define the proof problem
theorem wall_length_proof : total_brick_volume = wall_height * wall_thickness * 800 :=
sorry

end wall_length_proof_l120_120777


namespace arithmetic_sequence_problem_l120_120288

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) (h_incr : ∀ n, a (n + 1) > a n) (h_prod : a 4 * a 5 = 13) : a 3 * a 6 = -275 := 
sorry

end arithmetic_sequence_problem_l120_120288


namespace max_trig_expression_l120_120218

open Real

theorem max_trig_expression (x y z : ℝ) :
  (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 := sorry

end max_trig_expression_l120_120218


namespace no_generating_combination_l120_120484

-- Representing Rubik's Cube state as a type (assume a type exists)
axiom CubeState : Type

-- A combination of turns represented as a function on states
axiom A : CubeState → CubeState

-- Simple rotations
axiom P : CubeState → CubeState
axiom Q : CubeState → CubeState

-- Rubik's Cube property of generating combination (assuming generating implies all states achievable)
def is_generating (A : CubeState → CubeState) :=
  ∀ X : CubeState, ∃ m n : ℕ, P X = A^[m] X ∧ Q X = A^[n] X

-- Non-commutativity condition
axiom non_commutativity : ∀ X : CubeState, P (Q X) ≠ Q (P X)

-- Formal statement of the problem
theorem no_generating_combination : ¬ ∃ A : CubeState → CubeState, is_generating A :=
by sorry

end no_generating_combination_l120_120484


namespace ending_point_divisible_by_9_l120_120316

theorem ending_point_divisible_by_9 (n : ℕ) (ending_point : ℕ) 
  (h1 : n = 11110) 
  (h2 : ∃ k : ℕ, 10 + 9 * k = ending_point) : 
  ending_point = 99999 := 
  sorry

end ending_point_divisible_by_9_l120_120316


namespace coin_flip_ways_l120_120338

theorem coin_flip_ways : ∃ (n : ℕ), ∀ (k : ℕ), k = 10 → (number_of_valid_sequences k = 42) :=
by sorry


end coin_flip_ways_l120_120338


namespace theater_revenue_l120_120194

theorem theater_revenue 
  (seats : ℕ)
  (capacity_percentage : ℝ)
  (ticket_price : ℝ)
  (days : ℕ)
  (H1 : seats = 400)
  (H2 : capacity_percentage = 0.8)
  (H3 : ticket_price = 30)
  (H4 : days = 3)
  : (seats * capacity_percentage * ticket_price * days = 28800) :=
by
  sorry

end theater_revenue_l120_120194


namespace distinct_ordered_pairs_count_l120_120257

theorem distinct_ordered_pairs_count :
  ∃ S : Finset (ℕ × ℕ), 
    (∀ p ∈ S, 1 ≤ p.1 ∧ 1 ≤ p.2 ∧ (1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6)) ∧
    S.card = 9 := 
by
  sorry

end distinct_ordered_pairs_count_l120_120257


namespace monthly_average_decrease_rate_l120_120865

-- Conditions
def january_production : Float := 1.6 * 10^6
def march_production : Float := 0.9 * 10^6
def rate_decrease : Float := 0.25

-- Proof Statement: we need to prove that the monthly average decrease rate x = 0.25 satisfies the given condition
theorem monthly_average_decrease_rate :
  january_production * (1 - rate_decrease) * (1 - rate_decrease) = march_production := by
  sorry

end monthly_average_decrease_rate_l120_120865


namespace algebraic_expression_value_l120_120260

theorem algebraic_expression_value (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : m ^ 2 = 25) :
  m^2 - 100*a - 99*b - b*c*d + |c*d - 2| = -74 :=
by
  sorry

end algebraic_expression_value_l120_120260


namespace unit_prices_l120_120897

theorem unit_prices (x y : ℕ) (h1 : 5 * x + 4 * y = 139) (h2 : 4 * x + 5 * y = 140) :
  x = 15 ∧ y = 16 :=
by
  -- Proof will go here
  sorry

end unit_prices_l120_120897


namespace speed_ratio_thirteen_l120_120759

noncomputable section

def speed_ratio (vNikita vCar : ℝ) : ℝ := vCar / vNikita

theorem speed_ratio_thirteen :
  ∀ (vNikita vCar : ℝ),
  (65 * vNikita = 5 * vCar) →
  speed_ratio vNikita vCar = 13 :=
by
  intros vNikita vCar h
  unfold speed_ratio
  sorry

end speed_ratio_thirteen_l120_120759


namespace trapezoid_area_l120_120033

noncomputable def area_trapezoid (B1 B2 h : ℝ) : ℝ := (1 / 2 * (B1 + B2) * h)

theorem trapezoid_area
    (h1 : ∀ x : ℝ, 3 * x = 10 → x = 10 / 3)
    (h2 : ∀ x : ℝ, 3 * x = 5 → x = 5 / 3)
    (h3 : B1 = 10 / 3)
    (h4 : B2 = 5 / 3)
    (h5 : h = 5)
    : area_trapezoid B1 B2 h = 12.5 := by
  sorry

end trapezoid_area_l120_120033


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l120_120710

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l120_120710


namespace total_oranges_picked_l120_120425

-- Defining the number of oranges picked by Mary, Jason, and Sarah
def maryOranges := 122
def jasonOranges := 105
def sarahOranges := 137

-- The theorem to prove that the total number of oranges picked is 364
theorem total_oranges_picked : maryOranges + jasonOranges + sarahOranges = 364 := by
  sorry

end total_oranges_picked_l120_120425


namespace largest_in_given_numbers_l120_120041

noncomputable def A := 5.14322
noncomputable def B := 5.1432222222222222222 -- B = 5.143(bar)2
noncomputable def C := 5.1432323232323232323 -- C = 5.14(bar)32
noncomputable def D := 5.1432432432432432432 -- D = 5.1(bar)432
noncomputable def E := 5.1432143214321432143 -- E = 5.(bar)4321

theorem largest_in_given_numbers : D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_in_given_numbers_l120_120041


namespace min_paint_steps_l120_120122

-- Checkered square of size 2021x2021 where all cells initially white.
-- Ivan selects two cells and paints them black.
-- Cells with at least one black neighbor by side are painted black simultaneously each step.

-- Define a function to represent the steps required to paint the square black
noncomputable def min_steps_to_paint_black (n : ℕ) (a b : ℕ × ℕ) : ℕ :=
  sorry -- Placeholder for the actual function definition, as we're focusing on the statement.

-- Define the specific instance of the problem
def square_size := 2021
def initial_cells := ((505, 1010), (1515, 1010))

-- Theorem statement: Proving the minimal number of steps required is 1515
theorem min_paint_steps : min_steps_to_paint_black square_size initial_cells.1 initial_cells.2 = 1515 :=
sorry

end min_paint_steps_l120_120122


namespace third_consecutive_even_number_l120_120163

theorem third_consecutive_even_number (n : ℕ) (h : n % 2 = 0) (sum_eq : n + (n + 2) + (n + 4) = 246) : (n + 4) = 84 :=
by
  -- This statement sets up the conditions and the goal of the proof.
  sorry

end third_consecutive_even_number_l120_120163


namespace alton_weekly_profit_l120_120496

-- Definitions of the given conditions
def dailyEarnings : ℕ := 8
def daysInWeek : ℕ := 7
def weeklyRent : ℕ := 20

-- The proof problem: Prove that the total profit every week is $36
theorem alton_weekly_profit : (dailyEarnings * daysInWeek) - weeklyRent = 36 := by
  sorry

end alton_weekly_profit_l120_120496


namespace arrange_p_q_r_l120_120699

theorem arrange_p_q_r (p : ℝ) (h : 1 < p ∧ p < 1.1) : p < p^p ∧ p^p < p^(p^p) :=
by
  sorry

end arrange_p_q_r_l120_120699


namespace calculate_average_fish_caught_l120_120859

-- Definitions based on conditions
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def Toph_fish : ℕ := 12

-- Total fish and average calculation
def total_fish : ℕ := Aang_fish + Sokka_fish + Toph_fish
def number_of_people : ℕ := 3
def average_fish_per_person : ℕ := total_fish / number_of_people

-- Theorem to prove
theorem calculate_average_fish_caught : average_fish_per_person = 8 := 
by 
  -- Proof steps are skipped with 'sorry', but the statement is set up correctly
  sorry

end calculate_average_fish_caught_l120_120859


namespace least_trees_l120_120345

theorem least_trees (N : ℕ) (h1 : N % 7 = 0) (h2 : N % 6 = 0) (h3 : N % 4 = 0) (h4 : N ≥ 100) : N = 168 :=
sorry

end least_trees_l120_120345


namespace fraction_equal_l120_120862

theorem fraction_equal {a b x : ℝ} (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  (a + b) / (a - b) = (x + 1) / (x - 1) := 
by
  sorry

end fraction_equal_l120_120862


namespace cost_of_items_l120_120805

theorem cost_of_items (x y z : ℝ)
  (h1 : 20 * x + 3 * y + 2 * z = 32)
  (h2 : 39 * x + 5 * y + 3 * z = 58) :
  5 * (x + y + z) = 30 := by
  sorry

end cost_of_items_l120_120805


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l120_120706

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l120_120706


namespace eval_x_power_x_power_x_at_3_l120_120211

theorem eval_x_power_x_power_x_at_3 : (3^3)^(3^3) = 27^27 := by
    sorry

end eval_x_power_x_power_x_at_3_l120_120211


namespace median_of_scores_l120_120269

theorem median_of_scores : ∀ (scores : List ℚ),
  scores = [90, 78, 82, 85, 90] → median scores = 85 :=
begin
  intros scores h,
  rw h,
  sorry,
end

end median_of_scores_l120_120269


namespace min_sum_of_factors_l120_120160

theorem min_sum_of_factors (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 3960) : 
  a + b + c = 72 :=
sorry

end min_sum_of_factors_l120_120160


namespace vertex_angle_double_angle_triangle_l120_120525

theorem vertex_angle_double_angle_triangle 
  {α β : ℝ} (h1 : α + β + β = 180) (h2 : α = 2 * β ∨ β = 2 * α) :
  α = 36 ∨ α = 90 :=
by
  sorry

end vertex_angle_double_angle_triangle_l120_120525


namespace delores_money_left_l120_120371

theorem delores_money_left (initial_amount spent_computer spent_printer : ℝ) 
    (h1 : initial_amount = 450) 
    (h2 : spent_computer = 400) 
    (h3 : spent_printer = 40) : 
    initial_amount - (spent_computer + spent_printer) = 10 := 
by 
    sorry

end delores_money_left_l120_120371


namespace stream_speed_l120_120326

theorem stream_speed (c v : ℝ) (h1 : c - v = 9) (h2 : c + v = 12) : v = 1.5 :=
by
  sorry

end stream_speed_l120_120326


namespace ford_younger_than_christopher_l120_120385

variable (G C F Y : ℕ)

-- Conditions
axiom h1 : G = C + 8
axiom h2 : F = C - Y
axiom h3 : G + C + F = 60
axiom h4 : C = 18

-- Target statement
theorem ford_younger_than_christopher : Y = 2 :=
sorry

end ford_younger_than_christopher_l120_120385


namespace shoe_price_on_monday_l120_120567

theorem shoe_price_on_monday
  (price_on_thursday : ℝ)
  (price_increase : ℝ)
  (discount : ℝ)
  (price_on_friday : ℝ := price_on_thursday * (1 + price_increase))
  (price_on_monday : ℝ := price_on_friday * (1 - discount))
  (price_on_thursday_eq : price_on_thursday = 50)
  (price_increase_eq : price_increase = 0.2)
  (discount_eq : discount = 0.15) :
  price_on_monday = 51 :=
by
  sorry

end shoe_price_on_monday_l120_120567


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l120_120708

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l120_120708


namespace factorial_div_sum_l120_120952

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l120_120952


namespace counting_number_leaves_remainder_of_6_l120_120255

theorem counting_number_leaves_remainder_of_6:
  ∃! d : ℕ, d > 6 ∧ d ∣ (53 - 6) ∧ 53 % d = 6 :=
begin
  sorry
end

end counting_number_leaves_remainder_of_6_l120_120255


namespace third_generation_tail_length_l120_120277

theorem third_generation_tail_length (tail_length : ℕ → ℕ) (h0 : tail_length 0 = 16)
    (h_next : ∀ n, tail_length (n + 1) = tail_length n + (25 * tail_length n) / 100) :
    tail_length 2 = 25 :=
by
  sorry

end third_generation_tail_length_l120_120277


namespace mrs_heine_dogs_l120_120882

-- Define the number of biscuits per dog
def biscuits_per_dog : ℕ := 3

-- Define the total number of biscuits
def total_biscuits : ℕ := 6

-- Define the number of dogs
def number_of_dogs : ℕ := 2

-- Define the proof statement
theorem mrs_heine_dogs : total_biscuits / biscuits_per_dog = number_of_dogs :=
by
  sorry

end mrs_heine_dogs_l120_120882


namespace largest_number_of_gold_coins_l120_120616

theorem largest_number_of_gold_coins 
  (num_friends : ℕ)
  (extra_coins : ℕ)
  (total_coins : ℕ) :
  num_friends = 13 →
  extra_coins = 3 →
  total_coins < 150 →
  ∀ k : ℕ, total_coins = num_friends * k + extra_coins →
  total_coins ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l120_120616


namespace slope_of_line_l120_120670

theorem slope_of_line (x y : ℝ) :
  (4 * x - 7 * y = 28) → (slope (line_eq := 4 * x - 7 * y) = 4 / 7) :=
by
  sorry

end slope_of_line_l120_120670


namespace sum_of_integers_divisible_by_15_less_than_999_l120_120831

theorem sum_of_integers_divisible_by_15_less_than_999 : 
  ∑ k in finset.range 67, (15 * k) = 33165 := 
by
  sorry

end sum_of_integers_divisible_by_15_less_than_999_l120_120831


namespace ellipse_hyperbola_foci_l120_120449

theorem ellipse_hyperbola_foci (a b : ℝ) 
  (h1 : ∃ (a b : ℝ), b^2 - a^2 = 25 ∧ a^2 + b^2 = 64) : 
  |a * b| = (Real.sqrt 3471) / 2 :=
by
  sorry

end ellipse_hyperbola_foci_l120_120449


namespace sum_reciprocals_of_roots_l120_120222

-- Problem statement: Prove that the sum of the reciprocals of the roots of the quadratic equation x^2 - 11x + 6 = 0 is 11/6.
theorem sum_reciprocals_of_roots : 
  ∀ (p q : ℝ), p + q = 11 → p * q = 6 → (1 / p + 1 / q = 11 / 6) :=
by
  intro p q hpq hprod
  sorry

end sum_reciprocals_of_roots_l120_120222


namespace number_of_children_l120_120443

theorem number_of_children (A V S : ℕ) (x : ℕ → ℕ) (n : ℕ) 
  (h1 : (A / 2) + V = (A + V + S + (Finset.range (n - 3)).sum x) / n)
  (h2 : S + A = V + (Finset.range (n - 3)).sum x) : 
  n = 6 :=
sorry

end number_of_children_l120_120443


namespace real_roots_prime_equation_l120_120099

noncomputable def has_rational_roots (p q : ℕ) : Prop :=
  ∃ x : ℚ, x^2 + p^2 * x + q^3 = 0

theorem real_roots_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  has_rational_roots p q ↔ (p = 3 ∧ q = 2) :=
sorry

end real_roots_prime_equation_l120_120099


namespace journey_time_difference_l120_120342

theorem journey_time_difference :
  let t1 := (100:ℝ) / 60
  let t2 := (400:ℝ) / 40
  let T1 := t1 + t2
  let T2 := (500:ℝ) / 50
  let difference := (T1 - T2) * 60
  abs (difference - 100) < 0.01 :=
by
  sorry

end journey_time_difference_l120_120342


namespace sum_of_series_l120_120973

theorem sum_of_series :
  (∑' n : ℕ, (3^n) / (3^(3^n) + 1)) = 1 / 2 :=
sorry

end sum_of_series_l120_120973


namespace contrapositive_proof_l120_120676

theorem contrapositive_proof (m : ℕ) (h_pos : 0 < m) :
  (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

end contrapositive_proof_l120_120676


namespace oblique_asymptote_l120_120465

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 8 * x + 12) / (3 * x + 4)

theorem oblique_asymptote :
  (∃ b : ℝ, ∀ x : ℝ, ∥f x - (x + b)∥ < ε) := by
sorry

end oblique_asymptote_l120_120465


namespace evie_collected_shells_for_6_days_l120_120373

theorem evie_collected_shells_for_6_days (d : ℕ) (h1 : 10 * d - 2 = 58) : d = 6 := by
  sorry

end evie_collected_shells_for_6_days_l120_120373


namespace average_fish_per_person_l120_120857

theorem average_fish_per_person (Aang Sokka Toph : ℕ) 
  (haang : Aang = 7) (hsokka : Sokka = 5) (htoph : Toph = 12) : 
  (Aang + Sokka + Toph) / 3 = 8 := by
  sorry

end average_fish_per_person_l120_120857


namespace smallest_digit_divisible_by_9_l120_120824

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (0 ≤ d ∧ d < 10) ∧ (∃ k : ℕ, 26 + d = 9 * k) ∧ d = 1 :=
by
  sorry

end smallest_digit_divisible_by_9_l120_120824


namespace domain_log_base_5_range_3_pow_neg_x_l120_120154

theorem domain_log_base_5 (x : ℝ) :
  (∃ y : ℝ, y = logBase 5 (1 - x)) ↔ x < 1 :=
by
  sorry

theorem range_3_pow_neg_x (y : ℝ) :
  (∃ x : ℝ, y = 3 ^ (-x)) ↔ y > 0 :=
by
  sorry

end domain_log_base_5_range_3_pow_neg_x_l120_120154


namespace meryll_questions_l120_120754

theorem meryll_questions (M P : ℕ) (h1 : (3/5 : ℚ) * M + (2/3 : ℚ) * P = 31) (h2 : P = 15) : M = 35 :=
sorry

end meryll_questions_l120_120754


namespace at_least_one_term_le_one_l120_120748

theorem at_least_one_term_le_one
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) :
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
  sorry

end at_least_one_term_le_one_l120_120748


namespace find_ratio_l120_120772

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Given conditions
axiom sum_arithmetic_a (n : ℕ) : S n = n / 2 * (a 1 + a n)
axiom sum_arithmetic_b (n : ℕ) : T n = n / 2 * (b 1 + b n)
axiom sum_ratios (n : ℕ) : S n / T n = (2 * n + 1) / (3 * n + 2)

-- The proof problem
theorem find_ratio : (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := 
sorry

end find_ratio_l120_120772


namespace find_a₃_l120_120267

variable (a₁ a₂ a₃ a₄ a₅ : ℝ)
variable (S₅ : ℝ) (a_seq : ℕ → ℝ)

-- Define the conditions for arithmetic sequence and given sum
def is_arithmetic_sequence (a_seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_seq (n+1) - a_seq n = a_seq 1 - a_seq 0

axiom sum_first_five_terms (S₅ : ℝ) (hS : S₅ = 20) : 
  S₅ = (5 * (a₁ + a₅)) / 2

-- Main theorem we need to prove
theorem find_a₃ (hS₅ : S₅ = 20) (h_seq : is_arithmetic_sequence a_seq) :
  (∃ (a₃ : ℝ), a₃ = 4) :=
sorry

end find_a₃_l120_120267


namespace smallest_solution_of_abs_eq_l120_120825

theorem smallest_solution_of_abs_eq (x : ℝ) : 
  (x * |x| = 3 * x + 2 → x ≥ 0 → x = (3 + Real.sqrt 17) / 2) ∧
  (x * |x| = 3 * x + 2 → x < 0 → x = -2) ∧
  (x * |x| = 3 * x + 2 → x = -2 → x = -2) :=
by
  sorry

end smallest_solution_of_abs_eq_l120_120825


namespace no_absolute_winner_l120_120513

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end no_absolute_winner_l120_120513


namespace problem_solution_l120_120236

noncomputable def circle_constant : ℝ := Real.pi
noncomputable def natural_base : ℝ := Real.exp 1

theorem problem_solution (π : ℝ) (e : ℝ) (h₁ : π = Real.pi) (h₂ : e = Real.exp 1) :
  π * Real.log e / Real.log 3 > 3 * Real.log e / Real.log π := by
  sorry

end problem_solution_l120_120236


namespace eccentricity_range_l120_120389

variable {a b c : ℝ} (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (e : ℝ)

-- Assume a > 0, b > 0, and the eccentricity of the hyperbola is given by c = e * a.
variable (a_pos : 0 < a) (b_pos : 0 < b) (hyperbola : (P.1 / a)^2 - (P.2 / b)^2 = 1)
variable (on_right_branch : P.1 > 0)
variable (foci_condition : dist P F₁ = 4 * dist P F₂)
variable (eccentricity_def : c = e * a)

theorem eccentricity_range : 1 < e ∧ e ≤ 5 / 3 := by
  sorry

end eccentricity_range_l120_120389


namespace triangle_abs_simplification_l120_120698

theorem triangle_abs_simplification
  (x y z : ℝ)
  (h1 : x + y > z)
  (h2 : y + z > x)
  (h3 : x + z > y) :
  |x + y - z| - 2 * |y - x - z| = -x + 3 * y - 3 * z :=
by
  sorry

end triangle_abs_simplification_l120_120698


namespace school_count_l120_120265

theorem school_count (n : ℕ) (h1 : 2 * n - 1 = 69) (h2 : n < 76) (h3 : n > 29) : (2 * n - 1) / 3 = 23 :=
by
  sorry

end school_count_l120_120265


namespace min_square_value_l120_120417

theorem min_square_value (a b : ℤ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ r : ℤ, r^2 = 15 * a + 16 * b)
  (h2 : ∃ s : ℤ, s^2 = 16 * a - 15 * b) : 
  231361 ≤ min (15 * a + 16 * b) (16 * a - 15 * b) :=
sorry

end min_square_value_l120_120417


namespace circumcircle_area_of_isosceles_triangle_l120_120482

open Real

theorem circumcircle_area_of_isosceles_triangle:
  (∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
    (dist A B = 4) ∧ (dist A C = 4) ∧ (dist B C = 3) →
    (circle_area A B C = (256 / 13.75) * π)) :=
by sorry

end circumcircle_area_of_isosceles_triangle_l120_120482


namespace num_play_both_l120_120178

-- Definitions based on the conditions
def total_members : ℕ := 30
def play_badminton : ℕ := 17
def play_tennis : ℕ := 19
def play_neither : ℕ := 2

-- The statement we want to prove
theorem num_play_both :
  play_badminton + play_tennis - 8 = total_members - play_neither := by
  -- Omitted proof
  sorry

end num_play_both_l120_120178


namespace grass_field_width_l120_120798

theorem grass_field_width (w : ℝ) (length_field : ℝ) (path_width : ℝ) (area_path : ℝ) :
  length_field = 85 → path_width = 2.5 → area_path = 1450 →
  (90 * (w + path_width * 2) - length_field * w = area_path) → w = 200 :=
by
  intros h_length_field h_path_width h_area_path h_eq
  sorry

end grass_field_width_l120_120798


namespace find_hours_hired_l120_120424

def hourly_rate : ℝ := 15
def tip_rate : ℝ := 0.20
def total_paid : ℝ := 54

theorem find_hours_hired (h : ℝ) : 15 * h + 0.20 * 15 * h = 54 → h = 3 :=
by
  sorry

end find_hours_hired_l120_120424


namespace find_m_l120_120110

def A (m : ℝ) : Set ℝ := {x | x^2 - m * x + m^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def C : Set ℝ := {2, -4}

theorem find_m (m : ℝ) : (A m ∩ B).Nonempty ∧ (A m ∩ C) = ∅ → m = -2 := by
  sorry

end find_m_l120_120110


namespace simplify_expression_l120_120017

theorem simplify_expression (x : ℕ) (h : x = 100) :
  (x + 1) * (x - 1) + x * (2 - x) + (x - 1) ^ 2 = 10000 := by
  sorry

end simplify_expression_l120_120017


namespace flag_blue_area_l120_120643

theorem flag_blue_area (A C₁ C₃ : ℝ) (h₀ : A = 1.0) (h₁ : C₁ + C₃ = 0.36 * A) :
  C₃ = 0.02 * A := by
  sorry

end flag_blue_area_l120_120643


namespace shorter_piece_length_l120_120791

-- Definitions according to conditions in a)
variables (x : ℝ) (total_length : ℝ := 140)
variables (ratio : ℝ := 5 / 2)

-- Statement to be proved
theorem shorter_piece_length : x + ratio * x = total_length → x = 40 := 
by
  intros h
  sorry

end shorter_piece_length_l120_120791


namespace abs_diff_eq_sqrt_l120_120091

theorem abs_diff_eq_sqrt (x1 x2 a b : ℝ) (h1 : x1 + x2 = a) (h2 : x1 * x2 = b) : 
  |x1 - x2| = Real.sqrt (a^2 - 4 * b) :=
by
  sorry

end abs_diff_eq_sqrt_l120_120091


namespace sector_perimeter_l120_120409

noncomputable def radius : ℝ := 2
noncomputable def central_angle_deg : ℝ := 120
noncomputable def expected_perimeter : ℝ := (4 / 3) * Real.pi + 4

theorem sector_perimeter (r : ℝ) (θ : ℝ) (h_r : r = radius) (h_θ : θ = central_angle_deg) :
    let arc_length := θ / 360 * 2 * Real.pi * r
    let perimeter := arc_length + 2 * r
    perimeter = expected_perimeter :=
by
  -- Skip the proof
  sorry

end sector_perimeter_l120_120409


namespace quadratic_polynomials_exist_l120_120372

-- Definitions of the polynomials
def p1 (x : ℝ) := (x - 10)^2 - 1
def p2 (x : ℝ) := x^2 - 1
def p3 (x : ℝ) := (x + 10)^2 - 1

-- The theorem to prove
theorem quadratic_polynomials_exist :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ p1 x1 = 0 ∧ p1 x2 = 0) ∧
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ p2 y1 = 0 ∧ p2 y2 = 0) ∧
  (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ p3 z1 = 0 ∧ p3 z2 = 0) ∧
  (∀ x : ℝ, p1 x + p2 x ≠ 0 ∧ p1 x + p3 x ≠ 0 ∧ p2 x + p3 x ≠ 0) :=
by
  sorry

end quadratic_polynomials_exist_l120_120372


namespace cristina_running_pace_4point2_l120_120758

theorem cristina_running_pace_4point2 :
  ∀ (nicky_pace head_start time_after_start cristina_pace : ℝ),
    nicky_pace = 3 →
    head_start = 12 →
    time_after_start = 30 →
    cristina_pace = 4.2 →
    (time_after_start = head_start + 30 →
    cristina_pace * time_after_start = nicky_pace * (head_start + 30)) :=
by
  sorry

end cristina_running_pace_4point2_l120_120758


namespace scientific_notation_of_3300000000_l120_120514

theorem scientific_notation_of_3300000000 :
  3300000000 = 3.3 * 10^9 :=
sorry

end scientific_notation_of_3300000000_l120_120514


namespace find_k_l120_120220

variable (x y z k : ℝ)

def fractions_are_equal : Prop := (9 / (x + y) = k / (x + z) ∧ k / (x + z) = 15 / (z - y))

theorem find_k (h : fractions_are_equal x y z k) : k = 24 := by
  sorry

end find_k_l120_120220


namespace oranges_in_shop_l120_120134

-- Define the problem conditions
def ratio (M O A : ℕ) : Prop := (10 * O = 2 * M) ∧ (10 * A = 3 * M)

noncomputable def numMangoes : ℕ := 120
noncomputable def numApples : ℕ := 36

-- Statement of the problem
theorem oranges_in_shop (ratio_factor : ℕ) (h_ratio : ratio numMangoes (2 * ratio_factor) numApples) :
  (2 * ratio_factor) = 24 := by
  sorry

end oranges_in_shop_l120_120134


namespace odd_function_decreasing_function_max_min_values_on_interval_l120_120563

variable (f : ℝ → ℝ)

axiom func_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom func_negative_for_positive : ∀ x : ℝ, (0 < x) → f x < 0
axiom func_value_at_one : f 1 = -2

theorem odd_function : ∀ x : ℝ, f (-x) = -f x := by
  have f_zero : f 0 = 0 := by sorry
  sorry

theorem decreasing_function : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → f x₁ > f x₂ := by sorry

theorem max_min_values_on_interval :
  (f (-3) = 6) ∧ (f 3 = -6) := by sorry

end odd_function_decreasing_function_max_min_values_on_interval_l120_120563


namespace polynomial_divisibility_l120_120085

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x : ℝ, (x ^ 4 + a * x ^ 2 + b * x + c) = (x - 1) ^ 3 * (x + 1) →
  a = 0 ∧ b = 2 ∧ c = -1) :=
by
  intros x h
  sorry

end polynomial_divisibility_l120_120085


namespace distance_from_apex_to_larger_cross_section_l120_120779

noncomputable def area1 : ℝ := 324 * Real.sqrt 2
noncomputable def area2 : ℝ := 648 * Real.sqrt 2
def distance_between_planes : ℝ := 12

theorem distance_from_apex_to_larger_cross_section
  (area1 area2 : ℝ)
  (distance_between_planes : ℝ)
  (h_area1 : area1 = 324 * Real.sqrt 2)
  (h_area2 : area2 = 648 * Real.sqrt 2)
  (h_distance : distance_between_planes = 12) :
  ∃ (H : ℝ), H = 24 + 12 * Real.sqrt 2 :=
by sorry

end distance_from_apex_to_larger_cross_section_l120_120779


namespace cubic_sum_identity_l120_120116

variables (x y z : ℝ)

theorem cubic_sum_identity (h1 : x + y + z = 10) (h2 : xy + xz + yz = 30) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 100 :=
sorry

end cubic_sum_identity_l120_120116


namespace difference_in_roi_l120_120816

theorem difference_in_roi (E_investment : ℝ) (B_investment : ℝ) (E_rate : ℝ) (B_rate : ℝ) (years : ℕ) :
  E_investment = 300 → B_investment = 500 → E_rate = 0.15 → B_rate = 0.10 → years = 2 →
  (B_rate * B_investment * years) - (E_rate * E_investment * years) = 10 :=
by
  intros E_investment_eq B_investment_eq E_rate_eq B_rate_eq years_eq
  sorry

end difference_in_roi_l120_120816


namespace initial_manufacturing_cost_l120_120365

theorem initial_manufacturing_cost
  (P : ℝ) -- selling price
  (initial_cost new_cost : ℝ)
  (initial_profit new_profit : ℝ)
  (h1 : initial_profit = 0.25 * P)
  (h2 : new_profit = 0.50 * P)
  (h3 : new_cost = 50)
  (h4 : new_profit = P - new_cost)
  (h5 : initial_profit = P - initial_cost) :
  initial_cost = 75 := 
by
  sorry

end initial_manufacturing_cost_l120_120365


namespace weekly_allowance_l120_120283

theorem weekly_allowance (A : ℝ) (h1 : A / 2 + 6 = 11) : A = 10 := 
by 
  sorry

end weekly_allowance_l120_120283


namespace perceived_temperature_difference_l120_120205

theorem perceived_temperature_difference (N : ℤ) (M L : ℤ)
  (h1 : M = L + N)
  (h2 : M - 11 - (L + 5) = 6 ∨ M - 11 - (L + 5) = -6) :
  N = 22 ∨ N = 10 := by
  sorry

end perceived_temperature_difference_l120_120205


namespace shaded_fraction_is_correct_l120_120799

-- Definitions based on the identified conditions
def initial_fraction_shaded : ℚ := 4 / 9
def geometric_series_sum (a r : ℚ) : ℚ := a / (1 - r)
def infinite_series_fraction_shaded : ℚ := 4 / 9 * (4 / 3)

-- The theorem stating the problem
theorem shaded_fraction_is_correct :
  infinite_series_fraction_shaded = 16 / 27 :=
by
  sorry -- proof to be provided

end shaded_fraction_is_correct_l120_120799


namespace number_of_children_proof_l120_120440

-- Let A be the number of mushrooms Anya has
-- Let V be the number of mushrooms Vitya has
-- Let S be the number of mushrooms Sasha has
-- Let xs be the list of mushrooms of other children

def mushrooms_distribution (A V S : ℕ) (xs : List ℕ) : Prop :=
  let n := 3 + xs.length
  -- First condition
  let total_mushrooms := A + V + S + xs.sum
  let equal_share := total_mushrooms / n
  (A / 2 = equal_share) ∧ (V + A / 2 = equal_share) ∧ (S = equal_share) ∧
  (∀ x ∈ xs, x = equal_share) ∧
  -- Second condition
  (S + A = V + xs.sum)

theorem number_of_children_proof (A V S : ℕ) (xs : List ℕ) :
  mushrooms_distribution A V S xs → 3 + xs.length = 6 :=
by
  intros h
  sorry

end number_of_children_proof_l120_120440


namespace no_absolute_winner_prob_l120_120500

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end no_absolute_winner_prob_l120_120500


namespace consecutive_days_probability_l120_120460

noncomputable def probability_of_consecutive_days : ℚ :=
  let total_days := 5
  let combinations := Nat.choose total_days 2
  let consecutive_pairs := 4
  consecutive_pairs / combinations

theorem consecutive_days_probability :
  probability_of_consecutive_days = 2 / 5 :=
by
  sorry

end consecutive_days_probability_l120_120460


namespace age_multiple_l120_120576

variables {R J K : ℕ}

theorem age_multiple (h1 : R = J + 6) (h2 : R = K + 3) (h3 : (R + 4) * (K + 4) = 108) :
  ∃ M : ℕ, R + 4 = M * (J + 4) ∧ M = 2 :=
sorry

end age_multiple_l120_120576


namespace unique_positive_real_b_l120_120379

noncomputable def is_am_gm_satisfied (r s t : ℝ) (a : ℝ) : Prop :=
  r + s + t = 2 * a ∧ r * s * t = 2 * a ∧ (r+s+t)/3 = ((r * s * t) ^ (1/3))

noncomputable def poly_roots_real (r s t : ℝ) : Prop :=
  ∀ x : ℝ, (x = r ∨ x = s ∨ x = t)

theorem unique_positive_real_b :
  ∃ b a : ℝ, 0 < a ∧ 0 < b ∧
  (∃ r s t : ℝ, (r ≥ 0 ∧ s ≥ 0 ∧ t ≥ 0 ∧ poly_roots_real r s t) ∧
   is_am_gm_satisfied r s t a ∧
   (x^3 - 2*a*x^2 + b*x - 2*a = (x - r) * (x - s) * (x - t)) ∧
   b = 9) := sorry

end unique_positive_real_b_l120_120379


namespace quadratic_root_relationship_l120_120380

theorem quadratic_root_relationship (a b c : ℝ) (α β : ℝ)
  (h1 : a ≠ 0)
  (h2 : α + β = -b / a)
  (h3 : α * β = c / a)
  (h4 : β = 3 * α) : 
  3 * b^2 = 16 * a * c :=
sorry

end quadratic_root_relationship_l120_120380


namespace find_two_digit_number_l120_120166

theorem find_two_digit_number (n : ℕ) (h1 : n % 9 = 7) (h2 : n % 7 = 5) (h3 : n % 3 = 1) (h4 : 10 ≤ n) (h5 : n < 100) : n = 61 := 
by
  sorry

end find_two_digit_number_l120_120166


namespace solution_set_of_abs_inequality_is_real_l120_120550

theorem solution_set_of_abs_inequality_is_real (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| + m - 7 > 0) ↔ m > 4 :=
by
  sorry

end solution_set_of_abs_inequality_is_real_l120_120550


namespace contrapositive_proposition_l120_120446

def proposition (x : ℝ) : Prop := x < 0 → x^2 > 0

theorem contrapositive_proposition :
  (∀ x : ℝ, proposition x) → (∀ x : ℝ, x^2 ≤ 0 → x ≥ 0) :=
by
  sorry

end contrapositive_proposition_l120_120446


namespace Harriett_total_money_l120_120174

open Real

theorem Harriett_total_money :
    let quarters := 14 * 0.25
    let dimes := 7 * 0.10
    let nickels := 9 * 0.05
    let pennies := 13 * 0.01
    let half_dollars := 4 * 0.50
    quarters + dimes + nickels + pennies + half_dollars = 6.78 :=
by
    sorry

end Harriett_total_money_l120_120174


namespace largest_number_of_gold_coins_l120_120624

theorem largest_number_of_gold_coins (n : ℕ) :
  (∃ k : ℕ, n = 13 * k + 3 ∧ n < 150) → n ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l120_120624


namespace apples_per_basket_l120_120321

theorem apples_per_basket (total_apples : ℕ) (baskets : ℕ) (h1 : total_apples = 629) (h2 : baskets = 37) :
  total_apples / baskets = 17 :=
by
  sorry

end apples_per_basket_l120_120321


namespace compare_y_values_l120_120693

variable (a : ℝ) (y₁ y₂ : ℝ)
variable (h : a > 0)
variable (p1 : y₁ = a * (-1 : ℝ)^2 - 4 * a * (-1 : ℝ) + 2)
variable (p2 : y₂ = a * (1 : ℝ)^2 - 4 * a * (1 : ℝ) + 2)

theorem compare_y_values : y₁ > y₂ :=
by {
  sorry
}

end compare_y_values_l120_120693


namespace distance_between_Jay_and_Sarah_l120_120737

theorem distance_between_Jay_and_Sarah 
  (time_in_hours : ℝ)
  (jay_speed_per_12_minutes : ℝ)
  (sarah_speed_per_36_minutes : ℝ)
  (total_distance : ℝ) :
  time_in_hours = 2 →
  jay_speed_per_12_minutes = 1 →
  sarah_speed_per_36_minutes = 3 →
  total_distance = 20 :=
by
  intros time_in_hours_eq jay_speed_eq sarah_speed_eq
  sorry

end distance_between_Jay_and_Sarah_l120_120737


namespace log12_div_log15_eq_2m_n_div_1_m_n_l120_120259

variable (m n : Real)

theorem log12_div_log15_eq_2m_n_div_1_m_n 
  (h1 : Real.log 2 = m) 
  (h2 : Real.log 3 = n) : 
  Real.log 12 / Real.log 15 = (2 * m + n) / (1 - m + n) :=
by sorry

end log12_div_log15_eq_2m_n_div_1_m_n_l120_120259


namespace minimum_obtuse_triangles_in_triangulation_of_2003gon_l120_120809

-- Defining the concepts of a polygon, a circle, and a triangulation
def polygon (n : ℕ) := { v : ℕ // 3 ≤ n }

def inscribed_circle (n : ℕ) (P : polygon n) := 
  ∃ C : ℝ × ℝ, ∀ v ∈ P, ∃ r : ℝ, (circle_equation C r v)

def triangulation (n : ℕ) (P : polygon n) := 
  ∀ t, t ∈ (triangularization P) → (obtuse t) ∨ (acute t ∨ right t)

-- Main statement we want to prove
theorem minimum_obtuse_triangles_in_triangulation_of_2003gon : 
  ∀ (P : polygon 2003), inscribed_circle 2003 P → ∃ k, k = 1999 ∧
  (∀ t ∈ (triangularization P), obtuse t → k = 1999) :=
by
  sorry

end minimum_obtuse_triangles_in_triangulation_of_2003gon_l120_120809


namespace rectangle_breadth_l120_120307

theorem rectangle_breadth (l b : ℕ) (hl : l = 15) (h : l * b = 15 * b) (h2 : l - b = 10) : b = 5 := 
sorry

end rectangle_breadth_l120_120307


namespace calculation_result_l120_120038

theorem calculation_result : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 :=
by
  sorry

end calculation_result_l120_120038


namespace collinear_probability_in_grid_l120_120067

-- Define the dimensions of the grid
def rows : ℕ := 4
def columns : ℕ := 5
def totalDots : ℕ := 20

-- Define the total number of ways to choose 4 dots from 20
def totalWaysToChoose4Dots : ℕ := Nat.choose totalDots 4

-- Define the number of sets of 4 collinear dots (horizontally, vertically, diagonally)
def collinearSets : ℕ := 17

-- Define the probability of four randomly chosen dots being collinear
def collinearProbability : ℚ := collinearSets / totalWaysToChoose4Dots

-- Main statement to be proved
theorem collinear_probability_in_grid :
  collinearProbability = 17 / 4845 := by
  sorry

end collinear_probability_in_grid_l120_120067


namespace boys_seen_l120_120853

theorem boys_seen (total_eyes : ℕ) (eyes_per_boy : ℕ) (h1 : total_eyes = 46) (h2 : eyes_per_boy = 2) : total_eyes / eyes_per_boy = 23 := 
by 
  sorry

end boys_seen_l120_120853


namespace mark_new_phone_plan_cost_l120_120006

theorem mark_new_phone_plan_cost (old_cost : ℕ) (h_old_cost : old_cost = 150) : 
  let new_cost := old_cost + (0.3 * old_cost) in 
  new_cost = 195 :=
by 
  sorry

end mark_new_phone_plan_cost_l120_120006


namespace no_d1_d2_multiple_of_7_l120_120742
open Function

theorem no_d1_d2_multiple_of_7 (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 100) :
  let d1 := a^2 + 3^a + a * 3^((a+1)/2)
  let d2 := a^2 + 3^a - a * 3^((a+1)/2)
  ¬(d1 * d2 % 7 = 0) :=
by
  let d1 := a^2 + 3^a + a * 3^((a+1)/2)
  let d2 := a^2 + 3^a - a * 3^((a+1)/2)
  sorry

end no_d1_d2_multiple_of_7_l120_120742


namespace q_value_l120_120597

-- Define the problem conditions
def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b

-- Statement of the problem
theorem q_value (p q : ℕ) (hp : prime p) (hq : prime q) (h1 : q = 13 * p + 2) (h2 : is_multiple_of (q - 1) 3) : q = 67 :=
sorry

end q_value_l120_120597


namespace problem_solution_l120_120170

theorem problem_solution : 15 * ((1 / 3 : ℚ) + (1 / 4) + (1 / 6))⁻¹ = 20 := by
  have h1 : (1 / 3 : ℚ) + (1 / 4) + (1 / 6) = 3 / 4 := sorry
  have h2 : ((3 / 4)⁻¹ : ℚ) = 4 / 3 := sorry
  calc
    15 * ((1 / 3 : ℚ) + (1 / 4) + (1 / 6))⁻¹
        = 15 * (3 / 4)⁻¹ : by rw [h1]
    ... = 15 * (4 / 3)   : by rw [h2]
    ... = 20             : by norm_num

end problem_solution_l120_120170


namespace Isabella_redeem_day_l120_120415

def is_coupon_day_closed_sunday (start_day : ℕ) (num_coupons : ℕ) (cycle_days : ℕ) : Prop :=
  ∃ n, n < num_coupons ∧ (start_day + n * cycle_days) % 7 = 0

theorem Isabella_redeem_day: 
  ∀ (day : ℕ), day ≡ 1 [MOD 7]
  → ¬ is_coupon_day_closed_sunday day 6 11 :=
by
  intro day h_mod
  simp [is_coupon_day_closed_sunday]
  sorry

end Isabella_redeem_day_l120_120415


namespace prove_mutually_exclusive_l120_120086

def bag : List String := ["red", "red", "red", "black", "black"]

def at_least_one_black (drawn : List String) : Prop :=
  "black" ∈ drawn

def all_red (drawn : List String) : Prop :=
  ∀ b ∈ drawn, b = "red"

def events_mutually_exclusive : Prop :=
  ∀ drawn, at_least_one_black drawn → ¬all_red drawn

theorem prove_mutually_exclusive :
  events_mutually_exclusive
:= by
  sorry

end prove_mutually_exclusive_l120_120086


namespace eggs_in_each_basket_l120_120752

theorem eggs_in_each_basket :
  ∃ (n : ℕ), (n ∣ 30) ∧ (n ∣ 45) ∧ (n ≥ 5) ∧
    (∀ m : ℕ, (m ∣ 30) ∧ (m ∣ 45) ∧ (m ≥ 5) → m ≤ n) ∧ n = 15 :=
by
  -- Condition 1: n divides 30
  -- Condition 2: n divides 45
  -- Condition 3: n is greater than or equal to 5
  -- Condition 4: n is the largest such divisor
  -- Therefore, n = 15
  sorry

end eggs_in_each_basket_l120_120752


namespace smallest_integer_in_set_l120_120869

def avg_seven_consecutive_integers (n : ℤ) : ℤ :=
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7

theorem smallest_integer_in_set : ∃ (n : ℤ), n = 0 ∧ (n + 6 < 3 * avg_seven_consecutive_integers n) :=
by
  sorry

end smallest_integer_in_set_l120_120869


namespace hotel_room_count_l120_120802

theorem hotel_room_count {total_lamps lamps_per_room : ℕ} (h_total_lamps : total_lamps = 147) (h_lamps_per_room : lamps_per_room = 7) : total_lamps / lamps_per_room = 21 := by
  -- We will insert this placeholder auto-proof, as the actual arithmetic proof isn't the focus.
  sorry

end hotel_room_count_l120_120802


namespace question_one_question_two_l120_120242

variable (b x : ℝ)
def f (x : ℝ) : ℝ := x^2 - b * x + 3

theorem question_one (h : f b 0 = f b 4) : ∃ x1 x2 : ℝ, f b x1 = 0 ∧ f b x2 = 0 ∧ (x1 = 3 ∧ x2 = 1) ∨ (x1 = 1 ∧ x2 = 3) := by 
  sorry

theorem question_two (h1 : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f b x1 = 0 ∧ f b x2 = 0) : b > 4 := by
  sorry

end question_one_question_two_l120_120242


namespace number_of_three_digit_integers_congruent_mod4_l120_120720

def integer_congruent_to_mod (a b n : ℕ) : Prop := ∃ k : ℤ, n = a * k + b

theorem number_of_three_digit_integers_congruent_mod4 :
  (finset.filter (λ n, integer_congruent_to_mod 4 2 (n : ℕ)) 
   (finset.Icc (100 : ℕ) (999 : ℕ))).card = 225 :=
by
  sorry

end number_of_three_digit_integers_congruent_mod4_l120_120720


namespace find_M_plus_N_l120_120258

theorem find_M_plus_N (M N : ℕ) 
  (h1 : 5 / 7 = M / 63) 
  (h2 : 5 / 7 = 70 / N) : 
  M + N = 143 :=
by
  sorry

end find_M_plus_N_l120_120258


namespace cost_of_fencing_l120_120626

noncomputable def fencingCost :=
  let π := 3.14159
  let diameter := 32
  let costPerMeter := 1.50
  let circumference := π * diameter
  let totalCost := costPerMeter * circumference
  totalCost

theorem cost_of_fencing :
  let roundedCost := (fencingCost).round
  roundedCost = 150.80 :=
by
  sorry

end cost_of_fencing_l120_120626


namespace election_winner_percentage_l120_120337

theorem election_winner_percentage :
    let votes_candidate1 := 2500
    let votes_candidate2 := 5000
    let votes_candidate3 := 15000
    let total_votes := votes_candidate1 + votes_candidate2 + votes_candidate3
    let winning_votes := votes_candidate3
    (winning_votes / total_votes) * 100 = 75 := 
by 
    sorry

end election_winner_percentage_l120_120337


namespace smallest_value_of_x_l120_120916

theorem smallest_value_of_x :
  ∃ x, (12 * x^2 - 58 * x + 70 = 0) ∧ x = 7 / 3 :=
by
  sorry

end smallest_value_of_x_l120_120916


namespace quadrants_contain_points_l120_120374

def satisfy_inequalities (x y : ℝ) : Prop :=
  y > -3 * x ∧ y > x + 2

def in_quadrant_I (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def in_quadrant_II (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem quadrants_contain_points (x y : ℝ) :
  satisfy_inequalities x y → (in_quadrant_I x y ∨ in_quadrant_II x y) :=
sorry

end quadrants_contain_points_l120_120374


namespace find_a1_l120_120266

-- Definitions of the conditions
def Sn (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of the sequence
def a₁ : ℤ := sorry          -- First term of the sequence

axiom S_2016_eq_2016 : Sn 2016 = 2016
axiom diff_seq_eq_2000 : (Sn 2016 / 2016) - (Sn 16 / 16) = 2000

-- Proof statement
theorem find_a1 : a₁ = -2014 :=
by
  -- The proof would go here
  sorry

end find_a1_l120_120266


namespace b_plus_c_neg_seven_l120_120536

theorem b_plus_c_neg_seven {A B : Set ℝ} (hA : A = {x : ℝ | x > 3 ∨ x < -1}) (hB : B = {x : ℝ | -1 ≤ x ∧ x ≤ 4})
  (h_union : A ∪ B = Set.univ) (h_inter : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 4}) :
  ∃ b c : ℝ, (∀ x, x^2 + b * x + c ≤ 0 ↔ x ∈ B) ∧ b + c = -7 :=
by
  sorry

end b_plus_c_neg_seven_l120_120536


namespace quadratic_min_max_l120_120694

noncomputable def quadratic (x : ℝ) : ℝ := (x - 3)^2 - 1

theorem quadratic_min_max :
  ∃ min max, (∀ x ∈ set.Icc 1 4, quadratic x ≥ min) ∧ min = (-1) ∧
             (∀ x ∈ set.Icc 1 4, quadratic x ≤ max) ∧ max = 3 :=
by
  sorry

end quadratic_min_max_l120_120694


namespace construct_triangle_l120_120207

variable (h_a h_b h_c : ℝ)

noncomputable def triangle_exists_and_similar :=
  ∃ (a b c : ℝ), (a = h_b) ∧ (b = h_a) ∧ (c = h_a * h_b / h_c) ∧
  (∃ (area : ℝ), area = 1/2 * a * (h_a * h_c / h_b) ∧ area = 1/2 * b * (h_b * h_c / h_a) ∧ area = 1/2 * c * h_c)

theorem construct_triangle (h_a h_b h_c : ℝ) :
  ∃ a b c, a = h_b ∧ b = h_a ∧ c = h_a * h_b / h_c ∧
  ∃ area, area = 1/2 * a * (h_a * h_c / h_b) ∧ area = 1/2 * b * (h_b * h_c / h_a) ∧ area = 1/2 * c * h_c := 
  sorry

end construct_triangle_l120_120207


namespace maximize_profit_l120_120060

noncomputable def profit_function (x : ℝ) : ℝ := -3 * x^2 + 252 * x - 4860

theorem maximize_profit :
  (∀ x : ℝ, 30 ≤ x ∧ x ≤ 54 → profit_function x ≤ 432) ∧ profit_function 42 = 432 := sorry

end maximize_profit_l120_120060


namespace train_length_l120_120353

theorem train_length
  (speed_kmph : ℕ) (time_s : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_s = 12) :
  speed_kmph * (1000 / 3600 : ℕ) * time_s = 240 :=
by
  sorry

end train_length_l120_120353


namespace base8_subtraction_l120_120823

-- Define the base 8 notation for the given numbers
def b8_256 := 256
def b8_167 := 167
def b8_145 := 145

-- Define the sum of 256_8 and 167_8 in base 8
def sum_b8 := 435

-- Define the result of subtracting 145_8 from the sum in base 8
def result_b8 := 370

-- Prove that the result of the entire operation is 370_8
theorem base8_subtraction : sum_b8 - b8_145 = result_b8 := by
  sorry

end base8_subtraction_l120_120823


namespace find_y_l120_120735

-- Conditions as definitions in Lean 4
def angle_AXB : ℝ := 180
def angle_AX : ℝ := 70
def angle_BX : ℝ := 40
def angle_CY : ℝ := 130

-- The Lean statement for the proof problem
theorem find_y (angle_AXB_eq : angle_AXB = 180)
               (angle_AX_eq : angle_AX = 70)
               (angle_BX_eq : angle_BX = 40)
               (angle_CY_eq : angle_CY = 130) : 
               ∃ y : ℝ, y = 60 :=
by
  sorry -- The actual proof goes here.

end find_y_l120_120735


namespace probability_of_drawing_white_ball_is_zero_l120_120320

theorem probability_of_drawing_white_ball_is_zero
  (red_balls blue_balls : ℕ)
  (h1 : red_balls = 3)
  (h2 : blue_balls = 5)
  (white_balls : ℕ)
  (h3 : white_balls = 0) : 
  (0 / (red_balls + blue_balls + white_balls) = 0) :=
sorry

end probability_of_drawing_white_ball_is_zero_l120_120320


namespace least_number_divisible_l120_120377

-- Define the numbers as given in the conditions
def given_number : ℕ := 3072
def divisor1 : ℕ := 57
def divisor2 : ℕ := 29
def least_number_to_add : ℕ := 234

-- Define the LCM
noncomputable def lcm_57_29 : ℕ := Nat.lcm divisor1 divisor2

-- Prove that adding least_number_to_add to given_number makes it divisible by both divisors
theorem least_number_divisible :
  (given_number + least_number_to_add) % divisor1 = 0 ∧ 
  (given_number + least_number_to_add) % divisor2 = 0 := 
by
  -- Proof should be provided here
  sorry

end least_number_divisible_l120_120377


namespace remainder_proof_l120_120036

theorem remainder_proof : 1234567 % 12 = 7 := sorry

end remainder_proof_l120_120036


namespace sugar_needed_for_40_cookies_l120_120874

def num_cookies_per_cup_flour (a : ℕ) (b : ℕ) : ℕ := a / b

def cups_of_flour_needed (num_cookies : ℕ) (cookies_per_cup : ℕ) : ℕ := num_cookies / cookies_per_cup

def cups_of_sugar_needed (cups_flour : ℕ) (flour_to_sugar_ratio_num : ℕ) (flour_to_sugar_ratio_denom : ℕ) : ℚ := 
  (flour_to_sugar_ratio_denom * cups_flour : ℚ) / flour_to_sugar_ratio_num

theorem sugar_needed_for_40_cookies :
  let num_flour_to_make_24_cookies := 3
  let cookies := 24
  let ratio_num := 3
  let ratio_denom := 2
  num_cookies_per_cup_flour cookies num_flour_to_make_24_cookies = 8 →
  cups_of_flour_needed 40 8 = 5 →
  cups_of_sugar_needed 5 ratio_num ratio_denom = 10 / 3 :=
by 
  sorry

end sugar_needed_for_40_cookies_l120_120874


namespace roberto_outfits_l120_120016

-- Define the conditions
def trousers := 5
def shirts := 8
def jackets := 4

-- Define the total number of outfits
def total_outfits : ℕ := trousers * shirts * jackets

-- The theorem stating the actual problem and answer
theorem roberto_outfits : total_outfits = 160 :=
by
  -- skip the proof for now
  sorry

end roberto_outfits_l120_120016


namespace equidistant_point_x_coord_l120_120933

theorem equidistant_point_x_coord :
  ∃ x y : ℝ, y = x ∧ dist (x, y) (x, 0) = dist (x, y) (0, y) ∧ dist (x, y) (0, y) = dist (x, y) (x, 5 - x)
    → x = 5 / 2 :=
by sorry

end equidistant_point_x_coord_l120_120933


namespace num_integer_terms_sequence_l120_120523

noncomputable def sequence_starting_at_8820 : Nat := 8820

def divide_by_5 (n : Nat) : Nat := n / 5

theorem num_integer_terms_sequence :
  let seq := [sequence_starting_at_8820, divide_by_5 sequence_starting_at_8820]
  seq = [8820, 1764] →
  seq.length = 2 := by
  sorry

end num_integer_terms_sequence_l120_120523


namespace selling_price_of_book_l120_120347

   theorem selling_price_of_book
     (cost_price : ℝ)
     (profit_rate : ℝ)
     (profit := (profit_rate / 100) * cost_price)
     (selling_price := cost_price + profit)
     (hp : cost_price = 50)
     (hr : profit_rate = 60) :
     selling_price = 80 := sorry
   
end selling_price_of_book_l120_120347


namespace line_equation_l120_120315

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_sq := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_sq) • u

theorem line_equation :
  ∀ (x y : ℝ), projection (4, 3) (x, y) = (-4, -3) → y = (-4 / 3) * x - 25 / 3 :=
by
  intros x y h
  sorry

end line_equation_l120_120315


namespace product_roots_example_l120_120206

def cubic_eq (a b c d : ℝ) (x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

noncomputable def product_of_roots (a b c d : ℝ) : ℝ := -d / a

theorem product_roots_example : product_of_roots 4 (-2) (-25) 36 = -9 := by
  sorry

end product_roots_example_l120_120206


namespace pyramid_volume_QEFGH_l120_120571

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * EF * FG * QE

theorem pyramid_volume_QEFGH :
  let EF := 10
  let FG := 5
  let QE := 9
  volume_of_pyramid EF FG QE = 150 := by
  sorry

end pyramid_volume_QEFGH_l120_120571


namespace sector_COD_area_ratio_l120_120427

-- Define the given angles
def angle_AOC : ℝ := 30
def angle_DOB : ℝ := 45
def angle_AOB : ℝ := 180

-- Define the full circle angle
def full_circle_angle : ℝ := 360

-- Calculate the angle COD
def angle_COD : ℝ := angle_AOB - angle_AOC - angle_DOB

-- State the ratio of the area of sector COD to the area of the circle
theorem sector_COD_area_ratio :
  angle_COD / full_circle_angle = 7 / 24 := by
  sorry

end sector_COD_area_ratio_l120_120427


namespace average_fish_per_person_l120_120856

theorem average_fish_per_person (Aang Sokka Toph : ℕ) 
  (haang : Aang = 7) (hsokka : Sokka = 5) (htoph : Toph = 12) : 
  (Aang + Sokka + Toph) / 3 = 8 := by
  sorry

end average_fish_per_person_l120_120856


namespace ROI_difference_l120_120814

-- Definitions based on the conditions
def Emma_investment : ℝ := 300
def Briana_investment : ℝ := 500
def Emma_yield : ℝ := 0.15
def Briana_yield : ℝ := 0.10
def years : ℕ := 2

-- The goal is to prove that the difference between their 2-year ROI is $10
theorem ROI_difference :
  let Emma_ROI := Emma_investment * Emma_yield * years
  let Briana_ROI := Briana_investment * Briana_yield * years
  (Briana_ROI - Emma_ROI) = 10 :=
by
  sorry

end ROI_difference_l120_120814


namespace find_p_q_r_sum_l120_120235

theorem find_p_q_r_sum (p q r : ℕ) (hpq_rel_prime : Nat.gcd p q = 1) (hq_nonzero : q ≠ 0) 
  (h1 : ∃ t, (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4) 
  (h2 : ∃ t, (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r) : 
  p + q + r = 7 :=
sorry

end find_p_q_r_sum_l120_120235


namespace john_ate_10_chips_l120_120872

variable (c p : ℕ)

/-- Given the total calories from potato chips and the calories increment of cheezits,
prove the number of potato chips John ate. -/
theorem john_ate_10_chips (h₀ : p * c = 60)
  (h₁ : ∃ c_cheezit, (c_cheezit = (4 / 3 : ℝ) * c))
  (h₂ : ∀ c_cheezit, p * c + 6 * c_cheezit = 108) :
  p = 10 :=
by {
  sorry
}

end john_ate_10_chips_l120_120872


namespace triangle_area_l120_120272

open Real

-- Define the conditions
variables (a : ℝ) (B : ℝ) (cosA : ℝ)
variable (S : ℝ)

-- Given conditions of the problem
def triangle_conditions : Prop :=
  a = 5 ∧ B = π / 3 ∧ cosA = 11 / 14

-- State the theorem to be proved
theorem triangle_area (h : triangle_conditions a B cosA) : S = 10 * sqrt 3 :=
sorry

end triangle_area_l120_120272


namespace factor_difference_of_squares_l120_120073

theorem factor_difference_of_squares (x : ℝ) :
  x^2 - 169 = (x - 13) * (x + 13) := by
  have h : 169 = 13^2 := by norm_num
  rw h
  exact by ring

end factor_difference_of_squares_l120_120073


namespace width_decrease_l120_120637

-- Given conditions and known values
variable (L W : ℝ) -- original length and width
variable (P : ℝ)   -- percentage decrease in width

-- The known condition for the area comparison
axiom area_condition : 1.4 * (L * (W * (1 - P / 100))) = 1.1199999999999999 * (L * W)

-- The property we want to prove
theorem width_decrease (L W: ℝ) (h : L > 0) (h1 : W > 0) :
  P = 20 := 
by
  sorry

end width_decrease_l120_120637


namespace trapezium_distance_l120_120820

theorem trapezium_distance (h : ℝ) (a b A : ℝ) 
  (h_area : A = 95) (h_a : a = 20) (h_b : b = 18) :
  A = (1/2 * (a + b) * h) → h = 5 :=
by
  sorry

end trapezium_distance_l120_120820


namespace total_amount_proof_l120_120551

def total_shared_amount : ℝ :=
  let z := 250
  let y := 1.20 * z
  let x := 1.25 * y
  x + y + z

theorem total_amount_proof : total_shared_amount = 925 :=
by
  sorry

end total_amount_proof_l120_120551


namespace first_ring_time_l120_120838

-- Define the properties of the clock
def rings_every_three_hours : Prop := ∀ n : ℕ, 3 * n < 24
def rings_eight_times_a_day : Prop := ∀ n : ℕ, n = 8 → 3 * n = 24

-- The theorem statement
theorem first_ring_time : rings_every_three_hours → rings_eight_times_a_day → (∀ n : ℕ, n = 1 → 3 * n = 3) := 
    sorry

end first_ring_time_l120_120838


namespace find_x_y_l120_120422

theorem find_x_y 
  (x y : ℚ)
  (h1 : (x / 6) * 12 = 10)
  (h2 : (y / 4) * 8 = x) :
  x = 5 ∧ y = 2.5 :=
by
  sorry

end find_x_y_l120_120422


namespace different_result_l120_120470

theorem different_result :
  let A := -2 - (-3)
  let B := 2 - 3
  let C := -3 + 2
  let D := -3 - (-2)
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B = C ∧ B = D :=
by
  sorry

end different_result_l120_120470


namespace num_divisors_47_gt_6_l120_120256

theorem num_divisors_47_gt_6 : (finset.filter (λ d, d > 6) (finset.divisors 47)).card = 1 :=
by 
  sorry

end num_divisors_47_gt_6_l120_120256


namespace find_meeting_time_l120_120880

-- Define the context and the problem parameters
def lisa_speed : ℝ := 9  -- Lisa's speed in mph
def adam_speed : ℝ := 7  -- Adam's speed in mph
def initial_distance : ℝ := 6  -- Initial distance in miles

-- The time in minutes for Lisa to meet Adam
theorem find_meeting_time : (initial_distance / (lisa_speed + adam_speed)) * 60 = 22.5 := by
  -- The proof is omitted for this statement
  sorry

end find_meeting_time_l120_120880


namespace area_of_circumcircle_of_isosceles_triangle_l120_120481

theorem area_of_circumcircle_of_isosceles_triangle :
  let AB := 4
  let AC := 4
  let BC := 3
  let AD := (√(AB^2 - (BC/2)^2))
  let radius := AD
  let area := π * radius^2
  area = 16 * π :=
by
  sorry

end area_of_circumcircle_of_isosceles_triangle_l120_120481


namespace system1_solution_system2_solution_l120_120581

-- System (1)
theorem system1_solution (x y : ℚ) (h1 : 3 * y - 4 * x = 0) (h2 : 4 * x + y = 8) : 
  x = 3 / 2 ∧ y = 2 :=
by
  sorry

-- System (2)
theorem system2_solution (x y : ℚ) (h1 : x + y = 3) (h2 : (x - 1) / 4 + y / 2 = 3 / 4) : 
  x = 2 ∧ y = 1 :=
by
  sorry

end system1_solution_system2_solution_l120_120581


namespace no_common_elements_in_sequences_l120_120911

theorem no_common_elements_in_sequences :
  ∀ (k : ℕ), (∃ n : ℕ, k = n^2 - 1) ∧ (∃ m : ℕ, k = m^2 + 1) → False :=
by sorry

end no_common_elements_in_sequences_l120_120911


namespace sun_xing_zhe_problem_l120_120046

theorem sun_xing_zhe_problem (S X Z : ℕ) (h : S < 10 ∧ X < 10 ∧ Z < 10)
  (hprod : (100 * S + 10 * X + Z) * (100 * Z + 10 * X + S) = 78445) :
  (100 * S + 10 * X + Z) + (100 * Z + 10 * X + S) = 1372 := 
by
  sorry

end sun_xing_zhe_problem_l120_120046


namespace left_handed_jazz_lovers_count_l120_120444

noncomputable def club_members := 30
noncomputable def left_handed := 11
noncomputable def like_jazz := 20
noncomputable def right_handed_dislike_jazz := 4

theorem left_handed_jazz_lovers_count : 
  ∃ x, x + (left_handed - x) + (like_jazz - x) + right_handed_dislike_jazz = club_members ∧ x = 5 :=
by
  sorry

end left_handed_jazz_lovers_count_l120_120444


namespace right_triangle_area_l120_120015

theorem right_triangle_area (a : ℝ) (r : ℝ) (area : ℝ) :
  a = 3 → r = 3 / 8 → area = 21 / 16 :=
by 
  sorry

end right_triangle_area_l120_120015


namespace find_bicycle_speed_l120_120297

-- Let's define the conditions first
def distance := 10  -- Distance in km
def time_diff := 1 / 3  -- Time difference in hours
def speed_of_bicycle (x : ℝ) := x
def speed_of_car (x : ℝ) := 2 * x

-- Prove the equation using the given conditions
theorem find_bicycle_speed (x : ℝ) (h : x ≠ 0) :
  (distance / speed_of_bicycle x) = (distance / speed_of_car x) + time_diff :=
by {
  sorry
}

end find_bicycle_speed_l120_120297


namespace inradius_triangle_l120_120313

theorem inradius_triangle (p A : ℝ) (h1 : p = 39) (h2 : A = 29.25) :
  ∃ r : ℝ, A = (1 / 2) * r * p ∧ r = 1.5 := by
  sorry

end inradius_triangle_l120_120313


namespace scientific_notation_correct_l120_120298

-- Defining the given number in terms of its scientific notation components.
def million : ℝ := 10^6
def num_million : ℝ := 15.276

-- Expressing the number 15.276 million using its definition.
def fifteen_point_two_seven_six_million : ℝ := num_million * million

-- Scientific notation representation to be proved.
def scientific_notation : ℝ := 1.5276 * 10^7

-- The theorem statement.
theorem scientific_notation_correct :
  fifteen_point_two_seven_six_million = scientific_notation :=
by
  sorry

end scientific_notation_correct_l120_120298


namespace Dacid_weighted_average_l120_120068

noncomputable def DacidMarks := 86 * 3 + 85 * 4 + 92 * 4 + 87 * 3 + 95 * 3 + 89 * 2 + 75 * 1
noncomputable def TotalCreditHours := 3 + 4 + 4 + 3 + 3 + 2 + 1
noncomputable def WeightedAverageMarks := (DacidMarks : ℝ) / (TotalCreditHours : ℝ)

theorem Dacid_weighted_average :
  WeightedAverageMarks = 88.25 :=
sorry

end Dacid_weighted_average_l120_120068


namespace car_trip_cost_proof_l120_120950

def car_trip_cost 
  (d1 d2 d3 d4 : ℕ) 
  (efficiency : ℕ) 
  (cost_per_gallon : ℕ) 
  (total_distance : ℕ) 
  (gallons_used : ℕ) 
  (cost : ℕ) : Prop :=
  d1 = 8 ∧
  d2 = 6 ∧
  d3 = 12 ∧
  d4 = 2 * d3 ∧
  efficiency = 25 ∧
  cost_per_gallon = 250 ∧
  total_distance = d1 + d2 + d3 + d4 ∧
  gallons_used = total_distance / efficiency ∧
  cost = gallons_used * cost_per_gallon ∧
  cost = 500

theorem car_trip_cost_proof : car_trip_cost 8 6 12 (2 * 12) 25 250 (8 + 6 + 12 + (2 * 12)) ((8 + 6 + 12 + (2 * 12)) / 25) (((8 + 6 + 12 + (2 * 12)) / 25) * 250) :=
by 
  sorry

end car_trip_cost_proof_l120_120950


namespace total_time_is_10_l120_120800

-- Definitions based on conditions
def total_distance : ℕ := 224
def first_half_distance : ℕ := total_distance / 2
def second_half_distance : ℕ := total_distance / 2
def speed_first_half : ℕ := 21
def speed_second_half : ℕ := 24

-- Definition of time taken for each half of the journey
def time_first_half : ℚ := first_half_distance / speed_first_half
def time_second_half : ℚ := second_half_distance / speed_second_half

-- Total time is the sum of time taken for each half
def total_time : ℚ := time_first_half + time_second_half

-- Theorem stating the total time taken for the journey
theorem total_time_is_10 : total_time = 10 := by
  sorry

end total_time_is_10_l120_120800


namespace largest_gold_coins_l120_120619

noncomputable def max_gold_coins (n : ℕ) : ℕ :=
  if h : ∃ k : ℕ, n = 13 * k + 3 ∧ n < 150 then
    n
  else 0

theorem largest_gold_coins : max_gold_coins 146 = 146 :=
by
  sorry

end largest_gold_coins_l120_120619


namespace cody_spent_tickets_l120_120945

theorem cody_spent_tickets (initial_tickets lost_tickets remaining_tickets : ℝ) (h1 : initial_tickets = 49.0) (h2 : lost_tickets = 6.0) (h3 : remaining_tickets = 18.0) :
  initial_tickets - lost_tickets - remaining_tickets = 25.0 :=
by
  sorry

end cody_spent_tickets_l120_120945


namespace decrement_from_observation_l120_120900

theorem decrement_from_observation 
  (n : ℕ) (mean_original mean_updated : ℚ)
  (h1 : n = 50)
  (h2 : mean_original = 200)
  (h3 : mean_updated = 194)
  : (mean_original - mean_updated) = 6 :=
by
  sorry

end decrement_from_observation_l120_120900


namespace Nadine_pebbles_l120_120009

theorem Nadine_pebbles :
  ∀ (white red blue green x : ℕ),
    white = 20 →
    red = white / 2 →
    blue = red / 3 →
    green = blue + 5 →
    red = (1/5) * x →
    x = 50 :=
by
  intros white red blue green x h_white h_red h_blue h_green h_percentage
  sorry

end Nadine_pebbles_l120_120009


namespace algebraic_expression_equivalence_l120_120664

theorem algebraic_expression_equivalence (x : ℝ) : 
  x^2 - 6*x + 10 = (x - 3)^2 + 1 := 
by 
  sorry

end algebraic_expression_equivalence_l120_120664


namespace cost_of_article_l120_120627

variable (C G : ℝ)

theorem cost_of_article (h1 : 340 = C + G) (h2 : 350 = C + G + 0.05 * G) : C = 140 :=
by
  have h3 : 0.05 * G = 10 :=
    calc 0.05 * G = (350 - 340) : by linarith [h2, h1]
                 ... = 10 : by norm_num
  have h4 : G = 200 := by linarith
  linarith

end cost_of_article_l120_120627


namespace angle_complement_l120_120397

-- Conditions: The complement of angle A is 60 degrees
def complement (α : ℝ) : ℝ := 90 - α 

theorem angle_complement (A : ℝ) : complement A = 60 → A = 30 :=
by
  sorry

end angle_complement_l120_120397


namespace frankie_pets_total_l120_120384

theorem frankie_pets_total
  (C S P D : ℕ)
  (h_snakes : S = C + 6)
  (h_parrots : P = C - 1)
  (h_dogs : D = 2)
  (h_total : C + S + P + D = 19) :
  C + (C + 6) + (C - 1) + 2 = 19 := by
  sorry

end frankie_pets_total_l120_120384


namespace find_angle_y_l120_120413

open Real

theorem find_angle_y 
    (angle_ABC angle_BAC : ℝ)
    (h1 : angle_ABC = 70)
    (h2 : angle_BAC = 50)
    (triangle_sum : ∀ {A B C : ℝ}, A + B + C = 180)
    (right_triangle_sum : ∀ D E : ℝ, D + E = 90) :
    30 = 30 :=
by
    -- Given, conditions, and intermediate results (skipped)
    sorry

end find_angle_y_l120_120413


namespace sin_alpha_plus_beta_eq_33_by_65_l120_120230

theorem sin_alpha_plus_beta_eq_33_by_65 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (hcosα : Real.cos α = 12 / 13) 
  (hcos_2α_β : Real.cos (2 * α + β) = 3 / 5) :
  Real.sin (α + β) = 33 / 65 := 
by 
  sorry

end sin_alpha_plus_beta_eq_33_by_65_l120_120230


namespace eraser_cost_l120_120325

theorem eraser_cost (initial_money : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) (erasers_count : ℕ) (remaining_money : ℕ) :
    initial_money = 100 →
    scissors_count = 8 →
    scissors_price = 5 →
    erasers_count = 10 →
    remaining_money = 20 →
    (initial_money - scissors_count * scissors_price - remaining_money) / erasers_count = 4 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end eraser_cost_l120_120325


namespace brad_start_time_after_maxwell_l120_120294

-- Assuming time is measured in hours, distance in kilometers, and speed in km/h
def meet_time (d : ℕ) (v_m : ℕ) (v_b : ℕ) (t_m : ℕ) : ℕ :=
  let d_m := t_m * v_m
  let t_b := t_m - 1
  let d_b := t_b * v_b
  d_m + d_b

theorem brad_start_time_after_maxwell (d : ℕ) (v_m : ℕ) (v_b : ℕ) (t_m : ℕ) :
  d = 54 → v_m = 4 → v_b = 6 → t_m = 6 → 
  meet_time d v_m v_b t_m = 54 :=
by
  intros hd hv_m hv_b ht_m
  have : meet_time d v_m v_b t_m = t_m * v_m + (t_m - 1) * v_b := rfl
  rw [hd, hv_m, hv_b, ht_m] at this
  sorry

end brad_start_time_after_maxwell_l120_120294


namespace parabola_equation_l120_120029

theorem parabola_equation (vertex focus : ℝ × ℝ) 
  (h_vertex : vertex = (0, 0)) 
  (h_focus_line : ∃ x y : ℝ, focus = (x, y) ∧ x - y + 2 = 0) 
  (h_symmetry_axis : ∃ axis : ℝ × ℝ → ℝ, ∀ p : ℝ × ℝ, axis p = 0): 
  ∃ k : ℝ, k > 0 ∧ (∀ x y : ℝ, y^2 = -8*x ∨ x^2 = 8*y) :=
by {
  sorry
}

end parabola_equation_l120_120029


namespace no_primes_in_range_l120_120991

theorem no_primes_in_range (n : ℕ) (hn : n > 2) : 
  ∀ k, n! + 2 < k ∧ k < n! + n + 1 → ¬Prime k := 
sorry

end no_primes_in_range_l120_120991


namespace school_survey_l120_120641

theorem school_survey (n k smallest largest : ℕ) (h1 : n = 24) (h2 : k = 4) (h3 : smallest = 3) (h4 : 1 ≤ smallest ∧ smallest ≤ n) (h5 : largest - smallest = (k - 1) * (n / k)) : 
  largest = 21 :=
by {
  sorry
}

end school_survey_l120_120641


namespace domain_of_log_base_5_range_of_3_pow_neg_l120_120153

theorem domain_of_log_base_5 (x : ℝ) : (1 - x > 0) -> (x < 1) :=
sorry

theorem range_of_3_pow_neg (y : ℝ) : (∃ x : ℝ, y = 3 ^ (-x)) -> (y > 0) :=
sorry

end domain_of_log_base_5_range_of_3_pow_neg_l120_120153


namespace selling_price_of_cycle_l120_120786

theorem selling_price_of_cycle (cp : ℝ) (loss_percentage : ℝ) (sp : ℝ) : 
  cp = 1400 → loss_percentage = 20 → sp = cp - (loss_percentage / 100) * cp → sp = 1120 :=
by 
  intro h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end selling_price_of_cycle_l120_120786


namespace banana_pieces_l120_120322

theorem banana_pieces (B G P : ℕ) 
  (h1 : P = 4 * G)
  (h2 : G = B + 5)
  (h3 : P = 192) : B = 43 := 
by
  sorry

end banana_pieces_l120_120322


namespace percentage_increase_in_y_l120_120147

variable (x y k q : ℝ) (h1 : x * y = k) (h2 : x' = x * (1 - q / 100))

theorem percentage_increase_in_y (h1 : x * y = k) (h2 : x' = x * (1 - q / 100)) :
  (y * 100 / (100 - q) - y) / y * 100 = (100 * q) / (100 - q) :=
by
  sorry

end percentage_increase_in_y_l120_120147


namespace largest_coins_l120_120615

theorem largest_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n = 146 :=
by
  sorry

end largest_coins_l120_120615


namespace number_of_children_l120_120442

theorem number_of_children (A V S : ℕ) (x : ℕ → ℕ) (n : ℕ) 
  (h1 : (A / 2) + V = (A + V + S + (Finset.range (n - 3)).sum x) / n)
  (h2 : S + A = V + (Finset.range (n - 3)).sum x) : 
  n = 6 :=
sorry

end number_of_children_l120_120442


namespace sequence_n_5_l120_120084

theorem sequence_n_5 (a : ℤ) (n : ℕ → ℤ) 
  (h1 : ∀ i > 1, n i = 2 * n (i - 1) + a)
  (h2 : n 2 = 5)
  (h3 : n 8 = 257) : n 5 = 33 :=
by
  sorry

end sequence_n_5_l120_120084


namespace number_of_three_digit_integers_congruent_mod4_l120_120723

def integer_congruent_to_mod (a b n : ℕ) : Prop := ∃ k : ℤ, n = a * k + b

theorem number_of_three_digit_integers_congruent_mod4 :
  (finset.filter (λ n, integer_congruent_to_mod 4 2 (n : ℕ)) 
   (finset.Icc (100 : ℕ) (999 : ℕ))).card = 225 :=
by
  sorry

end number_of_three_digit_integers_congruent_mod4_l120_120723


namespace range_of_x_satisfies_conditions_l120_120393

theorem range_of_x_satisfies_conditions (x : ℝ) (h : x^2 - 4 < 0 ∨ |x| = 2) : -2 ≤ x ∧ x ≤ 2 := 
by
  sorry

end range_of_x_satisfies_conditions_l120_120393


namespace find_m_given_root_of_quadratic_l120_120239

theorem find_m_given_root_of_quadratic (m : ℝ) : (∃ x : ℝ, x = 3 ∧ x^2 - m * x - 6 = 0) → m = 1 := 
by
  sorry

end find_m_given_root_of_quadratic_l120_120239


namespace geometric_sequence_third_term_l120_120674

theorem geometric_sequence_third_term (a b c d : ℕ) (r : ℕ) 
  (h₁ : d * r = 81) 
  (h₂ : 81 * r = 243) 
  (h₃ : r = 3) : c = 27 :=
by
  -- Insert proof here
  sorry

end geometric_sequence_third_term_l120_120674


namespace permutations_eq_factorial_l120_120189

theorem permutations_eq_factorial (n : ℕ) : 
  (∃ Pn : ℕ, Pn = n!) := 
sorry

end permutations_eq_factorial_l120_120189


namespace eq_x_add_q_l120_120855

theorem eq_x_add_q (x q : ℝ) (h1 : abs (x - 5) = q) (h2 : x > 5) : x + q = 5 + 2*q :=
by {
  sorry
}

end eq_x_add_q_l120_120855


namespace triangle_angle_y_l120_120035

theorem triangle_angle_y (y : ℝ) (h : y + 3 * y + 45 = 180) : y = 33.75 :=
by
  have h1 : 4 * y + 45 = 180 := by sorry
  have h2 : 4 * y = 135 := by sorry
  have h3 : y = 33.75 := by sorry
  exact h3

end triangle_angle_y_l120_120035


namespace carol_packs_l120_120065

theorem carol_packs (n_invites n_per_pack : ℕ) (h1 : n_invites = 12) (h2 : n_per_pack = 4) : n_invites / n_per_pack = 3 :=
by
  sorry

end carol_packs_l120_120065


namespace original_price_of_cupcakes_l120_120208

theorem original_price_of_cupcakes
  (revenue : ℕ := 32) 
  (cookies_sold : ℕ := 8) 
  (cupcakes_sold : ℕ := 16) 
  (cookie_price: ℕ := 2)
  (half_price_of_cookie: ℕ := 1) :
  (x : ℕ) → (16 * (x / 2)) + (8 * 1) = 32 → x = 3 := 
by
  sorry

end original_price_of_cupcakes_l120_120208


namespace sum_in_base_8_l120_120079

theorem sum_in_base_8 (a b : ℕ) (h_a : a = 3 * 8^2 + 2 * 8 + 7)
                                  (h_b : b = 7 * 8 + 3) :
  (a + b) = 4 * 8^2 + 2 * 8 + 2 :=
by
  sorry

end sum_in_base_8_l120_120079


namespace k_value_l120_120119

theorem k_value (k : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |k * x - 4| ≤ 2) → k = 2 := 
by
  intros h
  sorry

end k_value_l120_120119


namespace pyramid_four_triangular_faces_area_l120_120607

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l120_120607


namespace factorial_div_sum_l120_120951

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l120_120951


namespace train_length_l120_120351

theorem train_length (speed_km_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_hr = 72) (h_time : time_sec = 12) : 
  ∃ length_m : ℕ, length_m = 240 := 
by
  sorry

end train_length_l120_120351


namespace pyramid_face_area_total_l120_120603

theorem pyramid_face_area_total 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base_edge : base_edge = 8) 
  (h_lateral_edge : lateral_edge = 7) : 
  4 * (1 / 2 * base_edge * real.sqrt (lateral_edge^2 - (base_edge / 2)^2)) = 16 * real.sqrt 33 :=
by
  sorry

end pyramid_face_area_total_l120_120603


namespace find_a_l120_120997

theorem find_a (a : ℤ) : 0 ≤ a ∧ a ≤ 13 ∧ (51^2015 + a) % 13 = 0 → a = 1 :=
by { sorry }

end find_a_l120_120997


namespace carla_highest_final_number_l120_120649

def alice_final_number (initial : ℕ) : ℕ :=
  let step1 := initial * 2
  let step2 := step1 - 3
  let step3 := step2 / 3
  step3 + 4

def bob_final_number (initial : ℕ) : ℕ :=
  let step1 := initial + 5
  let step2 := step1 * 2
  let step3 := step2 - 4
  step3 / 2

def carla_final_number (initial : ℕ) : ℕ :=
  let step1 := initial - 2
  let step2 := step1 * 2
  let step3 := step2 + 3
  step3 * 2

theorem carla_highest_final_number : carla_final_number 12 > bob_final_number 12 ∧ carla_final_number 12 > alice_final_number 12 :=
  by
  have h_alice : alice_final_number 12 = 11 := by rfl
  have h_bob : bob_final_number 12 = 15 := by rfl
  have h_carla : carla_final_number 12 = 46 := by rfl
  sorry

end carla_highest_final_number_l120_120649


namespace water_displaced_volume_square_l120_120932

-- Given conditions:
def radius : ℝ := 5
def height : ℝ := 10
def cube_side : ℝ := 6

-- Theorem statement for the problem
theorem water_displaced_volume_square (r h s : ℝ) (w : ℝ) 
  (hr : r = 5) 
  (hh : h = 10) 
  (hs : s = 6) : 
  (w * w) = 13141.855 :=
by 
  sorry

end water_displaced_volume_square_l120_120932


namespace box_growth_factor_l120_120328

/-
Problem: When a large box in the shape of a cuboid measuring 6 centimeters (cm) wide,
4 centimeters (cm) long, and 1 centimeters (cm) high became larger into a volume of
30 centimeters (cm) wide, 20 centimeters (cm) long, and 5 centimeters (cm) high,
find how many times it has grown.
-/

def original_box_volume (w l h : ℕ) : ℕ := w * l * h
def larger_box_volume (w l h : ℕ) : ℕ := w * l * h

theorem box_growth_factor :
  original_box_volume 6 4 1 * 125 = larger_box_volume 30 20 5 :=
by
  -- Proof goes here
  sorry

end box_growth_factor_l120_120328


namespace disc_thickness_l120_120642

theorem disc_thickness (r_sphere : ℝ) (r_disc : ℝ) (h : ℝ)
  (h_radius_sphere : r_sphere = 3)
  (h_radius_disc : r_disc = 10)
  (h_volume_constant : (4/3) * Real.pi * r_sphere^3 = Real.pi * r_disc^2 * h) :
  h = 9 / 25 :=
by
  sorry

end disc_thickness_l120_120642


namespace work_problem_l120_120182

theorem work_problem (x : ℕ) (b_work : ℕ) (a_b_together_work : ℕ) (h1: b_work = 24) (h2: a_b_together_work = 8) :
  (1 / x) + (1 / b_work) = (1 / a_b_together_work) → x = 12 :=
by 
  intros h_eq
  have h_b : b_work = 24 := h1
  have h_ab : a_b_together_work = 8 := h2
  -- Full proof is omitted
  sorry

end work_problem_l120_120182


namespace max_value_sincos_sum_l120_120217

theorem max_value_sincos_sum (x y z : ℝ) :
  (∀ x y z, (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5) :=
by sorry

end max_value_sincos_sum_l120_120217


namespace complete_the_square_l120_120917

theorem complete_the_square :
  ∀ (x : ℝ), (x^2 + 14 * x + 24 = 0) → (∃ c d : ℝ, (x + c)^2 = d ∧ d = 25) :=
by
  intro x h
  sorry

end complete_the_square_l120_120917


namespace range_of_a_l120_120689

theorem range_of_a : 
  ∀ (a : ℝ), 
  (∀ (x : ℝ), ((a^2 - 1) * x^2 + (a + 1) * x + 1) > 0) → 1 ≤ a ∧ a ≤ 5 / 3 := 
by
  sorry

end range_of_a_l120_120689


namespace quadratic_solution_sum_l120_120458

theorem quadratic_solution_sum (m n p : ℕ) (h : m.gcd (n.gcd p) = 1)
  (h₀ : ∀ x, x * (5 * x - 11) = -6 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :
  m + n + p = 70 :=
sorry

end quadratic_solution_sum_l120_120458


namespace isosceles_triangle_circumcircle_area_l120_120478

noncomputable def area_of_circumcircle (a b c : ℝ) : ℝ :=
  let BD := real.sqrt (a^2 - ((c / 2)^2))
  let OD := (2 / 3) * BD
  let r := real.sqrt (a^2 - OD^2)
  real.pi * r^2

theorem isosceles_triangle_circumcircle_area :
  area_of_circumcircle 4 4 3 = 9.8889 * real.pi :=
sorry

end isosceles_triangle_circumcircle_area_l120_120478


namespace part1_mean_and_variance_part2_probability_l120_120121

noncomputable def mean_and_variance_for_20_students (μ_A σ2_A μ_B σ2_B : ℝ) (n_A n_B : ℝ) : ℝ × ℝ :=
let μ := (n_A * μ_A + n_B * μ_B) / (n_A + n_B),
    σ2 := (n_A * (σ2_A + (μ_A - μ)^2) + n_B * (σ2_B + (μ_B - μ)^2)) / (n_A + n_B) in
(μ, σ2)

theorem part1_mean_and_variance :
  mean_and_variance_for_20_students 1 1 1.5 0.25 12 8 = (1.2, 0.76) :=
by sorry

-- Probabilities of drawing questions in sequences
noncomputable def probability_A (p1 p2 p3 : ℝ) (pA_given_b1 pA_given_b2 pA_given_b3 : ℝ) : ℝ :=
p1 * pA_given_b1 + p2 * pA_given_b2 + p3 * pA_given_b3

noncomputable def conditional_probability (p : ℝ) (p_given : ℝ) : ℝ :=
(p_given * p) / p

theorem part2_probability :
  let p1 := 2 / 5,
      p2 := 8 / 15,
      p3 := 1 / 15,
      pA_given_b1 := 5 / 8,
      pA_given_b2 := 8 / 15,
      pA_given_b3 := 3 / 8 in
  conditional_probability (probability_A p1 p2 p3 pA_given_b1 pA_given_b2 pA_given_b3) pA_given_b1 = 6 / 13 :=
by sorry

end part1_mean_and_variance_part2_probability_l120_120121


namespace compute_value_l120_120972

def Δ (p q : ℕ) : ℕ := p^3 - q

theorem compute_value : Δ (5^Δ 2 7) (4^Δ 4 8) = 125 - 4^56 := by
  sorry

end compute_value_l120_120972


namespace division_value_l120_120188

theorem division_value (x : ℚ) (h : (5 / 2) / x = 5 / 14) : x = 7 :=
sorry

end division_value_l120_120188


namespace tensor_A_B_eq_l120_120659

-- Define sets A and B
def A : Set ℕ := {0, 2}
def B : Set ℕ := {x | x^2 - 3 * x + 2 = 0}

-- Define set operation ⊗
def tensor (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

-- Prove that A ⊗ B = {0, 2, 4}
theorem tensor_A_B_eq : tensor A B = {0, 2, 4} :=
by
  sorry

end tensor_A_B_eq_l120_120659


namespace fraction_finding_l120_120913

theorem fraction_finding (x : ℝ) (h : (3 / 4) * x * (2 / 3) = 0.4) : x = 0.8 :=
sorry

end fraction_finding_l120_120913


namespace train_speed_excluding_stoppages_l120_120663

-- Define the speed of the train excluding stoppages and including stoppages
variables (S : ℕ) -- S is the speed of the train excluding stoppages
variables (including_stoppages_speed : ℕ := 40) -- The speed including stoppages is 40 kmph

-- The train stops for 20 minutes per hour. This means it runs for (60 - 20) minutes per hour.
def running_time_per_hour := 40

-- Converting 40 minutes to hours
def running_fraction_of_hour : ℚ := 40 / 60

-- Formulate the main theorem:
theorem train_speed_excluding_stoppages
    (H1 : including_stoppages_speed = 40)
    (H2 : running_fraction_of_hour = 2 / 3) :
    S = 60 :=
by
    sorry

end train_speed_excluding_stoppages_l120_120663


namespace Alton_profit_l120_120498

variable (earnings_per_day : ℕ)
variable (days_per_week : ℕ)
variable (rent_per_week : ℕ)

theorem Alton_profit (h1 : earnings_per_day = 8) (h2 : days_per_week = 7) (h3 : rent_per_week = 20) :
  earnings_per_day * days_per_week - rent_per_week = 36 := 
by sorry

end Alton_profit_l120_120498


namespace Haley_initial_trees_l120_120538

theorem Haley_initial_trees (T : ℕ) (h1 : T - 4 ≥ 0) (h2 : (T - 4) + 5 = 10): T = 9 :=
by
  -- proof goes here
  sorry

end Haley_initial_trees_l120_120538


namespace find_prime_q_l120_120462

theorem find_prime_q (p q r : ℕ) 
  (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q)
  (prime_r : Nat.Prime r)
  (eq_r : q - p = r)
  (cond_p : 5 < p ∧ p < 15)
  (cond_q : q < 15) :
  q = 13 :=
sorry

end find_prime_q_l120_120462


namespace smallest_positive_divisor_l120_120999

theorem smallest_positive_divisor
  (a b x₀ y₀ : ℤ)
  (h₀ : a ≠ 0 ∨ b ≠ 0)
  (h₁ : ∀ x y, a * x₀ + b * y₀ ≤ 0 ∨ a * x + b * y ≥ a * x₀ + b * y₀)
  (h₂ : 0 < a * x₀ + b * y₀):
  ∀ x y : ℤ, a * x₀ + b * y₀ ∣ a * x + b * y := 
sorry

end smallest_positive_divisor_l120_120999


namespace square_side_length_l120_120020

theorem square_side_length :
  ∀ (s : ℝ), (∃ w l : ℝ, w = 6 ∧ l = 24 ∧ s^2 = w * l) → s = 12 := by 
  sorry

end square_side_length_l120_120020


namespace inverse_of_B_cubed_l120_120854

theorem inverse_of_B_cubed
  (B_inv : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![3, -1],
    ![0, 5]
  ]) :
  (B_inv ^ 3) = ![
    ![27, -49],
    ![0, 125]
  ] := 
by
  sorry

end inverse_of_B_cubed_l120_120854


namespace math_problem_l120_120336

theorem math_problem :
    (50 + 5 * (12 / (180 / 3))^2) * Real.sin (Real.pi / 6) = 25.1 :=
by
  sorry

end math_problem_l120_120336


namespace circle_area_l120_120463

-- Definition of the given circle equation
def circle_eq (x y : ℝ) : Prop := 3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0

-- Prove the area of the circle defined by circle_eq (x y) is 25/4 * π
theorem circle_area (x y : ℝ) (h : circle_eq x y) : ∃ r : ℝ, r = 5 / 2 ∧ π * r^2 = 25 / 4 * π :=
by
  sorry

end circle_area_l120_120463


namespace find_ab_l120_120675

-- Define the polynomials involved
def poly1 (x : ℝ) (a b : ℝ) : ℝ := a * x^4 + b * x^2 + 1
def poly2 (x : ℝ) : ℝ := x^2 - x - 2

-- Define the roots of the second polynomial
def root1 : ℝ := 2
def root2 : ℝ := -1

-- State the theorem to prove
theorem find_ab (a b : ℝ) :
  poly1 root1 a b = 0 ∧ poly1 root2 a b = 0 → a = 1/4 ∧ b = -5/4 :=
by
  -- Skipping the proof here
  sorry

end find_ab_l120_120675


namespace find_C_coordinates_l120_120012

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 11, y := 9 }
def B : Point := { x := 2, y := -3 }
def D : Point := { x := -1, y := 3 }

-- Define the isosceles property
def is_isosceles (A B C : Point) : Prop :=
  Real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2) = Real.sqrt ((A.x - C.x) ^ 2 + (A.y - C.y) ^ 2)

-- Define the midpoint property
def is_midpoint (D B C : Point) : Prop :=
  D.x = (B.x + C.x) / 2 ∧ D.y = (B.y + C.y) / 2

theorem find_C_coordinates (C : Point)
  (h_iso : is_isosceles A B C)
  (h_mid : is_midpoint D B C) :
  C = { x := -4, y := 9 } := 
  sorry

end find_C_coordinates_l120_120012


namespace inequality_solution_l120_120771

theorem inequality_solution (x : ℝ) : 
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1 / 2 < x ∧ x ≤ 1 :=
sorry

end inequality_solution_l120_120771


namespace other_asymptote_of_hyperbola_l120_120143

theorem other_asymptote_of_hyperbola (a b : ℝ) : 
  (∀ x : ℝ, y = 2 * x + 3) → 
  (∃ y : ℝ, x = -4) → 
  (∀ x : ℝ, y = - (1 / 2) * x - 7) := 
by {
  -- The proof will go here
  sorry
}

end other_asymptote_of_hyperbola_l120_120143


namespace theater_ticket_sales_l120_120197

theorem theater_ticket_sales (O B : ℕ) 
  (h1 : O + B = 370) 
  (h2 : 12 * O + 8 * B = 3320) : 
  B - O = 190 := 
sorry

end theater_ticket_sales_l120_120197


namespace expand_polynomial_l120_120212

theorem expand_polynomial : 
  (∀ (x : ℝ), (5 * x^3 + 7) * (3 * x + 4) = 15 * x^4 + 20 * x^3 + 21 * x + 28) :=
by
  intro x
  sorry

end expand_polynomial_l120_120212


namespace find_linear_function_b_l120_120548

theorem find_linear_function_b (b : ℝ) :
  (∃ b, (∀ x y, y = 2 * x + b - 2 → (x = -1 ∧ y = 0)) → b = 4) :=
sorry

end find_linear_function_b_l120_120548


namespace max_total_length_of_cuts_l120_120340

theorem max_total_length_of_cuts (A : ℕ) (n : ℕ) (m : ℕ) (P : ℕ) (Q : ℕ)
  (h1 : A = 30 * 30)
  (h2 : n = 225)
  (h3 : m = A / n)
  (h4 : m = 4)
  (h5 : Q = 4 * 30)
  (h6 : P = 225 * 10 - Q)
  (h7 : P / 2 = 1065) :
  P / 2 = 1065 :=
by 
  exact h7

end max_total_length_of_cuts_l120_120340


namespace oblique_asymptote_l120_120464

theorem oblique_asymptote :
  ∀ x : ℝ, (∃ δ > 0, ∀ y > x, (abs (3 * y^2 + 8 * y + 12) / (3 * y + 4) - (y + 4 / 3)) < δ) :=
sorry

end oblique_asymptote_l120_120464


namespace function_behavior_l120_120898

noncomputable def f (x : ℝ) : ℝ := abs (2^x - 2)

theorem function_behavior :
  (∀ x y : ℝ, x < y ∧ y ≤ 1 → f x ≥ f y) ∧ (∀ x y : ℝ, x < y ∧ x ≥ 1 → f x ≤ f y) :=
by
  sorry

end function_behavior_l120_120898


namespace fill_time_correct_l120_120050

-- Define the conditions
def rightEyeTime := 2 * 24 -- hours
def leftEyeTime := 3 * 24 -- hours
def rightFootTime := 4 * 24 -- hours
def throatTime := 6       -- hours

def rightEyeRate := 1 / rightEyeTime
def leftEyeRate := 1 / leftEyeTime
def rightFootRate := 1 / rightFootTime
def throatRate := 1 / throatTime

-- Combined rate calculation
def combinedRate := rightEyeRate + leftEyeRate + rightFootRate + throatRate

-- Goal definition
def fillTime := 288 / 61 -- hours

-- Prove that the calculated time to fill the pool matches the given answer
theorem fill_time_correct : (1 / combinedRate) = fillTime :=
by {
  sorry
}

end fill_time_correct_l120_120050


namespace income_of_first_member_l120_120593

-- Define the number of family members
def num_members : ℕ := 4

-- Define the average income per member
def avg_income : ℕ := 10000

-- Define the known incomes of the other three members
def income2 : ℕ := 15000
def income3 : ℕ := 6000
def income4 : ℕ := 11000

-- Define the total income of the family
def total_income : ℕ := avg_income * num_members

-- Define the total income of the other three members
def total_other_incomes : ℕ := income2 + income3 + income4

-- Define the income of the first member
def income1 : ℕ := total_income - total_other_incomes

-- The theorem to prove
theorem income_of_first_member : income1 = 8000 := by
  sorry

end income_of_first_member_l120_120593


namespace intersection_on_y_axis_l120_120156

theorem intersection_on_y_axis (k : ℝ) (x y : ℝ) :
  (2 * x + 3 * y - k = 0) →
  (x - k * y + 12 = 0) →
  (x = 0) →
  k = 6 ∨ k = -6 :=
by
  sorry

end intersection_on_y_axis_l120_120156


namespace necessary_and_sufficient_condition_l120_120543

theorem necessary_and_sufficient_condition (a : ℝ) : (a > 0) ↔ (a + 1 / a ≥ 2) :=
sorry

end necessary_and_sufficient_condition_l120_120543


namespace largest_sum_fraction_l120_120066

theorem largest_sum_fraction :
  max (max (max (max ((1/3) + (1/2)) ((1/3) + (1/4))) ((1/3) + (1/5))) ((1/3) + (1/7))) ((1/3) + (1/9)) = 5/6 :=
by
  sorry

end largest_sum_fraction_l120_120066


namespace find_original_numbers_l120_120942

-- Definitions corresponding to the conditions in a
def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ x y : ℕ, x + y = 8 ∧ n = 10 * x + y

-- Definitions to state the condition about the swapped number and their product
def swapped_number (n : ℕ) : ℕ :=
  let x := n / 10 in
  let y := n % 10 in
  10 * y + x

def product_of_numbers (n : ℕ) : Prop :=
  n * swapped_number(n) = 1855

-- Statement combining conditions and correct answer
theorem find_original_numbers (n : ℕ) :
  is_valid_number n ∧ product_of_numbers n → n = 35 ∨ n = 53 :=
sorry

end find_original_numbers_l120_120942


namespace small_supermarkets_sample_count_l120_120553

def large := 300
def medium := 600
def small := 2100
def sample_size := 100
def total := large + medium + small

theorem small_supermarkets_sample_count :
  small * (sample_size / total) = 70 := by
  sorry

end small_supermarkets_sample_count_l120_120553


namespace line_slope_intercept_l120_120027

theorem line_slope_intercept :
  ∃ k b, (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 → y = k * x + b) ∧ k = 2/3 ∧ b = 2 :=
by
  sorry

end line_slope_intercept_l120_120027


namespace bottles_per_case_l120_120486

theorem bottles_per_case (total_bottles : ℕ) (total_cases : ℕ) (h1 : total_bottles = 60000) (h2 : total_cases = 12000) :
  total_bottles / total_cases = 5 :=
by
  -- Using the given problem, so steps from the solution are not required here
  sorry

end bottles_per_case_l120_120486


namespace students_exam_percentage_l120_120120

theorem students_exam_percentage 
  (total_students : ℕ) 
  (avg_assigned_day : ℚ) 
  (avg_makeup_day : ℚ)
  (overall_avg : ℚ) 
  (h_total : total_students = 100)
  (h_avg_assigned_day : avg_assigned_day = 0.60) 
  (h_avg_makeup_day : avg_makeup_day = 0.80) 
  (h_overall_avg : overall_avg = 0.66) : 
  ∃ x : ℚ, x = 70 / 100 :=
by
  sorry

end students_exam_percentage_l120_120120


namespace factorize_expression_l120_120819

variable (x y : ℝ)

theorem factorize_expression : 9 * x^2 * y - y = y * (3 * x + 1) * (3 * x - 1) := 
by
  sorry

end factorize_expression_l120_120819


namespace factorial_expression_l120_120953

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l120_120953


namespace calc_nabla_l120_120947

noncomputable def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem calc_nabla : (op_nabla (op_nabla 2 3) 4) = 11 / 9 :=
by
  unfold op_nabla
  sorry

end calc_nabla_l120_120947


namespace fundraiser_successful_l120_120990

-- Defining the conditions
def num_students_bringing_brownies := 30
def brownies_per_student := 12
def num_students_bringing_cookies := 20
def cookies_per_student := 24
def num_students_bringing_donuts := 15
def donuts_per_student := 12
def price_per_treat := 2

-- Calculating the total number of each type of treat
def total_brownies := num_students_bringing_brownies * brownies_per_student
def total_cookies := num_students_bringing_cookies * cookies_per_student
def total_donuts := num_students_bringing_donuts * donuts_per_student

-- Calculating the total number of treats
def total_treats := total_brownies + total_cookies + total_donuts

-- Calculating the total money raised
def total_money_raised := total_treats * price_per_treat

theorem fundraiser_successful : total_money_raised = 2040 := by
    -- We introduce a sorry here because we are not providing the proof steps.
    sorry

end fundraiser_successful_l120_120990


namespace johns_photo_world_sitting_fee_l120_120939

variable (J : ℝ)

theorem johns_photo_world_sitting_fee
  (h1 : ∀ n : ℝ, n = 12 → 2.75 * n + J = 1.50 * n + 140) : J = 125 :=
by
  -- We will skip the proof since it is not required by the problem statement.
  sorry

end johns_photo_world_sitting_fee_l120_120939


namespace mark_ate_in_first_four_days_l120_120137

-- Definitions based on conditions
def total_fruit : ℕ := 10
def fruit_kept : ℕ := 2
def fruit_brought_on_friday : ℕ := 3

-- Statement to be proved
theorem mark_ate_in_first_four_days : total_fruit - fruit_kept - fruit_brought_on_friday = 5 := 
by sorry

end mark_ate_in_first_four_days_l120_120137


namespace determine_k_l120_120691

theorem determine_k 
  (k : ℝ) 
  (r s : ℝ) 
  (h1 : r + s = -k) 
  (h2 : r * s = 6) 
  (h3 : (r + 5) + (s + 5) = k) : 
  k = 5 := 
by 
  sorry

end determine_k_l120_120691


namespace avg_words_per_hour_l120_120797

theorem avg_words_per_hour (words hours : ℝ) (h_words : words = 40000) (h_hours : hours = 80) :
  words / hours = 500 :=
by
  rw [h_words, h_hours]
  norm_num
  done

end avg_words_per_hour_l120_120797


namespace line_forms_equivalence_l120_120983

noncomputable def points (P Q : ℝ × ℝ) : Prop := 
  ∃ m c, ∃ b d, P = (b, m * b + c) ∧ Q = (d, m * d + c)

theorem line_forms_equivalence :
  points (-2, 3) (4, -1) →
  (∀ x y : ℝ, (y + 1) / (3 + 1) = (x - 4) / (-2 - 4)) ∧
  (∀ x y : ℝ, y + 1 = - (2 / 3) * (x - 4)) ∧
  (∀ x y : ℝ, y = - (2 / 3) * x + 5 / 3) ∧
  (∀ x y : ℝ, x / (5 / 2) + y / (5 / 3) = 1) :=
  sorry

end line_forms_equivalence_l120_120983


namespace calculate_ray_grocery_bill_l120_120430

noncomputable def ray_grocery_total_cost : ℝ :=
let hamburger_meat_price := 5.0
let crackers_price := 3.5
let frozen_vegetables_price := 2.0 * 4
let cheese_price := 3.5
let chicken_price := 6.5
let cereal_price := 4.0
let wine_price := 10.0
let cookies_price := 3.0

let discount_hamburger_meat := hamburger_meat_price * 0.10
let discount_crackers := crackers_price * 0.10
let discount_frozen_vegetables := frozen_vegetables_price * 0.10
let discount_cheese := cheese_price * 0.05
let discount_chicken := chicken_price * 0.05
let discount_wine := wine_price * 0.15

let discounted_hamburger_meat_price := hamburger_meat_price - discount_hamburger_meat
let discounted_crackers_price := crackers_price - discount_crackers
let discounted_frozen_vegetables_price := frozen_vegetables_price - discount_frozen_vegetables
let discounted_cheese_price := cheese_price - discount_cheese
let discounted_chicken_price := chicken_price - discount_chicken
let discounted_wine_price := wine_price - discount_wine

let total_discounted_price :=
  discounted_hamburger_meat_price +
  discounted_crackers_price +
  discounted_frozen_vegetables_price +
  discounted_cheese_price +
  discounted_chicken_price +
  cereal_price +
  discounted_wine_price +
  cookies_price

let food_items_total_price :=
  discounted_hamburger_meat_price +
  discounted_crackers_price +
  discounted_frozen_vegetables_price +
  discounted_cheese_price +
  discounted_chicken_price +
  cereal_price +
  cookies_price

let food_sales_tax := food_items_total_price * 0.06
let wine_sales_tax := discounted_wine_price * 0.09

let total_with_tax := total_discounted_price + food_sales_tax + wine_sales_tax

total_with_tax

theorem calculate_ray_grocery_bill :
  ray_grocery_total_cost = 42.51 :=
sorry

end calculate_ray_grocery_bill_l120_120430


namespace pirate_flag_minimal_pieces_l120_120540

theorem pirate_flag_minimal_pieces (original_stripes : ℕ) (desired_stripes : ℕ) (cuts_needed : ℕ) : 
  original_stripes = 12 →
  desired_stripes = 10 →
  cuts_needed = 1 →
  ∃ pieces : ℕ, pieces = 2 ∧ 
  (∀ (top_stripes bottom_stripes: ℕ), top_stripes + bottom_stripes = original_stripes → top_stripes = desired_stripes → 
   pieces = 1 + (if bottom_stripes = original_stripes - desired_stripes then 1 else 0)) :=
by intros;
   sorry

end pirate_flag_minimal_pieces_l120_120540


namespace probability_no_absolute_winner_l120_120505

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end probability_no_absolute_winner_l120_120505


namespace calories_in_200_grams_is_137_l120_120383

-- Define the grams of ingredients used.
def lemon_juice_grams := 100
def sugar_grams := 100
def water_grams := 400

-- Define the calories per 100 grams of each ingredient.
def lemon_juice_calories_per_100_grams := 25
def sugar_calories_per_100_grams := 386
def water_calories_per_100_grams := 0

-- Calculate the total calories in the entire lemonade mixture.
def total_calories : Nat :=
  (lemon_juice_grams * lemon_juice_calories_per_100_grams / 100) + 
  (sugar_grams * sugar_calories_per_100_grams / 100) +
  (water_grams * water_calories_per_100_grams / 100)

-- Calculate the total weight of the lemonade mixture.
def total_weight : Nat := lemon_juice_grams + sugar_grams + water_grams

-- Calculate the caloric density (calories per gram).
def caloric_density := total_calories / total_weight

-- Calculate the calories in 200 grams of lemonade.
def calories_in_200_grams := (caloric_density * 200)

-- The theorem to prove
theorem calories_in_200_grams_is_137 : calories_in_200_grams = 137 :=
by sorry

end calories_in_200_grams_is_137_l120_120383


namespace lizzie_scored_six_l120_120731

-- Definitions based on the problem conditions
def lizzie_score : Nat := sorry
def nathalie_score := lizzie_score + 3
def aimee_score := 2 * (lizzie_score + nathalie_score)

-- Total score condition
def total_score := 50
def teammates_score := 17
def combined_score := total_score - teammates_score

-- Proven statement
theorem lizzie_scored_six:
  (lizzie_score + nathalie_score + aimee_score = combined_score) → lizzie_score = 6 :=
by sorry

end lizzie_scored_six_l120_120731


namespace jackie_apples_l120_120648

theorem jackie_apples (a : ℕ) (j : ℕ) (h1 : a = 9) (h2 : a = j + 3) : j = 6 :=
by
  sorry

end jackie_apples_l120_120648


namespace probability_neither_red_nor_purple_l120_120625

theorem probability_neither_red_nor_purple 
    (total_balls : ℕ)
    (white_balls : ℕ) 
    (green_balls : ℕ) 
    (yellow_balls : ℕ) 
    (red_balls : ℕ) 
    (purple_balls : ℕ) 
    (h_total : total_balls = white_balls + green_balls + yellow_balls + red_balls + purple_balls)
    (h_counts : white_balls = 50 ∧ green_balls = 30 ∧ yellow_balls = 8 ∧ red_balls = 9 ∧ purple_balls = 3):
    (88 : ℚ) / 100 = 0.88 :=
by
  sorry

end probability_neither_red_nor_purple_l120_120625


namespace largest_number_of_gold_coins_l120_120623

theorem largest_number_of_gold_coins (n : ℕ) :
  (∃ k : ℕ, n = 13 * k + 3 ∧ n < 150) → n ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l120_120623


namespace relationship_of_rationals_l120_120845

theorem relationship_of_rationals (a b c : ℚ) (h1 : a - b > 0) (h2 : b - c > 0) : c < b ∧ b < a :=
by {
  sorry
}

end relationship_of_rationals_l120_120845


namespace investment_calculation_l120_120892

theorem investment_calculation :
  ∃ (x : ℝ), x * (1.04 ^ 14) = 1000 := by
  use 571.75
  sorry

end investment_calculation_l120_120892


namespace net_income_difference_l120_120306

-- Define Terry's and Jordan's daily income and working days
def terryDailyIncome : ℝ := 24
def terryWorkDays : ℝ := 7
def jordanDailyIncome : ℝ := 30
def jordanWorkDays : ℝ := 6

-- Define the tax rate
def taxRate : ℝ := 0.10

-- Calculate weekly gross incomes
def terryGrossWeeklyIncome : ℝ := terryDailyIncome * terryWorkDays
def jordanGrossWeeklyIncome : ℝ := jordanDailyIncome * jordanWorkDays

-- Calculate tax deductions
def terryTaxDeduction : ℝ := taxRate * terryGrossWeeklyIncome
def jordanTaxDeduction : ℝ := taxRate * jordanGrossWeeklyIncome

-- Calculate net weekly incomes
def terryNetWeeklyIncome : ℝ := terryGrossWeeklyIncome - terryTaxDeduction
def jordanNetWeeklyIncome : ℝ := jordanGrossWeeklyIncome - jordanTaxDeduction

-- Calculate the difference
def incomeDifference : ℝ := jordanNetWeeklyIncome - terryNetWeeklyIncome

-- The theorem to be proven
theorem net_income_difference :
  incomeDifference = 10.80 :=
by
  sorry

end net_income_difference_l120_120306


namespace counting_numbers_leave_remainder_6_divide_53_l120_120254

theorem counting_numbers_leave_remainder_6_divide_53 :
  ∃! n : ℕ, (∃ k : ℕ, 53 = n * k + 6) ∧ n > 6 :=
sorry

end counting_numbers_leave_remainder_6_divide_53_l120_120254


namespace range_of_a_l120_120096

variable (a : ℝ)
def is_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0
def z : ℂ := 4 - 2 * Complex.I

theorem range_of_a (ha : is_second_quadrant ((z + a * Complex.I) ^ 2)) : a > 6 := by
  sorry

end range_of_a_l120_120096


namespace circle_area_isosceles_triangle_l120_120476

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end circle_area_isosceles_triangle_l120_120476


namespace find_xz_over_y_squared_l120_120834

variable {x y z : ℝ}

noncomputable def k : ℝ := 7

theorem find_xz_over_y_squared
    (h1 : x + k * y + 4 * z = 0)
    (h2 : 4 * x + k * y - 3 * z = 0)
    (h3 : x + 3 * y - 2 * z = 0)
    (h_nz : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
    (x * z) / (y ^ 2) = 26 / 9 :=
by sorry

end find_xz_over_y_squared_l120_120834


namespace find_fraction_l120_120261

variable (F N : ℚ)

-- Defining the conditions
def condition1 : Prop := (1 / 3) * F * N = 18
def condition2 : Prop := (3 / 10) * N = 64.8

-- Proof statement
theorem find_fraction (h1 : condition1 F N) (h2 : condition2 N) : F = 1 / 4 := by 
  sorry

end find_fraction_l120_120261


namespace fundraiser_total_money_l120_120988

def number_of_items (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student : ℕ) : ℕ :=
  (students1 * brownies_per_student) + (students2 * cookies_per_student) + (students3 * donuts_per_student)

def total_money_raised (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) : ℕ :=
  number_of_items students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student * price_per_item

theorem fundraiser_total_money (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) :
  students1 = 30 → students2 = 20 → students3 = 15 → brownies_per_student = 12 → cookies_per_student = 24 → donuts_per_student = 12 → price_per_item = 2 → 
  total_money_raised students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item = 2040 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end fundraiser_total_money_l120_120988


namespace find_a_l120_120396

noncomputable def binomial_expansion_term_coefficient
  (n : ℕ) (r : ℕ) (a : ℝ) (x : ℝ) : ℝ :=
  (2^(n-r)) * ((-a)^r) * (Nat.choose n r) * (x^(n - 2*r))

theorem find_a 
  (a : ℝ)
  (h : binomial_expansion_term_coefficient 7 5 a 1 = 84) 
  : a = -1 :=
sorry

end find_a_l120_120396


namespace polynomial_divisibility_l120_120929

theorem polynomial_divisibility (n : ℕ) : (∀ x : ℤ, (x^2 + x + 1 ∣ x^(2*n) + x^n + 1)) ↔ (3 ∣ n) := by
  sorry

end polynomial_divisibility_l120_120929


namespace average_fish_per_person_l120_120858

theorem average_fish_per_person (Aang Sokka Toph : ℕ) 
  (haang : Aang = 7) (hsokka : Sokka = 5) (htoph : Toph = 12) : 
  (Aang + Sokka + Toph) / 3 = 8 := by
  sorry

end average_fish_per_person_l120_120858


namespace total_lobster_pounds_l120_120251

variable (lobster_other_harbor1 : ℕ)
variable (lobster_other_harbor2 : ℕ)
variable (lobster_hooper_bay : ℕ)

-- Conditions
axiom h_eq : lobster_hooper_bay = 2 * (lobster_other_harbor1 + lobster_other_harbor2)
axiom other_harbors_eq : lobster_other_harbor1 = 80 ∧ lobster_other_harbor2 = 80

-- Proof statement
theorem total_lobster_pounds : 
  lobster_other_harbor1 + lobster_other_harbor2 + lobster_hooper_bay = 480 :=
by
  sorry

end total_lobster_pounds_l120_120251


namespace part_3_l120_120692

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x - m * x^2
noncomputable def g (x : ℝ) (m : ℝ) := (1/2) * m * x^2 + x
noncomputable def F (x : ℝ) (m : ℝ) := f x m + g x m

theorem part_3 (x₁ x₂ : ℝ) (m : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hm : m = -2)
  (hF : F x₁ m + F x₂ m + x₁ * x₂ = 0) : x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end part_3_l120_120692


namespace find_x_squared_plus_y_squared_plus_z_squared_l120_120100

theorem find_x_squared_plus_y_squared_plus_z_squared
  (x y z : ℤ)
  (h1 : x + y + z = 3)
  (h2 : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 57 :=
by
  sorry

end find_x_squared_plus_y_squared_plus_z_squared_l120_120100


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l120_120709

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l120_120709


namespace average_percent_score_l120_120426

theorem average_percent_score (num_students : ℕ)
    (students_95 students_85 students_75 students_65 students_55 students_45 : ℕ)
    (h : students_95 + students_85 + students_75 + students_65 + students_55 + students_45 = 120) :
  ((95 * students_95 + 85 * students_85 + 75 * students_75 + 65 * students_65 + 55 * students_55 + 45 * students_45) / 120 : ℚ) = 72.08 := 
by {
  sorry
}

end average_percent_score_l120_120426


namespace graph_of_equation_is_two_lines_l120_120040

theorem graph_of_equation_is_two_lines : 
  ∀ (x y : ℝ), (x - y)^2 = x^2 - y^2 ↔ (x = 0 ∨ y = 0) := 
by
  sorry

end graph_of_equation_is_two_lines_l120_120040


namespace quadratic_inequality_no_real_roots_l120_120864

theorem quadratic_inequality_no_real_roots (a b c : ℝ) (h : a ≠ 0) (h_Δ : b^2 - 4 * a * c < 0) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) :=
sorry

end quadratic_inequality_no_real_roots_l120_120864


namespace pencils_purchased_l120_120552

theorem pencils_purchased (n : ℕ) (h1: n ≤ 10) 
  (h2: 2 ≤ 10) 
  (h3: (10 - 2) / 10 * (10 - 2 - 1) / (10 - 1) * (10 - 2 - 2) / (10 - 2) = 0.4666666666666667) :
  n = 3 :=
sorry

end pencils_purchased_l120_120552


namespace solve_quadratic_equation_l120_120434

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2 := by
sorry

end solve_quadratic_equation_l120_120434


namespace rectangular_prism_surface_area_l120_120599

/-- The surface area of a rectangular prism with edge lengths 2, 3, and 4 is 52. -/
theorem rectangular_prism_surface_area :
  let a := 2
  let b := 3
  let c := 4
  2 * (a * b + a * c + b * c) = 52 :=
by
  let a := 2
  let b := 3
  let c := 4
  show 2 * (a * b + a * c + b * c) = 52
  sorry

end rectangular_prism_surface_area_l120_120599


namespace fuchsia_to_mauve_l120_120203

theorem fuchsia_to_mauve (F : ℝ) :
  (5 / 8) * F + (3 * 26.67 : ℝ) = (3 / 8) * F + (5 / 8) * F →
  F = 106.68 :=
by
  intro h
  -- Step to implement the solution would go here
  sorry

end fuchsia_to_mauve_l120_120203


namespace pages_already_read_l120_120522

theorem pages_already_read (total_pages : ℕ) (pages_left : ℕ) (h_total : total_pages = 563) (h_left : pages_left = 416) :
  total_pages - pages_left = 147 :=
by
  sorry

end pages_already_read_l120_120522


namespace pyramid_total_area_l120_120600

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

theorem pyramid_total_area 
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base_edge : base_edge = pyramid_base_edge)
  (h_lateral_edge : lateral_edge = pyramid_lateral_edge) 
: 4 * (1 / 2 * base_edge * real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * real.sqrt 33 := 
sorry

end pyramid_total_area_l120_120600


namespace perfect_square_sequence_l120_120209

theorem perfect_square_sequence (x : ℕ → ℤ) (h₀ : x 0 = 0) (h₁ : x 1 = 3) 
  (h₂ : ∀ n, x (n + 1) + x (n - 1) = 4 * x n) : 
  ∀ n, ∃ k : ℤ, x (n + 1) * x (n - 1) + 9 = k^2 :=
by 
  sorry

end perfect_square_sequence_l120_120209


namespace log_domain_l120_120152

theorem log_domain (x : ℝ) : x + 2 > 0 ↔ x ∈ Set.Ioi (-2) :=
by
  sorry

end log_domain_l120_120152


namespace g_bounded_l120_120131

noncomputable def problem (f f' f'' : ℝ → ℝ) (g : ℝ → ℝ) :=
  (∀ x ≥ 0, f'' x = 1 / (x^2 + (f' x)^2 + 1)) ∧
  f 0 = 0 ∧ f' 0 = 0 ∧
  (∀ x ≥ 0, g x = f x / x) ∧ g 0 = 0

theorem g_bounded {f f' f'' g : ℝ → ℝ} (h : problem f f' f'' g) :
  ∀ x ≥ 0, g x ≤ Real.pi / 2 :=
sorry

end g_bounded_l120_120131


namespace smallest_number_with_divisibility_condition_l120_120915

theorem smallest_number_with_divisibility_condition :
  ∃ x : ℕ, (x + 7) % 24 = 0 ∧ (x + 7) % 36 = 0 ∧ (x + 7) % 50 = 0 ∧ (x + 7) % 56 = 0 ∧ (x + 7) % 81 = 0 ∧ x = 113393 :=
by {
  -- sorry is used to skip the proof.
  sorry
}

end smallest_number_with_divisibility_condition_l120_120915


namespace bryson_new_shoes_l120_120804

-- Define the conditions as variables and constant values
def pairs_of_shoes : ℕ := 2 -- Number of pairs Bryson bought
def shoes_per_pair : ℕ := 2 -- Number of shoes per pair

-- Define the theorem to prove the question == answer
theorem bryson_new_shoes : pairs_of_shoes * shoes_per_pair = 4 :=
by
  sorry -- Proof placeholder

end bryson_new_shoes_l120_120804


namespace exists_integers_m_n_l120_120684

theorem exists_integers_m_n (x y : ℝ) (hxy : x ≠ y) : 
  ∃ (m n : ℤ), (m * x + n * y > 0) ∧ (n * x + m * y < 0) :=
sorry

end exists_integers_m_n_l120_120684


namespace total_distance_traveled_l120_120052

noncomputable def totalDistance
  (d1 d2 : ℝ) (s1 s2 : ℝ) (average_speed : ℝ) (total_time : ℝ) : ℝ := 
  average_speed * total_time

theorem total_distance_traveled :
  let d1 := 160
  let s1 := 64
  let d2 := 160
  let s2 := 80
  let average_speed := 71.11111111111111
  let total_time := d1 / s1 + d2 / s2
  totalDistance d1 d2 s1 s2 average_speed total_time = 320 :=
by
  -- This is the main statement theorem
  sorry

end total_distance_traveled_l120_120052


namespace equation_of_circle_given_diameter_l120_120392

def is_on_circle (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

theorem equation_of_circle_given_diameter :
  ∀ (A B : ℝ × ℝ), A = (-3,0) → B = (1,0) → 
  (∃ (x y : ℝ), is_on_circle (-1, 0) 2 (x, y)) ↔ (x + 1)^2 + y^2 = 4 :=
by
  sorry

end equation_of_circle_given_diameter_l120_120392


namespace probability_no_adjacent_birch_l120_120056

theorem probability_no_adjacent_birch (m n : ℕ):
  let maple_trees := 5
  let oak_trees := 4
  let birch_trees := 6
  let total_trees := maple_trees + oak_trees + birch_trees
  (∀ (prob : ℚ), prob = (2 : ℚ) / 45) → (m + n = 47) := by
  sorry

end probability_no_adjacent_birch_l120_120056


namespace no_absolute_winner_probability_l120_120502

-- Define the probabilities of matches
def P_AB : ℝ := 0.6  -- Probability Alyosha wins against Borya
def P_BV : ℝ := 0.4  -- Probability Borya wins against Vasya

-- Define the event C that there is no absolute winner
def event_C (P_AV : ℝ) (P_VB : ℝ) : ℝ :=
  let scenario1 := P_AB * P_BV * P_AV in
  let scenario2 := P_AB * P_VB * (1 - P_AV) in
  scenario1 + scenario2

-- Main theorem to prove
theorem no_absolute_winner_probability : 
  event_C 1 0.6 = 0.24 :=
by
  rw [event_C]
  simp
  norm_num
  sorry

end no_absolute_winner_probability_l120_120502


namespace three_n_plus_two_not_perfect_square_l120_120300

theorem three_n_plus_two_not_perfect_square (n : ℕ) : ¬ ∃ (a : ℕ), 3 * n + 2 = a * a :=
by
  sorry

end three_n_plus_two_not_perfect_square_l120_120300


namespace fraction_of_number_is_one_fifth_l120_120784

theorem fraction_of_number_is_one_fifth (N : ℕ) (f : ℚ) 
    (hN : N = 90) 
    (h : 3 + (1 / 2) * (1 / 3) * f * N = (1 / 15) * N) : 
  f = 1 / 5 := by 
  sorry

end fraction_of_number_is_one_fifth_l120_120784


namespace a_n_b_n_T_n_correct_l120_120165

noncomputable def a_n (n : ℕ) : ℕ := n

noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1)

noncomputable def S (n : ℕ) : ℕ := (Finset.range n).sum a_n

def condition1 := S 2 * b_n 2 = 6
def condition2 := b_n 2 + S 3 = 8

noncomputable def T (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, a_n (i+1) * b_n (i+1))

theorem a_n_b_n_T_n_correct :
  condition1 →
  condition2 →
  (∀ n, a_n n = n) ∧ 
  (∀ n, b_n n = 2^(n-1)) ∧ 
  (∀ n, T n = 1 + (n-1) * 2^n) :=
by
  sorry

end a_n_b_n_T_n_correct_l120_120165


namespace dogs_food_consumption_l120_120778

theorem dogs_food_consumption :
  (let cups_per_meal_momo_fifi := 1.5
   let meals_per_day := 3
   let cups_per_meal_gigi := 2
   let cups_to_pounds := 3
   let daily_food_momo_fifi := cups_per_meal_momo_fifi * meals_per_day * 2
   let daily_food_gigi := cups_per_meal_gigi * meals_per_day
   daily_food_momo_fifi + daily_food_gigi) / cups_to_pounds = 5 :=
by
  sorry

end dogs_food_consumption_l120_120778


namespace value_of_a_b_c_l120_120387

noncomputable def absolute_value (x : ℤ) : ℤ := abs x

theorem value_of_a_b_c (a b c : ℤ)
  (ha : absolute_value a = 1)
  (hb : absolute_value b = 2)
  (hc : absolute_value c = 3)
  (h : a > b ∧ b > c) :
  a + b - c = 2 ∨ a + b - c = 0 :=
by
  sorry

end value_of_a_b_c_l120_120387


namespace planted_area_ratio_l120_120795

noncomputable def ratio_of_planted_area_to_total_area : ℚ := 145 / 147

theorem planted_area_ratio (h : ∃ (S : ℚ), 
  (∃ (x y : ℚ), x * x + y * y ≤ S * S) ∧
  (∃ (a b : ℚ), 3 * a + 4 * b = 12 ∧ (3 * x + 4 * y - 12) / 5 = 2)) :
  ratio_of_planted_area_to_total_area = 145 / 147 :=
sorry

end planted_area_ratio_l120_120795


namespace find_urn_yellow_balls_l120_120063

theorem find_urn_yellow_balls :
  ∃ (M : ℝ), 
    (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
    M = 111 := 
sorry

end find_urn_yellow_balls_l120_120063


namespace emily_cleaning_time_l120_120565

noncomputable def total_time : ℝ := 8 -- total time in hours
noncomputable def lilly_fiona_time : ℝ := 1/4 * total_time -- Lilly and Fiona's combined time in hours
noncomputable def jack_time : ℝ := 1/3 * total_time -- Jack's time in hours
noncomputable def emily_time : ℝ := total_time - lilly_fiona_time - jack_time -- Emily's time in hours
noncomputable def emily_time_minutes : ℝ := emily_time * 60 -- Emily's time in minutes

theorem emily_cleaning_time :
  emily_time_minutes = 200 := by
  sorry

end emily_cleaning_time_l120_120565


namespace combined_dog_years_difference_l120_120610

theorem combined_dog_years_difference 
  (Max_age : ℕ) 
  (small_breed_rate medium_breed_rate large_breed_rate : ℕ) 
  (Max_turns_age : ℕ) 
  (small_breed_diff medium_breed_diff large_breed_diff combined_diff : ℕ) :
  Max_age = 3 →
  small_breed_rate = 5 →
  medium_breed_rate = 7 →
  large_breed_rate = 9 →
  Max_turns_age = 6 →
  small_breed_diff = small_breed_rate * Max_turns_age - Max_turns_age →
  medium_breed_diff = medium_breed_rate * Max_turns_age - Max_turns_age →
  large_breed_diff = large_breed_rate * Max_turns_age - Max_turns_age →
  combined_diff = small_breed_diff + medium_breed_diff + large_breed_diff →
  combined_diff = 108 :=
by
  intros
  sorry

end combined_dog_years_difference_l120_120610


namespace find_r_floor_r_add_r_eq_18point2_l120_120214

theorem find_r_floor_r_add_r_eq_18point2 (r : ℝ) (h : ⌊r⌋ + r = 18.2) : r = 9.2 := 
sorry

end find_r_floor_r_add_r_eq_18point2_l120_120214


namespace orthogonality_implies_x_value_l120_120225

theorem orthogonality_implies_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, -1)
  a.1 * b.1 + a.2 * b.2 = 0 → x = 1 :=
sorry

end orthogonality_implies_x_value_l120_120225


namespace area_in_terms_of_diagonal_l120_120903

variables (l w d : ℝ)

-- Given conditions
def length_to_width_ratio := l / w = 5 / 2
def diagonal_relation := d^2 = l^2 + w^2

-- Proving the area is kd^2 with k = 10 / 29
theorem area_in_terms_of_diagonal 
    (ratio : length_to_width_ratio l w)
    (diag_rel : diagonal_relation l w d) :
  ∃ k, k = 10 / 29 ∧ (l * w = k * d^2) :=
sorry

end area_in_terms_of_diagonal_l120_120903


namespace problem_l120_120407

variable {m n r t : ℚ}

theorem problem (h1 : m / n = 5 / 4) (h2 : r / t = 8 / 15) : (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 2 :=
by
  sorry

end problem_l120_120407


namespace factorial_division_l120_120955

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l120_120955


namespace part1_part2_l120_120681

theorem part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) : 
  (1 / a) + (1 / (b + 1)) ≥ 4 / 5 := 
by 
  sorry

theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 8) : 
  a + b ≥ 4 := 
by 
  sorry

end part1_part2_l120_120681


namespace xiao_ming_climb_stairs_8_l120_120138

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | (n+2) => fibonacci n + fibonacci (n + 1)

theorem xiao_ming_climb_stairs_8 :
  fibonacci 8 = 34 :=
sorry

end xiao_ming_climb_stairs_8_l120_120138


namespace spring_expenses_l120_120767

noncomputable def expense_by_end_of_february : ℝ := 0.6
noncomputable def expense_by_end_of_may : ℝ := 1.8
noncomputable def spending_during_spring_months := expense_by_end_of_may - expense_by_end_of_february

-- Lean statement for the proof problem
theorem spring_expenses : spending_during_spring_months = 1.2 := by
  sorry

end spring_expenses_l120_120767


namespace new_shoes_last_for_two_years_l120_120794

theorem new_shoes_last_for_two_years :
  let cost_repair := 11.50
  let cost_new := 28.00
  let increase_factor := 1.2173913043478261
  (cost_new / ((increase_factor) * cost_repair)) ≠ 0 :=
by
  sorry

end new_shoes_last_for_two_years_l120_120794


namespace O_l120_120910

theorem O'Hara_triple_49_16_y : 
  (∃ y : ℕ, (49 : ℕ).sqrt + (16 : ℕ).sqrt = y) → y = 11 :=
by
  sorry

end O_l120_120910


namespace reciprocal_sum_hcf_lcm_l120_120044

variables (m n : ℕ)

def HCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem reciprocal_sum_hcf_lcm (h₁ : HCF m n = 6) (h₂ : LCM m n = 210) (h₃ : m + n = 60) :
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 21 :=
by
  -- The proof will be inserted here.
  sorry

end reciprocal_sum_hcf_lcm_l120_120044


namespace find_sin_expression_l120_120736

noncomputable def trigonometric_identity (γ : ℝ) : Prop :=
  3 * (Real.tan γ)^2 + 3 * (1 / (Real.tan γ))^2 + 2 / (Real.sin γ)^2 + 2 / (Real.cos γ)^2 = 19

theorem find_sin_expression (γ : ℝ) (h : trigonometric_identity γ) : 
  (Real.sin γ)^4 - (Real.sin γ)^2 = -1 / 5 :=
sorry

end find_sin_expression_l120_120736


namespace prime_square_plus_two_is_prime_iff_l120_120835

theorem prime_square_plus_two_is_prime_iff (p : ℕ) (hp : Prime p) : Prime (p^2 + 2) ↔ p = 3 :=
sorry

end prime_square_plus_two_is_prime_iff_l120_120835


namespace find_n_l120_120665

theorem find_n (n : ℕ) (h : (n + 1) * n.factorial = 5040) : n = 6 := 
by sorry

end find_n_l120_120665


namespace work_efficiency_ratio_l120_120923

variables (A_eff B_eff : ℚ) (a b : Type)

theorem work_efficiency_ratio (h1 : B_eff = 1 / 33)
  (h2 : A_eff + B_eff = 1 / 11) :
  A_eff / B_eff = 2 :=
by 
  sorry

end work_efficiency_ratio_l120_120923


namespace rectangle_perimeter_l120_120640

variable (L W : ℝ) 

theorem rectangle_perimeter (h1 : L > 4) (h2 : W > 4) (h3 : (L * W) - ((L - 4) * (W - 4)) = 168) : 
  2 * (L + W) = 92 := 
  sorry

end rectangle_perimeter_l120_120640


namespace find_n_l120_120077

theorem find_n (n : ℕ) (h : n * Nat.factorial n + Nat.factorial n = 5040) : n = 6 :=
sorry

end find_n_l120_120077


namespace probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5_l120_120019

open BigOperators

/-- Suppose 30 balls are tossed independently and at random into one 
of the 6 bins. Let p be the probability that one bin ends up with 3 
balls, another with 6 balls, another with 5, another with 4, another 
with 2, and the last one with 10 balls. Let q be the probability 
that each bin ends up with 5 balls. Calculate p / q. 
-/
theorem probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5 :
  (Nat.factorial 5 ^ 6 : ℚ) / ((Nat.factorial 3:ℚ) * Nat.factorial 6 * Nat.factorial 5 * Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 10) = 0.125 := 
sorry

end probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5_l120_120019


namespace area_of_circle_passing_through_vertices_l120_120477

noncomputable def circle_area_through_isosceles_triangle_vertices 
  (a b c : ℝ) (h_isosceles: (a = b) (h_sides: a = 4) (h_base: c = 3) : ℝ :=
π *(√((4^2 - (3/2)^2)/2 + (3/2))^2

theorem area_of_circle_passing_through_vertices :
  circle_area_through_isosceles_triangle_vertices 4 4 3 = 5.6875 * π :=
sorry

end area_of_circle_passing_through_vertices_l120_120477


namespace smallest_digit_for_divisibility_by_9_l120_120672

theorem smallest_digit_for_divisibility_by_9 : 
  ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ (18 + d) % 9 = 0 ∧ ∀ d' : ℕ, (0 ≤ d' ∧ d' ≤ 9 ∧ (18 + d') % 9 = 0) → d' ≥ d :=
sorry

end smallest_digit_for_divisibility_by_9_l120_120672


namespace trigonometric_expression_evaluation_l120_120102

theorem trigonometric_expression_evaluation
  (α : ℝ)
  (h1 : Real.tan α = -3 / 4) :
  (3 * Real.sin (α / 2) ^ 2 + 
   2 * Real.sin (α / 2) * Real.cos (α / 2) + 
   Real.cos (α / 2) ^ 2 - 2) / 
  (Real.sin (π / 2 + α) * Real.tan (-3 * π + α) + 
   Real.cos (6 * π - α)) = -7 := 
by 
  sorry
  -- This will skip the proof and ensure the Lean code can be built successfully.

end trigonometric_expression_evaluation_l120_120102


namespace coin_collection_l120_120868

def initial_ratio (G S : ℕ) : Prop := G = S / 3
def new_ratio (G S : ℕ) (addedG : ℕ) : Prop := G + addedG = S / 2
def total_coins_after (G S addedG : ℕ) : ℕ := G + addedG + S

theorem coin_collection (G S : ℕ) (addedG : ℕ) 
  (h1 : initial_ratio G S) 
  (h2 : addedG = 15) 
  (h3 : new_ratio G S addedG) : 
  total_coins_after G S addedG = 135 := 
by {
  sorry
}

end coin_collection_l120_120868


namespace largest_independent_subsets_l120_120093

theorem largest_independent_subsets {n : ℕ} : 
    (∀ (a b : Finset (Fin n)), a ⊆ b → a = b) → 
    set.card (Finset.powerset (Fin n) \ {s ∈ Finset (Fin n) | ∃ t, t ∈ Finset.powerset (Fin n) ∧ t ⊂ s}) = nat.choose n (n/2) :=
by
  sorry

end largest_independent_subsets_l120_120093


namespace at_least_one_le_one_l120_120746

theorem at_least_one_le_one (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end at_least_one_le_one_l120_120746


namespace sin_minus_cos_eq_one_sol_l120_120074

theorem sin_minus_cos_eq_one_sol (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = 1) :
  x = Real.pi / 2 ∨ x = Real.pi :=
sorry

end sin_minus_cos_eq_one_sol_l120_120074


namespace problem_l120_120530

theorem problem (x : ℝ) (h : 3 * x^2 - 2 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2 / 3) = 3 :=
by
  sorry

end problem_l120_120530


namespace find_parallel_and_perpendicular_lines_through_A_l120_120233

def point_A : ℝ × ℝ := (2, 2)

def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 20 = 0

def parallel_line_l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 14 = 0

def perpendicular_line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y - 2 = 0

theorem find_parallel_and_perpendicular_lines_through_A :
  (∀ x y, line_l x y → parallel_line_l1 x y) ∧
  (∀ x y, line_l x y → perpendicular_line_l2 x y) :=
by
  sorry

end find_parallel_and_perpendicular_lines_through_A_l120_120233


namespace find_b_l120_120075

variable (x : ℝ)

noncomputable def d : ℝ := 3

theorem find_b (b c : ℝ) :
  (7 * x^2 - 5 * x + 11 / 4) * (d * x^2 + b * x + c) = 21 * x^4 - 26 * x^3 + 34 * x^2 - 55 / 4 * x + 33 / 4 →
  b = -11 / 7 :=
by
  sorry

end find_b_l120_120075


namespace gazprom_rd_expense_l120_120519

theorem gazprom_rd_expense
  (R_and_D_t : ℝ) (ΔAPL_t_plus_1 : ℝ)
  (h1 : R_and_D_t = 3289.31)
  (h2 : ΔAPL_t_plus_1 = 1.55) :
  R_and_D_t / ΔAPL_t_plus_1 = 2122 := 
by
  sorry

end gazprom_rd_expense_l120_120519


namespace paint_house_18_women_4_days_l120_120127

theorem paint_house_18_women_4_days :
  (∀ (m1 m2 : ℕ) (d1 d2 : ℕ), m1 * d1 = m2 * d2) →
  (12 * 6 = 72) →
  (72 = 18 * d) →
  d = 4.0 :=
by
  sorry

end paint_house_18_women_4_days_l120_120127


namespace at_least_one_term_le_one_l120_120747

theorem at_least_one_term_le_one
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) :
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
  sorry

end at_least_one_term_le_one_l120_120747


namespace delores_money_left_l120_120369

def initial : ℕ := 450
def computer_cost : ℕ := 400
def printer_cost : ℕ := 40
def money_left (initial computer_cost printer_cost : ℕ) : ℕ := initial - (computer_cost + printer_cost)

theorem delores_money_left : money_left initial computer_cost printer_cost = 10 := by
  sorry

end delores_money_left_l120_120369


namespace fundraiser_total_money_l120_120987

def number_of_items (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student : ℕ) : ℕ :=
  (students1 * brownies_per_student) + (students2 * cookies_per_student) + (students3 * donuts_per_student)

def total_money_raised (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) : ℕ :=
  number_of_items students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student * price_per_item

theorem fundraiser_total_money (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) :
  students1 = 30 → students2 = 20 → students3 = 15 → brownies_per_student = 12 → cookies_per_student = 24 → donuts_per_student = 12 → price_per_item = 2 → 
  total_money_raised students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item = 2040 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end fundraiser_total_money_l120_120987


namespace largest_coins_l120_120614

theorem largest_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n = 146 :=
by
  sorry

end largest_coins_l120_120614


namespace expression_value_l120_120303

theorem expression_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 :=
by
  sorry

end expression_value_l120_120303


namespace factorial_division_l120_120956

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l120_120956


namespace set_notation_nat_lt_3_l120_120457

theorem set_notation_nat_lt_3 : {x : ℕ | x < 3} = {0, 1, 2} := 
sorry

end set_notation_nat_lt_3_l120_120457


namespace ratio_in_sequence_l120_120109

theorem ratio_in_sequence (a1 a2 b1 b2 b3 : ℝ)
  (h1 : ∃ d, a1 = 1 + d ∧ a2 = 1 + 2 * d ∧ 9 = 1 + 3 * d)
  (h2 : ∃ r, b1 = 1 * r ∧ b2 = 1 * r^2 ∧ b3 = 1 * r^3 ∧ 9 = 1 * r^4) :
  b2 / (a1 + a2) = 3 / 10 := by
  sorry

end ratio_in_sequence_l120_120109


namespace rectangle_measurement_error_l120_120733

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) 
  (h1 : 0 < L) (h2 : 0 < W) 
  (h3 : A = L * W)
  (h4 : A' = L * (1 + x / 100) * W * (1 - 4 / 100))
  (h5 : A' = A * (100.8 / 100)) :
  x = 5 :=
by
  sorry

end rectangle_measurement_error_l120_120733


namespace pyramid_total_area_l120_120609

/-- The total area of the four triangular faces of a right, square-based pyramid
whose base edges measure 8 units and lateral edges measure 7 units is 16√33. -/
theorem pyramid_total_area :
  let base_edge := 8
  let lateral_edge := 7
  4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 16 * Real.sqrt 33 := 
by
  let base_edge := 8
  let lateral_edge := 7
  have h1 : 4 * (1 / 2 * base_edge * Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 
              4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) := 
    by sorry
  have h2 : 4 * (1 / 2 * 8 * Real.sqrt (49 - 16)) = 4 * (4 * Real.sqrt 33) := 
    by sorry
  have h3 : 4 * (4 * Real.sqrt 33) = 16 * Real.sqrt 33 := 
    by sorry
  exact eq.trans (eq.trans h1 h2) h3

end pyramid_total_area_l120_120609


namespace find_range_of_a_l120_120148

def setA (x : ℝ) : Prop := 1 < x ∧ x < 2
def setB (x : ℝ) : Prop := 3 / 2 < x ∧ x < 4
def setUnion (x : ℝ) : Prop := 1 < x ∧ x < 4
def setP (a x : ℝ) : Prop := a < x ∧ x < a + 2

theorem find_range_of_a (a : ℝ) :
  (∀ x, setP a x → setUnion x) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end find_range_of_a_l120_120148


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l120_120704

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l120_120704


namespace quadrilateral_area_l120_120391

/-
Proof Statement: For a square with a side length of 8 cm, each of whose sides is divided by a point into two equal segments, 
prove that the area of the quadrilateral formed by connecting these points is 32 cm².
-/

theorem quadrilateral_area (side_len : ℝ) (h : side_len = 8) :
  let quadrilateral_area := (side_len * side_len) / 2
  quadrilateral_area = 32 :=
by
  sorry

end quadrilateral_area_l120_120391


namespace miquels_theorem_l120_120789

-- Define a triangle ABC with points D, E, F on sides BC, CA, and AB respectively
variables {A B C D E F : Type}

-- Assume we have a function that checks for collinearity of points
def is_on_side (X Y Z: Type) : Bool := sorry

-- Assume a function that returns the circumcircle of a triangle formed by given points
def circumcircle (X Y Z: Type) : Type := sorry 

-- Define the function that checks the intersection of circumcircles
def have_common_point (circ1 circ2 circ3: Type) : Bool := sorry

-- The theorem statement
theorem miquels_theorem (A B C D E F : Type) 
  (hD: is_on_side D B C) 
  (hE: is_on_side E C A) 
  (hF: is_on_side F A B) : 
  have_common_point (circumcircle A E F) (circumcircle B D F) (circumcircle C D E) :=
sorry

end miquels_theorem_l120_120789


namespace largest_sample_number_l120_120123

theorem largest_sample_number (n : ℕ) (start interval total : ℕ) (h1 : start = 7) (h2 : interval = 25) (h3 : total = 500) (h4 : n = total / interval) : 
(start + interval * (n - 1) = 482) :=
sorry

end largest_sample_number_l120_120123


namespace conversion_points_worth_two_l120_120871

theorem conversion_points_worth_two
  (touchdowns_per_game : ℕ := 4)
  (points_per_touchdown : ℕ := 6)
  (games_in_season : ℕ := 15)
  (total_touchdowns_scored : ℕ := touchdowns_per_game * games_in_season)
  (total_points_from_touchdowns : ℕ := total_touchdowns_scored * points_per_touchdown)
  (old_record_points : ℕ := 300)
  (points_above_record : ℕ := 72)
  (total_points_scored : ℕ := old_record_points + points_above_record)
  (conversions_scored : ℕ := 6)
  (total_points_from_conversions : ℕ := total_points_scored - total_points_from_touchdowns) :
  total_points_from_conversions / conversions_scored = 2 := by
sorry

end conversion_points_worth_two_l120_120871


namespace sufficient_but_not_necessary_condition_l120_120998

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 2) → (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 1 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l120_120998


namespace xiaoming_department_store_profit_l120_120472

theorem xiaoming_department_store_profit:
  let P₁ := 40000   -- average monthly profit in Q1
  let L₂ := -15000  -- average monthly loss in Q2
  let L₃ := -18000  -- average monthly loss in Q3
  let P₄ := 32000   -- average monthly profit in Q4
  let P_total := (P₁ * 3 + L₂ * 3 + L₃ * 3 + P₄ * 3)
  P_total = 117000 := by
  sorry

end xiaoming_department_store_profit_l120_120472


namespace tail_length_third_generation_l120_120282

theorem tail_length_third_generation (initial_length : ℕ) (growth_rate : ℕ) :
  initial_length = 16 ∧ growth_rate = 25 → 
  let sec_len := initial_length * (100 + growth_rate) / 100 in
  let third_len := sec_len * (100 + growth_rate) / 100 in
  third_len = 25 := by
  intros h
  sorry

end tail_length_third_generation_l120_120282


namespace card_sequence_probability_l120_120595

-- Let's define the conditions
def deck_size : ℕ := 52
def hearts_count : ℕ := 13
def spades_count : ℕ := 13
def clubs_count : ℕ := 13

-- Define the probability of the desired sequence of cards being drawn.
def desired_probability : ℚ := (13 : ℚ) / deck_size * (13 : ℚ) / (deck_size - 1) * (13 : ℚ) / (deck_size - 2)

-- The statement that needs to be proved.
theorem card_sequence_probability :
  desired_probability = 2197 / 132600 :=
by
  sorry

end card_sequence_probability_l120_120595


namespace negation_of_universal_proposition_l120_120234

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 2 → x^3 - 8 > 0)) ↔ (∃ x : ℝ, x > 2 ∧ x^3 - 8 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l120_120234


namespace find_m_given_slope_condition_l120_120238

variable (m : ℝ)

theorem find_m_given_slope_condition
  (h : (m - 4) / (3 - 2) = 1) : m = 5 :=
sorry

end find_m_given_slope_condition_l120_120238


namespace negation_of_existence_statement_l120_120311

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x^2 - 8 * x + 18 < 0) ↔ (∀ x : ℝ, x^2 - 8 * x + 18 ≥ 0) :=
by
  sorry

end negation_of_existence_statement_l120_120311


namespace car_speed_l120_120140

theorem car_speed {vp vc : ℚ} (h1 : vp = 7 / 2) (h2 : vc = 6 * vp) : 
  vc = 21 := 
by 
  sorry

end car_speed_l120_120140


namespace necessary_but_not_sufficient_condition_l120_120532

-- Given conditions and translated inequalities
variable {x : ℝ}
variable (h_pos : 0 < x) (h_bound : x < π / 2)
variable (h_sin_pos : 0 < Real.sin x) (h_sin_bound : Real.sin x < 1)

-- Define the inequalities we are dealing with
def ineq_1 (x : ℝ) := Real.sqrt x - 1 / Real.sin x < 0
def ineq_2 (x : ℝ) := 1 / Real.sin x - x > 0

-- The main proof statement
theorem necessary_but_not_sufficient_condition 
  (h1 : ineq_1 x) 
  (hx : 0 < x) (hπ : x < π/2) : 
  ineq_2 x → False := by
  sorry

end necessary_but_not_sufficient_condition_l120_120532


namespace points_on_parabola_l120_120993

theorem points_on_parabola (a : ℝ) (y1 y2 y3 : ℝ) 
  (h_a : a < -1) 
  (h1 : y1 = (a - 1)^2) 
  (h2 : y2 = a^2) 
  (h3 : y3 = (a + 1)^2) : 
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end points_on_parabola_l120_120993


namespace rectangle_area_ratio_l120_120925

theorem rectangle_area_ratio (s x y : ℝ) (h_square : s > 0)
    (h_side_ae : x > 0) (h_side_ag : y > 0)
    (h_ratio_area : x * y = (1 / 4) * s^2) :
    ∃ (r : ℝ), r > 0 ∧ r = x / y := 
sorry

end rectangle_area_ratio_l120_120925


namespace no_absolute_winner_prob_l120_120499

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end no_absolute_winner_prob_l120_120499


namespace common_ratio_q_l120_120906

noncomputable def Sn (n : ℕ) (a1 q : ℝ) := a1 * (1 - q^n) / (1 - q)

theorem common_ratio_q (a1 : ℝ) (q : ℝ) (h : q ≠ 1) (h1 : 6 * Sn 4 a1 q = Sn 5 a1 q + 5 * Sn 6 a1 q) : q = -6/5 := by
  sorry

end common_ratio_q_l120_120906


namespace find_sum_l120_120493

theorem find_sum (P : ℕ) (h_total : P * (4/100 + 6/100 + 8/100) = 2700) : P = 15000 :=
by
  sorry

end find_sum_l120_120493


namespace non_empty_subsets_count_l120_120111

def odd_set : Finset ℕ := {1, 3, 5, 7, 9}
def even_set : Finset ℕ := {2, 4, 6, 8}

noncomputable def num_non_empty_subsets_odd : ℕ := 2 ^ odd_set.card - 1
noncomputable def num_non_empty_subsets_even : ℕ := 2 ^ even_set.card - 1

theorem non_empty_subsets_count :
  num_non_empty_subsets_odd + num_non_empty_subsets_even = 46 :=
by sorry

end non_empty_subsets_count_l120_120111


namespace population_size_in_15th_year_l120_120769

theorem population_size_in_15th_year
  (a : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = a * Real.logb 2 (x + 1))
  (h2 : y 1 = 100) :
  y 15 = 400 :=
by
  sorry

end population_size_in_15th_year_l120_120769


namespace isosceles_triangle_circle_area_l120_120480

theorem isosceles_triangle_circle_area 
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : a = 4) 
  (h3 : c = 3) 
  (h4 : a = 4) 
  (h5 : b = 4)
  (h6 : c ≠ a)
  (h7 : c ≠ b) :
  let r := 4 in π * r ^ 2 = 16 * π :=
by
  sorry

end isosceles_triangle_circle_area_l120_120480


namespace compute_fraction_power_l120_120656

theorem compute_fraction_power : (45000 ^ 3 / 15000 ^ 3) = 27 :=
by
  sorry

end compute_fraction_power_l120_120656


namespace range_of_m_minimum_value_ab_l120_120891

-- Define the given condition as a predicate on the real numbers
def domain_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0

-- Define the first part of the proof problem: range of m
theorem range_of_m :
  (∀ m : ℝ, domain_condition m) → ∀ m : ℝ, m ≤ 6 :=
sorry

-- Define the second part of the proof problem: minimum value of 4a + 7b
theorem minimum_value_ab (n : ℝ) (a b : ℝ) (h : n = 6) :
  (∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (4 / (a + 5 * b) + 1 / (3 * a + 2 * b) = n)) → 
  ∃ (a b : ℝ), 4 * a + 7 * b = 3 / 2 :=
sorry

end range_of_m_minimum_value_ab_l120_120891


namespace circle_area_difference_l120_120539

noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_difference :
  let radius1 := 20
  let diameter2 := 20
  let radius2 := diameter2 / 2
  area radius1 - area radius2 = 300 * Real.pi :=
by
  sorry

end circle_area_difference_l120_120539


namespace ROI_difference_l120_120815

-- Definitions based on the conditions
def Emma_investment : ℝ := 300
def Briana_investment : ℝ := 500
def Emma_yield : ℝ := 0.15
def Briana_yield : ℝ := 0.10
def years : ℕ := 2

-- The goal is to prove that the difference between their 2-year ROI is $10
theorem ROI_difference :
  let Emma_ROI := Emma_investment * Emma_yield * years
  let Briana_ROI := Briana_investment * Briana_yield * years
  (Briana_ROI - Emma_ROI) = 10 :=
by
  sorry

end ROI_difference_l120_120815


namespace boxes_contain_same_number_of_apples_l120_120577

theorem boxes_contain_same_number_of_apples (total_apples boxes : ℕ) (h1 : total_apples = 49) (h2 : boxes = 7) : 
  total_apples / boxes = 7 :=
by
  sorry

end boxes_contain_same_number_of_apples_l120_120577


namespace greatest_integer_with_gcd_6_l120_120172

theorem greatest_integer_with_gcd_6 (x : ℕ) :
  x < 150 ∧ gcd x 12 = 6 → x = 138 :=
by
  sorry

end greatest_integer_with_gcd_6_l120_120172


namespace car_owners_without_motorcycles_l120_120124

theorem car_owners_without_motorcycles
  (total_adults : ℕ)
  (car_owners : ℕ)
  (motorcycle_owners : ℕ)
  (all_owners : total_adults = 400)
  (john_owns_cars : car_owners = 370)
  (john_owns_motorcycles : motorcycle_owners = 50)
  (all_adult_owners : total_adults = car_owners + motorcycle_owners - (car_owners - motorcycle_owners)) : 
  (car_owners - (car_owners + motorcycle_owners - total_adults) = 350) :=
by {
  sorry
}

end car_owners_without_motorcycles_l120_120124


namespace calc_2002_sq_minus_2001_mul_2003_l120_120064

theorem calc_2002_sq_minus_2001_mul_2003 : 2002 ^ 2 - 2001 * 2003 = 1 := 
by
  sorry

end calc_2002_sq_minus_2001_mul_2003_l120_120064


namespace Alton_profit_l120_120497

variable (earnings_per_day : ℕ)
variable (days_per_week : ℕ)
variable (rent_per_week : ℕ)

theorem Alton_profit (h1 : earnings_per_day = 8) (h2 : days_per_week = 7) (h3 : rent_per_week = 20) :
  earnings_per_day * days_per_week - rent_per_week = 36 := 
by sorry

end Alton_profit_l120_120497


namespace factorial_div_sum_l120_120967

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l120_120967


namespace min_value_of_function_l120_120668

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  (x + 1/x + x^2 + 1/x^2 + 1 / (x + 1/x + x^2 + 1/x^2)) = 4.25 := by
  sorry

end min_value_of_function_l120_120668


namespace johns_improvement_l120_120558

-- Declare the variables for the initial and later lap times.
def initial_minutes : ℕ := 50
def initial_laps : ℕ := 25
def later_minutes : ℕ := 54
def later_laps : ℕ := 30

-- Calculate the initial and later lap times in seconds, and the improvement.
def initial_lap_time_seconds := (initial_minutes * 60) / initial_laps 
def later_lap_time_seconds := (later_minutes * 60) / later_laps
def improvement := initial_lap_time_seconds - later_lap_time_seconds

-- State the theorem to prove the improvement is 12 seconds per lap.
theorem johns_improvement : improvement = 12 := by
  sorry

end johns_improvement_l120_120558


namespace number_of_real_solutions_eq_2_l120_120661

theorem number_of_real_solutions_eq_2 :
  ∃! (x : ℝ), (6 * x) / (x^2 + 2 * x + 5) + (7 * x) / (x^2 - 7 * x + 5) = -5 / 3 :=
sorry

end number_of_real_solutions_eq_2_l120_120661


namespace distance_between_trains_l120_120598

theorem distance_between_trains
  (v1 v2 : ℕ) (d_diff : ℕ)
  (h_v1 : v1 = 50) (h_v2 : v2 = 60) (h_d_diff : d_diff = 100) :
  ∃ d, d = 1100 :=
by
  sorry

-- Explanation:
-- v1 is the speed of the first train.
-- v2 is the speed of the second train.
-- d_diff is the difference in the distances traveled by the two trains at the time of meeting.
-- h_v1 states that the speed of the first train is 50 kmph.
-- h_v2 states that the speed of the second train is 60 kmph.
-- h_d_diff states that the second train travels 100 km more than the first train.
-- The existential statement asserts that there exists a distance d such that d equals 1100 km.

end distance_between_trains_l120_120598


namespace nested_expression_rational_count_l120_120082

theorem nested_expression_rational_count : 
  let count := Nat.card {n : ℕ // 1 ≤ n ∧ n ≤ 2021 ∧ ∃ m : ℕ, m % 2 = 1 ∧ m * m = 1 + 4 * n}
  count = 44 := 
by sorry

end nested_expression_rational_count_l120_120082


namespace max_a_2017_2018_ge_2017_l120_120875

def seq_a (a : ℕ → ℤ) (b : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ (∀ n, n ≥ 1 → 
  (b (n-1) = 1 → a (n+1) = a n * b n + a (n-1)) ∧ 
  (b (n-1) > 1 → a (n+1) = a n * b n - a (n-1)))

theorem max_a_2017_2018_ge_2017 (a : ℕ → ℤ) (b : ℕ → ℕ) (h : seq_a a b) :
  max (a 2017) (a 2018) ≥ 2017 :=
sorry

end max_a_2017_2018_ge_2017_l120_120875


namespace find_breadth_of_cuboid_l120_120981

variable (l : ℝ) (h : ℝ) (surface_area : ℝ) (b : ℝ)

theorem find_breadth_of_cuboid (hL : l = 10) (hH : h = 6) (hSA : surface_area = 480) 
  (hFormula : surface_area = 2 * (l * b + b * h + h * l)) : b = 11.25 := by
  sorry

end find_breadth_of_cuboid_l120_120981


namespace three_digit_integers_congruent_to_2_mod_4_l120_120712

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l120_120712


namespace find_a_l120_120751

-- Definitions of universal set U, set P, and complement of P in U
def U (a : ℤ) : Set ℤ := {2, 4, 3 - a^2}
def P (a : ℤ) : Set ℤ := {2, a^2 - a + 2}
def complement_U_P (a : ℤ) : Set ℤ := {-1}

-- The Lean statement asserting the conditions and the proof goal
theorem find_a (a : ℤ) (h_union : U a = P a ∪ complement_U_P a) : a = -1 :=
sorry

end find_a_l120_120751


namespace cubical_tank_fraction_filled_l120_120922

theorem cubical_tank_fraction_filled (a : ℝ) (h1 : ∀ a:ℝ, (a * a * 1 = 16) )
  : (1 / 4) = (16 / (a^3)) :=
by
  sorry

end cubical_tank_fraction_filled_l120_120922


namespace number_of_neutrons_eq_l120_120870

variable (A n x : ℕ)

/-- The number of neutrons N in the nucleus of an atom R, given that:
  1. A is the atomic mass number of R.
  2. The ion RO3^(n-) contains x outer electrons. -/
theorem number_of_neutrons_eq (N : ℕ) (h : A - N + 24 + n = x) : N = A + n + 24 - x :=
by sorry

end number_of_neutrons_eq_l120_120870


namespace m_equals_p_of_odd_prime_and_integers_l120_120290

theorem m_equals_p_of_odd_prime_and_integers (p m : ℕ) (x y : ℕ) (hp : p > 1 ∧ ¬ (p % 2 = 0)) 
    (hx : x > 1) (hy : y > 1) 
    (h : (x ^ p + y ^ p) / 2 = ((x + y) / 2) ^ m): 
    m = p := 
by 
  sorry

end m_equals_p_of_odd_prime_and_integers_l120_120290


namespace face_value_of_shares_l120_120485

-- Define the problem conditions
variables (F : ℝ) (D R : ℝ)

-- Assume conditions
axiom h1 : D = 0.155 * F
axiom h2 : R = 0.25 * 31
axiom h3 : D = R

-- State the theorem
theorem face_value_of_shares : F = 50 :=
by 
  -- Here should be the proof which we are skipping
  sorry

end face_value_of_shares_l120_120485


namespace mod_remainder_7_10_20_3_20_l120_120822

theorem mod_remainder_7_10_20_3_20 : (7 * 10^20 + 3^20) % 9 = 7 := sorry

end mod_remainder_7_10_20_3_20_l120_120822


namespace probability_three_white_two_black_eq_eight_seventeen_l120_120341
-- Import Mathlib library to access combinatorics functions.

-- Define the total number of white and black balls.
def total_white := 8
def total_black := 7

-- The key function to calculate combinations.
noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem conditions as constants.
def total_balls := total_white + total_black
def chosen_balls := 5
def white_balls_chosen := 3
def black_balls_chosen := 2

-- Calculate number of combinations.
noncomputable def total_combinations : ℕ := choose total_balls chosen_balls
noncomputable def white_combinations : ℕ := choose total_white white_balls_chosen
noncomputable def black_combinations : ℕ := choose total_black black_balls_chosen

-- Calculate the probability as a rational number.
noncomputable def probability_exact_three_white_two_black : ℚ :=
  (white_combinations * black_combinations : ℚ) / total_combinations

-- The theorem we want to prove
theorem probability_three_white_two_black_eq_eight_seventeen :
  probability_exact_three_white_two_black = 8 / 17 := by
  sorry

end probability_three_white_two_black_eq_eight_seventeen_l120_120341


namespace circle_radius_zero_l120_120821

theorem circle_radius_zero (x y : ℝ) :
  4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0 → ∃ c : ℝ × ℝ, ∃ r : ℝ, (x - c.1)^2 + (y - c.2)^2 = r^2 ∧ r = 0 :=
by
  sorry

end circle_radius_zero_l120_120821


namespace maximum_of_function_l120_120667

theorem maximum_of_function :
  ∃ x y : ℝ, 
    (1/3 ≤ x ∧ x ≤ 2/5 ∧ 1/4 ≤ y ∧ y ≤ 5/12) ∧ 
    (∀ x' y' : ℝ, 1/3 ≤ x' ∧ x' ≤ 2/5 ∧ 1/4 ≤ y' ∧ y' ≤ 5/12 → 
                (xy / (x^2 + y^2) ≤ x' * y' / (x'^2 + y'^2))) ∧ 
    (xy / (x^2 + y^2) = 20 / 41) := 
sorry

end maximum_of_function_l120_120667


namespace calc_expression_l120_120359

theorem calc_expression :
  let a := 3^456
  let b := 9^5 / 9^3
  a - b = 3^456 - 81 :=
by
  let a := 3^456
  let b := 9^5 / 9^3
  sorry

end calc_expression_l120_120359


namespace solve_quadratic_l120_120435

theorem solve_quadratic : ∃ x : ℝ, (x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2) :=
sorry

end solve_quadratic_l120_120435


namespace fraction_furniture_spent_l120_120566

theorem fraction_furniture_spent (S T : ℕ) (hS : S = 600) (hT : T = 300) : (S - T) / S = 1 / 2 :=
by
  sorry

end fraction_furniture_spent_l120_120566


namespace final_price_is_correct_l120_120652

-- Define the conditions as constants
def price_smartphone : ℝ := 300
def price_pc : ℝ := price_smartphone + 500
def price_tablet : ℝ := price_smartphone + price_pc
def total_price : ℝ := price_smartphone + price_pc + price_tablet
def discount : ℝ := 0.10 * total_price
def price_after_discount : ℝ := total_price - discount
def sales_tax : ℝ := 0.05 * price_after_discount
def final_price : ℝ := price_after_discount + sales_tax

-- Theorem statement asserting the final price value
theorem final_price_is_correct : final_price = 2079 := by sorry

end final_price_is_correct_l120_120652


namespace smallest_even_sum_equals_200_l120_120164

theorem smallest_even_sum_equals_200 :
  ∃ (x : ℤ), (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 200) ∧ (x = 36) :=
by
  sorry

end smallest_even_sum_equals_200_l120_120164


namespace tye_bills_l120_120780

theorem tye_bills : 
  ∀ (total_amount withdrawn_money_per_bank bill_value number_of_banks: ℕ), 
  withdrawn_money_per_bank = 300 → 
  bill_value = 20 → 
  number_of_banks = 2 → 
  total_amount = withdrawn_money_per_bank * number_of_banks → 
  (total_amount / bill_value) = 30 :=
by
  intros total_amount withdrawn_money_per_bank bill_value number_of_banks 
  intro h_withdrawn_eq_300
  intro h_bill_eq_20
  intro h_banks_eq_2
  intro h_total_eq_mult
  have h_total_eq_600 : total_amount = 600 := by rw [h_total_eq_mult, h_withdrawn_eq_300, h_banks_eq_2]; norm_num
  have h_bills_eq_30 := h_total_eq_600.symm ▸ div_eq_of_eq_mul_left (ne_of_gt (by norm_num : 0 < 20)) (by norm_num : 600 = 20 * 30)
  exact h_bills_eq_30
  sorry

end tye_bills_l120_120780


namespace hayley_stickers_l120_120930

theorem hayley_stickers (S F x : ℕ) (hS : S = 72) (hF : F = 9) (hx : x = S / F) : x = 8 :=
by
  sorry

end hayley_stickers_l120_120930


namespace Sue_waited_in_NY_l120_120582

-- Define the conditions as constants and assumptions
def T_NY_SF : ℕ := 24
def T_total : ℕ := 58
def T_NO_NY : ℕ := (3 * T_NY_SF) / 4

-- Define the waiting time
def T_wait : ℕ := T_total - T_NO_NY - T_NY_SF

-- Theorem stating the problem
theorem Sue_waited_in_NY :
  T_wait = 16 :=
by
  -- Implicitly using the given conditions
  sorry

end Sue_waited_in_NY_l120_120582


namespace SameFunction_l120_120919

noncomputable def f : ℝ → ℝ := λ x, x^2
noncomputable def g : ℝ → ℝ := λ x, (x^6)^(1/3)

theorem SameFunction : ∀ x : ℝ, f x = g x :=
by
  intro x
  sorry

end SameFunction_l120_120919


namespace machines_make_2550_copies_l120_120487

def total_copies (rate1 rate2 : ℕ) (time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

theorem machines_make_2550_copies :
  total_copies 30 55 30 = 2550 :=
by
  unfold total_copies
  decide

end machines_make_2550_copies_l120_120487


namespace pq_sum_eight_l120_120101

theorem pq_sum_eight
  (p q : ℤ)
  (hp1 : p > 1)
  (hq1 : q > 1)
  (hs1 : (2 * q - 1) % p = 0)
  (hs2 : (2 * p - 1) % q = 0) : p + q = 8 := 
sorry

end pq_sum_eight_l120_120101


namespace pyramid_triangular_face_area_l120_120602

theorem pyramid_triangular_face_area 
  (base : ℝ) (lateral : ℝ)
  (h_base : base = 8) (h_lateral : lateral = 7) :
  let height := sqrt (lateral^2 - (base / 2)^2) in
  4 * (1 / 2 * base * height) = 16 * sqrt 33 := 
by
  sorry

end pyramid_triangular_face_area_l120_120602


namespace alpha_and_2beta_l120_120995

theorem alpha_and_2beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (h_tan_alpha : Real.tan α = 1 / 8) (h_sin_beta : Real.sin β = 1 / 3) :
  α + 2 * β = Real.arctan (15 / 56) := by
  sorry

end alpha_and_2beta_l120_120995


namespace monotonicity_f_inequality_proof_l120_120878

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem monotonicity_f :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x < f (x + ε)) ∧ (∀ x : ℝ, 1 < x → f (x - ε) > f x) := 
sorry

theorem inequality_proof (x : ℝ) (hx : 1 < x) :
  1 < (x - 1) / Real.log x ∧ (x - 1) / Real.log x < x :=
sorry

end monotonicity_f_inequality_proof_l120_120878


namespace greatest_distance_P_D_l120_120190

noncomputable def greatest_distance_from_D (P : ℝ × ℝ) (A B C : ℝ × ℝ) (D : ℝ × ℝ) : ℝ :=
  let u := (P.1 - A.1)^2 + (P.2 - A.2)^2
  let v := (P.1 - B.1)^2 + (P.2 - B.2)^2
  let w := (P.1 - C.1)^2 + (P.2 - C.2)^2
  if u + v = w + 1 then ((P.1 - D.1)^2 + (P.2 - D.2)^2).sqrt else 0

theorem greatest_distance_P_D (P : ℝ × ℝ) (u v w : ℝ)
  (h1 : u^2 + v^2 = w^2 + 1) :
  greatest_distance_from_D P (0,0) (2,0) (2,2) (0,2) = 5 :=
sorry

end greatest_distance_P_D_l120_120190


namespace product_of_four_integers_l120_120837

theorem product_of_four_integers (A B C D : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) (h_pos_D : 0 < D)
  (h_sum : A + B + C + D = 36)
  (h_eq1 : A + 2 = B - 2)
  (h_eq2 : B - 2 = C * 2)
  (h_eq3 : C * 2 = D / 2) :
  A * B * C * D = 3840 :=
by
  sorry

end product_of_four_integers_l120_120837


namespace pyramid_volume_l120_120574

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * (EF * FG) * QE

theorem pyramid_volume
  (EF FG QE : ℝ)
  (h1 : EF = 10)
  (h2 : FG = 5)
  (h3 : QE = 9) :
  volume_of_pyramid EF FG QE = 150 :=
by
  simp [volume_of_pyramid, h1, h2, h3]
  sorry

end pyramid_volume_l120_120574


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l120_120711

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l120_120711


namespace calculate_average_fish_caught_l120_120861

-- Definitions based on conditions
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def Toph_fish : ℕ := 12

-- Total fish and average calculation
def total_fish : ℕ := Aang_fish + Sokka_fish + Toph_fish
def number_of_people : ℕ := 3
def average_fish_per_person : ℕ := total_fish / number_of_people

-- Theorem to prove
theorem calculate_average_fish_caught : average_fish_per_person = 8 := 
by 
  -- Proof steps are skipped with 'sorry', but the statement is set up correctly
  sorry

end calculate_average_fish_caught_l120_120861


namespace Mrs_Fredricksons_chickens_l120_120296

theorem Mrs_Fredricksons_chickens (C : ℕ) (h1 : 1/4 * C + 1/4 * (3/4 * C) = 35) : C = 80 :=
by
  sorry

end Mrs_Fredricksons_chickens_l120_120296


namespace find_y_l120_120542

theorem find_y (y : ℕ) (h : 2^10 = 32^y) : y = 2 :=
by {
  sorry
}

end find_y_l120_120542


namespace triangle_area_given_lines_l120_120660

theorem triangle_area_given_lines 
  (L1 : ℝ → ℝ) (L2 : ℝ → ℝ) (L3 : ℝ → ℝ)
  (hL1 : ∀ x, L1 x = 3)
  (hL2 : ∀ x, L2 x = 2 + 2 * x)
  (hL3 : ∀ x, L3 x = 2 - 2 * x) :
  let intersect1 := (0.5, 3)
  let intersect2 := (-0.5, 3)
  let intersect3 := (0, 2)
  Shoelace.area [intersect1, intersect2, intersect3] = 0.5 :=
by
  sorry

end triangle_area_given_lines_l120_120660


namespace determinant_is_zero_l120_120977

-- Define the matrix
def my_matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, x + z, y - z],
    ![1, x + y + z, y - z],
    ![1, x + z, x + y]]

-- Define the property to prove
theorem determinant_is_zero (x y z : ℝ) :
  Matrix.det (my_matrix x y z) = 0 :=
by sorry

end determinant_is_zero_l120_120977


namespace minimum_value_is_two_sqrt_two_l120_120078

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sqrt (x^2 + (2 - x)^2)) + (Real.sqrt ((2 - x)^2 + x^2))

theorem minimum_value_is_two_sqrt_two :
  ∃ x : ℝ, minimum_value_expression x = 2 * Real.sqrt 2 :=
by 
  sorry

end minimum_value_is_two_sqrt_two_l120_120078


namespace total_lobster_pounds_l120_120253

theorem total_lobster_pounds
  (combined_other_harbors : ℕ)
  (hooper_bay : ℕ)
  (H1 : combined_other_harbors = 160)
  (H2 : hooper_bay = 2 * combined_other_harbors) :
  combined_other_harbors + hooper_bay = 480 :=
by
  -- proof goes here
  sorry

end total_lobster_pounds_l120_120253


namespace quadratic_has_two_distinct_real_roots_l120_120469

-- Given the discriminant condition Δ = b^2 - 4ac > 0
theorem quadratic_has_two_distinct_real_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) := 
  sorry

end quadratic_has_two_distinct_real_roots_l120_120469


namespace total_area_of_triangular_faces_l120_120608

noncomputable def pyramid_base_edge : ℝ := 8
noncomputable def pyramid_lateral_edge : ℝ := 7

structure Pyramid where
  base_edge : ℝ
  lateral_edge : ℝ

def myPyramid : Pyramid :=
{ base_edge := pyramid_base_edge,
  lateral_edge := pyramid_lateral_edge }

theorem total_area_of_triangular_faces :
  4 * (1 / 2) * myPyramid.base_edge * (Real.sqrt (myPyramid.lateral_edge^2 - (myPyramid.base_edge / 2)^2)) = 16 * Real.sqrt 33 :=
by
  sorry

end total_area_of_triangular_faces_l120_120608


namespace symmetric_circle_eq_l120_120232

theorem symmetric_circle_eq (C_1_eq : ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 1)
    (line_eq : ∀ x y : ℝ, x - y - 2 = 0) :
    ∀ x y : ℝ, (x - 1)^2 + y^2 = 1 :=
sorry

end symmetric_circle_eq_l120_120232


namespace no_absolute_winner_prob_l120_120510

open_locale probability

-- Define the probability of Alyosha winning against Borya
def P_A_wins_B : ℝ := 0.6

-- Define the probability of Borya winning against Vasya
def P_B_wins_V : ℝ := 0.4

-- There are no ties, and each player plays with each other once
-- Conditions ensure that all pairs have played exactly once

-- Define the event that there will be no absolute winner
def P_no_absolute_winner : ℝ := P_A_wins_B * P_B_wins_V * 1 + P_A_wins_B * (1 - P_B_wins_V) * (1 - 1)

-- Statement of the problem: Prove that the probability of event C is 0.24
theorem no_absolute_winner_prob :
  P_no_absolute_winner = 0.24 :=
  by
    -- Placeholder for proof
    sorry

end no_absolute_winner_prob_l120_120510


namespace triangle_angle_sum_l120_120554

theorem triangle_angle_sum {A B C : Type} 
  (angle_ABC : ℝ) (angle_BAC : ℝ) (angle_BCA : ℝ) (x : ℝ) 
  (h1: angle_ABC = 90) 
  (h2: angle_BAC = 3 * x) 
  (h3: angle_BCA = x + 10)
  : x = 20 :=
by
  sorry

end triangle_angle_sum_l120_120554


namespace total_games_played_l120_120762

def games_lost : ℕ := 4
def games_won : ℕ := 8

theorem total_games_played : games_lost + games_won = 12 :=
by
  -- Proof is omitted
  sorry

end total_games_played_l120_120762


namespace candy_mixture_l120_120636

theorem candy_mixture (x : ℝ) (h1 : x * 3 + 64 * 2 = (x + 64) * 2.2) : x + 64 = 80 :=
by sorry

end candy_mixture_l120_120636


namespace Cindy_hourly_rate_l120_120362

theorem Cindy_hourly_rate
    (num_courses : ℕ)
    (weekly_hours : ℕ) 
    (monthly_earnings : ℕ) 
    (weeks_in_month : ℕ)
    (monthly_hours_per_course : ℕ)
    (hourly_rate : ℕ) :
    num_courses = 4 →
    weekly_hours = 48 →
    monthly_earnings = 1200 →
    weeks_in_month = 4 →
    monthly_hours_per_course = (weekly_hours / num_courses) * weeks_in_month →
    hourly_rate = monthly_earnings / monthly_hours_per_course →
    hourly_rate = 25 := by
  sorry

end Cindy_hourly_rate_l120_120362


namespace jacket_initial_reduction_l120_120456

theorem jacket_initial_reduction (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.9 * 1.481481481481481 = P → x = 25 :=
by
  sorry

end jacket_initial_reduction_l120_120456


namespace total_handshakes_eq_900_l120_120516

def num_boys : ℕ := 25
def handshakes_per_pair : ℕ := 3

theorem total_handshakes_eq_900 : (num_boys * (num_boys - 1) / 2) * handshakes_per_pair = 900 := by
  sorry

end total_handshakes_eq_900_l120_120516


namespace word_count_with_a_l120_120271

-- Defining the constants for the problem
def alphabet_size : ℕ := 26
def no_a_size : ℕ := 25

-- Calculating words that contain 'A' for lengths 1 to 5
def words_with_a (len : ℕ) : ℕ :=
  alphabet_size ^ len - no_a_size ^ len

-- The main theorem statement
theorem word_count_with_a : words_with_a 1 + words_with_a 2 + words_with_a 3 + words_with_a 4 + words_with_a 5 = 2186085 :=
by
  -- Calculations are established in the problem statement
  sorry

end word_count_with_a_l120_120271


namespace neither_sufficient_nor_necessary_l120_120876

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem neither_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  ¬ ((q > 1) ↔ is_increasing_sequence a) :=
sorry

end neither_sufficient_nor_necessary_l120_120876


namespace bus_people_final_count_l120_120793

theorem bus_people_final_count (initial_people : ℕ) (people_on : ℤ) (people_off : ℤ) :
  initial_people = 22 → people_on = 4 → people_off = -8 → initial_people + people_on + people_off = 18 :=
by
  intro h_initial h_on h_off
  rw [h_initial, h_on, h_off]
  norm_num

end bus_people_final_count_l120_120793


namespace pyramid_area_l120_120604

theorem pyramid_area :
  ∀ (a b : ℝ), a = 8 → b = 7 → 4 * (1/2 * a * sqrt (b^2 - (a/2)^2)) = 16 * sqrt 33 :=
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : a / 2 = 4 := by norm_num [ha]
  have h2 : b^2 - (a / 2)^2 = 33 :=
    by
      calc
        b^2 - (a / 2)^2 = 49 - 16 := by norm_num [hb]
        ... = 33 := by norm_num
  rw [h1, h2, sqrt 33, mul_one, mul_one, half_mul, mul_comm (1/2) a, ←mul_assoc, mul_comm 4 4]
  norm_num
  sorry

end pyramid_area_l120_120604


namespace nine_chapters_problem_l120_120179

theorem nine_chapters_problem (n x : ℤ) (h1 : 8 * n = x + 3) (h2 : 7 * n = x - 4) :
  (x + 3) / 8 = (x - 4) / 7 :=
  sorry

end nine_chapters_problem_l120_120179


namespace parameterization_function_l120_120452

theorem parameterization_function (f : ℝ → ℝ) 
  (parameterized_line : ∀ t : ℝ, (f t, 20 * t - 10))
  (line_eq : ∀ x y : ℝ, y = 2 * x - 30) :
  f = λ t, 10 * t + 10 :=
by
  sorry

end parameterization_function_l120_120452


namespace youngest_person_age_l120_120774

theorem youngest_person_age (total_age_now : ℕ) (total_age_when_born : ℕ) (Y : ℕ) (h1 : total_age_now = 210) (h2 : total_age_when_born = 162) : Y = 48 :=
by
  sorry

end youngest_person_age_l120_120774


namespace largest_number_of_gold_coins_l120_120617

theorem largest_number_of_gold_coins 
  (num_friends : ℕ)
  (extra_coins : ℕ)
  (total_coins : ℕ) :
  num_friends = 13 →
  extra_coins = 3 →
  total_coins < 150 →
  ∀ k : ℕ, total_coins = num_friends * k + extra_coins →
  total_coins ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l120_120617


namespace problem_1_problem_2_l120_120247

theorem problem_1 (a : ℝ) : (∀ x1 x2 : ℝ, (a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) → x1 = x2) → (a = 0 ∨ a = 1) :=
by sorry

theorem problem_2 (a : ℝ) : (∀ x1 x2 : ℝ, (a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) → x1 = x2 ∨ ¬ ∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a ≥ 1 ∨ a = 0) :=
by sorry

end problem_1_problem_2_l120_120247


namespace garden_length_l120_120333

-- Define the perimeter and breadth
def perimeter : ℕ := 900
def breadth : ℕ := 190

-- Define a function to calculate the length using given conditions
def length (P : ℕ) (B : ℕ) : ℕ := (P / 2) - B

-- Theorem stating that for the given perimeter and breadth, the length is 260.
theorem garden_length : length perimeter breadth = 260 :=
by
  -- placeholder for proof
  sorry

end garden_length_l120_120333


namespace pyramid_volume_l120_120573

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * (EF * FG) * QE

theorem pyramid_volume
  (EF FG QE : ℝ)
  (h1 : EF = 10)
  (h2 : FG = 5)
  (h3 : QE = 9) :
  volume_of_pyramid EF FG QE = 150 :=
by
  simp [volume_of_pyramid, h1, h2, h3]
  sorry

end pyramid_volume_l120_120573


namespace greatest_number_of_bouquets_l120_120888

/--
Sara has 42 red flowers, 63 yellow flowers, and 54 blue flowers.
She wants to make bouquets with the same number of each color flower in each bouquet.
Prove that the greatest number of bouquets she can make is 21.
-/
theorem greatest_number_of_bouquets (red yellow blue : ℕ) (h_red : red = 42) (h_yellow : yellow = 63) (h_blue : blue = 54) :
  Nat.gcd (Nat.gcd red yellow) blue = 21 :=
by
  rw [h_red, h_yellow, h_blue]
  sorry

end greatest_number_of_bouquets_l120_120888


namespace problem_statement_l120_120561

theorem problem_statement (r p q : ℝ) (h1 : r > 0) (h2 : p * q ≠ 0) (h3 : p^2 * r > q^2 * r) : p^2 > q^2 := 
sorry

end problem_statement_l120_120561


namespace intersection_eq_two_l120_120631

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

theorem intersection_eq_two : A ∩ B = {2} := by
  sorry

end intersection_eq_two_l120_120631


namespace koala_fiber_eaten_l120_120286

-- Definitions based on conditions
def absorbs_percentage : ℝ := 0.40
def fiber_absorbed : ℝ := 12

-- The theorem statement to prove the total amount of fiber eaten
theorem koala_fiber_eaten : 
  (fiber_absorbed / absorbs_percentage) = 30 :=
by 
  sorry

end koala_fiber_eaten_l120_120286


namespace simplify_logical_expression_l120_120145

variables (A B C : Bool)

theorem simplify_logical_expression :
  (A && !B || B && !C || B && C || A && B) = (A || B) :=
by { sorry }

end simplify_logical_expression_l120_120145


namespace rectangle_ratio_l120_120934

noncomputable def ratio_of_sides (a b : ℝ) : ℝ := a / b

theorem rectangle_ratio (a b d : ℝ) (h1 : d = Real.sqrt (a^2 + b^2)) (h2 : (a/b)^2 = b/d) : 
  ratio_of_sides a b = (Real.sqrt 5 - 1) / 3 :=
by sorry

end rectangle_ratio_l120_120934


namespace smallest_n_circle_l120_120884

theorem smallest_n_circle (n : ℕ) 
    (h1 : ∀ i j : ℕ, i < j → j - i = 3 ∨ j - i = 4 ∨ j - i = 5) :
    n = 7 :=
sorry

end smallest_n_circle_l120_120884


namespace cost_price_of_computer_table_l120_120591

variable (C : ℝ) (SP : ℝ)
variable (h1 : SP = 5400)
variable (h2 : SP = C * 1.32)

theorem cost_price_of_computer_table : C = 5400 / 1.32 :=
by
  -- We are required to prove C = 5400 / 1.32
  sorry

end cost_price_of_computer_table_l120_120591


namespace brianna_books_gift_l120_120803

theorem brianna_books_gift (books_per_month : ℕ) (months_per_year : ℕ) (books_bought : ℕ) 
  (borrow_difference : ℕ) (books_reread : ℕ) (total_books_needed : ℕ) : 
  (books_per_month * months_per_year = total_books_needed) →
  ((books_per_month * months_per_year) - books_reread - 
  (books_bought + (books_bought - borrow_difference)) = 
  books_given) →
  books_given = 6 := 
by
  intro h1 h2
  sorry

end brianna_books_gift_l120_120803


namespace correct_operation_l120_120785

theorem correct_operation (a b : ℝ) : 2 * a^2 * b - a^2 * b = a^2 * b :=
by
  sorry

end correct_operation_l120_120785


namespace z_completion_time_l120_120629

variable (x y z : Type) [NormedField x] [NormedField y] [NormedField z]

theorem z_completion_time : 
  (x := 40) → (y := 30) → 
  (x_worked : ∀ (days_worked: ℕ) (total_days: ℕ), total_days = 40 → 
    days_worked = 8 → x := 1 / total_days * days_worked) →
  (y_and_z_wr: ∀ (days_worked: ℕ), (y := 1 / 30 + 1/z * days_worked) → 
    days_worked = 20 → (x := 8 / 40 * 8) → 
    4 / 5 = 20 * (1 / 30 + 1 / d)) → (d = 150) := 
  sorry

end z_completion_time_l120_120629


namespace window_width_is_28_l120_120662

noncomputable def window_width (y : ℝ) : ℝ :=
  12 * y + 4

theorem window_width_is_28 : ∃ (y : ℝ), window_width y = 28 :=
by
  -- The proof goes here
  sorry

end window_width_is_28_l120_120662


namespace negation_of_existential_prop_l120_120312

theorem negation_of_existential_prop :
  (¬ ∃ (x₀ : ℝ), x₀^2 + x₀ + 1 < 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_prop_l120_120312


namespace no_absolute_winner_prob_l120_120501

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end no_absolute_winner_prob_l120_120501


namespace total_books_to_put_away_l120_120128

-- Definitions based on the conditions
def books_per_shelf := 4
def shelves_needed := 3

-- The proof problem translates to finding the total number of books
theorem total_books_to_put_away : shelves_needed * books_per_shelf = 12 := by
  sorry

end total_books_to_put_away_l120_120128


namespace find_possible_values_of_b_l120_120418

def good_number (x : ℕ) : Prop :=
  ∃ p n : ℕ, Nat.Prime p ∧ n ≥ 2 ∧ x = p^n

theorem find_possible_values_of_b (b : ℕ) : 
  (b ≥ 4) ∧ good_number (b^2 - 2 * b - 3) ↔ b = 87 := sorry

end find_possible_values_of_b_l120_120418


namespace third_generation_tail_length_is_25_l120_120279

def first_generation_tail_length : ℝ := 16
def growth_rate : ℝ := 0.25

def second_generation_tail_length : ℝ := first_generation_tail_length * (1 + growth_rate)
def third_generation_tail_length : ℝ := second_generation_tail_length * (1 + growth_rate)

theorem third_generation_tail_length_is_25 :
  third_generation_tail_length = 25 := by
  sorry

end third_generation_tail_length_is_25_l120_120279


namespace reciprocal_real_roots_l120_120224

theorem reciprocal_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 * x2 = 1 ∧ x1 + x2 = 2 * (m + 2)) ∧ 
  (x1^2 - 2 * (m + 2) * x1 + (m^2 - 4) = 0) → m = Real.sqrt 5 := 
sorry

end reciprocal_real_roots_l120_120224


namespace four_students_three_classes_l120_120702

-- Define the function that calculates the number of valid assignments
def valid_assignments (students : ℕ) (classes : ℕ) : ℕ :=
  if students = 4 ∧ classes = 3 then 36 else 0  -- Using given conditions to return 36 when appropriate

-- Define the theorem to prove that there are 36 valid ways
theorem four_students_three_classes : valid_assignments 4 3 = 36 :=
  by
  -- The proof is not required, so we use sorry to skip it
  sorry

end four_students_three_classes_l120_120702


namespace tapanga_corey_candies_l120_120764

theorem tapanga_corey_candies (corey_candies : ℕ) (tapanga_candies : ℕ) 
                              (h1 : corey_candies = 29) 
                              (h2 : tapanga_candies = corey_candies + 8) : 
                              corey_candies + tapanga_candies = 66 :=
by
  rw [h1, h2]
  sorry

end tapanga_corey_candies_l120_120764


namespace polygon_interior_sum_sum_of_exterior_angles_l120_120690

theorem polygon_interior_sum (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

theorem sum_of_exterior_angles (n : ℕ) : 360 = 360 :=
by
  sorry

end polygon_interior_sum_sum_of_exterior_angles_l120_120690


namespace complement_of_67_is_23_l120_120445

-- Define complement function
def complement (x : ℝ) : ℝ := 90 - x

-- State the theorem
theorem complement_of_67_is_23 : complement 67 = 23 := 
by
  sorry

end complement_of_67_is_23_l120_120445


namespace enclosed_area_correct_l120_120022

noncomputable def enclosed_area_closed_curve : ℝ :=
  let r := 5 / 4 in  -- From the length of arcs
  let s := 3 in      -- Side length of the octagon
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2 in
  let sector_area := (5 * Real.pi * r^2) / 6 in
  octagon_area + 12 * sector_area / 2 - 12 * sector_area / 4 -- Approximate enclosed area

theorem enclosed_area_correct :
  enclosed_area_closed_curve = 54 + 54 * Real.sqrt 2 + Real.pi :=
by
  sorry

end enclosed_area_correct_l120_120022


namespace number_of_sets_A_l120_120386

/-- Given conditions about intersections and unions of set A, we want to find the number of 
  possible sets A that satisfy the given conditions. Specifically, prove the following:
  - A ∩ {-1, 0, 1} = {0, 1}
  - A ∪ {-2, 0, 2} = {-2, 0, 1, 2}
  Total number of such sets A is 4.
-/
theorem number_of_sets_A : ∃ (As : Finset (Finset ℤ)), 
  (∀ A ∈ As, A ∩ {-1, 0, 1} = {0, 1} ∧ A ∪ {-2, 0, 2} = {-2, 0, 1, 2}) ∧
  As.card = 4 := 
sorry

end number_of_sets_A_l120_120386


namespace no_absolute_winner_l120_120512

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end no_absolute_winner_l120_120512


namespace largest_angle_l120_120773

-- Definitions for our conditions
def right_angle : ℝ := 90
def sum_of_two_angles (a b : ℝ) : Prop := a + b = (4 / 3) * right_angle
def angle_difference (a b : ℝ) : Prop := b = a + 40

-- Statement of the problem to be proved
theorem largest_angle (a b c : ℝ) (h_sum : sum_of_two_angles a b) (h_diff : angle_difference a b) (h_triangle : a + b + c = 180) : c = 80 :=
by sorry

end largest_angle_l120_120773


namespace last_digit_11_power_11_last_digit_9_power_9_last_digit_9219_power_9219_last_digit_2014_power_2014_l120_120167

-- Definition of function to calculate the last digit of a number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Proof statements
theorem last_digit_11_power_11 : last_digit (11 ^ 11) = 1 := sorry

theorem last_digit_9_power_9 : last_digit (9 ^ 9) = 9 := sorry

theorem last_digit_9219_power_9219 : last_digit (9219 ^ 9219) = 9 := sorry

theorem last_digit_2014_power_2014 : last_digit (2014 ^ 2014) = 6 := sorry

end last_digit_11_power_11_last_digit_9_power_9_last_digit_9219_power_9219_last_digit_2014_power_2014_l120_120167


namespace part1_part2_l120_120732

noncomputable def choose (n : ℕ) (k : ℕ) : ℕ :=
  n.choose k

theorem part1 :
  let internal_medicine_doctors := 12
  let surgeons := 8
  let total_doctors := internal_medicine_doctors + surgeons
  let team_size := 5
  let doctors_left := total_doctors - 1 - 1 -- as one internal medicine must participate and one surgeon cannot
  choose doctors_left (team_size - 1) = 3060 := by
  sorry

theorem part2 :
  let internal_medicine_doctors := 12
  let surgeons := 8
  let total_doctors := internal_medicine_doctors + surgeons
  let team_size := 5
  let only_internal_medicine := choose internal_medicine_doctors team_size
  let only_surgeons := choose surgeons team_size
  let total_ways := choose total_doctors team_size
  total_ways - only_internal_medicine - only_surgeons = 14656 := by
  sorry

end part1_part2_l120_120732


namespace trapezoid_height_l120_120076

variables (a b h : ℝ)

def is_trapezoid (a b h : ℝ) (angle_diag : ℝ) (angle_ext : ℝ) : Prop :=
a < b ∧ angle_diag = 90 ∧ angle_ext = 45

theorem trapezoid_height
  (a b : ℝ) (ha : a < b)
  (angle_diag : ℝ) (h_angle_diag : angle_diag = 90)
  (angle_ext : ℝ) (h_angle_ext : angle_ext = 45)
  (h_def : is_trapezoid a b h angle_diag angle_ext) :
  h = a * b / (b - a) :=
sorry

end trapezoid_height_l120_120076


namespace remainder_when_four_times_n_minus_nine_divided_by_7_l120_120092

theorem remainder_when_four_times_n_minus_nine_divided_by_7 (n : ℤ) (h : n % 7 = 3) : (4 * n - 9) % 7 = 3 := by
  sorry

end remainder_when_four_times_n_minus_nine_divided_by_7_l120_120092


namespace area_of_circumscribed_circle_isosceles_triangle_l120_120479

theorem area_of_circumscribed_circle_isosceles_triangle :
  ∃ (r : ℝ), (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi) :=
by
  -- Consider the isosceles triangle conditions
  let a : ℝ := 4
  let b : ℝ := 4
  let c : ℝ := 3
  let BD := Real.sqrt(a^2 - (c/2)^2)
  let r := 8 / BD
  have h1 : BD = Real.sqrt 13.75 := by 
    -- Calculate the altitude BD
    calc
      BD = Real.sqrt(a^2 -  (c/2)^2) : rfl
      ... = Real.sqrt(16 - (3/2)^2) : rfl
      ... = Real.sqrt 13.75 : rfl
  
  use r
  have h2 : r = 8 / Real.sqrt 13.75 := by 
    -- Simplify the radius expression
    sorry

  have h3 : Real.pi * r ^ 2 = 256 / 55 * Real.pi := by 
    -- Calculate the area
    calc
      Real.pi * r ^ 2 = Real.pi * (8 / Real.sqrt 13.75) ^ 2 : by rw h2
      ... = Real.pi * (64 / 13.75) : by rw [pow_two, mul_div_assoc, mul_one, div_mul_div_same]
      ... = (256 / 54.6875) * Real.pi : by rw mul_comm
      ...   = (256 / 55) * Real.pi : by norm_num
    sorry
  
  show (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi),
  from ⟨h2, h3⟩
  sorry

end area_of_circumscribed_circle_isosceles_triangle_l120_120479


namespace number_of_children_proof_l120_120441

-- Let A be the number of mushrooms Anya has
-- Let V be the number of mushrooms Vitya has
-- Let S be the number of mushrooms Sasha has
-- Let xs be the list of mushrooms of other children

def mushrooms_distribution (A V S : ℕ) (xs : List ℕ) : Prop :=
  let n := 3 + xs.length
  -- First condition
  let total_mushrooms := A + V + S + xs.sum
  let equal_share := total_mushrooms / n
  (A / 2 = equal_share) ∧ (V + A / 2 = equal_share) ∧ (S = equal_share) ∧
  (∀ x ∈ xs, x = equal_share) ∧
  -- Second condition
  (S + A = V + xs.sum)

theorem number_of_children_proof (A V S : ℕ) (xs : List ℕ) :
  mushrooms_distribution A V S xs → 3 + xs.length = 6 :=
by
  intros h
  sorry

end number_of_children_proof_l120_120441


namespace find_x_parallel_find_x_perpendicular_l120_120249

def a (x : ℝ) : ℝ × ℝ := (x, x + 2)
def b : ℝ × ℝ := (1, 2)

-- Given that a vector is proportional to another
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Given that the dot product is zero
def are_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_x_parallel (x : ℝ) (h : are_parallel (a x) b) : x = 2 :=
by sorry

theorem find_x_perpendicular (x : ℝ) (h : are_perpendicular (a x - b) b) : x = (1 / 3 : ℝ) :=
by sorry

end find_x_parallel_find_x_perpendicular_l120_120249


namespace product_of_cubes_91_l120_120453

theorem product_of_cubes_91 :
  ∃ (a b : ℤ), (a = 3 ∨ a = 4) ∧ (b = 3 ∨ b = 4) ∧ (a^3 + b^3 = 91) ∧ (a * b = 12) :=
by
  sorry

end product_of_cubes_91_l120_120453


namespace total_selling_price_is_correct_l120_120057

def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

def discount : ℝ := discount_rate * original_price
def sale_price : ℝ := original_price - discount
def tax : ℝ := tax_rate * sale_price
def total_selling_price : ℝ := sale_price + tax

theorem total_selling_price_is_correct : total_selling_price = 96.6 := by
  sorry

end total_selling_price_is_correct_l120_120057


namespace algebra_eq_iff_sum_eq_one_l120_120149

-- Definitions from conditions
def expr1 (a b c : ℝ) : ℝ := a + b * c
def expr2 (a b c : ℝ) : ℝ := (a + b) * (a + c)

-- Lean statement for the proof problem
theorem algebra_eq_iff_sum_eq_one (a b c : ℝ) : expr1 a b c = expr2 a b c ↔ a + b + c = 1 :=
by
  sorry

end algebra_eq_iff_sum_eq_one_l120_120149


namespace find_x_l120_120679

/-- Given vectors a and b, and a is parallel to b -/
def vectors (x : ℝ) : Prop :=
  let a := (x, 2)
  let b := (2, 1)
  a.1 * b.2 = a.2 * b.1

theorem find_x: ∀ x : ℝ, vectors x → x = 4 :=
by
  intros x h
  sorry

end find_x_l120_120679


namespace solve_quadratic_l120_120436

theorem solve_quadratic : ∃ x : ℝ, (x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2) :=
sorry

end solve_quadratic_l120_120436


namespace speed_of_stream_l120_120634

variable (b s : ℝ)

-- Define the conditions from the problem
def downstream_condition := (100 : ℝ) / 4 = b + s
def upstream_condition := (75 : ℝ) / 15 = b - s

theorem speed_of_stream (h1 : downstream_condition b s) (h2: upstream_condition b s) : s = 10 := 
by 
  sorry

end speed_of_stream_l120_120634


namespace sum_of_roots_cubic_l120_120782

theorem sum_of_roots_cubic :
  let a := 3
  let b := 7
  let c := -12
  let d := -4
  let roots_sum := -(b / a)
  roots_sum = -2.33 :=
by
  sorry

end sum_of_roots_cubic_l120_120782
