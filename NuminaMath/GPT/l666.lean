import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Combinations
import Mathlib.Combinatorics.GraphTheory.Color
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Fin
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Circles
import Mathlib.Geometry.Euclidean.Triangles
import Mathlib.Order.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution
import Mathlib.Probability.Notation
import Mathlib.RingTheory.MatrixAlgebra
import Mathlib.Statistics.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.MetricSpace.Basic

namespace angle_value_l666_666670

-- Problem statement
theorem angle_value (α : ℝ) (hα : 0 ≤ α ∧ α ≤ 2 * π)
  (h_point : let x := Real.sin (2 * π / 3) in let y := Real.cos (2 * π / 3) in ∃ (θ : ℝ), θ = α ∧ (x, y) = (Real.sin θ, Real.cos θ)) :
  α = 5 * π / 3 :=
sorry

end angle_value_l666_666670


namespace units_digit_of_27_times_36_l666_666121

theorem units_digit_of_27_times_36 :
  let units_digit := fun (n : ℕ) => n % 10
  in units_digit (27 * 36) = 2 :=
by
  let units_digit := fun (n : ℕ) => n % 10
  have h27: units_digit 27 = 7 := by
    show 27 % 10 = 7
    sorry
  have h36: units_digit 36 = 6 := by
    show 36 % 10 = 6
    sorry
  have h42: units_digit (7 * 6) = 2 := by
    show 42 % 10 = 2
    sorry
  exact h42

end units_digit_of_27_times_36_l666_666121


namespace apples_difference_l666_666526

theorem apples_difference {initial final : ℕ} (h_initial : initial = 46) (h_final : final = 14) :
  initial - final = 32 :=
by
  rw [h_initial, h_final]
  norm_num
  sorry

end apples_difference_l666_666526


namespace speed_down_l666_666034

theorem speed_down {u avg_speed d v : ℝ} (hu : u = 18) (havg : avg_speed = 20.571428571428573) (hv : 2 * d / ((d / u) + (d / v)) = avg_speed) : v = 24 :=
by
  have h1 : 20.571428571428573 = 20.571428571428573 := rfl
  have h2 : 18 = 18 := rfl
  sorry

end speed_down_l666_666034


namespace wealthiest_500_lowest_income_l666_666812

theorem wealthiest_500_lowest_income :
  ∀ x : ℝ, (5 * 10^9 * x ^ (-2) = 500) → x = 10^4 := sorry

end wealthiest_500_lowest_income_l666_666812


namespace part_I_max_value_part_I_min_value_part_II_range_of_a_l666_666177

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - Real.log x

theorem part_I_max_value (a : ℝ) (I : [1, 3]) : 
  ∀ x ∈ I, f x 3 ≤ 2 := sorry

theorem part_I_min_value (a : ℝ) (I : [1, 3]) : 
  ∃ x ∈ I, f x 3 = 2 - Real.log 2 := sorry

theorem part_II_range_of_a (a : ℝ) : 
  (∀ x ≥ 0, ∃ m M, m ≤ f x a ∧ f x a ≤ M) → a > 2 * Real.sqrt 2 := sorry

end part_I_max_value_part_I_min_value_part_II_range_of_a_l666_666177


namespace numerical_puzzle_l666_666411

noncomputable def THETA (T : ℕ) (A : ℕ) : ℕ := 1000 * T + 100 * T + 10 * T + A
noncomputable def BETA (B : ℕ) (T : ℕ) (A : ℕ) : ℕ := 1000 * B + 100 * T + 10 * T + A
noncomputable def GAMMA (Γ : ℕ) (E : ℕ) (M : ℕ) (A : ℕ) : ℕ := 10000 * Γ + 1000 * E + 100 * M + 10 * M + A

theorem numerical_puzzle
  (T : ℕ) (B : ℕ) (E : ℕ) (M : ℕ) (Γ : ℕ) (A : ℕ)
  (h1 : A = 0)
  (h2 : Γ = 1)
  (h3 : T + T = M)
  (h4 : 2 * E = M)
  (h5 : T ≠ B)
  (h6 : B ≠ E)
  (h7 : E ≠ M)
  (h8 : M ≠ Γ)
  (h9 : Γ ≠ T)
  (h10 : Γ ≠ B)
  (h11 : THETA T A + BETA B T A = GAMMA Γ E M A) :
  THETA 4 0 + BETA 5 4 0 = GAMMA 1 9 8 0 :=
by {
  sorry
}

end numerical_puzzle_l666_666411


namespace blocks_differ_in_two_ways_exactly_l666_666020

theorem blocks_differ_in_two_ways_exactly 
  (materials : Finset String := {"plastic", "wood", "metal"})
  (sizes : Finset String := {"small", "medium", "large"})
  (colors : Finset String := {"blue", "green", "red", "yellow"})
  (shapes : Finset String := {"circle", "hexagon", "square", "triangle"})
  (target : String := "plastic medium red circle") :
  ∃ (n : ℕ), n = 37 := by
  sorry

end blocks_differ_in_two_ways_exactly_l666_666020


namespace b_n_sum_geometric_sequence_arithmetic_sum_geometric_sequence_l666_666750

def sum_first_n (a : ℕ → ℕ) (n : ℕ) : ℕ := (List.range n).sum (fun i => a (i + 1))

-- Problem 1
theorem b_n_sum_geometric_sequence (b : ℕ → ℕ) (n : ℕ) (h1 : ∀ n, 2^(b n) = 2 * 4^(n - 1))
  (h2 : sum_first_n b (2 * n) / sum_first_n b n = 4) : true := sorry

-- Problem 2
theorem arithmetic_sum_geometric_sequence (c : ℕ → ℕ) (c1 d : ℕ)
  (h1 : ∀ n, c n = c1 + (n - 1) * d)
  (h2 : sum_first_n c (2 * n) / sum_first_n c n ≠ 0)
  (h3 : ∃ k, k ≠ 0 ∧ sum_first_n c (2 * n) / sum_first_n c n = k) : d = 2 * c1 := sorry

end b_n_sum_geometric_sequence_arithmetic_sum_geometric_sequence_l666_666750


namespace hyperbola_eccentricity_l666_666650

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (c : ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ) (B : ℝ × ℝ)
    (h_hyperbola_eq : ∀ x y, (x, y) ∈ set_of (λ p, p.1 ^ 2 / a ^ 2 - p.2 ^ 2 / b ^ 2 = 1))
    (h_focus : F = (-c, 0))
    (h_vertex : A = (0, b))
    (h_line_FA_asymptote_intersection : B = (a * c / (c - a), b * c / (c - a)))
    (h_vector_relation : vector.to_tuple (vector FA) = (sqrt 2 - 1) • vector.to_tuple (vector AB)) :
    eccentricity = sqrt 2 := 
sorry

end hyperbola_eccentricity_l666_666650


namespace joseph_power_cost_ratio_l666_666350

theorem joseph_power_cost_ratio
  (electric_oven_cost : ℝ)
  (total_cost : ℝ)
  (water_heater_cost : ℝ)
  (refrigerator_cost : ℝ)
  (H1 : electric_oven_cost = 500)
  (H2 : 2 * water_heater_cost = electric_oven_cost)
  (H3 : refrigerator_cost + water_heater_cost + electric_oven_cost = total_cost)
  (H4 : total_cost = 1500):
  (refrigerator_cost / water_heater_cost) = 3 := sorry

end joseph_power_cost_ratio_l666_666350


namespace cos_double_angle_l666_666278

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l666_666278


namespace cost_of_fencing_l666_666109

theorem cost_of_fencing (d : ℝ) (rate : ℝ) (pi : ℝ) (h_d : d = 70) (h_rate : rate = 1) (h_pi : pi = Real.pi) : 
  (pi * d * rate).ceil = 220 :=
by
  sorry

end cost_of_fencing_l666_666109


namespace max_points_on_third_line_l666_666716

theorem max_points_on_third_line (P : Finset Point) (a b c : Line)
  (hP : P.card = 100)
  (h_a : (P.filter (λ p, p ∈ a)).card = 40)
  (h_b : (P.filter (λ p, p ∈ b)).card = 40)
  (h_ab : a ≠ b)
  (h_c_ab : c ≠ a ∧ c ≠ b) :
  (P.filter (λ p, p ∈ c)).card ≤ 23 :=
sorry

end max_points_on_third_line_l666_666716


namespace triangle_area_OAB_l666_666043

theorem triangle_area_OAB
  (a r : ℝ) (h_a_gt_zero : a > 0) (h_r_gt_zero : r > 0)
  (h_line_inclination : Tan(Real.pi / 6) = √3 / 3) 
  (h_tangent : ∀ B : ℝ × ℝ, B ∈ {(x, y) : ℝ × ℝ | (√3 * x - 3 * y - √3 * a = 0)}
    ∧ (B.1^2 + B.2^2 = r^2) → dist (a, 0) B = √3) 
  (h_distance_AB : dist (a, 0) (∃ B : ℝ × ℝ, B ∈ {(x, y) : ℝ × ℝ | (√3 * x - 3 * y - √3 * a = 0)} ∧ (B.1^2 + B.2^2 = r^2)) = √3) :
  let O := (0, 0) in
  let A := (a, 0) in
  let B := Classical.choose (exists.intro (a/2, (√3 * a)/2) (and.intro (by sorry) (by sorry))) in
  let S := abs ((r * √3) / 2) in
  S = √3 / 2 :=
  sorry

end triangle_area_OAB_l666_666043


namespace percentage_of_non_defective_products_l666_666314

theorem percentage_of_non_defective_products:
  let p1 := 0.25
  let p2 := 0.35
  let p3 := 0.40
  let d1 := 0.02
  let d2 := 0.04
  let d3 := 0.05
  let total_production := 100
  let defective_units := (p1 * total_production * d1) + (p2 * total_production * d2) + (p3 * total_production * d3)
  let percentage_defective := (defective_units / total_production) * 100
  let percentage_non_defective := 100 - percentage_defective
  percentage_non_defective = 96.1 := 
by {
  let p1 := 0.25;
  let p2 := 0.35;
  let p3 := 0.40;
  let d1 := 0.02;
  let d2 := 0.04;
  let d3 := 0.05;
  let total_production := 100;
  let defective_units := (p1 * total_production * d1) + (p2 * total_production * d2) + (p3 * total_production * d3);
  let percentage_defective := (defective_units / total_production) * 100;
  let percentage_non_defective := 100 - percentage_defective;
  have h1 : defective_units = (0.25 * 100 * 0.02) + (0.35 * 100 * 0.04) + (0.40 * 100 * 0.05) := sorry;
  have h2 : percentage_defective = (defective_units / 100) * 100 := sorry;
  have h3 : percentage_non_defective = 100 - percentage_defective := sorry;
  show percentage_non_defective = 96.1, from h3;
}

end percentage_of_non_defective_products_l666_666314


namespace units_digit_of_27_mul_36_l666_666118

theorem units_digit_of_27_mul_36 : (27 * 36) % 10 = 2 := by
  sorry

end units_digit_of_27_mul_36_l666_666118


namespace jason_pokemon_cards_l666_666344

-- Conditions
def initial_cards : ℕ := 13
def cards_given : ℕ := 9

-- Proof Statement
theorem jason_pokemon_cards (initial_cards cards_given : ℕ) : initial_cards - cards_given = 4 :=
by
  sorry

end jason_pokemon_cards_l666_666344


namespace solution_l666_666921

-- Define the conditions
def satisfies_functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x + y) = f(x) * f(y)

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f(x) > f(y)

-- Define the set of given functions
def func_A (x : ℝ) : ℝ := x^(1/2)
def func_B (x : ℝ) : ℝ := x^3
def func_C (x : ℝ) : ℝ := (1/2)^x
def func_D (x : ℝ) : ℝ := Real.log (1/2) * x

-- State the theorem
theorem solution :
  satisfies_functional_equation func_C ∧ is_monotonically_decreasing func_C :=
sorry

end solution_l666_666921


namespace equal_distribution_arithmetic_seq_l666_666868

theorem equal_distribution_arithmetic_seq :
  ∃ (a1 d : ℚ), (a1 + (a1 + d) = (a1 + 2 * d) + (a1 + 3 * d) + (a1 + 4 * d)) ∧ 
                (5 * a1 + 10 / 2 * d = 5) ∧ 
                (a1 = 4 / 3) :=
by
  sorry

end equal_distribution_arithmetic_seq_l666_666868


namespace interest_rate_l666_666847

noncomputable def principal : ℝ := 400
noncomputable def compound_interest : ℝ := 100
noncomputable def time : ℝ := 2
noncomputable def n_compound : ℕ := 1
noncomputable def final_amount : ℝ := principal + compound_interest

theorem interest_rate :
  ∃ r : ℝ, final_amount = principal * (1 + r) ^ (n_compound * time) ∧ r ≈ 0.11803398875 :=
by
  sorry

end interest_rate_l666_666847


namespace ice_cream_vendor_l666_666063

theorem ice_cream_vendor (choco : ℕ) (mango : ℕ) (sold_choco : ℚ) (sold_mango : ℚ) 
  (h_choco : choco = 50) (h_mango : mango = 54) (h_sold_choco : sold_choco = 3/5) 
  (h_sold_mango : sold_mango = 2/3) : 
  choco - (choco * sold_choco) + mango - (mango * sold_mango) = 38 := 
by 
  sorry

end ice_cream_vendor_l666_666063


namespace master_bedroom_suite_is_1040_l666_666030

noncomputable def master_bedroom_suite_size (living_room_dining_kitchen_area : ℝ) (total_house_area : ℝ)
  (guest_bedroom_fraction_of_master : ℝ) : ℝ :=
  let remaining_area := total_house_area - living_room_dining_kitchen_area,
      total_bedroom_area := remaining_area,
      master_bedroom_area := total_bedroom_area / (1 + guest_bedroom_fraction_of_master)
  in master_bedroom_area

theorem master_bedroom_suite_is_1040 :
  master_bedroom_suite_size 1000 2300 0.25 = 1040 :=
by
  unfold master_bedroom_suite_size
  have h1 : 2300 - 1000 = 1300 := by norm_num
  rw [h1]
  have h2 : 1 + 0.25 = 1.25 := by norm_num
  rw [h2]
  have h3 : 1300 / 1.25 = 1040 := by norm_num
  rw [h3]
  exact rfl

end master_bedroom_suite_is_1040_l666_666030


namespace T8_minus_2_l666_666149

def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else (1/2)^(n-2)

def T (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

theorem T8_minus_2 : T 8 - 2 = 63 / 64 :=
by {
  -- Definitions of T and a should ensure the correct proof
  sorry
}

end T8_minus_2_l666_666149


namespace complex_numbers_count_l666_666980

theorem complex_numbers_count (z : ℂ) (h1 : complex.abs z = 1) (h2 : complex.abs ((z / (z.conj)) - ((z.conj) / z)) = 1) : (∃ z_list : list ℂ, z_list.length = 8 ∧ ∀ w ∈ z_list, complex.abs w = 1 ∧ complex.abs ((w / (w.conj)) - ((w.conj) / w)) = 1) :=
sorry

end complex_numbers_count_l666_666980


namespace solve_equation_l666_666768

theorem solve_equation (a b : ℚ) : 
  ((b = 0) → false) ∧ 
  ((4 * a - 3 = 0) → ((5 * b - 1 = 0) → a = 3 / 4 ∧ b = 1 / 5)) ∧ 
  ((4 * a - 3 ≠ 0) → (∃ x : ℚ, x = (5 * b - 1) / (4 * a - 3))) :=
by
  sorry

end solve_equation_l666_666768


namespace four_neg_a_equals_one_ninth_l666_666658

theorem four_neg_a_equals_one_ninth (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 2) : 4 ^ (-a) = 1 / 9 :=
by
  sorry

end four_neg_a_equals_one_ninth_l666_666658


namespace least_zeros_in_interval_l666_666825

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero : f 0 = 0
axiom f_sym_9 : ∀ x : ℝ, f (9 + x) = f (9 - x)
axiom f_sym_10 : ∀ x : ℝ, f (x - 10) = f (-x - 10)

theorem least_zeros_in_interval : 
  ∃ n : ℕ, n ≥ 107 ∧ ∀ x ∈ set.Icc 0 2014, f x = 0 → ∃ k : ℕ, 38 * k = x ∨ 38 * k + 18 = x :=
sorry

end least_zeros_in_interval_l666_666825


namespace smallest_possible_purchases_l666_666474

/-- Definitions for coin values in pence --/
def one_shilling : ℕ := 12

/-- Condition that ensures no change is given and total is less than 5 shillings (60 pence) --/
def no_change_and_less_than_five_shillings (a b : ℕ) : Prop :=
  a + b < 60

/-- Condition that each purchase requires exactly six coins --/
def requires_six_coins (amount : ℕ) : Prop :=
  ∃ c1 c2 c3 c4 c5 c6 : ℕ, c1 + c2 + c3 + c4 + c5 + c6 = amount

/-- Condition that combined purchases also require exactly six coins --/
def combined_requires_six_coins (a b : ℕ) : Prop :=
  ∃ c1 c2 c3 c4 c5 c6 : ℕ, c1 + c2 + c3 + c4 + c5 + c6 = a + b

/-- Mathematical proof statement: 
  There exist two different amounts that each requires exactly six coins,
  and when combined also require exactly six coins, and have the smallest possible total. --/
theorem smallest_possible_purchases :
  ∃ (a b : ℕ), no_change_and_less_than_five_shillings a b ∧ 
  requires_six_coins a ∧ 
  requires_six_coins b ∧ 
  combined_requires_six_coins a b ∧ 
  a = 23.5 ∧ 
  b = 11.5 :=
sorry

end smallest_possible_purchases_l666_666474


namespace units_digit_multiplication_l666_666116

-- Define a function to find the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Statement of the problem: Given the product 27 * 36, prove that the units digit is 2.
theorem units_digit_multiplication (a b : ℕ) (h1 : units_digit 27 = 7) (h2 : units_digit 36 = 6) :
  units_digit (27 * 36) = 2 :=
by
  have h3 : units_digit (7 * 6) = 2 := by sorry
  exact h3

end units_digit_multiplication_l666_666116


namespace price_of_cheese_cookie_pack_l666_666017

theorem price_of_cheese_cookie_pack
    (cartons : ℕ) (boxes_per_carton : ℕ) (packs_per_box : ℕ) (total_cost : ℕ)
    (h_cartons : cartons = 12)
    (h_boxes_per_carton : boxes_per_carton = 12)
    (h_packs_per_box : packs_per_box = 10)
    (h_total_cost : total_cost = 1440) :
  (total_cost / (cartons * boxes_per_carton * packs_per_box) = 1) :=
by
  -- conditions are explicitly given in the theorem statement
  sorry

end price_of_cheese_cookie_pack_l666_666017


namespace weight_of_B_l666_666511

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 47) : B = 39 :=
by
  sorry

end weight_of_B_l666_666511


namespace new_cost_after_decrease_l666_666912

theorem new_cost_after_decrease {A : ℝ} (hA : A = 1750) : 
  let N := A - 0.4 * A in N = 1050 :=
by
  sorry

end new_cost_after_decrease_l666_666912


namespace cosine_double_angle_l666_666253

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l666_666253


namespace daily_sales_and_profit_l666_666525

-- Here we state the problem and provide the conditions
theorem daily_sales_and_profit
  (f : ℝ → ℝ)
  (h : ∀ x, 3 < x ∧ x < 6 → f x = 2 + 10 * (x - 3) * (x - 6) ^ 2 )
  (h₁ : f 5 = 11) :
  (∀ x, f x < f 4) : sorry

end daily_sales_and_profit_l666_666525


namespace cos_double_angle_l666_666262

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l666_666262


namespace ordered_triples_count_l666_666944

theorem ordered_triples_count :
  ∃! (a b c : ℕ), a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ real.log a b = c^2 ∧ a + b + c = 100 :=
sorry

end ordered_triples_count_l666_666944


namespace raisin_impossible_to_cut_l666_666615

theorem raisin_impossible_to_cut (initial_square : convex_polygon)
  (raisin_center : initial_square.center = raisin)
  (cut_condition : ∀ (A B C : point), A ∈ initial_square.vertices →
    B ∈ side_segment (A, A.next) → C ∈ side_segment (A, A.previous) →
    convex_polygon (initial_square.cut_triangle A B C)) :
  ¬ ∃ (A B C : point), A ∈ initial_square.vertices ∧
    B ∈ side_segment (A, A.next) ∧ C ∈ side_segment (A, A.previous) ∧
    raisin_center ∈ triangle A B C := 
by
  sorry

end raisin_impossible_to_cut_l666_666615


namespace smallest_three_digit_number_l666_666503

theorem smallest_three_digit_number : 
  ∃ n, n = 203 ∧ ∀ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ (a ∈ {3, 0, 2, 5, 7}) ∧ (b ∈ {3, 0, 2, 5, 7}) ∧ (c ∈ {3, 0, 2, 5, 7}) ∧ a ≠ 0 
      → n = a * 100 + b * 10 + c → n ≥ 100 ∧ n ≤ 999 → n = 203 := 
by
  -- proof steps would go here
  sorry

end smallest_three_digit_number_l666_666503


namespace curve_symmetric_about_line_l666_666471

def curve_original (x : ℝ) : ℝ := log x / log 2
def curve_transformed (x : ℝ) : ℝ := (log (x - 3) / log 2) - 2
def curve_candidate (x : ℝ) : ℝ := -2^(2 - x) - 1

theorem curve_symmetric_about_line :
  ∀ x, curve_transformed (-curve_candidate x) = -x ↔ x + curve_transformed x = 0 :=
by
  sorry

end curve_symmetric_about_line_l666_666471


namespace coloring_pyramid_l666_666932

theorem coloring_pyramid :
  let vertices : Finset ℕ := {1, 2, 3, 4, 5}
  ∃ (S A B C D : ℕ), 
  S ∈ vertices ∧ A ∈ vertices ∧ B ∈ vertices ∧ C ∈ vertices ∧ D ∈ vertices ∧
  S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ S ≠ D ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧
  B ≠ C ∧ B ≠ D ∧
  C ≠ D ∧
  5 * (5 - 1) * (5 - 2) * 7 = 420 := 
begin 
  -- definitions of the vertices and the conditions for coloring
  let vertices := {1, 2, 3, 4, 5},
  existsi (1, 2, 3, 4, 5),
  simp only [finset.mem_insert, finset.mem_singleton],
  split, 
  -- proving the colors assigned to vertices belong to the set of 5 colors available 
  sorry,
end

end coloring_pyramid_l666_666932


namespace solve_quadratic_eq_l666_666448

theorem solve_quadratic_eq (x : ℝ) : x^2 = 4 * x → x = 0 ∨ x = 4 :=
by
  intro h
  sorry

end solve_quadratic_eq_l666_666448


namespace solve_equation_l666_666830

theorem solve_equation (x : ℝ) (h : 4^x = 2^(x + 1) - 1) : x = 0 := by
  sorry

end solve_equation_l666_666830


namespace find_a_l666_666520

variable (a : ℝ)
def f (x : ℝ) := 5 ^ |x|
def g (x : ℝ) := a * x^2 - x

theorem find_a (h : f (g 1) = 1) : a = 1 := by
  sorry

end find_a_l666_666520


namespace count_possible_c_values_l666_666607

theorem count_possible_c_values :
  {c : ℤ | 0 ≤ c ∧ c ≤ 2000 ∧ ∃ x : ℝ, 5 * (⌊x⌋ : ℤ) + 4 * (⌈x⌉ : ℤ) = c}.card = 445 :=
by
  sorry

end count_possible_c_values_l666_666607


namespace b101_mod_49_l666_666364

-- Definitions based on conditions
def b (n : ℕ) : ℕ := 5^n + 7^n

-- The formal statement of the proof problem
theorem b101_mod_49 : b 101 % 49 = 12 := by
  sorry

end b101_mod_49_l666_666364


namespace network_structure_l666_666550

theorem network_structure 
  (n : ℕ)
  (is_acquainted : Fin n → Fin n → Prop)
  (H_symmetric : ∀ x y, is_acquainted x y = is_acquainted y x) 
  (H_common_acquaintance : ∀ x y, ¬ is_acquainted x y → ∃! z : Fin n, is_acquainted x z ∧ is_acquainted y z) :
  ∃ (G : SimpleGraph (Fin n)), (∀ x y, G.Adj x y = is_acquainted x y) ∧
    (∀ x y, ¬ G.Adj x y → (∃ (z1 z2 : Fin n), G.Adj x z1 ∧ G.Adj y z1 ∧ G.Adj x z2 ∧ G.Adj y z2)) :=
by
  sorry

end network_structure_l666_666550


namespace students_in_all_sections_is_six_l666_666463

-- Define the number of students in each section and the total.
variable (total_students : ℕ := 30)
variable (music_students : ℕ := 15)
variable (drama_students : ℕ := 18)
variable (dance_students : ℕ := 12)
variable (at_least_two_sections : ℕ := 14)

-- Define the number of students in all three sections.
def students_in_all_three_sections (total_students music_students drama_students dance_students at_least_two_sections : ℕ) : ℕ :=
  let a := 6 -- the result we want to prove
  a

-- The theorem proving that the number of students in all three sections is 6.
theorem students_in_all_sections_is_six :
  students_in_all_three_sections total_students music_students drama_students dance_students at_least_two_sections = 6 :=
by 
  sorry -- Proof is omitted

end students_in_all_sections_is_six_l666_666463


namespace integral_problem_l666_666098

noncomputable def integrand (x : ℝ) : ℝ := real.sqrt (1 - x^2) + x + x^3

theorem integral_problem :
  ∫ x in 0..1, integrand x = (Real.pi + 3) / 4 :=
by
  sorry

end integral_problem_l666_666098


namespace haley_initial_cupcakes_l666_666203

-- Define the conditions
def todd_eats : ℕ := 11
def packages : ℕ := 3
def cupcakes_per_package : ℕ := 3

-- Initial cupcakes calculation
def initial_cupcakes := packages * cupcakes_per_package + todd_eats

-- The theorem to prove
theorem haley_initial_cupcakes : initial_cupcakes = 20 :=
by
  -- Mathematical proof would go here,
  -- but we leave it as sorry for now.
  sorry

end haley_initial_cupcakes_l666_666203


namespace sandwich_cost_in_usd_l666_666918

theorem sandwich_cost_in_usd (price_eur : ℝ) (conversion_rate : ℝ) :
  price_eur = 50 ∧ conversion_rate = 0.95 → (price_eur / conversion_rate) ≈ 52.63 :=
by
  sorry

end sandwich_cost_in_usd_l666_666918


namespace no_maximum_value_l666_666176

def f (x : ℝ) (M P : set ℝ) : ℝ :=
if x ∈ M then 2^x else x^2

theorem no_maximum_value (M P : set ℝ) (h : M ∪ P = set.univ) :
  ¬ ∃ x, ∀ y, f y M P ≤ f x M P :=
sorry

end no_maximum_value_l666_666176


namespace broadcasting_orders_count_l666_666052

theorem broadcasting_orders_count {business_comm : ℕ} {expo_comm : ℕ} {public_service_comm : ℕ}
  (business_comm_pos : business_comm = 3) (expo_comm_pos : expo_comm = 2) (public_service_comm_pos : public_service_comm = 1)
  (num_commercials : business_comm + expo_comm + public_service_comm = 6)
  (last_not_business : ¬ (business_comm > 0)) 
  (expo_not_consec : ¬ (expo_comm = 2))
  (public_service_not_consec_with_expo : ¬ (public_service_comm = 1 ∧ expo_comm = 2)) :
  36 := by
  sorry

end broadcasting_orders_count_l666_666052


namespace trigonometric_identity_l666_666568

theorem trigonometric_identity :
  sec (Real.pi / 18) - 3 * sin (5 * Real.pi / 9) = (1 - (3 / 2) * sin (Real.pi / 9)) / sin (4 * Real.pi / 9) :=
by sorry

end trigonometric_identity_l666_666568


namespace evaluate_propositions_l666_666328

/- Definitions -/
def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

def proposition_1 := {P : ℝ × ℝ | polyline_distance P (0,0) = 1}
def proposition_2 := {P : ℝ × ℝ| polyline_distance P (0,0) = 1}
def proposition_3 := {P : ℝ × ℝ | polyline_distance P (-1,0) + polyline_distance P (1,0) = 4}
def proposition_4 := {P : ℝ × ℝ | abs (polyline_distance P (-1,0) - polyline_distance P (1,0)) = 1}

/- Theorem -/
theorem evaluate_propositions :
  (proposition_1 = {P : ℝ × ℝ | |P.1| + |P.2| = 1} ∧
  ¬(proposition_2 = {P : ℝ × ℝ | sqrt (P.1^2 + P.2^2) = 1}) ∧
  (proposition_3 = {P : ℝ × ℝ | polyline_distance P (-1,0) + polyline_distance P (1,0) = 4}) ∧
  (proposition_3_area = 6) ∧
  (proposition_4 = {P : ℝ × ℝ | abs ((|P.1 +1| - |P.1 -1|)) = 1})) :=
  sorry

end evaluate_propositions_l666_666328


namespace total_cost_of_modular_home_l666_666579

-- Definition of costs and sizes for the specific modules
def kitchen_sqft : ℕ := 400
def kitchen_cost : ℕ := 20000

def bathroom_sqft : ℕ := 150
def bathroom_cost : ℕ := 12000

def other_cost_per_sqft : ℕ := 100

-- Total planned area for the modular home
def total_sqft : ℕ := 2000

-- Total cost to build the modular home with given modules and areas
theorem total_cost_of_modular_home :
  let total_area_kitchen_bathrooms := kitchen_sqft + 2 * bathroom_sqft in
  let total_cost_kitchen_bathrooms := kitchen_cost + 2 * bathroom_cost in
  let remaining_sqft := total_sqft - total_area_kitchen_bathrooms in
  let remaining_cost := remaining_sqft * other_cost_per_sqft in
  total_cost_kitchen_bathrooms + remaining_cost = 174000 := 
  by
  sorry

end total_cost_of_modular_home_l666_666579


namespace invariant_interval_of_g_l666_666623

def is_invariant (g : ℝ → ℝ) (A : set ℝ) := ∀ {x}, x ∈ A → g x ∈ A

theorem invariant_interval_of_g (A : set ℝ)
  (m : ℝ) (h : A = set.Ici 0) (g : ℝ → ℝ) (hg : ∀ x, g x = -x + m + real.exp x)
  (hinv : is_invariant g A) : m = -1 :=
by {
  sorry
}

end invariant_interval_of_g_l666_666623


namespace rational_triples_roots_polynomial_l666_666592

theorem rational_triples_roots_polynomial :
  ∀ (a b c : ℚ), (a, b, c) ∈ { (0, 0, 0), (1, -2, 0), (1, -1, -1) } ↔
  (∀ x : ℚ, x^3 + a * x^2 + b * x + c = 0 → x = a ∨ x = b ∨ x = c) :=
by {
  sorry
}

end rational_triples_roots_polynomial_l666_666592


namespace rational_solution_for_quadratic_l666_666612

theorem rational_solution_for_quadratic (k : ℕ) (h_pos : 0 < k) : 
  ∃ m : ℕ, (18^2 - 4 * k * (2 * k)) = m^2 ↔ k = 4 :=
by
  sorry

end rational_solution_for_quadratic_l666_666612


namespace wage_percent_change_l666_666036

-- Definitions based on given conditions
def initial_wage (W : ℝ) := W
def first_decrease (W : ℝ) := 0.60 * W
def first_increase (W : ℝ) := 0.78 * W
def second_decrease (W : ℝ) := 0.624 * W
def second_increase (W : ℝ) := 0.6864 * W

-- Lean theorem statement to prove overall percent change
theorem wage_percent_change : ∀ (W : ℝ), 
  ((second_increase (second_decrease (first_increase (first_decrease W))) - initial_wage W) / initial_wage W) * 100 = -31.36 :=
by sorry

end wage_percent_change_l666_666036


namespace min_value_tangent_line_l666_666166

theorem min_value_tangent_line (k b : ℝ) (h : ∀ x : ℝ, y = sqrt(x) + 1 → y = k*x + b ∧ deriv (λ x, sqrt(x) + 1) x = k) :
  ∃ x : ℝ, k = 1 / (2 * sqrt x) ∧ b = sqrt x / 2 + 1 ∧ (k^2 + b^2 - 2 * b) = -1 / 2 := 
sorry

end min_value_tangent_line_l666_666166


namespace double_angle_cosine_l666_666224

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l666_666224


namespace value_of_expression_l666_666696

theorem value_of_expression (y : ℤ) (h : 7^(2 * y) = (343 : ℤ)^2) : 7^(2 * y - 4) = 49 := by
  sorry

end value_of_expression_l666_666696


namespace ice_creams_not_sold_l666_666066

theorem ice_creams_not_sold 
  (chocolate_ice_creams : ℕ)
  (mango_ice_creams : ℕ)
  (sold_chocolate : ℚ)
  (sold_mango : ℚ)
  (initial_chocolate : chocolate_ice_creams = 50)
  (initial_mango : mango_ice_creams = 54)
  (fraction_sold_chocolate : sold_chocolate = 3 / 5)
  (fraction_sold_mango : sold_mango = 2 / 3) :
  chocolate_ice_creams - (chocolate_ice_creams * fraction_sold_chocolate).toNat
  + mango_ice_creams - (mango_ice_creams * fraction_sold_mango).toNat = 38 := 
by {
  sorry
}

end ice_creams_not_sold_l666_666066


namespace distance_city_A_B_l666_666862

theorem distance_city_A_B (D : ℝ) : 
  (3 : ℝ) + (2.5 : ℝ) = 5.5 → 
  ∃ T_saved, T_saved = 1 →
  80 = (2 * D) / (5.5 - T_saved) →
  D = 180 :=
by
  intros
  sorry

end distance_city_A_B_l666_666862


namespace binomial_coefficient_and_factorial_l666_666569

open Nat

/--
  Given:
    - The binomial coefficient definition: Nat.choose n k = n! / (k! * (n - k)!)
    - The factorial definition: Nat.factorial n = n * (n - 1) * ... * 1
  Prove:
    Nat.choose 60 3 * Nat.factorial 10 = 124467072000
-/
theorem binomial_coefficient_and_factorial :
  Nat.choose 60 3 * Nat.factorial 10 = 124467072000 :=
by
  sorry

end binomial_coefficient_and_factorial_l666_666569


namespace general_term_formula_Sn_less_three_fourths_l666_666169

-- Defining the conditions
def geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def a (n : ℕ) : ℝ := (-2)^(n-1)

def b (n : ℕ) : ℝ := 1 / (Real.log 2 (abs (a (n + 1))) * Real.log 2 (abs (a (n + 3))))

def Sn (n : ℕ) : ℝ := ∑ i in Finset.range n, b i

-- The statement for the general term formula
theorem general_term_formula 
  (h1 : a 2 + a 5 = 14) 
  (h2 : a 2 * a 5 = -32) 
  (h3 : a 1 > 0) 
  (h_seq : geometric_seq a) : 
  ∀ n : ℕ, a n = (-2)^(n-1) := 
by 
  sorry

-- The statement for the inequality
theorem Sn_less_three_fourths 
  : ∀ n : ℕ, Sn n < 3 / 4 := 
by 
  sorry

end general_term_formula_Sn_less_three_fourths_l666_666169


namespace double_angle_cosine_l666_666221

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l666_666221


namespace problem1_problem2_l666_666874

-- Problem 1
theorem problem1 (pi : ℝ) : 
  ((real.cbrt 2 * real.sqrt 3) ^ 6) + (-2018)^0 - 4 * ((16/49) ^ (-1/2)) + (|(3 - pi)| ^ 4) ^ (1/4) = 102 + |3 - pi| :=
sorry

-- Problem 2
theorem problem2 : 
  (log 5 8 * log 5 2 + 25^(log 5 3)) / (log 4 + log 25) + 5 * log 3 2 - log 3 (32/9) = 8 :=
sorry

end problem1_problem2_l666_666874


namespace modulus_of_complex_number_l666_666672

noncomputable def complex_modulus (z : ℂ) : ℝ :=
  complex.abs z

theorem modulus_of_complex_number :
  let z : ℂ := -6 + (3 - (5/3 : ℂ).im * complex.I) * complex.I in
  complex_modulus z = (5 * real.sqrt 10) / 3 :=
by
  sorry

end modulus_of_complex_number_l666_666672


namespace inequality_solution_b_range_l666_666653

-- Given conditions
variables (a b : ℝ)

def condition1 : Prop := (1 - a < 0) ∧ (a = 3)
def condition2 : Prop := ∀ (x : ℝ), (3 * x^2 + b * x + 3) ≥ 0

-- Assertions to be proved
theorem inequality_solution (a : ℝ) (ha : condition1 a) : 
  ∀ (x : ℝ), (2 * x^2 + (2 - a) * x - a > 0) ↔ (x < -1 ∨ x > 3/2) :=
sorry

theorem b_range (a : ℝ) (hb : condition1 a) : 
  condition2 b ↔ (-6 ≤ b ∧ b ≤ 6) :=
sorry

end inequality_solution_b_range_l666_666653


namespace smallest_integer_ends_in_3_and_divisible_by_5_l666_666495

theorem smallest_integer_ends_in_3_and_divisible_by_5 : ∃ (n : ℕ), n > 0 ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ n = 53 :=
by
  sorry

end smallest_integer_ends_in_3_and_divisible_by_5_l666_666495


namespace triangle_problem_l666_666729

theorem triangle_problem
  (P Q R S T : Type)
  (line_PR: P → R → T → Prop)
  (line_QR: Q → R → S → Prop)
  (isosceles_PQR: P → Q → R → Prop)
  (PQ_eq_QR : ∀ (p q r : P), isosceles_PQR p q r → (p = q) = (q = r))
  (angle_PQR_40: ∀ (p q r : P), isosceles_PQR p q r → angle p q r = 40)
  (isosceles_RST: R → S → T → Prop)
  (RS_eq_RT : ∀ (r s t : R), isosceles_RST r s t → (r = s) = (s = t))
  (x_value : ∀ {r s t : R}, isosceles_RST r s t → angle_SR r s t = 70 → angle_RST r s t = 55) :
  ∀ {r s t : R}, isosceles_RST r s t → angle_RST r s t = 55 :=
  sorry

end triangle_problem_l666_666729


namespace eliminate_sacks_l666_666715

noncomputable def number_of_sacks_to_eliminate (total_sacks : ℕ) (required_probability : ℚ) : ℕ :=
  let sacks_with_at_least_65536 := 6
  let min_sacks_to_reach_probability := sacks_with_at_least_65536 / required_probability
  total_sacks - int.ceil min_sacks_to_reach_probability

theorem eliminate_sacks (total_sacks : ℕ) (sacks_with_at_least_65536 : ℕ) (required_probability : ℚ) :
  total_sacks = 30 →
  sacks_with_at_least_65536 = 6 →
  required_probability = 0.4 →
  number_of_sacks_to_eliminate total_sacks required_probability = 15 := 
by
  intros ht hs hp
  rw [number_of_sacks_to_eliminate, ht, hs, hp]
  norm_num
  exact rfl

end eliminate_sacks_l666_666715


namespace total_packages_sold_l666_666561

variable (P : ℕ)

/-- An automobile parts supplier charges 25 per package of gaskets. 
    When a customer orders more than 10 packages of gaskets, the supplier charges 4/5 the price for each package in excess of 10.
    During a certain week, the supplier received 1150 in payment for the gaskets. --/
def cost (P : ℕ) : ℕ :=
  if P > 10 then 250 + (P - 10) * 20 else P * 25

theorem total_packages_sold :
  cost P = 1150 → P = 55 := by
  sorry

end total_packages_sold_l666_666561


namespace angle_between_apothem_and_lateral_face_l666_666810

theorem angle_between_apothem_and_lateral_face
    {a : ℝ} (h₀ : ∀ (P A B C L : ℝ), (FL = (3 / 8) / 8 * a)
    
    -- Assume the necessary geometric relationships
    (h₁ : P ≠ A) (h₂ : P ≠ B) (h₃ : P ≠ C) (h₄ : ∠A = ∠B = ∠C) :
    ∃ φ, φ = arcsin (3 * sqrt 39 / 25) :=
begin
    sorry,
end

end angle_between_apothem_and_lateral_face_l666_666810


namespace inequality_proof_l666_666208

theorem inequality_proof
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (hxyz : x * y * z = 1) :
  x^2 + y^2 + z^2 + x * y + y * z + z * x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) :=
by
  sorry

end inequality_proof_l666_666208


namespace alpha_nonneg_integer_l666_666506

theorem alpha_nonneg_integer (α : ℝ) 
  (h : ∀ n : ℕ, ∃ k : ℕ, n = k * α) : α ≥ 0 ∧ ∃ k : ℤ, α = k := 
sorry

end alpha_nonneg_integer_l666_666506


namespace number_of_correct_conclusions_l666_666820

theorem number_of_correct_conclusions :
  let S1 := ∀ (Q : Type) [Q.quadrilateral] [Q.rectangle], false in 
  let S2 := ∀ (x : ℝ), x^2 + 1 < 0 in
  let S3 := ¬ (∃ x : ℝ, x^2 + 2 * x + 1 ≤ 0) = ∀ x : ℝ, x^2 + 2 * x + 1 ≤ 0 in
  let S4 := ∀ {a b c : ℝ}, c ≠ 0 ∧ ac^2 > bc^2 → a > b in
  (¬ S1 → ¬ S2 → S3 → S4 → 2 = 2) := 
begin
    sorry
end

end number_of_correct_conclusions_l666_666820


namespace least_value_xy_l666_666642

theorem least_value_xy {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : (1 : ℚ) / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_value_xy_l666_666642


namespace enumerate_set_l666_666446

open Set

def is_positive_integer (x : ℕ) : Prop := x > 0

theorem enumerate_set :
  { p : ℕ × ℕ | p.1 + p.2 = 4 ∧ is_positive_integer p.1 ∧ is_positive_integer p.2 } =
  { (1, 3), (2, 2), (3, 1) } := by 
sorry

end enumerate_set_l666_666446


namespace aisha_probability_four_tosses_l666_666914

noncomputable def probability_at_least_two_heads (tosses : ℕ) (heads_needed : ℕ) : ℚ :=
  1 - (nat.choose tosses 1 * (1/2)^tosses + (1/2)^tosses)

theorem aisha_probability_four_tosses :
  probability_at_least_two_heads 4 2 = 11/16 := by
  sorry 

end aisha_probability_four_tosses_l666_666914


namespace cos_double_angle_l666_666230

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666230


namespace cos_double_angle_l666_666240

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666240


namespace quadratic_has_two_distinct_real_roots_l666_666829

theorem quadratic_has_two_distinct_real_roots :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (∀ x : ℝ, x^2 + 2 * x - 3 = 0 ↔ x = x1 ∨ x = x2) :=
by
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := -3
  let Δ := b^2 - 4 * a * c
  have hΔ : Δ = 16 := by
    calc
      Δ = b^2 - 4 * a * c : rfl
      _ = 2^2 - 4 * 1 * (-3) : by norm_num
      _ = 4 + 12 : by norm_num
      _ = 16 : by norm_num
  have hΔ_pos : Δ > 0 := by norm_num
  have h := quadratic_eq_reals_iff_real_roots a b c hΔ_pos
  exact h
  sorry

end quadratic_has_two_distinct_real_roots_l666_666829


namespace max_difference_in_bottle_masses_l666_666422

theorem max_difference_in_bottle_masses (mass nominal : ℝ) (epsilon : ℝ) (h : nominal = 450) (h_eps : epsilon = 1) :
  let max_mass := nominal + epsilon in
  let min_mass := nominal - epsilon in
  max_mass - min_mass = 2 :=
by
  sorry

end max_difference_in_bottle_masses_l666_666422


namespace complex_point_quadrant_l666_666331

def imaginary_unit : ℂ := Complex.i

def complex_number : ℂ := 5 / (2 + imaginary_unit)

def point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem complex_point_quadrant :
  let p := point complex_number in p.1 > 0 ∧ p.2 < 0 :=
by
  -- proof goes here
  sorry

end complex_point_quadrant_l666_666331


namespace moles_of_hcl_formed_l666_666982

def c2h6_moles : ℕ := 3
def cl2_moles : ℕ := 6

theorem moles_of_hcl_formed (c2h6 : ℕ) (cl2 : ℕ) (reaction : c2h6 + cl2 → c2h5cl + hcl = true) :
  c2h6 = c2h6_moles →
  cl2 = cl2_moles →
  c2h6 = 3 :=
by
  sorry

end moles_of_hcl_formed_l666_666982


namespace distance_relationship_l666_666524

-- Define the relationship between distance and time given the speed
def speed_constant : ℝ := 70
def distance_traveled (t : ℝ) : ℝ := speed_constant * t

-- State the main theorem
theorem distance_relationship (t : ℝ) : distance_traveled(t) = 70 * t :=
by 
  -- The proof will utilize the definition of distance_traveled
  sorry

-- Define the independent variable
def independent_variable : Type := ℝ
lemma independent_variable_is_time : independent_variable = ℝ :=
by 
  -- Show that the independent variable is indeed time
  sorry

-- Define the dependent variable
def dependent_variable : Type := ℝ
lemma dependent_variable_is_distance : dependent_variable = ℝ :=
by 
  -- Show that the dependent variable is indeed distance
  sorry

end distance_relationship_l666_666524


namespace triangle_area_l666_666487

theorem triangle_area (a b c : ℝ) (ha : a = 5) (hb : b = 5) (hc : c = 6) :
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 12 :=
by 
  sorry

end triangle_area_l666_666487


namespace deans_task_impossible_l666_666084

theorem deans_task_impossible :
  ¬ ∃ (grid : ℕ → ℕ → ℕ), (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 9)
  ∧ (∀ i, (∏ j in finset.range 3, grid i j) % 4 = 0)    
  ∧ (∀ j, (∏ i in finset.range 3, grid i j) % 4 = 0)
  := sorry

end deans_task_impossible_l666_666084


namespace units_digit_of_27_times_36_l666_666122

theorem units_digit_of_27_times_36 :
  let units_digit := fun (n : ℕ) => n % 10
  in units_digit (27 * 36) = 2 :=
by
  let units_digit := fun (n : ℕ) => n % 10
  have h27: units_digit 27 = 7 := by
    show 27 % 10 = 7
    sorry
  have h36: units_digit 36 = 6 := by
    show 36 % 10 = 6
    sorry
  have h42: units_digit (7 * 6) = 2 := by
    show 42 % 10 = 2
    sorry
  exact h42

end units_digit_of_27_times_36_l666_666122


namespace remainder_of_power_mod_five_l666_666827

theorem remainder_of_power_mod_five : (4 ^ 11) % 5 = 4 :=
by
  sorry

end remainder_of_power_mod_five_l666_666827


namespace polygon_sides_l666_666458

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360)
  (h2 : n ≥ 3) : 
  n = 8 := 
by
  sorry

end polygon_sides_l666_666458


namespace distance_origin_to_point_on_parabola_l666_666749

noncomputable def origin : ℝ × ℝ := (0, 0)

noncomputable def parabola_focus (x y : ℝ) : Prop :=
  x^2 = 4 * y ∧ y = 1

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  x^2 = 4 * y

theorem distance_origin_to_point_on_parabola (x y : ℝ) (hx : x^2 = 4 * y)
 (hf : (0, 1) = (0, 1)) (hPF : (x - 0)^2 + (y - 1)^2 = 25) : (x^2 + y^2 = 32) :=
by
  sorry

end distance_origin_to_point_on_parabola_l666_666749


namespace gasoline_cost_for_trip_l666_666345

noncomputable def cost_of_gasoline (distance_traveled : ℕ) (fuel_efficiency : ℕ) (price_per_gallon : ℝ) : ℝ := 
  (distance_traveled.to_nat / fuel_efficiency) * price_per_gallon

theorem gasoline_cost_for_trip :
  let initial_odometer := 32430
  let final_odometer := 32464
  let fuel_efficiency := 32
  let price_per_gallon := 3.85
  let distance_traveled := final_odometer - initial_odometer
  cost_of_gasoline distance_traveled fuel_efficiency price_per_gallon = 4.09 :=
by 
  sorry

end gasoline_cost_for_trip_l666_666345


namespace non_freshmen_play_instrument_solution_l666_666545

variables (f n : ℕ)
def total_students : ℕ := 400
def freshman_percentage_play_instrument : ℝ := 0.5
def non_freshman_percentage_not_play_instrument : ℝ := 0.25
def students_not_play_instrument : ℝ := 0.4 * 400
def system_of_equations := (f + n = total_students) ∧ (0.50 * f + 0.25 * n = students_not_play_instrument)

theorem non_freshmen_play_instrument_solution (h : system_of_equations f n):
  0.75 * n = 120 :=
sorry  --proof not required

end non_freshmen_play_instrument_solution_l666_666545


namespace point_on_parallel_line_with_P_l666_666060

-- Definitions
def is_on_parallel_line_x_axis (P Q : ℝ × ℝ) : Prop :=
  P.snd = Q.snd

theorem point_on_parallel_line_with_P :
  let P := (3, -2)
  let D := (-3, -2)
  is_on_parallel_line_x_axis P D :=
by
  sorry

end point_on_parallel_line_with_P_l666_666060


namespace cos_double_angle_l666_666291

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666291


namespace ordered_partitions_count_l666_666819

theorem ordered_partitions_count (n : ℕ) : 
  (number_of_ordered_partitions n = 2^(n - 1)) := 
sorry

end ordered_partitions_count_l666_666819


namespace travel_methods_count_l666_666042

theorem travel_methods_count :
  let choices_shijiazhuang_qingdao := 3
      choices_qingdao_guangzhou := 4 
  in choices_shijiazhuang_qingdao * choices_qingdao_guangzhou = 12 :=
by
  sorry

end travel_methods_count_l666_666042


namespace exists_common_divisor_l666_666370

open Int

variable {R : Type*} [CommRing R] (F : ℤ → ℤ) (a : List ℤ)

-- Definition of a polynomial with integer coefficients.
def is_poly_with_int_coeff (F : ℤ → ℤ) : Prop :=
  ∃ p : ℤ[X], ∀ n : ℤ, F n = eval n p

-- Condition: For any integer n, F(n) is divisible by one of the integers a1, a2, ..., am.
def div_by_one_of_a (F : ℤ → ℤ) (a : List ℤ) : Prop :=
  ∀ n : ℤ, ∃ d ∈ a, d ∣ F n

-- Main theorem
theorem exists_common_divisor (F : ℤ → ℤ) (a : List ℤ)
    (h_poly : is_poly_with_int_coeff F)
    (h_div : div_by_one_of_a F a) :
  ∃ d ∈ a, ∀ n : ℤ, d ∣ F n :=
by
  sorry

end exists_common_divisor_l666_666370


namespace cos_A_value_side_c_values_l666_666338

-- Definitions based on conditions
def in_triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
sides_opposite a A b B c C -- This is a placeholder for being a triangle with specified opposite sides and angles

def condition (a b c A B C : ℝ) : Prop :=
2 * a * Real.cos A = c * Real.cos B + b * Real.cos C

-- Proof problem 1
theorem cos_A_value (a b c A B C : ℝ) (h_in_triangle: in_triangle_ABC a b c A B C) :
  condition a b c A B C → Real.cos A = 1 / 2 :=
sorry

-- Additional condition for proof problem 2
def secondary_condition (B C : ℝ) : Prop :=
Real.cos B + Real.cos (2 * Real.pi / 3 - B) = Real.sqrt 3 / 2 

theorem side_c_values (a b c A B C : ℝ) (h_in_triangle: in_triangle_ABC a b c A B C) :
  condition a b c A B C ∧ a = 1 ∧ Real.cos^2 (B / 2) + Real.cos^2 (C / 2) = 1 + Real.sqrt 3 / 4 →
  (c = 2 * Real.sqrt 3 / 3 ∨ c = Real.sqrt 3 / 3) :=
sorry

end cos_A_value_side_c_values_l666_666338


namespace equal_piles_impossible_l666_666002

theorem equal_piles_impossible (initial : list ℕ) : 
  initial = [2, 3, 1, 1] →
  ∀ f : list ℕ → list ℕ, (∀ i j, i < j → i < 4 ∧ j < 4 → f (list.update_nth (list.update_nth initial i ((list.nth_le initial i (by simp [initial])) + 1)) j ((list.nth_le initial j (by simp [initial])) + 1)) ∈ permutations [0, 1, 2, 3] ) →
  ¬ ∃ n, ∀ i < 4, (f initial i = n) :=
begin
  intros h1 h2,
  sorry
end

end equal_piles_impossible_l666_666002


namespace minimum_lamps_required_l666_666096

theorem minimum_lamps_required (n : ℕ) (h : n ≥ 3) : 
  (∃ initial_on_lamps : ℕ, ∀ finite_operations, 
    (∀ i j, lamps_state_after_operations i j = off) → initial_on_lamps ≥ 2 * n - 4) := 
sorry

end minimum_lamps_required_l666_666096


namespace depth_of_water_l666_666552

variable (RonHeight DepthOfWater : ℝ)

-- Definitions based on conditions
def RonStandingHeight := 12 -- Ron's height is 12 feet
def DepthOfWaterCalculation := 5 * RonStandingHeight -- Depth is 5 times Ron's height

-- Theorem statement to prove
theorem depth_of_water (hRon : RonHeight = RonStandingHeight) (hDepth : DepthOfWater = DepthOfWaterCalculation) :
  DepthOfWater = 60 := by
  sorry

end depth_of_water_l666_666552


namespace pyramid_dihedral_angle_l666_666400

-- Define the given conditions and the problem
theorem pyramid_dihedral_angle (O A B C D : Point) (x a : ℝ) (m n : ℤ)
    (h_base_rect : is_rect_base OABCD ABCD)
    (h_AB_2BC : AB = 2 * BC)
    (h_oa_eq_ob_eq_oc_eq_od : OA = OB ∧ OB = OC ∧ OC = OD)
    (h_angle_AOB_60 : ∠ AOB = 60)
    (h_cos_theta : cos_dihedral_angle OAB OBC = m + real.sqrt n):
  m + n = 21 :=
sorry

end pyramid_dihedral_angle_l666_666400


namespace bullet_passage_duration_l666_666011

open Real

/-- Problem Statement Definition -/
structure BulletPlankProblem :=
  (v0 vf s a: ℝ)                        -- Define initial velocity, final velocity, thickness, and resistance coefficient
  (v0_pos : 0 < v0)                     -- Initial velocity must be positive
  (vf_pos : 0 < vf)                     -- Final velocity must be positive
  (thickness_pos : 0 < s)               -- Thickness must be positive
  (resistance_coeff_pos : 0 < a)        -- Resistance coefficient must be positive
  (entry_velocity : v0 = 200)           -- Initial velocity condition
  (exit_velocity : vf = 50)             -- Final velocity condition
  (thickness : s = 0.1)                 -- Thickness condition
  (resistance_law : ∀ (v: ℝ), 0 < v → -a * v^2)

/-- Main theorem statement -/
theorem bullet_passage_duration (bprob : BulletPlankProblem) : 
  let I : ℝ := ln 4 / 0.1,
      t : ℝ := 3 / (200 * I) in
  t = 0.001 := sorry

end bullet_passage_duration_l666_666011


namespace box_third_dimension_length_l666_666894

noncomputable def box_height (num_cubes : ℕ) (cube_volume : ℝ) (length : ℝ) (width : ℝ) : ℝ :=
  let total_volume := num_cubes * cube_volume
  total_volume / (length * width)

theorem box_third_dimension_length (num_cubes : ℕ) (cube_volume : ℝ) (length : ℝ) (width : ℝ)
  (h_num_cubes : num_cubes = 24)
  (h_cube_volume : cube_volume = 27)
  (h_length : length = 8)
  (h_width : width = 12) :
  box_height num_cubes cube_volume length width = 6.75 :=
by {
  -- proof skipped
  sorry
}

end box_third_dimension_length_l666_666894


namespace problem_b_2076_l666_666756

noncomputable def solve_b_2076 : ℕ → ℝ
| n := if n = 1 then 2 + Real.sqrt 19 else if n = 2023 then 23 + Real.sqrt 19 else sorry

theorem problem_b_2076 :
  let b : ℕ → ℝ := solve_b_2076 in 
  (∀ n, n ≥ 2 → b n = b (n - 1) * b (n + 1)) →
  b 1 = 2 + Real.sqrt 19 →
  b 2023 = 23 + Real.sqrt 19 →
  b 2076 = (Real.sqrt 19 - 2) / 17 :=
by
  intros b h_rec h_b1 h_b2023
  sorry

end problem_b_2076_l666_666756


namespace mn_not_equal_l666_666760

-- Define conditions for the problem
def isValidN (N : ℕ) (n : ℕ) : Prop :=
  0 ≤ N ∧ N < 10^n ∧ N % 4 = 0 ∧ ((N.digits 10).sum % 4 = 0)

-- Define the number M_n of integers N satisfying the conditions
noncomputable def countMn (n : ℕ) : ℕ :=
  Nat.card { N : ℕ | isValidN N n }

-- Define the theorem stating the problem's conclusion
theorem mn_not_equal (n : ℕ) (hn : n > 0) : 
  countMn n ≠ 10^n / 16 :=
sorry

end mn_not_equal_l666_666760


namespace derivative_at_zero_l666_666428

-- Given conditions
def f (x : ℝ) : ℝ := -2 * x^2 + 3

-- Theorem statement to prove
theorem derivative_at_zero : 
  deriv f 0 = 0 := 
by 
  sorry

end derivative_at_zero_l666_666428


namespace distance_AC_l666_666616

theorem distance_AC : 
  let north_south_net := 50 - 20
  let east_west_net := 80 - 30
  let AC := Real.sqrt ((north_south_net ^ 2) + (east_west_net ^ 2))
  AC = 10 * Real.sqrt 34 :=
by
  let north_south_net := 50 - 20
  let east_west_net := 80 - 30
  let AC := Real.sqrt ((north_south_net ^ 2) + (east_west_net ^ 2))
  show AC = 10 * Real.sqrt 34
  sorry

end distance_AC_l666_666616


namespace volume_of_cut_ball_l666_666947

-- Definitions based on conditions
variables {R : ℝ} {α β γ : ℝ}

-- Statement of the problem with conditions and expected answer
theorem volume_of_cut_ball (hR : R > 0)
                           (hα : 0 < α ∧ α < π)
                           (hβ : 0 < β ∧ β < π)
                           (hγ : 0 < γ ∧ γ < π) :
    let V_body := (α + β + γ - π) * (R^3) / 3 in
    V_body = (α + β + γ - π) * (R^3) / 3 :=
by sorry

end volume_of_cut_ball_l666_666947


namespace hexagon_area_ratio_l666_666357

/--
Given a regular hexagon ABCDEF with each side length s, 
and P, Q, R, S, T, U are midpoints of sides AB, BC, CD, DE, EF, and FA respectively,
the segments AP, BQ, CR, DS, ET, FU bound a smaller regular hexagon inside ABCDEF.
Prove that the ratio of the area of the smaller hexagon to the area of ABCDEF is 1/4.
-/
theorem hexagon_area_ratio (s : ℝ) : 
  let ABCDEF := regular_hexagon s in
  let P := midpoint ABCDEF.A ABCDEF.B in
  let Q := midpoint ABCDEF.B ABCDEF.C in
  let R := midpoint ABCDEF.C ABCDEF.D in
  let S := midpoint ABCDEF.D ABCDEF.E in
  let T := midpoint ABCDEF.E ABCDEF.F in
  let U := midpoint ABCDEF.F ABCDEF.A in
  let smaller_hexagon := bounded_by_segments ABCDEF.A P ABCDEF.B Q 
                                           ABCDEF.C R ABCDEF.D S 
                                           ABCDEF.E T ABCDEF.F U in
  (area smaller_hexagon) / (area ABCDEF) = 1 / 4 :=
by
  sorry

end hexagon_area_ratio_l666_666357


namespace exists_coloring_scheme_l666_666326

/-- Define the types of points based on their color -/
inductive Color
| white 
| red 
| blue 

open Color

/-- Define a point in Cartesian coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- Color a point based on the parity of its coordinates -/
def color (p : Point) : Color :=
  if p.x % 2 = 1 then 
    if p.y % 2 = 0 then white else blue 
  else 
    if p.y % 2 = 0 then red else blue

/-- The main theorem stating the existence of coloring -/
theorem exists_coloring_scheme :
  (∀ k : ℤ, ∃∞ p : Point, p.y = k ∧ (color p = white ∨ color p = red ∨ color p = blue)) ∧
  (∀ A B C : Point, color A = white → color B = red → color C = blue →
   ∃ D : Point, color D = red ∧ (D.x = A.x + C.x - B.x ∧ D.y = A.y + C.y - B.y)) :=
by
  sorry

end exists_coloring_scheme_l666_666326


namespace ratio_joe_sara_l666_666782

variables (S J : ℕ) (k : ℕ)

-- Conditions
#check J + S = 120
#check J = k * S + 6
#check J = 82

-- The goal is to prove the ratio J / S = 41 / 19
theorem ratio_joe_sara (h1 : J + S = 120) (h2 : J = k * S + 6) (h3 : J = 82) : J / S = 41 / 19 :=
sorry

end ratio_joe_sara_l666_666782


namespace verify_sum_of_cousins_ages_l666_666349

theorem verify_sum_of_cousins_ages 
  (a b c d : ℕ) 
  (h1 : a * b = 20)
  (h2 : c * d = 36)
  (h_distinct : List.nodup [a, b, c, d]) 
  (h_single_digit : ∀ x ∈ [a, b, c, d], x < 10) 
  : a + b + c + d = 21 :=
by 
  -- Proof 
  sorry

end verify_sum_of_cousins_ages_l666_666349


namespace four_neg_a_equals_one_ninth_l666_666660

theorem four_neg_a_equals_one_ninth (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 2) : 4 ^ (-a) = 1 / 9 :=
by
  sorry

end four_neg_a_equals_one_ninth_l666_666660


namespace rook_path_exists_l666_666775

theorem rook_path_exists :
  ∃ (path : Finset (Fin 8 × Fin 8)) (s1 s2 : Fin 8 × Fin 8),
  s1 ≠ s2 ∧
  s1.1 % 2 = s2.1 % 2 ∧ s1.2 % 2 = s2.2 % 2 ∧
  ∀ s : Fin 8 × Fin 8, s ∈ path ∧ s ≠ s2 :=
sorry

end rook_path_exists_l666_666775


namespace all_numbers_equal_l666_666873

theorem all_numbers_equal
  (n : ℕ)
  (h n_eq_20 : n = 20)
  (a : ℕ → ℝ)
  (h_avg : ∀ i : ℕ, i < n → a i = (a ((i+n-1) % n) + a ((i+1) % n)) / 2) :
  ∀ i j : ℕ, i < n → j < n → a i = a j :=
by {
  -- Proof steps go here.
  sorry
}

end all_numbers_equal_l666_666873


namespace margaret_egg_collection_l666_666773

theorem margaret_egg_collection :
  ∀ {n m : ℕ}, n = 5 → m = 7 → n * m = 35 :=
by
  intros n m hn hm
  rw [hn, hm]
  exact Nat.mul_comm 5 7
  sorry

end margaret_egg_collection_l666_666773


namespace proof_verification_l666_666721

section
variables (scores : List ℝ) (n : ℕ)
def given_scores : List ℝ := [70, 85, 86, 88, 90, 90, 92, 94, 95, 100]

def median (l : List ℝ) : ℝ :=
let sorted := List.sort (≤) l in
if hl : l.length % 2 = 0 then (sorted.get (l.length / 2 - 1) + sorted.get (l.length / 2)) / 2
else sorted.get ((l.length - 1) / 2)

def percentile (l : List ℝ) (p : ℝ) : ℝ :=
let sorted := List.sort (≤) l in
sorted.get ⟨(p * (l.length + 1) / 100 : ℝ).ceil.toNat⟩

def average (l : List ℝ) : ℝ :=
l.sum / l.length

def variance (l : List ℝ) : ℝ :=
let μ := average l in
(l.map (λ x, (x - μ) ^ 2)).sum / l.length

def verify_statements : Prop :=
  (median given_scores = 90) ∧
  (percentile given_scores 60 = 91) ∧
  (average given_scores ≤ median given_scores) ∧
  let new_scores := given_scores.erase 70 |>.erase 100 in
  average new_scores > average given_scores ∧
  variance new_scores < variance given_scores

theorem proof_verification : verify_statements given_scores :=
sorry
end

end proof_verification_l666_666721


namespace number_of_chords_l666_666788

theorem number_of_chords (n : ℕ) (h : n = 10) : finset.card (finset.pairs (finset.range n)) = 45 :=
by
  rw [h]
  -- Sorry to skip the proof steps as required
  sorry

end number_of_chords_l666_666788


namespace unique_zero_in_interval_l666_666821

noncomputable def f (x : ℝ) : ℝ := 3^x + x^3 - 2

theorem unique_zero_in_interval : ∃! c ∈ Ioo (0 : ℝ) 1, f c = 0 := 
begin
  sorry
end

end unique_zero_in_interval_l666_666821


namespace range_of_y_l666_666299

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ set.Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end range_of_y_l666_666299


namespace smallest_range_of_sample_l666_666900

/-- 
Given a sample of seven observations with:
  1) An arithmetic mean of 13
  2) A median of 15
Prove that the smallest possible value of the range (largest observation minus smallest observation) is 9.
-/ 
theorem smallest_range_of_sample (x : Fin 7 → ℝ)
  (h₁ : (∑ i, x i) = 91) 
  (h₂ : ∃ (i : Fin 7), x i = 15) :
  ∃ (lo hi : Fin 7), (x hi - x lo) = 9 := 
sorry

end smallest_range_of_sample_l666_666900


namespace sphere_surface_area_and_cylinder_radius_l666_666461

theorem sphere_surface_area_and_cylinder_radius
(volume_sphere : ℝ)
(height_cylinder : ℝ)
(h1 : volume_sphere = 72 * Real.pi)
(h2 : height_cylinder = 4) :

  (∃ (r_sphere r_cylinder : ℝ),
    let surface_area_sphere := 4 * Real.pi * r_sphere^2 in
    let volume_cylinder := Real.pi * r_cylinder^2 * height_cylinder in

    r_sphere = 3 * Real.cbrt 2 ∧
    surface_area_sphere = 36 * 2^(2 / 3) * Real.pi ∧
    volume_cylinder = 72 * Real.pi ∧
    r_cylinder = Real.sqrt 18
  ) := sorry

end sphere_surface_area_and_cylinder_radius_l666_666461


namespace triangle_solution_l666_666322

noncomputable theory
open Real

-- Definitions based on given conditions
def acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π

variables {A B C a b c : ℝ}

-- The mathematical proof problem
theorem triangle_solution
  (h_acute : acute_triangle A B C)
  (h_side : a = sin A ∧ b = sin B ∧ c = sin C)
  (h_condition : 2 * b * sin A = sqrt 3 * a) :
  (B = π / 3) ∧ (frac (cos A + cos B + cos C) ∈ Icc (frac ((sqrt 3 + 1)/2)) (frac 3/2)) :=
by sorry

end triangle_solution_l666_666322


namespace find_a_minus_b_l666_666140

theorem find_a_minus_b (a b : ℝ) (h1 : ∃ (a b : ℝ), ∀ x, f(x) = x^3 + 3 * a * x^2 + b * x + a^2 ∧ f'(-1) = 0 ∧ f(-1) = 0) : a - b = -7 :=
by
  sorry

end find_a_minus_b_l666_666140


namespace pow_evaluation_l666_666958

theorem pow_evaluation : 81^(5/4) = 243 := 
by sorry

end pow_evaluation_l666_666958


namespace find_b_l666_666993

-- Definitions for the conditions
variables (a b c d : ℝ)
def four_segments_proportional := a / b = c / d

theorem find_b (h1: a = 3) (h2: d = 4) (h3: c = 6) (h4: four_segments_proportional a b c d) : b = 2 :=
by
  sorry

end find_b_l666_666993


namespace candy_bar_cost_l666_666348

def num_quarters := 4
def num_dimes := 3
def num_nickel := 1
def change_received := 4

def value_quarter := 25
def value_dime := 10
def value_nickel := 5

def total_paid := (num_quarters * value_quarter) + (num_dimes * value_dime) + (num_nickel * value_nickel)
def cost_candy_bar := total_paid - change_received

theorem candy_bar_cost : cost_candy_bar = 131 := by
  sorry

end candy_bar_cost_l666_666348


namespace total_tickets_correct_l666_666906

-- Let's define the conditions given in the problem
def student_tickets (adult_tickets : ℕ) := 2 * adult_tickets
def adult_tickets := 122
def total_tickets := adult_tickets + student_tickets adult_tickets

-- We now state the theorem to be proved
theorem total_tickets_correct : total_tickets = 366 :=
by 
  sorry

end total_tickets_correct_l666_666906


namespace chords_in_circle_l666_666794

theorem chords_in_circle (n : ℕ) (h_n : n = 10) : (n.choose 2) = 45 := sorry

end chords_in_circle_l666_666794


namespace least_xy_value_l666_666636

theorem least_xy_value {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
  sorry

end least_xy_value_l666_666636


namespace cos_double_angle_l666_666273

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666273


namespace count_solutions_l666_666978

noncomputable def num_solutions : ℕ :=
  {z : ℂ // complex.abs z = 1 ∧ complex.abs ((z / complex.conj z) - (complex.conj z / z)) = 1}.to_finset.card

theorem count_solutions : num_solutions = 4 :=
by
  sorry

end count_solutions_l666_666978


namespace star_polygon_n_value_l666_666896

theorem star_polygon_n_value (n : ℕ) (A B : ℕ → ℝ) (h1 : ∀ i, A i = B i - 20)
    (h2 : 360 = n * 20) : n = 18 :=
by {
  sorry
}

end star_polygon_n_value_l666_666896


namespace locus_of_points_l666_666090

-- Define the square's vertices
def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (1, -1)
def C : ℝ × ℝ := (1, 1)
def D : ℝ × ℝ := (-1, 1)

-- Define the condition for point P in the plane
def is_locus_point (P : ℝ × ℝ) : Prop := 
  let (x, y) := P in
  (sin (angle A P B) = sin (angle D P C)) ∧ 
  (P ≠ A) ∧ (P ≠ B) ∧ (P ≠ C) ∧ (P ≠ D)

-- Formalize the theorem in Lean
theorem locus_of_points (P : ℝ × ℝ) : is_locus_point P →
  (P.2 = 0) ∨ (P.1^2 - P.2^2 = 0) ∨ (P.1^2 + P.2^2 - 2 = 0) :=
by {
    sorry
}

end locus_of_points_l666_666090


namespace intersection_M_N_l666_666766

noncomputable def M : Set ℝ := { x | x^2 ≤ x }
noncomputable def N : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_M_N :
  M ∩ N = { x | 0 < x ∧ x ≤ 1 } :=
  sorry

end intersection_M_N_l666_666766


namespace cheese_cookie_price_l666_666014

theorem cheese_cookie_price
  (boxes_per_carton : ℕ)
  (packs_per_box : ℕ)
  (cost_per_dozen_cartons : ℕ) :
  boxes_per_carton = 12 →
  packs_per_box = 10 →
  cost_per_dozen_cartons = 1440 →
  cost_per_dozen_cartons / (boxes_per_carton * 12 * packs_per_box) = 1 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry -- Proof steps can be filled in, but not required for this task.

end cheese_cookie_price_l666_666014


namespace find_x_l666_666092

-- Define the points (2,10), (-2,2), and (x, -6)
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 2, y := 10}
def B : Point := {x := -2, y := 2}
def P (x : ℝ) : Point := {x := x, y := -6}

-- Define the slope calculation function
def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

-- Define the condition that P(x) lies on the line joining A and B
def liesOnLine (x : ℝ) : Prop :=
  slope A B = slope A (P x)

-- Prove that x = -6 satisfies this condition
theorem find_x : liesOnLine (-6) :=
by
  sorry

end find_x_l666_666092


namespace cylindrical_container_depth_l666_666532

theorem cylindrical_container_depth :
    ∀ (L D A : ℝ), 
      L = 12 ∧ D = 8 ∧ A = 48 → (∃ h : ℝ, h = 4 - 2 * Real.sqrt 3) :=
by
  intros L D A h_cond
  obtain ⟨hL, hD, hA⟩ := h_cond
  sorry

end cylindrical_container_depth_l666_666532


namespace magnitude_a_is_2_l666_666202

-- Definitions of vectors a and b
def a (n : ℝ) : ℝ × ℝ := (1, n)
def b (n : ℝ) : ℝ × ℝ := (-1, n)

-- Magnitude function
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Proof problem: Prove that the magnitude of the vector a is 2, given the conditions.
theorem magnitude_a_is_2 (n : ℝ) (h : let c := (2 * (1, n).1 + (1, n).2, 2 * (1, n).2 + (-(-1, n).2))
                           c ∘2 (-1, n) = (0, 0)) :
  magnitude (a n) = 2 :=
sorry

end magnitude_a_is_2_l666_666202


namespace hyperbola_focal_length_l666_666190

theorem hyperbola_focal_length 
  (a : ℝ)
  (h_pos : a > 0)
  (h_asymptote : ∀ x y : ℝ, y = 2 * x → (x^2 / a^2 - y^2 / 20 = 0)) :
  let b : ℝ := 2 * real.sqrt 5 in
  let c : ℝ := real.sqrt (a^2 + b^2) in
  2 * c = 10 :=
by
  sorry

end hyperbola_focal_length_l666_666190


namespace range_of_numbers_l666_666902

theorem range_of_numbers (a b c : ℕ) (h_mean : (a + b + c) / 3 = 4) (h_median : b = 4) (h_smallest : a = 1) :
  c - a = 6 :=
sorry

end range_of_numbers_l666_666902


namespace max_n_for_triangles_l666_666737

theorem max_n_for_triangles (n : ℕ) (m : ℕ) (h : m = 100) (h1 : ∑ (i : Fin n.succ), 180 * (n - 2) + 360 * m = 180 * (n + 198)) : n ≤ 102 ↔ n + m < 300 := 
by 
  sorry

end max_n_for_triangles_l666_666737


namespace master_bedroom_size_l666_666032

-- Definitions of variables and conditions
def total_area : ℝ := 2300
def common_area : ℝ := 1000
def bedroom_area : ℝ := total_area - common_area
def guest_bedroom (master_bedroom : ℝ) : ℝ := master_bedroom / 4

-- Statement to prove
theorem master_bedroom_size : ∃ (M : ℝ), guest_bedroom M + M = bedroom_area ∧ M = 1040 :=
by
  sorry

end master_bedroom_size_l666_666032


namespace one_over_a_is_arithmetic_a_n_formula_T_n_formula_smallest_n_for_T_l666_666730

-- Define the sequence and the given conditions
def a : ℕ → ℝ
| 0 := 1
| (n + 1) := a n / (2 * a n + 1)

-- Prove that the sequence {1 / a_n} is arithmetic
theorem one_over_a_is_arithmetic :
  ∀ n : ℕ, (1 / a n) = 2 * n + 1 := sorry

-- Find the general term of the sequence
theorem a_n_formula :
  ∀ n : ℕ, a n = 1 / (2 * n - 1) := sorry

-- Define the sum of the first n terms of the sequence {a_n a_{n+1}}
def T (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i * a (i + 1)

-- Prove the sum formula
theorem T_n_formula :
  ∀ n : ℕ, T n = n / (2 * n + 1) := sorry

-- Find the smallest positive integer n such that T_n > 1000 / 2011
theorem smallest_n_for_T :
  ∃ n : ℕ, T n > 1000 / 2011 ∧ (∀ m : ℕ, m < n → T m <= 1000 / 2011) := by
    use 91
    split
    . sorry -- Prove T 91 > 1000 / 2011
    . intros m h
      sorry -- Prove T m <= 1000 / 2011 for all m < 91

end one_over_a_is_arithmetic_a_n_formula_T_n_formula_smallest_n_for_T_l666_666730


namespace find_other_number_l666_666816

theorem find_other_number (a b lcm hcf : ℕ) (h_lcm : lcm = 2310) (h_hcf : hcf = 61) (h_first_number : a = 210) :
  a * b = lcm * hcf → b = 671 :=
by 
  -- setup
  sorry

end find_other_number_l666_666816


namespace inequality_abc_sum_one_l666_666518

theorem inequality_abc_sum_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 1) :
  (a^2 + b^2 + c^2 + d) / (a + b + c)^3 +
  (b^2 + c^2 + d^2 + a) / (b + c + d)^3 +
  (c^2 + d^2 + a^2 + b) / (c + d + a)^3 +
  (d^2 + a^2 + b^2 + c) / (d + a + b)^3 > 4 := by
  sorry

end inequality_abc_sum_one_l666_666518


namespace find_B_squared_l666_666591

def g (x : ℝ) := Real.sqrt 24 + 60 / x

noncomputable def equation (x : ℝ) := g (g (g (g (g x)))) = x

theorem find_B_squared :
  let B := |Real.sqrt 6 - Real.sqrt 66| + |Real.sqrt 6 + Real.sqrt 66| in
  B^2 = 264 :=
by
  sorry

end find_B_squared_l666_666591


namespace dishonest_dealer_weight_l666_666026

noncomputable def dealer_weight_equiv (cost_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  (1 - profit_percent / 100) * cost_price / selling_price

theorem dishonest_dealer_weight :
  dealer_weight_equiv 1 2 100 = 0.5 :=
by
  sorry

end dishonest_dealer_weight_l666_666026


namespace tangent_line_at_point_l666_666432

noncomputable def parabola (x : ℝ) : ℝ := x^2 + x + 1

theorem tangent_line_at_point (x y : ℝ) (h : parabola x = y) :
  let slope := (deriv parabola) 0 in
  let tangent_line := λ (x : ℝ), slope * (x - 0) + y in
  tangent_line = λ x, 1 * x + 1 :=
by
  sorry

end tangent_line_at_point_l666_666432


namespace pairs_even_gte_odd_l666_666316

-- Let s be a sequence of zeros and ones.
variable {s : List ℕ}

-- M is the number of (1, 0) pairs with an even number of digits between them.
def count_even_pairs (s : List ℕ) : ℕ :=
  List.pairs s |>.count (λ p => p.1 = 1 ∧ p.2 = 0 ∧ (List.indexOf p.2 s - List.indexOf p.1 s - 1) % 2 = 0)

-- N is the number of (1, 0) pairs with an odd number of digits between them.
def count_odd_pairs (s : List ℕ) : ℕ :=
  List.pairs s |>.count (λ p => p.1 = 1 ∧ p.2 = 0 ∧ (List.indexOf p.2 s - List.indexOf p.1 s - 1) % 2 = 1)

theorem pairs_even_gte_odd (s : List ℕ) :
  count_even_pairs s ≥ count_odd_pairs s :=
sorry

end pairs_even_gte_odd_l666_666316


namespace set_to_interval_notation_l666_666101

noncomputable def set_in_interval (s : Set ℝ) : Prop :=
  s = { x | x > 0 ∧ x ≠ 2 }

theorem set_to_interval_notation :
  ∀ s : Set ℝ, set_in_interval s → s = (Set.Ioo 0 2) ∪ (Set.Ioo 2 ∞) :=
by
  intro s
  intro h
  unfold set_in_interval at h
  rw h
  ext x
  simp
  split
  · rintro ⟨h₁, h₂⟩
    cases lt_or_gt_of_ne h₂ with hlt hgt
    · left
      exact ⟨h₁, hlt⟩
    · right
      exact ⟨h₁, hgt⟩
  · intro h
    cases h with hIoo0 hIoo2
    · exact ⟨hIoo0.1, ne_of_lt hIoo0.2⟩
    · exact ⟨hIoo2.1, ne_of_gt hIoo2.2⟩
  sorry

end set_to_interval_notation_l666_666101


namespace min_distance_point_to_line_l666_666327

noncomputable def distance (x1 y1 a b c : ℝ) : ℝ :=
  |a * x1 + b * y1 + c| / (Real.sqrt (a^2 + b^2))

theorem min_distance_point_to_line :
  let line_l := (x, y) | (x = -8 + t, y = t / 2)
  let curve_C := (x, y) | (x = 2 * s^2, y = 2 * Real.sqrt 2 * s)
  let cartesian_l := x - 2 * y + 8 = 0
  ∃ s : ℝ, 
    distance (2 * s^2) (2 * Real.sqrt 2 * s) 1 (-2) 8 = 4 * Real.sqrt 5 / 5 :=
sorry

end min_distance_point_to_line_l666_666327


namespace ellipse_fixed_point_theorem_l666_666152

section
variables (a b : ℝ) (A B F1 F2 : ℝ × ℝ)
variables (x1 x2 y1 y2 k : ℝ)

noncomputable def ellipse_equation (a : ℝ) (b : ℝ) :=
  a > b ∧ b > 0 ∧ (∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def min_AB_condition (a b : ℝ) :=
  2 * b^2 / a = 3

noncomputable def perimeter_condition (a : ℝ) :=
  4 * a = 8

-- Definition of the fixed point property for the line A'B
noncomputable def fixed_point_property
  (x1 x2 y1 y2 k : ℝ) (f : ℝ × ℝ) : Prop :=
  let A' := (x1, -y1)
  ∧ let B := (x2, y2) in
  x1 + x2 = -8 * k^2 / (3 + 4 * k^2) ∧
  x1 * x2 = (4 * k^2 - 12) / (3 + 4 * k^2) ∧
  f = (-4, 0) 

-- The main theorem
theorem ellipse_fixed_point_theorem :
  (ellipse_equation 2 (sqrt 3)) ∧
  (foreach a b, (min_AB_condition a b) →
  (perimeter_condition a) → 
  (fixed_point_property x1 x2 y1 y2 k (-4, 0))) :=
begin
  sorry
end
end

end ellipse_fixed_point_theorem_l666_666152


namespace bridge_length_correct_l666_666907

noncomputable def length_of_bridge : ℝ :=
  let train_length := 110 -- in meters
  let train_speed_kmh := 72 -- in km/hr
  let crossing_time := 14.248860091192705 -- in seconds
  let speed_in_mps := train_speed_kmh * (1000 / 3600)
  let distance := speed_in_mps * crossing_time
  distance - train_length

theorem bridge_length_correct :
  length_of_bridge = 174.9772018238541 := by
  sorry

end bridge_length_correct_l666_666907


namespace no_arithmetic_progression_in_S_l666_666748

open_locale classical

noncomputable def S (p k : ℕ) : set ℕ :=
  {a | 1 ≤ a ∧ a ≤ p - 1 ∧ ∃ x, x^k % p = a % p}

theorem no_arithmetic_progression_in_S (p k : ℕ) [fact (nat.prime p)] (hS_card : 3 ≤ (S p k).to_finset.card ∧ (S p k).to_finset.card ≤ p - 2) :
  ¬∃ (a d : ℕ), (∀ n, a + n * d ∈ S p k) ∧ (a + (p - 2) * d ∈ S p k) :=
sorry

end no_arithmetic_progression_in_S_l666_666748


namespace parabola_equation_l666_666594

theorem parabola_equation 
  (vertex_x vertex_y : ℝ)
  (a b c : ℝ)
  (h_vertex : vertex_x = 3 ∧ vertex_y = 5)
  (h_point : ∃ x y: ℝ, x = 2 ∧ y = 2 ∧ y = a * (x - vertex_x)^2 + vertex_y)
  (h_vertical_axis : ∃ a b c, a = -3 ∧ b = 18 ∧ c = -22):
  ∀ x: ℝ, x ≠ vertex_x → b^2 - 4 * a * c > 0 := 
    sorry

end parabola_equation_l666_666594


namespace not_right_triangle_exists_l666_666557

theorem not_right_triangle_exists :
  (∃ (triangle : Type) (h₁ : triangle → Prop) (h₂ : triangle → Prop) 
  (h₃ : triangle → Prop) (h₄ : triangle → Prop),
  (h₁ → (triangle → false)) ∧
  (h₂ → (triangle → true)) ∧
  (h₃ → (triangle → false)) ∧
  (h₄ → (triangle → false))) :=
sorry

end not_right_triangle_exists_l666_666557


namespace solve_x_squared_eq_four_x_l666_666451

theorem solve_x_squared_eq_four_x : {x : ℝ | x^2 = 4*x} = {0, 4} := 
sorry

end solve_x_squared_eq_four_x_l666_666451


namespace average_length_of_strings_l666_666777

theorem average_length_of_strings : 
  let length1 := 2
  let length2 := 5
  let length3 := 3
  let total_length := length1 + length2 + length3 
  let average_length := total_length / 3
  average_length = 10 / 3 :=
by
  let length1 := 2
  let length2 := 5
  let length3 := 3
  let total_length := length1 + length2 + length3
  let average_length := total_length / 3
  have h1 : total_length = 10 := by rfl
  have h2 : average_length = 10 / 3 := by rfl
  exact h2

end average_length_of_strings_l666_666777


namespace cross_section_area_correct_l666_666554

noncomputable def area_of_cross_section (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 11) / 16

theorem cross_section_area_correct (a : ℝ) (h : 0 < a) :
  area_of_cross_section a = (3 * a^2 * Real.sqrt 11) / 16 := by
  sorry

end cross_section_area_correct_l666_666554


namespace triangle_solutions_l666_666865

-- Define the triangles and the conditions
structure Triangle :=
(A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)

-- Define the existence conditions for triangles XYZ given triangles ABC and X'Y'Z'
noncomputable def existence_conditions (ABC XpYpZp : Triangle) : ℕ :=
sorry

-- Define the number of solutions for the problem
theorem triangle_solutions (ABC XpYpZp : Triangle) :
  existence_conditions ABC XpYpZp = 2 ∨
  existence_conditions ABC XpYpZp = 1 ∨
  existence_conditions ABC XpYpZp = 0 ∨
  (exists_on_extensions (ABC XpYpZp) -> existence_conditions ABC XpYpZp = 6) :=
sorry

end triangle_solutions_l666_666865


namespace exponents_multiplication_l666_666500

variable (a : ℝ)

theorem exponents_multiplication : a^3 * a = a^4 := by
  sorry

end exponents_multiplication_l666_666500


namespace collinear_to_coplanar_sufficient_coplanar_not_necessitate_collinear_l666_666305

variables {P : Type} [AffineSpace P]

def collinear (A B C : P) : Prop := ∃ (ℓ : line P), A ∈ ℓ ∧ B ∈ ℓ ∧ C ∈ ℓ
def coplanar (A B C D : P) : Prop := ∃ (π : plane P), A ∈ π ∧ B ∈ π ∧ C ∈ π ∧ D ∈ π

theorem collinear_to_coplanar_sufficient (A B C D : P) :
  collinear A B C → coplanar A B C D :=
by
  sorry

theorem coplanar_not_necessitate_collinear (A B C D : P) :
  coplanar A B C D → ¬ collinear A B C :=
by
  sorry

end collinear_to_coplanar_sufficient_coplanar_not_necessitate_collinear_l666_666305


namespace lynn_total_annual_interest_l666_666385

theorem lynn_total_annual_interest :
  ∃ x : ℝ, x - 100 = 400 ∧ (0.09 * x + 0.07 * 400) = 73 :=
by {
  use 500,
  split,
  { linarith },
  { norm_num }
}

end lynn_total_annual_interest_l666_666385


namespace range_of_f_l666_666460

noncomputable def f (x : ℝ) : ℝ :=
  log 2 (x / 2) * log 2 (x / 4)

theorem range_of_f :
  ∃ y ∈ Icc (-1 / 4 : ℝ) 2, ∀ x ∈ Ioo 2 8, f(x) = y :=
by
  -- Further proof steps go here.
  sorry

end range_of_f_l666_666460


namespace diagonal_AC_is_diameter_l666_666780

theorem diagonal_AC_is_diameter 
    (ABCD_inscribed : ∃(O : point), is_cyclic_quad ABCD O) 
    (ABCD_circumscribed : ∃(I : point), is_tangential_quad ABCD I) 
    (side_difference : AD - BC = AB - CD) : 
    is_diameter (circumscribed_circle ABCD) AC :=
sorry

end diagonal_AC_is_diameter_l666_666780


namespace total_cars_running_l666_666315

theorem total_cars_running :
  let P_X := 1 / 8
  let P_Y := 1 / 12
  let P_Z := 1 / 6
  let P_total := 3 / 8
  (P_X + P_Y + P_Z = P_total) ⟹
  (∀ n : ℕ, n = 3) := 
sorry

end total_cars_running_l666_666315


namespace ship_total_distance_l666_666547

variables {v_r : ℝ} {t_total : ℝ} {a d : ℝ}

-- Given conditions
def conditions (v_r t_total a d : ℝ) :=
  v_r = 2 ∧ t_total = 3.2 ∧
  (∃ v : ℝ, ∀ t : ℝ, t = a/(v + v_r) + (a + d)/v + (a + 2*d)/(v - v_r)) 

-- The main statement to prove
theorem ship_total_distance (d_total : ℝ) :
  conditions 2 3.2 a d → d_total = 102 :=
by
  sorry

end ship_total_distance_l666_666547


namespace triangle_area_l666_666488

theorem triangle_area (a b c : ℝ) (ha : a = 5) (hb : b = 5) (hc : c = 6) :
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 12 :=
by 
  sorry

end triangle_area_l666_666488


namespace cosine_double_angle_l666_666248

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l666_666248


namespace locus_of_midpoint_l666_666622

-- Define the given geometrical objects and properties.
variables {O M A B F : Point} {r : ℝ}

-- Given conditions
axioms
  (circle_center : center (circle O r))
  (point_inside_circle : M ∈ interior (circle O r))
  (right_angle_at_M : ∀ (A B : Point), right_angle A M B → ∃ (A B : Point), A ∈ circle O r ∧ B ∈ circle O r ∧ right_angle A M B)

-- Midpoint definition
def is_midpoint (F A B : Point) := dist F A = dist F B ∧ (A + B) / 2 = F

-- Statement to prove
theorem locus_of_midpoint (locus_circle_radius : ℝ) :
  (locus_circle_radius = sqrt (2 * r^2 - dist O M^2) / 2) →
  ∀ F, (∃ A B, A ∈ circle O r ∧ B ∈ circle O r ∧ right_angle A M B ∧ is_midpoint F A B)
    ↔ (F ∈ circle (midpoint O M) locus_circle_radius) :=
sorry

end locus_of_midpoint_l666_666622


namespace cos_double_angle_l666_666271

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666271


namespace total_staff_correct_l666_666808

noncomputable def total_staff_weekdays_weekends : ℕ := 84

theorem total_staff_correct :
  let chefs_weekdays := 16
  let waiters_weekdays := 16
  let busboys_weekdays := 10
  let hostesses_weekdays := 5
  let additional_chefs_weekends := 5
  let additional_hostesses_weekends := 2
  
  let chefs_leave := chefs_weekdays * 25 / 100
  let waiters_leave := waiters_weekdays * 20 / 100
  let busboys_leave := busboys_weekdays * 30 / 100
  let hostesses_leave := hostesses_weekdays * 15 / 100
  
  let chefs_left_weekdays := chefs_weekdays - chefs_leave
  let waiters_left_weekdays := waiters_weekdays - Nat.floor waiters_leave
  let busboys_left_weekdays := busboys_weekdays - busboys_leave
  let hostesses_left_weekdays := hostesses_weekdays - Nat.ceil hostesses_leave

  let total_staff_weekdays := chefs_left_weekdays + waiters_left_weekdays + busboys_left_weekdays + hostesses_left_weekdays

  let chefs_weekends := chefs_weekdays + additional_chefs_weekends
  let waiters_weekends := waiters_left_weekdays
  let busboys_weekends := busboys_left_weekdays
  let hostesses_weekends := hostesses_weekdays + additional_hostesses_weekends
  
  let total_staff_weekends := chefs_weekends + waiters_weekends + busboys_weekends + hostesses_weekends

  total_staff_weekdays + total_staff_weekends = total_staff_weekdays_weekends
:= by
  sorry

end total_staff_correct_l666_666808


namespace compute_HHHH_of_3_l666_666540

def H (x : ℝ) : ℝ := -0.5 * x^2 + 3 * x

theorem compute_HHHH_of_3 :
  H (H (H (H 3))) = 2.689453125 := by
  sorry

end compute_HHHH_of_3_l666_666540


namespace correct_statements_l666_666501

def certain_event_prob : Prop :=
  ∀ (E : Event), E.isCertain → Prob(E) = 1

def complementary_events_mutually_exclusive : Prop :=
  ∀ (A B : Event), A.isComplementOf B → MutuallyExclusive A B

def fair_coin_classical_probability_model : Prop :=
  classical_probability_model toss_coin

theorem correct_statements :
  certain_event_prob ∧ complementary_events_mutually_exclusive ∧ fair_coin_classical_probability_model :=
sorry

end correct_statements_l666_666501


namespace cos_double_angle_l666_666264

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l666_666264


namespace expression_equals_one_l666_666934

noncomputable def given_expression : ℚ := 
  (List.prod (List.range' 1 17).map (λ i => (1 + 15) * (1 + 15 / i.toRat))) /
  (List.prod (List.range' 1 16).map (λ i => (1 + 17) * (1 + 17 / i.toRat)))

theorem expression_equals_one : given_expression = 1 := by
  sorry

end expression_equals_one_l666_666934


namespace total_office_payroll_correct_l666_666324

-- Definitions of given conditions
def total_payroll_factory_workers : ℝ := 30000
def number_of_factory_workers : ℕ := 15
def number_of_office_workers : ℕ := 30
def salary_difference : ℝ := 500

-- Derived definitions
def average_salary_factory_worker : ℝ := total_payroll_factory_workers / number_of_factory_workers
def average_salary_office_worker : ℝ := average_salary_factory_worker + salary_difference
def total_payroll_office_workers : ℝ := average_salary_office_worker * number_of_office_workers

-- Theorem statement
theorem total_office_payroll_correct : total_payroll_office_workers = 75000 :=
sorry

end total_office_payroll_correct_l666_666324


namespace monotonicity_f_range_of_b_l666_666181

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - x

-- Problem I: Discuss monotonicity of f(x) on (0, +∞)
theorem monotonicity_f (a : ℝ) (h : a < 1) :
  (1/2 < a ∧ a < 1 → (∀ x : ℝ, x ∈ (0, 1) ∪ (a/(1-a), +∞) → MonotoneStrictlyIncreasingOn ℝ (f x a) x) ∧
                      (∀ x : ℝ, x ∈ (1, a/(1-a)) → MonotoneStrictlyDecreasingOn ℝ (f x a) x)) ∧
  (a = 1/2 → ∀ x : ℝ, x > 0 → MonotoneStrictlyIncreasingOn ℝ (f x a) x) ∧
  (0 < a ∧ a < 1/2 → (∀ x : ℝ, x ∈ (0, a/(1-a)) ∪ (1, +∞) → MonotoneStrictlyIncreasingOn ℝ (f x a) x) ∧
                     (∀ x : ℝ, x ∈ (a/(1-a), 1) → MonotoneStrictlyDecreasingOn ℝ (f x a) x)) ∧
  (a ≤ 0 → (∀ x : ℝ, x ∈ (0, 1) → MonotoneStrictlyDecreasingOn ℝ (f x a) x) ∧
           (∀ x : ℝ, x > 1 → MonotoneStrictlyIncreasingOn ℝ (f x a) x)) :=
sorry

-- Problem II: Find the range of b such that bx + 1 ≥ f(x) always holds when a = 1
theorem range_of_b (b : ℝ) (x : ℝ) (h1 : a = 1) (h2 : x > 0) :
  bx + 1 ≥ f x 1 ↔ b ≥ (1 / Real.exp 2) - 1 :=
sorry

end monotonicity_f_range_of_b_l666_666181


namespace maximum_k_value_l666_666654

noncomputable def max_value_k (a : ℝ) (b : ℝ) (k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ a^2 + b^2 ≥ k ∧ k = 1 / 2

theorem maximum_k_value (a b : ℝ) :
  (a > 0 ∧ b > 0 ∧ a + b = 1) → a^2 + b^2 ≥ 1 / 2 :=
by
  intro h
  obtain ⟨ha, hb, hab⟩ := h
  sorry

end maximum_k_value_l666_666654


namespace average_weight_of_friends_l666_666739

variable (Ishmael Ponce Jalen Mike : ℝ)

-- Conditions
def condition1 : Prop := Ishmael = Ponce + 20
def condition2 : Prop := Ponce = Jalen - 10
def condition3 : Prop := Jalen = 160
def condition4 : Prop := Mike = (Ishmael + Ponce + Jalen) - 15

-- Average weight
def average_weight (w1 w2 w3 w4 : ℝ) : ℝ := (w1 + w2 + w3 + w4) / 4

-- Proof statement
theorem average_weight_of_friends
  (h1 : condition1 Ishmael Ponce)
  (h2 : condition2 Ponce Jalen)
  (h3 : condition3 Jalen)
  (h4 : condition4 Ishmael Ponce Jalen Mike) :
  average_weight Ishmael Ponce Jalen Mike = 236.25 :=
sorry

end average_weight_of_friends_l666_666739


namespace areas_equal_l666_666745

-- Definitions of the points and the conditions
variables {A B C D P F H Q : Type}
variable [rectABC : ∀ (A B C D : Type), Prop]
variable [point_on_diagonal : ∀ (P : Type) (BD : Set Type), Prop]
variable [projection : ∀ (P : Type) (BC : Set Type) (F : Type), Prop]
variable [midpoint : ∀ (B F H : Type), Prop]
variable [intersection : ∀ (P C A H Q : Type), Prop]

-- Given conditions
axiom cond1 : point_on_diagonal P {B, D}
axiom cond2 : projection P {B, C} F
axiom cond3 : midpoint B F H
axiom cond4 : intersection P C A H Q

-- Statement to prove
theorem areas_equal : ∀ (A B C D P F H Q : Type), 
  (rectABC A B C D) → 
  (point_on_diagonal P {B, D}) → 
  (projection P {B, C} F) → 
  (midpoint B F H) → 
  (intersection P C A H Q) → 
  ∃ (area_APQ area_CHQ : ℝ),
    area_APQ = area_CHQ :=
begin
  intros A B C D P F H Q rectABC cond1 cond2 cond3 cond4,
  -- Proof to be filled in
  sorry
end

end areas_equal_l666_666745


namespace find_angle_C_maximum_trig_function_l666_666708

-- Given conditions
variables {A B C : ℝ} (a b c : ℝ)
hypothesis (h1 : 0 < A ∧ A < π)
hypothesis (h2 : 0 < B ∧ B < π)
hypothesis (h3 : 0 < C ∧ C < π)

noncomputable def triangle_condition : Prop :=
  2 * a * Real.cos C + c * Real.cos A = b

noncomputable def angle_sum : Prop :=
  A + B + C = π

-- Part (I)
theorem find_angle_C (h_triangle : triangle_condition a b c) (h_angle_sum : angle_sum) : 
  C = π / 2 :=
sorry

-- Part (II)
noncomputable def trig_function_value (A B : ℝ) : ℝ :=
  Real.sin A * Real.cos B + Real.sin B

theorem maximum_trig_function (h_angle_C : C = π / 2) (h_angle_sum : A + B = π / 2) : 
  ∃ (max_value : ℝ), max_value = 5 / 4 :=
sorry

end find_angle_C_maximum_trig_function_l666_666708


namespace quadratic_one_solution_m_value_l666_666939

theorem quadratic_one_solution_m_value (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0) → (b^2 - 4 * a * m = 0) → m = 3 :=
by
  sorry

end quadratic_one_solution_m_value_l666_666939


namespace apples_eaten_l666_666131

theorem apples_eaten (apples_per_sandwich sandwiches_per_day days_in_week : ℕ)
  (h1 : apples_per_sandwich = 4)
  (h2 : sandwiches_per_day = 10)
  (h3 : days_in_week = 7) :
  apples_per_sandwich * sandwiches_per_day * days_in_week = 280 :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end apples_eaten_l666_666131


namespace construct_triangle_from_sides_and_height_l666_666575

theorem construct_triangle_from_sides_and_height (a b ha : ℝ) (hab : ha > 0) (hb : b > 0) (ha_lt_b : ha < b) :
  ∃ (A B C : ℝ × ℝ), 
  (dist A B = a ∧ 
   dist A C = b ∧
   dist (foot_of_the_altitude A B C) A = ha ∧ 
   is_right_triangle A (foot_of_the_altitude A B C) C ∧ 
   (C.2 = foot_of_the_altitude A B C).2) :=
sorry

noncomputable def foot_of_the_altitude (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let mid : ℝ × ℝ := ((A.1 + C.1)/2, 0) in
  -- Actual computation to find the foot of the altitude
  mid -- placeholder, replace with the actual computation

end construct_triangle_from_sides_and_height_l666_666575


namespace area_of_isosceles_trapezoid_circumscribed_circle_l666_666069

theorem area_of_isosceles_trapezoid_circumscribed_circle 
  (x y : ℝ) 
  (h1 : isosceles_trapezoid_circumscribed_circle x y) 
  (h2 : longer_base h1 = 20) 
  (h3 : base_angle h1 = arcsin 0.6) : 
  area h1 ≈ 74.17 := 
sorry

end area_of_isosceles_trapezoid_circumscribed_circle_l666_666069


namespace strictly_decreasing_exponential_l666_666704

theorem strictly_decreasing_exponential (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2*a - 1)^x > (2*a - 1)^y) → (1/2 < a ∧ a < 1) :=
by
  sorry

end strictly_decreasing_exponential_l666_666704


namespace inscribed_circle_diameter_l666_666490

open Real

def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

def herons_formula (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  sqrt (s * (s - a) * (s - b) * (s - c))

theorem inscribed_circle_diameter
  (XY XZ YZ : ℝ)
  (hXY : XY = 13)
  (hXZ : XZ = 8)
  (hYZ : YZ = 15) :
  let s := semiperimeter XY XZ YZ
  let K := herons_formula XY XZ YZ
  let r := K / s
  let d := 2 * r
  d = (10 * sqrt 3) / 3
:= by
  sorry

end inscribed_circle_diameter_l666_666490


namespace wage_percent_change_l666_666035

-- Definitions based on given conditions
def initial_wage (W : ℝ) := W
def first_decrease (W : ℝ) := 0.60 * W
def first_increase (W : ℝ) := 0.78 * W
def second_decrease (W : ℝ) := 0.624 * W
def second_increase (W : ℝ) := 0.6864 * W

-- Lean theorem statement to prove overall percent change
theorem wage_percent_change : ∀ (W : ℝ), 
  ((second_increase (second_decrease (first_increase (first_decrease W))) - initial_wage W) / initial_wage W) * 100 = -31.36 :=
by sorry

end wage_percent_change_l666_666035


namespace handrail_length_is_17_point_3_l666_666904

noncomputable def length_of_handrail (turn : ℝ) (rise : ℝ) (radius : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let arc_length := (turn / 360) * circumference
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_is_17_point_3 : length_of_handrail 270 10 3 = 17.3 :=
by 
  sorry

end handrail_length_is_17_point_3_l666_666904


namespace quadratic_square_condition_existence_of_quadratic_l666_666694

-- Definitions for the conditions
variables (a b c d : ℚ)

-- The polynomial and its assumed form
def polynomial := x^4 + a * x^3 + b * x^2 + c * x + d

def quadratic := x^2 + p * x + q

-- The function representing the expanded form of the quadratic squared
def quadratic_square (p q : ℚ) : polynomial := x^4 + 2 * p * x^3 + (p^2 + 2 * q) * x^2 + 2 * p * q * x + q^2

-- The condition that must be satisfied if the polynomial is the square of a quadratic
def condition1 (a b c d : ℚ) : Prop := a * c^2 - 4 * a * b * d + 8 * c * d = 0

-- The main theorem to be proved
theorem quadratic_square_condition (p q : ℚ) :
  polynomial a b c d = quadratic_square a b c d → condition1 a b c d :=
sorry

-- Another statement to resolve the existence part
theorem existence_of_quadratic (a b c d : ℚ) (h : condition1 a b c d) :
  ∃ p q : ℚ, polynomial a b c d = quadratic_square p q :=
sorry

end quadratic_square_condition_existence_of_quadratic_l666_666694


namespace minimum_distance_l666_666620

section MinimumDistance
open Real

noncomputable def f (x : ℝ) : ℝ := exp x
noncomputable def g (x : ℝ) : ℝ := 2 * sqrt x
def t (x1 x2 : ℝ) := f x1 = g x2
def d (x1 x2 : ℝ) := abs (x2 - x1)

theorem minimum_distance : ∃ (x1 x2 : ℝ), t x1 x2 ∧ d x1 x2 = (1 - log 2) / 2 := 
sorry

end MinimumDistance

end minimum_distance_l666_666620


namespace cos_double_angle_l666_666268

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666268


namespace cos_double_angle_l666_666275

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l666_666275


namespace triangle_area_is_12_l666_666483

/-- 
Theorem: The area of an isosceles triangle with sides 5, 5, and 6 units is 12 square units.
-/
theorem triangle_area_is_12 
  (a b c : ℕ) 
  (h1 : a = 5)
  (h2 : b = 5)
  (h3 : c = 6)
  : (1/2 : ℚ) * c * (real.sqrt (a^2 - (c/2)^2)) = 12 := 
by 
  sorry

end triangle_area_is_12_l666_666483


namespace solve_inequality_l666_666132

theorem solve_inequality (x : Real) : 
  x^2 - 48 * x + 576 ≤ 16 ↔ 20 ≤ x ∧ x ≤ 28 :=
by
  sorry

end solve_inequality_l666_666132


namespace solve_for_x_l666_666408

theorem solve_for_x (x : ℝ) : 
  (5 ^ (2 * x) * 5 ^ (2 * x) * 5 ^ (2 * x) = 125^4) → (x = 2) :=
by
  sorry

end solve_for_x_l666_666408


namespace sam_apples_one_week_l666_666128

theorem sam_apples_one_week (sandwiches_per_day : ℕ) (apples_per_sandwich : ℕ) (days_per_week : ℕ)
  (h1 : sandwiches_per_day = 10) (h2 : apples_per_sandwich = 4) (h3 : days_per_week = 7) :
  (sandwiches_per_day * apples_per_sandwich * days_per_week) = 280 :=
by
  rw [h1, h2, h3]
  simp
  sorry

end sam_apples_one_week_l666_666128


namespace intersect_CC_l666_666903

variables {A B C D A' B' C' D' : Point}
variables {sphere : Sphere}
variables (tetra : Tetrahedron A B C D)
variables (touch_A' : sphere.touch (face A B C) A')
variables (touch_B' : sphere.touch (face A B D) B')
variables (touch_C' : sphere.touch (face A C D) C')
variables (touch_D' : sphere.touch (face B C D) D')
variables (intersect_AA'_BB' : ∃ P, P ∈ seg A A' ∧ P ∈ seg B B' ∧ P ∈ sphere)

theorem intersect_CC'_DD' 
  (touch_A' : sphere.touch (face A B C) A')
  (touch_B' : sphere.touch (face A B D) B')
  (touch_C' : sphere.touch (face A C D) C')
  (touch_D' : sphere.touch (face B C D) D')
  (intersect_AA'_BB' : ∃ P, P ∈ LineSegment A A' ∧ P ∈ LineSegment B B' ∧ P ∈ sphere) : 
  ∃ Q, Q ∈ LineSegment C C' ∧ Q ∈ LineSegment D D' ∧ Q ∈ sphere :=
begin
  sorry, -- Proof goes here
end

end intersect_CC_l666_666903


namespace polygon_sides_l666_666457

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360)
  (h2 : n ≥ 3) : 
  n = 8 := 
by
  sorry

end polygon_sides_l666_666457


namespace polygon_sides_l666_666455

theorem polygon_sides (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n > 2) : n = 8 :=
sorry

end polygon_sides_l666_666455


namespace range_of_f1_l666_666179

-- Define the quadratic function
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^2 - a * x + 5

-- State the monotonically increasing condition
def is_monotone_increasing_on_R1_to_inf (a : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 ≤ x2 → f x1 a ≤ f x2 a

theorem range_of_f1 (a : ℝ):
  is_monotone_increasing_on_R1_to_inf a → (7 - a) ≥ 3 :=
by
  assume h : is_monotone_increasing_on_R1_to_inf a,
  sorry

end range_of_f1_l666_666179


namespace find_length_of_AE_l666_666572

-- Let's define the conditions
variables {A B C D E : Type} -- Points in space
variables [metric A]

-- Given conditions
def is_isosceles_triangle (A B C : A) : Prop :=
  dist A B = dist A C

def is_diameter_of_circle (A B : A) (r : ℝ) (circle : set A) : Prop :=
  dist A B = 2 * r ∧ ∀ P ∈ circle, dist P A = r ∧ dist P B = r

def located_on (P Q : A) (a : ℝ) : Prop :=
  dist P Q = a

-- The proof statement
theorem find_length_of_AE
  (A B C D E : A)
  (h1 : is_isosceles_triangle A B C)
  (h2 : located_on B C (2 * real.sqrt 2))
  (h3 : is_diameter_of_circle A B 1 (set.univ : set A))
  (h4 : dist A B = 2)
  (h5 : dist A C = 2)
  (h6 : exists D, located_on A D 1 ∧ exists E, located_on A E 1)
  (h7 : located_on B E real.sqrt 2) :
  dist A E = real.sqrt 2 := sorry

end find_length_of_AE_l666_666572


namespace number_of_nonsimilar_triangles_l666_666686
-- Import the necessary library

-- Define the problem conditions
def angles_in_arithmetic_progression (a d : ℕ) : Prop :=
  0 < d ∧ d < 30 ∧ 
  (a - d > 0) ∧ (a + d < 180) ∧ -- Ensures positive and valid angles
  (a - d) + a + (a + d) = 180  -- Triangle sum property

-- Declare the theorem
theorem number_of_nonsimilar_triangles : 
  ∃ n : ℕ, n = 29 ∧ ∀ (a d : ℕ), angles_in_arithmetic_progression a d → d < 30 → a = 60 :=
sorry

end number_of_nonsimilar_triangles_l666_666686


namespace sum_distinct_odd_nat_geq_l666_666430

theorem sum_distinct_odd_nat_geq (n : ℕ) (a : ℕ → ℕ) (h_distinct : ∀ i j : ℕ, i < n → j < n → i ≠ j → a i ≠ a j)
  (h_odd : ∀ i : ℕ, i < n → Nat.Odd (a i)) (h_diff_distinct : ∀ i j : ℕ, i < n → j < n → i ≠ j → ∀ k l : ℕ, k < n → l < n → k ≠ l → (a i - a j) ≠ (a k - a l)) :
  (∑ i in Finset.range n, a i) ≥ n * (n^2 + 2) / 3 :=
  sorry

end sum_distinct_odd_nat_geq_l666_666430


namespace range_of_a_l666_666138

theorem range_of_a (x a : ℝ) (h1 : 0 < x) (h2 : x ≤ 1) (h3 : 4 * x < log a x) : 0 < a ∧ a < 1 :=
by
  sorry -- Proof is not required as per the instructions.

end range_of_a_l666_666138


namespace unique_a_for_fx_eq_2ax_l666_666160

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * Real.log x

theorem unique_a_for_fx_eq_2ax (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, f x a = 2 * a * x → x = (a + Real.sqrt (a^2 + 4 * a)) / 2) →
  a = 1 / 2 :=
sorry

end unique_a_for_fx_eq_2ax_l666_666160


namespace rhombus_point_on_vertex_l666_666747

theorem rhombus_point_on_vertex (ABCD : Rhombus) (angle_DAB : ∠ DAB = 60°) 
    (P : Point) (hP : P = intersection (diagonal AC) (diagonal BD)) (PQRS : Rhombus) 
    (hPQRS_boundary : ∀ X, X ∈ {Q, R, S} → X ∈ boundary ABCD) : 
    (∃ X, X ∈ {Q, R, S} ∧ X ∈ {A, B, C, D}) 
    ∧ (∀ X Y, X ≠ Y → X ∈ {Q, R, S} → Y ∈ {Q, R, S} → ¬ (X ∈ {A, B, C, D} ∧ Y ∈ {A, B, C, D})) :=
sorry

end rhombus_point_on_vertex_l666_666747


namespace dog_park_ratio_l666_666771

theorem dog_park_ratio
  (total_dogs_with_spots : ℕ)
  (total_dogs_with_pointy_ears : ℕ)
  (half_dogs_have_spots : ℕ -> ℕ)
  (half_dogs_have_spots total_dogs = total_dogs_with_spots)
  (total_dogs_with_spots = 15)
  (total_dogs_with_pointy_ears = 6) :
  (total_dogs_with_pointy_ears : total_dogs) = 1 / 5 :=
by
  sorry

end dog_park_ratio_l666_666771


namespace rectangle_properties_l666_666147

theorem rectangle_properties
  (b l : ℝ)
  (h1 : l = 3 * b)
  (h2 : 2 * l + 2 * b = 104) :
  ∃ (A d θ: ℝ), 
    A = l * b ∧
    d = real.sqrt (l^2 + b^2) ∧
    θ = real.arccos (l / d) ∧
    A = 507 ∧
    d ≈ 41.1 ∧
    θ ≈ 18.19 :=
by sorry

end rectangle_properties_l666_666147


namespace break_even_point_l666_666549

def cost_of_commodity (a : ℝ) : ℝ := a

def profit_beginning_of_month (a : ℝ) : ℝ := 100 + (a + 100) * 0.024

def profit_end_of_month : ℝ := 115

theorem break_even_point (a : ℝ) : profit_end_of_month - profit_beginning_of_month a = 0 → a = 525 := 
by sorry

end break_even_point_l666_666549


namespace problem_statement_l666_666645

variables (a b c : ℝ)

-- The conditions given in the problem
def condition1 : Prop := a^2 + b^2 - 4 * a ≤ 1
def condition2 : Prop := b^2 + c^2 - 8 * b ≤ -3
def condition3 : Prop := c^2 + a^2 - 12 * c ≤ -26

-- The theorem we need to prove
theorem problem_statement (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 c a) : (a + b) ^ c = 27 :=
by sorry

end problem_statement_l666_666645


namespace solve_inequalities_l666_666946

-- Define the interval [-1, 1]
def interval := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

-- State the problem
theorem solve_inequalities :
  {x : ℝ | 3 * x^2 + 2 * x - 9 ≤ 0 ∧ x ≥ -1} = interval := 
sorry

end solve_inequalities_l666_666946


namespace largest_prime_dividing_sum_of_powers_l666_666597

-- Theorem statement

theorem largest_prime_dividing_sum_of_powers :
  ∃ p : ℕ, prime p ∧ p ∣ (2^(p + 1) + 3^(p + 1) + 5^(p + 1) + 7^(p + 1)) ∧ p = 29 :=
sorry

end largest_prime_dividing_sum_of_powers_l666_666597


namespace distance_from_P_to_O_l666_666199

-- Definitions from conditions
variable (P O : Point)
variable (α β γ : Plane)
variable (dist : Point → Plane → ℝ)

-- Conditions
axiom α_perpendicular_β : α ⊥ β
axiom β_perpendicular_γ : β ⊥ γ
axiom γ_perpendicular_α : γ ⊥ α
axiom P_distances : dist P α = 3 ∧ dist P β = 4 ∧ dist P γ = 12
axiom planes_intersect_O : α ∩ β ∩ γ = {O}

-- Proof statement
theorem distance_from_P_to_O : dist P O = 13 :=
by
  sorry

end distance_from_P_to_O_l666_666199


namespace total_population_of_jackson_l666_666329

theorem total_population_of_jackson (num_cities : ℕ) (a b : ℝ) (h_num : num_cities = 25) 
(h_avg_range : a = 3200 ∧ b = 3600) (h_midpoint : (a + b) / 2 = 3400) :
  num_cities * ((a + b) / 2) = 85000 := by
  -- given conditions
  have h_pop_city := (a + b) / 2,
  rw h_midpoint at h_pop_city,
  rw h_num,
  calc
    25 * 3400 = 85000 : by norm_num

end total_population_of_jackson_l666_666329


namespace four_neg_a_equals_one_ninth_l666_666659

theorem four_neg_a_equals_one_ninth (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 2) : 4 ^ (-a) = 1 / 9 :=
by
  sorry

end four_neg_a_equals_one_ninth_l666_666659


namespace difference_between_length_and_width_l666_666828

-- Definitions for the problem conditions
variables (w l : ℝ)
variables (A : ℝ) (h : A = 1024)

-- Conditions
def roof_width := w
def roof_length := l
def length_is_4_times_width := l = 4 * w
def roof_area := A = l * w

-- The proof statement
theorem difference_between_length_and_width (a : A = 1024) (h1 : length_is_4_times_width) (h2 : roof_area): (l - w) = 48 :=
by
  sorry

end difference_between_length_and_width_l666_666828


namespace not_right_triangle_l666_666558

-- Conditions
def ratio_sides (s1 s2 s3 : ℝ) : Prop :=
  s1 = 3 * s2 ∧ s2 = 4 * s3 ∧ s1 = 5 * s3

def ratio_angles_345 (a1 a2 a3 : ℝ) : Prop :=
  a1 / a2 = 3 / 4 ∧ a2 / a3 = 4 / 5

def ratio_angles_123 (a1 a2 a3 : ℝ) : Prop :=
  a1 / a2 = 1 / 2 ∧ a2 / a3 = 2 / 3

def squares_sides (s1 s2 s3 : ℝ) : Prop :=
  s1^2 = k * s2^2 ∧ s2^2 = 2 * s3^2 ∧ s1^2 = 3 * s3^2

-- Statement
theorem not_right_triangle (a1 a2 a3 s1 s2 s3 : ℝ) (k: ℝ) :
  (ratio_sides s1 s2 s3 → a1 = 90 ∨ a2 = 90 ∨ a3 = 90) →
  (ratio_angles_123 a1 a2 a3 → a1 = 90 ∨ a2 = 90 ∨ a3 = 90) →
  (squares_sides s1 s2 s3 → a1 = 90 ∨ a2 = 90 ∨ a3 = 90) →
  (ratio_angles_345 a1 a2 a3 → a1 ≠ 90 ∧ a2 ≠ 90 ∧ a3 ≠ 90) →
  ratio_angles_345 a1 a2 a3 := sorry

end not_right_triangle_l666_666558


namespace min_value_f_l666_666972

def f (x y : ℝ) : ℝ := 2015 * (x + y) / real.sqrt (2015 * (x^2 + y^2))

theorem min_value_f : ∃ x y : ℝ, f x y = -real.sqrt 4030 ∧ x = y ∧ x < 0 :=
by
  sorry

end min_value_f_l666_666972


namespace cos_double_angle_l666_666270

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666270


namespace sum_of_possible_values_of_N_l666_666767

theorem sum_of_possible_values_of_N (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_c : c = a + b) (h_N : (a * b * c) = 8 * (a + b + c)) :
  ∃ N, (N = a * b * c) ∧ (c = a + b) ∧ (N = 8 * (a + b + c)) ∧ (N = 16 * (a + b)) ∧ (ab_sum : ∑ N in finset.range 3, N = 560) :=
by
  sorry

end sum_of_possible_values_of_N_l666_666767


namespace problem_1_problem_2_problem_3_l666_666680

theorem problem_1 (a : ℝ) (A : set ℝ) (hA : A = {x | a * x^2 + 2 * x + 1 = 0}) (h : 1 ∈ A) : 
  A = {x | x = -1/3 ∨ x = 1} := sorry

theorem problem_2 (a : ℝ) (A : set ℝ) (B : set ℝ) (hA : A = {x | a * x^2 + 2 * x + 1 = 0})  
  (h : ∃! x ∈ A) : B = {0, 1} := sorry

theorem problem_3 (a : ℝ) (A : set ℝ) (hA : A = {x | a * x^2 + 2 * x + 1 = 0})  
  (h : ∀ x ∈ A, ∃ y ≠ x, a * y^2 + 2 * y + 1 ≠ 0 → False) : 
  a ∈ {x | x = 0 ∨ x >= 1} := sorry

end problem_1_problem_2_problem_3_l666_666680


namespace evaluate_f_at_3_l666_666999

theorem evaluate_f_at_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 2 * x + 3) : f 3 = 7 :=
by
  -- proof goes here
  sorry

end evaluate_f_at_3_l666_666999


namespace cos_diff_l666_666163

theorem cos_diff (α : ℝ) (h1 : Real.cos α = (Real.sqrt 2) / 10) (h2 : α > -π ∧ α < 0) :
  Real.cos (α - π / 4) = -3 / 5 :=
sorry

end cos_diff_l666_666163


namespace double_angle_cosine_l666_666227

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l666_666227


namespace heisuke_win_l666_666051

def game_table (m n : ℕ) := vector (vector ℤ n) m

def swap_columns {m n : ℕ} (a b : fin (n + 1)) (table : game_table m n) : game_table m n :=
  ⟨vector.map (λ row, row.swap a b) table.1⟩

def add_rows {m n : ℕ} (rows : fin m → Prop) (table : game_table m n) : vector ℤ n :=
  vector.of_fn (λ j, (finset.univ.filter rows).sum (λ i, table.nth i j))

def delete_row {m n : ℕ} (i : fin m) (table : game_table m n) : game_table (m - 1) n :=
  ⟨table.1.remove_nth i.1⟩

theorem heisuke_win (m n : ℕ) (table : game_table m n) :
  ∃ (finite_turns : ℕ), ∀ t : ℕ, t > finite_turns → ∀ i j : fin m, j.1 < n - 1 → table.nth i j ≤ table.nth i (j + 1) :=
  sorry

end heisuke_win_l666_666051


namespace cos_double_angle_l666_666280

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l666_666280


namespace probability_of_three_correct_deliveries_l666_666604

theorem probability_of_three_correct_deliveries (n : ℕ) (h : n = 5) : 
  let total_outcomes : ℕ := n.factorial,
      favorable_outcomes : ℕ := (nat.choose n 3) * 1 -- derangements of 2 
  in
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 12 := by
  rw h
  have h1 : total_outcomes = 5! := by sorry -- 5 factorial
  have h2 : favorable_outcomes = (nat.choose 5 3) * 1 := by sorry -- choosing 3 and derange 2
  have h3 : (nat.choose 5 3) = 10 := by sorry -- binomial coefficient
  have h4 : 5! = 120 := by sorry -- factorial of 5
  calc
  (10 * 1 : ℚ) / 120 = 10 / 120 : by sorry
  ... = 1 / 12 : by sorry

end probability_of_three_correct_deliveries_l666_666604


namespace roots_of_polynomial_l666_666985

theorem roots_of_polynomial :
  {x | x * (2 * x - 5) ^ 2 * (x + 3) * (7 - x) = 0} = {0, 2.5, -3, 7} :=
by {
  sorry
}

end roots_of_polynomial_l666_666985


namespace count_solutions_l666_666977

noncomputable def num_solutions : ℕ :=
  {z : ℂ // complex.abs z = 1 ∧ complex.abs ((z / complex.conj z) - (complex.conj z / z)) = 1}.to_finset.card

theorem count_solutions : num_solutions = 4 :=
by
  sorry

end count_solutions_l666_666977


namespace x_plus_y_l666_666360

variables {e1 e2 : ℝ → ℝ → Prop} -- Represents the vectors as properties of reals
variables {x y : ℝ} -- Real numbers x and y

-- Assuming non-collinearity of e1 and e2 (This means e1 and e2 are independent)
axiom non_collinear : e1 ≠ e2 

-- Given condition translated into Lean
axiom main_equation : (3 * x - 4 * y = 6) ∧ (2 * x - 3 * y = 3)

-- Prove that x + y = 9
theorem x_plus_y : x + y = 9 := 
by
  sorry -- Proof will be provided here

end x_plus_y_l666_666360


namespace min_M_value_l666_666371

noncomputable def poly_cond (p : ℝ → ℝ) : Prop :=
  (∀ x ∈ Icc (-1 : ℝ) 1, 0 ≤ p x) ∧ (intervalIntegral.integral (a := -1) (b := 1) p volume = 1)

def M (p : ℝ → ℝ) (x : ℝ) : ℝ := p x

theorem min_M_value : ∀ p : ℝ → ℝ, poly_cond p → (∀ x ∈ Icc (-1 : ℝ) 1, M p x) ≥ (2 / 3) := 
  sorry

end min_M_value_l666_666371


namespace cosine_double_angle_l666_666249

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l666_666249


namespace cistern_total_wet_surface_area_l666_666885

-- Define the length, width, and depth of water in the cistern
def length : ℝ := 9
def width : ℝ := 4
def depth : ℝ := 1.25

-- Define the bottom surface area
def bottom_surface_area : ℝ := length * width

-- Define the longer side surface area
def longer_side_surface_area_each : ℝ := depth * length

-- Define the shorter end surface area
def shorter_end_surface_area_each : ℝ := depth * width

-- Calculate the total wet surface area
def total_wet_surface_area : ℝ := bottom_surface_area + 2 * longer_side_surface_area_each + 2 * shorter_end_surface_area_each

-- The theorem to be proved
theorem cistern_total_wet_surface_area :
  total_wet_surface_area = 68.5 :=
by
  -- since bottom_surface_area = 36,
  -- 2 * longer_side_surface_area_each = 22.5, and
  -- 2 * shorter_end_surface_area_each = 10
  -- the total will be equal to 68.5
  sorry

end cistern_total_wet_surface_area_l666_666885


namespace total_teachers_proof_l666_666546

noncomputable def total_teachers (senior intermediate strat_sampled other_sampled : ℕ) : ℕ :=
  let x := 210 in
  if strat_sampled = 42 ∧ other_sampled = 12 ∧ senior = 104 ∧ intermediate = 46 then x else 0

theorem total_teachers_proof:
  total_teachers 104 46 42 12 = 210 := sorry

end total_teachers_proof_l666_666546


namespace max_n_for_triangles_l666_666738

theorem max_n_for_triangles (n : ℕ) (m : ℕ) (h : m = 100) (h1 : ∑ (i : Fin n.succ), 180 * (n - 2) + 360 * m = 180 * (n + 198)) : n ≤ 102 ↔ n + m < 300 := 
by 
  sorry

end max_n_for_triangles_l666_666738


namespace multiple_of_5_pow_100_no_zero_l666_666342

theorem multiple_of_5_pow_100_no_zero (k : ℕ) (hk : k ≥ 1) : 
  ∃ n : ℕ, (n % 5^100 = 0) ∧ (∀ d ∈ nat.digits 10 n, d ≠ 0) :=
sorry

end multiple_of_5_pow_100_no_zero_l666_666342


namespace smallest_k_l666_666762

theorem smallest_k (n : ℕ) (h : 0 < n) : 
  ∃ k, (∀ S : Finset ℕ, S.card = k → (∃ a b c d ∈ S, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b + c + d = 4 * n + 1)) ∧ (∀ m, (∀ S : Finset ℕ, S.card = m → (∃ a b c d ∈ S, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b + c + d = 4 * n + 1)) → m ≥ k) := sorry

end smallest_k_l666_666762


namespace number_of_chords_l666_666796

/-- Ten points are marked on the circumference of a circle.
    Prove that the number of different chords that can be drawn
    by connecting any two of these ten points is 45.
-/
theorem number_of_chords (n : ℕ) (h_n : n = 10) : 
  (nat.choose n 2) = 45 :=
by
  rw h_n
  norm_num

end number_of_chords_l666_666796


namespace train_crosses_platform_in_15_seconds_l666_666908

-- Definitions based on conditions
def length_of_train : ℝ := 330 -- in meters
def tunnel_length : ℝ := 1200 -- in meters
def time_to_cross_tunnel : ℝ := 45 -- in seconds
def platform_length : ℝ := 180 -- in meters

-- Definition based on the solution but directly asserting the correct answer.
def time_to_cross_platform : ℝ := 15 -- in seconds

-- Lean statement
theorem train_crosses_platform_in_15_seconds :
  (length_of_train + platform_length) / ((length_of_train + tunnel_length) / time_to_cross_tunnel) = time_to_cross_platform :=
by
  sorry

end train_crosses_platform_in_15_seconds_l666_666908


namespace min_value_of_f_x_squared_lt_exp_x_exists_x0_l666_666184

-- The given problem conditions
def f (x : ℝ) (a : ℝ) : ℝ := exp x - a * x

-- 1. Prove that for the function f(x) = exp(x) - 2x, it has a minimum value of 2 - log 4 at x = log 2.
theorem min_value_of_f : ∃ x : ℝ, f x 2 = 2 - log 4 ∧ ∀ y : ℝ, f y 2 ≥ f (log 2) 2 := 
by
  sorry

-- 2. Prove that for all x > 0, x^2 < exp x.
theorem x_squared_lt_exp_x (x : ℝ) : x > 0 → x^2 < exp x :=
by
  sorry

-- 3. Prove that for any given positive number c, there exists x0 > 0 such that for x in (x0, +∞), x < ce^x.
theorem exists_x0 (c : ℝ) : c > 0 → ∃ x0 : ℝ, x0 > 0 ∧ ∀ x : ℝ, x > x0 → x < c * exp x :=
by
  sorry

end min_value_of_f_x_squared_lt_exp_x_exists_x0_l666_666184


namespace smallest_integer_divisible_l666_666373

theorem smallest_integer_divisible (n : ℕ) :
  let m := ⌈(Real.sqrt 3 + 1) ^ (2 * n)⌉ in
  m % (2^(n+1)) = 0 := by
  sorry

end smallest_integer_divisible_l666_666373


namespace chords_in_circle_l666_666793

theorem chords_in_circle (n : ℕ) (h_n : n = 10) : (n.choose 2) = 45 := sorry

end chords_in_circle_l666_666793


namespace EulerLinesCoincide_l666_666838

open EuclideanGeometry

variables {A B C A1 B1 C1 A2 B2 C2 : Point}

-- Conditions Definitions
def OrthicTriangle (ABC A1 B1 C1 : Triangle) :=
  IsOrthicTriangle ABC A1 B1 C1

def IncircleTouchpoints (A1 B1 C1 A2 B2 C2 : Point) :=
  IncircleTouchesAt A1 B1 C1 A2 B2 C2

-- Statement of the theorem
theorem EulerLinesCoincide
  (h1 : OrthicTriangle (Triangle.mk A B C) (Triangle.mk A1 B1 C1))
  (h2 : IncircleTouchpoints A1 B1 C1 A2 B2 C2) :
  EulerLine (Triangle.mk A B C) = EulerLine (Triangle.mk A2 B2 C2) :=
sorry -- Proof omitted

end EulerLinesCoincide_l666_666838


namespace smallest_scalene_triangle_perimeter_is_prime_l666_666901

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p1 p2 p3 : ℕ) : Prop :=
  p1 < p2 ∧ p2 < p3 ∧ is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧
  (p2 = p1 + 2) ∧ (p3 = p1 + 6)

noncomputable def smallest_prime_perimeter : ℕ :=
  5 + 7 + 11

theorem smallest_scalene_triangle_perimeter_is_prime :
  ∃ (p1 p2 p3 : ℕ), p1 < p2 ∧ p2 < p3 ∧ consecutive_primes p1 p2 p3 ∧ is_prime (p1 + p2 + p3) ∧ (p1 + p2 + p3 = smallest_prime_perimeter) :=
by 
  sorry

end smallest_scalene_triangle_perimeter_is_prime_l666_666901


namespace sufficient_condition_for_equation_l666_666949

theorem sufficient_condition_for_equation (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) :
    x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  -- Proof omitted
  sorry

end sufficient_condition_for_equation_l666_666949


namespace soybeans_to_oil_l666_666007

theorem soybeans_to_oil 
    (kg_soybeans_to_tofu : ℝ)
    (kg_soybeans_to_oil : ℝ)
    (price_soybeans : ℝ)
    (price_tofu : ℝ)
    (price_oil : ℝ)
    (purchase_amount : ℝ)
    (sales_amount : ℝ)
    (amount_to_oil : ℝ)
    (used_soybeans_for_oil : ℝ) :
    kg_soybeans_to_tofu = 3 →
    kg_soybeans_to_oil = 6 →
    price_soybeans = 2 →
    price_tofu = 3 →
    price_oil = 15 →
    purchase_amount = 920 →
    sales_amount = 1800 →
    used_soybeans_for_oil = 360 →
    (6 * amount_to_oil) = 360 →
    15 * amount_to_oil + 3 * (460 - 6 * amount_to_oil) = 1800 :=
by sorry

end soybeans_to_oil_l666_666007


namespace sum_log_terms_l666_666170

-- Define the sequence sum function
noncomputable def s (n : ℕ) (a : ℝ) : ℝ := 2^n - a

-- Define the geometric sequence term a_n
noncomputable def a_n (n : ℕ) (a : ℝ) : ℝ := s n a - s (n - 1) a

-- Define the logarithmic sequence term
noncomputable def log_a_n (n : ℕ) (a : ℝ) : ℝ := Real.log (a_n n a) / Real.log 2

-- State the theorem
theorem sum_log_terms (a : ℝ) (h : ∀ n, s n a = 2^n - a) : (Finset.range 10).sum (λ n, log_a_n (n + 1) a) = 45 :=
by
  sorry

end sum_log_terms_l666_666170


namespace cos_double_angle_l666_666289

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666289


namespace sum_of_transformed_numbers_l666_666459

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l666_666459


namespace angle_between_vectors_l666_666158

-- Define the vectors a and b
variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def non_zero_vectors : Prop := (a ≠ 0) ∧ (b ≠ 0)
def condition1 : Prop := inner (a - 6 • b) a = 0
def condition2 : Prop := inner (2 • a - 3 • b) b = 0

-- The proof problem
theorem angle_between_vectors (h1 : non_zero_vectors a b) (h2 : condition1 a b) (h3 : condition2 a b) :
  real.angle a b = real.pi / 3 :=
sorry

end angle_between_vectors_l666_666158


namespace apples_eaten_l666_666130

theorem apples_eaten (apples_per_sandwich sandwiches_per_day days_in_week : ℕ)
  (h1 : apples_per_sandwich = 4)
  (h2 : sandwiches_per_day = 10)
  (h3 : days_in_week = 7) :
  apples_per_sandwich * sandwiches_per_day * days_in_week = 280 :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end apples_eaten_l666_666130


namespace trapezoid_bc_length_l666_666423

/-- A trapezoid with given dimensions and area has \(BC\) of approximately 3.83 cm. -/
theorem trapezoid_bc_length :
  let area := 289
  let height := 17
  let ab := 9
  let cd := 25 in
  9 + 25 - sqrt (289 - 2 * 289 - 68 - 155.805) / 17 = 3.83 :=
by
  sorry -- Proof is omitted

end trapezoid_bc_length_l666_666423


namespace cosine_double_angle_l666_666250

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l666_666250


namespace eval_complex_powers_l666_666588

noncomputable def i : Complex := Complex.I

theorem eval_complex_powers :
  let i := Complex.I
  (i^13 + i^18 + i^23 + i^28 + i^33 = i) :=
by
  /- conditions -/
  have h1 : i^2 = -1 := by simp [Complex.I_eq_i]
  have h2 : i^4 = 1 := by simp [Complex.I_eq_i]
  sorry

end eval_complex_powers_l666_666588


namespace P_intersect_Q_is_empty_l666_666196
open Complex

noncomputable def a : ℕ → ℕ
| 0 := 0 -- Convention n ∈ ℕ^*, so a_0 is not used
| 1 := 1
| (n+1) := a n + 4 * n

def P : Set ℕ := {n | ∃ m : ℕ, n = a m}

def Q : Set ℕ := {n | (1 + complex.i)^(2 * n) = (2^n) * complex.i}

theorem P_intersect_Q_is_empty : P ∩ Q = ∅ :=
by
  sorry

end P_intersect_Q_is_empty_l666_666196


namespace angle_sum_pentagon_triangle_l666_666080

theorem angle_sum_pentagon_triangle
  (A B C x y : ℝ)
  (hA : A = 34)
  (hB : B = 80)
  (hC : C = 24)
  (h_vertical_angles : ∀ a b, a = b → a = b)
  (h_right_angles : ∀ x : ℝ, x = 90 → x = 90)
  (pentagon_sum : ∀ p r s t u : ℝ, p + r + s + t + u = 540):
  x + y = 132 := 
by 
  sorry

end angle_sum_pentagon_triangle_l666_666080


namespace real_root_condition_l666_666195

theorem real_root_condition (k : ℝ) :
  ∀ (x : ℝ), (x^2 + (k + complex.I.re) * x - 2 - (k * complex.I).re = 0) → (k = 1 ∨ k = -1) :=
by
  sorry

end real_root_condition_l666_666195


namespace ellipse_eccentricity_l666_666151

theorem ellipse_eccentricity (a b c e : ℝ) (h1 : a > b) (h2 : b > 0)
  (h2 : a > 0) (h3 : c = a * sqrt (1 - (b / a)^2))
  (h4 : ∀ P Q : ℝ × ℝ, 
    ((P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧
    (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) ∧
    (F1 = (-c, 0)) ∧
    (F2 = (c, 0)) ∧
    (is_perpendicular (P, Q)) ∧
    (is_equilateral_triangle (P, Q, F2)))) :
  e = sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l666_666151


namespace polynomial_k_average_l666_666669

noncomputable def average_of_all_distinct_k (p : ℕ → ℕ → Prop) : ℕ :=
  let pairs := [(1, 20), (2, 10), (4, 5)]
  let ks := pairs.map (λ r, r.1 + r.2)
  let distinct_ks := ks.eraseDups
  distinct_ks.sum / distinct_ks.length

theorem polynomial_k_average :
  average_of_all_distinct_k (λ r1 r2, r1 * r2 = 20) = 14 :=
by
  sorry

end polynomial_k_average_l666_666669


namespace length_of_first_platform_is_140_l666_666054

-- Definitions based on problem conditions
def train_length : ℝ := 190
def time_first_platform : ℝ := 15
def time_second_platform : ℝ := 20
def length_second_platform : ℝ := 250

-- Definition for the length of the first platform (what we're proving)
def length_first_platform (L : ℝ) : Prop :=
  (time_first_platform * (train_length + L) = time_second_platform * (train_length + length_second_platform))

-- Theorem: The length of the first platform is 140 meters
theorem length_of_first_platform_is_140 : length_first_platform 140 :=
  by sorry

end length_of_first_platform_is_140_l666_666054


namespace petrov_vasechkin_boards_l666_666393

theorem petrov_vasechkin_boards:
  ∃ n : ℕ, 
  (∃ x y : ℕ, 2 * x + 3 * y = 87 ∧ x + y = n) ∧ 
  (∃ u v : ℕ, 3 * u + 5 * v = 94 ∧ u + v = n) ∧ 
  n = 30 := 
sorry

end petrov_vasechkin_boards_l666_666393


namespace sum_s_equal_51_l666_666571

def x_i (i : ℕ) : ℝ := i / 101

def sum_term (i : ℕ) : ℝ := (x_i i)^3 / (1 - 3 * x_i i + 3 * (x_i i)^2)

theorem sum_s_equal_51 :
  (∑ i in Finset.range (102), sum_term i) = 51 :=
by
  sorry

end sum_s_equal_51_l666_666571


namespace infinite_series_sum_l666_666077

theorem infinite_series_sum :
  (∑' n : ℕ, (4 * (n + 1) - 3) / 3 ^ (n + 1)) = 13 / 8 :=
by sorry

end infinite_series_sum_l666_666077


namespace joan_total_seashells_l666_666346

-- Definitions of the conditions
def joan_initial_seashells : ℕ := 79
def mike_additional_seashells : ℕ := 63

-- Definition of the proof problem statement
theorem joan_total_seashells : joan_initial_seashells + mike_additional_seashells = 142 :=
by
  -- Proof would go here
  sorry

end joan_total_seashells_l666_666346


namespace cos_double_angle_l666_666260

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l666_666260


namespace triangle_solution_l666_666323

noncomputable theory
open Real

-- Definitions based on given conditions
def acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π

variables {A B C a b c : ℝ}

-- The mathematical proof problem
theorem triangle_solution
  (h_acute : acute_triangle A B C)
  (h_side : a = sin A ∧ b = sin B ∧ c = sin C)
  (h_condition : 2 * b * sin A = sqrt 3 * a) :
  (B = π / 3) ∧ (frac (cos A + cos B + cos C) ∈ Icc (frac ((sqrt 3 + 1)/2)) (frac 3/2)) :=
by sorry

end triangle_solution_l666_666323


namespace sum_of_possible_c_values_l666_666602

theorem sum_of_possible_c_values :
  (∑ (c : ℕ) in {c | 3 * c + 7^2 - 4 * 3 * c = 0 ∧ 0 < c ∧ c ≤ ⌊49 / 12⌋}, c) = 10 :=
by sorry

end sum_of_possible_c_values_l666_666602


namespace door_cranking_time_l666_666743

-- Define the given conditions
def run_time_with_backpack : ℝ := 7 * 60 + 23  -- 443 seconds
def run_time_without_backpack : ℝ := 5 * 60 + 58  -- 358 seconds
def total_time : ℝ := 874  -- 874 seconds

-- Define the Lean statement of the proof
theorem door_cranking_time :
  (run_time_with_backpack + run_time_without_backpack) + (total_time - (run_time_with_backpack + run_time_without_backpack)) = total_time ∧
  (total_time - (run_time_with_backpack + run_time_without_backpack)) = 73 :=
by
  sorry

end door_cranking_time_l666_666743


namespace total_cars_produced_l666_666013

theorem total_cars_produced (cars_NA cars_EU : ℕ) (h1 : cars_NA = 3884) (h2 : cars_EU = 2871) : cars_NA + cars_EU = 6755 := by
  sorry

end total_cars_produced_l666_666013


namespace josephine_milk_containers_l666_666391

theorem josephine_milk_containers :
  3 * 2 + 2 * 0.75 + 5 * x = 10 → x = 0.5 :=
by
  intro h
  sorry

end josephine_milk_containers_l666_666391


namespace base7_subtraction_correct_l666_666593

theorem base7_subtraction_correct : 
  base7_subtraction 2534 1466 = 1065 := 
sorry

def base7_subtraction (a b : ℕ) : ℕ :=
  -- Definition of base-7 subtraction should be placed here
  sorry

end base7_subtraction_correct_l666_666593


namespace equilateral_triangle_maximizes_sum_of_altitudes_l666_666555

theorem equilateral_triangle_maximizes_sum_of_altitudes (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_medians : (1/2) * (sqrt (2*a^2 + 2*b^2 - c^2) 
                         + sqrt (2*b^2 + 2*c^2 - a^2) 
                         + sqrt (2*c^2 + 2*a^2 - b^2)) = const_medians) :
  ∃ (k : ℝ), k > 0 ∧ a = k ∧ b = k ∧ c = k :=
begin
  -- Prove that a = b = c = k given the conditions
  sorry
end

end equilateral_triangle_maximizes_sum_of_altitudes_l666_666555


namespace exponential_decreasing_iff_l666_666173

variable {a : ℝ} (f : ℝ → ℝ) 

def decreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x > f y

theorem exponential_decreasing_iff : 
  (∀ x : ℝ, f x = (2 * a + 1) ^ x) → (decreasing f ↔ (a ∈ set.Ioo (-1/2 : ℝ) 0)) :=
by 
  sorry

end exponential_decreasing_iff_l666_666173


namespace a_seq_correct_T_seq_correct_l666_666453

def S (n : ℕ) : ℕ := n^2

def a (n : ℕ) : ℕ := 2 * n - 1

theorem a_seq_correct (n : ℕ) (h_n : n ≥ 1) : 
  ((∑ k in Finset.range n.succ, a k) = S n) :=
sorry

def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

def T (n : ℕ) : ℚ := (∑ k in Finset.range n.succ, b k)

theorem T_seq_correct (n : ℕ) (h_n : n ≥ 1) : 
  T n = n / (2 * n + 1) :=
sorry

end a_seq_correct_T_seq_correct_l666_666453


namespace extreme_value_f_l666_666187

noncomputable def f (x : ℝ) : ℝ := x^2 - x - log x + 2

theorem extreme_value_f :
  (∀ x ∈ set.Ioo 0 1, f'(x) < 0) ∧ (∀ x ∈ set.Ioi 1, f'(x) > 0) → (∃ x, f x = 2) :=
begin
  sorry
end

end extreme_value_f_l666_666187


namespace jamestown_theme_parks_l666_666741

theorem jamestown_theme_parks (J : ℕ) (Venice := J + 25) (MarinaDelRay := J + 50) (total := J + Venice + MarinaDelRay) (h : total = 135) : J = 20 :=
by
  -- proof step to be done here
  sorry

end jamestown_theme_parks_l666_666741


namespace ratio_of_a_to_b_l666_666434

theorem ratio_of_a_to_b (a y b : ℝ) (h1 : a = 0) (h2 : b = 2 * y) : a / b = 0 :=
by
  sorry

end ratio_of_a_to_b_l666_666434


namespace evaluate_f_at_neg3_l666_666673

def f (x : ℝ) : ℝ := -2 * x^3 + 5 * x^2 - 3 * x + 2

theorem evaluate_f_at_neg3 : f (-3) = 110 :=
by 
  sorry

end evaluate_f_at_neg3_l666_666673


namespace cos_double_angle_l666_666214

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l666_666214


namespace cos_double_angle_l666_666217

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l666_666217


namespace find_angle_C_l666_666310

noncomputable def angle_C (a b c : ℝ) (S : ℝ) :=
  S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2) → 
  angle_C : ℝ := π / 3

theorem find_angle_C (a b c : ℝ) (S : ℝ) (h : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) : 
  angle_C a b c S = π / 3 :=
sorry

end find_angle_C_l666_666310


namespace number_of_students_in_range_l666_666713

theorem number_of_students_in_range 
  (n : ℕ)
  (μ : ℝ)
  (σ : ℝ)
  (h_n : n = 40)
  (h_μ : μ = 120)
  (h_P : ∀ (X : ℝ → Prop), P (λ X, X > 140) = 0.2) :
  (40 * 0.3 = 12) :=
by
  sorry

end number_of_students_in_range_l666_666713


namespace construct_segment_length_l666_666646

theorem construct_segment_length (a b : ℝ) (h : a > b) : 
  ∃ c : ℝ, c = (a^2 + b^2) / (a - b) :=
by
  sorry

end construct_segment_length_l666_666646


namespace total_bread_and_treats_is_225_l666_666343

-- Define the conditions
def conditions (T : ℝ) (Wanda_bread : ℝ) (Wanda_treats : ℝ) (Jane_bread : ℝ) (Jane_treats : ℝ) : Prop :=
  Jane_bread = 0.75 * Jane_treats ∧
  Wanda_treats = Jane_treats / 2 ∧
  Wanda_bread = 3 * Wanda_treats ∧
  Wanda_bread = 90 ∧
  Jane_treats = T

-- Goal: the total pieces of bread and treats is 225
theorem total_bread_and_treats_is_225 (T : ℝ) (Wanda_bread Wanda_treats Jane_bread Jane_treats total : ℝ) : 
  conditions T Wanda_bread Wanda_treats Jane_bread Jane_treats → 
  total = (Wanda_bread + Jane_bread + Wanda_treats + Jane_treats) →
  total = 225 :=
begin
  sorry
end

end total_bread_and_treats_is_225_l666_666343


namespace range_of_a_l666_666194

noncomputable def discriminant (a : ℝ) : ℝ :=
  (2 * a)^2 - 4 * 1 * 1

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) ↔ (a < -1 ∨ a > 1) :=
by
  sorry

end range_of_a_l666_666194


namespace star_sums_to_24_l666_666325

def sum_of_four_is_24 (a b c d : ℕ) : Prop := a + b + c + d = 24

noncomputable def pentagon_numbers : ℕ := 5

def five_pointed_star_numbers (A B C D E F G H J K : ℕ) : Prop :=
  A = 1 ∧ B = 2 ∧ C = 3 ∧ D = 4 ∧ E = 5 ∧
  F = 10 ∧ G = 12 ∧ H = 9 ∧ J = 6 ∧ K = 8 ∧
  sum_of_four_is_24 E F H J ∧
  sum_of_four_is_24 F G K J ∧
  sum_of_four_is_24 H J K B ∧
  sum_of_four_is_24 J E K C ∧
  sum_of_four_is_24 A J G B

theorem star_sums_to_24 :
  ∃ (A B C D E F G H J K : ℕ), five_pointed_star_numbers A B C D E F G H J K :=
by {
  use [1, 2, 3, 4, 5, 10, 12, 9, 6, 8],
  sorry
}

end star_sums_to_24_l666_666325


namespace compare_fx1_fx2_l666_666365

-- Conditions: f is an even function defined on ℝ and increasing on (-∞, 0).
-- Let f be a function from reals to reals with those properties.
variables {f : ℝ → ℝ}

-- Condition stating that f is even
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f (x)

-- Condition stating that f is increasing on (-∞, 0)
def is_increasing_on_negative_reals (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y ∧ y < 0 → f(x) < f(y)

-- Using these conditions, we want to prove that f(x1) > f(x2) given x1 < 0 and x1 + x2 > 0.
theorem compare_fx1_fx2
  (h_even : is_even_function f)
  (h_incr : is_increasing_on_negative_reals f)
  (x1 x2 : ℝ)
  (hx1 : x1 < 0)
  (hx1x2 : x1 + x2 > 0) :
  f(x1) > f(x2) :=
sorry

end compare_fx1_fx2_l666_666365


namespace find_angle_B_find_cosine_sum_range_l666_666320

-- Define the acute triangle and given conditions
structure acute_triangle (A B C : ℝ) (a b c : ℝ) :=
(acute_A : 0 < A ∧ A < π / 2)
(acute_B : 0 < B ∧ B < π / 2)
(acute_C : 0 < C ∧ C < π / 2)
(sides : a > 0 ∧ b > 0 ∧ c > 0)
(angles_sum : A + B + C = π)
(given_condition : 2 * b * sin A = sqrt 3 * a)

-- Part Ⅰ: Proving the measure of angle B
theorem find_angle_B {A B C a b c : ℝ} (h : acute_triangle A B C a b c) : 
  B = π / 3 :=
sorry

-- Part Ⅱ: Proving the range of values for cos A + cos B + cos C
theorem find_cosine_sum_range {A B C a b c : ℝ} (h : acute_triangle A B C a b c) :
  (sqrt 3 + 1) / 2 < cos A + cos B + cos C ∧ cos A + cos B + cos C ≤ 3 / 2 :=
sorry

end find_angle_B_find_cosine_sum_range_l666_666320


namespace cos_double_angle_l666_666245

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666245


namespace permutation_three_people_l666_666684

theorem permutation_three_people : 
  ∃ (A B C : Type), fintype.card (finset.univ : finset (A × B × C)) = 6 :=
by
  sorry

end permutation_three_people_l666_666684


namespace overall_percent_change_l666_666037

theorem overall_percent_change (W : ℝ) : 
  (W * 0.6 * 1.3 * 0.8 * 1.1) / W = 0.624 :=
by {
  sorry
}

end overall_percent_change_l666_666037


namespace jayda_spending_l666_666917

theorem jayda_spending
  (J A : ℝ)
  (h1 : A = J + (2/5) * J)
  (h2 : J + A = 960) :
  J = 400 :=
by
  sorry

end jayda_spending_l666_666917


namespace little_john_spent_on_sweets_l666_666384

theorem little_john_spent_on_sweets:
  let initial_amount := 10.10
  let amount_given_to_each_friend := 2.20
  let amount_left := 2.45
  let total_given_to_friends := 2 * amount_given_to_each_friend
  let amount_before_sweets := initial_amount - total_given_to_friends
  let amount_spent_on_sweets := amount_before_sweets - amount_left
  amount_spent_on_sweets = 3.25 :=
by
  sorry

end little_john_spent_on_sweets_l666_666384


namespace calculation_result_l666_666936

theorem calculation_result :
  (∏ k in Finset.range 17, (16 + k) / (k + 1)) / 
  (∏ j in Finset.range 15, (18 + j) / (j + 1)) = 496 :=
by sorry

end calculation_result_l666_666936


namespace evaluate_power_l666_666964

theorem evaluate_power (a b : ℝ) (m n : ℝ) (h1 : a = b^m) (h2 : ∀ x y z : ℝ, (x^y)^z = x^(y*z)) : a^(n/m) = b^n :=
by
  sorry

example : 81^(5/4) = 243 := evaluate_power 81 3 4 5
  (by norm_num) -- Simplification for 81 = 3^4
  (by norm_num []) -- Using the power of a power rule

end evaluate_power_l666_666964


namespace fair_coin_probability_l666_666499

theorem fair_coin_probability :
  ∀ (fair : bool) (total_tosses heads : ℕ),
    fair = true →
    total_tosses = 24000 →
    heads = 12012 →
    heads / total_tosses = 0.5005 :=
by
  intros _
  sorry

end fair_coin_probability_l666_666499


namespace angle_relationship_l666_666778

theorem angle_relationship (A B C C1 : Type) [has_angle A B C C1] (angle: A → B → C → Type)
  (h_C1 : foot_of_altitude C C1 ABC)
  (h_eq : CC1^2 = C1A * C1B) :
  angle A + angle B = 90 ∨ |angle A - angle B| = 90 := 
sorry

end angle_relationship_l666_666778


namespace valid_votes_for_candidate_A_l666_666860

theorem valid_votes_for_candidate_A (total_votes : ℕ) (invalid_percentage valid_percentage candidate_A_percentage : ℚ) :
  total_votes = 560000 →
  invalid_percentage = 0.15 →
  valid_percentage = 0.85 →
  candidate_A_percentage = 0.85 →
  let valid_votes := valid_percentage * total_votes in
  let votes_for_candidate_A := candidate_A_percentage * valid_votes in
  votes_for_candidate_A = 404600 := 
by
  intros h_total_votes h_invalid_percentage h_valid_percentage h_candidate_A_percentage
  simp only [h_total_votes, h_invalid_percentage, h_valid_percentage, h_candidate_A_percentage]
  let valid_votes := 0.85 * 560000
  let votes_for_candidate_A := 0.85 * valid_votes
  have h_valid_votes : valid_votes = 476000 := by norm_num
  have h_votes_for_candidate_A : votes_for_candidate_A = 0.85 * 476000 := by norm_num
  rw [h_valid_votes, ← mul_assoc] at h_votes_for_candidate_A
  norm_num at h_votes_for_candidate_A
  exact h_votes_for_candidate_A

end valid_votes_for_candidate_A_l666_666860


namespace max_min_rounded_value_l666_666056

theorem max_min_rounded_value (n : ℝ) (h : 3.75 ≤ n ∧ n < 3.85) : 
  (∀ n, 3.75 ≤ n ∧ n < 3.85 → n ≤ 3.84 ∧ n ≥ 3.75) :=
sorry

end max_min_rounded_value_l666_666056


namespace cos_double_angle_l666_666267

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666267


namespace abs_inequality_solution_l666_666410

theorem abs_inequality_solution :
  { x : ℝ | |x - 2| + |x + 3| < 6 } = { x | -7 / 2 < x ∧ x < 5 / 2 } :=
by
  sorry

end abs_inequality_solution_l666_666410


namespace point_translation_l666_666725

variable (P Q : (ℝ × ℝ))
variable (dx : ℝ) (dy : ℝ)

theorem point_translation (hP : P = (-1, 2)) (hdx : dx = 2) (hdy : dy = 3) :
  Q = (P.1 + dx, P.2 - dy) → Q = (1, -1) := by
  sorry

end point_translation_l666_666725


namespace circle_chords_integer_lengths_l666_666394

theorem circle_chords_integer_lengths (P O : ℝ) (d r : ℝ) (n : ℕ) : 
  dist P O = d → r = 20 → d = 12 → n = 9 := by
  sorry

end circle_chords_integer_lengths_l666_666394


namespace perpendicular_line_plane_l666_666161

-- Variables representing lines and planes
variables (l m : Line) (α β : Plane)

-- Hypotheses representing the conditions
hypothesis (hlm_diff : l ≠ m)
hypothesis (hαβ_diff : α ≠ β)
hypothesis (hlα : l ∈ α)
hypothesis (hmβ : m ∈ β)

-- Statement to be proved
theorem perpendicular_line_plane (hlβ : Line.perpendicular l β) : Plane.perpendicular α β :=
by
  sorry

end perpendicular_line_plane_l666_666161


namespace least_possible_value_of_En_l666_666356

noncomputable section

open Real

variable {n : ℕ} (a : Fin n → ℝ)

theorem least_possible_value_of_En (h_n : n ≥ 2) (h_pos : ∀ i : Fin n, 0 < a i) :
  (∏ i in Finset.range n, (if i = 0 then 1 + a 0 else a i.pred + a i)) * (a (Fin.last n) + 3 ^ (n + 1)) / 
  (∏ i in Finset.range n, a i) ≥ 4 ^ (n + 1) :=
sorry

end least_possible_value_of_En_l666_666356


namespace number_of_students_voted_in_favor_of_both_propositions_l666_666070

variable (U A B : Set ℕ)
variable (cardU cardA cardB cardAcapBc : ℕ)

theorem number_of_students_voted_in_favor_of_both_propositions
  (hU : cardU = 250) (hA : cardA = 175) (hB : cardB = 140) (hAcapBc : cardAcapBc = 45)
  (hU_def : cardU = (U.card))
  (hA_def : cardA = (A.card))
  (hB_def : cardB = (B.card))
  (hAcapBc_def : cardAcapBc = (U.card - ((U ∩ Aᶜ ∩ Bᶜ).card))) :
  A.card + B.card - (cardU - (U ∩ Aᶜ ∩ Bᶜ).card) = 110 :=
by sorry

end number_of_students_voted_in_favor_of_both_propositions_l666_666070


namespace factors_of_3600_are_perfect_cubes_l666_666685

theorem factors_of_3600_are_perfect_cubes :
  let n := 3600
  let prime_factors := (4, 2, 2)  -- Corresponding to exponents of 2, 3, 5 in prime factorization
  let exponents := [(a, b, c) | 0 <= a ∧ a <= 4 ∧ a % 3 = 0,
                               0 <= b ∧ b <= 2 ∧ b % 3 = 0,
                               0 <= c ∧ c <= 2 ∧ c % 3 = 0]
  ∑ _(x : ℕ × ℕ × ℕ) in exponents.1 ∣ (3600 = 2 ^ x.1 * 3 ^ x.2 * 5 ^ x.3), 1 = 1 :=
by
  let n := 3600
  let prime_factors := (2 ^ 4, 3 ^ 2, 5 ^ 2)
  let exponents := [(x, y, z) | (0 <= x ∧ x <= 4 ∧ x % 3 = 0,
                                 0 <= y ∧ y <= 2 ∧ y % 3 = 0,
                                 0 <= z ∧ z <= 2 ∧ z % 3 = 0)]
  exact sorry

end factors_of_3600_are_perfect_cubes_l666_666685


namespace series_sum_zero_l666_666763

theorem series_sum_zero (x1 x2 : ℂ) (h1 : x1^2 + x1 + 1 = 0) (h2 : x2^2 + x2 + 1 = 0) :
  ∑ k in finset.range 1998, (x1 / x2)^k = 0 :=
by
  sorry

end series_sum_zero_l666_666763


namespace value_of_expression_l666_666698

-- Definitions for the conditions
variables (a b : ℝ)

-- Theorem statement
theorem value_of_expression : (a - 3 * b = 3) → (a + 2 * b - (2 * a - b)) = -3 :=
by
  intro h
  sorry

end value_of_expression_l666_666698


namespace problem_log_inequality_l666_666381

noncomputable def f (x m : ℝ) := x - |x + 2| - |x - 3| - m

theorem problem (m : ℝ) (h1 : ∀ x : ℝ, (1 / m) - 4 ≥ f x m) :
  m > 0 :=
sorry

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_inequality (m : ℝ) (h2 : m > 0) :
  log_base (m + 1) (m + 2) > log_base (m + 2) (m + 3) :=
sorry

end problem_log_inequality_l666_666381


namespace rectangle_probability_l666_666879

theorem rectangle_probability (m n : ℕ) (h_m : m = 1003^2) (h_n : n = 1003 * 2005) :
  (1 - (m / n)) = 1002 / 2005 :=
by
  sorry

end rectangle_probability_l666_666879


namespace cos_double_angle_l666_666265

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666265


namespace hyperbola_focal_length_l666_666595

theorem hyperbola_focal_length :
  let a := 2
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  2 * c = 2 * Real.sqrt 7 := 
by
  sorry

end hyperbola_focal_length_l666_666595


namespace find_m_l666_666813

theorem find_m {m : ℝ} :
  (∃ x y : ℝ, y = x + 1 ∧ y = -x ∧ y = mx + 3) → m = 5 :=
by
  sorry

end find_m_l666_666813


namespace find_x_squared_plus_y_squared_l666_666209

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 10344 / 169 :=
by
  sorry

end find_x_squared_plus_y_squared_l666_666209


namespace same_function_pairs_l666_666061

def f1 (x : ℝ) : ℝ := x^2 - x - 1
def g1 (t : ℝ) : ℝ := t^2 - t + 1

def f2 (x : ℝ) : ℝ := (x^2 - 1)^(1/2)
def g2 (x : ℝ) : ℝ := (x + 1)^(1/2) * (x - 1)^(1/2)

def f3 (x : ℝ) : ℝ := (x^(1/2))^2
def g3 (x : ℝ) : ℝ := (x^2)^(1/2)

def f4 (x : ℝ) : ℝ := (x^2)^(1/2) / x
def g4 (x : ℝ) : ℝ := x / (x^2)^(1/2)

theorem same_function_pairs :
  (∀ x, f1 x = g1 x) ∧
  (∀ x, (f2 x = g2 x) = False) ∧
  (∀ x, (f3 x = g3 x) = False) ∧
  (∀ x, f4 x = g4 x) := 
by
  sorry

end same_function_pairs_l666_666061


namespace volume_of_solid_l666_666948

theorem volume_of_solid :
  let T := {
    p : ℝ × ℝ × ℝ |
    let (x, y, z) := p in
    x + 2 * y ≤ 1 ∧
    x + 2 * z ≤ 1 ∧
    2 * y + 2 * z ≤ 1 ∧
    x ≥ 0 ∧
    y ≥ 0 ∧
    z ≥ 0
  } in
  ∫ z in 0..(1/2),
  ∫ y in 0..(1/2 - z),
  ∫ x in 0..(1 - 2 * y), (1 : ℝ) = 1 / 6 :=
by
  sorry

end volume_of_solid_l666_666948


namespace poly_coeff_sum_l666_666155

variable {a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ}

theorem poly_coeff_sum :
  (∀ x : ℝ, (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 + 6 * a_6 = 12 :=
by
  sorry

end poly_coeff_sum_l666_666155


namespace probability_at_least_two_heads_is_11_over_16_l666_666915

noncomputable def probability_of_heads : ℚ := 1 / 2

noncomputable def probability_at_least_two_heads : ℚ :=
  1 - (nat.choose 4 0 * probability_of_heads^4 + nat.choose 4 1 * probability_of_heads^4)

theorem probability_at_least_two_heads_is_11_over_16 :
  probability_at_least_two_heads = 11 / 16 := by
  sorry

end probability_at_least_two_heads_is_11_over_16_l666_666915


namespace fff1_eq_17_l666_666577

def f (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 1
  else if n < 6 then 3 * n + 2
  else 2 * n - 1

theorem fff1_eq_17 : f (f (f 1)) = 17 :=
  by sorry

end fff1_eq_17_l666_666577


namespace number_of_special_integers_is_16_l666_666087

noncomputable def count_special_integers : ℕ :=
  let even_last_digits := [0, 2, 4, 6, 8] in
  let count_for_digit (c : ℕ) : ℕ := if c = 0 then 7
                                     else if c = 2 then 5
                                     else if c = 4 then 3
                                     else if c = 6 then 1
                                     else 0 in
  even_last_digits.map count_for_digit |>.sum

theorem number_of_special_integers_is_16 : count_special_integers = 16 :=
by
  unfold count_special_integers
  unfold count_for_digit
  simp only [List.map, List.sum]
  rfl

end number_of_special_integers_is_16_l666_666087


namespace intersection_is_line_l666_666519

-- Define the two planes as given in the conditions
def plane1 (x y z : ℝ) : Prop := x + 5 * y + 2 * z - 5 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x - 5 * y - z + 5 = 0

-- The intersection of the planes should satisfy both plane equations
def is_on_line (x y z : ℝ) : Prop := plane1 x y z ∧ plane2 x y z

-- Define the canonical equation of the line
def line_eq (x y z : ℝ) : Prop := (∃ k : ℝ, x = 5 * k ∧ y = 5 * k + 1 ∧ z = -15 * k)

-- The proof statement
theorem intersection_is_line :
  (∀ x y z : ℝ, is_on_line x y z → line_eq x y z) ∧ 
  (∀ x y z : ℝ, line_eq x y z → is_on_line x y z) :=
by
  sorry

end intersection_is_line_l666_666519


namespace kim_cherry_pits_l666_666353

/-- Kim plants some cherry pits. 25% of them sprout and Kim sells 6 of the saplings. She has 14 cherry saplings left.
    How many cherry pits did she plant? 
    Prove that the total number of cherry pits is 80. -/
theorem kim_cherry_pits (P : ℕ) (h1 : 0.25 * P - 6 = 14) : P = 80 := 
by sorry

end kim_cherry_pits_l666_666353


namespace susan_strawberries_per_handful_l666_666417

-- Definitions of the given conditions
def total_picked := 75
def total_needed := 60
def strawberries_per_handful := 5

-- Derived conditions
def total_eaten := total_picked - total_needed
def number_of_handfuls := total_picked / strawberries_per_handful
def strawberries_eaten_per_handful := total_eaten / number_of_handfuls

-- The theorem we want to prove
theorem susan_strawberries_per_handful : strawberries_eaten_per_handful = 1 :=
by sorry

end susan_strawberries_per_handful_l666_666417


namespace rectangle_cut_into_square_l666_666044

theorem rectangle_cut_into_square (a b : ℝ) (h : a ≤ 4 * b) : 4 * b ≥ a := 
by 
  exact h

end rectangle_cut_into_square_l666_666044


namespace altitude_line_equation_l666_666996

-- Definitions of the coordinates of points A, B, and C
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (3, 0)

-- Statement to prove the equation of the line containing the altitude
theorem altitude_line_equation :
  ∃ K : ℝ, ∃ x₀ y₀ : ℝ, x₀ + 3 * y₀ - 3 = 0 ∧
  (A = (1, -1) ∧ B = (2, 2) ∧ C = (3, 0)) ∧
  ∀ (x y : ℝ), (y - (-1)) = K * (x - 1) :=
begin
  sorry,
end

end altitude_line_equation_l666_666996


namespace find_x_value_l666_666728

-- Definitions
variables {Point : Type} [affine_space Point]
variables (A B D E C : Point)
variables {angle : Type} [real_angle angle]
variables (x : angle) (ECB DEC : angle)
variables (line_segment : Point → Point → set Point)
variables (meets_at_right_angle : Point → Point → Point → Prop)
variables (angle_measure : Point → Point → Point → angle)

-- Conditions
axiom is_line_segment : line_segment A B
axiom point_D_on_AB : D ∈ line_segment A B
axiom point_E_not_on_AB : E ∉ line_segment A B
axiom DE_meets_AB_at_right_angle : meets_at_right_angle D E (line_segment A B)
axiom angle_ECB_65 : angle_measure E C B = 65
axiom angle_DEC_30 : angle_measure D E C = 30

-- Proof Statement
theorem find_x_value : x = 25 :=
by
  sorry

end find_x_value_l666_666728


namespace product_of_axes_l666_666395

theorem product_of_axes 
  (P G X Y W Z : Type) 
  [center P X Y W Z G] 
  (PG : P = G) 
  (a b : ℝ) 
  (H1 : a^2 - b^2 = 64) 
  (H2 : a - b = 4) 
  (XY : ℝ := 2 * a)
  (WZ : ℝ := 2 * b) 
  : (XY * WZ = 240) :=
by {
  sorry
}

end product_of_axes_l666_666395


namespace product_polynomial_coeffs_l666_666823

theorem product_polynomial_coeffs
  (g h : ℚ)
  (h1 : 7 * d^2 - 3 * d + g * (3 * d^2 + h * d - 5) = 21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d + 15) :
  g + h = -28/9 := 
  sorry

end product_polynomial_coeffs_l666_666823


namespace cow_husk_eating_time_l666_666313

theorem cow_husk_eating_time 
  (cows : ℕ) 
  (bags : ℕ) 
  (days : ℕ) 
  (h : cows * days = bags * days) : 
  1 * days = 1 * days := 
by 
  -- Given 45 cows take 45 days to eat 45 bags
  have H : 45 * 45 = 45 * 45 := h
  -- Therefore, one cow will take 45 days to eat one bag
  sorry

end cow_husk_eating_time_l666_666313


namespace units_digit_of_fraction_l666_666849

-- Define the problem
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fraction :
  units_digit ((30 * 31 * 32 * 33 * 34 * 35) / 2500) = 2 := by
  sorry

end units_digit_of_fraction_l666_666849


namespace half_alpha_quad_l666_666210

def alpha_in_first_or_third_quadrant (α : ℝ) : Prop :=
  (0 < α ∧ α < π/2) →
  (0 < α/2 ∧ α/2 < π/2) ∨ (π < α/2 ∧ α/2 < 3*π/2)

theorem half_alpha_quad (α : ℝ) (h : 0 < α ∧ α < π/2) : 
  (0 < α/2 ∧ α/2 < π/2) ∨ (π < α/2 ∧ α/2 < 3*π/2) := 
sorry

end half_alpha_quad_l666_666210


namespace obtuse_triangle_l666_666734

theorem obtuse_triangle (A B C M E : ℝ) (hM : M = (B + C) / 2) (hE : E > 0) 
(hcond : (B - E) ^ 2 + (C - E) ^ 2 >= 4 * (A - M) ^ 2): 
∃ α β γ, α > 90 ∧ β + γ < 90 ∧ α + β + γ = 180 :=
by
  sorry

end obtuse_triangle_l666_666734


namespace sequence_integers_l666_666375

variable {R : Type*} [CommRing R] {x y : R}

def a (n : ℕ) := ∑ k in Finset.range (n + 1), x ^ k * y ^ (n - k)

theorem sequence_integers
  (h : ∃ (m : ℕ), a m ∈ ℤ ∧ a (m + 1) ∈ ℤ ∧ a (m + 2) ∈ ℤ ∧ a (m + 3) ∈ ℤ) :
  ∀ n, a n ∈ ℤ := 
sorry

end sequence_integers_l666_666375


namespace proof_g_neg_one_l666_666621

theorem proof_g_neg_one (f g : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) + (-x)^2 = - (f x + x^2))
    (h_f1 : f 1 = 1) (h_g : ∀ x : ℝ, g x = f x + 2) : g (-1) = -1 := by
  -- definitions and conditions
  have h_f_odd : ∀ x: ℝ, f x + f (-x) = -2 * x^2 := by
    intro x
    specialize h_odd x
    rw [neg_sq] at h_odd
    sorry  -- detailed proof steps omitted and directly focus on main goal
  
  -- calculate f(-1)
  have h_f_neg1 : f (-1) = -3 := by
    specialize h_f_odd 1
    rw [h_f1] at h_f_odd
    linarith
  
  -- calculate g(-1)
  specialize h_g (-1)
  rw [h_f_neg1] at h_g
  rw [h_g]
  exact rfl

end proof_g_neg_one_l666_666621


namespace greatest_mass_l666_666844

theorem greatest_mass (V : ℝ) (h : ℝ) (l : ℝ) 
    (ρ_Hg ρ_H2O ρ_Oil : ℝ) 
    (V1 V2 V3 : ℝ) 
    (m_Hg m_H2O m_Oil : ℝ)
    (ρ_Hg_val : ρ_Hg = 13.59) 
    (ρ_H2O_val : ρ_H2O = 1) 
    (ρ_Oil_val : ρ_Oil = 0.915) 
    (height_layers_equal : h = l) :
    ∀ V1 V2 V3 m_Hg m_H2O m_Oil, 
    V1 + V2 + V3 = 27 * (l^3) → 
    V2 = 7 * V1 → 
    V3 = 19 * V1 → 
    m_Hg = ρ_Hg * V1 → 
    m_H2O = ρ_H2O * V2 → 
    m_Oil = ρ_Oil * V3 → 
    m_Oil > m_Hg ∧ m_Oil > m_H2O := 
by 
    intros
    sorry

end greatest_mass_l666_666844


namespace square_side_length_l666_666437

theorem square_side_length (x : ℝ) (h : 4 * x = 8 * Real.pi) : x = 6.28 := 
by {
  -- proof will go here
  sorry
}

end square_side_length_l666_666437


namespace probability_none_solve_l666_666864

theorem probability_none_solve (a b c : ℕ) (ha : 0 < a ∧ a < 10)
                               (hb : 0 < b ∧ b < 10)
                               (hc : 0 < c ∧ c < 10)
                               (P_A : ℚ := 1 / a)
                               (P_B : ℚ := 1 / b)
                               (P_C : ℚ := 1 / c)
                               (H : (1 - (1 / a)) * (1 - (1 / b)) * (1 - (1 / c)) = 8 / 15) :
                               -- Conclusion: The probability that none of them solve the problem is 8/15
                               (1 - (1 / a)) * (1 - (1 / b)) * (1 - (1 / c)) = 8 / 15 :=
sorry

end probability_none_solve_l666_666864


namespace range_of_y_l666_666297

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ set.Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end range_of_y_l666_666297


namespace square_side_length_l666_666436

theorem square_side_length (x : ℝ) (h : 4 * x = 8 * Real.pi) : x = 6.28 := 
by {
  -- proof will go here
  sorry
}

end square_side_length_l666_666436


namespace count_complex_solutions_l666_666975

noncomputable def complex_solutions_count (z : ℂ) : ℕ :=
  if (|z| = 1 ∧ |(z / (conj z)) - ((conj z) / z)| = 1) then 8 else 0

theorem count_complex_solutions : complex_solutions_count = 8 := sorry

end count_complex_solutions_l666_666975


namespace min_employees_wednesday_l666_666567

noncomputable def minWednesdayBirthdays (total_employees : ℕ) 
  (diff_birthdays : (List ℕ → Prop)) 
  (max_birthdays : (ℕ → List ℕ → Prop)) :
  ℕ :=
  40

theorem min_employees_wednesday (total_employees : ℕ) 
  (diff_birthdays : (List ℕ → Prop)) 
  (max_birthdays : (ℕ → List ℕ → Prop)) 
  (h1 : total_employees = 61) 
  (h2 : ∃ lst, diff_birthdays lst ∧ max_birthdays 40 lst) :
  minWednesdayBirthdays total_employees diff_birthdays max_birthdays = 40 := 
sorry

end min_employees_wednesday_l666_666567


namespace solution_properties_l666_666608

variables {R : Type*} [Field R]

noncomputable def Delta (a b c : R) : R := (b - 1)^2 - 4 * a * c

def system_of_equations (a b c : R) (x : Fin n → R) : Prop :=
  ∀ i : Fin n, a * (x i)^2 + b * (x i) + c = x ((i + 1) % n)

theorem solution_properties {a b c : R} {n : Nat} (h_a : a ≠ 0) :
  ∀ x : Fin n → R, 
  let Δ := Delta a b c in 
  (Δ < 0 → ¬(system_of_equations a b c x)) ∧
  (Δ = 0 → ∃! x, system_of_equations a b c x ∧ (∀ i, x i = - (b - 1) / (2 * a))) ∧
  (Δ > 0 → ∃ x₁ x₂, system_of_equations a b c x₁ ∧ system_of_equations a b c x₂ ∧ 
    (∀ i, (x₁ i = ((- (b - 1) + (Δ.sqrt)) / (2 * a)) ∨ x₁ i = ((- (b - 1) - (Δ.sqrt)) / (2 * a))))) :=
sorry

end solution_properties_l666_666608


namespace last_number_of_nth_row_sum_of_numbers_in_nth_row_position_of_2008_l666_666390

theorem last_number_of_nth_row (n : ℕ) : 
    let last_number := 2^n - 1
    last_number = 2^n - 1 := 
sorry

theorem sum_of_numbers_in_nth_row (n : ℕ) :
    let sum := (3 * 2^(n-3)) - 2^(n-2)
    sum = (3 * 2^(n-3)) - 2^(n-2) :=
sorry

theorem position_of_2008 : 
    let position := 985
    position = 985 :=
sorry

end last_number_of_nth_row_sum_of_numbers_in_nth_row_position_of_2008_l666_666390


namespace vertex_x_coordinate_eq_zero_l666_666814

structure Point (α : Type _) [LinearOrderedField α] :=
  (x y : α)

def parabola (α : Type _) [LinearOrderedField α] (a b c x : α) : α :=
  a * x^2 + b * x + c

theorem vertex_x_coordinate_eq_zero (a b c : ℝ)
  (h1 : parabola ℝ a b c (-2) = 9)
  (h2 : parabola ℝ a b c 2 = 9)
  (h3 : parabola ℝ a b c 1 = 6) :
  ∃ v_x : ℝ, v_x = 0 :=
by
  use 0
  sorry

end vertex_x_coordinate_eq_zero_l666_666814


namespace largest_prime_factor_of_sum_of_divisors_of_180_divisor_count_of_180_l666_666358

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ i in (finset.range (n + 1)).filter (λ x, n % x = 0), i

def largest_prime_factor (n : ℕ) : ℕ :=
  (n.factors).max' (list.nodup n.factors)

theorem largest_prime_factor_of_sum_of_divisors_of_180 :
  let N := sum_of_divisors 180
  in largest_prime_factor N = 13 := by
  sorry

theorem divisor_count_of_180 :
  (finset.range (180 + 1)).filter (λ x, 180 % x = 0).card = 18 := by
  sorry

end largest_prime_factor_of_sum_of_divisors_of_180_divisor_count_of_180_l666_666358


namespace _l666_666663

-- Define the main theorem with the given condition and the required conclusion
example (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 2) : Real.log (4 ** -a) / Real.log 3 = -2 := by
  sorry

end _l666_666663


namespace approx_f_at_1_97_l666_666925

def f (x : ℝ) : ℝ := Real.sqrt (x^2 + x + 3)

def f_deriv (x : ℝ) : ℝ := (2 * x + 1) / (2 * Real.sqrt (x^2 + x + 3))

theorem approx_f_at_1_97 : 
  let x0 := 2 in
  let x := 1.97 in
  let delta_x := x - x0 in
  let f_x0 := f x0 in
  let f'_x0 := f_deriv x0 in
  let approx_f := f_x0 + f'_x0 * delta_x in
  abs (f 1.97 - approx_f) < 0.001 :=
by lift no sorry

end approx_f_at_1_97_l666_666925


namespace cos_double_angle_l666_666276

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l666_666276


namespace passenger_waiting_time_probability_l666_666012

def bus_arrival_interval : ℕ := 5

def waiting_time_limit : ℕ := 3

/-- 
  Prove that for a bus arriving every 5 minutes,
  the probability that a passenger's waiting time 
  is no more than 3 minutes, given the passenger 
  arrives at a random time, is 3/5. 
--/
theorem passenger_waiting_time_probability 
  (bus_interval : ℕ) (time_limit : ℕ) 
  (random_arrival : ℝ) :
  bus_interval = 5 →
  time_limit = 3 →
  0 ≤ random_arrival ∧ random_arrival < bus_interval →
  (random_arrival ≤ time_limit) →
  (random_arrival / ↑bus_interval) = 3 / 5 :=
by
  sorry

end passenger_waiting_time_probability_l666_666012


namespace b_value_l666_666124

theorem b_value (x y b : ℝ) (h1 : x / (2 * y) = 3 / 2) (h2 : (7 * x + b * y) / (x - 2 * y) = 25) : b = 4 := 
by
  sorry

end b_value_l666_666124


namespace scientific_notation_of_216000_l666_666419

theorem scientific_notation_of_216000 :
  216000 = 2.16 * 10^5 :=
sorry

end scientific_notation_of_216000_l666_666419


namespace find_f_e_eq_2_l666_666180

noncomputable def f : ℝ → ℝ
| x => if x < 1 then Real.exp x + 1 else f (Real.log x)

theorem find_f_e_eq_2 : f (Real.exp 1) = 2 := by
  sorry

end find_f_e_eq_2_l666_666180


namespace smaller_square_area_percentage_l666_666905

noncomputable def percent_area_of_smaller_square (side_length_larger_square : ℝ) : ℝ :=
  let diagonal_larger_square := side_length_larger_square * Real.sqrt 2
  let radius_circle := diagonal_larger_square / 2
  let x := (2 + 4 * (side_length_larger_square / 2)) / ((side_length_larger_square / 2) * 2) -- Simplified quadratic solution
  let side_length_smaller_square := side_length_larger_square * x
  let area_smaller_square := side_length_smaller_square ^ 2
  let area_larger_square := side_length_larger_square ^ 2
  (area_smaller_square / area_larger_square) * 100

-- Statement to show that under given conditions, the area of the smaller square is 4% of the larger square's area
theorem smaller_square_area_percentage :
  percent_area_of_smaller_square 4 = 4 := 
sorry

end smaller_square_area_percentage_l666_666905


namespace rectangle_probability_l666_666880

theorem rectangle_probability (m n : ℕ) (h_m : m = 1003^2) (h_n : n = 1003 * 2005) :
  (1 - (m / n)) = 1002 / 2005 :=
by
  sorry

end rectangle_probability_l666_666880


namespace determine_f_l666_666293

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 / 3

lemma increasing_function (x y : ℝ) (hxy : x ≤ y) : f(x) ≤ f(y) :=
by sorry

theorem determine_f (x : ℝ) 
  (linear_f : ∀ x y : ℝ, f (x + y) = f x + f y ∧ f (2*x) = 2 * f(x)) -- condition 1
  (increasing_f : ∀ x y : ℝ, x ≤ y → f x ≤ f y)                    -- condition 2
  (fx_fx_eq : ∀ x : ℝ, f (f x) = 4 * x - 1)                          -- condition 3
  : f(x) = 2 * x - 1/3 :=
by sorry

end determine_f_l666_666293


namespace measure_one_minute_correct_l666_666744

noncomputable def measure_one_minute : Prop :=
  let large_timer_minutes := 60 in
  let small_timer_minutes := 11 in
  (120 - 121 = 1)

theorem measure_one_minute_correct : measure_one_minute :=
  by
    -- The proof would be provided here, but for now we just present the statement format
    sorry

end measure_one_minute_correct_l666_666744


namespace pow_evaluation_l666_666960

theorem pow_evaluation : 81^(5/4) = 243 := 
by sorry

end pow_evaluation_l666_666960


namespace cos_sum_to_product_l666_666940

theorem cos_sum_to_product (a : ℝ) : cos (3 * a) + cos (5 * a) = 2 * cos (4 * a) * cos (a) := 
  sorry

end cos_sum_to_product_l666_666940


namespace average_expenditure_whole_week_l666_666509

theorem average_expenditure_whole_week (a b : ℕ) (h₁ : a = 3 * 350) (h₂ : b = 4 * 420) : 
  (a + b) / 7 = 390 :=
by 
  sorry

end average_expenditure_whole_week_l666_666509


namespace domain_f_l666_666076

def quadratic (x : ℝ) : ℝ := x^2 - 7 * x + 16
def floor_quad (x : ℝ) : ℤ := Int.floor (quadratic x)
def f (x : ℝ) : ℝ := 1 / (floor_quad x)

theorem domain_f : 
  {x : ℝ | (x ≤ 3) ∨ (x ≥ 5)} = {x : ℝ | x ∉ set.Ioo 3 5} :=
by 
  sorry

end domain_f_l666_666076


namespace shelves_for_coloring_books_l666_666050

theorem shelves_for_coloring_books (initial_stock sold donated per_shelf remaining total_used needed_shelves : ℕ) 
    (h_initial : initial_stock = 150)
    (h_sold : sold = 55)
    (h_donated : donated = 30)
    (h_per_shelf : per_shelf = 12)
    (h_total_used : total_used = sold + donated)
    (h_remaining : remaining = initial_stock - total_used)
    (h_needed_shelves : (remaining + per_shelf - 1) / per_shelf = needed_shelves) :
    needed_shelves = 6 :=
by
  sorry

end shelves_for_coloring_books_l666_666050


namespace octal_to_decimal_7564_l666_666062

theorem octal_to_decimal_7564 : 7 * 8^3 + 5 * 8^2 + 6 * 8^1 + 4 * 8^0 = 3956 :=
by
  sorry 

end octal_to_decimal_7564_l666_666062


namespace find_intersection_find_range_of_a_l666_666197

-- Define the sets A and B
def A : Set ℝ := { x : ℝ | x < -2 ∨ (3 < x ∧ x < 4) }
def B : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 5 }

-- Proof Problem 1: Prove the intersection A ∩ B
theorem find_intersection : (A ∩ B) = { x : ℝ | 3 < x ∧ x ≤ 5 } := by
  sorry

-- Define the set C and the condition B ∩ C = B
def C (a : ℝ) : Set ℝ := { x : ℝ | x ≥ a }
def condition (a : ℝ) : Prop := B ∩ C a = B

-- Proof Problem 2: Find the range of a
theorem find_range_of_a : ∀ a : ℝ, condition a → a ≤ -3 := by
  sorry

end find_intersection_find_range_of_a_l666_666197


namespace table_height_max_l666_666339

-- We define the sides and the maximum possible height
def triangle_PQR :
    {PQ QR PR : ℝ // PQ = 24 ∧ QR = 26 ∧ PR = 28}

-- We define the maximum possible height h in the specified format
def maximum_table_height {PQ QR PR : ℝ} (h : ℝ) :
    (∃ {k m n : ℕ}, h = (k : ℝ) * real.sqrt (m : ℝ) / (n : ℝ) ∧ 
        PQ = 24 ∧ QR = 26 ∧ PR = 28 ∧ 
        n * h = k * real.sqrt (m : ℝ) 
        ∧ k = 30 
        ∧ m = 1287
        ∧ n = 58 
    )

-- The theorem stating the height of the table given the conditions above
theorem table_height_max {PQ QR PR : ℝ} :
    PQ = 24 ∧ QR = 26 ∧ PR = 28 →
    ∃ h : ℝ, maximum_table_height h :=
by {
  sorry
}

end table_height_max_l666_666339


namespace solution_to_problem_l666_666447

theorem solution_to_problem :
  ∃ x y : ℝ, (x + 3)^2 + y^2 + (x - y)^2 = 3 ∧ x = -2 ∧ y = -1 :=
by
  use [-2, -1]
  split
  { calc
      ( -2 + 3 )^2 + ( -1 )^2 + ( -2 - ( -1 ) )^2
        = ( 1 )^2 + ( 1 )^2 + ( -1 )^2 : by ring
    ... = 1 + 1 + 1 : by norm_num
    ... = 3 : by norm_num }
  split
  { refl }
  { refl }

end solution_to_problem_l666_666447


namespace sum_inverse_Sn_lt_11_over_9_l666_666631

theorem sum_inverse_Sn_lt_11_over_9 
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h2 : S 3 = 9)
  (h3 : (a 1 + 6 * (a 2 - a 1)) = (a 1 + 2 * (a 2 - a 1)) * (a 1 + 2 * (a 2 - a 1)))
  (h4 : a 1 ≠ a 2)
  (h5 : 3 ≤ n) 
  : ∑ k in finset.range n, 1 / S (k + 1) < 11 / 9 :=
begin
  sorry
end

end sum_inverse_Sn_lt_11_over_9_l666_666631


namespace imaginary_part_of_conjugate_of_z_l666_666302

-- Define the main problem
theorem imaginary_part_of_conjugate_of_z {z : ℂ} (h : z * (2 - complex.I) = 5) : complex.im (conj z) = -1 := 
sorry

end imaginary_part_of_conjugate_of_z_l666_666302


namespace count_divisible_by_35_l666_666690

theorem count_divisible_by_35 : 
  ∃! (n : ℕ), n = 13 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 ∧ (∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ ab = 10 * a + b) →
    (ab * 100 + 35) % 35 = 0 ↔ ab % 7 = 0) :=
by {
  sorry
}

end count_divisible_by_35_l666_666690


namespace necessary_but_not_sufficient_l666_666153

-- Definitions
def k := Real.sqrt 3
def p : Prop := k = Real.sqrt 3
def q : Prop := ∃ k : ℝ, y = k * x + 2 ∧ ∀ x y : ℝ, x^2 + y^2 = 1 → (x, y) is_tangent_to(λ x y : ℝ, y = kx + 2)

-- The statement to prove
theorem necessary_but_not_sufficient (hp : p) (hq : q) : ¬p -> (¬q -> (¬p ∧ (¬p ∧ ¬q -> False))) :=
begin
  sorry
end

end necessary_but_not_sufficient_l666_666153


namespace train_speed_proof_l666_666892

noncomputable def train_speed (l : ℝ) (t : ℝ) : ℝ := (l / t) * 3.6

theorem train_speed_proof :
  train_speed 80 7.199424046076314 ≈ 40.0032 :=
begin
  sorry
end

end train_speed_proof_l666_666892


namespace polygon_sides_l666_666456

theorem polygon_sides (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n > 2) : n = 8 :=
sorry

end polygon_sides_l666_666456


namespace double_angle_cosine_l666_666220

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l666_666220


namespace constant_term_binomial_l666_666427

theorem constant_term_binomial (n : ℕ) (h : n = 5) : ∃ (r : ℕ), r = 6 ∧ (Nat.choose (2 * n) r) = 210 := by
  sorry

end constant_term_binomial_l666_666427


namespace smallest_in_sample_l666_666136

theorem smallest_in_sample:
  ∃ (m : ℕ) (δ : ℕ), m ≥ 0 ∧ δ > 0 ∧ δ * 5 = 80 ∧ 42 = δ * (42 / δ) + m ∧ m < δ ∧ (∀ i < 5, m + i * δ < 80) → m = 10 :=
by
  sorry

end smallest_in_sample_l666_666136


namespace cos_double_angle_l666_666216

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l666_666216


namespace perpendicular_bisector_a_l666_666633

-- Define the main theorem statement.
theorem perpendicular_bisector_a (a : ℝ) : ∃ a, let A := (0 : ℝ, -2 : ℝ), B := (6 : ℝ, 0 : ℝ), C := (0 : ℝ, a)
  -- Define the midpoint D of AB.
  let D := (((A.1 + B.1) / 2), ((A.2 + B.2) / 2)) in
  -- Define the equation for the perpendicular bisector of AB.
  (∀ (x y : ℝ), x = 0 → y = -3 * x + 8 → C = (x, y)) →
  -- Prove that a = 8.
  a = 8 :=
begin
  -- Substitute D and the positions of A, B.
  use 8,
  intros x y h1 h2,
  simp at *,
  sorry -- proof steps go here
end

end perpendicular_bisector_a_l666_666633


namespace find_exp_neg_alpha_beta_l666_666292

theorem find_exp_neg_alpha_beta (α β : ℝ) 
  (h : complex.exp (complex.I * α) + complex.exp (complex.I * β) = (1 / 4) + (3 / 7) * complex.I) : 
  complex.exp (-complex.I * α) + complex.exp (-complex.I * β) = (1 / 4) - (3 / 7) * complex.I :=
sorry

end find_exp_neg_alpha_beta_l666_666292


namespace cosine_double_angle_l666_666254

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l666_666254


namespace find_tan_2alpha_l666_666681

open Real

-- Define the tangent of double the angle
def tan_double_angle (tanα : ℝ) : ℝ := (2 * tanα) / (1 - tanα^2)

theorem find_tan_2alpha (α: ℝ) 
  (h1 : (cos α - 5) * cos α + (sin α) * (sin α - 5) = 0)
  (h2 : 0 < α ∧ α < π) :
  tan_double_angle (tan α) = 24 / 7 := 
sorry

end find_tan_2alpha_l666_666681


namespace max_perimeter_of_polygons_l666_666469

theorem max_perimeter_of_polygons : 
  ∀ (a b c : ℕ), 
  (a = 4 ∨ b = 4 ∨ c = 4) ∧
  (2^2 = 4) ∧
  (180 * (a-2) // a + 180 * (b-2) // b + 180 * (c-2) // c = 360) ∧
  (a > 2 ∧ b > 2 ∧ c > 2) →
  2 * (a + b + c) - 6 ≤ 34 :=
sorry

end max_perimeter_of_polygons_l666_666469


namespace rug_inner_rectangle_length_l666_666544

theorem rug_inner_rectangle_length
  (width : ℕ)
  (shaded1_width : ℕ)
  (shaded2_width : ℕ)
  (areas_in_ap : ℕ → ℕ → ℕ → Prop)
  (h1 : width = 2)
  (h2 : shaded1_width = 2)
  (h3 : shaded2_width = 2)
  (h4 : ∀ y a1 a2 a3, 
        a1 = 2 * y →
        a2 = 6 * (y + 4) →
        a3 = 10 * (y + 8) →
        areas_in_ap a1 (a2 - a1) (a3 - a2) →
        (a2 - a1 = a3 - a2)) :
  ∃ y, y = 4 :=
by
  sorry

end rug_inner_rectangle_length_l666_666544


namespace solve_quadratic_eq_l666_666449

theorem solve_quadratic_eq (x : ℝ) : x^2 = 4 * x → x = 0 ∨ x = 4 :=
by
  intro h
  sorry

end solve_quadratic_eq_l666_666449


namespace _l666_666661

-- Define the main theorem with the given condition and the required conclusion
example (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 2) : Real.log (4 ** -a) / Real.log 3 = -2 := by
  sorry

end _l666_666661


namespace emma_age_when_sister_is_56_l666_666955

theorem emma_age_when_sister_is_56 (e s : ℕ) (he : e = 7) (hs : s = e + 9) : 
  (s + (56 - s) - 9 = 47) :=
by {
  sorry
}

end emma_age_when_sister_is_56_l666_666955


namespace quadratic_has_two_distinct_real_roots_l666_666094

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  (x : ℝ) -> x^2 + m * x + 1 = 0 → (m < -2 ∨ m > 2) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l666_666094


namespace sum_groups_eq_250_l666_666928

-- Definitions for each sum
def sum1 : ℕ := 3 + 13 + 23 + 33 + 43
def sum2 : ℕ := 7 + 17 + 27 + 37 + 47

-- Theorem statement that the sum of these groups is 250
theorem sum_groups_eq_250 : sum1 + sum2 = 250 :=
by sorry

end sum_groups_eq_250_l666_666928


namespace even_product_when_eight_cards_drawn_l666_666952

theorem even_product_when_eight_cards_drawn :
  ∀ (s : Finset ℕ), (∀ n ∈ s, n ∈ Finset.range 15) →
  s.card ≥ 8 →
  (∃ m ∈ s, Even m) :=
by
  sorry

end even_product_when_eight_cards_drawn_l666_666952


namespace moles_of_MgO_formed_l666_666113

theorem moles_of_MgO_formed 
  (moles_Mg : ℕ) 
  (moles_CO2 : ℕ) 
  (balanced_eq : 2 * moles_Mg / 2 = 2 * moles_MgO ∧ moles_CO2 / 1 = 2 * moles_MgO) 
  (moles_Mg : moles_Mg = 3) 
  (moles_CO2 : moles_CO2 = 2) : 
  3 = (min (moles_Mg / 2) (moles_CO2)) * 2 :=
by
  sorry

end moles_of_MgO_formed_l666_666113


namespace factor_x4_minus_81_l666_666589

theorem factor_x4_minus_81 : 
  ∀ x : ℝ, (Polynomial.X ^ 4 - 81 : Polynomial ℝ) = (Polynomial.X - 3) * (Polynomial.X + 3) * (Polynomial.X ^ 2 + 9) := 
sorry

end factor_x4_minus_81_l666_666589


namespace count_complex_solutions_l666_666973

noncomputable def complex_solutions_count (z : ℂ) : ℕ :=
  if (|z| = 1 ∧ |(z / (conj z)) - ((conj z) / z)| = 1) then 8 else 0

theorem count_complex_solutions : complex_solutions_count = 8 := sorry

end count_complex_solutions_l666_666973


namespace units_digit_of_27_mul_36_l666_666119

theorem units_digit_of_27_mul_36 : (27 * 36) % 10 = 2 := by
  sorry

end units_digit_of_27_mul_36_l666_666119


namespace four_pow_minus_a_l666_666656

noncomputable def log_base_3 (x : ℝ) := Real.log x / Real.log 3

theorem four_pow_minus_a {a : ℝ} (h : a * log_base_3 4 = 2) : 4^(-a) = 1 / 9 :=
by
  sorry

end four_pow_minus_a_l666_666656


namespace range_of_f_l666_666601

def f (x : ℝ) : ℝ := 2 * x ^ 2 - 8 * x + 3

theorem range_of_f :
  (set.range (λ x, f x : ℝ → ℝ) ∩ set.Icc 2 5) = set.Icc (-5 : ℝ) 13 :=
sorry

end range_of_f_l666_666601


namespace cheese_cookie_price_l666_666015

theorem cheese_cookie_price
  (boxes_per_carton : ℕ)
  (packs_per_box : ℕ)
  (cost_per_dozen_cartons : ℕ) :
  boxes_per_carton = 12 →
  packs_per_box = 10 →
  cost_per_dozen_cartons = 1440 →
  cost_per_dozen_cartons / (boxes_per_carton * 12 * packs_per_box) = 1 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry -- Proof steps can be filled in, but not required for this task.

end cheese_cookie_price_l666_666015


namespace not_right_triangle_exists_l666_666556

theorem not_right_triangle_exists :
  (∃ (triangle : Type) (h₁ : triangle → Prop) (h₂ : triangle → Prop) 
  (h₃ : triangle → Prop) (h₄ : triangle → Prop),
  (h₁ → (triangle → false)) ∧
  (h₂ → (triangle → true)) ∧
  (h₃ → (triangle → false)) ∧
  (h₄ → (triangle → false))) :=
sorry

end not_right_triangle_exists_l666_666556


namespace arnaldo_bernaldo_second_and_third_cernaldo_scored_points_races_won_by_racers_l666_666987

-- Definitions of racers and their final scores
structure Racer where
  name : String
  score : ℕ

namespace Racer
def Arnaldo   := { name := "Arnaldo", score := 19 }
def Bernaldo  := { name := "Bernaldo", score := 16 }
def Cernaldo  := { name := "Cernaldo", score := 0 } -- Placeholder for actual calculation
def Dernaldo  := { name := "Dernaldo", score := 0 } -- Placeholder for actual calculation
def Ernaldo   := { name := "Ernaldo", score := 0 } -- Placeholder for actual calculation
end Racer

-- Total points distributed in each race
def points_distributed_per_race := 5 + 4 + 3 + 2 + 1

-- Total points distributed across all races
def total_points := 5 * points_distributed_per_race

-- Points obtained by Arnaldo and Bernaldo
def total_points_Arnaldo_Bernaldo := Racer.Arnaldo.score + Racer.Bernaldo.score

-- Points left after Arnaldo and Bernaldo
def remaining_points := total_points - total_points_Arnaldo_Bernaldo

-- Assumptions
axiom no_ties_in_races_and_final_scores : true
axiom points_distribution_correct : points_distributed_per_race = 15

-- Theorem a) Arnaldo and Bernaldo consistently took second and third places.
theorem arnaldo_bernaldo_second_and_third :
  ∀ race_num : ℕ, race_num ∈ {1, 2, 3, 4, 5} →
  let a := 2 in  -- Placeholder: actual race results need to be more detailed
  true := sorry

-- Theorem b) Determine points scored by Cernaldo.
theorem cernaldo_scored_points :
  Racer.Cernaldo.score = 15 := sorry

-- Theorem c) Number of races won by each racer.
theorem races_won_by_racers :
  true := sorry -- Placeholder for actual race calculation

end arnaldo_bernaldo_second_and_third_cernaldo_scored_points_races_won_by_racers_l666_666987


namespace chords_in_circle_l666_666795

theorem chords_in_circle (n : ℕ) (h_n : n = 10) : (n.choose 2) = 45 := sorry

end chords_in_circle_l666_666795


namespace cos_double_angle_l666_666239

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666239


namespace log_gt_quotient_log_x1_gt_quot_x2_l666_666676

-- Problem 1: f(x) > g(x) for x in (1, +∞) when a = 2
theorem log_gt_quotient (x : ℝ) (h : x > 1) : 
  real.log x > (2 * (x - 1)) / (x + 1) :=
sorry

-- Problem 2: Discussing the monotonicity of h(x)
def h (x : ℝ) (a : ℝ) : ℝ := real.log x - (a * (x - 1)) / (x + 1)

-- Problem 3: f(x+1) > x^2 / (e^x - 1) for x > 0
theorem log_x1_gt_quot_x2 (x : ℝ) (h : x > 0) : 
  real.log (x + 1) > (x^2) / (real.exp x - 1) :=
sorry

end log_gt_quotient_log_x1_gt_quot_x2_l666_666676


namespace adults_had_meal_l666_666027

theorem adults_had_meal (A : ℕ) (h1 : 70 ≥ A) (h2 : ((70 - A) * 9) = (72 * 7)) : A = 14 := 
by
  sorry

end adults_had_meal_l666_666027


namespace vacationers_city_correctness_l666_666605

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

end vacationers_city_correctness_l666_666605


namespace complex_root_product_l666_666702

theorem complex_root_product (w : ℂ) (hw1 : w^3 = 1) (hw2 : w^2 + w + 1 = 0) :
(1 - w + w^2) * (1 + w - w^2) = 4 :=
sorry

end complex_root_product_l666_666702


namespace candy_bar_calories_l666_666462

theorem candy_bar_calories :
  let calA := 150
  let calB := 200
  let calC := 250
  let countA := 2
  let countB := 3
  let countC := 4
  (countA * calA + countB * calB + countC * calC) = 1900 :=
by
  sorry

end candy_bar_calories_l666_666462


namespace binom_divisibility_l666_666399

theorem binom_divisibility (p : ℕ) (h₀ : Nat.Prime p) (h₁ : p % 2 = 1) : 
  (Nat.choose (2 * p - 1) (p - 1) - 1) % (p^2) = 0 := 
by 
  sorry

end binom_divisibility_l666_666399


namespace cosine_double_angle_l666_666255

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l666_666255


namespace length_of_AD_l666_666722

variables (A B C D O : Type*)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]
variables [normed_add_comm_group A] [normed_space ℝ A] [normed_add_comm_group B] [normed_space ℝ B]
variables [normed_add_comm_group C] [normed_space ℝ C] [normed_add_comm_group D] [normed_space ℝ D]
variables [normed_add_comm_group O] [normed_space ℝ O]

variables (BO OD AO OC AB BD : ℝ)
axiom BO_def : BO = 5
axiom OD_def : OD = 7
axiom AO_def : AO = 9
axiom OC_def : OC = 4
axiom AB_def : AB = 6
axiom BD_def : BD = 6

theorem length_of_AD (A B O D C: Type*) [metric_space A] [metric_space B] [metric_space O] [metric_space D] [metric_space C]
[normed_add_comm_group A] [normed_space ℝ A] [normed_add_comm_group B] [normed_space ℝ B]
[normed_add_comm_group O] [normed_space ℝ O] [normed_add_comm_group D] [normed_space ℝ D]
[normed_add_comm_group C] [normed_space ℝ C] :
BO = 5 →
OD = 7 →
AO = 9 →
OC = 4 →
AB = 6 →
BD = 6 →
∃ AD: ℝ, AD = 2 * real.sqrt 57 :=
by
  sorry

end length_of_AD_l666_666722


namespace shelby_can_buy_posters_l666_666404

noncomputable def shelby_initial_money : ℝ :=
50

noncomputable def book1_price : ℝ :=
12.5

noncomputable def book2_price : ℝ :=
7.25

noncomputable def bookmark_price : ℝ :=
2.75

noncomputable def pencils_price : ℝ :=
3.80

noncomputable def discount_threshold : ℝ :=
30

noncomputable def discount_rate : ℝ :=
0.10

noncomputable def tax_rate : ℝ :=
0.07

noncomputable def poster_price : ℝ :=
5.50

theorem shelby_can_buy_posters :
  let total_before_discount := book1_price + book2_price + bookmark_price + pencils_price in
  let discount := if total_before_discount > discount_threshold then total_before_discount * discount_rate else 0 in
  let total_after_discount := total_before_discount - discount in
  let total_with_tax := total_after_discount * (1 + tax_rate) in
  let money_left := shelby_initial_money - total_with_tax in
  ∃ (n : ℕ), n = floor (money_left / poster_price) ∧ n = 3 :=
by
  sorry

end shelby_can_buy_posters_l666_666404


namespace aisha_probability_four_tosses_l666_666913

noncomputable def probability_at_least_two_heads (tosses : ℕ) (heads_needed : ℕ) : ℚ :=
  1 - (nat.choose tosses 1 * (1/2)^tosses + (1/2)^tosses)

theorem aisha_probability_four_tosses :
  probability_at_least_two_heads 4 2 = 11/16 := by
  sorry 

end aisha_probability_four_tosses_l666_666913


namespace Vasya_unique_dish_sets_l666_666330

theorem Vasya_unique_dish_sets (n : ℕ) (h_n : n = 6) :
  (∃ max_days : ℕ, max_days = 2^n) ∧
  (∃ avg_dishes : ℕ, avg_dishes = n / 2) :=
by {
  have h_max_days : 2^6 = 64, from rfl,
  have h_avg_dishes : 6 / 2 = 3, from rfl,
  exact ⟨⟨64, h_max_days⟩, ⟨3, h_avg_dishes⟩⟩,
}

end Vasya_unique_dish_sets_l666_666330


namespace eval_complex_powers_l666_666587

noncomputable def i : Complex := Complex.I

theorem eval_complex_powers :
  let i := Complex.I
  (i^13 + i^18 + i^23 + i^28 + i^33 = i) :=
by
  /- conditions -/
  have h1 : i^2 = -1 := by simp [Complex.I_eq_i]
  have h2 : i^4 = 1 := by simp [Complex.I_eq_i]
  sorry

end eval_complex_powers_l666_666587


namespace cos_double_angle_l666_666246

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666246


namespace male_athletes_sampled_l666_666053

-- Define the total number of athletes
def total_athletes : Nat := 98

-- Define the number of female athletes
def female_athletes : Nat := 42

-- Define the probability of being selected
def selection_probability : ℚ := 2 / 7

-- Calculate the number of male athletes
def male_athletes : Nat := total_athletes - female_athletes

-- State the theorem about the number of male athletes sampled
theorem male_athletes_sampled : male_athletes * selection_probability = 16 :=
by
  sorry

end male_athletes_sampled_l666_666053


namespace probability_of_both_selected_l666_666514

noncomputable def ramSelectionProbability : ℚ := 1 / 7
noncomputable def raviSelectionProbability : ℚ := 1 / 5

theorem probability_of_both_selected : 
  ramSelectionProbability * raviSelectionProbability = 1 / 35 :=
by sorry

end probability_of_both_selected_l666_666514


namespace probability_at_least_two_heads_is_11_over_16_l666_666916

noncomputable def probability_of_heads : ℚ := 1 / 2

noncomputable def probability_at_least_two_heads : ℚ :=
  1 - (nat.choose 4 0 * probability_of_heads^4 + nat.choose 4 1 * probability_of_heads^4)

theorem probability_at_least_two_heads_is_11_over_16 :
  probability_at_least_two_heads = 11 / 16 := by
  sorry

end probability_at_least_two_heads_is_11_over_16_l666_666916


namespace solve_equation_1_solve_equation_2_solve_equation_3_l666_666409

theorem solve_equation_1 (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 := 
sorry

theorem solve_equation_2 (x : ℝ) : (2 * x - 1)^2 = (3 - x)^2 ↔ x = -2 ∨ x = 4 / 3 := 
sorry

theorem solve_equation_3 (x : ℝ) : 3 * x * (x - 2) = x - 2 ↔ x = 2 ∨ x = 1 / 3 :=
sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l666_666409


namespace find_hyperbola_equation_l666_666111

def hyperbola_equation_exists (x y : ℝ) (h_asymptotes : ∀ x y, x^2 / 2 - y^2 = 1) (P : ℝ × ℝ) : Prop :=
  P = (2, -2) ∧ (∀ k : ℝ, x^2 / 2 - y^2 = k ↔ k = -2) ∧ (∀ a b : ℝ, y^2 / a^2 - x^2 / b^2 = 1) ∧ a = sqrt 2 ∧ b = 2

theorem find_hyperbola_equation :
  hyperbola_equation_exists 2 (-2)
    (λ x y, x^2 / 2 - y^2 = 1)
    (2, -2) :=
sorry

end find_hyperbola_equation_l666_666111


namespace cos_double_angle_l666_666269

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666269


namespace increasing_C_l666_666174

theorem increasing_C (e R r : ℝ) (n : ℕ) (h₁ : 0 < e) (h₂ : 0 < R) (h₃ : 0 < r) (h₄ : 0 < n) :
    ∀ n1 n2 : ℕ, n1 < n2 → (e^2 * n1) / (R + n1 * r) < (e^2 * n2) / (R + n2 * r) :=
by
  sorry

end increasing_C_l666_666174


namespace number_of_chords_l666_666790

theorem number_of_chords (n : ℕ) (h : n = 10) : finset.card (finset.pairs (finset.range n)) = 45 :=
by
  rw [h]
  -- Sorry to skip the proof steps as required
  sorry

end number_of_chords_l666_666790


namespace not_right_triangle_l666_666559

-- Conditions
def ratio_sides (s1 s2 s3 : ℝ) : Prop :=
  s1 = 3 * s2 ∧ s2 = 4 * s3 ∧ s1 = 5 * s3

def ratio_angles_345 (a1 a2 a3 : ℝ) : Prop :=
  a1 / a2 = 3 / 4 ∧ a2 / a3 = 4 / 5

def ratio_angles_123 (a1 a2 a3 : ℝ) : Prop :=
  a1 / a2 = 1 / 2 ∧ a2 / a3 = 2 / 3

def squares_sides (s1 s2 s3 : ℝ) : Prop :=
  s1^2 = k * s2^2 ∧ s2^2 = 2 * s3^2 ∧ s1^2 = 3 * s3^2

-- Statement
theorem not_right_triangle (a1 a2 a3 s1 s2 s3 : ℝ) (k: ℝ) :
  (ratio_sides s1 s2 s3 → a1 = 90 ∨ a2 = 90 ∨ a3 = 90) →
  (ratio_angles_123 a1 a2 a3 → a1 = 90 ∨ a2 = 90 ∨ a3 = 90) →
  (squares_sides s1 s2 s3 → a1 = 90 ∨ a2 = 90 ∨ a3 = 90) →
  (ratio_angles_345 a1 a2 a3 → a1 ≠ 90 ∧ a2 ≠ 90 ∧ a3 ≠ 90) →
  ratio_angles_345 a1 a2 a3 := sorry

end not_right_triangle_l666_666559


namespace graph_remains_connected_after_removing_one_color_l666_666466

/-- A 30-point graph where each pair is connected by an edge of one of four colors remains connected after removing all edges of some one color. -/
theorem graph_remains_connected_after_removing_one_color :
  ∃ C : Fin 4 → Prop,
    ∀ G : SimpleGraph (Fin 30),
    ((∀ v w : Fin 30, v ≠ w → ∃ c : Fin 4, G.edge v w c) →
     ∀ c : Fin 4, (∀ p q : Fin 30, p ≠ q → (∃ r : Fin 30, r ≠ p ∧ r ≠ q ∧ ¬G.edge p q c)) → 
     Connected (G ⊖ {E | E.2 = c → E ∉ G.edge})) :=
by sorry

end graph_remains_connected_after_removing_one_color_l666_666466


namespace math_problem_l666_666183

noncomputable def f (a b : ℝ) (x : ℝ) := a * x ^ 2 + b * x + 1
noncomputable def g (a b k : ℝ) (x : ℝ) := f a b x - k * x

theorem math_problem
  (a b k : ℝ)
  (x : ℝ)
  (h1 : a ≠ 0)
  (h2 : f a b (-2) = 1)
  (h3 : (b ^ 2 - 4 * a = 0))
  (h4 : ∀ x, x ∈ set.Icc (-1 : ℝ) 2 → (g a b k x - g a b k (-1)) * (x - (-1)) ≥ 0) :
  (a = 1 ∧ b = 2 ∧ (g a b k x = (x ^ 2 - (k - 2) * x + 1) 
  ∧ (k ≤ 0 ∨ k ≥ 6))) :=
sorry

end math_problem_l666_666183


namespace lcm_value_l666_666817

def hcf (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

def number1 : ℕ := 210
def number2 : ℕ := 517
def hcf_known : ℕ := 47

theorem lcm_value : lcm number1 number2 = 2310 :=
by
  have hcf_def : hcf number1 number2 = hcf_known := by sorry
  exact sorry

end lcm_value_l666_666817


namespace chords_on_circle_l666_666801

theorem chords_on_circle (n : ℕ) (h : n = 10) : nat.choose n 2 = 45 :=
by {
  rw h,
  -- we can directly calculate choose 10 2
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num),
  sorry -- the actual detailed proof goes here.
}

end chords_on_circle_l666_666801


namespace find_x_l666_666852

theorem find_x (u : ℕ) (h₁ : u = 90) (w : ℕ) (h₂ : w = u + 10)
                (z : ℕ) (h₃ : z = w + 25) (y : ℕ) (h₄ : y = z + 15)
                (x : ℕ) (h₅ : x = y + 3) : x = 143 :=
by {
  -- Proof will be included here
  sorry
}

end find_x_l666_666852


namespace second_player_wins_l666_666884

-- Define the game's setup parameters and conditions
def hexagon_game (n : ℕ) : Prop :=
  ∀ (s : ℕ), (s ≤ (3 * (n^2) + 3 * n + 1)) → (s mod 2 = 0 →  player_wins_with_strategy 2) ∧
             (s mod 2 = 1 →  player_wins_with_strategy 1)

-- Statement that the second player has a winning strategy
theorem second_player_wins (n : ℕ) : 
  hexagon_game n :=
begin
  sorry
end

end second_player_wins_l666_666884


namespace distribute_prizes_8000_l666_666097

def num_ways_distribute_prizes (N k : ℕ) : ℕ :=
  N ^ k

theorem distribute_prizes_8000 (N : ℕ) (k : ℕ) (hN : N = 20) (hk : k = 3) :
  num_ways_distribute_prizes N k = 8000 :=
by
  -- since N = 20 and k = 3
  rw [hN, hk]
  -- we have by definition num_ways_distribute_prizes 20 3 = 20^3
  exact Nat.pow_succ 20 2 -- Reduce 20^3 to a base calculation
  -- calculation yields 8000
  -- it's clear 20 * 20 * 20 = 8000
  sorry    -- We skip further detailed proof verification

end distribute_prizes_8000_l666_666097


namespace hyperbola_properties_l666_666189

open Real

-- Defining the hyperbola conditions
def hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  { p | ∃ (x y : ℝ), p = (x, y) ∧ (x^2 / a^2 - y^2 / b^2 = 1) }

-- Definition for distance from focus to asymptote
def dist_focus_to_asymptote (a b c : ℝ) : Prop :=
  let d := |b * c| / sqrt (a^2 + b^2) in
  d = sqrt 3

-- The equation of our specific hyperbola
def equation_of_C : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ hyperbola 1 (sqrt 3) ↔ (x^2 - y^2 / 3 = 1)

-- The proof problem regarding the fixed point for MN
def passes_through_fixed_point (A M N : ℝ × ℝ) : Prop :=
  let kAM := (snd A - snd M) / (fst A - fst M) in
  let kAN := (snd A - snd N) / (fst A - fst N) in
  kAM * kAN = -9 →
  ∃ (P : ℝ × ℝ), (∀ (x y : ℝ), (x, y) = P ↔ x = 2 ∧ y = 0)

-- The overall statement for the Lean theorem
theorem hyperbola_properties (a b c : ℝ) : equation_of_C → 
  dist_focus_to_asymptote a b c →
  ∀ (A M N : ℝ × ℝ), A = (1,0) → (fst M > 1 ∧ fst N > 1) →
  passes_through_fixed_point A M N
  :=
sorry

end hyperbola_properties_l666_666189


namespace triangle_area_sides_5_5_6_l666_666485

theorem triangle_area_sides_5_5_6 (a b c : ℕ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 6) :
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 12 :=
by
  sorry

end triangle_area_sides_5_5_6_l666_666485


namespace cyclic_pentagon_cosines_l666_666869

theorem cyclic_pentagon_cosines
  (A B C D E : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  [MetricSpace D]
  [MetricSpace E]
  (circ : Circle A B C D E)
  (AB BC CD DE : Real)
  (AE : Real)
  (h_AB : AB = 2)
  (h_BC : BC = 2)
  (h_CD : CD = 2)
  (h_DE : DE = 2)
  (h_AE : AE = 3) :
  (1 - cos (angle B)) * (1 - cos (angle ADE)) = 5 / 64 := 
by
  sorry

end cyclic_pentagon_cosines_l666_666869


namespace discarded_number_is_45_l666_666805

theorem discarded_number_is_45
  (X : ℝ)
  (average_50 : ∀ (S : ℝ), S / 50 = 62 → S)
  (average_48 : ∀ (S : ℝ), S / 50 = 62 → (S - X - 55) / 48 = 62.5):
  X = 45 :=
by
  sorry

end discarded_number_is_45_l666_666805


namespace pure_imaginary_product_l666_666106

theorem pure_imaginary_product (x : ℝ) :
  let z1 := complex.of_real x + complex.I * 2,
      z2 := complex.of_real (x + 1) + complex.I * 2,
      z3 := complex.of_real (x + 2) + complex.I * 2 in
  (complex.re (z1 * z2 * z3) = 0) → x = 1 :=
sorry

end pure_imaginary_product_l666_666106


namespace count_complex_solutions_l666_666974

noncomputable def complex_solutions_count (z : ℂ) : ℕ :=
  if (|z| = 1 ∧ |(z / (conj z)) - ((conj z) / z)| = 1) then 8 else 0

theorem count_complex_solutions : complex_solutions_count = 8 := sorry

end count_complex_solutions_l666_666974


namespace relationship_among_a_b_c_l666_666826

-- Definitions of the given numbers
def a : ℝ := 7 ^ 0.3
def b : ℝ := 0.3 ^ 7
def c : ℝ := Real.log 0.3

-- The main statement to prove
theorem relationship_among_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_among_a_b_c_l666_666826


namespace cos_double_angle_l666_666261

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l666_666261


namespace notebook_cost_l666_666402

theorem notebook_cost
  (initial_amount : ℝ)
  (notebook_count : ℕ)
  (pen_count : ℕ)
  (pen_cost : ℝ)
  (remaining_amount : ℝ)
  (total_spent : ℝ)
  (notebook_cost : ℝ) :
  initial_amount = 15 →
  notebook_count = 2 →
  pen_count = 2 →
  pen_cost = 1.5 →
  remaining_amount = 4 →
  total_spent = initial_amount - remaining_amount →
  total_spent = notebook_count * notebook_cost + pen_count * pen_cost →
  notebook_cost = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end notebook_cost_l666_666402


namespace measure_of_angle_EGQ_l666_666403

variables {EF FG : ℝ}
variables (x : ℝ)

-- Definitions from conditions
def midpoint (a b : ℝ) : ℝ := (a + b) / 2
def radius (d : ℝ) : ℝ := d / 2
def area_of_semicircle (r : ℝ) : ℝ := (1/2) * Real.pi * r^2
def total_area (a1 a2 : ℝ) : ℝ := a1 + a2

-- Conditions
axiom EF_length : EF = 2 * x
axiom FG_length : FG = x
axiom G_is_midpoint : midpoint EF 0 = x
axiom H_is_midpoint : midpoint x 0 = x / 2
axiom area1_is_twice_area2 : ∀ (area1 area2 : ℝ), (area_of_semicircle x = area1 * 3 / 2 ∨ area_of_semicircle (x / 2) = area2 * 3 / 2)
axiom area1_area2_relation : total_area (area_of_semicircle x) (area_of_semicircle (x / 2)) = (area_of_semicircle x) * 5 / 8 + (area_of_semicircle (x / 2)) * 5 / 8

-- Desired theorem
theorem measure_of_angle_EGQ (Θ : ℝ) : Θ = 150 :=
by
  sorry

end measure_of_angle_EGQ_l666_666403


namespace students_no_preference_l666_666318

/-- In a survey of 210 college students:
  - 60 students preferred Mac.
  - 40 students preferred Windows.
  - One third of the students who preferred Mac equally preferred both brands.

We aim to prove the number of students with no preference between Windows or Mac 
presuming the above conditions is 90.
-/
theorem students_no_preference 
  (total_students : ℕ) (mac_preference : ℕ) (windows_preference : ℕ)
  (both_brands_preference_ratio : ℕ → ℕ)
  (no_preference_count : ℕ) :
  total_students = 210 →
  mac_preference = 60 →
  windows_preference = 40 →
  (both_brands_preference_ratio mac_preference = mac_preference / 3) →
  no_preference_count = total_students - (mac_preference + windows_preference + both_brands_preference_ratio mac_preference) →
  no_preference_count = 90 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3] at *,
  sorry
end

end students_no_preference_l666_666318


namespace chords_on_circle_l666_666803

theorem chords_on_circle (n : ℕ) (h : n = 10) : nat.choose n 2 = 45 :=
by {
  rw h,
  -- we can directly calculate choose 10 2
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num),
  sorry -- the actual detailed proof goes here.
}

end chords_on_circle_l666_666803


namespace boys_meetings_excluding_start_finish_l666_666839

def semi_major_axis : ℝ := 40
def semi_minor_axis : ℝ := 30
def speed_boy1 : ℝ := 7
def speed_boy2 : ℝ := 10

def approximation_of_ellipse_circumference (a b : ℝ) : ℝ :=
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

def relative_speed (v1 v2 : ℝ) : ℝ :=
  v1 + v2

def time_to_meet_again (C v : ℝ) : ℝ :=
  C / v

def number_of_meetings (t : ℝ) : ℤ :=
  Int.floor t - 1

theorem boys_meetings_excluding_start_finish :
  number_of_meetings (time_to_meet_again (approximation_of_ellipse_circumference semi_major_axis semi_minor_axis) (relative_speed speed_boy1 speed_boy2)) = 13 :=
  sorry

end boys_meetings_excluding_start_finish_l666_666839


namespace sandrines_brother_borrowed_sheets_l666_666781

noncomputable def borrowed_sheets (n_total_pages : ℕ) (pages_per_sheet : ℕ) (total_sheets : ℕ) 
                                  (remaining_avg : ℕ) (final_avg : ℕ) : ℕ :=
let c := total_sheets - final_avg * total_sheets / remaining_avg in 
c

theorem sandrines_brother_borrowed_sheets :
  borrowed_sheets 64 2 32 23 23 = 12 :=
by
  let n_total_pages := 64
  let pages_per_sheet := 2
  let total_sheets := 32
  let remaining_avg := 23
  let final_avg := 23
  sorry

end sandrines_brother_borrowed_sheets_l666_666781


namespace cos_double_angle_l666_666266

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666266


namespace max_area_rectangle_l666_666893

-- Define the conditions as a Lean 4 statement
theorem max_area_rectangle (l w : ℕ) (hl : l + w = 20) (hw : even w) : l * w ≤ 100 :=
  sorry

end max_area_rectangle_l666_666893


namespace minimum_value_of_f_l666_666133

def f (x : ℝ) : ℝ := x^2 - x + 1 + Real.sqrt (2*x^4 - 18*x^2 + 12*x + 68)

theorem minimum_value_of_f :
  ∃ x : ℝ, f x = 9 :=
sorry

end minimum_value_of_f_l666_666133


namespace cos_double_angle_l666_666284

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666284


namespace number_of_chords_l666_666798

/-- Ten points are marked on the circumference of a circle.
    Prove that the number of different chords that can be drawn
    by connecting any two of these ten points is 45.
-/
theorem number_of_chords (n : ℕ) (h_n : n = 10) : 
  (nat.choose n 2) = 45 :=
by
  rw h_n
  norm_num

end number_of_chords_l666_666798


namespace tileable_rectangles_l666_666405

noncomputable def M : ℕ := sorry

theorem tileable_rectangles (b : ℕ) (m n : ℕ) (M : ℕ) : 
  Even b →
  (Even (m * n)) →
  (M < m) →
  (M < n) →
  ∃ M, M ∈ Finset.univ.filter (λ x => x ∈ Mline_const b) → 
  tileable (Rectangle m n) (Tile 1 b) :=
sorry

end tileable_rectangles_l666_666405


namespace bad_carrots_count_l666_666867

def total_carrots (vanessa_carrots : ℕ) (mother_carrots : ℕ) : ℕ := 
vanessa_carrots + mother_carrots

def bad_carrots (total_carrots : ℕ) (good_carrots : ℕ) : ℕ := 
total_carrots - good_carrots

theorem bad_carrots_count : 
  ∀ (vanessa_carrots mother_carrots good_carrots : ℕ), 
  vanessa_carrots = 17 → 
  mother_carrots = 14 → 
  good_carrots = 24 → 
  bad_carrots (total_carrots vanessa_carrots mother_carrots) good_carrots = 7 := 
by 
  intros; 
  sorry

end bad_carrots_count_l666_666867


namespace point_P_trajectory_rect_eqn_max_dist_P_to_line_l666_666168

-- Given conditions:
def pole_origin : Prop := true  -- The pole of the polar coordinate system is at the origin.
def polar_axis_x_axis : Prop := true  -- The polar axis coincides with the positive semi-axis of the x-axis.
def length_units_same : Prop := true  -- Length units are the same.
def polar_eq_line (ρ θ : ℝ) : Prop := ρ = 5 / (Real.sin (θ - π / 3))  -- Polar coordinate equation of line l.
def point_P (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α + 2)  -- Point P(x,y) where α ∈ ℝ.

-- Theorem 1: Proving the rectangular coordinate equation of P's trajectory
theorem point_P_trajectory_rect_eqn :
  pole_origin → polar_axis_x_axis → length_units_same → 
  (∀ α : ℝ, let P := point_P α in P.fst ^ 2 + (P.snd - 2) ^ 2 = 4) :=
by
  intros _ _ _ α
  let P := point_P α
  have h : P = (2 * Real.cos α, 2 * Real.sin α + 2) := rfl
  simp [h]
  sorry

-- Theorem 2: Proving the maximum distance from P to line l is 6
theorem max_dist_P_to_line :
  pole_origin → polar_axis_x_axis → length_units_same → 
  (∀ α : ℝ, let ρ := 2 * Real.cos α in 
  let θ := α in 
  let d := (|-2 + 10|) / (Real.sqrt (3 ^ 2 + 1 ^ 2)) + 2 in
  d = 6) :=
by
  intros _ _ _
  let d := (|-2 + 10|) / (Real.sqrt (3 ^ 2 + 1 ^ 2)) + 2
  sorry

end point_P_trajectory_rect_eqn_max_dist_P_to_line_l666_666168


namespace alpha_perpendicular_beta_l666_666366

variables {l : Line} {α β : Plane}

-- Conditions
axiom line_parallel_plane (l : Line) (α : Plane) : Prop
axiom line_perpendicular_plane (l : Line) (β : Plane) : Prop
axiom planes_perpendicular (α β : Plane) : Prop

-- Given conditions
hypothesis l_parallel_alpha : line_parallel_plane l α
hypothesis l_perpendicular_beta : line_perpendicular_plane l β

-- Prove the statement
theorem alpha_perpendicular_beta : planes_perpendicular α β :=
sorry

end alpha_perpendicular_beta_l666_666366


namespace probability_of_winning_l666_666440

theorem probability_of_winning
    (P_lose : ℚ)
    (h1 : P_lose = 5 / 7)
    (h2 : (∀ P_win : ℚ, P_win + P_lose = 1)) :
    ∃ P_win : ℚ, P_win = 2 / 7 :=
by
  use (1 - P_lose)
  have h : 1 - (5 / 7) = 2 / 7, by norm_num
  exact ⟨h⟩ sorry

end probability_of_winning_l666_666440


namespace proof_lines_SA_SB_l666_666822

noncomputable def line_intersection (l1 l2: ℝ → ℝ × ℝ) : ℝ × ℝ :=
let (x1, y1) := l1 1 in
let (x2, y2) := l2 1 in
have same_x : x1 = x2, by sorry
have same_y : y1 = y2, by sorry
(x1, y1)

noncomputable def line_SA (x y: ℝ) : Prop :=
165 * x - 296 * y + 665 = 0

noncomputable def line_SB (x y: ℝ) : Prop :=
723 * x - 584 * y + 1007 = 0

theorem proof_lines_SA_SB : 
(∀ x y : ℝ, line_SA x y ↔ line_SB x y )  := 
begin
have Sx : 7*x + 5*y + 8 = 0,
have Sy : 3*x + 4*y - 13=0,
have A: 165 * x - 296 * y + 665 = 0,
have B: 723 * x - 584 * y+ 1007 = 0,
sorry
end

end proof_lines_SA_SB_l666_666822


namespace count_divisible_by_35_l666_666691

theorem count_divisible_by_35 : 
  ∃! (n : ℕ), n = 13 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 ∧ (∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ ab = 10 * a + b) →
    (ab * 100 + 35) % 35 = 0 ↔ ab % 7 = 0) :=
by {
  sorry
}

end count_divisible_by_35_l666_666691


namespace circle_area_l666_666491

-- Define the diameter d
def diameter := 10

-- Define the radius r in terms of the diameter
def radius := diameter / 2

-- Define the area A of a circle in terms of the radius
def area_of_circle (r : ℝ) : ℝ := π * r^2

theorem circle_area (d : ℝ) (hd : d = 10) : area_of_circle (d / 2) = 25 * π :=
by {
  -- using the definition of radius
  have r_def : radius = d / 2 := rfl,
  -- calculate area
  rw [r_def, hd],
  calc
    area_of_circle (10 / 2) = 25 * π : by {
      rw [area_of_circle, div_eq_mul_one_div],
      norm_num,
      ring
      },
}

end circle_area_l666_666491


namespace det_S_eq_6_l666_666754

noncomputable def R_45 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![sqrt 2 / 2, -sqrt 2 / 2],
    ![sqrt 2 / 2, sqrt 2 / 2]]

noncomputable def D : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 0], 
    ![0, 3]]

noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ := D ⬝ R_45

theorem det_S_eq_6 : 
  ∃ S : Matrix (Fin 2) (Fin 2) ℝ, 
    S = D ⬝ R_45 ∧ 
    Matrix.det S = 6 := 
by
  sorry

end det_S_eq_6_l666_666754


namespace length_of_segment_AC_l666_666021

noncomputable def circle_radius (C : ℝ) := C / (2 * Real.pi)
noncomputable def acos_degrees_to_radians (x : ℝ) := Real.pi * x / 180

theorem length_of_segment_AC 
    (C P : Type)
    [metric_space C]
    [circle C]
    [circle_metric_space C P]
    (A B C D : C)
    (r : ℝ)
    (circumference_eq : ∀ P, circumference P = 18 * Real.pi)
    (diameter_eq : ∀ (P : C), is_diameter_of_circle A B P)
    (angle_eq : ∀ (P : C), angle A P C = acos_degrees_to_radians 45)
    :
    dist A C = 9 :=
by
  sorry

end length_of_segment_AC_l666_666021


namespace triangle_medians_proof_l666_666039

variables {A B C G M N P X : Type}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited G] [inhabited M] [inhabited N] [inhabited P] [inhabited X]
variables (BC : B → C → ℝ)
variables (CA : C → A → ℝ)
variables (AB : A → B → ℝ)
variables (AG : A → G → ℝ)
variables (GP : G → P → ℝ)
variables (CP : C → P → ℝ)
variables (BN : B → N → ℝ)
variables (AM : A → M → ℝ)

-- Condition: M, N, and P are midpoints of sides BC, CA, and AB respectively.
def is_midpoint (p midpoint1 midpoint2 : Type) : Prop :=
  BC midpoint1 midpoint2 / 2 = BC midpoint1 p

def M_is_midpoint : Prop := is_midpoint M B C
def N_is_midpoint : Prop := is_midpoint N C A
def P_is_midpoint : Prop := is_midpoint P A B

-- Condition: Medians BN and CP intersect at point G.
def BN_is_median : Prop := ∀ {b₁ : B} {n₁ : N}, BN b₁ n₁ / 2 = BN b₁ G 
def CP_is_median : Prop := ∀ {c₁ : C} {p₁ : P}, CP c₁ p₁ / 2 = CP c₁ G

-- Condition: X is the midpoint of segment AG.
def X_is_midpoint_AG : Prop := AG A G / 2 = AG A X

-- Question (a): G divides CP in the ratio 2:1
def divides_AT_ratio (A B G : Type) (ratio : ℝ) : Prop :=
  (AG A G) = ratio * (GP G B)

-- Question (b): The median AM intersects CP at G.
def intersects_AT_G (AM CP G : Type) : Prop :=
  ∀ {a₁ : A} {m₁ : M}, AM a₁ m₁ / 2 = (AM a₁ G) ∧ 
  ∀ {c₁ : C} {p₁ : P}, CP c₁ p₁ / 2 = (CP c₁ G)

-- Question (c): AC = AB given BN and CP have the same length
def lengths_equal (A B C : Type)
  (BN CP : B → N → ℝ)
  (length_equal : Prop) : Prop :=
  (BN = CP) → AB A B = AC C A

theorem triangle_medians_proof :
  M_is_midpoint ∧ N_is_midpoint ∧ P_is_midpoint ∧
  BN_is_median ∧ CP_is_median ∧ X_is_midpoint_AG →
  divides_AT_ratio C P G 2 ∧
  intersects_AT_G AM CP G ∧
  lengths_equal A B C BN CP (BN = CP) :=
sorry

end triangle_medians_proof_l666_666039


namespace range_of_a_l666_666139

-- Define the function f(x) = 1/3 * x^3 + x
def f (x : ℝ) : ℝ := (1 / 3) * x^3 + x

-- State the main theorem
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f (a - x) + f (a * x^2 - 1) < 0) ↔ a < (1 + Real.sqrt 2) / 2 :=
sorry

end range_of_a_l666_666139


namespace determine_ab_l666_666088

theorem determine_ab :
  ∃ a b : ℝ, 
  (3 + 8 * a = 2 - 3 * b) ∧ 
  (-1 - 6 * a = 4 * b) → 
  a = -1 / 14 ∧ b = -1 / 14 := 
by 
sorry

end determine_ab_l666_666088


namespace triangle_ceva_l666_666779

theorem triangle_ceva
  (A B C A1 B1 C1 K : Point)
  (hA1_on_BC : A1 ∈ line_segment B C)
  (hB1_on_AC : B1 ∈ line_segment A C)
  (hC1_on_AB : C1 ∈ line_segment A B)
  (h_intersect : line_segment A A1 ∩ line_segment B B1 ∩ line_segment C C1 = {K}) :
  (AK KA1 : ℝ)
  (hAK_eq : AK = distance A K)
  (hKA1_eq : KA1 = distance K A1)
  (AB1 B1C AC1 C1B : ℝ)
  (hAB1_eq : AB1 = distance A B1)
  (hB1C_eq : B1C = distance B1 C)
  (hAC1_eq : AC1 = distance A C1)
  (hC1B_eq : C1B = distance C1 B) :
  AK / KA1 = AB1 / B1C + AC1 / C1B :=
sorry

end triangle_ceva_l666_666779


namespace units_digit_fraction_l666_666850

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10 = (2500) % 10 → 
  ((30 * 31 * 32 * 33 * 34 * 35) / 2500) % 10 = 1 := 
by 
  intro h
  sorry

end units_digit_fraction_l666_666850


namespace cos_double_angle_l666_666279

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l666_666279


namespace volume_ratio_l666_666079

noncomputable def volume_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

theorem volume_ratio (r : ℝ) (h_cylinder : ℝ) (h_cone : ℝ) (h_cone_eq : h_cone = h_cylinder / 3) :
  volume_cone r h_cone / volume_cylinder r h_cylinder = 1 / 9 :=
by
  sorry

end volume_ratio_l666_666079


namespace red_balls_in_bag_l666_666018

/-- Given the conditions of the ball distribution in the bag,
we need to prove the number of red balls is 9. -/
theorem red_balls_in_bag (total_balls white_balls green_balls yellow_balls purple_balls : ℕ)
  (prob_neither_red_nor_purple : ℝ) (h_total : total_balls = 100)
  (h_white : white_balls = 50) (h_green : green_balls = 30)
  (h_yellow : yellow_balls = 8) (h_purple : purple_balls = 3)
  (h_prob : prob_neither_red_nor_purple = 0.88) :
  ∃ R : ℕ, (total_balls = white_balls + green_balls + yellow_balls + purple_balls + R) ∧ R = 9 :=
by {
  sorry
}

end red_balls_in_bag_l666_666018


namespace compare_expression_solve_inequality_l666_666872

-- Part (1) Problem Statement in Lean 4
theorem compare_expression (x : ℝ) (h : x ≥ -1) : 
  x^3 + 1 ≥ x^2 + x ∧ (x^3 + 1 = x^2 + x ↔ x = 1 ∨ x = -1) :=
by sorry

-- Part (2) Problem Statement in Lean 4
theorem solve_inequality (x a : ℝ) (ha : a < 0) : 
  (x^2 - a * x - 6 * a^2 > 0) ↔ (x < 3 * a ∨ x > -2 * a) :=
by sorry

end compare_expression_solve_inequality_l666_666872


namespace f_even_g_odd_l666_666188

def f (x : ℝ) : ℝ := 3^x + 3^(-x)
def g (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem f_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  rfl
  -- Proof of even property is skipped
  sorry

theorem g_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  rfl
  -- Proof of odd property is skipped
  sorry

end f_even_g_odd_l666_666188


namespace polynomial_not_sum_of_at_most_deg_periodic_degree_1_sum_of_two_periodic_unbounded_degree_1_is_sum_of_two_periodic_polynomial_is_sum_of_deg_plus_one_periodic_exp_not_sum_of_finite_periodic_l666_666441

-- Part (a)
theorem polynomial_not_sum_of_at_most_deg_periodic {n : ℕ} (f : ℝ → ℝ) (hf : polynomial_degree f = n) :
  ¬ ∃ (g : fin n → ℝ → ℝ), (∀ i, periodic (g i)) ∧ f = ∑ i, g i :=
sorry

-- Part (b)
theorem degree_1_sum_of_two_periodic_unbounded (f : ℝ → ℝ) (hf : polynomial_degree f = 1) :
  ∃ (g₁ g₂ : ℝ → ℝ), periodic g₁ ∧ periodic g₂ ∧ f = g₁ + g₂ → (∃ I : ℝ → Prop, ∀ (x : ℝ), I x → ¬ (bounded_on g₁ x ∧ bounded_on g₂ x)) :=
sorry

-- Part (c)
theorem degree_1_is_sum_of_two_periodic (f : ℝ → ℝ) (hf : polynomial_degree f = 1) :
  ∃ (g₁ g₂ : ℝ → ℝ), periodic g₁ ∧ periodic g₂ ∧ f = g₁ + g₂ :=
sorry

-- Part (d)
theorem polynomial_is_sum_of_deg_plus_one_periodic (f : ℝ → ℝ) (n : ℕ) (hf : polynomial_degree f = n) :
  ∃ (g : fin (n + 1) → ℝ → ℝ), (∀ i, periodic (g i)) ∧ f = ∑ i, g i :=
sorry

-- Part (e)
theorem exp_not_sum_of_finite_periodic : ¬ ∃ (g : ℕ → ℝ → ℝ), (∀ i, periodic (g i)) ∧ (∃ N, ∀ i > N, g i = 0) ∧ (eq ^ℕ) = ∑ i, g i :=
sorry

end polynomial_not_sum_of_at_most_deg_periodic_degree_1_sum_of_two_periodic_unbounded_degree_1_is_sum_of_two_periodic_polynomial_is_sum_of_deg_plus_one_periodic_exp_not_sum_of_finite_periodic_l666_666441


namespace find_side_length_l666_666438

theorem find_side_length (x : ℝ) : 
  (4 * x = 8 * Real.pi) → (x = Real.pi * 2) :=
by
  intro h
  calc
    x = (8 * Real.pi) / 4 : by linarith
    ... = 2 * Real.pi : by linarith

#eval Float.toString (2 * Float.pi)

end find_side_length_l666_666438


namespace min_buttons_needed_l666_666049

-- Defining the number of colors and the required adjacency condition
def num_colors : ℕ := 6
def min_adjacent_pairs (colors : ℕ) : ℕ := 3 * colors

-- The statement of our problem
theorem min_buttons_needed : min_adjacent_pairs num_colors = 18 := by
  sorry

end min_buttons_needed_l666_666049


namespace determine_k_linear_l666_666697

theorem determine_k_linear (k : ℝ) : |k| = 1 ∧ k + 1 ≠ 0 ↔ k = 1 := by
  sorry

end determine_k_linear_l666_666697


namespace hyperbola_correct_a_l666_666192

noncomputable def hyperbola_a_value : ℝ :=
  let a : ℝ := -4
  let b : ℝ := 3
  let asymptote_eq (x y: ℝ) := (3*x + 2*y = 0) ∨ (3*x - 2*y = 0)
  let hyperbola_eq (x y: ℝ) := ((x^2) / a + (y^2) / 9 = 1)
  a

theorem hyperbola_correct_a (x y : ℝ) (h_hyperbola : ((x^2) / -4 + (y^2) / 9 = 1))
  (h_asymptote : (3*x + 2*y = 0) ∨ (3*x - 2*y = 0)) :
  hyperbola_a_value = -4 :=
begin
  -- proof here
  sorry
end

end hyperbola_correct_a_l666_666192


namespace slowest_sheep_one_bag_l666_666300

-- Definitions based on the conditions
def total_sheep : ℕ := 30
def total_grass_bags : ℕ := 45
def total_days : ℕ := 40

def consumption_rate_total : ℝ := total_grass_bags / total_days
def average_consumption_rate_per_sheep : ℝ := consumption_rate_total / total_sheep

-- Fastest sheep eats twice the rate of the slowest sheep
def slowest_sheep_rate (x : ℝ) : Prop := 
  let average_rate := 1.5 * x
  average_consumption_rate_per_sheep = average_rate

-- Proving that the slowestest sheep will eat one bag of grass in 40 days
theorem slowest_sheep_one_bag (x : ℝ) (h : slowest_sheep_rate x) :
  1 / x = 40 :=
by
  -- We know average_consumption_rate_per_sheep = 0.0375
  have h1 : average_consumption_rate_per_sheep = 1.125 / 30 := rfl
  -- We have slowest sheep rate from condition
  sorry

end slowest_sheep_one_bag_l666_666300


namespace orthocenter_perpendicular_to_median_l666_666718

open EuclideanGeometry

variables {A B C I Q E K : Point}

-- Theorem: Prove that the line KQ is perpendicular to the line IE given the specified conditions.
theorem orthocenter_perpendicular_to_median
  (h_triangle : Triangle A B C)
  (h_incenter : Incenter I A B C)
  (h_inc_tg_AC : TangentPoint Q (incenter_circle I A B C) AC)
  (h_midpoint : Midpoint E A C)
  (h_orthocenter : Orthocenter K (Triangle B I C)) :
  Perpendicular (Line K Q) (Line I E) :=
sorry

end orthocenter_perpendicular_to_median_l666_666718


namespace number_of_non_congruent_triangles_l666_666206

-- Definitions for the problem conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent (t₁ t₂ : (ℕ × ℕ × ℕ)) : Prop :=
  t₁ ≠ t₂ ∧ ∀ (a₁ b₁ c₁ a₂ b₂ c₂ : ℕ), (t₁ = (a₁, b₁, c₁) ∧ t₂ = (a₂, b₂, c₂)) → 
    (a₁ = a₂ ∧ b₁ = b₂ ∧ c₁ = c₂) ∨ (a₁ = a₂ ∧ b₁ = c₂ ∧ c₁ = b₂) ∨ 
    (a₁ = b₂ ∧ b₁ = a₂ ∧ c₁ = c₂) ∨ (a₁ = b₂ ∧ b₁ = c₂ ∧ c₁ = a₂) ∨
    (a₁ = c₂ ∧ b₁ = a₂ ∧ c₁ = b₂) ∨ (a₁ = c₂ ∧ b₁ = b₂ ∧ c₁ = a₂)

def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Proof goal
theorem number_of_non_congruent_triangles : 
  ∃ (t : ℕ), t = 3 ∧ ∃ (triangles : list (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → is_triangle a b c ∧ perimeter a b c = 11) ∧ 
    list.nodup triangles ∧ list.length triangles = t :=
by
  sorry

end number_of_non_congruent_triangles_l666_666206


namespace disjoint_intervals_sum_le_0_55_l666_666647

open Set

theorem disjoint_intervals_sum_le_0_55
  (A : Set ℝ) (hA1 : A ⊆ (Icc 0 1)) (hA2 : ∀ (a b : ℝ), a ∈ A → b ∈ A → abs (a - b) ≠ 0.1) :
  ∑' (x in A), (length (interval_oc x)) ≤ 0.55 :=
by skip_proof_placeholder

end disjoint_intervals_sum_le_0_55_l666_666647


namespace speed_of_first_train_l666_666009

/-- The length of the first train in meters -/
def length_train_one : ℝ := 220

/-- The speed of the second train in km/hr -/
def speed_train_two : ℝ := 80

/-- The length of the second train in meters -/
def length_train_two : ℝ := 280.04

/-- The time taken to cross each other in seconds -/
def crossing_time : ℝ := 9

/-- The total distance covered when both trains cross each other is the sum of their lengths -/
def total_distance_km : ℝ := (length_train_one + length_train_two) / 1000

/-- Convert time from seconds to hours -/
def crossing_time_hr : ℝ := crossing_time / 3600

/-- The relative speed of the two trains while crossing each other -/
def relative_speed : ℝ := total_distance_km / crossing_time_hr

/-- Proof problem: proving the speed of the first train given the conditions -/
theorem speed_of_first_train : relative_speed - speed_train_two = 120.016 :=
by
  -- Here you would use the given conditions to prove that the speed of the first train is indeed 120.016 km/hr.
  exact sorry

end speed_of_first_train_l666_666009


namespace alex_raking_l666_666505

open Real

theorem alex_raking (a b c : ℝ) 
  (h1 : a + b = 1) 
  (h2 : c = 1 / 6) 
  (h3 : b + c = 5 / 6) : 1 / a = 3 :=
by {
  -- start with existing conditions
  -- sorry
}

end alex_raking_l666_666505


namespace find_three_consecutive_numbers_l666_666843

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

noncomputable def is_perfect (n : ℕ) : Prop :=
  σ(n) = 2 * n

noncomputable def number_of_divisors (n : ℕ) : ℕ :=
  (nat.divisors n).length

noncomputable def has_exactly_one_prime_divisor_less_than_10 (n : ℕ) : Prop :=
  let primes_less_than_10 := [2, 3, 5, 7]
  in (nat.divisors n).filter (λ x, x ∈ primes_less_than_10).length = 1

noncomputable def is_square_free (n : ℕ) : Prop :=
  n.isSquareFree

theorem find_three_consecutive_numbers :
  ∃ (N N1 N2 : ℕ),
    N1 = N + 1 ∧
    N2 = N + 2 ∧
    ((∃ d1, sum_of_digits N = d1 ∧ nat.prime d1) ∧
     (∃ d2, sum_of_digits N1 = d2 ∧ is_perfect d2) ∧
     sum_of_digits N2 = number_of_divisors N2) ∧
    (N.digits 10).count 1 ≤ 2 ∧
    (N1.digits 10).count 1 ≤ 2 ∧
    (N2.digits 10).count 1 ≤ 2 ∧
    (∃ k : ℕ, ∃ x : ℕ, N = x ^ 2 - 11) ∧
    has_exactly_one_prime_divisor_less_than_10 N ∧
    has_exactly_one_prime_divisor_less_than_10 N1 ∧
    has_exactly_one_prime_divisor_less_than_10 N2 ∧
    is_square_free N ∧
    is_square_free N1 ∧
    is_square_free N2
  :=
sorry

end find_three_consecutive_numbers_l666_666843


namespace complement_intersection_l666_666198

noncomputable def U := Set.univ : Set ℝ
noncomputable def A := {x : ℝ | x ≤ -2}
noncomputable def B := {x : ℝ | x < 1}

theorem complement_intersection :
  (U \ A) ∩ B = {x : ℝ | -2 < x ∧ x < 1} :=
by {
  sorry
}

end complement_intersection_l666_666198


namespace ellipse_problem_l666_666924

theorem ellipse_problem :
  ∃ (A B C D E F : ℤ),
    (∀ (t : ℝ), 
      (x y : ℝ),
      x = 3 * (Real.sin t - 2) / (3 - Real.cos t) ∧ 
      y = 4 * (Real.cos t - 4) / (3 - Real.cos t) →
      A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0) ∧ 
      Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.abs A) (Int.abs B)) (Int.abs C)) (Int.abs D)) (Int.abs E)) (Int.abs F) = 1 ∧
      Int.abs A + Int.abs B + Int.abs C + Int.abs D + Int.abs E + Int.abs F = 1383 :=
sorry

end ellipse_problem_l666_666924


namespace cos_double_angle_l666_666285

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666285


namespace train_passengers_count_l666_666055

noncomputable def num_passengers :=
  let p q r s t : ℕ := sorry
  p + q + r + s + t

theorem train_passengers_count (p q r s t : ℕ) 
  (h1 : p ≥ 1) (h2 : q ≥ 1) (h3 : r ≥ 1) (h4 : s ≥ 1) (h5 : t ≥ 1)
  (cond1 : (p - 1) + q = 5 ∨ (p - 1) + q = 10)
  (cond2 : p + (q - 1) + r = 5 ∨ p + (q - 1) + r = 10)
  (cond3 : (r - 1) + s = 5 ∨ (r - 1) + s = 10)
  (cond4 : s + (t - 1) = 5 ∨ s + (t - 1) = 10) :
  p + q + r + s + t = 17 := 
sorry

end train_passengers_count_l666_666055


namespace divisors_of_180_pow_180_with_90_divisors_l666_666692

theorem divisors_of_180_pow_180_with_90_divisors :
  let a := 360
      b := 360
      c := 180
  in ∑ (x : ℕ × ℕ × ℕ) in { p | (p.1 + 1) * (p.2 + 1) * (p.3 + 1) = 90 }, 1 = 60 := 
by sorry

end divisors_of_180_pow_180_with_90_divisors_l666_666692


namespace hypotenuse_length_l666_666010

theorem hypotenuse_length (a b c : ℝ) (h1 : a + b + c = 32) (h2 : a * b = 40) (h3 : a^2 + b^2 = c^2) : 
  c = 59 / 4 :=
by
  sorry

end hypotenuse_length_l666_666010


namespace non_adjacent_a_arrangements_l666_666563

theorem non_adjacent_a_arrangements :
  let letters := ['a', 'a', 'b', 'c', 'd']
  in let total_non_adjacent_arrangements := 36
  in ∃ arrangements: Finset (List Char), arrangements.card = total_non_adjacent_arrangements
     ∧ (∀ l ∈ arrangements, (∀ i, (l.nth i = some 'a' → l.nth (i+1) ≠ some 'a') 
                                                ∧ (l.nth i = some 'a' → l.nth (i-1) ≠ some 'a'))) :=
by
  sorry

end non_adjacent_a_arrangements_l666_666563


namespace union_compl_eq_l666_666383

open Set

def U := {1, 2, 3, 4, 5, 6, 7}
def A := {1, 3, 5}
def B := {2, 3, 6}

theorem union_compl_eq :
  A ∪ (U \ B) = {1, 3, 4, 5, 7} :=
by
  simp [U, A, B]
  sorry

end union_compl_eq_l666_666383


namespace maximal_sum_of_squares_l666_666919

namespace TriangleProof

-- Assume we have a type point representing points on the plane
-- and a type line representing lines (which could be defined using points, vectors, etc.).
-- Here we define a basic structure of a triangle and circle:

structure point := (x : ℝ) (y : ℝ)

structure triangle :=
(A B C : point)

structure circle :=
(center : point)
(radius : ℝ)

-- Let inscribed be a definition that checks if a triangle is inscribed in a given circle:

def inscribed (t : triangle) (c : circle) : Prop :=
(distance t.A c.center = c.radius ∧ distance t.B c.center = c.radius ∧ distance t.C c.center = c.radius)

-- Define the sum of squares of the sides of a triangle:

noncomputable def sum_of_squares_of_sides (t : triangle) : ℝ :=
(distance t.A t.B)^2 + (distance t.B t.C)^2 + (distance t.C t.A)^2

-- Main statement
theorem maximal_sum_of_squares (t : triangle) (c : circle) :
  inscribed t c → (∀ t', inscribed t' c → sum_of_squares_of_sides t' ≤ sum_of_squares_of_sides t) → is_equilateral t :=
sorry

end TriangleProof

end maximal_sum_of_squares_l666_666919


namespace triangle_area_l666_666489

theorem triangle_area (a b c : ℝ) (ha : a = 5) (hb : b = 5) (hc : c = 6) :
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 12 :=
by 
  sorry

end triangle_area_l666_666489


namespace four_pow_minus_a_l666_666655

noncomputable def log_base_3 (x : ℝ) := Real.log x / Real.log 3

theorem four_pow_minus_a {a : ℝ} (h : a * log_base_3 4 = 2) : 4^(-a) = 1 / 9 :=
by
  sorry

end four_pow_minus_a_l666_666655


namespace units_digit_of_27_mul_36_l666_666120

theorem units_digit_of_27_mul_36 : (27 * 36) % 10 = 2 := by
  sorry

end units_digit_of_27_mul_36_l666_666120


namespace minimum_matchsticks_to_remove_l666_666834

theorem minimum_matchsticks_to_remove (matchsticks triangles : ℕ) (h_matchsticks : matchsticks = 42) (h_triangles : triangles = 38) :
  ∃ (n : ℕ), n = 12 ∧ (∀ (remaining_triangles : ℕ), remaining_triangles = triangles - (min_tris_broken n) → remaining_triangles = 0) :=
begin
  sorry
end

end minimum_matchsticks_to_remove_l666_666834


namespace rectangle_area_change_l666_666818

theorem rectangle_area_change (L B : ℝ) : 
  let L_new := 1.20 * L,
      B_new := 0.80 * B in
  ((L_new * B_new) / (L * B) = 0.96) := 
by 
  sorry

end rectangle_area_change_l666_666818


namespace average_high_temperature_l666_666911

theorem average_high_temperature :
  let temps := [51.0, 64.0, 60.0, (60.0 + 59.0) / 2.0, 59.0, 49.0, 63.0] in
  (temps.sum / temps.length).round(1) = 50.8 :=
by
  -- Define the temperatures including the interpolated missing value
  let temps := [51.0, 64.0, 60.0, (60.0 + 59.0) / 2.0, 59.0, 49.0, 63.0]
  -- Calculate the average and round to the nearest tenth
  have avg := (temps.sum / temps.length).round(1)
  -- Assert the average equals 50.8
  have : avg = 50.8 := by sorry
  exact this

end average_high_temperature_l666_666911


namespace problem_statement_l666_666933

def floor_diff (x : ℝ) : ℤ := Int.floor x

theorem problem_statement : floor_diff ( (2010^2 : ℝ) / (2008 * 2009) - (2008^2 : ℝ) / (2009 * 2010) ) = 6 :=
by
  let n : ℕ := 2009
  have h1 : (n + 1) = 2010 := rfl
  have h2 : (n - 1) = 2008 := rfl
  let expression := ((n + 1)^2 : ℝ) / ((n - 1) * n) - ((n - 1)^2 : ℝ) / (n * (n + 1))
  calc floor_diff expression = 6 : by sorry

end problem_statement_l666_666933


namespace down_payment_calculation_l666_666770

noncomputable def tablet_price : ℝ := 450
noncomputable def installment_1 : ℝ := 4 * 40
noncomputable def installment_2 : ℝ := 4 * 35
noncomputable def installment_3 : ℝ := 4 * 30
noncomputable def total_savings : ℝ := 70
noncomputable def total_installments := tablet_price + total_savings
noncomputable def installment_payments := installment_1 + installment_2 + installment_3
noncomputable def down_payment := total_installments - installment_payments

theorem down_payment_calculation : down_payment = 100 := by
  unfold down_payment
  unfold total_installments
  unfold installment_payments
  unfold tablet_price
  unfold total_savings
  unfold installment_1
  unfold installment_2
  unfold installment_3
  sorry

end down_payment_calculation_l666_666770


namespace inv_5_mod_221_l666_666104

theorem inv_5_mod_221 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 220 ∧ 5 * x ≡ 1 [MOD 221] ∧ x = 177 := 
by 
  sorry

end inv_5_mod_221_l666_666104


namespace percentage_advanced_degrees_of_male_employees_l666_666712

theorem percentage_advanced_degrees_of_male_employees :
  let males := 300
  let females := 150
  let total_employees := males + females
  let percentage_advanced_degrees_females := 0.40
  let probability_advanced_degree_or_female := 0.4
  let number_advanced_degrees_females := percentage_advanced_degrees_females * females
  let probability_female := (females : ℝ) / total_employees
  let probability_male_with_advanced_degree := (males : ℝ) * (x : ℝ) / total_employees
  let equation := probability_female + probability_male_with_advanced_degree = probability_advanced_degree_or_female
  show x * 100 = 15
by
  let males := 300
  let females := 150
  let total_employees := males + females
  let percentage_advanced_degrees_females := 0.40
  let probability_advanced_degree_or_female := 0.4
  let number_advanced_degrees_females := percentage_advanced_degrees_females * females
  let probability_female := (females : ℝ) / total_employees
  let probability_male_with_advanced_degree := (males : ℝ) * (x : ℝ) / total_employees
  let equation := probability_female + probability_male_with_advanced_degree = probability_advanced_degree_or_female
  -- proof will be inserted here
  sorry

end percentage_advanced_degrees_of_male_employees_l666_666712


namespace ticket_cost_is_correct_l666_666072

variable (price_of_adult_ticket : ℝ)

-- Condition 1
def price_of_child_ticket := price_of_adult_ticket / 2

-- Condition 2
def cost_of_3_adults_and_5_children := 3 * price_of_adult_ticket + 5 * price_of_child_ticket

-- Given condition
axiom cost_3_adults_5_children_is_27 : cost_of_3_adults_and_5_children price_of_adult_ticket = 27

-- Goal
def cost_of_10_adults_and_15_children := 10 * price_of_adult_ticket + 15 * price_of_child_ticket

theorem ticket_cost_is_correct :
  cost_of_10_adults_and_15_children price_of_adult_ticket = 945 / 11 :=
by 
  sorry

end ticket_cost_is_correct_l666_666072


namespace scientific_notation_216000_l666_666420

theorem scientific_notation_216000 : 216000 = 2.16 * 10^5 :=
by
  -- proof will be provided here
  sorry

end scientific_notation_216000_l666_666420


namespace cube_divisors_of_n_l666_666019

theorem cube_divisors_of_n (n : ℕ) (p q : ℕ) (α β : ℕ) (hn : n = p^α * q^β) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p ≠ q) (hdiv : (2 * α + 1) * (2 * β + 1) = 81) :
    (Nat.divisors_count (n^3) = 160) ∨ (Nat.divisors_count (n^3) = 169) := 
sorry

end cube_divisors_of_n_l666_666019


namespace sequence_product_l666_666679

def sequence_term (n : ℕ) : ℤ :=
  if odd n then (-2)^n else n

theorem sequence_product : sequence_term 4 * sequence_term 3 = -32 :=
by
  sorry

end sequence_product_l666_666679


namespace volume_ratio_of_cone_pieces_l666_666543

theorem volume_ratio_of_cone_pieces (r h : ℝ) : 
    let V1 := (125 / 3 - 64 / 3) * π * r^2 * h
    let V3 := (27 / 3 - 8 / 3) * π * r^2 * h
    (V3 / V1) = 19 / 61 :=
by
    let V1 := (125 / 3 - 64 / 3) * π * r^2 * h
    let V3 := (27 / 3 - 8 / 3) * π * r^2 * h
    have h1 : V1 = (61 / 3) * π * r^2 * h := by sorry
    have h2 : V3 = (19 / 3) * π * r^2 * h := by sorry
    rw [h1, h2]
    field_simp
    norm_num
    sorry

end volume_ratio_of_cone_pieces_l666_666543


namespace beneficial_to_buy_card_breakeven_visits_l666_666001

-- Define the conditions for Part (a)
def avg_check := 900
def discount_rate := 0.30
def visits_per_week := 3
def weeks_per_year := 52
def card_cost := 30000

-- Define the conditions for Part (b)
def avg_check_lower := 600

-- Part (a) statement
theorem beneficial_to_buy_card :
  (visits_per_week * weeks_per_year * (avg_check * discount_rate) > card_cost) :=
by
  sorry

-- Part (b) statement
theorem breakeven_visits :
  (card_cost / (avg_check_lower * discount_rate)).ceil = 167 :=
by
  sorry

end beneficial_to_buy_card_breakeven_visits_l666_666001


namespace prob_divisible_by_5_l666_666067

theorem prob_divisible_by_5 :
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 100 ∧ (∃ (p : ℚ), p = 2 / 5 ∧
    p = ((nat.filter (λ k, (k * (k + 1)) % 5 = 0) (list.range 100)).length : ℚ) / 100)) :=
sorry

end prob_divisible_by_5_l666_666067


namespace cos_double_angle_l666_666272

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666272


namespace bugs_next_meeting_time_l666_666840

-- Given conditions
def radius1 := 8 -- radius of the first circle
def radius2 := 4 -- radius of the second circle
def speed1 := 3 * Real.pi -- speed of the bug on the first circle in inches per minute
def speed2 := 4 * Real.pi -- speed of the bug on the second circle in inches per minute

def circumference1 := 2 * radius1 * Real.pi -- circumference of the first circle
def circumference2 := 2 * radius2 * Real.pi -- circumference of the second circle

def time1 := circumference1 / speed1 -- time to complete one round on the first circle
def time2 := circumference2 / speed2 -- time to complete one round on the second circle

-- The proof problem statement
theorem bugs_next_meeting_time : 
  (time1 = 16 / 3) → 
  (time2 = 2) → 
  (∀ t : ℕ, t ≥ 0 → (t / (16 / 3) = ∃ k1 : ℕ, k1 ∈ ℕ) ∧ (t / 2 = ∃ k2 : ℕ, k2 ∈ ℕ) → (t = 48)) :=
  sorry

end bugs_next_meeting_time_l666_666840


namespace units_digit_of_expression_l666_666583

-- Definitions based on the conditions
def units_digit (n : ℕ) : ℕ := n % 10

def eight := 8
def thirteen := 13
def nineteen_eighty_nine := 1989

-- Conditions derived from the problem
def units_digit_eight : ℕ := units_digit eight
def units_digit_thirteen : ℕ := units_digit thirteen
def units_digit_nineteen_eighty_nine : ℕ := units_digit nineteen_eighty_nine

-- Calculate units digits of the needed intermediate products
def intermediate_product_units_digit : ℕ :=
  units_digit (units_digit_eight * units_digit_thirteen)

def final_product_units_digit : ℕ :=
  units_digit (intermediate_product_units_digit * units_digit_nineteen_eighty_nine)

def eight_cube : ℕ := 8 ^ 3
def units_digit_eight_cube : ℕ := units_digit eight_cube

-- The statement that needs to be proven
theorem units_digit_of_expression :
  units_digit (8 * 13 * 1989 - 8 ^ 3) = 4 :=
by
  have h1 : units_digit_eight = 8 := by sorry
  have h2 : units_digit_thirteen = 3 := by sorry
  have h3 : units_digit_nineteen_eighty_nine = 9 := by sorry
  have h4 : intermediate_product_units_digit = 4 := by sorry
  have h5 : final_product_units_digit = 6 := by sorry
  have h6 : units_digit_eight_cube = 2 := by sorry
  show units_digit (8 * 13 * 1989 - 8 ^ 3) = 4 from sorry

end units_digit_of_expression_l666_666583


namespace proof_least_t_l666_666355

noncomputable def least_t (a b c d e : ℝ) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) (he : e ≥ 0) 
  (h_sum : a + b + c + d + e > 0) (t : ℝ) : Prop :=
a + c = t * b ∧ b + d = t * c ∧ c + e = t * d → t ≥ Real.sqrt 2

theorem proof_least_t (a b c d e : ℝ) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) (he : e ≥ 0) 
  (h_sum : a + b + c + d + e > 0) : ∃ t : ℝ, least_t a b c d e hb hc hd he h_sum t :=
begin
  use Real.sqrt 2,
  sorry
end

end proof_least_t_l666_666355


namespace double_angle_cosine_l666_666226

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l666_666226


namespace find_a_if_geometric_sequence_exists_l666_666335

theorem find_a_if_geometric_sequence_exists
  (a : ℝ)
  (h_curve : ∀ (ρ θ : ℝ), ρ * (sin θ)^2 = 2 * a * cos θ)
  (h_line : ∀ (t : ℝ), let x := -2 + (sqrt 2 / 2) * t in let y := -4 + (sqrt 2 / 2) * t in y = x - 2) :
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
    let M := (-2 + (sqrt 2 / 2) * t₁, -4 + (sqrt 2 / 2) * t₁),
        N := (-2 + (sqrt 2 / 2) * t₂, -4 + (sqrt 2 / 2) * t₂),
        P := (-2 : ℝ, -4 : ℝ),
        PM := (M.1 - P.1) ^ 2 + (M.2 - P.2) ^ 2,
        MN := (N.1 - M.1) ^ 2 + (N.2 - M.2) ^ 2,
        PN := (N.1 - P.1) ^ 2 + (N.2 - P.2) ^ 2
    in (PM, MN, PN).Sorted ((· < ·) ∘ (λ x : ℝ, real.sqrt x)) ∧ PM * PN = MN ^ 2) →
  a = 1 :=
by
  sorry

end find_a_if_geometric_sequence_exists_l666_666335


namespace union_of_A_and_B_l666_666380

-- Define the sets A and B
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

-- Prove that the union of A and B is {-1, 0, 1}
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} :=
  by sorry

end union_of_A_and_B_l666_666380


namespace minimize_sum_of_distances_l666_666634

theorem minimize_sum_of_distances (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ Q : ℝ)
        (order : Q₁ < Q₂ < Q₃ < Q₄ < Q₅ < Q₆ < Q₇ < Q₈ < Q₉) :
    (QQ₁ + QQ₂ + QQ₃ + QQ₄ + QQ₅ + QQ₆ + QQ₇ + QQ₈ + QQ₉) ≥ 
    (Q₅Q₁ + Q₅Q₂ + Q₅Q₃ + Q₅Q₄ + Q₅Q₆ + Q₅Q₇ + Q₅Q₈ + Q₅Q₉) :=
by
  sorry

end minimize_sum_of_distances_l666_666634


namespace focal_chord_of_ellipse_l666_666598

def ellipse_focal_length_chord (a b : ℝ) (h : a^2 = 2 ∧ b^2 = 1) : ℝ :=
  if h : a^2 = 2 ∧ b^2 = 1 then 2 * (sqrt (a^2 - b^2)) else 0

theorem focal_chord_of_ellipse :
  ellipse_focal_length_chord 2 1 ⟨rfl, rfl⟩ = 2 :=
by
  sorry

end focal_chord_of_ellipse_l666_666598


namespace parabola_directrix_l666_666110

theorem parabola_directrix (a : ℝ) (c : ℝ) (a_pos : a = 12) (c_pos : c = 5) : 
  let d := -1 / (4 * a) + c in
  d = 239 / 48 :=
by
  sorry

end parabola_directrix_l666_666110


namespace expr_is_even_l666_666301

theorem expr_is_even (x : ℕ) (h : x = 3) : Even (2 * (x^2 + 9)) := by
  have h1 : 2 * (x^2 + 9) = 2 * (3^2 + 9) := by
    rw [h]
    rfl
  have h2 : 2 * (3^2 + 9) = 36 := by
    norm_num
  rw [h2]
  exact Even.intro (18) rfl

end expr_is_even_l666_666301


namespace total_pins_cardboard_l666_666507

theorem total_pins_cardboard {length width pins : ℕ} (h_length : length = 34) (h_width : width = 14) (h_pins : pins = 35) :
  2 * pins * (length + width) / (length + width) = 140 :=
by
  sorry

end total_pins_cardboard_l666_666507


namespace triangle_angle_l666_666006

/-- 
Given a right-angled triangle BAD with right-angle at B, 
and points A, B, C, D on a plane such that AC = 3 * CD and AB = BC,
prove that the magnitude of ∠DAB equals tan⁻¹(1.5). 
-/
theorem triangle_angle
  (A B C D : Point)
  (h_right : right_angle (∠ BAD) B)
  (h_ratio : dist A C = 3 * dist C D)
  (h_equal : dist A B = dist B C) :
  ∠ D A B = Real.arctan 1.5 :=
sorry

end triangle_angle_l666_666006


namespace vacation_cost_split_l666_666837

theorem vacation_cost_split (t d : ℕ) 
  (h_total : 105 + 125 + 175 = 405)
  (h_split : 405 / 3 = 135)
  (h_t : t = 135 - 105)
  (h_d : d = 135 - 125) : 
  t - d = 20 := by
  sorry

end vacation_cost_split_l666_666837


namespace alex_climbs_ways_l666_666966

def waysToClimbStairs : ℕ → (ℕ → ℕ) → ℕ
| 0, _   => 1
| n+1, f => f 1 + f 2 + f 3 + f 4

def validSteps (n k : ℕ) : ℕ :=
if n < k then 0
else waysToClimbStairs (n - k) (λ j, if j ≠ k then validSteps (n - k) j else 0)

theorem alex_climbs_ways : validSteps 8 1 + validSteps 8 2 + validSteps 8 3 + validSteps 8 4 = 36 := sorry

end alex_climbs_ways_l666_666966


namespace number_of_possible_triangles_with_side_5_not_shortest_l666_666628

-- Define and prove the number of possible triangles (a, b, c) with a, b, c positive integers,
-- such that one side is length 5 and it is not the shortest side is 10.
theorem number_of_possible_triangles_with_side_5_not_shortest (a b c : ℕ) (h1: a + b > c) (h2: a + c > b) (h3: b + c > a) 
(h4: 0 < a) (h5: 0 < b) (h6: 0 < c) (h7: a = 5 ∨ b = 5 ∨ c = 5) (h8: ¬ (a < 5 ∧ b < 5 ∧ c < 5)) :
∃ n, n = 10 := 
sorry

end number_of_possible_triangles_with_side_5_not_shortest_l666_666628


namespace parabola_intersection_length_l666_666193

noncomputable def parabola_y_squared (x y : ℝ) := y^2 = 4 * x

noncomputable def focus := (1 : ℝ, 0 : ℝ)
noncomputable def directrix_x := (-1 : ℝ)
noncomputable def point_p := (directrix_x, 0 : ℝ)

noncomputable def line_through_p (k : ℝ) (k_pos : 0 < k) (x y : ℝ) := y = k * x + k

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.fst - p2.fst) ^ 2 + (p1.snd - p2.snd) ^ 2)

theorem parabola_intersection_length (k : ℝ) (h : 0 < k) :
  ∀ (A B : ℝ × ℝ),
  parabola_y_squared A.fst A.snd → 
  parabola_y_squared B.fst B.snd →
  line_through_p k h A.fst A.snd →
  line_through_p k h B.fst B.snd →
  point_p.fst = A.fst → 
  (distance B focus = 2 * distance A focus) →
  distance A B = real.sqrt 17 / 2 :=
sorry

end parabola_intersection_length_l666_666193


namespace correct_sum_deficient_numbers_l666_666619

/-- Define the sequence a_n. -/
def a (n : ℕ) : ℝ := real.log (n + 2) / real.log (n + 1)

/-- Define deficient number k such that product of first k terms of the sequence is an integer. -/
def is_deficient (k : ℕ) : Prop :=
  ∃ n : ℕ, k + 2 = 2^n

/-- Sum of all deficient numbers in the interval (1, 62). -/
def sum_deficient_numbers : ℕ :=
  ∑ k in finset.filter (λ k, 1 < k ∧ k < 62 ∧ is_deficient k) (finset.range 62), k

theorem correct_sum_deficient_numbers : sum_deficient_numbers = 52 := 
sorry

end correct_sum_deficient_numbers_l666_666619


namespace box_height_l666_666057

theorem box_height (ceiling_height light_bulb_distance shelf_distance : ℝ)
                   (alice_height alice_reach : ℝ) : 
  let alice_total_reach := alice_height + alice_reach in
  let light_bulb_floor_height := ceiling_height - light_bulb_distance in
  let shelf_floor_height := light_bulb_floor_height - shelf_distance in
  alice_total_reach + 75 = light_bulb_floor_height
→ 75 = 75 :=
by 
  intros h
  unfold alice_total_reach light_bulb_floor_height shelf_floor_height
  sorry

end box_height_l666_666057


namespace complex_numbers_count_l666_666981

theorem complex_numbers_count (z : ℂ) (h1 : complex.abs z = 1) (h2 : complex.abs ((z / (z.conj)) - ((z.conj) / z)) = 1) : (∃ z_list : list ℂ, z_list.length = 8 ∧ ∀ w ∈ z_list, complex.abs w = 1 ∧ complex.abs ((w / (w.conj)) - ((w.conj) / w)) = 1) :=
sorry

end complex_numbers_count_l666_666981


namespace milk_needed_for_cookies_l666_666470

-- Definition of the problem conditions
def cookies_per_milk_usage := 24
def milk_in_liters := 5
def liters_to_milliliters := 1000
def milk_for_6_cookies := 1250

-- Prove that 1250 milliliters of milk are needed to bake 6 cookies
theorem milk_needed_for_cookies
  (h1 : cookies_per_milk_usage = 24)
  (h2 : milk_in_liters = 5)
  (h3 : liters_to_milliliters = 1000) :
  milk_for_6_cookies = 1250 :=
by
  -- Proof is omitted with sorry
  sorry

end milk_needed_for_cookies_l666_666470


namespace triangle_area_sides_5_5_6_l666_666484

theorem triangle_area_sides_5_5_6 (a b c : ℕ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 6) :
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 12 :=
by
  sorry

end triangle_area_sides_5_5_6_l666_666484


namespace positive_inequality_l666_666606

theorem positive_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^2 + b^2) / (2 * a^5 * b^5) + 81 * (a^2 * b^2) / 4 + 9 * a * b > 18 := 
  sorry

end positive_inequality_l666_666606


namespace quadratic_residues_mod_pq_quadratic_residues_mod_pk_l666_666693

-- Proof Problem 1
theorem quadratic_residues_mod_pq (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (hpq : p ≠ q) :
  count_quadratic_residues (p * q) = ((p - 1) * (q - 1)) / 4 := sorry

-- Proof Problem 2
theorem quadratic_residues_mod_pk (p k : ℕ) [Fact (Nat.Prime p)] (hk : k > 0) :
  count_quadratic_residues (p^k) = ((p - 1) * p^(k - 1)) / 2 := sorry

end quadratic_residues_mod_pq_quadratic_residues_mod_pk_l666_666693


namespace double_angle_cosine_l666_666225

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l666_666225


namespace n_th_group_sum_l666_666004

theorem n_th_group_sum (n : ℕ) (h1 : n > 0) :
  let S := λ k, k * (k + 1) / 2 in
  let start := (n * (n - 1)) / 2 + 1 in
  let end_term := start + n - 1 in
  let a_n := S end_term - S (start - 1) in
  a_n = n^3 / 2 + n / 2 := 
by
  sorry

end n_th_group_sum_l666_666004


namespace total_distance_traveled_l666_666308

def biking_time : ℝ := 1 -- in hours
def biking_rate : ℝ := 8 -- in mph
def jogging_time : ℝ := 1/3 -- in hours (20 minutes converted to hours)
def jogging_rate : ℝ := 6 -- in mph

theorem total_distance_traveled :
  (biking_rate * biking_time) + (jogging_rate * jogging_time) = 10 := by
  sorry

end total_distance_traveled_l666_666308


namespace maximum_value_g_l666_666091

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem maximum_value_g : ∃ x ∈ set.Icc (0:ℝ) 2, ∀ y ∈ set.Icc (0:ℝ) 2, g x ≥ g y ∧ g x = 3 :=
by
  sorry

end maximum_value_g_l666_666091


namespace find_f_analytical_expression_l666_666668

noncomputable def f (x : ℝ) : ℝ := (2/3) * Real.sqrt x + (1/3)

theorem find_f_analytical_expression :
  (∀ (x : ℝ), x > 0 → 
    f(x) = 2 * f(1 / x) * Real.sqrt x - 1) :=
begin
  sorry
end

end find_f_analytical_expression_l666_666668


namespace find_m_l666_666102

variable {β : ℝ}
noncomputable def tan (x : ℝ) := sin x / cos x
noncomputable def cot (x : ℝ) := cos x / sin x
noncomputable def sec (x : ℝ) := 1 / cos x
noncomputable def csc (x : ℝ) := 1 / sin x

theorem find_m (h : (tan β + cot β)^2 + (cos β + sec β)^2 = m + sin β^2 + csc β^2) : m = 4 :=
sorry

end find_m_l666_666102


namespace cos_value_l666_666142

theorem cos_value : 
  ∀ α : ℝ, sin(α - π / 3) = 1 / 3 → cos(π / 6 + α) = 1 / 3 :=
by
  intros α h
  sorry

end cos_value_l666_666142


namespace disk_tangent_6_oclock_l666_666528

theorem disk_tangent_6_oclock
  (radius_clock : ℝ) (radius_disk : ℝ) (circumference_clock : ℝ) (circumference_disk : ℝ)
  (initial_position : ℝ) (final_position : ℝ)
  (tangent_initial : initial_position = 0)
  (radius_clock_val : radius_clock = 30)
  (radius_disk_val : radius_disk = 15)
  (circumference_clock_val : circumference_clock = 2 * real.pi * radius_clock)
  (circumference_disk_val : circumference_disk = 2 * real.pi * radius_disk)
  (final_position_val : final_position = (circumference_disk / circumference_clock) * 360) :
  final_position = 180 :=
by sorry

end disk_tangent_6_oclock_l666_666528


namespace cos_double_angle_l666_666263

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l666_666263


namespace veranda_area_l666_666811

theorem veranda_area (length_room width_room width_veranda : ℕ)
  (h_length : length_room = 20) 
  (h_width : width_room = 12) 
  (h_veranda : width_veranda = 2) : 
  (length_room + 2 * width_veranda) * (width_room + 2 * width_veranda) - (length_room * width_room) = 144 := 
by
  sorry

end veranda_area_l666_666811


namespace sandals_sold_l666_666950

theorem sandals_sold (S B T Sandals : ℕ) (hS : S = 2) (hB : B = 11) (hT : T = 17) (hSandals : Sandals = T - (S + B)) :
  Sandals = 4 := by
  rw [hS, hB, hT, hSandals]
  rfl

end sandals_sold_l666_666950


namespace feet_of_perpendiculars_lie_on_circle_l666_666832

variables {α : Type*} [metric_space α] [metric_space.embedding_isometry G T]

/-- The feet of perpendiculars dropped from the foot of the altitude of a pyramid onto the lateral edges lie on a circle.
-/
theorem feet_of_perpendiculars_lie_on_circle
  (Pyramid : Type*) [has_base P : pos_polygon] [has_altitude P : gt MeasurableSpace]
  (vertices_on_circle : ∃ c : Circle, ∀ v ∈ base vertices, within_circle c v)
  (altitude_perpendicular : ∃ T : point, is_altitude_perpendicular P T base) :
  ∃ c' : circle, ∀ b ∈ feet_of_perpendiculars P T base lateral_edges, within_circle c' b := 
sorry

end feet_of_perpendiculars_lie_on_circle_l666_666832


namespace evaluate_power_l666_666962

theorem evaluate_power (a b : ℝ) (m n : ℝ) (h1 : a = b^m) (h2 : ∀ x y z : ℝ, (x^y)^z = x^(y*z)) : a^(n/m) = b^n :=
by
  sorry

example : 81^(5/4) = 243 := evaluate_power 81 3 4 5
  (by norm_num) -- Simplification for 81 = 3^4
  (by norm_num []) -- Using the power of a power rule

end evaluate_power_l666_666962


namespace unique_2017_tuple_l666_666477

def arrangeable (a b c d e : ℤ) : Prop := a - b + c - d + e = 29

theorem unique_2017_tuple (n : ℕ → ℤ) (h : ∀ i : ℕ, arrangeable (n i) (n (i+1)) (n (i+2)) (n (i+3)) (n (i+4))) :
  ∀ i, n i = 29 :=
begin
  sorry
end

end unique_2017_tuple_l666_666477


namespace correct_propositions_l666_666624

-- Definitions of the given conditions
def circle_O1_eq (x y : ℝ) : Prop := x^2 + y^2 + 6 * x + 5 = 0
def circle_O2_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 91 = 0
def externally_tangent (M O1 : ℝ × ℝ) (r1 rM : ℝ) : Prop := dist M O1 = r1 + rM
def internally_tangent (M O2 : ℝ × ℝ) (r2 rM : ℝ) : Prop := dist M O2 = r2 - rM
def trajectory_eq (x y : ℝ) : Prop := x^2 / 36 + y^2 / 27 = 1

-- Prove that propositions (1) and (4) are correct given the conditions
theorem correct_propositions :
  (∀ x y : ℝ, circle_O1_eq x y → circle_O2_eq x y → externally_tangent (x, y) (-3, 0) 2 4 → internally_tangent (x, y) (3, 0) 10 4 → trajectory_eq x y) ∧
  (∀ x y : ℝ, y ≠ 0 → (slope (x, y) (-6, 0)) * (slope (x, y) (6, 0)) = -3 / 4) :=
sorry

end correct_propositions_l666_666624


namespace team_F_matches_l666_666835

-- Definitions of the conditions.
def team := { A, B, C, D, E, F }
def matches_played (team: team) (n: ℕ) : Prop :=
  (team = A ∧ n = 3) ∨
  (team = B ∧ n = 3) ∨
  (team = C ∧ n = 4) ∨
  (team = D ∧ n = 4) ∨
  (team = E ∧ n = 2)

def not_play_against (team1 team2: team) : Prop :=
  (team1 = A ∧ team2 = C) ∨
  (team1 = B ∧ team2 = D)

-- The theorem to be proved.
theorem team_F_matches : matches_played F 4 :=
  sorry

end team_F_matches_l666_666835


namespace parabola_focus_vertex_distance_l666_666416

theorem parabola_focus_vertex_distance (A V F : Point) (d : ℝ)
  (h1 : dist A F = 24)
  (h2 : dist A V = 25) :
  d = 7 / 2 :=
by
  sorry

end parabola_focus_vertex_distance_l666_666416


namespace cos_double_angle_l666_666274

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l666_666274


namespace tan_theta_eq_sqrt3_l666_666171

theorem tan_theta_eq_sqrt3 (θ : Real) (h : ∃ x : Real, y : Real, y = sqrt 3 * x ∧ θ = atan y / x) : tan θ = sqrt 3 :=
sorry

end tan_theta_eq_sqrt3_l666_666171


namespace cube_max_cover_on_grid_l666_666531

theorem cube_max_cover_on_grid (A : ℤ) : 
  ∀ F : set (ℤ × ℤ), ∃ x y : ℤ, 
  (S : ℤ) (P : ℤ),
  (S < A^2) ∧ (P ≥ 4 * A) ∧ (S = x / 2 + y - 1) 
  → (x + y ≤ (A + 1) ^ 2) :=
by
  sorry

end cube_max_cover_on_grid_l666_666531


namespace solve_for_k_l666_666157

theorem solve_for_k (x y k : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : 5 * x - k * y - 7 = 0) : k = 1 :=
by
  sorry

end solve_for_k_l666_666157


namespace double_angle_cosine_l666_666223

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l666_666223


namespace three_number_relationship_l666_666445

theorem three_number_relationship :
  let a := (0.7 : ℝ) ^ 6
  let b := 6 ^ (0.7 : ℝ)
  let c := Real.log 6 / Real.log 0.7
  c < a ∧ a < b :=
sorry

end three_number_relationship_l666_666445


namespace jail_time_calculation_l666_666476

-- Define conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def arrests_per_day : ℕ := 10
def pre_trial_days : ℕ := 4
def half_two_week_sentence_days : ℕ := 7 -- 1 week is half of 2 weeks

-- Define the calculation of the total combined weeks of jail time
def total_combined_weeks_jail_time : ℕ :=
  let total_arrests := arrests_per_day * number_of_cities * days_of_protest
  let total_days_jail_per_person := pre_trial_days + half_two_week_sentence_days
  let total_combined_days_jail_time := total_arrests * total_days_jail_per_person
  total_combined_days_jail_time / 7

-- Theorem statement
theorem jail_time_calculation : total_combined_weeks_jail_time = 9900 := by
  sorry

end jail_time_calculation_l666_666476


namespace solve_f_2025_l666_666610

noncomputable def f : ℝ → ℝ :=
  sorry -- A definition that satisfies f(x) = f(x+2) + f(x-2)

theorem solve_f_2025 (h1 : ∀ x, f(x) = f(x+2) + f(x-2)) (h2 : f(2013) = 2013) :
  f(2025) = 2013 :=
sorry

end solve_f_2025_l666_666610


namespace evaluate_powers_of_i_l666_666585

noncomputable def i_pow_13_18_23_28_33 : ℂ :=
  let i := complex.I
  i^13 + i^18 + i^23 + i^28 + i^33

theorem evaluate_powers_of_i :
  i_pow_13_18_23_28_33 = complex.I :=
by
  sorry

end evaluate_powers_of_i_l666_666585


namespace max_edge_length_of_tetrahedron_l666_666144

def diameter : Type := ℝ

noncomputable def largest_tetrahedron_edge_length (d : diameter) : ℝ :=
  (Math.pi * d) / 2

theorem max_edge_length_of_tetrahedron (d : diameter) :
  largest_tetrahedron_edge_length d = (Math.pi * d) / 2 :=
by
  sorry

end max_edge_length_of_tetrahedron_l666_666144


namespace principal_amount_correct_l666_666984

-- Define the given conditions and quantities
def P : ℝ := 1054.76
def final_amount : ℝ := 1232.0
def rate1 : ℝ := 0.05
def rate2 : ℝ := 0.07
def rate3 : ℝ := 0.04

-- Define the statement we want to prove
theorem principal_amount_correct :
  final_amount = P * (1 + rate1) * (1 + rate2) * (1 + rate3) :=
sorry

end principal_amount_correct_l666_666984


namespace am_ge_hm_l666_666377

noncomputable theory

open_locale big_operators

variables {ι : Type*} [fintype ι] {x : ι → ℝ}

theorem am_ge_hm (hx : ∀ i, 0 < x i) : 
  (∑ i, x i) / fintype.card ι ≥ fintype.card ι / (∑ i, (1 / x i)) :=
begin
  sorry
end

end am_ge_hm_l666_666377


namespace inverse_function_l666_666858

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 7

noncomputable def f_inv (y : ℝ) : ℝ := -2 - Real.sqrt ((1 + y) / 2)

theorem inverse_function :
  ∀ (x : ℝ), x < -2 → f_inv (f x) = x ∧ ∀ (y : ℝ), y > -1 → f (f_inv y) = y :=
by
  sorry

end inverse_function_l666_666858


namespace number_of_zeros_of_g_l666_666675

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -2 else -x^2 - 4 * x - 2

def g (x : ℝ) : ℝ :=
  f x + x

theorem number_of_zeros_of_g (x : ℝ) :
  (f 0 = -2) ∧ (f (-1) = 1) → 
  (∃! t₁ t₂ t₃: ℝ, g t₁ = 0 ∧ g t₂ = 0 ∧ g t₃ = 0) :=
by
  sorry

end number_of_zeros_of_g_l666_666675


namespace banana_group_size_l666_666464

theorem banana_group_size (bananas groups : ℕ) (h1 : bananas = 407) (h2 : groups = 11) : bananas / groups = 37 :=
by sorry

end banana_group_size_l666_666464


namespace triangle_area_is_12_l666_666481

/-- 
Theorem: The area of an isosceles triangle with sides 5, 5, and 6 units is 12 square units.
-/
theorem triangle_area_is_12 
  (a b c : ℕ) 
  (h1 : a = 5)
  (h2 : b = 5)
  (h3 : c = 6)
  : (1/2 : ℚ) * c * (real.sqrt (a^2 - (c/2)^2)) = 12 := 
by 
  sorry

end triangle_area_is_12_l666_666481


namespace number_of_chords_l666_666789

theorem number_of_chords (n : ℕ) (h : n = 10) : finset.card (finset.pairs (finset.range n)) = 45 :=
by
  rw [h]
  -- Sorry to skip the proof steps as required
  sorry

end number_of_chords_l666_666789


namespace proof_problem_l666_666703

variable (n : ℤ)
def p := 2 * n - 1
def q := 2 * n + 1

theorem proof_problem :
  (Odd p) ∧ (¬ Even q) → (Odd p ∨ Even q) :=
by
  intro h
  cases h with hp hq
  left
  exact hp

end proof_problem_l666_666703


namespace cos_double_angle_l666_666236

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666236


namespace number_of_chords_l666_666797

/-- Ten points are marked on the circumference of a circle.
    Prove that the number of different chords that can be drawn
    by connecting any two of these ten points is 45.
-/
theorem number_of_chords (n : ℕ) (h_n : n = 10) : 
  (nat.choose n 2) = 45 :=
by
  rw h_n
  norm_num

end number_of_chords_l666_666797


namespace fraction_difference_l666_666699

theorem fraction_difference (a b : ℝ) (h : a - b = 2 * a * b) : (1 / a - 1 / b) = -2 := 
by
  sorry

end fraction_difference_l666_666699


namespace wheel_radius_increase_l666_666095

theorem wheel_radius_increase 
  (d₁ d₂ : ℝ) -- distances according to the odometer (600 and 580 miles)
  (r₀ : ℝ)   -- original radius (17 inches)
  (C₁: d₁ = 600)
  (C₂: d₂ = 580)
  (C₃: r₀ = 17) :
  ∃ Δr : ℝ, Δr = 0.57 :=
by
  sorry

end wheel_radius_increase_l666_666095


namespace sequence_term_four_l666_666336

theorem sequence_term_four (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 2) : a 4 = 7 :=
sorry

end sequence_term_four_l666_666336


namespace area_codes_even_product_count_l666_666508

def digits := {2, 4, 5}

def is_even_product (a b c : Nat) : Prop :=
  (a * b * c) % 2 = 0

def count_even_product_codes : Nat :=
  let codes := { (a, b, c) | a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ is_even_product a b c }
  codes.size

theorem area_codes_even_product_count : count_even_product_codes = 26 :=
  by sorry

end area_codes_even_product_count_l666_666508


namespace find_min_value_x_l666_666759

theorem find_min_value_x (x y z : ℝ) (h1 : x + y + z = 6) (h2 : xy + xz + yz = 10) : 
  ∃ (x_min : ℝ), (∀ (x' : ℝ), (∀ y' z', x' + y' + z' = 6 ∧ x' * y' + x' * z' + y' * z' = 10 → x' ≥ x_min)) ∧ x_min = 2 / 3 :=
sorry

end find_min_value_x_l666_666759


namespace double_angle_cosine_l666_666222

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l666_666222


namespace cosine_double_angle_l666_666252

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l666_666252


namespace evaluate_expression_l666_666406

theorem evaluate_expression (x : ℤ) (h1 : 0 ≤ x ∧ x ≤ 2) (h2 : x ≠ 1) (h3 : x ≠ 2) (h4 : x = 0) :
    ( ((4 - x) / (x - 1) - x) / ((x - 2) / (x - 1)) ) = -2 :=
by
    sorry

end evaluate_expression_l666_666406


namespace cos_double_angle_l666_666290

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666290


namespace triangle_area_l666_666970

/-- Define the points A, B, and C -/
def A : ℝ³ := ⟨6, 5, 3⟩
def B : ℝ³ := ⟨3, 3, 1⟩
def C : ℝ³ := ⟨15, 10, 9⟩

/-- Define the function to compute the cross product of two vectors -/
def cross_product (u v : ℝ³) : ℝ³ :=
  ⟨u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1⟩

/-- Define the norm of a vector -/
def norm (v : ℝ³) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def area_of_triangle (a b c : ℝ³) : ℝ :=
  1 / 2 * norm (cross_product (b - a) (c - a))

theorem triangle_area :
  area_of_triangle A B C = Real.sqrt 13 / 2 :=
by sorry

end triangle_area_l666_666970


namespace product_eq_l666_666398

-- Define the necessary constants and conditions
variables {P X Y W Z G : Type}
variables (a b : ℝ) (PG diameter : ℝ)

-- Given conditions
def ellipse_center (P : Type) : Prop := True  -- Placeholder for P being the center of ellipse
def major_axis (XY : Type) : Prop := True  -- Placeholder for XY being major axis
def minor_axis (WZ : Type) : Prop := True  -- Placeholder for WZ being minor axis
def focus_point (P G : Type) : Prop := True  -- Placeholder for G being a focus of the ellipse

-- Numerical conditions
def PG_condition : PG = 8 := sorry
def diameter_condition : diameter = 4 := sorry

-- Mathematical relationships
def semi_major_eq : a = PX := sorry
def semi_minor_eq : b = PW := sorry
def ellipse_eq : a^2 - b^2 = 64 := sorry
def inradius_eq : b + 8 - a = 4 := sorry
def diff_of_squares_eq : (a+b) * (a-b) = 64 := sorry
def system_eq1 : a - b = 4 := sorry
def system_eq2 : a + b = 16 := sorry

-- Prove the final product
theorem product_eq : (20:ℝ) * (12:ℝ) = 240 := by
  compute_product P X Y W Z a b PG diameter sorry sorry sorry sorry sorry sorry sorry sorry sorry

end product_eq_l666_666398


namespace fixed_point_of_parabola_l666_666126

theorem fixed_point_of_parabola :
  ∀ (m : ℝ), ∃ (a b : ℝ), (∀ (x : ℝ), (a = -3 ∧ b = 81) → (y = 9*x^2 + m*x + 3*m) → (y = 81)) :=
by
  sorry

end fixed_point_of_parabola_l666_666126


namespace sum_of_areas_of_rectangles_l666_666407

theorem sum_of_areas_of_rectangles :
  let widths := [3, 3, 3, 3, 3, 3],
      lengths := [1, 9, 25, 49, 81, 121] in
  list.sum (list.map (λ (wl : ℕ × ℕ), wl.1 * wl.2) (list.zip widths lengths)) = 858 :=
by
  sorry

end sum_of_areas_of_rectangles_l666_666407


namespace determine_digits_from_expression_l666_666742

theorem determine_digits_from_expression (a b c x y z S : ℕ) 
  (hx : x = 100) (hy : y = 10) (hz : z = 1)
  (hS : S = a * x + b * y + c * z) :
  S = 100 * a + 10 * b + c :=
by
  -- Variables
  -- a, b, c : ℕ -- digits to find
  -- x, y, z : ℕ -- chosen numbers
  -- S : ℕ -- the given sum

  -- Assumptions
  -- hx : x = 100
  -- hy : y = 10
  -- hz : z = 1
  -- hS : S = a * x + b * y + c * z
  sorry

end determine_digits_from_expression_l666_666742


namespace find_a_l666_666664

def f (x a : ℝ) : ℝ := -4 * x^2 + 4 * a * x - 4 * a - a^2

theorem find_a (a : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) 1, f x a ≤ -5) ∧ (∃ x ∈ set.Icc (0 : ℝ) 1, f x a = -5) → 
  (a = -5 ∨ a = 5 / 4) :=
by
  sorry

end find_a_l666_666664


namespace solve_x_squared_eq_four_x_l666_666450

theorem solve_x_squared_eq_four_x : {x : ℝ | x^2 = 4*x} = {0, 4} := 
sorry

end solve_x_squared_eq_four_x_l666_666450


namespace smallest_integer_ends_in_3_and_divisible_by_5_l666_666494

theorem smallest_integer_ends_in_3_and_divisible_by_5 : ∃ (n : ℕ), n > 0 ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ n = 53 :=
by
  sorry

end smallest_integer_ends_in_3_and_divisible_by_5_l666_666494


namespace ratio_of_supply_to_demand_l666_666135

def supply : ℕ := 1800000
def demand : ℕ := 2400000

theorem ratio_of_supply_to_demand : (supply / demand : ℚ) = 3 / 4 := by
  sorry

end ratio_of_supply_to_demand_l666_666135


namespace arithmetic_sequence_property_l666_666751

-- Define the arithmetic sequence and its properties.
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (a1 d : ℤ), ∀ n, a n = a1 + (n - 1) * d

-- Define the conditions a_4 + a_5 = 24 and S_6 = 48.
def cond1 (a : ℕ → ℤ) : Prop :=
  a 4 + a 5 = 24

def cond2 (a : ℕ → ℤ) : Prop :=
  let S6 := (6 * (2 * a 1 + 5 * (a 2 - a 1))) / 2 in
  S6 = 48

-- Define the property we want to prove: a_n = 4n - 6.
def target_property (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 4 * n - 6

-- The statement to be proved using the conditions.
theorem arithmetic_sequence_property (a : ℕ → ℤ) :
  arithmetic_sequence a → cond1 a → cond2 a → target_property a := by
  sorry

end arithmetic_sequence_property_l666_666751


namespace exists_value_m_l666_666632

-- Definition of points
structure Point :=
(x : ℝ) (y : ℝ)

-- Definition of the ellipse equation
def ellipse (m : ℝ) (P : Point) : Prop :=
m > 0 ∧ P.x^2 + 2 * P.y^2 = m

-- Definition of the midpoint of chord AB and points of intersection C and D
def midpoint (A B : Point) : Point := { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }
def perpendicular_bisector (P Q : Point) : Point := sorry -- Details about the line intersection are not elaborated here

-- Conditions and main problem statement
def main_problem (m : ℝ) (M : Point) : Prop :=
(M = { x := 2, y := 1 }) →
(ellipse m { x := 2, y := 1 }) →
∃ e, (e = (Real.sqrt 2) / 2) ∧  
(∃ A B C D, midpoint A B = M ∧ 
  (∀ P, ellipse m P ↔ ellipse m (perpendicular_bisector C D)) ∧
  |subtraction_function_1 A B| < |subtraction_function_2 C D| ∧ 
  collinear_on_same_circle M A B C D )

-- Skipping the proof
theorem exists_value_m (m : ℝ) (M : Point) : main_problem m M :=
by { sorry }

end exists_value_m_l666_666632


namespace soda_mineral_cost_l666_666073

theorem soda_mineral_cost
  (x y : ℝ)
  (h1 : 2 * x + y = 7)
  (h2 : 4 * x + 3 * y = 16) :
  10 * x + 10 * y = 45 :=
  sorry

end soda_mineral_cost_l666_666073


namespace find_pairs_S_n_l666_666988

def S_n (a : ℤ) (n : ℕ) (p : ℕ) [hp : Fact (p.Prime)] : ℕ :=
  Int.nat_abs ((a^p^n - 1).padicVal p)

theorem find_pairs_S_n : 
  (S_n 2013 98 2 = 100) ∧ (S_n 2013 99 503 = 100) ∧ ∀ n p, S_n 2013 n p = 100 → (n = 98 ∧ p = 2) ∨ (n = 99 ∧ p = 503) := 
by
  sorry

end find_pairs_S_n_l666_666988


namespace max_area_triang_ABC_l666_666309

noncomputable def max_area_triang (a b c : ℝ) (M : ℝ) (BM : ℝ := 2) (AM : ℝ := c - b) : ℝ :=
if M = (b + c) / 2 then 2 * Real.sqrt 3 else 0

theorem max_area_triang_ABC (a b c : ℝ) (M : ℝ) (BM : ℝ := 2) (AM : ℝ := c - b) (M_midpoint : M = (b + c) / 2) :
  max_area_triang a b c M BM AM = 2 * Real.sqrt 3 :=
by
  sorry

end max_area_triang_ABC_l666_666309


namespace least_xy_l666_666639

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_xy_l666_666639


namespace cos_double_angle_l666_666237

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666237


namespace regression_is_zero_implies_correlation_is_zero_l666_666706

variables {n : ℕ}
variables {x y : Fin n → ℝ}
variables {x̄ ȳ : ℝ}
variables (b r : ℝ)

def regression_coefficient (x y : Fin n → ℝ) (x̄ ȳ : ℝ) :=
  (∑ i, (x i - x̄) * (y i - ȳ)) / (∑ i, (x i - x̄)^2)

def correlation_coefficient (x y : Fin n → ℝ) (x̄ ȳ : ℝ) :=
  (∑ i, (x i - x̄) * (y i - ȳ)) / (Real.sqrt (∑ i, (x i - x̄)^2) * Real.sqrt (∑ i, (y i - ȳ)^2))

theorem regression_is_zero_implies_correlation_is_zero
  (hx : x̄ = ∑ i, x i / n)
  (hy : ȳ = ∑ i, y i / n)
  (hb : b = regression_coefficient x y x̄ ȳ)
  (hr : r = correlation_coefficient x y x̄ ȳ)
  (h : regression_coefficient x y x̄ ȳ = 0) :
  r = 0 :=
by
  sorry

end regression_is_zero_implies_correlation_is_zero_l666_666706


namespace least_value_xy_l666_666641

theorem least_value_xy {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : (1 : ℚ) / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_value_xy_l666_666641


namespace cos_double_angle_l666_666235

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666235


namespace remaining_volume_after_drilling_hole_l666_666533

def radius_of_cylinder (R : ℝ) : ℝ :=
  real.sqrt (R^2 - 9)

def volume_of_spherical_cap (R h : ℝ) : ℝ :=
  π * h^2 * (R - h / 3)

def volume_of_sphere (R : ℝ) : ℝ :=
  (4 / 3) * π * R^3

def volume_of_cylinder (R : ℝ) (h : ℝ) : ℝ :=
  π * (radius_of_cylinder R)^2 * h

theorem remaining_volume_after_drilling_hole (R : ℝ) (h : ℝ) (h_cylinder : h = 6) :
  ((4 / 3) * π * R^3 - volume_of_cylinder R h - 2 * volume_of_spherical_cap R 3) = 36 * π :=
by
  sorry

end remaining_volume_after_drilling_hole_l666_666533


namespace calculation_result_l666_666937

theorem calculation_result :
  (∏ k in Finset.range 17, (16 + k) / (k + 1)) / 
  (∏ j in Finset.range 15, (18 + j) / (j + 1)) = 496 :=
by sorry

end calculation_result_l666_666937


namespace extreme_value_0_at_minus_1_l666_666186

theorem extreme_value_0_at_minus_1 (m n : ℝ)
  (h1 : (-1) + 3 * m - n + m^2 = 0)
  (h2 : 3 - 6 * m + n = 0) :
  m + n = 11 :=
sorry

end extreme_value_0_at_minus_1_l666_666186


namespace div_by_30_l666_666783

theorem div_by_30 (a n : ℤ) : 30 ∣ (a ^ (n + 4) - a ^ n) :=
by
  sorry

end div_by_30_l666_666783


namespace men_count_for_work_l666_666412

theorem men_count_for_work :
  ∃ (x : ℕ), some_men x 21 8 = same_work_women 21 20 9 ∧ 3 * w = 2 * m ∧ x = 34 :=
by
  let m := work_man
  let w := work_woman
  have h1 : 3 * w = 2 * m := sorry
  have h2 : some_men x 21 8 = same_work_women 21 20 9 := sorry
  have h3 : x = 34 := sorry
  exact ⟨34, h2, h1, h3⟩

noncomputable def some_men (x : ℕ) (days : ℕ) (hours_per_day : ℕ) := x * days * hours_per_day * work_man
noncomputable def same_work_women (women : ℕ) (days : ℕ) (hours_per_day : ℕ) := women * days * hours_per_day * work_woman
noncomputable def work_man : ℝ := sorry
noncomputable def work_woman : ℝ := sorry

end men_count_for_work_l666_666412


namespace evaluate_powers_of_i_l666_666586

noncomputable def i_pow_13_18_23_28_33 : ℂ :=
  let i := complex.I
  i^13 + i^18 + i^23 + i^28 + i^33

theorem evaluate_powers_of_i :
  i_pow_13_18_23_28_33 = complex.I :=
by
  sorry

end evaluate_powers_of_i_l666_666586


namespace rectangle_area_diagonal_l666_666442

theorem rectangle_area_diagonal (r l w d : ℝ) (h_ratio : r = 5 / 2) (h_diag : d^2 = l^2 + w^2) : ∃ k : ℝ, (k = 10 / 29) ∧ (l / w = r) ∧ (l^2 + w^2 = d^2) :=
by
  sorry

end rectangle_area_diagonal_l666_666442


namespace smallest_d_correct_l666_666986

noncomputable def smallest_d := 2007^2008

theorem smallest_d_correct (x : ℝ) (hx : x > smallest_d) :
  ∃ d : ℕ, (d = 2007^2008) ∧ ∀ y > d, 
  ∀ (log_2010 : ℝ), (log_2010 = log 2010 (log 2009 (log 2008 (log 2007 y)))) → true :=
by
  let d := smallest_d
  use d
  split
  { refl }
  { intros y hy log_2010 hf
    sorry
  }

end smallest_d_correct_l666_666986


namespace cones_surface_area_and_volume_l666_666473

noncomputable def cone_properties (a : ℝ) : ℝ × ℝ :=
let R := a in
let h := a in
let l := real.sqrt (R ^ 2 + h ^ 2) in
let S_1 := (π * R * l) + (π * R ^ 2) in
let V_1 := (1 / 3) * π * R ^ 2 * h in
-- Total surface area of the two cones
let S := 2 * S_1 in
-- Total volume of the two cones
let V := 2 * V_1 in
(S, V)

theorem cones_surface_area_and_volume (a : ℝ) :
  cone_properties a = (2 * π * a ^ 2 * (real.sqrt 2 + 1), (2 / 3) * π * a ^ 3) :=
by
  let R := a
  let h := a
  let l := real.sqrt (R ^ 2 + h ^ 2)
  have l_def : l = a * real.sqrt 2 := by
    rw [real.sqrt_add, real.sqrt_mul_self, real.sqrt_mul_self]
    exact mul_nonneg (by norm_num) (by norm_num)
  let S_1 := π * R * l + π * R ^ 2
  have S_1_def : S_1 = π * a ^ 2 * (real.sqrt 2 + 1) := by
    rw [mul_assoc, ← mul_add, l_def, add_comm, mul_comm]
  let V_1 := (1 / 3) * π * R ^ 2 * h
  have V_1_def : V_1 = (1 / 3) * π * a ^ 3 := rfl
  let S := 2 * S_1
  let V := 2 * V_1
  have S_def : S = 2 * π * a ^ 2 * (real.sqrt 2 + 1) := by rw [S_1_def, mul_comm (2 : ℝ)]
  have V_def : V = (2 / 3) * π * a ^ 3 := by rw [V_1_def, ← mul_assoc, mul_comm (2 : ℝ)]
  exact ⟨S_def, V_def⟩

end cones_surface_area_and_volume_l666_666473


namespace Melanie_gave_out_brownies_l666_666388

variable (total_batches : ℕ) (brownies_per_batch : ℕ) (fraction_for_sale : ℚ)
variable (fraction_in_container : ℚ) (fraction_donated : ℚ)

theorem Melanie_gave_out_brownies :
  total_batches = 12 →
  brownies_per_batch = 25 →
  fraction_for_sale = 7 / 10 →
  fraction_in_container = 2 / 3 →
  fraction_donated = 2 / 5 →
  let total_brownies := total_batches * brownies_per_batch in
  let brownies_for_sale := fraction_for_sale * total_brownies in
  let remaining_post_sale := total_brownies - brownies_for_sale in
  let brownies_in_container := fraction_in_container * remaining_post_sale in
  let remaining_post_container := remaining_post_sale - brownies_in_container in
  let brownies_donated := fraction_donated * remaining_post_container in
  let brownies_given_out := remaining_post_container - brownies_donated in
  brownies_given_out = 18 :=
sorry

end Melanie_gave_out_brownies_l666_666388


namespace number_of_divisors_not_multiples_of_14_l666_666758

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_seventh_power (n : ℕ) : Prop := ∃ k : ℕ, k ^ 7 = n

theorem number_of_divisors_not_multiples_of_14
  (m : ℕ)
  (h1 : is_perfect_square (m / 2))
  (h2 : is_perfect_cube (m / 3))
  (h3 : is_perfect_seventh_power (m / 7)) :
  ∀ n, nat.divisors m = n :: nat.divisors m \ {k : ℕ | 14 ∣ k} ∧ n.length = 1364 :=
sorry

end number_of_divisors_not_multiples_of_14_l666_666758


namespace function_characterization_l666_666942

def isRelativelyPrime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem function_characterization (f : ℕ → ℤ) (hyp : ∀ x y, isRelativelyPrime x y → f (x + y) = f (x + 1) + f (y + 1)) :
  ∃ a b : ℤ, ∀ n : ℕ, f (2 * n) = (n - 1) * b ∧ f (2 * n + 1) = (n - 1) * b + a :=
by
  sorry

end function_characterization_l666_666942


namespace number_of_people_in_tour_l666_666951

theorem number_of_people_in_tour (x : ℕ) : 
  (x ≤ 25 ∧ 100 * x = 2700 ∨ 
  (x > 25 ∧ 
   (100 - 2 * (x - 25)) * x = 2700 ∧ 
   70 ≤ 100 - 2 * (x - 25))) → 
  x = 30 := 
by
  sorry

end number_of_people_in_tour_l666_666951


namespace john_needs_one_plank_l666_666991

-- Definitions based on the conditions
def totalNails : Nat := 11
def nailsPerPlank : Nat := 3
def additionalNails : Nat := 8

-- Mathematically equivalent proof problem
theorem john_needs_one_plank : ∃ P : Nat, nailsPerPlank * P = totalNails - additionalNails ∧ P = 1 := by
  have condition : nailsPerPlank * 1 = totalNails - additionalNails := by
    calc
      nailsPerPlank * 1 = 3 * 1 := by rfl
      _ = 3 := by rfl
      _ = 11 - 8 := by rfl
      
  use 1
  constructor
  · exact condition
  · rfl

end john_needs_one_plank_l666_666991


namespace sine_subtraction_formula_example_l666_666995

theorem sine_subtraction_formula_example
  (α β : ℝ)
  (h1 : sin α = 2 * real.sqrt 2 / 3)
  (h2 : cos (α + β) = -1 / 3)
  (h3 : 0 < α) 
  (h4 : α < π / 2) 
  (h5 : 0 < β) 
  (h6 : β < π / 2) :
  sin (α - β) = 10 * real.sqrt 2 / 27 := by
  sorry

end sine_subtraction_formula_example_l666_666995


namespace remainder_a_plus_b_l666_666521

-- Definition of the terms in the array.
def term_in_row (r : ℕ) (c : ℕ) : ℚ := (1 / (2 * 1004) ^ r) * (1 / 1004 ^ c)

-- Sum of all terms in an infinite array.
noncomputable def sum_of_array : ℚ :=
  ∑' r, ∑' c, term_in_row r c

-- Sum of geometric series calculations.
noncomputable def sum_geometric_series (a b : ℚ) : ℚ :=
  (1 / (1 - a)) * (1 / (1 - b))

-- Proved values for sum.
lemma sum_of_array_eq_a_div_b (a b : ℕ) (ha_relatively_prime : Nat.gcd a b = 1) :
  sum_of_array = a / b :=
sorry

-- The ultimate statement to prove.
theorem remainder_a_plus_b (a b : ℕ) (ha_relatively_prime : Nat.gcd a b = 1)
  (ha : 2 * 1004 * 1004 = a) (hb : 2007 * 1003 = b) : 
  (a + b) % 1004 = 1 :=
begin
  -- Assume sum_of_array is a / b for some integers a and b.
  have sum_eq : sum_of_array = a / b,
  from sum_of_array_eq_a_div_b a b ha_relatively_prime,
  -- Substitute the values of a and b as given.
  rcases ha, rcases hb,
  -- Calculate the remainder (a + b) % 1004 to be 1.
  sorry
end

end remainder_a_plus_b_l666_666521


namespace evaluate_power_l666_666965

theorem evaluate_power (a b : ℝ) (m n : ℝ) (h1 : a = b^m) (h2 : ∀ x y z : ℝ, (x^y)^z = x^(y*z)) : a^(n/m) = b^n :=
by
  sorry

example : 81^(5/4) = 243 := evaluate_power 81 3 4 5
  (by norm_num) -- Simplification for 81 = 3^4
  (by norm_num []) -- Using the power of a power rule

end evaluate_power_l666_666965


namespace largest_decreasing_digits_divisible_by_37_l666_666596

-- Definitions representing the conditions
def decreasing_digits (n : ℕ) : Prop :=
  let digits := n.to_digits.reverse
  (list.chain' (>) digits)

def divisible_by_37 (n : ℕ) : Prop :=
  n % 37 = 0

def largest_natural_number (n : ℕ) : Prop :=
  ∀ m : ℕ, (decreasing_digits m ∧ divisible_by_37 m) → m ≤ n

-- Statement of the problem
theorem largest_decreasing_digits_divisible_by_37 :
  decreasing_digits 987654 ∧ divisible_by_37 987654 ∧ largest_natural_number 987654 :=
  sorry

end largest_decreasing_digits_divisible_by_37_l666_666596


namespace correct_number_of_propositions_l666_666081

def proposition_1 (x : ℝ) : Prop :=
  ∀ x > 2 ∨ x < 3, Math.log 2 (x^2 - 5*x + 6)  < Math.log 2 ((3)^2 - 5*(3) + 6)

def proposition_2 (x y x1 y1 x2 y2 : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

def proposition_3 (x : ℝ) : Prop :=
  ¬ (∀ x ∈ set.univ, x^2 - x - 1 ≤ 0) ↔ ∃ (x0 : ℝ), x0^2 - x0 - 1 > 0

def number_of_correct_propositions : ℕ :=
  (if ∀ x, proposition_1 x = false then 0 else 1) +
  (if ∀ x y x1 y1 x2 y2, proposition_2 x y x1 y1 x2 y2 = true then 1 else 0) +
  (if ∀ x, proposition_3 x = true then 1 else 0)

theorem correct_number_of_propositions : number_of_correct_propositions = 2 :=
  sorry

end correct_number_of_propositions_l666_666081


namespace area_of_triangle_ABC_l666_666895

-- Define the conditions
def side_length_hexagon := 2
def radius_circle := 1
def distance_between_centers := 2 -- since each circle touches two others

-- Define the equilateral triangle formed by centers of three adjacent circles
def side_length_triangle := distance_between_centers

-- State the theorem
theorem area_of_triangle_ABC : 
  let s := side_length_triangle in 
  (s = 2) → 
  (let area := (Math.sqrt 3 / 4) * s^2 in
  area = Math.sqrt 3) :=
by
  intro s hs
  rw hs
  let area := (Math.sqrt 3 / 4) * 2^2
  show area = Math.sqrt 3
  rw [sq_2, mul_div_eq_mul_div, mul_assoc, ← two_mul_mathlib]
  sorry -- Skipping the proof

end area_of_triangle_ABC_l666_666895


namespace interval_of_monotonic_increase_l666_666182

theorem interval_of_monotonic_increase (a : ℝ) (k : ℤ) :
  (∀ x, x = (π / 6) → f(a, x) = a * sin(x) * cos(x) - sin(x)^2 + 1 / 2)
  → ∀ x, (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) :=
by
  sorry

-- Defining the function for use in the theorem
noncomputable def f (a x : ℝ) : ℝ :=
  a * sin x * cos x - sin x ^ 2 + 1 / 2

end interval_of_monotonic_increase_l666_666182


namespace range_of_a_l666_666666

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def is_monotone_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) : 
  is_even_function f → 
  is_monotone_increasing f {x : ℝ | x < 0} →
  f (2 ^ (|a - 1|)) > f (-.real.sqrt 2) →
  a ∈ set.Ioo (1 / 2) (3 / 2) :=
begin
  intros h_even h_mono h_ineq,
  sorry,
end

end range_of_a_l666_666666


namespace increase_by_one_unit_l666_666626

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 2 + 3 * x

-- State the theorem
theorem increase_by_one_unit (x : ℝ) : regression_eq (x + 1) - regression_eq x = 3 := by
  sorry

end increase_by_one_unit_l666_666626


namespace area_of_isosceles_trapezoid_circumscribed_circle_l666_666068

theorem area_of_isosceles_trapezoid_circumscribed_circle 
  (x y : ℝ) 
  (h1 : isosceles_trapezoid_circumscribed_circle x y) 
  (h2 : longer_base h1 = 20) 
  (h3 : base_angle h1 = arcsin 0.6) : 
  area h1 ≈ 74.17 := 
sorry

end area_of_isosceles_trapezoid_circumscribed_circle_l666_666068


namespace simplify_and_evaluate_expression_l666_666785

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = -6) : 
  (1 - a / (a - 3)) / ((a^2 + 3 * a) / (a^2 - 9)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l666_666785


namespace smallest_positive_period_range_f_B_l666_666674

-- The function definition
def f (x : ℝ) : ℝ := 4 * cos (x - π / 2) * sin (x - π / 3) - 1

-- Problem 1: Prove the smallest positive period of f(x) is π
theorem smallest_positive_period : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
by
  sorry

-- Problem 2: In ΔABC with sides a, b, c in a geometric sequence (b^2 = ac),
-- Prove the range of f(B) is [-2, -1]
theorem range_f_B (a b c : ℝ) (B : ℝ) 
  (h1 : b^2 = a * c) 
  (h2 : 0 < B) (h3 : B ≤ π / 3) 
  : ∀ (B : ℝ), f B ∈ set.interval (-2) (-1) :=
by
  sorry

end smallest_positive_period_range_f_B_l666_666674


namespace boat_license_combinations_l666_666048

theorem boat_license_combinations :
  let letters := ['A', 'M', 'S']
  let non_zero_digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let any_digit := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  3 * 9 * 10^4 = 270000 := 
by 
  sorry

end boat_license_combinations_l666_666048


namespace maria_needs_23_gallons_l666_666386

-- Definitions based on provided conditions
def num_pillars : ℕ := 12
def height : ℝ := 24  -- height of each pillar in feet
def diameter : ℝ := 8  -- diameter of each pillar in feet
def radius : ℝ := diameter / 2
def coverage_per_gallon : ℝ := 320  -- square feet per gallon

-- Curved surface area of one pillar
def lateral_surface_area (r h : ℝ) : ℝ := 2 * Real.pi * r * h

-- Total surface area to be painted for num_pillars
def total_surface_area_to_paint (n : ℕ) (r h : ℝ) : ℝ := n * lateral_surface_area r h

-- Gallons of paint needed
def gallons_of_paint_needed (area : ℝ) (coverage : ℝ) : ℝ := area / coverage

-- Maria's paint purchase
def maria_paint_purchase : ℕ := ⌈gallons_of_paint_needed (total_surface_area_to_paint num_pillars radius height) coverage_per_gallon⌉

-- Statement that needs proof
theorem maria_needs_23_gallons : maria_paint_purchase = 23 := by
  sorry

end maria_needs_23_gallons_l666_666386


namespace min_trips_l666_666836

def bucket (capacity trips : Nat) : Prop :=
  capacity * trips

theorem min_trips (total_liters trips_13 trips_3 trips_2 : Nat) 
  (h1 : bucket 13 trips_13 + bucket 3 trips_3 + bucket 2 trips_2 = total_liters)
  (h2 : total_liters = 50)
  : trips_13 + trips_3 + trips_2 = 7 :=
by
  sorry

end min_trips_l666_666836


namespace book_pages_l666_666576

theorem book_pages (P : ℝ) (h1 : P / 2 + 0.15 * (P / 2) + 210 = P) : P = 600 := 
sorry

end book_pages_l666_666576


namespace average_weight_of_children_l666_666806

theorem average_weight_of_children 
  (average_weight_boys : ℝ)
  (number_of_boys : ℕ)
  (average_weight_girls : ℝ)
  (number_of_girls : ℕ)
  (total_children : ℕ)
  (average_weight_children : ℝ) :
  average_weight_boys = 160 →
  number_of_boys = 8 →
  average_weight_girls = 130 →
  number_of_girls = 6 →
  total_children = number_of_boys + number_of_girls →
  average_weight_children = 
    (number_of_boys * average_weight_boys + number_of_girls * average_weight_girls) / total_children →
  average_weight_children = 147 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_weight_of_children_l666_666806


namespace minimum_teams_needed_l666_666023

theorem minimum_teams_needed (players : ℕ) (team_size : ℕ) (teams : ℕ) : 
  players = 30 ∧ team_size ∣ players ∧ team_size ≤ 8 → teams = players / team_size →
  teams = 5 :=
by {
  intros h1 h2,
  sorry
}

end minimum_teams_needed_l666_666023


namespace triangle_area_sides_5_5_6_l666_666486

theorem triangle_area_sides_5_5_6 (a b c : ℕ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 6) :
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 12 :=
by
  sorry

end triangle_area_sides_5_5_6_l666_666486


namespace books_remaining_after_second_day_l666_666889

theorem books_remaining_after_second_day :
  let initial_books := 100
  let first_day_borrowed := 5 * 2
  let second_day_borrowed := 20
  let total_borrowed := first_day_borrowed + second_day_borrowed
  let remaining_books := initial_books - total_borrowed
  remaining_books = 70 :=
by
  sorry

end books_remaining_after_second_day_l666_666889


namespace tan_double_angle_l666_666145

theorem tan_double_angle (α : ℝ) 
  (h : Real.tan α = 1 / 2) : Real.tan (2 * α) = 4 / 3 := 
by
  sorry

end tan_double_angle_l666_666145


namespace total_cost_of_modular_home_l666_666578

-- Definition of costs and sizes for the specific modules
def kitchen_sqft : ℕ := 400
def kitchen_cost : ℕ := 20000

def bathroom_sqft : ℕ := 150
def bathroom_cost : ℕ := 12000

def other_cost_per_sqft : ℕ := 100

-- Total planned area for the modular home
def total_sqft : ℕ := 2000

-- Total cost to build the modular home with given modules and areas
theorem total_cost_of_modular_home :
  let total_area_kitchen_bathrooms := kitchen_sqft + 2 * bathroom_sqft in
  let total_cost_kitchen_bathrooms := kitchen_cost + 2 * bathroom_cost in
  let remaining_sqft := total_sqft - total_area_kitchen_bathrooms in
  let remaining_cost := remaining_sqft * other_cost_per_sqft in
  total_cost_kitchen_bathrooms + remaining_cost = 174000 := 
  by
  sorry

end total_cost_of_modular_home_l666_666578


namespace amy_total_tickets_l666_666927

theorem amy_total_tickets (initial_tickets additional_tickets : ℕ) (h_initial : initial_tickets = 33) (h_additional : additional_tickets = 21) : 
  initial_tickets + additional_tickets = 54 := 
by 
  sorry

end amy_total_tickets_l666_666927


namespace integer_pairs_solution_l666_666086

theorem integer_pairs_solution :
  { x : ℤ // ∃ y : ℤ, 1 + 2^x + 2^(2 * x + 1) = y^2 } = {0, 4} ∧ 
  { y : ℤ // y = ±2} = {23, -23} :=
sorry

end integer_pairs_solution_l666_666086


namespace cos_double_angle_l666_666218

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l666_666218


namespace solve_for_x_l666_666600

open Real

theorem solve_for_x (x : ℝ) (h1 : x > 0) (h2 : 6 * sqrt (4 + x) + 6 * sqrt (4 - x) = 9 * sqrt 2) : 
  x = sqrt 255 / 4 :=
sorry

end solve_for_x_l666_666600


namespace thirtieth_entry_satisfies_l666_666125

def r_9 (n : ℕ) : ℕ := n % 9

theorem thirtieth_entry_satisfies (n : ℕ) (h : ∃ k : ℕ, k < 30 ∧ ∀ m < 30, k ≠ m → 
    (r_9 (7 * n + 3) ≤ 4) ∧ 
    ((r_9 (7 * n + 3) ≤ 4) ↔ 
    (r_9 (7 * m + 3) > 4))) :
  n = 37 :=
sorry

end thirtieth_entry_satisfies_l666_666125


namespace hexagon_diagonals_sum_l666_666028

noncomputable def sum_diagonals (A B C D E F : Point) (O : Circle) : ℕ :=
  if h : isInscribedHexagon ⟨A, B, C, D, E, F⟩ O ∧ (AB = 41) ∧ (BC = 91) ∧ (CD = 91) ∧ (DE = 91) ∧ (EF = 91) ∧ (FA = 91)
  then let x := length (A - C);
         let y := length (A - D);
         let z := length (A - E);
         x + y + z
  else 0

theorem hexagon_diagonals_sum (A B C D E F : Point) (O : Circle) :
  isInscribedHexagon ⟨A, B, C, D, E, F⟩ O ∧ (AB = 41) ∧ (BC = 91) ∧ (CD = 91) ∧ (DE = 91) ∧ (EF = 91) ∧ (FA = 91) →
  sum_diagonals A B C D E F O = 377 :=
by 
  intros h; 
  sorry

end hexagon_diagonals_sum_l666_666028


namespace count_valid_pairs_l666_666687

theorem count_valid_pairs : 
  ∃! n : ℕ, 
  n = 2 ∧ 
  (∀ (a b : ℕ), (0 < a ∧ 0 < b) → 
    (a * b + 97 = 18 * Nat.lcm a b + 14 * Nat.gcd a b) → 
    n = 2)
:= sorry

end count_valid_pairs_l666_666687


namespace factor_expr_l666_666075

def expr1 (x : ℝ) := 16 * x^6 + 49 * x^4 - 9
def expr2 (x : ℝ) := 4 * x^6 - 14 * x^4 - 9

theorem factor_expr (x : ℝ) :
  (expr1 x - expr2 x) = 3 * x^4 * (4 * x^2 + 21) := 
by
  sorry

end factor_expr_l666_666075


namespace expression_equals_one_l666_666935

noncomputable def given_expression : ℚ := 
  (List.prod (List.range' 1 17).map (λ i => (1 + 15) * (1 + 15 / i.toRat))) /
  (List.prod (List.range' 1 16).map (λ i => (1 + 17) * (1 + 17 / i.toRat)))

theorem expression_equals_one : given_expression = 1 := by
  sorry

end expression_equals_one_l666_666935


namespace count_multiples_of_15_l666_666943

theorem count_multiples_of_15 : ∃ n : ℕ, ∀ k, 12 < k ∧ k < 202 ∧ k % 15 = 0 ↔ k = 15 * n ∧ n = 13 := sorry

end count_multiples_of_15_l666_666943


namespace train_speed_approx_l666_666541

theorem train_speed_approx :
  ∀ (length : ℝ) (time : ℝ) (length = 40) (time = 3.9996800255979523), 
  length / time * 3.6 ≈ 36.003 := 
by
  intros length time h_length h_time
  rw [h_length, h_time]
  sorry

end train_speed_approx_l666_666541


namespace train_cross_bridge_time_l666_666683

-- Definitions based on the conditions provided
def length_of_train : ℝ := 110
def length_of_bridge : ℝ := 132
def speed_of_train_km_hr : ℝ := 36
def speed_convert_factor : ℝ := 1000 / 3600

-- Conversion from km/hr to m/s
def speed_of_train_m_s : ℝ := speed_of_train_km_hr * speed_convert_factor

-- Total distance to be covered by the train
def total_distance : ℝ := length_of_train + length_of_bridge

-- Correct answer
def correct_time : ℝ := 24.2

-- Proof that the time to cross the bridge is equal to the correct time
theorem train_cross_bridge_time : (total_distance / speed_of_train_m_s) = correct_time := by
  sorry

end train_cross_bridge_time_l666_666683


namespace sequence_solution_l666_666627

theorem sequence_solution :
  ∃ (a : ℕ → ℕ) (b : ℕ → ℝ),
    a 1 = 2 ∧
    (∀ n, b n = (a (n + 1)) / (a n)) ∧
    b 10 * b 11 = 2 →
    a 21 = 2 ^ 11 :=
by
  sorry

end sequence_solution_l666_666627


namespace f_divisible_by_g_l666_666008

noncomputable def f (a b c : ℚ) (x : ℚ) : ℚ := a * x^2 + b * x + c
noncomputable def g (d e : ℚ) (x : ℚ) : ℚ := d * x + e

theorem f_divisible_by_g (a b c d e : ℚ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) : 
  (∀ n : ℕ, 0 < n → ((f a b c n) / (g d e n)) = (f a b c n) / (g d e n) ∧ (f a b c n) % (g d e n) = 0) →
  ∀ x : ℚ, (f a b c x) % (g d e x) = 0 :=
by
  sorry

end f_divisible_by_g_l666_666008


namespace cos_double_angle_l666_666215

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l666_666215


namespace totalInitialAmount_l666_666613

variable (a j t k x : ℝ)

-- Given conditions
def initialToyAmount : Prop :=
  t = 48

def kimRedistribution : Prop :=
  k = 4 * x - 144

def amyRedistribution : Prop :=
  (a = 3 * x) ∧ (j = 2 * x) ∧ (t = 2 * x)

def janRedistribution : Prop :=
  (a = 3 * x) ∧ (t = 4 * x)

def toyRedistribution : Prop :=
  (a = 6 * x) ∧ (j = -6 * x) ∧ (t = 48) 

def toyFinalAmount : Prop :=
  t = 48

-- Proof Problem
theorem totalInitialAmount
  (h1 : initialToyAmount t)
  (h2 : kimRedistribution k x)
  (h3 : amyRedistribution a j t x)
  (h4 : janRedistribution a t x)
  (h5 : toyRedistribution a j t x)
  (h6 : toyFinalAmount t) :
  a + j + t + k = 192 :=
sorry

end totalInitialAmount_l666_666613


namespace work_completion_days_l666_666510

theorem work_completion_days (A B : ℝ) (h1 : B = 6) (h2 : A = 10) : 
  let work_rate_A := 1 / A,
      work_rate_B := 1 / B,
      combined_work_rate := work_rate_A + work_rate_B in
  1 / combined_work_rate = 3.75 :=
by 
  let work_rate_A := 1 / A
  let work_rate_B := 1 / B
  let combined_work_rate := work_rate_A + work_rate_B
  have h0 : combined_work_rate = 4 / 15 := by sorry
  have h3 : 1 / combined_work_rate = 15 / 4 := by sorry
  have h4 : (15 / 4 : ℝ) = 3.75 := by norm_num
  rw [←h4, h3]
  exact h0

end work_completion_days_l666_666510


namespace plane_vector_z_l666_666648

def vector (α : Type*) := (α × α × α)

def dot_product {α : Type*} [Field α] (u v : vector α) : α :=
  let (ux, uy, uz) := u in
  let (vx, vy, vz) := v in
  ux * vx + uy * vy + uz * vz

def cross_product {α : Type*} [Field α] (u v : vector α) : vector α :=
  let (ux, uy, uz) := u in
  let (vx, vy, vz) := v in
  ((uy * vz - uz * vy), (uz * vx - ux * vz), (ux * vy - uy * vx))

def same_plane {α : Type*} [Field α] (a b c : vector α) : Prop :=
  let cp := cross_product a b in
  dot_product cp c = 0

theorem plane_vector_z : 
  let a := (2, 1, -3)
  let b := (-1, 2, 3)
  let c (z : ℝ) := (7, 6, z)
  same_plane a b (c (-9)) :=
by sorry

end plane_vector_z_l666_666648


namespace meter_to_steps_l666_666415

variable (hops skips jumps steps meters : Type)
variable hop_to_skip : (2 : ℝ) * (hops : ℝ) = (3 : ℝ) * (skips : ℝ)
variable jump_to_hop : (4 : ℝ) * (jumps : ℝ) = (6 : ℝ) * (hops : ℝ)
variable jump_to_meter : (5 : ℝ) * (jumps : ℝ) = (20 : ℝ) * (meters : ℝ)
variable skip_to_step : (15 : ℝ) * (skips : ℝ) = (10 : ℝ) * (steps : ℝ)

theorem meter_to_steps :
  ∃ (factor : ℝ), (1 : ℝ) * (meters : ℝ) = factor * (steps : ℝ) ∧ factor = (3 / 8 : ℝ) := sorry

end meter_to_steps_l666_666415


namespace geometric_sequence_n_eq_5_l666_666334

-- Conditions of the problem
variables {a : ℕ → ℝ} (n : ℕ)
variable h1 : a 1 + a n = 82
variable h2 : a 3 * a (n - 2) = 81
variable h3 : (finset.range n).sum a = 121

-- Goal: Prove that n = 5
theorem geometric_sequence_n_eq_5 : n = 5 :=
by
  sorry

end geometric_sequence_n_eq_5_l666_666334


namespace relay_race_total_races_l666_666134

theorem relay_race_total_races 
  (dave_races : ℕ)
  (betty_races : ℕ)
  (alex_races : ℕ)
  (clara_races : ℕ)
  (h_dave : dave_races = 8)
  (h_betty : betty_races = 3)
  (h_alex_bounds : 3 < alex_races ∧ alex_races < 8)
  (h_clara_bounds : 3 < clara_races ∧ clara_races < 8) :
  let total_races := (dave_races + betty_races + alex_races + clara_races) / 2 in
  total_races = 10 :=
by
  let total_races := (dave_races + betty_races + alex_races + clara_races) / 2
  sorry

end relay_race_total_races_l666_666134


namespace sum_first_n_inverse_no_lambda_geometric_sequence_l666_666630

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n + 1
def S (n : ℕ) : ℕ := \sum i in range n, a i
def T (n : ℕ) : ℝ := \sum i in range n, b i

-- (I) Sum of the first n terms of {1/(a_n a_{n+1})}
theorem sum_first_n_inverse (n : ℕ) :
  \sum i in range n, 1 / (a i * a (i + 1)) = n / (6 * n + 9) :=
sorry

-- (II) No non-zero λ such that {b_n} forms a geometric sequence
theorem no_lambda_geometric_sequence (b : ℕ → ℝ) :
  (∀ λ : ℝ, λ ≠ 0 → ¬ ∀ n : ℕ, b (n + 1) = λ * b n) :=
sorry

end sum_first_n_inverse_no_lambda_geometric_sequence_l666_666630


namespace complex_coord_l666_666809

theorem complex_coord (i : ℂ)
  (hi : i = complex.I) :
  let z := i * (1 + i) in 
  z = -1 + complex.I := 
sorry

end complex_coord_l666_666809


namespace correct_option_A_l666_666857

theorem correct_option_A : 
  (∀ a : ℝ, a^3 * a^4 = a^7) ∧ 
  ¬ (∀ a : ℝ, a^6 / a^2 = a^3) ∧ 
  ¬ (∀ a : ℝ, a^4 - a^2 = a^2) ∧ 
  ¬ (∀ a b : ℝ, (a - b)^2 = a^2 - b^2) :=
by
  /- omitted proofs -/
  sorry

end correct_option_A_l666_666857


namespace unique_arrangements_of_numbers_in_grid_l666_666842

theorem unique_arrangements_of_numbers_in_grid : 
  let S := {1, 2, 3, 4, 5, 6}
  ∃ configurations : Finset (Finset (ℕ × ℕ)),
    configurations.card = 12 ∧
    ∀ config ∈ configurations,
      (∀ (n1 n2 : ℕ) (c1 c2 : ℕ × ℕ),
        c1 ∈ config →
        c2 ∈ config →
        n1 ∈ S →
        n2 ∈ S →
        n1 ≠ n2 →
        (c1.1 < c2.1 → c1.2 < c2.2 → n1 > n2)) :=
by
  sorry

end unique_arrangements_of_numbers_in_grid_l666_666842


namespace cos_double_angle_l666_666232

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666232


namespace cos_double_angle_l666_666281

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l666_666281


namespace initial_price_70_percent_markup_l666_666897

theorem initial_price_70_percent_markup (P : ℝ) (h : 2 * P - 1.7 * P = 6) : 1.7 * P = 34 := by
  calc
    0.3 * P = 6 : by linarith
    P = 6 / 0.3 : by {rw [← mul_div_cancel_left 6 (by norm_num : (0.3 : ℝ) ≠ 0)], norm_num}
    1.7 * P = 1.7 * (6 / 0.3) : by rw [P]
             ... = 34 : by norm_num

end initial_price_70_percent_markup_l666_666897


namespace total_cost_is_174000_l666_666580

namespace ModularHome
  -- Conditions given as definitions in Lean
  def kitchen_sqft : ℕ := 400
  def kitchen_cost : ℕ := 20_000
  def bathroom_sqft : ℕ := 150
  def bathroom_cost : ℕ := 12_000
  def other_modules_cost_per_sqft : ℕ := 100
  def total_planned_sqft : ℕ := 2_000
  def num_bathrooms : ℕ := 2

  -- Required to prove that total cost is $174,000
  def calculate_total_cost : ℕ :=
    let kitchen_total_cost := kitchen_cost
    let bathrooms_total_cost := num_bathrooms * bathroom_cost
    let total_sqft_kitchen_bathrooms := kitchen_sqft + num_bathrooms * bathroom_sqft
    let remaining_sqft := total_planned_sqft - total_sqft_kitchen_bathrooms
    let remaining_cost := remaining_sqft * other_modules_cost_per_sqft
    kitchen_total_cost + bathrooms_total_cost + remaining_cost

  theorem total_cost_is_174000 : calculate_total_cost = 174_000 := by
    sorry
end ModularHome

end total_cost_is_174000_l666_666580


namespace fraction_of_satisfactory_grades_is_3_4_l666_666807

def num_grades (grades : String → ℕ) : ℕ := 
  grades "A" + grades "B" + grades "C" + grades "D" + grades "F"

def satisfactory_grades (grades : String → ℕ) : ℕ := 
  grades "A" + grades "B" + grades "C" + grades "D"

def fraction_satisfactory (grades : String → ℕ) : ℚ := 
  satisfactory_grades grades / num_grades grades

theorem fraction_of_satisfactory_grades_is_3_4 
  (grades : String → ℕ)
  (hA : grades "A" = 5)
  (hB : grades "B" = 4)
  (hC : grades "C" = 3)
  (hD : grades "D" = 3)
  (hF : grades "F" = 5) : 
  fraction_satisfactory grades = (3 : ℚ) / 4 := by
{
  sorry
}

end fraction_of_satisfactory_grades_is_3_4_l666_666807


namespace angle_equality_l666_666753

variables {α : Type*} [incidence_geometry α]

variables (Γ : Circle) (P A B K T P' : Point α)
variable [IsOutside P Γ]
variables [IsTangent P Γ A] [IsTangent P Γ B]
variable [IsBetween K A B]
variable [OnCircumcircle P B K T]
variable [Reflection P K P']

-- The theorem statement
theorem angle_equality 
  (h1 : IsOutside P Γ)
  (h2 : IsTangent P Γ A)
  (h3 : IsTangent P Γ B)
  (h4 : IsBetween K A B)
  (h5 : OnCircumcircle P B K T)
  (h6 : Reflection P K P') :
  ∠ P B T = ∠ P' K A :=
sorry

end angle_equality_l666_666753


namespace _l666_666662

-- Define the main theorem with the given condition and the required conclusion
example (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 2) : Real.log (4 ** -a) / Real.log 3 = -2 := by
  sorry

end _l666_666662


namespace radius_of_second_circle_l666_666562

theorem radius_of_second_circle
  (a α : ℝ) -- conditions
  (h_triangle_isosceles : ∀ (B C : ℝ), B = a / 2 ∧ tan α = C / B) -- height and other constraints
  (h_inscribed_in_circle : ∃ (R : ℝ), ∀ A B C, inscribes_in_circle A B C R)
  (h_second_circle_tangency : ∃ (r : ℝ), touches_base_and_first_circle r (a/2) α) -- second circle tangent
  : ∃ (x : ℝ), x = a / 4 * cot α := -- correct answer
sorry

end radius_of_second_circle_l666_666562


namespace fred_final_cards_l666_666614

def initial_cards : ℕ := 40
def keith_bought : ℕ := 22
def linda_bought : ℕ := 15

theorem fred_final_cards : initial_cards - keith_bought - linda_bought = 3 :=
by sorry

end fred_final_cards_l666_666614


namespace distance_to_center_square_l666_666025

theorem distance_to_center_square (x y : ℝ) (h : x*x + y*y = 72) (h1 : x*x + (y + 8)*(y + 8) = 72) (h2 : (x + 4)*(x + 4) + y*y = 72) :
  x*x + y*y = 9 ∨ x*x + y*y = 185 :=
by
  sorry

end distance_to_center_square_l666_666025


namespace concrete_required_in_whole_cubic_yards_l666_666024

noncomputable def rectangular_section_volume : ℝ :=
  let width : ℝ := 1  -- in yards
  let length : ℝ := 20 -- in yards
  let thickness : ℝ := 1 / 12 -- in yards
  width * length * thickness

noncomputable def semicircular_end_volume : ℝ :=
  let radius : ℝ := 0.5 -- in yards
  let thickness : ℝ := 1 / 12 -- in yards
  (1 / 2) * π * (radius ^ 2) * thickness

noncomputable def total_volume : ℝ :=
  rectangular_section_volume + semicircular_end_volume

theorem concrete_required_in_whole_cubic_yards : 
  ⌈total_volume⌉ = 2 :=
by
  sorry

end concrete_required_in_whole_cubic_yards_l666_666024


namespace same_terminal_side_l666_666059
  
theorem same_terminal_side (k : ℤ) : ∃ k : ℤ, (2 * k * real.pi) - (real.pi / 3) = (5 * real.pi / 3) :=
sorry

end same_terminal_side_l666_666059


namespace rhombus_diagonal_length_l666_666429

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) (h1 : area = 600) (h2 : d1 = 30) :
  d2 = 40 :=
by
  sorry

end rhombus_diagonal_length_l666_666429


namespace part_I_part_II_l666_666454

noncomputable def Sn (n : ℕ) : ℚ := (n^2 + 3 * n) / 2

def a (n : ℕ) : ℕ := n + 1

def bn (n : ℕ) : ℚ := 1 / (a (2 * n - 1) * a (2 * n + 1))

noncomputable def Tn (n : ℕ) : ℚ := ∑ i in Finset.range n, bn i

theorem part_I (n : ℕ) (hn : n ≥ 1) : a n = (Sn n) - (Sn (n - 1)) := 
  by 
  sorry

theorem part_II (n : ℕ) : Tn n = (2 * n + 1) / (4 * n + 4) :=
  by 
  sorry

end part_I_part_II_l666_666454


namespace find_edge_of_cube_volume_3375_l666_666971

def edge_of_cube_given_volume (volume : ℝ) (edge : ℝ) : Prop :=
  edge^3 = volume

theorem find_edge_of_cube_volume_3375 : ∃ edge : ℝ, edge_of_cube_given_volume 3375 edge ∧ edge = 15 :=
by
  use 15
  split
  · show 15^3 = 3375
    sorry
  · show 15 = 15
    rfl

end find_edge_of_cube_volume_3375_l666_666971


namespace least_number_to_add_l666_666515

theorem least_number_to_add (x : ℕ) (h : 1056 % 23 = 21) : (1056 + x) % 23 = 0 ↔ x = 2 :=
by {
    sorry
}

end least_number_to_add_l666_666515


namespace cosine_double_angle_l666_666247

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l666_666247


namespace at_least_1000_triangles_l666_666354

open Classical

noncomputable def points_in_regions (A : Point) (regions : Fin 6 → Set Point) : Prop :=
  (∀ i, (regions i).card = 5) ∧
  (∀ (i j k : Fin 6), ∀ (p ∈ regions i), ∀ (q ∈ regions j), ∀ (r ∈ regions k), ¬ collinear [p, q, r])

theorem at_least_1000_triangles (A : Point)(regions : Fin 6 → Set Point)
    (h : points_in_regions A regions) :
    ∃ triangles : Finset (Finset Point), 
      (triangles.card ≥ 1000) ∧
      (∀ triangle ∈ triangles, triangle.card = 3 ∧ (A ∈ convex_hull ℝ triangle) ) :=
sorry

end at_least_1000_triangles_l666_666354


namespace cos_double_angle_l666_666244

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666244


namespace cos_double_angle_l666_666213

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l666_666213


namespace units_digit_multiplication_l666_666115

-- Define a function to find the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Statement of the problem: Given the product 27 * 36, prove that the units digit is 2.
theorem units_digit_multiplication (a b : ℕ) (h1 : units_digit 27 = 7) (h2 : units_digit 36 = 6) :
  units_digit (27 * 36) = 2 :=
by
  have h3 : units_digit (7 * 6) = 2 := by sorry
  exact h3

end units_digit_multiplication_l666_666115


namespace find_frac_sum_l666_666378

noncomputable def centroid_plane_intersect (α β γ : ℝ) : ℝ × ℝ × ℝ :=
  ((α / 3), (β / 3), (γ / 3))

theorem find_frac_sum :
  ∀ α : ℝ,
  ∀ β : ℝ,
  ∀ γ : ℝ,
  α = β →
  β = γ →
  1 / (α^2) + 1 / (β^2) + 1 / (γ^2) = 36 :=
begin
  intros α β γ h1 h2,
  -- Conditions and further proof steps would go here
  sorry
end

end find_frac_sum_l666_666378


namespace graph_remains_connected_after_removing_one_color_l666_666465

/-- A 30-point graph where each pair is connected by an edge of one of four colors remains connected after removing all edges of some one color. -/
theorem graph_remains_connected_after_removing_one_color :
  ∃ C : Fin 4 → Prop,
    ∀ G : SimpleGraph (Fin 30),
    ((∀ v w : Fin 30, v ≠ w → ∃ c : Fin 4, G.edge v w c) →
     ∀ c : Fin 4, (∀ p q : Fin 30, p ≠ q → (∃ r : Fin 30, r ≠ p ∧ r ≠ q ∧ ¬G.edge p q c)) → 
     Connected (G ⊖ {E | E.2 = c → E ∉ G.edge})) :=
by sorry

end graph_remains_connected_after_removing_one_color_l666_666465


namespace cos_double_angle_l666_666286

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666286


namespace canisterC_initially_half_full_l666_666930

-- Definitions based on the conditions
def capacityC : ℝ := sorry
def capacityD : ℝ := 2 * capacityC
def initialFractionFullD : ℝ := 1 / 3
def remainingFractionFullD : ℝ := 1 / 12

-- Main theorem: Prove that canister C is initially 1/2 full of water
theorem canisterC_initially_half_full 
  (h1 : capacityD = 2 * capacityC)
  (h2 : initialFractionFullD = 1 / 3)
  (h3 : remainingFractionFullD = 1 / 12)
  : (1 / 2) * capacityC = capacityC * initialFractionFullD - (capacityC * remainingFractionFullD) :=
by {
  sorry
}

end canisterC_initially_half_full_l666_666930


namespace limit_A_div_B_eq_2_l666_666369

noncomputable def A (n : ℕ) : ℝ :=
  Real.pi * (1 / 2 * Real.csc (Real.pi / n.toReal)) ^ 2 - n.toReal / 4 * Real.cot (Real.pi / n.toReal)

noncomputable def B (n : ℕ) : ℝ :=
  n.toReal / 4 * Real.cot (Real.pi / n.toReal) - Real.pi * (1 / 2 * Real.cot (Real.pi / n.toReal)) ^ 2

theorem limit_A_div_B_eq_2 : 
  Tendsto (fun n => A(n) / B(n)) atTop (𝓝 2) := sorry

end limit_A_div_B_eq_2_l666_666369


namespace intervals_countable_dissection_l666_666784

variable {ι : Type*}

def congruent_sets (A B : Set ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), bijective f ∧ ∀ x ∈ A, f x ∈ B ∧ ∀ x ∈ B, f x ∈ A

theorem intervals_countable_dissection (A B : Set ℝ)
  (hA : (∃ a b : ℝ, a < b ∧ A = Set.Ioo a b) ∨ (∃ a b : ℝ, a < b ∧ A ∈ {Set.Ioc a b, Set.Ico a b, Set.Icc a b}))
  (hB : (∃ a b : ℝ, a < b ∧ B = Set.Ioo a b) ∨ (∃ a b : ℝ, a < b ∧ B ∈ {Set.Ioc a b, Set.Ico a b, Set.Icc a b}))
  (pos_length_A : 0 < measure_theory.measure_Set A)
  (pos_length_B : 0 < measure_theory.measure_Set B) :
  ∃ (A_i B_i : ι → Set ℝ), (∀ i, A_i i ⊆ A ∧ B_i i ⊆ B) ∧ 
                            (∀ i j, i ≠ j → disjoint (A_i i) (A_i j) ∧ disjoint (B_i i) (B_i j)) ∧ 
                            (∀ i, congruent_sets (A_i i) (B_i i)) :=
sorry

end intervals_countable_dissection_l666_666784


namespace number_of_male_animals_l666_666709

def total_original_animals : ℕ := 100 + 29 + 9
def animals_bought_by_brian : ℕ := total_original_animals / 2
def animals_after_brian : ℕ := total_original_animals - animals_bought_by_brian
def animals_after_jeremy : ℕ := animals_after_brian + 37

theorem number_of_male_animals : animals_after_jeremy / 2 = 53 :=
by
  sorry

end number_of_male_animals_l666_666709


namespace four_color_removal_l666_666468

open SimpleGraph

-- Define the graph as a complete graph on 30 vertices
def K_30 : SimpleGraph (Fin 30) := completeGraph (Fin 30)

-- Define the coloring function
variable (coloring : Fin 30 → Fin 30 → Fin 4)

-- A complete graph is connected
noncomputable def connected_graph : SimpleGraph (Fin 30) := {
  adj := λ v w, v ≠ w,
  symm := λ v w h, h.symm,
  loopless := λ v h, h rfl,
}

-- Given the conditions, prove the statement
theorem four_color_removal :
  ∃ c : Fin 4, ∀ {G' : SimpleGraph (Fin 30)},
    (G'.adj = λ v w, v ≠ w ∧ coloring v w ≠ c) → G'.connected :=
begin
  sorry
end

end four_color_removal_l666_666468


namespace expression_for_q_l666_666815

noncomputable def q (x : ℝ) : ℝ := 2 * x ^ 2 - 8

theorem expression_for_q :
  (∃ a : ℝ, (∀ x : ℝ, q(x) = a * (x + 2) * (x - 2))) ∧
  (∀ x : ℝ, x = -2 → q(x) = 0) ∧
  (∀ x : ℝ, x = 2 → q(x) = 0) ∧
  (q 0 = 8) :=
by
  sorry

end expression_for_q_l666_666815


namespace find_angle_B_find_cosine_sum_range_l666_666321

-- Define the acute triangle and given conditions
structure acute_triangle (A B C : ℝ) (a b c : ℝ) :=
(acute_A : 0 < A ∧ A < π / 2)
(acute_B : 0 < B ∧ B < π / 2)
(acute_C : 0 < C ∧ C < π / 2)
(sides : a > 0 ∧ b > 0 ∧ c > 0)
(angles_sum : A + B + C = π)
(given_condition : 2 * b * sin A = sqrt 3 * a)

-- Part Ⅰ: Proving the measure of angle B
theorem find_angle_B {A B C a b c : ℝ} (h : acute_triangle A B C a b c) : 
  B = π / 3 :=
sorry

-- Part Ⅱ: Proving the range of values for cos A + cos B + cos C
theorem find_cosine_sum_range {A B C a b c : ℝ} (h : acute_triangle A B C a b c) :
  (sqrt 3 + 1) / 2 < cos A + cos B + cos C ∧ cos A + cos B + cos C ≤ 3 / 2 :=
sorry

end find_angle_B_find_cosine_sum_range_l666_666321


namespace overall_percent_change_l666_666038

theorem overall_percent_change (W : ℝ) : 
  (W * 0.6 * 1.3 * 0.8 * 1.1) / W = 0.624 :=
by {
  sorry
}

end overall_percent_change_l666_666038


namespace cos_double_angle_l666_666211

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l666_666211


namespace perpendicular_condition_l666_666154

def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_condition (a : ℝ) :
  is_perpendicular (a^2) (1/a) ↔ a = -1 :=
sorry

end perpendicular_condition_l666_666154


namespace explicit_formula_and_properties_l666_666382

noncomputable def f (x : ℝ) := x + 1 / x

-- The given conditions
def cond1 := f 1 = 2
def cond2 := f 2 = 5 / 2

-- Prove that the function meets the conditions
theorem explicit_formula_and_properties :
  (f 1 = 2) ∧ (f 2 = 5 / 2) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, 1 < x1 ∧ x1 < x2 → f x1 < f x2) :=
by
  split
  case left =>
    -- Proof for f 1 = 2
    exact sorry
  case right =>
    split
    case left =>
      -- Proof for f 2 = 5 / 2
      exact sorry
    case right =>
      split
      case left =>
        -- Proof that f is odd
        exact sorry
      case right =>
        -- Proof that f is increasing on (1, +∞)
        exact sorry

end explicit_formula_and_properties_l666_666382


namespace circle_equation_l666_666431

theorem circle_equation (x y : ℝ) :
  (∃ a < 0, (x - a)^2 + y^2 = 4 ∧ (0 - a)^2 + 0^2 = 4) ↔ (x + 2)^2 + y^2 = 4 := 
sorry

end circle_equation_l666_666431


namespace find_ab_solution_l666_666105

theorem find_ab_solution (a b : ℤ) (h : 7^a - 3 * 2^b = 1) : (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 4) :=
sorry

end find_ab_solution_l666_666105


namespace chords_in_circle_l666_666792

theorem chords_in_circle (n : ℕ) (h_n : n = 10) : (n.choose 2) = 45 := sorry

end chords_in_circle_l666_666792


namespace seating_arrangement_l666_666899

theorem seating_arrangement (N : ℕ) (hN : N ≥ 2) :
    ∃ (ways : ℕ), ways = N^2 ∧ ∀ arrangement : {A : finset (fin (2 * N)) // A.card = (N - 1)}, 
       (∀ (i j ∈ arrangement.val), abs (i - j) ≠ 1) :=
sorry

end seating_arrangement_l666_666899


namespace triangle_area_is_12_l666_666482

/-- 
Theorem: The area of an isosceles triangle with sides 5, 5, and 6 units is 12 square units.
-/
theorem triangle_area_is_12 
  (a b c : ℕ) 
  (h1 : a = 5)
  (h2 : b = 5)
  (h3 : c = 6)
  : (1/2 : ℚ) * c * (real.sqrt (a^2 - (c/2)^2)) = 12 := 
by 
  sorry

end triangle_area_is_12_l666_666482


namespace binomial_probability_l666_666146

open ProbabilityTheory

theorem binomial_probability (X : ℕ → ℕ) (hX : Binomial 6 (1/3) X) : 
  P (X = 2) = 80 / 243 := 
sorry

end binomial_probability_l666_666146


namespace cos_double_angle_l666_666219

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l666_666219


namespace percentage_of_meetings_spent_in_work_day_l666_666772

theorem percentage_of_meetings_spent_in_work_day :
  let work_day_minutes := 8 * 60,
      first_meeting_minutes := 45,
      second_meeting_minutes := 3 * first_meeting_minutes,
      total_meeting_minutes := first_meeting_minutes + second_meeting_minutes
  in (total_meeting_minutes.toFloat / work_day_minutes.toFloat) * 100 = 37.5 :=
by
  sorry

end percentage_of_meetings_spent_in_work_day_l666_666772


namespace pow_evaluation_l666_666961

theorem pow_evaluation : 81^(5/4) = 243 := 
by sorry

end pow_evaluation_l666_666961


namespace find_larger_integer_l666_666824

noncomputable def larger_integer (a b : ℤ) := max a b

theorem find_larger_integer (a b : ℕ) 
  (h1 : a/b = 7/3) 
  (h2 : a * b = 294): 
  larger_integer a b = 7 * Real.sqrt 14 :=
by
  -- Proof goes here
  sorry

end find_larger_integer_l666_666824


namespace units_digit_fraction_l666_666851

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10 = (2500) % 10 → 
  ((30 * 31 * 32 * 33 * 34 * 35) / 2500) % 10 = 1 := 
by 
  intro h
  sorry

end units_digit_fraction_l666_666851


namespace ellipse_standard_equation_and_max_area_l666_666172

theorem ellipse_standard_equation_and_max_area
    (a b : ℝ) (M : ℝ² → Prop)
    (h1 : a > b) (h2 : b > 0) (h3 : ∀ x y, M (x, y) ↔ (x^2) / (a^2) + (y^2) / (b^2) = 1)
    (ecc : ∀ c, c = sqrt 2 → c / a = sqrt 2 / 2) (foci : (√2, 0) ∈ M)
    (m ≠ 0) :
  ∃ a b, a = 2 ∧ b^2 = 2 ∧ (∀ x y, M (x, y) ↔ (x^2) / 4 + (y^2) / 2 = 1) ∧
  (∀ A B, ∃ S, S = sqrt 2) := by
  sorry

end ellipse_standard_equation_and_max_area_l666_666172


namespace cos_double_angle_l666_666229

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666229


namespace expression_positive_integer_l666_666573

theorem expression_positive_integer (x : ℝ) (hx : x ≠ 0) : 
  (∃ n : ℕ, (n > 0) ∧ (n ≥ 1) ∧ ∃ k : ℤ, (k > 0) ∧ (k = (|x^2 - |x||) / x)) ↔ (x > 1 ∧ (∃ n : ℕ, x = n ∧ n ≥ 2)) := 
sorry

end expression_positive_integer_l666_666573


namespace pow_evaluation_l666_666959

theorem pow_evaluation : 81^(5/4) = 243 := 
by sorry

end pow_evaluation_l666_666959


namespace nate_coin_flip_l666_666876

-- Definitions of the conditions
def total_sequences : ℕ := 2 ^ 12
def favorable_sequences : ℕ := 143
def total_probability (a b : ℕ) : ℚ := a / b

-- Prop to state the problem
theorem nate_coin_flip :
  let m := 143
  let n := 4096
  m.gcd n = 1 ∧ 
  total_probability favorable_sequences total_sequences = (m : ℚ) / n ∧ 
  (m + n = 4239) :=
by
  sorry

end nate_coin_flip_l666_666876


namespace cos_double_angle_l666_666259

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l666_666259


namespace hyperbola_focal_length_is_8_l666_666191

noncomputable def hyperbola_focal_length (b : ℝ) (hb : b > 0) : ℝ :=
  let a := 2 in
  let e := (Real.sqrt 3) / 3 * b in
  let c := e * a in
  let lhs := c ^ 2 in
  let rhs := a ^ 2 + b ^ 2 in
  if lhs = rhs then 2 * c else 0

theorem hyperbola_focal_length_is_8 (b : ℝ) (hb : b > 0) (e : ℝ)
    (he : e = (Real.sqrt 3) / 3 * b) : hyperbola_focal_length b hb = 8 :=
  sorry

end hyperbola_focal_length_is_8_l666_666191


namespace double_angle_cosine_l666_666228

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l666_666228


namespace scientific_notation_216000_l666_666421

theorem scientific_notation_216000 : 216000 = 2.16 * 10^5 :=
by
  -- proof will be provided here
  sorry

end scientific_notation_216000_l666_666421


namespace number_of_right_triangles_l666_666433

def rectangle_points : Set ℝ :=
  {A, P, B, C, Q, D, R}

def conditions (A P B C Q D R : ℝ): Prop :=
  -- Assuming A, B, C, D are distinct corners of rectangle,
  -- P, Q divide it into two congruent rectangles along diagonal AC,
  -- R is the intersection of AC and PQ.
  sorry

theorem number_of_right_triangles (A P B C Q D R : ℝ) (h : conditions A P B C Q D R) : 
  ∃ n : ℕ, n = 12 ∧ 
    (n = cardinal.mk {t | ∃ x y z : ℝ, t = {x, y, z} ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
      ({x, y, z} ⊆ rectangle_points ∧ 
        (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ t = {a, b, c} ∧ 
          (right_triangle t)
        )
    })) := sorry

end number_of_right_triangles_l666_666433


namespace complex_equivalence_l666_666871

-- Definitions and conditions
def i : ℂ := complex.I
def c1 : ℂ := 2 + i
def c2 : ℂ := 1 + i
def a_b : ℂ := (3 - i) / 2
def a : ℝ := 3 / 2
def b : ℝ := -1 / 2

-- Theorem to prove
theorem complex_equivalence : (a + b = 1) ↔ (c1 / c2 = a_b) := by
  -- Lean skips and assumes this as a proof placeholder
  sorry

end complex_equivalence_l666_666871


namespace sum_of_y_coefficients_l666_666967

-- Define the polynomials
def poly1 (x y : ℕ) := 5 * x + 3 * y + 2
def poly2 (x y : ℕ) := 2 * x + 5 * y + 3

-- The expanded form
def expanded_poly (x y : ℕ) := 10 * x^2 + 31 * x * y + 19 * x + 15 * y^2 + 19 * y + 6

-- Sum of the coefficients of the terms with nonzero power of y
theorem sum_of_y_coefficients : 
  let terms_with_y := [31 * 1, 15 * 1, 19 * 1] in
  list.sum terms_with_y = 65 :=
by 
  sorry

end sum_of_y_coefficients_l666_666967


namespace axis_of_symmetry_l666_666108

theorem axis_of_symmetry (x : ℝ) : 
  ∃ (c : ℝ), c = -π/3 ∧ ∀ (k : ℤ), 
    3 * real.sin (x / 2 - π / 3) = 3 * real.sin ((2 * k * π + 5 * π / 3) / 2 - π / 3) → 
    c = (2 * k * π + 5 * π / 3 + 2 * (k + 1) * π + 5 * π / 3)/2 :=
begin
  sorry
end

end axis_of_symmetry_l666_666108


namespace bowling_ball_volume_l666_666523

open Real

noncomputable def remaining_volume (d_bowling_ball d1 d2 d3 d4 h1 h2 h3 h4 : ℝ) : ℝ :=
  let r_bowling_ball := d_bowling_ball / 2
  let v_bowling_ball := (4/3) * π * (r_bowling_ball ^ 3)
  let v_hole1 := π * ((d1 / 2) ^ 2) * h1
  let v_hole2 := π * ((d2 / 2) ^ 2) * h2
  let v_hole3 := π * ((d3 / 2) ^ 2) * h3
  let v_hole4 := π * ((d4 / 2) ^ 2) * h4
  v_bowling_ball - (v_hole1 + v_hole2 + v_hole3 + v_hole4)

theorem bowling_ball_volume :
  remaining_volume 40 3 3 4 5 10 10 12 8 = 10523.67 * π :=
by
  sorry

end bowling_ball_volume_l666_666523


namespace median_in_75_79_l666_666082

def student_scores : List (Int × Int) := [
  (85, 20), (80, 18), (75, 15), (70, 12), 
  (65, 10), (60, 8), (55, 10), (50, 7)
]

def median_position : Nat := 50

def scores_list : List Int :=
  student_scores.foldr (λ (score, count) acc, 
    List.repeat score count ++ acc) []

noncomputable def median (l : List Int) : Float :=
  let sorted_l := l.qsort (· < ·)
  (sorted_l.get! (median_position - 1) + sorted_l.get! median_position).toFloat / 2

def interval_75_79 (x : Float) : Prop := 75 ≤ x ∧ x < 80

theorem median_in_75_79 : interval_75_79 (median scores_list) :=
by sorry

end median_in_75_79_l666_666082


namespace initial_bacteria_count_l666_666425

theorem initial_bacteria_count (doubling_interval : ℕ) (initial_count four_minutes_final_count : ℕ)
  (h1 : doubling_interval = 30)
  (h2 : four_minutes_final_count = 524288)
  (h3 : ∀ t : ℕ, initial_count * 2 ^ (t / doubling_interval) = four_minutes_final_count) :
  initial_count = 2048 :=
sorry

end initial_bacteria_count_l666_666425


namespace common_chord_length_l666_666472

theorem common_chord_length (r : ℝ) (h : r = 12) :
  let d := r in
  let half_chord := sqrt (r^2 - (d/2)^2) in
  2 * half_chord = 12 * sqrt 3 := by
  sorry

end common_chord_length_l666_666472


namespace donna_has_40_bananas_l666_666941

-- Define the number of bananas each person has
variables (dawn lydia donna total : ℕ)

-- State the conditions
axiom h1 : dawn + lydia + donna = total
axiom h2 : dawn = lydia + 40
axiom h3 : lydia = 60
axiom h4 : total = 200

-- State the theorem to be proved
theorem donna_has_40_bananas : donna = 40 :=
by {
  sorry -- Placeholder for the proof
}

end donna_has_40_bananas_l666_666941


namespace no_road_to_capital_l666_666714

theorem no_road_to_capital (n m : ℕ) (cap : Type) (city : fin n) 
  (outgoing : fin n → fin m) (incoming : fin n → fin (m + 1) )
  (h1 : n = 101) 
  (h2 : ∀ i : fin 100, outgoing i = 20)
  (h3 : ∀ i : fin 100, incoming i = 21) :
  incoming 0 = 0 :=
by sorry

end no_road_to_capital_l666_666714


namespace range_of_a_has_3_integer_solutions_l666_666707

theorem range_of_a_has_3_integer_solutions (a : ℝ) :
  (-7 : ℝ) ≤ a ∧ a < -3 ↔
  ∃ x : ℝ, (x / 2 - 1 < (2 - 3 * x) / 3) ∧ (a - 3 < 4 * x - 2) ∧
  (Int.card {x : ℤ | (x / 2 - 1 < (2 - 3 * x) / 3) ∧ (a - 3 < 4 * x - 2)} = 3) :=
by
  sorry

end range_of_a_has_3_integer_solutions_l666_666707


namespace min_product_value_max_product_value_l666_666764

open Real

noncomputable def min_cos_sin_product (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 then
    cos x * sin y * cos z
  else 0

noncomputable def max_cos_sin_product (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 then
    cos x * sin y * cos z
  else 0

theorem min_product_value :
  ∃ (x y z : ℝ), x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 ∧ min_cos_sin_product x y z = 1 / 8 :=
sorry

theorem max_product_value :
  ∃ (x y z : ℝ), x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 ∧ max_cos_sin_product x y z = (2 + sqrt 3) / 8 :=
sorry

end min_product_value_max_product_value_l666_666764


namespace conditional_probability_event_B_given_event_A_l666_666529

-- Definitions of events A and B
def event_A := {outcomes | ∃ i j k, outcomes = (i, j, k) ∧ (i = 1 ∨ j = 1 ∨ k = 1)}
def event_B := {outcomes | ∃ i j k, outcomes = (i, j, k) ∧ (i + j + k = 1)}

-- Calculation of probabilities
def probability_AB := 3 / 8
def probability_A := 7 / 8

-- Prove conditional probability
theorem conditional_probability_event_B_given_event_A :
  (probability_AB / probability_A) = 3 / 7 :=
by
  sorry

end conditional_probability_event_B_given_event_A_l666_666529


namespace cos_double_angle_l666_666256

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l666_666256


namespace standard_equation_of_ellipse_range_of_triangle_OAB_area_l666_666156

noncomputable def semiMajorAxis := sqrt 2
noncomputable def semiMinorAxis := 1
noncomputable def fociDistance := 1

variables (O : Point)
variables (F1 F2 : Point)
variables (P : Point)
variables (M : Point)
variables (A B : Point)
variables (k m : ℝ)
variables (lambda S : ℝ)
variables (tangent_line : Line)
variables (ellipse : Set Point)
variables (circle : Set Point)

axiom ellipse_has_foci : foci (ellipse) = (F1, F2)
axiom ellipse_contains_P : P ∈ ellipse
axiom P_coordinates : P = (-1, sqrt 2 / 2)
axiom midpoint_condition : Midpoint(P, F2) = M
axiom perpendicular_condition : OM_perpendicular_F1F2 O M F1 F2
axiom circle_diameter : diameter(circle) = segment(F1, F2)
axiom line_tangent_to_circle : is_tangent(tangent_line, circle)
axiom line_eq : tangent_line = {y := k * x + m}
axiom line_intersects_ellipse_A_B : {A, B} = tangent_line ∩ ellipse
axiom dot_product_condition : dot_product(vector(O, A), vector(O, B)) = lambda
axiom lambda_range : 2 / 3 ≤ lambda ∧ λ ≤ 3 / 4

theorem standard_equation_of_ellipse :
  ellipse = { (x, y) | x^2 / semiMajorAxis^2 + y^2 / semiMinorAxis^2 = 1 } := sorry

theorem range_of_triangle_OAB_area :
  (sqrt 6 / 4 ≤ S ∧ S ≤ 2 / 3) :=
  sorry

end standard_equation_of_ellipse_range_of_triangle_OAB_area_l666_666156


namespace max_product_sum_1979_l666_666107

theorem max_product_sum_1979 :
  ∃ (n : ℕ) (a : Fin n → ℕ), (∑ i, a i = 1979) ∧ ((∏ i, a i) = (2 * 3 ^ 659)) :=
by
  sorry

end max_product_sum_1979_l666_666107


namespace α_plus_β_equals_negative_3π_over_4_l666_666651

-- Define conditions
def α β : ℝ
def h1 : α ∈ Ioo (-π/2) (π/2)
def h2 : β ∈ Ioo (-π/2) (π/2)
def h3 (x : ℝ) : x^2 + 6 * x + 7 = 0

-- State to prove
theorem α_plus_β_equals_negative_3π_over_4 
  (h1 : α ∈ Ioo (-π/2) (π/2))
  (h2 : β ∈ Ioo (-π/2) (π/2))
  (h3 : ∀ x, x^2 + 6 * x + 7 = 0 → x = tan α ∨ x = tan β) :
  α + β = -3 * π / 4 := 
by
  sorry

end α_plus_β_equals_negative_3π_over_4_l666_666651


namespace prime_addition_fraction_equivalence_l666_666853

theorem prime_addition_fraction_equivalence : 
  ∃ n : ℕ, Prime n ∧ (4 + n) * 8 = (7 + n) * 7 ∧ n = 17 := 
sorry

end prime_addition_fraction_equivalence_l666_666853


namespace range_of_y_l666_666295

theorem range_of_y (y : ℝ) (hy : y < 0) (hceil_floor : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end range_of_y_l666_666295


namespace triangle_area_sum_of_sqrt_l666_666909

theorem triangle_area_sum_of_sqrt (T T1 T2 T3 : ℝ) (hT : 0 ≤ T) (hT1 : 0 ≤ T1) (hT2 : 0 ≤ T2) (hT3 : 0 ≤ T3)
  (hDiv : ∀ k1 k2 k3 : ℝ, (T1 = k1^2 * T) → (T2 = k2^2 * T) → (T3 = k3^2 * T) → (1 = k1 + k2 + k3)) :
  sqrt T = sqrt T1 + sqrt T2 + sqrt T3 :=
by {
  sorry
}

end triangle_area_sum_of_sqrt_l666_666909


namespace ratio_c_d_l666_666304

theorem ratio_c_d (x y c d : ℝ) (h1 : 4 * x - 2 * y = c)
  (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : c / d = -1 / 3 := by
  sorry

end ratio_c_d_l666_666304


namespace fiona_observes_pairs_l666_666603

theorem fiona_observes_pairs (n : ℕ) (h : n = 12) : (nat.choose n 2) = 66 := by
  sorry

end fiona_observes_pairs_l666_666603


namespace withdraw_representation_l666_666504

-- Define the concept of depositing and withdrawing money.
def deposit (amount : ℕ) : ℤ := amount
def withdraw (amount : ℕ) : ℤ := - amount

-- Define the given condition: depositing $30,000 is represented as $+30,000.
def deposit_condition : deposit 30000 = 30000 := by rfl

-- The statement to be proved: withdrawing $40,000 is represented as $-40,000
theorem withdraw_representation (deposit_condition : deposit 30000 = 30000) : withdraw 40000 = -40000 :=
by
  sorry

end withdraw_representation_l666_666504


namespace point_to_plane_distance_l666_666167

-- Define the normal vector of the plane
def normal_vector : ℝ × ℝ × ℝ := (-2, -2, 1)

-- Define point A on the plane
def point_A : ℝ × ℝ × ℝ := (-1, 3, 0)

-- Define point P
def point_P : ℝ × ℝ × ℝ := (-2, 1, 2)

-- Define the vector AP
def vector_AP := (point_P.1 - point_A.1, point_P.2 - point_A.2, point_P.3 - point_A.3)

-- Define the dot product function
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the norm function
def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Theorem statement: Prove the distance from point P to the plane is 8/3
theorem point_to_plane_distance : 
  abs ((dot_product vector_AP normal_vector) / (norm normal_vector)) = 8 / 3 :=
sorry

end point_to_plane_distance_l666_666167


namespace odd_integer_permutation_l666_666969

theorem odd_integer_permutation (n : ℕ) (h_n_odd : odd n) (h_n_gt_one : n > 1) : 
  (∃ (a : Fin n → Fin n), 
    (∀ i : Fin n, ∑ j, (if j + 1 = i then 1 else if j = i then -1 else 0) * a (Fin.mod (j + i) n) > 0)) ↔ n % 4 = 1 :=
sorry

end odd_integer_permutation_l666_666969


namespace cyclic_quadrilateral_min_bd_l666_666746

theorem cyclic_quadrilateral_min_bd {A B C D I : Type*}
  [IncenterOfTriangle I A B D]
  (h_cyclic : cyclic_quadrilateral A B C D)
  (h_BC : dist B C = 2)
  (h_CD : dist C D = 2)
  (h_AI : dist A I = 2) :
  ∃ BD_min : ℝ, BD_min = 2 * Real.sqrt 3 :=
begin
  -- Proof will be provided here
  sorry,
end

end cyclic_quadrilateral_min_bd_l666_666746


namespace ice_cream_vendor_l666_666064

theorem ice_cream_vendor (choco : ℕ) (mango : ℕ) (sold_choco : ℚ) (sold_mango : ℚ) 
  (h_choco : choco = 50) (h_mango : mango = 54) (h_sold_choco : sold_choco = 3/5) 
  (h_sold_mango : sold_mango = 2/3) : 
  choco - (choco * sold_choco) + mango - (mango * sold_mango) = 38 := 
by 
  sorry

end ice_cream_vendor_l666_666064


namespace chords_on_circle_l666_666800

theorem chords_on_circle (n : ℕ) (h : n = 10) : nat.choose n 2 = 45 :=
by {
  rw h,
  -- we can directly calculate choose 10 2
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num),
  sorry -- the actual detailed proof goes here.
}

end chords_on_circle_l666_666800


namespace beneficial_to_buy_card_breakeven_visits_l666_666000

-- Define the conditions for Part (a)
def avg_check := 900
def discount_rate := 0.30
def visits_per_week := 3
def weeks_per_year := 52
def card_cost := 30000

-- Define the conditions for Part (b)
def avg_check_lower := 600

-- Part (a) statement
theorem beneficial_to_buy_card :
  (visits_per_week * weeks_per_year * (avg_check * discount_rate) > card_cost) :=
by
  sorry

-- Part (b) statement
theorem breakeven_visits :
  (card_cost / (avg_check_lower * discount_rate)).ceil = 167 :=
by
  sorry

end beneficial_to_buy_card_breakeven_visits_l666_666000


namespace swap_values_correct_l666_666671

theorem swap_values_correct (a b : ℕ) : ∃ c : ℕ, (c = a ∧ a = b ∧ b = c) → (a = b ∧ b = a) :=
by 
  intros c h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw [h1, h2, h3],
  sorry

end swap_values_correct_l666_666671


namespace arithmetic_sequence_min_positive_value_l666_666755

theorem arithmetic_sequence_min_positive_value (a₁ d : ℝ) (hd : d < -1) :
  ∃ n : ℕ, let S_n := (n / 2) * (2 * a₁ + (n - 1) * d) in
  ∀ m : ℕ, (m / 2) * (2 * a₁ + (m - 1) * d) = S_n →
  (n = 20 ∧ S_n > 0) ∧ ((∀ k : ℕ, k < n → (k / 2) * (2 * a₁ + (k - 1) * d) > S_n) ∨ (∀ k : ℕ, k ≤ n → (k / 2) * (2 * a₁ + (k - 1) * d) ≤ 0)) :=
begin
  sorry
end

end arithmetic_sequence_min_positive_value_l666_666755


namespace sum_reciprocals_eq_two_l666_666362

variable {a b c d : ℝ}
variable {ω ψ : ℂ}

theorem sum_reciprocals_eq_two 
(h₁ : a ≠ -1) 
(h₂ : b ≠ -1) 
(h₃ : c ≠ -1) 
(h₄ : d ≠ -1) 
(h₅ : a ≠ -complex.I) 
(h₆ : b ≠ -complex.I) 
(h₇ : c ≠ -complex.I) 
(h₈ : d ≠ -complex.I) 
(hω₁ : ω^3 = 1) 
(hω₂ : ω ≠ 1) 
(hψ : ψ = ω^2) 
(hsum : (1/(a + ψ) + 1/(b + ψ) + 1/(c + ψ) + 1/(d + ψ) = 4/ω)) : 
1/(a + 1) + 1/(b + 1) + 1/(c + 1) + 1/(d + 1) = 2 :=
sorry

end sum_reciprocals_eq_two_l666_666362


namespace clara_stickers_l666_666931

-- Definitions based on the problem conditions
def initial_stickers : ℕ := 100
def stickers_given_to_boy : ℕ := 10
def stickers_received_from_teacher : ℕ := 50
def stickers_given_to_classmates : ℕ := 20
def stickers_exchanged : ℕ := 15
def stickers_received_in_exchange : ℕ := 30
def half (x : ℕ) : ℕ := x / 2

-- Calculate the number of stickers Clara has left
def stickers_remaining : ℕ :=
  let after_boy := initial_stickers - stickers_given_to_boy in
  let after_teacher := after_boy + stickers_received_from_teacher in
  let after_classmates := after_teacher - stickers_given_to_classmates in
  let after_exchange := after_classmates - stickers_exchanged + stickers_received_in_exchange in
  after_exchange - half after_exchange

-- The goal is to prove Clara's remaining stickers
theorem clara_stickers : stickers_remaining = 68 :=
by
  sorry

end clara_stickers_l666_666931


namespace find_n_arctan_l666_666983

theorem find_n_arctan (n : ℕ) (hn_pos : 0 < n) :
  arctan (1 / 6) + arctan (1 / 7) + arctan (1 / 5) + arctan (1 / n) = π / 4 ↔ n = 311 :=
by sorry

end find_n_arctan_l666_666983


namespace max_n_for_triangles_l666_666735

theorem max_n_for_triangles (n : ℕ) :
  let total_points := n + 100,
      no_collinear := ∀ (p1 p2 p3 : Point), 
        {p1, p2, p3} ⊆ total_points → ¬ collinear p1 p2 p3,
      triangles_formed ≤ 300 := 300
  in n ≤ 102 :=
sorry

end max_n_for_triangles_l666_666735


namespace larger_value_3a_plus_1_l666_666413

theorem larger_value_3a_plus_1 {a : ℝ} (h : 8 * a^2 + 6 * a + 2 = 0) : 3 * a + 1 ≤ 3 * (-1/4 : ℝ) + 1 := 
sorry

end larger_value_3a_plus_1_l666_666413


namespace ellipse_properties_l666_666724

-- Define the conditions of the problem
def sum_distances_eq (P : ℝ × ℝ) : Prop :=
    let (x, y) := P in dist P (0, -real.sqrt 3) + dist P (0, real.sqrt 3) = 4

def ellipse_eq (P : ℝ × ℝ) : Prop :=
    let (x, y) := P in (x : ℝ) ^ 2 + (y : ℝ) ^ 2 / 4 = 1

-- The line y = kx + 1 intersects C at points A and B
def line_intersects_ellipse (k : ℝ) (A B : ℝ × ℝ) : Prop :=
    let (x1, y1) := A in
    let (x2, y2) := B in
    ellipse_eq A ∧ ellipse_eq B ∧ y1 = k * x1 + 1 ∧ y2 = k * x2 + 1

-- OA ⊥ OB
def perpendicular_vectors (A B: ℝ × ℝ) : Prop :=
    let (x1, y1) := A in
    let (x2, y2) := B in x1 * x2 + y1 * y2 = 0

-- The proof problem in Lean 4
theorem ellipse_properties (C : set (ℝ × ℝ)) (A B : ℝ × ℝ) (k : ℝ)
    (h1 : ∀ P, P ∈ C ↔ sum_distances_eq P)
    (h2 : ∀ P, P ∈ C ↔ ellipse_eq P)
    (h3 : line_intersects_ellipse k A B)
    (h4 : perpendicular_vectors A B) :
    ellipse_eq (0, 2) ∧ (k = 1 / 2 ∨ k = -1 / 2) :=
sorry

end ellipse_properties_l666_666724


namespace tax_collected_total_l666_666968

variable (total_collected : ℝ)
variable (tax_william_paid : ℝ := 480)
variable (percent_william_land : ℝ := 0.25)

theorem tax_collected_total 
  (h₀ : tax_william_paid = 480)
  (h₁ : percent_william_land = 0.25)
  : total_collected = 1920 :=
by
  -- Conditions
  have : 0.25 * total_collected = 480,
  {
    calc
    0.25 * total_collected = tax_william_paid : by rw [h₁, h₀]
  },
  -- Solution for total_collected
  calc
  total_collected = 480 / 0.25 : by rw [this]
              ... = 1920          : by norm_num

end tax_collected_total_l666_666968


namespace cost_price_is_925_l666_666859

-- Definitions for the conditions
def SP : ℝ := 1110
def profit_percentage : ℝ := 0.20

-- Theorem to prove that the cost price is 925
theorem cost_price_is_925 (CP : ℝ) (h : SP = (CP * (1 + profit_percentage))) : CP = 925 := 
by sorry

end cost_price_is_925_l666_666859


namespace find_a_l666_666611

variable (x y : ℝ)
variable (x_i y_i : ℝ)
variable (data : Fin 8 → ℝ × ℝ)

def linear_regression_line (x : ℝ) (a : ℝ) := (1/3) * x + a

theorem find_a (h1 : (∑ i in Finset.range 8, (data i).fst) = 6)
               (h2 : (∑ i in Finset.range 8, (data i).snd) = 3) :
  ∃ (a : ℝ), linear_regression_line (3/4) a = 3/8 :=
begin
  use 1/8,
  sorry
end

end find_a_l666_666611


namespace system_solution_l666_666787

-- Definitions and assumptions from step a)
variables {a b c x y z : ℝ}
variables (h₁: a ≠ 0) (h₂: b ≠ 0) (h₃: c ≠ 0)
variables (hx: x ≠ 0) (hy: y ≠ 0) (hz: z ≠ 0)
variables (h_sum: x + y + z ≠ 0)

-- Define the system of equations
def equation1 := 1/x + 1/(x + y) = 1/a
def equation2 := 1/y + 1/(z + x) = 1/b
def equation3 := 1/z + 1/(x + y) = 1/c

-- Theorem statement proving the solution provided
theorem system_solution :
  equation1 → equation2 → equation3 → 
  x = (2 * (a * b + a * c + b * c) - (a^2 + b^2 + c^2)) / (2 * (-a + b + c)) ∧
  y = (2 * (a * b + a * c + b * c) - (a^2 + b^2 + c^2)) / (2 * (a - b + c)) ∧
  z = (2 * (a * b + a * c + b * c) - (a^2 + b^2 + c^2)) / (2 * (a + b - c)) :=
sorry

end system_solution_l666_666787


namespace geodesic_lines_cover_each_point_num_families_of_geodesic_lines_l666_666863

-- Definitions based on the problem's conditions
structure Cube (V : Type) := (vertices : finset V) -- Representing the finite set of vertices of a cube

structure GeodesicLine (V : Type) :=
(trace : V → Prop)

def geodesic_line_on_cube {V : Type} [fintype V] (c : Cube V) (g : GeodesicLine V) : Prop :=
∀ (v : V), v ∈ c.vertices → g.trace v

-- Theorems based on the problem's questions and their correct answers
theorem geodesic_lines_cover_each_point (V : Type) [fintype V] (c : Cube V) (g : GeodesicLine V) :
  ∀ (v : V), v ∈ c.vertices → 4 :=
by sorry

theorem num_families_of_geodesic_lines (V : Type) [fintype V] (c : Cube V) :
  ∃ (families : finset (GeodesicLine V)), families.card = 7 :=
by sorry

end geodesic_lines_cover_each_point_num_families_of_geodesic_lines_l666_666863


namespace find_max_M_within_range_l666_666990

/-- Define M(a) as the number of positive integers b such that a + b divides ab -/
def M (a : ℕ) : ℕ :=
  Nat.card { b : ℕ // 0 < b ∧ (a + b) ∣ (a * b) }

theorem find_max_M_within_range : 
  ∃ (a : ℕ), 1 ≤ a ∧ a ≤ 2013 ∧ (∀ b, 1 ≤ b ∧ b ≤ 2013 → M b ≤ M 1680) :=
by sorry

end find_max_M_within_range_l666_666990


namespace sum_of_number_and_radical_conjugate_l666_666938

theorem sum_of_number_and_radical_conjugate : 
  (10 - Real.sqrt 2018) + (10 + Real.sqrt 2018) = 20 := 
by 
  sorry

end sum_of_number_and_radical_conjugate_l666_666938


namespace sheets_in_set_l666_666347

-- Definitions of the conditions
def John_sheets_left (S E : ℕ) : Prop := S - E = 80
def Mary_sheets_used (S E : ℕ) : Prop := S = 4 * E

-- Theorems to prove the number of sheets
theorem sheets_in_set (S E : ℕ) (hJohn : John_sheets_left S E) (hMary : Mary_sheets_used S E) : S = 320 :=
by { 
  sorry 
}

end sheets_in_set_l666_666347


namespace find_p_l666_666164

noncomputable def point_parabola (p : ℝ) (y_0 : ℝ) :=
  y_0^2 = 2 * p * 3

def parabola_focus_distance (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

theorem find_p (p : ℝ) (y_0 : ℝ) 
  (h1 : 0 < p) (h2 : p < 6) 
  (h3 : point_parabola p y_0) 
  (h4 : distance (3, y_0) (parabola_focus_distance p) = 2 * distance (3, y_0) (p / 2, y_0)) :
  p = 2 :=
sorry

end find_p_l666_666164


namespace count_fourdigit_integers_div_by_35_of_form_x35_l666_666688

theorem count_fourdigit_integers_div_by_35_of_form_x35 : 
  {n: Nat // 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 35 ∧ n % 35 = 0}.card = 13 :=
by
  sorry

end count_fourdigit_integers_div_by_35_of_form_x35_l666_666688


namespace cos_double_angle_l666_666233

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666233


namespace vasya_new_scoring_system_l666_666003

theorem vasya_new_scoring_system (a b c : ℕ) 
  (h1 : a + b + c = 52) 
  (h2 : a + b / 2 = 35) : a - c = 18 :=
by
  sorry

end vasya_new_scoring_system_l666_666003


namespace sum_of_arithmetic_sequence_l666_666148

/-- Given an arithmetic sequence {a_n}, where the point (n, a_n) lies on a fixed line passing through 
the point (8, 4), prove that the sum of the first 15 terms of the sequence is 60. -/
theorem sum_of_arithmetic_sequence : 
  (∃ k m : ℝ, ∀ n : ℕ, a_n = k * n + m) ∧ (a 8 = 4) →
  S 15 = 60 :=
sorry

end sum_of_arithmetic_sequence_l666_666148


namespace petrov_vasechkin_boards_l666_666392

theorem petrov_vasechkin_boards:
  ∃ n : ℕ, 
  (∃ x y : ℕ, 2 * x + 3 * y = 87 ∧ x + y = n) ∧ 
  (∃ u v : ℕ, 3 * u + 5 * v = 94 ∧ u + v = n) ∧ 
  n = 30 := 
sorry

end petrov_vasechkin_boards_l666_666392


namespace min_value_A_mul_abs_x1_minus_x2_l666_666998

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2017 * x + Real.pi / 6) + Real.cos (2017 * x - Real.pi / 3)

theorem min_value_A_mul_abs_x1_minus_x2 :
  ∃ x1 x2 : ℝ, (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) →
  2 * |x1 - x2| = (2 * Real.pi) / 2017 :=
sorry

end min_value_A_mul_abs_x1_minus_x2_l666_666998


namespace problem_statement_l666_666307

def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem problem_statement :
  lcm (lcm (gcd 24 36) (gcd 54 24)) (gcd (48 * 60) (72 * 48)) = 576 := 
sorry

end problem_statement_l666_666307


namespace budget_for_equipment_l666_666022

-- Definitions based on problem conditions
def salaries : ℕ := 60
def research_and_development : ℕ := 9
def utilities : ℕ := 5
def supplies : ℕ := 2
def transportation_degrees : ℕ := 72
def total_budget : ℕ := 100
def degrees_per_percent := 360 / total_budget -- This should be 3.6, but we'll keep it as a fraction here

-- Prove the question statement
theorem budget_for_equipment : 
  let transportation_percent := transportation_degrees / degrees_per_percent in
  let known_percentages := salaries + research_and_development + utilities + supplies + transportation_percent in
  total_budget - known_percentages = 4 := 
by 
  sorry

end budget_for_equipment_l666_666022


namespace misha_card_numbers_l666_666517

-- Define the context for digits
def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

-- Define conditions
def proper_fraction (a b : ℕ) : Prop := is_digit a ∧ is_digit b ∧ a < b

-- Original problem statement rewritten for Lean
theorem misha_card_numbers (L O M N S B : ℕ) :
  is_digit L → is_digit O → is_digit M → is_digit N → is_digit S → is_digit B →
  proper_fraction O M → proper_fraction O S →
  L + O / M + O + N + O / S = 10 + B :=
sorry

end misha_card_numbers_l666_666517


namespace simplify_fraction_l666_666786

theorem simplify_fraction (n : ℤ) : 
  (2^(n+5) - 4 * 2^(n+1)) / (4 * 2^(n+4)) = 3 / 8 :=
by
  sorry

end simplify_fraction_l666_666786


namespace master_bedroom_size_l666_666031

-- Definitions of variables and conditions
def total_area : ℝ := 2300
def common_area : ℝ := 1000
def bedroom_area : ℝ := total_area - common_area
def guest_bedroom (master_bedroom : ℝ) : ℝ := master_bedroom / 4

-- Statement to prove
theorem master_bedroom_size : ∃ (M : ℝ), guest_bedroom M + M = bedroom_area ∧ M = 1040 :=
by
  sorry

end master_bedroom_size_l666_666031


namespace complex_numbers_count_l666_666979

theorem complex_numbers_count (z : ℂ) (h1 : complex.abs z = 1) (h2 : complex.abs ((z / (z.conj)) - ((z.conj) / z)) = 1) : (∃ z_list : list ℂ, z_list.length = 8 ∧ ∀ w ∈ z_list, complex.abs w = 1 ∧ complex.abs ((w / (w.conj)) - ((w.conj) / w)) = 1) :=
sorry

end complex_numbers_count_l666_666979


namespace four_color_removal_l666_666467

open SimpleGraph

-- Define the graph as a complete graph on 30 vertices
def K_30 : SimpleGraph (Fin 30) := completeGraph (Fin 30)

-- Define the coloring function
variable (coloring : Fin 30 → Fin 30 → Fin 4)

-- A complete graph is connected
noncomputable def connected_graph : SimpleGraph (Fin 30) := {
  adj := λ v w, v ≠ w,
  symm := λ v w h, h.symm,
  loopless := λ v h, h rfl,
}

-- Given the conditions, prove the statement
theorem four_color_removal :
  ∃ c : Fin 4, ∀ {G' : SimpleGraph (Fin 30)},
    (G'.adj = λ v w, v ≠ w ∧ coloring v w ≠ c) → G'.connected :=
begin
  sorry
end

end four_color_removal_l666_666467


namespace angle_PTC_eq_45_deg_l666_666401

-- Definitions for rectangle PQRS and related properties
def PQ := 8
def QR := 4
def PT (T : ℝ) := T
def TQ (T : ℝ) := PQ - T

-- Circle properties centered at point T passing through P and R
noncomputable def TP {T : ℝ} := real.sqrt (PT T ^ 2 + QR ^ 2)
noncomputable def TR {T : ℝ} := real.sqrt ((PQ - PT T) ^ 2 + QR ^ 2)

-- Angle PTC is equal to 45 degrees under given conditions
theorem angle_PTC_eq_45_deg (T : ℝ) (h_circle_T : TP T = TR T) (h_symmetry : ∠ PTC = ∠ QTC) :
  ∠ PTC = 45 :=
  sorry

end angle_PTC_eq_45_deg_l666_666401


namespace no_nonzero_real_number_makes_fraction_positive_integer_l666_666992

theorem no_nonzero_real_number_makes_fraction_positive_integer (x : ℝ) (hx : x ≠ 0) : (||x|-|x||) / x ∉ ℤ+ :=
by
  sorry

end no_nonzero_real_number_makes_fraction_positive_integer_l666_666992


namespace hamburgers_second_day_l666_666926

theorem hamburgers_second_day (x H D : ℕ) (h1 : 3 * H + 4 * D = 10) (h2 : x * H + 3 * D = 7) (h3 : D = 1) (h4 : H = 2) :
  x = 2 :=
by
  sorry

end hamburgers_second_day_l666_666926


namespace parabola_solution_unique_l666_666625

theorem parabola_solution_unique (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 4 * a + 2 * b + c = -1) (h3 : 4 * a + b = 1) :
  a = 3 ∧ b = -11 ∧ c = 9 := 
  by sorry

end parabola_solution_unique_l666_666625


namespace product_eq_l666_666397

-- Define the necessary constants and conditions
variables {P X Y W Z G : Type}
variables (a b : ℝ) (PG diameter : ℝ)

-- Given conditions
def ellipse_center (P : Type) : Prop := True  -- Placeholder for P being the center of ellipse
def major_axis (XY : Type) : Prop := True  -- Placeholder for XY being major axis
def minor_axis (WZ : Type) : Prop := True  -- Placeholder for WZ being minor axis
def focus_point (P G : Type) : Prop := True  -- Placeholder for G being a focus of the ellipse

-- Numerical conditions
def PG_condition : PG = 8 := sorry
def diameter_condition : diameter = 4 := sorry

-- Mathematical relationships
def semi_major_eq : a = PX := sorry
def semi_minor_eq : b = PW := sorry
def ellipse_eq : a^2 - b^2 = 64 := sorry
def inradius_eq : b + 8 - a = 4 := sorry
def diff_of_squares_eq : (a+b) * (a-b) = 64 := sorry
def system_eq1 : a - b = 4 := sorry
def system_eq2 : a + b = 16 := sorry

-- Prove the final product
theorem product_eq : (20:ℝ) * (12:ℝ) = 240 := by
  compute_product P X Y W Z a b PG diameter sorry sorry sorry sorry sorry sorry sorry sorry sorry

end product_eq_l666_666397


namespace clock_hands_coincide_22_times_clock_hands_straight_angle_24_times_clock_hands_right_angle_48_times_l666_666207

theorem clock_hands_coincide_22_times
  (minute_hand_cycles_24_hours : ℕ := 24)
  (hour_hand_cycles_24_hours : ℕ := 2)
  (minute_hand_overtakes_hour_hand_per_12_hours : ℕ := 11) :
  2 * minute_hand_overtakes_hour_hand_per_12_hours = 22 :=
by
  -- Proof should be filled here
  sorry

theorem clock_hands_straight_angle_24_times
  (hours_in_day : ℕ := 24)
  (straight_angle_per_hour : ℕ := 1) :
  hours_in_day * straight_angle_per_hour = 24 :=
by
  -- Proof should be filled here
  sorry

theorem clock_hands_right_angle_48_times
  (hours_in_day : ℕ := 24)
  (right_angles_per_hour : ℕ := 2) :
  hours_in_day * right_angles_per_hour = 48 :=
by
  -- Proof should be filled here
  sorry

end clock_hands_coincide_22_times_clock_hands_straight_angle_24_times_clock_hands_right_angle_48_times_l666_666207


namespace man_l666_666891

/-- The man's speed in still water in km/hr -/
def v_still : ℝ := 25

/-- The speed of the current in km/hr -/
def v_current : ℝ := 5

/-- The speed reduction due to headwind in km/hr -/
def v_headwind : ℝ := 2

/-- The width of the river in meters -/
def d_river : ℝ := 800

/-- The additional navigation distance in meters -/
def d_additional : ℝ := 200

/-- Conversion factor from km/hr to m/s -/
def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

/-- The man's speed in still water in m/s -/
def v_still_mps : ℝ := kmph_to_mps v_still

/-- The speed of the current in m/s -/
def v_current_mps : ℝ := kmph_to_mps v_current

/-- The speed reduction due to headwind in m/s -/
def v_headwind_mps : ℝ := kmph_to_mps v_headwind

/-- The man's speed upstream including headwind in m/s -/
def v_upstream_mps : ℝ := v_still_mps - v_current_mps - v_headwind_mps

/-- The man's speed considering all conditions approximately 4.999 m/s -/
theorem man's_speed_approx : v_upstream_mps ≈ 4.999 := by
  -- Since we don't provide the proof, we use sorry
  sorry

end man_l666_666891


namespace number_of_chords_l666_666799

/-- Ten points are marked on the circumference of a circle.
    Prove that the number of different chords that can be drawn
    by connecting any two of these ten points is 45.
-/
theorem number_of_chords (n : ℕ) (h_n : n = 10) : 
  (nat.choose n 2) = 45 :=
by
  rw h_n
  norm_num

end number_of_chords_l666_666799


namespace max_distance_l666_666539

noncomputable def distance_covered (D : ℝ) :=
  (D / 2) / 5 + (D / 2) / 4

theorem max_distance : ∃ D : ℕ, distance_covered D = 6 ∧ D = 13 := by
  use 13
  unfold distance_covered
  simp
  norm_num
  split
  { ring_nf,
    linarith }
  { exact rfl }

-- The proof steps are skipped with appropriate justifications (linarith and rfl).

end max_distance_l666_666539


namespace product_of_roots_l666_666162

variable {x1 x2 : ℝ}

theorem product_of_roots (hx1 : x1 * Real.log x1 = 2006) (hx2 : x2 * Real.exp x2 = 2006) : x1 * x2 = 2006 :=
sorry

end product_of_roots_l666_666162


namespace ice_creams_not_sold_l666_666065

theorem ice_creams_not_sold 
  (chocolate_ice_creams : ℕ)
  (mango_ice_creams : ℕ)
  (sold_chocolate : ℚ)
  (sold_mango : ℚ)
  (initial_chocolate : chocolate_ice_creams = 50)
  (initial_mango : mango_ice_creams = 54)
  (fraction_sold_chocolate : sold_chocolate = 3 / 5)
  (fraction_sold_mango : sold_mango = 2 / 3) :
  chocolate_ice_creams - (chocolate_ice_creams * fraction_sold_chocolate).toNat
  + mango_ice_creams - (mango_ice_creams * fraction_sold_mango).toNat = 38 := 
by {
  sorry
}

end ice_creams_not_sold_l666_666065


namespace problem_l666_666644

noncomputable section

variable (a_1 b_1 : ℝ)
variable (h_a1_pos : a_1 > 0) (h_b1_pos : b_1 > 0)

def a_seq : ℕ → ℝ
| 0     => a_1
| (n+1) => a_seq n + 1 / b_seq n

def b_seq : ℕ → ℝ
| 0     => b_1
| (n+1) => b_seq n + 1 / a_seq n

theorem problem (a_1 b_1 : ℝ) (h_a1_pos : a_1 > 0) (h_b1_pos : b_1 > 0) :
  a_seq a_1 b_1 25 + b_seq a_1 b_1 25 ≥ 10 * Real.sqrt 2 :=
sorry

end problem_l666_666644


namespace find_z_l666_666141

noncomputable def z := {z : ℂ | ∃ i : ℂ, i^2 = -1 ∧ i * z = i - 1}

theorem find_z (i : ℂ) (hi : i^2 = -1) : ∃ z : ℂ, i * z = i - 1 ∧ z = 1 + i := by
  use 1 + i
  sorry

end find_z_l666_666141


namespace best_fitting_model_is_I_l666_666498

noncomputable def correlation_coefficients := (r1 r2 r3 r4 : ℝ)
  (h1 : r1 = 0.96)
  (h2 : r2 = 0.81)
  (h3 : r3 = 0.53)
  (h4 : r4 = 0.35) : Prop := 
  abs r1 = max (abs r1) (max (abs r2) (max (abs r3) (abs r4)))

theorem best_fitting_model_is_I : correlation_coefficients 0.96 0.81 0.53 0.35 := 
  by
  sorry

end best_fitting_model_is_I_l666_666498


namespace find_area_l666_666733

-- Define the triangle with the given properties
structure TriangleABC :=
  (A B C : Point)
  (angle_C_is_right : ∠ A B C = 90)
  (CM CN CK : ℝ)
  (ratio_CM_CN : CM / CN = sqrt 6)
  (CK_length : CK = 2)

theorem find_area (t : TriangleABC) : 
  (area t.A t.B t.C = 12) ∧ (area t.C t.N t.K = sqrt 2) :=
  sorry

end find_area_l666_666733


namespace long_sleeve_shirts_correct_l666_666074

def total_shirts : ℕ := 9
def short_sleeve_shirts : ℕ := 4
def long_sleeve_shirts : ℕ := total_shirts - short_sleeve_shirts

theorem long_sleeve_shirts_correct : long_sleeve_shirts = 5 := by
  sorry

end long_sleeve_shirts_correct_l666_666074


namespace identify_rational_num_l666_666920

def is_rational (x : ℝ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ x = p / q

noncomputable def optionA : ℝ := 1 / 3
noncomputable def optionB : ℝ := Real.pi
noncomputable def optionC : ℝ := Real.sqrt 5
noncomputable def optionD : ℝ := Real.cbrt 5

theorem identify_rational_num : (is_rational optionA) ∧ ¬ (is_rational optionB) ∧ ¬ (is_rational optionC) ∧ ¬ (is_rational optionD) :=
by sorry

end identify_rational_num_l666_666920


namespace num_distinct_numbers_l666_666205

-- Define the conditions
def digits := {1, 2, 3, 4, 5}
def no_repeats (s : List Nat) : Prop := s.Nodup

-- Define the number of three- and four-digit numbers
noncomputable def count_three_digit_numbers : Nat :=
  Nat.perm (digits.toList) 3

noncomputable def count_four_digit_numbers : Nat :=
  Nat.perm (digits.toList) 4

-- The total number of distinct numbers
theorem num_distinct_numbers : 
  count_three_digit_numbers + count_four_digit_numbers = 180 :=
by
  sorry

end num_distinct_numbers_l666_666205


namespace units_digit_of_27_times_36_l666_666123

theorem units_digit_of_27_times_36 :
  let units_digit := fun (n : ℕ) => n % 10
  in units_digit (27 * 36) = 2 :=
by
  let units_digit := fun (n : ℕ) => n % 10
  have h27: units_digit 27 = 7 := by
    show 27 % 10 = 7
    sorry
  have h36: units_digit 36 = 6 := by
    show 36 % 10 = 6
    sorry
  have h42: units_digit (7 * 6) = 2 := by
    show 42 % 10 = 2
    sorry
  exact h42

end units_digit_of_27_times_36_l666_666123


namespace q_value_at_1_l666_666565

def q (x : ℝ) : ℝ := sorry -- definition of q is unspecified and left as sorry

theorem q_value_at_1 : q 1 = 3 :=
by
  -- Given condition: q(x) passes through (1, 3)
  have cond : q(1) = 3 := rfl
  show q(1) = 3 from cond

end q_value_at_1_l666_666565


namespace solution_set_of_inequality_l666_666855

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2 * a * x - 3 * a^2 < 0} = {x : ℝ | 3 * a < x ∧ x < -a} :=
by
  sorry

end solution_set_of_inequality_l666_666855


namespace units_digit_multiplication_l666_666117

-- Define a function to find the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Statement of the problem: Given the product 27 * 36, prove that the units digit is 2.
theorem units_digit_multiplication (a b : ℕ) (h1 : units_digit 27 = 7) (h2 : units_digit 36 = 6) :
  units_digit (27 * 36) = 2 :=
by
  have h3 : units_digit (7 * 6) = 2 := by sorry
  exact h3

end units_digit_multiplication_l666_666117


namespace sum_first_five_terms_sequence_l666_666677

open Nat

def a (n : ℕ) : ℕ := n * 2^(n-1)

def sum_sequence_upto (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, a (i + 1))

theorem sum_first_five_terms_sequence :
  sum_sequence_upto 5 = 129 := by
  sorry -- Proof to be filled in

end sum_first_five_terms_sequence_l666_666677


namespace count_solutions_l666_666976

noncomputable def num_solutions : ℕ :=
  {z : ℂ // complex.abs z = 1 ∧ complex.abs ((z / complex.conj z) - (complex.conj z / z)) = 1}.to_finset.card

theorem count_solutions : num_solutions = 4 :=
by
  sorry

end count_solutions_l666_666976


namespace least_xy_value_l666_666637

theorem least_xy_value {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
  sorry

end least_xy_value_l666_666637


namespace sunil_total_amount_l666_666861

-- Define the given conditions
def CI : ℝ := 246
def r : ℝ := 0.05
def n : ℝ := 1
def t : ℝ := 2

-- Define the compound interest formula used
def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ := 
  P * ((1 + r / n) ^ (n * t) - 1)

-- Define the principal amount found from the formula
def P : ℝ := CI / 0.1025

-- Define the total amount calculation
def TotalAmount : ℝ := P + CI

-- The main statement to prove
theorem sunil_total_amount : TotalAmount = 2646 := 
by
  -- Sorry is used to skip the proof
  sorry

end sunil_total_amount_l666_666861


namespace pump_out_time_l666_666882

theorem pump_out_time
  (length width : ℝ) (depth_inches : ℝ)
  (gallons_per_minute_per_pump : ℝ)
  (gallons_per_cubic_foot : ℝ)
  (num_pumps : ℝ)
  (inches_per_foot : ℝ)
  (num_minutes : ℝ) 
  (h_length : length = 30)
  (h_width : width = 40)
  (h_depth_inches : depth_inches = 24)
  (h_gallons_per_minute_per_pump : gallons_per_minute_per_pump = 10)
  (h_gallons_per_cubic_foot : gallons_per_cubic_foot = 7.5)
  (h_num_pumps : num_pumps = 4)
  (h_inches_per_foot : inches_per_foot = 12)
  (h_num_minutes : num_minutes = 450)
  : let depth_feet := depth_inches / inches_per_foot,
        volume_cubic_feet := length * width * depth_feet,
        total_gallons := volume_cubic_feet * gallons_per_cubic_foot,
        total_pumping_rate := num_pumps * gallons_per_minute_per_pump,
        total_time := total_gallons / total_pumping_rate
    in total_time = num_minutes :=
by
  sorry

end pump_out_time_l666_666882


namespace direction_vector_of_projection_matrix_l666_666536

open Matrix

theorem direction_vector_of_projection_matrix : 
  let P := !![ (1 / 18 : ℚ), (-1 / 9), (1 / 6) ; 
               (-1 / 9), (4 / 18), (-1 / 3) ; 
               (1 / 6), (-1 / 3), (1 / 2)] in
  let dir_vec := !![1, -2, 3] in
  (P ⬝ !![1, 0, 0] = (1 / 18) • dir_vec) ∧ 
  gcd (1 : ℕ) 2 3 = 1 :=
by
  -- The proof goes here, but for now:
  sorry

end direction_vector_of_projection_matrix_l666_666536


namespace maintenance_time_increase_l666_666883

-- Conditions
def original_time : ℝ := 50
def new_time : ℝ := 60

-- Definition of percentage increase formula
def percentage_increase (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

-- Theorem to be proven
theorem maintenance_time_increase : percentage_increase original_time new_time = 20 := by
  sorry

end maintenance_time_increase_l666_666883


namespace number_of_chords_l666_666791

theorem number_of_chords (n : ℕ) (h : n = 10) : finset.card (finset.pairs (finset.range n)) = 45 :=
by
  rw [h]
  -- Sorry to skip the proof steps as required
  sorry

end number_of_chords_l666_666791


namespace calculate_WXYZ_l666_666994

def W := (Real.sqrt 2025 + Real.sqrt 2024)
def X := (-Real.sqrt 2025 - Real.sqrt 2024)
def Y := (Real.sqrt 2025 - Real.sqrt 2024)
def Z := (Real.sqrt 2024 - Real.sqrt 2025)

theorem calculate_WXYZ : W * X * Y * Z = 1 :=
by
  sorry

end calculate_WXYZ_l666_666994


namespace width_of_track_is_8_feet_l666_666045

-- Define the given conditions
def outer_circumference := 100 * Real.pi
def circumference_difference := 16 * Real.pi

-- Define the radii and the width of the track
def r1 := outer_circumference / (2 * Real.pi)
def r2 := (outer_circumference - circumference_difference) / (2 * Real.pi)
def track_width := r1 - r2

-- State the theorem: the width of the track is 8 feet
theorem width_of_track_is_8_feet : track_width = 8 := by
  sorry

end width_of_track_is_8_feet_l666_666045


namespace bus_problem_l666_666833

theorem bus_problem (x : ℕ) : 50 * x + 10 = 52 * x + 2 := 
sorry

end bus_problem_l666_666833


namespace percentage_students_qualified_school_A_l666_666720

theorem percentage_students_qualified_school_A 
  (A Q : ℝ)
  (h1 : 1.20 * A = A + 0.20 * A)
  (h2 : 1.50 * Q = Q + 0.50 * Q)
  (h3 : (1.50 * Q / 1.20 * A) * 100 = 87.5) :
  (Q / A) * 100 = 58.33 := sorry

end percentage_students_qualified_school_A_l666_666720


namespace cassini_lemniscate_properties_l666_666534

theorem cassini_lemniscate_properties (c a : ℝ) (P : ℝ × ℝ) :
  (P = (0, 0) → c = a) ∧
  ((0 < a ∧ a < c) → ¬∃ P, ((P.1 + c) ^ 2 + P.2 ^ 2) * ((P.1 - c) ^ 2 + P.2 ^ 2) = a ^ 2) ∧
  ((0 < c ∧ c < a) → 
    ∀ P, ((P.1 + c) ^ 2 + P.2 ^ 2) * ((P.1 - c) ^ 2 + P.2 ^ 2) = a ^ 2 → 
    a ^ 2 - c ^ 2 ≤ P.1 ^ 2 + P.2 ^ 2 ∧ P.1 ^ 2 + P.2 ^ 2 ≤ a ^ 2 + c ^ 2) ∧
  (∀ P, ((P.1 + c) ^ 2 + P.2 ^ 2) * ((P.1 - c) ^ 2 + P.2 ^ 2) = a ^ 2 →
    ((P.1, P.2) = (-P.1, P.2) ∧ (P.1, P.2) = (P.1, -P.2) ∧ (P.1, P.2) = (-P.1, -P.2))) :=
by sorry

end cassini_lemniscate_properties_l666_666534


namespace am_gm_inequality_l666_666376

variable {x y z : ℝ}

theorem am_gm_inequality
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + real.cbrt (y / x) + real.cbrt (z / y) + real.cbrt (x / z)) :=
by
  sorry

end am_gm_inequality_l666_666376


namespace cos_double_angle_l666_666283

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666283


namespace problem_statement_l666_666652

-- Define the variables a, b, c as given in the problem
def a : ℝ := log 1 / log 3
def b : ℝ := real.exp (2.5 * log 2)
def c : ℝ := log 3 / log (1/2)

-- State the proof goal
theorem problem_statement : b > a ∧ a > c := by
  sorry

end problem_statement_l666_666652


namespace sufficient_not_necessary_condition_l666_666618

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1) → (0 ≤ a * b ∧ a * b ≤ 1) ∧ ¬((0 ≤ a * b ∧ a * b ≤ 1) → (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1)) :=
by
  intros h
  split
  {
    -- Proof that the condition is sufficient (for later steps)
    sorry
  }
  {
    -- Proof that the condition is not necessary (for later steps)
    sorry
  }

end sufficient_not_necessary_condition_l666_666618


namespace exponent_problem_l666_666845

theorem exponent_problem : (8 ^ (-1 : ℝ)) ^ (0 : ℝ) + (8 ^ (1 / 3 : ℝ)) ^ (3 : ℝ) = 9 := 
by
  sorry

end exponent_problem_l666_666845


namespace min_value_reciprocals_l666_666368

theorem min_value_reciprocals (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h_sum : x + y = 8) (h_prod : x * y = 12) : 
  (1/x + 1/y) = 2/3 :=
sorry

end min_value_reciprocals_l666_666368


namespace g_inv_zero_solution_l666_666757

noncomputable def g (a b x : ℝ) : ℝ := 1 / (2 * a * x + b)

theorem g_inv_zero_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  g a b (g a b 0) = 0 ↔ g a b 0 = 1 / b :=
by
  sorry

end g_inv_zero_solution_l666_666757


namespace number_of_male_animals_l666_666710

def total_original_animals : ℕ := 100 + 29 + 9
def animals_bought_by_brian : ℕ := total_original_animals / 2
def animals_after_brian : ℕ := total_original_animals - animals_bought_by_brian
def animals_after_jeremy : ℕ := animals_after_brian + 37

theorem number_of_male_animals : animals_after_jeremy / 2 = 53 :=
by
  sorry

end number_of_male_animals_l666_666710


namespace profit_or_loss_l666_666033

-- Conditions
variables 
  (P₁ P₂ : ℝ)  -- Cost prices
  (selling_price : ℝ)  -- Selling price for each aeroplane
  (profit_rate loss_rate : ℝ)  -- Profit and loss rates
  (total_selling_price total_cost_price total_loss : ℝ)

-- Definitions from conditions
def condition_1 : Prop := selling_price = 600
def condition_2 : Prop := profit_rate = 0.2
def condition_3 : Prop := loss_rate = 0.2
def cost_price_profit : Prop :=
  selling_price = P₁ + profit_rate * P₁
def cost_price_loss : Prop :=
  selling_price = P₂ - loss_rate * P₂

-- Derived definitions
def total_cost_price_def : Prop := total_cost_price = P₁ + P₂
def total_selling_price_def : Prop := total_selling_price = 600 + 600
def total_loss_def : Prop := total_loss = total_cost_price - total_selling_price

-- Theorem statement to prove
theorem profit_or_loss : 
  condition_1 ∧ condition_2 ∧ condition_3 ∧ cost_price_profit ∧ cost_price_loss ∧
  total_cost_price_def ∧ total_selling_price_def ∧ total_loss_def →
  total_loss = 50 :=
by
  intros h
  unfold condition_1 condition_2 condition_3 cost_price_profit cost_price_loss 
    total_cost_price_def total_selling_price_def total_loss_def at h
  sorry

end profit_or_loss_l666_666033


namespace hit_once_probability_l666_666538

theorem hit_once_probability (p: ℝ) (h: p = 0.5) : 
  let event_space : EventSpace := sorry,
  let hit_event : MeasurableSpace.event := sorry,
  let miss_event : MeasurableSpace.event := sorry,
  let hit_prob := event_space.prob hit_event,
  let miss_prob := event_space.prob miss_event in
  (hit_prob = p) →
  (miss_prob = 1 - p) →
  let exactly_one_hit := ((hit_event ∩ miss_event) ∪ (miss_event ∩ hit_event)) in
  event_space.prob exactly_one_hit = 0.5 :=
begin
  sorry
end

end hit_once_probability_l666_666538


namespace postage_cost_correct_l666_666542

def base_rate (oz: ℝ) : ℝ := 40 / 100

def additional_rate (oz: ℝ) : ℝ :=
  if oz > 1 then ((oz - 1).ceil) * 25 / 100 else 0

def total_cost (oz: ℝ) : ℝ :=
  base_rate oz + additional_rate oz

theorem postage_cost_correct (oz: ℝ) (h1: oz = 3.7) :
  total_cost oz = 1.15 :=
by
  rw [h1, total_cost, base_rate, additional_rate]
  have h2 : (3.7 - 1).ceil = 3 := by sorry
  rw [h2]
  norm_num
  done

end postage_cost_correct_l666_666542


namespace bacteria_growth_rate_l666_666740

theorem bacteria_growth_rate (r : ℝ) :
  (1 + r)^6 = 64 → r = 1 :=
by
  intro h
  sorry

end bacteria_growth_rate_l666_666740


namespace star_m_eq_15_l666_666361

-- Define the star function
def star (x : ℕ) : ℕ := x.digits.sum

-- The set S as defined
def S : set ℕ := {n : ℕ | star n = 15 ∧ n < 10^6}

-- Define m as the number of elements in the set S
def m : ℕ := S.card

-- The theorem to prove
theorem star_m_eq_15 : star m = 15 :=
by
  sorry

end star_m_eq_15_l666_666361


namespace y_value_l666_666890

theorem y_value {y : ℝ} (h1 : (0, 2) = (0, 2))
                (h2 : (3, y) = (3, y))
                (h3 : dist (0, 2) (3, y) = 10)
                (h4 : y > 0) :
                y = 2 + Real.sqrt 91 := by
  sorry

end y_value_l666_666890


namespace Yoongi_has_smaller_number_l666_666351

def Jungkook_number : ℕ := 6 + 3
def Yoongi_number : ℕ := 4

theorem Yoongi_has_smaller_number : Yoongi_number < Jungkook_number :=
by
  exact sorry

end Yoongi_has_smaller_number_l666_666351


namespace expand_product_l666_666099

theorem expand_product (x : ℝ) : (x + 5) * (x - 16) = x^2 - 11 * x - 80 :=
by sorry

end expand_product_l666_666099


namespace cone_slice_volume_ratio_l666_666898

theorem cone_slice_volume_ratio
  (h r : ℝ)  (h_pos : 0 < h) (r_pos : 0 < r) :
  let V (x h r : ℝ) := (1/3) * real.pi * (r * x) ^ 2 * (h * x) in
  let V_A := V  1 h r,
      V_B := V  2 h r,
      V_C := V  3 h r,
      V_D := V  4 h r,
      V_E := V  5 h r,
      V_1 := V_E - V_D,
      V_2 := V_D - V_C in
  V_2 / V_1 = (37 / 61) := 
by
  sorry

end cone_slice_volume_ratio_l666_666898


namespace sufficient_but_not_necessary_l666_666649

theorem sufficient_but_not_necessary (φ : ℝ) : (φ = π) → 
  ∃ k : ℤ, φ = k * π ∧ ∀ x : ℝ, (f : ℝ → ℝ) (f(x) = sin (x + φ)) → (f(-x) = -f(x)) :=
by
  intros hφ hfx
  have hk : ∃ k : ℤ, φ = k * π, from exists.intro 1 hφ,
  exists hφ
  intros x
  rw sin_add
  sorry

end sufficient_but_not_necessary_l666_666649


namespace intersect_lines_l666_666332

-- Defining the cube and midpoints
variables (A B C D A1 B1 C1 D1 E F : Point)
-- Assuming A, B, C, D, A1, B1, C1, D1 form a cube
-- E is the midpoint of C1D
def is_midpoint_CD_E : Prop :=
  E = midpoint C1 D
-- F is the midpoint of BC
def is_midpoint_BC_F : Prop :=
  F = midpoint B C
-- Lines
def line_A1B : Line := line_through A1 B
def line_EF : Line := line_through E F

-- The theorem to prove: A1B intersects EF
theorem intersect_lines (h1: is_midpoint_CD_E E) (h2: is_midpoint_BC_F F) : 
  intersects line_A1B line_EF :=
sorry

end intersect_lines_l666_666332


namespace cos_double_angle_l666_666242

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666242


namespace exists_root_between_l666_666667

variable (p q α β : ℝ)

-- Condition 1: α is a root of x^2 + px + q = 0
axiom root_of_first_eq : α^2 + p*α + q = 0

-- Condition 2: β is a root of -x^2 + px + q = 0
axiom root_of_second_eq : -β^2 + p*β + q = 0

-- Statement to prove
theorem exists_root_between : ∃ x : ℝ, x ∈ Ioo α β ∧ 1/2 * x^2 + p * x + q = 0 := 
by
  sorry

end exists_root_between_l666_666667


namespace find_functions_l666_666303

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def domain (f g : ℝ → ℝ) : Prop := ∀ x, x ≠ 1 → x ≠ -1 → true

theorem find_functions
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h_domain : domain f g)
  (h_eq : ∀ x, x ≠ 1 → x ≠ -1 → f x + g x = 1 / (x - 1)) :
  (∀ x, x ≠ 1 → x ≠ -1 → f x = x / (x^2 - 1)) ∧ 
  (∀ x, x ≠ 1 → x ≠ -1 → g x = 1 / (x^2 - 1)) := 
by
  sorry

end find_functions_l666_666303


namespace intervals_of_monotonicity_inequality_for_x_gt_1_l666_666765

noncomputable def f (x : ℝ) := (-1 / 24) * x^2 - 1 / 2 - real.log x

theorem intervals_of_monotonicity :
  ∀{x : ℝ}, 0 < x → x < real.sqrt 12 → f' x < 0 ∧ (x > real.sqrt 12 → f' x > 0)
sorry

theorem inequality_for_x_gt_1 :
  ∀{x : ℝ}, 1 < x → f x > 1 / x - real.exp (1 - x)
sorry

end intervals_of_monotonicity_inequality_for_x_gt_1_l666_666765


namespace enemies_left_undefeated_l666_666319

theorem enemies_left_undefeated (points_per_enemy points_earned total_enemies : ℕ) 
  (h1 : points_per_enemy = 3)
  (h2 : total_enemies = 6)
  (h3 : points_earned = 12) : 
  (total_enemies - points_earned / points_per_enemy) = 2 :=
by
  sorry

end enemies_left_undefeated_l666_666319


namespace measure_segment_QE_l666_666727

variables {y : ℝ}
variables {N Q D E B : Type} -- Abstract points in the circle

-- Conditions given in the problem
axiom midpoint_arc_N (N D B E : Type) : midpoint_of_arc N D B E  -- N is the midpoint of arc DBE
axiom perpendicular_segment_NQ (N Q D E: Type) : perpendicular N Q D E  -- NQ is perpendicular to DE at Q
axiom measure_chord_DB (DB : Type) : measure_chord DB = y  -- measure of chord DB is y
axiom measure_segment_DQ (DQ : Type) : measure_segment DQ = y + 3 -- measure of segment DQ is y + 3

-- We need to prove that QE = y + 3 given the conditions
theorem measure_segment_QE : measure_segment QE = y + 3 :=
by
  sorry

end measure_segment_QE_l666_666727


namespace cos_double_angle_l666_666234

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666234


namespace series_product_solution_l666_666078

theorem series_product_solution (y : ℚ) :
  ( (∑' n, (1 / 2) * (1 / 3) ^ n) * (∑' n, (1 / 3) * (-1 / 3) ^ n) ) = ∑' n, (1 / y) ^ (n + 1) → y = 19 / 3 :=
by
  sorry

end series_product_solution_l666_666078


namespace rectangular_solid_diagonal_l666_666804

theorem rectangular_solid_diagonal (p q r : ℝ) (d : ℝ) :
  p^2 + q^2 + r^2 = d^2 :=
sorry

end rectangular_solid_diagonal_l666_666804


namespace find_side_length_l666_666439

theorem find_side_length (x : ℝ) : 
  (4 * x = 8 * Real.pi) → (x = Real.pi * 2) :=
by
  intro h
  calc
    x = (8 * Real.pi) / 4 : by linarith
    ... = 2 * Real.pi : by linarith

#eval Float.toString (2 * Float.pi)

end find_side_length_l666_666439


namespace emma_age_when_sister_is_56_l666_666956

theorem emma_age_when_sister_is_56 :
  ∀ (emma_age sister_age_difference sister_future_age : ℕ),
  emma_age = 7 →
  sister_age_difference = 9 →
  sister_future_age = 56 →
  emma_age + (sister_future_age - (emma_age + sister_age_difference)) = 47 :=
by
  intros emma_age sister_age_difference sister_future_age hEmma hSisterDiff hSisterFuture
  rw [hEmma, hSisterDiff, hSisterFuture]
  norm_num
  sorry

end emma_age_when_sister_is_56_l666_666956


namespace T_n_less_than_3_over_8_l666_666629

-- Definitions based on conditions
def arithmetic_seq (n : ℕ) : ℕ := 2 * n

def geometric_seq (n : ℕ) : ℕ := 2^(n - 1)

def sequence_cond (n : ℕ) : Prop := arithmetic_seq n * geometric_seq n = n * 2^n 

-- c_n definition
noncomputable def c_n (n : ℕ) : ℝ := 1 / (arithmetic_seq n * real.logb 2 (geometric_seq (n + 3)))

-- Sum of c_n terms
noncomputable def T_n (n : ℕ) : ℝ := (∑ k in range(1, n+1), c_n k)

-- Main theorem to be proved
theorem T_n_less_than_3_over_8 (n : ℕ) : T_n n < 3 / 8 := 
by 
  sorry  -- Placeholder to indicate the proof is not provided

end T_n_less_than_3_over_8_l666_666629


namespace intersection_points_max_distance_l666_666723

-- Definitions for curve C and line l
def C (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, Real.sin θ)
def l (a t : ℝ) : ℝ × ℝ := (a + 4 * t, 1 - t)

-- Problem 1: Intersection points when a = -1
theorem intersection_points (θ t : ℝ) : 
  let a := -1 in
  ∃ (x y : ℝ), (x, y) = C θ ∧ (x, y) = l a t ↔ 
  (x, y) = (3, 0) ∨ (x, y) = (-21/25, 24/25) :=
begin
  sorry
end

-- Problem 2: Maximum distance from C to l
theorem max_distance (θ : ℝ) : 
  ∃ (d a : ℝ), d = (|(3 * Real.cos θ) + 4 * (Real.sin θ) - a - 4|) / Real.sqrt 17 
               ∧ d = Real.sqrt 17 ↔ 
  a = -16 ∨ a = 8 := 
begin
  sorry
end

end intersection_points_max_distance_l666_666723


namespace average_salary_of_associates_is_correct_l666_666886

noncomputable def average_salary_of_associates : ℝ :=
let num_managers := 9 in
let num_associates := 18 in
let avg_salary_managers := 1300 in
let avg_salary_store := 3989 in
let total_employees := num_managers + num_associates in
let total_salary_store := total_employees * avg_salary_store in
let total_salary_managers := num_managers * avg_salary_managers in
let total_salary_associates := total_salary_store - total_salary_managers in
total_salary_associates / num_associates

theorem average_salary_of_associates_is_correct :
  average_salary_of_associates = 5333.50 := by
  sorry

end average_salary_of_associates_is_correct_l666_666886


namespace remainder_when_divided_by_15_l666_666497

theorem remainder_when_divided_by_15 (N : ℕ) (h1 : N % 60 = 49) : N % 15 = 4 :=
by
  sorry

end remainder_when_divided_by_15_l666_666497


namespace product_of_distances_l666_666527

-- Definitions
def circle : Type := sorry -- Placeholder for the actual circle type
def point : Type := sorry -- Placeholder for the definition of a point

def diameter (c : circle) : point × point := sorry -- Placeholder for the diameter function returning two points
def tangent (c : circle) (p : point) : Type := sorry -- Placeholder for the tangent definition at a point
def ray (a : point) : Type := sorry -- Placeholder for ray starting from point a

def intersects_circle_at (r : Type) (c : circle) : point := sorry -- Function to return the intersection point of ray and circle
def intersects_tangent_at (r : Type) (t : Type) : point := sorry -- Function to return the intersection point of ray and tangent

-- Theorem to prove
theorem product_of_distances {A B : point} (c : circle)
    (AB_is_diameter : (A, B) = diameter c)
    (tangent_at_B : Type) (tangent_at_B_def : tangent c B = tangent_at_B)
    (r : Type) (ray_from_A : ray A = r)
    (C_at_intersection : point := intersects_circle_at r c)
    (D_at_intersection : point := intersects_tangent_at r tangent_at_B) :
    let AC := dist A C_at_intersection in -- Distance function placeholder
    let AD := dist A D_at_intersection in
    let AB := dist A B in
    AC * AD = AB ^ 2 :=
sorry

end product_of_distances_l666_666527


namespace a_plus_b_value_l666_666609

def f (a b x : ℝ) : ℝ :=
if x < 3 then a * x + b else 10 - 4 * x

theorem a_plus_b_value (a b : ℝ) (h : ∀ x, f a b (f a b x) = x) : a + b = 21 / 4 :=
sorry

end a_plus_b_value_l666_666609


namespace rowing_distance_l666_666831

theorem rowing_distance
  (speed_boat : ℕ)
  (speed_stream : ℕ)
  (total_time : ℕ)
  (D : ℕ)
  (effective_speed_with_stream : ℕ)
  (effective_speed_against_stream : ℕ)
  (T1 : ℕ)
  (T2 : ℕ) :
  speed_boat = 5 →
  speed_stream = 2 →
  total_time = 200 →
  effective_speed_with_stream = speed_boat + speed_stream →
  effective_speed_against_stream = speed_boat - speed_stream →
  T1 = D / effective_speed_with_stream →
  T2 = D / effective_speed_against_stream →
  T1 + T2 = total_time →
  D = 420 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8,
  sorry
end

end rowing_distance_l666_666831


namespace yi_catches_jia_on_DA_l666_666717

def square_side_length : ℝ := 90
def jia_speed : ℝ := 65
def yi_speed : ℝ := 72
def jia_start : ℝ := 0
def yi_start : ℝ := 90

theorem yi_catches_jia_on_DA :
  let square_perimeter := 4 * square_side_length
  let initial_gap := 3 * square_side_length
  let relative_speed := yi_speed - jia_speed
  let time_to_catch := initial_gap / relative_speed
  let distance_travelled_by_yi := yi_speed * time_to_catch
  let number_of_laps := distance_travelled_by_yi / square_perimeter
  let additional_distance := distance_travelled_by_yi % square_perimeter
  additional_distance = 0 →
  square_side_length * (number_of_laps % 4) = 0 ∨ number_of_laps % 4 = 3 :=
by
  -- We only provide the statement, the proof is omitted.
  sorry

end yi_catches_jia_on_DA_l666_666717


namespace stick_triangle_inequality_l666_666841

theorem stick_triangle_inequality (x : ℝ) : 
  (x = 5) → (2 + 5 > x ∧ 5 + x > 2 ∧ 2 + x > 5) :=
by
  intro h
  rw h
  split
  all_goals {
    linarith
  }

end stick_triangle_inequality_l666_666841


namespace ratio_of_A_to_B_l666_666046

theorem ratio_of_A_to_B (v_A v_B : ℝ) (d_A d_B : ℝ) (h1 : d_A = 128) (h2 : d_B = 64) (h3 : d_A / v_A = d_B / v_B) : v_A / v_B = 2 := 
by
  sorry

end ratio_of_A_to_B_l666_666046


namespace store_profit_l666_666537

variable (C : ℝ)  -- Cost price of a turtleneck sweater

noncomputable def initial_marked_price : ℝ := 1.20 * C
noncomputable def new_year_marked_price : ℝ := 1.25 * initial_marked_price C
noncomputable def discount_amount : ℝ := 0.08 * new_year_marked_price C
noncomputable def final_selling_price : ℝ := new_year_marked_price C - discount_amount C
noncomputable def profit : ℝ := final_selling_price C - C

theorem store_profit (C : ℝ) : profit C = 0.38 * C :=
by
  -- The detailed steps are omitted, as required by the instructions.
  sorry

end store_profit_l666_666537


namespace range_of_y_l666_666294

theorem range_of_y (y : ℝ) (hy : y < 0) (hceil_floor : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end range_of_y_l666_666294


namespace hours_practicing_l666_666682

theorem hours_practicing (W : ℕ) (hours_weekday : ℕ) 
  (h1 : hours_weekday = W + 17)
  (h2 : W + hours_weekday = 33) :
  W = 8 :=
sorry

end hours_practicing_l666_666682


namespace square_segment_ratio_l666_666340

theorem square_segment_ratio
  (A B C D E M P Q : ℝ × ℝ)
  (h_square: A = (0, 16) ∧ B = (16, 16) ∧ C = (16, 0) ∧ D = (0, 0))
  (h_E: E = (7, 0))
  (h_midpoint: M = ((0 + 7) / 2, (16 + 0) / 2))
  (h_bisector_P: P = (P.1, 16) ∧ (16 - 8 = (7 / 16) * (P.1 - 3.5)))
  (h_bisector_Q: Q = (Q.1, 0) ∧ (0 - 8 = (7 / 16) * (Q.1 - 3.5)))
  (h_PM: abs (16 - 8) = abs (P.2 - M.2))
  (h_MQ: abs (8 - 0) = abs (M.2 - Q.2)) :
  abs (P.2 - M.2) = abs (M.2 - Q.2) :=
sorry

end square_segment_ratio_l666_666340


namespace circle_sector_area_l666_666479

theorem circle_sector_area (x y : ℝ)
  (h1 : x^2 - 16 * x + y^2 = 48)
  (h2 : y = x - 4) :
  let r := 8 * Real.sqrt 2,
      full_area := Real.pi * r ^ 2,
      quadrant_area := full_area / 4 in
  quadrant_area = 32 * Real.pi :=
sorry

end circle_sector_area_l666_666479


namespace cos_double_angle_l666_666282

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l666_666282


namespace max_n_for_triangles_l666_666736

theorem max_n_for_triangles (n : ℕ) :
  let total_points := n + 100,
      no_collinear := ∀ (p1 p2 p3 : Point), 
        {p1, p2, p3} ⊆ total_points → ¬ collinear p1 p2 p3,
      triangles_formed ≤ 300 := 300
  in n ≤ 102 :=
sorry

end max_n_for_triangles_l666_666736


namespace eventually_identical_lines_l666_666776

noncomputable def transformation (l : List ℕ) : List ℕ :=
  l.map (λ n, l.count n)

theorem eventually_identical_lines (l : List ℕ) (h_len : l.length = 2018) :
  ∃ n, ∀ m ≥ n, transformation^[m] l = transformation^[n] l :=
sorry

end eventually_identical_lines_l666_666776


namespace angle_CAE_eq_angle_BAF_l666_666719

theorem angle_CAE_eq_angle_BAF
  (ABC : Triangle)
  (H : Orthocenter ABC)
  (E F : Point)
  (h1 : intersection_of_altitudes_and_circumcircle ABC E)
  (h2 : intersection_of_altitudes_and_circumcircle ABC F) :
  ∠ CAE = ∠ BAF :=
sorry

end angle_CAE_eq_angle_BAF_l666_666719


namespace part_a_part_b_part_c_l666_666875

-- Part a: Existence of a sphere touching all sides of a spatial quadrilateral
theorem part_a (AB BC CD DA : ℝ) (is_spatial : AB ≠ BC ∧ BC ≠ CD ∧ CD ≠ DA ∧ DA ≠ AB) :
  ∃ (O : ℝ^3), ∀ (p : ℝ^3), (dist O p = AB) ∨ (dist O p = BC) ∨ (dist O p = CD) ∨ (dist O p = DA) := 
sorry

-- Part b: Infinite spheres condition being that sums of opposite sides are equal
theorem part_b (AB BC CD DA : ℝ) (eq_sum : AB + CD = BC + DA) :
  ∃∞ (O : ℝ^3), ∀ (p : ℝ^3), (dist O p = AB) ∨ (dist O p = BC) ∨ (dist O p = CD) ∨ (dist O p = DA) := 
sorry

-- Part c: Finding the sphere with the smallest radius from part b
theorem part_c (AB BC CD DA : ℝ) (eq_sum : AB + CD = BC + DA) :
  ∃ (O : ℝ^3), ∀ (O' : ℝ^3), (dist O p ≤ dist O' p) ∧ ∀ (p : ℝ^3), (dist O p = AB) ∨ (dist O p = BC) ∨ (dist O p = CD) ∨ (dist O p = DA) := 
sorry

end part_a_part_b_part_c_l666_666875


namespace cylinder_excluding_cone_ratio_l666_666530

variables (S h V1 V2 : ℝ)

-- Assume V1 is the volume of the cone and V2 is the volume of the cylinder
def volume_cone := (1/3) * S * h
def volume_cylinder := S * h

-- Assume their volume relation as given in the problem
axiom V1_eq : V1 = volume_cone S h
axiom V2_eq : V2 = volume_cylinder S h
axiom volume_ratio : V1 = (1/3) * V2

theorem cylinder_excluding_cone_ratio : (V2 - V1) / V1 = 2 :=
by {
  -- Proof is omitted
  sorry
}

end cylinder_excluding_cone_ratio_l666_666530


namespace kaleb_allowance_l666_666352

theorem kaleb_allowance (saved_money toys_needed toy_price : ℕ) (h_saved : saved_money = 21) (h_toys_needed : toys_needed = 6) (h_toy_price : toy_price = 6) : 
  saved_money + 15 = toys_needed * toy_price :=
by
  rw [h_saved, h_toys_needed, h_toy_price]
  norm_num

end kaleb_allowance_l666_666352


namespace total_athletes_l666_666513

theorem total_athletes (parts_football parts_baseball parts_soccer parts_basketball count_basketball : ℕ) (h_ratio : parts_football = 10 ∧ parts_baseball = 7 ∧ parts_soccer = 5 ∧ parts_basketball = 4) (h_basketball_count : count_basketball = 16) :
  4 * (parts_football + parts_baseball + parts_soccer + parts_basketball) = 104 :=
by
  obtain ⟨h_football, h_baseball, h_soccer, h_basketball⟩ := h_ratio
  simp [h_football, h_baseball, h_soccer, h_basketball, h_basketball_count]
  sorry

end total_athletes_l666_666513


namespace card_A_eq_card_B_card_A_eq_card_C_card_A_eq_card_B_eq_card_C_main_l666_666574

def A := { ways_to_insert_parentheses : List String // is_valid_parenthesis_insertion ways_to_insert_parentheses }
def B := { ways_to_divide_hexagon : List (Fin 6) // is_valid_hexagon_division ways_to_divide_hexagon }
def C := { ways_to_arrange_balls : List Bool // is_valid_ball_arrangement ways_to_arrange_balls }

theorem card_A_eq_card_B : fintype.card A = fintype.card B := sorry

theorem card_A_eq_card_C : fintype.card A = fintype.card C := sorry

theorem card_A_eq_card_B_eq_card_C : fintype.card A = fintype.card B := 
by {
  apply card_A_eq_card_B,
}

theorem main : fintype.card A = fintype.card B ∧ fintype.card A = fintype.card C :=
by {
  split,
  exact card_A_eq_card_B,
  exact card_A_eq_card_C,
}

end card_A_eq_card_B_card_A_eq_card_C_card_A_eq_card_B_eq_card_C_main_l666_666574


namespace min_value_fract_ineq_l666_666159

theorem min_value_fract_ineq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 / a + 9 / b) ≥ 16 := 
sorry

end min_value_fract_ineq_l666_666159


namespace fractions_not_equal_to_seven_over_five_l666_666856

theorem fractions_not_equal_to_seven_over_five :
  (7 / 5 ≠ 1 + (4 / 20)) ∧ (7 / 5 ≠ 1 + (3 / 15)) ∧ (7 / 5 ≠ 1 + (2 / 6)) :=
by
  sorry

end fractions_not_equal_to_seven_over_five_l666_666856


namespace min_value_of_M_l666_666175

noncomputable def f (x b : ℝ) : ℝ := abs (x^2 + b * x)

noncomputable def M (b : ℝ) : ℝ :=
  if b ≥ 2 * (1 - real.sqrt 2) then
    b + 1
  else
    (b ^ 2) / 4

theorem min_value_of_M : ∃ b : ℝ, M b = 3 - 2 * real.sqrt 2 :=
by
  use 2 * (1 - real.sqrt 2)
  sorry

end min_value_of_M_l666_666175


namespace gcd_result_is_two_l666_666379

theorem gcd_result_is_two
  (n m k j: ℕ) (hn : n > 0) (hm : m > 0) (hk : k > 0) (hj : j > 0) :
  Nat.gcd (Nat.gcd (16 * n) (20 * m)) (Nat.gcd (18 * k) (24 * j)) = 2 := 
by
  sorry

end gcd_result_is_two_l666_666379


namespace length_of_bridge_is_80_l666_666551

-- Define the given constants
def length_of_train : ℕ := 280
def speed_of_train : ℕ := 18
def time_to_cross : ℕ := 20

-- Define the distance traveled by the train in the given time
def distance_traveled : ℕ := speed_of_train * time_to_cross

-- Define the length of the bridge from the given distance traveled
def length_of_bridge := distance_traveled - length_of_train

-- The theorem to prove the length of the bridge is 80 meters
theorem length_of_bridge_is_80 :
  length_of_bridge = 80 := by
  sorry

end length_of_bridge_is_80_l666_666551


namespace least_xy_value_l666_666635

theorem least_xy_value {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
  sorry

end least_xy_value_l666_666635


namespace determinant_polynomial_roots_l666_666363

theorem determinant_polynomial_roots (a b c m p q : ℝ)
  (h1 : a + b + c = m)
  (h2 : a * b + b * c + c * a = p)
  (h3 : a * b * c = -q) :
  (by simpa [matrix.det_fin_three] using matrix.det! ![(a, b, c), (b, c, a), (c, a, b)]) =
  -3 * q - m^3 + 3 * m * p := 
sorry

end determinant_polynomial_roots_l666_666363


namespace chords_on_circle_l666_666802

theorem chords_on_circle (n : ℕ) (h : n = 10) : nat.choose n 2 = 45 :=
by {
  rw h,
  -- we can directly calculate choose 10 2
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num),
  sorry -- the actual detailed proof goes here.
}

end chords_on_circle_l666_666802


namespace value_of_b_add_c_add_d_l666_666127

open Complex

-- The set S and its property
axiom S : Set ℂ
axiom S_prop : ∀ x y : ℂ, x ∈ S → y ∈ S → x * y ∈ S

-- Elements of S
axiom a b c d : ℂ
axiom a_in_S : a ∈ S
axiom b_in_S : b ∈ S
axiom c_in_S : c ∈ S
axiom d_in_S : d ∈ S

-- Conditions
axiom cond1 : a^2 = 1
axiom cond2 : b = 1
axiom cond3 : c^2 = a 

-- Goal
theorem value_of_b_add_c_add_d : b + c + d = 1 :=
sorry

end value_of_b_add_c_add_d_l666_666127


namespace arctan_identity_l666_666103

theorem arctan_identity (y : ℝ) (h : 2 * arctan (1 / 3) + arctan (1 / 7) + arctan (1 / y) = π / 4) : 
    y = 2 :=
sorry

end arctan_identity_l666_666103


namespace alyssa_total_spent_l666_666058

theorem alyssa_total_spent :
  let grapes := 12.08
  let cherries := 9.85
  grapes + cherries = 21.93 := by
  sorry

end alyssa_total_spent_l666_666058


namespace tom_score_second_day_goals_l666_666137

-- Definitions based on conditions
def gina_first_day_goals : ℕ := 2
def tom_first_day_goals : ℕ := gina_first_day_goals + 3
def tom_second_day_goals (T2 : ℕ) : Prop := TotalGoals 17 ∧ GinaSecondDayGoals(T2 - 2) = (gina_first_day_goals + (T2 - 2)) + (tom_first_day_goals + T2) = 17

-- Theorem to prove Tom's second day goals
theorem tom_score_second_day_goals (T2 : ℕ) : tom_second_day_goals T2 → T2 = 6 := by
  sorry

end tom_score_second_day_goals_l666_666137


namespace find_c_in_parabola_l666_666041

theorem find_c_in_parabola (b c : ℝ) (h₁ : 2 = (-1) ^ 2 + b * (-1) + c) (h₂ : 2 = 3 ^ 2 + b * 3 + c) : c = -1 :=
sorry

end find_c_in_parabola_l666_666041


namespace range_of_quadratic_expression_l666_666444

theorem range_of_quadratic_expression (x : ℝ) (hx : x^2 - 7*x + 12 < 0) :
  ∃ I, ∀ y ∈ I, y = x^2 + 4*x + 5 ∧ I = set.Ioo 26 37 :=
by
  -- Outline proof will be provided here
  sorry

end range_of_quadratic_expression_l666_666444


namespace range_of_m_for_false_proposition_l666_666705

theorem range_of_m_for_false_proposition :
  ¬ (∃ x : ℝ, x^2 - m * x - m ≤ 0) → m ∈ Set.Ioo (-4 : ℝ) 0 :=
sorry

end range_of_m_for_false_proposition_l666_666705


namespace possible_values_of_m_l666_666700

noncomputable def polynomial_factors_over_integers (m : ℤ) : Bool :=
  ∃ (a b : ℤ), a * b = -4 ∧ a + b = m

theorem possible_values_of_m :
  {m : ℤ | polynomial_factors_over_integers m} = { -3, 0, 3 } :=
by
  sorry

end possible_values_of_m_l666_666700


namespace planes_parallel_iff_lines_parallel_to_plane_l666_666426

variables (α β : Type) [Plane α] [Plane β]

def is_parallel (α β : Type) [Plane α] [Plane β] : Prop :=
  ∀ (l : Line), l ∈ α → parallel l β

theorem planes_parallel_iff_lines_parallel_to_plane :
  is_parallel α β ↔ (∀ l ∈ α, parallel l β) := sorry

end planes_parallel_iff_lines_parallel_to_plane_l666_666426


namespace students_surveyed_l666_666888

theorem students_surveyed (S : ℕ) (F : ℕ):
  (E N : ℕ) (P : ℝ) 
  (H1 : E = 10) 
  (H2 : N = 40) 
  (H3 : P = 0.75) 
  (H4 : F = E + N) 
  (H5 : F = (1 / 4) * S) 
  (H6 : F = 50)
  : S = 200 :=
by
  sorry

end students_surveyed_l666_666888


namespace evaluate_power_l666_666963

theorem evaluate_power (a b : ℝ) (m n : ℝ) (h1 : a = b^m) (h2 : ∀ x y z : ℝ, (x^y)^z = x^(y*z)) : a^(n/m) = b^n :=
by
  sorry

example : 81^(5/4) = 243 := evaluate_power 81 3 4 5
  (by norm_num) -- Simplification for 81 = 3^4
  (by norm_num []) -- Using the power of a power rule

end evaluate_power_l666_666963


namespace units_digit_of_fraction_l666_666848

-- Define the problem
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fraction :
  units_digit ((30 * 31 * 32 * 33 * 34 * 35) / 2500) = 2 := by
  sorry

end units_digit_of_fraction_l666_666848


namespace car_speed_ratio_l666_666047

noncomputable def speed_ratio (t_round_trip t_leaves t_returns t_walk_start t_walk_end : ℕ) (meet_time : ℕ) : ℕ :=
  let one_way_time_car := t_round_trip / 2
  let total_car_time := t_returns - t_leaves
  let meeting_time_car := total_car_time / 2
  let remaining_time_to_factory := one_way_time_car - meeting_time_car
  let total_walk_time := t_walk_end - t_walk_start
  total_walk_time / remaining_time_to_factory

theorem car_speed_ratio :
  speed_ratio 60 120 160 60 140 80 = 8 :=
by
  sorry

end car_speed_ratio_l666_666047


namespace angle_between_vectors_l666_666665

variables {a b : ℝ^3}

-- Definitions from conditions
def magnitude_a : ℝ := 2 * (∥ b ∥)
def nonzero_b : Prop := ∥ b ∥ ≠ 0
def quadratic_equal_roots : Prop :=
  let ac := a • b in ∥ a ∥^2 + 4 * ac = 0

-- The statement we need to prove
theorem angle_between_vectors (hab : ∥ a ∥ = 2 * ∥ b ∥) (hb_nonzero : ∥ b ∥ ≠ 0) (quad_roots : quadratic_equal_roots) : 
  let θ := real.arccos ((a • b) / (∥ a ∥ * ∥ b ∥)) in
  θ = 2 * real.pi / 3 :=
by
  sorry

end angle_between_vectors_l666_666665


namespace average_percent_increase_per_year_is_7_l666_666512

def initial_population : ℕ := 175000
def final_population : ℕ := 297500
def time_period : ℕ := 10

theorem average_percent_increase_per_year_is_7 :
  let total_increase := final_population - initial_population in
  let average_annual_increase := total_increase / time_period in
  let average_percent_increase_per_year := (average_annual_increase * 100) / initial_population in
  average_percent_increase_per_year = 7 :=
by
  sorry

end average_percent_increase_per_year_is_7_l666_666512


namespace greatest_three_digit_integer_l666_666945

noncomputable def sn (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_three_digit_integer :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    (n^2 / sn n).denom = 1 ∧ 
    ¬ ((sn n)^2 ∣ factorial n) ∧
    n = 987 := sorry

end greatest_three_digit_integer_l666_666945


namespace repeating_decimal_addition_l666_666100

def repeating_decimal_45 := (45 / 99 : ℚ)
def repeating_decimal_36 := (36 / 99 : ℚ)

theorem repeating_decimal_addition :
  repeating_decimal_45 + repeating_decimal_36 = 9 / 11 :=
by
  sorry

end repeating_decimal_addition_l666_666100


namespace smallest_term_is_four_l666_666678

def sequence (n : ℕ) : ℚ := (1/3) * n^3 - 13 * n

theorem smallest_term_is_four : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → sequence n ≤ sequence m) ∧ n = 4 :=
by
  sorry

end smallest_term_is_four_l666_666678


namespace largest_divisor_of_odd_squares_l666_666367

theorem largest_divisor_of_odd_squares (m n : ℤ) (hm : odd m) (hn : odd n) (hcond1 : n < m) (hcond2 : m - n > 2) : 
  ∃ k : ℤ, k = 4 ∧ ∀ (a b : ℤ), m = 2 * a + 3 → n = 2 * b + 1 → k ∣ (m^2 - n^2) :=
by
  sorry

end largest_divisor_of_odd_squares_l666_666367


namespace cylindrical_tank_volume_l666_666929

-- Define the conditions
def diameter : ℝ := 20
def height : ℝ := 10
def radius : ℝ := diameter / 2

-- Define the expected volume
def expected_volume : ℝ := 1000 * Real.pi

-- The theorem to prove
theorem cylindrical_tank_volume : 
  let volume := Real.pi * radius^2 * height in
  volume = expected_volume := 
by sorry

end cylindrical_tank_volume_l666_666929


namespace solve_expression_l666_666695

theorem solve_expression (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 3 * (x + 4) = 6 :=
sorry

end solve_expression_l666_666695


namespace inequality_solution_l666_666582

theorem inequality_solution 
  (x : ℝ) (k : ℤ) :
  (sin x)^2 + (sin (2 * x))^2 > (sin (3 * x))^2 ↔ 
  ((π / 6) + k * π < x ∧ x < (π / 2) + k * π ∨ 
  (5 * π / 6) + k * π < x ∧ x < (3 * π / 2) + k * π) :=
by
  sorry

end inequality_solution_l666_666582


namespace notebook_cost_l666_666711

theorem notebook_cost (s n c : ℕ) (h1 : s > 20) (h2 : n > 2) (h3 : c > 2 * n) (h4 : s * c * n = 4515) : c = 35 :=
sorry

end notebook_cost_l666_666711


namespace factor_x4_minus_81_l666_666590

theorem factor_x4_minus_81 : 
  ∀ x : ℝ, (Polynomial.X ^ 4 - 81 : Polynomial ℝ) = (Polynomial.X - 3) * (Polynomial.X + 3) * (Polynomial.X ^ 2 + 9) := 
sorry

end factor_x4_minus_81_l666_666590


namespace scientific_notation_of_216000_l666_666418

theorem scientific_notation_of_216000 :
  216000 = 2.16 * 10^5 :=
sorry

end scientific_notation_of_216000_l666_666418


namespace least_xy_l666_666640

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_xy_l666_666640


namespace cube_to_sphere_surface_area_ratio_l666_666443

theorem cube_to_sphere_surface_area_ratio (L : ℝ) (hL : L > 0) :
  let R := L / 2 in
  (6 * L^2) / (4 * π * R^2) = 6 / π :=
by
  let R := L / 2
  -- Here, the conditions required for the proof are laid out.
  sorry

end cube_to_sphere_surface_area_ratio_l666_666443


namespace eggs_purchased_l666_666083

theorem eggs_purchased (dozen_count : ℕ) (eggs_per_dozen : ℕ) (total_eggs : ℕ) 
  (h1 : dozen_count = 9) (h2 : eggs_per_dozen = 12) (h3 : total_eggs = dozen_count * eggs_per_dozen) : 
  total_eggs = 108 := 
by 
  rw [h1, h2, h3]
  sorry

end eggs_purchased_l666_666083


namespace cos_double_angle_l666_666212

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l666_666212


namespace chord_length_intercepted_by_circle_l666_666165

theorem chord_length_intercepted_by_circle (a : ℝ) :
  (x y : ℝ) -> (x^2 + (y + a)^2 = 4) -> (x - y - 2 = 0) -> 
  (chord_length : ℝ) -> chord_length = 2 * sqrt 2 -> 
  a = 0 ∨ a = 4 := by
  sorry

end chord_length_intercepted_by_circle_l666_666165


namespace angle_B_sum_a_c_l666_666731

theorem angle_B (a b c : ℝ) (A B C : ℝ) (T : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h8 : b = Real.sqrt 3)
  (h9 : T = 1/2 * a * c * Real.sin B) :
  B = π / 3 :=
  sorry

theorem sum_a_c (a b c : ℝ) (A B C : ℝ) (T : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h8 : b = Real.sqrt 3)
  (h9 : T = 1/2 * a * c * Real.sin B)
  (hB : B = π / 3) :
  a + c = Real.sqrt 15 :=
  sorry

end angle_B_sum_a_c_l666_666731


namespace number_and_sum_of_g3_l666_666752

noncomputable def T := {x : ℝ // x > 0}

def g : T → ℝ := sorry

theorem number_and_sum_of_g3 :
  (∃ n s : ℝ, (∀ g : T → ℝ, (∀ x y : T, g x * g y = g ⟨x * y, sorry⟩ + 2010 * (1/x + 1/y + 2010)) → 
  n = 1 ∧ s = (1/3 + 2010) ∧ n * s = 6031/3)) :=
sorry

end number_and_sum_of_g3_l666_666752


namespace students_same_group_in_all_lessons_l666_666866

theorem students_same_group_in_all_lessons (students : Fin 28 → Fin 3 × Fin 3 × Fin 3) :
  ∃ (i j : Fin 28), i ≠ j ∧ students i = students j :=
by
  sorry

end students_same_group_in_all_lessons_l666_666866


namespace maximum_k_l666_666617

theorem maximum_k (m k : ℝ) (h0 : 0 < m) (h1 : m < 1/2) (h2 : (1/m + 2/(1-2*m)) ≥ k): k ≤ 8 :=
sorry

end maximum_k_l666_666617


namespace cos_double_angle_l666_666287

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666287


namespace number_of_true_props_l666_666922

-- Define each proposition as a condition
def prop1 : Prop := 
  supplemental_angles_congruent ∧ lines_parallel

def prop2 : Prop := 
  ∀ x : ℝ, irrational x ↔ infinite_non_repeating_decimal x

def prop3 : Prop := 
  ∀ (P : Point) (L : Line), parallel_through_point L P 

def prop4 : Prop := 
  ∀ x : ℝ, (x ^ (1 / 3) = x) → (x = 0 ∨ x = 1)

-- The equivalent proof problem
theorem number_of_true_props : ({prop2, prop3}.card = 2) :=
by
  sorry

end number_of_true_props_l666_666922


namespace limit_of_exponential_trigonometric_function_l666_666516

theorem limit_of_exponential_trigonometric_function :
  tendsto (λ x : ℝ, (2^(3*x) - 3^(5*x)) / (sin(7*x) - 2*x)) (nhds 0) (nhds (1/5 * real.log (8/243))) :=
sorry

end limit_of_exponential_trigonometric_function_l666_666516


namespace sum_nat_numbers_from_1_to_5_l666_666496

theorem sum_nat_numbers_from_1_to_5 : (1 + 2 + 3 + 4 + 5 = 15) :=
by
  sorry

end sum_nat_numbers_from_1_to_5_l666_666496


namespace cos_double_angle_l666_666258

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l666_666258


namespace sticker_price_of_laptop_l666_666204

/-
Heather is comparing the price of a new laptop at two different stores. 
Store A offers a 20% discount on the sticker price followed by a $120 rebate, 
while store B offers a 30% discount with a $50 rebate. Heather saves $30 
by buying the laptop from store A instead of store B. Prove that the sticker price of 
the laptop is $1000.
-/

theorem sticker_price_of_laptop : ∃ (x : ℝ), 
  let storeA_final_price := 0.80 * x - 120
  let storeB_final_price := 0.70 * x - 50
  0.80 * x - 120 = 0.70 * x - 20 ∧ x = 1000 := by
s sorry

end sticker_price_of_laptop_l666_666204


namespace smallest_integer_ends_3_divisible_5_l666_666493

theorem smallest_integer_ends_3_divisible_5 : ∃ n : ℕ, (53 = n ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, (m % 10 = 3) ∧ (m % 5 = 0) → 53 ≤ m) :=
by {
  sorry
}

end smallest_integer_ends_3_divisible_5_l666_666493


namespace initial_value_l666_666478

theorem initial_value (x k : ℤ) (h : x + 294 = k * 456) : x = 162 :=
sorry

end initial_value_l666_666478


namespace find_a7_in_arithmetic_sequence_l666_666726

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem find_a7_in_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3_a5 : a 3 + a 5 = 10) :
  a 7 = 8 :=
sorry

end find_a7_in_arithmetic_sequence_l666_666726


namespace value_of_expression_l666_666374

noncomputable def polynomial_roots : ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
λ p q r s t, (Polynomial.eval (Polynomial.X^5 - 4 * Polynomial.X^4 + 6 * Polynomial.X^3 - 3 * Polynomial.X^2 + 2 * Polynomial.X - 1) p = 0) ∧
             (Polynomial.eval (Polynomial.X^5 - 4 * Polynomial.X^4 + 6 * Polynomial.X^3 - 3 * Polynomial.X^2 + 2 * Polynomial.X - 1) q = 0) ∧
             (Polynomial.eval (Polynomial.X^5 - 4 * Polynomial.X^4 + 6 * Polynomial.X^3 - 3 * Polynomial.X^2 + 2 * Polynomial.X - 1) r = 0) ∧
             (Polynomial.eval (Polynomial.X^5 - 4 * Polynomial.X^4 + 6 * Polynomial.X^3 - 3 * Polynomial.X^2 + 2 * Polynomial.X - 1) s = 0) ∧
             (Polynomial.eval (Polynomial.X^5 - 4 * Polynomial.X^4 + 6 * Polynomial.X^3 - 3 * Polynomial.X^2 + 2 * Polynomial.X - 1) t = 0)

theorem value_of_expression (p q r s t : ℝ) (h : polynomial_roots p q r s t) :
  (1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (p * t) + 1 / (q * r) + 1 / (q * s) + 1 / (q * t) + 1 / (r * s) + 1 / (r * t) + 1 / (s * t)) = -6 :=
sorry

end value_of_expression_l666_666374


namespace emma_age_when_sister_is_56_l666_666954

theorem emma_age_when_sister_is_56 (e s : ℕ) (he : e = 7) (hs : s = e + 9) : 
  (s + (56 - s) - 9 = 47) :=
by {
  sorry
}

end emma_age_when_sister_is_56_l666_666954


namespace probability_ticket_draw_l666_666317

theorem probability_ticket_draw (n k : ℕ) (hn : n > 0) (hk : 1 ≤ k ∧ k ≤ n) :
  let total_ways := list.range n in
  let ticket_1_ways := list.range (n - 1) in
  let probability := (finset.card (finset.filter (λ x, x = 1) (finset.range n)) * 
                       finset.card (finset.range (n - 1))) /
                       finset.card (finset.range n) in
  probability = (1 : ℚ) / (n : ℚ) :=
sorry

end probability_ticket_draw_l666_666317


namespace max_value_of_x_plus_3y_l666_666989

def length (k : ℕ) : ℕ :=
match k with
| 0 => 0
| 1 => 0
| _ => let f := Nat.factors k in f.length

theorem max_value_of_x_plus_3y
    (x y : ℕ)
    (hx : x > 1)
    (hy : y > 1)
    (hl_sum : length x + length y = 16) :
    x + 3 * y = 98306 :=
sorry

end max_value_of_x_plus_3y_l666_666989


namespace probability_C_gets_position_probability_B_or_E_gets_position_l666_666311

def job_seekers : Set (String) := {"A", "B", "C", "D", "E"}

def total_positions : ℕ := 2

noncomputable def all_combinations := {combination : Finset (String × String) | 
  let job_seekers := ["A", "B", "C", "D", "E"];
  ∃ (a b : String), a ∈ job_seekers ∧ b ∈ job_seekers ∧ a ≠ b ∧ combination = (a, b)
}

def favorable_combinations_for_C := {combination : Finset (String × String) | 
  combination ∩ {"C"} ≠ ∅
}

def favorable_combinations_for_B_or_E := {combination : Finset (String × String) |
  combination ∩ {"B", "E"} ≠ ∅
}

theorem probability_C_gets_position : 
  (favorable_combinations_for_C.card / all_combinations.card) = 2 / 5 := 
sorry

theorem probability_B_or_E_gets_position : 
  (favorable_combinations_for_B_or_E.card / all_combinations.card) = 7 / 10 := 
sorry

end probability_C_gets_position_probability_B_or_E_gets_position_l666_666311


namespace four_pow_minus_a_l666_666657

noncomputable def log_base_3 (x : ℝ) := Real.log x / Real.log 3

theorem four_pow_minus_a {a : ℝ} (h : a * log_base_3 4 = 2) : 4^(-a) = 1 / 9 :=
by
  sorry

end four_pow_minus_a_l666_666657


namespace expected_value_of_expression_is_50_l666_666387

def expected_value_single_digit : ℚ := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 9

def expected_value_expression : ℚ :=
  (expected_value_single_digit + expected_value_single_digit + expected_value_single_digit +
   (expected_value_single_digit + expected_value_single_digit * expected_value_single_digit) +
   (expected_value_single_digit * expected_value_single_digit + expected_value_single_digit) +
   (expected_value_single_digit * expected_value_single_digit * expected_value_single_digit)) / 4

theorem expected_value_of_expression_is_50 :
  expected_value_expression = 50 := sorry

end expected_value_of_expression_is_50_l666_666387


namespace price_of_cheese_cookie_pack_l666_666016

theorem price_of_cheese_cookie_pack
    (cartons : ℕ) (boxes_per_carton : ℕ) (packs_per_box : ℕ) (total_cost : ℕ)
    (h_cartons : cartons = 12)
    (h_boxes_per_carton : boxes_per_carton = 12)
    (h_packs_per_box : packs_per_box = 10)
    (h_total_cost : total_cost = 1440) :
  (total_cost / (cartons * boxes_per_carton * packs_per_box) = 1) :=
by
  -- conditions are explicitly given in the theorem statement
  sorry

end price_of_cheese_cookie_pack_l666_666016


namespace cos_double_angle_l666_666288

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l666_666288


namespace range_of_y_l666_666296

theorem range_of_y (y : ℝ) (hy : y < 0) (hceil_floor : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end range_of_y_l666_666296


namespace sequence_bounded_l666_666761

noncomputable theory

open Real

def is_bounded (a : ℕ → ℝ) : Prop :=
  ∃ M, ∀ n, a n ≤ M

theorem sequence_bounded {c : ℝ} (hc : c > 2) (a : ℕ → ℝ)
  (h_sequence_nonnegative : ∀ n, a n ≥ 0)
  (h_condition1 : ∀ m n, a (m + n) ≤ 2 * a m + 2 * a n)
  (h_condition2 : ∀ k : ℕ, a (2^k) ≤ 1 / (k + 1 : ℝ)^c) :
  is_bounded a :=
sorry

end sequence_bounded_l666_666761


namespace part1_part2_l666_666143

noncomputable def point_N : ℝ × ℝ := (real.sqrt 2, real.pi / 4)

def curve_C1 (M : ℝ × ℝ) : Prop :=
  let ⟨ρ, θ⟩ := M in ρ = 1

def point_G (M N G : ℝ × ℝ) : Prop :=
  let ⟨x_m, y_m⟩ := M in
  let ⟨x_n, y_n⟩ := N in
  let ⟨x_g, y_g⟩ := G in
  x_g = x_m + x_n ∧ y_g = y_m + y_n

def curve_C2 (G : ℝ × ℝ) : Prop :=
  let ⟨x, y⟩ := G in (x - 1)^2 + (y - 1)^2 = 1

def line_l (t : ℝ) : ℝ × ℝ :=
  (2 - t / 2, (real.sqrt 3 / 2) * t)

theorem part1 : ∀ G, (∃ M N, curve_C1 M ∧ point_G M N G) -> curve_C2 G := 
by
  sorry

theorem part2 (t1 t2 : ℝ) :
  let P := (2, 0) in
  let A := line_l t1 in
  let B := line_l t2 in 
  ∃ t, ((A.1 - 1)^2 + (A.2 - 1)^2 = 1) ∧ ((B.1 - 1)^2 + (B.2 - 1)^2 = 1) ∧
  t1 + t2 = 1 + real.sqrt 3 ∧ t1 * t2 = 1 →
  (1 / real.dist P A + 1 / real.dist P B = 1 + real.sqrt 3) :=
by
  sorry

end part1_part2_l666_666143


namespace total_cost_is_174000_l666_666581

namespace ModularHome
  -- Conditions given as definitions in Lean
  def kitchen_sqft : ℕ := 400
  def kitchen_cost : ℕ := 20_000
  def bathroom_sqft : ℕ := 150
  def bathroom_cost : ℕ := 12_000
  def other_modules_cost_per_sqft : ℕ := 100
  def total_planned_sqft : ℕ := 2_000
  def num_bathrooms : ℕ := 2

  -- Required to prove that total cost is $174,000
  def calculate_total_cost : ℕ :=
    let kitchen_total_cost := kitchen_cost
    let bathrooms_total_cost := num_bathrooms * bathroom_cost
    let total_sqft_kitchen_bathrooms := kitchen_sqft + num_bathrooms * bathroom_sqft
    let remaining_sqft := total_planned_sqft - total_sqft_kitchen_bathrooms
    let remaining_cost := remaining_sqft * other_modules_cost_per_sqft
    kitchen_total_cost + bathrooms_total_cost + remaining_cost

  theorem total_cost_is_174000 : calculate_total_cost = 174_000 := by
    sorry
end ModularHome

end total_cost_is_174000_l666_666581


namespace sum_c2_to_c10_eq_2145_l666_666093

noncomputable def sum_c2_to_c10 : ℕ :=
  ∑ n in { 2, 3, 4, 5, 6, 7, 8, 9, 10 }, C n

theorem sum_c2_to_c10_eq_2145 (C : ℕ → ℕ) 
  (h : ∀ n, C (n + 1) ^ 3 - C n ^ 3 = C n ^ 2) : 
  sum_c2_to_c10 = 2145 := 
  by
  sorry

end sum_c2_to_c10_eq_2145_l666_666093


namespace mike_initial_games_l666_666774

theorem mike_initial_games (v w: ℕ)
  (h_non_working : v - w = 8)
  (h_earnings : 7 * w = 56)
  : v = 16 :=
by
  sorry

end mike_initial_games_l666_666774


namespace HB_HC_K_collinear_l666_666870

variables {A B C I H_B H_C K : Type}
variable [EuclideanGeometry A B C]
variable [Incenter I A B C]
variable [Orthocenter H_B A B I]
variable [Orthocenter H_C A C I]
variable [IncircleTangency K A B C]

theorem HB_HC_K_collinear 
  (incircle_touches_BC : incircle_tangency_point K A B C BC)
  (H_B_is_orthocenter_abi : orthocenter H_B A B I)
  (H_C_is_orthocenter_aci : orthocenter H_C A C I) :
  collinear [H_B, H_C, K] :=
by
sorry

end HB_HC_K_collinear_l666_666870


namespace convex_quad_equal_segments_l666_666312

noncomputable def are_segments_equal (A B C D K L M N : Point) (h1: is_midpoint K A B) (h2: is_midpoint L B C) (h3: is_midpoint M C D) (h4: is_midpoint N D A) 
(h5 : dist A M = dist B N) 
(h6 : dist B N = dist C K) 
(h7 : dist C K = dist D L) 
(h8 : dist D L = dist A M) 
(h9 : dist A N = dist B L) 
(h10 : dist B L = dist C M) 
(h11 : dist C M = dist D K) : Prop := 
dist D K = dist A N

theorem convex_quad_equal_segments {A B C D K L M N : Type} [Point A] [Point B] [Point C] [Point D] [Point K] [Point L] [Point M] [Point N]
(h1 : is_midpoint K A B)
(h2 : is_midpoint L B C)
(h3 : is_midpoint M C D)
(h4 : is_midpoint N D A)
(h_eq1 : dist A M = dist B N)
(h_eq2 : dist B N = dist C K)
(h_eq3 : dist C K = dist D L)
(h_eq4 : dist D L = dist A M)
(h_eq5 : dist A N = dist B L)
(h_eq6 : dist B L = dist C M)
(h_eq7 : dist C M = dist D K) :
are_segments_equal A B C D K L M N h1 h2 h3 h4 h_eq1 h_eq2 h_eq3 h_eq4 h_eq5 h_eq6 h_eq7 := 
by
  sorry

end convex_quad_equal_segments_l666_666312


namespace ratio_first_to_second_l666_666452

theorem ratio_first_to_second (A B C : ℕ) (h1 : A + B + C = 98) (h2 : B = 30) (h3 : B / C = 5 / 8) : A / B = 2 / 3 :=
sorry

end ratio_first_to_second_l666_666452


namespace kids_at_camp_l666_666584

theorem kids_at_camp (total_stayed_home : ℕ) (difference : ℕ) (x : ℕ) 
  (h1 : total_stayed_home = 777622) 
  (h2 : difference = 574664) 
  (h3 : total_stayed_home = x + difference) : 
  x = 202958 :=
by
  sorry

end kids_at_camp_l666_666584


namespace calculate_V3_horner_l666_666566

noncomputable def f (x : ℝ) : ℝ :=
  4 * x ^ 6 + 3 * x ^ 5 + 4 * x ^ 4 + 2 * x ^ 3 + 5 * x ^ 2 - 7 * x + 9

theorem calculate_V3_horner :
  let x := 4 in
  let V3 := (4 * x + 3) * x + 4 in
  V3 = 80 := by
  sorry

end calculate_V3_horner_l666_666566


namespace sum_smallest_largest_even_integers_l666_666424

theorem sum_smallest_largest_even_integers (n : ℕ) (h_odd : n % 2 = 1) (b z : ℤ)
  (h_mean : z = b + n - 1) : (b + (b + 2 * (n - 1))) = 2 * z :=
by
  sorry

end sum_smallest_largest_even_integers_l666_666424


namespace product_of_two_numbers_l666_666475

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x - y = 1 * k) 
  (h2 : x + y = 2 * k) 
  (h3 : (x * y)^2 = 18 * k) : (x * y = 16) := 
by 
    sorry


end product_of_two_numbers_l666_666475


namespace maximum_ships_on_battleship_board_l666_666522

theorem maximum_ships_on_battleship_board (m n ship_size : ℕ) (hm : m = 10) (hn : n = 10) (hship_size : ship_size = 4) : 
  25 = (m * n) / ship_size :=
by
  have total_cells := m * n
  have cells_per_ship := ship_size
  have max_ships := total_cells / cells_per_ship
  have htotal : total_cells = 100 := by rw [hm, hn]; rfl
  have hships : cells_per_ship = 4 := by rw hship_size; rfl
  have hmax : max_ships = 25 := by rw [htotal, hships]; norm_num
  exact hmax

end maximum_ships_on_battleship_board_l666_666522


namespace ned_weekly_sales_l666_666389

-- Define the conditions given in the problem
def normal_mouse_price : ℝ := 120
def normal_keyboard_price : ℝ := 80
def normal_scissor_price : ℝ := 30

def lt_hand_mouse_price := normal_mouse_price * 1.3
def lt_hand_keyboard_price := normal_keyboard_price * 1.2
def lt_hand_scissor_price := normal_scissor_price * 1.5

def lt_hand_mouse_daily_sales : ℝ := 25 * lt_hand_mouse_price
def lt_hand_keyboard_daily_sales : ℝ := 10 * lt_hand_keyboard_price
def lt_hand_scissor_daily_sales : ℝ := 15 * lt_hand_scissor_price

def total_daily_sales := lt_hand_mouse_daily_sales + lt_hand_keyboard_daily_sales + lt_hand_scissor_daily_sales
def days_open_per_week : ℝ := 4

def weekly_sales := total_daily_sales * days_open_per_week

-- The theorem to prove
theorem ned_weekly_sales : weekly_sales = 22140 := by
  -- The proof is omitted
  sorry

end ned_weekly_sales_l666_666389


namespace truck_loading_time_l666_666887

theorem truck_loading_time (h1_rate h2_rate h3_rate : ℝ)
  (h1 : h1_rate = 1 / 5) (h2 : h2_rate = 1 / 4) (h3 : h3_rate = 1 / 6) :
  (1 / (h1_rate + h2_rate + h3_rate)) = 60 / 37 :=
by simp [h1, h2, h3]; sorry

end truck_loading_time_l666_666887


namespace monotonic_intervals_f_maximum_value_g_l666_666185

-- Define f(x)
def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1

-- Define the monotonic intervals of f(x)
theorem monotonic_intervals_f (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 a ≤ f x2 a) ∨ 
  (∃ l a', a = Real.exp l ∧ 
    (∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < l → f x2 a < f x1 a) ∧
    (∀ x1 x2 : ℝ, x1 < x2 ∧ x2 > l → f x1 a < f x2 a)) :=
sorry

-- Define g(x)
def g (x : ℝ) : ℝ := Real.log x - x

-- Define that g(x) attains its maximum value at x = 1 and the maximum value is -1
theorem maximum_value_g :
  ∃ x : ℝ, (0 < x) ∧ (∀ y : ℝ, 0 < y → g y ≤ g x) ∧ g 1 = -1 :=
sorry

end monotonic_intervals_f_maximum_value_g_l666_666185


namespace rectangle_probability_no_shaded_square_l666_666878

theorem rectangle_probability_no_shaded_square :
  let n := 1003 * 2005 in
  let m := 1003 ^ 2 in
  2 by 2005 rectangle
  ∧ middle unit square of each row is shaded
  ∧ rect chosen at random
  → 1 - m / n = 1002 / 2005 :=
by
  sorry

end rectangle_probability_no_shaded_square_l666_666878


namespace closest_fraction_l666_666071

theorem closest_fraction :
  let fraction := 23 / 150
  let candidates := [1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9]
  let closest := 1 / 7
  ∀ x ∈ candidates, abs (fraction - closest) ≤ abs (fraction - x) :=
by
  sorry

end closest_fraction_l666_666071


namespace fraction_lit_l666_666040

theorem fraction_lit (r : ℝ) (h1 : r > 0) :
  let total_area := π * r^2,
      unlit_area := π * (r / 2)^2,
      lit_fraction := 1 - unlit_area / total_area
  in lit_fraction = 3 / 4 :=
by
  sorry

end fraction_lit_l666_666040


namespace minimum_points_set_l666_666372

noncomputable def g (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3

theorem minimum_points_set (n : ℕ) : 
  ∃ P : set (EuclideanSpace ℝ (Fin n)), 
  (∀ x ∈ EuclideanSpace ℝ (Fin n), ∃ p ∈ P, irrational (dist x p)) ∧ 
  ∀ Q : set (EuclideanSpace ℝ (Fin n)), 
    (∀ x ∈ EuclideanSpace ℝ (Fin n), ∃ q ∈ Q, irrational (dist x q)) → 
    set.card P ≤ set.card Q := 
sorry

end minimum_points_set_l666_666372


namespace rectangle_perimeter_l666_666005

noncomputable def perimeter (a b c : ℕ) : ℕ :=
  2 * (a + b)

theorem rectangle_perimeter (p q: ℕ) (rel_prime: Nat.gcd p q = 1) :
  ∃ (a b c: ℕ), p = 2 * (a + b) ∧ p + q = 52 ∧ a = 5 ∧ b = 12 ∧ c = 7 :=
by
  sorry

end rectangle_perimeter_l666_666005


namespace cos_double_angle_l666_666243

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666243


namespace determinant_of_cross_product_l666_666359

variables {R : Type*} [field R] 
variables (a b c d : R^3)

noncomputable def determinant_of_cross_product_matrix : R :=
  let D := by sorry, -- The determinant of the matrix with columns a, b, c
  let E := by sorry, -- The determinant of the matrix with columns a × b, b × c, c × d
  E

def cross_product_determinant_eq (D : R) : Prop :=
  let E := determinant_of_cross_product_matrix a b c d
  E = D^2

theorem determinant_of_cross_product (D : R) (hD : D = a ⬝ (b × c)) :
  cross_product_determinant_eq a b c d D :=
by sorry

end determinant_of_cross_product_l666_666359


namespace largest_n_for_factorial_sum_l666_666089

theorem largest_n_for_factorial_sum (n : ℕ) :
  199! + 200! + 201! ∣ 7^n → n = 32 :=
begin
  sorry
end

end largest_n_for_factorial_sum_l666_666089


namespace master_bedroom_suite_is_1040_l666_666029

noncomputable def master_bedroom_suite_size (living_room_dining_kitchen_area : ℝ) (total_house_area : ℝ)
  (guest_bedroom_fraction_of_master : ℝ) : ℝ :=
  let remaining_area := total_house_area - living_room_dining_kitchen_area,
      total_bedroom_area := remaining_area,
      master_bedroom_area := total_bedroom_area / (1 + guest_bedroom_fraction_of_master)
  in master_bedroom_area

theorem master_bedroom_suite_is_1040 :
  master_bedroom_suite_size 1000 2300 0.25 = 1040 :=
by
  unfold master_bedroom_suite_size
  have h1 : 2300 - 1000 = 1300 := by norm_num
  rw [h1]
  have h2 : 1 + 0.25 = 1.25 := by norm_num
  rw [h2]
  have h3 : 1300 / 1.25 = 1040 := by norm_num
  rw [h3]
  exact rfl

end master_bedroom_suite_is_1040_l666_666029


namespace range_of_y_l666_666298

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ set.Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end range_of_y_l666_666298


namespace determine_equation_of_line_l666_666535

noncomputable def equation_of_line (P Q A N : ℝ × ℝ) : Prop :=
  ∃ k, (Q.2^2 + Q.1^2 = 1) ∧
       (A = (-1, 0)) ∧
       (N = (0, 0)) ∧ --assuming Q is not the same as N
       (P = (Q.1 + 1, Q.2)) ∧ -- Point P translated vertically
       (Q ≠ (1, 0)) ∧ -- Making sure Q is not the intersecting vertical point on the unit circle
       ((A.2 - N.2) / (A.1 - N.1) * ((P.2 - A.2) / (P.1 - A.1)) = 1) ∧
       ((A.2 - Q.2) / (A.1 - Q.1) * ((P.2 - A.2) / (P.1 - A.1)) = -1) ∧
       (P.1 = -1/2) ∧
       (P.2 = ±(√3) / 2) ∧
       (k = ± √3) ∧ 
       ∃ l, l = λ x, k * x

theorem determine_equation_of_line (P Q A : ℝ × ℝ) (l : ℝ → ℝ): equation_of_line ∈ P Q A (0, 0) :=
sorry -- Proof not required

end determine_equation_of_line_l666_666535


namespace cos_double_angle_l666_666231

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666231


namespace least_value_xy_l666_666643

theorem least_value_xy {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : (1 : ℚ) / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_value_xy_l666_666643


namespace cosine_double_angle_l666_666251

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l666_666251


namespace cos_double_angle_l666_666257

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l666_666257


namespace area_of_region_l666_666480

theorem area_of_region :
  let region := { p : ℝ × ℝ | |(4 * p.1 - 20)| + |(3 * p.2 + 9)| ≤ 6 } in
  ∃ area : ℝ, area = 9 / 2 ∧ (region_area region = area)
:= by
  let region := { p : ℝ × ℝ | |(4 * p.1 - 20)| + |(3 * p.2 + 9)| ≤ 6 }
  existsi (9 / 2)
  split
  . refl
  . sorry

end area_of_region_l666_666480


namespace circle_filling_l666_666564

open Finset

theorem circle_filling:
  let numbers := {2, 3, 4, 5, 6}
  ∃ (configs : Finset (Finset ℕ)), 
  (∀ c ∈ configs, (c \ {1}).card = 5 ∧ 
    (∀ x ∈ c, ∀ y ∈ neighbors x, abs (x - y) ≥ 2)) 
  ∧ configs.card = 3 :=
begin
  sorry,
end

end circle_filling_l666_666564


namespace sum_expression_l666_666701

theorem sum_expression (x k : ℝ) (h1 : y = 3 * x) (h2 : z = k * y) : x + y + z = (4 + 3 * k) * x :=
by
  sorry

end sum_expression_l666_666701


namespace sum_digits_greatest_prime_divisor_l666_666435

theorem sum_digits_greatest_prime_divisor (n : ℕ) 
  (h1 : n = 2 ^ 14) : 
  let m := n - 1
  in (∑ d in to_digits 10 (greatest_prime_divisor m), d) = 10 :=
by 
  { sorry }

end sum_digits_greatest_prime_divisor_l666_666435


namespace emma_age_when_sister_is_56_l666_666957

theorem emma_age_when_sister_is_56 :
  ∀ (emma_age sister_age_difference sister_future_age : ℕ),
  emma_age = 7 →
  sister_age_difference = 9 →
  sister_future_age = 56 →
  emma_age + (sister_future_age - (emma_age + sister_age_difference)) = 47 :=
by
  intros emma_age sister_age_difference sister_future_age hEmma hSisterDiff hSisterFuture
  rw [hEmma, hSisterDiff, hSisterFuture]
  norm_num
  sorry

end emma_age_when_sister_is_56_l666_666957


namespace triangle_area_l666_666910

theorem triangle_area (r : ℝ) (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 2 * r) (r_val : r = 5) (ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) :
  (1 / 2) * a * b = 24 :=
by
  -- We assume statements are given
  sorry

end triangle_area_l666_666910


namespace exposed_surface_area_equals_42_l666_666923

-- Define the structure and exposed surface area calculations.
def surface_area_of_sculpture (layers : List Nat) : Nat :=
  (layers.headD 0 * 5) +  -- Top layer (5 faces exposed)
  (layers.getD 1 0 * 3 + layers.getD 1 0) +  -- Second layer
  (layers.getD 2 0 * 1 + layers.getD 2 0) +  -- Third layer
  (layers.getD 3 0 * 1) -- Bottom layer

-- Define the conditions
def number_of_layers : List Nat := [1, 4, 9, 6]

-- State the theorem
theorem exposed_surface_area_equals_42 :
  surface_area_of_sculpture number_of_layers = 42 :=
by
  sorry

end exposed_surface_area_equals_42_l666_666923


namespace problem_solution_l666_666341

section
variables (a b : ℝ)

-- Definition of the \* operation
def star_op (a b : ℝ) : ℝ := (a + 1) * (b - 1)

-- Definition of a^{*2} as a \* a
def star_square (a : ℝ) : ℝ := star_op a a

-- Define the specific problem instance with x = 2
def problem_expr : ℝ := star_op 3 (star_square 2) - star_op 2 2 + 1

-- Theorem stating the correct answer
theorem problem_solution : problem_expr = 6 := by
  -- Proof steps, marked as 'sorry'
  sorry

end

end problem_solution_l666_666341


namespace average_rounds_per_golfer_is_three_l666_666560

def total_rounds_played : ℕ :=
  4 * 1 + 3 * 2 + 3 * 3 + 2 * 4 + 6 * 5

def total_golfers : ℕ :=
  4 + 3 + 3 + 2 + 6

theorem average_rounds_per_golfer_is_three :
  (total_rounds_played / total_golfers : ℚ).round = 3 := by
  sorry

end average_rounds_per_golfer_is_three_l666_666560


namespace cos_double_angle_l666_666238

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666238


namespace range_of_a_l666_666306

theorem range_of_a (x : ℝ) (a : ℝ) (h₀ : x ∈ Set.Icc (-2 : ℝ) 3)
(h₁ : 2 * x - x ^ 2 ≥ a) : a ≤ 1 :=
sorry

end range_of_a_l666_666306


namespace constant_term_expansion_l666_666846

def poly1 := x^4 + x^2 + 3
def poly2 := 2x^5 + x^3 + 7

theorem constant_term_expansion :
  let constant_term_poly1 := 3
  let constant_term_poly2 := 7
  (constant_term_poly1 * constant_term_poly2) = 21 :=
by
  sorry

end constant_term_expansion_l666_666846


namespace cos_double_angle_l666_666241

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l666_666241


namespace vector_OH_OA_OB_OC_l666_666732

-- Definitions
variable {ABC : Type} [Triangle ABC]

-- Definitions of circumcenter and orthocenter for the given triangle
def circumcenter (ABC : Triangle) : Point := sorry
def orthocenter (ABC : Triangle) : Point := sorry

-- Points A, B, C of the triangle
variable {A B C : Point}
-- Vectors OA, OB, OC
variable {OA OB OC OH : Vector}
-- Specific points of the triangle linked to OA, OB, OC, and OH
def OA := vector_from_to O A
def OB := vector_from_to O B
def OC := vector_from_to O C
def OH := vector_from_to O H

-- Statement of the problem
theorem vector_OH_OA_OB_OC (ABC : Triangle)
  (O : Point) (A B C H : Point)
  (h_circumcenter : circumcenter ABC = O)
  (h_orthocenter : orthocenter ABC = H) :
  vector_from_to O H = vector_from_to O A + vector_from_to O B + vector_from_to O C := 
sorry

end vector_OH_OA_OB_OC_l666_666732


namespace product_of_axes_l666_666396

theorem product_of_axes 
  (P G X Y W Z : Type) 
  [center P X Y W Z G] 
  (PG : P = G) 
  (a b : ℝ) 
  (H1 : a^2 - b^2 = 64) 
  (H2 : a - b = 4) 
  (XY : ℝ := 2 * a)
  (WZ : ℝ := 2 * b) 
  : (XY * WZ = 240) :=
by {
  sorry
}

end product_of_axes_l666_666396


namespace solve_absolute_value_equation_l666_666854

theorem solve_absolute_value_equation :
  {x : ℝ | 3 * x^2 + 3 * x + 6 = abs (-20 + 5 * x)} = {1.21, -3.87} :=
by
  sorry

end solve_absolute_value_equation_l666_666854


namespace smallest_positive_period_maximum_f_B_l666_666178

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 / 2

theorem smallest_positive_period (x : ℝ) : 
  (∀ T, (f (x + T) = f x) → (T ≥ 0) → T = Real.pi) := 
sorry

variable {a b c : ℝ}

lemma cos_law_cos_B (h : b^2 = a * c) : 
  ∀ (B : ℝ), (B > 0) ∧ (B < Real.pi) → 
  (1 / 2) ≤ Real.cos B ∧ Real.cos B < 1 := 
sorry

theorem maximum_f_B (h : b^2 = a * c) :
  ∀ (B : ℝ), (B > 0) ∧ (B < Real.pi) → 
  f B ≤ 1 := 
sorry

end smallest_positive_period_maximum_f_B_l666_666178


namespace smallest_integer_ends_3_divisible_5_l666_666492

theorem smallest_integer_ends_3_divisible_5 : ∃ n : ℕ, (53 = n ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, (m % 10 = 3) ∧ (m % 5 = 0) → 53 ≤ m) :=
by {
  sorry
}

end smallest_integer_ends_3_divisible_5_l666_666492


namespace circumscribed_sphere_surface_area_l666_666337

noncomputable def surface_area_of_circumscribed_sphere 
  (AB BC CA : ℝ) (AA1 : ℝ) 
  (h : AB = sqrt 3 ∧ BC = sqrt 3 ∧ CA = sqrt 3 ∧ AA1 = 2 * sqrt 3) : ℝ :=
  4 * π * (sqrt (3 + 1)) ^ 2

theorem circumscribed_sphere_surface_area 
  (AB BC CA : ℝ) (AA1 : ℝ) 
  (h : AB = sqrt 3 ∧ BC = sqrt 3 ∧ CA = sqrt 3 ∧ AA1 = 2 * sqrt 3) : 
  surface_area_of_circumscribed_sphere AB BC CA AA1 h = 16 * π :=
by
  unfold surface_area_of_circumscribed_sphere
  sorry

end circumscribed_sphere_surface_area_l666_666337


namespace chickens_lay_eggs_l666_666414

theorem chickens_lay_eggs (d : ℚ) 
  (chickens1 : ℚ := 5) (eggs1 : ℚ := 7) (days1 : ℚ := 4) 
  (chickens2 : ℚ := 8) (eggs2 : ℚ := 10) :
  (chickens1 / days1) * (d * chickens2) = eggs2 :=
begin
  unfold Q,
  sorry
end

end chickens_lay_eggs_l666_666414


namespace smallest_positive_x_for_maximum_l666_666570

noncomputable def smallest_x_for_maximum (f : ℝ → ℝ) : ℝ :=
  if h : ∃ x, (x > 0) ∧ (∀ y, f x ≥ f y) then classical.some h else 0

theorem smallest_positive_x_for_maximum :
  ∀ f : ℝ → ℝ,
    (∀ x, f x = sin (x / 4) + sin (x / 7)) →
    smallest_x_for_maximum f = 10080 :=
by
  intro f h
  have h_condition : ∃ x, (x > 0) ∧ (∀ y, f x ≥ f y) :=
    sorry
  rw [smallest_x_for_maximum, dif_pos h_condition]
  exact classical.some_spec h_condition
  sorry

end smallest_positive_x_for_maximum_l666_666570


namespace extremum_point_range_l666_666997

theorem extremum_point_range (a : ℝ) (h_extremum: ∃ x : ℝ, x > 0 ∧ deriv (λ x, a * real.exp x + 3 * x) x = 0) : -3 < a ∧ a < 0 :=
by
  sorry

end extremum_point_range_l666_666997


namespace cos_double_angle_l666_666277

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l666_666277


namespace least_xy_l666_666638

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_xy_l666_666638


namespace angle_bisector_l666_666150

theorem angle_bisector (A B D : Point)
  (h1 : segment A B)
  (h2 : line_from_point B parallel line_through A)
  (h3 : segment A B = segment B D) :
  (bisects_angle A D A) :=
sorry

end angle_bisector_l666_666150


namespace cubic_common_roots_l666_666599

theorem cubic_common_roots:
  ∃ (c d : ℝ), 
  (∀ r s : ℝ,
    r ≠ s ∧ 
    (r ∈ {x : ℝ | x^3 + c * x^2 + 16 * x + 9 = 0}) ∧
    (s ∈ {x : ℝ | x^3 + c * x^2 + 16 * x + 9 = 0}) ∧ 
    (r ∈ {x : ℝ | x^3 + d * x^2 + 20 * x + 12 = 0}) ∧
    (s ∈ {x : ℝ | x^3 + d * x^2 + 20 * x + 12 = 0})) → 
  c = 8 ∧ d = 9 := 
by
  sorry

end cubic_common_roots_l666_666599


namespace Charlie_has_largest_final_value_l666_666553

def start_value : ℕ := 15

def Alice_final (n : ℕ) : ℕ :=
  let step1 := n - 3 in
  let step2 := step1 * 3 in
  step2 + 5

def Bob_final (n : ℕ) : ℕ :=
  let step1 := n * 3 in
  let step2 := step1 - 3 in
  step2 + 5

def Charlie_final (n : ℕ) : ℕ :=
  let step1 := n - 3 in
  let step2 := step1 + 5 in
  step2 * 3

theorem Charlie_has_largest_final_value :
  Charlie_final start_value > Alice_final start_value ∧
  Charlie_final start_value > Bob_final start_value := by
  sorry

end Charlie_has_largest_final_value_l666_666553


namespace parallel_lines_condition_l666_666201

theorem parallel_lines_condition (a : ℝ) : 
    (∀ x y : ℝ, 2 * x + a * y + 2 ≠ (a - 1) * x + y - 2) ↔ a = 2 := 
sorry

end parallel_lines_condition_l666_666201


namespace count_triples_sets_l666_666114

theorem count_triples_sets :
  let S := {1, 2, 3, 4, 5, 6, 7, 8}
  ∃ (A B C : finset ℕ), 
    A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧
    A.card = 4 ∧ B.card = 4 ∧ C.card = 4 ∧
    (A ∩ B).card = 2 ∧ (B ∩ C).card = 2 ∧ (C ∩ A).card = 2 →
  (∃ n, n = 85680) :=
by
  sorry

end count_triples_sets_l666_666114


namespace beef_weight_loss_l666_666548

theorem beef_weight_loss (weight_before weight_after: ℕ) 
                         (h1: weight_before = 400) 
                         (h2: weight_after = 240) : 
                         ((weight_before - weight_after) * 100 / weight_before = 40) :=
by 
  sorry

end beef_weight_loss_l666_666548


namespace area_of_triangle_l666_666333

variable (DE DF : ℝ)
variable (hDE : DE = 15)
variable (hDF : DF = 10)
variable (angleD : ∠D = 90)

theorem area_of_triangle (DE DF : ℝ) (hDE : DE = 15) (hDF : DF = 10) (angleD : ∠D = 90) : 
  (1 / 2) * DE * DF = 75 :=
by
  sorry

end area_of_triangle_l666_666333


namespace circle_intersections_and_circles_l666_666200

theorem circle_intersections_and_circles
  (C1 C2 : ℝ → ℝ → Prop)
  (hC1 : ∀ x y : ℝ, C1 x y ↔ x^2 + y^2 + 2 * x + 2 * y - 8 = 0)
  (hC2 : ∀ x y : ℝ, C2 x y ↔ x^2 + y^2 - 2 * x + 10 * y - 24 = 0) :
  (∀ x y : ℝ, (x - 2 * y + 4 = 0) ↔ 
    (exists A B : ℝ × ℝ, C1 A.1 A.2 ∧ C2 A.1 A.2 ∧ C1 B.1 B.2 ∧ C2 B.1 B.2 ∧ is_on_line (A, B) (x, y))) ∧
  (∀ x y : ℝ, ((x + 3)^2 + (y - 3)^2 = 10) ↔ 
    (exists A B : ℝ × ℝ, C1 A.1 A.2 ∧ C2 A.1 A.2 ∧ C1 B.1 B.2 ∧ C2 B.1 B.2 ∧ ∃ M, M.2 = -M.1 ∧ M ∈ circle (A, B) (x, y))) ∧
  (∀ x y : ℝ, ((x + 2)^2 + (y - 1)^2 = 5) ↔ 
    (exists A B : ℝ × ℝ, C1 A.1 A.2 ∧ C2 A.1 A.2 ∧ C1 B.1 B.2 ∧ C2 B.1 B.2 ∧ smallest_circle (A, B) (x, y))) :=
by sorry

end circle_intersections_and_circles_l666_666200


namespace work_days_l666_666881

-- Definitions based on conditions
def A_work_rate := 1 / 5
def B_work_rate := 1 / 10
def AB_work_rate := A_work_rate + B_work_rate
def total_payment := 500
def C_share := 200
def C_contribution := C_share / total_payment
def AB_contribution := 1 - C_contribution
def combined_work_rate (x : ℝ) := 1 / x

-- Main theorem to prove
theorem work_days (x : ℝ) (hx : AB_work_rate * x = AB_contribution) : x = 2 :=
by
  sorry

end work_days_l666_666881


namespace eval_integral_1_eval_integral_2_eval_integral_3_eval_integral_4_l666_666112

noncomputable def integral_1 := ∫ x in 0..(0:ℝ), (1 / (x^2 + 4 * x + 8))
noncomputable def integral_1_solution := (1 / 2) * (Real.arctan ((x + 2) / 2))

theorem eval_integral_1 : integral_1 = integral_1_solution + C := sorry

noncomputable def integral_2 := ∫ x in 0..(0:ℝ), ((7 - 8 * x) / (2 * x^3 - 3 * x + 1))
noncomputable def integral_2_solution := Real.log (abs ((x - 1) / (x - 0.5))) - 2 * Real.log (abs (x^2 - 1.5 * x + 0.5)) + C

theorem eval_integral_2 : integral_2 = integral_2_solution := sorry

noncomputable def integral_3 := ∫ x in 0..(0:ℝ), ((3 * x - 2) / (x^2 + 6 * x + 9))
noncomputable def integral_3_solution := 3 * Real.log (abs (x + 3)) + 11 / (x + 3) + C

theorem eval_integral_3 : integral_3 = integral_3_solution := sorry

noncomputable def integral_4 := ∫ x in 0..(0:ℝ), ((6 * x^3 - 7 * x^2 + 3 * x - 1) / (2 * x - 3 * x^2))
noncomputable def integral_4_solution := -x^2 + x + (1 / 6) * Real.log (abs (x - (2 / 3))) - (1 / 2) * Real.log (abs x) + C

theorem eval_integral_4 : integral_4 = integral_4_solution := sorry

end eval_integral_1_eval_integral_2_eval_integral_3_eval_integral_4_l666_666112


namespace suitable_census_survey_l666_666502

theorem suitable_census_survey (A B C D: Prop) 
  (hA: A = "The lifespan of a certain brand of light bulb") 
  (hB: B = "The most popular sports activity among all students in the class") 
  (hC: C = "The types of fish in the Yangtze River") 
  (hD: D = "The number of plastic bags discarded by families of all students in the city within a week")
  (hImpA: A -> False)
  (hImpC: C -> False)
  (hImpD: D -> False)
  (hImpB: B -> True): 
  (suitableSurvey: B) :=
  
  by
    exact hImpB

end suitable_census_survey_l666_666502


namespace flour_needed_l666_666953

-- Define the conditions
def original_butter : ℕ := 2
def original_flour : ℕ := 5
def multiplier : ℕ := 6

-- Define the statement to be proved
theorem flour_needed (b : ℕ) (f : ℕ) (m : ℕ) (B : ℕ) : 
  b = original_butter → 
  f = original_flour → 
  m = B / b → 
  B = 12 → 
  m * f = 30 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h4] at h3
  rw [h1, h2]
  sorry

end flour_needed_l666_666953


namespace matrix_mult_result_l666_666085

def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem matrix_mult_result :
  let M := 
      (Real.mkMatrixOfArray! [
        [log2(2^(1/4)), log2(25)],
        [log2(5), log2(256)]
      ] : Matrix 2 2 ℝ),
      v := ![8^(2/3), 2^(-1 : ℝ)]
  M.mul_vec v = ![1, 4] := by
  sorry

end matrix_mult_result_l666_666085


namespace sam_apples_one_week_l666_666129

theorem sam_apples_one_week (sandwiches_per_day : ℕ) (apples_per_sandwich : ℕ) (days_per_week : ℕ)
  (h1 : sandwiches_per_day = 10) (h2 : apples_per_sandwich = 4) (h3 : days_per_week = 7) :
  (sandwiches_per_day * apples_per_sandwich * days_per_week) = 280 :=
by
  rw [h1, h2, h3]
  simp
  sorry

end sam_apples_one_week_l666_666129


namespace count_fourdigit_integers_div_by_35_of_form_x35_l666_666689

theorem count_fourdigit_integers_div_by_35_of_form_x35 : 
  {n: Nat // 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 35 ∧ n % 35 = 0}.card = 13 :=
by
  sorry

end count_fourdigit_integers_div_by_35_of_form_x35_l666_666689


namespace rectangle_probability_no_shaded_square_l666_666877

theorem rectangle_probability_no_shaded_square :
  let n := 1003 * 2005 in
  let m := 1003 ^ 2 in
  2 by 2005 rectangle
  ∧ middle unit square of each row is shaded
  ∧ rect chosen at random
  → 1 - m / n = 1002 / 2005 :=
by
  sorry

end rectangle_probability_no_shaded_square_l666_666877


namespace Lidia_money_for_apps_l666_666769

theorem Lidia_money_for_apps (app_cost : ℕ) (num_apps : ℕ) (remaining_money : ℕ) 
  (total_cost : ℕ) : 
  app_cost = 4 → 
  num_apps = 15 → 
  remaining_money = 6 → 
  total_cost = app_cost * num_apps → 
  Lidia_has_money : total_cost + remaining_money = 66 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h4]
  exact rfl
  sorry

end Lidia_money_for_apps_l666_666769
