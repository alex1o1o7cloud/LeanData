import Mathlib

namespace number_of_cities_l1300_130019

theorem number_of_cities (n : ℕ) (h : n * (n - 1) / 2 = 15) : n = 6 :=
sorry

end number_of_cities_l1300_130019


namespace equilateral_triangle_side_length_l1300_130092

theorem equilateral_triangle_side_length 
  (x1 y1 : ℝ) 
  (hx1y1 : y1 = - (1 / 4) * x1^2)
  (h_eq_tri: ∃ (x2 y2 : ℝ), x2 = -x1 ∧ y2 = y1 ∧ (x2, y2) ≠ (x1, y1) ∧ ((x1 - x2)^2 + (y1 - y2)^2 = x1^2 + y1^2 ∧ (x1 - 0)^2 + (y1 - 0)^2 = (x1 - x2)^2 + (y1 - y2)^2)):
  2 * x1 = 8 * Real.sqrt 3 := 
sorry

end equilateral_triangle_side_length_l1300_130092


namespace dice_sum_not_possible_l1300_130002

   theorem dice_sum_not_possible (a b c d : ℕ) :
     (1 ≤ a ∧ a ≤ 6) → (1 ≤ b ∧ b ≤ 6) → (1 ≤ c ∧ c ≤ 6) → (1 ≤ d ∧ d ≤ 6) →
     (a * b * c * d = 360) → ¬ (a + b + c + d = 20) :=
   by
     intros ha hb hc hd prod eq_sum
     -- Proof skipped
     sorry
   
end dice_sum_not_possible_l1300_130002


namespace tan_theta_eq_neg_sqrt_3_l1300_130063

theorem tan_theta_eq_neg_sqrt_3 (theta : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h_a : a = (Real.cos theta, Real.sin theta))
  (h_b : b = (Real.sqrt 3, 1))
  (h_perpendicular : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.tan theta = -Real.sqrt 3 :=
sorry

end tan_theta_eq_neg_sqrt_3_l1300_130063


namespace recipe_flour_requirement_l1300_130083

def sugar_cups : ℕ := 9
def salt_cups : ℕ := 40
def flour_initial_cups : ℕ := 4
def additional_flour : ℕ := sugar_cups + 1
def total_flour_cups : ℕ := additional_flour

theorem recipe_flour_requirement : total_flour_cups = 10 := by
  sorry

end recipe_flour_requirement_l1300_130083


namespace minimum_value_S_l1300_130053

noncomputable def S (x a : ℝ) : ℝ := (x - a)^2 + (Real.log x - a)^2

theorem minimum_value_S : ∃ x a : ℝ, x > 0 ∧ (S x a = 1 / 2) := by
  sorry

end minimum_value_S_l1300_130053


namespace dan_balloons_l1300_130005

theorem dan_balloons (fred_balloons sam_balloons total_balloons dan_balloons : ℕ) 
  (h₁ : fred_balloons = 10) 
  (h₂ : sam_balloons = 46) 
  (h₃ : total_balloons = 72) : 
  dan_balloons = total_balloons - (fred_balloons + sam_balloons) :=
by
  sorry

end dan_balloons_l1300_130005


namespace probability_of_all_female_l1300_130094

noncomputable def probability_all_females_final (females males total chosen : ℕ) : ℚ :=
  (females.choose chosen) / (total.choose chosen)

theorem probability_of_all_female:
  probability_all_females_final 5 3 8 3 = 5 / 28 :=
by
  sorry

end probability_of_all_female_l1300_130094


namespace fg_eval_l1300_130087

def f (x : ℤ) : ℤ := x^3
def g (x : ℤ) : ℤ := 4 * x + 5

theorem fg_eval : f (g (-2)) = -27 := by
  sorry

end fg_eval_l1300_130087


namespace max_sector_area_central_angle_l1300_130008

theorem max_sector_area_central_angle (radius arc_length : ℝ) :
  (arc_length + 2 * radius = 20) ∧ (arc_length = 20 - 2 * radius) ∧
  (arc_length / radius = 2) → 
  arc_length / radius = 2 :=
by
  intros h 
  sorry

end max_sector_area_central_angle_l1300_130008


namespace scientific_notation_l1300_130058

def billion : ℝ := 10^9
def fifteenPointSeventyFiveBillion : ℝ := 15.75 * billion

theorem scientific_notation :
  fifteenPointSeventyFiveBillion = 1.575 * 10^10 :=
  sorry

end scientific_notation_l1300_130058


namespace sufficient_but_not_necessary_condition_l1300_130095

open Real

theorem sufficient_but_not_necessary_condition :
  ∀ (m : ℝ),
  (∀ x, (x^2 - 3*x - 4 ≤ 0) → (x^2 - 6*x + 9 - m^2 ≤ 0)) ∧
  (∃ x, ¬(x^2 - 3*x - 4 ≤ 0) ∧ (x^2 - 6*x + 9 - m^2 ≤ 0)) ↔
  m ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
by
  sorry

end sufficient_but_not_necessary_condition_l1300_130095


namespace difference_of_squares_l1300_130079

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := 
by
  sorry

end difference_of_squares_l1300_130079


namespace width_of_jordan_rectangle_l1300_130072

def carol_length := 5
def carol_width := 24
def jordan_length := 2
def jordan_area := carol_length * carol_width

theorem width_of_jordan_rectangle : ∃ (w : ℝ), jordan_length * w = jordan_area ∧ w = 60 :=
by
  use 60
  simp [carol_length, carol_width, jordan_length, jordan_area]
  sorry

end width_of_jordan_rectangle_l1300_130072


namespace cube_volume_l1300_130048

theorem cube_volume {V : ℝ} (x : ℝ) (hV : V = x^3) (hA : 2 * V = 6 * x^2) : V = 27 :=
by
  -- Proof goes here
  sorry

end cube_volume_l1300_130048


namespace determine_m_l1300_130014

-- Define the fractional equation condition
def fractional_eq (m x : ℝ) : Prop := (m/(x - 2) + 2*x/(x - 2) = 1)

-- Define the main theorem statement
theorem determine_m (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ x ≠ 2 ∧ fractional_eq m x) : m = -4 :=
sorry

end determine_m_l1300_130014


namespace total_amount_proof_l1300_130012

def total_shared_amount : ℝ :=
  let z := 250
  let y := 1.20 * z
  let x := 1.25 * y
  x + y + z

theorem total_amount_proof : total_shared_amount = 925 :=
by
  sorry

end total_amount_proof_l1300_130012


namespace bananas_to_oranges_l1300_130067

variables (banana apple orange : Type) 
variables (cost_banana : banana → ℕ) 
variables (cost_apple : apple → ℕ)
variables (cost_orange : orange → ℕ)

-- Conditions given in the problem
axiom cond1 : ∀ (b1 b2 b3 : banana) (a1 a2 : apple), cost_banana b1 = cost_banana b2 → cost_banana b2 = cost_banana b3 → 3 * cost_banana b1 = 2 * cost_apple a1
axiom cond2 : ∀ (a3 a4 a5 a6 : apple) (o1 o2 : orange), cost_apple a3 = cost_apple a4 → cost_apple a4 = cost_apple a5 → cost_apple a5 = cost_apple a6 → 6 * cost_apple a3 = 4 * cost_orange o1

-- Prove that 8 oranges cost as much as 18 bananas
theorem bananas_to_oranges (b1 b2 b3 : banana) (a1 a2 a3 a4 a5 a6 : apple) (o1 o2 : orange) :
    3 * cost_banana b1 = 2 * cost_apple a1 →
    6 * cost_apple a3 = 4 * cost_orange o1 →
    18 * cost_banana b1 = 8 * cost_orange o2 := 
sorry

end bananas_to_oranges_l1300_130067


namespace negation_of_all_cars_are_fast_l1300_130032

variable {α : Type} -- Assume α is the type of entities
variable (car fast : α → Prop) -- car and fast are predicates on entities

theorem negation_of_all_cars_are_fast :
  ¬ (∀ x, car x → fast x) ↔ ∃ x, car x ∧ ¬ fast x :=
by sorry

end negation_of_all_cars_are_fast_l1300_130032


namespace find_number_l1300_130073

theorem find_number (N : ℝ) (h : 0.1 * 0.3 * 0.5 * N = 90) : N = 6000 :=
by
  sorry

end find_number_l1300_130073


namespace operation_B_is_correct_l1300_130025

theorem operation_B_is_correct (a b x : ℝ) : 
  2 * (a^2) * b * 4 * a * (b^3) = 8 * (a^3) * (b^4) :=
by
  sorry

-- Conditions for incorrect operations
lemma operation_A_is_incorrect (x : ℝ) : 
  x^8 / x^2 ≠ x^4 :=
by
  sorry

lemma operation_C_is_incorrect (x : ℝ) : 
  (-x^5)^4 ≠ -x^20 :=
by
  sorry

lemma operation_D_is_incorrect (a b : ℝ) : 
  (a + b)^2 ≠ a^2 + b^2 :=
by
  sorry

end operation_B_is_correct_l1300_130025


namespace greatest_divisor_of_546_smaller_than_30_and_factor_of_126_l1300_130069

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem greatest_divisor_of_546_smaller_than_30_and_factor_of_126 :
  ∃ (d : ℕ), d < 30 ∧ is_factor d 546 ∧ is_factor d 126 ∧ ∀ e : ℕ, e < 30 ∧ is_factor e 546 ∧ is_factor e 126 → e ≤ d := 
sorry

end greatest_divisor_of_546_smaller_than_30_and_factor_of_126_l1300_130069


namespace last_digit_to_appear_mod9_l1300_130047

def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

def fib_mod9 (n : ℕ) : ℕ :=
  (fib n) % 9

theorem last_digit_to_appear_mod9 :
  ∃ n : ℕ, ∀ m : ℕ, m < n → fib_mod9 m ≠ 0 ∧ fib_mod9 n = 0 :=
sorry

end last_digit_to_appear_mod9_l1300_130047


namespace medal_allocation_l1300_130052

-- Define the participants
inductive Participant
| Jiri
| Vit
| Ota

open Participant

-- Define the medals
inductive Medal
| Gold
| Silver
| Bronze

open Medal

-- Define a structure to capture each person's statement
structure Statements :=
  (Jiri : Prop)
  (Vit : Prop)
  (Ota : Prop)

-- Define the condition based on their statements
def statements (m : Participant → Medal) : Statements :=
  {
    Jiri := m Ota = Gold,
    Vit := m Ota = Silver,
    Ota := (m Ota ≠ Gold ∧ m Ota ≠ Silver)
  }

-- Define the condition for truth-telling and lying based on medals
def truths_and_lies (m : Participant → Medal) (s : Statements) : Prop :=
  (m Jiri = Gold → s.Jiri) ∧ (m Jiri = Bronze → ¬ s.Jiri) ∧
  (m Vit = Gold → s.Vit) ∧ (m Vit = Bronze → ¬ s.Vit) ∧
  (m Ota = Gold → s.Ota) ∧ (m Ota = Bronze → ¬ s.Ota)

-- Define the final theorem to be proven
theorem medal_allocation : 
  ∃ (m : Participant → Medal), 
    truths_and_lies m (statements m) ∧ 
    m Vit = Gold ∧ 
    m Ota = Silver ∧ 
    m Jiri = Bronze := 
sorry

end medal_allocation_l1300_130052


namespace sin_eq_cos_510_l1300_130003

theorem sin_eq_cos_510 (n : ℤ) (h1 : -180 ≤ n ∧ n ≤ 180) (h2 : Real.sin (n * Real.pi / 180) = Real.cos (510 * Real.pi / 180)) :
  n = -60 :=
sorry

end sin_eq_cos_510_l1300_130003


namespace simplify_and_evaluate_expr_evaluate_at_zero_l1300_130088

theorem simplify_and_evaluate_expr (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1)) = (2 + x) / (2 - x) :=
by
  sorry

theorem evaluate_at_zero :
  (2 + 0 : ℝ) / (2 - 0) = 1 :=
by
  norm_num

end simplify_and_evaluate_expr_evaluate_at_zero_l1300_130088


namespace triangle_PR_eq_8_l1300_130045

open Real

theorem triangle_PR_eq_8 (P Q R M : ℝ) 
  (PQ QR PM : ℝ) 
  (hPQ : PQ = 6) (hQR : QR = 10) (hPM : PM = 5) 
  (M_midpoint : M = (Q + R) / 2) :
  dist P R = 8 :=
by
  sorry

end triangle_PR_eq_8_l1300_130045


namespace remainder_of_3_pow_102_mod_101_l1300_130044

theorem remainder_of_3_pow_102_mod_101 : (3^102) % 101 = 9 :=
by
  sorry

end remainder_of_3_pow_102_mod_101_l1300_130044


namespace simplify_expression_l1300_130039

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : (2 * x⁻¹ + 3 * y⁻¹)⁻¹ = (x * y) / (2 * y + 3 * x) :=
by sorry

end simplify_expression_l1300_130039


namespace total_seeds_planted_l1300_130022

def number_of_flowerbeds : ℕ := 9
def seeds_per_flowerbed : ℕ := 5

theorem total_seeds_planted : number_of_flowerbeds * seeds_per_flowerbed = 45 :=
by
  sorry

end total_seeds_planted_l1300_130022


namespace b_n_plus_1_eq_2a_n_l1300_130038

/-- Definition of binary sequences of length n that do not contain 0, 1, 0 -/
def a_n (n : ℕ) : ℕ := -- specify the actual counting function, placeholder below
  sorry

/-- Definition of binary sequences of length n that do not contain 0, 0, 1, 1 or 1, 1, 0, 0 -/
def b_n (n : ℕ) : ℕ := -- specify the actual counting function, placeholder below
  sorry

/-- Proof statement that for all positive integers n, b_{n+1} = 2a_n -/
theorem b_n_plus_1_eq_2a_n (n : ℕ) (hn : 0 < n) : b_n (n + 1) = 2 * a_n n :=
  sorry

end b_n_plus_1_eq_2a_n_l1300_130038


namespace number_is_three_l1300_130015

theorem number_is_three (n : ℝ) (h : 4 * n - 7 = 5) : n = 3 :=
by sorry

end number_is_three_l1300_130015


namespace store_profit_l1300_130020

theorem store_profit (m n : ℝ) (hmn : m > n) : 
  let selling_price := (m + n) / 2
  let profit_a := 40 * (selling_price - m)
  let profit_b := 60 * (selling_price - n)
  let total_profit := profit_a + profit_b
  total_profit > 0 :=
by sorry

end store_profit_l1300_130020


namespace find_solutions_l1300_130046

theorem find_solutions (n k : ℕ) (hn : n > 0) (hk : k > 0) : 
  n! + n = n^k → (n, k) = (2, 2) ∨ (n, k) = (3, 2) ∨ (n, k) = (5, 3) :=
sorry

end find_solutions_l1300_130046


namespace hot_dogs_remainder_l1300_130050

theorem hot_dogs_remainder :
  25197625 % 4 = 1 :=
by
  sorry

end hot_dogs_remainder_l1300_130050


namespace max_min_value_l1300_130001

theorem max_min_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 12) (h5 : x * y + y * z + z * x = 30) :
  ∃ n : ℝ, n = min (x * y) (min (y * z) (z * x)) ∧ n = 2 :=
sorry

end max_min_value_l1300_130001


namespace range_of_m_l1300_130071

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + |x - 1| ≥ (m + 2) * x - 1) ↔ (-3 - 2 * Real.sqrt 2) ≤ m ∧ m ≤ 0 := 
sorry

end range_of_m_l1300_130071


namespace triangle_third_side_possibilities_l1300_130062

theorem triangle_third_side_possibilities (x : ℕ) : 
  (6 + 8 > x) ∧ (x + 6 > 8) ∧ (x + 8 > 6) → 
  3 ≤ x ∧ x < 14 → 
  ∃ n, n = 11 :=
by
  sorry

end triangle_third_side_possibilities_l1300_130062


namespace james_total_fish_catch_l1300_130043

-- Definitions based on conditions
def weight_trout : ℕ := 200
def weight_salmon : ℕ := weight_trout + (60 * weight_trout / 100)
def weight_tuna : ℕ := 2 * weight_trout
def weight_bass : ℕ := 3 * weight_salmon
def weight_catfish : ℚ := weight_tuna / 3

-- Total weight of the fish James caught
def total_weight_fish : ℚ := 
  weight_trout + weight_salmon + weight_tuna + weight_bass + weight_catfish 

-- The theorem statement
theorem james_total_fish_catch : total_weight_fish = 2013.33 := by
  sorry

end james_total_fish_catch_l1300_130043


namespace total_combinations_l1300_130055

def varieties_of_wrapping_paper : Nat := 10
def colors_of_ribbon : Nat := 4
def types_of_gift_cards : Nat := 5
def kinds_of_decorative_stickers : Nat := 2

theorem total_combinations : varieties_of_wrapping_paper * colors_of_ribbon * types_of_gift_cards * kinds_of_decorative_stickers = 400 := by
  sorry

end total_combinations_l1300_130055


namespace smallest_n_leq_l1300_130081

theorem smallest_n_leq (n : ℤ) : (n ^ 2 - 13 * n + 40 ≤ 0) → (n = 5) :=
sorry

end smallest_n_leq_l1300_130081


namespace factorization_correct_l1300_130065

noncomputable def original_poly (x : ℝ) : ℝ := 12 * x ^ 2 + 18 * x - 24
noncomputable def factored_poly (x : ℝ) : ℝ := 6 * (2 * x - 1) * (x + 4)

theorem factorization_correct (x : ℝ) : original_poly x = factored_poly x :=
by
  sorry

end factorization_correct_l1300_130065


namespace angle_coloring_min_colors_l1300_130076

  theorem angle_coloring_min_colors (n : ℕ) : 
    (∃ c : ℕ, (c = 2 ↔ n % 2 = 0) ∧ (c = 3 ↔ n % 2 = 1)) :=
  by
    sorry
  
end angle_coloring_min_colors_l1300_130076


namespace remaining_lemons_proof_l1300_130018

-- Definitions for initial conditions
def initial_lemons_first_tree   := 15
def initial_lemons_second_tree  := 20
def initial_lemons_third_tree   := 25

def sally_picked_first_tree     := 7
def mary_picked_second_tree     := 9
def tom_picked_first_tree       := 12

def lemons_fell_each_tree       := 4
def animals_eaten_per_tree      := lemons_fell_each_tree / 2

-- Definitions for intermediate calculations
def remaining_lemons_first_tree_full := initial_lemons_first_tree - sally_picked_first_tree - tom_picked_first_tree
def remaining_lemons_first_tree      := if remaining_lemons_first_tree_full < 0 then 0 else remaining_lemons_first_tree_full

def remaining_lemons_second_tree := initial_lemons_second_tree - mary_picked_second_tree

def mary_picked_third_tree := (remaining_lemons_second_tree : ℚ) / 2
def remaining_lemons_third_tree_full := (initial_lemons_third_tree : ℚ) - mary_picked_third_tree
def remaining_lemons_third_tree      := Nat.floor remaining_lemons_third_tree_full

-- Adjusting for fallen and eaten lemons
def final_remaining_lemons_first_tree_full := remaining_lemons_first_tree - lemons_fell_each_tree + animals_eaten_per_tree
def final_remaining_lemons_first_tree      := if final_remaining_lemons_first_tree_full < 0 then 0 else final_remaining_lemons_first_tree_full

def final_remaining_lemons_second_tree     := remaining_lemons_second_tree - lemons_fell_each_tree + animals_eaten_per_tree

def final_remaining_lemons_third_tree_full := remaining_lemons_third_tree - lemons_fell_each_tree + animals_eaten_per_tree
def final_remaining_lemons_third_tree      := if final_remaining_lemons_third_tree_full < 0 then 0 else final_remaining_lemons_third_tree_full

-- Lean 4 statement to prove the equivalence
theorem remaining_lemons_proof :
  final_remaining_lemons_first_tree = 0 ∧
  final_remaining_lemons_second_tree = 9 ∧
  final_remaining_lemons_third_tree = 18 :=
by
  -- The proof is omitted as per the requirement
  sorry

end remaining_lemons_proof_l1300_130018


namespace product_divisible_by_third_l1300_130061

theorem product_divisible_by_third (a b c : Int)
    (h1 : (a + b + c)^2 = -(a * b + a * c + b * c))
    (h2 : a + b ≠ 0) (h3 : b + c ≠ 0) (h4 : a + c ≠ 0) :
    ((a + b) * (a + c) % (b + c) = 0) ∧ ((a + b) * (b + c) % (a + c) = 0) ∧ ((a + c) * (b + c) % (a + b) = 0) :=
  sorry

end product_divisible_by_third_l1300_130061


namespace geometric_sequence_sum_l1300_130091

theorem geometric_sequence_sum (a_1 q n S : ℕ) (h1 : a_1 = 2) (h2 : q = 2) (h3 : S = 126) 
    (h4 : S = (a_1 * (1 - q^n)) / (1 - q)) : 
    n = 6 :=
by
  sorry

end geometric_sequence_sum_l1300_130091


namespace calc_fraction_product_l1300_130037

theorem calc_fraction_product : 
  (7 / 4) * (8 / 14) * (14 / 8) * (16 / 40) * (35 / 20) * (18 / 45) * (49 / 28) * (32 / 64) = 49 / 200 := 
by sorry

end calc_fraction_product_l1300_130037


namespace find_b12_l1300_130016

noncomputable def seq (b : ℕ → ℤ) : Prop :=
  b 1 = 2 ∧ 
  ∀ m n : ℕ, m > 0 → n > 0 → b (m + n) = b m + b n + (m * n * n)

theorem find_b12 (b : ℕ → ℤ) (h : seq b) : b 12 = 98 := 
by
  sorry

end find_b12_l1300_130016


namespace married_fraction_l1300_130027

variables (M W N : ℕ)

def married_men : Prop := 2 * M = 3 * N
def married_women : Prop := 3 * W = 5 * N
def total_population : ℕ := M + W
def married_population : ℕ := 2 * N

theorem married_fraction (h1: married_men M N) (h2: married_women W N) :
  (married_population N : ℚ) / (total_population M W : ℚ) = 12 / 19 :=
by sorry

end married_fraction_l1300_130027


namespace trees_planted_l1300_130017

theorem trees_planted (yard_length : ℕ) (distance_between_trees : ℕ) (n_trees : ℕ) 
  (h1 : yard_length = 434) 
  (h2 : distance_between_trees = 14) 
  (h3 : n_trees = yard_length / distance_between_trees + 1) : 
  n_trees = 32 :=
by
  sorry

end trees_planted_l1300_130017


namespace number_of_correct_statements_l1300_130080

-- Definitions of the conditions from the problem
def seq_is_graphical_points := true  -- Statement 1
def seq_is_finite (s : ℕ → ℝ) := ∀ n, s n = 0 -- Statement 2
def seq_decreasing_implies_finite (s : ℕ → ℝ) := (∀ n, s (n + 1) ≤ s n) → seq_is_finite s -- Statement 3

-- Prove the number of correct statements is 1
theorem number_of_correct_statements : (seq_is_graphical_points = true ∧ ¬(∃ s: ℕ → ℝ, ¬seq_is_finite s) ∧ ∃ s : ℕ → ℝ, ¬seq_decreasing_implies_finite s) → 1 = 1 :=
by
  sorry

end number_of_correct_statements_l1300_130080


namespace total_squares_in_6x6_grid_l1300_130033

theorem total_squares_in_6x6_grid : 
  let size := 6
  let total_squares := (size * size) + ((size - 1) * (size - 1)) + ((size - 2) * (size - 2)) + ((size - 3) * (size - 3)) + ((size - 4) * (size - 4)) + ((size - 5) * (size - 5))
  total_squares = 91 :=
by
  let size := 6
  let total_squares := (size * size) + ((size - 1) * (size - 1)) + ((size - 2) * (size - 2)) + ((size - 3) * (size - 3)) + ((size - 4) * (size - 4)) + ((size - 5) * (size - 5))
  have eqn : total_squares = 91 := sorry
  exact eqn

end total_squares_in_6x6_grid_l1300_130033


namespace num_correct_conclusions_l1300_130029

-- Definitions and conditions from the problem
variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}
variable (n : ℕ)
variable (hSn_eq : S n + S (n + 1) = n ^ 2)

-- Assert the conditions described in the comments
theorem num_correct_conclusions (hSn_eq : ∀ n, S n + S (n + 1) = n ^ 2) :
  (1:ℕ) = 3 ↔
  (-- Conclusion 1
   ¬(∀ n, a (n + 2) - a n = 2) ∧
   -- Conclusion 2: If a_1 = 0, then S_50 = 1225
   (S 50 = 1225) ∧
   -- Conclusion 3: If a_1 = 1, then S_50 = 1224
   (S 50 = 1224) ∧
   -- Conclusion 4: Monotonically increasing sequence
   (∀ a_1, (-1/4 : ℚ) < a_1 ∧ a_1 < 1/4)) :=
by
  sorry

end num_correct_conclusions_l1300_130029


namespace expr_simplify_l1300_130060

variable {a b c d m : ℚ}
variable {b_nonzero : b ≠ 0}
variable {m_nat : ℕ}
variable {m_bound : 0 ≤ m_nat ∧ m_nat < 2}

def expr_value (a b c d m : ℚ) : ℚ :=
  m - (c * d) + (a + b) / 2023 + a / b

theorem expr_simplify (h1 : a = -b) (h2 : c * d = 1) (h3 : m = (m_nat : ℚ)) :
  expr_value a b c d m = -1 ∨ expr_value a b c d m = -2 := by
  sorry

end expr_simplify_l1300_130060


namespace ed_total_pets_l1300_130064

theorem ed_total_pets (num_dogs num_cats : ℕ) (h_dogs : num_dogs = 2) (h_cats : num_cats = 3) :
  ∃ num_fish : ℕ, (num_fish = 2 * (num_dogs + num_cats)) ∧ (num_dogs + num_cats + num_fish) = 15 :=
by
  sorry

end ed_total_pets_l1300_130064


namespace bananas_oranges_equivalence_l1300_130074

theorem bananas_oranges_equivalence :
  (3 / 4) * 12 * banana_value = 9 * orange_value →
  (2 / 3) * 6 * banana_value = 4 * orange_value :=
by
  intros h
  sorry

end bananas_oranges_equivalence_l1300_130074


namespace coloring_problem_l1300_130098

def condition (m n : ℕ) : Prop :=
  2 ≤ m ∧ m ≤ 31 ∧ 2 ≤ n ∧ n ≤ 31 ∧ m ≠ n ∧ m % n = 0

def color (f : ℕ → ℕ) : Prop :=
  ∀ m n, condition m n → f m ≠ f n

theorem coloring_problem :
  ∃ (k : ℕ) (f : ℕ → ℕ), (∀ n, 2 ≤ n ∧ n ≤ 31 → f n ≤ k) ∧ color f ∧ k = 4 :=
by
  sorry

end coloring_problem_l1300_130098


namespace alice_unanswered_questions_l1300_130006

-- Declare variables for the proof
variables (c w u : ℕ)

-- State the problem in Lean
theorem alice_unanswered_questions :
  50 + 5 * c - 2 * w = 100 ∧
  40 + 7 * c - w - u = 120 ∧
  6 * c + 3 * u = 130 ∧
  c + w + u = 25 →
  u = 20 :=
by
  intros h
  sorry

end alice_unanswered_questions_l1300_130006


namespace timeTakenByBobIs30_l1300_130035

-- Define the conditions
def timeTakenByAlice : ℕ := 40
def fractionOfTimeBobTakes : ℚ := 3 / 4

-- Define the statement to be proven
theorem timeTakenByBobIs30 : (fractionOfTimeBobTakes * timeTakenByAlice : ℚ) = 30 := 
by
  sorry

end timeTakenByBobIs30_l1300_130035


namespace avg_of_multiples_of_10_eq_305_l1300_130070

theorem avg_of_multiples_of_10_eq_305 (N : ℕ) (h : N % 10 = 0) (h_avg : (10 + N) / 2 = 305) : N = 600 :=
sorry

end avg_of_multiples_of_10_eq_305_l1300_130070


namespace value_of_A_l1300_130011

def clubsuit (A B : ℕ) := 3 * A + 2 * B + 5

theorem value_of_A (A : ℕ) (h : clubsuit A 7 = 82) : A = 21 :=
by
  sorry

end value_of_A_l1300_130011


namespace sum_of_first_39_natural_numbers_l1300_130078

theorem sum_of_first_39_natural_numbers : (39 * (39 + 1)) / 2 = 780 :=
by
  sorry

end sum_of_first_39_natural_numbers_l1300_130078


namespace percentage_of_rotten_oranges_l1300_130030

theorem percentage_of_rotten_oranges
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (percentage_good_condition : ℕ)
  (rotted_percentage_bananas : ℕ)
  (total_fruits : ℕ)
  (good_condition_fruits : ℕ)
  (rotted_fruits : ℕ)
  (rotted_bananas : ℕ)
  (rotted_oranges : ℕ)
  (percentage_rotten_oranges : ℕ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : percentage_good_condition = 89)
  (h4 : rotted_percentage_bananas = 5)
  (h5 : total_fruits = total_oranges + total_bananas)
  (h6 : good_condition_fruits = percentage_good_condition * total_fruits / 100)
  (h7 : rotted_fruits = total_fruits - good_condition_fruits)
  (h8 : rotted_bananas = rotted_percentage_bananas * total_bananas / 100)
  (h9 : rotted_oranges = rotted_fruits - rotted_bananas)
  (h10 : percentage_rotten_oranges = rotted_oranges * 100 / total_oranges) : 
  percentage_rotten_oranges = 15 := 
by
  sorry

end percentage_of_rotten_oranges_l1300_130030


namespace area_triangle_ABC_correct_l1300_130089

noncomputable def rectangle_area : ℝ := 42

noncomputable def area_triangle_outside_I : ℝ := 9
noncomputable def area_triangle_outside_II : ℝ := 3.5
noncomputable def area_triangle_outside_III : ℝ := 12

noncomputable def area_triangle_ABC : ℝ :=
  rectangle_area - (area_triangle_outside_I + area_triangle_outside_II + area_triangle_outside_III)

theorem area_triangle_ABC_correct : area_triangle_ABC = 17.5 := by 
  sorry

end area_triangle_ABC_correct_l1300_130089


namespace product_abc_l1300_130054

theorem product_abc 
  (a b c : ℝ)
  (h1 : a + b + c = 1) 
  (h2 : 3 * (4 * a + 2 * b + c) = 15) 
  (h3 : 5 * (9 * a + 3 * b + c) = 65) :
  a * b * c = -4 :=
by
  sorry

end product_abc_l1300_130054


namespace gcd_equivalence_l1300_130056

theorem gcd_equivalence : 
  let m := 2^2100 - 1
  let n := 2^2091 + 31
  gcd m n = gcd (2^2091 + 31) 511 :=
by
  sorry

end gcd_equivalence_l1300_130056


namespace sum_of_three_consecutive_odds_is_69_l1300_130085

-- Definition for the smallest of three consecutive odd numbers
def smallest_consecutive_odd := 21

-- Define the three consecutive odd numbers based on the smallest one
def first_consecutive_odd := smallest_consecutive_odd
def second_consecutive_odd := smallest_consecutive_odd + 2
def third_consecutive_odd := smallest_consecutive_odd + 4

-- Calculate the sum of these three consecutive odd numbers
def sum_consecutive_odds := first_consecutive_odd + second_consecutive_odd + third_consecutive_odd

-- Theorem statement that the sum of these three consecutive odd numbers is 69
theorem sum_of_three_consecutive_odds_is_69 : 
  sum_consecutive_odds = 69 := by
    sorry

end sum_of_three_consecutive_odds_is_69_l1300_130085


namespace isosceles_triangle_relationship_l1300_130077

theorem isosceles_triangle_relationship (x y : ℝ) (h1 : 2 * x + y = 30) (h2 : 7.5 < x) (h3 : x < 15) : 
  y = 30 - 2 * x :=
  by sorry

end isosceles_triangle_relationship_l1300_130077


namespace train_length_l1300_130000

-- Define the given speeds and time
def train_speed_km_per_h := 25
def man_speed_km_per_h := 2
def crossing_time_sec := 36

-- Convert speeds to m/s
def km_per_h_to_m_per_s (v : ℕ) : ℕ := (v * 1000) / 3600
def train_speed_m_per_s := km_per_h_to_m_per_s train_speed_km_per_h
def man_speed_m_per_s := km_per_h_to_m_per_s man_speed_km_per_h

-- Define the relative speed in m/s
def relative_speed_m_per_s := train_speed_m_per_s + man_speed_m_per_s

-- Theorem to prove the length of the train
theorem train_length : (relative_speed_m_per_s * crossing_time_sec) = 270 :=
by
  -- sorry is used to skip the proof
  sorry

end train_length_l1300_130000


namespace travel_west_l1300_130082

-- Define the condition
def travel_east (d: ℝ) : ℝ := d

-- Define the distance for east
def east_distance := (travel_east 3 = 3)

-- The theorem to prove that traveling west for 2km should be -2km
theorem travel_west (d: ℝ) (h: east_distance) : travel_east (-d) = -d := 
by
  sorry

-- Applying this theorem to the specific case of 2km travel
example (h: east_distance): travel_east (-2) = -2 :=
by 
  apply travel_west 2 h

end travel_west_l1300_130082


namespace roots_numerically_equal_opposite_signs_l1300_130041

theorem roots_numerically_equal_opposite_signs
  (a b c : ℝ) (k : ℝ)
  (h : (∃ x : ℝ, x^2 - (b+1) * x ≠ 0) →
    ∃ x : ℝ, x ≠ 0 ∧ x ∈ {x : ℝ | (k+2)*(x^2 - (b+1)*x) = (k-2)*((a+1)*x - c)} ∧ -x ∈ {x : ℝ | (k+2)*(x^2 - (b+1)*x) = (k-2)*((a+1)*x - c)}) :
  k = (-2 * (b - a)) / (b + a + 2) :=
by
  sorry

end roots_numerically_equal_opposite_signs_l1300_130041


namespace cube_without_lid_configurations_l1300_130021

-- Introduce assumption for cube without a lid
structure CubeWithoutLid

-- Define the proof statement
theorem cube_without_lid_configurations : 
  ∃ (configs : Nat), (configs = 8) :=
by
  sorry

end cube_without_lid_configurations_l1300_130021


namespace polynomial_min_value_l1300_130034

noncomputable def poly (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

theorem polynomial_min_value : 
  ∃ x y : ℝ, poly x y = -18 :=
by
  sorry

end polynomial_min_value_l1300_130034


namespace Lakers_win_in_7_games_l1300_130066

-- Variables for probabilities given in the problem
variable (p_Lakers_win : ℚ := 1 / 4) -- Lakers' probability of winning a single game
variable (p_Celtics_win : ℚ := 3 / 4) -- Celtics' probability of winning a single game

-- Probabilities and combinations
def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_Lakers_win_game7 : ℚ :=
  let first_6_games := binom 6 3 * (p_Lakers_win ^ 3) * (p_Celtics_win ^ 3)
  let seventh_game := p_Lakers_win
  first_6_games * seventh_game

theorem Lakers_win_in_7_games : probability_Lakers_win_game7 = 540 / 16384 := by
  sorry

end Lakers_win_in_7_games_l1300_130066


namespace find_tricycles_l1300_130010

noncomputable def number_of_tricycles (w b t : ℕ) : ℕ := t

theorem find_tricycles : ∃ (w b t : ℕ), 
  (w + b + t = 10) ∧ 
  (2 * b + 3 * t = 25) ∧ 
  (number_of_tricycles w b t = 5) :=
  by 
    sorry

end find_tricycles_l1300_130010


namespace value_of_expression_l1300_130049

theorem value_of_expression 
  (x : ℝ) 
  (h : 7 * x^2 + 6 = 5 * x + 11) 
  : (8 * x - 5)^2 = (2865 - 120 * Real.sqrt 165) / 49 := 
by 
  sorry

end value_of_expression_l1300_130049


namespace largest_divisor_of_m_l1300_130040

-- Definitions
def positive_integer (m : ℕ) : Prop := m > 0
def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement
theorem largest_divisor_of_m (m : ℕ) (h1 : positive_integer m) (h2 : divisible_by (m^2) 54) : ∃ k : ℕ, k = 9 ∧ k ∣ m := 
sorry

end largest_divisor_of_m_l1300_130040


namespace airplane_rows_l1300_130028

theorem airplane_rows (r : ℕ) (h1 : ∀ (seats_per_row total_rows : ℕ), seats_per_row = 8 → total_rows = r →
  ∀ occupied_seats : ℕ, occupied_seats = (3 * seats_per_row) / 4 →
  ∀ unoccupied_seats : ℕ, unoccupied_seats = seats_per_row * total_rows - occupied_seats * total_rows →
  unoccupied_seats = 24): 
  r = 12 :=
by
  sorry

end airplane_rows_l1300_130028


namespace george_earnings_l1300_130084

theorem george_earnings (cars_sold : ℕ) (price_per_car : ℕ) (lego_set_price : ℕ) (h1 : cars_sold = 3) (h2 : price_per_car = 5) (h3 : lego_set_price = 30) :
  cars_sold * price_per_car + lego_set_price = 45 :=
by
  sorry

end george_earnings_l1300_130084


namespace parallelogram_smaller_angle_proof_l1300_130024

noncomputable def smaller_angle (x : ℝ) : Prop :=
  let larger_angle := x + 120
  let angle_sum := x + larger_angle + x + larger_angle = 360
  angle_sum

theorem parallelogram_smaller_angle_proof (x : ℝ) (h1 : smaller_angle x) : x = 30 := by
  sorry

end parallelogram_smaller_angle_proof_l1300_130024


namespace range_of_a_values_l1300_130009

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - |x + 1| + 3 * a ≥ 0

theorem range_of_a_values (a : ℝ) : range_of_a a ↔ a ≥ 1/2 :=
by
  sorry

end range_of_a_values_l1300_130009


namespace probability_X_eq_3_l1300_130023

def number_of_ways_to_choose (n k : ℕ) : ℕ :=
  Nat.choose n k

def P_X_eq_3 : ℚ :=
  (number_of_ways_to_choose 5 3) * (number_of_ways_to_choose 3 1) / (number_of_ways_to_choose 8 4)

theorem probability_X_eq_3 : P_X_eq_3 = 3 / 7 := by
  sorry

end probability_X_eq_3_l1300_130023


namespace distance_origin_to_point_l1300_130007

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l1300_130007


namespace find_f_2_l1300_130099

theorem find_f_2 (f : ℝ → ℝ) (h : ∀ x, f (1 / x + 1) = 2 * x + 3) : f 2 = 5 :=
by
  sorry

end find_f_2_l1300_130099


namespace rods_needed_to_complete_6_step_pyramid_l1300_130031

def rods_in_step (n : ℕ) : ℕ :=
  16 * n

theorem rods_needed_to_complete_6_step_pyramid (rods_1_step rods_2_step : ℕ) :
  rods_1_step = 16 → rods_2_step = 32 → rods_in_step 6 - rods_in_step 4 = 32 :=
by
  intros h1 h2
  sorry

end rods_needed_to_complete_6_step_pyramid_l1300_130031


namespace possible_values_of_m_l1300_130090

theorem possible_values_of_m (m : ℝ) (A B : Set ℝ) (hA : A = {-1, 1}) (hB : B = {x | m * x = 1}) (hUnion : A ∪ B = A) : m = 0 ∨ m = 1 ∨ m = -1 :=
sorry

end possible_values_of_m_l1300_130090


namespace fourth_equation_pattern_l1300_130004

theorem fourth_equation_pattern :
  36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2 :=
by
  sorry

end fourth_equation_pattern_l1300_130004


namespace pizza_area_increase_l1300_130086

theorem pizza_area_increase 
  (r : ℝ) 
  (A_medium A_large : ℝ) 
  (h_medium_area : A_medium = Real.pi * r^2)
  (h_large_area : A_large = Real.pi * (1.40 * r)^2) : 
  ((A_large - A_medium) / A_medium) * 100 = 96 := 
by 
  sorry

end pizza_area_increase_l1300_130086


namespace cos_angle_B_bounds_l1300_130093

theorem cos_angle_B_bounds {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ)
  (angle_ADC : ℝ) (angle_B : ℝ)
  (h1 : AB = 2) (h2 : BC = 3) (h3 : CD = 2) (h4 : angle_ADC = 180 - angle_B) :
  (1 / 4) < Real.cos angle_B ∧ Real.cos angle_B < (3 / 4) := 
sorry -- Proof to be provided

end cos_angle_B_bounds_l1300_130093


namespace quadratic_eq_coeff_l1300_130075

theorem quadratic_eq_coeff (x : ℝ) : 
  (x^2 + 2 = 3 * x) = (∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 2 ∧ (a * x^2 + b * x + c = 0)) :=
by
  sorry

end quadratic_eq_coeff_l1300_130075


namespace find_g_inverse_sum_l1300_130096

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 2 then x^2 - 2 * x + 2 else 3 - x

theorem find_g_inverse_sum :
  (∃ x, g x = -2 ∧ x = 5) ∧
  (∃ x, g x = 0 ∧ x = 3) ∧
  (∃ x, g x = 2 ∧ x = 0) ∧
  (5 + 3 + 0 = 8) := by
  sorry

end find_g_inverse_sum_l1300_130096


namespace theorem_227_l1300_130057

theorem theorem_227 (a b c d : ℤ) (k : ℤ) (h : b ≡ c [ZMOD d]) :
  (a + b ≡ a + c [ZMOD d]) ∧
  (a - b ≡ a - c [ZMOD d]) ∧
  (a * b ≡ a * c [ZMOD d]) :=
by
  sorry

end theorem_227_l1300_130057


namespace original_number_l1300_130042

theorem original_number (x : ℝ) (h1 : 268 * 74 = 19732) (h2 : x * 0.74 = 1.9832) : x = 2.68 :=
by
  sorry

end original_number_l1300_130042


namespace deriv_prob1_deriv_prob2_l1300_130068

noncomputable def prob1 (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem deriv_prob1 : ∀ x, deriv prob1 x = -x * Real.sin x :=
by 
  sorry

noncomputable def prob2 (x : ℝ) : ℝ := x / (Real.exp x - 1)

theorem deriv_prob2 : ∀ x, x ≠ 0 → deriv prob2 x = (Real.exp x * (1 - x) - 1) / (Real.exp x - 1)^2 :=
by
  sorry

end deriv_prob1_deriv_prob2_l1300_130068


namespace solve_inequality_l1300_130051

theorem solve_inequality :
  {x : ℝ | -x^2 + 5 * x > 6} = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end solve_inequality_l1300_130051


namespace jake_peaches_count_l1300_130036

-- Define Jill's peaches
def jill_peaches : ℕ := 5

-- Define Steven's peaches based on the condition that Steven has 18 more peaches than Jill
def steven_peaches : ℕ := jill_peaches + 18

-- Define Jake's peaches based on the condition that Jake has 6 fewer peaches than Steven
def jake_peaches : ℕ := steven_peaches - 6

-- The theorem to prove that Jake has 17 peaches
theorem jake_peaches_count : jake_peaches = 17 := by
  sorry

end jake_peaches_count_l1300_130036


namespace determine_triangle_value_l1300_130013

theorem determine_triangle_value (p : ℕ) (triangle : ℕ) (h1 : triangle + p = 67) (h2 : 3 * (triangle + p) - p = 185) : triangle = 51 := by
  sorry

end determine_triangle_value_l1300_130013


namespace complex_multiplication_l1300_130026

variable (i : ℂ)
axiom i_square : i^2 = -1

theorem complex_multiplication : i * (1 + i) = -1 + i :=
by
  sorry

end complex_multiplication_l1300_130026


namespace only_solution_l1300_130097

theorem only_solution (a : ℤ) : 
  (∀ x : ℤ, x > 0 → 2 * x > 4 * x - 8 → 3 * x - a > -9 → x = 2) →
  (12 ≤ a ∧ a < 15) :=
by
  sorry

end only_solution_l1300_130097


namespace domain_of_function_l1300_130059

noncomputable def domain : Set ℝ := {x | x ≥ 1/2 ∧ x ≠ 1}

theorem domain_of_function : ∀ (x : ℝ), (2 * x - 1 ≥ 0) ∧ (x ^ 2 + x - 2 ≠ 0) ↔ (x ∈ domain) :=
by 
  sorry

end domain_of_function_l1300_130059
