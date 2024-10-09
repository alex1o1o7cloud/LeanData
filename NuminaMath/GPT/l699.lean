import Mathlib

namespace total_price_of_order_l699_69908

-- Define the price of each item
def price_ice_cream_bar : ℝ := 0.60
def price_sundae : ℝ := 1.40

-- Define the quantity of each item
def quantity_ice_cream_bar : ℕ := 125
def quantity_sundae : ℕ := 125

-- Calculate the costs
def cost_ice_cream_bar := quantity_ice_cream_bar * price_ice_cream_bar
def cost_sundae := quantity_sundae * price_sundae

-- Calculate the total cost
def total_cost := cost_ice_cream_bar + cost_sundae

-- Statement of the theorem
theorem total_price_of_order : total_cost = 250 := 
by {
  sorry
}

end total_price_of_order_l699_69908


namespace product_of_odd_primes_mod_32_l699_69961

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l699_69961


namespace find_pairs_of_real_numbers_l699_69922

theorem find_pairs_of_real_numbers (x y : ℝ) :
  (∀ n : ℕ, n > 0 → x * ⌊n * y⌋ = y * ⌊n * x⌋) →
  (x = y ∨ x = 0 ∨ y = 0 ∨ (∃ a b : ℤ, x = a ∧ y = b)) :=
by
  sorry

end find_pairs_of_real_numbers_l699_69922


namespace min_value_correct_l699_69913

noncomputable def min_value (x y : ℝ) : ℝ :=
x * y / (x^2 + y^2)

theorem min_value_correct :
  ∃ x y : ℝ,
    (2 / 5 : ℝ) ≤ x ∧ x ≤ (1 / 2 : ℝ) ∧
    (1 / 3 : ℝ) ≤ y ∧ y ≤ (3 / 8 : ℝ) ∧
    min_value x y = (6 / 13 : ℝ) :=
by sorry

end min_value_correct_l699_69913


namespace Caitlin_age_l699_69983

theorem Caitlin_age (Aunt_Anna_age : ℕ) (h1 : Aunt_Anna_age = 54) (Brianna_age : ℕ) (h2 : Brianna_age = (2 * Aunt_Anna_age) / 3) (Caitlin_age : ℕ) (h3 : Caitlin_age = Brianna_age - 7) : 
  Caitlin_age = 29 := 
  sorry

end Caitlin_age_l699_69983


namespace find_the_number_l699_69981

theorem find_the_number :
  ∃ N : ℝ, ((4/5 : ℝ) * 25 = 20) ∧ (0.40 * N = 24) ∧ (N = 60) :=
by
  sorry

end find_the_number_l699_69981


namespace original_number_l699_69993

-- Define the three-digit number and its permutations under certain conditions.
-- Prove the original number given the specific conditions stated.
theorem original_number (a b c : ℕ)
  (ha : a % 2 = 1) -- a being odd
  (m : ℕ := 100 * a + 10 * b + c)
  (sum_permutations : 100*a + 10*b + c + 100*a + 10*c + b + 100*b + 10*c + a + 
                      100*c + 10*a + b + 100*b + 10*a + c + 100*c + 10*b + a = 3300) :
  m = 192 := 
sorry

end original_number_l699_69993


namespace parabola_focus_coordinates_l699_69907

theorem parabola_focus_coordinates (y x : ℝ) (h : y^2 = 8 * x) : (x, y) = (2, 0) :=
sorry

end parabola_focus_coordinates_l699_69907


namespace find_discriminant_l699_69947

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l699_69947


namespace team_order_l699_69990

-- Define the points of teams
variables (A B C D : ℕ)

-- State the conditions
def condition1 := A + C = B + D
def condition2 := B + A + 5 ≤ D + C
def condition3 := B + C ≥ A + D + 3

-- Statement of the theorem
theorem team_order (h1 : condition1 A B C D) (h2 : condition2 A B C D) (h3 : condition3 A B C D) :
  C > D ∧ D > B ∧ B > A :=
sorry

end team_order_l699_69990


namespace negation_equiv_no_solution_l699_69959

-- Definition of there is at least one solution
def at_least_one_solution (P : α → Prop) : Prop := ∃ x, P x

-- Definition of no solution
def no_solution (P : α → Prop) : Prop := ∀ x, ¬ P x

-- Problem statement to prove that the negation of at_least_one_solution is equivalent to no_solution
theorem negation_equiv_no_solution (P : α → Prop) :
  ¬ at_least_one_solution P ↔ no_solution P := 
sorry

end negation_equiv_no_solution_l699_69959


namespace bananas_to_pears_l699_69952

theorem bananas_to_pears : ∀ (cost_banana cost_apple cost_pear : ℚ),
  (5 * cost_banana = 3 * cost_apple) →
  (9 * cost_apple = 6 * cost_pear) →
  (25 * cost_banana = 10 * cost_pear) :=
by
  intros cost_banana cost_apple cost_pear h1 h2
  sorry

end bananas_to_pears_l699_69952


namespace gcd_seq_coprime_l699_69968

def seq (n : ℕ) : ℕ := 2^(2^n) + 1

theorem gcd_seq_coprime (n k : ℕ) (hnk : n ≠ k) : Nat.gcd (seq n) (seq k) = 1 :=
by
  sorry

end gcd_seq_coprime_l699_69968


namespace problem_M_plus_N_l699_69979

theorem problem_M_plus_N (M N : ℝ) (H1 : 4/7 = M/77) (H2 : 4/7 = 98/(N^2)) : M + N = 57.1 := 
sorry

end problem_M_plus_N_l699_69979


namespace series_sum_correct_l699_69945

open Classical

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (2 * (k+1)) / 4^(k+1)

theorem series_sum_correct :
  series_sum = 8 / 9 :=
by
  sorry

end series_sum_correct_l699_69945


namespace sum_abs_eq_l699_69997

theorem sum_abs_eq (a b : ℝ) (ha : |a| = 10) (hb : |b| = 7) (hab : a > b) : a + b = 17 ∨ a + b = 3 :=
sorry

end sum_abs_eq_l699_69997


namespace caller_wins_both_at_35_l699_69901

theorem caller_wins_both_at_35 (n : ℕ) :
  ∀ n, (n % 5 = 0 ∧ n % 7 = 0) ↔ n = 35 :=
by
  sorry

end caller_wins_both_at_35_l699_69901


namespace commutativity_l699_69978

universe u

variable {M : Type u} [Nonempty M]
variable (star : M → M → M)

axiom star_assoc_right {a b : M} : (star (star a b) b) = a
axiom star_assoc_left {a b : M} : star a (star a b) = b

theorem commutativity (a b : M) : star a b = star b a :=
by sorry

end commutativity_l699_69978


namespace problem_statement_l699_69956

-- Define the power function f and the property that it is odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the given conditions
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_cond : f 3 < f 2)

-- The statement we need to prove
theorem problem_statement : f (-3) > f (-2) := by
  sorry

end problem_statement_l699_69956


namespace min_value_fraction_ineq_l699_69966

-- Define the conditions and statement to be proved
theorem min_value_fraction_ineq (x : ℝ) (hx : x > 4) : 
  ∃ M, M = 4 * Real.sqrt 5 ∧ ∀ y : ℝ, y > 4 → (y + 16) / Real.sqrt (y - 4) ≥ M := 
sorry

end min_value_fraction_ineq_l699_69966


namespace min_sum_of_squares_l699_69976

theorem min_sum_of_squares 
  (x_1 x_2 x_3 : ℝ)
  (h1: x_1 + 3 * x_2 + 4 * x_3 = 72)
  (h2: x_1 = 3 * x_2)
  (h3: 0 < x_1)
  (h4: 0 < x_2)
  (h5: 0 < x_3) : 
  x_1^2 + x_2^2 + x_3^2 = 347.04 := 
sorry

end min_sum_of_squares_l699_69976


namespace tile_area_l699_69948

-- Define the properties and conditions of the tile

structure Tile where
  sides : Fin 9 → ℝ 
  six_of_length_1 : ∀ i : Fin 6, sides i = 1 
  congruent_quadrilaterals : Fin 3 → Quadrilateral

structure Quadrilateral where
  length : ℝ
  width : ℝ

-- Given the tile structure, calculate the area
noncomputable def area_of_tile (t: Tile) : ℝ := sorry

-- Statement: Prove the area of the tile given the conditions
theorem tile_area (t : Tile) : area_of_tile t = (4 * Real.sqrt 3 / 3) :=
  sorry

end tile_area_l699_69948


namespace range_of_a_l699_69991

def set_A (a : ℝ) : Set ℝ := {-1, 0, a}
def set_B : Set ℝ := {x : ℝ | 1/3 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : (set_A a) ∩ set_B ≠ ∅) : 1/3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l699_69991


namespace sale_price_same_as_original_l699_69988

theorem sale_price_same_as_original (x : ℝ) :
  let increased_price := 1.25 * x
  let sale_price := 0.8 * increased_price
  sale_price = x := 
by
  let increased_price := 1.25 * x
  let sale_price := 0.8 * increased_price
  sorry

end sale_price_same_as_original_l699_69988


namespace minimum_shirts_to_save_money_l699_69902

-- Definitions for the costs
def EliteCost (n : ℕ) : ℕ := 30 + 8 * n
def OmegaCost (n : ℕ) : ℕ := 10 + 12 * n

-- Theorem to prove the given solution
theorem minimum_shirts_to_save_money : ∃ n : ℕ, 30 + 8 * n < 10 + 12 * n ∧ n = 6 :=
by {
  sorry
}

end minimum_shirts_to_save_money_l699_69902


namespace gcd_10010_15015_l699_69980

theorem gcd_10010_15015 :
  let n1 := 10010
  let n2 := 15015
  ∃ d, d = Nat.gcd n1 n2 ∧ d = 5005 :=
by
  let n1 := 10010
  let n2 := 15015
  -- ... omitted proof steps
  sorry

end gcd_10010_15015_l699_69980


namespace sin_alpha_value_l699_69933

open Real

theorem sin_alpha_value (α β : ℝ) 
  (h1 : cos (α - β) = 3 / 5) 
  (h2 : sin β = -5 / 13) 
  (h3 : 0 < α ∧ α < π / 2) 
  (h4 : -π / 2 < β ∧ β < 0) 
  : sin α = 33 / 65 :=
sorry

end sin_alpha_value_l699_69933


namespace sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l699_69914

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l699_69914


namespace min_ab_diff_value_l699_69929

noncomputable def min_ab_diff (x y z : ℝ) : ℝ :=
  let A := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12)
  let B := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)
  A^2 - B^2

theorem min_ab_diff_value : ∀ (x y z : ℝ),
  0 ≤ x → 0 ≤ y → 0 ≤ z → min_ab_diff x y z = 36 :=
by
  intros x y z hx hy hz
  sorry

end min_ab_diff_value_l699_69929


namespace molecular_weight_of_compound_l699_69960

-- Definitions of the atomic weights.
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

-- Proof statement of the molecular weight of the compound.
theorem molecular_weight_of_compound :
  (1 * atomic_weight_K) + (1 * atomic_weight_Br) + (3 * atomic_weight_O) = 167.00 :=
  by
    sorry

end molecular_weight_of_compound_l699_69960


namespace average_age_new_students_l699_69994

theorem average_age_new_students (O A_old A_new_avg A_new : ℕ) 
  (hO : O = 8) 
  (hA_old : A_old = 40) 
  (hA_new_avg : A_new_avg = 36)
  (h_total_age_before : O * A_old = 8 * 40)
  (h_total_age_after : (O + 8) * A_new_avg = 16 * 36)
  (h_age_new_students : (16 * 36) - (8 * 40) = A_new * 8) :
  A_new = 32 := 
by 
  sorry

end average_age_new_students_l699_69994


namespace right_angled_triangles_l699_69955

theorem right_angled_triangles (x y z : ℕ) : (x - 6) * (y - 6) = 18 ∧ (x^2 + y^2 = z^2)
  → (3 * (x + y + z) = x * y) :=
sorry

end right_angled_triangles_l699_69955


namespace max_pieces_of_pie_l699_69915

theorem max_pieces_of_pie : ∃ (PIE PIECE : ℕ), 10000 ≤ PIE ∧ PIE < 100000
  ∧ 10000 ≤ PIECE ∧ PIECE < 100000
  ∧ ∃ (n : ℕ), n = 7 ∧ PIE = n * PIECE := by
  sorry

end max_pieces_of_pie_l699_69915


namespace new_person_weight_l699_69992

theorem new_person_weight (n : ℕ) (k : ℝ) (w_old w_new : ℝ) 
  (h_n : n = 6) 
  (h_k : k = 4.5) 
  (h_w_old : w_old = 75) 
  (h_avg_increase : w_new - w_old = n * k) : 
  w_new = 102 := 
sorry

end new_person_weight_l699_69992


namespace min_value_144_l699_69977

noncomputable def min_expression (a b c d : ℝ) : ℝ :=
  (a + b + c) / (a * b * c * d)

theorem min_value_144 (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_pos_d : 0 < d) (h_sum : a + b + c + d = 2) : min_expression a b c d ≥ 144 :=
by
  sorry

end min_value_144_l699_69977


namespace f_even_of_g_odd_l699_69971

theorem f_even_of_g_odd (g : ℝ → ℝ) (f : ℝ → ℝ) (h1 : ∀ x, g (-x) = -g x) (h2 : ∀ x, f x = |g (x^5)|) : ∀ x, f (-x) = f x := 
by
  sorry

end f_even_of_g_odd_l699_69971


namespace negation_of_proposition_l699_69928

theorem negation_of_proposition (x : ℝ) : 
  ¬ (∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := 
sorry

end negation_of_proposition_l699_69928


namespace compare_sums_of_square_roots_l699_69937

theorem compare_sums_of_square_roots
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (M : ℝ := Real.sqrt a + Real.sqrt b) 
  (N : ℝ := Real.sqrt (a + b)) :
  M > N :=
by
  sorry

end compare_sums_of_square_roots_l699_69937


namespace goldbach_10000_l699_69984

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

theorem goldbach_10000 :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p q : ℕ), (p, q) ∈ S → is_prime p ∧ is_prime q ∧ p + q = 10000) ∧ S.card > 3 :=
sorry

end goldbach_10000_l699_69984


namespace park_area_correct_l699_69936

noncomputable def rect_park_area (speed_km_hr : ℕ) (time_min : ℕ) (ratio_l_b : ℕ) : ℕ := by
  let speed_m_min := speed_km_hr * 1000 / 60
  let perimeter := speed_m_min * time_min
  let B := perimeter * 3 / 8
  let L := B / 3
  let area := L * B
  exact area

theorem park_area_correct : rect_park_area 12 8 3 = 120000 := by
  sorry

end park_area_correct_l699_69936


namespace triangle_inequality_l699_69940

variable (a b c : ℝ)
variable (h1 : a * b + b * c + c * a = 18)
variable (h2 : 1 < a)
variable (h3 : 1 < b)
variable (h4 : 1 < c)

theorem triangle_inequality :
  (1 / (a - 1)^3 + 1 / (b - 1)^3 + 1 / (c - 1)^3) > (1 / (a + b + c - 3)) :=
by
  sorry

end triangle_inequality_l699_69940


namespace distance_from_wall_to_picture_edge_l699_69974

theorem distance_from_wall_to_picture_edge
  (wall_width : ℕ)
  (picture_width : ℕ)
  (centered : Prop)
  (h1 : wall_width = 22)
  (h2 : picture_width = 4)
  (h3 : centered) :
  ∃ x : ℕ, x = 9 :=
by
  sorry

end distance_from_wall_to_picture_edge_l699_69974


namespace three_zeros_of_f_l699_69934

noncomputable def f (a x b : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + a + 2) * x + (2 * a + 2) * (Real.log x) + b

theorem three_zeros_of_f (a b : ℝ) (h1 : a > 3) (h2 : a^2 + a + 1 < b) (h3 : b < 2 * a^2 - 2 * a + 2) : 
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 b = 0 ∧ f a x2 b = 0 ∧ f a x3 b = 0 :=
by
  sorry

end three_zeros_of_f_l699_69934


namespace simplify_sum_l699_69926

theorem simplify_sum : 
  (-1: ℤ)^(2010) + (-1: ℤ)^(2011) + (1: ℤ)^(2012) + (-1: ℤ)^(2013) = -2 := by
  sorry

end simplify_sum_l699_69926


namespace largest_of_seven_consecutive_l699_69969

theorem largest_of_seven_consecutive (n : ℕ) (h1 : (7 * n + 21 = 3020)) : (n + 6 = 434) :=
sorry

end largest_of_seven_consecutive_l699_69969


namespace angela_initial_action_figures_l699_69975

theorem angela_initial_action_figures (X : ℕ) (h1 : X - (1/4 : ℚ) * X - (1/3 : ℚ) * (3/4 : ℚ) * X = 12) : X = 24 :=
sorry

end angela_initial_action_figures_l699_69975


namespace propA_neither_sufficient_nor_necessary_l699_69951

def PropA (a b : ℕ) : Prop := a + b ≠ 4
def PropB (a b : ℕ) : Prop := a ≠ 1 ∧ b ≠ 3

theorem propA_neither_sufficient_nor_necessary (a b : ℕ) : 
  ¬((PropA a b → PropB a b) ∧ (PropB a b → PropA a b)) :=
by {
  sorry
}

end propA_neither_sufficient_nor_necessary_l699_69951


namespace frog_escape_probability_l699_69963

def jump_probability (N : ℕ) : ℚ := N / 14

def survival_probability (P : ℕ → ℚ) (N : ℕ) : ℚ :=
  if N = 0 then 0
  else if N = 14 then 1
  else jump_probability N * P (N - 1) + (1 - jump_probability N) * P (N + 1)

theorem frog_escape_probability :
  ∃ (P : ℕ → ℚ), P 0 = 0 ∧ P 14 = 1 ∧ (∀ (N : ℕ), 0 < N ∧ N < 14 → survival_probability P N = P N) ∧ P 3 = 325 / 728 :=
sorry

end frog_escape_probability_l699_69963


namespace symmetrical_circle_l699_69919

-- Defining the given circle's equation
def given_circle_eq (x y: ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Defining the equation of the symmetrical circle
def symmetrical_circle_eq (x y: ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Proving the symmetry property
theorem symmetrical_circle (x y : ℝ) : 
  (given_circle_eq x y) → (symmetrical_circle_eq (-x) (-y)) :=
by
  sorry

end symmetrical_circle_l699_69919


namespace range_of_ab_l699_69911

theorem range_of_ab (a b : ℝ) 
  (h1: ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0 → (2 * a * x - b * y + 2 = 0)) : 
  ab ≤ 0 :=
sorry

end range_of_ab_l699_69911


namespace cos_alpha_minus_pi_over_6_l699_69982

theorem cos_alpha_minus_pi_over_6 (α : Real) 
  (h1 : Real.pi / 2 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.sin (α + Real.pi / 6) = 3 / 5) : 
  Real.cos (α - Real.pi / 6) = (3 * Real.sqrt 3 - 4) / 10 := 
by 
  sorry

end cos_alpha_minus_pi_over_6_l699_69982


namespace production_average_l699_69918

theorem production_average (n : ℕ) (P : ℕ) (hP : P = n * 50)
  (h1 : (P + 95) / (n + 1) = 55) : n = 8 :=
by
  -- skipping the proof
  sorry

end production_average_l699_69918


namespace denomination_of_remaining_coins_l699_69920

/-
There are 324 coins total.
The total value of the coins is Rs. 70.
There are 220 coins of 20 paise each.
Find the denomination of the remaining coins.
-/

def total_coins := 324
def total_value := 7000 -- Rs. 70 converted into paise
def num_20_paise_coins := 220
def value_20_paise_coin := 20
  
theorem denomination_of_remaining_coins :
  let total_remaining_value := total_value - (num_20_paise_coins * value_20_paise_coin)
  let num_remaining_coins := total_coins - num_20_paise_coins
  num_remaining_coins > 0 →
  total_remaining_value / num_remaining_coins = 25 :=
by
  sorry

end denomination_of_remaining_coins_l699_69920


namespace ratio_of_shares_l699_69938

-- Definitions for the given conditions
def capital_A : ℕ := 4500
def capital_B : ℕ := 16200
def months_A : ℕ := 12
def months_B : ℕ := 5 -- B joined after 7 months

-- Effective capital contributions
def effective_capital_A : ℕ := capital_A * months_A
def effective_capital_B : ℕ := capital_B * months_B

-- Defining the statement to prove
theorem ratio_of_shares : effective_capital_A / Nat.gcd effective_capital_A effective_capital_B = 2 ∧ effective_capital_B / Nat.gcd effective_capital_A effective_capital_B = 3 := by
  sorry

end ratio_of_shares_l699_69938


namespace calc_expr_l699_69944

noncomputable def expr_val : ℝ :=
  Real.sqrt 4 - |(-(1 / 4 : ℝ))| + (Real.pi - 2)^0 + 2^(-2 : ℝ)

theorem calc_expr : expr_val = 3 := by
  sorry

end calc_expr_l699_69944


namespace ratio_equivalence_l699_69987

theorem ratio_equivalence (x : ℚ) (h : x / 360 = 18 / 12) : x = 540 :=
by
  -- Proof goes here, to be filled in
  sorry

end ratio_equivalence_l699_69987


namespace zero_of_F_when_a_is_zero_range_of_a_if_P_and_Q_l699_69973

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x
noncomputable def g (a x : ℝ) : ℝ := Real.log (x^2 - 2*x + a)
noncomputable def F (a x : ℝ) : ℝ := f a x + g a x

theorem zero_of_F_when_a_is_zero (x : ℝ) : a = 0 → F a x = 0 → x = 3 := by
  sorry

theorem range_of_a_if_P_and_Q (a : ℝ) :
  (∀ x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ), a - 1/x ≤ 0) ∧
  (∀ x : ℝ, (x^2 - 2*x + a) > 0) →
  1 < a ∧ a ≤ 2 := by
  sorry

end zero_of_F_when_a_is_zero_range_of_a_if_P_and_Q_l699_69973


namespace notebook_cost_l699_69985

theorem notebook_cost (s n c : ℕ) (h1 : s > 20) (h2 : n > 2) (h3 : c > 2 * n) (h4 : s * c * n = 4515) : c = 35 :=
sorry

end notebook_cost_l699_69985


namespace prob_point_closer_to_six_than_zero_l699_69910

theorem prob_point_closer_to_six_than_zero : 
  let interval_start := 0
  let interval_end := 7
  let closer_to_six := fun x => x > ((interval_start + 6) / 2)
  let total_length := interval_end - interval_start
  let length_closer_to_six := interval_end - (interval_start + 6) / 2
  total_length > 0 -> length_closer_to_six / total_length = 4 / 7 :=
by
  sorry

end prob_point_closer_to_six_than_zero_l699_69910


namespace find_z_l699_69912

theorem find_z (a b p q : ℝ) (z : ℝ) 
  (cond : (z + a + b = q * (p * z - a - b))) : 
  z = (a + b) * (q + 1) / (p * q - 1) :=
sorry

end find_z_l699_69912


namespace exists_four_numbers_with_equal_sum_l699_69970

theorem exists_four_numbers_with_equal_sum (S : Finset ℕ) (hS : S.card = 16) (h_range : ∀ n ∈ S, n ≤ 100) :
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ c ≠ d ∧ a ≠ c ∧ b ≠ d ∧ a + b = c + d :=
by
  sorry

end exists_four_numbers_with_equal_sum_l699_69970


namespace no_positive_integers_m_n_l699_69972

theorem no_positive_integers_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  m^3 + 11^3 ≠ n^3 :=
sorry

end no_positive_integers_m_n_l699_69972


namespace combined_rate_mpg_900_over_41_l699_69931

-- Declare the variables and conditions
variables {d : ℕ} (h_d_pos : d > 0)

def combined_mpg (d : ℕ) : ℚ :=
  let anna_car_gasoline := (d : ℚ) / 50
  let ben_car_gasoline  := (d : ℚ) / 20
  let carl_car_gasoline := (d : ℚ) / 15
  let total_gasoline    := anna_car_gasoline + ben_car_gasoline + carl_car_gasoline
  ((3 : ℚ) * d) / total_gasoline

-- Define the theorem statement
theorem combined_rate_mpg_900_over_41 :
  ∀ d : ℕ, d > 0 → combined_mpg d = 900 / 41 :=
by
  intros d h_d_pos
  rw [combined_mpg]
  -- Steps following the solution
  sorry -- proof omitted

end combined_rate_mpg_900_over_41_l699_69931


namespace islanders_liars_l699_69942

theorem islanders_liars (n : ℕ) (h : n = 450) : (∃ L : ℕ, (L = 150 ∨ L = 450)) :=
sorry

end islanders_liars_l699_69942


namespace triangle_area_l699_69954

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end triangle_area_l699_69954


namespace graph_passes_through_point_l699_69967

theorem graph_passes_through_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ∃ p : ℝ × ℝ, p = (2, 0) ∧ ∀ x, (x = 2 → a ^ (x - 2) - 1 = 0) :=
by
  sorry

end graph_passes_through_point_l699_69967


namespace sum_of_legs_l699_69939

theorem sum_of_legs (x : ℕ) (h : x^2 + (x + 1)^2 = 41^2) : x + (x + 1) = 57 :=
sorry

end sum_of_legs_l699_69939


namespace cars_without_paying_l699_69905

theorem cars_without_paying (total_cars : ℕ) (percent_with_tickets : ℚ) (fraction_with_passes : ℚ)
  (h1 : total_cars = 300)
  (h2 : percent_with_tickets = 0.75)
  (h3 : fraction_with_passes = 1/5) :
  let cars_with_tickets := percent_with_tickets * total_cars
  let cars_with_passes := fraction_with_passes * cars_with_tickets
  total_cars - (cars_with_tickets + cars_with_passes) = 30 :=
by
  -- Placeholder proof
  sorry

end cars_without_paying_l699_69905


namespace no_solution_set_1_2_4_l699_69958

theorem no_solution_set_1_2_4 
  (f : ℝ → ℝ) 
  (hf : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c)
  (t : ℝ) : ¬ ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (|x1 - t|) = 0 ∧ f (|x2 - t|) = 0 ∧ f (|x3 - t|) = 0 ∧ (x1 = 1 ∧ x2 = 2 ∧ x3 = 4) := 
sorry

end no_solution_set_1_2_4_l699_69958


namespace units_digit_of_42_pow_3_add_24_pow_3_l699_69964

theorem units_digit_of_42_pow_3_add_24_pow_3 :
    (42 ^ 3 + 24 ^ 3) % 10 = 2 :=
by
    have units_digit_42 := (42 % 10 = 2)
    have units_digit_24 := (24 % 10 = 4)
    sorry

end units_digit_of_42_pow_3_add_24_pow_3_l699_69964


namespace min_distance_PQ_l699_69943

theorem min_distance_PQ :
  ∀ (P Q : ℝ × ℝ), (P.1 - P.2 - 4 = 0) → (Q.1^2 = 4 * Q.2) →
  ∃ (d : ℝ), d = dist P Q ∧ d = 3 * Real.sqrt 2 / 2 :=
sorry

end min_distance_PQ_l699_69943


namespace replace_digits_and_check_divisibility_l699_69923

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

end replace_digits_and_check_divisibility_l699_69923


namespace find_f_of_2011_l699_69946

-- Define the function f
def f (x : ℝ) (a b c : ℝ) := a * x^5 + b * x^3 + c * x + 7

-- The main statement we need to prove
theorem find_f_of_2011 (a b c : ℝ) (h : f (-2011) a b c = -17) : f 2011 a b c = 31 :=
by
  sorry

end find_f_of_2011_l699_69946


namespace probability_of_a_b_c_l699_69965

noncomputable def probability_condition : ℚ :=
  5 / 6 * 5 / 6 * 7 / 8

theorem probability_of_a_b_c : 
  let a_outcome := 6
  let b_outcome := 6
  let c_outcome := 8
  (1 / a_outcome) * (1 / b_outcome) * (1 / c_outcome) = probability_condition :=
sorry

end probability_of_a_b_c_l699_69965


namespace ratio_area_square_circle_eq_pi_l699_69998

theorem ratio_area_square_circle_eq_pi
  (a r : ℝ)
  (h : 4 * a = 4 * π * r) :
  (a^2 / (π * r^2)) = π := by
  sorry

end ratio_area_square_circle_eq_pi_l699_69998


namespace common_difference_arithmetic_seq_l699_69925

theorem common_difference_arithmetic_seq (S n a1 d : ℕ) (h_sum : S = 650) (h_n : n = 20) (h_a1 : a1 = 4) :
  S = (n / 2) * (2 * a1 + (n - 1) * d) → d = 3 := by
  intros h_formula
  sorry

end common_difference_arithmetic_seq_l699_69925


namespace find_room_dimension_l699_69927

noncomputable def unknown_dimension_of_room 
  (cost_per_sq_ft : ℕ)
  (total_cost : ℕ)
  (w : ℕ)
  (l : ℕ)
  (h : ℕ)
  (door_h : ℕ)
  (door_w : ℕ)
  (window_h : ℕ)
  (window_w : ℕ)
  (num_windows : ℕ) : ℕ := sorry

theorem find_room_dimension :
  unknown_dimension_of_room 10 9060 25 15 12 6 3 4 3 3 = 25 :=
sorry

end find_room_dimension_l699_69927


namespace find_n_l699_69989

theorem find_n (n : ℝ) (h1 : ∀ x y : ℝ, (n + 1) * x^(n^2 - 5) = y) 
               (h2 : ∀ x > 0, (n + 1) * x^(n^2 - 5) > 0) :
               n = 2 :=
by
  sorry

end find_n_l699_69989


namespace sum_of_bases_l699_69935

theorem sum_of_bases (R₁ R₂ : ℕ) 
    (h1 : (4 * R₁ + 5) / (R₁^2 - 1) = (3 * R₂ + 4) / (R₂^2 - 1))
    (h2 : (5 * R₁ + 4) / (R₁^2 - 1) = (4 * R₂ + 3) / (R₂^2 - 1)) : 
    R₁ + R₂ = 23 := 
sorry

end sum_of_bases_l699_69935


namespace no_p_dependence_l699_69999

theorem no_p_dependence (m : ℕ) (p : ℕ) (hp : Prime p) (hm : m < p)
  (n : ℕ) (hn : 0 < n) (k : ℕ) 
  (h : m^2 + n^2 + p^2 - 2*m*n - 2*m*p - 2*n*p = k^2) : 
  ∀ q : ℕ, Prime q → m < q → (m^2 + n^2 + q^2 - 2*m*n - 2*m*q - 2*n*q = k^2) :=
by sorry

end no_p_dependence_l699_69999


namespace rachel_picture_shelves_l699_69949

-- We define the number of books per shelf
def books_per_shelf : ℕ := 9

-- We define the number of mystery shelves
def mystery_shelves : ℕ := 6

-- We define the total number of books
def total_books : ℕ := 72

-- We create a theorem that states Rachel had 2 shelves of picture books
theorem rachel_picture_shelves : ∃ (picture_shelves : ℕ), 
  (mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = total_books) ∧
  picture_shelves = 2 := by
  sorry

end rachel_picture_shelves_l699_69949


namespace positive_diff_solutions_abs_eq_12_l699_69921

theorem positive_diff_solutions_abs_eq_12 : 
  ∀ (x1 x2 : ℤ), (|x1 - 4| = 12) ∧ (|x2 - 4| = 12) ∧ (x1 > x2) → (x1 - x2 = 24) :=
by
  sorry

end positive_diff_solutions_abs_eq_12_l699_69921


namespace tenth_term_arith_seq_l699_69909

variable (a1 d : Int) -- Initial term and common difference
variable (n : Nat) -- nth term

-- Definition of the nth term in an arithmetic sequence
def arithmeticSeq (a1 d : Int) (n : Nat) : Int :=
  a1 + (n - 1) * d

-- Specific values for the problem
def a_10 : Int :=
  arithmeticSeq 10 (-3) 10

-- The theorem we want to prove
theorem tenth_term_arith_seq : a_10 = -17 := by
  sorry

end tenth_term_arith_seq_l699_69909


namespace radius_calculation_l699_69962

noncomputable def radius_of_circle (n : ℕ) : ℝ :=
if 2 ≤ n ∧ n ≤ 11 then
  if n ≤ 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  1.61
else
  0  -- Outside the specified range

theorem radius_calculation (n : ℕ) (hn : 2 ≤ n ∧ n ≤ 11) :
  radius_of_circle n =
  if n ≤ 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  1.61 :=
sorry

end radius_calculation_l699_69962


namespace trader_profit_percent_equal_eight_l699_69930

-- Defining the initial conditions
def original_price (P : ℝ) := P
def purchased_price (P : ℝ) := 0.60 * original_price P
def selling_price (P : ℝ) := 1.80 * purchased_price P

-- Statement to be proved
theorem trader_profit_percent_equal_eight (P : ℝ) (h : P > 0) :
  ((selling_price P - original_price P) / original_price P) * 100 = 8 :=
by
  sorry

end trader_profit_percent_equal_eight_l699_69930


namespace total_swim_distance_five_weeks_total_swim_time_five_weeks_l699_69904

-- Definitions of swim distances and times based on Jasmine's routine 
def monday_laps : ℕ := 10
def tuesday_laps : ℕ := 15
def tuesday_aerobics_time : ℕ := 20
def wednesday_laps : ℕ := 12
def wednesday_time_per_lap : ℕ := 2
def thursday_laps : ℕ := 18
def friday_laps : ℕ := 20

-- Proving total swim distance for five weeks
theorem total_swim_distance_five_weeks : (5 * (monday_laps + tuesday_laps + wednesday_laps + thursday_laps + friday_laps)) = 375 := 
by 
  sorry

-- Proving total swim time for five weeks (partially solvable)
theorem total_swim_time_five_weeks : (5 * (tuesday_aerobics_time + wednesday_laps * wednesday_time_per_lap)) = 220 := 
by 
  sorry

end total_swim_distance_five_weeks_total_swim_time_five_weeks_l699_69904


namespace total_distance_is_3_miles_l699_69924

-- Define conditions
def running_speed := 6   -- mph
def walking_speed := 2   -- mph
def running_time := 20 / 60   -- hours
def walking_time := 30 / 60   -- hours

-- Define total distance
def total_distance := (running_speed * running_time) + (walking_speed * walking_time)

theorem total_distance_is_3_miles : total_distance = 3 :=
by
  sorry

end total_distance_is_3_miles_l699_69924


namespace tangent_line_value_l699_69957

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2 * x

theorem tangent_line_value (a b : ℝ) (h : a ≤ 0) 
  (h_tangent : ∀ x : ℝ, f a x = 2 * x + b) : a - 2 * b = 2 :=
sorry

end tangent_line_value_l699_69957


namespace no_egg_arrangements_possible_l699_69953

noncomputable def num_egg_arrangements 
  (total_eggs : ℕ) 
  (type_A_eggs : ℕ) 
  (type_B_eggs : ℕ)
  (type_C_eggs : ℕ)
  (groups : ℕ)
  (ratio_A : ℕ) 
  (ratio_B : ℕ) 
  (ratio_C : ℕ) : ℕ :=
if (total_eggs = type_A_eggs + type_B_eggs + type_C_eggs) ∧ 
   (type_A_eggs / groups = ratio_A) ∧ 
   (type_B_eggs / groups = ratio_B) ∧ 
   (type_C_eggs / groups = ratio_C) then 0 else 0

theorem no_egg_arrangements_possible :
  num_egg_arrangements 35 15 12 8 5 2 3 1 = 0 := 
by sorry

end no_egg_arrangements_possible_l699_69953


namespace slope_of_line_passing_through_MN_l699_69995

theorem slope_of_line_passing_through_MN :
  let M := (-2, 1)
  let N := (1, 4)
  ∃ m : ℝ, m = (N.2 - M.2) / (N.1 - M.1) ∧ m = 1 :=
by
  sorry

end slope_of_line_passing_through_MN_l699_69995


namespace isosceles_triangle_area_48_l699_69986

noncomputable def isosceles_triangle_area (b h s : ℝ) : ℝ :=
  (1 / 2) * (2 * b) * h

theorem isosceles_triangle_area_48 :
  ∀ (b s : ℝ),
  b ^ 2 + 8 ^ 2 = s ^ 2 ∧ s + b = 16 →
  isosceles_triangle_area b 8 s = 48 :=
by
  intros b s h
  unfold isosceles_triangle_area
  sorry

end isosceles_triangle_area_48_l699_69986


namespace min_value_proof_l699_69906

noncomputable def min_value_expression (a b : ℝ) : ℝ :=
  (1 / (12 * a + 1)) + (1 / (8 * b + 1))

theorem min_value_proof (a b : ℝ) (h1 : 3 * a + 2 * b = 1) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  min_value_expression a b = 2 / 3 :=
sorry

end min_value_proof_l699_69906


namespace intersection_M_N_l699_69950

open Set

def M := { x : ℝ | 0 < x ∧ x < 3 }
def N := { x : ℝ | x^2 - 5 * x + 4 ≥ 0 }

theorem intersection_M_N :
  { x | x ∈ M ∧ x ∈ N } = { x | 0 < x ∧ x ≤ 1 } :=
sorry

end intersection_M_N_l699_69950


namespace difference_is_three_l699_69932

-- Define the range for two-digit numbers
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define whether a number is a multiple of three
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

-- Identify the smallest and largest two-digit multiples of three
def smallest_two_digit_multiple_of_three : ℕ := 12
def largest_two_digit_multiple_of_three : ℕ := 99

-- Identify the smallest and largest two-digit non-multiples of three
def smallest_two_digit_non_multiple_of_three : ℕ := 10
def largest_two_digit_non_multiple_of_three : ℕ := 98

-- Calculate Joey's sum
def joeys_sum : ℕ := smallest_two_digit_multiple_of_three + largest_two_digit_multiple_of_three

-- Calculate Zoë's sum
def zoes_sum : ℕ := smallest_two_digit_non_multiple_of_three + largest_two_digit_non_multiple_of_three

-- Prove the difference between Joey's and Zoë's sums is 3
theorem difference_is_three : joeys_sum - zoes_sum = 3 :=
by
  -- The proof is not given, so we use sorry here
  sorry

end difference_is_three_l699_69932


namespace find_a17_a18_a19_a20_l699_69900

variable {α : Type*} [Field α]

-- Definitions based on the given conditions:
def geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a n = a 0 * r ^ n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (Finset.range n).sum a

-- Problem statement based on the question and conditions:
theorem find_a17_a18_a19_a20 (a S : ℕ → α) (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S) (hS4 : S 4 = 1) (hS8 : S 8 = 3) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
sorry

end find_a17_a18_a19_a20_l699_69900


namespace transform_fraction_l699_69996

theorem transform_fraction (x : ℝ) (h₁ : x ≠ 3) : - (1 / (3 - x)) = (1 / (x - 3)) := 
    sorry

end transform_fraction_l699_69996


namespace determine_m_l699_69916

variable {x y z : ℝ}

theorem determine_m (h : (5 / (x + y)) = (m / (x + z)) ∧ (m / (x + z)) = (13 / (z - y))) : m = 18 :=
by
  sorry

end determine_m_l699_69916


namespace draw_from_unit_D_l699_69941

variable (d : ℕ)

-- Variables representing the number of questionnaires drawn from A, B, C, and D
def QA : ℕ := 30 - d
def QB : ℕ := 30
def QC : ℕ := 30 + d
def QD : ℕ := 30 + 2 * d

-- Total number of questionnaires drawn
def TotalDrawn : ℕ := QA d + QB + QC d + QD d

theorem draw_from_unit_D :
  (TotalDrawn d = 150) →
  QD d = 60 := sorry

end draw_from_unit_D_l699_69941


namespace determine_y_l699_69903

def diamond (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

theorem determine_y (y : ℝ) (h : diamond 4 y = 30) : y = 5 / 3 :=
by sorry

end determine_y_l699_69903


namespace math_problem_l699_69917

def otimes (a b : ℚ) : ℚ := (a^3) / (b^2)

theorem math_problem : ((otimes (otimes 2 4) 6) - (otimes 2 (otimes 4 6))) = -23327 / 288 := by sorry

end math_problem_l699_69917
