import Mathlib

namespace NUMINAMATH_GPT_hyperbola_eccentricity_l251_25181

theorem hyperbola_eccentricity 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (focus_on_x_axis : ∃ c : ℝ, c > 0)
  (asymptote_eq : ∀ x y : ℝ, (4 + 3 * y = 0) ∨ (4 - 3 * y = 0)) :
  ∃ e : ℝ, e = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l251_25181


namespace NUMINAMATH_GPT_find_k_l251_25146

def f (a b c x : Int) : Int := a * x^2 + b * x + c

theorem find_k (a b c k : Int)
  (h₁ : f a b c 2 = 0)
  (h₂ : 100 < f a b c 7 ∧ f a b c 7 < 110)
  (h₃ : 120 < f a b c 8 ∧ f a b c 8 < 130)
  (h₄ : 6000 * k < f a b c 100 ∧ f a b c 100 < 6000 * (k + 1)) :
  k = 0 := 
sorry

end NUMINAMATH_GPT_find_k_l251_25146


namespace NUMINAMATH_GPT_root_relationship_l251_25173

theorem root_relationship (a x₁ x₂ : ℝ) 
  (h_eqn : x₁^2 - (2*a + 1)*x₁ + a^2 + 2 = 0)
  (h_roots : x₂ = 2*x₁)
  (h_vieta1 : x₁ + x₂ = 2*a + 1)
  (h_vieta2 : x₁ * x₂ = a^2 + 2) : 
  a = 4 := 
sorry

end NUMINAMATH_GPT_root_relationship_l251_25173


namespace NUMINAMATH_GPT_wire_ratio_bonnie_roark_l251_25143

-- Definitions from the conditions
def bonnie_wire_length : ℕ := 12 * 8
def bonnie_volume : ℕ := 8 ^ 3
def roark_cube_side : ℕ := 2
def roark_cube_volume : ℕ := roark_cube_side ^ 3
def num_roark_cubes : ℕ := bonnie_volume / roark_cube_volume
def roark_wire_length_per_cube : ℕ := 12 * roark_cube_side
def roark_total_wire_length : ℕ := num_roark_cubes * roark_wire_length_per_cube

-- Statement to prove
theorem wire_ratio_bonnie_roark : 
  ((bonnie_wire_length : ℚ) / roark_total_wire_length) = (1 / 16) :=
by
  sorry

end NUMINAMATH_GPT_wire_ratio_bonnie_roark_l251_25143


namespace NUMINAMATH_GPT_max_sheep_pen_area_l251_25100

theorem max_sheep_pen_area :
  ∃ x y : ℝ, 15 * 2 = 30 ∧ (x + 2 * y = 30) ∧
  (x > 0 ∧ y > 0) ∧
  (x * y = 112) := by
  sorry

end NUMINAMATH_GPT_max_sheep_pen_area_l251_25100


namespace NUMINAMATH_GPT_inequality_proof_l251_25192

variable (m : ℕ) (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1)

theorem inequality_proof :
    (m > 0) →
    (x^m / ((1 + y) * (1 + z)) + y^m / ((1 + x) * (1 + z)) + z^m / ((1 + x) * (1 + y)) >= 3/4) :=
by
  intro hm_pos
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_inequality_proof_l251_25192


namespace NUMINAMATH_GPT_distributive_laws_fail_for_all_l251_25102

def has_op_hash (a b : ℝ) : ℝ := a + 2 * b

theorem distributive_laws_fail_for_all (x y z : ℝ) : 
  ¬ (∀ x y z, has_op_hash x (y + z) = has_op_hash x y + has_op_hash x z) ∧
  ¬ (∀ x y z, x + has_op_hash y z = has_op_hash (x + y) (x + z)) ∧
  ¬ (∀ x y z, has_op_hash x (has_op_hash y z) = has_op_hash (has_op_hash x y) (has_op_hash x z)) := 
sorry

end NUMINAMATH_GPT_distributive_laws_fail_for_all_l251_25102


namespace NUMINAMATH_GPT_complex_number_solution_l251_25199

theorem complex_number_solution (i : ℂ) (h : i^2 = -1) : (5 / (2 - i) - i = 2) :=
  sorry

end NUMINAMATH_GPT_complex_number_solution_l251_25199


namespace NUMINAMATH_GPT_injective_function_identity_l251_25165

theorem injective_function_identity (f : ℕ → ℕ) (h_inj : Function.Injective f)
  (h : ∀ (m n : ℕ), 0 < m → 0 < n → f (n * f m) ≤ n * m) : ∀ x : ℕ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_injective_function_identity_l251_25165


namespace NUMINAMATH_GPT_garage_motorcycles_l251_25120

theorem garage_motorcycles (bicycles cars motorcycles total_wheels : ℕ)
  (hb : bicycles = 20)
  (hc : cars = 10)
  (hw : total_wheels = 90)
  (wb : bicycles * 2 = 40)
  (wc : cars * 4 = 40)
  (wm : motorcycles * 2 = total_wheels - (bicycles * 2 + cars * 4)) :
  motorcycles = 5 := 
  by 
  sorry

end NUMINAMATH_GPT_garage_motorcycles_l251_25120


namespace NUMINAMATH_GPT_find_k_l251_25142

theorem find_k (k : ℝ) (A B : ℝ × ℝ) 
  (hA : A = (2, 3)) (hB : B = (4, k)) 
  (hAB_parallel : A.2 = B.2) : k = 3 := 
by 
  have hA_def : A = (2, 3) := hA 
  have hB_def : B = (4, k) := hB 
  have parallel_condition: A.2 = B.2 := hAB_parallel
  simp at parallel_condition
  sorry

end NUMINAMATH_GPT_find_k_l251_25142


namespace NUMINAMATH_GPT_min_x_prime_factorization_sum_eq_31_l251_25150

theorem min_x_prime_factorization_sum_eq_31
    (x y a b c d : ℕ)
    (hx : x > 0)
    (hy : y > 0)
    (h : 7 * x^5 = 11 * y^13)
    (hx_prime_fact : ∃ a c b d : ℕ, x = a^c * b^d) :
    a + b + c + d = 31 :=
by
 sorry
 
end NUMINAMATH_GPT_min_x_prime_factorization_sum_eq_31_l251_25150


namespace NUMINAMATH_GPT_evaluate_fg_of_2_l251_25189

def f (x : ℝ) : ℝ := x ^ 3
def g (x : ℝ) : ℝ := 4 * x + 5

theorem evaluate_fg_of_2 : f (g 2) = 2197 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fg_of_2_l251_25189


namespace NUMINAMATH_GPT_complete_the_square_l251_25149

theorem complete_the_square (y : ℝ) : (y^2 + 12*y + 40) = (y + 6)^2 + 4 := by
  sorry

end NUMINAMATH_GPT_complete_the_square_l251_25149


namespace NUMINAMATH_GPT_distinct_primes_eq_1980_l251_25172

theorem distinct_primes_eq_1980 (p q r A : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
    (hne1 : p ≠ q) (hne2 : q ≠ r) (hne3 : p ≠ r) 
    (h1 : 2 * p * q * r + 50 * p * q = A)
    (h2 : 7 * p * q * r + 55 * p * r = A)
    (h3 : 8 * p * q * r + 12 * q * r = A) : 
    A = 1980 := by {
  sorry
}

end NUMINAMATH_GPT_distinct_primes_eq_1980_l251_25172


namespace NUMINAMATH_GPT_determine_constants_l251_25155

theorem determine_constants (α β : ℝ) (h_eq : ∀ x, (x - α) / (x + β) = (x^2 - 96 * x + 2210) / (x^2 + 65 * x - 3510))
  (h_num : ∀ x, x^2 - 96 * x + 2210 = (x - 34) * (x - 62))
  (h_denom : ∀ x, x^2 + 65 * x - 3510 = (x - 45) * (x + 78)) :
  α + β = 112 :=
sorry

end NUMINAMATH_GPT_determine_constants_l251_25155


namespace NUMINAMATH_GPT_ax_by_powers_l251_25103

theorem ax_by_powers (a b x y : ℝ) (h1 : a * x + b * y = 5) 
                      (h2: a * x^2 + b * y^2 = 11)
                      (h3: a * x^3 + b * y^3 = 25)
                      (h4: a * x^4 + b * y^4 = 59) : 
                      a * x^5 + b * y^5 = 145 := 
by 
  -- Include the proof steps here if needed 
  sorry

end NUMINAMATH_GPT_ax_by_powers_l251_25103


namespace NUMINAMATH_GPT_train_length_l251_25163

theorem train_length 
    (t : ℝ) 
    (s_kmh : ℝ) 
    (s_mps : ℝ)
    (h1 : t = 2.222044458665529) 
    (h2 : s_kmh = 162) 
    (h3 : s_mps = s_kmh * (5 / 18))
    (L : ℝ)
    (h4 : L = s_mps * t) : 
  L = 100 := 
sorry

end NUMINAMATH_GPT_train_length_l251_25163


namespace NUMINAMATH_GPT_correct_decimal_product_l251_25122

theorem correct_decimal_product : (0.125 * 3.2 = 4.0) :=
sorry

end NUMINAMATH_GPT_correct_decimal_product_l251_25122


namespace NUMINAMATH_GPT_largest_n_condition_l251_25190

theorem largest_n_condition :
  ∃ n : ℤ, (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧ ∃ k : ℤ, 2 * n + 99 = k^2 ∧ ∀ x : ℤ, 
  (∃ m' : ℤ, x^2 = (m' + 1)^3 - m'^3) ∧ ∃ k' : ℤ, 2 * x + 99 = k'^2 → x ≤ 289 :=
sorry

end NUMINAMATH_GPT_largest_n_condition_l251_25190


namespace NUMINAMATH_GPT_total_money_spent_l251_25183

-- Assume Keanu gave dog 40 fish
def dog_fish := 40

-- Assume Keanu gave cat half as many fish as he gave to his dog
def cat_fish := dog_fish / 2

-- Assume each fish cost $4
def cost_per_fish := 4

-- Prove that total amount of money spent is $240
theorem total_money_spent : (dog_fish + cat_fish) * cost_per_fish = 240 := 
by
  sorry

end NUMINAMATH_GPT_total_money_spent_l251_25183


namespace NUMINAMATH_GPT_find_k_value_l251_25114

theorem find_k_value : 
  (∃ (x y k : ℝ), x = -6.8 ∧ 
  (y = 0.25 * x + 10) ∧ 
  (k = -3 * x + y) ∧ 
  k = 32.1) :=
sorry

end NUMINAMATH_GPT_find_k_value_l251_25114


namespace NUMINAMATH_GPT_power_function_value_at_minus_two_l251_25107

-- Define the power function assumption and points
variable (f : ℝ → ℝ)
variable (hf : f (1 / 2) = 8)

-- Prove that the given condition implies the required result
theorem power_function_value_at_minus_two : f (-2) = -1 / 8 := 
by {
  -- proof to be filled here
  sorry
}

end NUMINAMATH_GPT_power_function_value_at_minus_two_l251_25107


namespace NUMINAMATH_GPT_two_to_the_n_plus_3_is_perfect_square_l251_25153

theorem two_to_the_n_plus_3_is_perfect_square (n : ℕ) (h : ∃ a : ℕ, 2^n + 3 = a^2) : n = 0 := 
sorry

end NUMINAMATH_GPT_two_to_the_n_plus_3_is_perfect_square_l251_25153


namespace NUMINAMATH_GPT_fractions_order_and_non_equality_l251_25161

theorem fractions_order_and_non_equality:
  (37 / 29 < 41 / 31) ∧ (41 / 31 < 31 / 23) ∧ 
  ((37 / 29 ≠ 4 / 3) ∧ (41 / 31 ≠ 4 / 3) ∧ (31 / 23 ≠ 4 / 3)) := by
  sorry

end NUMINAMATH_GPT_fractions_order_and_non_equality_l251_25161


namespace NUMINAMATH_GPT_phase_and_initial_phase_theorem_l251_25125

open Real

noncomputable def phase_and_initial_phase (x : ℝ) : ℝ := 3 * sin (-x + π / 6)

theorem phase_and_initial_phase_theorem :
  ∃ φ : ℝ, ∃ ψ : ℝ,
    ∀ x : ℝ, phase_and_initial_phase x = 3 * sin (x + φ) ∧
    (φ = 5 * π / 6) ∧ (ψ = φ) :=
sorry

end NUMINAMATH_GPT_phase_and_initial_phase_theorem_l251_25125


namespace NUMINAMATH_GPT_min_training_iterations_l251_25164

/-- The model of exponentially decaying learning rate is given by L = L0 * D^(G / G0)
    where
    L  : the learning rate used in each round of optimization,
    L0 : the initial learning rate,
    D  : the decay coefficient,
    G  : the number of training iterations,
    G0 : the decay rate.

    Given:
    - the initial learning rate L0 = 0.5,
    - the decay rate G0 = 18,
    - when G = 18, L = 0.4,

    Prove: 
    The minimum number of training iterations required for the learning rate to decay to below 0.1 (excluding 0.1) is 130.
-/
theorem min_training_iterations
  (L0 : ℝ) (G0 : ℝ) (D : ℝ) (G : ℝ) (L : ℝ)
  (h1 : L0 = 0.5)
  (h2 : G0 = 18)
  (h3 : L = 0.4)
  (h4 : G = 18)
  (h5 : L0 * D^(G / G0) = 0.4)
  : ∃ G, G ≥ 130 ∧ L0 * D^(G / G0) < 0.1 := sorry

end NUMINAMATH_GPT_min_training_iterations_l251_25164


namespace NUMINAMATH_GPT_number_of_good_carrots_l251_25184

def total_carrots (nancy_picked : ℕ) (mom_picked : ℕ) : ℕ :=
  nancy_picked + mom_picked

def bad_carrots := 14

def good_carrots (total : ℕ) (bad : ℕ) : ℕ :=
  total - bad

theorem number_of_good_carrots :
  good_carrots (total_carrots 38 47) bad_carrots = 71 := by
  sorry

end NUMINAMATH_GPT_number_of_good_carrots_l251_25184


namespace NUMINAMATH_GPT_smallest_even_number_of_seven_l251_25115

-- Conditions: The sum of seven consecutive even numbers is 700.
-- We need to prove that the smallest of these numbers is 94.

theorem smallest_even_number_of_seven (n : ℕ) (hn : 7 * n = 700) :
  ∃ (a b c d e f g : ℕ), 
  (2 * a + 4 * b + 6 * c + 8 * d + 10 * e + 12 * f + 14 * g = 700) ∧ 
  (a = b - 1) ∧ (b = c - 1) ∧ (c = d - 1) ∧ (d = e - 1) ∧ (e = f - 1) ∧ 
  (f = g - 1) ∧ (g = 100) ∧ (a = 94) :=
by
  -- This is the theorem statement. 
  sorry

end NUMINAMATH_GPT_smallest_even_number_of_seven_l251_25115


namespace NUMINAMATH_GPT_find_prime_pairs_l251_25101

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_pair (p q : ℕ) : Prop := 
  p < 2023 ∧ q < 2023 ∧ 
  p ∣ q^2 + 8 ∧ q ∣ p^2 + 8

theorem find_prime_pairs : 
  ∀ (p q : ℕ), is_prime p → is_prime q → valid_pair p q → 
    (p = 2 ∧ q = 2) ∨ 
    (p = 17 ∧ q = 3) ∨ 
    (p = 11 ∧ q = 5) :=
by 
  sorry

end NUMINAMATH_GPT_find_prime_pairs_l251_25101


namespace NUMINAMATH_GPT_smallest_x_for_non_prime_expression_l251_25157

/-- The smallest positive integer x for which x^2 + x + 41 is not a prime number is 40. -/
theorem smallest_x_for_non_prime_expression : ∃ x : ℕ, x > 0 ∧ x^2 + x + 41 = 41 * 41 ∧ (∀ y : ℕ, 0 < y ∧ y < x → Prime (y^2 + y + 41)) := 
sorry

end NUMINAMATH_GPT_smallest_x_for_non_prime_expression_l251_25157


namespace NUMINAMATH_GPT_initial_cases_purchased_l251_25195

open Nat

-- Definitions based on conditions

def group1_children := 14
def group2_children := 16
def group3_children := 12
def group4_children := (group1_children + group2_children + group3_children) / 2
def total_children := group1_children + group2_children + group3_children + group4_children

def bottles_per_child_per_day := 3
def days := 3
def total_bottles_needed := total_children * bottles_per_child_per_day * days

def additional_bottles_needed := 255

def bottles_per_case := 24
def initial_bottles := total_bottles_needed - additional_bottles_needed

def cases_purchased := initial_bottles / bottles_per_case

-- Theorem to prove the number of cases purchased initially
theorem initial_cases_purchased : cases_purchased = 13 :=
  sorry

end NUMINAMATH_GPT_initial_cases_purchased_l251_25195


namespace NUMINAMATH_GPT_original_price_of_coat_l251_25119

theorem original_price_of_coat (P : ℝ) (h : 0.70 * P = 350) : P = 500 :=
sorry

end NUMINAMATH_GPT_original_price_of_coat_l251_25119


namespace NUMINAMATH_GPT_sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l251_25147

variable (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_52 : (Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l251_25147


namespace NUMINAMATH_GPT_arithmetic_sum_S11_l251_25191

theorem arithmetic_sum_S11 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ)
  (h_arith : ∀ n, a (n+1) - a n = d) -- The sequence is arithmetic with common difference d
  (h_sum : S n = n * (a 1 + a n) / 2) -- Sum of the first n terms definition
  (h_condition: a 3 + a 6 + a 9 = 54) :
  S 11 = 198 := 
sorry

end NUMINAMATH_GPT_arithmetic_sum_S11_l251_25191


namespace NUMINAMATH_GPT_range_of_m_l251_25123

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^x 
else if 1 < x ∧ x ≤ 2 then Real.log (x - 1) 
else 0 -- function is not defined outside the given range

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 
  (x ≤ 1 → 2^x ≤ 4 - m * x) ∧ 
  (1 < x ∧ x ≤ 2 → Real.log (x - 1) ≤ 4 - m * x)) → 
  0 ≤ m ∧ m ≤ 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l251_25123


namespace NUMINAMATH_GPT_ratio_of_shares_l251_25197

theorem ratio_of_shares (A B C : ℝ) (x : ℝ):
  A = 240 → 
  A + B + C = 600 →
  A = x * (B + C) →
  B = (2/3) * (A + C) →
  A / (B + C) = 2 / 3 :=
by
  intros hA hTotal hFraction hB
  sorry

end NUMINAMATH_GPT_ratio_of_shares_l251_25197


namespace NUMINAMATH_GPT_find_x_when_y_is_sqrt_8_l251_25148

theorem find_x_when_y_is_sqrt_8
  (x y : ℝ)
  (h : ∀ x y : ℝ, (x^2 * y^4 = 1600) ↔ (x = 10 ∧ y = 2)) :
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_when_y_is_sqrt_8_l251_25148


namespace NUMINAMATH_GPT_max_y_coordinate_l251_25154

noncomputable def y_coordinate (θ : Real) : Real :=
  let u := Real.sin θ
  3 * u - 4 * u^3

theorem max_y_coordinate : ∃ θ, y_coordinate θ = 1 := by
  use Real.arcsin (1 / 2)
  sorry

end NUMINAMATH_GPT_max_y_coordinate_l251_25154


namespace NUMINAMATH_GPT_peaches_division_l251_25111

theorem peaches_division (n k r : ℕ) 
  (h₁ : 100 = n * k + 10)
  (h₂ : 1000 = n * k * 11 + r) :
  r = 10 :=
by sorry

end NUMINAMATH_GPT_peaches_division_l251_25111


namespace NUMINAMATH_GPT_worker_savings_multiple_l251_25158

theorem worker_savings_multiple 
  (P : ℝ)
  (P_gt_zero : P > 0)
  (save_fraction : ℝ := 1/3)
  (not_saved_fraction : ℝ := 2/3)
  (total_saved : ℝ := 12 * (save_fraction * P)) :
  ∃ multiple : ℝ, total_saved = multiple * (not_saved_fraction * P) ∧ multiple = 6 := 
by 
  sorry

end NUMINAMATH_GPT_worker_savings_multiple_l251_25158


namespace NUMINAMATH_GPT_number_of_people_got_off_at_third_stop_l251_25175

-- Definitions for each stop
def initial_passengers : ℕ := 0
def passengers_after_first_stop : ℕ := initial_passengers + 7
def passengers_after_second_stop : ℕ := passengers_after_first_stop - 3 + 5
def passengers_after_third_stop (x : ℕ) : ℕ := passengers_after_second_stop - x + 4

-- Final condition stating there are 11 passengers after the third stop
def final_passengers : ℕ := 11

-- Proof goal
theorem number_of_people_got_off_at_third_stop (x : ℕ) :
  passengers_after_third_stop x = final_passengers → x = 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_number_of_people_got_off_at_third_stop_l251_25175


namespace NUMINAMATH_GPT_infinite_equal_pairs_of_equal_terms_l251_25127

theorem infinite_equal_pairs_of_equal_terms {a : ℤ → ℤ}
  (h : ∀ n, a n = (a (n - 1) + a (n + 1)) / 4)
  (i j : ℤ) (hij : a i = a j) :
  ∃ (infinitely_many_pairs : ℕ → ℤ × ℤ), ∀ k, a (infinitely_many_pairs k).1 = a (infinitely_many_pairs k).2 :=
sorry

end NUMINAMATH_GPT_infinite_equal_pairs_of_equal_terms_l251_25127


namespace NUMINAMATH_GPT_paper_cut_square_l251_25110

noncomputable def proof_paper_cut_square : Prop :=
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ ((2 * x - 2 = 2 - x) ∨ (2 * (2 * x - 2) = 2 - x)) ∧ (x = 1.2 ∨ x = 1.5)

theorem paper_cut_square : proof_paper_cut_square :=
sorry

end NUMINAMATH_GPT_paper_cut_square_l251_25110


namespace NUMINAMATH_GPT_find_tangent_line_at_neg1_l251_25130

noncomputable def tangent_line (x : ℝ) : ℝ := 2 * x^2 + 3

theorem find_tangent_line_at_neg1 :
  let x := -1
  let m := 4 * x
  let y := 2 * x^2 + 3
  let tangent := y + m * (x - x)
  tangent = -4 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_tangent_line_at_neg1_l251_25130


namespace NUMINAMATH_GPT_find_number_l251_25167

theorem find_number : ∃ n : ℝ, 50 + (5 * n) / (180 / 3) = 51 ∧ n = 12 := 
by
  use 12
  sorry

end NUMINAMATH_GPT_find_number_l251_25167


namespace NUMINAMATH_GPT_ratio_of_distances_l251_25144

theorem ratio_of_distances 
  (x : ℝ) -- distance walked by the first lady
  (h1 : 4 + x = 12) -- combined total distance walked is 12 miles 
  (h2 : ¬(x < 0)) -- distance cannot be negative
  (h3 : 4 ≠ 0) : -- the second lady walked 4 miles which is not zero
  x / 4 = 2 := -- the ratio of the distances is 2
by
  sorry

end NUMINAMATH_GPT_ratio_of_distances_l251_25144


namespace NUMINAMATH_GPT_men_earnings_l251_25113

-- Definitions based on given problem conditions
variables (M rm W rw B rb X : ℝ)
variables (h1 : 5 > 0) (h2 : X > 0) (h3 : 8 > 0) -- positive quantities
variables (total_earnings : 5 * M * rm + X * W * rw + 8 * B * rb = 180)

-- The theorem we want to prove
theorem men_earnings (h1 : 5 > 0) (h2 : X > 0) (h3 : 8 > 0) (total_earnings : 5 * M * rm + X * W * rw + 8 * B * rb = 180) : 
  ∃ men_earnings : ℝ, men_earnings = 5 * M * rm :=
by 
  -- Proof is omitted
  exact Exists.intro (5 * M * rm) rfl

end NUMINAMATH_GPT_men_earnings_l251_25113


namespace NUMINAMATH_GPT_cricket_initial_overs_l251_25169

-- Definitions based on conditions
def run_rate_initial : ℝ := 3.2
def run_rate_remaining : ℝ := 12.5
def target_runs : ℝ := 282
def remaining_overs : ℕ := 20

-- Mathematical statement to prove
theorem cricket_initial_overs (x : ℝ) (y : ℝ)
    (h1 : y = run_rate_initial * x)
    (h2 : y + run_rate_remaining * remaining_overs = target_runs) :
    x = 10 :=
sorry

end NUMINAMATH_GPT_cricket_initial_overs_l251_25169


namespace NUMINAMATH_GPT_base10_to_base4_of_255_l251_25174

theorem base10_to_base4_of_255 :
  (255 : ℕ) = 3 * 4^3 + 3 * 4^2 + 3 * 4^1 + 3 * 4^0 :=
by
  sorry

end NUMINAMATH_GPT_base10_to_base4_of_255_l251_25174


namespace NUMINAMATH_GPT_ratio_of_ages_l251_25152

variable (F S : ℕ)

-- Condition 1: The product of father's age and son's age is 756
def cond1 := F * S = 756

-- Condition 2: The ratio of their ages after 6 years will be 2
def cond2 := (F + 6) / (S + 6) = 2

-- Theorem statement: The current ratio of the father's age to the son's age is 7:3
theorem ratio_of_ages (h1 : cond1 F S) (h2 : cond2 F S) : F / S = 7 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_ages_l251_25152


namespace NUMINAMATH_GPT_problem_statement_l251_25168

-- Define the function f1 as the square of the sum of the digits of k
def f1 (k : Nat) : Nat :=
  let sum_digits := (Nat.digits 10 k).sum
  sum_digits * sum_digits

-- Define the recursive function f_{n+1}(k) = f1(f_n(k))
def fn : Nat → Nat → Nat
| 0, k => k
| n+1, k => f1 (fn n k)

theorem problem_statement : fn 1991 (2^1990) = 256 :=
sorry

end NUMINAMATH_GPT_problem_statement_l251_25168


namespace NUMINAMATH_GPT_sum_of_quotient_and_reciprocal_l251_25105

theorem sum_of_quotient_and_reciprocal (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 500) : 
    (x / y + y / x) = 41 / 20 := 
sorry

end NUMINAMATH_GPT_sum_of_quotient_and_reciprocal_l251_25105


namespace NUMINAMATH_GPT_volume_correct_l251_25171

open Set Real

-- Define the conditions: the inequality and the constraints on x, y, z
def region (x y z : ℝ) : Prop :=
  abs (z + x + y) + abs (z + x - y) ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

-- Define the volume calculation
def volume_of_region : ℝ :=
  62.5

-- State the theorem
theorem volume_correct : ∀ (x y z : ℝ), region x y z → volume_of_region = 62.5 :=
by
  intro x y z h
  sorry

end NUMINAMATH_GPT_volume_correct_l251_25171


namespace NUMINAMATH_GPT_tank_leak_time_l251_25104

/--
The rate at which the tank is filled without a leak is R = 1/5 tank per hour.
The effective rate with the leak is 1/6 tank per hour.
Prove that the time it takes for the leak to empty the full tank is 30 hours.
-/
theorem tank_leak_time (R : ℝ) (L : ℝ) (h1 : R = 1 / 5) (h2 : R - L = 1 / 6) :
  1 / L = 30 :=
by
  sorry

end NUMINAMATH_GPT_tank_leak_time_l251_25104


namespace NUMINAMATH_GPT_calculation_result_l251_25126

theorem calculation_result:
  (-1:ℤ)^3 - 8 / (-2) + 4 * abs (-5) = 23 := by
  sorry

end NUMINAMATH_GPT_calculation_result_l251_25126


namespace NUMINAMATH_GPT_arithmetic_sequence_a11_l251_25138

theorem arithmetic_sequence_a11 (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 2) - a n = 6) : 
  a 11 = 31 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a11_l251_25138


namespace NUMINAMATH_GPT_original_count_l251_25187

-- Conditions
def original_count_eq (ping_pong_balls shuttlecocks : ℕ) : Prop :=
  ping_pong_balls = shuttlecocks

def removal_count (x : ℕ) : Prop :=
  5 * x - 3 * x = 16

-- Theorem to prove the original number of ping-pong balls and shuttlecocks
theorem original_count (ping_pong_balls shuttlecocks : ℕ) (x : ℕ) (h1 : original_count_eq ping_pong_balls shuttlecocks) (h2 : removal_count x) : ping_pong_balls = 40 ∧ shuttlecocks = 40 :=
  sorry

end NUMINAMATH_GPT_original_count_l251_25187


namespace NUMINAMATH_GPT_find_some_value_l251_25151

theorem find_some_value (m n : ℝ) (some_value : ℝ) 
  (h₁ : m = n / 2 - 2 / 5)
  (h₂ : m + 2 = (n + some_value) / 2 - 2 / 5) :
  some_value = 4 := 
sorry

end NUMINAMATH_GPT_find_some_value_l251_25151


namespace NUMINAMATH_GPT_distance_between_A_and_C_l251_25162

theorem distance_between_A_and_C :
  ∀ (AB BC CD AD AC : ℝ),
  AB = 3 → BC = 2 → CD = 5 → AD = 6 → AC = 1 := 
by
  intros AB BC CD AD AC hAB hBC hCD hAD
  have h1 : AD = AB + BC + CD := by sorry
  have h2 : 6 = 3 + 2 + AC := by sorry
  have h3 : 6 = 5 + AC := by sorry
  have h4 : AC = 1 := by sorry
  exact h4

end NUMINAMATH_GPT_distance_between_A_and_C_l251_25162


namespace NUMINAMATH_GPT_translation_of_exponential_l251_25118

noncomputable def translated_function (a : ℝ × ℝ) (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (x - a.1) + a.2

theorem translation_of_exponential :
  translated_function (2, 3) (λ x => Real.exp x) = λ x => Real.exp (x - 2) + 3 :=
by
  sorry

end NUMINAMATH_GPT_translation_of_exponential_l251_25118


namespace NUMINAMATH_GPT_inequality_1_inequality_2_inequality_4_l251_25194

theorem inequality_1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := sorry

theorem inequality_2 (a : ℝ) : a * (1 - a) ≤ 1 / 4 := sorry

theorem inequality_4 (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_inequality_4_l251_25194


namespace NUMINAMATH_GPT_largest_AB_under_conditions_l251_25166

def is_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem largest_AB_under_conditions :
  ∃ A B C D : ℕ, is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (A + B) % (C + D) = 0 ∧
    is_prime (A + B) ∧ is_prime (C + D) ∧
    (A + B) = 11 :=
sorry

end NUMINAMATH_GPT_largest_AB_under_conditions_l251_25166


namespace NUMINAMATH_GPT_vegan_non_soy_fraction_l251_25129

theorem vegan_non_soy_fraction (total_menu : ℕ) (vegan_dishes soy_free_vegan_dish : ℕ) 
  (h1 : vegan_dishes = 6) (h2 : vegan_dishes = total_menu / 3) (h3 : soy_free_vegan_dish = vegan_dishes - 5) :
  (soy_free_vegan_dish / total_menu = 1 / 18) :=
by
  sorry

end NUMINAMATH_GPT_vegan_non_soy_fraction_l251_25129


namespace NUMINAMATH_GPT_stickers_total_correct_l251_25156

-- Define the conditions
def stickers_per_page : ℕ := 10
def pages_total : ℕ := 22

-- Define the total number of stickers
def total_stickers : ℕ := pages_total * stickers_per_page

-- The statement we want to prove
theorem stickers_total_correct : total_stickers = 220 :=
by {
  sorry
}

end NUMINAMATH_GPT_stickers_total_correct_l251_25156


namespace NUMINAMATH_GPT_ratio_of_areas_l251_25106

-- Definition of sides and given condition
variables {a b c d : ℝ}
-- Given condition in the problem.
axiom condition : a / c = 3 / 5 ∧ b / d = 3 / 5

-- Statement of the theorem to be proved in Lean 4
theorem ratio_of_areas (h : a / c = 3 / 5) (h' : b / d = 3 / 5) : (a * b) / (c * d) = 9 / 25 :=
by sorry

end NUMINAMATH_GPT_ratio_of_areas_l251_25106


namespace NUMINAMATH_GPT_principal_amount_correct_l251_25136

noncomputable def initial_amount (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (A * 100) / (R * T + 100)

theorem principal_amount_correct : initial_amount 950 9.230769230769232 5 = 650 := by
  sorry

end NUMINAMATH_GPT_principal_amount_correct_l251_25136


namespace NUMINAMATH_GPT_find_M_l251_25185

theorem find_M (A M C : ℕ) (h1 : (100 * A + 10 * M + C) * (A + M + C) = 2040)
(h2 : (A + M + C) % 2 = 0)
(h3 : A ≤ 9) (h4 : M ≤ 9) (h5 : C ≤ 9) :
  M = 7 := 
sorry

end NUMINAMATH_GPT_find_M_l251_25185


namespace NUMINAMATH_GPT_calculate_expression_l251_25141

theorem calculate_expression :
  (-0.125) ^ 2009 * (8 : ℝ) ^ 2009 = -1 :=
sorry

end NUMINAMATH_GPT_calculate_expression_l251_25141


namespace NUMINAMATH_GPT_avg_age_increase_l251_25117

theorem avg_age_increase 
    (student_count : ℕ) (avg_student_age : ℕ) (teacher_age : ℕ) (new_count : ℕ) (new_avg_age : ℕ) (age_increase : ℕ)
    (hc1 : student_count = 23)
    (hc2 : avg_student_age = 22)
    (hc3 : teacher_age = 46)
    (hc4 : new_count = student_count + 1)
    (hc5 : new_avg_age = ((avg_student_age * student_count + teacher_age) / new_count))
    (hc6 : age_increase = new_avg_age - avg_student_age) :
  age_increase = 1 := 
sorry

end NUMINAMATH_GPT_avg_age_increase_l251_25117


namespace NUMINAMATH_GPT_or_is_true_given_p_true_q_false_l251_25140

theorem or_is_true_given_p_true_q_false (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q :=
by
  sorry

end NUMINAMATH_GPT_or_is_true_given_p_true_q_false_l251_25140


namespace NUMINAMATH_GPT_result_l251_25178

def problem : Float :=
  let sum := 78.652 + 24.3981
  let diff := sum - 0.025
  Float.round (diff * 100) / 100

theorem result :
  problem = 103.03 := by
  sorry

end NUMINAMATH_GPT_result_l251_25178


namespace NUMINAMATH_GPT_percentage_increase_in_expenses_l251_25145

theorem percentage_increase_in_expenses:
  ∀ (S : ℝ) (original_save_percentage new_savings : ℝ), 
  S = 5750 → 
  original_save_percentage = 0.20 →
  new_savings = 230 →
  (original_save_percentage * S - new_savings) / (S - original_save_percentage * S) * 100 = 20 :=
by
  intros S original_save_percentage new_savings HS Horiginal_save_percentage Hnew_savings
  rw [HS, Horiginal_save_percentage, Hnew_savings]
  sorry

end NUMINAMATH_GPT_percentage_increase_in_expenses_l251_25145


namespace NUMINAMATH_GPT_lisa_goal_l251_25132

theorem lisa_goal 
  (total_quizzes : ℕ) 
  (target_percentage : ℝ) 
  (completed_quizzes : ℕ) 
  (earned_A : ℕ) 
  (remaining_quizzes : ℕ) : 
  total_quizzes = 40 → 
  target_percentage = 0.9 → 
  completed_quizzes = 25 → 
  earned_A = 20 → 
  remaining_quizzes = (total_quizzes - completed_quizzes) → 
  (earned_A + remaining_quizzes ≥ target_percentage * total_quizzes) → 
  remaining_quizzes - (total_quizzes * target_percentage - earned_A) = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_lisa_goal_l251_25132


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l251_25198

variable (a x : ℝ)

-- Define the quadratic expression and the inequality condition
def quadratic_inequality (a x : ℝ) : Prop := 
  x^2 - (2 * a + 1) * x + a^2 + a < 0

-- Define the interval in which the inequality holds
def solution_set (a x : ℝ) : Prop :=
  a < x ∧ x < a + 1

-- The main statement to be proven
theorem quadratic_inequality_solution :
  ∀ a x, quadratic_inequality a x ↔ solution_set a x :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l251_25198


namespace NUMINAMATH_GPT_john_beats_per_minute_l251_25121

theorem john_beats_per_minute :
  let hours_per_day := 2
  let days := 3
  let total_beats := 72000
  let minutes_per_hour := 60
  total_beats / (days * hours_per_day * minutes_per_hour) = 200 := 
by 
  sorry

end NUMINAMATH_GPT_john_beats_per_minute_l251_25121


namespace NUMINAMATH_GPT_area_of_triangle_PDE_l251_25134

noncomputable def length (a b : Point) : ℝ := -- define length between two points
sorry

def distance_from_line (P D E : Point) : ℝ := -- define perpendicular distance from P to line DE
sorry

structure Point :=
(x : ℝ)
(y : ℝ)

def area_triangle (P D E : Point) : ℝ :=
0.5 -- define area given conditions

theorem area_of_triangle_PDE (D E : Point) (hD_E : D ≠ E) :
  { P : Point | area_triangle P D E = 0.5 } =
  { P : Point | distance_from_line P D E = 1 / (length D E) } :=
sorry

end NUMINAMATH_GPT_area_of_triangle_PDE_l251_25134


namespace NUMINAMATH_GPT_integer_distances_implies_vertex_l251_25159

theorem integer_distances_implies_vertex (M A B C D : ℝ × ℝ × ℝ)
  (a b c d : ℕ)
  (h_tetrahedron: 
    dist A B = 2 ∧ dist B C = 2 ∧ dist C D = 2 ∧ dist D A = 2 ∧ 
    dist A C = 2 ∧ dist B D = 2)
  (h_distances: 
    dist M A = a ∧ dist M B = b ∧ dist M C = c ∧ dist M D = d) :
  M = A ∨ M = B ∨ M = C ∨ M = D := 
  sorry

end NUMINAMATH_GPT_integer_distances_implies_vertex_l251_25159


namespace NUMINAMATH_GPT_group4_exceeds_group2_group4_exceeds_group3_l251_25116

-- Define conditions
def score_group1 : Int := 100
def score_group2 : Int := 150
def score_group3 : Int := -400
def score_group4 : Int := 350
def score_group5 : Int := -100

-- Theorem 1: Proving Group 4 exceeded Group 2 by 200 points
theorem group4_exceeds_group2 :
  score_group4 - score_group2 = 200 := by
  sorry

-- Theorem 2: Proving Group 4 exceeded Group 3 by 750 points
theorem group4_exceeds_group3 :
  score_group4 - score_group3 = 750 := by
  sorry

end NUMINAMATH_GPT_group4_exceeds_group2_group4_exceeds_group3_l251_25116


namespace NUMINAMATH_GPT_sum_of_fractions_limit_one_l251_25131

theorem sum_of_fractions_limit_one :
  (∑' (a : ℕ), ∑' (b : ℕ), (1 : ℝ) / ((a + 1) : ℝ) ^ (b + 1)) = 1 := 
sorry

end NUMINAMATH_GPT_sum_of_fractions_limit_one_l251_25131


namespace NUMINAMATH_GPT_g_eq_g_inv_solution_l251_25180

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5
noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem g_eq_g_inv_solution (x : ℝ) : g x = g_inv x ↔ x = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_g_eq_g_inv_solution_l251_25180


namespace NUMINAMATH_GPT_projection_non_ambiguity_l251_25160

theorem projection_non_ambiguity 
    (a b c : ℝ) 
    (theta : ℝ) 
    (h : a^2 = b^2 + c^2 - 2 * b * c * Real.cos theta) : 
    ∃ (c' : ℝ), c' = c * Real.cos theta ∧ a^2 = b^2 + c^2 + 2 * b * c' := 
sorry

end NUMINAMATH_GPT_projection_non_ambiguity_l251_25160


namespace NUMINAMATH_GPT_smallest_single_discount_more_advantageous_l251_25188

theorem smallest_single_discount_more_advantageous (n : ℕ) :
  (∀ n, 0 < n -> (1 - (n:ℝ)/100) < 0.64 ∧ (1 - (n:ℝ)/100) < 0.658503 ∧ (1 - (n:ℝ)/100) < 0.63) → 
  n = 38 := 
sorry

end NUMINAMATH_GPT_smallest_single_discount_more_advantageous_l251_25188


namespace NUMINAMATH_GPT_transform_sequence_zero_l251_25135

theorem transform_sequence_zero 
  (n : ℕ) 
  (a : ℕ → ℝ) 
  (h_nonempty : n > 0) :
  ∃ k : ℕ, k ≤ n ∧ ∀ k' ≤ k, ∃ α : ℝ, (∀ i, i < n → |a i - α| = 0) := 
sorry

end NUMINAMATH_GPT_transform_sequence_zero_l251_25135


namespace NUMINAMATH_GPT_smallest_n_for_sqrt_50n_is_integer_l251_25133

theorem smallest_n_for_sqrt_50n_is_integer :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (50 * n) = k * k) ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_sqrt_50n_is_integer_l251_25133


namespace NUMINAMATH_GPT_number_of_people_purchased_only_book_A_l251_25179

-- Definitions based on the conditions
variable (A B x y z w : ℕ)
variable (h1 : z = 500)
variable (h2 : z = 2 * y)
variable (h3 : w = z)
variable (h4 : x + y + z + w = 2500)
variable (h5 : A = x + z)
variable (h6 : B = y + z)
variable (h7 : A = 2 * B)

-- The statement we want to prove
theorem number_of_people_purchased_only_book_A :
  x = 1000 :=
by
  -- The proof steps will be filled here
  sorry

end NUMINAMATH_GPT_number_of_people_purchased_only_book_A_l251_25179


namespace NUMINAMATH_GPT_circles_intersect_l251_25112

def circle1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}
def circle2 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 4 * p.2 + 3 = 0}

theorem circles_intersect : ∃ (p : ℝ × ℝ), p ∈ circle1 ∧ p ∈ circle2 :=
by
  sorry

end NUMINAMATH_GPT_circles_intersect_l251_25112


namespace NUMINAMATH_GPT_solve_for_b_l251_25193

theorem solve_for_b (b : ℚ) (h : 2 * b + b / 4 = 5 / 2) : b = 10 / 9 :=
by sorry

end NUMINAMATH_GPT_solve_for_b_l251_25193


namespace NUMINAMATH_GPT_acute_triangle_probability_l251_25177

noncomputable def probability_acute_triangle : ℝ := sorry

theorem acute_triangle_probability :
  probability_acute_triangle = 1 / 4 := sorry

end NUMINAMATH_GPT_acute_triangle_probability_l251_25177


namespace NUMINAMATH_GPT_intersection_M_N_l251_25124

theorem intersection_M_N :
  let M := { x : ℝ | abs x ≤ 2 }
  let N := {-1, 0, 2, 3}
  M ∩ N = {-1, 0, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l251_25124


namespace NUMINAMATH_GPT_slope_of_l_l251_25176

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def parallel_lines (slope : ℝ) : Prop :=
  ∃ m : ℝ, ∀ x y : ℝ, y = slope * x + m

def intersects_ellipse (slope : ℝ) : Prop :=
  parallel_lines slope ∧ ∃ x y : ℝ, ellipse x y ∧ y = slope * x + (y - slope * x)

theorem slope_of_l {l_slope : ℝ} :
  (∃ (m : ℝ) (x y : ℝ), intersects_ellipse (1 / 4) ∧ (y - l_slope * x = m)) →
  (l_slope = -2) :=
sorry

end NUMINAMATH_GPT_slope_of_l_l251_25176


namespace NUMINAMATH_GPT_sum_ratio_arithmetic_sequence_l251_25170

theorem sum_ratio_arithmetic_sequence
  (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : ∀ n, S n = (n / 2) * (a 1 + a n))
  (h2 : ∀ k : ℕ, a (k + 1) - a k = a 2 - a 1)
  (h3 : a 4 / a 8 = 2 / 3) :
  S 7 / S 15 = 14 / 45 :=
sorry

end NUMINAMATH_GPT_sum_ratio_arithmetic_sequence_l251_25170


namespace NUMINAMATH_GPT_graphs_intersect_exactly_one_point_l251_25196

theorem graphs_intersect_exactly_one_point (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 5 * x + 4 = 2 * x - 6 → x = (7 / (2 * k))) ↔ k = (49 / 40) := 
by
  sorry

end NUMINAMATH_GPT_graphs_intersect_exactly_one_point_l251_25196


namespace NUMINAMATH_GPT_find_radius_of_sector_l251_25108

noncomputable def radius_of_sector (P : ℝ) (θ : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem find_radius_of_sector :
  radius_of_sector 144 180 = 144 / (Real.pi + 2) :=
by
  unfold radius_of_sector
  sorry

end NUMINAMATH_GPT_find_radius_of_sector_l251_25108


namespace NUMINAMATH_GPT_one_fourth_difference_l251_25109

theorem one_fourth_difference :
  (1 / 4) * ((9 * 5) - (7 + 3)) = 35 / 4 :=
by sorry

end NUMINAMATH_GPT_one_fourth_difference_l251_25109


namespace NUMINAMATH_GPT_range_of_k_for_intersecting_circles_l251_25186

/-- Given circle \( C \) with equation \( x^2 + y^2 - 8x + 15 = 0 \) and a line \( y = kx - 2 \),
    prove that if there exists at least one point on the line such that a circle with this point
    as the center and a radius of 1 intersects with circle \( C \), then \( 0 \leq k \leq \frac{4}{3} \). -/
theorem range_of_k_for_intersecting_circles (k : ℝ) :
  (∃ (x y : ℝ), y = k * x - 2 ∧ (x - 4) ^ 2 + y ^ 2 - 1 ≤ 1) → 0 ≤ k ∧ k ≤ 4 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_k_for_intersecting_circles_l251_25186


namespace NUMINAMATH_GPT_power_mod_equiv_l251_25139

theorem power_mod_equiv :
  2^1000 % 17 = 1 := by
  sorry

end NUMINAMATH_GPT_power_mod_equiv_l251_25139


namespace NUMINAMATH_GPT_smallest_b_value_minimizes_l251_25182

noncomputable def smallest_b_value (a b : ℝ) (c : ℝ := 2) : ℝ :=
  if (1 < a) ∧ (a < b) ∧ (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) ∧ (¬ (1/b + 1/a > c ∧ 1/a + c > 1/b ∧ c + 1/b > 1/a)) then b else 0

theorem smallest_b_value_minimizes (a b : ℝ) (c : ℝ := 2) :
  (1 < a) ∧ (a < b) ∧ (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) ∧ (¬ (1/b + 1/a > c ∧ 1/a + c > 1/b ∧ c + 1/b > 1/a)) →
  b = 2 :=
by sorry

end NUMINAMATH_GPT_smallest_b_value_minimizes_l251_25182


namespace NUMINAMATH_GPT_tangent_line_slope_through_origin_l251_25128

theorem tangent_line_slope_through_origin :
  (∃ a : ℝ, (a^3 + a + 16 = (3 * a^2 + 1) * a ∧ a = 2)) →
  (3 * (2 : ℝ)^2 + 1 = 13) :=
by
  intro h
  -- Detailed proof goes here
  sorry

end NUMINAMATH_GPT_tangent_line_slope_through_origin_l251_25128


namespace NUMINAMATH_GPT_ring_cost_l251_25137

theorem ring_cost (total_cost : ℕ) (rings : ℕ) (h1 : total_cost = 24) (h2 : rings = 2) : total_cost / rings = 12 :=
by
  sorry

end NUMINAMATH_GPT_ring_cost_l251_25137
