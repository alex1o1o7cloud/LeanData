import Mathlib

namespace NUMINAMATH_GPT_limit_a_n_l1013_101309

open Nat Real

noncomputable def a_n (n : ℕ) : ℝ := (7 * n - 1) / (n + 1)

theorem limit_a_n : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - 7| < ε := 
by {
  -- The proof would go here.
  sorry
}

end NUMINAMATH_GPT_limit_a_n_l1013_101309


namespace NUMINAMATH_GPT_div_pow_eq_l1013_101399

theorem div_pow_eq : 23^11 / 23^5 = 148035889 := by
  sorry

end NUMINAMATH_GPT_div_pow_eq_l1013_101399


namespace NUMINAMATH_GPT_rectangle_image_l1013_101357

-- A mathematically equivalent Lean 4 proof problem statement

variable (x y : ℝ)

def rectangle_OABC (x y : ℝ) : Prop :=
  (x = 0 ∧ (0 ≤ y ∧ y ≤ 3)) ∨
  (y = 0 ∧ (0 ≤ x ∧ x ≤ 2)) ∨
  (x = 2 ∧ (0 ≤ y ∧ y ≤ 3)) ∨
  (y = 3 ∧ (0 ≤ x ∧ x ≤ 2))

def transform_u (x y : ℝ) : ℝ := x^2 - y^2 + 1
def transform_v (x y : ℝ) : ℝ := x * y

theorem rectangle_image (u v : ℝ) :
  (∃ (x y : ℝ), rectangle_OABC x y ∧ u = transform_u x y ∧ v = transform_v x y) ↔
  (u, v) = (-8, 0) ∨
  (u, v) = (1, 0) ∨
  (u, v) = (5, 0) ∨
  (u, v) = (-4, 6) :=
sorry

end NUMINAMATH_GPT_rectangle_image_l1013_101357


namespace NUMINAMATH_GPT_bisection_interval_length_l1013_101368

theorem bisection_interval_length (n : ℕ) : 
  (1 / (2:ℝ)^n) ≤ 0.01 → n ≥ 7 :=
by 
  sorry

end NUMINAMATH_GPT_bisection_interval_length_l1013_101368


namespace NUMINAMATH_GPT_correct_conclusions_l1013_101398

theorem correct_conclusions :
  (∀ n : ℤ, n < -1 -> n < -1) ∧
  (¬ ∀ a : ℤ, abs (a + 2022) > 0) ∧
  (∀ a b : ℤ, a + b = 0 -> a * b < 0) ∧
  (∀ n : ℤ, abs n = n -> n ≥ 0) :=
sorry

end NUMINAMATH_GPT_correct_conclusions_l1013_101398


namespace NUMINAMATH_GPT_karen_locks_l1013_101311

theorem karen_locks : 
  let L1 := 5
  let L2 := 3 * L1 - 3
  let Lboth := 5 * L2
  Lboth = 60 :=
by
  let L1 := 5
  let L2 := 3 * L1 - 3
  let Lboth := 5 * L2
  sorry

end NUMINAMATH_GPT_karen_locks_l1013_101311


namespace NUMINAMATH_GPT_min_value_of_expression_l1013_101353

-- positive real numbers a and b
variables (a b : ℝ)
variables (ha : 0 < a) (hb : 0 < b)
-- given condition: 1/a + 9/b = 6
variable (h : 1 / a + 9 / b = 6)

theorem min_value_of_expression : (a + 1) * (b + 9) ≥ 16 := by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1013_101353


namespace NUMINAMATH_GPT_difference_longest_shortest_worm_l1013_101379

theorem difference_longest_shortest_worm
  (A B C D E : ℝ)
  (hA : A = 0.8)
  (hB : B = 0.1)
  (hC : C = 1.2)
  (hD : D = 0.4)
  (hE : E = 0.7) :
  (max C (max A (max E (max D B))) - min B (min D (min E (min A C)))) = 1.1 :=
by
  sorry

end NUMINAMATH_GPT_difference_longest_shortest_worm_l1013_101379


namespace NUMINAMATH_GPT_factorization_result_l1013_101342

theorem factorization_result (a b : ℤ) (h1 : 25 * x^2 - 160 * x - 336 = (5 * x + a) * (5 * x + b)) :
  a + 2 * b = 20 :=
by
  sorry

end NUMINAMATH_GPT_factorization_result_l1013_101342


namespace NUMINAMATH_GPT_initial_population_l1013_101382

theorem initial_population (P : ℝ)
  (h1 : P * 1.25 * 0.75 = 18750) : P = 20000 :=
sorry

end NUMINAMATH_GPT_initial_population_l1013_101382


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1013_101356

-- Definitions representing the conditions
def setA : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def setB : Set ℝ := {x | x < 2}

-- Proof problem statement
theorem intersection_of_A_and_B : setA ∩ setB = {x | -1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1013_101356


namespace NUMINAMATH_GPT_solve_x_l1013_101301

theorem solve_x : ∃ x : ℝ, 65 + (5 * x) / (180 / 3) = 66 ∧ x = 12 := by
  sorry

end NUMINAMATH_GPT_solve_x_l1013_101301


namespace NUMINAMATH_GPT_piglet_weight_l1013_101336

variable (C K P L : ℝ)

theorem piglet_weight (h1 : C = K + P) (h2 : P + C = L + K) (h3 : L = 30) : P = 15 := by
  sorry

end NUMINAMATH_GPT_piglet_weight_l1013_101336


namespace NUMINAMATH_GPT_initial_cake_pieces_l1013_101321

-- Define the initial number of cake pieces
variable (X : ℝ)

-- Define the conditions as assumptions
def cake_conditions (X : ℝ) : Prop :=
  0.60 * X + 3 * 32 = X 

theorem initial_cake_pieces (X : ℝ) (h : cake_conditions X) : X = 240 := sorry

end NUMINAMATH_GPT_initial_cake_pieces_l1013_101321


namespace NUMINAMATH_GPT_last_digit_of_7_to_the_7_l1013_101387

theorem last_digit_of_7_to_the_7 :
  (7 ^ 7) % 10 = 3 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_of_7_to_the_7_l1013_101387


namespace NUMINAMATH_GPT_at_least_one_nonnegative_l1013_101361

theorem at_least_one_nonnegative
  (a1 a2 a3 a4 a5 a6 a7 a8 : ℝ)
  (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (h3 : a3 ≠ 0) (h4 : a4 ≠ 0)
  (h5 : a5 ≠ 0) (h6 : a6 ≠ 0) (h7 : a7 ≠ 0) (h8 : a8 ≠ 0)
  : (a1 * a3 + a2 * a4 ≥ 0) ∨ (a1 * a5 + a2 * a6 ≥ 0) ∨ (a1 * a7 + a2 * a8 ≥ 0) ∨
    (a3 * a5 + a4 * a6 ≥ 0) ∨ (a3 * a7 + a4 * a8 ≥ 0) ∨ (a5 * a7 + a6 * a8 ≥ 0) := 
sorry

end NUMINAMATH_GPT_at_least_one_nonnegative_l1013_101361


namespace NUMINAMATH_GPT_number_of_meetings_l1013_101366

-- Definitions based on the given conditions
def track_circumference : ℕ := 300
def boy1_speed : ℕ := 7
def boy2_speed : ℕ := 3
def both_start_simultaneously := true

-- The theorem to prove
theorem number_of_meetings (h1 : track_circumference = 300) (h2 : boy1_speed = 7) (h3 : boy2_speed = 3) (h4 : both_start_simultaneously) : 
  ∃ n : ℕ, n = 1 := 
sorry

end NUMINAMATH_GPT_number_of_meetings_l1013_101366


namespace NUMINAMATH_GPT_race_distance_l1013_101347

theorem race_distance (D : ℝ) (h1 : (D / 36) * 45 = D + 20) : D = 80 :=
by
  sorry

end NUMINAMATH_GPT_race_distance_l1013_101347


namespace NUMINAMATH_GPT_license_plate_count_correct_l1013_101375

def rotokas_letters : Finset Char := {'A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U'}

def valid_license_plate_count : ℕ :=
  let first_letter_choices := 2 -- Letters A or E
  let last_letter_fixed := 1 -- Fixed as P
  let remaining_letters := rotokas_letters.erase 'V' -- Exclude V
  let second_letter_choices := (remaining_letters.erase 'P').card - 1 -- Exclude P and first letter
  let third_letter_choices := second_letter_choices - 1
  let fourth_letter_choices := third_letter_choices - 1
  2 * 9 * 8 * 7

theorem license_plate_count_correct :
  valid_license_plate_count = 1008 := by
  sorry

end NUMINAMATH_GPT_license_plate_count_correct_l1013_101375


namespace NUMINAMATH_GPT_floor_ineq_l1013_101302

theorem floor_ineq (α β : ℝ) : ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ :=
sorry

end NUMINAMATH_GPT_floor_ineq_l1013_101302


namespace NUMINAMATH_GPT_tangency_splits_segments_l1013_101385

def pentagon_lengths (a b c d e : ℕ) (h₁ : a = 1) (h₃ : c = 1) (x1 x2 : ℝ) :=
x1 + x2 = b ∧ x1 = 1/2 ∧ x2 = 1/2

theorem tangency_splits_segments {a b c d e : ℕ} (h₁ : a = 1) (h₃ : c = 1) :
    ∃ x1 x2 : ℝ, pentagon_lengths a b c d e h₁ h₃ x1 x2 :=
    by 
    sorry

end NUMINAMATH_GPT_tangency_splits_segments_l1013_101385


namespace NUMINAMATH_GPT_original_number_is_correct_l1013_101313

noncomputable def original_number : ℝ :=
  let x := 11.26666666666667
  let y := 30.333333333333332
  x + y

theorem original_number_is_correct (x y : ℝ) (h₁ : 10 * x + 22 * y = 780) (h₂ : y = 30.333333333333332) : 
  original_number = 41.6 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_correct_l1013_101313


namespace NUMINAMATH_GPT_quadratic_root_k_value_l1013_101360

theorem quadratic_root_k_value 
  (k : ℝ) 
  (h_roots : ∀ x : ℝ, (5 * x^2 + 7 * x + k = 0) → (x = ( -7 + Real.sqrt (-191) ) / 10 ∨ x = ( -7 - Real.sqrt (-191) ) / 10)) : 
  k = 12 :=
sorry

end NUMINAMATH_GPT_quadratic_root_k_value_l1013_101360


namespace NUMINAMATH_GPT_sum_of_transformed_numbers_l1013_101303

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_transformed_numbers_l1013_101303


namespace NUMINAMATH_GPT_grandmother_age_five_times_lingling_l1013_101319

theorem grandmother_age_five_times_lingling (x : ℕ) :
  let lingling_age := 8
  let grandmother_age := 60
  (grandmother_age + x = 5 * (lingling_age + x)) ↔ (x = 5) := by
  sorry

end NUMINAMATH_GPT_grandmother_age_five_times_lingling_l1013_101319


namespace NUMINAMATH_GPT_overall_gain_percent_l1013_101369

theorem overall_gain_percent (cp1 cp2 cp3: ℝ) (sp1 sp2 sp3: ℝ) (h1: cp1 = 840) (h2: cp2 = 1350) (h3: cp3 = 2250) (h4: sp1 = 1220) (h5: sp2 = 1550) (h6: sp3 = 2150) : 
  (sp1 + sp2 + sp3 - (cp1 + cp2 + cp3)) / (cp1 + cp2 + cp3) * 100 = 10.81 := 
by 
  sorry

end NUMINAMATH_GPT_overall_gain_percent_l1013_101369


namespace NUMINAMATH_GPT_a_4_is_zero_l1013_101346

def a_n (n : ℕ) : ℕ := n^2 - 2*n - 8

theorem a_4_is_zero : a_n 4 = 0 := 
by
  sorry

end NUMINAMATH_GPT_a_4_is_zero_l1013_101346


namespace NUMINAMATH_GPT_find_x_from_roots_l1013_101344

variable (x m : ℕ)

theorem find_x_from_roots (h1 : (m + 3)^2 = x) (h2 : (2 * m - 15)^2 = x) : x = 49 := by
  sorry

end NUMINAMATH_GPT_find_x_from_roots_l1013_101344


namespace NUMINAMATH_GPT_prop_disjunction_is_true_l1013_101367

variable (p q : Prop)
axiom hp : p
axiom hq : ¬q

theorem prop_disjunction_is_true (hp : p) (hq : ¬q) : p ∨ q :=
by
  sorry

end NUMINAMATH_GPT_prop_disjunction_is_true_l1013_101367


namespace NUMINAMATH_GPT_Pam_current_balance_l1013_101351

-- Given conditions as definitions
def initial_balance : ℕ := 400
def tripled_balance : ℕ := 3 * initial_balance
def current_balance : ℕ := tripled_balance - 250

-- The theorem to be proved
theorem Pam_current_balance : current_balance = 950 := by
  sorry

end NUMINAMATH_GPT_Pam_current_balance_l1013_101351


namespace NUMINAMATH_GPT_exists_unique_i_l1013_101393

theorem exists_unique_i (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) 
  (a : ℤ) (ha1 : 2 ≤ a) (ha2 : a ≤ p - 2) : 
  ∃! (i : ℤ), 2 ≤ i ∧ i ≤ p - 2 ∧ (i * a) % p = 1 ∧ Nat.gcd (i.natAbs) (a.natAbs) = 1 :=
sorry

end NUMINAMATH_GPT_exists_unique_i_l1013_101393


namespace NUMINAMATH_GPT_quadratic_root_value_of_b_l1013_101329

theorem quadratic_root_value_of_b :
  (∃ r1 r2 : ℝ, 2 * r1^2 + b * r1 - 20 = 0 ∧ r1 = -5 ∧ r1 * r2 = -10 ∧ r1 + r2 = -b / 2) → b = 6 :=
by
  intro h
  obtain ⟨r1, r2, h_eq1, h_r1, h_prod, h_sum⟩ := h
  sorry

end NUMINAMATH_GPT_quadratic_root_value_of_b_l1013_101329


namespace NUMINAMATH_GPT_coprime_divisibility_l1013_101324

theorem coprime_divisibility (p q r P Q R : ℕ)
  (hpq : Nat.gcd p q = 1) (hpr : Nat.gcd p r = 1) (hqr : Nat.gcd q r = 1)
  (h : ∃ k : ℤ, (P:ℤ) * (q*r) + (Q:ℤ) * (p*r) + (R:ℤ) * (p*q) = k * (p*q * r)) :
  ∃ a b c : ℤ, (P:ℤ) = a * (p:ℤ) ∧ (Q:ℤ) = b * (q:ℤ) ∧ (R:ℤ) = c * (r:ℤ) :=
by
  sorry

end NUMINAMATH_GPT_coprime_divisibility_l1013_101324


namespace NUMINAMATH_GPT_savings_per_egg_l1013_101350

def price_per_organic_egg : ℕ := 50 
def cost_of_tray : ℕ := 1200 -- in cents
def number_of_eggs_in_tray : ℕ := 30

theorem savings_per_egg : 
  price_per_organic_egg - (cost_of_tray / number_of_eggs_in_tray) = 10 := 
by
  sorry

end NUMINAMATH_GPT_savings_per_egg_l1013_101350


namespace NUMINAMATH_GPT_two_digit_numbers_equal_three_times_product_of_digits_l1013_101372

theorem two_digit_numbers_equal_three_times_product_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 3 * a * b} = {15, 24} :=
by
  sorry

end NUMINAMATH_GPT_two_digit_numbers_equal_three_times_product_of_digits_l1013_101372


namespace NUMINAMATH_GPT_hat_cost_l1013_101327

noncomputable def cost_of_hat (H : ℕ) : Prop :=
  let cost_shirts := 3 * 5
  let cost_jeans := 2 * 10
  let cost_hats := 4 * H
  let total_cost := 51
  cost_shirts + cost_jeans + cost_hats = total_cost

theorem hat_cost : ∃ H : ℕ, cost_of_hat H ∧ H = 4 :=
by 
  sorry

end NUMINAMATH_GPT_hat_cost_l1013_101327


namespace NUMINAMATH_GPT_number_tower_proof_l1013_101363

theorem number_tower_proof : 123456 * 9 + 7 = 1111111 := 
  sorry

end NUMINAMATH_GPT_number_tower_proof_l1013_101363


namespace NUMINAMATH_GPT_moles_of_Cl2_l1013_101395

def chemical_reaction : Prop :=
  ∀ (CH4 Cl2 HCl : ℕ), 
  (CH4 = 1) → 
  (HCl = 4) →
  -- Given the balanced equation: CH4 + 2Cl2 → CHCl3 + 4HCl
  (CH4 + 2 * Cl2 = CH4 + 2 * Cl2) →
  (4 * HCl = 4 * HCl) → -- This asserts the product side according to the balanced equation
  (Cl2 = 2)

theorem moles_of_Cl2 (CH4 Cl2 HCl : ℕ) (hCH4 : CH4 = 1) (hHCl : HCl = 4)
  (h_balanced : CH4 + 2 * Cl2 = CH4 + 2 * Cl2) (h_product : 4 * HCl = 4 * HCl) :
  Cl2 = 2 := by {
    sorry
}

end NUMINAMATH_GPT_moles_of_Cl2_l1013_101395


namespace NUMINAMATH_GPT_walter_hushpuppies_per_guest_l1013_101397

variables (guests hushpuppies_per_batch time_per_batch total_time : ℕ)

def batches (total_time time_per_batch : ℕ) : ℕ :=
  total_time / time_per_batch

def total_hushpuppies (batches hushpuppies_per_batch : ℕ) : ℕ :=
  batches * hushpuppies_per_batch

def hushpuppies_per_guest (total_hushpuppies guests : ℕ) : ℕ :=
  total_hushpuppies / guests

theorem walter_hushpuppies_per_guest :
  ∀ (guests hushpuppies_per_batch time_per_batch total_time : ℕ),
    guests = 20 →
    hushpuppies_per_batch = 10 →
    time_per_batch = 8 →
    total_time = 80 →
    hushpuppies_per_guest (total_hushpuppies (batches total_time time_per_batch) hushpuppies_per_batch) guests = 5 :=
by 
  intros _ _ _ _ h_guests h_hpb h_tpb h_tt
  sorry

end NUMINAMATH_GPT_walter_hushpuppies_per_guest_l1013_101397


namespace NUMINAMATH_GPT_square_garden_perimeter_l1013_101374

theorem square_garden_perimeter (A : ℝ) (h : A = 450) : ∃ P : ℝ, P = 60 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_square_garden_perimeter_l1013_101374


namespace NUMINAMATH_GPT_tape_pieces_needed_l1013_101307

-- Define the setup: cube edge length and tape width
def edge_length (n : ℕ) : ℕ := n
def tape_width : ℕ := 1

-- Define the statement we want to prove
theorem tape_pieces_needed (n : ℕ) (h₁ : edge_length n > 0) : 2 * n = 2 * (edge_length n) :=
  by
  sorry

end NUMINAMATH_GPT_tape_pieces_needed_l1013_101307


namespace NUMINAMATH_GPT_sqrt_diff_inequality_l1013_101391

open Real

theorem sqrt_diff_inequality (a : ℝ) (h : a ≥ 3) : 
  sqrt a - sqrt (a - 1) < sqrt (a - 2) - sqrt (a - 3) :=
sorry

end NUMINAMATH_GPT_sqrt_diff_inequality_l1013_101391


namespace NUMINAMATH_GPT_cone_base_radius_l1013_101389

theorem cone_base_radius (angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) 
(h1 : angle = 216)
(h2 : sector_radius = 15)
(h3 : 2 * π * base_radius = (3 / 5) * 2 * π * sector_radius) :
base_radius = 9 := 
sorry

end NUMINAMATH_GPT_cone_base_radius_l1013_101389


namespace NUMINAMATH_GPT_negation_of_proposition_l1013_101352

-- Given condition
def original_statement (a : ℝ) : Prop :=
  ∃ x : ℝ, a*x^2 - 2*a*x + 1 ≤ 0

-- Correct answer (negation statement)
def negated_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - 2*a*x + 1 > 0

-- Statement to prove
theorem negation_of_proposition (a : ℝ) :
  ¬ (original_statement a) ↔ (negated_statement a) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1013_101352


namespace NUMINAMATH_GPT_probability_sum_of_10_l1013_101323

theorem probability_sum_of_10 (total_outcomes : ℕ) 
  (h1 : total_outcomes = 6^4) : 
  (46 / total_outcomes) = 23 / 648 := by
  sorry

end NUMINAMATH_GPT_probability_sum_of_10_l1013_101323


namespace NUMINAMATH_GPT_find_fraction_l1013_101383

noncomputable def distinct_real_numbers (a b : ℝ) : Prop :=
  a ≠ b

noncomputable def equation_condition (a b : ℝ) : Prop :=
  (2 * a / (3 * b)) + ((a + 12 * b) / (3 * b + 12 * a)) = (5 / 3)

theorem find_fraction (a b : ℝ) (h1 : distinct_real_numbers a b) (h2 : equation_condition a b) : a / b = -93 / 49 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l1013_101383


namespace NUMINAMATH_GPT_intersection_AB_l1013_101343

def setA : Set ℝ := { x | x^2 - 2*x - 3 < 0}
def setB : Set ℝ := { x | x > 1 }
def intersection : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem intersection_AB : setA ∩ setB = intersection :=
by
  sorry

end NUMINAMATH_GPT_intersection_AB_l1013_101343


namespace NUMINAMATH_GPT_added_number_is_6_l1013_101325

theorem added_number_is_6 : ∃ x : ℤ, (∃ y : ℤ, y = 9 ∧ (2 * y + x) * 3 = 72) → x = 6 := 
by
  sorry

end NUMINAMATH_GPT_added_number_is_6_l1013_101325


namespace NUMINAMATH_GPT_avgPercentageSpentOnFoodCorrect_l1013_101333

-- Definitions for given conditions
def JanuaryIncome : ℕ := 3000
def JanuaryPetrolExpenditure : ℕ := 300
def JanuaryHouseRentPercentage : ℕ := 14
def JanuaryClothingPercentage : ℕ := 10
def JanuaryUtilityBillsPercentage : ℕ := 5
def FebruaryIncome : ℕ := 4000
def FebruaryPetrolExpenditure : ℕ := 400
def FebruaryHouseRentPercentage : ℕ := 14
def FebruaryClothingPercentage : ℕ := 10
def FebruaryUtilityBillsPercentage : ℕ := 5

-- Calculate percentage spent on food over January and February
noncomputable def avgPercentageSpentOnFood : ℝ :=
  let totalIncome := (JanuaryIncome + FebruaryIncome: ℝ)
  let totalFoodExpenditure :=
    let remainingJan := (JanuaryIncome - JanuaryPetrolExpenditure: ℝ) 
                         - (JanuaryHouseRentPercentage / 100 * (JanuaryIncome - JanuaryPetrolExpenditure: ℝ))
                         - (JanuaryClothingPercentage / 100 * JanuaryIncome)
                         - (JanuaryUtilityBillsPercentage / 100 * JanuaryIncome)
    let remainingFeb := (FebruaryIncome - FebruaryPetrolExpenditure: ℝ)
                         - (FebruaryHouseRentPercentage / 100 * (FebruaryIncome - FebruaryPetrolExpenditure: ℝ))
                         - (FebruaryClothingPercentage / 100 * FebruaryIncome)
                         - (FebruaryUtilityBillsPercentage / 100 * FebruaryIncome)
    remainingJan + remainingFeb
  (totalFoodExpenditure / totalIncome) * 100

theorem avgPercentageSpentOnFoodCorrect : avgPercentageSpentOnFood = 62.4 := by
  sorry

end NUMINAMATH_GPT_avgPercentageSpentOnFoodCorrect_l1013_101333


namespace NUMINAMATH_GPT_total_customers_in_line_l1013_101337

-- Definition of the number of people standing in front of the last person
def num_people_in_front : Nat := 8

-- Definition of the last person in the line
def last_person : Nat := 1

-- Statement to prove
theorem total_customers_in_line : num_people_in_front + last_person = 9 := by
  sorry

end NUMINAMATH_GPT_total_customers_in_line_l1013_101337


namespace NUMINAMATH_GPT_af_cd_ratio_l1013_101378

theorem af_cd_ratio (a b c d e f : ℝ) 
  (h1 : a * b * c = 130) 
  (h2 : b * c * d = 65) 
  (h3 : c * d * e = 750) 
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_af_cd_ratio_l1013_101378


namespace NUMINAMATH_GPT_find_a_l1013_101371

theorem find_a 
  (a b c : ℤ) 
  (h_vertex : ∀ x, (a * (x - 2)^2 + 5 = a * x^2 + b * x + c))
  (h_point : ∀ y, y = a * (1 - 2)^2 + 5)
  : a = -1 := by
  sorry

end NUMINAMATH_GPT_find_a_l1013_101371


namespace NUMINAMATH_GPT_find_m_l1013_101326

-- Definition of vectors in terms of the condition
def vec_a (m : ℝ) : ℝ × ℝ := (2 * m + 1, m)
def vec_b (m : ℝ) : ℝ × ℝ := (1, m)

-- Condition that vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2) = 0

-- Problem statement: find m such that vec_a is perpendicular to vec_b
theorem find_m (m : ℝ) (h : perpendicular (vec_a m) (vec_b m)) : m = -1 := by
  sorry

end NUMINAMATH_GPT_find_m_l1013_101326


namespace NUMINAMATH_GPT_gh_of_2_l1013_101317

def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem gh_of_2 :
  g (h 2) = 3269 :=
by
  sorry

end NUMINAMATH_GPT_gh_of_2_l1013_101317


namespace NUMINAMATH_GPT_solve_for_x_l1013_101388

theorem solve_for_x (x : ℤ) (h : 3 * x - 5 = 4 * x + 10) : x = -15 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1013_101388


namespace NUMINAMATH_GPT_solve_triples_l1013_101314

theorem solve_triples (a b c n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n) :
  a^2 + b^2 = n * Nat.lcm a b + n^2 ∧
  b^2 + c^2 = n * Nat.lcm b c + n^2 ∧
  c^2 + a^2 = n * Nat.lcm c a + n^2 →
  ∃ k : ℕ, 0 < k ∧ a = k ∧ b = k ∧ c = k :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_triples_l1013_101314


namespace NUMINAMATH_GPT_asymptote_of_hyperbola_l1013_101316

theorem asymptote_of_hyperbola :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) → y = x / 2 ∨ y = - x / 2 :=
sorry

end NUMINAMATH_GPT_asymptote_of_hyperbola_l1013_101316


namespace NUMINAMATH_GPT_max_three_topping_pizzas_l1013_101358

-- Define the combinations function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Assert the condition and the question with the expected answer
theorem max_three_topping_pizzas : combination 8 3 = 56 :=
by
  sorry

end NUMINAMATH_GPT_max_three_topping_pizzas_l1013_101358


namespace NUMINAMATH_GPT_var_power_l1013_101349

theorem var_power {a b c x y z : ℝ} (h1 : x = a * y^4) (h2 : y = b * z^(1/3)) :
  ∃ n : ℝ, x = c * z^n ∧ n = 4/3 := by
  sorry

end NUMINAMATH_GPT_var_power_l1013_101349


namespace NUMINAMATH_GPT_continuous_function_identity_l1013_101345

theorem continuous_function_identity (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_func_eq : ∀ x y : ℝ, 2 * f (x + y) = f x * f y)
  (h_f1 : f 1 = 10) :
  ∀ x : ℝ, f x = 2 * 5^x :=
by
  sorry

end NUMINAMATH_GPT_continuous_function_identity_l1013_101345


namespace NUMINAMATH_GPT_probability_inequality_up_to_99_l1013_101338

theorem probability_inequality_up_to_99 :
  (∀ x : ℕ, 1 ≤ x ∧ x < 100 → (2^x / x!) > x^2) →
    (∃ n : ℕ, (1 ≤ n ∧ n < 100) ∧ (2^n / n!) > n^2) →
      ∃ p : ℚ, p = 1/99 :=
by
  sorry

end NUMINAMATH_GPT_probability_inequality_up_to_99_l1013_101338


namespace NUMINAMATH_GPT_greatest_of_six_consecutive_mixed_numbers_l1013_101392

theorem greatest_of_six_consecutive_mixed_numbers (A : ℚ) :
  let B := A + 1
  let C := A + 2
  let D := A + 3
  let E := A + 4
  let F := A + 5
  (A + B + C + D + E + F = 75.5) →
  F = 15 + 1/12 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_of_six_consecutive_mixed_numbers_l1013_101392


namespace NUMINAMATH_GPT_no_single_two_three_digit_solution_l1013_101364

theorem no_single_two_three_digit_solution :
  ¬ ∃ (x y z : ℕ),
    (1 ≤ x ∧ x ≤ 9) ∧
    (10 ≤ y ∧ y ≤ 99) ∧
    (100 ≤ z ∧ z ≤ 999) ∧
    (1/x : ℝ) = 1/y + 1/z :=
by
  sorry

end NUMINAMATH_GPT_no_single_two_three_digit_solution_l1013_101364


namespace NUMINAMATH_GPT_loan_percentage_correct_l1013_101354

-- Define the parameters and conditions of the problem
def house_initial_value : ℕ := 100000
def house_increase_percentage : ℝ := 0.25
def new_house_cost : ℕ := 500000
def loan_percentage : ℝ := 75.0

-- Define the theorem we want to prove
theorem loan_percentage_correct :
  let increase_value := house_initial_value * house_increase_percentage
  let sale_price := house_initial_value + increase_value
  let loan_amount := new_house_cost - sale_price
  let loan_percentage_computed := (loan_amount / new_house_cost) * 100
  loan_percentage_computed = loan_percentage :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_loan_percentage_correct_l1013_101354


namespace NUMINAMATH_GPT_part1_part2_l1013_101310

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.exp (2 * x) - a * Real.exp x - x * Real.exp x

theorem part1 :
  (∀ x : ℝ, f a x ≥ 0) → a = 1 := sorry

theorem part2 (h : a = 1) :
  ∃ x₀ : ℝ, (∀ x : ℝ, f a x ≤ f a x₀) ∧
    (Real.log 2 / (2 * Real.exp 1) + 1 / (4 * Real.exp (2 * 1)) ≤ f a x₀ ∧
    f a x₀ < 1 / 4) := sorry

end NUMINAMATH_GPT_part1_part2_l1013_101310


namespace NUMINAMATH_GPT_certain_number_l1013_101394

theorem certain_number (G : ℕ) (N : ℕ) (H1 : G = 129) 
  (H2 : N % G = 9) (H3 : 2206 % G = 13) : N = 2202 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_l1013_101394


namespace NUMINAMATH_GPT_modulo_17_residue_l1013_101318

theorem modulo_17_residue : (392 + 6 * 51 + 8 * 221 + 3^2 * 23) % 17 = 11 :=
by 
  sorry

end NUMINAMATH_GPT_modulo_17_residue_l1013_101318


namespace NUMINAMATH_GPT_least_number_to_make_divisible_l1013_101381

theorem least_number_to_make_divisible (k : ℕ) (h : 1202 + k = 1204) : (2 ∣ 1204) := 
by
  sorry

end NUMINAMATH_GPT_least_number_to_make_divisible_l1013_101381


namespace NUMINAMATH_GPT_annie_blocks_walked_l1013_101339

theorem annie_blocks_walked (x : ℕ) (h1 : 7 * 2 = 14) (h2 : 2 * x + 14 = 24) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_annie_blocks_walked_l1013_101339


namespace NUMINAMATH_GPT_solve_ff_eq_x_l1013_101340

def f (x : ℝ) : ℝ := x^2 + 2 * x - 5

theorem solve_ff_eq_x :
  ∀ x : ℝ, f (f x) = x ↔ (x = ( -1 + Real.sqrt 21 ) / 2) ∨ (x = ( -1 - Real.sqrt 21 ) / 2) ∨
                          (x = ( -3 + Real.sqrt 17 ) / 2) ∨ (x = ( -3 - Real.sqrt 17 ) / 2) := 
by
  sorry

end NUMINAMATH_GPT_solve_ff_eq_x_l1013_101340


namespace NUMINAMATH_GPT_remy_sold_110_bottles_l1013_101315

theorem remy_sold_110_bottles 
    (price_per_bottle : ℝ)
    (total_evening_sales : ℝ)
    (evening_more_than_morning : ℝ)
    (nick_fewer_than_remy : ℝ)
    (R : ℝ) 
    (total_morning_sales_is : ℝ) :
    price_per_bottle = 0.5 →
    total_evening_sales = 55 →
    evening_more_than_morning = 3 →
    nick_fewer_than_remy = 6 →
    total_morning_sales_is = total_evening_sales - evening_more_than_morning →
    (R * price_per_bottle) + ((R - nick_fewer_than_remy) * price_per_bottle) = total_morning_sales_is →
    R = 110 :=
by
  intros
  sorry

end NUMINAMATH_GPT_remy_sold_110_bottles_l1013_101315


namespace NUMINAMATH_GPT_all_children_receive_candy_l1013_101355

-- Define f(x) function
def f (x n : ℕ) : ℕ := ((x * (x + 1)) / 2) % n

-- Define the problem statement: prove that all children receive at least one candy if n is a power of 2.
theorem all_children_receive_candy (n : ℕ) (h : ∃ m, n = 2^m) : 
    ∀ i : ℕ, i < n → ∃ x : ℕ, i = f x n := 
sorry

end NUMINAMATH_GPT_all_children_receive_candy_l1013_101355


namespace NUMINAMATH_GPT_num_cubes_with_more_than_one_blue_face_l1013_101380

-- Define the parameters of the problem
def block_length : ℕ := 5
def block_width : ℕ := 3
def block_height : ℕ := 1

def total_cubes : ℕ := 15
def corners : ℕ := 4
def edges : ℕ := 6
def middles : ℕ := 5

-- Define the condition that the total number of cubes painted on more than one face.
def cubes_more_than_one_blue_face : ℕ := corners + edges

-- Prove that the number of cubes painted on more than one face is 10
theorem num_cubes_with_more_than_one_blue_face :
  cubes_more_than_one_blue_face = 10 :=
by
  show (4 + 6) = 10
  sorry

end NUMINAMATH_GPT_num_cubes_with_more_than_one_blue_face_l1013_101380


namespace NUMINAMATH_GPT_blake_change_given_l1013_101335

theorem blake_change_given :
  let oranges := 40
  let apples := 50
  let mangoes := 60
  let total_amount := 300
  let total_spent := oranges + apples + mangoes
  let change_given := total_amount - total_spent
  change_given = 150 :=
by
  sorry

end NUMINAMATH_GPT_blake_change_given_l1013_101335


namespace NUMINAMATH_GPT_ratio_constant_l1013_101322

theorem ratio_constant (a b c d : ℕ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d)
    (h : ∀ k : ℕ, ∃ m : ℤ, a + c * k = m * (b + d * k)) :
    ∃ m : ℤ, ∀ k : ℕ, a + c * k = m * (b + d * k) :=
    sorry

end NUMINAMATH_GPT_ratio_constant_l1013_101322


namespace NUMINAMATH_GPT_theodore_pays_10_percent_in_taxes_l1013_101320

-- Defining the quantities
def num_stone_statues : ℕ := 10
def num_wooden_statues : ℕ := 20
def price_per_stone_statue : ℕ := 20
def price_per_wooden_statue : ℕ := 5
def total_earnings_after_taxes : ℕ := 270

-- Assertion: Theodore pays 10% of his earnings in taxes
theorem theodore_pays_10_percent_in_taxes :
  (num_stone_statues * price_per_stone_statue + num_wooden_statues * price_per_wooden_statue) - total_earnings_after_taxes
  = (10 * (num_stone_statues * price_per_stone_statue + num_wooden_statues * price_per_wooden_statue)) / 100 := 
by
  sorry

end NUMINAMATH_GPT_theodore_pays_10_percent_in_taxes_l1013_101320


namespace NUMINAMATH_GPT_moe_mowing_time_l1013_101396

noncomputable def effective_swath_width_inches : ℝ := 30 - 6
noncomputable def effective_swath_width_feet : ℝ := (effective_swath_width_inches / 12)
noncomputable def lawn_width : ℝ := 180
noncomputable def lawn_length : ℝ := 120
noncomputable def walking_rate : ℝ := 4500
noncomputable def total_strips : ℝ := lawn_width / effective_swath_width_feet
noncomputable def total_distance : ℝ := total_strips * lawn_length
noncomputable def time_required : ℝ := total_distance / walking_rate

theorem moe_mowing_time :
  time_required = 2.4 := by
  sorry

end NUMINAMATH_GPT_moe_mowing_time_l1013_101396


namespace NUMINAMATH_GPT_pizza_remained_l1013_101305

noncomputable def number_of_people := 15
noncomputable def fraction_eating_pizza := 3 / 5
noncomputable def total_pizza_pieces := 50
noncomputable def pieces_per_person := 4
noncomputable def pizza_remaining := total_pizza_pieces - (pieces_per_person * (fraction_eating_pizza * number_of_people))

theorem pizza_remained :
  pizza_remaining = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_pizza_remained_l1013_101305


namespace NUMINAMATH_GPT_fortieth_sequence_number_l1013_101308

theorem fortieth_sequence_number :
  (∃ r n : ℕ, ((r * (r + 1)) - 40 = n) ∧ (40 ≤ r * (r + 1)) ∧ (40 > (r - 1) * r) ∧ n = 2 * r) :=
sorry

end NUMINAMATH_GPT_fortieth_sequence_number_l1013_101308


namespace NUMINAMATH_GPT_mul_same_base_exp_ten_pow_1000_sq_l1013_101341

theorem mul_same_base_exp (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by
  sorry

-- Given specific constants for this problem
theorem ten_pow_1000_sq : (10:ℝ)^(1000) * (10)^(1000) = (10)^(2000) := by
  exact mul_same_base_exp 10 1000 1000

end NUMINAMATH_GPT_mul_same_base_exp_ten_pow_1000_sq_l1013_101341


namespace NUMINAMATH_GPT_smallest_product_l1013_101312

theorem smallest_product (S : Set ℤ) (hS : S = { -8, -3, -2, 2, 4 }) :
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a * b = -32 ∧ ∀ (x y : ℤ), x ∈ S → y ∈ S → x * y ≥ -32 :=
by
  sorry

end NUMINAMATH_GPT_smallest_product_l1013_101312


namespace NUMINAMATH_GPT_ten_thousands_written_correctly_ten_thousands_truncated_correctly_l1013_101330

-- Definitions to be used in the proof
def ten_thousands_description := "Three thousand nine hundred seventy-six ten thousands"
def num_written : ℕ := 39760000
def truncated_num : ℕ := 3976

-- Theorems to be proven
theorem ten_thousands_written_correctly :
  (num_written = 39760000) :=
sorry

theorem ten_thousands_truncated_correctly :
  (truncated_num = 3976) :=
sorry

end NUMINAMATH_GPT_ten_thousands_written_correctly_ten_thousands_truncated_correctly_l1013_101330


namespace NUMINAMATH_GPT_min_value_f_exists_min_value_f_l1013_101362

noncomputable def f (a b c : ℝ) := 1 / (b^2 + b * c) + 1 / (c^2 + c * a) + 1 / (a^2 + a * b)

theorem min_value_f (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) : f a b c ≥ 3 / 2 :=
  sorry

theorem exists_min_value_f : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧ f a b c = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_min_value_f_exists_min_value_f_l1013_101362


namespace NUMINAMATH_GPT_finalCostCalculation_l1013_101384

-- Define the inputs
def tireRepairCost : ℝ := 7
def salesTaxPerTire : ℝ := 0.50
def numberOfTires : ℕ := 4

-- The total cost should be $30
theorem finalCostCalculation : 
  let repairTotal := tireRepairCost * numberOfTires
  let salesTaxTotal := salesTaxPerTire * numberOfTires
  repairTotal + salesTaxTotal = 30 := 
by {
  sorry
}

end NUMINAMATH_GPT_finalCostCalculation_l1013_101384


namespace NUMINAMATH_GPT_largest_num_blocks_l1013_101373

-- Define the volume of the box
def volume_box (l₁ w₁ h₁ : ℕ) : ℕ :=
  l₁ * w₁ * h₁

-- Define the volume of the block
def volume_block (l₂ w₂ h₂ : ℕ) : ℕ :=
  l₂ * w₂ * h₂

-- Define the function to calculate maximum blocks
def max_blocks (V_box V_block : ℕ) : ℕ :=
  V_box / V_block

theorem largest_num_blocks :
  max_blocks (volume_box 5 4 6) (volume_block 3 3 2) = 6 :=
by
  sorry

end NUMINAMATH_GPT_largest_num_blocks_l1013_101373


namespace NUMINAMATH_GPT_smallest_k_divides_ab_l1013_101334

theorem smallest_k_divides_ab (S : Finset ℕ) (hS : S = Finset.range 51)
  (k : ℕ) : (∀ T : Finset ℕ, T ⊆ S → T.card = k → ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ (a + b) ∣ (a * b)) ↔ k = 39 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_divides_ab_l1013_101334


namespace NUMINAMATH_GPT_find_a_m_range_c_l1013_101300

noncomputable def f (x a : ℝ) := x^2 - 2*x + 2*a
def solution_set (f : ℝ → ℝ) (m : ℝ) := {x : ℝ | -2 ≤ x ∧ x ≤ m ∧ f x ≤ 0}

theorem find_a_m (a m : ℝ) : 
  (∀ x, f x a ≤ 0 ↔ -2 ≤ x ∧ x ≤ m) → a = -4 ∧ m = 4 := by
  sorry

theorem range_c (c : ℝ) : 
  (∀ x, (c - 4) * x^2 + 2 * (c - 4) * x - 1 < 0) → 13 / 4 < c ∧ c < 4 := by
  sorry

end NUMINAMATH_GPT_find_a_m_range_c_l1013_101300


namespace NUMINAMATH_GPT_alice_prank_combinations_l1013_101306

theorem alice_prank_combinations : 
  let monday_choices := 1
  let tuesday_choices := 3
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 60 :=
by
  let monday_choices := 1
  let tuesday_choices := 3
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  exact (show 1 * 3 * 5 * 4 * 1 = 60 from sorry)

end NUMINAMATH_GPT_alice_prank_combinations_l1013_101306


namespace NUMINAMATH_GPT_calculate_present_worth_l1013_101377

variable (BG : ℝ) (r : ℝ) (t : ℝ)

theorem calculate_present_worth (hBG : BG = 24) (hr : r = 0.10) (ht : t = 2) : 
  ∃ PW : ℝ, PW = 120 := 
by
  sorry

end NUMINAMATH_GPT_calculate_present_worth_l1013_101377


namespace NUMINAMATH_GPT_chickens_rabbits_l1013_101331

theorem chickens_rabbits (c r : ℕ) 
  (h1 : c = r - 20)
  (h2 : 4 * r = 6 * c + 10) :
  c = 35 := by
  sorry

end NUMINAMATH_GPT_chickens_rabbits_l1013_101331


namespace NUMINAMATH_GPT_tom_strokes_over_par_l1013_101390

theorem tom_strokes_over_par 
  (rounds : ℕ) 
  (holes_per_round : ℕ) 
  (avg_strokes_per_hole : ℕ) 
  (par_value_per_hole : ℕ) 
  (h1 : rounds = 9) 
  (h2 : holes_per_round = 18) 
  (h3 : avg_strokes_per_hole = 4) 
  (h4 : par_value_per_hole = 3) : 
  (rounds * holes_per_round * avg_strokes_per_hole - rounds * holes_per_round * par_value_per_hole = 162) :=
by { 
  sorry 
}

end NUMINAMATH_GPT_tom_strokes_over_par_l1013_101390


namespace NUMINAMATH_GPT_distinct_integer_values_b_for_quadratic_l1013_101304

theorem distinct_integer_values_b_for_quadratic :
  ∃ (S : Finset ℤ), ∀ b ∈ S, (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0) ∧ S.card = 16 :=
sorry

end NUMINAMATH_GPT_distinct_integer_values_b_for_quadratic_l1013_101304


namespace NUMINAMATH_GPT_closest_integer_to_cube_root_of_500_l1013_101386

theorem closest_integer_to_cube_root_of_500 :
  ∃ n : ℤ, (∀ m : ℤ, |m^3 - 500| ≥ |8^3 - 500|) := 
sorry

end NUMINAMATH_GPT_closest_integer_to_cube_root_of_500_l1013_101386


namespace NUMINAMATH_GPT_circle_equation_through_points_l1013_101365

-- Line and circle definitions
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 15 = 0

-- Intersection point definition
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ circle1 x y

-- Revised circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 28 * x - 15 * y = 0

-- Proof statement
theorem circle_equation_through_points :
  (∀ x y, intersection_point x y → circle_equation x y) ∧ circle_equation 0 0 :=
sorry

end NUMINAMATH_GPT_circle_equation_through_points_l1013_101365


namespace NUMINAMATH_GPT_find_constant_l1013_101348

theorem find_constant (t : ℝ) (constant : ℝ) :
  (x = constant - 3 * t) → (y = 2 * t - 3) → (t = 0.8) → (x = y) → constant = 1 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_constant_l1013_101348


namespace NUMINAMATH_GPT_smallest_zarks_l1013_101376

theorem smallest_zarks (n : ℕ) : (n^2 > 15 * n) → (n ≥ 16) := sorry

end NUMINAMATH_GPT_smallest_zarks_l1013_101376


namespace NUMINAMATH_GPT_pens_in_each_pack_l1013_101359

-- Given the conditions
def Kendra_packs : ℕ := 4
def Tony_packs : ℕ := 2
def pens_kept_each : ℕ := 2
def friends : ℕ := 14

-- Theorem statement
theorem pens_in_each_pack : ∃ (P : ℕ), Kendra_packs * P + Tony_packs * P - pens_kept_each * 2 - friends = 0 ∧ P = 3 := by
  sorry

end NUMINAMATH_GPT_pens_in_each_pack_l1013_101359


namespace NUMINAMATH_GPT_distribution_methods_l1013_101332

theorem distribution_methods (n m k : Nat) (h : n = 23) (h1 : m = 10) (h2 : k = 2) :
  (∃ d : Nat, d = Nat.choose m 1 + 2 * Nat.choose m 2 + Nat.choose m 3) →
  ∃ x : Nat, x = 220 :=
by
  sorry

end NUMINAMATH_GPT_distribution_methods_l1013_101332


namespace NUMINAMATH_GPT_parabola_vector_sum_distance_l1013_101328

noncomputable def parabola_focus (x y : ℝ) : Prop := x^2 = 8 * y

noncomputable def on_parabola (x y : ℝ) : Prop := parabola_focus x y

theorem parabola_vector_sum_distance :
  ∀ (A B C : ℝ × ℝ) (F : ℝ × ℝ),
  on_parabola A.1 A.2 ∧ on_parabola B.1 B.2 ∧ on_parabola C.1 C.2 ∧
  F = (0, 2) ∧
  ((A.1 - F.1)^2 + (A.2 - F.2)^2) + ((B.1 - F.1)^2 + (B.2 - F.2)^2) + ((C.1 - F.1)^2 + (C.2 - F.2)^2) = 0
  → (abs ((A.2 + F.2)) + abs ((B.2 + F.2)) + abs ((C.2 + F.2))) = 12 :=
by sorry

end NUMINAMATH_GPT_parabola_vector_sum_distance_l1013_101328


namespace NUMINAMATH_GPT_eight_div_pow_64_l1013_101370

theorem eight_div_pow_64 (h : 64 = 8^2) : 8^15 / 64^7 = 8 := by
  sorry

end NUMINAMATH_GPT_eight_div_pow_64_l1013_101370
