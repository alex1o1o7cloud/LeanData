import Mathlib

namespace NUMINAMATH_GPT_towel_bleach_volume_decrease_l1585_158563

theorem towel_bleach_volume_decrease :
  ∀ (L B T : ℝ) (L' B' T' : ℝ),
  (L' = L * 0.75) →
  (B' = B * 0.70) →
  (T' = T * 0.90) →
  (L * B * T = 1000000) →
  ((L * B * T - L' * B' * T') / (L * B * T) * 100) = 52.75 :=
by
  intros L B T L' B' T' hL' hB' hT' hV
  sorry

end NUMINAMATH_GPT_towel_bleach_volume_decrease_l1585_158563


namespace NUMINAMATH_GPT_fraction_to_decimal_l1585_158542

theorem fraction_to_decimal :
  (3 / 8 : ℝ) = 0.375 :=
sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1585_158542


namespace NUMINAMATH_GPT_number_of_squirrels_l1585_158543

/-
Problem: Some squirrels collected 575 acorns. If each squirrel needs 130 acorns to get through the winter, each squirrel needs to collect 15 more acorns. 
Question: How many squirrels are there?
Conditions:
 1. Some squirrels collected 575 acorns.
 2. Each squirrel needs 130 acorns to get through the winter.
 3. Each squirrel needs to collect 15 more acorns.
Answer: 5 squirrels
-/

theorem number_of_squirrels (acorns_total : ℕ) (acorns_needed : ℕ) (acorns_short : ℕ) (S : ℕ)
  (h1 : acorns_total = 575)
  (h2 : acorns_needed = 130)
  (h3 : acorns_short = 15)
  (h4 : S * (acorns_needed - acorns_short) = acorns_total) :
  S = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_squirrels_l1585_158543


namespace NUMINAMATH_GPT_small_glass_cost_l1585_158503

theorem small_glass_cost 
  (S : ℝ)
  (small_glass_cost : ℝ)
  (large_glass_cost : ℝ := 5)
  (initial_money : ℝ := 50)
  (num_small : ℝ := 8)
  (change : ℝ := 1)
  (num_large : ℝ := 5)
  (spent_money : ℝ := initial_money - change)
  (total_large_cost : ℝ := num_large * large_glass_cost)
  (total_cost : ℝ := num_small * S + total_large_cost)
  (total_cost_eq : total_cost = spent_money) :
  S = 3 :=
by
  sorry

end NUMINAMATH_GPT_small_glass_cost_l1585_158503


namespace NUMINAMATH_GPT_union_A_B_eq_C_l1585_158597

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
noncomputable def C : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}

theorem union_A_B_eq_C : A ∪ B = C := by
  sorry

end NUMINAMATH_GPT_union_A_B_eq_C_l1585_158597


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1585_158528

theorem quadratic_inequality_solution :
  {x : ℝ | -3 < x ∧ x < 1} = {x : ℝ | x * (x + 2) < 3} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1585_158528


namespace NUMINAMATH_GPT_part_I_part_II_l1585_158557

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + (1 - a) * x

theorem part_I (a : ℝ) (h_a : a ≠ 0) :
  (∃ x : ℝ, (x * (f a (1/x))) = 4 * x - 3 ∧ ∀ y, x = y → (x * (f a (1/x))) = 4 * x - 3) →
  a = 2 :=
sorry

noncomputable def f2 (x : ℝ) : ℝ := 2 / x - x

theorem part_II : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f2 x1 > f2 x2 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1585_158557


namespace NUMINAMATH_GPT_remainder_when_divided_by_8_l1585_158589

theorem remainder_when_divided_by_8 (x : ℤ) (h : ∃ k : ℤ, x = 72 * k + 19) : x % 8 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_8_l1585_158589


namespace NUMINAMATH_GPT_negation_of_proposition_l1585_158504

theorem negation_of_proposition
  (h : ∀ x : ℝ, x^2 - 2 * x + 2 > 0) :
  ∃ x : ℝ, x^2 - 2 * x + 2 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1585_158504


namespace NUMINAMATH_GPT_systematic_sample_seat_number_l1585_158509

theorem systematic_sample_seat_number (total_students sample_size interval : ℕ) (seat1 seat2 seat3 : ℕ) 
  (H_total_students : total_students = 56)
  (H_sample_size : sample_size = 4)
  (H_interval : interval = total_students / sample_size)
  (H_seat1 : seat1 = 3)
  (H_seat2 : seat2 = 31)
  (H_seat3 : seat3 = 45) :
  ∃ seat4 : ℕ, seat4 = 17 :=
by 
  sorry

end NUMINAMATH_GPT_systematic_sample_seat_number_l1585_158509


namespace NUMINAMATH_GPT_diophantine_no_nonneg_solutions_l1585_158540

theorem diophantine_no_nonneg_solutions {a b : ℕ} (ha : 0 < a) (hb : 0 < b) (h_gcd : Nat.gcd a b = 1) :
  ∃ (c : ℕ), (a * b - a - b + 1) / 2 = (a - 1) * (b - 1) / 2 := 
sorry

end NUMINAMATH_GPT_diophantine_no_nonneg_solutions_l1585_158540


namespace NUMINAMATH_GPT_max_non_overlapping_areas_l1585_158510

theorem max_non_overlapping_areas (n : ℕ) (h : 0 < n) :
  ∃ k : ℕ, k = 4 * n + 1 :=
sorry

end NUMINAMATH_GPT_max_non_overlapping_areas_l1585_158510


namespace NUMINAMATH_GPT_part1_part2_l1585_158599

universe u
variable {α : Type u}

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {3, 5, 6}

theorem part1 : A ∩ B = {3, 5} := by
  sorry

theorem part2 : (U \ A) ∪ B = {3, 4, 5, 6} := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1585_158599


namespace NUMINAMATH_GPT_proof_C_D_values_l1585_158552

-- Given the conditions
def denominator_factorization (x : ℝ) : Prop :=
  3 * x ^ 2 - x - 14 = (3 * x + 7) * (x - 2)

def fraction_equality (x : ℝ) (C D : ℝ) : Prop :=
  (3 * x ^ 2 + 7 * x - 20) / (3 * x ^ 2 - x - 14) =
  C / (x - 2) + D / (3 * x + 7)

-- The values to be proven
def values_C_D : Prop :=
  ∃ C D : ℝ, C = -14 / 13 ∧ D = 81 / 13 ∧ ∀ x : ℝ, (denominator_factorization x → fraction_equality x C D)

theorem proof_C_D_values : values_C_D :=
sorry

end NUMINAMATH_GPT_proof_C_D_values_l1585_158552


namespace NUMINAMATH_GPT_smallest_n_l1585_158574

theorem smallest_n (r g b n : ℕ) 
  (h1 : 12 * r = 14 * g)
  (h2 : 14 * g = 15 * b)
  (h3 : 15 * b = 20 * n)
  (h4 : ∀ n', (12 * r = 14 * g ∧ 14 * g = 15 * b ∧ 15 * b = 20 * n') → n ≤ n') :
  n = 21 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1585_158574


namespace NUMINAMATH_GPT_game_of_24_l1585_158554

theorem game_of_24 : 
  let a := 3
  let b := -5
  let c := 6
  let d := -8
  ((b + c / a) * d = 24) :=
by
  let a := 3
  let b := -5
  let c := 6
  let d := -8
  show (b + c / a) * d = 24
  sorry

end NUMINAMATH_GPT_game_of_24_l1585_158554


namespace NUMINAMATH_GPT_find_primes_l1585_158548

theorem find_primes (p : ℕ) (x y : ℕ) (hx : x > 0) (hy : y > 0) (hp : Nat.Prime p) : 
  (x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ (p = 2 ∨ p = 3 ∨ p = 7) := sorry

end NUMINAMATH_GPT_find_primes_l1585_158548


namespace NUMINAMATH_GPT_box_dimensions_l1585_158521

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by
  sorry

end NUMINAMATH_GPT_box_dimensions_l1585_158521


namespace NUMINAMATH_GPT_mouse_shortest_path_on_cube_l1585_158560

noncomputable def shortest_path_length (edge_length : ℝ) : ℝ :=
  2 * edge_length * Real.sqrt 2

theorem mouse_shortest_path_on_cube :
  shortest_path_length 2 = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_mouse_shortest_path_on_cube_l1585_158560


namespace NUMINAMATH_GPT_g_45_l1585_158517

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y
axiom g_30 : g 30 = 30

theorem g_45 : g 45 = 20 := by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_g_45_l1585_158517


namespace NUMINAMATH_GPT_inequality_solution_l1585_158523

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1585_158523


namespace NUMINAMATH_GPT_value_of_k_l1585_158532

theorem value_of_k (k : ℝ) (h1 : k ≠ 0) (h2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k * x₁ - 100) < (k * x₂ - 100)) : k = 1 :=
by
  have h3 : k > 0 :=
    sorry -- We know that if y increases as x increases, then k > 0
  have h4 : k = 1 :=
    sorry -- For this specific problem, we can take k = 1 which satisfies the conditions
  exact h4

end NUMINAMATH_GPT_value_of_k_l1585_158532


namespace NUMINAMATH_GPT_units_digit_k_squared_plus_pow2_k_l1585_158533

def n : ℕ := 4016
def k : ℕ := n^2 + 2^n

theorem units_digit_k_squared_plus_pow2_k :
  (k^2 + 2^k) % 10 = 7 := sorry

end NUMINAMATH_GPT_units_digit_k_squared_plus_pow2_k_l1585_158533


namespace NUMINAMATH_GPT_sqrt_product_simplified_l1585_158522

theorem sqrt_product_simplified (q : ℝ) (hq : 0 < q) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_product_simplified_l1585_158522


namespace NUMINAMATH_GPT_camel_water_ratio_l1585_158512

theorem camel_water_ratio (gallons_water : ℕ) (ounces_per_gallon : ℕ) (traveler_ounces : ℕ)
  (total_ounces : ℕ) (camel_ounces : ℕ) (ratio : ℕ) 
  (h1 : gallons_water = 2) 
  (h2 : ounces_per_gallon = 128) 
  (h3 : traveler_ounces = 32) 
  (h4 : total_ounces = gallons_water * ounces_per_gallon) 
  (h5 : camel_ounces = total_ounces - traveler_ounces)
  (h6 : ratio = camel_ounces / traveler_ounces) : 
  ratio = 7 := 
by
  sorry

end NUMINAMATH_GPT_camel_water_ratio_l1585_158512


namespace NUMINAMATH_GPT_meaningful_expression_range_l1585_158518

theorem meaningful_expression_range {x : ℝ} : (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end NUMINAMATH_GPT_meaningful_expression_range_l1585_158518


namespace NUMINAMATH_GPT_find_difference_l1585_158534

variable (k1 k2 t1 t2 : ℝ)

theorem find_difference (h1 : t1 = 5 / 9 * (k1 - 32))
                        (h2 : t2 = 5 / 9 * (k2 - 32))
                        (h3 : t1 = 105)
                        (h4 : t2 = 80) :
  k1 - k2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_l1585_158534


namespace NUMINAMATH_GPT_expansion_number_of_terms_l1585_158501

theorem expansion_number_of_terms (A B : Finset ℕ) (hA : A.card = 4) (hB : B.card = 5) : (A.card * B.card = 20) :=
by 
  sorry

end NUMINAMATH_GPT_expansion_number_of_terms_l1585_158501


namespace NUMINAMATH_GPT_question_equals_answer_l1585_158576

def heartsuit (a b : ℤ) : ℤ := |a + b|

theorem question_equals_answer : heartsuit (-3) (heartsuit 5 (-8)) = 0 := 
by
  sorry

end NUMINAMATH_GPT_question_equals_answer_l1585_158576


namespace NUMINAMATH_GPT_eq_of_symmetric_translation_l1585_158526

noncomputable def parabola (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 5

noncomputable def translate_left (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f (x + k)

noncomputable def translate_up (g : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := g x + k

noncomputable def translate_parabola (x : ℝ) : ℝ := translate_up (translate_left parabola 3) 2 x

noncomputable def symmetric_parabola (h : ℝ → ℝ) (x : ℝ) : ℝ := h (-x)

theorem eq_of_symmetric_translation :
  symmetric_parabola translate_parabola x = 2 * x^2 - 8 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_eq_of_symmetric_translation_l1585_158526


namespace NUMINAMATH_GPT_train_speed_in_kmph_l1585_158595

variable (L V : ℝ) -- L is the length of the train in meters, and V is the speed of the train in m/s.

-- Conditions given in the problem
def crosses_platform_in_30_seconds : Prop := L + 200 = V * 30
def crosses_man_in_20_seconds : Prop := L = V * 20

-- Length of the platform
def platform_length : ℝ := 200

-- The proof problem: Prove the speed of the train is 72 km/h
theorem train_speed_in_kmph 
  (h1 : crosses_man_in_20_seconds L V) 
  (h2 : crosses_platform_in_30_seconds L V) : 
  V * 3.6 = 72 := 
by 
  sorry

end NUMINAMATH_GPT_train_speed_in_kmph_l1585_158595


namespace NUMINAMATH_GPT_sum_of_roots_equals_18_l1585_158585

-- Define the conditions
variable (f : ℝ → ℝ)
variable (h_symmetry : ∀ x : ℝ, f (3 + x) = f (3 - x))
variable (h_distinct_roots : ∃ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0))

-- The theorem statement
theorem sum_of_roots_equals_18 (f : ℝ → ℝ) 
  (h_symmetry : ∀ x : ℝ, f (3 + x) = f (3 - x)) 
  (h_distinct_roots : ∃ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0)) :
  ∀ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0) → xs.sum id = 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_equals_18_l1585_158585


namespace NUMINAMATH_GPT_sum_mod_9_is_6_l1585_158525

noncomputable def sum_modulo_9 : ℤ :=
  1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999

theorem sum_mod_9_is_6 : sum_modulo_9 % 9 = 6 := 
  by
    sorry

end NUMINAMATH_GPT_sum_mod_9_is_6_l1585_158525


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1585_158572

variable (a : ℝ)

theorem sufficient_but_not_necessary_condition :
  (a > 2 → a^2 > 2 * a)
  ∧ (¬(a^2 > 2 * a → a > 2)) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1585_158572


namespace NUMINAMATH_GPT_max_bananas_l1585_158541

theorem max_bananas (a o b : ℕ) (h_a : a ≥ 1) (h_o : o ≥ 1) (h_b : b ≥ 1) (h_eq : 3 * a + 5 * o + 8 * b = 100) : b ≤ 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_bananas_l1585_158541


namespace NUMINAMATH_GPT_teresa_age_at_michiko_birth_l1585_158586

noncomputable def Teresa_age_now : ℕ := 59
noncomputable def Morio_age_now : ℕ := 71
noncomputable def Morio_age_at_Michiko_birth : ℕ := 38

theorem teresa_age_at_michiko_birth :
  (Teresa_age_now - (Morio_age_now - Morio_age_at_Michiko_birth)) = 26 := 
by
  sorry

end NUMINAMATH_GPT_teresa_age_at_michiko_birth_l1585_158586


namespace NUMINAMATH_GPT_max_pencils_to_buy_l1585_158555

-- Definition of costs and budget
def pin_cost : ℕ := 3
def pen_cost : ℕ := 4
def pencil_cost : ℕ := 9
def total_budget : ℕ := 72

-- Minimum purchase required: one pin and one pen
def min_purchase : ℕ := pin_cost + pen_cost

-- Remaining budget after minimum purchase
def remaining_budget : ℕ := total_budget - min_purchase

-- Maximum number of pencils can be bought with the remaining budget
def max_pencils := remaining_budget / pencil_cost

-- Theorem stating the maximum number of pencils Alice can purchase
theorem max_pencils_to_buy : max_pencils = 7 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_max_pencils_to_buy_l1585_158555


namespace NUMINAMATH_GPT_problem_l1585_158567

variable {R : Type} [Field R]

def f1 (a b c d : R) : R := a + b + c + d
def f2 (a b c d : R) : R := (1 / a) + (1 / b) + (1 / c) + (1 / d)
def f3 (a b c d : R) : R := (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) + (1 / (1 - d))

theorem problem (a b c d : R) (h1 : f1 a b c d = 2) (h2 : f2 a b c d = 2) : f3 a b c d = 2 :=
by sorry

end NUMINAMATH_GPT_problem_l1585_158567


namespace NUMINAMATH_GPT_least_N_bench_sections_l1585_158566

-- First, define the problem conditions
def bench_capacity_adult (N : ℕ) : ℕ := 7 * N
def bench_capacity_child (N : ℕ) : ℕ := 11 * N

-- Define the problem statement to be proven
theorem least_N_bench_sections :
  ∃ N : ℕ, (N > 0) ∧ (bench_capacity_adult N = bench_capacity_child N → N = 77) :=
sorry

end NUMINAMATH_GPT_least_N_bench_sections_l1585_158566


namespace NUMINAMATH_GPT_at_least_one_greater_than_one_l1585_158558

open Classical

variable (x y : ℝ)

theorem at_least_one_greater_than_one (h : x + y > 2) : x > 1 ∨ y > 1 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_greater_than_one_l1585_158558


namespace NUMINAMATH_GPT_sum_100_consecutive_from_neg49_l1585_158515

noncomputable def sum_of_consecutive_integers (n : ℕ) (first_term : ℤ) : ℤ :=
  n * ( first_term + (first_term + n - 1) ) / 2

theorem sum_100_consecutive_from_neg49 : sum_of_consecutive_integers 100 (-49) = 50 :=
by sorry

end NUMINAMATH_GPT_sum_100_consecutive_from_neg49_l1585_158515


namespace NUMINAMATH_GPT_ineq_triples_distinct_integers_l1585_158590

theorem ineq_triples_distinct_integers 
  (x y z : ℤ) (h₁ : x ≠ y) (h₂ : y ≠ z) (h₃ : z ≠ x) : 
  ( ( (x - y)^7 + (y - z)^7 + (z - x)^7 - (x - y) * (y - z) * (z - x) * ((x - y)^4 + (y - z)^4 + (z - x)^4) )
  / ( (x - y)^5 + (y - z)^5 + (z - x)^5 ) ) ≥ 3 :=
sorry

end NUMINAMATH_GPT_ineq_triples_distinct_integers_l1585_158590


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1585_158519

-- For System (1)
theorem system1_solution (x y : ℝ) (h1 : y = 2 * x) (h2 : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 :=
by
  sorry

-- For System (2)
theorem system2_solution (s t : ℝ) (h1 : 2 * s - 3 * t = 2) (h2 : (s + 2 * t) / 3 = 3 / 2) : s = 5 / 2 ∧ t = 1 :=
by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1585_158519


namespace NUMINAMATH_GPT_jordan_width_45_l1585_158511

noncomputable def carolRectangleLength : ℕ := 15
noncomputable def carolRectangleWidth : ℕ := 24
noncomputable def jordanRectangleLength : ℕ := 8
noncomputable def carolRectangleArea : ℕ := carolRectangleLength * carolRectangleWidth
noncomputable def jordanRectangleWidth (area : ℕ) : ℕ := area / jordanRectangleLength

theorem jordan_width_45 : jordanRectangleWidth carolRectangleArea = 45 :=
by sorry

end NUMINAMATH_GPT_jordan_width_45_l1585_158511


namespace NUMINAMATH_GPT_denominator_of_first_fraction_l1585_158539

theorem denominator_of_first_fraction (y x : ℝ) (h : y > 0) (h_eq : (9 * y) / x + (3 * y) / 10 = 0.75 * y) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_denominator_of_first_fraction_l1585_158539


namespace NUMINAMATH_GPT_PropositionA_PropositionD_l1585_158559

-- Proposition A: a > 1 is a sufficient but not necessary condition for 1/a < 1.
theorem PropositionA (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by sorry

-- PropositionD: a ≠ 0 is a necessary but not sufficient condition for ab ≠ 0.
theorem PropositionD (a b : ℝ) (h : a ≠ 0) : a * b ≠ 0 :=
by sorry
 
end NUMINAMATH_GPT_PropositionA_PropositionD_l1585_158559


namespace NUMINAMATH_GPT_solve_for_x_l1585_158536

theorem solve_for_x : 
  (35 / (6 - (2 / 5)) = 25 / 4) := 
by
  sorry 

end NUMINAMATH_GPT_solve_for_x_l1585_158536


namespace NUMINAMATH_GPT_amelia_painted_faces_l1585_158556

def faces_of_cuboid : ℕ := 6
def number_of_cuboids : ℕ := 6

theorem amelia_painted_faces : faces_of_cuboid * number_of_cuboids = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_amelia_painted_faces_l1585_158556


namespace NUMINAMATH_GPT_jessica_balloons_l1585_158500

-- Given conditions
def joan_balloons : Nat := 9
def sally_balloons : Nat := 5
def total_balloons : Nat := 16

-- The theorem to prove the number of balloons Jessica has
theorem jessica_balloons : (total_balloons - (joan_balloons + sally_balloons) = 2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jessica_balloons_l1585_158500


namespace NUMINAMATH_GPT_reflection_matrix_values_l1585_158506

theorem reflection_matrix_values (a b : ℝ) (I : Matrix (Fin 2) (Fin 2) ℝ) :
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![![a, 9/26], ![b, 17/26]]
  (R * R = I) → a = -17/26 ∧ b = 0 :=
by
  sorry

end NUMINAMATH_GPT_reflection_matrix_values_l1585_158506


namespace NUMINAMATH_GPT_prob_diff_fruit_correct_l1585_158535

noncomputable def prob_same_all_apple : ℝ := (0.4)^3
noncomputable def prob_same_all_orange : ℝ := (0.3)^3
noncomputable def prob_same_all_banana : ℝ := (0.2)^3
noncomputable def prob_same_all_grape : ℝ := (0.1)^3

noncomputable def prob_same_fruit_all_day : ℝ := 
  prob_same_all_apple + prob_same_all_orange + prob_same_all_banana + prob_same_all_grape

noncomputable def prob_diff_fruit (prob_same : ℝ) : ℝ := 1 - prob_same

theorem prob_diff_fruit_correct :
  prob_diff_fruit prob_same_fruit_all_day = 0.9 :=
by
  sorry

end NUMINAMATH_GPT_prob_diff_fruit_correct_l1585_158535


namespace NUMINAMATH_GPT_number_of_divisors_M_l1585_158562

def M : ℕ := 2^5 * 3^4 * 5^2 * 7^3 * 11^1

theorem number_of_divisors_M : (M.factors.prod.divisors.card = 720) :=
sorry

end NUMINAMATH_GPT_number_of_divisors_M_l1585_158562


namespace NUMINAMATH_GPT_unique_solution_f_l1585_158520

def f : ℕ → ℕ
  := sorry

namespace ProofProblem

theorem unique_solution_f (f : ℕ → ℕ)
  (h1 : ∀ (m n : ℕ), f m + f n - m * n ≠ 0)
  (h2 : ∀ (m n : ℕ), f m + f n - m * n ∣ m * f m + n * f n)
  : (∀ n : ℕ, f n = n^2) :=
sorry

end ProofProblem

end NUMINAMATH_GPT_unique_solution_f_l1585_158520


namespace NUMINAMATH_GPT_second_eq_value_l1585_158508

variable (x y z w : ℝ)

theorem second_eq_value (h1 : 4 * x * z + y * w = 3) (h2 : (2 * x + y) * (2 * z + w) = 15) : 
  x * w + y * z = 6 :=
by
  sorry

end NUMINAMATH_GPT_second_eq_value_l1585_158508


namespace NUMINAMATH_GPT_sum_of_three_consecutive_odd_integers_l1585_158565

theorem sum_of_three_consecutive_odd_integers (a : ℤ) (h₁ : a + (a + 4) = 150) :
  (a + (a + 2) + (a + 4) = 225) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_odd_integers_l1585_158565


namespace NUMINAMATH_GPT_eq_expression_l1585_158545

theorem eq_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
by
  sorry

end NUMINAMATH_GPT_eq_expression_l1585_158545


namespace NUMINAMATH_GPT_Jasper_height_in_10_minutes_l1585_158507

noncomputable def OmarRate : ℕ := 240 / 12
noncomputable def JasperRate : ℕ := 3 * OmarRate
noncomputable def JasperHeight (time: ℕ) : ℕ := JasperRate * time

theorem Jasper_height_in_10_minutes :
  JasperHeight 10 = 600 :=
by
  sorry

end NUMINAMATH_GPT_Jasper_height_in_10_minutes_l1585_158507


namespace NUMINAMATH_GPT_right_triangle_area_perimeter_ratio_l1585_158547

theorem right_triangle_area_perimeter_ratio :
  let a := 4
  let b := 8
  let area := (1/2) * a * b
  let c := Real.sqrt (a^2 + b^2)
  let perimeter := a + b + c
  let ratio := area / perimeter
  ratio = 3 - Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_perimeter_ratio_l1585_158547


namespace NUMINAMATH_GPT_Billy_has_10_fish_l1585_158581

def Billy_has_fish (Bobby Sarah Tony Billy : ℕ) : Prop :=
  Bobby = 2 * Sarah ∧
  Sarah = Tony + 5 ∧
  Tony = 3 * Billy ∧
  Bobby + Sarah + Tony + Billy = 145

theorem Billy_has_10_fish : ∃ (Billy : ℕ), Billy_has_fish (2 * (3 * Billy + 5)) (3 * Billy + 5) (3 * Billy) Billy ∧ Billy = 10 :=
by
  sorry

end NUMINAMATH_GPT_Billy_has_10_fish_l1585_158581


namespace NUMINAMATH_GPT_seismic_activity_mismatch_percentage_l1585_158544

theorem seismic_activity_mismatch_percentage
  (total_days : ℕ)
  (quiet_days_percentage : ℝ)
  (prediction_accuracy : ℝ)
  (predicted_quiet_days_percentage : ℝ)
  (quiet_prediction_correctness : ℝ)
  (active_days_percentage : ℝ)
  (incorrect_quiet_predictions : ℝ) :
  quiet_days_percentage = 0.8 →
  predicted_quiet_days_percentage = 0.64 →
  quiet_prediction_correctness = 0.7 →
  active_days_percentage = 0.2 →
  incorrect_quiet_predictions = predicted_quiet_days_percentage - (quiet_prediction_correctness * quiet_days_percentage) →
  (incorrect_quiet_predictions / active_days_percentage) * 100 = 40 := by
  sorry

end NUMINAMATH_GPT_seismic_activity_mismatch_percentage_l1585_158544


namespace NUMINAMATH_GPT_find_x_l1585_158530

theorem find_x (x n q r : ℕ) (h_n : n = 220080) (h_sum : n = (x + 445) * (2 * (x - 445)) + r) (h_r : r = 80) : 
  x = 555 :=
by
  have eq1 : n = 220080 := h_n
  have eq2 : n =  (x + 445) * (2 * (x - 445)) + r := h_sum
  have eq3 : r = 80 := h_r
  sorry

end NUMINAMATH_GPT_find_x_l1585_158530


namespace NUMINAMATH_GPT_no_solution_equation_l1585_158594

theorem no_solution_equation (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 → x ≠ 8 → (x - 3) / (x - 4) = (x - m) / (x - 8) → false) ↔ m = 7 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_equation_l1585_158594


namespace NUMINAMATH_GPT_find_function_l1585_158569

theorem find_function (a : ℝ) (f : ℝ → ℝ) (h : ∀ x y, (f x * f y - f (x * y)) / 4 = 2 * x + 2 * y + a) : a = -3 ∧ ∀ x, f x = x + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_function_l1585_158569


namespace NUMINAMATH_GPT_problem_1163_prime_and_16424_composite_l1585_158584

theorem problem_1163_prime_and_16424_composite :
  let x := 1910 * 10000 + 1112
  let a := 1163
  let b := 16424
  x = a * b →
  Prime a ∧ ¬ Prime b :=
by
  intros h
  sorry

end NUMINAMATH_GPT_problem_1163_prime_and_16424_composite_l1585_158584


namespace NUMINAMATH_GPT_GuntherFreeTime_l1585_158524

def GuntherCleaning : Nat := 45 + 60 + 30 + 15

def TotalFreeTime : Nat := 180

theorem GuntherFreeTime : TotalFreeTime - GuntherCleaning = 30 := by
  sorry

end NUMINAMATH_GPT_GuntherFreeTime_l1585_158524


namespace NUMINAMATH_GPT_customers_left_l1585_158591

theorem customers_left (original_customers remaining_tables people_per_table customers_left : ℕ)
  (h1 : original_customers = 44)
  (h2 : remaining_tables = 4)
  (h3 : people_per_table = 8)
  (h4 : original_customers - remaining_tables * people_per_table = customers_left) :
  customers_left = 12 :=
by
  sorry

end NUMINAMATH_GPT_customers_left_l1585_158591


namespace NUMINAMATH_GPT_triangle_cosine_sum_l1585_158598

theorem triangle_cosine_sum (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hsum : A + B + C = π) : 
  (Real.cos A + Real.cos B + Real.cos C > 1) :=
sorry

end NUMINAMATH_GPT_triangle_cosine_sum_l1585_158598


namespace NUMINAMATH_GPT_mixing_solutions_l1585_158577

theorem mixing_solutions (Vx : ℝ) :
  (0.10 * Vx + 0.30 * 900 = 0.25 * (Vx + 900)) ↔ Vx = 300 := by
  sorry

end NUMINAMATH_GPT_mixing_solutions_l1585_158577


namespace NUMINAMATH_GPT_white_squares_in_20th_row_l1585_158578

def num_squares_in_row (n : ℕ) : ℕ :=
  3 * n

def num_white_squares (n : ℕ) : ℕ :=
  (num_squares_in_row n - 2) / 2

theorem white_squares_in_20th_row: num_white_squares 20 = 30 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_white_squares_in_20th_row_l1585_158578


namespace NUMINAMATH_GPT_Li_Minghui_should_go_to_supermarket_B_for_300_yuan_cost_equal_for_500_yuan_l1585_158582

def cost_supermarket_A (x : ℝ) : ℝ :=
  200 + 0.8 * (x - 200)

def cost_supermarket_B (x : ℝ) : ℝ :=
  100 + 0.85 * (x - 100)

theorem Li_Minghui_should_go_to_supermarket_B_for_300_yuan :
  cost_supermarket_B 300 < cost_supermarket_A 300 := by
  sorry

theorem cost_equal_for_500_yuan :
  cost_supermarket_A 500 = cost_supermarket_B 500 := by
  sorry

end NUMINAMATH_GPT_Li_Minghui_should_go_to_supermarket_B_for_300_yuan_cost_equal_for_500_yuan_l1585_158582


namespace NUMINAMATH_GPT_timber_volume_after_two_years_correct_l1585_158516

-- Definitions based on the conditions in the problem
variables (a p b : ℝ) -- Assume a, p, and b are real numbers

-- Timber volume after one year
def timber_volume_one_year (a p b : ℝ) : ℝ := a * (1 + p) - b

-- Timber volume after two years
def timber_volume_two_years (a p b : ℝ) : ℝ := (timber_volume_one_year a p b) * (1 + p) - b

-- Prove that the timber volume after two years is equal to the given expression
theorem timber_volume_after_two_years_correct (a p b : ℝ) :
  timber_volume_two_years a p b = a * (1 + p)^2 - (2 + p) * b := sorry

end NUMINAMATH_GPT_timber_volume_after_two_years_correct_l1585_158516


namespace NUMINAMATH_GPT_workers_together_time_l1585_158596

theorem workers_together_time (A_time B_time : ℝ) (hA : A_time = 8) (hB : B_time = 10) :
  let rateA := 1 / A_time
  let rateB := 1 / B_time
  let combined_rate := rateA + rateB
  combined_rate * (40 / 9) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_workers_together_time_l1585_158596


namespace NUMINAMATH_GPT_annual_return_l1585_158588

theorem annual_return (initial_price profit : ℝ) (h₁ : initial_price = 5000) (h₂ : profit = 400) : 
  ((profit / initial_price) * 100 = 8) := by
  -- Lean's substitute for proof
  sorry

end NUMINAMATH_GPT_annual_return_l1585_158588


namespace NUMINAMATH_GPT_sum_min_max_z_l1585_158575

theorem sum_min_max_z (x y : ℝ) 
  (h1 : x - y - 2 ≥ 0) 
  (h2 : x - 5 ≤ 0) 
  (h3 : y + 2 ≥ 0) :
  ∃ (z_min z_max : ℝ), z_min = 2 ∧ z_max = 34 ∧ z_min + z_max = 36 :=
by
  sorry

end NUMINAMATH_GPT_sum_min_max_z_l1585_158575


namespace NUMINAMATH_GPT_piece_length_is_111_l1585_158573

-- Define the conditions
axiom condition1 : ∃ (x : ℤ), 9 * x ≤ 1000
axiom condition2 : ∃ (x : ℤ), 9 * x ≤ 1100

-- State the problem: Prove that the length of each piece is 111 centimeters
theorem piece_length_is_111 (x : ℤ) (h1 : 9 * x ≤ 1000) (h2 : 9 * x ≤ 1100) : x = 111 :=
by sorry

end NUMINAMATH_GPT_piece_length_is_111_l1585_158573


namespace NUMINAMATH_GPT_range_of_x_minus_2y_l1585_158513

theorem range_of_x_minus_2y (x y : ℝ) (h₁ : -1 ≤ x) (h₂ : x < 2) (h₃ : 0 < y) (h₄ : y ≤ 1) :
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 :=
sorry

end NUMINAMATH_GPT_range_of_x_minus_2y_l1585_158513


namespace NUMINAMATH_GPT_land_per_person_l1585_158593

noncomputable def total_land_area : ℕ := 20000
noncomputable def num_people_sharing : ℕ := 5

theorem land_per_person (Jose_land : ℕ) (h : Jose_land = total_land_area / num_people_sharing) :
  Jose_land = 4000 :=
by
  sorry

end NUMINAMATH_GPT_land_per_person_l1585_158593


namespace NUMINAMATH_GPT_find_n_l1585_158592

variable {a : ℕ → ℝ}  -- Defining the sequence

-- Defining the conditions:
def a1 : Prop := a 1 = 1 / 3
def a2_plus_a5 : Prop := a 2 + a 5 = 4
def a_n_eq_33 (n : ℕ) : Prop := a n = 33

theorem find_n (n : ℕ) : a 1 = 1 / 3 → (a 2 + a 5 = 4) → (a n = 33) → n = 50 := 
by 
  intros h1 h2 h3 
  -- the complete proof can be done here
  sorry

end NUMINAMATH_GPT_find_n_l1585_158592


namespace NUMINAMATH_GPT_find_n_l1585_158551

theorem find_n (n : ℕ) : (Nat.lcm n 10 = 36) ∧ (Nat.gcd n 10 = 5) → n = 18 :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_find_n_l1585_158551


namespace NUMINAMATH_GPT_line_intersects_parabola_at_one_point_l1585_158564

theorem line_intersects_parabola_at_one_point (k : ℝ) :
    (∃ y : ℝ, x = -3 * y^2 - 4 * y + 7) ↔ (x = k) := by
  sorry

end NUMINAMATH_GPT_line_intersects_parabola_at_one_point_l1585_158564


namespace NUMINAMATH_GPT_second_alloy_amount_l1585_158549

theorem second_alloy_amount (x : ℝ) : 
  (0.10 * 15 + 0.08 * x = 0.086 * (15 + x)) → 
  x = 35 := by 
sorry

end NUMINAMATH_GPT_second_alloy_amount_l1585_158549


namespace NUMINAMATH_GPT_digit_theta_l1585_158580

noncomputable def theta : ℕ := 7

theorem digit_theta (Θ : ℕ) (h1 : 378 / Θ = 40 + Θ + Θ) : Θ = theta :=
by {
  sorry
}

end NUMINAMATH_GPT_digit_theta_l1585_158580


namespace NUMINAMATH_GPT_distance_from_O_is_450_l1585_158553

noncomputable def find_distance_d (A B C P Q O : Type) 
  (side_length : ℝ) (PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ : ℝ) : ℝ :=
    if h : PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧ side_length = 600 ∧
           dist_OA = dist_OB ∧ dist_OB = dist_OC ∧ dist_OC = dist_OP ∧ dist_OP = dist_OQ ∧
           -- condition of 120 degree dihedral angle translates to specific geometric constraints
           true -- placeholder for the actual geometrical configuration that proves the problem
    then 450
    else 0 -- default or indication of inconsistency in conditions

-- Assuming all conditions hold true
theorem distance_from_O_is_450 (A B C P Q O : Type) 
  (side_length : ℝ) (PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ : ℝ)
  (h : PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧ side_length = 600 ∧
       dist_OA = dist_OB ∧ dist_OB = dist_OC ∧ dist_OC = dist_OP ∧ dist_OP = dist_OQ ∧
       -- adding condition of 120 degree dihedral angle
       true) -- true is a placeholder, the required proof to be filled in
  : find_distance_d A B C P Q O side_length PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ = 450 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_distance_from_O_is_450_l1585_158553


namespace NUMINAMATH_GPT_intersection_M_N_l1585_158527

noncomputable def M : Set ℕ := { x | 0 < x ∧ x < 8 }
def N : Set ℕ := { x | ∃ n : ℕ, x = 2 * n + 1 }
def K : Set ℕ := { 1, 3, 5, 7 }

theorem intersection_M_N : M ∩ N = K :=
by sorry

end NUMINAMATH_GPT_intersection_M_N_l1585_158527


namespace NUMINAMATH_GPT_find_range_m_l1585_158561

-- Definitions of the conditions
def p (m : ℝ) : Prop := ∃ x y : ℝ, (x + y - m = 0) ∧ ((x - 1)^2 + y^2 = 1)
def q (m : ℝ) : Prop := ∃ x : ℝ, (x^2 - x + m - 4 = 0) ∧ x ≠ 0 ∧ ∀ y : ℝ, (y^2 - y + m - 4 = 0) → x * y < 0

theorem find_range_m (m : ℝ) : (p m ∨ q m) ∧ ¬p m → (m ≤ 1 - Real.sqrt 2 ∨ 1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
by
  sorry

end NUMINAMATH_GPT_find_range_m_l1585_158561


namespace NUMINAMATH_GPT_simplified_value_of_f_l1585_158505

variable (x : ℝ)

noncomputable def f : ℝ := 3 * x + 5 - 4 * x^2 + 2 * x - 7 + x^2 - 3 * x + 8

theorem simplified_value_of_f : f x = -3 * x^2 + 2 * x + 6 := by
  unfold f
  sorry

end NUMINAMATH_GPT_simplified_value_of_f_l1585_158505


namespace NUMINAMATH_GPT_number_of_classes_l1585_158571

-- Define the conditions
def first_term : ℕ := 27
def common_diff : ℤ := -2
def total_students : ℕ := 115

-- Define and prove the main statement
theorem number_of_classes : ∃ n : ℕ, n > 0 ∧ (first_term + (n - 1) * common_diff) * n / 2 = total_students ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_classes_l1585_158571


namespace NUMINAMATH_GPT_FGH_supermarkets_total_l1585_158537

theorem FGH_supermarkets_total (US Canada : ℕ) 
  (h1 : US = 49) 
  (h2 : US = Canada + 14) : 
  US + Canada = 84 := 
by 
  sorry

end NUMINAMATH_GPT_FGH_supermarkets_total_l1585_158537


namespace NUMINAMATH_GPT_largest_t_value_maximum_t_value_l1585_158502

noncomputable def largest_t : ℚ :=
  (5 : ℚ) / 2

theorem largest_t_value (t : ℚ) :
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  t ≤ (5 : ℚ) / 2 :=
sorry

theorem maximum_t_value (t : ℚ) :
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  (5 : ℚ) / 2 = largest_t :=
sorry

end NUMINAMATH_GPT_largest_t_value_maximum_t_value_l1585_158502


namespace NUMINAMATH_GPT_taxi_ride_total_cost_l1585_158568

theorem taxi_ride_total_cost :
  let base_fee := 1.50
  let cost_per_mile := 0.25
  let distance1 := 5
  let distance2 := 8
  let distance3 := 3
  let cost1 := base_fee + distance1 * cost_per_mile
  let cost2 := base_fee + distance2 * cost_per_mile
  let cost3 := base_fee + distance3 * cost_per_mile
  cost1 + cost2 + cost3 = 8.50 := sorry

end NUMINAMATH_GPT_taxi_ride_total_cost_l1585_158568


namespace NUMINAMATH_GPT_solution_set_quadratic_inequality_l1585_158531

def quadraticInequalitySolutionSet 
  (x : ℝ) : Prop := 
  3 + 5 * x - 2 * x^2 > 0

theorem solution_set_quadratic_inequality :
  { x : ℝ | quadraticInequalitySolutionSet x } = 
  { x : ℝ | - (1:ℝ) / 2 < x ∧ x < 3 } :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_quadratic_inequality_l1585_158531


namespace NUMINAMATH_GPT_tangent_line_circle_sol_l1585_158587

theorem tangent_line_circle_sol (r : ℝ) (h_pos : r > 0)
  (h_tangent : ∀ x y : ℝ, x^2 + y^2 = 2 * r → x + 2 * y = r) : r = 10 := 
sorry

end NUMINAMATH_GPT_tangent_line_circle_sol_l1585_158587


namespace NUMINAMATH_GPT_only_powers_of_2_satisfy_condition_l1585_158538

theorem only_powers_of_2_satisfy_condition:
  ∀ (n : ℕ), n ≥ 2 →
  (∃ (x : ℕ → ℕ), 
    ∀ (i j : ℕ), 
      0 < i ∧ i < n → 0 < j ∧ j < n → i ≠ j ∧ (n ∣ (2 * i + j)) → x i < x j) ↔
      ∃ (s : ℕ), n = 2^s ∧ s ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_only_powers_of_2_satisfy_condition_l1585_158538


namespace NUMINAMATH_GPT_max_value_of_n_l1585_158570

theorem max_value_of_n : 
  ∃ n : ℕ, 
    (∀ m : ℕ, m ≤ n → (2 / 3)^(m - 1) * (1 / 3) ≥ 1 / 60) 
      ∧ 
    (∀ k : ℕ, k > n → (2 / 3)^(k - 1) * (1 / 3) < 1 / 60) 
      ∧ 
    n = 8 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_n_l1585_158570


namespace NUMINAMATH_GPT_negation_of_proposition_l1585_158529

theorem negation_of_proposition :
  (∀ x : ℝ, x > 0 → x + (1 / x) ≥ 2) →
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + (1 / x₀) < 2) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1585_158529


namespace NUMINAMATH_GPT_room_height_l1585_158514

-- Define the conditions
def total_curtain_length : ℕ := 101
def extra_material : ℕ := 5

-- Define the statement to be proven
theorem room_height : total_curtain_length - extra_material = 96 :=
by
  sorry

end NUMINAMATH_GPT_room_height_l1585_158514


namespace NUMINAMATH_GPT_actual_number_of_sides_l1585_158550

theorem actual_number_of_sides (apparent_angle : ℝ) (distortion_factor : ℝ)
  (sum_exterior_angles : ℝ) (actual_sides : ℕ) :
  apparent_angle = 18 ∧ distortion_factor = 1.5 ∧ sum_exterior_angles = 360 ∧ 
  apparent_angle / distortion_factor = sum_exterior_angles / actual_sides →
  actual_sides = 30 :=
by
  sorry

end NUMINAMATH_GPT_actual_number_of_sides_l1585_158550


namespace NUMINAMATH_GPT_juniors_to_freshmen_ratio_l1585_158579

variable (f s j : ℕ)

def participated_freshmen := 3 * f / 7
def participated_sophomores := 5 * s / 7
def participated_juniors := j / 2

-- The statement
theorem juniors_to_freshmen_ratio
    (h1 : participated_freshmen = participated_sophomores)
    (h2 : participated_freshmen = participated_juniors) :
    j = 6 * f / 7 ∧ f = 7 * j / 6 :=
by
  sorry

end NUMINAMATH_GPT_juniors_to_freshmen_ratio_l1585_158579


namespace NUMINAMATH_GPT_triangle_inequalities_l1585_158583

theorem triangle_inequalities (a b c h_a h_b h_c : ℝ) (ha_eq : h_a = b * Real.sin (arc_c)) (hb_eq : h_b = a * Real.sin (arc_c)) (hc_eq : h_c = a * Real.sin (arc_b)) (h : a > b) (h2 : b > c) :
  (a + h_a > b + h_b) ∧ (b + h_b > c + h_c) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequalities_l1585_158583


namespace NUMINAMATH_GPT_sum_of_integers_l1585_158546

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1585_158546
