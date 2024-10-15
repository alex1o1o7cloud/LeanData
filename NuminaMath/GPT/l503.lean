import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_b_l503_50350

theorem geometric_sequence_b (b : ℝ) (h1 : b > 0) (h2 : 30 * (b / 30) = b) (h3 : b * (b / 30) = 9 / 4) :
  b = 3 * Real.sqrt 30 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_b_l503_50350


namespace NUMINAMATH_GPT_lana_extra_flowers_l503_50311

theorem lana_extra_flowers (tulips roses used total extra : ℕ) 
  (h1 : tulips = 36) 
  (h2 : roses = 37) 
  (h3 : used = 70) 
  (h4 : total = tulips + roses) 
  (h5 : extra = total - used) : 
  extra = 3 := 
sorry

end NUMINAMATH_GPT_lana_extra_flowers_l503_50311


namespace NUMINAMATH_GPT_sin_870_eq_half_l503_50382

theorem sin_870_eq_half : Real.sin (870 * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_870_eq_half_l503_50382


namespace NUMINAMATH_GPT_compressor_distances_distances_when_a_15_l503_50324

theorem compressor_distances (a : ℝ) (x y z : ℝ) (h1 : x + y = 2 * z) (h2 : x + z = y + a) (h3 : x + z = 75) :
  0 < a ∧ a < 100 → 
  let x := (75 + a) / 3;
  let y := 75 - a;
  let z := 75 - x;
  x + y = 2 * z ∧ x + z = y + a ∧ x + z = 75 :=
sorry

theorem distances_when_a_15 (x y z : ℝ) (h : 15 = 15) :
  let x := (75 + 15) / 3;
  let y := 75 - 15;
  let z := 75 - x;
  x = 30 ∧ y = 60 ∧ z = 45 :=
sorry

end NUMINAMATH_GPT_compressor_distances_distances_when_a_15_l503_50324


namespace NUMINAMATH_GPT_gcd_factorial_eight_nine_eq_8_factorial_l503_50322

theorem gcd_factorial_eight_nine_eq_8_factorial : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_factorial_eight_nine_eq_8_factorial_l503_50322


namespace NUMINAMATH_GPT_find_number_chosen_l503_50399

theorem find_number_chosen (x : ℤ) (h : 4 * x - 138 = 102) : x = 60 := sorry

end NUMINAMATH_GPT_find_number_chosen_l503_50399


namespace NUMINAMATH_GPT_num_positive_integer_N_l503_50342

def num_valid_N : Nat := 7

theorem num_positive_integer_N (N : Nat) (h_pos : N > 0) :
  (∃ k : Nat, k > 3 ∧ N = k - 3 ∧ 48 % k = 0) ↔ (N < 45) ∧ (num_valid_N = 7) := 
by
sorry

end NUMINAMATH_GPT_num_positive_integer_N_l503_50342


namespace NUMINAMATH_GPT_angles_of_triangle_arith_seq_l503_50328

theorem angles_of_triangle_arith_seq (A B C a b c : ℝ) (h1 : A + B + C = 180) (h2 : A = B - (B - C)) (h3 : (1 / a + 1 / c) / 2 = 1 / b) : 
  A = 60 ∧ B = 60 ∧ C = 60 :=
sorry

end NUMINAMATH_GPT_angles_of_triangle_arith_seq_l503_50328


namespace NUMINAMATH_GPT_well_diameter_l503_50307

theorem well_diameter 
  (h : ℝ) 
  (P : ℝ) 
  (C : ℝ) 
  (V : ℝ) 
  (r : ℝ) 
  (d : ℝ) 
  (π : ℝ) 
  (h_eq : h = 14)
  (P_eq : P = 15)
  (C_eq : C = 1484.40)
  (V_eq : V = C / P)
  (volume_eq : V = π * r^2 * h)
  (radius_eq : r^2 = V / (π * h))
  (diameter_eq : d = 2 * r) : 
  d = 3 :=
by
  sorry

end NUMINAMATH_GPT_well_diameter_l503_50307


namespace NUMINAMATH_GPT_square_areas_l503_50300

theorem square_areas (s1 s2 s3 : ℕ)
  (h1 : s3 = s2 + 1)
  (h2 : s3 = s1 + 2)
  (h3 : s2 = 18)
  (h4 : s1 = s2 - 1) :
  s3^2 = 361 ∧ s2^2 = 324 ∧ s1^2 = 289 :=
by {
sorry
}

end NUMINAMATH_GPT_square_areas_l503_50300


namespace NUMINAMATH_GPT_dot_product_bounds_l503_50351

theorem dot_product_bounds
  (A : ℝ × ℝ)
  (hA : A.1 ^ 2 + (A.2 - 1) ^ 2 = 1) :
  -2 ≤ A.1 * 2 ∧ A.1 * 2 ≤ 2 := 
sorry

end NUMINAMATH_GPT_dot_product_bounds_l503_50351


namespace NUMINAMATH_GPT_range_of_a_l503_50354

theorem range_of_a 
    (x y a : ℝ) 
    (hx_pos : 0 < x) 
    (hy_pos : 0 < y) 
    (hxy : x + y = 1) 
    (hineq : ∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → (1 / x + a / y) ≥ 4) :
    a ≥ 1 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l503_50354


namespace NUMINAMATH_GPT_closest_multiple_of_15_to_2023_is_2025_l503_50362

theorem closest_multiple_of_15_to_2023_is_2025 (n : ℤ) (h : 15 * n = 2025) : 
  ∀ m : ℤ, abs (2023 - 2025) ≤ abs (2023 - 15 * m) :=
by
  exact sorry

end NUMINAMATH_GPT_closest_multiple_of_15_to_2023_is_2025_l503_50362


namespace NUMINAMATH_GPT_area_union_square_circle_l503_50313

noncomputable def side_length_square : ℝ := 12
noncomputable def radius_circle : ℝ := 15
noncomputable def area_union : ℝ := 144 + 168.75 * Real.pi

theorem area_union_square_circle : 
  let area_square := side_length_square ^ 2
  let area_circle := Real.pi * radius_circle ^ 2
  let area_quarter_circle := area_circle / 4
  area_union = area_square + area_circle - area_quarter_circle :=
by
  -- The actual proof is omitted
  sorry

end NUMINAMATH_GPT_area_union_square_circle_l503_50313


namespace NUMINAMATH_GPT_percent_increase_surface_area_l503_50376

theorem percent_increase_surface_area (a b c : ℝ) :
  let S := 2 * (a * b + b * c + a * c)
  let S' := 2 * (1.8 * a * 1.8 * b + 1.8 * b * 1.8 * c + 1.8 * c * 1.8 * a)
  (S' - S) / S * 100 = 224 := by
  sorry

end NUMINAMATH_GPT_percent_increase_surface_area_l503_50376


namespace NUMINAMATH_GPT_area_of_triangle_formed_by_tangent_line_l503_50316
-- Import necessary libraries from Mathlib

-- Set up the problem
theorem area_of_triangle_formed_by_tangent_line
  (f : ℝ → ℝ) (h_f : ∀ x, f x = x^2) :
  let slope := (deriv f 1)
  let tangent_line (x : ℝ) := slope * (x - 1) + f 1
  let x_intercept := (0 : ℝ)
  let y_intercept := tangent_line 0
  let area := 0.5 * abs x_intercept * abs y_intercept
  area = 1 / 4 :=
by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_area_of_triangle_formed_by_tangent_line_l503_50316


namespace NUMINAMATH_GPT_larger_cylinder_volume_l503_50305

theorem larger_cylinder_volume (v: ℝ) (r: ℝ) (R: ℝ) (h: ℝ) (hR : R = 2 * r) (hv : v = 100) : 
  π * R^2 * h = 4 * v := 
by 
  sorry

end NUMINAMATH_GPT_larger_cylinder_volume_l503_50305


namespace NUMINAMATH_GPT_find_k_l503_50332

noncomputable def k_val : ℝ := 19.2

theorem find_k (k : ℝ) :
  (4 + ∑' n : ℕ, (4 + n * k) / (5^(n + 1))) = 10 ↔ k = k_val :=
  sorry

end NUMINAMATH_GPT_find_k_l503_50332


namespace NUMINAMATH_GPT_angles_arith_prog_tangent_tangent_parallel_euler_line_l503_50363

-- Define a non-equilateral triangle with angles in arithmetic progression
structure Triangle :=
  (A B C : ℝ) -- Angles in a non-equilateral triangle
  (non_equilateral : A ≠ B ∨ B ≠ C ∨ A ≠ C)
  (angles_arith_progression : (2 * B = A + C))

-- Additional geometry concepts will be assumptions as their definition 
-- would involve extensive axiomatic setups

-- The main theorem to state the equivalence
theorem angles_arith_prog_tangent_tangent_parallel_euler_line (Δ : Triangle)
  (common_tangent_parallel_euler : sorry) : 
  ((Δ.A = 60) ∨ (Δ.B = 60) ∨ (Δ.C = 60)) :=
sorry

end NUMINAMATH_GPT_angles_arith_prog_tangent_tangent_parallel_euler_line_l503_50363


namespace NUMINAMATH_GPT_jerry_total_bill_l503_50346

-- Definitions for the initial bill and late fees
def initial_bill : ℝ := 250
def first_fee_rate : ℝ := 0.02
def second_fee_rate : ℝ := 0.03

-- Function to calculate the total bill after applying the fees
def total_bill (init : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let first_total := init * (1 + rate1)
  first_total * (1 + rate2)

-- Theorem statement
theorem jerry_total_bill : total_bill initial_bill first_fee_rate second_fee_rate = 262.65 := by
  sorry

end NUMINAMATH_GPT_jerry_total_bill_l503_50346


namespace NUMINAMATH_GPT_sequence_nth_term_16_l503_50366

theorem sequence_nth_term_16 (n : ℕ) (sqrt2 : ℝ) (h_sqrt2 : sqrt2 = Real.sqrt 2) (a_n : ℕ → ℝ) 
  (h_seq : ∀ n, a_n n = sqrt2 ^ (n - 1)) :
  a_n n = 16 → n = 9 := by
  sorry

end NUMINAMATH_GPT_sequence_nth_term_16_l503_50366


namespace NUMINAMATH_GPT_tangent_line_equation_l503_50304

theorem tangent_line_equation (P : ℝ × ℝ) (hP : P = (-4, -3)) :
  ∃ (a b c : ℝ), a * -4 + b * -3 + c = 0 ∧ a * a + b * b = (5:ℝ)^2 ∧ 
                 a = 4 ∧ b = 3 ∧ c = 25 := 
sorry

end NUMINAMATH_GPT_tangent_line_equation_l503_50304


namespace NUMINAMATH_GPT_marts_income_percentage_of_juans_l503_50370

variable (T J M : Real)
variable (h1 : M = 1.60 * T)
variable (h2 : T = 0.40 * J)

theorem marts_income_percentage_of_juans : M = 0.64 * J :=
by
  sorry

end NUMINAMATH_GPT_marts_income_percentage_of_juans_l503_50370


namespace NUMINAMATH_GPT_sum_two_triangular_numbers_iff_l503_50393

theorem sum_two_triangular_numbers_iff (m : ℕ) : 
  (∃ a b : ℕ, m = (a * (a + 1)) / 2 + (b * (b + 1)) / 2) ↔ 
  (∃ x y : ℕ, 4 * m + 1 = x * x + y * y) :=
by sorry

end NUMINAMATH_GPT_sum_two_triangular_numbers_iff_l503_50393


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l503_50387

theorem problem1 (α : ℝ) (h₁ : Real.sin α > 0) (h₂ : Real.tan α > 0) :
  α ∈ { x : ℝ | x >= 0 ∧ x < π/2 } := sorry

theorem problem2 (α : ℝ) (h₁ : Real.tan α * Real.sin α < 0) :
  α ∈ { x : ℝ | (x > π/2 ∧ x < π) ∨ (x > π ∧ x < 3 * π / 2) } := sorry

theorem problem3 (α : ℝ) (h₁ : Real.sin α * Real.cos α < 0) :
  α ∈ { x : ℝ | (x > π/2 ∧ x < π) ∨ (x > 3 * π / 2 ∧ x < 2 * π) } := sorry

theorem problem4 (α : ℝ) (h₁ : Real.cos α * Real.tan α > 0) :
  α ∈ { x : ℝ | x >= 0 ∧ x < π ∨ x > π ∧ x < 3 * π / 2 } := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l503_50387


namespace NUMINAMATH_GPT_evaluate_expression_l503_50329

def a := 3 + 6 + 9
def b := 2 + 5 + 8
def c := 3 + 6 + 9
def d := 2 + 5 + 8

theorem evaluate_expression : (a / b) - (d / c) = 11 / 30 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l503_50329


namespace NUMINAMATH_GPT_remainder_1425_1427_1429_mod_12_l503_50386

theorem remainder_1425_1427_1429_mod_12 :
  (1425 * 1427 * 1429) % 12 = 11 :=
by
  sorry

end NUMINAMATH_GPT_remainder_1425_1427_1429_mod_12_l503_50386


namespace NUMINAMATH_GPT_angelina_journey_equation_l503_50303

theorem angelina_journey_equation (t : ℝ) :
    4 = t + 15/60 + (4 - 15/60 - t) →
    60 * t + 90 * (15/4 - t) = 255 :=
    by
    sorry

end NUMINAMATH_GPT_angelina_journey_equation_l503_50303


namespace NUMINAMATH_GPT_negate_original_is_correct_l503_50314

-- Define the original proposition
def original_proposition (a b : ℕ) : Prop := (a * b = 0) → (a = 0 ∨ b = 0)

-- Define the negated proposition
def negated_proposition (a b : ℕ) : Prop := (a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0)

-- The theorem stating that the negation of the original proposition is the given negated proposition
theorem negate_original_is_correct (a b : ℕ) : ¬ original_proposition a b ↔ negated_proposition a b := by
  sorry

end NUMINAMATH_GPT_negate_original_is_correct_l503_50314


namespace NUMINAMATH_GPT_evaluate_expression_l503_50302

theorem evaluate_expression : 4 * 5 - 3 + 2^3 - 3 * 2 = 19 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l503_50302


namespace NUMINAMATH_GPT__l503_50309

noncomputable def waiter_fraction_from_tips (S T I : ℝ) : Prop :=
  T = (5 / 2) * S ∧
  I = S + T ∧
  T / I = 5 / 7

lemma waiter_tips_fraction_theorem (S T I : ℝ) : waiter_fraction_from_tips S T I → T / I = 5 / 7 :=
by
  intro h
  rw [waiter_fraction_from_tips] at h
  obtain ⟨h₁, h₂, h₃⟩ := h
  exact h₃

end NUMINAMATH_GPT__l503_50309


namespace NUMINAMATH_GPT_min_value_a4b3c2_l503_50341

theorem min_value_a4b3c2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 := 
sorry

end NUMINAMATH_GPT_min_value_a4b3c2_l503_50341


namespace NUMINAMATH_GPT_exactly_three_assertions_l503_50394

theorem exactly_three_assertions (x : ℕ) : 
  10 ≤ x ∧ x < 100 ∧
  ((x % 3 = 0) ∧ (x % 5 = 0) ∧ (x % 9 ≠ 0) ∧ (x % 15 = 0) ∧ (x % 25 ≠ 0) ∧ (x % 45 ≠ 0)) ↔
  (x = 15 ∨ x = 30 ∨ x = 60) :=
by
  sorry

end NUMINAMATH_GPT_exactly_three_assertions_l503_50394


namespace NUMINAMATH_GPT_reading_ratio_l503_50344

theorem reading_ratio (x : ℕ) (h1 : 10 * x + 5 * (75 - x) = 500) : 
  (10 * x) / 500 = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_reading_ratio_l503_50344


namespace NUMINAMATH_GPT_fraction_value_l503_50360

theorem fraction_value (p q x : ℚ) (h₁ : p / q = 4 / 5) (h₂ : 2 * q + p ≠ 0) (h₃ : 2 * q - p ≠ 0) :
  x + (2 * q - p) / (2 * q + p) = 2 → x = 11 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l503_50360


namespace NUMINAMATH_GPT_prime_problem_l503_50339

open Nat

-- Definition of primes and conditions based on the problem
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The formalized problem and conditions
theorem prime_problem (p q s : ℕ) 
  (p_prime : is_prime p) 
  (q_prime : is_prime q) 
  (s_prime : is_prime s) 
  (h1 : p + q = s + 4) 
  (h2 : 1 < p) 
  (h3 : p < q) : 
  p = 2 :=
sorry

end NUMINAMATH_GPT_prime_problem_l503_50339


namespace NUMINAMATH_GPT_max_value_of_f_l503_50391

open Real

noncomputable def f (θ : ℝ) : ℝ :=
  sin (θ / 2) * (1 + cos θ)

theorem max_value_of_f : 
  (∃ θ : ℝ, 0 < θ ∧ θ < π ∧ (∀ θ' : ℝ, 0 < θ' ∧ θ' < π → f θ' ≤ f θ) ∧ f θ = 4 * sqrt 3 / 9) := 
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l503_50391


namespace NUMINAMATH_GPT_round_trip_percentage_l503_50337

-- Definitions based on the conditions
variable (P : ℝ) -- Total number of passengers
variable (R : ℝ) -- Number of round-trip ticket holders

-- First condition: 20% of passengers held round-trip tickets and took their cars aboard
def condition1 := 0.20 * P = 0.60 * R

-- Second condition: 40% of passengers with round-trip tickets did not take their cars aboard (implies 60% did)
theorem round_trip_percentage (h1 : condition1 P R) : (R / P) * 100 = 33.33 := by
  sorry

end NUMINAMATH_GPT_round_trip_percentage_l503_50337


namespace NUMINAMATH_GPT_simplify_expression_l503_50361

variables (a b : ℝ)

theorem simplify_expression : 
  a^(2/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a := by
  -- proof here
  sorry

end NUMINAMATH_GPT_simplify_expression_l503_50361


namespace NUMINAMATH_GPT_fraction_sum_eq_l503_50331

-- Given conditions
variables (w x y : ℝ)
axiom hx : w / x = 1 / 6
axiom hy : w / y = 1 / 5

-- Proof goal
theorem fraction_sum_eq : (x + y) / y = 11 / 5 :=
by sorry

end NUMINAMATH_GPT_fraction_sum_eq_l503_50331


namespace NUMINAMATH_GPT_no_solution_for_equation_l503_50365

theorem no_solution_for_equation : 
  ∀ x : ℝ, (x ≠ 3) → (x-1)/(x-3) = 2 - 2/(3-x) → False :=
by
  intro x hx heq
  sorry

end NUMINAMATH_GPT_no_solution_for_equation_l503_50365


namespace NUMINAMATH_GPT_rectangle_area_l503_50379

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l503_50379


namespace NUMINAMATH_GPT_cubic_roots_identity_l503_50317

theorem cubic_roots_identity 
  (x1 x2 x3 p q : ℝ) 
  (hq : ∀ x, x^3 + p * x + q = (x - x1) * (x - x2) * (x - x3))
  (h_sum : x1 + x2 + x3 = 0)
  (h_prod : x1 * x2 + x2 * x3 + x3 * x1 = p):
  x2^2 + x2 * x3 + x3^2 = -p ∧ x1^2 + x1 * x3 + x3^2 = -p ∧ x1^2 + x1 * x2 + x2^2 = -p :=
sorry

end NUMINAMATH_GPT_cubic_roots_identity_l503_50317


namespace NUMINAMATH_GPT_corset_total_cost_l503_50367

def purple_bead_cost : ℝ := 50 * 20 * 0.12
def blue_bead_cost : ℝ := 40 * 18 * 0.10
def gold_bead_cost : ℝ := 80 * 0.08
def red_bead_cost : ℝ := 30 * 15 * 0.09
def silver_bead_cost : ℝ := 100 * 0.07

def total_cost : ℝ := purple_bead_cost + blue_bead_cost + gold_bead_cost + red_bead_cost + silver_bead_cost

theorem corset_total_cost : total_cost = 245.90 := by
  sorry

end NUMINAMATH_GPT_corset_total_cost_l503_50367


namespace NUMINAMATH_GPT_binary_addition_l503_50381

theorem binary_addition :
  (0b1101 : Nat) + 0b101 + 0b1110 + 0b111 + 0b1010 = 0b10101 := by
  sorry

end NUMINAMATH_GPT_binary_addition_l503_50381


namespace NUMINAMATH_GPT_parallel_vectors_x_value_l503_50308

def vectors_are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, vectors_are_parallel (-1, 4) (x, 2) → x = -1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_value_l503_50308


namespace NUMINAMATH_GPT_isosceles_triangle_l503_50323

theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ) (hAcosB : a * Real.cos B = b * Real.cos A) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (a = b ∨ b = c ∨ a = c) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_l503_50323


namespace NUMINAMATH_GPT_no_partition_exists_l503_50364

theorem no_partition_exists : ¬ ∃ (x y : ℕ), 
    (1 ≤ x ∧ x ≤ 15) ∧ 
    (1 ≤ y ∧ y ≤ 15) ∧ 
    (x * y = 120 - x - y) :=
by
  sorry

end NUMINAMATH_GPT_no_partition_exists_l503_50364


namespace NUMINAMATH_GPT_P_never_77_l503_50383

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_never_77 (x y : ℤ) : P x y ≠ 77 := sorry

end NUMINAMATH_GPT_P_never_77_l503_50383


namespace NUMINAMATH_GPT_solve_for_y_l503_50375

theorem solve_for_y (x y : ℝ) (h₁ : x^(2 * y) = 64) (h₂ : x = 8) : y = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l503_50375


namespace NUMINAMATH_GPT_find_radius_of_circle_l503_50369

theorem find_radius_of_circle :
  ∀ (r : ℝ) (α : ℝ) (ρ : ℝ) (θ : ℝ), r > 0 →
  (∀ (x y : ℝ), x = r * Real.cos α ∧ y = r * Real.sin α → x^2 + y^2 = r^2) →
  (∃ (x y: ℝ), x - y + 2 = 0 ∧ 2 * Real.sqrt (r^2 - 2) = 2 * Real.sqrt 2) →
  r = 2 :=
by
  intro r α ρ θ r_pos curve_eq polar_eq
  sorry

end NUMINAMATH_GPT_find_radius_of_circle_l503_50369


namespace NUMINAMATH_GPT_problem1_eval_problem2_eval_l503_50345

theorem problem1_eval : (1 * (Real.pi - 3.14)^0 - |2 - Real.sqrt 3| + (-1 / 2)^2) = Real.sqrt 3 - 3 / 4 :=
  sorry

theorem problem2_eval : (Real.sqrt (1 / 3) + Real.sqrt 6 * (1 / Real.sqrt 2 + Real.sqrt 8)) = 16 * Real.sqrt 3 / 3 :=
  sorry

end NUMINAMATH_GPT_problem1_eval_problem2_eval_l503_50345


namespace NUMINAMATH_GPT_user_count_exceed_50000_l503_50358

noncomputable def A (t : ℝ) (k : ℝ) := 500 * Real.exp (k * t)

theorem user_count_exceed_50000 :
  (∃ k : ℝ, A 10 k = 2000) →
  (∀ t : ℝ, A t k > 50000) →
  ∃ t : ℝ, t >= 34 :=
by
  sorry

end NUMINAMATH_GPT_user_count_exceed_50000_l503_50358


namespace NUMINAMATH_GPT_b_10_eq_64_l503_50392

noncomputable def a (n : ℕ) : ℕ := -- Definition of the sequence a_n
  sorry

noncomputable def b (n : ℕ) : ℕ := -- Definition of the sequence b_n
  a n + a (n + 1)

theorem b_10_eq_64 (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a n * a (n + 1) = 2^n) :
  b 10 = 64 :=
sorry

end NUMINAMATH_GPT_b_10_eq_64_l503_50392


namespace NUMINAMATH_GPT_total_beads_correct_l503_50398

-- Definitions of the problem conditions
def blue_beads : ℕ := 5
def red_beads : ℕ := 2 * blue_beads
def white_beads : ℕ := blue_beads + red_beads
def silver_beads : ℕ := 10

-- Definition of the total number of beads
def total_beads : ℕ := blue_beads + red_beads + white_beads + silver_beads

-- The main theorem statement
theorem total_beads_correct : total_beads = 40 :=
by 
  sorry

end NUMINAMATH_GPT_total_beads_correct_l503_50398


namespace NUMINAMATH_GPT_max_divisor_f_l503_50349

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_f (m : ℕ) : (∀ n : ℕ, m ∣ f n) → m = 36 :=
sorry

end NUMINAMATH_GPT_max_divisor_f_l503_50349


namespace NUMINAMATH_GPT_count_teams_of_6_l503_50380

theorem count_teams_of_6 
  (students : Fin 12 → Type)
  (played_together_once : ∀ (s : Finset (Fin 12)) (h : s.card = 5), ∃! t : Finset (Fin 12), t.card = 6 ∧ s ⊆ t) :
  (∃ team_count : ℕ, team_count = 132) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_count_teams_of_6_l503_50380


namespace NUMINAMATH_GPT_larger_number_is_22_l503_50315

theorem larger_number_is_22 (x y : ℕ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_22_l503_50315


namespace NUMINAMATH_GPT_insulation_cost_of_rectangular_tank_l503_50389

theorem insulation_cost_of_rectangular_tank
  (l w h cost_per_sq_ft : ℕ)
  (hl : l = 4) (hw : w = 5) (hh : h = 3) (hc : cost_per_sq_ft = 20) :
  2 * l * w + 2 * l * h + 2 * w * h * 20 = 1880 :=
by
  sorry

end NUMINAMATH_GPT_insulation_cost_of_rectangular_tank_l503_50389


namespace NUMINAMATH_GPT_largest_number_l503_50325

theorem largest_number (a b c d : ℤ)
  (h1 : (a + b + c) / 3 + d = 17)
  (h2 : (a + b + d) / 3 + c = 21)
  (h3 : (a + c + d) / 3 + b = 23)
  (h4 : (b + c + d) / 3 + a = 29) :
  d = 21 := 
sorry

end NUMINAMATH_GPT_largest_number_l503_50325


namespace NUMINAMATH_GPT_find_m_ineq_soln_set_min_value_a2_b2_l503_50320

-- Problem 1
theorem find_m_ineq_soln_set (m x : ℝ) (h1 : m - |x - 2| ≥ 1) (h2 : x ∈ Set.Icc 0 4) : m = 3 := by
  sorry

-- Problem 2
theorem min_value_a2_b2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) : a^2 + b^2 ≥ 9 / 2 := by
  sorry

end NUMINAMATH_GPT_find_m_ineq_soln_set_min_value_a2_b2_l503_50320


namespace NUMINAMATH_GPT_distinct_numbers_mean_inequality_l503_50355

open Nat

theorem distinct_numbers_mean_inequality (n m : ℕ) (h_n_m : m ≤ n)
  (a : Fin m → ℕ) (ha_distinct : Function.Injective a)
  (h_cond : ∀ (i j : Fin m), i ≠ j → i.val + j.val ≤ n → ∃ (k : Fin m), a i + a j = a k) :
  (1 : ℝ) / m * (Finset.univ.sum (fun i => a i)) ≥  (n + 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_distinct_numbers_mean_inequality_l503_50355


namespace NUMINAMATH_GPT_cocos_August_bill_l503_50318

noncomputable def total_cost (a_monthly_cost: List (Float × Float)) :=
a_monthly_cost.foldr (fun x acc => (x.1 * x.2 * 0.09) + acc) 0

theorem cocos_August_bill :
  let oven        := (2.4, 25)
  let air_cond    := (1.6, 150)
  let refrigerator := (0.15, 720)
  let washing_mach := (0.5, 20) 
  total_cost [oven, air_cond, refrigerator, washing_mach] = 37.62 :=
by
  sorry

end NUMINAMATH_GPT_cocos_August_bill_l503_50318


namespace NUMINAMATH_GPT_even_sum_sufficient_not_necessary_l503_50340

theorem even_sum_sufficient_not_necessary (m n : ℤ) : 
  (∀ m n : ℤ, (Even m ∧ Even n) → Even (m + n)) 
  ∧ (∀ a b : ℤ, Even (a + b) → ¬ (Odd a ∧ Odd b)) :=
by
  sorry

end NUMINAMATH_GPT_even_sum_sufficient_not_necessary_l503_50340


namespace NUMINAMATH_GPT_smallest_cube_dividing_pq2r4_l503_50347

-- Definitions of conditions
variables {p q r : ℕ} [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] [Fact (Nat.Prime r)]
variables (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)

-- Definitions used in the proof
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

def smallest_perfect_cube_dividing (n k : ℕ) : Prop :=
  is_perfect_cube k ∧ n ∣ k ∧ ∀ k', is_perfect_cube k' ∧ n ∣ k' → k ≤ k'

-- The proof problem
theorem smallest_cube_dividing_pq2r4 (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  smallest_perfect_cube_dividing (p * q^2 * r^4) ((p * q * r^2)^3) :=
sorry

end NUMINAMATH_GPT_smallest_cube_dividing_pq2r4_l503_50347


namespace NUMINAMATH_GPT_find_y_coordinate_of_first_point_l503_50334

theorem find_y_coordinate_of_first_point :
  ∃ y1 : ℝ, ∀ k : ℝ, (k = 0.8) → (k = (0.8 - y1) / (5 - (-1))) → y1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_y_coordinate_of_first_point_l503_50334


namespace NUMINAMATH_GPT_spherical_to_rectangular_conversion_l503_50368

theorem spherical_to_rectangular_conversion :
  ∃ x y z : ℝ, 
    x = -Real.sqrt 2 ∧ 
    y = 0 ∧ 
    z = Real.sqrt 2 ∧ 
    (∃ rho theta phi : ℝ, 
      rho = 2 ∧
      theta = π ∧
      phi = π/4 ∧
      x = rho * Real.sin phi * Real.cos theta ∧
      y = rho * Real.sin phi * Real.sin theta ∧
      z = rho * Real.cos phi) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_conversion_l503_50368


namespace NUMINAMATH_GPT_solve_inequality_correct_l503_50336

noncomputable def solve_inequality (a x : ℝ) : Set ℝ :=
  if a > 1 ∨ a < 0 then {x | x ≤ a ∨ x ≥ a^2 }
  else if a = 1 ∨ a = 0 then {x | True}
  else {x | x ≤ a^2 ∨ x ≥ a}

theorem solve_inequality_correct (a x : ℝ) :
  (x^2 - (a^2 + a) * x + a^3 ≥ 0) ↔ 
    (if a > 1 ∨ a < 0 then x ≤ a ∨ x ≥ a^2
      else if a = 1 ∨ a = 0 then True
      else x ≤ a^2 ∨ x ≥ a) :=
by sorry

end NUMINAMATH_GPT_solve_inequality_correct_l503_50336


namespace NUMINAMATH_GPT_line_fixed_point_l503_50372

theorem line_fixed_point (m : ℝ) : ∃ x y, (∀ m, y = m * x + (2 * m + 1)) ↔ (x = -2 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_line_fixed_point_l503_50372


namespace NUMINAMATH_GPT_line_equation_unique_l503_50327

theorem line_equation_unique (m b k : ℝ) (h_intersect_dist : |(k^2 + 6*k + 5) - (m*k + b)| = 7)
  (h_passing_point : 8 = 2*m + b) (hb_nonzero : b ≠ 0) :
  y = 10*x - 12 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_unique_l503_50327


namespace NUMINAMATH_GPT_car_speed_correct_l503_50374

noncomputable def car_speed (d v_bike t_delay : ℝ) (h1 : v_bike > 0) (h2 : t_delay > 0): ℝ := 2 * v_bike

theorem car_speed_correct:
  ∀ (d v_bike : ℝ) (t_delay : ℝ) (h1 : v_bike > 0) (h2 : t_delay > 0),
    (d / v_bike - t_delay = d / (car_speed d v_bike t_delay h1 h2)) → 
    car_speed d v_bike t_delay h1 h2 = 0.6 :=
by
  intros
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_car_speed_correct_l503_50374


namespace NUMINAMATH_GPT_intersection_A_complement_B_l503_50390

def set_A : Set ℝ := {x | 1 < x ∧ x < 4}
def set_B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_Complement_B : Set ℝ := {x | x < -1 ∨ x > 3}
def set_Intersection : Set ℝ := {x | set_A x ∧ set_Complement_B x}

theorem intersection_A_complement_B : set_Intersection = {x | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_GPT_intersection_A_complement_B_l503_50390


namespace NUMINAMATH_GPT_other_bill_denomination_l503_50319

-- Define the conditions of the problem
def cost_shirt : ℕ := 80
def ten_dollar_bills : ℕ := 2
def other_bills (x : ℕ) : ℕ := ten_dollar_bills + 1

-- The amount paid with $10 bills
def amount_with_ten_dollar_bills : ℕ := ten_dollar_bills * 10

-- The total amount should match the cost of the shirt
def total_amount (x : ℕ) : ℕ := amount_with_ten_dollar_bills + (other_bills x) * x

-- Statement to prove
theorem other_bill_denomination : 
  ∃ (x : ℕ), total_amount x = cost_shirt ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_other_bill_denomination_l503_50319


namespace NUMINAMATH_GPT_middle_number_of_five_consecutive_numbers_l503_50353

theorem middle_number_of_five_consecutive_numbers (n : ℕ) 
  (h : (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 60) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_middle_number_of_five_consecutive_numbers_l503_50353


namespace NUMINAMATH_GPT_smallest_root_equation_l503_50306

theorem smallest_root_equation :
  ∃ x : ℝ, (3 * x) / (x - 2) + (2 * x^2 - 28) / x = 11 ∧ ∀ y, (3 * y) / (y - 2) + (2 * y^2 - 28) / y = 11 → x ≤ y ∧ x = (-1 - Real.sqrt 17) / 2 :=
sorry

end NUMINAMATH_GPT_smallest_root_equation_l503_50306


namespace NUMINAMATH_GPT_parity_of_f_l503_50357

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ :=
  x * (x - 2) * (x - 1) * x * (x + 1) * (x + 2)

theorem parity_of_f :
  is_even_function f ∧ ¬ (∃ g : ℝ → ℝ, g = f ∧ (∀ x : ℝ, g (-x) = -g x)) :=
by
  sorry

end NUMINAMATH_GPT_parity_of_f_l503_50357


namespace NUMINAMATH_GPT_find_x_l503_50356

theorem find_x (x : ℕ) (h1 : x % 6 = 0) (h2 : x^2 > 144) (h3 : x < 30) : x = 18 ∨ x = 24 :=
sorry

end NUMINAMATH_GPT_find_x_l503_50356


namespace NUMINAMATH_GPT_sunlovers_happy_days_l503_50384

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end NUMINAMATH_GPT_sunlovers_happy_days_l503_50384


namespace NUMINAMATH_GPT_smallest_perimeter_iso_triangle_l503_50348

theorem smallest_perimeter_iso_triangle :
  ∃ (x y : ℕ), (PQ = PR ∧ PQ = x ∧ PR = x ∧ QR = y ∧ QJ = 10 ∧ PQ + PR + QR = 416 ∧ 
  PQ = PR ∧ y = 8 ∧ 2 * (x + y) = 416 ∧ y^2 - 50 > 0 ∧ y < 10) :=
sorry

end NUMINAMATH_GPT_smallest_perimeter_iso_triangle_l503_50348


namespace NUMINAMATH_GPT_eulers_polyhedron_theorem_l503_50395

theorem eulers_polyhedron_theorem 
  (V E F t h : ℕ) (T H : ℕ) :
  (F = 30) →
  (t = 20) →
  (h = 10) →
  (T = 3) →
  (H = 2) →
  (E = (3 * t + 6 * h) / 2) →
  (V - E + F = 2) →
  100 * H + 10 * T + V = 262 :=
by
  intros F_eq t_eq h_eq T_eq H_eq E_eq euler_eq
  rw [F_eq, t_eq, h_eq, T_eq, H_eq, E_eq] at *
  sorry

end NUMINAMATH_GPT_eulers_polyhedron_theorem_l503_50395


namespace NUMINAMATH_GPT_angle_of_inclination_l503_50378

theorem angle_of_inclination (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, 3)) : 
  ∃ θ : ℝ, θ = (3 * Real.pi) / 4 ∧ (∃ k : ℝ, k = (A.2 - B.2) / (A.1 - B.1) ∧ Real.tan θ = k) :=
by
  sorry

end NUMINAMATH_GPT_angle_of_inclination_l503_50378


namespace NUMINAMATH_GPT_order_of_products_l503_50330

theorem order_of_products (x a b : ℝ) (h1 : x < a) (h2 : a < b) (h3 : b < 0) : b * x > a * x ∧ a * x > a ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_order_of_products_l503_50330


namespace NUMINAMATH_GPT_new_cube_weight_l503_50385

-- Define the weight function for a cube given side length and density.
def weight (ρ : ℝ) (s : ℝ) : ℝ := ρ * s^3

-- Given conditions: the weight of the original cube.
axiom original_weight : ∃ ρ s : ℝ, weight ρ s = 7

-- The goal is to prove that a new cube with sides twice as long weighs 56 pounds.
theorem new_cube_weight : 
  (∃ ρ s : ℝ, weight ρ (2 * s) = 56) := by
  sorry

end NUMINAMATH_GPT_new_cube_weight_l503_50385


namespace NUMINAMATH_GPT_translation_result_l503_50312

variables (P : ℝ × ℝ) (P' : ℝ × ℝ)

def translate_left (P : ℝ × ℝ) (units : ℝ) := (P.1 - units, P.2)
def translate_down (P : ℝ × ℝ) (units : ℝ) := (P.1, P.2 - units)

theorem translation_result :
    P = (-4, 3) -> P' = translate_down (translate_left P 2) 2 -> P' = (-6, 1) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_translation_result_l503_50312


namespace NUMINAMATH_GPT_find_c_and_d_l503_50359

theorem find_c_and_d (c d : ℝ) (h : ℝ → ℝ) (f : ℝ → ℝ) (finv : ℝ → ℝ) 
  (h_def : ∀ x, h x = 6 * x - 5)
  (finv_eq : ∀ x, finv x = 6 * x - 3)
  (f_def : ∀ x, f x = c * x + d)
  (inv_prop : ∀ x, f (finv x) = x ∧ finv (f x) = x) :
  4 * c + 6 * d = 11 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_c_and_d_l503_50359


namespace NUMINAMATH_GPT_germany_fraction_closest_japan_fraction_closest_l503_50388

noncomputable def fraction_approx (a b : ℕ) : ℚ := a / b

theorem germany_fraction_closest :
  abs (fraction_approx 23 150 - fraction_approx 1 7) < 
  min (abs (fraction_approx 23 150 - fraction_approx 1 5))
      (min (abs (fraction_approx 23 150 - fraction_approx 1 6))
           (min (abs (fraction_approx 23 150 - fraction_approx 1 8))
                (abs (fraction_approx 23 150 - fraction_approx 1 9)))) :=
by sorry

theorem japan_fraction_closest :
  abs (fraction_approx 27 150 - fraction_approx 1 6) < 
  min (abs (fraction_approx 27 150 - fraction_approx 1 5))
      (min (abs (fraction_approx 27 150 - fraction_approx 1 7))
           (min (abs (fraction_approx 27 150 - fraction_approx 1 8))
                (abs (fraction_approx 27 150 - fraction_approx 1 9)))) :=
by sorry

end NUMINAMATH_GPT_germany_fraction_closest_japan_fraction_closest_l503_50388


namespace NUMINAMATH_GPT_greatest_least_S_T_l503_50301

theorem greatest_least_S_T (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) (triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  4 ≤ (a + b + c)^2 / (b * c) ∧ (a + b + c)^2 / (b * c) ≤ 9 :=
by sorry

end NUMINAMATH_GPT_greatest_least_S_T_l503_50301


namespace NUMINAMATH_GPT_program_output_l503_50338

theorem program_output :
  let a := 1
  let b := 3
  let a := a + b
  let b := b * a
  a = 4 ∧ b = 12 :=
by
  sorry

end NUMINAMATH_GPT_program_output_l503_50338


namespace NUMINAMATH_GPT_points_on_circle_l503_50335

theorem points_on_circle (t : ℝ) :
  let x := (t^3 - 1) / (t^3 + 1);
  let y := (2 * t^3) / (t^3 + 1);
  x^2 + y^2 = 1 :=
by
  let x := (t^3 - 1) / (t^3 + 1)
  let y := (2 * t^3) / (t^3 + 1)
  have h1 : x^2 + y^2 = ((t^3 - 1) / (t^3 + 1))^2 + ((2 * t^3) / (t^3 + 1))^2 := by rfl
  have h2 : (x^2 + y^2) = ( (t^3 - 1)^2 + (2 * t^3)^2 ) / (t^3 + 1)^2 := by sorry
  have h3 : (x^2 + y^2) = ( t^6 - 2 * t^3 + 1 + 4 * t^6 ) / (t^3 + 1)^2 := by sorry
  have h4 : (x^2 + y^2) = 1 := by sorry
  exact h4

end NUMINAMATH_GPT_points_on_circle_l503_50335


namespace NUMINAMATH_GPT_smallest_part_is_correct_l503_50321

-- Conditions
def total_value : ℕ := 360
def proportion1 : ℕ := 5
def proportion2 : ℕ := 7
def proportion3 : ℕ := 4
def proportion4 : ℕ := 8
def total_parts := proportion1 + proportion2 + proportion3 + proportion4
def value_per_part := total_value / total_parts
def smallest_proportion : ℕ := proportion3

-- Theorem to prove
theorem smallest_part_is_correct : value_per_part * smallest_proportion = 60 := by
  dsimp [total_value, total_parts, value_per_part, smallest_proportion]
  norm_num
  sorry

end NUMINAMATH_GPT_smallest_part_is_correct_l503_50321


namespace NUMINAMATH_GPT_parallel_lines_implies_m_no_perpendicular_lines_solution_l503_50352

noncomputable def parallel_slopes (m : ℝ) : Prop :=
  let y₁ := -m
  let y₂ := -2 / m
  y₁ = y₂

noncomputable def perpendicular_slopes (m : ℝ) : Prop :=
  let y₁ := -m
  let y₂ := -2 / m
  y₁ * y₂ = -1

theorem parallel_lines_implies_m (m : ℝ) : parallel_slopes m ↔ m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
by
  sorry

theorem no_perpendicular_lines_solution (m : ℝ) : perpendicular_slopes m → false :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_implies_m_no_perpendicular_lines_solution_l503_50352


namespace NUMINAMATH_GPT_c_amount_correct_b_share_correct_l503_50343

-- Conditions
def total_sum : ℝ := 246    -- Total sum of money
def c_share : ℝ := 48      -- C's share in Rs
def c_per_rs : ℝ := 0.40   -- C's amount per Rs

-- Expressing the given condition c_share = total sum * c_per_rs
theorem c_amount_correct : c_share = total_sum * c_per_rs := 
  by
  -- Substitute that can be more elaboration of the calculations done
  sorry

-- Additional condition for the total per Rs distribution
axiom a_b_c_total : ∀ (a b : ℝ), a + b + c_per_rs = 1

-- Proving B's share per Rs is approximately 0.4049
theorem b_share_correct : ∃ a b : ℝ, c_share = 246 * 0.40 ∧ a + b + 0.40 = 1 ∧ b = 1 - (48 / 246) - 0.40 := 
  by
  -- Substitute that can be elaboration of the proof arguments done in the translated form
  sorry

end NUMINAMATH_GPT_c_amount_correct_b_share_correct_l503_50343


namespace NUMINAMATH_GPT_variable_cost_per_book_fixed_cost_l503_50333

theorem variable_cost_per_book_fixed_cost (fixed_costs : ℝ) (selling_price_per_book : ℝ) 
(number_of_books : ℝ) (total_costs total_revenue : ℝ) (variable_cost_per_book : ℝ) 
(h1 : fixed_costs = 35630) (h2 : selling_price_per_book = 20.25) (h3 : number_of_books = 4072)
(h4 : total_costs = fixed_costs + variable_cost_per_book * number_of_books)
(h5 : total_revenue = selling_price_per_book * number_of_books)
(h6 : total_costs = total_revenue) : variable_cost_per_book = 11.50 := by
  sorry

end NUMINAMATH_GPT_variable_cost_per_book_fixed_cost_l503_50333


namespace NUMINAMATH_GPT_interest_rate_difference_l503_50377

theorem interest_rate_difference (P T : ℝ) (R1 R2 : ℝ) (I_diff : ℝ) (hP : P = 2100) 
  (hT : T = 3) (hI : I_diff = 63) :
  R2 - R1 = 0.01 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_difference_l503_50377


namespace NUMINAMATH_GPT_possible_values_of_a_l503_50326

theorem possible_values_of_a (a : ℝ) :
  (∃ x, ∀ y, (y = x) ↔ (a * y^2 + 2 * y + a = 0))
  → (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l503_50326


namespace NUMINAMATH_GPT_Carol_mother_carrots_l503_50397

theorem Carol_mother_carrots (carol_picked : ℕ) (total_good : ℕ) (total_bad : ℕ) (total_carrots : ℕ) (mother_picked : ℕ) :
  carol_picked = 29 → total_good = 38 → total_bad = 7 → total_carrots = total_good + total_bad → mother_picked = total_carrots - carol_picked → mother_picked = 16 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end NUMINAMATH_GPT_Carol_mother_carrots_l503_50397


namespace NUMINAMATH_GPT_top_card_is_club_probability_l503_50373

-- Definitions based on the conditions
def deck_size := 52
def suit_count := 4
def cards_per_suit := deck_size / suit_count

-- The question we want to prove
theorem top_card_is_club_probability :
  (13 : ℝ) / (52 : ℝ) = 1 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_top_card_is_club_probability_l503_50373


namespace NUMINAMATH_GPT_Cornelia_three_times_Kilee_l503_50396

variable (x : ℕ)

def Kilee_current_age : ℕ := 20
def Cornelia_current_age : ℕ := 80

theorem Cornelia_three_times_Kilee (x : ℕ) :
  Cornelia_current_age + x = 3 * (Kilee_current_age + x) ↔ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_Cornelia_three_times_Kilee_l503_50396


namespace NUMINAMATH_GPT_min_fraction_sum_is_15_l503_50310

theorem min_fraction_sum_is_15
  (A B C D : ℕ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
  (h_nonzero_int : ∃ k : ℤ, k ≠ 0 ∧ (A + B : ℤ) = k * (C + D))
  : C + D = 15 :=
sorry

end NUMINAMATH_GPT_min_fraction_sum_is_15_l503_50310


namespace NUMINAMATH_GPT_complete_work_together_in_days_l503_50371

noncomputable def a_days := 16
noncomputable def b_days := 6
noncomputable def c_days := 12

noncomputable def work_rate (days: ℕ) : ℚ := 1 / days

theorem complete_work_together_in_days :
  let combined_rate := (work_rate a_days) + (work_rate b_days) + (work_rate c_days)
  let days_to_complete := 1 / combined_rate
  days_to_complete = 3.2 :=
  sorry

end NUMINAMATH_GPT_complete_work_together_in_days_l503_50371
