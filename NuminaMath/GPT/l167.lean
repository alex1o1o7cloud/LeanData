import Mathlib

namespace compare_f_values_l167_167844

noncomputable def f : ℝ → ℝ := sorry

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_monotonically_decreasing_on_nonnegative (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → x1 < x2 → f x2 < f x1

axiom even_property : is_even_function f
axiom decreasing_property : is_monotonically_decreasing_on_nonnegative f

theorem compare_f_values : f 3 < f (-2) ∧ f (-2) < f 1 :=
by {
  sorry
}

end compare_f_values_l167_167844


namespace rhombus_perimeter_l167_167039

theorem rhombus_perimeter {a b c : ℝ} (h_eq : a * a - 14 * a + 48 = 0) (h_eq': b * b - 14 * b + 48 = 0) (ac_bd_pos : a ≠ 0 ∧ b ≠ 0):
  (4 * (Real.sqrt ((a / 2) ^ 2 + (b / 2) ^ 2))) = 20 :=
by
  have ac_bd : a = 8 ∧ b = 6 ∨ a = 6 ∧ b = 8 := sorry -- roots calculation
  cases ac_bd with h1 h2
  · rw [h1.1, h1.2]
    have d := Real.sqrt ((8 / 2) ^ 2 + (6 / 2) ^ 2)
    -- calculate distance using Pythagorean theorem
    rw [Real.sqrt_eq_iff_sq_eq] at d
    rw [d]
    norm_num
  · rw [h2.1, h2.2]
    have d := Real.sqrt ((6 / 2) ^ 2 + (8 / 2) ^ 2)
    -- calculate distance using Pythagorean theorem
    rw [Real.sqrt_eq_iff_sq_eq] at d
    rw [d]
    norm_num

end rhombus_perimeter_l167_167039


namespace problem_equivalent_l167_167548

noncomputable def f : ℚ+ → ℝ := sorry

theorem problem_equivalent (f : ℚ+ → ℝ)
  (h1 : ∀ x y : ℚ+, f x * f y ≥ f (x * y))
  (h2 : ∀ x y : ℚ+, f (x + y) ≥ f x + f y)
  (h3 : ∃ a : ℚ+, a > 1 ∧ f a = a) :
  ∀ x : ℚ+, f x = x := sorry

end problem_equivalent_l167_167548


namespace find_monotonic_bijections_l167_167132

variable {f : ℝ → ℝ}

-- Define the properties of the function f
def bijective (f : ℝ → ℝ) : Prop :=
  Function.Bijective f

def condition (f : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, f t + f (f t) = 2 * t

theorem find_monotonic_bijections (f : ℝ → ℝ) (hf_bij : bijective f) (hf_cond : condition f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end find_monotonic_bijections_l167_167132


namespace reflection_matrix_correct_l167_167538

noncomputable def reflection_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1/3, 4/3, 2/3], ![5/3, 1/3, 2/3], ![1/3, 4/3, 2/3]]

theorem reflection_matrix_correct (u : Fin 3 → ℝ) :
  let n : Fin 3 → ℝ := ![2, -1, 1]
  let q := u - (u ⬝ᴛ n) / (n ⬝ᴛ n) • n
  let s := 2 • q - u
  reflection_matrix ⬝ u = s :=
by
  let n : Fin 3 → ℝ := ![2, -1, 1]
  have h_n_dot_n : n ⬝ᴛ n = 6 := sorry
  let q := u - (u ⬝ᴛ n) / (n ⬝ᴛ n) • n
  let s := 2 • q - u
  have h_projection : reflection_matrix ⬝ u = s := sorry
  exact h_projection
  sorry

#check @reflection_matrix_correct

end reflection_matrix_correct_l167_167538


namespace period_of_y_l167_167658

noncomputable def y (x : ℝ) := Real.tan (2 * x) + Real.cot (2 * x)

theorem period_of_y : ∀ x, y (x) = y (x + π / 2) :=
by
  intro x
  -- We know that y(x) can be simplified to 2 / sin(4x)
  have h1 : y x = 2 / Real.sin (4 * x) := sorry
  -- Now we need to show that y(x) = y(x + π / 2)
  -- Note that sin(4 (x + π / 8)) = sin(4x + π / 2) = cos(4x)
  have h2 : Real.sin (4 * (x + π / 8)) = Real.cos (4 * x) := sorry
  -- Use h2 to show that y x = y (x + π / 2)
  rw [h1]
  rw [h1 (x + π / 2)]
  rw [h2]
  sorry

end period_of_y_l167_167658


namespace solve_inequality_l167_167175

theorem solve_inequality (a : ℝ) :
  let inequality := λ x : ℝ, a * x^2 + (2 - a) * x - 2 < 0 in
  (a = 0 → {x : ℝ | x < 1} = {x | inequality x}) ∧
  (-2 < a ∧ a < 0 → {x : ℝ | x < 1 ∨ x > -2 / a} = {x | inequality x}) ∧
  (a = -2 → {x : ℝ | x ≠ 1} = {x | inequality x}) ∧
  (a < -2 → {x : ℝ | x < -2 / a ∨ x > 1} = {x | inequality x}) ∧
  (a > 0 → {x : ℝ | -2 / a < x ∧ x < 1} = {x | inequality x}) :=
begin
  sorry
end

end solve_inequality_l167_167175


namespace expected_value_of_X_is_7_l167_167709

-- Define the random variable X that always outputs 7
noncomputable def X : ℝ := 7

-- Define the probability function for X
def P (x : ℝ) : ℝ := if x = 7 then 1 else 0

-- Define the expected value function
noncomputable def E (X : ℝ) (P : ℝ → ℝ) : ℝ := ∑ (x : ℝ) in {X}, x * P x

-- The theorem to prove the expected value of X is 7
theorem expected_value_of_X_is_7 : E X P = 7 :=
by
  sorry

end expected_value_of_X_is_7_l167_167709


namespace find_line_equation_for_given_area_l167_167766

theorem find_line_equation_for_given_area :
  ∃ b : ℝ, 
    (let line1 := (λ x, (3/4 : ℝ) * x + b),
       line2 := (λ x, (3/4 : ℝ) * x - b) in
     ((∀ x, y = line1 x → y - (3/4 : ℝ) * x = b) ∨ 
      (∀ x, y = line2 x → y - (3/4 : ℝ) * x = -b)) ∧ 
     (1 / 2 * abs( - (4 / 3) * b) * abs( b ) = 6)
    ) :=
begin
  sorry
end

end find_line_equation_for_given_area_l167_167766


namespace shaded_area_square_with_circles_l167_167408

-- Defining the conditions
def side_length : ℝ := 8
def radius : ℝ := 3

/-
The goal is to prove that the shaded area is equal to 
64 - 16 * real.sqrt 7 - 18 * real.arcsin (real.sqrt 7 / 4)
-/
theorem shaded_area_square_with_circles :
  let A_shaded := side_length^2 - 4 * (2 * real.sqrt(4^2 - radius^2)) - 2 * (4 * radius^2 * real.arcsin(real.sqrt(4^2 - radius^2) / 4)) / π
  A_shaded = 64 - 16 * real.sqrt 7 - 18 * real.arcsin (real.sqrt 7 / 4) := 
by {
  sorry
}

end shaded_area_square_with_circles_l167_167408


namespace log_base_3_l167_167353

theorem log_base_3 (h : (1 / 81 : ℝ) = 3 ^ (-4 : ℝ)) : Real.logBase 3 (1 / 81) = -4 := 
by sorry

end log_base_3_l167_167353


namespace problem_statement_l167_167908

def atOp (a b : ℝ) := a * b ^ (1 / 2)

theorem problem_statement : atOp ((2 * 3) ^ 2) ((3 * 5) ^ 2 / 9) = 180 := by
  sorry

end problem_statement_l167_167908


namespace profit_percentage_is_25_l167_167305

variable (CP SP Profit ProfitPercentage : ℝ)
variable (hCP : CP = 192)
variable (hSP : SP = 240)
variable (hProfit : Profit = SP - CP)
variable (hProfitPercentage : ProfitPercentage = (Profit / CP) * 100)

theorem profit_percentage_is_25 :
  hCP → hSP → hProfit → hProfitPercentage → ProfitPercentage = 25 := by
  intros hCP hSP hProfit hProfitPercentage
  sorry

end profit_percentage_is_25_l167_167305


namespace solve_equation_l167_167371

theorem solve_equation (x : ℝ) : x^4 + (4 - x)^4 = 272 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 := 
by sorry

end solve_equation_l167_167371


namespace domain_ln_neg_x_plus_1_l167_167150

theorem domain_ln_neg_x_plus_1 :
  ∀ x : ℝ, (f x = ln (-x + 1)) → ((-x + 1 > 0) ↔ (x < 1)) :=
begin
  sorry
end

end domain_ln_neg_x_plus_1_l167_167150


namespace problem_1_problem_2_l167_167046

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |x + 1|

theorem problem_1 : {x : ℝ | f x < 4} = {x : ℝ | -4 / 3 < x ∧ x < 4 / 3} :=
by 
  sorry

theorem problem_2 (x₀ : ℝ) (h : ∀ t : ℝ, f x₀ < |m + t| + |t - m|) : 
  {m : ℝ | ∃ x t, f x < |m + t| + |t - m|} = {m : ℝ | m < -3 / 4 ∨ m > 3 / 4} :=
by 
  sorry

end problem_1_problem_2_l167_167046


namespace log_base_3_l167_167351

theorem log_base_3 (h : (1 / 81 : ℝ) = 3 ^ (-4 : ℝ)) : Real.logBase 3 (1 / 81) = -4 := 
by sorry

end log_base_3_l167_167351


namespace union_eq_l167_167016

def A : Set ℤ := {-1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

theorem union_eq : A ∪ B = {-1, 0, 1, 2, 3} := 
by 
  sorry

end union_eq_l167_167016


namespace smallest_n_partitions_l167_167378

theorem smallest_n_partitions (n : ℕ) :
  (∀ A B : finset ℕ, A ∪ B = finset.range (n + 1) ∧ A ∩ B = ∅ →
  (∃ a b c ∈ A, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c) ∨
  (∃ a b c ∈ B, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c)) ↔ n = 96 :=
sorry

end smallest_n_partitions_l167_167378


namespace distinct_three_digit_even_integers_count_l167_167466

theorem distinct_three_digit_even_integers_count : 
  let even_digits := {0, 2, 4, 6, 8}
  ∃ h : Finset ℕ, h = {2, 4, 6, 8} ∧ 
     (∏ x in h, 5 * 5) = 100 :=
by
  let even_digits : Finset ℕ := {0, 2, 4, 6, 8}
  let h : Finset ℕ := {2, 4, 6, 8}
  have : ∏ x in h, 5 * 5 = 100 := sorry
  exact ⟨h, rfl, this⟩

end distinct_three_digit_even_integers_count_l167_167466


namespace number_of_bags_needed_l167_167885

-- Definitions for the conditions:
def red_jellybeans_in_bag : ℕ := 24
def white_jellybeans_in_bag : ℕ := 18
def total_guess_jellybeans : ℕ := 126

-- Lean statement where we prove the number of bags needed is 3
theorem number_of_bags_needed : 
  ((red_jellybeans_in_bag + white_jellybeans_in_bag) * 3 = total_guess_jellybeans) := 
by 
  calc
    (red_jellybeans_in_bag + white_jellybeans_in_bag) * 3 = (24 + 18) * 3 : by rw [red_jellybeans_in_bag, white_jellybeans_in_bag]
    ... = 42 * 3 : by norm_num
    ... = total_guess_jellybeans : by rw total_guess_jellybeans

end number_of_bags_needed_l167_167885


namespace tinplates_to_match_l167_167645

theorem tinplates_to_match (x : ℕ) (y : ℕ) (total_tinplates : ℕ) 
    (h1 : total_tinplates = 36)
    (h2 : 25 * x = 40 * (total_tinplates - x) / 2) :
    x = 16 ∧ total_tinplates - x = 20 :=
by
  have eq1 : total_tinplates - x = 20, sorry
  have eq2 : x = 16, sorry
  exact ⟨eq2, eq1⟩

end tinplates_to_match_l167_167645


namespace inequality_holds_l167_167925

theorem inequality_holds (k n : ℕ) (x : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ 1) :
  (1 - (1 - x)^n)^k ≥ 1 - (1 - x^k)^n :=
by
  sorry

end inequality_holds_l167_167925


namespace number_of_triplets_l167_167412

theorem number_of_triplets (N : ℕ) (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 2017 ≥ 10 * a) (h5 : 10 * a ≥ 100 * b) (h6 : 100 * b ≥ 1000 * c) : 
  N = 574 := 
sorry

end number_of_triplets_l167_167412


namespace smallest_n_digits_l167_167545

theorem smallest_n_digits (n : ℕ) (h1 : 18 ∣ n) (h2 : ∃ k : ℕ, n^2 = k^3) (h3 : ∃ m : ℕ, n^3 = m^2) : 
  nat.digits 10 n = 8 :=
sorry

end smallest_n_digits_l167_167545


namespace sequence_sum_eq_one_over_36_l167_167541

def b (n : ℕ) : ℕ
| 0     := 2
| 1     := 3
| (n+2) := b n + b (n + 1)

theorem sequence_sum_eq_one_over_36 : 
  (∑ n in Finset.range (100), (b n) / (9 ^ (n + 1))) = 1 / 36 := 
sorry

end sequence_sum_eq_one_over_36_l167_167541


namespace integral_div_f_prime_diverges_l167_167135

noncomputable def f : ℝ → ℝ := sorry -- We define f as a placeholder.

theorem integral_div_f_prime_diverges 
  (h₁ : ∀ x > 1, f x ≤ x^2 * real.log x)
  (h₂ : ∀ x > 1, 0 < deriv f x) :
  ∫ x in 1..∞, 1 / deriv f x = ∞ :=
by {
  sorry
}

end integral_div_f_prime_diverges_l167_167135


namespace true_propositions_l167_167822

-- Definitions of the conditions

def prop1 (lines1 lines2 : Type) (plane1 plane2 : Type) : Prop :=
  ∀ (l1 l2 : lines1) (l3 l4 : lines2), (l1 ⊆ plane1) ∧ (l2 ⊆ plane1) ∧ (l1 ∥ l3) ∧ (l2 ∥ l4) → plane1 ∥ plane2

def prop2 (line : Type) (plane1 plane2 : Type) : Prop :=
  ∀ (m : line), (∀ a ∈ plane1, a ⊥ m) → (∀ a ∈ plane2, a ⊥ m)

def prop3 (line : Type) (plane1 plane2 : Type) : Prop :=
  ∀ (m : line), (∀ a ∈ plane1, a ⊥ m) ∧ (plane1 ⊥ plane2) → m ∥ plane2

def prop4 (line intersectionLine : Type) (plane1 plane2 : Type) : Prop :=
  ∀ (l : line) (intL : intersectionLine), (plane1 ⊥ plane2) ∧ (l ⊆ plane1) ∧ (¬ (l ⊥ intL)) → (¬ (l ⊥ plane2))

-- The theorem to prove Propositions ② and ④ are true
theorem true_propositions : prop2 line plane1 plane2 ∧ prop4 line intersectionLine plane1 plane2 := 
  by
  sorry

end true_propositions_l167_167822


namespace total_handshakes_l167_167724

def people := 40
def groupA := 25
def groupB := 15
def knownByGroupB (x : ℕ) : ℕ := 5
def interactionsWithinGroupB : ℕ := 105
def interactionsBetweenGroups : ℕ := 75

theorem total_handshakes : (groupB * knownByGroupB 0) + interactionsWithinGroupB = 180 :=
by
  sorry

end total_handshakes_l167_167724


namespace sum_of_powers_pattern_l167_167329

theorem sum_of_powers_pattern :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 5^32 + 7^32 :=
  sorry

end sum_of_powers_pattern_l167_167329


namespace integral_of_circle_l167_167726

theorem integral_of_circle (a : ℂ) (R : ℝ) (n : ℤ) :
  ∫ (z : ℂ) in (circleIntegral ℂ (fun z => 1 / (z - a) ^ n) a R), (circle_integral (1 / (z - a) ^ n)) = 
  if n = 1 then 2 * Complex.pi * Complex.i else 0 :=
sorry

end integral_of_circle_l167_167726


namespace inequality_solution_l167_167176

theorem inequality_solution (x : ℝ) :
  2 * real.sqrt((4 * x - 9) ^ 2) + real.sqrt(3 * real.sqrt(x) - 5 + 2 * |x - 2|) ≤ 18 - 8 * x ↔ x = 0 :=
by
  sorry

end inequality_solution_l167_167176


namespace even_digit_numbers_count_eq_100_l167_167453

-- Definition for the count of distinct three-digit positive integers with only even digits
def count_even_digit_three_numbers : ℕ :=
  let hundreds_place := {2, 4, 6, 8}.card
  let tens_units_place := {0, 2, 4, 6, 8}.card
  hundreds_place * tens_units_place * tens_units_place

-- Theorem stating the count of distinct three-digit positive integers with only even digits is 100
theorem even_digit_numbers_count_eq_100 : count_even_digit_three_numbers = 100 :=
by sorry

end even_digit_numbers_count_eq_100_l167_167453


namespace santa_gifts_distribution_l167_167581

theorem santa_gifts_distribution :
  let bags := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_gifts := 36
  count_subsets_divisible_by_8 (bags : Finset ℕ) (H_distinct : bags.card = 8)
  (H_sum : bags.sum = total_gifts) = 31 :=
  sorry

end santa_gifts_distribution_l167_167581


namespace tangent_parabola_line_l167_167047

theorem tangent_parabola_line (a x₀ y₀ : ℝ) 
  (h_line : x₀ - y₀ - 1 = 0)
  (h_parabola : y₀ = a * x₀^2)
  (h_tangent_slope : 2 * a * x₀ = 1) : 
  a = 1 / 4 :=
sorry

end tangent_parabola_line_l167_167047


namespace eiffel_tower_model_height_l167_167253

theorem eiffel_tower_model_height 
  (H1 : ℝ) (W1 : ℝ) (W2 : ℝ) (H2 : ℝ)
  (h1 : H1 = 324)
  (w1 : W1 = 8000000)  -- converted 8000 tons to 8000000 kg
  (w2 : W2 = 1)
  (h_eq : (H2 / H1)^3 = W2 / W1) : 
  H2 = 1.62 :=
by
  rw [h1, w1, w2] at h_eq
  sorry

end eiffel_tower_model_height_l167_167253


namespace drama_club_students_l167_167084

theorem drama_club_students (
  total_students : ℕ,
  math_students : ℕ,
  physics_students : ℕ,
  both_students : ℕ
) : total_students = 60 ∧ math_students = 40 ∧ physics_students = 35 ∧ both_students = 25 →
    total_students - (math_students - both_students + physics_students - both_students + both_students) = 10 := by
  intro h
  cases h with ht h
  cases h with hm h
  cases h with hp hb
  sorry

end drama_club_students_l167_167084


namespace problem_B_decreasing_l167_167783

def decreasing_on_interval (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f y < f x

theorem problem_B_decreasing (a : ℝ) (ha : a > 0) : decreasing_on_interval (λ x, x^2 - 2 * a * x + 1) {x | 0 < x ∧ x < a} :=
sorry

end problem_B_decreasing_l167_167783


namespace bus_on_time_at_least_two_days_probability_l167_167814

namespace ProbabilityProof

def bus_on_time_probability_each_day : ℚ := 3 / 5
def days_commuting : ℕ := 3

theorem bus_on_time_at_least_two_days_probability :
  (binomial 3 2) * (bus_on_time_probability_each_day ^ 2) * ((1 - bus_on_time_probability_each_day) ^ 1) + 
  (binomial 3 3) * (bus_on_time_probability_each_day ^ 3) = 81 / 125 := 
by 
  sorry

end ProbabilityProof

end bus_on_time_at_least_two_days_probability_l167_167814


namespace evaluate_exponent_expression_l167_167758

theorem evaluate_exponent_expression (a m n : ℚ) (h_a : a ≠ 0) (h_m : m = 4) (h_n : n = -4) :
  (a ^ m) * (a ^ n) = 1 := by
  have h_exp : m + n = 0 := by
    rw [h_m, h_n]
    norm_num
  have h_pow_zero : a ^ (m + n) = 1 := by
    rw [h_exp]
    exact pow_zero h_a
  rw [← mul_assoc, h_pow_zero]
  exact one_mul _

-- Applying the specific values of a, m, and n
example : (5/6 : ℚ) ^ 4 * (5/6 : ℚ) ^ (-4) = 1 :=
  evaluate_exponent_expression (5/6) 4 (-4) (by norm_num) rfl rfl

end evaluate_exponent_expression_l167_167758


namespace solution_of_inequality_l167_167204

theorem solution_of_inequality (x : ℝ) : (sqrt (x^2 - 2 * x + 1) > 2 * x) ↔ (x < 1 / 3) := by sorry

end solution_of_inequality_l167_167204


namespace total_wood_carvings_l167_167592

theorem total_wood_carvings (carvings_per_shelf : Nat) (shelves_filled : Nat) : carvings_per_shelf = 8 → shelves_filled = 7 → carvings_per_shelf * shelves_filled = 56 :=
by
  intro h1 h2
  rw [h1, h2]
  norm_num

end total_wood_carvings_l167_167592


namespace sq_length_QP_eq_245_l167_167509

-- Definitions for the given conditions
def radius1 : ℝ := 10
def radius2 : ℝ := 7
def center_distance : ℝ := 15
def cos_angle : ℝ := -19 / 35

-- Proof statement
theorem sq_length_QP_eq_245 (x : ℝ) (h1 : chord_length_eq (radius1 * radius1 - 2 * radius1 * x * cos_angle) (radius2 * radius2 - 2 * radius2 * x * cos_angle)) :
  x^2 = 245 :=
sorry

-- Definition to express the chord lengths being equal
def chord_length_eq (v1 : ℝ) (v2 : ℝ) : Prop :=
v1 = v2

end sq_length_QP_eq_245_l167_167509


namespace max_min_S_l167_167917

-- Define the expression S
def S (x y z : ℝ) : ℝ := 2 * x^2 * y^2 + 2 * x^2 * z^2 + 2 * y^2 * z^2 - x^4 - y^4 - z^4

-- Define the conditions for x, y, z
def valid_range (x y z : ℝ) : Prop := 5 ≤ x ∧ x ≤ 8 ∧ 5 ≤ y ∧ y ≤ 8 ∧ 5 ≤ z ∧ z ≤ 8

-- State the theorem to prove maximum and minimum values of S
theorem max_min_S : 
  ∃ x y z : ℝ, valid_range x y z ∧ S x y z = 4096 ∧ 
  ∃ a b c : ℝ, valid_range a b c ∧ S a b c = -375 := 
begin 
  sorry 
end

end max_min_S_l167_167917


namespace circle_M_equation_line_GH_fixed_point_l167_167872

noncomputable def center_on_curve (x : ℝ) : ℝ :=
sqrt(3) / x

def circle_eq (x y cx cy r : ℝ) : Prop :=
(x - cx)^2 + (y - cy)^2 = r^2

def line_eq (x y k b : ℝ) : Prop :=
y = k * x + b

theorem circle_M_equation
  (cx cy r : ℝ)
  (h_center_curve : cy = center_on_curve cx)
  (h_origin : circle_eq 0 0 cx cy r)
  (h_line : ∀ x, circle_eq x (- sqrt(3) / 3 * x + 4) cx cy r → | x | = r ) :
  (x-1)^2 + (y-sqrt(3))^2 = 4 :=
sorry

theorem line_GH_fixed_point
  (cx cy : ℝ)
  (h_center_curve : cy = center_on_curve cx)
  (h_origin : circle_eq 0 0 cx cy 2)
  (h_intersect : ∀ x y,
    y = sqrt(3) →
    ∃ E F G H P, 
    line_eq x y 0 5 →
    | G - E | = | P - E | ∧
    | H - F | = | P - E | ∧
    (x-2) ∈ [ - sqrt(3) .. sqrt(3) ] ) :
  ∃ x y, x = 2 ∧ y = sqrt(3) :=
sorry

end circle_M_equation_line_GH_fixed_point_l167_167872


namespace class_student_difference_l167_167634

theorem class_student_difference (A B : ℕ) (h : A - 4 = B + 4) : A - B = 8 := by
  sorry

end class_student_difference_l167_167634


namespace krakozyabrs_count_l167_167111

theorem krakozyabrs_count :
  ∀ (K : Type) [fintype K] (has_horns : K → Prop) (has_wings : K → Prop), 
  (∀ k, has_horns k ∨ has_wings k) →
  (∃ n, (∀ k, has_horns k → has_wings k) → fintype.card ({k | has_horns k} : set K) = 5 * n) →
  (∃ n, (∀ k, has_wings k → has_horns k) → fintype.card ({k | has_wings k} : set K) = 4 * n) →
  (∃ T, 25 < T ∧ T < 35 ∧ T = 32) := 
by
  intros K _ has_horns has_wings _ _ _
  sorry

end krakozyabrs_count_l167_167111


namespace lines_intersection_l167_167613

theorem lines_intersection (m : ℝ) (x y : ℝ) :
  (3 * 4 - 2 * y = m) ∧ (-4 - 2 * y = -10) → m = 6 :=
by
  intro h
  cases h with h1 h2
  sorry

end lines_intersection_l167_167613


namespace max_gcd_l167_167311

theorem max_gcd (n : ℕ) (h : 0 < n) : ∀ n, ∃ d ≥ 1, d ∣ 13 * n + 4 ∧ d ∣ 8 * n + 3 → d ≤ 9 :=
begin
  sorry
end

end max_gcd_l167_167311


namespace gain_percent_is_40_l167_167168

def purchase_price : ℝ := 800
def repair_costs : ℝ := 200
def selling_price : ℝ := 1400

def total_cost := purchase_price + repair_costs
def gain := selling_price - total_cost
def gain_percent := (gain / total_cost) * 100

theorem gain_percent_is_40 : gain_percent = 40 := 
by
  -- add proof here
  sorry

end gain_percent_is_40_l167_167168


namespace cos_alpha_value_l167_167029

theorem cos_alpha_value (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 3 / 5) :
  Real.cos α = 4 / 5 :=
by
  sorry

end cos_alpha_value_l167_167029


namespace reflection_matrix_correct_l167_167530

-- Given conditions
def normal_vector : EuclideanSpace ℝ (Fin 3) := ![2, -1, 1]

-- Define the reflection matrix
def reflection_matrix := ![
  ![-1/3, -1/3, 1/3],
  ![1/3, 2/3, -1/3],
  ![-5/3, 5/3, 2/3]
]

-- Problem statement: Proving that this matrix correctly reflects any vector through the plane with the given normal vector.
theorem reflection_matrix_correct (u : EuclideanSpace ℝ (Fin 3)) :
  let S := reflection_matrix,
      Q := normal_vector
  in ∀ u, S • u = u - 2 * ((u ⬝ Q) / (Q ⬝ Q)) • Q :=
sorry

end reflection_matrix_correct_l167_167530


namespace slower_train_cross_time_l167_167644

-- Definitions based on given conditions
def speed_train1 : ℝ := 60        -- Speed of slower train in kmph
def speed_train2 : ℝ := 90        -- Speed of faster train in kmph
def length_train1 : ℝ := 1.10     -- Length of slower train in km
def length_train2 : ℝ := 0.90     -- Length of faster train in km

-- Conversion factor and relative speed
def kmph_to_mps (v : ℝ) : ℝ := v * (5 / 18)
def relative_speed : ℝ := kmph_to_mps (speed_train1 + speed_train2)

-- Total length in meters
def total_length_m : ℝ := (length_train1 + length_train2) * 1000

-- Time to cross in seconds
noncomputable def time_to_cross := total_length_m / relative_speed

-- Lean statement to prove
theorem slower_train_cross_time : time_to_cross = 48 := by
  -- Proof goes here
  sorry

end slower_train_cross_time_l167_167644


namespace max_2ab_plus_2bc_sqrt2_l167_167540

theorem max_2ab_plus_2bc_sqrt2 (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt 3 :=
sorry

end max_2ab_plus_2bc_sqrt2_l167_167540


namespace total_weight_is_1kg_total_weight_in_kg_eq_1_l167_167282

theorem total_weight_is_1kg 
  (weight_msg : ℕ := 80)
  (weight_salt : ℕ := 500)
  (weight_detergent : ℕ := 420) :
  (weight_msg + weight_salt + weight_detergent) = 1000 := by
sorry

theorem total_weight_in_kg_eq_1 
  (total_weight_g : ℕ := weight_msg + weight_salt + weight_detergent) :
  (total_weight_g = 1000) → (total_weight_g / 1000 = 1) := by
sorry

end total_weight_is_1kg_total_weight_in_kg_eq_1_l167_167282


namespace last_score_is_65_l167_167501

-- Define the scores and the problem conditions
def scores := [65, 72, 75, 80, 85, 88, 92]
def total_sum := 557
def remaining_sum (score : ℕ) : ℕ := total_sum - score

-- Define a property to check divisibility
def divisible_by (n d : ℕ) : Prop := n % d = 0

-- The main theorem statement
theorem last_score_is_65 :
  (∀ s ∈ scores, divisible_by (remaining_sum s) 6) ∧ divisible_by total_sum 7 ↔ scores = [65, 72, 75, 80, 85, 88, 92] :=
sorry

end last_score_is_65_l167_167501


namespace constant_term_expansion_eq_70_l167_167093

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem: Prove the constant term in the expansion of (x - 1/x)^8 is 70
theorem constant_term_expansion_eq_70 : 
  (∃ (r : ℕ), r ≤ 8 ∧ (8 - 2 * r = 0) ∧ binom 8 r * (-1) ^ r = 70) :=
by
  sorry

end constant_term_expansion_eq_70_l167_167093


namespace collinear_vectors_x_value_l167_167782

theorem collinear_vectors_x_value :
  ∀ (x : ℝ), (∃ k : ℝ, (1, 2) = k • (x, 1)) → x = 1 / 2 :=
by
  intro x
  intro hx
  -- skipping the proof
  sorry

end collinear_vectors_x_value_l167_167782


namespace find_p_for_arithmetic_sequence_l167_167067

theorem find_p_for_arithmetic_sequence :
  ∃ (p : ℝ), (∀ x, polynomial.eval x (polynomial.C 1 * x^3 - polynomial.C (6 * p) * x^2 + polynomial.C (5 * p) * x + polynomial.C 88) = 0) →
  p = 2 :=
sorry

end find_p_for_arithmetic_sequence_l167_167067


namespace fixed_circle_center_sphere_l167_167891

def midpoint (A B : Point) : Point := sorry

def skew_lines (L M : Line) : Prop := sorry

variables (L M : Line)
variables (A B P Q O C : Point)

-- Define conditions
axiom cond1 : skew_lines L M
axiom cond2 : A ∈ L
axiom cond3 : B ∈ M
axiom cond4 : (AB ⊥ L)
axiom cond5 : (AB ⊥ M)
axiom cond6 : P ∈ L
axiom cond7 : Q ∈ M
axiom cond8 : (PQ has_constant_length)
axiom cond9 : P ≠ A
axiom cond10 : Q ≠ B
axiom def_O : O = midpoint A B

theorem fixed_circle_center_sphere :
  center_of_sphere_through A B P Q ∈ circle_centered_at O :=
sorry

end fixed_circle_center_sphere_l167_167891


namespace decomposition_addition_l167_167390

theorem decomposition_addition (m p : ℕ) 
  (hm : m^2 = 1 + 3 + 5 + 7 + 9 + 11) 
  (hp : ∃ (n : ℕ), p^3 = 21 + n * 2 + ∑ (k : ℕ) in (range n).succ, 2 * k + 1) :
  m + p = 11 := 
sorry

end decomposition_addition_l167_167390


namespace woman_waits_time_after_passing_l167_167280

-- Definitions based only on the conditions in a)
def man_speed : ℝ := 5 -- in miles per hour
def woman_speed : ℝ := 25 -- in miles per hour
def waiting_time_man_minutes : ℝ := 20 -- in minutes

-- Equivalent proof problem statement
theorem woman_waits_time_after_passing :
  let waiting_time_man_hours := waiting_time_man_minutes / 60
  let distance_man : ℝ := man_speed * waiting_time_man_hours
  let relative_speed : ℝ := woman_speed - man_speed
  let time_woman_covers_distance_hours := distance_man / relative_speed
  let time_woman_covers_distance_minutes := time_woman_covers_distance_hours * 60
  time_woman_covers_distance_minutes = 5 :=
by
  sorry

end woman_waits_time_after_passing_l167_167280


namespace point_N_in_second_quadrant_l167_167921

theorem point_N_in_second_quadrant (a b : ℝ) (h1 : 1 + a < 0) (h2 : 2 * b - 1 < 0) :
    (a - 1 < 0) ∧ (1 - 2 * b > 0) :=
by
  -- Insert proof here
  sorry

end point_N_in_second_quadrant_l167_167921


namespace infinite_sequences_B_intersect_with_A_infinite_l167_167557

theorem infinite_sequences_B_intersect_with_A_infinite (A B : ℕ → ℕ) (d : ℕ) :
  (∀ n, A n = 5 * n - 2) ∧ (∀ k, B k = k * d + 7 - d) →
  (∃ d, ∀ m, ∃ k, A m = B k) :=
by
  sorry

end infinite_sequences_B_intersect_with_A_infinite_l167_167557


namespace exists_special_quadrilateral_l167_167979

-- Define the convex 2550-gon with the specified vertex coloring.
structure ConvexNGon (n : ℕ) :=
  (vertices : Fin n → Color)
  (is_convex : Convex vertices)
  (coloring_pattern : ∀ i : Fin n, coloring_rule i vertices(i))

-- Define the coloring pattern as a specific repetition sequence.
inductive Color
  | Black
  | White

def coloring_rule (i : ℕ) (c : Color) : Prop :=
  if i % 101 < 1275 then
    if (i % 101) % 2 == 0 then c = Color.Black else c = Color.White
  else
    if (i % 101) % 2 == 0 then c = Color.White else c = Color.Black

-- Define the divide-into-quadrilateral property.
def divides_into_quadrilaterals (g : ConvexNGon 2550) : Prop := sorry

-- Prove the statement.
theorem exists_special_quadrilateral :
  ∀ (g : ConvexNGon 2550), divides_into_quadrilaterals g →
    ∃ (quad : Quadrilateral), adjacent_black_white quad :=
by
  intro g h
  sorry

-- Define quadrilateral and adjacency properties.
structure Quadrilateral :=
  (v1 v2 v3 v4 : Fin 2550)

def adjacent_black_white (quad : Quadrilateral) : Prop :=
  ((g.vertices quad.v1 = Color.Black ∧ g.vertices quad.v2 = Color.Black) 
  ∧ (g.vertices quad.v3 = Color.White ∧ g.vertices quad.v4 = Color.White))
  ∨ ((g.vertices quad.v1 = Color.White ∧ g.vertices quad.v2 = Color.White)
  ∧ (g.vertices quad.v3 = Color.Black ∧ g.vertices quad.v4 = Color.Black))

end exists_special_quadrilateral_l167_167979


namespace outcome_sum_eq_4_l167_167637

-- Define the outcomes for the given conditions
def outcome_A : Prop := ∑ = 4 ∧ ((one die shows 3 points ∧ the other die shows 1 point) ∨ (one die shows 1 point ∧ the other die shows 3 points))
def outcome_B : Prop := ∑ = 4 ∧ (both dice show 2 points)
def outcome_C : Prop := ∑ = 4 ∧ (both dice show 4 points)
def outcome_D : Prop := ∑ = 4 ∧ ((one die shows 3 points ∧ the other die shows 1 point) ∨ (one die shows 1 point ∧ the other die shows 3 points) ∨ (both dice show 2 points))

-- Lean statement for the problem
theorem outcome_sum_eq_4 :
  ∑ = 4 → outcome_A ∨ outcome_B ∨ outcome_D :=
sorry

end outcome_sum_eq_4_l167_167637


namespace six_points_six_lines_configuration_l167_167915

-- Problem: Mark 6 distinct points on a plane and draw 6 lines 
-- such that each line has exactly 2 marked points on it, 
-- and there are 2 marked points on each side of every line.

theorem six_points_six_lines_configuration :
  ∃ (P : Fin 6 → Point) (L : Fin 6 → Line),
  (∀ i, (L i).contains_two_points (P 0) (P 1) ∧ others_conditions P L) :=
sorry

end six_points_six_lines_configuration_l167_167915


namespace rectangle_is_both_axisymmetric_and_centrally_symmetric_l167_167237

-- Define the properties of shapes

-- Definitions for axisymmetry and central symmetry.
def axisymmetric (shape : Type) : Prop := sorry -- Define axisymmetry
def centrally_symmetric (shape : Type) : Prop := sorry -- Define central symmetry

-- Shapes as Types
inductive Shape
| EquilateralTriangle
| Parallelogram
| Rectangle
| RegularPentagon

open Shape

-- Define the property of each shape
def is_axisymmetric : Shape → Prop
| EquilateralTriangle := true
| Parallelogram       := false
| Rectangle           := true
| RegularPentagon     := true

def is_centrally_symmetric : Shape → Prop
| EquilateralTriangle := false
| Parallelogram       := true
| Rectangle           := true
| RegularPentagon     := false

-- The theorem we want to prove
theorem rectangle_is_both_axisymmetric_and_centrally_symmetric :
  ∀ s, (is_axisymmetric s ∧ is_centrally_symmetric s) ↔ (s = Rectangle) :=
sorry

end rectangle_is_both_axisymmetric_and_centrally_symmetric_l167_167237


namespace find_length_of_side_c_find_measure_of_angle_B_l167_167514

variable {A B C a b c : ℝ}

def triangle_problem (a b c A B C : ℝ) :=
  a * Real.cos B = 3 ∧
  b * Real.cos A = 1 ∧
  A - B = Real.pi / 6 ∧
  a^2 + c^2 - b^2 - 6 * c = 0 ∧
  b^2 + c^2 - a^2 - 2 * c = 0

theorem find_length_of_side_c (h : triangle_problem a b c A B C) :
  c = 4 :=
sorry

theorem find_measure_of_angle_B (h : triangle_problem a b c A B C) :
  B = Real.pi / 6 :=
sorry

end find_length_of_side_c_find_measure_of_angle_B_l167_167514


namespace product_inequality_l167_167553

variable (n : ℕ)
variable (a : Fin n → ℝ) (σ : Equiv.Perm (Fin n))

theorem product_inequality (h_nonneg : ∀ i, 0 ≤ a i) :
  (∏ i, (a i)^2 + a (σ i)) ≥ (∏ i, (a i)^2 + a i) :=
sorry

end product_inequality_l167_167553


namespace sequence_value_x_l167_167875

theorem sequence_value_x (x : ℕ) (h1 : 1 + 3 = 4) (h2 : 4 + 3 = 7) (h3 : 7 + 3 = 10) (h4 : 10 + 3 = x) (h5 : x + 3 = 16) : x = 13 := by
  sorry

end sequence_value_x_l167_167875


namespace log3_1_over_81_l167_167360

theorem log3_1_over_81 : log 3 (1 / 81) = -4 := by
  have h1 : 1 / 81 = 3 ^ (-4) := by
    -- provide a proof or skip with "sory"
    sorry
  have h2 : log 3 (3 ^ (-4)) = -4 := by
    -- provide a proof or skip with "sorry"
    sorry
  exact eq.trans (log 3) (congr_fun (h1.symm h2))

end log3_1_over_81_l167_167360


namespace radius_circumscribed_sphere_l167_167603

-- Define a pyramid with the specified properties
def equilateral_triangle_side (a : ℝ) : Prop :=
∀ (P Q R : ℝ × ℝ), P ≠ Q ∧ Q ≠ R ∧ R ≠ P → (dist P Q = a ∧ dist Q R = a ∧ dist R P = a)

def perpendicular_lateral_edge (b : ℝ) (O A : ℝ × ℝ × ℝ) : Prop :=
O.2 = 0 ∧ dist O A = b ∧ A.2 = √(a^2 - ((1/2) * a)^2)

-- Define the radius of the circumscribed sphere
def radius_of_circumscribed_sphere (a b : ℝ) : ℝ :=
√((3 * a^2 + 4 * b^2) / 12)

-- State the theorem to be proved
theorem radius_circumscribed_sphere
  (a b : ℝ)
  (P Q R : ℝ × ℝ)
  (O : ℝ × ℝ × ℝ)
  (h1 : equilateral_triangle_side a P Q R)
  (h2 : perpendicular_lateral_edge b O (P.1, P.2)):
  radius_of_circumscribed_sphere a b = √((3 * a^2 + 4 * b^2) / 12) :=
sorry

end radius_circumscribed_sphere_l167_167603


namespace point_C_locations_l167_167576

noncomputable def AB_distance : Real := 10
noncomputable def triangle_area : Real := 20

theorem point_C_locations :
  ∃ (A B C : ℝ × ℝ),
    dist A B = AB_distance ∧
    ∃ (h : ℝ), 1 / 2 * AB_distance * h = triangle_area ∧
    (∃ (x : Real), (C = (x, 4) ∨ C = (x, -4)) ∧
    (dist A B = 10) ∧ (1 / 2 * (10 * h) = 20)) ∧
    (list.count [C] = 8) :=
by
  sorry

end point_C_locations_l167_167576


namespace find_prime_p_l167_167761

theorem find_prime_p (p : ℕ) (hp : nat.prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) → (p = 2 ∨ p = 3) :=
by
  sorry

end find_prime_p_l167_167761


namespace unique_x0_implies_a_in_range_l167_167826

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x * (3 * x - 1) - a * x + a

theorem unique_x0_implies_a_in_range :
  ∃ x0 : ℤ, f x0 a ≤ 0 ∧ a < 1 -> a ∈ Set.Ico (2 / Real.exp 1) 1 := 
sorry

end unique_x0_implies_a_in_range_l167_167826


namespace ticket_cost_l167_167293

theorem ticket_cost 
  (V G : ℕ)
  (h1 : V + G = 320)
  (h2 : V = G - 212) :
  40 * V + 15 * G = 6150 := 
by
  sorry

end ticket_cost_l167_167293


namespace max_value_y_l167_167608

noncomputable def y (x : ℝ) : ℝ := 3 * x - x^3

theorem max_value_y : 
  ∃ x0 ∈ set.Ioi (0 : ℝ), (∀ x ∈ set.Ioi (0 : ℝ), y x ≤ y x0) ∧ y x0 = 2 :=
sorry

end max_value_y_l167_167608


namespace compute_ab_val_l167_167609

variables (a b : ℝ)

theorem compute_ab_val
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 868.5 :=
sorry

end compute_ab_val_l167_167609


namespace ratio_of_shaded_area_l167_167215

/-- Define the geometric setup of the problem. -/
structure Triangle :=
(A B C D E F G H : Point)
(ratio : ℚ)

axiom right_isosceles {T : Triangle} : T.AB = T.BC
axiom trisection_points_df {T : Triangle} : (T.AD = T.DF) ∧ (T.DF = T.FB) ∧ (T.DF = T.AB / 3)
axiom midpoint_e {T : Triangle} : T.E = midpoint T.B T.C
axiom midpoint_g {T : Triangle} : T.G = midpoint T.D T.E
axiom midpoint_h {T : Triangle} : T.H = midpoint T.F T.E

/-- The proof goal is to show the required ratio of shaded to non-shaded areas. -/
theorem ratio_of_shaded_area {T : Triangle} :
  triangle.right_isosceles →
  triangle.trisection_points_df →
  triangle.midpoint_e →
  triangle.midpoint_g →
  triangle.midpoint_h →
  T.ratio = 25 / 47 :=
sorry

end ratio_of_shaded_area_l167_167215


namespace tangent_fixed_circle_l167_167922

variables {α : Type*} [MetricSpace α] [Plane α]
variables (A B C D M N I1 I2 : α)
variables (ω : Circle)
variables (hA : A ∈ ω) (hB : B ∈ ω)
variables (hC : C ∈ Arc A B) (hD : D ∈ Arc A B)
variables (hCD_const : ∀ C D ∈ Arc A B, dist C D = dist C D)
variables (hI1 : IsIncenter (Triangle.mk A B C) I1)
variables (hI2 : IsIncenter (Triangle.mk A B D) I2)

theorem tangent_fixed_circle :
    ∃ γ : Circle, ∀ C D ∈ Arc A B, dist C D = dist C D →
    IsTangentLine (Line.mk I1 I2) γ :=
sorry

end tangent_fixed_circle_l167_167922


namespace solution_set_for_f_x_l167_167427

variables {f : ℝ → ℝ}

theorem solution_set_for_f_x (h₀ : f(1) = 1) (h₁ : ∀ x, f' x < 1 / 2) :
  ∀ x, f x < x / 2 + 1 / 2 ↔ x > 1 :=
sorry

end solution_set_for_f_x_l167_167427


namespace arithmetic_sequence_a8_l167_167091

noncomputable def a_n (n : ℕ) : ℕ := 1 + (n - 1) * 3

theorem arithmetic_sequence_a8 :
  let a1 := 1
  let S5 := 35
  let d := 3
  (a_5 : ℕ := (1 + 2 + 3 + 4 + 5) * 3 / 5)
  in
  (1 + 7 * d = 22) :=
by
  sorry

end arithmetic_sequence_a8_l167_167091


namespace inflation_two_years_correct_real_rate_of_return_correct_l167_167667

-- Define the calculation for inflation over two years
def inflation_two_years (r : ℝ) : ℝ :=
  ((1 + r)^2 - 1) * 100

-- Define the calculation for the real rate of return
def real_rate_of_return (r : ℝ) (infl_rate : ℝ) : ℝ :=
  ((1 + r * r) / (1 + infl_rate / 100) - 1) * 100

-- Prove the inflation over two years is 3.0225%
theorem inflation_two_years_correct :
  inflation_two_years 0.015 = 3.0225 :=
by
  sorry

-- Prove the real yield of the bank deposit is 11.13%
theorem real_rate_of_return_correct :
  real_rate_of_return 0.07 3.0225 = 11.13 :=
by
  sorry

end inflation_two_years_correct_real_rate_of_return_correct_l167_167667


namespace correct_length_of_AB_l167_167869

noncomputable def length_AB (AB AC BC CD DE CE : ℝ) : Prop :=
  ∃ AB AC BC CD DE CE,
    -- Conditions from the problem
    is_isosceles_triangle ABC ↔ (AC = BC) ∧
    is_isosceles_triangle CDE ↔ (CD = DE) ∧
    perimeter AC BC CD = 24 ∧
    perimeter CD DE CE = 22 ∧
    CE = 9 ∧
    -- Given final correct answer
    AB = 11

-- You might define helper functions or inline assumptions here (if needed)
-- However, the exact definitions (e.g., is_isosceles_triangle, perimeter) should be handled correctly in Lean according to library definitions.

theorem correct_length_of_AB : 
  (∃ AB AC BC CD DE CE : ℝ, 
    (is_isosceles_triangle ABC → AC = BC) ∧ 
    (is_isosceles_triangle CDE → CD = DE) ∧ 
    (perimeter_triangle ABC 24) ∧ 
    (perimeter_triangle CDE 22) ∧ 
    (CE = 9) ∧ 
    AB = 11) :=
sorry

end correct_length_of_AB_l167_167869


namespace how_many_more_apples_suraya_picked_than_mia_l167_167563

def surayaPickedMore (Kayla Caleb Suraya Mia: ℕ): Prop :=
  (2 * Caleb = Kayla - 5) ∧
  (Suraya = 3 * Caleb) ∧
  (Mia = 2 * Caleb) ∧
  (Kayla = 20) ∧
  (Suraya - Mia = 5)

theorem how_many_more_apples_suraya_picked_than_mia :
  ∃ Kayla Caleb Suraya Mia, surayaPickedMore Kayla Caleb Suraya Mia :=
by
  have Kayla := 20
  have Caleb := (Kayla / 2) - 5
  have Suraya := 3 * Caleb
  have Mia := 2 * Caleb
  use [Kayla, Caleb, Suraya, Mia]
  sorry

end how_many_more_apples_suraya_picked_than_mia_l167_167563


namespace range_of_a_l167_167138

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 2 > 0) → a > 9 / 8 :=
by
  sorry

end range_of_a_l167_167138


namespace Carlos_candy_share_l167_167159

theorem Carlos_candy_share (total_candy : ℚ) (num_piles : ℕ) (piles_for_Carlos : ℕ)
  (h_total_candy : total_candy = 75 / 7)
  (h_num_piles : num_piles = 5)
  (h_piles_for_Carlos : piles_for_Carlos = 2) :
  (piles_for_Carlos * (total_candy / num_piles) = 30 / 7) :=
by
  sorry

end Carlos_candy_share_l167_167159


namespace percentage_increase_l167_167853

-- defining the given values
def Z := 150
def total := 555
def x_from_y (Y : ℝ) := 1.25 * Y

-- defining the condition that x gets 25% more than y and z out of 555 is Rs. 150
def condition1 (X Y : ℝ) := X = x_from_y Y
def condition2 (X Y : ℝ) := X + Y + Z = total

-- theorem to prove
theorem percentage_increase (Y : ℝ) :
  condition1 (x_from_y Y) Y →
  condition2 (x_from_y Y) Y →
  ((Y - Z) / Z) * 100 = 20 :=
by
  sorry

end percentage_increase_l167_167853


namespace fixed_point_of_fn_l167_167192

theorem fixed_point_of_fn (n : ℕ) : (∃ c : ℝ, f c = c) :=
by
  let f : ℝ → ℝ := λ x, x ^ n + 1
  have h : f 1 = 1 ^ n + 1 := rfl
  rw [pow_one] at h
  exact h⟩
s sorry

end fixed_point_of_fn_l167_167192


namespace estimated_probability_l167_167811

def is_scored (digit : ℕ) : Bool :=
  digit = 1 ∨ digit = 2 ∨ digit = 3 ∨ digit = 4

def count_succ_shots (group : List ℕ) : ℕ :=
  group.countp is_scored

def successful_groups (groups : List (List ℕ)) : List (List ℕ) :=
  groups.filter (fun group => count_succ_shots group = 2)

def probability_of_success (groups : List (List ℕ)) : ℚ :=
  (successful_groups groups).length / groups.length

def groups : List (List ℕ) :=
  [[9, 0, 7], [9, 6, 6], [1, 9, 1], [9, 2, 5], [2, 7, 1], 
   [9, 3, 2], [8, 1, 2], [4, 5, 8], [5, 6, 9], [6, 8, 3], 
   [4, 3, 1], [2, 5, 7], [3, 9, 3], [0, 2, 7], [5, 5, 6], 
   [4, 8, 8], [7, 3, 0], [1, 1, 3], [5, 3, 7], [9, 8, 9]]

theorem estimated_probability :
  probability_of_success groups = 1 / 4 :=
  by
    sorry

end estimated_probability_l167_167811


namespace simplify_expression_l167_167172

theorem simplify_expression : 
  (√6 / √10) * (√5 / √15) * (√8 / √14) = (2 * √35) / 35 := 
by
  sorry

end simplify_expression_l167_167172


namespace fraction_pow_zero_l167_167651

theorem fraction_pow_zero :
  (4310000 / -21550000 : ℝ) ≠ 0 →
  (4310000 / -21550000 : ℝ) ^ 0 = 1 :=
by
  intro h
  sorry

end fraction_pow_zero_l167_167651


namespace fixed_point_l167_167823

-- Define the function f(x) = 4 + a^(x - 1) with a > 0 and a ≠ 1
def f (a x : ℝ) := 4 + a^(x - 1)

-- State the fixed point condition
theorem fixed_point (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : f a 1 = 5 :=
by
  unfold f
  have : a ^ (1 - 1) = 1 := by norm_num
  rw [this]
  norm_num
  sorry

end fixed_point_l167_167823


namespace factor_expression_l167_167370

theorem factor_expression (x : ℝ) : 
  3 * x * (x - 5) + 4 * (x - 5) = (3 * x + 4) * (x - 5) :=
by
  sorry

end factor_expression_l167_167370


namespace lindsey_money_left_l167_167912

-- Conditions as definitions
def savings_september := 50
def savings_october := 37
def savings_november := 11
def savings_december := savings_november + 0.1 * savings_november
def total_savings := savings_september + savings_october + savings_november + savings_december
def additional_from_mom := 0.2 * total_savings
def total_with_mom := total_savings + additional_from_mom
def spent_on_bundle := 0.75 * total_with_mom

-- The amount of money Lindsey has left after buying the video game bundle
def money_left := total_with_mom - spent_on_bundle

-- The theorem statement
theorem lindsey_money_left : money_left = 33.03 := by
  sorry

end lindsey_money_left_l167_167912


namespace cos_double_angle_l167_167032

-- Define terms and conditions
def is_acute_angle (A : ℝ) : Prop := A > 0 ∧ A < π / 2
def satisfies_equation (A : ℝ) : Prop := 3 * Real.cos A - 8 * Real.tan A = 0

-- The statement to prove
theorem cos_double_angle (A : ℝ) (h₁ : is_acute_angle A) (h₂ : satisfies_equation A) : Real.cos (2 * A) = 7 / 9 :=
sorry

end cos_double_angle_l167_167032


namespace color_2011_is_white_l167_167201

-- Define the colors as an inductive type
inductive Color
| black | white

open Color

-- Define a function that assigns a color to each integer
def color : ℤ → Color

-- Define the conditions
axiom color_one_is_white : color 1 = white
axiom color_diff_sum : ∀ (a b : ℤ), color a = white → color b = white → color (a - b) ≠ color (a + b)

-- Formulate the theorem that needs to be proven
theorem color_2011_is_white : color 2011 = white :=
sorry

end color_2011_is_white_l167_167201


namespace triangle_properties_l167_167833

open Real

/-- For a triangle ABC with sides a, b, and c opposite to angles A, B, and C,
    satisfying 2a * cos A = c * cos B + b * cos C and circumradius R = 2:
    1. A = π/3
    2. If b^2 + c^2 = 18, then the area of triangle ABC is 3√3/2 -/
theorem triangle_properties
  (a b c A B C R : ℝ)
  (h1 : 2 * a * cos A = c * cos B + b * cos C)
  (h2 : R = 2)
  (h3 : b^2 + c^2 = 18) :
  A = π / 3 ∧ (sqrt (3 / 4) * 6 = 3 * sqrt(3) / 2) :=
  by sorry

end triangle_properties_l167_167833


namespace intersection_A_B_l167_167484

def set_A (x : ℝ) : Prop := 2 * x + 1 > 0
def set_B (x : ℝ) : Prop := abs (x - 1) < 2

theorem intersection_A_B : 
  {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | -1/2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l167_167484


namespace period_of_y_l167_167657

noncomputable def y (x : ℝ) := Real.tan (2 * x) + Real.cot (2 * x)

theorem period_of_y : ∀ x, y (x) = y (x + π / 2) :=
by
  intro x
  -- We know that y(x) can be simplified to 2 / sin(4x)
  have h1 : y x = 2 / Real.sin (4 * x) := sorry
  -- Now we need to show that y(x) = y(x + π / 2)
  -- Note that sin(4 (x + π / 8)) = sin(4x + π / 2) = cos(4x)
  have h2 : Real.sin (4 * (x + π / 8)) = Real.cos (4 * x) := sorry
  -- Use h2 to show that y x = y (x + π / 2)
  rw [h1]
  rw [h1 (x + π / 2)]
  rw [h2]
  sorry

end period_of_y_l167_167657


namespace choose_three_cards_of_different_suits_l167_167475

/-- The number of ways to choose 3 cards from a standard deck of 52 cards,
if all three cards must be of different suits -/
theorem choose_three_cards_of_different_suits :
  let n := 4
  let r := 3
  let suits_combinations := Nat.choose n r
  let cards_per_suit := 13
  let total_ways := suits_combinations * (cards_per_suit ^ r)
  total_ways = 8788 :=
by
  sorry

end choose_three_cards_of_different_suits_l167_167475


namespace max_value_of_expression_l167_167940

variables (x y : ℝ)

theorem max_value_of_expression (hx : 0 < x) (hy : 0 < y) (h : x^2 - 2*x*y + 3*y^2 = 12) : x^2 + 2*x*y + 3*y^2 ≤ 24 + 24*sqrt 3 :=
sorry

end max_value_of_expression_l167_167940


namespace complement_of_angle_correct_l167_167818

noncomputable def complement_of_angle (α : ℝ) : ℝ := 90 - α

theorem complement_of_angle_correct (α : ℝ) (h : complement_of_angle α = 125 + 12 / 60) :
  complement_of_angle α = 35 + 12 / 60 :=
by
  sorry

end complement_of_angle_correct_l167_167818


namespace total_slices_sold_l167_167209

theorem total_slices_sold (sold_yesterday served_today : ℕ) (h1 : sold_yesterday = 5) (h2 : served_today = 2) :
  sold_yesterday + served_today = 7 :=
by
  -- Proof skipped
  exact sorry

end total_slices_sold_l167_167209


namespace f_is_even_l167_167142

variable (g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h x

def is_periodic (h : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, h (x + p) = h x

def is_even (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = h x

def f (x : ℝ) : ℝ := abs (g (x^5))

theorem f_is_even (hg_odd : is_odd g) (hg_periodic : ∃ p, is_periodic g p) : is_even f :=
  sorry

end f_is_even_l167_167142


namespace min_segment_length_l167_167101

theorem min_segment_length 
  (angle : ℝ) (P : ℝ × ℝ)
  (dist_x : ℝ) (dist_y : ℝ) 
  (hx : P.1 ≤ dist_x ∧ P.2 = dist_y)
  (hy : P.2 ≤ dist_y ∧ P.1 = dist_x)
  (right_angle : angle = 90) 
  : ∃ (d : ℝ), d = 10 :=
by
  sorry

end min_segment_length_l167_167101


namespace age_ratio_l167_167964

variable (V A : ℕ)

def condition1 : Prop := V - 5 = 16
def condition2 : Prop := (V + 4) * 2 = (A + 4) * 5

theorem age_ratio (h1 : condition1 V) (h2 : condition2 V A) : V = 21 ∧ A = 6 ∧ V / nat.gcd V A = 7 ∧ A / nat.gcd V A = 2 := by
  sorry

end age_ratio_l167_167964


namespace number_of_valid_pairs_l167_167506

open Nat

theorem number_of_valid_pairs :
  let S := {2, 3, 4, 5, 6, 7, 8, 9}
  let even_set := {2, 4, 6, 8}
  let odd_set := {3, 5, 7, 9}
  S.card = 8 ∧ even_set.card = 4 ∧ odd_set.card = 4 →
  gcd (x, y) ≠ 2 → 
  ∃! f : even_set → odd_set, 
    function.bijective f →
    (∑ (e : even_set), (some (f e)).val / e = 36) :=
begin
  sorry
end

end number_of_valid_pairs_l167_167506


namespace concurrency_of_lines_l167_167525

variable {A B C X Y Z A1 A2 B1 B2 C1 C2 : Point}
variable {ABC : Triangle}
variable {l_A l_B l_C : Line}

-- Definitions of the problem conditions
def acute_triangle (ABC : Triangle) : Prop := 
  ∠A < 90 ∧ ∠B < 90 ∧ ∠C < 90

def altitude (A B C X : Point) : Prop := 
  ∠BAX = 90 ∧ ∠CAX = 90

def line_parallel (l : Line) (m : Line) : Prop := 
  parallel l m

-- Given conditions
axiom H1 : acute_triangle ABC
axiom H2 : altitude A B C X
axiom H3 : altitude B C A Y
axiom H4 : altitude C A B Z
axiom H5 : line_parallel l_A (YZ : Line)
axiom H6 : l_A.intersects (CA : Line) A1
axiom H7 : l_A.intersects (AB : Line) A2
axiom H8 : line_parallel l_B (ZX : Line)
axiom H9 : l_B.intersects (AB : Line) B1
axiom H10 : l_B.intersects (BC : Line) B2
axiom H11 : line_parallel l_C (XY : Line)
axiom H12 : l_C.intersects (BC : Line) C1
axiom H13 : l_C.intersects (CA : Line) C2
axiom H14 : perimeter (Triangle A A1 A2) = length CA + length AB
axiom H15 : perimeter (Triangle B B1 B2) = length AB + length BC
axiom H16 : perimeter (Triangle C C1 C2) = length BC + length CA

-- Proof goal
theorem concurrency_of_lines : concurrent l_A l_B l_C :=
sorry

end concurrency_of_lines_l167_167525


namespace combined_weight_is_170_l167_167482

namespace weight_problem

variable (K : ℝ) -- Define Kendra's current weight as a variable
variable (L : ℝ) -- Define Leo's current weight as a variable
variable (combined_weight : ℝ) -- Define their combined current weight

-- Conditions
axiom leo_current_weight : L = 98
axiom leo_gain_weight_condition : L + 10 = 1.5 * K

-- Theorem statement: Their combined current weight is 170 pounds
theorem combined_weight_is_170 : combined_weight = (L + K) → combined_weight = 170 :=
begin
  intros h,
  subst h,
  sorry
end

end weight_problem

end combined_weight_is_170_l167_167482


namespace f_neg_2_eq_neg_4_f_x_expression_for_neg_x_g_t_min_value_l167_167037

-- Conditions
def f (x : ℝ) : ℝ := 
  if x >= 0 then x^2 - 4 * x 
  else x^2 + 4 * x

def g (t : ℝ) : ℝ :=
  if 1 < t ∧ t <= 2 then (t - 1)^2 - 4 * (t - 1)
  else (t + 1)^2 - 4 * (t + 1)

-- Proof goals
theorem f_neg_2_eq_neg_4 : f (-2) = -4 := sorry

theorem f_x_expression_for_neg_x (x : ℝ) (h : x < 0) : f x = x^2 + 4 * x := sorry

theorem g_t_min_value (t : ℝ) (h : t > 1) : 
  g(t) = if t ≤ 2 then t^2 - 6 * t + 5 else t^2 - 2 * t - 3 ∧ g t = -3 := sorry

end f_neg_2_eq_neg_4_f_x_expression_for_neg_x_g_t_min_value_l167_167037


namespace number_of_possible_values_of_M_minus_m_l167_167493

def isRed (n : ℤ) : Prop := n ≥ 3 ∧ n ≤ 7
def isGreen (n : ℤ) : Prop := n ≥ 3 ∧ n ≤ 7

theorem number_of_possible_values_of_M_minus_m : 
  (∃ M m : ℤ, isRed M ∧ isGreen m ∧ ∀ k : ℤ, (k = M - m) → k ∈ (set.range (λ i : ℕ, -4 + i)) ∧ (set.range (λ i : ℕ, -4 + i)).card = 9)
:=
sorry

end number_of_possible_values_of_M_minus_m_l167_167493


namespace anna_always_wins_l167_167572

def proper_divisor (n k : ℕ) : Prop :=
  k < n ∧ n % k = 0

def move (count : ℕ) : ℕ → Prop :=
  λ d, proper_divisor count d

theorem anna_always_wins :
  ∃ strategy : (ℕ → ℕ) → (_, _), 
    ∀ count ≤ 2024, 
    (∀ d, proper_divisor count d → strategy count d = count + d) →
    strategy count count = count + 1350 →
    ((count + d) > 2024) →
    wins strategy count :=
begin
  sorry
end

end anna_always_wins_l167_167572


namespace fuel_fill_up_cost_l167_167650

-- Define capacities of the tanks in gallons
def truck_tank_capacity : ℝ := 25
def car_tank_capacity : ℝ := 15

-- Define current amounts of fuel in the tanks
def current_truck_diesel : ℝ := truck_tank_capacity / 2
def current_car_gasoline : ℝ := car_tank_capacity / 3

-- Define prices per gallon
def price_per_gallon_diesel : ℝ := 3.50
def price_per_gallon_gasoline : ℝ := 3.20

-- Define how much fuel is needed to fill the tanks
def diesel_needed : ℝ := truck_tank_capacity - current_truck_diesel
def gasoline_needed : ℝ := car_tank_capacity - current_car_gasoline

-- Define costs to fill the tanks
def cost_to_fill_diesel : ℝ := diesel_needed * price_per_gallon_diesel
def cost_to_fill_gasoline : ℝ := gasoline_needed * price_per_gallon_gasoline

-- Define the total cost
def total_cost : ℝ := cost_to_fill_diesel + cost_to_fill_gasoline

theorem fuel_fill_up_cost :
  total_cost = 75.75 :=
by
  sorry

end fuel_fill_up_cost_l167_167650


namespace max_gcd_13n_plus_4_8n_plus_3_l167_167316

theorem max_gcd_13n_plus_4_8n_plus_3 : ∃ n : ℕ, n > 0 ∧ Int.gcd (13 * n + 4) (8 * n + 3) = 11 := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l167_167316


namespace f_zero_f_two_f_three_f_four_f_nat_squared_l167_167555

-- Definition of the function f satisfying the given condition
def f (x : ℝ) : ℝ := sorry

-- Condition: for any real numbers x and y, f(x + y) = f(x) + f(y) + 2xy
axiom f_add (x y : ℝ) : f(x + y) = f(x) + f(y) + 2 * x * y

-- Condition: f(1) = 1
axiom f_one : f(1) = 1

-- Prove: f(0) = 0
theorem f_zero : f(0) = 0 := 
by
  sorry

-- Prove: f(2) = 4
theorem f_two : f(2) = 4 := 
by
  sorry

-- Prove: f(3) = 9
theorem f_three : f(3) = 9 := 
by
  sorry

-- Prove: f(4) = 16
theorem f_four : f(4) = 16 := 
by
  sorry

-- Prove by induction: ∀ n ∈ ℕ, f(n) = n^2
theorem f_nat_squared (n : ℕ) : f(n) = n^2 := 
by
  induction n with
  | zero => 
    sorry
  | succ k ih => 
    sorry

end f_zero_f_two_f_three_f_four_f_nat_squared_l167_167555


namespace number_of_boys_in_class_l167_167966

theorem number_of_boys_in_class
  (g_ratio : ℕ) (b_ratio : ℕ) (total_students : ℕ)
  (h_ratio : g_ratio / b_ratio = 4 / 3)
  (h_total_students : g_ratio + b_ratio = 7 * (total_students / 56)) :
  total_students = 56 → 3 * (total_students / (4 + 3)) = 24 :=
by
  intros total_students_56
  sorry

end number_of_boys_in_class_l167_167966


namespace smallest_k_for_abk_l167_167689

theorem smallest_k_for_abk : ∃ (k : ℝ), (∀ (a b : ℝ), a + b = k ∧ ab = k → k = 4) :=
sorry

end smallest_k_for_abk_l167_167689


namespace inflation_two_years_real_rate_of_return_l167_167664

-- Proof Problem for Question 1
theorem inflation_two_years :
  ((1 + 0.015)^2 - 1) * 100 = 3.0225 :=
by
  sorry

-- Proof Problem for Question 2
theorem real_rate_of_return :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by
  sorry

end inflation_two_years_real_rate_of_return_l167_167664


namespace andy_ends_with_problem_126_l167_167308

theorem andy_ends_with_problem_126
  (starting_number : ℕ)
  (problems_solved : ℕ)
  (h1 : starting_number = 70)
  (h2 : problems_solved = 56) :
  starting_number + problems_solved = 126 :=
by { rw [h1, h2], norm_num, }

end andy_ends_with_problem_126_l167_167308


namespace baker_sales_difference_l167_167268

/-!
  Prove that the difference in dollars between the baker's daily average sales and total sales for today is 48 dollars.
-/

theorem baker_sales_difference :
  let price_pastry := 2
  let price_bread := 4
  let avg_pastries := 20
  let avg_bread := 10
  let today_pastries := 14
  let today_bread := 25
  let daily_avg_sales := avg_pastries * price_pastry + avg_bread * price_bread
  let today_sales := today_pastries * price_pastry + today_bread * price_bread
  daily_avg_sales - today_sales = 48 :=
sorry

end baker_sales_difference_l167_167268


namespace construct_circles_parallel_construct_circles_perpendicular_l167_167252

-- Definitions and given conditions
variable (P : Type) [MetricSpace P] -- Ambient Space
variable (e₁ e₂ e₃ : Set P) -- Given lines

-- Define the tangency condition
structure Tangent (c : Set P) (l : Set P) : Prop := (tangent_point : ∃ p : P, Metric.inf_dist p l = Metric.inf_dist p c)

-- Problem settings
noncomputable def circle_collection_exists (e₁ e₂ e₃ : Set P) : Prop :=
  ∃ (k₁ k₂ k₃ : Set P),
    Tangent k₁ e₂ ∧ Tangent k₁ e₃ ∧ Tangent k₁ k₂ ∧ Tangent k₁ k₃ ∧
    Tangent k₂ e₃ ∧ Tangent k₂ e₁ ∧ Tangent k₂ k₃ ∧
    Tangent k₃ e₁ ∧ Tangent k₃ e₂

-- Part (a): When e₁, e₂, e₃ are parallel
theorem construct_circles_parallel (e₁ e₂ e₃ : Set P) (h_parallel : ∀ p₁ p₂ ∈ e₁, ∀ q₁ q₂ ∈ e₂, ∀ r₁ r₂ ∈ e₃, 
  Line.parallel e₁ e₂ ∧ Line.parallel e₂ e₃ ∧ Line.parallel e₁ e₃):
  circle_collection_exists e₁ e₂ e₃ := sorry

-- Part (b): When e₃ is perpendicular to both e₁ and e₂
theorem construct_circles_perpendicular (e₁ e₂ e₃ : Set P) (h_perpendicular : ∀ p₁ p₂ ∈ e₁, ∀ q₁ q₂ ∈ e₂, ∀ r₁ r₂ ∈ e₃, 
  (Metric.angle p₁ r₁ p₂) = 90 ∧ (Metric.angle q₁ r₁ q₂) = 90):
  circle_collection_exists e₁ e₂ e₃ := sorry

end construct_circles_parallel_construct_circles_perpendicular_l167_167252


namespace households_subscribing_to_F_l167_167495

theorem households_subscribing_to_F
  (x y : ℕ)
  (hx : x ≥ 1)
  (h_subscriptions : 1 + 4 + 2 + 2 + 2 + y = 2 + 2 + 4 + 3 + 5 + x)
  : y = 6 :=
sorry

end households_subscribing_to_F_l167_167495


namespace length_YJ_l167_167877

-- Define the lengths of the sides of triangle XYZ
def XY : ℝ := 17
def XZ : ℝ := 19
def YZ : ℝ := 16

-- Define the inradius r and the lengths of segments a, b, c from the solution
def s : ℝ := (XY + XZ + YZ) / 2
def area_XYZ : ℝ := 90 -- Using the area calculated from Heron's formula
def r : ℝ := 45 / 13
def a : ℝ := 10
def b : ℝ := 7
def c : ℝ := 9

-- Prove that YJ = (91 / 13)
theorem length_YJ : ∀ (XY XZ YZ : ℝ) (a b c r : ℝ), 
  XY = 17 ∧ 
  XZ = 19 ∧ 
  YZ = 16 ∧ 
  a = 10 ∧ 
  b = 7 ∧ 
  c = 9 ∧ 
  r = 45 / 13 -> 
  YJ = (91 / 13) :=
by {
  intros,
  sorry
}

end length_YJ_l167_167877


namespace Brazil_wins_10_l167_167097

/-- In the year 3000, the World Hockey Championship will follow new rules: 12 points will be awarded for a win, 
5 points will be deducted for a loss, and no points will be awarded for a draw. If the Brazilian team plays 
38 matches, scores 60 points, and loses at least once, then the number of wins they can achieve is 10. 
List all possible scenarios and justify why there cannot be any others. -/
theorem Brazil_wins_10 (x y z : ℕ) 
    (h1: x + y + z = 38) 
    (h2: 12 * x - 5 * y = 60) 
    (h3: y ≥ 1)
    (h4: z ≥ 0): 
  x = 10 :=
by
  sorry

end Brazil_wins_10_l167_167097


namespace last_score_84_l167_167083

theorem last_score_84 (scores : List ℕ) (ordered_scores : scores = [65, 69, 78, 84, 92]) 
  (avg_always_integer : ∀ i, 1 ≤ i ∧ i ≤ 5 → (i list_sum (take i scores)) % i = 0) : 
  ∃ last_score, last_score = 84 := 
by 
  cases scores 
  case nil => sorry 
  case cons => sorry

end last_score_84_l167_167083


namespace ott_fractional_part_l167_167158

theorem ott_fractional_part (x : ℝ) :
  let moe_initial := 6 * x
  let loki_initial := 5 * x
  let nick_initial := 4 * x
  let ott_initial := 1
  
  let moe_given := (x : ℝ)
  let loki_given := (x : ℝ)
  let nick_given := (x : ℝ)
  
  let ott_returned_each := (1 / 10) * x
  
  let moe_effective := moe_given - ott_returned_each
  let loki_effective := loki_given - ott_returned_each
  let nick_effective := nick_given - ott_returned_each
  
  let ott_received := moe_effective + loki_effective + nick_effective
  let ott_final_money := ott_initial + ott_received
  
  let total_money_original := moe_initial + loki_initial + nick_initial + ott_initial
  let fraction_ott_final := ott_final_money / total_money_original
  
  ott_final_money / total_money_original = (10 + 27 * x) / (150 * x + 10) :=
by
  sorry

end ott_fractional_part_l167_167158


namespace pet_shop_dogs_l167_167249

theorem pet_shop_dogs (D C B : ℕ) (x : ℕ) (h1 : D = 3 * x) (h2 : C = 5 * x) (h3 : B = 9 * x) (h4 : D + B = 204) : D = 51 := by
  -- omitted proof
  sorry

end pet_shop_dogs_l167_167249


namespace hexagon_chord_length_l167_167277

noncomputable def chord_length (r α β : ℝ) : ℝ :=
  2 * r * sin (3 * α)

theorem hexagon_chord_length
  (r : ℝ)
  (α β : ℝ)
  (h_hexagon : 6 * α + 6 * β = 360)
  (h_sides_3 : 2 * r * sin α = 3)
  (h_sides_5 : 2 * r * sin β = 5)
  (gcd_relatively_prime : ∀ (m n : ℕ), nat.gcd m n = 1)
  (m n : ℕ)
  (h_mn : m + n = 409) :
  chord_length r α β = m / n :=
sorry

end hexagon_chord_length_l167_167277


namespace color_graph_two_colors_l167_167796

theorem color_graph_two_colors (G : Type) [graph G] (N : ℕ) 
  (hN : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ N → ∀ (V' ⊆ finset.univ : finset G.vertex), ∀ S ⊂ V', V'.card = k → G.edges S ≤ 2 * k - 2) :
  ∃ (f : G.edge → bool), ∀ C ∈ G.cycles, ∃ e ∈ C, ∃ b ∈ {tt, ff}, f e = b ∨ f e ≠ b := 
sorry

end color_graph_two_colors_l167_167796


namespace triangle_properties_l167_167951

noncomputable theory
open_locale big_operators

structure Triangle :=
  (A B C : ℝ)
  (angle_A : ℝ)
  (circumradius : ℝ)

def radius_of_inscribed_circle (T : Triangle) (BD DC : ℝ) : ℝ := 
  2 * (sqrt (T.A * T.B * T.C * (T.A + T.B + T.C))) / (T.A + T.B + T.C)

def area_of_triangle (T : Triangle) (BD DC : ℝ) : ℝ :=
  (sqrt (T.A * T.B * T.C * (T.A + T.B + T.C))) / 2

theorem triangle_properties (T : Triangle) (BD DC : ℝ)
  (h₁ : BD = 5)
  (h₂ : T.angle_A = 60 * π / 180)
  (h₃ : T.circumradius = 7 / sqrt 3)
  (h₄ : T.A > 0) (h₅ : T.B > 0) (h₆ : T.C > 0):
  radius_of_inscribed_circle T BD DC = sqrt 3 ∧
  area_of_triangle T BD DC = 10 * sqrt 3 :=
sorry

end triangle_properties_l167_167951


namespace intersection_empty_l167_167830

def M := {l | l ∈ set.univ ∧ is_line l}
def N := {c | c ∈ set.univ ∧ is_circle c}

theorem intersection_empty : M ∩ N = ∅ :=
by sorry

noncomputable def is_line : Type → Prop := sorry -- definition for a line goes here
noncomputable def is_circle : Type → Prop := sorry -- definition for a circle goes here

end intersection_empty_l167_167830


namespace optimal_allocation_l167_167676

variables {weeks : ℕ → ℕ}

-- Definitions for scores based on the number of weeks allocated
def ideological_score (x : ℕ) : ℕ :=
  match x with
  | 0 => 20
  | 1 => 40
  | 2 => 55
  | 3 => 65
  | 4 => 72
  | 5 => 78
  | 6 => 80
  | 7 => 82
  | 8 => 83
  | 9 => 84
  | 10 => 85
  | _ => 85

def foreign_language_score (y : ℕ) : ℕ :=
  match y with
  | 0 => 30
  | 1 => 45
  | 2 => 53
  | 3 => 58
  | 4 => 62
  | 5 => 65
  | 6 => 68
  | 7 => 70
  | 8 => 72
  | 9 => 74
  | 10 => 75
  | _ => 75

def professional_course_score (z : ℕ) : ℕ :=
  match z with
  | 0 => 50
  | 1 => 70
  | 2 => 85
  | 3 => 90
  | 4 => 93
  | 5 => 95
  | 6 => 96
  | _ => 96

-- Total weeks available for review
def total_weeks : ℕ := 11

-- Proof problem: Prove that (weeks allocated to Ideological and Political Education, weeks allocated to Foreign Language, weeks allocated to Professional Courses) = (5, 4, 2) maximizes Zhenhua's total score.
theorem optimal_allocation : (5, 4, 2) = (5, 4, 2) → 
(((ideological_score 5) + (foreign_language_score 4) + (professional_course_score 2)) ≥ (ideological_score x + foreign_language_score y + professional_course_score z) 
    ∀ x y z, x + y + z = total_weeks) :=
sorry

end optimal_allocation_l167_167676


namespace first_number_is_seven_l167_167628

variable (x y : ℝ)

theorem first_number_is_seven (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) : x = 7 :=
sorry

end first_number_is_seven_l167_167628


namespace part1_part2_l167_167798

noncomputable section

open Complex

theorem part1 (a : ℝ) (h : (a + (1 : ℂ) * I)^2 = -2 * I) : a = -1 :=
sorry

theorem part2 (a : ℝ) (h1 : (a + (1 : ℂ) * I) / (1 - I) = (a - 1) / 2 + ((a + 1) / 2) * I) 
    (h2 : (a - 1) / 2 = 0) : 
    (let z := (a + 1) / 2 * I
    (sum (range 2024) (λ k, z ^ k)) = -1 :=
sorry

end part1_part2_l167_167798


namespace prime_x_difference_l167_167184

theorem prime_x_difference 
  (x : ℕ) 
  (h₁ : (45 + x) / 2 = 50) 
  (h₂ : Prime x) : |45 - x| = 8 := sorry

end prime_x_difference_l167_167184


namespace average_pages_per_day_is_correct_l167_167010

-- Definitions based on the given conditions
def first_book_pages := 249
def first_book_days := 3

def second_book_pages := 379
def second_book_days := 5

def third_book_pages := 480
def third_book_days := 6

-- Definition of total pages read
def total_pages := first_book_pages + second_book_pages + third_book_pages

-- Definition of total days spent reading
def total_days := first_book_days + second_book_days + third_book_days

-- Definition of expected average pages per day
def expected_average_pages_per_day := 79.14

-- The theorem to prove
theorem average_pages_per_day_is_correct : (total_pages.toFloat / total_days.toFloat) = expected_average_pages_per_day :=
by
  sorry

end average_pages_per_day_is_correct_l167_167010


namespace linda_paint_cans_l167_167155

theorem linda_paint_cans (wall_area : ℝ) (coverage_per_gallon : ℝ) (coats : ℝ) 
  (h1 : wall_area = 600) 
  (h2 : coverage_per_gallon = 400) 
  (h3 : coats = 2) : 
  (ceil (wall_area * coats / coverage_per_gallon) = 3) := 
by 
  sorry

end linda_paint_cans_l167_167155


namespace probability_two_digit_gt_30_l167_167042

open Finset
open Rat

-- Definition of digits set
def digits : Finset ℕ := {1, 2, 3}

-- Definition of valid two-digit numbers (in decimal)
def two_digit_numbers : Finset (ℕ × ℕ) := (digits.product digits).filter (λ p, p.1 ≠ p.2)

-- Function to check if a two-digit number made from a pair is greater than 30
def is_greater_than_30 (p : ℕ × ℕ) : Bool :=
  let num := p.1 * 10 + p.2
  num > 30

-- Set of valid two-digit numbers greater than 30
def two_digit_numbers_gt_30 : Finset (ℕ × ℕ) :=
  (two_digit_numbers.filter is_greater_than_30)

-- The proof goal stating the probability
theorem probability_two_digit_gt_30 : 
  (two_digit_numbers_gt_30.card : ℚ) / (two_digit_numbers.card : ℚ) = 1 / 3 :=
by 
  -- Begin proof steps here
  sorry

end probability_two_digit_gt_30_l167_167042


namespace length_of_AB_l167_167704

-- Defining the parabola and the condition on x1 and x2
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def condition (x1 x2 : ℝ) : Prop := x1 + x2 = 9

-- The main statement to prove |AB| = 11
theorem length_of_AB (x1 x2 y1 y2 : ℝ) (h1 : parabola x1 y1) (h2 : parabola x2 y2) (hx : condition x1 x2) :
  abs (x1 - x2) + abs (y1 - y2) = 11 :=
sorry

end length_of_AB_l167_167704


namespace num_digits_multiple_of_4_l167_167005

theorem num_digits_multiple_of_4 : (Finset.card (Finset.filter (λ C : Fin 10, (10 * C + 4) % 4 = 0) Finset.univ) = 5) := 
by {
  sorry -- This is where the proof would go.
}

end num_digits_multiple_of_4_l167_167005


namespace find_reflection_line_l167_167217

theorem find_reflection_line :
  (P Q R P' Q' R' : Point) →
  P = (2, 2) →
  Q = (6, 6) →
  R = (-3, 5) →
  P' = (2, -4) →
  Q' = (6, -8) →
  R' = (-3, -7) →
  ∃ L : Line, L.equation = "y = -1" := by
sorry

end find_reflection_line_l167_167217


namespace sum_of_three_smallest_a_l167_167763

theorem sum_of_three_smallest_a : 
  let a_values := {a : ℝ | ∀ x : ℝ, (x < -4 → (x^2 + (a + 1) * x + a ≥ 0) → (x^2 + 5 * x + 4 > 0))
                              ∧ (x ∈ [-a, -1) → (x^2 + (a + 1) * x + a ≥ 0) → (x^2 + 5 * x + 4 > 0))
                              ∧ (x > -1  → (x^2 + (a + 1) * x + a ≥ 0) → (x^2 + 5 * x + 4 > 0)) }
  in (∃ a1 a2 a3 ∈ a_values, a1 + a2 + a3 = 9) := sorry

end sum_of_three_smallest_a_l167_167763


namespace joes_average_score_l167_167522

theorem joes_average_score (
  A B C : ℕ,
  h1 : A ≥ 30,
  h2 : B ≥ 30,
  h3 : C ≥ 30,
  h_sum : A + B + C = 150
) : (A + B + C + 30) / 4 = 45 :=
by sorry

end joes_average_score_l167_167522


namespace distance_from_origin_is_correct_l167_167285

-- Define the point (x, y) with given conditions
variables (x y : ℝ)

-- Given conditions
axiom h1 : y = 20
axiom h2 : dist (x, y) (2, 15) = 15
axiom h3 : x > 2

-- The theorem to prove
theorem distance_from_origin_is_correct :
  dist (x, y) (0, 0) = Real.sqrt (604 + 40 * Real.sqrt 2) :=
by
  -- Set h1, h2, and h3 as our constraints
  sorry

end distance_from_origin_is_correct_l167_167285


namespace exists_n_ge_1_le_2020_l167_167577

theorem exists_n_ge_1_le_2020
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j : ℕ, 1 ≤ i → i ≤ 2020 → 1 ≤ j → j ≤ 2020 → i ≠ j → a i ≠ a j)
  (h_periodic1 : a 2021 = a 1)
  (h_periodic2 : a 2022 = a 2) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2020 ∧ a n ^ 2 + a (n + 1) ^ 2 ≥ a (n + 2) ^ 2 + n ^ 2 + 3 := 
sorry

end exists_n_ge_1_le_2020_l167_167577


namespace sum_of_sines_leq_3_sqrt3_over_2_l167_167397

theorem sum_of_sines_leq_3_sqrt3_over_2 (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Real.sin α + Real.sin β + Real.sin γ ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end sum_of_sines_leq_3_sqrt3_over_2_l167_167397


namespace math_problem_solution_l167_167238

-- Definitions for statement A
def line1 (a x y : ℝ) : Prop := a * x - y + 1 = 0
def line2 (a x y : ℝ) : Prop := x - a * y - 2 = 0

-- Definitions for statement B
def pointA : ℝ × ℝ := (2, 1)
def pointB : ℝ × ℝ := (-1, 2 * Real.sqrt 3)
def pointP : ℝ × ℝ := (1, 0)

-- Definitions for statement C
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2 * x = 0
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 4 * x - 8 * y + m = 0

-- Definitions for statement D
def circle3 (x y : ℝ) : Prop := x^2 + y^2 = 2
def line3 (x y : ℝ) : Prop := x - y + 1 = 0

-- The final proof statement in Lean
theorem math_problem_solution :
(
  (∀ a x y, line1 a x y → line2 a x y → a = 1) = False ∧
  (∀ t, ∃ k, k ∈ Set.Icc (Float.pi / 4) (2 * Float.pi / 3) ∧ 
            (pointA.fst - pointB.fst) * t + pointA.snd = (k * t) + pointB.snd) = True ∧
  (∀ x y m, circle1 x y ∧ circle2 x y m → 4 < m ∧ m < 20) = True ∧
  (∃ x y, circle3 x y ∧ line3 x y → False) = False
) :=
by
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

end math_problem_solution_l167_167238


namespace fifth_term_equals_31_l167_167738

-- Define the sequence of sums of consecutive powers of 2
def sequence_sum (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (λ i, 2^i)

-- State the theorem: The fifth term of the sequence equals 31
theorem fifth_term_equals_31 : sequence_sum 4 = 31 := by
  sorry

end fifth_term_equals_31_l167_167738


namespace tom_miles_per_day_l167_167996

theorem tom_miles_per_day (miles_per_day_first_183 : ℕ) (days_first : ℕ) (total_year_miles : ℕ) (total_year_days : ℕ) 
                          (miles_per_day_first_183 = 30) (days_first = 183) (total_year_miles = 11860) (total_year_days = 365) :
  let remaining_miles := total_year_miles - miles_per_day_first_183 * days_first in
  let remaining_days := total_year_days - days_first in
  remaining_miles / remaining_days = 35 := 
by
  sorry

end tom_miles_per_day_l167_167996


namespace number_of_cows_l167_167059

-- Define the total number of legs and number of legs per cow
def total_legs : ℕ := 460
def legs_per_cow : ℕ := 4

-- Mathematical proof problem as a Lean 4 statement
theorem number_of_cows : total_legs / legs_per_cow = 115 := by
  -- This is the proof statement place. We use 'sorry' as a placeholder for the actual proof.
  sorry

end number_of_cows_l167_167059


namespace angle_between_vectors_l167_167549

variables {V : Type*} [inner_product_space ℝ V]

/-- The angle between vectors a and b in a real inner product space, given certain norms -/
theorem angle_between_vectors {a b : V} 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h1 : ∥a∥ = 2 * ∥a + b∥) 
  (h2 : ∥b∥ = 2 * ∥a + b∥) : 
  real.angle a b = real.arccos (-7 / 8) :=
sorry

end angle_between_vectors_l167_167549


namespace find_b2023_l167_167146

noncomputable def seq (n : ℕ) : ℝ
def b1 : ℝ := 2 + Real.sqrt 8  
def b1800 : ℝ := 14 + Real.sqrt 8
axiom seq_recurrence (n : ℕ) (h : 2 ≤ n) : seq n = seq (n - 1) * seq (n + 1)

theorem find_b2023 : seq 2023 = 3 * Real.sqrt 8 - 5 :=
by
  sorry

end find_b2023_l167_167146


namespace distinct_three_digit_numbers_with_even_digits_l167_167458

theorem distinct_three_digit_numbers_with_even_digits : 
  let even_digits := {0, 2, 4, 6, 8} in
  (∃ (hundreds options : Finset ℕ) (x : ℕ), 
    hundreds = {2, 4, 6, 8} ∧ 
    options = even_digits ∧ 
    x = Finset.card hundreds * Finset.card options * Finset.card options ∧ 
    x = 100) :=
by
  let even_digits := {0, 2, 4, 6, 8}
  exact ⟨{2, 4, 6, 8}, even_digits, 100, rfl, rfl, sorry, rfl⟩

end distinct_three_digit_numbers_with_even_digits_l167_167458


namespace bouncy_ball_pack_count_l167_167128

theorem bouncy_ball_pack_count
  (x : ℤ)  -- Let x be the number of bouncy balls in each pack
  (r : ℤ := 7 * x)  -- Total number of red bouncy balls
  (y : ℤ := 6 * x)  -- Total number of yellow bouncy balls
  (h : r = y + 18)  -- Condition: 7x = 6x + 18
  : x = 18 := sorry

end bouncy_ball_pack_count_l167_167128


namespace part_I_part_II_part_III_l167_167439

-- Definitions
def a_seq (n : ℕ) : ℤ := 1 + 2 * n
def S_n (n : ℕ) : ℤ := (1 + a_seq n) * n / 2

-- Statement for part (I)
theorem part_I (n : ℕ) : S_n n = n^2 := sorry

-- Definitions for part (II)
def b_seq (n : ℕ) : ℝ := 2 ^ (n - 1)
def T_n (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), b_seq k

-- Statement for part (II)
theorem part_II (q : ℝ) (h : q > 1) (H : ∀ n, T_n (n + 1) ≤ 4 * b_seq n) : q = 2 := sorry

-- Definitions for part (III)
def condition_i (p : ℕ) : Prop := ∀ n, (n ≠ p) ∨ (a_seq n < a_seq (n - 1))
def condition_ii : Prop := ∀ m, T_n m > 0

-- Statement for part (III)
theorem part_III (p : ℕ) (H1 : condition_i p) (H2 : condition_ii) :
  (∃ m : ℕ, S_n (m + 1) = 4 * b_seq m) ↔ (p ≥ 3 ∧ m = 1) := sorry


end part_I_part_II_part_III_l167_167439


namespace number_of_triangles_l167_167222

/- 
  Given that we have five lines in general position (none of them are parallel, 
  and no three lines intersect at one point), which divides the plane into 
  exactly 16 regions, prove the minimum and maximum number of triangular regions are 3 and 5 respectively.
-/

theorem number_of_triangles (h : ∀ (n : ℕ), n = 5 → 
    (none_parallel_and_no_three_intersect_at_one_point n) → 
    divide_plane_with_lines n = 16) : 
  ∃ (t_min t_max : ℕ), t_min = 3 ∧ t_max = 5 := 
begin
  sorry
end

end number_of_triangles_l167_167222


namespace thomas_payment_weeks_l167_167989

theorem thomas_payment_weeks 
    (weekly_rate : ℕ) 
    (total_amount_paid : ℕ) 
    (h1 : weekly_rate = 4550) 
    (h2 : total_amount_paid = 19500) :
    (19500 / 4550 : ℕ) = 4 :=
by {
  sorry
}

end thomas_payment_weeks_l167_167989


namespace log_base_4_of_2_l167_167366

theorem log_base_4_of_2 : log 4 2 = 1 / 2 := by
  sorry

end log_base_4_of_2_l167_167366


namespace part_i_part_ii_l167_167880

-- Part (i)
theorem part_i (a b : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = a * x^2 + b * x)
  (h2 : (∂ f / ∂ x) 3 = 24)
  (h3 : (∂ f / ∂ x) 1 = 0) :
  { x : ℝ | x > 1 } = { x : ℝ | 6 * x - 18 ≤ 0 } := by sorry

-- Part (ii)
theorem part_ii (b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = x^2 + b * x) 
  (h2 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (∂ f / ∂ x) x ≤ 0) :
  b ≤ -2 := by sorry

end part_i_part_ii_l167_167880


namespace distinct_three_digit_even_integers_count_l167_167463

theorem distinct_three_digit_even_integers_count : 
  let even_digits := {0, 2, 4, 6, 8}
  ∃ h : Finset ℕ, h = {2, 4, 6, 8} ∧ 
     (∏ x in h, 5 * 5) = 100 :=
by
  let even_digits : Finset ℕ := {0, 2, 4, 6, 8}
  let h : Finset ℕ := {2, 4, 6, 8}
  have : ∏ x in h, 5 * 5 = 100 := sorry
  exact ⟨h, rfl, this⟩

end distinct_three_digit_even_integers_count_l167_167463


namespace difference_in_sales_l167_167264

def daily_pastries : ℕ := 20
def daily_bread : ℕ := 10
def today_pastries : ℕ := 14
def today_bread : ℕ := 25
def price_pastry : ℕ := 2
def price_bread : ℕ := 4

theorem difference_in_sales : (daily_pastries * price_pastry + daily_bread * price_bread) - (today_pastries * price_pastry + today_bread * price_bread) = -48 :=
by
  -- Proof will go here
  sorry

end difference_in_sales_l167_167264


namespace f_neg_one_eq_three_l167_167786

noncomputable def f : ℝ → ℝ 
| x => if x < 6 then f (x + 3) else Real.log x / Real.log 2

theorem f_neg_one_eq_three : f (-1) = 3 := 
by
  sorry

end f_neg_one_eq_three_l167_167786


namespace find_line_equation_l167_167954

-- Let's define Point and Line structures for better clarity
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  equation : ℝ → ℝ → Prop

def passes_through (L : Line) (P : Point) : Prop :=
  L.equation P.x P.y

def has_equal_intercepts (L : Line) : Prop :=
  ∃ c : ℝ, c ≠ 0 ∧ L.equation c 0 ∧ L.equation 0 c

theorem find_line_equation (A : Point) (L : Line) (hA : A = ⟨1, 2⟩)
  (h1 : passes_through L A) (h2 : has_equal_intercepts L) :
  (L.equation = (λ x y, 2 * x - y = 0))
  ∨ (L.equation = (λ x y, x + y - 3 = 0)) := by
  sorry

end find_line_equation_l167_167954


namespace problem1_problem2_l167_167730

-- Define the first problem
theorem problem1 : ( (9 / 4) ^ (1 / 2) - (-8.6) ^ 0 - (8 / 27) ^ (-1 / 3)) = -1 := by
  sorry

-- Define the second problem
theorem problem2 : log 10 25 + log 10 4 + 7 ^ (log 7 2) + 2 * log 3 (sqrt 3) = 5 := by
  sorry

end problem1_problem2_l167_167730


namespace intersection_of_P_and_Q_l167_167831

def P : set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def Q : set (ℝ × ℝ) := {q | q.1 - q.2 = 4}

theorem intersection_of_P_and_Q :
  P ∩ Q = {(3 : ℝ, -1 : ℝ)} :=
by
  sorry

end intersection_of_P_and_Q_l167_167831


namespace inequality_proof_l167_167762

theorem inequality_proof : ∀ (a b c d : ℝ), 
  (-1 ≤ a) → (-1 ≤ b) → (-1 ≤ c) → (-1 ≤ d) → 
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3 / 4) * (a + b + c + d) :=
begin
  sorry
end

end inequality_proof_l167_167762


namespace distance_between_A_and_B_eq_1656_l167_167890

variables (A B C E D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space E] [metric_space D]
          (v_jun v_ping : ℝ) (AC BC DC : ℝ)

-- Constants
def CE := 100
def CD := 360

-- Ratios given in the problem
def ratio_speeds := 9 / 14
def ratio_parts := 23

-- Conditions
theorem distance_between_A_and_B_eq_1656 
  (h1 : v_jun / v_ping = ratio_speeds)
  (h2 : AC / BC = 14 / 9)
  (h3 : DC = CD / 5 * 14)
  (h4 : A |-> B = ratio_parts * CD / 5) : 
  A |-> B = 1656 := 
by sorry

end distance_between_A_and_B_eq_1656_l167_167890


namespace solve_for_x_y_l167_167876

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def triangle_ABC (A B C E F : V) (x y : ℝ) : Prop :=
  (E - A) = (1 / 2) • (B - A) ∧
  (C - F) = (2 : ℝ) • (A - F) ∧
  (E - F) = x • (B - A) + y • (C - A)

theorem solve_for_x_y (A B C E F : V) (x y : ℝ) :
  triangle_ABC A B C E F x y →
  x + y = - (1 / 6 : ℝ) :=
by
  sorry

end solve_for_x_y_l167_167876


namespace _l167_167680

noncomputable def speed_kmh := 60
noncomputable def time_s := 18

noncomputable theorem train_length :
  let speed_mps := speed_kmh * (1000 / 3600)
  let length := speed_mps * time_s
  length = 300.06 := by 
  sorry

end _l167_167680


namespace system_of_equations_solution_l167_167590

variable {x y : ℝ}

theorem system_of_equations_solution
  (h1 : x^2 + x * y * Real.sqrt (x * y) + y^2 = 25)
  (h2 : x^2 - x * y * Real.sqrt (x * y) + y^2 = 9) :
  (x, y) = (1, 4) ∨ (x, y) = (4, 1) ∨ (x, y) = (-1, -4) ∨ (x, y) = (-4, -1) :=
by
  sorry

end system_of_equations_solution_l167_167590


namespace max_value_m_l167_167807

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 2
noncomputable def g (x a b : ℝ) : ℝ := Real.exp x + b * x^2 + a

theorem max_value_m (a b m : ℝ) 
  (tangent_cond : ∀ x, f x = a * x + b - Real.log 2) 
  (ineq_cond : ∀ x ∈ set.Icc 1 2, m ≤ g x a b ∧ g x a b ≤ m^2 - 2) :
  m ≤ Real.exp 1 + 1 := 
sorry

end max_value_m_l167_167807


namespace sequence_bound_l167_167131

theorem sequence_bound {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0) (M : ℝ) (h_bound : ∀ n, (∑ i in Finset.range n, (a i)^2) < M * (a n.succ)^2) :
  ∃ M' > 0, ∀ n, (∑ i in Finset.range n, a i) < M' * a n.succ := 
sorry

end sequence_bound_l167_167131


namespace sqrt_expr_eq_neg_x_l167_167338

theorem sqrt_expr_eq_neg_x (x : ℝ) (h : x < -2) : 
  sqrt (x / (1 + (x + 1)/(x + 2))) = -x :=
sorry

end sqrt_expr_eq_neg_x_l167_167338


namespace no_infinite_primes_sequence_l167_167928

theorem no_infinite_primes_sequence (p : ℕ → ℕ)
  (prime : ∀ n : ℕ, Nat.Prime (p n))
  (increasing : ∀ m n : ℕ, m < n → p m < p n)
  (relation : ∀ k : ℕ, p (k + 1) = 2 * p k ± 1) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → Nat.Prime (p n) = false := 
sorry

end no_infinite_primes_sequence_l167_167928


namespace light_path_length_l167_167527

/--
Let $ABCD$ and $BCFG$ be two faces of a cube with $AB=10$. 
A beam of light emanates from vertex $A$ and reflects off face $BCFG$ at point $P$, 
which is 6 units from $\overline{BG}$ and 3 units from $\overline{BC}$. 
The beam continues to be reflected off the faces of the cube. 
Prove that the length of the light path from the time it leaves point $A$ until it next reaches a vertex of the cube is $10\sqrt{145}$, 
and thus $m+n=155$.
-/
theorem light_path_length (A B C D F G P: ℝ)
  (H1 : dist A B = 10)
  (H2 : dist P (line_through B G) = 6)
  (H3 : dist P (line_through B C) = 3)
  : ∃ m n : ℤ, m * real.sqrt n = 10 * real.sqrt 145 ∧ (n % (real.sqrt n).nat_abs ^ 2 ≠ 0) ∧ m + n = 155 :=
begin
  sorry
end

end light_path_length_l167_167527


namespace g_fifty_eq_zero_l167_167151

noncomputable def g : ℝ → ℝ := sorry

theorem g_fifty_eq_zero (g : ℝ → ℝ)
(hg_pos : ∀ x, 0 < x → 0 < g x → 0)
(h_fe : ∀ x y, 0 < x → 0 < y → x * g y - y * g x = g (x / y) + g (x * y)) :
g 50 = 0 :=
sorry

end g_fifty_eq_zero_l167_167151


namespace find_a_b_l167_167049

noncomputable def parabola_props (a b : ℝ) : Prop :=
a ≠ 0 ∧ 
∀ x : ℝ, a * x^2 + b * x - 4 = (1 / 2) * x^2 + x - 4

theorem find_a_b {a b : ℝ} (h1 : parabola_props a b) : 
a = 1 / 2 ∧ b = -1 :=
sorry

end find_a_b_l167_167049


namespace exists_odd_cycle_l167_167936

-- Conditions and definitions
def cities : Type := fin 1988
def airlines : Type := fin 10

-- Full connectivity and bidirectional operation by airlines
axiom fully_connected (a : airlines) (c1 c2 : cities) : c1 ≠ c2 → (c1 → c2) ∧ (c2 → c1)

-- Statement to prove
theorem exists_odd_cycle :
  ∃ a : airlines, ∃ c : list cities, (cycle c) ∧ (list.length c % 2 = 1) :=
sorry

end exists_odd_cycle_l167_167936


namespace probability_no_correct_letter_for_7_envelopes_l167_167983

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

noncomputable def probability_no_correct_letter (n : ℕ) : ℚ :=
  derangement n / factorial n

theorem probability_no_correct_letter_for_7_envelopes :
  probability_no_correct_letter 7 = 427 / 1160 :=
by sorry

end probability_no_correct_letter_for_7_envelopes_l167_167983


namespace solve_for_one_star_one_l167_167389

theorem solve_for_one_star_one (a b c : ℚ) :
  (3 * a + 5 * b + c = 15) → 
  (4 * a + 7 * b + c = 28) → 
  (∀ x y : ℚ, x ⬝ y = a * x + b * y + c) → 
  (1 ⬝ 1 = -11) :=
by
  intros h1 h2 op_def
  sorry

end solve_for_one_star_one_l167_167389


namespace max_value_f_l167_167767

def f (x : ℝ) := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem max_value_f : 
  ∃ x ∈ Set.Icc (Real.pi/4) (Real.pi/2), f x = 3/2 :=
by
  sorry

end max_value_f_l167_167767


namespace smallest_positive_period_l167_167750

def det2x2 (a1 a2 a3 a4 : ℝ) : ℝ :=
  a1 * a4 - a2 * a3

def f (x : ℝ) : ℝ :=
  det2x2 (Real.sin x) (-1) 1 (Real.cos x)

theorem smallest_positive_period :
  ∀ x, f (x + π) = f x := by
  sorry

end smallest_positive_period_l167_167750


namespace ratio_of_volumes_l167_167290

theorem ratio_of_volumes (h r : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) : 
  let V_t := (1 / 3) * π * (r / 4) ^ 2 * (h / 4),
      V_m := (1 / 3) * π * (r / 2) ^ 2 * (h / 4) - (1 / 3) * π * (r / 4) ^ 2 * (h / 4),
      V_b := (1 / 3) * π * r ^ 2 * (h / 2) - (1 / 3) * π * (r / 2) ^ 2 * (h / 4) in
  V_m / V_b = 3 / 16 := by sorry

end ratio_of_volumes_l167_167290


namespace hyperbola_equation_same_asymptotes_l167_167765

theorem hyperbola_equation_same_asymptotes (λ : ℝ) (x y : ℝ)
  (h_eq1 : ∃ (k : ℝ), k * (x^2 / 9 - y^2 / 16) = 1)
  (h_point : ∃ (a b : ℝ), a = -3 ∧ b = 2 * real.sqrt 3 ∧ a^2 / 9 - b^2 / 16 = λ) :
  ∃ (k : ℝ), k * (4 * x^2 / 9 - y^2 / 4) = 1 :=
by sorry

end hyperbola_equation_same_asymptotes_l167_167765


namespace question_solution_l167_167728

variable (a b : ℝ)

theorem question_solution : 2 * a - 3 * (a - b) = -a + 3 * b := by
  sorry

end question_solution_l167_167728


namespace more_cookies_l167_167562

theorem more_cookies (x y : ℕ) 
  (h1 : x / 2 = 40) 
  (h2 : 3 * y / 5 = 25) : 
  y - x = -38 := 
sorry

end more_cookies_l167_167562


namespace max_value_of_expression_l167_167941

variables (x y : ℝ)

theorem max_value_of_expression (hx : 0 < x) (hy : 0 < y) (h : x^2 - 2*x*y + 3*y^2 = 12) : x^2 + 2*x*y + 3*y^2 ≤ 24 + 24*sqrt 3 :=
sorry

end max_value_of_expression_l167_167941


namespace max_p_plus_q_l167_167033

theorem max_p_plus_q (p q : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → 2 * p * x^2 + q * x - p + 1 ≥ 0) : p + q ≤ 2 :=
sorry

end max_p_plus_q_l167_167033


namespace necessary_but_not_sufficient_condition_l167_167411

variable {α : Type} [Plane α]
variable {m n : Line α}

/--
Given lines m, n, and plane α, 
if the angles formed by m and n with α are equal, 
it is a necessary but not sufficient condition for m ∥ n.
-/
theorem necessary_but_not_sufficient_condition (h : angle_with_plane m α = angle_with_plane n α) : ¬ (m ∥ n) :=
sorry

end necessary_but_not_sufficient_condition_l167_167411


namespace find_a_given_solution_l167_167808

theorem find_a_given_solution (a : ℝ) (x : ℝ) (h : x = 1) (eqn : a * (x + 1) = 2 * (2 * x - a)) : a = 1 := 
by
  sorry

end find_a_given_solution_l167_167808


namespace prob_B_given_A_l167_167993

/-
Define events A and B:
- A: Blue die results in 4 or 6
- B: Sum of results of red die and blue die is greater than 8
-/

def event_A (blue : ℕ) : Prop := blue = 4 ∨ blue = 6
def event_B (red blue : ℕ) : Prop := red + blue > 8

theorem prob_B_given_A :
  let outcomes := (1..6).bind (fun red => (1..6).map (fun blue => (red, blue))) in
  let PA := (outcomes.filter (fun ⟨r, b⟩ => event_A b)).length / outcomes.length.toFloat in
  let PAB := (outcomes.filter (fun ⟨r, b⟩ => event_A b ∧ event_B r b)).length / outcomes.length.toFloat in
  PAB / PA = 1 / 2 := sorry


end prob_B_given_A_l167_167993


namespace eval_expr_at_x_eq_neg6_l167_167584

-- Define the given condition
def x : ℤ := -4

-- Define the expression to be simplified and evaluated
def expr (x y : ℤ) : ℤ := ((x + y)^2 - y * (2 * x + y) - 8 * x) / (2 * x)

-- The theorem stating the result of the evaluated expression
theorem eval_expr_at_x_eq_neg6 (y : ℤ) : expr (-4) y = -6 := 
by
  sorry

end eval_expr_at_x_eq_neg6_l167_167584


namespace arithmetic_binom_difference_l167_167001

theorem arithmetic_binom_difference (k : ℕ) (h : k ≥ 2) :
  let diff := Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1)) in
  (2^(3*k) ∣ diff) ∧ ¬ (2^(3*k+1) ∣ diff) :=
by
  sorry

end arithmetic_binom_difference_l167_167001


namespace arlo_stationery_count_l167_167622

theorem arlo_stationery_count (books pens : ℕ) (ratio_books_pens : ℕ × ℕ) (total_books : ℕ)
  (h_ratio : ratio_books_pens = (7, 3)) (h_books : total_books = 280) :
  books + pens = 400 :=
by
  sorry

end arlo_stationery_count_l167_167622


namespace sum_equals_fraction_minus_constant_l167_167224

theorem sum_equals_fraction_minus_constant :
  (∑ k in finset.range 50.succ, (-1)^k * (k^3 + k^2 + k + 1) / (k.factorial)) =
  (2602 / (50.factorial)) - 1 :=
sorry

end sum_equals_fraction_minus_constant_l167_167224


namespace max_avg_speed_1_5_to_2_l167_167611

-- Define the distance function
def distance_function (t : ℝ) : ℝ := sorry -- Placeholder for the actual distance function

-- Define the average speed over a half-hour period
def avg_speed (d : ℝ → ℝ) (start_time : ℝ) : ℝ :=
  (d (start_time + 0.5) - d start_time) / 0.5

-- The statement to prove
theorem max_avg_speed_1_5_to_2 (d : ℝ → ℝ) :
  (∀ s, 0 ≤ s ∧ s < 1.5 → avg_speed d s ≤ avg_speed d 1.5) →
  avg_speed d 1.5 > avg_speed d 2 :=
sorry

end max_avg_speed_1_5_to_2_l167_167611


namespace krakozyabrs_count_l167_167109

theorem krakozyabrs_count :
  ∀ (K : Type) [fintype K] (has_horns : K → Prop) (has_wings : K → Prop), 
  (∀ k, has_horns k ∨ has_wings k) →
  (∃ n, (∀ k, has_horns k → has_wings k) → fintype.card ({k | has_horns k} : set K) = 5 * n) →
  (∃ n, (∀ k, has_wings k → has_horns k) → fintype.card ({k | has_wings k} : set K) = 4 * n) →
  (∃ T, 25 < T ∧ T < 35 ∧ T = 32) := 
by
  intros K _ has_horns has_wings _ _ _
  sorry

end krakozyabrs_count_l167_167109


namespace even_digit_numbers_count_eq_100_l167_167456

-- Definition for the count of distinct three-digit positive integers with only even digits
def count_even_digit_three_numbers : ℕ :=
  let hundreds_place := {2, 4, 6, 8}.card
  let tens_units_place := {0, 2, 4, 6, 8}.card
  hundreds_place * tens_units_place * tens_units_place

-- Theorem stating the count of distinct three-digit positive integers with only even digits is 100
theorem even_digit_numbers_count_eq_100 : count_even_digit_three_numbers = 100 :=
by sorry

end even_digit_numbers_count_eq_100_l167_167456


namespace profit_percentage_l167_167284

theorem profit_percentage (C S : ℝ) (hC : C = 800) (hS : S = 1080) :
  ((S - C) / C) * 100 = 35 := 
by
  sorry

end profit_percentage_l167_167284


namespace largest_among_given_numbers_l167_167718

theorem largest_among_given_numbers (a b c d : ℝ) (ha : a = -real.pi) (hb : b = real.sqrt 25) (hc : c = abs (-8)) (hd : d = 0) :
  c = 8 ∧ c > b ∧ c > d ∧ c > a := by
  sorry

end largest_among_given_numbers_l167_167718


namespace proof_problem_l167_167845

noncomputable def f : ℝ → ℝ := sorry -- This should be defined according to the given conditions in the full proof, omitted here for brevity.

theorem proof_problem (f : ℝ → ℝ) (h1 : ∀ a b : ℝ, f (a + b) = f a * f b) (h2 : f 1 = 2) :
    (Finset.range (1008)).sum (λ k, f (2 * (k + 1)) / f (2 * (k + 1) - 1)) = 2016 := 
by
  sorry -- Proof is not required here.

end proof_problem_l167_167845


namespace cos_alpha_of_gp_and_ap_l167_167852

theorem cos_alpha_of_gp_and_ap 
  (α β γ : ℝ)
  (h1 : β = 2 * α) 
  (h2 : γ = 4 * α)
  (h3 : sin β = (sin α + sin γ) / 2) :
  cos α = -1 / 2 :=
by
  sorry

end cos_alpha_of_gp_and_ap_l167_167852


namespace area_of_rectangle_l167_167893

variable (AB BC CD AE: ℝ)
variable (E: Point)
variable (rectangle: Rectangle)
variable (midpoint: Midpoint BC E)
variable [HasLength AE 9]
variable [HasSum (AB + BC + CD) 20]

theorem area_of_rectangle (AB BC CD AE: ℝ) (E: Point) [rectangle AB BC CD] [midpoint BC E] (h₁: AE = 9) (h₂: AB + BC + CD = 20): 2 * AB * (BC / 2) = 19 :=
by
  simplify
  field_simp
  ring
  sorry

end area_of_rectangle_l167_167893


namespace count_distinct_three_digit_even_numbers_l167_167470

theorem count_distinct_three_digit_even_numbers : 
  let even_digits := {0, 2, 4, 6, 8} in
  let first_digit_choices := {2, 4, 6, 8} in
  let second_and_third_digit_choices := even_digits in
  (finset.card first_digit_choices) * 
  (finset.card second_and_third_digit_choices) *
  (finset.card second_and_third_digit_choices) = 100 := by
  let even_digits := {0, 2, 4, 6, 8}
  let first_digit_choices := {2, 4, 6, 8}
  let second_and_third_digit_choices := even_digits
  have h1 : finset.card first_digit_choices = 4 := by simp
  have h2 : finset.card second_and_third_digit_choices = 5 := by simp
  calc (finset.card first_digit_choices) * 
       (finset.card second_and_third_digit_choices) *
       (finset.card second_and_third_digit_choices)
       = 4 * 5 * 5 : by rw [h1, h2]
    ... = 100 : by norm_num

end count_distinct_three_digit_even_numbers_l167_167470


namespace total_kilometers_ridden_l167_167165

theorem total_kilometers_ridden :
  ∀ (d1 d2 d3 d4 : ℕ),
    d1 = 40 →
    d2 = 50 →
    d3 = d2 - d2 / 2 →
    d4 = d1 + d3 →
    d1 + d2 + d3 + d4 = 180 :=
by 
  intros d1 d2 d3 d4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_kilometers_ridden_l167_167165


namespace tea_garden_grain_field_l167_167179

theorem tea_garden_grain_field (x y : ℝ) (total_land : ℝ) (vegetable_percentage : ℝ)
  (vx : x = 2 * y - 3) (vy : x + y = 54) (vt : total_land = 60) (vp : vegetable_percentage = 0.1) :
  x + y = total_land - vegetable_percentage * total_land ∧ x = 2 * y - 3 :=
by
  have h1 : 60 - 0.1 * 60 = 54, by norm_num,
  rw [vt, vp] at h1,
  split,
  { exact vy },
  { exact vx }

end tea_garden_grain_field_l167_167179


namespace neighboring_minutes_l167_167274

def sector_state : Type := list bool -- true represents white, false represents red.

noncomputable def repaint (s : sector_state) (i : ℕ) : sector_state :=
let (start, end) := s.splitAt i in
let (middle, rest) := end.splitAt 500 in
start ++ list.map bnot middle ++ rest

-- Condition 1: Initial state of the sectors (all white)
def initial_state : sector_state := list.repeat true 1000

-- Condition 2: Repainting mechanics (defined in repaint function)

-- Condition 3: In some minute, the number of white sectors did not change
def unchanged_white_sectors (s1 s2 : sector_state) : Prop :=
list.count true s1 = list.count true s2

theorem neighboring_minutes (t : ℕ) :
  (unchanged_white_sectors (repaint (initial_state) t) (initial_state)) →
  (unchanged_white_sectors (repaint (initial_state) (t - 1)) (initial_state) ∨ 
  unchanged_white_sectors (repaint (initial_state) (t + 1)) (initial_state)) :=
sorry

end neighboring_minutes_l167_167274


namespace log_base_4_of_2_l167_167365

theorem log_base_4_of_2 : log 4 2 = 1 / 2 :=
by sorry

end log_base_4_of_2_l167_167365


namespace reflection_matrix_correct_l167_167531

-- Given conditions
def normal_vector : EuclideanSpace ℝ (Fin 3) := ![2, -1, 1]

-- Define the reflection matrix
def reflection_matrix := ![
  ![-1/3, -1/3, 1/3],
  ![1/3, 2/3, -1/3],
  ![-5/3, 5/3, 2/3]
]

-- Problem statement: Proving that this matrix correctly reflects any vector through the plane with the given normal vector.
theorem reflection_matrix_correct (u : EuclideanSpace ℝ (Fin 3)) :
  let S := reflection_matrix,
      Q := normal_vector
  in ∀ u, S • u = u - 2 * ((u ⬝ Q) / (Q ⬝ Q)) • Q :=
sorry

end reflection_matrix_correct_l167_167531


namespace distinct_three_digit_even_integers_count_l167_167467

theorem distinct_three_digit_even_integers_count : 
  let even_digits := {0, 2, 4, 6, 8}
  ∃ h : Finset ℕ, h = {2, 4, 6, 8} ∧ 
     (∏ x in h, 5 * 5) = 100 :=
by
  let even_digits : Finset ℕ := {0, 2, 4, 6, 8}
  let h : Finset ℕ := {2, 4, 6, 8}
  have : ∏ x in h, 5 * 5 = 100 := sorry
  exact ⟨h, rfl, this⟩

end distinct_three_digit_even_integers_count_l167_167467


namespace max_gcd_13n_plus_4_8n_plus_3_l167_167315

theorem max_gcd_13n_plus_4_8n_plus_3 : ∃ n : ℕ, n > 0 ∧ Int.gcd (13 * n + 4) (8 * n + 3) = 11 := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l167_167315


namespace find_a_and_b_l167_167433

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the curve equation
def curve (a b x : ℝ) : ℝ := x^3 + a * x + b

-- Define the derivative of the curve
def curve_derivative (a x : ℝ) : ℝ := 3 * x^2 + a

-- Main theorem to prove a = -1 and b = 3 given tangency conditions
theorem find_a_and_b 
  (k : ℝ) (a b : ℝ) (tangent_point : ℝ × ℝ)
  (h_tangent : tangent_point = (1, 3))
  (h_line : line k tangent_point.1 = tangent_point.2)
  (h_curve : curve a b tangent_point.1 = tangent_point.2)
  (h_slope : curve_derivative a tangent_point.1 = k) : 
  a = -1 ∧ b = 3 := 
by
  sorry

end find_a_and_b_l167_167433


namespace inflation_two_years_correct_real_rate_of_return_correct_l167_167669

-- Define the calculation for inflation over two years
def inflation_two_years (r : ℝ) : ℝ :=
  ((1 + r)^2 - 1) * 100

-- Define the calculation for the real rate of return
def real_rate_of_return (r : ℝ) (infl_rate : ℝ) : ℝ :=
  ((1 + r * r) / (1 + infl_rate / 100) - 1) * 100

-- Prove the inflation over two years is 3.0225%
theorem inflation_two_years_correct :
  inflation_two_years 0.015 = 3.0225 :=
by
  sorry

-- Prove the real yield of the bank deposit is 11.13%
theorem real_rate_of_return_correct :
  real_rate_of_return 0.07 3.0225 = 11.13 :=
by
  sorry

end inflation_two_years_correct_real_rate_of_return_correct_l167_167669


namespace distribution_plans_l167_167309

theorem distribution_plans (teachers schools : ℕ) (no_more_than_2 : ∀ s, multiset.countp (λ t, t = s) teachers ≤ 2) : 
  (teachers = 3) → (schools = 4) → ∃ plans : ℕ, plans = 60 :=
by
  intros h1 h2
  existsi 60
  sorry

end distribution_plans_l167_167309


namespace find_A_l167_167199

theorem find_A (A B : ℕ) (h1 : 15 = 3 * A) (h2 : 15 = 5 * B) : A = 5 := 
by 
  sorry

end find_A_l167_167199


namespace repeating_block_length_7_div_13_l167_167327

theorem repeating_block_length_7_div_13 : ∀ n : ℕ, (n = 7 / 13) → (length_of_repeating_block (decimal_expansion n) = 6) :=
begin
  sorry
end

end repeating_block_length_7_div_13_l167_167327


namespace krakozyabr_count_l167_167117

variable (n H W T : ℕ)
variable (h1 : H = 5 * n) -- 20% of the 'krakozyabrs' with horns also have wings
variable (h2 : W = 4 * n) -- 25% of the 'krakozyabrs' with wings also have horns
variable (h3 : T = H + W - n) -- Total number of 'krakozyabrs' using inclusion-exclusion
variable (h4 : 25 < T)
variable (h5 : T < 35)

theorem krakozyabr_count : T = 32 := by
  sorry

end krakozyabr_count_l167_167117


namespace jill_sold_fraction_l167_167888

variables (num_single_calves : ℕ) (num_twin_calves : ℕ) 
variables (trade_calves : ℕ) (trade_adults : ℕ)
variables (total_after_sale : ℕ) (num_sold : ℕ)

-- Given conditions:
def jill_conditions (num_single_calves = 9) (num_twin_calves = 5)
  (trade_calves = 8) (trade_adults = 2) (total_after_sale = 18) : Prop :=
  let total_calves := num_single_calves + (num_twin_calves * 2)
  let initial_adults := total_after_sale - (total_calves - trade_calves + trade_adults)
  let total_before_sale := total_calves + initial_adults - trade_calves + trade_adults
  let num_sold := total_before_sale - total_after_sale
  ∃ f : ℚ, f = (num_sold : ℚ) / total_before_sale ∧ f = 4 / 13

-- Theorem to prove:
theorem jill_sold_fraction :
  jill_conditions num_single_calves num_twin_calves trade_calves trade_adults total_after_sale :=
sorry

end jill_sold_fraction_l167_167888


namespace mindy_tax_rate_l167_167567

variables (M : ℝ) -- Denote M as a real number for Mork's income.
variables (r : ℝ) -- Denote r as a real number for Mindy's tax rate.

theorem mindy_tax_rate (h1 : 0.45 * M) -- Mork's tax amount
                        (h2 : 4 * M) -- Mindy's income
                        (h3 : (0.45 * M + 4 * M * r) / (M + 4 * M) = 0.29) -- Combined tax rate condition
                        : r = 0.25 := sorry

end mindy_tax_rate_l167_167567


namespace find_c_l167_167955

-- Define the function f
def f (z : ℂ) : ℂ := ((1 - complex.i * real.sqrt 2) * z + (3 * real.sqrt 2 - 5 * complex.i)) / 4

-- Define the constant c
def c : ℂ := (9 * real.sqrt 2 - 15 * complex.i * real.sqrt 2 + 10 - 6 * complex.i) / 11

-- State the theorem
theorem find_c : f c = c :=
by
  -- (Proof will go here)
  sorry

end find_c_l167_167955


namespace new_paint_intensity_l167_167935

theorem new_paint_intensity : 
  let I_original : ℝ := 0.5
  let I_added : ℝ := 0.2
  let replacement_fraction : ℝ := 1 / 3
  let remaining_fraction : ℝ := 2 / 3
  let I_new := remaining_fraction * I_original + replacement_fraction * I_added
  I_new = 0.4 :=
by
  -- sorry is used to skip the actual proof
  sorry

end new_paint_intensity_l167_167935


namespace smaller_number_l167_167971

theorem smaller_number (x y : ℤ) (h1 : x + y = 12) (h2 : x - y = 20) : y = -4 := 
by 
  sorry

end smaller_number_l167_167971


namespace percent_neither_question_l167_167065

def inclusion_exclusion (A B Both : ℝ) : ℝ := A + B - Both

theorem percent_neither_question (A B Both Neither : ℝ) :
    A = 0.75 → 
    B = 0.30 → 
    Both = 0.25 → 
    Neither = 1 - inclusion_exclusion A B Both → 
    Neither = 0.20 :=
by
  intros hA hB hBoth hNeither
  rw [hA, hB, hBoth, inclusion_exclusion, hNeither]
  linarith

end percent_neither_question_l167_167065


namespace myopia_gender_independence_test_l167_167994

/-- To investigate the situation of myopia among middle school students, a school has 150 male
students, among whom 80 are myopic, and 140 female students, among whom 70 are myopic. 
The most convincing method to test whether myopia among these middle school students is related 
to gender is the Independence Test. -/
theorem myopia_gender_independence_test (male_students : ℕ) (male_myopic : ℕ)
  (female_students : ℕ) (female_myopic : ℕ) :
  male_students = 150 ∧ male_myopic = 80 ∧
  female_students = 140 ∧ female_myopic = 70 →
  "most_convincing_method" = "C: Independence Test" :=
by
  intro h
  cases h with hmales hmconv
  cases hmconv with hmalesmyopic hfemales
  cases hfemales with hfemconv hfemalesmyopic
  exact sorry

end myopia_gender_independence_test_l167_167994


namespace distinct_paper_count_l167_167984

theorem distinct_paper_count (n : ℕ) :
  let sides := 4  -- 4 rotations and 4 reflections
  let identity_fixed := n^25 
  let rotation_90_fixed := n^7
  let rotation_270_fixed := n^7
  let rotation_180_fixed := n^13
  let reflection_fixed := n^15
  (1 / 8) * (identity_fixed + 4 * reflection_fixed + rotation_180_fixed + 2 * rotation_90_fixed) 
  = (1 / 8) * (n^25 + 4 * n^15 + n^13 + 2 * n^7) :=
  by 
    sorry

end distinct_paper_count_l167_167984


namespace euler_totient_divisibility_l167_167583

theorem euler_totient_divisibility (n : ℕ) (h : n > 0) : n ∣ Nat.totient (2^n - 1) := by
  sorry

end euler_totient_divisibility_l167_167583


namespace max_gcd_l167_167313

theorem max_gcd (n : ℕ) (h : 0 < n) : ∀ n, ∃ d ≥ 1, d ∣ 13 * n + 4 ∧ d ∣ 8 * n + 3 → d ≤ 9 :=
begin
  sorry
end

end max_gcd_l167_167313


namespace smallest_value_proof_l167_167064

noncomputable def smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) : Prop :=
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/x^2

theorem smallest_value_proof (x : ℝ) (h : 0 < x ∧ x < 1) : smallest_value x h :=
  sorry

end smallest_value_proof_l167_167064


namespace repeating_decimal_denominators_count_l167_167597

theorem repeating_decimal_denominators_count
  (a b c : ℕ)
  (h₀ : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)
  (h₁ : a ≠ 9 ∨ b ≠ 9 ∨ c ≠ 9)
  (h_digits : ∀ x ∈ [a, b, c], x < 10) :
  {d : ℕ | ∃ k : ℕ, 0.overline abc = k / d ∧ nat.gcd (abc a b c) 999 = 1}.card = 7 :=
sorry

end repeating_decimal_denominators_count_l167_167597


namespace complex_modulus_condition_l167_167072

theorem complex_modulus_condition (a : ℝ) : 
  abs ((a-2) + (a+2)*complex.I) = 4 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end complex_modulus_condition_l167_167072


namespace nat_nums_division_by_7_l167_167760

theorem nat_nums_division_by_7 (n : ℕ) : 
  (∃ q r, n = 7 * q + r ∧ q = r ∧ 1 ≤ r ∧ r < 7) ↔ 
  n = 8 ∨ n = 16 ∨ n = 24 ∨ n = 32 ∨ n = 40 ∨ n = 48 := by
  sorry

end nat_nums_division_by_7_l167_167760


namespace mask_production_july_l167_167205

theorem mask_production_july 
  (initial_production : ℕ)
  (doubled_each_month : ∀ n : ℕ, initial_production * (2 ^ n))
  (march_production : initial_production = 3000) : 
  doubled_each_month 4 = 48000 := 
by
  -- initial_production = 3000
  -- doubled_each_month 0 = initial_production = 3000
  -- doubled_each_month 1 = initial_production * 2 = 6000
  -- doubled_each_month 2 = initial_production * 2^2 = 12000
  -- doubled_each_month 3 = initial_production * 2^3 = 24000
  -- doubled_each_month 4 = initial_production * 2^4 = 48000
  sorry

end mask_production_july_l167_167205


namespace bridge_length_l167_167679

variable (speed : ℝ) (time_minutes : ℝ)
variable (time_hours : ℝ := time_minutes / 60)

theorem bridge_length (h1 : speed = 5) (h2 : time_minutes = 15) : 
  speed * time_hours = 1.25 := by
  sorry

end bridge_length_l167_167679


namespace a0_a1_sum_zero_l167_167969

noncomputable def sum_geometric_series (r n : ℕ) : ℤ :=
  (r^(n + 1) - 1) / (r - 1)

def base_representation (k : ℕ) (n : ℤ) : List ℕ :=
  List.unfoldr (λ x, if x = 0 then none else some (x % k, x / k)) n

theorem a0_a1_sum_zero :
  let S := sum_geometric_series 2018 (2016 * 2017^2)
  let S_mod := S % 2017
  (S_mod = 0) → (let a := base_representation 2017 S_mod in a.headD 0 + a.tail.toList.headD 0 = 0) :=
by
  sorry

end a0_a1_sum_zero_l167_167969


namespace new_efficiency_is_correct_l167_167189

-- Initial conditions and known information
variable (T_H T_C : ℝ)
variable (T_H_pos : 0 < T_H)
variable (T_C_pos : 0 < T_C)
variable (initial_efficiency : ℝ)
variable (T_H_prime : ℝ := 1.5 * T_H)
variable (T_C_prime : ℝ := 0.5 * T_C)

-- Initial efficiency is 50%
def initial_efficiency_condition : Prop := initial_efficiency = 0.5

-- Efficiency formula for an ideal heat engine
def efficiency_formula (T_C T_H : ℝ) : ℝ := 1 - T_C / T_H

-- New efficiency calculation
def new_efficiency (T_C_prime T_H_prime : ℝ) : ℝ := efficiency_formula T_C_prime T_H_prime

-- Theorem to prove
theorem new_efficiency_is_correct
  (T_H T_C : ℝ)
  (T_H_pos : 0 < T_H)
  (T_C_pos : 0 < T_C)
  (initial_efficiency_condition : initial_efficiency = 0.5)
  (T_H_prime : ℝ := 1.5 * T_H)
  (T_C_prime : ℝ := 0.5 * T_C)
  : new_efficiency T_C_prime T_H_prime = 0.8333 :=
begin
  sorry
end

end new_efficiency_is_correct_l167_167189


namespace zuzka_paint_containers_l167_167242

def side_lengths : list ℕ := [1, 2, 3, 4, 5]
def total_surface_area (l : list ℕ) : ℕ :=
  6 * (l.map (λ a, a * a)).sum
def unpainted_area (l : list ℕ) : ℕ :=
  (2 * (l.dropLast.map (λ a, a * a)).sum) + (l.lastD 0) * (l.lastD 0)
def painted_surface_area (l : list ℕ) : ℕ :=
  total_surface_area l - unpainted_area l
def number_of_paint_containers (l : list ℕ) (container_area : ℕ) : ℕ :=
  (painted_surface_area l + container_area - 1) / container_area -- ceiling division 

theorem zuzka_paint_containers : number_of_paint_containers side_lengths 5 = 49 :=
by
  sorry

end zuzka_paint_containers_l167_167242


namespace symmetric_point_xOy_l167_167512

theorem symmetric_point_xOy (m n p : ℝ) : 
    symmetric_in_xOy m n p = (m, n, -p) := 
by 
  sorry

noncomputable def symmetric_in_xOy (x y z: ℝ) : ℝ × ℝ × ℝ :=
(m, n, -z)

end symmetric_point_xOy_l167_167512


namespace work_done_by_force_l167_167283

theorem work_done_by_force:
  ∫ x in 0..1, (1 + Real.exp x) = Real.exp 1 :=
by
  sorry

end work_done_by_force_l167_167283


namespace length_DE_l167_167214

def AB : ℝ := 7
def BC : ℝ := 8
def EF (DE : ℝ) : ℝ := DE / 2
def area_rectangle : ℝ := AB * BC
def area_triangle (DE : ℝ) : ℝ := (DE * EF DE) / 2

theorem length_DE (DE : ℝ) (h : area_rectangle = area_triangle DE) : DE = 4 * Real.sqrt 14 :=
by
  unfold area_rectangle at h
  unfold area_triangle at h
  sorry

end length_DE_l167_167214


namespace average_games_played_l167_167332

theorem average_games_played :
  let games := [1, 3, 5, 7, 9]
  let num_teams := [4, 3, 2, 4, 6]
  let total_games := List.zipWith (*) games num_teams |>.sum
  let total_teams := num_teams.sum
  let average_games := (total_games : ℚ) / total_teams
  let rounded_average := Int.ofRat(average_games)
  rounded_average = 6 := 
by 
  sorry

end average_games_played_l167_167332


namespace number_of_valid_elements_is_130_l167_167561

open Set

def A := { (x1 : ℤ, x2 : ℤ, x3 : ℤ, x4 : ℤ, x5 : ℤ) |
  x1 ∈ {-1, 0, 1} ∧
  x2 ∈ {-1, 0, 1} ∧
  x3 ∈ {-1, 0, 1} ∧
  x4 ∈ {-1, 0, 1} ∧
  x5 ∈ {-1, 0, 1}
}

noncomputable def countValidElements : ℕ :=
  card { (x1, x2, x3, x4, x5) ∈ A | 1 ≤ |x1| + |x2| + |x3| + |x4| + |x5| ∧
                                          |x1| + |x2| + |x3| + |x4| + |x5| ≤ 3 }

theorem number_of_valid_elements_is_130 :
  countValidElements = 130 := 
sorry

end number_of_valid_elements_is_130_l167_167561


namespace M_coordinates_l167_167077

noncomputable def coordinates_of_M (m : ℝ) : (ℝ × ℝ) :=
(4 * m + 4, 3 * m - 6)

def coordinates_of_N : (ℝ × ℝ) := (-8, 12)

theorem M_coordinates :
  ∃ m : ℝ, coordinates_of_M m = (28, 12) :=
by
  -- given conditions
  have h1 : ∀ m : ℝ, (3 * m - 6) = 12 → m = 6, by
    intro m h
    rw [←sub_eq_zero, sub_eq_add_neg, neg_add_eq_sub, ←mul_right_inj' (show (3 : ℝ) ≠ 0, by norm_num)] at h
    exact h

  existsi 6
  simp [coordinates_of_M]
  exact ⟨rfl, rfl⟩


end M_coordinates_l167_167077


namespace determine_x_l167_167337

theorem determine_x (x : ℚ) (n : ℤ) (d : ℚ) 
  (h_cond : x = n + d)
  (h_floor : n = ⌊x⌋)
  (h_d : 0 ≤ d ∧ d < 1)
  (h_eq : ⌊x⌋ + x = 17 / 4) :
  x = 9 / 4 := sorry

end determine_x_l167_167337


namespace cut_into_segments_l167_167992

noncomputable theory

-- Define the lengths of the sticks and the condition on their sum
variables (n : ℕ) (a1 a2 a3 : ℕ)
-- Assume each stick's length is at least n
hypothesis h1 : a1 ≥ n
hypothesis h2 : a2 ≥ n
hypothesis h3 : a3 ≥ n
-- Assume the sum of the lengths of the three sticks equals n(n+1)/2
hypothesis h_sum : a1 + a2 + a3 = n * (n + 1) / 2

-- Proof goal: show it is possible to cut the sticks into lengths 1, 2,..., n
theorem cut_into_segments : ∃ (segments : list ℕ), 
    (∀ x ∈ segments, x ∈ (list.range (n + 1)).tail) ∧ 
    a1 + a2 + a3 = segments.sum :=
sorry

end cut_into_segments_l167_167992


namespace jasmine_longest_book_l167_167884

theorem jasmine_longest_book (S L : ℕ)
  (h1 : S = 1 / 4 * L)
  (h2 : 297 = 3 * S) :
  L = 396 :=
by 
  -- introduce assumption to convert the fractional multiplication to integer multiplication
  have h₁ : 4 * S = L := sorry,
  -- substitute h2 into h₁
  have h₃ : 4 * (297 / 3) = L := sorry,
  -- calculate the value L
  sorry

end jasmine_longest_book_l167_167884


namespace evaluate_f_pi_over_4_l167_167824

noncomputable def f (x φ : ℝ) : ℝ := sin (x + 2 * φ) - 2 * sin φ * cos (x + φ)

theorem evaluate_f_pi_over_4 (φ : ℝ) : f (π / 4) φ = √2 / 2 :=
by
  sorry

end evaluate_f_pi_over_4_l167_167824


namespace buddy_baseball_cards_l167_167570

theorem buddy_baseball_cards:
  let Monday_cards := 30 in
  let Tuesday_cards := Monday_cards / 2 in
  let Thursday_cards := 32 in
  let Thursday_purchased := Tuesday_cards / 3 in
  let Wednesday_cards := Thursday_cards - Thursday_purchased in
  let Wednesday_purchased := Wednesday_cards - Tuesday_cards in
  Wednesday_purchased = 12 :=
by
  sorry

end buddy_baseball_cards_l167_167570


namespace equilateral_triangle_ABC_l167_167574

-- Let O be the vertex of the angle.
variables (O A A1 B B1 C : Point)

-- Distances from the vertex.
variables (p q : ℝ)

-- Assume the points follow given distances and relationships.
-- Point A is at distance p from the vertex.
def distance_A : dist O A = p := sorry
-- Point A1 is at distance 2q from the vertex.
def distance_A1 : dist O A1 = 2 * q := sorry
-- Point B is at distance q from the vertex.
def distance_B : dist O B = q := sorry
-- Point B1 is at distance 2p from the vertex.
def distance_B1 : dist O B1 = 2 * p := sorry
-- Angle AOB is 60 degrees.
def angle_AOB : angle O A B = 60 := sorry
-- Angle A1OB1 is 60 degrees.
def angle_A1OB1 : angle O A1 B1 = 60 := sorry
-- Point C is the midpoint of A1B1.
def midpoint_C : midpoint A1 B1 C := sorry

-- We need to prove that triangle ABC is equilateral.
theorem equilateral_triangle_ABC : 
  is_equilateral (triangle A B C) := sorry

end equilateral_triangle_ABC_l167_167574


namespace Natalia_total_distance_l167_167162

variable (d_m d_t d_w d_r d_total : ℕ)

-- Conditions
axiom cond1 : d_m = 40
axiom cond2 : d_t = 50
axiom cond3 : d_w = d_t / 2
axiom cond4 : d_r = d_m + d_w

-- Question and answer
theorem Natalia_total_distance : 
  d_total = d_m + d_t + d_w + d_r → 
  d_total = 180 := 
by
  intros h
  simp [cond1, cond2, cond3, cond4] at h
  rw [cond1, cond2, cond3, cond4] in h
  simp at h
  exact h

end Natalia_total_distance_l167_167162


namespace find_m_l167_167050

-- Definitions based on the problem conditions
def a (m : ℝ) : ℝ × ℝ := (2 * m + 1, -1 / 2)
def b (m : ℝ) : ℝ × ℝ := (2 * m, 1)

-- The condition that the vectors are parallel
def vectors_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Lean statement using the conditions to prove m = -1 / 3
theorem find_m (m : ℝ) (h : vectors_parallel (a m) (b m)) : 
  m = -1 / 3 :=
sorry

end find_m_l167_167050


namespace weight_of_smallest_box_l167_167206

variables (M S L : ℕ)

theorem weight_of_smallest_box
  (h1 : M + S = 83)
  (h2 : L + S = 85)
  (h3 : L + M = 86) :
  S = 41 :=
sorry

end weight_of_smallest_box_l167_167206


namespace problem_1_problem_2_l167_167098

-- Problem (1)
theorem problem_1 (A : ℝ) (a b c : ℝ) (h1 : a = 2 * real.sqrt 6) (h2 : b = 3) (h3 : c - b = 2 * b * real.cos A) : 
  c = 5 :=
sorry

-- Problem (2)
theorem problem_2 (A B C : ℝ) (h1 : C = real.pi / 2) (h2 : A + B = real.pi / 2) : 
  B = real.pi / 6 :=
sorry

end problem_1_problem_2_l167_167098


namespace inheritance_value_l167_167161

def inheritance_proof (x : ℝ) (federal_tax_ratio : ℝ) (state_tax_ratio : ℝ) (total_tax : ℝ) : Prop :=
  let federal_taxes := federal_tax_ratio * x
  let remaining_after_federal := x - federal_taxes
  let state_taxes := state_tax_ratio * remaining_after_federal
  let total_taxes := federal_taxes + state_taxes
  total_taxes = total_tax

theorem inheritance_value :
  inheritance_proof 41379 0.25 0.15 15000 :=
by
  sorry

end inheritance_value_l167_167161


namespace divide_triangle_into_two_equal_parts_l167_167406

-- Definitions for the problem
variable {A B C P Q R : Type} [PlaneGeometry A B C]

-- Additional conditions required
axiom P_on_perimeter_But_Not_Vertex : lies_on_perimeter_but_not_vertex P A B C
axiom Q_inside_triangle : lies_inside_triangle Q A B C
axiom R_on_perimeter : lies_on_perimeter R A B C

theorem divide_triangle_into_two_equal_parts :
  ∃ R : Point, 
    lies_on_perimeter R (triangle A B C) ∧ 
    polygonal_line_divides_area_eq P Q R (triangle A B C) :=
begin
  sorry
end

end divide_triangle_into_two_equal_parts_l167_167406


namespace fifth_term_is_31_l167_167737

/-- 
  The sequence is formed by summing consecutive powers of 2. 
  Define the sequence and prove the fifth term is 31.
--/
def sequence_sum (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, 2^k)

theorem fifth_term_is_31 : sequence_sum 5 = 31 :=
by sorry

end fifth_term_is_31_l167_167737


namespace distinct_triangles_count_l167_167056

def num_combinations (n k : ℕ) : ℕ := n.choose k

def count_collinear_sets_in_grid (grid_size : ℕ) : ℕ :=
  let rows := grid_size
  let cols := grid_size
  let diagonals := 2
  rows + cols + diagonals

noncomputable def distinct_triangles_in_grid (grid_size n k : ℕ) : ℕ :=
  num_combinations n k - count_collinear_sets_in_grid grid_size

theorem distinct_triangles_count :
  distinct_triangles_in_grid 3 9 3 = 76 := 
by 
  sorry

end distinct_triangles_count_l167_167056


namespace reflection_matrix_correct_l167_167532

-- Given conditions
def normal_vector : EuclideanSpace ℝ (Fin 3) := ![2, -1, 1]

-- Define the reflection matrix
def reflection_matrix := ![
  ![-1/3, -1/3, 1/3],
  ![1/3, 2/3, -1/3],
  ![-5/3, 5/3, 2/3]
]

-- Problem statement: Proving that this matrix correctly reflects any vector through the plane with the given normal vector.
theorem reflection_matrix_correct (u : EuclideanSpace ℝ (Fin 3)) :
  let S := reflection_matrix,
      Q := normal_vector
  in ∀ u, S • u = u - 2 * ((u ⬝ Q) / (Q ⬝ Q)) • Q :=
sorry

end reflection_matrix_correct_l167_167532


namespace intersection_of_symmetric_set_and_naturals_l167_167848

namespace SymmetricSet

open Set

-- Definitions
def symmetric_set (A : Set ℤ) : Prop := ∀ x : ℤ, x ∈ A → -x ∈ A

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k ∨ x = 0 ∨ x = k^2 + k}

def B : Set ℕ := {n | n ∈ ℕ}

-- Correct answer
def answer : Set ℤ := {0, 6}

-- Proof statement
theorem intersection_of_symmetric_set_and_naturals (hA : symmetric_set A) : 
  (A ∩ B : Set ℤ) = answer := 
by
  sorry
  
end SymmetricSet

end intersection_of_symmetric_set_and_naturals_l167_167848


namespace sum_of_digits_133131_l167_167620

noncomputable def extract_digits_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.foldl (· + ·) 0

theorem sum_of_digits_133131 :
  let ABCDEF := 665655 / 5
  extract_digits_sum ABCDEF = 12 :=
by
  sorry

end sum_of_digits_133131_l167_167620


namespace inequality_proof_l167_167803

variable {α β γ : ℝ}

theorem inequality_proof (h1 : β * γ ≠ 0) (h2 : (1 - γ^2) / (β * γ) ≥ 0) :
  10 * (α^2 + β^2 + γ^2 - β * γ^2) ≥ 2 * α * β + 5 * α * γ :=
sorry

end inequality_proof_l167_167803


namespace range_of_a_l167_167018

noncomputable def f (x a : ℝ) : ℝ := (x^2 + (a - 1) * x + 1) * Real.exp x

theorem range_of_a :
  (∀ x, f x a + Real.exp 2 ≥ 0) ↔ (-2 ≤ a ∧ a ≤ Real.exp 3 + 3) :=
sorry

end range_of_a_l167_167018


namespace move_possible_without_adjacent_l167_167919

open Function
open FiniteCardinals
open Set

-- Define the board and the placement rules
def board_position (n : ℕ) := (Fin n) × (Fin n)

def no_adjacent (n : ℕ) (positions : Finset (board_position n)) : Prop :=
  ∀ pos₁ pos₂ ∈ positions, pos₁ ≠ pos₂ → ¬ adjacent_cells pos₁ pos₂

def adjacent_cells {n : ℕ} (pos₁ pos₂ : board_position n) : Prop :=
  abs (pos₁.fst - pos₂.fst) = 1 ∧ pos₁.snd = pos₂.snd ∨
  pos₁.fst = pos₂.fst ∧ abs (pos₁.snd - pos₂.snd) = 1

-- Lean statement for the math proof problem
theorem move_possible_without_adjacent (n : ℕ) (positions : Finset (board_position n)) :
  positions.card = n - 1 →
  no_adjacent n positions →
  ∃ pos' ∈ positions, ∃ new_pos : board_position n, adjacent_cells pos' new_pos ∧ no_adjacent n (positions.erase pos' ∪ {new_pos}) :=
sorry

end move_possible_without_adjacent_l167_167919


namespace two_fiftyth_term_of_omitted_sequence_l167_167652

def perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def omitted_sequence_term (k : ℕ) : ℕ :=
  Nat.find (λ n, k = n - (Nat.sqrt n - 1))

theorem two_fiftyth_term_of_omitted_sequence : omitted_sequence_term 250 = 265 :=
sorry

end two_fiftyth_term_of_omitted_sequence_l167_167652


namespace u_divisible_l167_167918

-- Define u(n, k) based on the given conditions
noncomputable def u : ℕ → ℕ → ℕ
| 1, 1 := 1
| n, k := if n > 1 ∧ k ≥ 1 ∧ k ≤ n then (nat.choose n k - ∑ d in (finset.filter (λ d, d > 1) (finset.divisors n)), if d ∣ k then u (n / d) (k / d) else 0) else 0

-- Prove that n divides u(n, k) for all 1 ≤ k ≤ n
theorem u_divisible (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) : n ∣ u n k := 
sorry

end u_divisible_l167_167918


namespace complex_root_sixth_power_sum_equals_38908_l167_167139

noncomputable def omega : ℂ :=
  -- By definition, omega should satisfy the below properties.
  -- The exact value of omega is not being defined, we will use algebraic properties in the proof.
  sorry

theorem complex_root_sixth_power_sum_equals_38908 : 
  ∀ (ω : ℂ), ω^3 = 1 ∧ ¬(ω.re = 1) → (2 - ω + 2 * ω^2)^6 + (2 + ω - 2 * ω^2)^6 = 38908 :=
by
  -- Proof will utilize given conditions:
  -- 1. ω^3 = 1
  -- 2. ω is not real (or ω.re is not 1)
  sorry

end complex_root_sixth_power_sum_equals_38908_l167_167139


namespace semicircle_radius_is_10_l167_167743
-- Import the necessary libraries

-- Define the problem conditions in Lean 4
noncomputable def isosceles_triangle_radius : ℝ :=
  let AC := 20 in  -- base of the triangle
  let AD := AC / 2 in  -- half of the base since D is the midpoint
  let AB := 20 in  -- the legs are equal to the height
  let BD := real.sqrt (AB^2 - AD^2) in  -- using the Pythagorean theorem
  AC / 2  -- radius of the inscribed semicircle

-- Statement to prove
theorem semicircle_radius_is_10 :
  isosceles_triangle_radius = 10 :=
by
  sorry  -- The proof can be completed here

end semicircle_radius_is_10_l167_167743


namespace PE_eq_QE_l167_167712

variable (A B C D E M N P Q : Point)
variable (circle_CAMN circle_NMBD : Circle)
variable (line_CD line_AB : Line)

-- conditions
axiom tangent_ab1 : Tangent line_AB circle_CAMN
axiom tangent_ab2 : Tangent line_AB circle_NMBD
axiom line_cd_parallel_ab : Parallel line_CD line_AB
axiom m_between_c_d : Between C M D
axiom chord_na_cm_intersect_p : Intersect (Chord circle_CAMN A N) (Chord circle_CAMN C M) P
axiom chord_nb_md_intersect_q : Intersect (Chord circle_NMBD B N) (Chord circle_NMBD M D) Q
axiom rays_ca_db_meet_e : Meet (Ray C A) (Ray D B) E

-- objective
theorem PE_eq_QE : dist P E = dist Q E := 
by sorry

end PE_eq_QE_l167_167712


namespace solution_of_system_l167_167717

theorem solution_of_system :
  ∃ (x y : ℤ), (x + y = 2) ∧ (x + 2 * y = 3) ∧ (x = 1) ∧ (y = 1) :=
by
  use 1, 1
  split; norm_num
  split; norm_num
sorry

end solution_of_system_l167_167717


namespace total_distance_l167_167381

open Real

/-- Define the distance function between two points in 2D space -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

/-- Define the points in the problem -/
def point1 : ℝ × ℝ := (2, 2)
def point2 : ℝ × ℝ := (5, 9)
def point3 : ℝ × ℝ := (10, 12)

theorem total_distance :
  distance point1 point2 + distance point2 point3 = real.sqrt 58 + real.sqrt 34 := by
  sorry

end total_distance_l167_167381


namespace log_base_3_of_reciprocal_81_l167_167356

theorem log_base_3_of_reciprocal_81 : log 3 (1 / 81) = -4 :=
by
  sorry

end log_base_3_of_reciprocal_81_l167_167356


namespace no_perfect_square_l167_167897

theorem no_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h : ∃ (a : ℕ), p + q^2 = a^2) : ∀ (n : ℕ), n > 0 → ¬ (∃ (b : ℕ), p^2 + q^n = b^2) := 
by
  sorry

end no_perfect_square_l167_167897


namespace reflection_matrix_correct_l167_167537

noncomputable def reflection_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1/3, 4/3, 2/3], ![5/3, 1/3, 2/3], ![1/3, 4/3, 2/3]]

theorem reflection_matrix_correct (u : Fin 3 → ℝ) :
  let n : Fin 3 → ℝ := ![2, -1, 1]
  let q := u - (u ⬝ᴛ n) / (n ⬝ᴛ n) • n
  let s := 2 • q - u
  reflection_matrix ⬝ u = s :=
by
  let n : Fin 3 → ℝ := ![2, -1, 1]
  have h_n_dot_n : n ⬝ᴛ n = 6 := sorry
  let q := u - (u ⬝ᴛ n) / (n ⬝ᴛ n) • n
  let s := 2 • q - u
  have h_projection : reflection_matrix ⬝ u = s := sorry
  exact h_projection
  sorry

#check @reflection_matrix_correct

end reflection_matrix_correct_l167_167537


namespace asymptotes_of_hyperbola_l167_167088

theorem asymptotes_of_hyperbola (a : ℝ) :
  (∃ x y : ℝ, y^2 = 12 * x ∧ (x = 3) ∧ (y = 0)) →
  (a^2 = 9) →
  (∀ b c : ℝ, (b, c) ∈ ({(a, b) | (b = a/3 ∨ b = -a/3)})) :=
by
  intro h_focus_coincides vertex_condition
  sorry

end asymptotes_of_hyperbola_l167_167088


namespace inequality_proof_l167_167395

theorem inequality_proof (n : ℕ) (a b : ℕ → ℝ) (l k : ℕ) (h_pos_b : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < b i) (h_nonneg_a : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ a i) (hl : 0 < l) (hk : 0 < k) :
  (∑ i in finset.range n, (a i)^(l+k) / (b i)^l) ≥ 
  ((∑ i in finset.range n, a i)^(l+k) / (n^(k-1) * (∑ i in finset.range n, b i)^l)) :=
by
  sorry

end inequality_proof_l167_167395


namespace find_f_neg_one_l167_167043

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * real.sin x + 1

theorem find_f_neg_one :
  ∀ (a b : ℝ), f 1 a b = 5 → f (-1) a b = -3 := 
by
  sorry

end find_f_neg_one_l167_167043


namespace polar_coordinates_of_midpoint_l167_167438

-- Define the initial points A and B with their Cartesian coordinates
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 2 }
def B : Point := { x := -4, y := 0 }

-- Define the midpoint M of points A and B
def midpoint (P1 P2 : Point) : Point :=
  { x := (P1.x + P2.x) / 2, y := (P1.y + P2.y) / 2 }

noncomputable def M : Point := midpoint A B

-- Prove that the polar coordinates of M are (sqrt(5), π - atan (1/2))
theorem polar_coordinates_of_midpoint :
  let ρ := Real.sqrt ((M.x)^2 + (M.y)^2)
  let θ := Real.pi - Real.arctan (1 / 2)
  (ρ, θ) = (Real.sqrt 5, Real.pi - Real.arctan (1 / 2)) := by
  -- Adding proof part to establish the theorem; skipping it with sorry
  sorry

end polar_coordinates_of_midpoint_l167_167438


namespace count_distinct_three_digit_even_numbers_l167_167472

theorem count_distinct_three_digit_even_numbers : 
  let even_digits := {0, 2, 4, 6, 8} in
  let first_digit_choices := {2, 4, 6, 8} in
  let second_and_third_digit_choices := even_digits in
  (finset.card first_digit_choices) * 
  (finset.card second_and_third_digit_choices) *
  (finset.card second_and_third_digit_choices) = 100 := by
  let even_digits := {0, 2, 4, 6, 8}
  let first_digit_choices := {2, 4, 6, 8}
  let second_and_third_digit_choices := even_digits
  have h1 : finset.card first_digit_choices = 4 := by simp
  have h2 : finset.card second_and_third_digit_choices = 5 := by simp
  calc (finset.card first_digit_choices) * 
       (finset.card second_and_third_digit_choices) *
       (finset.card second_and_third_digit_choices)
       = 4 * 5 * 5 : by rw [h1, h2]
    ... = 100 : by norm_num

end count_distinct_three_digit_even_numbers_l167_167472


namespace carpet_dimensions_l167_167299

theorem carpet_dimensions (a b : ℕ) 
  (h1 : a^2 + b^2 = 38^2 + 55^2) 
  (h2 : a^2 + b^2 = 50^2 + 55^2) 
  (h3 : a ≤ b) : 
  (a = 25 ∧ b = 50) ∨ (a = 50 ∧ b = 25) :=
by {
  -- The proof would go here
  sorry
}

end carpet_dimensions_l167_167299


namespace lowest_fraction_taskA_l167_167863

-- Define the individual completion times.
def times : List ℝ := [3, 4, 6, 8, 12]

-- Define the work rates for each person (tasks per hour).
def work_rate (t : ℝ) : ℝ := 1 / t
def work_rates : List ℝ := times.map work_rate

-- Define the combined work rate of the three slowest workers for Task B.
def combined_work_rate_taskB : ℝ := 
  let slowest_three := List.take 3 (work_rates.qsort (· ≤ ·))
  slowest_three.foldr (· + ·) 0

-- Define the combined work rate for Task A.
def combined_work_rate_taskA : ℝ := combined_work_rate_taskB / 2

-- Prove that the lowest fraction of Task A that can be completed in 1 hour by exactly 3 of the people is 3/16.
theorem lowest_fraction_taskA : combined_work_rate_taskA = 3 / 16 := by sorry

end lowest_fraction_taskA_l167_167863


namespace eq_C1_curves_symmetric_l167_167149

-- Definitions
def curve_C (x : ℝ) : ℝ := x^3 - x

def curve_C1 (x t s : ℝ) : ℝ := (x - t)^3 - (x - t) + s

variables (x t s : ℝ)

-- Claim 1: Equation of curve C1
theorem eq_C1 : curve_C1 x t s = (x - t)^3 - (x - t) + s := 
  by sorry

-- Symmetry definitions
def point_A (t s : ℝ) : ℝ × ℝ := (t / 2, s / 2)

def symmetric_point (x1 y1 t s : ℝ) : ℝ × ℝ := (t - x1, s - y1)

-- Claim 2: Curves are symmetric around point A(t / 2, s / 2)
theorem curves_symmetric (x1 y1 : ℝ) (h_C : curve_C x1 = y1) :
  let pA := point_A t s in
  let p2 := symmetric_point x1 y1 t s in
  curve_C1 p2.1 t s = p2.2 :=
  by sorry

end eq_C1_curves_symmetric_l167_167149


namespace parabola_solution_l167_167087

variable (x b c : ℝ)
variable (x_range : ∀ x, -1 ≤ x ∧ x ≤ 4)

-- Part 1: Parabola passing through (3,0) and (0,-3)
def equation_of_parabola := y = x^2 + b * x + c
def point1 := equation_of_parabola 3 = 0
def point2 := equation_of_parabola 0 = -3

-- Part 2: Find the maximum and minimum values within the range -1 ≤ x ≤ 4
def min_value_of_quadratic := -4
def max_value_of_quadratic := 5

-- Part 3: Specific geometric conditions to solve for m
def m_value_case1 := m = (1 - Real.sqrt 29) / 2 ∨ m = (3 + Real.sqrt 13) / 2
def m_value_case2 := m = 0 ∨ m = 1 + Real.sqrt 10

theorem parabola_solution : ∃ b c, (point1 ∧ point2) ∧
  (∀ x, -1 ≤ x ∧ x ≤ 4 → (min_value_of_quadratic ≤ x^2 + b * x + c ∧ x^2 + b * x + c ≤ max_value_of_quadratic)) ∧
  m_value_case1 ∧
  m_value_case2 := sorry

end parabola_solution_l167_167087


namespace carpet_dimensions_l167_167301
open Real

theorem carpet_dimensions (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : ∃ k: ℝ, y = k * x)
  (h4 : ∃ α β: ℝ, α + k * β = 50 ∧ k * α + β = 55)
  (h5 : ∃ γ δ: ℝ, γ + k * δ = 38 ∧ k * γ + δ = 55) :
  x = 25 ∧ y = 50 :=
by sorry

end carpet_dimensions_l167_167301


namespace solution_set_inequality_l167_167387

open Real

theorem solution_set_inequality (x : ℝ) (n : ℕ) (h : n ≠ 0) (hx : n ≤ x ∧ x < n + 1) :
  4 * (⎣x⎦ : ℤ)^2 - 36 * (⎣x⎦ : ℤ) + 45 < 0 ↔ 2 ≤ x ∧ x < 8 := by
  sorry

end solution_set_inequality_l167_167387


namespace area_of_parallelogram_l167_167502

theorem area_of_parallelogram (EF_height_perpendicular : ℝ) (base_EF : ℝ) (angle_theta : ℝ) : 
  base_EF = 6 → EF_height_perpendicular = 5 → angle_theta = 30 → 
  (base_EF * EF_height_perpendicular * Real.sin (Real.pi / 6) = 15) :=
by
  intros hEF h_height h_theta
  rw [hEF, h_height, h_theta]
  simp
  rw [Real.sin_pi_div_six]
  simp
  norm_num

end area_of_parallelogram_l167_167502


namespace prove_a_zero_l167_167615

-- Define two natural numbers a and b
variables (a b : ℕ)

-- Condition: For every natural number n, 2^n * a + b is a perfect square
def condition := ∀ n : ℕ, ∃ k : ℕ, 2^n * a + b = k^2

-- Statement to prove: a = 0
theorem prove_a_zero (h : condition a b) : a = 0 := sorry

end prove_a_zero_l167_167615


namespace sin_sum_triangle_l167_167400

theorem sin_sum_triangle (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_l167_167400


namespace sum_of_eight_numbers_l167_167849

theorem sum_of_eight_numbers (avg : ℝ) (num_of_items : ℕ) (h_avg : avg = 5.3) (h_items : num_of_items = 8) :
  avg * num_of_items = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l167_167849


namespace construct_triangle_l167_167746

theorem construct_triangle (h_a m_a : ℝ) (A : Real.Angle) :
  ∃ (A B C : Point) (M : Point), 
    (∃ (AH : LineSegment), length AH = h_a ∧ is_altitude_to_side A B C) ∧
    (∃ (AM : LineSegment), length AM = m_a ∧ is_median_to_side A B C) ∧
    ∠BAC = A := by
  sorry

end construct_triangle_l167_167746


namespace line_intersects_yz_plane_l167_167376

theorem line_intersects_yz_plane :
  ∃ t : ℝ, 
  let p := (⟨1 + 3 * t, 2 + 3 * t, 3 + 3 * t⟩ : ℝ × ℝ × ℝ) in
  (p.1 = 0) ∧ (p = (0, 1, 2)) :=
by
  sorry

end line_intersects_yz_plane_l167_167376


namespace evaporated_water_l167_167173

theorem evaporated_water 
  (E : ℝ)
  (h₁ : 0 < 10) -- initial mass is positive
  (h₂ : 10 * 0.3 + 10 * 0.7 = 3 + 7) -- Solution Y composition check
  (h₃ : (3 + 0.3 * E) / (10 - E + 0.7 * E) = 0.36) -- New solution composition
  : E = 0.9091 := 
sorry

end evaporated_water_l167_167173


namespace even_digit_numbers_count_eq_100_l167_167457

-- Definition for the count of distinct three-digit positive integers with only even digits
def count_even_digit_three_numbers : ℕ :=
  let hundreds_place := {2, 4, 6, 8}.card
  let tens_units_place := {0, 2, 4, 6, 8}.card
  hundreds_place * tens_units_place * tens_units_place

-- Theorem stating the count of distinct three-digit positive integers with only even digits is 100
theorem even_digit_numbers_count_eq_100 : count_even_digit_three_numbers = 100 :=
by sorry

end even_digit_numbers_count_eq_100_l167_167457


namespace find_B_l167_167632

theorem find_B (A B : ℕ) (h₁ : 6 * A + 10 * B + 2 = 77) (h₂ : A ≤ 9) (h₃ : B ≤ 9) : B = 1 := sorry

end find_B_l167_167632


namespace num_divisible_terms_eq_gcd_l167_167924

theorem num_divisible_terms_eq_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (function.comp List.length (List.filter (λ n, b ∣ n) [a, 2 * a .. b * a])) = Nat.gcd a b := 
by
  sorry

end num_divisible_terms_eq_gcd_l167_167924


namespace north_pond_ducks_l167_167585

-- Definitions based on the conditions
def ducks_lake_michigan : ℕ := 100
def twice_ducks_lake_michigan : ℕ := 2 * ducks_lake_michigan
def additional_ducks : ℕ := 6
def ducks_north_pond : ℕ := twice_ducks_lake_michigan + additional_ducks

-- Theorem to prove the answer
theorem north_pond_ducks : ducks_north_pond = 206 :=
by
  sorry

end north_pond_ducks_l167_167585


namespace Natalia_total_distance_l167_167163

variable (d_m d_t d_w d_r d_total : ℕ)

-- Conditions
axiom cond1 : d_m = 40
axiom cond2 : d_t = 50
axiom cond3 : d_w = d_t / 2
axiom cond4 : d_r = d_m + d_w

-- Question and answer
theorem Natalia_total_distance : 
  d_total = d_m + d_t + d_w + d_r → 
  d_total = 180 := 
by
  intros h
  simp [cond1, cond2, cond3, cond4] at h
  rw [cond1, cond2, cond3, cond4] in h
  simp at h
  exact h

end Natalia_total_distance_l167_167163


namespace domain_of_v_l167_167226

-- Define the function v
noncomputable def v (x y : ℝ) : ℝ := 1 / (x^(2/3) - y^(2/3))

-- State the domain of v
def domain_v : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≠ p.2 }

-- State the main theorem
theorem domain_of_v :
  ∀ x y : ℝ, x ≠ y ↔ (x, y) ∈ domain_v :=
by
  intro x y
  -- We don't need to provide proof
  sorry

end domain_of_v_l167_167226


namespace difference_between_percentages_l167_167259

noncomputable def number : ℝ := 140

noncomputable def percentage_65 (x : ℝ) : ℝ := 0.65 * x

noncomputable def fraction_4_5 (x : ℝ) : ℝ := 0.8 * x

theorem difference_between_percentages 
  (x : ℝ) 
  (hx : x = number) 
  : (fraction_4_5 x) - (percentage_65 x) = 21 := 
by 
  sorry

end difference_between_percentages_l167_167259


namespace cloth_sold_l167_167291

theorem cloth_sold (C S M : ℚ) (P : ℚ) (hP : P = 1 / 3) (hG : 10 * S = (1 / 3) * (M * C)) (hS : S = (4 / 3) * C) : M = 40 := by
  sorry

end cloth_sold_l167_167291


namespace num_digits_for_2C4_multiple_of_4_l167_167002

theorem num_digits_for_2C4_multiple_of_4 : (finset.univ.filter (λ C : ℕ, (C * 10 + 4) % 4 = 0)).card = 5 :=
by
  -- The proof is omitted as we are only required to write the statement.
  sorry

end num_digits_for_2C4_multiple_of_4_l167_167002


namespace regular_heptagon_collinear_points_l167_167546

open_locale classical

noncomputable theory
open set

def is_regular_heptagon (A : fin 7 → ℝ × ℝ) : Prop :=
  ∃ r (o : ℝ × ℝ), r > 0 ∧ ∀ i j, i ≠ j → dist (A i) (A j) = 2 * r * sin (π / 7 * abs (i - j))

def intersection_point (A B C D : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- Intersection of line segments AB and CD (definition omitted for brevity)

def collinear (A B C D : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, C = (1 - k) • A + k • B ∧ D = (1 - k) • A + k • B

theorem regular_heptagon_collinear_points
  (A : fin 7 → ℝ × ℝ)
  (h_reg_hept : is_regular_heptagon A) :
  let B1 := intersection_point (A 1) (A 3) (A 2) (A 7),
      B4 := intersection_point (A 3) (A 5) (A 4) (A 6),
      C1 := intersection_point (A 1) (A 4) (A 3) (A 7),
      C3 := intersection_point (A 3) (A 6) (A 2) (A 5) in
  collinear B1 B4 C1 C3 :=
begin
  sorry -- The proof block is omitted
end

end regular_heptagon_collinear_points_l167_167546


namespace problem_A_problem_B_problem_C_problem_D_correct_options_l167_167806

variables (a b c d : ℝ)

theorem problem_A (hab : a * b > 0) (hbc_ad : b * c - a * d > 0) : c / a - d / b > 0 :=
sorry

theorem problem_B (hab : a * b > 0) (hbc_ad : b * c - a * d > 0) : c / a - d / b < 0 :=
sorry

theorem problem_C (hab : a * b > 0) (hca_db : c / a - d / b > 0) : b * c - a * d > 0 :=
sorry

theorem problem_D (hbc_ad : b * c - a * d > 0) (hca_db : c / a - d / b > 0) : a * b > 0 :=
sorry

-- Equivalencies stating which options are correct
theorem correct_options : problem_A = true ∧ problem_B = false ∧ problem_C = true ∧ problem_D = true :=
sorry

end problem_A_problem_B_problem_C_problem_D_correct_options_l167_167806


namespace product_evaluation_l167_167369

theorem product_evaluation :
  (∏ n in Finset.range (15 - 2 + 1) + 2, (1 - 1 / (n ^ 2))) = (8 / 15) :=
by
  sorry

end product_evaluation_l167_167369


namespace set_A_even_and_large_l167_167899

def is_tens_digit_one (m : ℕ) : Prop :=
  (m / 10) % 10 = 1

def is_member_of_set_A (m : ℕ) : Prop :=
  (16 = String.length (ToString m)) ∧ (∃ n : ℕ, m = n^2) ∧ is_tens_digit_one m

theorem set_A_even_and_large :
  (∀ m ∈ {m : ℕ | is_member_of_set_A m}, Even m) ∧ ({m : ℕ | is_member_of_set_A m}.card > 10^6) :=
  by
    sorry

end set_A_even_and_large_l167_167899


namespace sine_angle_AC_BD_eq_one_l167_167792

variables {A B C D : Type} [MetricSpace A]

def AB := dist (A : A) (B : B) = 2
def AD := dist (A : A) (D : D) = 11/2
def BC := dist (B : B) (C : C) = 8
def CD := dist (C : C) (D : D) = 19/2

theorem sine_angle_AC_BD_eq_one : 
  AB ∧ AD ∧ BC ∧ CD → real.sin (angle (line (A : A) (C : C)) (line (B : B) (D : D))) = 1 :=
by
  sorry

end sine_angle_AC_BD_eq_one_l167_167792


namespace candy_necklaces_l167_167883

theorem candy_necklaces (blocks : ℕ) (candies_per_block : ℕ) (friends : ℕ)
    (blocks_eq : blocks = 3) (candies_per_block_eq : candies_per_block = 30) (friends_eq : friends = 8) : 
  (blocks * candies_per_block) / friends = 11 :=
by
  rw [blocks_eq, candies_per_block_eq, friends_eq]
  norm_num
  sorry

end candy_necklaces_l167_167883


namespace amount_for_second_shop_l167_167580

-- Definitions based on conditions
def books_from_first_shop : Nat := 65
def amount_first_shop : Float := 1160.0
def books_from_second_shop : Nat := 50
def avg_price_per_book : Float := 18.08695652173913
def total_books : Nat := books_from_first_shop + books_from_second_shop
def total_amount_spent : Float := avg_price_per_book * (total_books.toFloat)

-- The Lean statement to prove
theorem amount_for_second_shop : total_amount_spent - amount_first_shop = 920.0 := by
  sorry

end amount_for_second_shop_l167_167580


namespace sum_of_sines_leq_3_sqrt3_over_2_l167_167398

theorem sum_of_sines_leq_3_sqrt3_over_2 (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Real.sin α + Real.sin β + Real.sin γ ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end sum_of_sines_leq_3_sqrt3_over_2_l167_167398


namespace intersection_eq_l167_167436

def A : Set ℕ := {1, 2, 4, 6, 8}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_eq : A ∩ B = {2, 4, 8} := by
  sorry

end intersection_eq_l167_167436


namespace sqrt_eight_is_two_sqrt_two_l167_167182

theorem sqrt_eight_is_two_sqrt_two : sqrt 8 = 2 * sqrt 2 :=
by
  sorry

end sqrt_eight_is_two_sqrt_two_l167_167182


namespace increasing_involution_eq_identity_l167_167547

theorem increasing_involution_eq_identity {f : ℝ → ℝ} (h_increasing : ∀ x y : ℝ, x < y → f(x) < f(y))
  (h_involution : ∀ x : ℝ, f(f(x)) = x) : ∀ x : ℝ, f(x) = x :=
sorry

end increasing_involution_eq_identity_l167_167547


namespace dividend_calculation_l167_167859

theorem dividend_calculation (quotient divisor remainder : ℕ) (h1 : quotient = 36) (h2 : divisor = 85) (h3 : remainder = 26) : 
    (divisor * quotient + remainder = 3086) := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end dividend_calculation_l167_167859


namespace quadratic_radical_identification_l167_167236

theorem quadratic_radical_identification :
  (∃ (a : ℝ), a = sqrt 3) ∧
  (∀ (b : ℝ), b ≠ sqrt[3] 4) ∧
  (∀ (c : ℝ), c ≠ sqrt (- (4 ^ 2))) ∧
  (∀ (d : ℝ), d ≠ sqrt (-5)) :=
by {
  split, 
  { use sqrt 3, },
  split, 
  { intros b hb, 
    exact sorry, },
  split,
  { intros c hc,
    exact sorry, },
  { intros d hd,
    exact sorry, }
  
}

end quadratic_radical_identification_l167_167236


namespace sum_floor_log2_l167_167341

theorem sum_floor_log2 :
  (∑ N in Finset.range 2048 \ + 1, \ ())
  (Nat.floor (logBase 2 N)) = 14324 :=
  sorry

end sum_floor_log2_l167_167341


namespace log_base_3_of_reciprocal_81_l167_167358

theorem log_base_3_of_reciprocal_81 : log 3 (1 / 81) = -4 :=
by
  sorry

end log_base_3_of_reciprocal_81_l167_167358


namespace sum_of_numbers_l167_167960

theorem sum_of_numbers {a b c : ℝ} (h1 : b = 7) (h2 : (a + b + c) / 3 = a + 8) (h3 : (a + b + c) / 3 = c - 20) : a + b + c = 57 :=
sorry

end sum_of_numbers_l167_167960


namespace max_gcd_is_one_l167_167202

-- Defining the sequence a_n
def a_n (n : ℕ) : ℕ := 101 + n^3

-- Defining the gcd function for a_n and a_(n+1)
def d_n (n : ℕ) : ℕ := Nat.gcd (a_n n) (a_n (n + 1))

-- The theorem stating the maximum value of d_n is 1
theorem max_gcd_is_one : ∀ n : ℕ, d_n n = 1 := by
  -- Proof is omitted as per instructions
  sorry

end max_gcd_is_one_l167_167202


namespace tangent_condition_l167_167383

def curve1 (x y : ℝ) : Prop := y = x ^ 3 + 2
def curve2 (x y m : ℝ) : Prop := y^2 - m * x = 1

theorem tangent_condition (m : ℝ) (h : ∃ x y : ℝ, curve1 x y ∧ curve2 x y m) :
  m = 4 + 2 * Real.sqrt 3 :=
sorry

end tangent_condition_l167_167383


namespace graph_passes_through_fixed_point_l167_167957

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ y : ℝ, y = a^0 + 3 ∧ (0, y) = (0, 4)) :=
by
  use 4
  have h : a^0 = 1 := by simp
  rw [h]
  simp
  sorry

end graph_passes_through_fixed_point_l167_167957


namespace arthur_walk_distance_l167_167721

def blocks_east : ℕ := 8
def blocks_north : ℕ := 15
def block_length : ℚ := 1 / 4

theorem arthur_walk_distance :
  (blocks_east + blocks_north) * block_length = 23 * (1 / 4) := by
  sorry

end arthur_walk_distance_l167_167721


namespace sum_on_simple_interest_is_1400_l167_167626

noncomputable def sum_placed_on_simple_interest : ℝ :=
  let P_c := 4000
  let r := 0.10
  let n := 1
  let t_c := 2
  let t_s := 3
  let A := P_c * (1 + r / n)^(n * t_c)
  let CI := A - P_c
  let SI := CI / 2
  100 * SI / (r * t_s)

theorem sum_on_simple_interest_is_1400 : sum_placed_on_simple_interest = 1400 := by
  sorry

end sum_on_simple_interest_is_1400_l167_167626


namespace distinct_two_digit_squares_count_l167_167838

theorem distinct_two_digit_squares_count :
  ∃ S : Finset (Fin 100), S.card = 44 ∧ ∀ d : Fin 100, (d * d) % (100 : ℕ) ∈ S :=
by
  -- assume S is the set of all distinct last two digits of squares
  let S := {z | ∃ (d : ℕ), 0 ≤ d ∧ d < 100 ∧ z = (d * d) % 100}.to_finset
  have hS : S = {z | ∃ (d : ℕ), 0 ≤ d ∧ d < 100 ∧ z = (d * d) % 100}.to_finset, from rfl
  existsi S,
  split,
  -- To prove: S.card = 44
  sorry,
  -- To prove: ∀ d : Fin 100, (d * d) % (100 : ℕ) ∈ S
  intros d,
  simp,
  existsi (d : ℕ),
  split,
  exact nat.zero_le d,
  split,
  exact nat.lt_of_lt_of_le (Fin.is_lt d) (by norm_num), -- 100 > d
  refl,

end distinct_two_digit_squares_count_l167_167838


namespace dad_strawberries_final_weight_l167_167914

variable {M D : ℕ}

theorem dad_strawberries_final_weight :
  M + D = 22 →
  36 - M + 30 + D = D' →
  D' = 46 :=
by
  intros h h1
  sorry

end dad_strawberries_final_weight_l167_167914


namespace range_of_a_not_monotonic_l167_167045

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^3 + (1 - a) * x^2 - a * (a + 2) * x 

-- Define the derivative f'(x)
def f_prime (x a : ℝ) : ℝ := 3 * x^2 + (2 - 2 * a) * x - a * (a + 2)

-- Statement for the given theorem
theorem range_of_a_not_monotonic :
  ∀ a : ℝ, 
  (∃ x : ℝ, -2 < x ∧ x < 2 ∧ f_prime x a = 0) ↔ (a ∈ Set.Ioo (-8 : ℝ) (-1 / 2) ∪ Set.Ioo (-1 / 2) 4) :=
begin
  sorry,
end

end range_of_a_not_monotonic_l167_167045


namespace locus_midpoints_l167_167137

noncomputable def circle_radius_locus (r : ℝ) : ℝ := r * Real.sqrt 3 / 6

theorem locus_midpoints (r : ℝ) (P O : EuclideanSpace ℝ (Fin 2))
  (K : Metric.sphere O r)
  (d : ℝ) (h1 : d = r / 3)
  (h2 : dist P O = d) :
  ∃ L : Metric.sphere P (circle_radius_locus r), 
    ∀ (A B : EuclideanSpace ℝ (Fin 2)), 
      (A ≠ B) → 
      (dist O A = r) ∧ 
      (dist O B = r) ∧ 
      (dist A B = r) ∧ 
      Segment A B P ∧ 
      ∃ M : Midpoint A B, Midpoint.dist O M = circle_radius_locus r := 
sorry

end locus_midpoints_l167_167137


namespace candy_crush_ratio_l167_167054

theorem candy_crush_ratio :
  ∃ m : ℕ, (400 + (400 - 70) + (400 - 70) * m = 1390) ∧ (m = 2) :=
by
  sorry

end candy_crush_ratio_l167_167054


namespace price_of_mixture_l167_167599

theorem price_of_mixture (P1 P2 P3 : ℝ) (h1 : P1 = 126) (h2 : P2 = 135) (h3 : P3 = 175.5) : 
  (P1 + P2 + 2 * P3) / 4 = 153 :=
by 
  -- Main goal is to show (126 + 135 + 2 * 175.5) / 4 = 153
  sorry

end price_of_mixture_l167_167599


namespace triangle_angle_contradiction_l167_167674

theorem triangle_angle_contradiction (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A > 60) (h₃ : B > 60) (h₄ : C > 60) :
  false :=
by
  sorry

end triangle_angle_contradiction_l167_167674


namespace gcd_pow_diff_l167_167227

theorem gcd_pow_diff (m n: ℤ) (H1: m = 2^2025 - 1) (H2: n = 2^2016 - 1) : Int.gcd m n = 511 := by
  sorry

end gcd_pow_diff_l167_167227


namespace fifth_term_is_31_l167_167735

/-- 
  The sequence is formed by summing consecutive powers of 2. 
  Define the sequence and prove the fifth term is 31.
--/
def sequence_sum (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, 2^k)

theorem fifth_term_is_31 : sequence_sum 5 = 31 :=
by sorry

end fifth_term_is_31_l167_167735


namespace find_point_O_l167_167375

-- Definitions and Conditions
structure Triangle (α : Type*) :=
(A B C : α)

noncomputable def center_of_mass {α : Type*} [add_comm_group α] [vector_space ℝ α]
  (p q: ℝ) (ABC: Triangle α) (O: α) : Prop :=
∃ x_a x_c : ℝ, x_a + x_c = 1 ∧ ∀ (K L : α),
  (K = (p • ABC.A + x_a • ABC.B) / (p + x_a) ∧
   L = (x_c • ABC.B + q • ABC.C) / (x_c + q)) →
  (O = (p + x_a) • K / (p + x_a + q + x_c) + (q + x_c) • L / (p + x_a + q + x_c)) ∧
  p * (((ABC.A - K) • K) / (K - ABC.B) • (ABC.B - K)) + 
  q * (((ABC.C - L) • L) / (L - ABC.B) • (ABC.B - L)) = 1

-- The Lean statement
theorem find_point_O {α : Type*} [add_comm_group α] [vector_space ℝ α]
  (ABC : Triangle α) (p q : ℝ) (h_p : p > 0) (h_q : q > 0) :
  ∃ O : α, center_of_mass p q ABC O :=
begin
  sorry
end

end find_point_O_l167_167375


namespace symmetry_condition_f1_symmetric_f2_not_symmetric_composite_fn_symmetric_add_composite_fn_not_symmetric_mul_l167_167980

variable {D : Type*} [Nonempty : Nonempty D]
variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem symmetry_condition (hf : ∀ x ∈ D, f(a + x) + f(a - x) = 2 * b) :
  ∀ x ∈ D, f(x) = f(x - 2 * a) + 2 * (b - f(a)) :=
sorry

def f1 (x : ℝ) : ℝ := 2^x - 2^(-x)
def f2 (x : ℝ) : ℝ := x^2

theorem f1_symmetric : ∀ x, f1 x = f1 (-x) :=
sorry

theorem f2_not_symmetric : ¬∃ (a b : ℝ), ∀ x, f2(a + x) + f2(a - x) = 2 * b :=
sorry

variable {f g : ℝ → ℝ}
variable {a b : ℝ}

theorem composite_fn_symmetric_add 
  (hf : ∀ x, f(a + x) + f(a - x) = 2 * b)
  (hg : ∀ x, g(a + x) + g(a - x) = 2 * b) :
  ∀ x, (f(x) + g(x)) = f(x - 2 * a) + g(x - 2 * a) + 4 * (b - f(a)) :=
sorry

theorem composite_fn_not_symmetric_mul :
  ∃ a b, ¬∀ x, (f(x) * g(x)) = f(x - 2 * a) * g(x - 2 * a) :=
sorry

end symmetry_condition_f1_symmetric_f2_not_symmetric_composite_fn_symmetric_add_composite_fn_not_symmetric_mul_l167_167980


namespace parabola_intersections_l167_167642

-- Definitions of the parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 1
def parabola2 (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

-- The statement to be proven
theorem parabola_intersections :
    (parabola1 (2 + Real.sqrt 6) = 13 + 3 * Real.sqrt 6) ∧ 
    (parabola1 (2 - Real.sqrt 6) = 13 - 3 * Real.sqrt 6) ∧
    (parabola2 (2 + Real.sqrt ) = 13 + 3 * Real.sqrt 6) ∧ 
    (parabola2 (2 - Real.sqrt 6) = 13 - 3 * Real.sqrt 6) := 
by
  sorry

end parabola_intersections_l167_167642


namespace fallen_sheets_l167_167273

/-- The number of sheets that fell out of a book given the first page is 163
    and the last page contains the same digits but arranged in a different 
    order and ends with an even digit.
-/
theorem fallen_sheets (h1 : ∃ n, n = 163 ∧ 
                        ∃ m, m ≠ n ∧ (m = 316) ∧ 
                        m % 2 = 0 ∧ 
                        (∃ p1 p2 p3 q1 q2 q3, 
                         (p1, p2, p3) ≠ (q1, q2, q3) ∧ 
                         p1 ≠ q1 ∧ p2 ≠ q2 ∧ p3 ≠ q3 ∧ 
                         n = p1 * 100 + p2 * 10 + p3 ∧ 
                         m = q1 * 100 + q2 * 10 + q3)) :
  ∃ k, k = 77 :=
by
  sorry

end fallen_sheets_l167_167273


namespace differentiable_increasing_implication_l167_167191

theorem differentiable_increasing_implication (f : ℝ → ℝ) (a b : ℝ) (h_diff : ∀ x ∈ Ioo a b, differentiable_at ℝ f x) (h_inc : ∀ x y ∈ Ioo a b, x < y → f x < f y) :
  ∀ x ∈ Ioo a b, deriv f x ≥ 0 :=
sorry

end differentiable_increasing_implication_l167_167191


namespace spherical_to_rectangular_conversion_l167_167335

noncomputable def convert_spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  convert_spherical_to_rectangular 8 (5 * Real.pi / 4) (Real.pi / 4) = (-4, -4, 4 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l167_167335


namespace percentage_increase_in_gross_sales_l167_167702

theorem percentage_increase_in_gross_sales
  (P N : ℝ)
  (hP : P > 0) 
  (hN : N > 0)
  (first_discount : ℝ := 0.20)
  (first_increase : ℝ := 0.80)
  (second_discount : ℝ := 0.15)
  (second_increase : ℝ := 0.50) :
  let new_price1 := P * (1 - first_discount)
      new_sales1 := N * (1 + first_increase)
      gross_sales1 := new_price1 * new_sales1
      new_price2 := new_price1 * (1 - second_discount)
      new_sales2 := new_sales1 * (1 + second_increase)
      gross_sales2 := new_price2 * new_sales2
      original_gross_sales := P * N
      percentage_increase := ((gross_sales2 - original_gross_sales) / original_gross_sales) * 100
  in percentage_increase = 83.6 :=
by {
  sorry
}

end percentage_increase_in_gross_sales_l167_167702


namespace hexagon_chord_splits_l167_167498

noncomputable def find_coprime_sum (m n : ℕ) : ℕ :=
m + n

theorem hexagon_chord_splits :
  ∃ (m n : ℕ), coprime m n ∧
  find_coprime_sum m n = 409 ∧
  (∃ (a b : ℝ), 
    -- conditions representing the problem setup
    ∀ A B : set (ℝ × ℝ), 
    ((A ∩ B = ∅) ∧ 
    (A ∪ B = set.univ) ∧ 
    (∃ hexagon : hexagon, 
      (hexagon.has_sides [3, 3, 3, 5, 5, 5])
    ) ∧
    (∃ chord : chord, 
      (chord.splits_hexagon_into_quadrilaterals_with_sides [3, 3, 3] [5, 5, 5])
    )))) := 
sorry

end hexagon_chord_splits_l167_167498


namespace find_a_value_l167_167820

theorem find_a_value :
  let slope_y_curve : ℝ → ℝ := λ x, (x + 1) / (x - 1)
  let slope_tangent (x : ℝ) : ℝ := -2 / ((x - 1)^2)
  let slope_at_p : ℝ := slope_tangent 2
  let given_slope : ℝ := -a
  let slope_equation : Prop := slope_at_p * given_slope = -1
  ∃ a : ℝ, slope_equation ∧ a = -1 / 2 :=
sorry

end find_a_value_l167_167820


namespace arithmetic_progression_power_of_two_l167_167304

theorem arithmetic_progression_power_of_two 
  (a d : ℤ) (n : ℕ) (k : ℕ) 
  (Sn : ℤ)
  (h_sum : Sn = 2^k)
  (h_ap : Sn = n * (2 * a + (n - 1) * d) / 2)  :
  ∃ m : ℕ, n = 2^m := 
sorry

end arithmetic_progression_power_of_two_l167_167304


namespace evaluate_expression_l167_167143

theorem evaluate_expression (x z : ℝ) (h1 : x ≠ 0) (h2 : z ≠ 0) (y : ℝ) (h3 : y = 1 / x + z) : 
    (x - 1 / x) * (y + 1 / y) = (x^2 - 1) * (1 + 2 * x * z + x^2 * z^2 + x^2) / (x^2 * (1 + x * z)) := by
  sorry

end evaluate_expression_l167_167143


namespace roots_product_sum_eq_l167_167904

open Polynomial

theorem roots_product_sum_eq {p q r : ℝ} :
  (root_of_polynomial (6 * X^3 - 9 * X^2 + 14 * X - 10) = [p, q, r]) →
  (p * q + p * r + q * r = 7 / 3) :=
begin
  sorry
end

end roots_product_sum_eq_l167_167904


namespace sqrt_expression_value_l167_167231

theorem sqrt_expression_value : sqrt (25 * sqrt (15 * sqrt 9)) = 5 * real.root 4 3 :=
by sorry

end sqrt_expression_value_l167_167231


namespace sets_relationship_l167_167911

def M : Set ℝ := {x | (x + 3) / (x - 1) ≤ 0}
def N : Set ℝ := {x | abs (x + 1) ≤ 2}
def P : Set ℝ := {x | (1/2) ^ (x^2 + 2 * x - 3) ≥ 1}

theorem sets_relationship : M ⊆ N ∧ N = P :=
  sorry

end sets_relationship_l167_167911


namespace combined_tax_rate_correct_l167_167683

noncomputable def combined_tax_rate (income_john income_ingrid tax_rate_john tax_rate_ingrid : ℝ) : ℝ :=
  let tax_john := tax_rate_john * income_john
  let tax_ingrid := tax_rate_ingrid * income_ingrid
  let total_tax := tax_john + tax_ingrid
  let combined_income := income_john + income_ingrid
  total_tax / combined_income * 100

theorem combined_tax_rate_correct :
  combined_tax_rate 56000 74000 0.30 0.40 = 35.69 := by
  sorry

end combined_tax_rate_correct_l167_167683


namespace power_function_even_iff_l167_167488

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ), f x = f (-x)

theorem power_function_even_iff (m : ℝ) :
(is_even_function (λ x, (m^2 - m - 1) * x^(1 - m))) ↔ (m = -1) :=
by
  sorry

end power_function_even_iff_l167_167488


namespace infinite_series_value_l167_167733

theorem infinite_series_value :
  ∑' n : ℕ, (n^3 + 2*n^2 + 5*n + 2) / (3^n * (n^3 + 3)) = 1 / 2 :=
sorry

end infinite_series_value_l167_167733


namespace log3_1_over_81_l167_167363

theorem log3_1_over_81 : log 3 (1 / 81) = -4 := by
  have h1 : 1 / 81 = 3 ^ (-4) := by
    -- provide a proof or skip with "sory"
    sorry
  have h2 : log 3 (3 ^ (-4)) = -4 := by
    -- provide a proof or skip with "sorry"
    sorry
  exact eq.trans (log 3) (congr_fun (h1.symm h2))

end log3_1_over_81_l167_167363


namespace inflation_over_two_years_real_yield_deposit_second_year_l167_167661

-- Inflation problem setup and proof
theorem inflation_over_two_years :
  ((1 + 0.015) ^ 2 - 1) * 100 = 3.0225 :=
by sorry

-- Real yield problem setup and proof
theorem real_yield_deposit_second_year :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by sorry

end inflation_over_two_years_real_yield_deposit_second_year_l167_167661


namespace acute_triangle_inequality_l167_167866

theorem acute_triangle_inequality
  (ABC : Triangle)
  (acute : ABC.isAcute)
  (R : ℝ)
  (h1 h2 h3 t1 t2 t3 : ℝ)
  (AD BE CF : LineSegment)
  (altitudes : ABC.altitudes = (AD, BE, CF))
  (altitude_lengths : AD.length = h1 ∧ BE.length = h2 ∧ CF.length = h3)
  (tangents : TangentLengths ABC.Circumcircle (A, B, C) = (t1, t2, t3)) :
  (t1 / Real.sqrt h1)^2 + (t2 / Real.sqrt h2)^2 + (t3 / Real.sqrt h3)^2 ≤ (3 / 2) * R :=
by
  sorry

end acute_triangle_inequality_l167_167866


namespace jimmy_fill_pool_time_l167_167124

theorem jimmy_fill_pool_time (pool_gallons : ℕ) (bucket_gallons : ℕ) (time_per_trip_sec : ℕ) (sec_per_min : ℕ) :
  pool_gallons = 84 → 
  bucket_gallons = 2 → 
  time_per_trip_sec = 20 → 
  sec_per_min = 60 → 
  (pool_gallons / bucket_gallons) * time_per_trip_sec / sec_per_min = 14 :=
by
  sorry

end jimmy_fill_pool_time_l167_167124


namespace height_ratio_width_ratio_l167_167194

variable (HeightSculpture : ℝ) -- actual height of the sculpture in feet
variable (HeightModel : ℝ) -- height of the model in inches
variable (WidthSculpture : ℝ) -- actual width of the sculpture in feet
variable (WidthModel : ℝ) -- width of the model in inches

-- Given values
def height_sculpture_value : HeightSculpture = 120
def height_model_value : HeightModel = 8
def width_sculpture_value : WidthSculpture = 40
def width_model_value : WidthModel = 2

-- Proving the height ratio
theorem height_ratio (hS : HeightSculpture = 120) (hM : HeightModel = 8) : (HeightSculpture / HeightModel) = 15 := by
  sorry

-- Proving the width ratio
theorem width_ratio (wS : WidthSculpture = 40) (wM : WidthModel = 2) : (WidthSculpture / WidthModel) = 20 := by
  sorry

end height_ratio_width_ratio_l167_167194


namespace area_of_50th_ring_l167_167333

-- Definitions based on conditions:
def garden_area : ℕ := 9
def ring_area (n : ℕ) : ℕ := 9 * ((2 * n + 1) ^ 2 - (2 * (n - 1) + 1) ^ 2) / 2

-- Theorem to prove:
theorem area_of_50th_ring : ring_area 50 = 1800 := by sorry

end area_of_50th_ring_l167_167333


namespace maximize_ratio_l167_167134

open Matrix

noncomputable def a : ℝ := 1 + 10 ^ (-4)

def x_i (A : Matrix (Fin 2023) (Fin 2023) ℝ) (i : Fin 2023) : ℝ :=
  ∑ j, A i j

def y_i (A : Matrix (Fin 2023) (Fin 2023) ℝ) (i : Fin 2023) : ℝ :=
  ∑ j, A j i

theorem maximize_ratio (A : Matrix (Fin 2023) (Fin 2023) ℝ)
  (h : ∀ i j, 1 ≤ A i j ∧ A i j ≤ a) :
  ∃ (A : Matrix (Fin 2023) (Fin 2023) ℝ), 
    ∀ i j, 1 ≤ A i j ∧ A i j ≤ a ∧
    (∑ j, A i j) = x_i A i ∧
    (∑ j, A j i) = y_i A i ∧
    ∑ i, x_i A i = ∑ i, y_i A i ∧ 
    (x_i A) = y_i A → 
    ((∏ i, y_i A i) / (∏ i, x_i A i) = 1) := sorry

end maximize_ratio_l167_167134


namespace probability_red_or_blue_l167_167987

-- Definitions and assumptions
def Total : ℕ := 84
def P_W : ℚ := 1 / 4
def P_G : ℚ := 2 / 7

-- Theorem to prove
theorem probability_red_or_blue : (1 - (P_W + P_G) = (13:ℚ) / 28) := by
  sorry

end probability_red_or_blue_l167_167987


namespace num_digits_multiple_of_4_l167_167004

theorem num_digits_multiple_of_4 : (Finset.card (Finset.filter (λ C : Fin 10, (10 * C + 4) % 4 = 0) Finset.univ) = 5) := 
by {
  sorry -- This is where the proof would go.
}

end num_digits_multiple_of_4_l167_167004


namespace trig_identity_l167_167243

theorem trig_identity (θ : ℝ) (hθ : θ = π / 12) :
  cos θ ^ 2 - sin θ ^ 2 = sqrt 3 / 2 :=
by 
  sorry

end trig_identity_l167_167243


namespace fifth_boy_pays_l167_167774

def problem_conditions (a b c d e : ℝ) : Prop :=
  d = 20 ∧
  a = (1 / 3) * (b + c + d + e) ∧
  b = (1 / 4) * (a + c + d + e) ∧
  c = (1 / 5) * (a + b + d + e) ∧
  a + b + c + d + e = 120 

theorem fifth_boy_pays (a b c d e : ℝ) (h : problem_conditions a b c d e) : 
  e = 35 :=
sorry

end fifth_boy_pays_l167_167774


namespace krakozyabrs_count_l167_167108

theorem krakozyabrs_count :
  ∀ (K : Type) [fintype K] (has_horns : K → Prop) (has_wings : K → Prop), 
  (∀ k, has_horns k ∨ has_wings k) →
  (∃ n, (∀ k, has_horns k → has_wings k) → fintype.card ({k | has_horns k} : set K) = 5 * n) →
  (∃ n, (∀ k, has_wings k → has_horns k) → fintype.card ({k | has_wings k} : set K) = 4 * n) →
  (∃ T, 25 < T ∧ T < 35 ∧ T = 32) := 
by
  intros K _ has_horns has_wings _ _ _
  sorry

end krakozyabrs_count_l167_167108


namespace cyclist_C_speed_l167_167998

noncomputable def speed_of_cyclist_C : ℝ :=
  let v_C : ℝ := 10 in
  let v_D := v_C + 5 in
  let distance_C := 72 in
  let distance_D := 108 in
  if (72 / v_C = 108 / v_D) then v_C else 0

theorem cyclist_C_speed : speed_of_cyclist_C = 10 :=
begin
  unfold speed_of_cyclist_C,
  split_ifs,
  refl,
  sorry -- This will be replaced with the actual proof in a full verification case.
end

end cyclist_C_speed_l167_167998


namespace circumscribed_triangle_BCK_tangent_to_AB_l167_167698

-- Define objects in Lean
variables {A B C D K : Point}

-- Definitions of geometric conditions
def is_cyclic_trapezoid (ABCD : Trapezoid) : Prop :=
  ∃ (circle : Circle), circle.passes_through A ∧ circle.passes_through B ∧ circle.passes_through D

def intersecting_point (circ : Circle) (CD : Line) : Prop :=
  circ.intersects CD ∧ K ∈ circ ∧ K ∈ CD

def parallel_sides (AD BC : Line) : Prop :=
  AD.parallel BC

-- Main theorem statement
theorem circumscribed_triangle_BCK_tangent_to_AB
  (h1 : is_cyclic_trapezoid ABCD)
  (h2 : intersecting_point circle CD)
  (h3 : parallel_sides AD BC) :
  circle_circumscribed_around_triangle B C K.tangent_to AB :=
sorry

end circumscribed_triangle_BCK_tangent_to_AB_l167_167698


namespace cos_double_angle_l167_167843

theorem cos_double_angle (α : ℝ) (h : Real.sin (π + α) = 2 / 3) : Real.cos (2 * α) = 1 / 9 := 
by
  sorry

end cos_double_angle_l167_167843


namespace correct_statements_l167_167719

def correlation_relationship (x y : Type) :=
  ∀ a b : x, ∃ c : y, c has certain randomness when a is fixed

def scatter_plot (x y : Type) :=
  ∃ points : set (x × y), points represents a set of data in a Cartesian coordinate system with a correlation relationship

def linear_regression_best_fit (x y : Type) :=
  linear_regression_line_equation represents the linear correlation relationship between x and y

-- The following condition states that statement 4 is incorrect:
def regression_line_representative_meaning (x y : Type) :=
  ∃ observed_values : set (x × y), regression_line_equation has representative meaning

theorem correct_statements {x y : Type} :
  correlation_relationship x y ∧ scatter_plot x y ∧ linear_regression_best_fit x y ∧ ¬ regression_line_representative_meaning x y → true :=
by
  intros,
  sorry

end correct_statements_l167_167719


namespace part1_part2_interval_part2_axis_of_symmetry_l167_167834

-- Define the vectors
def vector_a (θ : ℝ) : ℝ × ℝ := (Real.sin θ, 1)
def vector_b (θ : ℝ) : ℝ × ℝ := (1, Real.cos θ)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Part 1: Proving the perpendicular condition implies θ = -π/4
theorem part1 (θ : ℝ) :
  dot_product (vector_a θ) (vector_b θ) = 0 ∧ -π / 2 < θ ∧ θ < π / 2 → θ = -π / 4 :=
sorry

-- Part 2: Proving the interval of monotonic increase and the axis of symmetry
theorem part2_interval (k : ℤ) (θ : ℝ) :
  2 * k * π - 3 * π / 4 ≤ θ ∧ θ ≤ 2 * k * π + π / 4 →
  is_monotonic_increasing (λ θ, Real.sqrt (2 * (Real.sin θ + Real.cos θ) + 3)) θ :=
sorry

theorem part2_axis_of_symmetry (k : ℤ) :
  ∃ x, x = k * π + π / 4 :=
sorry

end part1_part2_interval_part2_axis_of_symmetry_l167_167834


namespace log_base_3_of_one_over_81_l167_167345

theorem log_base_3_of_one_over_81 : log 3 (1 / 81) = -4 := by
  sorry

end log_base_3_of_one_over_81_l167_167345


namespace final_remaining_money_l167_167442

-- Define conditions as given in the problem
def monthly_income : ℕ := 2500
def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50
def expenses_total : ℕ := rent + car_payment + utilities + groceries
def remaining_money : ℕ := monthly_income - expenses_total
def retirement_contribution : ℕ := remaining_money / 2

-- State the theorem to be proven
theorem final_remaining_money : (remaining_money - retirement_contribution) = 650 := by
  sorry

end final_remaining_money_l167_167442


namespace sum_even_coeffs_l167_167842

theorem sum_even_coeffs (a : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ)
  (h_eq : ((a : ℝ) * x + 1)^5 * (x + 2)^4 = 
           a_0 * (x + 2) ^ 9 + a_1 * (x + 2) ^ 8 + 
           a_2 * (x + 2) ^ 7 + a_3 * (x + 2) ^ 6 + 
           a_4 * (x + 2) ^ 5 + a_5 * (x + 2) ^ 4 + 
           a_6 * (x + 2) ^ 3 + a_7 * (x + 2) ^ 2 + 
           a_8 * (x + 2) + a_9)
  (h_sum : (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + 
           a_6 + a_7 + a_8 + a_9 = 1024)) :
  a_0 + a_2 + a_4 + a_6 + a_8 = (2^10 - 14^5) / 2 :=
sorry

end sum_even_coeffs_l167_167842


namespace count_quadratic_functions_l167_167827

def valid_coefficients (a b c : ℕ) : Prop := 
  (a ≠ 0) ∧ (a = 1 ∨ a = 2) ∧ (b ∈ {0, 1, 2}) ∧ (c ∈ {0, 1, 2})

theorem count_quadratic_functions : 
  (finset.univ.filter (λ x : ℕ × ℕ × ℕ, valid_coefficients x.1 x.2.1 x.2.2)).card = 18 := 
by 
  sorry

end count_quadratic_functions_l167_167827


namespace smallest_n_l167_167905

theorem smallest_n (n : ℕ) (x : fin n → ℝ)
  (h₀ : ∀ i, 0 ≤ x i)
  (h₁ : ∑ i, x i = 1)
  (h₂ : (∑ i, (x i)^2) ≤ 1/50) :
  n ≥ 50 :=
sorry

end smallest_n_l167_167905


namespace range_of_a_range_of_t_l167_167956

noncomputable def f (x a : ℝ) := log (9^x + 3^x - a)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 9^x + 3^x - a > 0) ↔ a ≤ 0 :=
sorry

noncomputable def g (x : ℝ) : ℝ :=
if x > 0 then 3^x
else if x < 0 then -3^(-x)
else 0

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, g (x^2 - 2*t*x + 3) / g x = |g x| → ∃ x1 x2 : ℝ, x1 ≠ x2) ↔
  t ∈ (-∞, -1 - sqrt 3) ∪ (sqrt 3 - 1, +∞) :=
sorry

end range_of_a_range_of_t_l167_167956


namespace completion_time_l167_167269

-- Define the problem conditions
def A_time : ℝ := 4 -- A can do a piece of work in 4 hours
def B_time : ℝ := 3 -- B can do a piece of work in 3 hours
def C_time : ℝ := 2 -- C can do another piece of work in 2 hours
def A_B_together_C_time : ℝ := 1.5 -- A and B together can do C's work in 1.5 hours

-- Define the time it takes for everyone to complete their tasks
def total_time := max A_time (max B_time C_time)

-- Define the theorem to be proved
theorem completion_time : total_time = 4 :=
by
  sorry

end completion_time_l167_167269


namespace linda_paint_cans_l167_167154

theorem linda_paint_cans (wall_area : ℝ) (coverage_per_gallon : ℝ) (coats : ℝ) 
  (h1 : wall_area = 600) 
  (h2 : coverage_per_gallon = 400) 
  (h3 : coats = 2) : 
  (ceil (wall_area * coats / coverage_per_gallon) = 3) := 
by 
  sorry

end linda_paint_cans_l167_167154


namespace paint_needed_l167_167157

theorem paint_needed (wall_area : ℕ) (coverage_per_gallon : ℕ) (number_of_coats : ℕ) (h_wall_area : wall_area = 600) (h_coverage_per_gallon : coverage_per_gallon = 400) (h_number_of_coats : number_of_coats = 2) : 
    ((number_of_coats * wall_area) / coverage_per_gallon) = 3 :=
by
  sorry

end paint_needed_l167_167157


namespace price_for_two_bracelets_l167_167241

theorem price_for_two_bracelets
    (total_bracelets : ℕ)
    (price_per_bracelet : ℕ)
    (total_earned_for_single : ℕ)
    (total_earned : ℕ)
    (bracelets_sold_single : ℕ)
    (bracelets_left : ℕ)
    (remaining_earned : ℕ)
    (pairs_sold : ℕ)
    (price_per_pair : ℕ) :
    total_bracelets = 30 →
    price_per_bracelet = 5 →
    total_earned_for_single = 60 →
    total_earned = 132 →
    bracelets_sold_single = total_earned_for_single / price_per_bracelet →
    bracelets_left = total_bracelets - bracelets_sold_single →
    remaining_earned = total_earned - total_earned_for_single →
    pairs_sold = bracelets_left / 2 →
    price_per_pair = remaining_earned / pairs_sold →
    price_per_pair = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end price_for_two_bracelets_l167_167241


namespace period_of_tan2x_plus_cot2x_l167_167656

noncomputable theory

open Real

theorem period_of_tan2x_plus_cot2x :
  ∃ T > 0, ∀ x, tan (2 * (x + T)) + cot (2 * (x + T)) = tan (2 * x) + cot (2 * x) :=
sorry

end period_of_tan2x_plus_cot2x_l167_167656


namespace log_base_3_of_one_over_81_l167_167344

theorem log_base_3_of_one_over_81 : log 3 (1 / 81) = -4 := by
  sorry

end log_base_3_of_one_over_81_l167_167344


namespace factorable_and_nonneg_discriminant_l167_167377

theorem factorable_and_nonneg_discriminant (x y k : ℤ) :
  (∃ (A B C D E F : ℤ), (x^2 + 4 * x * y + 2 * x + k * y - 3 * k = (A * x + B * y + C) * (D * x + E * y + F)) ∧
  (let Δ := (4 : ℤ)^2 - 4 * (1 : ℤ) * (0 : ℤ) in Δ ≥ 0)) ↔ k = 0 := 
sorry

end factorable_and_nonneg_discriminant_l167_167377


namespace problem_statement_l167_167028

noncomputable def solve_equations : ℝ :=
  let x1 := some (classical.indefinite_description _ (exists_unique λ x, x + Real.log10 x = 3))
  let x2 := some (classical.indefinite_description _ (exists_unique λ x, x + 10^x = 3))
  x1 + x2

theorem problem_statement :
  let x1 := some (classical.indefinite_description _ (exists_unique λ x, x + Real.log10 x = 3))
  let x2 := some (classical.indefinite_description _ (exists_unique λ x, x + 10^x = 3))
  x1 + x2 = 3 := 
sorry

end problem_statement_l167_167028


namespace number_of_true_propositions_is_one_l167_167423

def proposition_1_not_complementary : Prop :=
  let A := {ω | ω = (true, true)}
  let B := {ω | ω = (false, false)}
  ¬ (A ∪ B = {ω | true})

def proposition_2_mutually_exclusive : Prop :=
  let A := {ω | ω = (true, true)}
  let B := {ω | ω = (false, false)}
  A ∩ B = ∅

def proposition_3_not_mutually_exclusive : Prop :=
  let Ω := ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10} : Set ℕ)
  let combinations := {S : Set ℕ | S ⊆ Ω ∧ S.card = 3}
  let A := {S ∈ combinations | S.filter (λx => x <= 3).card ≤ 2}
  let B := {S ∈ combinations | S.filter (λx => x <= 3).card ≥ 2}
  ∃ S ∈ combinations, S ∈ A ∧ S ∈ B

theorem number_of_true_propositions_is_one :
  (∃ (p1_false : proposition_1_not_complementary) (p2_true : proposition_2_mutually_exclusive)
   (p3_false : proposition_3_not_mutually_exclusive), true) →
  1 = 1 := sorry

end number_of_true_propositions_is_one_l167_167423


namespace solution_l167_167023

noncomputable def problem_statement : Prop :=
  ∃ (C1 C2 : ℝ → ℝ → Prop)
    (F1 F2 : ℝ × ℝ)
    (a b c : ℝ),
  (∀ x y, C1 x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∀ x y, C2 x y ↔ y^2 = 8 * x) ∧
  F1 = (-2, 0) ∧
  F2 = (2, 0) ∧
  a > b ∧ b > 0 ∧
  2 * sqrt 2 = max (λ P, 0.5 * dist P F1 * dist P F2) (C1) ∧
  ∃ k,
  (∀ m (T : ℝ × ℝ), T.fst = -3 → 
    let F1 := (-2, 0) 
        calcDistance : ℝ := sqrt (dist T F1)
        intersectPoints : ℝ × ℝ := -- Placeholder for the actual logic to find M and N
        |mn| : ℝ := sqrt (dist intersectPoints.fst intersectPoints.snd),
    calcDistance / |mn| ≥ sqrt 3 / 3)

theorem solution : problem_statement :=
sorry

end solution_l167_167023


namespace area_quadrilateral_OMPN_l167_167180

-- Given definitions and conditions
variables (A B C D O P M N : Type)
variables [EuclideanGeometry A B C D P M N]

-- Specific geometry conditions
variables (h_area_trap : TrapezoidArea A B C D = 90)
variables (h_AD_is_2BC : Length AD = 2 * Length BC)
variables (h_diagonals_intersect : Intersect AC BD O)
variables (h_P_midpoint : Midpoint P AD)
variables (h_M_B_intersect : Intersect BP AC M)
variables (h_N_C_intersect : Intersect CP BD N)

-- Goal to prove
theorem area_quadrilateral_OMPN : AreaQuadrilateral O M P N = 10 :=
by sorry

end area_quadrilateral_OMPN_l167_167180


namespace numberOfValidColoringsIs_l167_167055

def validColoringsOnBranch (colors : Finset (List (Fin Color))) : Prop :=
  ∀ c ∈ colors, c.count (λ x, x = Color.yellow) = 2 ∧ c.count (λ x, x = Color.blue) = 2

def coloringDistinctUnderSymmetry (colorings : Finset (List (Fin Color))) : Prop :=
  ∀ c1 c2 ∈ colorings, (c1 ≠ c2 ∨ c1.transformedBy (rotationOrReflection) ≠ c2)

theorem numberOfValidColoringsIs (b : Finset (List (Fin Color))) (h : validColoringsOnBranch b) (hsym : coloringDistinctUnderSymmetry b) : 
  b.card = 696 := 
sorry

end numberOfValidColoringsIs_l167_167055


namespace compound_interest_doubling_l167_167483

theorem compound_interest_doubling (P : ℝ) (t : ℝ) (r n : ℝ) :
  t = 10 → n = 1 → P ≠ 0 → 2 * P = P * (1 + r / n)^(n * t) → 
  r ≈ 0.071773462 → 
  n = 1 → (∃ r, A = P * (1 + r)^10) → true := 
by
  sorry

end compound_interest_doubling_l167_167483


namespace symmetric_axis_and_vertex_l167_167629

theorem symmetric_axis_and_vertex (x : ℝ) : 
  (∀ x y, y = (1 / 2) * (x - 1)^2 + 6 → x = 1) 
  ∧ (1, 6) = (1, 6) :=
by 
  sorry

end symmetric_axis_and_vertex_l167_167629


namespace geometric_seq_monotonic_decreasing_maximized_prod_l167_167500

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := 
∀ n, a n = a 1 * (q ^ (n - 1)) 

theorem geometric_seq_monotonic_decreasing_maximized_prod 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, 0 < a n)
  (h_a1 : a 1 = 20)
  (h_cond : 2 * a 4 + a 3 - a 2 = 0) : 
  (∀ n, a n > a (n + 1)) ∧ (T n = ∏ i in finset.range n, a (i + 1) → T 5 = max) :=
begin
  sorry
end

end geometric_seq_monotonic_decreasing_maximized_prod_l167_167500


namespace difference_in_sales_l167_167265

def daily_pastries : ℕ := 20
def daily_bread : ℕ := 10
def today_pastries : ℕ := 14
def today_bread : ℕ := 25
def price_pastry : ℕ := 2
def price_bread : ℕ := 4

theorem difference_in_sales : (daily_pastries * price_pastry + daily_bread * price_bread) - (today_pastries * price_pastry + today_bread * price_bread) = -48 :=
by
  -- Proof will go here
  sorry

end difference_in_sales_l167_167265


namespace thirteen_pow_seven_mod_nine_l167_167593

theorem thirteen_pow_seven_mod_nine : (13^7 % 9 = 4) :=
by {
  sorry
}

end thirteen_pow_seven_mod_nine_l167_167593


namespace probability_in_interval_l167_167636

noncomputable def probs_correct_uniform (X : ℝ → ℝ) [measure_theory.distrib.uniform X] (M : ∀ i, ∫ x in i, X x ∂ℙ = 4) (σ : ∀ i, ∫ x in i, (X x - 4)^2 ∂ℙ = 16) : Prop :=
  ∫ x in (5, 12), X x ∂ℙ = 0.428

noncomputable def probs_correct_exponential (X : ℝ → ℝ) [measure_theory.distrib.exponential X] (M : ∀ i, ∫ x in i, X x ∂ℙ = 4) (σ : ∀ i, ∫ x in i, (X x - 4)^2 ∂ℙ = 16) : Prop :=
  ∫ x in (5, 12), X x ∂ℙ = 0.2367

noncomputable def probs_correct_normal (X : ℝ → ℝ) [measure_theory.distrib.normal X] (M : ∀ i, ∫ x in i, X x ∂ℙ = 4) (σ : ∀ i, ∫ x in i, (X x - 4)^2 ∂ℙ = 16) : Prop :=
  ∫ x in (5, 12), X x ∂ℙ = 0.3785

-- Theorem to encapsulate all conditions and results for uniform, exponential and normal distributions
theorem probability_in_interval (X₁ X₂ X₃ : ℝ → ℝ) 
  [measure_theory.distrib.uniform X₁] 
  [measure_theory.distrib.exponential X₂]
  [measure_theory.distrib.normal X₃]
  (M₁ : ∀ i, ∫ x in i, X₁ x ∂ℙ = 4)
  (σ₁ : ∀ i, ∫ x in i, (X₁ x - 4)^2 ∂ℙ = 16)
  (M₂ : ∀ i, ∫ x in i, X₂ x ∂ℙ = 4)
  (σ₂ : ∀ i, ∫ x in i, (X₂ x - 4)^2 ∂ℙ = 16)
  (M₃ : ∀ i, ∫ x in i, X₃ x ∂ℙ = 4)
  (σ₃ : ∀ i, ∫ x in i, (X₃ x - 4)^2 ∂ℙ = 16) :
  probs_correct_uniform X₁ M₁ σ₁ ∧ probs_correct_exponential X₂ M₂ σ₂ ∧ probs_correct_normal X₃ M₃ σ₃ :=
by sorry

end probability_in_interval_l167_167636


namespace _l167_167407

noncomputable def a_sequence (n : ℕ+) : ℤ :=
  if n = 1 then 3 else 3 * (-2) ^ (n - 1)

noncomputable def S_sequence (n : ℕ+) : ℚ :=
  2 / 3 * a_sequence n + 1

noncomputable theorem part_I (n : ℕ+) : a_sequence n = if n = 1 then 3 else 3 * (-2) ^ (n - 1) := 
by sorry

noncomputable def T_sequence (n : ℕ+) : ℚ :=
  3 + 3 * n * 2^n - 3 * 2^n

noncomputable theorem part_II (n : ℕ+) : 
  T_sequence n = 3 + 3 * n * 2^n - 3 * 2^n :=
by sorry

end _l167_167407


namespace sin_alpha_def_cos_alpha_def_tan_alpha_def_l167_167405

variable {α : Type*}
variable (m : ℝ) (h : m < 0)

def P : ℝ × ℝ := (3 * m, -2 * m)

theorem sin_alpha_def : ∃ (α : ℝ), sin α = (2 * Real.sqrt 13) / 13 := by
  use α
  sorry

theorem cos_alpha_def : ∃ (α : ℝ), cos α = -(3 * Real.sqrt 13) / 13 := by
  use α
  sorry

theorem tan_alpha_def : ∃ (α : ℝ), tan α = -2 / 3 := by
  use α
  sorry

end sin_alpha_def_cos_alpha_def_tan_alpha_def_l167_167405


namespace intersection_with_x_axis_l167_167372

noncomputable def f (x : ℝ) : ℝ := 
  (3 * x - 1) * (Real.sqrt (9 * x^2 - 6 * x + 5) + 1) + 
  (2 * x - 3) * (Real.sqrt (4 * x^2 - 12 * x + 13)) + 1

theorem intersection_with_x_axis :
  ∃ x : ℝ, f x = 0 ∧ x = 4 / 5 :=
by
  sorry

end intersection_with_x_axis_l167_167372


namespace max_value_proof_l167_167943

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + 2 * x * y + 3 * y^2

theorem max_value_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 12) : 
  max_value x y = 24 + 12 * Real.sqrt 3 := 
sorry

end max_value_proof_l167_167943


namespace problem1_problem2_problem3_problem4_l167_167302

-- Definitions based on given conditions
def non_coincident {A : Type*} (x y : A) : Prop := ¬(x = y)
def perpendicular {A : Type*} [inner_product_space ℝ A] (x y : A) : Prop := ⟪x, y⟫ = 0
def intersection {A : Type*} [linear_ordered_add_comm_group A] (x y : set A) : set A := x ∩ y

-- Problem ①
theorem problem1 (m n : ℝ) (α β : set ℝ) :
  non_coincident m n →
  non_coincident α β →
  perpendicular α β →
  intersection α β = {n} →
  perpendicular m n →
  ¬perpendicular m β :=
sorry

-- Problem ②
theorem problem2 (r : ℝ) : 
  (|r| ≈ 1) → 
  stronger_linear_correlation r :=
sorry

-- Problem ③
def f (x : ℝ) : ℝ := 208 + 9 * x^2 + 6 * x^4 + x^6
noncomputable def v (x v0 : ℝ) := 
  let v1 := 1 * x in
  let v2 := v1 * x + v0 in
  v2
theorem problem3 : 
  v (-4) 6 = 22 :=
sorry

-- Problem ④
theorem problem4 : 
  ∃ (l : ℝ -> ℝ), (∀ A B : ℝ × ℝ, (parabola y^2 = 4x) l (sum_x_coord  A B) = 4) → 
  (number_of_lines l = 2) :=
sorry

end problem1_problem2_problem3_problem4_l167_167302


namespace missing_digits_and_thousandth_place_l167_167517

def sequence : ℕ → ℕ
| 0     := 2
| 1     := 3
| n + 2 := let prod := (sequence n) * (sequence (n + 1))
           in if prod < 10 then prod else prod % 10

def repeating_part : ℕ → ℕ := λ n, [4, 8, 3, 2, 6, 1, 2, 2].nth! (n % 8)

-- Define the combined sequence taking into account the initial and repeating part
def combined_sequence : ℕ → ℕ
| n := if n < 9 then sequence n else repeating_part (n - 9)

theorem missing_digits_and_thousandth_place :
  -- The missing digits are 0, 5, 7, 9
  (∀ d: ℕ, d ∈ [0, 5, 7, 9] ∧ ∀ n : ℕ, combined_sequence n ≠ d) ∧
  -- The digit at the 1000th position is 2
  (combined_sequence 999 = 2) :=
by
  sorry

end missing_digits_and_thousandth_place_l167_167517


namespace max_gcd_l167_167312

theorem max_gcd (n : ℕ) (h : 0 < n) : ∀ n, ∃ d ≥ 1, d ∣ 13 * n + 4 ∧ d ∣ 8 * n + 3 → d ≤ 9 :=
begin
  sorry
end

end max_gcd_l167_167312


namespace gaoan_total_revenue_in_scientific_notation_l167_167294

theorem gaoan_total_revenue_in_scientific_notation :
  (21 * 10^9 : ℝ) = 2.1 * 10^9 :=
sorry

end gaoan_total_revenue_in_scientific_notation_l167_167294


namespace repeating_block_length_7_div_13_l167_167326

theorem repeating_block_length_7_div_13 : ∀ n : ℕ, (n = 7 / 13) → (length_of_repeating_block (decimal_expansion n) = 6) :=
begin
  sorry
end

end repeating_block_length_7_div_13_l167_167326


namespace find_c_for_circle_radius_l167_167007

theorem find_c_for_circle_radius :
  ∃ c : ℝ, (∀ (x y : ℝ), (x^2 + 8*x + y^2 - 2*y + c = 0 → (x + 4)^2 + (y - 1)^2 = 25)) :=
begin
  use -8,
  intros x y h,
  calc
    (x + 4)^2 + (y - 1)^2
        = (x^2 + 8*x + 16) + (y^2 - 2*y + 1) : by ring
    ... = x^2 + 8*x + y^2 - 2*y + 17 : by ring
    ... = 25                         : by { rw h, ring }
end

end find_c_for_circle_radius_l167_167007


namespace quarts_of_water_needed_l167_167639

-- Definitions of conditions
def total_parts := 5 + 2 + 1
def total_gallons := 3
def quarts_per_gallon := 4
def water_parts := 5

-- Lean proof statement
theorem quarts_of_water_needed :
  (water_parts : ℚ) * ((total_gallons * quarts_per_gallon) / total_parts) = 15 / 2 :=
by sorry

end quarts_of_water_needed_l167_167639


namespace chomp_game_configurations_l167_167499

/-- Number of valid configurations such that 0 ≤ a_1 ≤ a_2 ≤ ... ≤ a_5 ≤ 7 is 330 -/
theorem chomp_game_configurations :
  let valid_configs := {a : Fin 6 → Fin 8 // (∀ i j, i ≤ j → a i ≤ a j)}
  Fintype.card valid_configs = 330 :=
sorry

end chomp_game_configurations_l167_167499


namespace leopards_points_l167_167078

variables (x y : ℕ)

theorem leopards_points (h₁ : x + y = 50) (h₂ : x - y = 28) : y = 11 := by
  sorry

end leopards_points_l167_167078


namespace remainder_when_divided_l167_167234

noncomputable def y : ℝ := 19.999999999999716
def quotient : ℝ := 76.4
def remainder : ℝ := 8

theorem remainder_when_divided (x : ℝ) (hx : x = y * 76 + y * 0.4) : x % y = 8 :=
by
  -- Proof is omitted
  sorry

end remainder_when_divided_l167_167234


namespace area_of_right_triangle_l167_167195

theorem area_of_right_triangle : 
  ∀ (h : ℝ) (θ : ℝ), h = 15 ∧ θ = 45 → 
  let l := h / Real.sqrt 2 in 
  1 / 2 * l * l = 112.5 :=
by
  intro h θ
  rintro ⟨h_eq, θ_eq⟩
  let l := h / Real.sqrt 2
  sorry

end area_of_right_triangle_l167_167195


namespace collinearity_A2_B2_C2_ratio_A2B2_B2C2_l167_167687

theorem collinearity_A2_B2_C2 :
  (∀ (A B C A_0 B_0 C_0 H A_1 B_1 C_1 A_2 B_2 C_2: Point), 
    acute_triangle A B C ∧ BC > CA > AB ∧ inscribed_circle A B C A_0 B_0 C_0 ∧ orthocenter A B C H 
    ∧ midpoint A H A_1 ∧ midpoint H B B_1 ∧ midpoint H C C_1 
    ∧ reflection A_1 (line B_0 C_0) A_2 
    ∧ reflection B_1 (line C_0 A_0) B_2 
    ∧ reflection C_1 (line A_0 B_0) C_2 → 
  collinear A_2 B_2 C_2) := sorry

theorem ratio_A2B2_B2C2 :
  (∀ (A B C A_0 B_0 C_0 H A_1 B_1 C_1 A_2 B_2 C_2: Point),
    acute_triangle A B C ∧ BC > CA > AB ∧ inscribed_circle A B C A_0 B_0 C_0 ∧ orthocenter A B C H
    ∧ midpoint A H A_1 ∧ midpoint H B B_1 ∧ midpoint H C C_1
    ∧ reflection A_1 (line B_0 C_0) A_2
    ∧ reflection B_1 (line C_0 A_0) B_2
    ∧ reflection C_1 (line A_0 B_0) C_2 →
  (dist A_2 B_2 / dist B_2 C_2) = (tan (A / 2) - tan (B / 2)) / (tan (B / 2) - tan (C / 2))) := sorry

end collinearity_A2_B2_C2_ratio_A2B2_B2C2_l167_167687


namespace probability_arithmetic_sequence_l167_167013

-- Define the set of numbers from which the selection is made
def num_set : Set ℕ := { n | 1 ≤ n ∧ n ≤ 20 }

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a b c : ℕ) : Prop := a + c = 2 * b

-- Define the count of selecting 3 distinct numbers from the set
def count_total_triplets : ℕ := Nat.choose 20 3

-- Define the count of arithmetic sequence triplets
noncomputable def count_arithmetic_sequences : ℕ :=
  (10.choose 2) * 2

-- The probability calculation
theorem probability_arithmetic_sequence : 
  (count_arithmetic_sequences : ℚ) / count_total_triplets = 3 / 38 :=
by
  -- Conversion and calculation to be proved here
  sorry

end probability_arithmetic_sequence_l167_167013


namespace burrito_calories_l167_167126

theorem burrito_calories :
  ∀ (C : ℕ), 
  (10 * C = 6 * (250 - 50)) →
  C = 120 :=
by
  intros C h
  sorry

end burrito_calories_l167_167126


namespace pqrs_product_l167_167529

noncomputable def P := (Real.sqrt 2007 + Real.sqrt 2008)
noncomputable def Q := (-Real.sqrt 2007 - Real.sqrt 2008)
noncomputable def R := (Real.sqrt 2007 - Real.sqrt 2008)
noncomputable def S := (-Real.sqrt 2008 + Real.sqrt 2007)

theorem pqrs_product : P * Q * R * S = -1 := by
  sorry

end pqrs_product_l167_167529


namespace log_base_3_l167_167350

theorem log_base_3 (h : (1 / 81 : ℝ) = 3 ^ (-4 : ℝ)) : Real.logBase 3 (1 / 81) = -4 := 
by sorry

end log_base_3_l167_167350


namespace marble_weights_total_l167_167920

theorem marble_weights_total:
  0.33 + 0.33 + 0.08 + 0.25 + 0.02 + 0.12 + 0.15 = 1.28 :=
by {
  sorry
}

end marble_weights_total_l167_167920


namespace sum_first_21_terms_arithmetic_progression_l167_167380

theorem sum_first_21_terms_arithmetic_progression :
  let a := 3
  let d := 7
  let n := 21
  S_n = n / 2 * (2 * a + (n - 1) * d) := 1533 := 
by
  sorry

end sum_first_21_terms_arithmetic_progression_l167_167380


namespace area_ratio_probability_l167_167410

open set

variables {A B C D E F : Point}
variables {Parallelogram : Points → Prop}
variables {Midpoint : Point → Point → Point → Prop}
variables {ChosenOn : Point → Line → Prop}
variables {ProbAreaRatio : Line → Line → ℝ}

theorem area_ratio_probability (h1 : Parallelogram {A, B, C, D})
                               (h2 : E = Midpoint B C)
                               (h3 : ChosenOn F AB):
  ProbAreaRatio (AD : Line) (BE : Line) = 2 / 3 :=
begin
  sorry,
end

end area_ratio_probability_l167_167410


namespace range_of_a_l167_167426

-- Defining the function f(x)
def f (a x : ℝ) := x^2 + (a^2 - 1) * x + (a - 2)

-- The statement of the problem in Lean 4
theorem range_of_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : -2 < a ∧ a < 1 :=
by
  sorry -- Proof is omitted

end range_of_a_l167_167426


namespace removed_term_is_a11_l167_167090

noncomputable def sequence_a (n : ℕ) (a1 d : ℤ) := a1 + (n - 1) * d

def sequence_sum (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem removed_term_is_a11 :
  ∃ d : ℤ, ∀ a1 d : ℤ, 
            a1 = -5 ∧ 
            sequence_sum 11 a1 d = 55 ∧ 
            (sequence_sum 11 a1 d - sequence_a 11 a1 d) / 10 = 4 
          → sequence_a 11 a1 d = removed_term :=
sorry

end removed_term_is_a11_l167_167090


namespace inflation_two_years_correct_real_rate_of_return_correct_l167_167668

-- Define the calculation for inflation over two years
def inflation_two_years (r : ℝ) : ℝ :=
  ((1 + r)^2 - 1) * 100

-- Define the calculation for the real rate of return
def real_rate_of_return (r : ℝ) (infl_rate : ℝ) : ℝ :=
  ((1 + r * r) / (1 + infl_rate / 100) - 1) * 100

-- Prove the inflation over two years is 3.0225%
theorem inflation_two_years_correct :
  inflation_two_years 0.015 = 3.0225 :=
by
  sorry

-- Prove the real yield of the bank deposit is 11.13%
theorem real_rate_of_return_correct :
  real_rate_of_return 0.07 3.0225 = 11.13 :=
by
  sorry

end inflation_two_years_correct_real_rate_of_return_correct_l167_167668


namespace determine_true_proposition_l167_167910

def proposition_p : Prop :=
  ∃ x : ℝ, Real.tan x > 1

def proposition_q : Prop :=
  let focus_distance := 3/4 -- Distance from the focus to the directrix in y = (1/3)x^2
  focus_distance = 1/6

def true_proposition : Prop :=
  proposition_p ∧ ¬proposition_q

theorem determine_true_proposition :
  (proposition_p ∧ ¬proposition_q) = true_proposition :=
by
  sorry -- Proof will go here

end determine_true_proposition_l167_167910


namespace intersection_ST_l167_167153

def S : Set ℝ := { x : ℝ | x < -5 } ∪ { x : ℝ | x > 5 }
def T : Set ℝ := { x : ℝ | -7 < x ∧ x < 3 }

theorem intersection_ST : S ∩ T = { x : ℝ | -7 < x ∧ x < -5 } := 
by 
  sorry

end intersection_ST_l167_167153


namespace Pete_latest_time_to_LA_l167_167166

def minutesInHour := 60
def minutesOfWalk := 10
def minutesOfTrain := 80
def departureTime := 7 * minutesInHour + 30

def latestArrivalTime : Prop :=
  9 * minutesInHour = departureTime + minutesOfWalk + minutesOfTrain 

theorem Pete_latest_time_to_LA : latestArrivalTime :=
by
  sorry

end Pete_latest_time_to_LA_l167_167166


namespace fraction_value_l167_167747

def at (a b : ℤ) : ℤ := a * b - b^3
def hash (a b : ℤ) : ℤ := a + 2 * b - a * b^3

theorem fraction_value :
  at 7 3 / hash 7 3 = 3 / 88 :=
by sorry

end fraction_value_l167_167747


namespace trigonometric_identity_proof_l167_167551

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h : Real.tan α = (1 + Real.sin β) / Real.cos β)

theorem trigonometric_identity_proof : 2 * α - β = π / 2 := 
by 
  sorry

end trigonometric_identity_proof_l167_167551


namespace max_gcd_13n_plus_4_8n_plus_3_l167_167318

theorem max_gcd_13n_plus_4_8n_plus_3 : 
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 7 ∧ ∀ k : ℕ, k = gcd (13 * n + 4) (8 * n + 3) → k ≤ d :=
by
  sorry

end max_gcd_13n_plus_4_8n_plus_3_l167_167318


namespace find_coordinates_of_P_l167_167812

-- Definitions based on the conditions:
-- Point P has coordinates (a, 2a-1) and lies on the line y = x.

def lies_on_bisector (a : ℝ) : Prop :=
  (2 * a - 1) = a -- This is derived from the line y = x for the given point coordinates.

-- The final statement to prove:
theorem find_coordinates_of_P (a : ℝ) (P : ℝ × ℝ) (h1 : P = (a, 2 * a - 1)) (h2 : lies_on_bisector a) :
  P = (1, 1) :=
by
  -- Proof steps are omitted and replaced with sorry.
  sorry

end find_coordinates_of_P_l167_167812


namespace solution_y_amount_to_achieve_25_percent_l167_167932

noncomputable def alcohol_concentration (x y: ℝ) (vol_x vol_y: ℝ) : ℝ :=
  (x * vol_x + y * vol_y) / (vol_x + vol_y)

theorem solution_y_amount_to_achieve_25_percent:
  ∀ (vol_x vol_y: ℝ),
    vol_x = 200 →
    vol_y = 600 →
    alcohol_concentration 0.10 0.30 vol_x vol_y = 0.25 :=
by
  intros vol_x vol_y Hx Hy
  rw [Hx, Hy]
  unfold alcohol_concentration
  calc
    (0.10 * 200 + 0.30 * 600) / (200 + 600)
      = (20 + 180) / 800 : by simp
      = 200 / 800 : by linarith
      = 0.25 : by norm_num

end solution_y_amount_to_achieve_25_percent_l167_167932


namespace intersection_distance_from_GH_l167_167507

variables {t : ℝ} (h_t : t > 0)

def square_coords (t : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((0, 0), (t, 0), (t, t), (0, t))

def arc1_center := (0, 0)
def arc2_center := (t, 0)

def arc1_eq (x y : ℝ) : Prop := x^2 + y^2 = (t/2)^2
def arc2_eq (x y : ℝ) : Prop := (x - t)^2 + y^2 = (3*t/2)^2

theorem intersection_distance_from_GH :
  ∃ (x y : ℝ), arc1_eq t x y ∧ arc2_eq t x y ∧ y = t :=
sorry

end intersection_distance_from_GH_l167_167507


namespace total_amount_shared_l167_167272

variables (a b c : ℝ)

-- Conditions
def cond1 : Prop := a = (1 / 3) * (b + c)
def cond2 : Prop := b = (2 / 7) * (a + c)
def cond3 : Prop := a = b + 35

-- Goal
theorem total_amount_shared (h1 : cond1 a b c) (h2 : cond2 a b c) (h3 : cond3 a b c) : a + b + c = 1260 := 
sorry

end total_amount_shared_l167_167272


namespace shoe_store_total_shoes_l167_167624

theorem shoe_store_total_shoes (b k : ℕ) (h1 : b = 22) (h2 : k = 2 * b) : b + k = 66 :=
by
  sorry

end shoe_store_total_shoes_l167_167624


namespace ratio_of_areas_l167_167950

theorem ratio_of_areas (AB CD AH BG CF DG S_ABCD S_KLMN : ℕ)
  (h1 : AB = 15)
  (h2 : CD = 19)
  (h3 : DG = 17)
  (condition1 : S_ABCD = 17 * (AH + BG))
  (midpoints_AH_CF : AH = BG)
  (midpoints_CF_CD : CF = CD/2)
  (condition2 : (∃ h₁ h₂ : ℕ, S_KLMN = h₁ * AH + h₂ * CF / 2))
  (h_case1 : (S_KLMN = (AH + BG + CD)))
  (h_case2 : (S_KLMN = (AB + (CD - DG)))) :
  (S_ABCD / S_KLMN = 2 / 3 ∨ S_ABCD / S_KLMN = 2) :=
  sorry

end ratio_of_areas_l167_167950


namespace length_of_PQ_l167_167492

-- Define the type for right-angled triangle properties
structure RightAngledTriangle (P Q R : Type) :=
(angle_P : ℝ)
(tan_R : ℝ)
(QR : ℝ)
(PQ : ℝ)

-- Create an instance given the conditions
def triangle_PQR : RightAngledTriangle :=
{ angle_P := 90,
  tan_R := 8,
  QR := 80,
  PQ := 79.384 }

-- The theorem statement
theorem length_of_PQ (t : RightAngledTriangle)
  (h1 : t.angle_P = 90)
  (h2 : t.tan_R = 8)
  (h3 : t.QR = 80) : t.PQ = 79.384 := sorry

end length_of_PQ_l167_167492


namespace maximum_value_is_l167_167939

noncomputable def maximum_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) : ℝ :=
  x^2 + 2 * x * y + 3 * y^2

theorem maximum_value_is (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) :
  maximum_value x y h₁ h₂ h₃ ≤ 18 + 12 * Real.sqrt 3 :=
sorry

end maximum_value_is_l167_167939


namespace sales_difference_l167_167261

-- Definitions of the conditions
def daily_avg_sales_pastries := 20 * 2
def daily_avg_sales_bread := 10 * 4
def daily_avg_sales := daily_avg_sales_pastries + daily_avg_sales_bread

def today_sales_pastries := 14 * 2
def today_sales_bread := 25 * 4
def today_sales := today_sales_pastries + today_sales_bread

-- Statement to be proved
theorem sales_difference : today_sales - daily_avg_sales = 48 :=
by {
  -- Unpack the definitions
  simp [daily_avg_sales_pastries, daily_avg_sales_bread, daily_avg_sales],
  simp [today_sales_pastries, today_sales_bread, today_sales],
  -- Computation,
  -- daily_avg_sales == 20 * 2 + 10 * 4 == 80,
  -- today_sales == 14 * 2 + 25 * 4 == 128
  -- therefore, 128 - 80 == 48,
  -- QED.
  sorry
}

end sales_difference_l167_167261


namespace standard_equation_of_ellipse_equation_of_line_through_chord_l167_167793

-- Condition 1: Ellipse equation
def ellipse (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

-- Condition 2: Eccentricity
def eccentricity (a c : ℝ) := c / a

-- Given minor axis length and eccentricity
def minor_axis_length := 4
def given_eccentricity := sqrt 3 / 2

-- Problem 1: Prove the standard equation of the ellipse
theorem standard_equation_of_ellipse (a b c : ℝ) (h1: b = 2) (h2: eccentricity a c = given_eccentricity)
  (h3: c^2 = a^2 - b^2): ellipse a b = (λ x y, (x^2 / 16) + (y^2 / 4) = 1) :=
sorry

-- Problem 2: Prove the equation of the line containing the chord passing through P(2, 1)
theorem equation_of_line_through_chord (x1 y1 x2 y2 : ℝ) (a b : ℝ) 
  (h1: x1 + x2 = 4) (h2: y1 + y2 = 2) 
  (h3: ellipse a b x1 y1) (h4: ellipse a b x2 y2) 
  (P : ℝ × ℝ) (h5: P = (2, 1)): (λ x y, x + 2 * y - 4 = 0) :=
sorry

end standard_equation_of_ellipse_equation_of_line_through_chord_l167_167793


namespace particle_intersection_distance_l167_167505

theorem particle_intersection_distance :
  let p1 := (1 : ℝ, 2, 2)
  let p2 := (-2 : ℝ, -4, -5)
  let line_traj (t : ℝ) := (1 - 3 * t, 2 - 6 * t, 2 - 7 * t)
  let unit_sphere (x y z : ℝ) := x^2 + y^2 + z^2 = 1
  let t1 := (-49 + sqrt 705) / 47
  let t2 := (-49 - sqrt 705) / 47
  let point1 := line_traj t1
  let point2 := line_traj t2
  let dist_point1_point2 := Real.sqrt (3^2 * (t1 - t2)^2 + 6^2 * (t1 - t2)^2 + 7^2 * (t1 - t2)^2)
  dist_point1_point2 = 24 * Real.sqrt 3 / Real.sqrt 47 :=
by
  sorry

end particle_intersection_distance_l167_167505


namespace change_first_digit_to_8_gives_largest_l167_167235

theorem change_first_digit_to_8_gives_largest (x : ℕ) (h : x = 0.1234567):
    maximized_value (λ y, y ∈ {0.8234567, 0.1834567, 0.1284567, 0.1238567, 0.1234867, 0.1234587, 0.1234568}) 0.8234567 := 
by 
  sorry

end change_first_digit_to_8_gives_largest_l167_167235


namespace circle_radius_l167_167052

-- Define the main geometric scenario in Lean 4
theorem circle_radius 
  (O P A B : Type) 
  (r OP PA PB : ℝ)
  (h1 : PA * PB = 24)
  (h2 : OP = 5)
  : r = 7 
:= sorry

end circle_radius_l167_167052


namespace m_plus_n_l167_167898

noncomputable def S : set (ℤ × ℤ × ℤ) := 
  { p | let (x, y, z) := p in 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 0 ≤ z ∧ z ≤ 5 }

def midpoint (p1 p2 : ℤ × ℤ × ℤ) : ℤ × ℤ × ℤ := 
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)

def valid_midpoint (p1 p2 : ℤ × ℤ × ℤ) : Prop := 
  midpoint p1 p2 ∈ S

noncomputable def total_points : ℕ := 
  (4) * (5) * (6)

noncomputable def total_valid_pairs : ℕ := 
  10 * 13 * 18

noncomputable def total_pairs : ℕ := 
  (total_points * (total_points - 1)) / 2

noncomputable def probability_midpoint_in_S : ℚ := 
  (total_valid_pairs - total_points) / total_pairs

noncomputable def m := 37
noncomputable def n := 119

theorem m_plus_n : m + n = 156 := 
  by sorry

end m_plus_n_l167_167898


namespace solve_for_x_l167_167174

theorem solve_for_x (n m x : ℕ) (h1 : 5 / 7 = n / 91) (h2 : 5 / 7 = (m + n) / 105) (h3 : 5 / 7 = (x - m) / 140) :
    x = 110 :=
sorry

end solve_for_x_l167_167174


namespace inequality_solution_set_minimum_value_l167_167828

noncomputable def f (x a : ℝ) : ℝ := |2 * x - 1| - |2 * x - a|

theorem inequality_solution_set (x : ℝ) (h : f x 2 ≥ (1/2) * x) : x ∈ set.Icc (-∞) (-2) ∪ set.Icc (6/7) 2 := 
sorry

theorem minimum_value (a b c : ℝ) (h₁ : a > 1) (h₂ : f x a ≤ a - 1) (h₃ : 1/b + 2/c = a - (a - 1)) : 
  2 / (b - 1) + 1 / (c - 2) = 2 :=
sorry

end inequality_solution_set_minimum_value_l167_167828


namespace krakozyabrs_total_count_l167_167104

theorem krakozyabrs_total_count :
  (∃ (T : ℕ), 
  (∀ (H W n : ℕ),
    0.2 * H = n ∧ 0.25 * W = n ∧ H = 5 * n ∧ W = 4 * n ∧ T = H + W - n ∧ 25 < T ∧ T < 35) ∧ 
  T = 32) :=
by sorry

end krakozyabrs_total_count_l167_167104


namespace f_periodic_f_def_interval_f_def_interval_find_f_l167_167544

noncomputable def f : ℝ → ℝ
| x => if h : -1 ≤ x ∧ x < 0 then -4 * x ^ 2 + 2
       else if h : 0 ≤ x ∧ x < 1 then x
       else f (x - 2 * ⌊(x + 1) / 2⌋)

theorem f_periodic (x : ℝ) : f (x + 2) = f x :=
by sorry

theorem f_def_interval (x : ℝ) (h1 : -1 ≤ x) (h2 : x < 0) : f x = -4 * x ^ 2 + 2 :=
by sorry

theorem f_def_interval' (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) : f x = x :=
by sorry

theorem find_f (x : ℝ) : f (3/2) = 1 :=
by
  have hx : f (3/2) = f (-1/2) := by
    rw ← f_periodic (3/2 - 2 * ⌊(3/2 + 1) / 2⌋) -- Using periodicity
  have h : -1 ≤ -1/2 ∧ -1/2 < 0 := by
    exact ⟨by linarith, by linarith⟩
  rw f_def_interval (-1/2) h.1 h.2
  norm_num
  rfl
  sorry

end f_periodic_f_def_interval_f_def_interval_find_f_l167_167544


namespace magnitude_a_plus_b_lambda_values_l167_167440

open ComplexConjugate RealAngle

variables {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)
variables (λ : ℝ)

-- Definitions and conditions
def norm_a : ℝ := ‖a‖ = 1
def norm_b : ℝ := ‖b‖ = 2
def angle_ab : Real.Angle := real.cos (Real.angle a b) = 1 / 2

-- First proof problem
theorem magnitude_a_plus_b :
  norm_a a → norm_b b → angle_ab a b → ‖a + b‖ = Real.sqrt 7 := by
  sorry

-- Second proof problem
theorem lambda_values :
  norm_a a → norm_b b → angle_ab a b → 
  (inner (λ • a - 6 • b) (λ • a + b) = 0 ↔ λ = 8 ∨ λ = -3) := by
  sorry

end magnitude_a_plus_b_lambda_values_l167_167440


namespace find_number_picked_by_person_5_l167_167342

-- Define the cyclic group of friends and their chosen numbers and averages
def friends := Fin 8 → ℕ
noncomputable def averages := [2, 4, 5, 7, 9, 11, 12, 1] : List ℕ

-- Function to calculate the average based on neighbors
def neighbor_average (b : friends) (i : Fin 8) : ℕ :=
  (b (i - 1) + b (i + 1)) / 2

-- Main theorem statement
theorem find_number_picked_by_person_5 (b : friends)
  (h : ∀ i : Fin 8, neighbor_average b i = averages[i]) :
  b 4 = 13 := sorry

end find_number_picked_by_person_5_l167_167342


namespace card_53_is_king_l167_167575

def card_sequence : List String := ["K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2", "A"]

def nth_card (n : Nat) : String :=
  card_sequence.get! ((n - 1) % card_sequence.length)

theorem card_53_is_king :
  nth_card 53 = "K" :=
by
  -- Definitions and calculations
  have mod_result : 53 % 13 = 1 := by norm_num
  rw [nth_card, mod_result]
  norm_num
  exact List.get_cons_self_nth card_sequence 1 sorry

end card_53_is_king_l167_167575


namespace volume_vector_equation_l167_167906

noncomputable def volume (P Q R S : ℝ × ℝ × ℝ) : ℝ :=
sorry

noncomputable def vector (A B : ℝ × ℝ × ℝ) : (ℝ × ℝ × ℝ) :=
(A.1 - B.1, A.2 - B.2, A.3 - B.3)

noncomputable def dot_product (u v : (ℝ × ℝ × ℝ)) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem volume_vector_equation 
(A B C D M : ℝ × ℝ × ℝ) 
(hM : ∃ α β γ δ : ℝ, α + β + γ + δ = 1 ∧ M = (α • A + β • B + γ • C + δ • D)) : 
    dot_product (vector M A) (volume M B C D) + 
    dot_product (vector M B) (volume M A C D) + 
    dot_product (vector M C) (volume M A B D) + 
    dot_product (vector M D) (volume M A B C) = 0 :=
sorry

end volume_vector_equation_l167_167906


namespace max_value_proof_l167_167944

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + 2 * x * y + 3 * y^2

theorem max_value_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 12) : 
  max_value x y = 24 + 12 * Real.sqrt 3 := 
sorry

end max_value_proof_l167_167944


namespace fraction_divisible_by_n_l167_167518

theorem fraction_divisible_by_n (a b n : ℕ) (h1 : a ≠ b) (h2 : n > 0) (h3 : n ∣ (a^n - b^n)) : n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end fraction_divisible_by_n_l167_167518


namespace total_books_read_l167_167678

-- Given conditions
variables (c s : ℕ) -- variable c represents the number of classes, s represents the number of students per class

-- Main statement to prove
theorem total_books_read (h1 : ∀ a, a = 7) (h2 : ∀ b, b = 12) :
  84 * c * s = 84 * c * s :=
by
  sorry

end total_books_read_l167_167678


namespace elans_initial_speed_l167_167638

theorem elans_initial_speed
    (dist : ℝ)
    (tim_speed : ℝ)
    (tim_meet_dist : ℝ)
    (elan_speed : ℝ) :
    dist = 120 →
    tim_speed = 10 →
    tim_meet_dist = 80 →
    elan_speed = dist / 4 →
    elan_speed = 40 / 3 :=
by
  intro h1 h2 h3 h4
  rw [h4]
  field_simp [dist]
  sorry

end elans_initial_speed_l167_167638


namespace min_value_of_function_l167_167198

noncomputable def f (x : ℝ) : ℝ := 3 * x + 12 / x^2

theorem min_value_of_function :
  ∀ x > 0, f x ≥ 9 :=
by
  intro x hx_pos
  sorry

end min_value_of_function_l167_167198


namespace sum_of_acute_angles_l167_167817

theorem sum_of_acute_angles (α β : ℝ) (t : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_tanα : Real.tan α = 2 / t) (h_tanβ : Real.tan β = t / 15)
  (h_min : 10 * Real.tan α + 3 * Real.tan β = 4) :
  α + β = π / 4 :=
sorry

end sum_of_acute_angles_l167_167817


namespace major_airlines_internet_percentage_eq_both_l167_167591

noncomputable def percentage_of_major_airlines_with_internet : ℝ := 35
noncomputable def percentage_of_major_airlines_with_snacks : ℝ := 70
noncomputable def percentage_of_major_airlines_with_both : ℝ := 35

theorem major_airlines_internet_percentage_eq_both (W S IcapS : ℝ):
  S = percentage_of_major_airlines_with_snacks →
  IcapS = percentage_of_major_airlines_with_both →
  W = IcapS :=
begin
  intros hS hIcapS,
  sorry
end

end major_airlines_internet_percentage_eq_both_l167_167591


namespace λ_is_integer_gcd_λ_n_λ_n_plus_1_l167_167133
open Nat

/-- Problem statement expressing the conditions and the goal to prove -/

variables {m : ℤ} (hm : m % 2 = 1)   -- Assuming m is an odd integer
variables {α β : ℝ} (hαβ : ∀ x: ℝ, x^2 + (m:ℝ) * x - 1 = 0 -> (x = α ∨ x = β))

def λ (n : ℕ) : ℝ :=
  if n = 0 then α^0 + β^0
  else if n = 1 then α^1 + β^1
  else α^n + β^n

theorem λ_is_integer (n : ℕ) : ∃ k : ℤ, λ n = k :=
  sorry  -- Proof to show λ_n is an integer

theorem gcd_λ_n_λ_n_plus_1 (n : ℕ) : gcd (int.natAbs (λ n)) (int.natAbs (λ (n + 1))) = 1 :=
  sorry  -- Proof to show gcd(λ_n, λ_{n+1}) = 1

end λ_is_integer_gcd_λ_n_λ_n_plus_1_l167_167133


namespace correct_statements_count_l167_167946

/-- The Fibonacci sequence -/
def Fibonacci (n : ℕ) : ℕ :=
  Nat.fib (n + 1)

theorem correct_statements_count :
  (∃ m : ℕ, m > 0 ∧ (Fibonacci m + Fibonacci (m+2) = 2 * Fibonacci (m+1))) ∧
  ¬ (∃ m : ℕ, m > 0 ∧ (Fibonacci (m+1) * Fibonacci (m+1) = Fibonacci m * Fibonacci (m+2))) ∧
  (∃ t : ℚ, t = 3/2 ∧ ∀ n : ℕ, n > 0 → (Fibonacci n + Fibonacci (n+4) = 2 * t * Fibonacci (n+2))) ∧
  (∃ (i₁ i₂ : ℕ), 1≤i₁ ∧ i₁<i₂ ∧ (Fibonacci i₁ + Fibonacci i₂ = 2023)) →
  ∃ n : ℕ, n = 3 :=
sorry

end correct_statements_count_l167_167946


namespace range_of_a_l167_167060

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x < -1 ↔ x ≤ a) ↔ a < -1 :=
by
  sorry

end range_of_a_l167_167060


namespace leq_sum_l167_167396

open BigOperators

theorem leq_sum (x : Fin 3 → ℝ) (hx_pos : ∀ i, 0 < x i) (hx_sum : ∑ i, x i = 1) :
  (∑ i, 1 / (1 + (x i)^2)) ≤ 27 / 10 :=
sorry

end leq_sum_l167_167396


namespace fastest_slowest_difference_l167_167991

-- Given conditions
def length_A : ℕ := 8
def length_B : ℕ := 10
def length_C : ℕ := 6
def section_length : ℕ := 2

def sections_A : ℕ := 24
def sections_B : ℕ := 25
def sections_C : ℕ := 27

-- Calculate number of cuts required
def cuts_per_segment_A := length_A / section_length - 1
def cuts_per_segment_B := length_B / section_length - 1
def cuts_per_segment_C := length_C / section_length - 1

-- Calculate total number of cuts
def total_cuts_A := cuts_per_segment_A * (sections_A / (length_A / section_length))
def total_cuts_B := cuts_per_segment_B * (sections_B / (length_B / section_length))
def total_cuts_C := cuts_per_segment_C * (sections_C / (length_C / section_length))

-- Finding min and max cuts
def max_cuts := max total_cuts_A (max total_cuts_B total_cuts_C)
def min_cuts := min total_cuts_A (min total_cuts_B total_cuts_C)

-- Prove that the difference between max cuts and min cuts is 2
theorem fastest_slowest_difference :
  max_cuts - min_cuts = 2 := by
  sorry

end fastest_slowest_difference_l167_167991


namespace unique_parallel_line_exists_infinitely_many_parallel_planes_exist_l167_167988

open Set

-- Define the line, point, plane, and the concept of parallel lines and planes.
section Geometry
variable {Point Line Plane : Type}
variable (L : Line) (P : Point) (Pl : Plane)
variable (isParallelLine : Line → Line → Prop)
variable (isParallelPlaneLine : Plane → Line → Prop)
variable (isOutsideLine : Point → Line → Prop)

-- Given conditions
variable (p_outside : isOutsideLine P L)

-- Definitions
def unique_parallel_line (L : Line) (P : Point) [h : isOutsideLine P L] : Prop :=
  ∃! (L' : Line), isParallelLine L L' ∧ isOutsideLine P L'

def infinitely_many_parallel_planes (L : Line) (P : Point) [h : isOutsideLine P L] : Prop :=
  ∃ (S : Set Plane), infinite S ∧ ∀ Pl ∈ S, isParallelPlaneLine Pl L ∧ isOutsideLine P L

-- Theorem statements
theorem unique_parallel_line_exists : unique_parallel_line L P :=
sorry

theorem infinitely_many_parallel_planes_exist : infinitely_many_parallel_planes L P :=
sorry

end Geometry

end unique_parallel_line_exists_infinitely_many_parallel_planes_exist_l167_167988


namespace circumferences_ratio_l167_167181

theorem circumferences_ratio (r1 r2 : ℝ) (h : (π * r1 ^ 2) / (π * r2 ^ 2) = 49 / 64) : r1 / r2 = 7 / 8 :=
sorry

end circumferences_ratio_l167_167181


namespace tinplate_distribution_correct_l167_167647

-- Conditions
def box_body_per_tinplate := 25
def box_bottom_per_tinplate := 40
def total_tinplates := 36
def body_to_bottom_ratio := 2

-- Variables to prove:
def x (tinplates_for_bodies : ℕ) := tinplates_for_bodies
def y (tinplates_for_bottoms : ℕ) := total_tinplates - tinplates_for_bodies

-- Original conditions as equations:
lemma tinplates_used_correctly (bodies bottom_ratio bottoms : ℕ) (h1 : bodies = box_body_per_tinplate * x bodies) (h2 : bottom_ratio * bodies = box_bottom_per_tinplate * y bottoms) : Prop :=
  2 * bodies = box_bottom_per_tinplate * y bottoms

-- Theorem to prove
theorem tinplate_distribution_correct : ∃ (tinplates_for_bodies tinplates_for_bottoms : ℕ),
  total_tinplates = tinplates_for_bodies + tinplates_for_bottoms ∧
  body_to_bottom_ratio * box_body_per_tinplate * tinplates_for_bodies = box_bottom_per_tinplate * tinplates_for_bottoms ∧
  tinplates_for_bodies = 16 ∧
  tinplates_for_bottoms = 20 :=
by
  use 16, 20
  split; sorry  -- Skip the proof using sorry for now to complete the statement

end tinplate_distribution_correct_l167_167647


namespace sheep_problem_system_l167_167089

theorem sheep_problem_system :
  (∃ (x y : ℝ), 5 * x - y = -90 ∧ 50 * x - y = 0) ↔ 
  (5 * x - y = -90 ∧ 50 * x - y = 0) := 
by
  sorry

end sheep_problem_system_l167_167089


namespace profit_maximization_problem_l167_167706

theorem profit_maximization_problem :
  ∀ (x : ℝ), 
  let G := λ x, (2 + x) in
  let R := 
    λ x, if 0 ≤ x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x - 0.8 
         else 10.2 in
  let f := 
    λ x, if 0 ≤ x ∧ x ≤ 5 then -0.4 * x^2 + 3.2 * x - 2.8 
         else 8.2 - x in
  (f = λ x, R x - G x) ∧
  (0 ≤ x → x ≤ 5 → f x = -0.4 * (x - 4)^2 + 3.6) ∧
  (x > 5 → f x < 3.2) ∧
  (f 4 = 3.6) :=
by
  intro x G R f
  sorry

end profit_maximization_problem_l167_167706


namespace math_problem_l167_167437

variable {p q r x y : ℝ}

theorem math_problem (h1 : p / q = 6 / 7)
                     (h2 : p / r = 8 / 9)
                     (h3 : q / r = x / y) :
                     x = 28 ∧ y = 27 ∧ 2 * p + q = (19 / 6) * p := 
by 
  sorry

end math_problem_l167_167437


namespace problem1_problem2_l167_167255

-- Problem 1
theorem problem1 : 
  4 ^ (3 / 2) + ((Real.sqrt 2) * (Real.cbrt 3)) ^ 6 - 16 * ((2 + 10 / 27) : ℚ) ^ (- 2 / 3) = 71 :=
by sorry

-- Problem 2
theorem problem2 : 
  3 ^ (Real.log 2 / Real.log 3) - (2 / 3) * (Real.log 3 / Real.log 2) * (Real.log 8 / Real.log 3) + 
  (1 / 3) * (Real.log 8 / Real.log 6) + 2 * (Real.log (Real.sqrt 3) / Real.log 6) = 1 :=
by sorry

end problem1_problem2_l167_167255


namespace blue_eyed_kittens_percentage_approx_l167_167160

/-
We make all the conditions explicit, specifying how many blue-eyed and brown-eyed kittens each cat has.
Then, we will state what needs to be proven: That the percentage of blue-eyed kittens is approx. 41.67%
-/

noncomputable def blue_eyed_kittens (cat_num: ℕ) : ℕ :=
match cat_num with
| 1 => 5
| 2 => 6
| 3 => 4
| 4 => 7
| 5 => 3
| _ => 0
end

noncomputable def brown_eyed_kittens (cat_num: ℕ) : ℕ :=
match cat_num with
| 1 => 7
| 2 => 8
| 3 => 6
| 4 => 9
| 5 => 5
| _ => 0
end

noncomputable def total_blue_eyed_kittens : ℕ :=
(1 to 5).sum (λ i, blue_eyed_kittens i)

noncomputable def total_kittens : ℕ :=
(1 to 5).sum (λ i, blue_eyed_kittens i + brown_eyed_kittens i)

noncomputable def blue_percentage : ℝ :=
(total_blue_eyed_kittens.toReal / total_kittens.toReal) * 100

theorem blue_eyed_kittens_percentage_approx : abs (blue_percentage - 41.67) < 0.01 :=
by
  sorry

end blue_eyed_kittens_percentage_approx_l167_167160


namespace cobbler_work_percentage_l167_167725

def cobbler_initial_work_cost : ℝ := 75 * 8
def cobbler_mold_cost : ℝ := 250
def total_cost_without_discount : ℝ := cobbler_initial_work_cost + cobbler_mold_cost
def amount_paid_by_bobby : ℝ := 730
def discount_received : ℝ := total_cost_without_discount - amount_paid_by_bobby
def percentage_charged : ℝ := (discount_received / cobbler_initial_work_cost) * 100

theorem cobbler_work_percentage : percentage_charged = 20 :=
by
  sorry

end cobbler_work_percentage_l167_167725


namespace average_age_combined_l167_167948

theorem average_age_combined (nA nB nC : ℕ) (ageA ageB ageC : ℕ)
  (hA : nA = 8) (hB : nB = 5) (hC : nC = 7)
  (avgA : ageA = 30) (avgB : ageB = 35) (avgC : ageC = 40) :
  (nA * ageA + nB * ageB + nC * ageC) / (nA + nB + nC) = 34.75 :=
by
  have h1 : nA * ageA = 240 := by rw [hA, avgA]; sorry
  have h2 : nB * ageB = 175 := by rw [hB, avgB]; sorry
  have h3 : nC * ageC = 280 := by rw [hC, avgC]; sorry
  have h4 : nA + nB + nC = 20 := by rw [hA, hB, hC]; sorry
  have h5 : (nA * ageA + nB * ageB + nC * ageC) = 695 := by rw [h1, h2, h3]; sorry
  have h6 : (nA * ageA + nB * ageB + nC * ageC) / (nA + nB + nC) = 34.75 := by rw [h5, h4]; sorry
  exact h6

end average_age_combined_l167_167948


namespace range_of_inverse_sums_l167_167481

open Real

theorem range_of_inverse_sums (a : ℝ) (x_1 x_2 : ℝ) (hx1 : a > 1)
  (hx2 : f : ℝ → ℝ := λ x, a ^ x + x - 4)
  (hx3 : g : ℝ → ℝ := λ x, log a x + x - 4)
  (root_f : f x_1 = 0)
  (root_g : g x_2 = 0) :
  ∃ y, y = (1 / x_1 + 1 / x_2) ∧ ∀ x, x ∈ Set.Ici y ↔ x ≥ 1 :=
by sorry

end range_of_inverse_sums_l167_167481


namespace velocity_at_t2_l167_167048

variable (t : ℝ) (s : ℝ)

-- Define the motion equation
def motion_equation (t : ℝ) : ℝ := t^2 + 3 / t

-- Define the velocity as the derivative of the motion equation
noncomputable def velocity (t : ℝ) := (deriv motion_equation) t

-- Statement to prove
theorem velocity_at_t2 : velocity 2 = 13 / 4 :=
by {
  sorry
}

end velocity_at_t2_l167_167048


namespace find_MN_l167_167096

variables (a b c : ℝ^3)
-- a b c are vectors representing OA, OB, OC respectively

def M : ℝ^3 := (2 / 3) • a
-- Point M on OA such that OM = 2 * MA

def N : ℝ^3 := (1 / 2) • (b + c)
-- Point N is the midpoint of BC

def MN : ℝ^3 := N - M

theorem find_MN : MN a b c = (1 / 2) • b + (1 / 2) • c - (2 / 3) • a :=
by
  sorry

end find_MN_l167_167096


namespace cos_y_value_l167_167881

-- Define the conditions as given
def alpha : ℝ := Real.arccos (5 / 9)

-- The main statement we need to prove
theorem cos_y_value (y : ℝ) 
  (h1 : ∀ (x z : ℝ), x = y - alpha ∧ z = y + alpha ∧ 
     (1 + Real.cos (y - alpha)) * (1 + Real.cos (y + alpha)) = (1 + Real.cos y) * (1 + Real.cos y)) :
  Real.cos y = -7 / 9 := 
sorry

end cos_y_value_l167_167881


namespace isosceles_triangle_BDG_l167_167643

theorem isosceles_triangle_BDG (A B C D E G H I J : Point)
  (sq1 sq2 : Square) (EG_eq_EB : dist E G = dist E B)
  (common_side_DE : sq1.side = sq2.side) :
  is_isosceles B D G :=
by
  -- all the proof steps needed to demonstrate the theorem
  sorry

end isosceles_triangle_BDG_l167_167643


namespace max_value_of_f_l167_167394

open Real

theorem max_value_of_f :
  (∀ x ∈ Icc (π / 4) (5 * π / 12), 
    let f := λ x, (√2 * cos x * sin (x + π / 4)) / (sin (2 * x)) in
    f x ≤ 1) ∧ 
  (∃ x ∈ Icc (π / 4) (5 * π / 12), 
    let f := λ x, (√2 * cos x * sin (x + π / 4)) / (sin (2 * x)) in
    f x = 1) := by
  sorry

end max_value_of_f_l167_167394


namespace range_of_a_l167_167805

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a_seq n = a + n - 1)
  (h2 : ∀ n : ℕ, b n = (1 + a_seq n) / a_seq n)
  (h3 : ∀ n : ℕ, n > 0 → b n ≤ b 5) :
  -4 < a ∧ a < -3 :=
by
  sorry

end range_of_a_l167_167805


namespace StockPriceAdjustment_l167_167598

theorem StockPriceAdjustment (P₀ P₁ P₂ P₃ P₄ : ℝ) (january_increase february_decrease march_increase : ℝ) :
  P₀ = 150 →
  january_increase = 0.10 →
  february_decrease = 0.15 →
  march_increase = 0.30 →
  P₁ = P₀ * (1 + january_increase) →
  P₂ = P₁ * (1 - february_decrease) →
  P₃ = P₂ * (1 + march_increase) →
  142.5 <= P₃ * (1 - 0.17) ∧ P₃ * (1 - 0.17) <= 157.5 :=
by
  intros hP₀ hJanuaryIncrease hFebruaryDecrease hMarchIncrease hP₁ hP₂ hP₃
  sorry

end StockPriceAdjustment_l167_167598


namespace problem_sequences_l167_167630

noncomputable def a (n : ℕ) : ℕ := n
noncomputable def b (n : ℕ) : ℕ := 2^(n-1)
def S (n : ℕ) : ℕ := n * (n+1) / 2
def T (n : ℕ) : ℕ := 1 + (n-1) * 2^n

theorem problem_sequences (n : ℕ) :
  (a 1 = 1) ∧ (b 1 = 1) ∧ (b 2 * S 2 = 6) ∧ (b 2 + S 3 = 8) ∧
  (∀ n, T(n) = finset.sum (finset.range n) (λ k, (a (k + 1)) * (b (k + 1)))) :=
by
  sorry

end problem_sequences_l167_167630


namespace min_sum_sides_l167_167303

theorem min_sum_sides (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  minimum_sum_of_sides m n = m + n - gcd m n :=
sorry

end min_sum_sides_l167_167303


namespace linda_total_distance_l167_167565

theorem linda_total_distance
  (miles_per_gallon : ℝ) (tank_capacity : ℝ) (initial_distance : ℝ) (refuel_amount : ℝ) (final_tank_fraction : ℝ)
  (fuel_used_first_segment : ℝ := initial_distance / miles_per_gallon)
  (initial_fuel_full : fuel_used_first_segment = tank_capacity)
  (total_fuel_after_refuel : ℝ := 0 + refuel_amount)
  (remaining_fuel_stopping : ℝ := final_tank_fraction * tank_capacity)
  (fuel_used_second_segment : ℝ := total_fuel_after_refuel - remaining_fuel_stopping)
  (distance_second_leg : ℝ := fuel_used_second_segment * miles_per_gallon) :
  initial_distance + distance_second_leg = 637.5 := by
  sorry

end linda_total_distance_l167_167565


namespace charging_bull_rounds_l167_167965

theorem charging_bull_rounds 
    (racing_magic_placeholder_seconds : Nat)
    (time_to_meet_minutes : Nat)
    (hours_to_minutes : Nat)
    (laps_by_charging_bull : Nat) : 
    racing_magic_placeholder_seconds = 60 →
    time_to_meet_minutes = 6 →
    hours_to_minutes = 60 →
    laps_by_charging_bull = 70 :=
begin
    sorry
end

end charging_bull_rounds_l167_167965


namespace imaginary_part_of_z_l167_167402

noncomputable def complex_imag_part (z : ℂ) : Prop := (4 + 3 * complex.i) * z = -complex.i

theorem imaginary_part_of_z (z : ℂ) (h : complex_imag_part z) : complex.im z = -4 / 25 :=
by sorry

end imaginary_part_of_z_l167_167402


namespace parabola_equation_l167_167789

theorem parabola_equation (focus_x : ℝ) (focus_y : ℝ) (a b c : ℝ) (y x : ℝ) 
  (h_vertex : (0, 0)) (h_focus : (focus_x, focus_y)) (h_line : y = 2 * x + 1)
  (h_chord_length : (abs (2 * focus_x - 4) * (focus_x - 2) - 1 = 15)) :
  (y^2 = -4 * x ∨ y^2 = 12 * x) :=
by
  sorry

end parabola_equation_l167_167789


namespace domain_of_sqrt_log_eq_l167_167188

theorem domain_of_sqrt_log_eq (x : ℝ) : (sqrt (log (1 / 2) (4 * x - 3))) = (x ∈ set.Ioc 3/4 1) :=
by 
  sorry

end domain_of_sqrt_log_eq_l167_167188


namespace smallest_value_of_3a_plus_2_l167_167478

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) : 
  ∃ (x : ℝ), x = 3 * a + 2 ∧ x = -1 :=
by
  sorry

end smallest_value_of_3a_plus_2_l167_167478


namespace order_of_roots_l167_167145

noncomputable def a : ℝ := some (Real.solve_eq (λ x, x + log x / log 2 - 2))
noncomputable def b : ℝ := some (Real.solve_eq (λ x, x + log x / log 3 - 2))
noncomputable def c : ℝ := some (Real.solve_eq (λ x, x + log x / log 2 - 1))

theorem order_of_roots : c < a ∧ a < b :=
by
  -- sorry will be replaced by proof during formal verification
  sorry

end order_of_roots_l167_167145


namespace minimum_pieces_to_form_3x3_square_l167_167129

theorem minimum_pieces_to_form_3x3_square (shape : fin 9 → ℕ) 
  (goal :ℕ)
  (condition_is_all_squares_rearranged: ∀ s, set.range shape = finset.range 9):  
  goal = 3 ↔ 
  (∃ pieces : list (finset ℕ), (∀ p ∈ pieces, p.card ≤ 3) ∧ (∀ i j, shape i ≠ shape j → ∃ p ∈ pieces, i ∈ p.to_finset ∧ j ∈ p.to_finset)) :=
sorry

end minimum_pieces_to_form_3x3_square_l167_167129


namespace speed_difference_between_lucy_and_sam_l167_167296

noncomputable def average_speed (distance : ℚ) (time_minutes : ℚ) : ℚ :=
  distance / (time_minutes / 60)

theorem speed_difference_between_lucy_and_sam :
  let distance := 6
  let lucy_time := 15
  let sam_time := 45
  let lucy_speed := average_speed distance lucy_time
  let sam_speed := average_speed distance sam_time
  (lucy_speed - sam_speed) = 16 :=
by
  sorry

end speed_difference_between_lucy_and_sam_l167_167296


namespace count_special_primes_l167_167764

/--
  Prove that the count of numbers less than 1000, divisible by 4, prime,
  and not containing the digits 6, 7, 8, 9, or 0, is equal to 31.
-/
theorem count_special_primes : 
  let valid_digit (d : ℕ) := d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5
  let contains_valid_digits (n : ℕ) := ∀ d ∈ (n.to_string.to_list.map (λ x => x.to_nat - '0'.to_nat)), valid_digit d
  let is_valid_prime (n : ℕ) := nat.prime n ∧ (n % 4 = 0) ∧ (n < 1000) ∧ contains_valid_digits n
  (finset.range 1000).filter is_valid_prime).card = 31 :=
begin
  sorry
end

end count_special_primes_l167_167764


namespace distinct_digits_mean_l167_167600

theorem distinct_digits_mean (M : ℕ) :
  (∀ n, n ∈ {9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999} → M = (9 + 99 + 999 + 9999 + 99999 + 999999 + 9999999 + 99999999 + 999999999) / 9) →
  M = 123456789 ∧ (∀ d : ℕ, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → d ≠ 0 → d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) :=
by 
  sorry

end distinct_digits_mean_l167_167600


namespace sum_of_valid_m_values_l167_167744

-- Given conditions as definitions:
def distinct_addition (s : Set ℤ) (m : ℤ) : Prop :=
  m ∉ s ∧ (median (s ∪ {m}) = mean (s ∪ {m}))

-- The set in question:
def base_set : Set ℤ := {4, 7, 8, 12}

-- Sum of all valid m values for which median equals mean:
theorem sum_of_valid_m_values : ∑ m in (Filter (distinct_addition base_set)).toFinset, m = 13 := 
  sorry

end sum_of_valid_m_values_l167_167744


namespace average_speed_round_trip_l167_167685

theorem average_speed_round_trip (D : ℝ) (hD_nonzero: D ≠ 0):
  let T1 := D / 60
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  average_speed := total_distance / total_time
  average_speed = 45 := by
  sorry

end average_speed_round_trip_l167_167685


namespace sum_of_three_integers_mod_53_l167_167671

theorem sum_of_three_integers_mod_53 (a b c : ℕ) (h1 : a % 53 = 31) 
                                     (h2 : b % 53 = 22) (h3 : c % 53 = 7) : 
                                     (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_three_integers_mod_53_l167_167671


namespace contrapositive_sine_l167_167186

variable (α : ℝ)

theorem contrapositive_sine (h : α = π / 3 → sin α = sqrt 3 / 2) :
  sin α ≠ sqrt 3 / 2 → α ≠ π / 3 :=
by
  sorry

end contrapositive_sine_l167_167186


namespace rectangular_sheet_integers_l167_167710

noncomputable def at_least_one_integer (a b : ℝ) : Prop :=
  ∃ i : ℤ, a = i ∨ b = i

theorem rectangular_sheet_integers (a b : ℝ)
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_cut_lines : ∀ x y : ℝ, (∃ k : ℤ, x = k ∧ y = 1 ∨ y = k ∧ x = 1) → (∃ z : ℤ, x = z ∨ y = z)) :
  at_least_one_integer a b :=
sorry

end rectangular_sheet_integers_l167_167710


namespace inscribed_sphere_radius_l167_167692

theorem inscribed_sphere_radius
  (a : ℝ) (A B C D M N : ℝ) 
  (h1 : M = center_of_face_ADC)
  (h2 : N = midpoint B C)
  :
  r = \(\frac{5\sqrt{6} - 3\sqrt{2}}{48} a\) := 
by
  sorry

end inscribed_sphere_radius_l167_167692


namespace count_distinct_three_digit_even_numbers_l167_167469

theorem count_distinct_three_digit_even_numbers : 
  let even_digits := {0, 2, 4, 6, 8} in
  let first_digit_choices := {2, 4, 6, 8} in
  let second_and_third_digit_choices := even_digits in
  (finset.card first_digit_choices) * 
  (finset.card second_and_third_digit_choices) *
  (finset.card second_and_third_digit_choices) = 100 := by
  let even_digits := {0, 2, 4, 6, 8}
  let first_digit_choices := {2, 4, 6, 8}
  let second_and_third_digit_choices := even_digits
  have h1 : finset.card first_digit_choices = 4 := by simp
  have h2 : finset.card second_and_third_digit_choices = 5 := by simp
  calc (finset.card first_digit_choices) * 
       (finset.card second_and_third_digit_choices) *
       (finset.card second_and_third_digit_choices)
       = 4 * 5 * 5 : by rw [h1, h2]
    ... = 100 : by norm_num

end count_distinct_three_digit_even_numbers_l167_167469


namespace krakozyabrs_proof_l167_167114

-- Defining necessary variables and conditions
variable {n : ℕ}

-- Conditions given in the problem
def condition1 (H : ℕ) : Prop := 0.2 * H = n
def condition2 (W : ℕ) : Prop := 0.25 * W = n

-- Definition using the principle of inclusion-exclusion
def total_krakozyabrs (H W : ℕ) : ℕ := H + W - n

-- Statements reflecting the conditions and the final conclusion
theorem krakozyabrs_proof (H W : ℕ) (n : ℕ) : condition1 H → condition2 W → (25 < total_krakozyabrs H W) ∧ (total_krakozyabrs H W < 35) → total_krakozyabrs H W = 32 :=
by
  -- We skip the proof here
  sorry

end krakozyabrs_proof_l167_167114


namespace complex_subtraction_l167_167797

theorem complex_subtraction (z1 z2 : ℂ) (h1 : z1 = 3 + 4 * complex.I) (h2 : z2 = 3 - 4 * complex.I) : 
  z1 - z2 = 8 * complex.I :=
by
  sorry

end complex_subtraction_l167_167797


namespace probability_of_event_3a_minus_1_gt_0_l167_167020

noncomputable def probability_event : ℝ :=
if h : 0 <= 1 then (1 - 1/3) else 0

theorem probability_of_event_3a_minus_1_gt_0 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) : 
  probability_event = 2 / 3 :=
by
  sorry

end probability_of_event_3a_minus_1_gt_0_l167_167020


namespace francine_normal_frogs_l167_167862

theorem francine_normal_frogs (F : ℕ) (hF : 0.33 * F = 9) : 0.67 * F = 18 :=
sorry

end francine_normal_frogs_l167_167862


namespace distinct_three_digit_numbers_with_even_digits_l167_167459

theorem distinct_three_digit_numbers_with_even_digits : 
  let even_digits := {0, 2, 4, 6, 8} in
  (∃ (hundreds options : Finset ℕ) (x : ℕ), 
    hundreds = {2, 4, 6, 8} ∧ 
    options = even_digits ∧ 
    x = Finset.card hundreds * Finset.card options * Finset.card options ∧ 
    x = 100) :=
by
  let even_digits := {0, 2, 4, 6, 8}
  exact ⟨{2, 4, 6, 8}, even_digits, 100, rfl, rfl, sorry, rfl⟩

end distinct_three_digit_numbers_with_even_digits_l167_167459


namespace option_b_incorrect_l167_167732

theorem option_b_incorrect : ¬ ( (sqrt 5 - 1) / 2 < 0.5 ) :=
by
  sorry

end option_b_incorrect_l167_167732


namespace krakozyabr_count_l167_167119

variable (n H W T : ℕ)
variable (h1 : H = 5 * n) -- 20% of the 'krakozyabrs' with horns also have wings
variable (h2 : W = 4 * n) -- 25% of the 'krakozyabrs' with wings also have horns
variable (h3 : T = H + W - n) -- Total number of 'krakozyabrs' using inclusion-exclusion
variable (h4 : 25 < T)
variable (h5 : T < 35)

theorem krakozyabr_count : T = 32 := by
  sorry

end krakozyabr_count_l167_167119


namespace sum_of_solutions_l167_167136

theorem sum_of_solutions :
  (∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    ((x, y) = (-1, 0) ∨ (x, y) = (2, 3) ∨ (x, y) = (7, 2))) →
  ((∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    (1 + 1 = 3 ∨ true)) → 
  (∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    (x, y) = (-1, 0) ∨ (x, y) = (2, 3) ∨ (x, y) = (7, 2))) →
  (-1) + 0 + 2 + 3 + 7 + 2 = 13 :=
by
  sorry

end sum_of_solutions_l167_167136


namespace vector_expression_value_l167_167835

theorem vector_expression_value 
  (θ : ℝ) 
  (cosθ : ℝ := cos θ)
  (sinθ : ℝ := sin θ)
  (h_parallel: sinθ = -2 * cosθ) :
  (2 * sinθ - cosθ) / (sinθ + cosθ) = 5 :=
by
  sorry

end vector_expression_value_l167_167835


namespace exists_C_l167_167413

def A : set ℤ := { x | -1 ≤ x ∧ x ≤ 2 }
def B : set ℤ := { x | -3 < x ∧ x < -1 }

def valid_elements : set ℤ := { x | -3 < x ∧ x ≤ 2 }

theorem exists_C (C : set ℤ) : 
  (∀ x, x ∈ A ∪ B → x ∈ valid_elements) → 
  (∀ x, x ∈ valid_elements → x ∈ A ∪ B) →
  (C ⊆ valid_elements) → 
  (|C| = 3) → 
  (C ∩ B ≠ ∅ ) →
  C = {-2, -1, 0} ∨ C = {-2, -1, 1} ∨ C = {-2, -1, 2} ∨ C = {-2, 0, 1} ∨ C = {-2, 0, 2} ∨ C = {-2, 1, 2} :=
sorry

end exists_C_l167_167413


namespace kaleb_lost_lives_l167_167240

theorem kaleb_lost_lives :
  ∀ (initial_lives remaining_lives lives_lost : ℕ),
    initial_lives = 98 →
    remaining_lives = 73 →
    lives_lost = initial_lives - remaining_lives →
    lives_lost = 25 :=
by
  intros initial_lives remaining_lives lives_lost
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3
  sorry

end kaleb_lost_lives_l167_167240


namespace area_of_region_eq_50pi_l167_167602

def upper_function (x : ℝ) : ℝ := sin (2 * x) + 10
def lower_function (x : ℝ) : ℝ := 2 * cos (4 * x)
def interval_start : ℝ := 0
def interval_end : ℝ := 5 * Real.pi

theorem area_of_region_eq_50pi : 
  ∫ x in interval_start..interval_end, (upper_function x - lower_function x) = 50 * Real.pi := by
  sorry

end area_of_region_eq_50pi_l167_167602


namespace canteen_needs_bananas_l167_167270

-- Define the given conditions
def total_bananas := 9828
def weeks := 9
def days_in_week := 7
def bananas_in_dozen := 12

-- Calculate the required value and prove the equivalence
theorem canteen_needs_bananas : 
  (total_bananas / (weeks * days_in_week)) / bananas_in_dozen = 13 :=
by
  -- This is where the proof would go
  sorry

end canteen_needs_bananas_l167_167270


namespace percent_less_than_m_plus_s_l167_167861

-- Define the parameters and conditions
variables {α : Type*} [LinearOrder α]
variables (m : α) (s : α)
variables (dist : Set α)

-- The distribution is symmetric about the mean m
def is_symmetric_about_mean (dist : Set α) (m : α) : Prop :=
  ∀ x ∈ dist, (2 * m - x) ∈ dist

-- 68% of the distribution lies within one standard deviation s of the mean
def within_one_std_dev (dist : Set ℝ) (m s : ℝ) : Prop :=
  (∃ P Q : Set ℝ, P ∪ Q = dist ∧
                  (∀ x ∈ P, m - s ≤ x ∧ x ≤ m + s) ∧
                  (∀ x ∈ Q, x < m - s ∨ x > m + s) ∧
                  P.fraction 68)

-- The theorem to be proven
theorem percent_less_than_m_plus_s (h_sym : is_symmetric_about_mean dist m) (h_std : within_one_std_dev dist m s) :
  ∃ (percent : ℝ), percent = 84 := 
sorry

end percent_less_than_m_plus_s_l167_167861


namespace derivative_y_l167_167686

variables {α β x : ℝ}

def y (α β x : ℝ) : ℝ := (exp (α * x) * (α * sin (β * x) - β * cos (β * x))) / (α^2 + β^2)

theorem derivative_y :
  ∀ (α β x : ℝ), deriv (λ x, y α β x) x = exp (α * x) * sin (β * x) :=
by sorry

end derivative_y_l167_167686


namespace points_count_l167_167082

theorem points_count (A B C : ℝ × ℝ) :
  dist A B = 12 →
  let a := (0, 0)
  let b := (12, 0)
  let perimeter : ℝ := 12 + dist a C + dist b C
  let area : ℝ := 1/2 * 12 * |C.snd|
  (perimeter = 60 ∧ area = 72) →
  (C = (6, 12) ∨ C = (6, -12)) → 
  ∃! C, (perimeter = 60 ∧ area = 72) :=
by
  intros hAB ha hb perimeter_def area_def hCond
  sorry

end points_count_l167_167082


namespace log_base_3_l167_167352

theorem log_base_3 (h : (1 / 81 : ℝ) = 3 ^ (-4 : ℝ)) : Real.logBase 3 (1 / 81) = -4 := 
by sorry

end log_base_3_l167_167352


namespace different_8_digit_integers_count_l167_167446

theorem different_8_digit_integers_count :
  let first_digit_choices := 9 in
  let other_digit_choices := 5 in
  first_digit_choices * other_digit_choices ^ 7 = 703125 :=
by
  sorry

end different_8_digit_integers_count_l167_167446


namespace sum_of_values_of_b_l167_167754

theorem sum_of_values_of_b (b : ℝ) (h : ∀ x, 3 * x^2 + (b + 12) * x + 5 = 0 → x * 3 + (b + 12) * x + 5) :
  (∀ x1 x2, 3 * x1^2 + (bx + 12x) * x1 + 5 = 0 ∧ 3 * x2^2 + (bx + 12x) * x2 + 5 = 0 → x1 = x2) →
  b = -12 + 2 * real.sqrt 15 ∨ b = -12 - 2 * real.sqrt 15 →
  b = -24 :=
sorry

end sum_of_values_of_b_l167_167754


namespace total_area_of_storage_units_l167_167604

theorem total_area_of_storage_units (total_units remaining_units : ℕ) 
    (size_8_by_4 length width unit_area_200 : ℕ)
    (h1 : total_units = 42)
    (h2 : remaining_units = 22)
    (h3 : length = 8)
    (h4 : width = 4)
    (h5 : unit_area_200 = 200) 
    (h6 : ∀ i : ℕ, i < 20 → unit_area_8_by_4 = length * width) 
    (h7 : ∀ j : ℕ, j < 22 → unit_area_200 = 200) :
    total_area_of_all_units = 5040 :=
by
  let unit_area_8_by_4 := length * width
  let total_area_20_units := 20 * unit_area_8_by_4
  let total_area_22_units := 22 * unit_area_200
  let total_area_of_all_units := total_area_20_units + total_area_22_units
  sorry

end total_area_of_storage_units_l167_167604


namespace krakozyabr_count_l167_167120

variable (n H W T : ℕ)
variable (h1 : H = 5 * n) -- 20% of the 'krakozyabrs' with horns also have wings
variable (h2 : W = 4 * n) -- 25% of the 'krakozyabrs' with wings also have horns
variable (h3 : T = H + W - n) -- Total number of 'krakozyabrs' using inclusion-exclusion
variable (h4 : 25 < T)
variable (h5 : T < 35)

theorem krakozyabr_count : T = 32 := by
  sorry

end krakozyabr_count_l167_167120


namespace circles_are_separate_l167_167752

def circle1_center : ℝ × ℝ := (-1/2, -1)
def circle2_center (θ : ℝ) : ℝ × ℝ := (Real.sin θ, 1)
def radius1 : ℝ := Real.sqrt (1/2)
def radius2 : ℝ := 1/4
def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circles_are_separate (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2) :
  distance circle1_center (circle2_center θ) > radius1 + radius2 :=
by
  sorry

end circles_are_separate_l167_167752


namespace percentage_shaded_grid_l167_167232

theorem percentage_shaded_grid :
  let n := 7
  let total_squares := n * n
  let unshaded_squares_row := n
  let unshaded_squares_col := n - 1 -- exclude the intersecting unshaded square
  let shaded_squares := total_squares - (unshaded_squares_row + unshaded_squares_col)
  (shaded_squares : ℝ) / total_squares * 100 ≈ 73.47 :=
by
  sorry

end percentage_shaded_grid_l167_167232


namespace delegate_rooms_l167_167723

-- Define the parameters and the problem itself
theorem delegate_rooms (delegates : Fin 1000 → Type) (speaks_language : ∀ (d : Fin 1000), Set (Fin 1000)) :
  (∀ (d1 d2 d3 : Fin 1000), ∃ (l ∈ speaks_language d1), l ∈ speaks_language d2 ∧ l ∈ speaks_language d3) →
  ∃ (rooms : Fin 500 → (Fin 1000 × Fin 1000)),
    (∀ (r : Fin 500), (rooms r).fst ≠ (rooms r).snd ∧
    (rooms r).fst < (rooms r).snd ∧
    (∃ l ∈ speaks_language (rooms r).fst, l ∈ speaks_language (rooms r).snd)) := 
sorry

end delegate_rooms_l167_167723


namespace hyperbola_parabola_focus_l167_167071

theorem hyperbola_parabola_focus (m : ℝ) :
  (m + (m - 2) = 4) → m = 3 :=
by
  intro h
  sorry

end hyperbola_parabola_focus_l167_167071


namespace distinct_three_digit_numbers_with_even_digits_l167_167461

theorem distinct_three_digit_numbers_with_even_digits : 
  let even_digits := {0, 2, 4, 6, 8} in
  (∃ (hundreds options : Finset ℕ) (x : ℕ), 
    hundreds = {2, 4, 6, 8} ∧ 
    options = even_digits ∧ 
    x = Finset.card hundreds * Finset.card options * Finset.card options ∧ 
    x = 100) :=
by
  let even_digits := {0, 2, 4, 6, 8}
  exact ⟨{2, 4, 6, 8}, even_digits, 100, rfl, rfl, sorry, rfl⟩

end distinct_three_digit_numbers_with_even_digits_l167_167461


namespace krakozyabr_count_l167_167118

variable (n H W T : ℕ)
variable (h1 : H = 5 * n) -- 20% of the 'krakozyabrs' with horns also have wings
variable (h2 : W = 4 * n) -- 25% of the 'krakozyabrs' with wings also have horns
variable (h3 : T = H + W - n) -- Total number of 'krakozyabrs' using inclusion-exclusion
variable (h4 : 25 < T)
variable (h5 : T < 35)

theorem krakozyabr_count : T = 32 := by
  sorry

end krakozyabr_count_l167_167118


namespace total_games_played_l167_167684

theorem total_games_played (n : ℕ) (h : n = 7) : (n.choose 2) = 21 := by
  sorry

end total_games_played_l167_167684


namespace divisible_by_4_of_product_and_sum_exists_integers_with_product_and_sum_zero_l167_167245

-- Part (a)
theorem divisible_by_4_of_product_and_sum
  {n : ℕ} (a : fin n → ℤ) (h1 : (∏ i, a i) = n) (h2 : (∑ i, a i) = 0) : 4 ∣ n :=
sorry

-- Part (b)
theorem exists_integers_with_product_and_sum_zero
  {n : ℕ} (hn : 4 ∣ n) : ∃ (a : fin n → ℤ), (∏ i, a i) = n ∧ (∑ i, a i) = 0 :=
sorry

end divisible_by_4_of_product_and_sum_exists_integers_with_product_and_sum_zero_l167_167245


namespace sum_of_coefficients_is_7_l167_167909

noncomputable def sequence_sum_of_coefficients (v : ℕ → ℕ) :=
  v 1 = 7 ∧ ∀ n : ℕ, v (n + 1) - v n = 5 + 6 * (n - 1) →
  let a := 3 in
  let b := -4 in
  let c := 8 in
  a + b + c = 7

theorem sum_of_coefficients_is_7 (v : ℕ → ℕ) :
  sequence_sum_of_coefficients v :=
by
  sorry

end sum_of_coefficients_is_7_l167_167909


namespace odd_function_f_neg_one_l167_167022

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.logBase 2 (x + 3) else - Real.logBase 2 (-x + 3)

theorem odd_function_f_neg_one : f (-1) = -2 := 
by {
  -- Given that f is odd and f(x) = log₂(x + 3) for x > 0
  have h1 : ∀ x, f (-x) = -f x := by sorry,
  have h2 : ∀ x, x > 0 -> f x = Real.logBase 2 (x + 3) := by sorry,
  -- Now to show that f(-1) = -2
  have h3 : f 1 = 2 := by sorry,
  sorry
}

end odd_function_f_neg_one_l167_167022


namespace cos_double_angle_identity_l167_167026

theorem cos_double_angle_identity (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 - 2 * α) = -7 / 9 := 
sorry

end cos_double_angle_identity_l167_167026


namespace tom_original_weight_l167_167213

theorem tom_original_weight (W : ℕ) (h1 : 2 * (2 * W * 1.10) = 352) : W = 80 :=
by
  sorry

end tom_original_weight_l167_167213


namespace sequence_continues_indefinitely_and_never_zero_l167_167973

noncomputable def ternary_seq := (ℕ+ → ℝ) × (ℕ+ → ℝ) × (ℕ+ → ℝ)

def x_1 : ℝ := 2
def y_1 : ℝ := 4
def z_1 : ℝ := (6 / 7 : ℝ)

def x_recursion (x : ℕ+ → ℝ) (n : ℕ+) : ℝ := 2 * x n / (x n ^ 2 - 1)
def y_recursion (y : ℕ+ → ℝ) (n : ℕ+) : ℝ := 2 * y n / (y n ^ 2 - 1)
def z_recursion (z : ℕ+ → ℝ) (n : ℕ+) : ℝ := 2 * z n / (z n ^ 2 - 1)

theorem sequence_continues_indefinitely_and_never_zero (x y z : ternary_seq) :
  (∀ n > 1, x n = x_recursion x (n - 1) ∧ y n = y_recursion y (n - 1) ∧ z n = z_recursion z (n - 1)) →
  (x 1 = x_1 ∧ y 1 = y_1 ∧ z 1 = z_1) →
  (∀ n, x n ≠ 1 ∧ x n ≠ -1 ∧ y n ≠ 1 ∧ y n ≠ -1 ∧ z n ≠ 1 ∧ z n ≠ -1) ∧
  (∀ n, x n + y n + z n ≠ 0) :=
sorry

end sequence_continues_indefinitely_and_never_zero_l167_167973


namespace prove_p_value_l167_167307

noncomputable def binomial_coefficient (n k: ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Assign the given binomial coefficient
def C_7_4 : ℕ := binomial_coefficient 7 4

-- Define v in terms of binomial distribution
def v (p : ℝ) : ℝ := C_7_4 * p^4 * (1 - p)^3

-- Given condition
axiom v_value : v 0.7 = 343 / 3125

theorem prove_p_value : ∃ p : ℝ, v p = 343 / 3125 ∧ p = 0.7 :=
by
  sorry

end prove_p_value_l167_167307


namespace find_max_marks_l167_167244

theorem find_max_marks 
  (M : ℕ) 
  (passing_percentage : ℝ) 
  (marks_obtained : ℕ) 
  (marks_failed_by : ℕ) 
  (passing_marks_eq : marks_obtained + marks_failed_by = (passing_percentage * M).to_nat) 
  (passing_percentage_eq : passing_percentage = 0.33)
  (marks_obtained_eq : marks_obtained = 175)
  (marks_failed_by_eq : marks_failed_by = 89) 
  : M = 800 :=
by
  sorry

end find_max_marks_l167_167244


namespace c_10_value_l167_167552

def c : ℕ → ℤ
| 0 => 3
| 1 => 9
| (n + 1) => c n * c (n - 1)

theorem c_10_value : c 10 = 3^89 :=
by
  sorry

end c_10_value_l167_167552


namespace inequality_for_any_x_l167_167196

theorem inequality_for_any_x (a : ℝ) (h : ∀ x : ℝ, |3 * x + 2 * a| + |2 - 3 * x| - |a + 1| > 2) :
  a < -1/3 ∨ a > 5 := 
sorry

end inequality_for_any_x_l167_167196


namespace calculate_expression_l167_167230

theorem calculate_expression : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end calculate_expression_l167_167230


namespace krakozyabrs_total_count_l167_167103

theorem krakozyabrs_total_count :
  (∃ (T : ℕ), 
  (∀ (H W n : ℕ),
    0.2 * H = n ∧ 0.25 * W = n ∧ H = 5 * n ∧ W = 4 * n ∧ T = H + W - n ∧ 25 < T ∧ T < 35) ∧ 
  T = 32) :=
by sorry

end krakozyabrs_total_count_l167_167103


namespace even_digit_numbers_count_eq_100_l167_167454

-- Definition for the count of distinct three-digit positive integers with only even digits
def count_even_digit_three_numbers : ℕ :=
  let hundreds_place := {2, 4, 6, 8}.card
  let tens_units_place := {0, 2, 4, 6, 8}.card
  hundreds_place * tens_units_place * tens_units_place

-- Theorem stating the count of distinct three-digit positive integers with only even digits is 100
theorem even_digit_numbers_count_eq_100 : count_even_digit_three_numbers = 100 :=
by sorry

end even_digit_numbers_count_eq_100_l167_167454


namespace conjugate_in_third_quadrant_l167_167421

noncomputable def z (i : ℂ) : ℂ :=
  i / (1 - i)

theorem conjugate_in_third_quadrant
  (i : ℂ) (hi : i^2 = -1) (z : ℂ) (hz : z * (i - 1) = i) :
  (z.conj.re < 0) ∧ (z.conj.im < 0) :=
sorry

end conjugate_in_third_quadrant_l167_167421


namespace savings_amount_correct_l167_167295

-- Define the expenses
def rent := 5000
def milk := 1500
def groceries := 4500
def education := 2500
def petrol := 2000
def miscellaneous := 5650
def medical := 3000
def utilities := 4000
def entertainment := 1000

-- Define the total expenses
def total_expenses := rent + milk + groceries + education + petrol + miscellaneous + medical + utilities + entertainment

-- Define the savings percentage
def savings_percentage := 0.20

-- Define the total salary
def total_salary := total_expenses / (1 - savings_percentage)

-- Define the savings
def savings := savings_percentage * total_salary

-- Theorem to prove the correct savings amount
theorem savings_amount_correct : savings = 7537.50 :=
by
  -- The tactic proof goes here
  sorry

end savings_amount_correct_l167_167295


namespace find_ratio_l167_167434

-- Define vectors and their norms and dot products
variables {x1 y1 x2 y2 : ℝ}
def a : EuclideanSpace ℝ (Fin 2) := ![x1, y1]
def b : EuclideanSpace ℝ (Fin 2) := ![x2, y2]

theorem find_ratio : ∥a∥ = 3 → ∥b∥ = 4 → ⟪a, b⟫ = -12 → (x1 + y1) / (x2 + y2) = -3/4 :=
by
  sorry

end find_ratio_l167_167434


namespace find_highway_speed_l167_167271

def car_local_distance := 40
def car_local_speed := 20
def car_highway_distance := 180
def average_speed := 44
def speed_of_car_on_highway := 60

theorem find_highway_speed :
  car_local_distance / car_local_speed + car_highway_distance / speed_of_car_on_highway = (car_local_distance + car_highway_distance) / average_speed :=
by
  sorry

end find_highway_speed_l167_167271


namespace krakozyabrs_total_count_l167_167106

theorem krakozyabrs_total_count :
  (∃ (T : ℕ), 
  (∀ (H W n : ℕ),
    0.2 * H = n ∧ 0.25 * W = n ∧ H = 5 * n ∧ W = 4 * n ∧ T = H + W - n ∧ 25 < T ∧ T < 35) ∧ 
  T = 32) :=
by sorry

end krakozyabrs_total_count_l167_167106


namespace base4_arithmetic_proof_l167_167325

theorem base4_arithmetic_proof :
  let b321 := (3 * 4^2 + 2 * 4^1 + 1) in
  let b21 := (2 * 4^1 + 1) in
  let b3 := (3) in
  let div_result := (b321 / b3) in
  let div_result_base4 := (div_result % 4 + (div_result / 4) % 4 * 10 + (div_result / 16) % 4 * 100) in
  let mul_res := (div_result_base4 * b21) in
  let final_res_base4 := (mul_res % 4 + (mul_res / 4) % 4 * 10 + (mul_res / 16) % 4 * 100 + (mul_res / 64) % 4 * 1000) in
  final_res_base4 = 2 * 1000 + 2 * 100 + 2 * 10 + 3 := 
sorry

end base4_arithmetic_proof_l167_167325


namespace cos_diff_alpha_beta_l167_167393

variables (α β : Real)
variables (h1 : sin (α - π / 6) = 2 / 3)
variables (h2 : π < α ∧ α < 3 * π / 2)
variables (h3 : cos (π / 3 + β) = 5 / 13)
variables (h4 : 0 < β ∧ β < π)

theorem cos_diff_alpha_beta : cos (β - α) = - (10 + 12 * Real.sqrt 5) / 39 :=
by
  sorry

end cos_diff_alpha_beta_l167_167393


namespace largest_domain_l167_167748

def domain_condition (g : ℝ → ℝ) (x : ℝ) : Prop :=
  g x + g (1 / x^2) = x^2

theorem largest_domain (g : ℝ → ℝ) :
  (∀ x, x ≠ 0 → domain_condition g x) ↔ (domain_condition g ≠ 0) := by
  sorry

end largest_domain_l167_167748


namespace rel_prime_probability_30_l167_167641

def is_rel_prime (a b : ℕ) : Prop := gcd a b = 1

def count_rel_prime (n : ℕ) : ℕ :=
  (List.range n).count (λ i, is_rel_prime (i+1) n)

def probability_rel_prime (n : ℕ) : ℚ :=
  count_rel_prime n / n

theorem rel_prime_probability_30 : probability_rel_prime 30 = 4 / 15 := 
  sorry

end rel_prime_probability_30_l167_167641


namespace range_of_a_l167_167816

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → a < x ∧ x < 5) → a ≤ 1 := 
sorry

end range_of_a_l167_167816


namespace larger_number_l167_167606

theorem larger_number (A B : ℝ) (h1 : A - B = 1650) (h2 : 0.075 * A = 0.125 * B) : A = 4125 :=
sorry

end larger_number_l167_167606


namespace measure_angle_PRQ_l167_167513

theorem measure_angle_PRQ :
  ∀ (P Q R S : Type)
    (distance : P → P → ℝ)
    (angle : P → P → P → ℝ),
    (distance P R = distance R S) ∧
    (distance R S = distance S Q) ∧
    (distance S Q = distance Q R) ∧
    (angle P Q R = 150) →
    angle P R Q = 35 :=
begin
  sorry
end

end measure_angle_PRQ_l167_167513


namespace inverse_h_l167_167542

-- definitions of f, g, and h
def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := 3 * x + 7
def h (x : ℝ) : ℝ := f (g x)

-- statement of the problem
theorem inverse_h (x : ℝ) : (∃ y : ℝ, h y = x) ∧ ∀ y : ℝ, h y = x → y = (x - 23) / 12 :=
by
  sorry

end inverse_h_l167_167542


namespace krakozyabr_count_l167_167121

variable (n H W T : ℕ)
variable (h1 : H = 5 * n) -- 20% of the 'krakozyabrs' with horns also have wings
variable (h2 : W = 4 * n) -- 25% of the 'krakozyabrs' with wings also have horns
variable (h3 : T = H + W - n) -- Total number of 'krakozyabrs' using inclusion-exclusion
variable (h4 : 25 < T)
variable (h5 : T < 35)

theorem krakozyabr_count : T = 32 := by
  sorry

end krakozyabr_count_l167_167121


namespace percentage_difference_l167_167841

theorem percentage_difference :
  ((75 / 100 : ℝ) * 40 - (4 / 5 : ℝ) * 25) = 10 := 
by
  sorry

end percentage_difference_l167_167841


namespace dragon_rope_problem_l167_167701

theorem dragon_rope_problem :
  ∃ (p q r : ℕ), 
  (r ∣ p - 30) ∧ 
  (r \ ∣ q \- 900) ∧
  (prime r) ∧
  (30 : ℝ)^2 - (6: ℝ)^2 = (10: ℝ)^2 + ((p - √q) / r) ∧
  pq + r = 993 :=
sorry

end dragon_rope_problem_l167_167701


namespace no_parallel_lines_l1_l2_l167_167418

theorem no_parallel_lines_l1_l2 (m: ℝ):
  ¬ ∃ m: ℝ, ∀ x y: ℝ,
  (x + (1 + m) * y + (m - 2) = 0 ∧ m * x + 2 * y + 6 = 0) →
  (-1 / (1 + m) = -m / 2) :=
begin
  sorry
end

end no_parallel_lines_l1_l2_l167_167418


namespace triangle_inequality_l167_167099

theorem triangle_inequality 
  (A B C L K : Point) 
  (hABC : Triangle A B C)
  (hBL_bisector : AngleBisector B L A C)
  (hLK_eq_AB : Length L K = Length A B)
  (hAK_parallel_BC : Parallel (Segment A K) (Segment B C)) :
  Length A B > Length B C :=
sorry

end triangle_inequality_l167_167099


namespace smallest_digit_never_in_units_place_of_divisible_by_5_l167_167660

theorem smallest_digit_never_in_units_place_of_divisible_by_5 : 
    (∃ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ d ≠ 0 ∧ d ≠ 5 ∧ (∀ n, n % 10 ≠ d)) → 
    d = 1 := 
by
  sorry

end smallest_digit_never_in_units_place_of_divisible_by_5_l167_167660


namespace baker_sales_difference_l167_167267

/-!
  Prove that the difference in dollars between the baker's daily average sales and total sales for today is 48 dollars.
-/

theorem baker_sales_difference :
  let price_pastry := 2
  let price_bread := 4
  let avg_pastries := 20
  let avg_bread := 10
  let today_pastries := 14
  let today_bread := 25
  let daily_avg_sales := avg_pastries * price_pastry + avg_bread * price_bread
  let today_sales := today_pastries * price_pastry + today_bread * price_bread
  daily_avg_sales - today_sales = 48 :=
sorry

end baker_sales_difference_l167_167267


namespace midpoint_BC_l167_167025

variables {A B C H N M : Point}
variable (P Q R : Triangle)
variable (O : Circle)

def is_orthocenter (H : Point) (Δ : Triangle) : Prop := sorry
def is_circumcircle (O : Circle) (Δ : Triangle) : Prop := sorry
def diameter_circle_intersects (cir1 cir2 : Circle) (A H : Point) : Point := sorry
def line_intersection (P Q : Point) (R : Line) : Point := sorry
def is_midpoint (M B C : Point) : Prop := sorry

theorem midpoint_BC
  (h1 : P = Triangle.mk A B C)
  (h2 : is_orthocenter H P)
  (h3 : is_circumcircle O P)
  (N_inters : N = diameter_circle_intersects (circle_diameter A H) O A H)
  (M_inters : M = line_intersection N H (line BC)) :
  is_midpoint M B C :=
sorry

end midpoint_BC_l167_167025


namespace find_y_l167_167080

-- Defining the changes percentage
def first_week_increase := 0.30
def second_week_decrease := 0.10
def third_week_increase := 0.40
def final_percentage_decrease := 0.05

-- Initial price
noncomputable def P0 := 100

-- Price after each change
noncomputable def P1 := P0 * (1 + first_week_increase)
noncomputable def P2 := P1 * (1 - second_week_decrease)
noncomputable def P3 := P2 * (1 + third_week_increase)
noncomputable def P4 := P3 * (1 - y / 100)

-- The final price condition
def final_condition : Prop := P4 = P0 * (1 - final_percentage_decrease)

-- The unknown percentage decrease
variable (y : ℝ)

theorem find_y : final_condition → y = 42 :=
by
  sorry

end find_y_l167_167080


namespace problem_part1_problem_part2_l167_167477

variable {x y : ℝ}

theorem problem_part1 
  (h1 : (x + y)^2 = 4)
  (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
sorry

theorem problem_part2 
  (h1 : (x + y)^2 = 4)
  (h2 : x * y = -1) :
  x - y = 2 * sqrt 2 :=
sorry

end problem_part1_problem_part2_l167_167477


namespace area_ratio_l167_167289

open Real

-- Define the regular dodecagon and its properties
def is_regular_dodecagon (vertices : Fin 12 → ℝ × ℝ) : Prop :=
  ∀ i, dist (vertices i) (vertices (Fin.mod 12 (i + 1))) = dist (vertices 0) (vertices 1) ∧
       is_regular_polygon vertices 12

-- Define the quadrilateral formed by vertices A, C, E, G
def quadrilateral_aceg (vertices : Fin 12 → ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  (vertices 0, vertices 2, vertices 4, vertices 6)

-- Define the areas m and n
noncomputable def area_of_quadrilateral (quad : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry
noncomputable def area_of_dodecagon (vertices : Fin 12 → ℝ × ℝ) : ℝ := sorry

-- Define the problem statement
theorem area_ratio (vertices : Fin 12 → ℝ × ℝ)
  (h : is_regular_dodecagon vertices) :
  let m := area_of_quadrilateral (quadrilateral_aceg vertices),
      n := area_of_dodecagon vertices in
  m / n = 1 / (3 * sqrt 3) := sorry

end area_ratio_l167_167289


namespace a_5_value_l167_167041

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (h : ∀ n : ℕ, S n = n * (2 * n + 1))

theorem a_5_value (h : ∀ n : ℕ, S n = n * (2 * n + 1)) : 
  (S 5) - (S 4) = 19 :=
by
  -- Substituted value below for clarity but it should be computed as in the problem
  calc
    S 5 = 5 * (2 * 5 + 1) := h 5
    _ = 55             := by rfl
    S 4 = 4 * (2 * 4 + 1) := h 4
    _ = 36             := by rfl
    55 - 36 = 19       := by rfl
  sorry

end a_5_value_l167_167041


namespace krakozyabrs_proof_l167_167112

-- Defining necessary variables and conditions
variable {n : ℕ}

-- Conditions given in the problem
def condition1 (H : ℕ) : Prop := 0.2 * H = n
def condition2 (W : ℕ) : Prop := 0.25 * W = n

-- Definition using the principle of inclusion-exclusion
def total_krakozyabrs (H W : ℕ) : ℕ := H + W - n

-- Statements reflecting the conditions and the final conclusion
theorem krakozyabrs_proof (H W : ℕ) (n : ℕ) : condition1 H → condition2 W → (25 < total_krakozyabrs H W) ∧ (total_krakozyabrs H W < 35) → total_krakozyabrs H W = 32 :=
by
  -- We skip the proof here
  sorry

end krakozyabrs_proof_l167_167112


namespace ratio_volumes_equal_ratio_areas_l167_167926

-- Defining necessary variables and functions
variables (R : ℝ) (S_sphere S_cone V_sphere V_cone : ℝ)

-- Conditions
def surface_area_sphere : Prop := S_sphere = 4 * Real.pi * R^2
def volume_sphere : Prop := V_sphere = (4 / 3) * Real.pi * R^3
def volume_polyhedron : Prop := V_cone = (S_cone * R) / 3

-- Theorem statement
theorem ratio_volumes_equal_ratio_areas
  (h1 : surface_area_sphere R S_sphere)
  (h2 : volume_sphere R V_sphere)
  (h3 : volume_polyhedron R S_cone V_cone)
  : (V_sphere / V_cone) = (S_sphere / S_cone) :=
sorry

end ratio_volumes_equal_ratio_areas_l167_167926


namespace prove_sum_series_l167_167331

noncomputable def infinite_sum_series : ℝ :=
  ∑' (j : ℕ) (k : ℕ), 2 ^ (- (3 * k + 2 * j + (k + j) ^ 2))

theorem prove_sum_series :
  infinite_sum_series = 4 / 3 :=
  sorry

end prove_sum_series_l167_167331


namespace maximize_volume_l167_167715

-- Define the volume of the pyramid
def pyramid_volume (a : ℝ) (α : ℝ) : ℝ :=
  let b := 2 * a * Real.sin(α / 2)
  let base_area := (Real.sqrt 3 / 4) * b^2
  let h := a * Real.sqrt((3 - 4 * (Real.sin (α / 2))^2) / 3)
  1 / 3 * base_area * h

-- The theorem statement
theorem maximize_volume (a : ℝ) (α : ℝ) : 
  (α = 90 * (Real.pi / 180)) → 
  (pyramid_volume a α = a^3 / 6) := 
  by 
  sorry

end maximize_volume_l167_167715


namespace problem_one_l167_167691

def S_n (n : Nat) : Nat := 
  List.foldl (fun acc x => acc * 10 + 2) 0 (List.replicate n 2)

theorem problem_one : ∃ n ∈ Finset.range 2011, S_n n % 2011 = 0 := 
  sorry

end problem_one_l167_167691


namespace number_of_valid_points_l167_167840

open Real

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the tangent line at a point (x0, x0^2)
def tangent_line (x0 : ℝ) (x : ℝ) : ℝ := 2 * x0 * x - x0^2

-- The set of valid points on the parabola y = x^2 whose tangents meet the specified condition
def valid_points : set ℝ :=
  {x0 | ∃ y0 : ℝ, y0 = parabola x0 ∧
                  ∃ x_int y_int : ℤ, 
                  x_int ≤ 2020 ∧ -2020 ≤ x_int ∧ 
                  y_int ≤ 2020 ∧ -2020 ≤ y_int ∧ 
                  2 * x0 * x_int = x0^2 ∧ y_int = -x0^2 ∧
                  x0 ≠ 0 ∧ x0 / 2 ∈ ℤ}

-- The goal is to prove that the cardinality of valid_points is 44.
theorem number_of_valid_points : ∃ n : ℤ, n = 44 ∧ n = 2 * ↑((set.filter (λ x0, x0 ≠ 0 ∧ x0 / 2 ∈ ℤ ∧ -44 ≤ x0 ∧ x0 ≤ 44) Icc).card) :=
sorry

end number_of_valid_points_l167_167840


namespace find_m_l167_167428

noncomputable def f : ℝ → ℝ := λ x, 2^x - 3

theorem find_m (m : ℝ) (h : f (m + 1) = 5) : m = 2 :=
by {
  sorry
}

end find_m_l167_167428


namespace ducks_at_North_Pond_l167_167587

theorem ducks_at_North_Pond :
  (∀ (ducks_Lake_Michigan : ℕ), ducks_Lake_Michigan = 100 → (6 + 2 * ducks_Lake_Michigan) = 206) :=
by
  intros ducks_Lake_Michigan hL
  rw [hL]
  norm_num
  rfl

end ducks_at_North_Pond_l167_167587


namespace sin_cos_sixth_power_l167_167901

theorem sin_cos_sixth_power (θ : ℝ) 
  (h : Real.sin (3 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11 / 12 :=
  sorry

end sin_cos_sixth_power_l167_167901


namespace sin_x_plus_pi_l167_167787

theorem sin_x_plus_pi (x : ℝ) (h1 : x ∈ Ioo (-π / 2) 0) (h2 : Real.tan x = -4 / 3) :
  Real.sin (x + π) = 4 / 5 := 
sorry

end sin_x_plus_pi_l167_167787


namespace max_gcd_13n_plus_4_8n_plus_3_l167_167314

theorem max_gcd_13n_plus_4_8n_plus_3 : ∃ n : ℕ, n > 0 ∧ Int.gcd (13 * n + 4) (8 * n + 3) = 11 := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l167_167314


namespace area_midpoint_quadrilateral_l167_167403

theorem area_midpoint_quadrilateral (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : ∀ quadrilateral with diagonals AC and BD, intersecting perpendicularly, 
    and having lengths AC = a and BD = b) : 
    area_midpoint_quadrilateral == (1 / 4) * a * b := 
sorry

end area_midpoint_quadrilateral_l167_167403


namespace find_M_remainder_l167_167000

def h (n : ℕ) : ℕ :=
  (n.to_digits 5).sum

def j (n : ℕ) : ℕ :=
  (h n).to_digits 7 .sum

def satisfies_condition (n : ℕ) : Prop :=
  j n > 5

theorem find_M_remainder :
  let M := 86 in
  satisfies_condition M ∧ (M % 1000 = 86) :=
begin
  sorry
end

end find_M_remainder_l167_167000


namespace carpet_dimensions_l167_167300
open Real

theorem carpet_dimensions (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : ∃ k: ℝ, y = k * x)
  (h4 : ∃ α β: ℝ, α + k * β = 50 ∧ k * α + β = 55)
  (h5 : ∃ γ δ: ℝ, γ + k * δ = 38 ∧ k * γ + δ = 55) :
  x = 25 ∧ y = 50 :=
by sorry

end carpet_dimensions_l167_167300


namespace question_l167_167616

def gcd (a b : ℕ) : ℕ :=
if b = 0 then a else gcd b (a % b)

def relatively_prime (p q : ℕ) : Prop := gcd p q = 1

noncomputable def a : ℝ := sorry
noncomputable def p : ℕ := sorry
noncomputable def q : ℕ := sorry

def floor (x : ℝ) : ℤ := Int.floor x
def frac (x : ℝ) : ℝ := x - floor x

def condition (a : ℝ) (x : ℝ) : Prop :=
floor x * frac x = a * x^2

theorem question {p q : ℕ} (hpq : relatively_prime p q)
  (h : ∀ x, condition (p / q : ℝ) x → sum (filter (condition (p / q : ℝ)) (⋃ i, [i, i + 1))) = 500) :
  p + 3 * q = 2914 :=
sorry

end question_l167_167616


namespace solve_system_l167_167934

noncomputable def is_solution (x y : ℝ) : Prop :=
  let eq1 := ((- x^7 / y) ^ Real.log (-y)) = x ^ (2 * Real.log (x * y^2))
  let eq2 := y^2 + 2 * x * y - 3 * x^2 + 12 * x + 4 * y = 0
  eq1 ∧ eq2

theorem solve_system : 
  { x y : ℝ // is_solution x y } = 
  { (2, -2), (3, -9), 
    ( (Real.sqrt 17 - 1) / 2, (Real.sqrt 17 - 9) / 2) } :=
by {
  sorry
}

end solve_system_l167_167934


namespace proof_problem_l167_167100

open Real

noncomputable def triangle := Type

structure Triangle (α : Type) where
  a b c : α
  A B C : Real

variables {a b c : ℝ} {A B C : Real}

def conditions (T : Triangle Real) :=
  (a = T.a) ∧ 
  (b = T.b) ∧
  (c = T.c) ∧
  (A = T.A) ∧
  (B = T.B) ∧
  (C = T.C) ∧
  A < π/2 ∧
  3 * b = 5 * a * sin B

theorem proof_problem (T : Triangle Real) (h : conditions T) :
  (sin A)^2 + (cos ((B + C) / 2))^2 = 53/50 ∧ 
  (a = sqrt 2) → 
  (area T = 3/2) → 
  b = sqrt 5 ∧ c = sqrt 5
sorry

end proof_problem_l167_167100


namespace simplify_fraction_and_rationalize_l167_167169
    
theorem simplify_fraction_and_rationalize :
  (sqrt 6 / sqrt 10) * (sqrt 5 / sqrt 15) * (sqrt 8 / sqrt 14) = 2 * sqrt 7 / 7 := by
sorry

end simplify_fraction_and_rationalize_l167_167169


namespace at_least_one_basketball_selected_l167_167779

theorem at_least_one_basketball_selected (balls : Finset ℕ) (basketballs : Finset ℕ) (volleyballs : Finset ℕ) :
  basketballs.card = 6 → volleyballs.card = 2 → balls ⊆ (basketballs ∪ volleyballs) →
  balls.card = 3 → ∃ b ∈ balls, b ∈ basketballs :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end at_least_one_basketball_selected_l167_167779


namespace log2_denominator_of_tournament_probability_l167_167210

noncomputable def tournament_outcome_prob : ℝ :=
  (Nat.factorial 35 : ℝ) / (2 ^ (Nat.choose 35 2) : ℝ)

theorem log2_denominator_of_tournament_probability :
  ∃ (m n : ℕ), Nat.coprime m n ∧ (m : ℝ) / (n : ℝ) = tournament_outcome_prob ∧ Real.log2 (n : ℝ) = 564 :=
sorry

end log2_denominator_of_tournament_probability_l167_167210


namespace sum_first_4_terms_of_arithmetic_sequence_eq_8_l167_167040

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 1 - a 0)

def S4 (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_first_4_terms_of_arithmetic_sequence_eq_8
  (a : ℕ → ℤ) 
  (h_seq : arithmetic_seq a) 
  (h_a2 : a 1 = 1) 
  (h_a3 : a 2 = 3) :
  S4 a = 8 :=
by
  sorry

end sum_first_4_terms_of_arithmetic_sequence_eq_8_l167_167040


namespace jonathan_should_start_on_tuesday_l167_167127

def day_of_week : Type := ℕ

def redeem_day (initial_day : day_of_week) (interval : ℕ) (n : ℕ) : day_of_week :=
  (initial_day + n * interval) % 7

def valid_redeem_schedule (initial_day : day_of_week) (interval : ℕ) (vouchers : ℕ) : Prop :=
  ∀ n, n < vouchers → redeem_day initial_day interval n ≠ 6

theorem jonathan_should_start_on_tuesday :
  valid_redeem_schedule 2 12 7 :=
by
  sorry

end jonathan_should_start_on_tuesday_l167_167127


namespace find_y_l167_167073

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 8) : y = 1 :=
by
  sorry

end find_y_l167_167073


namespace fraction_day_crew_loaded_boxes_l167_167320

-- Define the necessary variables and conditions
variables (D W : ℕ)
variables (boxes_loaded_by_day_worker : ℕ)
variables (workers_day : ℕ)
variables (workers_night : ℕ)

-- Define the conditions given in the problem
def boxes_loaded_by_night_worker := (3 / 4 : ℚ) * boxes_loaded_by_day_worker
def workers_night := (1 / 2 : ℚ) * workers_day

-- Define the assertion for the fraction of all the boxes loaded by the day crew as 8/11
theorem fraction_day_crew_loaded_boxes:
  ∀ (D W : ℕ), (D * W) / ((D * W) + ((3 / 4 : ℚ) * D * ((1 / 2 : ℚ) * W))) = (8 / 11 : ℚ) :=
begin
  intros,
  -- proof omitted
  sorry
end

end fraction_day_crew_loaded_boxes_l167_167320


namespace sun_radius_scientific_notation_l167_167621

theorem sun_radius_scientific_notation : 
  ∀ (r : ℝ), r = 696000000 → ∃ a n, (1 ≤ |a| ∧ |a| < 10) ∧ r = a * 10^n := by
  intros r hr
  use [6.96, 8]
  split
  { split
    { exact le_of_lt (abs_pos_of_pos (by norm_num))
    { norm_num [abs_of_pos (by norm_num)] } }
  { rw hr
    norm_num }
  sorry

end sun_radius_scientific_notation_l167_167621


namespace correct_statements_count_l167_167556

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

theorem correct_statements_count (x : ℝ) :
  let C := graph f in
  let stmt1 := ∀ x, f(-(x - 11 * Real.pi / 12)) = f(x + 11 * Real.pi / 12) in
  let stmt2 := ∀ x, x ∈ Ioo (5 * Real.pi / 12) (11 * Real.pi / 12) -> (f' x) < 0 in
  let stmt3 := ∀ x, f (x - Real.pi / 3) = 3 * Real.sin (2 * x - 2 * Real.pi / 3) in
  (stmt1 ∧ stmt2 ∧ ¬stmt3) ↔ (2 = 2) :=
by sorry

end correct_statements_count_l167_167556


namespace find_a_plus_b_l167_167431

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * Real.log x

theorem find_a_plus_b (a b : ℝ) :
  (∃ x : ℝ, x = 1 ∧ f a b x = 1 / 2 ∧ (deriv (f a b)) 1 = 0) →
  a + b = -1/2 :=
by
  sorry

end find_a_plus_b_l167_167431


namespace purely_imaginary_z_real_z_second_quadrant_z_l167_167008

-- Problem 1: m that makes z a purely imaginary number
theorem purely_imaginary_z (m : ℝ) : 
  (real.log (m^2 - 2*m - 2) = 0 ∧ m^2 + 3*m + 2 ≠ 0) ↔ m = 3 :=
sorry

-- Problem 2: m that makes z a real number
theorem real_z (m : ℝ) : 
  (m^2 + 3*m + 2 = 0) ↔ m = -1 ∨ m = -2 :=
sorry

-- Problem 3: m range for the point in the second quadrant
theorem second_quadrant_z (m : ℝ) : 
  (real.log (m^2 - 2*m - 2) < 0 ∧ m^2 + 3*m + 2 > 0) ↔ -1 < m ∧ m < 3 :=
sorry

end purely_imaginary_z_real_z_second_quadrant_z_l167_167008


namespace length_of_first_train_is_270_l167_167697

/-- 
Given:
1. Speed of the first train = 120 kmph
2. Speed of the second train = 80 kmph
3. Time to cross each other = 9 seconds
4. Length of the second train = 230.04 meters
  
Prove that the length of the first train is 270 meters.
-/
theorem length_of_first_train_is_270
  (speed_first_train : ℝ := 120)
  (speed_second_train : ℝ := 80)
  (time_to_cross : ℝ := 9)
  (length_second_train : ℝ := 230.04)
  (conversion_factor : ℝ := 1000/3600) :
  (length_second_train + (speed_first_train + speed_second_train) * conversion_factor * time_to_cross - length_second_train) = 270 :=
by
  sorry

end length_of_first_train_is_270_l167_167697


namespace Brianna_books_difference_correct_l167_167324

noncomputable def Brianna_books_difference : ℕ :=
let books_per_month := 2 in
let months_in_year := 12 in
let given_books := 6 in
let bought_books := 8 in
let old_books_reread := 4 in
let total_needed_books := books_per_month * months_in_year in
let total_new_books := given_books + bought_books in
let total_books_with_old := total_new_books + old_books_reread in
let books_to_borrow := total_needed_books - total_books_with_old in
bought_books - books_to_borrow

theorem Brianna_books_difference_correct :
  Brianna_books_difference = 2 :=
by {
  // Proof steps here
  sorry
}

end Brianna_books_difference_correct_l167_167324


namespace hacker_guaranteed_hack_671_computers_l167_167986

theorem hacker_guaranteed_hack_671_computers
  (n : ℕ)
  (h_n : n = 2008)
  (connected : Graph.connected (graph : Type))
  (no_common_vertex_in_cycles : ∀ c1 c2 : set (graph.verts), (is_cycle c1) → (is_cycle c2) → (c1 ∩ c2 = ∅)) :
  ∃ m : ℕ, m = 671 ∧ (hacker_can_guarantee_hack m (graph : Type) (game_rules : Type)) :=
by
  sorry

end hacker_guaranteed_hack_671_computers_l167_167986


namespace inflation_two_years_real_rate_of_return_l167_167666

-- Proof Problem for Question 1
theorem inflation_two_years :
  ((1 + 0.015)^2 - 1) * 100 = 3.0225 :=
by
  sorry

-- Proof Problem for Question 2
theorem real_rate_of_return :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by
  sorry

end inflation_two_years_real_rate_of_return_l167_167666


namespace normal_dist_P_xi_l167_167790

variable (ξ : ℝ → ℝ) (μ σ : ℝ)
-- Condition for normal distribution with mean μ and variance σ^2
axiom normal_distribution (μ : ℝ) (σ : ℝ) : Prop

-- Condition given in the problem
axiom P_condition (H : normal_distribution μ σ): 
  (∀ x : ℝ, (P (ξ < 1) = P (ξ > 3)))

-- The theorem we want to prove
theorem normal_dist_P_xi (H : normal_distribution μ σ) (H_cond : P_condition H) : 
  (P (ξ > 2) = 1/2) := 
  sorry

end normal_dist_P_xi_l167_167790


namespace tan_330_eq_sec_330_eq_l167_167727

noncomputable def tan_deg (deg : ℝ) := Real.tan (deg * Real.pi / 180)
noncomputable def sec_deg (deg : ℝ) := 1 / Real.cos (deg * Real.pi / 180)

def cos_30 := Real.cos (30 * Real.pi / 180)
def tan_30 := Real.tan (30 * Real.pi / 180)

theorem tan_330_eq : tan_deg 330 = -1 / Real.sqrt 3 :=
by
  have h1 : tan_deg 330 = tan_deg (360 - 30), by sorry
  have h2 : tan_deg (360 - 30) = -tan_30, by sorry
  have h3 : tan_30 = 1 / Real.sqrt 3, by sorry
  exact Eq.trans h1 (Eq.trans h2 (Eq.symm h3))

theorem sec_330_eq : sec_deg 330 = 2 * Real.sqrt 3 / 3 :=
by
  have h1 : sec_deg 330 = 1 / Real.cos (330 * Real.pi / 180), by sorry
  have h2 : Real.cos (330 * Real.pi / 180) = Real.cos (360 * Real.pi / 180 - 30 * Real.pi / 180), by sorry
  have h3 : Real.cos (360 * Real.pi / 180 - 30 * Real.pi / 180) = cos_30, by sorry
  have h4 : cos_30 = Real.sqrt 3 / 2, by sorry
  exact Eq.trans h1 (Eq.trans (Eq.trans h2 h3) (Eq.symm (div_eq_iff_mul_eq.mpr h4)))

end tan_330_eq_sec_330_eq_l167_167727


namespace marks_lost_wrong_answer_is_one_l167_167504

variables (marks_lost_per_wrong_answer : ℕ)
variables (correct_answers wrong_answers total_questions total_marks : ℕ)

-- Given conditions
def condition_correct_answers := correct_answers = 40
def condition_total_questions := total_questions = 60
def condition_total_marks := total_marks = 140
def condition_score_per_correct := 4

-- Define calculations based on conditions
def marks_from_correct := correct_answers * condition_score_per_correct
def num_wrong_answers := total_questions - correct_answers
def marks_lost := wrong_answers * marks_lost_per_wrong_answer
def total_score := marks_from_correct - marks_lost

-- Proof problem statement
theorem marks_lost_wrong_answer_is_one :
  condition_correct_answers →
  condition_total_questions →
  condition_total_marks →
  total_score = total_marks →
  marks_lost_per_wrong_answer = 1 :=
by
  intros h1 h2 h3 h4
  -- mark this part with sorry since the proof is not required
  sorry

end marks_lost_wrong_answer_is_one_l167_167504


namespace path_shorter_factor_l167_167571

-- Declare variables
variables (x y z : ℝ)

-- Define conditions as hypotheses
def condition1 := x = 3 * (y + z)
def condition2 := 4 * y = z + x

-- State the proof statement
theorem path_shorter_factor (condition1 : x = 3 * (y + z)) (condition2 : 4 * y = z + x) :
  (4 * y) / z = 19 :=
sorry

end path_shorter_factor_l167_167571


namespace shortest_minor_arc_line_equation_l167_167799

noncomputable def pointM : (ℝ × ℝ) := (1, -2)
noncomputable def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

theorem shortest_minor_arc_line_equation :
  (∀ x y : ℝ, (x + 2 * y + 3 = 0) ↔ 
  ((x = 1 ∧ y = -2) ∨ ∃ (k_l : ℝ), (k_l * (2) = -1) ∧ (y + 2 = -k_l * (x - 1)))) :=
sorry

end shortest_minor_arc_line_equation_l167_167799


namespace animal_population_l167_167975

def total_population (L P E : ℕ) : ℕ :=
L + P + E

theorem animal_population 
    (L P E : ℕ) 
    (h1 : L = 2 * P) 
    (h2 : E = (L + P) / 2) 
    (h3 : L = 200) : 
  total_population L P E = 450 := 
  by 
    sorry

end animal_population_l167_167975


namespace socks_cost_l167_167009

noncomputable def pants_cost : ℕ := 20
noncomputable def shirt_cost : ℕ := 2 * pants_cost
noncomputable def tie_cost : ℕ := shirt_cost / 5

theorem socks_cost (S : ℕ) (total_spending : ℕ) (h : total_spending = 355) : S = 3 :=
by
  have total_cost_one_uniform : ℕ := pants_cost + shirt_cost + tie_cost + S
  have total_cost_five_uniforms := 5 * total_cost_one_uniform
  have h1 : total_cost_five_uniforms = 340 + 5 * S := by
    simp [pants_cost, shirt_cost, tie_cost, total_cost_one_uniform]
  have h2 : total_spending = 340 + 5 * S := by
    simp [h, total_cost_five_uniforms]
  have h3 : S = 3 := by
    linarith
  exact h3

end socks_cost_l167_167009


namespace triangle_sides_proof_sin_B_plus_pi_over_6_proof_l167_167854

noncomputable def triangle_side_a_c (A : ℝ) (b : ℝ) (area : ℝ) (a c : ℝ) :=
  A = 2 * π / 3 ∧ b = 1 ∧ area = sqrt 3 ∧ c = 4 ∧ a = sqrt 21

noncomputable def sin_B_pi_over_6 (A : ℝ) (b : ℝ) (area : ℝ) (sin_B_plus_pi_over_6 : ℝ) :=
  A = 2 * π / 3 ∧ b = 1 ∧ area = sqrt 3 ∧ sin_B_plus_pi_over_6 = sqrt 21 / 7

-- Using the defined conditions and values to create the proof problem:

theorem triangle_sides_proof :
  ∃ a c, triangle_side_a_c (2 * π / 3) 1 (sqrt 3) a c :=
  sorry

theorem sin_B_plus_pi_over_6_proof :
  ∃ sin_B_plus_pi_over_6, sin_B_pi_over_6 (2 * π / 3) 1 (sqrt 3) sin_B_plus_pi_over_6 :=
  sorry

end triangle_sides_proof_sin_B_plus_pi_over_6_proof_l167_167854


namespace animal_population_l167_167976

theorem animal_population
  (number_of_lions : ℕ)
  (number_of_leopards : ℕ)
  (number_of_elephants : ℕ)
  (h1 : number_of_lions = 200)
  (h2 : number_of_lions = 2 * number_of_leopards)
  (h3 : number_of_elephants = (number_of_lions + number_of_leopards) / 2) :
  number_of_lions + number_of_leopards + number_of_elephants = 450 :=
sorry

end animal_population_l167_167976


namespace exists_infinite_B_with_property_l167_167559

-- Definition of the sequence A
def seqA (n : ℕ) : ℤ := 5 * n - 2

-- Definition of the sequence B with its general form
def seqB (k : ℕ) (d : ℤ) : ℤ := k * d + 7 - d

-- The proof problem statement
theorem exists_infinite_B_with_property :
  ∃ (B : ℕ → ℤ) (d : ℤ), B 1 = 7 ∧ 
  (∀ k, k > 1 → B k = B (k - 1) + d) ∧
  (∀ n : ℕ, ∃ (k : ℕ), seqB k d = seqA n) :=
sorry

end exists_infinite_B_with_property_l167_167559


namespace task1_correct_task2_task3_l167_167409

-- Definition of the sum of products for a given n x n table
def sum_of_products (n : ℕ) (A : Fin n → Fin n → ℤ) : ℤ :=
  let x (i : Fin n) : ℤ := ∏ j, A i j
  let y (j : Fin n) : ℤ := ∏ i, A i j
  ∑ i, x i + ∑ j, y j

-- 1. Prove the sum of products S for the given 4x4 table is 0
def task1 : ℤ :=
  let A : Fin 4 → Fin 4 → ℤ :=
    fun i j =>
      match i, j with
      | 0, 0 => 1 | 0, 1 => 1 | 0, 2 => -1 | 0, 3 => -1
      | 1, 0 => 1 | 1, 1 => -1 | 1, 2 => 1 | 1, 3 => 1
      | 2, 0 => 1 | 2, 1 => -1 | 2, 2 => -1 | 2, 3 => 1
      | 3, 0 => -1 | 3, 1 => -1 | 3, 2 => 1 | 3, 3 => 1
      | _, _ => 0 -- Should never hit this case
  sum_of_products 4 A

theorem task1_correct : task1 = 0 := by
  sorry

-- 2. Prove there does not exist a 3x3 table such that S = 0
theorem task2 : ¬ ∃ (A : Fin 3 → Fin 3 → ℤ), sum_of_products 3 A = 0 := by
  sorry

-- 3. Prove the possible values of S for n = 10
theorem task3 : ∀ (A : Fin 10 → Fin 10 → ℤ), sum_of_products 10 A ∈ {-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20} := by
  sorry

end task1_correct_task2_task3_l167_167409


namespace range_of_x_l167_167749

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_x (f_even : ∀ x : ℝ, f x = f (-x))
                   (f_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
                   (f_at_one_third : f (1/3) = 0) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | (0 < x ∧ x < 1/2) ∨ 2 < x} :=
sorry

end range_of_x_l167_167749


namespace arith_seq_a_minus_2_pow_n_seq_b_condition_l167_167511

-- Definitions and conditions
def seq_a (n : ℕ) : ℕ :=
  if n = 1 then 2 else seq_a(n - 1) + 2^(n - 1) + 1

def seq_diff (n : ℕ) : ℕ :=
  2^n

def seq_b (n : ℕ) : ℕ :=
  if n = 1 then log 2 else log (seq_a n + 1 - n)

-- (1) Prove that the sequence {a_n - 2^n} is arithmetic with first term 0 and common difference 1
theorem arith_seq_a_minus_2_pow_n : ∀ n, seq_a n - seq_diff n = 1 * n :=
  sorry

-- (2) Prove the inequality: sum (λ i, 1 / (b_i * b_(i + 2))) < 3 / 4
theorem seq_b_condition (n : ℕ) (h : 0 < n) :
  let seq_b := λ i, log (seq_a i + 1 - i) in
  (finset.range n).sum (λ i, 1 / (seq_b i * seq_b (i + 2))) < 3 / 4 :=
  sorry

end arith_seq_a_minus_2_pow_n_seq_b_condition_l167_167511


namespace ducks_at_North_Pond_l167_167588

theorem ducks_at_North_Pond :
  (∀ (ducks_Lake_Michigan : ℕ), ducks_Lake_Michigan = 100 → (6 + 2 * ducks_Lake_Michigan) = 206) :=
by
  intros ducks_Lake_Michigan hL
  rw [hL]
  norm_num
  rfl

end ducks_at_North_Pond_l167_167588


namespace krakozyabrs_total_count_l167_167102

theorem krakozyabrs_total_count :
  (∃ (T : ℕ), 
  (∀ (H W n : ℕ),
    0.2 * H = n ∧ 0.25 * W = n ∧ H = 5 * n ∧ W = 4 * n ∧ T = H + W - n ∧ 25 < T ∧ T < 35) ∧ 
  T = 32) :=
by sorry

end krakozyabrs_total_count_l167_167102


namespace fifth_term_equals_31_l167_167739

-- Define the sequence of sums of consecutive powers of 2
def sequence_sum (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (λ i, 2^i)

-- State the theorem: The fifth term of the sequence equals 31
theorem fifth_term_equals_31 : sequence_sum 4 = 31 := by
  sorry

end fifth_term_equals_31_l167_167739


namespace ineq_abc_l167_167476

theorem ineq_abc (a b c : ℝ) (h1 : a ∈ set.Icc (-1 : ℝ) 1)
                              (h2 : b ∈ set.Icc (-1 : ℝ) 1)
                              (h3 : c ∈ set.Icc (-1 : ℝ) 1)
                              (h4 : a + b + c + a * b * c = 0) :
  a^2 + b^2 + c^2 ≥ 3 * (a + b + c) ∧ (a^2 + b^2 + c^2 = 3 * (a + b + c) ↔ (a = 1 ∧ b = 1 ∧ c = -1)) :=
sorry

end ineq_abc_l167_167476


namespace remainder_when_m_divided_by_1000_is_8_l167_167617

theorem remainder_when_m_divided_by_1000_is_8 :
  (∃ m n : ℕ, m > n ∧ (∀ i (h1 : 1 ≤ i ∧ i ≤ 12), 
  let a_i := i + 2 * (k_n i) + 1 } :=
  (∀ i, a i ≤ a (i+1) ∧ a 12 ≤ 2005) ∧ 
  binom m n = binom 1008 12))
  → m % 1000 = 8 := 
by
  sorry

end remainder_when_m_divided_by_1000_is_8_l167_167617


namespace theta_digit_l167_167225

theorem theta_digit (Θ : ℕ) (h : Θ ≠ 0) (h1 : 252 / Θ = 10 * 4 + Θ + Θ) : Θ = 5 :=
  sorry

end theta_digit_l167_167225


namespace combined_boys_girls_ratio_l167_167982

def num_students_Pascal : ℕ := 400
def ratio_boys_girls_Pascal : ℕ × ℕ := (3, 2)
def num_students_Fermat : ℕ := 600
def ratio_boys_girls_Fermat : ℕ × ℕ := (2, 3)

theorem combined_boys_girls_ratio :
  let boys_Pascal := (ratio_boys_girls_Pascal.1 * num_students_Pascal) / (ratio_boys_girls_Pascal.1 + ratio_boys_girls_Pascal.2),
      girls_Pascal := (ratio_boys_girls_Pascal.2 * num_students_Pascal) / (ratio_boys_girls_Pascal.1 + ratio_boys_girls_Pascal.2),
      boys_Fermat := (ratio_boys_girls_Fermat.1 * num_students_Fermat) / (ratio_boys_girls_Fermat.1 + ratio_boys_girls_Fermat.2),
      girls_Fermat := (ratio_boys_girls_Fermat.2 * num_students_Fermat) / (ratio_boys_girls_Fermat.1 + ratio_boys_girls_Fermat.2),
      total_boys := boys_Pascal + boys_Fermat,
      total_girls := girls_Pascal + girls_Fermat
  in (total_boys * 13) = (total_girls * 12) :=
by 
  sorry

end combined_boys_girls_ratio_l167_167982


namespace simplify_expression_l167_167930

noncomputable def π : ℝ := real.pi

theorem simplify_expression : sqrt ((π - 4)^2) + (π - 3)^(1 / 3) = 1 :=
by sorry

end simplify_expression_l167_167930


namespace jenny_ate_more_than_thrice_mike_l167_167123

theorem jenny_ate_more_than_thrice_mike :
  let mike_ate := 20
  let jenny_ate := 65
  jenny_ate - 3 * mike_ate = 5 :=
by
  let mike_ate := 20
  let jenny_ate := 65
  have : jenny_ate - 3 * mike_ate = 5 := by
    sorry
  exact this

end jenny_ate_more_than_thrice_mike_l167_167123


namespace find_budget_l167_167122

variable (B : ℝ)

-- Conditions provided
axiom cond1 : 0.30 * B = 300

theorem find_budget : B = 1000 :=
by
  -- Notes:
  -- The proof will go here.
  sorry

end find_budget_l167_167122


namespace smallest_possible_N_l167_167858

/--
In a circle, all natural numbers from 1 to \( N \) (where \( N \ge 2 \)) are written in some order.
For any pair of neighboring numbers, there is at least one digit that appears in the decimal representation of both numbers.
Prove that the smallest possible value of \( N \) is 29.
-/
theorem smallest_possible_N : ∃ (N : ℕ), (N = 29) ∧ ∀ (arrangement : list ℕ), 
  (∀ (i : ℕ), i < arrangement.length →
    let x := arrangement.nth_le i _;
    let y := arrangement.nth_le ((i + 1) % arrangement.length) _ in
    ∃ d : char, d ∈ x.to_digits ∧ d ∈ y.to_digits) ∧ 
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ N → n ∈ arrangement) :=
begin
  sorry
end

end smallest_possible_N_l167_167858


namespace average_age_of_9_students_l167_167183

theorem average_age_of_9_students
  (avg_20_students : ℝ)
  (n_20_students : ℕ)
  (avg_10_students : ℝ)
  (n_10_students : ℕ)
  (age_20th_student : ℝ)
  (total_age_20_students : ℝ := avg_20_students * n_20_students)
  (total_age_10_students : ℝ := avg_10_students * n_10_students)
  (total_age_9_students : ℝ := total_age_20_students - total_age_10_students - age_20th_student)
  (n_9_students : ℕ)
  (expected_avg_9_students : ℝ := total_age_9_students / n_9_students)
  (H1 : avg_20_students = 20)
  (H2 : n_20_students = 20)
  (H3 : avg_10_students = 24)
  (H4 : n_10_students = 10)
  (H5 : age_20th_student = 61)
  (H6 : n_9_students = 9) :
  expected_avg_9_students = 11 :=
sorry

end average_age_of_9_students_l167_167183


namespace largest_reflections_l167_167288

-- Define initial conditions
variables (A B D : Point)  -- Points in the plane
variable (n : ℕ) -- Number of reflections
variable (a : ℝ) -- Angle at reflection

-- Define reflection conditions
variable (AD CD : Line) -- Lines involved
variable (angleCDA : ℝ) -- Given angle

-- Lean statement of the equivalent proof problem
theorem largest_reflections (h_angle: angleCDA = 8)
                           (h_angle_unit: angleCDA * n ≤ 90):
  n ≤ 11 := 
  -- Given conditions above and problem's context
  sorry

end largest_reflections_l167_167288


namespace taxi_fare_A_to_B_l167_167972

-- Define the distance thresholds and fares
def fare_up_to_2_km : ℝ := 5
def fare_per_km_2_to_8 : ℝ := 1.6
def fare_per_km_above_8 : ℝ := 2.4
def distance_A_to_B : ℝ := 10

-- Define the total fare calculation based on the given distance
def total_fare (d : ℝ) : ℝ :=
  if d <= 2 then
    fare_up_to_2_km
  else if d <= 8 then
    fare_up_to_2_km + (d - 2) * fare_per_km_2_to_8
  else
    fare_up_to_2_km + 6 * fare_per_km_2_to_8 + (d - 8) * fare_per_km_above_8

-- Theorem stating the total fare from A to B is equal to 19.4 yuan
theorem taxi_fare_A_to_B : total_fare distance_A_to_B = 19.4 := by
  -- Proof omitted
  sorry

end taxi_fare_A_to_B_l167_167972


namespace inequality_sum_cube_div_pos_l167_167923

theorem inequality_sum_cube_div_pos 
  (α β : ℝ) (x : ℕ → ℝ) (n : ℕ) 
  (h_pos : ∀ i, 1 ≤ i → i ≤ n → 0 < x i)
  (h_sum : (∑ i in Finset.range n, x (i + 1)) = 1)
  (h_npos : 0 < n) :
  (∑ i in Finset.range n, x (i + 1) ^ 3 / (α * x (i + 1) + β * x ((i + 1) % n + 1))) 
  ≥ 1 / (n * (α + β)) :=
sorry

end inequality_sum_cube_div_pos_l167_167923


namespace line_intersects_y_axis_at_correct_point_l167_167321

theorem line_intersects_y_axis_at_correct_point :
  let x := 0
  let y := 3
  in 5 * y + 3 * x = 15 :=
by
  sorry

end line_intersects_y_axis_at_correct_point_l167_167321


namespace sum_of_values_of_m_l167_167490

-- Define the inequality conditions
def condition1 (x m : ℝ) : Prop := (x - m) / 2 ≥ 0
def condition2 (x : ℝ) : Prop := x + 3 < 3 * (x - 1)

-- Define the equation constraint for y
def fractional_equation (y m : ℝ) : Prop := (3 - y) / (2 - y) + m / (y - 2) = 3

-- Sum function for the values of m
def sum_of_m (m1 m2 m3 : ℝ) : ℝ := m1 + m2 + m3

-- Main theorem
theorem sum_of_values_of_m : sum_of_m 3 (-3) (-1) = -1 := 
by { sorry }

end sum_of_values_of_m_l167_167490


namespace perpendicular_distances_sum_l167_167334

noncomputable def sum_of_recip_distances (a b : ℝ) (n : ℕ) (c : ℝ) (dist : Fin n → ℝ) : ℝ :=
  ∑ i, 1 / dist i
  
theorem perpendicular_distances_sum 
  (a b : ℝ) (h : a > b ∧ b > 0) (n : ℕ) 
  (c : ℝ) (dist : Fin n → ℝ) 
  (h_c : c = Real.sqrt (a^2 - b^2)) 
  (h_dist : ∀ i : Fin n, 
    dist i = dist 0 * Real.cos ((2 * i.1 * Real.pi) / n)) :
  sum_of_recip_distances a b n c dist = n * c / b^2 := 
sorry

end perpendicular_distances_sum_l167_167334


namespace max_value_proof_l167_167945

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + 2 * x * y + 3 * y^2

theorem max_value_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 12) : 
  max_value x y = 24 + 12 * Real.sqrt 3 := 
sorry

end max_value_proof_l167_167945


namespace minimum_value_f_l167_167768

def f (x : ℝ) : ℝ := Real.sin x - x

theorem minimum_value_f : 
  ∀ x ∈ set.Icc 0 (Real.pi / 2), (f x) ≥ 1 - (Real.pi / 2) ∧ 
  ∃ x ∈ set.Icc 0 (Real.pi / 2), (f x) = 1 - (Real.pi / 2) :=
by
  sorry

end minimum_value_f_l167_167768


namespace sum_first_100_even_numbers_divisible_by_6_l167_167773

-- Define the sequence of even numbers divisible by 6 between 100 and 300 inclusive.
def even_numbers_divisible_by_6 (n : ℕ) : ℕ := 102 + n * 6

-- Define the sum of the first 100 even numbers divisible by 6.
def sum_even_numbers_divisible_by_6 (k : ℕ) : ℕ := k / 2 * (102 + (102 + (k - 1) * 6))

-- Define the problem statement as a theorem.
theorem sum_first_100_even_numbers_divisible_by_6 :
  sum_even_numbers_divisible_by_6 100 = 39900 :=
by
  sorry

end sum_first_100_even_numbers_divisible_by_6_l167_167773


namespace find_x_average_l167_167949

theorem find_x_average (x : ℚ) : 
  let S := (149 * (149 + 1)) / 2
  in (S + x) / 150 = 150 * x → x = 11175 / 22499 :=
by
  intro h
  have : S = 11175 := by norm_num
  rw this at h
  sorry

end find_x_average_l167_167949


namespace john_additional_tax_l167_167081

-- Define the old and new tax rates
def old_tax (income : ℕ) : ℕ :=
  if income ≤ 500000 then income * 20 / 100
  else if income ≤ 1000000 then 100000 + (income - 500000) * 25 / 100
  else 225000 + (income - 1000000) * 30 / 100

def new_tax (income : ℕ) : ℕ :=
  if income ≤ 500000 then income * 30 / 100
  else if income ≤ 1000000 then 150000 + (income - 500000) * 35 / 100
  else 325000 + (income - 1000000) * 40 / 100

-- Calculate the tax for rental income after deduction
def rental_income_tax (rental_income : ℕ) : ℕ :=
  let taxable_rental_income := rental_income - rental_income * 10 / 100
  taxable_rental_income * 40 / 100

-- Calculate the tax for investment income
def investment_income_tax (investment_income : ℕ) : ℕ :=
  investment_income * 25 / 100

-- Calculate the tax for self-employment income
def self_employment_income_tax (self_employment_income : ℕ) : ℕ :=
  self_employment_income * 15 / 100

-- Define the total additional tax John pays
def additional_tax_paid (old_main_income new_main_income rental_income investment_income self_employment_income : ℕ) : ℕ :=
  let old_tax_main := old_tax old_main_income
  let new_tax_main := new_tax new_main_income
  let rental_tax := rental_income_tax rental_income
  let investment_tax := investment_income_tax investment_income
  let self_employment_tax := self_employment_income_tax self_employment_income
  (new_tax_main - old_tax_main) + rental_tax + investment_tax + self_employment_tax

-- Prove John pays $352,250 more in taxes under the new system
theorem john_additional_tax (main_income_old main_income_new rental_income investment_income self_employment_income : ℕ) :
  main_income_old = 1000000 →
  main_income_new = 1500000 →
  rental_income = 100000 →
  investment_income = 50000 →
  self_employment_income = 25000 →
  additional_tax_paid main_income_old main_income_new rental_income investment_income self_employment_income = 352250 :=
by
  intros h_old h_new h_rental h_invest h_self
  rw [h_old, h_new, h_rental, h_invest, h_self]
  -- calculation steps are omitted
  sorry

end john_additional_tax_l167_167081


namespace sum_perpendiculars_in_regular_hexagon_l167_167256

open EuclideanGeometry

variables {P Q R A B C D E F O : Point}

-- Definition of regular hexagon and its properties
def is_regular_hexagon (A B C D E F : Point) : Prop :=
  is_regular_polygon 6 [A, B, C, D, E, F]

-- Perpendicular drops
def perpendicular_from_point_to_line (A P : Point) (l : Line) : Prop :=
  is_perpendicular (Line.through A P) l

-- Regular Hexagon and perpendicular properties
variables (h1 : is_regular_hexagon A B C D E F)
          (h2 : perpendicular_from_point_to_line A P (Line.through C D))
          (h3 : perpendicular_from_point_to_line A Q (Line.through E F))
          (h4 : perpendicular_from_point_to_line A R (Line.through B C))
          (h5 : center_of A B C D E F = O)
          (h6 : dist O P = 1)

theorem sum_perpendiculars_in_regular_hexagon (h1 h2 h3 h4 h5 h6) : 
  dist A P + dist A Q + dist A R = 3 * Real.sqrt 3 := 
sorry

end sum_perpendiculars_in_regular_hexagon_l167_167256


namespace smaller_square_area_l167_167167

theorem smaller_square_area (A B C D : Point) (S : Square) 
  (h_midpoints : A = midpoint S.side1 ∧ B = midpoint S.side2 ∧ C = midpoint S.side3 ∧ D = midpoint S.side4)
  (h_area : S.area = 60) : 
  smaller_square S A B C D).area = 30 := 
by
  sorry

end smaller_square_area_l167_167167


namespace complex_conjugate_in_fourth_quadrant_l167_167035

theorem complex_conjugate_in_fourth_quadrant (z : ℂ) (h : (z + 2) * (1 - I) = 6 - 4 * I) :
  z.conj.im < 0 ∧ z.conj.re > 0 :=
sorry

end complex_conjugate_in_fourth_quadrant_l167_167035


namespace distribute_tourists_l167_167212

theorem distribute_tourists (guides tourists : ℕ) (hguides : guides = 3) (htourists : tourists = 8) :
  ∃ k, k = 5796 := by
  sorry

end distribute_tourists_l167_167212


namespace sally_bought_20_cards_l167_167929

variable (initial_cards : ℕ) (dan_cards : ℕ) (total_cards : ℕ) (bought_cards : ℕ)

def sally_bought_cards (initial_cards dan_cards total_cards bought_cards : ℕ) : Prop :=
  initial_cards = 27 ∧
  dan_cards = 41 ∧
  total_cards = 88 ∧
  bought_cards = total_cards - (initial_cards + dan_cards)

theorem sally_bought_20_cards : 
  sally_bought_cards 27 41 88 20 :=
by
  unfold sally_bought_cards
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact rfl }
  sorry

end sally_bought_20_cards_l167_167929


namespace gcd_pair_sum_ge_prime_l167_167579

theorem gcd_pair_sum_ge_prime
  (n : ℕ)
  (h_prime: Prime (2*n - 1))
  (a : Fin n → ℕ)
  (h_distinct: ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∃ i j : Fin n, i ≠ j ∧ (a i + a j) / Nat.gcd (a i) (a j) ≥ 2*n - 1 := sorry

end gcd_pair_sum_ge_prime_l167_167579


namespace inflation_over_two_years_real_yield_deposit_second_year_l167_167663

-- Inflation problem setup and proof
theorem inflation_over_two_years :
  ((1 + 0.015) ^ 2 - 1) * 100 = 3.0225 :=
by sorry

-- Real yield problem setup and proof
theorem real_yield_deposit_second_year :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by sorry

end inflation_over_two_years_real_yield_deposit_second_year_l167_167663


namespace waiter_tables_l167_167711

theorem waiter_tables (initial_customers remaining_customers people_per_table : ℕ) (hc : initial_customers = 44) (hl : remaining_customers = initial_customers - 12) (ht : people_per_table = 8) :
  remaining_customers / people_per_table = 4 :=
by {
  rw [hc, hl, ht],
  exact rfl,
}

end waiter_tables_l167_167711


namespace max_gcd_13n_plus_4_8n_plus_3_l167_167317

theorem max_gcd_13n_plus_4_8n_plus_3 : 
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 7 ∧ ∀ k : ℕ, k = gcd (13 * n + 4) (8 * n + 3) → k ≤ d :=
by
  sorry

end max_gcd_13n_plus_4_8n_plus_3_l167_167317


namespace john_naps_per_week_l167_167889

theorem john_naps_per_week :
  ∃ (naps_per_week : ℕ), naps_per_week = 3 ∧
  (∀ (days : ℕ) (hours_in_70_days : ℕ) (weeks_per_70_days : ℕ),
    days = 70 → hours_in_70_days = 60 →
    weeks_per_70_days = 10 →
    (∃ (hours_per_week : ℕ), hours_per_week = 6 ∧
    ∀ (hours_per_nap : ℕ), hours_per_nap = 2 →
    hours_per_week / hours_per_nap = naps_per_week)) :=
begin
  sorry
end

end john_naps_per_week_l167_167889


namespace polygon_angle_multiple_l167_167223

theorem polygon_angle_multiple (m : ℕ) (h : m ≥ 3) : 
  (∃ k : ℕ, (2 * m - 2) * 180 = k * ((m - 2) * 180)) ↔ (m = 3 ∨ m = 4) :=
by sorry

end polygon_angle_multiple_l167_167223


namespace ordered_pair_solution_l167_167339

theorem ordered_pair_solution :
  ∃ x y : ℚ, 6 * x + 3 * y = -12 ∧ 4 * x = 5 * y + 10 ∧ x = -(5/7) ∧ y = -(18/7) :=
by {
  use [- (5/7), - (18/7)],
  split,
  { norm_num, linarith, },
  split,
  { norm_num, linarith, },
  split;
  { norm_num, },
  sorry
}

end ordered_pair_solution_l167_167339


namespace wheel_radius_problem_l167_167388

def wheelRadius (total_distance : ℝ) (revolutions : ℝ) := total_distance / (2 * Real.pi * revolutions)

theorem wheel_radius_problem : 
  wheelRadius 88000 1000 = 88 / (2 * Real.pi) := by
  sorry

end wheel_radius_problem_l167_167388


namespace area_of_circle_B_l167_167330

-- Define circles A and B
def Circle (r : ℝ) := { c : ℝ × ℝ | (fst c) ^ 2 + (snd c) ^ 2 = r ^ 2 }

-- Definitions based on conditions
def circleA_area (A : Circle) : Prop := Pi * (radius A) ^ 2 = 16 * Pi
def tangent_at_one_point (A B : Circle) : Prop := 
  ∃ C : ℝ × ℝ, C ∈ Boundary A ∧ C ∈ Boundary B ∧
  (∀ P ∈ Boundary A, P ≠ C → (P - C) ∉ Tangent A B)

-- Circle B's area should be 64 Pi when the given conditions are satisfied.
theorem area_of_circle_B : 
  ∀ (A B : Circle), circleA_area A → tangent_at_one_point A B → (Pi * (radius B) ^ 2 = 64 * Pi) :=
by
  sorry

end area_of_circle_B_l167_167330


namespace range_f_iff_l167_167424

noncomputable def f (m x : ℝ) : ℝ :=
  Real.log ((m^2 - 3 * m + 2) * x^2 + 2 * (m - 1) * x + 5)

theorem range_f_iff (m : ℝ) :
  (∀ y ∈ Set.univ, ∃ x, f m x = y) ↔ (m = 1 ∨ (2 < m ∧ m ≤ 9/4)) := 
by
  sorry

end range_f_iff_l167_167424


namespace total_unique_values_f_l167_167382

noncomputable def f (x : ℝ) : ℤ :=
  int.floor x + int.floor (2 * x) + int.floor (5 * x / 3) + int.floor (3 * x) + int.floor (4 * x)

theorem total_unique_values_f : ∀ (x : ℝ), 0 ≤ x → x ≤ 100 → 
  ∃ n : ℤ, n = 734 := sorry

end total_unique_values_f_l167_167382


namespace count_special_two_digit_numbers_l167_167058

theorem count_special_two_digit_numbers :
  let numbers := {n : ℕ | 30 ≤ n ∧ n ≤ 80 ∧ let t := n / 10 in 
                              let o := n % 10 in 
                              t > o ∧ t + o = 10} in 
  numbers.card = 4 :=
by {
  sorry -- Proof is omitted.
}

end count_special_two_digit_numbers_l167_167058


namespace number_of_possible_values_correct_l167_167961

noncomputable def number_of_possible_values (s : ℚ) : ℕ :=
  let lower_bound := 11 / 60
  let upper_bound := 9 / 40
  let range_start := 1833 / 10000
  let range_end := 2250 / 10000
  classical.some (nat.find (λ n : ℚ, n ≥ range_start ∧ n ≤ range_end)) - 
  classical.some (nat.find (λ n : ℚ, n > range_end)) +
  1

theorem number_of_possible_values_correct : 
  number_of_possible_values 0 = 418 :=
by
  sorry

end number_of_possible_values_correct_l167_167961


namespace fifth_term_equals_31_l167_167741

-- Define the sequence of sums of consecutive powers of 2
def sequence_sum (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (λ i, 2^i)

-- State the theorem: The fifth term of the sequence equals 31
theorem fifth_term_equals_31 : sequence_sum 4 = 31 := by
  sorry

end fifth_term_equals_31_l167_167741


namespace fred_limes_l167_167011

theorem fred_limes (limes_total : ℕ) (alyssa_limes : ℕ) (nancy_limes : ℕ) (fred_limes : ℕ)
  (h_total : limes_total = 103)
  (h_alyssa : alyssa_limes = 32)
  (h_nancy : nancy_limes = 35)
  (h_fred : fred_limes = limes_total - (alyssa_limes + nancy_limes)) :
  fred_limes = 36 :=
by
  sorry

end fred_limes_l167_167011


namespace calculate_y_position_l167_167404

/--
Given a number line with equally spaced markings, if eight steps are taken from \( 0 \) to \( 32 \),
then the position \( y \) after five steps can be calculated.
-/
theorem calculate_y_position : 
    ∃ y : ℕ, (∀ (step length : ℕ), (8 * step = 32) ∧ (y = 5 * length) → y = 20) :=
by
  -- Provide initial definitions based on the conditions
  let step := 4
  let length := 4
  use (5 * length)
  sorry

end calculate_y_position_l167_167404


namespace triangle_angle_bac_l167_167074

theorem triangle_angle_bac {
  (BC : ℝ) (AB : ℝ) (BAC ABC ACB : ℝ)
  (h1 : ABC = 2 * ACB)
  (h2 : BC = 2 * AB)
  (h3 : ∠BAC = 180 - ACB - (2 * ACB))
} : ∠BAC = 90 :=
  sorry

end triangle_angle_bac_l167_167074


namespace length_CD_l167_167623

theorem length_CD (L : ℝ) (r : ℝ) (V_total : ℝ) (h : ℝ) :
  r = 4 ∧ V_total = 320 * real.pi ∧
  V_total = (real.pi * r^2 * L) + ((2/3) * real.pi * r^3) + ((1/3) * real.pi * r^2 * r) →
  L = 16 :=
by
  intro h
  sorry

end length_CD_l167_167623


namespace regression_line_equation_l167_167832

-- Define conditions
variable (slope : ℝ) (x₀ y₀ : ℝ)

-- The conditions given in the problem
def slope_of_regression_line := slope = -1
def center_of_sample_points := (x₀, y₀) = (1, 2)

-- The proof problem
theorem regression_line_equation (h1 : slope_of_regression_line slope) (h2 : center_of_sample_points x₀ y₀) : 
  ∀ x y : ℝ, y - y₀ = slope * (x - x₀) → y = -x + 3 :=
by 
  intro x y h
  simp [slope_of_regression_line, center_of_sample_points] at h ⊢
  sorry

end regression_line_equation_l167_167832


namespace abcd_product_l167_167902

noncomputable def a := sqrt (4 + sqrt (5 - a))
noncomputable def b := sqrt (4 + sqrt (5 + b))
noncomputable def c := sqrt (4 - sqrt (5 - c))
noncomputable def d := sqrt (4 - sqrt (5 + d))

theorem abcd_product (a b c d : Real) 
  (h₁ : a = sqrt (4 + sqrt (5 - a)))
  (h₂ : b = sqrt (4 + sqrt (5 + b)))
  (h₃ : c = sqrt (4 - sqrt (5 - c)))
  (h₄ : d = sqrt (4 - sqrt (5 + d))) : 
  a * b * c * d = 11 :=
  sorry

end abcd_product_l167_167902


namespace quotient_of_P_div_Q_l167_167769

def P : Polynomial ℝ := 8 * X^4 + 7 * X^3 - 2 * X^2 - 9 * X + 5
def Q : Polynomial ℝ := X - 1
def R : Polynomial ℝ := 8 * X^3 + 15 * X^2 + 13 * X + 4

theorem quotient_of_P_div_Q :
  P / Q = R := by
  sorry

end quotient_of_P_div_Q_l167_167769


namespace certain_event_at_least_one_good_l167_167778

noncomputable def total_products := 12
noncomputable def good_products := 10
noncomputable def defective_products := 2
noncomputable def picked_products := 3

theorem certain_event_at_least_one_good: 
  (∃ g d : ℕ, g + d = picked_products ∧ g ≤ good_products ∧ d ≤ defective_products) :=
begin
  sorry
end

end certain_event_at_least_one_good_l167_167778


namespace sum_of_three_lowest_scores_l167_167589

theorem sum_of_three_lowest_scores 
  (scores : List ℝ) 
  (h_len : scores.length = 6) 
  (h_mean : list.sum scores / 6 = 85) 
  (h_median : (scores.nth_le 2 (by sorry) + scores.nth_le 3 (by sorry)) / 2 = 88) 
  (h_mode : ∃ n, list.count n scores = list.maximum (list.map (λ x, list.count x scores) scores) ∧ n = 90) :
  list.sum (list.take 3 (scores.qsort (≤))) = 242 := 
sorry

end sum_of_three_lowest_scores_l167_167589


namespace event_classification_l167_167177

-- Define the conditions: things that will certainly happen and things that will certainly not happen.
def certain_event (e : Prop) : Prop := ∀ (e : Prop), e
def impossible_event (e : Prop) : Prop := ∀ (e : Prop), ¬ e

-- Prove the defined conditions correspond to the correct labels.
theorem event_classification (e : Prop) :
  (certain_event e = (∀ (e : Prop), e)) ∧
  (impossible_event e = (∀ (e : Prop), ¬ e)) :=
by
  split
  -- Prove things we can be certain will happen in advance are called certain events
  case left {
    unfold certain_event
    sorry
  }
  -- Prove things we can be certain will not happen in advance are called impossible events
  case right {
    unfold impossible_event
    sorry
  }

end event_classification_l167_167177


namespace ella_total_revenue_l167_167757

variable (x y w v : Nat)
variable (a b : Rational)
variable (z t : Nat)

-- Given conditions
def ella_has_bags := 
  (x = 4 * 20) ∧ 
  (y = 6 * 25) ∧ 
  (w = 3 * 15) ∧ 
  (v = 5 * 30) ∧
  (a = 0.5) ∧
  (b = 0.75) ∧ 
  (z = 200) ∧ 
  (t = 75)

-- Prove the total revenue equals 156.25
theorem ella_total_revenue : ella_has_bags x y w v a b z t → ((z * a) + (t * b) = 156.25) :=
by
  sorry

end ella_total_revenue_l167_167757


namespace triangle_side_length_eq_sqrt3_l167_167516

theorem triangle_side_length_eq_sqrt3
  (P Q R: Type) [Point P] [Point Q] [Point R]
  (triangle_PQR: Triangle P Q R)
  (angle_QPR: angle P Q R = 60) 
  (perpendiculars_drawn : ∃ A B: Point, perpendicular_from P to QR = A ∧ perpendicular_from R to PQ = B)
  (intersection_point_equal_distance: ∃ M: Point, dist M P = 1 ∧ dist M Q = 1) :
  side_length P Q R = sqrt 3 := 
sorry

end triangle_side_length_eq_sqrt3_l167_167516


namespace no_values_of_z_satisfy_conditions_l167_167148

open Complex

theorem no_values_of_z_satisfy_conditions (f : ℂ → ℂ) (h_f : ∀ z, f z = 3 * I * conj z) :
  ∀ z : ℂ, (abs z = 4) ∧ (f z = z) → false :=
by
  sorry

end no_values_of_z_satisfy_conditions_l167_167148


namespace fifth_term_is_31_l167_167734

/-- 
  The sequence is formed by summing consecutive powers of 2. 
  Define the sequence and prove the fifth term is 31.
--/
def sequence_sum (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, 2^k)

theorem fifth_term_is_31 : sequence_sum 5 = 31 :=
by sorry

end fifth_term_is_31_l167_167734


namespace sum_angles_bisected_l167_167012

theorem sum_angles_bisected (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : 0 < θ₁) (h₂ : 0 < θ₂) (h₃ : 0 < θ₃) (h₄ : 0 < θ₄)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = 360) :
  (θ₁ / 2 + θ₃ / 2 = 180 ∨ θ₂ / 2 + θ₄ / 2 = 180) ∧ (θ₂ / 2 + θ₄ / 2 = 180 ∨ θ₁ / 2 + θ₃ / 2 = 180) := 
by 
  sorry

end sum_angles_bisected_l167_167012


namespace a_zero_sufficient_not_necessary_l167_167416

def is_even_function {α : Type*} [AddCommGroup α] (f : α → α) : Prop :=
  ∀ x, f (-x) = f x

def f (a b : ℝ) (x : ℝ) : ℝ :=
  x^2 + a * |x| + b

theorem a_zero_sufficient_not_necessary (a b : ℝ) :
  (∀ x, is_even_function (f a b)) ↔ (a = 0 → is_even_function (f a b)) ∧ (∃ a' : ℝ, a' ≠ 0 ∧ is_even_function (f a' b)) :=
sorry

end a_zero_sufficient_not_necessary_l167_167416


namespace minimum_distance_l167_167068

noncomputable def point_on_curve (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

noncomputable def point_on_line (x : ℝ) : ℝ := x + 2

theorem minimum_distance 
  (a b c d : ℝ) 
  (hP : b = point_on_curve a) 
  (hQ : d = point_on_line c) 
  : (a - c)^2 + (b - d)^2 = 8 :=
by
  sorry

end minimum_distance_l167_167068


namespace max_remainder_209_lt_120_l167_167690

theorem max_remainder_209_lt_120 : 
  ∃ n : ℕ, n < 120 ∧ (209 % n = 104) := 
sorry

end max_remainder_209_lt_120_l167_167690


namespace half_angle_in_quadrant_l167_167480

theorem half_angle_in_quadrant (k : ℤ) (θ : ℝ) (h : θ ∈ set.Ioo (2 * k * real.pi + real.pi / 2) (2 * k * real.pi + real.pi)) : 
  ∃ n : ℤ, (∃ θ' : ℝ, θ' = θ / 2 ∧ θ' ∈ set.Ioo (n * real.pi + real.pi / 4) (n * real.pi + real.pi / 2)) :=
begin
  sorry
end

end half_angle_in_quadrant_l167_167480


namespace krakozyabrs_count_l167_167110

theorem krakozyabrs_count :
  ∀ (K : Type) [fintype K] (has_horns : K → Prop) (has_wings : K → Prop), 
  (∀ k, has_horns k ∨ has_wings k) →
  (∃ n, (∀ k, has_horns k → has_wings k) → fintype.card ({k | has_horns k} : set K) = 5 * n) →
  (∃ n, (∀ k, has_wings k → has_horns k) → fintype.card ({k | has_wings k} : set K) = 4 * n) →
  (∃ T, 25 < T ∧ T < 35 ∧ T = 32) := 
by
  intros K _ has_horns has_wings _ _ _
  sorry

end krakozyabrs_count_l167_167110


namespace count_distinct_three_digit_even_numbers_l167_167471

theorem count_distinct_three_digit_even_numbers : 
  let even_digits := {0, 2, 4, 6, 8} in
  let first_digit_choices := {2, 4, 6, 8} in
  let second_and_third_digit_choices := even_digits in
  (finset.card first_digit_choices) * 
  (finset.card second_and_third_digit_choices) *
  (finset.card second_and_third_digit_choices) = 100 := by
  let even_digits := {0, 2, 4, 6, 8}
  let first_digit_choices := {2, 4, 6, 8}
  let second_and_third_digit_choices := even_digits
  have h1 : finset.card first_digit_choices = 4 := by simp
  have h2 : finset.card second_and_third_digit_choices = 5 := by simp
  calc (finset.card first_digit_choices) * 
       (finset.card second_and_third_digit_choices) *
       (finset.card second_and_third_digit_choices)
       = 4 * 5 * 5 : by rw [h1, h2]
    ... = 100 : by norm_num

end count_distinct_three_digit_even_numbers_l167_167471


namespace total_distance_l167_167596

-- Define points as coordinates in two-dimensional space
def Point : Type := ℝ × ℝ

-- Define the distance function using the Euclidean distance formula
noncomputable def distance (p1 p2 : Point) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the specific points involved
def start_point : Point := (-3, 6)
def origin_point : Point := (0, 0)
def end_point : Point := (6, -3)

-- Formulate the total distance problem statement
theorem total_distance :
  distance start_point origin_point + distance origin_point end_point = 2 * real.sqrt 45 :=
by
  sorry

end total_distance_l167_167596


namespace problem_inequality_l167_167751

theorem problem_inequality (p q : ℝ) (h1 : p > 3) (h2 : q > 0) : 
  7 * (pq^2 + p^2q + 3q^2 + 3pq) / (p + q) > 3 * p^2q :=
by sorry

end problem_inequality_l167_167751


namespace north_pond_ducks_l167_167586

-- Definitions based on the conditions
def ducks_lake_michigan : ℕ := 100
def twice_ducks_lake_michigan : ℕ := 2 * ducks_lake_michigan
def additional_ducks : ℕ := 6
def ducks_north_pond : ℕ := twice_ducks_lake_michigan + additional_ducks

-- Theorem to prove the answer
theorem north_pond_ducks : ducks_north_pond = 206 :=
by
  sorry

end north_pond_ducks_l167_167586


namespace num_of_distinct_three_digit_integers_with_even_digits_l167_167451

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def valid_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (is_even_digit (n / 100 % 10)) ∧
  (is_even_digit (n / 10 % 10)) ∧
  (is_even_digit (n % 10)) ∧
  (n / 100 % 10 ≠ 0)

theorem num_of_distinct_three_digit_integers_with_even_digits : 
  {n : ℕ | valid_three_digit_integer n}.finite.to_finset.card = 100 :=
sorry

end num_of_distinct_three_digit_integers_with_even_digits_l167_167451


namespace smallest_x_solution_l167_167770

def min_solution (a b : Real) : Real := if a < b then a else b

theorem smallest_x_solution (x : Real) (h : x * abs x = 3 * x - 2) : 
  x = min_solution (min_solution 1 2) ( (-3 - Real.sqrt 17) / 2) :=
sorry

end smallest_x_solution_l167_167770


namespace hyperbola_sum_l167_167079

theorem hyperbola_sum :
  let h := -2
  let k := 0
  let a := abs (-2 - (-5))  -- distance between center and vertex
  let c := abs (-2 - (-2 + Real.sqrt 34))  -- distance between center and focus
  let b := Real.sqrt (c ^ 2 - a ^ 2)  -- derived from the relationship c^2 = a^2 + b^2
  h + k + a + b = 6 :=
begin
  sorry
end

end hyperbola_sum_l167_167079


namespace tickets_difference_vip_general_l167_167292

theorem tickets_difference_vip_general (V G : ℕ) 
  (h1 : V + G = 320) 
  (h2 : 40 * V + 10 * G = 7500) : G - V = 34 := 
by
  sorry

end tickets_difference_vip_general_l167_167292


namespace find_even_monotonically_decreasing_function_l167_167716

-- Define the functions
def f1 (x : ℝ) : ℝ := x^3
def f2 (x : ℝ) : ℝ := Real.exp (-x)
def f3 (x : ℝ) : ℝ := -x^2 + 1
def f4 (x : ℝ) : ℝ := Real.log (Real.abs x)

-- State the main theorem
theorem find_even_monotonically_decreasing_function :
  f3 ∈ { f | (∀ x : ℝ, f (-x) = f x) ∧ (∀ x y : ℝ, 0 < x ∧ x < y → f x > f y) } :=
by
  -- sorry is used to skip the proof
  sorry

end find_even_monotonically_decreasing_function_l167_167716


namespace max_value_of_expression_l167_167942

variables (x y : ℝ)

theorem max_value_of_expression (hx : 0 < x) (hy : 0 < y) (h : x^2 - 2*x*y + 3*y^2 = 12) : x^2 + 2*x*y + 3*y^2 ≤ 24 + 24*sqrt 3 :=
sorry

end max_value_of_expression_l167_167942


namespace general_eqn_C_general_eqn_l_max_PA_min_PA_l167_167819

noncomputable theory
open Real

-- Definition of the curve C using parametric equations
def C (θ : ℝ) : ℝ × ℝ := (3 * cos θ, 2 * sin θ)

-- Definition of the line l using parametric equations
def l (t : ℝ) : ℝ × ℝ := (2 + t, 2 - 2 * t)

-- Proving the general equation of the curve C
theorem general_eqn_C : ∀ (x y : ℝ), (∃ θ : ℝ, C θ = (x, y)) ↔ (x^2 / 9 + y^2 / 4 = 1) :=
by sorry

-- Proving the general equation of the line l
theorem general_eqn_l : ∀ (x y : ℝ), (∃ t : ℝ, l t = (x, y)) ↔ (2 * x + y - 6 = 0) :=
by sorry

-- Proving the maximum value of |PA|
theorem max_PA : 
  ∀ (θ : ℝ), 
  let P := C θ in 
  let d := (sqrt 5 / 5) * abs (6 * cos θ + 2 * sin θ - 6) in 
  let PA := (2 * sqrt 10 / 5) * abs (sqrt 10 * sin (θ + atan 3) - 3) in
  PA ≤ (6 * sqrt 10 + 20) / 5 :=
by sorry

-- Proving the minimum value of |PA|
theorem min_PA : 
  ∀ (θ : ℝ), 
  let P := C θ in 
  let d := (sqrt 5 / 5) * abs (6 * cos θ + 2 * sin θ - 6) in 
  let PA := (2 * sqrt 10 / 5) * abs (sqrt 10 * sin (θ + atan 3) - 3) in
  PA ≥ (20 - 6 * sqrt 10) / 5 :=
by sorry

end general_eqn_C_general_eqn_l_max_PA_min_PA_l167_167819


namespace percent_slacks_in_hamper_l167_167233

-- Define the given conditions
def total_blouses : ℕ := 12
def total_skirts : ℕ := 6
def total_slacks : ℕ := 8
def percent_blouses_in_hamper : ℝ := 75 / 100
def percent_skirts_in_hamper : ℝ := 50 / 100
def pieces_needed_for_washer : ℕ := 14

-- Formalize the question as a Lean theorem
theorem percent_slacks_in_hamper :
  (pieces_needed_for_washer - (total_blouses * percent_blouses_in_hamper + total_skirts * percent_skirts_in_hamper)) / total_slacks * 100 = 25 := by 
  sorry

end percent_slacks_in_hamper_l167_167233


namespace product_of_solutions_l167_167228

theorem product_of_solutions (a b c x : ℝ) (h1 : -x^2 - 4 * x + 10 = 0) :
  x * (-4 - x) = -10 :=
by
  sorry

end product_of_solutions_l167_167228


namespace f_zero_and_negative_f_monotonic_a_range_l167_167554

-- Conditions stated as definitions
def f : ℝ → ℝ := sorry

axiom f_property : ∀ m n : ℝ, f(m + n) = f(m) * f(n)
axiom f_condition : ∀ x : ℝ, x > 0 → 0 < f(x) < 1

-- Proving part (1)
theorem f_zero_and_negative (x : ℝ) (h : x < 0) : f(0) = 1 ∧ f(x) > 1 := by
  sorry

-- Proving part (2)
theorem f_monotonic : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) > f(x2) := by
  sorry

-- Proving part (3)
def A : set (ℝ × ℝ) := {p | f(p.1^2) * f(p.2^2) > f(1)}
def B (a : ℝ) : set (ℝ × ℝ) := {p | f(a * p.1 - p.2 + 2) = 1}

theorem a_range (a : ℝ) : (A ∩ B(a) = ∅) → (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) := by
  sorry

end f_zero_and_negative_f_monotonic_a_range_l167_167554


namespace aaronFoundCards_l167_167310

-- Given conditions
def initialCardsAaron : ℕ := 5
def finalCardsAaron : ℕ := 67

-- Theorem statement
theorem aaronFoundCards : finalCardsAaron - initialCardsAaron = 62 :=
by
  sorry

end aaronFoundCards_l167_167310


namespace linear_eq_solution_l167_167486

theorem linear_eq_solution (m x : ℝ) (h : |m| = 1) (h1: 1 - m ≠ 0):
  x = -(1/2) :=
sorry

end linear_eq_solution_l167_167486


namespace equal_chance_of_selection_l167_167780

-- Define the set of all students
def total_students : ℕ := 2006

-- Define the number of students to be eliminated initially
def eliminated_students : ℕ := 6

-- Define the remaining students after elimination
def remaining_students : ℕ := total_students - eliminated_students

-- Define the number of students to be selected for the visiting group
def selected_students : ℕ := 50

-- Theorem: given the conditions, the chance of each person being selected is equal
theorem equal_chance_of_selection (total_students eliminated_students remaining_students selected_students : ℕ) 
    (h1 : total_students = 2006) 
    (h2 : eliminated_students = 6) 
    (h3 : remaining_students = total_students - eliminated_students) 
    (h4 : selected_students = 50) 
    : ∀ person, (person ∈ (range 2006) → (person ∈ (range 2000)) → probability_of_selection total_students eliminated_students selected_students person = 1 / 2000) := 
sorry

end equal_chance_of_selection_l167_167780


namespace joe_initial_tests_l167_167523

theorem joe_initial_tests (S n : ℕ) (h1 : S = 60 * n) (h2 : (S - 45) = 65 * (n - 1)) : n = 4 :=
by {
  sorry
}

end joe_initial_tests_l167_167523


namespace not_divisible_by_44_l167_167178

theorem not_divisible_by_44 (k : ℤ) (n : ℤ) (h1 : n = k * (k + 1) * (k + 2)) (h2 : 11 ∣ n) : ¬ (44 ∣ n) :=
sorry

end not_divisible_by_44_l167_167178


namespace f_yz_product_l167_167432

noncomputable section
open_locale classical -- Enable classical logic

-- Declare the function f(x)
def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- State the conditions
variables (y z : ℝ)
variable h1 : -1 < y ∧ y < 1
variable h2 : -1 < z ∧ z < 1
variable h3 : f ((y + z) / (1 + y * z)) = 1
variable h4 : f ((y - z) / (1 - y * z)) = 2

-- The main statement to prove
theorem f_yz_product : f y * f z = -3 / 4 :=
sorry

end f_yz_product_l167_167432


namespace largest_of_three_consecutive_integers_sum_90_is_31_l167_167970

theorem largest_of_three_consecutive_integers_sum_90_is_31 :
  ∃ (a b c : ℤ), (a + b + c = 90) ∧ (b = a + 1) ∧ (c = b + 1) ∧ (c = 31) :=
by
  sorry

end largest_of_three_consecutive_integers_sum_90_is_31_l167_167970


namespace RU_squared_l167_167508

-- Definitions of the geometric elements and givens
variables (L M N O P Q T R S U V : Type)
variables (LQ LT side : ℝ)
variables (LM : ℝ → ℝ → Prop) (LN : ℝ → ℝ → Prop) (LO : ℝ → ℝ → Prop) 
variables (RU QT SV : ℝ)

-- Given conditions
axiom h1 : LM 2 2                   -- Square's side length is 2
axiom h2 : LQ = LT                  -- LQ equals LT
axiom h3 : side = 2                 -- Side length of square is 2
axiom h4 : ∀ A B, A ≠ B → (A = B ↔ false) -- Difference of distinct points
axiom area_LQT : 1                  -- Area of triangle LQT
axiom area_MRUQ : 1                 -- Area of quadrilateral MRUQ
axiom area_PSVT : 1                 -- Area of quadrilateral PSVT
axiom area_NRSUV : 1                -- Area of pentagon NRSUV
axiom perpendicular_RU_QT : RU ⟂ QT -- RU is perpendicular to QT
axiom perpendicular_SV_QT : SV ⟂ QT -- SV is perpendicular to QT

-- Statement to prove
theorem RU_squared : RU^2 = 8 - 4 * sqrt 2 := by
  sorry

end RU_squared_l167_167508


namespace triangle_properties_l167_167257

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def longest_side_triangle 
  (A B C : ℝ × ℝ) 
  (hA : A = (1, 3)) 
  (hB : B = (4, 7)) 
  (hC : C = (4, 3)) : ℝ :=
  max (distance A B) (max (distance A C) (distance B C))

noncomputable def perimeter_triangle
  (A B C : ℝ × ℝ) 
  (hA : A = (1, 3)) 
  (hB : B = (4, 7)) 
  (hC : C = (4, 3)) : ℝ :=
  distance A B + distance A C + distance B C

theorem triangle_properties :
  ∀ (A B C : ℝ × ℝ), 
  A = (1, 3) → 
  B = (4, 7) → 
  C = (4, 3) →
  longest_side_triangle A B C = 5 ∧ perimeter_triangle A B C = 12 := 
by
  intros A B C hA hB hC
  -- proof will go here
  sorry

end triangle_properties_l167_167257


namespace actual_distance_traveled_l167_167681

theorem actual_distance_traveled 
  (D : ℝ)
  (h1 : ∃ (D : ℝ), D/12 = (D + 36)/20)
  : D = 54 :=
sorry

end actual_distance_traveled_l167_167681


namespace log_base_4_of_2_l167_167367

theorem log_base_4_of_2 : log 4 2 = 1 / 2 := by
  sorry

end log_base_4_of_2_l167_167367


namespace tinplates_to_match_l167_167646

theorem tinplates_to_match (x : ℕ) (y : ℕ) (total_tinplates : ℕ) 
    (h1 : total_tinplates = 36)
    (h2 : 25 * x = 40 * (total_tinplates - x) / 2) :
    x = 16 ∧ total_tinplates - x = 20 :=
by
  have eq1 : total_tinplates - x = 20, sorry
  have eq2 : x = 16, sorry
  exact ⟨eq2, eq1⟩

end tinplates_to_match_l167_167646


namespace percent_of_y_l167_167248

theorem percent_of_y (y : ℝ) (h : y > 0) : (2 * y) / 10 + (3 * y) / 10 = (50 / 100) * y :=
by
  sorry

end percent_of_y_l167_167248


namespace bob_total_questions_l167_167323

theorem bob_total_questions (q1 q2 q3 : ℕ) : 
  q1 = 13 ∧ q2 = 2 * q1 ∧ q3 = 2 * q2 → q1 + q2 + q3 = 91 :=
by
  intros
  sorry

end bob_total_questions_l167_167323


namespace ellipse_properties_l167_167794

/-- Given the foci of an ellipse and the condition that the sum of the distances from 
    any point on the ellipse to the two foci is constant and equal to twice the length 
    of the semi-major axis, find the equation of the ellipse and the area of the triangle 
    formed by the point on the ellipse in the second quadrant, and the two foci. -/
theorem ellipse_properties (F₁ F₂ : ℝ × ℝ) (hF₁ : F₁ = (-1, 0)) (hF₂ : F₂ = (1, 0))
  (h_distance : 2 * euclidean_distance F₁ F₂ = euclidean_distance P F₁ + euclidean_distance P F₂)
  (h_F1F2 : euclidean_distance F₁ F₂ = 2) (P : ℝ × ℝ) (hP_second_quadrant : P.1 < 0 ∧ P.2 > 0)
  (h_angle : ∠ P F₁ F₂ = 120) :
  (∃ a b : ℝ, a = 2 ∧ b = sqrt 3 ∧ ∀ x y : ℝ, (x, y) ∈ ellipse a b ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ area : ℝ, area = 3 * sqrt 3 / 5) := 
sorry

end ellipse_properties_l167_167794


namespace reflection_matrix_correct_l167_167534

-- Define the normal vector n
def n : ℝ^3 := ⟨2, -1, 1⟩

-- Define the reflection matrix S
def S : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    [-1/3, 4/3, 4/3],
    [4/3, 5/3, 7/3],
    [4/3, 4/3, 5/3]
  ]

-- Define the function that calculates the reflection of u through the plane Q
def reflection_through_plane (u : ℝ^3) : ℝ^3 := 
  let proj_n := (2 * u.1 - u.2 + u.3) / 6 * n
  let q := u - proj_n
  2 * q - u

-- Proposition that the reflection matrix S correctly reflects any vector through plane Q
theorem reflection_matrix_correct (u : ℝ^3) : S.mulVec u = reflection_through_plane u := by
  sorry

end reflection_matrix_correct_l167_167534


namespace a_seq_general_term_b_seq_general_term_sum_T_n_l167_167829

/-- Given conditions for sequences a_n and b_n --/
def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2 * n + 1

def b_seq (n : ℕ) : ℕ :=
  if n = 1 then 4 else 2 * n + 1

/-- Sum of first n terms of sequence {1/(b_n * b_(n+1))} --/
def T_n (n : ℕ) : ℚ :=
  (6 * n - 1) / (20 * (2 * n + 3))

/-- Prove the general term a_n --/
theorem a_seq_general_term (n : ℕ) : a_seq n = 2 * n + 1 :=
  sorry

/-- Prove the general term b_n --/
theorem b_seq_general_term (n : ℕ) : b_seq n = if n = 1 then 4 else 2 * n + 1 :=
  sorry

/-- Prove the sum of the first n terms T_n --/
theorem sum_T_n (n : ℕ) : 
  let seq_prod_sum := (1/(b_seq 1) * (b_seq 2)) + (∑ i in finset.range (n - 1), 1 / (b_seq (i+1) * b_seq (i+2))) 
  in seq_prod_sum = T_n n :=
  sorry

end a_seq_general_term_b_seq_general_term_sum_T_n_l167_167829


namespace tinplate_distribution_correct_l167_167648

-- Conditions
def box_body_per_tinplate := 25
def box_bottom_per_tinplate := 40
def total_tinplates := 36
def body_to_bottom_ratio := 2

-- Variables to prove:
def x (tinplates_for_bodies : ℕ) := tinplates_for_bodies
def y (tinplates_for_bottoms : ℕ) := total_tinplates - tinplates_for_bodies

-- Original conditions as equations:
lemma tinplates_used_correctly (bodies bottom_ratio bottoms : ℕ) (h1 : bodies = box_body_per_tinplate * x bodies) (h2 : bottom_ratio * bodies = box_bottom_per_tinplate * y bottoms) : Prop :=
  2 * bodies = box_bottom_per_tinplate * y bottoms

-- Theorem to prove
theorem tinplate_distribution_correct : ∃ (tinplates_for_bodies tinplates_for_bottoms : ℕ),
  total_tinplates = tinplates_for_bodies + tinplates_for_bottoms ∧
  body_to_bottom_ratio * box_body_per_tinplate * tinplates_for_bodies = box_bottom_per_tinplate * tinplates_for_bottoms ∧
  tinplates_for_bodies = 16 ∧
  tinplates_for_bottoms = 20 :=
by
  use 16, 20
  split; sorry  -- Skip the proof using sorry for now to complete the statement

end tinplate_distribution_correct_l167_167648


namespace final_remaining_money_l167_167443

-- Define conditions as given in the problem
def monthly_income : ℕ := 2500
def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50
def expenses_total : ℕ := rent + car_payment + utilities + groceries
def remaining_money : ℕ := monthly_income - expenses_total
def retirement_contribution : ℕ := remaining_money / 2

-- State the theorem to be proven
theorem final_remaining_money : (remaining_money - retirement_contribution) = 650 := by
  sorry

end final_remaining_money_l167_167443


namespace num_of_distinct_three_digit_integers_with_even_digits_l167_167450

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def valid_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (is_even_digit (n / 100 % 10)) ∧
  (is_even_digit (n / 10 % 10)) ∧
  (is_even_digit (n % 10)) ∧
  (n / 100 % 10 ≠ 0)

theorem num_of_distinct_three_digit_integers_with_even_digits : 
  {n : ℕ | valid_three_digit_integer n}.finite.to_finset.card = 100 :=
sorry

end num_of_distinct_three_digit_integers_with_even_digits_l167_167450


namespace center_of_symmetry_max_omega_for_increasing_g_l167_167607

open Real

/-- Definition of the function f with given parameters -/
def f (A ω φ x : ℝ) : ℝ := A * sin (ω * x + φ)

/-- Center of symmetry problem setup -/
theorem center_of_symmetry (A ω φ : ℝ)
  (hA : A > 0)
  (hw : 0 < ω ∧ ω < 16)
  (hφ : 0 < φ ∧ φ < π / 2)
  (hmax : ∀ x, abs (f A ω φ x) ≤ sqrt 2)
  (h0 : f A ω φ 0 = 1)
  (hp : f A ω φ (π / 8) = sqrt 2) :
  ∃ k ∈ ℤ, is_symmetry_center (f A ω φ) (k * (π / 2) - π / 8) :=
sorry

/-- Increasing function problem setup -/
theorem max_omega_for_increasing_g :
  exists ω, (4 * sin (4 * x - π / (2 * ω) + π / 4)) ∧ (0 ≤ x ∧ x ≤ π / 8) ∧ (ω = 2) :=
sorry

end center_of_symmetry_max_omega_for_increasing_g_l167_167607


namespace find_tuples_l167_167386

theorem find_tuples 
  (n : ℕ) 
  (a : Fin (n + 1) → ℤ)
  (b : Fin (n + 1) → ℕ) 
  (b_def : ∀ k, b k = (Finset.univ.filter (λ j, a j = k)).card)
  (c : Fin (n + 1) → ℕ)
  (c_def : ∀ k, c k = (Finset.univ.filter (λ j, b j = k)).card)
: (∀ k : Fin (n + 1), a k = c k) := 
sorry

end find_tuples_l167_167386


namespace fraction_of_orange_juice_l167_167999

def pitcher1_volume := 500
def pitcher2_volume := 800
def juice_in_pitcher1 := (1 / 4 : ℚ)
def juice_in_pitcher2 := (1 / 2 : ℚ)
def full_volume := pitcher1_volume + pitcher2_volume

theorem fraction_of_orange_juice :
  let orange_juice_pitcher1 := pitcher1_volume * juice_in_pitcher1
      orange_juice_pitcher2 := pitcher2_volume * juice_in_pitcher2
      total_orange_juice := orange_juice_pitcher1 + orange_juice_pitcher2
  in total_orange_juice / full_volume = (21 / 52 : ℚ) :=
by
  sorry

end fraction_of_orange_juice_l167_167999


namespace area_of_region_divided_by_chord_l167_167497

-- Definitions of the given conditions
def circle (R: ℝ) := set (λ p, p.1 ^ 2 + p.2 ^ 2 = R^2)
def is_chord (c: set (ℝ × ℝ)) (l: ℝ) := ∃ A B, A ≠ B ∧ dist A B = l ∧ ∀ p ∈ c, dist p (0, 0) < 30

-- The main theorem we need to prove
theorem area_of_region_divided_by_chord :
  ∀ P : ℝ × ℝ, 
  (dist P (0, 0) = 12) ∧ circle 30 → 
  (∃ c1 c2 : set (ℝ × ℝ), is_chord c1 50 ∧ is_chord c2 40 ∧ c1 ∩ c2 = {P}) →
  area_of_region_divided_by_chord = 306 * real.pi - 122.86 := 
by
  sorry

end area_of_region_divided_by_chord_l167_167497


namespace find_xyz_l167_167649

theorem find_xyz : ∃ (x y z : ℕ), x + y + z = 12 ∧ 7 * x + 5 * y + 8 * z = 79 ∧ x = 5 ∧ y = 4 ∧ z = 3 :=
by
  sorry

end find_xyz_l167_167649


namespace baking_time_ratio_l167_167566

noncomputable def usual_assemble_time : ℝ := 1
noncomputable def usual_baking_time : ℝ := 1.5
noncomputable def usual_decorating_time : ℝ := 1
noncomputable def failed_oven_total_time : ℝ := 5

theorem baking_time_ratio :
  let usual_total_time := usual_assemble_time + usual_baking_time + usual_decorating_time
  let failed_oven_baking_time := failed_oven_total_time - (usual_assemble_time + usual_decorating_time)
  failed_oven_baking_time / usual_baking_time = 2 :=
by
  have usual_total_time := usual_assemble_time + usual_baking_time + usual_decorating_time
  have failed_oven_baking_time := failed_oven_total_time - (usual_assemble_time + usual_decorating_time)
  show failed_oven_baking_time / usual_baking_time = 2
  sorry

end baking_time_ratio_l167_167566


namespace sin_sum_triangle_l167_167399

theorem sin_sum_triangle (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_l167_167399


namespace polygon_area_inequality_l167_167286

variable {B A C : ℝ}

theorem polygon_area_inequality (hB : 0 < B) (hA : 0 < A) (hC : 0 < C)
  (polygon_inscribed_in_circle : B.inscribedCircle = A)
  (polygon_circumscribed_around_circle : B.circumscribedCircle = C) :
  2 * B ≤ A + C :=
sorry

end polygon_area_inequality_l167_167286


namespace reflection_matrix_correct_l167_167535

-- Define the normal vector n
def n : ℝ^3 := ⟨2, -1, 1⟩

-- Define the reflection matrix S
def S : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    [-1/3, 4/3, 4/3],
    [4/3, 5/3, 7/3],
    [4/3, 4/3, 5/3]
  ]

-- Define the function that calculates the reflection of u through the plane Q
def reflection_through_plane (u : ℝ^3) : ℝ^3 := 
  let proj_n := (2 * u.1 - u.2 + u.3) / 6 * n
  let q := u - proj_n
  2 * q - u

-- Proposition that the reflection matrix S correctly reflects any vector through plane Q
theorem reflection_matrix_correct (u : ℝ^3) : S.mulVec u = reflection_through_plane u := by
  sorry

end reflection_matrix_correct_l167_167535


namespace orthogonal_diagonals_l167_167254

variables {A1 A2 A3 A4 B1 B2 B3 B4 P Q R S : Point}

-- Define the points
variables {A1A2 A2A3 A3A4 A4A1 B1B2 B2B3 B3B4 B4B1 A1B1 A2B2 A3B3 A4B4 : Segment}
-- Define squares and midpoints
variables (squareA : is_square A1 A2 A3 A4) (squareB : is_square B1 B2 B3 B4)

-- Define perpendicular bisectors intersecting at P, Q, R, S
variables (P Q R S : Point) (C1 C2 C3 C4 : Circle)

-- The perpendicular bisectors of segments \(A_iB_i\)
variables (bisector1 : is_perpendicular_bisector A1B1 P)
variables (bisector2 : is_perpendicular_bisector A2B2 Q)
variables (bisector3 : is_perpendicular_bisector A3B3 R)
variables (bisector4 : is_perpendicular_bisector A4B4 S)

theorem orthogonal_diagonals :
  is_square A1 A2 A3 A4 →
  is_square B1 B2 B3 B4 →
  is_perpendicular_bisector A1B1 P →
  is_perpendicular_bisector A2B2 Q →
  is_perpendicular_bisector A3B3 R →
  is_perpendicular_bisector A4B4 S →
  orthogonal (segment PR) (segment QS) :=
by
  sorry

end orthogonal_diagonals_l167_167254


namespace fifth_term_equals_31_l167_167740

-- Define the sequence of sums of consecutive powers of 2
def sequence_sum (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (λ i, 2^i)

-- State the theorem: The fifth term of the sequence equals 31
theorem fifth_term_equals_31 : sequence_sum 4 = 31 := by
  sorry

end fifth_term_equals_31_l167_167740


namespace min_value_of_quadratic_expression_l167_167595

variable (x y z : ℝ)

theorem min_value_of_quadratic_expression 
  (h1 : 2 * x + 2 * y + z + 8 = 0) : 
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 = 9 :=
sorry

end min_value_of_quadratic_expression_l167_167595


namespace total_drums_filled_l167_167564

theorem total_drums_filled (
  monday : ℕ := 324,
  tuesday : ℕ := 358,
  wednesday : ℕ := 389,
  thursday : ℕ := 415,
  friday : ℕ := 368,
  saturday : ℕ := 402,
  sunday : ℕ := 440
) : 
  monday + tuesday + wednesday + thursday + friday + saturday + sunday = 2696 :=
by
  sorry

end total_drums_filled_l167_167564


namespace how_many_numbers_fit_in_A_l167_167695

theorem how_many_numbers_fit_in_A :
  let A_values := {A | 1 ≤ A ∧ A ≤ 9 ∧ (57 * 7 > 65 * A) } in
  Fintype.card A_values = 6 :=
by
  sorry

end how_many_numbers_fit_in_A_l167_167695


namespace maximum_value_is_l167_167937

noncomputable def maximum_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) : ℝ :=
  x^2 + 2 * x * y + 3 * y^2

theorem maximum_value_is (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) :
  maximum_value x y h₁ h₂ h₃ ≤ 18 + 12 * Real.sqrt 3 :=
sorry

end maximum_value_is_l167_167937


namespace degree_of_given_polynomial_l167_167952

def polynomial_degree (p : Polynomial ℤ) : ℕ :=
  p.natDegree

def poly : Polynomial (Polynomial ℤ) := (C (C 1) + C (C 2) * X * Y - C (C 3) * X * Y^3)

theorem degree_of_given_polynomial :
  polynomial_degree poly = 4 :=
sorry

end degree_of_given_polynomial_l167_167952


namespace more_subsets_with_product_greater_than_2009_l167_167894

noncomputable def M : Set ℕ := {1, 2, 3, 4, 6, 8, 12, 16, 24, 48}

theorem more_subsets_with_product_greater_than_2009 :
  let subsets := M.powerset.filter (λ s, s.size = 4) in
  let subsets_with_product_gt_2009 := subsets.filter (λ s, s.prod id > 2009) in
  let subsets_with_product_lt_2009 := subsets.filter (λ s, s.prod id < 2009) in
  subsets_with_product_gt_2009.card > subsets_with_product_lt_2009.card :=
sorry

end more_subsets_with_product_greater_than_2009_l167_167894


namespace medium_pizza_promotion_price_l167_167125

-- Define the conditions
def regular_price_medium_pizza : ℝ := 18
def total_savings : ℝ := 39
def number_of_medium_pizzas : ℝ := 3

-- Define the goal
theorem medium_pizza_promotion_price : 
  ∃ P : ℝ, 3 * regular_price_medium_pizza - 3 * P = total_savings ∧ P = 5 := 
by
  sorry

end medium_pizza_promotion_price_l167_167125


namespace cycle_selling_price_l167_167700

theorem cycle_selling_price 
  (CP : ℝ) (gain_percent : ℝ) (SP : ℝ) 
  (h1 : CP = 840) 
  (h2 : gain_percent = 45.23809523809524 / 100)
  (h3 : SP = CP * (1 + gain_percent)) :
  SP = 1220 :=
sorry

end cycle_selling_price_l167_167700


namespace greatest_length_of_pieces_l167_167520

theorem greatest_length_of_pieces (a b c : ℕ) (ha : a = 48) (hb : b = 60) (hc : c = 72) :
  Nat.gcd (Nat.gcd a b) c = 12 := by
  sorry

end greatest_length_of_pieces_l167_167520


namespace angle_of_inclination_l2_l167_167419

-- Definitions and conditions
def slope (m : ℝ) := m
def perpendicular_slopes (m1 m2 : ℝ) := m1 * m2 = -1

theorem angle_of_inclination_l2 (m1 : ℝ) (m2 : ℝ) (θ : ℝ) :
  slope m1 = 1 → perpendicular_slopes m1 m2 → θ = 135 :=
by
  intro h1 h2
  sorry

end angle_of_inclination_l2_l167_167419


namespace combine_radicals_l167_167489

theorem combine_radicals (x : ℝ) (h : sqrt (x + 1) = - (1/2) * sqrt (2 * x)) : x = 1 :=
by {
  sorry
}

end combine_radicals_l167_167489


namespace num_of_distinct_three_digit_integers_with_even_digits_l167_167452

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def valid_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (is_even_digit (n / 100 % 10)) ∧
  (is_even_digit (n / 10 % 10)) ∧
  (is_even_digit (n % 10)) ∧
  (n / 100 % 10 ≠ 0)

theorem num_of_distinct_three_digit_integers_with_even_digits : 
  {n : ℕ | valid_three_digit_integer n}.finite.to_finset.card = 100 :=
sorry

end num_of_distinct_three_digit_integers_with_even_digits_l167_167452


namespace range_of_x_l167_167753

theorem range_of_x :
  ∀ (b : Fin 20 → ℕ), (∀ i, b i = 0 ∨ b i = 3) →
  let x := ∑ i in Finset.range 20, b i / (4 : ℕ) ^ (i + 1)
  in (0 ≤ x ∧ x < 1 / 4) ∨ (3 / 4 ≤ x ∧ x < 1) :=
by
  intros b hb x hx
  sorry

end range_of_x_l167_167753


namespace square_free_sum_eq_l167_167306

-- Define what it means for an integer to be square-free
def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

-- Define the set of positive square-free integers
def S : set ℕ := {n | n > 0 ∧ is_square_free n}

-- Define the main theorem to be proved
theorem square_free_sum_eq :
  (∑ k in S, nat.floor (real.sqrt (10^10 / k))) = 10^10 :=
sorry

end square_free_sum_eq_l167_167306


namespace range_of_k_l167_167800

theorem range_of_k (k : ℝ) :
  let P := (1 : ℝ, 2 : ℝ),
      circle := λ x y, x^2 + y^2 + 2*x + y + 2*k - 1 in
  (2*k - 1 < (P.1^2 + 2*P.1 + P.2^2 + P.2)) ∧ 
  (P.1^2 + P.2^2 - 2*(2*k - 1) > 0) → 
  -4 < k ∧ k < 9/8 :=
sorry

end range_of_k_l167_167800


namespace sum_of_sins_is_zero_l167_167379

variable {x y z : ℝ}

theorem sum_of_sins_is_zero
  (h1 : Real.sin x = Real.tan y)
  (h2 : Real.sin y = Real.tan z)
  (h3 : Real.sin z = Real.tan x) :
  Real.sin x + Real.sin y + Real.sin z = 0 :=
sorry

end sum_of_sins_is_zero_l167_167379


namespace simplify_expression_l167_167171

theorem simplify_expression : 
  (√6 / √10) * (√5 / √15) * (√8 / √14) = (2 * √35) / 35 := 
by
  sorry

end simplify_expression_l167_167171


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_l167_167021

theorem arithmetic_sequence_general_formula :
  (∃ a_n : ℕ → ℕ, a_n 2 = 2 ∧ a_n 5 = 8 ∧ ∀ n : ℕ, a_n n = 2 * n - 2) :=
by
  let a_n := λ n : ℕ, 2 * n - 2
  use a_n
  simp [a_n]

theorem geometric_sequence_sum (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (a_2_eq : a_n 2 = 2) (a_5_eq : a_n 5 = 8) 
  (b_1_eq : b_n 1 = 1) (b_2_b_3_eq : b_n 2 + b_n 3 = a_n 4)
  (h_a_n : ∀ n : ℕ, a_n n = 2 * n - 2) :
  ∃ T_n : ℕ → ℕ, ∀ n : ℕ, T_n n = 2 ^ n - 1 :=
by
  let T_n := λ n : ℕ, 2 ^ n - 1
  use T_n
  sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_l167_167021


namespace measure_angle_KDA_l167_167967

-- Definitions of points and angles:
variables {A B C D M K : Point}

-- Given conditions as hypotheses:
axiom rectangle_ABCD : rectangle A B C D
axiom AD_eq_2AB : dist A D = 2 * dist A B
axiom M_midpoint_AD : midpoint M A D
axiom angle_AMK_80 : angle A M K = 80
axiom KD_angle_bisector_MKC : angle_bisector K D M C

-- Statement to prove:
theorem measure_angle_KDA :
  angle K D A = 35 :=
sorry

end measure_angle_KDA_l167_167967


namespace b_is_neg_38_l167_167963

-- Define the polynomial P(x) = 2x^3 + ax^2 + bx + c
def P (x a b c : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + c

-- Conditions
variables (a b c : ℝ)
variables (meanZeros productZeros sumCoefficients : ℝ)

-- The mean of the zeros, the product of the zeros, and the sum of the coefficients are all equal
axiom mean_product_sum_eq : meanZeros = productZeros ∧ productZeros = sumCoefficients

-- The y-intercept is 8
axiom y_intercept_is_8 : P 0 a b c = 8

-- The product of the roots for the given polynomial
axiom product_of_roots : productZeros = -c / 2

-- The mean of the zeros (sum of zeros divided by 3 is equal to meanZeros)
axiom mean_of_zeros : meanZeros = -c / 2

-- The sum of the zeros
axiom sum_of_zeros : -b / 2 = 3 * mean_of_zeros

-- The sum of the coefficients is equal to the mean of the zeros (which is given to be -4)
axiom sum_of_coefficients : 2 + a + b + c = -4

-- Prove that b = -38
theorem b_is_neg_38 : b = -38 :=
sorry

end b_is_neg_38_l167_167963


namespace find_k_l167_167836

variables (a b : ℝ × ℝ) (k : ℝ)

-- Given vectors
def a := (1, 2)
def b := (-3, 2)

-- Definition of parallel vectors
def parallel (u v : ℝ × ℝ) : Prop := ∃ c : ℝ, u = (c * v.1, c * v.2)

-- Problem statement in Lean
theorem find_k (h₁ : k • a + b = (k - 3, 2 * k + 2))
    (h₂ : a + 3 • b = (-8, 8))
    (h₃ : parallel (k • a + b) (a + 3 • b)) :
  k = 1 / 3 :=
sorry


end find_k_l167_167836


namespace log_base_4_of_2_l167_167364

theorem log_base_4_of_2 : log 4 2 = 1 / 2 :=
by sorry

end log_base_4_of_2_l167_167364


namespace price_of_orange_is_60_l167_167722

-- Definitions from the conditions
def price_of_apple : ℕ := 40 -- The price of each apple is 40 cents
def total_fruits : ℕ := 10 -- Mary selects a total of 10 apples and oranges
def avg_price_initial : ℕ := 48 -- The average price of the 10 pieces of fruit is 48 cents
def put_back_oranges : ℕ := 2 -- Mary puts back 2 oranges
def avg_price_remaining : ℕ := 45 -- The average price of the remaining fruits is 45 cents

-- Variable definition for the price of an orange which will be solved for
variable (price_of_orange : ℕ)

-- Theorem: proving the price of each orange is 60 cents given the conditions
theorem price_of_orange_is_60 : 
  (∀ a o : ℕ, a + o = total_fruits →
  40 * a + price_of_orange * o = total_fruits * avg_price_initial →
  40 * a + price_of_orange * (o - put_back_oranges) = (total_fruits - put_back_oranges) * avg_price_remaining)
  → price_of_orange = 60 :=
by
  -- Proof is omitted
  sorry

end price_of_orange_is_60_l167_167722


namespace bells_start_time_l167_167258

theorem bells_start_time :
  ∃ t: ℕ, t = 56 ∧ 
  ∃ lcm, lcm = Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))) ∧
  t % (lcm / 60) = 4 :=
begin
  sorry
end

end bells_start_time_l167_167258


namespace log_base_3_of_reciprocal_81_l167_167357

theorem log_base_3_of_reciprocal_81 : log 3 (1 / 81) = -4 :=
by
  sorry

end log_base_3_of_reciprocal_81_l167_167357


namespace systematic_sampling_second_group_l167_167995

theorem systematic_sampling_second_group
    (N : ℕ) (n : ℕ) (k : ℕ := N / n)
    (number_from_16th_group : ℕ)
    (number_from_1st_group : ℕ := number_from_16th_group - 15 * k)
    (number_from_2nd_group : ℕ := number_from_1st_group + k) :
    N = 160 → n = 20 → number_from_16th_group = 123 → number_from_2nd_group = 11 :=
by
  sorry

end systematic_sampling_second_group_l167_167995


namespace sqrt_div_add_l167_167384

theorem sqrt_div_add :
  let sqrt_0_81 := 0.9
  let sqrt_1_44 := 1.2
  let sqrt_0_49 := 0.7
  (Real.sqrt 1.1 / sqrt_0_81) + (sqrt_1_44 / sqrt_0_49) = 2.8793 :=
by
  -- Prove equality using the given conditions
  sorry

end sqrt_div_add_l167_167384


namespace QR_passes_through_fixed_point_l167_167401

open EuclideanGeometry

noncomputable def fixed_point_QR_passing_through (Ω : Circle) (A B C P Q R D : Point) (O : Point) :=
  -- Given conditions
  Tangent Ω A B ∧
  Tangent Ω A C ∧
  Tangent Ω P Q ∧
  Tangent Ω P R ∧
  Tangent Ω Q R ∧
  Parallel P AC R ∧
  Intersects P AC R B C ∧
  CenterOf Ω O ∧
  FootOfPerpendicular O A D ∧
  FootOfPerpendicular O B D ∧
  FootOfPerpendicular O C D ∧
  (QR_passes_through_fixed_point : ∀ R Q D, Collinear Q R D)

-- We claim that QR passes through a fixed point D
theorem QR_passes_through_fixed_point (Ω : Circle) (A B C P Q R D : Point) (O : Point) : 
  fixed_point_QR_passing_through Ω A B C P Q R D O :=
sorry

end QR_passes_through_fixed_point_l167_167401


namespace max_gcd_13n_plus_4_8n_plus_3_l167_167319

theorem max_gcd_13n_plus_4_8n_plus_3 : 
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 7 ∧ ∀ k : ℕ, k = gcd (13 * n + 4) (8 * n + 3) → k ≤ d :=
by
  sorry

end max_gcd_13n_plus_4_8n_plus_3_l167_167319


namespace right_triangle_hypotenuse_l167_167503

theorem right_triangle_hypotenuse (a b : ℕ) (a_val : a = 4) (b_val : b = 5) :
    ∃ c : ℝ, c^2 = (a:ℝ)^2 + (b:ℝ)^2 ∧ c = Real.sqrt 41 :=
by
  sorry

end right_triangle_hypotenuse_l167_167503


namespace f_periodic_odd_condition_l167_167193

theorem f_periodic_odd_condition (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 4) = f x) (h_one : f 1 = 5) : f 2015 = -5 :=
by
  sorry

end f_periodic_odd_condition_l167_167193


namespace sum_of_coefficients_eq_3125_l167_167846

theorem sum_of_coefficients_eq_3125 
  {b_5 b_4 b_3 b_2 b_1 b_0 : ℤ}
  (h : (2 * x + 3)^5 = b_5 * x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0) :
  b_5 + b_4 + b_3 + b_2 + b_1 + b_0 = 3125 := 
by 
  sorry

end sum_of_coefficients_eq_3125_l167_167846


namespace findSinA_findArea_l167_167855

noncomputable def sinValue : ℝ := sqrt 14 / 8
noncomputable def areaValue : ℝ := sqrt 7 / 4

variables {A B C : ℝ}
variables (a b c : ℝ)

-- Conditions
def anglesAndSides : Prop :=
  a = 1 ∧ c = sqrt 2 ∧ cos C = 3 / 4

-- Prove the values
theorem findSinA (h : anglesAndSides) : b = 2 → sin A = sinValue :=
sorry

theorem findArea (h : anglesAndSides) : (12 + b² - 3 * b) ∧ sin A = sinValue → 
  1/2 * a * b * sqrt 7 / 4 = areaValue :=
sorry

end findSinA_findArea_l167_167855


namespace integer_points_on_circle_l167_167474

theorem integer_points_on_circle :
  let center := (7 : ℝ, 3 : ℝ)
  let radius := 10
  let inside_or_on_circle (x y : ℝ) : Prop := (x - center.1)^2 + (y - center.2)^2 ≤ radius^2
  let point_on_line (x : ℝ) := (x, -2 * x)
  let condition := ∀ x, point_on_line x ∈ {p : ℝ × ℝ | inside_or_on_circle p.fst p.snd}
  ∃! n : ℕ, n = 6 := sorry

end integer_points_on_circle_l167_167474


namespace find_a_l167_167850

-- The conditions of the problem
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem find_a (a : ℝ) (h : isPureImaginary ((a - complex.I) / (1 + complex.I))) : a = 1 :=
by
  sorry

end find_a_l167_167850


namespace circles_position_l167_167036

noncomputable def positional_relationship
  (d : ℝ) 
  (r1 r2 : ℝ)
  (h_d : d = 3)
  (h_radii_roots : ∀ x, x^2 - 5 * x + 3 = (x - r1) * (x - r2))
  (h_distinct_radii : r1 ≠ r2) : Prop :=
  r1 + r2 = 5 ∧ ∥r1 - r2∥ = √13 / 2 → d < √13 → "one inside the other"

theorem circles_position
  (d = 3)
  (r1 r2 : ℝ)
  (h_radii_roots : ∀ x, x^2 - 5 * x + 3 = (x - r1) * (x - r2))
  (h_distinct_radii : r1 ≠ r2) : positional_relationship 3 r1 r2 d h_radii_roots h_distinct_radii :=
by
  sorry

end circles_position_l167_167036


namespace original_number_abc_l167_167874

theorem original_number_abc (a b c : ℕ)
  (h : 100 * a + 10 * b + c = 528)
  (N : ℕ)
  (h1 : N + (100 * a + 10 * b + c) = 222 * (a + b + c))
  (hN : N = 2670) :
  100 * a + 10 * b + c = 528 := by
  sorry

end original_number_abc_l167_167874


namespace find_a_of_tangent_parallel_l167_167491

theorem find_a_of_tangent_parallel :
  ∃ a : ℝ, (∃ f : ℝ → ℝ, f = (λ x, x^2 + a * x) ∧ 
    (∀ x, deriv f x = 2 * x + a) ∧ 
    (∀ a, deriv (λ x, x^2 + a * x) 1 = 2 + a) ∧ 
    (2 + a = 7)) → 
  a = 5 :=
by
  sorry

end find_a_of_tangent_parallel_l167_167491


namespace triangle_perimeter_l167_167487

theorem triangle_perimeter (a : ℕ) (h1 : a < 8) (h2 : a > 4) (h3 : a % 2 = 0) : 2 + 6 + a = 14 :=
  by
  sorry

end triangle_perimeter_l167_167487


namespace decreasing_omega_range_l167_167069

open Real

theorem decreasing_omega_range {ω : ℝ} (h1 : 1 < ω) :
  (∀ x y : ℝ, π ≤ x ∧ x ≤ y ∧ y ≤ (5 * π) / 4 → 
    (|sin (ω * y + π / 3)| ≤ |sin (ω * x + π / 3)|)) → 
  (7 / 6 ≤ ω ∧ ω ≤ 4 / 3) :=
by
  sorry

end decreasing_omega_range_l167_167069


namespace chord_length_of_parabola_l167_167279

noncomputable def length_of_chord (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem chord_length_of_parabola :
  let line := λ x : ℝ, -sqrt 3 * (x - 1)
  ∃ A B : ℝ × ℝ, (A.1 = 1 ∧ A.2 = 0) ∧ 
                 (A.2 = -sqrt 3 * (A.1 - 1)) ∧
                 (B.1 = 3 ∧ B.2 = -2*sqrt 3) ∧
                 (B.2 = -sqrt 3 * (B.1 - 1)) ∧
                 (A.1, A.2) ≠ (B.1, B.2) ∧
                 (A.2^2 = 4 * A.1) ∧
                 (B.2^2 = 4 * B.1) ∧
                 length_of_chord A.1 A.2 B.1 B.2 = 16 / 3 := 
by
  sorry

end chord_length_of_parabola_l167_167279


namespace sum_of_remainders_mod_l167_167672

theorem sum_of_remainders_mod (a b c : ℕ) (h1 : a % 53 = 31) (h2 : b % 53 = 22) (h3 : c % 53 = 7) :
  (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_remainders_mod_l167_167672


namespace extreme_value_of_f_intersection_range_l167_167430

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - Real.log x) / x
noncomputable def g (x : ℝ) : ℝ := -1

theorem extreme_value_of_f (a : ℝ) : 
  ∃ x : ℝ, x = Real.exp (a + 1) ∧ f a x = -Real.exp (-a - 1) :=
by
  sorry

theorem intersection_range (a : ℝ) :
  (∃ x ∈ Ioc 0 Real.exp(1), f a x = -1) ↔ (a ≤ -1 ∨ 0 ≤ a ∧ a ≤ Real.exp 1) :=
by
  sorry

end extreme_value_of_f_intersection_range_l167_167430


namespace range_of_eccentricity_ellipse_l167_167414

theorem range_of_eccentricity_ellipse (a b : ℝ) (ha : a > b) (hb : b > 0) :
  (∃ (A B : ℝ×ℝ), (let F1 := (c, 0) in let F2 := (-c, 0) in (A ∈ ellipse a b) ∧ (B ∈ ellipse a b) ∧
  (∥A - F1∥ = 3 * ∥B - F2∥))) →
  (let e := (1 - b^2 / a^2).sqrt in 0 < e ∧ e < 1/3) :=
sorry

def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | let (x, y) := p in x^2 / a^2 + y^2 / b^2 = 1}

def c (a b : ℝ) : ℝ := (a^2 - b^2).sqrt

end range_of_eccentricity_ellipse_l167_167414


namespace num_of_distinct_three_digit_integers_with_even_digits_l167_167449

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def valid_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (is_even_digit (n / 100 % 10)) ∧
  (is_even_digit (n / 10 % 10)) ∧
  (is_even_digit (n % 10)) ∧
  (n / 100 % 10 ≠ 0)

theorem num_of_distinct_three_digit_integers_with_even_digits : 
  {n : ℕ | valid_three_digit_integer n}.finite.to_finset.card = 100 :=
sorry

end num_of_distinct_three_digit_integers_with_even_digits_l167_167449


namespace angle_ACD_eq_90_l167_167867

theorem angle_ACD_eq_90 
  (A B C D X Y Z : Type)
  [Points A B C D X Y Z]
  (h1 : Midpoint X A B)
  (h2 : Midpoint Y A D)
  (h3 : Midpoint Z B C)
  (h4 : Perpendicular XY AB)
  (h5 : Perpendicular YZ BC)
  (h6 : Angle ABC = 100) :
  Angle ACD = 90 :=
sorry

end angle_ACD_eq_90_l167_167867


namespace fortieth_term_of_sequence_l167_167968

-- Definition of the sequence based on the conditions
def is_valid_term (n : ℕ) : Prop :=
  n % 3 = 0 ∧ ∃ (d : ℕ), d ∈ List.ofString (n.repr) ∧ d = 2

def valid_terms : List ℕ :=
  List.filter is_valid_term (List.range 1000)  -- Assuming a check up to 999

-- Main theorem statement
theorem fortieth_term_of_sequence : valid_terms.nth 39 = some 210 :=
by {
  sorry -- Proof is skipped as per instructions
}

end fortieth_term_of_sequence_l167_167968


namespace distinct_three_digit_numbers_with_even_digits_l167_167462

theorem distinct_three_digit_numbers_with_even_digits : 
  let even_digits := {0, 2, 4, 6, 8} in
  (∃ (hundreds options : Finset ℕ) (x : ℕ), 
    hundreds = {2, 4, 6, 8} ∧ 
    options = even_digits ∧ 
    x = Finset.card hundreds * Finset.card options * Finset.card options ∧ 
    x = 100) :=
by
  let even_digits := {0, 2, 4, 6, 8}
  exact ⟨{2, 4, 6, 8}, even_digits, 100, rfl, rfl, sorry, rfl⟩

end distinct_three_digit_numbers_with_even_digits_l167_167462


namespace sufficiency_nessessity_l167_167220

-- Definitions of arithmetic sequences and their common elements.
def arithmetic_seq (a d : ℕ) : ℕ → ℕ := λ n, a + n * d

def A (a d : ℕ) : Set ℕ := { x | ∃ n, x = arithmetic_seq a d n }
def B (a d : ℕ) : Set ℕ := { x | ∃ n, x = arithmetic_seq a d n }
def C (a d1 d2 : ℕ) : Set ℕ := A a d1 ∩ B a d2

-- The statement of the problem.
theorem sufficiency_nessessity (a d1 d2 : ℕ) (h1 : 0 < d1) (h2 : 0 < d2) :
  ( ∀ k ∈ C a d1 d2, ∃ m, k = a + m * Nat.lcm d1 d2 )
  ∧
  ¬(a = ∃ a', a ∈ C a' d1 d2)
  ∧
  ( ∀ a' d1' d2', ¬(a' = a) → (∃ k ∈ C a' d1' d2', is_arith_seq C a' d1' d2') )
sorry

end sufficiency_nessessity_l167_167220


namespace average_monthly_balance_l167_167714

def january_balance : ℕ := 150
def february_balance : ℕ := 300
def march_balance : ℕ := 450
def april_balance : ℕ := 300
def number_of_months : ℕ := 4

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance) / number_of_months = 300 := by
  sorry

end average_monthly_balance_l167_167714


namespace probability_not_same_color_shirt_l167_167582

theorem probability_not_same_color_shirt :
  let colors := 5,
      total_outcomes := colors * colors * colors,
      same_color_outcomes := colors
  in (total_outcomes - same_color_outcomes) / total_outcomes = 24 / 25 :=
by
  let colors := 5
  let total_outcomes := colors * colors * colors
  let same_color_outcomes := colors
  have total_outcomes := 125  -- 5 * 5 * 5
  have same_color_outcomes := 5  -- 5 colors where all wear the same one
  have remaining_outcomes := total_outcomes - same_color_outcomes  -- 125 - 5
  have probability := remaining_outcomes / total_outcomes
  show (120 / 125 = 24 / 25)
  sorry

end probability_not_same_color_shirt_l167_167582


namespace determine_m_value_l167_167804

theorem determine_m_value 
  (a b m : ℝ)
  (h1 : 2^a = m)
  (h2 : 5^b = m)
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := 
sorry

end determine_m_value_l167_167804


namespace fixed_point_PQ_passes_l167_167095

theorem fixed_point_PQ_passes (P Q : ℝ × ℝ) (x1 x2 : ℝ)
  (hP : P = (x1, x1^2))
  (hQ : Q = (x2, x2^2))
  (hC1 : x1 ≠ 0)
  (hC2 : x2 ≠ 0)
  (hSlopes : (x2 / x2^2 * (2 * x1)) = -2) :
  ∃ D : ℝ × ℝ, D = (0, 1) ∧
    ∀ (x y : ℝ), (y = x1^2 + (x1 - (1 / x1)) * (x - x1)) → ((x, y) = P ∨ (x, y) = Q) := sorry

end fixed_point_PQ_passes_l167_167095


namespace max_balls_in_cube_l167_167654

noncomputable def volume_of_cube : ℝ := (5 : ℝ)^3

noncomputable def volume_of_ball : ℝ := (4 / 3) * Real.pi * (1 : ℝ)^3

theorem max_balls_in_cube (c_length : ℝ) (b_radius : ℝ) (h1 : c_length = 5)
  (h2 : b_radius = 1) : 
  ⌊volume_of_cube / volume_of_ball⌋ = 29 := 
by
  sorry

end max_balls_in_cube_l167_167654


namespace harold_final_remaining_money_l167_167445

def harold_monthly_income : ℝ := 2500.00
def rent : ℝ := 700.00
def car_payment : ℝ := 300.00
def utilities_cost (car_payment : ℝ) : ℝ := car_payment / 2
def groceries : ℝ := 50.00
def total_expenses (rent car_payment utilities_cost groceries : ℝ) : ℝ :=
  rent + car_payment + utilities_cost + groceries
def remaining_money (income total_expenses : ℝ) : ℝ := income - total_expenses
def retirement_savings (remaining_money : ℝ) : ℝ := remaining_money / 2
def final_remaining (remaining_money retirement_savings : ℝ) : ℝ :=
  remaining_money - retirement_savings

theorem harold_final_remaining_money :
  final_remaining (remaining_money harold_monthly_income (total_expenses rent car_payment (utilities_cost car_payment) groceries))
         (retirement_savings (remaining_money harold_monthly_income (total_expenses rent car_payment (utilities_cost car_payment) groceries))) = 650.00 :=
by
  sorry

end harold_final_remaining_money_l167_167445


namespace log3_1_over_81_l167_167361

theorem log3_1_over_81 : log 3 (1 / 81) = -4 := by
  have h1 : 1 / 81 = 3 ^ (-4) := by
    -- provide a proof or skip with "sory"
    sorry
  have h2 : log 3 (3 ^ (-4)) = -4 := by
    -- provide a proof or skip with "sorry"
    sorry
  exact eq.trans (log 3) (congr_fun (h1.symm h2))

end log3_1_over_81_l167_167361


namespace average_weight_of_arun_l167_167682

variable (weight : ℝ)

def arun_constraint := 61 < weight ∧ weight < 72
def brother_constraint := 60 < weight ∧ weight < 70
def mother_constraint := weight ≤ 64
def father_constraint := 62 < weight ∧ weight < 73
def sister_constraint := 59 < weight ∧ weight < 68

theorem average_weight_of_arun : 
  (∃ w : ℝ, arun_constraint w ∧ brother_constraint w ∧ mother_constraint w ∧ father_constraint w ∧ sister_constraint w) →
  (63.5 = (63 + 64) / 2) := 
by
  sorry

end average_weight_of_arun_l167_167682


namespace percentage_loss_of_person_l167_167707

theorem percentage_loss_of_person :
  let cost_bowls := 115 * 18
  let discount_bowls := 0.10 * cost_bowls
  let cost_bowls_after_discount := cost_bowls - discount_bowls
  let cost_coasters := 20 * 12
  let total_cost_before_tax := cost_bowls_after_discount + cost_coasters
  let sales_tax := 0.12 * total_cost_before_tax
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  let revenue_bowls := 104 * 20
  let revenue_coasters := 15 * 15
  let total_revenue := revenue_bowls + revenue_coasters
  let net_gain_loss := total_revenue - total_cost_after_tax
  let percentage_loss := (net_gain_loss / total_cost_after_tax) * 100
  percentage_loss ≈ -2.14 :=
by
  sorry

end percentage_loss_of_person_l167_167707


namespace distinct_three_digit_even_integers_count_l167_167464

theorem distinct_three_digit_even_integers_count : 
  let even_digits := {0, 2, 4, 6, 8}
  ∃ h : Finset ℕ, h = {2, 4, 6, 8} ∧ 
     (∏ x in h, 5 * 5) = 100 :=
by
  let even_digits : Finset ℕ := {0, 2, 4, 6, 8}
  let h : Finset ℕ := {2, 4, 6, 8}
  have : ∏ x in h, 5 * 5 = 100 := sorry
  exact ⟨h, rfl, this⟩

end distinct_three_digit_even_integers_count_l167_167464


namespace transformed_data_average_and_variance_l167_167485

variables (n : ℕ) (a : ℕ → ℝ)
hypothesis (h_avg : (∑ i in finset.range n, a i) / n = 10)
hypothesis (h_var : (∑ i in finset.range n, (a i - 10) ^ 2) / n = 4)

theorem transformed_data_average_and_variance :
  (∑ i in finset.range n, (2 * a i + 3)) / n = 23 ∧
  (∑ i in finset.range n, ((2 * a i + 3) - 23) ^ 2) / n = 16 :=
sorry

end transformed_data_average_and_variance_l167_167485


namespace proof_problem_l167_167051

open Set

variable (U : Set ℕ)
variable (P : Set ℕ)
variable (Q : Set ℕ)

noncomputable def problem_statement : Set ℕ :=
  compl (P ∪ Q) ∩ U

theorem proof_problem :
  U = {1, 2, 3, 4} →
  P = {1, 2} →
  Q = {2, 3} →
  compl (P ∪ Q) ∩ U = {4} :=
by
  intros hU hP hQ
  rw [hU, hP, hQ]
  sorry

end proof_problem_l167_167051


namespace david_marks_in_physics_l167_167336

def marks_in_physics (marks_eng marks_math marks_chem marks_biol : ℕ) (avg_marks : ℚ) : ℕ :=
  let total_marks := avg_marks * 5
  in total_marks - (marks_eng + marks_math + marks_chem + marks_biol)

theorem david_marks_in_physics :
  let marks_eng := 70
  let marks_math := 60
  let marks_chem := 60
  let marks_biol := 65
  let avg_marks := 66.6
  marks_in_physics marks_eng marks_math marks_chem marks_biol avg_marks = 78 :=
by
  -- Use the conditions:
  -- marks_eng = 70, marks_math = 60, marks_chem = 60, marks_biol = 65, avg_marks = 66.6
  -- Therefore, marks_in_physics = ?
  sorry

end david_marks_in_physics_l167_167336


namespace product_of_primes_l167_167659

theorem product_of_primes :
  (7 * 97 * 89) = 60431 :=
by
  sorry

end product_of_primes_l167_167659


namespace find_r_minus_p_l167_167152

-- Define the variables and conditions
variables (p q r A1 A2 : ℝ)
noncomputable def arithmetic_mean (x y : ℝ) := (x + y) / 2

-- Given conditions in the problem
axiom hA1 : arithmetic_mean p q = 10
axiom hA2 : arithmetic_mean q r = 25

-- Statement to prove
theorem find_r_minus_p : r - p = 30 :=
by {
  -- write the necessary proof steps here
  sorry
}

end find_r_minus_p_l167_167152


namespace pizza_percentage_increase_l167_167708

def area (r : ℝ) : ℝ := π * r^2

theorem pizza_percentage_increase :
  let R1 := 5
  let R2 := 2
  let A1 := area R1
  let A2 := area R2
  let ΔA := A1 - A2
  let percentage_increase := (ΔA / A2) * 100
  percentage_increase = 525 :=
by
  sorry

end pizza_percentage_increase_l167_167708


namespace four_digit_combinations_total_four_digit_odd_combinations_four_digit_even_combinations_four_digit_greater_than_2000_combinations_l167_167777

theorem four_digit_combinations_total : ∃ L : List ℕ, L.length = 24 ∧ L.nodup ∧ 
  (∀ n ∈ L, (1 ≤ n / 1000) ∧ (n / 1000 ≤ 5) ∧ ((n / 100) % 10 ≠ n / 1000) ∧ ((n / 10) % 10 ≠ n / 100) ∧ (n % 10 ≠ n / 10 % 10)) := sorry

theorem four_digit_odd_combinations : ∃ L : List ℕ, L.length = 18 ∧ L.nodup ∧ 
  (∀ n ∈ L, n % 2 = 1 ∧ (1 ≤ n / 1000) ∧ (n / 1000 ≤ 5) ∧ ((n / 100) % 10 ≠ n / 1000) ∧ ((n / 10) % 10 ≠ n / 100) ∧ (n % 10 ≠ n / 10 % 10)) := sorry

theorem four_digit_even_combinations : ∃ L : List ℕ, L.length = 6 ∧ L.nodup ∧ 
  (∀ n ∈ L, n % 2 = 0 ∧ (1 ≤ n / 1000) ∧ (n / 1000 ≤ 5) ∧ ((n / 100) % 10 ≠ n / 1000) ∧ ((n / 10) % 10 ≠ n / 100) ∧ (n % 10 ≠ n / 10 % 10)) := sorry

theorem four_digit_greater_than_2000_combinations : ∃ L : List ℕ, L.length = 18 ∧ L.nodup ∧ 
  (∀ n ∈ L, n ≥ 2000 ∧ (1 ≤ n / 1000) ∧ (n / 1000 ≤ 5) ∧ ((n / 100) % 10 ≠ n / 1000) ∧ ((n / 10) % 10 ≠ n / 100) ∧ (n % 10 ≠ n / 10 % 10)) := sorry

end four_digit_combinations_total_four_digit_odd_combinations_four_digit_even_combinations_four_digit_greater_than_2000_combinations_l167_167777


namespace anne_clean_house_in_12_hours_l167_167246

theorem anne_clean_house_in_12_hours (B A : ℝ) (h1 : 4 * (B + A) = 1) (h2 : 3 * (B + 2 * A) = 1) : A = 1 / 12 ∧ (1 / A) = 12 :=
by
  -- We will leave the proof as a placeholder
  sorry

end anne_clean_house_in_12_hours_l167_167246


namespace min_dist_l167_167809

noncomputable def parabola_focus : Point := ⟨1, 0⟩ -- Focus of y^2 = 4x.

structure Point where
  x : ℝ
  y : ℝ

def parabola (M : Point) : Prop :=
  M.y ^ 2 = 4 * M.x

def distance (A B : Point) : ℝ :=
  real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2)

def is_projection (D M : Point) : Prop :=
  D.x = -1 ∧ D.y = M.y -- Directrix x = -1

def collinear (A B C : Point) : Prop :=
  (B.y - A.y) * (C.x - A.x) = (C.y - A.y) * (B.x - A.x)

theorem min_dist (M : Point)
  (h_parabola : parabola M)
  (P : Point)
  (hP : P = ⟨4, 1⟩)
  (F : Point)
  (hF : F = parabola_focus) :
  ∃ D : Point, is_projection D M ∧ collinear D M P ∧ distance M P + distance M F = 6 :=
sorry

end min_dist_l167_167809


namespace sum_of_squares_formula_l167_167879

theorem sum_of_squares_formula (a b : ℕ) (h₁ : a = 1) (h₂ : b = 4) :
    ∀ n : ℕ, 0 < n → (∑ i in finset.range n.succ, (i + 1)^2 / ((2 * (i + 1) - 1) * (2 * (i + 1) + 1)) = (a * n^2 + n) / (b * n + 2)) :=
by
  sorry

end sum_of_squares_formula_l167_167879


namespace largest_is_D_l167_167208

-- Definitions based on conditions
def A : ℕ := 27
def B : ℕ := A + 7
def C : ℕ := B - 9
def D : ℕ := 2 * C

-- Theorem stating D is the largest
theorem largest_is_D : D = max (max A B) (max C D) :=
by
  -- Inserting sorry because the proof is not required.
  sorry

end largest_is_D_l167_167208


namespace find_k_l167_167815

theorem find_k (k : ℝ) : 
  (1 : ℝ)^2 + k * 1 - 3 = 0 → k = 2 :=
by
  intro h
  sorry

end find_k_l167_167815


namespace total_tips_fraction_l167_167250

variables {A : ℚ} -- average monthly tips in other months

theorem total_tips_fraction (A : ℚ) :
  let august_tips := 8 * A in
  let other_months_tips := 6 * A in
  let total_tips := other_months_tips + august_tips + A in
  august_tips / total_tips = 8 / 15 :=
by
  sorry

end total_tips_fraction_l167_167250


namespace countable_finite_sequences_l167_167927

open Set Function

noncomputable def finite_sequences : Set (List ℕ) := { l : List ℕ | true }

theorem countable_finite_sequences :
  Countable (finite_sequences) :=
begin
  -- Given conditions:
  -- 1. Pairs of natural numbers are countable.
  have h_pairs : Countable (ℕ × ℕ) := by sorry,
  
  -- 2. Finite sequences of fixed length k are countable.
  have h_fixed_length : ∀ (k : ℕ), Countable ({ l : List ℕ | l.length = k }) := by sorry,
  
  -- 3. Countable union of countable sets is countable.
  have h_countable_union : ∀ (S : Set (Set ℕ)), (∀ s ∈ S, Countable s) → Countable (⋃₀ S) := by sorry,

  -- Using these conditions, prove the set of all finite sequences is countable.
  sorry
end

end countable_finite_sequences_l167_167927


namespace possible_values_quotient_l167_167287

theorem possible_values_quotient (α : ℝ) (h_pos : α > 0) (h_rounded : ∃ (n : ℕ) (α1 : ℝ), α = n / 100 + α1 ∧ 0 ≤ α1 ∧ α1 < 1 / 100) :
  ∃ (values : List ℝ), values = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
                                  0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                                  0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
                                  0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                                  0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
                                  1.00] :=
  sorry

end possible_values_quotient_l167_167287


namespace second_athlete_high_jump_eq_eight_l167_167218

theorem second_athlete_high_jump_eq_eight :
  let first_athlete_long_jump := 26
  let first_athlete_triple_jump := 30
  let first_athlete_high_jump := 7
  let second_athlete_long_jump := 24
  let second_athlete_triple_jump := 34
  let winner_average_jump := 22
  (first_athlete_long_jump + first_athlete_triple_jump + first_athlete_high_jump) / 3 < winner_average_jump →
  ∃ (second_athlete_high_jump : ℝ), 
    second_athlete_high_jump = 
    (winner_average_jump * 3 - (second_athlete_long_jump + second_athlete_triple_jump)) ∧ 
    second_athlete_high_jump = 8 :=
by
  intros 
  sorry

end second_athlete_high_jump_eq_eight_l167_167218


namespace length_BI_l167_167216

-- Hypothesis definitions
def right_triangle (A B C : Type) (AB AC : ℝ) (∠BAC : ℝ) : Prop := 
  AB = 4 ∧ AC = 3 ∧ ∠BAC = 90

def midpoint (M B C : Type) (BC : ℝ) : Prop :=
  BC = real.sqrt(4^2 + 3^2) ∧ BC = 5 ∧ (2 * M = B + C)

def cyclic_quadrilateral (A I M E : Type) : Prop :=
  ∃ (AI AE : ℝ), AI > AE ∧ ∃(area : ℝ), area_triangle(M I E) = 3 / 2

-- The target statement to prove
theorem length_BI (a b c : ℕ) : 
  ∃ (A B C M I E : Type), right_triangle A B C 4 3 90 ∧ midpoint M B C 5 ∧ cyclic_quadrilateral A I M E ∧
  let BI := real.sqrt(16 - (real.sqrt(37) / 2)^2) in
  BI = (3 - real.sqrt(3)) / 2 ∧ a + b + c = 48 :=
sorry

end length_BI_l167_167216


namespace special_hash_calculation_l167_167618

-- Definition of the operation #
def special_hash (a b : ℤ) : ℚ := 2 * a + (a / b) + 3

-- Statement of the proof problem
theorem special_hash_calculation : special_hash 7 3 = 19 + 1/3 := 
by 
  sorry

end special_hash_calculation_l167_167618


namespace smallest_n_in_geometric_sequence_l167_167094

theorem smallest_n_in_geometric_sequence (a1 q : ℕ) (S : ℕ → ℕ) (h_a1 : a1 = 3) 
(h_q : q = 2) (h_S : ∀ n, S n = a1 * (q^n - 1)) :
  ∃ n : ℕ, S n > 1000 ∧ ∀ m : ℕ, S m > 1000 → n ≤ m :=
begin
  sorry
end

end smallest_n_in_geometric_sequence_l167_167094


namespace problem_solved_by_half_trainees_l167_167856

theorem problem_solved_by_half_trainees 
  (n m : ℕ) (hn_pos : 0 < n)
  (h_trainee_solved : ∀ t : ℕ, t < m → ∑ p in finset.range n, ite (solved p t) 1 0 ≥ n / 2) :
  ∃ p, ∑ t in finset.range m, ite (solved p t) 1 0 ≥ m / 2 :=
by
  sorry


end problem_solved_by_half_trainees_l167_167856


namespace determine_angle_XZY_l167_167878

variable (X Y Z P Q R : Point)
variable (θ : ℝ)

-- Define the conditions
def XY_eq_3XZ (X Y Z : Point) : Prop := dist X Y = 3 * dist X Z
def angle_XPQ_eq_angle_ZQP (X P Q Z : Point) : Prop := ∠ X P Q = ∠ Z Q P
def triangle_ZQR_equilateral (Z Q R : Point) : Prop := 
  dist Z Q = dist Q R ∧ dist Q R = dist R Z ∧ dist R Z = dist Z Q 

-- Main theorem statement
theorem determine_angle_XZY 
  (h1 : XY_eq_3XZ X Y Z)
  (h2 : angle_XPQ_eq_angle_ZQP X P Q Z)
  (h3 : triangle_ZQR_equilateral Z Q R) : 
  ∠ X Z Y = 30 :=
sorry

end determine_angle_XZY_l167_167878


namespace solve_problem_l167_167030

noncomputable def proof_problem (x1 x2 A : ℂ) [ne : x1 ≠ x2] : Prop :=
  (x1 * (x1 + 1) = A) ∧
  (x2 * (x2 + 1) = A) ∧
  (x1^4 + 3 * x1^3 + 5 * x1 = x2^4 + 3 * x2^3 + 5 * x2) →
  A = -7

theorem solve_problem (x1 x2 : ℂ) [ne : x1 ≠ x2] (A : ℂ) : proof_problem x1 x2 A :=
by {
  sorry
}

end solve_problem_l167_167030


namespace simple_interest_is_correct_l167_167247

def Principal : ℝ := 10000
def Rate : ℝ := 0.09
def Time : ℝ := 1

theorem simple_interest_is_correct :
  Principal * Rate * Time = 900 := by
  sorry

end simple_interest_is_correct_l167_167247


namespace probability_correct_l167_167997

-- Define the sum of two 8-sided dice
def sum_of_two_8sided_dice (dice1 dice2 : ℕ) : ℕ := dice1 + dice2

-- Define the condition for the area of the circle being less than the circumference
def area_less_than_circumference (d : ℕ) : Prop :=
  d > 0 ∧ d < 4

-- Define the probability that the sum of two 8-sided dice determines a circle's diameter such that the area is less than the circumference
def probability_area_less_than_circumference : ℚ :=
  ∑ i in finset.range 9, ∑ j in finset.range 9, if area_less_than_circumference (sum_of_two_8sided_dice i j) then 1 else 0

-- State the theorem
theorem probability_correct :
  probability_area_less_than_circumference = 3 / 64 :=
sorry

end probability_correct_l167_167997


namespace exists_infinite_B_with_property_l167_167560

-- Definition of the sequence A
def seqA (n : ℕ) : ℤ := 5 * n - 2

-- Definition of the sequence B with its general form
def seqB (k : ℕ) (d : ℤ) : ℤ := k * d + 7 - d

-- The proof problem statement
theorem exists_infinite_B_with_property :
  ∃ (B : ℕ → ℤ) (d : ℤ), B 1 = 7 ∧ 
  (∀ k, k > 1 → B k = B (k - 1) + d) ∧
  (∀ n : ℕ, ∃ (k : ℕ), seqB k d = seqA n) :=
sorry

end exists_infinite_B_with_property_l167_167560


namespace krakozyabrs_count_l167_167107

theorem krakozyabrs_count :
  ∀ (K : Type) [fintype K] (has_horns : K → Prop) (has_wings : K → Prop), 
  (∀ k, has_horns k ∨ has_wings k) →
  (∃ n, (∀ k, has_horns k → has_wings k) → fintype.card ({k | has_horns k} : set K) = 5 * n) →
  (∃ n, (∀ k, has_wings k → has_horns k) → fintype.card ({k | has_wings k} : set K) = 4 * n) →
  (∃ T, 25 < T ∧ T < 35 ∧ T = 32) := 
by
  intros K _ has_horns has_wings _ _ _
  sorry

end krakozyabrs_count_l167_167107


namespace sum_infinite_geometric_l167_167795

theorem sum_infinite_geometric (a r : ℝ) (ha : a = 2) (hr : r = 1/3) : 
  ∑' n : ℕ, a * r^n = 3 := by
  sorry

end sum_infinite_geometric_l167_167795


namespace roots_difference_squared_l167_167539

-- Defining the solutions to the quadratic equation
def quadratic_equation_roots (a b : ℚ) : Prop :=
  (2 * a^2 - 7 * a + 6 = 0) ∧ (2 * b^2 - 7 * b + 6 = 0)

-- The main theorem we aim to prove
theorem roots_difference_squared (a b : ℚ) (h : quadratic_equation_roots a b) :
    (a - b)^2 = 1 / 4 := 
  sorry

end roots_difference_squared_l167_167539


namespace subcommittee_with_at_least_two_teachers_l167_167203

-- Definitions based on the problem conditions
def total_committee_members : ℕ := 12
def total_teachers : ℕ := 5
def total_non_teachers : ℕ := total_committee_members - total_teachers
def subcommittee_size : ℕ := 5

-- Main theorem to be proved
theorem subcommittee_with_at_least_two_teachers : 
  (∃ (count : ℕ), (count = @nat.choose total_committee_members subcommittee_size - 
    (nat.choose total_non_teachers subcommittee_size + 
    (nat.choose total_teachers 1 * nat.choose total_non_teachers (subcommittee_size - 1))) 
  ) ∧ count = 596) :=
by {
  sorry,
}

end subcommittee_with_at_least_two_teachers_l167_167203


namespace problem_statement_l167_167017

variable (P : Prop) (Q : Prop)

axiom P_def : P = (2 + 2 = 5)
axiom Q_def : Q = (3 > 2)

theorem problem_statement : (P ∨ Q) ∧ ¬(¬Q) :=
by
  rw [P_def, Q_def]
  -- we would continue the proof here usually
  sorry

end problem_statement_l167_167017


namespace minimize_sum_of_squares_l167_167745

open Real

-- Assume x, y are positive real numbers and x + y = s
variables {x y s : ℝ}
variables (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : x + y = s)

theorem minimize_sum_of_squares :
  (x = y) ∧ (2 * x * x = s * s / 2) → (x = s / 2 ∧ y = s / 2 ∧ x^2 + y^2 = s^2 / 2) :=
by
  sorry

end minimize_sum_of_squares_l167_167745


namespace num_repeating_decimals_l167_167385

theorem num_repeating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) : 
  (∑ k in Finset.filter (λ m, m ∈ finset.range(202) \ 
     {2,4,5,8,10,16,20,25,32,40,50,64,80,100,125,128,160,200}) 
     finset.univ, 1) = 182 := 
by 
  sorry

end num_repeating_decimals_l167_167385


namespace chord_length_consistent_l167_167821

noncomputable def ellipse_eqn : ℝ := 
  let a := 2
  let b := 1
  ∀ (x y : ℝ), \(\frac{x^2}{4} + y^2 = 1\)

noncomputable def midpoint : ℝ -> ℝ -> ℝ -> ℝ -> ℝ :=
  fun m n := (0, (2*n / (m + 2), 0))

theorem chord_length_consistent (b : ℝ) (m n : ℝ) (h₁ : m ≠ ±2) 
                             (h₂ : (m*n)^2 = 1 - \((m/2)^2\)) :
  let M := midpoint m (2 * n / (m + 2))
  let N := midpoint -m (2 * -n / (-m + 2))
  (chord_length (m,n) (M, N)) = 2 := sorry

end chord_length_consistent_l167_167821


namespace num_digits_for_2C4_multiple_of_4_l167_167003

theorem num_digits_for_2C4_multiple_of_4 : (finset.univ.filter (λ C : ℕ, (C * 10 + 4) % 4 = 0)).card = 5 :=
by
  -- The proof is omitted as we are only required to write the statement.
  sorry

end num_digits_for_2C4_multiple_of_4_l167_167003


namespace problem_part1_problem_part2_l167_167892

-- Define the conditions
def eq1 (a b : ℝ) : Prop := a + b = 0
def eq2 (a b : ℝ) : Prop := 2b * (Math.exp (a + b)) = -2
def volume_eq(a b : ℝ) : Prop :=
    a = 1 ∧ b = -1 ∧
    (∀ V : ℝ,
      V = 2 * Real.pi * (∫ x in 0..1, x * (Math.exp (1 - x^2) - x^2)) - (2 * Real.pi * (1/4))

theorem problem_part1 (a b : ℝ) : eq1 a b ∧ eq2 a b → a = 1 ∧ b = -1 :=
by 
  sorry

theorem problem_part2 : volume_eq 1 -1 :=
by 
  sorry


end problem_part1_problem_part2_l167_167892


namespace hexagon_area_l167_167528

variables {P A B C D E F G H I : Type}

-- Define the conditions
variables [regular_hexagon P A B C D E F]
variables [segment_division A B G (2/3)]
variables [segment_division D C H (1/3)]
variables [segment_division F E I (2/3)]
variables [triangle_area G H I 300]

-- Define the theorem statement
theorem hexagon_area
  (h1 : regular_hexagon P A B C D E F)
  (h2 : segment_division A B G (2/3))
  (h3 : segment_division D C H (1/3))
  (h4 : segment_division F E I (2/3))
  (h5 : triangle_area G H I 300) :
  hexagon_area P A B C D E F = 800 :=
sorry

end hexagon_area_l167_167528


namespace reflection_matrix_correct_l167_167533

-- Define the normal vector n
def n : ℝ^3 := ⟨2, -1, 1⟩

-- Define the reflection matrix S
def S : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    [-1/3, 4/3, 4/3],
    [4/3, 5/3, 7/3],
    [4/3, 4/3, 5/3]
  ]

-- Define the function that calculates the reflection of u through the plane Q
def reflection_through_plane (u : ℝ^3) : ℝ^3 := 
  let proj_n := (2 * u.1 - u.2 + u.3) / 6 * n
  let q := u - proj_n
  2 * q - u

-- Proposition that the reflection matrix S correctly reflects any vector through plane Q
theorem reflection_matrix_correct (u : ℝ^3) : S.mulVec u = reflection_through_plane u := by
  sorry

end reflection_matrix_correct_l167_167533


namespace f_neg_9_over_2_eq_neg_3_over_4_l167_167903

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1
then x * (1 + x)
else if 0 ≤ -x ∧ -x ≤ 1
then -((-x) * (1 + (-x)))
else f (x + 4 * if x < 0 then -1 else 1)

theorem f_neg_9_over_2_eq_neg_3_over_4 : f (-9 / 2) = -3 / 4 :=
sorry

end f_neg_9_over_2_eq_neg_3_over_4_l167_167903


namespace calculate_expr_at_3_l167_167729

-- Definition of the expression
def expr (x : ℕ) : ℕ := (x + x * x^(x^2)) * 3

-- The proof statement
theorem calculate_expr_at_3 : expr 3 = 177156 := 
by
  sorry

end calculate_expr_at_3_l167_167729


namespace greatest_b_not_in_range_l167_167653

theorem greatest_b_not_in_range (b : ℤ) : ∀ x : ℝ, ¬ (x^2 + (b : ℝ) * x + 20 = -9) ↔ b ≤ 10 :=
by
  sorry

end greatest_b_not_in_range_l167_167653


namespace gina_total_pay_l167_167391

noncomputable def gina_painting_pay : ℕ :=
let roses_per_hour := 6
let lilies_per_hour := 7
let rose_order := 6
let lily_order := 14
let pay_per_hour := 30

-- Calculate total time (in hours) Gina spends to complete the order
let time_for_roses := rose_order / roses_per_hour
let time_for_lilies := lily_order / lilies_per_hour
let total_time := time_for_roses + time_for_lilies

-- Calculate the total pay
let total_pay := total_time * pay_per_hour

total_pay

-- The theorem that Gina gets paid $90 for the order
theorem gina_total_pay : gina_painting_pay = 90 := by
  sorry

end gina_total_pay_l167_167391


namespace num_subsets_with_sum_mod_2006_l167_167374

open Set

-- Define the set S from 1 to 2005
def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2005}

-- Define a function that computes the sum of elements in a set modulo 2048
def sum_mod_2048 (B : Set ℕ) : ℕ :=
  (∑ x in B, x) % 2048

-- Define the target remainder
def target_remainder : ℕ := 2006

-- State the problem: the number of subsets B of S such that sum of elements in B % 2048 is 2006
theorem num_subsets_with_sum_mod_2006 : 
  (card {B | B ⊆ S ∧ sum_mod_2048 B = target_remainder}) = 2 ^ 1994 := 
by
  sorry

end num_subsets_with_sum_mod_2006_l167_167374


namespace sum_of_gray_areas_eq_half_area_l167_167756

theorem sum_of_gray_areas_eq_half_area (a : ℝ) (p q : ℝ) :
  let A := (-a, 0) 
  let B := (a, 0) 
  let C := (0, Real.sqrt 3 * a) 
  let P := (p, q)
  let area_triangle (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) := 
    1 / 2 * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂))
  let side_Area_triangle := area_triangle (-a) 0 p q x y + area_triangle p y a 0 q p + area_triangle a 0 0 (Real.sqrt 3 * a) p q in
  2 * side_Area_triangle (-a 0 a 0 0 (Real.sqrt 3 * a)) = area_triangle (-a) 0 a 0 p q := by
  sorry

end sum_of_gray_areas_eq_half_area_l167_167756


namespace count_valid_integers_l167_167473

theorem count_valid_integers : 
  (n : ℤ) (h1 : -30 < n^2) (h2: n^2 < 30) → 
  (finset.Icc (-5 : ℤ) 5).card = 11 := 
by
  sorry

end count_valid_integers_l167_167473


namespace P_is_sufficient_but_not_necessary_for_q_l167_167801

theorem P_is_sufficient_but_not_necessary_for_q
  (k : ℝ)
  (P : |k - 1/2| > 1/2)
  (q : ∀ x : ℝ, x^2 - 2*k*x + k > 0) :
  (P → ∀ x : ℝ, x^2 - 2*k*x + k > 0) ∧
  (q → ¬ P) :=
by
  sorry

end P_is_sufficient_but_not_necessary_for_q_l167_167801


namespace slope_at_two_l167_167813

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2
noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem slope_at_two (a b : ℝ) (h1 : f' 1 a b = 0) (h2 : f 1 a b = 10) :
  f' 2 4 (-11) = 17 :=
sorry

end slope_at_two_l167_167813


namespace find_x_l167_167200

theorem find_x (x : ℝ) (h : x - 1/10 = x / 10) : x = 1 / 9 := 
  sorry

end find_x_l167_167200


namespace magnitude_b_magnitude_c_area_l167_167075

-- Define the triangle ABC and parameters
variables {A B C : ℝ} {a b c : ℝ}
variables (A_pos : 0 < A) (A_lt_pi_div2 : A < Real.pi / 2)
variables (triangle_condition : a = Real.sqrt 15) (sin_A : Real.sin A = 1 / 4)

-- Problem 1
theorem magnitude_b (cos_B : Real.cos B = Real.sqrt 5 / 3) :
  b = (8 * Real.sqrt 15) / 3 := by
  sorry

-- Problem 2
theorem magnitude_c_area (b_eq_4a : b = 4 * a) :
  c = 15 ∧ (1 / 2 * b * c * Real.sin A = (15 / 2) * Real.sqrt 15) := by
  sorry

end magnitude_b_magnitude_c_area_l167_167075


namespace happy_children_count_l167_167569

theorem happy_children_count (total_children sad_children neither_children total_boys total_girls happy_boys sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : sad_children = 10)
  (h3 : neither_children = 20)
  (h4 : total_boys = 18)
  (h5 : total_girls = 42)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4) :
  ∃ happy_children, happy_children = 30 :=
  sorry

end happy_children_count_l167_167569


namespace triangle_perimeter_is_correct_l167_167034

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (S : ℝ)

def triangle_perimeter (a b c : ℝ) := a + b + c

theorem triangle_perimeter_is_correct :
  c = sqrt 7 → C = π / 3 → S = 3 * sqrt 3 / 2 →
  S = (1 / 2) * a * b * sin (C) → c^2 = a^2 + b^2 - 2 * a * b * cos (C) →
  ∃ a b : ℝ, triangle_perimeter a b c = 5 + sqrt 7 :=
  by
    intros h1 h2 h3 h4 h5
    sorry

end triangle_perimeter_is_correct_l167_167034


namespace log_base_3_of_reciprocal_81_l167_167354

theorem log_base_3_of_reciprocal_81 : log 3 (1 / 81) = -4 :=
by
  sorry

end log_base_3_of_reciprocal_81_l167_167354


namespace connect_paths_by_arc_l167_167221

noncomputable def feasible_connection (M N : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
∃ (C : ℝ × ℝ),
is_perpendicular_to M A ∧ is_perpendicular_to N B ∧
dist C A = dist C B

theorem connect_paths_by_arc (M N : ℝ → ℝ) (A B : ℝ × ℝ) :
  ¬ feasible_connection M N A B ∨
  (parallel M N ∧ common_perpendicular M N A B) :=
sorry

end connect_paths_by_arc_l167_167221


namespace min_distance_symmetry_l167_167610

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 + x + 1

def line (x y : ℝ) : Prop := 2 * x - y = 3

theorem min_distance_symmetry :
  ∀ (P Q : ℝ × ℝ),
    line P.1 P.2 → line Q.1 Q.2 →
    (exists (x : ℝ), P = (x, f x)) ∧
    (exists (x : ℝ), Q = (x, f x)) →
    ∃ (d : ℝ), d = 2 * Real.sqrt 5 :=
sorry

end min_distance_symmetry_l167_167610


namespace T_10_mod_5_eq_3_l167_167776

def a_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in A
sorry

def b_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in B
sorry

def c_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in C
sorry

def T (n : ℕ) : ℕ := -- Number of valid sequences of length n
  a_n n + b_n n

theorem T_10_mod_5_eq_3 :
  T 10 % 5 = 3 :=
sorry

end T_10_mod_5_eq_3_l167_167776


namespace center_value_in_array_l167_167693

theorem center_value_in_array : 
  ∀ (a1 a7 : ℕ), 
  (∀ n, ∃ (r1 r7 : list ℕ), 
    r1 = (list.range 7).map (λ k, a1 + k * 6) ∧
    r7 = (list.range 7).map (λ k, a7 + k * 8) ∧
    list.nth r1 6 = some 39 ∧
    list.nth r7 6 = some 58) →
  ∃ (Y : ℕ), Y = 27 :=
by
  sorry

end center_value_in_array_l167_167693


namespace min_time_one_ball_l167_167207

noncomputable def children_circle_min_time (n : ℕ) := 98

theorem min_time_one_ball (n : ℕ) (h1 : n = 99) : 
  children_circle_min_time n = 98 := 
by 
  sorry

end min_time_one_ball_l167_167207


namespace monotonic_f_l167_167825

def f (x a : ℝ) : ℝ := 3 * x - (a / x)

def is_symmetric (x0 y0 : ℝ) : Prop :=
  x0 + (4 + x0) = 0 ∧ y0 + (x0 + y0) = 0

theorem monotonic_f (a x0 y0 : ℝ) (h_symmetric : is_symmetric x0 y0) (h_eq : a = 14) :
  (∀ x : ℝ, x < 0 → ∀ y : ℝ, y < 0 → x < y → f x a < f y a) ∧ (∀ x : ℝ, x > 0 → ∀ y : ℝ, y > 0 → x < y → f x a < f y a) :=
by
  sorry

end monotonic_f_l167_167825


namespace sales_difference_l167_167260

-- Definitions of the conditions
def daily_avg_sales_pastries := 20 * 2
def daily_avg_sales_bread := 10 * 4
def daily_avg_sales := daily_avg_sales_pastries + daily_avg_sales_bread

def today_sales_pastries := 14 * 2
def today_sales_bread := 25 * 4
def today_sales := today_sales_pastries + today_sales_bread

-- Statement to be proved
theorem sales_difference : today_sales - daily_avg_sales = 48 :=
by {
  -- Unpack the definitions
  simp [daily_avg_sales_pastries, daily_avg_sales_bread, daily_avg_sales],
  simp [today_sales_pastries, today_sales_bread, today_sales],
  -- Computation,
  -- daily_avg_sales == 20 * 2 + 10 * 4 == 80,
  -- today_sales == 14 * 2 + 25 * 4 == 128
  -- therefore, 128 - 80 == 48,
  -- QED.
  sorry
}

end sales_difference_l167_167260


namespace harold_final_remaining_money_l167_167444

def harold_monthly_income : ℝ := 2500.00
def rent : ℝ := 700.00
def car_payment : ℝ := 300.00
def utilities_cost (car_payment : ℝ) : ℝ := car_payment / 2
def groceries : ℝ := 50.00
def total_expenses (rent car_payment utilities_cost groceries : ℝ) : ℝ :=
  rent + car_payment + utilities_cost + groceries
def remaining_money (income total_expenses : ℝ) : ℝ := income - total_expenses
def retirement_savings (remaining_money : ℝ) : ℝ := remaining_money / 2
def final_remaining (remaining_money retirement_savings : ℝ) : ℝ :=
  remaining_money - retirement_savings

theorem harold_final_remaining_money :
  final_remaining (remaining_money harold_monthly_income (total_expenses rent car_payment (utilities_cost car_payment) groceries))
         (retirement_savings (remaining_money harold_monthly_income (total_expenses rent car_payment (utilities_cost car_payment) groceries))) = 650.00 :=
by
  sorry

end harold_final_remaining_money_l167_167444


namespace find_f_e_l167_167785

noncomputable def f (x : ℝ) : ℝ := f' 1 + x * Real.log x

theorem find_f_e : f e = 1 + e :=
  sorry

end find_f_e_l167_167785


namespace log3_1_over_81_l167_167362

theorem log3_1_over_81 : log 3 (1 / 81) = -4 := by
  have h1 : 1 / 81 = 3 ^ (-4) := by
    -- provide a proof or skip with "sory"
    sorry
  have h2 : log 3 (3 ^ (-4)) = -4 := by
    -- provide a proof or skip with "sorry"
    sorry
  exact eq.trans (log 3) (congr_fun (h1.symm h2))

end log3_1_over_81_l167_167362


namespace miles_driven_before_gas_l167_167568

variable (totalDistance remainingDistance drivenBeforeGas : ℕ)
variable (h_totalDistance : totalDistance = 78)
variable (h_remainingDistance : remainingDistance = 46)
variable (h_equation : drivenBeforeGas = totalDistance - remainingDistance)

theorem miles_driven_before_gas : drivenBeforeGas = 32 := by
  rw [h_totalDistance, h_remainingDistance, h_equation]
  rw [h_totalDistance, h_remainingDistance] at h_equation
  have : drivenBeforeGas = 78 - 46 := by rw [h_totalDistance, h_remainingDistance]
  exact h_equation.trans this

end miles_driven_before_gas_l167_167568


namespace animal_population_l167_167974

def total_population (L P E : ℕ) : ℕ :=
L + P + E

theorem animal_population 
    (L P E : ℕ) 
    (h1 : L = 2 * P) 
    (h2 : E = (L + P) / 2) 
    (h3 : L = 200) : 
  total_population L P E = 450 := 
  by 
    sorry

end animal_population_l167_167974


namespace forest_segment_proportion_l167_167251

-- Definition: A type to represent the modulo 3 results
inductive Mod3
| Zero
| One
| Two

-- Function to calculate the modulo 3 result
def mod3 (n : Nat) : Mod3 :=
  match n % 3 with
  | 0 => Mod3.Zero
  | 1 => Mod3.One
  | 2 => Mod3.Two
  | _ => panic! "Unreachable"

-- Function to determine the forest portion based on n
def forest_portion (n : Nat) : ℚ :=
  match mod3 n with
  | Mod3.Zero => 0
  | Mod3.One  => 2 / 3
  | Mod3.Two  => 1 / 3

-- Proof placeholder
theorem forest_segment_proportion (n : Nat) : forest_portion n = 
  if n % 3 = 0 then 0
  else if n % 3 = 1 then 2 / 3
  else if n % 3 = 2 then 1 / 3
  else 0 :=
by sorry


end forest_segment_proportion_l167_167251


namespace coefficient_of_x4_in_expansion_l167_167870

theorem coefficient_of_x4_in_expansion :
  (nat.choose 10 3) * ((-1 / (2 : ℝ)))^3 * 1 = -15 := by 
sorry

end coefficient_of_x4_in_expansion_l167_167870


namespace krakozyabrs_proof_l167_167115

-- Defining necessary variables and conditions
variable {n : ℕ}

-- Conditions given in the problem
def condition1 (H : ℕ) : Prop := 0.2 * H = n
def condition2 (W : ℕ) : Prop := 0.25 * W = n

-- Definition using the principle of inclusion-exclusion
def total_krakozyabrs (H W : ℕ) : ℕ := H + W - n

-- Statements reflecting the conditions and the final conclusion
theorem krakozyabrs_proof (H W : ℕ) (n : ℕ) : condition1 H → condition2 W → (25 < total_krakozyabrs H W) ∧ (total_krakozyabrs H W < 35) → total_krakozyabrs H W = 32 :=
by
  -- We skip the proof here
  sorry

end krakozyabrs_proof_l167_167115


namespace sales_difference_l167_167262

-- Definitions of the conditions
def daily_avg_sales_pastries := 20 * 2
def daily_avg_sales_bread := 10 * 4
def daily_avg_sales := daily_avg_sales_pastries + daily_avg_sales_bread

def today_sales_pastries := 14 * 2
def today_sales_bread := 25 * 4
def today_sales := today_sales_pastries + today_sales_bread

-- Statement to be proved
theorem sales_difference : today_sales - daily_avg_sales = 48 :=
by {
  -- Unpack the definitions
  simp [daily_avg_sales_pastries, daily_avg_sales_bread, daily_avg_sales],
  simp [today_sales_pastries, today_sales_bread, today_sales],
  -- Computation,
  -- daily_avg_sales == 20 * 2 + 10 * 4 == 80,
  -- today_sales == 14 * 2 + 25 * 4 == 128
  -- therefore, 128 - 80 == 48,
  -- QED.
  sorry
}

end sales_difference_l167_167262


namespace count_squarish_is_zero_l167_167275

/-
  A five-digit number (base 8) is squarish if it satisfies the following conditions:
  (i) none of its digits are zero;
  (ii) it is a perfect square;
  (iii) the first two digits, the middle digit, and the last two digits
  of the number are all perfect squares when considered as two-digit or one-digit numbers in base 8;
  (iv) the sum of all its digits is a perfect square.
  -/
def is_squarish (N : ℕ) : Prop :=
  let digits := List.take 5 (nat.digits 8 N) in
  (N < 8^5) ∧
  (∀ d ∈ digits, d ≠ 0) ∧
  (∃ x, x^2 = N) ∧
  (∃ a b c d e, digits = [a, b, c, d, e] ∧
    (∃ y, y^2 = a) ∧ 
    (∃ z, z^2 = c) ∧ 
    (∃ w, w^2 = e) ∧ 
    (b < 8) ∧ 
    (d < 8) ∧
    (∃ u, u^2 = (a + b + c + d + e))
  )

theorem count_squarish_is_zero : finset.card (finset.filter is_squarish (finset.range (8^5))) = 0 :=
  sorry

end count_squarish_is_zero_l167_167275


namespace krakozyabrs_proof_l167_167116

-- Defining necessary variables and conditions
variable {n : ℕ}

-- Conditions given in the problem
def condition1 (H : ℕ) : Prop := 0.2 * H = n
def condition2 (W : ℕ) : Prop := 0.25 * W = n

-- Definition using the principle of inclusion-exclusion
def total_krakozyabrs (H W : ℕ) : ℕ := H + W - n

-- Statements reflecting the conditions and the final conclusion
theorem krakozyabrs_proof (H W : ℕ) (n : ℕ) : condition1 H → condition2 W → (25 < total_krakozyabrs H W) ∧ (total_krakozyabrs H W < 35) → total_krakozyabrs H W = 32 :=
by
  -- We skip the proof here
  sorry

end krakozyabrs_proof_l167_167116


namespace crosses_4x4_grid_l167_167076

theorem crosses_4x4_grid :
  ∃ (ways : ℕ), 
    ways = 432 ∧ 
    (∀ (g : ℕ × ℕ → bool),
      (∑ i in finRange 4, ∑ j in finRange 4, if g (i, j) then 1 else 0) = 5 → 
      ∀ i ∈ finRange 4, ∃ j ∈ finRange 4, g (i, j) = true →
      ∀ j ∈ finRange 4, ∃ i ∈ finRange 4, g (i, j) = true) := sorry

end crosses_4x4_grid_l167_167076


namespace single_elimination_games_l167_167085

theorem single_elimination_games (players : ℕ) (h : players = 512) : 
  (games_needed : ℕ) :=
  sorry

end single_elimination_games_l167_167085


namespace total_passengers_l167_167633

variable (P : ℕ) (men women : ℕ)

-- Conditions
def two_thirds_women (P : ℕ) : Prop := women = (2 * P) / 3
def one_eighth_men_standing (men : ℕ) : Prop := men_standing = men / 8
def fourteen_seated_men (men : ℕ) : Prop := men_seated = 14

-- Sum constraints
def men_eq_one_third_passengers (P men : ℕ) : Prop := men = P / 3
def men_seating_relation (men men_seated : ℕ) : Prop := men_seated = (7 * men) / 8

theorem total_passengers (P men : ℕ) 
  (h1 : two_thirds_women P)
  (h2 : one_eighth_men_standing men)
  (h3 : fourteen_seated_men men)
  (h4 : men_eq_one_third_passengers P men)
  (h5 : men_seating_relation men men_seated)
  : P = 48 := 
  by
    sorry

end total_passengers_l167_167633


namespace find_beta_l167_167392

open Real

theorem find_beta (α β : ℝ) (h1 : cos α = 1 / 7) (h2 : cos (α - β) = 13 / 14)
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π / 2) : β = π / 3 :=
by
  sorry

end find_beta_l167_167392


namespace find_m_l167_167063

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

theorem find_m (a m : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : is_odd_function (λ x, 1 / (a^x + 1) - m)) :
  m = 1 / 2 :=
by
  sorry

end find_m_l167_167063


namespace simplify_fraction_and_rationalize_l167_167170
    
theorem simplify_fraction_and_rationalize :
  (sqrt 6 / sqrt 10) * (sqrt 5 / sqrt 15) * (sqrt 8 / sqrt 14) = 2 * sqrt 7 / 7 := by
sorry

end simplify_fraction_and_rationalize_l167_167170


namespace infinite_sequences_B_intersect_with_A_infinite_l167_167558

theorem infinite_sequences_B_intersect_with_A_infinite (A B : ℕ → ℕ) (d : ℕ) :
  (∀ n, A n = 5 * n - 2) ∧ (∀ k, B k = k * d + 7 - d) →
  (∃ d, ∀ m, ∃ k, A m = B k) :=
by
  sorry

end infinite_sequences_B_intersect_with_A_infinite_l167_167558


namespace range_ratio_a_b_l167_167865

theorem range_ratio_a_b (A B C a b c : ℝ) (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (h_angles : A + B + C = π) (h_A_eq_2B : A = 2 * B) :
  (sqrt 2) < a / b ∧ a / b < (sqrt 3) := by
  sorry

end range_ratio_a_b_l167_167865


namespace option_d_correct_l167_167810

-- Define the monotonic function f and its properties
variable {R : Type*} [linear_ordered_field R] (f : R → R)

-- Define the inverse function f⁻¹
noncomputable def f_inv : R → R := function.inverse f

-- Definition reflecting the condition that f is monotonic
def is_monotonic (f : R → R) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

-- Definition of the main theorem to prove
theorem option_d_correct (hmono : is_monotonic f) :
  ∀ x y : R, f (x + 1) = y ↔ f_inv x - 1 = f_inv y :=
sorry

end option_d_correct_l167_167810


namespace carpet_dimensions_l167_167298

theorem carpet_dimensions (a b : ℕ) 
  (h1 : a^2 + b^2 = 38^2 + 55^2) 
  (h2 : a^2 + b^2 = 50^2 + 55^2) 
  (h3 : a ≤ b) : 
  (a = 25 ∧ b = 50) ∨ (a = 50 ∧ b = 25) :=
by {
  -- The proof would go here
  sorry
}

end carpet_dimensions_l167_167298


namespace num_digits_of_x_l167_167479

theorem num_digits_of_x (x : ℝ) (h : log 2 (log 3 (log 2 x)) = 2) : 
  let d := ⌈81 * log 10 2⌉ in d = 25 :=
by 
  sorry

end num_digits_of_x_l167_167479


namespace A_can_finish_remaining_work_in_6_days_l167_167696

-- Condition: A can finish the work in 18 days
def A_work_rate := 1 / 18

-- Condition: B can finish the work in 15 days
def B_work_rate := 1 / 15

-- Given B worked for 10 days
def B_days_worked := 10

-- Calculation of the remaining work
def remaining_work := 1 - B_days_worked * B_work_rate

-- Calculation of the time for A to finish the remaining work
def A_remaining_days := remaining_work / A_work_rate

-- The theorem to prove
theorem A_can_finish_remaining_work_in_6_days : A_remaining_days = 6 := 
by 
  -- The proof is not required, so we use sorry to skip it.
  sorry

end A_can_finish_remaining_work_in_6_days_l167_167696


namespace sum_fractions_l167_167731

theorem sum_fractions : 
  (1/2 + 1/6 + 1/12 + 1/20 + 1/30 + 1/42 = 6/7) :=
by
  sorry

end sum_fractions_l167_167731


namespace time_to_cross_platform_l167_167694

-- Definitions of the given conditions
def train_length : ℝ := 900
def time_to_cross_pole : ℝ := 18
def platform_length : ℝ := 1050

-- Goal statement in Lean 4 format
theorem time_to_cross_platform : 
  let speed := train_length / time_to_cross_pole;
  let total_distance := train_length + platform_length;
  let time := total_distance / speed;
  time = 39 := 
by
  sorry

end time_to_cross_platform_l167_167694


namespace num_unique_triangle_areas_correct_l167_167755

noncomputable def num_unique_triangle_areas : ℕ :=
  let A := 0
  let B := 1
  let C := 3
  let D := 6
  let E := 0
  let F := 2
  let base_lengths := [1, 2, 3, 5, 6]
  (base_lengths.eraseDups).length

theorem num_unique_triangle_areas_correct : num_unique_triangle_areas = 5 :=
  by sorry

end num_unique_triangle_areas_correct_l167_167755


namespace find_p_plus_q_l167_167978

open BigOperators

noncomputable section

structure SquareWithOctagon where
  center : ℝ × ℝ
  side : ℝ
  length_AB : ℝ

def octagon_area (swo : SquareWithOctagon) : ℝ :=
  let base := swo.length_AB
  let height := swo.side / 2
  8 * (1 / 2 * base * height)

theorem find_p_plus_q (swo : SquareWithOctagon) (h : swo.side = 2) (hAB : swo.length_AB = 37 / 80) :
    ∃ p q : ℕ, RelativelyPrime p q ∧ octagon_area swo = p / q ∧ p + q = 57 := 
  sorry

end find_p_plus_q_l167_167978


namespace find_difference_l167_167594

variable (d : ℕ) (A B : ℕ)
open Nat

theorem find_difference (hd : d > 7)
  (hAB : d * A + B + d * A + A = d * d + 7 * d + 4)  (hA_gt_B : A > B):
  A - B = 3 :=
sorry

end find_difference_l167_167594


namespace product_of_equal_numbers_l167_167947

theorem product_of_equal_numbers (a b : ℕ) (mean : ℕ) (sum : ℕ)
  (h1 : mean = 20)
  (h2 : a = 22)
  (h3 : b = 34)
  (h4 : sum = 4 * mean)
  (h5 : sum - a - b = 2 * x)
  (h6 : sum = 80)
  (h7 : x = 12) 
  : x * x = 144 :=
by
  sorry

end product_of_equal_numbers_l167_167947


namespace domain_of_log_l167_167373

def quadratic_expr (x : ℝ) := x^2 - 3*x + 2

theorem domain_of_log 
  (f : ℝ → ℝ) 
  (h : ∀ x, f x = log (quadratic_expr x)) :
  ∀ x, x ∈ set_of (λ x, (quadratic_expr x > 0)) → 
  x ∈ set.Iio 1 ∪ set.Ioi 2 :=
by
  sorry

end domain_of_log_l167_167373


namespace scientific_notation_8_500_000_l167_167713

theorem scientific_notation_8_500_000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 8_500_000 = a * 10^n ∧ a = 8.5 ∧ n = 6 :=
by
  sorry

end scientific_notation_8_500_000_l167_167713


namespace solve_for_x_l167_167933

theorem solve_for_x (x : ℝ) : (7 : ℝ)^(3 * x + 2) = (1 / 49 : ℝ) → x = -4 / 3 :=
begin
  sorry
end

end solve_for_x_l167_167933


namespace project_completion_by_B_l167_167519

-- Definitions of the given conditions
def person_A_work_rate := 1 / 10
def person_B_work_rate := 1 / 15
def days_A_worked := 3

-- Definition of the mathematical proof problem
theorem project_completion_by_B {x : ℝ} : person_A_work_rate * days_A_worked + person_B_work_rate * x = 1 :=
by
  sorry

end project_completion_by_B_l167_167519


namespace probability_at_least_3_boys_or_exactly_half_girls_l167_167521

noncomputable def probability_boys_or_half_girls : ℚ :=
  let total_outcomes := 2^6
  let prob_3_boys := (binomial 6 3) * (1/2)^6
  let prob_4_boys := (binomial 6 4) * (1/2)^6
  let prob_5_boys := (binomial 6 5) * (1/2)^6
  let prob_6_boys := (binomial 6 6) * (1/2)^6
  let prob_3_boys_or_more := prob_3_boys + prob_4_boys + prob_5_boys + prob_6_boys
  prob_3_boys_or_more

theorem probability_at_least_3_boys_or_exactly_half_girls :
  probability_boys_or_half_girls = 21 / 32 :=
begin
  -- proof will follow
  sorry
end

end probability_at_least_3_boys_or_exactly_half_girls_l167_167521


namespace krakozyabrs_proof_l167_167113

-- Defining necessary variables and conditions
variable {n : ℕ}

-- Conditions given in the problem
def condition1 (H : ℕ) : Prop := 0.2 * H = n
def condition2 (W : ℕ) : Prop := 0.25 * W = n

-- Definition using the principle of inclusion-exclusion
def total_krakozyabrs (H W : ℕ) : ℕ := H + W - n

-- Statements reflecting the conditions and the final conclusion
theorem krakozyabrs_proof (H W : ℕ) (n : ℕ) : condition1 H → condition2 W → (25 < total_krakozyabrs H W) ∧ (total_krakozyabrs H W < 35) → total_krakozyabrs H W = 32 :=
by
  -- We skip the proof here
  sorry

end krakozyabrs_proof_l167_167113


namespace suff_not_nec_cond_l167_167014

theorem suff_not_nec_cond (a : ℝ) : (a > 6 → a^2 > 36) ∧ (a^2 > 36 → (a > 6 ∨ a < -6)) := by
  sorry

end suff_not_nec_cond_l167_167014


namespace period_of_tan2x_plus_cot2x_l167_167655

noncomputable theory

open Real

theorem period_of_tan2x_plus_cot2x :
  ∃ T > 0, ∀ x, tan (2 * (x + T)) + cot (2 * (x + T)) = tan (2 * x) + cot (2 * x) :=
sorry

end period_of_tan2x_plus_cot2x_l167_167655


namespace triangle_height_decrease_l167_167640

theorem triangle_height_decrease
(base_b height_b : ℝ) 
(base_a height_a : ℝ)
(percentage_less : ℝ) :
  base_a = 1.2 * base_b →
  (0.5 * base_a * height_a) = 0.9975 * (0.5 * base_b * height_b) →
  percentage_less = 100 - ((height_a / height_b) * 100) →
  percentage_less = 16.875 := 
begin
  intros h1 h2 h3,
  sorry
end

end triangle_height_decrease_l167_167640


namespace value_of_2x_plus_3y_l167_167847

theorem value_of_2x_plus_3y {x y : ℝ} (h1 : 2 * x - 1 = 5) (h2 : 3 * y + 2 = 17) : 2 * x + 3 * y = 21 :=
by
  sorry

end value_of_2x_plus_3y_l167_167847


namespace even_digit_numbers_count_eq_100_l167_167455

-- Definition for the count of distinct three-digit positive integers with only even digits
def count_even_digit_three_numbers : ℕ :=
  let hundreds_place := {2, 4, 6, 8}.card
  let tens_units_place := {0, 2, 4, 6, 8}.card
  hundreds_place * tens_units_place * tens_units_place

-- Theorem stating the count of distinct three-digit positive integers with only even digits is 100
theorem even_digit_numbers_count_eq_100 : count_even_digit_three_numbers = 100 :=
by sorry

end even_digit_numbers_count_eq_100_l167_167455


namespace circle_range_of_a_l167_167190

theorem circle_range_of_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * a * x - 4 * y + (a^2 + a) = 0 → (x - h)^2 + (y - k)^2 = r^2) ↔ (a < 4) :=
sorry

end circle_range_of_a_l167_167190


namespace cannot_determine_movies_watched_l167_167981

theorem cannot_determine_movies_watched:
  (m b r : ℕ) (h1 : m = 17) (h2 : b = 11) (h3 : r = 13) (h4 : m = b + 6) :
  ∃ (x : ℕ), x ≤ m := 
by {
    -- The statement asserts there exists an x (number of movies watched) such that x ≤ 17
    cases h1, -- Use the fact that m = 17
    existsi m, -- There exists m which is equal to 17
    sorry
}

end cannot_determine_movies_watched_l167_167981


namespace range_of_mn_proof_l167_167688

-- Definitions of the problem
def side_length (a : ℝ) := a > 0
def angle_120_deg := 120 * (π / 180)  -- 120 degrees in radians
def range_of_MN (a : ℝ) := [real.sqrt(3) / 2 * a, a]

theorem range_of_mn_proof (a : ℝ) (h : side_length a):
  (∃ M N : ℝ, 
    M ∈ [0, a] ∧ N ∈ [0, a] ∧ 
    AM = λ * a ∧ FN = λ * a ∧
    λ ∈ [0, 1] ∧ 
    dihedral_angle_between_planes = angle_120_deg ∧
    MN ∈ range_of_MN a) :=
sorry

end range_of_mn_proof_l167_167688


namespace total_kids_on_soccer_field_l167_167635

theorem total_kids_on_soccer_field : 
    let initial_kids := 14 in
    let joining_kids := 22 in
    initial_kids + joining_kids = 36 := 
by
  sorry

end total_kids_on_soccer_field_l167_167635


namespace extreme_value_of_f_range_of_a_log_inequality_l167_167053

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * log x + (a - 1) * x

theorem extreme_value_of_f (a : ℝ) (h : 0 < a) : (∃ x : ℝ, f x a = (a * (1 - log a)) / 2) :=
sorry

theorem range_of_a (h : ∃ x y : ℝ, g x > 0 ∧ g y > 0 ∧ f 1 a < 0 ∧ e⁻¹ < x ∧ x < e ∧ e⁻¹ < y ∧ y < e) : 
  (2 * e - 1) / (2 * e^2 + 2 * e) < a ∧ a < 1/2 :=
sorry

theorem log_inequality (x : ℝ) (h : 0 < x) : log x + (3 / (4 * x^2)) - (1 / exp x) > 0 :=
sorry

end extreme_value_of_f_range_of_a_log_inequality_l167_167053


namespace initial_distance_between_projectiles_l167_167219

noncomputable def initial_distance
  (speed_proj1_km_per_h : ℝ)
  (speed_proj2_km_per_h : ℝ)
  (time_to_meet_min : ℝ) : ℝ :=
let speed_proj1_km_per_min := speed_proj1_km_per_h / 60 in
let speed_proj2_km_per_min := speed_proj2_km_per_h / 60 in
let distance_proj1 := speed_proj1_km_per_min * time_to_meet_min in
let distance_proj2 := speed_proj2_km_per_min * time_to_meet_min in
distance_proj1 + distance_proj2

theorem initial_distance_between_projectiles :
  initial_distance 460 525 72 = 1182.24 :=
by
  unfold initial_distance
  sorry

end initial_distance_between_projectiles_l167_167219


namespace replace_last_s_l167_167699

def alphabet := fin 26

def shift_letter (c : alphabet) (shift : ℕ) : alphabet :=
  ⟨(c.1 + shift) % 26, nat.mod_lt _ (by norm_num)⟩

def occurrence_shift (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def message_s : string := "Lee's sis is a Mississippi miss, Chriss!"

def count_occurrences (msg : string) (char : char) : ℕ :=
  msg.fold 0 (λ acc c, if c = char then acc + 1 else acc)

def nth_s_shift (msg : string) : ℕ :=
  occurrence_shift (count_occurrences msg 's')

theorem replace_last_s : 
  shift_letter ⟨18, nat.lt_succ_self 26⟩ (nth_s_shift message_s) = ⟨18, nat.lt_succ_self 26⟩ := 
sorry

end replace_last_s_l167_167699


namespace baker_sales_difference_l167_167266

/-!
  Prove that the difference in dollars between the baker's daily average sales and total sales for today is 48 dollars.
-/

theorem baker_sales_difference :
  let price_pastry := 2
  let price_bread := 4
  let avg_pastries := 20
  let avg_bread := 10
  let today_pastries := 14
  let today_bread := 25
  let daily_avg_sales := avg_pastries * price_pastry + avg_bread * price_bread
  let today_sales := today_pastries * price_pastry + today_bread * price_bread
  daily_avg_sales - today_sales = 48 :=
sorry

end baker_sales_difference_l167_167266


namespace log_base_3_of_one_over_81_l167_167348

theorem log_base_3_of_one_over_81 : log 3 (1 / 81) = -4 := by
  sorry

end log_base_3_of_one_over_81_l167_167348


namespace find_median_l167_167705

theorem find_median (n : ℕ) (l : List ℝ) 
  (h_mode : l.mode = 36)
  (h_mean : l.mean = 28)
  (h_min : l.minimum = 12)
  (m : ℝ)
  (h_median : l.median = m)
  (h_replace_plus8_mean : (l.replace m (m+8)).mean = 30)
  (h_replace_plus8_median : (l.replace m (m+8)).median = m+8)
  (h_replace_minus10_median : (l.replace m (m-10)).median = m-5) :
  m = 34 :=
sorry

end find_median_l167_167705


namespace wrongly_copied_value_l167_167959

theorem wrongly_copied_value (mean_initial mean_correct : ℕ) (n : ℕ) 
  (wrong_copied_value : ℕ) (total_sum_initial total_sum_correct : ℕ) : 
  (mean_initial = 150) ∧ (mean_correct = 151) ∧ (n = 30) ∧ 
  (wrong_copied_value = 135) ∧ (total_sum_initial = n * mean_initial) ∧ 
  (total_sum_correct = n * mean_correct) → 
  (total_sum_correct - (total_sum_initial - wrong_copied_value) + wrong_copied_value = 300) :=
by
  intros h
  have h1 : mean_initial = 150 := by sorry
  have h2 : mean_correct = 151 := by sorry
  have h3 : n = 30 := by sorry
  have h4 : wrong_copied_value = 135 := by sorry
  have h5 : total_sum_initial = n * mean_initial := by sorry
  have h6 : total_sum_correct = n * mean_correct := by sorry
  sorry -- This is where the proof would go, but is not required per instructions.

end wrongly_copied_value_l167_167959


namespace general_term_a_general_term_b_sum_c_l167_167791

-- Problem 1: General term formula for the sequence {a_n}
theorem general_term_a (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 2 - a n) :
  ∀ n, a n = (1 / 2) ^ (n - 1) := 
sorry

-- Problem 2: General term formula for the sequence {b_n}
theorem general_term_b (b : ℕ → ℝ) (a : ℕ → ℝ) (h_b1 : b 1 = 1)
  (h_b : ∀ n, b (n + 1) = b n + a n) (h_a : ∀ n, a n = (1 / 2) ^ (n - 1)) :
  ∀ n, b n = 3 - 2 * (1 / 2) ^ (n - 1) := 
sorry

-- Problem 3: Sum of the first n terms for the sequence {c_n}
theorem sum_c (c : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h_b : ∀ n, b n = 3 - 2 * (1 / 2) ^ (n - 1)) (h_c : ∀ n, c n = n * (3 - b n)) :
  ∀ n, T n = 8 - (8 + 4 * n) * (1 / 2) ^ n := 
sorry

end general_term_a_general_term_b_sum_c_l167_167791


namespace rectangle_diagonal_length_l167_167962

theorem rectangle_diagonal_length 
  (P : ℝ) (L W : ℝ) (H1 : P = 2 * (L + W)) (H2 : L / W = 5 / 2) : d = sqrt 725 :=
by
  sorry

end rectangle_diagonal_length_l167_167962


namespace interval_of_monotonic_increase_l167_167031

noncomputable def m : ℝ := sorry -- defined as some real number.

def f (x : ℝ) : ℝ := x^2 * (x - m)

theorem interval_of_monotonic_increase
  (h : deriv f (-1) = -1) :
  (∀ x, deriv f x > 0 → x < -(4 / 3) ∨ x > 0) :=
sorry

end interval_of_monotonic_increase_l167_167031


namespace carousel_ratio_l167_167185

theorem carousel_ratio (P : ℕ) (h : 3 + P + 2*P + P/3 = 33) : P / 3 = 3 := 
by 
  sorry

end carousel_ratio_l167_167185


namespace eval_f_neg1_eval_f_2_l167_167550

def piecewise_fn (x : ℝ) : ℝ :=
if x < 0 then 3 * x - 7 else x^2 + 4 * x + 4

theorem eval_f_neg1 : piecewise_fn (-1) = -10 := 
by 
  -- proof omitted
  sorry

theorem eval_f_2 : piecewise_fn 2 = 16 := 
by 
  -- proof omitted
  sorry

end eval_f_neg1_eval_f_2_l167_167550


namespace distinct_digits_mean_l167_167601

theorem distinct_digits_mean (M : ℕ) :
  (∀ n, n ∈ {9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999} → M = (9 + 99 + 999 + 9999 + 99999 + 999999 + 9999999 + 99999999 + 999999999) / 9) →
  M = 123456789 ∧ (∀ d : ℕ, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → d ≠ 0 → d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) :=
by 
  sorry

end distinct_digits_mean_l167_167601


namespace inflation_over_two_years_real_yield_deposit_second_year_l167_167662

-- Inflation problem setup and proof
theorem inflation_over_two_years :
  ((1 + 0.015) ^ 2 - 1) * 100 = 3.0225 :=
by sorry

-- Real yield problem setup and proof
theorem real_yield_deposit_second_year :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by sorry

end inflation_over_two_years_real_yield_deposit_second_year_l167_167662


namespace log_base_3_of_reciprocal_81_l167_167355

theorem log_base_3_of_reciprocal_81 : log 3 (1 / 81) = -4 :=
by
  sorry

end log_base_3_of_reciprocal_81_l167_167355


namespace quotient_of_division_l167_167229

theorem quotient_of_division (dividend divisor remainder quotient: ℕ) : 
  dividend = 140 ∧ divisor = 15 ∧ remainder = 5 → 
  quotient = (dividend - remainder) / divisor :=
begin
  intros h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw [h1, h3, h4],
  have h_calc : (140 - 5) / 15 = 9, by norm_num,
  exact h_calc,
end

end quotient_of_division_l167_167229


namespace quadratic_function_opens_downwards_l167_167070

theorem quadratic_function_opens_downwards (m : ℝ) (h₁ : m - 1 < 0) (h₂ : m^2 + 1 = 2) : m = -1 :=
by {
  -- Proof would go here.
  sorry
}

end quadratic_function_opens_downwards_l167_167070


namespace prime_divides_sequence_implies_greater_l167_167130

theorem prime_divides_sequence_implies_greater :
  ∀ (p n : ℕ), prime p → odd p →
  (∀ (a : ℕ → ℕ), 
    (a 1 = 2) →
    (∀ n, a (n + 1) = (a n)^3 - (a n) + 1) →
    (p ∣ a n) → p > n ) :=
by sorry

end prime_divides_sequence_implies_greater_l167_167130


namespace find_omega_l167_167197

-- Define the function and conditions
def f (ω x : ℝ) : ℝ := 3 * sin (ω * x + π / 6)

-- Statement of the theorem
theorem find_omega (ω : ℝ) (h_pos : ω > 0) (h_period : (∀ x : ℝ, f ω (x + π) = f ω x)) : ω = 2 :=
sorry

end find_omega_l167_167197


namespace ln_x_add_1_le_x_f_x_le_1_l167_167044

theorem ln_x_add_1_le_x (x : ℝ) (h₀ : 0 < x) : ln x + 1 ≤ x := sorry

theorem f_x_le_1 (x : ℝ) (h₀ : 0 < x) : (1 + ln x) / x ≤ 1 := 
by
  have h₁ : ln x + 1 ≤ x := ln_x_add_1_le_x x h₀
  sorry

end ln_x_add_1_le_x_f_x_le_1_l167_167044


namespace ai_gt_1_ai_powers_of_3_ai_gt_bi_unique_primitive_representation_l167_167420

def Cantor_set (C : Set ℕ) : Prop :=
  ∀ n ∈ C, ∀ k, 0 ≤ k ∧ k < 3^(nat.log n / nat.log 3 + 1) → digit n k ≠ 1

def similarity_transformation (A : Set ℕ) (a b : ℤ) : Set ℕ :=
  {n | ∃ m ∈ A, n = a * m + b}

variables {C : Set ℕ} (a_i b_i : ℤ) (i : ℕ)

-- Given set C consisting of non-negative integers whose ternary representation contains no 1s
axiom Cantor_set_axiom : Cantor_set C

-- Given similarity transformation defined for a set A subseteq ℤ by aA + b
axiom similarity_transform_axiom (A : Set ℕ) : similarity_transformation A a_i b_i

theorem ai_gt_1 : ∀ i, a_i > 1 :=
by  sorry

theorem ai_powers_of_3 : ∀ i, ∃ r : ℕ, a_i = 3^r :=
by sorry

theorem ai_gt_bi : ∀ i, a_i > b_i :=
by sorry

theorem unique_primitive_representation : C = (similarity_transformation C 3 0) ∪ (similarity_transformation C 3 2) :=
by sorry

end ai_gt_1_ai_powers_of_3_ai_gt_bi_unique_primitive_representation_l167_167420


namespace cubes_sum_identity_l167_167907

variable {a b : ℝ}

theorem cubes_sum_identity (h : (a / (1 + b) + b / (1 + a) = 1)) : a^3 + b^3 = a + b :=
sorry

end cubes_sum_identity_l167_167907


namespace slope_negative_l167_167871

theorem slope_negative (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → mx1 + 5 > mx2 + 5) → m < 0 :=
by
  sorry

end slope_negative_l167_167871


namespace log_relationships_l167_167422

theorem log_relationships (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) :
  9 * (Real.log y / Real.log c)^2 + 5 * (Real.log y / Real.log d)^2 = 18 * (Real.log y)^2 / (Real.log c * Real.log d) →
  d = c^(1 / Real.sqrt 3) ∨ d = c^(Real.sqrt 3) ∨ d = c^(1 / Real.sqrt (6 / 10)) ∨ d = c^(Real.sqrt (6 / 10)) :=
sorry

end log_relationships_l167_167422


namespace ratio_of_perimeters_not_integer_l167_167578

theorem ratio_of_perimeters_not_integer
    (a k l : ℕ)
    (h_area : a^2 = k * l)
    (hk_pos : 0 < k)
    (hl_pos : 0 < l)
    (ha_pos : 0 < a) :
    ∀ n : ℤ, 2 * (k + l) ≠ 4 * n * a :=
sorry

end ratio_of_perimeters_not_integer_l167_167578


namespace sum_of_b_values_l167_167340

theorem sum_of_b_values :
  let discriminant (b : ℝ) := (b + 6) ^ 2 - 4 * 3 * 12
  ∃ b1 b2 : ℝ, discriminant b1 = 0 ∧ discriminant b2 = 0 ∧ b1 + b2 = -12 :=
by sorry

end sum_of_b_values_l167_167340


namespace minimum_discount_percentage_l167_167187

theorem minimum_discount_percentage (cost_price marked_price : ℝ) (profit_margin : ℝ) (discount : ℝ) :
  cost_price = 400 ∧ marked_price = 600 ∧ profit_margin = 0.05 ∧ 
  (marked_price * (1 - discount / 100) - cost_price) / cost_price ≥ profit_margin → discount ≤ 30 := 
by
  intros h
  rcases h with ⟨hc, hm, hp, hineq⟩
  sorry

end minimum_discount_percentage_l167_167187


namespace exists_Q_square_l167_167895

open Polynomial

theorem exists_Q_square (P : Polynomial ℤ) (h_monic : P.monic) (h_even_deg : even P.nat_degree)
  (h_inf_square : ∃^{∞} x : ℤ, ∃ k : ℕ+, P.eval x = k^2) :
  ∃ Q : Polynomial ℤ, P = Q^2 :=
sorry

end exists_Q_square_l167_167895


namespace inequality_always_holds_l167_167006

theorem inequality_always_holds (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 < log a (x+1)) ↔ (1 < a ∧ a ≤ 2) :=
by
  sorry

end inequality_always_holds_l167_167006


namespace geometric_seq_condition_l167_167510

noncomputable def is_geometric_seq_with_ratio (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n ≥ 1, a (n + 1) = r * a n

def condition (a : ℕ → ℝ) : Prop :=
∀ n ≥ 2, a n = 2 * a (n - 1)

theorem geometric_seq_condition (a : ℕ → ℝ) :
  condition a ↔ (∀ (n ≥ 2), is_geometric_seq_with_ratio a 2 → condition a) :=
sorry

end geometric_seq_condition_l167_167510


namespace fifth_term_is_31_l167_167736

/-- 
  The sequence is formed by summing consecutive powers of 2. 
  Define the sequence and prove the fifth term is 31.
--/
def sequence_sum (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, 2^k)

theorem fifth_term_is_31 : sequence_sum 5 = 31 :=
by sorry

end fifth_term_is_31_l167_167736


namespace slope_angle_of_line_through_origin_and_point_l167_167958

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_angle_of_line_through_origin_and_point :
  let p1: ℝ × ℝ := (0, 0)
  let p2: ℝ × ℝ := (-1, -1)
  let k : ℝ := slope p1 p2
  ∃ α : ℝ, tan α = k ∧ α = π / 4 :=
by
  have h : slope (0, 0) (-1, -1) = 1 := by
    simp [slope]
  use π / 4
  split
  . simp [Real.tan_pi_div_four]
  . sorry

end slope_angle_of_line_through_origin_and_point_l167_167958


namespace elijah_total_cards_l167_167343

-- Define the conditions
def num_decks : ℕ := 6
def cards_per_deck : ℕ := 52

-- The main statement that we need to prove
theorem elijah_total_cards : num_decks * cards_per_deck = 312 := by
  -- We skip the proof
  sorry

end elijah_total_cards_l167_167343


namespace length_of_ae_l167_167677

-- Definition of points and lengths between them
variables (a b c d e : Type)
variables (bc cd de ab ac : ℝ)

-- Given conditions
axiom H1 : bc = 3 * cd
axiom H2 : de = 8
axiom H3 : ab = 5
axiom H4 : ac = 11
axiom H5 : bc = ac - ab
axiom H6 : cd = bc / 3

-- Theorem to prove
theorem length_of_ae : ∀ ab bc cd de : ℝ, ae = ab + bc + cd + de := by
  sorry

end length_of_ae_l167_167677


namespace log_base_3_of_one_over_81_l167_167346

theorem log_base_3_of_one_over_81 : log 3 (1 / 81) = -4 := by
  sorry

end log_base_3_of_one_over_81_l167_167346


namespace x_1997_eq_23913_l167_167526

noncomputable def x_seq : ℕ → ℤ
| 1     := 1
| (n+1) := x_seq n + ⌊(x_seq n : ℚ) / n⌋ + 2

theorem x_1997_eq_23913 : x_seq 1997 = 23913 := by
  sorry

end x_1997_eq_23913_l167_167526


namespace sum_possible_values_y_zero_l167_167278

variable {A B C D O : Type} [F : Field A] -- Define the type and field
variables AB AD AC BD x : A
variables (r y : A)
variables (k : Kite A B C D) -- Assuming a type for Kite

-- Define the conditions
def kite_conditions (AB AD AC BD x r y : A) : Prop :=
  AB = AD ∧
  AC = 10 ∧
  BD = x ∧
  (∃ r, r < 1 ∧ AB = 5 ∧ BC = 5 * r ∧ CD = 5 * r^2 ∧ DA = 5 * r^3) ∧
  2 ≤ x ∧ x ≤ 10

-- The theorem to be proved
theorem sum_possible_values_y_zero (A B C D : Type) [Field A]
  (AB AD AC BD x y r : A) (k : Kite A B C D) :
  kite_conditions AB AD AC BD x r y → 
  (r < 1 → y ≠ y) → 
  ∑ i in {y | r < 1}, i = 0 := sorry

end sum_possible_values_y_zero_l167_167278


namespace cube_inequality_sufficient_and_necessary_l167_167627

theorem cube_inequality_sufficient_and_necessary (a b : ℝ) :
  (a > b ↔ a^3 > b^3) := 
sorry

end cube_inequality_sufficient_and_necessary_l167_167627


namespace log_base_3_l167_167349

theorem log_base_3 (h : (1 / 81 : ℝ) = 3 ^ (-4 : ℝ)) : Real.logBase 3 (1 / 81) = -4 := 
by sorry

end log_base_3_l167_167349


namespace A_beats_B_by_A_beats_C_by_C_beats_B_by_l167_167857

-- Define the conditions in the problem.
def time_A : ℕ := 40
def time_B : ℕ := 50
def time_C : ℕ := 45

-- State the questions and the expected answers as proof goals.
theorem A_beats_B_by : (time_B - time_A) = 10 := by
  sorry

theorem A_beats_C_by : (time_C - time_A) = 5 := by
  sorry

theorem C_beats_B_by : (time_C - time_B) = -5 := by
  sorry


end A_beats_B_by_A_beats_C_by_C_beats_B_by_l167_167857


namespace hadassah_painting_time_l167_167441

theorem hadassah_painting_time :
  let small_painting_time := 6 / 12
  let large_painting_time := 8 / 6
  let small_paintings := 15
  let large_paintings := 10
  let total_paintings := small_paintings + large_paintings
  let breaks := total_paintings / 3
  let break_time := 0.5 * breaks.floor
  let painting_time := (small_paintings * small_painting_time) + (large_paintings * large_painting_time)
  let total_time := painting_time + break_time
  total_time = 24.8 :=
begin
  sorry
end

end hadassah_painting_time_l167_167441


namespace product_ends_in_zero_l167_167882

-- Definitions from problem conditions
variable (Ж B И H U Pi У X : ℕ)
variable (МЁД : ℕ)
variable (digits_used : Finset ℕ := {1, 0, 8, 9})
variable (remaining_digits : Finset ℕ := {2, 3, 4, 5, 6, 7})

-- Conditions provided in the problem
axiom Ж_condition : Ж ∈ (Finset.range 10)
axiom equ_JJ_J_MJD : 12 * Ж = МЁД
axiom unique_digits : ∀ a b : ℕ, a ∈ digits_used → b ∈ digits_used → a ≠ b → Ж = a → False
axiom distinct_remaining_digits : remaining_digits ∩ digits_used = ∅
axiom digit_range : ∀ x : ℕ, x ∈ remaining_digits → x < 10

noncomputable def product_end_digit : ℕ := 
        (B * И * H * H * U * Pi * У * X) % 10

-- Lean statement (proof not included)
theorem product_ends_in_zero 
    (hBrem : B ∈ remaining_digits)
    (hIrem : И ∈ remaining_digits)
    (hHrem : H ∈ remaining_digits)
    (hUrem : U ∈ remaining_digits)
    (hPirem : Pi ∈ remaining_digits)
    (hVrem : У ∈ remaining_digits)
    (hXrem : X ∈ remaining_digits)
    (h2inrem : 2 ∈ remaining_digits)
    (h5inrem : 5 ∈ remaining_digits) : 
    product_end_digit Ж B И H U Pi У X = 0 := 
sorry

end product_ends_in_zero_l167_167882


namespace total_kilometers_ridden_l167_167164

theorem total_kilometers_ridden :
  ∀ (d1 d2 d3 d4 : ℕ),
    d1 = 40 →
    d2 = 50 →
    d3 = d2 - d2 / 2 →
    d4 = d1 + d3 →
    d1 + d2 + d3 + d4 = 180 :=
by 
  intros d1 d2 d3 d4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_kilometers_ridden_l167_167164


namespace range_of_a_l167_167802

noncomputable def p (a : ℝ) : Prop :=
  ∀ m ∈ set.Icc (-1 : ℝ) 1, a^2 - 5 * a - 3 ≥ real.sqrt (m^2 + 8)

noncomputable def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a * x + 2 < 0

theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬ (p a ∧ q a) → (a ∈ set.Icc (-2 * real.sqrt 2 : ℝ) (-1 : ℝ) ∨ a ∈ set.Ioo (2 * real.sqrt 2 : ℝ) 6) :=
by
  sorry

end range_of_a_l167_167802


namespace rectangular_box_surface_area_l167_167631

theorem rectangular_box_surface_area
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 200)
  (h2 : sqrt (a^2 + b^2 + c^2) = 25) :
  2 * (a * b + b * c + c * a) = 1875 :=
by 
  sorry

end rectangular_box_surface_area_l167_167631


namespace geometric_sequence_sum_l167_167328

theorem geometric_sequence_sum :
  let a := (1/2 : ℚ)
  let r := (1/3 : ℚ)
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 243 :=
by
  sorry

end geometric_sequence_sum_l167_167328


namespace agatha_amount_left_l167_167297

noncomputable def initial_amount : ℝ := 60
noncomputable def frame_cost : ℝ := 15 * (1 - 0.10)
noncomputable def wheel_cost : ℝ := 25 * (1 - 0.05)
noncomputable def seat_cost : ℝ := 8 * (1 - 0.15)
noncomputable def handlebar_tape_cost : ℝ := 5
noncomputable def bell_cost : ℝ := 3
noncomputable def hat_cost : ℝ := 10 * (1 - 0.25)

noncomputable def total_cost : ℝ :=
  frame_cost + wheel_cost + seat_cost + handlebar_tape_cost + bell_cost + hat_cost

noncomputable def amount_left : ℝ := initial_amount - total_cost

theorem agatha_amount_left : amount_left = 0.45 :=
by
  -- interim calculations would go here
  sorry

end agatha_amount_left_l167_167297


namespace maximum_value_is_l167_167938

noncomputable def maximum_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) : ℝ :=
  x^2 + 2 * x * y + 3 * y^2

theorem maximum_value_is (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) :
  maximum_value x y h₁ h₂ h₃ ≤ 18 + 12 * Real.sqrt 3 :=
sorry

end maximum_value_is_l167_167938


namespace sum_of_coefficients_of_rational_terms_in_expansion_l167_167772

theorem sum_of_coefficients_of_rational_terms_in_expansion :
  (∑ r in {0, 2, 5}.to_finset, if (r = 2 ∨ r = 5) then (nat.choose 5 r) * (if r = 2 then 2 else 1) else 0) = 21 :=
by sorry

end sum_of_coefficients_of_rational_terms_in_expansion_l167_167772


namespace traveling_speed_l167_167887

namespace ProofProblem

-- Conditions
def distance_to_ny : ℝ := 300
def total_trip_time : ℝ := 7
def rest_time_each_interval : ℝ := 30 / 60
def interval_duration : ℝ := 2

-- Theorem statement
theorem traveling_speed : 
  let intervals := total_trip_time / interval_duration
  let full_intervals := intervals.to_floor
  let rest_periods := full_intervals
  let total_rest_time := rest_periods * rest_time_each_interval
  let total_driving_time := total_trip_time - total_rest_time
  let speed := distance_to_ny / total_driving_time
  abs (speed - 54.55) < 1e-2 := 
by sorry

end ProofProblem

end traveling_speed_l167_167887


namespace sum_of_remainders_mod_l167_167673

theorem sum_of_remainders_mod (a b c : ℕ) (h1 : a % 53 = 31) (h2 : b % 53 = 22) (h3 : c % 53 = 7) :
  (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_remainders_mod_l167_167673


namespace number_of_integers_l167_167057

theorem number_of_integers (n : ℤ) : (200 < n ∧ n < 300 ∧ ∃ r : ℤ, n % 7 = r ∧ n % 9 = r) ↔ 
  n = 252 ∨ n = 253 ∨ n = 254 ∨ n = 255 ∨ n = 256 ∨ n = 257 ∨ n = 258 :=
by {
  sorry
}

end number_of_integers_l167_167057


namespace employee_salaries_l167_167990

theorem employee_salaries 
  (x y z : ℝ)
  (h1 : x + y + z = 638)
  (h2 : x = 1.20 * y)
  (h3 : z = 0.80 * y) :
  x = 255.20 ∧ y = 212.67 ∧ z = 170.14 :=
sorry

end employee_salaries_l167_167990


namespace mutually_exclusive_not_complementary_l167_167494

open Finset

-- Definitions for the number of white and black balls in a bag.
def white_balls : ℕ := 3
def black_balls : ℕ := 4
def total_balls : ℕ := white_balls + black_balls

-- Definitions of events ①, ②, ③, and ④.
def event1 (drawn : Finset ℕ) : Prop :=
  (drawn.card = 1 ∧ (drawn ⊆ range white_balls) ∧ (drawn = range white_balls))

def event2 (drawn : Finset ℕ) : Prop :=
  (∃ ball, ball ∈ drawn ∧ ball < white_balls) ∧ ∀ ball, ball ∈ drawn → ball ≥ white_balls

def event3 (drawn : Finset ℕ) : Prop :=
  (∃ ball, ball ∈ drawn ∧ ball < white_balls) ∧ drawn.card ≥ 2

def event4 (drawn : Finset ℕ) : Prop :=
  (∃ ball, ball ∈ drawn ∧ ball < white_balls) ∧ (∃ ball, ball ∈ drawn ∧ ball ≥ white_balls)

-- The question to prove.
theorem mutually_exclusive_not_complementary :
  (∀ e1 e2, event1 e1 → event1 e2 → e1 = e2 ∨ disjoint e1 e2) ∧
  ¬(∀ e, (event2 e ∨ event3 e ∨ event4 e) → event1 e) :=
sorry

end mutually_exclusive_not_complementary_l167_167494


namespace probability_all_distinct_l167_167781

def set_of_cards := finset.range 1 101
def number_of_draws := 20
def probability_distinct_draws (cards: finset ℕ) (draws: ℕ) : ℝ := 
  ∏ i in finset.range draws, ((cards.card - i): ℝ) / cards.card

theorem probability_all_distinct (p : ℝ) :
  p = probability_distinct_draws set_of_cards number_of_draws →
  p < (9/10)^19 ∧ (9/10)^19 < 1/real.exp 2 :=
by
  intros hp
  sorry

end probability_all_distinct_l167_167781


namespace smallest_x_solution_l167_167771

def min_solution (a b : Real) : Real := if a < b then a else b

theorem smallest_x_solution (x : Real) (h : x * abs x = 3 * x - 2) : 
  x = min_solution (min_solution 1 2) ( (-3 - Real.sqrt 17) / 2) :=
sorry

end smallest_x_solution_l167_167771


namespace largest_c_value_l167_167141

theorem largest_c_value (c : ℚ) (h : (3 * c + 7) * (c - 2) = 9 * c) : c ≤ 2 :=
begin
  sorry
end

example : ∃ c : ℚ, (3 * c + 7) * (c - 2) = 9 * c ∧ c = 2 :=
begin
  use 2,
  split,
  calc (3 * (2 : ℚ) + 7) * ((2 : ℚ) - 2) = (3 * (2 : ℚ) + 7) * (0 : ℚ) : by ring
                                      ... = (0 : ℚ) : by ring,
  ring,
end

end largest_c_value_l167_167141


namespace possible_values_x2_y2_z2_l167_167019

theorem possible_values_x2_y2_z2 {x y z : ℤ}
    (h1 : x + y + z = 3)
    (h2 : x^3 + y^3 + z^3 = 3) : (x^2 + y^2 + z^2 = 3) ∨ (x^2 + y^2 + z^2 = 57) :=
by sorry

end possible_values_x2_y2_z2_l167_167019


namespace triangle_third_side_length_l167_167864

theorem triangle_third_side_length (a b : ℝ) (theta : ℝ) (h_a : a = 10) (h_b : b = 15) (h_theta : theta = 100) :
  ∃ c : ℝ, c ≈ 19.42 :=
by
  sorry

end triangle_third_side_length_l167_167864


namespace num_of_distinct_three_digit_integers_with_even_digits_l167_167448

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def valid_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (is_even_digit (n / 100 % 10)) ∧
  (is_even_digit (n / 10 % 10)) ∧
  (is_even_digit (n % 10)) ∧
  (n / 100 % 10 ≠ 0)

theorem num_of_distinct_three_digit_integers_with_even_digits : 
  {n : ℕ | valid_three_digit_integer n}.finite.to_finset.card = 100 :=
sorry

end num_of_distinct_three_digit_integers_with_even_digits_l167_167448


namespace num_rectangles_in_4x4_grid_l167_167742

theorem num_rectangles_in_4x4_grid : 
  let n := 4 
  (choose n 2) * (choose n 2) = 36 :=
by
  let n := 4
  have h1 : choose n 2 = 6 := dvds.choose_eq_combination n 2
  have h2 : 6 * 6 = 36
  show (choose n 2) * (choose n 2) = 36
  rw [h1, h1]
  exact h2

end num_rectangles_in_4x4_grid_l167_167742


namespace remainder_a83_l167_167140

def a_n (n : ℕ) : ℕ := 6^n + 8^n

theorem remainder_a83 (n : ℕ) : 
  a_n 83 % 49 = 35 := sorry

end remainder_a83_l167_167140


namespace mean_volume_of_cubes_l167_167211

theorem mean_volume_of_cubes (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) :
  ((a^3 + b^3 + c^3) / 3) = 135 :=
by
  -- known cube volumes and given edge lengths conditions
  sorry

end mean_volume_of_cubes_l167_167211


namespace perimeter_of_shaded_region_l167_167868

noncomputable def circle_center : Type := sorry -- Define the object type for circle's center
noncomputable def radius_length : ℝ := 10 -- Define the radius length as 10
noncomputable def central_angle : ℝ := 270 -- Define the central angle corresponding to the arc RS

-- Function to calculate the perimeter of the shaded region
noncomputable def perimeter_shaded_region (radius : ℝ) (angle : ℝ) : ℝ :=
  2 * radius + (angle / 360) * 2 * Real.pi * radius

-- Theorem stating that the perimeter of the shaded region is 20 + 15π given the conditions
theorem perimeter_of_shaded_region : 
  perimeter_shaded_region radius_length central_angle = 20 + 15 * Real.pi :=
by
  -- skipping the actual proof
  sorry

end perimeter_of_shaded_region_l167_167868


namespace jessica_quarters_l167_167886

theorem jessica_quarters (initial_quarters borrowed_quarters remaining_quarters : ℕ)
  (h1 : initial_quarters = 8)
  (h2 : borrowed_quarters = 3) :
  remaining_quarters = initial_quarters - borrowed_quarters → remaining_quarters = 5 :=
by
  intro h3
  rw [h1, h2] at h3
  exact h3

end jessica_quarters_l167_167886


namespace sum_as_fraction_l167_167759

noncomputable def sum_of_decimals := 0.01 + 0.002 + 0.0003 + 0.00004 + 0.000005

theorem sum_as_fraction :
  sum_of_decimals = (2469 / 200000 : ℚ) :=
by
  sorry

end sum_as_fraction_l167_167759


namespace angle_A_area_of_triangle_l167_167515

/-
  Given:
  1. a = sqrt 19
  2. (sin B + sin C) / (cos B + cos A) = (cos B - cos A) / sin C
  3. BD / DC = 3 / 4
  4. AD ⊥ AC
  Prove that:
  1. A = 2π / 3
  2. Area of triangle ABC = 3√3 / 2
-/

-- Definitions based on given conditions
noncomputable def side_a := Real.sqrt 19

axiom trigonometric_equation (B C A : Real) : 
  (Real.sin B + Real.sin C) / (Real.cos B + Real.cos A) = 
  (Real.cos B - Real.cos A) / (Real.sin C)

axiom ratio_BD_DC (BD DC : Real) : BD / DC = 3 / 4

axiom perpendicular_AD_AC (AD AC : Real) : AD * AC = 0

-- Theorem statements for the proof problems
theorem angle_A {B C : Real} (A : Real) (h1 : side_a = Real.sqrt 19) (h2 : trigonometric_equation B C A) :
  A = 2 * Real.pi / 3 := sorry

theorem area_of_triangle {B C : Real} (A : Real) (BD DC AD AC : Real) (h1 : side_a = Real.sqrt 19) 
  (h3 : ratio_BD_DC BD DC) (h4 : perpendicular_AD_AC AD AC) : 
  triangle_area B C A = 3 * Real.sqrt 3 / 2 := sorry

end angle_A_area_of_triangle_l167_167515


namespace ordered_pair_solution_l167_167985

-- Definitions for the problem
def v1 := (3 : ℝ, -1 : ℝ)
def v2 := (0 : ℝ, 2 : ℝ)
def u1 := (8 : ℝ, -3 : ℝ)
def u2 := (-1 : ℝ, 4 : ℝ)
def x := -15 / 29
def y := 33 / 29

theorem ordered_pair_solution :
  v1 + x • u1 = v2 + y • u2 :=
by
  -- Here should be the proof which we're not providing as per instructions
  sorry

end ordered_pair_solution_l167_167985


namespace max_profit_l167_167086

noncomputable def fixed_cost := 20000
noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 8 then (1/3) * x^2 + 2 * x else 7 * x + 100 / x - 37
noncomputable def sales_price_per_unit : ℝ := 6
noncomputable def profit (x : ℝ) : ℝ :=
  let revenue := sales_price_per_unit * x
  let cost := fixed_cost / 10000 + variable_cost x
  revenue - cost

theorem max_profit : ∃ x : ℝ, (0 < x) ∧ (15 = profit 10) :=
by {
  sorry
}

end max_profit_l167_167086


namespace paint_needed_l167_167156

theorem paint_needed (wall_area : ℕ) (coverage_per_gallon : ℕ) (number_of_coats : ℕ) (h_wall_area : wall_area = 600) (h_coverage_per_gallon : coverage_per_gallon = 400) (h_number_of_coats : number_of_coats = 2) : 
    ((number_of_coats * wall_area) / coverage_per_gallon) = 3 :=
by
  sorry

end paint_needed_l167_167156


namespace baron_boasting_l167_167322

noncomputable def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)

theorem baron_boasting (A B C D : ℝ) (AB AC AD BD DC : ℝ)
    (h1: is_isosceles AB AC)
    (h2: AB = AC)
    (h3: BD ≠ DC)
    (h4: D ≠ B)
    (h5: D ≠ C)
    : ∀ (T1 T2 : ℝ), ¬ is_isosceles (T1 + T2) (T1 + T2) := by
  sorry

end baron_boasting_l167_167322


namespace workers_to_build_cars_l167_167066

theorem workers_to_build_cars (W : ℕ) (hW : W > 0) : 
  (∃ D : ℝ, D = 63 / W) :=
by
  sorry

end workers_to_build_cars_l167_167066


namespace sin_cos_third_quadrant_l167_167061

-- Definitions
def is_in_third_quadrant (α : ℝ) : Prop := α ∈ (set.Ioo (π) (3 * π / 2))

-- Theorem statement
theorem sin_cos_third_quadrant (α : ℝ) (h_alpha: is_in_third_quadrant α) : sin α + cos α = -1 :=
sorry

end sin_cos_third_quadrant_l167_167061


namespace f_neg1_lt_f1_l167_167038

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x * f' 2

theorem f_neg1_lt_f1 (h_diff : ∀ x : ℝ, differentiable_at ℝ f x) :
  f (-1) < f 1 := by
  -- Applying the given conditions and definitions
  sorry

end f_neg1_lt_f1_l167_167038


namespace total_roses_given_to_friends_l167_167837

-- Definition of the given conditions
def total_money : ℝ := 300
def cost_per_rose : ℝ := 2
def total_roses := total_money / cost_per_rose

def roses_for_jenna := (1 / 3) * total_roses
def roses_for_imma := (1 / 2) * total_roses

-- Theorem stating the total number of roses given to friends
theorem total_roses_given_to_friends : roses_for_jenna + roses_for_imma = 125 :=
by
  sorry

end total_roses_given_to_friends_l167_167837


namespace sum_f_251point6_l167_167543

noncomputable def f (x : ℝ) : ℝ := 3 / (4^x + 3)

theorem sum_f_251point6 :
  (∑ k in Finset.range 502, f ((k + 1) / 503)) = 251.6 :=
by
  sorry

end sum_f_251point6_l167_167543


namespace sin_alpha_obtuse_l167_167415

theorem sin_alpha_obtuse (α : ℝ) (h1 : α > π / 2 ∧ α < π) (h2 : 3 * sin (2 * α) = cos α) : sin α = 1 / 6 :=
by
  sorry

end sin_alpha_obtuse_l167_167415


namespace fraction_equality_l167_167368

theorem fraction_equality : (16 : ℝ) / (8 * 17) = (1.6 : ℝ) / (0.8 * 17) := 
sorry

end fraction_equality_l167_167368


namespace tangent_line_at_1_decreasing_interval_l167_167015

-- Define the function y = x^2 - 3 ln(x)
def f (x : ℝ) : ℝ := x^2 - 3 * Real.log x

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := (2 * x^2 - 3) / x

-- Define the tangent line equation condition
def is_tangent_line (f : ℝ → ℝ) (x : ℝ) (line : ℝ → ℝ) : Prop :=
  line x = f x ∧ ∀ h, h ≠ x → (line h - f h) / (h - x) = f' x

-- Define the function's decreasing interval condition
def is_decreasing_on (f' : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → f' x < 0

-- Theorem for Part (1): the tangent line at x = 1 is x + y - 2 = 0
theorem tangent_line_at_1 : is_tangent_line f 1 (λ x, -x + 2) :=
  sorry

-- Theorem for Part (2): the function is decreasing on the interval (0, sqrt(6)/2)
theorem decreasing_interval : is_decreasing_on f' 0 (Real.sqrt 6 / 2) :=
  sorry

end tangent_line_at_1_decreasing_interval_l167_167015


namespace simplify_expr1_simplify_expr2_l167_167931

variable (x y : ℝ)

theorem simplify_expr1 : 
  3 * x^2 - 2 * x * y + y^2 - 3 * x^2 + 3 * x * y = x * y + y^2 :=
by
  sorry

theorem simplify_expr2 : 
  (7 * x^2 - 3 * x * y) - 6 * (x^2 - 1/3 * x * y) = x^2 - x * y :=
by
  sorry

end simplify_expr1_simplify_expr2_l167_167931


namespace forty_percent_of_number_l167_167573

theorem forty_percent_of_number (N : ℚ)
  (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 15) : 0.40 * N = 180 := 
by 
  sorry

end forty_percent_of_number_l167_167573


namespace volume_of_rotated_isosceles_right_triangle_l167_167720

theorem volume_of_rotated_isosceles_right_triangle (h leg_length : ℝ) (π : ℝ) 
  (h₁ : isosceles_right_triangle leg_length) 
  (h₂ : leg_length = 1) 
  (h₃ : h = 1) :
  volume_of_cone (1 / 3 * π * leg_length ^ 2 * h) = π / 3 :=
by
  sorry

end volume_of_rotated_isosceles_right_triangle_l167_167720


namespace distinct_three_digit_numbers_with_even_digits_l167_167460

theorem distinct_three_digit_numbers_with_even_digits : 
  let even_digits := {0, 2, 4, 6, 8} in
  (∃ (hundreds options : Finset ℕ) (x : ℕ), 
    hundreds = {2, 4, 6, 8} ∧ 
    options = even_digits ∧ 
    x = Finset.card hundreds * Finset.card options * Finset.card options ∧ 
    x = 100) :=
by
  let even_digits := {0, 2, 4, 6, 8}
  exact ⟨{2, 4, 6, 8}, even_digits, 100, rfl, rfl, sorry, rfl⟩

end distinct_three_digit_numbers_with_even_digits_l167_167460


namespace lola_mini_cupcakes_l167_167913

theorem lola_mini_cupcakes :
  ∃ c : ℕ, (c + 10 + 8 + 16 + 12 + 14 = 73) ∧ (c = 13) :=
begin
  use 13,
  split,
  { simp },
  { refl }
end

end lola_mini_cupcakes_l167_167913


namespace sum_of_three_integers_mod_53_l167_167670

theorem sum_of_three_integers_mod_53 (a b c : ℕ) (h1 : a % 53 = 31) 
                                     (h2 : b % 53 = 22) (h3 : c % 53 = 7) : 
                                     (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_three_integers_mod_53_l167_167670


namespace probability_exactly_one_each_is_correct_l167_167860

def probability_one_each (total forks spoons knives teaspoons : ℕ) : ℚ :=
  (forks * spoons * knives * teaspoons : ℚ) / ((total.choose 4) : ℚ)

theorem probability_exactly_one_each_is_correct :
  probability_one_each 34 8 9 10 7 = 40 / 367 :=
by sorry

end probability_exactly_one_each_is_correct_l167_167860


namespace triangle_equality_l167_167144

theorem triangle_equality {A B C D B' C' D' K L M : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space B'] [metric_space C'] [metric_space D'] [metric_space K] [metric_space L] [metric_space M] 
  (h1 : is_triangle A B C) 
  (h2 : is_regular_triangle B C D) 
  (h3 : measure_along_ray D B A B' ∧ measure_along_ray D C C_1 C') 
  (h4 : measure_along_ray B' C' A D') 
  (h5 : ∃ K, same_line D D' ∧ on_line K BC) 
  (h6 : parallel_through_point K AB AC L) 
  (h7 : foot_of_perpendicular K AB M) :
  KL = KM :=
sorry

end triangle_equality_l167_167144


namespace even_function_inequality_l167_167027

open Real

noncomputable def f (x a : ℝ) : ℝ := 2^|x - a|

theorem even_function_inequality (f_even : ∀ x : ℝ, f x a = f (-x) a) (a_zero : a = 0) :
  let log2_3 := Real.log 3 / Real.log 2
  let log05_5 := Real.log 5 / Real.log (1/2)
  f a_zero < f log2_3 < f log05_5 :=
by
  -- Placeholder for the actual proof
  sorry

end even_function_inequality_l167_167027


namespace cost_price_of_computer_table_l167_167619

/-- The owner of a furniture shop charges 20% more than the cost price. 
    Given that the customer paid Rs. 3000 for the computer table, 
    prove that the cost price of the computer table was Rs. 2500. -/
theorem cost_price_of_computer_table (CP SP : ℝ) (h1 : SP = CP + 0.20 * CP) (h2 : SP = 3000) : CP = 2500 :=
by {
  sorry
}

end cost_price_of_computer_table_l167_167619


namespace sum_inferior_numbers_2016_l167_167435

def is_inferior (n : ℕ) : Prop :=
  (n ≠ 0) ∧ (∃ k : ℕ, n = 2^k - 2)

def sum_inferior_numbers (max_n : ℕ) : ℕ :=
  Nat.sum (List.filter is_inferior (List.range (max_n + 1)))

theorem sum_inferior_numbers_2016 : sum_inferior_numbers 2016 = 2026 :=
  sorry

end sum_inferior_numbers_2016_l167_167435


namespace inflation_two_years_real_rate_of_return_l167_167665

-- Proof Problem for Question 1
theorem inflation_two_years :
  ((1 + 0.015)^2 - 1) * 100 = 3.0225 :=
by
  sorry

-- Proof Problem for Question 2
theorem real_rate_of_return :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by
  sorry

end inflation_two_years_real_rate_of_return_l167_167665


namespace nature_of_roots_indeterminate_l167_167062

variable (a b c : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem nature_of_roots_indeterminate (h : b^2 - 4 * a * c = 0) : 
  ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) = 0) ∨ ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) < 0) ∨ ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) > 0) :=
sorry

end nature_of_roots_indeterminate_l167_167062


namespace compare_abc_l167_167784

noncomputable def a : ℝ := (0.9)^(1/3)
noncomputable def b : ℝ := (1/3)^(0.9)
noncomputable def c : ℝ := (1/2) * (Real.log 9 / Real.log 27)

theorem compare_abc :
  c < b ∧ b < a :=
by
  sorry

end compare_abc_l167_167784


namespace range_of_a_l167_167425

noncomputable def f (a x : ℝ) : ℝ := log a (a * x ^ 2 - x + 3)

theorem range_of_a (a : ℝ) :
  (2 ≤ x ∧ x ≤ 4 → f a x > 0) →
  ((a > 1 ∧ (∀ x, 2 ≤ x ∧ x ≤ 4 → (a * x ^ 2 - x + 3) > (a * (x + 1) ^ 2 - (x + 1) + 3))) ∨
   (0 < a ∧ a < 1 ∧ (∀ x, 2 ≤ x ∧ x ≤ 4 → (a * x ^ 2 - x + 3) < (a * (x + 1) ^ 2 - (x + 1) + 3)))) →
  a ∈ Ioo (1 / 16 : ℝ) (1 / 8) ∪ Ioi 1 :=
begin
  sorry
end

end range_of_a_l167_167425


namespace otimes_example_l167_167775

variable (x y z : ℝ)

def otimes (x y z : ℝ) : ℝ := x / (y - z)

theorem otimes_example :
  otimes (otimes 2 5 3 ^ 2) (otimes 4 6 2) (otimes 5 2 6) = 4 / 9 := by
  sorry

end otimes_example_l167_167775


namespace distance_between_city_A_and_B_is_180_l167_167953

theorem distance_between_city_A_and_B_is_180
  (D : ℝ)
  (h1 : ∀ T_C : ℝ, T_C = D / 30)
  (h2 : ∀ T_D : ℝ, T_D = T_C - 1)
  (h3 : ∀ V_D : ℝ, V_D > 36 → T_D = D / V_D) :
  D = 180 := 
by
  sorry

end distance_between_city_A_and_B_is_180_l167_167953


namespace number_of_license_plates_l167_167092

-- Define the alphabet size and digit size constants.
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the number of letters in the license plate.
def letters_in_plate : ℕ := 3

-- Define the number of digits in the license plate.
def digits_in_plate : ℕ := 4

-- Calculating the total number of license plates possible as (26^3) * (10^4).
theorem number_of_license_plates : 
  (num_letters ^ letters_in_plate) * (num_digits ^ digits_in_plate) = 175760000 :=
by
  sorry

end number_of_license_plates_l167_167092


namespace four_digit_numbers_count_l167_167447

theorem four_digit_numbers_count : 
  ∃ n : ℕ, n = 9 ∧ 
  (∀ (d1 d2 d3 d4 : ℕ), 
     multiset.of_list [d1, d2, d3, d4] = multiset.of_list [2, 0, 2, 5] → 
     d1 ≠ 0 → 
     (d1 * 1000 + d2 * 100 + d3 * 10 + d4).to_string.length = 4) :=
sorry

end four_digit_numbers_count_l167_167447


namespace initial_volume_of_mixture_l167_167281

-- Define the initial condition volumes for p and q
def initial_volumes (x : ℕ) : ℕ × ℕ := (3 * x, 2 * x)

-- Define the final condition volumes for p and q after adding 2 liters of q
def final_volumes (x : ℕ) : ℕ × ℕ := (3 * x, 2 * x + 2)

-- Define the initial total volume of the mixture
def initial_volume (x : ℕ) : ℕ := 5 * x

-- The theorem stating the solution
theorem initial_volume_of_mixture (x : ℕ) (h : 3 * x / (2 * x + 2) = 5 / 4) : 5 * x = 25 := 
by sorry

end initial_volume_of_mixture_l167_167281


namespace set_A_elements_l167_167024

variable {M : Set ℝ} (h : ∀ a ∈ M, (1 + a) / (1 - a) ∈ M)

theorem set_A_elements : 
  (2 ∈ M) → 
  (−3 ∈ M) → 
  (−(1 / 2) ∈ M) → 
  ((1 / 3) ∈ M) → 
  M = {2, -3, -1 / 2, 1 / 3} :=
by
  sorry  -- Proof omitted

end set_A_elements_l167_167024


namespace reflection_matrix_correct_l167_167536

noncomputable def reflection_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1/3, 4/3, 2/3], ![5/3, 1/3, 2/3], ![1/3, 4/3, 2/3]]

theorem reflection_matrix_correct (u : Fin 3 → ℝ) :
  let n : Fin 3 → ℝ := ![2, -1, 1]
  let q := u - (u ⬝ᴛ n) / (n ⬝ᴛ n) • n
  let s := 2 • q - u
  reflection_matrix ⬝ u = s :=
by
  let n : Fin 3 → ℝ := ![2, -1, 1]
  have h_n_dot_n : n ⬝ᴛ n = 6 := sorry
  let q := u - (u ⬝ᴛ n) / (n ⬝ᴛ n) • n
  let s := 2 • q - u
  have h_projection : reflection_matrix ⬝ u = s := sorry
  exact h_projection
  sorry

#check @reflection_matrix_correct

end reflection_matrix_correct_l167_167536


namespace sum_of_200_consecutive_integers_l167_167239

theorem sum_of_200_consecutive_integers 
  (n : ℕ) (H : n ∈ {2000200000, 3000300000, 4000400000, 5000500000, 6000600000}) :
  ¬∃ k : ℕ, n = 200 * k + 20100 := 
by
  sorry

end sum_of_200_consecutive_integers_l167_167239


namespace PQ_eq_RS_l167_167612

-- Definitions and conditions
variables {P B C U V S Q W R : Type*}

-- Incircle touches BC at U and PC at V
-- S is on BC such that BS = CU
-- PS meets the incircle nearer to P at Q
-- W on PC such that PW = CV
-- BW and PS meet at R
axiom incircle_touches_BC_PC_at_U_V (triangle : P → B → C) (incircle : incircle P B C) :
  touches incircle B C U ∧ touches incircle P C V

axiom S_on_BC_such_that_BS_eq_CU (S : BC) (BS CU :_length) : equiv_length BS CU

axiom PS_meets_incircle_at_Q (PS : S → ⟦circle P⟧) :
  meets_near P PS Q

axiom W_on_PC_such_that_PW_eq_CV (W : PC) (PW CV :length) : equiv_length PW CV

axiom BW_PS_meet_at_R (BW PS : line) :
  intersect BW PS R

-- Prove that PQ = RS
theorem PQ_eq_RS :
  ∀ {P B C U V S Q W R : Type*}
  (incircle_touches_BC_PC_at_U_V : touches incircle B C U ∧ touches incircle P C V)
  (S_on_BC_such_that_BS_eq_CU : equiv_length (BS S) (CU S))
  (PS_meets_incircle_at_Q: meets_near P PS Q)
  (W_on_PC_such_that_PW_eq_CV: equiv_length (PW W) (CV W))
  (BW_PS_meet_at_R: intersect BW PS R),
  equiv_length (PQ Q) (RS R) :=
sorry

end PQ_eq_RS_l167_167612


namespace organization_members_count_l167_167496

-- Definition of the problem
def members_belong_two_committees (members : Type) (committees : Type) [Fintype members] [Fintype committees] [DecidableEq members] [DecidableEq committees] : Prop :=
  ∀ m : members, ∃ c1 c2 : committees, c1 ≠ c2 ∧ ∀ c : committees, c = c1 ∨ c = c2

def pairs_of_committees_unique_member (members : Type) (committees : Type) [Fintype members] [Fintype committees] [DecidableEq members] [DecidableEq committees] : Prop :=
  ∀ c1 c2 : committees, c1 ≠ c2 → ∃! m : members, (m belongs to c1) ∧ (m belongs to c2)

theorem organization_members_count (members committees : Type)
  [Fintype members] [Fintype committees] [DecidableEq members] [DecidableEq committees]
  [card_committees : Fintype.card committees = 5] :
    members_belong_two_committees members committees →
    pairs_of_committees_unique_member members committees →
    Fintype.card members = 10 :=
by
  sorry

end organization_members_count_l167_167496


namespace exists_line_satisfying_conditions_l167_167703

def pointP : ℝ × ℝ := (4 / 3, 2)

def trianglePerimeter (a b : ℝ) : ℝ := a + b + Real.sqrt (a^2 + b^2)
def triangleArea (a b : ℝ) : ℝ := (a * b) / 2

theorem exists_line_satisfying_conditions :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  (trianglePerimeter a b = 12 ∧ triangleArea a b = 6) ∧
  (4 / 3 / a + 2 / b = 1) ∧ (3 - 3 * a = 0) ∧ (4 - 4 * b = 0) :=
by
  have eq1: trianglePerimeter = sorry,
  have eq2: triangleArea = sorry,
  sorry

end exists_line_satisfying_conditions_l167_167703


namespace min_max_values_l167_167614

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sin x

theorem min_max_values : 
  ∃ (m M : ℝ), (∀ x : ℝ, f(x) ≥ m) ∧ (∀ x : ℝ, f(x) ≤ M) ∧ m = -3 ∧ M = 3 / 2 :=
by
  sorry

end min_max_values_l167_167614


namespace sum_of_four_squares_express_689_as_sum_of_squares_l167_167417

theorem sum_of_four_squares (m n : ℕ) (hmn : m ≠ n) : 
  ∃ a b c d : ℕ, m^4 + 4 * n^4 = a^2 + b^2 + c^2 + d^2 :=
sorry

theorem express_689_as_sum_of_squares : 
  ∃ a b c d : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ 689 = a^2 + b^2 + c^2 + d^2 :=
begin
  use [21, 14, 6, 4],
  split, norm_num, split, norm_num, split, norm_num, norm_num,
end

end sum_of_four_squares_express_689_as_sum_of_squares_l167_167417


namespace percentage_of_required_amount_l167_167625

variable (P Q R : ℝ)
-- Defining the original price (P) and the required quantity (Q)

-- The net increase in price
def new_price := 1.25 * P

-- The expenditure change
def original_expenditure := P * Q
def new_expenditure := new_price * (R / 100) * Q

-- The net difference in expenditure is given as 20
def net_difference := original_expenditure - new_expenditure = 20

-- We need to prove that R = 16 in this setup
theorem percentage_of_required_amount :
  net_difference → R = 16 := 
sorry

end percentage_of_required_amount_l167_167625


namespace log_base_3_of_one_over_81_l167_167347

theorem log_base_3_of_one_over_81 : log 3 (1 / 81) = -4 := by
  sorry

end log_base_3_of_one_over_81_l167_167347


namespace krakozyabrs_total_count_l167_167105

theorem krakozyabrs_total_count :
  (∃ (T : ℕ), 
  (∀ (H W n : ℕ),
    0.2 * H = n ∧ 0.25 * W = n ∧ H = 5 * n ∧ W = 4 * n ∧ T = H + W - n ∧ 25 < T ∧ T < 35) ∧ 
  T = 32) :=
by sorry

end krakozyabrs_total_count_l167_167105


namespace number_of_valid_numbers_l167_167896

theorem number_of_valid_numbers (n : ℕ) (h : 0 < n) :
  let S := { x : ℕ | (nat.digits 10 x).length = n ∧ (∀ d ∈ (nat.digits 10 x), d ∈ {3, 5, 7, 9}) ∧ x % 3 = 0 } in
  |S| = (4^n + 2) / 3 :=
by
  sorry

end number_of_valid_numbers_l167_167896


namespace animal_population_l167_167977

theorem animal_population
  (number_of_lions : ℕ)
  (number_of_leopards : ℕ)
  (number_of_elephants : ℕ)
  (h1 : number_of_lions = 200)
  (h2 : number_of_lions = 2 * number_of_leopards)
  (h3 : number_of_elephants = (number_of_lions + number_of_leopards) / 2) :
  number_of_lions + number_of_leopards + number_of_elephants = 450 :=
sorry

end animal_population_l167_167977


namespace circle_inscribed_angles_l167_167605

theorem circle_inscribed_angles (O : Type) (circle : Set O) (A B C D E F G H I J K L : O) 
  (P : ℕ) (n : ℕ) (x_deg_sum y_deg_sum : ℝ)  
  (h1 : n = 12) 
  (h2 : x_deg_sum = 45) 
  (h3 : y_deg_sum = 75) :
  x_deg_sum + y_deg_sum = 120 :=
by
  /- Proof steps are not required -/
  apply sorry

end circle_inscribed_angles_l167_167605


namespace difference_in_sales_l167_167263

def daily_pastries : ℕ := 20
def daily_bread : ℕ := 10
def today_pastries : ℕ := 14
def today_bread : ℕ := 25
def price_pastry : ℕ := 2
def price_bread : ℕ := 4

theorem difference_in_sales : (daily_pastries * price_pastry + daily_bread * price_bread) - (today_pastries * price_pastry + today_bread * price_bread) = -48 :=
by
  -- Proof will go here
  sorry

end difference_in_sales_l167_167263


namespace distinct_three_digit_even_integers_count_l167_167465

theorem distinct_three_digit_even_integers_count : 
  let even_digits := {0, 2, 4, 6, 8}
  ∃ h : Finset ℕ, h = {2, 4, 6, 8} ∧ 
     (∏ x in h, 5 * 5) = 100 :=
by
  let even_digits : Finset ℕ := {0, 2, 4, 6, 8}
  let h : Finset ℕ := {2, 4, 6, 8}
  have : ∏ x in h, 5 * 5 = 100 := sorry
  exact ⟨h, rfl, this⟩

end distinct_three_digit_even_integers_count_l167_167465


namespace parallel_lines_solution_count_l167_167788

noncomputable def number_of_parallel_lines
  (circle : Type) 
  (a b i : line)
  (ha : intersects a circle)
  (hb : intersects b circle) : ℕ :=
sorry

theorem parallel_lines_solution_count
  (circle : Type)
  (a b i : line)
  (ha : intersects a circle)
  (hb : intersects b circle)
  : (number_of_parallel_lines circle a b i ha hb) ∈ {0, 1, 2, 3} :=
sorry

end parallel_lines_solution_count_l167_167788


namespace nancy_potatoes_l167_167916

theorem nancy_potatoes (sandy_potatoes total_potatoes : ℕ) (h1 : sandy_potatoes = 7) (h2 : total_potatoes = 13) :
    total_potatoes - sandy_potatoes = 6 :=
by
  sorry

end nancy_potatoes_l167_167916


namespace equation_curve_C_tangent_constant_value_l167_167873

-- Define the essential conditions
def circle_O : Set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 3 }
def M : ℝ × ℝ := (real.sqrt 2, 0)
def M' : ℝ × ℝ := (-real.sqrt 2, 0)

-- Define the locus condition for N and curve C
def N' (N : ℝ × ℝ) : Prop :=
  (∃ x y, N = (x, y) ∧
  let midpoint := ((N.1 + M.1) / 2, (N.2 + M.2) / 2) in
  distance (0, 0) midpoint + distance (0, 0) (N.1 - midpoint.1, N.2 - midpoint.2) = distance (N.1, N.2) midpoint ∧
  midpoint ∈ circle_O)

def curve_C : Set (ℝ × ℝ) :=  { p | (p.1^2 / 3) + p.2^2 = 1 }

-- Proof Problem (1): Prove the equation of curve C
theorem equation_curve_C : 
  ∀ N, N' N → N ∈ curve_C :=
sorry

-- Additional conditions for part (2)
def P : Set (ℝ×ℝ) :=  curve_C 
def tangent_circle (P : ℝ × ℝ) : Set (ℝ × ℝ) := { x | (x.1 - P.1)^2 + (x.2 - P.2)^2 = (real.sqrt 3 / 2)^2}

def is_tangent (O : ℝ × ℝ) (A B : Set (ℝ × ℝ)) : Prop := ∀ x ∈ A, ∀ y ∈ B, distance O x + distance O y = distance x y

-- Proof Problem (2): Prove |OA|^2 + |OB|^2 is constant
theorem tangent_constant_value :
  ∀ (P : ℝ × ℝ) (A B : ℝ × ℝ), 
  P ∈ curve_C ∧
  is_tangent (0,0) (tangent_circle P) A ∧
  A ∈ curve_C ∧
  B ∈ curve_C 
  → (distance (0,0) A)^2 + (distance (0,0) B)^2 = 4 :=
sorry

end equation_curve_C_tangent_constant_value_l167_167873


namespace count_distinct_three_digit_even_numbers_l167_167468

theorem count_distinct_three_digit_even_numbers : 
  let even_digits := {0, 2, 4, 6, 8} in
  let first_digit_choices := {2, 4, 6, 8} in
  let second_and_third_digit_choices := even_digits in
  (finset.card first_digit_choices) * 
  (finset.card second_and_third_digit_choices) *
  (finset.card second_and_third_digit_choices) = 100 := by
  let even_digits := {0, 2, 4, 6, 8}
  let first_digit_choices := {2, 4, 6, 8}
  let second_and_third_digit_choices := even_digits
  have h1 : finset.card first_digit_choices = 4 := by simp
  have h2 : finset.card second_and_third_digit_choices = 5 := by simp
  calc (finset.card first_digit_choices) * 
       (finset.card second_and_third_digit_choices) *
       (finset.card second_and_third_digit_choices)
       = 4 * 5 * 5 : by rw [h1, h2]
    ... = 100 : by norm_num

end count_distinct_three_digit_even_numbers_l167_167468


namespace log3_1_over_81_l167_167359

theorem log3_1_over_81 : log 3 (1 / 81) = -4 := by
  have h1 : 1 / 81 = 3 ^ (-4) := by
    -- provide a proof or skip with "sory"
    sorry
  have h2 : log 3 (3 ^ (-4)) = -4 := by
    -- provide a proof or skip with "sorry"
    sorry
  exact eq.trans (log 3) (congr_fun (h1.symm h2))

end log3_1_over_81_l167_167359


namespace four_digit_numbers_count_l167_167839

open Nat

theorem four_digit_numbers_count :
  let valid_a := [5, 6]
  let valid_d := 0
  let valid_bc_pairs := [(3, 4), (3, 6)]
  valid_a.length * 1 * valid_bc_pairs.length = 4 :=
by
  sorry

end four_digit_numbers_count_l167_167839


namespace find_cost_10_pound_bag_l167_167276

def cost_5_pound_bag : ℝ := 13.82
def cost_25_pound_bag : ℝ := 32.25
def minimum_required_weight : ℝ := 65
def maximum_required_weight : ℝ := 80
def least_possible_cost : ℝ := 98.75
def cost_10_pound_bag (cost : ℝ) : Prop :=
  ∃ n m l, 
    (n * 5 + m * 10 + l * 25 ≥ minimum_required_weight) ∧
    (n * 5 + m * 10 + l * 25 ≤ maximum_required_weight) ∧
    (n * cost_5_pound_bag + m * cost + l * cost_25_pound_bag = least_possible_cost)

theorem find_cost_10_pound_bag : cost_10_pound_bag 2 := 
by
  sorry

end find_cost_10_pound_bag_l167_167276


namespace solution_l167_167147

noncomputable def problem (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  (∀ x : ℝ, x^2 - 14 * p * x - 15 * q = 0 → x = r ∨ x = s) ∧
  (∀ x : ℝ, x^2 - 14 * r * x - 15 * s = 0 → x = p ∨ x = q)

theorem solution (p q r s : ℝ) (h : problem p q r s) : p + q + r + s = 3150 :=
sorry

end solution_l167_167147


namespace new_computer_price_l167_167851

theorem new_computer_price (d : ℕ) (h : 2 * d = 560) : d + 3 * d / 10 = 364 :=
by
  sorry

end new_computer_price_l167_167851


namespace nonreal_root_of_z_cubed_eq_one_sum_expression_equals_12_l167_167900

noncomputable def omega : ℂ := -1 / 2 + complex.I * (real.sqrt 3) / 2

variable {n : ℕ}
variable (a : fin n → ℝ)

-- Conditions
theorem nonreal_root_of_z_cubed_eq_one : omega^3 = 1 := by sorry

-- Main theorem
theorem sum_expression_equals_12 (h : ∑ k, 1 / (a k + omega) = 4 + 3 * complex.I) :
  (∑ k, (3 * a k - 2) / ((a k)^2 - a k + 1)) = 12 := by sorry

end nonreal_root_of_z_cubed_eq_one_sum_expression_equals_12_l167_167900


namespace find_omega_monotonicity_f_l167_167429

noncomputable def period_condition (ω : ℝ) (hω : 0 < ω) :=
  ∃ T > 0, ∀ x, 4 * Real.cos (ω * x) * Real.cos (ω * x + Real.pi / 3) = 
  4 * Real.cos (ω * (x + T)) * Real.cos (ω * (x + T) + Real.pi / 3)

theorem find_omega (ω : ℝ) (hω : 0 < ω) (hT : period_condition ω hω) :
  ω = 1 :=
begin
  sorry
end

theorem monotonicity_f (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5 * Real.pi / 6) :
  let f (x : ℝ) := 2 * Real.cos (2 * x + Real.pi / 3) + 1 in
  if (0 ≤ x ∧ x ≤ Real.pi / 3)
  then ∀ a b, (a ≤ b ∧ 0 ≤ a ∧ b ≤ Real.pi / 3) → f a ≥ f b
  else ∀ a b, (a ≤ b ∧ Real.pi / 3 ≤ a ∧ b ≤ 5 * Real.pi / 6) → f a ≤ f b :=
begin
  sorry
end

end find_omega_monotonicity_f_l167_167429


namespace incorrect_statement_of_quadratic_eq_l167_167675

theorem incorrect_statement_of_quadratic_eq (a b c : ℝ) (h : a = 1 ∧ b = 1 ∧ c = 3) :
  let discriminant := b^2 - 4 * a * c in
  ¬ (discriminant = 0 ∧ ∃ x : ℝ, a * x^2 + b * x + c = 0) :=
by
  intro h_discriminant
  sorry

end incorrect_statement_of_quadratic_eq_l167_167675


namespace tasty_pair_iff_isogonal_conjugate_tasty_pair_T_at_T_B_T_C_l167_167524

noncomputable def isogonal_conjugate (P Q A B C : Point) : Prop := sorry

noncomputable def is_tasty_pair (P Q A B C : Point) : Prop := sorry

theorem tasty_pair_iff_isogonal_conjugate (A B C P Q : Point) (hABC : triangle A B C) (hscalene: scalene A B C) (hacute: acute A B C) :
  (is_tasty_pair P Q A B C) ↔ isogonal_conjugate P Q A B C := sorry

theorem tasty_pair_T_at_T_B_T_C (T_A T_B T_C P Q : Point) (hT: triangle T_A T_B T_C) (hscalene_T: scalene T_A T_B T_C) (hacute_T: acute T_A T_B T_C) (htasty: is_tasty_pair P Q T_A T_B T_C) :
  is_tasty_pair P Q T_A T_B T_C := sorry

end tasty_pair_iff_isogonal_conjugate_tasty_pair_T_at_T_B_T_C_l167_167524
