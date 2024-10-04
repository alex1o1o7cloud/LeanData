import Mathlib

namespace exterior_angle_DEF_eq_96_43_l80_80508

-- Define the conditions
def is_regular_heptagon (A B C D E : Type) : Prop := 
  -- Placeholder definition for regular heptagon
  sorry 

def is_regular_octagon (A E F G H I J : Type) : Prop := 
  -- Placeholder definition for regular octagon
  sorry 

def coplanar (A B C D E F G H I J : Type) : Prop := 
  -- Placeholder definition for coplanar property
  sorry 

def drawn_opposite_sides_of_segment (A E : Type) (ABCD EFGHIJ : Type) : Prop := 
  -- Placeholder definition for drawn on opposite sides of the segment
  sorry 

-- Define the theorem
theorem exterior_angle_DEF_eq_96_43 (A B C D E F G H I J : Type) 
  (h1 : is_regular_heptagon A B C D E) 
  (h2 : is_regular_octagon A E F G H I J) 
  (h3 : coplanar A B C D E F G H I J)
  (h4 : drawn_opposite_sides_of_segment A E (A B C D E) (A E F G H I J)) : 
  (exterior_angle DEF = 96.43) := 
begin
  -- Proof to be completed
  sorry
end

end exterior_angle_DEF_eq_96_43_l80_80508


namespace sum_of_factors_36_l80_80687

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80687


namespace value_of_a_sub_b_l80_80425

theorem value_of_a_sub_b (a b : ℝ) (h1 : abs a = 8) (h2 : abs b = 5) (h3 : a > 0) (h4 : b < 0) : a - b = 13 := 
  sorry

end value_of_a_sub_b_l80_80425


namespace p_is_rational_iff_a3_b2_l80_80927

-- Define the problem conditions
def is_rational (p : ℚ) : Prop := ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ p = (Real.sqrt 2 + Real.sqrt a) / (Real.sqrt 3 + Real.sqrt b)

-- State the theorem.
theorem p_is_rational_iff_a3_b2 : ∀ (a b : ℕ), is_rational ⟨a, b⟩ ↔ (a = 3 ∧ b = 2) :=
  by { sorry } -- Placeholder for the proof

end p_is_rational_iff_a3_b2_l80_80927


namespace matrix_zero_solution_l80_80946

theorem matrix_zero_solution (n : ℕ) (A : matrix (fin n) (fin n) ℝ) (h1 : 0 < n)
  (h2 : ∀ (λ : ℝ), is_eigenvalue A λ)
  (k : ℕ) (h3 : n ≤ k) (h4 : A + matrix.pow A k = Aᵀ) : A = 0 :=
by
  sorry

end matrix_zero_solution_l80_80946


namespace sum_of_factors_36_l80_80682

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80682


namespace sum_factors_36_l80_80794

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80794


namespace solve_for_x_l80_80133

variable {R : Type*} [Field R]

def matrix_3x3 (x : R) : Matrix (Fin 3) (Fin 3) R :=
  !![3,  1, -1;
     4,  x,  2;
     1,  3,  6]

theorem solve_for_x (x : R) :
  matrix_3x3 x.det = 0 → x = 52 / 19 := by
  sorry

end solve_for_x_l80_80133


namespace odd_factors_of_450_l80_80404

theorem odd_factors_of_450 : 
  let factors_count (n : ℕ) := (n.factors.count (λ d, d % 2 = 1))
  factors_count 450 = 9 :=
by
  sorry

end odd_factors_of_450_l80_80404


namespace sum_series_eq_4_div_9_l80_80182

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80182


namespace sum_series_eq_four_ninths_l80_80203

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80203


namespace train_crossing_time_l80_80086

noncomputable def speed_kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000) / 3600

theorem train_crossing_time
  (train_speed_kmh : ℝ) (train_time_pole : ℝ) (stationary_train_length : ℝ)
  (train_speed_ms := speed_kmh_to_ms train_speed_kmh)
  (train_length := train_speed_ms * train_time_pole)
  (total_length := train_length + stationary_train_length) :
  train_speed_kmh = 72 → train_time_pole = 12 → stationary_train_length = 300 →
  (total_length / train_speed_ms) = 27 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have speed_ms : train_speed_ms = speed_kmh_to_ms 72 := by
    unfold speed_kmh_to_ms
    norm_num
  rw [speed_ms]
  have train_length_eq : (speed_kmh_to_ms 72) * 12 = 240 := by
    unfold speed_kmh_to_ms
    norm_num
  rw [train_length_eq]
  norm_num
  sorry

end train_crossing_time_l80_80086


namespace sum_of_factors_36_l80_80685

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80685


namespace find_n_l80_80935

theorem find_n (n : ℤ) (h1 : -180 ≤ n ∧ n ≤ 180) (h2 : real.sin (n * real.pi / 180) = real.cos (810 * real.pi / 180)) : n ∈ {-180, 0, 180} :=
sorry

end find_n_l80_80935


namespace probability_point_in_sphere_l80_80074

theorem probability_point_in_sphere:
  (∫ x in -2..2, ∫ y in -2..2, ∫ z in -2..2, if x^2 + y^2 + z^2 ≤ 4 then 1 else 0) / (8 * 8 * 8)
  = π / 6 :=
sorry

end probability_point_in_sphere_l80_80074


namespace infinite_composite_tendsto_max_f_l80_80834

/-- The number of divisors of a number -/
def tau (a : ℕ) : ℕ := a.divisors.card

/-- Define the function f -/
def f (n : ℕ) : ℕ := tau (n!) - tau ((n - 1)!)

/-- Proof that there exist infinitely many composite numbers n such that for any positive integer m < n, f(m) < f(n) -/
theorem infinite_composite_tendsto_max_f :
  ∃ᶠ n in filter.at_top, ¬nat.prime n ∧ ∀ m < n, f(m) < f(n) :=
sorry

end infinite_composite_tendsto_max_f_l80_80834


namespace max_white_pieces_l80_80865

theorem max_white_pieces (m n : ℕ) : 
  ∃ (w : ℕ), w = m + n - 1 ∧ ∀ (white_pos : set (ℕ × ℕ)), 
  (∀ ⟨i₁, j₁⟩ ∈ white_pos, ∀ ⟨i₂, j₂⟩ ∈ white_pos, (i₁ ≠ i₂ ∨ j₁ ≠ j₂)) → 
  (w = card white_pos) → 
  (∀ (x y : ℕ), (x ≤ m ∧ y ≤ n) → 
   (white_pos ⊆ (univ ∩ (range m).product (range n))) → 
   (∀ ⟨i, j⟩ ∈ white_pos, (∀ k, (i, k) ≠ ⟨i, j⟩ → black ⟨i, k⟩) ∧ (∀ k, (k, j) ≠ ⟨i, j⟩ → black ⟨k, j⟩))) → sorry

end max_white_pieces_l80_80865


namespace sum_factors_36_l80_80763

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80763


namespace groceries_delivered_l80_80096

variables (S C P g T G : ℝ)
theorem groceries_delivered (hS : S = 14500) (hC : C = 14600) (hP : P = 1.5) (hg : g = 0.05) (hT : T = 40) :
  G = 800 :=
by {
  sorry
}

end groceries_delivered_l80_80096


namespace sum_series_eq_4_div_9_l80_80180

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80180


namespace log_equation_l80_80343

theorem log_equation (x y : ℝ) (h1 : 3^x = 2) (h2 : log 3 (9 / 4) = y) : 2 * x + y = 2 :=
by
  sorry

end log_equation_l80_80343


namespace sin_negative_angle_periodic_l80_80120

theorem sin_negative_angle_periodic :
  sin (-17 * Real.pi / 3) = (Real.sqrt 3) / 2 :=
by
  sorry

end sin_negative_angle_periodic_l80_80120


namespace lowestCommonMultiple8and12_l80_80890

/-- Define a predicate for a number being divisible by 8 and 12. --/
def isDivisibleBy8And12 (n : ℕ) : Prop :=
  n % 8 = 0 ∧ n % 12 = 0

/-- Prove that the smallest natural number that is divisible by both 8 and 12 is 24. --/
theorem lowestCommonMultiple8and12 : ∃ n, isDivisibleBy8And12 n ∧ ∀ m, isDivisibleBy8And12 m → n ≤ m :=
  ⟨ 24, ⟨ by norm_num, by norm_num ⟩, 
    by {
      intro m,
      intro h,
      cases h with h8 h12,
      have hcm : 24 ≤ m := Nat.le_of_dvd (Nat.le_of_dvd zero_lt_one _) sorry,
      exact hcm,
    } ⟩

end lowestCommonMultiple8and12_l80_80890


namespace sum_series_equals_4_div_9_l80_80217

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80217


namespace sum_of_factors_36_eq_91_l80_80582

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80582


namespace sum_of_factors_36_l80_80806

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80806


namespace series_sum_eq_four_ninths_l80_80241

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80241


namespace triangle_hypotenuse_l80_80540

theorem triangle_hypotenuse (h : ℝ) :
  ∃ h, (∀ (log3_64 log3_16 : ℝ), log3_64 = real.log 64 / real.log 3 ∧ log3_16 = real.log 16 / real.log 3 →
    (h = (real.sqrt (log3_64 ^ 2 + log3_16 ^ 2)) / (real.sqrt (log3_64 ^ 2 + log3_16 ^ 2) / (real.sqrt (log3_64 ^ 2 + log3_16 ^ 2))) →
    real.pow 3 h = 32)) :=
begin
  sorry
end

end triangle_hypotenuse_l80_80540


namespace sum_of_factors_36_l80_80617

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80617


namespace sample_size_l80_80869

theorem sample_size (n : ℕ) (h1 : n ∣ 36) (h2 : 36 / n ∣ 6) (h3 : (n + 1) ∣ 35) : n = 6 := 
sorry

end sample_size_l80_80869


namespace cloves_of_garlic_needed_l80_80055

def cloves_needed_for_vampires (vampires : ℕ) : ℕ :=
  (vampires * 3) / 2

def cloves_needed_for_wights (wights : ℕ) : ℕ :=
  (wights * 3) / 3

def cloves_needed_for_vampire_bats (vampire_bats : ℕ) : ℕ :=
  (vampire_bats * 3) / 8

theorem cloves_of_garlic_needed (vampires wights vampire_bats : ℕ) :
  cloves_needed_for_vampires 30 + cloves_needed_for_wights 12 + 
  cloves_needed_for_vampire_bats 40 = 72 :=
by
  sorry

end cloves_of_garlic_needed_l80_80055


namespace sum_factors_36_l80_80768

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80768


namespace sum_series_equals_4_div_9_l80_80222

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80222


namespace equal_functions_pair_A_equal_functions_pair_B_equal_functions_pair_D_pairs_A_B_D_same_l80_80825

theorem equal_functions_pair_A (x : ℝ) (h : x ≤ 0) : 
  (sqrt (-2 * x^3) = x * sqrt (-2 * x)) := sorry

theorem equal_functions_pair_B (x : ℝ) : 
  (|x| = sqrt (x^2)) := sorry

theorem equal_functions_pair_D (x : ℝ) (hx : x ≠ 0) : 
  (x / x = x^0) := sorry

theorem pairs_A_B_D_same : 
  (∀ x : ℝ, x ≤ 0 → sqrt (-2 * x^3) = x * sqrt (-2 * x)) ∧ 
  (∀ x : ℝ, |x| = sqrt (x^2)) ∧ 
  (∀ x : ℝ, x ≠ 0 → x / x = x^0) := 
by
  apply and.intro
  · intro x hx
    exact equal_functions_pair_A x hx
  · apply and.intro
    · intro x
      exact equal_functions_pair_B x
    · intro x hx
      exact equal_functions_pair_D x hx

end equal_functions_pair_A_equal_functions_pair_B_equal_functions_pair_D_pairs_A_B_D_same_l80_80825


namespace crayons_total_l80_80119

theorem crayons_total (Billy_crayons : ℝ) (Jane_crayons : ℝ)
  (h1 : Billy_crayons = 62.0) (h2 : Jane_crayons = 52.0) :
  Billy_crayons + Jane_crayons = 114.0 := 
by
  sorry

end crayons_total_l80_80119


namespace sum_series_eq_four_ninths_l80_80205

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80205


namespace sum_series_eq_4_div_9_l80_80183

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80183


namespace sum_factors_36_eq_91_l80_80610

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80610


namespace cos_value_given_sin_l80_80980

theorem cos_value_given_sin (α : Real) (h : sin (π / 6 - α) = 1 / 4) : 
  cos (2 * α + 2 * π / 3) = -7 / 8 :=
by
  sorry

end cos_value_given_sin_l80_80980


namespace distribute_items_5_in_4_identical_bags_l80_80112

theorem distribute_items_5_in_4_identical_bags : 
  let items := 5 in 
  let bags := 4 in 
  number_of_ways_to_distribute items bags = 36 := 
by sorry

end distribute_items_5_in_4_identical_bags_l80_80112


namespace carolyn_spends_36_dollars_on_lace_l80_80903

-- Definitions of the conditions
def length_cuff : ℕ := 50 -- 50 cm for each cuff
def length_hem : ℕ := 300 -- 300 cm for the hem
def waist_fraction : ℝ := 1 / 3 -- fraction for waist length
def length_ruffle : ℕ := 20 -- 20 cm for each ruffle

def num_cuffs : ℕ := 2
def num_ruffles : ℕ := 5

def cost_per_meter : ℝ := 6 -- $6 per meter

-- Derived total lace in cm
def total_lace_cm : ℕ :=
  let waist_length := (length_hem : ℝ) * waist_fraction
  (num_cuffs * length_cuff) + length_hem + (waist_length.toNat) + (num_ruffles * length_ruffle)

-- Derived total lace in meters
def total_lace_m : ℝ := (total_lace_cm : ℝ) / 100

-- Calculate the total cost
theorem carolyn_spends_36_dollars_on_lace : 
  (total_lace_m * cost_per_meter) = 36 := by
  sorry

end carolyn_spends_36_dollars_on_lace_l80_80903


namespace sum_k_over_4k_l80_80153

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80153


namespace perimeter_of_isosceles_triangle_l80_80088

def is_triangle (a b c : ℝ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem perimeter_of_isosceles_triangle (a b : ℝ) (h1 : a = 3 ∨ a = 5) (h2 : b = 3 ∨ b = 5) :
  ∃ p, (p = 11 ∨ p = 13) ∧ is_triangle a a b :=
begin
  sorry
end

end perimeter_of_isosceles_triangle_l80_80088


namespace sum_of_series_l80_80170

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80170


namespace sum_factors_36_eq_91_l80_80611

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80611


namespace expected_difference_after_10_days_l80_80024

-- Define the initial state and transitions
noncomputable def initial_prob (k : ℤ) : ℝ :=
if k = 0 then 1 else 0

noncomputable def transition_prob (k : ℤ) (n : ℕ) : ℝ :=
0.5 * initial_prob k +
0.25 * initial_prob (k - 1) +
0.25 * initial_prob (k + 1)

-- Define event probability for having any wealth difference after n days
noncomputable def p_k_n (k : ℤ) (n : ℕ) : ℝ :=
if n = 0 then initial_prob k
else transition_prob k (n - 1)

-- Use expected value of absolute difference between wealths 
noncomputable def expected_value_abs_diff (n : ℕ) : ℝ :=
Σ' k, |k| * p_k_n k n

-- Finally, state the theorem
theorem expected_difference_after_10_days :
expected_value_abs_diff 10 = 1 :=
by
  sorry

end expected_difference_after_10_days_l80_80024


namespace parabola_directrix_l80_80997

theorem parabola_directrix (p : ℝ) (hp : p > 0) 
  (x1 x2 t : ℝ) 
  (h_intersect : ∃ y1 y2, y1 = x1 + t ∧ y2 = x2 + t ∧ x1^2 = 2 * p * y1 ∧ x2^2 = 2 * p * y2)
  (h_midpoint : (x1 + x2) / 2 = 2) :
  p = 2 → ∃ d : ℝ, d = -1 := 
by
  sorry

end parabola_directrix_l80_80997


namespace sum_series_eq_4_div_9_l80_80291

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80291


namespace sum_of_abs_roots_l80_80144

-- Definitions as per conditions
def polynomial : Polynomial ℝ := 
  Polynomial.C 1 * Polynomial.X ^ 4 - 
  Polynomial.C 6 * Polynomial.X ^ 3 + 
  Polynomial.C 9 * Polynomial.X ^ 2 + 
  Polynomial.C 2 * Polynomial.X - 
  Polynomial.C 4

noncomputable def roots := Roots polynomial -- Assuming a function that gets roots
  
noncomputable def absSum : ℝ := (roots.map Real.abs).sum

-- The theorem for proof
theorem sum_of_abs_roots : absSum = 4 :=
by 
  sorry

end sum_of_abs_roots_l80_80144


namespace sum_of_factors_36_eq_91_l80_80586

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80586


namespace decreasing_interval_l80_80543

noncomputable theory
open Real

def f (x : ℝ) := x + 2 * cos x

theorem decreasing_interval :
  ∀ x ∈ (0, 2 * π), 
    (1 - 2 * sin x < 0) ↔ (x ∈ (π / 6, 5 * π / 6)) :=
by
  sorry

end decreasing_interval_l80_80543


namespace equilateral_triangle_properties_l80_80446

noncomputable def equilateral_triangle_perimeter (a : ℝ) : ℝ :=
3 * a

noncomputable def equilateral_triangle_bisector_length (a : ℝ) : ℝ :=
(a * Real.sqrt 3) / 2

theorem equilateral_triangle_properties (a : ℝ) (h : a = 10) :
  equilateral_triangle_perimeter a = 30 ∧
  equilateral_triangle_bisector_length a = 5 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_properties_l80_80446


namespace ordered_quadruples_count_l80_80319

noncomputable def count_ordered_quadruples : ℕ :=
  {n | ∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
     a^2 + b^2 + c^2 + d^2 = 9 ∧
     (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81 ∧
     n = 15}.card

theorem ordered_quadruples_count : count_ordered_quadruples = 15 := sorry

end ordered_quadruples_count_l80_80319


namespace cafeteria_apples_l80_80525

theorem cafeteria_apples (handed_out: ℕ) (pies: ℕ) (apples_per_pie: ℕ) 
(h1: handed_out = 27) (h2: pies = 5) (h3: apples_per_pie = 4) : handed_out + pies * apples_per_pie = 47 :=
by
  -- The proof will be provided here if needed
  sorry

end cafeteria_apples_l80_80525


namespace sum_of_factors_36_eq_91_l80_80750

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80750


namespace sum_of_factors_of_36_l80_80779

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80779


namespace sum_factors_36_l80_80795

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80795


namespace infinite_series_sum_l80_80227

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80227


namespace sum_factors_36_l80_80641

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80641


namespace series_sum_eq_four_ninths_l80_80247

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80247


namespace series_sum_eq_four_ninths_l80_80246

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80246


namespace remainder_of_5_pow_2023_mod_17_l80_80327

theorem remainder_of_5_pow_2023_mod_17 :
  5^2023 % 17 = 11 :=
by
  have h1 : 5^2 % 17 = 8 := by sorry
  have h2 : 5^4 % 17 = 13 := by sorry
  have h3 : 5^8 % 17 = -1 := by sorry
  have h4 : 5^16 % 17 = 1 := by sorry
  have h5 : 2023 = 16 * 126 + 7 := by sorry
  sorry

end remainder_of_5_pow_2023_mod_17_l80_80327


namespace num_with_square_factors_24_l80_80401

theorem num_with_square_factors_24 : 
  let S := { n | 1 ≤ n ∧ n ≤ 60 };
  let perfect_square_factors := {4, 9, 16, 25, 36, 49};
  (∑ s in S, (∃ f ∈ perfect_square_factors, f ∣ s)) = 24 := 
  by
  let S := { n | 1 ≤ n ∧ n ≤ 60 };
  let perfect_square_factors := {4, 9, 16, 25, 36, 49};
  sorry

end num_with_square_factors_24_l80_80401


namespace quadratic_eq_roots_quadratic_eq_positive_integer_roots_l80_80972

theorem quadratic_eq_roots (m : ℝ) (hm : m ≠ 0 ∧ m ≤ 9 / 8) :
  ∃ x1 x2 : ℝ, (m * x1^2 + 3 * x1 + 2 = 0) ∧ (m * x2^2 + 3 * x2 + 2 = 0) :=
sorry

theorem quadratic_eq_positive_integer_roots (m : ℕ) (hm : m = 1) :
  ∃ x1 x2 : ℝ, (x1 = -1) ∧ (x2 = -2) ∧ (m * x1^2 + 3 * x1 + 2 = 0) ∧ (m * x2^2 + 3 * x2 + 2 = 0) :=
sorry

end quadratic_eq_roots_quadratic_eq_positive_integer_roots_l80_80972


namespace sum_of_factors_36_l80_80627

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80627


namespace sum_factors_36_l80_80769

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80769


namespace red_points_diameter_l80_80062

-- Define the problem with given conditions.
theorem red_points_diameter :
  ∃ (A B: ℝ), 
  (A > 0) ∧ (B > 0) ∧ -- Ensure we are working with positive points
  ∀ (points : List ℝ), points.length = 2019 →  -- exactly 2019 points
  (∀ (arc_length : ℝ), arc_length ∈ {1, 2, 3} ∧ -- arc lengths are 1, 2, or 3
   (points.filter (λ p, p = arc_length)).length = 673) → -- 673 arcs of each length
  -- conclusion we want to prove
  (∃ (P Q : ℝ), (P ∈ points ∧ Q ∈ points) ∧ (P ≠ Q) ∧ is_diameter P Q) := 
by
  sorry

end red_points_diameter_l80_80062


namespace segment_covering_l80_80498

theorem segment_covering (segments : list (ℝ × ℝ))
  (h_cover : ∃ cover : list (ℝ × ℝ), (∀ x ∈ cover, 0 ≤ x.1 ∧ x.2 ≤ 1) ∧ (∀ y ∈ cover, y ∈ segments) ∧ (∃ (l r : ℝ), l = 0 ∧ r = 1 ∧ ∀ x ∈ cover, l ≤ x.1 ∧ x.2 ≤ r)) :
  ∃ selected : list (ℝ × ℝ), (∀ s ∈ selected, s ∈ segments) ∧ (∀ (l r : ℝ), l = 0 ∧ r = 1 ∧ ∃ c ∈ selected, l ≤ c.1 ∧ c.2 ≤ r) ∧ (∑ s in selected, s.2 - s.1 ≤ 2) :=
  sorry

end segment_covering_l80_80498


namespace fraction_product_l80_80129

theorem fraction_product : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l80_80129


namespace trajectory_is_parabola_l80_80519

theorem trajectory_is_parabola (C : ℝ × ℝ) (M : ℝ × ℝ) (l : ℝ → ℝ)
  (hM : M = (0, 3)) (hl : ∀ y, l y = -3)
  (h : dist C M = |C.2 + 3|) : C.1^2 = 12 * C.2 := by
  sorry

end trajectory_is_parabola_l80_80519


namespace arrangement_problem_l80_80090

-- Definitions for the problem setup
def numPeople : ℕ := 5
def totalArrangements : ℕ := factorial numPeople

-- Theorem stating the proof problem
theorem arrangement_problem : 
  let validArrangements := totalArrangements / 2 in
  validArrangements = 60 := 
by
  sorry

end arrangement_problem_l80_80090


namespace sum_of_factors_36_l80_80707

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80707


namespace M_subset_N_cond_l80_80351

theorem M_subset_N_cond (a : ℝ) (h : 0 < a) :
  (∀ p : ℝ × ℝ, p ∈ {p : ℝ × ℝ | p.fst^2 + p.snd^2 = a^2} → p ∈ {p : ℝ × ℝ | |p.fst + p.snd| + |p.fst - p.snd| ≤ 2}) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end M_subset_N_cond_l80_80351


namespace sum_factors_36_l80_80636

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80636


namespace sum_k_over_4k_l80_80161

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80161


namespace sum_factors_36_l80_80643

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80643


namespace number_of_channels_in_first_package_l80_80467

-- Definitions based on the given conditions
constant first_package_cost : ℕ
constant second_100_channels_cost : ℕ
constant james_payment : ℕ
constant total_cost : ℕ

axiom first_package_cost_def : first_package_cost = 100
axiom second_100_channels_cost_def : second_100_channels_cost = 50
axiom james_payment_def : james_payment = 75
axiom total_cost_def : total_cost = 150

-- Theorem to prove
theorem number_of_channels_in_first_package :
  ∃ n, total_cost = first_package_cost + second_100_channels_cost ∧ n = 100 :=
by
  rw [first_package_cost_def, second_100_channels_cost_def, <-total_cost_def]
  sorry

end number_of_channels_in_first_package_l80_80467


namespace a_pow_neg_one_eq_one_l80_80417

theorem a_pow_neg_one_eq_one : ∀ (a : ℝ), a⁻¹ = (-1 : ℝ)^0 → a = 1 := 
begin
  intros a h,
  have h1 : (-1 : ℝ)^0 = 1 := by norm_num,
  rw h1 at h,
  rw inv_eq_one_iff at h,
  exact h,
end

end a_pow_neg_one_eq_one_l80_80417


namespace sum_of_factors_36_eq_91_l80_80743

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80743


namespace sum_to_product_l80_80297

theorem sum_to_product (c d : ℝ) : 
  cos (c + d) + cos (c - d) = 2 * cos c * cos d :=
by
  sorry

end sum_to_product_l80_80297


namespace part1_part2_part3_l80_80843

-- Define the conditions
variable (cost_per_item : ℤ := 8)
variable (x : ℤ) (h_x : 8 ≤ x ∧ x ≤ 15)
variable (y : ℤ := -5 * x + 150)

-- Statements to prove
theorem part1 : ∀ x, 8 ≤ x ∧ x ≤ 15 → y = -5 * x + 150 := 
  begin
    intros x hx,
    exact rfl
  end

theorem part2 : ∃ x, (x - 8) * (-5 * x + 150) = 480 ∧ 8 ≤ x ∧ x ≤ 15 → x = 14 :=
  sorry

theorem part3 : ∃ x, x = 15 ∧ (x - 8) * (-5 * x + 150) = 525 :=
  sorry

end part1_part2_part3_l80_80843


namespace sum_k_over_4k_l80_80158

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80158


namespace sum_factors_36_eq_91_l80_80599

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80599


namespace minimum_value_on_interval_l80_80533

noncomputable def f (x : ℝ) : ℝ := (1/4) * x^4 + (1/3) * x^3 + (1/2) * x^2

theorem minimum_value_on_interval : 
  ∃ x ∈ set.Icc (-1 : ℝ) 1, ∀ y ∈ set.Icc (-1 : ℝ) 1, f x ≤ f y ∧ f x = 0 :=
by
  sorry

end minimum_value_on_interval_l80_80533


namespace sum_infinite_series_l80_80193

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80193


namespace shaded_region_area_correct_l80_80079

noncomputable def hexagon_side : ℝ := 4
noncomputable def major_axis : ℝ := 4
noncomputable def minor_axis : ℝ := 2

noncomputable def hexagon_area := (3 * Real.sqrt 3 / 2) * hexagon_side^2

noncomputable def semi_ellipse_area : ℝ :=
  (1 / 2) * Real.pi * major_axis * minor_axis

noncomputable def total_semi_ellipse_area := 4 * semi_ellipse_area 

noncomputable def shaded_region_area := hexagon_area - total_semi_ellipse_area

theorem shaded_region_area_correct : shaded_region_area = 48 * Real.sqrt 3 - 16 * Real.pi :=
by
  sorry

end shaded_region_area_correct_l80_80079


namespace validate_f_l80_80551

-- Noncomputable definition of the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1.5 then 0
  else if -1.5 < x ∧ x ≤ -0.5 then 0.5 * x + 0.75
  else if -0.5 < x ∧ x ≤ 0.5 then 0.5
  else if 0.5 < x ∧ x ≤ 1.5 then -0.5 * x + 0.75
  else 0

-- Main statement: Validate the analytical expression for f(x) over the entire real number line
theorem validate_f :
  (∀ x, f x = 0 ∨ f x = 0.5 * x + 0.75 ∨ f x = 0.5 ∨ f x = -0.5 * x + 0.75) ∧
  (∫ x in -∞..∞, f x = 1) :=
  sorry

end validate_f_l80_80551


namespace sum_of_factors_36_eq_91_l80_80596

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80596


namespace infinite_series_sum_l80_80263

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80263


namespace sum_squares_of_roots_of_polynomial_l80_80907

noncomputable def roots (n : ℕ) (p : Polynomial ℂ) : List ℂ :=
  if h : n = p.natDegree then Multiset.toList p.roots else []

theorem sum_squares_of_roots_of_polynomial :
  (roots 2018 (Polynomial.C 404 + Polynomial.C 3 * X ^ 3 + Polynomial.C 44 * X ^ 2015 + X ^ 2018)).sum = 0 :=
by
  sorry

end sum_squares_of_roots_of_polynomial_l80_80907


namespace sum_of_factors_36_eq_91_l80_80593

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80593


namespace sum_of_factors_36_l80_80693

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80693


namespace sum_of_factors_36_l80_80683

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80683


namespace sum_of_factors_36_l80_80726

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80726


namespace mortgage_loan_amount_l80_80410

/-- Given the initial payment is 1,800,000 rubles and it represents 30% of the property cost C, 
    prove that the mortgage loan amount is 4,200,000 rubles. -/
theorem mortgage_loan_amount (C : ℝ) (h : 0.3 * C = 1800000) : C - 1800000 = 4200000 :=
by
  sorry

end mortgage_loan_amount_l80_80410


namespace sum_of_factors_36_eq_91_l80_80594

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80594


namespace sum_of_factors_of_36_l80_80666

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80666


namespace sum_series_equals_4_div_9_l80_80221

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80221


namespace ab_equiv_l80_80344

theorem ab_equiv (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 7) : a / b = 10 / 7 :=
by
  sorry

end ab_equiv_l80_80344


namespace length_of_DE_l80_80457

theorem length_of_DE
  (AB AD : ℝ) (D E C : ℝ × ℝ)
  (h_AB : AB = 8)
  (h_AD : AD = 7)
  (h_DC : D.1 = C.1 ∧ abs (D.2 - C.2) = 8)
  (area_equal : 1/2 * abs (C.1 - E.1) * abs (C.2 - E.2) = AB * AD) :
  ∃ DE : ℝ, DE = 2 * real.sqrt 65 :=
by
  sorry

end length_of_DE_l80_80457


namespace discount_rate_at_1000_actual_discount_rate_condition_l80_80083

-- Define the discount rate function y for x in (0, 1000]
def discount_rate (x : ℝ) : ℝ :=
  if x < 625 then 0.8
  else if 625 ≤ x ∧ x ≤ 1000 then (0.8 * x - 100) / x
  else 0 -- undefined otherwise, for safety

-- Define the property that the actual discount rate is as expected for x in (0, 1000]
theorem discount_rate_at_1000 :
  discount_rate 1000 = 0.7 :=
by
  -- Proof to be provided
  sorry

-- Define another property for x in [2500, 3500]
def discount_condition (x : ℝ) : Prop :=
  (2500 ≤ x ∧ x < 3000 → (0.8 * x - 400) / x < 2 / 3) ∧
  (3125 ≤ x ∧ x ≤ 3500 → (0.8 * x - 500) / x < 2 / 3)

-- Theorem for discount condition within the given intervals
theorem actual_discount_rate_condition (x : ℝ) (h1: 2500 ≤ x) (h2: x ≤ 3500) :
  (discount_condition x) :=
by
  -- Proof to be provided
  sorry

end discount_rate_at_1000_actual_discount_rate_condition_l80_80083


namespace parabola_properties_l80_80930

-- Define the conditions
def vertex (f : ℝ → ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ (x : ℝ), f (v.1) ≤ f x

def vertical_axis_of_symmetry (f : ℝ → ℝ) (h : ℝ) : Prop :=
  ∀ (x : ℝ), f x = f (2 * h - x)

def contains_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

-- Define f as the given parabola equation
def f (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 16

-- The main statement to prove
theorem parabola_properties :
  vertex f (3, -2) ∧ vertical_axis_of_symmetry f 3 ∧ contains_point f (6, 16) := sorry

end parabola_properties_l80_80930


namespace first_player_wins_l80_80048

def highest_power_of_two (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else 
    let rec find_power (k acc : ℕ) : ℕ :=
      if k % 2 = 1 then acc
      else find_power (k / 2) (acc + 1)
    find_power n 0

theorem first_player_wins (M N : ℕ) : 
  (highest_power_of_two M ≠ highest_power_of_two N) ↔ 
  (exists strategy : ℕ × ℕ → ℕ × ℕ, 
    let winning_position := ∀ (M N : ℕ), strategy (M, N) ≠ (1, 1) in 
    winning_position M N) :=
sorry

end first_player_wins_l80_80048


namespace sum_of_factors_36_l80_80684

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80684


namespace sum_factors_36_l80_80791

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80791


namespace find_sin_value_l80_80367

theorem find_sin_value (α : ℝ) (h1 : -π < α) (h2 : α < -π / 2) (h3 : cos (5 * π / 12 + α) = 1 / 3) : 
  sin (2 * α + 5 * π / 6) = -4 * real.sqrt 2 / 9 := 
sorry

end find_sin_value_l80_80367


namespace solve_for_t_l80_80925

theorem solve_for_t : ∃ t : ℚ, (8 * t ^ 3 + 17 * t ^ 2 + 2 * t - 3 = 0) ∧ t = 1 / 2 :=
begin
  use 1 / 2,  -- Provide the candidate solution as t = 1/2
  split,
  { -- Show that this t satisfies the polynomial equation
    norm_num,
  },
  { -- Establish t = 1/2
    refl,
  }
end

end solve_for_t_l80_80925


namespace lines_and_planes_l80_80103

variables {Plane Line : Type}
variables (m n : Line) (α : Plane)

-- Conditions
def is_within_plane (m : Line) (α : Plane) := sorry
def parallel_lines (m n : Line) := sorry
def line_outside_plane (n : Line) (α : Plane) := sorry 
def parallel_line_and_plane (n : Line) (α : Plane) := sorry

-- Hypotheses / Assumptions
axiom m_within_alpha : is_within_plane m α
axiom m_parallel_n : parallel_lines m n
axiom n_outside_alpha : line_outside_plane n α

-- The proof goal
theorem lines_and_planes (m n : Line) (α : Plane) 
  (h1 : is_within_plane m α) 
  (h2 : parallel_lines m n)
  (h3 : line_outside_plane n α) : 
  parallel_line_and_plane n α := 
sorry

end lines_and_planes_l80_80103


namespace fractional_product_l80_80124

theorem fractional_product :
  ((3/4) * (4/5) * (5/6) * (6/7) * (7/8)) = 3/8 :=
by
  sorry

end fractional_product_l80_80124


namespace domain_of_function1_domain_of_function2_l80_80308

noncomputable def domain1 := { x : ℝ | x > -1 ∧ x ≠ 0 }
noncomputable def domain2 := { x : ℝ | (2 : ℝ) / 3 < x ∧ x ≤ 1 }

theorem domain_of_function1 :
  ∀ x : ℝ, (x + 1 > 0 ∧ (4 - x)^(1/3) / (x + 1)^(1/2) - 1 ≠ 0) ↔ (x > -1 ∧ x ≠ 0) :=
sorry

theorem domain_of_function2 :
  ∀ x : ℝ, (log (1 / 2) (3 * x - 2) ≥ 0) ↔ ((2 : ℝ) / 3 < x ∧ x ≤ 1) :=
sorry

end domain_of_function1_domain_of_function2_l80_80308


namespace categorize_numbers_l80_80299

def given_numbers : List ℚ := [12, 0, -4, 3.14, |-\frac{2}{3}|, 0.618, -3.5, 2.71, 6/100, 0.3]

theorem categorize_numbers :
  let positive_numbers := [12, 3.14, |-\frac{2}{3}|, 0.618, 2.71, 6/100, 0.3];
  let negative_fractions := [-3.5];
  let nonneg_integers := [12, 0];
  (∀ n ∈ given_numbers, n ∈ positive_numbers → n > 0) ∧ 
  (∀ n ∈ given_numbers, n ∈ negative_fractions → n < 0) ∧
  (∀ n ∈ given_numbers, n ∈ nonneg_integers → n ≥ 0 ∧ n.den = 1) := by
  sorry

end categorize_numbers_l80_80299


namespace sum_of_factors_of_36_l80_80784

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80784


namespace expected_absolute_difference_after_10_days_l80_80029

def cat_fox_wealth_difference : ℝ := 1

theorem expected_absolute_difference_after_10_days :
  let p_cat_wins := 0.25
  let p_fox_wins := 0.25
  let p_both_police := 0.5
  let num_days := 10
  ∃ (X : ℕ → ℕ), 
    (X 0 = 0) ∧
    ∀ n, (X (n + 1) = (if (X n = 0) then 0.5 else 0) * X n) →
    (∑ k in range (num_days + 1), (k : ℝ) * (0.5 ^ k)) = cat_fox_wealth_difference := 
sorry

end expected_absolute_difference_after_10_days_l80_80029


namespace sum_factors_36_l80_80789

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80789


namespace sum_of_series_l80_80168

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80168


namespace sum_of_factors_of_36_l80_80782

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80782


namespace sum_series_equals_4_div_9_l80_80213

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80213


namespace num_ways_to_distribute_items_l80_80116

theorem num_ways_to_distribute_items : 
  let items := 5
  let bags := 4
  let distinct_items := 5
  let identical_bags := 4
  (number_of_ways_to_distribute_items_in_4_identical_bags distinct_items identical_bags = 36) := sorry

end num_ways_to_distribute_items_l80_80116


namespace odd_factors_of_450_l80_80405

theorem odd_factors_of_450 : 
  let factors_count (n : ℕ) := (n.factors.count (λ d, d % 2 = 1))
  factors_count 450 = 9 :=
by
  sorry

end odd_factors_of_450_l80_80405


namespace collinear_BCAD_l80_80473

variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

theorem collinear_BCAD (h : ∀ P : Type, dist P A + dist P D ≥ dist P B + dist P C) : 
  collinear ℝ ({B, C, A, D} : set ℝ) :=
sorry

end collinear_BCAD_l80_80473


namespace carolyn_spends_36_dollars_on_lace_l80_80904

-- Definitions of the conditions
def length_cuff : ℕ := 50 -- 50 cm for each cuff
def length_hem : ℕ := 300 -- 300 cm for the hem
def waist_fraction : ℝ := 1 / 3 -- fraction for waist length
def length_ruffle : ℕ := 20 -- 20 cm for each ruffle

def num_cuffs : ℕ := 2
def num_ruffles : ℕ := 5

def cost_per_meter : ℝ := 6 -- $6 per meter

-- Derived total lace in cm
def total_lace_cm : ℕ :=
  let waist_length := (length_hem : ℝ) * waist_fraction
  (num_cuffs * length_cuff) + length_hem + (waist_length.toNat) + (num_ruffles * length_ruffle)

-- Derived total lace in meters
def total_lace_m : ℝ := (total_lace_cm : ℝ) / 100

-- Calculate the total cost
theorem carolyn_spends_36_dollars_on_lace : 
  (total_lace_m * cost_per_meter) = 36 := by
  sorry

end carolyn_spends_36_dollars_on_lace_l80_80904


namespace sum_of_factors_36_l80_80740

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80740


namespace sum_of_factors_of_36_l80_80787

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80787


namespace graded_percentage_on_Monday_l80_80492

theorem graded_percentage_on_Monday (P : ℝ) 
  (h1 : ∀ (P : ℝ), 
    (120 - P / 100 * 120 - 
    0.75 * (1 - P / 100) * 120 = 12)) : 
  P = 40 :=
by
  intro P
  specialize h1 P
  exact sorry

end graded_percentage_on_Monday_l80_80492


namespace cos_minus_sin_eq_neg_one_fifth_l80_80368

theorem cos_minus_sin_eq_neg_one_fifth
  (α : ℝ)
  (h1 : Real.sin (2 * α) = 24 / 25)
  (h2 : π < α ∧ α < 5 * π / 4) :
  Real.cos α - Real.sin α = -1 / 5 := sorry

end cos_minus_sin_eq_neg_one_fifth_l80_80368


namespace groceries_value_l80_80092

-- Conditions
def alex_saved : ℝ := 14500
def car_cost : ℝ := 14600
def trip_charge : ℝ := 1.5
def earn_percentage : ℝ := 0.05
def num_trips : ℝ := 40

-- Proof Statement
theorem groceries_value
  (alex_saved : ℝ)
  (car_cost : ℝ)
  (trip_charge : ℝ)
  (earn_percentage : ℝ)
  (num_trips : ℝ)
  (h_saved : alex_saved = 14500)
  (h_car_cost : car_cost = 14600)
  (h_trip_charge : trip_charge = 1.5)
  (h_earn_percentage : earn_percentage = 0.05)
  (h_num_trips : num_trips = 40) :

  let needed_savings := car_cost - alex_saved in
  let earnings_from_trips := num_trips * trip_charge in
  let earnings_from_groceries := needed_savings - earnings_from_trips in
  let total_value_of_groceries := earnings_from_groceries / earn_percentage in
  total_value_of_groceries = 800 := by {
    sorry
  }

end groceries_value_l80_80092


namespace proposition2_l80_80369

variables {a b : Type}  -- Types for lines
variables {γ : Type}    -- Type for plane

-- Indicating that a and b are lines, γ is a plane
variables [is_line a] [is_line b] [is_plane γ]

-- Definitions for perpendicular and parallel relationships
variable (perpendicular : ∀ {l : Type} {p : Type}, is_line l → is_plane p → Prop)
variable (parallel : ∀ {l1 : Type} {l2 : Type}, is_line l1 → is_line l2 → Prop)

theorem proposition2 (ha : is_line a) (hb : is_line b) (hγ : is_plane γ)
  (h1 : perpendicular a γ) (h2 : perpendicular b γ) : parallel a b :=
sorry

end proposition2_l80_80369


namespace sum_factors_36_l80_80771

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80771


namespace sum_factors_36_l80_80644

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80644


namespace triangle_sum_correct_l80_80520

def triangle_op (a b c : ℕ) : ℕ :=
  a * b / c

theorem triangle_sum_correct :
  triangle_op 4 8 2 + triangle_op 5 10 5 = 26 :=
by
  sorry

end triangle_sum_correct_l80_80520


namespace sum_of_factors_36_eq_91_l80_80756

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80756


namespace noncongruent_triangles_l80_80504

noncomputable def isosceles_triangle (A B C : Type*) [MetricSpace A] 
  [MetricSpace B] [MetricSpace C] := 
  (dist A B = dist A C) ∧ (dist B C ≠ dist A B)

noncomputable def midpoint (A B : α) : α := sorry
noncomputable def perpendicular_foot (A B C : α) : α := sorry

noncomputable def point_configuration (A B C D E F : Type*) :=
  isosceles_triangle A B C ∧
  (D = midpoint A B) ∧
  (E = midpoint A C) ∧
  (F = perpendicular_foot A B C)

theorem noncongruent_triangles (A B C D E F : Type*) 
  [MetricSpace A] [MetricSpace B] 
  [MetricSpace C] [MetricSpace D]
  [MetricSpace E] [MetricSpace F] 
  (h : point_configuration A B C D E F) :
  ∃ n, n = 4 :=
by
  sorry

end noncongruent_triangles_l80_80504


namespace sum_factors_36_l80_80790

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80790


namespace two_cos_is_even_with_period_l80_80535

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

theorem two_cos_is_even_with_period :
  is_even_function (λ x : ℝ, 2 * Real.cos x) ∧ has_period (λ x : ℝ, 2 * Real.cos x) (2 * Real.pi) :=
by
  sorry

end two_cos_is_even_with_period_l80_80535


namespace find_a_l80_80432

def f (a : ℝ) (x : ℝ) := 1 / (3^x + 1) + a

theorem find_a (a : ℝ) 
  (h_odd : ∀ x : ℝ, f a x = - f a (-x)) : 
  a = - 1 / 2 :=
by
  sorry

end find_a_l80_80432


namespace sum_of_factors_36_l80_80625

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80625


namespace sum_series_l80_80259

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80259


namespace elderly_teachers_in_sample_l80_80547

-- Definitions based on the conditions
def numYoungTeachersSampled : ℕ := 320
def ratioYoungToElderly : ℚ := 16 / 9

-- The theorem that needs to be proved
theorem elderly_teachers_in_sample :
  ∃ numElderlyTeachersSampled : ℕ, 
    numYoungTeachersSampled * (9 / 16) = numElderlyTeachersSampled := 
by
  use 180
  sorry

end elderly_teachers_in_sample_l80_80547


namespace no_real_root_of_sqrt_eq_l80_80916

theorem no_real_root_of_sqrt_eq (x : ℝ) :
  sqrt (x + 9) - sqrt (x - 2) + 2 = 0 → False :=
by sorry

end no_real_root_of_sqrt_eq_l80_80916


namespace sum_factors_36_l80_80757

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80757


namespace greatest_possible_sum_l80_80875

noncomputable def eight_products_sum_max : ℕ :=
  let a := 3
  let b := 4
  let c := 5
  let d := 8
  let e := 6
  let f := 7
  7 * (c + d) * (e + f)

theorem greatest_possible_sum (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) :
  eight_products_sum_max = 1183 :=
by
  sorry

end greatest_possible_sum_l80_80875


namespace cartesian_eq_polar_eq_distance_AB_main_theorem_l80_80389

noncomputable def param_line (t : ℝ) : ℝ × ℝ :=
(2 + t, 2 * t)

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ :=
(rho * Math.cos theta, rho * Math.sin theta)

theorem cartesian_eq_polar_eq (x y theta : ℝ) (h : y = ρ * Math.sin θ ∧ x = ρ * Math.cos θ)
: y^2 = 8 * x ↔ ρ = 8 * Math.cos θ / (Math.sin θ)^2 :=
by sorry

theorem distance_AB (t1 t2 : ℝ) (t1t2_solution : t1 + t2 = 2 ∧ t1 * t2 = -4) : 
|t1 - t2| = 2 * sqrt 5 :=
begin
  sorry
end

theorem main_theorem (t : ℝ) (x y : ℝ)
  (line_eq : (x, y) = param_line t)
  (polar_eq : y^2 = 8 * x) 
  (t1t2_solution : (∑ (t1 t2 : ℝ), t1 + t2 = 2 ∧ t1 * t2 = -4) :
 ∃ t1 t2 : ℝ, t1 - t2 = 2*sqrt(5) :=
begin 
  sorry
end

end cartesian_eq_polar_eq_distance_AB_main_theorem_l80_80389


namespace inhabitants_in_Hut_l80_80118

theorem inhabitants_in_Hut : 
  (∃ C O M : ℕ, 
    O + M = 2 ∧ 
    C + M = 2 ∧ 
    C + O = 2 ∧
    C + O + M = 3) :=
begin
  sorry
end

end inhabitants_in_Hut_l80_80118


namespace range_of_m_cond_l80_80428

noncomputable def quadratic_inequality (x m : ℝ) : Prop :=
  x^2 + m * x + 2 * m - 3 ≥ 0

theorem range_of_m_cond (m : ℝ) (h1 : 2 ≤ m) (h2 : m ≤ 6) (x : ℝ) :
  quadratic_inequality x m :=
sorry

end range_of_m_cond_l80_80428


namespace area_of_trapezoid_l80_80929

def trapezoid_area (a b h : ℕ) : ℕ := (a + b) * h / 2

theorem area_of_trapezoid :
  let AD := 44
  let BC := 16
  let AB := 17
  let CD := 25
  let S_△CKD := 210
  let KD := AD - BC
  let CM := 2 * S_△CKD / KD
  trapezoid_area AD BC CM = 450 :=
by
  let AD := 44
  let BC := 16
  let AB := 17
  let CD := 25
  let S_△CKD := 210
  let KD := AD - BC
  let CM := 2 * S_△CKD / KD
  have h1 : KD = 28 :=
    by linarith
  have h2 : CM = 15 :=
    by
      calc
        CM = 2 * 210 / 28 := by linarith
  have h3 : trapezoid_area AD BC CM = 450 :=
    by
      calc
        trapezoid_area AD BC CM = (AD + BC) * CM / 2 := by rfl
                                     ... = (44 + 16) * 15 / 2 := by rw [AD, BC, CM]
                                     ... = 450 := by norm_num
  exact h3

end area_of_trapezoid_l80_80929


namespace problem1_div_expr_problem2_div_expr_l80_80091

-- Problem 1
theorem problem1_div_expr : (1 / 30) / ((2 / 3) - (1 / 10) + (1 / 6) - (2 / 5)) = 1 / 10 :=
by 
  -- sorry is added to mark the spot for the proof
  sorry

-- Problem 2
theorem problem2_div_expr : (-1 / 20) / (-(1 / 4) - (2 / 5) + (9 / 10) - (3 / 2)) = 1 / 25 :=
by 
  -- sorry is added to mark the spot for the proof
  sorry

end problem1_div_expr_problem2_div_expr_l80_80091


namespace cos_angle_is_correct_k_values_for_perpendicular_l80_80961

def vector_a : ℝ × ℝ × ℝ := (3, 2, -1)
def vector_b : ℝ × ℝ × ℝ := (2, 1, 2)

def cos_angle_between (a b : ℝ × ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) / ((Real.sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)) * (Real.sqrt (b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2)))

theorem cos_angle_is_correct :
  cos_angle_between vector_a vector_b = 2 / Real.sqrt 14 :=
by
  sorry

theorem k_values_for_perpendicular :
  ∃ k : ℝ, (k = 3 / 2 ∨ k = -2 / 3) ∧ (let ka_plus_b := (k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2, k * vector_a.3 + vector_b.3) in
  let a_minus_kb := (vector_a.1 - k * vector_b.1, vector_a.2 - k * vector_b.2, vector_a.3 - k * vector_b.3) in
  (ka_plus_b.1 * a_minus_kb.1 + ka_plus_b.2 * a_minus_kb.2 + ka_plus_b.3 * a_minus_kb.3 = 0)) :=
by
  sorry

end cos_angle_is_correct_k_values_for_perpendicular_l80_80961


namespace sum_of_factors_36_l80_80694

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80694


namespace quadratic_inequality_solution_l80_80557

theorem quadratic_inequality_solution 
  (a b : ℝ) 
  (h1 : (∀ x : ℝ, x^2 + a * x + b > 0 → (x < 3 ∨ x > 1))) :
  ∀ x : ℝ, a * x + b < 0 → x > 3 / 4 := 
by 
  sorry

end quadratic_inequality_solution_l80_80557


namespace shortest_distance_between_circles_l80_80917

-- Define circles using their equations
def circle1 (x y : ℝ) : Prop := x^2 - 8 * x + y^2 + 6 * y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + 16 * x + y^2 - 2 * y + 34 = 0

-- Prove the shortest distance between the circles
theorem shortest_distance_between_circles :
  (∃ (d : ℝ), d = 4*ℝ.sqrt 10 - (2*ℝ.sqrt 6 + ℝ.sqrt 31)) :=
sorry

end shortest_distance_between_circles_l80_80917


namespace sum_of_factors_36_l80_80621

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80621


namespace system_sol_l80_80969

theorem system_sol {x y : ℝ} (h1 : x + 2 * y = -1) (h2 : 2 * x + y = 3) : x - y = 4 := by
  sorry

end system_sol_l80_80969


namespace evaluate_series_sum_l80_80280

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80280


namespace remainder_of_power_mod_l80_80328

theorem remainder_of_power_mod :
  (5^2023) % 17 = 15 :=
begin
  sorry
end

end remainder_of_power_mod_l80_80328


namespace tangent_spheres_common_tangent_plane_l80_80822

theorem tangent_spheres_common_tangent_plane (r₁ r₂ r₃ : ℝ) 
  (h₁ : 0 ≤ r₁) (h₂ : 0 ≤ r₂) (h₃ : 0 ≤ r₃)
  (h₁₂ : r₁ ≥ r₂) (h₂₃ : r₂ ≥ r₃)
  (tangent_condition : ∀ (s₁ s₂ s₃ : Sphere), 
    are_pairwise_tangent s₁ s₂ s₃ ↔ 
    (∃ (r₁ r₂ r₃ : ℝ), 
    s₁.radius = r₁ ∧ 
    s₂.radius = r₂ ∧ 
    s₃.radius = r₃ ∧ 
    r₁ ≥ r₂ ∧ 
    r₂ ≥ r₃)) : 
  r₃ ≥ (r₁ * r₂) / ((Real.sqrt r₁ + Real.sqrt r₂) ^ 2) := 
by
  sorry

end tangent_spheres_common_tangent_plane_l80_80822


namespace sum_of_factors_36_eq_91_l80_80581

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80581


namespace sum_of_factors_of_36_l80_80663

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80663


namespace problem1_problem2_l80_80387

-- Define the conditions for the parabola
def parabola (y x p : ℝ) := y^2 = 2 * p * x
variable (p : ℝ) (hp : p > 0)
def focus (p : ℝ) := (p / 2, 0)
variable {P : ℝ × ℝ} (hP : P = (1, m))
variable (m : ℝ)
def on_parabola : Prop := parabola P.2 P.1 p
def distance_to_focus_eq_2 : Prop := Real.sqrt ((P.1 - p/2)^2 + (P.2)^2) = 2

-- Define the ellipse sharing the same focus as the parabola
def ellipse (x y : ℝ) := x^2 / 4 + y^2 / 3 = 1
def ellipse_focus_eq_parabola_focus : Prop := focus p = focus 1

-- Problem 1: Prove the equation of the ellipse
theorem problem1 (h : ellipse_focus_eq_parabola_focus) : ellipse P.1 P.2 := sorry

-- Define the hyperbola with asymptotes along lines OA and OB and passing through P
def hyperbola (x y : ℝ) := 3 * x^2 - y^2 / 2 = 1
def intersections (A B : ℝ × ℝ) : Prop := 
  ellipse A.1 A.2 ∧ parabola A.2 A.1 p ∧ ellipse B.1 B.2 ∧ parabola B.2 B.1 p
def asymptotes (A B : ℝ × ℝ) (x y : ℝ) := y = (A.2 / A.1) * x ∨ y = (B.2 / B.1) * x

-- Problem 2: Prove the equation of the hyperbola
theorem problem2 {A B : ℝ × ℝ} (hAB : intersections A B) (hP : on_parabola) 
  (hPF : Real.sqrt ((P.1 - p/2)^2 + (P.2)^2) = 2) : 
  hyperbola P.1 P.2 := sorry

end problem1_problem2_l80_80387


namespace sum_of_factors_36_eq_91_l80_80741

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80741


namespace team_count_l80_80542

theorem team_count (girls boys : ℕ) (g : girls = 4) (b : boys = 6) :
  (Nat.choose 4 2) * (Nat.choose 6 2) = 90 := by
  sorry

end team_count_l80_80542


namespace solve_inequality_l80_80516

theorem solve_inequality (x : ℝ) : 
  (-9 * x^2 + 6 * x + 15 > 0) ↔ (x > -1 ∧ x < 5/3) := 
sorry

end solve_inequality_l80_80516


namespace verify_propositions_correct_l80_80999

noncomputable def propositions_correct : Prop :=
  ∀ (a b c : Type) 
    (α β γ : Type) 
    [Parallelism a b] [Parallelism b c] [Intersection α γ b] [Intersection β γ a],
    (parallelism_trans : a ∥ b → b ∥ c → a ∥ c) →
    (line_plane_parallelism : α ∥ β → α ∩ γ = b → β ∩ γ = a → a ∥ b) →
    (a ∥ b ∧ b ∥ c → a ∥ c) ∧
    (α ∥ β ∧ α ∩ γ = b ∧ β ∩ γ = a → a ∥ b) ∧
    ¬(a ⟂ b ∧ b ⟂ c → a ⟂ c) ∧
    ¬(a ⟂ α ∧ α ⟂ β → a ∥ β)

theorem verify_propositions_correct :
  propositions_correct :=
  sorry

end verify_propositions_correct_l80_80999


namespace distance_ran_l80_80842

theorem distance_ran (time_minutes : ℝ) (speed_kmh : ℝ) (time_hours : ℝ) : 
  time_minutes = 45 → speed_kmh = 2 → time_hours = time_minutes / 60 → 
  (speed_kmh * time_hours) = 1.5 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end distance_ran_l80_80842


namespace sum_of_factors_36_l80_80723

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80723


namespace trajectory_of_M_l80_80358

-- Definitions and conditions
variables {P A B C M : Type}
variable [h_cone : IsRightCircularCone P A B C]
variable [h_triangle : IsEquilateralTriangle A B C]
variable [h_point_inside : IsPointInsideOrOnBoundary M A B C]
variable {d a : ℝ}

-- Arithmetic sequence of distances
variable [h_distances_seq : DistancesFormArithmeticSequence M P A B C d a]
-- Constant volume condition
variable [h_volume : VolumeCondition P A B C d]

-- Theorem statement
theorem trajectory_of_M (P A B C M : Type) [h_cone : IsRightCircularCone P A B C]
  [h_triangle : IsEquilateralTriangle A B C]
  [h_point_inside : IsPointInsideOrOnBoundary M A B C]
  {d a : ℝ}
  [h_distances_seq : DistancesFormArithmeticSequence M P A B C d a]
  [h_volume : VolumeCondition P A B C d] : 
  IsLineSegment (TrajectoryOfPoint M) := 
sorry

end trajectory_of_M_l80_80358


namespace mike_picked_peaches_l80_80490

theorem mike_picked_peaches (initial_peaches total_peaches picked_peaches : ℕ) 
  (h1 : initial_peaches = 34) 
  (h2 : total_peaches = 86) 
  : picked_peaches = total_peaches - initial_peaches :=
by
  have h : picked_peaches = 52,
  sorry

end mike_picked_peaches_l80_80490


namespace proper_subsets_number_l80_80390

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def Z : Set ℤ := {z | True}  -- This denotes the set of all integers

theorem proper_subsets_number : 
  let A_Z := (λ x, ∃ (n : ℤ), (x:ℝ) = n ∧ n ∈ A) in
  (Finset.card (Finset.erase_univ (Finset.map (Set.to_finset_embedding A_Z) Finset.univ)) = 3) := by
  sorry

end proper_subsets_number_l80_80390


namespace eval_F_at_4_f_5_l80_80422

def f (a : ℤ) : ℤ := 3 * a - 6
def F (a : ℤ) (b : ℤ) : ℤ := 2 * b ^ 2 + 3 * a

theorem eval_F_at_4_f_5 : F 4 (f 5) = 174 := by
  sorry

end eval_F_at_4_f_5_l80_80422


namespace sum_eq_zero_l80_80052

theorem sum_eq_zero (m n : ℕ) (h : m < n) : 
  ∑ k in Finset.range (n + 1), (-1)^k * k^m * Nat.choose n k = 0 := 
sorry

end sum_eq_zero_l80_80052


namespace smallest_possible_fourth_chair_l80_80954

/--
Four special prizes at a school sports day are hidden beneath each chair numbered with two-digit positive integers. 
The first three numbers found are 45, 26, and 63, but the label on the last chair is partially ripped, leaving the number unreadable. 
The sum of the digits of all four numbers equals one-third of the sum of all four numbers. 
Additionally, the total sum of all numbers must be a multiple of 7. 
Prove that the smallest possible number for the fourth chair is 37.
-/
theorem smallest_possible_fourth_chair : 
  ∃ (x : ℕ), (134 + x) % 7 = 0 ∧ x < 100 ∧ ( (26 + x.digits.sum) * 3 = 134 + x ) ∧ x = 37 := 
by 
  sorry

end smallest_possible_fourth_chair_l80_80954


namespace sum_infinite_series_l80_80196

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80196


namespace Cara_skate_distance_l80_80568

-- Definitions corresponding to the conditions
def distance_CD : ℝ := 150
def speed_Cara : ℝ := 10
def speed_Dan : ℝ := 6
def angle_Cara_CD : ℝ := 45

-- main theorem based on the problem and given conditions
theorem Cara_skate_distance : ∃ t : ℝ, distance_CD = 150 ∧ speed_Cara = 10 ∧ speed_Dan = 6
                            ∧ angle_Cara_CD = 45 
                            ∧ 10 * t = 253.5 :=
by
  sorry

end Cara_skate_distance_l80_80568


namespace find_f_f_3_l80_80967

noncomputable def f : ℝ → ℝ :=
λ x, if x < 3 then 3 * Real.exp(x - 1) else Real.logb 3 (x^2 - 6)

theorem find_f_f_3 : f (f 3) = 3 := by
  sorry

end find_f_f_3_l80_80967


namespace first_digit_base_5_of_2197_l80_80577

theorem first_digit_base_5_of_2197 : 
  ∃ k : ℕ, 2197 = k * 625 + r ∧ k = 3 ∧ r < 625 :=
by
  -- existence of k and r follows from the division algorithm
  -- sorry is used to indicate the part of the proof that needs to be filled in
  sorry

end first_digit_base_5_of_2197_l80_80577


namespace sum_of_factors_36_l80_80620

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80620


namespace range_of_a_l80_80998

theorem range_of_a (a : ℝ) : (¬ (∃ x : ℝ, sin x > a)) → (a ≥ 1) :=
by {
  sorry
}

end range_of_a_l80_80998


namespace sum_factors_36_l80_80772

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80772


namespace probability_all_heads_or_all_tails_in_three_flips_l80_80823

theorem probability_all_heads_or_all_tails_in_three_flips : 
  let p := 1/2 in
  let prob_hhh := p * p * p in
  let prob_ttt := p * p * p in
  prob_hhh + prob_ttt = 1/4 :=
by
  sorry

end probability_all_heads_or_all_tails_in_three_flips_l80_80823


namespace energy_equivalence_l80_80988

def solar_energy_per_sqm := 1.3 * 10^8
def china_land_area := 9.6 * 10^6
def expected_coal_energy := 1.248 * 10^15

theorem energy_equivalence : 
  solar_energy_per_sqm * china_land_area = expected_coal_energy := 
by
  sorry

end energy_equivalence_l80_80988


namespace sum_factors_36_l80_80760

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80760


namespace sum_of_factors_36_l80_80690

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80690


namespace sum_factors_36_eq_91_l80_80600

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80600


namespace triangle_condition_proof_l80_80965

-- Given triangle ABC
variable (A B C : Point)

-- Additional points P, Q, R
variable (P Q R : Point)

-- Angles in degrees
def angle_deg (x y z : Point) : Float := sorry

-- Conditions
def cond1 : angle_deg P B C = 45 := sorry
def cond2 : angle_deg C A Q = 45 := sorry
def cond3 : angle_deg B C P = 30 := sorry
def cond4 : angle_deg Q C A = 30 := sorry
def cond5 : angle_deg A B R = 15 := sorry
def cond6 : angle_deg R A B = 15 := sorry

-- Proving the required properties
theorem triangle_condition_proof (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) (h5 : cond5) (h6 : cond6) :
  angle_deg P R Q = 90 ∧ distance Q R = distance P R := sorry


end triangle_condition_proof_l80_80965


namespace smallest_third_term_geometric_progression_l80_80082

theorem smallest_third_term_geometric_progression :
  ∃ d : ℝ, (a d b : ℝ)  
  (hp1 : a = 5) 
  (hp2 : b = 8 + d)
  (hp3 : a * (17 + 2 * d) = b^2)
  , 17 + 2 * d = 0 :=
begin
  use -17/2, -- The value of d which makes the third term 0
  split, repeat {unfold_projs, trivial},
  rw [hp3], ring, linarith
end

end smallest_third_term_geometric_progression_l80_80082


namespace determine_m_l80_80365

theorem determine_m (m : ℝ) : (A = {1, 2, m}) → (B = {2, 3}) → (A ∪ B = {1, 2, 3}) → m = 3 :=
by
  intro hA hB hUnion
  sorry

end determine_m_l80_80365


namespace diet_cola_cost_l80_80489

theorem diet_cola_cost (T C : ℝ) 
  (h1 : T + 6 + C = 2 * T)
  (h2 : (T + 6 + C) + T = 24) : C = 2 := 
sorry

end diet_cola_cost_l80_80489


namespace sum_factors_of_36_l80_80645

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80645


namespace largest_difference_l80_80572

theorem largest_difference : ∀ X Y : ℕ, 
  (∃ d1 d2 d3 d4 d5 : ℕ, 
    {d1, d2, d3, d4, d5} = {1, 3, 7, 8, 9} ∧ 
    X = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
    Y = d5 * 100 + d3 * 10 + d1) →
  X - Y = 9868 :=
by
  sorry

end largest_difference_l80_80572


namespace major_axis_of_ellipse_with_given_foci_and_line_intersection_l80_80363

noncomputable def major_axis_length (a : ℝ) : ℝ := 2 * a

theorem major_axis_of_ellipse_with_given_foci_and_line_intersection :
  (∃ a > 2, ∀ (x y : ℝ), (x / a) ^ 2 + (y / (a^2 - 4)) ^ 2 = 1 ∧ x + sqrt 3 * y + 4 = 0) →
  (∃ l : ℝ, l = major_axis_length (sqrt 7)) :=
by sorry

end major_axis_of_ellipse_with_given_foci_and_line_intersection_l80_80363


namespace find_n_int_sin_eq_cos_l80_80931

theorem find_n_int_sin_eq_cos (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) : 
  (sin (n : ℝ) * π / 180 = cos (810 * π / 180)) ↔ (n = -180 ∨ n = 0 ∨ n = 180) := 
by
  sorry

end find_n_int_sin_eq_cos_l80_80931


namespace sum_of_factors_36_eq_91_l80_80752

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80752


namespace bouquet_combinations_count_l80_80849

noncomputable def valid_bouquets_count : ℕ :=
  let count_valid_r := (5 ≤ r ∧ r ≤ 30 ∧ (120 - 4 * r) % 3 = 0).count 
  in count_valid_r

theorem bouquet_combinations_count :
  valid_bouquets_count = 9 := sorry

end bouquet_combinations_count_l80_80849


namespace sum_factors_36_eq_91_l80_80598

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80598


namespace sum_of_series_l80_80174

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80174


namespace sum_of_factors_36_l80_80679

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80679


namespace problem1_problem2_l80_80347

-- Defining the conditions
variables (x y : ℝ)
def m : ℝ := 5  -- Example value satisfying m + n = 7
def n : ℝ := 2  -- Example value satisfying m + n = 7

def f (x : ℝ) := |x - 1| - |x + 1|
def F (x y : ℝ) := max (|x^2 - 4*y + m|) (|y^2 - 2*x + n|)

-- Statement 1: solution to the inequality f(x) ≥ (m + n)x is x ≤ 0
theorem problem1 (x : ℝ) : 
    f x ≥ (m + n) * x ↔ x ≤ 0 :=
by 
  sorry

-- Statement 2: minimum value of F is 1
theorem problem2 : 
    F x y ≥ 1 :=
by 
  sorry

end problem1_problem2_l80_80347


namespace minimize_total_cost_l80_80071

noncomputable def gasoline_cost_per_hour (x : ℝ) : ℝ := 6 * (2 + x^2 / 360)
noncomputable def driver_wage_per_hour : ℝ := 18

noncomputable def total_cost (d : ℝ) (x : ℝ) : ℝ :=
  let t := d / x
  t * gasoline_cost_per_hour x + t * driver_wage_per_hour

theorem minimize_total_cost :
  ∀ (d : ℝ) (x : ℝ),
  40 ≤ x ∧ x ≤ 100 ∧ d = 130 →
  (total_cost d x = 130 * (30 / x + x / 60)) →
  ∃ (x : ℝ), x = 30 * real.sqrt 2 ∧ total_cost d x = 130 * real.sqrt 2 :=
begin
  intros d x h1 h2,
  use 30 * real.sqrt 2,
  split,
  { refl },
  { sorry }
end

end minimize_total_cost_l80_80071


namespace sum_factors_of_36_l80_80647

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80647


namespace sum_of_factors_of_36_l80_80785

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80785


namespace water_ratio_horse_pig_l80_80900

-- Definitions based on conditions
def num_pigs : ℕ := 8
def water_per_pig : ℕ := 3
def num_horses : ℕ := 10
def water_for_chickens : ℕ := 30
def total_water : ℕ := 114

-- Statement of the problem
theorem water_ratio_horse_pig : 
  (total_water - (num_pigs * water_per_pig) - water_for_chickens) / num_horses / water_per_pig = 2 := 
by sorry

end water_ratio_horse_pig_l80_80900


namespace count_n_values_l80_80860

theorem count_n_values :
  (finset.Icc 10 2017).count (λ n, (n - 10) * (n - 9) * (n - 8) * (n - 7) = 8 * n * (n - 1) * (n - 2) * (n - 3)) = 450 :=
sorry

end count_n_values_l80_80860


namespace lowest_possible_sale_price_percentage_l80_80829

theorem lowest_possible_sale_price_percentage :
  let P : ℝ := 80
  let D1 : ℝ := 0.50
  let D2 : ℝ := 0.20
  let initial_discounted_price : ℝ := P * (1 - D1)
  let additional_discount : ℝ := P * D2
  let final_price : ℝ := initial_discounted_price - additional_discount
  let percentage_of_list_price : ℝ := (final_price / P) * 100
  in percentage_of_list_price = 30 :=
by {
  sorry
}

end lowest_possible_sale_price_percentage_l80_80829


namespace sum_of_roots_tan_interval_l80_80331

open Real

theorem sum_of_roots_tan_interval :
  ∑ θ in {θ | 0 ≤ θ ∧ θ ≤ 2 * π ∧ (tan θ) ^ 2 - 5 * (tan θ) + 6 = 0}.to_finset ∩ Icc 0 (2 * π), θ 
  = 3 * π :=
by
  sorry

end sum_of_roots_tan_interval_l80_80331


namespace min_sum_of_distances_is_at_center_l80_80506

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2))

def regular_tetrahedron_vertices (a : ℝ) : List (ℝ × ℝ × ℝ) :=
  [(-a, -a, -a), (-a, a, a), (a, -a, a), (a, a, -a)]

def sum_of_distances (p : ℝ × ℝ × ℝ) (a : ℝ) : ℝ :=
  (regular_tetrahedron_vertices a).foldl (λ acc v => acc + distance p v) 0

theorem min_sum_of_distances_is_at_center (a : ℝ) :
  ∀ (x y z : ℝ), 4 * a * Real.sqrt 3 ≤ sum_of_distances (x, y, z) a :=
  sorry

end min_sum_of_distances_is_at_center_l80_80506


namespace tan_theta_eq_neg3_l80_80415

theorem tan_theta_eq_neg3 (θ : Real) (h : (sin (Real.pi - θ) + cos (θ - 2 * Real.pi)) / (sin θ + cos (Real.pi + θ)) = 1 / 2) :
  tan θ = -3 :=
begin
    sorry
end

end tan_theta_eq_neg3_l80_80415


namespace proof_problem_l80_80976

theorem proof_problem : 
  (∑ k in Finset.range 11, (1:ℚ) / (Nat.factorial k * Nat.factorial (20 - k))) = (N : ℚ) / Nat.factorial 21 → 
  Int.floor ((N : ℤ) / 100) = 10485 := by
  sorry

end proof_problem_l80_80976


namespace fraction_product_l80_80127

theorem fraction_product :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := 
by
  -- Detailed proof steps would go here
  sorry

end fraction_product_l80_80127


namespace geometrical_proof_l80_80851

noncomputable def isosceles_triangle (A B C : ℝ) : Prop :=
  A = C ∧ B = C

noncomputable def line_intersects_base_and_circle 
    (A B C M N : ℝ) (isosceles_triangle_ABC : isosceles_triangle A B C) 
    (line_CM_intersects_AB : Prop) 
    (line_CM_intersects_circle : Prop) : Prop :=
  isosceles_triangle_ABC ∧ line_CM_intersects_AB ∧ line_CM_intersects_circle

theorem geometrical_proof (A B C M N : ℝ)
    (h_iso : isosceles_triangle A C B)
    (h_line_intersects : line_intersects_base_and_circle A B C M N h_iso 
        (line_CM_intersects_AB : Prop) 
        (line_CM_intersects_circle : Prop)) :
    (CM * CN = AC ^ 2) ∧ (CM / CN = (AM * BM) / (AN * BN)) :=
begin
  sorry
end

end geometrical_proof_l80_80851


namespace sum_of_factors_36_l80_80688

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80688


namespace collinear_c1_c2_l80_80106

def a : ℝ × ℝ × ℝ := (-1, 2, 8)
def b : ℝ × ℝ × ℝ := (3, 7, -1)

def c1 := (4 * a.1 - 3 * b.1, 4 * a.2 - 3 * b.2, 4 * a.3 - 3 * b.3)
def c2 := (9 * b.1 - 12 * a.1, 9 * b.2 - 12 * a.2, 9 * b.3 - 12 * a.3)

theorem collinear_c1_c2 : ∃ γ : ℝ, c1 = (γ * c2.1, γ * c2.2, γ * c2.3) :=
sorry

end collinear_c1_c2_l80_80106


namespace sum_of_factors_of_36_l80_80670

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80670


namespace groceries_value_l80_80094

-- Conditions
def alex_saved : ℝ := 14500
def car_cost : ℝ := 14600
def trip_charge : ℝ := 1.5
def earn_percentage : ℝ := 0.05
def num_trips : ℝ := 40

-- Proof Statement
theorem groceries_value
  (alex_saved : ℝ)
  (car_cost : ℝ)
  (trip_charge : ℝ)
  (earn_percentage : ℝ)
  (num_trips : ℝ)
  (h_saved : alex_saved = 14500)
  (h_car_cost : car_cost = 14600)
  (h_trip_charge : trip_charge = 1.5)
  (h_earn_percentage : earn_percentage = 0.05)
  (h_num_trips : num_trips = 40) :

  let needed_savings := car_cost - alex_saved in
  let earnings_from_trips := num_trips * trip_charge in
  let earnings_from_groceries := needed_savings - earnings_from_trips in
  let total_value_of_groceries := earnings_from_groceries / earn_percentage in
  total_value_of_groceries = 800 := by {
    sorry
  }

end groceries_value_l80_80094


namespace find_larger_number_l80_80502

theorem find_larger_number :
  ∃ (x y : ℝ), (y = x + 10) ∧ (x = y / 2) ∧ (x + y = 34) → y = 20 :=
by
  sorry

end find_larger_number_l80_80502


namespace sum_series_l80_80255

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80255


namespace expected_absolute_difference_after_10_days_l80_80028

def cat_fox_wealth_difference : ℝ := 1

theorem expected_absolute_difference_after_10_days :
  let p_cat_wins := 0.25
  let p_fox_wins := 0.25
  let p_both_police := 0.5
  let num_days := 10
  ∃ (X : ℕ → ℕ), 
    (X 0 = 0) ∧
    ∀ n, (X (n + 1) = (if (X n = 0) then 0.5 else 0) * X n) →
    (∑ k in range (num_days + 1), (k : ℝ) * (0.5 ^ k)) = cat_fox_wealth_difference := 
sorry

end expected_absolute_difference_after_10_days_l80_80028


namespace eval_f_neg_x_plus_f_x_l80_80994

def f (x : ℝ) : ℝ := 1 / (1 + 2^x)

theorem eval_f_neg_x_plus_f_x (x : ℝ) : f (-x) + f x = 1 := 
  sorry

end eval_f_neg_x_plus_f_x_l80_80994


namespace algorithm_output_is_2_l80_80992

-- Define the variables
variables a b c : ℝ
variable m : ℝ

-- Define the value m based on the condition given in the problem
def calc_m (a b c : ℝ) : ℝ := (4 * a * c - b ^ 2) / (4 * a)

-- Given specific values for a, b, and c
def a := 1
def b := 2
def c := 3

-- The proof that given a = 1, b = 2, c = 3, the output of the algorithm is 2
theorem algorithm_output_is_2 : calc_m a b c = 2 :=
by
  sorry

end algorithm_output_is_2_l80_80992


namespace perimeter_equilateral_triangle_l80_80549

theorem perimeter_equilateral_triangle 
(one_side_eq : ∀ (s : ℕ), 
  let peri_isosceles : ℕ := s + s + 30 in
  peri_isosceles = 70) :
  ∃ (P : ℕ), P = 60 := 
by {
  sorry
}

end perimeter_equilateral_triangle_l80_80549


namespace exists_k_divide_l80_80479

def sequence : ℕ → ℕ
| 0 := 5
| (n+1) := 2 * sequence n + 1

theorem exists_k_divide (n : ℕ) : ∃ k : ℕ, k ≠ n ∧ sequence n ∣ sequence k :=
by
    sorry

end exists_k_divide_l80_80479


namespace proof_l80_80955

-- Lean definitions for the provided conditions
variables {f g : ℝ → ℝ}  -- functions f and g are defined on real numbers
variables {x1 x2 x3 x4 : ℝ}  -- x-intercepts as real numbers

-- Conditions as Lean statements
def conditions 
  (h1 : ∀ x, g x = - f (120 - x))
  (h2 : g (vertex_x f) = f (vertex_x f))
  (h3 : x1 < x2 ∧ x2 < x3 ∧ x3 < x4)
  (h4 : x3 - x2 = 120)
  : Prop := 
  x4 - x1 = 360 + 240 * real.sqrt 2

-- Lean theorem stating the requirement
theorem proof
  (h1 : ∀ x, g x = - f (120 - x))
  (h2 : g (vertex_x f) = f (vertex_x f))
  (h3 : x1 < x2 ∧ x2 < x3 ∧ x3 < x4)
  (h4 : x3 - x2 = 120) :
  x4 - x1 = 360 + 240 * real.sqrt 2 :=
sorry

end proof_l80_80955


namespace biggest_number_in_ratio_l80_80952

theorem biggest_number_in_ratio (A B C D : ℕ) (h1 : 2 * D = 5 * A) (h2 : 3 * D = 5 * B) (h3 : 4 * D = 5 * C) (h_sum : A + B + C + D = 1344) : D = 480 := 
by
  sorry

end biggest_number_in_ratio_l80_80952


namespace rhombus_inscribed_circle_side_length_l80_80859

theorem rhombus_inscribed_circle_side_length
  (O : Type*) [nonempty O]
  (rhombus : O → O → O → O → Prop)
  (inscribed : O → O → ℝ → Prop)
  (radius : ℝ)
  (side_length : ℝ)
  (ABCD : O) (A B C D : O)
  (h_rhombus : rhombus A B C D)
  (h_inscribed : inscribed ABCD radius)
  (h_radius : radius = 100 * real.sqrt 2)
  (h_side1 : side_length = 100)
  (h_side2 : side_length = 100)
  (h_side3 : side_length = 100) :
  side_length = 100 := by
  sorry

end rhombus_inscribed_circle_side_length_l80_80859


namespace groceries_delivered_amount_l80_80098

noncomputable def alex_saved_up : ℝ := 14500
noncomputable def car_cost : ℝ := 14600
noncomputable def charge_per_trip : ℝ := 1.5
noncomputable def percentage_charge : ℝ := 0.05
noncomputable def number_of_trips : ℕ := 40

theorem groceries_delivered_amount :
  ∃ G : ℝ, charge_per_trip * number_of_trips + percentage_charge * G = car_cost - alex_saved_up ∧ G = 800 :=
by {
  use 800,
  rw [mul_comm (800 : ℝ), mul_assoc],
  norm_num,
  exact add_comm 60 (40 : ℝ),
  sorry
}

end groceries_delivered_amount_l80_80098


namespace sin_double_angle_l80_80418

open Real

theorem sin_double_angle (θ : ℝ) (h : cos (π / 4 - θ) = 1 / 2) : sin (2 * θ) = -1 / 2 := 
by 
  sorry

end sin_double_angle_l80_80418


namespace ratio_A_to_B_l80_80866

/--
Proof problem statement:
Given that A and B together can finish the work in 4 days,
and B alone can finish the work in 24 days,
prove that the ratio of the time A takes to finish the work to the time B takes to finish the work is 1:5.
-/
theorem ratio_A_to_B
  (A_time B_time working_together_time : ℝ) 
  (h1 : working_together_time = 4)
  (h2 : B_time = 24)
  (h3 : 1 / A_time + 1 / B_time = 1 / working_together_time) :
  A_time / B_time = 1 / 5 :=
sorry

end ratio_A_to_B_l80_80866


namespace power_division_example_l80_80412

theorem power_division_example (x y : ℝ) (h1 : 3^x = 15) (h2 : 3^y = 5) : 3^(x - y) = 3 := by
  sorry

end power_division_example_l80_80412


namespace minimum_colors_hexagon_tessellation_l80_80009

-- Definitions for problem conditions
def hexagon_shares_six_sides (h : Hexagon) : Prop :=
  h.adjacent.size = 6

def valid_coloring (coloring : Hexagon → Color) : Prop :=
  ∀ h₁ h₂ : Hexagon, h₁.adjacent.contains h₂ → coloring h₁ ≠ coloring h₂

-- The proof problem statement
theorem minimum_colors_hexagon_tessellation (coloring : Hexagon → Color) :
  (∀ h : Hexagon, hexagon_shares_six_sides h) →
  valid_coloring coloring →
  (∃ n : ℕ, n = 3 ∧ (∀ h₁ h₂ : Hexagon, h₁.adjacent.contains h₂ → coloring h₁ ≠ coloring h₂)) :=
sorry

end minimum_colors_hexagon_tessellation_l80_80009


namespace sum_series_l80_80258

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80258


namespace pentagon_to_rectangle_l80_80505

theorem pentagon_to_rectangle :
  ∀ (p : Type) [regular_pentagon p], 
    ∃ (parts : list (set p)), 
      (parts.length = 4) ∧ 
      (∀ i j, i ≠ j → parts.nodup ∧ parts.foldr (∪) ∅ = p) ∧ 
      (rectangle (parts)) :=
by sorry

end pentagon_to_rectangle_l80_80505


namespace sum_of_factors_36_l80_80706

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80706


namespace particle_motion_inverse_relationship_l80_80857

theorem particle_motion_inverse_relationship 
  {k : ℝ} 
  (inverse_relationship : ∀ {n : ℕ}, ∃ t_n d_n, d_n = k / t_n)
  (second_mile : ∃ t_2 d_2, t_2 = 2 ∧ d_2 = 1) : 
  ∃ t_4 d_4, t_4 = 4 ∧ d_4 = 0.5 :=
by
  sorry

end particle_motion_inverse_relationship_l80_80857


namespace sum_series_eq_four_ninths_l80_80212

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80212


namespace sum_series_equals_4_div_9_l80_80218

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80218


namespace large_cube_side_length_painted_blue_l80_80850

   theorem large_cube_side_length_painted_blue (n : ℕ) (h : 6 * n^2 = (1 / 3) * 6 * n^3) : n = 3 :=
   by
     sorry
   
end large_cube_side_length_painted_blue_l80_80850


namespace sum_first_n_terms_geometric_sequence_l80_80400

variables {a : ℕ → ℝ}
variables {S : ℕ → ℝ}

def geometric_series_sum (u : ℕ → ℝ) (a₁ : ℝ) (r : ℝ) :=
  ∀ n, u 1 = a₁ ∧ (∀ k > 0, u (k + 1) = u k * r) →
    S n = (a₁ * (1 - r^n)) / (1 - r)

theorem sum_first_n_terms_geometric_sequence :
  geometric_series_sum a 1 (1/5) →
  S n = (5 / 4) * (1 - (1 / 5)^n) :=
sorry

end sum_first_n_terms_geometric_sequence_l80_80400


namespace sum_of_factors_of_36_l80_80780

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80780


namespace sum_of_factors_36_l80_80696

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80696


namespace sum_of_factors_36_l80_80686

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80686


namespace sum_of_factors_36_eq_91_l80_80592

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80592


namespace sum_of_factors_36_l80_80737

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80737


namespace find_largest_number_l80_80883

theorem find_largest_number (a b c d : ℝ) 
  (h1 : a = Real.sqrt 5)
  (h2 : b = -1.6)
  (h3 : c = 0)
  (h4 : d = 2) 
  (h5 : b < c)
  (h6 : c < d)
  (h7 : d < a) :
  ∃ x, x = a ∧ (x > b) ∧ (x > c) ∧ (x > d) :=
by
  use a
  split
  · assumption
  split
  · linarith
  split
  · linarith
  · linarith

end find_largest_number_l80_80883


namespace sum_of_factors_36_l80_80616

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80616


namespace good_sequence_unique_l80_80081

theorem good_sequence_unique (a : ℕ → ℕ) (h1 : ∀ n : ℕ, n > 0 → a (fact n) = list.prod (list.map a (list.range n.succ)))
    (h2 : ∀ n : ℕ, n > 0 → ∃ b : ℕ, a n = b ^ n) : ∀ n : ℕ, n > 0 → a n = 1 :=
sorry

end good_sequence_unique_l80_80081


namespace expected_difference_after_10_days_l80_80034

noncomputable def cat_fox_expected_difference : ℕ → ℝ
| 0     := 0
| (n+1) := 0.25 * (cat_fox_expected_difference n + 1)  -- cat wins
                + 0.25 * (cat_fox_expected_difference n + 1)  -- fox wins
                + 0.5 * 0  -- both go to police, difference resets

theorem expected_difference_after_10_days :
  cat_fox_expected_difference 10 = 1 :=
sorry

end expected_difference_after_10_days_l80_80034


namespace total_nails_l80_80574

-- Definitions based on the conditions
def Violet_nails : ℕ := 27
def Tickletoe_nails : ℕ := (27 - 3) / 2

-- Theorem to prove the total number of nails
theorem total_nails : Violet_nails + Tickletoe_nails = 39 := by
  sorry

end total_nails_l80_80574


namespace expected_difference_after_10_days_l80_80035

noncomputable def cat_fox_expected_difference : ℕ → ℝ
| 0     := 0
| (n+1) := 0.25 * (cat_fox_expected_difference n + 1)  -- cat wins
                + 0.25 * (cat_fox_expected_difference n + 1)  -- fox wins
                + 0.5 * 0  -- both go to police, difference resets

theorem expected_difference_after_10_days :
  cat_fox_expected_difference 10 = 1 :=
sorry

end expected_difference_after_10_days_l80_80035


namespace amy_school_year_hours_l80_80884

noncomputable def summer_hours_per_week := 40
noncomputable def summer_weeks := 8
noncomputable def summer_earnings := 3200
noncomputable def school_year_weeks := 32
noncomputable def school_year_earnings_needed := 4800

theorem amy_school_year_hours
  (H1 : summer_earnings = summer_hours_per_week * summer_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  (H2 : school_year_earnings_needed = school_year_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  : (school_year_earnings_needed / school_year_weeks / (summer_earnings / (summer_hours_per_week * summer_weeks))) = 15 :=
by
  sorry

end amy_school_year_hours_l80_80884


namespace sum_factors_36_eq_91_l80_80601

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80601


namespace angles_counted_are_right_angles_l80_80894

-- Definitions based on the given conditions
def rectangular_park : Type := {angle : Type}
def square_field : Type := {angle : Type}
def angles_counted (place : Type) : Nat := if place = rectangular_park ∨ place = square_field then 4 else 0

-- Given condition: sum of angles counted in both places is 8
axiom sum_of_angles_eq_8 : angles_counted(rectangular_park) + angles_counted(square_field) = 8

-- Theorem: The type of angles that Avery counted is right angles
theorem angles_counted_are_right_angles : ∃ angles : Type, angles = angles_counted /\ angles = 4 :=
by
  sorry

end angles_counted_are_right_angles_l80_80894


namespace sum_of_factors_of_36_l80_80777

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80777


namespace sin_eq_cos_810_l80_80938

theorem sin_eq_cos_810 (n : ℤ) (h_range : -180 ≤ n ∧ n ≤ 180) : 
  (sin (n : ℝ) * (Real.pi / 180) = cos (810 * (Real.pi / 180))) ↔ (n = -180 ∨ n = 0 ∨ n = 180) := 
by
  sorry

end sin_eq_cos_810_l80_80938


namespace hiker_final_distance_l80_80069

-- Definitions of the movements
def northward_movement : ℤ := 20
def southward_movement : ℤ := 8
def westward_movement : ℤ := 15
def eastward_movement : ℤ := 10

-- Definitions of the net movements
def net_north_south_movement : ℤ := northward_movement - southward_movement
def net_east_west_movement : ℤ := westward_movement - eastward_movement

-- The proof statement
theorem hiker_final_distance : 
  (net_north_south_movement^2 + net_east_west_movement^2) = 13^2 := by 
    sorry

end hiker_final_distance_l80_80069


namespace sum_factors_36_l80_80762

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80762


namespace sum_k_over_4k_l80_80162

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80162


namespace sin_alpha_value_l80_80957

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end sin_alpha_value_l80_80957


namespace sum_of_factors_36_l80_80626

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80626


namespace sum_series_eq_4_div_9_l80_80181

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80181


namespace sum_factors_36_l80_80631

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80631


namespace natural_number_is_integer_l80_80993

theorem natural_number_is_integer
  (natural_numbers_are_integers: ∀ n : ℕ, n ∈ ℤ)
  (two_is_natural: 2 ∈ ℕ)
  : 2 ∈ ℤ :=
sorry

end natural_number_is_integer_l80_80993


namespace sum_of_factors_36_eq_91_l80_80745

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80745


namespace num_divisors_of_32_eq_6_l80_80315

theorem num_divisors_of_32_eq_6 : ∀ n:ℕ, n = 32 → n.divisors.length = 6 :=
by
  intros n h
  rw h
  sorry

end num_divisors_of_32_eq_6_l80_80315


namespace quadratic_roots_equation_l80_80016

open Real

noncomputable def quadratic_roots_are_squares (α : ℝ) : Prop :=
let a := sin (2 * α)
let b := -2 * (sin α + cos α)
let c := (2 : ℝ) in
let d := sqrt ((b^2) - 4 * a * c) / (2 * a) in
let x1 := (-b + d) / (2*a)
let x2 := (-b - d) / (2*a) in
let z1 := x1^2
let z2 := x2^2 in
(cos α * sin α)^4 * z^2 - z + 1 = 0

theorem quadratic_roots_equation (α : ℝ) :
  ∃ z : ℝ, quadratic_roots_are_squares α :=
sorry

end quadratic_roots_equation_l80_80016


namespace sum_series_eq_4_div_9_l80_80188

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80188


namespace sum_of_factors_36_l80_80710

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80710


namespace find_t_given_V_S_l80_80414

variables (g V V0 S S0 a t : ℝ)

theorem find_t_given_V_S :
  (V = g * (t - a) + V0) →
  (S = (1 / 2) * g * (t - a) ^ 2 + V0 * (t - a) + S0) →
  t = a + (V - V0) / g :=
by
  intros h1 h2
  sorry

end find_t_given_V_S_l80_80414


namespace sum_factors_36_l80_80761

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80761


namespace sum_infinite_series_l80_80194

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80194


namespace sum_of_perimeters_equilateral_triangles_l80_80831

noncomputable def sum_of_perimeters (S1 : ℝ) (T : ℕ → ℝ) : ℝ :=
  let P := λ n, 3 * (S1 / (2^n))
  ∑' n, P n

theorem sum_of_perimeters_equilateral_triangles : sum_of_perimeters 40 (λ n, 40 / (2^n)) = 240 :=
by
  sorry

end sum_of_perimeters_equilateral_triangles_l80_80831


namespace sum_of_factors_36_l80_80739

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80739


namespace maximum_x_y_value_l80_80977

theorem maximum_x_y_value (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h1 : x + 2 * y ≤ 6) (h2 : 2 * x + y ≤ 6) : x + y ≤ 4 := 
sorry

end maximum_x_y_value_l80_80977


namespace number_of_valid_sequences_l80_80478

def validSequence (a : List ℕ) : Prop :=
  (List.length a = 10) ∧
  ∀ i, 1 < i ∧ i < 11 → (a.nth (i - 1)).isSome ∧
  ((a.nth (i - 2)).isSome → (a.nth (i - 1)).iget + 1 = (a.nth (i - 2)).iget ∨ (a.nth (i - 1)).iget - 1 = (a.nth (i - 2)).iget)

theorem number_of_valid_sequences :
  ∃ (l : List (List ℕ)), (∀ a, a ∈ l → validSequence a) ∧ (l.length = 512) := sorry

end number_of_valid_sequences_l80_80478


namespace infinite_series_sum_l80_80265

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80265


namespace total_hunts_l80_80442

-- Conditions
def Sam_hunts : ℕ := 6
def Rob_hunts := Sam_hunts / 2
def combined_Rob_Sam_hunts := Rob_hunts + Sam_hunts
def Mark_hunts := combined_Rob_Sam_hunts / 3
def Peter_hunts := 3 * Mark_hunts

-- Question and proof statement
theorem total_hunts : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 := by
  sorry

end total_hunts_l80_80442


namespace a_range_l80_80383

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.log x - (1 / 2) * x^2 + 3 * x

def is_monotonic_on_interval (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a (a + 1), 4 / x - x + 3 > 0

theorem a_range (a : ℝ) :
  is_monotonic_on_interval a → (0 < a ∧ a ≤ 3) :=
by 
  sorry

end a_range_l80_80383


namespace smallest_k_is_5_l80_80143

noncomputable def smallest_k : ℕ :=
  Classical.choose (Exists.intro 5 sorry)

theorem smallest_k_is_5 : smallest_k = 5 :=
sorry

end smallest_k_is_5_l80_80143


namespace min_value_a_plus_9b_l80_80981

theorem min_value_a_plus_9b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 1 / b = 1) : a + 9 * b ≥ 16 :=
  sorry

end min_value_a_plus_9b_l80_80981


namespace condo_total_units_l80_80072

theorem condo_total_units (total_floors : ℕ) (top_penthouse_floors : ℕ) (regular_units_per_floor : ℕ) (penthouse_units_per_floor : ℕ) 
  (h1 : total_floors = 23) (h2 : top_penthouse_floors = 2) (h3 : regular_units_per_floor = 12) (h4 : penthouse_units_per_floor = 2) :
  let regular_floors := total_floors - top_penthouse_floors in
  let regular_units := regular_floors * regular_units_per_floor in
  let penthouse_units := top_penthouse_floors * penthouse_units_per_floor in
  let total_units := regular_units + penthouse_units in
  total_units = 256 :=
by
  sorry

end condo_total_units_l80_80072


namespace increased_side_percent_l80_80541

-- We define the areas and the given conditions
def area_square (side : ℝ) : ℝ := side^2

def increased_side (side : ℝ) (percentage : ℝ) : ℝ := side * (1 + percentage / 100)

theorem increased_side_percent (a : ℝ) (x : ℝ) :
  let b := increased_side a 100 in
  let sum_areas := area_square a + area_square b in
  let c := increased_side b x in
  area_square c = sum_areas * (1 + 159.2 / 100) ->
  x = 80 := by
  sorry

end increased_side_percent_l80_80541


namespace sum_of_factors_36_l80_80730

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80730


namespace maximum_m_value_l80_80986

noncomputable def is_triangular (S : Set ℕ) : Prop := 
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≤ b → b ≤ c → a + b > c

theorem maximum_m_value :
  ∀ S : Set ℕ, (∀ T : Finset ℕ, T.card = 10 → T ⊆ S → is_triangular T) →
  S = { x : ℕ | 4 ≤ x ∧ x ≤ 253 } →
  ∀ T : Finset ℕ, T ⊆ S → T.card = 10 → is_triangular T :=
begin
  sorry
end

end maximum_m_value_l80_80986


namespace binomial_expansion_sum_abs_coefficients_l80_80342

theorem binomial_expansion_sum_abs_coefficients :
  let expansion := fun (x : ℝ) => (1 - 2 * x)^7
  ∃ a : ℕ → ℝ, expansion = ∑ i in Finset.range 8, a i * x^i ∧
    (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| = 2187) := sorry

end binomial_expansion_sum_abs_coefficients_l80_80342


namespace library_visitors_equation_l80_80494

-- Variables representing conditions
variable (x : ℝ)  -- Monthly average growth rate
variable (first_month_visitors : ℝ) -- Visitors in the first month
variable (total_visitors_by_third_month : ℝ) -- Total visitors by the third month

-- Setting specific values for conditions
def first_month_visitors := 600
def total_visitors_by_third_month := 2850

-- The Lean statement that the specified equation holds
theorem library_visitors_equation :
  first_month_visitors + first_month_visitors * (1 + x) + first_month_visitors * (1 + x)^2 = total_visitors_by_third_month :=
sorry

end library_visitors_equation_l80_80494


namespace evaluate_series_sum_l80_80279

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80279


namespace sum_factors_36_l80_80799

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80799


namespace center_value_is_36_l80_80888

noncomputable def first_row_term_n (n : ℕ) : ℤ :=
2 + (n - 1) * 6

noncomputable def sixth_row_term_n (n : ℕ) : ℤ :=
12 + (n - 1) * 12

noncomputable def center_value : ℤ :=
let a1 := (first_row_term_n 4) in
let a6 := (sixth_row_term_n 4) in
a1 + ((3 - 1) * (a6 - a1) / 2)

theorem center_value_is_36 : 
  center_value = 36 :=
sorry

end center_value_is_36_l80_80888


namespace sum_of_factors_36_eq_91_l80_80755

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80755


namespace data_set_conditions_l80_80528

theorem data_set_conditions (x : ℕ → ℝ) (n : ℕ) (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → x i > 0)
  (h_len : n = 4) (h_mean : (x 1 + x 2 + x 3 + x 4) / n = 2)
  (h_median : (x 2 + x 3) / 2 = 2)
  (h_std_dev : (x 1 - 2)^2 + (x 2 - 2)^2 + (x 3 - 2)^2 + (x 4 - 2)^2 = 4) :
  ∃ y, (y = [1, 1, 3, 3]) ∧ (∀ i, 1 ≤ i ∧ i ≤ n → x i = y (i-1)) := 
begin
  sorry
end

end data_set_conditions_l80_80528


namespace fractions_sum_to_decimal_l80_80051

theorem fractions_sum_to_decimal :
  (2 / 10) + (4 / 100) + (6 / 1000) = 0.246 :=
by 
  sorry

end fractions_sum_to_decimal_l80_80051


namespace series_sum_eq_four_ninths_l80_80238

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80238


namespace max_cubes_fit_l80_80579

theorem max_cubes_fit (L S : ℕ) (hL : L = 10) (hS : S = 2) : (L * L * L) / (S * S * S) = 125 := by
  sorry

end max_cubes_fit_l80_80579


namespace sum_of_factors_of_36_l80_80776

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80776


namespace boat_speed_in_still_water_l80_80122

variables (V_b V_c V_w : ℝ)

-- Conditions from the problem
def speed_upstream (V_b V_c V_w : ℝ) : ℝ := V_b - V_c - V_w
def water_current_range (V_c : ℝ) : Prop := 2 ≤ V_c ∧ V_c ≤ 4
def wind_resistance_range (V_w : ℝ) : Prop := -1 ≤ V_w ∧ V_w ≤ 1
def upstream_speed : Prop := speed_upstream V_b 4 (2 - (-1)) + (2 - -1) = 4

-- Statement of the proof problem
theorem boat_speed_in_still_water :
  (∀ V_c V_w, water_current_range V_c → wind_resistance_range V_w → speed_upstream V_b V_c V_w = 4) → V_b = 7 :=
by
  sorry

end boat_speed_in_still_water_l80_80122


namespace exponent_rules_application_l80_80898

theorem exponent_rules_application : (π - 3) ^ 0 + (1 / 2) ^ (-1) = 3 := by
  sorry

end exponent_rules_application_l80_80898


namespace tan_2beta_l80_80345

theorem tan_2beta {α β : ℝ} 
  (h₁ : Real.tan (α + β) = 2) 
  (h₂ : Real.tan (α - β) = 3) : 
  Real.tan (2 * β) = -1 / 7 :=
by 
  sorry

end tan_2beta_l80_80345


namespace identify_irrational_number_l80_80014

-- Definitions extracted from the problem
def is_irrational (x : ℝ) : Prop := ¬∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem identify_irrational_number :
  ∀ (x : ℝ), x ∈ {Real.sqrt 16, -2, -Real.pi, 0} → is_irrational x ↔ x = -Real.pi :=
by
  intro x
  intro hx
  split
  case hp =>
    intro hi
    have h_set : x = Real.sqrt 16 ∨ x = -2 ∨ x = -Real.pi ∨ x = 0 := by
      exact Set.mem_insert_iff.mp hx
    cases h_set
    case inl h =>
      simp [Real.sqrt] at hi
      sorry
    case inr h0 =>
      cases h0
      case inl h =>
        simp at hi
        sorry
      case inr h1 =>
        cases h1
        case inl h =>
          exact h
        case inr h =>
          simp at hi
          sorry
  case hq =>
    intro heq
    rw heq
    apply Rational.not_is_irrational
    rw Rat.ilog_ofIrrationalPi
    exact Rat.inv_pos_lt_absReal_pi

end identify_irrational_number_l80_80014


namespace proj_compute_l80_80476

open Real

variables (v w : ℝ^3)

def proj_w_v : ℝ^3 := (3, -2, 1)

theorem proj_compute :
  (proj_w w (3 * v + w)) = (9 + w.1, -6 + w.2, 3 + w.3) :=
by sorry

end proj_compute_l80_80476


namespace infinite_series_sum_l80_80268

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80268


namespace sum_of_factors_36_l80_80729

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80729


namespace sum_factors_36_eq_91_l80_80597

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80597


namespace probability_of_sum_equals_4_l80_80559

noncomputable def probability_sum_equals_4 : ℝ :=
let total := 3.5 in
let intervals := [((0 : ℝ), 0.5), (0.5, 1.5)] in
let total_length := 3.5 in
let sum_length := intervals.foldl (λ acc (a, b), acc + (b - a)) 0 in
sum_length / total_length

theorem probability_of_sum_equals_4 : probability_sum_equals_4 = 3 / 7 :=
by
  sorry

end probability_of_sum_equals_4_l80_80559


namespace cloves_of_garlic_needed_l80_80056

def cloves_needed_for_vampires (vampires : ℕ) : ℕ :=
  (vampires * 3) / 2

def cloves_needed_for_wights (wights : ℕ) : ℕ :=
  (wights * 3) / 3

def cloves_needed_for_vampire_bats (vampire_bats : ℕ) : ℕ :=
  (vampire_bats * 3) / 8

theorem cloves_of_garlic_needed (vampires wights vampire_bats : ℕ) :
  cloves_needed_for_vampires 30 + cloves_needed_for_wights 12 + 
  cloves_needed_for_vampire_bats 40 = 72 :=
by
  sorry

end cloves_of_garlic_needed_l80_80056


namespace sum_series_eq_4_div_9_l80_80186

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80186


namespace percentage_error_in_area_l80_80832

theorem percentage_error_in_area (s : ℝ) (h_s_pos: s > 0) :
  let measured_side := 1.01 * s
  let actual_area := s ^ 2
  let measured_area := measured_side ^ 2
  let error_in_area := measured_area - actual_area
  (error_in_area / actual_area) * 100 = 2.01 :=
by
  sorry

end percentage_error_in_area_l80_80832


namespace hyperbola_eccentricity_proof_l80_80995

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (B_y : ℝ) (B_y_def : B_y = (sqrt 15 / 3) * b) : ℝ :=
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  if h : (8 * a^2 - 5 * c^2 + 6 * a * c = 0) then e else e

theorem hyperbola_eccentricity_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (B_y : ℝ) (B_y_def : B_y = (sqrt 15 /3) * b)
  (h : ∃ e, 5 * e^2 - 6 * e - 8 = 0) :
  hyperbola_eccentricity a b ha hb B_y B_y_def = 2 :=
begin
  sorry
end

end hyperbola_eccentricity_proof_l80_80995


namespace sum_of_series_l80_80175

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80175


namespace func_max_l80_80012

-- Condition: The function defined as y = 2 * cos x - 1
def func (x : ℝ) : ℝ := 2 * Real.cos x - 1

-- Proof statement: Prove that the function achieves its maximum value when x = 2 * k * π for k ∈ ℤ
theorem func_max (x : ℝ) (k : ℤ) : (∃ x, func x = 2 * 1 - 1) ↔ (∃ k : ℤ, x = 2 * k * Real.pi) := sorry

end func_max_l80_80012


namespace triangle_exists_l80_80349

def point : Type := ℕ -- Using natural numbers to denote points.

def is_segment (segments : set (point × point)) (p1 p2 : point) : Prop :=
(p1, p2) ∈ segments ∨ (p2, p1) ∈ segments

noncomputable def segments (n : ℕ) : set (point × point) := sorry
-- We assume segments are a subset of pairs of points. The specific set is not provided here.

def mutual_triple (segments : set (point × point)) (p1 p2 p3 : point) : Prop :=
is_segment segments p1 p2 ∧ is_segment segments p1 p3 ∧ is_segment segments p2 p3

theorem triangle_exists (n : ℕ) (segments : set (point × point)) (h : 2 * n)
  (segment_count : (set.to_finset segments).card = n^2 + 1)
  (no_mutual_triple : ∀ p1 p2 p3, ¬ mutual_triple segments p1 p2 p3) :
  false :=
sorry

end triangle_exists_l80_80349


namespace compute_BC_l80_80135

-- Definition of a parallelogram and relevant points on it
structure Parallelogram (A B C D : Type) :=
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ)
  (AB_gt_BC : AB > BC)
  (E_on_AB : E : Point A B)
  (F_on_CD : F : Point C D)

-- Properties related to circles passing through given points
structure Circle (ω : Type) :=
  (passes_through : Point ω → Prop)
  (ω1 : passes_through A ∧ passes_through D ∧ passes_through E ∧ passes_through F)
  (ω2 : passes_through B ∧ passes_through C ∧ passes_through E ∧ passes_through F)

-- Segment and partition lengths on line BD
structure Partition (BD : ℝ) :=
  (BX : ℝ)
  (XY : ℝ)
  (YD : ℝ)

-- Lean 4 theorem statement
theorem compute_BC (A B C D E F X Y : Point) (AB BD : Line ℝ) 
  (ω1 ω2 : Circle ℝ) (partition : Partition BD)
  (h1 : AB > BC)
  (h2 : ω1.passes_through A ∧ ω1.passes_through D ∧ ω1.passes_through E ∧ ω1.passes_through F)
  (h3 : ω2.passes_through B ∧ ω2.passes_through C ∧ ω2.passes_through E ∧ ω2.passes_through F)
  (h4 : partition.BX = 200)
  (h5 : partition.XY = 9)
  (h6 : partition.YD = 80)
  : BC = 51 :=
sorry

end compute_BC_l80_80135


namespace C1C2_parallel_AB_l80_80050

variables {O A B P A1 B1 A2 B2 C1 C2 : Type}
variables [Circle O]

-- Conditions
def midpoint (P : Type) (A B : Type) : Prop := -- Definition stub
def chord_through_point (A1 B1 A2 B2 P : Type) : Prop := -- Definition stub
def tangents_intersection (C1 C2 A1 B1 A2 B2 : Type) : Prop := -- Definition stub
def parallel (C1 C2 AB : Type) : Prop := -- Definition stub

-- Given Conditions
axiom AB_is_chord : chord O A B
axiom AB_not_diameter : ¬diameter O A B
axiom P_midpoint_AB : midpoint P A B
axiom chords_through_P : chord_through_point A1 B1 A2 B2 P
axiom C1_tangent_intersection : tangents_intersection C1 O A1 B1
axiom C2_tangent_intersection : tangents_intersection C2 O A2 B2

-- Proof Goal
theorem C1C2_parallel_AB :
  parallel C1 C2 AB := sorry

end C1C2_parallel_AB_l80_80050


namespace amy_school_year_hours_l80_80885

noncomputable def summer_hours_per_week := 40
noncomputable def summer_weeks := 8
noncomputable def summer_earnings := 3200
noncomputable def school_year_weeks := 32
noncomputable def school_year_earnings_needed := 4800

theorem amy_school_year_hours
  (H1 : summer_earnings = summer_hours_per_week * summer_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  (H2 : school_year_earnings_needed = school_year_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  : (school_year_earnings_needed / school_year_weeks / (summer_earnings / (summer_hours_per_week * summer_weeks))) = 15 :=
by
  sorry

end amy_school_year_hours_l80_80885


namespace neo_tokropolis_monthly_population_change_l80_80458

theorem neo_tokropolis_monthly_population_change :
  (∃ (nb_hours : ℕ) (nd_hours : ℕ) (hours_per_day : ℕ) (days_per_month : ℕ),
    nb_hours = 12 ∧ nd_hours = 36 ∧ hours_per_day = 24 ∧ days_per_month = 30 ∧
    (days_per_month * (hours_per_day / nb_hours - hours_per_day / nd_hours)) = 40) :=
begin
  let nb_hours := 12,
  let nd_hours := 36,
  let hours_per_day := 24,
  let days_per_month := 30,
  have h1: nb_hours = 12, by simp,
  have h2: nd_hours = 36, by simp,
  have h3: hours_per_day = 24, by simp,
  have h4: days_per_month = 30, by simp,
  have h5: days_per_month * (hours_per_day / nb_hours - hours_per_day / nd_hours) = 40, sorry,
  exact ⟨nb_hours, nd_hours, hours_per_day, days_per_month, h1, h2, h3, h4, h5⟩
end

end neo_tokropolis_monthly_population_change_l80_80458


namespace concentration_is_correct_l80_80870

def concentration_of_alcohol_in_mixture (a1 a2 v1 v2 total_volume: ℝ) : ℝ :=
  (a1 * v1 + a2 * v2) / total_volume * 100

theorem concentration_is_correct :
  concentration_of_alcohol_in_mixture 0.35 0.50 2 6 8 = 46.25 := 
by
  sorry

end concentration_is_correct_l80_80870


namespace different_weights_detectable_l80_80101

theorem different_weights_detectable (k : ℕ) (coins : set ℝ) :
  coins.card = 2^k →
  (∀ a b c ∈ coins, a ≠ b → b ≠ c → c ≠ a → (a = b ∨ b = c ∨ c = a)) →
  ∃ (lighter heavier : ℝ), lighter ∈ coins ∧ heavier ∈ coins ∧ lighter < heavier :=
by
  intros h_card h_no_three_diff
  -- We'll proceed by using the given conditions and perform k measurements
  sorry

end different_weights_detectable_l80_80101


namespace sum_of_series_l80_80169

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80169


namespace fraction_product_l80_80130

theorem fraction_product : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l80_80130


namespace min_distance_from_ellipse_to_line_l80_80355

noncomputable def minimum_distance_point_to_line_on_ellipse : ℝ :=
  let ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 9) = 1
  let line (x y : ℝ) := 2 * x + y - 10 = 0
  if ∃ x y, ellipse x y ∧ ∃ u v, line u v then sqrt 5 else 0

theorem min_distance_from_ellipse_to_line :
  minimum_distance_point_to_line_on_ellipse = sqrt 5 :=
  sorry

end min_distance_from_ellipse_to_line_l80_80355


namespace sum_series_eq_four_ninths_l80_80209

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80209


namespace sin_eq_cos_810_l80_80939

theorem sin_eq_cos_810 (n : ℤ) (h_range : -180 ≤ n ∧ n ≤ 180) : 
  (sin (n : ℝ) * (Real.pi / 180) = cos (810 * (Real.pi / 180))) ↔ (n = -180 ∨ n = 0 ∨ n = 180) := 
by
  sorry

end sin_eq_cos_810_l80_80939


namespace hare_to_12th_cell_l80_80068

noncomputable def hare_jumps : ℕ → ℕ
| 0     := 0  -- This is to handle the 0-cell case which is not needed
| 1     := 1
| 2     := 1
| (n+3) := hare_jumps (n+2) + hare_jumps (n+1)

theorem hare_to_12th_cell : hare_jumps 12 = 144 :=
sorry

end hare_to_12th_cell_l80_80068


namespace sum_reciprocal_roots_l80_80480

noncomputable def polynomial : Polynomial ℝ :=
  x^2018 + x^2017 + x^2016 + ... + x^2 + x -  1345

theorem sum_reciprocal_roots :
  let a := polynomialRoots polynomial
  ( ∑ n in a, 1 / (1 - n) ) = 3027 := 
sorry

end sum_reciprocal_roots_l80_80480


namespace sum_series_l80_80256

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80256


namespace evaluate_series_sum_l80_80282

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80282


namespace sum_series_l80_80253

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80253


namespace locus_of_center_l80_80309

-- Define point A
def PointA : ℝ × ℝ := (-2, 0)

-- Define the tangent line
def TangentLine : ℝ := 2

-- The condition to prove the locus equation
theorem locus_of_center (x₀ y₀ : ℝ) :
  (∃ r : ℝ, abs (x₀ - TangentLine) = r ∧ (x₀ + 2)^2 + y₀^2 = r^2) →
  y₀^2 = -8 * x₀ := by
  sorry

end locus_of_center_l80_80309


namespace daily_charges_correct_l80_80511

def days_in_non_leap_year : ℕ := 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 4

def total_amount_paid : ℝ := 7038

def daily_coaching_charges : ℝ := total_amount_paid / days_in_non_leap_year

theorem daily_charges_correct : daily_coaching_charges = 22.86 := by
  have h1 : days_in_non_leap_year = 308 := rfl
  have h2 : total_amount_paid = 7038 := rfl
  have h3 : daily_coaching_charges = 7038 / 308 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end daily_charges_correct_l80_80511


namespace ellipse_equation_line_slope_l80_80973

theorem ellipse_equation
  (a b : ℝ) (a_gt_b : a > b) (ellipse_pass : (a = sqrt 5) ∧ (b > 0) ∧ ((-(sqrt 2), 1) ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 })) :
  a^2 * x^2 + b^2 * y^2 = 5 := sorry

theorem line_slope
  (k : ℝ) (C : ℝ × ℝ) (C_coord : C = (-1, 0))
  (ellipse : ∀ p : ℝ × ℝ, p ∈ { p : ℝ × ℝ | p.1^2 + 3 * (p.2^2) = 5 })
  (line_slope : ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + B.1) / 2 = -1 / 2 ∧
                                     ∃ k : ℝ, ∃ C : ℝ × ℝ, C = (-1, 0) ∧ (∀ p : ℝ × ℝ, (p.2 = k * (p.1 + 1)) → p ∈ { p : ℝ × ℝ | p.1^2 + 3 * (p.2^2) = 5 })) :
  k = (sqrt 3 / 3) ∨ k = -(sqrt 3 / 3) := sorry

end ellipse_equation_line_slope_l80_80973


namespace divide_square_into_trapezoids_l80_80920

theorem divide_square_into_trapezoids (s : ℝ) (h1 h2 h3 h4 : ℝ) :
  s = 7 ∧ h1 = 1 ∧ h2 = 2 ∧ h3 = 3 ∧ h4 = 4 →
  ∃ (a b c d : ℝ), a = h1 ∧ b = h2 ∧ c = h3 ∧ d = h4 ∧ a + b + c + d = s :=
by
  intro h,
  rcases h with ⟨hs, h1, h2, h3, h4⟩,
  use h1, h2, h3, h4,
  exact ⟨h1, h2, h3, h4, by linarith⟩,
  sorry -- Proof steps would go here

end divide_square_into_trapezoids_l80_80920


namespace sum_of_factors_36_l80_80725

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80725


namespace fraction_of_boxes_loaded_by_day_crew_l80_80891

theorem fraction_of_boxes_loaded_by_day_crew
    (dayCrewBoxesPerWorker : ℚ)
    (dayCrewWorkers : ℚ)
    (nightCrewBoxesPerWorker : ℚ := (3 / 4) * dayCrewBoxesPerWorker)
    (nightCrewWorkers : ℚ := (3 / 4) * dayCrewWorkers) :
    (dayCrewBoxesPerWorker * dayCrewWorkers) / ((dayCrewBoxesPerWorker * dayCrewWorkers) + (nightCrewBoxesPerWorker * nightCrewWorkers)) = 16 / 25 :=
by
  sorry

end fraction_of_boxes_loaded_by_day_crew_l80_80891


namespace sin_eq_cos_810_l80_80937

theorem sin_eq_cos_810 (n : ℤ) (h_range : -180 ≤ n ∧ n ≤ 180) : 
  (sin (n : ℝ) * (Real.pi / 180) = cos (810 * (Real.pi / 180))) ↔ (n = -180 ∨ n = 0 ∨ n = 180) := 
by
  sorry

end sin_eq_cos_810_l80_80937


namespace number_of_marked_points_l80_80512

theorem number_of_marked_points
  (a1 a2 b1 b2 : ℕ)
  (hA : a1 * a2 = 50)
  (hB : b1 * b2 = 56)
  (h_sum : a1 + a2 = b1 + b2) :
  a1 + a2 + 1 = 16 :=
sorry

end number_of_marked_points_l80_80512


namespace sum_of_n_when_f_is_prime_l80_80947

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def f (n : ℕ) : ℕ := n^3 - 180 * n^2 + 40 * n + 400

theorem sum_of_n_when_f_is_prime (a b : ℕ) (ha : is_prime (f a)) (hb : is_prime (f b)) (hapos : a > 0) (hbpos : b > 0) :
  ∑ n in {n | is_prime (f n) ∧ n > 0}.to_finset, n = a + b :=
sorry

end sum_of_n_when_f_is_prime_l80_80947


namespace nonnegative_solution_positive_solution_l80_80481

/-- For k > 7, there exist non-negative integers x and y such that 5*x + 3*y = k. -/
theorem nonnegative_solution (k : ℤ) (hk : k > 7) : ∃ x y : ℕ, 5 * x + 3 * y = k :=
sorry

/-- For k > 15, there exist positive integers x and y such that 5*x + 3*y = k. -/
theorem positive_solution (k : ℤ) (hk : k > 15) : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 5 * x + 3 * y = k :=
sorry

end nonnegative_solution_positive_solution_l80_80481


namespace correct_propositions_l80_80378

variable (P1 P2 P3 P4 : Prop)

-- Proposition 1: The negation of ∀ x ∈ ℝ, cos(x) > 0 is ∃ x ∈ ℝ such that cos(x) ≤ 0. 
def prop1 : Prop := 
  (¬ (∀ x : ℝ, Real.cos x > 0)) ↔ (∃ x : ℝ, Real.cos x ≤ 0)

-- Proposition 2: If 0 < a < 1, then the equation x^2 + a^x - 3 = 0 has only one real root.
def prop2 : Prop := 
  ∀ a : ℝ, (0 < a ∧ a < 1) → (∃! x : ℝ, x^2 + a^x - 3 = 0)

-- Proposition 3: For any real number x, if f(-x) = f(x) and f'(x) > 0 when x > 0, then f'(x) < 0 when x < 0.
def prop3 (f : ℝ → ℝ) : Prop := 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, x > 0 → deriv f x > 0) →
  (∀ x : ℝ, x < 0 → deriv f x < 0)

-- Proposition 4: For a rectangle with area S and perimeter l, the pair of real numbers (6, 8) is a valid (S, l) pair.
def prop4 : Prop :=
  ∃ (a b : ℝ), (a * b = 6) ∧ (2 * (a + b) = 8)

theorem correct_propositions (P1_def : prop1)
                            (P3_def : ∀ f : ℝ → ℝ, prop3 f) :
                          P1 ∧ P3 :=
by
  sorry

end correct_propositions_l80_80378


namespace series_sum_eq_four_ninths_l80_80237

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80237


namespace sufficient_but_not_necessary_condition_l80_80366

-- Step d: Lean 4 statement
theorem sufficient_but_not_necessary_condition 
  (m n : ℕ) (e : ℚ) (h₁ : m = 5) (h₂ : n = 4) (h₃ : e = 3 / 5)
  (ellipse_eq : ∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) :
  (m = 5 ∧ n = 4) → (e = 3 / 5) ∧ (¬(e = 3 / 5 → m = 5 ∧ n = 4)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l80_80366


namespace sum_of_factors_of_36_l80_80773

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80773


namespace determine_s_l80_80147

noncomputable def quadratic_root_conjugate (p s : ℝ) : Prop :=
  let root1 := 4 + 3 * Complex.i in
  let root2 := 4 - 3 * Complex.i in
  let sum_roots := root1 + root2 in
  let prod_roots := root1 * root2 in
  let quadratic_eq := ∀ x, 3 * x^2 + p * x + s = 0 in
  sum_roots = 8 ∧ prod_roots = 25 ∧ s = 75

theorem determine_s (p s : ℝ) (h : quadratic_root_conjugate p s) : s = 75 :=
by
  sorry

end determine_s_l80_80147


namespace find_a_l80_80303

theorem find_a (a : ℝ) :
  (∃ (r d : ℂ), r - d, r, r + d are the roots of the polynomial x^3 - 6 * x^2 + 21 * x + a ∧
  ¬ (r - d).im = 0 ∧ ¬ r.im = 0 ∧ ¬ (r + d).im = 0 ∧
  (r - d) + r + (r + d) = 6 ∧
  (r - d) * r + (r - d) * (r + d) + r * (r + d) = 21) ↔
  (a = -26) := 
sorry

end find_a_l80_80303


namespace expected_difference_after_10_days_l80_80023

-- Define the initial state and transitions
noncomputable def initial_prob (k : ℤ) : ℝ :=
if k = 0 then 1 else 0

noncomputable def transition_prob (k : ℤ) (n : ℕ) : ℝ :=
0.5 * initial_prob k +
0.25 * initial_prob (k - 1) +
0.25 * initial_prob (k + 1)

-- Define event probability for having any wealth difference after n days
noncomputable def p_k_n (k : ℤ) (n : ℕ) : ℝ :=
if n = 0 then initial_prob k
else transition_prob k (n - 1)

-- Use expected value of absolute difference between wealths 
noncomputable def expected_value_abs_diff (n : ℕ) : ℝ :=
Σ' k, |k| * p_k_n k n

-- Finally, state the theorem
theorem expected_difference_after_10_days :
expected_value_abs_diff 10 = 1 :=
by
  sorry

end expected_difference_after_10_days_l80_80023


namespace hyperbola_slope_sum_l80_80384

variable {x y k : ℝ}

def on_hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 2) = 1

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

theorem hyperbola_slope_sum :
  ∀ A B C D E F : ℝ × ℝ,
  on_hyperbola A.1 A.2 → on_hyperbola B.1 B.2 → on_hyperbola C.1 C.2 →
  D = midpoint A B → E = midpoint B C → F = midpoint A C →
  slope (0, 0) D + slope (0, 0) E + slope (0, 0) F = -1 →
  1 / slope A B + 1 / slope B C + 1 / slope A C = -2 :=
by
  intros
  sorry

end hyperbola_slope_sum_l80_80384


namespace trigonometric_evaluation_monotonic_intervals_l80_80053

-- Statement for (1)
theorem trigonometric_evaluation : 
  sin (120 * Real.pi / 180) ^ 2 + cos (180 * Real.pi / 180) + tan (45 * Real.pi / 180)
  - cos (-330 * Real.pi / 180) ^ 2 + sin (-210 * Real.pi / 180) = 1 / 2 :=
by
  sorry

-- Statement for (2)
theorem monotonic_intervals :
  ∃ k : ℤ, 
    (∀ x ∈ (Icc (-(Real.pi / 2) + 2 * k * Real.pi) (Real.pi / 2 + 2 * k * Real.pi)), 
    (1/3) ^ sin x ≤ (1/3) ^ sin (x + Real.pi))
∧ ∃ k : ℤ, 
    (∀ x ∈ (Icc (Real.pi / 2 + 2 * k * Real.pi) (3 * Real.pi / 2 + 2 * k * Real.pi)), 
    (1/3) ^ sin x ≥ (1/3) ^ sin (x + Real.pi)) :=
by
  sorry

end trigonometric_evaluation_monotonic_intervals_l80_80053


namespace leakage_empty_time_l80_80828

variables (a : ℝ) (h1 : a > 0) -- Assuming a is positive for the purposes of the problem

theorem leakage_empty_time (h : 7 * a > 0) : (7 * a) / 6 = 7 * a / 6 :=
by
  sorry

end leakage_empty_time_l80_80828


namespace pictures_per_album_l80_80837

-- Define the conditions
def uploaded_pics_phone : ℕ := 22
def uploaded_pics_camera : ℕ := 2
def num_albums : ℕ := 4

-- Define the total pictures uploaded
def total_pictures : ℕ := uploaded_pics_phone + uploaded_pics_camera

-- Define the target statement as the theorem
theorem pictures_per_album : (total_pictures / num_albums) = 6 := by
  sorry

end pictures_per_album_l80_80837


namespace compute_fraction_l80_80908

theorem compute_fraction (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) : 
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end compute_fraction_l80_80908


namespace sequence_is_arithmetic_l80_80376

/-- Definition of the sum of the first n terms of a sequence -/
def Sn (n : ℕ) : ℕ := n^2

/-- Definition of the nth term of the sequence -/
def an (n : ℕ) : ℕ :=
  if n = 0 then 0 else Sn n - Sn (n - 1)

theorem sequence_is_arithmetic :
  ∀ (n : ℕ), an (n + 1) - an n = 2 :=
begin
  sorry,
end

end sequence_is_arithmetic_l80_80376


namespace sum_of_factors_36_l80_80705

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80705


namespace sqrt_expr_evaluation_l80_80152

theorem sqrt_expr_evaluation :
  sqrt (5 + 6 * sqrt 2) - sqrt (5 - 6 * sqrt 2) = 4 * sqrt 2 :=
by
  sorry

end sqrt_expr_evaluation_l80_80152


namespace arrange_polynomial_descending_l80_80107

variable (a b : ℤ)

def polynomial := -a + 3 * a^5 * b^3 + 5 * a^3 * b^5 - 9 + 4 * a^2 * b^2 

def rearranged_polynomial := 3 * a^5 * b^3 + 5 * a^3 * b^5 + 4 * a^2 * b^2 - a - 9

theorem arrange_polynomial_descending :
  polynomial a b = rearranged_polynomial a b :=
sorry

end arrange_polynomial_descending_l80_80107


namespace sum_of_factors_of_36_l80_80775

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80775


namespace cubic_root_sum_l80_80945

theorem cubic_root_sum :
  let a := (25 + 10 * Real.sqrt 5)^(1/3)
  let b := (25 - 10 * Real.sqrt 5)^(1/3)
  let x := a + b
  x = 5 :=
by
  let a := (25 + 10 * Real.sqrt 5)^(1/3)
  let b := (25 - 10 * Real.sqrt 5)^(1/3)
  let x := a + b
  have h1 : a^3 = 25 + 10 * Real.sqrt 5 := sorry
  have h2 : b^3 = 25 - 10 * Real.sqrt 5 := sorry
  have h3 : a^3 + b^3 = 50 := sorry
  have h4 : a * b = 5 := sorry
  have h5 : x^3 = 50 + 15 * x := sorry
  have h6 : x = 5 := by
    apply Real.solve_cubic_eq; assumption
  exact h6

end cubic_root_sum_l80_80945


namespace sum_of_factors_36_eq_91_l80_80587

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80587


namespace sum_factors_36_l80_80633

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80633


namespace sum_factors_of_36_l80_80646

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80646


namespace monotonically_increasing_interval_l80_80536

theorem monotonically_increasing_interval {k : ℤ} :
  ∀ x, (-π / 6 : ℝ) ≤ x ∧ x ≤ (5 * π / 6 : ℝ) → monotone_on (λ x, sin (x - π / 3)) (Icc (-π / 6) (5 * π / 6)) :=
sorry

end monotonically_increasing_interval_l80_80536


namespace number_of_quadruples_l80_80318

/-- The number of ordered quadruples (a, b, c, d) of nonnegative real numbers such that
    a² + b² + c² + d² = 9 and (a + b + c + d)(a³ + b³ + c³ + d³) = 81 is 15. -/
theorem number_of_quadruples :
  ∃ (s : Finset (ℝ × ℝ × ℝ × ℝ)),
    (∀ (p : ℝ × ℝ × ℝ × ℝ), p ∈ s → 
     let (a, b, c, d) := p in 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 
     a^2 + b^2 + c^2 + d^2 = 9 ∧ 
     (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81) ∧ 
    s.card = 15 :=
by
  sorry

end number_of_quadruples_l80_80318


namespace sum_series_eq_4_div_9_l80_80287

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80287


namespace proof_problem_l80_80983

def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem proof_problem : f(2018) / f(2018⁻¹) = -1 := by
  sorry

end proof_problem_l80_80983


namespace alpha_range_l80_80979

theorem alpha_range (α : ℝ) (h1 : 0 < α ∧ α < 2 * π)
  (h2 : Real.sin α > 0)
  (h3 : Real.cos α < 0) :
  α ∈ Ioo (π / 2) π :=
by
  sorry

end alpha_range_l80_80979


namespace find_c_for_equal_real_roots_l80_80431

theorem find_c_for_equal_real_roots
  (c : ℝ)
  (h : ∀ x : ℝ, x^2 + 6 * x + c = 0 → x = -3) : c = 9 :=
sorry

end find_c_for_equal_real_roots_l80_80431


namespace sum_of_factors_of_36_l80_80672

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80672


namespace calculate_fg1_l80_80423

def f (x : ℝ) : ℝ := 4 - 3 * x
def g (x : ℝ) : ℝ := x^3 + 1

theorem calculate_fg1 : f (g 1) = -2 :=
by
  sorry

end calculate_fg1_l80_80423


namespace sum_of_factors_36_eq_91_l80_80595

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80595


namespace sum_factors_of_36_l80_80651

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80651


namespace cos_210_eq_neg_sqrt3_over_2_l80_80897

theorem cos_210_eq_neg_sqrt3_over_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_over_2_l80_80897


namespace man_savings_l80_80853

theorem man_savings (I : ℝ) (S : ℝ) (h1 : S = 0.35) (h2 : 2 * (0.65 * I) = 0.65 * I + 0.70 * I) :
  S = 0.35 :=
by
  -- Introduce necessary assumptions
  let savings_first_year := S * I
  let expenditure_first_year := I - savings_first_year
  let savings_second_year := 2 * savings_first_year

  have h3 : expenditure_first_year = 0.65 * I := by sorry
  have h4 : savings_first_year = 0.35 * I := by sorry

  -- Using given condition to resolve S
  exact h1

end man_savings_l80_80853


namespace sum_series_equals_4_div_9_l80_80214

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80214


namespace series_sum_eq_four_ninths_l80_80244

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80244


namespace find_cos_C_find_circumcircle_radius_l80_80462

variable {a b c : ℝ}
variable {C : ℝ}

-- Given: In triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively.
-- Condition 1: a^2 + b^2 - c^2 = 1/5 * a * b
-- Condition 2: c = 3 * real.sqrt 11
-- To Prove:
-- 1) cos C = 1/10
-- 2) The radius of the circumcircle R = 5

theorem find_cos_C (h : a^2 + b^2 - c^2 = 1 / 5 * a * b) : Real.cos C = 1 / 10 :=
by
  -- Proof will be here
  sorry

theorem find_circumcircle_radius (h_cos : Real.cos C = 1 / 10) (h_c : c = 3 * Real.sqrt 11) (h_sin : Real.sin C = 3 * Real.sqrt 11 / 10) : ∃ R : ℝ, R = 5 :=
by
  -- Proof will be here
  use 5
  sorry

end find_cos_C_find_circumcircle_radius_l80_80462


namespace odd_factors_450_l80_80402

theorem odd_factors_450 : ∃ n: ℕ, n = 9 ∧ (∀ p, prime p ∧ 450 % p = 0 → odd p) := sorry

end odd_factors_450_l80_80402


namespace infinite_series_sum_l80_80225

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80225


namespace sum_of_factors_36_l80_80724

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80724


namespace sum_series_eq_four_ninths_l80_80211

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80211


namespace distance_proof_l80_80482

-- Definitions for given conditions
def skew_lines (l m : Type) : Prop := sorry
def on_line (pt : Type) (l : Type) : Prop := sorry
def equal_distance (pt1 pt2 pt3 : Type) : Prop := sorry
def perpendicular_distance (pt1 pt2 : Type) (dist : ℝ) : Prop := sorry

-- Given conditions
variables {l m A B C D E F G H K : Type}
variable h1 : skew_lines l m
variable h2 : on_line A l
variable h3 : on_line B l
variable h4 : on_line C l
variable h5 : equal_distance A B C
variable h6 : perpendicular_distance A D (real.sqrt 15)
variable h7 : perpendicular_distance B E (7 / 2)
variable h8 : perpendicular_distance C F (real.sqrt 10)

-- Problem statement to prove the distance
noncomputable def distance_between_lines : ℝ := real.sqrt 6

-- Theorem to prove
theorem distance_proof : ∀ {l m : Type} {A B C D E F G H K : Type},
    skew_lines l m →
    on_line A l →
    on_line B l →
    on_line C l →
    equal_distance A B C →
    perpendicular_distance A D (real.sqrt 15) →
    perpendicular_distance B E (7 / 2) →
    perpendicular_distance C F (real.sqrt 10) →
    distance_between_lines = real.sqrt 6 :=
begin
    sorry
end

end distance_proof_l80_80482


namespace inner_automorphism_is_automorphism_l80_80472

variable {G : Type*} [Group G]

def inner_automorphism (x : G) : G → G :=
  λ y, x⁻¹ * y * x

theorem inner_automorphism_is_automorphism (G : Type*) [Group G] (x : G) :
  function.is_automorphism (inner_automorphism x) :=
sorry

end inner_automorphism_is_automorphism_l80_80472


namespace sum_of_factors_36_l80_80731

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80731


namespace max_digit_sum_24_hour_format_l80_80067

theorem max_digit_sum_24_hour_format : ∃ maxSum : ℕ, maxSum = 19 ∧
  (∀ (h m : ℕ), h < 24 → m < 60 →
    let digits_sum := (h / 10) + (h % 10) + (m / 10) + (m % 10) in digits_sum ≤ maxSum) :=
by
  sorry

end max_digit_sum_24_hour_format_l80_80067


namespace sum_of_factors_36_l80_80624

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80624


namespace sum_k_over_4k_l80_80164

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80164


namespace sum_of_factors_36_l80_80623

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80623


namespace range_of_gauss_f_l80_80341

def gauss_function (x : ℝ) := ⌊x⌋

def f (x : ℝ) := 2 ^ (x + 1) / (1 + 2 ^ (2 * x))

theorem range_of_gauss_f : 
  {y : ℤ | ∃ x : ℝ, y = gauss_function (f x)} = {0, 1} := 
by
  sorry

end range_of_gauss_f_l80_80341


namespace find_u_l80_80924

open Real

noncomputable def u_vec : ℝ × ℝ × ℝ :=
  let x := (5 * sqrt 3 - 6 * sqrt 10) / 18
  let z := - sqrt 10 / 2
  (x, 0, z)

lemma u_is_unit_vector : 
  let ⟨x, y, z⟩ := u_vec in y = 0 ∧ x^2 + y^2 + z^2 = 1 :=
by 
  let ⟨x, y, z⟩ := u_vec
  exact sorry

lemma u_angle_with_first_vector :
  let ⟨x, y, z⟩ := u_vec in 
  arccos ((3 * x + 0 * y + -2 * z) / 
    (sqrt (x^2 + y^2 + z^2) * sqrt (3^2 + 1^2 + (-2)^2))) = π / 6 :=
by 
  let ⟨x, y, z⟩ := u_vec
  exact sorry

lemma u_angle_with_second_vector :
  let ⟨x, y, z⟩ := u_vec in 
  arccos ((0 * x + 2 * y + -1 * z) / 
    (sqrt (x^2 + y^2 + z^2) * sqrt (0^2 + 2^2 + (-1)^2))) = π / 4 :=
by 
  let ⟨x, y, z⟩ := u_vec
  exact sorry

theorem find_u :
  ∃ (u : ℝ × ℝ × ℝ), 
    let ⟨x, y, z⟩ := u in 
    y = 0 ∧
    x^2 + z^2 = 1 ∧
    arccos ((3 * x + 0 * y + -2 * z) / (sqrt 10)) = π / 6 ∧
    arccos ((0 * x + 2 * y + -1 * z) / (sqrt 5)) = π / 4 ∧
    u = u_vec :=
by 
  use u_vec
  exact ⟨u_is_unit_vector, u_angle_with_first_vector, u_angle_with_second_vector⟩

end find_u_l80_80924


namespace distribution_of_items_l80_80109

open Finset

theorem distribution_of_items :
  ∃ (count : ℕ), count = 52 ∧
  ∃ (ways : Finset (Multiset (Multiset ℕ))), ways.card = 52 ∧
  (∀ (items : Finset ℕ) (bags : Finset (Finset (Finset ℕ))),
    items.card = 5 →
    bags.card = 4 →
    (∃ way : Multiset (Multiset ℕ), way ∈ ways) →
    bags ⊆ ways.toFinset) :=
begin
  sorry
end

end distribution_of_items_l80_80109


namespace probability_of_line_intersecting_both_squares_l80_80470

open MeasureTheory

theorem probability_of_line_intersecting_both_squares :
  let R := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1} in
  let P := uniformMeasure R in
  ∫ p in R, indicator (set_of (λ p : ℝ × ℝ, ∃ b : ℝ, p.2 = 1 / 2 * p.1 + b ∧ -1 / 2 ≤ b ∧ b ≤ 1 / 2 ∧ 0 ≤ 1 / 2 + b ∧ 1 / 2 + b ≤ 1)) p ∂P = 3 / 4 := sorry

end probability_of_line_intersecting_both_squares_l80_80470


namespace find_k_l80_80398

open Real

theorem find_k
  (a : ℝ × ℝ := (3, 1))
  (b : ℝ × ℝ := (1, 3))
  (c : ℝ × ℝ)
  (k : ℝ := 5) :
  (a - c = (3 - k, -6) → ∃ (λ : ℝ), a - c = λ • b) → c = (k, 7) → k = 5 := by
  sorry

end find_k_l80_80398


namespace sum_factors_36_l80_80630

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80630


namespace sum_of_factors_36_l80_80733

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80733


namespace sum_factors_36_l80_80759

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80759


namespace solve_wire_cut_problem_l80_80042

def wire_cut_problem : Prop :=
  ∃ x y : ℝ, x + y = 35 ∧ y = (2/5) * x ∧ x = 25

theorem solve_wire_cut_problem : wire_cut_problem := by
  sorry

end solve_wire_cut_problem_l80_80042


namespace sum_of_factors_36_l80_80703

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80703


namespace sum_factors_36_eq_91_l80_80605

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80605


namespace sum_of_factors_of_36_l80_80788

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80788


namespace geom_seq_log_sum_l80_80990

theorem geom_seq_log_sum :
  (∀ n : ℕ, (∑ i in Finset.range n, (2:ℝ)^i) = (2:ℝ)^n - 1) →
  (∑ i in Finset.range 12, Real.logb 2 (2:ℝ)^i) = 66 :=
by
  intro geometric_sum
  sorry

end geom_seq_log_sum_l80_80990


namespace base9_product_correct_l80_80121

theorem base9_product_correct :
  ∀ (n1 n2 : ℕ), n1 = 3 * 9^2 + 4 * 9^1 + 7 * 9^0 → n2 = 6 * 9^0 → 
  let product := 286 * 6 in 
  let base9_product := 2 * 9^3 + 3 * 9^2 + 1 * 9^1 + 6 * 9^0 in 
  (product = 1716) ∧ (base9_product = 2316) :=
by 
  intros n1 n2 hn1 hn2
  let product := 286 * 6
  have h₁ : product = 1716 := by rfl
  let base9_product := 2 * 9^3 + 3 * 9^2 + 1 * 9^1 + 6 * 9^0
  have h₂ : base9_product = 2 * 9^3 + 3 * 9^2 + 1 * 9^1 + 6 * 9^0 := 
    by rfl
  exact ⟨h₁, h₂⟩

end base9_product_correct_l80_80121


namespace sine_angle_BAC_l80_80370

variables {A B C O : Type} [InnerProductSpace ℝ V] [MetricSpace A] [NormedAddTorsor V A]

-- We define points A, B, C, and O and the condition that O is the circumcenter of ABC
variables (A B C O : A)
-- Define vectors for the points A, B, and C
variables (a b c o : V)
variables (hA : A = a) (hB : B = b) (hC : C = c) (hO : O = o)

-- Define the vector condition for the circumcenter
variables (hVec : o - a = (b - a) + 2 * (c - a))

-- Prove that the sine of angle BAC equals sqrt(10)/4
theorem sine_angle_BAC : sin (angle (b - a) (c - a)) = sqrt 10 / 4 :=
sorry

end sine_angle_BAC_l80_80370


namespace series_sum_eq_four_ninths_l80_80242

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80242


namespace find_n_l80_80934

theorem find_n (n : ℤ) (h1 : -180 ≤ n ∧ n ≤ 180) (h2 : real.sin (n * real.pi / 180) = real.cos (810 * real.pi / 180)) : n ∈ {-180, 0, 180} :=
sorry

end find_n_l80_80934


namespace brothers_save_time_l80_80566

-- Define the conditions
variables (v : ℕ) -- Walking speed in minutes per kilometer
def v_b := v - 12 -- Bicycling speed

-- Define the total walking and biking times
def t_walk := 10 * v
def t_bike := 5 * v + 5 * v_b

-- Define the time saved
def t_saved := t_walk - t_bike

-- State the theorem
theorem brothers_save_time (v : ℕ) (hv : v ≥ 12) : t_saved v = 60 :=
by
  -- Placeholder for the proof
  sorry

end brothers_save_time_l80_80566


namespace g_50_l80_80537

noncomputable def g : ℝ → ℝ :=
sorry

axiom functional_equation (x y : ℝ) : g (x * y) = x * g y
axiom g_2 : g 2 = 10

theorem g_50 : g 50 = 250 :=
sorry

end g_50_l80_80537


namespace sum_infinite_series_l80_80191

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80191


namespace product_2017_l80_80555

def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 3 else
    let p := sequence_a (n - 1)
    (-p + 1) / (p - 1)

def product_sequence_A (n : ℕ) : ℚ :=
  (List.range n).map (λ i => sequence_a (i + 1)).prod

theorem product_2017 : product_sequence_A 2017 = 3 :=
  sorry

end product_2017_l80_80555


namespace number_of_regions_in_convex_polygon_l80_80354

theorem number_of_regions_in_convex_polygon (n : ℕ) (h : n ≥ 3) :
  (∑ i in finset.range (n - 3 - 1), i) + 2 = (n - 3) * (n - 4) / 2 + 2 := by
  sorry

end number_of_regions_in_convex_polygon_l80_80354


namespace sum_factors_36_l80_80632

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80632


namespace snail_return_origin_int_hours_l80_80863

theorem snail_return_origin_int_hours (c : constant_speed) (p : starts_at_origin) (t : turns_every_half_hour (60 * degree)) : 
  ∃ n : ℤ, returns_to_origin n :=
sorry

end snail_return_origin_int_hours_l80_80863


namespace sum_of_factors_of_36_l80_80662

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80662


namespace scientific_notation_of_15510000_l80_80874

/--
Express 15,510,000 in scientific notation.

Theorem: 
Given that the scientific notation for large numbers is of the form \(a \times 10^n\) where \(1 \leq |a| < 10\),
prove that expressing 15,510,000 in scientific notation results in 1.551 × 10^7.
-/
theorem scientific_notation_of_15510000 : ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 15510000 = a * 10 ^ n ∧ a = 1.551 ∧ n = 7 :=
by
  sorry

end scientific_notation_of_15510000_l80_80874


namespace quadratic_inequality_m_range_l80_80386

theorem quadratic_inequality_m_range (m : ℝ) : (∀ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ (m ≠ 0) :=
by
  sorry

end quadratic_inequality_m_range_l80_80386


namespace min_value_of_x_plus_y_l80_80970

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)  
  (h : 19 / x + 98 / y = 1) : x + y ≥ 203 :=
sorry

end min_value_of_x_plus_y_l80_80970


namespace Amy_work_hours_l80_80886

theorem Amy_work_hours (summer_weeks: ℕ) (summer_hours_per_week: ℕ) (summer_total_earnings: ℕ)
                       (school_weeks: ℕ) (school_total_earnings: ℕ) (hourly_wage: ℕ) 
                       (school_hours_per_week: ℕ):
    summer_weeks = 8 →
    summer_hours_per_week = 40 →
    summer_total_earnings = 3200 →
    school_weeks = 32 →
    school_total_earnings = 4800 →
    hourly_wage = summer_total_earnings / (summer_weeks * summer_hours_per_week) →
    school_hours_per_week = school_total_earnings / (hourly_wage * school_weeks) →
    school_hours_per_week = 15 :=
by
  intros
  sorry

end Amy_work_hours_l80_80886


namespace sum_series_eq_4_div_9_l80_80296

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80296


namespace sum_factors_36_l80_80802

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80802


namespace find_value_of_f_f_1_l80_80379

def f (x : ℝ) : ℝ :=
  if x > 1 then real.log2 x else x^2 + 1

theorem find_value_of_f_f_1 : f (f 1) = 1 := 
by 
  sorry

end find_value_of_f_f_1_l80_80379


namespace infinite_series_sum_l80_80264

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80264


namespace sum_of_factors_of_36_l80_80669

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80669


namespace sum_of_series_l80_80165

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80165


namespace sum_of_factors_36_l80_80622

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80622


namespace max_levels_up_to_3_levels_l80_80435

noncomputable def dice_faces := [1, 2, 3, 4, 5, 6]

def dice_roll_sum (n : ℕ) := list.sum (list.replicate n (6 : ℕ))

def pass_level (n : ℕ) := dice_roll_sum n > 2^n

theorem max_levels (n : ℕ) : n ≤ 4 := by
  assume h : n > 4
  have h1 : dice_roll_sum 5 = 5 * 6 := by sorry
  have h2 : 2^5 = 32 := by sorry
  have h3 : dice_roll_sum 5 < 2^5 := by
    rw [h1, h2]
    exact dec_trivial
  exact absurd h3 (pass_level 5)

theorem up_to_3_levels : (2 / 3) * (7 / 12) * (7 / 27) = 49 / 486 := by sorry

end max_levels_up_to_3_levels_l80_80435


namespace sum_of_factors_36_l80_80805

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80805


namespace common_tangents_l80_80451

noncomputable def C1 : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 + 4*p.2 - 4 = 0}

noncomputable def C2 : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 2*p.2 - 2 = 0}

theorem common_tangents (C1 C2 : set (ℝ × ℝ)) :
  (C1 = {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 + 4*p.2 - 4 = 0}) →
  (C2 = {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 2*p.2 - 2 = 0}) →
  (2 : ℕ) :=
by
  intros hC1 hC2
  sorry

end common_tangents_l80_80451


namespace sin_690_eq_neg_0_5_l80_80821

theorem sin_690_eq_neg_0_5 : Real.sin (690 * Real.pi / 180) = -0.5 := by
  sorry

end sin_690_eq_neg_0_5_l80_80821


namespace geometric_sequence_seventh_term_l80_80847

-- Define the conditions of the problem
def geometric_sequence (a r : ℕ) (n : ℕ) : ℕ := a * r^n

theorem geometric_sequence_seventh_term :
  (a r : ℕ) (a1_eq_3 : a = 3) (a6_eq_972 : geometric_sequence a r 5 = 972) :
  geometric_sequence a r 6 = 2187 :=
by
  -- The proof is to be filled in
  sorry

end geometric_sequence_seventh_term_l80_80847


namespace part_I_part_II_l80_80987

-- Definitions for given vectors
def OA := (2 : ℝ, 5 : ℝ)
def OB := (3 : ℝ, 1 : ℝ)
def OC (x : ℝ) := (x, 3)

-- Definition of vectors A, B, C
def AB := (OB.1 - OA.1, OB.2 - OA.2)  -- (3 - 2, 1 - 5) = (1, -4)
def BC (x : ℝ) := (OC x).1 - OB.1, (OC x).2 - OB.2  -- (x - 3, 3 - 1) = (x - 3, 2)

theorem part_I (x : ℝ) : 
  (x = 5 / 2 ∨ x ≠ 5 / 2) :=
begin
  sorry,
end

-- Definition of vector OM based on lambda
def OM (λ : ℝ) := (6 * λ, 3 * λ)

-- Definitions for questions in part II
def MA (λ : ℝ) := (OM λ).1 - OA.1, (OM λ).2 - OA.2
def MB (λ : ℝ) := (OM λ).1 - OB.1, (OM λ).2 - OB.2

-- The orthogonality condition
def ortho_condition (λ : ℝ) := 
  (MA λ).1 * (MB λ).1 + (MA λ).2 * (MB λ).2 = 0

theorem part_II (λ : ℝ) : 
  ortho_condition λ → (OM λ = (2, 1) ∨ OM λ = (22 / 5, 11 / 5)) :=
begin
  sorry,
end

end part_I_part_II_l80_80987


namespace percentage_profit_l80_80430

theorem percentage_profit 
  (C S : ℝ) 
  (h : 29 * C = 24 * S) : 
  ((S - C) / C) * 100 = 20.83 := 
by
  sorry

end percentage_profit_l80_80430


namespace sum_of_factors_36_eq_91_l80_80588

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80588


namespace ratio_AD_over_AB_l80_80456

-- Define the context and necessary variables
variables (A B C D E : Type) [linear_ordered_field ℝ] [has_scalar ℝ ℝ]
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables (AB AC AD DE : ℝ)
axiom angle_A : real.pi / 3 -- 60 degrees
axiom angle_B : real.pi / 4 -- 45 degrees
axiom angle_ADE : real.pi / 4 -- 45 degrees
axiom line_DE_divides_triangle_ABC_into_equal_areas : DE ∈ segment B C

-- The target statement to prove
theorem ratio_AD_over_AB (A B C D E : Type) 
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  [linear_ordered_field ℝ] [has_scalar ℝ ℝ]
  (angle_A : real.pi / 3) (angle_B : real.pi / 4) (angle_ADE : real.pi / 4) (DE_on_segment_AC : ℝ)
  (line_DE_divides_triangle_ABC_into_equal_areas : ∃ DE, DE ∈ segment B C) : 
  AD / AB = 1 / real.sqrt 3 :=
sorry

end ratio_AD_over_AB_l80_80456


namespace no_real_solution_abs_eq_quadratic_l80_80928

theorem no_real_solution_abs_eq_quadratic (x : ℝ) : abs (2 * x - 6) ≠ x^2 - x + 2 := by
  sorry

end no_real_solution_abs_eq_quadratic_l80_80928


namespace ellipse_equation_circumscribed_circle_equation_l80_80362

def ellipse_eq_foci (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  let a := 1
  let b := 1
  let c := 1
  (F1 = (-1, 0)) ∧ (F2 = (1, 0)) ∧ (dist (1, sqrt(2) / 2)) = a + b ∧
  ∀ (x y: ℝ), x^2 / (b^2 + 1) + y^2 / b^2 = 1

theorem ellipse_equation :
  ellipse_eq_foci (-1, 0) (1, 0) (1, sqrt 2 / 2) :=
sorry

def circumscribed_circle_eq (P A B C D : ℝ × ℝ) : Prop :=
  let m := 2
  let r := sqrt 10 / 3
  let center := (-1 / 3, 0)
  let circle_eq (x y : ℝ) := (x + 1 / 3)^2 + y^2 = r^2
  P = (-2, 0) ∧
  ∃ m : ℝ, m² > 2 ∧ l = (x * m - 2) ∧
  ∃ A B : ℝ × ℝ, (intersects (A, B, l, E)) ∧ (P B = 3 P A) ∧
  B = (0, 1) ∧ symmetric_to_x_axis A B ∧
  quadrilateral ACDB_center center ∧ circle_eq

theorem circumscribed_circle_equation :
  circumscribed_circle_eq (-2, 0) (⟨_, _⟩) (⟨_, _⟩) (⟨_, _⟩) (⟨_, _⟩) :=
sorry

end ellipse_equation_circumscribed_circle_equation_l80_80362


namespace shaded_region_area_correct_l80_80454

noncomputable def area_shaded_region : ℝ := 
  let side_length := 2
  let radius := 1
  let area_square := side_length^2
  let area_circle := Real.pi * radius^2
  area_square - area_circle

theorem shaded_region_area_correct : area_shaded_region = 4 - Real.pi :=
  by
    sorry

end shaded_region_area_correct_l80_80454


namespace sum_infinite_series_l80_80198

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80198


namespace g_has_extreme_at_0_h_has_extreme_at_0_l80_80882

def f (x : ℝ) : ℝ := x^3
def g (x : ℝ) : ℝ := x^2 + 1
def h (x : ℝ) : ℝ := abs x
def k (x : ℝ) : ℝ := 2^x

theorem g_has_extreme_at_0 : ∃ c, g 0 = c ∧ (∀ x, g x ≥ c) ∨ (∀ x, g x ≤ c) := sorry

theorem h_has_extreme_at_0 : ∃ c, h 0 = c ∧ (∀ x, h x ≥ c) ∨ (∀ x, h x ≤ c) := sorry

end g_has_extreme_at_0_h_has_extreme_at_0_l80_80882


namespace smallest_fraction_numerator_l80_80879

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (a * 4 > b * 3) ∧ (a = 73) := 
sorry

end smallest_fraction_numerator_l80_80879


namespace PG_parallel_AE_l80_80892

noncomputable def Point := (ℝ × ℝ)

structure Square :=
(A B C D : Point)
(is_square : ∃ s, A = (0, s) ∧ B = (s, s) ∧ C = (s, 0) ∧ D = (0, 0))

structure IsoscelesTriangle :=
(AB E : Point)
(base_angle : Real.angle)
(is_isosceles : E = (3/2, (2 + Real.sqrt 3) / 2))

structure ProblemData :=
(square : Square)
(iso_triangle : IsoscelesTriangle)
(M : Point)
(N : Point)
(P : Point)
(G : Point)
(M_condition : M = (field.tangent base_angle / 2 + (2 + Real.sqrt 3) / 2, 1))
(N_condition : N = (field.tangent base_angle / 2 + (2 + Real.sqrt 3) / 2, 1))
(P_condition : some_condition_here)
(G_condition : G = ((0 + 1 + 3/2) / 3, (1 + 1 + (2 + Real.sqrt 3)/2) / 3))

theorem PG_parallel_AE : 
    ∀ (data : ProblemData), 
    let G := data.G in 
    let P := data.P in
    let E := data.iso_triangle.E in 
    let A := data.square.A in 
    let slope_AE := (E.2 - A.2) / (E.1 - A.1) in
    let slope_PG := (G.2 - P.2) / (G.1 - P.1) in
    slope_AE = slope_PG :=
by sorry

end PG_parallel_AE_l80_80892


namespace non_athletic_parents_l80_80562

-- Define the conditions
variables (total_students athletic_dads athletic_moms both_athletic : ℕ)

-- Assume the given conditions
axiom h1 : total_students = 45
axiom h2 : athletic_dads = 17
axiom h3 : athletic_moms = 20
axiom h4 : both_athletic = 11

-- Statement to be proven
theorem non_athletic_parents : total_students - (athletic_dads - both_athletic + athletic_moms - both_athletic + both_athletic) = 19 :=
by {
  -- We intentionally skip the proof here
  sorry
}

end non_athletic_parents_l80_80562


namespace PC_eq_PD_l80_80871

def quadrilateral : Type := 
  {A B C D : ℝ × ℝ} 
  (convex : convex_hull ℝ (finset.cons A (finset.cons B (finset.cons C (finset.singleton D)))) = 
             finset.cons A (finset.cons B (finset.cons C (finset.singleton D))))

variables  {A B C D P : ℝ × ℝ}
variables  (hBAC: angle A B C = 30)
           (hCAD: angle C A D = 20)
           (hABD: angle A B D = 50)
           (hDBC: angle D B C = 30)
           (Pexists : (exists P, P = line_intersection (line A C) (line B D)))

theorem PC_eq_PD : dist P C = dist P D :=
by sorry

end PC_eq_PD_l80_80871


namespace Amy_work_hours_l80_80887

theorem Amy_work_hours (summer_weeks: ℕ) (summer_hours_per_week: ℕ) (summer_total_earnings: ℕ)
                       (school_weeks: ℕ) (school_total_earnings: ℕ) (hourly_wage: ℕ) 
                       (school_hours_per_week: ℕ):
    summer_weeks = 8 →
    summer_hours_per_week = 40 →
    summer_total_earnings = 3200 →
    school_weeks = 32 →
    school_total_earnings = 4800 →
    hourly_wage = summer_total_earnings / (summer_weeks * summer_hours_per_week) →
    school_hours_per_week = school_total_earnings / (hourly_wage * school_weeks) →
    school_hours_per_week = 15 :=
by
  intros
  sorry

end Amy_work_hours_l80_80887


namespace series_sum_eq_four_ninths_l80_80248

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80248


namespace distributing_items_into_bags_l80_80893

theorem distributing_items_into_bags :
  ∃ (n : ℕ), n = 14 ∧
    ∑ x in (finset.powerset (finset.range 4)), 
    if x.card ≤ 3 then 1 else 0 = 14 := by
  sorry

end distributing_items_into_bags_l80_80893


namespace convert_time_l80_80408

-- Definitions of conditions
def minutes_to_seconds (minutes : ℝ) : ℝ :=
  minutes * 60

def minutes_to_hours (minutes : ℝ) : ℝ :=
  minutes / 60

-- Given 12.5 minutes, prove the equivalent seconds and hours
theorem convert_time (h : 12.5 = 12.5) :
  minutes_to_seconds 12.5 = 750 ∧ minutes_to_hours 12.5 = 0.20833 :=
by
  sorry

end convert_time_l80_80408


namespace infinite_series_sum_l80_80261

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80261


namespace common_sum_of_cube_faces_l80_80548

theorem common_sum_of_cube_faces (A B C D E F G H : ℕ) (h : {A, B, C, D, E, F, G, H} ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  (4 * (A + B + C + D + E + F + G + H) / 3 = 18) :=
by {
  have h1 : A + B + C + D + E + F + G + H = 36, {
    -- sum of 1 to 8
    sorry
  },
  have h2 : 4 * (A + B + C + D + E + F + G + H) = 3 * 18, {
    -- use A + B + C + D + E + F + G + H = 36
    sorry
  },
  have h3 : 4 * (36 / 3) = 18, {
    -- dividing by 3 to get the common sum
    sorry
  },
  exact h3,
}

end common_sum_of_cube_faces_l80_80548


namespace no_sequence_for_k_eq_5_l80_80465

-- Defining the problem statement
theorem no_sequence_for_k_eq_5 :
  ¬(∃ n : Fin 6 → ℕ, ∀ i : Fin 6, (n i = i) ∧ (∑ i : Fin 6, n i = 6)) :=
sorry

end no_sequence_for_k_eq_5_l80_80465


namespace select_82_numbers_ensure_pair_sum_multiple_of_5_l80_80340

theorem select_82_numbers_ensure_pair_sum_multiple_of_5 :
  ∀ (S : Finset ℕ), (∀ x ∈ S, x ≤ 200) → S.card = 82 →
  ∃ (a b ∈ S), a ≠ b ∧ (a + b) % 5 = 0 :=
by
  intro S h1 h2
  sorry

end select_82_numbers_ensure_pair_sum_multiple_of_5_l80_80340


namespace expected_difference_after_10_days_l80_80026

-- Define the initial state and transitions
noncomputable def initial_prob (k : ℤ) : ℝ :=
if k = 0 then 1 else 0

noncomputable def transition_prob (k : ℤ) (n : ℕ) : ℝ :=
0.5 * initial_prob k +
0.25 * initial_prob (k - 1) +
0.25 * initial_prob (k + 1)

-- Define event probability for having any wealth difference after n days
noncomputable def p_k_n (k : ℤ) (n : ℕ) : ℝ :=
if n = 0 then initial_prob k
else transition_prob k (n - 1)

-- Use expected value of absolute difference between wealths 
noncomputable def expected_value_abs_diff (n : ℕ) : ℝ :=
Σ' k, |k| * p_k_n k n

-- Finally, state the theorem
theorem expected_difference_after_10_days :
expected_value_abs_diff 10 = 1 :=
by
  sorry

end expected_difference_after_10_days_l80_80026


namespace sum_of_factors_36_l80_80807

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80807


namespace expected_value_absolute_difference_after_10_days_l80_80040

/-- Define the probability space and outcome -/
noncomputable def probability_cat_wins : ℝ := 0.25
noncomputable def probability_fox_wins : ℝ := 0.25
noncomputable def probability_both_police : ℝ := 0.50

/-- Define the random variable for absolute difference in wealth -/
noncomputable def X_n (n : ℕ) : ℝ := sorry

/-- Define the probability p_{0, n} -/
noncomputable def p (k n : ℕ) : ℝ := sorry

/-- Given the above conditions, the expected value of the absolute difference -/
theorem expected_value_absolute_difference_after_10_days : (∑ k in finset.range 11, k * p k 10) = 1 :=
sorry

end expected_value_absolute_difference_after_10_days_l80_80040


namespace sum_of_factors_36_eq_91_l80_80742

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80742


namespace sum_of_factors_36_l80_80820

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80820


namespace sum_of_factors_36_l80_80700

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80700


namespace sum_of_factors_36_l80_80628

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80628


namespace infinite_series_sum_l80_80234

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80234


namespace power_mod_l80_80322

theorem power_mod (a : ℕ) : 5 ^ 2023 % 17 = 2 := by
  sorry

end power_mod_l80_80322


namespace thabo_hardcover_nonfiction_books_l80_80521

constant H : ℕ -- Number of hardcover nonfiction books
constant P : ℕ -- Number of paperback nonfiction books
constant F : ℕ -- Number of paperback fiction books

axiom condition1 : P = H + 30
axiom condition2 : F = 3 * P
axiom condition3 : H + P + F = 500

theorem thabo_hardcover_nonfiction_books : H = 76 :=
by
  sorry

end thabo_hardcover_nonfiction_books_l80_80521


namespace find_angle_θ_l80_80371

-- Define the conditions
def sin_3pi_over_4 : ℝ := Real.sin (3 * Real.pi / 4)
def cos_3pi_over_4 : ℝ := Real.cos (3 * Real.pi / 4)
def point_on_terminal_side (θ : ℝ) : Prop := θ ∈ Set.Ico 0 (2 * Real.pi) ∧ P = (Real.sin θ, Real.cos θ) where
  P := (sin_3pi_over_4, cos_3pi_over_4)

-- Statement of the proof problem
theorem find_angle_θ (θ : ℝ) (h : point_on_terminal_side θ) : θ = 7 * Real.pi / 4 := sorry

end find_angle_θ_l80_80371


namespace sqrt_difference_l80_80011

noncomputable def sqrt49plus49 : ℝ := real.sqrt (49 + 49)
noncomputable def sqrt36plus25 : ℝ := real.sqrt (36 + 25)

theorem sqrt_difference :
  sqrt49plus49 - sqrt36plus25 = 7 * real.sqrt 2 - real.sqrt 61 :=
by
  sorry

end sqrt_difference_l80_80011


namespace infinite_series_sum_l80_80229

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80229


namespace intersection_complement_l80_80413

-- Definitions and conditions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 3}
def B : Set ℕ := {1, 3, 4}
def C_U (B : Set ℕ) : Set ℕ := {x ∈ U | x ∉ B}

-- Theorem statement
theorem intersection_complement :
  (C_U B) ∩ A = {0, 2} := 
by
  -- Proof is not required, so we use sorry
  sorry

end intersection_complement_l80_80413


namespace sin_double_theta_l80_80420

-- Given condition
def given_condition (θ : ℝ) : Prop :=
  Real.cos (Real.pi / 4 - θ) = 1 / 2

-- The statement we want to prove: sin(2θ) = -1/2
theorem sin_double_theta (θ : ℝ) (h : given_condition θ) : Real.sin (2 * θ) = -1 / 2 :=
sorry

end sin_double_theta_l80_80420


namespace sum_of_factors_36_eq_91_l80_80584

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80584


namespace infinite_series_sum_l80_80262

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80262


namespace min_containers_needed_l80_80065

def container_capacity : ℕ := 500
def required_tea : ℕ := 5000

theorem min_containers_needed (n : ℕ) : n * container_capacity ≥ required_tea → n = 10 :=
sorry

end min_containers_needed_l80_80065


namespace circle_equation_l80_80558

def circle_center := (-5 : ℝ, 4 : ℝ)
def radius := 4

theorem circle_equation (x y : ℝ) :
  ((x + 5)^2 + (y - 4)^2 = 16) :=
sorry

end circle_equation_l80_80558


namespace anna_more_heads_than_tails_l80_80889

noncomputable def probability_more_heads_than_tails (n : ℕ) (p_head : ℚ) (p_tail : ℚ) : ℚ :=
∑ k in finset.range(n+1), if k > n/2 then nat.choose n k * (p_head^k) * (p_tail^(n-k)) else 0

theorem anna_more_heads_than_tails : probability_more_heads_than_tails 9 (2 / 5) (3 / 5) = 0.267 := 
sorry

end anna_more_heads_than_tails_l80_80889


namespace initial_percentage_increase_l80_80862

theorem initial_percentage_increase (x : ℝ) :
  let final_price_gain_percent := 4.040000000000006 in
  let successive_discounts_prod := 0.9 * 0.85 in
  ((1 + x / 100) * successive_discounts_prod = 1 + final_price_gain_percent / 100) →
  x = 35.95 :=
sorry

end initial_percentage_increase_l80_80862


namespace sum_k_over_4k_l80_80160

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80160


namespace sum_frac_zeta_eq_3_div_4_l80_80134

noncomputable def zeta (x : ℝ) : ℝ := ∑' n, 1 / (n : ℝ) ^ x

-- Define the fractional part function
def frac_part (x : ℝ) : ℝ := x - x.floor

theorem sum_frac_zeta_eq_3_div_4 :
  (∑' k, frac_part (zeta (2 * k))) = 3 / 4 :=
sorry

end sum_frac_zeta_eq_3_div_4_l80_80134


namespace cos_angle_is_correct_k_values_for_perpendicular_l80_80960

def vector_a : ℝ × ℝ × ℝ := (3, 2, -1)
def vector_b : ℝ × ℝ × ℝ := (2, 1, 2)

def cos_angle_between (a b : ℝ × ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) / ((Real.sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)) * (Real.sqrt (b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2)))

theorem cos_angle_is_correct :
  cos_angle_between vector_a vector_b = 2 / Real.sqrt 14 :=
by
  sorry

theorem k_values_for_perpendicular :
  ∃ k : ℝ, (k = 3 / 2 ∨ k = -2 / 3) ∧ (let ka_plus_b := (k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2, k * vector_a.3 + vector_b.3) in
  let a_minus_kb := (vector_a.1 - k * vector_b.1, vector_a.2 - k * vector_b.2, vector_a.3 - k * vector_b.3) in
  (ka_plus_b.1 * a_minus_kb.1 + ka_plus_b.2 * a_minus_kb.2 + ka_plus_b.3 * a_minus_kb.3 = 0)) :=
by
  sorry

end cos_angle_is_correct_k_values_for_perpendicular_l80_80960


namespace sum_infinite_series_l80_80189

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80189


namespace median_and_mode_l80_80861

def data_set : Set Int := {2, -1, 0, 3, -3, 2}

def sorted_data_set : List Int := [-3, -1, 0, 2, 2, 3]

def median (s : List Int) : ℚ :=
  let n := s.length
  if n % 2 = 0 then
    (s.get! (n / 2 - 1) + s.get! (n / 2)) / 2
  else s.get! (n / 2)

def mode (s : List Int) : Int :=
  s.maxBy (λ x => s.filter (· = x)).length

theorem median_and_mode :
  median sorted_data_set = 1 ∧ mode sorted_data_set = 2 := by
  sorry

end median_and_mode_l80_80861


namespace sum_of_factors_36_eq_91_l80_80589

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80589


namespace square_area_is_25_l80_80496

-- Declare two points on the Cartesian coordinate system
def P1 : ℝ × ℝ := (1, 2)
def P2 : ℝ × ℝ := (5, 5)

-- Define the distance formula between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the side length of the square as the distance between P1 and P2
def side_length : ℝ := distance P1 P2

-- Define the area of the square as the side length squared
def area (s : ℝ) : ℝ := s^2

-- Theorem statement: the area of the square is 25 given the points P1 and P2 are adjacent points
theorem square_area_is_25 : area side_length = 25 := 
sorry

end square_area_is_25_l80_80496


namespace cos_420_degrees_l80_80332

-- Define the problem conditions in Lean 4
def cos_periodic (θ : ℝ) : Prop := cos θ = cos (θ + 360)

def cos_special (θ : ℝ) (value : ℝ) : Prop := cos θ = value

-- State the main problem given the conditions
theorem cos_420_degrees : cos_periodic 420 ∧ cos_special 60 (1 / 2) → cos 420 = 1 / 2 :=
by
  sorry

end cos_420_degrees_l80_80332


namespace correct_option_is_C_l80_80104

-- Define the conditions
def coefficient_of_x_is_1 : Prop := (coe 1 : ℤ)
def degree_of_x_is_1 : Prop := (degree (X : polynomial ℤ) = 1)
def negative_2010_is_monomial : Prop := is_monomial (-2010 : polynomial ℤ)
def coefficient_matrix_is_not_negative_7 : Prop := (coeff (monomial 1 7 : polynomial ℤ) 1 ≠ -7)

-- The main theorem, stating that option C is correct
theorem correct_option_is_C (h₁ : coefficient_of_x_is_1)
                           (h₂ : degree_of_x_is_1)
                           (h₃ : negative_2010_is_monomial)
                           (h₄ : coefficient_matrix_is_not_negative_7) : 
                            option_C_is_correct := h₃ Sorry.

end correct_option_is_C_l80_80104


namespace find_t_l80_80959

-- Definitions of the vectors involved
def vector_AB : ℝ × ℝ := (2, 3)
def vector_AC (t : ℝ) : ℝ × ℝ := (3, t)
def vector_BC (t : ℝ) : ℝ × ℝ := ((vector_AC t).1 - (vector_AB).1, (vector_AC t).2 - (vector_AB).2)

-- Condition for orthogonality
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Main statement to be proved
theorem find_t : ∃ t : ℝ, is_perpendicular vector_AB (vector_BC t) ∧ t = 7 / 3 :=
by
  sorry

end find_t_l80_80959


namespace probability_exceeding_90_points_expected_value_E_X_expected_value_E_Y_compare_E_X_E_Y_l80_80569

-- Definition of the score data for students A and B
def scores_A : List ℕ := [80, 78, 82, 86, 95, 93]
def scores_B : List ℕ := [76, 81, 80, 85, 89, 96, 94]

-- Condition for a score to be considered excellent
def is_excellent (score : ℕ) : Bool := score >= 85

-- Part 1: Proving the probability of selecting a test score exceeding 90 points
theorem probability_exceeding_90_points : 
  let scores := scores_A ++ scores_B
  let exceeding_90 := scores.filter (λ s => s > 90)
  (exceeding_90.length : ℚ) / (scores.length : ℚ) = 4 / 13 := sorry

-- Part 2: Expected value E(X) for number of excellent scores in 4 tests from A's 6 tests
noncomputable def choose_4_from_A := (scores_A.combinations 4).map (λ l => l.count is_excellent)
noncomputable def E_X := 
  (choose_4_from_A.sum : ℚ) / (choose_4_from_A.length : ℚ)

theorem expected_value_E_X : E_X = 2 := sorry

-- Part 3: Expected value E(Y) for number of excellent scores in 3 tests from B's 7 tests
noncomputable def choose_3_from_B := (scores_B.combinations 3).map (λ l => l.count is_excellent)
noncomputable def E_Y := 
  (choose_3_from_B.sum : ℚ) / (choose_3_from_B.length : ℚ)

theorem expected_value_E_Y : E_Y = 12 / 7 := sorry

-- Part 4: Comparing E(X) and E(Y)
theorem compare_E_X_E_Y : E_X > E_Y := sorry

end probability_exceeding_90_points_expected_value_E_X_expected_value_E_Y_compare_E_X_E_Y_l80_80569


namespace sum_series_eq_4_div_9_l80_80187

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80187


namespace prove_a_eq_zero_l80_80836

open Nat Int

theorem prove_a_eq_zero (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 := 
begin
  sorry
end

end prove_a_eq_zero_l80_80836


namespace part1_part2_l80_80433

-- Define the necessary conditions and prove the first question
theorem part1 (a b c : ℝ) (A B C : ℝ) (h1 : c * Real.cos B = (2 * a - b) * Real.cos C) : 
  C = Real.pi / 3 := 
sorry

-- Define the necessary conditions and prove the second question
theorem part2 (a b : ℝ) (C : ℝ) (h1: C = Real.pi / 3) (h2 : a * b = 16) : 
  let S := (Real.sqrt 3 / 4) * a * b in 
  S = 4 * Real.sqrt 3 := 
sorry

end part1_part2_l80_80433


namespace compare_numbers_l80_80132

def convert_base_6_to_base_10 (n : ℕ) : ℕ :=
  (4 * 6^2) + (0 * 6^1) + (3 * 6^0) -- 403 in base 6 == 147 in base 10

def convert_base_8_to_base_10 (n : ℕ) : ℕ :=
  (2 * 8^2) + (1 * 8^1) + (7 * 8^0) -- 217 in base 8 == 143 in base 10

theorem compare_numbers : convert_base_6_to_base_10 403 > convert_base_8_to_base_10 217 := by
  have h1 : convert_base_6_to_base_10 403 = 147 := by refl
  have h2 : convert_base_8_to_base_10 217 = 143 := by refl
  show 147 > 143
  sorry

end compare_numbers_l80_80132


namespace area_of_region_l80_80306

noncomputable def tangent_line (x : ℝ) : ℝ := 
  4 * x - 4

theorem area_of_region :
  let curve := λ x : ℝ, x ^ 2 in
  let region_area := (∫ x in 0..1, curve x) + (∫ x in 1..2, curve x - tangent_line x) in
  region_area = (2 / 3 : ℝ) :=
by
  sorry

end area_of_region_l80_80306


namespace max_value_of_S_max_value_of_S_achieved_l80_80826

theorem max_value_of_S (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (min x (min (1 / y) (y + 1 / x))) ≤ sqrt 2 :=
sorry

theorem max_value_of_S_achieved (x y : ℝ) (hx : x = sqrt 2) (hy : y = 1 / sqrt 2) : 
  (min x (min (1 / y) (y + 1 / x))) = sqrt 2 :=
sorry

end max_value_of_S_max_value_of_S_achieved_l80_80826


namespace positive_diff_A_l80_80136

def A' : ℕ := ∑ i in (Finset.range 22), (2*i + 1) * (2*i + 2) + 1 * 2 + 43
def B' : ℕ := ∑ i in (Finset.range 21), (2*i + 1) * (2*i + 2) + 43

theorem positive_diff_A'_B' : |A' - B'| = 925 := by
  sorry

end positive_diff_A_l80_80136


namespace hyperbola_lambda_range_l80_80996

open Real

theorem hyperbola_lambda_range (λ : ℝ) (h₁ : 0 < λ) (h₂ : λ < 1) (h₃ : 1 < 1 / sqrt λ) (h₄ : 1 / sqrt λ < 2) :
  λ ∈ Ioo (1 / 4) 1 :=
by
  sorry

end hyperbola_lambda_range_l80_80996


namespace remainder_of_power_mod_l80_80329

theorem remainder_of_power_mod :
  (5^2023) % 17 = 15 :=
begin
  sorry
end

end remainder_of_power_mod_l80_80329


namespace infinite_series_sum_l80_80269

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80269


namespace sum_of_factors_36_l80_80720

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80720


namespace purely_imaginary_iff_x_eq_one_l80_80304

theorem purely_imaginary_iff_x_eq_one :
  ∀ x : ℝ, (x + 2*complex.I) * (x + 1 + 3*complex.I) * (x + 2 + 4*complex.I) ∈ (set.range (λ c : ℝ, c * complex.I)) ↔ x = 1 :=
by
    intro x
    split
    sorry
    sorry

end purely_imaginary_iff_x_eq_one_l80_80304


namespace biggest_number_in_ratio_l80_80951

theorem biggest_number_in_ratio (x : ℕ) (h_sum : 2 * x + 3 * x + 4 * x + 5 * x = 1344) : 5 * x = 480 := 
by
  sorry

end biggest_number_in_ratio_l80_80951


namespace sum_infinite_series_l80_80197

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80197


namespace hyperbola_eccentricity_l80_80385

variable (x y a b c : ℝ)

-- Conditions given in the problem
def hyperbola (x y a b : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ ((x^2 / a^2) - (y^2 / b^2) = 1)

def asymptote_through_point (a b : ℝ) (point_x point_y : ℝ) : Prop :=
  (point_y / point_x = b / a)

def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 + (b^2 / a^2))

-- Given that the point (sqrt(2), sqrt(6)) is on asymptote
def point := (sqrt 2, sqrt 6)

-- The main statement to prove the eccentricity is 2
theorem hyperbola_eccentricity :
  ∀ a b : ℝ, hyperbola x y a b ∧ asymptote_through_point a b (fst point) (snd point) → 
  eccentricity a b = 2 :=
by
  sorry

end hyperbola_eccentricity_l80_80385


namespace distribution_of_items_l80_80111

open Finset

theorem distribution_of_items :
  ∃ (count : ℕ), count = 52 ∧
  ∃ (ways : Finset (Multiset (Multiset ℕ))), ways.card = 52 ∧
  (∀ (items : Finset ℕ) (bags : Finset (Finset (Finset ℕ))),
    items.card = 5 →
    bags.card = 4 →
    (∃ way : Multiset (Multiset ℕ), way ∈ ways) →
    bags ⊆ ways.toFinset) :=
begin
  sorry
end

end distribution_of_items_l80_80111


namespace sum_factors_36_eq_91_l80_80603

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80603


namespace find_square_divisible_by_four_l80_80300

/-- There exists an x such that x is a square number, x is divisible by four, 
and 39 < x < 80, and that x = 64 is such a number. --/
theorem find_square_divisible_by_four : ∃ (x : ℕ), (∃ (n : ℕ), x = n^2) ∧ (x % 4 = 0) ∧ (39 < x ∧ x < 80) ∧ x = 64 :=
  sorry

end find_square_divisible_by_four_l80_80300


namespace distribution_of_items_l80_80110

open Finset

theorem distribution_of_items :
  ∃ (count : ℕ), count = 52 ∧
  ∃ (ways : Finset (Multiset (Multiset ℕ))), ways.card = 52 ∧
  (∀ (items : Finset ℕ) (bags : Finset (Finset (Finset ℕ))),
    items.card = 5 →
    bags.card = 4 →
    (∃ way : Multiset (Multiset ℕ), way ∈ ways) →
    bags ⊆ ways.toFinset) :=
begin
  sorry
end

end distribution_of_items_l80_80110


namespace sum_of_factors_36_eq_91_l80_80746

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80746


namespace sum_of_factors_36_l80_80699

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80699


namespace find_values_of_real_numbers_l80_80352

theorem find_values_of_real_numbers (x y : ℝ)
  (h : 2 * x - 1 + (y + 1) * Complex.I = x - y - (x + y) * Complex.I) :
  x = 3 ∧ y = -2 :=
sorry

end find_values_of_real_numbers_l80_80352


namespace gcd_of_1230_and_990_l80_80312

theorem gcd_of_1230_and_990 : Nat.gcd 1230 990 = 30 :=
by
  sorry

end gcd_of_1230_and_990_l80_80312


namespace matthews_crackers_l80_80488

theorem matthews_crackers (initial_cakes friends : ℕ) (cakes_each : ℕ) (ate_cakes : ℕ) (equal_crackers_and_cakes : bool) :
  initial_cakes = 8 → friends = 4 → cakes_each = 2 → ate_cakes = friends * cakes_each →
  equal_crackers_and_cakes = true → ate_cakes = initial_cakes → 
  ∃ initial_crackers, initial_crackers = 8 :=
by
  intros h_initial_cakes h_friends h_cakes_each h_ate_cakes h_equal_crackers_and_cakes h_ate_matches_initial
  use 8
  exact_Sorry

end matthews_crackers_l80_80488


namespace infinite_series_sum_l80_80232

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80232


namespace sum_series_eq_4_div_9_l80_80179

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80179


namespace total_bottles_l80_80848

theorem total_bottles (regular diet : ℕ) (h1 : regular = 28) (h2 : diet = 2) : regular + diet = 30 :=
by
  rw [h1, h2]
  rfl

end total_bottles_l80_80848


namespace maximize_area_center_coordinates_l80_80531

theorem maximize_area_center_coordinates (k : ℝ) :
  (∃ r : ℝ, r^2 = 1 - (3/4) * k^2 ∧ r ≥ 0) →
  ((k = 0) → ∃ a b : ℝ, (a = 0 ∧ b = -1)) :=
by
  sorry

end maximize_area_center_coordinates_l80_80531


namespace cos_square_theta_plus_pi_over_4_eq_one_fourth_l80_80964

variable (θ : ℝ)

theorem cos_square_theta_plus_pi_over_4_eq_one_fourth
  (h : Real.tan θ + 1 / Real.tan θ = 4) :
  Real.cos (θ + Real.pi / 4) ^ 2 = 1 / 4 :=
sorry

end cos_square_theta_plus_pi_over_4_eq_one_fourth_l80_80964


namespace tangent_line_at_A_tangent_lines_through_origin_l80_80382

noncomputable def f (x : ℝ) : ℝ := x^3 + 4 * x^2

-- Proof for Part 1
theorem tangent_line_at_A :
  f 1 = 5 → ∃ m b, (∀ x, f' x = 3 * x^2 + 8 * x) ∧ (m = 11) ∧ (b = -6) ∧ (f 1 = 5) → (∀ x, f x = m * x + b) :=
by sorry

-- Proof for Part 2
theorem tangent_lines_through_origin :
  (∃ x y, x = 0 ∧ y = 0 ∧ (∀ x, f' x = 3 * x^2 + 8 * x) ∧ f x = y) ∧
  (∃ x y, x = -2 ∧ y = 8 ∧ (∀ x, f' x = 3 * x^2 + 8 * x) ∧ f x = y) :=
by sorry

end tangent_line_at_A_tangent_lines_through_origin_l80_80382


namespace sum_series_eq_4_div_9_l80_80289

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80289


namespace sum_of_factors_36_l80_80732

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80732


namespace could_not_be_diagonal_lengths_l80_80824

-- Definitions of the diagonal conditions
def diagonal_condition (s : List ℕ) : Prop :=
  match s with
  | [x, y, z] => x^2 + y^2 > z^2 ∧ x^2 + z^2 > y^2 ∧ y^2 + z^2 > x^2
  | _ => false

-- Statement of the problem
theorem could_not_be_diagonal_lengths : 
  ¬ diagonal_condition [5, 6, 8] :=
by 
  sorry

end could_not_be_diagonal_lengths_l80_80824


namespace candy_division_l80_80339

/-- Given Frank had 16 pieces of candy and he put them equally into 2 bags,
    prove that there are 8 pieces of candy in each bag. --/
def pieces_in_each_bag (total_candy : ℕ) (num_bags : ℕ) (pieces_per_bag : ℕ) : Prop :=
  total_candy = 16 ∧ num_bags = 2 → pieces_per_bag = 8

-- The statement for the provided problem
theorem candy_division : pieces_in_each_bag 16 2 8 :=
by
  intro h
  cases h with h_total h_bags
  rw [h_total, h_bags]
  sorry

end candy_division_l80_80339


namespace sum_factors_36_l80_80804

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80804


namespace series_sum_eq_four_ninths_l80_80245

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80245


namespace instantaneous_rate_of_change_at_6_l80_80554

def f (x : ℝ) : ℝ := (3 * x / (4 * Real.exp 1)) ^ (1 / 3 : ℝ)

theorem instantaneous_rate_of_change_at_6 :
  (Real.deriv f 6) = (1 / 9) * (4 * Real.exp 2 / 3) ^ (1 / 3 : ℝ) :=
by
  sorry

end instantaneous_rate_of_change_at_6_l80_80554


namespace carolyn_lace_cost_l80_80902

theorem carolyn_lace_cost : 
  let cuffs_length := 50
  let hem_length := 300
  let waist_ratio := 1/3
  let ruffle_count := 5
  let ruffle_length := 20
  let cost_per_meter := 6
  let total_cuff_length := 2 * cuffs_length
  let total_waist_length := waist_ratio * hem_length
  let total_neckline_length := ruffle_count * ruffle_length
  let total_lace_length_cm := total_cuff_length + hem_length + total_waist_length + total_neckline_length
  let total_lace_length_m := total_lace_length_cm / 100
  let total_cost := total_lace_length_m * cost_per_meter
  in total_cost = 36 := by
    -- specify the proof here
    sorry

end carolyn_lace_cost_l80_80902


namespace evaluate_series_sum_l80_80274

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80274


namespace interval_length_of_solutions_l80_80914

theorem interval_length_of_solutions (a b : ℝ) :
  (∃ x : ℝ, a ≤ 3*x + 6 ∧ 3*x + 6 ≤ b) ∧ (∃ (l : ℝ), l = (b - a) / 3 ∧ l = 15) → b - a = 45 :=
by sorry

end interval_length_of_solutions_l80_80914


namespace sum_of_factors_36_l80_80728

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80728


namespace increasing_function_range_of_a_monotonicity_intervals_l80_80534

-- Definitions based on the conditions provided in the problem
noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

-- Lean statements to prove
theorem increasing_function_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ Real.exp x - a) → a ≤ 0 :=
sorry
  
theorem monotonicity_intervals (x : ℝ) :
  let f1 := λ x : ℝ, Real.exp x - x - 1 in
  ((∀ x : ℝ, 0 < x → 0 < Real.exp x - 1) ∧ (∀ x : ℝ, x < 0 → Real.exp x - 1 < 0)) :=
sorry

end increasing_function_range_of_a_monotonicity_intervals_l80_80534


namespace jeremy_time_on_bus_l80_80468

def total_time_away (bus_departure : ℕ) (bus_return : ℕ) : ℕ :=
  (bus_return - bus_departure) * 60

def total_school_time (num_classes : ℕ) (class_duration : ℕ) (lunch_time : ℕ) (additional_time_h : ℕ) : ℕ :=
  (num_classes * class_duration) + lunch_time + (additional_time_h * 60)

def time_on_bus : Prop :=
  let away_minutes := total_time_away 7 17 in
  let school_minutes := total_school_time 7 45 45 2 in
  away_minutes - school_minutes = 105

#eval total_time_away 7 17 -- Evaluates to 600
#eval total_school_time 7 45 45 2 -- Evaluates to 495
#eval time_on_bus -- Evaluates to true if the properties hold

theorem jeremy_time_on_bus : time_on_bus :=
by {
  sorry
}

end jeremy_time_on_bus_l80_80468


namespace sum_of_factors_36_eq_91_l80_80748

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80748


namespace power_mod_l80_80323

theorem power_mod (a : ℕ) : 5 ^ 2023 % 17 = 2 := by
  sorry

end power_mod_l80_80323


namespace sum_of_factors_of_36_l80_80664

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80664


namespace sum_factors_of_36_l80_80654

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80654


namespace pen_sales_average_l80_80500

theorem pen_sales_average :
  ∃ d : ℕ, (48 = (96 + 44 * d) / (d + 1)) → d = 12 :=
by
  sorry

end pen_sales_average_l80_80500


namespace expected_value_absolute_difference_after_10_days_l80_80038

/-- Define the probability space and outcome -/
noncomputable def probability_cat_wins : ℝ := 0.25
noncomputable def probability_fox_wins : ℝ := 0.25
noncomputable def probability_both_police : ℝ := 0.50

/-- Define the random variable for absolute difference in wealth -/
noncomputable def X_n (n : ℕ) : ℝ := sorry

/-- Define the probability p_{0, n} -/
noncomputable def p (k n : ℕ) : ℝ := sorry

/-- Given the above conditions, the expected value of the absolute difference -/
theorem expected_value_absolute_difference_after_10_days : (∑ k in finset.range 11, k * p k 10) = 1 :=
sorry

end expected_value_absolute_difference_after_10_days_l80_80038


namespace integral_f_l80_80346

def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then x^2 else
  if 0 < x ∧ x ≤ 1 then 1 else 0

-- Statement of the theorem
theorem integral_f : ∫ x in -1..1, f x = 4 / 3 :=
by sorry

end integral_f_l80_80346


namespace length_of_YZ_in_triangle_XYZ_l80_80460

theorem length_of_YZ_in_triangle_XYZ
    (XY XZ XM : ℝ)
    (hXY : XY = 6)
    (hXZ : XZ = 9)
    (hXM : XM = 6)
    : ∃ YZ : ℝ, YZ = 3 * Real.sqrt 10 :=
begin
  sorry
end

end length_of_YZ_in_triangle_XYZ_l80_80460


namespace b11_eq_4_l80_80477

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d r : ℤ} {a1 : ℤ}

-- Define non-zero arithmetic sequence {a_n} with common difference d
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define geometric sequence {b_n} with common ratio r
def is_geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ n, b (n + 1) = b n * r

-- The given conditions
axiom a1_minus_a7_sq_plus_a13_eq_zero : a 1 - (a 7) ^ 2 + a 13 = 0
axiom b7_eq_a7 : b 7 = a 7

-- The problem statement to prove: b 11 = 4
theorem b11_eq_4
  (arith_seq : is_arithmetic_sequence a d)
  (geom_seq : is_geometric_sequence b r)
  (a1_non_zero : a1 ≠ 0) :
  b 11 = 4 :=
sorry

end b11_eq_4_l80_80477


namespace one_way_ticket_cost_l80_80073

theorem one_way_ticket_cost (x : ℝ) (h : 50 / 26 < x) : x >= 2 :=
by sorry

end one_way_ticket_cost_l80_80073


namespace fixed_point_for_all_parabolas_l80_80140

theorem fixed_point_for_all_parabolas : ∃ (x y : ℝ), (∀ t : ℝ, y = 4 * x^2 + 2 * t * x - 3 * t) ∧ x = 1 ∧ y = 4 :=
by 
  sorry

end fixed_point_for_all_parabolas_l80_80140


namespace sum_factors_36_l80_80635

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80635


namespace sum_of_numbers_with_lcm_and_ratio_l80_80046

theorem sum_of_numbers_with_lcm_and_ratio 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 48)
  (h_ratio : a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3) : 
  a + b = 80 := 
by sorry

end sum_of_numbers_with_lcm_and_ratio_l80_80046


namespace inequality_proof_l80_80484

noncomputable def sequence_x : ℕ → ℝ 
| 0 := √2
| (n + 1) := sequence_x n + (1 / sequence_x n)

theorem inequality_proof : 
  let lhs := ∑ n in Finset.range 2020, (sequence_x n) ^ 2 / (2 * sequence_x n * sequence_x (n + 1) - 1)
  let rhs := 2019 ^ 2 / (sequence_x 2019 ^ 2 + 1 / sequence_x 2019 ^ 2)
  lhs > rhs :=
begin
  sorry
end

end inequality_proof_l80_80484


namespace sum_factors_of_36_l80_80657

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80657


namespace sum_of_factors_36_l80_80722

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80722


namespace cat_fox_wealth_difference_l80_80021

noncomputable def prob_coin_toss : ℕ := ((1/4 : ℝ) + (1/4 : ℝ) - (1/2 : ℝ))

-- define the random variable X_n representing the "absolute difference in wealth at end of n-th day"
noncomputable def X (n : ℕ) : ℝ := sorry

-- statement of the proof problem
theorem cat_fox_wealth_difference : ∃ E : ℝ, E = 1 ∧ E = classical.some X 10 := 
sorry

end cat_fox_wealth_difference_l80_80021


namespace sum_factors_36_l80_80764

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80764


namespace sin_neg_600_eq_sqrt_3_div_2_l80_80146

theorem sin_neg_600_eq_sqrt_3_div_2 :
  Real.sin (-(600 * Real.pi / 180)) = Real.sqrt 3 / 2 :=
sorry

end sin_neg_600_eq_sqrt_3_div_2_l80_80146


namespace infinite_series_sum_l80_80266

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80266


namespace sum_of_factors_of_36_l80_80674

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80674


namespace manuscript_fee_l80_80856

noncomputable def tax (x : ℝ) : ℝ :=
  if x ≤ 800 then 0
  else if x <= 4000 then 0.14 * (x - 800)
  else 0.11 * x

theorem manuscript_fee (x : ℝ) (h₁ : tax x = 420)
  (h₂ : 800 < x ∧ x ≤ 4000 ∨ x > 4000) :
  x = 3800 :=
sorry

end manuscript_fee_l80_80856


namespace sum_of_digits_N_l80_80138

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

def term (d : ℕ) : ℕ := (10^d - 1) / 9

def N' : ℕ := (Finset.range 200).sum (λ k => term (k + 1)) + term 150

theorem sum_of_digits_N' : sum_of_digits N' = 360 :=
sorry

end sum_of_digits_N_l80_80138


namespace smallest_n_with_all_digits_odd_of_97n_l80_80471

def all_digits_odd (n : ℕ) : Prop :=
  ∀ (d : ℕ), (d ∈ (nat.digits 10 n)) → d % 2 = 1

theorem smallest_n_with_all_digits_odd_of_97n : ∃ n : ℕ, n > 1 ∧ all_digits_odd (97 * n) ∧ ∀ m : ℕ, m > 1 ∧ all_digits_odd (97 * m) → n ≤ m :=
begin
  use 35,
  split,
  { -- n > 1
    linarith },
  split,
  { -- all_digits_odd (97 * 35)
    intros d hd,
    simp [nat.digits_def] at hd,
    cases hd,
    { -- d = 3
      linarith },
    cases hd,
    { -- d = 3
      linarith },
    cases hd,
    { -- d = 9
      linarith },
    cases hd,
    { -- d = 5
      linarith },
    exfalso,
    exact hd },
  { -- ∀ m > 1 it implies n ≤ m
    intros m hm hoddm,
    cases m,
    linarith,
    cases m,
    simp [all_digits_odd] at hoddm,
    split_ifs at hoddm; linarith }
end

end smallest_n_with_all_digits_odd_of_97n_l80_80471


namespace number_of_solutions_l80_80915

def satisfies_equation (z : ℂ) : Prop :=
  complex.exp z = (z + complex.I) / (z - complex.I)

def is_within_radius (z : ℂ) : Prop :=
  complex.abs z < 20

theorem number_of_solutions :
  ∃ S : finset ℂ, (∀ z ∈ S, satisfies_equation z ∧ is_within_radius z) ∧ S.card = 14 :=
sorry

end number_of_solutions_l80_80915


namespace reflection_matrix_y_eq_x_l80_80314

theorem reflection_matrix_y_eq_x :
  (matrix.of_vec [0, 1, 1, 0] : matrix (fin 2) (fin 2) ℝ) = λ (i j : fin 2), if i = j then 0 else 1 := 
sorry

end reflection_matrix_y_eq_x_l80_80314


namespace sum_series_l80_80252

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80252


namespace sum_of_series_l80_80166

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80166


namespace division_of_203_by_single_digit_l80_80054

theorem division_of_203_by_single_digit (d : ℕ) (h : 1 ≤ d ∧ d < 10) : 
  ∃ q : ℕ, q = 203 / d ∧ (10 ≤ q ∧ q < 100 ∨ 100 ≤ q ∧ q < 1000) := 
by
  sorry

end division_of_203_by_single_digit_l80_80054


namespace carolyn_lace_cost_l80_80901

theorem carolyn_lace_cost : 
  let cuffs_length := 50
  let hem_length := 300
  let waist_ratio := 1/3
  let ruffle_count := 5
  let ruffle_length := 20
  let cost_per_meter := 6
  let total_cuff_length := 2 * cuffs_length
  let total_waist_length := waist_ratio * hem_length
  let total_neckline_length := ruffle_count * ruffle_length
  let total_lace_length_cm := total_cuff_length + hem_length + total_waist_length + total_neckline_length
  let total_lace_length_m := total_lace_length_cm / 100
  let total_cost := total_lace_length_m * cost_per_meter
  in total_cost = 36 := by
    -- specify the proof here
    sorry

end carolyn_lace_cost_l80_80901


namespace sum_series_eq_4_div_9_l80_80295

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80295


namespace sum_series_eq_4_div_9_l80_80178

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80178


namespace horner_method_evaluation_l80_80356

noncomputable def f (x : ℕ) : ℕ := 2 * x^7 + x^6 + x^4 + x^2 + 1

theorem horner_method_evaluation : 
  let x := 2 in
  let v0 := x in
  let v1 := 2 * v0 + 1 in
  let V2 := v1 * x in
  V2 = 10 := 
by 
  sorry

end horner_method_evaluation_l80_80356


namespace number_of_black_boxcars_l80_80510

def red_boxcars : Nat := 3
def blue_boxcars : Nat := 4
def black_boxcar_capacity : Nat := 4000
def boxcar_total_capacity : Nat := 132000

def blue_boxcar_capacity : Nat := 2 * black_boxcar_capacity
def red_boxcar_capacity : Nat := 3 * blue_boxcar_capacity

def red_boxcar_total_capacity : Nat := red_boxcars * red_boxcar_capacity
def blue_boxcar_total_capacity : Nat := blue_boxcars * blue_boxcar_capacity

def other_total_capacity : Nat := red_boxcar_total_capacity + blue_boxcar_total_capacity
def remaining_capacity : Nat := boxcar_total_capacity - other_total_capacity
def expected_black_boxcars : Nat := remaining_capacity / black_boxcar_capacity

theorem number_of_black_boxcars :
  expected_black_boxcars = 7 := by
  sorry

end number_of_black_boxcars_l80_80510


namespace sum_series_l80_80257

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80257


namespace ordered_quadruples_count_l80_80320

noncomputable def count_ordered_quadruples : ℕ :=
  {n | ∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
     a^2 + b^2 + c^2 + d^2 = 9 ∧
     (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81 ∧
     n = 15}.card

theorem ordered_quadruples_count : count_ordered_quadruples = 15 := sorry

end ordered_quadruples_count_l80_80320


namespace P_plus_Q_is_expected_l80_80395

-- defining the set P
def P : Set ℝ := { x | x ^ 2 - 3 * x - 4 ≤ 0 }

-- defining the set Q
def Q : Set ℝ := { x | x ^ 2 - 2 * x - 15 > 0 }

-- defining the set P + Q
def P_plus_Q : Set ℝ := { x | (x ∈ P ∨ x ∈ Q) ∧ ¬(x ∈ P ∧ x ∈ Q) }

-- the expected result
def expected_P_plus_Q : Set ℝ := { x | x < -3 } ∪ { x | -1 ≤ x ∧ x ≤ 4 } ∪ { x | x > 5 }

-- theorem stating that P + Q equals the expected result
theorem P_plus_Q_is_expected : P_plus_Q = expected_P_plus_Q := by
  sorry

end P_plus_Q_is_expected_l80_80395


namespace sum_of_factors_36_eq_91_l80_80585

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80585


namespace maximum_k_secondary_diagonal_l80_80450

/-- In an n x n table, with k cells filled with 1 and the rest with 0,
    prove the maximum value k such that all 1's can be positioned above or on
    the secondary diagonal by swapping rows and columns. -/
theorem maximum_k_secondary_diagonal (n : ℕ) (n_pos : n > 0) :
  ∃ k, (∀ (table : Array (Array ℕ)), table.size = n ∧ (∀ row, row.size = n) ∧
    (∃ i j, table[i][j] = 1) →
    (∃ rperm cperm, 
        (Perm.rperm rperm).size = n ∧ (Perm.cperm cperm).size = n ∧ 
        (∀ i j, table[(Perm.rperm rperm) i][(Perm.cperm cperm) j] = 1) → 
        ((Perm.rperm rperm) i + (Perm.cperm cperm) j ≤ n - 1))) 
        ∧ k = n + 1) ∧ 
  (∀ (m : ℕ), m > n + 1 → ¬(∃ table : Array (Array ℕ),
    table.size = n ∧ (∀ row, row.size = n) ∧
    (count_ones table = m) ∧
    (∃ rperm cperm, 
        (Perm.rperm rperm).size = n ∧ (Perm.cperm cperm).size = n ∧ 
        (∀ i j, table[(Perm.rperm rperm) i][(Perm.cperm cperm) j] = 1) → 
        ((Perm.rperm rperm) i + (Perm.cperm cperm) j ≤ n - 1)))) :=
sorry

end maximum_k_secondary_diagonal_l80_80450


namespace sum_of_factors_36_l80_80678

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80678


namespace sum_series_l80_80249

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80249


namespace percentage_cross_pollinated_l80_80845

-- Definitions and known conditions:
variables (F C T : ℕ)
variables (h1 : F + C = 221)
variables (h2 : F = 3 * T / 4)
variables (h3 : T = F + 39 + C)

-- Theorem statement for the percentage of cross-pollinated trees
theorem percentage_cross_pollinated : ((C : ℚ) / T) * 100 = 10 :=
by sorry

end percentage_cross_pollinated_l80_80845


namespace polynomial_integer_roots_a_values_l80_80302

theorem polynomial_integer_roots_a_values :
  ∀ (a : ℤ),
  (∃ (x : ℤ), x^3 + 3 * x^2 + a * x + 8 = 0) ↔
  a ∈ {-89, -39, -30, -14, -12, -6, -2, 10} :=
by {
  sorry
}

end polynomial_integer_roots_a_values_l80_80302


namespace gcd_1230_990_l80_80311

theorem gcd_1230_990 : Int.gcd 1230 990 = 30 := by
  sorry

end gcd_1230_990_l80_80311


namespace parallel_lines_transitive_l80_80507

theorem parallel_lines_transitive {m1 m2 l : Line} 
  (h1 : m1 ∥ l) 
  (h2 : m2 ∥ l) : 
  m1 ∥ m2 := 
sorry

end parallel_lines_transitive_l80_80507


namespace groceries_delivered_amount_l80_80099

noncomputable def alex_saved_up : ℝ := 14500
noncomputable def car_cost : ℝ := 14600
noncomputable def charge_per_trip : ℝ := 1.5
noncomputable def percentage_charge : ℝ := 0.05
noncomputable def number_of_trips : ℕ := 40

theorem groceries_delivered_amount :
  ∃ G : ℝ, charge_per_trip * number_of_trips + percentage_charge * G = car_cost - alex_saved_up ∧ G = 800 :=
by {
  use 800,
  rw [mul_comm (800 : ℝ), mul_assoc],
  norm_num,
  exact add_comm 60 (40 : ℝ),
  sorry
}

end groceries_delivered_amount_l80_80099


namespace cara_neighbors_l80_80899

theorem cara_neighbors (friends : Finset Person) (mark : Person) (cara : Person) (h_mark : mark ∈ friends) (h_len : friends.card = 8) :
  ∃ pairs : Finset (Person × Person), pairs.card = 6 ∧
    ∀ (p : Person × Person), p ∈ pairs → p.1 = mark ∨ p.2 = mark :=
by
  -- The proof goes here.
  sorry

end cara_neighbors_l80_80899


namespace sum_factors_36_l80_80638

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80638


namespace infinite_series_sum_l80_80271

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80271


namespace intersection_with_complement_l80_80392

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {0, 2, 4}

theorem intersection_with_complement (hU : U = {0, 1, 2, 3, 4})
                                     (hA : A = {0, 1, 2, 3})
                                     (hB : B = {0, 2, 4}) :
  A ∩ (U \ B) = {1, 3} :=
by sorry

end intersection_with_complement_l80_80392


namespace sum_factors_36_l80_80793

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80793


namespace sum_of_factors_36_l80_80719

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80719


namespace evaluate_series_sum_l80_80276

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80276


namespace sum_of_series_l80_80176

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80176


namespace find_n_l80_80936

theorem find_n (n : ℤ) (h1 : -180 ≤ n ∧ n ≤ 180) (h2 : real.sin (n * real.pi / 180) = real.cos (810 * real.pi / 180)) : n ∈ {-180, 0, 180} :=
sorry

end find_n_l80_80936


namespace infinite_series_sum_l80_80231

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80231


namespace example_problem_l80_80881

theorem example_problem
  (h1 : 0.25 < 1) 
  (h2 : 0.15 < 0.25) : 
  3.04 / 0.25 > 1 :=
by
  sorry

end example_problem_l80_80881


namespace verify_closest_point_on_line_l80_80942

noncomputable def closest_point_on_line : Prop := 
  let line_point : ℝ × ℝ × ℝ := (5, -1, 2)
  let direction_vector : ℝ × ℝ × ℝ := (-3, 4, -2)
  let target_point : ℝ × ℝ × ℝ := (3, -2, 5)
  let closest_point : ℝ × ℝ × ℝ := (163/29, -45/29, 66/29)
  ∃ t : ℝ, 
    line_point.1 + t * direction_vector.1 = closest_point.1 ∧ 
    line_point.2 + t * direction_vector.2 = closest_point.2 ∧ 
    line_point.3 + t * direction_vector.3 = closest_point.3 ∧ 
    (let vector := (closest_point.1 - target_point.1, closest_point.2 - target_point.2, closest_point.3 - target_point.3)
    vector.1 * direction_vector.1 + vector.2 * direction_vector.2 + vector.3 * direction_vector.3 = 0)

-- proof of the statement is omitted (denoted by 'sorry')
theorem verify_closest_point_on_line : closest_point_on_line := 
  sorry

end verify_closest_point_on_line_l80_80942


namespace domain_range_sum_l80_80148

theorem domain_range_sum (m n : ℝ) 
  (h1 : ∀ x, m ≤ x ∧ x ≤ n → 3 * m ≤ -x ^ 2 + 2 * x ∧ -x ^ 2 + 2 * x ≤ 3 * n)
  (h2 : -m ^ 2 + 2 * m = 3 * m)
  (h3 : -n ^ 2 + 2 * n = 3 * n) :
  m = -1 ∧ n = 0 ∧ m + n = -1 := 
by 
  sorry

end domain_range_sum_l80_80148


namespace evaluate_series_sum_l80_80273

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80273


namespace find_acute_angle_l80_80416

theorem find_acute_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) 
    (h3 : Real.sin α = 1 - Real.sqrt 3 * Real.tan (π / 18) * Real.sin α) : 
    α = π / 3 * 5 / 9 :=
by
  sorry

end find_acute_angle_l80_80416


namespace number_of_quadruples_l80_80317

/-- The number of ordered quadruples (a, b, c, d) of nonnegative real numbers such that
    a² + b² + c² + d² = 9 and (a + b + c + d)(a³ + b³ + c³ + d³) = 81 is 15. -/
theorem number_of_quadruples :
  ∃ (s : Finset (ℝ × ℝ × ℝ × ℝ)),
    (∀ (p : ℝ × ℝ × ℝ × ℝ), p ∈ s → 
     let (a, b, c, d) := p in 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 
     a^2 + b^2 + c^2 + d^2 = 9 ∧ 
     (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81) ∧ 
    s.card = 15 :=
by
  sorry

end number_of_quadruples_l80_80317


namespace sum_series_eq_4_div_9_l80_80286

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80286


namespace sum_of_factors_of_36_l80_80786

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80786


namespace mn_parallel_ad_l80_80108

variables {A B C D M N P Q O Z : Type} -- Points in our geometric space.

-- Definitions and conditions
variables (abcTriangle: Triangle A B C)
variables (circumcircleABC: Circle O A B C)
variables (angleBisectorAD: IsAngleBisector A B C D)
variables (midpointM: Midpoint B C M)
variables (circumcircleADM: Circle Z A D M intersects A B P ∧ intersects A C Q)
variables (midpointN: Midpoint P Q N)

-- The theorem we need to prove: MN is parallel to AD
theorem mn_parallel_ad (h1: ∃ O, Circle O A B C)
                     (h2: IsAngleBisector A B C D)
                     (h3: midpointM)
                     (h4: ∃ Z, Circle Z A D M ∧ intersects A B P ∧ intersects A C Q)
                     (h5: midpointN):
  Parallel MN AD :=
begin
  sorry
end

end mn_parallel_ad_l80_80108


namespace power_mod_l80_80324

theorem power_mod (a : ℕ) : 5 ^ 2023 % 17 = 2 := by
  sorry

end power_mod_l80_80324


namespace triangle_count_l80_80463

theorem triangle_count (ABC : Triangle) 
  (D1 D2 D3 : Point) (E1 E2 E3 : Point)
  (hD1 : D1 ∈ ABC.side AB) (hD2 : D2 ∈ ABC.side AB) (hD3 : D3 ∈ ABC.side AB)
  (hE1 : E1 ∈ ABC.side AC) (hE2 : E2 ∈ ABC.side AC) (hE3 : E3 ∈ ABC.side AC) :
  count_triangles ABC D1 D2 D3 E1 E2 E3 = 64 := 
sorry

end triangle_count_l80_80463


namespace g_has_two_zeros_in_interval_l80_80381

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) - sqrt 3 * cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3) - x

theorem g_has_two_zeros_in_interval : ∃ a b : ℝ, (0 ≤ a ∧ a ≤ π) ∧ (0 ≤ b ∧ b ≤ π) ∧ a ≠ b ∧ g a = 0 ∧ g b = 0 := by
  sorry

end g_has_two_zeros_in_interval_l80_80381


namespace hyperbola_sufficient_condition_l80_80333

-- Define the condition for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  (3 - k) * (k - 1) < 0

-- Lean 4 statement to prove that k > 3 is a sufficient condition for the given equation
theorem hyperbola_sufficient_condition (k : ℝ) (h : k > 3) :
  represents_hyperbola k :=
sorry

end hyperbola_sufficient_condition_l80_80333


namespace usamo_2012_function_problem_l80_80301

-- Define the set of positive integers
def ℤ₊ := {n : ℤ // n > 0}

-- Define the function type f: ℤ₊ → ℤ₊
def f (n : ℤ₊) : ℤ₊ := sorry

-- The first condition
def condition1 (f : ℤ₊ → ℤ₊) : Prop :=
  ∀ n : ℤ₊, f ⟨factorial n.val, nat.factorial_pos n.property⟩ = ⟨factorial (f n).val, nat.factorial_pos (classical.some_spec (f n).property)⟩

-- The second condition
def condition2 (f : ℤ₊ → ℤ₊) : Prop :=
  ∀ {m n: ℤ₊}, m ≠ n → (m.val - n.val) ∣ (f m).val - (f n).val

-- The problem statement equivalent in Lean 4
theorem usamo_2012_function_problem:
  ∀ f: ℤ₊ → ℤ₊, 
  condition1 f ∧ condition2 f → 
  (f = λ n, ⟨1, sorry⟩ ∨ f = λ n, ⟨2, sorry⟩ ∨ f = λ n, n) := sorry

end usamo_2012_function_problem_l80_80301


namespace senior_citizen_tickets_l80_80004

theorem senior_citizen_tickets (A S : ℕ) 
  (h1 : A + S = 510) 
  (h2 : 21 * A + 15 * S = 8748) : 
  S = 327 :=
by 
  -- Proof steps are omitted as instructed
  sorry

end senior_citizen_tickets_l80_80004


namespace cat_fox_wealth_difference_l80_80020

noncomputable def prob_coin_toss : ℕ := ((1/4 : ℝ) + (1/4 : ℝ) - (1/2 : ℝ))

-- define the random variable X_n representing the "absolute difference in wealth at end of n-th day"
noncomputable def X (n : ℕ) : ℝ := sorry

-- statement of the proof problem
theorem cat_fox_wealth_difference : ∃ E : ℝ, E = 1 ∧ E = classical.some X 10 := 
sorry

end cat_fox_wealth_difference_l80_80020


namespace clothes_donation_l80_80876

variable (initial_clothes : ℕ)
variable (clothes_thrown_away : ℕ)
variable (final_clothes : ℕ)
variable (x : ℕ)

theorem clothes_donation (h1 : initial_clothes = 100) 
                        (h2 : clothes_thrown_away = 15) 
                        (h3 : final_clothes = 65) 
                        (h4 : 4 * x = initial_clothes - final_clothes - clothes_thrown_away) :
  x = 5 := by
  sorry

end clothes_donation_l80_80876


namespace total_hunts_l80_80440

-- Conditions
def Sam_hunts : ℕ := 6
def Rob_hunts := Sam_hunts / 2
def combined_Rob_Sam_hunts := Rob_hunts + Sam_hunts
def Mark_hunts := combined_Rob_Sam_hunts / 3
def Peter_hunts := 3 * Mark_hunts

-- Question and proof statement
theorem total_hunts : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 := by
  sorry

end total_hunts_l80_80440


namespace roots_cubed_sum_l80_80475

noncomputable def polynomial := 8 * x^3 + 2012 * x + 2013

theorem roots_cubed_sum : 
  ∀ (α β γ : ℂ), 
    (polynomial.eval α = 0 ∧ polynomial.eval β = 0 ∧ polynomial.eval γ = 0 ∧ (α + β + γ = 0)) →
    ((α + β)^3 + (β + γ)^3 + (γ + α)^3) = 6039 / 8 :=
by 
  intros α β γ h
  sorry

end roots_cubed_sum_l80_80475


namespace biggest_number_in_ratio_l80_80953

theorem biggest_number_in_ratio (A B C D : ℕ) (h1 : 2 * D = 5 * A) (h2 : 3 * D = 5 * B) (h3 : 4 * D = 5 * C) (h_sum : A + B + C + D = 1344) : D = 480 := 
by
  sorry

end biggest_number_in_ratio_l80_80953


namespace distribute_items_5_in_4_identical_bags_l80_80114

theorem distribute_items_5_in_4_identical_bags : 
  let items := 5 in 
  let bags := 4 in 
  number_of_ways_to_distribute items bags = 36 := 
by sorry

end distribute_items_5_in_4_identical_bags_l80_80114


namespace manager_salary_l80_80523

def avg_salary_employees := 1500
def num_employees := 20
def avg_salary_increase := 600
def num_total_people := num_employees + 1

def total_salary_employees := num_employees * avg_salary_employees
def new_avg_salary := avg_salary_employees + avg_salary_increase
def total_salary_with_manager := num_total_people * new_avg_salary

theorem manager_salary : total_salary_with_manager - total_salary_employees = 14100 :=
by
  sorry

end manager_salary_l80_80523


namespace least_positive_integer_reducible_fraction_l80_80941

theorem least_positive_integer_reducible_fraction :
  ∃ n : ℕ, (n > 0) ∧ Nat.gcd (n - 17) (3 * n + 4) > 1 ∧ ∀ m : ℕ, (m > 0) ∧ Nat.gcd (m - 17) (3 * m + 4) > 1 → n ≤ m :=
begin
  use 22,
  split,
  { exact nat.succ_pos' 21 },
  split,
  { calc
      Nat.gcd (22 - 17) (3 * 22 + 4)
          = Nat.gcd 5 70 : by rw [sub_eq_add_neg, add_comm 17]; norm_num
      ... = 5 : by norm_num },
  { intros m hm,
    cases hm with _ h,
    have h1 : 3 * m + 4 = 3 * m + 4, by refl,
    rw [←Nat.gcd_zero_left (3 * m + 4), ←mod_eq_sub_mod hm.1, gcd_add_mul_right_left _ _ 3, gcd_comm_self, gcd_great_or_equal_iff_eq_if_dvd] at h,
    { cases h,
      { subst m },
      { refl } },
    { exact Iff.find_left sorry (Nat.gcd_pos_of_pos_right _ (nat.succ_pos' m)) } }
end

end least_positive_integer_reducible_fraction_l80_80941


namespace num_first_and_second_year_students_total_l80_80006

-- Definitions based on conditions
def num_sampled_students : ℕ := 55
def num_first_year_students_sampled : ℕ := 10
def num_second_year_students_sampled : ℕ := 25
def num_third_year_students_total : ℕ := 400

-- Given that 20 students from the third year are sampled
def num_third_year_students_sampled := num_sampled_students - num_first_year_students_sampled - num_second_year_students_sampled

-- Proportion equality condition
theorem num_first_and_second_year_students_total (x : ℕ) :
  20 / 55 = 400 / (x + num_third_year_students_total) →
  x = 700 :=
by
  sorry

end num_first_and_second_year_students_total_l80_80006


namespace share_y_is_18_l80_80085

-- Definitions from conditions
def total_amount := 70
def ratio_x := 100
def ratio_y := 45
def ratio_z := 30
def total_ratio := ratio_x + ratio_y + ratio_z
def part_value := total_amount / total_ratio
def share_y := ratio_y * part_value

-- Statement to be proved
theorem share_y_is_18 : share_y = 18 :=
by
  -- Placeholder for the proof
  sorry

end share_y_is_18_l80_80085


namespace g_at_3_l80_80486

def g (x : ℝ) : ℝ := x^3 - 2 * x^2 + x

theorem g_at_3 : g 3 = 12 := by
  sorry

end g_at_3_l80_80486


namespace sequence_general_term_l80_80359

noncomputable def S : ℕ → ℤ
| n => 3 * (n ^ 2) - 2 * n

def a_n (n : ℕ) : ℤ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem sequence_general_term : ∀ n : ℕ, n > 0 → a_n n = 6 * n - 5 :=
by
  intros n hn
  sorry

end sequence_general_term_l80_80359


namespace sum_k_over_4k_l80_80156

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80156


namespace area_of_trapezoid_l80_80868

namespace TrapezoidArea

def height : ℝ := 13
def base1 : ℝ := 13
def base2 : ℝ := 13 / 2

theorem area_of_trapezoid :
  (1 / 2) * (base1 + base2) * height = 126.75 :=
by
  -- Proof omitted as requested
  sorry

end TrapezoidArea

end area_of_trapezoid_l80_80868


namespace sum_series_eq_4_div_9_l80_80285

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80285


namespace imaginary_part_of_z_l80_80353

noncomputable def z : ℂ := Complex.mk (3/5) (-4/5)

theorem imaginary_part_of_z :
  let z : ℂ := Complex.mk (3/5) (-4/5) in 
  (3 - 4 * Complex.I) * Complex.conj z = Complex.abs (4 + 3 * Complex.I) → 
  Complex.im z = -4/5 :=
by
  sorry

end imaginary_part_of_z_l80_80353


namespace part1_cos_angle_part2_find_k_l80_80962

noncomputable def vector_a : ℝ × ℝ × ℝ := (3, 2, -1)
noncomputable def vector_b : ℝ × ℝ × ℝ := (2, 1, 2)

theorem part1_cos_angle :
  let dot_prod := (3 * 2) + (2 * 1) + (-1 * 2),
      mag_a := Real.sqrt (3^2 + 2^2 + (-1)^2),
      mag_b := Real.sqrt (2^2 + 1^2 + 2^2)
  in (dot_prod / (mag_a * mag_b) = 2 / Real.sqrt 14) := sorry

theorem part2_find_k (k : ℝ) :
  let a := vector_a,
      b := vector_b,
      condition := (k * a.1 + b.1) * (a.1 - k * b.1) + (k * a.2 + b.2) * (a.2 - k * b.2) + (k * a.3 + b.3) * (a.3 - k * b.3)
  in condition = 0 → (k = 3 / 2 ∨ k = -2 / 3) := sorry

end part1_cos_angle_part2_find_k_l80_80962


namespace sum_of_factors_36_l80_80680

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80680


namespace infinite_series_sum_l80_80270

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80270


namespace expected_difference_after_10_days_l80_80032

noncomputable def cat_fox_expected_difference : ℕ → ℝ
| 0     := 0
| (n+1) := 0.25 * (cat_fox_expected_difference n + 1)  -- cat wins
                + 0.25 * (cat_fox_expected_difference n + 1)  -- fox wins
                + 0.5 * 0  -- both go to police, difference resets

theorem expected_difference_after_10_days :
  cat_fox_expected_difference 10 = 1 :=
sorry

end expected_difference_after_10_days_l80_80032


namespace infinite_series_sum_l80_80230

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80230


namespace sum_factors_36_eq_91_l80_80602

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80602


namespace solve_inequality_l80_80514

-- Define the inequality as a function
def inequality_holds (x : ℝ) : Prop :=
  (2 * x + 3) / (x + 4) > (4 * x + 5) / (3 * x + 10)

-- Define the solution set as intervals excluding the points
def solution_set (x : ℝ) : Prop :=
  x < -5 / 2 ∨ x > -2

theorem solve_inequality (x : ℝ) : inequality_holds x ↔ solution_set x :=
by sorry

end solve_inequality_l80_80514


namespace binary_to_decimal_and_octal_conversion_l80_80911

theorem binary_to_decimal_and_octal_conversion : 
  let binary_num := 101101 in
  let decimal_num := 45 in
  let octal_num := 55 in
  (Nat.ofDigits 2 [1, 0, 1, 1, 0, 1] = decimal_num) ∧
  (Nat.toDigits 8 decimal_num = [5, 5]) :=
by
  sorry

end binary_to_decimal_and_octal_conversion_l80_80911


namespace uniform_CDF_equation_l80_80553

def uniformCDF (x : ℝ) : ℝ :=
  if x ≤ -3 then 0
  else if x < 2 then (x + 3) / 5
  else 1

theorem uniform_CDF_equation (x : ℝ) :
  uniformCDF x = 
  if x ≤ -3 then 0
  else if x < 2 then (x + 3) / 5
  else 1 := by
  sorry

end uniform_CDF_equation_l80_80553


namespace find_AB_l80_80063

theorem find_AB 
  (A B C Q N : Point)
  (h_AQ_QC : AQ / QC = 5 / 2)
  (h_CN_NB : CN / NB = 5 / 2)
  (h_QN : QN = 5 * Real.sqrt 2) : 
  AB = 7 * Real.sqrt 5 :=
sorry

end find_AB_l80_80063


namespace sum_series_eq_4_div_9_l80_80290

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80290


namespace sum_factors_of_36_l80_80648

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80648


namespace positive_integer_triples_satisfying_conditions_l80_80926

theorem positive_integer_triples_satisfying_conditions :
  ∀ (a b c : ℕ), a^2 + b^2 + c^2 = 2005 ∧ a ≤ b ∧ b ≤ c →
  (a, b, c) = (23, 24, 30) ∨
  (a, b, c) = (12, 30, 31) ∨
  (a, b, c) = (9, 30, 32) ∨
  (a, b, c) = (4, 30, 33) ∨
  (a, b, c) = (15, 22, 36) ∨
  (a, b, c) = (9, 18, 40) ∨
  (a, b, c) = (4, 15, 42) :=
sorry

end positive_integer_triples_satisfying_conditions_l80_80926


namespace sum_factors_of_36_l80_80656

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80656


namespace sum_of_factors_36_l80_80698

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80698


namespace sum_of_factors_36_l80_80691

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80691


namespace tan_pi_minus_x_l80_80958

theorem tan_pi_minus_x (x : ℝ) (h1 : cos (π + x) = 3 / 5) (h2 : π < x ∧ x < 2 * π) : 
  tan (π - x) = 4 / 3 :=
by
  sorry

end tan_pi_minus_x_l80_80958


namespace cloves_needed_l80_80058

theorem cloves_needed (cv_fp : 3 / 2 = 1.5) (cw_fp : 3 / 3 = 1) (vc_fp : 3 / 8 = 0.375) : 
  let cloves_for_vampires := 45
  let cloves_for_wights := 12
  let cloves_for_bats := 15
  30 * (3 / 2) + 12 * (3 / 3) + 40 * (3 / 8) = 72 := by
  sorry

end cloves_needed_l80_80058


namespace problem_1_problem_2_problem_3_l80_80375

theorem problem_1 (h1 : (x + m / x) ^ n = 256) : n = 8 :=
by {
  have h_sum := sum_binomial_coeff (x + m/x) n,
  rw h_sum at h1,
  exact eq_of_pow_eq_pow 8 (by norm_num) h1,
  sorry -- proof details omitted
}

theorem problem_2 (h2 : (x + m / x) ^ 8 = 35 / 8) : m = ± 1 / 2 :=
by {
  have h_const := const_term (x + m/x)^8,
  rw h_const at h2,
  exact solve_eqn (by norm_num) h2,
  sorry -- proof details omitted
}

theorem problem_3 (h3 : coeff_largest_term (x + m / x) ^ 8 = coeff_6_and_7) : m = 2 :=
by {
  have h_coeff := compute_coeff (x + m/x)^8,
  rw h_coeff at h3,
  exact solve_eqn_coeff_67 (by norm_num) h3,
  sorry -- proof details omitted
}

end problem_1_problem_2_problem_3_l80_80375


namespace swimmers_pass_each_other_l80_80570

/-- 
  Consider two swimmers A and B swimming in a 120-foot pool. Swimmer A swims 
  at 3 feet per second, and swimmer B swims at 4 feet per second. Swimmer B 
  starts 10 seconds after swimmer A. Both swim continuously back and forth in 
  the pool for 900 seconds (15 minutes) without any time loss at turns.
  In this setup, they pass each other 38 times within the 15-minute period.
-/
theorem swimmers_pass_each_other (length_pool : ℕ) (speed_A speed_B : ℕ) (start_delay : ℕ) (total_time : ℕ) :
  length_pool = 120 →
  speed_A = 3 →
  speed_B = 4 →
  start_delay = 10 →
  total_time = 900 →
  ∃ n, n = 38 :=
by
  intros,
  sorry

end swimmers_pass_each_other_l80_80570


namespace sum_factors_36_l80_80798

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80798


namespace expected_value_absolute_difference_after_10_days_l80_80041

/-- Define the probability space and outcome -/
noncomputable def probability_cat_wins : ℝ := 0.25
noncomputable def probability_fox_wins : ℝ := 0.25
noncomputable def probability_both_police : ℝ := 0.50

/-- Define the random variable for absolute difference in wealth -/
noncomputable def X_n (n : ℕ) : ℝ := sorry

/-- Define the probability p_{0, n} -/
noncomputable def p (k n : ℕ) : ℝ := sorry

/-- Given the above conditions, the expected value of the absolute difference -/
theorem expected_value_absolute_difference_after_10_days : (∑ k in finset.range 11, k * p k 10) = 1 :=
sorry

end expected_value_absolute_difference_after_10_days_l80_80041


namespace divide_and_round_l80_80840

noncomputable def round_up_to_nearest_cent (x : ℝ) : ℝ :=
  if x = (x.toRational.ceiling : ℝ) then x else (x.toRational.ceiling : ℝ)

theorem divide_and_round (total_bill : ℝ) (number_of_people : ℕ) (expected_share : ℝ)
  (h_bill : total_bill = 314.12) (h_people : number_of_people = 8) :
  round_up_to_nearest_cent (total_bill / number_of_people) = expected_share :=
by
  sorry

end divide_and_round_l80_80840


namespace calculate_clothing_fraction_l80_80854

noncomputable def salary : ℝ := 190000
noncomputable def foodCost(salary: ℝ) : ℝ := (1 / 5) * salary
noncomputable def rentCost(salary: ℝ) : ℝ := (1 / 10) * salary
noncomputable def clothesCost(salary: ℝ) (C : ℝ) : ℝ := C * salary

theorem calculate_clothing_fraction : 
  (∀ (s: ℝ), s ≈ salary) →
  (∀ (s: ℝ), clothesCost s C = s - 19000 - foodCost s - rentCost s) → 
  (C = 0.6) := sorry

end calculate_clothing_fraction_l80_80854


namespace part1_part2_l80_80461

theorem part1 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a + 2 * c = b * Real.cos C + Real.sqrt 3 * b * Real.sin C) : 
  B = 2 * Real.pi / 3 := 
sorry

theorem part2 
  (a b c : ℝ) 
  (A C : ℝ) 
  (h1 : a + 2 * c = b * Real.cos C + Real.sqrt 3 * b * Real.sin C)
  (h2 : b = 3) : 
  6 < (a + b + c) ∧ (a + b + c) ≤ 3 + 2 * Real.sqrt 3 :=
sorry

end part1_part2_l80_80461


namespace effective_annual_interest_rate_is_correct_l80_80518

noncomputable def quarterly_interest_rate : ℝ := 0.02

noncomputable def annual_interest_rate (quarterly_rate : ℝ) : ℝ :=
  ((1 + quarterly_rate) ^ 4 - 1) * 100

theorem effective_annual_interest_rate_is_correct :
  annual_interest_rate quarterly_interest_rate = 8.24 :=
by
  sorry

end effective_annual_interest_rate_is_correct_l80_80518


namespace sum_of_factors_36_eq_91_l80_80747

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80747


namespace expected_value_absolute_difference_after_10_days_l80_80039

/-- Define the probability space and outcome -/
noncomputable def probability_cat_wins : ℝ := 0.25
noncomputable def probability_fox_wins : ℝ := 0.25
noncomputable def probability_both_police : ℝ := 0.50

/-- Define the random variable for absolute difference in wealth -/
noncomputable def X_n (n : ℕ) : ℝ := sorry

/-- Define the probability p_{0, n} -/
noncomputable def p (k n : ℕ) : ℝ := sorry

/-- Given the above conditions, the expected value of the absolute difference -/
theorem expected_value_absolute_difference_after_10_days : (∑ k in finset.range 11, k * p k 10) = 1 :=
sorry

end expected_value_absolute_difference_after_10_days_l80_80039


namespace boundary_inner_relation_l80_80485

open Nat

-- Condition: n is an integer such that n ≥ 4
variables (n : ℕ) (h1 : n ≥ 4)

-- Condition: convex n-gon where no four vertices lie on the same circle
variables (convex_ngon: List (ℕ × ℕ))
(h2: ∀ (a b c d : ℕ), a ≠ b → b ≠ c → c ≠ d → d ≠ a → (a, b, c, d) ∉ (convex_ngon.powerset 4))

-- Definition of circumscribed circle passing through 3 points
def circumscribed_circle (a b c : ℕ) := 
  -- Assuming definition exists for the circle passing through a, b, c and containing all other vertices
  true

-- Definition of boundary circle
def boundary_circle (a b c : ℕ) : Prop :=
  circumscribed_circle n a b c ∧ (b = a + 1) ∧ (c = b + 1)

-- Definition of inner circle
def inner_circle (a b c : ℕ) : Prop :=
  circumscribed_circle n a b c ∧ (c - a > 1 ∧ c - b > 1)

-- Definition: counting the boundary circles
def count_boundary_circles : ℕ :=
  Nat.choose n 3

-- Definition: counting the inner circles
def count_inner_circles : ℕ :=
  Nat.choose n 3 - n

-- Proving the main statement
theorem boundary_inner_relation : 
  count_boundary_circles n h1 h2 = count_inner_circles n h1 h2 + 2 := 
sorry

end boundary_inner_relation_l80_80485


namespace sum_of_factors_36_l80_80701

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80701


namespace sum_of_factors_36_eq_91_l80_80754

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80754


namespace sum_factors_of_36_l80_80652

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80652


namespace fractional_product_l80_80125

theorem fractional_product :
  ((3/4) * (4/5) * (5/6) * (6/7) * (7/8)) = 3/8 :=
by
  sorry

end fractional_product_l80_80125


namespace distance_of_intersections_l80_80388

open Real

section

def line_parametric (t : ℝ) : ℝ × ℝ := (t, t - 1)

def curve_polar (ρ θ : ℝ) : Prop := ρ * sin θ^2 - 4 * cos θ = 0 ∧ ρ ≥ 0 ∧ θ ≥ 0 ∧ θ < 2 * π

theorem distance_of_intersections :
  let line_intersections := {t : ℝ | let (x, y) := line_parametric t in y^2 = 4 * x}
  ∃ t₁ t₂ ∈ line_intersections, t₁ ≠ t₂ ∧ |t₁ - t₂| = 8 :=
sorry

end

end distance_of_intersections_l80_80388


namespace sum_series_equals_4_div_9_l80_80216

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80216


namespace sum_factors_36_l80_80639

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80639


namespace lamplighter_monkey_distance_traveled_l80_80841

-- Define the parameters
def running_speed : ℕ := 15
def running_time : ℕ := 5
def swinging_speed : ℕ := 10
def swinging_time : ℕ := 10

-- Define the proof statement
theorem lamplighter_monkey_distance_traveled :
  (running_speed * running_time) + (swinging_speed * swinging_time) = 175 := by
  sorry

end lamplighter_monkey_distance_traveled_l80_80841


namespace sum_x_bounds_l80_80971

theorem sum_x_bounds (n : ℕ) (x : ℕ → ℝ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → x i ≥ 0) 
  (h1 : ∑ i in finset.range(n), x i ^ 2 + 2 * ∑ i in finset.range(n), ∑ j in finset.Icc(i+1,n), (sqrt i / sqrt j) * (x i) * (x j) = 1) :
  1 ≤ ∑ i in finset.range(n), x i ∧ ∑ i in finset.range(n), x i ≤ (∑ k in finset.range(n), (sqrt(k+1) - sqrt k) ^2) ^ (1/2) :=
begin
  sorry
end

end sum_x_bounds_l80_80971


namespace sum_series_equals_4_div_9_l80_80215

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80215


namespace exists_inequality_l80_80350

theorem exists_inequality {n : ℕ} (h1 : n > 1) (x : Fin n → ℝ) (h2 : ∀ (i : Fin n), 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i : Fin (n-1), x i * (1 - x ⟨i + 1, sorry⟩) ≥ x ⟨0, sorry⟩ * (1 - x ⟨n-1, sorry⟩) / 4 :=
by
  sorry

end exists_inequality_l80_80350


namespace sum_of_factors_of_36_l80_80661

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80661


namespace abs_diff_31st_term_l80_80002

-- Define the sequences C and D
def C (n : ℕ) : ℤ := 40 + 20 * (n - 1)
def D (n : ℕ) : ℤ := 40 - 20 * (n - 1)

-- Question: What is the absolute value of the difference between the 31st term of C and D?
theorem abs_diff_31st_term : |C 31 - D 31| = 1200 := by
  sorry

end abs_diff_31st_term_l80_80002


namespace math_inequality_l80_80411

variable {n : ℕ} (a : Fin n → ℝ) (k : ℕ)

theorem math_inequality (hpos : ∀ i, 0 < a i) (hn : 0 < n) :
  (Finset.univ.sum (λ i, (a i / a (i + 1) % n) ^ k)) ≥ 
  (Finset.univ.sum (λ i, a i / a (i + 1) % n)) :=
sorry

end math_inequality_l80_80411


namespace probability_two_foreign_language_speakers_l80_80102

theorem probability_two_foreign_language_speakers :
  let total_students := 7
  let foreign_language_speakers := 3
  let total_pairs := nat.choose total_students 2
  let foreign_language_pairs := nat.choose foreign_language_speakers 2
  total_pairs > 0 → (foreign_language_pairs / total_pairs : ℚ) = 1 / 7 :=
by
  intros
  sorry

end probability_two_foreign_language_speakers_l80_80102


namespace greatest_x_value_l80_80578

theorem greatest_x_value : 
  (∃ x : ℝ, 2 * x^2 + 7 * x + 3 = 5 ∧ ∀ y : ℝ, (2 * y^2 + 7 * y + 3 = 5) → y ≤ x) → x = 1 / 2 :=
by
  sorry

end greatest_x_value_l80_80578


namespace sum_of_factors_36_l80_80819

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80819


namespace sum_of_factors_36_l80_80809

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80809


namespace sum_of_factors_36_l80_80718

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80718


namespace total_cost_is_correct_l80_80575

noncomputable def total_cost : ℝ :=
  let palm_fern_cost := 15.00
  let creeping_jenny_cost := 4.00
  let geranium_cost := 3.50
  let elephant_ear_cost := 7.00
  let purple_fountain_grass_cost := 6.00
  let pots := 6
  let sales_tax := 0.07
  let cost_one_pot := palm_fern_cost 
                   + 4 * creeping_jenny_cost 
                   + 4 * geranium_cost 
                   + 2 * elephant_ear_cost 
                   + 3 * purple_fountain_grass_cost
  let total_pots_cost := pots * cost_one_pot
  let tax := total_pots_cost * sales_tax
  total_pots_cost + tax

theorem total_cost_is_correct : total_cost = 494.34 :=
by
  -- This is where the proof would go, but we are adding sorry to skip the proof
  sorry

end total_cost_is_correct_l80_80575


namespace second_boy_marbles_l80_80564

theorem second_boy_marbles (x : ℚ) :
  (4 * x + 2) + (3 * x - 1) + (5 * x + 3) = 128 → (3 * x - 1) = 30 :=
begin
  sorry,
end

end second_boy_marbles_l80_80564


namespace num_ways_to_distribute_items_l80_80115

theorem num_ways_to_distribute_items : 
  let items := 5
  let bags := 4
  let distinct_items := 5
  let identical_bags := 4
  (number_of_ways_to_distribute_items_in_4_identical_bags distinct_items identical_bags = 36) := sorry

end num_ways_to_distribute_items_l80_80115


namespace sum_of_factors_36_l80_80817

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80817


namespace sum_of_factors_of_36_l80_80665

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80665


namespace cloves_needed_l80_80057

theorem cloves_needed (cv_fp : 3 / 2 = 1.5) (cw_fp : 3 / 3 = 1) (vc_fp : 3 / 8 = 0.375) : 
  let cloves_for_vampires := 45
  let cloves_for_wights := 12
  let cloves_for_bats := 15
  30 * (3 / 2) + 12 * (3 / 3) + 40 * (3 / 8) = 72 := by
  sorry

end cloves_needed_l80_80057


namespace polygon_number_of_sides_l80_80075

-- Definitions based on conditions
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def exterior_angle (angle : ℕ) : ℕ := 30

-- The theorem statement
theorem polygon_number_of_sides (n : ℕ) (angle : ℕ) 
  (h1 : sum_of_exterior_angles n = 360)
  (h2 : exterior_angle angle = 30) : 
  n = 12 := 
by
  sorry

end polygon_number_of_sides_l80_80075


namespace selling_price_l80_80527

def cost_price : ℝ := 1500
def loss_percentage : ℝ := 20

theorem selling_price :
  let loss_amount := (loss_percentage / 100) * cost_price in
  let selling_price := cost_price - loss_amount in
  selling_price = 1200 :=
by
  sorry

end selling_price_l80_80527


namespace intersection_of_curves_l80_80944

theorem intersection_of_curves (x : ℝ) (y : ℝ) (h₁ : y = 9 / (x^2 + 3)) (h₂ : x + y = 3) : x = 0 :=
sorry

end intersection_of_curves_l80_80944


namespace sum_factors_36_eq_91_l80_80609

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80609


namespace increasing_interval_of_f_l80_80539

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - Real.log x

theorem increasing_interval_of_f :
  ∀ x : ℝ, x > 0 → (f' x > 0 ↔ x > 1 / 2)
    where 
      f' : ℝ → ℝ
      | x => 6 * x ^ 2 - 1 / x :=
  sorry

end increasing_interval_of_f_l80_80539


namespace sum_of_factors_36_l80_80697

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80697


namespace total_earnings_l80_80044

theorem total_earnings (x y : ℕ) 
  (h1 : 2 * x * y = 250) : 
  58 * (x * y) = 7250 := 
by
  sorry

end total_earnings_l80_80044


namespace sum_of_factors_36_l80_80721

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80721


namespace expected_difference_after_10_days_l80_80033

noncomputable def cat_fox_expected_difference : ℕ → ℝ
| 0     := 0
| (n+1) := 0.25 * (cat_fox_expected_difference n + 1)  -- cat wins
                + 0.25 * (cat_fox_expected_difference n + 1)  -- fox wins
                + 0.5 * 0  -- both go to police, difference resets

theorem expected_difference_after_10_days :
  cat_fox_expected_difference 10 = 1 :=
sorry

end expected_difference_after_10_days_l80_80033


namespace proof_problem_l80_80334

noncomputable def S (d : ℤ) : set ℤ :=
  { n | ∃ m n0 : ℤ, n = m^2 + d * n0^2 }

theorem proof_problem (d : ℤ) (p q : ℤ) (hp_prime : p.prime)
  (hp : p ∈ S d) (hq : q ∈ S d) (hpq : p ∣ q) :
  (q / p) ∈ S d :=
sorry

end proof_problem_l80_80334


namespace cosine_angle_between_vectors_l80_80399

noncomputable def vector_cosine (a b : ℝ × ℝ) : ℝ :=
  let dot_product := (a.1 * b.1 + a.2 * b.2)
  let magnitude_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / (magnitude_a * magnitude_b)

theorem cosine_angle_between_vectors : ∀ (k : ℝ), 
  let a := (3, 1)
  let b := (1, 3)
  let c := (k, -2)
  (3 - k) / 3 = 1 →
  vector_cosine a c = Real.sqrt 5 / 5 := by
  intros
  sorry

end cosine_angle_between_vectors_l80_80399


namespace sum_of_factors_36_l80_80811

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80811


namespace sum_of_factors_36_l80_80808

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80808


namespace sum_series_eq_four_ninths_l80_80210

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80210


namespace sum_factors_36_l80_80796

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80796


namespace sum_series_eq_four_ninths_l80_80208

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80208


namespace coefficient_x3_expansion_l80_80526

theorem coefficient_x3_expansion : 
  (∀ x : ℝ, (1 - x) * (2 * x + 1) ^ 4).coeff 3 = 8 := 
by 
  sorry

end coefficient_x3_expansion_l80_80526


namespace sum_series_eq_four_ninths_l80_80207

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80207


namespace sum_of_factors_36_l80_80702

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80702


namespace sum_of_factors_36_l80_80815

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80815


namespace cat_fox_wealth_difference_l80_80019

noncomputable def prob_coin_toss : ℕ := ((1/4 : ℝ) + (1/4 : ℝ) - (1/2 : ℝ))

-- define the random variable X_n representing the "absolute difference in wealth at end of n-th day"
noncomputable def X (n : ℕ) : ℝ := sorry

-- statement of the proof problem
theorem cat_fox_wealth_difference : ∃ E : ℝ, E = 1 ∧ E = classical.some X 10 := 
sorry

end cat_fox_wealth_difference_l80_80019


namespace points_on_line_y1_gt_y2_l80_80974

theorem points_on_line_y1_gt_y2 (y1 y2 : ℝ) : 
    (∀ x y, y = -x + 3 → 
    ((x = -4 → y = y1) ∧ (x = 2 → y = y2))) → 
    y1 > y2 :=
by
  sorry

end points_on_line_y1_gt_y2_l80_80974


namespace sum_of_factors_36_eq_91_l80_80583

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80583


namespace infinite_series_sum_l80_80272

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80272


namespace sequence_length_l80_80556

def sequence (a₀ : ℕ) (sequence : ℕ → ℕ) : ℕ := 
  if a₀ = 8640 then 
    Nat.find_greatest (λ n, (8640 / 2^n) ∈ ℕ) (λ n, 8640 / 2^n = 0)

theorem sequence_length : 
  sequence 8640 (λ n, 8640 / 2^n) = 7 :=
sorry

end sequence_length_l80_80556


namespace sum_of_factors_36_l80_80735

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80735


namespace checkerboard_probability_l80_80497

def checkerboard_size : ℕ := 10

def total_squares (n : ℕ) : ℕ := n * n

def perimeter_squares (n : ℕ) : ℕ := 4 * n - 4

def inner_squares (n : ℕ) : ℕ := total_squares n - perimeter_squares n

def probability_not_touching_edge (n : ℕ) : ℚ := inner_squares n / total_squares n

theorem checkerboard_probability :
  probability_not_touching_edge checkerboard_size = 16 / 25 := by
  sorry

end checkerboard_probability_l80_80497


namespace evaluate_series_sum_l80_80281

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80281


namespace vector_subtraction_l80_80396

-- Define the given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

-- State the theorem that the vector subtraction b - a equals (2, -1)
theorem vector_subtraction : b - a = (2, -1) :=
by
  -- Proof is omitted and replaced with sorry
  sorry

end vector_subtraction_l80_80396


namespace sum_of_factors_36_l80_80727

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80727


namespace function_value_at_2018_l80_80372

theorem function_value_at_2018 (f : ℝ → ℝ)
  (h1 : f 4 = 2 - Real.sqrt 3)
  (h2 : ∀ x, f (x + 2) = 1 / (- f x)) :
  f 2018 = -2 - Real.sqrt 3 :=
by
  sorry

end function_value_at_2018_l80_80372


namespace lawn_width_l80_80078

theorem lawn_width 
  (length_lawn : ℕ) 
  (cost_per_sqm : ℕ) 
  (total_cost : ℕ) 
  (width_road : ℕ) 
  (w : ℕ)
  (h_length : length_lawn = 55) 
  (h_cost_per_sqm : cost_per_sqm = 75) 
  (h_total_cost : total_cost = 258) 
  (h_width_road : width_road = 4)
  (h_total_area : 0.75 * (55 * 4 + 4 * w - 16) = 258) : 
  w = 35 :=
by
  sorry

end lawn_width_l80_80078


namespace sum_of_factors_36_l80_80711

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80711


namespace sum_of_factors_36_l80_80692

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80692


namespace rotated_vector_90_degrees_about_origin_passes_xy_plane_l80_80561

noncomputable def initial_vector : ℝ^4 := ![2, 1, 3, 1]

def rotated_vector (v : ℝ^4) : Prop :=
  ∃ (x y z w : ℝ), v = ![x, y, z, w] ∧
                     (2 * x + y + 3 * z + w = 0) ∧
                     (x^2 + y^2 + z^2 + w^2 = 15) ∧
                     (z = 0) ∧ (w = -3 * x) ∧
                     (x = real.sqrt (15 / 11)) ∧
                     (y = real.sqrt (15 / 11))

theorem rotated_vector_90_degrees_about_origin_passes_xy_plane :
  rotated_vector ![real.sqrt (15 / 11), real.sqrt (15 / 11), 0, -3 * real.sqrt (15 / 11)] :=
sorry

end rotated_vector_90_degrees_about_origin_passes_xy_plane_l80_80561


namespace total_toothpicks_in_grid_l80_80000

theorem total_toothpicks_in_grid (l w : ℕ) (h₁ : l = 50) (h₂ : w = 20) : 
  (l + 1) * w + (w + 1) * l + 2 * (l * w) = 4070 :=
by
  sorry

end total_toothpicks_in_grid_l80_80000


namespace angle_bisector_length_l80_80560

theorem angle_bisector_length (a b : ℝ) (θ : ℝ) (ha : a = 10) (hb : b = 15) :
  (2 * a * b * Real.cos(θ / 2) / (a + b)) < 12 :=
by
  simp [ha, hb]
  have h_cos : Real.cos(θ / 2) ≤ 1 := Real.cos_le_one (θ / 2)
  linarith [h_cos]

end angle_bisector_length_l80_80560


namespace product_comparison_l80_80906

/-
  Let \(A\) be the following product:
  \[
  A = \frac{100}{101} \times \frac{102}{103} \times \ldots \times \frac{2020}{2021} \times \frac{2022}{2023}
  \]
  Prove that \( A \) is less than \( \frac{5}{16} \) given that for all \( n \geq 100 \),
  \[
  \frac{n}{n+1} < \frac{n+2}{n+3}
  \]
-/

theorem product_comparison : 
  (A : ℝ) (hA : A = (List.range' 100 922).prod (λ n => (n * 2 + 100) / (n * 2 + 101))) :
  (∀ n : ℕ, n ≥ 100 → (n : ℝ) / (n + 1 : ℝ) < (n + 2 : ℝ) / (n + 3 : ℝ)) →
  A < 5 / 16 :=
sorry

end product_comparison_l80_80906


namespace senior_citizen_tickets_l80_80005

theorem senior_citizen_tickets (A S : ℕ) 
  (h1 : A + S = 510) 
  (h2 : 21 * A + 15 * S = 8748) : 
  S = 327 :=
by 
  -- Proof steps are omitted as instructed
  sorry

end senior_citizen_tickets_l80_80005


namespace infinite_series_sum_l80_80228

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80228


namespace not_necessarily_true_l80_80966

variable {a b c : ℝ}

def conditions : Prop := c < b ∧ b < a ∧ a * c < 0

theorem not_necessarily_true (h : conditions) : ¬(cb^2 < ca^2) :=
  sorry

end not_necessarily_true_l80_80966


namespace area_of_trapezoid_l80_80453

theorem area_of_trapezoid (PQ PR: ℝ) (PQR_area smallest_triangle_area: ℝ)
  (num_smallest_triangles: ℕ) (area_PST: ℝ) :
  PQ = PR →
  smallest_triangle_area = 2 →
  num_smallest_triangles = 10 →
  PQR_area = 80 →
  area_PST = 3 * smallest_triangle_area →
  (PQR_area - area_PST = 74) :=
begin
  intros h1 h2 h3 h4 h5,
  rw h2 at h5,
  rw h4,
  rw h5,
  norm_num,
end

end area_of_trapezoid_l80_80453


namespace sum_factors_36_eq_91_l80_80607

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80607


namespace cost_of_each_pair_of_shorts_l80_80517

variable (C : ℝ)
variable (h_discount : 3 * C - 2.7 * C = 3)

theorem cost_of_each_pair_of_shorts : C = 10 :=
by 
  sorry

end cost_of_each_pair_of_shorts_l80_80517


namespace sum_factors_36_eq_91_l80_80608

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80608


namespace base_5_division_quotient_l80_80298

theorem base_5_division_quotient :
  base_to_nat 5 [2, 4, 3, 4, 2] / base_to_nat 5 [2, 3] = base_to_nat 5 [4, 3] :=
sorry

end base_5_division_quotient_l80_80298


namespace sum_of_factors_36_l80_80812

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80812


namespace sum_infinite_series_l80_80199

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80199


namespace sin_double_angle_l80_80419

open Real

theorem sin_double_angle (θ : ℝ) (h : cos (π / 4 - θ) = 1 / 2) : sin (2 * θ) = -1 / 2 := 
by 
  sorry

end sin_double_angle_l80_80419


namespace fixed_point_exists_l80_80337

theorem fixed_point_exists :
  ∀ (m : ℝ), ∃ (x y : ℝ), (x^2 + y^2 - 2mx - 4my + 6m - 2 = 0) ∧
  ((x = 1 ∧ y = 1) ∨ (x = 1/5 ∧ y = 7/5)) := 
sorry

end fixed_point_exists_l80_80337


namespace sum_series_eq_four_ninths_l80_80201

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80201


namespace find_n_int_sin_eq_cos_l80_80933

theorem find_n_int_sin_eq_cos (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) : 
  (sin (n : ℝ) * π / 180 = cos (810 * π / 180)) ↔ (n = -180 ∨ n = 0 ∨ n = 180) := 
by
  sorry

end find_n_int_sin_eq_cos_l80_80933


namespace infinite_series_sum_l80_80233

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80233


namespace circle_radius_triple_area_l80_80522

/-- Given the area of a circle is tripled when its radius r is increased by n, prove that 
    r = n * (sqrt(3) - 1) / 2 -/
theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 1) / 2 :=
sorry

end circle_radius_triple_area_l80_80522


namespace sum_series_eq_4_div_9_l80_80288

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80288


namespace remainder_of_sum_of_binomials_l80_80546

open Nat

theorem remainder_of_sum_of_binomials (hprime : Prime 2023) :
    (∑ k in Finset.range 65, binomial 2020 k) % 2023 = 1089 := sorry

end remainder_of_sum_of_binomials_l80_80546


namespace f_behavior_on_interval_l80_80984

theorem f_behavior_on_interval :
  (∀ x, f (-x) = -f x) →
  (∀ x > 0, f x = x^2 - 4 * x + 2) →
  (∀ x ∈ set.Icc (-4 : ℝ) (-2 : ℝ), ∃ y ∈ set.Icc (-4 : ℝ) (-2 : ℝ), 
    f x = y^2 - 4 * y + 2) →
  (∀ x, ∀ y ∈ set.Icc (-4 : ℝ) (-2 : ℝ), (y > x → f y > f x)) ∧ 
  (∃ x ∈ set.Icc (-4 : ℝ) (-2 : ℝ), f x = 2) :=
sorry

end f_behavior_on_interval_l80_80984


namespace exists_three_numbers_sum_not_exists_three_numbers_sum_1001_l80_80348

theorem exists_three_numbers_sum (A : Finset ℕ) (h_card : A.card = 1002) (h_ub : ∀ a ∈ A, a ≤ 2000) :
  ∃ a b c ∈ A, a + b = c ∧ a ≠ b :=
sorry

theorem not_exists_three_numbers_sum_1001 (A : Finset ℕ) (h_card : A.card = 1001) (h_ub : ∀ a ∈ A, a ≤ 2000) :
  ¬ ∃ a b c ∈ A, a + b = c ∧ a ≠ b :=
begin
  -- Construct a counterexample as described in the solution
  let B := Finset.range' 1000 1001,
  have hB_card : B.card = 1001 := by simp,
  have hB_ub : ∀ b ∈ B, b ≤ 2000 := by {
    intros b hb,
    rw Finset.mem_range' at hb,
    linarith,
  },
  use B,
  split; assumption,
end

end exists_three_numbers_sum_not_exists_three_numbers_sum_1001_l80_80348


namespace complex_number_coordinates_l80_80452

theorem complex_number_coordinates :
  ∃ z : ℂ, z = (⟨-1, 1⟩ : ℂ) ∧ z = (2 * complex.I) / (1 - complex.I) :=
by
  sorry

end complex_number_coordinates_l80_80452


namespace gcd_of_1230_and_990_l80_80313

theorem gcd_of_1230_and_990 : Nat.gcd 1230 990 = 30 :=
by
  sorry

end gcd_of_1230_and_990_l80_80313


namespace washing_whiteboards_l80_80949

/-- Define the conditions from the problem:
1. Four kids can wash three whiteboards in 20 minutes.
2. It takes one kid 160 minutes to wash a certain number of whiteboards. -/
def four_kids_wash_in_20_min : ℕ := 3
def time_per_batch : ℕ := 20
def one_kid_time : ℕ := 160
def intervals : ℕ := one_kid_time / time_per_batch

/-- Proving the answer based on the conditions:
one kid can wash six whiteboards in 160 minutes given these conditions. -/
theorem washing_whiteboards : intervals * (four_kids_wash_in_20_min / 4) = 6 :=
by
  sorry

end washing_whiteboards_l80_80949


namespace mortgage_loan_amount_l80_80409

/-- Given the initial payment is 1,800,000 rubles and it represents 30% of the property cost C, 
    prove that the mortgage loan amount is 4,200,000 rubles. -/
theorem mortgage_loan_amount (C : ℝ) (h : 0.3 * C = 1800000) : C - 1800000 = 4200000 :=
by
  sorry

end mortgage_loan_amount_l80_80409


namespace sum_series_l80_80254

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80254


namespace find_n_l80_80940

theorem find_n : ∃ n : ℤ, (-90 ≤ n ∧ n ≤ 90) ∧ sin (n : ℝ * (Real.pi / 180)) = cos (390 * (Real.pi / 180)) :=
by
  use 60
  sorry

end find_n_l80_80940


namespace sum_of_factors_36_l80_80712

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80712


namespace regular_polygon_inscribed_circle_area_l80_80080

theorem regular_polygon_inscribed_circle_area
  (n : ℕ) (R : ℝ) (hR : R ≠ 0) (h_area : (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2) :
  n = 20 :=
by 
  sorry

end regular_polygon_inscribed_circle_area_l80_80080


namespace identify_digits_l80_80835

theorem identify_digits 
  (A C E G I O R T U V : ℕ)
  (hC: C = 1)
  (hU: U = 7)
  (hA: A = 0)
  (hT: T = 5)
  (hR: R = 4)
  (hO: O = 9)
  (hI: I = 8)
  (hN: N = 2)
  (hE: E = 6)
  (hValidDigits: A ≠ C ∧ A ≠ E ∧ A ≠ G ∧ A ≠ I ∧ A ≠ N ∧ A ≠ O ∧ A ≠ R ∧ A ≠ T ∧ A ≠ U ∧ A ≠ V
    ∧ C ≠ E ∧ C ≠ G ∧ C ≠ I ∧ C ≠ N ∧ C ≠ O ∧ C ≠ R ∧ C ≠ T ∧ C ≠ U ∧ C ≠ V
    ∧ E ≠ G ∧ E ≠ I ∧ E ≠ N ∧ E ≠ O ∧ E ≠ R ∧ E ≠ T ∧ E ≠ U ∧ E ≠ V
    ∧ G ≠ I ∧ G ≠ N ∧ G ≠ O ∧ G ≠ R ∧ G ≠ T ∧ G ≠ U ∧ G ≠ V
    ∧ I ≠ N ∧ I ≠ O ∧ I ≠ R ∧ I ≠ T ∧ I ≠ U ∧ I ≠ V
    ∧ N ≠ O ∧ N ≠ R ∧ N ≠ T ∧ N ≠ U ∧ N ≠ V
    ∧ O ≠ R ∧ O ≠ T ∧ O ≠ U ∧ O ≠ V
    ∧ R ≠ T ∧ R ≠ U ∧ R ≠ V
    ∧ T ≠ U ∧ T ≠ V
    ∧ U ≠ V)
  (hDigitsRange:  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ E ∧ E ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9
    ∧ 0 ≤ I ∧ I ≤ 9 ∧ 0 ≤ N ∧ N ≤ 9 ∧ 0 ≤ O ∧ O ≤ 9 ∧ 0 ≤ R ∧ R ≤ 9 
    ∧ 0 ≤ T ∧ T ≤ 9 ∧ 0 ≤ U ∧ U ≤ 9 ∧ 0 ≤ V ∧ V ≤ 9)
  (hSum:  (C * 100000 + U * 10000 + A * 1000 + T * 100 + R * 10 + O) +
            (C * 100000 + U * 10000 + A * 1000 + T * 100 + R * 10 + O) +
            (C * 100000 + U * 10000 + A * 1000 + A * 100 + T * 10 + R * 1 + O * 0) +
            (C * 100000 + U * 10000 + A * 1000 + A * 100 + T * 10 + R * 1 + O * 0) +
            (C * 100000 + U * 10000 + A * 1000 + T * 100 + R * 10 + O * 0)
            = C * 100000 + U * 10000 + I * 1000 + N * 100 + T * 10 + E * 1) :
  ∃ A C E G I N O R T U V, true := 
  sorry

end identify_digits_l80_80835


namespace infinite_series_sum_l80_80267

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l80_80267


namespace sum_of_factors_36_eq_91_l80_80744

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80744


namespace sum_slope_y_intercept_BM_is_negative_four_thirds_l80_80001

-- Definitions for vertices and midpoint
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 8⟩
def B : Point := ⟨2, 0⟩
def C : Point := ⟨10, 0⟩

def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def M : Point := midpoint A C

-- Definitions for slope and y-intercept calculation
def slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

def line_through (P Q : Point) (x : ℝ) : ℝ :=
  slope P Q * (x - P.x) + P.y

def y_intercept (P Q : Point) : ℝ :=
  line_through P Q 0

def sum_slope_y_intercept (P Q : Point) : ℝ :=
  slope P Q + y_intercept P Q

-- Problem statement
theorem sum_slope_y_intercept_BM_is_negative_four_thirds :
  sum_slope_y_intercept B M = -4 / 3 :=
sorry

end sum_slope_y_intercept_BM_is_negative_four_thirds_l80_80001


namespace sum_of_reciprocals_less_than_two_l80_80563

open Nat

theorem sum_of_reciprocals_less_than_two 
  (n : ℕ) 
  (a : Fin n → ℕ) 
  (h_positive : ∀ i, 0 < a i) 
  (h_less_than_1951 : ∀ i, a i < 1951) 
  (h_lcm_greater_than_1951 : ∀ i j, i ≠ j → lcm (a i) (a j) > 1951) : 
  (∑ i, (1 : ℚ) / a i) < 2 := 
by 
  sorry

end sum_of_reciprocals_less_than_two_l80_80563


namespace series_sum_eq_four_ninths_l80_80240

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80240


namespace sum_of_factors_of_36_l80_80675

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80675


namespace original_set_cardinality_l80_80491

-- Definitions based on conditions
def is_reversed_error (n : ℕ) : Prop :=
  ∃ (A B C : ℕ), 100 * A + 10 * B + C = n ∧ 100 * C + 10 * B + A = n + 198 ∧ C - A = 2

-- The theorem to prove
theorem original_set_cardinality : ∃ n : ℕ, is_reversed_error n ∧ n = 10 := by
  sorry

end original_set_cardinality_l80_80491


namespace derek_april_savings_l80_80139

theorem derek_april_savings :
  ∀ (savings : ℕ → ℕ),
  savings 0 = 2 ∧ 
  savings 1 = 4 ∧ 
  savings 2 = 8 ∧ 
  ∀ n, savings (n + 1) = savings n * 2 → 
  savings 3 = 16 :=
by {
  assume savings,
  assume h0 : savings 0 = 2,
  assume h1 : savings 1 = 4,
  assume h2 : savings 2 = 8,
  assume pattern : ∀ n, savings (n + 1) = savings n * 2,
  sorry
}

end derek_april_savings_l80_80139


namespace sum_factors_36_l80_80797

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80797


namespace find_common_difference_l80_80448

noncomputable def arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (d : ℝ) (a1 a2 a5 : ℝ) : Prop := 
a 1 = a1 ∧ a 2 = a2 ∧ a 5 = a5 ∧ 
(∀ n, a (n + 1) = a n + d) ∧ 
a 1 + a 2 + a 5 = 13 ∧ 
(a 1 + (a 2 - a 1)) ^ 2 = a 1 * (a 1 + 4 * (a 2 - a 1)) 

theorem find_common_difference :
  ∃ d : ℝ, ∀ a : ℝ → ℝ, 
    ∃ a1 a2 a5 : ℝ, 
      arithmetic_sequence_common_difference a d a1 a2 a5 → 
      d = 2 :=
begin
  sorry
end

end find_common_difference_l80_80448


namespace state_tax_deduction_equals_100_cents_per_hour_l80_80895

-- Define the given conditions
def hourlyWageInCents: ℕ := 2500  -- 25 dollars per hour in cents
def percentageTaxRate: ℚ := 2 / 100  -- 2% tax rate
def fixedTaxInCents: ℕ := 50  -- fixed tax in cents per hour

-- Define the theorem to prove the total tax in cents per hour
theorem state_tax_deduction_equals_100_cents_per_hour:
  let percentageTaxDeduction := (percentageTaxRate * hourlyWageInCents : ℚ).natAbs in
  let totalTax := percentageTaxDeduction + fixedTaxInCents in
  totalTax = 100 :=
by
  -- Placeholder proof
  sorry

end state_tax_deduction_equals_100_cents_per_hour_l80_80895


namespace effective_simple_interest_rate_proof_l80_80045

noncomputable def effective_simple_interest_rate : ℝ :=
  let P := 1
  let r1 := 0.10 / 2 -- Half-yearly rate for year 1
  let t1 := 2 -- number of compounding periods semi-annual
  let A1 := P * (1 + r1) ^ t1

  let r2 := 0.12 / 2 -- Half-yearly rate for year 2
  let t2 := 2
  let A2 := A1 * (1 + r2) ^ t2

  let r3 := 0.14 / 2 -- Half-yearly rate for year 3
  let t3 := 2
  let A3 := A2 * (1 + r3) ^ t3

  let r4 := 0.16 / 2 -- Half-yearly rate for year 4
  let t4 := 2
  let A4 := A3 * (1 + r4) ^ t4

  let CI := 993
  let P_actual := CI / (A4 - P)
  let effective_simple_interest := (CI / P_actual) * 100
  effective_simple_interest

theorem effective_simple_interest_rate_proof :
  effective_simple_interest_rate = 65.48 := by
  sorry

end effective_simple_interest_rate_proof_l80_80045


namespace expected_absolute_difference_after_10_days_l80_80030

def cat_fox_wealth_difference : ℝ := 1

theorem expected_absolute_difference_after_10_days :
  let p_cat_wins := 0.25
  let p_fox_wins := 0.25
  let p_both_police := 0.5
  let num_days := 10
  ∃ (X : ℕ → ℕ), 
    (X 0 = 0) ∧
    ∀ n, (X (n + 1) = (if (X n = 0) then 0.5 else 0) * X n) →
    (∑ k in range (num_days + 1), (k : ℝ) * (0.5 ^ k)) = cat_fox_wealth_difference := 
sorry

end expected_absolute_difference_after_10_days_l80_80030


namespace average_of_x_y_l80_80007

def x : ℝ := 0.4
def y : ℝ := 0.005

theorem average_of_x_y : (x + y) / 2 = 0.2025 := 
by
  rw [x, y]
  norm_num
  sorry

end average_of_x_y_l80_80007


namespace circles_intersect_l80_80550

-- Definitions of the circles and their properties
def circle1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }
def circle2 := { p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 4)^2 = 16 }

def center1 := (0, 0) : ℝ × ℝ
def center2 := (-3, 4) : ℝ × ℝ

def radius1 := 2
def radius2 := 4

def distance_centers := (5 : ℝ)

-- Proof statement that the positional relationship between the circles is they intersect
theorem circles_intersect
  (c1_eq : circle1 = { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 })
  (c2_eq : circle2 = { p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 4)^2 = 16 })
  (center1_def: center1 = (0, 0))
  (center2_def: center2 = (-3, 4))
  (radius1_def: radius1 = 2)
  (radius2_def: radius2 = 4)
  (distance_def: distance_centers = 5) :
  distance_centers < radius1 + radius2 ∧ distance_centers > abs (radius1 - radius2) :=
by {
  sorry
}

end circles_intersect_l80_80550


namespace evaluate_series_sum_l80_80275

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80275


namespace sum_of_money_l80_80469

theorem sum_of_money (jimin_100_won : ℕ) (jimin_50_won : ℕ) (seokjin_100_won : ℕ) (seokjin_10_won : ℕ) 
  (h1 : jimin_100_won = 5) (h2 : jimin_50_won = 1) (h3 : seokjin_100_won = 2) (h4 : seokjin_10_won = 7) :
  jimin_100_won * 100 + jimin_50_won * 50 + seokjin_100_won * 100 + seokjin_10_won * 10 = 820 :=
by
  sorry

end sum_of_money_l80_80469


namespace square_tile_area_l80_80151

-- Definition and statement of the problem
theorem square_tile_area (side_length : ℝ) (h : side_length = 7) : 
  (side_length * side_length) = 49 :=
by
  sorry

end square_tile_area_l80_80151


namespace analytical_expression_of_odd_function_l80_80373

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2 * x + 3 
else if x = 0 then 0 
else -x^2 - 2 * x - 3

theorem analytical_expression_of_odd_function :
  ∀ x : ℝ, f x =
    if x > 0 then x^2 - 2 * x + 3 
    else if x = 0 then 0 
    else -x^2 - 2 * x - 3 :=
by
  sorry

end analytical_expression_of_odd_function_l80_80373


namespace probability_two_dice_divisible_by_5_l80_80013

-- Defining the sides of the 10-sided dice
def DieSides : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Defining the event that the die roll is divisible by 5
def DivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

-- Probability Mass Function for a fair 10-sided die
def DiePMF : PMF ℕ := PMF.ofFinsetUniform DieSides (by simp)

-- Probability that a number is divisible by 5 in the context of the PMF
def ProbabilityDivisibleBy5 := DiePMF.prob {n | DivisibleBy5 n}

theorem probability_two_dice_divisible_by_5 :
  ProbabilityDivisibleBy5 * ProbabilityDivisibleBy5 = 1 / 25 := by
sorry

end probability_two_dice_divisible_by_5_l80_80013


namespace sum_of_factors_36_l80_80816

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80816


namespace maximum_subset_size_not_divisible_by_seven_l80_80864

theorem maximum_subset_size_not_divisible_by_seven :
  ∃ S : Finset ℕ, S ⊆ (Finset.range 51).filter (λ n, n > 0) ∧ (∀ x y ∈ S, x ≠ y → (x + y) % 7 ≠ 0) ∧ S.card = 22 :=
sorry

end maximum_subset_size_not_divisible_by_seven_l80_80864


namespace cost_of_largest_pot_l80_80833

theorem cost_of_largest_pot
    (x : ℝ)
    (hx : 6 * x + (0.1 + 0.2 + 0.3 + 0.4 + 0.5) = 8.25) :
    (x + 0.5) = 1.625 :=
sorry

end cost_of_largest_pot_l80_80833


namespace sum_factors_of_36_l80_80650

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80650


namespace cat_fox_wealth_difference_l80_80017

noncomputable def prob_coin_toss : ℕ := ((1/4 : ℝ) + (1/4 : ℝ) - (1/2 : ℝ))

-- define the random variable X_n representing the "absolute difference in wealth at end of n-th day"
noncomputable def X (n : ℕ) : ℝ := sorry

-- statement of the proof problem
theorem cat_fox_wealth_difference : ∃ E : ℝ, E = 1 ∧ E = classical.some X 10 := 
sorry

end cat_fox_wealth_difference_l80_80017


namespace sum_of_factors_36_l80_80695

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80695


namespace cube_traversal_count_l80_80909

-- Defining the cube traversal problem
def cube_traversal (num_faces : ℕ) (adj_faces : ℕ) (visits : ℕ) : ℕ :=
  if (num_faces = 6 ∧ adj_faces = 4) then
    4 * 2
  else
    0

-- Theorem statement
theorem cube_traversal_count : 
  cube_traversal 6 4 1 = 8 :=
by
  -- Skipping the proof with sorry for now
  sorry

end cube_traversal_count_l80_80909


namespace candle_height_relation_l80_80873

theorem candle_height_relation (t : ℕ) (h : ℝ)
  (h₀ : h = 30 - (1/2)*t) :
  ∀ t, ∃ h, h = 30 - (1/2)*t :=
by
  assume t,
  existsi (30 - (1/2)*t),
  exact h₀

end candle_height_relation_l80_80873


namespace series_sum_eq_four_ninths_l80_80243

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80243


namespace polynomial_k_correct_l80_80047

noncomputable def find_k (roots : List ℝ) : ℕ :=
  let a := roots[0]
  let a' := roots[1]
  let b := roots[2]
  let b' := roots[3]
  let aa' := a * a'
  let bb' := b * b'
  if aa' = -32 ∧ a + a' + b + b' = 18 ∧ aa' * bb' = -1984 then
    - 32 + 62 + 4 * 14
  else
    0

theorem polynomial_k_correct :
  ∃ (roots : List ℝ), (roots.length = 4 ∧ 
  let a := roots[0] in 
  let a' := roots[1] in 
  let b := roots[2] in 
  let b' := roots[3] in 
  a * a' = -32 ∧ a + a' + b + b' = 18 ∧
  a * a' * b * b' = -1984) → 
  find_k roots = 86 := by
  sorry

end polynomial_k_correct_l80_80047


namespace dot_product_value_l80_80397

def vector_a (k : ℝ) : ℝ × ℝ := (1, k)
def vector_b : ℝ × ℝ := (2, 2)

def colinear (a b : ℝ × ℝ) : Prop :=
  ∃ (λ : ℝ), a = (λ * b.1, λ * b.2)

theorem dot_product_value (k : ℝ) (h : colinear (vector_a k) (vector_b)) :
  let a := (1, k)
  let b := (2, 2)
  k = 1 → a.1 * b.1 + a.2 * b.2 = 4 :=
by
  intros a b h k_eq
  dsimp at a b
  sorry

end dot_product_value_l80_80397


namespace sum_of_factors_36_l80_80717

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80717


namespace sum_of_factors_36_l80_80736

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80736


namespace find_w_plus_one_l80_80427

theorem find_w_plus_one (w : ℝ) (h : (w + 15)^2 = (4w + 9) * (3w + 6)) :
  w + 1 = 4.1 ∨ w + 1 = -4.0 := by
  sorry

end find_w_plus_one_l80_80427


namespace groceries_delivered_l80_80097

variables (S C P g T G : ℝ)
theorem groceries_delivered (hS : S = 14500) (hC : C = 14600) (hP : P = 1.5) (hg : g = 0.05) (hT : T = 40) :
  G = 800 :=
by {
  sorry
}

end groceries_delivered_l80_80097


namespace sum_factors_of_36_l80_80660

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80660


namespace sum_of_factors_36_l80_80681

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80681


namespace intersection_correct_l80_80393

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℤ := {-2, 0, 2}

theorem intersection_correct : M ∩ N = {0, 2} := 
  sorry

end intersection_correct_l80_80393


namespace sum_of_factors_36_l80_80619

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80619


namespace sum_series_l80_80260

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80260


namespace sum_of_series_l80_80173

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80173


namespace infinite_perfect_squares_in_arithmetic_sequence_l80_80501

theorem infinite_perfect_squares_in_arithmetic_sequence 
  (a d : ℕ) 
  (h_exists_perfect_square : ∃ (n₀ k : ℕ), a + n₀ * d = k^2) 
  : ∃ (S : ℕ → ℕ), (∀ n, ∃ t, S n = a + t * d ∧ ∃ k, S n = k^2) ∧ (∀ m n, S m = S n → m = n) :=
sorry

end infinite_perfect_squares_in_arithmetic_sequence_l80_80501


namespace max_A_value_l80_80336

-- Variables
variables {x1 x2 x3 y1 y2 y3 z1 z2 z3 : ℝ}

-- Assumptions
axiom pos_x1 : 0 < x1
axiom pos_x2 : 0 < x2
axiom pos_x3 : 0 < x3
axiom pos_y1 : 0 < y1
axiom pos_y2 : 0 < y2
axiom pos_y3 : 0 < y3
axiom pos_z1 : 0 < z1
axiom pos_z2 : 0 < z2
axiom pos_z3 : 0 < z3

-- Statement
theorem max_A_value :
  ∃ A : ℝ, 
    (∀ x1 x2 x3 y1 y2 y3 z1 z2 z3, 
    (0 < x1) → (0 < x2) → (0 < x3) →
    (0 < y1) → (0 < y2) → (0 < y3) →
    (0 < z1) → (0 < z2) → (0 < z3) →
    (x1^3 + x2^3 + x3^3 + 1) * (y1^3 + y2^3 + y3^3 + 1) * (z1^3 + z2^3 + z3^3 + 1) ≥
    A * (x1 + y1 + z1) * (x2 + y2 + z2) * (x3 + y3 + z3)) ∧ 
    A = 9/2 := 
by 
  exists 9/2 
  sorry

end max_A_value_l80_80336


namespace sum_of_factors_36_l80_80689

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80689


namespace sum_of_factors_of_36_l80_80676

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80676


namespace num_ways_to_distribute_items_l80_80117

theorem num_ways_to_distribute_items : 
  let items := 5
  let bags := 4
  let distinct_items := 5
  let identical_bags := 4
  (number_of_ways_to_distribute_items_in_4_identical_bags distinct_items identical_bags = 36) := sorry

end num_ways_to_distribute_items_l80_80117


namespace yogurt_combinations_l80_80089

-- Define the conditions from a)
def num_flavors : ℕ := 5
def num_toppings : ℕ := 8
def num_sizes : ℕ := 3

-- Define the problem in a theorem statement
theorem yogurt_combinations : num_flavors * ((num_toppings * (num_toppings - 1)) / 2) * num_sizes = 420 :=
by
  -- sorry is used here to skip the proof
  sorry

end yogurt_combinations_l80_80089


namespace exists_positive_integer_n_l80_80338

theorem exists_positive_integer_n (f : ℝ → ℝ) (t : ℝ) (n : ℕ) :
  (∀ t ∈ Ioc 0 4, f t = (n-1) * t^2 - 10 * t + 10) →
  (∀ t ∈ Ioc 0 4, 0 < f t ∧ f t ≤ 30) →
  ∃ n : ℕ, ∀ t ∈ Ioc 0 4, 0 < (n-1) * t^2 - 10 * t + 10 ∧ (n-1) * t^2 - 10 * t + 10 ≤ 30 :=
by
  sorry

end exists_positive_integer_n_l80_80338


namespace sum_of_factors_36_l80_80715

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80715


namespace invalid_vote_percentage_l80_80449

theorem invalid_vote_percentage (total_votes : ℕ) (valid_votes_other_candidate : ℝ) (valid_votes_percentage_candidate1 : ℝ) 
  (valid_votes_percentage_candidate2 : ℝ) (correct_percentage_invalid_votes : ℝ) :
  total_votes = 9000 →
  valid_votes_other_candidate = 2519.9999999999995 →
  valid_votes_percentage_candidate1 = 0.60 →
  valid_votes_percentage_candidate2 = 0.40 →
  correct_percentage_invalid_votes = 30 →
  let V := valid_votes_other_candidate / valid_votes_percentage_candidate2 in
  let invalid_votes := total_votes - V in
  (invalid_votes / total_votes) * 100 = correct_percentage_invalid_votes := 
by
  intros
  let V := valid_votes_other_candidate / valid_votes_percentage_candidate2
  let invalid_votes := total_votes - V
  have h1 : (invalid_votes / total_votes) * 100 = correct_percentage_invalid_votes := sorry
  exact h1

end invalid_vote_percentage_l80_80449


namespace train_length_l80_80830

theorem train_length (speed_kmh : ℕ) (time_sec : ℕ) (length_m : ℝ) :
  speed_kmh = 56 →
  time_sec = 9 →
  length_m = (56 * 1000 / 3600) * 9 →
  length_m = 140.04 :=
by
  intros h_speed h_time h_length
  rw [h_speed, h_time]
  have speed_ms : ℝ := (56 * 1000 / 3600)
  change length_m = speed_ms * 9 at h_length
  rw [←h_length]
  norm_num
  sorry

end train_length_l80_80830


namespace area_of_polygon_is_correct_l80_80576

noncomputable def area_of_intersection_polygon (a b : ℝ) (circle : a^2 + b^2 = 16) (ellipse : (a-2)^2 + 4*b^2 = 36) : ℝ :=
  sorry

theorem area_of_polygon_is_correct : area_of_intersection_polygon _ _ _ _ = (8 * (Real.sqrt 80)) / 3 :=
by
  sorry

end area_of_polygon_is_correct_l80_80576


namespace sum_factors_36_l80_80792

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80792


namespace sum_series_eq_four_ninths_l80_80206

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80206


namespace library_visitors_equation_l80_80493

-- Variables representing conditions
variable (x : ℝ)  -- Monthly average growth rate
variable (first_month_visitors : ℝ) -- Visitors in the first month
variable (total_visitors_by_third_month : ℝ) -- Total visitors by the third month

-- Setting specific values for conditions
def first_month_visitors := 600
def total_visitors_by_third_month := 2850

-- The Lean statement that the specified equation holds
theorem library_visitors_equation :
  first_month_visitors + first_month_visitors * (1 + x) + first_month_visitors * (1 + x)^2 = total_visitors_by_third_month :=
sorry

end library_visitors_equation_l80_80493


namespace sum_factors_of_36_l80_80658

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80658


namespace vector_on_line_l80_80070

variable (p q : Vector ℝ) -- assuming vectors are from the reals

theorem vector_on_line (h : p ≠ q) (m : ℝ) (hm : m = 5/8) :
  ∃ k : ℝ, k = 3/8 ∧ (k * p + m * q) lies_on_line p q := 
begin
  sorry
end

end vector_on_line_l80_80070


namespace sum_factors_36_l80_80765

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80765


namespace sum_k_over_4k_l80_80163

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80163


namespace ordered_quadruples_count_l80_80321

noncomputable def count_ordered_quadruples : ℕ :=
  {n | ∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
     a^2 + b^2 + c^2 + d^2 = 9 ∧
     (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81 ∧
     n = 15}.card

theorem ordered_quadruples_count : count_ordered_quadruples = 15 := sorry

end ordered_quadruples_count_l80_80321


namespace fraction_product_l80_80128

theorem fraction_product :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := 
by
  -- Detailed proof steps would go here
  sorry

end fraction_product_l80_80128


namespace sum_factors_of_36_l80_80655

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80655


namespace fractional_product_l80_80123

theorem fractional_product :
  ((3/4) * (4/5) * (5/6) * (6/7) * (7/8)) = 3/8 :=
by
  sorry

end fractional_product_l80_80123


namespace sum_infinite_series_l80_80195

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80195


namespace circle_in_square_radius_l80_80084

noncomputable def radius_of_circle_in_quadrilateral 
  (side_length : ℝ) (M_is_midpoint : Prop) 
  (circle_touches_AM_CD_DA : Prop) : ℝ :=
3 - real.sqrt 5

theorem circle_in_square_radius (ABCD_square : Prop)
  (side_length_eq : side_length = 2)
  (M_midpoint_of_BC : M_is_midpoint)
  (circle_touches_sides : circle_touches_AM_CD_DA)
  : radius_of_circle_in_quadrilateral side_length M_is_midpoint circle_touches_AM_CD_DA 
  = 3 - real.sqrt 5 := 
sorry

end circle_in_square_radius_l80_80084


namespace sum_of_roots_of_quadratic_l80_80580

noncomputable def sum_of_undefined_denominator_roots : ℕ := sorry

theorem sum_of_roots_of_quadratic :
  let P : ℤ → ℤ := λ y, y^2 - 6*y + 8 
  (∀ y : ℤ, P y = 0 → y = 2 ∨ y = 4) →
  sum_of_undefined_denominator_roots = 6 :=
begin
  sorry
end

end sum_of_roots_of_quadratic_l80_80580


namespace sum_factors_36_eq_91_l80_80604

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80604


namespace part1_cos_angle_part2_find_k_l80_80963

noncomputable def vector_a : ℝ × ℝ × ℝ := (3, 2, -1)
noncomputable def vector_b : ℝ × ℝ × ℝ := (2, 1, 2)

theorem part1_cos_angle :
  let dot_prod := (3 * 2) + (2 * 1) + (-1 * 2),
      mag_a := Real.sqrt (3^2 + 2^2 + (-1)^2),
      mag_b := Real.sqrt (2^2 + 1^2 + 2^2)
  in (dot_prod / (mag_a * mag_b) = 2 / Real.sqrt 14) := sorry

theorem part2_find_k (k : ℝ) :
  let a := vector_a,
      b := vector_b,
      condition := (k * a.1 + b.1) * (a.1 - k * b.1) + (k * a.2 + b.2) * (a.2 - k * b.2) + (k * a.3 + b.3) * (a.3 - k * b.3)
  in condition = 0 → (k = 3 / 2 ∨ k = -2 / 3) := sorry

end part1_cos_angle_part2_find_k_l80_80963


namespace sum_factors_36_l80_80629

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80629


namespace sum_of_factors_36_l80_80614

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80614


namespace sum_of_factors_36_l80_80677

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l80_80677


namespace braking_distance_non_alcohol_reaction_time_after_alcohol_braking_distance_after_alcohol_increase_in_braking_distance_max_reaction_time_l80_80922

section braking_distance

variables {t k v s : ℝ}

-- Problem 1
theorem braking_distance_non_alcohol: 
  (t = 0.5) ∧ (v = 10) ∧ (k = 0.1) → (s = t * v + k * v^2) → (s = 15) :=
by intros; sorry

-- Problem 2a
theorem reaction_time_after_alcohol:
  (v = 15) ∧ (s = 52.5) ∧ (k = 0.1) → (s = t * v + k * v^2) → (t = 2) :=
by intros; sorry

-- Problem 2b
theorem braking_distance_after_alcohol:
  (t = 2) ∧ (v = 10) ∧ (k = 0.1) → (s = t * v + k * v^2) → (s = 30) :=
by intros; sorry

-- Problem 2c
theorem increase_in_braking_distance:
  (s_after = 30) ∧ (s_before = 15) → (diff = s_after - s_before) → (diff = 15) :=
by intros; sorry

-- Problem 3
theorem max_reaction_time:
  (v = 12) ∧ (k = 0.1) ∧ (s ≤ 42) → (s = t * v + k * v^2) → (t ≤ 2.3) :=
by intros; sorry

end braking_distance

end braking_distance_non_alcohol_reaction_time_after_alcohol_braking_distance_after_alcohol_increase_in_braking_distance_max_reaction_time_l80_80922


namespace total_different_lines_l80_80509

noncomputable def count_different_triplets : ℕ :=
  let S := {1, 3, 5, 7, 9}
  let triplet_count := (list.product (list.product S.to_list S.to_list) S.to_list).countp (λ t, t.1.1 ≠ t.1.2 ∧ t.1.1 ≠ t.2 ∧ t.1.2 ≠ t.2)
  triplet_count

theorem total_different_lines :
  count_different_triplets = 60 :=
by
  sorry

end total_different_lines_l80_80509


namespace biggest_number_in_ratio_l80_80950

theorem biggest_number_in_ratio (x : ℕ) (h_sum : 2 * x + 3 * x + 4 * x + 5 * x = 1344) : 5 * x = 480 := 
by
  sorry

end biggest_number_in_ratio_l80_80950


namespace sum_factors_36_eq_91_l80_80606

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80606


namespace sum_factors_36_l80_80640

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80640


namespace infinite_series_sum_l80_80236

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80236


namespace smallest_fraction_numerator_l80_80878

theorem smallest_fraction_numerator (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : (a * 4) > (b * 3)) : a = 73 :=
  sorry

end smallest_fraction_numerator_l80_80878


namespace expected_value_absolute_difference_after_10_days_l80_80037

/-- Define the probability space and outcome -/
noncomputable def probability_cat_wins : ℝ := 0.25
noncomputable def probability_fox_wins : ℝ := 0.25
noncomputable def probability_both_police : ℝ := 0.50

/-- Define the random variable for absolute difference in wealth -/
noncomputable def X_n (n : ℕ) : ℝ := sorry

/-- Define the probability p_{0, n} -/
noncomputable def p (k n : ℕ) : ℝ := sorry

/-- Given the above conditions, the expected value of the absolute difference -/
theorem expected_value_absolute_difference_after_10_days : (∑ k in finset.range 11, k * p k 10) = 1 :=
sorry

end expected_value_absolute_difference_after_10_days_l80_80037


namespace sum_of_factors_36_l80_80814

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80814


namespace area_of_rhombus_l80_80049

theorem area_of_rhombus (EF FD : ℝ) (hEF : EF = 2) (hFD : FD = 1) 
  (ABCD_is_rhombus : rhombus ABCD) 
  (E : Point) (hE : intersection E AC BD) 
  (F : Point) (hF : lies_on F AD) 
  (perpendicular_EF_FD : is_perpendicular EF FD) : 
  area ABCD = 20 := sorry

end area_of_rhombus_l80_80049


namespace eval_otimes_l80_80912

def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem eval_otimes : otimes 4 2 = 18 :=
by
  sorry

end eval_otimes_l80_80912


namespace product_of_odd_integers_less_than_500_l80_80010

theorem product_of_odd_integers_less_than_500 :
  (∏ i in (finset.filter (λ x, x % 2 = 1) (finset.range 500)), (i : ℕ)) = (499! / (2^249 * 249!)) := by
  sorry

end product_of_odd_integers_less_than_500_l80_80010


namespace sum_of_factors_36_eq_91_l80_80590

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80590


namespace sum_series_equals_4_div_9_l80_80220

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80220


namespace constant_term_expansion_l80_80455

noncomputable def constant_term := 
  let T := λ (r : ℕ), (Nat.choose 10 r) * (-1)^r * x^(10-5*r) in
  let r := 2 in
  let term := T r in
  if 10 - 5 * r = 0 then term else 0

theorem constant_term_expansion :
  constant_term = 45 := 
sorry

end constant_term_expansion_l80_80455


namespace sum_of_factors_36_l80_80704

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80704


namespace groceries_delivered_l80_80095

variables (S C P g T G : ℝ)
theorem groceries_delivered (hS : S = 14500) (hC : C = 14600) (hP : P = 1.5) (hg : g = 0.05) (hT : T = 40) :
  G = 800 :=
by {
  sorry
}

end groceries_delivered_l80_80095


namespace work_completed_in_4_days_l80_80061

theorem work_completed_in_4_days (A_days B_days C_days : ℝ) (hA : A_days = 8) (hB : B_days = 12) (hC : C_days = 24) :
  let A_work_per_day := 1 / A_days in
  let B_work_per_day := 1 / B_days in
  let C_work_per_day := 1 / C_days in
  let total_work_per_day := A_work_per_day + B_work_per_day + C_work_per_day in
  total_work_per_day = 1 / 4 :=
by
  sorry

end work_completed_in_4_days_l80_80061


namespace sum_k_over_4k_l80_80154

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80154


namespace sum_of_factors_of_36_l80_80774

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80774


namespace average_speed_of_train_l80_80087

theorem average_speed_of_train (d1 d2: ℝ) (t1 t2: ℝ) (h_d1: d1 = 250) (h_d2: d2 = 350) (h_t1: t1 = 2) (h_t2: t2 = 4) :
  (d1 + d2) / (t1 + t2) = 100 := by
  sorry

end average_speed_of_train_l80_80087


namespace smallest_fraction_numerator_l80_80880

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (a * 4 > b * 3) ∧ (a = 73) := 
sorry

end smallest_fraction_numerator_l80_80880


namespace sum_of_min_max_values_l80_80968

open Real

-- Definitions used in the conditions
def f (ω x : ℝ) : ℝ := sin (ω * x) * cos (ω * x) - sqrt 3 * (cos (ω * x)) ^ 2

def y (ω x : ℝ) : ℝ := f ω x + (2 + sqrt 3) / 2

def minMaxSum (ω : ℝ) (x_interval : Set ℝ) (sum_val : ℝ) : Prop :=
  ∀ x ∈ x_interval, 
  let f_val := f ω x in
  let min_val := -sqrt 3 in
  let max_val := 1 - sqrt 3 / 2 in
  sum_val = min_val + max_val

theorem sum_of_min_max_values (ω : ℝ) (x1 x2 : ℝ)
  (hω : 0 < ω)
  (hx1_eq_zero : y ω x1 = 0)
  (hx2_eq_zero : y ω x2 = 0)
  (h_diff : x2 - x1 = π)
  (sum_val : ℝ) :
  minMaxSum ω (Set.Icc 0 (7 * π / 12)) sum_val → sum_val = (2 - 3 * sqrt 3) / 2 :=
by
  sorry

end sum_of_min_max_values_l80_80968


namespace domain_of_function_l80_80529

noncomputable def function_domain : Set ℝ :=
  { x | ∃ k : ℤ, (2 * k * Real.pi < x) ∧ (x < Real.pi / 2 + 2 * k * Real.pi) }

theorem domain_of_function :
  ∀ x : ℝ,
  (sin x ≥ 0 ∧ cos x > 0 ∧ tan x ≠ 0) ↔
  x ∈ function_domain :=
by
  sorry

end domain_of_function_l80_80529


namespace total_hunts_is_21_l80_80439

-- Define the initial conditions
def Sam_hunts : Nat := 6
def Rob_hunts : Nat := Sam_hunts / 2
def Rob_Sam_total_hunt : Nat := Sam_hunts + Rob_hunts
def Mark_hunts : Nat := Rob_Sam_total_hunt / 3
def Peter_hunts : Nat := Mark_hunts * 3

-- The main theorem to prove
theorem total_hunts_is_21 : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 :=
by
  sorry

end total_hunts_is_21_l80_80439


namespace probability_of_negative_product_l80_80003

-- Define the set of integers
def integer_set := {-5, -8, 7, 4, -2, 6, -3}

-- Define a function to calculate the number of negative integers
def num_negatives (s : Set Int) : Nat :=
  s.count (λ x => x < 0)

-- Define a function to calculate the number of positive integers
def num_positives (s : Set Int) : Nat :=
  s.count (λ x => x > 0)

-- Define a function to calculate the probability of a product being negative
noncomputable def probability_product_negative (s : Set Int) : Rat :=
  let total_pairs := (s.card * (s.card - 1)) / 2
  let neg_pos_pairs := num_negatives s * num_positives s
  neg_pos_pairs / total_pairs

-- Statement to prove
theorem probability_of_negative_product :
  probability_product_negative integer_set = 4 / 7 :=
by
  sorry

end probability_of_negative_product_l80_80003


namespace domain_f_l80_80530

noncomputable def f (x : ℝ) : ℝ := sqrt (x + 1) + 1 / x

theorem domain_f :
  {x : ℝ | x + 1 ≥ 0 ∧ x ≠ 0} = {x : ℝ | -1 ≤ x ∧ x < 0 ∨ 0 < x} :=
by
  sorry

end domain_f_l80_80530


namespace correct_choice_l80_80391

def M (x y : ℝ) : Prop := 27^x = (1/9) * 3^y

theorem correct_choice : M 1 5 :=
by 
  -- We know that:
  -- M = { (x, y) | 27^x = (1/9) * 3^y }
  -- We can simplify:
  -- 27^x = 3^(3x) and (1/9) * 3^y = 3^(-2) * 3^y = 3^(y - 2)
  -- So we have: 3^(3x) = 3^(y - 2)
  -- Hence, 3x = y - 2
  -- Therefore, 3x - y + 2 = 0
  -- We need to check if (1, 5) satisfies this equation
  have h : 3 * 1 - 5 + 2 = 0,
    calc
      3 * 1 - 5 + 2 = 3 - 5 + 2  : by ring
                  ... = 0         : by ring,
  -- Thus, (1, 5) ∈ M
  exact h

end correct_choice_l80_80391


namespace sum_k_over_4k_l80_80155

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80155


namespace coin_flip_solution_l80_80844

theorem coin_flip_solution (p : ℚ) (m n : ℕ) (h_mn_coprime : Nat.coprime m n) (hp : p > 0) (h1p : 1 - p > 0)
    (h_prob : Nat.choose 8 3 * p ^ 3 * (1 - p) ^ 5 = (1 / 25) * Nat.choose 8 5 * p ^ 5 * (1 - p) ^ 3) : m + n = 11 :=
sorry

end coin_flip_solution_l80_80844


namespace divisors_of_2_pow_48_minus_1_l80_80545

theorem divisors_of_2_pow_48_minus_1 :
  ∃ (a b : ℕ), (60 ≤ a ∧ a ≤ 70) ∧ (60 ≤ b ∧ b ≤ 70) ∧ (a ≠ b) ∧
  (a = 63 ∨ a = 65) ∧ (b = 63 ∨ b = 65) ∧ (2^48 - 1) % a = 0 ∧ (2^48 - 1) % b = 0 :=
by {
  use [63, 65],
  split,
  exact (by norm_num : 60 ≤ 63 ∧ 63 ≤ 70),
  split,
  exact (by norm_num : 60 ≤ 65 ∧ 65 ≤ 70),
  split,
  exact (by norm_num : 63 ≠ 65),
  split,
  exact or.inl rfl,
  split,
  exact or.inr rfl,
  split,
  exact (by norm_num : (2^48 - 1) % 63 = 0),
  exact (by norm_num : (2^48 - 1) % 65 = 0)
}

end divisors_of_2_pow_48_minus_1_l80_80545


namespace sum_of_factors_of_36_l80_80671

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80671


namespace max_n_inequality_l80_80360

noncomputable def a : ℕ → ℕ
| 0       := 0  
| 1       := 2
| (n + 2) := a (n + 1) + 2

noncomputable def S : ℕ → ℕ
| 0       := 0
| (n + 1) := (n + 1) * a (n + 2) - (n + 1) * (n + 2)

theorem max_n_inequality :
  ∀ n : ℕ, 0 < n → a1 = 2 → 
  (∀ n : ℕ, 0 < n → S n / n = a (n + 1) - (n + 1)) → 
  a n * S n ≤ 2200 → n ≤ 10 := 
by
  intros n hn ha1 hSn hineq
  sorry

end max_n_inequality_l80_80360


namespace boat_travel_time_downstream_l80_80827

-- Define the given conditions and statement to prove
theorem boat_travel_time_downstream (B : ℝ) (C : ℝ) (Us : ℝ) (Ds : ℝ) :
  (C = B / 4) ∧ (Us = B - C) ∧ (Ds = B + C) ∧ (Us = 3) ∧ (15 / Us = 5) ∧ (15 / Ds = 3) :=
by
  -- Provide the proof here; currently using sorry to skip the proof
  sorry

end boat_travel_time_downstream_l80_80827


namespace sum_of_factors_36_l80_80810

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80810


namespace count_empty_intersection_image_l80_80975

open Finset

-- Define the set A
def A : Finset ℕ := {1, 2, 3}

-- Define the function type from A to A
def FuncType (A : Type) := A → A

theorem count_empty_intersection_image :
  ∃ (n : ℕ), n = 42 ∧ ∀ (f g : FuncType ℕ), f '' (A : Set ℕ) ∩ g '' (A : Set ℕ) = ∅ :=
  sorry

end count_empty_intersection_image_l80_80975


namespace eraser_crayon_difference_l80_80503

def initial_crayons : Nat := 601
def initial_erasers : Nat := 406
def final_crayons : Nat := 336
def final_erasers : Nat := initial_erasers

theorem eraser_crayon_difference :
  final_erasers - final_crayons = 70 :=
by
  sorry

end eraser_crayon_difference_l80_80503


namespace sum_of_factors_36_l80_80716

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80716


namespace cat_fox_wealth_difference_l80_80018

noncomputable def prob_coin_toss : ℕ := ((1/4 : ℝ) + (1/4 : ℝ) - (1/2 : ℝ))

-- define the random variable X_n representing the "absolute difference in wealth at end of n-th day"
noncomputable def X (n : ℕ) : ℝ := sorry

-- statement of the proof problem
theorem cat_fox_wealth_difference : ∃ E : ℝ, E = 1 ∧ E = classical.some X 10 := 
sorry

end cat_fox_wealth_difference_l80_80018


namespace sum_factors_36_l80_80803

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80803


namespace sum_first_40_terms_l80_80447

-- Given: The sum of the first 10 terms of a geometric sequence is 9
axiom S_10 : ℕ → ℕ
axiom sum_S_10 : S_10 10 = 9 

-- Given: The sum of the terms from the 11th to the 20th is 36
axiom S_20 : ℕ → ℕ
axiom sum_S_20 : S_20 20 - S_10 10 = 36

-- Let Sn be the sum of the first n terms in the geometric sequence
def Sn (n : ℕ) : ℕ := sorry

-- Prove: The sum of the first 40 terms is 144
theorem sum_first_40_terms : Sn 40 = 144 := sorry

end sum_first_40_terms_l80_80447


namespace sum_factors_36_l80_80800

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80800


namespace sum_series_eq_4_div_9_l80_80184

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80184


namespace brokerage_percentage_is_0_22_l80_80008

noncomputable def brokerage_percentage (face_value : ℝ) (discount_percentage : ℝ) (cost_price : ℝ) : ℝ :=
  let discounted_price := face_value - (face_value * discount_percentage / 100)
  in (cost_price - discounted_price) / discounted_price * 100

theorem brokerage_percentage_is_0_22 (face_value discount_percentage cost_price : ℝ) : 
  face_value = 100 ∧ discount_percentage = 9 ∧ cost_price = 91.2 →
  (brokerage_percentage face_value discount_percentage cost_price) ≈ 0.22 := 
by
  sorry

end brokerage_percentage_is_0_22_l80_80008


namespace count_perfect_cubes_between_bounds_l80_80406

theorem count_perfect_cubes_between_bounds : 
  let lower_bound := 2^6 + 1
  let upper_bound := 2^{12} + 1
  ∃ n : ℕ, n = 12 ∧ 
           (∀ k : ℕ, lower_bound ≤ k^3 → k^3 ≤ upper_bound → k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10 ∨ k = 11 ∨ k = 12 ∨ k = 13 ∨ k = 14 ∨ k = 15 ∨ k = 16) :=
by
  sorry

end count_perfect_cubes_between_bounds_l80_80406


namespace smallest_c_log_expression_well_defined_l80_80919

theorem smallest_c_log_expression_well_defined :
  ∃ (c : ℝ), (∀ x : ℝ, x > c → (log 10 (log 9 (log 8 (log 7 x)))) > 0) ∧ c = 7^(8^9) :=
by
  sorry

end smallest_c_log_expression_well_defined_l80_80919


namespace sum_series_eq_four_ninths_l80_80202

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80202


namespace expected_absolute_difference_after_10_days_l80_80027

def cat_fox_wealth_difference : ℝ := 1

theorem expected_absolute_difference_after_10_days :
  let p_cat_wins := 0.25
  let p_fox_wins := 0.25
  let p_both_police := 0.5
  let num_days := 10
  ∃ (X : ℕ → ℕ), 
    (X 0 = 0) ∧
    ∀ n, (X (n + 1) = (if (X n = 0) then 0.5 else 0) * X n) →
    (∑ k in range (num_days + 1), (k : ℝ) * (0.5 ^ k)) = cat_fox_wealth_difference := 
sorry

end expected_absolute_difference_after_10_days_l80_80027


namespace sum_of_factors_of_36_l80_80668

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80668


namespace slope_of_asymptotes_l80_80394

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 2)^2 / 144 - (y + 3)^2 / 81 = 1

-- The theorem stating the slope of the asymptotes
theorem slope_of_asymptotes : ∀ x y : ℝ, hyperbola x y → (∃ m : ℝ, m = 3 / 4) :=
by
  sorry

end slope_of_asymptotes_l80_80394


namespace evaluate_series_sum_l80_80277

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80277


namespace sum_of_factors_of_36_l80_80673

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80673


namespace solve_inequality_l80_80515

theorem solve_inequality (a : ℝ) : (6 * x^2 + a * x - a^2 < 0) ↔
  ((a > 0) ∧ (-a / 2 < x ∧ x < a / 3)) ∨
  ((a < 0) ∧ (a / 3 < x ∧ x < -a / 2)) ∨
  ((a = 0) ∧ false) :=
by 
  sorry

end solve_inequality_l80_80515


namespace inverse_function_ratio_l80_80137

def f (x : ℝ) : ℝ := (3 * x - 2) / (2 * x + 4)

noncomputable def f_inv (x : ℝ) : ℝ := (-4 * x - 2) / (2 * x - 3)

theorem inverse_function_ratio (a b c d : ℝ) 
  (h : ∀ x : ℝ, f_inv (f x) = x) 
  (ha : a = -4) 
  (hc : c = 2) : 
  a / c = -2 :=
by 
  rw [ha, hc]
  norm_num

end inverse_function_ratio_l80_80137


namespace area_ratio_trapezoids_l80_80565

/-- 
Given trapezoid ABCD formed by three congruent isosceles triangles DAO, AOB, and OBC, 
where AD = AO = OB = BC = 13 and AB = DO = OC = 15, 
with points X and Y as midpoints of AD and BC respectively, 
prove that the simplified ratio of the areas of trapezoid ABYX to trapezoid XYCD is 1:1, 
and hence the sum p+q of the ratio p:q is 2.
 -/
theorem area_ratio_trapezoids (AD AO OB BC AB DO OC : ℝ) (X Y : ℝ) (ratio p q : ℕ) 
  (hcong : AD = AO ∧ AO = OB ∧ OB = BC ∧ BC = 13) 
  (hparallel : DO = OC ∧ OC = AB ∧ AB = 15)
  (hmidX : X = (AD / 2)) (hmidY : Y = (BC / 2))
  (hxy : X = Y)
  (hratio : ratio = (1:1))
  : (p + q = 2) := 
sorry

end area_ratio_trapezoids_l80_80565


namespace sara_rent_amount_l80_80495

def ticketPrice : Real := 10.62
def numTickets : Int := 2
def boughtMovieCost : Real := 13.95
def totalSpent : Real := 36.78

def spentOnTickets : Real := ticketPrice * numTickets.toReal
def spentOnBoughtMovie : Real := boughtMovieCost
def spentOnMoviesWithoutRent : Real := spentOnTickets + spentOnBoughtMovie
def rentAmount : Real := totalSpent - spentOnMoviesWithoutRent

theorem sara_rent_amount : rentAmount = 1.59 :=
  by
    -- proof steps will go here
    sorry

end sara_rent_amount_l80_80495


namespace smallest_fraction_numerator_l80_80877

theorem smallest_fraction_numerator (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : (a * 4) > (b * 3)) : a = 73 :=
  sorry

end smallest_fraction_numerator_l80_80877


namespace gcd_1230_990_l80_80310

theorem gcd_1230_990 : Int.gcd 1230 990 = 30 := by
  sorry

end gcd_1230_990_l80_80310


namespace geometric_arithmetic_sequence_l80_80982

theorem geometric_arithmetic_sequence 
  (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (q : ℝ) 
  (h0 : 0 < q) (h1 : q ≠ 1)
  (h2 : ∀ n, a_n n = a_n 1 * q ^ (n - 1)) -- a_n is a geometric sequence
  (h3 : 2 * a_n 3 * a_n 5 = a_n 4 * (a_n 3 + a_n 5)) -- a3, a5, a4 form an arithmetic sequence
  (h4 : ∀ n, S n = a_n 1 * (1 - q^n) / (1 - q)) -- S_n is the sum of the first n terms
  : S 6 / S 3 = 9 / 8 :=
by
  sorry

end geometric_arithmetic_sequence_l80_80982


namespace total_hunts_is_21_l80_80438

-- Define the initial conditions
def Sam_hunts : Nat := 6
def Rob_hunts : Nat := Sam_hunts / 2
def Rob_Sam_total_hunt : Nat := Sam_hunts + Rob_hunts
def Mark_hunts : Nat := Rob_Sam_total_hunt / 3
def Peter_hunts : Nat := Mark_hunts * 3

-- The main theorem to prove
theorem total_hunts_is_21 : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 :=
by
  sorry

end total_hunts_is_21_l80_80438


namespace sum_k_over_4k_l80_80157

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80157


namespace Tn_max_value_exists_x_star_l80_80839

-- Part (1):
theorem Tn_max_value (T : ℝ → ℝ) (n : ℕ) (x : ℝ)
  (h_poly : ∀ x, T x = (1 / 2^n) * ((x + (sqrt (1 - x^2)) * I)^n + (x - (sqrt (1 - x^2)) * I)^n))
  (h_interval : -1 ≤ x ∧ x ≤ 1)
  (h_leading_coefficient : ∀ x, leading_coeff (T.polynomial) = 1) :
  ∃ x_max ∈ set.Icc (-1 : ℝ) (1 : ℝ), T x_max = 1 / 2^(n-1) :=
sorry


-- Part (2):
theorem exists_x_star (p : ℝ → ℝ) (n : ℕ)
  (h_poly : p = (λ x, x^n + ∑ i in finset.range n, a i * x^i))
  (h_leading_coefficient : ∀ x, a n = 1)
  (h_bounds : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), p x > -1 / 2^(n-1)) :
  ∃ x_star ∈ set.Icc (-1 : ℝ) (1 : ℝ), p x_star ≥ 1 / 2^(n-1) :=
sorry

end Tn_max_value_exists_x_star_l80_80839


namespace sum_of_factors_36_eq_91_l80_80749

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80749


namespace sum_series_l80_80250

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80250


namespace AE_length_correct_l80_80064

noncomputable def length_of_AE : ℝ :=
let radius_A := 10 in
let radius_B := 4 in
-- distance between centers A and B
let AB := radius_A + radius_B in
let AD := radius_A in
let DE := 4 * real.sqrt 10 in
real.sqrt (AD^2 + DE^2)

theorem AE_length_correct :
  length_of_AE = 2 * real.sqrt 65 :=
by
  -- The full segment can be computed as described in the full solution
  unfold length_of_AE
  simp -- Ensure the definition is unfolded for computation
  sorry -- Calculation steps go here

end AE_length_correct_l80_80064


namespace fourth_derivative_l80_80307

noncomputable def y (x : ℝ) : ℝ :=
  Real.exp (-x) * (Real.cos (2 * x) - 3 * Real.sin (2 * x))

theorem fourth_derivative (x : ℝ) :
  (iterated_deriv 4 y) x = -Real.exp (-x) * (79 * Real.cos (2 * x) + 3 * Real.sin (2 * x)) :=
by
  sorry

end fourth_derivative_l80_80307


namespace sum_factors_36_l80_80634

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80634


namespace sum_of_factors_36_l80_80818

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80818


namespace possible_to_cut_hole_l80_80464

-- Define the conditions and the question
def can_cut_hole (sheet : Type) (person : Type) : Prop :=
  ∃ (cutting_process : sheet → sheet), 
  (cutting_process ⟨sheet⟩) 
  ∧ (folding_and_cutting_steps cutting_process)
  ∧ (maximized_edge_length cutting_process)
  ∧ (person_can_fit_through_hole cutting_process person)

-- State the theorem to be proven
theorem possible_to_cut_hole : can_cut_hole notebook_sheet human :=
sorry

end possible_to_cut_hole_l80_80464


namespace a_1994_value_l80_80077

noncomputable def sequence (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := (sequence n * real.sqrt 3 + 1) / (real.sqrt 3 - sequence n)

theorem a_1994_value (a : ℝ) : 
  sequence a 1994 = (a + real.sqrt 3) / (1 - a * real.sqrt 3) :=
sorry

end a_1994_value_l80_80077


namespace factorize_problem_1_factorize_problem_2_l80_80923

theorem factorize_problem_1 (a b : ℝ) : -3 * a ^ 3 + 12 * a ^ 2 * b - 12 * a * b ^ 2 = -3 * a * (a - 2 * b) ^ 2 := 
sorry

theorem factorize_problem_2 (m n : ℝ) : 9 * (m + n) ^ 2 - (m - n) ^ 2 = 4 * (2 * m + n) * (m + 2 * n) := 
sorry

end factorize_problem_1_factorize_problem_2_l80_80923


namespace rectangle_coords_sum_l80_80956

-- Problem statement in Lean 4
theorem rectangle_coords_sum (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 7) (hy1 : y1 = 19) (hx2 : x2 = 11) (hy2 : y2 = -2) :
  (x1 + x2) = 18 ∧ (y1 + y2) = 17 := 
by 
  rw [hx1, hy1, hx2, hy2]
  -- Calculate the sums
  have hx_sum : 7 + 11 = 18 := rfl
  have hy_sum : 19 + (-2) = 17 := rfl
  -- Conclude the proof
  exact ⟨hx_sum, hy_sum⟩

end rectangle_coords_sum_l80_80956


namespace sum_abs_nonconstant_coeffs_eqn_l80_80838

theorem sum_abs_nonconstant_coeffs_eqn :
  let p := (3 * x - 1) ^ 7
  in |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| = 4 ^ 7 :=
sorry

end sum_abs_nonconstant_coeffs_eqn_l80_80838


namespace sum_series_l80_80251

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l80_80251


namespace sum_series_equals_4_div_9_l80_80219

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80219


namespace fraction_product_l80_80131

theorem fraction_product : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l80_80131


namespace sum_of_factors_36_l80_80714

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80714


namespace negation_exists_zero_product_l80_80544

variable {R : Type} [LinearOrderedField R]

variable (f g : R → R)

theorem negation_exists_zero_product :
  (¬ ∃ x : R, f x * g x = 0) ↔ ∀ x : R, f x ≠ 0 ∧ g x ≠ 0 :=
by
  sorry

end negation_exists_zero_product_l80_80544


namespace expected_difference_after_10_days_l80_80036

noncomputable def cat_fox_expected_difference : ℕ → ℝ
| 0     := 0
| (n+1) := 0.25 * (cat_fox_expected_difference n + 1)  -- cat wins
                + 0.25 * (cat_fox_expected_difference n + 1)  -- fox wins
                + 0.5 * 0  -- both go to police, difference resets

theorem expected_difference_after_10_days :
  cat_fox_expected_difference 10 = 1 :=
sorry

end expected_difference_after_10_days_l80_80036


namespace proof_problem_l80_80985

variable {a1 a2 b1 b2 b3 : ℝ}

-- Condition: -2, a1, a2, -8 form an arithmetic sequence
def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
  a2 - a1 = -2 / 3 * (-2 - 8)

-- Condition: -2, b1, b2, b3, -8 form a geometric sequence
def geometric_sequence (b1 b2 b3 : ℝ) : Prop :=
  b2^2 = (-2) * (-8) ∧ b1^2 = (-2) * b2 ∧ b3^2 = b2 * (-8)

theorem proof_problem (h1 : arithmetic_sequence a1 a2) (h2 : geometric_sequence b1 b2 b3) : b2 * (a2 - a1) = 8 :=
by
  admit -- Convert to sorry to skip the proof

end proof_problem_l80_80985


namespace percentage_increase_l80_80150

variables (A B C D E : ℝ)
variables (A_inc B_inc C_inc D_inc E_inc : ℝ)

-- Conditions
def conditions (A_inc B_inc C_inc D_inc E_inc : ℝ) :=
  A_inc = 0.1 * A ∧
  B_inc = (1/15) * B ∧
  C_inc = 0.05 * C ∧
  D_inc = 0.04 * D ∧
  E_inc = (1/30) * E ∧
  B = 1.5 * A ∧
  C = 2 * A ∧
  D = 2.5 * A ∧
  E = 3 * A

-- Theorem to prove
theorem percentage_increase (A B C D E : ℝ) (A_inc B_inc C_inc D_inc E_inc : ℝ) :
  conditions A B C D E A_inc B_inc C_inc D_inc E_inc →
  (A_inc + B_inc + C_inc + D_inc + E_inc) / (A + B + C + D + E) = 0.05 :=
by
  sorry

end percentage_increase_l80_80150


namespace sum_of_factors_of_36_l80_80778

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80778


namespace stratified_sample_medium_supermarkets_l80_80436

theorem stratified_sample_medium_supermarkets (large medium small total sample : ℕ)
  (h_large : large = 200)
  (h_medium : medium = 400)
  (h_small : small = 1400)
  (h_total : total = large + medium + small)
  (h_sample : sample = 100) :
  (sample * medium / total = 20) := by
  -- Given conditions
  rw [h_large, h_medium, h_small] at h_total,
  rw [h_medium, h_total, h_sample],
  norm_num,
  sorry

end stratified_sample_medium_supermarkets_l80_80436


namespace sum_series_eq_4_div_9_l80_80177

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80177


namespace sum_of_factors_36_eq_91_l80_80591

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l80_80591


namespace remainder_of_5_pow_2023_mod_17_l80_80326

theorem remainder_of_5_pow_2023_mod_17 :
  5^2023 % 17 = 11 :=
by
  have h1 : 5^2 % 17 = 8 := by sorry
  have h2 : 5^4 % 17 = 13 := by sorry
  have h3 : 5^8 % 17 = -1 := by sorry
  have h4 : 5^16 % 17 = 1 := by sorry
  have h5 : 2023 = 16 * 126 + 7 := by sorry
  sorry

end remainder_of_5_pow_2023_mod_17_l80_80326


namespace sum_of_factors_36_eq_91_l80_80753

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80753


namespace sum_of_factors_36_l80_80613

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80613


namespace number_of_quadruples_l80_80316

/-- The number of ordered quadruples (a, b, c, d) of nonnegative real numbers such that
    a² + b² + c² + d² = 9 and (a + b + c + d)(a³ + b³ + c³ + d³) = 81 is 15. -/
theorem number_of_quadruples :
  ∃ (s : Finset (ℝ × ℝ × ℝ × ℝ)),
    (∀ (p : ℝ × ℝ × ℝ × ℝ), p ∈ s → 
     let (a, b, c, d) := p in 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 
     a^2 + b^2 + c^2 + d^2 = 9 ∧ 
     (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81) ∧ 
    s.card = 15 :=
by
  sorry

end number_of_quadruples_l80_80316


namespace sum_of_factors_36_l80_80618

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80618


namespace sum_of_factors_36_l80_80713

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80713


namespace plane_altitude_correct_l80_80858

noncomputable theory

def altitude_of_plane (speed : ℝ) (alpha beta : ℝ) (time : ℝ) : ℝ :=
  let
    alpha_rad := (34 + 28/60) * real.pi / 180,  -- Convert degrees to radians
    beta_rad := (23 + 41/60) * real.pi / 180,   -- Convert degrees to radians
    R0R := speed * (time / 60),  -- Distance traveled in 1 minute
    d0 := λ x, x * (1 / real.tan alpha_rad),    -- Initial distance calculation
    d := λ x, x * (1 / real.tan beta_rad)       -- Final distance calculation
  in
    sqrt ((6.5 ^ 2) / (real.cot beta_rad ^ 2 - real.cot alpha_rad ^ 2))

theorem plane_altitude_correct :
  altitude_of_plane 390 (34 + 28/60) (23 + 41/60) 1 = 3700 :=
by 
  sorry  -- Proof omitted.

end plane_altitude_correct_l80_80858


namespace smallest_k_multiple_of_175_l80_80466

theorem smallest_k_multiple_of_175 (k : ℕ) :
  (1 ≤ k) ∧ (∃ (n : ℕ), 1^2 + 2^2 + ... + k^2 = n * 175) ↔ k = 1200 :=
by sorry

end smallest_k_multiple_of_175_l80_80466


namespace cross_section_area_leq_largest_face_area_l80_80573

variables {A B C D : Point}
variables {P : Plane}

noncomputable theory

def largest_face_area (D A B C : Point) : ℝ := max (Area (Triangle A B C)) (max (Area (Triangle D A B)) (max (Area (Triangle D B C)) (Area (Triangle D C A))))

theorem cross_section_area_leq_largest_face_area (D A B C : Point) (P : Plane)
  (h_tetra : Tetrahedron D A B C) :
  ∃ S : ℝ, S = largest_face_area D A B C ∧ ∀ T: Triangle, (CrossSection T P (Tetrahedron D A B C)) → Area T ≤ S :=
by
  sorry

end cross_section_area_leq_largest_face_area_l80_80573


namespace person_A_arrives_first_both_arrive_same_time_l80_80567

noncomputable def travel_times (m n S : ℝ) (h_mn : m ≠ n) :=
let t_1 := 2 * S / (m + n) in
let t_2 := S * (m + n) / (2 * m * n) in
(t_1, t_2)

theorem person_A_arrives_first (m n S : ℝ) (h_mn : m ≠ n) :
  let (t_1, t_2) := travel_times m n S h_mn
  in t_1 < t_2 :=
by {
  let (t_1, t_2) := travel_times m n S h_mn,
  dsimp [t_1, t_2],
  sorry
}

theorem both_arrive_same_time (m n S : ℝ) (h_mn : m = n) :
  let (t_1, t_2) := travel_times m n S h_mn
  in t_1 = t_2 :=
by {
  let (t_1, t_2) := travel_times m n S h_mn,
  dsimp [t_1, t_2],
  sorry
}

end person_A_arrives_first_both_arrive_same_time_l80_80567


namespace fraction_problem_l80_80532

theorem fraction_problem (a : ℕ) (h1 : (a:ℚ)/(a + 27) = 865/1000) : a = 173 := 
by
  sorry

end fraction_problem_l80_80532


namespace sum_series_eq_4_div_9_l80_80294

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80294


namespace expected_absolute_difference_after_10_days_l80_80031

def cat_fox_wealth_difference : ℝ := 1

theorem expected_absolute_difference_after_10_days :
  let p_cat_wins := 0.25
  let p_fox_wins := 0.25
  let p_both_police := 0.5
  let num_days := 10
  ∃ (X : ℕ → ℕ), 
    (X 0 = 0) ∧
    ∀ n, (X (n + 1) = (if (X n = 0) then 0.5 else 0) * X n) →
    (∑ k in range (num_days + 1), (k : ℝ) * (0.5 ^ k)) = cat_fox_wealth_difference := 
sorry

end expected_absolute_difference_after_10_days_l80_80031


namespace injective_function_equality_l80_80846

def injective (f : ℕ → ℕ) : Prop :=
  ∀ ⦃a b : ℕ⦄, f a = f b → a = b

theorem injective_function_equality
  {f : ℕ → ℕ}
  (h_injective : injective f)
  (h_eq : ∀ n m : ℕ, (1 / f n) + (1 / f m) = 4 / (f n + f m)) :
  ∀ n m : ℕ, m = n :=
by
  sorry

end injective_function_equality_l80_80846


namespace projectile_trajectory_line_l80_80076

noncomputable def trajectory_highest_points_tracing_line (v g : ℝ) : Prop :=
  let θ := (real.pi / 4)
  let x (t : ℝ) := v * t * real.cos θ
  let y (t : ℝ) := (v * t * real.sin θ) - (1 / 2) * g * t^2
  ∀ x_max y_max, (∃ t_max, x_max = x t_max ∧ y_max = y t_max) →
  y_max = x_max / 2

theorem projectile_trajectory_line (v g : ℝ) :
  trajectory_highest_points_tracing_line v g :=
by 
  sorry

end projectile_trajectory_line_l80_80076


namespace sum_series_eq_4_div_9_l80_80292

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80292


namespace sum_infinite_series_l80_80190

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80190


namespace Sn_eq_S9_l80_80991

-- Definition of the arithmetic sequence sum formula.
def Sn (n a1 d : ℕ) : ℕ := (n * a1) + (n * (n - 1) / 2 * d)

theorem Sn_eq_S9 (a1 d : ℕ) (h1 : Sn 3 a1 d = 9) (h2 : Sn 6 a1 d = 36) : Sn 9 a1 d = 81 := by
  sorry

end Sn_eq_S9_l80_80991


namespace series_sum_eq_four_ninths_l80_80239

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l80_80239


namespace total_animals_hunted_l80_80443

theorem total_animals_hunted :
  let sam_hunts := 6
  let rob_hunts := sam_hunts / 2
  let total_sam_rob := sam_hunts + rob_hunts
  let mark_hunts := total_sam_rob / 3
  let peter_hunts := mark_hunts * 3
  sam_hunts + rob_hunts + mark_hunts + peter_hunts = 21 :=
by
  sorry

end total_animals_hunted_l80_80443


namespace sum_of_factors_of_36_l80_80667

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l80_80667


namespace minimum_prime_divisor_of_polynomial_l80_80483

theorem minimum_prime_divisor_of_polynomial :
  ∃ (p : ℕ), Nat.Prime p ∧ 
    (∀ (n : ℕ), n > 0 → ¬ (p ∣ (n^2 + 7 * n + 23))) →
    p = 11 :=
begin
  sorry
end

end minimum_prime_divisor_of_polynomial_l80_80483


namespace total_animals_hunted_l80_80444

theorem total_animals_hunted :
  let sam_hunts := 6
  let rob_hunts := sam_hunts / 2
  let total_sam_rob := sam_hunts + rob_hunts
  let mark_hunts := total_sam_rob / 3
  let peter_hunts := mark_hunts * 3
  sam_hunts + rob_hunts + mark_hunts + peter_hunts = 21 :=
by
  sorry

end total_animals_hunted_l80_80444


namespace sum_series_eq_4_div_9_l80_80293

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l80_80293


namespace sum_of_factors_36_eq_91_l80_80751

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l80_80751


namespace probability_of_same_color_l80_80060

theorem probability_of_same_color (total_balls white_balls black_balls n chosen_balls : ℕ)
  (h_total : total_balls = white_balls + black_balls)
  (h_white_balls : white_balls = 8)
  (h_black_balls : black_balls = 7)
  (h_chosen_balls : n = 5) :
  n = chosen_balls →
  let total_ways := Nat.choose total_balls n,
      ways_white := Nat.choose white_balls n,
      ways_black := Nat.choose black_balls n,
      favorable_outcomes := ways_white + ways_black,
      probability := favorable_outcomes / total_ways in
  probability = 77 / 3003 :=
by
  sorry

end probability_of_same_color_l80_80060


namespace sum_factors_36_l80_80642

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80642


namespace cannot_equalize_pieces_l80_80499

structure Board where
  size : ℕ
  pieces : ℕ
  cells : Fin size.succ × Fin size.succ → ℕ
  n_neighbors : (Fin size.succ × Fin size.succ) → Fin size.succ × Fin size.succ → Bool

def neighbors (size : ℕ) (cell : Fin size.succ × Fin size.succ) : List (Fin size.succ × Fin size.succ) :=
  let (i, j) := cell
  [ 
    ((i + size) % size.succ, j), 
    ((i + 1) % size.succ, j), 
    (i, (j + size) % size.succ), 
    (i, (j + 1) % size.succ)
  ]

def invariant_sum (b: Board) : ℤ :=
  ∑ i in (Fin n).attach, ∑ j in (Fin n).attach, b.cells (i.1, j.1) * ((i.1 : ℤ) + (j.1 : ℤ))

theorem cannot_equalize_pieces :
  ∀ (b : Board), 
  b.size = 10 → 
  b.pieces = 484 →
  (∀ (cell : Fin size.succ × Fin size.succ), b.n_neighbors cell = neighbors b.size cell) →
  ∀ (initial_distribution : Fin b.size.succ × Fin b.size.succ → ℕ), 
  ¬ ∃ (final_distribution : Fin b.size.succ × Fin b.size.succ → ℕ), 
    (∀ (cell : Fin b.size.succ × Fin b.size.succ), final_distribution cell = b.pieces / (size.succ * size.succ)) 
    ∧ (invariant_sum { Board with cells := initial_distribution, ..b } = invariant_sum { Board with cells := final_distribution, ..b }) := by sorry

end cannot_equalize_pieces_l80_80499


namespace distribute_items_5_in_4_identical_bags_l80_80113

theorem distribute_items_5_in_4_identical_bags : 
  let items := 5 in 
  let bags := 4 in 
  number_of_ways_to_distribute items bags = 36 := 
by sorry

end distribute_items_5_in_4_identical_bags_l80_80113


namespace sum_factors_of_36_l80_80653

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80653


namespace calculate_added_water_l80_80059

-- Defining conditions as Lean definitions
def initial_volume : ℝ := 90
def initial_percentage_jasmine : ℝ := 0.05
def added_jasmine : ℝ := 8
def final_percentage_jasmine : ℝ := 0.125

-- The final statement we need to prove, including the equality condition
theorem calculate_added_water (initial_volume = 90) (initial_percentage_jasmine = 0.05) 
  (added_jasmine = 8) (final_percentage_jasmine = 0.125) : 
  let initial_jasmine := initial_volume * initial_percentage_jasmine
  let final_jasmine := initial_jasmine + added_jasmine
  let final_volume := final_jasmine / final_percentage_jasmine
  final_volume - initial_volume - added_jasmine = 2 :=
by
  sorry

end calculate_added_water_l80_80059


namespace total_hunts_is_21_l80_80437

-- Define the initial conditions
def Sam_hunts : Nat := 6
def Rob_hunts : Nat := Sam_hunts / 2
def Rob_Sam_total_hunt : Nat := Sam_hunts + Rob_hunts
def Mark_hunts : Nat := Rob_Sam_total_hunt / 3
def Peter_hunts : Nat := Mark_hunts * 3

-- The main theorem to prove
theorem total_hunts_is_21 : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 :=
by
  sorry

end total_hunts_is_21_l80_80437


namespace area_triangle_45_90_45_l80_80459

theorem area_triangle_45_90_45 (A B C D : Type) [triangle ABC] [altitude_from C_to_AB D] 
  (h1 : angle A = 45) (h2 : CD = 2) : 
  area ABC = 2 * real.sqrt 2 := 
sorry

end area_triangle_45_90_45_l80_80459


namespace product_comparison_l80_80905

/-
  Let \(A\) be the following product:
  \[
  A = \frac{100}{101} \times \frac{102}{103} \times \ldots \times \frac{2020}{2021} \times \frac{2022}{2023}
  \]
  Prove that \( A \) is less than \( \frac{5}{16} \) given that for all \( n \geq 100 \),
  \[
  \frac{n}{n+1} < \frac{n+2}{n+3}
  \]
-/

theorem product_comparison : 
  (A : ℝ) (hA : A = (List.range' 100 922).prod (λ n => (n * 2 + 100) / (n * 2 + 101))) :
  (∀ n : ℕ, n ≥ 100 → (n : ℝ) / (n + 1 : ℝ) < (n + 2 : ℝ) / (n + 3 : ℝ)) →
  A < 5 / 16 :=
sorry

end product_comparison_l80_80905


namespace sum_series_eq_four_ninths_l80_80204

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l80_80204


namespace bacteria_doubling_time_l80_80524

open Real

theorem bacteria_doubling_time (initial_bacteria : ℕ) (final_bacteria : ℕ) (doubling_time : ℕ) (t : ℕ) : 
  initial_bacteria = 400 → 
  final_bacteria = 25600 → 
  doubling_time = 3 → 
  (log 2 (final_bacteria / initial_bacteria)) * doubling_time = t → 
  t = 18 :=
by
  intros h1 h2 h3 h4
  rw ← h1, ← h2, ← h3 at h4
  sorry

end bacteria_doubling_time_l80_80524


namespace sum_factors_36_eq_91_l80_80612

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l80_80612


namespace infinite_series_sum_l80_80235

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80235


namespace sum_of_series_l80_80171

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80171


namespace negate_at_most_two_l80_80105

def atMost (n : Nat) : Prop := ∃ k : Nat, k ≤ n
def atLeast (n : Nat) : Prop := ∃ k : Nat, k ≥ n

theorem negate_at_most_two : ¬ atMost 2 ↔ atLeast 3 := by
  sorry

end negate_at_most_two_l80_80105


namespace largest_power_of_7_in_sum_l80_80335

-- Define factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Problem statement
theorem largest_power_of_7_in_sum :
  let s := factorial 47 + factorial 48 + factorial 49 in
  let n := 10 in
  ∃ k, 7^k ∣ s ∧ n = k :=
sorry

end largest_power_of_7_in_sum_l80_80335


namespace sum_factors_36_l80_80767

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80767


namespace triplet_solution_l80_80145

theorem triplet_solution (a b c : ℝ) 
  (h1 : a * (b^2 + c) = c * (c + a * b))
  (h2 : b * (c^2 + a) = a * (a + b * c))
  (h3 : c * (a^2 + b) = b * (b + a * c)) :
  ∃ k : ℝ, a = k ∧ b = k ∧ c = k :=
begin
  sorry
end

end triplet_solution_l80_80145


namespace values_of_z_l80_80948

theorem values_of_z (z : ℤ) (hz : 0 < z) :
  (z^2 - 50 * z + 550 ≤ 10) ↔ (20 ≤ z ∧ z ≤ 30) := sorry

end values_of_z_l80_80948


namespace sum_of_factors_of_36_l80_80781

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80781


namespace evaluate_series_sum_l80_80284

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80284


namespace points_opposite_sides_of_line_l80_80374

theorem points_opposite_sides_of_line (m : ℝ) : 
  (let f (p : ℝ × ℝ) := 3 * p.1 - p.2 + m in 
  (f (1, 2)) * (f (1, 1)) < 0) ↔ -2 < m ∧ m < -1 :=
by
  let f := λ p : ℝ × ℝ => 3 * p.1 - p.2 + m
  sorry

end points_opposite_sides_of_line_l80_80374


namespace race_outcomes_l80_80872

def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Fiona"]

theorem race_outcomes (h : ¬ "Fiona" ∈ ["Abe", "Bobby", "Charles", "Devin", "Edwin"]) : 
  (participants.length - 1) * (participants.length - 2) * (participants.length - 3) = 60 :=
by
  sorry

end race_outcomes_l80_80872


namespace adult_ticket_cost_l80_80571

def num_total_tickets : ℕ := 510
def cost_senior_ticket : ℕ := 15
def total_receipts : ℤ := 8748
def num_senior_tickets : ℕ := 327
def num_adult_tickets : ℕ := num_total_tickets - num_senior_tickets
def revenue_senior : ℤ := num_senior_tickets * cost_senior_ticket
def revenue_adult (cost_adult_ticket : ℤ) : ℤ := num_adult_tickets * cost_adult_ticket

theorem adult_ticket_cost : 
  ∃ (cost_adult_ticket : ℤ), 
    revenue_adult cost_adult_ticket + revenue_senior = total_receipts ∧ 
    cost_adult_ticket = 21 :=
by
  sorry

end adult_ticket_cost_l80_80571


namespace sum_factors_36_l80_80770

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80770


namespace groceries_value_l80_80093

-- Conditions
def alex_saved : ℝ := 14500
def car_cost : ℝ := 14600
def trip_charge : ℝ := 1.5
def earn_percentage : ℝ := 0.05
def num_trips : ℝ := 40

-- Proof Statement
theorem groceries_value
  (alex_saved : ℝ)
  (car_cost : ℝ)
  (trip_charge : ℝ)
  (earn_percentage : ℝ)
  (num_trips : ℝ)
  (h_saved : alex_saved = 14500)
  (h_car_cost : car_cost = 14600)
  (h_trip_charge : trip_charge = 1.5)
  (h_earn_percentage : earn_percentage = 0.05)
  (h_num_trips : num_trips = 40) :

  let needed_savings := car_cost - alex_saved in
  let earnings_from_trips := num_trips * trip_charge in
  let earnings_from_groceries := needed_savings - earnings_from_trips in
  let total_value_of_groceries := earnings_from_groceries / earn_percentage in
  total_value_of_groceries = 800 := by {
    sorry
  }

end groceries_value_l80_80093


namespace sum_k_over_4k_l80_80159

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l80_80159


namespace quadratic_polynomial_solution_is_zero_l80_80407

-- Definitions based on given conditions
variables (a b c r s : ℝ)
variables (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
variables (h2 : a ≠ b ∧ a ≠ c ∧ b ≠ c)
variables (h3 : r + s = -b / a)
variables (h4 : r * s = c / a)

-- Proposition matching the equivalent proof problem
theorem quadratic_polynomial_solution_is_zero :
  ¬ ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  (∃ r s : ℝ, (r + s = -b / a) ∧ (r * s = c / a) ∧ (c = r * s ∨ b = r * s ∨ a = r * s) ∧
  (a = r ∨ a = s)) :=
sorry

end quadratic_polynomial_solution_is_zero_l80_80407


namespace groceries_delivered_amount_l80_80100

noncomputable def alex_saved_up : ℝ := 14500
noncomputable def car_cost : ℝ := 14600
noncomputable def charge_per_trip : ℝ := 1.5
noncomputable def percentage_charge : ℝ := 0.05
noncomputable def number_of_trips : ℕ := 40

theorem groceries_delivered_amount :
  ∃ G : ℝ, charge_per_trip * number_of_trips + percentage_charge * G = car_cost - alex_saved_up ∧ G = 800 :=
by {
  use 800,
  rw [mul_comm (800 : ℝ), mul_assoc],
  norm_num,
  exact add_comm 60 (40 : ℝ),
  sorry
}

end groceries_delivered_amount_l80_80100


namespace rise_in_water_level_l80_80043

def edge_length : ℝ := 5  -- cm
def base_length : ℝ := 10  -- cm
def base_width  : ℝ := 5  -- cm

def volume_cube (a : ℝ) : ℝ := a^3
def base_area (l w : ℝ) : ℝ := l * w

theorem rise_in_water_level :
  let V_cube := volume_cube edge_length,
      A_base := base_area base_length base_width
  in V_cube / A_base = 2.5 := by
  sorry

end rise_in_water_level_l80_80043


namespace evaluate_series_sum_l80_80278

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80278


namespace remainder_of_power_mod_l80_80330

theorem remainder_of_power_mod :
  (5^2023) % 17 = 15 :=
begin
  sorry
end

end remainder_of_power_mod_l80_80330


namespace sum_infinite_series_l80_80200

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80200


namespace sum_factors_of_36_l80_80649

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80649


namespace max_strong_boys_l80_80426

theorem max_strong_boys (n : ℕ) (h : n = 100) (a b : Fin n → ℕ) 
  (ha : ∀ i j : Fin n, i < j → a i > a j) 
  (hb : ∀ i j : Fin n, i < j → b i < b j) : 
  ∃ k : ℕ, k = n := 
sorry

end max_strong_boys_l80_80426


namespace train_parking_methods_l80_80867

theorem train_parking_methods :
  (∃ (track_count train_count : ℕ), track_count = 8 ∧ train_count = 4 ∧ 
  (∏ i in finset.range train_count, (track_count - i)) = 1680) :=
begin
  use [8, 4],
  split,
  { refl, },
  split,
  { refl, },
  calc 
    ∏ i in finset.range 4, (8 - i) = (8 * 7 * 6 * 5) : by norm_num
    ... = 1680 : by norm_num,
end

end train_parking_methods_l80_80867


namespace function_strictly_decreasing_l80_80141

open Real

theorem function_strictly_decreasing :
  ∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → deriv (λ x, 1 / x + 2 * log x) x < 0 :=
by
  -- Proof is omitted
  sorry

end function_strictly_decreasing_l80_80141


namespace sum_series_equals_4_div_9_l80_80224

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80224


namespace remainder_of_5_pow_2023_mod_17_l80_80325

theorem remainder_of_5_pow_2023_mod_17 :
  5^2023 % 17 = 11 :=
by
  have h1 : 5^2 % 17 = 8 := by sorry
  have h2 : 5^4 % 17 = 13 := by sorry
  have h3 : 5^8 % 17 = -1 := by sorry
  have h4 : 5^16 % 17 = 1 := by sorry
  have h5 : 2023 = 16 * 126 + 7 := by sorry
  sorry

end remainder_of_5_pow_2023_mod_17_l80_80325


namespace last_two_nonzero_digits_100_l80_80142

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def factors_of_10 (n! : ℕ) : ℕ :=
  let fives := (n / 5) + (n / 25) + (n / 125) -- Here n is 100
  let twos := (n / 2^1) -- Simplification to show count
  min fives twos 

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  (factorial n) / (10^factors_of_10 (factorial n)) % 100

theorem last_two_nonzero_digits_100 :
  last_two_nonzero_digits 100 = 76 :=
sorry

end last_two_nonzero_digits_100_l80_80142


namespace range_of_2a_minus_b_l80_80978

theorem range_of_2a_minus_b (a b : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : 2 < b) (h4 : b < 4) : 
  -2 < 2 * a - b ∧ 2 * a - b < 4 := 
by 
  sorry

end range_of_2a_minus_b_l80_80978


namespace sum_of_series_l80_80172

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80172


namespace fraction_product_l80_80126

theorem fraction_product :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := 
by
  -- Detailed proof steps would go here
  sorry

end fraction_product_l80_80126


namespace count_first_digit_7_l80_80474

noncomputable def U := {x : ℕ | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 3000 ∧ x = 7^k}

theorem count_first_digit_7 :
  (∃ n : ℕ, n = {x ∈ U | ∃ d : ℤ, 7^(3000-d) = x ∧ Nat.digits 10 x.head = 7}.card ∧ n = 491) :=
sorry

end count_first_digit_7_l80_80474


namespace transform_f_to_g_exists_varphi_l80_80513

def f (x : ℝ) := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)
def g (x : ℝ) := 2 * Real.sin x

theorem transform_f_to_g_exists_varphi (φ : ℝ) (hφ : 2 * Real.sin (2 * (x - φ) + Real.pi / 3) = 2 * Real.sin x) : φ = Real.pi / 6 :=
sorry

end transform_f_to_g_exists_varphi_l80_80513


namespace sum_of_series_l80_80167

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l80_80167


namespace polar_coordinates_intersection_parametric_equation_given_chord_length_l80_80989

-- Definitions for conditions
def polar_equiv_cartesian_pole_origin : Prop :=
  -- coordinates' pole and origin coincide
  ∀ (ρ θ : ℝ) (x y : ℝ), x = ρ * cos θ ∧ y = ρ * sin θ ∧ ρ = sqrt (x^2 + y^2)

def parametric_equation_line (alpha : ℝ) (l : ℝ → ℝ × ℝ) : Prop :=
  -- parametric equation of line l
  ∀ (t : ℝ), l t = (-1 + t * cos alpha, 1 + t * sin alpha)

def polar_equation_curve : ℝ → ℝ → Prop :=
  -- polar equation of curve C
  λ ρ θ, ρ = 4 * cos θ

def slope_negative_one (alpha : ℝ) : Prop :=
  -- slope of line l is -1
  tan alpha = -1

def chord_length (k : ℝ) : ℝ → Prop :=
  -- chord length of the intersection
  λ length, length = 2 * sqrt 3

-- Proof statements based on the above definitions and conditions:

theorem polar_coordinates_intersection (alpha : ℝ) (l : ℝ → ℝ × ℝ) :
  slope_negative_one(alpha) ∧ parametric_equation_line(alpha, l) →
  ∃ (A B : ℝ × ℝ), polar_equiv_cartesian_pole_origin A.1 A.2 B.1 B.2 ∧
                   A = (0, 0) ∧ B = (2 * sqrt 2, 7 / 4 * Real.pi) :=
by
  sorry

theorem parametric_equation_given_chord_length (k : ℝ) (l : ℝ → ℝ × ℝ) :
  chord_length(k) → 
  ∃ (l₁ l₂ : ℝ → ℝ × ℝ), 
    parametric_equation_line(0, l₁) ∧ parametric_equation_line(pi + -atan 3/4, l₂) :=
by
  sorry

end polar_coordinates_intersection_parametric_equation_given_chord_length_l80_80989


namespace sum_of_factors_36_l80_80738

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80738


namespace different_quantifiers_not_equiv_l80_80015

theorem different_quantifiers_not_equiv {x₀ : ℝ} :
  (∃ x₀ : ℝ, x₀^2 > 3) ↔ ¬ (∀ x₀ : ℝ, x₀^2 > 3) :=
by
  sorry

end different_quantifiers_not_equiv_l80_80015


namespace sin_double_theta_l80_80421

-- Given condition
def given_condition (θ : ℝ) : Prop :=
  Real.cos (Real.pi / 4 - θ) = 1 / 2

-- The statement we want to prove: sin(2θ) = -1/2
theorem sin_double_theta (θ : ℝ) (h : given_condition θ) : Real.sin (2 * θ) = -1 / 2 :=
sorry

end sin_double_theta_l80_80421


namespace sum_infinite_series_l80_80192

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l80_80192


namespace expected_difference_after_10_days_l80_80022

-- Define the initial state and transitions
noncomputable def initial_prob (k : ℤ) : ℝ :=
if k = 0 then 1 else 0

noncomputable def transition_prob (k : ℤ) (n : ℕ) : ℝ :=
0.5 * initial_prob k +
0.25 * initial_prob (k - 1) +
0.25 * initial_prob (k + 1)

-- Define event probability for having any wealth difference after n days
noncomputable def p_k_n (k : ℤ) (n : ℕ) : ℝ :=
if n = 0 then initial_prob k
else transition_prob k (n - 1)

-- Use expected value of absolute difference between wealths 
noncomputable def expected_value_abs_diff (n : ℕ) : ℝ :=
Σ' k, |k| * p_k_n k n

-- Finally, state the theorem
theorem expected_difference_after_10_days :
expected_value_abs_diff 10 = 1 :=
by
  sorry

end expected_difference_after_10_days_l80_80022


namespace total_animals_hunted_l80_80445

theorem total_animals_hunted :
  let sam_hunts := 6
  let rob_hunts := sam_hunts / 2
  let total_sam_rob := sam_hunts + rob_hunts
  let mark_hunts := total_sam_rob / 3
  let peter_hunts := mark_hunts * 3
  sam_hunts + rob_hunts + mark_hunts + peter_hunts = 21 :=
by
  sorry

end total_animals_hunted_l80_80445


namespace total_hunts_l80_80441

-- Conditions
def Sam_hunts : ℕ := 6
def Rob_hunts := Sam_hunts / 2
def combined_Rob_Sam_hunts := Rob_hunts + Sam_hunts
def Mark_hunts := combined_Rob_Sam_hunts / 3
def Peter_hunts := 3 * Mark_hunts

-- Question and proof statement
theorem total_hunts : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 := by
  sorry

end total_hunts_l80_80441


namespace sum_factors_36_l80_80766

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80766


namespace Linda_purchase_cost_l80_80487

def price_peanuts : ℝ := sorry
def price_berries : ℝ := sorry
def price_coconut : ℝ := sorry
def price_dates : ℝ := sorry

theorem Linda_purchase_cost:
  ∃ (p b c d : ℝ), 
    (p + b + c + d = 30) ∧ 
    (3 * p = d) ∧
    ((p + b) / 2 = c) ∧
    (b + c = 65 / 9) :=
sorry

end Linda_purchase_cost_l80_80487


namespace sum_of_factors_36_l80_80813

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l80_80813


namespace sum_of_factors_36_l80_80734

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80734


namespace sin_symmetry_y_axis_l80_80538

theorem sin_symmetry_y_axis :
  Symmetric (λ x : ℝ, sin (x + 3 * Real.pi / 2)) :=
by
  sorry

end sin_symmetry_y_axis_l80_80538


namespace drum_oil_capacity_l80_80921

theorem drum_oil_capacity (C : ℝ) (Y : ℝ) 
  (hX : DrumX_Oil = 0.5 * C) 
  (hY : DrumY_Cap = 2 * C) 
  (hY_filled : Y + 0.5 * C = 0.65 * (2 * C)) :
  Y = 0.8 * C :=
by
  sorry

end drum_oil_capacity_l80_80921


namespace no_integer_k_such_that_f_k_eq_8_l80_80357

theorem no_integer_k_such_that_f_k_eq_8 
  {a1 a2 a3 ... an : ℤ} {a b c d : ℤ} (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_poly: ∀ x, f(x) = x ^ n + a1 * x ^ (n - 1) + a2 * x ^ (n - 2) + ... + an)
  (h_roots: f(a) = 5 ∧ f(b) = 5 ∧ f(c) = 5 ∧ f(d) = 5) : 
  ¬ ∃ k : ℤ, f(k) = 8 :=
sorry

-- Define f as the given polynomial
def f(x : ℤ) : ℤ := x ^ n + a1 * x ^ (n - 1) + a2 * x ^ (n - 2) + ... + an

end no_integer_k_such_that_f_k_eq_8_l80_80357


namespace sum_factors_36_l80_80801

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l80_80801


namespace expected_difference_after_10_days_l80_80025

-- Define the initial state and transitions
noncomputable def initial_prob (k : ℤ) : ℝ :=
if k = 0 then 1 else 0

noncomputable def transition_prob (k : ℤ) (n : ℕ) : ℝ :=
0.5 * initial_prob k +
0.25 * initial_prob (k - 1) +
0.25 * initial_prob (k + 1)

-- Define event probability for having any wealth difference after n days
noncomputable def p_k_n (k : ℤ) (n : ℕ) : ℝ :=
if n = 0 then initial_prob k
else transition_prob k (n - 1)

-- Use expected value of absolute difference between wealths 
noncomputable def expected_value_abs_diff (n : ℕ) : ℝ :=
Σ' k, |k| * p_k_n k n

-- Finally, state the theorem
theorem expected_difference_after_10_days :
expected_value_abs_diff 10 = 1 :=
by
  sorry

end expected_difference_after_10_days_l80_80025


namespace sum_of_factors_36_l80_80615

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l80_80615


namespace cloud_layer_height_l80_80066

theorem cloud_layer_height (total_height h volume_ratio : ℝ) (V : ℝ) :
  total_height = 10000 →
  volume_ratio = 1 / 10 →
  V = (1/3) * π * ((h / total_height) ^ 2) * total_height →
  h / total_height = real.sqrt( real.sqrt(real.sqrt(volume_ratio))) →
  total_height - (h / total_height *10000) = 5360 := 
by
  intros ht10000 v10 Vdef hdef
  sorry

end cloud_layer_height_l80_80066


namespace find_n_int_sin_eq_cos_l80_80932

theorem find_n_int_sin_eq_cos (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) : 
  (sin (n : ℝ) * π / 180 = cos (810 * π / 180)) ↔ (n = -180 ∨ n = 0 ∨ n = 180) := 
by
  sorry

end find_n_int_sin_eq_cos_l80_80932


namespace odd_int_squares_eq_n4_iff_l80_80305

theorem odd_int_squares_eq_n4_iff (n : ℕ) (h_odd : n % 2 = 1) :
  (∃ (x : ℕ → ℕ), (∀ i, i < n → x i % 2 = 1) ∧ (∑ i in finset.range n, (x i)^2) = n^4) ↔ (n % 8 = 1) :=
begin
  sorry,
end

end odd_int_squares_eq_n4_iff_l80_80305


namespace multiple_choice_test_ways_l80_80855

theorem multiple_choice_test_ways : (8^8) = 16777216 := by
  have total_choices : ℕ := 8^8
  calc
    total_choices = 8^8   : rfl
                ... = 16777216 : sorry

end multiple_choice_test_ways_l80_80855


namespace infinite_series_sum_l80_80226

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l80_80226


namespace sum_of_integers_l80_80429

theorem sum_of_integers : (∀ (x y : ℤ), x = -4 ∧ y = -5 ∧ x - y = 1 → x + y = -9) := 
by 
  intros x y
  sorry

end sum_of_integers_l80_80429


namespace evaluate_series_sum_l80_80283

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l80_80283


namespace odd_factors_450_l80_80403

theorem odd_factors_450 : ∃ n: ℕ, n = 9 ∧ (∀ p, prime p ∧ 450 % p = 0 → odd p) := sorry

end odd_factors_450_l80_80403


namespace sum_factors_36_l80_80758

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l80_80758


namespace sum_of_factors_36_l80_80708

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l80_80708


namespace smallest_C_inequality_l80_80918

theorem smallest_C_inequality :
  ∃ C : ℝ, C = 5^15 ∧ ∀ (x1 x2 x3 x4 x5 : ℝ), 
    (0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧ 0 < x5) → 
    C * (x1^2005 + x2^2005 + x3^2005 + x4^2005 + x5^2005) ≥ 
    (x1 * x2 * x3 * x4 * x5) * (x1^125 + x2^125 + x3^125 + x4^125 + x5^125)^16 :=
by
  existsi (5^15)
  intros x1 x2 x3 x4 x5 H_pos
  sorry

end smallest_C_inequality_l80_80918


namespace min_product_f_l80_80552

-- Definitions for conditions
variables (a b c x₁ x₂ x₃ : ℝ)
variable (f : ℝ → ℝ)

-- Condition 1: f(x) = ax^2 + bx + c
def quadratic_function := ∀ x : ℝ, f(x) = a*x^2 + b*x + c

-- Condition 2: f(-1) = 0
def condition_1 := f (-1) = 0

-- Condition 3: ∀ x ∈ ℝ, f(x) ≥ x
def condition_2 := ∀ x : ℝ, f(x) ≥ x

-- Condition 4: For x in (0, 2), f(x) ≤ (x+1)^2 / 4
def condition_3 := ∀ x : ℝ, 0 < x ∧ x < 2 → f(x) ≤ (x + 1)^2 / 4

-- Additional conditions
def in_range := 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ 0 < x₃ ∧ x₃ < 2
def harmonic_mean := 1 / x₁ + 1 / x₂ + 1 / x₃ = 3

-- Proof statement
theorem min_product_f :
  quadratic_function f a b c →
  condition_1 f →
  condition_2 f →
  condition_3 f →
  in_range x₁ x₂ x₃ →
  harmonic_mean x₁ x₂ x₃ →
  f(x₁) * f(x₂) * f(x₃) = 1 :=
sorry

end min_product_f_l80_80552


namespace ellipse_min_value_8_ellipse_equation_min_l80_80377

noncomputable def ellipse_min_value (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : (3 * Real.sqrt 3)^2 / m^2 + 1 / n^2 = 1) : ℝ := m + n

theorem ellipse_min_value_8 : ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ (3 * Real.sqrt 3)^2 / m^2 + 1 / n^2 = 1 ∧ ellipse_min_value m n _ _ _ = 8 :=
begin
  -- Proof is not required
  sorry
end

noncomputable def ellipse_equation (x y m n : ℝ) (h : m + n = 8) : Prop :=
  x^2 / (6^2) + y^2 / (2^2) = 1

theorem ellipse_equation_min :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ (3 * Real.sqrt 3)^2 / m^2 + 1 / n^2 = 1 ∧ m + n = 8 ∧ ∀ x y, ellipse_equation x y m n _ :=
begin
  -- Proof is not required
  sorry
end

end ellipse_min_value_8_ellipse_equation_min_l80_80377


namespace sum_series_eq_4_div_9_l80_80185

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l80_80185


namespace sum_of_factors_36_l80_80709

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l80_80709


namespace part1_part2_l80_80380

open Real

def f (x : ℝ) (hx : x > 1) : ℝ := log (x + 1) - log (x - 1)

theorem part1 :
  ∀ x1 x2 : ℝ, (1 < x1) → (1 < x2) → (x1 < x2) → (f x1 (by linarith) > f x2 (by linarith)) :=
sorry

noncomputable def g (a x : ℝ) : ℝ := f (a^(2*x) - 2*a^x) (by linarith [a, x])

theorem part2 (a : ℝ) (ha : 0 < a) :
  (a > 1 → ∀ x : ℝ, (g a x < log 2) ↔ (x > log 3 / log a)) ∧
  (0 < a ∧ a < 1 → ∀ x : ℝ, (g a x < log 2) ↔ (x < log 3 / log a)) ∧
  (a = 1 → ∀ x : ℝ, ¬ (g a x < log 2)) :=
sorry

end part1_part2_l80_80380


namespace value_of_m_l80_80364

-- Define the two circles
def circle_C (x y : ℝ) (m : ℝ) := (x - 2)^2 + (y - 2)^2 = 8 - m
def circle_D (x y : ℝ) := (x + 1)^2 + (y + 2)^2 = 1

theorem value_of_m (m : ℝ) :
  (∀ x y, circle_C x y m) ∧ (∀ x y, circle_D x y) ∧ 
  (sqrt ((3)^2 + (4)^2) = sqrt (8 - m) + 1) → 
  m = -8 :=
by {
  sorry
}

end value_of_m_l80_80364


namespace construct_triangle_l80_80910

-- Definition of given points A, O and L with respective properties
variables (A O L : Point)
-- Assume O is the circumcenter
axiom circumcenter_O : is_circumcenter O
-- Assume L is the Lemoine point
axiom lemoine_point_L : is_lemoine_point L

-- The theorem statement for constructing triangle ABC
theorem construct_triangle (A O L : Point) (circumcenter_O : is_circumcenter O) (lemoine_point_L : is_lemoine_point L) : 
  ∃ (B C : Point), is_triangle A B C :=
sorry -- proof is omitted, as per instructions

end construct_triangle_l80_80910


namespace min_chips_left_l80_80149

theorem min_chips_left (n : ℕ) (board_size : ℕ) (chips : Fin board_size^2 → bool)
  (h_even_neighbors : ∀ (i : Fin board_size^2), even (neighbors i) → chips i = false)
  (h_board : board_size = 2017) : 
  ∃ (min_chips : ℕ), min_chips = 2 ∧ ∀ chips, (min_chips' : ℕ) = count_chips chips -> min_chips' ≥ 2 :=
by
  sorry

def neighbors (i : Fin (board_size * board_size)) : ℕ := sorry
def count_chips (chips: Fin (board_size * board_size) → bool) : ℕ := sorry

end min_chips_left_l80_80149


namespace apples_total_l80_80896

theorem apples_total (Benny_picked Dan_picked : ℕ) (hB : Benny_picked = 2) (hD : Dan_picked = 9) : Benny_picked + Dan_picked = 11 :=
by
  -- Definitions
  sorry

end apples_total_l80_80896


namespace sum_of_factors_of_36_l80_80783

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l80_80783


namespace sum_factors_36_l80_80637

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l80_80637


namespace sum_series_equals_4_div_9_l80_80223

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l80_80223


namespace largest_possible_number_after_deletion_l80_80913

def original_sequence : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 
87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

-- Given the problem of deleting 100 digits from the sequence 
-- "123456789101112...979899100" to obtain the largest possible number
-- we want to prove the expected result after deletion as the largest possible number
theorem largest_possible_number_after_deletion :
  ∃ (remaining_digits : List Nat), 
    (remaining_digits.length = 92 ∧ remaining_digits = [9, 9, 9, 9, 9, 7, 8, 5, 9, 6, 0, 6, 1, ..., 9, 9, 1, 0, 0]) :=
sorry

end largest_possible_number_after_deletion_l80_80913


namespace remainder_of_polynomial_division_l80_80943

noncomputable def evaluate_polynomial (x : ℂ) : ℂ :=
  x^100 + x^75 + x^50 + x^25 + 1

noncomputable def divisor_polynomial (x : ℂ) : ℂ :=
  x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_of_polynomial_division : 
  ∀ β : ℂ, divisor_polynomial β = 0 → evaluate_polynomial β = -1 :=
by
  intros β hβ
  sorry

end remainder_of_polynomial_division_l80_80943


namespace imaginary_number_properties_l80_80424

theorem imaginary_number_properties {x y : ℝ} (h : x + y * complex.I = complex.I) : x = 0 ∧ y ≠ 0 :=
by sorry

end imaginary_number_properties_l80_80424


namespace farthest_distance_return_trip_cost_l80_80852

-- Define the distances travelled at different times
def distances : List Int := [-3, 8, -9, 10, 4, -6, -2]

-- Define the fuel consumption per kilometer
def fuel_consumption_per_km : Float := 0.25

-- Define the cost of gasoline per liter
def cost_per_liter : Float := 6

theorem farthest_distance (distances : List Int) : 
  let positions := List.scanl (+) 0 distances
  let abs_positions := positions.map Int.natAbs
  ∃ (t : ℕ) (d : ℤ), t < distances.length ∧ abs_positions t = d ∧ d = 10 := by
  sorry

theorem return_trip_cost (distances : List Int) (fuel_consumption_per_km : Float) (cost_per_liter : Float) : 
  let total_distance := distances.map Int.natAbs |>.sum
  total_distance * fuel_consumption_per_km * cost_per_liter = 66 := by
  sorry

end farthest_distance_return_trip_cost_l80_80852


namespace arithmetic_sequence_problem_l80_80361

/--
Given an arithmetic sequence {a_n} with the sum of the first n terms S_n,
where a_3 + a_6 = a_5 + 4 and a_2, a_4, 2a_4 form a geometric sequence:
1. Prove the general formula for {a_n} is a_n = n.
2. Prove the sum of the first n terms S_n is S_n = n * (n + 1) / 2.
3. Prove the number of elements in the set {m | T_m = 100 - a_k, k ∈ ℕ+, m ∈ ℕ+, and k ≥ 10} is 4.
-/
theorem arithmetic_sequence_problem (a : ℕ → ℕ) (S T : ℕ → ℕ) :
  (∀ n, a n = n) ∧
  (∀ n, S n = n * (n + 1) / 2) ∧
  (set.card {m : ℕ+ | ∃ k : ℕ+, T m = 100 - a k ∧ k ≥ 10}  = 4) :=
by sorry

end arithmetic_sequence_problem_l80_80361


namespace eccentricity_of_ellipse_passes_through_C_l80_80434

theorem eccentricity_of_ellipse_passes_through_C
  (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB BC : Real)
  (cos_B : Real)
  (h1 : AB = BC)
  (h2 : cos_B = -7 / 18)
  (h3 : ∃ (e : Real),
    ∃ (f : A × B × C),
      (e += B + C = AB * BC) ∧ ∃ (AC AB BC foci : Real),
        (AC = Real.sqrt (AB^2 + BC^2 - 2 * AB * BC * cos_B))
        (foci = e / AC)) :
  (eccentricity_of_ellipse_passes_through_C = 3 / 8) := sorry

end eccentricity_of_ellipse_passes_through_C_l80_80434


namespace sum_factors_of_36_l80_80659

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l80_80659
