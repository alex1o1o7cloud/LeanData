import Mathlib

namespace sum_of_possible_N_l595_595042

theorem sum_of_possible_N (x y : ℝ) (N : set ℝ) (h_nonzero : y ≠ 0)
    (h_sets_equal : { (x + y)^2, (x - y)^2, x * y, x / y } = { 4, 12.8, 28.8, N }) :
    ∑ n in N, n = 85.2 := 
sorry

end sum_of_possible_N_l595_595042


namespace sin_30_eq_half_l595_595365

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595365


namespace judy_expense_correct_l595_595511

noncomputable def judy_expense : ℝ :=
  let carrots := 5 * 1
  let milk := 3 * 3
  let pineapples := 2 * 4
  let original_flour_price := 5
  let discount := original_flour_price * 0.25
  let discounted_flour_price := original_flour_price - discount
  let flour := 2 * discounted_flour_price
  let ice_cream := 7
  let total_no_coupon := carrots + milk + pineapples + flour + ice_cream
  if total_no_coupon >= 30 then total_no_coupon - 10 else total_no_coupon

theorem judy_expense_correct : judy_expense = 26.5 := by
  sorry

end judy_expense_correct_l595_595511


namespace product_le_neg_inv_nat_l595_595095

theorem product_le_neg_inv_nat (n : ℕ) (x : Fin n → ℝ) (hn : ∑ i, x i = 0) (hnsq : ∑ i, (x i)^2 = 1) :
  ∃ (i j : Fin n), x i * x j ≤ -1 / n :=
sorry

end product_le_neg_inv_nat_l595_595095


namespace find_a_monotonicity_f_l595_595607

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := a - (2 / x)

-- Define the first theorem to find the value of a
theorem find_a :
  (∃ a : ℝ, 2 * (f 1 a) = f 2 a) ↔ a = 3 := by
  sorry

-- Define the second theorem to determine the monotonicity
theorem monotonicity_f :
  ∀ (x1 x2 : ℝ), x1 ∈ set.Ioo (-∞ : ℝ) 0 → x2 ∈ set.Ioo (-∞ : ℝ) 0 → x1 < x2 → f x1 3 < f x2 3 := by
  sorry

end find_a_monotonicity_f_l595_595607


namespace sin_30_eq_one_half_l595_595312

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595312


namespace general_binomial_sum_l595_595870

open Nat

theorem general_binomial_sum (n : ℕ) (h : n ≠ 0) :
  (Finset.filter (λ k => k % 4 = 1) (Finset.range (4 * n + 2))).sum (λ k => Nat.choose (4 * n + 1) k) =
  2^(4 * n - 1) + (-1)^n * 2^(2 * n - 1) :=
  sorry

end general_binomial_sum_l595_595870


namespace closest_ratio_adults_children_l595_595894

theorem closest_ratio_adults_children (a c : ℕ) 
  (h1 : 30 * a + 15 * c = 2250) 
  (h2 : a ≥ 50) 
  (h3 : c ≥ 20) : a = 50 ∧ c = 50 :=
by {
  sorry
}

end closest_ratio_adults_children_l595_595894


namespace incorrect_correlation_statement_l595_595121

-- Definitions and conditions
def linear_regression_line_passes_through_center (x̄ ȳ : ℝ) (n : ℕ) (xi yi : Fin n → ℝ) :=
  ∃ a b, ∀ i, yi i = b * xi i + a → Sum xi / n = x̄ ∧ Sum yi / n = ȳ

def least_squares_minimizes (n : ℕ) (yi xi : Fin n → ℝ) (a b : ℝ) : Prop :=
  ∀ a' b', Sum (fun i => (yi i - b' * xi i - a')^2) ≥ Sum (fun i => (yi i - b * xi i - a)^2)

def r_value (n : ℕ) (x y : Fin n → ℝ) : ℝ := sorry -- Assume a definition for correlation coefficient r

def r_squared (n : ℕ) (y y_predicted : Fin n → ℝ) (ȳ : ℝ) : ℝ :=
  1 - (Sum (fun i => (y i - y_predicted i)^2) / Sum (fun i => (y i - ȳ)^2))

-- Problem in Lean statement
theorem incorrect_correlation_statement (n : ℕ) (x y : Fin n → ℝ) (r : ℝ) :
  linear_regression_line_passes_through_center (Sum (fun i => x i) / n) (Sum (fun i => y i) / n) n x y →
  (∃ a b, least_squares_minimizes n y x a b) →
  R_squared n y (λ i, r * x i) (Sum y / n) → 
  ¬ (the_abs_smaller_the_r_weaker_the_correlation r) :=
sorry

end incorrect_correlation_statement_l595_595121


namespace n_le_60_l595_595848

theorem n_le_60 (n : ℕ) (h1 : ∃ k : ℤ, (1 / 2 + 1 / 3 + 1 / 5 + 1 / n) = ↑k) : n ≤ 60 := 
sorry

end n_le_60_l595_595848


namespace compositional_pairs_count_l595_595856

open Polynomial

noncomputable def number_of_compositional_pairs (p : ℕ) (hp : Nat.Prime p) (hp_gt_two : p > 2) : ℕ :=
  if hp_gt_two then 4 * p * (p - 1) else 0

theorem compositional_pairs_count (p : ℕ)
  (hp : Nat.Prime p) (hp_gt_two : p > 2) :
  number_of_compositional_pairs p hp hp_gt_two = 4 * p * (p - 1) := by
  sorry

end compositional_pairs_count_l595_595856


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595639

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595639


namespace volume_of_spherical_region_in_first_octant_l595_595007

theorem volume_of_spherical_region_in_first_octant
  (S : Set (ℝ × ℝ × ℝ))
  (h1 : ∀ (x y z : ℝ), x^2 + y^2 + z^2 ≤ 1 → x ≥ 0 → y ≥ 0 → z ≥ 0 → (x, y, z) ∈ S) :
  ∫ (x : ℝ) in 0..1, ∫ (y : ℝ) in 0..sqrt (1 - x^2), ∫ (z : ℝ) in 0..sqrt (1 - x^2 - y^2), z dx dy dz = π / 6 :=
sorry

end volume_of_spherical_region_in_first_octant_l595_595007


namespace count_ordered_triples_2023_l595_595155

-- Definitions of the right rectangular prism and the conditions
def is_similar (a b c : ℕ) (x y z : ℕ) : Prop :=
  x * b = a * y ∧ x * c = a * z

def ordered_triples (b : ℕ) : ℕ :=
  let D := 2023 * 2023 in
  (Nat.divisors D).count (λ d, d < b ∧ D / d > b)

-- The theorem to prove the number of such ordered triples
theorem count_ordered_triples_2023 :
  ordered_triples 2023 = 7 :=
by
  sorry

end count_ordered_triples_2023_l595_595155


namespace sin_30_eq_half_l595_595208

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595208


namespace sin_thirty_deg_l595_595228

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595228


namespace connie_initial_marbles_l595_595493

def marbles_connie_gave_to_juan : ℝ := 183.5
def marbles_connie_gave_to_maria : ℝ := 245.7
def marbles_connie_received_from_mike : ℝ := 50.3
def marbles_connie_has_now : ℝ := 593.2

theorem connie_initial_marbles : 
  let x := marbles_connie_has_now + (marbles_connie_gave_to_juan + marbles_connie_gave_to_maria - marbles_connie_received_from_mike) in
  x = 972.1 :=
by
  sorry

end connie_initial_marbles_l595_595493


namespace exists_set_condition_l595_595960

theorem exists_set_condition (n : ℕ) (p : ℕ) (h₁ : n ≥ 2) (h₂ : prime p) (h₃ : p ∣ n) :
  ∃ (A : Finset ℕ), (∀ a b ∈ A, a * b % ∑ x in (Finset.powersetLen p A), x = 0) :=
sorry

end exists_set_condition_l595_595960


namespace omega_range_l595_595592

theorem omega_range (omega : ℝ) :
  (∀ x : ℝ, -π/2 < x ∧ x < π/2 → (tan (omega * x)).monotone_decreasing) →
  -1 ≤ omega ∧ omega < 0 :=
by
  sorry

end omega_range_l595_595592


namespace sum_of_all_possible_values_f1_l595_595031

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_all_possible_values_f1 :
  (∀ x y : ℝ, f(f(x - y)) = f(x) * f(y) - f(x) + f(y) - x * y) →
  (f 1 = -1) :=
by
  sorry

end sum_of_all_possible_values_f1_l595_595031


namespace sin_30_is_half_l595_595334

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595334


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595713

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595713


namespace sin_30_eq_half_l595_595474

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595474


namespace sin_30_eq_half_l595_595364

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595364


namespace find_maximum_value_on_interval_l595_595498

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : f(x + y) = f(x) + f(y)
axiom positivity (x : ℝ) : 0 < x → 0 < f(x)
axiom value_at_two : f(2) = 4

theorem find_maximum_value_on_interval : ∃ M, ∀ x ∈ Icc (-2012) (-100), f(x) ≤ M ∧ M = -200 := sorry

end find_maximum_value_on_interval_l595_595498


namespace sin_30_eq_half_l595_595451

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595451


namespace product_of_k_l595_595958

theorem product_of_k (k : ℝ) (h : ∀ x : ℝ, k * x^2 + (k + 5) * x + 5 = 0 → (x - ((10 - 10) / 2) = 0 → k = 5)) :
  ∃ k, k = 5 ∧ ∏ x in {k}, (k: ℝ) = 5 := by
  sorry

end product_of_k_l595_595958


namespace inequality_solution_l595_595091

noncomputable def solution_set (x : ℝ) : Prop :=
  (∃ t : ℝ, t = (\frac{-1 - Real.sqrt 2}{5}) ∧ t ≤ x ∧ x < -\frac{2}{5}) ∨
  (∃ t : ℝ, t = (\frac{Real.sqrt 2 - 1}{5}) ∧ t ≤ x ∧ x < ∞)

theorem inequality_solution (x : ℝ) :
  log 5 (5*x^2 + 2*x) * log 5 (5 + 2/x) > log 5 (5*x^2) ↔ solution_set x :=
sorry

end inequality_solution_l595_595091


namespace find_x_eq_nine_fourths_l595_595535

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l595_595535


namespace count_triples_satisfying_conditions_l595_595501

theorem count_triples_satisfying_conditions :
  (∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ ab + bc = 72 ∧ ac + bc = 35) → 
  ∃! t : (ℕ × ℕ × ℕ), 0 < t.1 ∧ 0 < t.2.1 ∧ 0 < t.2.2 ∧ 
                     t.1 * t.2.1 + t.2.1 * t.2.2 = 72 ∧ 
                     t.1 * t.2.2 + t.2.1 * t.2.2 = 35 :=
by sorry

end count_triples_satisfying_conditions_l595_595501


namespace find_x_of_floor_plus_x_eq_17_over_4_l595_595526

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l595_595526


namespace angle_BCD_in_quadrilateral_l595_595190

theorem angle_BCD_in_quadrilateral :
  ∀ (A B D BCD : ℝ), A = 60 ∧ B = 30 ∧ D = 20 → A + B + D + BCD = 360 → BCD = 250 :=
by
  intros A B D BCD h1 h2
  cases h1 with hA hrest
  cases hrest with hB hD
  rw [hA, hB, hD] at h2
  linarith

-- Skipping the proof for the purpose of this task

end angle_BCD_in_quadrilateral_l595_595190


namespace unoccupied_seats_in_business_class_l595_595163

def airplane_seating (fc bc ec : ℕ) (pfc pbc pec : ℕ) : Nat :=
  let num_ec := ec / 2
  let num_bc := num_ec - pfc
  bc - num_bc

theorem unoccupied_seats_in_business_class :
  airplane_seating 10 30 50 3 (50/2) (50/2) = 8 := by
    sorry

end unoccupied_seats_in_business_class_l595_595163


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595632

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595632


namespace perfect_square_after_dividing_l595_595110

theorem perfect_square_after_dividing (n : ℕ) (h : n = 16800) : ∃ m : ℕ, (n / 21) = m * m :=
by {
  sorry
}

end perfect_square_after_dividing_l595_595110


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595725

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595725


namespace sin_30_eq_half_l595_595424

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595424


namespace sin_30_eq_half_l595_595439

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595439


namespace sin_of_30_degrees_l595_595265

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595265


namespace min_value_of_S_diff_l595_595570

def a_seq (a : ℕ → ℤ) := ∀ n : ℕ, n > 0 → (a n = (2 * n - 5) * (n - 6))

def S_n (S : ℕ → ℤ) (a : ℕ → ℤ) := ∀ n : ℕ, S n = ∑ i in Finset.range (n + 1), a i

def min_S_diff (S : ℕ → ℤ) (m n : ℕ) (h_nm : n > m) := S n - S m

theorem min_value_of_S_diff :
  ∃ (S : ℕ → ℤ) (a : ℕ → ℤ), 
  a_seq a ∧ 
  S_n S a ∧ 
  ∀ (n m : ℕ), n > m → min_S_diff S m n = -14 :=
sorry

end min_value_of_S_diff_l595_595570


namespace find_x_l595_595541

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l595_595541


namespace solution_l595_595491

noncomputable def problem (i : ℂ) (h : i^2 = -1) : ℂ :=
  ∑ k in finset.range (202), i^(k - 101)

theorem solution (i : ℂ) (h : i^2 = -1) :
  problem i h = -i :=
by
  sorry

end solution_l595_595491


namespace min_A_plus_n_l595_595564

theorem min_A_plus_n {n : ℕ} (a : Fin n → ℕ) (h_distinct : Function.Injective a) 
  (h_positive : ∀ i, 0 < a i) (h_sum : ∑ i in Finset.univ, a i = 2000) 
  (A : ℕ) (h_max : A = Finset.max' (Finset.image a Finset.univ) (Finset.univ_nonempty.subset (Finset.image_subset (λ i j h, h) (function.Injective.apply_implies_eq_iff.mpr h_distinct)))) : 
  A + n ≥ 110 :=
by
  sorry

end min_A_plus_n_l595_595564


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595715

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595715


namespace increasing_ln_plus_2_on_pos_reals_l595_595184

theorem increasing_ln_plus_2_on_pos_reals :
  (∀ x : ℝ, 0 < x → (λ y, -real.sqrt (y + 1)) x ≤ (λ y, -real.sqrt (y + 1)) (x + 1)) ∧
  (∀ x : ℝ, 0 < x → (λ y, (1/2 : ℝ)^y) x ≤ (λ y, (1/2 : ℝ)^y) (x + 1)) ∧
  (∀ x : ℝ, 0 < x → 0 < x ∧ x < 1 → (λ y, y + 1/y) x ≤ (λ y, y + 1/y) (x + 1)) ∧
  (∀ x : ℝ, 1 < x → (λ y, y + 1/y) x ≤ (λ y, y + 1/y) (x + 1)) →
  (∀ x : ℝ, 0 < x → (λ y, real.log y + 2) x < (λ y, real.log y + 2) (x + 1)) :=
  sorry

end increasing_ln_plus_2_on_pos_reals_l595_595184


namespace sin_30_eq_half_l595_595411

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595411


namespace sin_30_eq_half_l595_595242

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595242


namespace savings_at_end_of_year_l595_595012

def income := 10000
def tax_rate := 0.15
def investment_rate := 0.10
def expenditure_ratio := (3, 5)

theorem savings_at_end_of_year :
  let taxes := tax_rate * income in
  let investments := investment_rate * income in
  let total_deductions := taxes + investments in
  let remaining_income := income - total_deductions in
  ∃ (x : ℝ), let expenditures := (expenditure_ratio.1 / expenditure_ratio.2) * income in
             remaining_income - expenditures = 1500 :=
by
  let taxes := tax_rate * income
  let investments := investment_rate * income
  let total_deductions := taxes + investments
  let remaining_income := income - total_deductions
  let expenditures := (expenditure_ratio.1 / expenditure_ratio.2) * income
  exists 2000
  have H : remaining_income - expenditures = 1500
  exact sorry

end savings_at_end_of_year_l595_595012


namespace sin_30_eq_one_half_l595_595314

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595314


namespace AD_square_eq_mb_square_plus_nc_square_subtracted_mna_square_l595_595819

theorem AD_square_eq_mb_square_plus_nc_square_subtracted_mna_square 
  {α : Type*} [linear_ordered_field α] 
  {A B C D : α} {a b c m n : α} 
  (hD_on_BC : D ∈ segment B C) 
  (BD_eq_ma : dist B D = m * a) 
  (DC_eq_na : dist D C = n * a) 
  (m_add_n_eq_one : m + n = 1) 
  : dist A D ^ 2 = m * (dist A B ^ 2) + n * (dist A C ^ 2) - m * n * a ^ 2 := 
by skip -- sorry to skip the proof.

end AD_square_eq_mb_square_plus_nc_square_subtracted_mna_square_l595_595819


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595722

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595722


namespace at_least_one_greater_l595_595842

theorem at_least_one_greater (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = a * b * c) :
  a > 17 / 10 ∨ b > 17 / 10 ∨ c > 17 / 10 :=
sorry

end at_least_one_greater_l595_595842


namespace exists_solution_l595_595038

noncomputable def smallest_c0 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 1) : ℕ :=
  a * b - a - b + 1

theorem exists_solution (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 1) :
  ∃ c0, (c0 = smallest_c0 a b ha hb h) ∧ ∀ c : ℕ, c ≥ c0 → ∃ x y : ℕ, a * x + b * y = c :=
sorry

end exists_solution_l595_595038


namespace sin_30_eq_half_l595_595475

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595475


namespace sin_30_eq_half_l595_595457

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595457


namespace distinct_prime_factors_of_sigma_450_l595_595676
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595676


namespace sin_thirty_deg_l595_595219

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595219


namespace infinite_primes_of_form_l595_595843

theorem infinite_primes_of_form (p : ℕ) (hp : Nat.Prime p) (hpodd : p % 2 = 1) :
  ∃ᶠ n in at_top, Nat.Prime (2 * p * n + 1) :=
sorry

end infinite_primes_of_form_l595_595843


namespace simplify_fraction_rationalize_denominator_l595_595884

theorem simplify_fraction_rationalize_denominator :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = 5 * Real.sqrt 2 / 28 :=
by
  have sqrt_50 : Real.sqrt 50 = 5 * Real.sqrt 2 := sorry
  have sqrt_8 : 3 * Real.sqrt 8 = 6 * Real.sqrt 2 := sorry
  have sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := sorry
  sorry

end simplify_fraction_rationalize_denominator_l595_595884


namespace sin_30_eq_half_l595_595392

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595392


namespace max_sum_index_l595_595572

-- Sequence definition and initial conditions
def seq {α : Type} [Add α] [Mul α] [HasSmul ℕ α] (a n : ℕ → α) : Prop :=
  ∀ n, a n + a (n + 2) = 2 * a (n + 1)

def initial_conditions {α : Type} [HasZero α] [HasZero α] (a : ℕ → α) : Prop :=
  a 1 = 13 ∧ a 2 = 11

-- The statement that encapsulates the problem and its maximum point
theorem max_sum_index (a : ℕ → ℤ) (n : ℕ) (Sn : ℕ → ℤ)
  (h_seq : seq a n)
  (h_initial_conditions : initial_conditions a)
  (h_Sn : ∀ n, Sn n = 13 * n + (n * (n - 1) / 2) * (-2)) :
  ∃ m, ∀ n, Sn m ≥ Sn n ∧ m = 7 :=
sorry

end max_sum_index_l595_595572


namespace writing_rate_l595_595832

theorem writing_rate (nathan_rate : ℕ) (jacob_rate : ℕ) : nathan_rate = 25 → jacob_rate = 2 * nathan_rate → (nathan_rate + jacob_rate) * 10 = 750 :=
by
  assume h1 : nathan_rate = 25,
  assume h2 : jacob_rate = 2 * nathan_rate,
  have combined_rate : nathan_rate + jacob_rate = 75, from sorry, -- From calculation in solution step
  show (nathan_rate + jacob_rate) * 10 = 750, from sorry -- Multiplying by 10 as per solution step


end writing_rate_l595_595832


namespace dow_jones_morning_value_l595_595069

theorem dow_jones_morning_value 
  (end_of_day_value : ℝ) 
  (percentage_fall : ℝ)
  (expected_morning_value : ℝ) 
  (h1 : end_of_day_value = 8722) 
  (h2 : percentage_fall = 0.02) 
  (h3 : expected_morning_value = 8900) :
  expected_morning_value = end_of_day_value / (1 - percentage_fall) :=
sorry

end dow_jones_morning_value_l595_595069


namespace expansion_constant_term_l595_595075

/-
Problem Statement:
Prove that the constant term in the expansion of (x^2 + 1/x^2 - 2)^3 is -20.
-/

theorem expansion_constant_term :
  let f (x : ℚ) := (x^2 + x⁻² - 2)^3 in
  coeff 0 (expansion f) = -20 :=
by
  sorry

end expansion_constant_term_l595_595075


namespace dice_probability_l595_595864

noncomputable def probability_event (event_count : ℕ) (total_count : ℕ) : ℚ := 
  event_count / total_count

theorem dice_probability :
  let event_first_die := 3
  let event_second_die := 3
  let total_outcomes_first := 8
  let total_outcomes_second := 8
  probability_event event_first_die total_outcomes_first * probability_event event_second_die total_outcomes_second = 9 / 64 :=
by
  sorry

end dice_probability_l595_595864


namespace sum_of_s_and_t_eq_neg11_l595_595790

theorem sum_of_s_and_t_eq_neg11 (s t : ℝ) 
  (h1 : ∀ x, x = 3 → x^2 + s * x + t = 0)
  (h2 : ∀ x, x = -4 → x^2 + s * x + t = 0) :
  s + t = -11 :=
sorry

end sum_of_s_and_t_eq_neg11_l595_595790


namespace find_d_l595_595896

theorem find_d (
  x : ℝ
) (
  h1 : 3 * x + 8 = 5
) (
  d : ℝ
) (
  h2 : d * x - 15 = -7
) : d = -8 :=
by
  sorry

end find_d_l595_595896


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595758

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595758


namespace intersection_height_proof_l595_595925

-- Define the conditions
def pole_height_1 : ℝ := 30
def pole_height_2 : ℝ := 50
def pole_distance : ℝ := 120

-- Define the problem statement using Lean's theorem construct
theorem intersection_height_proof :
  let line1 := (λ x : ℝ, - (1 / 4 : ℝ) * x + pole_height_1)
  let line2 := (λ x : ℝ, (5 / 12 : ℝ) * x)
  ∃ x y : ℝ, x = 45 ∧ line1 x = y ∧ line2 x = y ∧ y = 18.75 :=
by
  sorry

end intersection_height_proof_l595_595925


namespace distinct_prime_factors_sum_divisors_450_l595_595744

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595744


namespace num_distinct_prime_factors_sum_divisors_450_l595_595654

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595654


namespace distinct_permutations_of_12233_l595_595627

def numFiveDigitIntegers : ℕ :=
  Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2)

theorem distinct_permutations_of_12233 : numFiveDigitIntegers = 30 := by
  sorry

end distinct_permutations_of_12233_l595_595627


namespace sequence_bounds_for_all_l595_595005

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 2 else 1/2 * sequence (n-1) + 3

theorem sequence_bounds_for_all (n : ℕ) : 2 ≤ sequence n ∧ sequence n < sequence (n+1) ∧ sequence (n+1) < 6 :=
by {
  sorry
}

end sequence_bounds_for_all_l595_595005


namespace find_x_eq_nine_fourths_l595_595532

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l595_595532


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595667

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595667


namespace sin_30_eq_half_l595_595413

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595413


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595636

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595636


namespace sin_30_eq_half_l595_595449

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595449


namespace thirteen_in_base_two_l595_595516

theorem thirteen_in_base_two : nat.to_binary 13 = "1101" :=
sorry

end thirteen_in_base_two_l595_595516


namespace time_to_reach_ship_l595_595127

def scubaDivingTime (depth rate: ℕ) : ℕ := depth / rate

theorem time_to_reach_ship 
  (rate_of_descent : ℕ)
  (depth_of_ship : ℕ)
  (H_rate : rate_of_descent = 30)
  (H_depth : depth_of_ship = 2400):
  scubaDivingTime depth_of_ship rate_of_descent = 80 :=
by
  rw [H_rate, H_depth]
  unfold scubaDivingTime
  norm_num
  sorry

end time_to_reach_ship_l595_595127


namespace sin_thirty_deg_l595_595234

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595234


namespace sin_30_eq_half_l595_595409

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595409


namespace people_between_Tara_Uma_l595_595885

-- We define the names as constants as we would do similarly in the problem statement.
constant Pat Qasim Roman Sam Tara Uma : Type

-- Conditions as given in the problem
constant positions : Fin 6 → Type
constant between (x y z : Type) : Prop

axiom three_between : between Pat Qasim (positions 1) ∧ between Pat Qasim (positions 2) ∧ between Pat Qasim (positions 3)
axiom two_between : between Qasim Roman (positions 1) ∧ between Qasim Roman (positions 2)
axiom one_between : between Roman Sam (positions 1)
axiom not_at_end : ∀ p, ¬(p = Sam)

-- The proof goal
theorem people_between_Tara_Uma : ∃ k, between Tara Uma (positions k) ∧ k = 2 := 
sorry

end people_between_Tara_Uma_l595_595885


namespace sum_of_divisors_prime_factors_450_l595_595729

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595729


namespace sin_thirty_degree_l595_595276

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595276


namespace find_other_parallel_side_l595_595547

-- Defining the conditions and proof statement
def trapezium_area (a b h : ℕ) : ℕ := (a + b) * h / 2

theorem find_other_parallel_side :
  ∃ (b : ℕ), let a := 20, h := 20, A := 380 in
             trapezium_area a b h = A ∧ b = 18 :=
by
  sorry

end find_other_parallel_side_l595_595547


namespace correct_operations_l595_595200

variable (x : ℚ)

def incorrect_equation := ((x - 5) * 3) / 7 = 10

theorem correct_operations :
  incorrect_equation x → (3 * x - 5) / 7 = 80 / 7 :=
by
  intro h
  sorry

end correct_operations_l595_595200


namespace second_drain_rate_l595_595961

theorem second_drain_rate :
  ∀ (pipe_rate drain1_rate net_fill_rate : ℝ), pipe_rate = 0.5 ∧ drain1_rate = 0.25 ∧ net_fill_rate = 0.0833 → 
  ∃ (x : ℝ), 0.5 - 0.25 - x = net_fill_rate ∧ x = 0.1667 :=
by
  intros pipe_rate drain1_rate net_fill_rate hpipeline
  rcases hpipeline with ⟨hpipe, hdrain1, hnet⟩
  use 0.1667
  split
  {
    rw [hpipe, hdrain1, hnet],
    norm_num,
  }
  {
    -- Proof steps would go here
    sorry,
  }

end second_drain_rate_l595_595961


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595669

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595669


namespace triangle_side_inequality_l595_595039

theorem triangle_side_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) : 
  (a / (b + c - a)) + (b / (c + a - b)) + (c / (a + b - c)) ≥ 3 := 
begin
  sorry
end

end triangle_side_inequality_l595_595039


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595761

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595761


namespace area_AFCH_l595_595871

-- Definition of dimensions and areas
def AB := 9
def BC := 5
def EF := 3
def FG := 10

def area (length width: ℕ) : ℕ := length * width

def area_ABCD := area AB BC
def area_EFGH := area EF FG
def area_overlap := area BC EF

-- Proof that the area of AFCH is 52.5
theorem area_AFCH : (area_ABCD + area_EFGH - area_overlap) / 2 + area_overlap = 52.5 := 
by
  sorry  -- Proof omitted

end area_AFCH_l595_595871


namespace sin_30_eq_one_half_l595_595320

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595320


namespace part1_part2_l595_595609

section

variables {x m : ℝ}

def f (x m : ℝ) : ℝ := 3 * x^2 + (4 - m) * x - 6 * m
def g (x m : ℝ) : ℝ := 2 * x^2 - x - m

theorem part1 (m : ℝ) (h : m = 1) : 
  {x : ℝ | f x m > 0} = {x : ℝ | x < -2 ∨ x > 1} :=
sorry

theorem part2 (m : ℝ) (h : m > 0) : 
  {x : ℝ | f x m ≤ g x m} = {x : ℝ | -5 ≤ x ∧ x ≤ m} :=
sorry
     
end

end part1_part2_l595_595609


namespace sin_30_eq_half_l595_595301
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595301


namespace sin_thirty_deg_l595_595220

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595220


namespace sin_30_deg_l595_595351

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595351


namespace sin_30_eq_half_l595_595489

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595489


namespace sin_30_eq_half_l595_595476

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595476


namespace sin_30_eq_half_l595_595210

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595210


namespace unique_parallel_line_through_point_l595_595567

-- Define the setup
variables (α : Type*) [plane α] (l : Type*) [line l] (P : α)

-- Assumptions
variable (h1 : parallel_to_plane l α)
variable (h2 : P ∈ α)

-- Proof Statement
theorem unique_parallel_line_through_point (h1 : parallel_to_plane l α) (h2 : ∈ α) :
  ∃! m : Type*, [line m] ∧ (P ∈ m) ∧ (parallel_to m l) ∧ (m ⊆ α) := 
  sorry

end unique_parallel_line_through_point_l595_595567


namespace geometric_series_reciprocal_sum_l595_595585

theorem geometric_series_reciprocal_sum : 
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ), 
    (a 1 = 1) →
    (∀ n, a (n + 1) = 2 ^ n) →
    (∀ n, S n = (∑ i in range n, a i)) →
    9 * S 3 = S 6 →
  let b (n : ℕ) := (λ n, 1 / a n) in
  (∑ i in range 5, b i) = 31 / 16 :=
by
  assume a S ha ha' hS hS_eq
  let b := λ n, 1 / a n
  sorry

end geometric_series_reciprocal_sum_l595_595585


namespace sin_30_eq_half_l595_595416

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595416


namespace constant_term_binomial_expansion_l595_595892

open Nat

theorem constant_term_binomial_expansion : 
  let expr := (2*x - (1/real.sqrt x))
  let power := 6
  ∀ r, (expr ^ power)^.binomial 6 r = (-1)^r * (6.choose r) * 2^(6-r) * x^(6 - (3/2)*r)
  ∃ r, (6 - (3/2)*r = 0) → ((-1)^r * (6.choose r) * 2^(6-r) * x^(6 - (3/2)*r) = 60) := by
  sorry

end constant_term_binomial_expansion_l595_595892


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595719

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595719


namespace find_angle_bisector_length_l595_595818

noncomputable def length_of_angle_bisector (a b c : ℝ) (cosA : ℝ) : ℝ :=
  let cos_half_A := Real.sqrt ((1 + cosA) / 2) in
  (2 * a * b * cos_half_A) / (a + b)

theorem find_angle_bisector_length :
  ∀ (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C],
    ∀ (a b c cosA : ℝ),
      a = 4 →
      b = 8 →
      cosA = 1 / 10 →
      length_of_angle_bisector a b c cosA = (16 * Real.sqrt 0.55) / 3 
:=
by
  intros A B C _ _ _ a b c cosA ha hb hcosA
  simp [length_of_angle_bisector, ha, hb, hcosA]
  sorry

end find_angle_bisector_length_l595_595818


namespace sin_of_30_degrees_l595_595258

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595258


namespace max_f_on_interval_l595_595085

noncomputable def f : ℝ → ℝ := λ x, (Real.sqrt 3) * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

theorem max_f_on_interval : 
  ∃ x ∈ set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 :=
begin
  -- Proof omitted
  sorry
end

end max_f_on_interval_l595_595085


namespace value_of_y_l595_595117

theorem value_of_y (y : ℝ) : 8^12 + 8^12 + 8^12 = 2^y → y = 36 + Real.log2 3 :=
by
  sorry

end value_of_y_l595_595117


namespace pieces_left_to_place_l595_595049

noncomputable def total_pieces : ℕ := 300
noncomputable def reyn_pieces : ℕ := 25
noncomputable def rhys_pieces : ℕ := 2 * reyn_pieces
noncomputable def rory_pieces : ℕ := 3 * reyn_pieces
noncomputable def placed_pieces : ℕ := reyn_pieces + rhys_pieces + rory_pieces
noncomputable def remaining_pieces : ℕ := total_pieces - placed_pieces

theorem pieces_left_to_place : remaining_pieces = 150 :=
by sorry

end pieces_left_to_place_l595_595049


namespace sin_30_eq_half_l595_595446

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595446


namespace fixed_point_of_transformed_exponential_l595_595605

noncomputable theory

open Function

theorem fixed_point_of_transformed_exponential (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  ∃ P : ℝ × ℝ, (let f := λ x : ℝ, 4 + a^(x - 1) in P = (1, 5) ∧ ∀ x, f(x) = P.snd ↔ x = P.fst) := 
by
  sorry

end fixed_point_of_transformed_exponential_l595_595605


namespace decimal_truncation_division_results_l595_595812

theorem decimal_truncation_division_results (α : ℝ) (hα : α > 0) :
  let n := (α * 100).toInt
  let α₁ := α - (n / 100)
  let truncated := n / 100
  let quotient := (truncated / α).floor / 100
  quotient ∈ {0, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 
              0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 
              0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 
              0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1} :=
by
  sorry


end decimal_truncation_division_results_l595_595812


namespace unoccupied_business_class_seats_l595_595162

theorem unoccupied_business_class_seats :
  ∀ (first_class_seats business_class_seats economy_class_seats economy_fullness : ℕ)
  (first_class_occupancy : ℕ) (total_business_first_combined economy_occupancy : ℕ),
  first_class_seats = 10 →
  business_class_seats = 30 →
  economy_class_seats = 50 →
  economy_fullness = economy_class_seats / 2 →
  economy_occupancy = economy_fullness →
  total_business_first_combined = economy_occupancy →
  first_class_occupancy = 3 →
  total_business_first_combined = first_class_occupancy + business_class_seats - (business_class_seats - total_business_first_combined + first_class_occupancy) →
  business_class_seats - (total_business_first_combined - first_class_occupancy) = 8 :=
by
  intros first_class_seats business_class_seats economy_class_seats economy_fullness 
         first_class_occupancy total_business_first_combined economy_occupancy
         h1 h2 h3 h4 h5 h6 h7 h8 
  rw [h1, h2, h3, h4, h5, h6, h7] at h8
  sorry

end unoccupied_business_class_seats_l595_595162


namespace sin_thirty_degree_l595_595274

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595274


namespace find_x_of_floor_plus_x_eq_17_over_4_l595_595529

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l595_595529


namespace sin_thirty_degree_l595_595279

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595279


namespace max_apples_apples_difference_l595_595997

theorem max_apples (A B C : ℕ) (h1 : A + B + C < 100) (h2 : B = (5 * A) / 12) (h3 : C = (7 * A) / 12) :
  A ≤ 48 := 
sorry

theorem apples_difference (A B C : ℕ) (h1 : A + B + C < 100) (h2 : B = (5 * A) / 12) (h3 : C = (7 * A) / 12) :
  B - C = -(A / 6) :=
sorry

end max_apples_apples_difference_l595_595997


namespace num_distinct_prime_factors_sum_divisors_450_l595_595656

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595656


namespace geometric_sequence_sum_l595_595615

variable (a : ℕ → ℝ)

def is_geometric (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) * a (n - 1) = a n ^ 2

def S_9 (a : ℕ → ℝ) : ℝ :=
∑ i in finset.range 9, a (i + 1)

theorem geometric_sequence_sum (h_geom : is_geometric a)
  (h_sum1 : a 1 + a 2 + a 3 = 40)
  (h_sum2 : a 4 + a 5 + a 6 = 20) :
  S_9 a = 70 :=
sorry

end geometric_sequence_sum_l595_595615


namespace distinct_prime_factors_of_sigma_450_l595_595672
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595672


namespace sin_30_eq_half_l595_595482

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595482


namespace equal_share_expense_l595_595019

theorem equal_share_expense (L B C X : ℝ) : 
  let T := L + B + C - X
  let share := T / 3 
  L + (share - L) == (B + C - X - 2 * L) / 3 := 
by
  sorry

end equal_share_expense_l595_595019


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595707

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595707


namespace count_valid_a_values_l595_595555

theorem count_valid_a_values : 
  ∃ (a_values : List ℕ), 
    (∀ (a ∈ a_values), 0 < a ∧ a ≤ 50 ∧ 
      ∃ (k : ℤ), 5 * (a : ℤ)^2 + 12 * a + 4 = k^2) ∧ 
    a_values.length = 2 := 
  sorry

end count_valid_a_values_l595_595555


namespace probability_sum_is_5_l595_595923

open ProbabilityTheory MeasureTheory

-- Define the dice with numbers 0 through 5.
def diceNumber : Fin 6 := {
  zero,
  one,
  two,
  three,
  four,
  five
}

-- Define the sample space for a single die (six faces).
def dieOutcome := {0, 1, 2, 3, 4, 5}

-- Define the sample space for two dice.
def sampleSpace := (dieOutcome × dieOutcome)

-- Define the event of interest: Sum of top faces is 5.
def eventSum5 (outcome : sampleSpace) : Prop := 
  outcome.1 + outcome.2 = 5

-- Define the probability space.
def probabilitySpace : ProbabilitySpace sampleSpace := prodUniformSpace dieOutcome dieOutcome

-- Define the probability of an event.
noncomputable def probability (e : set sampleSpace) : ℝ :=
  (measureOf probabilitySpace).measure e / (measureOf probabilitySpace).measure univ

theorem probability_sum_is_5 : probability {outcome | eventSum5 outcome} = 1 / 3 :=
by sorry

end probability_sum_is_5_l595_595923


namespace no_square_number_solution_sum_zero_l595_595089

theorem no_square_number_solution_sum_zero :
  ∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x * y * z = 4 * (x + y + z)) ∧ (x * y = z + x) →
  (∄ N : ℕ, (N = x * y * z) ∧ (∃ k : ℕ, N = k * k)) →
  0 := 
by
  intros x y z hconds h_no_square
  exact 0

end no_square_number_solution_sum_zero_l595_595089


namespace inverse_function_value_l595_595604

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 4 then x^2 + 1 else if -4 ≤ x ∧ x < 0 then 2^x else 0

noncomputable def f_inv (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x ≤ 17 then real.sqrt (x - 1) else if (1/16 : ℝ) ≤ x ∧ x < 1 then real.log x / real.log 2 else 0

theorem inverse_function_value :
  f_inv 4 + f_inv (1 / 4) = real.sqrt 3 - 2 :=
by sorry

end inverse_function_value_l595_595604


namespace sin_30_eq_half_l595_595399

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595399


namespace sin_30_eq_half_l595_595402

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595402


namespace tangent_line_at_zero_l595_595895

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.exp x

theorem tangent_line_at_zero :
  ∃ (m b : ℝ), (∀ x y, y = f(x) → y = m * x + b) ∧ (m = 1) ∧ (b = -1) :=
by
  sorry

end tangent_line_at_zero_l595_595895


namespace sin_30_eq_half_l595_595236

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595236


namespace question1_question2_l595_595177

-- Definitions of the required geometric entities.
variables {A B C P T: Type} -- Points
variables [EquilateralTriangle A B C] -- ABC is an equilateral triangle
variables [Circle (center := A) (radius := (2: ℝ))] -- S is the circle with diameter AB
variables {R: ℝ} -- Radius of circle S
variables [PointOnSegment P A C] -- P is on AC
variables [TangentCircle P C (radius := PC)] -- Circle with center P and radius PC touches S at T

-- Statement for the first part
theorem question1 (h₁ : EquilateralTriangle ABC) (h₂ : Circle S (Diameter AB)) (h₃ : PointOnSegment P AC)
  (h₄ : TangentCircle P C (radius := PC)) :
  AP / AC = 4 / 5 := sorry

-- Statement for the second part
theorem question2 (h₁ : EquilateralTriangle ABC) (h₂ : Circle S (Diameter AB)) (h₃ : PointOnSegment P AC)
  (h₄ : TangentCircle P C (radius := PC)) :
  AT / AC = sqrt (3 / 7) := sorry

end question1_question2_l595_595177


namespace proof_min_k_l595_595509

-- Define the number of teachers
def num_teachers : ℕ := 200

-- Define what it means for a teacher to send a message to another teacher.
-- Represent this as a function where each teacher sends a message to exactly one other teacher.
def sends_message (teachers : Fin num_teachers → Fin num_teachers) : Prop :=
  ∀ i : Fin num_teachers, ∃ j : Fin num_teachers, teachers i = j

-- Define the main proposition: there exists a group of 67 teachers where no one sends a message to anyone else in the group.
def min_k (teachers : Fin num_teachers → Fin num_teachers) : Prop :=
  ∃ (k : ℕ) (reps : Fin k → Fin num_teachers), k ≥ 67 ∧
  ∀ (i j : Fin k), i ≠ j → teachers (reps i) ≠ reps j

theorem proof_min_k : ∀ (teachers : Fin num_teachers → Fin num_teachers),
  sends_message teachers → min_k teachers :=
sorry

end proof_min_k_l595_595509


namespace distinct_prime_factors_of_sigma_450_l595_595675
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595675


namespace area_of_triangle_XOY_l595_595820

theorem area_of_triangle_XOY {XYZ : Type} [metric_space XYZ] 
  (M midpoint_of XY : XYZ) 
  (N midpoint_of XM : XYZ)
  (O midpoint_of NX : XYZ)
  (area_XYZ : Real := 162) :
  ∃ (A_XOY : Real), A_XOY = 20.25 :=
  sorry

end area_of_triangle_XOY_l595_595820


namespace closest_fraction_l595_595191

theorem closest_fraction :
  let f := 23 / 120 : ℝ in
  let options := [1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8] : List ℝ in
  let closest := List.foldl (λ (closest : ℝ × ℝ) (x : ℝ), if abs (f - x) < abs (f - closest.1) then (x, abs (f - x)) else closest) (1 / 4, abs (f - 1 / 4)) options in
  closest.1 = 1 / 5 :=
by
  -- proof steps go here
  sorry

end closest_fraction_l595_595191


namespace find_S_l595_595580

theorem find_S (a b : ℝ) (R : ℝ) (S : ℝ)
  (h1 : a + b = R) 
  (h2 : a^2 + b^2 = 12)
  (h3 : R = 2)
  (h4 : S = a^3 + b^3) : S = 32 :=
by
  sorry

end find_S_l595_595580


namespace redPoints_l595_595004

open Nat

def isRedPoint (x y : ℕ) : Prop :=
  (y = (x - 36) * (x - 144) - 1991) ∧ (∃ m : ℕ, y = m * m)

theorem redPoints :
  {p : ℕ × ℕ | isRedPoint p.1 p.2} = { (2544, 6017209), (444, 120409) } :=
by
  sorry

end redPoints_l595_595004


namespace max_min_distance_rectangle_l595_595569

theorem max_min_distance_rectangle 
  (a b : ℝ) (h_ab : a ≥ b) 
  (h_points : ∀ (X Y Z : Prod ℝ ℝ), point_in_or_on_rectangle X a b ∧ point_in_or_on_rectangle Y a b ∧ point_in_or_on_rectangle Z a b) :
  let max_min_distance := if (a / b) ≥ (2 / Real.sqrt 3) then (Real.sqrt ((a ^ 2) / 4 + b ^ 2))
                          else (2 * Real.sqrt (a ^ 2 + b ^ 2 - Real.sqrt 3 * a * b)) in
  true :=
sorry

def point_in_or_on_rectangle (P : Prod ℝ ℝ) (a b : ℝ) : Prop :=
  let (x, y) := P in 0 ≤ x ∧ x ≤ b ∧ 0 ≤ y ∧ y ≤ a

end max_min_distance_rectangle_l595_595569


namespace minimum_days_needed_l595_595980

theorem minimum_days_needed (n : ℕ) (h1 : 0 < n) : 
  (∑ i in finset.range n, 3 ^ i) ≥ 100 ↔ n ≥ 5 :=
by 
  sorry

end minimum_days_needed_l595_595980


namespace sin_30_eq_half_l595_595396

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595396


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595633

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595633


namespace length_of_iterated_graph_l595_595024

def f (x : ℝ) : ℝ :=
  if x ≤ 1/2 then 2 * x else 2 - 2 * x

def f_iter (n : ℕ) (x : ℝ) : ℝ :=
  nat.iterate f n x

theorem length_of_iterated_graph :
  ∀ n : ℕ, n = 2012 → 
  (λ x, f_iter n x) '' set.Icc 0 1 → 2 ^ 2011 := 
sorry

end length_of_iterated_graph_l595_595024


namespace find_x_log_eqn_l595_595517

theorem find_x_log_eqn (x: ℝ) (h : log 9 (2*x - 7) = 3/2) : x = 17 :=
by
  sorry

end find_x_log_eqn_l595_595517


namespace distinct_prime_factors_of_sigma_450_l595_595680
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595680


namespace digit_9_count_in_house_numbers_l595_595988

theorem digit_9_count_in_house_numbers (n : ℕ) (h : 1 ≤ n ∧ n ≤ 70) :
  (List.length (List.filter (λ x, x = 9) (List.map (fun x => x % 10) (List.range' 1 70)))) = 7 :=
by 
  sorry

end digit_9_count_in_house_numbers_l595_595988


namespace combinatorial_proof_example_l595_595066

theorem combinatorial_proof_example :
  nat.choose 13 3 = 286 := 
by
  sorry

end combinatorial_proof_example_l595_595066


namespace simplify_fraction_rationalize_denominator_l595_595883

theorem simplify_fraction_rationalize_denominator :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = 5 * Real.sqrt 2 / 28 :=
by
  have sqrt_50 : Real.sqrt 50 = 5 * Real.sqrt 2 := sorry
  have sqrt_8 : 3 * Real.sqrt 8 = 6 * Real.sqrt 2 := sorry
  have sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := sorry
  sorry

end simplify_fraction_rationalize_denominator_l595_595883


namespace sin_30_eq_one_half_l595_595317

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595317


namespace factorial_less_power_l595_595906

open Nat

noncomputable def factorial_200 : ℕ := 200!

noncomputable def power_100_200 : ℕ := 100 ^ 200

theorem factorial_less_power : factorial_200 < power_100_200 :=
by
  -- Proof goes here
  sorry

end factorial_less_power_l595_595906


namespace sin_30_eq_half_l595_595426

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595426


namespace find_x_of_floor_plus_x_eq_17_over_4_l595_595524

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l595_595524


namespace axis_of_symmetry_closest_y_l595_595082

-- Define the function and conditions
def trig_fun (x : ℝ) (ω : ℝ) : ℝ := Real.sin (ω * x + 5 * Real.pi / 6)

-- Define the conditions
def ω_in_range (ω : ℝ) : Prop := 0 < ω ∧ ω < Real.pi
def intersects_at_origin (ω : ℝ) : Prop := trig_fun 0 ω = 1 / 2
def intersects_at_half (ω : ℝ) : Prop := trig_fun (1 / 2) ω = 0

-- Define what needs to be proven
theorem axis_of_symmetry_closest_y (ω : ℝ) (h1 : ω_in_range ω) (h2 : intersects_at_origin ω) (h3 : intersects_at_half ω) :
  ∃ x : ℝ, x = -1 :=
sorry

end axis_of_symmetry_closest_y_l595_595082


namespace squirrels_supplies_l595_595887

theorem squirrels_supplies :
  ∀ (m f_Z f_P f_K h_Z h_P h_K : ℕ),
  m = 48 →
  f_Z + f_P + f_K = 180 →
  h_Z + h_P + h_K = 180 →
  h_Z = 2 * h_P →
  h_K = h_P + 20 →
  m + h_Z + f_Z = 168 →
  m + h_P + f_P = 168 →
  m + h_K + f_K = 168 →
  (m = 48 ∧ h_Z = 80 ∧ f_Z = 40 ∧ 
   m = 48 ∧ h_P = 40 ∧ f_P = 80 ∧ 
   m = 48 ∧ h_K = 60 ∧ f_K = 60) := 
by
  intros m f_Z f_P f_K h_Z h_P h_K
  intro hm
  intro hf_total
  intro hh_total
  intro hz
  intro hk
  intro hZ_total
  intro hP_total
  intro hK_total
  split; repeat {split}; exact hm <|> sorry

end squirrels_supplies_l595_595887


namespace sum_of_divisors_prime_factors_450_l595_595735

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595735


namespace sin_30_eq_one_half_l595_595311

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595311


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595630

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595630


namespace sin_30_eq_half_l595_595294
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595294


namespace cut_paper_to_k_pieces_l595_595016

theorem cut_paper_to_k_pieces {n : ℕ} (a : Fin n → ℝ) (h : ∀ i, 0 ≤ a i ∧ a i ≤ 1) (k : ℕ) (hk : k < n) :
  ∃ (P : List (List ℝ)), P.length = k ∧ (∀ xs ∈ P, xs ≠ []) ∧ (∀ xs ys ∈ P, |List.sum xs - List.sum ys| ≤ 1) :=
by
  sorry

end cut_paper_to_k_pieces_l595_595016


namespace mean_of_all_students_l595_595053

theorem mean_of_all_students (M A : ℕ) (m a : ℕ) (hM : M = 88) (hA : A = 68) (hRatio : m * 5 = 2 * a) : 
  (176 * a + 340 * a) / (7 * a) = 74 :=
by sorry

end mean_of_all_students_l595_595053


namespace sin_30_eq_half_l595_595370

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595370


namespace sin_30_eq_half_l595_595485

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595485


namespace sin_30_eq_half_l595_595389

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595389


namespace adjusted_speed_in_still_water_l595_595977

def original_upstream_speed := 25 -- kmph
def original_downstream_speed := 45 -- kmph
def wind_resistance_upstream_reduction := 2 -- kmph
def wind_resistance_downstream_increase := 2 -- kmph

theorem adjusted_speed_in_still_water:
  let V := 35 in -- correct answer from solution
  let S := 10 in
  (original_upstream_speed - wind_resistance_upstream_reduction + 
   (original_downstream_speed + wind_resistance_downstream_increase)) / 2 = V :=
begin
  sorry, -- proof omitted
end

end adjusted_speed_in_still_water_l595_595977


namespace sin_30_eq_half_l595_595241

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595241


namespace convert_13_to_binary_l595_595513

theorem convert_13_to_binary : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by sorry

end convert_13_to_binary_l595_595513


namespace distinct_prime_factors_sum_divisors_450_l595_595748

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595748


namespace sum_of_divisors_prime_factors_450_l595_595737

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595737


namespace jessica_monthly_car_insurance_payment_l595_595018

theorem jessica_monthly_car_insurance_payment
  (rent_last_year : ℤ := 1000)
  (food_last_year : ℤ := 200)
  (car_insurance_last_year : ℤ)
  (rent_increase_rate : ℕ := 3 / 10)
  (food_increase_rate : ℕ := 1 / 2)
  (car_insurance_increase_rate : ℕ := 3)
  (additional_expenses_this_year : ℤ := 7200) :
  car_insurance_last_year = 300 :=
by
  sorry

end jessica_monthly_car_insurance_payment_l595_595018


namespace choir_members_total_l595_595966

theorem choir_members_total
  (first_group second_group third_group : ℕ)
  (h1 : first_group = 25)
  (h2 : second_group = 30)
  (h3 : third_group = 15) :
  first_group + second_group + third_group = 70 :=
by
  sorry

end choir_members_total_l595_595966


namespace solution_set_of_f_of_x_neg_half_l595_595033

noncomputable def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom odd_f : ∀ x, f(-x) = -f(x)
axiom f_translation : ∀ x, f(x + 2) = -f(x)
axiom f_initial_segment : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(x) = 1/2 * x

-- Theorem
theorem solution_set_of_f_of_x_neg_half :
  {x : ℝ | f(x) = -1/2} = {x | ∃ k : ℤ, x = 4 * k - 1} := sorry

end solution_set_of_f_of_x_neg_half_l595_595033


namespace find_x_l595_595538

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l595_595538


namespace sin_30_eq_half_l595_595295
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595295


namespace fifteenth_prime_is_47_l595_595916

theorem fifteenth_prime_is_47 (h : ∀ n, Prime (prime_of_nat 6) ≠ 13 → n ≠ 6 → prime_of_nat n ≠ 47) : 
  prime_of_nat 15 = 47 := 
sorry

end fifteenth_prime_is_47_l595_595916


namespace intersection_of_A_and_B_l595_595621

def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} :=
by
  sorry

end intersection_of_A_and_B_l595_595621


namespace kitten_initial_length_l595_595188

def kitten_length_when_found (current_length : ℝ) (doubling_periods : ℕ) : ℝ :=
current_length / (2 ^ doubling_periods)

theorem kitten_initial_length (current_length : ℝ) (doubling_periods: ℕ) : kitte_length_when_found current_length doubling_periods = 4 :=
sorry

lemma initial_length_example : kitten_initial_length 16 2 := 
by sorry

end kitten_initial_length_l595_595188


namespace min_max_value_of_M_l595_595858

theorem min_max_value_of_M (x y : ℝ) :
  let E1 := x^2 + x * y + y^2,
      E2 := x^2 + x * (y - 1) + (y - 1)^2,
      E3 := (x - 1)^2 + (x - 1) * y + y^2,
      E4 := (x - 1)^2 + (x - 1) * (y - 1) + (y - 1)^2
  in max (max E1 E2) (max E3 E4) = (3 / 4) := 
sorry

end min_max_value_of_M_l595_595858


namespace nathan_subtracts_79_l595_595869

theorem nathan_subtracts_79 (a b : ℤ) (h₁ : a = 40) (h₂ : b = 1) :
  (a - b) ^ 2 = a ^ 2 - 79 := 
by
  sorry

end nathan_subtracts_79_l595_595869


namespace distinct_prime_factors_of_sigma_450_l595_595769

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595769


namespace sin_30_eq_half_l595_595479

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595479


namespace max_lg_sum_lemma_l595_595035

noncomputable def max_lg_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 4 * y = 40) : ℝ := 
  lg (x * y)

theorem max_lg_sum_lemma : 
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (x + 4 * y = 40) → max_lg_sum x y = 8 * lg 2 :=
by
  intros x y h1 h2 h3
  sorry

end max_lg_sum_lemma_l595_595035


namespace sin_30_eq_half_l595_595483

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595483


namespace initial_solution_weight_100kg_l595_595140

theorem initial_solution_weight_100kg
  (W : ℝ)
  (initial_salt_percentage : ℝ)
  (added_salt : ℝ)
  (final_salt_percentage : ℝ)
  (H1 : initial_salt_percentage = 0.10)
  (H2 : added_salt = 12.5)
  (H3 : final_salt_percentage = 0.20)
  (H4 : 0.20 * (W + 12.5) = 0.10 * W + 12.5) :
  W = 100 :=   
by 
  sorry

end initial_solution_weight_100kg_l595_595140


namespace jacob_nathan_total_letters_l595_595829

/-- Jacob and Nathan's combined writing output in 10 hours. -/
theorem jacob_nathan_total_letters (jacob_speed nathan_speed : ℕ) (h1 : jacob_speed = 2 * nathan_speed) (h2 : nathan_speed = 25) : jacob_speed + nathan_speed = 75 → (jacob_speed + nathan_speed) * 10 = 750 :=
by
  intros h3
  rw [h1, h2] at h3
  simp at h3
  rw [h3]
  norm_num

end jacob_nathan_total_letters_l595_595829


namespace sin_30_eq_half_l595_595464

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595464


namespace sin_30_eq_half_l595_595366

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595366


namespace unoccupied_seats_in_business_class_l595_595164

def airplane_seating (fc bc ec : ℕ) (pfc pbc pec : ℕ) : Nat :=
  let num_ec := ec / 2
  let num_bc := num_ec - pfc
  bc - num_bc

theorem unoccupied_seats_in_business_class :
  airplane_seating 10 30 50 3 (50/2) (50/2) = 8 := by
    sorry

end unoccupied_seats_in_business_class_l595_595164


namespace sin_30_eq_half_l595_595455

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595455


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595755

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595755


namespace trig_identity_l595_595097

theorem trig_identity : sin (π / 12) - (sqrt 3) * cos (π / 12) = -sqrt 2 :=
by
  sorry

end trig_identity_l595_595097


namespace sin_30_eq_half_l595_595356

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595356


namespace average_price_of_dvds_l595_595508

theorem average_price_of_dvds :
  let num_dvds_box1 := 10
  let price_per_dvd_box1 := 2.00
  let num_dvds_box2 := 5
  let price_per_dvd_box2 := 5.00
  let total_cost_box1 := num_dvds_box1 * price_per_dvd_box1
  let total_cost_box2 := num_dvds_box2 * price_per_dvd_box2
  let total_dvds := num_dvds_box1 + num_dvds_box2
  let total_cost := total_cost_box1 + total_cost_box2
  (total_cost / total_dvds) = 3.00 := 
sorry

end average_price_of_dvds_l595_595508


namespace curve_C_rect_eq_EA_plus_EB_inv_l595_595595

-- Define the conditions and prove the equivalent statement

def polar_eq_C (θ : ℝ) : ℝ := 2 * (Real.cos θ + Real.sin θ)

def rect_eq_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

def parametric_eq_l (t : ℝ) : ℝ × ℝ := (1/2 * t, 1 + (Real.sqrt 3)/2 * t)

theorem curve_C_rect_eq :
  ∀ (θ x y : ℝ), polar_eq_C θ = Real.sqrt (x^2 + y^2) →
  θ = Real.acos (x / Real.sqrt (x^2 + y^2)) → rect_eq_C x y :=
by
  intros θ x y h₁ h₂
  sorry

-- Definitions for intersection points A and B
def E : (ℝ × ℝ) := (0, 1)

def A : ℝ × ℝ := parametric_eq_l (-1)
def B : ℝ × ℝ := parametric_eq_l 1

def EA : ℝ := Real.abs (E.2 - A.2)
def EB : ℝ := Real.abs (E.2 - B.2)

theorem EA_plus_EB_inv :
  1/EA + 1/EB = Real.sqrt 3 :=
by
  sorry

end curve_C_rect_eq_EA_plus_EB_inv_l595_595595


namespace domain_of_sqrt_l595_595503

theorem domain_of_sqrt (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end domain_of_sqrt_l595_595503


namespace sarah_probability_reach_5_return_0_l595_595880

/-
Sarah starts at 0 on the real number line and tosses a fair coin 10 times.
When she gets heads, she moves 1 unit in the positive direction; when she gets tails, she moves 1 unit in the negative direction.
Prove that the probability that she reaches exactly 5 at some point during her tosses and returns to 0 by the end of her 10 tosses is 63/256.
-/
theorem sarah_probability_reach_5_return_0 :
  let probability : ℚ := (63 : ℚ) / 256
  in
  probability = (63 : ℚ) / 256 :=
by
  -- Proof to be filled in here.
  sorry

end sarah_probability_reach_5_return_0_l595_595880


namespace sin_thirty_degrees_l595_595372

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595372


namespace F_inequality_prime_counting_bound_l595_595849

-- Defining the problem and conditions
variable {r : ℕ}
variable {p : Fin r → ℕ}
-- Assume each p_i is prime
axiom primes : ∀ i : Fin r, Nat.Prime (p i)
variable {F : ℕ → ℕ}
-- F(n) is the number of integers less than n whose prime divisors are from (p_i)
axiom F_def : ∀ n : ℕ, F n = {m | m < n ∧ (∀ d, Nat.Prime d → d | m → ∃ i : Fin r, d = p i)}.toFinset.card

-- The main inequality to prove
theorem F_inequality (n : ℕ) : F n ≤ 2^r * Nat.sqrt n :=
sorry

-- Definition of prime counting function π(x)
def π (x : ℝ) : ℝ := {p | Nat.Prime p ∧ (p : ℝ) ≤ x}.toFinset.card

-- Proving the existence of constant c > 0
theorem prime_counting_bound (c : ℝ) (hc : c > 0) : 
  (∀ x : ℝ, 0 < x → π x ≥ c * Real.log x) :=
sorry

end F_inequality_prime_counting_bound_l595_595849


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595759

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595759


namespace sum_first_three_terms_arithmetic_sequence_l595_595897

theorem sum_first_three_terms_arithmetic_sequence
  (a₅ a₆ a₇ : ℤ)
  (h₅: a₅ = 7)
  (h₆: a₆ = 12)
  (h₇: a₇ = 17)
  (h : ∀ n : ℤ, (n = 6 → a₆ = a₅ + (a₅ - a₆)) ∧  (n = 7 → a₇ = a₆ + (a₆ - a₅))) :
  (∃ (a₁ a₂ a₃: ℤ), a₁ + a₂ + a₃ = -24) :=
begin
  sorry
end

end sum_first_three_terms_arithmetic_sequence_l595_595897


namespace find_a_l595_595546

variable (f : ℝ → ℝ)

noncomputable def satisfies_condition (a : ℝ) : Prop :=
  ∀ x y, f (x - f y) = f x + a * ⌊y⟩

theorem find_a (a : ℝ) :
  (∃ f : ℝ → ℝ, satisfies_condition f a) → (∃ n : ℤ, a = -n^2) :=
by
  sorry

end find_a_l595_595546


namespace quadratic_sum_of_coefficients_l595_595999

theorem quadratic_sum_of_coefficients (x : ℝ) : 
  let a := 1
  let b := 1
  let c := -4
  a + b + c = -2 :=
by
  sorry

end quadratic_sum_of_coefficients_l595_595999


namespace sin_30_eq_half_l595_595249

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595249


namespace ratio_of_triangle_areas_l595_595003

theorem ratio_of_triangle_areas
    (AB CD : ℝ)
    (β : ℝ)
    (E : Type)
    (intersect : E)
    (diameter_condition : AB = 2)
    (parallel_condition : CD = CD)
    (angle_condition : ∀ [plane_geometry], angle AED = 90 - β) :
    (area_of_triangle CDE) / (area_of_triangle ABE) = sin(β)^2 :=
by
  sorry

end ratio_of_triangle_areas_l595_595003


namespace sqrt_number_is_169_l595_595094

theorem sqrt_number_is_169 (a b : ℝ) 
  (h : a^2 + b^2 + (4 * a - 6 * b + 13) = 0) : 
  (a^2 + b^2)^2 = 169 :=
sorry

end sqrt_number_is_169_l595_595094


namespace Morio_age_when_Michiko_was_born_l595_595067

theorem Morio_age_when_Michiko_was_born (Teresa_age_now : ℕ) (Teresa_age_when_Michiko_born : ℕ) (Morio_age_now : ℕ)
  (hTeresa : Teresa_age_now = 59) (hTeresa_born : Teresa_age_when_Michiko_born = 26) (hMorio : Morio_age_now = 71) :
  Morio_age_now - (Teresa_age_now - Teresa_age_when_Michiko_born) = 38 :=
by
  sorry

end Morio_age_when_Michiko_was_born_l595_595067


namespace c_put_15_oxen_l595_595176

theorem c_put_15_oxen (x : ℕ):
  (10 * 7 + 12 * 5 + 3 * x = 130 + 3 * x) →
  (175 * 3 * x / (130 + 3 * x) = 45) →
  x = 15 :=
by
  intros h1 h2
  sorry

end c_put_15_oxen_l595_595176


namespace sin_thirty_deg_l595_595231

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595231


namespace correct_answer_l595_595192

/-- Let N be a positive integer such that 300 ≤ N ≤ 600. 
  Let N_4 and N_7 be the base-4 and base-7 representations of N, respectively. 
  Define S to be the sum of the base-10 interpretations of N_4 and N_7. 
  Consider the number of integers N for which the two rightmost digits of S 
  are the same as those of 3N (i.e., S % 100 = (3*N) % 100). 
-/
def problem_condition (N : ℕ) : Prop :=
  300 ≤ N ∧ N ≤ 600

def base_4_to_10 (N : ℕ) : ℕ :=
  let a1 := (N / 16) % 4
  let a2 := (N / 4) % 4
  let a3 := N % 4
  16 * a1 + 4 * a2 + a3

def base_7_to_10 (N : ℕ) : ℕ :=
  let b1 := (N / 49) % 7
  let b2 := (N / 7) % 7
  let b3 := N % 7
  49 * b1 + 7 * b2 + b3

def compute_S (N : ℕ) : ℕ :=
  (base_4_to_10 N) + (base_7_to_10 N)

def check_rightmost_digits (N : ℕ) : Prop :=
  (compute_S N) % 100 = (3 * N) % 100

def num_valid_N : nat :=
  (finset.Icc 300 600).filter (λ N, check_rightmost_digits N).card

theorem correct_answer : num_valid_N = 15 := sorry

end correct_answer_l595_595192


namespace sin_thirty_degree_l595_595281

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595281


namespace sin_thirty_deg_l595_595221

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595221


namespace max_cubes_line_pass_through_l595_595928

theorem max_cubes_line_pass_through (n : ℕ) (h : 2 ≤ n) : 
  ∃ k : ℕ, (∀ line : ℝ → ℝ × ℝ × ℝ, ∃ unit_cubes : set (ℝ × ℝ × ℝ), line_passing_through_unit_cubes line unit_cubes ∧ size unit_cubes ≤ k) ∧ k = 3 * n - 2 := 
sorry

end max_cubes_line_pass_through_l595_595928


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595708

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595708


namespace find_A_l595_595994

noncomputable def telephone_number_satisfies_conditions (A B C D E F G H I J : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J ∧
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  E = D - 2 ∧ F = D - 4 ∧ -- Given D, E, F are consecutive even digits
  H = G - 2 ∧ I = G - 4 ∧ J = G - 6 ∧ -- Given G, H, I, J are consecutive odd digits
  A + B + C = 9

theorem find_A :
  ∃ (A B C D E F G H I J : ℕ), telephone_number_satisfies_conditions A B C D E F G H I J ∧ A = 8 :=
by {
  sorry
}

end find_A_l595_595994


namespace sin_30_eq_half_l595_595471

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595471


namespace dow_original_value_l595_595071

-- Given conditions
def Dow_end := 8722
def percentage_fall := 0.02
def final_percentage := 1 - percentage_fall -- 98% of the original value

-- To prove: the original value of the Dow Jones Industrial Average equals 8900
theorem dow_original_value :
  (Dow_end: ℝ) = (final_percentage * 8900) := 
by sorry

end dow_original_value_l595_595071


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595721

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595721


namespace largest_divisible_sequence_l595_595099

def is_divisible (a b : ℕ) : Prop :=
  a % b = 0 ∨ b % a = 0

theorem largest_divisible_sequence :
  ∃ (seq : List ℕ), seq.nodup ∧
    seq ⊆ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    (∀ i, i < seq.length - 1 → is_divisible (seq[i]) (seq[i + 1])) ∧
    seq.length = 8 :=
  sorry

end largest_divisible_sequence_l595_595099


namespace sin_thirty_degrees_l595_595377

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595377


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595765

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595765


namespace sin_30_eq_half_l595_595429

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595429


namespace sin_of_30_degrees_l595_595256

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595256


namespace find_BP_l595_595132

theorem find_BP
  (A B C D P : Type)
  (on_circle : ∀ x ∈ {A, B, C, D}, x ∈ Circle)
  (intersects : ∀ A C B D, segments_intersect AC BD P)
  (AP : length (segment A P) = 6)
  (PC : length (segment P C) = 2)
  (BD : length (segment B D) = 7)
  (BP_lt_DP : ∀ BP DP, BP < DP) :
  ∃ BP, BP = 3 :=
begin
  sorry
end

end find_BP_l595_595132


namespace values_of_a_and_b_minimum_value_of_g_g_minimum_at_zero_l595_595613

section
variable (a b x : ℝ)

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.exp x - x^2 + a

-- Derivative of f(x)
def f' (x : ℝ) : ℝ := Real.exp x - 2 * x

-- Define the tangent line at x = 0
def tangent_line (x : ℝ) : ℝ := b * x

-- Condition for tangent line at x = 0
def tangent_condition (x : ℝ) : Prop := f(0) = 0 ∧ f'(0) = b

-- Calculate values of a and b
theorem values_of_a_and_b : a = -1 ∧ b = 1 :=
by
  sorry

-- Define the function g(x)
def g (x : ℝ) : ℝ := f(x) + x^2 - x

-- Derivative of g(x)
def g' (x : ℝ) : ℝ := Real.exp x - 1

-- Prove minimum value of g(x)
theorem minimum_value_of_g : ∀ x, g(x) ≥ 0 :=
by
  sorry

-- Prove the minimum value of g(x) is 0
theorem g_minimum_at_zero : g(0) = 0 :=
by
  sorry

end

end values_of_a_and_b_minimum_value_of_g_g_minimum_at_zero_l595_595613


namespace sin_30_eq_one_half_l595_595315

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595315


namespace car_distance_in_30_minutes_l595_595783

theorem car_distance_in_30_minutes 
  (train_speed : ℝ) 
  (car_speed_fraction : ℝ) 
  (time_minutes : ℝ)
  (train_speed_value : train_speed = 90)
  (car_speed_fraction_value : car_speed_fraction = 2/3)
  (time_minutes_value : time_minutes = 30) :
  let car_speed := car_speed_fraction * train_speed
  let car_distance := car_speed * (time_minutes / 60)
  car_distance = 30 := 
by
  intros train_speed car_speed_fraction time_minutes
  intros train_speed_value car_speed_fraction_value time_minutes_value
  simp only [train_speed_value, car_speed_fraction_value, time_minutes_value]
  -- here we would perform the actual calculations
  sorry

end car_distance_in_30_minutes_l595_595783


namespace floor_plus_x_eq_17_over_4_l595_595523

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l595_595523


namespace year_2022_form_l595_595822

theorem year_2022_form :
  ∃ (a b c d e f g h i j : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10 ∧
    2001 ≤ (a + b * c * d * e) / (f + g * h * i * j) ∧ (a + b * c * d * e) / (f + g * h * i * j) ≤ 2100 ∧
    (a + b * c * d * e) / (f + g * h * i * j) = 2022 :=
sorry

end year_2022_form_l595_595822


namespace max_sand_weight_l595_595147

-- Define the problem conditions
variables (cart1_capacity : ℕ) (cart2_capacity : ℕ) (sack_weights : list ℝ)

-- Hypotheses as given conditions
def conditions : Prop :=
  cart1_capacity = 8 ∧
  cart2_capacity = 9 ∧
  (∀ w ∈ sack_weights, w ≤ 1) ∧ 
  (list.sum sack_weights > 17)

-- Statement of the theorem proving the maximum weight
theorem max_sand_weight (h : conditions cart1_capacity cart2_capacity sack_weights) :
  ∃ max_weight : ℝ, max_weight = 15.3 ∧ 
  ∀ (partition : (list ℝ × list ℝ)), 
    partition.1 ++ partition.2 = sack_weights →
    list.sum partition.1 ≤ cart1_capacity →
    list.sum partition.2 ≤ cart2_capacity →
    list.sum partition.1 + list.sum partition.2 ≤ max_weight :=
sorry

end max_sand_weight_l595_595147


namespace count_sequences_of_length_15_l595_595785

def countingValidSequences (n : ℕ) : ℕ := sorry

theorem count_sequences_of_length_15 :
  countingValidSequences 15 = 266 :=
  sorry

end count_sequences_of_length_15_l595_595785


namespace area_circle_minus_square_l595_595171

theorem area_circle_minus_square {r : ℝ} (h : r = 1/2) : 
  (π * r^2) - (1^2) = (π / 4) - 1 :=
by
  rw [h]
  sorry

end area_circle_minus_square_l595_595171


namespace range_of_m_l595_595576

variable (m : ℝ)

/-- Proposition p: For any x in ℝ, x^2 + 1 > m -/
def p := ∀ x : ℝ, x^2 + 1 > m

/-- Proposition q: The linear function f(x) = (2 - m) * x + 1 is an increasing function -/
def q := (2 - m) > 0

theorem range_of_m (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 1 < m ∧ m < 2 := 
sorry

end range_of_m_l595_595576


namespace halfway_point_distance_l595_595125

-- Definitions from conditions
def distance_house_to_hospital_km := 1
def distance_house_to_hospital_m := 700
def distance_hospital_to_school_m := 900
def km_to_m (km : Int) : Int := km * 1000

-- Converting kilometers to meters
def distance_house_to_hospital_m_converted : Int := km_to_m(distance_house_to_hospital_km) + distance_house_to_hospital_m

-- Total distance
def total_distance_m : Int := distance_house_to_hospital_m_converted + distance_hospital_to_school_m

-- The statement to prove
theorem halfway_point_distance :
  total_distance_m / 2 = 1300 := by
  -- The proof is not needed as per the task requirements.
  sorry

end halfway_point_distance_l595_595125


namespace minimal_people_next_to_seated_l595_595143

theorem minimal_people_next_to_seated (N : ℕ) 
  (circular_table : ℕ -> Prop) 
  (seated_people_condition : (p : ℕ) -> p < 80 -> N -> Prop)
  (next_person_condition : (p : ℕ) -> p < 80 -> Prop) :
  ( ∀ p, 0 ≤ p < 80 → circular_table p) →
  ( ∀ p, 0 ≤ p < 80 → next_person_condition p → 
         ∃ i, i < N ∧ seated_people_condition i N) →
  N = 20 :=
by
  sorry

end minimal_people_next_to_seated_l595_595143


namespace jamies_shoes_cost_l595_595922

-- Define the costs of items and the total cost.
def cost_total : ℤ := 110
def cost_coat : ℤ := 40
def cost_one_pair_jeans : ℤ := 20

-- Define the number of pairs of jeans.
def num_pairs_jeans : ℕ := 2

-- Define the cost of Jamie's shoes (to be proved).
def cost_jamies_shoes : ℤ := cost_total - (cost_coat + num_pairs_jeans * cost_one_pair_jeans)

theorem jamies_shoes_cost : cost_jamies_shoes = 30 :=
by
  -- Insert proof here
  sorry

end jamies_shoes_cost_l595_595922


namespace sin_30_deg_l595_595344

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595344


namespace green_balls_count_l595_595141

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def yellow_balls : ℕ := 2
def red_balls : ℕ := 15
def purple_balls : ℕ := 3
def probability_neither_red_nor_purple : ℝ := 0.7

theorem green_balls_count (G : ℕ) :
  (white_balls + G + yellow_balls) / total_balls = probability_neither_red_nor_purple →
  G = 18 := 
by
  sorry

end green_balls_count_l595_595141


namespace problem_I_problem_II_l595_595608

namespace MathProof

-- Define the function f(x) given m
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2 * |x + 1|

-- Problem (I)
theorem problem_I (x : ℝ) : (5 - |x - 1| - 2 * |x + 1| > 2) ↔ (-4/3 < x ∧ x < 0) := 
sorry

-- Define the quadratic function
def y (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Problem (II)
theorem problem_II (m : ℝ) : (∀ x : ℝ, ∃ t : ℝ, t = x^2 + 2*x + 3 ∧ t = f m x) ↔ (m ≥ 4) :=
sorry

end MathProof

end problem_I_problem_II_l595_595608


namespace sin_30_eq_half_l595_595292
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595292


namespace sin_30_eq_half_l595_595211

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595211


namespace probability_of_tie_l595_595807

open ProbabilityTheory

-- Define the number of supporters
def num_supporters (candidate : Type) := 5

-- Define the probability for voting and not voting
def voting_probability : ℚ := 1 / 2
def non_voting_probability : ℚ := 1 / 2

-- Define random variables X_A and X_B for the number of votes for candidates A and B, respectively
noncomputable def X_A : ℕ → ℚ := λ k, (nat.choose 5 k : ℕ) * (voting_probability ^ k) * (non_voting_probability ^ (5 - k))
noncomputable def X_B : ℕ → ℚ := λ k, (nat.choose 5 k : ℕ) * (voting_probability ^ k) * (non_voting_probability ^ (5 - k))

-- Define the probability of a tie
noncomputable def prob_of_tie : ℚ :=
  ∑ k in finset.range 6, (X_A k) * (X_B k)
  
theorem probability_of_tie :
  prob_of_tie = 63 / 256 := sorry

end probability_of_tie_l595_595807


namespace sum_of_divisors_prime_factors_450_l595_595734

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595734


namespace parabola_polygon_focus_sum_l595_595026

theorem parabola_polygon_focus_sum
  (n : ℕ) (p : ℝ) (h_n : n ≥ 3)
  (F : ℝ × ℝ := (p / 2, 0))
  (polygon : fin n → ℝ × ℝ)
  (h_focus : ∀ i, ((polygon i).fst, (polygon i).snd) ≠ F)
  (h_no_x_axis : ∀ i, (polygon i).snd ≠ 0)
  (B : fin n → ℝ × ℝ)
  (h_parabola : ∀ i, (B i).snd^2 = 2 * p * (B i).fst)
  (h_intersects : ∀ i, ∃ k, B i = (k * polygon i).fst + F.fst, (k * polygon i).snd + F.snd)
  : (finset.univ.sum (λ i, ((F.1 - (B i).1)^2 + (F.2 - (B i).2)^2)^0.5)) > n * p := sorry

end parabola_polygon_focus_sum_l595_595026


namespace tournament_duration_in_hours_l595_595008

-- Define the conditions
def number_of_amateurs : ℕ := 5
def game_duration_minutes : ℕ := 45
def break_duration_minutes : ℕ := 15
def number_of_games (n : ℕ) : ℕ := (n * (n - 1)) / 2
def total_game_time_minutes (n_games : ℕ) (game_duration : ℕ) : ℕ := n_games * game_duration
def total_break_time_minutes (n_games : ℕ) (break_duration : ℕ) : ℕ := (n_games - 1) * break_duration

-- Proving the question equals the answer
theorem tournament_duration_in_hours :
  let n_games := number_of_games number_of_amateurs
  let total_game_time := total_game_time_minutes n_games game_duration_minutes
  let total_break_time := total_break_time_minutes n_games break_duration_minutes
  let total_time_minutes := total_game_time + total_break_time
  let total_time_hours := (total_time_minutes : ℚ) / 60 in
  total_time_hours = 9.75 :=
by
  sorry

end tournament_duration_in_hours_l595_595008


namespace sin_30_eq_half_l595_595453

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595453


namespace find_x_l595_595588

variable (x : ℝ)
variable (l : ℝ) (w : ℝ)

def length := 4 * x + 1
def width := x + 7

theorem find_x (h1 : l = length x) (h2 : w = width x) (h3 : l * w = 2 * (2 * l + 2 * w)) :
  x = (-9 + Real.sqrt 481) / 8 :=
by
  subst_vars
  sorry

end find_x_l595_595588


namespace sin_30_eq_half_l595_595245

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595245


namespace sin_30_eq_half_l595_595287
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595287


namespace sin_30_deg_l595_595349

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595349


namespace ivan_apples_leftover_l595_595824

theorem ivan_apples_leftover (initial_apples : ℕ) (mini_pies : ℕ) (apples_per_mini_pie : ℚ) 
  (used_apples : ℕ) (leftover_apples : ℕ) :
  initial_apples = 48 →
  mini_pies = 24 →
  apples_per_mini_pie = 1/2 →
  used_apples = (mini_pies * apples_per_mini_pie).toNat →
  leftover_apples = initial_apples - used_apples →
  leftover_apples = 36 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  simp at h4
  rw h4 at h5
  exact h5

end ivan_apples_leftover_l595_595824


namespace sin_30_eq_half_l595_595422

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595422


namespace probability_allison_greater_l595_595180

open Probability

-- Definitions for the problem
noncomputable def die_A : ℕ := 4

noncomputable def die_B : PMF ℕ :=
  PMF.uniform_of_fin (fin 6) -- Could also explicitly write as PMF.of_list [(1, 1/6), ..., (6, 1/6)]

noncomputable def die_N : PMF ℕ :=
  PMF.of_list [(3, 3/6), (5, 3/6)]

-- The target event: Allison's roll > Brian's roll and Noah's roll
noncomputable def event (roll_A : ℕ) (roll_B roll_N : PMF ℕ) :=
  ∀ b n, b ∈ [1, 2, 3] → n ∈ [3] → roll_A > b ∧ roll_A > n

-- The probability calculation
theorem probability_allison_greater :
  (∑' (b : ℕ) (h_b : b < 4) (n : ℕ) (h_n : n = 3), die_B b * die_N n)
  = 1 / 4 :=
by
  -- assumed rolls
  let roll_A := 4
  let prob_B := 1 / 2
  let prob_N := 1 / 2
  
  -- skip proof, but assert the correct result.
  sorry

end probability_allison_greater_l595_595180


namespace sin_30_eq_half_l595_595459

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595459


namespace arithmetic_square_root_16_l595_595073

theorem arithmetic_square_root_16 : ∀ x : ℝ, x ≥ 0 → x^2 = 16 → x = 4 :=
by
  intro x hx h
  sorry

end arithmetic_square_root_16_l595_595073


namespace number_of_soccer_balls_in_first_set_l595_595959

noncomputable def cost_of_soccer_ball : ℕ := 50
noncomputable def first_cost_condition (F c : ℕ) : Prop := 3 * F + c = 155
noncomputable def second_cost_condition (F : ℕ) : Prop := 2 * F + 3 * cost_of_soccer_ball = 220

theorem number_of_soccer_balls_in_first_set (F : ℕ) :
  (first_cost_condition F 50) ∧ (second_cost_condition F) → 1 = 1 :=
by
  sorry

end number_of_soccer_balls_in_first_set_l595_595959


namespace sin_thirty_degrees_l595_595385

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595385


namespace number_of_pictures_l595_595124

theorem number_of_pictures (x : ℕ) (h : x - (x / 2 - 1) = 25) : x = 48 :=
sorry

end number_of_pictures_l595_595124


namespace log_expression_eq_three_trig_expression_eq_combined_value_l595_595553

theorem log_expression_eq_three :
  log 5^2 + (2/3) * log 8 + log 5 * log 20 + log 2 ^ 2 = 3 := 
sorry

theorem trig_expression_eq_combined_value :
  cos (17 * pi / 4) + sin (13 * pi / 3) + tan (25 * pi / 6) = (3 * real.sqrt(2) + 5 * real.sqrt(3)) / 6 := 
sorry

end log_expression_eq_three_trig_expression_eq_combined_value_l595_595553


namespace Dr_Fu_Manchu_bank_account_interest_rate_l595_595507

theorem Dr_Fu_Manchu_bank_account_interest_rate :
  let annual_rate : ℝ := 0.08,
      tax_rate : ℝ := 0.15,
      quarters : ℝ := 4,
      quarterly_rate := annual_rate / quarters,
      effective_quarterly_rate := quarterly_rate * (1 - tax_rate),
      effective_annual_rate := (1 + effective_quarterly_rate)^quarters - 1 in
  (effective_annual_rate * 100).round = 6.99 := by
  sorry

end Dr_Fu_Manchu_bank_account_interest_rate_l595_595507


namespace sin_30_eq_half_l595_595441

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595441


namespace weight_of_brand_a_correct_l595_595098

noncomputable def weight_a_of_brand_a : ℝ :=
  sorry

theorem weight_of_brand_a_correct:
  (∃ W_a : ℝ, weight_a_of_brand_a = W_a ∧ 
    (let mixture_weight := 2460 in 
     let brand_b_weight := 850 in
     let W_a_grams := 
       ((3/5) * 3 * W_a + (2/5) * 2 * brand_b_weight) in
    W_a_grams = mixture_weight)) → weight_a_of_brand_a = 988.888888888889 :=
by
  sorry

end weight_of_brand_a_correct_l595_595098


namespace problem1_problem2_l595_595955

-- First problem: Calculation
theorem problem1 :
  |real.sqrt 3 - 1| - 4 * real.sin (real.pi / 6) + (1/2)^(-1 : ℤ) + (4 - real.pi)^0 = real.sqrt 3 :=
by 
  sorry

-- Second problem: Factorization
theorem problem2 (a : ℝ) :
  2 * a^3 - 12 * a^2 + 18 * a = 2 * a * (a - 3)^2 :=
by 
  sorry

end problem1_problem2_l595_595955


namespace sin_30_eq_half_l595_595487

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595487


namespace sin_30_eq_half_l595_595293
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595293


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595756

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595756


namespace intersection_of_sets_eq_l595_595624

noncomputable def set_intersection (M N : Set ℝ): Set ℝ :=
  {x | x ∈ M ∧ x ∈ N}

theorem intersection_of_sets_eq :
  let M := {x : ℝ | -2 < x ∧ x < 2}
  let N := {x : ℝ | x^2 - 2 * x - 3 < 0}
  set_intersection M N = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end intersection_of_sets_eq_l595_595624


namespace sin_30_eq_half_l595_595465

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595465


namespace find_c_l595_595900

-- Define the two points as given in the problem
def pointA : ℝ × ℝ := (-6, 1)
def pointB : ℝ × ℝ := (-3, 4)

-- Define the direction vector as subtraction of the two points
def directionVector : ℝ × ℝ := (pointB.1 - pointA.1, pointB.2 - pointA.2)

-- Define the target direction vector format with unknown c
def targetDirectionVector (c : ℝ) : ℝ × ℝ := (3, c)

-- The theorem stating that c must be 3
theorem find_c : ∃ c : ℝ, directionVector = targetDirectionVector c ∧ c = 3 := 
by
  -- Prove the statement or show it is derivable
  sorry

end find_c_l595_595900


namespace determine_n_l595_595791

theorem determine_n (m n : ℝ)
  (h : (∀ x : ℝ, (x + 3) * (x + m) = x^2 + n * x + 12)) :
  n = 7 := 
begin
  sorry
end

end determine_n_l595_595791


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595704

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595704


namespace sin_thirty_degree_l595_595271

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595271


namespace negation_equiv_l595_595620

variable (p : Prop) [Nonempty ℝ]

def proposition := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

def negation_of_proposition : Prop := ∀ x : ℝ, Real.exp x - x - 1 > 0

theorem negation_equiv
  (h : proposition = p) : (¬ proposition) = negation_of_proposition := by
  sorry

end negation_equiv_l595_595620


namespace sin_30_eq_half_l595_595215

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595215


namespace centroid_area_ratio_l595_595794

theorem centroid_area_ratio (A B C M : Type) [InnerProductSpace ℝ M] (area : M → M → M → ℝ) 
    (h : area A B M + area B C M + area C A M = 0) :
    (area A B M) / (area A B C) = 1 / 3 :=
sorry

end centroid_area_ratio_l595_595794


namespace sin_30_eq_half_l595_595486

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595486


namespace sum_g_7_l595_595847

noncomputable def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

noncomputable def g (y : ℝ) : ℝ := 3 * y - 1

theorem sum_g_7 : 
  let x₁ := (5 + Real.sqrt 29) / 2,
      x₂ := (5 - Real.sqrt 29) / 2,
      g₁ := 3 * x₁ - 1,
      g₂ := 3 * x₂ - 1
  in g₁ + g₂ = 13 :=
begin
  sorry
end

end sum_g_7_l595_595847


namespace distance_B_G_l595_595152

-- Define the variables and constants based on the conditions
variables {A B C D E F G H I J K : Type} [metric_space A] [metric_space B]
  [metric_space C] [metric_space D] [metric_space E] [metric_space F]
  [metric_space G] [metric_space H] [metric_space I] [metric_space J] [metric_space K]

-- Define relevant distances
noncomputable def distance_AK : ℝ := 56
noncomputable def max_distance : ℝ := 12
noncomputable def min_distance : ℝ := 17

-- Hypotheses based on the conditions
axiom dist_AC_le_12 : dist A C ≤ max_distance
axiom dist_BD_le_12 : dist B D ≤ max_distance
axiom dist_CE_le_12 : dist C E ≤ max_distance
axiom dist_IK_le_12 : dist I K ≤ max_distance
axiom dist_AD_ge_17 : dist A D ≥ min_distance
axiom dist_BE_ge_17 : dist B E ≥ min_distance
axiom dist_CF_ge_17 : dist C F ≥ min_distance
axiom dist_HK_ge_17 : dist H K ≥ min_distance

-- Goal: Prove the distance between B and G is 29 units
theorem distance_B_G : dist B G = 29 :=
by
  sorry

end distance_B_G_l595_595152


namespace cory_fruit_orders_l595_595496

theorem cory_fruit_orders (n_apples n_oranges n_bananas : ℕ)
    (h_apples : n_apples = 4)
    (h_oranges : n_oranges = 3)
    (h_bananas : n_bananas = 2) :
    let total_fruits := n_apples + n_oranges + n_bananas in
    let factorial := Nat.factorial in
    total_fruits = 9 → 
    (factorial total_fruits) / (factorial n_apples * factorial n_oranges * factorial n_bananas) = 1260 := by
  intros h_total
  sorry

end cory_fruit_orders_l595_595496


namespace sin_of_30_degrees_l595_595260

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595260


namespace max_value_g_l595_595905

theorem max_value_g (a b : ℝ) (h1 : ∀ x, a * real.sin x + b ≤ 1) (h2 : ∀ x, a * real.sin x + b ≥ -7) :
  ∃ (m : ℝ), m = -3 ∨ m = 4 ∧ ∀ x, b * (real.sin x)^2 - a * (real.cos x)^2 ≤ m :=
sorry

end max_value_g_l595_595905


namespace minimum_value_of_2x_3y_l595_595596

noncomputable def minimum_value (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : (2/x) + (3/y) = 1) : ℝ :=
  2*x + 3*y

theorem minimum_value_of_2x_3y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : (2/x) + (3/y) = 1) : minimum_value x y hx hy hxy = 25 :=
sorry

end minimum_value_of_2x_3y_l595_595596


namespace last_non_zero_digit_of_30_factorial_l595_595550

theorem last_non_zero_digit_of_30_factorial : 
  ∃ d: ℕ, d < 10 ∧ d ≠ 0 ∧ (30.factorial % 10^8 % 10 = d) := 
begin
  use 8,
  split,
  { exact nat.lt.base (10-1), },
  split,
  { norm_num, },
  { sorry }  -- Proof omitted
end

end last_non_zero_digit_of_30_factorial_l595_595550


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595757

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595757


namespace exists_diameter_not_intersecting_broken_line_l595_595924

-- Definitions for the problem
variables {P : Type*} [metric_space P] [is_connected_space P]
variables {O : P} {A B : P} {r : ℝ} (hcircle : dist O A = r ∧ dist O B = r)

-- The given condition for the broken line being shorter than the diameter
variables {n : ℕ} (X : fin n.succ → P) (hX0 : X 0 = A) (hXn : X n = B)
(hbroken : (∑ i in finset.range n, dist (X i) (X (i + 1))) < 2 * r)

-- The theorem to prove there exists a diameter that does not intersect the broken line
theorem exists_diameter_not_intersecting_broken_line :
  ∃ d : set P, is_diameter O d ∧ (∀ i : fin n, ¬ (X i ∈ d ∧ X (i + 1) ∈ d)) :=
sorry

end exists_diameter_not_intersecting_broken_line_l595_595924


namespace greatest_difference_l595_595806

theorem greatest_difference (a b c d : ℤ) (h₁ : b ≠ -a) (h₂ : d ≠ -c) (h₃ : b + c ≠ 0) (h₄ : b + d ≠ 0) (h₅ : a + c ≠ 0) :
  ( (a * d - b * c : ℚ) / ((a + b) * (c + d)) )
  ≥ ( max 
        ((a * (a + b) - c * (c + d)) / ((c + d) * (a + b))) 
        (max 
          (a * (a + b) - c * (b + c)) / ((a + b) * (b + c)) 
          ((a * (b + d) - c * (a + c)) / ((b + d) * (a + c))))) := 
by
  sorry

end greatest_difference_l595_595806


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595638

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595638


namespace distinct_prime_factors_of_sigma_450_l595_595777

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595777


namespace range_of_c_l595_595577

variable (c : ℝ)

def p := 2 < 3 * c
def q := ∀ x : ℝ, 2 * x^2 + 4 * c * x + 1 > 0

theorem range_of_c (hp : p c) (hq : q c) : (2 / 3) < c ∧ c < (Real.sqrt 2 / 2) :=
by
  sorry

end range_of_c_l595_595577


namespace concurrency_of_MiSi_l595_595953

variables {A1 A2 A3 : Type}
variables {a1 a2 a3 : Line A1 A2 A3} -- sides opposite to vertices A1, A2, A3
variables (M1 M2 M3 : Midpoint A1 A2 A3 a1 a2 a3) -- midpoints of sides a1, a2, a3
variables (T1 T2 T3 : IncirclePoints A1 A2 A3 a1 a2 a3) -- tangent points to sides a1, a2, a3
variables (S1 S2 S3 : Reflections A1 A2 A3 T1 T2 T3) -- reflections of T1, T2, T3 over angle bisectors of ∠ A1, ∠ A2, ∠ A3.

theorem concurrency_of_MiSi
  (h1 : ¬ EquilateralTriangle A1 A2 A3) -- condition: non-equilateral triangle
  (h2 : a1.opposite A1) (h3 : a2.opposite A2) (h4 : a3.opposite A3) -- sides opposite to vertices
  (h5 : M1.midpoint a1) (h6 : M2.midpoint a2) (h7 : M3.midpoint a3) -- midpoints of sides
  (h8 : T1.tangentPointIncircle a1) (h9 : T2.tangentPointIncircle a2) (h10 : T3.tangentPointIncircle a3) -- tangent points
  (h11 : S1.reflectionOverBisector T1 (angle_bisector A1)) (h12 : S2.reflectionOverBisector T2 (angle_bisector A2)) (h13 : S3.reflectionOverBisector T3 (angle_bisector A3)) -- reflections of tangent points
  : Concurrent (M1.lineThrough S1) (M2.lineThrough S2) (M3.lineThrough S3) := sorry

end concurrency_of_MiSi_l595_595953


namespace sin_30_eq_half_l595_595391

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595391


namespace greatest_integer_solution_l595_595862

theorem greatest_integer_solution (x : ℝ) (h : x^3 = 7 - 2x) (h' : x - 2 < x) : x ≤ 3 :=
by
  sorry

end greatest_integer_solution_l595_595862


namespace no_consecutive_pairs_subset_count_l595_595135

def fib : ℕ → ℕ 
| 0 => 0 
| 1 => 1 
| n + 2 => fib (n + 1) + fib n

theorem no_consecutive_pairs_subset_count : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
  (card {T : set ℕ // T ⊆ S ∧ ∀ x ∈ T, ∀ y ∈ T, x ≠ y → abs (x - y) ≠ 2}) = 273 :=
by
  sorry

end no_consecutive_pairs_subset_count_l595_595135


namespace distinct_prime_factors_sum_divisors_450_l595_595750

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595750


namespace dot_product_expanded_dot_product_l595_595586

-- Definitions for the conditions
variables (a b : ℝ^3) (theta : ℝ)
hypothesis h1 : ∥a∥ = 4
hypothesis h2 : ∥b∥ = 2
hypothesis h3 : theta = 2 * Real.pi / 3  -- 120 degrees in radians

-- Definition for the angle condition
hypothesis angle_condition : Real.cos theta = -1 / 2

-- Question 1: a · b = -4
theorem dot_product (a b : ℝ^3) [h1 : ∥a∥ = 4] [h2 : ∥b∥ = 2] [h3 : theta = 2 * Real.pi / 3] [angle_condition : Real.cos theta = -1 / 2] :
  a ⬝ b = -4 := sorry

-- Question 2: (a - 2b) · (a + b) = 12
theorem expanded_dot_product (a b : ℝ^3) [h1 : ∥a∥ = 4] [h2 : ∥b∥ = 2] [h3 : theta = 2 * Real.pi / 3] [angle_condition : Real.cos theta = -1 / 2] :
  (a - 2 • b) ⬝ (a + b) = 12 := sorry

end dot_product_expanded_dot_product_l595_595586


namespace sin_of_30_degrees_l595_595257

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595257


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595641

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595641


namespace sin_30_is_half_l595_595329

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595329


namespace ivan_apples_leftover_l595_595823

theorem ivan_apples_leftover (initial_apples : ℕ) (mini_pies : ℕ) (apples_per_mini_pie : ℚ) 
  (used_apples : ℕ) (leftover_apples : ℕ) :
  initial_apples = 48 →
  mini_pies = 24 →
  apples_per_mini_pie = 1/2 →
  used_apples = (mini_pies * apples_per_mini_pie).toNat →
  leftover_apples = initial_apples - used_apples →
  leftover_apples = 36 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  simp at h4
  rw h4 at h5
  exact h5

end ivan_apples_leftover_l595_595823


namespace caitlins_team_number_l595_595921

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the two-digit prime numbers
def two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ is_prime n

-- Lean statement
theorem caitlins_team_number (h_date birthday_before today birthday_after : ℕ)
  (p₁ p₂ p₃ : ℕ)
  (h1 : two_digit_prime p₁)
  (h2 : two_digit_prime p₂)
  (h3 : two_digit_prime p₃)
  (h4 : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h5 : p₁ + p₂ = today ∨ p₁ + p₃ = today ∨ p₂ + p₃ = today)
  (h6 : (p₁ + p₂ = birthday_before ∨ p₁ + p₃ = birthday_before ∨ p₂ + p₃ = birthday_before)
       ∧ birthday_before < today)
  (h7 : (p₁ + p₂ = birthday_after ∨ p₁ + p₃ = birthday_after ∨ p₂ + p₃ = birthday_after)
       ∧ birthday_after > today) :
  p₃ = 11 := by
  sorry

end caitlins_team_number_l595_595921


namespace convert_13_to_binary_l595_595514

theorem convert_13_to_binary : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by sorry

end convert_13_to_binary_l595_595514


namespace non_negative_real_inequality_l595_595853

theorem non_negative_real_inequality
  {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
by
  sorry

end non_negative_real_inequality_l595_595853


namespace k_value_monotonicity_and_inequality_solution_l595_595611

-- Define the function f(x) with given conditions
def f (x : ℝ) (k : ℝ) (a : ℝ) := k * (a^x) - (a^(-x))

-- Assume a > 0 and a ≠ 1
variables (a : ℝ) (h_a1 : a > 0) (h_a2 : a ≠ 1)

-- State that f(x) is an odd function
axiom f_odd : ∀ x : ℝ, f x k a = -f (-x) k a

-- Prove the value of k
theorem k_value : ∃ k : ℝ, (f 0 k a = 0) → k = 1 :=
by
  sorry

-- Assuming a > 1, determine monotonicity and solve the inequality
axiom a_gt_1 : a > 1

theorem monotonicity_and_inequality_solution :
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 1 a < f x2 1 a) ∧
  (∃ s : set ℝ, s = {x : ℝ | -1 - real.sqrt 2 < x ∧ x < -1 + real.sqrt 2} ∧
  ∀ x ∈ s, f (x^2) 1 a + f (2*x - 1) 1 a < 0) :=
by
  sorry

end k_value_monotonicity_and_inequality_solution_l595_595611


namespace ordered_pair_solution_l595_595502

theorem ordered_pair_solution 
  (x y : ℤ) 
  (h₁ : 16 * x + 24 * y = 32) 
  (h₂ : 24 * x + 16 * y = 48) :
  (x = 2 ∧ y = 0) := 
by { unfold, sorry }

end ordered_pair_solution_l595_595502


namespace non_negative_real_inequality_l595_595854

theorem non_negative_real_inequality
  {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
by
  sorry

end non_negative_real_inequality_l595_595854


namespace only_A_is_direct_proportion_l595_595119

def direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def func_A : ℝ → ℝ := λ x, 2 * x
def func_B : ℝ → ℝ := λ x, (1 / x) + 2
def func_C : ℝ → ℝ := λ x, (1 / 2) * x - (2 / 3)
def func_D : ℝ → ℝ := λ x, 2 * x^2

theorem only_A_is_direct_proportion :
  direct_proportion func_A ∧
  ¬ direct_proportion func_B ∧
  ¬ direct_proportion func_C ∧
  ¬ direct_proportion func_D :=
by
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

end only_A_is_direct_proportion_l595_595119


namespace sin_30_eq_half_l595_595355

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595355


namespace sin_30_eq_half_l595_595214

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595214


namespace sum_of_divisors_prime_factors_l595_595690

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595690


namespace sum_of_divisors_prime_factors_450_l595_595740

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595740


namespace sin_30_deg_l595_595341

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595341


namespace probability_of_2_reds_before_3_greens_l595_595805

theorem probability_of_2_reds_before_3_greens :
  let total_chips := 7
  let red_chips := 4
  let green_chips := 3
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose 5 2
  (favorable_arrangements / total_arrangements : ℚ) = (2 / 7 : ℚ) :=
by
  let total_chips := 7
  let red_chips := 4
  let green_chips := 3
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose 5 2
  have fraction_computation :
    (favorable_arrangements : ℚ) / (total_arrangements : ℚ) = (2 / 7 : ℚ)
  {
    sorry
  }
  exact fraction_computation

end probability_of_2_reds_before_3_greens_l595_595805


namespace sin_30_eq_half_l595_595445

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595445


namespace interval_between_births_l595_595912

noncomputable theory

theorem interval_between_births (Y I : ℕ) (H1 : Y = 2) (H2 : Y + (Y + I) + (Y + 2 * I) + (Y + 3 * I) + (Y + 4 * I) = 90) : I = 8 := 
by 
  sorry

end interval_between_births_l595_595912


namespace minimal_area_circle_equation_l595_595566

theorem minimal_area_circle_equation :
  ∃ (x₀ y₀ r : ℝ), 
    (∀ x y : ℝ, (x + 4 + y * 2 = 0 → x ^ 2 + y ^ 2 + 2 * x - 4 * y + 1 = 0 → (x - x₀) ^ 2 + (y - y₀) ^ 2 = r ^ 2)) ∧
    x₀ = -13 / 5 ∧ y₀ = 6 / 5 ∧ r = sqrt (4 / 5) :=
sorry

end minimal_area_circle_equation_l595_595566


namespace sin_30_eq_one_half_l595_595310

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595310


namespace A_inter_complement_B_is_empty_l595_595625

open Set Real

noncomputable def U : Set Real := univ

noncomputable def A : Set Real := { x : Real | ∃ (y : Real), y = sqrt (Real.log x) }

noncomputable def B : Set Real := { y : Real | ∃ (x : Real), y = sqrt x }

theorem A_inter_complement_B_is_empty :
  A ∩ (U \ B) = ∅ :=
by
    sorry

end A_inter_complement_B_is_empty_l595_595625


namespace perpendicular_collinear_l595_595106

-- Given two circles with different radii O1 and O2 tangent to a larger circle O at points S and T respectively.
variables (O O1 O2 : Circle)
variables (S T M N : Point)
variables (different_radii : O1.radius ≠ O2.radius)
variables (tangent_S : O1.isTangent O S)
variables (tangent_T : O2.isTangent O T)
variables (intersect_points : {M, N} ⊆ (O1.intersection O2))

-- Prove the necessary and sufficient condition for OM ⊥ MN is that S, N, T are collinear
theorem perpendicular_collinear (h : OM.perpendicular MN) : collinear {S, N, T} ↔ OM ⊥ MN :=
sorry

end perpendicular_collinear_l595_595106


namespace sin_30_eq_half_l595_595369

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595369


namespace dot_product_AB_AD_l595_595804

variable (A B C D : Type)
variable [VectorSpace ℝ A]
variable [InnerProductSpace ℝ A]

theorem dot_product_AB_AD (AB BC CD AD BD : A) 
  (h₁ : ∥AB∥ = 3)
  (h₂ : CD + 2 • DB = 0)
  (h₃ : AD = AB + BD) :
  (AB ⬝ AD) = 15 / 2 := sorry

end dot_product_AB_AD_l595_595804


namespace largest_n_divisibility_l595_595108

theorem largest_n_divisibility (n : ℕ) (h : n = 14749) : (n^4 + 119) % (n + 11) = 0 :=
by
  rw h
  sorry

end largest_n_divisibility_l595_595108


namespace sin_30_is_half_l595_595333

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595333


namespace triangle_classification_l595_595597

theorem triangle_classification 
  (a b c : ℝ) 
  (h : (b^2 + a^2) * (b - a) = b * c^2 - a * c^2) : 
  (a = b ∨ a^2 + b^2 = c^2) :=
by sorry

end triangle_classification_l595_595597


namespace partition_groups_l595_595041

theorem partition_groups (n : ℕ) (a : ℕ → ℝ) (d : ℕ) 
    (h_sum : (∑ i in range d, a i) = n)
    (h_bounds : ∀ i, i < d → 0 ≤ a i ∧ a i ≤ 1) :
    ∃ (k : ℕ), (∀ (partition : fin (k+1) → finset (fin d)), 
        (∀ i, (∑ j in partition i, a j) ≤ 1)) 
    → k = 2 * n - 1 := 
sorry

end partition_groups_l595_595041


namespace roots_quadratic_l595_595600

theorem roots_quadratic (m x₁ x₂ : ℝ) (h : m < 0) (h₁ : x₁ < x₂) (hx : ∀ x, (x^2 - x - 6 = m) ↔ (x = x₁ ∨ x = x₂)) : 
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 :=
by {
  sorry
}

end roots_quadratic_l595_595600


namespace sin_30_eq_half_l595_595250

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595250


namespace sin_30_eq_half_l595_595400

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595400


namespace sum_of_drawn_vegetable_oil_and_fruits_vegetables_l595_595149

-- Definitions based on conditions
def varieties_of_grains : ℕ := 40
def varieties_of_vegetable_oil : ℕ := 10
def varieties_of_animal_products : ℕ := 30
def varieties_of_fruits_vegetables : ℕ := 20
def total_sample_size : ℕ := 20

def sampling_fraction : ℚ := total_sample_size / (varieties_of_grains + varieties_of_vegetable_oil + varieties_of_animal_products + varieties_of_fruits_vegetables)

def expected_drawn_vegetable_oil : ℚ := varieties_of_vegetable_oil * sampling_fraction
def expected_drawn_fruits_vegetables : ℚ := varieties_of_fruits_vegetables * sampling_fraction

-- The theorem to be proved
theorem sum_of_drawn_vegetable_oil_and_fruits_vegetables : 
  expected_drawn_vegetable_oil + expected_drawn_fruits_vegetables = 6 := 
by 
  -- Placeholder for proof
  sorry

end sum_of_drawn_vegetable_oil_and_fruits_vegetables_l595_595149


namespace sin_of_30_degrees_l595_595268

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595268


namespace monic_poly_exists_l595_595542

theorem monic_poly_exists : 
  ∃ (p : Polynomial ℚ), p.monic ∧ p.degree = 4 ∧ Polynomial.aeval (√2 + √5) p = 0 :=
sorry

end monic_poly_exists_l595_595542


namespace distinct_prime_factors_of_sigma_450_l595_595679
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595679


namespace chords_intersect_midpoint_l595_595859

noncomputable def midpoint (A B O: ℝ) := (A + B) / 2

theorem chords_intersect_midpoint (A B K L M N O: ℝ):
  midpoint A B = O →
  midpoint K L = O →
  midpoint M N = O →
  (∃ P₁ P₂ : ℝ, midpoint P₁ P₂ = O ∧ P₁ = K ∧ P₂ = N ∧ midpoint P₁ P₂ = O ∧ P₁ = M ∧ P₂ = L)
:=
sorry

end chords_intersect_midpoint_l595_595859


namespace floor_plus_x_eq_17_over_4_l595_595518

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l595_595518


namespace sin_30_eq_half_l595_595359

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595359


namespace sum_of_divisors_prime_factors_l595_595696

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595696


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595764

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595764


namespace triangle_circumcircles_l595_595059

variable {α : Type} [metric_space α]

-- Points A, B, C, D
variables (A B C D A₁ B₁ C₁ : α)

-- Definitions of segments as metrics
variables (AD AA₁ BD BB₁ CD CC₁ : ℝ)

-- Hypotheses based on conditions
hypothesis h1 : ∀ a b : α, dist A D = AD
hypothesis h2 : ∀ a b : α, dist A A₁ = AA₁
hypothesis h3 : ∀ a b : α, dist B D = BD
hypothesis h4 : ∀ a b : α, dist B B₁ = BB₁
hypothesis h5 : ∀ a b : α, dist C D = CD
hypothesis h6 : ∀ a b : α, dist C C₁ = CC₁

-- Problem to be proved
theorem triangle_circumcircles :
  (AD / AA₁) + (BD / BB₁) + (CD / CC₁) = 1 :=
sorry

end triangle_circumcircles_l595_595059


namespace calculate_expression_l595_595948

noncomputable def a : ℤ := -(1 * 10^2011 + 4 * (10^2010 - 1) // 9 + 6)
noncomputable def b : ℤ := 5 * 10^2011 + 4 * (10^2010 - 1) // 9 + 4444

theorem calculate_expression : 
  (2 * a * b * (a^3 - b^3) / (a^2 + a * b + b^2) - (a - b) * (a^4 - b^4) / (a^2 - b^2)) = 343 :=
  sorry

end calculate_expression_l595_595948


namespace sin_30_eq_one_half_l595_595308

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595308


namespace sin_30_eq_half_l595_595447

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595447


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595760

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595760


namespace flagpole_height_l595_595972

theorem flagpole_height
  (AB : ℝ) (AD : ℝ) (BC : ℝ)
  (h1 : AB = 10)
  (h2 : BC = 3)
  (h3 : 2 * AD^2 = AB^2 + BC^2) :
  AD = Real.sqrt 54.5 :=
by 
  -- Proof omitted
  sorry

end flagpole_height_l595_595972


namespace sin_30_eq_half_l595_595240

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595240


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595705

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595705


namespace sequence_product_property_l595_595941

theorem sequence_product_property (a : ℕ) (n : ℕ) : 
  ∃ u v : ℕ, (u > n) ∧ (x_n a n = x_n a u * x_n a v) :=
by 
  sorry

def x_n (a n : ℕ) : ℝ := n / (n + a)

end sequence_product_property_l595_595941


namespace sin_30_is_half_l595_595332

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595332


namespace sin_30_eq_half_l595_595463

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595463


namespace sin_of_30_degrees_l595_595269

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595269


namespace sin_thirty_degree_l595_595272

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595272


namespace impossible_to_arrange_hexagon_l595_595154

open Finset

theorem impossible_to_arrange_hexagon (vertices : Finset ℕ) (distinct_integers : vertices.card = 7) :
  ¬ ∃ (A B C D E F G : ℕ), 
    A ∈ vertices ∧ B ∈ vertices ∧ C ∈ vertices ∧ D ∈ vertices ∧ E ∈ vertices ∧ F ∈ vertices ∧ G ∈ vertices ∧
    distinct_integers ∧ 
    (A ≠ B ∧ A ≠ C ∧ ... ∧ G ≠ F) ∧
    (A ≤ G ∧ G ≤ B → A < B) ∧ 
    (B ≤ G ∧ G ≤ C → B < C) ∧ 
    (C ≤ G ∧ G ≤ D → C < D) ∧
    (D ≤ G ∧ G ≤ E → D < E) ∧
    (E ≤ G ∧ G ≤ F → E < F) ∧
    (F ≤ G ∧ G ≤ A → F < A) :=
by 
  sorry

end impossible_to_arrange_hexagon_l595_595154


namespace sin_30_is_half_l595_595322

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595322


namespace sin_30_eq_half_l595_595417

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595417


namespace sin_30_eq_one_half_l595_595309

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595309


namespace sin_30_eq_half_l595_595431

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595431


namespace sin_30_eq_half_l595_595467

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595467


namespace sequence_sum_l595_595907

theorem sequence_sum:
  ∀ (y : ℕ → ℕ), 
  (y 1 = 100) → 
  (∀ k ≥ 2, y k = y (k - 1) ^ 2 + 2 * y (k - 1) + 1) →
  ( ∑' n, 1 / (y n + 1) = 1 / 101 ) :=
by
  sorry

end sequence_sum_l595_595907


namespace work_completion_time_l595_595963

theorem work_completion_time (A_works_in : ℕ) (A_works_days : ℕ) (B_works_remainder_in : ℕ) (total_days : ℕ) :
  (A_works_in = 60) → (A_works_days = 15) → (B_works_remainder_in = 30) → (total_days = 24) := 
by
  intros hA_work hA_days hB_work
  sorry

end work_completion_time_l595_595963


namespace BP_tangent_circumcircle_PAD_l595_595590

variables {A B C D E P : Type*}
variable [euclidean_geometry A B C D E P]

-- Points D and E lie on the sidelines AB and BC of triangle ABC, respectively.
variables (AB BC : line)
variable (hD : D ∈ AB)
variable (hE : E ∈ BC)

-- Point P is in the interior of triangle ABC such that PE = PC
variable (hP_interior : P ∈ interior (triangle A B C))
variable (hPE_equal_PC : dist P E = dist P C)

-- Triangle DEP ∼ Triangle PCA
variable (h_similar_triangles : similar_triangles (triangle D E P) (triangle P C A))

-- The goal to prove: BP is tangent to the circumcircle of triangle PAD
theorem BP_tangent_circumcircle_PAD 
  (hD : D ∈ line_through A B) 
  (hE : E ∈ line_through B C)
  (hP_interior : P ∈ interior (triangle A B C))
  (hPE_equal_PC : dist P E = dist P C) 
  (h_similar_triangles : similar_triangles (triangle D E P) (triangle P C A)) : 
  is_tangent (line_through B P) (circumcircle (triangle P A D)) :=
sorry

end BP_tangent_circumcircle_PAD_l595_595590


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595634

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595634


namespace sum_of_divisors_prime_factors_450_l595_595730

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595730


namespace sin_30_eq_half_l595_595247

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595247


namespace race_course_length_to_finish_at_same_time_l595_595964

variable (v : ℝ) -- speed of B
variable (d : ℝ) -- length of the race course

-- A's speed is 4 times B's speed and A gives B a 75-meter head start.
theorem race_course_length_to_finish_at_same_time (h1 : v > 0) (h2 : d > 75) : 
  (1 : ℝ) / 4 * (d / v) = ((d - 75) / v) ↔ d = 100 := 
sorry

end race_course_length_to_finish_at_same_time_l595_595964


namespace part1_part2_l595_595044

-- Defining the sequence and the conditions
def sequence (x : ℕ → ℝ) := (x 0 = 1) ∧ ∀ n, 0 < x (n + 1) ≤ x n

-- Sum definition
def S (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  finset.sum (finset.range n) (λ k, (x k) ^ 2 / (x (k + 1)))

-- Part 1: Proving the existence of n such that S_n ≥ 3.999
theorem part1 (x : ℕ → ℝ) (h : sequence x) : ∃ n ≥ 1, S x n ≥ 3.999 :=
begin
  sorry
end

-- Part 2: Providing a sequence such that S_n < 4 for any n ≥ 1
def example_sequence (n : ℕ) : ℝ :=
  if n = 0 then 1 else (1 / 2) ^ n

theorem part2 : sequence example_sequence ∧ ∀ n ≥ 1, S example_sequence n < 4 :=
begin
  sorry
end

end part1_part2_l595_595044


namespace num_distinct_prime_factors_sum_divisors_450_l595_595650

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595650


namespace sin_of_30_degrees_l595_595261

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595261


namespace num_distinct_prime_factors_sum_divisors_450_l595_595655

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595655


namespace sin_30_eq_half_l595_595462

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595462


namespace sin_30_eq_half_l595_595435

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595435


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595710

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595710


namespace inappropriate_character_choice_l595_595940

-- Definitions and conditions
def is_main_character (c : String) : Prop := 
  c = "Gryphon" ∨ c = "Mock Turtle"

def characters : List String := ["Lobster", "Gryphon", "Mock Turtle"]

-- Theorem statement
theorem inappropriate_character_choice : 
  ¬ is_main_character "Lobster" :=
by 
  sorry

end inappropriate_character_choice_l595_595940


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595663

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595663


namespace karl_miles_driven_l595_595017

def gas_consumption_per_mile : ℚ := 1 / 30
def full_tank : ℚ := 16
def initial_drive_miles : ℚ := 360
def additional_gas : ℚ := 10
def final_tank_fraction : ℚ := 3 / 4

theorem karl_miles_driven : 
  let initial_gas := full_tank in
  let gas_used_first_leg := initial_drive_miles * gas_consumption_per_mile in
  let remaining_gas_first_leg := initial_gas - gas_used_first_leg in
  let gas_after_refuel := remaining_gas_first_leg + additional_gas in
  let final_gas_amount := final_tank_fraction * full_tank in
  let gas_used_second_leg := gas_after_refuel - final_gas_amount in
  let second_leg_miles := gas_used_second_leg / gas_consumption_per_mile in
  initial_drive_miles + second_leg_miles = 420 :=
by
  sorry

end karl_miles_driven_l595_595017


namespace sin_30_eq_half_l595_595419

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595419


namespace atm_passwords_count_l595_595185

theorem atm_passwords_count :
  let total_passwords := 10^4 in
  let restricted_passwords := 10 in
  total_passwords - restricted_passwords = 9990 :=
by
  let total_passwords := 10^4
  let restricted_passwords := 10
  show total_passwords - restricted_passwords = 9990
  sorry

end atm_passwords_count_l595_595185


namespace distinct_prime_factors_of_sigma_450_l595_595773

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595773


namespace distinct_prime_factors_of_sigma_450_l595_595681
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595681


namespace linear_function_difference_l595_595080

variable {f : ℝ → ℝ}

theorem linear_function_difference (h_lin : ∀ d : ℝ, f(d+2) - f(d) = 6) : f(1) - f(7) = -18 :=
by 
  sorry

end linear_function_difference_l595_595080


namespace mark_total_cents_l595_595868

theorem mark_total_cents (dimes nickels : ℕ) (h1 : nickels = dimes + 3) (h2 : dimes = 5) : 
  dimes * 10 + nickels * 5 = 90 := by
  sorry

end mark_total_cents_l595_595868


namespace recurrence_relation_divisible_by_5_l595_595846

-- Define the function f(n)
def f : ℕ → ℕ
| 0       := 1  -- Base case f(0)
| 1       := 2  -- Base case f(1)
| n + 2 := f n + f (n + 1)

-- Prove that f(n) = f(n-1) + f(n-2) for n ≥ 3
theorem recurrence_relation (n : ℕ) (h : n ≥ 3) : 
    f n = f (n - 1) + f (n - 2) :=
by {
  induction n with n ih; 
  -- Use induction and base cases here to structure proof
  sorry
}

-- Prove that f(5k + 3) is divisible by 5
theorem divisible_by_5 (k : ℕ) : 
    5 ∣ f (5 * k + 3) :=
by {
  induction k with k ih; 
  -- Use induction and divisibility properties here
  sorry
}

end recurrence_relation_divisible_by_5_l595_595846


namespace draw_at_least_two_first_class_l595_595183

theorem draw_at_least_two_first_class :
  let total_products := 9
  let first_class := 4
  let second_class := 3
  let third_class := 2
  let draw_count := 4
  let at_least_two_first_class : ℕ :=
    (nat.choose 4 2) * (nat.choose 5 2)
    + (nat.choose 4 3) * (nat.choose 5 1)
    + (nat.choose 4 4) * (nat.choose 5 0)
  at_least_two_first_class = (nat.choose 9 4) := 
sorry

end draw_at_least_two_first_class_l595_595183


namespace fyodor_can_not_always_win_l595_595927

-- Define structures for sandwiches and moves
structure Sandwich :=
(contains_sausage : Bool)
(contains_cheese : Bool)

-- Define the game state structure
structure GameState :=
(total_sandwiches : Nat)
(eaten_sandwiches : Nat)
(sandwiches : List Sandwich)

-- Define the initial game state
def initial_game_state (N : ℕ) : GameState := {
  total_sandwiches := 100 * N,
  eaten_sandwiches := 0,
  sandwiches := List.replicate (100 * N) { contains_sausage := true, contains_cheese := true }
}

-- Define what it means for Uncle Fyodor to win
def FyodorWins (state : GameState) : Prop :=
  state.eaten_sandwiches = state.total_sandwiches ∧ List.length state.sandwiches > 0 → (List.<|image_sentinel|>head! state.sandwiches).contains_sausage

-- Define the game dynamics (skipping the details)
def step_game (state : GameState) : GameState := sorry

-- Final assertion
theorem fyodor_can_not_always_win : ∀ (N : ℕ),
  ¬(∀ state, FyodorWins (step_game state)) := sorry

end fyodor_can_not_always_win_l595_595927


namespace sin_30_deg_l595_595338

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595338


namespace find_product_l595_595011

-- Define the conditions in the problem
variables (D E F D' E' F' P : Type)
variables (DD' EE' FF' : P) (concurrent_at_P : DD' = EE' = FF' = P)
variables (DP PD' EP PE' FP PF' : ℝ) (h : DP / PD' + EP / PE' + FP / PF' = 94)

-- Define the theorem to prove
theorem find_product : DP / PD' * EP / PE' * FP / PF' = 92 :=
by {
  sorry -- Proof omitted as requested
}

end find_product_l595_595011


namespace sin_30_eq_half_l595_595403

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595403


namespace sin_of_30_degrees_l595_595263

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595263


namespace sum_of_squares_of_coeffs_l595_595115

theorem sum_of_squares_of_coeffs : 
  let expr := (3 : ℝ) * (X ^ 3 - 4 * X + 5) - (5 : ℝ) * (X ^ 2 - 2 * X + 3),
      coeffs := Polynomial.coeffs expr,
      sum_squares := coeffs.map (λ c, c ^ 2).sum in
  sum_squares = 38 := 
by
  let expr := (3 : ℝ) * (X ^ 3 - 4 * X + 5) - (5 : ℝ) * (X ^ 2 - 2 * X + 3)
  let expected_expr := (3 : ℝ) * X ^ 3 - 5 * X ^ 2 - 2 * X
  have : expr = expected_expr, from sorry
  let coeffs := Polynomial.coeffs expected_expr
  have coeffs_x3 : coeffs.toList.nthLe 0 0 = 3, from sorry
  have coeffs_x2 : coeffs.toList.nthLe 1 0 = -5, from sorry
  have coeffs_x : coeffs.toList.nthLe 2 0 = -2, from sorry
  let sum_squares := coeffs.map (λ c, c ^ 2).toList.sum
  have : sum_squares = 3^2 + (-5)^2 + (-2)^2, from sorry
  have expected_sum : 3^2 + (-5)^2 + (-2)^2 = 38, from by norm_num
  exact expected_sum

end sum_of_squares_of_coeffs_l595_595115


namespace instantaneous_velocity_at_t_eq_3_l595_595619

noncomputable def velocity (t : ℝ) : ℝ :=
  deriv (λ t, (1 / 9) * t^3 + t) t

theorem instantaneous_velocity_at_t_eq_3 :
  velocity 3 = 4 :=
by
  sorry

end instantaneous_velocity_at_t_eq_3_l595_595619


namespace sin_30_deg_l595_595353

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595353


namespace identify_linear_equations_l595_595899

theorem identify_linear_equations :
  let eq1 := "4 + 8 = 12"
  let eq2 := "5x + 3 = 4"
  let eq3 := "2x + 3y = 0"
  let eq4 := "2a - 1 = 3 + 5a"
  let eq5 := "2x^2 + x = 1"
  true :=
by {
  -- Define a linear equation as one that can be written in the form ax + by = c
  -- where a, b, and c are constants, x and y are variables.
  let is_linear (eq : String) : Bool :=
    match eq with
    | "5x + 3 = 4" => true
    | "2a - 1 = 3 + 5a" => true
    | _ => false

  have eq1_non_linear : ¬ is_linear eq1 := by simp
  have eq2_linear : is_linear eq2 := by simp
  have eq3_non_linear : ¬ is_linear eq3 := by simp
  have eq4_linear : is_linear eq4 := by simp
  have eq5_non_linear : ¬ is_linear eq5 := by simp

  trivial
}

end identify_linear_equations_l595_595899


namespace sin_30_eq_one_half_l595_595306

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595306


namespace window_ratio_area_l595_595970

/-- Given a rectangle with semicircles at either end, if the ratio of AD to AB is 3:2,
    and AB is 30 inches, then the ratio of the area of the rectangle to the combined 
    area of the semicircles is 6 : π. -/
theorem window_ratio_area (AD AB r : ℝ) (h1 : AB = 30) (h2 : AD / AB = 3 / 2) (h3 : r = AB / 2) :
    (AD * AB) / (π * r^2) = 6 / π :=
by
  sorry

end window_ratio_area_l595_595970


namespace total_number_of_legs_l595_595057

theorem total_number_of_legs :
  let horses := 2 * 4 in
  let dogs := 5 * 4 in
  let cats := 7 * 4 in
  let turtles := 3 * 4 in
  let goat := 1 * 4 in
  let snakes := 4 * 0 in
  let spiders := 2 * 8 in
  let birds := 3 * 2 in
  horses + dogs + cats + turtles + goat + snakes + spiders + birds = 94 :=
by
  sorry

end total_number_of_legs_l595_595057


namespace intersection_I_intersection_II_l595_595134

-- Definitions for the sets
def A1 : Set (ℝ × ℝ) := { p | ∃ x, p = (x, x^2 + 2) }
def B1 : Set (ℝ × ℝ) := { p | ∃ x, p = (x, 6 - x^2) }
def result1 := { (Real.sqrt 2, 4), (-Real.sqrt 2, 4) }

def A2 : Set ℝ := { y | ∃ x, y = x^2 + 2 }
def B2 : Set ℝ := { y | ∃ x, y = 6 - x^2 }
def result2 := { y | 2 ≤ y ∧ y ≤ 6 }

-- Problem (I): Lean statement
theorem intersection_I : A1 ∩ B1 = result1 := sorry

-- Problem (II): Lean statement
theorem intersection_II : A2 ∩ B2 = result2 := sorry

end intersection_I_intersection_II_l595_595134


namespace mark_lloyd_ratio_l595_595866

theorem mark_lloyd_ratio (M L C : ℕ) (h1 : M = L) (h2 : M = C - 10) (h3 : C = 100) (h4 : M + L + C + 80 = 300) : M = L :=
by {
  sorry -- proof steps go here
}

end mark_lloyd_ratio_l595_595866


namespace sin_of_30_degrees_l595_595264

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595264


namespace sin_30_is_half_l595_595330

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595330


namespace lexie_older_than_brother_l595_595863

-- Conditions as given in the problem
variables (L B S : ℕ)
axiom h1 : L = 8
axiom h2 : S = 2 * L
axiom h3 : S - B = 14

-- Proof goal: L - B = 6
theorem lexie_older_than_brother : L - B = 6 :=
by
  have hL : L = 8 := h1
  have hS : S = 2 * L := h2
  have hSB : S - B = 14 := h3
  rw [hL, hS] at hSB
  have hS' : S = 16 := by rw [hL, hS]; exact rfl
  have hB : B = 2 := by linarith
  have hLB : L - B = 6 := by linarith
  exact hLB

end lexie_older_than_brother_l595_595863


namespace sin_30_eq_half_l595_595442

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595442


namespace num_roots_of_unity_real_power_l595_595103

theorem num_roots_of_unity_real_power :
  let z (k : ℕ) : ℂ := Complex.exp (2 * Real.pi * Complex.I * k / 30)
  in (Finset.filter (λ k, Complex.real? ((z k) ^ 10)) (Finset.range 30)).card = 10 := by
  sorry

end num_roots_of_unity_real_power_l595_595103


namespace no_infinite_arithmetic_progression_partitions_l595_595013

theorem no_infinite_arithmetic_progression_partitions :
  ∃ (A B : Set ℕ), 
    (∀ (a : ℕ) (d : ℕ), ¬ (∀ k : ℕ, a + k * d ∈ A)) ∧ 
    (∀ (a : ℕ) (d : ℕ), ¬ (∀ k : ℕ, a + k * d ∈ B)) :=
begin
  sorry
end

end no_infinite_arithmetic_progression_partitions_l595_595013


namespace sin_30_eq_half_l595_595251

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595251


namespace unoccupied_seats_in_business_class_l595_595167

/-
Define the numbers for each class and the number of people in each.
-/
def first_class_seats : Nat := 10
def business_class_seats : Nat := 30
def economy_class_seats : Nat := 50
def people_in_first_class : Nat := 3
def people_in_economy_class : Nat := economy_class_seats / 2
def people_in_business_and_first_class : Nat := people_in_economy_class
def people_in_business_class : Nat := people_in_business_and_first_class - people_in_first_class

/-
Prove that the number of unoccupied seats in business class is 8.
-/
theorem unoccupied_seats_in_business_class :
  business_class_seats - people_in_business_class = 8 :=
sorry

end unoccupied_seats_in_business_class_l595_595167


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595700

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595700


namespace black_ball_exists_l595_595102

theorem black_ball_exists (balls : List Bool) (h_black_count : (balls.count (λ b, b = tt)) = 5) (h_white_count : (balls.count (λ b, b = ff)) = 4) :
  ∃ i, balls.get i = tt ∧ (balls.drop (i + 1)).count (λ b, b = ff) = (balls.drop (i + 1)).count (λ b, b = tt) := 
sorry

end black_ball_exists_l595_595102


namespace sin_30_deg_l595_595343

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595343


namespace length_of_platform_l595_595137

theorem length_of_platform {train_length : ℕ} {time_to_cross_pole : ℕ} {time_to_cross_platform : ℕ} 
  (h1 : train_length = 300) 
  (h2 : time_to_cross_pole = 18) 
  (h3 : time_to_cross_platform = 45) : 
  ∃ platform_length : ℕ, platform_length = 450 :=
by
  sorry

end length_of_platform_l595_595137


namespace expected_value_remainder_l595_595027

theorem expected_value_remainder (p : ℕ) (h_prime : Nat.prime p) (h_p : p = 2017) 
  (m n : ℕ) (h_mn : 9 * 6 ^ 2018 - 5 ^ 2019 = m ∧ 6 ^ 2018 = n):
  (m + n) % p = 235 :=
by
  have h_m := h_mn.1
  have h_n := h_mn.2
  rw [h_p] at h_prime
  rw [h_m, h_n]
  sorry

end expected_value_remainder_l595_595027


namespace last_digit_of_2_pow_2010_l595_595054

-- Define the pattern of last digits of powers of 2
def last_digit_of_power_of_2 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 0 -- This case is redundant as n % 4 ∈ {0, 1, 2, 3}

-- Main theorem stating the problem's assertion
theorem last_digit_of_2_pow_2010 : last_digit_of_power_of_2 2010 = 4 :=
by
  -- The proof is omitted
  sorry

end last_digit_of_2_pow_2010_l595_595054


namespace find_A_in_phone_number_l595_595991

theorem find_A_in_phone_number
  (A B C D E F G H I J : ℕ)
  (h_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧ 
            B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧ 
            C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧ 
            D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧ 
            E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧ 
            F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧ 
            G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
            H ≠ I ∧ H ≠ J ∧
            I ≠ J)
  (h_dec_ABC : A > B ∧ B > C)
  (h_dec_DEF : D > E ∧ E > F)
  (h_dec_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_consec_even_DEF : D % 2 = 0 ∧ E % 2 = 0 ∧ F % 2 = 0 ∧ E = D - 2 ∧ F = E - 2)
  (h_consec_odd_GHIJ : G % 2 = 1 ∧ H % 2 = 1 ∧ I % 2 = 1 ∧ J % 2 = 1 ∧ H = G - 2 ∧ I = H - 2 ∧ J = I - 2)
  (h_sum : A + B + C = 9) :
  A = 8 :=
sorry

end find_A_in_phone_number_l595_595991


namespace distinct_prime_factors_of_sigma_450_l595_595775

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595775


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595699

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595699


namespace sin_30_deg_l595_595354

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595354


namespace sin_30_eq_half_l595_595484

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595484


namespace complex_modulus_problem_l595_595583

-- Define the imaginary unit and the relevant complex numbers
def i : ℂ := complex.I

-- Define the complex number z
def z : ℂ := (2 - i) / (1 + i)

-- Define the modulus function for a complex number
def modulus (z : ℂ) : ℝ := complex.abs z

-- State the theorem to be proved
theorem complex_modulus_problem : modulus z = \frac{\sqrt{10}}{2} :=
by
  sorry

end complex_modulus_problem_l595_595583


namespace cookies_with_flour_l595_595020

theorem cookies_with_flour (cookies_per_4cups : ℕ) (h : cookies_per_4cups = 24) : 
  let cookies_per_5cups := (cookies_per_4cups * 5) / 4 in
  cookies_per_5cups = 30 := 
by 
  simp [h]
  sorry

end cookies_with_flour_l595_595020


namespace polyhedron_space_diagonals_l595_595969

theorem polyhedron_space_diagonals (V E F T Q : ℕ) (hV : V = 30) (hE : E = 72) (hF : F = 44) (hT : T = 30) (hQ : Q = 14) :
  let number_of_space_diagonals := (V * (V - 1) / 2) - E - 2 * Q in
  number_of_space_diagonals = 335 :=
by
  -- Introducing the given conditions
  intro V E F T Q hV hE hF hT hQ
  rw [hV, hE, hQ] -- Substituting the given values directly into the space diagonals expression
  let number_of_space_diagonals := (30 * (30 - 1) / 2) - 72 - 2 * 14
  -- It remains to show that number_of_space_diagonals equals 335
  have : number_of_space_diagonals = 335 := rfl
  exact this

end polyhedron_space_diagonals_l595_595969


namespace A_is_largest_l595_595120

-- Definitions of A, B, and C based on the problem conditions
def A : ℝ := (3004 / 3003) + (3004 / 3005)
def B : ℝ := (3006 / 3005) + (3006 / 3007)
def C : ℝ := (3005 / 3004) + (3005 / 3006)

-- The theorem stating that A is the largest
theorem A_is_largest : A > B ∧ A > C := by
  sorry

end A_is_largest_l595_595120


namespace arrangements_count_l595_595100

-- Define the conditions
def male_students_together (arrangement : List String) : Prop :=
  ∃ l1 l2 l3 r, arrangement = l1 ++ ["M1", "M2", "M3"] ++ l2 ++ ["F1", "F2", "F3", "F4"] ++ l3 ++ ["T1", "T2"] ++ r

def female_students_not_next_to_each_other (arrangement : List String) : Prop :=
  ∀ i, (i < arrangement.length - 1) → 
  arrangement.nth i ≠ some "F" ∨ arrangement.nth (i + 1) ≠ some "F"

def female_students_ordered (arrangement : List String) : Prop :=
  ∀ i j, (i < j ∧ arrangement.nth i = some "F1" ∧ arrangement.nth j = some "F4") → i < j

def teachers_not_at_ends (arrangement : List String) : Prop :=
  arrangement.head ≠ some "T1" ∧ arrangement.head ≠ some "T2" ∧ arrangement.last ≠ some "T1" ∧ arrangement.last ≠ some "T2"

def male_students_in_middle (arrangement : List String) : Prop :=
  ∃ l r, arrangement = l ++ ["M1", "M2", "M3"] ++ r

-- Final Lean 4 statement
theorem arrangements_count (arrangement : List String) :
  male_students_together arrangement ∧
  female_students_not_next_to_each_other arrangement ∧
  female_students_ordered arrangement ∧
  teachers_not_at_ends arrangement ∧
  male_students_in_middle arrangement →
  arrangement.length = 1728 :=
sorry

end arrangements_count_l595_595100


namespace sin_30_eq_half_l595_595217

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595217


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595718

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595718


namespace distinct_prime_factors_of_sigma_450_l595_595776

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595776


namespace exists_coloring_if_and_only_if_odd_l595_595802

/--
For a convex n-gon P, we color every edge and diagonal in one of n colors. 
We prove that there exists a coloring such that for any three different colors,
one can find a triangle with its vertices being vertices of the polygon P and 
its three edges colored with these three different colors if and only if n is odd.
-/
theorem exists_coloring_if_and_only_if_odd (n : ℕ)
  (hn : ∃ P : list (ℕ × ℕ), True) -- treating convex n-gon P as a list of edges
  (H : ∀ (i j : ℕ), i < n → j < n → i ≠ j → (∃ c : ℕ, c < n)) :
  (∃ (c : ℕ → ℕ → ℕ), ∀ (a b c : ℕ), 
    a < n → b < n → c < n → a ≠ b → b ≠ c → a ≠ c → 
    (∃ (i j k : ℕ), i < n → j < n → k < n → 
     c i j = a ∧ c j k = b ∧ c k i = c)) ↔ odd n := sorry

end exists_coloring_if_and_only_if_odd_l595_595802


namespace sin_30_eq_half_l595_595468

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595468


namespace infinite_sum_eq_n_l595_595029

-- Define the greatest integer function (floor function).
def floor (x : ℝ) : ℤ := Int.floor x

-- Define the infinite summation problem given n a positive integer.
theorem infinite_sum_eq_n (n : ℕ) (h : n > 0) : 
  ∑ k in Finset.range (Nat.log2 n + 1), floor ((n + 2^k) / (2^(k+1))) = n :=
by sorry

end infinite_sum_eq_n_l595_595029


namespace distinct_prime_factors_of_sigma_450_l595_595779

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595779


namespace find_interest_rate_l595_595048

noncomputable def principal : ℝ := 5331.168831168831
def total_amount_returned : ℝ := 8210
def time_period : ℕ := 9

theorem find_interest_rate :
  let total_interest := total_amount_returned - principal in
  let interest_rate := (total_interest * 100) / (principal * time_period) in
  interest_rate = 6 :=
by
  sorry

end find_interest_rate_l595_595048


namespace after_school_program_spots_l595_595799

theorem after_school_program_spots (B G : ℕ)
  (h1 : 5 * G = 13 * B)
  (h2 : G = B + 64) :
  let participating_girls := (7 * G + 9) / 10,  -- 70% of girls rounded up (using (7*G+9) / 10 for ceiling)
      participating_boys := 4 * B / 5           -- 80% of boys
  in participating_girls + participating_boys = 105 :=
by
  sorry

end after_school_program_spots_l595_595799


namespace sin_30_eq_half_l595_595209

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595209


namespace line_bisects_iff_equal_lengths_l595_595811

noncomputable def convex_quadrilateral (A B C D : Point) : Prop :=
has_convex_hull A B C D ∧ ¬(parallel A D B C)

noncomputable def circle_tangent_to (Γ : Circle) (A B : Point) : Prop :=
tangent Γ A ∧ tangent Γ B

noncomputable def segment_bisects (P M A B : Point) : Prop :=
midpoint M A B ∧ incidence P M (line A B)

theorem line_bisects_iff_equal_lengths
    {A B C D K L P : Point}
    {Γ : Circle} :
    convex_quadrilateral A B C D →
    circle_tangent_to Γ C D → intersects Γ A B K L →
    incidence A C (line A C) → incidence B D (line B D) →
    incidence P (line A C) → incidence P (line B D) →
    (∃ M, segment_bisects K P C D ↔ length L C = length L D) :=
by
  sorry

end line_bisects_iff_equal_lengths_l595_595811


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595724

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595724


namespace sin_30_eq_half_l595_595473

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595473


namespace chord_length_l595_595889

noncomputable def standard_equation_line_l (x y : ℝ) : Prop :=
  (√3 * x + y = 2 * √3)

noncomputable def standard_equation_circle_C (x y : ℝ) : Prop :=
  (x^2 + y^2 - 4 * x - 2 * y = 0)

noncomputable def length_of_chord_AB (r d : ℝ) : ℝ :=
  2 * (real.sqrt (r^2 - d^2))

theorem chord_length (x y : ℝ) (r d : ℝ) : 
    standard_equation_line_l x y ∧ 
    standard_equation_circle_C x y ∧ 
    r = real.sqrt (x-2)^2 + (y-1)^2 ∧
    d = abs((√3 * 2 + 1 - 2 * √3) / 2) →
    length_of_chord_AB r d = √19 :=
sorry

end chord_length_l595_595889


namespace sin_30_is_half_l595_595325

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595325


namespace unique_isogonal_conjugates_of_convex_pentagon_l595_595875

theorem unique_isogonal_conjugates_of_convex_pentagon
  (A : ℕ → Point)
  (h_convex : ∀ (i : ℕ), 1 ≤ i → i ≤ 5 → convex (A i) (A (i + 1)) (A (i + 2)))
  : ∃! (P Q : Point), ∀ (i : ℕ), 1 ≤ i → i ≤ 5 → 
      angle (P A i A (i - 1)) = angle (A (i + 1) A i Q) :=
sorry

end unique_isogonal_conjugates_of_convex_pentagon_l595_595875


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595709

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595709


namespace sin_30_eq_half_l595_595362

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595362


namespace sin_30_is_half_l595_595323

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595323


namespace num_distinct_prime_factors_sum_divisors_450_l595_595653

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595653


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595766

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595766


namespace sin_30_is_half_l595_595324

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595324


namespace minimum_c_value_l595_595792

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ n : ℕ, n * n * n = n

theorem minimum_c_value
    (a b c d e : ℕ)
    (h1 : a = c - 2)
    (h2 : b = c - 1)
    (h3 : d = c + 1)
    (h4 : e = c + 2)
    (h5 : is_perfect_square (b + c + d))
    (h6 : is_perfect_cube (a + b + c + d + e)) : c = 675 :=
begin
  sorry
end

end minimum_c_value_l595_595792


namespace sin_30_eq_half_l595_595418

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595418


namespace factorial_expression_value_l595_595116

theorem factorial_expression_value :
  (13.factorial - 11.factorial) / 10.factorial = 1705 := by
  sorry

end factorial_expression_value_l595_595116


namespace cot_30_deg_l595_595193

theorem cot_30_deg : (Real.cot (Real.pi / 6)) = Real.sqrt 3 := by
  sorry

end cot_30_deg_l595_595193


namespace net_percentage_decrease_l595_595128

-- Define the original salary
def original_salary (S : ℝ) := S

-- Define the salary after a 15% increase
def increased_salary (S : ℝ) := S * 1.15

-- Define the salary after a 15% reduction
def reduced_salary (S : ℝ) := (increased_salary S) * 0.85

-- Define the net change
def net_change (S : ℝ) := reduced_salary S - original_salary S

-- Define the net percentage change
def net_percentage_change (S : ℝ) := (net_change S / original_salary S) * 100

-- The main theorem to prove
theorem net_percentage_decrease (S : ℝ) : net_percentage_change S = -2.25 :=
by
  sorry

end net_percentage_decrease_l595_595128


namespace sin_30_eq_half_l595_595488

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595488


namespace isosceles_triangle_solution_l595_595574

noncomputable def isosceles_triangle_sides (x y : ℝ) : Prop :=
(x + 1/2 * y = 6 ∧ 1/2 * x + y = 12) ∨ (x + 1/2 * y = 12 ∧ 1/2 * x + y = 6)

theorem isosceles_triangle_solution :
  ∃ (x y : ℝ), isosceles_triangle_sides x y ∧ x = 8 ∧ y = 2 :=
sorry

end isosceles_triangle_solution_l595_595574


namespace distinct_prime_factors_of_sigma_450_l595_595684
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595684


namespace sin_30_eq_half_l595_595410

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595410


namespace flower_shop_february_roses_l595_595898

theorem flower_shop_february_roses (roses_oct : ℕ) (roses_nov : ℕ) (roses_dec : ℕ) (roses_jan : ℕ) (d : ℕ) :
  roses_oct = 108 →
  roses_nov = 120 →
  roses_dec = 132 →
  roses_jan = 144 →
  roses_nov - roses_oct = d →
  roses_dec - roses_nov = d →
  roses_jan - roses_dec = d →
  (roses_jan + d = 156) :=
by
  intros h_oct h_nov h_dec h_jan h_diff1 h_diff2 h_diff3
  rw [h_jan, h_diff1] at *
  sorry

end flower_shop_february_roses_l595_595898


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595660

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595660


namespace series_sum_remainder_l595_595195

theorem series_sum_remainder : 
  let S := ∑ k in Finset.range 503, 2014 / ((4 * (k+1) - 1) * (4 * (k+1) + 3))
  in S.toNearestInt % 5 = 3 :=
begin
  sorry
end

end series_sum_remainder_l595_595195


namespace turtle_speed_l595_595865

-- Define the conditions
def distance_to_park : ℝ := 2640  -- meters
def rabbit_jump_distance : ℝ := 36  -- meters per minute
def rabbit_rest_time (k : ℕ) : ℝ := 0.5 * k -- minutes for the k-th rest
def time_difference : ℝ := 3 + 20/60  -- 3 minutes and 20 seconds

-- Define a fictional equivalent proof problem
theorem turtle_speed :
  ∃ (turtle_speed : ℝ),
    (∀ (rabbit_jumps : ℝ) (rabbit_rests : ℕ),
      rabbit_jumps = distance_to_park / rabbit_jump_distance →
      rabbit_rests = (⌊rabbit_jumps / 3⌋:ℝ).to_nat →
      let rabbit_total_rest_time := (list.range rabbit_rests).sum rabbit_rest_time in
      let rabbit_total_time := rabbit_jumps + rabbit_total_rest_time in
      let turtle_time := rabbit_total_time - time_difference in
      turtle_speed = distance_to_park / turtle_time)
      :=
begin
  use 12,
  sorry
end

end turtle_speed_l595_595865


namespace sin_thirty_degrees_l595_595374

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595374


namespace problem1_problem2_l595_595573

variables {α : Type*} [inner_product_space ℝ α] [normed_group α] [normed_space ℝ α]
variables (A B C M N P I T Q I1 I2 : α)
variables (Γ : set α)
variables [is_circumcircle Γ A B C]
variables [in_bounds_circle A (circumcircle_center A B C) (circumradius A B C)]
variables (h1 : angle A < angle B)
variables (h2 : midpoint_arc Γ B C M)
variables (h3 : midpoint_arc Γ A C N)
variables (h4 : segment C P ||| segment M N)
variables (h5 : intersects_at (segment C P) Γ P)
variables (h6 : is_incenter I A B C)
variables (h7 : intersects_at_extended (segment P I) Γ T)
variables (h8 : on_different_arc AB Q A B)
variables (h9 : Q ≠ A ∧ Q ≠ T ∧ Q ≠ B)
variables (hI1 : is_incenter I1 A Q C)
variables (hI2 : is_incenter I2 Q C B)

-- Part 1: Prove MP · MT = NP · NT
theorem problem1 (MP MT NP NT : ℝ) : MP * MT = NP * NT :=
sorry

-- Part 2: Prove that points Q, I1, I2, and T are concyclic
theorem problem2 : ∃ circle_center_radius T, circle_center_radius (circle_center T) (circle_radius T) ∧
Q ∈ circle_center_radius ∧ I1 ∈ circle_center_radius ∧ I2 ∈ circle_center_radius :=
sorry

end problem1_problem2_l595_595573


namespace sin_30_eq_half_l595_595303
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595303


namespace sum_of_divisors_prime_factors_450_l595_595728

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595728


namespace chloe_carrots_l595_595130

variable (initial_picked : Nat) (threw_out : Nat) (picked_next_day : Nat)

theorem chloe_carrots (h1 : initial_picked = 48)
                     (h2 : threw_out = 45)
                     (h3 : picked_next_day = 42) :
  (initial_picked - threw_out + picked_next_day) = 45 :=
  
by 
  rw [h1, h2, h3]
  norm_num
  sorry

end chloe_carrots_l595_595130


namespace sin_30_eq_half_l595_595460

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595460


namespace sequence_contains_perfect_square_l595_595078

noncomputable def f (n : ℕ) : ℕ := n + Nat.floor (Real.sqrt n)

theorem sequence_contains_perfect_square (m : ℕ) : ∃ k : ℕ, ∃ p : ℕ, f^[k] m = p * p := by
  sorry

end sequence_contains_perfect_square_l595_595078


namespace integral_sin_pi_l595_595954

-- We need to define a statement to assert the integral of sin x from 0 to π equals 2.
theorem integral_sin_pi: ∫ x in 0..π, sin x = 2 := by
  sorry  -- proof to be filled in later

end integral_sin_pi_l595_595954


namespace expected_number_of_1s_rolling_two_dice_l595_595107

def prob_roll_1 := (1 : ℚ) / 6
def prob_roll_not_1 := (5 : ℚ) / 6
def prob_zero_1s := prob_roll_not_1 ^ 2
def prob_two_1s := prob_roll_1 ^ 2
def prob_one_1  := 1 - prob_zero_1s - prob_two_1s

theorem expected_number_of_1s_rolling_two_dice : 
  ∑ i in {0, 1, 2}, i * (if i = 0 then prob_zero_1s else if i = 1 then prob_one_1 else prob_two_1s) = 1 / 3 :=
by
  sorry

end expected_number_of_1s_rolling_two_dice_l595_595107


namespace quadrilateral_cyclic_bisectors_intersect_midpoint_l595_595949

theorem quadrilateral_cyclic_bisectors_intersect_midpoint
  (A B C D O : Point)
  (h_cyclic : CyclicQuadrilateral A B C D O)
  (b1 : Line bisects ∠DAB)
  (b2 : Line bisects ∠ABC)
  (b3 : Line bisects ∠BCD)
  (b4 : Line bisects ∠CDA)
  (W X Y Z : Point)
  (h1 : W ∈ Circle(A, B, C, D)) (h2 : X ∈ Circle(A, B, C, D))
  (h3 : Y ∈ Circle(A, B, C, D)) (h4 : Z ∈ Circle(A, B, C, D))
  (hW : W ∈ b1) (hX : X ∈ b2) (hY : Y ∈ b3) (hZ : Z ∈ b4) :
  midpoint (W, Y) = midpoint (X, Z) :=
sorry

end quadrilateral_cyclic_bisectors_intersect_midpoint_l595_595949


namespace sin_thirty_degree_l595_595286

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595286


namespace distinct_3x3x3_cube_constructions_l595_595975

theorem distinct_3x3x3_cube_constructions (white_cubes blue_cubes total_cubes : ℕ)
  (H1 : white_cubes = 13) (H2 : blue_cubes = 14) (H3 : total_cubes = 27) :
  (white_cubes + blue_cubes = total_cubes) →
  ∃ (n : ℕ), n = 3466866 :=
by
  intro h
  use 3466866
  sorry

end distinct_3x3x3_cube_constructions_l595_595975


namespace units_digit_7_pow_2024_l595_595933

theorem units_digit_7_pow_2024 : (7 ^ 2024) % 10 = 1 := 
by
  have cycle : List Int := [7, 9, 3, 1]
  have len_cycle : cycle.length = 4 := by decide
  have mod_result : 2024 % 4 = 0 := by norm_num
  have units_digit := cycle.get! (0)
  rw [List.get!]
  sorry

end units_digit_7_pow_2024_l595_595933


namespace sin_30_eq_half_l595_595216

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595216


namespace part_I_part_II_l595_595010

-- Definitions and conditions from the problem statement
variables {A B C : ℝ}
variables {a b c : ℝ}
variables (cos_A : ℝ) (cos_B : ℝ) (sin_A : ℝ) (sin_B : ℝ) (sin_C : ℝ)

-- Given conditions
axiom given_condition1 : cos_A / a + cos_B / b = sin_C / c
axiom given_condition2 : b^2 + c^2 - a^2 = (6 / 5) * b * c

-- Prove part (I)
theorem part_I (hA : cos_A = Math.cos A) (ha : sin_A = Math.sin A) (hb : sin_B = Math.sin B) (hc : sin_C = Math.sin C) :
  sin_A * sin_B = sin_C :=
by
  have : cos_A / a + cos_B / b = sin_C / c := given_condition1
  sorry

-- Prove part (II)
theorem part_II (hA : cos_A = Math.cos A) (hcos_A : cos_A = 2 / 5) (hb : b ≠ 0) (hc : c ≠ 0) :
  Real.tan B = (√21 / (√21 - 2)) :=
by
  have : b^2 + c^2 - a^2 = (6 / 5) * b * c := given_condition2
  sorry

end part_I_part_II_l595_595010


namespace polygon_angle_ratio_pairs_count_l595_595926

theorem polygon_angle_ratio_pairs_count :
  ∃ (m n : ℕ), (∃ (k : ℕ), (k > 0) ∧ (180 - 360 / ↑m) / (180 - 360 / ↑n) = 4 / 3
  ∧ Prime n ∧ (m - 6) * (n + 8) = 48 ∧ 
  ∃! (m n : ℕ), (180 - 360 / ↑m = (4 * (180 - 360 / ↑n)) / 3)) :=
sorry  -- Proof omitted, providing only the statement

end polygon_angle_ratio_pairs_count_l595_595926


namespace find_alpha_l595_595602

open Real

-- Conditions: Parametric equations of the line
def parametric_line (t α : ℝ) : ℝ × ℝ := (-2 + t * cos α, 1 + t * sin α)

-- Condition on alpha
def alpha_condition (α : ℝ) : Prop := 0 ≤ α ∧ α < π / 2

-- Polar equation of the curve
def polar_equation (ρ θ : ℝ) : Prop := ρ^2 - 4 * ρ * cos θ - 2 * ρ * sin θ - 4 = 0

-- Cartesian equation of the line
def cartesian_line (x y α : ℝ) : Prop := x * tan α - y + 2 * tan α + 1 = 0

-- Cartesian equation of the curve
def cartesian_curve (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 9

-- Problem: 
theorem find_alpha (α : ℝ) (t ρ θ : ℝ) (x y : ℝ) 
  (h1 : parametric_line t α = (x, y))
  (h2 : alpha_condition α)
  (h3 : polar_equation ρ θ)
  (h4 : cartesian_curve x y)
  (h5 : |(x - 2)^2 + (y - 1)^2| = 9)
  (h6 : |x * tan α - y + 2 * tan α + 1| = 2 * sqrt 2) : α = π / 4 := by
  sorry

end find_alpha_l595_595602


namespace sum_of_divisors_prime_factors_l595_595687

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595687


namespace compute_factorial_expression_l595_595492

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem compute_factorial_expression :
  factorial 9 - factorial 8 - factorial 7 + factorial 6 = 318240 := by
  sorry

end compute_factorial_expression_l595_595492


namespace area_ratio_proof_l595_595021

-- Definitions and conditions
def inscribed_circle_area := (5 + Real.sqrt 5) / 10 * Real.pi
def golden_ratio := (1 + Real.sqrt 5) / 2

-- Prove the result
theorem area_ratio_proof : 
  (∃ (a b c : ℤ), (c > 0) ∧ (∀ (s : ℝ), s = 92 → (s = 100 * a + 10 * b + c))) :=
by
  sorry

end area_ratio_proof_l595_595021


namespace sum_of_divisors_prime_factors_450_l595_595732

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595732


namespace elves_closed_eyes_l595_595056

theorem elves_closed_eyes :
  ∃ (age: ℕ → ℕ), -- Function assigning each position an age
  (∀ n, 1 ≤ n ∧ n ≤ 100 → (age n < age ((n % 100) + 1) ∧ age n < age (n - 1 % 100 + 1)) ∨
                          (age n > age ((n % 100) + 1) ∧ age n > age (n - 1 % 100 + 1))) :=
by
  sorry

end elves_closed_eyes_l595_595056


namespace unique_prime_digit_l595_595909

def is_prime (n : ℕ) : Prop := sorry -- assume a proper primality test

theorem unique_prime_digit :
  ∃! (B : ℕ), B ∈ {1, 3, 7, 9} ∧ is_prime (305200 + B) :=
sorry

end unique_prime_digit_l595_595909


namespace count_valid_n_l595_595494

theorem count_valid_n : 
  ∃ n : ℕ, (∑ k in finset.range (n + 1), 8) = 8000 ↔ n = 108 := 
sorry

end count_valid_n_l595_595494


namespace sin_thirty_degrees_l595_595375

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595375


namespace sin_of_30_degrees_l595_595259

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595259


namespace puzzle_pieces_left_l595_595051

theorem puzzle_pieces_left (total_pieces : ℕ) (pieces_each : ℕ) (R_pieces : ℕ) (Rhys_pieces : ℕ) (Rory_pieces : ℕ)
  (h1 : total_pieces = 300)
  (h2 : pieces_each = total_pieces / 3)
  (h3 : R_pieces = 25)
  (h4 : Rhys_pieces = 2 * R_pieces)
  (h5 : Rory_pieces = 3 * R_pieces) :
  total_pieces - (R_pieces + Rhys_pieces + Rory_pieces) = 150 :=
begin
  sorry
end

end puzzle_pieces_left_l595_595051


namespace sin_thirty_degree_l595_595285

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595285


namespace sum_of_divisors_prime_factors_450_l595_595731

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595731


namespace sin_thirty_deg_l595_595233

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595233


namespace solve_for_lambda_l595_595796

theorem solve_for_lambda :
  ∀ (λ : ℝ), let a := (1, λ, 1)
             let b := (2, -1, -2)
             let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
             let mag_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)
             let mag_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2)
             let cos_theta := dot_product / (mag_a * mag_b)
             cos_theta = Real.sqrt(2) / 6 →
             λ = -Real.sqrt(2) :=
by
  intros λ a b dot_product mag_a mag_b cos_theta h
  sorry

end solve_for_lambda_l595_595796


namespace sequence_value_l595_595815

def sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a (n + 2) + a n

theorem sequence_value :
  ∀ (a : ℕ → ℤ),
    sequence a →
    a 1 = 2 →
    a 2 = 5 →
    a 6 = -3 :=
by
  intros a h_seq h_a1 h_a2
  sorry

end sequence_value_l595_595815


namespace max_faces_of_intersection_l595_595036

def W : Set (ℝ^4) := { p | ∀ i, 0 ≤ p i ∧ p i ≤ 1}

-- Main theorem statement
theorem max_faces_of_intersection :
  ∃ d : ℝ, ∀ H : Set (ℝ^4), (∀ p, H p ↔ (p ∈ W ∧ p 0 + p 1 + p 2 + p 3 = d)) → (number_of_faces H = 8) :=
sorry

end max_faces_of_intersection_l595_595036


namespace unoccupied_seats_in_business_class_l595_595166

/-
Define the numbers for each class and the number of people in each.
-/
def first_class_seats : Nat := 10
def business_class_seats : Nat := 30
def economy_class_seats : Nat := 50
def people_in_first_class : Nat := 3
def people_in_economy_class : Nat := economy_class_seats / 2
def people_in_business_and_first_class : Nat := people_in_economy_class
def people_in_business_class : Nat := people_in_business_and_first_class - people_in_first_class

/-
Prove that the number of unoccupied seats in business class is 8.
-/
theorem unoccupied_seats_in_business_class :
  business_class_seats - people_in_business_class = 8 :=
sorry

end unoccupied_seats_in_business_class_l595_595166


namespace sum_of_coefficients_l595_595512

theorem sum_of_coefficients :
  let d : ℝ := 0
  ∃ (polynomial : ℝ → ℝ), polynomial d = -(4 - 2 * d) * (d + 3 * (4 - 2 * d)) ∧
  polynomial.coeff 20 = -14 :=
by
  sorry

end sum_of_coefficients_l595_595512


namespace inscribed_sphere_cone_radius_l595_595169

theorem inscribed_sphere_cone_radius :
  ∃ (a c : ℝ), (∀ (r : ℝ), r = a * real.sqrt c - a →
    (cone_base_radius = 12 ∧ cone_height = 24) → (a + c = 11)) := 
sorry

end inscribed_sphere_cone_radius_l595_595169


namespace sin_30_eq_half_l595_595405

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595405


namespace chairs_in_fifth_row_l595_595002

theorem chairs_in_fifth_row : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 14 ∧ 
    a 2 = 23 ∧ 
    a 3 = 32 ∧ 
    a 4 = 41 ∧ 
    a 6 = 59 ∧ 
    (∀ n, a (n + 1) = a n + 9) → 
  a 5 = 50 :=
by
  sorry

end chairs_in_fifth_row_l595_595002


namespace sin_30_eq_one_half_l595_595305

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595305


namespace total_travel_time_l595_595175

theorem total_travel_time (x y : ℝ) : 
  (x / 50 + y / 70 + 0.5) = (7 * x + 5 * y) / 350 + 0.5 :=
by
  sorry

end total_travel_time_l595_595175


namespace pieces_left_to_place_l595_595050

noncomputable def total_pieces : ℕ := 300
noncomputable def reyn_pieces : ℕ := 25
noncomputable def rhys_pieces : ℕ := 2 * reyn_pieces
noncomputable def rory_pieces : ℕ := 3 * reyn_pieces
noncomputable def placed_pieces : ℕ := reyn_pieces + rhys_pieces + rory_pieces
noncomputable def remaining_pieces : ℕ := total_pieces - placed_pieces

theorem pieces_left_to_place : remaining_pieces = 150 :=
by sorry

end pieces_left_to_place_l595_595050


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595702

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595702


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595768

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595768


namespace sin_30_eq_half_l595_595248

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595248


namespace modified_pyramid_volume_l595_595151

-- Defining the initial volume and the transformation of dimensions
def pyramid_initial_volume : ℝ := 60
def length_scale_factor : ℝ := 3
def width_scale_factor : ℝ := 1 / 2
def height_scale_factor : ℝ := 1  -- Height remains the same

-- The formula for the volume of a pyramid
def pyramid_volume (l w h : ℝ) : ℝ := (1 / 3) * l * w * h

-- Given conditions
variable (l w h : ℝ)
variable (h_nonneg : 0 ≤ h)

-- Prove the new volume is 90 cubic inches
theorem modified_pyramid_volume :
  let l_new := length_scale_factor * l
      w_new := width_scale_factor * w
      h_new := height_scale_factor * h
  in pyramid_volume l_new w_new h_new = 90 := by
  sorry

end modified_pyramid_volume_l595_595151


namespace sum_of_divisors_prime_factors_l595_595691

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595691


namespace cody_steps_l595_595201

theorem cody_steps (S steps_week1 steps_week2 steps_week3 steps_week4 total_steps_4weeks : ℕ) 
  (h1 : steps_week1 = 7 * S) 
  (h2 : steps_week2 = 7 * (S + 1000)) 
  (h3 : steps_week3 = 7 * (S + 2000)) 
  (h4 : steps_week4 = 7 * (S + 3000)) 
  (h5 : total_steps_4weeks = steps_week1 + steps_week2 + steps_week3 + steps_week4) 
  (h6 : total_steps_4weeks = 70000) : 
  S = 1000 := 
    sorry

end cody_steps_l595_595201


namespace sin_thirty_deg_l595_595224

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595224


namespace sin_30_eq_half_l595_595205

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595205


namespace distinct_prime_factors_sum_divisors_450_l595_595751

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595751


namespace distinct_prime_factors_of_sigma_450_l595_595778

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595778


namespace jacob_nathan_total_letters_l595_595828

/-- Jacob and Nathan's combined writing output in 10 hours. -/
theorem jacob_nathan_total_letters (jacob_speed nathan_speed : ℕ) (h1 : jacob_speed = 2 * nathan_speed) (h2 : nathan_speed = 25) : jacob_speed + nathan_speed = 75 → (jacob_speed + nathan_speed) * 10 = 750 :=
by
  intros h3
  rw [h1, h2] at h3
  simp at h3
  rw [h3]
  norm_num

end jacob_nathan_total_letters_l595_595828


namespace function_transformation_l595_595793

noncomputable def transformed_function (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x, f (2 * (x - (π / 3)))

theorem function_transformation (f : ℝ → ℝ) :
  (transformed_function (λ x, sin (x - π / 4))) = (λ x, sin (x / 2 + π / 12)) :=
sorry

end function_transformation_l595_595793


namespace quadratic_roots_bounds_l595_595598

theorem quadratic_roots_bounds (m x1 x2 : ℝ) (h : m < 0)
  (hx : x1 < x2) 
  (hr : ∀ x, x^2 - x - 6 = m → x = x1 ∨ x = x2) :
  -2 < x1 ∧ x2 < 3 :=
by
  sorry

end quadratic_roots_bounds_l595_595598


namespace slope_angle_of_line_l595_595910

theorem slope_angle_of_line : 
  let P := (5, 3)
  let Q := (-2, 4)
  let slope := (Q.2 - P.2) / (Q.1 - P.1) 
  let alpha := Real.pi - Real.arctan((1 : ℝ) / 7)
  slope = -1 / 7 → alpha = Real.pi - Real.arctan((1 : ℝ) / 7) :=
by 
  intros
  sorry

end slope_angle_of_line_l595_595910


namespace expression_value_l595_595194

theorem expression_value (x y z : ℕ) (hx : x = 2) (hy : y = 5) (hz : z = 3) :
  (3 * x^5 + 4 * y^3 + z^2) / 7 = 605 / 7 := by
  rw [hx, hy, hz]
  sorry

end expression_value_l595_595194


namespace sin_thirty_deg_l595_595226

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595226


namespace total_profit_l595_595962

variable (A_contribution : ℝ) (A_time : ℝ)
variable (B_contribution : ℝ) (B_time : ℝ)
variable (A_profit_share : ℝ)

theorem total_profit (h1 : A_contribution = 5000) (h2 : A_time = 8) 
  (h3 : B_contribution = 6000) (h4 : B_time = 5)
  (h5 : A_profit_share = 4800) : 
  ∃ P : ℝ, P = 8400 :=
by
  let A_effort := A_contribution * A_time
  let B_effort := B_contribution * B_time
  let total_effort := A_effort + B_effort
  have hA_effort : A_effort = 40000 := by simp [h1, h2]
  have hB_effort : B_effort = 30000 := by simp [h3, h4]
  have A_share := A_effort / total_effort * (1 : ℝ)
  have total_ratio : total_effort = 70000 := by simp [hA_effort, hB_effort]
  have A_ratio := 4/7 : ℝ
  have A_profit_share_ratio : 4/7 * A_profit_share = 4800 := by 
    simp [h5]
  use 8400
  simp
  sorry

end total_profit_l595_595962


namespace sin_30_eq_half_l595_595357

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595357


namespace ivan_apples_leftover_l595_595826

theorem ivan_apples_leftover :
  ∀ (total_apples : ℕ) (mini_pie_apples : ℚ) (num_mini_pies : ℕ),
    total_apples = 48 →
    mini_pie_apples = 1 / 2 →
    num_mini_pies = 24 →
    total_apples - (num_mini_pies * mini_pie_apples) = 36 :=
by
  intros total_apples mini_pie_apples num_mini_pies ht hm hn
  rw [ht, hm, hn]
  norm_num
  sorry

end ivan_apples_leftover_l595_595826


namespace sin_30_eq_half_l595_595404

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595404


namespace jackson_emma_probability_l595_595803

open_locale big_operators

theorem jackson_emma_probability :
  ∃ (p : ℚ), p = 55 / 303 ∧ jackson_emma_probability = 55 / 303 :=
begin
  let total_ways := (nat.choose 12 6) * (nat.choose 12 6),
  let successful_ways := (nat.choose 12 3) * (nat.choose 9 3) * (nat.choose 9 3),
  let probability := (successful_ways : ℚ) / total_ways,
  use probability,
  split,
  { refl },
  { 
    norm_num1,
    sorry 
  }
end

end jackson_emma_probability_l595_595803


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595706

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595706


namespace roots_are_conjugates_eq_zero_l595_595857

theorem roots_are_conjugates_eq_zero (a b : ℝ) :
  (∀ z : ℂ, z^2 + (6 + a*complex.I) * z + (13 + b*complex.I) = 0 →
  ∃ w : ℂ, w = complex.conj z) →
  (a = 0 ∧ b = 0) :=
by
  sorry

end roots_are_conjugates_eq_zero_l595_595857


namespace sin_30_eq_half_l595_595290
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595290


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595642

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595642


namespace pigeons_in_park_l595_595148

noncomputable def park_pigeons_total (B : ℕ) : ℕ :=
  let black_male_pigeons := 0.20 * B
  let black_female_pigeons := black_male_pigeons + 21
  let total_black_pigeons := black_male_pigeons + black_female_pigeons
  if total_black_pigeons = B then 2 * B else 0

theorem pigeons_in_park (B : ℕ) (h1 : 0.20 * B + (0.20 * B + 21) = B) : park_pigeons_total B = 70 :=
  sorry

end pigeons_in_park_l595_595148


namespace sin_thirty_degrees_l595_595384

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595384


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595762

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595762


namespace circumradius_geq_3_times_inradius_l595_595850

-- Define the variables representing the circumradius and inradius
variables {R r : ℝ}

-- Assume the conditions that R is the circumradius and r is the inradius of a tetrahedron
def tetrahedron_circumradius (R : ℝ) : Prop := true
def tetrahedron_inradius (r : ℝ) : Prop := true

-- State the theorem
theorem circumradius_geq_3_times_inradius (hR : tetrahedron_circumradius R) (hr : tetrahedron_inradius r) : R ≥ 3 * r :=
sorry

end circumradius_geq_3_times_inradius_l595_595850


namespace sin_30_eq_half_l595_595288
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595288


namespace distance_midpoint_chord_l595_595084

noncomputable def midpoint_chord_distance (k : ℝ) (A B : ℝ × ℝ)
  (h_line : 4 * k * A.fst - 4 * A.snd - k = 0 ∧ 4 * k * B.fst - 4 * B.snd - k = 0)
  (h_parabola : A.snd ^ 2 = A.fst ∧ B.snd ^ 2 = B.fst)
  (h_AB_dist : real.dist A B = 4) : real :=
  let M := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2) in
  real.abs (M.fst + 1 / 2)

theorem distance_midpoint_chord (k : ℝ) (A B : ℝ × ℝ)
  (h_line : 4 * k * A.fst - 4 * A.snd - k = 0 ∧ 4 * k * B.fst - 4 * B.snd - k = 0)
  (h_parabola : A.snd ^ 2 = A.fst ∧ B.snd ^ 2 = B.fst)
  (h_AB_dist : real.dist A B = 4) :
  midpoint_chord_distance k A B h_line h_parabola h_AB_dist = 9 / 4 :=
sorry

end distance_midpoint_chord_l595_595084


namespace sin_30_eq_half_l595_595450

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595450


namespace sum_of_divisors_prime_factors_l595_595694

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595694


namespace iso_triangle_perimeter_l595_595593

theorem iso_triangle_perimeter :
  ∃ p : ℕ, (p = 11 ∨ p = 13) ∧ ∃ a b : ℕ, a ≠ b ∧ a^2 - 8 * a + 15 = 0 ∧ b^2 - 8 * b + 15 = 0 :=
by
  sorry

end iso_triangle_perimeter_l595_595593


namespace max_value_fraction_l595_595551

theorem max_value_fraction (x y : ℝ) (hx : 1 / 3 ≤ x ∧ x ≤ 3 / 5) (hy : 1 / 4 ≤ y ∧ y ≤ 1 / 2) :
  (∃ x y, (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) ∧ (xy / (x^2 + y^2) = 6 / 13)) :=
by
  sorry

end max_value_fraction_l595_595551


namespace ratio_safety_gear_to_test_tubes_l595_595146

-- Definitions
def budget : ℕ := 325
def cost_flasks : ℕ := 150
def cost_test_tubes : ℕ := (2 * cost_flasks) / 3
def remaining_budget : ℕ := 25
def total_spent : ℕ := budget - remaining_budget
def cost_safety_gear : ℕ := total_spent - (cost_flasks + cost_test_tubes)

-- Theorem statement
theorem ratio_safety_gear_to_test_tubes : cost_safety_gear / cost_test_tubes = 1 / 2 :=
begin
  -- Using the given conditions directly in our statement
  unfold cost_safety_gear cost_flasks cost_test_tubes total_spent remaining_budget budget,
  sorry
end

end ratio_safety_gear_to_test_tubes_l595_595146


namespace sin_30_eq_half_l595_595437

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595437


namespace sin_thirty_degree_l595_595280

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595280


namespace problem_1_problem_2_problem_3_l595_595568

variable {α : Type*} [LinearOrder α] [Nontrivial α] [Field α]

-- Definitions and conditions from part a)
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_sequence (a b c : α) : Prop :=
  b * b = a * c

variable (a : ℕ → α) (S : ℕ → α) (t : α)

axiom h1 : is_arithmetic_sequence a 2
axiom h2 : S 3 = a 4 + 4
axiom h3 : is_geometric_sequence (a 2) (a 6) (a 18)

-- Statement for problem 1
theorem problem_1 : ∀ n, a n = 2 * n :=
sorry

-- Definitions related to problem 2
def b (n : ℕ) : α := a n / 2 ^ n
def T (n : ℕ) : α := ∑ i in Finset.range n, b (i + 1)

-- Statement for problem 2
theorem problem_2 : ∀ n, T n = 4 - (2 * n + 4) / 2 ^ n :=
sorry

-- Definitions related to problem 3
def c (n : ℕ) : α := (S n + t) ^ (0.5 : ℤ)

-- Statement for problem 3
theorem problem_3 : (∀ n, is_arithmetic_sequence (λ n, c n) 1) ↔ t = (1 / 4 : α) :=
sorry

end problem_1_problem_2_problem_3_l595_595568


namespace find_x_of_floor_plus_x_eq_17_over_4_l595_595527

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l595_595527


namespace geometric_seq_a7_l595_595001

-- Definitions for the geometric sequence and conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ}
axiom a1 : a 1 = 2
axiom a3 : a 3 = 4
axiom geom_seq : geometric_sequence a

-- Statement to prove
theorem geometric_seq_a7 : a 7 = 16 :=
by
  -- proof will be filled in here
  sorry

end geometric_seq_a7_l595_595001


namespace original_quantity_ghee_mixture_is_correct_l595_595801

-- Define the variables
def percentage_ghee (x : ℝ) := 0.55 * x
def percentage_vanasapati (x : ℝ) := 0.35 * x
def percentage_palm_oil (x : ℝ) := 0.10 * x
def new_mixture_weight (x : ℝ) := x + 20
def final_vanasapati_percentage (x : ℝ) := 0.30 * (new_mixture_weight x)

-- State the theorem
theorem original_quantity_ghee_mixture_is_correct (x : ℝ) 
  (h1 : percentage_ghee x = 0.55 * x)
  (h2 : percentage_vanasapati x = 0.35 * x)
  (h3 : percentage_palm_oil x = 0.10 * x)
  (h4 : percentage_vanasapati x = final_vanasapati_percentage x) :
  x = 120 := 
sorry

end original_quantity_ghee_mixture_is_correct_l595_595801


namespace sin_30_eq_half_l595_595438

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595438


namespace find_positive_integer_l595_595545

-- Definition of the sum of the digits of a number.
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

-- The theorem we want to prove.
theorem find_positive_integer (m : ℕ) (h : m = 36018) : 2001 * sum_of_digits m = m :=
by
  sorry

end find_positive_integer_l595_595545


namespace find_x_l595_595536

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l595_595536


namespace max_a4b2c_l595_595845

-- Define the conditions and required statement
theorem max_a4b2c (a b c : ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a + b + c = 1) :
    a^4 * b^2 * c ≤ 1024 / 117649 :=
sorry

end max_a4b2c_l595_595845


namespace components_produced_ge_152_l595_595144

-- Define the conditions
def cost_per_component := 80
def shipping_per_component := 3
def fixed_monthly_costs := 16500
def selling_price_per_component := 191.67

-- Define the expression for the total cost
def total_cost (x : ℕ) : ℝ := (cost_per_component + shipping_per_component) * x + fixed_monthly_costs

-- Define the expression for revenue
def revenue (x : ℕ) : ℝ := selling_price_per_component * x

-- State the main theorem
theorem components_produced_ge_152 (x : ℕ) : revenue x ≥ total_cost x → x ≥ 152 :=
by
  -- Insert proof here (skipped)
  sorry

end components_produced_ge_152_l595_595144


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595670

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595670


namespace unique_circle_arrangement_l595_595025

open Function

variables (k : ℕ) (S : Set ℕ)

-- Assume k is a positive integer and S is a finite set of odd prime numbers
variable [k_pos : k > 0]
variable [fin_S : S.Finite]
variable [odd_S : ∀ p ∈ S, Nat.Prime p ∧ p % 2 = 1]

-- Lean statement to prove uniqueness of arrangement
theorem unique_circle_arrangement :
  ∃! f : S → S,
    (Bijective f ∧
    ∀ x ∈ S, ∃ y ∈ S, (f x = y ∨ f y = x) ∧
    ∃ x', x' > 0 ∧ x * y = x'^2 + x' + k) ∨
    (Bijective f ∧
    ∀ x ∈ S, ∃ y ∈ S, ∃ x', (f x = y ∨ f y = x) ∧ x' > 0 ∧ x * y = x'^2 + x' + k) := sorry

end unique_circle_arrangement_l595_595025


namespace distinct_prime_factors_of_sigma_450_l595_595682
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595682


namespace sum_first_n_c_b_l595_595861

noncomputable def a : ℕ → ℝ
| 1     := 7
| 2     := 3
| (n+1) := 3 * a n - 2

def b : ℕ → ℝ :=
λ n, (a n - 1) / 2

def c : ℕ → ℝ :=
λ n, Real.logb 3 (b n)

def T : ℕ → ℝ
| 1     := 3
| n     := match n with 
           | 1 => 3
           | n+1 => (n-2) * (3^(n-2)) + T n
           end

theorem sum_first_n_c_b :
  ∀ n : ℕ, T n = 
  (2 * n - 5) / 4 * 3^(n - 1) + 15 / 4 :=
sorry

end sum_first_n_c_b_l595_595861


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595658

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595658


namespace extreme_values_find_a_range_l595_595606

open Real

-- Define the function f(x) = ln(x + 1) - a * x
def f (a x : ℝ) : ℝ := ln (x + 1) - a * x

-- Problem 1: Prove the properties of extreme values
theorem extreme_values (a : ℝ) :
  (a ≤ 0 → (∀ x, ¬∃ y, y = f a x ∧ (∀ z, f a z ≤ y) ∨ (∀ z, f a z ≥ y))) ∧
  (a > 0 → (∃ y, y = f a ((1 / a) - 1) ∧ ∀ x, f a x ≤ y ∧ (¬∃ z, f a z < y))) :=
by
  sorry

-- Problem 2: Prove the range of a for the given inequality
theorem find_a_range (a : ℝ) :
  (∀ x ∈ Icc 0 (∞ : ℝ), (f a x + a * x) / exp x ≤ a * x) ↔ 1 ≤ a :=
by
  sorry

end extreme_values_find_a_range_l595_595606


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595717

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595717


namespace sin_30_eq_half_l595_595390

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595390


namespace friends_pay_amount_l595_595893

theorem friends_pay_amount (total_bill : ℝ) (friends : ℕ) (silas_percentage : ℝ) (tip_percentage : ℝ) :
  total_bill = 200 → 
  friends = 8 →
  silas_percentage = 0.60 →
  tip_percentage = 0.15 → 
  let silas_pay := silas_percentage * total_bill in
  let remaining_amount := total_bill - silas_pay in
  let tip_amount := tip_percentage * total_bill in
  let total_amount := remaining_amount + tip_amount in
  let remaining_friends := friends - 1 in
  let amount_per_friend := total_amount / remaining_friends in
  amount_per_friend ≈ 15.71 :=
by
  intros h1 h2 h3 h4
  let silas_pay := silas_percentage * total_bill
  let remaining_amount := total_bill - silas_pay
  let tip_amount := tip_percentage * total_bill
  let total_amount := remaining_amount + tip_amount
  let remaining_friends := friends - 1
  let amount_per_friend := total_amount / remaining_friends
  have : amount_per_friend ≈ 15.71
  sorry

end friends_pay_amount_l595_595893


namespace first_day_exceeding_100_paperclips_l595_595836

def paperclips_day (k : ℕ) : ℕ := 3 * 2^k

theorem first_day_exceeding_100_paperclips :
  ∃ (k : ℕ), paperclips_day k > 100 ∧ k = 6 := by
  sorry

end first_day_exceeding_100_paperclips_l595_595836


namespace nonneg_real_inequality_l595_595852

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 := 
by
  sorry

end nonneg_real_inequality_l595_595852


namespace min_distance_eq_one_l595_595578

open Real

/-- Given real numbers a and b satisfy ln(b+1) + a - 3b = 0,
    and real numbers c and d satisfy 2d - c + sqrt(5) = 0,
    then the minimum value of (a-c)^2 + (b-d)^2 is 1. -/
theorem min_distance_eq_one
    (a b c d : ℝ)
    (h1 : ln(b + 1) + a - 3 * b = 0)
    (h2 : 2 * d - c + sqrt(5) = 0) :
    ∃ x0 y0, (x0 = a - c) ∧ (y0 = b - d) ∧ (x0^2 + y0^2 = 1) := 
sorry

end min_distance_eq_one_l595_595578


namespace log_exp_problem_l595_595787

theorem log_exp_problem (a : ℝ) (h : log 2 (a + 2) = 16) : 3^a = 3^(2^16 - 2) :=
by
  sorry

end log_exp_problem_l595_595787


namespace car_speed_in_first_hour_l595_595093

theorem car_speed_in_first_hour (x : ℝ) 
  (second_hour_speed : ℝ := 40)
  (average_speed : ℝ := 60)
  (h : (x + second_hour_speed) / 2 = average_speed) :
  x = 80 := 
by
  -- Additional steps needed to solve this theorem
  sorry

end car_speed_in_first_hour_l595_595093


namespace minimum_value_of_k_l595_595617

theorem minimum_value_of_k (x y : ℝ) (h : x * (x - 1) ≤ y * (1 - y)) : x^2 + y^2 ≤ 2 :=
sorry

end minimum_value_of_k_l595_595617


namespace sin_thirty_degrees_l595_595382

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595382


namespace area_quadrilateral_is_correct_l595_595808

noncomputable def area_quadrilateral
  (AB BC CD : ℝ) (angle_B angle_C : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_CD : CD = 5)
  (h_angle_B : angle_B = 120) (h_angle_C : angle_C = 120) : ℝ :=
  let sin_120 := Real.sin (Real.pi * 120 / 180) in
  let area_ABC := 1 / 2 * AB * BC * sin_120 in
  let area_BCD := 1 / 2 * BC * CD * sin_120 in
  area_ABC + area_BCD

theorem area_quadrilateral_is_correct :
  ∀ (AB BC CD : ℝ) (angle_B angle_C : ℝ),
    AB = 3 → BC = 4 → CD = 5 → angle_B = 120 → angle_C = 120 →
    area_quadrilateral AB BC CD angle_B angle_C AB BC CD angle_B angle_C = 8 * Real.sqrt 3 :=
by
  intros AB BC CD angle_B angle_C h_AB h_BC h_CD h_angle_B h_angle_C
  rw [area_quadrilateral, h_AB, h_BC, h_CD, h_angle_B, h_angle_C]
  sorry

end area_quadrilateral_is_correct_l595_595808


namespace sum_is_nine_given_probability_l595_595145

-- Define the event of rolling two dice.
def roll_dice : Type := ℕ × ℕ

-- Define the sum of the two dice.
def sum_dice (d : roll_dice) : ℕ := d.1 + d.2

-- Define the event space and probability function.
def event_space : Finset roll_dice := { (i, j) | i ∈ Finset.range 1 (6 + 1) ∧ j ∈ Finset.range 1 (6 + 1) }.toFinset

def probability (s : Finset roll_dice) : ℚ := (s.card : ℚ) / event_space.card.toRat

theorem sum_is_nine_given_probability :
  probability {d ∈ event_space | sum_dice d = 9} = 1/9 :=
sorry

end sum_is_nine_given_probability_l595_595145


namespace find_x_eq_nine_fourths_l595_595534

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l595_595534


namespace thirteen_in_base_two_l595_595515

theorem thirteen_in_base_two : nat.to_binary 13 = "1101" :=
sorry

end thirteen_in_base_two_l595_595515


namespace distinct_prime_factors_sum_divisors_450_l595_595754

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595754


namespace average_monthly_balance_l595_595998

theorem average_monthly_balance :
  let balances := [200, 250, 300, 350, 400] in
  (List.sum balances) / (balances.length) = 300 := by
  sorry

end average_monthly_balance_l595_595998


namespace distinct_prime_factors_of_sigma_450_l595_595671
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595671


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595631

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595631


namespace sin_of_30_degrees_l595_595254

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595254


namespace range_of_f_on_interval_l595_595562

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x

theorem range_of_f_on_interval : set.range (λ x : {x // 0 ≤ x ∧ x ≤ 2}, f x) = set.Icc 0 4 := by
  sorry

end range_of_f_on_interval_l595_595562


namespace john_unanswered_questions_l595_595838

theorem john_unanswered_questions
  (x y z : ℕ)
  (h1 : 5 * x + 2 * z = 93)
  (h2 : 4 * x - y = 54)
  (h3 : x + y + z = 30) : z = 9 :=
sorry

end john_unanswered_questions_l595_595838


namespace sin_30_eq_one_half_l595_595304

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595304


namespace sin_30_eq_half_l595_595427

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595427


namespace prob_Allison_wins_l595_595181

open ProbabilityTheory

def outcome_space (n : ℕ) := {k : ℕ // k < n}

def Brian_roll : outcome_space 6 := sorry -- Representing Brian's possible outcomes: 1, 2, 3, 4, 5, 6
def Noah_roll : outcome_space 2 := sorry -- Representing Noah's possible outcomes: 3 or 5 (3 is 0 and 5 is 1 in the index for simplicity)

noncomputable def prob_Brian_less_than_4 := (|{k : ℕ // k < 3}| : ℝ) / (|{k : ℕ // k < 6}|) -- Probability Brian rolls 1, 2, or 3
noncomputable def prob_Noah_less_than_4 := (|{k : ℕ // k = 0}| : ℝ) / (|{k : ℕ // k < 2}|)  -- Probability Noah rolls 3 (index 0)

theorem prob_Allison_wins : (prob_Brian_less_than_4 * prob_Noah_less_than_4) = 1 / 4 := by
  sorry

end prob_Allison_wins_l595_595181


namespace sin_30_deg_l595_595340

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595340


namespace sin_30_eq_half_l595_595428

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595428


namespace prod_second_largest_and_smallest_is_120_l595_595915

def numbers : List ℕ := [10, 11, 12, 13]

def second_largest (l : List ℕ) : ℕ :=
  (l.qsort (· <= ·)).get l.length.pred.pred

def smallest (l : List ℕ) : ℕ :=
  (l.qsort (· <= ·)).head!

theorem prod_second_largest_and_smallest_is_120 :
  second_largest numbers * smallest numbers = 120 := by
  sorry

end prod_second_largest_and_smallest_is_120_l595_595915


namespace total_basketballs_donated_l595_595158

theorem total_basketballs_donated 
  (total_donations : ℕ) 
  (hoops_donated : ℕ) 
  (half_hoops_with_basketballs : hoops_donated / 2 = 30) 
  (pool_floats_donated : ℕ) 
  (quarter_pool_floats_damaged : pool_floats_donated / 4 = 30)
  (footballs_donated : ℕ) 
  (tennis_balls_donated : ℕ) 
  (remaining_as_basketballs : total_donations - (hoops_donated + (pool_floats_donated - quarter_pool_floats_damaged) + footballs_donated + tennis_balls_donated) = 60) 
  : total_donations = 300 ∧ hoops_donated = 60 ∧ pool_floats_donated = 120 ∧ footballs_donated = 50 ∧ tennis_balls_donated = 40 → 
  ∃ total_basketballs : ℕ, total_basketballs = 90 :=
by
  intros
  sorry

end total_basketballs_donated_l595_595158


namespace distinct_prime_factors_of_sigma_450_l595_595780

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595780


namespace find_x_eq_nine_fourths_l595_595533

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l595_595533


namespace sin_thirty_degree_l595_595278

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595278


namespace moles_of_C6H6_l595_595784

def balanced_reaction (a b c d : ℕ) : Prop :=
  a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ a + b + c + d = 4

theorem moles_of_C6H6 (a b c d : ℕ) (h_balanced : balanced_reaction a b c d) :
  a = 1 := 
by 
  sorry

end moles_of_C6H6_l595_595784


namespace linear_function_not_in_fourth_quadrant_l595_595902

theorem linear_function_not_in_fourth_quadrant (a b : ℝ) (h : a = 2 ∧ b = 1) :
  ∀ (x : ℝ), (2 * x + 1 < 0 → x > 0) := 
sorry

end linear_function_not_in_fourth_quadrant_l595_595902


namespace sin_30_eq_half_l595_595252

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595252


namespace sin_30_eq_half_l595_595239

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595239


namespace problem1_problem2_l595_595622

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}

noncomputable def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

variable (a : ℝ) (x : ℝ)

-- Problem 1: Proving intersection of sets when a = 2
theorem problem1 (ha : a = 2) : (A a ∩ B a) = {x | 4 < x ∧ x < 5} :=
sorry

-- Problem 2: Proving the range of a for which B is a subset of A
theorem problem2 : {a | B a ⊆ A a} = {a | (1 < a ∧ a ≤ 3) ∨ a = -1} :=
sorry

end problem1_problem2_l595_595622


namespace intersection_condition_l595_595950

theorem intersection_condition 
  (α1 α2 β1 β2 γ1 γ2 : ℝ) 
  (h₁ : (1 + γ1) * ∥AC₁∥ = ∥AB∥)
  (h₂ : (1 + γ2) * ∥C₂B∥ = ∥AB∥)
  (h₃ : (1 + α1) * ∥BA1∥ = ∥BC∥)
  (h₄ : (1 + α2) * ∥BA2∥ = ∥BC∥)
  (h₅ : (1 + β1) * ∥CB1∥ = ∥CA∥)
  (h₆ : (1 + β2) * ∥CB2∥ = ∥CA∥) :
  (α1 * β1 * γ1 + α2 * β2 * γ2 + α1 * α2 + β1 * β2 + γ1 * γ2 = 1) ↔ 
  (∃ P, A2B1 ≠ B2C1 ∧ B2C1 ≠ C2A1 ∧ C2A1 ≠ A2B1 ∧ is_intersecting_at P (A2B1) (B2C1) (C2A1)) :=
sorry

end intersection_condition_l595_595950


namespace sin_30_eq_half_l595_595425

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595425


namespace sum_of_divisors_prime_factors_450_l595_595736

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595736


namespace sector_angle_measure_l595_595795

-- Define the variables
variables (r α : ℝ)

-- Define the conditions
def perimeter_condition := (2 * r + r * α = 4)
def area_condition := (1 / 2 * α * r^2 = 1)

-- State the theorem
theorem sector_angle_measure (h1 : perimeter_condition r α) (h2 : area_condition r α) : α = 2 :=
sorry

end sector_angle_measure_l595_595795


namespace fifteenth_prime_is_47_l595_595919

/-- Statement: Given that thirteen is the sixth prime number, we will prove that the fifteenth prime number is 47. -/
theorem fifteenth_prime_is_47 : (nat.find (λ n, n = 15 ∧ n.prime_at (15 - 1).prime) = 47) :=
by
  sorry

end fifteenth_prime_is_47_l595_595919


namespace sum_of_divisors_prime_factors_l595_595698

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595698


namespace sin_30_deg_l595_595339

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595339


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595763

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595763


namespace sum_of_divisors_prime_factors_l595_595695

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595695


namespace exists_triangle_with_given_vertex_centroid_circumcenter_l595_595575

variable (A S O : EuclideanGeometry.Point)

theorem exists_triangle_with_given_vertex_centroid_circumcenter :
  ∃ (B C : EuclideanGeometry.Point), 
    EuclideanGeometry.is_triangle A B C ∧
    ∃ G, EuclideanGeometry.is_centroid A B C S ∧
    EuclideanGeometry.is_circumcenter A B C O :=
sorry

end exists_triangle_with_given_vertex_centroid_circumcenter_l595_595575


namespace find_divisor_of_x_l595_595935

theorem find_divisor_of_x (x : ℕ) (q p : ℕ) (h1 : x % n = 5) (h2 : 4 * x % n = 2) : n = 9 :=
by
  sorry

end find_divisor_of_x_l595_595935


namespace rational_roots_of_quadratic_l595_595587

theorem rational_roots_of_quadratic (r : ℝ) (h : ∃ m : ℤ, r = m / 2) :
  r = 2 ∨ r = -2 ∨ r = -4 :=
begin
  sorry
end

end rational_roots_of_quadratic_l595_595587


namespace sin_30_eq_half_l595_595394

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595394


namespace letters_written_l595_595835

theorem letters_written (nathan_rate : ℕ) (jacob_rate : ℕ) (combined_rate : ℕ) (hours : ℕ) :
  nathan_rate = 25 →
  jacob_rate = 2 * nathan_rate →
  combined_rate = nathan_rate + jacob_rate →
  hours = 10 →
  combined_rate * hours = 750 :=
by
  intros
  sorry

end letters_written_l595_595835


namespace g_satisfies_functional_equation_l595_595034

noncomputable def g : ℝ → ℝ :=
  fun x => x + 4

theorem g_satisfies_functional_equation :
  ∀ x y : ℝ, (g x * g y - g(x * y)) / 4 = x + y + 3 :=
by
  intros x y
  rw [g, g, g]
  sorry

end g_satisfies_functional_equation_l595_595034


namespace correct_product_l595_595173

theorem correct_product (incorrect_product : ℝ) (incorrect_product_val : incorrect_product = 12.04) : ∃ correct_product : ℝ, correct_product = 0.1204 :=
by {
  use 0.1204,
  exact sorry
}

end correct_product_l595_595173


namespace generating_function_equality_gaussian_polynomial_equality_l595_595855

-- Given the generating function definitions
def generating_function (P : ℕ × ℕ → ℕ → ℕ) (k l : ℕ) (x : ℕ) : ℕ :=
  ∑ n in finset.range (k*l + 1), (x^n) * (P (k, l) n)

-- Define conditions for f(k, l, x)
theorem generating_function_equality (P : ℕ × ℕ → ℕ → ℕ) (k l : ℕ) (x : ℕ) :
  generating_function P k l x = generating_function P (k-1) l x + x^k * generating_function P k (l-1) x ∧
  generating_function P k l x = generating_function P k (l-1) x + x^l * generating_function P (k-1) l x :=
sorry

-- Define the relationship with the Gaussian polynomial
def gaussian_polynomial (g : ℕ × ℕ → ℕ → ℕ) (k l : ℕ) (x : ℕ) : ℕ :=
  ∑ n in finset.range (k*l + 1), (x^n) * (g (k, l) n)

theorem gaussian_polynomial_equality (g : ℕ × ℕ → ℕ → ℕ) (P : ℕ × ℕ → ℕ → ℕ) (k l : ℕ) (x : ℕ) :
  generating_function P k l x = gaussian_polynomial g k l x :=
sorry

end generating_function_equality_gaussian_polynomial_equality_l595_595855


namespace num_distinct_prime_factors_sum_divisors_450_l595_595644

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595644


namespace salary_of_N_l595_595947

theorem salary_of_N (total_salary : ℝ) (percent_M_from_N : ℝ) (N_salary : ℝ) : 
  (percent_M_from_N * N_salary + N_salary = total_salary) → (N_salary = 280) :=
by
  sorry

end salary_of_N_l595_595947


namespace distinct_prime_factors_of_sigma_450_l595_595674
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595674


namespace pi_is_irrational_l595_595936

theorem pi_is_irrational :
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ π = p / q) :=
by
  sorry

end pi_is_irrational_l595_595936


namespace find_x_when_y_equals_2_l595_595914

theorem find_x_when_y_equals_2 :
  ∀ (y x k : ℝ),
  (y * (Real.sqrt x + 1) = k) →
  (y = 5 → x = 1 → k = 10) →
  (y = 2 → x = 16) := by
  intros y x k h_eq h_initial h_final
  sorry

end find_x_when_y_equals_2_l595_595914


namespace sin_30_eq_half_l595_595238

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595238


namespace fly_least_distance_l595_595986

noncomputable def least_distance_fly_crawled (radius height dist_start dist_end : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let slant_height := Real.sqrt (radius^2 + height^2)
  let angle := circumference / slant_height
  let half_angle := angle / 2
  let start_x := dist_start
  let end_x := dist_end * Real.cos half_angle
  let end_y := dist_end * Real.sin half_angle
  Real.sqrt ((end_x - start_x)^2 + end_y^2)

theorem fly_least_distance : least_distance_fly_crawled 500 (300 * Real.sqrt 3) 150 (450 * Real.sqrt 2) = 486.396 := by
  sorry

end fly_least_distance_l595_595986


namespace sin_30_is_half_l595_595328

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595328


namespace find_x_l595_595537

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l595_595537


namespace writing_rate_l595_595831

theorem writing_rate (nathan_rate : ℕ) (jacob_rate : ℕ) : nathan_rate = 25 → jacob_rate = 2 * nathan_rate → (nathan_rate + jacob_rate) * 10 = 750 :=
by
  assume h1 : nathan_rate = 25,
  assume h2 : jacob_rate = 2 * nathan_rate,
  have combined_rate : nathan_rate + jacob_rate = 75, from sorry, -- From calculation in solution step
  show (nathan_rate + jacob_rate) * 10 = 750, from sorry -- Multiplying by 10 as per solution step


end writing_rate_l595_595831


namespace sin_30_deg_l595_595342

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595342


namespace sin_30_eq_half_l595_595407

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595407


namespace sin_30_eq_half_l595_595371

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595371


namespace sum_of_divisors_prime_factors_l595_595697

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595697


namespace sin_30_eq_half_l595_595469

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595469


namespace sin_thirty_degrees_l595_595386

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595386


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595666

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595666


namespace cone_radius_is_four_l595_595987

-- Definition of the problem and the conditions
def arc_length : ℝ := 8 * Real.pi
def cone_base_radius (circumference : ℝ) : ℝ := circumference / (2 * Real.pi)

-- The theorem we need to prove
theorem cone_radius_is_four : cone_base_radius arc_length = 4 := by
  simp only [arc_length, cone_base_radius]
  norm_num

end cone_radius_is_four_l595_595987


namespace sin_30_eq_half_l595_595478

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595478


namespace sin_30_eq_one_half_l595_595307

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595307


namespace math_problem_l595_595603

-- Define the propositions as Lean definitions
def prop1 (x : ℝ) : Prop := x > 0 → x > Real.sin x
def prop2 : Prop := ¬ (∀ x > 0, x - Real.log x > 0) ↔ (∃ x > 0, x - Real.log x ≤ 0)
def prop3 (p q : Prop) : Prop := (p ∨ q) → (p ∧ q)
def prop4 (a b : ℝ) : Prop := (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ ((a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0)

-- The equivalence proof problem
theorem math_problem :
  (prop1 ∧ prop4) ∧ ¬(prop2 ∧ prop3) := 
by
  sorry

end math_problem_l595_595603


namespace cost_of_fencing_per_meter_in_cents_l595_595090

-- Definitions for the conditions
def ratio_length_width : ℕ := 3
def ratio_width_length : ℕ := 2
def total_area : ℕ := 3750
def total_fencing_cost : ℕ := 175

-- Main theorem statement with proof omitted
theorem cost_of_fencing_per_meter_in_cents :
  (ratio_length_width = 3) →
  (ratio_width_length = 2) →
  (total_area = 3750) →
  (total_fencing_cost = 175) →
  ∃ (cost_per_meter_in_cents : ℕ), cost_per_meter_in_cents = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_of_fencing_per_meter_in_cents_l595_595090


namespace minimum_value_of_z_l595_595584

/-- Given the constraints: 
1. x - y + 5 ≥ 0,
2. x + y ≥ 0,
3. x ≤ 3,

Prove that the minimum value of z = (x + y + 2) / (x + 3) is 1/3.
-/
theorem minimum_value_of_z : 
  ∀ (x y : ℝ), 
    (x - y + 5 ≥ 0) ∧ 
    (x + y ≥ 0) ∧ 
    (x ≤ 3) → 
    ∃ (z : ℝ), 
      z = (x + y + 2) / (x + 3) ∧
      z = 1 / 3 :=
by
  intros x y h
  sorry

end minimum_value_of_z_l595_595584


namespace sin_30_eq_half_l595_595300
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595300


namespace sin_30_is_half_l595_595336

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595336


namespace area_of_triangle_QPS_l595_595813

noncomputable def triangle_area (QR PR : ℝ) (angle_PRQ angle_RPS : ℝ) : ℝ :=
  let QS := QR + (2 * PR) -- QR + RS where RS = 2 * PR
  let height := (sqrt 3 / 2) * PR
  0.5 * QS * height

theorem area_of_triangle_QPS :
  ∀ (QR PR : ℝ) (angle_PRQ angle_RPS : ℝ),
  QR = 8 → PR = 12 → angle_PRQ = 120 → angle_RPS = 90 → 
  triangle_area QR PR angle_PRQ angle_RPS = 96 * sqrt 3 :=
by
  intros QR PR angle_PRQ angle_RPS hQR hPR hAngle_PRQ hAngle_RPS
  -- Conditions implied in the problem:
  -- - QR = 8
  -- - PR = 12
  -- - angle_PRQ = 120 degrees
  -- - angle_RPS = 90 degrees
  have h1: QR = 8 := hQR
  have h2: PR = 12 := hPR
  have h3: angle_PRQ = 120 := hAngle_PRQ
  have h4: angle_RPS = 90 := hAngle_RPS

  sorry

end area_of_triangle_QPS_l595_595813


namespace sin_30_is_half_l595_595331

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595331


namespace fifteenth_prime_is_47_l595_595918

/-- Statement: Given that thirteen is the sixth prime number, we will prove that the fifteenth prime number is 47. -/
theorem fifteenth_prime_is_47 : (nat.find (λ n, n = 15 ∧ n.prime_at (15 - 1).prime) = 47) :=
by
  sorry

end fifteenth_prime_is_47_l595_595918


namespace sin_30_eq_half_l595_595363

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595363


namespace variance_of_angles_l595_595874

theorem variance_of_angles (α β γ : ℝ) (h_sum: α + β + γ = 2 * Real.pi)
  (h_inside_triangle : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α < Real.pi ∧ β < Real.pi ∧ γ < Real.pi) :
  let mean := (α + β + γ) / 3 in
  let variance := (1 / 3) * ((α - mean)^2 + (β - mean)^2 + (γ - mean)^2) in
  variance < 10 * Real.pi^2 / 27 ∧ variance < 2 * Real.pi^2 / 9 :=
begin
  sorry
end

end variance_of_angles_l595_595874


namespace sin_30_eq_one_half_l595_595316

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595316


namespace num_distinct_prime_factors_sum_divisors_450_l595_595646

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595646


namespace distinct_prime_factors_of_sigma_450_l595_595771

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595771


namespace find_A_l595_595993

noncomputable def telephone_number_satisfies_conditions (A B C D E F G H I J : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J ∧
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  E = D - 2 ∧ F = D - 4 ∧ -- Given D, E, F are consecutive even digits
  H = G - 2 ∧ I = G - 4 ∧ J = G - 6 ∧ -- Given G, H, I, J are consecutive odd digits
  A + B + C = 9

theorem find_A :
  ∃ (A B C D E F G H I J : ℕ), telephone_number_satisfies_conditions A B C D E F G H I J ∧ A = 8 :=
by {
  sorry
}

end find_A_l595_595993


namespace sin_30_eq_half_l595_595472

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595472


namespace beetle_crawls_100th_segment_in_1300_seconds_l595_595981

def segment_length (n : ℕ) : ℕ :=
  (n / 4) + 1

def total_length (s : ℕ) : ℕ :=
  (s / 4) * 4 * (segment_length (s - 1)) * (segment_length (s - 1) + 1) / 2

theorem beetle_crawls_100th_segment_in_1300_seconds :
  total_length 100 = 1300 :=
  sorry

end beetle_crawls_100th_segment_in_1300_seconds_l595_595981


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595640

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595640


namespace sin_thirty_deg_l595_595222

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595222


namespace puzzle_pieces_left_l595_595052

theorem puzzle_pieces_left (total_pieces : ℕ) (pieces_each : ℕ) (R_pieces : ℕ) (Rhys_pieces : ℕ) (Rory_pieces : ℕ)
  (h1 : total_pieces = 300)
  (h2 : pieces_each = total_pieces / 3)
  (h3 : R_pieces = 25)
  (h4 : Rhys_pieces = 2 * R_pieces)
  (h5 : Rory_pieces = 3 * R_pieces) :
  total_pieces - (R_pieces + Rhys_pieces + Rory_pieces) = 150 :=
begin
  sorry
end

end puzzle_pieces_left_l595_595052


namespace range_of_a_for_monotonicity_l595_595504

theorem range_of_a_for_monotonicity (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x ∈ Ioo 1 2, f x = log a (2 - a / x)) → 
  (∀ x ∈ Ioo 1 2, f x < 0 ∧ f' x > 0) ↔ 
  1 < a ∧ a < 2 :=
sorry

end range_of_a_for_monotonicity_l595_595504


namespace annie_budget_l595_595189

theorem annie_budget :
  let budget := 120
  let hamburger_count := 8
  let milkshake_count := 6
  let hamburgerA := 4
  let milkshakeA := 5
  let hamburgerB := 3.5
  let milkshakeB := 6
  let hamburgerC := 5
  let milkshakeC := 4
  let costA := hamburgerA * hamburger_count + milkshakeA * milkshake_count
  let costB := hamburgerB * hamburger_count + milkshakeB * milkshake_count
  let costC := hamburgerC * hamburger_count + milkshakeC * milkshake_count
  let min_cost := min costA (min costB costC)
  budget - min_cost = 58 :=
by {
  sorry
}

end annie_budget_l595_595189


namespace average_of_multiples_of_10_l595_595930

theorem average_of_multiples_of_10 (a1 an d n avg : ℕ) (h1 : a1 = 10) (h2 : an = 160) (h3 : d = 10) (h4 : n = 16) :
  avg = (a1 + an) / 2 :=
by
  rw [h1, h2]
  have h_avg : avg = (10 + 160) / 2 := rfl
  exact h_avg

end average_of_multiples_of_10_l595_595930


namespace trigonometric_identity_solution_l595_595559

theorem trigonometric_identity_solution (θ : ℝ) 
(h1 : sin (θ + π / 4) = sqrt 2 / 4)
(h2 : θ ∈ Ioo (-π / 2) 0) :
  sin θ * cos θ = -3 / 8 ∧ cos θ - sin θ = sqrt 7 / 2 :=
by
  sorry

end trigonometric_identity_solution_l595_595559


namespace find_angle_l595_595920

variable (r1 r2 r3 : ℝ) (S U : ℝ) (θ : ℝ)

noncomputable def area_of_circle (r : ℝ) : ℝ := π * r^2

axiom radii : r1 = 4 ∧ r2 = 3 ∧ r3 = 2
axiom total_area : area_of_circle r1 + area_of_circle r2 + area_of_circle r3 = 29 * π
axiom shaded_unshaded_relation : S = (3 / 5) * U ∧ S + U = 29 * π
axiom area_contributions : S = 11 * θ + 9 * π

theorem find_angle (r1 r2 r3 S U θ : ℝ)
  (hradii : r1 = 4 ∧ r2 = 3 ∧ r3 = 2)
  (htotal_area : area_of_circle r1 + area_of_circle r2 + area_of_circle r3 = 29 * π)
  (hshaded_unshaded_relation : S = (3 / 5) * U ∧ S + U = 29 * π)
  (harea_contributions : S = 11 * θ + 9 * π) : θ = π / 8 := by
  sorry

end find_angle_l595_595920


namespace eccentricity_of_hyperbola_eq_sqrt2_l595_595495

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (line_through_focus : ∃ c : ℝ, c > 0 ∧ (∀ x y : ℝ, y = c * x) → c = 1) : ℝ :=
  let c := (a^2 + b^2).sqrt in
  c / a

theorem eccentricity_of_hyperbola_eq_sqrt2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (hyperbola_eq : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → 
      ∃ m: ℝ, (m * x - y = 0) ∧ (m = 1)) : 
    hyperbola_eccentricity a b ha hb hyperbola_eq = Real.sqrt 2 := 
  sorry

end eccentricity_of_hyperbola_eq_sqrt2_l595_595495


namespace series_sum_remainder_l595_595196

theorem series_sum_remainder : 
  let S := ∑ k in Finset.range 503, 2014 / ((4 * (k+1) - 1) * (4 * (k+1) + 3))
  in S.toNearestInt % 5 = 3 :=
begin
  sorry
end

end series_sum_remainder_l595_595196


namespace distinct_prime_factors_sum_divisors_450_l595_595747

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595747


namespace sin_30_eq_half_l595_595237

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595237


namespace jacob_nathan_total_letters_l595_595827

/-- Jacob and Nathan's combined writing output in 10 hours. -/
theorem jacob_nathan_total_letters (jacob_speed nathan_speed : ℕ) (h1 : jacob_speed = 2 * nathan_speed) (h2 : nathan_speed = 25) : jacob_speed + nathan_speed = 75 → (jacob_speed + nathan_speed) * 10 = 750 :=
by
  intros h3
  rw [h1, h2] at h3
  simp at h3
  rw [h3]
  norm_num

end jacob_nathan_total_letters_l595_595827


namespace suzannes_book_pages_l595_595888

-- Conditions
def pages_read_on_monday : ℕ := 15
def pages_read_on_tuesday : ℕ := 31
def pages_left : ℕ := 18

-- Total number of pages in the book
def total_pages : ℕ := pages_read_on_monday + pages_read_on_tuesday + pages_left

-- Problem statement
theorem suzannes_book_pages : total_pages = 64 :=
by
  -- Proof is not required, only the statement
  sorry

end suzannes_book_pages_l595_595888


namespace distinct_prime_factors_of_sum_of_divisors_450_l595_595767

def sum_of_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)

theorem distinct_prime_factors_of_sum_of_divisors_450 :
  ∃ (factors : ℕ), factors = (3) ∧ -- Number of distinct prime factors is 3
  sum_of_divisors 450 = 1209 ∧
  (450 = 2 * 3^2 * 5^2) :=
begin
  sorry
end

end distinct_prime_factors_of_sum_of_divisors_450_l595_595767


namespace sin_thirty_degrees_l595_595387

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595387


namespace sin_30_eq_half_l595_595243

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595243


namespace find_A_in_phone_number_l595_595992

theorem find_A_in_phone_number
  (A B C D E F G H I J : ℕ)
  (h_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧ 
            B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧ 
            C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧ 
            D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧ 
            E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧ 
            F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧ 
            G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
            H ≠ I ∧ H ≠ J ∧
            I ≠ J)
  (h_dec_ABC : A > B ∧ B > C)
  (h_dec_DEF : D > E ∧ E > F)
  (h_dec_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_consec_even_DEF : D % 2 = 0 ∧ E % 2 = 0 ∧ F % 2 = 0 ∧ E = D - 2 ∧ F = E - 2)
  (h_consec_odd_GHIJ : G % 2 = 1 ∧ H % 2 = 1 ∧ I % 2 = 1 ∧ J % 2 = 1 ∧ H = G - 2 ∧ I = H - 2 ∧ J = I - 2)
  (h_sum : A + B + C = 9) :
  A = 8 :=
sorry

end find_A_in_phone_number_l595_595992


namespace sin_of_30_degrees_l595_595253

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595253


namespace distinct_prime_factors_sum_divisors_450_l595_595741

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595741


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595726

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595726


namespace writing_rate_l595_595830

theorem writing_rate (nathan_rate : ℕ) (jacob_rate : ℕ) : nathan_rate = 25 → jacob_rate = 2 * nathan_rate → (nathan_rate + jacob_rate) * 10 = 750 :=
by
  assume h1 : nathan_rate = 25,
  assume h2 : jacob_rate = 2 * nathan_rate,
  have combined_rate : nathan_rate + jacob_rate = 75, from sorry, -- From calculation in solution step
  show (nathan_rate + jacob_rate) * 10 = 750, from sorry -- Multiplying by 10 as per solution step


end writing_rate_l595_595830


namespace volunteers_meet_again_in_360_days_l595_595510

-- Definitions of the given values for the problem
def ella_days := 5
def fiona_days := 6
def george_days := 8
def harry_days := 9

-- Statement of the problem in Lean 4
theorem volunteers_meet_again_in_360_days :
  Nat.lcm (Nat.lcm (Nat.lcm ella_days fiona_days) george_days) harry_days = 360 :=
by
  sorry

end volunteers_meet_again_in_360_days_l595_595510


namespace letters_written_l595_595834

theorem letters_written (nathan_rate : ℕ) (jacob_rate : ℕ) (combined_rate : ℕ) (hours : ℕ) :
  nathan_rate = 25 →
  jacob_rate = 2 * nathan_rate →
  combined_rate = nathan_rate + jacob_rate →
  hours = 10 →
  combined_rate * hours = 750 :=
by
  intros
  sorry

end letters_written_l595_595834


namespace floor_plus_x_eq_17_over_4_l595_595520

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l595_595520


namespace tangent_line_at_1_minimum_value_l595_595612

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def f_deriv (x : ℝ) : ℝ := Real.log x + 1

theorem tangent_line_at_1 :
  ∀ x y : ℝ, (x - y - 1 = 0) →
    ∀ a : ℝ, (f(1) = 0) → 
      (f_deriv 1 = 1) →
        y = x - 1 :=
sorry

theorem minimum_value :
  ∃ x : ℝ, f_deriv x = 0 ∧ f x = -1 / Real.exp 1 :=
sorry

end tangent_line_at_1_minimum_value_l595_595612


namespace sin_30_eq_half_l595_595202

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595202


namespace sin_30_deg_l595_595348

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595348


namespace angle_A_eq_60_l595_595009

open Real EuclideanGeometry

-- Define an arbitrary triangle ABC with incenter I, bisectors of angles B and C intersecting sides, and distances IP, IQ given
noncomputable def problem_triangle (A B C : Point)
  (incenter : Point)
  (P : Point)
  (Q : Point)
  (h1 : (AB : Line) > (AC : Line)) -- AB > AC
  (h2 : bisector (angle B) ∩ AC = P) -- Bisector of ∠B intersects AC at P
  (h3 : bisector (angle C) ∩ AB = Q) -- Bisector of ∠C intersects AB at Q
  (h4 : incenter_of_triangle incenter A B C) -- I is the incenter
  (h5 : distance incenter P = distance incenter Q) : Prop := -- IP = IQ
  angle A = 60 -- ∠A = 60 degrees

-- The theorem statement based on the above definitions
theorem angle_A_eq_60 (A B C : Point)
  (incenter : Point)
  (P : Point)
  (Q : Point)
  (h1 : (AB : Line) > (AC : Line))
  (h2 : bisector (angle B) ∩ AC = P)
  (h3 : bisector (angle C) ∩ AB = Q)
  (h4 : incenter_of_triangle incenter A B C)
  (h5 : distance incenter P = distance incenter Q) : 
  angle A = 60 :=
sorry

end angle_A_eq_60_l595_595009


namespace manufacturing_percentage_is_60_l595_595945

-- Define the conditions
def full_circle_angle : ℝ := 360
def manufacturing_sector_angle : ℝ := 216

-- Define the target result
def percentage_in_manufacturing : ℝ := (manufacturing_sector_angle / full_circle_angle) * 100

-- Prove that the percentage of employees in manufacturing is 60%
theorem manufacturing_percentage_is_60 :
  percentage_in_manufacturing = 60 :=
by
  -- Placeholder for the proof
  sorry

end manufacturing_percentage_is_60_l595_595945


namespace line_passes_fixed_point_l595_595618

-- Define the line equation and the proof goal
theorem line_passes_fixed_point (a ℝ): ∃ (x y: ℝ), ∀ a: ℝ, ay = (3a-1)x - 1 → (x, y) = (-1, -3) :=
begin
  sorry
end

end line_passes_fixed_point_l595_595618


namespace area_of_large_square_l595_595890

theorem area_of_large_square (s : ℝ) (h : 2 * s^2 = 14) : 9 * s^2 = 63 := by
  sorry

end area_of_large_square_l595_595890


namespace sin_30_eq_half_l595_595481

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595481


namespace sin_30_eq_half_l595_595444

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595444


namespace x_gt_1_sufficient_but_not_necessary_x_gt_0_l595_595952

theorem x_gt_1_sufficient_but_not_necessary_x_gt_0 (x : ℝ) :
  (x > 1 → x > 0) ∧ ¬(x > 0 → x > 1) :=
by
  sorry

end x_gt_1_sufficient_but_not_necessary_x_gt_0_l595_595952


namespace sin_30_eq_half_l595_595360

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595360


namespace correct_statement_l595_595122

def statement_A : Prop := ∀ (R : Type) [LinearOrderedField R] 
  (x y : R), regression_analysis x y → ¬(narrow_band x y ⟹ worse_regression_effect x y)

def statement_B : Prop := ∀ (R : Type) [LinearOrderedField R], 
  ∃ (p : R), p = 0.9 ∧ ¬rained_on_june_9th ⟹ ¬scientific_weather_forecast p

def statement_C : Prop := let data1 : List ℕ := [2, 3, 4, 5] 
                          let data2 : List ℕ := [4, 6, 8, 10] 
                          variance data1 = (1/2:ℝ) * variance data2

def statement_D : Prop := ∀ (x : ℝ), 
  let y : ℝ := regression_line (0.1 * x + 10) 
  ∀ (increment : ℝ), y = 0.1 * increment 

theorem correct_statement : statement_A = false → statement_B = false → statement_C = false → statement_D = true :=
sorry

end correct_statement_l595_595122


namespace largest_common_term_in_range_l595_595186

theorem largest_common_term_in_range :
  ∃ (a : ℕ), a < 150 ∧ (∃ (n : ℕ), a = 3 + 8 * n) ∧ (∃ (n : ℕ), a = 5 + 9 * n) ∧ a = 131 :=
by
  sorry

end largest_common_term_in_range_l595_595186


namespace sin_30_eq_half_l595_595297
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595297


namespace distinct_prime_factors_sum_divisors_450_l595_595743

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595743


namespace sin_30_eq_half_l595_595448

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595448


namespace number_of_valid_pairs_l595_595136

def digits_same (n m : ℕ) : Prop :=
  let to_digit_list := λ x : ℕ => (to_string x).to_list.filter_map (λ c, c.to_nat?)
  list.perm (to_digit_list n) (to_digit_list m)

def in_range (n m : ℕ) : Prop :=
  (1000 ≤ n ∧ n ≤ 9999) ∧ (1000 ≤ m ∧ m ≤ 9999)

def valid_pair (n m : ℕ) : Prop :=
  in_range n m ∧ digits_same n m ∧ (n - m) % 2010 = 0

theorem number_of_valid_pairs : 
  ∃ pairs : ℕ, pairs = 50 ∧
  ∀ (n m : ℕ), valid_pair n m ->
  pairs = 50 :=
by
  sorry

end number_of_valid_pairs_l595_595136


namespace sin_30_eq_half_l595_595207

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595207


namespace min_blocks_for_wall_l595_595157

-- Definitions based on conditions
def length_of_wall := 120
def height_of_wall := 6
def block_height := 1
def block_lengths := [1, 3]
def blocks_third_row := 3

-- Function to calculate the total blocks given the constraints from the conditions
noncomputable def min_blocks_needed : Nat := 164 + 80

-- Theorem assertion that the minimum number of blocks required is 244
theorem min_blocks_for_wall : min_blocks_needed = 244 := by
  -- The proof would go here
  sorry

end min_blocks_for_wall_l595_595157


namespace business_fraction_l595_595978

theorem business_fraction (x : ℚ) (H1 : 3 / 4 * x * 60000 = 30000) : x = 2 / 3 :=
by sorry

end business_fraction_l595_595978


namespace shot_put_distance_l595_595976

theorem shot_put_distance :
  (∃ x : ℝ, (y = - 1 / 12 * x^2 + 2 / 3 * x + 5 / 3) ∧ y = 0) ↔ x = 10 := 
by
  sorry

end shot_put_distance_l595_595976


namespace range_of_a_l595_595040

-- Define the properties and conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
if 0 ≤ x then x^2 else -x^2

theorem range_of_a :
  (∀ x : ℝ, odd_function f) →
  (∀ x : ℝ, 0 ≤ x → f x = x^2) →
  (∀ (a : ℝ), (∀ x ∈ set.Icc a (a + 2), f (x + a) ≥ 2 * f x) → a ≥ real.sqrt 2) :=
by
  intros oddF fx h
  sorry

end range_of_a_l595_595040


namespace fifteenth_prime_is_47_l595_595917

theorem fifteenth_prime_is_47 (h : ∀ n, Prime (prime_of_nat 6) ≠ 13 → n ≠ 6 → prime_of_nat n ≠ 47) : 
  prime_of_nat 15 = 47 := 
sorry

end fifteenth_prime_is_47_l595_595917


namespace sin_30_eq_one_half_l595_595318

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595318


namespace largest_x_value_l595_595901

theorem largest_x_value (b c m n : ℝ) :
  let p := λ x : ℝ, x^5 - 8 * x^4 + 20 * x^3 + 6 * x^2 - b * x + c
  let l := λ x : ℝ, m * x + n
  (∀ x : ℝ, p x = l x → (count x (roots (p - l)) = 4)) ∧
  (∃ x : ℝ, p.derivative x = l.derivative x)
  → 3 = max (roots (p - l)) := 
sorry

end largest_x_value_l595_595901


namespace correct_options_l595_595810

structure Point where
  x : ℝ
  y : ℝ

def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

def A : Point := { x := -1, y := 0 }
def B : Point := { x := 1, y := 0 }
def C : Point := { x := 0, y := 7 }
def P : Point := sorry -- here we'll assume P satisfies the given condition in the proof

lemma locus_equation (P : Point) (h : dist P A = Real.sqrt 2 * dist P B) : 
  (P.x - 3) ^ 2 + P.y ^ 2 = 8 := 
sorry

lemma maximum_angle (P : Point) (h : dist P A = Real.sqrt 2 * dist P B) : 
  PA = 2 * Real.sqrt 2 → sorry -- detailed max angle proof required
sorry

lemma minimum_distance_to_line (P : Point) (h : dist P A = Real.sqrt 2 * dist P B) : 
  Line.distance_to (3, 0) (7 * x - y + 7) - Real.sqrt 8 = 4 * Real.sqrt 2 / 5 :=
sorry

theorem correct_options (P : Point) (h : dist P A = Real.sqrt 2 * dist P B) :
  (locus_equation P h) ∧ (maximum_angle P h) ∧ (minimum_distance_to_line P h) :=
sorry

end correct_options_l595_595810


namespace find_x_satisfying_conditions_l595_595543

theorem find_x_satisfying_conditions :
  ∃ x : ℕ, (x % 2 = 1) ∧ (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) ∧ x = 59 :=
by
  sorry

end find_x_satisfying_conditions_l595_595543


namespace sum_of_divisors_prime_factors_l595_595685

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595685


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595637

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595637


namespace work_together_time_l595_595942

-- Conditions in the problem
def man_days : ℝ := 15
def father_days : ℝ := 20
def son_days : ℝ := 25

-- Converting the days to work rates
def man_rate : ℝ := 1 / man_days
def father_rate : ℝ := 1 / father_days
def son_rate : ℝ := 1 / son_days

-- Combined work rate
def combined_rate : ℝ := man_rate + father_rate + son_rate

-- Time to complete the job together
def total_days : ℝ := 1 / combined_rate

-- Proof statement
theorem work_together_time : total_days ≈ 6.38 := by 
  -- numerical verification so as not to deal with approximation
  sorry

end work_together_time_l595_595942


namespace sin_thirty_degree_l595_595282

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595282


namespace num_distinct_prime_factors_sum_divisors_450_l595_595647

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595647


namespace find_rate_of_interest_l595_595174

-- Define the conditions given in the problem
def simple_interest (SI P T R : ℝ) : Prop :=
  SI = (P * R * T) / 100

-- Theorem statement: Given the Simple Interest, Principal, and Time, prove the rate of interest is 9%
theorem find_rate_of_interest :
  ∀ (SI P T : ℝ), SI = 4034.25 → P = 8965 → T = 5 → ∃ R, simple_interest SI P T R ∧ R = 9 :=
by {
  intros SI P T hSI hP hT,
  -- Proof can be filled in later
  sorry
}

end find_rate_of_interest_l595_595174


namespace num_distinct_prime_factors_sum_divisors_450_l595_595651

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595651


namespace circumference_of_circle_l595_595985

-- Define the dimensions of the rectangle
def length := 7
def width := 24

-- Define the diameter of the circle as the diagonal of the rectangle
def diameter := Math.sqrt (length^2 + width^2)

-- Define the circumference of the circle
def circumference := Real.pi * diameter

-- Theorem stating that the circumference of the circle is 25π cm
theorem circumference_of_circle : circumference = 25 * Real.pi := by
  sorry

end circumference_of_circle_l595_595985


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595657

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595657


namespace sin_thirty_deg_l595_595230

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595230


namespace function_monotonic_increase_after_shift_l595_595881

theorem function_monotonic_increase_after_shift :
  ∀ x : ℝ, y = 2 * sin (2 * x + π / 3) →
  (∀ x : ℝ, x ∈ set.Icc (π / 12) (7 * π / 12) → strictly_monotone (λ x, 2 * sin (2 * x - 2 * π / 3))) := sorry

end function_monotonic_increase_after_shift_l595_595881


namespace num_distinct_prime_factors_sum_divisors_450_l595_595645

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595645


namespace sum_of_divisors_prime_factors_450_l595_595727

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595727


namespace graph_of_g_neg_x_l595_595616

-- Define the piecewise function g(x)
def g (x : ℝ) : ℝ :=
  if h₁ : -4 ≤ x ∧ x ≤ -1 then 2 * x + 8
  else if h₂ : -1 ≤ x ∧ x ≤ 1 then -x^2 + 2
  else if h₃ : 1 ≤ x ∧ x ≤ 4 then -2 * (x - 3)
  else 0

-- Define the function for the reflected graph g(-x)
def g_neg_x (x : ℝ) : ℝ := g (-x)

-- This is the statement that needs to be proved
theorem graph_of_g_neg_x :
  ∀ x : ℝ,
    (if -4 ≤ x ∧ x ≤ -1 then g_neg_x x = -2 * x + 8
     else if -1 ≤ x ∧ x ≤ 1 then g_neg_x x = -x^2 + 2
     else if 1 ≤ x ∧ x ≤ 4 then g_neg_x x = 2 * x + 6
     else g_neg_x x = 0) :=
by
  sorry

end graph_of_g_neg_x_l595_595616


namespace sin_thirty_degrees_l595_595380

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595380


namespace find_range_OM_l595_595594

noncomputable def distance (a b : ℝ) := (a - b) ^ 2

theorem find_range_OM (m n : ℝ) (hne : ¬ (m = 0 ∧ n = 0)) :
  let line_eq : ℝ → ℝ → Prop := λ x y, 2 * m * x - (4 * m + n) * y + 2 * n = 0 in
  let P : (ℝ × ℝ) := (2, 6) in
  let O : (ℝ × ℝ) := (0, 0) in
  let d := distance (distance O.1 P.1 + distance O.2 P.2) in
  let OM := sqrt ((O.1 - P.1) ^ 2 + (O.2 - P.2) ^ 2) in
  5 - sqrt 5 ≤ OM ∧ OM ≤ 5 + sqrt 5 :=
sorry

end find_range_OM_l595_595594


namespace trapezoid_perimeter_l595_595817

theorem trapezoid_perimeter (AB CD BC : ℝ) (h_parallels: AB = CD) (h_AB: AB = 4) (h_CD: CD = 4) (height: ℝ) (h_height: height = 2) (h_BC: BC = 6) : 
  let AC := Real.sqrt (height^2 + (AB/2)^2) in
  let AD := AC in
  Perimeter := AB + BC + CD + AD in
  Perimeter =  14 + 4 * Real.sqrt 2 :=
by sorry

end trapezoid_perimeter_l595_595817


namespace sin_two_alpha_correct_l595_595560

noncomputable def sin_two_alpha (α : ℝ) : ℝ :=
  2 * tan α / (1 + (tan α)^2)

theorem sin_two_alpha_correct (α : ℝ) (h : tan (π / 4 + α) = 2) : sin_two_alpha α = 3 / 5 := by
  sorry

end sin_two_alpha_correct_l595_595560


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595659

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595659


namespace distinct_prime_factors_of_sigma_450_l595_595781

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595781


namespace sin_thirty_deg_l595_595235

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595235


namespace unoccupied_business_class_seats_l595_595161

theorem unoccupied_business_class_seats :
  ∀ (first_class_seats business_class_seats economy_class_seats economy_fullness : ℕ)
  (first_class_occupancy : ℕ) (total_business_first_combined economy_occupancy : ℕ),
  first_class_seats = 10 →
  business_class_seats = 30 →
  economy_class_seats = 50 →
  economy_fullness = economy_class_seats / 2 →
  economy_occupancy = economy_fullness →
  total_business_first_combined = economy_occupancy →
  first_class_occupancy = 3 →
  total_business_first_combined = first_class_occupancy + business_class_seats - (business_class_seats - total_business_first_combined + first_class_occupancy) →
  business_class_seats - (total_business_first_combined - first_class_occupancy) = 8 :=
by
  intros first_class_seats business_class_seats economy_class_seats economy_fullness 
         first_class_occupancy total_business_first_combined economy_occupancy
         h1 h2 h3 h4 h5 h6 h7 h8 
  rw [h1, h2, h3, h4, h5, h6, h7] at h8
  sorry

end unoccupied_business_class_seats_l595_595161


namespace apprentice_time_l595_595904

theorem apprentice_time
  (x y : ℝ)
  (h1 : 7 * x + 4 * y = 5 / 9)
  (h2 : 11 * x + 8 * y = 17 / 18)
  (hy : y > 0) :
  1 / y = 24 :=
by
  sorry

end apprentice_time_l595_595904


namespace first_worker_time_l595_595996

theorem first_worker_time
  (T : ℝ) 
  (hT : T ≠ 0)
  (h_comb : (T + 8) / (8 * T) = 1 / 3.428571428571429) :
  T = 8 / 7 :=
by
  sorry

end first_worker_time_l595_595996


namespace marie_finishes_fourth_task_at_1240_PM_l595_595867

theorem marie_finishes_fourth_task_at_1240_PM :
  ∀ (start_time : Time) (second_task_end_time : Time), 
    start_time = Time.mk 8 0 →
    second_task_end_time = Time.mk 10 20 →
    (∀ (task_duration : Nat), 
      task_duration = 70 →
      marie_finishes_fourth_task_at_1240_PM := 
        start_time + 4 * task_duration = Time.mk 12 40) :=
by
  simp
  sorry

end marie_finishes_fourth_task_at_1240_PM_l595_595867


namespace hyperbola_equation_l595_595589

theorem hyperbola_equation (a b : ℝ) (x y : ℝ) (h_asymptote : y = sqrt 3 * x)
  (h_focus_on_directrix : sqrt (a^2 + b^2) = 12 ∧ b = sqrt 3 * a ∧ a > 0 ∧ b > 0) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = 36 ∧ b^2 = 108 ∧ (x^2 / a^2 - y^2 / b^2 = 1)) :=
begin
  -- Initial setup for the solution steps
  rcases h_focus_on_directrix with ⟨hc, hb, ha_gt, hb_gt⟩,
  use [6, sqrt 3 * 6],
  split,
  { exact zero_lt_six },
  { split,
    { exact mul_pos (sqrt 3) zero_lt_six },
    { split,
      { norm_num },
      { norm_num } } },
  sorry
end

end hyperbola_equation_l595_595589


namespace part_I_geometric_part_I_general_formula_part_II_sum_sequence_part_III_inequality_range_l595_595571

-- Define the sequence {a_n}
def a_seq : ℕ → ℤ
| 0     := 1
| (n+1) := 2 * a_seq n + 1

-- Define the sequence {b_n} where b_n = n / (a_n + 1)
def b_seq (n : ℕ) : ℤ :=
  n / (a_seq n + 1)

-- Sum of the first n terms of {b_n}, denoted as S_n
def S_n (n : ℕ) : ℤ :=
  (finset.range n).sum (λ i, b_seq (i + 1))

-- Main theorems based on the conditions

-- Part (I)
theorem part_I_geometric : 
  ∀ n : ℕ, ∃ k : ℕ, a_seq n + 1 = 2^k := sorry

theorem part_I_general_formula : 
  ∀ n : ℕ, a_seq n = 2^n - 1 := sorry

-- Part (II)
theorem part_II_sum_sequence :
  ∀ n : ℕ, S_n n = 2 - (2 + n) / 2^n := sorry

-- Part (III)
theorem part_III_inequality_range (a : ℝ) :
  (- (1 / 2) < a ∧ a < 3 / 4) ↔ 
  (∀ n : ℕ, S_n n + (n + 1) / 2^n - 1 > (-1)^n * a) := sorry

end part_I_geometric_part_I_general_formula_part_II_sum_sequence_part_III_inequality_range_l595_595571


namespace cart_coord_plane_l595_595055

theorem cart_coord_plane (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x^2 - x^3) :
    ({x : ℝ | f x = 0} = {0, 1}) → 
    (∃ p1 p2 : ℝ × ℝ, f p1.1 = 0 ∧ f p2.1 = 0 ∧ 
     (∀ x : ℝ, (0, 1) ⊆ set.real_univ) ∧
     p1 = (0, 0) ∧ p2 = (1, 0)) := 
sorry

end cart_coord_plane_l595_595055


namespace sin_30_eq_half_l595_595298
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595298


namespace sin_thirty_degrees_l595_595379

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595379


namespace floor_plus_x_eq_17_over_4_l595_595521

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l595_595521


namespace nonneg_real_inequality_l595_595851

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 := 
by
  sorry

end nonneg_real_inequality_l595_595851


namespace distinct_prime_factors_sum_divisors_450_l595_595752

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595752


namespace sin_30_eq_half_l595_595358

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595358


namespace problem_max_binomial_term_problem_sum_Sn_l595_595558

theorem problem_max_binomial_term (x : ℝ) (n : ℕ) (h : (1 + 3) ^ n - 2 ^ n = 992) : 
  (2 ^ n - 32) * (2 ^ n + 31) = 0 → n = 5 := 
by 
  sorry

theorem problem_sum_Sn (n : ℕ) : 
  (S_n : ℕ) = ∑ k in range(1, n+1), (C n k) * 2^(k-1) 
  → S_n = (3^n - 1) / 2 := 
by 
  sorry

end problem_max_binomial_term_problem_sum_Sn_l595_595558


namespace sin_30_eq_half_l595_595395

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595395


namespace sin_30_eq_half_l595_595204

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595204


namespace sin_30_eq_one_half_l595_595319

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595319


namespace sin_30_eq_half_l595_595393

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595393


namespace quadratic_roots_bounds_l595_595599

theorem quadratic_roots_bounds (m x1 x2 : ℝ) (h : m < 0)
  (hx : x1 < x2) 
  (hr : ∀ x, x^2 - x - 6 = m → x = x1 ∨ x = x2) :
  -2 < x1 ∧ x2 < 3 :=
by
  sorry

end quadratic_roots_bounds_l595_595599


namespace total_pages_l595_595061

def reading_rate (pages : ℕ) (minutes : ℕ) : ℝ :=
  pages / minutes

def total_pages_read (rate : ℝ) (minutes : ℕ) : ℝ :=
  rate * minutes

theorem total_pages (t : ℕ) (rene_pages : ℕ) (rene_minutes : ℕ) (lulu_pages : ℕ) (lulu_minutes : ℕ) (cherry_pages : ℕ) (cherry_minutes : ℕ) :
  t = 240 →
  rene_pages = 30 →
  rene_minutes = 60 →
  lulu_pages = 27 →
  lulu_minutes = 60 →
  cherry_pages = 25 →
  cherry_minutes = 60 →
  total_pages_read (reading_rate rene_pages rene_minutes) t +
  total_pages_read (reading_rate lulu_pages lulu_minutes) t +
  total_pages_read (reading_rate cherry_pages cherry_minutes) t = 328 :=
by
  intros t_val rene_p_val rene_m_val lulu_p_val lulu_m_val cherry_p_val cherry_m_val
  rw [t_val, rene_p_val, rene_m_val, lulu_p_val, lulu_m_val, cherry_p_val, cherry_m_val]
  simp [reading_rate, total_pages_read]
  norm_num
  sorry

end total_pages_l595_595061


namespace sin_30_eq_half_l595_595246

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595246


namespace complement_intersection_condition_l595_595087

variable {U : Type} -- Universal set U as a type
variable {M N : Set U} -- Subsets M and N of U

theorem complement_intersection_condition (x : U) :
  x ∈ (U \ (M ∩ N)) ↔ (x ∈ (U \ M) ∨ x ∈ (U \ N)) := sorry

end complement_intersection_condition_l595_595087


namespace values_of_a_l595_595623

open Set

noncomputable def A : Set ℝ := { x | x^2 - 2*x - 3 = 0 }
noncomputable def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else { x | a * x = 1 }

theorem values_of_a (a : ℝ) : (B a ⊆ A) ↔ (a = -1 ∨ a = 0 ∨ a = 1/3) :=
by 
  sorry

end values_of_a_l595_595623


namespace sin_30_eq_half_l595_595367

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595367


namespace equal_volume_pipes_l595_595974

theorem equal_volume_pipes (h : ℝ) (π : ℝ) :
  let r12 := 12 / 2
  let V12 := π * (r12 ^ 2) * h
  let r2 := 2 / 2
  let V2 := π * (r2 ^ 2) * h
  ∃ n : ℕ, V12 = n * V2 := by
  sorry

end equal_volume_pipes_l595_595974


namespace sum_odd_divisors_300_l595_595114

theorem sum_odd_divisors_300 :
  let odd_divisors := {d : ℕ | d ∣ 300 ∧ d % 2 = 1}
  (∑ x in odd_divisors, x) = 124 :=
by {
  let prime_factorization := (2^2) * 3 * (5^2),
  have h1 : 2^2 * 3 * 5^2 = 300 := by norm_num,
  let odd_divisors := {d : ℕ | d ∣ 300 ∧ d % 2 = 1},
  -- The steps are omitted for conciseness; main focus is on ensuring Lean can parse the statement
  suffices h2 : ∑ (x : ℕ) in {d : ℕ | d ∣ 300 ∧ d % 2 = 1}.to_finset, x = 124,
  {
    exact h2,
  },
  -- Placeholder for actual proof
  sorry
}

end sum_odd_divisors_300_l595_595114


namespace sin_30_eq_half_l595_595302
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595302


namespace sum_of_divisors_prime_factors_l595_595689

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595689


namespace sin_30_is_half_l595_595327

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595327


namespace laptop_turn_off_time_l595_595178

-- Define the initial conditions
def start_time := Time.mk 15 20 -- 3:20 pm
def movie_duration := 3 * 60 -- movie duration is 180 minutes
def remaining_movie_time := 36 -- 36 minutes remaining

-- Calculate the watched time
def watched_time := movie_duration - remaining_movie_time -- 144 minutes

-- Convert the start time and watched time into final time
def end_time := start_time.add_minutes watched_time

-- Expected end time
def expected_end_time := Time.mk 17 44 -- 5:44 pm

-- Statement to prove
theorem laptop_turn_off_time :
  end_time = expected_end_time :=
sorry

end laptop_turn_off_time_l595_595178


namespace sin_30_eq_half_l595_595480

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595480


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595720

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595720


namespace sin_30_eq_half_l595_595218

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595218


namespace gcd_polynomials_l595_595500

def even_multiple_of_2927 (a : ℤ) : Prop := ∃ k : ℤ, a = 2 * 2927 * k

theorem gcd_polynomials (a : ℤ) (h : even_multiple_of_2927 a) :
  Int.gcd (3 * a ^ 2 + 61 * a + 143) (a + 19) = 7 :=
by
  sorry

end gcd_polynomials_l595_595500


namespace equation_satisfying_solution_l595_595092

theorem equation_satisfying_solution (x y : ℤ) :
  (x = 1 ∧ y = 4 → x + 3 * y ≠ 7) ∧
  (x = 2 ∧ y = 1 → x + 3 * y ≠ 7) ∧
  (x = -2 ∧ y = 3 → x + 3 * y = 7) ∧
  (x = 4 ∧ y = 2 → x + 3 * y ≠ 7) :=
by
  sorry

end equation_satisfying_solution_l595_595092


namespace area_percentage_of_smaller_square_l595_595170

theorem area_percentage_of_smaller_square 
  (radius : ℝ)
  (a A O B: ℝ)
  (side_length_larger_square side_length_smaller_square : ℝ) 
  (hyp1 : side_length_larger_square = 4)
  (hyp2 : radius = 2 * Real.sqrt 2)
  (hyp3 : a = 4) 
  (hyp4 : A = 2 + side_length_smaller_square / 4)
  (hyp5 : O = 2 * Real.sqrt 2)
  (hyp6 : side_length_smaller_square = 0.8) :
  (side_length_smaller_square^2 / side_length_larger_square^2) = 0.04 :=
by
  sorry

end area_percentage_of_smaller_square_l595_595170


namespace geometric_sequence_problem_l595_595814

noncomputable def a₂ (a₁ q : ℝ) : ℝ := a₁ * q
noncomputable def a₃ (a₁ q : ℝ) : ℝ := a₁ * q^2
noncomputable def a₄ (a₁ q : ℝ) : ℝ := a₁ * q^3
noncomputable def S₆ (a₁ q : ℝ) : ℝ := (a₁ * (1 - q^6)) / (1 - q)

theorem geometric_sequence_problem
  (a₁ q : ℝ)
  (h1 : a₁ * a₂ a₁ q * a₃ a₁ q = 27)
  (h2 : a₂ a₁ q + a₄ a₁ q = 30)
  : ((a₁ = 1 ∧ q = 3) ∨ (a₁ = -1 ∧ q = -3))
    ∧ (if a₁ = 1 ∧ q = 3 then S₆ a₁ q = 364 else true)
    ∧ (if a₁ = -1 ∧ q = -3 then S₆ a₁ q = -182 else true) :=
by
  -- Proof goes here
  sorry

end geometric_sequence_problem_l595_595814


namespace sin_30_eq_half_l595_595398

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595398


namespace B_days_to_do_work_alone_l595_595139

variable (W : ℝ) -- The total work
variable (A_rate : ℝ := W / 20) -- A's rate of work
variable (C_rate : ℝ := W / 10) -- C's rate of work
variable (B_days : ℝ)
variable (B_rate : ℝ := W / B_days) -- B's rate of work

-- Equation representing the total work done by A, B, and C
def work_done_by_all : ℝ :=
  A_rate * 2 + C_rate * 4 + B_rate * 15.000000000000002

-- Theorem to prove
theorem B_days_to_do_work_alone (h : work_done_by_all W B_days = W) : B_days = 30 := 
by
  sorry

end B_days_to_do_work_alone_l595_595139


namespace minimum_circumcircle_area_is_5pi_over_4_l595_595581

noncomputable def minimum_circumcircle_area : ℝ :=
  let l : ℝ → ℝ → Prop := λ x y, x - 2 * y + 4 = 0
  let circle_center := (1 : ℝ, 0 : ℝ)
  let circle_radius := 1
  let distance_to_line (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
    abs (fst p * 1 + snd p * (-2) + 4) / real.sqrt (1 ^ 2 + (-2) ^ 2)
  let minimum_distance := distance_to_line circle_center l
  let circumcircle_radius := minimum_distance / 2
  let circumcircle_area := real.pi * circumcircle_radius ^ 2
  circumcircle_area

theorem minimum_circumcircle_area_is_5pi_over_4 : minimum_circumcircle_area = 5 * real.pi / 4 :=
  sorry

end minimum_circumcircle_area_is_5pi_over_4_l595_595581


namespace sin_of_30_degrees_l595_595255

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595255


namespace largest_non_sum_of_42_and_composite_l595_595109

open Nat

def isPrime (n : Nat) : Prop := Prime n

def valid_n (n : Nat) : Prop :=
∃ (a b : Nat), n = 42 * a + b ∧ b < 42 ∧ ∀ k < a, isPrime (b + 42 * k)

theorem largest_non_sum_of_42_and_composite :
  ∃ (n : Nat), valid_n n ∧ ∀ (m : Nat), valid_n m → m ≤ n :=
sorry

end largest_non_sum_of_42_and_composite_l595_595109


namespace f_analytic_expression_f_max_value_l595_595626

noncomputable def a (x m : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, m + cos x)
noncomputable def b (x m : ℝ) : ℝ × ℝ := (cos x, -m + cos x)
noncomputable def f (x m : ℝ) : ℝ := (a x m).1 * (b x m).1 + (a x m).2 * (b x m).2

theorem f_analytic_expression (x m : ℝ) : f x m = sin (2 * x + π / 6) + 1 / 2 - m^2 :=
by sorry

theorem f_max_value (m : ℝ) :
  ∃ x ∈ Icc (-π / 6) (π / 3), f x m = 3 / 2 - m^2 ∧ ∀ y ∈ Icc (-π / 6) (π / 3), f y m ≤ f x m :=
by sorry

end f_analytic_expression_f_max_value_l595_595626


namespace parabola_directrix_l595_595079

theorem parabola_directrix (x y : ℝ) (h : y = x^2) : 4 * y + 1 = 0 := 
sorry

end parabola_directrix_l595_595079


namespace sin_30_eq_half_l595_595397

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595397


namespace sin_30_eq_half_l595_595454

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595454


namespace total_travel_time_l595_595839

variable (d : ℕ) (speed_out speed_back total_distance : ℕ)

theorem total_travel_time :
  speed_out = 24 →
  speed_back = 18 →
  total_distance = 72 →
  2 * d = total_distance →
  d / speed_out + d / speed_back = 3.5 :=
by
  sorry

end total_travel_time_l595_595839


namespace total_pages_read_l595_595063

-- Define the reading rates
def ReneReadingRate : ℕ := 30  -- pages in 60 minutes
def LuluReadingRate : ℕ := 27  -- pages in 60 minutes
def CherryReadingRate : ℕ := 25  -- pages in 60 minutes

-- Total time in minutes
def totalTime : ℕ := 240  -- minutes

-- Define a function to calculate pages read in given time
def pagesRead (rate : ℕ) (time : ℕ) : ℕ :=
  rate * (time / 60)

-- Theorem to prove the total number of pages read
theorem total_pages_read :
  pagesRead ReneReadingRate totalTime +
  pagesRead LuluReadingRate totalTime +
  pagesRead CherryReadingRate totalTime = 328 :=
by
  -- Proof is not required, hence replaced with sorry
  sorry

end total_pages_read_l595_595063


namespace sin_30_eq_half_l595_595432

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595432


namespace recommendation_plans_count_l595_595809

noncomputable theory

-- Conditions of the problem
def recommendation_spots := 5
def russian_spots := 2
def japanese_spots := 2
def spanish_spot := 1

def males := 3
def females := 2

-- Proof problem statement
theorem recommendation_plans_count :
  ∃ (plans : ℕ), plans = 24 ∧
  (recommendation_spots = russian_spots + japanese_spots + spanish_spot) ∧
  (russian_spots ≥ 2) ∧
  (japanese_spots ≥ 2) ∧
  (males = 3) ∧
  ((males + females) = 5) :=
sorry

end recommendation_plans_count_l595_595809


namespace sin_30_is_half_l595_595326

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595326


namespace original_number_is_16_l595_595943

theorem original_number_is_16 (x : ℤ) (h1 : 3 * (2 * x + 5) = 111) : x = 16 :=
by
  sorry

end original_number_is_16_l595_595943


namespace sin_thirty_deg_l595_595229

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595229


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595662

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595662


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595716

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595716


namespace num_distinct_prime_factors_sum_divisors_450_l595_595643

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595643


namespace unoccupied_seats_in_business_class_l595_595168

/-
Define the numbers for each class and the number of people in each.
-/
def first_class_seats : Nat := 10
def business_class_seats : Nat := 30
def economy_class_seats : Nat := 50
def people_in_first_class : Nat := 3
def people_in_economy_class : Nat := economy_class_seats / 2
def people_in_business_and_first_class : Nat := people_in_economy_class
def people_in_business_class : Nat := people_in_business_and_first_class - people_in_first_class

/-
Prove that the number of unoccupied seats in business class is 8.
-/
theorem unoccupied_seats_in_business_class :
  business_class_seats - people_in_business_class = 8 :=
sorry

end unoccupied_seats_in_business_class_l595_595168


namespace num_distinct_prime_factors_sum_divisors_450_l595_595652

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595652


namespace sin_30_eq_half_l595_595212

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595212


namespace sin_30_is_half_l595_595321

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595321


namespace sin_30_eq_half_l595_595433

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595433


namespace remaining_travel_time_approx_l595_595096

noncomputable def total_time : ℚ := 29 / 4 -- 7.25 hours

noncomputable def avg_speed : ℚ := 50 -- miles per hour

noncomputable def first_part_time: ℚ := 2 -- hours

noncomputable def first_part_speed : ℚ := 80 -- miles per hour

noncomputable def remaining_part_speed : ℚ := 40 -- miles per hour

noncomputable def total_distance : ℚ := avg_speed * total_time

noncomputable def first_part_distance : ℚ := first_part_speed * first_part_time

noncomputable def remaining_distance : ℚ := total_distance - first_part_distance

noncomputable def remaining_time : ℚ := remaining_distance / remaining_part_speed

theorem remaining_travel_time_approx :
  Real.floor((remaining_time * 100) + 0.5) / 100 = 5.06 := by
  sorry

end remaining_travel_time_approx_l595_595096


namespace sum_of_elements_of_S_l595_595841

noncomputable def sum_of_S : ℝ :=
∑ (x : ℝ) in {x | ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ x = a * 100 + b * 10 + c} / 999

theorem sum_of_elements_of_S : sum_of_S = 360 := 
sorry

end sum_of_elements_of_S_l595_595841


namespace triangles_from_vertex_A_are_multiplicative_l595_595951

def is_multiplicative_triangle (a b c : ℝ) : Prop :=
  (a * b = c) ∨ (b * c = a) ∨ (c * a = b)

def regular_polygon (vertices : ℕ) (side_length : ℝ) : Prop :=
  vertices ≥ 3 ∧ side_length = 1

theorem triangles_from_vertex_A_are_multiplicative
  (n : ℕ) (h1 : regular_polygon n 1)
  (h2 : n ≥ 4) :
  ∀ (A B C ... X Y Z : ℝ), 
  let diagonals := n - 3
  in 
  ∀ (triangle : fin (n-2) → (ℝ × ℝ × ℝ)), 
  (∀ i, is_multiplicative_triangle (triangle i).1 (triangle i).2 (triangle i).3) := 
sorry

end triangles_from_vertex_A_are_multiplicative_l595_595951


namespace sin_30_eq_half_l595_595421

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595421


namespace sin_30_eq_half_l595_595406

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595406


namespace correct_formula_for_xy_l595_595816

theorem correct_formula_for_xy :
  (∀ x y, (x = 1 ∧ y = 3) ∨ (x = 2 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ (x = 4 ∧ y = 21) ∨ (x = 5 ∧ y = 31) →
    y = x^2 + x + 1) :=
sorry

end correct_formula_for_xy_l595_595816


namespace determine_coefficients_l595_595821

noncomputable def triangle_XYZ (X Y Z : Point) (W : Point) (V : Point)
  (hW_on_YZ : W ∈ line(Y, Z))
  (h1 : (∀ {W Y Z}, W:Y = 4 → Z:W = 1))
  (V : Point) (hV_on_XZ : V ∈ line(X, Z))
  (h2 : (∀ {V X Z}, X:V = 2 → Z:V = 3))
  : Prop :=
∀ (Q : Point), (Q = intersection(line(Y, V), line(X, W)))
  → ∃ (x y z : ℚ), (x + y + z = 1) ∧ (Q = x • X + y • Y + z • Z) 
  ∧ (x = 5 / 3) ∧ (y = (-2 / 15)) ∧ (z = (8 / 15))

theorem determine_coefficients (X Y Z W V Q : Point)
  (hW_on_YZ : W ∈ line(Y, Z))
  (hYW_WZ : (YW:WZ = 4:1))
  (hV_on_XZ : V ∈ line(X, Z))
  (hXV_VZ : (XV:VZ = 2:3))
  : triangle_XYZ X Y Z W V hW_on_YZ hYW_WZ V hV_on_XZ hXV_VZ :=
begin
  intros Q hQ,
  use [5 / 3, -2 / 15, 8 / 15],
  split,
  { ring },
  split,
  { sorry },
  { split; ring },
end

end determine_coefficients_l595_595821


namespace sin_thirty_degree_l595_595284

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595284


namespace sin_30_deg_l595_595347

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595347


namespace pipe_A_rate_l595_595058

theorem pipe_A_rate (A : ℝ) : 
  (∃ (rateA : ℝ),
    -- Conditions
    let capacity := 950 in
    let rateB := 30 in
    let rateC := 20 in
    let cycle_duration := 3 in
    let total_duration := 57 in
    -- Equation from the problem has the tank full in 57 minutes
    let total_cycles := total_duration / cycle_duration in
    let net_addition_per_cycle := A + rateB - rateC in
    let total_addition := total_cycles * net_addition_per_cycle in
    total_addition = capacity ∧
    -- Question and answer
    rateA = A) ↔ A = 40 :=
begin
  sorry -- Proof is to be constructed here
end

end pipe_A_rate_l595_595058


namespace distinct_prime_factors_of_sigma_450_l595_595677
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595677


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595703

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595703


namespace sin_30_eq_half_l595_595420

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595420


namespace geraldine_more_than_jazmin_l595_595557

def geraldine_dolls : ℝ := 2186.0
def jazmin_dolls : ℝ := 1209.0
def difference_dolls : ℝ := 977.0

theorem geraldine_more_than_jazmin : geraldine_dolls - jazmin_dolls = difference_dolls :=
by sorry

end geraldine_more_than_jazmin_l595_595557


namespace exists_integer_m_l595_595088

noncomputable def W (x a b : ℤ) : ℤ := x^2 + a * x + b

theorem exists_integer_m (a b : ℤ) 
  (h : ∀ p : ℕ, nat.prime p → ∃ k : ℤ, W k a b ≡ 0 [ZMOD p] ∧ W (k + 1) a b ≡ 0 [ZMOD p]) :
  ∃ m : ℤ, W m a b = 0 ∧ W (m + 1) a b = 0 :=
sorry

end exists_integer_m_l595_595088


namespace sequence_less_than_inverse_l595_595045

-- Define the sequence and conditions given in the problem
variables {a : ℕ → ℝ}
axiom positive_sequence (n : ℕ) : 0 < a n
axiom sequence_inequality (n : ℕ) : a n ^ 2 ≤ a n - a (n + 1)

theorem sequence_less_than_inverse (n : ℕ) : a n < 1 / n := 
sorry

end sequence_less_than_inverse_l595_595045


namespace sum_of_divisors_prime_factors_450_l595_595733

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595733


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595701

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595701


namespace passing_or_failing_outcomes_l595_595957

  theorem passing_or_failing_outcomes (n : ℕ) : ∃ k : ℕ, k = 2^n :=
  by
    use 2^n
    sorry
  
end passing_or_failing_outcomes_l595_595957


namespace sin_30_eq_half_l595_595361

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595361


namespace sin_of_30_degrees_l595_595262

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595262


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595665

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595665


namespace sin_thirty_degrees_l595_595376

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595376


namespace satisfactory_grades_fraction_l595_595156

def total_satisfactory_students (gA gB gC gD gE : Nat) : Nat :=
  gA + gB + gC + gD + gE

def total_students (gA gB gC gD gE gF : Nat) : Nat :=
  total_satisfactory_students gA gB gC gD gE + gF

def satisfactory_fraction (gA gB gC gD gE gF : Nat) : Rat :=
  total_satisfactory_students gA gB gC gD gE / total_students gA gB gC gD gE gF

theorem satisfactory_grades_fraction :
  satisfactory_fraction 3 5 4 2 1 4 = (15 : Rat) / 19 :=
by
  sorry

end satisfactory_grades_fraction_l595_595156


namespace difference_between_max_and_min_iterative_averages_l595_595187

-- We define the input list of numbers.
def nums : List ℚ := [2, 4, 6, 8, 10]

-- Define a function to compute the iterative average of a given list of numbers.
def iterative_average : List ℚ → ℚ
| [] => 0
| [x] => x
| (x :: y :: xs) => iterative_average ((x + y) / 2 :: xs)

-- Define the problem statement.
theorem difference_between_max_and_min_iterative_averages :
  let max_avg := max ((List.permutations nums).map iterative_average)
  let min_avg := min ((List.permutations nums).map iterative_average)
  max_avg - min_avg = 4.25 := sorry

end difference_between_max_and_min_iterative_averages_l595_595187


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595629

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595629


namespace course_program_count_l595_595172

theorem course_program_count : 
  let courses := ["English", "Algebra", "Geometry", "History", "Art", "Latin", "Biology", "Chemistry"]
  let required_courses := ["English"]
  let math_courses := ["Algebra", "Geometry"]
  let science_courses := ["Biology", "Chemistry"]
  (∃ (s : Finset (String)) (h₁ : "English" ∈ s) (h₂ : ∃ (m : String), m ∈ math_courses ∧ m ∈ s) (h₃ : ∃ (sc : String), sc ∈ science_courses ∧ sc ∈ s), 
    s.card = 5 ∧ (s ⊆ courses.toFinset)) := 25 :=
by
  sorry

end course_program_count_l595_595172


namespace sin_30_eq_half_l595_595436

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595436


namespace triangle_sides_possible_k_l595_595614

noncomputable def f (x k : ℝ) : ℝ := x^2 - 4*x + 4 + k^2

theorem triangle_sides_possible_k (a b c k : ℝ) (ha : 0 ≤ a) (hb : a ≤ 3) (ha' : 0 ≤ b) (hb' : b ≤ 3) (ha'' : 0 ≤ c) (hb'' : c ≤ 3) :
  (f a k + f b k > f c k) ∧ (f a k + f c k > f b k) ∧ (f b k + f c k > f a k) ↔ k = 3 ∨ k = 4 :=
by
  sorry

end triangle_sides_possible_k_l595_595614


namespace sin_30_eq_half_l595_595412

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595412


namespace find_natural_numbers_l595_595544

theorem find_natural_numbers (n : ℕ) (h : n > 1) : 
  ((n - 1) ∣ (n^3 - 3)) ↔ (n = 2 ∨ n = 3) := 
by 
  sorry

end find_natural_numbers_l595_595544


namespace sin_thirty_degree_l595_595270

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595270


namespace dog_distance_in_2_hours_dog_time_to_run_50_km_l595_595982

-- Defining the conditions
def speed (d t : ℝ) : ℝ := d / t  -- speed = distance / time
def constant_speed := speed 89.6 4  -- The speed of the dog is 89.6 km / 4 hours

-- Problem (1): Distance in 2 hours
theorem dog_distance_in_2_hours : speed 44.8 2 = constant_speed := by
  sorry

-- Problem (2): Time taken to run 50 kilometers
theorem dog_time_to_run_50_km : (round (speed 50 constant_speed * 10) / 10) = 2.2 := by
  sorry

end dog_distance_in_2_hours_dog_time_to_run_50_km_l595_595982


namespace triangle_new_position_after_two_rotations_l595_595153

-- Mathematical definitions and conditions
def regular_pentagon_interior_angle : ℝ := (5 - 2) * 180 / 5
def rotation_per_movement (rect_angle: ℝ, pent_angle: ℝ) : ℝ := 360 - (rect_angle + pent_angle)
def total_rotation (movement_rotation: ℝ) (num_movements: ℕ) : ℝ := num_movements * movement_rotation
def final_rotation (complete_rotation: ℝ) : ℝ := complete_rotation % 360

-- Given conditions
def initial_triangle_position : ℕ := 0 -- assuming 0 for "Top Left", 1 for "Top Right", 2 for "Bottom Right", 3 for "Bottom Left"
def movements_in_full_rotation : ℕ := 5
def full_rotation_degrees : ℝ := 360
def rectangle_angle_rotation : ℝ := 90 -- internal angle of a rectangle corner

-- Let's prove the position after completing two full rotations
theorem triangle_new_position_after_two_rotations : initial_triangle_position = 0 →
  final_rotation (total_rotation (rotation_per_movement rectangle_angle_rotation regular_pentagon_interior_angle) (2 * movements_in_full_rotation)) = 180 →
  1 :=
by
  intro h1 h2
  exact sorry

end triangle_new_position_after_two_rotations_l595_595153


namespace statement1_statement2_statement3_l595_595800

-- Definition of the character types
inductive Character
| knight : Character
| liar : Character

-- Predicate indicating if a character always speaks the truth or always lies
def tells_truth : Character -> Prop
| Character.knight := true
| Character.liar := false

-- The existence of the two types of characters in the group
axiom two_knights_two_liars (c1 c2 c3 c4 : Character) : 
  (tells_truth c1 ∧ tells_truth c2 ∧ ¬tells_truth c3 ∧ ¬tells_truth c4) ∨ 
  (tells_truth c1 ∧ tells_truth c3 ∧ ¬tells_truth c2 ∧ ¬tells_truth c4) ∨ 
  (tells_truth c1 ∧ tells_truth c4 ∧ ¬tells_truth c2 ∧ ¬tells_truth c3) ∨ 
  (tells_truth c2 ∧ tells_truth c3 ∧ ¬tells_truth c1 ∧ ¬tells_truth c4) ∨ 
  (tells_truth c2 ∧ tells_truth c4 ∧ ¬tells_truth c1 ∧ ¬tells_truth c3) ∨ 
  (tells_truth c3 ∧ tells_truth c4 ∧ ¬tells_truth c1 ∧ ¬tells_truth c2)

-- Problem statements
-- 1. Prove that if a person says "Among us, all are knights", then the person is a liar
theorem statement1 (p : Character) : (∀ x, p = x → tells_truth x) ↔ ¬tells_truth p := sorry

-- 2. Prove that if a person says "Among you, there is exactly one knight", then the person is a knight
theorem statement2 (p : Character) (q1 q2 q3 : Character) : 
  tells_truth p → ¬tells_truth q1 → (q2 = q1 ∧ tells_truth q3) ∨ (tells_truth q2 ∧ tells_truth q3 → tells_truth q1) := sorry

-- 3. Prove that neither a knight nor a liar can say "Among you, there are exactly two knights"
theorem statement3 (p : Character) (q1 q2 q3 : Character) :
  tells_truth p ∧ tells_truth q1 ∧ tells_truth q2 -> ¬(∀ r, r = q3 → tells_truth r) := sorry

end statement1_statement2_statement3_l595_595800


namespace dow_jones_morning_value_l595_595068

theorem dow_jones_morning_value 
  (end_of_day_value : ℝ) 
  (percentage_fall : ℝ)
  (expected_morning_value : ℝ) 
  (h1 : end_of_day_value = 8722) 
  (h2 : percentage_fall = 0.02) 
  (h3 : expected_morning_value = 8900) :
  expected_morning_value = end_of_day_value / (1 - percentage_fall) :=
sorry

end dow_jones_morning_value_l595_595068


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595723

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595723


namespace cost_to_marked_price_ratio_l595_595150

theorem cost_to_marked_price_ratio (p : ℚ) :
  let selling_price := (3 / 4 : ℚ) * p,
      cost_price := (5 / 8 : ℚ) * selling_price in
  cost_price / p = 15 / 32 :=
by
  let selling_price := (3 / 4 : ℚ) * p
  let cost_price := (5 / 8 : ℚ) * selling_price
  sorry

end cost_to_marked_price_ratio_l595_595150


namespace sum_of_divisors_450_has_three_distinct_prime_factors_l595_595635

def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := if n = 450 then [(2, 1), (3, 2), (5, 2)] else [];
  factors.foldl (λ acc (p, e), acc * (Finset.range (e + 1)).sum (λ k, p ^ k)) 1

noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors (sum_of_divisors n)).toFinset.card

theorem sum_of_divisors_450_has_three_distinct_prime_factors :
  num_distinct_prime_factors 450 = 3 := by
    sorry

end sum_of_divisors_450_has_three_distinct_prime_factors_l595_595635


namespace sin_30_eq_half_l595_595470

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595470


namespace unoccupied_business_class_seats_l595_595160

theorem unoccupied_business_class_seats :
  ∀ (first_class_seats business_class_seats economy_class_seats economy_fullness : ℕ)
  (first_class_occupancy : ℕ) (total_business_first_combined economy_occupancy : ℕ),
  first_class_seats = 10 →
  business_class_seats = 30 →
  economy_class_seats = 50 →
  economy_fullness = economy_class_seats / 2 →
  economy_occupancy = economy_fullness →
  total_business_first_combined = economy_occupancy →
  first_class_occupancy = 3 →
  total_business_first_combined = first_class_occupancy + business_class_seats - (business_class_seats - total_business_first_combined + first_class_occupancy) →
  business_class_seats - (total_business_first_combined - first_class_occupancy) = 8 :=
by
  intros first_class_seats business_class_seats economy_class_seats economy_fullness 
         first_class_occupancy total_business_first_combined economy_occupancy
         h1 h2 h3 h4 h5 h6 h7 h8 
  rw [h1, h2, h3, h4, h5, h6, h7] at h8
  sorry

end unoccupied_business_class_seats_l595_595160


namespace initial_mean_calculated_l595_595086

-- Definitions of the conditions
def num_values : ℕ := 30
def incorrect_value : ℕ := 135
def correct_value : ℕ := 165
def correct_mean : ℚ := 251

-- Theorem statement to prove the initial mean
theorem initial_mean_calculated :
  (let incorrect_mean := (num_values * correct_mean - (correct_value - incorrect_value)) / num_values in
  incorrect_mean = 250) :=
by
  sorry -- The proof steps are omitted

end initial_mean_calculated_l595_595086


namespace sin_30_deg_l595_595352

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595352


namespace sum_of_divisors_prime_factors_450_l595_595738

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595738


namespace constant_term_in_polynomial_expansion_eq_18_l595_595499

theorem constant_term_in_polynomial_expansion_eq_18 :
  let poly := (4 * (x^2) - 2) * (1 + (1 / (x^2))) ^ 5
  in constant_term poly = 18 :=
by
  sorry

end constant_term_in_polynomial_expansion_eq_18_l595_595499


namespace sum_of_midpoint_coordinates_l595_595197

def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

theorem sum_of_midpoint_coordinates : 
  let midpoint_coords := midpoint 8 (-2) 2 10 in
  midpoint_coords.1 + midpoint_coords.2 = 9 :=
by
  sorry

end sum_of_midpoint_coordinates_l595_595197


namespace actual_time_when_watch_reads_10pm_l595_595199

-- Definitions of the problem conditions
def watch_loss_rate := 57.6 / 60  -- Watch time per actual hour

def watch_reading_10pm := 600  -- Minutes from noon at watch reading 10:00 PM

-- Mathematical Proof Statement
theorem actual_time_when_watch_reads_10pm :
  ∃ actual_time_when_watch_reads_10pm, 
  actual_time_when_watch_reads_10pm = watch_reading_10pm / watch_loss_rate := 
sorry

end actual_time_when_watch_reads_10pm_l595_595199


namespace sin_30_eq_one_half_l595_595313

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l595_595313


namespace sin_30_eq_half_l595_595213

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595213


namespace triangle_problems_l595_595797

variables {A B C : ℝ} {a b c : ℝ} {l : ℝ}
noncomputable def cos := sorry -- Assume existence of cos function for real numbers
noncomputable def sin := sorry -- Assume existence of sin function for real numbers

open_locale real

theorem triangle_problems (A B C a b c : ℝ)
  (h₁ : b * cos C = a - 1/2 * c)
  (h₂ : b = 1):
  B = π/3 ∧ (l = a + b + c → (2 < l ∧ l ≤ 3)) :=
begin
  -- Skipping the proof and adding sorry
  sorry,
end

end triangle_problems_l595_595797


namespace sin_30_eq_half_l595_595456

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595456


namespace distinct_prime_factors_of_sigma_450_l595_595678
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595678


namespace sum_of_divisors_prime_factors_l595_595693

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595693


namespace sin_30_eq_half_l595_595477

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595477


namespace AK_eq_CK_l595_595131

variable {α : Type*} [LinearOrder α] [LinearOrder ℝ]

variable (A B C L K : ℝ)
variable (triangle : ℝ)
variable (h₁ : AL = LB)
variable (h₂ : AK = CL)

--  Given that in triangle ABC,
--     AL is a bisector such that AL = LB,
--     and AK is on ray AL with AK = CL,
--     prove that AK = CK.
theorem AK_eq_CK (h₁ : AL = LB) (h₂ : AK = CL) : AK = CK := by
  sorry

end AK_eq_CK_l595_595131


namespace option_a_incorrect_option_b_correct_option_c_correct_option_d_correct_l595_595938

-- Define the conditions and theorems to be proven
theorem option_a_incorrect (n p : ℝ) (x : ℝ) (hx : x ≈ B(n, p)) (H1 : E(x) = 30) (H2 : D(x) = 20) : 
  p ≠ 2 / 3 := sorry

theorem option_b_correct (a : ℝ) (n : ℕ) (H1 : binomial_coefficient (9, 2) = binomial_coefficient (9, 7))
  (H2 : constant_term ((a * x + 1 / sqrt(x))^9) = 84) :
  a = 1 := sorry

theorem option_c_correct (ξ : ℝ → Prop) (H : ξ ~ N(0,1)) (H1 : P(ξ > 1) = p) :
  P(-1 < ξ ∧ ξ < 0) = 1/2 - p := sorry

theorem option_d_correct :
  ∃ (arrangements_count : ℕ), arrangements_count = 48 := 
sorry

end option_a_incorrect_option_b_correct_option_c_correct_option_d_correct_l595_595938


namespace sin_30_eq_half_l595_595299
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595299


namespace tetrahedron_dihedral_sum_gt_360_l595_595876

theorem tetrahedron_dihedral_sum_gt_360 (T : Tetrahedron) : 
  (Σ A B C, DihedralAngle A B C T) > 360 := 
sorry

end tetrahedron_dihedral_sum_gt_360_l595_595876


namespace largest_common_remainder_l595_595983

theorem largest_common_remainder : 
  ∃ n r, 2013 ≤ n ∧ n ≤ 2156 ∧ (n % 5 = r) ∧ (n % 11 = r) ∧ (n % 13 = r) ∧ (r = 4) := 
by
  sorry

end largest_common_remainder_l595_595983


namespace distinct_prime_factors_of_sigma_450_l595_595772

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595772


namespace sin_30_eq_half_l595_595466

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595466


namespace sin_thirty_degrees_l595_595378

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595378


namespace sin_thirty_degree_l595_595283

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595283


namespace floor_plus_x_eq_17_over_4_l595_595519

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l595_595519


namespace abs_sum_l595_595789

theorem abs_sum (a b c : ℚ) (h₁ : a = -1/4) (h₂ : b = -2) (h₃ : c = -11/4) :
  |a| + |b| - |c| = -1/2 :=
by {
  sorry
}

end abs_sum_l595_595789


namespace find_b_value_l595_595000

-- Definitions based on given conditions
def original_line (x : ℝ) (b : ℝ) : ℝ := 2 * x + b
def shifted_line (x : ℝ) (b : ℝ) : ℝ := 2 * (x - 2) + b
def passes_through_origin (b : ℝ) := shifted_line 0 b = 0

-- Main proof statement
theorem find_b_value (b : ℝ) (h : passes_through_origin b) : b = 4 := by
  sorry

end find_b_value_l595_595000


namespace distinct_prime_factors_of_sigma_450_l595_595683
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595683


namespace sqrt_expression_value_l595_595552

theorem sqrt_expression_value (y : ℝ) (hy : y = 0.81) :
  (sqrt 1.21 / sqrt y) + (sqrt 1.00 / sqrt 0.49) = 2.650793650793651 :=
by {
  rw hy,
  norm_num
}

end sqrt_expression_value_l595_595552


namespace sin_thirty_deg_l595_595232

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595232


namespace sin_thirty_degree_l595_595275

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595275


namespace smallest_three_digit_multiple_of_9_with_digit_sum_27_l595_595113

def digits_sum (n : ℕ) : ℕ := n.digits.sum

theorem smallest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n % 9 = 0) ∧ (digits_sum n = 27) ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (m % 9 = 0) ∧ (digits_sum m = 27) → n ≤ m :=
sorry

end smallest_three_digit_multiple_of_9_with_digit_sum_27_l595_595113


namespace sequence_term_number_l595_595006

theorem sequence_term_number (n : ℕ) (a_n : ℕ) (h : a_n = 2 * n ^ 2 - 3) : a_n = 125 → n = 8 :=
by
  sorry

end sequence_term_number_l595_595006


namespace inscribed_hexagon_area_l595_595929

theorem inscribed_hexagon_area (r : ℝ) (h : π * r^2 = 16 * π) : 
  let side := r in
  let triangle_area := (side^2 * Real.sqrt 3) / 4 in
  let hexagon_area := 6 * triangle_area in
  hexagon_area = 24 * Real.sqrt 3 := 
by
  have r_squared_eq : r^2 = 16 := by sorry
  have side_def : side = r := by sorry
  have tri_area_def : triangle_area = (side^2 * Real.sqrt 3) / 4 := by sorry
  have hex_area_def : hexagon_area = 6 * triangle_area := by sorry
  have step1 : r = 4 := by sorry
  have step2 : side = 4 := by sorry
  have step3 : triangle_area = 4 * Real.sqrt 3 := by sorry
  have step4 : hexagon_area = 24 * Real.sqrt 3 := by sorry
  exact step4

end inscribed_hexagon_area_l595_595929


namespace profit_function_correct_l595_595965

-- Definitions based on Conditions
def selling_price {R : Type*} [LinearOrderedField R] : R := 45
def profit_max {R : Type*} [LinearOrderedField R] : R := 450
def price_no_sales {R : Type*} [LinearOrderedField R] : R := 60
def quadratic_profit {R : Type*} [LinearOrderedField R] (x : R) : R := -2 * (x - 30) * (x - 60)

-- The statement we need to prove.
theorem profit_function_correct {R : Type*} [LinearOrderedField R] :
  quadratic_profit (selling_price : R) = profit_max ∧ quadratic_profit (price_no_sales : R) = 0 := 
sorry

end profit_function_correct_l595_595965


namespace right_angled_triangle_l595_595937

theorem right_angled_triangle (a b c : ℕ) (h₀ : a = 7) (h₁ : b = 9) (h₂ : c = 13) :
  a^2 + b^2 ≠ c^2 :=
by
  sorry

end right_angled_triangle_l595_595937


namespace sin_30_eq_half_l595_595368

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595368


namespace sin_thirty_degrees_l595_595383

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595383


namespace sin_30_eq_half_l595_595434

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595434


namespace matrix_exponentiation_l595_595579

theorem matrix_exponentiation (b m : ℕ) :
  let A := ![
    ![1, 3, b],
    ![0, 1, 5],
    ![0, 0, 1]
  ],
  result := ![
    ![1, 27, 3050],
    ![0, 1, 45],
    ![0, 0, 1]
  ] in
  (A ^ m = result) → (b + m = 287) :=
by
  intros A result h
  sorry

end matrix_exponentiation_l595_595579


namespace sin_thirty_degree_l595_595273

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595273


namespace wire_cut_problem_l595_595126

variable (x : ℝ)

theorem wire_cut_problem 
  (h₁ : x + (5 / 2) * x = 49) : x = 14 :=
by
  sorry

end wire_cut_problem_l595_595126


namespace num_distinct_prime_factors_sum_divisors_450_l595_595649

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595649


namespace intersection_point_l595_595840

noncomputable theory

def f (x : ℝ) := (x^2 - 4*x + 4) / (x - 2)
def g (x : ℝ) : ℝ := (-x^2 + 5*x + 7) / (x - 2)

theorem intersection_point :
  ∃ x y : ℝ, f x = y ∧ g x = y ∧ x ≠ 3 ∧ x = (9 - Real.sqrt 105) / 4 :=
by
  sorry

end intersection_point_l595_595840


namespace prob_Allison_wins_l595_595182

open ProbabilityTheory

def outcome_space (n : ℕ) := {k : ℕ // k < n}

def Brian_roll : outcome_space 6 := sorry -- Representing Brian's possible outcomes: 1, 2, 3, 4, 5, 6
def Noah_roll : outcome_space 2 := sorry -- Representing Noah's possible outcomes: 3 or 5 (3 is 0 and 5 is 1 in the index for simplicity)

noncomputable def prob_Brian_less_than_4 := (|{k : ℕ // k < 3}| : ℝ) / (|{k : ℕ // k < 6}|) -- Probability Brian rolls 1, 2, or 3
noncomputable def prob_Noah_less_than_4 := (|{k : ℕ // k = 0}| : ℝ) / (|{k : ℕ // k < 2}|)  -- Probability Noah rolls 3 (index 0)

theorem prob_Allison_wins : (prob_Brian_less_than_4 * prob_Noah_less_than_4) = 1 / 4 := by
  sorry

end prob_Allison_wins_l595_595182


namespace shirts_per_pants_l595_595047

-- Definitions based on conditions
def num_pants : ℕ := 40
def total_clothes : ℕ := 280

-- The theorem to prove
theorem shirts_per_pants (S : ℕ) (h : 40 * S + 40 = 280) : S = 6 :=
by {
  sorry,
}

end shirts_per_pants_l595_595047


namespace sin_30_eq_half_l595_595401

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595401


namespace area_of_given_triangle_l595_595995

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem area_of_given_triangle :
  area_of_triangle (-2) 3 7 (-3) 4 6 = 31.5 :=
by
  sorry

end area_of_given_triangle_l595_595995


namespace sin_thirty_deg_l595_595225

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595225


namespace sin_30_deg_l595_595345

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595345


namespace total_respondents_l595_595944

theorem total_respondents (X Y : ℕ) (h1 : X = 60) (h2 : 3 * Y = X) : X + Y = 80 :=
by
  sorry

end total_respondents_l595_595944


namespace coin_flips_probability_equal_heads_l595_595014

def fair_coin (p : ℚ) := p = 1 / 2
def second_coin (p : ℚ) := p = 3 / 5
def third_coin (p : ℚ) := p = 2 / 3

theorem coin_flips_probability_equal_heads :
  ∀ p1 p2 p3, fair_coin p1 → second_coin p2 → third_coin p3 →
  ∃ m n, m + n = 119 ∧ m / n = 29 / 90 :=
by
  sorry

end coin_flips_probability_equal_heads_l595_595014


namespace solution_is_correct_l595_595886

theorem solution_is_correct : ∃ (a b c d : ℕ), 4^a * 5^b - 3^c * 11^d = 1 ∧ a = 1 ∧ b = 2 ∧ c = 2 ∧ d = 1 :=
by
  use (1, 2, 2, 1)
  split
  -- First part of the proof: show 4^1 * 5^2 - 3^2 * 11^1 = 1
  calc
    4^1 * 5^2 - 3^2 * 11^1 = 4 * 25 - 9 * 11 : by rfl
    ... = 100 - 99 : by norm_num
    ... = 1 : by norm_num
  -- Second part of the proof: show a = 1, b = 2, c = 2, and d = 1
  split; rfl

end solution_is_correct_l595_595886


namespace tangent_line_at_zero_monotonicity_of_func_l595_595081

def func (x a : ℝ) : ℝ := x^3 - a * x - 1

theorem tangent_line_at_zero (a : ℝ) (h : a = 8) :
  ∃ m b, (m = 8) ∧ (b = -1) ∧ (∀ x y, y = func x 8 → y + 1 = m * x) :=
by 
  sorry

theorem monotonicity_of_func (a : ℝ) :
  (a ≤ 0 → ∀ x y, func x a ≤ func y a → x ≤ y) ∧ 
  (a > 0 → (∀ x, x < -sqrt (a / 3) ∨ x > sqrt (a / 3) → ∀ y, func x a < func y a → x < y) ∧
          (∀ x, -sqrt (a / 3) < x ∧ x < sqrt (a / 3) → ∀ y, func x a > func y a → x > y)) :=
by 
  sorry

end tangent_line_at_zero_monotonicity_of_func_l595_595081


namespace six_by_six_board_partition_l595_595133

theorem six_by_six_board_partition (P : Prop) (Q : Prop) 
(board : ℕ × ℕ) (domino : ℕ × ℕ) 
(h1 : board = (6, 6)) 
(h2 : domino = (2, 1)) 
(h3 : P → Q ∧ Q → P) :
  ∃ R₁ R₂ : ℕ × ℕ, (R₁ = (p, q) ∧ R₂ = (r, s) ∧ ((R₁.1 * R₁.2 + R₂.1 * R₂.2) = 36)) :=
sorry

end six_by_six_board_partition_l595_595133


namespace sum_G_inverse_l595_595844

def G : ℕ → ℚ
| 0       := 1
| 1       := 4 / 3
| n + 2   := 3 * G (n + 1) - 2 * G n

theorem sum_G_inverse : (∑' n : ℕ, 1 / G (3 ^ n)) = 1 := 
by 
  sorry

end sum_G_inverse_l595_595844


namespace irrational_product_rational_l595_595123

-- Definitions of irrational and rational for clarity
def irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q
def rational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Statement of the problem in Lean 4
theorem irrational_product_rational (a b : ℕ) (ha : irrational (Real.sqrt a)) (hb : irrational (Real.sqrt b)) :
  rational ((Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b)) :=
by
  sorry

end irrational_product_rational_l595_595123


namespace sum_of_divisors_prime_factors_l595_595692

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595692


namespace find_x_l595_595539

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l595_595539


namespace letters_written_l595_595833

theorem letters_written (nathan_rate : ℕ) (jacob_rate : ℕ) (combined_rate : ℕ) (hours : ℕ) :
  nathan_rate = 25 →
  jacob_rate = 2 * nathan_rate →
  combined_rate = nathan_rate + jacob_rate →
  hours = 10 →
  combined_rate * hours = 750 :=
by
  intros
  sorry

end letters_written_l595_595833


namespace max_S_family_size_l595_595023

open Set

variable {α : Type} [Fintype α]

def is_S_family (R : Finset α) (F : Finset (Finset α)) : Prop :=
(∀ X Y ∈ F, ¬ (X ⊆ Y) ∨ X = Y) ∧
(∀ X Y Z ∈ F, X ∪ Y ∪ Z ≠ R) ∧
(⋃₀ F = R)

theorem max_S_family_size {α : Type} [Fintype α] (R : Finset α) (hR: R.card = 6) :
  ∃ F : Finset (Finset α), is_S_family R F ∧ F.card = 3 :=
sorry

end max_S_family_size_l595_595023


namespace unoccupied_seats_in_business_class_l595_595165

def airplane_seating (fc bc ec : ℕ) (pfc pbc pec : ℕ) : Nat :=
  let num_ec := ec / 2
  let num_bc := num_ec - pfc
  bc - num_bc

theorem unoccupied_seats_in_business_class :
  airplane_seating 10 30 50 3 (50/2) (50/2) = 8 := by
    sorry

end unoccupied_seats_in_business_class_l595_595165


namespace total_value_l595_595129

/-- 
The total value of the item V can be determined based on the given conditions.
- The merchant paid an import tax of $109.90.
- The tax rate is 7%.
- The tax is only on the portion of the value above $1000.

Given these conditions, prove that the total value V is 2567.
-/
theorem total_value {V : ℝ} (h1 : 0.07 * (V - 1000) = 109.90) : V = 2567 :=
by
  sorry

end total_value_l595_595129


namespace sin_thirty_degree_l595_595277

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l595_595277


namespace least_integer_remainder_l595_595111

theorem least_integer_remainder (n : ℕ) 
  (h₁ : n > 1)
  (h₂ : n % 5 = 2)
  (h₃ : n % 6 = 2)
  (h₄ : n % 7 = 2)
  (h₅ : n % 8 = 2)
  (h₆ : n % 10 = 2): 
  n = 842 := 
by
  sorry

end least_integer_remainder_l595_595111


namespace max_elements_of_valid_set_l595_595022

def valid_set (M : Finset ℤ) : Prop :=
  ∀ (a b c : ℤ), a ∈ M → b ∈ M → c ∈ M → (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (a + b ∈ M ∨ a + c ∈ M ∨ b + c ∈ M)

theorem max_elements_of_valid_set (M : Finset ℤ) (h : valid_set M) : M.card ≤ 7 :=
sorry

end max_elements_of_valid_set_l595_595022


namespace sin_of_30_degrees_l595_595266

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595266


namespace cost_of_two_books_and_one_magazine_l595_595076

-- Definitions of the conditions
def condition1 (x y : ℝ) : Prop := 3 * x + 2 * y = 18.40
def condition2 (x y : ℝ) : Prop := 2 * x + 3 * y = 17.60

-- Proof problem
theorem cost_of_two_books_and_one_magazine (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  2 * x + y = 11.20 :=
sorry

end cost_of_two_books_and_one_magazine_l595_595076


namespace total_pages_l595_595062

def reading_rate (pages : ℕ) (minutes : ℕ) : ℝ :=
  pages / minutes

def total_pages_read (rate : ℝ) (minutes : ℕ) : ℝ :=
  rate * minutes

theorem total_pages (t : ℕ) (rene_pages : ℕ) (rene_minutes : ℕ) (lulu_pages : ℕ) (lulu_minutes : ℕ) (cherry_pages : ℕ) (cherry_minutes : ℕ) :
  t = 240 →
  rene_pages = 30 →
  rene_minutes = 60 →
  lulu_pages = 27 →
  lulu_minutes = 60 →
  cherry_pages = 25 →
  cherry_minutes = 60 →
  total_pages_read (reading_rate rene_pages rene_minutes) t +
  total_pages_read (reading_rate lulu_pages lulu_minutes) t +
  total_pages_read (reading_rate cherry_pages cherry_minutes) t = 328 :=
by
  intros t_val rene_p_val rene_m_val lulu_p_val lulu_m_val cherry_p_val cherry_m_val
  rw [t_val, rene_p_val, rene_m_val, lulu_p_val, lulu_m_val, cherry_p_val, cherry_m_val]
  simp [reading_rate, total_pages_read]
  norm_num
  sorry

end total_pages_l595_595062


namespace cotangent_half_angle_relationship_l595_595908

theorem cotangent_half_angle_relationship (a d : ℝ) (h : a > 0) (h_arith : a > d) :
  let b := a + d in
  let c := a + 2 * d in
  let s := (a + b + c) / 2 in
  let cot_B_half := Real.cot (Real.arcsin (a / (2 * (s - b)))) in
  let cot_A_half := Real.cot (Real.arcsin (b / (2 * (s - c)))) in
  cot_B_half * cot_A_half = 3 := 
sorry

end cotangent_half_angle_relationship_l595_595908


namespace sin_30_is_half_l595_595335

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595335


namespace lionel_initial_boxes_crackers_l595_595046

/--
Lionel went to the grocery store and bought some boxes of Graham crackers and 15 packets of Oreos. 
To make an Oreo cheesecake, Lionel needs 2 boxes of Graham crackers and 3 packets of Oreos. 
After making the maximum number of Oreo cheesecakes he can with the ingredients he bought, 
he had 4 boxes of Graham crackers left over. 

The number of boxes of Graham crackers Lionel initially bought is 14.
-/
theorem lionel_initial_boxes_crackers (G : ℕ) (h1 : G - 4 = 10) : G = 14 := 
by sorry

end lionel_initial_boxes_crackers_l595_595046


namespace person_a_catch_up_person_b_5_times_l595_595101

theorem person_a_catch_up_person_b_5_times :
  ∀ (num_flags laps_a laps_b : ℕ),
  num_flags = 2015 →
  laps_a = 23 →
  laps_b = 13 →
  (∃ t : ℕ, ∃ n : ℕ, 10 * t = num_flags * n ∧
             23 * t / 10 = k * num_flags ∧
             n % 2 = 0) →
  n = 10 →
  10 / (2 * 1) = 5 :=
by sorry

end person_a_catch_up_person_b_5_times_l595_595101


namespace g_lower_bound_l595_595563

def f (n : ℕ) : ℕ := 2 * n + 1
def g : ℕ → ℕ

axiom g_one : g 1 = 3
axiom g_condition : ∀ n ≥ 2, g n ≥ f (g (n - 1))

theorem g_lower_bound (n : ℕ) : g n ≥ 3 * 2^(n-1) - 1 :=
by
  sorry

end g_lower_bound_l595_595563


namespace stratified_sampling_correct_l595_595973

theorem stratified_sampling_correct (total_students first_grade second_grade third_grade sample_size : ℕ)
    (h_total: total_students = 900)
    (h_first: first_grade = 300)
    (h_second: second_grade = 200)
    (h_third: third_grade = 400)
    (h_sample_size: sample_size = 45) :
  let sampling_fraction := sample_size.to_rat / total_students.to_rat in
  let sample_first := (first_grade.to_rat * sampling_fraction).to_nat in
  let sample_second := (second_grade.to_rat * sampling_fraction).to_nat in
  let sample_third := (third_grade.to_rat * sampling_fraction).to_nat in
  sample_first = 15 ∧ sample_second = 10 ∧ sample_third = 20 :=
by
  sorry

end stratified_sampling_correct_l595_595973


namespace distinct_prime_factors_of_sigma_450_l595_595770

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595770


namespace sin_30_eq_half_l595_595458

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595458


namespace ratio_of_segments_l595_595060

theorem ratio_of_segments (E F G H : ℝ) (h_collinear : E < F ∧ F < G ∧ G < H)
  (hEF : F - E = 3) (hFG : G - F = 6) (hEH : H - E = 20) : (G - E) / (H - F) = 9 / 17 := by
  sorry

end ratio_of_segments_l595_595060


namespace sum_of_divisors_prime_factors_l595_595688

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595688


namespace sin_30_eq_half_l595_595490

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595490


namespace shirley_boxes_to_cases_l595_595882

theorem shirley_boxes_to_cases (boxes_sold : Nat) (boxes_per_case : Nat) (cases_needed : Nat) 
      (h1 : boxes_sold = 54) (h2 : boxes_per_case = 6) : cases_needed = 9 :=
by
  sorry

end shirley_boxes_to_cases_l595_595882


namespace problem_solve_l595_595032

theorem problem_solve (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f x - f (f y + f (-x)) + x) : 
  let n := 1 -- number of possible values of f(-2)
      s := 2 -- sum of all possible values of f(-2)
  in n * s = 2 :=
by sorry

end problem_solve_l595_595032


namespace sin_30_eq_half_l595_595423

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595423


namespace divides_five_iff_l595_595030

theorem divides_five_iff (a : ℤ) : (5 ∣ a^2) ↔ (5 ∣ a) := sorry

end divides_five_iff_l595_595030


namespace work_done_by_force_l595_595934

variable (k x : ℝ)
variable (F : ℝ) (A : ℝ)

-- Given conditions
def hookes_law : Prop := F = k * x
def given_values : Prop := (F = 10) ∧ (x = 0.02)
def work_integral (k : ℝ) : ℝ := ∫(t : ℝ) in (0..0.02), k * t

-- Prove the work done is 0.1J
theorem work_done_by_force : given_values ∧ hookes_law → A = 0.1 :=
by
  sorry

end work_done_by_force_l595_595934


namespace remainder_when_divided_by_5_l595_595872

-- Definitions of the conditions
def condition1 (N : ℤ) : Prop := ∃ R1 : ℤ, N = 5 * 2 + R1
def condition2 (N : ℤ) : Prop := ∃ Q2 : ℤ, N = 4 * Q2 + 2

-- Statement to prove
theorem remainder_when_divided_by_5 (N : ℤ) (R1 : ℤ) (Q2 : ℤ) :
  (N = 5 * 2 + R1) ∧ (N = 4 * Q2 + 2) → (R1 = 4) :=
by
  sorry

end remainder_when_divided_by_5_l595_595872


namespace gray_areas_trees_count_l595_595971

noncomputable def totalTreesInGrayAreas (T : ℕ) (white1 white2 white3 : ℕ) : ℕ :=
  let gray2 := T - white2
  let gray3 := T - white3
  gray2 + gray3

theorem gray_areas_trees_count (T : ℕ) :
  T = 100 → totalTreesInGrayAreas T 100 82 90 = 26 :=
by sorry

end gray_areas_trees_count_l595_595971


namespace gcd_840_1764_l595_595903

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 :=
by
  sorry

end gcd_840_1764_l595_595903


namespace find_x_of_floor_plus_x_eq_17_over_4_l595_595528

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l595_595528


namespace angle_bisector_l595_595891

-- Define the basic geometric elements
variables {Point : Type}
variables (O1 O2 P : Point) (r1 r2 : ℝ)

-- Assume the conditions
axiom circles_tangent_at_P : externally_tangent_circles O1 r1 O2 r2 P
axiom line1_tangent_to_circle2 : tangent_line_through_center O1 r1 O2 r2 ℓ1
axiom line2_tangent_to_circle1 : tangent_line_through_center O2 r2 O1 r1 ℓ2
axiom lines_not_parallel : ¬parallel ℓ1 ℓ2

-- Claim to prove
theorem angle_bisector :
  is_on_angle_bisector P ℓ1 ℓ2 :=
sorry

end angle_bisector_l595_595891


namespace probability_allison_greater_l595_595179

open Probability

-- Definitions for the problem
noncomputable def die_A : ℕ := 4

noncomputable def die_B : PMF ℕ :=
  PMF.uniform_of_fin (fin 6) -- Could also explicitly write as PMF.of_list [(1, 1/6), ..., (6, 1/6)]

noncomputable def die_N : PMF ℕ :=
  PMF.of_list [(3, 3/6), (5, 3/6)]

-- The target event: Allison's roll > Brian's roll and Noah's roll
noncomputable def event (roll_A : ℕ) (roll_B roll_N : PMF ℕ) :=
  ∀ b n, b ∈ [1, 2, 3] → n ∈ [3] → roll_A > b ∧ roll_A > n

-- The probability calculation
theorem probability_allison_greater :
  (∑' (b : ℕ) (h_b : b < 4) (n : ℕ) (h_n : n = 3), die_B b * die_N n)
  = 1 / 4 :=
by
  -- assumed rolls
  let roll_A := 4
  let prob_B := 1 / 2
  let prob_N := 1 / 2
  
  -- skip proof, but assert the correct result.
  sorry

end probability_allison_greater_l595_595179


namespace part_I_solution_part_II_solution_l595_595610

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

-- Part (I)
theorem part_I_solution (x : ℝ) :
  f 1 x > 4 ↔ x < -3 ∨ x > 1 :=
by sorry

-- Part (II)
theorem part_II_solution (a : ℝ) (h : a ≠ 0) :
  (∃ x ∈ set.Ioo 1 2, f a x = 0) ↔ -1 / 3 < a ∧ a < -1 / 8 :=
by sorry

end part_I_solution_part_II_solution_l595_595610


namespace range_of_f_l595_595561

noncomputable def f (x : ℝ) : ℝ := 3^(x - 2)

theorem range_of_f : Set.Icc 1 9 = {y : ℝ | ∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ f x = y} :=
by
  sorry

end range_of_f_l595_595561


namespace distance_from_P_to_AB_l595_595104

-- Definitions of conditions
def is_point_in_triangle (P A B C : ℝ×ℝ) : Prop := sorry
def parallel_to_base (P A B C : ℝ×ℝ) : Prop := sorry
def divides_area_in_ratio (P A B C : ℝ×ℝ) (r1 r2 : ℕ) : Prop := sorry

theorem distance_from_P_to_AB (P A B C : ℝ×ℝ) 
  (H_in_triangle : is_point_in_triangle P A B C)
  (H_parallel : parallel_to_base P A B C)
  (H_area_ratio : divides_area_in_ratio P A B C 1 3)
  (H_altitude : ∃ h : ℝ, h = 1) :
  ∃ d : ℝ, d = 3/4 :=
by
  sorry

end distance_from_P_to_AB_l595_595104


namespace sin_30_eq_half_l595_595452

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595452


namespace dow_original_value_l595_595070

-- Given conditions
def Dow_end := 8722
def percentage_fall := 0.02
def final_percentage := 1 - percentage_fall -- 98% of the original value

-- To prove: the original value of the Dow Jones Industrial Average equals 8900
theorem dow_original_value :
  (Dow_end: ℝ) = (final_percentage * 8900) := 
by sorry

end dow_original_value_l595_595070


namespace mischievous_chessboard_l595_595159

-- Definitions for the problem
def is_cycle (n : ℕ) (seq : List (ℕ × ℕ)) : Prop :=
  ∃ k, seq.length = k ∧ k ≥ 4 ∧ (∀ i : Fin k, (seq[i] = seq[(i + 1) % k] ∨ 
  abs (seq[i].fst - seq[(i + 1) % k].fst) = 1 ∨ abs (seq[i].snd - seq[(i + 1) % k].snd) = 1))
  
def is_mischievous (n : ℕ) (X : Set (ℕ × ℕ)) : Prop :=
  ∀ seq, is_cycle n seq → seq.toSet ∩ X ≠ ∅
 
-- Main conjecture restated as a Lean theorem
theorem mischievous_chessboard (C : ℝ) :
  (∀ n, n ≥ 2 → ∃ (X : Set (ℕ × ℕ)), is_mischievous n X ∧ X.card ≤ C * (n^2)) ↔ C ≥ (1 / 3) := 
sorry -- Proof omitted

end mischievous_chessboard_l595_595159


namespace find_x_l595_595540

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l595_595540


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595664

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595664


namespace sin_thirty_degrees_l595_595373

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595373


namespace hyperbola_condition_l595_595074

theorem hyperbola_condition (k : ℝ) : 
  (0 ≤ k ∧ k < 3) → (∃ a b : ℝ, a * b < 0 ∧ 
    (a = k + 1) ∧ (b = k - 5)) ∧ (∀ m : ℝ, -1 < m ∧ m < 5 → ∃ a b : ℝ, a * b < 0 ∧ 
    (a = m + 1) ∧ (b = m - 5)) :=
by
  sorry

end hyperbola_condition_l595_595074


namespace solve_quadratic_equation_solve_cubic_equation_l595_595198

-- First problem statement
theorem solve_quadratic_equation (x : ℝ) : 4 * (x - 1) ^ 2 = 8 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 :=
by
  sorry

-- Second problem statement
theorem solve_cubic_equation (x : ℝ) : 2 * x ^ 3 = 8 ↔ abs (x - Real.cbrt 4) < 0.01 :=
by
  sorry

end solve_quadratic_equation_solve_cubic_equation_l595_595198


namespace find_x_eq_nine_fourths_l595_595531

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l595_595531


namespace sin_30_eq_half_l595_595408

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595408


namespace sin_twenty_five_pi_over_six_l595_595505

theorem sin_twenty_five_pi_over_six : sin (25 * Real.pi / 6) = 1 / 2 := by
  sorry

end sin_twenty_five_pi_over_six_l595_595505


namespace distinct_prime_factors_sum_divisors_450_l595_595753

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595753


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595668

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595668


namespace derivative_of_constant_cos_l595_595077

theorem derivative_of_constant_cos : 
  let y := cos (π / 3) in deriv (λ x : ℝ, y) = 0 := 
by
  let y := cos (π / 3)
  sorry

end derivative_of_constant_cos_l595_595077


namespace divisibility_l595_595877

noncomputable def succession : ℕ → ℕ
| 0     := 6
| (n+1) := sorry 

theorem divisibility (n : ℕ) : 
  ∃ (a : ℕ → ℕ), (∀ i, a i ∈ {0,1,2,3,4,5,6,7,8,9}) ∧ a 0 = 6 ∧ 
  (∀ n, let x := ∑ i in finset.range n, a i * 10^i in (x^2 - x) % 10^n = 0) :=
sorry

end divisibility_l595_595877


namespace find_circle_eq_and_tangent_lines_l595_595565

variables {R : Type*} [linear_ordered_field R]

def circle_eq (x y : R) := (x - 1)^2 + y^2 = 1

noncomputable def line_tangent1 (x : R) := x = 2

noncomputable def line_tangent2 (k x y : R) := 4 * x - 3 * y + 1 = 0

theorem find_circle_eq_and_tangent_lines :
  (∀ (x y : R), (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 0) ∨
    (exists (m : R), y = m * (x - 1) ∨ 
    ((y = a ∧ a = 0 ∧ (∃ (r : R), x^2 + a^2 = r^2 ∨ 
    ((1 - x)^2 + (1 - a)^2 = x^2))))) → 
  circle_eq x y) ∧ 
  (∀ (x y : R), (x = 2 ∧ y = 3) →
  line_tangent1 x ∨ line_tangent2 (4 / 3) x y) :=
sorry

end find_circle_eq_and_tangent_lines_l595_595565


namespace log_base_5_3125_interval_sum_l595_595913

theorem log_base_5_3125_interval_sum : 
  ∃ (c d : ℤ), (4 < log 5 3125 ∧ log 5 3125 < 6) ∧ log 5 3125 = 5 ∧ (c = 4) ∧ (d = 5) ∧ c + d = 9 :=
by {
  sorry
}

end log_base_5_3125_interval_sum_l595_595913


namespace Ram_days_to_complete_task_l595_595878

-- Define the efficiency conditions
def Ram_efficiency_ratio : ℝ := 0.5

-- Define the condition that Ram and Krish together take 7 days to complete the task
def combined_days_to_complete_task : ℝ := 7

-- Define the theorem to prove that Ram takes 21 days to complete the task
theorem Ram_days_to_complete_task : 
  (Ram_efficiency_ratio * (1 / combined_days_to_complete_task)) ^ (-1) = 21 :=
by
  -- Use sorry to skip the proof
  sorry

end Ram_days_to_complete_task_l595_595878


namespace sin_of_30_degrees_l595_595267

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l595_595267


namespace calories_consumed_in_week_l595_595506

-- Define the calorie content of each type of burger
def calorie_A := 350
def calorie_B := 450
def calorie_C := 550

-- Define Dimitri's burger consumption over the 7 days
def consumption_day1 := (2 * calorie_A) + (1 * calorie_B)
def consumption_day2 := (1 * calorie_A) + (2 * calorie_B) + (1 * calorie_C)
def consumption_day3 := (1 * calorie_A) + (1 * calorie_B) + (2 * calorie_C)
def consumption_day4 := (3 * calorie_B)
def consumption_day5 := (1 * calorie_A) + (1 * calorie_B) + (1 * calorie_C)
def consumption_day6 := (2 * calorie_A) + (3 * calorie_C)
def consumption_day7 := (1 * calorie_B) + (2 * calorie_C)

-- Define the total weekly calorie consumption
def total_weekly_calories :=
  consumption_day1 + consumption_day2 + consumption_day3 +
  consumption_day4 + consumption_day5 + consumption_day6 + consumption_day7

-- State and prove the main theorem
theorem calories_consumed_in_week :
  total_weekly_calories = 11450 := 
by
  sorry

end calories_consumed_in_week_l595_595506


namespace sin_thirty_degrees_l595_595381

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595381


namespace probability_of_successful_meeting_l595_595979

noncomputable def meeting_probability : ℚ := 7 / 64

theorem probability_of_successful_meeting :
  (∃ x y z : ℝ,
     0 ≤ x ∧ x ≤ 2 ∧
     0 ≤ y ∧ y ≤ 2 ∧
     0 ≤ z ∧ z ≤ 2 ∧
     abs (x - z) ≤ 0.75 ∧
     abs (y - z) ≤ 1.5 ∧
     z ≥ x ∧
     z ≥ y) →
  meeting_probability = 7 / 64 := by
  sorry

end probability_of_successful_meeting_l595_595979


namespace team_selection_l595_595556

theorem team_selection :
  let teachers := 5
  let students := 10
  (teachers * students = 50) :=
by
  sorry

end team_selection_l595_595556


namespace sin_30_eq_half_l595_595291
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595291


namespace sin_30_eq_half_l595_595443

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595443


namespace sin_30_eq_half_l595_595203

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595203


namespace average_speed_correct_l595_595946

-- Definitions for the conditions
def distance_first_hour : ℝ := 80
def distance_second_hour : ℝ := 40
def time_first_hour : ℝ := 1
def time_second_hour : ℝ := 1

-- Definition for total distance and time
def total_distance : ℝ := distance_first_hour + distance_second_hour
def total_time : ℝ := time_first_hour + time_second_hour

-- Definition of average speed
def average_speed : ℝ := total_distance / total_time

-- Lean statement asserting the mathematical equivalence
theorem average_speed_correct :
  average_speed = 60 := by
  -- proof goes here
  sorry

end average_speed_correct_l595_595946


namespace jog_time_l595_595786

def time_to_jog (distance time : ℕ) : ℕ := 
  time * distance

theorem jog_time : 
  ∀ (time distance : ℕ), 
  distance = 1.5 * 2 * 1 → 
  time_to_jog 1.5 (time / distance) = 12 :=
sorry

end jog_time_l595_595786


namespace sin_30_eq_half_l595_595440

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l595_595440


namespace cutting_stick_ways_l595_595497

theorem cutting_stick_ways :
  ∃ (s : Finset (ℕ × ℕ)), 
  (∀ a ∈ s, 2 * a.1 + 3 * a.2 = 14) ∧
  s.card = 2 := 
by
  sorry

end cutting_stick_ways_l595_595497


namespace same_units_digit_pages_count_l595_595138

noncomputable def same_units_digit_pages : ℕ :=
  (filter (λ x, x % 10 = (64 - x) % 10) (list.range' 1 63)).length

theorem same_units_digit_pages_count : same_units_digit_pages = 13 := by
  sorry

end same_units_digit_pages_count_l595_595138


namespace carl_max_rocks_value_l595_595939

/-- 
Carl finds rocks of three different types:
  - 6-pound rocks worth $18 each.
  - 3-pound rocks worth $9 each.
  - 2-pound rocks worth $3 each.
There are at least 15 rocks available for each type.
Carl can carry at most 20 pounds.

Prove that the maximum value, in dollars, of the rocks Carl can carry out of the cave is $57.
-/
theorem carl_max_rocks_value : 
  (∃ x y z : ℕ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 6 * x + 3 * y + 2 * z ≤ 20 ∧ 18 * x + 9 * y + 3 * z = 57) :=
sorry

end carl_max_rocks_value_l595_595939


namespace sum_of_divisors_prime_factors_l595_595686

theorem sum_of_divisors_prime_factors (n : ℕ) (h : n = 2^1 * 3^2 * 5^2) :
  ∃ p : ℕ, nat.totient ( (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ) = 3 :=
by
  rw h
  sorry

end sum_of_divisors_prime_factors_l595_595686


namespace sum_of_solutions_l595_595931

theorem sum_of_solutions : {s : Int | 4 < (s - 2)^2 ∧ (s - 2)^2 < 36}.sum = 12 :=
by
  sorry

end sum_of_solutions_l595_595931


namespace sin_30_eq_half_l595_595206

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l595_595206


namespace sum_of_divisors_prime_factors_450_l595_595739

theorem sum_of_divisors_prime_factors_450 :
  let n := 450
  let prime_factors := [2^1 * 3^2 * 5^2]
  let sum_of_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  ∃ (p1 p2 p3 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : Prime p3),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∃ L: List ℕ, List.distinct L ∧ 
    (∀ x ∈ L, x = p1 ∨ x = p2 ∨ x = p3) ∧ L.length = 3 
  :=
by
  sorry

end sum_of_divisors_prime_factors_450_l595_595739


namespace sin_30_eq_half_l595_595430

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l595_595430


namespace least_number_subtracted_l595_595118

theorem least_number_subtracted (x : ℤ) (N : ℤ) :
  N = 2590 - x →
  (N % 9 = 6) →
  (N % 11 = 6) →
  (N % 13 = 6) →
  x = 10 :=
by
  sorry

end least_number_subtracted_l595_595118


namespace distinct_prime_factors_sum_divisors_450_l595_595746

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595746


namespace tower_height_proof_l595_595072

-- Definitions corresponding to given conditions
def elev_angle_A : ℝ := 45
def distance_AD : ℝ := 129
def elev_angle_D : ℝ := 60
def tower_height : ℝ := 305

-- Proving the height of Liaoning Broadcasting and Television Tower
theorem tower_height_proof (h : ℝ) (AC CD : ℝ) (h_eq_AC : h = AC) (h_eq_CD_sqrt3 : h = CD * (Real.sqrt 3)) (AC_CD_sum : AC + CD = 129) :
  h = 305 :=
by
  sorry

end tower_height_proof_l595_595072


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595711

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595711


namespace distinct_prime_factors_sum_divisors_450_l595_595749

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595749


namespace distinct_prime_factors_sum_divisors_450_l595_595742

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595742


namespace find_x_eq_nine_fourths_l595_595530

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l595_595530


namespace floor_plus_x_eq_17_over_4_l595_595522

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l595_595522


namespace sin_30_is_half_l595_595337

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l595_595337


namespace speed_upstream_calculation_l595_595911

def speed_boat_still_water : ℝ := 60
def speed_current : ℝ := 17

theorem speed_upstream_calculation : speed_boat_still_water - speed_current = 43 := by
  sorry

end speed_upstream_calculation_l595_595911


namespace sin_thirty_deg_l595_595227

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595227


namespace circle_radius_l595_595142

theorem circle_radius (r x y : ℝ) (hx : x = π * r^2) (hy : y = 2 * π * r) (h : x + y = 90 * π) : r = 9 := by
  sorry

end circle_radius_l595_595142


namespace sin_30_deg_l595_595350

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595350


namespace sin_30_eq_half_l595_595289
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595289


namespace well_diameter_approx_l595_595549

noncomputable def well_diameter (h : ℝ) (P : ℝ) (C : ℝ) : ℝ :=
  2 * real.sqrt (C / (P * π * h))

theorem well_diameter_approx :
  well_diameter 14 18 1781.28 ≈ 2.9966 := 
by
  sorry

end well_diameter_approx_l595_595549


namespace distinct_prime_factors_of_sigma_450_l595_595782

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595782


namespace sin_30_eq_half_l595_595461

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l595_595461


namespace cosine_of_angle_in_second_quadrant_l595_595788

theorem cosine_of_angle_in_second_quadrant
  (α : ℝ)
  (h1 : Real.sin α = 1 / 3)
  (h2 : π / 2 < α ∧ α < π) :
  Real.cos α = - (2 * Real.sqrt 2) / 3 :=
by
  sorry

end cosine_of_angle_in_second_quadrant_l595_595788


namespace unique_pos_four_digit_integers_l595_595628

theorem unique_pos_four_digit_integers : 
  (∃ digits : Fin 4 → ℕ, 
    digits 0 = 2 ∧ 
    digits 1 = 2 ∧ 
    digits 2 = 9 ∧ 
    digits 3 = 9 ∧ 
    ∀ i, i < 4 → digits i ∈ {2, 9}) → 
  (∃ n, n = 6) := 
by 
  sorry

end unique_pos_four_digit_integers_l595_595628


namespace distinct_lines_l595_595873

theorem distinct_lines (k n : ℕ) (h₁ : k > 1)
  (h₂ : (n + 1) * (k - 1) = 176)
  (h₃ : 2 * (k - 1) + (n - 1) * k + ((n - 2) * (n - 1)) / 2 = 221) :
  k = 17 ∧ n = 10 :=
by {
  sorry,
}

end distinct_lines_l595_595873


namespace smallest_mul_seven_perfect_square_l595_595112

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define the problem statement
theorem smallest_mul_seven_perfect_square :
  ∀ x : ℕ, x > 0 → (is_perfect_square (7 * x) ↔ x = 7) := 
by {
  sorry
}

end smallest_mul_seven_perfect_square_l595_595112


namespace paper_bag_cookie_capacity_l595_595984

/-- Edgar buys 292 cookies and needs 19 paper bags. Prove that one paper bag can hold 15 cookies. -/
theorem paper_bag_cookie_capacity : 
  ∀ (cookies paper_bags : ℕ), cookies = 292 → paper_bags = 19 → (cookies / paper_bags) = 15 :=
by
  intros cookies paper_bags hc hp
  rw [hc, hp]
  sorry

end paper_bag_cookie_capacity_l595_595984


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595661

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (finset.Icc 1 n), if (n % d = 0) then d else 0

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (n.factorization.keys.to_finset).card

theorem distinct_prime_factors_of_sum_of_divisors_of_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by
  sorry

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595661


namespace floor_sum_log_eq857_l595_595554

noncomputable def floor_sum_log : ℝ :=
  (∑ n in finset.range 244, ⌊real.log n / real.log 3⌋)

theorem floor_sum_log_eq857 : floor_sum_log = 857 := 
by
  -- statement only, proof is skipped
  sorry

end floor_sum_log_eq857_l595_595554


namespace Fran_speed_required_l595_595837

variable (s_J : ℝ) (t_J : ℝ) (b_J : ℝ) (t_F : ℝ) (b_F : ℝ) (s_F : ℝ)

-- Given conditions
axiom Joann_speed : s_J = 10
axiom Joann_total_time : t_J = 4
axiom Joann_break_time : b_J = 1
axiom Fran_total_time : t_F = 3
axiom Fran_break_time : b_F = 0.5

-- Effective riding time calculations
def Joann_effective_time : ℝ := t_J - b_J
def Fran_effective_time : ℝ := t_F - b_F

-- Distance Joann traveled
def Joann_distance : ℝ := s_J * Joann_effective_time s_J t_J b_J

-- Fran's required speed calculation
def Fran_required_speed : ℝ := Joann_distance s_J t_J b_J / Fran_effective_time t_F b_F

-- Theorem (what we need to prove)
theorem Fran_speed_required : Fran_required_speed s_J t_J b_J t_F b_F = 12 :=
by
  sorry

end Fran_speed_required_l595_595837


namespace cycle_count_bound_l595_595028

noncomputable def num_cycles (G : Type) [graph G] : Nat

variable (G : Type) [graph G]

theorem cycle_count_bound (r : Nat) (V : Finset G)
  (h1 : 0 < r)
  (h2 : ∀ (C : Finset G), C.card ≤ 2 * r → ¬(G.is_cycle C))
  : num_cycles G ≤ |V| ^ 2016 := sorry

end cycle_count_bound_l595_595028


namespace distinct_prime_factors_of_sum_of_divisors_of_450_l595_595712

noncomputable def sum_of_divisors (n: ℕ) : ℕ := 
  ∑ d in divisors n, d

theorem distinct_prime_factors_of_sum_of_divisors_of_450 : 
  (nat.factors (sum_of_divisors 450)).to_finset.card = 3 :=
by {
  -- prime factorization: 450 = 2^1 * 3^2 * 5^2
  have h_factorization: 450 = 2^1 * 3^2 * 5^2 := by norm_num1,
  rw [sum_of_divisors, div_eq_mul_invs, h_factorization],
  sorry
}

end distinct_prime_factors_of_sum_of_divisors_of_450_l595_595712


namespace sin_30_deg_l595_595346

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l595_595346


namespace sin_30_eq_half_l595_595244

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l595_595244


namespace distance_AB_l595_595548

def Point := (ℝ × ℝ × ℝ)

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

def A : Point := (0, 15, 5)
def B : Point := (8, 0, -3)

theorem distance_AB : distance A B = real.sqrt 353 := by
  sorry

end distance_AB_l595_595548


namespace solution_x_chemical_a_percentage_l595_595989

variable (Ax : ℝ) (A2 : ℝ := 0.40) (mixture : ℝ := 0.32) (Sx : ℝ := 0.80)

theorem solution_x_chemical_a_percentage (h : 0.80 * Ax + 0.20 * A2 = mixture) : 
  Ax = 0.30 :=
by
  have h1 : 0.80 * Ax + 0.08 = 0.32 := by simp [h, A2]
  have h2 : 0.80 * Ax = 0.24 := by linarith
  exact eq_div_of_mul_eq _ _ (by norm_num) h2

end solution_x_chemical_a_percentage_l595_595989


namespace sin_thirty_degrees_l595_595388

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l595_595388


namespace distinct_prime_factors_sum_divisors_450_l595_595745

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sum_divisors := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  (Nat.factors (Nat.sumDivisors n)).toFinset.card = 3 :=
by
  sorry

end distinct_prime_factors_sum_divisors_450_l595_595745


namespace triangle_area_ratio_l595_595879

structure RegularHexagon (A B C D E F G : Type) :=
(is_center : ∀ P, P = G → (Center_conditions P A B C D E F))

structure Center_conditions (P A B C D E F : Type) :=
(connects_to_mid_and_vertices : ∀ Q, Mid_and_vertices Q A B C D E F)

def Mid_and_vertices (Q A B C D E F : Type) : Prop := sorry

theorem triangle_area_ratio 
  (A B C D E F G : Type) 
  (hex : RegularHexagon A B C D E F G) 
  (center_cond : Center_conditions G A B C D E F)
  (conditions : ∀ P, Mid_and_vertices P A B C D E F) : 
  ([Triangle_area_ratio A B G] / [Triangle_area_ratio A D F]) = (1 / 12) := 
sorry

noncomputable def Triangle_area_ratio (A B G : Type) : ℝ := sorry

end triangle_area_ratio_l595_595879


namespace brooke_sidney_ratio_l595_595065

-- Definitions for the conditions
def sidney_monday : ℕ := 20
def sidney_tuesday : ℕ := 36
def sidney_wednesday : ℕ := 40
def sidney_thursday : ℕ := 50
def brooke_total : ℕ := 438

-- Total jumping jacks by Sidney
def sidney_total : ℕ := sidney_monday + sidney_tuesday + sidney_wednesday + sidney_thursday

-- The ratio of Brooke’s jumping jacks to Sidney's total jumping jacks
def ratio := brooke_total / sidney_total

-- The proof goal
theorem brooke_sidney_ratio : ratio = 3 :=
by
  sorry

end brooke_sidney_ratio_l595_595065


namespace sin_thirty_deg_l595_595223

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l595_595223


namespace roots_quadratic_l595_595601

theorem roots_quadratic (m x₁ x₂ : ℝ) (h : m < 0) (h₁ : x₁ < x₂) (hx : ∀ x, (x^2 - x - 6 = m) ↔ (x = x₁ ∨ x = x₂)) : 
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 :=
by {
  sorry
}

end roots_quadratic_l595_595601


namespace find_sum_of_valid_x_l595_595932

-- Definitions based on the conditions
def num_set := [3, 5, 7, 18]

def mean (x : ℝ) : ℝ := (33 + x) / 5

def median (x : ℝ) : ℝ :=
  -- Sorting the list and taking the third element (1-based index) for median
  let sorted := list.sort (num_set ++ [x]) 
  in sorted.nth_le 2 sorry

theorem find_sum_of_valid_x :
  (∑ x in {x : ℝ | median x = mean x}.to_finset, x) = -8 :=
sorry

end find_sum_of_valid_x_l595_595932


namespace sin_30_eq_half_l595_595415

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595415


namespace distinct_prime_factors_of_sigma_450_l595_595673
open Nat

theorem distinct_prime_factors_of_sigma_450 : 
  ∃ n, n = 3 ∧ ∃ divs : Finset ℕ, (divs = (divisors 450).sum ∧ (divs.val.primes.card = n)) := sorry

end distinct_prime_factors_of_sigma_450_l595_595673


namespace sin_30_eq_half_l595_595414

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l595_595414


namespace find_intercept_l595_595990

theorem find_intercept (avg_height : ℝ) (avg_shoe_size : ℝ) (a : ℝ)
  (h1 : avg_height = 170)
  (h2 : avg_shoe_size = 40) 
  (h3 : 3 * avg_shoe_size + a = avg_height) : a = 50 := 
by
  sorry

end find_intercept_l595_595990


namespace james_nickels_count_l595_595015

-- Definitions
def total_cents : ℕ := 685
def more_nickels_than_quarters := 11

-- Variables representing the number of nickels and quarters
variables (n q : ℕ)

-- Conditions
axiom h1 : 5 * n + 25 * q = total_cents
axiom h2 : n = q + more_nickels_than_quarters

-- Theorem stating the number of nickels
theorem james_nickels_count : n = 32 := 
by
  -- Proof will go here, marked as "sorry" to complete the statement
  sorry

end james_nickels_count_l595_595015


namespace solution_of_equation_value_of_expression_l595_595956

-- Part 1: Solve the equation 3x(x-2) = 2(2-x)
theorem solution_of_equation (x : ℝ) : 3 * x * (x - 2) = 2 * (2 - x) ↔ (x = -2/3 ∨ x = 2) :=
by sorry
    
-- Part 2: Calculate |-4| - 2cos 60° + (sqrt(3) - sqrt(2))^0 - (-3)^2
theorem value_of_expression : 
  |-4| - 2 * real.cos (real.pi / 3) + (real.sqrt 3 - real.sqrt 2)^0 - (-3)^2 = -5 :=
by sorry

end solution_of_equation_value_of_expression_l595_595956


namespace sum_of_divisors_has_three_distinct_prime_factors_l595_595714

-- Condition: Define the integer n and its prime factorization
def n : ℕ := 450
def prime_factorization_of_n : list ℕ := [2^1, 3^2, 5^2]

-- Sum of positive divisors
def sum_of_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)

-- Prove that the sum of the positive divisors has 3 distinct prime factors
theorem sum_of_divisors_has_three_distinct_prime_factors :
  ∃ (factors : list ℕ), factors = [3, 13, 31] ∧ factors.length = 3 ∧
                          ∀ p ∈ factors, p.prime :=
sorry

end sum_of_divisors_has_three_distinct_prime_factors_l595_595714


namespace total_pages_read_l595_595064

-- Define the reading rates
def ReneReadingRate : ℕ := 30  -- pages in 60 minutes
def LuluReadingRate : ℕ := 27  -- pages in 60 minutes
def CherryReadingRate : ℕ := 25  -- pages in 60 minutes

-- Total time in minutes
def totalTime : ℕ := 240  -- minutes

-- Define a function to calculate pages read in given time
def pagesRead (rate : ℕ) (time : ℕ) : ℕ :=
  rate * (time / 60)

-- Theorem to prove the total number of pages read
theorem total_pages_read :
  pagesRead ReneReadingRate totalTime +
  pagesRead LuluReadingRate totalTime +
  pagesRead CherryReadingRate totalTime = 328 :=
by
  -- Proof is not required, hence replaced with sorry
  sorry

end total_pages_read_l595_595064


namespace max_non_overlapping_areas_l595_595967

theorem max_non_overlapping_areas (n : ℕ) : 
  ∀ r s : set ℝ, (r.card = 3 * n) ∧ (s.card = 2) ∧ (∀ x ∈ s, x ≠ 0) → 
  ∃ A : ℕ, A = 6 * n + 1 := sorry

end max_non_overlapping_areas_l595_595967


namespace find_h_l595_595037

-- Definition of the operation \( a \star b \)
def star (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt b)

-- Assume the given condition
theorem find_h (h : ℝ) (H : star 8 h = 11) : h = 6 :=
sorry

end find_h_l595_595037


namespace odd_function_value_at_neg_one_l595_595591

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 + x else -(x^2 + x)

theorem odd_function_value_at_neg_one : f(-1) = -2 := by
  sorry

end odd_function_value_at_neg_one_l595_595591


namespace hawks_points_l595_595798

theorem hawks_points (x y : ℕ) (h1 : x + y = 50) (h2 : x + 4 - y = 12) : y = 21 :=
by
  sorry

end hawks_points_l595_595798


namespace number_of_good_partitions_is_8362_l595_595860

-- Define the set M
def M : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the condition for A_i sets
def is_partition (A1 A2 A3 : Finset ℕ) : Prop :=
  A1 ∪ A2 ∪ A3 = M ∧ A1 ∩ A2 = ∅ ∧ A1 ∩ A3 = ∅ ∧ A2 ∩ A3 = ∅ ∧ A1 ≠ ∅ ∧ A2 ≠ ∅ ∧ A3 ≠ ∅

-- Define the good partition condition
def is_good_partition (A1 A2 A3 : Finset ℕ) : Prop :=
  ∃ (i1 i2 i3 : Finset ℕ), 
    ({i1, i2, i3} = {A1, A2, A3} ∧ 
    (i1 ∪ i2 ∪ i3 = M ∧ max i1 > min i2 ∧ max i2 > min i3 ∧ max i3 > min i1))

-- The final statement asserting the number of good partitions is 8362
theorem number_of_good_partitions_is_8362 :
  ∃ (S : Finset (Finset ℕ × Finset ℕ × Finset ℕ)),
    (∀ (A1 A2 A3 : Finset ℕ), (A1, A2, A3) ∈ S ↔ is_partition A1 A2 A3 ∧ is_good_partition A1 A2 A3) ∧ 
    S.card = 8362 :=
sorry

end number_of_good_partitions_is_8362_l595_595860


namespace volume_and_surface_area_of_convex_body_l595_595083

noncomputable def volume_of_convex_body (a b c : ℝ) : ℝ := 
  (a^2 + b^2 + c^2)^3 / (6 * a * b * c)

noncomputable def surface_area_of_convex_body (a b c : ℝ) : ℝ :=
  (a^2 + b^2 + c^2)^(5/2) / (a * b * c)

theorem volume_and_surface_area_of_convex_body (a b c d : ℝ)
  (h : d^2 = a^2 + b^2 + c^2) :
  volume_of_convex_body a b c = (a^2 + b^2 + c^2)^3 / (6 * a * b * c) ∧
  surface_area_of_convex_body a b c = (a^2 + b^2 + c^2)^(5/2) / (a * b * c) :=
by
  sorry

end volume_and_surface_area_of_convex_body_l595_595083


namespace num_distinct_prime_factors_sum_divisors_450_l595_595648

theorem num_distinct_prime_factors_sum_divisors_450 : 
  ∀ (n : ℕ), n = 450 → (nat.num_distinct_prime_factors (nat.sigma n) = 3) :=
by
  intros n h
  rw h
  sorry

end num_distinct_prime_factors_sum_divisors_450_l595_595648


namespace y_comparison_l595_595582

theorem y_comparison :
  let y1 := (-1)^2 - 2*(-1) + 3
  let y2 := (-2)^2 - 2*(-2) + 3
  y2 > y1 := by
  sorry

end y_comparison_l595_595582


namespace ivan_apples_leftover_l595_595825

theorem ivan_apples_leftover :
  ∀ (total_apples : ℕ) (mini_pie_apples : ℚ) (num_mini_pies : ℕ),
    total_apples = 48 →
    mini_pie_apples = 1 / 2 →
    num_mini_pies = 24 →
    total_apples - (num_mini_pies * mini_pie_apples) = 36 :=
by
  intros total_apples mini_pie_apples num_mini_pies ht hm hn
  rw [ht, hm, hn]
  norm_num
  sorry

end ivan_apples_leftover_l595_595825


namespace sin_30_eq_half_l595_595296
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l595_595296


namespace sampling_methods_l595_595968
-- Import the necessary library

-- Definitions for the conditions of the problem:
def NumberOfFamilies := 500
def HighIncomeFamilies := 125
def MiddleIncomeFamilies := 280
def LowIncomeFamilies := 95
def SampleSize := 100

def FemaleStudentAthletes := 12
def NumberToChoose := 3

-- Define the appropriate sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Stating the proof problem in Lean 4
theorem sampling_methods :
  SamplingMethod.Stratified = SamplingMethod.Stratified ∧
  SamplingMethod.SimpleRandom = SamplingMethod.SimpleRandom :=
by
  -- Proof is omitted in this theorem statement
  sorry

end sampling_methods_l595_595968


namespace number_of_subsets_of_C_l595_595043

def A : set (ℝ × ℝ) := { p | p.1 + p.2 = 1 }
def B : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 2 }
def C : set (ℝ × ℝ) := A ∩ B

theorem number_of_subsets_of_C : fintype.card (set C) = 4 := 
sorry

end number_of_subsets_of_C_l595_595043


namespace distinct_prime_factors_of_sigma_450_l595_595774

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x => n % x = 0) (Finset.range (n+1))), d

theorem distinct_prime_factors_of_sigma_450 :
  ∃ pf : Finset ℕ, ((sigma 450).primeFactors = pf) ∧ pf.card = 3 :=
by
  sorry

end distinct_prime_factors_of_sigma_450_l595_595774


namespace find_x_of_floor_plus_x_eq_17_over_4_l595_595525

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l595_595525


namespace kenneth_and_ellen_l595_595105

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 1000)

def kenneth_sum (a : ℂ) : ℂ :=
  ∑ k in Finset.range 1000, 1 / (omega^k - a)

def ellen_sum (a : ℂ) : ℂ :=
  ∑ k in Finset.range 1000, 1 / omega^k - 1000 * a

theorem kenneth_and_ellen (a : ℂ) (h : kenneth_sum a = ellen_sum a) :
  a = 0 ∨ (a^1000 - a^998 - 1 = 0) :=
sorry

end kenneth_and_ellen_l595_595105
