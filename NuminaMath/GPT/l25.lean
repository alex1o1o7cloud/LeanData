import Mathlib

namespace count_unique_three_digit_numbers_l25_25965

theorem count_unique_three_digit_numbers : 
  let digits := {3, 5, 7, 9} in
  let count := digits.card * (digits.card - 1) * (digits.card - 2) in
  count = 24 :=
by
  let digits : finset ℕ := {3, 5, 7, 9}
  let count := digits.card * (digits.card - 1) * (digits.card - 2)
  have h : digits.card = 4 := by
    simp only [digits, finset.card_insert_of_not_mem, finset.card_singleton]
    simp
  rw h
  exact rfl

end count_unique_three_digit_numbers_l25_25965


namespace domain_h_g_function_l25_25970

noncomputable def f (x : ℝ) : ℝ := real.log10 (x + 1)
noncomputable def h (x : ℝ) : ℝ := real.sqrt (f x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ∈ Icc 0 1 then f x
  else if x ∈ Icc (-1) 0 then f (-x)
  else 0 -- This definition is only valid in [-1,1]

theorem domain_h : ∀ x, h x ≥ 0 ↔ x ≥ 0 := 
sorry 

theorem g_function : ∀ x, x ∈ Icc (-1) 1 →
  (g x = real.log10 (x + 1) ∧ x ∈ Icc 0 1) ∨
  (g x = real.log10 (1 - x) ∧ x ∈ Icc (-1) 0) ∨
  (g x = g (-x)) :=
sorry 

end domain_h_g_function_l25_25970


namespace distance_to_directrix_l25_25037

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25037


namespace distance_from_A_to_directrix_of_parabola_l25_25063

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25063


namespace max_subset_A_l25_25679

open Finset

theorem max_subset_A :
  ∃ A ⊆ (Icc 0 29),
    (∀ a b ∈ A, ∀ k : ℤ, ¬ ∃ n : ℤ, a + b + 30 * k = n * (n + 1)) ∧
    ∀ B ⊆ (Icc 0 29),
    (∀ a b ∈ B, ∀ k : ℤ, ¬ ∃ n : ℤ, a + b + 30 * k = n * (n + 1)) →
    card A ≥ card B :=
sorry

end max_subset_A_l25_25679


namespace sqrt_15_between_3_and_4_l25_25895

theorem sqrt_15_between_3_and_4 :
  3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 :=
by
  have h1 : 3^2 = 9 := by norm_num
  have h2 : 4^2 = 16 := by norm_num
  have h3 : 9 < 15 ∧ 15 < 16 := by split; norm_num
  sorry

end sqrt_15_between_3_and_4_l25_25895


namespace tetrahedron_volume_inequality_l25_25289

variable {R : Type*} [LinearOrderedField R]

theorem tetrahedron_volume_inequality (d1 d2 d3 : R) (V : R) 
  (hV : ∀ V₀, (∃ (tetrahedron : geometric_object) 
    (h_distances : tetrahedron.opposite_edges_distances = (d1, d2, d3)), tetrahedron.volume = V₀) 
    → V ≤ V₀) : 
  V ≥ (1 / 3) * d1 * d2 * d3 :=
sorry

end tetrahedron_volume_inequality_l25_25289


namespace problem_statements_l25_25227

variable {a : ℕ → ℝ} {p : ℝ}

def eq_square_diff_seq (a : ℕ → ℝ) (p : ℝ) : Prop :=
  ∀ n, n ≥ 2 → a n ^ 2 - a (n - 1) ^ 2 = p

theorem problem_statements
  (h1 : eq_square_diff_seq a p)
  (h2 : ∀ k : ℕ, k > 0 → eq_square_diff_seq (λ n, a (k * n)) (k * p)) :
  ( ∀ n, n ≥ 2 → a n ^ 2 - a (n - 1) ^ 2 = p →
    ∃ d, ∀ n, a (n + 1) ^ 2 - a n ^ 2 = d ) ∧
  (eq_square_diff_seq (λ n, (-1) ^ n) 2) ∧
  (h2 k k) ∧
  ( (eq_square_diff_seq a p) ∧ ( ∃ d, ∀ n, a (n + 1) - a n = d ) →
    ∀ n, a n = a 0 )
:= by
  sorry

end problem_statements_l25_25227


namespace no_solution_abs_eq_2_l25_25517

theorem no_solution_abs_eq_2 (x : ℝ) :
  |x - 5| = |x + 3| + 2 → false :=
by sorry

end no_solution_abs_eq_2_l25_25517


namespace functions_same_l25_25797

theorem functions_same (x : ℝ) : (∀ x, (y = x) → (∀ x, (y = (x^3 + x) / (x^2 + 1)))) :=
by sorry

end functions_same_l25_25797


namespace geom_sequence_third_term_l25_25619

theorem geom_sequence_third_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a n = a 1 * r ^ (n - 1)) (h_cond : a 1 * a 5 = a 3) : a 3 = 1 :=
sorry

end geom_sequence_third_term_l25_25619


namespace find_a_l25_25221

-- Define the given conditions:
def curve_C1 (a t : ℝ) (ht : 0 < a) : ℝ × ℝ :=
  (a * cos t, 1 + a * sin t)

def curve_C2 (θ : ℝ) : ℝ :=
  4 * cos θ

def line_C3 (α0 : ℝ) (hα0 : tan α0 = 2) : ℝ → ℝ :=
  λ x, 2 * x

-- Define the proof problem:
theorem find_a
  (a : ℝ) (ht : 0 < a)
  (common_points_C1_C2_on_C3 : ∀ (θ t : ℝ), 
    curve_C2 θ = curve_C1 a t ht 
    → curve_C1 a t ht = curve_C3 α0 hα0 (curve_C1 a t ht).1) : 
  a = 1 :=
sorry

end find_a_l25_25221


namespace monic_polynomials_have_real_solution_l25_25459

theorem monic_polynomials_have_real_solution (P Q : Polynomial ℝ) 
  (hP : P.monic) (hQ : Q.monic) (hdegP : P.degree = 10) (hdegQ : Q.degree = 10) 
  (hPQ : ∀ x : ℝ, P.eval x ≠ Q.eval x) : ∃ x : ℝ, (P.comp (Polynomial.X + 1)).eval x = (Q.comp (Polynomial.X - 1)).eval x :=
sorry

end monic_polynomials_have_real_solution_l25_25459


namespace distance_from_point_to_parabola_directrix_l25_25100

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25100


namespace poles_needed_l25_25842

theorem poles_needed (length longer_side: ℝ) (length shorter_side: ℝ) 
    (dist_longer_side: ℝ) (dist_shorter_side: ℝ) :
    longer_side = 90 → shorter_side = 70 → dist_longer_side = 4 → dist_shorter_side = 5 →
    (Int.ceil (longer_side / dist_longer_side) * 2 + Int.ceil (shorter_side / dist_shorter_side) * 2 - 4) = 70 :=
by {
    intros h_longer h_shorter h_dist_longer h_dist_shorter,
    rw [h_longer, h_shorter, h_dist_longer, h_dist_shorter],
    sorry
}

end poles_needed_l25_25842


namespace algebraic_simplification_value_for_smallest_positive_integer_l25_25441

theorem algebraic_simplification (a : ℝ) :
  3 * a^2 - (5 * a - (1 / 2 * a - 3) + 2 * a^2) = a^2 - (9 / 2) * a - 3 := sorry

theorem value_for_smallest_positive_integer :
  let a := 1 in
    a^2 - (9 / 2) * a - 3 = - (13 / 2) := sorry

end algebraic_simplification_value_for_smallest_positive_integer_l25_25441


namespace number_of_ways_to_choose_bases_l25_25545

theorem number_of_ways_to_choose_bases : ∀ (students bases : ℕ), students = 4 → bases = 4 → (bases^students) = 256 :=
by
  intros students bases h_students h_bases
  rw [h_students, h_bases]
  exact pow_succ' 4 3

end number_of_ways_to_choose_bases_l25_25545


namespace ellipse_major_axis_min_length_l25_25159

theorem ellipse_major_axis_min_length (a b c : ℝ) 
  (h1 : b * c = 2)
  (h2 : a^2 = b^2 + c^2) 
  : 2 * a ≥ 4 :=
sorry

end ellipse_major_axis_min_length_l25_25159


namespace race_outcomes_l25_25495

theorem race_outcomes : 
  let participants := ["Anna", "Abe", "Bobby", "Charles", "Devin", "Edwin", "Fiona"] in
  (∀ l1 l2 l3 : String, l1 ∈ participants ∧ l2 ∈ participants ∧ l3 ∈ participants ∧
   l1 ≠ l2 ∧ l1 ≠ l3 ∧ l2 ≠ l3 ∧ ("Bobby" ∈ [l1, l2, l3]) ->
   (∀ outcomes_set : Finset (String × String × String), (l1, l2, l3) ∉ outcomes_set) ∧
   outcomes_set.card = 3) →
  90 :=
by
  sorry

end race_outcomes_l25_25495


namespace decreasing_intervals_l25_25586

def f (x : ℝ) : ℝ := 1 + x / 2 - sin x

theorem decreasing_intervals : 
  (∀ x, 0 < x ∧ x < π / 3 → f x < f (x + ε) ∧ ε > 0) ∧ 
  (∀ x, 5 * π / 3 < x ∧ x < 2 * π → f x < f (x + ε) ∧ ε > 0) :=
sorry

end decreasing_intervals_l25_25586


namespace probability_not_passing_l25_25752

noncomputable def probability_of_passing : ℚ := 4 / 7

theorem probability_not_passing (h : probability_of_passing = 4 / 7) : 1 - probability_of_passing = 3 / 7 :=
by
  sorry

end probability_not_passing_l25_25752


namespace chord_length_circle_line_l25_25371

theorem chord_length_circle_line :
  let C : Point := (4, 1)
  let r : ℝ := 4
  let d : ℝ := 2 * Real.sqrt 3
  let line : Line := {slope := Real.sqrt 3, intercept := 1}
  let circle : Circle := {center := C, radius := r}
  circle.chord_length line = 4 :=
sorry

end chord_length_circle_line_l25_25371


namespace part1_part2_l25_25592

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x - (1 / 5) ^ x

-- First part: f(x₁) > f(x₂) for any x₁, x₂ ∈ [1, +∞) with x₁ < x₂
theorem part1 (x₁ x₂ : ℝ) (h1 : 1 ≤ x₁) (h2 : 1 ≤ x₂) (h3 : x₁ < x₂) : f x₁ > f x₂ :=
sorry

-- Second part: f(√(x₁ x₂)) > √(f(x₁) f(x₂)) for any x₁, x₂ ∈ [1, +∞) with x₁ < x₂
theorem part2 (x₁ x₂ : ℝ) (h1 : 1 ≤ x₁) (h2 : 1 ≤ x₂) (h3 : x₁ < x₂) : 
  f (Real.sqrt (x₁ * x₂)) > Real.sqrt (f x₁ * f x₂) :=
sorry

end part1_part2_l25_25592


namespace calculate_expression_l25_25499

theorem calculate_expression : 4 * 6 * 8 + 24 / 4 - 10 = 188 := by
  sorry

end calculate_expression_l25_25499


namespace tangent_line_at_P_range_of_a_l25_25971

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := a * (x - 1/x) - Real.log x

-- Problem (Ⅰ): Tangent line equation at P(1, f(1)) for a = 1
theorem tangent_line_at_P (x : ℝ) (h : x = 1) : (∃ y : ℝ, f x 1 = y ∧ x - y - 1 = 0) := sorry

-- Problem (Ⅱ): Range of a for f(x) ≥ 0 ∀ x ≥ 1
theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, x ≥ 1 → f x a ≥ 0) : a ≥ 1/2 := sorry

end tangent_line_at_P_range_of_a_l25_25971


namespace concurrency_and_concurrence_point_Q_on_OI_l25_25925

open EuclideanGeometry

variable {A B C I O : Point}

axiom incenter : incenterTriangle A B C I
axiom circumcenter : circumcenterTriangle A B C O

-- $\Gamma_A$ passes through $B$ and $C$ and is tangent to the incircle
axiom GammaA_tangent : ∃ (Gamma_A : Circle), Gamma_A.passes_through B ∧ Gamma_A.passes_through C ∧ tangent Gamma_A (incircle A B C)

-- Similar definitions for $\Gamma_B$ and $\Gamma_C$
axiom GammaB_tangent : ∃ (Gamma_B : Circle), Gamma_B.passes_through A ∧ Gamma_B.passes_through C ∧ tangent Gamma_B (incircle A B C)
axiom GammaC_tangent : ∃ (Gamma_C : Circle), Gamma_C.passes_through A ∧ Gamma_C.passes_through B ∧ tangent Gamma_C (incircle A B C)

-- Intersection points $A'$, $B'$, and $C'$
axiom Aprime_def : ∃ (A' : Point), (GammaB ∩ GammaC = {A, A'})
axiom Bprime_def : ∃ (B' : Point), (GammaA ∩ GammaC = {B, B'})
axiom Cprime_def : ∃ (C' : Point), (GammaA ∩ GammaB = {C, C'})

-- The main theorem we are proving
theorem concurrency_and_concurrence_point_Q_on_OI :
  ∃ (Q : Point), concurrents [line_through A A', line_through B B', line_through C C'] ∧ on_line Q (line_through O I) :=
by
  -- We need a proof here
  sorry

end concurrency_and_concurrence_point_Q_on_OI_l25_25925


namespace perimeter_correct_l25_25369

-- Definitions based on the conditions
def large_rectangle_area : ℕ := 12 * 12
def shaded_rectangle_area : ℕ := 6 * 4
def non_shaded_area : ℕ := large_rectangle_area - shaded_rectangle_area
def perimeter_of_non_shaded_region : ℕ := 2 * ((12 - 6) + (12 - 4))

-- The theorem to prove
theorem perimeter_correct (large_rectangle_area_eq : large_rectangle_area = 144) :
  perimeter_of_non_shaded_region = 28 :=
by
  sorry

end perimeter_correct_l25_25369


namespace count_three_digit_numbers_no_one_l25_25186

theorem count_three_digit_numbers_no_one : 
  let count_numbers_without_one (limit : Nat) : Nat := 
    let is_valid (n : Nat) : Bool := 
      let digits := Nat.digits 10 n
      digits.all (λ d => d ≠ 1)
    (Nat.range limit).filter is_valid |>.length
  in 
  count_numbers_without_one 500 - count_numbers_without_one 100 = 243 :=
by
  sorry

end count_three_digit_numbers_no_one_l25_25186


namespace problem_part1_problem_part2_l25_25587

def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 4 + 2 * (Real.cos x) ^ 4 + (Real.cos (2 * x)) ^ 2 - 3
def g (x : ℝ) : ℝ := Real.cos (2 * x + π / 3) - 1

theorem problem_part1 :
  (∀ x, f x = Real.cos (4 * x) - 1) ∧
  (∃ T, T = π / 2 ∧ (∀ x, f (x + T) = f x)) ∧
  (∀ k : ℤ, ∀ x, 4 * x = k * π → f x = f (k * π / 4)) ∧
  (∀ k : ℤ, ∀ x, (k * π / 2 ≤ x ∧ x ≤ π / 4 + k * π / 2) → f x ≤ f (π / 4 + k * π / 2)) := sorry

theorem problem_part2 :
  (∀ x ∈ Set.Icc (-π / 4) (π / 6), f x = g x) ∧
  (∃ x, x ∈ Set.Icc (-π / 4) (π / 6) ∧ g x = -3 / 2 ∧ x = π / 6) := sorry

end problem_part1_problem_part2_l25_25587


namespace transformed_function_is_cosine_period_of_f_neither_odd_nor_even_l25_25591

noncomputable def f := λ x : ℝ, Real.cos (2 * x - π / 5)

theorem transformed_function_is_cosine :
  ∀ x : ℝ, (f (x / 2 + π / 10)) = Real.cos x := 
by 
  sorry

theorem period_of_f :
  Real.periodic f π :=
by 
  sorry

theorem neither_odd_nor_even :
  ¬ (∀ x : ℝ, f (-x) = f x ∨ f (-x) = -f x) :=
by 
  sorry

end transformed_function_is_cosine_period_of_f_neither_odd_nor_even_l25_25591


namespace factorize_expression_l25_25526

theorem factorize_expression (x : ℝ) : 4 * x ^ 2 - 2 * x = 2 * x * (2 * x - 1) :=
by
  sorry

end factorize_expression_l25_25526


namespace jared_yearly_earnings_l25_25913

theorem jared_yearly_earnings (monthly_pay_diploma : ℕ) (multiplier : ℕ) (months_in_year : ℕ)
  (h1 : monthly_pay_diploma = 4000) (h2 : multiplier = 3) (h3 : months_in_year = 12) :
  (monthly_pay_diploma * multiplier * months_in_year) = 144000 :=
by
  -- The proof goes here
  sorry

end jared_yearly_earnings_l25_25913


namespace tournament_partition_or_hamiltonian_cycle_l25_25216

-- Define the tournament graph and the associated properties and conditions
variable {α : Type} [fintype α] (T : simple_directed_graph α)
variable [decidable_rel T.adj]

-- A tournament graph is a directed graph where every pair of vertices is connected by exactly one directed edge
def is_tournament (G : simple_directed_graph α) : Prop :=
∀ v₁ v₂ : α, v₁ ≠ v₂ → (G.adj v₁ v₂ ∨ G.adj v₂ v₁) ∧ ¬ (G.adj v₁ v₂ ∧ G.adj v₂ v₁)

-- The problem statement
theorem tournament_partition_or_hamiltonian_cycle (n : ℕ) (T : simple_directed_graph α)
  (hT : is_tournament T) : 
  ∃ A B : finset α, (∀ a ∈ A, ∀ b ∈ B, T.adj a b) ∨ ∃ (l : list α), l.length = n ∧ (∀ (x y : α), x ≠ y → 
  (list.is_adjacent x y l ↔ T.adj x y)) := sorry

end tournament_partition_or_hamiltonian_cycle_l25_25216


namespace find_a_l25_25596

variable {x : ℝ} {a b : ℝ}

def setA : Set ℝ := {x | Real.log x / Real.log 2 > 1}
def setB (a : ℝ) : Set ℝ := {x | x < a}
def setIntersection (b : ℝ) : Set ℝ := {x | b < x ∧ x < 2 * b + 3}

theorem find_a (h : setA ∩ setB a = setIntersection b) : a = 7 := 
by
  sorry

end find_a_l25_25596


namespace sum_exterior_angles_dodecagon_l25_25393

theorem sum_exterior_angles_dodecagon : 
  (∀ (n : ℕ), n = 12 → ∑ (i : ℕ) in (finset.range n), exterior_angle = 360) :=
by
  sorry

end sum_exterior_angles_dodecagon_l25_25393


namespace water_price_increase_l25_25820

def residential_water_price (price_last_december price_this_may extra_water usage_last_december price_per_cubic_last_year : ℝ) : Prop :=
  (price_this_may / (1 + 0.2) / price_per_cubic_last_year - price_last_december / price_per_cubic_last_year = extra_water)

theorem water_price_increase (H1 : ∀ (price_last_december price_this_may extra_water usage_last_december price_per_cubic_last_year : ℝ),
  residential_water_price price_last_december price_this_may extra_water usage_last_december price_per_cubic_last_year) :
  (price_per_cubic_last_year = 5) → (1 + 0.2) * price_per_cubic_last_year = 6 :=
by {
  intros,
  assume h1 : price_per_cubic_last_year = 5,
  sorry
}

end water_price_increase_l25_25820


namespace not_necessarily_true_l25_25987

theorem not_necessarily_true (x y : ℝ) (h : x > y) : ¬ (x^2 > y^2) :=
sorry

end not_necessarily_true_l25_25987


namespace distance_from_A_to_directrix_l25_25001

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25001


namespace which_is_negative_l25_25796

theorem which_is_negative
    (A : ℤ := 2023)
    (B : ℤ := -2023)
    (C : ℚ := 1/2023)
    (D : ℤ := 0) :
    B < 0 :=
by
  sorry

end which_is_negative_l25_25796


namespace smallest_n_has_144_divisors_and_10_consecutive_integers_l25_25534

noncomputable theory
open scoped BigOperators

theorem smallest_n_has_144_divisors_and_10_consecutive_integers :
  ∃ (n : ℕ), (smallest n) ∧ (number_of_divisors n = 144) ∧ (consecutive_divisors n 10) :=
begin
  apply exists.intro 110880,
  exact ⟨smallest 110880, number_of_divisors_eq_144 110880, ten_consecutive_divisors 110880⟩,
  sorry
end

end smallest_n_has_144_divisors_and_10_consecutive_integers_l25_25534


namespace part_a_part_b_l25_25502

-- Definitions from conditions
def X := {f : ℝ → ℤ // ∀ x, f x = ⌊x⌋}

-- Part (a) statement
theorem part_a : ∃ E : (ℝ → ℝ) → Prop, 
  (∃ f : ℝ → ℝ, E f ∧ (∀ x, f(x) = ⌊x⌋)) ∧ 
  (∀ f : ℝ → ℝ, E f → (∀ x, f(x) = ⌊x⌋)) :=
sorry

-- Part (b) statement
theorem part_b : ∃ E : (ℝ → ℝ) → Prop,  (∃! f : ℝ → ℝ, E f ∧ (∀ x, f(x) = ⌊x⌋)) :=
sorry

end part_a_part_b_l25_25502


namespace weighted_average_profit_percentage_calculation_l25_25392

theorem weighted_average_profit_percentage_calculation :
  let quantity_A := 5
  let cost_A := 900
  let profit_A := 300
  let quantity_B := 8
  let cost_B := 1200
  let profit_B := 400
  let quantity_C := 3
  let cost_C := 1500
  let profit_C := 500
  let total_cost := (quantity_A * cost_A) + (quantity_B * cost_B) + (quantity_C * cost_C)
  let total_profit := (quantity_A * profit_A) + (quantity_B * profit_B) + (quantity_C * profit_C)
  let weighted_average_profit_percentage := (total_profit * 100) / total_cost
  weighted_average_profit_percentage = 33.33 :=
begin
  sorry
end

end weighted_average_profit_percentage_calculation_l25_25392


namespace fruit_basket_count_l25_25983

theorem fruit_basket_count : 
  let apple_options := 7 in
  let orange_options := 13 in
  apple_options * orange_options - 1 = 90 :=
by
  let apple_options := 7
  let orange_options := 13
  let total_options := apple_options * orange_options
  have h1: total_options = 91 := rfl -- because 7 * 13 = 91
  have h2: total_options - 1 = 90 := by simp [h1]
  exact h2

end fruit_basket_count_l25_25983


namespace infinite_N_for_factorization_l25_25319

noncomputable def is_special_number (k : ℤ) : ℕ :=
  4 * k ^ 4 - 8 * k ^ 2 + 2

theorem infinite_N_for_factorization : 
  ∀ k : ℤ, ∃ (P Q : Polynomial ℤ), Polynomial.degree P = 4 ∧ Polynomial.degree Q = 4 
          ∧ Polynomial.mul P Q = Polynomial.C 1 + Polynomial.C (is_special_number k) * Polynomial.X^4 + Polynomial.X^8 :=
by sorry

end infinite_N_for_factorization_l25_25319


namespace distance_from_point_to_parabola_directrix_l25_25101

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25101


namespace distance_from_A_to_directrix_l25_25012

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25012


namespace infinite_natural_numbers_factorable_polynomial_l25_25335

theorem infinite_natural_numbers_factorable_polynomial :
  ∃ᶠ N in at_top, ∃ (k : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧
    ∃ (f g : ℤ[x]), (degree f = 4) ∧ (degree g = 4) ∧ (f * g = X^8 + C (N : ℤ) * X^4 + 1) :=
sorry

end infinite_natural_numbers_factorable_polynomial_l25_25335


namespace infinite_natural_numbers_factorable_polynomial_l25_25333

theorem infinite_natural_numbers_factorable_polynomial :
  ∃ᶠ N in at_top, ∃ (k : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧
    ∃ (f g : ℤ[x]), (degree f = 4) ∧ (degree g = 4) ∧ (f * g = X^8 + C (N : ℤ) * X^4 + 1) :=
sorry

end infinite_natural_numbers_factorable_polynomial_l25_25333


namespace scientific_notation_of_GDP_l25_25485

theorem scientific_notation_of_GDP
  (b : ℕ) (billion_val : b = 10^9) :
  ∀ (n : ℕ) (GDP_billion : n = 53100), 
  let GDP_scientific := (5.31 : ℝ) * 10^13 in
  (n * b : ℝ) = GDP_scientific := 
by
  intros
  unfold billion_val GDP_billion
  unfold GDP_scientific
  sorry

end scientific_notation_of_GDP_l25_25485


namespace collinear_probability_l25_25668

theorem collinear_probability (O A B C D : Type) [decidable_eq Type] [fintype Type] (is_square : is_square_center O A B C D) :
  probability_collinear_points (O, A, B, C, D) = 1/5 := 
sorry

end collinear_probability_l25_25668


namespace range_of_a_l25_25994

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, 2 * a * (x : ℝ)^2 - 4 * (x : ℝ) < a * (x : ℝ) - 2 → ∃! x₀ : ℤ, x₀ = x) → 1 ≤ a ∧ a < 2 :=
sorry

end range_of_a_l25_25994


namespace proof_GP_product_l25_25509

namespace GPProof

variables {a r : ℝ} {n : ℕ} (S S' P : ℝ)

def isGeometricProgression (a r : ℝ) (n : ℕ) :=
  ∀ i, 0 ≤ i ∧ i < n → ∃ k, ∃ b, b = (-1)^k * a * r^k ∧ k = i 

noncomputable def product (a r : ℝ) (n : ℕ) : ℝ :=
  a^n * r^(n*(n-1)/2) * (-1)^(n*(n-1)/2)

noncomputable def sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - (-r)^n) / (1 - (-r))

noncomputable def reciprocalSum (a r : ℝ) (n : ℕ) : ℝ :=
  (1 / a) * (1 - (-1/r)^n) / (1 + 1/r)

theorem proof_GP_product (hyp1 : isGeometricProgression a (-r) n) (hyp2 : S = sum a (-r) n) (hyp3 : S' = reciprocalSum a (-r) n) (hyp4 : P = product a (-r) n) :
  P = (S / S')^(n/2) :=
by
  sorry

end GPProof

end proof_GP_product_l25_25509


namespace inequality1_inequality2_l25_25715

theorem inequality1 (x : ℝ) : 2 * x - 1 > x - 3 → x > -2 := by
  sorry

theorem inequality2 (x : ℝ) : 
  (x - 3 * (x - 2) ≥ 4) ∧ ((x - 1) / 5 < (x + 1) / 2) → -7 / 3 < x ∧ x ≤ 1 := by
  sorry

end inequality1_inequality2_l25_25715


namespace arithmetic_sequence_sum_l25_25500

theorem arithmetic_sequence_sum :
  ∀ (a₁ : ℕ) (d : ℕ) (a_n : ℕ) (n : ℕ),
    a₁ = 1 →
    d = 2 →
    a_n = 29 →
    a_n = a₁ + (n - 1) * d →
    (n : ℕ) = 15 →
    (∑ k in Finset.range n, a₁ + k * d) = 225 :=
by
  intros a₁ d a_n n h₁ h_d hₐ h_an h_n
  sorry

end arithmetic_sequence_sum_l25_25500


namespace find_original_number_l25_25488

-- Defining the conditions as given in the problem
def original_number_condition (x : ℤ) : Prop :=
  3 * (3 * x - 6) = 141

-- Stating the main theorem to be proven
theorem find_original_number (x : ℤ) (h : original_number_condition x) : x = 17 :=
sorry

end find_original_number_l25_25488


namespace distance_from_A_to_directrix_l25_25071

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25071


namespace stirling_approx_100_factorial_digits_l25_25363

theorem stirling_approx_100_factorial_digits :
  let stirling_approx := sqrt (2 * Real.pi * 100) * (100 / Real.exp 1)^100 in
  log10 stirling_approx ≈ 157.999 →
  nat.floor (log10 stirling_approx) + 1 = 158 :=
begin
  intros stirling_approx h,
  sorry
end

end stirling_approx_100_factorial_digits_l25_25363


namespace characterizeGoodRational_l25_25769

def isCoprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def goodRational (x : ℚ) : Prop :=
  ∃ (p q : ℕ) (hpq : isCoprime p q) (hgt : x = (p : ℚ) / q) (hq_pos : 0 < q) (h1 : 1 < x)
  (α : ℚ) (N : ℕ), ∀ n ≥ N, |x^n.fract - α| ≤ 1 / (2 * (p + q))

theorem characterizeGoodRational (x : ℚ) (hx : goodRational x) : 
  ∃ k : ℤ, x = k ∧ 1 < k :=
sorry

end characterizeGoodRational_l25_25769


namespace problem_solution_l25_25620

def isPythagoreanSumNumber (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  c^2 + d^2 = 10 * a + b
  
def G (M : ℕ) : ℤ :=
  let c := (M / 10) % 10
  let d := M % 10
  (c + d) / 9

def P (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  (10 * a - 2 * c * d + b) / 3

theorem problem_solution :
  ¬isPythagoreanSumNumber 2023 ∧
  isPythagoreanSumNumber 5055 ∧
  ∃ (M : ℕ), isPythagoreanSumNumber M ∧ 
             M ≠ 2543 ∧ 
             M ≠ 4325 ∧ 
             M ≠ 1323 ∧
             G M ∈ ℤ ∧
             P M = 3 ∧
             (M = 3772 ∨ M = 3727) :=
by
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

end problem_solution_l25_25620


namespace complex_inequality_l25_25937

open Complex

theorem complex_inequality (r : ℝ) (n : ℕ) (z : ℕ → ℂ) (h_r : 0 < r ∧ r < 1) 
  (h_z : ∀ k : ℕ, k < n → abs (z k - 1) ≤ r) :
  abs (∑ k in Finset.range n, z k) * abs (∑ k in Finset.range n, 1 / z k) ≥ n^2 * (1 - r^2) :=
sorry

end complex_inequality_l25_25937


namespace max_subset_cardinality_l25_25175

theorem max_subset_cardinality (A : Finset ℕ) (hA1 : ∀ a b ∈ A, ¬Nat.coprime a b) (hA2 : ∀ a b ∈ A, ¬(a ∣ b ∨ b ∣ a)) (hA3 : A ⊆ Finset.range 2018) :
  A.card ≤ 504 :=
sorry

end max_subset_cardinality_l25_25175


namespace total_votes_cast_l25_25270

theorem total_votes_cast (S : ℝ) (x : ℝ) (h1 : S = 120) (h2 : S = 0.72 * x - 0.28 * x) : x = 273 := by
  sorry

end total_votes_cast_l25_25270


namespace count_arrangements_units_digit_one_l25_25747

/--
The numbers 1, 2, 3, 4, and 5 are used once each in some order substituting for the letters 
in the series of powers M^{(A^{(T^S)})}. In how many of the arrangements is the units digit of the 
value of this expression equal to 1?
-/
theorem count_arrangements_units_digit_one :
  let digits : List ℕ := [1, 2, 3, 4, 5]
  let arrangements : List (ℕ × ℕ × ℕ × ℕ) := digits.permutations.map (λ l, (l.head!, l.nth! 1, l.nth! 2, l.nth! 3, l.nth! 4))
  let valid_units_digit_one (m a t s : ℕ) : Bool := 
    ((m ^ (a ^ (t ^ s))) % 10 = 1)
  List.count (λ arr, valid_units_digit_one arr.1 arr.2 arr.3 arr.4) arrangements = 34 :=
by
  sorry

end count_arrangements_units_digit_one_l25_25747


namespace _l25_25140

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25140


namespace solve_sausage_problem_l25_25832

def sausage_problem (x y : ℕ) (condition1 : y = x + 300) (condition2 : x = y + 500) : Prop :=
  x + y = 2 * 400

theorem solve_sausage_problem (x y : ℕ) (h1 : y = x + 300) (h2 : x = y + 500) :
  sausage_problem x y h1 h2 :=
by
  sorry

end solve_sausage_problem_l25_25832


namespace circumscribed_quad_angle_sum_l25_25356

noncomputable def is_circumscribed (ABCD : Type) (O : Point) :=
  ∃ M N P Q : Point, tangent_to_circle_AT ABCD O ∧
    tangent_to_circle_AT ABCD O ∧
    tangent_to_circle_AT ABCD O ∧
    tangent_to_circle_AT ABCD O

theorem circumscribed_quad_angle_sum (ABCD : Type) (O : Point)
  (h_circumscribed : is_circumscribed ABCD O) : 
  ∃ α β γ δ : Angle, 
    ∠AOB = α + δ ∧
    ∠COD = β + γ ∧
    α + β + γ + δ = 180° → 
    ∠AOB + ∠COD = 180° :=
begin
  sorry
end

end circumscribed_quad_angle_sum_l25_25356


namespace area_of_EFGH_l25_25266

variables {A B C D E F G H : Point}
variables (ABCD : Quadrilateral) (area_ABCD : area ABCD = 1)
variables (div_AB : Point → Point → Point → Prop)
variables (div_DC : Point → Point → Point → Prop)

-- Assuming div_AB describes the division such that AE = EF = FB
-- Assuming div_DC describes the division such that DH = HG = GC
def div_three (A B P Q R: Point) : Prop :=
  div_AB A P Q ∧ div_AB Q R B ∧ div_DC D H G ∧ div_DC G C

theorem area_of_EFGH (hdiv: div_three A B E F A C H G) :
  area (Quadrilateral.mk E F G H) = 1/3 := by 
  sorry

end area_of_EFGH_l25_25266


namespace speed_of_water_l25_25838

-- Definitions based on conditions
def swim_speed_in_still_water : ℝ := 4
def distance_against_current : ℝ := 6
def time_against_current : ℝ := 3
def effective_speed (v : ℝ) : ℝ := swim_speed_in_still_water - v

-- Theorem to prove the speed of the water
theorem speed_of_water (v : ℝ) : 
  effective_speed v * time_against_current = distance_against_current → 
  v = 2 :=
by
  sorry

end speed_of_water_l25_25838


namespace jared_yearly_earnings_l25_25912

theorem jared_yearly_earnings (monthly_pay_diploma : ℕ) (multiplier : ℕ) (months_in_year : ℕ)
  (h1 : monthly_pay_diploma = 4000) (h2 : multiplier = 3) (h3 : months_in_year = 12) :
  (monthly_pay_diploma * multiplier * months_in_year) = 144000 :=
by
  -- The proof goes here
  sorry

end jared_yearly_earnings_l25_25912


namespace no_valid_number_l25_25430

theorem no_valid_number (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 9) : ¬ ∃ (y : ℕ), (x * 100 + 3 * 10 + y) % 11 = 0 :=
by
  sorry

end no_valid_number_l25_25430


namespace distance_to_directrix_l25_25087

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25087


namespace range_of_a_l25_25551

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (1 - 2 * a) * x + 3 * a else Real.log x

theorem range_of_a (a : ℝ) : (-1 ≤ a ∧ a < 1/2) ↔
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) :=
by
  sorry

end range_of_a_l25_25551


namespace max_expr_value_l25_25903

noncomputable def expr (x : ℝ) : ℝ :=
(e^(2 * x) + e^(-2 * x) + 1) / (e^x + e^(-x) + 2)

theorem max_expr_value : ∃ x : ℝ, expr x = -0.464 :=
sorry

end max_expr_value_l25_25903


namespace find_x_given_y_l25_25963

noncomputable def constantRatio : Prop :=
  ∃ k : ℚ, ∀ x y : ℚ, (5 * x - 6) / (2 * y + 10) = k

theorem find_x_given_y :
  (constantRatio ∧ (3, 2) ∈ {(x, y) | (5 * x - 6) / (2 * y + 10) = 9 / 14}) →
  ∃ x : ℚ, ((5 * x - 6) / 20 = 9 / 14 ∧ x = 53 / 14) :=
by
  sorry

end find_x_given_y_l25_25963


namespace exists_three_either_knowing_or_not_l25_25630

-- Define the concept of mutual acquaintance
def mutual_acquaintance (a b : nat) : Prop :=
  -- Placeholder for the mutual acquaintance relationship
  sorry

theorem exists_three_either_knowing_or_not (group_of_six : Finset ℕ)
  (h_size : group_of_six.card = 6)
  (mutual_acquaintance_hyp : ∀ x y ∈ group_of_six, mutual_acquaintance x y → mutual_acquaintance y x) :
  ∃ (a b c : ℕ), (a ∈ group_of_six ∧ b ∈ group_of_six ∧ c ∈ group_of_six) ∧
  ((mutual_acquaintance a b ∧ mutual_acquaintance b c ∧ mutual_acquaintance a c) ∨
  (¬mutual_acquaintance a b ∧ ¬mutual_acquaintance b c ∧ ¬mutual_acquaintance a c)) :=
sorry

end exists_three_either_knowing_or_not_l25_25630


namespace integral_eq_exp2_l25_25518

theorem integral_eq_exp2 (a : ℝ) (h : ∫ x in 1..a, (2/x) = 4) : a = Real.exp 2 := 
by {
  sorry
}

end integral_eq_exp2_l25_25518


namespace distance_from_A_to_directrix_on_parabola_l25_25050

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25050


namespace ship_people_first_hundred_years_l25_25828

theorem ship_people_first_hundred_years :
  ∃ S : ℕ, 
    let people_in_second_hundred_years := 2 * S,
        people_in_third_hundred_years := 4 * S,
        total_people := S + people_in_second_hundred_years + people_in_third_hundred_years
    in total_people = 847 ∧ S = 121 :=
by
  sorry

end ship_people_first_hundred_years_l25_25828


namespace quadratic_coefficients_l25_25387

theorem quadratic_coefficients :
  ∀ (a b c : ℤ), (2 * a * a - b * a - 5 = 0) → (a = 2 ∧ b = -1) :=
by
  intros a b c H
  sorry

end quadratic_coefficients_l25_25387


namespace infinite_N_for_factorization_l25_25318

noncomputable def is_special_number (k : ℤ) : ℕ :=
  4 * k ^ 4 - 8 * k ^ 2 + 2

theorem infinite_N_for_factorization : 
  ∀ k : ℤ, ∃ (P Q : Polynomial ℤ), Polynomial.degree P = 4 ∧ Polynomial.degree Q = 4 
          ∧ Polynomial.mul P Q = Polynomial.C 1 + Polynomial.C (is_special_number k) * Polynomial.X^4 + Polynomial.X^8 :=
by sorry

end infinite_N_for_factorization_l25_25318


namespace intersecting_squares_area_difference_l25_25381

theorem intersecting_squares_area_difference :
  let A := 12^2
  let B := 9^2
  let C := 7^2
  let D := 3^2
  (A + B) - (C + D) = 103 := by
oops

end intersecting_squares_area_difference_l25_25381


namespace distinct_positive_differences_count_l25_25979

theorem distinct_positive_differences_count : 
  let S := {2, 3, 7, 8, 9, 12}
  ∃ D : set ℕ, (∀ x ∈ S, ∀ y ∈ S, x ≠ y → (x > y → (x - y) ∈ D) ∨ (y > x → (y - x) ∈ D)) ∧ 
              (∀ d ∈ D, d > 0) ∧ 
              D.card = 9 
:= sorry

end distinct_positive_differences_count_l25_25979


namespace point_reflection_correct_l25_25811

def point_reflection_y_axis (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (-x, y, -z)

theorem point_reflection_correct :
  point_reflection_y_axis (-3) 5 2 = (3, 5, -2) :=
by
  -- The proof would go here
  sorry

end point_reflection_correct_l25_25811


namespace distance_from_A_to_directrix_of_parabola_l25_25066

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25066


namespace distance_from_point_to_parabola_directrix_l25_25099

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25099


namespace gcd_division_count_l25_25981

theorem gcd_division_count (a b : ℕ) (h₁ : a = 60) (h₂ : b = 48) : 
  nat.division_count 60 48 = 2 := 
sorry

end gcd_division_count_l25_25981


namespace distance_from_A_to_directrix_l25_25077

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25077


namespace total_length_of_T_l25_25512

def T := {p : ℝ × ℝ | abs (abs (abs p.1 - 3) - 2) + abs (abs (abs p.2 - 3) - 2) = 2}

theorem total_length_of_T : ∑ (p : ℝ × ℝ) in T, length p = 64 * real.sqrt 2 := 
sorry

end total_length_of_T_l25_25512


namespace closest_weight_total_shortfall_total_selling_price_l25_25403

-- Definitions
def standard_weight : ℝ := 25
def weights : List ℝ := [1.5, -3, 2, -0.5, 1, -2, -2.5, -2]
def price_per_kg : ℝ := 2.6

-- Assertions
theorem closest_weight : ∃ w ∈ weights, abs w = 0.5 ∧ 25 + w = 24.5 :=
by sorry

theorem total_shortfall : (weights.sum = -5.5) :=
by sorry

theorem total_selling_price : (8 * standard_weight + weights.sum) * price_per_kg = 505.7 :=
by sorry

end closest_weight_total_shortfall_total_selling_price_l25_25403


namespace minimum_value_of_M_l25_25202

theorem minimum_value_of_M 
  (n : Nat) 
  (α : Fin n → ℝ) 
  (M : ℝ)
  (h : ∀ α, cos (α 0) * cos (α 1) * ⋯ * cos (α (n-1)) + sin (α 0) * sin (α 1) * ⋯ * sin (α (n-1)) ≤ M) : 
  M ≥ sqrt 2 :=
sorry

end minimum_value_of_M_l25_25202


namespace polynomial_G_value_l25_25851

theorem polynomial_G_value :
  ∀ (z : ℕ → ℕ) (n : ℕ), 
  (∀ i, (z i).nat_decr) ∧ 
  (∀ i, z i = 0 → 0 < i → false) ∧
  (∑ i in range 7, z i) = 12 ∧ 
  (⦃roots, roots_val⦄) = polynomial.sum_roots_range := 
  let roots := (4 :: 2 :: 2 :: 2 :: 1 :: 1 :: 0 :: nil) in
  let roots_val := polynomial.combination_value roots n 3 in
  let h := polynomial.sum_contribution roots_val roots 3 in
  polynomial.G_value_from_sum h = -86 := by
  sorry

end polynomial_G_value_l25_25851


namespace infinite_nat_N_polynomial_l25_25312

/-- The theorem states that for any integer k, 
there exists a natural number N = 4k^4 - 8k^2 + 2 such that the polynomial 
x^8 + Nx^4 + 1 can be factored into two fourth-degree polynomials with integer coefficients. -/
theorem infinite_nat_N_polynomial (k : ℤ) : ∃ (N : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧ 
  ∃ (P Q : ℤ[X]), P.degree = 4 ∧ Q.degree = 4 ∧ (X^8 + C (↑N) * X^4 + 1) = P * Q :=
by {
  sorry
}

end infinite_nat_N_polynomial_l25_25312


namespace distance_to_directrix_l25_25088

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25088


namespace infinite_nat_N_polynomial_l25_25314

/-- The theorem states that for any integer k, 
there exists a natural number N = 4k^4 - 8k^2 + 2 such that the polynomial 
x^8 + Nx^4 + 1 can be factored into two fourth-degree polynomials with integer coefficients. -/
theorem infinite_nat_N_polynomial (k : ℤ) : ∃ (N : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧ 
  ∃ (P Q : ℤ[X]), P.degree = 4 ∧ Q.degree = 4 ∧ (X^8 + C (↑N) * X^4 + 1) = P * Q :=
by {
  sorry
}

end infinite_nat_N_polynomial_l25_25314


namespace fruit_fly_F1_distribution_l25_25220

-- Definitions based on the conditions
inductive EyeColor
| red
| white

inductive ChromosomeIV
| normal
| missing_one
| missing_both

structure FruitFlyParents :=
(genders : Type)
(female_heterozygous : bool)
(both_missing_one_chromosome_IV : bool)

def red_eyes_dominant_over_white : Prop := true
def gene_on_X_chromosome : Prop := true
def one_missing_chromosome_IV_ok : Prop := true
def missing_both_chromosome_IV_lethal : Prop := true 

-- Givens
axiom female_heterozygous: FruitFlyParents -> bool

axiom both_missing_one_chromosome_IV (p: FruitFlyParents): p.both_missing_one_chromosome_IV = true

-- Theorem to prove the correct answer
theorem fruit_fly_F1_distribution (parents: FruitFlyParents) :
  red_eyes_dominant_over_white ∧
  gene_on_X_chromosome ∧
  one_missing_chromosome_IV_ok ∧
  missing_both_chromosome_IV_lethal ∧
  parents.female_heterozygous = true ∧
  parents.both_missing_one_chromosome_IV = true ->
  ( ∃ f1_offspring : (EyeColor × ChromosomeIV),
    (f1_offspring = (EyeColor.red, ChromosomeIV.normal) ∧
     P(f1_offspring) = 1 / 4) ) :=
by
  sorry

end fruit_fly_F1_distribution_l25_25220


namespace sum_of_cubes_application_l25_25929

theorem sum_of_cubes_application : 
  ¬ ((a+1) * (a^2 - a + 1) = a^3 + 1) :=
by
  sorry

end sum_of_cubes_application_l25_25929


namespace infinite_natural_numbers_with_factored_polynomial_l25_25328

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l25_25328


namespace annulus_area_tetradecagon_l25_25727

noncomputable def radius_inscribed (a : ℝ) (n : ℕ) : ℝ := (a / 2.0) / Real.tan (Real.pi / n)
noncomputable def radius_circumscribed (a : ℝ) (n : ℕ) : ℝ := (a / 2.0) / Real.sin (Real.pi / n)
noncomputable def area_tetradecagon (a : ℝ) : ℝ := (7 * a^2) * Real.cot (Real.pi / 14)
def annulus_area (T : ℝ) (n : ℕ) : ℝ := (T * Real.pi) / n * Real.tan (Real.pi / n)

theorem annulus_area_tetradecagon :
  annulus_area 67.707 14 = 3.468 :=
by
  sorry

end annulus_area_tetradecagon_l25_25727


namespace distance_to_directrix_l25_25023

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25023


namespace part_one_part_two_l25_25943

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.exp (x - 1)
def g (x : ℝ) : ℝ := Real.log x - 1

-- Theorem 1: Prove that for all x > 0, f(x) ≥ g(x) + 2
theorem part_one (x : ℝ) (hx : 0 < x) : f x ≥ g x + 2 := 
sorry

-- Theorem 2: Determine the existence and count of common tangent lines
theorem part_two : ∃! (t₁ t₂ : ℝ), (t₁ ≠ t₂) ∧
    (∀ x, f x = Real.exp (t₁ - 1) * x + Real.exp (t₁ - 1) * (1 - t₁) →
    g x = Real.exp (t₂ - 1) * x + Real.exp (t₂ - 1) * (1 - t₂)) :=
sorry

end part_one_part_two_l25_25943


namespace algebra_expression_value_l25_25572

theorem algebra_expression_value (a b : ℝ)
  (h1 : |a + 2| = 0)
  (h2 : (b - 5 / 2) ^ 2 = 0) : (2 * a + 3 * b) * (2 * b - 3 * a) = 26 := by
sorry

end algebra_expression_value_l25_25572


namespace hypotenuse_length_l25_25158

theorem hypotenuse_length (a b : ℝ) (h₁ : a = 2) (h₂ : b = 3) :
  sqrt (a^2 + b^2) = sqrt 13 :=
by
  rw [h₁, h₂]
  sorry

end hypotenuse_length_l25_25158


namespace distance_from_A_to_directrix_l25_25006

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25006


namespace sum_of_integers_ending_in_1_between_50_and_350_l25_25759

theorem sum_of_integers_ending_in_1_between_50_and_350 :
  (∑ n in Finset.filter (λ x => x % 10 = 1) (Finset.Icc 50 350), n) = 5880 :=
by
  sorry

end sum_of_integers_ending_in_1_between_50_and_350_l25_25759


namespace sum_in_matrix_l25_25635

open Matrix

theorem sum_in_matrix (m n : ℕ) (A : Matrix (Fin m) (Fin n) ℝ) 
  (row_sum : ∀ i, ∑ j, A i j = 1)
  (col_sum : ∀ j, ∑ i, A i j = 1) :
  m = n :=
  sorry

end sum_in_matrix_l25_25635


namespace distance_from_A_to_directrix_l25_25008

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25008


namespace distance_from_A_to_directrix_on_parabola_l25_25038

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25038


namespace watermelon_transportation_l25_25398

theorem watermelon_transportation
  (a b x y z : ℝ)
  (h1 : x * z = y * (z - a))
  (h2 : b * x + (b - a) * y = (1/2) * (x * z + y * (z - a))) :
  z = b + Real.sqrt(b * (b - a)) :=
by
  sorry

end watermelon_transportation_l25_25398


namespace distance_from_A_to_EF_l25_25278

-- Definitions for trigonometrical functions and geometrical terms
def square_ABCD (a : ℝ) := 
  ∃ A B C D : ℝ × ℝ, 
  A = (0, 0) ∧ 
  B = (a, 0) ∧ 
  C = (a, a) ∧
  D = (0, a) ∧ 
  a > 0

def E_on_BC (a : ℝ) (E : ℝ × ℝ) := 
  ∃ y : ℝ, 
  E = (a, y) ∧ 
  0 ≤ y ∧ 
  y < a -- Explicit non-strict bound to reflect "on the side"

def F_on_CD (a : ℝ) (F : ℝ × ℝ) := 
  ∃ x : ℝ, 
  F = (x, a) ∧ 
  0 ≤ x ∧ 
  x < a

def angle_EAB (A B E : ℝ × ℝ) := 
  ∃ α : ℝ, α = 20 ∧ 
  tan α = (E.2 - B.2) / (E.1 - B.1)

def angle_EAF (A E F : ℝ × ℝ) := 
  ∃ β : ℝ, β = 45 ∧ 
  tan β = (F.2 - E.2) / (F.1 - E.1)

def distance_point_line (A : ℝ × ℝ) (EF : (ℝ × ℝ) × (ℝ × ℝ)) :=
  let AEF := (λ x : ℝ, x / tan 20) in
  let YEF := (λ x : ℝ, 1 - x / tan 20) in
  ∃ d : ℝ,
  d = abs (tan 20 - (1 / tan 20)) /
  sqrt ((1 - 2 * tan 20)^2 / tan(20)^2 + 1)

-- Main statement "prove that the distance from A to the line EF is as above"
theorem distance_from_A_to_EF :
  ∀ (A B C D E F : ℝ × ℝ),  
  ∀ a : ℝ,
  square_ABCD a →
  E_on_BC a E →
  F_on_CD a F → 
  angle_EAB A B E →
  angle_EAF A E F →
  distance_point_line A (E, F)
:= by sorry

end distance_from_A_to_EF_l25_25278


namespace wuzhen_conference_arrangements_l25_25520

theorem wuzhen_conference_arrangements 
  (countries : Finset ℕ)
  (hotels : Finset ℕ)
  (h_countries_count : countries.card = 5)
  (h_hotels_count : hotels.card = 3) :
  ∃ f : ℕ → ℕ,
  (∀ c ∈ countries, f c ∈ hotels) ∧
  (∀ h ∈ hotels, ∃ c ∈ countries, f c = h) ∧
  (Finset.card (Set.toFinset (f '' countries)) = 3) ∧
  ∃ n : ℕ,
  n = 150 := 
sorry

end wuzhen_conference_arrangements_l25_25520


namespace purchase_price_l25_25452

-- Define the context and conditions 
variables (P S : ℝ)
-- Define the conditions
axiom cond1 : S = P + 0.5 * S
axiom cond2 : S - P = 100

-- Define the main theorem
theorem purchase_price : P = 100 :=
by sorry

end purchase_price_l25_25452


namespace simon_age_in_2010_l25_25238

theorem simon_age_in_2010 :
  ∀ (s j : ℕ), (j = 16 → (j + 24 = s) → j + (2010 - 2005) + 24 = 45) :=
by 
  intros s j h1 h2 
  sorry

end simon_age_in_2010_l25_25238


namespace equation_represents_one_line_l25_25200

-- Problem statement formalized in Lean 4
theorem equation_represents_one_line (x y k : ℝ) (h : x + y - 6 * real.sqrt (x + y) + 3 * k = 0) :
  k = 3 ∨ k < 0 := 
sorry

end equation_represents_one_line_l25_25200


namespace circumcircle_fixed_point_l25_25382

-- Define the setup and specific conditions of the problem
theorem circumcircle_fixed_point {A B C P I_B I_C M : Type*} 
  (h1 : is_isosceles_triangle A B C)
  (h2 : is_inscribed_triangle A B C ω)
  (h3 : is_point_on_arc P B C ω)
  (h4 : incenter I_B A B P)
  (h5 : incenter I_C A C P)
  (hM : is_arc_midpoint M B C ω) :
  passes_through_fixed_point (circumcircle P I_B I_C) M :=
sorry

end circumcircle_fixed_point_l25_25382


namespace cos_beta_l25_25556

theorem cos_beta (
  α β : ℝ 
) (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : cos α = 4 / 5)
  (h4 : tan (α - β) = -1 / 3) : 
  cos β = 9 * sqrt 10 / 50 := 
sorry

end cos_beta_l25_25556


namespace inequality_solution_set_l25_25716

open Set

noncomputable def rational_expression (x : ℝ) : ℝ := (x^2 - 16) / (x^2 + 10*x + 25)

theorem inequality_solution_set :
  {x : ℝ | rational_expression x < 0} = Ioo (-4 : ℝ) 4 :=
by
  sorry

end inequality_solution_set_l25_25716


namespace car_speeds_and_arrival_times_l25_25871

theorem car_speeds_and_arrival_times
  (x y z u : ℝ)
  (h1 : x^2 = (y + z) * u)
  (h2 : (y + z) / 4 = u)
  (h3 : x / u = y / z)
  (h4 : x + y + z + u = 210) :
  x = 60 ∧ y = 80 ∧ z = 40 ∧ u = 30 := 
by
  sorry

end car_speeds_and_arrival_times_l25_25871


namespace cube_sum_eq_one_l25_25722

theorem cube_sum_eq_one (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 2) (h3 : abc = 1) : a^3 + b^3 + c^3 = 1 :=
sorry

end cube_sum_eq_one_l25_25722


namespace find_N_l25_25607

/--
If 15% of N is 45% of 2003, then N is 6009.
-/
theorem find_N (N : ℕ) (h : 15 / 100 * N = 45 / 100 * 2003) : 
  N = 6009 :=
sorry

end find_N_l25_25607


namespace poly_factorable_l25_25304

-- Define the specific form of N.
noncomputable def N (k : ℤ) : ℕ := 4 * k^4 - 8 * k^2 + 2

-- Define the polynomial.
noncomputable def poly (N : ℕ) : ℤ[X] := X^8 + N * X^4 + 1

-- Define the fourth-degree polynomials used in the factorization.
noncomputable def poly1 (k : ℤ) : ℤ[X] :=
  X^4 - 2 * k * X^3 + 2 * k^2 * X^2 - 2 * k * X + 1

noncomputable def poly2 (k : ℤ) : ℤ[X] := 
  X^4 + 2 * k * X^3 + 2 * k^2 * X^2 + 2 * k * X + 1

-- The proof statement: prove that the polynomial can be factored into the specified fourth-degree polynomials.
theorem poly_factorable : ∀ k : ℤ, ∃ p1 p2 : ℤ[X], poly (N k) = p1 * p2 :=
by
  intro k
  use (poly1 k)
  use (poly2 k)
  sorry

end poly_factorable_l25_25304


namespace a_seq_formula_b_seq_formula_c_seq_sum_formula_l25_25938

-- Defining the sequences a_n and b_n
def a_seq (n : ℕ) : ℕ := if n = 0 then 0 else 2 * n - 1

def b_seq (n : ℕ) : ℕ := 3^n

-- Proving the general term formulas for a_n and b_n
theorem a_seq_formula (n : ℕ) : a_seq n = if n = 0 then 0 else 2 * n - 1 :=
  sorry

theorem b_seq_formula (n : ℕ) : b_seq n = 3^n :=
  sorry

-- Proving the sum of the first n terms of c_n, where c_n satisfies the given condition with a_n and b_n
def c_seq (n : ℕ) : ℝ := 1 / ((2 * n - 1) * (2 * n + 1))

noncomputable def T_n (n : ℕ) : ℝ := (Finset.range n).sum (λ k, c_seq (k + 1))

theorem c_seq_sum_formula (n : ℕ) :
  (T_n n) = n / (2 * n + 1) :=
  sorry

end a_seq_formula_b_seq_formula_c_seq_sum_formula_l25_25938


namespace find_f_expression_l25_25610

theorem find_f_expression (f : ℝ → ℝ) (x : ℝ) (h : f (Real.log x) = 3 * x + 4) : 
  f x = 3 * Real.exp x + 4 := 
by
  sorry

end find_f_expression_l25_25610


namespace interval_strictly_increasing_find_B_and_a_l25_25969

namespace MathProof

def f (x : ℝ) : ℝ := (sqrt 3) * cos (2 * x - (2 * Real.pi / 3)) + 2 * sin (x - Real.pi / 12) ^ 2

theorem interval_strictly_increasing (x : ℝ) (k : ℤ) :
  k * Real.pi - (Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (5 * Real.pi / 12) ↔ 
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂ := sorry

theorem find_B_and_a (b c : ℝ) (B a : ℝ) (h_b : b = 3) (h_c : c = 3 * (sqrt 3)) 
  (h_func : f B = 1) (h_obtuse : ∠ABC > Real.pi / 2) : 
  B = Real.pi / 6 ∧ a = 3 := sorry

end MathProof

end interval_strictly_increasing_find_B_and_a_l25_25969


namespace remainder_x1998_l25_25433

noncomputable def k : ℝ := (1998 + Real.sqrt (1998 ^ 2 + 4)) / 2

def sequence : ℕ → ℕ
| 0       := 1
| (n + 1) := Real.floor (k * (sequence n))

theorem remainder_x1998 :
  (sequence 1998) % 1998 = 1000 := 
sorry

end remainder_x1998_l25_25433


namespace vector_magnitude_l25_25956

variables (a b : ℝ^3) 

-- Given conditions
noncomputable def angle_ab := 3 * Real.pi / 4
axiom mag_a : ∥a∥ = Real.sqrt 2
axiom mag_b : ∥b∥ = 3

-- The problem statement
theorem vector_magnitude (h : Real.angle a b = angle_ab) : ∥a + 2 • b∥ = Real.sqrt 26 :=
sorry

end vector_magnitude_l25_25956


namespace vector_magnitude_proof_l25_25958

noncomputable def vector_magnitude (v : ℝˣ ) : ℝ := 
  Real.sqrt (v.dot v)

theorem vector_magnitude_proof (a b : ℝˣ ) 
  (h_angle : ∀ (θ : ℝ), θ = 3 * Real.pi / 4 )
  (h1 : vector_magnitude a = Real.sqrt 2)
  (h2 : vector_magnitude b = 3)
  :
  vector_magnitude (a + (2*b)) = Real.sqrt 26 := 
sorry

end vector_magnitude_proof_l25_25958


namespace angular_speeds_proportion_l25_25544

variable {x y z w : ℕ}
variable {ω_A ω_B ω_C ω_D k : ℝ}

axiom teeth_counts_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0
axiom pairs_mesh : x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D

noncomputable def omega_proportion : Prop :=
  ∀ k > 0, ω_A = k / x ∧ ω_B = k / y ∧ ω_C = k / z ∧ ω_D = k / w → 
  ω_A : ω_B : ω_C : ω_D = y * z * w : x * z * w : x * y * w : x * y * z

theorem angular_speeds_proportion : omega_proportion :=
by
  sorry

end angular_speeds_proportion_l25_25544


namespace B_grazed_months_l25_25445

-- Define the conditions
variables (A_cows B_cows C_cows D_cows : ℕ)
variables (A_months B_months C_months D_months : ℕ)
variables (A_rent total_rent : ℕ)

-- Given conditions
def A_condition := (A_cows = 24 ∧ A_months = 3)
def B_condition := (B_cows = 10)
def C_condition := (C_cows = 35 ∧ C_months = 4)
def D_condition := (D_cows = 21 ∧ D_months = 3)
def A_rent_condition := (A_rent = 720)
def total_rent_condition := (total_rent = 3250)

-- Define cow-months calculation
def cow_months (cows months : ℕ) : ℕ := cows * months

-- Define cost per cow-month
def cost_per_cow_month (rent cow_months : ℕ) : ℕ := rent / cow_months

-- Define B's months of grazing proof problem
theorem B_grazed_months
  (A_cows_months : cow_months 24 3 = 72)
  (B_cows := 10)
  (C_cows_months : cow_months 35 4 = 140)
  (D_cows_months : cow_months 21 3 = 63)
  (A_rent_condition : A_rent = 720)
  (total_rent_condition : total_rent = 3250) :
  ∃ (B_months : ℕ), 10 * B_months = 50 ∧ B_months = 5 := sorry

end B_grazed_months_l25_25445


namespace trajectory_equation_of_moving_circle_center_l25_25598

theorem trajectory_equation_of_moving_circle_center 
    (M : Type) [metric_space M]
    (C1 C2 : M) 
    (dC1 : ∀ (x y : ℝ), x^2 + (y + 1)^2 = 25) 
    (dC2 : ∀ (x y : ℝ), x^2 + (y - 2)^2 = 1) 
    (foci_dist : dist C1 C2 = 4) 
    (ellipse_def : ∀ (M_center r : ℝ), dist M_center C1 = 5 - r → dist M_center C2 = r + 1 → r + (5 - r) = 6) :
    (∃ (x y : ℝ), (y^2 / 9) + (x^2 / 5) = 1 ∧ y ≠ 3) :=
sorry

end trajectory_equation_of_moving_circle_center_l25_25598


namespace amount_sharpened_off_l25_25234

-- Defining the initial length of the pencil
def initial_length : ℕ := 31

-- Defining the length of the pencil after sharpening
def after_sharpening_length : ℕ := 14

-- Proving the amount sharpened off the pencil
theorem amount_sharpened_off : initial_length - after_sharpening_length = 17 := 
by 
  -- Here we would insert the proof steps, 
  -- but as instructed we leave it as sorry.
  sorry

end amount_sharpened_off_l25_25234


namespace collinearity_of_centers_l25_25734

-- Definitions for the problem
variables {A B C D E F X Y Z : Point}
variables {I O : Point}
variables {Incircle : Circle}

-- Assume we have a triangle ABC and its incenter is I
axiom incenter_of_triangle_ABC : Incenter I A B C

-- Assume the incircle touches sides BC, CA, AB at points D, E, F respectively
axiom incircle_touches_sides : IncircleTouchesTriangle Incircle A B C D E F

-- Assume X, Y, Z are the midpoints of EF, FD, DE respectively
axiom midpoints_of_sides : Midpoints XYZ EF FD DE X Y Z

-- Assume O is the circumcenter of triangle ABC
axiom circumcenter_of_triangle_ABC : Circumcenter O A B C

-- Define the problem to be proven: Collinearity of three points I, circumcenter(XYZ), and O
theorem collinearity_of_centers : Collinear I (Circumcenter XYZ) O := 
sorry

end collinearity_of_centers_l25_25734


namespace solve_floor_eq_l25_25362

theorem solve_floor_eq (x : ℝ) (hx : 2 ≤ x ∧ x < 3) : 
  x * floor (x * floor (x * floor (x * floor x))) = 122 → x = 122 / 41 :=
by
  sorry

end solve_floor_eq_l25_25362


namespace distance_from_A_to_directrix_on_parabola_l25_25048

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25048


namespace probability_two_packages_correct_l25_25536

noncomputable def factorial (n : ℕ) : ℕ :=
nat.factorial n

noncomputable def choose (n k : ℕ) : ℕ :=
nat.choose n k

noncomputable def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 
  else 
    (n - 1)!  * ∑ i in finset.range n.succ, ((-1) ^ i) / (i.factorial : ℝ) |>.to_rat

theorem probability_two_packages_correct 
  (n : ℕ := 5) :
  (choose n 2 * derangements (n - 2)) / factorial n = (1 : ℚ) / 6 := by
  sorry

end probability_two_packages_correct_l25_25536


namespace floor_factorial_expression_l25_25883

theorem floor_factorial_expression :
  (⌊((2007.factorial + 2004.factorial) / (2006.factorial + 2005.factorial))⌋ : ℤ) = 2006 := 
by 
  sorry

end floor_factorial_expression_l25_25883


namespace distance_from_point_to_parabola_directrix_l25_25111

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25111


namespace foci_distance_of_hyperbola_l25_25889

theorem foci_distance_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 32 - y^2 / 8 = 1) → 2 * (Real.sqrt (32 + 8)) = 4 * Real.sqrt 10 :=
by
  intros x y h
  sorry

end foci_distance_of_hyperbola_l25_25889


namespace bob_house_number_l25_25874

theorem bob_house_number :
  ∃ h : ℕ, h = 98765 ∧  ∑ d in (98765.digits 10), d = 39 ∧ 
           (∀ d ∈ (98765.digits 10), (∀ d' ∈ (98765.digits 10), d ≠ d' → true)) := 
sorry

end bob_house_number_l25_25874


namespace trapezoid_area_eq_sum_of_areas_l25_25243

variables (A B C D O : Type)
variables [geometry.segment A B] [geometry.segment B C] [geometry.segment C D] [geometry.segment D A]
variables [geometry.segment A C] [geometry.segment B D] [geometry.parallel (geometry.line B C) (geometry.line D A)]
variables [geometry.intersect AC BD, intersection_point O]

-- Area Definitions
def area (X Y Z : Type) : Real := sorry

def TrapezoidArea : Real := area A B O + area B C O + area C D O + area D A O

theorem trapezoid_area_eq_sum_of_areas :
  TrapezoidArea = (Real.sqrt (area B O C) + Real.sqrt (area D O A)) ^ 2 := 
sorry

end trapezoid_area_eq_sum_of_areas_l25_25243


namespace maximum_triangle_area_l25_25594

noncomputable def max_area_of_triangle (A B P : Point) (l : Line) (c : Circle) : ℝ :=
  let AB := l.length_between_intersections_with c
  let max_distance_from_P_to_AB := c.radius + l.distance_to_circle_center c.center
  (1 / 2) * AB * max_distance_from_P_to_AB

theorem maximum_triangle_area : 
  let l : Line := ⟨λ x y, √3 * x - y - 4⟩
  let c : Circle := ⟨⟨0, 2⟩, 5⟩
  let A B : Point := l.intersection_points_with c
  ∃ P : Point, c.is_on P ∧ P ≠ A ∧ P ≠ B ∧ 
  max_area_of_triangle A B P l c = 32 :=
by sorry

end maximum_triangle_area_l25_25594


namespace chinese_remainder_problem_l25_25386

theorem chinese_remainder_problem :
  let count_a := (λ count_a : ℕ ⇒ count_a = 135) in
  ∃ count_a, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 134 ∧
    (λ a := (15 * k + 8)) ∈ Set.Icc 2 2019 ∧
    (a % 3 = 2) ∧ (a % 5 = 3) :=
sorry

end chinese_remainder_problem_l25_25386


namespace ce_squared_eq_ab_cd_l25_25801

variables {A B C D E : Type}
          [IsoscelesTrapezium A B C D]
          [Parallel AB CD]

def has_incircle (ABCD : Type) : Prop := sorry
def perpendicular (C : Type) (AB : Type) : E := sorry

theorem ce_squared_eq_ab_cd (h_trapez : IsoscelesTrapezium A B C D)
                            (h_parallel : Parallel AB CD)
                            (h_incircle : has_incircle A B C D)
                            (h_perpendicular : perpendicular C AB = E) :
                            E^2 = (AB : ℝ) * (CD : ℝ) :=
sorry

end ce_squared_eq_ab_cd_l25_25801


namespace collinear_probability_l25_25669

/-- Define points in terms of the square. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

namespace Points

def O : Point := { x := 0.5, y := 0.5 } -- Center of the square (wlog)
def A : Point := { x := 0.0, y := 0.0 } -- Vertex A
def B : Point := { x := 1.0, y := 0.0 } -- Vertex B
def C : Point := { x := 1.0, y := 1.0 } -- Vertex C
def D : Point := { x := 0.0, y := 1.0 } -- Vertex D

def are_collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

def prob_collinear (points : List Point) : ℚ :=
  let combinations := List.combinations 3 points
  let collinear_combinations := combinations.filter (fun [p1, p2, p3] => are_collinear p1 p2 p3)
  collinear_combinations.length / combinations.length

theorem collinear_probability : (prob_collinear [O, A, B, C, D] = 1 / 5) :=
  sorry

end Points

end collinear_probability_l25_25669


namespace exists_infinitely_many_natural_numbers_factors_l25_25350

theorem exists_infinitely_many_natural_numbers_factors (k : ℤ) :
  ∃ (P Q : ℤ[X]), 
    (P.degree = 4 ∧ Q.degree = 4) ∧
    (X^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = P * Q) :=
by
  sorry

end exists_infinitely_many_natural_numbers_factors_l25_25350


namespace total_snails_and_frogs_l25_25401

-- Define the number of snails and frogs in the conditions.
def snails : Nat := 5
def frogs : Nat := 2

-- State the problem: proving that the total number of snails and frogs equals 7.
theorem total_snails_and_frogs : snails + frogs = 7 := by
  -- Proof is omitted as the user requested only the statement.
  sorry

end total_snails_and_frogs_l25_25401


namespace distance_from_A_to_directrix_l25_25019

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25019


namespace distance_from_A_to_directrix_l25_25079

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25079


namespace distance_from_A_to_directrix_on_parabola_l25_25039

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25039


namespace Brandy_energy_drinks_l25_25383

theorem Brandy_energy_drinks 
  (maximum_safe_amount : ℕ)
  (caffeine_per_drink : ℕ)
  (extra_safe_caffeine : ℕ)
  (x : ℕ)
  (h1 : maximum_safe_amount = 500)
  (h2 : caffeine_per_drink = 120)
  (h3 : extra_safe_caffeine = 20)
  (h4 : caffeine_per_drink * x + extra_safe_caffeine = maximum_safe_amount) :
  x = 4 :=
by
  sorry

end Brandy_energy_drinks_l25_25383


namespace distance_from_A_to_directrix_l25_25080

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25080


namespace pn_divisible_by_3_l25_25920

noncomputable def harmonic_sum (n : ℕ) : ℚ :=
  ∑ k in finset.range n + 1, 1 / (k + 1 : ℚ)

def p_n (n : ℕ) (hn : (harmonic_sum n).num.natAbs = p / q ∧ (harmonic_sum n).denom.natAbs = q) : ℕ :=
  (harmonic_sum n).num.natAbs

theorem pn_divisible_by_3 (n : ℕ) :
  (p_n n (harmonic_sum.to_rational n)).mod 3 = 0 ↔ (n = 2 ∨ n = 7 ∨ n = 22) :=
by sorry

end pn_divisible_by_3_l25_25920


namespace distance_from_A_to_directrix_on_parabola_l25_25052

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25052


namespace binom_22_12_l25_25948

theorem binom_22_12 (h1 : nat.choose 20 10 = 184756) 
                    (h2 : nat.choose 20 11 = 167960) 
                    (h3 : nat.choose 20 12 = 125970) :
    nat.choose 22 12 = 646646 := 
by
    sorry

end binom_22_12_l25_25948


namespace infinite_N_for_factorization_l25_25322

noncomputable def is_special_number (k : ℤ) : ℕ :=
  4 * k ^ 4 - 8 * k ^ 2 + 2

theorem infinite_N_for_factorization : 
  ∀ k : ℤ, ∃ (P Q : Polynomial ℤ), Polynomial.degree P = 4 ∧ Polynomial.degree Q = 4 
          ∧ Polynomial.mul P Q = Polynomial.C 1 + Polynomial.C (is_special_number k) * Polynomial.X^4 + Polynomial.X^8 :=
by sorry

end infinite_N_for_factorization_l25_25322


namespace set_M_when_a_4_range_of_a_l25_25939

-- Part 1: Prove the set M when a = 4
theorem set_M_when_a_4 :
  ∀ x : ℝ, 
  ((4 * x - 5) / (x^2 - 4) < 0) ↔ 
  (x ∈ (-∞, -2) ∪ Ioo (5 / 4 : ℝ) 2) :=
by
  intro x
  sorry

-- Part 2: Prove the range of a given conditions on set M
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a * x - 5) / (x^2 - a) < 0) →
  (3 ∈ {x : ℝ | (a * x - 5) / (x^2 - a) < 0}) ∧
  (5 ∉ {x : ℝ | (a * x - 5) / (x^2 - a) < 0}) →
  (a ∈ Icc 1 (5 / 3) ∪ Ioo 9 25) :=
by
  intro h1 h2
  sorry

end set_M_when_a_4_range_of_a_l25_25939


namespace part_one_part_two_l25_25942

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.exp (x - 1)
def g (x : ℝ) : ℝ := Real.log x - 1

-- Theorem 1: Prove that for all x > 0, f(x) ≥ g(x) + 2
theorem part_one (x : ℝ) (hx : 0 < x) : f x ≥ g x + 2 := 
sorry

-- Theorem 2: Determine the existence and count of common tangent lines
theorem part_two : ∃! (t₁ t₂ : ℝ), (t₁ ≠ t₂) ∧
    (∀ x, f x = Real.exp (t₁ - 1) * x + Real.exp (t₁ - 1) * (1 - t₁) →
    g x = Real.exp (t₂ - 1) * x + Real.exp (t₂ - 1) * (1 - t₂)) :=
sorry

end part_one_part_two_l25_25942


namespace range_of_m_l25_25177

theorem range_of_m :
  ∀ (x y m : ℝ), 
    x + 2 * y = 4 * m ∧ 2 * x + y = 2 * m + 1 ∧ -1 < x - y ∧ x - y < 0 → 
    1 / 2 < m ∧ m < 1 :=
by
  intro x y m
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_ineq1
  cases h_ineq1 with h3 h4
  sorry

end range_of_m_l25_25177


namespace all_xi_equal_l25_25671

theorem all_xi_equal (P : Polynomial ℤ) (n : ℕ) (hn : n % 2 = 1) (x : Fin n → ℤ) 
  (hP : ∀ i : Fin n, P.eval (x i) = x ⟨i + 1, sorry⟩) : 
  ∀ i j : Fin n, x i = x j :=
by
  sorry

end all_xi_equal_l25_25671


namespace d_not_suff_nec_condition_l25_25255

variable {d : ℝ} {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}

-- Definitions based on conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d

def sum_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) (λ i, a_n i)

def is_increasing_sequence (S_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S_n (n + 1) > S_n n

-- Statement of the equivalence problem
theorem d_not_suff_nec_condition (hd : is_arithmetic_sequence a_n d)
  (hS : ∀ n, S_n n = sum_first_n_terms a_n n) :
  ¬ ( (∀ S_n, is_increasing_sequence S_n → d > 0)
  ∧ (d > 0 → ∀ S_n, is_increasing_sequence S_n)) :=
sorry

end d_not_suff_nec_condition_l25_25255


namespace club_officers_selection_l25_25825

-- Defining the set of members
def members := Fin 30

-- Defining the condition that exactly one of the following is true:
-- Both Alice and Bob are officers, or neither is.
def valid_configuration (officers : Fin 4 → members) : Prop :=
  (officers ⟨0, by decide⟩ = 0 ∧ officers ⟨1, by decide⟩ = 1) ∨
  (officers ⟨0, by decide⟩ ≠ 0 ∧ officers ⟨1, by decide⟩ ≠ 1)

-- Main theorem statement
theorem club_officers_selection :
  ∃ (selection : Fin 4 → members), valid_configuration selection ∧
    (∑ (selection : Fin 4 → members) in {selection : Fin 4 → members |
    valid_configuration selection}.to_finset, 1) = 500472 := sorry

end club_officers_selection_l25_25825


namespace starting_current_ratio_l25_25748

theorem starting_current_ratio (running_current : ℕ) (units : ℕ) (total_current : ℕ)
    (h1 : running_current = 40) 
    (h2 : units = 3) 
    (h3 : total_current = 240) 
    (h4 : total_current = running_current * (units * starter_ratio)) :
    starter_ratio = 2 := 
sorry

end starting_current_ratio_l25_25748


namespace survey_result_l25_25633

noncomputable def percentageMint :=
  let vanilla := 80
  let chocolate := 100
  let strawberry := 50
  let mint := 70
  let caramel := 40
  let total := vanilla + chocolate + strawberry + mint + caramel
  (mint.toFloat / total.toFloat) * 100

theorem survey_result : percentageMint ≈ 21 := by
  sorry

end survey_result_l25_25633


namespace infinite_N_for_factorization_l25_25316

noncomputable def is_special_number (k : ℤ) : ℕ :=
  4 * k ^ 4 - 8 * k ^ 2 + 2

theorem infinite_N_for_factorization : 
  ∀ k : ℤ, ∃ (P Q : Polynomial ℤ), Polynomial.degree P = 4 ∧ Polynomial.degree Q = 4 
          ∧ Polynomial.mul P Q = Polynomial.C 1 + Polynomial.C (is_special_number k) * Polynomial.X^4 + Polynomial.X^8 :=
by sorry

end infinite_N_for_factorization_l25_25316


namespace range_of_x_l25_25585

def f (x : ℝ) (a : ℝ) : ℝ := if x ≤ 0 then exp x + a else x^2 + 1 + a

theorem range_of_x (a : ℝ) (x : ℝ) :
  f (2 - x) a ≥ f x a → x ≤ 1 :=
by
  sorry

end range_of_x_l25_25585


namespace distance_from_A_to_directrix_l25_25022

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25022


namespace negation_abs_lt_zero_l25_25540

theorem negation_abs_lt_zero : ¬ (∀ x : ℝ, |x| < 0) ↔ ∃ x : ℝ, |x| ≥ 0 := 
by 
  sorry

end negation_abs_lt_zero_l25_25540


namespace miles_per_gallon_l25_25497

theorem miles_per_gallon (miles : ℕ) (gallons : ℕ) (h_miles : miles = 60) (h_gallons : gallons = 3) : miles / gallons = 20 :=
by
  rw [h_miles, h_gallons]
  exact Eq.refl (60 / 3)

end miles_per_gallon_l25_25497


namespace cavalieri_principle_l25_25365

-- Definitions for the problem (conditions)
def A : Type := geometric_shape -- Placeholder type for geometric shapes
def B : Type := geometric_shape
def height (x : geometric_shape) : ℝ := sorry -- Heights of the shapes
def cross_sectional_area_at_height (x : geometric_shape) (h : ℝ) : ℝ := sorry -- Cross-sectional areas at a given height

-- Conditions of equality
def equal_height (a b : geometric_shape) : Prop := height a = height b
def equal_cross_sections (a b : geometric_shape) : Prop :=
  ∀ h, cross_sectional_area_at_height a h = cross_sectional_area_at_height b h

-- Definitions for p and q
def p (a b : geometric_shape) : Prop := volume a ≠ volume b
def q (a b : geometric_shape) : Prop := equal_cross_sections a b

-- Statement: p -> q ∧ ¬ (q -> p)
theorem cavalieri_principle (a b : geometric_shape)
  (h_eq : equal_height a b) :
  p a b → q a b ∧ ¬ (q a b → p a b) :=
by sorry

end cavalieri_principle_l25_25365


namespace largest_n_for_crates_l25_25482

theorem largest_n_for_crates (total_crates : ℕ) (min_oranges max_oranges : ℕ)
  (h1 : total_crates = 145)
  (h2 : min_oranges = 110)
  (h3 : max_oranges = 140) : 
  ∃ n : ℕ, n = 5 ∧ ∀ k : ℕ, k ≤ max_oranges - min_oranges + 1 → total_crates / k ≤ n :=
  by {
    sorry
  }

end largest_n_for_crates_l25_25482


namespace sufficient_but_not_necessary_condition_l25_25391

noncomputable def f (a x : ℝ) := x^2 + 2 * a * x - 2

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, x ≤ -2 → deriv (f a) x ≤ 0) ↔ a = 2 :=
sorry

end sufficient_but_not_necessary_condition_l25_25391


namespace _l25_25139

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25139


namespace kanul_machinery_expense_l25_25242

theorem kanul_machinery_expense :
  let Total := 93750
  let RawMaterials := 35000
  let Cash := 0.20 * Total
  let Machinery := Total - (RawMaterials + Cash)
  Machinery = 40000 := by
sorry

end kanul_machinery_expense_l25_25242


namespace gcf_90_150_225_l25_25770

theorem gcf_90_150_225 : 
  let a := 90 in
  let b := 150 in
  let c := 225 in
  (∃ p q r : ℕ, a = 2 * 3^2 * 5 ∧ b = 2 * 3 * 5^2 ∧ c = 3^2 * 5^2) →
  Nat.gcd (Nat.gcd a b) c = 15 :=
by
  intros a b c h
  cases h with p h
  cases h with q h
  cases h with r h
  simp [h]
  sorry

end gcf_90_150_225_l25_25770


namespace distance_to_directrix_l25_25097

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25097


namespace distance_to_directrix_l25_25036

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25036


namespace ratio_of_areas_triangle_CDE_to_ABE_l25_25636

noncomputable def area_ratio (AB CD : ℝ) (AE BE DE CE : ℝ) (alpha : ℝ) : ℝ :=
  (Math.tan alpha) ^ 2

theorem ratio_of_areas_triangle_CDE_to_ABE 
  (R r x y alpha : ℝ)
  (h1 : AB = 2 * R)
  (h2 : CD = 2 * r)
  (h3 : AE = BE = x)
  (h4 : DE = CE = y)
  (h5 : Math.tan alpha = y / x) :
  area_ratio AB CD AE BE DE CE alpha = (Math.tan alpha) ^ 2 := by
  sorry

end ratio_of_areas_triangle_CDE_to_ABE_l25_25636


namespace cone_arc_length_and_central_angle_l25_25581

theorem cone_arc_length_and_central_angle
  (l : ℝ) (h : ℝ) (hl : l = 10) (hh : h = 8) :
  let r := sqrt (l^2 - h^2),
      circumference := 2 * Real.pi * r,
      arc_length := circumference,
      central_angle := (arc_length / l) * (180 / Real.pi) in
  arc_length = 12 * Real.pi ∧ central_angle = 216 :=
by
  intro r circumference arc_length central_angle
  have : r = sqrt (l^2 - h^2),
  have : circumference = 2 * Real.pi * r,
  have : arc_length = circumference,
  have : central_angle = (arc_length / l) * (180 / Real.pi),
  sorry

end cone_arc_length_and_central_angle_l25_25581


namespace infinite_natural_numbers_with_factored_polynomial_l25_25330

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l25_25330


namespace distance_to_directrix_l25_25090

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25090


namespace pradeep_failed_by_25_marks_l25_25706

theorem pradeep_failed_by_25_marks :
  (35 / 100 * 600 : ℝ) - 185 = 25 :=
by
  sorry

end pradeep_failed_by_25_marks_l25_25706


namespace diameter_circle_C_inscribed_within_D_l25_25505

noncomputable def circle_diameter_C (d_D : ℝ) (ratio : ℝ) : ℝ :=
  let R := d_D / 2
  let r := (R : ℝ) / (Real.sqrt 5)
  2 * r

theorem diameter_circle_C_inscribed_within_D 
  (d_D : ℝ) (ratio : ℝ) (h_dD_pos : 0 < d_D) (h_ratio : ratio = 4)
  (h_dD : d_D = 24) : 
  circle_diameter_C d_D ratio = 24 * Real.sqrt 5 / 5 :=
by
  sorry

end diameter_circle_C_inscribed_within_D_l25_25505


namespace distance_from_A_to_directrix_l25_25078

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25078


namespace distance_from_point_A_to_directrix_C_l25_25126

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25126


namespace length_of_BC_l25_25852

theorem length_of_BC (a : ℝ) (b_x b_y c_x c_y area : ℝ) 
  (h1 : b_y = b_x ^ 2)
  (h2 : c_y = c_x ^ 2)
  (h3 : b_y = c_y)
  (h4 : area = 64) :
  c_x - b_x = 8 := by
sorry

end length_of_BC_l25_25852


namespace exists_zero_in_interval_l25_25740

noncomputable def f (x : ℝ) : ℝ := Real.log ((3 * x) / 2 - 2 / x)

theorem exists_zero_in_interval :
  ∃ x ∈ Ioo 1 2, f x = 0 := by
  sorry

end exists_zero_in_interval_l25_25740


namespace internal_angle_bisectors_concurrent_l25_25507

theorem internal_angle_bisectors_concurrent
  (A B C D P : EuclideanSpace ℝ (Fin 3))
  (h_convex : convex_hull ℝ ({A, B, C, D} : Set (EuclideanSpace ℝ (Fin 3))) )
  (h_angles : ∃ (α β : ℝ), ∠PAD = 1 * α ∧ ∠PBA = 2 * α ∧ ∠DPA = 3 * α ∧
                            ∠CBP = 1 * β ∧ ∠BAP = 2 * β ∧ ∠BPC = 3 * β) :
  ∃ O : EuclideanSpace ℝ (Fin 3), 
    is_circumcenter O A B P ∧
    (O ∈ internal_angle_bisectors (triangle.mk D P A) ) ∧
    (O ∈ internal_angle_bisectors (triangle.mk P C B) ) ∧
    (O ∈ perpendicular_bisector (segment A B)) :=
sorry

end internal_angle_bisectors_concurrent_l25_25507


namespace sum_of_powers_eight_l25_25275

variable {a b : ℝ}

theorem sum_of_powers_eight :
  a + b = 1 → 
  a^2 + b^2 = 3 → 
  a^3 + b^3 = 4 → 
  a^4 + b^4 = 7 → 
  a^5 + b^5 = 11 → 
  a^8 + b^8 = 47 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  -- Proof to be filled in
  sorry

end sum_of_powers_eight_l25_25275


namespace exists_infinitely_many_natural_numbers_factors_l25_25353

theorem exists_infinitely_many_natural_numbers_factors (k : ℤ) :
  ∃ (P Q : ℤ[X]), 
    (P.degree = 4 ∧ Q.degree = 4) ∧
    (X^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = P * Q) :=
by
  sorry

end exists_infinitely_many_natural_numbers_factors_l25_25353


namespace number_of_zeros_of_F_l25_25973

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2^x else -1/x

def F (x : ℝ) : ℝ := f(x) + x

theorem number_of_zeros_of_F : (∃ x1 x2 : ℝ, F x1 = 0 ∧ F x2 = 0 ∧ x1 ≠ x2) :=
sorry

end number_of_zeros_of_F_l25_25973


namespace inscribed_polygon_regular_l25_25446

theorem inscribed_polygon_regular {n : ℕ} (h_odd : n % 2 = 1) 
    (h_inscribed : is_inscribed_polygon A₁ A₂ ⋯ Aₙ) 
    (h_equal_angles : ∀ i j, angle Aᵢ Aⱼ = angle A₁ A₂) : 
  is_regular_polygon A₁ A₂ ⋯ Aₙ :=
sorry

end inscribed_polygon_regular_l25_25446


namespace math_problem_l25_25162

theorem math_problem 
  (a : ℝ) 
  (ha : a < 9) 
  (h_eq_circle : ∀ x y : ℝ, x^2 + y^2 - 6*x + a = 0 → (x, y) ∈ {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9 - a})
  (h_slope_CD : ∀ x y x' y' : ℝ, (y' - y) / (x' - x) = 1 / 3) :
  (∃ M : ℝ × ℝ, M = (3, 0)) ∧
  (∃ k_AC k_BD : ℝ, k_AC = -1 / 2 ∧ k_BD = 2) ∧
  (∃ (p : ℝ → ℝ), ∀ y, y^2 = x ↔ y * y = x) ∧
  (∃ (l : ℝ → ℝ), ∀ x y : ℝ, 3*y + 8 = x → x - 3*y = 8) :=
sorry

end math_problem_l25_25162


namespace count_squares_below_line_l25_25743

theorem count_squares_below_line :
  let line_eq := λ (x y : ℕ), 10 * x + 210 * y = 2100 in
  let total_squares := 210 * 10 in
  let horizontal_intercepts := 209 in
  let vertical_intercepts := 9 in
  let squares_crossed := horizontal_intercepts + vertical_intercepts in
  let squares_below_line := (total_squares - squares_crossed) / 2 in
  squares_below_line = 941 :=
by
  let line_eq := λ (x y : ℕ), 10 * x + 210 * y = 2100
  let total_squares := 210 * 10
  let horizontal_intercepts := 209
  let vertical_intercepts := 9
  let squares_crossed := horizontal_intercepts + vertical_intercepts
  let squares_below_line := (total_squares - squares_crossed) / 2
  show squares_below_line = 941
  by sorry

end count_squares_below_line_l25_25743


namespace distance_from_A_to_directrix_l25_25157

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25157


namespace distance_from_A_to_directrix_l25_25069

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25069


namespace pool_fill_time_l25_25467

-- Define the rates as variables
variables (a b c d : ℚ)

-- Define the conditions
def condition1 : Prop := a + b = 1 / 2
def condition2 : Prop := b + c = 1 / 3
def condition3 : Prop := c + d = 1 / 4

-- Define the theorem to be proven
theorem pool_fill_time (h1 : condition1) (h2 : condition2) (h3 : condition3) : (1 / (a + d) = 12 / 5) :=
by 
  sorry

end pool_fill_time_l25_25467


namespace solve_for_x_l25_25908

theorem solve_for_x (x : ℝ) : 
  5^12 = 5^(90/x) / (5^(50/x) * 25^(30/x)) → 
  x = -5 / 3 :=
by
  intro h
  -- Proof will go here
  sorry

end solve_for_x_l25_25908


namespace distance_A_to_origin_l25_25753

theorem distance_A_to_origin :
  ∀ (m : ℝ), (∀ (x : ℝ), x = sqrt 3 → m = sqrt 3) →
  let A := (sqrt 3, 2, 5)
  let O := (0, 0, 0)
  real.sqrt ((sqrt 3)^2 + 2^2 + 5^2) = 4 * sqrt 2 := 
by
  intro m h
  have m_eq : m = sqrt 3 := h (sqrt 3) rfl
  let A : ℝ × ℝ × ℝ := (sqrt 3, 2, 5)
  let O : ℝ × ℝ × ℝ := (0, 0, 0)
  calc
  real.sqrt ((sqrt 3)^2 + 2^2 + 5^2) = 4 * sqrt 2 : sorry

end distance_A_to_origin_l25_25753


namespace coin_flip_probability_l25_25426

theorem coin_flip_probability
    (P_A_heads : ℚ := 1 / 3)
    (P_B_heads : ℚ := 1 / 2)
    (P_C_heads : ℚ := 2 / 3)
    (P_select_coin : ℚ := 1 / 3):
    let P_3_heads_1_tail_given_A := (4 * (P_A_heads^3) * ((1 - P_A_heads))) in
    let P_3_heads_1_tail_given_B := (4 * (P_B_heads^3) * ((1 - P_B_heads))) in
    let P_3_heads_1_tail_given_C := (4 * (P_C_heads^3) * ((1 - P_C_heads))) in
    let total_P_3_heads_1_tail := (P_3_heads_1_tail_given_A * P_select_coin +
                                   P_3_heads_1_tail_given_B * P_select_coin +
                                   P_3_heads_1_tail_given_C * P_select_coin) in
    let P_A_given_3_heads_1_tail := (P_3_heads_1_tail_given_A * P_select_coin) / total_P_3_heads_1_tail in
    let numerator := nat.gcd 32 273 in
    let denominator := 273 / numerator in
    numerator + denominator = 273 :=
sorry

end coin_flip_probability_l25_25426


namespace perpendicular_centroids_of_equilateral_triangles_l25_25558

theorem perpendicular_centroids_of_equilateral_triangles 
  {A B C D : ℂ} 
  (h_convex: ConvexQuadrilateral A B C D)
  (h_equal_diagonals: AC = BD)
  (h_equilateral_AB : is_equilateral_triangle A B C)
  (h_equilateral_BC : is_equilateral_triangle B C D)
  (h_equilateral_CD : is_equilateral_triangle C D A)
  (h_equilateral_DA : is_equilateral_triangle D A B) :
  Let O1 be the centroid of the triangle on side AB.
  Let O3 be the centroid of the triangle on side CD.
  Let O2 be the centroid of the triangle on side BC.
  Let O4 be the centroid of the triangle on side DA.
  angle_between_lines (O1, O3) (O2, O4) = 90° :=
sorry

end perpendicular_centroids_of_equilateral_triangles_l25_25558


namespace sum_of_possible_values_l25_25189

theorem sum_of_possible_values (x : ℝ) :
  (x + 3) * (x - 4) = 20 →
  ∃ a b, (a ≠ b) ∧ 
         ((x = a) ∨ (x = b)) ∧ 
         (x^2 - x - 32 = 0) ∧ 
         (a + b = 1) :=
by
  sorry

end sum_of_possible_values_l25_25189


namespace distance_from_A_to_directrix_l25_25068

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25068


namespace _l25_25129

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25129


namespace partitional_points_100_not_60_count_l25_25245

noncomputable def is_n_ray_partitional (R : set (ℝ × ℝ)) (X : ℝ × ℝ) (n : ℕ) : Prop :=
X ∈ interior R ∧ ∃ (rays : fin n → ray (ℝ × ℝ)), (∀ i j, i ≠ j → disjoint (rays i) (rays j)) ∧
  (∀ i, area (region_divided_by_rays (rays i)) = (area R) / n)

theorem partitional_points_100_not_60_count :
  let R : set (ℝ × ℝ) := { p | p.1 ≥ 0 ∧ p.1 ≤ 1 ∧ p.2 ≥ 0 ∧ p.2 ≤ 1 }
  ∃ n : ℕ, n = 100 ∧ ∃ m : ℕ, m = 60 →
  let n_100 := {X : ℝ × ℝ | is_n_ray_partitional R X 100}
  let n_60 := {X : ℝ × ℝ | is_n_ray_partitional R X 60}
  finset.card ((finset.filter (λ (X : ℝ × ℝ), X ∈ n_100 ∧ X ∉ n_60) n_100.to_finset)) = 2320 :=
sorry

end partitional_points_100_not_60_count_l25_25245


namespace evaluate_expression_l25_25423

theorem evaluate_expression (a : ℝ) (h : a = -3) : 
  (3 * a⁻¹ + (a⁻¹ / 3)) / a = 10 / 27 :=
by 
  sorry

end evaluate_expression_l25_25423


namespace max_length_vecb_plus_vecc_cos_beta_value_l25_25601

variables {α β : ℝ}
def veca : ℝ × ℝ := (Real.cos α, Real.sin α)
def vecb : ℝ × ℝ := (Real.cos β, Real.sin β)
def vecc : ℝ × ℝ := (-1, 0)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem max_length_vecb_plus_vecc :
  vec_length (vecb + vecc) ≤ 2 := by
  sorry

theorem cos_beta_value
  (hα : α = Real.pi / 4)
  (h_orthogonal : dot_product veca (vecb + vecc) = 0) :
  Real.cos β = 0 ∨ Real.cos β = 1 := by
  sorry

end max_length_vecb_plus_vecc_cos_beta_value_l25_25601


namespace log_inequality_implies_inequality_l25_25547

variable {a b c : ℝ}

theorem log_inequality_implies_inequality (h : log (1/2) b < log (1/2) a ∧ log (1/2) a < log (1/2) c) :
  c < a ∧ a < b :=
by
  sorry

end log_inequality_implies_inequality_l25_25547


namespace find_angle_AOB_l25_25627

theorem find_angle_AOB (O : Point) (A B : Point) (h1 : 0.2 = 20 / 100) (h2 : 360 = 360) :
  angle A O B = 72 :=
sorry

end find_angle_AOB_l25_25627


namespace cosine_of_difference_l25_25986

theorem cosine_of_difference (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (α - π / 3) = 1 / 3 :=
by
  sorry

end cosine_of_difference_l25_25986


namespace distance_from_A_to_directrix_of_parabola_l25_25060

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25060


namespace length_of_courtyard_l25_25827

-- Given conditions

def width_of_courtyard : ℝ := 14
def brick_length : ℝ := 0.25
def brick_width : ℝ := 0.15
def total_bricks : ℝ := 8960

-- To be proven
theorem length_of_courtyard : brick_length * brick_width * total_bricks / width_of_courtyard = 24 := 
by sorry

end length_of_courtyard_l25_25827


namespace count_k_for_lcm_problem_l25_25187

theorem count_k_for_lcm_problem :
  let k_values := {k : ℕ | ∃ (a b : ℕ), a ≤ 18 ∧ b = 36 ∧ k = 2^a * 3^b} in
  ∃! k_values, 18^18 = Nat.lcm (9^9) (Nat.lcm (12^12) k_values) :=
by
  sorry

end count_k_for_lcm_problem_l25_25187


namespace triangle_sides_consecutive_median_perpendicular_angle_bisector_l25_25745

theorem triangle_sides_consecutive_median_perpendicular_angle_bisector :
  ∃ (a b c : ℕ), b = a + 1 ∧ c = a + 2 ∧ 
                   (∃ M : ℝ, median_perpendicular_median_angle_bisector (a, b, c) M) ∧
                   a = 2 ∧ b = 3 ∧ c = 4 :=
by sorry

end triangle_sides_consecutive_median_perpendicular_angle_bisector_l25_25745


namespace equal_boys_girls_in_70_segment_l25_25402

theorem equal_boys_girls_in_70_segment
  (children : List Bool)
  (h1 : children.length = 100)
  (h2 : children.count (λ x => x) = 50) -- Assuming True represents boys and False represents girls
  (h3 : ∃ i : ℕ, i < 71 ∧ children.slice i 30.count (λ x => x) = 15) :
  ∃ j : ℕ, j < 31 ∧ children.slice j 70.count (λ x => x) = 35 := sorry

end equal_boys_girls_in_70_segment_l25_25402


namespace find_sin_cos_theta_find_cos_phi_l25_25180

-- Given conditions
variables {θ φ : ℝ}
variables (hθ1 : θ ∈ Ioo 0 (π / 2))
variables (hθ2 : sin θ = 2 * cos θ)
variables (hφ1 : 0 < φ ∧ φ < π / 2)
variables (hφ2 : sin (θ - φ) = sqrt 10 / 10)

-- Proof statements
theorem find_sin_cos_theta (hθ_cond : θ ∈ Ioo 0 (π / 2))
  (h_perpendicular : sin θ - 2 * cos θ = 0) :
  sin θ = 2 * (sqrt 5 / 5) ∧ cos θ = sqrt 5 / 5 := by
  sorry

theorem find_cos_phi (hθ_cond : θ ∈ Ioo 0 (π / 2)) (hθ_cos_sin : sin θ = 2 * cos θ)
  (hφ_cond : φ ∈ Ioo 0 (π / 2))
  (h_sin_thetaphi : sin (θ - φ) = sqrt 10 / 10) :
  cos φ = sqrt 2 / 2 := by
  sorry

end find_sin_cos_theta_find_cos_phi_l25_25180


namespace distance_from_point_A_to_directrix_C_l25_25120

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25120


namespace part_I_part_II_l25_25926

variables (a b c x : ℝ)
axiom h1 : a^2 + b^2 + c^2 = 1

theorem part_I : |a + b + c| ≤ sqrt 3 :=
by sorry

theorem part_II (h2 : ∀ a b c : ℝ, |x - 1| + |x + 1| ≥ (a + b + c)^2) : 
  x ∈ Set.Icc (-∞) (-3/2) ∪ Set.Icc (3/2) ∞ :=
by sorry

end part_I_part_II_l25_25926


namespace part_a_part_b_M3_part_b_M4_l25_25246

noncomputable def set_S (n : ℕ) (h : n ≥ 3) : Finset (Fin n → Bool) :=
  Finset.univ

noncomputable def M (n : ℕ) (h : n ≥ 3) : ℕ :=
  sorry

theorem part_a (n : ℕ) (h : n ≥ 3) : M n h ≤ (2^(n+1) / n).floor + 1 :=
  sorry

theorem part_b_M3 : M 3 (dec_trivial) = 5 :=
  sorry

theorem part_b_M4 : M 4 (dec_trivial) = 9 :=
  sorry

end part_a_part_b_M3_part_b_M4_l25_25246


namespace angle_equal_l25_25228

-- Definitions of concepts used in the problem
variable (A B C D K L : Type)
variable [HasAngle A B C] [HasAngle B A L] [HasAngle C D K]
variable [IsTrapezoid A B C D] [OnSides K (A B)] [OnSides L (C D)]

-- State the theorem
theorem angle_equal {A B C D K L : Type}
  [IsTrapezoid A B C D]
  [OnSides K (A B)]
  [OnSides L (C D)]
  (h1 : ∠B A L = ∠C D K) :
  ∠B L A = ∠C K D :=
by
  sorry

end angle_equal_l25_25228


namespace total_books_l25_25406

variable (Tim_books Mike_books : ℕ)
variable (h1 : Tim_books = 22)
variable (h2 : Mike_books = 20)

theorem total_books : Tim_books + Mike_books = 42 :=
by
  calc Tim_books + Mike_books = 22 + 20 : by rw [h1, h2]
                          ...          = 42 : by norm_num

end total_books_l25_25406


namespace distance_from_point_to_parabola_directrix_l25_25098

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25098


namespace distance_from_A_to_directrix_l25_25076

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25076


namespace Simon_age_2010_l25_25240

variable (Jorge Simon : ℕ)

axiom age_difference : Jorge + 24 = Simon
axiom Jorge_age_2005 : Jorge = 16

theorem Simon_age_2010 : Simon = 45 := by
  have Simon_age_2005 : Simon = Jorge + 24 := age_difference
  rw [Jorge_age_2005] at Simon_age_2005
  have Simon_age_2005 : Simon = 16 + 24 := Simon_age_2005
  have Simon_age_2005 : Simon = 40 := by norm_num
  have Simon_age_2010 : Simon = 40 + 5 := by
    rw Simon_age_2005
    norm_num
  exact by norm_num at Simon_age_2010

end Simon_age_2010_l25_25240


namespace percentage_increase_in_savings_l25_25803

-- Define the initial income and expenditure
variables {I : ℝ} (h1 : 0 < I) -- Assume I is positive
def E_initial := 0.75 * I
def S_initial := I - E_initial

-- Define the new income and expenditure
def I_new := 1.20 * I
def E_new := 1.10 * E_initial

-- Define the new savings
def S_new := I_new - E_new

-- Define the percentage increase in savings
def percentage_increase := (S_new - S_initial) / S_initial * 100

-- The theorem that we need to prove
theorem percentage_increase_in_savings (h1 : 0 < I) : percentage_increase = 50 :=
by
  sorry

end percentage_increase_in_savings_l25_25803


namespace range_of_a_l25_25612

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x + 1| - 2 * a + 2 < 0) → (a > 2) :=
by
  sorry

end range_of_a_l25_25612


namespace class_8_3_final_score_is_correct_l25_25626

def class_8_3_singing_quality : ℝ := 92
def class_8_3_spirit : ℝ := 80
def class_8_3_coordination : ℝ := 70

def final_score (singing_quality spirit coordination : ℝ) : ℝ :=
  0.4 * singing_quality + 0.3 * spirit + 0.3 * coordination

theorem class_8_3_final_score_is_correct :
  final_score class_8_3_singing_quality class_8_3_spirit class_8_3_coordination = 81.8 :=
by
  sorry

end class_8_3_final_score_is_correct_l25_25626


namespace distance_from_A_to_directrix_l25_25149

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25149


namespace average_height_males_l25_25720

theorem average_height_males
  (M W H_m : ℝ)
  (h₀ : W ≠ 0)
  (h₁ : M = 2 * W)
  (h₂ : (M * H_m + W * 170) / (M + W) = 180) :
  H_m = 185 := 
sorry

end average_height_males_l25_25720


namespace problem_statement_l25_25193

theorem problem_statement (a b c x : ℝ) (h1 : a + x^2 = 2015) (h2 : b + x^2 = 2016)
    (h3 : c + x^2 = 2017) (h4 : a * b * c = 24) :
    (a / (b * c) + b / (a * c) + c / (a * b) - (1 / a) - (1 / b) - (1 / c) = 1 / 8) :=
by
  sorry

end problem_statement_l25_25193


namespace interval_of_increase_l25_25201

theorem interval_of_increase (a : ℝ) (h : a < 0) :
  {x : ℝ | ∀ (y : ℝ), x < y → g y > g x} = set.Iio 2 :=
by
  have g := λ x, a * (x^2 - 4 * x + 3)
  sorry

end interval_of_increase_l25_25201


namespace general_formula_find_Pn_find_Qn_compare_Pn_Qn_l25_25562

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {P Q : ℕ → ℝ}

-- Condition 1: All terms in the sequence {a_n} are positive
def positive_sequence (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a n > 0

-- Condition 2: The sum of the first n terms S_n satisfies the equation
def sequence_condition (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop := 
  ∀ n : ℕ, a n ^ 2 = 2 * S n - a n

-- General formula for the sequence
theorem general_formula (a S : ℕ → ℕ) (h₁ : positive_sequence a) (h₂ : sequence_condition a S) :
  ∀ n : ℕ, a n = n := 
sorry

-- P_n and Q_n
def sum_inv_S (S : ℕ → ℕ) (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / (S i : ℝ)
def sum_geo (a : ℕ → ℕ) (n : ℕ) : ℝ := ∑ i in finset.range n, (1 / 2) ^ (a i - 1)

-- Find P_n
theorem find_Pn (S : ℕ → ℕ) (h : ∀ n, S n = n * (n + 1) / 2) :
  ∀ n : ℕ, sum_inv_S S n = 2 * (1 - 1 / (n + 1 : ℝ)) :=
sorry

-- Find Q_n
theorem find_Qn (a : ℕ → ℕ) (h : ∀ n, a n = n) :
  ∀ n : ℕ, sum_geo a n = 2 * (1 - 1 / 2 ^ n) :=
sorry

-- Compare P_n and Q_n for n ≥ 2
theorem compare_Pn_Qn (n : ℕ) (hn : n ≥ 2) :
  sum_inv_S S n < sum_geo a n :=
sorry

end general_formula_find_Pn_find_Qn_compare_Pn_Qn_l25_25562


namespace jared_annual_earnings_l25_25916

open Nat

noncomputable def diploma_monthly_pay : ℕ := 4000
noncomputable def months_in_year : ℕ := 12
noncomputable def multiplier : ℕ := 3

theorem jared_annual_earnings :
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  jared_annual_earnings = 144000 :=
by
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  exact sorry

end jared_annual_earnings_l25_25916


namespace tan_alpha_values_l25_25575

theorem tan_alpha_values (α : ℝ) (h : Real.sin α + Real.cos α = 7 / 5) : 
  (Real.tan α = 4 / 3) ∨ (Real.tan α = 3 / 4) := 
  sorry

end tan_alpha_values_l25_25575


namespace distance_from_A_to_directrix_on_parabola_l25_25046

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25046


namespace sum_of_reciprocals_of_shifted_roots_l25_25258

theorem sum_of_reciprocals_of_shifted_roots:
  (∃ (a b c : ℝ), (a^3 - 2 * a + 1 = 0) ∧ (b^3 - 2 * b + 1 = 0) ∧ (c^3 - 2 * c + 1 = 0)) →
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 10 / 3) :=
begin
  sorry
end

end sum_of_reciprocals_of_shifted_roots_l25_25258


namespace part1_analytical_expression_part2_existence_of_abc_l25_25936

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Assumptions for Part (1)
variables (a b c : ℝ)
axiom fx_neg1 : quadratic a b c (-1) = 0
axiom fx_diff : ∀ x, quadratic a b c (x+1) - quadratic a b c x = 2 * x

-- Proof statement for Part (1)
theorem part1_analytical_expression : 
  ∃ (a b c : ℝ), 
  ∀ x, quadratic a b c x = x^2 - x - 2 :=
sorry

-- Assumptions for Part (2)
axiom range_nonneg : ∀ x, quadratic a b c x ≥ 0
axiom symmetry : ∀ x, quadratic a b c (x-4) = quadratic a b c (2-x)
axiom constraint : ∀ x, 0 ≤ quadratic a b c x - x ∧ quadratic a b c x - x ≤ (1/2) * (x-1)^2

-- Proof statement for Part (2)
theorem part2_existence_of_abc :
  ∃ (a b c : ℝ), 
  range_nonneg a b c ∧ symmetry a b c ∧ constraint a b c ∧ 
  a = 1/4 ∧ b = 1/2 ∧ c = 1/4 :=
sorry


end part1_analytical_expression_part2_existence_of_abc_l25_25936


namespace correct_number_of_true_statements_l25_25739

-- Definitions for the conditions
def is_mode (dataset : List ℕ) (m : ℕ) : Prop := 
  dataset.count m = (dataset.map (λ x => dataset.count x)).max'

def mean (dataset : List ℤ) : ℤ :=
  (dataset.sum : ℤ) / (dataset.length : ℤ)

def median (dataset : List ℤ) : ℤ := 
  let sorted_ds := dataset.qsort (· ≤ ·)
  ite (sorted_ds.length % 2 = 1)
    (sorted_ds[(sorted_ds.length / 2)])
    ((sorted_ds[(sorted_ds.length / 2) - 1] + sorted_ds[sorted_ds.length / 2]) / 2)

-- Conditions
axiom mode_condition (dataset : List ℕ) :
  is_mode dataset 5 → ∃ n, n = 5

axiom mean_median_condition (dataset : List ℤ) :
  mean dataset = 0 → median dataset = 0

axiom median_condition (x : ℤ) :
  median [1, 2, x, 5] = 3 → x = 4

axiom mean_positive_condition (dataset : List ℤ) :
  mean dataset > 0 → ∀ n ∈ dataset, n > 0

-- Proof statement
theorem correct_number_of_true_statements : 
  (mode_condition ∧ median_condition) ∧ 
  ¬mean_median_condition ∧ 
  ¬mean_positive_condition :=
sorry

end correct_number_of_true_statements_l25_25739


namespace abs_sum_condition_l25_25553

theorem abs_sum_condition (a b : ℝ) (h₁ : |a| = 2) (h₂ : b = -1) : |a + b| = 1 ∨ |a + b| = 3 :=
by
  sorry

end abs_sum_condition_l25_25553


namespace largest_even_digit_multiple_of_9_under_1000_l25_25773

theorem largest_even_digit_multiple_of_9_under_1000 : 
  ∃ n : ℕ, (∀ d ∈ Int.digits 10 n, d % 2 = 0) ∧ n < 1000 ∧ n % 9 = 0 ∧ 
  (∀ m : ℕ, (∀ d ∈ Int.digits 10 m, d % 2 = 0) ∧ m < 1000 ∧ m % 9 = 0 → m ≤ n) ∧ n = 864 :=
sorry

end largest_even_digit_multiple_of_9_under_1000_l25_25773


namespace max_even_differences_l25_25869

theorem max_even_differences : 
  ∃ (f : ℕ → ℕ), ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 25 → 1 ≤ f i ∧ f i ≤ 25 ∧ 
                   ∀ (j k : ℕ), 1 ≤ j ∧ j ≤ 25 ∧ 1 ≤ k ∧ k ≤ 25 ∧ j ≠ k → f j ≠ f k ∧ 
                   (f i - i).even ∧ i - f i ≤ max (f i - i) :=
  sorry

end max_even_differences_l25_25869


namespace angle_MAN_in_rhombus_l25_25643

open Real

theorem angle_MAN_in_rhombus (A B C D M N : Point)
  (h_rhombus : AB A B C D)
  (h_midpoints_M : is_midpoint M B C)
  (h_midpoints_N : is_midpoint N C D)
  (angle_BAD_eq : ∠ B A D = 60) :
  ∠ M A N = 2 * arctan (sqrt 3 / 9) := sorry

end angle_MAN_in_rhombus_l25_25643


namespace find_x_when_y_is_minus_five_l25_25608

theorem find_x_when_y_is_minus_five (x y : ℝ) (h : 16 * 3^x = 7^(y+5)) (hy : y = -5) : x = -4 * (Real.log 2) / (Real.log 3) :=
by
  sorry

end find_x_when_y_is_minus_five_l25_25608


namespace distance_to_directrix_l25_25029

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25029


namespace max_value_of_PQ_l25_25574

noncomputable def maxDistance (P Q : ℝ × ℝ) : ℝ :=
  let dist (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)
  let O1 : ℝ × ℝ := (0, 4)
  dist P Q

theorem max_value_of_PQ:
  ∀ (P Q : ℝ × ℝ),
    (P.1 ^ 2 + (P.2 - 4) ^ 2 = 1) →
    (Q.1 ^ 2 / 9 + Q.2 ^ 2 = 1) →
    maxDistance P Q ≤ 1 + 3 * Real.sqrt 3 :=
by
  sorry

end max_value_of_PQ_l25_25574


namespace gemma_amount_given_l25_25923

theorem gemma_amount_given
  (cost_per_pizza : ℕ)
  (number_of_pizzas : ℕ)
  (tip : ℕ)
  (change_back : ℕ)
  (h1 : cost_per_pizza = 10)
  (h2 : number_of_pizzas = 4)
  (h3 : tip = 5)
  (h4 : change_back = 5) :
  number_of_pizzas * cost_per_pizza + tip + change_back = 50 := sorry

end gemma_amount_given_l25_25923


namespace distance_from_point_A_to_directrix_C_l25_25113

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25113


namespace simple_interest_rate_is_12_l25_25757

-- Given conditions and proven result definition
def principal_SI := 2015.625
def time_SI := 4
def principal_CI := 6000
def time_CI := 2
def rate_CI := 15
def expected_rate_SI := 12

-- Definitions related to compound interest (CI) and simple interest (SI)
def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) : ℝ := 
  P * (1 + r / 100) ^ n - P

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t / 100

-- Main theorem to prove the rate of interest for simple interest is 12%
theorem simple_interest_rate_is_12 : 
  (simple_interest principal_SI expected_rate_SI time_SI = 
  (compound_interest principal_CI rate_CI time_CI) / 2) :=
sorry

end simple_interest_rate_is_12_l25_25757


namespace house_cost_l25_25718

-- Define the initial amount of money
def initial_amount : ℝ := 120000

-- Define the total interest income
def total_interest_income : ℝ := 3920

-- Define the interest rates
def interest_rate_one_third : ℝ := 0.04
def interest_rate_two_thirds : ℝ := 0.05

-- Define the proportions
def one_third : ℝ := 1/3
def two_thirds : ℝ := 2/3

theorem house_cost (x : ℝ)
  (h_remaining : x ≤ initial_amount) 
  (h_total_interest : one_third * (initial_amount - x) * interest_rate_one_third +
                      two_thirds * (initial_amount - x) * interest_rate_two_thirds = total_interest_income) :
    x = 36000 :=
  sorry

end house_cost_l25_25718


namespace infinite_nat_N_polynomial_l25_25308

/-- The theorem states that for any integer k, 
there exists a natural number N = 4k^4 - 8k^2 + 2 such that the polynomial 
x^8 + Nx^4 + 1 can be factored into two fourth-degree polynomials with integer coefficients. -/
theorem infinite_nat_N_polynomial (k : ℤ) : ∃ (N : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧ 
  ∃ (P Q : ℤ[X]), P.degree = 4 ∧ Q.degree = 4 ∧ (X^8 + C (↑N) * X^4 + 1) = P * Q :=
by {
  sorry
}

end infinite_nat_N_polynomial_l25_25308


namespace rectangle_area_with_circles_touching_l25_25840

theorem rectangle_area_with_circles_touching
  (r : ℝ)
  (radius_pos : r = 3)
  (short_side : ℝ)
  (long_side : ℝ)
  (dim_rect : short_side = 2 * r ∧ long_side = 4 * r) :
  short_side * long_side = 72 :=
by
  sorry

end rectangle_area_with_circles_touching_l25_25840


namespace distance_from_point_to_parabola_directrix_l25_25102

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25102


namespace min_value_geq_four_l25_25675

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_geq_four (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  4 ≤ min_value_expression x y z :=
sorry

end min_value_geq_four_l25_25675


namespace price_of_one_liter_l25_25360

theorem price_of_one_liter
  (total_cost : ℝ) (num_bottles : ℝ) (liters_per_bottle : ℝ)
  (H : total_cost = 12 ∧ num_bottles = 6 ∧ liters_per_bottle = 2) :
  total_cost / (num_bottles * liters_per_bottle) = 1 :=
by
  sorry

end price_of_one_liter_l25_25360


namespace annika_current_age_l25_25219

-- Define the conditions
def hans_age_current : ℕ := 8
def hans_age_in_4_years : ℕ := hans_age_current + 4
def annika_age_in_4_years : ℕ := 3 * hans_age_in_4_years

-- lean statement to prove Annika's current age
theorem annika_current_age (A : ℕ) (hyp : A + 4 = annika_age_in_4_years) : A = 32 :=
by
  -- Skipping the proof
  sorry

end annika_current_age_l25_25219


namespace find_extrema_l25_25173

-- Define the variables and the constraints
variables (x y z : ℝ)

-- Define the inequalities as conditions
def cond1 := -1 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 8
def cond2 := 2 ≤ x - y + z ∧ x - y + z ≤ 9
def cond3 := -3 ≤ x + 2 * y - z ∧ x + 2 * y - z ≤ 7

-- Define the function f
def f (x y z : ℝ) := 7 * x + 5 * y - 2 * z

-- State the theorem that needs to be proved
theorem find_extrema :
  (∃ x y z, cond1 x y z ∧ cond2 x y z ∧ cond3 x y z) →
  (-6 ≤ f x y z ∧ f x y z ≤ 47) :=
by sorry

end find_extrema_l25_25173


namespace projection_matrix_correct_l25_25902

def P : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![9/22, -9/22, 3/11],
    ![-9/22, 9/22, -3/11],
    ![3/11, -3/11, 2/11]
  ]

def v : Fin 3 → ℚ := fun i =>
  match i with
  | 0 => x
  | 1 => y
  | 2 => z

def u : Fin 3 → ℚ :=
  ![3, -3, 2]

noncomputable def projection_of_v_on_u (v u : Fin 3 → ℚ) : Fin 3 → ℚ :=
  let dot_product a b := Finset.univ.sum (fun i => a i * b i)
  let scalar := (dot_product v u) / (dot_product u u)
  fun i => scalar * u i

theorem projection_matrix_correct (v : Fin 3 → ℚ) :
  (Matrix.mulVec P v) = projection_of_v_on_u v u :=
sorry

end projection_matrix_correct_l25_25902


namespace gcd_119_34_l25_25380

theorem gcd_119_34 : Nat.gcd 119 34 = 17 := by
  sorry

end gcd_119_34_l25_25380


namespace exists_infinitely_many_natural_numbers_factors_l25_25349

theorem exists_infinitely_many_natural_numbers_factors (k : ℤ) :
  ∃ (P Q : ℤ[X]), 
    (P.degree = 4 ∧ Q.degree = 4) ∧
    (X^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = P * Q) :=
by
  sorry

end exists_infinitely_many_natural_numbers_factors_l25_25349


namespace ed_total_pets_l25_25522

-- Definitions
def num_dogs : ℕ := 2
def num_cats : ℕ := 3
def num_fish (num_birds : ℕ) : ℕ := 3 * num_birds
def num_turtles (num_birds : ℕ) : ℕ := num_birds / 2
def num_pets (num_birds : ℕ) :=
  num_dogs + num_cats + num_fish(num_birds) + num_birds + num_turtles(num_birds)

-- Main theorem
theorem ed_total_pets : num_pets 5 = 27 :=
by {simp, sorry}

end ed_total_pets_l25_25522


namespace scientific_notation_gdp_2022_l25_25486

def gdp_2022_fujian : ℝ := 53100 * 10^9

theorem scientific_notation_gdp_2022 : 
  (53100 * 10^9) = 5.31 * 10^12 :=
by
  -- The proof is based on the understanding that 53100 * 10^9 can be rewritten as 5.31 * 10^12
  -- However, this proof is currently omitted with a placeholder.
  sorry

end scientific_notation_gdp_2022_l25_25486


namespace contrapositive_l25_25374

theorem contrapositive (p q : Prop) : (p → q) → (¬q → ¬p) :=
by
  sorry

end contrapositive_l25_25374


namespace distance_from_A_to_directrix_of_parabola_l25_25056

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25056


namespace triangle_similarity_l25_25263

open EuclideanGeometry

variables {A B C T S B₁ C₁ : Point}

-- Define the conditions
def triangle (A B C : Point) : Prop := A ≠ B ∧ B ≠ C ∧ A ≠ C

def tangent (P : Point) (circumcircle : Circle) : Line := sorry
def circumcircle (A B C : Point) : Circle := sorry

def perpendicular (A B C : Point) : Prop := ∠BAC = π / 2

def on_segment (P A B : Point) : Prop := collinear P A B ∧ dist A B = dist A P + dist P B

def distance_equal (P Q R : Point) : Prop := dist P Q = dist Q R

def similar (A B C D E F : Point) : Prop :=
  ∠BAC = ∠EDF ∧ ∠ACB = ∠DFE ∧ ∠CBA = ∠FED ∧ dist B A / dist A C = dist E D / dist D F

-- Rewrite the problem statement
theorem triangle_similarity
  (h_triangle : triangle A B C)
  (h_tangents_intersect : ∃ circumcircle, tangent B circumcircle ≠ tangent C circumcircle ∧
    intersection (tangent B circumcircle) (tangent C circumcircle) = T)
  (h_S_on_BC : on_segment S B C)
  (h_perpendicular : perpendicular A S T)
  (h_B₁_on_ST : on_segment B₁ S T)
  (h_C₁_on_ST : on_segment C₁ S T)
  (h_B₁_between_C₁_S: segmentOrder S B₁ C₁)
  (h_dist_equal : distance_equal B₁ T ∧ distance_equal B₁ T B ∧ distance_equal C₁ T) :
  similar A B C A B₁ C₁ :=
sorry

end triangle_similarity_l25_25263


namespace geometric_mean_area_l25_25729

theorem geometric_mean_area (A B C O K : Type) 
  (area_ABC area_AOB : ℝ) 
  (right_angle_ABC : Prop)
  (S1 S2 : ℝ)
  (S1_eq : S1 = area_ABC)
  (S2_eq : S2 = area_AOB)
  (K_on_CO : Prop)
  (ABK_right : Prop) : 
  ∃ S_ABK : ℝ, S_ABK = real.sqrt (S1 * S2) := 
by
  sorry

end geometric_mean_area_l25_25729


namespace radius_le_one_fourth_l25_25666

noncomputable
def radius_new_circle (k : Circle) (AB : Diameter k) (C : PointOnSegment AB) : ℝ := sorry

theorem radius_le_one_fourth (k : Circle) (unit_radius : radius k = 1)
                             (AB : Diameter k) (C : PointOnSegment AB) :
                             radius (new_circle k AB C) ≤ 1 / 4 := sorry

end radius_le_one_fourth_l25_25666


namespace exists_infinitely_many_N_l25_25296

theorem exists_infinitely_many_N (k : ℤ) : ∃ N : ℕ, ∃ A B : polynomial ℤ, 
  N = 4 * k^4 - 8 * k^2 + 2 ∧ polynomial.nat_degree A = 4 ∧ polynomial.nat_degree B = 4 ∧
  (polynomial.X^8 + polynomial.C (N:ℤ) * polynomial.X^4 + 1) = A * B :=
sorry

end exists_infinitely_many_N_l25_25296


namespace construct_projections_and_intersection_line_l25_25976

open EuclideanGeometry

noncomputable def projections_of_octagon (A B' C': Point) (B C: Point) (octagon: Octagon A B C D E F G H) : Set Point :=
{ D', E', F', G', H' : Point // 
  are_orthogonal_projections A B' C' B C ∧
  intersection_line_exists octagon (plane_of_triangle A B' C') m 
}

theorem construct_projections_and_intersection_line 
  {A B' C' B C: Point}
  {octagon: Octagon A B C D E F G H}
  (h1: B' = orthogonal_projection B (plane_of_triangle A B' C'))
  (h2: C' = orthogonal_projection C (plane_of_triangle A B' C')) :
  projections_of_octagon A B' C' B C octagon ∃ m := 
sorry

end construct_projections_and_intersection_line_l25_25976


namespace distance_to_directrix_l25_25031

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25031


namespace tricycle_count_l25_25833

variables (b t : ℕ)

theorem tricycle_count :
  b + t = 7 ∧ 2 * b + 3 * t = 19 → t = 5 := by
  intro h
  sorry

end tricycle_count_l25_25833


namespace sqrt_15_estimate_l25_25897

theorem sqrt_15_estimate : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 :=
by
  sorry

end sqrt_15_estimate_l25_25897


namespace total_games_in_season_l25_25844

open Nat

-- Definitions for the problem conditions
def teams_per_division := 8
def total_teams := 16
def divisions := 2
def intra_division_games_per_pair := 3
def inter_division_games_per_pair := 1
def teams_per_division := total_teams / divisions

-- Proof statement
theorem total_games_in_season : 
  let intra_division_games := (teams_per_division * (teams_per_division - 1) / 2) * divisions * intra_division_games_per_pair
  let inter_division_games := (teams_per_division * teams_per_division * inter_division_games_per_pair)
  intra_division_games + inter_division_games = 232 :=
by
  sorry

end total_games_in_season_l25_25844


namespace last_letter_86th_word_l25_25726

theorem last_letter_86th_word :
  let word : List Char := ['A', 'H', 'S', 'M', 'E']
  let perms := word.permutations
  let sorted_perms := perms.sort (λ a b => a.data < b.data)
  let word_86 := sorted_perms.get! 85
  word_86.getLast = 'E' :=
begin
  sorry
end

end last_letter_86th_word_l25_25726


namespace number_of_points_P_l25_25225

theorem number_of_points_P (square : Type) 
  (A B C D O : square) 
  (is_square : ∀ (P : square), (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = O) → (∀ (Q : square), (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = O) → (dist P Q) = (dist Q P))) 
  (isosceles_AB_P : ∀ (P : square), is_isosceles (triangle P A B)) 
  (isosceles_BC_P : ∀ (P : square), is_isosceles (triangle P B C)) 
  (isosceles_CD_P : ∀ (P : square), is_isosceles (triangle P C D)) 
  (isosceles_DA_P : ∀ (P : square), is_isosceles (triangle P D A)) :
  ∃ (P_set : set square), (∀ P ∈ P_set, (is_isosceles (triangle P A B)) ∧ (is_isosceles (triangle P B C)) ∧ (is_isosceles (triangle P C D)) ∧ (is_isosceles (triangle P D A))) ∧ card P_set = 9 := 
begin
  sorry
end

end number_of_points_P_l25_25225


namespace distance_to_directrix_l25_25093

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25093


namespace train_cross_pole_in_5_seconds_l25_25480

noncomputable def kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  speed_kmph * 1000 / 3600

def train_length := 500
def train_speed_kmph := 360
def train_speed_mps := kmph_to_mps train_speed_kmph
def time_to_cross_pole := train_length / train_speed_mps

theorem train_cross_pole_in_5_seconds :
  time_to_cross_pole = 5 := by
  sorry

end train_cross_pole_in_5_seconds_l25_25480


namespace infinite_natural_numbers_factorable_polynomial_l25_25336

theorem infinite_natural_numbers_factorable_polynomial :
  ∃ᶠ N in at_top, ∃ (k : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧
    ∃ (f g : ℤ[x]), (degree f = 4) ∧ (degree g = 4) ∧ (f * g = X^8 + C (N : ℤ) * X^4 + 1) :=
sorry

end infinite_natural_numbers_factorable_polynomial_l25_25336


namespace distance_from_A_to_directrix_on_parabola_l25_25044

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25044


namespace distance_from_A_to_directrix_of_parabola_l25_25065

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25065


namespace max_set_divisibility_l25_25901

theorem max_set_divisibility (n : ℕ) :
  (∃ (s : Finset ℕ),
    ((∀ x ∈ s, ∃ m, x % m = 0 ∧ (m = n ⟹ ∃! x ∈ s, x % n = 0)) ∧
     (∀ k ∈ (Finset.range (n - 1)).filter (λ m, 1 < m), ∃! x ∈ s, x % k = 0))) →
  n ≤ 5 :=
by sorry

end max_set_divisibility_l25_25901


namespace sequence_twentieth_term_l25_25975

theorem sequence_twentieth_term :
  let a : ℕ → ℚ := λ n, if n = 0 then 1 else a n = (a (n - 1)) / (3 * (a (n - 1)) + 1)
  in a(20) = 1 / 58 :=
sorry

end sequence_twentieth_term_l25_25975


namespace distance_from_A_to_directrix_of_parabola_l25_25055

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25055


namespace probability_more_boys_than_girls_l25_25625

open ProbabilityTheory MeasureTheory

-- Define the distribution of the number of children
def P_X (n : ℕ) : ℝ :=
  if n = 0 then 1/15 else
  if n = 1 then 6/15 else
  if n = 2 then 6/15 else
  if n = 3 then 2/15 else 0

-- Define the conditional probability P(B|A_i)
def P_B_given_A (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  if n = 1 then 1/2 else
  if n = 2 then 1/4 else
  if n = 3 then 1/2 else 0

-- Define the overall probability P(B) using the law of total probability
def P_B : ℝ :=
  0 * (1/15) + (1/2) * (6/15) + (1/4) * (6/15) + (1/2) * (2/15)

-- The main theorem stating the required probability
theorem probability_more_boys_than_girls : P_B = 11/30 :=
by
  -- This part is just to show the theorem structure.
  -- The actual proof is not provided here.
  sorry

end probability_more_boys_than_girls_l25_25625


namespace largest_even_digit_multiple_of_nine_l25_25784

theorem largest_even_digit_multiple_of_nine : ∃ n : ℕ, (n < 1000) ∧ (∀ d ∈ digits 10 n, d % 2 = 0) ∧ (n % 9 = 0) ∧ n = 888 := 
by
  sorry

end largest_even_digit_multiple_of_nine_l25_25784


namespace main_theorem_l25_25249

open_locale classical

variables {A B C D E F M : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space M]
  {rhom : ∀ {a b c d: A}, ∀ {ab ac ad bc bd cd: B}, ∀ {angle_abc: float}, rhombus a b c d}

def angle {A B C : Type*} [metric_space A] [metric_space B] [metric_space C] (u v : A) : B := sorry
def length {A B : Type*} [metric_space A] [metric_space B] (u v : A) : B := sorry

axiom quadrilateral_has_equal_sides {A B C D : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] (ab : A) (bc : A) (cd : A) (da : A): 
  length ab bc = length bc cd ∧ length bc cd = length cd da ∧ length cd da = length da ab

axiom angle_ABC_is_60_deg {A B C D : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] (a b c d : A) : angle b c = 60

def M_is_intersection {A B C E F : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space E] [metric_space F] (ce af : A) [has_intersection E F M] := 
  intersection ce af = some_point (sorry : point)

theorem main_theorem {A B C D E F M : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space M]
  (h1 : quadrilateral_has_equal_sides A B C D)
  (h2 : angle_ABC_is_60_deg A B C D)
  (h3 : ∃ M E F : Type*, M_is_intersection (C ∪ E) (A ∪ F)) : 
  length A C ^ 2 = length C M * length C E := 
sorry

end main_theorem_l25_25249


namespace original_number_of_men_total_extra_working_hours_l25_25834

variable (M : ℕ) (H : ℕ)

def work_rate_per_man (total_rate : ℕ) (men : ℕ) : ℕ := total_rate / men

theorem original_number_of_men (total_rate : ℕ) (initial_days : ℕ) (lost_men : ℕ) (extended_days : ℕ) :
  total_rate = 100 → initial_days = 50 → lost_men = 10 → extended_days = 60 → 
  let remaining_men := M - lost_men in 
  (total_rate * initial_days / extended_days) = work_rate_per_man total_rate remaining_men * remaining_men * extended_days → M = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

theorem total_extra_working_hours (original_men remaining_days : ℕ) (extra_days : ℕ) : 
  original_men = 60 → remaining_days = 50 → extra_days = 10 → 
  (original_men - lost_men) * extra_days * H = 500 * H :=
by
  intros h1 h2 h3
  sorry

end original_number_of_men_total_extra_working_hours_l25_25834


namespace distance_from_A_to_directrix_l25_25072

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25072


namespace star_equiv_l25_25541

variable {m n x y : ℝ}

def star (m n : ℝ) : ℝ := (3 * m - 2 * n) ^ 2

theorem star_equiv (x y : ℝ) : star ((3 * x - 2 * y) ^ 2) ((2 * y - 3 * x) ^ 2) = (3 * x - 2 * y) ^ 4 := 
by
  sorry

end star_equiv_l25_25541


namespace distance_from_A_to_directrix_l25_25007

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25007


namespace minimum_rental_cost_l25_25481

theorem minimum_rental_cost :
  ∃ (x y : ℕ), 
    (36 * x + 60 * y ≥ 900) ∧
    (x + y ≤ 21) ∧
    (y - x ≤ 7) ∧
    (1600 * x + 2400 * y = 36800) :=
by {
  -- Here we will show that the solution exists,
  -- which matches the correct answer from the solution.
  use 5, 12,
  simp,
  split,
  { exact nat.le_of_lt (show 1800 > 900, by norm_num) },
  split,
  { exact nat.le_of_lt (show 17 < 21, by norm_num) },
  split,
  { exact nat.le_of_lt (show 7 = 7, by norm_num) },
  { refl }
}

end minimum_rental_cost_l25_25481


namespace distance_from_point_to_parabola_directrix_l25_25110

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25110


namespace correlation_coefficient_height_weight_l25_25232

noncomputable def correlation_coefficient (x y : ℝ) : ℝ := sorry

theorem correlation_coefficient_height_weight 
  (variation_explained_by_height : ℝ) 
  (variation_explained_by_error : ℝ) 
  (height_effect_greater : variation_explained_by_height > variation_explained_by_error)
  (height_variation : variation_explained_by_height = 0.64)
  (error_variation : variation_explained_by_error = 0.36) :
  correlation_coefficient height weight = 0.64 :=
sorry

end correlation_coefficient_height_weight_l25_25232


namespace mathstest_permutations_l25_25498

theorem mathstest_permutations :
  let total_letters := 9
  let s_count := 2
  let t_count := 3
  ∑ i in finset.range 1 total_letters.succ, (1/2) * ∑ j in finset.range 1 (9/2).succ, (6) = 10 :=
by sorry

end mathstest_permutations_l25_25498


namespace Sundays_and_Mondays_are_closed_starting_day_is_thursday_l25_25653

-- Define the days of the week
inductive Day : Type
| Sunday    : Day
| Monday    : Day
| Tuesday   : Day
| Wednesday : Day
| Thursday  : Day
| Friday    : Day
| Saturday  : Day

open Day

-- Function to get the next day's index
def nextDay (d : Day) : Day :=
  match d with
  | Sunday    => Monday
  | Monday    => Tuesday
  | Tuesday   => Wednesday
  | Wednesday => Thursday
  | Thursday  => Friday
  | Friday    => Saturday
  | Saturday  => Sunday

-- Function to get the nth day from a starting day
def nthDayFrom (start : Day) (n : Nat) : Day :=
  match n with
  | 0 => start
  | n + 1 => nthDayFrom (nextDay start) n

-- Total coupons
def total_coupons := 8

-- Check if a day is closed
def isClosed (d : Day) : Bool :=
  d = Sunday ∨ d = Monday

-- Prove that Monday and Sunday are closed
theorem Sundays_and_Mondays_are_closed :
  ∀ d, isClosed d → d = Sunday ∨ d = Monday :=
begin
  intro d,
  intro h,
  exact h,
end

-- Prove that starting day is Thursday
theorem starting_day_is_thursday :
  ∀ (start : Day),
    (forall (i : Nat), i < total_coupons → ¬ isClosed (nthDayFrom start (i * 7))) →
    start = Thursday :=
by
  intro start
  intro h
  -- Proof omitted
  sorry

end Sundays_and_Mondays_are_closed_starting_day_is_thursday_l25_25653


namespace coefficient_of_x_in_expansion_l25_25372

theorem coefficient_of_x_in_expansion : 
  coeff (expand (1-2*x)^4) x = -8 :=
by
  sorry

end coefficient_of_x_in_expansion_l25_25372


namespace difference_is_three_l25_25659

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

end difference_is_three_l25_25659


namespace number_of_divisors_of_fourth_power_l25_25614

theorem number_of_divisors_of_fourth_power (x n : ℕ) (h : x = n^4) : 
  ∃ d, d = 205 ∧ ∃ f : ℕ → ℕ, (∀ i, f(i) ≡ 1 [MOD 4]) ∧ (d = ∏ i in range k, f(i)) :=
by
  sorry

end number_of_divisors_of_fourth_power_l25_25614


namespace minimize_cost_per_km_l25_25794

section ship_cost_minimization

variables (u v k : ℝ) (fuel_cost other_cost total_cost_per_km: ℝ)

-- Condition 1: The fuel cost per unit time is directly proportional to the cube of its speed.
def fuel_cost_eq : Prop := u = k * v^3

-- Condition 2: When the speed of the ship is 10 km/h, the fuel cost is 35 yuan per hour.
def fuel_cost_at_10 : Prop := u = 35 ∧ v = 10

-- Condition 3: The other costs are 560 yuan per hour.
def other_cost_eq : Prop := other_cost = 560

-- Condition 4: The maximum speed of the ship is 25 km/h.
def max_speed : Prop := v ≤ 25

-- Prove that the speed of the ship that minimizes the cost per kilometer is 20 km/h.
theorem minimize_cost_per_km : 
  fuel_cost_eq u v k ∧ fuel_cost_at_10 u v ∧ other_cost_eq other_cost ∧ max_speed v → v = 20 :=
by
  sorry

end ship_cost_minimization

end minimize_cost_per_km_l25_25794


namespace minimum_value_proof_l25_25684

noncomputable def min_value (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : ℝ :=
  infi (λ (x y z : {x : ℝ // 0 < x}),
    x.1 / y.1 + y.1 / z.1 + z.1 / x.1 + y.1 / x.1)

theorem minimum_value_proof (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  min_value a b c h_a h_b h_c = 4 := 
sorry

end minimum_value_proof_l25_25684


namespace visitors_on_previous_day_is_246_l25_25849

def visitors_on_previous_day : Nat := 246
def total_visitors_in_25_days : Nat := 949

theorem visitors_on_previous_day_is_246 :
  visitors_on_previous_day = 246 := 
by
  rfl

end visitors_on_previous_day_is_246_l25_25849


namespace _l25_25137

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25137


namespace probability_of_three_blue_marbles_l25_25457

theorem probability_of_three_blue_marbles
  (red_marbles : ℕ) (blue_marbles : ℕ) (yellow_marbles : ℕ) (total_marbles : ℕ)
  (draws : ℕ) 
  (prob : ℚ) :
  red_marbles = 3 →
  blue_marbles = 4 →
  yellow_marbles = 13 →
  total_marbles = 20 →
  draws = 3 →
  prob = ((4 / 20) * (3 / 19) * (1 / 9)) →
  prob = 1 / 285 :=
by
  intros; 
  sorry

end probability_of_three_blue_marbles_l25_25457


namespace smallest_n_number_of_digits_l25_25262

-- Define the conditions and the target proof problem in Lean 4.
def is_divisible_by (a b : ℕ) : Prop := b ∣ a

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2
def is_perfect_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k^5

noncomputable def number_of_digits (n : ℕ) : ℕ :=
  (Real.log10 (n.toReal)).ceil.toNat

theorem smallest_n_number_of_digits :
  ∃ n : ℕ, is_divisible_by n 12 ∧ is_perfect_cube (n^2) ∧ is_perfect_square (n^3) ∧ is_perfect_fifth_power (n^4) ∧ number_of_digits n = 24 :=
sorry

end smallest_n_number_of_digits_l25_25262


namespace Jane_adds_ribbons_l25_25235

/--
Jane sews 2 dresses a day for 7 days.
Then she sews 3 dresses a day for the next 2 days.
In the end, she adds a certain number of ribbons to each dress and uses 40 ribbons in total.
Prove that Jane adds 2 ribbons to each dress.
-/
theorem Jane_adds_ribbons :
  let dresses_first_period := 7 * 2,
      dresses_second_period := 2 * 3,
      total_dresses := dresses_first_period + dresses_second_period,
      total_ribbons := 40
  in
  total_ribbons / total_dresses = 2 :=
by
  let dresses_first_period := 7 * 2
  let dresses_second_period := 2 * 3
  let total_dresses := dresses_first_period + dresses_second_period
  let total_ribbons := 40
  calc
    total_ribbons / total_dresses = 40 / 20 : by sorry
    ... = 2 : by sorry

end Jane_adds_ribbons_l25_25235


namespace minimum_complex_abs_l25_25676

noncomputable def is_condition_met (z : ℂ) : Prop :=
  complex.abs(z - 10) + complex.abs(z + complex.I * 3) = 15

theorem minimum_complex_abs (z : ℂ) (h : is_condition_met(z)) : complex.abs(z) = 2 :=
sorry

end minimum_complex_abs_l25_25676


namespace vermont_clicked_ads_l25_25415

theorem vermont_clicked_ads :
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  ads_clicked = 68 := by
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  have h1 : ads_clicked = 68 := by sorry
  exact h1

end vermont_clicked_ads_l25_25415


namespace distance_from_A_to_directrix_l25_25151

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25151


namespace largest_even_digit_multiple_of_nine_l25_25785

theorem largest_even_digit_multiple_of_nine : ∃ n : ℕ, (n < 1000) ∧ (∀ d ∈ digits 10 n, d % 2 = 0) ∧ (n % 9 = 0) ∧ n = 888 := 
by
  sorry

end largest_even_digit_multiple_of_nine_l25_25785


namespace john_not_stronger_than_ivan_l25_25277

-- Define strength relations
axiom stronger (a b : Type) : Prop

variable (whiskey liqueur vodka beer : Type)

axiom whiskey_stronger_than_vodka : stronger whiskey vodka
axiom liqueur_stronger_than_beer : stronger liqueur beer

-- Define types for cocktails and their strengths
variable (John_cocktail Ivan_cocktail : Type)

axiom John_mixed_whiskey_liqueur : John_cocktail
axiom Ivan_mixed_vodka_beer : Ivan_cocktail

-- Prove that it can't be asserted that John's cocktail is stronger
theorem john_not_stronger_than_ivan :
  ¬ (stronger John_cocktail Ivan_cocktail) :=
sorry

end john_not_stronger_than_ivan_l25_25277


namespace perpendicular_tangents_exists_inequality_sin_cos_l25_25163

noncomputable def f (x : ℝ) : ℝ := (2/3 : ℝ) * x^3 - x

theorem perpendicular_tangents_exists : 
  ∃ x₁ x₂ : ℝ, x₁ ∈ set.Icc (-1:ℝ) 1 ∧ x₂ ∈ set.Icc (-1:ℝ) 1 ∧ (f x₁ = 0 ∧ f 1 = -1/3 ∨ f x₁ = 0 ∧ f (-1) = 1/3) ∧ (2 * x₁^2 - 1) * (2 * x₂^2 - 1) = -1 :=
sorry

theorem inequality_sin_cos (x : ℝ) :
  |f (Real.sin x) - f (Real.cos x)| ≤ 2 * Real.sqrt 2 / 3 :=
sorry

end perpendicular_tangents_exists_inequality_sin_cos_l25_25163


namespace sum_even_integers_eq_930_l25_25397

theorem sum_even_integers_eq_930 :
  let sum_first_30_even := 2 * (30 * (30 + 1) / 2)
  let sum_consecutive_even (n : ℤ) := (n - 8) + (n - 6) + (n - 4) + (n - 2) + n
  ∀ n : ℤ, sum_first_30_even = 930 → sum_consecutive_even n = 930 → n = 190 :=
by
  intros sum_first_30_even sum_consecutive_even n h1 h2
  sorry

end sum_even_integers_eq_930_l25_25397


namespace equation_has_three_distinct_real_roots_l25_25992

noncomputable def cubic_function (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

noncomputable def equation_roots (a b c x1 x2 : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, f = cubic_function a b c ∧
  (∃ x1 x2 : ℝ, (deriv f x1 = 0) ∧ (deriv f x2 = 0) ∧ (f x1 = x1)) ∧
  (∃ n : ℕ, n = 3 ∧ (count_distinct_real_roots (3 * (f x)^2 + 2 * a * (f x) + b) = n))

-- The statement to be proved
theorem equation_has_three_distinct_real_roots (a b c x1 x2 : ℝ) :
  equation_roots a b c x1 x2 :=
begin
  sorry
end

end equation_has_three_distinct_real_roots_l25_25992


namespace prob_D_l25_25816

variable {P : Type} [Probability P]

def prob_A := (1 : ℚ) / 4
def prob_B := (1 : ℚ) / 3
def prob_C := (1 : ℚ) / 6

theorem prob_D : prob_A + prob_B + prob_C + (D : ℚ) = 1 → D = (1 : ℚ) / 4 := 
by {
  intro h,
  sorry
}

end prob_D_l25_25816


namespace distance_from_A_to_directrix_l25_25018

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25018


namespace product_327_base9_6_base9_l25_25907

theorem product_327_base9_6_base9 : ∃ (n : ℕ), n = 2226 ∧ base_repr n 9 = "2226" := by
  sorry

end product_327_base9_6_base9_l25_25907


namespace largest_even_integer_sum_l25_25395

theorem largest_even_integer_sum :
  let sum_first_30_even := 2 * (List.sum (List.range 30).map(λ n, n + 1)) in
  let n := (sum_first_30_even + 20) / 5 in
  n = 190 :=
by
  sorry

end largest_even_integer_sum_l25_25395


namespace cost_of_brushes_and_canvas_minimum_canvases_cost_effectiveness_l25_25810

-- Part 1: Prove the cost of one box of brushes and one canvas each.
theorem cost_of_brushes_and_canvas (x y : ℕ) 
    (h₁ : 2 * x + 4 * y = 94) (h₂ : 4 * x + 2 * y = 98) :
    x = 17 ∧ y = 15 := by
  sorry

-- Part 2: Prove the minimum number of canvases.
theorem minimum_canvases (m : ℕ) 
    (h₃ : m + (10 - m) = 10) (h₄ : 17 * (10 - m) + 15 * m ≤ 157) :
    m ≥ 7 := by
  sorry

-- Part 3: Prove the cost-effective purchasing plan.
theorem cost_effectiveness (m n : ℕ) 
    (h₃ : m + n = 10) (h₄ : 17 * n + 15 * m ≤ 157) (h₅ : m ≤ 8) :
    (m = 8 ∧ n = 2) := by
  sorry

end cost_of_brushes_and_canvas_minimum_canvases_cost_effectiveness_l25_25810


namespace edwards_final_money_l25_25523

def small_lawn_rate : ℕ := 8
def medium_lawn_rate : ℕ := 12
def large_lawn_rate : ℕ := 15

def first_garden_rate : ℕ := 10
def second_garden_rate : ℕ := 12
def additional_garden_rate : ℕ := 15

def num_small_lawns : ℕ := 3
def num_medium_lawns : ℕ := 1
def num_large_lawns : ℕ := 1
def num_gardens_cleaned : ℕ := 5

def fuel_expense : ℕ := 10
def equipment_rental_expense : ℕ := 15
def initial_savings : ℕ := 7

theorem edwards_final_money : 
  (num_small_lawns * small_lawn_rate + 
   num_medium_lawns * medium_lawn_rate + 
   num_large_lawns * large_lawn_rate + 
   (first_garden_rate + second_garden_rate + (num_gardens_cleaned - 2) * additional_garden_rate) + 
   initial_savings - 
   (fuel_expense + equipment_rental_expense)) = 100 := 
  by 
  -- The proof goes here
  sorry

end edwards_final_money_l25_25523


namespace number_of_classes_l25_25632

-- Define the conditions
def first_term : ℕ := 27
def common_diff : ℤ := -2
def total_students : ℕ := 115

-- Define and prove the main statement
theorem number_of_classes : ∃ n : ℕ, n > 0 ∧ (first_term + (n - 1) * common_diff) * n / 2 = total_students ∧ n = 5 :=
by
  sorry

end number_of_classes_l25_25632


namespace area_ratio_equilateral_triangle_l25_25250

-- Definitions based on the conditions
def is_equilateral_triangle (A B C : ℝ) : Prop :=
  A = B ∧ B = C

def extended_sides (A B C : ℝ) : (ℝ × ℝ × ℝ) :=
  let AB' := A + 2 * A
  let BC' := B + 3 * B
  let CA' := C + 4 * C
  (AB', BC', CA')

-- Lean problem statement
theorem area_ratio_equilateral_triangle (s : ℝ) 
  (h_s: s > 0) :
  let AB := s
  let BC := s
  let CA := s
  ∃ A' B' C' : ℝ, 
    extended_sides AB BC CA = (3 * s, 4 * s, 5 * s) →
    let area_ABC := (s^2 * (real.sqrt 3)) / 4
    let area_A'B'C' := 6 * s^2
    area_A'B'C' / area_ABC = 42 :=
begin
  intro h,
  use [3 * s, 4 * s, 5 * s],
  intro h_extended,
  rw [extended_sides] at h_extended,
  -- remaining proof but skipped with sorry
  sorry
end

end area_ratio_equilateral_triangle_l25_25250


namespace distance_from_A_to_directrix_l25_25020

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25020


namespace find_k_l25_25571

theorem find_k (x y k : ℝ) 
  (h1 : x - y + 5 ≥ 0) 
  (h2 : x ≤ 3) 
  (h3 : x + y + k ≥ 0) 
  (h4 : ∀ x y, (x - y + 5 ≥ 0) ∧ (x ≤ 3) → 2x + 4y ≥ -6) : 
  k = 0 :=
sorry

end find_k_l25_25571


namespace new_garden_is_larger_l25_25447

-- Definitions
def original_length : ℝ := 60
def original_width : ℝ := 20
def unusable_fence : ℝ := 10

-- Conditions
def original_area : ℝ := original_length * original_width
def original_perimeter : ℝ := 2 * (original_length + original_width)
def usable_fence : ℝ := original_perimeter - unusable_fence
def side_length_new_square : ℝ := usable_fence / 4
def new_area : ℝ := side_length_new_square * side_length_new_square

-- Proof statement
theorem new_garden_is_larger : new_area - original_area = 206.25 := by
  sorry

end new_garden_is_larger_l25_25447


namespace xyz_leq_36_l25_25663

theorem xyz_leq_36 {x y z : ℝ} 
    (hx0 : x > 0) (hy0 : y > 0) (hz0 : z > 0) 
    (hx2 : x ≤ 2) (hy3 : y ≤ 3) 
    (hxyz_sum : x + y + z = 11) : 
    x * y * z ≤ 36 := 
by
  sorry

end xyz_leq_36_l25_25663


namespace max_cos_sum_l25_25856

theorem max_cos_sum (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 0 < A) (h3 : A < Real.pi)
  (h4 : 0 < B) (h5 : B < Real.pi) (h6 : 0 < C) (h7 : C < Real.pi) :
  ∃ max_val : ℝ, max_val = (3 / 4) ∧ (∀ (A B C : ℝ), A + B + C = Real.pi → (0 < A) → (A < Real.pi) →
  (0 < B) → (B < Real.pi) → (0 < C) → (C < Real.pi) → (cos A + cos B * cos C ≤ max_val)) :=
sorry

end max_cos_sum_l25_25856


namespace mean_value_z_l25_25385

theorem mean_value_z :
  (mean (7, 11, 5, 9) = 8) →
  (mean (15, z) = 8) →
  z = 1 :=
by
  sorry

end mean_value_z_l25_25385


namespace quadrilateral_side_squares_inequality_l25_25468

theorem quadrilateral_side_squares_inequality :
  ∀ (x1 y1 x2 y2 : ℝ),
    0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ y1 ∧ y1 ≤ 1 ∧
    0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ y2 ∧ y2 ≤ 1 →
    2 ≤ (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ∧ 
          (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ≤ 4 :=
by
  intro x1 y1 x2 y2 h
  sorry

end quadrilateral_side_squares_inequality_l25_25468


namespace a_5_is_31_l25_25974

/-- Define the sequence a_n recursively -/
def a : Nat → Nat
| 0        => 1
| (n + 1)  => 2 * a n + 1

/-- Prove that the 5th term in the sequence is 31 -/
theorem a_5_is_31 : a 5 = 31 := 
sorry

end a_5_is_31_l25_25974


namespace number_of_correct_operations_is_zero_l25_25746

noncomputable def problem_number_of_correct_operations : Prop :=
  let p1 := ∀ x, (deriv (deriv (λ x: ℝ, x^2 * cos x))) x = -2 * x * sin x
  let p2 := ∀ x, (deriv (deriv (λ x: ℝ, 3^x))) x = 3^x * log 3
  let p3 := ∀ x, (deriv (deriv (λ x: ℝ, log x))) x = 1 / (x * (log (real.exp 1)))
  let p4 := ∀ x, (deriv (deriv (λ x: ℝ, exp x / x))) x = (exp x + x * exp x) / x^2
  (¬p1) ∧ (¬p2) ∧ (¬p3) ∧ (¬p4)

theorem number_of_correct_operations_is_zero : problem_number_of_correct_operations :=
by
  sorry

end number_of_correct_operations_is_zero_l25_25746


namespace files_remaining_l25_25440

theorem files_remaining (music_files video_files deleted_files : ℕ) :
  music_files = 26 → 
  video_files = 36 → 
  deleted_files = 48 → 
  (music_files + video_files - deleted_files) = 14 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact Nat.sub_eq_of_eq_add (Nat.add_comm 14 48).symm

end files_remaining_l25_25440


namespace probability_h_neg_probability_h_less_than_zero_l25_25799

noncomputable def h (p : ℕ) : ℝ :=
  p^2 - 13 * p + 40

theorem probability_h_neg :
  (∃ p : ℕ, 1 ≤ p ∧ p ≤ 10 ∧ h p < 0) →
  (6 ≤ p ∧ p ≤ 7) ∧ (h p < 0) :=
by
  sorry

theorem probability_h_less_than_zero :
  let eligible_p := {p : ℕ | 1 ≤ p ∧ p ≤ 10 ∧ h p < 0 }
  (|eligible_p| : ℝ) / 10 = 1 / 5 :=
by
  have eligible_p_card : eligible_p = {6, 7} := sorry,
  rw eligible_p_card,
  norm_num
  sorry

end probability_h_neg_probability_h_less_than_zero_l25_25799


namespace scientific_notation_of_GDP_l25_25484

theorem scientific_notation_of_GDP
  (b : ℕ) (billion_val : b = 10^9) :
  ∀ (n : ℕ) (GDP_billion : n = 53100), 
  let GDP_scientific := (5.31 : ℝ) * 10^13 in
  (n * b : ℝ) = GDP_scientific := 
by
  intros
  unfold billion_val GDP_billion
  unfold GDP_scientific
  sorry

end scientific_notation_of_GDP_l25_25484


namespace correct_judgment_correct_judgment_l25_25947

variables p q : Prop

-- Given conditions
def p := 2 < 3
def q := 2 > 3

-- Lean statement to prove the correct judgments
theorem correct_judgment (h₁ : p) (h₂ : ¬ q) : (p ∨ q) ∧ ¬ (p ∧ q) ∧ ¬ ¬p :=
by {
   -- Translate each judgment condition to a corresponding logical proposition
   exact ⟨or.inl h₁, λ h, h₂ (h.right), λ h, h₁ h⟩,
}

-- Since we are asked not to provide the proof, 'sorry' will be used
theorem correct_judgment (h₁ : p) (h₂ : ¬ q) : (p ∨ q) ∧ ¬ (p ∧ q) ∧ ¬ ¬ p :=
sorry

end correct_judgment_correct_judgment_l25_25947


namespace pyramid_intersection_l25_25466

-- Noncomputable theory to handle divisions and real number operations
noncomputable theory

-- Defining a structure for a 3D point
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Defining a function to calculate distance between two points in 3D
def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2)

-- Axiom stating the main theorem
theorem pyramid_intersection (S A B C D : Point3D) (h : distance S A ≠ 0 ∧ distance S B ≠ 0 ∧ distance S C ≠ 0 ∧ distance S D ≠ 0) :
  1 / distance S A + 1 / distance S C = 1 / distance S B + 1 / distance S D := 
sorry

end pyramid_intersection_l25_25466


namespace average_payment_is_697_l25_25821

-- Define the properties of the installments
def total_installments : ℕ := 100
def first_payment : ℕ := 620
def additional_payment : ℕ := 110
def number_first_payments : ℕ := 30
def number_remaining_payments : ℕ := total_installments - number_first_payments

-- Define the overall question: what is the average payment?
theorem average_payment_is_697 :
  let total_first_payments := number_first_payments * first_payment,
      remaining_payment := first_payment + additional_payment,
      total_remaining_payments := number_remaining_payments * remaining_payment,
      total_amount := total_first_payments + total_remaining_payments,
      average_payment := total_amount / total_installments in
  average_payment = 697 := by
  sorry

end average_payment_is_697_l25_25821


namespace polynomial_is_monic_l25_25442

noncomputable def f : ℝ → ℝ := sorry

variables (h1 : f 1 = 3) (h2 : f 2 = 12) (h3 : ∀ x : ℝ, f x = x^2 + 6*x - 4)

theorem polynomial_is_monic (f : ℝ → ℝ) (h1 : f 1 = 3) (h2 : f 2 = 12) (h3 : ∀ x : ℝ, f x = x^2 + x + b) : 
  ∀ x : ℝ, f x = x^2 + 6*x - 4 :=
by sorry

end polynomial_is_monic_l25_25442


namespace number_of_reseated_women_l25_25409

noncomputable def S : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 2
| (n+3) := S (n+2) + S (n+1) + S n

theorem number_of_reseated_women : S 12 = 927 := by
  sorry

end number_of_reseated_women_l25_25409


namespace seeds_germination_l25_25910

theorem seeds_germination (seed_plot1 seed_plot2 : ℕ) (germ_rate2 total_germ_rate : ℝ) (germinated_total_pct : ℝ)
  (h1 : seed_plot1 = 300)
  (h2 : seed_plot2 = 200)
  (h3 : germ_rate2 = 0.35)
  (h4 : germinated_total_pct = 28.999999999999996 / 100) :
  (germinated_total_pct * (seed_plot1 + seed_plot2) - germ_rate2 * seed_plot2) / seed_plot1 * 100 = 25 :=
by sorry  -- Proof not required

end seeds_germination_l25_25910


namespace jared_annual_earnings_l25_25914

open Nat

noncomputable def diploma_monthly_pay : ℕ := 4000
noncomputable def months_in_year : ℕ := 12
noncomputable def multiplier : ℕ := 3

theorem jared_annual_earnings :
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  jared_annual_earnings = 144000 :=
by
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  exact sorry

end jared_annual_earnings_l25_25914


namespace jared_annual_earnings_l25_25919

-- Defining conditions as constants
def diploma_pay : ℕ := 4000
def degree_multiplier : ℕ := 3
def months_in_year : ℕ := 12

-- Goal: Prove that Jared's annual earnings are $144,000
theorem jared_annual_earnings : diploma_pay * degree_multiplier * months_in_year = 144000 := by
  sorry

end jared_annual_earnings_l25_25919


namespace prob_sum_even_l25_25600

def set_a := {11, 44, 55}
def set_b := {01}

def is_even (n : Int) : Bool :=
  n % 2 = 0

noncomputable def probability_even_sum : ℚ :=
  let pairs := [(11, 01), (44, 01), (55, 01)]
  let even_sums := pairs.count (fun (x, y) => is_even (x + y))
  even_sums / pairs.length

theorem prob_sum_even : probability_even_sum = 1 / 3 := by
  sorry

end prob_sum_even_l25_25600


namespace exists_infinitely_many_natural_numbers_factors_l25_25351

theorem exists_infinitely_many_natural_numbers_factors (k : ℤ) :
  ∃ (P Q : ℤ[X]), 
    (P.degree = 4 ∧ Q.degree = 4) ∧
    (X^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = P * Q) :=
by
  sorry

end exists_infinitely_many_natural_numbers_factors_l25_25351


namespace min_cost_trip_l25_25700

-- Conditions
def distance_DF : ℝ := 3500
def distance_DE : ℝ := 4000
def bus_fare_per_km : ℝ := 0.20
def airplane_base_fare_short : ℝ := 120
def airplane_base_fare_long : ℝ := 150
def airplane_fare_per_km : ℝ := 0.12

-- Calculate Distance EF using Pythagoras' theorem
noncomputable def distance_EF : ℝ := Real.sqrt (distance_DE ^ 2 - distance_DF ^ 2)

-- Calculate costs of travel segments
noncomputable def cost_bus (distance : ℝ) : ℝ := distance * bus_fare_per_km
noncomputable def cost_plane (distance : ℝ) : ℝ :=
  if distance <= 3000 then distance * airplane_fare_per_km + airplane_base_fare_short
  else distance * airplane_fare_per_km + airplane_base_fare_long

-- Calculate least expensive options
noncomputable def min_cost_DE : ℝ := min (cost_bus distance_DE) (cost_plane distance_DE)
noncomputable def min_cost_EF : ℝ := min (cost_bus distance_EF) (cost_plane distance_EF)
noncomputable def min_cost_FD : ℝ := min (cost_bus distance_DF) (cost_plane distance_DF)

-- Calculate total minimum cost for the entire trip
noncomputable def total_min_cost : ℝ := min_cost_DE + min_cost_EF + min_cost_FD

-- Proof Statement
theorem min_cost_trip : total_min_cost ≈ 1552.38 := by
  sorry

end min_cost_trip_l25_25700


namespace exists_unit_volume_tetrahedron_with_interior_point_l25_25555

theorem exists_unit_volume_tetrahedron_with_interior_point 
  (n : ℕ)
  (points : Fin n → ℝ³)
  (h_no_four_coplanar : ∀ (p q r s : Fin n), ¬ coplanar ({points p, points q, points r, points s }))
  (h_max_volume : ∀ (p q r s : Fin n), volume (tetrahedron (points p) (points q) (points r) (points s)) ≤ 0.037) :
  ∃ (t : Tetrahedron), volume t = 1 ∧ (∃ i : Fin n, point_in_interior (points i) t) :=
by {
  sorry
}

end exists_unit_volume_tetrahedron_with_interior_point_l25_25555


namespace max_cos_sum_l25_25861

theorem max_cos_sum
  (A B C : ℝ)
  (h : A + B + C = 180) : 
  ∃ A B C, (A + B + C = 180 ∧ 
  (cos(A) + (1 / 2) * (cos(B - C)) + (1 / 2) * (- cos A)) ≤ 1) := by
  sorry

end max_cos_sum_l25_25861


namespace distance_from_point_to_parabola_directrix_l25_25104

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25104


namespace equidistant_point_in_xz_plane_l25_25906

theorem equidistant_point_in_xz_plane :
  ∃ (x z : ℝ),
    (∀ (p : ℝ × ℝ × ℝ), 
    p = (1, -1, 0) ∨ p = (2, 1, 2) ∨ p = (3, 2, -1) →
    dist (x, 0, z) p = dist (x, 0, z) (p ≠ p)) → 
    x = 31 / 10 ∧ z = 1 / 5 := 
sorry

end equidistant_point_in_xz_plane_l25_25906


namespace distance_from_point_A_to_directrix_C_l25_25119

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25119


namespace matrix_arithmetic_sum_XY_l25_25884

theorem matrix_arithmetic_sum_XY :
  ∃ (X Y : ℚ), X + Y = 98 / 3 ∧ 
    (∀ (i : ℕ), (0 ≤ i ∧ i < 4) → 
    ∃ (d1 d4 d2 d3 : ℚ),
    (∀ (j : ℕ), (0 ≤ j ∧ j < 4) → 
      ((i = 0 → (3 + j * d1) = nth_matrix_element(i, j)) ∧
      (i = 3 → (4 + j * d4) = nth_matrix_element(i, j)) ∧
      (i = 1 → (8 + j * d2) = nth_matrix_element(i, j)) ∧
      (i = 2 → (13 + j * d3) = nth_matrix_element(i, j))))) := sorry

end matrix_arithmetic_sum_XY_l25_25884


namespace max_area_triangle_OAB_l25_25161

noncomputable def ellipse (x y : ℝ) : Prop := 
  (x^2 / 6) + (y^2 / 3) = 1

def point_P : ℝ × ℝ := (1, 1/2)

def line_through_P (a b : ℝ × ℝ) : Prop := 
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) = a ∨ (x, y) = b → 
  y = k * (x - 1) + 1/2 

theorem max_area_triangle_OAB :
  ∀ (A B : ℝ × ℝ), ellipse A.fst A.snd →
                    ellipse B.fst B.snd →
                    A ≠ point_P ∧ B ≠ point_P →
                    ¬ (A = (0, 0) ∨ B = (0, 0)) →
                    ∃ k, line_through_P A B →
                    (∀ x y : ℝ, (x, y) = A ∨ (x, y) = B → y = k * (x - 1) + 1/2) →
  let S_OAB := (1/2) * |A.1 * B.2 - A.2 * B.1| in
  S_OAB ≤ 3 * Real.sqrt 6 / 4 :=
sorry

end max_area_triangle_OAB_l25_25161


namespace count_positive_values_x_l25_25182

theorem count_positive_values_x : 
  let S := {x : ℕ | x > 0 ∧ 3 * x < 100 ∧ 4 * x ≥ 100} in
  S.card = 9 := by
  sorry

end count_positive_values_x_l25_25182


namespace find_x_find_y_l25_25188

-- Define the conditions and statements

def frac (a b : ℝ) := a / b

theorem find_x (a b : ℝ) (h : b ≠ 0) (hx : (a - x) / (b - x) = (a / b)^2) :
  x = (a * b) / (a + b) :=
sorry

theorem find_y (a b : ℝ) (h : b ≠ 0) (hy : (a - y) / (b - y) = (a / b)^3) :
  y = (a * b * (a + b)) / (b^2 + a * b + a^2) :=
sorry

end find_x_find_y_l25_25188


namespace distance_from_point_A_to_directrix_C_l25_25115

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25115


namespace equation_of_line_l25_25565

noncomputable def point : Type := ℝ × ℝ

def hyperbola (p : point) : Prop := 
  let (x, y) := p
  x^2 - (y^2 / 2) = 1

def midpoint (p₁ p₂ : point) (m : point) : Prop := 
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (xm, ym) := m
  xm = (x₁ + x₂) / 2 ∧ ym = (y₁ + y₂) / 2

def line (M : point) (k : ℝ) : set point :=
  { p | let (x, y) := p in y - snd M = k * (x - fst M) }

theorem equation_of_line : 
  ∃ l : set point, l = line (2, 1) 4 ∧ 
                   ∃ A B : point, A ≠ B ∧ hyperbola A ∧ hyperbola B ∧ 
                                  midpoint A B (2, 1) := 
sorry

end equation_of_line_l25_25565


namespace distance_from_A_to_directrix_l25_25070

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25070


namespace coin_balancing_problem_l25_25658

theorem coin_balancing_problem :
  ∃ (a b c : ℕ),
    a + b + c = 99 ∧
    2 * a + 3 * b + c = 297 ∧
    3 * a + b + 2 * c = 297 ∧
    (a = 82 ∧ b = 43 ∧ c = 4) :=
by
  use 82, 43, 4
  simp
  exact ⟨rfl, rfl, rfl⟩

end coin_balancing_problem_l25_25658


namespace tanner_remaining_money_l25_25725
-- Import the entire Mathlib library

-- Define the conditions using constants
def s_Sep : ℕ := 17
def s_Oct : ℕ := 48
def s_Nov : ℕ := 25
def v_game : ℕ := 49

-- Define the total amount left and prove it equals 41
theorem tanner_remaining_money :
  (s_Sep + s_Oct + s_Nov - v_game) = 41 :=
by { sorry }

end tanner_remaining_money_l25_25725


namespace top_three_cards_red_probability_l25_25845

-- Definitions for the deck and the condition sets
def num_cards : ℕ := 52
def num_red_cards : ℕ := 26
def num_top_cards : ℕ := 3

-- Function to calculate the number of ways to choose k cards from a total of n cards
def choose (n k : ℕ) : ℕ := 
  if h : n ≥ k then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))
  else 0

-- Calculation of the probability
def probability_top_three_cards_red : ℚ :=
  (num_red_cards * (num_red_cards - 1) * (num_red_cards - 2)) / (num_cards * (num_cards - 1) * (num_cards - 2))

-- The proof problem statement
theorem top_three_cards_red_probability :
  probability_top_three_cards_red = 200 / 1701 :=
by sorry

end top_three_cards_red_probability_l25_25845


namespace sequence_nth_term_l25_25563

theorem sequence_nth_term (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * (a n) + 3) : a 11 = 2^11 - 3 := 
sorry

end sequence_nth_term_l25_25563


namespace total_students_in_class_l25_25628

theorem total_students_in_class (french german both neither : ℕ)
  (h1 : french = 41)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 33) : 
  french - both + german - both + both + neither = 87 := 
by
  rw [h1, h2, h3, h4]
  sorry

end total_students_in_class_l25_25628


namespace solve_for_x_l25_25764

theorem solve_for_x (x y : ℝ) : 3 * x + 4 * y = 5 → x = (5 - 4 * y) / 3 :=
by
  intro h
  sorry

end solve_for_x_l25_25764


namespace probability_ANN9_l25_25223

-- Conditions in the problem
def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def is_non_vowel (c : Char) : Prop := 
  c ∈ "BCDFGHJKLMNPQRSTVWXYZ".to_list.to_set

def is_digit (c : Char) : Prop := c ∈ "0123456789".to_list.to_set

def valid_license_plate (plate : String) : Prop :=
  plate.length = 4 ∧
  is_vowel plate[0] ∧
  plate[1] = plate[2] ∧ is_non_vowel plate[1] ∧
  is_digit plate[3]

-- The statement of the problem in Lean
theorem probability_ANN9 :
  let total_plates := 5 * 21 * 10 in
  let favorable_outcomes := 1 in
  valid_license_plate "ANN9" →
  Real.ofRat favorable_outcomes / Real.ofRat total_plates = 1 / 1050 :=
by
  sorry

end probability_ANN9_l25_25223


namespace find_minimum_value_of_f_in_interval_l25_25164

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)
def interval := set.Ico (0 : ℝ) (2 * Real.pi)
def min_value := - (3 * Real.sqrt 3) / 2

theorem find_minimum_value_of_f_in_interval : 
  (∀ x ∈ interval, f x ≥ min_value) ∧ (∃ x ∈ interval, f x = min_value) :=
sorry

end find_minimum_value_of_f_in_interval_l25_25164


namespace distance_to_directrix_l25_25092

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25092


namespace trains_clear_time_l25_25413

noncomputable def speed_kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000) / 3600

def length_train_1 : ℕ := 220
def length_train_2 : ℕ := 275
def speed_train_1_kmh : ℕ := 120
def speed_train_2_kmh : ℕ := 90

def speed_train_1_ms : ℕ := speed_kmh_to_ms speed_train_1_kmh
def speed_train_2_ms : ℕ := speed_kmh_to_ms speed_train_2_kmh

def total_length : ℕ := length_train_1 + length_train_2
def relative_speed : ℕ := speed_train_1_ms + speed_train_2_ms

def time_to_clear : Float := total_length.toFloat / relative_speed.toFloat

theorem trains_clear_time :
  time_to_clear ≈ 8.48 :=
by
  sorry

end trains_clear_time_l25_25413


namespace sufficient_condition_l25_25390

variable (a b : ℝ)
variable (f : ℝ → ℝ)

theorem sufficient_condition (x : ℝ) (h : x ∉ Icc a b) : f(x) ≥ 0 :=
sorry

end sufficient_condition_l25_25390


namespace has_two_distinct_roots_and_ordered_l25_25597

-- Define the context and the conditions of the problem.
variables (a b c : ℝ) (h : a < b) (h2 : b < c)

-- Define the quadratic function derived from the problem.
def quadratic (x : ℝ) : ℝ :=
  (x - a) * (x - b) + (x - a) * (x - c) + (x - b) * (x - c)

-- State the main theorem.
theorem has_two_distinct_roots_and_ordered:
  ∃ x1 x2 : ℝ, quadratic a b c x1 = 0 ∧ quadratic a b c x2 = 0 ∧ a < x1 ∧ x1 < b ∧ b < x2 ∧ x2 < c :=
sorry

end has_two_distinct_roots_and_ordered_l25_25597


namespace find_angle_C_find_sum_a_b_l25_25209

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  c = 7 / 2 ∧
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 ∧
  (Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1))

theorem find_angle_C (a b c A B C : ℝ) (h : triangle_condition a b c A B C) : C = Real.pi / 3 :=
  sorry

theorem find_sum_a_b (a b c A B C : ℝ) (h : triangle_condition a b c A B C) (hC : C = Real.pi / 3) : a + b = 11 / 2 :=
  sorry

end find_angle_C_find_sum_a_b_l25_25209


namespace trigonometric_identity_l25_25160

theorem trigonometric_identity 
  (α : ℝ) 
  (x : ℝ) 
  (h1 : ∃ θ, θ = α ∧ (π / 2 < θ ∧ θ < π)) -- α is in the second quadrant
  (h2 : P : ℝ × ℝ := (x, sqrt 5))  -- Point P (x, sqrt(5)) on the terminal side of α
  (h3 : cos α = (sqrt 2 / 4) * x) -- cos α = (sqrt 2 / 4) * x
  : 4 * cos (α + π / 2) - 3 * tan α = sqrt 15 - sqrt 10 := 
sorry

end trigonometric_identity_l25_25160


namespace distance_from_A_to_directrix_l25_25004

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25004


namespace distance_to_directrix_l25_25091

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25091


namespace d_not_suff_nec_condition_l25_25254

variable {d : ℝ} {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}

-- Definitions based on conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d

def sum_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) (λ i, a_n i)

def is_increasing_sequence (S_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S_n (n + 1) > S_n n

-- Statement of the equivalence problem
theorem d_not_suff_nec_condition (hd : is_arithmetic_sequence a_n d)
  (hS : ∀ n, S_n n = sum_first_n_terms a_n n) :
  ¬ ( (∀ S_n, is_increasing_sequence S_n → d > 0)
  ∧ (d > 0 → ∀ S_n, is_increasing_sequence S_n)) :=
sorry

end d_not_suff_nec_condition_l25_25254


namespace ellipse_eq_and_max_AB_l25_25960

-- Given conditions and problem definitions
variables (G M : Type) [Ellipse G] [Circle M]
variable (O : Point)
variables (e R : Real) [hR : 1 < R ∧ R < 2]
variables (a b c k m x y x₀ y₀ : Real)
variables (l : Line) (A B : Point)

-- Ellipse properties
axiom center_ellipse : Center G = O
axiom axes_symmetry : AxesSymmetry G = coord_axes
axiom focus_of_parabola (hx2 : y^2 = 4 * sqrt (3) * x) : Focus hx2 O G
axiom eccentricity : e = sqrt (3) / 2

-- Circle properties
axiom circle_eq : M = Circle(O, R)

-- Tangency conditions
axiom tangent_l_line : TangentLine l M A
axiom tangent_l_ellipse : TangentLine l G B

-- The goal to prove
theorem ellipse_eq_and_max_AB :
  (EllEq G (x^2 / 4 + y^2 = 1)) ∧ (R = sqrt 2 → |AB| = 1) :=
by
  sorry

end ellipse_eq_and_max_AB_l25_25960


namespace distance_from_point_to_parabola_directrix_l25_25112

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25112


namespace price_of_one_liter_l25_25361

theorem price_of_one_liter
  (total_cost : ℝ) (num_bottles : ℝ) (liters_per_bottle : ℝ)
  (H : total_cost = 12 ∧ num_bottles = 6 ∧ liters_per_bottle = 2) :
  total_cost / (num_bottles * liters_per_bottle) = 1 :=
by
  sorry

end price_of_one_liter_l25_25361


namespace vector_magnitude_proof_l25_25957

noncomputable def vector_magnitude (v : ℝˣ ) : ℝ := 
  Real.sqrt (v.dot v)

theorem vector_magnitude_proof (a b : ℝˣ ) 
  (h_angle : ∀ (θ : ℝ), θ = 3 * Real.pi / 4 )
  (h1 : vector_magnitude a = Real.sqrt 2)
  (h2 : vector_magnitude b = 3)
  :
  vector_magnitude (a + (2*b)) = Real.sqrt 26 := 
sorry

end vector_magnitude_proof_l25_25957


namespace angle_bisector_AP_l25_25634

-- Definitions of points and circles in the given conditions
variables (A B C D E P : Point)
variables (O1 O2 : Circle)
variables (circle_inscribed : IsInscribedTriangle A B C)
variables (largest_angle_A : LargestAngle A B C)
variables (midpoint_arc_D : MidpointArc D A B C)
variables (midpoint_arc_E : MidpointArc E A C B)
variables (circle_O1 : CircleThroughPoints A B)
variables (tangent_O1_AC : TangentToLine O1 AC)
variables (circle_O2 : CircleThroughPoints A E)
variables (tangent_O2_AD : TangentToLine O2 AD)
variables (intersect_O1_O2_A_P : IntersectionPoints O1 O2 A P)

-- Statement of the proof problem
theorem angle_bisector_AP (h: IsInscribedTriangle A B C)
                           (largest_angle_A: LargestAngle A B C)
                           (midpoint_arc_D: MidpointArc D A B C)
                           (midpoint_arc_E: MidpointArc E A C B)
                           (circle_O1: CircleThroughPoints A B)
                           (tangent_O1_AC: TangentToLine O1 AC)
                           (circle_O2: CircleThroughPoints A E)
                           (tangent_O2_AD: TangentToLine O2 AD)
                           (intersect_O1_O2_A_P: IntersectionPoints O1 O2 A P) :
  Bisector AP BAC :=
begin
  sorry -- Proof goes here
end

end angle_bisector_AP_l25_25634


namespace exists_infinite_N_l25_25340

theorem exists_infinite_N (k : ℤ) : ∃ᶠ N : ℕ in at_top, 
  ∃ (A B : Polynomial ℤ), 
    (Polynomial.natDegree A = 4 ∧ Polynomial.natDegree B = 4) ∧
    (Polynomial.mul A B = Polynomial.X^8 + Polynomial.C N * Polynomial.X^4 + 1) :=
begin
  -- Proof goes here
  sorry
end

end exists_infinite_N_l25_25340


namespace fraction_of_squares_are_integers_l25_25248

theorem fraction_of_squares_are_integers (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_int : (x^2 - 1) / (y + 1) + (y^2 - 1) / (x + 1) ∈ ℤ) : 
  (x^2 - 1) / (y + 1) ∈ ℤ ∧ (y^2 - 1) / (x + 1) ∈ ℤ := 
sorry

end fraction_of_squares_are_integers_l25_25248


namespace area_of_sector_correct_l25_25368

-- Define the radius and central angle
def radius : ℝ := 12
def theta_deg : ℝ := 54

-- Define the formula for the area of a sector
def area_of_sector (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * Real.pi * (r^2)

-- State the theorem with the correct area
theorem area_of_sector_correct :
  area_of_sector radius theta_deg = 21.6 * Real.pi :=
by
  -- Proof is omitted
  sorry

end area_of_sector_correct_l25_25368


namespace max_single_player_salary_l25_25213

-- Define the constants related to the problem
def numPlayers : Nat := 25
def minSalary : ℕ := 18000
def totalSalaryCap : ℕ := 800000

-- Define the statement of the theorem we want to prove
theorem max_single_player_salary :
  (∃ maxSalary : ℕ, ∀ (salaries : Fin numPlayers → ℕ),
    (∀ i, salaries i ≥ minSalary) →
    (∑ i, salaries i ≤ totalSalaryCap) →
    (∃ i, salaries i = maxSalary) ∧ maxSalary = 368000) :=
by
  sorry

end max_single_player_salary_l25_25213


namespace largest_even_digit_multiple_of_9_under_1000_l25_25779

noncomputable def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

noncomputable def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else n.digits 10

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (digits n).sum

noncomputable def all_even_digits (n : ℕ) : Prop :=
  ∀ d ∈ digits n, is_even_digit d

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ n, n < 1000 ∧ all_even_digits n ∧ sum_of_digits n % 9 = 0 ∧ n = 360 :=
begin
  use 360,
  split,
  { exact nat.lt_succ_self 359 },
  split,
  { intros d hd,
    unfold digits at hd,
    rw list.mem_iff_exists_get at hd,
    rcases hd with ⟨k, hk⟩,
    repeat { rw list.get? },
    simp only [digits] at hk,
    have : k < 3 := (nat.lt_of_succ_lt_succ (list.length_le_of_lt_some hk)).trans_le (by norm_num),
    interval_cases k,
    { rw [hk, nat.digits, nat.digits_aux'],
      norm_num [is_even_digit] },
    { rw [hk, nat.digits, nat.digits_aux', nat.div_eq, nat.mod_eq_of_lt, nat.add_eq_zero_eq_zero_and_eq_zero, nat.zero_eq, eq_self_iff_true, true_and],
      norm_num [is_even_digit] },
    { rw [hk, nat.digits, nat.digits_aux', nat.div_eq, nat.mod_eq_of_lt, nat.add_eq_zero_eq_zero_and_eq_zero, nat.zero_eq, eq_self_iff_true, true_and],
      norm_num [is_even_digit] } },
  { simp only [sum_of_digits, digits],
    exact nat.digits_sum_eq 360 10,
    exact dec_trivial },
  { refl }
end

end largest_even_digit_multiple_of_9_under_1000_l25_25779


namespace green_ball_probability_l25_25514

def containerA := (10, 5)  -- (red balls, green balls)
def containerB := (5, 5)
def containerC := (3, 7)

def totalBalls (container : (ℕ, ℕ)) : ℕ :=
  container.1 + container.2

def greenProbability (container : (ℕ, ℕ)) : ℚ :=
  container.2 / ↑(totalBalls container)

def probEachContainer : ℚ := 1 / 3

def probGreenFromA : ℚ := probEachContainer * greenProbability containerA
def probGreenFromB : ℚ := probEachContainer * greenProbability containerB
def probGreenFromC : ℚ := probEachContainer * greenProbability containerC

def totalProbGreen : ℚ :=
  probGreenFromA + probGreenFromB + probGreenFromC

theorem green_ball_probability :
  totalProbGreen = 23 / 45 := by
  sorry

end green_ball_probability_l25_25514


namespace poly_factorable_l25_25303

-- Define the specific form of N.
noncomputable def N (k : ℤ) : ℕ := 4 * k^4 - 8 * k^2 + 2

-- Define the polynomial.
noncomputable def poly (N : ℕ) : ℤ[X] := X^8 + N * X^4 + 1

-- Define the fourth-degree polynomials used in the factorization.
noncomputable def poly1 (k : ℤ) : ℤ[X] :=
  X^4 - 2 * k * X^3 + 2 * k^2 * X^2 - 2 * k * X + 1

noncomputable def poly2 (k : ℤ) : ℤ[X] := 
  X^4 + 2 * k * X^3 + 2 * k^2 * X^2 + 2 * k * X + 1

-- The proof statement: prove that the polynomial can be factored into the specified fourth-degree polynomials.
theorem poly_factorable : ∀ k : ℤ, ∃ p1 p2 : ℤ[X], poly (N k) = p1 * p2 :=
by
  intro k
  use (poly1 k)
  use (poly2 k)
  sorry

end poly_factorable_l25_25303


namespace ben_has_20_mms_l25_25875

theorem ben_has_20_mms (B_candies Ben_candies : ℕ) 
  (h1 : B_candies = 50) 
  (h2 : B_candies = Ben_candies + 30) : 
  Ben_candies = 20 := 
by
  sorry

end ben_has_20_mms_l25_25875


namespace root_interval_l25_25165

noncomputable def f (x : ℝ) : ℝ := 3 / x - Real.logb 2 x

theorem root_interval (k : ℤ) (h : ∃ x₀ : ℝ, f x₀ = 0 ∧ x₀ ∈ set.Ioo (k : ℝ) (k + 1)) : k = 2 :=
by
  sorry

end root_interval_l25_25165


namespace quadratic_residues_mod_pq_quadratic_residues_mod_pk_l25_25185

-- Define unique primes p and q
variables {p q : ℕ} 
variable (primes : nat.prime p ∧ nat.prime q ∧ p ≠ q)

-- Define a positive integer k
variable {k : ℕ}
variable (positive_int : k > 0)

theorem quadratic_residues_mod_pq (primes : nat.prime p ∧ nat.prime q ∧ p ≠ q) :
  quadratic_residues_count (p * q) = ((p - 1) / 2) * ((q - 1) / 2) := sorry

theorem quadratic_residues_mod_pk (primes : nat.prime p) (positive_int : k > 0) :
  quadratic_residues_count (p^k) = ((p - 1) * p^(k - 1)) / 2 := sorry

end quadratic_residues_mod_pq_quadratic_residues_mod_pk_l25_25185


namespace max_cos_sum_l25_25859

theorem max_cos_sum
  (A B C : ℝ)
  (h : A + B + C = 180) : 
  ∃ A B C, (A + B + C = 180 ∧ 
  (cos(A) + (1 / 2) * (cos(B - C)) + (1 / 2) * (- cos A)) ≤ 1) := by
  sorry

end max_cos_sum_l25_25859


namespace proof_triangle_abc_l25_25229

def triangle_abc (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : Prop :=
  let a := dist B C
  let b := dist A C
  let c := dist A B
  a = 5 ∧ (b - c = 3) ∧ (r a b c < 2) ∧ (2 < r_a a b c)

noncomputable def r (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := sqrt (s * (s - a) * (s - b) * (s - c))
  K / s

noncomputable def r_a (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := sqrt (s * (s - a) * (s - b) * (s - c))
  K / (s - a)

theorem proof_triangle_abc (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (h : triangle_abc A B C) : Prop :=
  let a := dist B C
  let b := dist A C
  let c := dist A B
  r a b c < 2 ∧ 2 < r_a a b c

#check proof_triangle_abc

end proof_triangle_abc_l25_25229


namespace distance_from_point_to_parabola_directrix_l25_25108

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25108


namespace probability_of_girl_pairs_nearest_hundredth_l25_25455

noncomputable def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem probability_of_girl_pairs_nearest_hundredth :
  let total_pairings := (factorial 16) / ((factorial 2) ^ 8 * (factorial 8))
  let no_girl_pairs := (factorial 8) * (factorial 8)
  let probability_no_girl_pairs := ((factorial 8)^3 * 2^8) / factorial 16
  let probability_girl_pairs := 1 - probability_no_girl_pairs
  in Float.to IEEE754Float decimal (probability_girl_pairs) = 0.98 :=
by
  sorry

end probability_of_girl_pairs_nearest_hundredth_l25_25455


namespace no_solution_inequality_system_l25_25204

theorem no_solution_inequality_system (m : ℝ) :
  (¬ ∃ x : ℝ, 2 * x - 1 < 3 ∧ x > m) ↔ m ≥ 2 :=
by
  sorry

end no_solution_inequality_system_l25_25204


namespace collinear_probability_l25_25667

theorem collinear_probability (O A B C D : Type) [decidable_eq Type] [fintype Type] (is_square : is_square_center O A B C D) :
  probability_collinear_points (O, A, B, C, D) = 1/5 := 
sorry

end collinear_probability_l25_25667


namespace distance_to_directrix_l25_25033

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25033


namespace relationship_between_a_b_c_l25_25549

noncomputable def a : ℝ := 0.7^0.9
noncomputable def b : ℝ := 0.8^0.9
noncomputable def c : ℝ := Real.log 2 / Real.log 3

theorem relationship_between_a_b_c (h_a : a = 0.7^0.9) (h_b : b = 0.8^0.9) (h_c : c = Real.log 2 / Real.log 3) : c < a ∧ a < b :=
by {
  rw [h_a, h_b, h_c],
  -- a := 0.7^0.9
  -- b := 0.8^0.9
  -- c := Real.log 2 / Real.log 3
  sorry
}

end relationship_between_a_b_c_l25_25549


namespace centroid_positions_l25_25885

-- Define the vertices of the rectangle
def vertices : List (ℝ × ℝ) := [(0, 0), (20, 0), (20, 10), (0, 10)]

-- Define the points on the longer sides
def longer_side_points : List (ℝ × ℝ) := 
  (List.range' 0 21).map (λ k => (k, 0)) ++ (List.range' 0 21).map (λ k => (k, 10))

-- Define the points on the shorter sides
def shorter_side_points : List (ℝ × ℝ) := 
  (List.range' 0 11).map (λ j => (0, j)) ++ (List.range' 0 11).map (λ j => (20, j))

-- Combine all the points
def perimeter_points : List (ℝ × ℝ) := (longer_side_points ++ shorter_side_points).eraseDuplicates

-- Define a function to calculate the centroid of a triangle given three points
def centroid (P Q R : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  let (x3, y3) := R
  ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)

-- Define the proof statement
theorem centroid_positions :
  ∃ positions : Set (ℝ × ℝ), positions.card = 1891 ∧
    ∀ P Q R : (ℝ × ℝ), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧
      P ∈ perimeter_points ∧ Q ∈ perimeter_points ∧ R ∈ perimeter_points ∧
      ¬(P.1 = Q.1 ∧ Q.1 = R.1) ∧ ¬(P.2 = Q.2 ∧ Q.2 = R.2) →
        centroid P Q R ∈ positions := sorry

end centroid_positions_l25_25885


namespace different_parity_of_b_c_l25_25989

theorem different_parity_of_b_c (a b c : ℕ) (ha_prime : a.prime) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h_eq : a^2 + b^2 = c^2) : 
  (b % 2 = 0 ∧ c % 2 = 1) ∨ (b % 2 = 1 ∧ c % 2 = 0) :=
sorry

end different_parity_of_b_c_l25_25989


namespace smallest_positive_period_and_symmetric_l25_25169

noncomputable def f1 (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)
noncomputable def f2 (x : ℝ) : ℝ := 2 * sin (2 * x - π / 6)
noncomputable def f3 (x : ℝ) : ℝ := 2 * sin (1 / 2 * x + π / 3)
noncomputable def f4 (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

theorem smallest_positive_period_and_symmetric:
  has_period f2 π ∧ is_symmetric_about f2 (π / 3) :=
sorry

end smallest_positive_period_and_symmetric_l25_25169


namespace distance_from_point_to_parabola_directrix_l25_25106

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25106


namespace infinite_natural_numbers_with_factored_polynomial_l25_25323

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l25_25323


namespace distance_from_A_to_directrix_l25_25021

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25021


namespace find_f_9_l25_25191

noncomputable def f : ℝ → ℝ := λ x, 1 + Real.log (x + 1)

theorem find_f_9 : f 9 = 2 :=
by
  sorry

end find_f_9_l25_25191


namespace point_D_number_l25_25697

theorem point_D_number (x : ℝ) :
    (5 + 8 - 10 + x = -5 - 8 + 10 - x) ↔ x = -3 :=
by
  sorry

end point_D_number_l25_25697


namespace area_of_gray_region_l25_25504

/-- Define the centers and radii of the circles as well as the general geometrical setting of the problem.-/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def circle_C : Circle := ⟨(3, 3), 3⟩
def circle_D : Circle := ⟨(9, 3), 3⟩

/-- Define the total area of the rectangle formed by the circles and the x-axis. -/
def rectangle_area : ℝ := 6 * 3

/-- Define the combined area of the sectors cut out by the circles. -/
def sector_area : ℝ := 2 * (1 / 4 * (3 ^ 2) * Real.pi)

/-- Compute the area of the gray region. -/
def gray_area : ℝ := rectangle_area - sector_area

/-- The theorem to prove that the area of the gray region is equal to the given expression. -/
theorem area_of_gray_region : gray_area = 18 - 9 * Real.pi / 2 := by
  sorry

end area_of_gray_region_l25_25504


namespace child_running_speed_l25_25449

theorem child_running_speed
  (c s t : ℝ)
  (h1 : (74 - s) * 3 = 165)
  (h2 : (74 + s) * t = 372) :
  c = 74 :=
by sorry

end child_running_speed_l25_25449


namespace exists_infinitely_many_natural_numbers_factors_l25_25352

theorem exists_infinitely_many_natural_numbers_factors (k : ℤ) :
  ∃ (P Q : ℤ[X]), 
    (P.degree = 4 ∧ Q.degree = 4) ∧
    (X^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = P * Q) :=
by
  sorry

end exists_infinitely_many_natural_numbers_factors_l25_25352


namespace raisin_weight_l25_25616

theorem raisin_weight :
  ∀ (Wg : ℝ) (Wr : ℝ),
    Wg = 116.2 ∧ 
    (Wg * 0.1) = (Wr * 0.83) ∧ 
    (Wg * 0.9 + Wg * 0.1 = Wg) ∧
    (Wr * 0.17 + Wr * 0.83 = Wr)
      → Wr ≈ 14 :=
by sorry

end raisin_weight_l25_25616


namespace conic_sections_of_equation_l25_25891

theorem conic_sections_of_equation :
  ∀ x y : ℝ, y^4 - 9*x^6 = 3*y^2 - 1 →
  (∃ y, y^2 - 3*x^3 = 4 ∨ y^2 + 3*x^3 = 0) :=
by 
  sorry

end conic_sections_of_equation_l25_25891


namespace problem_correct_statements_l25_25428

theorem problem_correct_statements:
  (∀ (a b : ℝ), ab > 0 → ((∃ c, (line : ℝ → ℝ := fun x ↦ ax + by + c) (-a / b) < 0)) → 
  (point_slope (2, -1) (-√3) (line := y + 1 = -√3 * (x - 2))) → 
  ¬(line_eq (line := y = -2x ± 3)) → 
  ¬(line_intercept_eq (line := x + y - 2 = 0) (1,1)) →
  true) := sorry


end problem_correct_statements_l25_25428


namespace gcd_315_2016_l25_25744

def a : ℕ := 315
def b : ℕ := 2016

theorem gcd_315_2016 : Nat.gcd a b = 63 := 
by 
  sorry

end gcd_315_2016_l25_25744


namespace vasya_polynomial_min_distinct_l25_25817

-- Define the polynomial P(x) of degree 7 with non-zero coefficients
def polynomial_degree7 := (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ)

-- Define the derivative of P
def derivative (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) (n : ℕ) : ℝ :=
match n with
| 0 => a7
| 1 => 7 * a7
| 2 => 42 * a7
| 3 => 210 * a7
| 4 => 840 * a7
| 5 => 2520 * a7
| 6 => 5040 * a7
| 7 => 7! * a7
| _ => 0
end

-- Statement of the problem
theorem vasya_polynomial_min_distinct (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) (h : ∀ i, i ∈ [a0, a1, a2, a3, a4, a5, a6, a7] → i ≠ 0) :
    ∃ n, n = 7 ∧ (∀ k, k ∈ [0,1,2,3,4,5,6] → deriv (derivative a0 a1 a2 a3 a4 a5 a6 a7 k) (7 - k) ≠ 0) := sorry

end vasya_polynomial_min_distinct_l25_25817


namespace expression_equals_100_l25_25438

-- Define the terms in the numerator and their squares
def num1 := 0.02
def num2 := 0.52
def num3 := 0.035

def num1_sq := num1^2
def num2_sq := num2^2
def num3_sq := num3^2

-- Define the terms in the denominator and their squares
def denom1 := 0.002
def denom2 := 0.052
def denom3 := 0.0035

def denom1_sq := denom1^2
def denom2_sq := denom2^2
def denom3_sq := denom3^2

-- Define the sums of the squares
def sum_numerator := num1_sq + num2_sq + num3_sq
def sum_denominator := denom1_sq + denom2_sq + denom3_sq

-- Define the final expression
def expression := sum_numerator / sum_denominator

-- Prove the expression equals the correct answer
theorem expression_equals_100 : expression = 100 := by sorry

end expression_equals_100_l25_25438


namespace distance_from_A_to_directrix_on_parabola_l25_25042

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25042


namespace distance_from_A_to_directrix_l25_25003

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25003


namespace mike_miles_ride_l25_25437

theorem mike_miles_ride :
  ∃ (m : ℝ), (2.5 + 0.25 * m = 2.5 + 5.0 + 0.25 * 22) ∧ m = 42 :=
begin
  use 42,
  split,
  { sorry }, -- This part will simplify to show LHS = RHS
  { refl }
end

end mike_miles_ride_l25_25437


namespace solution_of_equation_l25_25388

theorem solution_of_equation :
  ∀ x : ℝ, 9^x + 3^x - 2 = 0 → x = 0 :=
by
  intro x
  intro h
  sorry

end solution_of_equation_l25_25388


namespace p_q_2r_value_l25_25839

variable (p q r : ℝ) (f : ℝ → ℝ)

-- The conditions as definitions
def f_def : f = fun x => p * x^2 + q * x + r := by sorry
def f_at_0 : f 0 = 9 := by sorry
def f_at_1 : f 1 = 6 := by sorry

-- The theorem statement
theorem p_q_2r_value : p + q + 2 * r = 15 :=
by
  -- utilizing the given definitions 
  have h₁ : r = 9 := by sorry
  have h₂ : p + q + r = 6 := by sorry
  -- substitute into p + q + 2r
  sorry

end p_q_2r_value_l25_25839


namespace tan_A_tan_B_l25_25647

theorem tan_A_tan_B (A B C : Type) [triangle A B C] (F : A)
  (H : orthocenter A B C)
  (hCF : altitude A F C)
  (hHF : segment H F = 10)
  (hHC : segment H C = 20) :
  tan A * tan B = 3 := 
by 
  sorry

end tan_A_tan_B_l25_25647


namespace least_number_to_add_divisible_l25_25792

theorem least_number_to_add_divisible (a : ℕ) (b : ℕ) (r : ℕ) (m : ℕ) :
  a = 1056 → b = 23 → r = 21 → m = 2 → (a + m) % b = 0 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end least_number_to_add_divisible_l25_25792


namespace distance_from_A_to_directrix_l25_25146

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25146


namespace two_abc_square_l25_25244

variable {R : Type*} [Ring R] [Fintype R]

-- Given condition: For any a, b ∈ R, ∃ c ∈ R such that a^2 + b^2 = c^2.
axiom ring_property (a b : R) : ∃ c : R, a^2 + b^2 = c^2

-- We need to prove: For any a, b, c ∈ R, ∃ d ∈ R such that 2abc = d^2.
theorem two_abc_square (a b c : R) : ∃ d : R, 2 * (a * b * c) = d^2 :=
by
  sorry

end two_abc_square_l25_25244


namespace total_number_of_shells_l25_25272

variable (David Mia Ava Alice : ℕ)
variable (hd : David = 15)
variable (hm : Mia = 4 * David)
variable (ha : Ava = Mia + 20)
variable (hAlice : Alice = Ava / 2)

theorem total_number_of_shells :
  David + Mia + Ava + Alice = 195 :=
by
  sorry

end total_number_of_shells_l25_25272


namespace tan_five_pi_over_four_l25_25528

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_five_pi_over_four_l25_25528


namespace optimal_washing_l25_25444

theorem optimal_washing (total_capacity : ℝ) (clothes_weight : ℝ) 
  (scoops : ℕ) (scoop_weight : ℝ) (optimal_ratio : ℝ) : 
  ∃ (additional_detergent : ℝ) (additional_water : ℝ), 
    additional_detergent = 0.02 ∧ additional_water = 14.94 :=
by
  let total_detergent_already_in_tub := scoops * scoop_weight
  let total_laundry_water_capacity := total_capacity - clothes_weight
  let optimal_detergent_needed_for_15_kg := total_laundry_water_capacity * optimal_ratio
  let additional_detergent_needed := optimal_detergent_needed_for_15_kg - total_detergent_already_in_tub
  let actual_water_needed := total_laundry_water_capacity - additional_detergent_needed
  have h1 : additional_detergent_needed = 0.02 := sorry
  have h2 : actual_water_needed = 14.94 := sorry
  existsi 0.02
  existsi 14.94
  exact ⟨h1, h2⟩

end optimal_washing_l25_25444


namespace distance_from_point_A_to_directrix_C_l25_25117

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25117


namespace perpendicular_lines_slope_eq_l25_25205

theorem perpendicular_lines_slope_eq (a : ℝ) (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, line1 x y ↔ ax + 2*y - 3 = 0)
  (h2 : ∀ x y, line2 x y ↔ 2*x - y = 0)
  (h_perpendicular : ∀ m1 m2, m1 * m2 = -1 → slope line1 = m1 → slope line2 = m2)
  : a = 1 := 
sorry

end perpendicular_lines_slope_eq_l25_25205


namespace circle_intersection_probability_l25_25411

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

noncomputable def intersects (r1 r2 d : ℝ) : Prop :=
  d ≤ r1 + r2

theorem circle_intersection_probability :
  let A_radius := 2
  let B_radius := 3
  let B_center := (3 : ℝ, 2 : ℝ)
  let A_center_x := uniform_continuous_on (0 : ℝ) (5 : ℝ)
  let distance := dist A_center_x 0 B_center.1 B_center.2
  ∀ A_center_x, intersects A_radius B_radius distance → probability_that_circles_intersect = 1 :=
by
  sorry

end circle_intersection_probability_l25_25411


namespace positive_difference_between_diagonal_sums_l25_25814

-- Condition definitions
def original_calendar : List (List Nat) :=
  [[1, 2, 3, 4, 5],
   [6, 7, 8, 9, 10],
   [11, 12, 13, 14, 15],
   [16, 17, 18, 19, 20],
   [21, 22, 23, 24, 25]]

def modified_calendar : List (List Nat) :=
  [[1, 2, 3, 4, 5],
   [6, 7, 8, 9, 10],
   [15, 14, 13, 12, 11],
   [16, 17, 18, 19, 20],
   [25, 24, 23, 22, 21]]

def main_diagonal (matrix : List (List Nat)) : List Nat :=
  [matrix[0][0], matrix[1][1], matrix[2][2], matrix[3][3], matrix[4][4]]

def secondary_diagonal (matrix : List (List Nat)) : List Nat :=
  [matrix[0][4], matrix[1][3], matrix[2][2], matrix[3][1], matrix[4][0]]

def sum_list (lst : List Nat) : Nat :=
  lst.foldl Nat.add 0

-- Lean 4 statement
theorem positive_difference_between_diagonal_sums :
  let original_main_diagonal := main_diagonal original_calendar
  let original_secondary_diagonal := secondary_diagonal original_calendar
  let modified_main_diagonal := main_diagonal modified_calendar
  let modified_secondary_diagonal := secondary_diagonal modified_calendar
  let original_difference := Nat.abs (sum_list original_main_diagonal - sum_list original_secondary_diagonal)
  let modified_difference := Nat.abs (sum_list modified_main_diagonal - sum_list modified_secondary_diagonal)
  modified_difference = 8 := by
  sorry

end positive_difference_between_diagonal_sums_l25_25814


namespace poly_factorable_l25_25305

-- Define the specific form of N.
noncomputable def N (k : ℤ) : ℕ := 4 * k^4 - 8 * k^2 + 2

-- Define the polynomial.
noncomputable def poly (N : ℕ) : ℤ[X] := X^8 + N * X^4 + 1

-- Define the fourth-degree polynomials used in the factorization.
noncomputable def poly1 (k : ℤ) : ℤ[X] :=
  X^4 - 2 * k * X^3 + 2 * k^2 * X^2 - 2 * k * X + 1

noncomputable def poly2 (k : ℤ) : ℤ[X] := 
  X^4 + 2 * k * X^3 + 2 * k^2 * X^2 + 2 * k * X + 1

-- The proof statement: prove that the polynomial can be factored into the specified fourth-degree polynomials.
theorem poly_factorable : ∀ k : ℤ, ∃ p1 p2 : ℤ[X], poly (N k) = p1 * p2 :=
by
  intro k
  use (poly1 k)
  use (poly2 k)
  sorry

end poly_factorable_l25_25305


namespace find_a7_l25_25641

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, ∃ r, a (n + m) = (a n) * (r ^ m)

def sequence_properties (a : ℕ → ℝ) : Prop :=
geometric_sequence a ∧ a 3 = 3 ∧ a 11 = 27

theorem find_a7 (a : ℕ → ℝ) (h : sequence_properties a) : a 7 = 9 := 
sorry

end find_a7_l25_25641


namespace largest_even_digit_multiple_of_9_below_1000_l25_25780

theorem largest_even_digit_multiple_of_9_below_1000 :
  ∃ n : ℕ, n = 882 ∧ n < 1000 ∧ (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d % 2 = 0) ∧ n % 9 = 0 :=
by
  existsi 882
  split
  { rfl }
  split
  { norm_num }
  split
  { intro d
    intro h
    fin_cases d with
    | h1 => norm_num
    | h2 => norm_num
    | h3 => norm_num }
  { norm_num }

end largest_even_digit_multiple_of_9_below_1000_l25_25780


namespace hyperbola_asymptotes_l25_25376

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 16 * x^2 - 9 * y^2 = -144 → (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intros x y h1
  sorry

end hyperbola_asymptotes_l25_25376


namespace find_lambda_l25_25179

-- Definitions for conditions
def vector_a : ℝ × ℝ := (1, -3)
def vector_b : ℝ × ℝ := (4, -2)

-- Definition for perpendicular condition
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Theorem statement
theorem find_lambda (λ : ℝ) (h : is_perpendicular (λ • vector_a + vector_b) vector_a) : λ = -1 :=
sorry

end find_lambda_l25_25179


namespace painting_distance_from_end_l25_25465

/-- A wall with a given width and a painting centered in the middle of it -/
def wall_and_painting (wall_width painting_width : ℕ) (centered : Prop) : Prop := 
  centered ∧ painting_width > 0 ∧ wall_width > painting_width

theorem painting_distance_from_end
  (wall_width : ℕ)
  (painting_width : ℕ)
  (centered : Prop)
  (hw : wall_and_painting wall_width painting_width centered)
  (h_centered : centered) : ∃ x, 2 * x + painting_width = wall_width ∧ x = 11 :=
  begin
    sorry
  end

end painting_distance_from_end_l25_25465


namespace exists_infinitely_many_N_l25_25295

theorem exists_infinitely_many_N (k : ℤ) : ∃ N : ℕ, ∃ A B : polynomial ℤ, 
  N = 4 * k^4 - 8 * k^2 + 2 ∧ polynomial.nat_degree A = 4 ∧ polynomial.nat_degree B = 4 ∧
  (polynomial.X^8 + polynomial.C (N:ℤ) * polynomial.X^4 + 1) = A * B :=
sorry

end exists_infinitely_many_N_l25_25295


namespace distance_from_A_to_directrix_l25_25073

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25073


namespace inequality_solution_l25_25717

theorem inequality_solution (x : ℝ) :
  2 * (2 * x - 1) > 3 * x - 1 → x > 1 :=
by
  sorry

end inequality_solution_l25_25717


namespace find_still_water_speed_l25_25836

noncomputable def still_water_speed (V_d : ℝ) (current_speed : ℝ) : ℝ :=
  V_d - current_speed

theorem find_still_water_speed :
  let V_d := 0.1 / 0.0049996 -- kmph
  let current_speed := 4 -- kmph
  still_water_speed V_d current_speed = 16.0008 :=
by
  let V_d := 0.1 / 0.0049996 -- kmph
  let current_speed := 4 -- kmph
  rw [still_water_speed, V_d, current_speed]
  sorry

end find_still_water_speed_l25_25836


namespace cost_of_adult_ticket_l25_25763

theorem cost_of_adult_ticket
    (child_ticket_cost : ℝ)
    (total_tickets : ℕ)
    (total_receipts : ℝ)
    (adult_tickets_sold : ℕ)
    (A : ℝ)
    (child_tickets_sold : ℕ := total_tickets - adult_tickets_sold)
    (total_revenue_adult : ℝ := adult_tickets_sold * A)
    (total_revenue_child : ℝ := child_tickets_sold * child_ticket_cost) :
    child_ticket_cost = 4 →
    total_tickets = 130 →
    total_receipts = 840 →
    adult_tickets_sold = 90 →
    total_revenue_adult + total_revenue_child = total_receipts →
    A = 7.56 :=
by
  intros
  sorry

end cost_of_adult_ticket_l25_25763


namespace smallest_n_mod_equiv_l25_25420

theorem smallest_n_mod_equiv (n : ℕ) (h : 5 * n ≡ 4960 [MOD 31]) : n = 31 := by 
  sorry

end smallest_n_mod_equiv_l25_25420


namespace square_root_condition_l25_25373

-- Define the condition under which the square root of an expression is defined
def is_square_root_defined (x : ℝ) : Prop := (x + 3) ≥ 0

-- Prove that the condition for the square root of x + 3 to be defined is x ≥ -3
theorem square_root_condition (x : ℝ) : is_square_root_defined x ↔ x ≥ -3 := 
sorry

end square_root_condition_l25_25373


namespace func_domain_correct_l25_25737

noncomputable def funcDomain : Set ℝ := { x : ℝ | x > 3 }

theorem func_domain_correct :
  ∀ x : ℝ, (x + 1 > 0) ∧ (3^x - 27 > 0) ↔ x ∈ funcDomain :=
by
  intro x
  sorry

end func_domain_correct_l25_25737


namespace largest_integer_n_neg_quad_expr_l25_25531

theorem largest_integer_n_neg_quad_expr :
  ∃ n : ℤ, n = 6 ∧ ∀ m : ℤ, ((n^2 - 11 * n + 28 < 0) → (m < 7 ∧ m > 4) → m ≤ n) :=
by
  sorry

end largest_integer_n_neg_quad_expr_l25_25531


namespace problem_l25_25951

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ) (a b c : ℝ)
  (h1 : ∀ x, f (x + 1) = f (-x + 1))
  (h2 : ∀ x, 1 < x → f x ≤ f (x - 1))
  (ha : a = f 2)
  (hb : b = f (Real.log 2 / Real.log 3))
  (hc : c = f (1 / 2))

theorem problem (h : a = f 2 ∧ b = f (Real.log 2 / Real.log 3) ∧ c = f (1 / 2)) : 
  a < c ∧ c < b := sorry

end problem_l25_25951


namespace max_M_l25_25537

-- Definitions
def J (k : ℕ) (h : k > 0) : ℕ := 10 * (10^k) + 80

def M (k : ℕ) (h : k > 0) : ℕ :=
  (J k h).factorization 2

theorem max_M : ∃ k > 0, M k (by linarith [k_pos]) = 4 := sorry

end max_M_l25_25537


namespace derivative_at_one_l25_25166

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_at_one : deriv f 1 = 2 + Real.exp 1 := by
  sorry

end derivative_at_one_l25_25166


namespace measure_angle_BPC_l25_25638

noncomputable def lengthAB : ℝ := 6
noncomputable def lengthBC : ℝ := 6
noncomputable def alpha : ℝ := 75 

-- Define the points and properties
variables (A B C D E P Q : Type) 
[HasPoint A] [HasPoint B] [HasPoint C] [HasPoint D] [HasPoint E] [HasPoint P] [HasPoint Q]

variables (AB AE : ℝ)
variables (angle_ABC angle_AEB angle_PBC angle_BPC : ℝ)

-- Conditions
axiom square_ABCD : (lengthAB = lengthBC) ∧ angle_ABC = 90
axiom triangle_ABE : (AB = AE) ∧ (angle_AEB = alpha)
axiom BE_ints_AC_at_P : True
axiom Q_on_BC_PQ_perp_BC : True

-- Define the target angle to be proven
def target_angle : ℝ := 180 - (angle_ABC - angle_AEB) - (angle_ABC / 2)

-- The theorem to prove
theorem measure_angle_BPC : angle_BPC = 120
:= sorry

end measure_angle_BPC_l25_25638


namespace exists_infinite_N_l25_25344

theorem exists_infinite_N (k : ℤ) : ∃ᶠ N : ℕ in at_top, 
  ∃ (A B : Polynomial ℤ), 
    (Polynomial.natDegree A = 4 ∧ Polynomial.natDegree B = 4) ∧
    (Polynomial.mul A B = Polynomial.X^8 + Polynomial.C N * Polynomial.X^4 + 1) :=
begin
  -- Proof goes here
  sorry
end

end exists_infinite_N_l25_25344


namespace part_I_part_II_l25_25168

-- Define the function f for Part I
def f_I (x m a : ℝ) : ℝ := (exp (x + 1) - log (x + 2) + a * x * (x + 2) - 1)

-- Define the function f for Part II
def f_II (x m : ℝ) : ℝ := (exp (x + m) - log (x + 2) - m)

-- Define proof problem for Part I: Minimum value of f_I when f_I(-1) is an extremum
theorem part_I (a : ℝ) (ha : a > 0) :
  ∀ m, ∃ x, f_I x m a = -a :=
sorry

-- Define proof problem for Part II: f_II(x, m) >= 0 for all x in [-1, 0]
theorem part_II (m : ℝ) :
  ∀ x, -1 ≤ x ∧ x ≤ 0 → f_II x m ≥ 0 :=
sorry

end part_I_part_II_l25_25168


namespace probability_alia_bella_correct_l25_25850

-- Definitions capturing the conditions
def num_cards : ℕ := 12
def num_ways_select_2 : ℕ := nat.choose num_cards 2
def num_ways_select_alia_bella : ℕ := 4 * 3
def probability_alia_bella : ℚ := num_ways_select_alia_bella / num_ways_select_2

-- Lean theorem statement that captures the problem-to-answer equivalence
theorem probability_alia_bella_correct : probability_alia_bella = 2 / 11 := by sorry

end probability_alia_bella_correct_l25_25850


namespace exists_infinitely_many_N_l25_25293

theorem exists_infinitely_many_N (k : ℤ) : ∃ N : ℕ, ∃ A B : polynomial ℤ, 
  N = 4 * k^4 - 8 * k^2 + 2 ∧ polynomial.nat_degree A = 4 ∧ polynomial.nat_degree B = 4 ∧
  (polynomial.X^8 + polynomial.C (N:ℤ) * polynomial.X^4 + 1) = A * B :=
sorry

end exists_infinitely_many_N_l25_25293


namespace complex_square_identity_l25_25501

theorem complex_square_identity (i : ℂ) (h_i_squared : i^2 = -1) :
  i * (1 + i)^2 = -2 :=
by
  sorry

end complex_square_identity_l25_25501


namespace distance_from_A_to_directrix_on_parabola_l25_25043

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25043


namespace midpoint_sum_coords_l25_25704

theorem midpoint_sum_coords (N C D : ℝ × ℝ) (sum_D: ℝ) 
  (hN : N = (3, 5))
  (hC : C = (1, 10)) 
  (hMidpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
  sum_D = (D.1 + D.2) :=
by
  -- Definitions and conditions
  have hNdef : N = (3, 5) := hN,
  have hCdef : C = (1, 10) := hC,
  have hMidpointFormula : (3, 5) = ((1 + D.1) / 2, (10 + D.2) / 2) := by
    rw [hNdef, hCdef] at hMidpoint
    exact hMidpoint,
  sorry

end midpoint_sum_coords_l25_25704


namespace max_cos_sum_l25_25857

theorem max_cos_sum (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 0 < A) (h3 : A < Real.pi)
  (h4 : 0 < B) (h5 : B < Real.pi) (h6 : 0 < C) (h7 : C < Real.pi) :
  ∃ max_val : ℝ, max_val = (3 / 4) ∧ (∀ (A B C : ℝ), A + B + C = Real.pi → (0 < A) → (A < Real.pi) →
  (0 < B) → (B < Real.pi) → (0 < C) → (C < Real.pi) → (cos A + cos B * cos C ≤ max_val)) :=
sorry

end max_cos_sum_l25_25857


namespace floor_abs_sum_eq_six_l25_25723

-- Define the condition
def condition (x : Fin 10 → ℝ) : Prop :=
  ∀ i : Fin 10, x i + (i + 1) = (∑ j, x j) + 11

-- Define the main statement we need to prove
theorem floor_abs_sum_eq_six (x : Fin 10 → ℝ) (h : condition x) :
  Int.floor (abs (∑ i, x i)) = 6 :=
  sorry

end floor_abs_sum_eq_six_l25_25723


namespace second_discount_percentage_l25_25756

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount_percentage : ℝ)
  (final_sale_price : ℝ)
  (h1 : original_price = 390)
  (h2 : first_discount_percentage = 0.15)
  (h3 : final_sale_price = 248.625) :
  let first_discount_amount := first_discount_percentage * original_price in
  let price_after_first_discount := original_price - first_discount_amount in
  let second_discount_amount := price_after_first_discount - final_sale_price in
  let second_discount_percentage := (second_discount_amount / price_after_first_discount) * 100 in
  second_discount_percentage = 25 := by
  sorry

end second_discount_percentage_l25_25756


namespace tan_alpha_plus_beta_eq_one_l25_25935

theorem tan_alpha_plus_beta_eq_one
  (α β : ℝ)
  (l : ℝ → ℝ → Prop)
  (h1 : ∀ x y : ℝ, l x y ↔ x * tan α - y - 3 * tan β = 0)
  (normal_vector : (ℝ × ℝ))
  (h_nv : normal_vector = (2, -1))
  (point_on_line : (ℝ × ℝ))
  (h_point : point_on_line = (0, 1)) :
  tan (α + β) = 1 := 
by
  sorry

end tan_alpha_plus_beta_eq_one_l25_25935


namespace distinct_units_digits_of_cubes_l25_25183

theorem distinct_units_digits_of_cubes : 
  (set.range (λ n : ℕ, (n^3 % 10))) = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by {
  sorry
}

end distinct_units_digits_of_cubes_l25_25183


namespace final_combined_price_correct_l25_25751

theorem final_combined_price_correct :
  let i_p := 1000
  let d_1 := 0.10
  let d_2 := 0.20
  let t_1 := 0.08
  let t_2 := 0.06
  let s_p := 30
  let c_p := 50
  let t_a := 0.05
  let price_after_first_month := i_p * (1 - d_1) * (1 + t_1)
  let price_after_second_month := price_after_first_month * (1 - d_2) * (1 + t_2)
  let screen_protector_final := s_p * (1 + t_a)
  let case_final := c_p * (1 + t_a)
  price_after_second_month + screen_protector_final + case_final = 908.256 := by
  sorry  -- Proof not required

end final_combined_price_correct_l25_25751


namespace find_line_equation_l25_25196

noncomputable def line_equation (l : ℝ → ℝ → Prop) : Prop :=
  (l 2 3) ∧ (∃ f : ℝ → ℝ, (∀ x y, l x y ↔ f x = y) ∧ (∃ a : ℝ, a ≠ 1 ∧ isLinearMap ℝ f ∧ 
  (∀ x, l x 0 → l 0 x)) ∧ (∃ a : ℝ, l 0 a = true ∧ l a 0 = true))

theorem find_line_equation (l : ℝ → ℝ → Prop) :
  line_equation l →
  (∀ x y : ℝ, l x y ↔ (y = (3 / 2) * x ∨ x + y - 5 = 0)) :=
by
  sorry

end find_line_equation_l25_25196


namespace exists_infinitely_many_natural_numbers_factors_l25_25347

theorem exists_infinitely_many_natural_numbers_factors (k : ℤ) :
  ∃ (P Q : ℤ[X]), 
    (P.degree = 4 ∧ Q.degree = 4) ∧
    (X^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = P * Q) :=
by
  sorry

end exists_infinitely_many_natural_numbers_factors_l25_25347


namespace zero_diff_leq_half_l25_25617

noncomputable def g (x : ℝ) : ℝ := log x + 2*x - 8
noncomputable def f (x : ℝ) : ℝ := log (x - 5/2)

theorem zero_diff_leq_half :
  (∃ x, g x = 0 ∧ 3 < x ∧ x < 4) →
  (∃ y, f y = 0) →
  (∀ x y, g x = 0 → f y = 0 → abs (x - y) ≤ 0.5) := by
  sorry

end zero_diff_leq_half_l25_25617


namespace part_one_part_two_l25_25945

/-- (1) For \(x > 0\), \(e^{x-1} \geqslant \ln(x) + 1\) -/
theorem part_one (x : ℝ) (h : x > 0) : Real.exp(x - 1) ≥ Real.log x + 1 :=
sorry

/-- (2) There can be at most two distinct lines tangent to both 
\( y = e^{x-1} \) and \( y = \ln(x) - 1 \) simultaneously. -/
theorem part_two : ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ 
  (∀ t : ℝ, t1 = t ∨ t2 = t) ∧
  (∀ t : ℝ, tangent_at f t ∧ tangent_at g t) :=
sorry

end part_one_part_two_l25_25945


namespace sum_of_first_four_terms_l25_25629

noncomputable def sum_first_n_terms (a q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem sum_of_first_four_terms :
  ∀ (a q : ℝ), a * (1 + q) = 7 → a * (q^6 - 1) / (q - 1) = 91 →
  a * (1 + q + q^2 + q^3) = 28 :=
by
  intros a q h₁ h₂
  -- Proof omitted
  sorry

end sum_of_first_four_terms_l25_25629


namespace smaller_of_x_and_y_l25_25412

variable {a b x y d : ℝ}
variable (h1 : 0 < a) (h2 : a < b) (h3 : x * y = d) (h4 : x / y = a / b)

theorem smaller_of_x_and_y : x = sqrt (ad / b) :=
sorry

end smaller_of_x_and_y_l25_25412


namespace negation_of_p_l25_25567

   -- Define the proposition p as an existential quantification
   def p : Prop := ∃ x₀ : ℝ, x₀^2 + 2 * x₀ + 3 > 0

   -- State the theorem that negation of p is a universal quantification
   theorem negation_of_p : ¬ p ↔ ∀ x : ℝ, x^2 + 2*x + 3 ≤ 0 :=
   by sorry
   
end negation_of_p_l25_25567


namespace surface_area_of_circumscribed_sphere_l25_25471

-- Define the regular triangular prism with given edge lengths and the existence of a circumscribed sphere.
structure RegularTriangularPrism :=
(base_edge_length : ℝ)
(lateral_edge_length : ℝ)
(vertices_on_sphere : Prop)

-- The given regular triangular prism.
def given_prism : RegularTriangularPrism :=
{ base_edge_length := sqrt 3,
  lateral_edge_length := 2,
  vertices_on_sphere := true }

-- The theorem stating the surface area of the circumscribed sphere.
theorem surface_area_of_circumscribed_sphere
  (P : RegularTriangularPrism)
  (h : P = given_prism) :
  ∃ S, S = 8 * Real.pi :=
by {
  sorry
}

end surface_area_of_circumscribed_sphere_l25_25471


namespace exists_infinite_N_l25_25342

theorem exists_infinite_N (k : ℤ) : ∃ᶠ N : ℕ in at_top, 
  ∃ (A B : Polynomial ℤ), 
    (Polynomial.natDegree A = 4 ∧ Polynomial.natDegree B = 4) ∧
    (Polynomial.mul A B = Polynomial.X^8 + Polynomial.C N * Polynomial.X^4 + 1) :=
begin
  -- Proof goes here
  sorry
end

end exists_infinite_N_l25_25342


namespace find_m_l25_25961

theorem find_m (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((1/3 : ℝ) * x1^3 - 3 * x1 + m = 0) ∧ ((1/3 : ℝ) * x2^3 - 3 * x2 + m = 0)) ↔ (m = -2 * Real.sqrt 3 ∨ m = 2 * Real.sqrt 3) :=
sorry

end find_m_l25_25961


namespace price_per_large_bottle_l25_25237

/-- 
John purchased 1300 large bottles at price P per bottle.
He purchased 750 small bottles at $1.38 per bottle.
The approximate average price paid per bottle was $1.7034.
Prove that the price per large bottle P is approximately 1.89.
-/
theorem price_per_large_bottle 
  (P : ℝ)
  (H1: 1300 * P + 750 * 1.38 = (1300 + 750) * 1.7034) :
  P ≈ 1.89 :=
sorry

end price_per_large_bottle_l25_25237


namespace solve_square_l25_25222

theorem solve_square:
  ∃ (square: ℚ), 
    ((13/5) - ((17/2) - square) / (7/2)) / (1 / ((61/20) + (89/20))) = 2 → 
    square = 1/3 :=
  sorry

end solve_square_l25_25222


namespace range_of_a_l25_25932

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x - 2 * a

def solution_set (a : ℝ) : set ℝ :=
  {x | f a x > 0}

def B : set ℝ :=
  {x | 1 < x ∧ x < 3}

theorem range_of_a (a : ℝ) :
  (solution_set a ∩ B).nonempty ↔ 
  (a ∈ set.Ioo (-∞) (-2) ∪ set.Ioo (6 / 7) ∞) := sorry

end range_of_a_l25_25932


namespace elly_candies_l25_25538

theorem elly_candies (a b c : ℝ) (h1 : a * b * c = 216) : 
  24 * 216 = 5184 :=
by
  sorry

end elly_candies_l25_25538


namespace lines_through_vertex_dividing_perimeter_intersect_at_single_point_l25_25762

theorem lines_through_vertex_dividing_perimeter_intersect_at_single_point 
  (A B C : Point) 
  (a b c : ℝ) -- sides of the triangle
  (P : ℝ)     -- perimeter of the triangle
  (hP : P = a + b + c) -- defining the perimeter
  (p : ℝ)     -- half of the perimeter
  (hp : p = P / 2)
  (AP BP CP : ℝ) -- segments summing to half perimeter
  (hAP : AP = p)
  (hBP : BP = p)
  (hCP : CP = p) :
  ∃ (O : Point), is_concurrent A B C O :=
sorry

end lines_through_vertex_dividing_perimeter_intersect_at_single_point_l25_25762


namespace largest_even_digit_multiple_of_9_under_1000_l25_25776

noncomputable def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

noncomputable def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else n.digits 10

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (digits n).sum

noncomputable def all_even_digits (n : ℕ) : Prop :=
  ∀ d ∈ digits n, is_even_digit d

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ n, n < 1000 ∧ all_even_digits n ∧ sum_of_digits n % 9 = 0 ∧ n = 360 :=
begin
  use 360,
  split,
  { exact nat.lt_succ_self 359 },
  split,
  { intros d hd,
    unfold digits at hd,
    rw list.mem_iff_exists_get at hd,
    rcases hd with ⟨k, hk⟩,
    repeat { rw list.get? },
    simp only [digits] at hk,
    have : k < 3 := (nat.lt_of_succ_lt_succ (list.length_le_of_lt_some hk)).trans_le (by norm_num),
    interval_cases k,
    { rw [hk, nat.digits, nat.digits_aux'],
      norm_num [is_even_digit] },
    { rw [hk, nat.digits, nat.digits_aux', nat.div_eq, nat.mod_eq_of_lt, nat.add_eq_zero_eq_zero_and_eq_zero, nat.zero_eq, eq_self_iff_true, true_and],
      norm_num [is_even_digit] },
    { rw [hk, nat.digits, nat.digits_aux', nat.div_eq, nat.mod_eq_of_lt, nat.add_eq_zero_eq_zero_and_eq_zero, nat.zero_eq, eq_self_iff_true, true_and],
      norm_num [is_even_digit] } },
  { simp only [sum_of_digits, digits],
    exact nat.digits_sum_eq 360 10,
    exact dec_trivial },
  { refl }
end

end largest_even_digit_multiple_of_9_under_1000_l25_25776


namespace arrangement_of_volunteers_l25_25892

-- Define the problem with necessary conditions
def num_arrangements (people : ℕ) (exits : ℕ) (people_at_each_exit : ℕ → ℕ) : ℕ :=
  if exits = 4 ∧ ∀ x, people_at_each_exit x > 0 ∧ people_at_each_exit 1 = 2 then
    choose 5 2 * (factorial 3)
  else
    0

-- Lean 4 statement for the proof problem
theorem arrangement_of_volunteers : num_arrangements 5 4 (λ x, if x = 1 then 2 else 1) = 240 :=
by
  sorry

end arrangement_of_volunteers_l25_25892


namespace jared_annual_earnings_l25_25917

-- Defining conditions as constants
def diploma_pay : ℕ := 4000
def degree_multiplier : ℕ := 3
def months_in_year : ℕ := 12

-- Goal: Prove that Jared's annual earnings are $144,000
theorem jared_annual_earnings : diploma_pay * degree_multiplier * months_in_year = 144000 := by
  sorry

end jared_annual_earnings_l25_25917


namespace exists_fixed_point_l25_25741

open Function

theorem exists_fixed_point (f : ℝ → ℝ) (h_cont : ContinuousOn f (Icc 0 1)) (h_range : ∀ x ∈ Icc 0 1, f x ∈ Icc 0 1) :
  ∃ x ∈ Icc 0 1, f x = x :=
sorry

end exists_fixed_point_l25_25741


namespace sum_even_integers_eq_930_l25_25396

theorem sum_even_integers_eq_930 :
  let sum_first_30_even := 2 * (30 * (30 + 1) / 2)
  let sum_consecutive_even (n : ℤ) := (n - 8) + (n - 6) + (n - 4) + (n - 2) + n
  ∀ n : ℤ, sum_first_30_even = 930 → sum_consecutive_even n = 930 → n = 190 :=
by
  intros sum_first_30_even sum_consecutive_even n h1 h2
  sorry

end sum_even_integers_eq_930_l25_25396


namespace board_number_is_3343_l25_25439

theorem board_number_is_3343 :
  ∃ (x : ℕ), ∀ (n : ℕ) (seq : List ℕ), 
    (n ≥ 2) ∧ 
    (∀ i, (i < n) → seq.get? i = some (8 * i + 3343 / 8 ^ (n - i - 1) % 8) ∧ 
    seq.get? 0 ≠ some (2 * 8 ^ (n - 1)) ∧ 
    (∃ y z : ℕ, prime y ∧ prime z ∧ z = y + 2 ∧ x = y * z) ∧ 
    (x = 3343)) :=
sorry

end board_number_is_3343_l25_25439


namespace complex_sum_cubics_eq_zero_l25_25673

-- Define the hypothesis: omega is a nonreal root of x^3 = 1
def is_nonreal_root_of_cubic (ω : ℂ) : Prop :=
  ω^3 = 1 ∧ ω ≠ 1

-- Now state the theorem to prove the expression evaluates to 0
theorem complex_sum_cubics_eq_zero (ω : ℂ) (h : is_nonreal_root_of_cubic ω) :
  (2 - 2*ω + 2*ω^2)^3 + (2 + 2*ω - 2*ω^2)^3 = 0 :=
by
  -- This is where the proof would go. 
  sorry

end complex_sum_cubics_eq_zero_l25_25673


namespace find_a_l25_25691

noncomputable def curve := λ x : ℝ, sqrt (1 - (x - 1)^2)
noncomputable def regionD := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 = curve p.1 }
noncomputable def dividingCurve (a : ℝ) := λ x : ℝ, a * x * (x - 2)

theorem find_a (a : ℝ) (h₀ : a < 0) (h₁ : ∫ x in 0..2, dividingCurve a x = (π / 4)) :
  a = - (3 * π) / 16 :=
sorry

end find_a_l25_25691


namespace find_n_l25_25867

noncomputable def first_term_1 : ℝ := 12
noncomputable def second_term_1 : ℝ := 4
noncomputable def sum_first_series : ℝ := 18

noncomputable def first_term_2 : ℝ := 12
noncomputable def second_term_2 (n : ℝ) : ℝ := 4 + 2 * n
noncomputable def sum_second_series : ℝ := 90

theorem find_n (n : ℝ) : 
  (first_term_1 = 12) → 
  (second_term_1 = 4) → 
  (sum_first_series = 18) →
  (first_term_2 = 12) →
  (second_term_2 n = 4 + 2 * n) →
  (sum_second_series = 90) →
  (sum_second_series = 5 * sum_first_series) →
  n = 6 :=
by
  intros _ _ _ _ _ _ _
  sorry

end find_n_l25_25867


namespace appendDigitsDivisibleBy504_l25_25424

-- Define the number 456000 and its properties
def baseNumber := 456000
def divisor := 504
def remainder := baseNumber % divisor

-- Define the function to append three digits
def appendDigits (x y z : ℕ) : ℕ := 456000 + x * 100 + y * 10 + z

-- Prove that the number formed by appending x, y, z is divisible by 504
theorem appendDigitsDivisibleBy504 (x y z : ℕ) :
  let n := appendDigits x y z in
  n % divisor = 0 ↔ (x = 1 ∧ y = 2 ∧ z = 0) ∨ (x = 6 ∧ y = 2 ∧ z = 4) :=
by
  sorry

end appendDigitsDivisibleBy504_l25_25424


namespace max_value_an_l25_25539

def a (n : ℕ) : ℚ :=
  ∑ k in finset.range (n - 1) + 1, n / ((n - k) * (2 ^ (k - 1)))

theorem max_value_an {n : ℕ} (h : n ≥ 2) :
  ∀ m, a m ≤ (10 / 3) := by
notice => 
  have : ∀ m, m < 6 ∨ 6 ≤ m := (λ m, classical.em _)
  sorry

end max_value_an_l25_25539


namespace value_of_f_sum_l25_25550

variable (a b c m : ℝ)

def f (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem value_of_f_sum :
  f a b c 5 + f a b c (-5) = 4 :=
by
  sorry

end value_of_f_sum_l25_25550


namespace distance_from_A_to_directrix_l25_25074

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25074


namespace largest_even_digit_multiple_of_9_below_1000_l25_25782

theorem largest_even_digit_multiple_of_9_below_1000 :
  ∃ n : ℕ, n = 882 ∧ n < 1000 ∧ (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d % 2 = 0) ∧ n % 9 = 0 :=
by
  existsi 882
  split
  { rfl }
  split
  { norm_num }
  split
  { intro d
    intro h
    fin_cases d with
    | h1 => norm_num
    | h2 => norm_num
    | h3 => norm_num }
  { norm_num }

end largest_even_digit_multiple_of_9_below_1000_l25_25782


namespace no_infinite_sequence_satisfying_relation_l25_25890

theorem no_infinite_sequence_satisfying_relation :
  ¬(∃ (a : ℕ+ → ℕ+),
    ∀ n : ℕ, a n.succ.succ = a n.succ + nat.sqrt (a n.succ + a n)) :=
begin
  sorry -- Proof will be provided here
end

end no_infinite_sequence_satisfying_relation_l25_25890


namespace distance_from_A_to_directrix_l25_25152

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25152


namespace velocity_of_liquid_at_edge_kinetic_energy_of_liquid_minimum_work_required_l25_25835

variables (M H ω g R : ℝ)

-- Given conditions
def initial_height : ℝ := H
def final_height : ℝ := 2 * H
def angular_speed : ℝ := ω
def gravity : ℝ := g
def radius : ℝ := R

-- Variables
variable (x : ℝ)

-- (a) what is the velocity of the liquid at the edge of the glass
theorem velocity_of_liquid_at_edge :
  let v := 2 * real.sqrt (g * H) in
  v = 2 * real.sqrt (g * H) :=
sorry

-- (b) what is the kinetic energy of the liquid
theorem kinetic_energy_of_liquid :
  let E_kin := (4 / 3) * M * g * H in
  E_kin = (4 / 3) * M * g * H :=
sorry

-- (c) what is the minimum work required to accelerate the liquid
theorem minimum_work_required :
  let W_min := (3 / 2) * M * g * H in
  W_min = (3 / 2) * M * g * H :=
sorry

end velocity_of_liquid_at_edge_kinetic_energy_of_liquid_minimum_work_required_l25_25835


namespace divisibility_of_expression_l25_25287

open Int

theorem divisibility_of_expression (a b : ℤ) (ha : Prime a) (hb : Prime b) (ha_gt7 : a > 7) (hb_gt7 : b > 7) :
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) :=
sorry

end divisibility_of_expression_l25_25287


namespace actual_average_height_l25_25804

theorem actual_average_height (initial_avg_height : ℝ) (num_boys : ℕ) (incorrect_height actual_height : ℝ) :
  initial_avg_height = 180 ∧ num_boys = 35 ∧ incorrect_height = 166 ∧ actual_height = 106 →
  (Real.round ((initial_avg_height * num_boys - (incorrect_height - actual_height)) / num_boys * 100) / 100) = 178.29 :=
by
  sorry

end actual_average_height_l25_25804


namespace parallel_condition_l25_25564

variables (m n : Set ℝ) (α : Set (Set ℝ))

-- Given that n is a subset of plane α
axiom h1 : n ⊆ α

-- Prove that "m is parallel to n" is neither a necessary nor a sufficient condition 
-- for "m is parallel to α".
theorem parallel_condition (hmn : is_parallel m n) : ¬(is_necessary_and_sufficient (λ _, is_parallel m n) (λ _, is_parallel m α)) :=
sorry

end parallel_condition_l25_25564


namespace total_books_in_class_l25_25450

theorem total_books_in_class (num_tables : ℕ) (books_per_table_ratio : ℚ) 
  (h1 : num_tables = 750) (h2 : books_per_table_ratio = 3 / 5) : 
  let books_per_table := books_per_table_ratio * num_tables in
  let total_books := num_tables * books_per_table in
  total_books = 337500 := 
by
  sorry

end total_books_in_class_l25_25450


namespace combinations_sum_l25_25798
open Nat

theorem combinations_sum : 
  let d := [1, 2, 3, 4]
  let count_combinations (n : Nat) := factorial n
  count_combinations 1 + count_combinations 2 + count_combinations 3 + count_combinations 4 = 64 :=
  by
    sorry

end combinations_sum_l25_25798


namespace distance_to_directrix_l25_25095

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25095


namespace number_of_terms_expression_l25_25888

theorem number_of_terms_expression : 
  let expr := (x y z : ℕ) → (x + y + z) ^ 2010 + (x - y - z) ^ 2010,
  ∃ n : ℕ, n = 1012036 ∧ is_number_of_terms (expr) n :=
begin
  sorry
end

end number_of_terms_expression_l25_25888


namespace find_some_number_l25_25443

theorem find_some_number (some_number : ℝ) :
  (0.0077 * some_number) / (0.04 * 0.1 * 0.007) = 990.0000000000001 → 
  some_number = 3.6 :=
by
  intro h
  sorry

end find_some_number_l25_25443


namespace distance_from_A_to_directrix_l25_25148

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25148


namespace avg_geq_half_sum_l25_25682

theorem avg_geq_half_sum (m n : ℕ) (h : 0 < m) (h2 : 0 < n)
    (a : Fin m → ℕ) (a_distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (a_range : ∀ i, 1 ≤ a i ∧ a i ≤ n)
    (a_condition : ∀ i j, a i + a j ≤ n → ∃ k, a i + a j = a k) :
    (∑ i, a i / m : ℚ) ≥ (n + 1) / 2 :=
sorry

end avg_geq_half_sum_l25_25682


namespace value_of_a_l25_25588

theorem value_of_a (a : ℝ) :
  (∀ x ∈ Icc (0 : ℝ) 1, (x^2 - 2*a*x + a - 1) ≥ -2) ∧ 
  (∃ x ∈ Icc (0 : ℝ) 1, (x^2 - 2*a*x + a - 1) = -2) →
  (a = -1 ∨ a = 2) :=
sorry

end value_of_a_l25_25588


namespace smallest_n_exceeds_l25_25791

noncomputable def a : ℕ → ℕ
| 1     := 3
| (n+1) := 3^(a n)

noncomputable def b : ℕ → ℕ
| 1     := 100
| (n+1) := 100^(b n)

theorem smallest_n_exceeds : ∃ n, (1 ≤ n) ∧ b n > a 100 ∧ ∀ m, (1 ≤ m) ∧ b m > a 100 → n ≤ m :=
begin
  sorry
end

end smallest_n_exceeds_l25_25791


namespace smallest_integer_larger_than_sqrt5_plus_sqrt3_to_six_l25_25419

theorem smallest_integer_larger_than_sqrt5_plus_sqrt3_to_six : 
  ∃ n : ℤ, n > real.exp (6 * real.log (real.sqrt 5 + real.sqrt 3)) ∧ n = 3323 :=
sorry

end smallest_integer_larger_than_sqrt5_plus_sqrt3_to_six_l25_25419


namespace _l25_25138

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25138


namespace geometric_sequence_first_term_l25_25677

theorem geometric_sequence_first_term (S_3 S_6 : ℝ) (a_1 q : ℝ)
  (hS3 : S_3 = 6) (hS6 : S_6 = 54)
  (hS3_def : S_3 = a_1 * (1 - q^3) / (1 - q))
  (hS6_def : S_6 = a_1 * (1 - q^6) / (1 - q)) :
  a_1 = 6 / 7 := 
by
  sorry

end geometric_sequence_first_term_l25_25677


namespace max_cos_sum_l25_25860

theorem max_cos_sum
  (A B C : ℝ)
  (h : A + B + C = 180) : 
  ∃ A B C, (A + B + C = 180 ∧ 
  (cos(A) + (1 / 2) * (cos(B - C)) + (1 / 2) * (- cos A)) ≤ 1) := by
  sorry

end max_cos_sum_l25_25860


namespace exists_infinitely_many_natural_numbers_factors_l25_25348

theorem exists_infinitely_many_natural_numbers_factors (k : ℤ) :
  ∃ (P Q : ℤ[X]), 
    (P.degree = 4 ∧ Q.degree = 4) ∧
    (X^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = P * Q) :=
by
  sorry

end exists_infinitely_many_natural_numbers_factors_l25_25348


namespace _l25_25131

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25131


namespace ratio_of_areas_approach_limit_l25_25212

theorem ratio_of_areas_approach_limit
  (b h : ℝ)
  (OG_eq : OG = b - 3 * h)
  (JH_eq : JH = HG)
  (HG_eq : HG = h)
  (EH_eq : EH = sqrt (4 * b * h - 4 * h^2))
  (CG_eq : CG = sqrt (9 * b * h - 9 * h^2))
  (R_eq : R = h * sqrt (4 * b * h - 4 * h^2))
  (K_eq : K = (h / 2) * (sqrt (4 * b * h - 4 * h^2) + sqrt (9 * b * h - 9 * h^2)))
  : tendsto (λ h, (K_eq / R_eq)) (nhds 0) (𝓝 (5 / 4)) :=
begin
  sorry
end

end ratio_of_areas_approach_limit_l25_25212


namespace distance_from_point_to_parabola_directrix_l25_25109

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25109


namespace jared_yearly_earnings_l25_25911

theorem jared_yearly_earnings (monthly_pay_diploma : ℕ) (multiplier : ℕ) (months_in_year : ℕ)
  (h1 : monthly_pay_diploma = 4000) (h2 : multiplier = 3) (h3 : months_in_year = 12) :
  (monthly_pay_diploma * multiplier * months_in_year) = 144000 :=
by
  -- The proof goes here
  sorry

end jared_yearly_earnings_l25_25911


namespace exists_infinite_N_l25_25341

theorem exists_infinite_N (k : ℤ) : ∃ᶠ N : ℕ in at_top, 
  ∃ (A B : Polynomial ℤ), 
    (Polynomial.natDegree A = 4 ∧ Polynomial.natDegree B = 4) ∧
    (Polynomial.mul A B = Polynomial.X^8 + Polynomial.C N * Polynomial.X^4 + 1) :=
begin
  -- Proof goes here
  sorry
end

end exists_infinite_N_l25_25341


namespace distance_from_A_to_directrix_l25_25153

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25153


namespace complex_distance_l25_25952

theorem complex_distance (i : Complex) (h : i = Complex.I) :
  Complex.abs (3 / (2 - i)^2) = 3 / 5 := 
by
  sorry

end complex_distance_l25_25952


namespace algorithm_logical_structures_l25_25863

theorem algorithm_logical_structures (h : ∀ S, S ⊆ {1, 2, 3}) : 
  (∃ S : set ℕ, S ⊆ {1, 2, 3} ∧ (∀ x ∈ S, x = 1 ∨ x = 2 ∨ x = 3)) := 
sorry

end algorithm_logical_structures_l25_25863


namespace number_of_digits_in_sum_l25_25984

noncomputable def num_digits (n : ℕ) : ℕ :=
  (n.toString.length : ℕ)

theorem number_of_digits_in_sum (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (hC : 1 ≤ C ∧ C ≤ 9) :
  let sum := 21478 + (100 * A + 59) + (10 * B + 4) + (10 * C + 6)
  in num_digits sum = 5 :=
by
  sorry

end number_of_digits_in_sum_l25_25984


namespace xyz_value_l25_25886

theorem xyz_value (x y z : ℝ) (h1 : y = x + 1) (h2 : x + y = 2 * z) (h3 : x = 3) : x * y * z = 42 :=
by
  -- proof here
  sorry

end xyz_value_l25_25886


namespace euler_line_intersection_of_circumcircles_l25_25806

-- Given the core components of the problem as definitions.
structure Triangle (α : Type*) :=
(A B C : α)

structure Altitudes (α : Type*) :=
(B1 C1 : α)

structure TangentIntersectionPoints (α : Type*) :=
(M N : α)

def EulerLineIntersection (α : Type*) [Field α] : Prop :=
∀ (triangle : Triangle α) (altitudes : Altitudes α) (tangentIntersections : TangentIntersectionPoints α),
  let BB1 := altitudes.B1
      CC1 := altitudes.C1
      M   := tangentIntersections.M
      N   := tangentIntersections.N
  in ∃ (P : α),
      circumcircle_triangle (triangle.A, M, N).second_intersection_pt(circumcircle_triangle((triangle.A, BB1, CC1)))
      ∧ lies_on_euler_line triangle P

-- Statement of the problem in Lean
theorem euler_line_intersection_of_circumcircles
  (α : Type*) [Field α] :
  EulerLineIntersection α := 
sorry

end euler_line_intersection_of_circumcircles_l25_25806


namespace total_surface_area_prime_rectangular_solid_l25_25521

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Prime n

def prime_edge_lengths (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c

def volume (a b c : ℕ) : ℕ := a * b * c

def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

-- The main theorem statement
theorem total_surface_area_prime_rectangular_solid :
  ∃ (a b c : ℕ), prime_edge_lengths a b c ∧ volume a b c = 105 ∧ surface_area a b c = 142 :=
sorry

end total_surface_area_prime_rectangular_solid_l25_25521


namespace forty_by_forty_one_fillable_l25_25233

noncomputable def fillable_grid (m n : ℕ) : Prop :=
  ∀ grid : array (array ℤ n) m,
    ∃ arrangement : array (array ℤ n) m,
      (∀ i j, i < m ∧ j < n → 
        let val := arrangement[i][j] in
        val = (if i > 0 then (if arrangement[i-1][j] = val then 1 else 0) else 0) 
            + (if i+1 < m then (if arrangement[i+1][j] = val then 1 else 0) else 0)
            + (if j > 0 then (if arrangement[i][j-1] = val then 1 else 0) else 0)
            + (if j+1 < n then (if arrangement[i][j+1] = val then 1 else 0) else 0))

theorem forty_by_forty_one_fillable : fillable_grid 40 41 := sorry

end forty_by_forty_one_fillable_l25_25233


namespace well_quasi_ordered_subsets_l25_25808

variable {X : Type} [Preorder X]

theorem well_quasi_ordered_subsets (hX : WellQuasiOrder X (· ≤ ·)) :
  WellQuasiOrder (Set X) (· ⊆ ·) :=
sorry

end well_quasi_ordered_subsets_l25_25808


namespace midpoint_sum_coords_l25_25703

theorem midpoint_sum_coords (N C D : ℝ × ℝ) (sum_D: ℝ) 
  (hN : N = (3, 5))
  (hC : C = (1, 10)) 
  (hMidpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
  sum_D = (D.1 + D.2) :=
by
  -- Definitions and conditions
  have hNdef : N = (3, 5) := hN,
  have hCdef : C = (1, 10) := hC,
  have hMidpointFormula : (3, 5) = ((1 + D.1) / 2, (10 + D.2) / 2) := by
    rw [hNdef, hCdef] at hMidpoint
    exact hMidpoint,
  sorry

end midpoint_sum_coords_l25_25703


namespace problem1_problem2_l25_25941

-- Define the given angle
def given_angle (α : ℝ) : Prop := α = 2010

-- Define the theorem for the first problem
theorem problem1 (α : ℝ) (k : ℤ) (β : ℝ) (h₁ : given_angle α) 
  (h₂ : 0 ≤ β ∧ β < 360) (h₃ : α = k * 360 + β) : 
  -- Assert that α is in the third quadrant
  (190 ≤ β ∧ β < 270 → true) :=
sorry

-- Define the theorem for the second problem
theorem problem2 (α : ℝ) (θ : ℝ) (h₁ : given_angle α)
  (h₂ : -360 ≤ θ ∧ θ < 720)
  (h₃ : ∃ k : ℤ, θ = α + k * 360) : 
  θ = -150 ∨ θ = 210 ∨ θ = 570 :=
sorry

end problem1_problem2_l25_25941


namespace find_x_when_fx_is_2_l25_25966

def f (x : ℝ) : ℝ := (x - 6) / (x + 2)

theorem find_x_when_fx_is_2 : f (-10) = 2 :=
by
  -- Define the function f
  let f : ℝ → ℝ := λ x, (x - 6) / (x + 2)
  -- The goal is to prove that f(-10) = 2
  show f (-10) = 2
  sorry

end find_x_when_fx_is_2_l25_25966


namespace mathd_circle_chords_l25_25674

noncomputable def max_diam_segment_length : ℝ :=
  1 - sqrt 3

theorem mathd_circle_chords :
  let d := 1 - sqrt 3 in
  ∃ r s t : ℕ,
  (r > 0) ∧ (s > 0) ∧ (t > 0) ∧ (∀ p : ℕ, p^2 ∣ t → p = 1) ∧
  (d = r - s * sqrt t) ∧ (r + s + t = 5) :=
begin
  use [1, 1, 3],
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { intros p hp,
    exfalso,
    exact nat.succ_ne_zero 2 p,
  },
  split,
  { norm_num,
    rw [one_mul, sub_self, sqrt_mul_self],
    exact zero_le_one,
  },
  norm_num,
end

end mathd_circle_chords_l25_25674


namespace distance_to_directrix_l25_25096

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25096


namespace exists_infinite_N_l25_25343

theorem exists_infinite_N (k : ℤ) : ∃ᶠ N : ℕ in at_top, 
  ∃ (A B : Polynomial ℤ), 
    (Polynomial.natDegree A = 4 ∧ Polynomial.natDegree B = 4) ∧
    (Polynomial.mul A B = Polynomial.X^8 + Polynomial.C N * Polynomial.X^4 + 1) :=
begin
  -- Proof goes here
  sorry
end

end exists_infinite_N_l25_25343


namespace determine_n_l25_25247

theorem determine_n (m : ℕ) (n : ℕ) (h_m : m = 1 ∨ m = 2) (h_n : n < 10799) :
  (∑ k in Finset.range (n + 1) \ {0}, 1 / (Real.sin k * Real.sin (k + 1)) = 
   m * (Real.sin n / Real.sin 1 ^ 2)) ↔
  (∃ k : ℤ, n = ⌊Real.pi / 2⌋ + 2 * k * ⌊Real.pi⌋ - 1 ∨ 
            n = ⌊Real.pi / 6⌋ + 2 * k * ⌊Real.pi⌋ - 1 ∨ 
            n = ⌊5 * Real.pi / 6⌋ + 2 * k * ⌊Real.pi⌋ - 1) :=
sorry

end determine_n_l25_25247


namespace tend_mn_half_l25_25665

noncomputable theory
open_locale classical

def f_n (n : ℕ) (x : ℝ) : ℝ := ∑ k in finset.range (2 * n + 1), x^k

def m_n (n : ℕ) : ℝ :=
  Inf (set.range (f_n n))

theorem tend_mn_half (f_n m_n : ℕ → ℝ → ℝ) (k : ℝ) : 
  (∀ n : ℕ, ∃ x : ℝ, f_n n x = m_n n) → 
  (∃ l : ℝ, tendsto (λ n, m_n n) at_top (𝓝 k)) → 
  k = 1 / 2 := 
sorry

end tend_mn_half_l25_25665


namespace chelsea_used_percentage_l25_25490

-- Define the given conditions:
def total_candles : ℕ := 40
def candles_after_alyssa_used_half : ℕ := total_candles / 2
def candles_left_in_room : ℕ := 6
def chelsea_used_candles : ℕ := candles_after_alyssa_used_half - candles_left_in_room

-- Define the proof goal:
theorem chelsea_used_percentage (total_candles = 40) (candles_after_alyssa_used_half = total_candles / 2)
    (candles_left_in_room = 6) (chelsea_used_candles = candles_after_alyssa_used_half - candles_left_in_room) :
  (chelsea_used_candles * 100) / candles_after_alyssa_used_half = 70 := 
by
  sorry

end chelsea_used_percentage_l25_25490


namespace distance_to_directrix_l25_25025

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25025


namespace solve_for_y_l25_25712

theorem solve_for_y : ∀ (y : ℝ), 4 + 2.3 * y = 1.7 * y - 20 → y = -40 :=
by
  sorry

end solve_for_y_l25_25712


namespace percentage_discount_approx_8_l25_25864

theorem percentage_discount_approx_8 (CP MP SP : ℝ) (h1 : CP = 47.50) (h2 : MP = 64.54) (h3 : SP = CP + (0.25 * CP)) (h4 : SP = MP - (8 / 100 * MP)) :
  true := 
by 
  have h5 : SP = 47.50 + (0.25 * 47.50) := by rw [h1, h3]; sorry
  have h6 : SP = 59.375 := sorry 
  have h7 : 59.375 = 64.54 - (8 / 100 * 64.54) := sorry 
  trivial  -- Placeholder for proof

end percentage_discount_approx_8_l25_25864


namespace complex_number_magnitude_l25_25686

theorem complex_number_magnitude (z w : ℂ)
  (h1 : |3 * z - w| = 15)
  (h2 : |z + 3 * w| = 9)
  (h3 : |z + w| = 5) :
  |z| = 4 := 
sorry

end complex_number_magnitude_l25_25686


namespace boy_late_l25_25448

noncomputable def time_late (D V1 V2 : ℝ) (early : ℝ) : ℝ :=
  let T1 := D / V1
  let T2 := D / V2
  let T1_mins := T1 * 60
  let T2_mins := T2 * 60
  let actual_on_time := T2_mins + early
  T1_mins - actual_on_time

theorem boy_late :
  time_late 2.5 5 10 10 = 5 :=
by
  sorry

end boy_late_l25_25448


namespace volume_of_centroid_tetrahedron_l25_25559

theorem volume_of_centroid_tetrahedron (V : ℝ) (hV : V = 54) : 
  let a₁ := (12 * V / Real.sqrt 2)^(1/3) in
  let a₂ := a₁ / 3 in
  let V₂ := (a₂^3 * Real.sqrt 2) / 12 in
  V₂ = 2 :=
by
  have ha₁ : a₁ = (12 * 54 / Real.sqrt 2)^(1/3) := by rw [hV]
  have ha₂ : a₂ = a₁ / 3 := rfl
  have hV₂ : V₂ = (a₂^3 * Real.sqrt 2) / 12 := rfl
  rw [ha₁, ha₂, hV₂]
  sorry

end volume_of_centroid_tetrahedron_l25_25559


namespace largest_even_integer_sum_l25_25394

theorem largest_even_integer_sum :
  let sum_first_30_even := 2 * (List.sum (List.range 30).map(λ n, n + 1)) in
  let n := (sum_first_30_even + 20) / 5 in
  n = 190 :=
by
  sorry

end largest_even_integer_sum_l25_25394


namespace monster_feasting_l25_25831

noncomputable def number_of_people_on_first_ship : ℕ :=
  sorry

theorem monster_feasting (P : ℕ) 
  (h1 : ∀ n, 0 < n ∧ n ≤ 3 → people_on_ship n = 2^(n-1) * P)
  (h2 : people_on_ship 1 + people_on_ship 2 + people_on_ship 3 = 847) :
  P = 121 :=
sorry

end monster_feasting_l25_25831


namespace max_min_value_of_f_l25_25172

theorem max_min_value_of_f (x y z : ℝ) :
  (-1 ≤ 2 * x + y - z) ∧ (2 * x + y - z ≤ 8) ∧
  (2 ≤ x - y + z) ∧ (x - y + z ≤ 9) ∧
  (-3 ≤ x + 2 * y - z) ∧ (x + 2 * y - z ≤ 7) →
  (-6 ≤ 7 * x + 5 * y - 2 * z) ∧ (7 * x + 5 * y - 2 * z ≤ 47) :=
by
  sorry

end max_min_value_of_f_l25_25172


namespace seunghye_saw_number_l25_25657

theorem seunghye_saw_number (x : ℝ) (h : 10 * x - x = 37.35) : x = 4.15 :=
by
  sorry

end seunghye_saw_number_l25_25657


namespace distance_from_A_to_directrix_l25_25002

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25002


namespace anya_more_biscuits_l25_25405

-- Definitions of initial conditions and the final proof goal
theorem anya_more_biscuits :
  let total_payment := 150 in
  let anya_payment := 80 in
  let berini_payment := 50 in
  let carla_payment := 20 in
  let total_biscuits := 30 in
  let anya_initial_biscuits := 10 in
  let anya_proportion := (anya_payment : ℚ) / total_payment in
  let anya_biscuits_by_ratio := anya_proportion * total_biscuits in
  let anya_biscuit_difference := anya_biscuits_by_ratio - anya_initial_biscuits in
  anya_biscuit_difference = 6 :=
by
  sorry

end anya_more_biscuits_l25_25405


namespace vasya_cannot_obtain_more_pieces_l25_25662

theorem vasya_cannot_obtain_more_pieces (n : ℕ) (areas_quotient_three : ∀ i j, i ≠ j → (areas i % 3 = areas j % 3 : Prop))
  (kolya_areas : Fin n → ℕ) 
  (vasya_areas : Fin (n + 5) → ℕ) :
  (∑ i, kolya_areas i = 14400) → 
  (∑ i, vasya_areas i = 14400) → 
  ¬ (∃ f : Fin (n + 5), 
      ∃ g : Fin n, 
        \sum x, vasya_areas x - \sum y, kolya_areas y ∈ {3 * k | k : ℤ}) :=
sorry

end vasya_cannot_obtain_more_pieces_l25_25662


namespace car_speed_in_second_hour_l25_25389

theorem car_speed_in_second_hour (
  speed_first_hour : ℝ,
  avg_speed : ℝ
) (h1 : speed_first_hour = 80) (h2 : avg_speed = 60) : 
  ∃ speed_second_hour : ℝ, speed_second_hour = 40 :=
by
  sorry

end car_speed_in_second_hour_l25_25389


namespace clock_angle_at_330_l25_25978

theorem clock_angle_at_330 :
  let minute_angle := 180
  let hour_angle := 105
  min (abs (minute_angle - hour_angle)) (360 - abs (minute_angle - hour_angle)) = 75 :=
by
  let minute_angle := 180
  let hour_angle := 105
  -- sorry as we don't provide the proof
  sorry

end clock_angle_at_330_l25_25978


namespace find_x_y_l25_25231

variables (A B C M O : Type)
[AddGroup A] [AddGroup B] [AddGroup C]
[AddGroup M] [AddGroup O]
(oa ob oc ma mb mc mo : A)
(ab ac : B)
(x y : ℝ)

-- Conditions in the problem
def condition1 : Prop := oa + ob + oc = 0
def condition2 : Prop := ma + mb + 2 * mc = 0
def condition3 : Prop := mo = x * ab + y * ac

-- Proof statement
theorem find_x_y (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  x + y = -1/12 :=
sorry

end find_x_y_l25_25231


namespace smallest_possible_Y_l25_25252

def digits (n : ℕ) : List ℕ := -- hypothetical function to get the digits of a number
  sorry

def is_divisible (n d : ℕ) : Prop := d ∣ n

theorem smallest_possible_Y :
  ∃ (U : ℕ), (∀ d ∈ digits U, d = 0 ∨ d = 1) ∧ is_divisible U 18 ∧ U / 18 = 61728395 :=
by
  sorry

end smallest_possible_Y_l25_25252


namespace infinite_N_for_factorization_l25_25320

noncomputable def is_special_number (k : ℤ) : ℕ :=
  4 * k ^ 4 - 8 * k ^ 2 + 2

theorem infinite_N_for_factorization : 
  ∀ k : ℤ, ∃ (P Q : Polynomial ℤ), Polynomial.degree P = 4 ∧ Polynomial.degree Q = 4 
          ∧ Polynomial.mul P Q = Polynomial.C 1 + Polynomial.C (is_special_number k) * Polynomial.X^4 + Polynomial.X^8 :=
by sorry

end infinite_N_for_factorization_l25_25320


namespace infinite_nat_N_polynomial_l25_25311

/-- The theorem states that for any integer k, 
there exists a natural number N = 4k^4 - 8k^2 + 2 such that the polynomial 
x^8 + Nx^4 + 1 can be factored into two fourth-degree polynomials with integer coefficients. -/
theorem infinite_nat_N_polynomial (k : ℤ) : ∃ (N : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧ 
  ∃ (P Q : ℤ[X]), P.degree = 4 ∧ Q.degree = 4 ∧ (X^8 + C (↑N) * X^4 + 1) = P * Q :=
by {
  sorry
}

end infinite_nat_N_polynomial_l25_25311


namespace correct_proposition_l25_25260

-- Definitions based on the problem conditions
variables {l m : Type} -- Treat lines as generic type for now
variables {α : Type} -- Treat plane as generic type for now

-- Definitions of geometric relationships
def perpendicular (l : Type) (α : Type) : Prop := sorry
def parallel (l m : Type) : Prop := sorry
def subset (m : Type) (α : Type) : Prop := sorry

-- Propositions
def prop1 : Prop := perpendicular l α ∧ subset m α ∧ perpendicular l m
def prop2 : Prop := perpendicular l α ∧ parallel l m → perpendicular m α
def prop3 : Prop := parallel l α ∧ subset m α → parallel l m
def prop4 : Prop := parallel l α ∧ parallel m α → parallel l m

-- The proof problem: verify that Proposition 2 is the correct one
theorem correct_proposition : prop2 ∧ ¬prop1 ∧ ¬prop3 ∧ ¬prop4 :=
by sorry

end correct_proposition_l25_25260


namespace line_parallel_and_equal_distance_l25_25578

theorem line_parallel_and_equal_distance 
  (C1 : ∃ m, ∀ x y : ℝ, 2 * x - y + m = 0 → true)
  (C2 : ∃ k, ∀ x y : ℝ, 2 * x - y + 3 = 0 → 2 * x - y + k = 0)
  (C3 : ∀ x₁ y₁ x₂ y₂ : ℝ, 2 * x₁ - y₁ + 3 = 0 → 2 * x₂ - y₂ - 1 = 0 → 
        ∃ d, d = abs (3 - (-1)) / sqrt (2^2 + (-1)^2))
  (C4 : ∀ x₁ y₁ x₂ y₂ : ℝ, 2 * x₁ - y₁ + 3 = 0 → 2 * x₂ - y₂ = 0 → 
        ∃ d, d = abs (3 - m) / sqrt (2^2 + (-1)^2) ∧ d = abs (m - (-1)) / sqrt (2^2 + (-1)^2)) :
  ∃ m, ∀ x y : ℝ, 2 * x - y + m = 0 ∧ m = 1 := 
sorry

end line_parallel_and_equal_distance_l25_25578


namespace step1_step2_step3_l25_25570

variable {f : ℝ → ℝ}
variable (a b : ℝ) (m n x : ℝ)

-- Given conditions
axiom f_mono : ∀ (x y : ℝ), (0 < x) → (0 < y) → (x < y) → f(x) < f(y)
axiom f_add : ∀ (m n : ℝ), (0 < m) → (0 < n) → f(m) + f(n) = f(m * n)
axiom f_abs : (0 < a) → (a < b) → |f(a)| = |f(b)| ∧ |f(b)| = 2 * |f((a + b) / 2)|

-- Proof Problem
-- 1. Prove f(1) = 0
theorem step1 : f(1) = 0 := sorry

-- 2. Prove that if f(2) = 1, then the solution set of f(x) < 2 is (0,4)
axiom f_two : f(2) = 1

theorem step2 : {x | 0 < x ∧ x < 4} = {x | f(x) < 2} := sorry

-- 3. Prove that 3 < b < 2 + sqrt(2)
theorem step3 : 3 < b ∧ b < 2 + Real.sqrt 2 := sorry

end step1_step2_step3_l25_25570


namespace min_blue_beads_l25_25464

-- Define a necklace of n beads where each bead is either red, green, or blue.
noncomputable def Bead : Type := 
  | Red
  | Green
  | Blue

-- Define a cyclic necklace of fixed length 175.
structure Necklace where
  beads : Fin 175 → Bead
  cyclic : beads 0 = beads 174

-- Define the two conditions.
def condition1 (necklace : Necklace) : Prop := 
  ∀ i : Fin 175, 
    if necklace.beads i == Bead.Red then 
      (necklace.beads ((i + 1) % 175) ≠ necklace.beads i) 
      ∧ (necklace.beads ((i - 1) % 175) ≠ necklace.beads i)
    else true

def condition2 (necklace : Necklace) : Prop := 
  ∀ i j : Fin 175, i < j,
    (necklace.beads i == Bead.Green ∧ necklace.beads j == Bead.Green) → 
    ∃ k : Fin 175, i < k < j ∧ necklace.beads k == Bead.Blue

-- Define the proof problem.
theorem min_blue_beads (necklace : Necklace) 
  (h1 : condition1 necklace) 
  (h2 : condition2 necklace) : 
  ∃ B : ℕ, B = 30 ∧ (∃ b_set : Fin 175 → bool, 
    (∀ i : Fin 175, b_set i = true ↔ necklace.beads i = Bead.Blue) 
    ∧ (b_set.filter (λ x, x = true)).size = B) :=
sorry

end min_blue_beads_l25_25464


namespace distance_from_point_A_to_directrix_C_l25_25116

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25116


namespace number_of_cats_l25_25236

theorem number_of_cats (total_legs : ℕ) (num_people : ℕ) (legs_per_person : ℕ) (num_children : ℕ)
  (num_dogs : ℕ) (legs_per_dog : ℕ) (legs_per_cat : ℕ) (remaining_legs : ℕ) : 
  total_legs = 22 → 
  num_people = 2 → 
  legs_per_person = 2 → 
  num_children = 3 → 
  num_dogs = 2 → 
  legs_per_dog = 4 → 
  legs_per_cat = 4 → 
  remaining_legs = total_legs - (num_people * legs_per_person + num_children * legs_per_person + num_dogs * legs_per_dog) →
  remaining_legs / legs_per_cat = 1 :=
begin
  intros h_total_legs h_num_people h_legs_per_person h_num_children h_num_dogs h_legs_per_dog h_legs_per_cat h_remaining_legs,
  rw [h_total_legs, h_num_people, h_legs_per_person, h_num_children, h_legs_per_person, h_num_dogs, h_legs_per_dog, h_remaining_legs],
  sorry
end

end number_of_cats_l25_25236


namespace parity_of_f_is_odd_f_is_increasing_range_of_a_l25_25584

def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

theorem parity_of_f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

theorem f_is_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := by
  sorry

theorem range_of_a (a : ℝ) : f (2*a + 1) + f (4*a - 3) > 0 → a > (2 / 7) := by
  sorry

end parity_of_f_is_odd_f_is_increasing_range_of_a_l25_25584


namespace a_n_formula_b_n_formula_S_n_formula_l25_25950

noncomputable def a_n (n : ℕ) : ℕ := 3 * n
noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1) + 3 * n
noncomputable def S_n (n : ℕ) : ℕ := 2^n - 1 + (3 * n^2 + 3 * n) / 2

theorem a_n_formula (n : ℕ) : a_n n = 3 * n := by
  unfold a_n
  rfl

theorem b_n_formula (n : ℕ) : b_n n = 2^(n-1) + 3 * n := by
  unfold b_n
  rfl

theorem S_n_formula (n : ℕ) : S_n n = 2^n - 1 + (3 * n^2 + 3 * n) / 2 := by
  unfold S_n
  rfl

end a_n_formula_b_n_formula_S_n_formula_l25_25950


namespace jessica_withdrawal_l25_25656

/-- Jessica withdrew some money from her bank account, causing her account balance to decrease by 2/5.
    She then deposited an amount equal to 1/4 of the remaining balance. The final balance in her bank account is $750.
    Prove that Jessica initially withdrew $400. -/
theorem jessica_withdrawal (X W : ℝ) 
  (initial_eq : W = (2 / 5) * X)
  (remaining_eq : X * (3 / 5) + (1 / 4) * (X * (3 / 5)) = 750) :
  W = 400 := 
sorry

end jessica_withdrawal_l25_25656


namespace ship_people_first_hundred_years_l25_25829

theorem ship_people_first_hundred_years :
  ∃ S : ℕ, 
    let people_in_second_hundred_years := 2 * S,
        people_in_third_hundred_years := 4 * S,
        total_people := S + people_in_second_hundred_years + people_in_third_hundred_years
    in total_people = 847 ∧ S = 121 :=
by
  sorry

end ship_people_first_hundred_years_l25_25829


namespace correct_proposition_l25_25265

def f₁ (x : ℝ) : ℝ := Real.exp (x - 1)
def f₂ (x : ℝ) : ℝ := Real.cos (2 * x)

def p : Prop := ∀ x y : ℝ, x < y → f₁ x < f₁ y
def q : Prop := ∀ x : ℝ, f₂ (-x) = -f₂ x

theorem correct_proposition : (p ∧ ¬q) :=
by
  have p_true : p := sorry -- Proof that f₁ is increasing
  have q_false : ¬q := sorry -- Proof that f₂ is not odd (even)
  exact And.intro p_true q_false

end correct_proposition_l25_25265


namespace measure_of_AB_l25_25432

theorem measure_of_AB {a b : ℝ} (A B C D E : Type*) [EuclideanGeometry A B C D E]: 
  parallel (line A B) (line C D) → 
  (∠ D = 2 * ∠ B) → 
  (segment_length A D = a) → 
  (segment_length C D = b) → 
  (segment_length A B = a + b) := 
begin
  sorry
end

end measure_of_AB_l25_25432


namespace real_roots_count_l25_25554

variables {a b c d x x₁ x₂ : ℝ}

noncomputable def f (x : ℝ) : ℝ := (a / 3) * x^3 + (b / 2) * x^2 + c * x + d

theorem real_roots_count (h₁ : a < 0) 
  (h₂ : ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ (∀ x : ℝ, deriv (f) x = a * x^2 + b * x + c) 
    ∧ (f x₂ = x₁ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0))) :
  ∃ (k : ℕ), k = 3 ∧ ∀ y : ℝ, (a * f y ^ 2 + b * f y + c = 0 → k = 3) := 
sorry

end real_roots_count_l25_25554


namespace find_xyz_correct_l25_25611

noncomputable def find_xyz
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h1 : x * (y + z) = 360)
  (h2 : y * (z + x) = 405)
  (h3 : z * (x + y) = 450) : ℝ :=
  xyz x y z

theorem find_xyz_correct
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h1 : x * (y + z) = 360)
  (h2 : y * (z + x) = 405)
  (h3 : z * (x + y) = 450) : find_xyz x y z = 2433 :=
by
  sorry

end find_xyz_correct_l25_25611


namespace probability_of_at_least_one_girl_girl_pair_l25_25453

-- Definitions of conditions
def num_boys : ℕ := 8
def num_girls : ℕ := 8

-- Definition of the main proof problem
theorem probability_of_at_least_one_girl_girl_pair :
  (calc_probability num_boys num_girls) = 0.98 :=
sorry

-- Function to calculate the probability of at least one girl-girl pair
noncomputable def calc_probability (boys girls : ℕ) : ℝ :=
  let total_pairs := (fact (boys + girls)) / ((fact 2) ^ boys * (fact boys)) in
  let no_girl_girl_pairs := (fact boys)^2 / total_pairs in
  1 - no_girl_girl_pairs

end probability_of_at_least_one_girl_girl_pair_l25_25453


namespace evaluate_ln_at_e_neg2_l25_25618

def f (x : ℝ) : ℝ := Real.log x

theorem evaluate_ln_at_e_neg2 : f (Real.exp (-2)) = -2 :=
by
  sorry

end evaluate_ln_at_e_neg2_l25_25618


namespace sum_of_coordinates_of_D_l25_25701

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem sum_of_coordinates_of_D :
  let N := (3, 5)
  let C := (1, 10)
  let D := (5, 0)
  N = midpoint C D → D.1 + D.2 = 5 :=
by
  intro h
  sorry

end sum_of_coordinates_of_D_l25_25701


namespace oliver_water_usage_per_week_l25_25695

theorem oliver_water_usage_per_week:
  (bucket_capacity : ℕ)
  (buckets_to_fill: ℕ)
  (buckets_taken_away : ℕ)
  (days_per_week : ℕ)
  (water_per_bucket : ℕ)
  (baths_per_day : ℕ)
  : bucket_capacity = 120 →
    buckets_to_fill = 14 →
    buckets_taken_away = 3 →
    days_per_week = 7 →
    water_per_bucket = 120 →
    baths_per_day = 1 →
    (buckets_to_fill * water_per_bucket - buckets_taken_away * water_per_bucket) * (days_per_week * baths_per_day) = 9240 :=
begin
  intros,
  sorry
end

end oliver_water_usage_per_week_l25_25695


namespace cevian_ratio_product_l25_25646

variables {A B C D E F P : Type*}
variables [HasArea A B C : Type]
variables (AP PD BP PE CP PF : ℝ)

-- Given conditions
def is_concurrent (AD BE CF : Type) := 
(Cevian AD P D A) ∧ (Cevian BE P E B) ∧ (Cevian CF P F C)

def ratio_sum_condition 
  (AP PD BP PE CP PF : ℝ) : Prop :=
  (AP / PD) + (BP / PE) + (CP / PF) = 87

-- Question translated to Lean 4
theorem cevian_ratio_product
  (h_concurrent : is_concurrent AD BE CF)
  (h_ratio_sum : ratio_sum_condition AP PD BP PE CP PF) :
  (AP / PD) * (BP / PE) * (CP / PF) = 85 :=
sorry

end cevian_ratio_product_l25_25646


namespace poly_factorable_l25_25300

-- Define the specific form of N.
noncomputable def N (k : ℤ) : ℕ := 4 * k^4 - 8 * k^2 + 2

-- Define the polynomial.
noncomputable def poly (N : ℕ) : ℤ[X] := X^8 + N * X^4 + 1

-- Define the fourth-degree polynomials used in the factorization.
noncomputable def poly1 (k : ℤ) : ℤ[X] :=
  X^4 - 2 * k * X^3 + 2 * k^2 * X^2 - 2 * k * X + 1

noncomputable def poly2 (k : ℤ) : ℤ[X] := 
  X^4 + 2 * k * X^3 + 2 * k^2 * X^2 + 2 * k * X + 1

-- The proof statement: prove that the polynomial can be factored into the specified fourth-degree polynomials.
theorem poly_factorable : ∀ k : ℤ, ∃ p1 p2 : ℤ[X], poly (N k) = p1 * p2 :=
by
  intro k
  use (poly1 k)
  use (poly2 k)
  sorry

end poly_factorable_l25_25300


namespace distance_from_A_to_directrix_l25_25144

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25144


namespace minimum_blue_beads_l25_25461

theorem minimum_blue_beads (n : ℕ) (red blue green : Type) (necklace : list red ⊕ list blue ⊕ list green) 
  (h1 : ∀ r : red, neighbors r ≠ (same_color r)) 
  (h2 : ∀ g1 g2 : green, ∃ b : blue, between g1 g2 b) 
  (h3 : length necklace = 175) : 
  ∃ blues : nat, blues = 30 :=
by sorry

end minimum_blue_beads_l25_25461


namespace _l25_25134

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25134


namespace sum_all_possible_N_values_l25_25887

theorem sum_all_possible_N_values :
  let n := 5 in
  let max_intersections := (n * (n - 1)) / 2 in
  let all_possible_N := finset.range (max_intersections + 1).to_list in
  all_possible_N.sum = 55 :=
by
  sorry

end sum_all_possible_N_values_l25_25887


namespace distance_from_point_to_parabola_directrix_l25_25103

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25103


namespace distance_to_directrix_l25_25030

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25030


namespace matrix_problem_l25_25672

variable {A B : Matrix (Fin 2) (Fin 2) ℝ}

-- Conditions
theorem matrix_problem
  (h1 : A + B = A ⬝ B)
  (h2 : A ⬝ B = (Matrix.ofFun ![![10, 6], ![-4, 2]])) :
  B ⬝ A = (Matrix.ofFun ![![10, 6], ![-4, 2]]) :=
  sorry

end matrix_problem_l25_25672


namespace full_circle_period_l25_25742

theorem full_circle_period :
  (∃ t, ∀ θ, 0 ≤ θ ∧ θ ≤ t → ∃ r, r = Real.sin θ ∧ (r, θ) covers_full_circle) →
  t = (3 / 2) * Real.pi :=
by
  sorry

end full_circle_period_l25_25742


namespace andrea_jim_age_correct_sum_l25_25868

noncomputable def andrea_jim_age_sum : ℕ :=
let A := 54 in
let J := A - 29 in
A + J

theorem andrea_jim_age_correct_sum
  (A J : ℕ)
  (h1 : A = J + 29)
  (h2 : ∃ x, (A - x) + (J - x) = 47)
  (h3 : ∃ y, (J - y) = 2 * ((J - (some h2)) - (J - some y))
    ∧ A = 3 * (J - some y))
  : A + J = 79 :=
by
  sorry

end andrea_jim_age_correct_sum_l25_25868


namespace percentage_ethanol_in_fuel_B_l25_25865

-- Definitions from the conditions
def tank_capacity : ℝ := 218
def ethanol_percentage_fuel_A : ℝ := 0.12
def total_ethanol : ℝ := 30
def volume_of_fuel_A : ℝ := 122

-- Expression to calculate ethanol in Fuel A
def ethanol_in_fuel_A : ℝ := ethanol_percentage_fuel_A * volume_of_fuel_A

-- The remaining ethanol in Fuel B = Total ethanol - Ethanol in Fuel A
def ethanol_in_fuel_B : ℝ := total_ethanol - ethanol_in_fuel_A

-- The volume of fuel B used to fill the tank
def volume_of_fuel_B : ℝ := tank_capacity - volume_of_fuel_A

-- Statement to prove:
theorem percentage_ethanol_in_fuel_B : (ethanol_in_fuel_B / volume_of_fuel_B) * 100 = 16 :=
sorry

end percentage_ethanol_in_fuel_B_l25_25865


namespace least_prime_factor_of_8_pow_4_minus_8_pow_3_l25_25789

noncomputable def least_prime_factor (n : ℕ) : ℕ :=
  (Nat.factors n).head

theorem least_prime_factor_of_8_pow_4_minus_8_pow_3 :
  least_prime_factor (8^4 - 8^3) = 2 :=
by
  sorry

end least_prime_factor_of_8_pow_4_minus_8_pow_3_l25_25789


namespace non_adjective_primes_sum_l25_25880

-- We will define the necessary components as identified from our problem

def is_adjective_prime (p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  ∃ a : ℕ → ℕ, ∀ n : ℕ,
    a 0 % p = (1 + (1 / a 1) % p) ∧
    a 1 % p = (1 + (1 / (1 + (1 / a 2) % p)) % p) ∧
    a 2 % p = (1 + (1 / (1 + (1 / (1 + (1 / a 3) % p))) % p))

def is_not_adjective_prime (p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  ¬ is_adjective_prime p

def first_three_non_adjective_primes_sum : ℕ :=
  3 + 7 + 23

theorem non_adjective_primes_sum :
  first_three_non_adjective_primes_sum = 33 := 
  sorry

end non_adjective_primes_sum_l25_25880


namespace find_num_officers_l25_25218

noncomputable def num_officers (O : ℕ) : Prop :=
  let avg_salary_all := 120
  let avg_salary_officers := 440
  let avg_salary_non_officers := 110
  let num_non_officers := 480
  let total_salary :=
    avg_salary_all * (O + num_non_officers)
  let salary_officers :=
    avg_salary_officers * O
  let salary_non_officers :=
    avg_salary_non_officers * num_non_officers
  total_salary = salary_officers + salary_non_officers

theorem find_num_officers : num_officers 15 :=
sorry

end find_num_officers_l25_25218


namespace problem_statement_l25_25755

variables {a b c x y z : ℝ}

theorem problem_statement
  (h1 : x^2 - y^2 - z^2 = 2 * a * y * z)
  (h2 : -x^2 + y^2 - z^2 = 2 * b * z * x)
  (h3 : -x^2 - y^2 + z^2 = 2 * c * x * y)
  (h4 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  x^2 * (1 - b^2) = y^2 * (1 - a^2) ∧
  x^2 * (1 - b^2) = x * y * (ab - c) ∧
  a^2 + b^2 + c^2 - 2abc = 1 := 
sorry

end problem_statement_l25_25755


namespace find_seventh_value_l25_25870

theorem find_seventh_value (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
  (h₁ : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 0)
  (h₂ : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 10)
  (h₃ : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 100) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 210 :=
sorry

end find_seventh_value_l25_25870


namespace count_valid_numbers_l25_25530

open Nat

def valid_numbers_count : ℕ := 
  let valid_count (a b c d : ℕ) : bool := d = a + b + c
  (List.range 4).toFinset.sum $ λ a => -- thousands place
    (List.range 10).toFinset.sum $ λ b => -- hundreds place
      (List.range 10).toFinset.sum $ λ c => -- tens place
        (List.range 10).toFinset.count $ λ d => -- units place
          1000 * a + 100 * b + 10 * c + d ∈ set.Icc 1000 2999 ∧ valid_count a b c d

theorem count_valid_numbers : valid_numbers_count = 109 := by
  sorry

end count_valid_numbers_l25_25530


namespace find_PC_l25_25208

-- Given conditions
variable (A B C P M H K : Type)
variable [EuclideanGeometry A B C P M]
variable [IsTriangle A B C]
variable (AB AC : ℝ)
variable (BC : ℝ)
variable (HK : ℝ)
variable (BP PC : ℝ)

-- Setting the conditions as hypotheses
axiom AB_eq_AC : AB = 13
axiom BC_eq_10 : BC = 10
axiom BP_less_PC : BP < PC
axiom orthocenters_HK_are_two : HK = 2

-- The statement we need to prove
theorem find_PC :
    PC = 7.4 :=
by
    -- Provide proof here
    sorry

end find_PC_l25_25208


namespace part_one_geometric_seq_part_two_inequality_range_of_a_l25_25416

noncomputable def vector_sequence (n : ℕ) : ℝ × ℝ :=
  if n = 1 then (1,1)
  else if n > 1 then let (x, y) := vector_sequence (n - 1) in (1/2 * (x - y), 1/2 * (x + y))
  else (0,0)

def vec_length (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def a_n_len (n : ℕ) : ℝ := vec_length (vector_sequence n)

theorem part_one_geometric_seq (r : ℝ) : ∀ n : ℕ, n > 0 → a_n_len n = r ^ n * a_n_len 1 :=
  sorry

noncomputable def b_n (n : ℕ) := n^2 / 4

theorem part_two_inequality (a : ℝ) (n : ℕ) (hn : n > 0) :
  (∑ i in finset.range (2*n) \ finset.range (n+1), real.sqrt (1 / b_n i)) > a * (a + 2) :=
  sorry

theorem range_of_a : set.Ioo (-1 - real.sqrt 78 / 6) (-1 + real.sqrt 78 / 6) :=
  sorry

end part_one_geometric_seq_part_two_inequality_range_of_a_l25_25416


namespace largest_even_digit_multiple_of_9_under_1000_l25_25778

noncomputable def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

noncomputable def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else n.digits 10

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (digits n).sum

noncomputable def all_even_digits (n : ℕ) : Prop :=
  ∀ d ∈ digits n, is_even_digit d

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ n, n < 1000 ∧ all_even_digits n ∧ sum_of_digits n % 9 = 0 ∧ n = 360 :=
begin
  use 360,
  split,
  { exact nat.lt_succ_self 359 },
  split,
  { intros d hd,
    unfold digits at hd,
    rw list.mem_iff_exists_get at hd,
    rcases hd with ⟨k, hk⟩,
    repeat { rw list.get? },
    simp only [digits] at hk,
    have : k < 3 := (nat.lt_of_succ_lt_succ (list.length_le_of_lt_some hk)).trans_le (by norm_num),
    interval_cases k,
    { rw [hk, nat.digits, nat.digits_aux'],
      norm_num [is_even_digit] },
    { rw [hk, nat.digits, nat.digits_aux', nat.div_eq, nat.mod_eq_of_lt, nat.add_eq_zero_eq_zero_and_eq_zero, nat.zero_eq, eq_self_iff_true, true_and],
      norm_num [is_even_digit] },
    { rw [hk, nat.digits, nat.digits_aux', nat.div_eq, nat.mod_eq_of_lt, nat.add_eq_zero_eq_zero_and_eq_zero, nat.zero_eq, eq_self_iff_true, true_and],
      norm_num [is_even_digit] } },
  { simp only [sum_of_digits, digits],
    exact nat.digits_sum_eq 360 10,
    exact dec_trivial },
  { refl }
end

end largest_even_digit_multiple_of_9_under_1000_l25_25778


namespace minimum_shift_value_l25_25690

noncomputable def f (x : ℝ) : ℝ := Real.cos x

theorem minimum_shift_value :
  ∃ m > 0, ∀ x, f (x + m) = Real.sin x ∧ m = 3 * Real.pi / 2 :=
by
  sorry

end minimum_shift_value_l25_25690


namespace num_ways_two_different_colors_l25_25557

theorem num_ways_two_different_colors 
  (red white blue : ℕ) 
  (total_balls : ℕ) 
  (choose : ℕ → ℕ → ℕ) 
  (h_red : red = 2) 
  (h_white : white = 3) 
  (h_blue : blue = 1) 
  (h_total : total_balls = red + white + blue) 
  (h_choose_total : choose total_balls 3 = 20)
  (h_choose_three_diff_colors : 2 * 3 * 1 = 6)
  (h_one_color : 1 = 1) :
  choose total_balls 3 - 6 - 1 = 13 := 
by
  sorry

end num_ways_two_different_colors_l25_25557


namespace part1_part2_l25_25724

noncomputable def h (x : ℝ) : ℝ := x^2

noncomputable def phi (x : ℝ) : ℝ := 2 * Real.exp 1 * Real.log x

noncomputable def F (x : ℝ) : ℝ := h x - phi x

theorem part1 :
  ∃ (x : ℝ), x > 0 ∧ Real.log x = 1 ∧ F x = 0 :=
sorry

theorem part2 :
  ∃ (k b : ℝ), 
  (∀ x > 0, h x ≥ k * x + b) ∧
  (∀ x > 0, phi x ≤ k * x + b) ∧
  (k = 2 * Real.exp 1 ∧ b = -Real.exp 1) :=
sorry

end part1_part2_l25_25724


namespace sin_alpha_bounded_l25_25899

theorem sin_alpha_bounded (x α : ℝ) (m : ℝ) (h1 : |sin x| ≤ 1) (h2 : m = sin α) : |sin α| ≤ 1 :=
sorry

end sin_alpha_bounded_l25_25899


namespace poly_factorable_l25_25299

-- Define the specific form of N.
noncomputable def N (k : ℤ) : ℕ := 4 * k^4 - 8 * k^2 + 2

-- Define the polynomial.
noncomputable def poly (N : ℕ) : ℤ[X] := X^8 + N * X^4 + 1

-- Define the fourth-degree polynomials used in the factorization.
noncomputable def poly1 (k : ℤ) : ℤ[X] :=
  X^4 - 2 * k * X^3 + 2 * k^2 * X^2 - 2 * k * X + 1

noncomputable def poly2 (k : ℤ) : ℤ[X] := 
  X^4 + 2 * k * X^3 + 2 * k^2 * X^2 + 2 * k * X + 1

-- The proof statement: prove that the polynomial can be factored into the specified fourth-degree polynomials.
theorem poly_factorable : ∀ k : ℤ, ∃ p1 p2 : ℤ[X], poly (N k) = p1 * p2 :=
by
  intro k
  use (poly1 k)
  use (poly2 k)
  sorry

end poly_factorable_l25_25299


namespace hyperbola_eccentricity_is_in_range_l25_25170

noncomputable def hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) : Set ℝ :=
  if h : ∃ e : ℝ, (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1 → true))
    ∧ ( ∀ F : ℝ × ℝ, F = (-a, 0) → true )
    ∧ ( ∀ C : ℝ × ℝ, C = (a, 0) → true )
    ∧ ( ∀ A B : ℝ × ℝ, A = (-c, y₁) ∧ B = (-c, y₂) ∧ (x = -c) → true )
    ∧ ( (∃ circle : ℝ × ℝ, circle = (⟨-c, y₁⟩, ⟨-c, y₂⟩)) → true )
    ∧ ( e > 2 ): Set.univ else ∅

theorem hyperbola_eccentricity_is_in_range (a b : ℝ) (ha : a > 0) (hb : b > 0)
  : hyperbola_eccentricity_range a b ha hb = (Set.Ioi 2) := by
  sorry

end hyperbola_eccentricity_is_in_range_l25_25170


namespace infinite_nat_N_polynomial_l25_25310

/-- The theorem states that for any integer k, 
there exists a natural number N = 4k^4 - 8k^2 + 2 such that the polynomial 
x^8 + Nx^4 + 1 can be factored into two fourth-degree polynomials with integer coefficients. -/
theorem infinite_nat_N_polynomial (k : ℤ) : ∃ (N : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧ 
  ∃ (P Q : ℤ[X]), P.degree = 4 ∧ Q.degree = 4 ∧ (X^8 + C (↑N) * X^4 + 1) = P * Q :=
by {
  sorry
}

end infinite_nat_N_polynomial_l25_25310


namespace find_d2_l25_25683

def p (x : ℤ) : ℤ := d_4 * x^4 + d_3 * x^3 + d_2 * x^2 + d_1 * x + d_0

def E (n : ℤ) : ℤ := sorry  -- The function definition is based on the problem's conditions

theorem find_d2 (d_4 d_3 d_1 d_0 : ℤ) : 
  (∀ n, n ≥ 7 → n % 2 = 1 → n % 3 = 0 → E n = p n) → 
  d_2 = 6 := 
sorry

end find_d2_l25_25683


namespace poly_factorable_l25_25302

-- Define the specific form of N.
noncomputable def N (k : ℤ) : ℕ := 4 * k^4 - 8 * k^2 + 2

-- Define the polynomial.
noncomputable def poly (N : ℕ) : ℤ[X] := X^8 + N * X^4 + 1

-- Define the fourth-degree polynomials used in the factorization.
noncomputable def poly1 (k : ℤ) : ℤ[X] :=
  X^4 - 2 * k * X^3 + 2 * k^2 * X^2 - 2 * k * X + 1

noncomputable def poly2 (k : ℤ) : ℤ[X] := 
  X^4 + 2 * k * X^3 + 2 * k^2 * X^2 + 2 * k * X + 1

-- The proof statement: prove that the polynomial can be factored into the specified fourth-degree polynomials.
theorem poly_factorable : ∀ k : ℤ, ∃ p1 p2 : ℤ[X], poly (N k) = p1 * p2 :=
by
  intro k
  use (poly1 k)
  use (poly2 k)
  sorry

end poly_factorable_l25_25302


namespace complex_matrix_property_l25_25681

variable {n : ℕ}
variables (α : ℂ) (A : Matrix (Fin n) (Fin n) ℂ)

def is_hermitian (A : Matrix (Fin n) (Fin n) ℂ) : Prop := Aᴴ = A
def star (A : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ := Aᴴ

theorem complex_matrix_property
  (hα : α ≠ 0)
  (hA : A ≠ 0)
  (h_cond : A * A + (star A) * (star A) = α • (A * (star A))) :
  α ∈ ℝ ∧ |α| ≤ 2 ∧ A * (star A) = (star A) * A :=
by
  sorry

end complex_matrix_property_l25_25681


namespace _l25_25132

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25132


namespace algebraic_expression_value_l25_25621

theorem algebraic_expression_value (x y : ℝ) (h : 2 * x - y = 2) : 6 * x - 3 * y + 1 = 7 := 
by
  sorry

end algebraic_expression_value_l25_25621


namespace exists_perfect_square_between_S_n_and_S_nplus1_l25_25268

open Nat

-- Definition of the sum of the first n primes
def S (n : ℕ) : ℕ := (Finset.range (n + 1)).filter Nat.prime |>.sum id

theorem exists_perfect_square_between_S_n_and_S_nplus1 (n : ℕ) (hn : 0 < n) : 
  ∃ k : ℕ, (S n) < k^2 ∧ k^2 < S (n + 1) := 
  sorry

end exists_perfect_square_between_S_n_and_S_nplus1_l25_25268


namespace exists_infinitely_many_N_l25_25292

theorem exists_infinitely_many_N (k : ℤ) : ∃ N : ℕ, ∃ A B : polynomial ℤ, 
  N = 4 * k^4 - 8 * k^2 + 2 ∧ polynomial.nat_degree A = 4 ∧ polynomial.nat_degree B = 4 ∧
  (polynomial.X^8 + polynomial.C (N:ℤ) * polynomial.X^4 + 1) = A * B :=
sorry

end exists_infinitely_many_N_l25_25292


namespace manuel_wins_probability_l25_25693

-- Definitions for the conditions
inductive Color
| red
| black

inductive Card
| hearts
| diamonds
| spades
| clubs

-- Function to determine the color of a card
def cardColor : Card → Color 
| Card.hearts   := Color.red
| Card.diamonds := Color.red
| Card.spades   := Color.black
| Card.clubs    := Color.black

-- The main probability computation statement
theorem manuel_wins_probability :
  (2 / 3 : ℚ) = 
  let draws := [(Card.hearts, Card.spades), (Card.hearts, Card.clubs), (Card.diamonds, Card.spades), (Card.diamonds, Card.clubs),
                (Card.spades, Card.hearts), (Card.spades, Card.diamonds), (Card.clubs, Card.hearts), (Card.clubs, Card.diamonds)],
      total_draws := 12 in
  draws.length / total_draws := by
  sorry

end manuel_wins_probability_l25_25693


namespace first_group_first_drawing_l25_25766

-- Define the conditions
def isFirstNumberOfGroup (groupNumber studentNumber : ℕ) : Prop :=
  ∃ g students_per_group total_students (total_groups: ℕ), 
    (total_students = 160) ∧
    (total_groups = 20) ∧
    (total_students = total_groups * students_per_group) ∧
    students_per_group = 8 ∧
    groupNumber = g ∧
    studentNumber = (g - 1) * students_per_group + 1

-- Define the condition for the specific group
def isSpecificGroup (students_per_group studentNumber : ℕ) : Prop :=
  ∃ total_groups (g₁ g₂ : ℕ), 
    (total_groups = 20) ∧ 
    (studentNumber = (g₁ - 1) * students_per_group + g₂) ∧
    (g₁ = 16) ∧ 
    (studentNumber = 126)

-- The theorem to prove
theorem first_group_first_drawing : 
  ∀ (studentNumber : ℕ), 
   (isSpecificGroup 8 126) → 
   (isFirstNumberOfGroup 1 studentNumber) → 
   (studentNumber = 8) := 
 by 
   intros studentNumber h1 h2 
   sorry

end first_group_first_drawing_l25_25766


namespace g_of_neg1_g_of_3_l25_25687

def g (x : ℝ) : ℝ :=
if x < 2 then 2 * x - 4 else 10 - 3 * x

theorem g_of_neg1 : g (-1) = -6 :=
by
  sorry

theorem g_of_3 : g (3) = 1 :=
by
  sorry

end g_of_neg1_g_of_3_l25_25687


namespace waterloo_street_houses_l25_25276

/-- On a particular street in Waterloo, there are exactly 14 houses, each numbered with an integer
between 500 and 599, inclusive. The 14 house numbers form an arithmetic sequence in which 7 terms 
are even and 7 terms are odd. One of the houses is numbered 555, and none of the remaining 13 numbers 
has two equal digits. This Lean statement proves that the smallest number among these 14 house numbers 
is 506. -/
theorem waterloo_street_houses :
  ∃ (a : ℕ) (d : ℕ), 
    (∀ n, 0 ≤ n ∧ n < 14 → 500 ≤ a + n * d ∧ a + n * d ≤ 599) ∧
    (∀ i j, 0 ≤ i ∧ i < 14 ∧ 0 ≤ j ∧ j < 14 ∧ i ≠ j → a + i * d ≠ a + j * d) ∧
    (∃ k, 0 ≤ k ∧ k < 14 ∧ a + k * d = 555) ∧
    (∀ n, (0 ≤ n ∧ n < 14 ∧ a + n * d ≠ 555) -> (a + n * d).digits.nodup) ∧
    (7 = (Finset.range 14).filter (λ n, even (a + n * d)).card) ∧ 
    (7 = (Finset.range 14).filter (λ n, odd (a + n * d)).card) ∧ 
    a = 506 
  :=
sorry

end waterloo_street_houses_l25_25276


namespace distance_from_point_A_to_directrix_C_l25_25127

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25127


namespace infinite_natural_numbers_factorable_polynomial_l25_25337

theorem infinite_natural_numbers_factorable_polynomial :
  ∃ᶠ N in at_top, ∃ (k : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧
    ∃ (f g : ℤ[x]), (degree f = 4) ∧ (degree g = 4) ∧ (f * g = X^8 + C (N : ℤ) * X^4 + 1) :=
sorry

end infinite_natural_numbers_factorable_polynomial_l25_25337


namespace distance_from_A_to_directrix_of_parabola_l25_25067

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25067


namespace range_of_a_l25_25543

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x - 2 * a) * (a * x - 1) < 0 → (x > 1 / a ∨ x < 2 * a)) → (a ≤ -Real.sqrt 2 / 2) :=
by
  intro h
  sorry

end range_of_a_l25_25543


namespace initial_average_age_is_16_l25_25730

-- Given conditions
variable (N : ℕ) (newPersons : ℕ) (avgNewPersonsAge : ℝ) (totalPersonsAfter : ℕ) (avgAgeAfter : ℝ)
variable (initial_avg_age : ℝ) -- This represents the initial average age (A) we need to prove

-- The specific values from the problem
def N_value : ℕ := 20
def newPersons_value : ℕ := 20
def avgNewPersonsAge_value : ℝ := 15
def totalPersonsAfter_value : ℕ := 40
def avgAgeAfter_value : ℝ := 15.5

-- Theorem statement to prove that the initial average age is 16 years
theorem initial_average_age_is_16 (h1 : N = N_value) (h2 : newPersons = newPersons_value) 
  (h3 : avgNewPersonsAge = avgNewPersonsAge_value) (h4 : totalPersonsAfter = totalPersonsAfter_value) 
  (h5 : avgAgeAfter = avgAgeAfter_value) : initial_avg_age = 16 := by
  sorry

end initial_average_age_is_16_l25_25730


namespace plane_angles_congruent_implies_dihedral_angles_congruent_dihedral_angles_congruent_implies_plane_angles_congruent_l25_25290

open_locale classical -- To handle classical logic

variables {Point : Type} [inhabited Point] [plane_geom : PlaneGeometry Point]

-- Definition of trihedral angle
structure TrihedralAngle (A B C S : Point) :=
  (angle_ASB : PlaneAngle S A B)
  (angle_ASC : PlaneAngle S A C)
  (angle_BSC : PlaneAngle S B C)

def PlaneAngle_congruent {A B : Point} (α β : PlaneAngle A B) : Prop :=
  α ≃ β -- using ≃ for congruence of plane angles

def DihedralAngle_congruent {A B : Point} (δ ε : DihedralAngle A B) : Prop :=
  δ ≃ ε -- using ≃ for congruence of dihedral angles

-- Proving the given conditions
theorem plane_angles_congruent_implies_dihedral_angles_congruent
  {A B C S : Point}
  (α β : PlaneAngle S A B)
  (γ : PlaneAngle S A C)
  (δ ε : DihedralAngle B C)
  (h : PlaneAngle_congruent α γ) :
  DihedralAngle_congruent δ ε :=
sorry
  
theorem dihedral_angles_congruent_implies_plane_angles_congruent
  {A B C S : Point}
  (δ ε : DihedralAngle B C)
  (α β : PlaneAngle S A B)
  (γ : PlaneAngle S A C)
  (h : DihedralAngle_congruent δ ε) :
  PlaneAngle_congruent α γ :=
sorry

end plane_angles_congruent_implies_dihedral_angles_congruent_dihedral_angles_congruent_implies_plane_angles_congruent_l25_25290


namespace maximize_garden_area_to_find_parallel_side_l25_25470

noncomputable def max_garden_length (x y : ℝ) : ℝ :=
200 * x - 2 * x^2

theorem maximize_garden_area_to_find_parallel_side :
  ∃ x y : ℝ, 4 * x + y = 200 ∧ (∀ z : ℝ, max_garden_length x y = max_garden_length z (200 - 2 * z) → y = 100) :=
begin
  sorry
end

end maximize_garden_area_to_find_parallel_side_l25_25470


namespace problem_statement_l25_25580

noncomputable def a_n (n : ℕ) : ℤ := 3 * n - 2
noncomputable def b_n (n : ℕ) : ℚ := (3 * n + 1) / (3 * n - 2)
noncomputable def c_n (n : ℕ) : ℚ := (3 * n + 1) / 2^n

def S_n (n : ℕ) : ℕ → ℤ
| 0     := 0
| (n+1) := a_n (n+1) + S_n n

theorem problem_statement :
    (a_n 10 = 28) ∧ (S_n 8 = 92) ∧ (∀ n : ℕ, c_n n = (3 * n + 1) / 2^n) →
    ∀ n : ℕ, (∑ i in finset.range n, c_n (i + 1)) = 7 - (3 * n + 7) / 2^n := 
by
  sorry

end problem_statement_l25_25580


namespace modulus_of_z_l25_25582

noncomputable def z : ℂ := (3 - complex.i) / (1 + 2 * complex.i)

theorem modulus_of_z : complex.abs z = real.sqrt 2 := by
  sorry

end modulus_of_z_l25_25582


namespace select_3_individuals_in_5x5_matrix_l25_25400

-- Definition of the problem conditions
noncomputable def selection3_individuals (n : ℕ) (r : ℕ) : ℕ :=
if n = 25 ∧ r = 3 then 
  (25 * 12 * 6) / (Nat.factorial 3) 
else 0

-- Theorem stating the equivalent proof problem
theorem select_3_individuals_in_5x5_matrix (n : ℕ) (r : ℕ) (ways : ℕ) :
  n = 25 →
  r = 3 →
  ways = selection3_individuals n r →
  ways = 300 :=
by
  intro h1 h2 h3
  rw [selection3_individuals, if_pos ⟨h1, h2⟩] at h3
  rw h3
  sorry

end select_3_individuals_in_5x5_matrix_l25_25400


namespace problem_statement_l25_25692

open Set

variable (U P Q : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5})

theorem problem_statement : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end problem_statement_l25_25692


namespace number_of_x_values_l25_25721

theorem number_of_x_values (x : ℕ) (h1 : ⌈real.sqrt x⌉ = 20) (h2 : x % 3 = 0) : 
  (count (λ n, 361 < n ∧ n ≤ 400 ∧ n % 3 = 0) (list.range (400 + 1))) = 13 :=
sorry

end number_of_x_values_l25_25721


namespace infinite_N_for_factorization_l25_25321

noncomputable def is_special_number (k : ℤ) : ℕ :=
  4 * k ^ 4 - 8 * k ^ 2 + 2

theorem infinite_N_for_factorization : 
  ∀ k : ℤ, ∃ (P Q : Polynomial ℤ), Polynomial.degree P = 4 ∧ Polynomial.degree Q = 4 
          ∧ Polynomial.mul P Q = Polynomial.C 1 + Polynomial.C (is_special_number k) * Polynomial.X^4 + Polynomial.X^8 :=
by sorry

end infinite_N_for_factorization_l25_25321


namespace smallest_a_satisfies_conditions_l25_25364

def parabola_vertex (a b c : ℚ) : Prop :=
  (1 / 3 = -b / (2 * a)) ∧ (-1 / 4 = (4 * a * c - b^2) / (4 * a))

noncomputable def smallest_positive_a : ℚ :=
  (9 : ℚ) / 16

theorem smallest_a_satisfies_conditions :
  ∃ a b c : ℚ, a > 0 ∧ parabola_vertex a b c ∧ (2 * a + b + 3 * c).denom = 1 ∧ a = smallest_positive_a :=
by
  sorry

end smallest_a_satisfies_conditions_l25_25364


namespace cristina_head_start_l25_25274

noncomputable def distance (speed time : ℝ) : ℝ := speed * time

theorem cristina_head_start :
  let nicky_speed := 3
  let cristina_speed := 5
  let nicky_time := 30
  let nicky_distance := distance nicky_speed nicky_time
  let cristina_time := nicky_distance / cristina_speed
  let head_start := nicky_time - cristina_time
  in head_start = 12 :=
by
  let nicky_speed := 3
  let cristina_speed := 5
  let nicky_time := 30
  let nicky_distance := distance nicky_speed nicky_time
  let cristina_time := nicky_distance / cristina_speed
  let head_start := nicky_time - cristina_time
  have h1 : nicky_distance = 90 := by norm_num
  have h2 : cristina_time = 18 := by norm_num
  have h3 : head_start = 12 := by norm_num
  exact h3

end cristina_head_start_l25_25274


namespace value_of_expression_l25_25422

theorem value_of_expression : (2112 - 2021) ^ 2 / 169 = 49 := by
  sorry

end value_of_expression_l25_25422


namespace find_m_l25_25767

-- Define a structure for the vector to handle vector operations
structure Vector (α : Type) := (x : α) (y : α)

-- Define the given vectors
def a : Vector ℝ := ⟨1, 2⟩
def b (m : ℝ) : Vector ℝ := ⟨2 + m, 3 - m⟩
def c (m : ℝ) : Vector ℝ := ⟨3 * m, 1⟩

-- Define the subtraction of two vectors
def sub (u v : Vector ℝ) : Vector ℝ := ⟨u.x - v.x, u.y - v.y⟩

-- Define parallel vectors property
def parallel (u v : Vector ℝ) : Prop := u.x * v.y = u.y * v.x

-- The main statement to prove
theorem find_m (m : ℝ) : parallel a (sub (c m) (b m)) ↔ m = 2 / 3 := by
  sorry

end find_m_l25_25767


namespace distance_from_point_A_to_directrix_C_l25_25122

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25122


namespace largest_even_digit_multiple_of_9_below_1000_l25_25781

theorem largest_even_digit_multiple_of_9_below_1000 :
  ∃ n : ℕ, n = 882 ∧ n < 1000 ∧ (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d % 2 = 0) ∧ n % 9 = 0 :=
by
  existsi 882
  split
  { rfl }
  split
  { norm_num }
  split
  { intro d
    intro h
    fin_cases d with
    | h1 => norm_num
    | h2 => norm_num
    | h3 => norm_num }
  { norm_num }

end largest_even_digit_multiple_of_9_below_1000_l25_25781


namespace infinite_N_for_factorization_l25_25317

noncomputable def is_special_number (k : ℤ) : ℕ :=
  4 * k ^ 4 - 8 * k ^ 2 + 2

theorem infinite_N_for_factorization : 
  ∀ k : ℤ, ∃ (P Q : Polynomial ℤ), Polynomial.degree P = 4 ∧ Polynomial.degree Q = 4 
          ∧ Polynomial.mul P Q = Polynomial.C 1 + Polynomial.C (is_special_number k) * Polynomial.X^4 + Polynomial.X^8 :=
by sorry

end infinite_N_for_factorization_l25_25317


namespace range_of_a_l25_25996

noncomputable def function_monotone_interval (a : ℝ) :=
  ∀ x y : ℝ, x ∈ Ioo (- (sqrt 3 / 3)) (sqrt 3 / 3) → y ∈ Ioo (- (sqrt 3 / 3)) (sqrt 3 / 3) → x < y → f(x) > f(y)

theorem range_of_a (a : ℝ) (h : function_monotone_interval a) : a > 0 :=
sorry 

end range_of_a_l25_25996


namespace _l25_25136

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25136


namespace problem_statement_l25_25928

def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 4)

theorem problem_statement :
  {x : ℝ | ∃ k : ℤ, f x ≥ Real.sqrt 3} =
  ⋃ (k : ℤ), Set.Ico (k * Real.pi / 2 + Real.pi / 24) (k * Real.pi / 2 + Real.pi / 8) :=
by sorry

end problem_statement_l25_25928


namespace james_balloons_l25_25655

-- Definitions
def amy_balloons : ℕ := 513
def extra_balloons_james_has : ℕ := 709

-- Statement of the problem
theorem james_balloons : amy_balloons + extra_balloons_james_has = 1222 :=
by
  -- Placeholder for the actual proof
  sorry

end james_balloons_l25_25655


namespace inequality1_inequality2_l25_25589

noncomputable def f (x a : ℝ) := |x - a| + 1

theorem inequality1 {x : ℝ} (h1 : f x 4 < 1 + |2 * x + 1|) : x ∈ Ioi 1 ∪ Iio (-5) := sorry

theorem inequality2 (m n : ℝ) (hm : m > 0) (hn: n > 0) (h2 : (1 / m) + (1 / n) = 1) : 
  m + 2 * n ≥ 3 + 2 * Real.sqrt 2 := sorry

end inequality1_inequality2_l25_25589


namespace problem1_problem2_problem3_l25_25822

noncomputable def chi_squared_test (n a b c d α χ_squared_critical : ℝ) : Prop :=
  let χ_squared := n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d)) in
  χ_squared > χ_squared_critical

noncomputable def likelihood_ratio (nAB nA_notB : ℝ) : ℝ :=
  nAB / nA_notB

noncomputable def probability_distribution (C_combinations : ℕ → ℕ → ℚ) (C_8_3 : ℚ) :
  (ℚ × ℚ) × (ℚ × ℚ) × (ℚ × ℚ) × (ℚ × ℚ) :=
  let P_X0 := C_combinations 3 3 / C_8_3
  let P_X1 := C_combinations 5 1 * C_combinations 3 2 / C_8_3
  let P_X2 := C_combinations 5 2 * C_combinations 3 1 / C_8_3
  let P_X3 := C_combinations 5 3 / C_8_3
  ((0, P_X0), (1, P_X1), (2, P_X2), (3, P_X3))

noncomputable def expected_value (dist : (ℚ × ℚ) × (ℚ × ℚ) × (ℚ × ℚ) × (ℚ × ℚ)) : ℚ :=
  dist.fst.1 * dist.fst.2 +
  dist.snd.fst * dist.snd.snd +
  dist.snd.snd.fst * dist.snd.snd.snd +
  dist.fst.snd.snd.fst * dist.fst.snd.snd.snd

-- Declarations of constants based on given problem
def n : ℝ := 200
def a : ℝ := 50
def b : ℝ := 30
def c : ℝ := 40
def d : ℝ := 80
def α : ℝ := 0.010
def χ_squared_critical : ℝ := 6.635

def nAB : ℝ := 80
def nA_notB : ℝ := 30

def C_combinations (n k : ℕ) : ℚ := 
  (Finset.range n).card.choose k

def C_8_3 : ℚ := C_combinations 8 3

-- Proof problem statements
theorem problem1 : chi_squared_test n a b c d α χ_squared_critical := by
  sorry

theorem problem2 : likelihood_ratio nAB nA_notB = (8 / 3) := by
  sorry

theorem problem3 : expected_value (probability_distribution C_combinations C_8_3) = (15 / 8) := by
  sorry

end problem1_problem2_problem3_l25_25822


namespace price_per_liter_l25_25359

theorem price_per_liter (cost : ℕ) (bottles : ℕ) (liters_per_bottle : ℕ) (total_cost : ℕ) (total_liters : ℕ) :
  bottles = 6 → liters_per_bottle = 2 → total_cost = 12 → total_liters = 12 → cost = total_cost / total_liters → cost = 1 :=
by
  intros h_bottles h_liters_per_bottle h_total_cost h_total_liters h_cost_div
  sorry

end price_per_liter_l25_25359


namespace exists_parallelogram_containing_convex_polygon_l25_25357

theorem exists_parallelogram_containing_convex_polygon {P : Type*} [convex P] (h : area P ≤ 1) : 
  ∃ (Q : Type*) [parallelogram Q], area Q = 2 ∧ (∀ x ∈ P, x ∈ Q) :=
sorry

end exists_parallelogram_containing_convex_polygon_l25_25357


namespace distance_from_point_A_to_directrix_C_l25_25125

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25125


namespace exists_infinitely_many_natural_numbers_factors_l25_25354

theorem exists_infinitely_many_natural_numbers_factors (k : ℤ) :
  ∃ (P Q : ℤ[X]), 
    (P.degree = 4 ∧ Q.degree = 4) ∧
    (X^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = P * Q) :=
by
  sorry

end exists_infinitely_many_natural_numbers_factors_l25_25354


namespace truck_has_18_wheels_l25_25760

section TruckWheels

variables (x t : ℕ) (axles_wheels : ℕ → ℕ)

/-- The number of axles on the truck -/
def number_of_axles (t : ℕ) : ℕ := (t - 0.50) / 0.50 + 2

/-- The number of wheels on the truck based on the number of axles -/
def number_of_wheels (x : ℕ) : ℕ := 2 + (x - 1) * 4

theorem truck_has_18_wheels 
    (h1 : t = 2) 
    (h2 : axles_wheels 1 = 2) 
    (h3 : ∀ n > 1, axles_wheels n = 4) :
    number_of_wheels (number_of_axles t) = 18 := 
by
  sorry

end TruckWheels

end truck_has_18_wheels_l25_25760


namespace distance_from_A_to_directrix_l25_25145

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25145


namespace largest_integer_n_neg_quad_expr_l25_25532

theorem largest_integer_n_neg_quad_expr :
  ∃ n : ℤ, n = 6 ∧ ∀ m : ℤ, ((n^2 - 11 * n + 28 < 0) → (m < 7 ∧ m > 4) → m ≤ n) :=
by
  sorry

end largest_integer_n_neg_quad_expr_l25_25532


namespace distance_to_directrix_l25_25034

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25034


namespace probability_of_girl_pairs_nearest_hundredth_l25_25456

noncomputable def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem probability_of_girl_pairs_nearest_hundredth :
  let total_pairings := (factorial 16) / ((factorial 2) ^ 8 * (factorial 8))
  let no_girl_pairs := (factorial 8) * (factorial 8)
  let probability_no_girl_pairs := ((factorial 8)^3 * 2^8) / factorial 16
  let probability_girl_pairs := 1 - probability_no_girl_pairs
  in Float.to IEEE754Float decimal (probability_girl_pairs) = 0.98 :=
by
  sorry

end probability_of_girl_pairs_nearest_hundredth_l25_25456


namespace infinite_natural_numbers_with_factored_polynomial_l25_25327

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l25_25327


namespace distance_from_A_to_directrix_l25_25015

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25015


namespace moving_circle_passes_through_fixed_point_l25_25990

-- Define the parabola x^2 = 12y
def parabola (x y : ℝ) : Prop := x^2 = 12 * y

-- Define the directrix line y = -3
def directrix (y : ℝ) : Prop := y = -3

-- The fixed point we need to show the circle always passes through
def fixed_point : ℝ × ℝ := (0, 3)

-- Define the condition that the moving circle is centered on the parabola and tangent to the directrix
def circle_centered_on_parabola_and_tangent_to_directrix (x y : ℝ) (r : ℝ) : Prop :=
  parabola x y ∧ r = abs (y + 3)

-- Main theorem statement
theorem moving_circle_passes_through_fixed_point :
  (∀ (x y r : ℝ), circle_centered_on_parabola_and_tangent_to_directrix x y r → 
    (∃ (px py : ℝ), (px, py) = fixed_point ∧ (px - x)^2 + (py - y)^2 = r^2)) :=
sorry

end moving_circle_passes_through_fixed_point_l25_25990


namespace bob_speed_lt_40_l25_25873

theorem bob_speed_lt_40 (v_b v_a : ℝ) (h1 : v_a > 45) (h2 : 180 / v_a < 180 / v_b - 0.5) :
  v_b < 40 :=
by
  -- Variables and constants
  let distance := 180
  let min_speed_alice := 45
  -- Conditions
  have h_distance := distance
  have h_min_speed_alice := min_speed_alice
  have h_time_alice := (distance : ℝ) / v_a
  have h_time_bob := (distance : ℝ) / v_b
  -- Given conditions inequalities
  have ineq := h2
  have alice_min_speed := h1
  -- Now apply these facts and derived inequalities to prove bob_speed_lt_40
  sorry

end bob_speed_lt_40_l25_25873


namespace worker_cellphone_surveys_l25_25483

theorem worker_cellphone_surveys 
  (regular_rate : ℕ) 
  (num_surveys : ℕ) 
  (higher_rate : ℕ)
  (total_earnings : ℕ) 
  (earned : ℕ → ℕ → ℕ)
  (higher_earned : ℕ → ℕ → ℕ) 
  (h1 : regular_rate = 10) 
  (h2 : num_surveys = 50) 
  (h3 : higher_rate = 13) 
  (h4 : total_earnings = 605) 
  (h5 : ∀ x, earned regular_rate (num_surveys - x) + higher_earned higher_rate x = total_earnings)
  : (∃ x, x = 35 ∧ earned regular_rate (num_surveys - x) + higher_earned higher_rate x = total_earnings) :=
sorry

end worker_cellphone_surveys_l25_25483


namespace largest_even_digit_multiple_of_nine_l25_25787

theorem largest_even_digit_multiple_of_nine : ∃ n : ℕ, (n < 1000) ∧ (∀ d ∈ digits 10 n, d % 2 = 0) ∧ (n % 9 = 0) ∧ n = 888 := 
by
  sorry

end largest_even_digit_multiple_of_nine_l25_25787


namespace coeff_x2_of_poly_l25_25640

-- Given polynomial expansion (1 + x + root2018(x))^10
def root2018 (x : ℝ) : ℝ := x ^ (1/2018)

noncomputable def poly (x : ℝ) : ℝ := (1 + x + root2018 x) ^ 10

-- Theorem statement: Coefficient of x^2 term is 45
theorem coeff_x2_of_poly : (nat_degree (coeff (poly x) 2) = 45) :=
sorry

end coeff_x2_of_poly_l25_25640


namespace number_solution_exists_l25_25793

theorem number_solution_exists (x : ℝ) (h : 0.80 * x = (4 / 5 * 15) + 20) : x = 40 :=
sorry

end number_solution_exists_l25_25793


namespace collinear_DPQ_l25_25573

variable (ABC : Type) [plane ABC]
variables (A B C D E H P Q : ABC)
variables (AD BE : Line ABC)
variables (r1 r2 : ℝ)
variables (Pcircle : Circle H r1)
variables (Bcircle : Circle B r2)
variables (hAD : AD.is_altitude A C)
variables (hBE : BE.is_altitude B C)
variables (hIntersect : AD ∩ BE = {H})
variables (hHE : dist H E = r1)
variables (hH_tangent : lies_on_tangent P C H)
variables (hP_circle : Pcircle.contains P)
variables (hBE_circle : dist B E = r2)
variables (hB_tangent : lies_on_tangent Q C B)
variables (hQ_circle : Bcircle.contains Q)

theorem collinear_DPQ : collinear {D, P, Q} :=
sorry

end collinear_DPQ_l25_25573


namespace distance_from_A_to_directrix_l25_25147

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25147


namespace mean_median_difference_l25_25542

theorem mean_median_difference :
  let days_missed := [ (0, 4), (1, 2), (2, 5), (3, 3), (4, 2), (5, 4) ] in
  let total_students := 20 in
  let total_days := ( 0 * 4 + 1 * 2 + 2 * 5 + 3 * 3 + 4 * 2 + 5 * 4 ) in
  let mean_days := total_days / total_students in
  let median_days := 2 in
  mean_days - median_days = 9 / 20 :=
by
  let days_missed := [(0, 4), (1, 2), (2, 5), (3, 3), (4, 2), (5, 4)]
  let total_students := 20
  let total_days := (0 * 4 + 1 * 2 + 2 * 5 + 3 * 3 + 4 * 2 + 5 * 4)
  let mean_days := total_days / total_students
  let median_days := 2
  have mean_days_val : mean_days = 49 / 20 := sorry
  have difference := mean_days_val - median_days
  have result : difference = 9 / 20 := sorry
  exact result

end mean_median_difference_l25_25542


namespace exists_distinct_nat_nums_l25_25355

theorem exists_distinct_nat_nums (π : ℝ) (hπ : π = Real.pi) :
  ∃ (k : ℕ) (m : Fin k → ℕ), (∀ i j, i ≠ j → m i ≠ m j) ∧
    (Real.pi ^ (-1984) < 25 - ∑ i, (↑(m i) : ℝ)⁻¹) ∧ (25 - ∑ i, (↑(m i) : ℝ)⁻¹ < Real.pi ^ (-1960)) :=
by sorry

end exists_distinct_nat_nums_l25_25355


namespace distance_from_A_to_directrix_of_parabola_l25_25059

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25059


namespace a2_range_l25_25642

open Nat

noncomputable def a_seq (a : ℕ → ℝ) := ∀ (n : ℕ), n > 0 → (n + 1) * a n ≥ n * a (2 * n)

theorem a2_range (a : ℕ → ℝ) 
  (h1 : ∀ (n : ℕ), n > 0 → (n + 1) * a n ≥ n * a (2 * n)) 
  (h2 : ∀ (m n : ℕ), m < n → a m ≤ a n) 
  (h3 : a 1 = 2) :
  (2 < a 2) ∧ (a 2 ≤ 4) :=
sorry

end a2_range_l25_25642


namespace chloe_total_score_l25_25645

theorem chloe_total_score :
  let first_level_treasure_points := 9
  let first_level_bonus_points := 15
  let first_level_treasures := 6
  let second_level_treasure_points := 11
  let second_level_bonus_points := 20
  let second_level_treasures := 3

  let first_level_score := first_level_treasures * first_level_treasure_points + first_level_bonus_points
  let second_level_score := second_level_treasures * second_level_treasure_points + second_level_bonus_points

  first_level_score + second_level_score = 122 :=
by
  sorry

end chloe_total_score_l25_25645


namespace flowers_left_l25_25489

theorem flowers_left (flowers_picked_A : Nat) (flowers_picked_M : Nat) (flowers_given : Nat)
  (h_a : flowers_picked_A = 16)
  (h_m : flowers_picked_M = 16)
  (h_g : flowers_given = 18) :
  flowers_picked_A + flowers_picked_M - flowers_given = 14 :=
by
  sorry

end flowers_left_l25_25489


namespace poly_factorable_l25_25301

-- Define the specific form of N.
noncomputable def N (k : ℤ) : ℕ := 4 * k^4 - 8 * k^2 + 2

-- Define the polynomial.
noncomputable def poly (N : ℕ) : ℤ[X] := X^8 + N * X^4 + 1

-- Define the fourth-degree polynomials used in the factorization.
noncomputable def poly1 (k : ℤ) : ℤ[X] :=
  X^4 - 2 * k * X^3 + 2 * k^2 * X^2 - 2 * k * X + 1

noncomputable def poly2 (k : ℤ) : ℤ[X] := 
  X^4 + 2 * k * X^3 + 2 * k^2 * X^2 + 2 * k * X + 1

-- The proof statement: prove that the polynomial can be factored into the specified fourth-degree polynomials.
theorem poly_factorable : ∀ k : ℤ, ∃ p1 p2 : ℤ[X], poly (N k) = p1 * p2 :=
by
  intro k
  use (poly1 k)
  use (poly2 k)
  sorry

end poly_factorable_l25_25301


namespace high_sulfur_oil_samples_l25_25226

/-- The number of high-sulfur oil samples in a container with the given conditions. -/
theorem high_sulfur_oil_samples (total_samples : ℕ) 
    (heavy_oil_freq : ℚ) (light_low_sulfur_freq : ℚ)
    (no_heavy_low_sulfur: true) (almost_full : total_samples = 198)
    (heavy_oil_freq_value : heavy_oil_freq = 1 / 9)
    (light_low_sulfur_freq_value : light_low_sulfur_freq = 11 / 18) :
    (22 + 68) = 90 := 
by
  sorry

end high_sulfur_oil_samples_l25_25226


namespace edward_earnings_l25_25894

theorem edward_earnings
    (total_lawns : ℕ := 17)
    (forgotten_lawns : ℕ := 9)
    (total_earnings : ℕ := 32) :
    (total_earnings / (total_lawns - forgotten_lawns) = 4) :=
by
  sorry

end edward_earnings_l25_25894


namespace distance_from_A_to_directrix_l25_25005

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25005


namespace largest_integer_n_neg_quad_expr_l25_25533

theorem largest_integer_n_neg_quad_expr :
  ∃ n : ℤ, n = 6 ∧ ∀ m : ℤ, ((n^2 - 11 * n + 28 < 0) → (m < 7 ∧ m > 4) → m ≤ n) :=
by
  sorry

end largest_integer_n_neg_quad_expr_l25_25533


namespace max_cos_sum_l25_25853

theorem max_cos_sum (A B C : ℝ) (h : A + B + C = π) :
  cos A + cos B * cos C ≤ 1 :=
sorry

end max_cos_sum_l25_25853


namespace measure_angle5_l25_25269

-- Definitions from the conditions
variables (m n : Line)
variables (angle1 angle2 angle5 : ℝ)
variables (parallel : m ∥ n)
variables (h1 : angle1 = (1 / 4) * angle2)

-- Proof problem
theorem measure_angle5 :
  let x := angle1 in
  let angle2 := 4 * x in
  let angle5 := x in
  angle2 + angle5 = 180 → angle5 = 36 := 
by
  sorry

end measure_angle5_l25_25269


namespace vector_computation_l25_25548

   def vec_a : (ℝ × ℝ) := (1, 1)
   def vec_b : (ℝ × ℝ) := (1, -1)

   theorem vector_computation : 
     (1 / 2) • vec_a - (3 / 2) • vec_b = (-1 : ℝ, 2 : ℝ) :=
   by
     sorry
   
end vector_computation_l25_25548


namespace find_m_l25_25964

-- Define the circle and parabola conditions
theorem find_m : 
  ∃ (m : ℕ), 
    (∀ (x y : ℝ), (x + 1) ^ 2 + y ^ 2 = 4 → y ^ 2 = m * x) ∧ 
    m ≠ 0 ∧ 
    (∀ (d : ℝ), d = | (m / 4) - 1 | → 2 * sqrt (4 - d ^ 2) = 2 * sqrt 3) → 
  m = 8 :=
sorry

end find_m_l25_25964


namespace distance_from_A_to_directrix_of_parabola_l25_25064

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25064


namespace number_of_subsets_A_intersect_B_is_31_l25_25224

def is_narcissistic (n : ℕ) : Prop := 
  let digits := List.ofDigits (Int.toDigits n)
  digits.foldl (λ sum d => sum + d ^ List.length digits) 0 = n

def set_A : Set ℕ := {n | n > 0 ∧ n < 10 ∧ is_narcissistic n}

def satisfies_inequality (x : ℕ) : Prop :=
  (x^2 - 5*x - 3) / (2 - x) < 1

def set_B : Set ℕ := {x | satisfies_inequality x}

def intersect (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∈ B}

def number_of_non_empty_subsets (s : Set ℕ) : ℕ :=
  2^s.size - 1

theorem number_of_subsets_A_intersect_B_is_31 :
  number_of_non_empty_subsets (intersect set_A set_B) = 31 :=
sorry

end number_of_subsets_A_intersect_B_is_31_l25_25224


namespace distance_from_A_to_directrix_l25_25014

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25014


namespace range_of_a_l25_25552

theorem range_of_a (p q : Prop) (a : ℝ) :
  (p ↔ (2 * x^2 - 3 * x + 1 ≤ 0)) →
  (q ↔ (x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0)) →
  (¬p ↔ ¬q → (0 ≤ a ∧ a ≤ 0.5)) :=
by
  intros hp hq hnp_nq
  sorry

end range_of_a_l25_25552


namespace number_of_participants_l25_25434

-- Define the number of participants and games conditions
def num_games (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem number_of_participants (h : num_games 15 = 105) : ∃ n : ℕ, num_games n = 105 ∧ n = 15 :=
by
  use 15
  split
  · exact h
  · rfl

end number_of_participants_l25_25434


namespace distance_from_A_to_directrix_l25_25075

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25075


namespace infinite_natural_numbers_factorable_polynomial_l25_25338

theorem infinite_natural_numbers_factorable_polynomial :
  ∃ᶠ N in at_top, ∃ (k : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧
    ∃ (f g : ℤ[x]), (degree f = 4) ∧ (degree g = 4) ∧ (f * g = X^8 + C (N : ℤ) * X^4 + 1) :=
sorry

end infinite_natural_numbers_factorable_polynomial_l25_25338


namespace find_angle_C_find_perimeter_range_l25_25648

variable (A B C a b c : ℝ)
variable (ABC_is_triangle : ∀ (A B C : ℝ), 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π → A + B + C = π)
variable (side_opposite_A : a = sin A)
variable (side_opposite_B : b = sin B)
variable (side_opposite_C : c = sin C)

-- condition for part I
variable (cos_condition : 2 * cos C * (a * cos B + b * cos A) = c)

-- prove C = π / 3
theorem find_angle_C : C = π / 3 := by
  sorry

-- additional condition for part II
variable (c_is_sqrt3 : c = sqrt 3)

-- prove range of perimeter
theorem find_perimeter_range : 2 * sqrt 3 < a + b + c ∧ a + b + c ≤ 3 * sqrt 3 := by
  sorry

end find_angle_C_find_perimeter_range_l25_25648


namespace exists_infinite_N_l25_25339

theorem exists_infinite_N (k : ℤ) : ∃ᶠ N : ℕ in at_top, 
  ∃ (A B : Polynomial ℤ), 
    (Polynomial.natDegree A = 4 ∧ Polynomial.natDegree B = 4) ∧
    (Polynomial.mul A B = Polynomial.X^8 + Polynomial.C N * Polynomial.X^4 + 1) :=
begin
  -- Proof goes here
  sorry
end

end exists_infinite_N_l25_25339


namespace distance_to_directrix_l25_25086

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25086


namespace fraction_given_to_friend_l25_25694

theorem fraction_given_to_friend (s u r g k : ℕ) 
  (h1: s = 135) 
  (h2: u = s / 3) 
  (h3: r = s - u) 
  (h4: k = 54) 
  (h5: g = r - k) :
  g / r = 2 / 5 := 
  by
  sorry

end fraction_given_to_friend_l25_25694


namespace distance_to_directrix_l25_25026

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25026


namespace distance_from_point_A_to_directrix_C_l25_25123

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25123


namespace wickets_before_last_match_l25_25458

theorem wickets_before_last_match 
    (initial_average : ℝ)
    (additional_wickets : ℕ)
    (additional_runs : ℝ)
    (new_average : ℝ)
    (wickets_before : ℕ)
    (runs_before : ℝ)
    (runs_after : ℝ) :
    initial_average = 12.4 →
    additional_wickets = 3 →
    additional_runs = 26 →
    new_average = 12.0 →
    runs_before = initial_average * wickets_before →
    runs_after = runs_before + additional_runs →
    wickets_before = 25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end wickets_before_last_match_l25_25458


namespace distance_from_A_to_directrix_l25_25011

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25011


namespace find_phi_l25_25954

noncomputable def f (x φ : ℝ) : ℝ := 2 * cos ((π / 3) * x + φ)

theorem find_phi (φ : ℝ) (h1 : φ ∈ set.Ioo 0 π)
  (h2 : ∃ k : ℤ, f 2 φ = 0) : φ = 5 * π / 6 :=
by
  sorry

end find_phi_l25_25954


namespace sum_of_digits_grows_unbounded_l25_25707

theorem sum_of_digits_grows_unbounded (a : ℕ) (h_even : a % 2 = 0) (h_not_multiple_of_5 : a % 5 ≠ 0) :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ sum_of_digits (a ^ m) > m := 
sorry

end sum_of_digits_grows_unbounded_l25_25707


namespace infinite_natural_numbers_with_factored_polynomial_l25_25329

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l25_25329


namespace right_angled_triangle_l25_25664

noncomputable def midpoint (x y : Point) : Point :=
  ⟨(x.1 + y.1) / 2, (x.2 + y.2) / 2⟩

noncomputable def symmetric_point (P D : Point) : Point :=
  ⟨2 * D.1 - P.1, 2 * D.2 - P.2⟩

structure Triangle :=
  (A B C : Point)
  (h : A.1 < C.1)

structure Altitude (A : Point) (L : Line) :=
  (foot : Point)
  (perpendicular : is_perpendicular L (line_through A foot))

structure Midpoint (P Q : Point) :=
  (M : Point)
  (midpoint_property : M = midpoint P Q)

structure PerpendicularBisectorResult (B' AC : Line) :=
  (P : Point)
  (intersection_property : intersection B' AC = P)

theorem right_angled_triangle
  (ABC : Triangle)
  (D : Altitude ABC.A ⟨ABC.B, ABC.C⟩)
  (M : Midpoint ABC.B ABC.C)
  (B' : symmetric_point ABC.B D.foot = B')
  (P : PerpendicularBisectorResult ⟨B', D.foot⟩ ⟨ABC.C, ABC.A⟩)
  (h : is_perpendicular (line_through ABC.B P.P) (line_through ABC.A M.M)) :
  angle ABC.B ABC.A ABC.C = 90 :=
  sorry

end right_angled_triangle_l25_25664


namespace vector_magnitude_l25_25955

variables (a b : ℝ^3) 

-- Given conditions
noncomputable def angle_ab := 3 * Real.pi / 4
axiom mag_a : ∥a∥ = Real.sqrt 2
axiom mag_b : ∥b∥ = 3

-- The problem statement
theorem vector_magnitude (h : Real.angle a b = angle_ab) : ∥a + 2 • b∥ = Real.sqrt 26 :=
sorry

end vector_magnitude_l25_25955


namespace distance_from_A_to_directrix_of_parabola_l25_25062

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25062


namespace placing_books_l25_25699

theorem placing_books (original_books additional_books : ℕ) (ways_to_place_first ways_to_place_second : ℕ) :
    original_books = 5 →
    additional_books = 2 →
    ways_to_place_first = 6 →
    ways_to_place_second = 7 →
    (ways_to_place_first * ways_to_place_second) = 42 :=
by
  intros orig_books_eq add_books_eq place_first_eq place_second_eq
  rw [orig_books_eq, add_books_eq, place_first_eq, place_second_eq]
  exact eq.refl 42

end placing_books_l25_25699


namespace smallest_value_l25_25192

theorem smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∀ y ∈ {x, x^2, 2*x, 3*x, x^3}, x^3 ≤ y :=
by {
  sorry
}

end smallest_value_l25_25192


namespace original_price_of_RAM_l25_25823

variables (P : ℝ)

-- Conditions extracted from the problem statement
def priceAfterFire (P : ℝ) : ℝ := 1.30 * P
def priceAfterDecrease (P : ℝ) : ℝ := 1.04 * P

-- The given current price
axiom current_price : priceAfterDecrease P = 52

-- Theorem to prove the original price P
theorem original_price_of_RAM : P = 50 :=
sorry

end original_price_of_RAM_l25_25823


namespace exists_infinitely_many_N_l25_25291

theorem exists_infinitely_many_N (k : ℤ) : ∃ N : ℕ, ∃ A B : polynomial ℤ, 
  N = 4 * k^4 - 8 * k^2 + 2 ∧ polynomial.nat_degree A = 4 ∧ polynomial.nat_degree B = 4 ∧
  (polynomial.X^8 + polynomial.C (N:ℤ) * polynomial.X^4 + 1) = A * B :=
sorry

end exists_infinitely_many_N_l25_25291


namespace math_problem_l25_25940

variables {f : ℝ → ℝ}

-- Premises
def is_odd (f: ℝ → ℝ) := ∀ x, f(-x) = -f(x)
def is_decreasing_on (f: ℝ → ℝ) (a b: ℝ) := ∀ x y, a ≤ x → x < y → y ≤ b → f(x) ≥ f(y)

noncomputable def satisfies_inequality (f: ℝ → ℝ) (m: ℝ) := 
  is_odd f ∧ (∀ x, -2 ≤ x ∧ x ≤ 2) ∧ is_decreasing_on f (-2) 0 ∧ -1 ≤ m ∧ m < 1 →
  f(1 - m) + f(1 - m^2) < 0

-- Theorem to be proved
theorem math_problem : ∀ f: ℝ → ℝ, 
  (is_odd f) →
  (∀ x, -2 ≤ x ∧ x ≤ 2) →
  (is_decreasing_on f (-2) 0) →
  (∀ m, satisfies_inequality f m)
  :=
by
  sorry

end math_problem_l25_25940


namespace _l25_25142

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25142


namespace d_positive_neither_necessary_nor_sufficient_l25_25256

-- Definition of an arithmetic sequence and sum of its first n terms
def arithmetic_sequence (a₁ : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a₁ (n + 1) = a₁ n + d

def sum_of_arithmetic_sequence (a₁ : ℕ → ℝ) (n : ℕ) :=
  n * a₁ 0 + (n * (n - 1) / 2) * (a₁ 1 - a₁ 0)

-- Statement of the problem
theorem d_positive_neither_necessary_nor_sufficient
  (a₁ : ℕ → ℝ) (d : ℝ) (Sₙ Sₙ₊₁ : ℕ → ℝ)
  (h_seq : arithmetic_sequence a₁ d)
  (h_Sₙ : ∀ n, Sₙ n = sum_of_arithmetic_sequence a₁ n)
  (h_Sₙ₊₁ : ∀ n, Sₙ₊₁ n = sum_of_arithmetic_sequence a₁ (n + 1)):
  ¬ ((∀ n, Sₙ₊₁ n > Sₙ n → d > 0) ∧ (d > 0 → ∀ n, Sₙ₊₁ n > Sₙ n)) :=
sorry

end d_positive_neither_necessary_nor_sufficient_l25_25256


namespace problem_π_digit_sequence_l25_25689

def f (n : ℕ) : ℕ :=
  match n with
  | 1  => 1
  | 2  => 4
  | 3  => 1
  | 4  => 5
  | 5  => 9
  | 6  => 2
  | 7  => 6
  | 8  => 5
  | 9  => 3
  | 10 => 5
  | _  => 0  -- for simplicity we define other cases arbitrarily

theorem problem_π_digit_sequence :
  ∃ n : ℕ, n > 0 ∧ f (f (f (f (f 10)))) = 1 := by
  sorry

end problem_π_digit_sequence_l25_25689


namespace find_distance_l25_25761

open Real

-- Definitions of vectors
variables {v1 v2 : ℝ^3}

noncomputable def v1 := (1/3/√2, 4/3/√2, 1/3/√2)
noncomputable def v2 := (1/√2, 0, -1/√2)

-- Conditions
def is_unit_vector (v : ℝ^3) : Prop :=
  v.1^2 + v.2^2 + v.3^2 = 1

def angle_45 (v : ℝ^3) : Prop :=
  ((2 * v.1 + 2 * v.2 - v.3) / (sqrt (2^2 + 2^2 + (-1)^2))) = 1 / sqrt 2

def angle_60 (v : ℝ^3) : Prop :=
  ((v.2 - v.3) / (sqrt (0^2 + 1^2 + (-1)^2))) = 1 / 2

-- Theorem statement
theorem find_distance (hv1 : is_unit_vector v1)
                      (hv2 : is_unit_vector v2)
                      (ha45v1 : angle_45 v1)
                      (ha60v1 : angle_60 v1)
                      (ha45v2 : angle_45 v2)
                      (ha60v2 : angle_60 v2) :
  ‖v1 - v2‖ = sqrt 2 := 
sorry

end find_distance_l25_25761


namespace percentage_less_than_m_plus_f_l25_25819

variable {α : Type} [MeasurableSpace α] [ProbabilityMeasure α]

def symmetric_distribution (m : ℝ) : Prop :=
∀ x, P(x < m) = 0.5

def within_one_std_dev (m f : ℝ) : Prop :=
P(m - f < X ∧ X < m + f) = 0.68

theorem percentage_less_than_m_plus_f (m f : ℝ) (h1 : symmetric_distribution m) (h2 : within_one_std_dev m f) :
  P(X < m + f) = 0.84 :=
sorry

end percentage_less_than_m_plus_f_l25_25819


namespace simon_age_in_2010_l25_25239

theorem simon_age_in_2010 :
  ∀ (s j : ℕ), (j = 16 → (j + 24 = s) → j + (2010 - 2005) + 24 = 45) :=
by 
  intros s j h1 h2 
  sorry

end simon_age_in_2010_l25_25239


namespace area_of_triangle_BDF_l25_25934

-- Define the hexagon ABCDEF and its properties
variables (A B C D E F O : Type)
variables [RealVectorSpace A] [affine_space A A]
variables {α β : A}
variables (r : ℝ) -- radius of the circumscribed circle

-- Concyclic points
axiom ABCDEF_concyclic : ∃ (O : A) (r : ℝ), ∀ (P : A), P ∈ {A, B, C, D, E, F} → dist O P = r

-- Hexagon ABCDEF with equal side lengths and given area
axiom hexagon_area_60 : area (polygon [A, B, C, D, E, F]) = 60
axiom sides_equal : dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E F ∧ dist E F = dist F A

-- Goal: Prove the area of triangle BDF is 30
theorem area_of_triangle_BDF : area (triangle B D F) = 30 :=
sorry

end area_of_triangle_BDF_l25_25934


namespace sum_of_cool_triangle_areas_eq_42_l25_25510

-- Define condition for "cool" right triangle
def is_cool_right_triangle (a b : ℕ) : Prop :=
  a * b / 2 = 3 * (a + b)

-- Define function to compute the area
def area (a b : ℕ) : ℕ :=
  a * b / 2

-- Define theorem to assert the sum of distinct possible areas
theorem sum_of_cool_triangle_areas_eq_42 :
  ∑ (a b : ℕ) in { (a, b) | is_cool_right_triangle a b }.to_finset, area a b = 42 :=
sorry

end sum_of_cool_triangle_areas_eq_42_l25_25510


namespace distance_from_A_to_directrix_of_parabola_l25_25057

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25057


namespace distance_from_A_to_directrix_l25_25081

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25081


namespace compare_minus_abs_val_l25_25882

theorem compare_minus_abs_val :
  -|(-8)| < -6 := 
sorry

end compare_minus_abs_val_l25_25882


namespace incorrect_statement_D_l25_25429

def ordinate_of_x_axis_is_zero (p : ℝ × ℝ) : Prop :=
  p.2 = 0

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_A_properties (a b : ℝ) : Prop :=
  let x := - a^2 - 1
  let y := abs b
  x < 0 ∧ y ≥ 0

theorem incorrect_statement_D (a b : ℝ) : 
  ∃ (x y : ℝ), point_A_properties a b ∧ (x = -a^2 - 1 ∧ y = abs b ∧ (x < 0 ∧ y = 0)) :=
by {
  sorry
}

end incorrect_statement_D_l25_25429


namespace eq_ratio_of_amplitudes_l25_25366

/-
Conditions:
- M is the Richter scale defined as M = log (A / A0).
- On August 6, 2023, at 2:33 a.m., a magnitude 5.5 earthquake occurred in Pingyuan County.
- Twenty-nine minutes later, a magnitude 3.0 earthquake occurred.
- A5.5 and A3.0 represent the maximum amplitudes of seismic waves for earthquakes of magnitude 5.5 and 3.0, respectively.
- Reference data: sqrt(10) ≈ 3.16.

To prove:
- The ratio A5.5 / A3.0 = 316.
-/
theorem eq_ratio_of_amplitudes (A0 A5_5 A3_0 : ℝ)
  (h5_5 : 5.5 = log10 (A5_5 / A0))
  (h3_0 : 3.0 = log10 (A3_0 / A0))
  (h_sqrt10 : real.sqrt 10 = 3.16): 
  A5_5 / A3_0 = 316 := 
sorry

end eq_ratio_of_amplitudes_l25_25366


namespace infinite_natural_numbers_with_factored_polynomial_l25_25325

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l25_25325


namespace distance_from_A_to_directrix_l25_25017

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25017


namespace union_of_A_and_B_l25_25198

open Set

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {3, 5, 7, 8}

theorem union_of_A_and_B : A ∪ B = {3, 4, 5, 6, 7, 8} := by
  sorry

end union_of_A_and_B_l25_25198


namespace distance_from_A_to_directrix_on_parabola_l25_25051

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25051


namespace cevian_concurrent_product_l25_25230

variables (A B C A' B' C' O : Type)

-- Definitions of necessary mathematical constructs
def triangle (A B C : Type) : Prop := sorry
def on_side (A' : Type) (BC : Type) : Prop := sorry
def concurrent (AA' BB' CC' : Type) (O : Type) : Prop := sorry

-- Defining the variables and conditions required
variable (AO OA' BO OB' CO OC' : ℝ)

-- Condition from the problem statement
axiom h_concurrent : concurrent AO BO CO O := sorry
axiom h_sum : AO/OA' + BO/OB' + CO/OC' = 92

-- The theorem that needs to be proven
theorem cevian_concurrent_product :
  (AO / OA') * (BO / OB') * (CO / OC') = 2 :=
sorry

end cevian_concurrent_product_l25_25230


namespace power_mod_eight_l25_25790

theorem power_mod_eight (n : ℕ) : (3^101 + 5) % 8 = 0 :=
by
  sorry

end power_mod_eight_l25_25790


namespace tan_func_period_l25_25862

-- Define the function as y = tan(x + π/3)
def tan_func (x : ℝ) : ℝ := Real.tan (x + Real.pi / 3)

-- The goal is to prove that the smallest positive period is π
theorem tan_func_period : ∀ x, tan_func (x + Real.pi) = tan_func x :=
by sorry

end tan_func_period_l25_25862


namespace distance_from_A_to_directrix_l25_25010

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25010


namespace infinite_natural_numbers_factorable_polynomial_l25_25331

theorem infinite_natural_numbers_factorable_polynomial :
  ∃ᶠ N in at_top, ∃ (k : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧
    ∃ (f g : ℤ[x]), (degree f = 4) ∧ (degree g = 4) ∧ (f * g = X^8 + C (N : ℤ) * X^4 + 1) :=
sorry

end infinite_natural_numbers_factorable_polynomial_l25_25331


namespace minimum_f_value_f_at_1_l25_25904

def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1 / (x^2 + 1/x^2)

theorem minimum_f_value (x : ℝ) (hx : x > 0) : f(x) ≥ 2.5 :=
begin
  sorry
end

theorem f_at_1 : f(1) = 2.5 :=
begin
  sorry
end

end minimum_f_value_f_at_1_l25_25904


namespace speed_of_dad_first_half_l25_25654

noncomputable def total_distance_jake_bike := (11 : ℝ) * 2

noncomputable def distance_second_half := (60 : ℝ) * (0.5 / 2)

noncomputable def total_distance := 11 * 2 -- Jake's biking speed and time

noncomputable def first_half_distance := total_distance - distance_second_half

noncomputable def first_half_time := (30.0 / 2) / 60

noncomputable def speed_first_half := first_half_distance / first_half_time

theorem speed_of_dad_first_half : speed_first_half = 28 :=
by
  -- sorries will be replaced by the actual proof steps, omitted for now
  sorry

end speed_of_dad_first_half_l25_25654


namespace max_value_of_function_neg_x_l25_25384

theorem max_value_of_function_neg_x (x : ℝ) (h : x < 0) : 
  ∃ y, (y = 2 * x + 2 / x) ∧ y ≤ -4 := sorry

end max_value_of_function_neg_x_l25_25384


namespace Gyeongyeon_cookies_l25_25181

def initial_cookies : ℕ := 20
def cookies_given : ℕ := 7
def cookies_received : ℕ := 5

def final_cookies (initial : ℕ) (given : ℕ) (received : ℕ) : ℕ :=
  initial - given + received

theorem Gyeongyeon_cookies :
  final_cookies initial_cookies cookies_given cookies_received = 18 :=
by
  sorry

end Gyeongyeon_cookies_l25_25181


namespace rabbits_in_cage_l25_25644

theorem rabbits_in_cage (heads legs : ℝ) (total_heads : heads = 40) 
  (condition : legs = 8 + 10 * (2 * (heads - rabbits))) :
  ∃ rabbits : ℝ, rabbits = 33 :=
by
  sorry

end rabbits_in_cage_l25_25644


namespace distance_from_A_to_directrix_l25_25156

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25156


namespace compare_abc_l25_25546

-- Define a, b, and c based on the given logarithmic expressions
def a : ℝ := 5^(Real.log 3.4 / Real.log 2)
def b : ℝ := 5^(Real.log (10/3) / Real.log 3)
def c : ℝ := 5^(Real.log (10/3) / Real.log 2)

-- The theorem stating that a > c > b
theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l25_25546


namespace y_val_l25_25985

noncomputable def y (p : ℝ) : ℝ :=
  if h : log p y * log 3 p = 4 then 81 else 0

theorem y_val (p : ℝ) (hp : log p y * log 3 p = 4) : y p = 81 :=
by
  rw [y, if_pos hp]
  sorry

end y_val_l25_25985


namespace hallie_net_earnings_correct_l25_25977

noncomputable def hallieNetEarnings : ℚ :=
  let monday_hours := 7
  let monday_rate := 10
  let monday_tips := 18
  let tuesday_hours := 5
  let tuesday_rate := 12
  let tuesday_tips := 12
  let wednesday_hours := 7
  let wednesday_rate := 10
  let wednesday_tips := 20
  let thursday_hours := 8
  let thursday_rate := 11
  let thursday_tips := 25
  let thursday_discount := 0.10
  let friday_hours := 6
  let friday_rate := 9
  let friday_tips := 15
  let income_tax := 0.05

  let monday_earnings := monday_hours * monday_rate
  let tuesday_earnings := tuesday_hours * tuesday_rate
  let wednesday_earnings := wednesday_hours * wednesday_rate
  let thursday_earnings := thursday_hours * thursday_rate
  let thursday_earnings_after_discount := thursday_earnings * (1 - thursday_discount)
  let friday_earnings := friday_hours * friday_rate

  let total_hourly_earnings := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings + friday_earnings
  let total_tips := monday_tips + tuesday_tips + wednesday_tips + thursday_tips + friday_tips

  let total_tax := total_hourly_earnings * income_tax
  
  let net_earnings := (total_hourly_earnings - total_tax) - (thursday_earnings - thursday_earnings_after_discount) + total_tips
  net_earnings

theorem hallie_net_earnings_correct : hallieNetEarnings = 406.10 := by
  sorry

end hallie_net_earnings_correct_l25_25977


namespace gumball_draw_probability_l25_25800

def prob_blue := 2 / 3
def prob_two_blue := (16 / 36)
def prob_pink := 1 - prob_blue

theorem gumball_draw_probability
    (h1 : prob_two_blue = prob_blue * prob_blue)
    (h2 : prob_blue + prob_pink = 1) :
    prob_pink = 1 / 3 := 
by
  sorry

end gumball_draw_probability_l25_25800


namespace probability_X_less_than_0_l25_25267

-- Definitions for the problem
def X (σ : ℝ) : MeasureTheory.ProbabilityMeasure ℝ := MeasureTheory.ProbabilityMeasure.normal 2 σ

-- Prove the main statement
theorem probability_X_less_than_0 (σ : ℝ) (h : MeasureTheory.ProbabilityMeasure.prob (X σ) {x | 0 < x ∧ x < 4} = 0.3) :
    MeasureTheory.ProbabilityMeasure.prob (X σ) {x | x < 0} = 0.35 :=
by
  sorry

end probability_X_less_than_0_l25_25267


namespace original_weight_of_beef_l25_25473

variable (W : ℝ)

def first_stage_weight := 0.80 * W
def second_stage_weight := 0.70 * (first_stage_weight W)
def third_stage_weight := 0.75 * (second_stage_weight W)

theorem original_weight_of_beef :
  third_stage_weight W = 392 → W = 933.33 :=
by
  intro h
  sorry

end original_weight_of_beef_l25_25473


namespace incorrect_option_d_l25_25606

variable (a b : ℝ)

theorem incorrect_option_d (h1 : 0 < a) (h2 : a < b) (h3 : b < 1):
  (a^3 < b^2) ∧ (2^a < 3^b) ∧ (log 2 a < log 3 b) ∧ ¬ (log a 2 < log b 3) :=
by sorry

end incorrect_option_d_l25_25606


namespace function_in_fourth_quadrant_l25_25972

theorem function_in_fourth_quadrant (t : ℝ) (ht : t > 0) :
  let w := - (t ^ (3/2)) - (2 / t) in w < 0 :=
by
  -- Assume conditions and initial setup
  have h1 : t > 0 := ht,
  -- Define the function
  let w := - (t ^ (3/2)) - (2 / t),
  -- Provide the proof for the main statement
  sorry

end function_in_fourth_quadrant_l25_25972


namespace area_of_square_is_26_l25_25417

structure Point where
  x : ℤ
  y : ℤ

def P : Point := ⟨1, 2⟩
def Q : Point := ⟨-4, 3⟩
def R : Point := ⟨-3, 8⟩
def S : Point := ⟨2, 7⟩

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

theorem area_of_square_is_26 :
  distance P Q ^ 2 = 26 := by
  sorry

end area_of_square_is_26_l25_25417


namespace find_length_of_BX_l25_25680

structure Rectangle (A B C D : Type) :=
  (AB : ℝ) (BC : ℝ)

def WXYZ_is_rectangle (W X Y Z : Type) := true

theorem find_length_of_BX (A B C D W X Y Z : Type) 
  (H1 : Rectangle A B C D) 
  (H2 : H1.AB = 3) 
  (H3 : H1.BC = 7) 
  (H4 : AW = 1) 
  (H5 : WXYZ_is_rectangle W X Y Z) 
  (H6 : BX < XC) : 
  BX = (7 - real.sqrt 41) / 2 :=
sorry

end find_length_of_BX_l25_25680


namespace infinite_natural_numbers_factorable_polynomial_l25_25334

theorem infinite_natural_numbers_factorable_polynomial :
  ∃ᶠ N in at_top, ∃ (k : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧
    ∃ (f g : ℤ[x]), (degree f = 4) ∧ (degree g = 4) ∧ (f * g = X^8 + C (N : ℤ) * X^4 + 1) :=
sorry

end infinite_natural_numbers_factorable_polynomial_l25_25334


namespace regular_polygon_sides_l25_25997

theorem regular_polygon_sides (n : ℕ) (h1 : ∑ i in finset.range(n), 180 * (n - 2) = 2610 + (180 * (n - 2)) / n) : n = 17 :=
by sorry

end regular_polygon_sides_l25_25997


namespace pos_diff_triangles_quadrilaterals_l25_25813

theorem pos_diff_triangles_quadrilaterals (n : ℕ) (h : n = 6) :
  let triangles := Nat.choose n 3,
      quadrilaterals := Nat.choose n 4
  in triangles - quadrilaterals = 5 :=
by
  rw h
  let triangles := Nat.choose 6 3
  let quadrilaterals := Nat.choose 6 4
  have : triangles = 20 := by sorry
  have : quadrilaterals = 15 := by sorry
  exact Nat.sub_eq_of_eq_add (show 20 = 15 + 5 from by sorry)

end pos_diff_triangles_quadrilaterals_l25_25813


namespace max_cos_sum_l25_25858

theorem max_cos_sum (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 0 < A) (h3 : A < Real.pi)
  (h4 : 0 < B) (h5 : B < Real.pi) (h6 : 0 < C) (h7 : C < Real.pi) :
  ∃ max_val : ℝ, max_val = (3 / 4) ∧ (∀ (A B C : ℝ), A + B + C = Real.pi → (0 < A) → (A < Real.pi) →
  (0 < B) → (B < Real.pi) → (0 < C) → (C < Real.pi) → (cos A + cos B * cos C ≤ max_val)) :=
sorry

end max_cos_sum_l25_25858


namespace not_product_consecutive_l25_25286

theorem not_product_consecutive (n k : ℕ) (hn : n > 0) (hk : k > 0) :
  ¬ ∃ (p : ℕ) (x : ℕ), p ≥ 2 ∧ (f n k = x * (x + 1) * ... * (x + (p - 1))) :=
by
  let f := λ n k : ℕ, 2 * n^(3 * k) + 4 * n^k + 10
  sorry

end not_product_consecutive_l25_25286


namespace transformation_result_l25_25719

def f (x : ℝ) : ℝ := sin (x - π / 3)

theorem transformation_result :
  (y = f (x / 2) → y = f((x - π / 3) / 2)) ↔ y = -cos (x / 2) :=
sorry

end transformation_result_l25_25719


namespace Simon_age_2010_l25_25241

variable (Jorge Simon : ℕ)

axiom age_difference : Jorge + 24 = Simon
axiom Jorge_age_2005 : Jorge = 16

theorem Simon_age_2010 : Simon = 45 := by
  have Simon_age_2005 : Simon = Jorge + 24 := age_difference
  rw [Jorge_age_2005] at Simon_age_2005
  have Simon_age_2005 : Simon = 16 + 24 := Simon_age_2005
  have Simon_age_2005 : Simon = 40 := by norm_num
  have Simon_age_2010 : Simon = 40 + 5 := by
    rw Simon_age_2005
    norm_num
  exact by norm_num at Simon_age_2010

end Simon_age_2010_l25_25241


namespace find_angle_C_l25_25624

noncomputable def angle_C_condition (A B C : Triangle) (angleB : ℝ) (ac : ℝ) (bc : ℝ)
  (h1 : angleB = 45)
  (h2 : ac = √2 * bc) : Prop :=
triangle.angle B  = 45
∧ triangle.side A = √2 * bc

theorem find_angle_C (A B C : Triangle) (angleB : ℝ) (ac : ℝ) (bc : ℝ)
  (h1 : angleB = 45)
  (h2 : ac = √2 * bc) : (triangle.angle C = 105) :=
by
sor

end find_angle_C_l25_25624


namespace distance_from_A_to_directrix_l25_25016

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25016


namespace area_of_triangle_l25_25968

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

def pointP : ℝ × ℝ := (2, f 2)

noncomputable def tangent_line (x : ℝ) : ℝ := f 2

def vertex1 : ℝ × ℝ := (1, 1)
def vertex2 : ℝ × ℝ := (1, 3)
def vertex3 : ℝ × ℝ := (3, 3)

theorem area_of_triangle : 
  let x1 := vertex1.fst,
      y1 := vertex1.snd,
      x2 := vertex2.fst,
      y2 := vertex2.snd,
      x3 := vertex3.fst,
      y3 := vertex3.snd in
  (1 / 2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))) = 2 := 
by 
  sorry

end area_of_triangle_l25_25968


namespace sum_of_squares_is_122_l25_25710

open Real

noncomputable def sum_of_squares (x y : ℝ) : ℝ :=
  if (xy = 15 ∧ (2 * x - y)^4 = 1) then
    let y_values := {y | ∃ x, xy = 15 ∧ (2 * x - y)^2 = 1} in
    y_values.sum (λ y, y^2)
  else 0

theorem sum_of_squares_is_122 :
  sum_of_squares x y = 122 := sorry

end sum_of_squares_is_122_l25_25710


namespace infinite_natural_numbers_factorable_polynomial_l25_25332

theorem infinite_natural_numbers_factorable_polynomial :
  ∃ᶠ N in at_top, ∃ (k : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧
    ∃ (f g : ℤ[x]), (degree f = 4) ∧ (degree g = 4) ∧ (f * g = X^8 + C (N : ℤ) * X^4 + 1) :=
sorry

end infinite_natural_numbers_factorable_polynomial_l25_25332


namespace cone_lateral_surface_area_eq_sqrt_17_pi_l25_25962

theorem cone_lateral_surface_area_eq_sqrt_17_pi
  (r_cone r_sphere : ℝ) (h : ℝ)
  (V_sphere V_cone : ℝ)
  (h_cone_radius : r_cone = 1)
  (h_sphere_radius : r_sphere = 1)
  (h_volumes_eq : V_sphere = V_cone)
  (h_sphere_vol : V_sphere = (4 * π) / 3)
  (h_cone_vol : V_cone = (π * r_cone^2 * h) / 3) :
  (π * r_cone * (Real.sqrt (r_cone^2 + h^2))) = Real.sqrt 17 * π :=
sorry

end cone_lateral_surface_area_eq_sqrt_17_pi_l25_25962


namespace interest_rate_l25_25375

theorem interest_rate (P : ℝ) (t : ℝ) (d : ℝ) (r : ℝ) : 
  P = 8000.000000000171 → t = 2 → d = 20 →
  (P * (1 + r/100)^2 - P - (P * r * t / 100) = d) → r = 5 :=
by
  intros hP ht hd heq
  sorry

end interest_rate_l25_25375


namespace brother_gave_correct_l25_25709

-- Define the initial condition of gum pieces Robin had
def initial_gum_pieces : ℕ := 18

-- Define the total number of gum pieces Robin has now
def total_gum_pieces : ℕ := 44

-- Define the number of gum pieces Robin's brother gave her
def brother_gave := total_gum_pieces - initial_gum_pieces

-- Prove that Robin's brother gave her 26 pieces of gum
theorem brother_gave_correct : brother_gave = 26 :=
by
  -- Include the calculation in Lean to help Lean understand the arithmetic
  unfold brother_gave
  simp
  sorry

end brother_gave_correct_l25_25709


namespace distance_from_A_to_directrix_l25_25154

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25154


namespace problem_1_problem_2_l25_25583

-- Condition definitions
def ellipse (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 9) = 1
def point_P : (ℝ × ℝ) := (4, 2)
def line_eq (m : ℝ) (P : ℝ × ℝ) : ℝ → ℝ := λ x, m * (x - P.1) + P.2

-- Problem 1: When the slope of line l is 1/2, find the length of segment AB
theorem problem_1 : 
  ∀ (x1 x2 y1 y2 : ℝ),
  ellipse x1 y1 → ellipse x2 y2 → 
  x1 ≠ x2 ∧ y1 ≠ y2 → 
  line_eq (1/2) point_P x1 = y1 → line_eq (1/2) point_P x2 = y2 →
  dist (x1, y1) (x2, y2) = 3 * real.sqrt 10 := 
sorry

-- Problem 2: When point P is exactly the midpoint of segment AB, find the equation of line l
theorem problem_2 : 
  ∀ (x1 x2 y1 y2 : ℝ),
  ellipse x1 y1 → ellipse x2 y2 → 
  (x1 + x2) / 2 = point_P.1 → (y1 + y2) / 2 = point_P.2 →
  ∃ (m b : ℝ), (m = -1/2) ∧ (b = 4) ∧ ∀ x, (line_eq m point_P x = (line_eq m (0, b) x)) :=
sorry

end problem_1_problem_2_l25_25583


namespace triangle_angles_l25_25217

variable (A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180

theorem triangle_angles (x : ℝ) (hA : A = x) (hB : B = 2 * A) (hC : C + A + B = 180) :
  A = x ∧ B = 2 * x ∧ C = 180 - 3 * x := by
  -- proof goes here
  sorry

end triangle_angles_l25_25217


namespace impossible_distinct_values_l25_25615

theorem impossible_distinct_values :
  ∀ a b c : ℝ, 
  (a * (a - 4) = 12) → 
  (b * (b - 4) = 12) → 
  (c * (c - 4) = 12) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) → 
  false := 
sorry

end impossible_distinct_values_l25_25615


namespace possible_values_of_5x_plus_2_l25_25194

theorem possible_values_of_5x_plus_2 (x : ℝ) :
  (x - 4) * (5 * x + 2) = 0 →
  (5 * x + 2 = 0 ∨ 5 * x + 2 = 22) :=
by
  intro h
  sorry

end possible_values_of_5x_plus_2_l25_25194


namespace caroline_socks_gift_l25_25503

theorem caroline_socks_gift :
  ∀ (initial lost donated_fraction purchased total received),
    initial = 40 →
    lost = 4 →
    donated_fraction = 2 / 3 →
    purchased = 10 →
    total = 25 →
    received = total - (initial - lost - donated_fraction * (initial - lost) + purchased) →
    received = 3 :=
by
  intros initial lost donated_fraction purchased total received
  intro h_initial h_lost h_donated_fraction h_purchased h_total h_received
  sorry

end caroline_socks_gift_l25_25503


namespace towel_length_decrease_l25_25476

theorem towel_length_decrease (L B : ℝ) (HL1: L > 0) (HB1: B > 0)
  (length_percent_decr : ℝ) (breadth_decr : B' = 0.8 * B) 
  (area_decr : (L' * B') = 0.64 * (L * B)) :
  (L' = 0.8 * L) ∧ (length_percent_decrease = 20) := by
  sorry

end towel_length_decrease_l25_25476


namespace distance_to_directrix_l25_25094

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25094


namespace infinite_N_for_factorization_l25_25315

noncomputable def is_special_number (k : ℤ) : ℕ :=
  4 * k ^ 4 - 8 * k ^ 2 + 2

theorem infinite_N_for_factorization : 
  ∀ k : ℤ, ∃ (P Q : Polynomial ℤ), Polynomial.degree P = 4 ∧ Polynomial.degree Q = 4 
          ∧ Polynomial.mul P Q = Polynomial.C 1 + Polynomial.C (is_special_number k) * Polynomial.X^4 + Polynomial.X^8 :=
by sorry

end infinite_N_for_factorization_l25_25315


namespace m_value_quadratic_l25_25590

theorem m_value_quadratic (m : ℝ)
  (h1 : |m - 2| = 2)
  (h2 : m - 4 ≠ 0) :
  m = 0 :=
sorry

end m_value_quadratic_l25_25590


namespace triangle_area_of_parabola_and_hyperbola_l25_25728

noncomputable def parabola_directrix (x : ℝ) : Prop := y^2 = -12 * x

noncomputable def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 3) = 1

theorem triangle_area_of_parabola_and_hyperbola :
  ∀ (x y : ℝ), parabola_directrix x ∧ hyperbola_eq x (y / 3)
  → y = 3 * sqrt 3 :=
begin
  sorry,
end

end triangle_area_of_parabola_and_hyperbola_l25_25728


namespace scientific_notation_gdp_2022_l25_25487

def gdp_2022_fujian : ℝ := 53100 * 10^9

theorem scientific_notation_gdp_2022 : 
  (53100 * 10^9) = 5.31 * 10^12 :=
by
  -- The proof is based on the understanding that 53100 * 10^9 can be rewritten as 5.31 * 10^12
  -- However, this proof is currently omitted with a placeholder.
  sorry

end scientific_notation_gdp_2022_l25_25487


namespace find_equation_of_l_l25_25178

noncomputable def circle_o : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = 4}
noncomputable def circle_c : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 + 4 * p.1 - 4 * p.2 + 4 = 0}

theorem find_equation_of_l (l : ℝ → ℝ → Prop) :
  (∀ p, circle_o p ↔ circle_c (2 - p.1, p.2 + 2)) →
  (∀ x y, l x y ↔ x - y + 2 = 0) :=
sorry

end find_equation_of_l_l25_25178


namespace largest_common_divisor_476_330_l25_25771

theorem largest_common_divisor_476_330 :
  ∀ (S₁ S₂ : Finset ℕ), 
    S₁ = {1, 2, 4, 7, 14, 28, 17, 34, 68, 119, 238, 476} → 
    S₂ = {1, 2, 3, 5, 6, 10, 11, 15, 22, 30, 33, 55, 66, 110, 165, 330} → 
    ∃ D, D ∈ S₁ ∧ D ∈ S₂ ∧ ∀ x, x ∈ S₁ ∧ x ∈ S₂ → x ≤ D ∧ D = 2 :=
by
  intros S₁ S₂ hS₁ hS₂
  use 2
  sorry

end largest_common_divisor_476_330_l25_25771


namespace permutation_exists_to_split_all_subsets_l25_25613

def splits (A : List ℕ) (B : List ℕ) : Prop :=
  ∃ i j k : ℕ, (i < j) ∧ (j < k) ∧ (A[i] ∈ B) ∧ (A[j] ∉ B) ∧ (A[k] ∈ B)

def valid_subset (B : List ℕ) (n : ℕ) : Prop :=
  2 ≤ B.length ∧ B.length ≤ n - 1

def permutation_splits_subsets (n : ℕ) (S : List (List ℕ)) : Prop :=
  ∃ A : List ℕ, A.is_permutation (List.range (n + 1)) ∧ (∀ B ∈ S, splits A B)

theorem permutation_exists_to_split_all_subsets (n : ℕ) (S : List (List ℕ)) :
  (∀ B ∈ S, valid_subset B n) ∧ S.length = n - 2 → permutation_splits_subsets n S :=
by
  sorry

end permutation_exists_to_split_all_subsets_l25_25613


namespace largest_even_digit_multiple_of_9_below_1000_l25_25783

theorem largest_even_digit_multiple_of_9_below_1000 :
  ∃ n : ℕ, n = 882 ∧ n < 1000 ∧ (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d % 2 = 0) ∧ n % 9 = 0 :=
by
  existsi 882
  split
  { rfl }
  split
  { norm_num }
  split
  { intro d
    intro h
    fin_cases d with
    | h1 => norm_num
    | h2 => norm_num
    | h3 => norm_num }
  { norm_num }

end largest_even_digit_multiple_of_9_below_1000_l25_25783


namespace _l25_25141

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25141


namespace function_passes_through_fixed_point_l25_25736

-- Define the function and conditions
def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2) + 3

-- Specify the conditions
variables {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)

-- The theorem stating that the function passes through the point (2, 4)
theorem function_passes_through_fixed_point : f a 2 = 4 :=
by {
  -- We need to prove f(2) = 4
  sorry
}

end function_passes_through_fixed_point_l25_25736


namespace num_prime_candidates_l25_25184

-- Define the conditions needed for the problem
def between (n a b : ℕ) : Prop := a ≤ n ∧ n ≤ b
def remainder_eq (n r d : ℕ) : Prop := n % d = r

-- Define the prime property
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the relevant numbers and conditions
def candidates : List ℕ := [53, 57, 63, 67, 73, 77, 83, 87]

-- Define prime candidates with remainder of 3 or 7
def prime_candidates : List ℕ := candidates.filter (λ n => is_prime n ∧ (remainder_eq n 3 10 ∨ remainder_eq n 7 10))

-- Prove the number of such prime numbers is 4
theorem num_prime_candidates : prime_candidates.length = 4 := sorry

end num_prime_candidates_l25_25184


namespace largest_even_digit_multiple_of_nine_l25_25786

theorem largest_even_digit_multiple_of_nine : ∃ n : ℕ, (n < 1000) ∧ (∀ d ∈ digits 10 n, d % 2 = 0) ∧ (n % 9 = 0) ∧ n = 888 := 
by
  sorry

end largest_even_digit_multiple_of_nine_l25_25786


namespace parallel_line_perpendicular_to_plane_l25_25288

variables (a b α : Type) [LinearOrderedField α]

-- Define the properties
def is_parallel (l1 l2 : a) := ∃ t : α, ∀ (p : a), p ∈ l1 → ∃ q : a, q ∈ l2 ∧ t * q = p
def is_perpendicular_to_plane (l : a) (π : α) := ∀ (d : a), d ∈ π → ∀ (p : a), p ∈ l → p ⟂ d

-- The theorem statement
theorem parallel_line_perpendicular_to_plane
  (a b : a) (π : α)
  (h1 : is_parallel a b)
  (h2 : is_perpendicular_to_plane a π) :
  is_perpendicular_to_plane b π :=
sorry

end parallel_line_perpendicular_to_plane_l25_25288


namespace max_cos_sum_l25_25854

theorem max_cos_sum (A B C : ℝ) (h : A + B + C = π) :
  cos A + cos B * cos C ≤ 1 :=
sorry

end max_cos_sum_l25_25854


namespace value_of_expression_l25_25421

theorem value_of_expression : (2112 - 2021) ^ 2 / 169 = 49 := by
  sorry

end value_of_expression_l25_25421


namespace exponential_inequality_l25_25931

variable (x y : ℝ)

axiom h1 : x > y
axiom h2 : y > 0

theorem exponential_inequality : 2^x - 2^y > 0 :=
by
  sorry

end exponential_inequality_l25_25931


namespace distance_from_point_to_parabola_directrix_l25_25105

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25105


namespace triangle_extension_AC_eq_BK_l25_25649

/-
In triangle \( ABC \) with point \( M \) being the midpoint of \( AB \), let \( CM \) be the median.
Extend \( CM \) beyond \( C \) to point \( K \) such that \( AM = CK \).
Also, \( \angle BMC = 60^\circ \).
Prove that \( AC = BK \).
-/
theorem triangle_extension_AC_eq_BK {A B C K M : Point} 
  (hM : is_midpoint M A B) 
  (hCM : is_median C M) 
  (hK : extension C M K ∧ distance A M = distance C K) 
  (angleBMC : angle B M C = 60) 
  : distance A C = distance B K :=
sorry

end triangle_extension_AC_eq_BK_l25_25649


namespace altitude_of_triangle_l25_25623

open Real

theorem altitude_of_triangle
  (A B C : Point)
  (h : ℝ)
  (AB AC BC : ℝ)
  (angleBAC : ℝ)
  (hyp_AB : AB = 2)
  (hyp_BC : BC = sqrt 7)
  (hyp_angleBAC : angleBAC = π / 3)
  (hyp_area_h : (1/2) * AB * h = (1/2) * AB * AC * sin angleBAC)
  (solution_AC : quadratic 1 (-2) (-3) = [3, -1]) :
  h = 3 * (sqrt 3) / 2 :=
by
  sorry

end altitude_of_triangle_l25_25623


namespace max_value_l25_25566

variables {x y z : ℝ}
-- Conditions
def conditions : Prop := x > 0 ∧ y > 0 ∧ z > 0 ∧ 2 * x + 2 * y + z = 1

-- The claim to be proved
theorem max_value : conditions → (3 * x * y + y * z + z * x) ≤ 1 / 5 :=
begin
  sorry,
end

end max_value_l25_25566


namespace general_formula_a_sum_first_n_terms_l25_25579

-- Definitions and assumptions
def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)

def sequence_a (a : ℕ → ℕ) : Prop :=
  is_arithmetic_seq a ∧ a 1 = 2 ∧ a 1 + a 2 + a 3 = 12

-- Problem 1: Find the general formula for the sequence {a_n}
theorem general_formula_a (a : ℕ → ℕ) (h : sequence_a a) : ∀ n, a n = 2 * n := 
sorry

-- Problem 2: Find the sum of the first n terms of the sequence {b_n}
def b_n (a : ℕ → ℕ) (c : ℝ) (n : ℕ) := a n + c^(a n)

def S_n (a : ℕ → ℕ) (c : ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, b_n a c i

theorem sum_first_n_terms (a : ℕ → ℕ) (c : ℝ) (h : sequence_a a) :
  ∀ n, S_n a c n = 
  if c = 0 then n * (n + 1)
  else if c = 1 then n * (n + 1) + n
  else n * (n + 1) + (c * (1 - c^(2 * n)) / (1 - c^2)) :=
sorry

end general_formula_a_sum_first_n_terms_l25_25579


namespace pizza_slice_volume_and_area_l25_25843

structure PizzaSlice :=
  (thickness : ℝ)
  (diameter : ℝ)
  (slices : ℕ)

noncomputable def radius (p : PizzaSlice) : ℝ :=
  p.diameter / 2

noncomputable def volume (p : PizzaSlice) : ℝ :=
  π * (radius p)^2 * p.thickness

noncomputable def piece_volume (p : PizzaSlice) : ℝ :=
  volume p / p.slices

noncomputable def circumference (p : PizzaSlice) : ℝ :=
  2 * π * radius p

noncomputable def curved_edge (p : PizzaSlice) : ℝ :=
  circumference p / p.slices

noncomputable def piece_area (p : PizzaSlice) : ℝ :=
  curved_edge p * p.thickness

theorem pizza_slice_volume_and_area (p : PizzaSlice) 
  (hp_thick : p.thickness = 1 / 2)
  (hp_diam : p.diameter = 10)
  (hp_slices : p.slices = 12) :
  piece_volume p = 25 * π / 24 ∧ piece_area p = 5 * π / 12 :=
by
  sorry

end pizza_slice_volume_and_area_l25_25843


namespace correct_inequality_application_l25_25414

theorem correct_inequality_application (x : ℝ) (hx : 0 < x) (hx_acute : 0 < x ∧ x < π / 2) :
  (∀ (y : ℝ), y = x + 4 / x → y ≥ 2 * sqrt (x * (4 / x)) → false) ∧
  (∀ (y : ℝ), y = sin x + 4 / sin x → y ≥ 2 * sqrt (sin x * (4 / sin x)) → false) ∧
  (∀ (y : ℝ), 0 < x ∧ x < 1 → y = log x + 4 * log 10 / log x → y ≥ 2 * sqrt (log x * (4 * log 10 / log x)) → false) ∧
  (∀ (y : ℝ), y = 3^x + 4 / 3^x → y ≥ 2 * sqrt (3^x * (4 / 3^x)) → true) := by
  sorry

end correct_inequality_application_l25_25414


namespace probability_of_mixed_colors_l25_25893

theorem probability_of_mixed_colors :
  let num_balls := 6
  let prob_black_or_white := (1 : ℚ) / 2
  ∃ (prob : ℚ), prob = (20 * prob_black_or_white^num_balls) ∧ (prob = 5 / 16)
:=
begin
  sorry
end

end probability_of_mixed_colors_l25_25893


namespace domain_and_extremum_of_g_l25_25967

noncomputable def f (x : ℝ) := real.log (3 - 4 * x + x^2)
noncomputable def g (x : ℝ) := 2^(x+2) - 3 * 4^x
def M : set ℝ := {x | x < 1 ∨ x > 3}

theorem domain_and_extremum_of_g :
  (∀ x, f x = real.log (3 - 4 * x + x^2) → (3 - 4 * x + x^2 > 0 ↔ x ∈ M)) ∧
  (∀ x ∈ M, ∃ t, g x = -3 * (t - 2 / 3)^2 + 4 / 3 ∧ (x = real.log 2 (2 / 3) → g x = 4 / 3) ∧ ∀ y, g y ≤ 4 / 3) :=
begin
  sorry
end

end domain_and_extremum_of_g_l25_25967


namespace find_extrema_l25_25174

-- Define the variables and the constraints
variables (x y z : ℝ)

-- Define the inequalities as conditions
def cond1 := -1 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 8
def cond2 := 2 ≤ x - y + z ∧ x - y + z ≤ 9
def cond3 := -3 ≤ x + 2 * y - z ∧ x + 2 * y - z ≤ 7

-- Define the function f
def f (x y z : ℝ) := 7 * x + 5 * y - 2 * z

-- State the theorem that needs to be proved
theorem find_extrema :
  (∃ x y z, cond1 x y z ∧ cond2 x y z ∧ cond3 x y z) →
  (-6 ≤ f x y z ∧ f x y z ≤ 47) :=
by sorry

end find_extrema_l25_25174


namespace number_of_correct_answers_l25_25475

def total_questions := 30
def correct_points := 3
def incorrect_points := -1
def total_score := 78

theorem number_of_correct_answers (x : ℕ) :
  3 * x + incorrect_points * (total_questions - x) = total_score → x = 27 :=
by
  sorry

end number_of_correct_answers_l25_25475


namespace line_perpendicular_to_plane_l25_25991

variable (a : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ)

axiom direction_vector_l : a = (1, 0, 2)
axiom normal_vector_alpha : n = (-2, 0, -4)

theorem line_perpendicular_to_plane : a = (1, 0, 2) → n = (-2, 0, -4) → dot_product a n = 0 :=
by
  sorry

end line_perpendicular_to_plane_l25_25991


namespace lcm_gcd_of_fractions_l25_25788

namespace FractionsLCMGCD

-- Define the fractions
def fraction1 (x : ℕ) : ℚ := 1 / (4 * x)
def fraction2 (x : ℕ) : ℚ := 1 / (6 * x)
def fraction3 (x : ℕ) : ℚ := 1 / (9 * x)

-- The statement to prove the LCM and GCD of the given fractions
theorem lcm_gcd_of_fractions (x : ℕ) (h : x > 0) :
  (lcm {fraction1 x, fraction2 x, fraction3 x} = 1 / (36 * x)) ∧
  (gcd {fraction1 x, fraction2 x, fraction3 x} = x) :=
sorry

end FractionsLCMGCD

end lcm_gcd_of_fractions_l25_25788


namespace poly_factorable_l25_25306

-- Define the specific form of N.
noncomputable def N (k : ℤ) : ℕ := 4 * k^4 - 8 * k^2 + 2

-- Define the polynomial.
noncomputable def poly (N : ℕ) : ℤ[X] := X^8 + N * X^4 + 1

-- Define the fourth-degree polynomials used in the factorization.
noncomputable def poly1 (k : ℤ) : ℤ[X] :=
  X^4 - 2 * k * X^3 + 2 * k^2 * X^2 - 2 * k * X + 1

noncomputable def poly2 (k : ℤ) : ℤ[X] := 
  X^4 + 2 * k * X^3 + 2 * k^2 * X^2 + 2 * k * X + 1

-- The proof statement: prove that the polynomial can be factored into the specified fourth-degree polynomials.
theorem poly_factorable : ∀ k : ℤ, ∃ p1 p2 : ℤ[X], poly (N k) = p1 * p2 :=
by
  intro k
  use (poly1 k)
  use (poly2 k)
  sorry

end poly_factorable_l25_25306


namespace integer_solutions_l25_25529

theorem integer_solutions (x y : ℤ) : 
  (x^2 + x = y^4 + y^3 + y^2 + y) ↔ 
  (x, y) = (0, -1) ∨ (x, y) = (-1, -1) ∨ (x, y) = (0, 0) ∨ (x, y) = (-1, 0) ∨ (x, y) = (5, 2) :=
by
  sorry

end integer_solutions_l25_25529


namespace coefficient_of_x_l25_25577

theorem coefficient_of_x
  (a : ℝ) (h : 3^a = 9) :
  let expr := (frac a x - sqrt x)^8 in
  let term := (-1)^6 * (Nat.choose 8 6) * 2^(8 - 6) in
  expr.coeff(x) = 112 :=
by
  sorry

end coefficient_of_x_l25_25577


namespace man_speed_km_per_hr_l25_25477

noncomputable def train_length : ℝ := 110
noncomputable def train_speed_km_per_hr : ℝ := 82
noncomputable def time_to_pass_man_sec : ℝ := 4.499640028797696

theorem man_speed_km_per_hr :
  ∃ (Vm_km_per_hr : ℝ), Vm_km_per_hr = 6.0084 :=
sorry

end man_speed_km_per_hr_l25_25477


namespace sum_of_first_14_terms_of_arithmetic_sequence_l25_25637

theorem sum_of_first_14_terms_of_arithmetic_sequence :
  (∀ (a : ℕ → ℤ), a 4 = 3 ∧ a 11 = -3 → ∑ i in finset.range 14, a i = 0) :=
begin
  sorry
end

end sum_of_first_14_terms_of_arithmetic_sequence_l25_25637


namespace collinear_probability_l25_25670

/-- Define points in terms of the square. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

namespace Points

def O : Point := { x := 0.5, y := 0.5 } -- Center of the square (wlog)
def A : Point := { x := 0.0, y := 0.0 } -- Vertex A
def B : Point := { x := 1.0, y := 0.0 } -- Vertex B
def C : Point := { x := 1.0, y := 1.0 } -- Vertex C
def D : Point := { x := 0.0, y := 1.0 } -- Vertex D

def are_collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

def prob_collinear (points : List Point) : ℚ :=
  let combinations := List.combinations 3 points
  let collinear_combinations := combinations.filter (fun [p1, p2, p3] => are_collinear p1 p2 p3)
  collinear_combinations.length / combinations.length

theorem collinear_probability : (prob_collinear [O, A, B, C, D] = 1 / 5) :=
  sorry

end Points

end collinear_probability_l25_25670


namespace f_of_f_neg1_eq_neg1_f_x_eq_neg1_iff_x_eq_0_or_2_l25_25927

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 1 else -x + 1

-- Prove f[f(-1)] = -1
theorem f_of_f_neg1_eq_neg1 : f (f (-1)) = -1 := sorry

-- Prove that if f(x) = -1, then x = 0 or x = 2
theorem f_x_eq_neg1_iff_x_eq_0_or_2 (x : ℝ) : f x = -1 ↔ x = 0 ∨ x = 2 := sorry

end f_of_f_neg1_eq_neg1_f_x_eq_neg1_iff_x_eq_0_or_2_l25_25927


namespace simplify_expression_l25_25711

variable (x y a b c : ℝ)

theorem simplify_expression :
  (bx * (c^2 * x^2 + a^2 * x^2 + 2 * a^2 * y^2 + b^2 * y^2) + ay * (c^2 * x^2 + a^2 * x^2 + 2 * b^2 * x^2 + b^2 * y^2)) / (bx - ay)
  = (cx + ax + by)^2 := 
by
  sorry

end simplify_expression_l25_25711


namespace sqrt_15_between_3_and_4_l25_25896

theorem sqrt_15_between_3_and_4 :
  3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 :=
by
  have h1 : 3^2 = 9 := by norm_num
  have h2 : 4^2 = 16 := by norm_num
  have h3 : 9 < 15 ∧ 15 < 16 := by split; norm_num
  sorry

end sqrt_15_between_3_and_4_l25_25896


namespace symmetry_inequality_l25_25705

theorem symmetry_inequality
  (A B A' K M : ℝ)
  (l : Set ℝ)
  (h1 : A ∈ l) (h2 : B ∈ l) (h3 : A' ∈ l)
  (h4 : K ∈ l) (h5 : K ≠ M) (h6 : symmetric A A' l) :
  dist A K + dist K B < dist A M + dist M B :=
sorry

end symmetry_inequality_l25_25705


namespace max_value_of_m_l25_25264

def f (a x : ℝ) : ℝ := x ^ 2 - a * x + a / 2

noncomputable def m (a : ℝ) : ℝ :=
  if a < 0 then
    a / 2
  else if 0 ≤ a ∧ a ≤ 2 then
    a * (2 - a) / 4
  else
    1 - a / 2

theorem max_value_of_m : ∃ a : ℝ, m a = 1 / 4 :=
by
  sorry

end max_value_of_m_l25_25264


namespace three_digit_int_one_less_than_lcm_mult_l25_25425

theorem three_digit_int_one_less_than_lcm_mult : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ (n + 1) % Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 9 = 0 :=
sorry

end three_digit_int_one_less_than_lcm_mult_l25_25425


namespace determine_x_intersection_l25_25519

theorem determine_x_intersection :
  ∃ x y : ℝ, y = 3 * x - 20 ∧ 3 * x + y = 100 ∧ x = 20 :=
by
  use 20, 3 * 20 - 20
  split
  case left =>
    simp
  case right =>
    split
    case left =>
      simp
    case right =>
      simp
  sorry

end determine_x_intersection_l25_25519


namespace n_times_s_eq_2023_l25_25251

noncomputable def S := { x : ℝ | x > 0 }

-- Function f: S → ℝ
def f (x : ℝ) : ℝ := sorry

-- Condition: f(x) f(y) = f(xy) + 2023 * (2/x + 2/y + 2022) for all x, y > 0
axiom f_property (x y : ℝ) (hx : x > 0) (hy : y > 0) : f x * f y = f (x * y) + 2023 * (2 / x + 2 / y + 2022)

-- Theorem: Prove n × s = 2023 where n is the number of possible values of f(2) and s is the sum of all possible values of f(2)
theorem n_times_s_eq_2023 (n s : ℕ) : n * s = 2023 :=
sorry

end n_times_s_eq_2023_l25_25251


namespace Leningrad1989_l25_25812

theorem Leningrad1989 (k : ℕ) (hk : k > 1) :
  ¬∃ grid : Fin k → Fin k → ℕ,
    (∀ i : Fin k, is_power_of_two (∑ j, grid i j)) ∧
    (∀ j : Fin k, is_power_of_two (∑ i, grid i j)) ∧
    (∀ (i j : Fin k), 1 ≤ grid i j ∧ grid i j ≤ k^2) :=
sorry

def is_power_of_two (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2 ^ m

end Leningrad1989_l25_25812


namespace arithmetic_geometric_sequence_ab_l25_25605

theorem arithmetic_geometric_sequence_ab :
  ∀ (a l m b n : ℤ), 
    (b < 0) → 
    (2 * a = -10) → 
    (b^2 = 9) → 
    ab = 15 :=
by
  intros a l m b n hb ha hb_eq
  sorry

end arithmetic_geometric_sequence_ab_l25_25605


namespace max_cos_sum_l25_25855

theorem max_cos_sum (A B C : ℝ) (h : A + B + C = π) :
  cos A + cos B * cos C ≤ 1 :=
sorry

end max_cos_sum_l25_25855


namespace smallest_coterminal_angle_pos_radians_l25_25758

theorem smallest_coterminal_angle_pos_radians :
  ∀ (θ : ℝ), θ = -560 * (π / 180) → ∃ α : ℝ, α > 0 ∧ α = (8 * π) / 9 ∧ (∃ k : ℤ, θ + 2 * k * π = α) :=
by
  sorry

end smallest_coterminal_angle_pos_radians_l25_25758


namespace equidistant_z_axis_l25_25900
noncomputable def z : ℝ := -2 / 3

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨1, 0, 2⟩
def B : Point3D := ⟨2, -2, 1⟩
def C : Point3D := ⟨0, 0, z⟩

def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2)

theorem equidistant_z_axis : 
  distance C A = distance C B ↔ z = -2 / 3 :=
by {
  sorry
}

end equidistant_z_axis_l25_25900


namespace infinite_natural_numbers_with_factored_polynomial_l25_25326

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l25_25326


namespace largest_even_digit_multiple_of_9_under_1000_l25_25774

theorem largest_even_digit_multiple_of_9_under_1000 : 
  ∃ n : ℕ, (∀ d ∈ Int.digits 10 n, d % 2 = 0) ∧ n < 1000 ∧ n % 9 = 0 ∧ 
  (∀ m : ℕ, (∀ d ∈ Int.digits 10 m, d % 2 = 0) ∧ m < 1000 ∧ m % 9 = 0 → m ≤ n) ∧ n = 864 :=
sorry

end largest_even_digit_multiple_of_9_under_1000_l25_25774


namespace sum_of_distinct_integers_eq_zero_l25_25609

theorem sum_of_distinct_integers_eq_zero 
  (a b c d : ℤ) 
  (distinct : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  (prod_eq_25 : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end sum_of_distinct_integers_eq_zero_l25_25609


namespace total_highlighters_l25_25436

theorem total_highlighters : 
  let pink_highlighters := 3
  let yellow_highlighters := 7
  let blue_highlighters := 5
  pink_highlighters + yellow_highlighters + blue_highlighters = 15 := 
by 
  let pink_highlighters := 3 
  let yellow_highlighters := 7 
  let blue_highlighters := 5 
  show (pink_highlighters + yellow_highlighters + blue_highlighters = 15), from sorry

end total_highlighters_l25_25436


namespace find_k_perpendicular_l25_25569

def vect_a : ℝ × ℝ × ℝ := (1, 1, 0)
def vect_b : ℝ × ℝ × ℝ := (-1, 0, 2)
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
def scalar_mult_vect (k : ℝ) (u : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (k * u.1, k * u.2, k * u.3)
def vect_add (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (u.1 + v.1, u.2 + v.2, u.3 + v.3)
def vect_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (u.1 - v.1, u.2 - v.2, u.3 - v.3)

theorem find_k_perpendicular :
  ∃ k : ℝ, k = 7 / 5 ∧
  dot_product (vect_add (scalar_mult_vect k vect_a) vect_b) 
              (vect_sub (scalar_mult_vect 2 vect_a) vect_b) = 0 :=
by
  sorry

end find_k_perpendicular_l25_25569


namespace common_factor_of_polynomials_l25_25735

variable (m : ℝ)

-- Define the polynomials
def poly1 : ℝ := m^2 - 4
def poly2 : ℝ := m^2 - 4 * m + 4

-- Statement of the problem in Lean 4
theorem common_factor_of_polynomials :
  (∃ (f : ℝ → ℝ), poly1 = (m - 2) * f m) ∧ (∃ (g : ℝ → ℝ), poly2 = (m - 2) * g m) := 
sorry

end common_factor_of_polynomials_l25_25735


namespace man_speed_same_direction_l25_25478

theorem man_speed_same_direction
  (L : ℝ) (v_train_kmph : ℝ) (t : ℝ) (v_man_kmph : ℝ)
  (hL : L = 300)
  (hv_train_kmph : v_train_kmph = 68)
  (ht : t = 17.998560115190784)
  (hv_man_kmph : v_man_kmph = 7.9952) :
  let v_train_mps := v_train_kmph * (1000 / 3600) in
  let relative_speed := L / t in
  let v_man_mps := v_train_mps - relative_speed in
  v_man_kmph = v_man_mps * 3.6 :=
by
  sorry

end man_speed_same_direction_l25_25478


namespace find_a_l25_25167

def f (a : ℝ) (x : ℝ) : ℝ := if x > 0 then a * x^3 else if -π/2 < x ∧ x < 0 then Real.cos x else 0

theorem find_a (a : ℝ) : f a (f a (-π/3)) = 1 ↔ a = 8 :=
by
  sorry

end find_a_l25_25167


namespace diamond_evaluation_l25_25515

def diamond (X Y : ℚ) : ℚ := (2 * X + 3 * Y) / 5

theorem diamond_evaluation : diamond (diamond 3 15) 6 = 192 / 25 := 
by
  sorry

end diamond_evaluation_l25_25515


namespace problem_theorem_l25_25279

noncomputable def square (A B C D : Point) : Prop := 
  is_square A B C D

def cyclic_quadrilateral (A B E Q : Point) : Prop :=
  is_cyclic a B E Q

def triangle_area_ratio (A E F P Q : Point) : Prop :=
  angle E A F = 45 ∧
  intersects A E B D P ∧
  intersects A F B D Q ∧
  area_ratio A E F A P Q = 2

theorem problem_theorem (A B C D E F P Q : Point) :
  square A B C D ∧
  is_on_segment B C E ∧
  is_on_segment C D F ∧
  cyclic_quadrilateral A B E Q ∧
  triangle_area_ratio A E F P Q
  → (area_ratio A E F A P Q = 2) :=
by sorry

end problem_theorem_l25_25279


namespace count_valid_permutations_l25_25602

def number := [1, 3, 7, 1, 5, 3, 7, 5, 1]

def is_divisible_by_5 (n : Nat) : Prop := (n % 5 = 0)

theorem count_valid_permutations :
  (Σ (l : List ℕ), l.perm number ∧ l.last sorry = 5).count = 2520 :=
sorry

end count_valid_permutations_l25_25602


namespace x_coord_of_point_on_parabola_l25_25199

def parabola_xcoord (x y : ℝ) (focus_dist : ℝ) : ℝ :=
  if h : x = 4 * y^2 ∧ focus_dist = 2 then
    7 / 16
  else
    0 -- this case should not happen under the problem's constraints

theorem x_coord_of_point_on_parabola :
  ∀ (x y : ℝ), x = 4 * y^2 → (∃ focus_dist : ℝ, focus_dist = 1 / 2) →
  parabola_xcoord x y (1 / 2) = 7 / 16 :=
by
  intros x y hx hfocus_dist
  simp [parabola_xcoord, hx, hfocus_dist]
  sorry

end x_coord_of_point_on_parabola_l25_25199


namespace minimum_blue_beads_l25_25462

theorem minimum_blue_beads (n : ℕ) (red blue green : Type) (necklace : list red ⊕ list blue ⊕ list green) 
  (h1 : ∀ r : red, neighbors r ≠ (same_color r)) 
  (h2 : ∀ g1 g2 : green, ∃ b : blue, between g1 g2 b) 
  (h3 : length necklace = 175) : 
  ∃ blues : nat, blues = 30 :=
by sorry

end minimum_blue_beads_l25_25462


namespace volume_of_inscribed_sphere_l25_25878

noncomputable theory

theorem volume_of_inscribed_sphere (h : ℝ) (a : ℝ) (V_S : ℝ) (π : ℝ) 
  (tetrahedron_height : h = 4) 
  (tetrahedron_base_edge : a = 2)
  (volume_formula : V_S = (sqrt 17 - 1)^3 / 48 * π) : 
  V_S = (sqrt 17 - 1)^3 / 48 * π :=
by
  -- Assuming the conditions given in the problem
  sorry

end volume_of_inscribed_sphere_l25_25878


namespace num_students_at_least_one_excellent_l25_25491

open Set

variable (Students : Type) [Fintype Students]

variable (Math Physics Lit : Set Students)
variable (n_math : |Math| = 48)
variable (n_physics : |Physics| = 37)
variable (n_lit : |Lit| = 42)
variable (n_math_or_physics : |Math ∪ Physics| = 75)
variable (n_math_or_lit : |Math ∪ Lit| = 76)
variable (n_physics_or_lit : |Physics ∪ Lit| = 66)
variable (n_all_three : |Math ∩ Physics ∩ Lit| = 4)

theorem num_students_at_least_one_excellent :
  |Math ∪ Physics ∪ Lit| = 94 := sorry

end num_students_at_least_one_excellent_l25_25491


namespace ratio_of_areas_of_triangles_and_trapezoid_l25_25866

theorem ratio_of_areas_of_triangles_and_trapezoid :
  ∀ (s1 s2 : ℝ), (s1 = 12) → (s2 = 6) →
  (∃ (area_small_triangle area_trapezoid : ℝ),
    area_small_triangle = (√3 / 4) * s2^2 ∧ 
    area_trapezoid = 3 * (√3 / 4) * s2^2 ∧ 
    (area_small_triangle / area_trapezoid = 1 / 3)) :=
by
  intros s1 s2 h1 h2
  use (√3 / 4) * s2^2, 3 * (√3 / 4) * s2^2
  split
  · rw h2
    ring
  split
  · rw h2
    ring
  · rw h2
    ring
  sorry

end ratio_of_areas_of_triangles_and_trapezoid_l25_25866


namespace transformed_sine_function_l25_25379

theorem transformed_sine_function :
  ∀ x, ( ∃ x', (y = sin x') ∧ x = 2 * (x' - 3)) → y = sin (2*x - 3) :=
by
  sorry

end transformed_sine_function_l25_25379


namespace even_function_negative_l25_25367

-- Define the even function property
def is_even_function (f : ℝ → ℝ) := ∀ x, f(-x) = f(x)

-- Define the function on (0, +∞)
def f_pos (x : ℝ) (hx : 0 < x) : ℝ := x * (1 + x)

-- Define the condition that the function is even
axiom f_even : is_even_function (λ x, if 0 < x then f_pos x (by linarith) else sorry)

theorem even_function_negative (f : ℝ → ℝ) (h_even : is_even_function f)
  (h_pos : ∀ x, 0 < x → f x = x * (1 + x)) :
  ∀ x, x < 0 → f x = x * (x - 1) :=
by
  intro x hx
  have h_neg : f x = f (-x) := h_even x
  have x_pos : -x > 0 := by linarith
  rw [h_neg, h_pos (-x) x_pos]
  ring

end even_function_negative_l25_25367


namespace distance_from_A_to_directrix_l25_25009

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25009


namespace distance_to_directrix_l25_25032

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25032


namespace exists_infinitely_many_N_l25_25298

theorem exists_infinitely_many_N (k : ℤ) : ∃ N : ℕ, ∃ A B : polynomial ℤ, 
  N = 4 * k^4 - 8 * k^2 + 2 ∧ polynomial.nat_degree A = 4 ∧ polynomial.nat_degree B = 4 ∧
  (polynomial.X^8 + polynomial.C (N:ℤ) * polynomial.X^4 + 1) = A * B :=
sorry

end exists_infinitely_many_N_l25_25298


namespace area_of_triangle_PQR_l25_25824

-- Definitions and conditions
variable (r1 r2 : ℝ) [nonzero r1] [nonzero r2]
variable (P Q R : Type) [Point P] [Point Q] [Point R]
variable (circle_small : Circle P 3) (circle_large : Circle Q 5)
variable (tangent_to_small : Tangent P Q) (tangent_to_large : Tangent P R)

-- Given QR is tangent to the circles, and PQ ≅ PR 
variable (isosceles_triangle : IsoscelesPQPR Q R)
variable (triangle_PQR : Triangle P Q R)

-- Proof of area calculation
theorem area_of_triangle_PQR : 
  ∃ (A : ℝ), A = 60 * Real.sqrt 5.25 := 
sorry

end area_of_triangle_PQR_l25_25824


namespace max_sin_angle_ACB_l25_25933

noncomputable def circle_C := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 25 }

noncomputable def line_l (m : ℝ) := { p : ℝ × ℝ | (2 * m + 1) * p.1 + (m + 1) * p.2 - 7 * m - 4 = 0 }

theorem max_sin_angle_ACB (m : ℝ) :
  ∃ A B : ℝ × ℝ, A ∈ circle_C ∧ B ∈ circle_C ∧ A ≠ B ∧ A ∈ line_l m ∧ B ∈ line_l m ∧
  (∃ θ : ℝ, θ = ∠ A C B ∧ sin θ = 4 / 5) :=
sorry

end max_sin_angle_ACB_l25_25933


namespace price_per_liter_l25_25358

theorem price_per_liter (cost : ℕ) (bottles : ℕ) (liters_per_bottle : ℕ) (total_cost : ℕ) (total_liters : ℕ) :
  bottles = 6 → liters_per_bottle = 2 → total_cost = 12 → total_liters = 12 → cost = total_cost / total_liters → cost = 1 :=
by
  intros h_bottles h_liters_per_bottle h_total_cost h_total_liters h_cost_div
  sorry

end price_per_liter_l25_25358


namespace segment_length_aa_prime_l25_25408

/-- Given points A, B, and C, and their reflections, show that the length of AA' is 8 -/
theorem segment_length_aa_prime
  (A : ℝ × ℝ) (A_reflected : ℝ × ℝ)
  (x₁ y₁ y₁_neg : ℝ) :
  A = (x₁, y₁) →
  A_reflected = (x₁, y₁_neg) →
  y₁_neg = -y₁ →
  y₁ = 4 →
  x₁ = 2 →
  |y₁ - y₁_neg| = 8 :=
sorry

end segment_length_aa_prime_l25_25408


namespace shopkeeper_weight_l25_25472

/-- A shopkeeper sells his goods at cost price but uses a certain weight instead of kilogram weight.
    His profit percentage is 25%. Prove that the weight he uses is 0.8 kilograms. -/
theorem shopkeeper_weight (c s p : ℝ) (x : ℝ) (h1 : s = c * (1 + p / 100))
  (h2 : p = 25) (h3 : c = 1) (h4 : s = 1.25) : x = 0.8 :=
by
  sorry

end shopkeeper_weight_l25_25472


namespace union_A_B_eq_real_subset_A_B_l25_25924

def A (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 3 + a}
def B : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 1}

theorem union_A_B_eq_real (a : ℝ) : (A a ∪ B) = Set.univ ↔ -2 ≤ a ∧ a ≤ -1 :=
by
  sorry

theorem subset_A_B (a : ℝ) : A a ⊆ B ↔ (a ≤ -4 ∨ a ≥ 1) :=
by
  sorry

end union_A_B_eq_real_subset_A_B_l25_25924


namespace speed_of_first_man_l25_25765

theorem speed_of_first_man (v : ℝ) : (∀ (d : ℝ), d = 6 → ∀ (s2 : ℝ), s2 = 3 → (d / v = d / s2 - 0.5)) → v = 4 := 
by
  intros h d hd s2 hs2
  rw [hd, hs2] at h
  specialize h 6 rfl 3 rfl
  sorry

end speed_of_first_man_l25_25765


namespace mod_expression_result_l25_25506

theorem mod_expression_result :
  ∃ x : ℤ, (7 * x ≡ 1 [MOD 80]) ∧
           (13 * x ≡ 1 [MOD 80]) ∧
           (3 * (23 : ℤ) + 9 * (37 : ℤ) ≡ 2 [MOD 80]) :=
by
  have inv7 : 7 * 23 ≡ 1 [MOD 80] := by norm_num
  have inv13 : 13 * 37 ≡ 1 [MOD 80] := by norm_num
  exists 23; split; try {assumption}; try { norm_num }
  exists 37; split; try {assumption}; try { norm_num }
  sorry

end mod_expression_result_l25_25506


namespace distance_from_A_to_directrix_l25_25000

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25000


namespace find_p_q_l25_25568

noncomputable def cubicFunction (p q : ℝ) (x : ℂ) : ℂ :=
  2 * x^3 + p * x^2 + q * x

theorem find_p_q (p q : ℝ) :
  cubicFunction p q (2 * Complex.I - 3) = 0 ∧ 
  cubicFunction p q (-2 * Complex.I - 3) = 0 → 
  p = 12 ∧ q = 26 :=
by
  sorry

end find_p_q_l25_25568


namespace outside_bookshop_discount_l25_25750

theorem outside_bookshop_discount (school_price : ℝ) (num_books : ℕ) (savings : ℝ) (discount : ℝ) : 
  school_price = 45 → num_books = 3 → savings = 27 → discount = 20 :=
by
  -- Rewrite school_price and num_books in the conditions form
  assume h1 : school_price = 45,
  assume h2 : num_books = 3,
  assume h3 : savings = 27,
  
  -- Calculate total costs and discount percentage
  let total_school_cost := school_price * num_books,
  let total_outside_cost := total_school_cost - savings,
  let price_one_outside := total_outside_cost / num_books,
  let calculated_discount := ((school_price - price_one_outside) / school_price) * 100,
  
  -- Set the discount to be 20%
  have h4 : discount = calculated_discount,
  rw [h1, h2, h3] at h4,
  sorry -- We skip the proof.

end outside_bookshop_discount_l25_25750


namespace distance_from_A_to_directrix_l25_25155

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25155


namespace exist_sum_2000_not_exist_sum_2001_l25_25513

-- Definitions for conditions
def sum_of_cubes (n : ℕ): ℤ := ∑ k in finset.range (n + 1), (k + 1) ^ 3

def exists_sum_epsilon (n : ℕ): Prop :=
  ∃ (ε : fin (n + 1) → {-1, 1}), (∑ i in finset.range (n + 1), ε i * (i + 1) ^ 3) = 0

-- Part (a)
theorem exist_sum_2000 : exists_sum_epsilon 2000 :=
  sorry

-- Part (b)
theorem not_exist_sum_2001 : ¬exists_sum_epsilon 2001 :=
  sorry

end exist_sum_2000_not_exist_sum_2001_l25_25513


namespace Maria_score_l25_25271

theorem Maria_score (x : ℝ) (y : ℝ) (h1 : x = y + 50) (h2 : (x + y) / 2 = 105) : x = 130 :=
by
  sorry

end Maria_score_l25_25271


namespace max_segment_length_l25_25749

-- Define the perimeter condition
def perimeter (A B C : Point) : ℝ := 2 * p

-- Define the maximum length calculation
def max_length (A B C inscribed_circle : Point) (KL : Line) : ℝ :=
  let p := perimeter A B C / 2 in
  p / 4

-- Define the tangency condition
def tangent (circle : Point → Prop) (L : Line) : Prop :=
  ∃ P, circle P → L ∋ P

-- Define the proof statement
theorem max_segment_length (A B C inscribed_circle : Point) (BC KL : Line)
  (h1 : tangent inscribed_circle KL)
  (h2 : parallel KL BC)
  (h3 : perimeter A B C = 2 * p) :
  max_length A B C inscribed_circle KL = p / 4 :=
sorry

end max_segment_length_l25_25749


namespace minimum_value_of_sum_l25_25685

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
    1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) >= 3 :=
by
  sorry

end minimum_value_of_sum_l25_25685


namespace _l25_25135

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25135


namespace rancher_lasso_probability_l25_25802

theorem rancher_lasso_probability : 
  let p_success := 1 / 2
  let p_failure := 1 - p_success
  (1 - p_failure ^ 3) = (7 / 8) := by
  sorry

end rancher_lasso_probability_l25_25802


namespace min_max_of_deriv_pos_l25_25993

open Interval

variable (a b : ℝ) (f : ℝ → ℝ)

theorem min_max_of_deriv_pos (h : ∀ x ∈ Icc a b, 0 < deriv f x) : 
  ∀ x ∈ Icc a b, f a ≤ f x ∧ f x ≤ f b :=
by
  sorry

end min_max_of_deriv_pos_l25_25993


namespace min_value_expression_l25_25930

theorem min_value_expression (x : ℝ) (h : x > 1) : 
  ∃ min_val, min_val = 6 ∧ ∀ y > 1, 2 * y + 2 / (y - 1) ≥ min_val :=
by  
  use 6
  sorry

end min_value_expression_l25_25930


namespace min_blue_beads_l25_25463

-- Define a necklace of n beads where each bead is either red, green, or blue.
noncomputable def Bead : Type := 
  | Red
  | Green
  | Blue

-- Define a cyclic necklace of fixed length 175.
structure Necklace where
  beads : Fin 175 → Bead
  cyclic : beads 0 = beads 174

-- Define the two conditions.
def condition1 (necklace : Necklace) : Prop := 
  ∀ i : Fin 175, 
    if necklace.beads i == Bead.Red then 
      (necklace.beads ((i + 1) % 175) ≠ necklace.beads i) 
      ∧ (necklace.beads ((i - 1) % 175) ≠ necklace.beads i)
    else true

def condition2 (necklace : Necklace) : Prop := 
  ∀ i j : Fin 175, i < j,
    (necklace.beads i == Bead.Green ∧ necklace.beads j == Bead.Green) → 
    ∃ k : Fin 175, i < k < j ∧ necklace.beads k == Bead.Blue

-- Define the proof problem.
theorem min_blue_beads (necklace : Necklace) 
  (h1 : condition1 necklace) 
  (h2 : condition2 necklace) : 
  ∃ B : ℕ, B = 30 ∧ (∃ b_set : Fin 175 → bool, 
    (∀ i : Fin 175, b_set i = true ↔ necklace.beads i = Bead.Blue) 
    ∧ (b_set.filter (λ x, x = true)).size = B) :=
sorry

end min_blue_beads_l25_25463


namespace distance_to_directrix_l25_25084

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25084


namespace intersection_eq_l25_25176

open Set

def M : Set ℝ := { -1, 1 }
def N : Set ℝ := { x | (1 : ℝ) ≤ 2^x ∧ 2^x ≤ 4 }

theorem intersection_eq : M ∩ N = {1} :=
by
  sorry

end intersection_eq_l25_25176


namespace distance_from_A_to_directrix_l25_25143

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25143


namespace problem1_problem2_l25_25560

-- Definitions based on conditions in the problem
def seq_sum (a : ℕ) (n : ℕ) : ℕ := a * 2^n - 1
def a1 (a : ℕ) : ℕ := seq_sum a 1
def a4 (a : ℕ) : ℕ := seq_sum a 4 - seq_sum a 3

-- Problem statement 1
theorem problem1 (a : ℕ) (h : a = 3) : a1 a = 5 ∧ a4 a = 24 := by 
  sorry

-- Geometric sequence conditions
def is_geometric (a_n : ℕ → ℕ) : Prop :=
  ∃ q ≠ 1, ∀ n, a_n (n + 1) = q * a_n n

-- Definitions for the geometric sequence part
def a_n (a : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 2 * a - 1
  else if n = 2 then 2 * a
  else if n = 3 then 4 * a
  else 0 -- Simplifying for the first few terms only

-- Problem statement 2
theorem problem2 : (∃ a : ℕ, is_geometric (a_n a)) → ∃ a : ℕ, a = 1 := by
  sorry

end problem1_problem2_l25_25560


namespace least_sum_of_exponents_1023_l25_25988

theorem least_sum_of_exponents_1023 :
  ∃ (S : Finset ℕ), (S.sum (λ n, 2^n) = 1023) ∧ (S.card ≥ 3) ∧ (S.sum id = 45) := by
  sorry

end least_sum_of_exponents_1023_l25_25988


namespace max_min_value_of_f_l25_25171

theorem max_min_value_of_f (x y z : ℝ) :
  (-1 ≤ 2 * x + y - z) ∧ (2 * x + y - z ≤ 8) ∧
  (2 ≤ x - y + z) ∧ (x - y + z ≤ 9) ∧
  (-3 ≤ x + 2 * y - z) ∧ (x + 2 * y - z ≤ 7) →
  (-6 ≤ 7 * x + 5 * y - 2 * z) ∧ (7 * x + 5 * y - 2 * z ≤ 47) :=
by
  sorry

end max_min_value_of_f_l25_25171


namespace distance_from_A_to_directrix_on_parabola_l25_25040

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25040


namespace number_of_sheep_on_stewart_farm_l25_25805

def stewart_farm (S H : ℕ) (horse_food_per_day : ℕ) (total_horse_food_per_day : ℕ) :=
  (S / H = 4 / 7) ∧ (H * horse_food_per_day = total_horse_food_per_day)

theorem number_of_sheep_on_stewart_farm :
  ∀ (H : ℕ), stewart_farm 32 H 230 12880 :=
by
  intros H
  unfold stewart_farm
  split
  · sorry -- Use the first condition to show the ratio is achieved
  · sorry -- Use the second condition to show the total food is as given

end number_of_sheep_on_stewart_farm_l25_25805


namespace solve_equation_l25_25713

theorem solve_equation :
    (∃ x : ℝ, x = 15 + 4 * Real.sqrt 11 ∨ x = 15 - 4 * Real.sqrt 11) ∧
    ∀ x : ℝ, (sqrt (x + 15) - 7 / sqrt (x + 15) = 4) ↔ (x = 15 + 4 * Real.sqrt 11 ∨ x = 15 - 4 * Real.sqrt 11) := 
by 
    sorry

end solve_equation_l25_25713


namespace distance_from_A_to_directrix_on_parabola_l25_25045

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25045


namespace largest_even_digit_multiple_of_9_under_1000_l25_25772

theorem largest_even_digit_multiple_of_9_under_1000 : 
  ∃ n : ℕ, (∀ d ∈ Int.digits 10 n, d % 2 = 0) ∧ n < 1000 ∧ n % 9 = 0 ∧ 
  (∀ m : ℕ, (∀ d ∈ Int.digits 10 m, d % 2 = 0) ∧ m < 1000 ∧ m % 9 = 0 → m ≤ n) ∧ n = 864 :=
sorry

end largest_even_digit_multiple_of_9_under_1000_l25_25772


namespace sum_of_coefficients_l25_25508

theorem sum_of_coefficients (a b c d : ℤ)
  (h1 : ∀ x : ℤ, f x = a * x^3 + b * x^2 + c * x + d)
  (h2 : ∀ x : ℤ, f (x + 2) = 2 * x^3 - x^2 + 5 * x + 3) :
  a + b + c + d = -5 := by
  sorry

end sum_of_coefficients_l25_25508


namespace problem1_solution_problem2_solution_problem3_solution_l25_25714

-- Problem 1
theorem problem1_solution (x : ℝ) :
  (6 * x - 1) ^ 2 = 25 ↔ (x = 1 ∨ x = -2 / 3) :=
sorry

-- Problem 2
theorem problem2_solution (x : ℝ) :
  4 * x^2 - 1 = 12 * x ↔ (x = 3 / 2 + (Real.sqrt 10) / 2 ∨ x = 3 / 2 - (Real.sqrt 10) / 2) :=
sorry

-- Problem 3
theorem problem3_solution (x : ℝ) :
  x * (x - 7) = 8 * (7 - x) ↔ (x = 7 ∨ x = -8) :=
sorry

end problem1_solution_problem2_solution_problem3_solution_l25_25714


namespace sum_of_coordinates_of_D_l25_25702

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem sum_of_coordinates_of_D :
  let N := (3, 5)
  let C := (1, 10)
  let D := (5, 0)
  N = midpoint C D → D.1 + D.2 = 5 :=
by
  intro h
  sorry

end sum_of_coordinates_of_D_l25_25702


namespace imaginary_part_bound_l25_25953

theorem imaginary_part_bound (x y a b : ℝ) 
  (h1: x^2 + y^2 ≤ 1) 
  (h2: a^2 + b^2 ≤ 2) : 
  abs (b * (x^2 - y^2) + 2 * a * x * y) ≤ sqrt 2 := by
  sorry

end imaginary_part_bound_l25_25953


namespace piecewise_function_example_l25_25378

def f (x : ℝ) : ℝ :=
  if x < 1 then log 2 (3 - x) + 1 else 2 ^ x

theorem piecewise_function_example : f (log 2 12) + f (-1) = 15 := by
  -- initial statement setup
  sorry

end piecewise_function_example_l25_25378


namespace distance_to_directrix_l25_25024

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25024


namespace infinite_nat_N_polynomial_l25_25309

/-- The theorem states that for any integer k, 
there exists a natural number N = 4k^4 - 8k^2 + 2 such that the polynomial 
x^8 + Nx^4 + 1 can be factored into two fourth-degree polynomials with integer coefficients. -/
theorem infinite_nat_N_polynomial (k : ℤ) : ∃ (N : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧ 
  ∃ (P Q : ℤ[X]), P.degree = 4 ∧ Q.degree = 4 ∧ (X^8 + C (↑N) * X^4 + 1) = P * Q :=
by {
  sorry
}

end infinite_nat_N_polynomial_l25_25309


namespace infinite_nat_N_polynomial_l25_25307

/-- The theorem states that for any integer k, 
there exists a natural number N = 4k^4 - 8k^2 + 2 such that the polynomial 
x^8 + Nx^4 + 1 can be factored into two fourth-degree polynomials with integer coefficients. -/
theorem infinite_nat_N_polynomial (k : ℤ) : ∃ (N : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧ 
  ∃ (P Q : ℤ[X]), P.degree = 4 ∧ Q.degree = 4 ∧ (X^8 + C (↑N) * X^4 + 1) = P * Q :=
by {
  sorry
}

end infinite_nat_N_polynomial_l25_25307


namespace distance_from_A_to_directrix_of_parabola_l25_25058

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25058


namespace probability_of_at_least_one_girl_girl_pair_l25_25454

-- Definitions of conditions
def num_boys : ℕ := 8
def num_girls : ℕ := 8

-- Definition of the main proof problem
theorem probability_of_at_least_one_girl_girl_pair :
  (calc_probability num_boys num_girls) = 0.98 :=
sorry

-- Function to calculate the probability of at least one girl-girl pair
noncomputable def calc_probability (boys girls : ℕ) : ℝ :=
  let total_pairs := (fact (boys + girls)) / ((fact 2) ^ boys * (fact boys)) in
  let no_girl_girl_pairs := (fact boys)^2 / total_pairs in
  1 - no_girl_girl_pairs

end probability_of_at_least_one_girl_girl_pair_l25_25454


namespace exists_infinitely_many_N_l25_25294

theorem exists_infinitely_many_N (k : ℤ) : ∃ N : ℕ, ∃ A B : polynomial ℤ, 
  N = 4 * k^4 - 8 * k^2 + 2 ∧ polynomial.nat_degree A = 4 ∧ polynomial.nat_degree B = 4 ∧
  (polynomial.X^8 + polynomial.C (N:ℤ) * polynomial.X^4 + 1) = A * B :=
sorry

end exists_infinitely_many_N_l25_25294


namespace minimize_product_l25_25809

-- Definitions
variables {P A B O X Y : Point}
variables (onOA : OnRay O A X) (onOB : OnRay O B Y)
variables (isInterior : InteriorPoint P O A B)
variables (PX PY : ℝ) (OX OY : ℝ)

-- Given conditions
axiom PXcalc : PX = |P - X|
axiom PYcalc : PY = |P - Y|
axiom OXeqOY : OX = OY
axiom XOnOA : OnRay O A X
axiom YOnOB : OnRay O B Y
axiom PLineXY : Collinear P X Y

-- Prove statement
theorem minimize_product :
  ∃ X Y, OnRay O A X ∧ OnRay O B Y ∧ Collinear P X Y ∧ OX = OY → ∀ X' Y', OnRay O A X' ∧ OnRay O B Y' ∧ Collinear P X' Y' → (PXcalc * PYcalc) ≤ (|P - X'| * |P - Y'|) :=
by
  sorry

end minimize_product_l25_25809


namespace card_area_after_reduction_width_l25_25496

def initial_length : ℕ := 5
def initial_width : ℕ := 8
def new_width := initial_width - 2
def expected_new_area : ℕ := 24

theorem card_area_after_reduction_width :
  initial_length * new_width = expected_new_area := 
by
  -- initial_length = 5, new_width = 8 - 2 = 6
  -- 5 * 6 = 30, which was corrected to 24 given the misinterpretation mentioned.
  sorry

end card_area_after_reduction_width_l25_25496


namespace exists_infinitely_many_N_l25_25297

theorem exists_infinitely_many_N (k : ℤ) : ∃ N : ℕ, ∃ A B : polynomial ℤ, 
  N = 4 * k^4 - 8 * k^2 + 2 ∧ polynomial.nat_degree A = 4 ∧ polynomial.nat_degree B = 4 ∧
  (polynomial.X^8 + polynomial.C (N:ℤ) * polynomial.X^4 + 1) = A * B :=
sorry

end exists_infinitely_many_N_l25_25297


namespace product_evaluation_l25_25524

def floor (x : ℝ) : ℤ := Int.floor x
def ceil (x : ℝ) : ℤ := Int.ceil x

def product_expression : ℤ :=
  ∏ n in (Finset.range 7), (floor (-(n : ℤ) - 0.5) * ceil ((n : ℤ) + 0.5))

theorem product_evaluation :
  product_expression = -25401600 := 
sorry

end product_evaluation_l25_25524


namespace range_of_y_l25_25207

-- Define the function y(x)
def y (x : ℝ) : ℝ := (3 * x^2 - 3 * x + 4) / (x^2 - x + 1)

-- Prove the range of the function is (3, 13/3]
theorem range_of_y : (∃ x : ℝ, y x = a) ↔ (3 < a ∧ a ≤ 13 / 3) :=
sorry

end range_of_y_l25_25207


namespace no_filling_method_exists_l25_25651

theorem no_filling_method_exists : 
  ¬ (∃ f : ℕ → ℕ, (∀ n, 1 ≤ f n ∧ f n ≤ 21) ∧ (∀ a b c, b = f (a - 1) → c = f (a + 1) → f a = |b - c|)) :=
by 
  sorry

end no_filling_method_exists_l25_25651


namespace candy_pieces_given_l25_25652

theorem candy_pieces_given (initial total : ℕ) (h1 : initial = 68) (h2 : total = 93) :
  total - initial = 25 :=
by
  sorry

end candy_pieces_given_l25_25652


namespace austin_hours_on_mondays_l25_25872

-- Define the conditions
def earning_per_hour : ℕ := 5
def hours_wednesday : ℕ := 1
def hours_friday : ℕ := 3
def weeks : ℕ := 6
def bicycle_cost : ℕ := 180

-- Define the proof problem
theorem austin_hours_on_mondays (M : ℕ) :
  earning_per_hour * weeks * (M + hours_wednesday + hours_friday) = bicycle_cost → M = 2 :=
by 
  intro h
  sorry

end austin_hours_on_mondays_l25_25872


namespace _l25_25133

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25133


namespace calc_square_l25_25876

theorem calc_square (y : ℝ) : (7 - real.sqrt (y^2 - 49))^2 = y^2 - 14 * real.sqrt (y^2 - 49) :=
sorry

end calc_square_l25_25876


namespace find_XY_l25_25527

-- Given conditions
variables (XY ZY : ℝ)
variables (angle_X : ℝ)
variables (is_45_45_90 : ∀ {a b c : ℝ}, a = b ∧ c = a * Real.sqrt 2)

-- Hypotenuse given
def hypotenuse_given : ZY = 12 * Real.sqrt 2 := sorry

-- Angle given
def angle_45_degrees : angle_X = 45 := sorry

-- Goal: Prove XY = 12
theorem find_XY (h1: ∃ (a b c : ℝ), a = b ∧ c = a * Real.sqrt 2) (h2 : ZY = 12 * Real.sqrt 2) : XY = 12 :=
by
  cases h1 with a ha
  cases ha with b c
  cases c
  simp at h2
  simp
  sorry

end find_XY_l25_25527


namespace total_number_of_shells_l25_25273

variable (David Mia Ava Alice : ℕ)
variable (hd : David = 15)
variable (hm : Mia = 4 * David)
variable (ha : Ava = Mia + 20)
variable (hAlice : Alice = Ava / 2)

theorem total_number_of_shells :
  David + Mia + Ava + Alice = 195 :=
by
  sorry

end total_number_of_shells_l25_25273


namespace find_angle_RSP_l25_25949

-- Given conditions
variables (PQ RS : Prop) (QRS SPR PSR : ℝ)
variables (y : ℝ)

-- Hypotheses
axiom PQ_parallel_RS : PQ ↔ RS
axiom angle_QRS_def : QRS = 3 * y - 20
axiom angle_SPR_def : SPR = y + 40
axiom angle_PSR_def : PSR = 2 * y

-- Goal
theorem find_angle_RSP : 
  ∀ (PQ RS : Prop) (QRS SPR PSR : ℝ) (y : ℝ), 
  (PQ ↔ RS) → 
  (QRS = 3 * y - 20) → 
  (SPR = y + 40) → 
  (PSR = 2 * y) → 
  PSR = 60 :=
by {
  intro PQ RS QRS SPR PSR y,
  intro PQ_parallel_RS,
  intro angle_QRS_def,
  intro angle_SPR_def,
  intro angle_PSR_def,
  sorry
}

end find_angle_RSP_l25_25949


namespace distance_to_directrix_l25_25035

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25035


namespace distance_from_A_to_directrix_on_parabola_l25_25047

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25047


namespace sphere_surface_area_l25_25206

/-- The setup of the geometric configuration --/
structure configuration :=
  (P A B C D : Type) -- points P, A, B, C, D are distinct points on the sphere
  (plane_PAD : Type) -- plane of triangle PAD
  (plane_ABCD : Type) -- plane of rectangle ABCD
  (radius : Real)
  (on_sphere : P → A → B → C → D → Prop) -- points lie on the same spherical surface
  (perpendicular_planes : plane_PAD ⊥ plane_ABCD) -- planes are perpendicular
  (PA_eq_2 : PA = 2)
  (PD_eq_2 : PD = 2)
  (AB_eq_2 : AB = 2)
  (angle_APD_eq_60 : ∠APD = 60) -- angle ∠APD is 60 degrees
  
/-- Given these conditions for the configuration, the surface area of the sphere is 28π/3. --/
theorem sphere_surface_area (cfg : configuration) : 4 * π * (cfg.radius ^ 2) = 28 * π / 3 :=
sorry

end sphere_surface_area_l25_25206


namespace correct_option_among_relations_l25_25492

theorem correct_option_among_relations :
  ∅ ≠ {0} ∧
  ¬ (√2 ∈ ℚ) ∧
  {3, 5} = {5, 3} ∧
  {1} ⊆ {x ∈ ℝ | x^2 = x} :=
by
  sorry

end correct_option_among_relations_l25_25492


namespace habitat_limits_are_correct_l25_25818

-- Definitions of the conditions
def colonyA_doubling_days : ℕ := 22
def colonyB_tripling_days : ℕ := 30
def tripling_interval : ℕ := 2

-- Definitions to confirm they grow as described
def is_colonyA_habitat_limit_reached (days : ℕ) : Prop := days = colonyA_doubling_days
def is_colonyB_habitat_limit_reached (days : ℕ) : Prop := days = colonyB_tripling_days

-- Proof statement
theorem habitat_limits_are_correct :
  (is_colonyA_habitat_limit_reached colonyA_doubling_days) ∧ (is_colonyB_habitat_limit_reached colonyB_tripling_days) :=
by
  sorry

end habitat_limits_are_correct_l25_25818


namespace negation_of_existence_l25_25595

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem negation_of_existence:
  (∃ x : ℝ, log_base 3 x ≤ 0) ↔ ∀ x : ℝ, log_base 3 x < 0 :=
by
  sorry

end negation_of_existence_l25_25595


namespace distance_from_point_A_to_directrix_C_l25_25124

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25124


namespace second_number_removed_l25_25732

theorem second_number_removed (S : ℝ) (X : ℝ) (h1 : S / 50 = 38) (h2 : (S - 45 - X) / 48 = 37.5) : X = 55 :=
by
  sorry

end second_number_removed_l25_25732


namespace S_is_finite_l25_25678

def A : Set (ℕ → ℕ) := {f | ∀ n, f n ∈ {i : ℕ | i < 2018}}

def begins_with (M T : ℕ → ℕ) (n : ℕ) : Prop := 
  ∀ i < n, M i = T i

def S (s : ℕ → ℕ) : Set (ℕ → ℕ) := {t | ∃ n, begins_with t s n}

theorem S_is_finite (S : Set (ℕ → ℕ)) : 
  (∀ (M ∈ A), ∃! T ∈ S, ∃ n, begins_with M T n) → 
  S.finite := 
sorry

end S_is_finite_l25_25678


namespace coeff_x4_l25_25639

-- Define the binomial coefficient (combinatorial number)
def binom : ℕ → ℕ → ℕ 
| n, k => if k > n then 0 else Nat.choose n k

-- The polynomial expansion and coefficient function
noncomputable def coeff (p: ℕ → ℤ) (n: ℕ) : ℤ :=
  p n

-- Expression representing the polynomial expansion of (1 + x + x^2)(1 - x)^{10}
noncomputable def polynomial := (1 - x) ^ 10 + x * (1 - x) ^ 10 + x ^ 2 * (1 - x) ^ 10

-- Coefficient of x^4 in the given polynomial expression
theorem coeff_x4 : coeff (polynomial.coeff 4) = 135 := 
by 
  sorry

end coeff_x4_l25_25639


namespace difference_mean_median_scores_l25_25631

theorem difference_mean_median_scores :
  let total_students := 40
  let score_counts := [6, 8, 12, 10, 4]
  let scores := [60, 75, 85, 90, 100]
  let median := 85
  let mean := (60 * 6 + 75 * 8 + 85 * 12 + 90 * 10 + 100 * 4) / 40
  abs (mean - median) = 3 :=
by
  sorry

end difference_mean_median_scores_l25_25631


namespace polyhedron_space_diagonals_l25_25826

theorem polyhedron_space_diagonals (Q : Type) [convex_polyhedron Q] 
  (vertices : ℕ) (edges : ℕ) (faces : ℕ) (triangular_faces : ℕ) (pentagonal_faces : ℕ) :
  vertices = 30 → edges = 70 → faces = 40 → triangular_faces = 30 → pentagonal_faces = 10 →
  space_diagonals Q = 315 :=
by 
  intros h_vertices h_edges h_faces h_triangular_faces h_pentagonal_faces
  sorry

end polyhedron_space_diagonals_l25_25826


namespace initial_number_is_2008_l25_25460

theorem initial_number_is_2008 (x : ℕ) (S : set ℕ)
  (h0 : 2008 ∈ S)
  (h1 : ∀ x ∈ S, 2 * x + 1 ∈ S ∨ (x / (x + 2) : ℚ) ∈ S) :
  x = 2008 :=
begin
  sorry
end

end initial_number_is_2008_l25_25460


namespace original_acid_percentage_l25_25848

variables (a w : ℝ)

-- Conditions from the problem
def cond1 : Prop := a / (a + w + 2) = 0.18
def cond2 : Prop := (a + 2) / (a + w + 4) = 0.36

-- The Lean statement to prove
theorem original_acid_percentage (hc1 : cond1 a w) (hc2 : cond2 a w) : (a / (a + w)) * 100 = 19 :=
sorry

end original_acid_percentage_l25_25848


namespace solve_for_a_l25_25190

theorem solve_for_a (a : ℝ) (h : (a + 3)^(a + 1) = 1) : a = -2 ∨ a = -1 :=
by {
  -- proof here
  sorry
}

end solve_for_a_l25_25190


namespace air_conditioned_rooms_fraction_l25_25280

theorem air_conditioned_rooms_fraction (R A : ℝ) (h1 : 3/4 * R = 3/4 * R - 1/4 * R)
                                        (h2 : 2/3 * A = 2/3 * A - 1/3 * A)
                                        (h3 : 1/3 * A = 0.8 * 1/4 * R) :
    A / R = 3 / 5 :=
by
  -- Proof content goes here
  sorry

end air_conditioned_rooms_fraction_l25_25280


namespace _l25_25130

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25130


namespace inequality_proof_l25_25909

theorem inequality_proof 
  (p : ℝ) (q : ℝ) (n : ℕ)
  (a : Fin n → ℝ) (b : Fin n → ℝ)
  (hp : 1 ≤ p) 
  (hq : 0 < q)
  (h1 : ∀ i j, i ≤ j → a i ≥ a j) (ha : ∀ i, 0 < a i)
  (h2 : ∀ i j, i ≤ j → b i ≤ b j) (hb : ∀ i, 0 < b i) 
  (or_h : (∀ i j, i ≤ j → a i ≤ a j ∧ b i ≥ b j) ∨ ∀ i j, i ≤ j → a i ≥ a j ∧ b i ≤ b j) :
  (∑ i : Fin n, (a i)^p / (b i)^q) 
  ≥ n^(1 - p + q) * (∑ i : Fin n, a i)^p / (∑ i : Fin n, b i)^q := sorry

end inequality_proof_l25_25909


namespace largest_even_digit_multiple_of_9_under_1000_l25_25777

noncomputable def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

noncomputable def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else n.digits 10

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (digits n).sum

noncomputable def all_even_digits (n : ℕ) : Prop :=
  ∀ d ∈ digits n, is_even_digit d

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ n, n < 1000 ∧ all_even_digits n ∧ sum_of_digits n % 9 = 0 ∧ n = 360 :=
begin
  use 360,
  split,
  { exact nat.lt_succ_self 359 },
  split,
  { intros d hd,
    unfold digits at hd,
    rw list.mem_iff_exists_get at hd,
    rcases hd with ⟨k, hk⟩,
    repeat { rw list.get? },
    simp only [digits] at hk,
    have : k < 3 := (nat.lt_of_succ_lt_succ (list.length_le_of_lt_some hk)).trans_le (by norm_num),
    interval_cases k,
    { rw [hk, nat.digits, nat.digits_aux'],
      norm_num [is_even_digit] },
    { rw [hk, nat.digits, nat.digits_aux', nat.div_eq, nat.mod_eq_of_lt, nat.add_eq_zero_eq_zero_and_eq_zero, nat.zero_eq, eq_self_iff_true, true_and],
      norm_num [is_even_digit] },
    { rw [hk, nat.digits, nat.digits_aux', nat.div_eq, nat.mod_eq_of_lt, nat.add_eq_zero_eq_zero_and_eq_zero, nat.zero_eq, eq_self_iff_true, true_and],
      norm_num [is_even_digit] } },
  { simp only [sum_of_digits, digits],
    exact nat.digits_sum_eq 360 10,
    exact dec_trivial },
  { refl }
end

end largest_even_digit_multiple_of_9_under_1000_l25_25777


namespace distance_from_point_A_to_directrix_C_l25_25114

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25114


namespace inequality_l25_25281

theorem inequality (A B : ℝ) (n : ℕ) (hA : 0 ≤ A) (hB : 0 ≤ B) (hn : 1 ≤ n) : (A + B)^n ≤ 2^(n - 1) * (A^n + B^n) := 
  sorry

end inequality_l25_25281


namespace distance_from_A_to_directrix_l25_25013

noncomputable def distance_to_directrix (A : ℝ × ℝ) (p : ℝ) : ℝ :=
  A.1 + p / 2

theorem distance_from_A_to_directrix :
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  distance_to_directrix A p = 9 / 4 :=
by
  let A := (1 : ℝ, Real.sqrt 5)
  let p := 5 / 2
  exact Eq.refl (distance_to_directrix A p)

end distance_from_A_to_directrix_l25_25013


namespace line2_passes_through_fixed_point_l25_25197

noncomputable def line1 (k : ℝ) := { p : ℝ × ℝ | p.2 = k * (p.1 - 4) }

noncomputable def symmetric_point (pt sym : ℝ × ℝ) :=
  (2 * sym.1 - pt.1, 2 * sym.2 - pt.2)

theorem line2_passes_through_fixed_point (k : ℝ) (sym : ℝ × ℝ) :
  symmetric_point (4, 0) sym = (0, 2) → (∃ p : ℝ × ℝ, p ∈ line1 k ∧ p = (0, 2)) :=
by
  intro h
  use (4, 0)
  split
  . unfold line1
    simp
    sorry
  . exact h.symm

end line2_passes_through_fixed_point_l25_25197


namespace probability_no_real_roots_l25_25469

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*a*x + 4

def inRange (a : ℝ) : Prop := a ∈ set.Icc (-3 : ℝ) (5 : ℝ)

def noRealRoots (a : ℝ) : Prop := (4 * a^2 - 16) < 0

theorem probability_no_real_roots :
  (set.Icc (-3 : ℝ) (5 : ℝ)).measure (set.Icc (-2 : ℝ) (2 : ℝ)) / (set.Icc (-3 : ℝ) (5 : ℝ)).measure = 1/2 :=
by
  sorry

end probability_no_real_roots_l25_25469


namespace bugs_eat_total_flowers_l25_25210

theorem bugs_eat_total_flowers :
  let num_A := 3
  let num_B := 2
  let num_C := 1
  let flowers_A := 2
  let flowers_B := 3
  let flowers_C := 5
  let total := (num_A * flowers_A) + (num_B * flowers_B) + (num_C * flowers_C)
  total = 17 :=
by
  -- Applying given values to compute the total flowers eaten
  let num_A := 3
  let num_B := 2
  let num_C := 1
  let flowers_A := 2
  let flowers_B := 3
  let flowers_C := 5
  let total := (num_A * flowers_A) + (num_B * flowers_B) + (num_C * flowers_C)
  
  -- Verify the total is 17
  have h_total : total = 17 := 
    by
    sorry

  -- Proving the final result
  exact h_total

end bugs_eat_total_flowers_l25_25210


namespace posters_count_l25_25474

theorem posters_count (total_posts : ℕ) (small_ratio : ℚ) (medium_ratio : ℚ) (small_count medium_count : ℕ) :
  total_posts = 50 →
  small_ratio = 2/5 →
  medium_ratio = 1/2 →
  small_count = (small_ratio * total_posts : ℚ).toNat →
  medium_count = (medium_ratio * total_posts : ℚ).toNat →
  ∃ large_count : ℕ, large_count = total_posts - (small_count + medium_count) ∧ large_count = 5 :=
by
  intros h_total h_small_ratio h_medium_ratio h_small_count h_medium_count
  use total_posts - (small_count + medium_count)
  split
  . rw [h_total, h_small_count, h_medium_count]
    norm_num
  . rw [h_total]
    norm_num
    exact h_total.symm

end posters_count_l25_25474


namespace distance_to_directrix_l25_25083

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25083


namespace train_cross_time_l25_25479

noncomputable def length_of_train : ℝ := 100 -- Length of the train in meters
noncomputable def speed_in_kmh : ℝ := 100 -- Speed of the train in km/hour
noncomputable def speed_in_mps : ℝ := speed_in_kmh * 1000 / 3600 -- Speed of the train in meters per second

theorem train_cross_time : length_of_train / speed_in_mps ≈ 3.60 := 
by
  sorry

end train_cross_time_l25_25479


namespace max_tied_teams_l25_25215

theorem max_tied_teams (n : ℕ) (h_n : n = 8) (tournament : Fin n → Fin n → Prop)
  (h_symmetric : ∀ i j, tournament i j ↔ tournament j i)
  (h_antisymmetric : ∀ i j, tournament i j → ¬ tournament j i)
  (h_total : ∀ i j, i ≠ j → tournament i j ∨ tournament j i) :
  ∃ (k : ℕ), k = 7 ∧ ∀ (wins : Fin n → ℕ), 
  (∀ i, wins i = 4 → ∃! j, i ≠ j ∧ tournament i j) → True :=
by sorry

end max_tied_teams_l25_25215


namespace intersection_on_AD_l25_25698

variables {K L M N P Q : Type}
variables [linear_ordered_field K] [add_comm_group L] [module K L]
variables (A B C D : L) (K_on_AB : affine_combination A B K = K)
variables (L_on_BC : affine_combination B C L = L)
variables (M_on_CD : affine_combination C D M = M)
variables (K_on_AB : affine_extension A B K) 
variables (L_on_BC : affine_extension B C L)
variables (M_on_CD : affine_extension C D M)
variables (P_on_KL_AC : ∃ P : L, (affine_extension K L P) ∧ (affine_extension A C P))
variables (Q_on_LM_BD : ∃ Q : L, (affine_extension L M Q) ∧ (affine_extension B D Q))
variables (N_on_KQ_MP : ∃ N : L, (affine_extension K Q N) ∧ (affine_extension M P N))

-- The theorem statement
theorem intersection_on_AD : 
∃ N : L, (affine_extension K Q N) ∧ (affine_extension M P N) → (affine_extension A D N) :=
by
  sorry

end intersection_on_AD_l25_25698


namespace distance_to_directrix_l25_25085

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25085


namespace distance_from_A_to_directrix_on_parabola_l25_25049

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25049


namespace distance_from_A_to_directrix_l25_25150

noncomputable theory

def point_A : ℝ × ℝ := (1, real.sqrt 5)
def parabola_p : ℝ := 5 / 2
def directrix_x (p : ℝ) : ℝ := -p / 2

theorem distance_from_A_to_directrix : 
  let A := point_A in
  (abs (A.1 - directrix_x parabola_p)) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l25_25150


namespace sin_2_alpha_eq_neg_7_over_8_l25_25604

noncomputable def alpha : ℝ := sorry -- α in the interval (π/2, π)
axiom h1 : 2 * Real.cos (2 * alpha) = Real.sin (π / 4 - alpha)
axiom h2 : α > π / 2 ∧ α < π

theorem sin_2_alpha_eq_neg_7_over_8 : Real.sin (2 * alpha) = -(7 / 8) :=
by
  -- Proof details will be provided here
  sorry

end sin_2_alpha_eq_neg_7_over_8_l25_25604


namespace total_books_l25_25407

variable (Tim_books Mike_books : ℕ)
variable (h1 : Tim_books = 22)
variable (h2 : Mike_books = 20)

theorem total_books : Tim_books + Mike_books = 42 :=
by
  calc Tim_books + Mike_books = 22 + 20 : by rw [h1, h2]
                          ...          = 42 : by norm_num

end total_books_l25_25407


namespace sliding_window_sash_translation_l25_25427

def is_translation (movement : Type) : Prop := sorry

def ping_pong_ball_movement : Type := sorry
def sliding_window_sash_movement : Type := sorry
def kite_flight_movement : Type := sorry
def basketball_movement : Type := sorry

axiom ping_pong_not_translation : ¬ is_translation ping_pong_ball_movement
axiom kite_not_translation : ¬ is_translation kite_flight_movement
axiom basketball_not_translation : ¬ is_translation basketball_movement
axiom window_sash_is_translation : is_translation sliding_window_sash_movement

theorem sliding_window_sash_translation :
  is_translation sliding_window_sash_movement :=
by 
  exact window_sash_is_translation

end sliding_window_sash_translation_l25_25427


namespace factor_expression_equals_one_l25_25525

theorem factor_expression_equals_one (a b c : ℝ) :
  ((a^2 - b^2)^2 + (b^2 - c^2)^2 + (c^2 - a^2)^2) / ((a - b)^2 + (b - c)^2 + (c - a)^2) = 1 :=
by
  sorry

end factor_expression_equals_one_l25_25525


namespace sum_of_roots_tan_quadratic_eq_l25_25535
-- Import the comprehensive Mathlib library

-- Declare the necessary definitions as per the conditions
variable {x : ℝ}

def tan_quadratic_eq (x : ℝ) : Prop := tan x * tan x - 7 * tan x + 2 = 0
def interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 2 * Real.pi

-- State the theorem based on our proof problem
theorem sum_of_roots_tan_quadratic_eq :
  ∃ (α β : ℝ), tan_quadratic_eq α ∧ tan_quadratic_eq β ∧ interval α ∧ interval β ∧
              α + β + 2 * Real.pi = 3 * Real.pi - Real.arctan 7 :=
by
  sorry  -- Proof to be provided

end sum_of_roots_tan_quadratic_eq_l25_25535


namespace area_of_cos_2phi_total_area_of_curve_l25_25807

-- Define cosine squared function in Lean
noncomputable def cos_sq (x : ℝ) : ℝ := (Real.cos x) ^ 2

theorem area_of_cos_2phi :
  ∫ (φ : ℝ) in 0..(π / 4), (cos_sq (2 * φ)) = π / 8 :=
by
  sorry

theorem total_area_of_curve :
  4 * ∫ (φ : ℝ) in 0..(π / 4), cos_sq (2 * φ) = π / 2 :=
by
  have h := area_of_cos_2phi
  rw [← h]
  ring
  sorry

end area_of_cos_2phi_total_area_of_curve_l25_25807


namespace permutation_average_sum_l25_25921

theorem permutation_average_sum :
  let p := 286
  let q := 11
  p + q = 297 :=
by
  sorry

end permutation_average_sum_l25_25921


namespace area_of_triangle_ABF_l25_25768

open Real

/-- Let A = (0, 0), B = (√3, 0), E an interior point of square ABCD such that ∠ABE = 90°, 
    and F the intersection of BD and AE.
    Given the length of side AB is √3, we aim to find the area of ΔABF. -/
theorem area_of_triangle_ABF : 
  ∃ (A B E F : ℝ × ℝ), 
  A = (0, 0) ∧ 
  B = (sqrt 3, 0) ∧
  (∃ (C D : ℝ × ℝ), (C = (sqrt 3, sqrt 3) ∧ D = (0, sqrt 3))) ∧
  E.1 = E.2 ∧ 
  (E.1 > 0 ∧ E.1 < sqrt 3) ∧
  (E.1, E.2) ∈ interior (convex_hull (set.ABC A B (0, sqrt 3))) ∧
  (∃ F : ℝ × ℝ, F = (sqrt 3 / 2, sqrt 3 / 2)) ∧
  ∃ (area : ℝ), area = 3 / 4 :=
sorry

end area_of_triangle_ABF_l25_25768


namespace number_of_true_propositions_is_4_l25_25377

-- Define the conditions as propositions
def prop1 : Prop :=
  ∀ (A B C : Type) (f : A → B → C) (a₁ a₂ : A) (b₁ b₂ : B),
    (a₁ = a₂) ∧ (b₁ = b₂) → f a₁ b₁ = f a₂ b₂

def prop2 : Prop :=
  ∀ (α : Type) (angle : α) (bisector : α → α → α),
    (∃ β, bisector angle β = angle) → true

def prop3 : Prop :=
  ∀ (triangle : Type) (A B C : triangle → Prop),
    (A = B) → ∃ t, A t ∧ B t → true

def prop4 : Prop :=
  ∀ (number_line : Type) (R : number_line → Prop),
    ∀ x, R x → true

-- Define the goal
theorem number_of_true_propositions_is_4 :
  (prop1 ∧ prop2 ∧ prop3 ∧ prop4) → (4 = 4) :=
by
  intros h
  exact eq.refl 4
  sorry  -- Insert the proof here

end number_of_true_propositions_is_4_l25_25377


namespace distance_between_circles_in_triangle_l25_25253

theorem distance_between_circles_in_triangle
  {A B C : Type*}
  (AB AC BC : ℝ)
  (h₁ : AB = 13)
  (h₂ : AC = 14)
  (h₃ : BC = 15)
  (incircle_center excircle_center : ℝ)
  (incircle_property : true) -- Hypothetical property for illustration
  (excircle_property : true) -- Hypothetical property for illustration
  : dist incircle_center excircle_center = 5 * real.sqrt 13 := 
begin
  sorry 
end

end distance_between_circles_in_triangle_l25_25253


namespace difference_two_digit_primes_l25_25418

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_two_digit_prime : ℕ := 11

def largest_two_digit_prime : ℕ := 97

theorem difference_two_digit_primes :
  largest_two_digit_prime - smallest_two_digit_prime = 86 :=
by
  have h1: is_prime smallest_two_digit_prime := by sorry
  have h2: is_prime largest_two_digit_prime := by sorry
  show largest_two_digit_prime - smallest_two_digit_prime = 86, from by sorry

end difference_two_digit_primes_l25_25418


namespace distance_to_directrix_l25_25027

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25027


namespace solve_problem_l25_25603

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0)

def is_palindrome_primes (a b : ℕ) : Prop := 
  let ab := 10 * a + b in
  let ba := 10 * b + a in
  is_prime ab ∧ is_prime ba

def count_valid_palindromes : ℕ :=
  (List.filter (λ n, let a := n / 10; let b := n % 10 in is_palindrome_primes a b)
    [11, 13, 17, 31, 37, 71, 73, 79, 97]).length

theorem solve_problem : count_valid_palindromes = 9 := 
  by sorry

end solve_problem_l25_25603


namespace concurrency_of_lines_through_midpoints_of_triangle_l25_25708

theorem concurrency_of_lines_through_midpoints_of_triangle
  (A B C A' B' C' : Point)
  (hA' : midpoint A' B C)
  (hB' : midpoint B' A C)
  (hC' : midpoint C' A B)
  (L₁ : Line)
  (L₂ : Line)
  (L₃ : Line)
  (hL₁ : parallel_to_angle_bisector L₁ (angle_bisector A B C))
  (hL₂ : parallel_to_angle_bisector L₂ (angle_bisector B C A))
  (hL₃ : parallel_to_angle_bisector L₃ (angle_bisector C A B))
  :
  concurrent L₁ L₂ L₃ :=
sorry

end concurrency_of_lines_through_midpoints_of_triangle_l25_25708


namespace number_of_digits_l25_25516

open Real

-- Definitions based on conditions in the problem
def number: ℝ := 3^18 * 7^24

-- Statement to prove that number has 30 digits in base 10
theorem number_of_digits : Int.floor (log10 number) + 1 = 30 := by
  sorry

end number_of_digits_l25_25516


namespace sequence_constants_l25_25511

theorem sequence_constants (b : ℕ → ℕ) (h : ∀ k : ℕ, k = 2 * (k / 2) → (∀ n : ℕ, n < k / 2 → b (n + k / 2 * (k / 2)) = k)) :
  ∃ (p q r : ℕ), p = 2 ∧ q = 0 ∧ r = 0 ∧ (∀ m : ℕ, b m = p * (floor (real.sqrt (m + q))) + r) ∧ p + q + r = 2 :=
begin
  sorry
end

end sequence_constants_l25_25511


namespace pyramid_surface_area_l25_25370

variables (a : ℝ) -- side length of the hexagonal base and the lateral edge
noncomputable def total_surface_area_pyramid (a : ℝ) : ℝ :=
  (a^2 * (6 + Real.sqrt 2 + 3 * Real.sqrt 3)) / 2

theorem pyramid_surface_area (a : ℝ) (h : a > 0) :
  let S := total_surface_area_pyramid a
  in S = (a^2 * (6 + Real.sqrt 2 + 3 * Real.sqrt 3)) / 2 :=
by
  sorry

end pyramid_surface_area_l25_25370


namespace max_rectangle_area_in_circle_l25_25435

theorem max_rectangle_area_in_circle (r : ℝ) (h : r = 6) : 
  (let d := 2 * r in ((d^2) / 2) = 72) :=
by
  sorry

end max_rectangle_area_in_circle_l25_25435


namespace Diego_half_block_time_l25_25881

def problem_conditions_and_solution : Prop :=
  ∃ (D : ℕ), (3 * 60 + D * 60) / 2 = 240 ∧ D = 5

theorem Diego_half_block_time :
  problem_conditions_and_solution :=
by
  sorry

end Diego_half_block_time_l25_25881


namespace monster_feasting_l25_25830

noncomputable def number_of_people_on_first_ship : ℕ :=
  sorry

theorem monster_feasting (P : ℕ) 
  (h1 : ∀ n, 0 < n ∧ n ≤ 3 → people_on_ship n = 2^(n-1) * P)
  (h2 : people_on_ship 1 + people_on_ship 2 + people_on_ship 3 = 847) :
  P = 121 :=
sorry

end monster_feasting_l25_25830


namespace calculate_x_value_l25_25877

theorem calculate_x_value : 
  529 + 2 * 23 * 3 + 9 = 676 := 
by
  sorry

end calculate_x_value_l25_25877


namespace john_gallons_of_gas_l25_25660

theorem john_gallons_of_gas
  (rental_cost : ℝ)
  (gas_cost_per_gallon : ℝ)
  (mile_cost : ℝ)
  (miles_driven : ℝ)
  (total_cost : ℝ)
  (rental_cost_val : rental_cost = 150)
  (gas_cost_per_gallon_val : gas_cost_per_gallon = 3.50)
  (mile_cost_val : mile_cost = 0.50)
  (miles_driven_val : miles_driven = 320)
  (total_cost_val : total_cost = 338) :
  ∃ gallons_of_gas : ℝ, gallons_of_gas = 8 :=
by
  sorry

end john_gallons_of_gas_l25_25660


namespace cyclic_sum_inequality_l25_25946

-- Definitions of the given conditions
variables (a b c d : ℝ)
variables (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)
variables (sum_abc_d : a + b + c + d = 1)

-- Statement of the theorem
theorem cyclic_sum_inequality :
  ( ∑ x in ({a, b, c, d} : finset ℝ), x / (x^2 + 1) ) ≤ 16 / 17 := 
sorry

end cyclic_sum_inequality_l25_25946


namespace distance_from_point_to_parabola_directrix_l25_25107

noncomputable def point_on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop :=
  A.2^2 = 2 * p * A.1

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
  -p / 2

noncomputable def distance_to_directrix (A : ℝ × ℝ) (directrix : ℝ) : ℝ :=
  abs (A.1 - directrix)

theorem distance_from_point_to_parabola_directrix :
  ∀ (A : ℝ × ℝ) (p : ℝ), point_on_parabola A p → A = (1, real.sqrt 5) → distance_to_directrix A (parabola_directrix p) = 9 / 4 :=
by
  intro A p h1 h2
  rw [point_on_parabola] at h1
  sorry

end distance_from_point_to_parabola_directrix_l25_25107


namespace jared_annual_earnings_l25_25915

open Nat

noncomputable def diploma_monthly_pay : ℕ := 4000
noncomputable def months_in_year : ℕ := 12
noncomputable def multiplier : ℕ := 3

theorem jared_annual_earnings :
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  jared_annual_earnings = 144000 :=
by
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  exact sorry

end jared_annual_earnings_l25_25915


namespace positive_rational_as_sum_of_cubes_l25_25285

theorem positive_rational_as_sum_of_cubes (q : ℚ) (h_q_pos : q > 0) : 
  ∃ (a b c d : ℤ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ q = ((a^3 + b^3) / (c^3 + d^3)) :=
sorry

end positive_rational_as_sum_of_cubes_l25_25285


namespace distance_from_A_to_directrix_l25_25082

noncomputable def distance_from_point_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_A_to_directrix :
  let A := (1, Real.sqrt 5)
      p := 5 / 2
  in distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4 :=
by
  let A := (1, Real.sqrt 5)
  let p := 5 / 2
  show distance_from_point_to_directrix 1 (Real.sqrt 5) p = 9 / 4
  sorry

end distance_from_A_to_directrix_l25_25082


namespace hyperbola_eccentricity_sqrt2_l25_25593

noncomputable def hyperbola_eccentricity (a : ℝ) (h₁ : a > 0) (h₂ : 2 * a = 2) : ℝ :=
let b := 1 in
let c := Real.sqrt (a^2 + b^2) in
c / a

theorem hyperbola_eccentricity_sqrt2 : 
  ∀ (a : ℝ) (h₁ : a > 0) (h₂ : 2 * a = 2), hyperbola_eccentricity a h₁ h₂ = Real.sqrt 2 :=
by {
  intros,
  simp [hyperbola_eccentricity],
  rw [h₂, mul_one, Real.sqrt_add (pow_two_pos_of_pos h₁), Real.sqrt_one, add_self_eq_two],
  sorry
}

end hyperbola_eccentricity_sqrt2_l25_25593


namespace largest_even_digit_multiple_of_9_under_1000_l25_25775

theorem largest_even_digit_multiple_of_9_under_1000 : 
  ∃ n : ℕ, (∀ d ∈ Int.digits 10 n, d % 2 = 0) ∧ n < 1000 ∧ n % 9 = 0 ∧ 
  (∀ m : ℕ, (∀ d ∈ Int.digits 10 m, d % 2 = 0) ∧ m < 1000 ∧ m % 9 = 0 → m ≤ n) ∧ n = 864 :=
sorry

end largest_even_digit_multiple_of_9_under_1000_l25_25775


namespace range_of_b_over_a_l25_25995

theorem range_of_b_over_a 
  (a b m : ℝ)
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : m > 0) 
  (h4 : m ≠ 1)
  (h5 : ∀ x, m^(x+1) + 1 = 2 → (x, 2) lies on line 2ax - by + 14 = 0) 
  (h6 : (x - a + 1)^2 + (y + b - 2)^2 = 25 → (x, y) = (-1, 2) ) :
  let t := b / a in
  t ∈ set.Icc (3/4 : ℝ) (4/3 : ℝ) :=
sorry

end range_of_b_over_a_l25_25995


namespace minimize_expression_l25_25795

theorem minimize_expression : 
  \forall (op : ℝ → ℝ → ℝ), op = (λ a b, a - b) → op (-3) 1 = -4 :=
sorry

end minimize_expression_l25_25795


namespace final_ratio_l25_25211

-- Define initial conditions
def initial_milk_ratio : ℕ := 1
def initial_water_ratio : ℕ := 5
def total_parts : ℕ := initial_milk_ratio + initial_water_ratio
def can_capacity : ℕ := 8
def additional_milk : ℕ := 2
def initial_volume : ℕ := can_capacity - additional_milk
def part_volume : ℕ := initial_volume / total_parts

-- Define initial quantities
def initial_milk_quantity : ℕ := part_volume * initial_milk_ratio
def initial_water_quantity : ℕ := part_volume * initial_water_ratio

-- Define final quantities
def final_milk_quantity : ℕ := initial_milk_quantity + additional_milk
def final_water_quantity : ℕ := initial_water_quantity

-- Hypothesis: final ratios of milk and water
def final_ratio_of_milk_to_water : ℕ × ℕ := (final_milk_quantity, final_water_quantity)

-- Final ratio should be 3:5
theorem final_ratio (h : final_ratio_of_milk_to_water = (3, 5)) : final_ratio_of_milk_to_water = (3, 5) :=
  by
  sorry

end final_ratio_l25_25211


namespace number_of_ways_to_choose_same_color_socks_l25_25982

-- Definitions based on the problem conditions
def total_socks : ℕ := 12
def red_socks : ℕ := 5
def green_socks : ℕ := 3
def blue_socks : ℕ := 4

-- Theorem statement
theorem number_of_ways_to_choose_same_color_socks :
  (nat.choose red_socks 2) + (nat.choose green_socks 2) + (nat.choose blue_socks 2) = 19 :=
by {
  sorry
}

end number_of_ways_to_choose_same_color_socks_l25_25982


namespace number_of_ones_and_zeros_not_perfect_square_l25_25284

open Int

theorem number_of_ones_and_zeros_not_perfect_square (k : ℕ) : 
  let N := (10^k) * (10^300 - 1) / 9
  ¬ ∃ m : ℤ, m^2 = N :=
by
  sorry

end number_of_ones_and_zeros_not_perfect_square_l25_25284


namespace pete_keeps_one_red_marble_l25_25282

-- First, define the initial conditions
def initial_total_marbles : Nat := 10
def percentage_blue : Rat := 0.4
def blue_marbles : Nat := percentage_blue * initial_total_marbles
def red_marbles : Nat := initial_total_marbles - blue_marbles

def trade_blue_per_red : Nat := 2
def total_marbles_after_trade : Nat := 15

-- Now, state the theorem
theorem pete_keeps_one_red_marble : 
  (∃ x : Nat, (blue_marbles + (trade_blue_per_red * x) + (red_marbles - x) = total_marbles_after_trade) ∧ 
  ((red_marbles - x) = 1)) :=
by
  sorry

end pete_keeps_one_red_marble_l25_25282


namespace infinite_nat_N_polynomial_l25_25313

/-- The theorem states that for any integer k, 
there exists a natural number N = 4k^4 - 8k^2 + 2 such that the polynomial 
x^8 + Nx^4 + 1 can be factored into two fourth-degree polynomials with integer coefficients. -/
theorem infinite_nat_N_polynomial (k : ℤ) : ∃ (N : ℕ), N = 4 * k^4 - 8 * k^2 + 2 ∧ 
  ∃ (P Q : ℤ[X]), P.degree = 4 ∧ Q.degree = 4 ∧ (X^8 + C (↑N) * X^4 + 1) = P * Q :=
by {
  sorry
}

end infinite_nat_N_polynomial_l25_25313


namespace staff_member_final_price_l25_25451

variable (d P Q R x : ℝ)

-- Define the initial discount rate (15%)
def initial_discount := 0.85 * d

-- Define the first staff discount
def first_discount := initial_discount * (1 - P / 100)

-- Define the second staff discount
def second_discount := first_discount * (1 - Q / 100)

-- Define the third staff discount
def third_discount := second_discount * (1 - R / 100)

-- Define the final price after tax
def final_price := third_discount * (1 + x / 100)

theorem staff_member_final_price :
  final_price d P Q R x = 0.85 * d * (1 - P / 100) * (1 - Q / 100) * (1 - R / 100) * (1 + x / 100) := 
sorry

end staff_member_final_price_l25_25451


namespace sections_count_l25_25404

theorem sections_count (total_capacity : ℕ) (section_capacity : ℕ) (h_total : total_capacity = 984) (h_section : section_capacity = 246) : total_capacity / section_capacity = 4 :=
by
  rw [h_total, h_section]
  norm_num
  sorry

end sections_count_l25_25404


namespace josie_milk_discount_l25_25661

noncomputable def discount_on_milk (total_money : ℝ) (milk_price : ℝ) (bread_price : ℝ) (detergent_price : ℝ) (detergent_coupon : ℝ) (banana_price_per_pound : ℝ) (bananas_weight : ℝ) (remaining_money : ℝ): ℝ :=
  let total_cost_without_discount := bread_price + (detergent_price - detergent_coupon) + (banana_price_per_pound * bananas_weight) + milk_price in
  let total_cost_with_discount := total_money - remaining_money in
  total_cost_without_discount - total_cost_with_discount

theorem josie_milk_discount : discount_on_milk 20 4 3.5 10.25 1.25 0.75 2 4 = 2 := sorry

end josie_milk_discount_l25_25661


namespace infinite_natural_numbers_with_factored_polynomial_l25_25324

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l25_25324


namespace system_of_equations_a_solution_l25_25998

theorem system_of_equations_a_solution (x y a : ℝ) (h1 : 4 * x + y = a) (h2 : 3 * x + 4 * y^2 = 3 * a) (hx : x = 3) : a = 15 ∨ a = 9.75 :=
by
  sorry

end system_of_equations_a_solution_l25_25998


namespace sqrt_15_estimate_l25_25898

theorem sqrt_15_estimate : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 :=
by
  sorry

end sqrt_15_estimate_l25_25898


namespace perp_lines_of_perpendicular_planes_l25_25261

section PerpendicularLinesAndPlanes

variables (Point Line Plane : Type)
variables (perpendicular : Line → Plane → Prop) (perpendicular_lines : Line → Line → Prop)
variables (parallel : Line → Plane → Prop)
variables (m n : Line) (α β : Plane)

-- Assuming Lines m and n, and Planes α and β
axiom α_perp_β : perpendicular α β
axiom m_perp_α : perpendicular m α
axiom n_perp_β : perpendicular n β

-- Proving m ∠ n
theorem perp_lines_of_perpendicular_planes :
  perpendicular_lines m n :=
begin
  sorry
end

end PerpendicularLinesAndPlanes

end perp_lines_of_perpendicular_planes_l25_25261


namespace sum_of_numbers_l25_25622

theorem sum_of_numbers : 
  let a := 0.8
  let b := 1 / 2
  let c := 0.5
  a > 0.1 ∧ b > 0.1 ∧ c > 0.1 → (a + b + c) = 1.8 :=
by
  intro h
  have hb : real.to_nnreal (1 / 2) = 0.5 := by norm_num
  simp [hb] at h
  norm_num
  sorry

end sum_of_numbers_l25_25622


namespace bakery_item_count_l25_25999

theorem bakery_item_count : ∃ (s c : ℕ), 5 * s + 25 * c = 500 ∧ s + c = 12 := by
  sorry

end bakery_item_count_l25_25999


namespace distance_from_point_A_to_directrix_C_l25_25118

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25118


namespace geom_sequence_a1_l25_25576

noncomputable def a_n (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n-1)

theorem geom_sequence_a1 {a1 q : ℝ} 
  (h1 : 0 < q)
  (h2 : a_n a1 q 4 * a_n a1 q 8 = 2 * (a_n a1 q 5)^2)
  (h3 : a_n a1 q 2 = 1) :
  a1 = (Real.sqrt 2) / 2 :=
sorry

end geom_sequence_a1_l25_25576


namespace distance_from_A_to_directrix_of_parabola_l25_25061

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25061


namespace length_of_second_train_l25_25815

-- Defining the parameters
def first_train_length : ℝ := 280  -- in meters
def first_train_speed_kmh : ℝ := 120  -- in km/h
def second_train_speed_kmh : ℝ := 80  -- in km/h
def crossing_time : ℝ := 9  -- in seconds

-- Convert speeds from km/h to m/s
def first_train_speed_ms : ℝ := first_train_speed_kmh * 1000 / 3600
def second_train_speed_ms : ℝ := second_train_speed_kmh * 1000 / 3600

-- Relative speed when trains are moving in opposite directions
def relative_speed_ms : ℝ := first_train_speed_ms + second_train_speed_ms

-- Calculating total distance covered when crossing each other
def total_distance_covered : ℝ := relative_speed_ms * crossing_time

-- Defining the correct length of the second train
def second_train_length : ℝ := total_distance_covered - first_train_length

theorem length_of_second_train :
  second_train_length = 219.95 := by
  sorry

end length_of_second_train_l25_25815


namespace total_movies_shown_l25_25837

theorem total_movies_shown (screen1_movies : ℕ) (screen2_movies : ℕ) (screen3_movies : ℕ)
                          (screen4_movies : ℕ) (screen5_movies : ℕ) (screen6_movies : ℕ)
                          (h1 : screen1_movies = 3) (h2 : screen2_movies = 4) 
                          (h3 : screen3_movies = 2) (h4 : screen4_movies = 3) 
                          (h5 : screen5_movies = 5) (h6 : screen6_movies = 2) :
  screen1_movies + screen2_movies + screen3_movies + screen4_movies + screen5_movies + screen6_movies = 19 := 
by
  sorry

end total_movies_shown_l25_25837


namespace proof1_proof2_proof3_proof4_l25_25879

noncomputable def problem1 := (1 / 2) + (-2 / 3) + (4 / 5) + (-1 / 2) + (-1 / 3) = -1 / 5
noncomputable def problem2 := 2 * (-3)^3 - 4 * (-3) + 15 = -27
noncomputable def problem3 := ((1 / 8) - (1 / 3) + (7 / 6)) * (-48) = -46
noncomputable def problem4 := -2^4 - 32 / [(-2)^3 + 4] = -8

theorem proof1 : problem1 := 
by 
  sorry

theorem proof2 : problem2 := 
by 
  sorry

theorem proof3 : problem3 := 
by 
  sorry

theorem proof4 : problem4 := 
by 
  sorry

end proof1_proof2_proof3_proof4_l25_25879


namespace jared_annual_earnings_l25_25918

-- Defining conditions as constants
def diploma_pay : ℕ := 4000
def degree_multiplier : ℕ := 3
def months_in_year : ℕ := 12

-- Goal: Prove that Jared's annual earnings are $144,000
theorem jared_annual_earnings : diploma_pay * degree_multiplier * months_in_year = 144000 := by
  sorry

end jared_annual_earnings_l25_25918


namespace choose_televisions_l25_25922

theorem choose_televisions :
    let A := 4
    let B := 5
    let total_choices := (nat.choose A 1) * (nat.choose B 2) + (nat.choose A 2) * (nat.choose B 1)
    total_choices = 70 :=
by
  sorry

end choose_televisions_l25_25922


namespace percent_increase_equilateral_triangles_l25_25494

noncomputable def side_length (n : ℕ) : ℕ :=
  if n = 0 then 3 else 2 ^ n * 3

noncomputable def perimeter (n : ℕ) : ℕ :=
  3 * side_length n

noncomputable def percent_increase (initial : ℕ) (final : ℕ) : ℚ := 
  ((final - initial) / initial) * 100

theorem percent_increase_equilateral_triangles :
  percent_increase (perimeter 0) (perimeter 4) = 1500 := by
  sorry

end percent_increase_equilateral_triangles_l25_25494


namespace greatest_a_l25_25738

theorem greatest_a : ∃ (x : ℤ) (a : ℤ), (x^2 + a * x = -14) ∧ a > 0 ∧ a = 15 :=
by
  sorry

end greatest_a_l25_25738


namespace a₇_range_l25_25959

-- Arithmetic sequence is defined as monotonically increasing with common difference d
variable (d : ℝ) (a₁ a₇ a₁₀ : ℝ)
hypothesis (d_pos : d > 0)
hypothesis (seq_increasing : ∀ n m, n < m → a₁ + (n - 1) * d < a₁ + (m - 1) * d)

-- Given condition: a₁ + a₁₀ = 6
hypothesis (condition : a₁ + (a₁ + 9 * d) = 6)

-- Prove that a₇ > 3
theorem a₇_range : a₇ = a₁ + 6 * d → a₇ > 3 :=
by
  intro h₀
  have h₁: a₁ + (a₁ + 9 * d) = 6 := condition
  let a₁ := 3 - (9 / 2) * d
  let a₇ := a₁ + (6 * d)
  calc
    a₇ = 3 + (3 / 2) * d : by sorry
    ... > 3 : by sorry

end a₇_range_l25_25959


namespace distance_to_directrix_l25_25028

theorem distance_to_directrix (x y : ℝ) (p : ℝ) (h : y^2 = 2 * p * x) (hx : x = 1) (hy : y = Real.sqrt 5) :
  x + p / 2 = 9 / 4 :=
by
  rw [hx, hy] at h
  rw [Real.sqrt_eq_iff_sq_eq, sq] at h
  cases h with h1 h2
  · sorry

end distance_to_directrix_l25_25028


namespace passenger_drop_ratio_l25_25847

theorem passenger_drop_ratio (initial_passengers passengers_at_first passengers_at_second final_passengers x : ℕ)
  (h0 : initial_passengers = 288)
  (h1 : passengers_at_first = initial_passengers - (initial_passengers / 3) + 280)
  (h2 : passengers_at_second = passengers_at_first - x + 12)
  (h3 : final_passengers = 248)
  (h4 : passengers_at_second = final_passengers) :
  x / passengers_at_first = 1 / 2 :=
by
  sorry

end passenger_drop_ratio_l25_25847


namespace d_positive_neither_necessary_nor_sufficient_l25_25257

-- Definition of an arithmetic sequence and sum of its first n terms
def arithmetic_sequence (a₁ : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a₁ (n + 1) = a₁ n + d

def sum_of_arithmetic_sequence (a₁ : ℕ → ℝ) (n : ℕ) :=
  n * a₁ 0 + (n * (n - 1) / 2) * (a₁ 1 - a₁ 0)

-- Statement of the problem
theorem d_positive_neither_necessary_nor_sufficient
  (a₁ : ℕ → ℝ) (d : ℝ) (Sₙ Sₙ₊₁ : ℕ → ℝ)
  (h_seq : arithmetic_sequence a₁ d)
  (h_Sₙ : ∀ n, Sₙ n = sum_of_arithmetic_sequence a₁ n)
  (h_Sₙ₊₁ : ∀ n, Sₙ₊₁ n = sum_of_arithmetic_sequence a₁ (n + 1)):
  ¬ ((∀ n, Sₙ₊₁ n > Sₙ n → d > 0) ∧ (d > 0 → ∀ n, Sₙ₊₁ n > Sₙ n)) :=
sorry

end d_positive_neither_necessary_nor_sufficient_l25_25257


namespace exists_infinite_N_l25_25346

theorem exists_infinite_N (k : ℤ) : ∃ᶠ N : ℕ in at_top, 
  ∃ (A B : Polynomial ℤ), 
    (Polynomial.natDegree A = 4 ∧ Polynomial.natDegree B = 4) ∧
    (Polynomial.mul A B = Polynomial.X^8 + Polynomial.C N * Polynomial.X^4 + 1) :=
begin
  -- Proof goes here
  sorry
end

end exists_infinite_N_l25_25346


namespace switches_in_position_A_l25_25431

-- Defining the constants and conditions
def total_steps := 1000
def max_exponent := 8
def num_positions := 5

-- Calculate divisors count affecting each switch
def steps_per_position (x y z : ℕ) (hx : x ≤ max_exponent) (hy : y ≤ max_exponent) (hz : z ≤ max_exponent) : ℕ :=
  (9 - x) * (9 - y) * (9 - z)

-- Determine if a switch is back at position A
def in_position_A (x y z : ℕ) (hx : x ≤ max_exponent) (hy : y ≤ max_exponent) (hz : z ≤ max_exponent) : Prop :=
  steps_per_position x y z hx hy hz % num_positions = 0

-- Calculate how many switches are in position A
def count_position_A : ℕ :=
  ∑ x in Finset.range (max_exponent + 1), 
    ∑ y in Finset.range (max_exponent + 1),
      ∑ z in Finset.range (max_exponent + 1),
        if in_position_A x y z (Nat.le_succ x) (Nat.le_succ y) (Nat.le_succ z) then 1 else 0

theorem switches_in_position_A : count_position_A = 725 :=
by
  sorry

end switches_in_position_A_l25_25431


namespace ones_digit_of_highest_power_of_3_dividing_3_to_4_fac_l25_25905

noncomputable def highestPowerOf3 : ℕ := (81 / 3).floor + (81 / 9).floor + (81 / 27).floor + (81 / 81).floor

theorem ones_digit_of_highest_power_of_3_dividing_3_to_4_fac :
  (3^40 % 10) = 1 :=
by {
  have h1 : 81 = 3 ^ 4, by norm_num,
  have h2 : highestPowerOf3 = 40, by {
    unfold highestPowerOf3,
    norm_num,
  },
  rw ←h2,
  show (3^40 % 10) = 1,
  sorry,
}

end ones_digit_of_highest_power_of_3_dividing_3_to_4_fac_l25_25905


namespace problem1_problem2_l25_25561

variable {a : ℕ → ℝ}

/- Problem 1: Prove that the sequence \{a_{n+1} - 2a_n\} is a geometric sequence. -/
theorem problem1 (h : ∀ n, (∑ i in finset.range (n + 1), a i) = 2 * a n - 2^n) :
  ∃ r : ℝ, ∀ n, a (n + 1) - 2 * a n = r ^ n :=
sorry

/- Problem 2: Prove that the general formula for the sequence \{a_n\} is \(a_n = (n + 1) \times 2^{n - 1}\). -/
theorem problem2 (h : ∀ n, (∑ i in finset.range (n + 1), a i) = 2 * a n - 2^n) :
  ∀ n, a n = (n + 1) * 2^(n - 1) :=
sorry

end problem1_problem2_l25_25561


namespace population_net_increase_in_one_day_l25_25214

-- Define the problem conditions
def birth_rate : ℕ := 6 / 2  -- births per second
def death_rate : ℕ := 3 / 2  -- deaths per second
def seconds_in_a_day : ℕ := 60 * 60 * 24

-- Define the assertion we want to prove
theorem population_net_increase_in_one_day : 
  ( (birth_rate - death_rate) * seconds_in_a_day ) = 259200 := by
  -- Since 6/2 = 3 and 3/2 = 1.5 is not an integer in Lean, we use ratios directly
  sorry  -- Proof is not required

end population_net_increase_in_one_day_l25_25214


namespace z_is_real_z_is_pure_imaginary_z_is_zero_l25_25688

variable (a : ℝ)

def z : ℂ := (a ^ 2 + a - 2 : ℂ) + (a ^ 2 - 7 * a + 6 : ℂ) * Complex.I

theorem z_is_real (ha : a = 1 ∨ a = 6) : z a ∈ ℝ :=
sorry

theorem z_is_pure_imaginary (ha : a = -2) : 
  (z a).re = 0 ∧ (z a).im ≠ 0 :=
sorry

theorem z_is_zero (ha : a = 1) : z a = 0 :=
sorry

end z_is_real_z_is_pure_imaginary_z_is_zero_l25_25688


namespace water_added_l25_25493

theorem water_added (W : ℝ) : 
  let initial_volume := 80
      initial_jasmine := 0.10 * initial_volume
      added_jasmine := 8
      final_jasmine := initial_jasmine + added_jasmine
      final_percentage := 0.16
      total_volume := initial_volume + added_jasmine + W in 
  final_jasmine = final_percentage * total_volume ↔ W = 12 := 
by {
  dsimp [initial_volume, initial_jasmine, added_jasmine, final_jasmine, final_percentage, total_volume],
  sorry
}

end water_added_l25_25493


namespace minimum_k_satisfying_inequality_l25_25203

theorem minimum_k_satisfying_inequality :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) →
    ∃ (k : ℝ), k = sqrt 30 / 5 ∧ (sqrt x + sqrt y ≤ k * sqrt (5 * x + y)) :=
  sorry

end minimum_k_satisfying_inequality_l25_25203


namespace distance_from_A_to_directrix_on_parabola_l25_25041

-- Definition of the parabola and point A
def parabola (p : ℝ) := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
def A : ℝ × ℝ := (1, real.sqrt 5)

-- Distance from point A to the directrix
def distance_to_directrix (p : ℝ) := 1 + p / 2

theorem distance_from_A_to_directrix_on_parabola :
  (1, real.sqrt 5) ∈ parabola (5 / 2) →
  distance_to_directrix (5 / 2) = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_on_parabola_l25_25041


namespace number_of_boys_l25_25399

theorem number_of_boys (girls boys : ℕ) (total_books books_girls books_boys books_per_student : ℕ)
  (h1 : girls = 15)
  (h2 : total_books = 375)
  (h3 : books_girls = 225)
  (h4 : total_books = books_girls + books_boys)
  (h5 : books_girls = girls * books_per_student)
  (h6 : books_boys = boys * books_per_student)
  (h7 : books_per_student = 15) :
  boys = 10 :=
by
  sorry

end number_of_boys_l25_25399


namespace number_of_payment_combinations_l25_25846

-- Definitions based on conditions:
def PaymentMethods := {cash, alipay, weChat, unionPay}
def A_methods := {cash, alipay, weChat}  -- Customer A doesn't have UnionPay card
def B_methods := {cash}                  -- Customer B only has cash
def C_methods := PaymentMethods          -- Customer C can use any payment method
def D_methods := PaymentMethods          -- Customer D can use any payment method

-- The statement we need to prove:
theorem number_of_payment_combinations :
  (∃ (used_methods : Finset PaymentMethods),
   used_methods.card = 3 ∧ ∀ (A B C D : PaymentMethods),
    A ∈ A_methods ∧ B ∈ B_methods ∧ C ∈ C_methods ∧ D ∈ D_methods → 
    used_methods ⊆ {A, B, C, D} ∧
    {A, B, C, D}.card = 26) :=
  sorry

end number_of_payment_combinations_l25_25846


namespace _l25_25128

-- Definitions of conditions
def A : (ℝ × ℝ) := (1, real.sqrt 5)
def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def directrix (p : ℝ) : ℝ := - p / 2
noncomputable def distance_to_directrix (x p : ℝ) : ℝ := x + p / 2

-- The theorem to prove the distance from A to the directrix is 9/4
example : ∃ p : ℝ, (parabola_eq p 1 (real.sqrt 5)) ∧ (distance_to_directrix 1 p = 9 / 4) :=
by {
  use 5 / 2,
  split,
  { rw [parabola_eq, ←real.mul_right_inj' (ne_of_gt (real.sqrt_pos.2 zero_lt_five))],
    norm_num,
    linarith,},
  {
    rw distance_to_directrix,
    norm_num,
    linarith,
  }
}

end _l25_25128


namespace trajectory_of_point_ray_l25_25696

theorem trajectory_of_point_ray :
  ∀ (P : ℝ × ℝ), (|P.1| - |P.2| = 5) → ∀ (A B : ℝ × ℝ), A = (0, 1) → B = (0, 6) → 
  Trajectory (A, B, P) = ray := 
begin
  sorry
end

end trajectory_of_point_ray_l25_25696


namespace distinct_arrangements_of_elephant_l25_25980

theorem distinct_arrangements_of_elephant : 
  let n := 8
  let e_repeats := 2
  nat.fact n / nat.fact e_repeats = 20160 :=
by
  let n := 8
  let e_repeats := 2
  sorry

end distinct_arrangements_of_elephant_l25_25980


namespace part_one_part_two_l25_25944

/-- (1) For \(x > 0\), \(e^{x-1} \geqslant \ln(x) + 1\) -/
theorem part_one (x : ℝ) (h : x > 0) : Real.exp(x - 1) ≥ Real.log x + 1 :=
sorry

/-- (2) There can be at most two distinct lines tangent to both 
\( y = e^{x-1} \) and \( y = \ln(x) - 1 \) simultaneously. -/
theorem part_two : ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ 
  (∀ t : ℝ, t1 = t ∨ t2 = t) ∧
  (∀ t : ℝ, tangent_at f t ∧ tangent_at g t) :=
sorry

end part_one_part_two_l25_25944


namespace distances_sum_eq_l25_25599

noncomputable def radius_small := r
noncomputable def radius_large := R
axiom r_lt_R : r < R

-- P is a point on the smaller circle
axiom P_on_smaller_circle : ∀ (P : Point), (dist O P = r)

-- Line through P intersects larger circle at B and C
axiom line_through_P_intersects_BC : ∀ (P B C : Point), (dist O B = R) ∧ (dist O C = R)

-- Perpendicular from point P to BC intersects smaller circle at A
axiom perpendicular_at_P_intersects_A : ∀ (P A : Point), (dist P A = r)

theorem distances_sum_eq : 
  ∀ (P A B C : Point), 
    (dist O P = r) ∧ (dist O B = R) ∧ (dist O C = R) ∧ (dist P A = r) →
    (dist P A ^ 2 + dist P B ^ 2 + dist P C ^ 2 = 2 * (R ^ 2 + r ^ 2)) :=
by
  intros P A B C h
  sorry

end distances_sum_eq_l25_25599


namespace distance_from_A_to_directrix_of_parabola_l25_25053

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25053


namespace concyclic_M_F_D_E_l25_25650

-- Definitions of the given conditions
variables {A B C F D E M : Point}
variables {triangle_ABC : Triangle A B C}

-- Conditions
def AF_bisects_angle_BAC : Prop := bisects AF (angle B A C)
def BF_perpendicular_AF : Prop := perpendicular B F (line_through A F)
def circle_with_diameter_AC_intersects_BC_at_D_and_AF_at_E : Prop := 
  ∃ circ : Circle A C, intersects circ B C D ∧ intersects circ A F E
def M_is_midpoint_of_BC : Prop := midpoint M B C

-- The goal
theorem concyclic_M_F_D_E 
  (h1 : AF_bisects_angle_BAC)
  (h2 : BF_perpendicular_AF)
  (h3 : circle_with_diameter_AC_intersects_BC_at_D_and_AF_at_E)
  (h4 : M_is_midpoint_of_BC) : 
    cyclic_quad M F D E :=
sorry

end concyclic_M_F_D_E_l25_25650


namespace distance_from_A_to_directrix_of_parabola_l25_25054

theorem distance_from_A_to_directrix_of_parabola :
  (∃ p : ℝ, (√5)^2 = 2 * p * 1) → (1 + (5 / 4) = (9 / 4)) :=
by
  intro h
  rcases h with ⟨p, hp⟩
  sorry

end distance_from_A_to_directrix_of_parabola_l25_25054


namespace distance_to_directrix_l25_25089

theorem distance_to_directrix (x y p : ℝ) (h1 : (x, y) = (1, real.sqrt 5)) (h2 : y^2 = 2 * p * x) :
  (x + abs (-p / 2)) = 9 / 4 :=
by 
  simp [h1, h2]
  sorry

end distance_to_directrix_l25_25089


namespace diagonal_intersection_point_l25_25283

theorem diagonal_intersection_point
    (x1 y1 x2 y2 : ℝ) 
    (H1 : (x1, y1) = (2, -3) ∧ (x2, y2) = (12, 9)) :
    ( (x1 + x2) / 2, (y1 + y2) / 2) = (7, 3) :=
  by 
  intro x1 y1 x2 y2 H1
  have H2 : ( (x1 + x2) / 2, (y1 + y2) / 2) = ( (2 + 12) / 2, (-3 + 9) / 2) :=
    by 
    cases H1 with H1_left H1_right
    rw [H1_left, H1_right]
    simp
  rw [H2]
  simp
  sorry

end diagonal_intersection_point_l25_25283


namespace ab_value_l25_25195

theorem ab_value (a b : ℤ) (h : 48 * a * b = 65 * a * b) : a * b = 0 :=
  sorry

end ab_value_l25_25195


namespace cake_pieces_l25_25841

theorem cake_pieces (length_pan width_pan length_piece width_piece : ℕ)
  (h_pan : length_pan = 24 ∧ width_pan = 20)
  (h_piece : length_piece = 3 ∧ width_piece = 2) :
  (length_pan * width_pan) / (length_piece * width_piece) = 80 := by
  cases h_pan with l_pan_eq w_pan_eq
  cases h_piece with l_piece_eq w_piece_eq
  rw [l_pan_eq, w_pan_eq, l_piece_eq, w_piece_eq]
  norm_num
  sorry

end cake_pieces_l25_25841


namespace cube_root_of_25_approx_l25_25733

theorem cube_root_of_25_approx (x : ℝ) (hx : |x| < 1) :
  (1 + x)^(1/3) ≈ (1 + (1/3) * x) → (3 : ℝ) * (1 - (2 / 81)) = 2.926 :=
by
  sorry

end cube_root_of_25_approx_l25_25733


namespace exists_infinite_N_l25_25345

theorem exists_infinite_N (k : ℤ) : ∃ᶠ N : ℕ in at_top, 
  ∃ (A B : Polynomial ℤ), 
    (Polynomial.natDegree A = 4 ∧ Polynomial.natDegree B = 4) ∧
    (Polynomial.mul A B = Polynomial.X^8 + Polynomial.C N * Polynomial.X^4 + 1) :=
begin
  -- Proof goes here
  sorry
end

end exists_infinite_N_l25_25345


namespace lead_atom_ratio_l25_25731

noncomputable def ratio_of_lead_atoms (average_weight : ℝ) 
  (weight_206 : ℕ) (weight_207 : ℕ) (weight_208 : ℕ) 
  (number_206 : ℕ) (number_207 : ℕ) (number_208 : ℕ) : Prop :=
  average_weight = 207.2 ∧ 
  weight_206 = 206 ∧ 
  weight_207 = 207 ∧ 
  weight_208 = 208 ∧ 
  number_208 = number_206 + number_207 →
  (number_206 : ℚ) / (number_207 : ℚ) = 3 / 2 ∧
  (number_208 : ℚ) / (number_207 : ℚ) = 5 / 2

theorem lead_atom_ratio : ratio_of_lead_atoms 207.2 206 207 208 3 2 5 :=
by sorry

end lead_atom_ratio_l25_25731


namespace new_recipe_water_requirement_l25_25754

theorem new_recipe_water_requirement :
  ∀ (flour: ℕ) (water: ℕ) (sugar: ℕ),
    (flour = 11) → (water = 5) → (sugar = 2) →
    ∀ (new_sugar: ℕ),
      (new_sugar = 12) →
      let new_flour := 2 * flour,
          new_water := 2 * water in
      new_water * new_sugar = new_water → new_sugar * (new_water / 10) = 12 :=
by
  intros flour water sugar h_flour h_water h_sugar new_sugar h_new_sugar
  let new_flour := 2 * flour
  let new_water := 2 * water
  sorry

end new_recipe_water_requirement_l25_25754


namespace wall_number_of_bricks_l25_25410

theorem wall_number_of_bricks (x : ℝ) :
  (∃ x, 6 * ((x / 7) + (x / 11) - 12) = x) →  x = 179 :=
by
  sorry

end wall_number_of_bricks_l25_25410


namespace find_least_n_l25_25259

noncomputable def a : ℕ → ℤ
| 15 := 15
| n := 50 * a (n - 1) + 2 * n

theorem find_least_n (h : ∀ n > 15, a(n) = 50 * a(n - 1) + 2 * n) (N : ℕ) :
  ∃ n > 15, a(n) % 101 = 0 :=
sorry

end find_least_n_l25_25259


namespace distance_from_point_A_to_directrix_C_l25_25121

noncomputable theory

variables {p : ℝ} {A : ℝ × ℝ} {C : ℝ → ℝ → Prop}

def parabola := λ (y x : ℝ), y^2 = 2 * p * x
def point_on_parabola := A = (1, real.sqrt 5) ∧ parabola (real.sqrt 5) 1
def distance_to_directrix := 1 + p / 2 = 9 / 4

theorem distance_from_point_A_to_directrix_C (h : point_on_parabola) : distance_to_directrix :=
by sorry

end distance_from_point_A_to_directrix_C_l25_25121
