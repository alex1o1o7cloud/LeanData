import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Field
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Matrix
import Mathlib.Analysis.Calculus.AbsMaxMin
import Mathlib.Analysis.Calculus.Circle
import Mathlib.Analysis.SpecialFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue
import Mathlib.NumberTheory.Congruences
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.DiscreteUniform
import Mathlib.ProbabilityTheory.Independence
import Mathlib.ProbabilityTheory.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Trigonometry.Basic
import probability_theory.identification_disk

namespace range_of_t_l434_434458

def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + x
else Real.log x / Real.log 0.3

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x ≤ t^2 / 4 - t + 1) ↔ (t ≤ 1 ∨ 3 ≤ t) :=
by sorry

end range_of_t_l434_434458


namespace circle_diameter_problem_circle_diameter_l434_434289

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l434_434289


namespace iterative_average_difference_l434_434339

def numbers : List ℚ := [2, 3, 5, 7, 11]

def iterative_average (lst : List ℚ) : ℚ :=
  lst.foldl (λ avg n, (avg + n) / 2) (lst.headI)

theorem iterative_average_difference :
  ∀ perm_lst : List (List ℚ), perm_lst.permutations.contains numbers →
  (iterative_average (perm_lst.maximum (λ x y, x < y)) - iterative_average (perm_lst.minimum (λ x y, x < y))) = 4.6875 :=
sorry

end iterative_average_difference_l434_434339


namespace pages_per_brochure_l434_434321

-- Define the conditions
def single_page_spreads := 20
def double_page_spreads := 2 * single_page_spreads
def pages_per_double_spread := 2
def pages_from_single := single_page_spreads
def pages_from_double := double_page_spreads * pages_per_double_spread
def total_pages_from_spreads := pages_from_single + pages_from_double
def ads_per_4_pages := total_pages_from_spreads / 4
def total_ads_pages := ads_per_4_pages
def total_pages := total_pages_from_spreads + total_ads_pages
def brochures := 25

-- The theorem we want to prove
theorem pages_per_brochure : total_pages / brochures = 5 :=
by
  -- This is a placeholder for the actual proof
  sorry

end pages_per_brochure_l434_434321


namespace tan_sum_eq_tan_product_l434_434336

theorem tan_sum_eq_tan_product {α β γ : ℝ} 
  (h_sum : α + β + γ = π) : 
    Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ :=
by
  sorry

end tan_sum_eq_tan_product_l434_434336


namespace sequence_increasing_and_bounded_sequence_decreasing_and_bounded_l434_434110

def recurrence_relation (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x (n + 1) = (2 * x n ^ 2 - x n) / (3 * (x n - 2))

-- For the first problem
theorem sequence_increasing_and_bounded (x : ℕ → ℝ) (h_rec : ∀ n, recurrence_relation x n)
  (h_initial : 4 < x 0 ∧ x 0 < 5) : ∀ n, x n < x (n + 1) ∧ x (n + 1) < 5 :=
by
  sorry

-- For the second problem
theorem sequence_decreasing_and_bounded (x : ℕ → ℝ) (h_rec : ∀ n, recurrence_relation x n)
  (h_initial : x 0 > 5) : ∀ n, 5 < x (n + 1) ∧ x (n + 1) < x n :=
by
  sorry

end sequence_increasing_and_bounded_sequence_decreasing_and_bounded_l434_434110


namespace domain_of_function_l434_434811

theorem domain_of_function (x : ℝ) : 
  {x | ∃ k : ℤ, - (Real.pi / 3) + (2 : ℝ) * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 3) + (2 : ℝ) * k * Real.pi} :=
by
  -- Proof omitted
  sorry

end domain_of_function_l434_434811


namespace geometric_sequence_properties_l434_434429

theorem geometric_sequence_properties 
  (a_n : ℕ → ℝ) 
  (is_geo_seq : ∀ n, a_n (n + 1) = q * a_n n) 
  (h1 : |a_n 2 - a_n 1| = 2)
  (h2 : a_n 1 * a_n 2 * a_n 3 = 8) :
  let q := a_n 2 / a_n 1,
      S_5 := a_n 1 * (1 - q^5) / (1 - q) in
  q = 1 / 2 ∧ S_5 = 31 / 4 := 
by
  -- skipping the proof
  sorry

end geometric_sequence_properties_l434_434429


namespace sum_of_cn_l434_434015

theorem sum_of_cn (n : ℕ) (a b : ℕ → ℕ) (c : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, a n = 3^n) →
  (∀ n, b n = 2 * n + 1) →
  (c = λ n, a n * b n) →
  (S = λ n, ∑ i in finset.range n, c (i + 1)) →
  S n = n * 3^(n+1) :=
by
  intro ha hb hc hS
  sorry

end sum_of_cn_l434_434015


namespace train_pass_jogger_in_41_seconds_l434_434232

-- Definitions based on conditions
def jogger_speed_kmh := 9 -- in km/hr
def train_speed_kmh := 45 -- in km/hr
def initial_distance_jogger := 200 -- in meters
def train_length := 210 -- in meters

-- Converting speeds from km/hr to m/s
def kmh_to_ms (kmh: ℕ) : ℕ := (kmh * 1000) / 3600

def jogger_speed_ms := kmh_to_ms jogger_speed_kmh -- in m/s
def train_speed_ms := kmh_to_ms train_speed_kmh -- in m/s

-- Relative speed of the train with respect to the jogger
def relative_speed := train_speed_ms - jogger_speed_ms -- in m/s

-- Total distance to be covered by the train to pass the jogger
def total_distance := initial_distance_jogger + train_length -- in meters

-- Time taken to pass the jogger
def time_to_pass (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem train_pass_jogger_in_41_seconds : time_to_pass total_distance relative_speed = 41 :=
by
  sorry

end train_pass_jogger_in_41_seconds_l434_434232


namespace smallest_number_mod_conditions_l434_434729

theorem smallest_number_mod_conditions :
  ∃ b : ℕ, b > 0 ∧ b % 3 = 2 ∧ b % 5 = 3 ∧ ∀ n : ℕ, (n > 0 ∧ n % 3 = 2 ∧ n % 5 = 3) → n ≥ b :=
begin
  use 8,
  split,
  { linarith },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  intros n h,
  cases h with n_pos h_cond,
  cases h_cond with h_mod3 h_mod5,
  have h := chinese_remainder_theorem 3 5 (by norm_num) (by norm_num_ge)
  (λ _, by norm_num) (λ _ _, by norm_num),
  specialize h 2 3,
  rcases h ⟨h_mod3, h_mod5⟩ with ⟨m, rfl⟩,
  linarith,
end

end smallest_number_mod_conditions_l434_434729


namespace find_prime_pair_l434_434399

theorem find_prime_pair (p q : ℕ) [Fact p.prime] [Fact q.prime] :
  (∃ r : ℕ, r.prime ∧ r = 1 + ((p^q - q^p) / (p + q))) ↔ (p = 2 ∧ q = 5) :=
by sorry

end find_prime_pair_l434_434399


namespace range_of_a_l434_434455

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ (e^x + 1) * (a * x + 2 * a - 2) < 2) → a < 4 / 3 :=
by
  sorry

end range_of_a_l434_434455


namespace smallest_perfect_cube_with_tax_l434_434778

theorem smallest_perfect_cube_with_tax :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k ^ 3) ∧ 1.06 * (n / 1.06) = 148877 :=
sorry

end smallest_perfect_cube_with_tax_l434_434778


namespace simplify_fraction_l434_434169

-- Define the given variables and their assigned values.
variable (b : ℕ)
variable (b_eq : b = 2)

-- State the theorem we want to prove
theorem simplify_fraction (b : ℕ) (h : b = 2) : 
  15 * b ^ 4 / (75 * b ^ 3) = 2 / 5 :=
by
  -- sorry indicates where the proof would be written.
  sorry

end simplify_fraction_l434_434169


namespace equal_numbers_possible_l434_434142

noncomputable def circle_operations (n : ℕ) (α : ℝ) : Prop :=
  (n ≥ 3) ∧ (∃ k : ℤ, α = 2 * Real.cos (k * Real.pi / n))

-- Statement of the theorem
theorem equal_numbers_possible (n : ℕ) (α : ℝ) (h1 : n ≥ 3) (h2 : α > 0) :
  circle_operations n α ↔ ∃ k : ℤ, α = 2 * Real.cos (k * Real.pi / n) :=
sorry

end equal_numbers_possible_l434_434142


namespace median_after_adding_9_l434_434300

/-- A collection of six positive integers has a mean of 5, a unique mode of 4, and 
a median of 5. Prove that if a 9 is added to the collection, the new median is 5.0.
-/
theorem median_after_adding_9 (numbers : List ℕ) (h_length : numbers.length = 6)
  (h_positive : ∀ x ∈ numbers, 0 < x)
  (h_mean : (numbers.sum : ℚ) / numbers.length = 5)
  (h_mode : ∃! n, ∃ k, k > 1 ∧ k = numbers.count n ∧ n = 4)
  (h_median : numbers.sorted (· ≤ ·) ∧ (numbers.nth (3 - 1)).getOrElse 0 = 5 ∧ (numbers.nth (4 - 1)).getOrElse 0 = 5)
  : let new_numbers := 9 :: numbers,
    new_median := (new_numbers.sorted (· ≤ ·)).nth (new_numbers.length / 2) = (some 5) := 
by
  intro new_numbers new_median 
  sorry

end median_after_adding_9_l434_434300


namespace variance_of_score_is_0_16_l434_434755

-- Define the probabilities for the single free throw event
variable (p_hit : ℝ) (p_miss : ℝ)
variable (X : Type) [Probability X]
variable (X : Probability.Space.Discrete)

def P_X_1 : ℝ := 0.8
def P_X_0 : ℝ := 0.2

-- Define the random variable and its expected value
def score (x : X) : ℝ :=
  if x then 1 else 0

def E_X : ℝ := E (score p_hit p_miss : X → ℝ)

-- Define the expected value of X squared
def E_X2 : ℝ := E (λ x, (score p_hit p_miss : X → ℝ) x ^ 2)

-- Define the variance of X
def Var_X : ℝ := E_X2 - E_X^2

-- Lean 4 statement to prove the variance is 0.16
theorem variance_of_score_is_0_16 (h1 : P_X_1 = 0.8) (h2 : P_X_0 = 0.2) : Var_X = 0.16 :=
by {
  sorry
}

end variance_of_score_is_0_16_l434_434755


namespace triangle_area_is_168_l434_434490

def curve (x : ℝ) : ℝ :=
  (x - 4)^2 * (x + 3)

noncomputable def x_intercepts : set ℝ :=
  {x | curve x = 0}

noncomputable def y_intercept : ℝ :=
  curve 0

theorem triangle_area_is_168 :
  let base := 7 in
  let height := y_intercept in
  let area := (1 / 2) * base * height in
  area = 168 :=
by
  sorry

end triangle_area_is_168_l434_434490


namespace range_of_a_l434_434924

variable (a : ℝ)

def z1 : ℂ := 2 + 3 * Complex.I
def z2 : ℂ := (a - 2) + Complex.I

theorem range_of_a (h : Complex.abs (z1 - z2) < Complex.abs z1) : 1 < a ∧ a < 7 := by
  sorry

end range_of_a_l434_434924


namespace good_numbers_count_up_to_2019_l434_434517

def isGoodNumber(n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^3 + y^3 = z^n

theorem good_numbers_count_up_to_2019 : 
  (setOf isGoodNumber).count (Icc 1 2019) = 1346 :=
sorry

end good_numbers_count_up_to_2019_l434_434517


namespace area_of_triangle_l434_434478

theorem area_of_triangle :
  let f : ℝ → ℝ := λ x, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := (0, f 0)
  let base := (x_intercepts[0] - x_intercepts[1]).abs
  let height := y_intercept.2
  1 / 2 * base * height = 168 :=
by
  sorry

end area_of_triangle_l434_434478


namespace euler_formula_for_connected_planar_graph_l434_434140

-- Definition of a connected planar graph
structure ConnectedPlanarGraph (G : Type*) :=
  (vertices : ℕ) -- number of vertices
  (faces : ℕ)    -- number of faces
  (edges : ℕ)    -- number of edges
  -- An actual graph structure and properties (connectivity and planarity) could be more rigorously defined

-- Euler's formula for connected planar graphs
theorem euler_formula_for_connected_planar_graph (G : ConnectedPlanarGraph) :
  G.vertices + G.faces - G.edges = 2 :=
  sorry

end euler_formula_for_connected_planar_graph_l434_434140


namespace circle_diameter_l434_434277

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434277


namespace tangent_line_equation_l434_434906

theorem tangent_line_equation (r x_0 y_0 : ℝ) (hx0_y0_on_circle : x_0^2 + y_0^2 = r^2) :
  ∃ (x y : ℝ), x_0 * x + y_0 * y = r^2 :=
begin
  sorry,
end

end tangent_line_equation_l434_434906


namespace determine_function_l434_434141

open Nat

-- Define the set of strictly positive integers
def N_star := {n : ℕ // n > 0}

-- Define the main theorem
theorem determine_function :
  ∀ (f : N_star → N_star), 
  (∀ (m n : N_star), n + f m ∣ f n + n.val * f m.val) →
  (f = (λ n => ⟨n.val * n.val, by { cases n, simp, apply nat.mul_self_pos, exact n_property }⟩) ∨ 
  f = (λ n => ⟨1, by { cases n, simp, exact nat.zero_lt_one }⟩)) :=
sorry

end determine_function_l434_434141


namespace f1_in_M_f2_in_M_f3_not_in_M_l434_434749

def is_in_M (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, 0 ≤ x → 0 ≤ f x) ∧ (∀ s t : ℝ, 0 ≤ s → 0 ≤ t → f s + f t ≤ f (s + t))

def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := 2^x - 1
def f3 (x : ℝ) : ℝ := Real.log (x + 1)

theorem f1_in_M : is_in_M f1 :=
sorry

theorem f2_in_M : is_in_M f2 :=
sorry

theorem f3_not_in_M : ¬ is_in_M f3 :=
sorry

example : (is_in_M f1) ∧ (is_in_M f2) ∧ ¬ (is_in_M f3) :=
by
  exact ⟨f1_in_M, f2_in_M, f3_not_in_M⟩

end f1_in_M_f2_in_M_f3_not_in_M_l434_434749


namespace ob_length_max_l434_434697

noncomputable def length_OB_maximize_volume (P A B O H C: ℝ) : ℝ :=
  if PA_eq_4 : PA = 4 then
    if H_eq_midPA : H = (PA / 2) then
      -- Further geometric details to connect relationships assumed
      sorry
    else
      false
  else
    false

theorem ob_length_max :
  ∀ (P A B O H C: ℝ), 
  P ≠ A ∧ A ≠ B ∧ B ≠ O ∧ O ≠ H ∧ H ≠ C ∧ PA = 4 ∧ H = (PA / 2) →
  length_OB_maximize_volume P A B O H C = (2 * sqrt 6) / 3 :=
begin
  sorry
end

end ob_length_max_l434_434697


namespace rational_powers_imply_integers_l434_434026

theorem rational_powers_imply_integers (a b : ℚ) (h_distinct : a ≠ b)
  (h_infinitely_many_n : ∃ᶠ (n : ℕ) in Filter.atTop, (n * (a^n - b^n) : ℚ).den = 1) :
  ∃ (a_int b_int : ℤ), a = a_int ∧ b = b_int := 
sorry

end rational_powers_imply_integers_l434_434026


namespace exists_point_perimeter_greater_sum_distances_l434_434978

noncomputable theory

variables {A B C D M N P Q Z : Type} [MetricSpace Z]

def is_convex_quadrilateral (A B C D : Z) : Prop :=
  -- Definition of a convex quadrilateral, omitted for brevity
  
def sum_distances_to_vertices (Z : Z) (A B C D : Z) : ℝ :=
  (dist Z A) + (dist Z B) + (dist Z C) + (dist Z D)

def sum_distances_to_marked_points (Z : Z) (M N P Q : Z) : ℝ :=
  (dist Z M) + (dist Z N) + (dist Z P) + (dist Z Q)

theorem exists_point_perimeter_greater_sum_distances 
  (A B C D M N P Q : Z) 
  (h1 : is_convex_quadrilateral A B C D) 
  (h2 : inside_quadrilateral M A B C D) 
  (h3 : inside_quadrilateral N A B C D) 
  (h4 : inside_quadrilateral P A B C D) 
  (h5 : inside_quadrilateral Q A B C D) : 
  ∃ Z : Z, (on_perimeter Z A B C D) ∧ 
    (sum_distances_to_vertices Z A B C D > sum_distances_to_marked_points Z M N P Q) :=
sorry

end exists_point_perimeter_greater_sum_distances_l434_434978


namespace largest_integer_less_log_sum_l434_434723

theorem largest_integer_less_log_sum :
  let s := ∑ k in Finset.range 3000, log10 ((k + 2) / (k + 1))
  in ⌊s⌋ = 3 :=
by
  let s := ∑ k in Finset.range 3000, log10 ((k + 2) / (k + 1))
  have h_sum : s = log10 3001 := sorry
  have h_range : 3 < log10 3001 := sorry
  have h_range_2 : log10 3001 < 4 := sorry
  rw h_sum
  exact floor_eq_iff.2 ⟨h_range, h_range_2⟩

end largest_integer_less_log_sum_l434_434723


namespace minerals_now_l434_434988

def minerals_yesterday (M : ℕ) : Prop := (M / 2 = 21)

theorem minerals_now (M : ℕ) (H : minerals_yesterday M) : (M + 6 = 48) :=
by 
  unfold minerals_yesterday at H
  sorry

end minerals_now_l434_434988


namespace Randy_trip_length_l434_434643

theorem Randy_trip_length :
  ∃ y : ℝ, (y / 4 + 30 + y / 3 = y) ∧ (y = 72) :=
begin
  sorry
end

end Randy_trip_length_l434_434643


namespace find_division_point_l434_434900

noncomputable def pressure_integral (a b : ℝ) : ℝ :=
  ∫ x in a..b, x

theorem find_division_point (c : ℝ) :
  (∫ x in 0..c, x) = (∫ x in c..6, x) → c = 3 * Real.sqrt 2 :=
by
  have h₀ : ∫ x in 0..c, x = c^2 / 2 := by sorry
  have h₁ : ∫ x in c..6, x = 18 - c^2 / 2 := by sorry
  intro h
  rw [h₀, h₁] at h
  linarith

end find_division_point_l434_434900


namespace sum_coordinates_eq_l434_434628

theorem sum_coordinates_eq : ∀ (A B : ℝ × ℝ), A = (0, 0) → (∃ x : ℝ, B = (x, 5)) → 
  (∀ x : ℝ, B = (x, 5) → (5 - 0) / (x - 0) = 3 / 4) → 
  let x := 20 / 3 in
  (x + 5 = 35 / 3) :=
by
  intros A B hA hB hslope
  have hx : ∃ x, B = (x, 5) := hB
  cases hx with x hx_def
  rw hx_def at hslope
  have : x = 20 / 3 := by
    rw hx_def at hslope
    field_simp at hslope
    linarith
  rw [this, hx_def]
  norm_num

end sum_coordinates_eq_l434_434628


namespace part1_part2_l434_434467

-- Define vectors and given conditions
def a : ℝ × ℝ := (1, Real.sqrt 3)
def b_magnitude : ℝ := 3
def dot_product_condition : ℝ := (2 * a.1 - 3 * b.1) * (2 * a.1 + b.1) + (2 * a.2 - 3 * b.2) * (2 * a.2 + b.2)

-- Puprove that |a + b| = sqrt(7)
theorem part1 (h1 : |a| = 2) (h2 : |b| = b_magnitude) (h3 : dot_product_condition = 1) :
  |a + b| = Real.sqrt 7 := 
sorry

-- Prove the projection of 2a - b on a
theorem part2 (h1 : |a| = 2) (h2 : |b| = b_magnitude) (h3 : dot_product_condition = 1) :
  proj (2 * a - b) a = ((11/4) * a.1, (11/4) * a.2) := 
sorry

end part1_part2_l434_434467


namespace option_C_is_not_a_fraction_l434_434741

-- Definitions for the numerical conditions
def A := - (1 / 2)
def B := 22 / 7
def C := Real.pi / 2
def D := 80 / 100

-- Proof statement
theorem option_C_is_not_a_fraction (hA : A = - (1 / 2))
                                    (hB : B = 22 / 7)
                                    (hC : C = Real.pi / 2)
                                    (hD : D = 80 / 100) :
                                   ¬ (∃ (p q : ℤ), q ≠ 0 ∧ C = p / q) :=
by { sorry }

end option_C_is_not_a_fraction_l434_434741


namespace set_union_unique_element_l434_434875

noncomputable theory

variable (A B : Set ℕ) (a : ℕ)

def A_def : Set ℕ := {1, 4}
def B_def : Set ℕ := {0, 1, a}
def union_def : Set ℕ := {0, 1, 4}

theorem set_union_unique_element (hA : A = A_def) (hB : B = B_def) (h_union : A ∪ B = union_def) : a = 4 :=
by
  sorry

end set_union_unique_element_l434_434875


namespace find_val_of_a_l434_434074

noncomputable def find_a (a : ℝ) : Prop :=
  (∃ (A B : ℝ × ℝ), A ≠ B ∧ (A.1 = a + sqrt 2 * cos (π / 6) ∧ A.2 = sqrt 2 * sin (π / 6)) ∧
  (B.1 = a + sqrt 2 * cos (π / 6) ∧ B.2 = sqrt 2 * sin (π / 6)) ∧
  dist A B = 2) ∧ a > 0

theorem find_val_of_a : (∃ a, find_a a) ↔ a = 2 :=
by
  sorry

end find_val_of_a_l434_434074


namespace find_a5_l434_434014

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

-- Given conditions
variable (a : ℕ → ℝ)
variable (h_arith : arithmetic_sequence a)
variable (h_a1 : a 0 = 2)
variable (h_sum : a 1 + a 3 = 8)

-- The target question
theorem find_a5 : a 4 = 6 :=
by
  sorry

end find_a5_l434_434014


namespace bologna_sandwiches_l434_434355

variable (C B P : ℕ)

theorem bologna_sandwiches (h1 : C = 1) (h2 : B = 7) (h3 : P = 8)
                          (h4 : C + B + P = 16) (h5 : 80 / 16 = 5) :
                          B * 5 = 35 :=
by
  -- omit the proof part
  sorry

end bologna_sandwiches_l434_434355


namespace triangle_ABI_ratio_l434_434961

theorem triangle_ABI_ratio:
  ∀ (AC BC : ℝ) (hAC : AC = 15) (hBC : BC = 20),
  let AB := Real.sqrt (AC^2 + BC^2) in
  let CD :=  (AC * BC) / AB in
  let r := CD / 2 in
  let x := Real.sqrt (r^2 + (Real.sqrt (r^2 + (AB/2)^2) - r)^2) in
  let P := 2 * x + AB in
  (P / AB = 177 / 100) ∧ (177 + 100 = 277) :=
by
  intros AC BC hAC hBC
  let AB := Real.sqrt (AC^2 + BC^2)
  let CD := (AC * BC) / AB
  let r := CD / 2
  let x := Real.sqrt (r^2 + (Real.sqrt (r^2 + (AB/2)^2) - r)^2)
  let P := 2 * x + AB
  have ratio : P / AB = 177 / 100 := sorry
  exact ⟨ratio, rfl⟩


end triangle_ABI_ratio_l434_434961


namespace smallest_positive_integer_remainder_l434_434731

theorem smallest_positive_integer_remainder : ∃ a : ℕ, 
  (a ≡ 2 [MOD 3]) ∧ (a ≡ 3 [MOD 5]) ∧ (a = 8) := 
by
  use 8
  split
  · exact Nat.modeq.symm (Nat.modeq.modeq_of_dvd _ (by norm_num))
  · split
    · exact Nat.modeq.symm (Nat.modeq.modeq_of_dvd _ (by norm_num))
    · rfl
  sorry  -- The detailed steps of the proof are omitted as per the instructions

end smallest_positive_integer_remainder_l434_434731


namespace hyperbola_equation_l434_434596

def equation_of_hyperbola (x y a b : ℝ) : Prop :=
  (x^2 / a^2 - y^2 / b^2 = 1) ∧ a > 0 ∧ b > 0

def line_passing_through_focus (x y b: ℝ) : Prop :=
  let l := λ x, -b * (x - 1) in
  l(0) = b ∧ l(1) = 0

def asymptotes (x y a b : ℝ) : Prop :=
  (∃ m : ℝ, m = b / a ∧ y = m * x) ∨ (∃ m : ℝ, m = - b / a ∧ y = m * x)

theorem hyperbola_equation {x y a b : ℝ} (h1 : equation_of_hyperbola x y a b)
    (h2 : line_passing_through_focus x y b)
    (h3 : ∀ (a b : ℝ), asymptotes x y a b → ((b/a = -b) ∨ (b/a * -b = -1 → x^2 - y^2 = 1))) :
  a = 1 ∧ b = 1 → x^2 - y^2 = 1 := sorry

end hyperbola_equation_l434_434596


namespace fraction_simplification_l434_434370

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l434_434370


namespace range_of_omega_l434_434460

    theorem range_of_omega 
      (ω : ℝ) (hω : 0 < ω) 
      (h : ∃ s : set ℝ, (∀ x ∈ s, 0 < x ∧ x < π) ∧ (∀ x ∈ s, f x = -1) ∧ (s.card = 4)) :
      ω ∈ set.Ioc (7/2 : ℝ) (25/6 : ℝ) :=
    by
      sorry

    noncomputable def f (x : ℝ) : ℝ :=
      sin (ω * x) - sqrt 3 * cos (ω * x)
    
end range_of_omega_l434_434460


namespace count_ordered_pairs_l434_434929

theorem count_ordered_pairs (a b : ℤ) :
  (a^2 + b^2 < 25) ∧ (a^2 + b^2 < 10 * a) ∧ (a^2 + b^2 < 10 * b) → 
  ∃ (count : ℕ), count = 15 :=
begin
  sorry
end

end count_ordered_pairs_l434_434929


namespace find_multiplying_number_l434_434318

variable (a b : ℤ)

theorem find_multiplying_number (h : a^2 * b = 3 * (4 * a + 2)) (ha : a = 1) :
  b = 18 := by
  sorry

end find_multiplying_number_l434_434318


namespace find_a_in_triangle_l434_434947

noncomputable def cosine_rule (A : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  real.sqrt (b * b + c * c - 2 * b * c * real.cos A)

theorem find_a_in_triangle (A : ℝ) (b : ℝ) (c : ℝ) (hA : A = real.pi / 3) (hb : b = 2) (hc : c = 1) :
  cosine_rule A b c = real.sqrt 3 :=
by
  rw [hA, hb, hc]
  sorry

end find_a_in_triangle_l434_434947


namespace area_triangle_ABC_is_l434_434108

variables {A B C D H K : Type*} [InnerProductSpace ℝ (EuclideanSpace ℝ [A, B, C, D, H, K])]

-- Define the given conditions
def AB = (dist A B : ℝ)
def BC = (dist B C : ℝ)
def AC = (dist A C : ℝ)
def AD = (dist A D : ℝ)
def AH = (dist A H : ℝ)
def CD = (dist C D : ℝ)
def BD = (dist B D : ℝ)
def AK = (dist A K : ℝ)

-- Assume the geometric conditions
axiom eq_AB_BC : AB = BC
axiom eq_AB_AC : AB = AC
axiom eq_AB_AD : AB = AD
axiom eq_AC_AD : AC = AD
axiom eq_AC_BC : AC = BC
axiom perp_AH_CD : isPerpendicular AH CD
axiom perp_KC_BC : isPerpendicular KC BC
axiom AK_intersect_H : intersectsAt AK H
axiom KC_intersect_H : intersectsAt KC H

-- Now the theorem to show
theorem area_triangle_ABC_is :
  S_triangle A B C = (sqrt 3 / 4) * (AK * BD) :=
by sorry

end area_triangle_ABC_is_l434_434108


namespace range_of_m_l434_434593

def p (m : ℝ) : Prop := ∀ x, (x > 1) → (2 / (x - m) < 2 / (x + 1))
def q (m : ℝ) : Prop := ∀ a, (-1 ≤ a ∧ a ≤ 1) → (m^2 + 5 * m - 3 ≥ (real.sqrt (a^2 + 8)))

theorem range_of_m (m : ℝ) (h1 : ¬ p m) (h2 : q m) : m > 1 :=
by {
  -- proof goes here
  sorry
}

end range_of_m_l434_434593


namespace sum_of_consecutive_numbers_l434_434197

/-- The sum of four consecutive numbers where the greatest of them is 27 is 102. -/
theorem sum_of_consecutive_numbers (a b c d : ℕ) (h : [a, b, c, d].sorted (≤)) (h1 : d = 27) (h2 : d = c + 1) (h3 : c = b + 1) (h4 : b = a + 1) :
  a + b + c + d = 102 :=
sorry

end sum_of_consecutive_numbers_l434_434197


namespace smallest_four_digit_number_l434_434410

def satisfies_congruences (x : ℕ) : Prop :=
  (9 * x ≡ 27 [MOD 15]) ∧
  (3 * x + 15 ≡ 21 [MOD 8]) ∧
  (-3 * x + 4 ≡ 2 * x + 5 [MOD 16])

theorem smallest_four_digit_number : ∃ x : ℕ, satisfies_congruences x ∧ 1000 ≤ x < 10000 ∧ x = 1053 :=
by
  sorry

end smallest_four_digit_number_l434_434410


namespace unique_a_one_root_l434_434847

theorem unique_a_one_root (k : ℝ) :
  (∃ a : ℝ, a ≠ 0 ∧ (x : ℝ → x^2 - (a^3 + 1 / a^3) * x + k = 0) ∧ 
             (∀ x₁ x₂, x₁ ≠ x₂ → ¬ (x₁ = x₂))) → k = 1 :=
by
  sorry

end unique_a_one_root_l434_434847


namespace number_of_true_statements_l434_434046

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

def statement1 := ∀ x, g (Real.pi / 6 - x) = - g (Real.pi / 6 + x)
def statement2 := ∀ x, g (Real.pi / 6 - x) = g (Real.pi / 6 + x)
def statement3 := ∀ x y, (Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ 5 * Real.pi / 12) → g x < g y

theorem number_of_true_statements : (if statement1 ∧ ¬ statement2 ∧ statement3 then 2 else 0) = 2 :=
sorry

end number_of_true_statements_l434_434046


namespace shuffleboard_total_games_l434_434564

theorem shuffleboard_total_games
    (jerry_wins : ℕ)
    (dave_wins : ℕ)
    (ken_wins : ℕ)
    (h1 : jerry_wins = 7)
    (h2 : dave_wins = jerry_wins + 3)
    (h3 : ken_wins = dave_wins + 5) :
    jerry_wins + dave_wins + ken_wins = 32 := 
by
  sorry

end shuffleboard_total_games_l434_434564


namespace isosceles_triangle_angle_sum_l434_434787

theorem isosceles_triangle_angle_sum (y : ℕ) (a : ℕ) (b : ℕ) 
  (h_isosceles : a = b ∨ a = y ∨ b = y)
  (h_sum : a + b + y = 180) :
  a = 80 → b = 80 → y = 50 ∨ y = 20 ∨ y = 80 → y + y + y = 150 :=
by
  sorry

end isosceles_triangle_angle_sum_l434_434787


namespace minimum_value_at_k_eq_1_l434_434915

def quadratic_function (k : ℤ) (x : ℝ) := x^2 - (2 * k + 3) * x + 2 * k^2 - k - 3

theorem minimum_value_at_k_eq_1 :
  let k := 1 in
  let x_vertex := 5 / 2 in
  quadratic_function k x_vertex = -33 / 4 :=
by
  sorry

end minimum_value_at_k_eq_1_l434_434915


namespace horizontal_asymptote_greater_than_one_l434_434388

noncomputable def horizontal_asymptote (a x : ℝ) : ℝ :=
(a * x^5 - 3 * x^3 + 7) / (x^5 - 2 * x^3 + x)

theorem horizontal_asymptote_greater_than_one (a : ℝ) (ha : a > 1) : 
  ∃ L : ℝ, (filter.tendsto (λ x, horizontal_asymptote a x) filter.at_top (nhds L)) ∧ L = a ∧ a > 1 :=
sorry

end horizontal_asymptote_greater_than_one_l434_434388


namespace determine_P_l434_434181

theorem determine_P (P Q R S T U : ℕ)
  (digits : {1, 2, 3, 4, 5, 6} = {P, Q, R, S, T, U})
  (h1 : (100 * P + 10 * Q + R) % 4 = 0)
  (h2 : (100 * Q + 10 * R + S) % 6 = 0)
  (h3 : (100 * R + 10 * S + T) % 3 = 0) :
  P = 5 :=
by
  sorry

end determine_P_l434_434181


namespace molecularWeightProof_molecularWeightOneMole_l434_434223

-- Define variables for molecular weights
def molecularWeightCai2 (m : ℕ) : ℝ := 294 * m

-- Define the main theorem
theorem molecularWeightProof : molecularWeightCai2 5 = 1470 :=
by
  -- Express the given condition
  have h : 294 * 5 = 1470 := by norm_num
  exact h

-- State the molecular weight of one mole as a corollary
theorem molecularWeightOneMole : molecularWeightCai2 1 = 294 :=
by
  -- It follows directly from the definition and the proven theorem
  exact rfl

end molecularWeightProof_molecularWeightOneMole_l434_434223


namespace monochromatic_tree_exists_l434_434093

theorem monochromatic_tree_exists (n : ℕ) (h : n ≥ 3) (P : Fin n → ℝ × ℝ)
  (hc : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ¬ collinear {P i, P j, P k})
  (C : Fin n → Fin n → Bool) :
  ∃ T : Finset (Fin n × Fin n), T.card = n - 1 ∧ (∀ {i j : Fin n}, (i, j) ∈ T → C i j = true) ∧ ∀ {i j k l : Fin n}, (i, j) ∈ T → (k, l) ∈ T → (i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l) → ¬ intersects_segment (P i, P j) (P k, P l) :=
by sorry

end monochromatic_tree_exists_l434_434093


namespace mara_correct_answers_l434_434602

theorem mara_correct_answers :
  let math_total    := 30
  let science_total := 20
  let history_total := 50
  let math_percent  := 0.85
  let science_percent := 0.75
  let history_percent := 0.65
  let math_correct  := math_percent * math_total
  let science_correct := science_percent * science_total
  let history_correct := history_percent * history_total
  let total_correct := math_correct + science_correct + history_correct
  let total_problems := math_total + science_total + history_total
  let overall_percent := total_correct / total_problems
  overall_percent = 0.73 :=
by
  sorry

end mara_correct_answers_l434_434602


namespace range_of_f_when_m_is_one_solution_set_of_inequality_l434_434598

-- Define the function f with m as a variable
def f (x m : ℝ) := |x+1| - m * |x-2|

-- Prove the range of f(x) is [-3, 3] when m = 1
theorem range_of_f_when_m_is_one : (∀ x : ℝ, f x (1 : ℝ) ∈ Icc (-3 : ℝ) 3) := by
  sorry

-- Define a new function f' for the case when m = -1
def f' (x : ℝ) := |x+1| + |x-2|

-- Prove the solution set of f'(x) > 3x is (-∞, 1)
theorem solution_set_of_inequality : (∀ x : ℝ, f' x > 3 * x ↔ x < 1) := by
  sorry

end range_of_f_when_m_is_one_solution_set_of_inequality_l434_434598


namespace sum_of_remainders_l434_434662

theorem sum_of_remainders : 
  let m (k : ℕ) := 1111 * k + 123 in
  (List.sum (List.map (fun k => m k % 51) [0, 1, 2, 3, 4, 5, 6])) = 156 :=
by
  let m (k : ℕ) := 1111 * k + 123
  sorry

end sum_of_remainders_l434_434662


namespace intersect_at_single_point_l434_434153

theorem intersect_at_single_point
  (A B C K L M N : Point)
  (circumcircle1 : Circle)
  (circumcircle2 : Circle)
  (BC : Line)
  (tangent1 : Line)
  (tangent2 : Line)
  (h1 : K ∈ BC)
  (h2 : Tangent_to_Circle A K circumcircle1 tangent1 L)
  (h3 : Parallel tangent1 (Line_through A B))
  (h4 : AL_line : Line := Line_through A L, AL_intersect_circumcircle2 : M ∈ (circumcircle2 : Set₁) (Line_through A M ≠ A))
  (h5 : Tangent_to_Circle N M circumcircle2 tangent2) :
  (intersect_line tangent1 BC).val = N ∧ (intersect_line BC tangent2).val = N ∧ (intersect_line tangent1 tangent2).val = N :=
by
  sorry

end intersect_at_single_point_l434_434153


namespace tony_initial_amount_l434_434211

-- Define the initial amount P
variable (P : ℝ)

-- Define the conditions
def initial_amount := P
def after_first_year := 1.20 * P
def after_half_taken := 0.60 * P
def after_second_year := 0.69 * P
def final_amount : ℝ := 690

-- State the theorem to prove
theorem tony_initial_amount : 
  (after_second_year P = final_amount) → (initial_amount P = 1000) :=
by 
  intro h
  sorry

end tony_initial_amount_l434_434211


namespace minimum_value_of_sequence_l434_434693

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = -4 / 3 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 2 * (n + 1) * a n / (a n + 2 * n)

theorem minimum_value_of_sequence :
  ∃ (a : ℕ → ℝ), sequence a ∧ ∀ n : ℕ, n > 0 ∧ n ≤ 2 → a n = -8 := by
  sorry

end minimum_value_of_sequence_l434_434693


namespace dice_probability_l434_434561

theorem dice_probability (p : ℚ) (h : p = (1 / 42)) : 
  p = 0.023809523809523808 := 
sorry

end dice_probability_l434_434561


namespace hyperbola_equation_l434_434839

theorem hyperbola_equation (x y : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (y^2 / a^2 - x^2 / b^2 = 1) ∧ y = -2 ∧ x = 2
              ∧ (a / b = sqrt 2 / 2) ∧ ((4 / a^2 - 4 / b^2) = 1)) →
  (y^2 / 2 - x^2 / 4 = 1) :=
by sorry

end hyperbola_equation_l434_434839


namespace avg_sum_is_286_over_11_l434_434414

def num_list := (1 : Finₓ 13)

theorem avg_sum_is_286_over_11 :
  let f (l : List (Finₓ 13)) := |l[0] - l[1]| + |l[2] - l[3]| + |l[4] - l[5]| + |l[6] - l[7]| + |l[8] - l[9]| + |l[10] - l[11]| in
  let all_permutations := List.permutations num_list in
  let all_sums := all_permutations.map f in
  ((all_sums.sum.toRational : ℚ) / (12.factorial : ℚ)) = (286 / 11) :=
sorry

end avg_sum_is_286_over_11_l434_434414


namespace game_winnable_iff_l434_434979

theorem game_winnable_iff (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) : 
  (∃ m : ℕ, ∃ strategy : (fin 2n)^k → Prop, ∀ wizard_response : (fin 2n)^k, (∃ (i j : fin n), i ≠ j ∧ strategy (array.set wizard_response i k) ∧ strategy (array.set wizard_response j k))) ↔ (n > k) :=
sorry

end game_winnable_iff_l434_434979


namespace number_of_valid_pairs_l434_434119

noncomputable def jane_age := 25

def is_digit_interchange (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), a = 10 * x + y ∧ b = 10 * y + x ∧ x < 10 ∧ y < 10

theorem number_of_valid_pairs :
  ∃ (pairs : ℕ), pairs = 25 ∧
  pairs = (finset.univ.filter (λ (d : ℕ × ℕ), 
    let d' := d.1 in
    let n := d.2 in
    d' > jane_age ∧ d' > 25 ∧
    let D := d' + n in
    let J := jane_age + n in
    10 ≤ D ∧ D < 100 ∧
    10 ≤ J ∧ J < 100 ∧
    is_digit_interchange D J)).card :=
by {
  sorry
}

end number_of_valid_pairs_l434_434119


namespace butterfly_distance_1007_l434_434252

noncomputable def butterfly_distance (n : ℕ) : ℝ :=
  let ω := Complex.exp (-Real.pi * Complex.i / 4)
  let z := (Finset.range (n + 1)).sum (λ k, (2 * k + 1 : ℂ) * ω ^ k)
  Complex.abs z / Complex.abs (ω - 1)

theorem butterfly_distance_1007 : butterfly_distance 1007 = 1008 + 1008 * Real.sqrt 2 :=
  sorry

end butterfly_distance_1007_l434_434252


namespace count_harmonious_numbers_l434_434571

def is_harmonious (B : ℕ) : Prop :=
  ∀ A : ℕ, (∃ n : ℕ, n ≥ 2 ∧ A.digits.length = n ∧ B ∣ A) →
  ∀ C : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i < n → C = A.insert B i → B ∣ C)

def harmonious_numbers (l : List ℕ) : List ℕ :=
  l.filter is_harmonious

theorem count_harmonious_numbers :
  harmonious_numbers [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 15, 66, 90] = [1, 2, 4, 5, 9, 10, 11, 12, 15, 66, 90] :=
sorry

end count_harmonious_numbers_l434_434571


namespace triangular_pyramid_has_no_circular_cross_section_l434_434742

inductive Solid
| Cone
| Cylinder
| Sphere
| TriangularPyramid

def has_circular_cross_section : Solid → Prop
| Solid.Cone := True
| Solid.Cylinder := True
| Solid.Sphere := True
| Solid.TriangularPyramid := False

theorem triangular_pyramid_has_no_circular_cross_section :
  has_circular_cross_section Solid.TriangularPyramid = False := by
  sorry

end triangular_pyramid_has_no_circular_cross_section_l434_434742


namespace problem_jerry_reaches_6_l434_434562

/-- Jerry starts at 0 on the real number line. He tosses a fair coin 10 times.
When he gets heads, he moves 1 unit in the positive direction; when he gets tails,
he moves 1 unit in the negative direction. The probability that he reaches 6 at some
time during this process is 45/512. Therefore, c + d = 557 where c and d are the
numerators and denominators of the reduced fraction, respectively. -/
theorem problem_jerry_reaches_6 :
  let c := 45,
      d := 512 in
  c + d = 557 := 
by
  sorry

end problem_jerry_reaches_6_l434_434562


namespace log_product_max_l434_434007

open Real

theorem log_product_max (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : log x + log y = 4) : log x * log y ≤ 4 := 
by
  sorry

end log_product_max_l434_434007


namespace part1_part2_part3_l434_434898

noncomputable def a_seq (n : ℕ) : ℝ := 4^n

noncomputable def b_seq (n : ℕ) : ℝ := 4^(2 * n) + 1/4^n

theorem part1 (n : ℕ) : b_seq n = 4^(2 * n) + 1/4^n :=
  sorry

theorem part2 (n : ℕ) : ∃ (r : ℝ), r = 4 ∧ ∀ n : ℕ, b_seq (n + 1)^2 - b_seq (2 * (n + 1)) = r * (b_seq n^2 - b_seq (2 * n)) :=
  sorry

theorem part3 (n : ℕ) : ∑ k in Finset.range n, Real.sqrt ((2 * k + 1) * (2 * k + 3) / (b_seq k^2 - b_seq (2 * k))) < 2 * Real.sqrt 2 :=
  sorry

end part1_part2_part3_l434_434898


namespace max_value_of_3x_plus_4y_l434_434024

theorem max_value_of_3x_plus_4y (x y : ℝ) (h : x^2 + y^2 = 10) : 
  ∃ z, z = 5 * Real.sqrt 10 ∧ z = 3 * x + 4 * y :=
by
  sorry

end max_value_of_3x_plus_4y_l434_434024


namespace andy_cavities_l434_434341

/-- Andy gets a cavity for every 4 candy canes he eats.
  He gets 2 candy canes from his parents, 
  3 candy canes each from 4 teachers.
  He uses his allowance to buy 1/7 as many candy canes as he was given.
  Prove the total number of cavities Andy gets from eating all his candy canes is 4. -/
theorem andy_cavities :
  let canes_per_teach := 3
  let teachers := 4
  let canes_from_parents := 2
  let bought_fraction := 1 / 7
  let cavities_per_cane := 1 / 4 in
  let canes_from_teachers := canes_per_teach * teachers in
  let total_given := canes_from_teachers + canes_from_parents in
  let canes_bought := total_given * bought_fraction in
  let total_canes := total_given + canes_bought in
  let cavities := total_canes * cavities_per_cane in
  cavities = 4 := 
begin
  sorry
end

end andy_cavities_l434_434341


namespace sum_mod_500_l434_434795

def S : ℕ :=
  ∑ n in Finset.range 501, (-1)^n * Nat.choose 1500 (3 * n)

theorem sum_mod_500 : S % 500 = r := sorry

end sum_mod_500_l434_434795


namespace sum_even_digits_1_to_500_l434_434797

def even_digits_sum (n : Nat) : Nat :=
  n.digits.filter (λ d, d % 2 = 0).sum

def sum_even_digits_up_to (m : Nat) : Nat :=
  (List.range (m + 1)).map even_digits_sum |> List.sum

theorem sum_even_digits_1_to_500 : sum_even_digits_up_to 500 = 2600 := 
by
  sorry

end sum_even_digits_1_to_500_l434_434797


namespace weekly_allowance_l434_434986

def allowance (A : ℝ) := A

def spent_on_movies (A : ℝ) := 0.40 * A
def remaining_after_movies (A : ℝ) := A - spent_on_movies A
def spent_on_snacks (A : ℝ) := 0.25 * remaining_after_movies A
def remaining_after_snacks (A : ℝ) := remaining_after_movies A - spent_on_snacks A
def spent_on_school_supplies := 3 * 1 + 4
def remaining_after_supplies (A : ℝ) := remaining_after_snacks A - spent_on_school_supplies
def earned_from_tasks := 6 + 8 + 3 + 12
def final_amount (A : ℝ) := remaining_after_supplies A + earned_from_tasks

theorem weekly_allowance : ∃ (A : ℝ), final_amount A = 42 :=
by
  refine ⟨44.44, _⟩
  sorry

end weekly_allowance_l434_434986


namespace sum_coords_B_l434_434624

def point (A B : ℝ × ℝ) : Prop :=
A = (0, 0) ∧ B.snd = 5 ∧ (B.snd - A.snd) / (B.fst - A.fst) = 3 / 4

theorem sum_coords_B (x : ℝ) :
  point (0, 0) (x, 5) → x + 5 = 35 / 3 :=
by
  intro h
  sorry

end sum_coords_B_l434_434624


namespace folded_triangle_segment_length_square_l434_434770

theorem folded_triangle_segment_length_square :
  ∀ (DEF : Triangle) (side_length : ℝ), 
    is_equilateral DEF →
    side DEF = side_length →
    let E := point_a DEF,
    let F := point_b DEF,
    let D := point_c DEF,
    let P := fold_point D E 7,
    let fold_segment_length_square := (fold_segment_length DEF P) ^ 2 in
  side_length = 15 → 
  distance E P = 7 →
  fold_segment_length_square = 28561 / 529 :=
by
  sorry

end folded_triangle_segment_length_square_l434_434770


namespace math_problem_l434_434417

variable {α : Type*} [LinearOrder α] [AddGroup α] [HasZero α]

variable {f : α → α}

theorem math_problem 
  (a b : α) 
  (h1 : a < b) (h2 : b < 0) 
  (h3 : ∀ x, f (-x) = -f x) -- f is odd
  (h4 : ∀ x y, -b ≤ x → x ≤ -a → x ≤ y → y ≤ -a → f y ≤ f x) -- f is monotonically decreasing on [-b, -a]
  (h5 : ∀ x, a ≤ x → x ≤ -a → 0 < f x) : 
  (∀ x, a ≤ x → x ≤ b → f x < 0) ∧ (∀ x y, a ≤ x → x ≤ y → y ≤ b → |f x| ≥ |f y|) :=
sorry

end math_problem_l434_434417


namespace quadratic_value_at_sum_of_roots_is_five_l434_434878

noncomputable def quadratic_func (a b x : ℝ) : ℝ := a * x^2 + b * x + 5

theorem quadratic_value_at_sum_of_roots_is_five
  (a b x₁ x₂ : ℝ)
  (hA : quadratic_func a b x₁ = 2023)
  (hB : quadratic_func a b x₂ = 2023)
  (ha : a ≠ 0) :
  quadratic_func a b (x₁ + x₂) = 5 :=
sorry

end quadratic_value_at_sum_of_roots_is_five_l434_434878


namespace triangle_area_l434_434475

-- Define the given curve
def curve (x : ℝ) : ℝ := (x - 4) ^ 2 * (x + 3)

-- x-intercepts occur when y = 0
def x_intercepts : set ℝ := { x | curve x = 0 }

-- y-intercept occurs when x = 0
def y_intercept : ℝ := curve 0

-- Base of the triangle is the distance between the x-intercepts
def base_of_triangle : ℝ := max (4 : ℝ) (-3) - min (4 : ℝ) (-3)

-- Height of the triangle is the y-intercept value
def height_of_triangle : ℝ := y_intercept

-- Area of the triangle
def area_of_triangle (b h : ℝ) : ℝ := 1 / 2 * b * h

theorem triangle_area : area_of_triangle base_of_triangle height_of_triangle = 168 := by
  -- Stating the problem requires definitions of x-intercepts and y-intercept
  have hx : x_intercepts = {4, -3} := by
    sorry -- The proof for finding x-intercepts

  have hy : y_intercept = 48 := by
    sorry -- The proof for finding y-intercept

  -- Setup base and height using the intercepts
  have b : base_of_triangle = 7 := by
    -- Calculate the base from x_intercepts
    rw [hx]
    exact calc
      4 - (-3) = 4 + 3 := by ring
      ... = 7 := rfl

  have h : height_of_triangle = 48 := by
    -- height_of_triangle should be y_intercept which is 48
    rw [hy]

  -- Finally calculate the area
  have A : area_of_triangle base_of_triangle height_of_triangle = 1 / 2 * 7 * 48 := by
    rw [b, h]

  -- Explicitly calculate the numerical value
  exact calc
    1 / 2 * 7 * 48 = 1 / 2 * 336 := by ring
    ... = 168 := by norm_num

end triangle_area_l434_434475


namespace fraction_evaluation_l434_434064

def h (x : ℤ) : ℤ := 3 * x + 4
def k (x : ℤ) : ℤ := 4 * x - 3

theorem fraction_evaluation :
  (h (k (h 3))) / (k (h (k 3))) = 151 / 121 :=
by sorry

end fraction_evaluation_l434_434064


namespace math_proof_problem_l434_434976

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (m n : ℝ × ℝ)
variable [Nontrivial ℝ]

-- Conditions
def conditions : Prop :=
  (m = (Real.cos A, a)) ∧ 
  (n = (2 * Real.cos C - 3 * Real.cos B, 3 * b - 2 * c)) ∧ 
  (m.1 * n.2 = m.2 * n.1)
  
-- Questions
def part1 : Prop :=
  ∃ b c, (conditions a b c A B C) → b / c = 3 / 2

def part2 : Prop :=
  ∃ a b c, (conditions a b c (2 * Real.pi / 3) B C) → 
  area ABC = 2 * Real.sqrt 3 / 3 → 
  a = 2 * Real.sqrt 19 / 3

-- Main statement combining both parts
theorem math_proof_problem : part1 a b c ∧ part2 a b c :=
by
  sorry

end math_proof_problem_l434_434976


namespace joes_monthly_income_l434_434247

theorem joes_monthly_income :
  ∃ (monthly_income : ℝ), (0.4 * monthly_income = 848) ∧ (monthly_income = 2120) :=
begin
  sorry
end

end joes_monthly_income_l434_434247


namespace natural_number_x_l434_434860

theorem natural_number_x (x : ℕ) (A : ℕ → ℕ) (h : 3 * (A (x + 1))^3 = 2 * (A (x + 2))^2 + 6 * (A (x + 1))^2) : x = 4 :=
sorry

end natural_number_x_l434_434860


namespace cherry_pie_degrees_l434_434611

theorem cherry_pie_degrees :
  ∀ (total_students chocolate_students apple_students blueberry_students : ℕ),
  total_students = 36 →
  chocolate_students = 12 →
  apple_students = 8 →
  blueberry_students = 6 →
  (total_students - chocolate_students - apple_students - blueberry_students) / 2 = 5 →
  ((5 : ℕ) * 360 / total_students) = 50 := 
by
  sorry

end cherry_pie_degrees_l434_434611


namespace find_f_2011_l434_434387

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Icc (-1 : ℝ) 1 then x^3
else if (x - 1) % 4 = 0 then f ((x - 1)/4)
else -f (-(x - 1)/4)

theorem find_f_2011 : 
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, f (1 + x) = f (1 - x)) →
  (∀ x ∈ Icc (-1 : ℝ) 1, f x = x^3) →
  f 2011 = -1 :=
by
  sorry

end find_f_2011_l434_434387


namespace polynomial_symmetric_proof_l434_434446

theorem polynomial_symmetric_proof (a b : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^3 + a * x^2 + b * x + 2)
  (h_symmetric : f 2 = 0) 
  (h_derivative : f' x = 3x^2 + 2*a*x + b)
  (h_point : f (2)) :
  f (1) = 1 := by
  sorry

end polynomial_symmetric_proof_l434_434446


namespace solution_set_ineq1_range_of_a_l434_434454

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

-- Prove the solution set for f(x) >= 1
theorem solution_set_ineq1 : 
  ∀ x : ℝ, f(x) ≥ 1 ↔ x ≤ -3/2 ∨ x ≥ 3/2 :=
sorry

-- Prove the range of a for f(x) >= a^2 - a - 2
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f(x) ≥ a^2 - a - 2) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end solution_set_ineq1_range_of_a_l434_434454


namespace hyperbola_tangent_circle_l434_434307

noncomputable def hyperbolaEquation := 
  ∀ (x y : ℝ), (x^2 / (11/3) - y^2 / 11 = 1) ↔ 
    (∃ (k : ℝ), k = √3 ∨ k = -√3) ∧ -- Asymptotes are y = ±√3 x
    (x = 2 ∧ y = 1) ∧ -- Passes through the point (2,1)
    ∀ c, (x^2+((y-2)^2)=1 → abs (2 / √(k^2+1)) = 1) -- Tangency condition with the circle

theorem hyperbola_tangent_circle : hyperbolaEquation :=
sorry

end hyperbola_tangent_circle_l434_434307


namespace union_area_equals_20_l434_434329

def point := (ℝ × ℝ)

def reflect_about_y_eq_1 (p : point) : point :=
  (p.1, 2 - p.2)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def original_triangle : (point × point × point) :=
  ((3, 4), (4, -2), (7, 0))

def reflected_triangle (A B C : point) := (reflect_about_y_eq_1 A, reflect_about_y_eq_1 B, reflect_about_y_eq_1 C)

theorem union_area_equals_20 :
  let A := (3, 4)
  let B := (4, -2)
  let C := (7, 0) in
  let A' := reflect_about_y_eq_1 A
  let B' := reflect_about_y_eq_1 B
  let C' := reflect_about_y_eq_1 C in
  triangle_area A B C + triangle_area A' B' C' = 20 :=
by
  sorry

end union_area_equals_20_l434_434329


namespace function_behavior_on_negative_interval_l434_434304

-- Define the necessary conditions and function properties
variables {f : ℝ → ℝ}

-- Conditions: f is even, increasing on [0, 7], and f(7) = 6
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def f7_eq_6 (f : ℝ → ℝ) : Prop := f 7 = 6

-- The theorem to prove
theorem function_behavior_on_negative_interval (h1 : even_function f) (h2 : increasing_on_interval f 0 7) (h3 : f7_eq_6 f) : 
  (∀ x y, -7 ≤ x → x ≤ y → y ≤ 0 → f x ≥ f y) ∧ (∀ x, -7 ≤ x → x ≤ 0 → f x ≤ 6) :=
sorry

end function_behavior_on_negative_interval_l434_434304


namespace circle_diameter_l434_434265

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l434_434265


namespace motorboat_speed_after_2_minutes_l434_434767

/-- 
Prove that the speed of the motorboat 2 minutes after turning off the engine is 
1.28 km/h, given:
1. The initial speed of the motorboat \( v_{0} = 20 \, \text{km/h} \).
2. The speed of the motorboat decreased to \( v_{1} = 8 \, \text{km/h} \) in 40 seconds.
3. The water resistance is proportional to the boat's speed.
-/
theorem motorboat_speed_after_2_minutes
  (v₀ : ℝ) (v₁ : ℝ) (t₁ : ℝ) (t₂ : ℝ) (k m : ℝ)
  (h1 : v₀ = 20)
  (h2 : v₁ = 8)
  (h3 : t₁ = 40/3600)
  (h4 : t₂ = 2/60)
  (h5 : ∀ (v : ℝ) (t : ℝ), v t = v₀ * Real.exp (- (k / m) * t)) :
  ∃ (vt: ℝ), vt = v₀ * (2/5)^(t₂ * (1/t₁)) ∧ vt = 1.28 := 
by
  sorry

end motorboat_speed_after_2_minutes_l434_434767


namespace positive_integer_solutions_3x_5y_eq_501_l434_434674

theorem positive_integer_solutions_3x_5y_eq_501 : ∃ n : ℕ, n = 34 ∧ ∀ (x y : ℕ), 3 * x + 5 * y = 501 → x > 0 ∧ y > 0 → n = 34 :=
by
  sorry

end positive_integer_solutions_3x_5y_eq_501_l434_434674


namespace find_n_l434_434853

open Real

def a : ℝ × ℝ := (1, 1)
def b (n : ℝ) : ℝ × ℝ := (2, n)

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

theorem find_n (n : ℝ) :
  vector_norm (vector_add a (b n)) = dot_product a (b n) → n = 3 := by
  sorry

end find_n_l434_434853


namespace sum_coordinates_eq_l434_434630

theorem sum_coordinates_eq : ∀ (A B : ℝ × ℝ), A = (0, 0) → (∃ x : ℝ, B = (x, 5)) → 
  (∀ x : ℝ, B = (x, 5) → (5 - 0) / (x - 0) = 3 / 4) → 
  let x := 20 / 3 in
  (x + 5 = 35 / 3) :=
by
  intros A B hA hB hslope
  have hx : ∃ x, B = (x, 5) := hB
  cases hx with x hx_def
  rw hx_def at hslope
  have : x = 20 / 3 := by
    rw hx_def at hslope
    field_simp at hslope
    linarith
  rw [this, hx_def]
  norm_num

end sum_coordinates_eq_l434_434630


namespace inscribed_circle_radius_l434_434724

theorem inscribed_circle_radius 
  (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
  (AB AC BC : ℝ)
  (hAB : AB = 24)
  (hAC : AC = 10)
  (hBC : BC = 26) : 
  ∃ r : ℝ, r = 4 :=
begin
  sorry
end

end inscribed_circle_radius_l434_434724


namespace inequality_k_bound_l434_434409

theorem inequality_k_bound (k : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → ∫ t in 0..x, (1 / (sqrt (3 + t^2)^3)) ≥ 
                 k * ∫ t in 0..x, (1 / sqrt (3 + t^2))) → 
  k ≤ 1 / (3 * sqrt 3 * real.log 3) :=
begin
  sorry
end

end inequality_k_bound_l434_434409


namespace compare_expr_l434_434004

theorem compare_expr (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : 
  (a + b) * (a^2 + b^2) ≤ 2 * (a^3 + b^3) :=
sorry

end compare_expr_l434_434004


namespace eric_return_journey_time_correct_l434_434393

variable (distance_skateboard : ℝ) (speed_skateboard : ℝ)
variable (rest_time_minutes : ℝ)
variable (distance_jog : ℝ) (speed_jog : ℝ)
variable (distance_walk : ℝ) (speed_walk : ℝ)
variable (total_return_distance : ℝ)
variable (expected_time : ℝ)

def return_journey_total_time 
  (distance_skateboard : ℝ) (speed_skateboard : ℝ)
  (rest_time_minutes : ℝ)
  (distance_jog : ℝ) (speed_jog : ℝ)
  (distance_walk : ℝ) (speed_walk : ℝ)
  (total_return_distance : ℝ) : ℝ :=
    distance_skateboard / speed_skateboard + 
    rest_time_minutes / 60 + 
    distance_jog / speed_jog + 
    distance_walk / speed_walk

theorem eric_return_journey_time_correct :
  return_journey_total_time 5 10 15 6 6 4 4 15 = 2.75 :=
by
  dsimp [return_journey_total_time]
  norm_num
  sorry

end eric_return_journey_time_correct_l434_434393


namespace area_of_triangle_from_intercepts_l434_434489

theorem area_of_triangle_from_intercepts :
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := ({4, -3} : Set ℝ)
  let y_intercept := f 0
  let base := 4 - (-3)
  let height := y_intercept
  let area := 1 / 2 * base * height
  area = 168 := 
by
  -- Define the function f
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  -- Calculate x-intercepts
  have hx1 : f 4 = 0 := by simp [f, pow_two, mul_assoc]
  have hx2 : f (-3) = 0 := by simp [f, pow_two]
  let x_intercepts := ({4, -3} : Set ℝ)
  -- Calculate y-intercept
  have hy : f 0 = 48 := by simp [f, pow_two]
  let y_intercept := 48
  -- Define base and height
  let base := 4 - (-3)
  let height := y_intercept
  -- Compute the area
  let area := 1 / 2 * base * height
  -- Show that the area is 168
  show area = 168
  by
    simp [base, height, hy]
    norm_num
    sorry -- Skip the full proof

end area_of_triangle_from_intercepts_l434_434489


namespace hyperbola_eccentricity_l434_434919

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  real.sqrt ((b^2 + a^2) / a^2)

theorem hyperbola_eccentricity {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
  (asymptotes : b / a = real.sqrt 3) :
  eccentricity a b = 2 := by
  sorry

end hyperbola_eccentricity_l434_434919


namespace area_square_II_is_6a_squared_l434_434661

-- Problem statement:
-- Given the diagonal of square I is 2a and the area of square II is three times the area of square I,
-- prove that the area of square II is 6a^2

noncomputable def area_square_II (a : ℝ) : ℝ :=
  let side_I := (2 * a) / Real.sqrt 2
  let area_I := side_I ^ 2
  3 * area_I

theorem area_square_II_is_6a_squared (a : ℝ) : area_square_II a = 6 * a ^ 2 :=
by
  sorry

end area_square_II_is_6a_squared_l434_434661


namespace meaningful_square_root_l434_434525

theorem meaningful_square_root (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 := 
by {
  -- This part would contain the proof
  sorry
}

end meaningful_square_root_l434_434525


namespace correct_mark_l434_434774

theorem correct_mark (n : ℕ) (increase : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ) :
  n = 68 → increase = 34 → wrong_mark = 79 → correct_mark = wrong_mark - increase → correct_mark = 45 :=
by
  intros h_n h_increase h_wrong_mark h_correct_mark_eq
  rw [h_n, h_increase, h_wrong_mark] at h_correct_mark_eq
  exact h_correct_mark_eq

end correct_mark_l434_434774


namespace find_a_and_b_l434_434076

noncomputable def find_ab (a b : ℝ) : Prop :=
  (3 - 2 * a + b = 0) ∧
  (27 + 6 * a + b = 0)

theorem find_a_and_b :
  ∃ (a b : ℝ), (find_ab a b) ∧ (a = -3) ∧ (b = -9) :=
by
  sorry

end find_a_and_b_l434_434076


namespace area_of_triangle_l434_434480

theorem area_of_triangle :
  let f : ℝ → ℝ := λ x, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := (0, f 0)
  let base := (x_intercepts[0] - x_intercepts[1]).abs
  let height := y_intercept.2
  1 / 2 * base * height = 168 :=
by
  sorry

end area_of_triangle_l434_434480


namespace exists_set_B_half_residues_l434_434584

open Set

variables (N : ℕ) (A : Finset (ZMod (N^2)))

theorem exists_set_B_half_residues (hA : A.card = N) :
  ∃ (B : Finset (ZMod (N^2))), B.card = N ∧ ((A.product B).image (λ p, p.1 + p.2)).card ≥ N^2 / 2 :=
sorry

end exists_set_B_half_residues_l434_434584


namespace min_value_of_expression_values_of_a_l434_434139

section Problem1
  variables {x y z : ℝ}
  hypothesis h_sum : x + y + z = 1

  theorem min_value_of_expression : 
    (x - 1)^2 + (y + 1)^2 + (z + 1)^2 ≥ 4/3 :=
  sorry
end Problem1

section Problem2
  variables {x y z a : ℝ}
  hypothesis h_sum : x + y + z = 1
  hypothesis h_ineq : (x - 2)^2 + (y - 1)^2 + (z - a)^2 ≥ 1/3

  theorem values_of_a : a ≤ -3 ∨ a ≥ -1 :=
  sorry
end Problem2

end min_value_of_expression_values_of_a_l434_434139


namespace ratio_rate_down_to_up_l434_434305

noncomputable def rate_up (r_up t_up: ℕ) : ℕ := r_up * t_up
noncomputable def rate_down (d_down t_down: ℕ) : ℕ := d_down / t_down
noncomputable def ratio (r_down r_up: ℕ) : ℚ := r_down / r_up

theorem ratio_rate_down_to_up :
  let r_up := 6
  let t_up := 2
  let d_down := 18
  let t_down := 2
  rate_up 6 2 = 12 ∧ rate_down 18 2 = 9 ∧ ratio 9 6 = 3 / 2 :=
by
  sorry

end ratio_rate_down_to_up_l434_434305


namespace packs_needed_l434_434980

def pouches_per_pack : ℕ := 6
def team_members : ℕ := 13
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people : ℕ := team_members + coaches + helpers

theorem packs_needed (people : ℕ) (pouches_per_pack : ℕ) : ℕ :=
  (people + pouches_per_pack - 1) / pouches_per_pack

example : packs_needed total_people pouches_per_pack = 3 :=
by
  have h1 : total_people = 18 := rfl
  have h2 : pouches_per_pack = 6 := rfl
  rw [h1, h2]
  norm_num
  sorry

end packs_needed_l434_434980


namespace greatest_integer_of_negative_fraction_l434_434719

-- Define the original fraction
def original_fraction : ℚ := -19 / 5

-- Define the greatest integer function
def greatest_integer_less_than (q : ℚ) : ℤ :=
  Int.floor q

-- The proof problem statement:
theorem greatest_integer_of_negative_fraction :
  greatest_integer_less_than original_fraction = -4 :=
sorry

end greatest_integer_of_negative_fraction_l434_434719


namespace triangle_ratio_perimeter_l434_434963

theorem triangle_ratio_perimeter (AC BC : ℝ) (CD : ℝ) (AB : ℝ) (m n : ℕ) :
  AC = 15 → BC = 20 → AB = 25 → CD = 10 * Real.sqrt 3 →
  gcd m n = 1 → (2 * Real.sqrt ((AC * BC) / AB) + AB) / AB = m / n → m + n = 7 :=
by
  intros hAC hBC hAB hCD hmn hratio
  sorry

end triangle_ratio_perimeter_l434_434963


namespace number_of_subsets_of_intersection_is_four_l434_434464

def set1 : set (ℝ × ℝ) := {p | p.2 = p.1 ^ 2}
def set2 : set (ℝ × ℝ) := {p | p.2 = real.sqrt p.1}
def intersection : set (ℝ × ℝ) := {(0,0), (1,1)}

theorem number_of_subsets_of_intersection_is_four :
  set.card (set.powerset intersection) = 4 :=
sorry

end number_of_subsets_of_intersection_is_four_l434_434464


namespace fraction_simplification_l434_434368

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l434_434368


namespace mn_calc_l434_434960

noncomputable def MN_length (A B C D M N : ℝ) : ℝ :=
  let AB := 15
  let BC := 15
  let CD := 30
  let DA := 30
  let angleBDC := 60
  let midpoint_M := (BC / 2)
  let midpoint_N := (DA / 2)
  15

theorem mn_calc (A B C D M N : ℝ):
  (ABCD.midpoint_BC = M) ∧
  (ABCD.midpoint_DA = N) ∧
  (ABCD.angleBDC = 60) ∧
  (ABCD.AB = 15) ∧
  (ABCD.BC = 15) ∧
  (ABCD.CD = 30) ∧
  (ABCD.DA = 30)
  → MN_length A B C D M N = 15 :=
by sorry

end mn_calc_l434_434960


namespace circle_diameter_l434_434270

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l434_434270


namespace solve_fractional_equation_l434_434195

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 4) : 
  (3 - x) / (x - 4) + 1 / (4 - x) = 1 → x = 3 :=
by {
  sorry
}

end solve_fractional_equation_l434_434195


namespace certain_number_divisibility_l434_434079

theorem certain_number_divisibility (n : ℕ) (p : ℕ) (h : p = 1) (h2 : 4864 * 9 * n % 12 = 0) : n = 43776 :=
by {
  sorry
}

end certain_number_divisibility_l434_434079


namespace sqrt_x_minus_1_domain_l434_434528

theorem sqrt_x_minus_1_domain (x : ℝ) : (∃ y, y^2 = x - 1) → (x ≥ 1) := 
by 
  sorry

end sqrt_x_minus_1_domain_l434_434528


namespace hexagon_area_l434_434757

theorem hexagon_area (A : ℝ) (h : A = π * 64 * (real.sqrt 3)) : A = 384 :=
sorry

end hexagon_area_l434_434757


namespace curve_symmetries_and_point_l434_434901

theorem curve_symmetries_and_point :
  ∀ (x y : ℝ), (x ^ 2 + x * y + y ^ 2 = 4) → 
  (∀ (x y : ℝ), x ^ 2 + x * y + y ^ 2 = 4 → (-x) ^ 2 + (-x) * (-y) + (-y) ^ 2 = 4) ∧ 
  (∀ (x y : ℝ), x ^ 2 + x * y + y ^ 2 = 4 → y ^ 2 + x * y + x ^ 2 = 4) ∧ 
  ( (2 : ℝ) ^ 2 + 2 * (-2 : ℝ) + (-2 : ℝ) ^ 2 = 4) :=
sorry

end curve_symmetries_and_point_l434_434901


namespace angles_C_and_A_l434_434959

variable (A B C D : Type)
variable [Parallelogram A B C D]
variable (∠B : ℝ)
variable (∠C ∠A : ℝ)

axiom parallelogram_consecutive_angles_supplementary (p : Parallelogram A B C D) : ∠B + ∠C = 180
axiom angle_B_given : ∠B = 135

theorem angles_C_and_A (p : Parallelogram A B C D) (hB : ∠B = 135) : ∠C = 45 ∧ ∠A = 45 :=
by
  sorry

end angles_C_and_A_l434_434959


namespace pure_imaginary_iff_l434_434437

noncomputable def imaginary_part_of_pure_imaginary {a : ℝ} : ℂ := 
  let z := a + (15 / (3 - (4 * complex.I))) in
  if a + 9/5 = 0 then z else 0

theorem pure_imaginary_iff (a : ℝ) : 
  (∃ z : ℂ, z = a + 15 / (3 - 4 * complex.I) ∧ (z.im ≠ 0 ∧ z.re = 0)) ↔ a = -9/5 :=
by sorry

end pure_imaginary_iff_l434_434437


namespace zero_of_y_l434_434040

def matrix_det (x : ℝ) : ℝ :=
  (2 ^ x * (-3) * (-1)) - (2 ^ x * 4 * 5) - (7 * 4 * (-1)) + (7 * 4 * 6) + (4 ^ x * 4 * 6) - (4 ^ x * (-3) * 4) 

def cofactor_32 (x : ℝ) : ℝ :=
  -((2 ^ x * 4) - (4 ^ x * 4))

def f (x : ℝ) : ℝ :=
  -2 ^ (x + 2) * (1 + 2 ^ x)

def y (x : ℝ) : ℝ :=
  1 + f(x)

theorem zero_of_y : ∃ x : ℝ, y(x) = 0 ∧ x = -1 :=
by 
  use -1 
  simp [y, f, cofactor_32] 
  sorry

end zero_of_y_l434_434040


namespace simplify_fraction_l434_434167

-- Define the given variables and their assigned values.
variable (b : ℕ)
variable (b_eq : b = 2)

-- State the theorem we want to prove
theorem simplify_fraction (b : ℕ) (h : b = 2) : 
  15 * b ^ 4 / (75 * b ^ 3) = 2 / 5 :=
by
  -- sorry indicates where the proof would be written.
  sorry

end simplify_fraction_l434_434167


namespace combined_mpg_l434_434644

theorem combined_mpg (m : ℕ) (ray_mpg tom_mpg : ℕ) (h1 : m = 200) (h2 : ray_mpg = 40) (h3 : tom_mpg = 20) :
  (m / (m / (2 * ray_mpg) + m / (2 * tom_mpg))) = 80 / 3 :=
by
  sorry

end combined_mpg_l434_434644


namespace find_N_l434_434829

theorem find_N (N : ℕ) (h1 : N > 1)
  (h2 : ∀ (a b c : ℕ), (a ≡ b [MOD N]) ∧ (b ≡ c [MOD N]) ∧ (a ≡ c [MOD N])
  → (a = 1743) ∧ (b = 2019) ∧ (c = 3008)) :
  N = 23 :=
by
  have h3 : Nat.gcd (2019 - 1743) (3008 - 2019) = 23 := sorry
  show N = 23 from sorry

end find_N_l434_434829


namespace find_fraction_l434_434072

theorem find_fraction 
  (f : ℚ) (t k : ℚ)
  (h1 : t = f * (k - 32)) 
  (h2 : t = 75)
  (h3 : k = 167) : 
  f = 5 / 9 :=
by
  sorry

end find_fraction_l434_434072


namespace angle_A_eq_pi_over_three_sin_B_plus_sin_C_l434_434869

variables (A B C a b c : ℝ)
variable (S : ℝ)
variable h1 : 2 * sin A ^ 2 + 3 * cos (B + C) = 0
variable h2 : S = 5 * sqrt 3
variable h3 : a = sqrt 21

theorem angle_A_eq_pi_over_three (h1 : 2 * sin A ^ 2 + 3 * cos (B + C) = 0) : 
  A = π / 3 :=
sorry

theorem sin_B_plus_sin_C (A : ℝ) (b c : ℝ)
  (h1 : S = 5 * sqrt 3)
  (h2 : a = sqrt 21)
  (h3 : 2 * sin A ^ 2 + 3 * cos (B + C) = 0)
  (hA : A = π / 3)
  : sin B + sin C = 9 * sqrt 7 / 14 :=
sorry

end angle_A_eq_pi_over_three_sin_B_plus_sin_C_l434_434869


namespace circle_diameter_l434_434262

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434262


namespace smallest_prime_fifth_term_of_arithmetic_sequence_l434_434845

theorem smallest_prime_fifth_term_of_arithmetic_sequence :
  ∃ (a d : ℕ) (seq : ℕ → ℕ), 
    (∀ n, seq n = a + n * d) ∧ 
    (∀ n < 5, Prime (seq n)) ∧ 
    d = 6 ∧ 
    a = 5 ∧ 
    seq 4 = 29 := by
  sorry

end smallest_prime_fifth_term_of_arithmetic_sequence_l434_434845


namespace harry_basketball_points_l434_434948

theorem harry_basketball_points :
  ∃ (x y : ℕ), 
    (x < 15) ∧ 
    (y < 15) ∧ 
    (62 + x) % 11 = 0 ∧ 
    (62 + x + y) % 12 = 0 ∧ 
    (x * y = 24) :=
by
  sorry

end harry_basketball_points_l434_434948


namespace parallelepiped_ratio_l434_434616

open Real

noncomputable def vector_norm_sq (v : ℝ × ℝ × ℝ) : ℝ := v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2

theorem parallelepiped_ratio (a b c : ℝ × ℝ × ℝ) :
  (\vector_norm_sq b + \vector_norm_sq c + \vector_norm_sq a + \vector_norm_sq (b - c) + \vector_norm_sq (b - a) + \vector_norm_sq (c - a)) / 
  (\vector_norm_sq ((b + c) / 2) + \vector_norm_sq ((b + a) / 2) + \vector_norm_sq ((c + a) / 2)) = 4 := 
sorry

end parallelepiped_ratio_l434_434616


namespace f_half_f_quarter_f_periodic_lim_ln_a_n_l434_434587

variable (f : ℝ → ℝ) (a : ℝ)

noncomputable def even_function := ∀ x, f x = f (-x)

noncomputable def symmetric_about_one := ∀ x, f x = f (2 - x)

noncomputable def functional_eq (x₁ x₂ : ℝ) := 
  0 ≤ x₁ ∧ x₁ ≤ 1/2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 1/2 → f (x₁ + x₂) = f x₁ * f x₂ 

axiom f_one: f 1 = a
axiom a_pos: a > 0

theorem f_half : f (1 / 2) = real.sqrt a := by sorry

theorem f_quarter : f (1 / 4) = real.sqrt (real.sqrt a) := by sorry

theorem f_periodic : ∀ x, f x = f (x + 2) := by sorry

theorem lim_ln_a_n: 
  let a_n := λ n : ℕ, f (2 * n + 1 / (2 * n : ℝ)) in 
  filter.tendsto (λ n : ℕ, real.log (a_n n)) filter.at_top (nhds 0) := by sorry

end f_half_f_quarter_f_periodic_lim_ln_a_n_l434_434587


namespace sin_alpha_correct_l434_434244

noncomputable def sin_alpha (α : ℝ) (h1 : α ∈ set.Ioo (3 * π / 2) (2 * π)) (h2 : Real.tan α = -4/3) : Real :=
  Real.sin α

theorem sin_alpha_correct
  (α : ℝ)
  (h1 : α ∈ set.Ioo (3 * π / 2) (2 * π))
  (h2 : Real.tan α = -4/3) :
  sin_alpha α h1 h2 = -4/5 :=
by
  sorry

end sin_alpha_correct_l434_434244


namespace function_intersects_y_axis_at_most_once_l434_434186

theorem function_intersects_y_axis_at_most_once (f : ℝ → ℝ) :
  (∃ y1 y2 : ℝ, f(0) = y1 ∧ f(0) = y2 → y1 = y2) :=
begin
  sorry
end

end function_intersects_y_axis_at_most_once_l434_434186


namespace students_choose_same_events_l434_434539

theorem students_choose_same_events (students : ℕ) (sport_options_1 sport_options_2 sport_options_3 : ℕ) :
  students = 50 →
  sport_options_1 = 4 →
  sport_options_2 = 3 →
  sport_options_3 = 2 →
  ∃ (n : ℕ), n ≥ 3 ∧ ∃ events, ∃ (count : ℕ), count = students / (sport_options_1 * sport_options_2 * sport_options_3) ∧ count ≥ n :=
by
  intros h_students h_sport_options_1 h_sport_options_2 h_sport_options_3
  sorry

end students_choose_same_events_l434_434539


namespace positive_difference_mean_median_l434_434202

def vertical_drops := [150, 125, 145, 280, 190]

def calculate_mean (lst : List ℕ) : ℕ := lst.sum / lst.length

def calculate_median (lst : List ℕ) : ℕ :=
  let sorted_lst := lst.qsort (λ a b => a < b)
  sorted_lst.get! (sorted_lst.length / 2)

theorem positive_difference_mean_median : 
  |calculate_mean vertical_drops - calculate_median vertical_drops| = 28 := by
  sorry

end positive_difference_mean_median_l434_434202


namespace distance_from_P_to_y_axis_l434_434904

theorem distance_from_P_to_y_axis 
  (x y : ℝ)
  (h1 : (x^2 / 16) + (y^2 / 25) = 1)
  (F1 : ℝ × ℝ := (0, -3))
  (F2 : ℝ × ℝ := (0, 3))
  (h2 : (F1.1 - x)^2 + (F1.2 - y)^2 = 9 ∨ (F2.1 - x)^2 + (F2.2 - y)^2 = 9 
          ∨ (F1.1 - x)^2 + (F1.2 - y)^2 + (F2.1 - x)^2 + (F2.2 - y)^2 = (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) :
  |x| = 16 / 5 :=
by
  sorry

end distance_from_P_to_y_axis_l434_434904


namespace angle_between_a_b_is_90_degrees_l434_434923

-- Define the vectors involved in the problem
def a : ℝ × ℝ × ℝ := (3, 4, -3)
def b : ℝ × ℝ × ℝ := (5, -3, 1)

-- Define the dot product of two vectors
def dot_product (x y : ℝ × ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2 + x.3 * y.3

-- Define a proof that shows the angle between a and b is 90 degrees
theorem angle_between_a_b_is_90_degrees : dot_product a b = 0 :=
by
  -- Sorry is used to skip the proof
  sorry

end angle_between_a_b_is_90_degrees_l434_434923


namespace sum_digits_of_10_pow_30_minus_36_l434_434225

theorem sum_digits_of_10_pow_30_minus_36 :
  let k := 10^30 - 36 in 
  (Nat.digits 10 k).sum = 262 :=
by
  sorry

end sum_digits_of_10_pow_30_minus_36_l434_434225


namespace card_2015_in_box_3_l434_434700

-- Define the pattern function for placing cards
def card_placement (n : ℕ) : ℕ :=
  let cycle_length := 12
  let cycle_pos := (n - 1) % cycle_length + 1
  if cycle_pos ≤ 7 then cycle_pos
  else 14 - cycle_pos

-- Define the theorem to prove the position of the 2015th card
theorem card_2015_in_box_3 : card_placement 2015 = 3 := by
  -- sorry is used to skip the proof
  sorry

end card_2015_in_box_3_l434_434700


namespace max_plus_min_value_of_f_l434_434599

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_plus_min_value_of_f :
  let M := Real.Sup (set.range f)
  let m := Real.Inf (set.range f)
  M + m = 2 := by
  let M := Real.Sup (set.range f)
  let m := Real.Inf (set.range f)
  sorry

end max_plus_min_value_of_f_l434_434599


namespace find_integer_l434_434842

def satisfies_conditions (x : ℕ) (m n : ℕ) : Prop :=
  x + 100 = m ^ 2 ∧ x + 168 = n ^ 2 ∧ m > 0 ∧ n > 0

theorem find_integer (x m n : ℕ) (h : satisfies_conditions x m n) : x = 156 :=
sorry

end find_integer_l434_434842


namespace bike_lock_combinations_l434_434994

theorem bike_lock_combinations :
  let n := 40 in
  let odd_count := n / 2 in
  let even_count := n / 2 in
  let multiples_of_4_count := n / 4 in
  let multiples_of_5_count := n / 5 in
  odd_count * even_count * multiples_of_4_count * multiples_of_5_count = 32000 :=
by
  sorry

end bike_lock_combinations_l434_434994


namespace max_value_of_f_and_count_within_range_l434_434183

def f : ℕ → ℕ
| 1     := 1
| (2*n) := f n
| (2*n+1) := f (2*n) + 1

theorem max_value_of_f_and_count_within_range :
  (∀ n ∈ finset.range(1990), f n ≤ 8) ∧
  (∃ n ∈ finset.range(1990), f n = 8) ∧
  (finset.count 8 (finset.image f (finset.range 1990)) = 5) := 
begin
  sorry
end

end max_value_of_f_and_count_within_range_l434_434183


namespace sum_of_first_n_terms_l434_434030

def arithmetic_sequence (a : ℕ → ℚ) :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def a : ℕ → ℚ
| 0       := 5 / 6
| (n + 1) := sorry -- This is defined by the arithmetic_sequence definition

theorem sum_of_first_n_terms (a : ℕ → ℚ) (n : ℕ) (h : arithmetic_sequence a)
  (h1 : a 0 = 5 / 6)
  (h15 : a 14 = -3 / 2) :
  ∑ k in Finset.range n, a k = (-n^2 + 11 * n) / 12 := by
  sorry

end sum_of_first_n_terms_l434_434030


namespace hourly_wage_increase_is_10_percent_l434_434319

theorem hourly_wage_increase_is_10_percent :
  ∀ (H W : ℝ), 
    ∀ (H' : ℝ), H' = H * (1 - 0.09090909090909092) →
    (H * W = H' * W') →
    (W' = (100 * W) / 90) := by
  sorry

end hourly_wage_increase_is_10_percent_l434_434319


namespace triangle_inequality_l434_434160

-- Variables and definitions for a generic triangle
variables {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
           (s : ℝ) (t : ℝ) (fa fb fc : ℝ) (sa sb sc : ℝ)
           
-- Hypotheses for the problem
def conditions (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (s : ℝ) (t : ℝ) (fa fb fc : ℝ) (sa sb sc : ℝ) : Prop :=
  s = (a + b + c) / 2 ∧ -- Semi-perimeter
  t = sqrt(s * (s - a) * (s - b) * (s - c)) ∧ -- Area using Heron's formula
  fa = 2 * sqrt(b * c * s * (s - a)) / (b + c) ∧ -- Angle bisector
  fb = 2 * sqrt(a * c * s * (s - b)) / (a + c) ∧
  fc = 2 * sqrt(a * b * s * (s - c)) / (a + b) ∧
  sa = sqrt(2 * b^2 + 2 * c^2 - a^2) / 2 ∧ -- Medians
  sb = sqrt(2 * a^2 + 2 * c^2 - b^2) / 2 ∧
  sc = sqrt(2 * a^2 + 2 * b^2 - c^2) / 2

-- The statement to be proved
theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (s : ℝ) (t : ℝ) (fa fb fc : ℝ) (sa sb sc : ℝ)
    (h : conditions ha hb hc s t fa fb fc sa sb sc) :
  fa * fb * fc ≤ s * t ∧ s * t ≤ sa * sb * sc :=
sorry

end triangle_inequality_l434_434160


namespace sin_reciprocal_sum_l434_434609

theorem sin_reciprocal_sum (n : ℕ) :
  (\sum i in Finset.range (n+1), (sin (i+1) * (π / (2*n+1))).recip ^ 2) = (4 / 3) * n * (n + 1) :=
sorry

end sin_reciprocal_sum_l434_434609


namespace max_height_reached_l434_434773

def h (t : ℝ) : ℝ := -20 * t ^ 2 + 120 * t + 36

theorem max_height_reached :
  ∃ t : ℝ, h t = 216 ∧ t = 3 :=
sorry

end max_height_reached_l434_434773


namespace wrongly_entered_mark_l434_434322

theorem wrongly_entered_mark (x : ℝ) : 
  (∀ (correct_mark avg_increase pupils : ℝ), 
  correct_mark = 45 ∧ avg_increase = 0.5 ∧ pupils = 80 ∧
  (avg_increase * pupils = (x - correct_mark)) →
  x = 85) :=
by 
  intro correct_mark avg_increase pupils
  rintro ⟨hc, ha, hp, h⟩
  sorry

end wrongly_entered_mark_l434_434322


namespace num_girls_l434_434701

variable (boys girls students : ℕ)

-- Given conditions
def total_boys : boys = 9 := rfl
def total_students : students = 7 * 3 := rfl

-- Prove that girls are remaining students after accounting for boys
theorem num_girls (boys girls students : ℕ) (h1 : boys = 9) (h2 : students = 7 * 3) : girls = students - boys := 
by
  sorry

end num_girls_l434_434701


namespace evaluate_expression_l434_434809

def g (x : ℝ) := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 2 + 2 * g (-4) = 177 := by
  sorry

end evaluate_expression_l434_434809


namespace area_of_triangle_l434_434320

theorem area_of_triangle (D E F Q : Type) 
  (area_u1 area_u2 area_u3 : Real) 
  (h1 : area_u1 = 16) 
  (h2 : area_u2 = 25)
  (h3 : area_u3 = 36)
  (hQ : Q ∈ triangle D E F)
  (h_parallel : ∀ P1 P2 P3, parallel P1 P2 P3 Q)
  (h_bisectors : ∀ A B C, angle_bisector A B C Q) :
  area (triangle D E F) = 1155 :=
by
  sorry

end area_of_triangle_l434_434320


namespace proof_problem_l434_434736

/-- 
  Given:
  - r, j, z are Ryan's, Jason's, and Zachary's earnings respectively.
  - Zachary sold 40 games at $5 each.
  - Jason received 30% more money than Zachary.
  - The total amount of money received by all three is $770.
  Prove:
  - Ryan received $50 more than Jason.
--/
def problem_statement : Prop :=
  ∃ (r j z : ℕ), 
    z = 40 * 5 ∧
    j = z + z * 30 / 100 ∧
    r + j + z = 770 ∧ 
    r - j = 50

theorem proof_problem : problem_statement :=
by 
  sorry

end proof_problem_l434_434736


namespace sum_even_integers_202_to_300_l434_434200

theorem sum_even_integers_202_to_300 :
  (∑ k in Finset.range 50, (2 * (k + 1)) - 1) = 12550 := by
  sorry

end sum_even_integers_202_to_300_l434_434200


namespace norm_of_w_l434_434003

variable (u v : EuclideanSpace ℝ (Fin 2)) 
variable (hu : ‖u‖ = 3) (hv : ‖v‖ = 5) 
variable (h_orthogonal : inner u v = 0)

theorem norm_of_w :
  ‖4 • u - 2 • v‖ = 2 * Real.sqrt 61 := by
  sorry

end norm_of_w_l434_434003


namespace problem_solution_l434_434440

noncomputable def proof_problem (n : ℕ) (x : Fin n → ℝ) (h₁ : ∀ i, 0 < x i) 
  (h₂ : (Finset.univ : Finset (Fin n)).sum x = 1) : Prop :=
  (Finset.univ : Finset (Fin n)).sum (λ i, x i / Real.sqrt (1 - x i)) ≥ 
  (Finset.univ : Finset (Fin n)).sum (λ i, Real.sqrt (x i)) / Real.sqrt (n - 1)

theorem problem_solution (n : ℕ) (x : Fin n → ℝ) (h₁ : ∀ i, 0 < x i) 
  (h₂ : (Finset.univ : Finset (Fin n)).sum x = 1) : proof_problem n x h₁ h₂ :=
sorry

end problem_solution_l434_434440


namespace prod_f_roots_l434_434905

def f (x : ℂ) : ℂ := x^2 + 1

theorem prod_f_roots :
  ∀ (x₁ x₂ x₃ x₄ x₅ : ℂ),
  (x₁^5 - x₁^2 + 5 = 0) →
  (x₂^5 - x₂^2 + 5 = 0) →
  (x₃^5 - x₃^2 + 5 = 0) →
  (x₄^5 - x₂^2 + 5 = 0) →
  (x₅^5 - x₅^2 + 5 = 0) →
  (∏ k in ({x₁, x₂, x₃, x₄, x₅} : Finset ℂ), f k) = 37 :=
by
  sorry

end prod_f_roots_l434_434905


namespace andy_cavities_l434_434340

/-- Andy gets a cavity for every 4 candy canes he eats.
  He gets 2 candy canes from his parents, 
  3 candy canes each from 4 teachers.
  He uses his allowance to buy 1/7 as many candy canes as he was given.
  Prove the total number of cavities Andy gets from eating all his candy canes is 4. -/
theorem andy_cavities :
  let canes_per_teach := 3
  let teachers := 4
  let canes_from_parents := 2
  let bought_fraction := 1 / 7
  let cavities_per_cane := 1 / 4 in
  let canes_from_teachers := canes_per_teach * teachers in
  let total_given := canes_from_teachers + canes_from_parents in
  let canes_bought := total_given * bought_fraction in
  let total_canes := total_given + canes_bought in
  let cavities := total_canes * cavities_per_cane in
  cavities = 4 := 
begin
  sorry
end

end andy_cavities_l434_434340


namespace probability_of_rerolling_two_dice_l434_434123

/-- Jason rolls three fair six-sided dice. Then he looks at the rolls and chooses a subset of the 
dice (possibly empty, possibly all three dice) to reroll. After rerolling, he wins if and only 
if the sum of the numbers face up on the three dice is exactly 7. Jason always plays to optimize 
his chances of winning. Prove that the probability he chooses to reroll exactly two of the dice 
is 7/36. -/
theorem probability_of_rerolling_two_dice :
  let win (dice : Fin 3 → ℕ) := (∑ i, dice i = 7) 
  let F := (Finset.finRange 7).val
  let ⦃optimize_strategy⦄ : Prop := sorry
  in (probability (reroll_exactly_two_and_win win F) = 7 / 36) :=
sorry

end probability_of_rerolling_two_dice_l434_434123


namespace proof_problem_l434_434017

variables (p q : Prop)

-- Assuming p is true and q is false
axiom p_is_true : p
axiom q_is_false : ¬ q

-- Proving that (¬p) ∨ (¬q) is true
theorem proof_problem : (¬p) ∨ (¬q) :=
by {
  sorry
}

end proof_problem_l434_434017


namespace geometric_sequence_sum_l434_434422

variables {α : Type*} [linear_ordered_field α]

theorem geometric_sequence_sum (a : ℕ → α)
  (h_geo : ∀ n, ∃ r, a (n + 1) = r * a n)
  (h_pos : ∀ n, 0 < a n)
  (h_cond : a 1 * a 3 + a 2 * a 4 + 2 * a 2 * a 3 = 49) : 
  a 2 + a 3 = 7 := 
sorry

end geometric_sequence_sum_l434_434422


namespace volunteer_selection_l434_434549

theorem volunteer_selection :
  let roles := ["translator", "tour guide", "etiquette", "driver"]
  let volunteers := ["A", "B", "C", "D", "E"] 
  -- The number of valid selection schemes where A and B can only be a translator or a tour guide.
  let valid_schemes := 5 * 2 * 2 in
  valid_schemes = 20 :=
by
  let roles := ["translator", "tour guide", "etiquette", "driver"]
  let volunteers := ["A", "B", "C", "D", "E"]
  let valid_schemes := 5 * 2 * 2
  have h : valid_schemes = 20, from sorry
  exact h

end volunteer_selection_l434_434549


namespace find_a_for_chord_length_l434_434073

theorem find_a_for_chord_length :
  ∀ a : ℝ, ((∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 ∧ (2 * x - y + a = 0)) 
  → ((2 * 1 - 1 + a = 0) → a = -1)) :=
by
  sorry

end find_a_for_chord_length_l434_434073


namespace area_of_triangle_l434_434503

-- Define the given curve equation
def curve (x : ℝ) := (x - 4)^2 * (x + 3)

-- The x and y intercepts of the curve, (x, y at x = 0)
def x_intercepts : set ℝ := {x | curve x = 0}
def y_intercept := curve 0

-- Vertices of the triangle
def vertex_A := (-3 : ℝ, 0 : ℝ)
def vertex_B := (4 : ℝ, 0 : ℝ)
def vertex_C := (0 : ℝ, y_intercept)

-- Calculate the base and height of the triangle
def base := 7
def height := y_intercept

-- The area of the triangle
def triangle_area := 1/2 * base * height

-- The theorem to prove
theorem area_of_triangle : triangle_area = 168 := by
  sorry

end area_of_triangle_l434_434503


namespace evaluate_function_at_3_l434_434857

theorem evaluate_function_at_3 :
  (∀ x : ℝ, f x = x^2) → f 3 = 9 := by
  intro h
  have : f 3 = 3^2 := by rw [h]
  rw [pow_two] at this
  exact this.symm.trans (by norm_num)

end evaluate_function_at_3_l434_434857


namespace exist_three_primes_sum_to_30_l434_434231

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def less_than_twenty (n : ℕ) : Prop := n < 20

theorem exist_three_primes_sum_to_30 : 
  ∃ A B C : ℕ, is_prime A ∧ is_prime B ∧ is_prime C ∧ 
  less_than_twenty A ∧ less_than_twenty B ∧ less_than_twenty C ∧ 
  A + B + C = 30 :=
by 
  -- assume A = 2, prime and less than 20
  -- find B, C such that B and C are primes less than 20 and A + B + C = 30
  sorry

end exist_three_primes_sum_to_30_l434_434231


namespace movie_theater_loss_l434_434311

theorem movie_theater_loss :
  let capacity := 50
  let ticket_price := 8.0
  let tickets_sold := 24
  (capacity * ticket_price - tickets_sold * ticket_price) = 208 := by
  sorry

end movie_theater_loss_l434_434311


namespace area_of_right_isosceles_triangle_l434_434100

-- Definitions of the conditions
def RightTriangle (A B C : Type) [inner_product_space ℝ A] (right_triangle : triangle A B C) : Prop :=
right_triangle.has_right_angle (angle A)

def IsoscelesRightTriangle (A B C : Type) [inner_product_space ℝ A] (isosceles_right_triangle : triangle A B C) : Prop :=
isosceles_right_triangle.has_right_angle (angle A) ∧
isosceles_right_triangle.angles_are_equal (angle B) (angle C)

def SideLength (A B : Type) [metric_space A] (length : dist A B) : ℝ :=
10

-- The proof problem statement
theorem area_of_right_isosceles_triangle {A B C : Type} [inner_product_space ℝ A] 
  (triangle_ABC : IsoscelesRightTriangle A B C)
  (length_AB : SideLength A B 10) : 
  AreaOfTriangle A B C = 50 := 
sorry

end area_of_right_isosceles_triangle_l434_434100


namespace sin_cos2x_decreasing_interval_l434_434813

theorem sin_cos2x_decreasing_interval :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ π) → 
  (∀ x' : ℝ, (π/8 ≤ x' ∧ x' ≤ 5*π/8) → 
   deriv (λ x : ℝ, sin (2*x) + cos (2*x)) x' ≤ 0) :=
by
  -- Proof will be filled here
  sorry

end sin_cos2x_decreasing_interval_l434_434813


namespace triangle_area_l434_434474

-- Define the given curve
def curve (x : ℝ) : ℝ := (x - 4) ^ 2 * (x + 3)

-- x-intercepts occur when y = 0
def x_intercepts : set ℝ := { x | curve x = 0 }

-- y-intercept occurs when x = 0
def y_intercept : ℝ := curve 0

-- Base of the triangle is the distance between the x-intercepts
def base_of_triangle : ℝ := max (4 : ℝ) (-3) - min (4 : ℝ) (-3)

-- Height of the triangle is the y-intercept value
def height_of_triangle : ℝ := y_intercept

-- Area of the triangle
def area_of_triangle (b h : ℝ) : ℝ := 1 / 2 * b * h

theorem triangle_area : area_of_triangle base_of_triangle height_of_triangle = 168 := by
  -- Stating the problem requires definitions of x-intercepts and y-intercept
  have hx : x_intercepts = {4, -3} := by
    sorry -- The proof for finding x-intercepts

  have hy : y_intercept = 48 := by
    sorry -- The proof for finding y-intercept

  -- Setup base and height using the intercepts
  have b : base_of_triangle = 7 := by
    -- Calculate the base from x_intercepts
    rw [hx]
    exact calc
      4 - (-3) = 4 + 3 := by ring
      ... = 7 := rfl

  have h : height_of_triangle = 48 := by
    -- height_of_triangle should be y_intercept which is 48
    rw [hy]

  -- Finally calculate the area
  have A : area_of_triangle base_of_triangle height_of_triangle = 1 / 2 * 7 * 48 := by
    rw [b, h]

  -- Explicitly calculate the numerical value
  exact calc
    1 / 2 * 7 * 48 = 1 / 2 * 336 := by ring
    ... = 168 := by norm_num

end triangle_area_l434_434474


namespace abs_inequality_solution_set_l434_434194

theorem abs_inequality_solution_set (x : ℝ) : -1 < x ∧ x < 1 ↔ |2*x - 1| - |x - 2| < 0 := by
  sorry

end abs_inequality_solution_set_l434_434194


namespace option_C_correct_l434_434065

theorem option_C_correct (m n : ℝ) (h : m > n) : (1/5) * m > (1/5) * n := 
by
  sorry

end option_C_correct_l434_434065


namespace problem_inequality_l434_434646

theorem problem_inequality (x y z : ℝ) (h1 : x + y + z = 0) (h2 : |x| + |y| + |z| ≤ 1) :
  x + (y / 2) + (z / 3) ≤ 1 / 3 :=
sorry

end problem_inequality_l434_434646


namespace rhombus_area_from_diagonals_l434_434543

def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_from_diagonals 
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 7) :
  rhombus_area d1 d2 = 70 :=
by
  rw [rhombus_area, h1, h2]
  norm_num

#check rhombus_area_from_diagonals

end rhombus_area_from_diagonals_l434_434543


namespace solve_a_l434_434080

theorem solve_a (a : ℝ) : 
  let Δ := (-2)^2 - 4 * 1 * (2 * a) in 
  Δ = 0 → a = 1 / 2 :=
by 
  let b := -2
  let c := 2 * a
  let Δ := b^2 - 4 * 1 * c
  intro h
  rw [←mul_assoc 4 1 c, ←mul_comm 1 c, mul_one] at Δ
  rw ←h at Δ
  sorry

end solve_a_l434_434080


namespace find_a_from_expansion_l434_434941

theorem find_a_from_expansion 
  (a : ℝ) 
  (h : (∑ r in finset.range 6, (-1) ^ r * nat.choose 5 r * (a ^ (5 - r) * x ^ r).eval x = -1)) 
  :
  a = 1 ∨ a = 9 := 
sorry

end find_a_from_expansion_l434_434941


namespace number_of_groups_l434_434699

-- Definitions of the conditions
def total_students := 32
def students_per_group := 6

-- Statement that needs to be proven
theorem number_of_groups (total_students = 32) (students_per_group = 6) : 
  32 / 6 = 5 := 
begin 
  rw [nat.div_eq_of_lt, nat.mul_sub_of_gt, nat.mul_sub_right_distrib, nat.mul_comm, nat.mul_comm];
  sorry
end

end number_of_groups_l434_434699


namespace solve_for_x_l434_434032

theorem solve_for_x (x : ℝ) (h : 3 * x + 1 = -(5 - 2 * x)) : x = -6 :=
by
  sorry

end solve_for_x_l434_434032


namespace Pascal_theorem_statement_l434_434641

noncomputable def is_collinear (A B C : Point) : Prop :=
  ∃ l : Line, A ∈ l ∧ B ∈ l ∧ C ∈ l

variables (A B C D E F G H K : Point)
variables (circle : Circle)
variables (hexagon : InscribedHexagon circle A B C D E F)

def PascalTheorem : Prop :=
  let G := intersect (line_through AB) (line_through DE) in
  let H := intersect (line_through BC) (line_through EF) in
  let K := intersect (line_through CD) (line_through FA) in
  is_collinear G H K

theorem Pascal_theorem_statement
  (hexagon : InscribedHexagon circle A B C D E F)
  : PascalTheorem A B C D E F G H K := 
sorry

end Pascal_theorem_statement_l434_434641


namespace minimum_boxes_to_eliminate_l434_434952

-- Defining the list representing the values of the boxes
def box_values : List Real := [0.05, 2, 10, 20, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                               500, 1000, 2000, 5000, 10000, 20000, 50000, 75000, 100000, 200000,
                               300000, 400000, 500000, 750000, 1000000]

-- Defining a predicate to check if a value is at least $75,000
def at_least_75000 (x : Real) : Prop := x >= 75000

-- Counting the number of boxes containing at least $75,000
def count_at_least_75000 : Nat := (box_values.filter at_least_75000).length

-- Proving the minimum number of boxes to eliminate to have one-third chance of holding at least $75,000
theorem minimum_boxes_to_eliminate : (30 - 27 = 3) :=
by sorry

end minimum_boxes_to_eliminate_l434_434952


namespace binomial_expansion_sum_eq_128_l434_434199

theorem binomial_expansion_sum_eq_128 (n : ℕ) :
  (∀ x : ℝ, (x^3 + 1 / real.sqrt x)^n).coeffs.sum = 128 → n = 7 :=
by 
  sorry

end binomial_expansion_sum_eq_128_l434_434199


namespace ratio_sum_pqr_uvw_l434_434514

theorem ratio_sum_pqr_uvw (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end ratio_sum_pqr_uvw_l434_434514


namespace find_p_l434_434891

-- Defining the given conditions as Lean statements
def point_A : ℝ × ℝ := (-2, 3)
def focus_P (p : ℝ) : ℝ × ℝ := (p, 0)
def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt (((B.1 - A.1) ^ 2) + ((B.2 - A.2) ^ 2))

theorem find_p (p : ℝ) (h : distance point_A (focus_P p) = 5) (hp : p > 0) : p = 2 :=
by
  sorry

end find_p_l434_434891


namespace sum_coords_B_l434_434623

def point (A B : ℝ × ℝ) : Prop :=
A = (0, 0) ∧ B.snd = 5 ∧ (B.snd - A.snd) / (B.fst - A.fst) = 3 / 4

theorem sum_coords_B (x : ℝ) :
  point (0, 0) (x, 5) → x + 5 = 35 / 3 :=
by
  intro h
  sorry

end sum_coords_B_l434_434623


namespace area_shaded_region_l434_434576

noncomputable def WX_square : ℝ := 6 * 6
noncomputable def side_length_WY : ℝ := 6
def area_square (side_length : ℝ) : ℝ := side_length * side_length
def radius_circle_from_WY (side_length_WY : ℝ) : ℝ := side_length_WY
def area_circle (radius : ℝ) : ℝ := Real.pi * radius * radius
def area_shaded (area_circle : ℝ) (area_square : ℝ) : ℝ := area_circle - area_square

theorem area_shaded_region :
  area_shaded (area_circle (radius_circle_from_WY side_length_WY)) (area_square side_length_WY) = 36 * Real.pi - 36 :=
  by sorry

end area_shaded_region_l434_434576


namespace triangle_area_is_168_l434_434494

def curve (x : ℝ) : ℝ :=
  (x - 4)^2 * (x + 3)

noncomputable def x_intercepts : set ℝ :=
  {x | curve x = 0}

noncomputable def y_intercept : ℝ :=
  curve 0

theorem triangle_area_is_168 :
  let base := 7 in
  let height := y_intercept in
  let area := (1 / 2) * base * height in
  area = 168 :=
by
  sorry

end triangle_area_is_168_l434_434494


namespace average_height_of_class_l434_434178

theorem average_height_of_class (n1 n2 : ℕ) (h1 h2 : ℕ) (total_students : ℕ) (avg1 avg2 : ℤ)
  (h_n1 : n1 = 30) (h_n2 : n2 = 10) (h_total : total_students = 40)
  (h_avg1 : avg1 = 160) (h_avg2 : avg2 = 156) :
  ((n1 * avg1 + n2 * avg2) / total_students = 159) := by
  -- number of girls in each group
  have h1 : n1 = 30, by rw h_n1
  have h2 : n2 = 10, by rw h_n2
  
  -- total number of students in class should be sum of n1 and n2
  have h_total : total_students = n1 + n2, by rw [h1, h2]; exact h_total
  
  -- average height of each group
  have avg1 : avg1 = 160, by rw h_avg1
  have avg2 : avg2 = 156, by rw h_avg2
  
  -- total height of each group
  let total1 := n1 * avg1
  let total2 := n2 * avg2
  
  -- total height of the whole class
  let total_height := total1 + total2
  
  -- average height of the whole class
  let class_avg := total_height / total_students
  
  -- final assertion
  exact calc
  class_avg = 6360 / 40 : sorry
  ... = 159 : sorry
  sorry

end average_height_of_class_l434_434178


namespace triangle_area_is_168_l434_434491

def curve (x : ℝ) : ℝ :=
  (x - 4)^2 * (x + 3)

noncomputable def x_intercepts : set ℝ :=
  {x | curve x = 0}

noncomputable def y_intercept : ℝ :=
  curve 0

theorem triangle_area_is_168 :
  let base := 7 in
  let height := y_intercept in
  let area := (1 / 2) * base * height in
  area = 168 :=
by
  sorry

end triangle_area_is_168_l434_434491


namespace value_of_expression_l434_434945

theorem value_of_expression (x : ℝ) (h : x = 1 + Real.sqrt 2) : x^4 - 4 * x^3 + 4 * x^2 + 4 = 5 := by
  sorry

end value_of_expression_l434_434945


namespace find_N_l434_434834

theorem find_N (N : ℕ) (hN : N > 1) (h1 : 2019 ≡ 1743 [MOD N]) (h2 : 3008 ≡ 2019 [MOD N]) : N = 23 :=
by
  sorry

end find_N_l434_434834


namespace remainder_3_pow_500_mod_17_l434_434224

theorem remainder_3_pow_500_mod_17 : (3^500) % 17 = 13 := 
by
  sorry

end remainder_3_pow_500_mod_17_l434_434224


namespace max_eccentricity_l434_434382

-- Given conditions
variables {a b c : ℝ} (hyp_a_pos : 0 < a) (hyp_b_pos : 0 < b)

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Focus of the hyperbola
def focus : Prop := c^2 = a^2 + b^2

-- Vertices of the hyperbola
def vertex_A : ℝ × ℝ := (-a, 0)
def vertex_B : ℝ × ℝ := (a, 0)

-- Point P on line x = c
variables {n : ℝ} {P : ℝ × ℝ} (hyp_P : P = (c, n))

-- Angle condition for APB
def angle_condition : Prop := ∠P A B = 60 * (Real.pi / 180)

-- Eccentricity
def eccentricity : ℝ := c / a

theorem max_eccentricity : 
  (∃ P, focus ∧ angle_condition ∧ (hyperbola vertex_A.1 vertex_A.2) ∧
  (hyperbola vertex_B.1 vertex_B.2)) → eccentricity ≤ (2 * Real.sqrt 3) / 3 :=
begin
  sorry 
end

end max_eccentricity_l434_434382


namespace champ_races_l434_434953

structure Conditions where
  total_athletes : ℕ := 216
  lanes_per_race : ℕ := 6
  athletes_per_lane : ℕ := 1
  first_place_advances : ∀ x, Prop

def num_of_races (c : Conditions) := 
  let first_round := c.total_athletes / c.lanes_per_race
  let second_round := first_round / c.lanes_per_race
  let final_round := 1
  first_round + second_round + final_round

theorem champ_races (c : Conditions) : num_of_races c = 43 :=
  sorry

end champ_races_l434_434953


namespace GOKU_cyclic_l434_434808

variable {A B C K U O G : Type}

-- Given conditions for the problem
variables (scalene_triangle : scalene_triangle A B C)
variables (AB_side_mediator_intersects : AB_side_mediator_intersects A B K U)
variables (AC_side_intersects : AC_side_intersects A C O G)

-- Definition of cyclic quadrilateral to be proven
theorem GOKU_cyclic (h1 : AB < AC) (h2 : AC < BC) (h3 : @scalene_triangle A B C)
(h4 : @AB_side_mediator_intersects A B K U) (h5 : @AC_side_intersects A C O G) :
  cyclic_quadrilateral G O K U :=
sorry

end GOKU_cyclic_l434_434808


namespace find_b_l434_434650

theorem find_b (b n : ℝ) (h_neg : b < 0) :
  (∀ x, x^2 + b * x + 1 / 4 = (x + n)^2 + 1 / 18) → b = - (Real.sqrt 7) / 3 :=
by
  sorry

end find_b_l434_434650


namespace smallest_number_mod_conditions_l434_434730

theorem smallest_number_mod_conditions :
  ∃ b : ℕ, b > 0 ∧ b % 3 = 2 ∧ b % 5 = 3 ∧ ∀ n : ℕ, (n > 0 ∧ n % 3 = 2 ∧ n % 5 = 3) → n ≥ b :=
begin
  use 8,
  split,
  { linarith },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  intros n h,
  cases h with n_pos h_cond,
  cases h_cond with h_mod3 h_mod5,
  have h := chinese_remainder_theorem 3 5 (by norm_num) (by norm_num_ge)
  (λ _, by norm_num) (λ _ _, by norm_num),
  specialize h 2 3,
  rcases h ⟨h_mod3, h_mod5⟩ with ⟨m, rfl⟩,
  linarith,
end

end smallest_number_mod_conditions_l434_434730


namespace circle_diameter_l434_434278

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434278


namespace tournament_chromatic_index_l434_434781

noncomputable def chromaticIndex {n : ℕ} (k : ℕ) (h₁ : 2^(k-1) < n) (h₂ : n ≤ 2^k) : ℕ :=
k

theorem tournament_chromatic_index (n k : ℕ) (h₁ : 2^(k-1) < n) (h₂ : n ≤ 2^k) :
  chromaticIndex k h₁ h₂ = k :=
by sorry

end tournament_chromatic_index_l434_434781


namespace cheetahs_increase_l434_434097

variables (C P : ℕ) (increase_pandas : ℕ) (x : ℕ)

-- Conditions
def initial_ratio := C = P / 3
def pandas_increase := increase_pandas = 6
def current_ratio := (C + x) / (P + increase_pandas) = 1 / 3

-- Proof goal: Increase in number of cheetahs is 2.
theorem cheetahs_increase (h₁ : initial_ratio) (h₂ : pandas_increase) (h₃ : current_ratio) : x = 2 :=
by
  -- Placeholder for proof
  sorry

end cheetahs_increase_l434_434097


namespace a_arithmetic_sequence_sum_b_sequence_l434_434011

noncomputable def a (n : ℕ) : ℕ := if n = 0 then 0 else 2 * n

axiom a1_eq : a 1 = 2
axiom recurrence_relation (n : ℕ) (hn : n ≠ 0) :
  (n + 1) * (a n)^2 + (a n) * (a (n+1)) - n * (a (n+1))^2 = 0

def b (n : ℕ) : ℚ := 4 / (a n * a (n+1))

def S (n : ℕ) : ℚ := ∑ i in finset.range n, b (i+1)

theorem a_arithmetic_sequence :
  ∀ n : ℕ, n ≠ 0 → a n = 2 * n := sorry

theorem sum_b_sequence (n : ℕ) : S n = n / (n + 1) := sorry

end a_arithmetic_sequence_sum_b_sequence_l434_434011


namespace average_score_of_juniors_l434_434089

theorem average_score_of_juniors :
  ∀ (N : ℕ) (junior_percent senior_percent overall_avg senior_avg : ℚ),
  junior_percent = 0.20 →
  senior_percent = 0.80 →
  overall_avg = 86 →
  senior_avg = 85 →
  (N * overall_avg - (N * senior_percent * senior_avg)) / (N * junior_percent) = 90 := 
by
  intros N junior_percent senior_percent overall_avg senior_avg
  intros h1 h2 h3 h4
  sorry

end average_score_of_juniors_l434_434089


namespace a_formula_correct_sum_b_correct_l434_434112

noncomputable def a : ℕ → ℕ
| 1 := 4
| (n+1) := a n + 2

def a_formula (n : ℕ) : ℕ := 2 * n + 2

theorem a_formula_correct (n : ℕ) (h : n > 0): a n = 2 * n + 2 :=
sorry

def b (n : ℕ) : ℕ := 2^n - 3 * n

def sum_b : ℕ := (List.range 10).sum (λ n, abs (b (n + 1)))

theorem sum_b_correct : sum_b = 1810 :=
sorry

end a_formula_correct_sum_b_correct_l434_434112


namespace completing_the_square_x_squared_plus_4x_plus_3_eq_0_l434_434226

theorem completing_the_square_x_squared_plus_4x_plus_3_eq_0 :
  (x : ℝ) → x^2 + 4 * x + 3 = 0 → (x + 2)^2 = 1 :=
by
  intros x h
  -- The actual proof will be provided here
  sorry

end completing_the_square_x_squared_plus_4x_plus_3_eq_0_l434_434226


namespace triangles_with_equal_perimeters_not_congruent_l434_434228

theorem triangles_with_equal_perimeters_not_congruent:
  ∃ (Δ1 Δ2 : Triangle), Δ1 ≠ Δ2 ∧ perimeter Δ1 = perimeter Δ2 :=
sorry

end triangles_with_equal_perimeters_not_congruent_l434_434228


namespace total_surface_area_correct_l434_434658

-- Definitions and assumptions based on the given conditions
def radius : ℝ := 10
def height : ℝ := 10

-- Total Surface Area calculation
def total_surface_area := 100 * Real.pi + 200 * Real.pi + 200 * Real.pi

-- Theorem statement to prove total surface area is 500π
theorem total_surface_area_correct :
  total_surface_area = 500 * Real.pi :=
by
  -- Proof goes here
  sorry

end total_surface_area_correct_l434_434658


namespace adam_and_simon_50_miles_apart_l434_434332

noncomputable def time_when_50_miles_apart (x : ℝ) : Prop :=
  let adam_distance := 10 * x
  let simon_distance := 8 * x
  (adam_distance^2 + simon_distance^2 = 50^2) 

theorem adam_and_simon_50_miles_apart : 
  ∃ x : ℝ, time_when_50_miles_apart x ∧ x = 50 / 12.8 := 
sorry

end adam_and_simon_50_miles_apart_l434_434332


namespace problem_proof_l434_434084
-- Import necessary libraries

-- Define the hypothesis and theorem
theorem problem_proof (A B C X C' P : Type) 
  [triangle : triangle A B C] 
  (hAX : angle_bisector A X)
  (hAC' : AC' = AC) 
  [linesegment : linesegment_on B C']
  [linesegment : linesegment_on A P] : 
  ∃ P ∈ AX, AP^2 = AC' * AB :=
sorry

end problem_proof_l434_434084


namespace sum_of_coordinates_l434_434634

theorem sum_of_coordinates (x y : ℚ) (h₁ : y = 5) (h₂ : (y - 0) / (x - 0) = 3/4) : x + y = 35/3 :=
by sorry

end sum_of_coordinates_l434_434634


namespace min_arg_z_l434_434037

noncomputable def z (x y : ℝ) := x + y * Complex.I

def satisfies_condition (x y : ℝ) : Prop :=
  Complex.abs (z x y + 3 - Real.sqrt 3 * Complex.I) = Real.sqrt 3

theorem min_arg_z (x y : ℝ) (h : satisfies_condition x y) :
  Complex.arg (z x y) = 5 * Real.pi / 6 := 
sorry

end min_arg_z_l434_434037


namespace value_of_expression_l434_434083

theorem value_of_expression (x y : ℤ) (h1 : x = -6) (h2 : y = -3) : 4 * (x - y) ^ 2 - x * y = 18 :=
by sorry

end value_of_expression_l434_434083


namespace ellipse_standard_form_constant_ab_for_E_l434_434428

variables {a b c k : ℝ} (x y : ℝ)
constants (h_a : a = sqrt(6)) (h_b : b = sqrt(2)) (h_eccentricity : c = 2)
noncomputable def ellipse_eq := (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_standard_form :
  ellipse_eq x y :=
by
  replace h_a : a^2 = 6 := by rw [h_a]; norm_num
  replace h_b : b^2 = 2 := by rw [h_b]; norm_num
  simp [ellipse_eq, h_a, h_b, h_eccentricity]
  sorry

variables (m : ℝ)
constants (A B : ℝ × ℝ)
def point_E := (7 / 3, 0)

theorem constant_ab_for_E :
  ∃ (m : ℝ), m = 7 / 3 ∧ (m^2 - 6) / (1 + 3 * k^2) = -5 / 9 :=
sorry

end ellipse_standard_form_constant_ab_for_E_l434_434428


namespace fraction_computation_l434_434375

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l434_434375


namespace ali_baba_max_coins_l434_434334

/-- Given a game setup where 100 gold coins are distributed into 10 piles of 10 coins each and Ali Baba can strategically place and rearrange cups of coins in selected piles, show that Ali Baba can ensure taking a maximum of 72 coins, even if the thief tries to maximize his own gain. -/
theorem ali_baba_max_coins :
  ∀ (piles : Fin 10 → ℕ),
    (∀ i, piles i = 10) →
    (AliBabaStrategy : ∀ (selected_piles : Fin 10 → Fin 4), Cups → RearrangedCups) →
    (ThiefStrategy : ∀ (rearranged_cups : RearrangedCups), RearrangedPiles) →
    ∃ (chosen_piles : Fin 10 → Fin 3), ∑ i in chosen_piles, piles i ≥ 72 :=
begin
  sorry
end

end ali_baba_max_coins_l434_434334


namespace minimum_k_l434_434879

variable {a b k : ℝ}

theorem minimum_k (h_a : a > 0) (h_b : b > 0) (h : ∀ a b : ℝ, a > 0 → b > 0 → (1 / a) + (1 / b) + (k / (a + b)) ≥ 0) : k ≥ -4 :=
sorry

end minimum_k_l434_434879


namespace distance_center_circle_line_l434_434403

theorem distance_center_circle_line :
  let center := (-1, 0)
  let line : ℝ → ℝ := λ x, x + 3
  let a := 1
  let b := -1
  let c := 3
  let d := (|a * center.1 + b * center.2 + c|) / (Real.sqrt (a ^ 2 + b ^ 2))
  (d = Real.sqrt 2) :=
by
  let center := (-1 : ℝ, 0)
  let a := 1
  let b := -1
  let c := 3
  let dist := (abs (a * center.1 + b * center.2 + c)) / (Real.sqrt (a ^ 2 + b ^ 2))
  show dist = Real.sqrt 2
  sorry

end distance_center_circle_line_l434_434403


namespace knights_enemies_structure_l434_434785

/-- There are n knights such that each knight has exactly three enemies, and the enemies of a knight's friends are also their enemies.
    Prove that n must be either 4 or 6. -/
theorem knights_enemies_structure (n : ℕ) (knights : Fin n → Fin n → Prop) 
  (H1 : ∀ i j, i ≠ j → (knights i j ∨ ¬knights i j)) 
  (H2 : ∀ i, (∑ j, if knights i j then 0 else 1) = 3) 
  (H3 : ∀ i j k, knights i j → ¬knights j k → ¬knights i k) : 
  n = 4 ∨ n = 6 := 
begin
  sorry
end

end knights_enemies_structure_l434_434785


namespace ways_to_distribute_5_balls_in_3_boxes_with_conditions_l434_434931

noncomputable def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) (min_in_box_c : ℕ) : ℕ :=
  let total_ways := boxes ^ balls
  let invalid_ways := (2 ^ balls) + (balls * (2 ^ (balls - 1)))
  total_ways - invalid_ways

theorem ways_to_distribute_5_balls_in_3_boxes_with_conditions : num_ways_to_distribute_balls 5 3 2 = 131 := by
  sorry

end ways_to_distribute_5_balls_in_3_boxes_with_conditions_l434_434931


namespace triangle_angle_sixty_degrees_l434_434638

theorem triangle_angle_sixty_degrees (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) : 
  ∃ (θ : ℝ), θ = 60 ∧ ∃ (a b c : ℝ), a * b * c ≠ 0 ∧ ∀ {α β γ : ℝ}, (a + b + c = α + β + γ + θ) := 
sorry

end triangle_angle_sixty_degrees_l434_434638


namespace angle_between_vectors_l434_434926

variable {V : Type} [RealInnerProductSpace V]

theorem angle_between_vectors (a b : V) (ha : ∥a∥ = 1) 
  (h₁ : inner (a + b) a = 0) (h₂ : inner (2 • a + b) b = 0) :
  angle a b = π * 3 / 4 :=
by
  sorry

end angle_between_vectors_l434_434926


namespace shape_is_plane_l434_434848

noncomputable def cylindrical_shape (r θ z c : ℝ) : Prop :=
  θ = c

theorem shape_is_plane (r z c : ℝ) :
  ∀ θ, cylindrical_shape r θ z c → ∃ (x y : ℝ), x = r * cos θ ∧ y = r * sin θ ∧ θ = c :=
by
  intros θ h
  use r * cos θ, r * sin θ
  simp [h]
  sorry

end shape_is_plane_l434_434848


namespace jane_performance_l434_434984

theorem jane_performance :
  ∃ (p w e : ℕ), 
  p + w + e = 15 ∧ 
  2 * p + 4 * w + 6 * e = 66 ∧ 
  e = p + 4 ∧ 
  w = 11 :=
by
  sorry

end jane_performance_l434_434984


namespace sum_of_series_l434_434380

theorem sum_of_series :
  ∑ n in Finset.range 99, (1 : ℝ) / (n + 1) / (n + 2) = 99 / 100 :=
by
  sorry

end sum_of_series_l434_434380


namespace journey_time_is_not_in_options_l434_434713

theorem journey_time_is_not_in_options :
  let d := 120
  let car_speed := 30
  let walk_speed := 5
  let t1 := d / 3 / car_speed -- initial car travel time till 40 miles (4/3 hours)
  let t2 := (d / walk_speed) -- total walk distance coverage by Harry
  let total_time := (260:ℝ)/15 -- total journey time (52/3 hours)
  (total_time = 17) ∨ (total_time = 18) ∨ (total_time = 19) ∨ (total_time = 16) = false :=
by
  -- Using the definitions and given conditions to setup the variables
  let d := 120
  let car_speed := 30
  let walk_speed := 5
  let t1 := (40:ℝ) / car_speed -- initial car travel time (4/3 hours)
  let t2 := total_time - t1 -- time taken for Harry's and Dick's adjustments
  let total_time := (260:ℝ) / 15

  -- Prove by contradiction that the calculated time does not match any options
  have h1 : total_time ≠ 16 := by sorry
  have h2 : total_time ≠ 17 := by sorry
  have h3 : total_time ≠ 18 := by sorry
  have h4 : total_time ≠ 19 := by sorry
  have h5 : total_time ≠ 16 ∧ total_time ≠ 17 ∧ total_time ≠ 18 ∧ total_time ≠ 19 := by
    exact ⟨h1, h2, h3, h4⟩
  exact h5

end journey_time_is_not_in_options_l434_434713


namespace athletes_meeting_time_and_overtakes_l434_434705

-- Define the constants for the problem
noncomputable def track_length : ℕ := 400
noncomputable def speed1 : ℕ := 155
noncomputable def speed2 : ℕ := 200
noncomputable def speed3 : ℕ := 275

-- The main theorem for the problem statement
theorem athletes_meeting_time_and_overtakes :
  ∃ (t : ℚ) (n_overtakes : ℕ), 
  (t = 80 / 3) ∧
  (n_overtakes = 13) ∧
  (∀ n : ℕ, n * (track_length / 45) = t) ∧
  (∀ k : ℕ, k * (track_length / 120) = t) ∧
  (∀ m : ℕ, m * (track_length / 75) = t) := 
sorry

end athletes_meeting_time_and_overtakes_l434_434705


namespace lines_parallel_a_2_or_3_l434_434521

-- Define the two lines l1 and l2
def l1 (a : ℝ) : (ℝ × ℝ) → Prop := λ p, (a-2) * p.1 + a * p.2 + 4 = 0
def l2 (a : ℝ) : (ℝ × ℝ) → Prop := λ p, (a-2) * p.1 + 3 * p.2 + 2 * a = 0

-- Define the condition for the lines to be parallel
def lines_parallel (a : ℝ) : Prop :=
  (a-2 ≠ 0) → (a ≠ 0) ∧ (3 * (a-2) = a * (a-2))

-- Define the theorem
theorem lines_parallel_a_2_or_3 (a : ℝ) : lines_parallel a → a = 2 ∨ a = 3 :=
by
  intros,
  sorry

end lines_parallel_a_2_or_3_l434_434521


namespace opposite_negative_nine_l434_434677

theorem opposite_negative_nine : 
  (∃ (y : ℤ), -9 + y = 0 ∧ y = 9) :=
by sorry

end opposite_negative_nine_l434_434677


namespace circle_diameter_problem_circle_diameter_l434_434285

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l434_434285


namespace paths_catalan_relation_catalan_paths_relation_l434_434997

def upstep := (1, 1)
def downstep := (1, -1)
def flatstep := (1, 0)

noncomputable def catalan (n : ℕ) : ℚ := (1 : ℚ) / (n + 1) * nat.choose (2 * n) n

def paths (n : ℕ) := 
  sorry -- Placeholder for paths function definition

theorem paths_catalan_relation (n : ℕ) : 
  paths n = ∑ k in Finset.range (n / 2 + 1), nat.choose n (2 * k) * catalan k := 
sorry

theorem catalan_paths_relation (n : ℕ) : 
  catalan n = ∑ i in Finset.range (2 * n + 1), (-1 : ℤ)^i * nat.choose (2 * n) i * paths (2 * n - i) := 
sorry

end paths_catalan_relation_catalan_paths_relation_l434_434997


namespace relative_positions_of_P_on_AB_l434_434324

theorem relative_positions_of_P_on_AB (A B P : ℝ) : 
  A ≤ B → (A ≤ P ∧ P ≤ B ∨ P = A ∨ P = B ∨ P < A ∨ P > B) :=
by
  intro hAB
  sorry

end relative_positions_of_P_on_AB_l434_434324


namespace max_value_trig_function_l434_434668

theorem max_value_trig_function :
  ∃ x : ℝ, cos x + sin x + cos x * sin x = 1/2 + real.sqrt 2 :=
sorry

end max_value_trig_function_l434_434668


namespace line_intersects_circle_and_focus_condition_l434_434671

variables {x y k : ℝ}

/-- The line l intersects the circle x^2 + y^2 + 2x - 4y + 1 = 0 at points A and B. If the midpoint of the chord AB is the focus of the parabola x^2 = 4y, then prove that the equation of the line l is x - y + 1 = 0. -/
theorem line_intersects_circle_and_focus_condition :
  (∃ l : ℝ → ℝ, (∀ x y : ℝ, l x = y) ∧
  (∀ A B : ℝ × ℝ, ∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (0, 1)) ∧
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧
  x^2 = 4*y ) → 
  (∀ x y : ℝ, x - y + 1 = 0) :=
sorry

end line_intersects_circle_and_focus_condition_l434_434671


namespace find_N_l434_434833

theorem find_N (N : ℕ) (hN : N > 1) (h1 : 2019 ≡ 1743 [MOD N]) (h2 : 3008 ≡ 2019 [MOD N]) : N = 23 :=
by
  sorry

end find_N_l434_434833


namespace oblique_asymptote_l434_434812

theorem oblique_asymptote :
  (∃ c: ℝ, ∀ x: ℝ, x ≠ -5/2 → abs ((3*x^2 + 10*x + 5)/(2*x + 5) - (3/2*x + 5/2)) < c/abs(x - (-5/2))) :=
sorry

end oblique_asymptote_l434_434812


namespace initial_books_in_cart_l434_434118

-- Definitions from the conditions
def history_books := 15
def fiction_books := 22
def childrens_books_before_find := 10
def childrens_books_misplaced := 5
def science_books_before_find := 8
def science_books_misplaced := 3
def biography_books := 12
def books_still_to_shelve := 20

-- Translate the proof problem into a Lean theorem statement
theorem initial_books_in_cart : (history_books + fiction_books + (childrens_books_before_find - childrens_books_misplaced) + (science_books_before_find - science_books_misplaced) + biography_books + books_still_to_shelve) = 79 := 
by 
  -- Subtract misplaced books from children and science sections to get effectively shelved books
  let childrens_books := childrens_books_before_find - childrens_books_misplaced
  let science_books := science_books_before_find - science_books_misplaced
  -- Calculate total books shelved and initial books in the cart
  have total_books_shelved := history_books + fiction_books + childrens_books + science_books + biography_books
  have initial_books := total_books_shelved + books_still_to_shelve
  -- Establish the given condition as the proof goal
  show initial_books = 79
  -- Simplify the expressions under the assumption
  sorry

end initial_books_in_cart_l434_434118


namespace expectation_fair_coin_5_tosses_l434_434714

noncomputable def fairCoinExpectation (n : ℕ) : ℚ :=
  n * (1/2)

theorem expectation_fair_coin_5_tosses :
  fairCoinExpectation 5 = 5 / 2 :=
by
  sorry

end expectation_fair_coin_5_tosses_l434_434714


namespace greatest_integer_less_than_neg_19_over_5_l434_434722

theorem greatest_integer_less_than_neg_19_over_5 : 
  let x := - (19 / 5 : ℚ) in
  ∃ n : ℤ, n < x ∧ (∀ m : ℤ, m < x → m ≤ n) := 
by 
  let x : ℚ := - (19 / 5)
  existsi (-4 : ℤ) 
  split 
  · norm_num 
    linarith
  · intros m hm 
    linarith

end greatest_integer_less_than_neg_19_over_5_l434_434722


namespace hyperbola_asymptotes_l434_434052

-- Define the parametric equations of the hyperbola C
def hyperbola_param_x (θ : ℝ) : ℝ := 3 * Real.sec θ
def hyperbola_param_y (θ : ℝ) : ℝ := 4 * Real.tan θ

-- Define the candidate parametric equations of lines
def line1_param_x (t : ℝ) : ℝ := -3 * t
def line1_param_y (t : ℝ) : ℝ := 4 * t

def line3_param_x (t : ℝ) : ℝ := (3 / 5) * t
def line3_param_y (t : ℝ) : ℝ := -(4 / 5) * t

def line5_param_x (t : ℝ) : ℝ := 3 + 3 * t
def line5_param_y (t : ℝ) : ℝ := -4 - 4 * t

theorem hyperbola_asymptotes :
  (∀ θ, ∃ x y, x = hyperbola_param_x θ ∧ y = hyperbola_param_y θ → (y / x = 4 / 3 ∨ y / x = -4 / 3)) ∧
  (∀ (t : ℝ), ((∃ x y, x = line1_param_x t ∧ y = line1_param_y t ∧ y / x = 4 / 3) ∨
                (∃ x y, x = line3_param_x t ∧ y = line3_param_y t ∧ y / x = -4 / 3) ∨
                (∃ x y, x = line5_param_x t ∧ y = line5_param_y t ∧ (y / x = 4 / 3 ∨ y / x = -4 / 3)))) := 
by
  sorry

end hyperbola_asymptotes_l434_434052


namespace greatest_integer_of_negative_fraction_l434_434720

-- Define the original fraction
def original_fraction : ℚ := -19 / 5

-- Define the greatest integer function
def greatest_integer_less_than (q : ℚ) : ℤ :=
  Int.floor q

-- The proof problem statement:
theorem greatest_integer_of_negative_fraction :
  greatest_integer_less_than original_fraction = -4 :=
sorry

end greatest_integer_of_negative_fraction_l434_434720


namespace no_2021_knights_possible_l434_434958

-- Define the general board size and the nature of its inhabitants
def board_size := 70
def occupants := { knight, liar }

-- Define the statements made by knights and liars.
def statement (cell : Fin board_size × Fin board_size) : Prop :=
  let (i, j) := cell in
  let row_knights := sorry in -- (count of knights in row i)
  let col_knights := sorry in -- (count of knights in column j)
  row_knights = col_knights

-- Main theorem: It is impossible to have exactly 2021 knights on a 70x70 board with given statements.
theorem no_2021_knights_possible 
: ∀ (cells : Fin board_size × Fin board_size → occupants), 
  (∀ (c : Fin board_size × Fin board_size), 
    (cells c = knight ↔ statement c)) → 
  (finset.univ.sum (λ (c : Fin board_size × Fin board_size), if cells c = knight then 1 else 0) ≠ 2021) :=
sorry

end no_2021_knights_possible_l434_434958


namespace find_sum_of_roots_l434_434583

open Real

theorem find_sum_of_roots (p q r s : ℝ): 
  r + s = 12 * p →
  r * s = 13 * q →
  p + q = 12 * r →
  p * q = 13 * s →
  p ≠ r →
  p + q + r + s = 2028 := by
  intros
  sorry

end find_sum_of_roots_l434_434583


namespace fraction_meaningful_range_l434_434075

variable (x : ℝ)

theorem fraction_meaningful_range (h : x - 2 ≠ 0) : x ≠ 2 :=
by
  sorry

end fraction_meaningful_range_l434_434075


namespace OA_squared_plus_OB_squared_l434_434887

-- Definitions of the conditions
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (x^2 / 4) + (y^2 / 3) = 1

def line_intersects_ellipse_at_two_points (l : ℝ → ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ is_on_ellipse A ∧ is_on_ellipse B ∧ ∀ t : ℝ, l(t) = t

def area_of_triangle_OAB (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  0.5 * abs (x1 * y2 - x2 * y1)

-- The proof goal
theorem OA_squared_plus_OB_squared (l : ℝ → ℝ) (A B : ℝ × ℝ) : 
  (¬ line_through_origin l) → 
  line_intersects_ellipse_at_two_points l → 
  area_of_triangle_OAB A B = sqrt 3 → 
  (A = (l 1, l 1)) →
  (B = (l 2, l 2)) →
  (dist (0, 0) A)^2 + (dist (0, 0) B)^2 = 7 :=
by
  sorry

/-- Definitions for preventing the line passing through the origin -/
def line_through_origin (l : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, 0 = l t

end OA_squared_plus_OB_squared_l434_434887


namespace triangle_area_eq_l434_434977

/-- In a triangle ABC, given that A = arccos(7/8), BC = a, and the altitude from vertex A 
     is equal to the sum of the other two altitudes, show that the area of triangle ABC 
     is (a^2 * sqrt(15)) / 4. -/
theorem triangle_area_eq (a : ℝ) (angle_A : ℝ) (h_angle : angle_A = Real.arccos (7/8))
    (BC : ℝ) (h_BC : BC = a) (H : ∀ (AC AB altitude_A altitude_C altitude_B : ℝ),
    AC = X → AB = Y → 
    altitude_A = (altitude_C + altitude_B) → 
    ∃ (S : ℝ), 
    S = (1/2) * X * Y * Real.sin angle_A ∧ 
    altitude_A = (2 * S / X) + (2 * S / Y) 
    → (X * Y) = 4 * (a^2) 
    → S = ((a^2 * Real.sqrt 15) / 4)) :
S = (a^2 * Real.sqrt 15) / 4 := sorry

end triangle_area_eq_l434_434977


namespace andy_cavities_l434_434342

def candy_canes_from_parents : ℕ := 2
def candy_canes_per_teacher : ℕ := 3
def number_of_teachers : ℕ := 4
def fraction_to_buy : ℚ := 1 / 7
def cavities_per_candies : ℕ := 4

theorem andy_cavities : (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers 
                         + (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers) * fraction_to_buy)
                         / cavities_per_candies = 4 := by
  sorry

end andy_cavities_l434_434342


namespace cost_per_night_l434_434989

variable (x : ℕ)

theorem cost_per_night (h : 3 * x - 100 = 650) : x = 250 :=
sorry

end cost_per_night_l434_434989


namespace thousandth_digit_sqrt_N_l434_434996

def N : ℕ := (10^1998 - 1) / 9

def sqrt_N : ℝ := real.sqrt N

theorem thousandth_digit_sqrt_N :
  (real.fract (sqrt_N * 10^1002)).to_digits_base 10 1000 = 3 :=
sorry

end thousandth_digit_sqrt_N_l434_434996


namespace range_of_k_for_no_fourth_quadrant_l434_434552

noncomputable def line_does_not_pass_through_fourth_quadrant_range (k : ℝ) : Prop :=
  ∀ x y : ℝ, y - 1 = k * (x - √3) → ¬(x > 0 ∧ y < 0)

theorem range_of_k_for_no_fourth_quadrant :
  {k : ℝ | line_does_not_pass_through_fourth_quadrant_range k} = set.Icc 0 (√3 / 3) :=
sorry

end range_of_k_for_no_fourth_quadrant_l434_434552


namespace problem_statement_l434_434135

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement :
  (∀ x : ℝ, f(x + 4) = f(x)) ∧
  (∀ x : ℝ, f(-x) = -f(x)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = 3x) →
  f(11.5) = -1.5 :=
begin
  sorry
end

end problem_statement_l434_434135


namespace minimum_value_expr_l434_434881

-- Define the conditions and the problem
axiom a_b_conditions (a b : ℝ) : a + b = 2 ∧ b > 0

-- Define the expression for which we want to find the minimum value
def expr (a b : ℝ) := (1 / (2 * |a|)) + (|a| / b)

-- State the theorem that proves the minimum value
theorem minimum_value_expr (a b : ℝ) (h : a + b = 2) (hb : b > 0) : 3 / 4 ≤ expr a b :=
by
  sorry  -- The proof is to be completed

end minimum_value_expr_l434_434881


namespace supreme_sports_package_channels_l434_434129

theorem supreme_sports_package_channels (c_start : ℕ) (c_removed1 : ℕ) (c_added1 : ℕ)
                                         (c_removed2 : ℕ) (c_added2 : ℕ)
                                         (c_final : ℕ)
                                         (net1 : ℕ) (net2 : ℕ) (c_mid : ℕ) :
  c_start = 150 →
  c_removed1 = 20 →
  c_added1 = 12 →
  c_removed2 = 10 →
  c_added2 = 8 →
  c_final = 147 →
  net1 = c_removed1 - c_added1 →
  net2 = c_removed2 - c_added2 →
  c_mid = c_start - net1 - net2 →
  c_final - c_mid = 7 :=
by
  intros
  sorry

end supreme_sports_package_channels_l434_434129


namespace limit_polynomial_fraction_l434_434238

theorem limit_polynomial_fraction : 
  (tendsto (λ x : ℝ, (3*x^4 - 2*x^3 + 6*x^2 + x - 2) / (9*x^4 + 5*x^3 + 7*x^2 + x + 1))
  at_top (𝓝 (1/3))) :=
by
  sorry

end limit_polynomial_fraction_l434_434238


namespace sqrt_x_minus_1_domain_l434_434530

theorem sqrt_x_minus_1_domain (x : ℝ) : (∃ y, y^2 = x - 1) → (x ≥ 1) := 
by 
  sorry

end sqrt_x_minus_1_domain_l434_434530


namespace find_initial_input_l434_434209

theorem find_initial_input :
  ∃ x : ℕ, 9 < x ∧ x < 100 ∧ let y := 2 * x in let z := (y % 10) * 10 + (y / 10) in z + 2 = 27 ∧ x = 26 :=
begin
  sorry
end

end find_initial_input_l434_434209


namespace triangle_area_l434_434498

-- Define the curve and points
noncomputable def curve (x : ℝ) : ℝ := (x-4)^2 * (x+3)

-- Define the x-intercepts
def x_intercept1 := 4
def x_intercept2 := -3

-- Define the y-intercept
def y_intercept := curve 0

-- Define the base and height of the triangle
def base : ℝ := x_intercept1 - x_intercept2
def height : ℝ := y_intercept

-- Statement of the problem: calculating the area of the triangle
theorem triangle_area : (1/2) * base * height = 168 := by
  sorry

end triangle_area_l434_434498


namespace length_of_PQ_l434_434106

theorem length_of_PQ (O P Q : Point) (circle : Circle) (square : Square) (M : Point)
  (r : ℝ) (A_circle : ℝ) (A_square : ℝ)
  (h1 : circle.center = O)
  (h2 : square.center = O)
  (h3 : r = 1)
  (h4 : A_circle = π * r^2)
  (h5 : A_square = square.side_length^2)
  (h6 : A_circle = A_square)
  (h7 : circle.intersects_side square P Q)
  (h8 : M = midpoint P Q)
  : length P Q = sqrt(4 - π) :=
sorry

end length_of_PQ_l434_434106


namespace fraction_computation_l434_434373

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l434_434373


namespace line_through_fixed_point_fixed_points_with_constant_slope_l434_434902

-- Point structure definition
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define curves C1 and C2
def curve_C1 (p : Point) : Prop :=
  p.x^2 + (p.y - 1/4)^2 = 1 ∧ p.y ≥ 1/4

def curve_C2 (p : Point) : Prop :=
  p.x^2 = 8 * p.y - 1 ∧ abs p.x ≥ 1

-- Line passing through fixed point for given perpendicularity condition
theorem line_through_fixed_point (A B M : Point) (l : ℝ → ℝ → Prop) :
  curve_C2 A → curve_C2 B →
  (∃ k b, ∀ x y, l x y ↔ y = k * x + b) →
  (M = ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩) →
  ((M.x = A.x ∧ M.y = (A.y + B.y) / 2) → A.x * B.x = -16) →
  ∀ x y, l x y → y = (17 / 8) := sorry

-- Existence of two fixed points on y-axis with constant slope product
theorem fixed_points_with_constant_slope (P T1 T2 M : Point) (l : ℝ → ℝ → Prop) :
  curve_C1 P →
  (T1 = ⟨0, -1⟩) →
  (T2 = ⟨0, 1⟩) →
  l P.x P.y →
  (∃ k b, ∀ x y, l x y ↔ y = k * x + b) →
  (M.y^2 - (M.x^2 / 16) = 1) →
  (M.x ≠ 0) →
  ((M.y + 1) / M.x) * ((M.y - 1) / M.x) = (1 / 16) := sorry

end line_through_fixed_point_fixed_points_with_constant_slope_l434_434902


namespace probability_exactly_three_heads_in_seven_tosses_l434_434302

def combinations (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) : ℚ :=
  (combinations n k) / (2^n : ℚ)

theorem probability_exactly_three_heads_in_seven_tosses :
  binomial_probability 7 3 = 35 / 128 := 
by 
  sorry

end probability_exactly_three_heads_in_seven_tosses_l434_434302


namespace number_of_integer_exponent_terms_l434_434971

noncomputable def terms_with_integer_exponent : ℕ → ℕ 
| n := (range n).countp (λ r, (12 - (5 * r / 6)) ∈ ℕ)

theorem number_of_integer_exponent_terms : terms_with_integer_exponent 24 = 5 := 
sorry

end number_of_integer_exponent_terms_l434_434971


namespace circle_diameter_l434_434259

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434259


namespace clara_sells_total_cookies_l434_434357

theorem clara_sells_total_cookies :
  let cookies_per_box_1 := 12
  let cookies_per_box_2 := 20
  let cookies_per_box_3 := 16
  let cookies_per_box_4 := 18
  let cookies_per_box_5 := 22

  let boxes_sold_1 := 50.5
  let boxes_sold_2 := 80.25
  let boxes_sold_3 := 70.75
  let boxes_sold_4 := 65.5
  let boxes_sold_5 := 55.25

  let total_cookies_1 := cookies_per_box_1 * boxes_sold_1
  let total_cookies_2 := cookies_per_box_2 * boxes_sold_2
  let total_cookies_3 := cookies_per_box_3 * boxes_sold_3
  let total_cookies_4 := cookies_per_box_4 * boxes_sold_4
  let total_cookies_5 := cookies_per_box_5 * boxes_sold_5

  let total_cookies := total_cookies_1 + total_cookies_2 + total_cookies_3 + total_cookies_4 + total_cookies_5

  total_cookies = 5737.5 :=
by
  sorry

end clara_sells_total_cookies_l434_434357


namespace age_comparison_l434_434765

variable (P A F X : ℕ)

theorem age_comparison :
  P = 50 →
  P = 5 / 4 * A →
  P = 5 / 6 * F →
  X = 50 - A →
  X = 10 :=
by { sorry }

end age_comparison_l434_434765


namespace arc_MBM_l434_434759

noncomputable def arc_degrees (A B C M : Point) (s : ℝ) : ℝ :=
  if is_right_isosceles_triangle ABC ∧
     (length (AB) = s) ∧
     (length (BC) = s) ∧
     is_tangent_circle B s BC ∧
     intersects_line_circle A C M B s
  then 180
  else 0

theorem arc_MBM'_degrees (A B C M M' : Point) (s : ℝ) :
  is_right_isosceles_triangle ABC ∧
  (length (AB) = s) ∧
  (length (BC) = s) ∧
  is_tangent_circle B s BC ∧
  intersects_line_circle A C M B s ∧
  reflection M BC M' → 
  arc_degrees A B C M s = 180 :=
by
  sorry

end arc_MBM_l434_434759


namespace constant_term_in_binomial_expansion_constant_term_expansion_l434_434134

noncomputable def integral_value : ℝ :=
  ∫ (x : ℝ) in -real.pi / 2 .. real.pi / 2, real.sqrt 2 * real.cos (x + real.pi / 4)

theorem constant_term_in_binomial_expansion :
  (∫ (x : ℝ) in -real.pi / 2 .. real.pi / 2, real.sqrt 2 * real.cos (x + real.pi / 4)) = 2 :=
begin
  -- Proof goes here
  sorry
end

theorem constant_term_expansion :
  let a := ∫ (x : ℝ) in -real.pi / 2 .. real.pi / 2, real.sqrt 2 * real.cos (x + real.pi / 4) in
  (a * real.sqrt(x) - 1 / real.sqrt(x))^6 = -160 :=
begin
  -- Proof goes here
  sorry
end

end constant_term_in_binomial_expansion_constant_term_expansion_l434_434134


namespace directly_above_156_is_133_l434_434734

def row_numbers (k : ℕ) : ℕ := 2 * k - 1

def total_numbers_up_to_row (k : ℕ) : ℕ := k * k

def find_row (n : ℕ) : ℕ :=
  Nat.sqrt (n + 1)

def position_in_row (n k : ℕ) : ℕ :=
  n - (total_numbers_up_to_row (k - 1)) + 1

def number_directly_above (n : ℕ) : ℕ :=
  let k := find_row n
  let pos := position_in_row n k
  (total_numbers_up_to_row (k - 1) - row_numbers (k - 1)) + pos + 1

theorem directly_above_156_is_133 : number_directly_above 156 = 133 := 
  by
  sorry

end directly_above_156_is_133_l434_434734


namespace james_pays_37_50_l434_434982

/-- 
James gets 20 singing lessons.
First lesson is free.
After the first 10 paid lessons, he only needs to pay for every other lesson.
Each lesson costs $5.
His uncle pays for half.
Prove that James pays $37.50.
--/

theorem james_pays_37_50 :
  let first_lessons := 1
  let total_lessons := 20
  let paid_lessons := 10
  let remaining_lessons := total_lessons - first_lessons - paid_lessons
  let paid_remaining_lessons := (remaining_lessons + 1) / 2
  let total_paid_lessons := paid_lessons + paid_remaining_lessons
  let cost_per_lesson := 5
  let total_payment := total_paid_lessons * cost_per_lesson
  let payment_by_james := total_payment / 2
  payment_by_james = 37.5 := 
by
  sorry

end james_pays_37_50_l434_434982


namespace circle_diameter_problem_circle_diameter_l434_434288

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l434_434288


namespace hyperbola_eccentricity_l434_434918

noncomputable def hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) := 
  { p : ℝ × ℝ | (p.1^2 / a^2 - p.2^2 / b^2 = 1) }

noncomputable def foci1 (c : ℝ) := (-c, 0)
noncomputable def foci2 (c : ℝ) := (c, 0)

theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : ∀ p ∈ hyperbola a b ha hb, ∃ q ∈ hyperbola a b ha hb, vector.parallel (p - q) ((c, 0) - (-c, 0)))
  (h2 : |(foci2 c - foci1 c)| = 3 * (λ p q : ℝ × ℝ, p.x - q.x = 0) (λ x : ℝ × ℝ, x ∈ hyperbola a b ha hb)) :
  ∃ e : ℝ, e = 3 := 
sorry

end hyperbola_eccentricity_l434_434918


namespace parametric_to_ray_l434_434686

-- Define the conditions
def parametric_conditions (t : ℝ) (x y : ℝ) :=
  x = real.sqrt t + 1 ∧ y = 1 - 2 * real.sqrt t

-- Define the desired conclusion
def represents_ray (x y : ℝ) :=
  ∃ (m c : ℝ), ∀ x, x >= 1 → y = m * x + c ∧ m = -2 ∧ c = 3

-- The main theorem stating the parametric equations describe a ray
theorem parametric_to_ray (t x y : ℝ) (h : parametric_conditions t x y) : represents_ray x y :=
by
  sorry

end parametric_to_ray_l434_434686


namespace program_output_l434_434794

def program_final_value (S N K : Nat) : Nat :=
  if K > 10 then S
  else program_final_value (S + N) (N + 2) (K + 1)

theorem program_output : 
  program_final_value 0 2 1 = 110 :=
by sorry

end program_output_l434_434794


namespace total_handshakes_is_316_l434_434788

def number_of_couples : ℕ := 15
def number_of_people : ℕ := number_of_couples * 2

def handshakes_among_men (n : ℕ) : ℕ := n * (n - 1) / 2
def handshakes_men_women (n : ℕ) : ℕ := n * (n - 1)
def handshakes_between_women : ℕ := 1
def total_handshakes (n : ℕ) : ℕ := handshakes_among_men n + handshakes_men_women n + handshakes_between_women

theorem total_handshakes_is_316 : total_handshakes number_of_couples = 316 :=
by
  sorry

end total_handshakes_is_316_l434_434788


namespace time_for_A_l434_434780

-- Given rates of pipes A, B, and C filling the tank
variable (A B C : ℝ)

-- Condition 1: Tank filled by all three pipes in 8 hours
def combined_rate := (A + B + C = 1/8)

-- Condition 2: Pipe C is twice as fast as B
def rate_C := (C = 2 * B)

-- Condition 3: Pipe B is twice as fast as A
def rate_B := (B = 2 * A)

-- Question: To prove that pipe A alone will take 56 hours to fill the tank
theorem time_for_A (h₁ : combined_rate A B C) (h₂ : rate_C B C) (h₃ : rate_B A B) : 
  1 / A = 56 :=
by {
  sorry
}

end time_for_A_l434_434780


namespace find_g_g2_l434_434515

def g (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 1

theorem find_g_g2 : g (g 2) = 2630 := by
  sorry

end find_g_g2_l434_434515


namespace determine_b_l434_434512

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (b : ℝ) (x : ℝ) : ℝ :=
  (sin (2 * x)) * log ((2 * x - 3) / (2 * x + b))

theorem determine_b : is_even_function (f 3) :=
sorry

end determine_b_l434_434512


namespace inverse_g_l434_434513

theorem inverse_g (g : ℝ → ℝ) (x : ℝ) (hx : g x = (-1/40)) 
  (hg : ∀ x, g x = (x^7 - 1) / 5) : 
  x = real.root 7 (7/8) :=
by
  sorry

end inverse_g_l434_434513


namespace carrot_servings_l434_434128

theorem carrot_servings (C : ℕ) 
  (H1 : ∀ (corn_servings : ℕ), corn_servings = 5 * C)
  (H2 : ∀ (green_bean_servings : ℕ) (corn_servings : ℕ), green_bean_servings = corn_servings / 2)
  (H3 : ∀ (plot_plants : ℕ), plot_plants = 9)
  (H4 : ∀ (total_servings : ℕ) 
         (carrot_servings : ℕ)
         (corn_servings : ℕ)
         (green_bean_servings : ℕ), 
         total_servings = carrot_servings + corn_servings + green_bean_servings ∧
         total_servings = 306) : 
  C = 4 := 
    sorry

end carrot_servings_l434_434128


namespace trailing_zeros_of_expression_l434_434509

theorem trailing_zeros_of_expression :
  let n := 999999
  let f := 6!
  (n^2 - f) % 10 = 0 -> False :=
begin
  -- Given definitions
  let n := 10^6 - 1,
  let f := 720,
  -- Expression: (999999^2 - 720)
  have h := (10^12 - 2 * 10^6 - 719),
  -- Analyze trailing zeros
  have h_trailing := h % 10,
  -- Prove trailing zeros count
  sorry
end

end trailing_zeros_of_expression_l434_434509


namespace square_root_domain_l434_434531

theorem square_root_domain (x : ℝ) : (∃ y, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end square_root_domain_l434_434531


namespace length_of_segment_AC_l434_434545

theorem length_of_segment_AC :
  ∀ (a b h: ℝ),
    (a = b) →
    (h = a * Real.sqrt 2) →
    (4 = (a + b - h) / 2) →
    a = 4 * Real.sqrt 2 + 8 :=
by
  sorry

end length_of_segment_AC_l434_434545


namespace graph_exists_two_vertices_with_same_degree_l434_434652

theorem graph_exists_two_vertices_with_same_degree (G : Graph) :
  ∃ (u v : G.vertices), u ≠ v ∧ G.degree u = G.degree v :=
by
  sorry

end graph_exists_two_vertices_with_same_degree_l434_434652


namespace hypotenuse_length_l434_434710

theorem hypotenuse_length (c : ℝ) (hC : c > 0) : hypotenuse = 2 * sqrt 3 * c :=
by 
  -- Defining the variables and conditions
  let x := sqrt 3 * c
  have eq1 : hypotenuse = 2 * x :=
    by 
      -- Here we would add a series of geometric and algebraic arguments as discussed in the solution
      -- Ultimately concluding with the value
      sorry

  show hypotenuse = 2 * sqrt 3 * c from eq1
  sorry

end hypotenuse_length_l434_434710


namespace fraction_computation_l434_434376

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l434_434376


namespace theater_loss_l434_434313

/-- 
A movie theater has a total capacity of 50 people and charges $8 per ticket.
On a Tuesday night, they only sold 24 tickets. 
Prove that the revenue lost by not selling out is $208.
-/
theorem theater_loss 
  (capacity : ℕ) 
  (price : ℕ) 
  (sold_tickets : ℕ) 
  (h_cap : capacity = 50) 
  (h_price : price = 8) 
  (h_sold : sold_tickets = 24) : 
  capacity * price - sold_tickets * price = 208 :=
by
  sorry

end theater_loss_l434_434313


namespace find_length_AH_l434_434973
noncomputable theory

-- Definitions as per the problem conditions
variable (O A B C D H: Type*)

-- Circumcenter of triangle ABC
def is_circumcenter (O : Type*) (A B C : Type*) : Prop := sorry

-- D is a point on the circumcircle of triangle OBC
def point_on_circumcircle (D O B C : Type*) : Prop := sorry

-- AB = AC
def is_isosceles (A B C : Type*) : Prop := sorry

-- H is the foot of the perpendicular from A to OD
def is_foot_of_perpendicular (A H : Type*) (OD : Type*) : Prop := sorry

-- Distances BD = 7 and DC = 3
def distance (X Y : Type*) (d : ℝ) : Prop := sorry

-- Stating the final goal
theorem find_length_AH (h1 : is_circumcenter O A B C)
                       (h2 : is_isosceles A B C)
                       (h3 : point_on_circumcircle D O B C)
                       (h4 : is_foot_of_perpendicular A H (OD : Type*))
                       (h5 : distance B D 7)
                       (h6 : distance D C 3) :
  distance A H 5 := sorry

end find_length_AH_l434_434973


namespace dress_designs_possible_l434_434301

theorem dress_designs_possible 
    (colors : ℕ) (materials_per_color : ℕ) (patterns : ℕ) 
    (h_colors : colors = 5) (h_materials : materials_per_color = 2) (h_patterns : patterns = 4) : 
    colors * materials_per_color * patterns = 40 := 
by
    -- fixed values from conditions
    have h1 : colors = 5 := h_colors,
    have h2 : materials_per_color = 2 := h_materials,
    have h3 : patterns = 4 := h_patterns,
    -- calculate the number of designs
    calc
    colors * materials_per_color * patterns 
        = 5 * 2 * 4 : by rw [h1, h2, h3]
    ... = 40 : by norm_num

end dress_designs_possible_l434_434301


namespace keiko_speed_l434_434991

theorem keiko_speed (w : ℕ) (t_diff : ℕ) (v : ℝ) : w = 6 ∧ t_diff = 36 ∧ v = 12 * π / t_diff → v = π / 3 := 
by 
  intros h 
  cases h with h1 h2 
  cases h2 with h3 h4 
  rw h1 at h4 
  rw h3 at h4 
  exact h4

end keiko_speed_l434_434991


namespace treadmill_sale_amount_l434_434712

variable T : ℝ

def chest_of_drawers := 0.5 * T
def television := 3 * T
def total := T + chest_of_drawers + television

theorem treadmill_sale_amount (h : total = 600) : T = 133.33 := 
by
  sorry

end treadmill_sale_amount_l434_434712


namespace problem_l434_434520

variables (f g : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f(x)
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g(x)

theorem problem (h_odd_f : is_odd f) (h_even_g : is_even g)
    (h : ∀ x, f x - g x = 2^x) :
    g 0 < f 2 ∧ f 2 < f 3 :=
begin
    sorry
end

end problem_l434_434520


namespace find_a_b_l434_434082

def inequality_solution (a b : ℝ) : Prop :=
  ∀ x, (sqrt x > a * x + (3/2)) ↔ (4 < x ∧ x < b)

theorem find_a_b (a b : ℝ) (h : inequality_solution a b) :
  a = 1 / 8 ∧ b = 36 :=
sorry

end find_a_b_l434_434082


namespace hair_cut_second_day_l434_434821

theorem hair_cut_second_day (cut_first_day cut_total : ℝ) (h1 : cut_first_day = 0.38) 
    (h2 : cut_total = 0.88) : ∃ cut_second_day : ℝ, cut_second_day = cut_total - cut_first_day ∧ cut_second_day = 0.50 := 
 by {
  use cut_total - cut_first_day,
  split,
  { refl },
  sorry
}

end hair_cut_second_day_l434_434821


namespace eval_sum_ceil_sqrt_l434_434823

theorem eval_sum_ceil_sqrt :
  (Int.ceil (sqrt 3) = 2) ∧ (Int.ceil (sqrt 33) = 6) ∧ (Int.ceil (sqrt 333) = 19) →
  (2 + 6 + 19) * 2 = 54 := by
  sorry

end eval_sum_ceil_sqrt_l434_434823


namespace circle_diameter_l434_434292

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l434_434292


namespace vector_difference_magnitude_l434_434468

variables {V : Type*} [InnerProductSpace ℝ V] (a b : V)

-- Conditions
def a_dot_b_zero : a ⬝ b = 0 := sorry
def norm_a_one : ∥a∥ = 1 := sorry
def norm_b_three : ∥b∥ = 3 := sorry

-- Statement
theorem vector_difference_magnitude :
  ∥a - b∥ = Real.sqrt 10 := by
  sorry

end vector_difference_magnitude_l434_434468


namespace ducks_cows_legs_l434_434950

theorem ducks_cows_legs (D C : ℕ) (L H X : ℤ)
  (hC : C = 13)
  (hL : L = 2 * D + 4 * C)
  (hH : H = D + C)
  (hCond : L = 3 * H + X) : X = 13 := by
  sorry

end ducks_cows_legs_l434_434950


namespace find_highest_score_l434_434179

-- Define the conditions for the proof
section
  variable {runs_innings : ℕ → ℕ}

  -- Total runs scored in 46 innings
  def total_runs (average num_innings : ℕ) : ℕ := average * num_innings
  def total_runs_46_innings := total_runs 60 46
  def total_runs_excluding_H_L := total_runs 58 44

  -- Evaluated difference and sum of scores
  def diff_H_and_L : ℕ := 180
  def sum_H_and_L : ℕ := total_runs_46_innings - total_runs_excluding_H_L

  -- Define the proof goal
  theorem find_highest_score (H L : ℕ)
    (h1 : H - L = diff_H_and_L)
    (h2 : H + L = sum_H_and_L) :
    H = 194 :=
  by
    sorry

end

end find_highest_score_l434_434179


namespace otimes_10_4_l434_434810

def otimes (a b : ℝ) : ℝ := a - (5 * a) / (2 * b)

theorem otimes_10_4 : otimes 10 4 = 3.75 := by
  sorry

end otimes_10_4_l434_434810


namespace parallel_lines_condition_l434_434855

-- Lean 4 statements
theorem parallel_lines_condition (a b l : ℝ) : (a ≠ 0 ∧ b ≠ 0) → (ab = 1) →
  (∀ x y : ℝ, ax + y = l ∧ x + by = 1 → ((a * by - 1 = 0) → (ax + y = l))) → 
  ((ax + y = l ∧ x + by = 1) ∧ ¬(a * b = 1 → ∃ x y : ℝ, (ax + y = l ∧ x + by = 1))) :=
  sorry

end parallel_lines_condition_l434_434855


namespace simplify_trig_expression_l434_434654

open Real

theorem simplify_trig_expression (α : ℝ) : 
  sin (2 * π - α)^2 + (cos (π + α) * cos (π - α)) + 1 = 2 := 
by 
  sorry

end simplify_trig_expression_l434_434654


namespace meaningful_square_root_l434_434526

theorem meaningful_square_root (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 := 
by {
  -- This part would contain the proof
  sorry
}

end meaningful_square_root_l434_434526


namespace find_N_l434_434824

theorem find_N :
  ∃ N : ℕ, N > 1 ∧
    (1743 % N = 2019 % N) ∧ (2019 % N = 3008 % N) ∧ N = 23 :=
by
  sorry

end find_N_l434_434824


namespace hyperbola_eccentricity_l434_434433

-- Define the given conditions
variables {a b c x₀ y₀ : ℝ} (F1 F2 A M N : ℝ × ℝ)
-- Define the hyperbola and its properties
def hyperbola : Prop := ∀ (x y : ℝ), (x / a)^2 - (y / b)^2 = 1

-- Define the foci F1 and F2 and the left vertex A of the hyperbola
def foci_left_right : Prop := F1 = (-(sqrt (a^2 + b^2)), 0) ∧ F2 = (sqrt (a^2 + b^2), 0)
def left_vertex : Prop := A = (-a, 0)

-- Circle with F1F2 as diameter
def circle_diameter : Prop := M = (a, b) ∧ N = (-a, -b) ∧ 
  (y₀ = b / a * x₀ ∧ x₀^2 + y₀^2 = c^2)

-- Angle at MAN given as 120 degrees
def angle_MAN : Prop := ∠ (A, M, N) = 120

-- Prove the eccentricity of the hyperbola
theorem hyperbola_eccentricity (h1 : hyperbola) (h2 : foci_left_right F1 F2) 
  (h3 : left_vertex A) (h4 : circle_diameter F1 F2 M N) (h5 : angle_MAN A M N) :
  e = sqrt(21) / 3 := sorry

end hyperbola_eccentricity_l434_434433


namespace sequence_property_l434_434591

theorem sequence_property 
  {a : ℕ → ℝ} {s : ℕ} 
  (h_pos : ∀ n, a n > 0)
  (h_max : ∀ n > s, a n = (finset.range (n - 1)).sup (λ k, a k + a (n - k))) :
  ∃ (ℓ N : ℕ), ℓ ≤ s ∧ (∀ n ≥ N, a n = a ℓ + a (n - ℓ)) :=
sorry

end sequence_property_l434_434591


namespace sin2alpha_div_1_plus_cos2alpha_eq_3_l434_434441

theorem sin2alpha_div_1_plus_cos2alpha_eq_3 (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 3 := 
  sorry

end sin2alpha_div_1_plus_cos2alpha_eq_3_l434_434441


namespace value_of_g_at_3_l434_434063

def g (x : ℝ) := x^2 + 1

theorem value_of_g_at_3 : g 3 = 10 := by
  sorry

end value_of_g_at_3_l434_434063


namespace primes_between_30_and_60_l434_434783

theorem primes_between_30_and_60 (list_of_primes : List ℕ) 
  (H1 : list_of_primes = [31, 37, 41, 43, 47, 53, 59]) :
  (list_of_primes.headI * list_of_primes.reverse.headI) = 1829 := by
  sorry

end primes_between_30_and_60_l434_434783


namespace point_B_coordinates_sum_l434_434620

theorem point_B_coordinates_sum (x : ℚ) (h1 : ∃ (B : ℚ × ℚ), B = (x, 5))
    (h2 : (5 - 0) / (x - 0) = 3/4) :
    x + 5 = 35/3 :=
by
  sorry

end point_B_coordinates_sum_l434_434620


namespace num_subintervals_minimum_l434_434116

noncomputable def f (x : ℝ) : ℝ := 1 / real.sqrt (x + 2)

def abs_error_midpoint_rule (a b : ℝ) (f f' : ℝ -> ℝ) (n : ℕ) (ε : ℝ) : Prop :=
  let M := real.sup (set.image (λ x, |f' x|) (set.Icc a b)) in
  (b - a) * (b - a) * M / (2 * n) ≤ ε

theorem num_subintervals_minimum (a b : ℝ) (ε : ℝ) (n : ℕ)
  (h_a : a = 2) (h_b : b = 7) (h_ε : ε = 0.1) 
  (h_n : n = 4) :
  abs_error_midpoint_rule a b f (λ x, -(1/2) * (x+2) ^ (-3/2)) n ε :=
by
  rw [h_a, h_b, h_ε, h_n]
  sorry

end num_subintervals_minimum_l434_434116


namespace valid_lineups_l434_434614

def total_players : ℕ := 15
def k : ℕ := 2  -- number of twins
def total_chosen : ℕ := 7
def remaining_players := total_players - k

def nCr (n r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.choose n r

def total_choices : ℕ := nCr total_players total_chosen
def restricted_choices : ℕ := nCr remaining_players (total_chosen - k)

theorem valid_lineups : total_choices - restricted_choices = 5148 := by
  sorry

end valid_lineups_l434_434614


namespace perpendicular_lines_not_both_perpendicular_to_plane_l434_434886

noncomputable theory

-- Definitions of entities
structure Line
structure Plane

-- Definitions from conditions
variable (a b : Line) (α : Plane) 

-- The property we need to prove
def lines_perpendicular (a b : Line) : Prop := sorry -- Define what it means for lines to be perpendicular
def line_perpendicular_to_plane (a : Line) (α: Plane) : Prop := sorry -- Define what it means for a line to be perpendicular to a plane

-- The theorem to prove
theorem perpendicular_lines_not_both_perpendicular_to_plane 
  (h1 : lines_perpendicular a b) 
  (h2 : line_perpendicular_to_plane a α)
  (h3 : line_perpendicular_to_plane b α) : False :=
sorry

end perpendicular_lines_not_both_perpendicular_to_plane_l434_434886


namespace arithmetic_sequence_problem_l434_434426

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_problem
  (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_cond : -a 2015 < a 1 ∧ a 1 < -a 2016) :
  sum_of_first_n_terms a 2015 > 0 ∧ sum_of_first_n_terms a 2016 < 0 :=
by
  sorry

end arithmetic_sequence_problem_l434_434426


namespace geoff_additional_votes_needed_l434_434094

-- Define the given conditions
def totalVotes : ℕ := 6000
def geoffPercentage : ℕ := 5 -- Represent 0.5% as 5 out of 1000 for better integer computation
def requiredPercentage : ℕ := 505 -- Represent 50.5% as 505 out of 1000 for better integer computation

-- Define the expressions for the number of votes received by Geoff and the votes required to win
def geoffVotes := (geoffPercentage * totalVotes) / 1000
def requiredVotes := (requiredPercentage * totalVotes) / 1000 + 1

-- The proposition to prove the additional number of votes needed for Geoff to win
theorem geoff_additional_votes_needed : requiredVotes - geoffVotes = 3001 := by sorry

end geoff_additional_votes_needed_l434_434094


namespace square_of_chord_length_l434_434104

noncomputable def length_of_chord_equality {r₁ r₂ d : ℝ} (h₁ : r₁ = 7)
  (h₂ : r₂ = 5) (h₃ : d = 10) (h₄ : r₁ + r₂ > d) (h₅ : r₁ < d + r₂) (h₆ : r₂ < d + r₁) : Prop :=
  let x := (P : ℝ) in
  QP = PR → (QP^2 = 98)

axiom QP : ℝ
axiom PR : ℝ
axiom cond_equal : QP = PR

theorem square_of_chord_length (r₁ r₂ d : ℝ) 
  (h₁: r₁ = 7) (h₂: r₂ = 5) (h₃: d = 10) 
  (h₄ : r₁ + r₂ > d) (h₅ : r₁ < d + r₂) (h₆ : r₂ < d + r₁)
  (h_eq: QP = PR) : QP^2 = 98 :=
by sorry

end square_of_chord_length_l434_434104


namespace line_through_fixed_point_l434_434002

theorem line_through_fixed_point 
  (A : Point := ⟨1, 0⟩)
  (B : Point)
  (C : Point)
  (M N : Point)
  (P : Point := ⟨1, 2⟩)
  (hB : on_x_axis B)
  (hAB_AC_eq : dist A B = dist A C)
  (hMidpoint : midpoint_on_y_axis B C)
  (hTrajectory_M : on_trajectory M (\(\Gamma\) := { (x, y) | y^2 = 4*x ∧ y ≠ 0 }))
  (hTrajectory_N : on_trajectory N (\Gamma\))
  (hSlopeSum : slope_sum_eq_two P M N)
  : passes_through_fixed_point (x := -1, y := 0) (line_eq M N) :=
sorry

def Point := (ℝ × ℝ)
def on_x_axis (B : Point) : Prop := B.2 = 0
def dist (p1 p2 : Point) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
def midpoint (p1 p2 : Point) : Point := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
def midpoint_on_y_axis (B C : Point) : Prop := (midpoint B C).1 = 0
def on_trajectory (P : Point) (Γ : set Point) : Prop := P ∈ Γ
def slope (p1 p2 : Point) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
def slope_sum_eq_two (P M N : Point) : Prop := slope P M + slope P N = 2
def line_eq (p1 p2 : Point) : set Point := { (x, y) | ∃m n, y = m*x + n }
def passes_through_fixed_point (fp : Point) (line : set Point) : Prop := fp ∈ line

instance : has_mem Point (set Point) := ⟨λ p s, s p⟩
instance : has_add ℝ := ⟨(+)⟩
instance : has_div ℝ := ⟨(/)⟩

end line_through_fixed_point_l434_434002


namespace point_lies_on_curve_l434_434044

def curve (x y : ℝ) : Prop := x^2 + x + y - 1 = 0
def point := (0 : ℝ, 1 : ℝ)

theorem point_lies_on_curve : curve point.1 point.2 :=
by 
  -- substitute and simplify the equation
  sorry

end point_lies_on_curve_l434_434044


namespace finite_set_condition_iff_l434_434836

theorem finite_set_condition_iff (S : Finset ℕ) : 
  (∃ (x : ℕ), x > 0 ∧ S = {x, 2 * x}) ↔ 
  (Finite S ∧ ∃ a b ∈ S, a ≠ b ∧ ∀ (a b ∈ S), a > b → (b^2 / (a - b) : ℕ) ∈ S) := by
  sorry

end finite_set_condition_iff_l434_434836


namespace each_piglet_ate_9_straws_l434_434715

theorem each_piglet_ate_9_straws (t : ℕ) (h_t : t = 300)
                                 (p : ℕ) (h_p : p = 20)
                                 (f : ℕ) (h_f : f = (3 * t / 5)) :
  f / p = 9 :=
by
  sorry

end each_piglet_ate_9_straws_l434_434715


namespace max_minus_min_value_l434_434916

noncomputable def f (x : ℝ) : ℝ :=
  4 * real.pi * real.arcsin x - (real.arccos (-x))^2

theorem max_minus_min_value : 
  let M := max (f 1) (f (-1)),
      m := min (f 1) (f (-1))
  in M - m = 3 * real.pi^2 := sorry

end max_minus_min_value_l434_434916


namespace number_of_power_functions_is_2_l434_434389

noncomputable def is_power_function (f : ℝ → ℝ) : Prop :=
∃ n : ℝ, ∀ x : ℝ, f x = x ^ n

def f1 : ℝ → ℝ := λ x, x^2
def f2 : ℝ → ℝ := λ x, (1/2)^x
def f3 : ℝ → ℝ := λ x, 4*x^2
def f4 : ℝ → ℝ := λ x, x^5 + 1
def f5 : ℝ → ℝ := λ x, (x - 1)^2
def f6 : ℝ → ℝ := λ x, x
def f7 (a : ℝ) (ha : 1 < a) : ℝ → ℝ := λ x, a^x

theorem number_of_power_functions_is_2 : 
  2 = (if (is_power_function f1) then 1 else 0)
      + (if (is_power_function f2) then 1 else 0)
      + (if (is_power_function f3) then 1 else 0)
      + (if (is_power_function f4) then 1 else 0)
      + (if (is_power_function f5) then 1 else 0)
      + (if (is_power_function f6) then 1 else 0)
      + (if ∀ a ha, is_power_function (f7 a ha) then 1 else 0) := 
by
  sorry

end number_of_power_functions_is_2_l434_434389


namespace area_of_proj_triangle_l434_434575

noncomputable def area_ratio (triangle_area : ℝ) (circumradius : ℝ) (op : ℝ) : ℝ :=
  (abs (circumradius ^ 2 - op ^ 2)) / (4 * circumradius ^ 2) * triangle_area

theorem area_of_proj_triangle (A B C P : ℝ × ℝ)
 (O : ℝ × ℝ)
 (circumradius : ℝ)
 (triangle_area : ℝ)
 (op_dist : ℝ)
 (h_p : op_dist = dist P O)
 (h_r : circumradius = dist A O / (2 * sin (angle_at_origin A B C)) )
 (perpendicular : ∀ (X Y Z P : ℝ × ℝ), ∃ (X1 Y1 Z1 : ℝ × ℝ),
    is_perpendicular (line_through P X1) YZ ∧
    is_perpendicular (line_through P Y1) ZX ∧
    is_perpendicular (line_through P Z1) XY ) :
  let A1 := some (perpendicular A B C P),
      B1 := some (perpendicular B C A P),
      C1 := some (perpendicular C A B P) in
  triangle_area_of A1 B1 C1 = area_ratio triangle_area circumradius op_dist :=
sorry

end area_of_proj_triangle_l434_434575


namespace primes_equal_if_sums_equal_l434_434636

theorem primes_equal_if_sums_equal
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : p + p^2 + p^3 + ... + p^q = q + q^2 + q^3 + ... + q^p)
  : p = q :=
sorry

end primes_equal_if_sums_equal_l434_434636


namespace correct_statements_l434_434049

-- Definitions based on the given problem conditions
noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.sin (3 * x + ϕ)
def ϕ_symmetry_condition : ℝ := -Real.pi / 4

-- Problem Conditions: 
def condition_1 : Prop := -Real.pi / 2 < ϕ_symmetry_condition ∧ ϕ_symmetry_condition < Real.pi / 2
def condition_2 : Prop := ∀ x, f (x + Real.pi / 12) ϕ_symmetry_condition = -f (-(x + Real.pi / 12)) ϕ_symmetry_condition
def condition_3 : Prop := ¬(∀ x ∈ Set.Icc (Real.pi / 12) (Real.pi / 3), f x ϕ_symmetry_condition ≤ f (x + 0.001) ϕ_symmetry_condition)
def condition_4 : Prop := |(3 * x₁ - Real.pi / 4) - (3 * x₂ - Real.pi / 4)| = 2 → |x₁ - x₂| = Real.pi / 3
def condition_5 : Prop := ∀ x, f (x - Real.pi / 4) ϕ_symmetry_condition ≠ -Real.cos (3 * x)

-- Combining all conditions into a single theorem statement
theorem correct_statements :
  condition_1 →
  condition_2 →
  condition_3 →
  condition_4 →
  condition_5 :=
by 
  sorry

end correct_statements_l434_434049


namespace final_speed_is_zero_l434_434328

-- Define physical constants and conversion
def initial_speed_kmh : ℝ := 189
def initial_speed_ms : ℝ := initial_speed_kmh * 0.277778
def deceleration : ℝ := -0.5
def distance : ℝ := 4000

-- The goal is to prove the final speed is 0 m/s
theorem final_speed_is_zero (v_i : ℝ) (a : ℝ) (d : ℝ) (v_f : ℝ) 
  (hv_i : v_i = initial_speed_ms) 
  (ha : a = deceleration) 
  (hd : d = distance) 
  (h : v_f^2 = v_i^2 + 2 * a * d) : 
  v_f = 0 := 
by 
  sorry 

end final_speed_is_zero_l434_434328


namespace circle_geometry_problem_l434_434802

variables {A B C D K O M N : Type} [inhabited A] [inhabited B] [inhabited C] [inhabited D]
variables [add_comm_group A] [module ℝ A]
variables (k : submodule ℝ A)

variables (circle : set A)
variables (center : A) 
variables (chord_AC chord_BD : set A) 

variables (K : A) 
variables (M_circ AKB : A) 
variables (N_circ CKD : A) 

variables (O : A)

-- Chords AC and BD of the circle with center O intersect at point K
variables (h1 : K ∈ chord_AC) (h2 : K ∈ chord_BD)

-- M and N are the centers of the circumscribed circles of triangles AKB and CKD respectively
variables (M_circ_center_AKB : is_circumcenter M_circ AKB)
variables (N_circ_center_CKD : is_circumcenter N_circ CKD)
variables (M_circ_center : is_in_circle M_circ circle)
variables (N_circ_center : is_in_circle N_circ circle)

-- To prove: OM = KN
theorem circle_geometry_problem (O M_circ N_circ : A) : dist O M_circ = dist K N_circ :=
sorry

end circle_geometry_problem_l434_434802


namespace hyperbola_equation_l434_434595

theorem hyperbola_equation (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_hyperbola : ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) 
  (h_focus : ∃ (p : ℝ × ℝ), p = (1, 0))
  (h_line_passing_focus : ∀ y, ∃ (m c : ℝ), y = -b * y + c)
  (h_parallel : ∀ x y : ℝ, b/a = -b)
  (h_perpendicular : ∀ x y : ℝ, b/a * (-b) = -1) : 
  ∀ x y : ℝ, x^2 - y^2 = 1 :=
by
  sorry

end hyperbola_equation_l434_434595


namespace Dima_broke_more_l434_434816

theorem Dima_broke_more (D F : ℕ) (h : 2 * D + 7 * F = 3 * (D + F)) : D = 4 * F :=
sorry

end Dima_broke_more_l434_434816


namespace smallest_positive_integer_remainder_l434_434732

theorem smallest_positive_integer_remainder : ∃ a : ℕ, 
  (a ≡ 2 [MOD 3]) ∧ (a ≡ 3 [MOD 5]) ∧ (a = 8) := 
by
  use 8
  split
  · exact Nat.modeq.symm (Nat.modeq.modeq_of_dvd _ (by norm_num))
  · split
    · exact Nat.modeq.symm (Nat.modeq.modeq_of_dvd _ (by norm_num))
    · rfl
  sorry  -- The detailed steps of the proof are omitted as per the instructions

end smallest_positive_integer_remainder_l434_434732


namespace prime_integer_roots_l434_434843

theorem prime_integer_roots (p : ℕ) (hp : Prime p) 
  (hroots : ∀ (x1 x2 : ℤ), x1 * x2 = -512 * p ∧ x1 + x2 = -p) : p = 2 :=
by
  -- Proof omitted
  sorry

end prime_integer_roots_l434_434843


namespace find_initial_height_l434_434786

theorem find_initial_height
  (bounce_height_ratio : ℝ)
  (total_distance : ℝ)
  (num_bounces : ℕ)
  (h : ℝ) :
  bounce_height_ratio = 0.5 →
  total_distance = 44.5 →
  num_bounces = 4 →
  (h + 2 * h * bounce_height_ratio + h * bounce_height_ratio^2 + (h * bounce_height_ratio^3 + h * bounce_height_ratio^4) / 2 + (h * bounce_height_ratio^4 / 4)) = total_distance →
  h = 9.9 :=
begin
  sorry
end

end find_initial_height_l434_434786


namespace sunny_weather_prob_correct_l434_434331

def rain_prob : ℝ := 0.45
def cloudy_prob : ℝ := 0.20
def sunny_prob : ℝ := 1 - rain_prob - cloudy_prob

theorem sunny_weather_prob_correct : sunny_prob = 0.35 := by
  sorry

end sunny_weather_prob_correct_l434_434331


namespace inradius_inscribed_circle_l434_434109

theorem inradius_inscribed_circle (PQ QR : ℝ) (angle_R_right : ∠PQR = 90) 
  (hPQ : PQ = 15) (hQR : QR = 8) : 
  let PR := Real.sqrt (PQ^2 + QR^2),
      A := (1/2) * PQ * QR,
      s := PQ + QR + PR,
      r := A / s 
  in r = 3 / 2 :=
by
  sorry

end inradius_inscribed_circle_l434_434109


namespace tangent_line_slope_range_l434_434689

def curve (x : ℝ) : ℝ := x^3 - real.sqrt 3 * x + 2

theorem tangent_line_slope_range :
  ∀ x : ℝ, 3 * x^2 - real.sqrt 3 ∈ set.Ioi (-real.sqrt 3) :=
by
  intro x
  -- Insert proof here
  sorry

end tangent_line_slope_range_l434_434689


namespace greatest_integer_less_than_neg_19_over_5_l434_434721

theorem greatest_integer_less_than_neg_19_over_5 : 
  let x := - (19 / 5 : ℚ) in
  ∃ n : ℤ, n < x ∧ (∀ m : ℤ, m < x → m ≤ n) := 
by 
  let x : ℚ := - (19 / 5)
  existsi (-4 : ℤ) 
  split 
  · norm_num 
    linarith
  · intros m hm 
    linarith

end greatest_integer_less_than_neg_19_over_5_l434_434721


namespace area_of_triangle_from_intercepts_l434_434488

theorem area_of_triangle_from_intercepts :
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := ({4, -3} : Set ℝ)
  let y_intercept := f 0
  let base := 4 - (-3)
  let height := y_intercept
  let area := 1 / 2 * base * height
  area = 168 := 
by
  -- Define the function f
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  -- Calculate x-intercepts
  have hx1 : f 4 = 0 := by simp [f, pow_two, mul_assoc]
  have hx2 : f (-3) = 0 := by simp [f, pow_two]
  let x_intercepts := ({4, -3} : Set ℝ)
  -- Calculate y-intercept
  have hy : f 0 = 48 := by simp [f, pow_two]
  let y_intercept := 48
  -- Define base and height
  let base := 4 - (-3)
  let height := y_intercept
  -- Compute the area
  let area := 1 / 2 * base * height
  -- Show that the area is 168
  show area = 168
  by
    simp [base, height, hy]
    norm_num
    sorry -- Skip the full proof

end area_of_triangle_from_intercepts_l434_434488


namespace area_of_triangle_from_intercepts_l434_434485

theorem area_of_triangle_from_intercepts :
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := ({4, -3} : Set ℝ)
  let y_intercept := f 0
  let base := 4 - (-3)
  let height := y_intercept
  let area := 1 / 2 * base * height
  area = 168 := 
by
  -- Define the function f
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  -- Calculate x-intercepts
  have hx1 : f 4 = 0 := by simp [f, pow_two, mul_assoc]
  have hx2 : f (-3) = 0 := by simp [f, pow_two]
  let x_intercepts := ({4, -3} : Set ℝ)
  -- Calculate y-intercept
  have hy : f 0 = 48 := by simp [f, pow_two]
  let y_intercept := 48
  -- Define base and height
  let base := 4 - (-3)
  let height := y_intercept
  -- Compute the area
  let area := 1 / 2 * base * height
  -- Show that the area is 168
  show area = 168
  by
    simp [base, height, hy]
    norm_num
    sorry -- Skip the full proof

end area_of_triangle_from_intercepts_l434_434485


namespace find_N_l434_434828

theorem find_N (N : ℕ) (h1 : N > 1)
  (h2 : ∀ (a b c : ℕ), (a ≡ b [MOD N]) ∧ (b ≡ c [MOD N]) ∧ (a ≡ c [MOD N])
  → (a = 1743) ∧ (b = 2019) ∧ (c = 3008)) :
  N = 23 :=
by
  have h3 : Nat.gcd (2019 - 1743) (3008 - 2019) = 23 := sorry
  show N = 23 from sorry

end find_N_l434_434828


namespace opposite_neg_9_l434_434678

theorem opposite_neg_9 : 
  ∃ y : Int, -9 + y = 0 ∧ y = 9 :=
by
  sorry

end opposite_neg_9_l434_434678


namespace expected_rank_of_winner_l434_434203

noncomputable def higher_rank_wins_probability := (3/5 : ℝ)

-- Function to determine the probability of a player winning in a given round
def win_probability (rank : ℕ) : ℝ :=
  if rank % 2 = 0 then higher_rank_wins_probability else 1 - higher_rank_wins_probability

-- Recursive function to compute the probability of winning all rounds
def total_win_probability (rank : ℕ) (rounds : ℕ) : ℝ :=
  nat.rec_on rounds
    1 -- Base case: probability of reaching the first round is 1
    (λ n prob, prob * win_probability rank) -- Recurrence: multiply by the win probability each round

-- Function to calculate the expected value of the winner's rank
def expected_winner_rank : ℝ :=
  let ranks := (list.range 256).map (λ r, r + 1) in
  let probabilities := ranks.map (λ r, total_win_probability r 8) in
  (ranks.zip probabilities).map (λ p, p.1 * p.2).sum

-- Theorem to assert the expected value is 103
theorem expected_rank_of_winner : expected_winner_rank = 103 := sorry

end expected_rank_of_winner_l434_434203


namespace log_decreasing_interval_correct_l434_434914

open Real

noncomputable def log_monotonically_decreasing_interval (a : ℝ) : Set ℝ := if h : a > 1 then {x | x < -3} else ∅

theorem log_decreasing_interval_correct (a : ℝ) (h₀ : a > 1) :
  (∀ x, log a (x^2 + 2 * x - 3) ∈ ℝ) →
  log_monotonically_decreasing_interval a = {x | x < -3} :=
by
  intros
  exact sorry

end log_decreasing_interval_correct_l434_434914


namespace star_problem_l434_434815

-- Define the star operation
def star (a b : ℝ) (h : a ≠ b) : ℝ := (a + b) / (a - b)

-- State the problem as a theorem
theorem star_problem : star (star 2 5 (by norm_num)) (-1) (by norm_num) = 5 / 2 := 
sorry

end star_problem_l434_434815


namespace number_of_valid_circle_arrangements_l434_434016

theorem number_of_valid_circle_arrangements (n : ℕ) (h_odd : odd n) (h_gt : n > 10) :
  let arrangements := {l : List ℕ | l.nodup ∧ l.length = n ∧ ∀ i, l.get (i % n) ∣ (l.get ((i - 1 + n) % n) + l.get ((i + 1) % n))} in
  (arrangements.card + arrangements.card.rotate = arrangements.card.reflect) = 2 :=
sorry

end number_of_valid_circle_arrangements_l434_434016


namespace quadratic_root_l434_434068

theorem quadratic_root (k : ℝ) (h : ∃ x : ℝ, x^2 - 2*k*x + k^2 = 0 ∧ x = -1) : k = -1 :=
sorry

end quadratic_root_l434_434068


namespace average_and_passing_percentage_l434_434098

def score_distribution : List (ℕ × ℕ) :=
  [ (95, 10), (85, 30), (75, 40), (65, 45), (55, 20), (45, 15) ]

def total_students : ℕ := 160

noncomputable def average_score : ℚ :=
  let total_score :=
    score_distribution.foldl (λ acc pair => acc + pair.1 * pair.2) 0
  total_score / total_students

noncomputable def percentage_scored_at_least_60 : ℚ :=
  let students_scored_at_least_60 :=
    score_distribution.foldl
      (λ acc pair => if pair.1 ≥ 60 then acc + pair.2 else acc)
      0
  (students_scored_at_least_60 / total_students) * 100

theorem average_and_passing_percentage :
  average_score = 70 ∧ percentage_scored_at_least_60 = 78.125 :=
by
  sorry

end average_and_passing_percentage_l434_434098


namespace area_above_line_l434_434221

-- Define the circle equation
def circle (x y : ℝ) : Prop := x^2 - 10*x + y^2 - 16*y + 56 = 0

-- Define the line above which we need to find the area
def line (y : ℝ) : Prop := y = 4

theorem area_above_line : 
  ∃ A : ℝ, (∃ O r : ℝ, (O = (5, 8) ∧ r = sqrt 33) ∧ ∃ f : ℝ → ℝ → ℝ, f = (λ x y, x^2 - 10*x + y^2 - 16*y + 56)) → A = 99 * (π / 4) :=
by
  sorry

end area_above_line_l434_434221


namespace product_expression_l434_434379

theorem product_expression :
  (3^4 - 1) / (3^4 + 1) * (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) * (6^4 - 1) / (6^4 + 1) * (7^4 - 1) / (7^4 + 1) = 880 / 91 := by
sorry

end product_expression_l434_434379


namespace interval_of_monotonic_increase_l434_434669

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem interval_of_monotonic_increase :
  (∃ α : ℝ, power_function α 2 = 4) →
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → power_function 2 x ≤ power_function 2 y) :=
by
  intro h
  sorry

end interval_of_monotonic_increase_l434_434669


namespace weekly_earnings_l434_434801

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def phone_repairs : ℕ := 5
def laptop_repairs : ℕ := 2
def computer_repairs : ℕ := 2

def total_earnings : ℕ := 
  phone_repairs * phone_repair_cost + 
  laptop_repairs * laptop_repair_cost + 
  computer_repairs * computer_repair_cost

theorem weekly_earnings : total_earnings = 121 := by
  sorry

end weekly_earnings_l434_434801


namespace geometric_sequence_solution_l434_434107

open_locale big_operators

variables {α : Type*} [field α] {a : ℕ → α} {q : α}

def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

variables (h_seq : geometric_sequence a q) (h_condition : a 4 + a 6 = 3)

theorem geometric_sequence_solution :
  a 5 * (a 3 + 2 * a 5 + a 7) = 9 :=
sorry

end geometric_sequence_solution_l434_434107


namespace problem_statement_l434_434137

def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 2 * x - 41

theorem problem_statement : g (g (g 10.5)) = g (g (g (-30.5))) := by
  sorry

end problem_statement_l434_434137


namespace parking_methods_count_l434_434349

theorem parking_methods_count : 
  let spaces := 6
  let cars := 3 
  (∀ i, 1 ≤ i → i ≤ spaces → (∃ j, 1 ≤ j → j ≤ cars → no_two_adjacent (i, j))) 
  → count_parking_methods(spaces, cars) = 24 := 
  by sorry

end parking_methods_count_l434_434349


namespace number_of_raccoons_l434_434985

/-- Jason pepper-sprays some raccoons and 6 times as many squirrels. 
Given that he pepper-sprays a total of 84 animals, the number of raccoons he pepper-sprays is 12. -/
theorem number_of_raccoons (R : Nat) (h1 : 84 = R + 6 * R) : R = 12 :=
by
  sorry

end number_of_raccoons_l434_434985


namespace quadratic_root_l434_434133

-- Problem conditions definitions
def isArithmeticSequence (a b c : ℝ) : Prop :=
  ∃ d : ℝ, b = a + d ∧ c = a + 2 * d

axiom nonneg_real {a b c : ℝ} (h : c ≥ b ∧ b ≥ a ∧ a ≥ 0)

-- Theorem statement
theorem quadratic_root (a b c : ℝ) (habc : isArithmeticSequence a b c) (nonneg : c ≥ b ∧ b ≥ a ∧ a ≥ 0) :
  ∃ r : ℝ, (c * r * r + b * r + a = 0) ∧ r = -1 - (Real.sqrt 3) / 3 :=
by
  sorry

end quadratic_root_l434_434133


namespace problem_statement_l434_434592

noncomputable def hyperbola (λ : ℝ) (P : ℝ × ℝ) : Prop :=
  P.1^2 - (P.2^2 / 2) = λ

def midpoint (P Q M : ℝ × ℝ) : Prop :=
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def perpendicular_bisector_intersect (P Q C D : ℝ × ℝ) (λ : ℝ) : Prop :=
  ∃ M : ℝ × ℝ, midpoint P Q M ∧ hyperbola λ M ∧ ∃ x : ℝ, M = (x, -x + 3)

def range_of_lambda (λ : ℝ) : Prop :=
  λ ∈ Ioo (-1 : ℝ) 0 ∪ Ioi 0

def points_concyclic (A B C D : ℝ × ℝ) : Prop :=
  ∃ O : ℝ × ℝ, ∃ r : ℝ, ∀ P ∈ {A, B, C, D}, (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2

theorem problem_statement (A B C D N : ℝ × ℝ) (λ : ℝ) :
  hyperbola λ A ∧ hyperbola λ B ∧ midpoint A B N ∧ N = (1, 2) ∧
  perpendicular_bisector_intersect A B C D λ →
  range_of_lambda λ ∧ points_concyclic A B C D :=
sorry

end problem_statement_l434_434592


namespace area_of_triangle_from_intercepts_l434_434487

theorem area_of_triangle_from_intercepts :
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := ({4, -3} : Set ℝ)
  let y_intercept := f 0
  let base := 4 - (-3)
  let height := y_intercept
  let area := 1 / 2 * base * height
  area = 168 := 
by
  -- Define the function f
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  -- Calculate x-intercepts
  have hx1 : f 4 = 0 := by simp [f, pow_two, mul_assoc]
  have hx2 : f (-3) = 0 := by simp [f, pow_two]
  let x_intercepts := ({4, -3} : Set ℝ)
  -- Calculate y-intercept
  have hy : f 0 = 48 := by simp [f, pow_two]
  let y_intercept := 48
  -- Define base and height
  let base := 4 - (-3)
  let height := y_intercept
  -- Compute the area
  let area := 1 / 2 * base * height
  -- Show that the area is 168
  show area = 168
  by
    simp [base, height, hy]
    norm_num
    sorry -- Skip the full proof

end area_of_triangle_from_intercepts_l434_434487


namespace max_ad_minus_bc_l434_434005

theorem max_ad_minus_bc (a b c d : ℤ) (ha : a ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hb : b ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hc : c ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hd : d ∈ Set.image (fun x => x) {(-1), 1, 2}) :
  ad - bc ≤ 6 :=
sorry

end max_ad_minus_bc_l434_434005


namespace movie_theater_loss_l434_434309

theorem movie_theater_loss :
  let capacity := 50
  let ticket_price := 8.0
  let tickets_sold := 24
  (capacity * ticket_price - tickets_sold * ticket_price) = 208 := by
  sorry

end movie_theater_loss_l434_434309


namespace even_function_k_value_l434_434942

theorem even_function_k_value (k : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = k * x^2 + (k - 1) * x + 2)
  (even_f : ∀ x : ℝ, f x = f (-x)) : k = 1 :=
by
  -- Proof would go here
  sorry

end even_function_k_value_l434_434942


namespace lottery_prize_l434_434601

-- Define the price of the first ticket and the increment for each successive ticket
def first_ticket := 1
def increment := 1

-- Define the number of tickets
def num_tickets := 5

-- Define the profit Lily plans to keep
def profit := 4

-- Define the function to calculate the price of the n-th ticket
def ticket_price (n : ℕ) : ℕ := first_ticket + (n * increment)

-- Define the total amount Lily collects from selling all 5 tickets
def total_amount : ℕ := (finset.range num_tickets).sum (λ n, ticket_price n)

-- Define the prize money for the winner
def prize_money : ℕ := total_amount - profit

-- State the theorem to prove
theorem lottery_prize : prize_money = 11 := by
  sorry

end lottery_prize_l434_434601


namespace day_of_week_after_1200_days_l434_434604

theorem day_of_week_after_1200_days (n : ℕ) (d : ℕ) 
  (h1 : n = 1200) 
  (h2 : d % 7 = 3) 
  (born_on_monday : (0 : ℕ) % 7 = 0) 
  : (d % 7 + 0) % 7 = 3 := 
by 
  rw [←h1, Nat.mod_eq_of_lt];
  norm_num;
  intros;
  sorry  -- Proof will be done here

end day_of_week_after_1200_days_l434_434604


namespace general_term_sum_b_sum_b_eq_l434_434463

variable (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
noncomputable theory

-- Given that a₁ = 2 and the sequence is arithmetic with a non-zero common difference.
axiom hav_a₁ : a 1 = 2
axiom arithmetic_seq (n : ℕ) : a (n+1) = a n + d
axiom non_zero_diff : d ≠ 0

-- Given that a₂, a₃, a₄ + 1 form a geometric sequence.
axiom geometric_seq : (a 2) * (a 4 + 1) = (a 3) ^ 2

-- Prove the general term formula for the arithmetic sequence aₙ.
theorem general_term : ∀ n, a n = 2 * n :=
begin
  sorry
end

-- Define the sequence bₙ based on the sequence aₙ.
def b (n : ℕ) : ℚ := 2 / (n * (a n + 2))

-- Prove the sum of the first n terms of the sequence bₙ.
theorem sum_b (n : ℕ) : S n = ∑ i in finset.range n, b (i + 1) := 
begin
  sorry
end

-- Prove that the sum of the first n terms of bₙ is n / (n + 1).
theorem sum_b_eq (n : ℕ) : S n = n / (n + 1) := 
begin
  sorry
end

end general_term_sum_b_sum_b_eq_l434_434463


namespace area_of_triangle_from_intercepts_l434_434484

theorem area_of_triangle_from_intercepts :
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := ({4, -3} : Set ℝ)
  let y_intercept := f 0
  let base := 4 - (-3)
  let height := y_intercept
  let area := 1 / 2 * base * height
  area = 168 := 
by
  -- Define the function f
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  -- Calculate x-intercepts
  have hx1 : f 4 = 0 := by simp [f, pow_two, mul_assoc]
  have hx2 : f (-3) = 0 := by simp [f, pow_two]
  let x_intercepts := ({4, -3} : Set ℝ)
  -- Calculate y-intercept
  have hy : f 0 = 48 := by simp [f, pow_two]
  let y_intercept := 48
  -- Define base and height
  let base := 4 - (-3)
  let height := y_intercept
  -- Compute the area
  let area := 1 / 2 * base * height
  -- Show that the area is 168
  show area = 168
  by
    simp [base, height, hy]
    norm_num
    sorry -- Skip the full proof

end area_of_triangle_from_intercepts_l434_434484


namespace athletes_meet_time_number_of_overtakes_l434_434708

-- Define the speeds of the athletes
def speed1 := 155 -- m/min
def speed2 := 200 -- m/min
def speed3 := 275 -- m/min

-- Define the total length of the track
def track_length := 400 -- meters

-- Prove the minimum time for the athletes to meet again is 80/3 minutes
theorem athletes_meet_time (speed1 speed2 speed3 track_length : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) :
  ∃ t : ℚ, t = (80 / 3 : ℚ) :=
by
  sorry

-- Prove the number of overtakes during this time is 13
theorem number_of_overtakes (speed1 speed2 speed3 track_length t : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) (h5 : t = 80 / 3) :
  ∃ n : ℕ, n = 13 :=
by
  sorry

end athletes_meet_time_number_of_overtakes_l434_434708


namespace opposite_of_neg_nine_is_nine_l434_434682

-- Define the predicate for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that needs to be proved
theorem opposite_of_neg_nine_is_nine : ∃ x, is_opposite (-9) x ∧ x = 9 := 
by {
  -- Use "sorry" to indicate the proof is not provided
  sorry
}

end opposite_of_neg_nine_is_nine_l434_434682


namespace sin_cos_identity_l434_434876

theorem sin_cos_identity (α : ℝ) (h : sin (π / 2 - 2 * α) = 3 / 5) : sin(α)^4 - cos(α)^4 = -3 / 5 :=
by
  sorry

end sin_cos_identity_l434_434876


namespace value_at_2pi_over_3_minimum_positive_period_interval_of_monotonic_increase_l434_434910

-- Define the function f
def f (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

-- Prove that f(2π/3) = 2
theorem value_at_2pi_over_3 : f (2 * Real.pi / 3) = 2 := by sorry

-- Prove that the minimum positive period of f is π
theorem minimum_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := by sorry

-- Prove the interval of monotonic increase
theorem interval_of_monotonic_increase : 
  ∀ k : ℤ, ∀ x, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3 → 
  ∀ y, x < y ∧ y < k * Real.pi + Real.pi / 6 ∨ k * Real.pi + 2 * Real.pi / 3 < y → f x < f y := by sorry

end value_at_2pi_over_3_minimum_positive_period_interval_of_monotonic_increase_l434_434910


namespace sum_row_12_pascals_triangle_sum_row_12_pascals_triangle_correct_l434_434846

theorem sum_row_12_pascals_triangle : ((∑ k in (finset.range 13), nat.choose 12 k) = 2^12) :=
by sorry

theorem sum_row_12_pascals_triangle_correct : ∑ k in (finset.range 13), nat.choose 12 k = 4096 :=
by 
  -- Using the property that sum of the elements in row n is 2^n
  have h : (∑ k in (finset.range 13), nat.choose 12 k) = 2^12 := sum_row_12_pascals_triangle
  -- Substituting 2^12 with its value 4096
  rw h
  simp

end sum_row_12_pascals_triangle_sum_row_12_pascals_triangle_correct_l434_434846


namespace initial_pups_per_mouse_l434_434793

-- Definitions from the problem's conditions
def initial_mice : ℕ := 8
def stress_factor : ℕ := 2
def second_round_pups : ℕ := 6
def total_mice : ℕ := 280

-- Define a variable for the initial number of pups each mouse had
variable (P : ℕ)

-- Lean statement to prove the number of initial pups per mouse
theorem initial_pups_per_mouse (P : ℕ) (initial_mice stress_factor second_round_pups total_mice : ℕ) :
  total_mice = initial_mice + initial_mice * P + (initial_mice + initial_mice * P) * second_round_pups - stress_factor * (initial_mice + initial_mice * P) → 
  P = 6 := 
by
  sorry

end initial_pups_per_mouse_l434_434793


namespace find_n_l434_434418

  def satisfies_condition (n : ℤ) : Prop :=
    (- (1 / 2 : ℚ))^n > (- (1 / 5 : ℚ))^n

  theorem find_n (n : ℤ) (h : n ∈ {-2, -1, 0, 1, 2, 3}) :
    satisfies_condition n ↔ n = -1 ∨ n = 2 :=
  by
    sorry
  
end find_n_l434_434418


namespace mutually_exclusive_events_l434_434737

-- Define the events based on the conditions
def at_least_two_heads (coin1 coin2 coin3 : Bool) : Prop :=
  list.count (=[true]) [coin1, coin2, coin3] ≥ 2

def no_more_than_one_head (coin1 coin2 coin3 : Bool) : Prop :=
  list.count (=[true]) [coin1, coin2, coin3] ≤ 1

-- Statement to be proven
theorem mutually_exclusive_events :
  ∀ (coin1 coin2 coin3 : Bool),
  ¬ (at_least_two_heads coin1 coin2 coin3 ∧ no_more_than_one_head coin1 coin2 coin3) :=
by sorry

end mutually_exclusive_events_l434_434737


namespace total_capacity_both_dressers_l434_434615

/-- Definition of drawers and capacity -/
def first_dresser_drawers : ℕ := 12
def first_dresser_capacity_per_drawer : ℕ := 8
def second_dresser_drawers : ℕ := 6
def second_dresser_capacity_per_drawer : ℕ := 10

/-- Theorem stating the total capacity of both dressers -/
theorem total_capacity_both_dressers :
  (first_dresser_drawers * first_dresser_capacity_per_drawer) +
  (second_dresser_drawers * second_dresser_capacity_per_drawer) = 156 :=
by sorry

end total_capacity_both_dressers_l434_434615


namespace power_function_characterization_l434_434522

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_characterization (f : ℝ → ℝ) (h : f 2 = Real.sqrt 2) : 
  ∀ x : ℝ, f x = x ^ (1 / 2) :=
sorry

end power_function_characterization_l434_434522


namespace peter_read_more_books_l434_434156

/-
Given conditions:
  Peter has 20 books.
  Peter has read 40% of them.
  Peter's brother has read 10% of them.
We aim to prove that Peter has read 6 more books than his brother.
-/

def total_books : ℕ := 20
def peter_read_fraction : ℚ := 0.4
def brother_read_fraction : ℚ := 0.1

def books_read_by_peter := total_books * peter_read_fraction
def books_read_by_brother := total_books * brother_read_fraction

theorem peter_read_more_books :
  books_read_by_peter - books_read_by_brother = 6 := by
  sorry

end peter_read_more_books_l434_434156


namespace induction_step_n_eq_1_l434_434635

theorem induction_step_n_eq_1 : (1 + 2 + 3 = (1+1)*(2*1+1)) :=
by
  -- Proof would go here
  sorry

end induction_step_n_eq_1_l434_434635


namespace penelope_mandm_candies_l434_434618

theorem penelope_mandm_candies (m n : ℕ) (r : ℝ) :
  (m / n = 5 / 3) → (n = 15) → (m = 25) :=
by
  sorry

end penelope_mandm_candies_l434_434618


namespace max_sum_reaches_at_8_or_9_l434_434541

noncomputable def geom_seq (a₃ a₅ : ℝ) (q : ℝ) (h1 : a₃ + a₅ = 5) (h2 : (a₃ / q) * (a₅ / q^3) = 4) (h3 : 0 < q ∧ q < 1) : ℕ → ℝ :=
λ n, 16 * q^(n-1)

def log_seq (a : ℕ → ℝ) : ℕ → ℝ :=
λ n, Real.log a n / Real.log 2

def sum_seq (b : ℕ → ℝ) : ℕ → ℝ :=
λ n, (n * (b 1 + b n)) / 2

def max_sum (S : ℕ → ℝ) : Prop :=
∀ n, ∑ k in finset.range(n+1), S k / (k+1) ≤ ∑ k in finset.range(10), S k / (k+1)

theorem max_sum_reaches_at_8_or_9 : 
  geom_seq 4 1 (1/2) (by norm_num) (by norm_num) ⟨by norm_num, by norm_num⟩ → 
  log_seq (geom_seq 4 1 (1/2) (by norm_num) (by norm_num) ⟨by norm_num, by norm_num⟩) → 
  sum_seq (log_seq (geom_seq 4 1 (1/2) (by norm_num) (by norm_num) ⟨by norm_num, by norm_num⟩)) →
  max_sum (sum_seq (log_seq (geom_seq 4 1 (1/2) (by norm_num) (by norm_num) ⟨by norm_num, by norm_num⟩))) :=
sorry

end max_sum_reaches_at_8_or_9_l434_434541


namespace equipment_value_decrease_l434_434696

theorem equipment_value_decrease (a : ℝ) (b : ℝ) (n : ℕ) :
  (a * (1 - b / 100)^n) = a * (1 - b/100)^n :=
sorry

end equipment_value_decrease_l434_434696


namespace fraction_computation_l434_434362

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l434_434362


namespace value_not_achievable_by_expression_l434_434738

theorem value_not_achievable_by_expression (x : ℝ) (hx : 0 < x) :
    ¬ (x^2 + 1/(4*x) = √3 - 1) := 
sorry

end value_not_achievable_by_expression_l434_434738


namespace triangle_area_l434_434477

-- Define the given curve
def curve (x : ℝ) : ℝ := (x - 4) ^ 2 * (x + 3)

-- x-intercepts occur when y = 0
def x_intercepts : set ℝ := { x | curve x = 0 }

-- y-intercept occurs when x = 0
def y_intercept : ℝ := curve 0

-- Base of the triangle is the distance between the x-intercepts
def base_of_triangle : ℝ := max (4 : ℝ) (-3) - min (4 : ℝ) (-3)

-- Height of the triangle is the y-intercept value
def height_of_triangle : ℝ := y_intercept

-- Area of the triangle
def area_of_triangle (b h : ℝ) : ℝ := 1 / 2 * b * h

theorem triangle_area : area_of_triangle base_of_triangle height_of_triangle = 168 := by
  -- Stating the problem requires definitions of x-intercepts and y-intercept
  have hx : x_intercepts = {4, -3} := by
    sorry -- The proof for finding x-intercepts

  have hy : y_intercept = 48 := by
    sorry -- The proof for finding y-intercept

  -- Setup base and height using the intercepts
  have b : base_of_triangle = 7 := by
    -- Calculate the base from x_intercepts
    rw [hx]
    exact calc
      4 - (-3) = 4 + 3 := by ring
      ... = 7 := rfl

  have h : height_of_triangle = 48 := by
    -- height_of_triangle should be y_intercept which is 48
    rw [hy]

  -- Finally calculate the area
  have A : area_of_triangle base_of_triangle height_of_triangle = 1 / 2 * 7 * 48 := by
    rw [b, h]

  -- Explicitly calculate the numerical value
  exact calc
    1 / 2 * 7 * 48 = 1 / 2 * 336 := by ring
    ... = 168 := by norm_num

end triangle_area_l434_434477


namespace point_B_coordinates_sum_l434_434621

theorem point_B_coordinates_sum (x : ℚ) (h1 : ∃ (B : ℚ × ℚ), B = (x, 5))
    (h2 : (5 - 0) / (x - 0) = 3/4) :
    x + 5 = 35/3 :=
by
  sorry

end point_B_coordinates_sum_l434_434621


namespace deficit_calculation_l434_434547

theorem deficit_calculation
    (L W : ℝ)  -- Length and Width
    (dW : ℝ)  -- Deficit in width
    (h1 : (1.08 * L) * (W - dW) = 1.026 * (L * W))  -- Condition on the calculated area
    : dW / W = 0.05 := 
by
    sorry

end deficit_calculation_l434_434547


namespace estimate_high_score_students_l434_434299

noncomputable def students_with_high_scores (num_students : ℕ) (mean : ℝ) (stddev : ℝ) (lower_bound upper_bound : ℝ) (prob_within_bounds : ℝ) : ℕ :=
  if (num_students = 50) ∧ (mean = 105) ∧ (stddev = 10) ∧ (lower_bound = 95) ∧ (upper_bound = 105) ∧ (prob_within_bounds = 0.32) then 9 else 0

theorem estimate_high_score_students :
  students_with_high_scores 50 105 10 95 105 0.32 = 9 :=
by sorry

end estimate_high_score_students_l434_434299


namespace zero_point_when_lambda_neg2_odd_function_lambda_value_lambda_range_on_interval_l434_434909

noncomputable def f (x : ℝ) (λ : ℝ) : ℝ := 3^x + λ * 3^(-x)

-- (1) Prove that when λ = -2, the zero point of the function f(x) is log_3(sqrt(2))
theorem zero_point_when_lambda_neg2 : 
  ∀ (x : ℝ), f x (-2) = 0 ↔ x = real.logb 3 (real.sqrt 2) := 
sorry

-- (2) Prove that if f(x) is an odd function, then λ = -1
theorem odd_function_lambda_value :
  (∀ x : ℝ, f (-x) λ = -f x λ) ↔ λ = -1 := 
sorry

-- (3) Prove that if 1/2 ≤ f(x) ≤ 4 on x ∈ [0,1], then λ ∈ [-1/2, 1] ∪ {3}
theorem lambda_range_on_interval :
  (∀ x ∈ Icc (0 : ℝ) 1, 1/2 ≤ f x λ ∧ f x λ ≤ 4) →
  λ ∈ set.Icc (-1/2) 1 ∪ {3} :=
sorry

end zero_point_when_lambda_neg2_odd_function_lambda_value_lambda_range_on_interval_l434_434909


namespace Product_a5_a6_arithmetic_log_sequence_l434_434944

theorem Product_a5_a6_arithmetic_log_sequence
  (a : ℕ → ℝ)
  (h1 : ∃ d : ℝ, ∀ n : ℕ, log 3 (a (n+1)) - log 3 (a n) = d)
  (h2 : (finset.range 10).sum (λ n, log 3 (a (n + 1))) = 10)
  : a 5 * a 6 = 9 :=
sorry

end Product_a5_a6_arithmetic_log_sequence_l434_434944


namespace triangle_area_l434_434476

-- Define the given curve
def curve (x : ℝ) : ℝ := (x - 4) ^ 2 * (x + 3)

-- x-intercepts occur when y = 0
def x_intercepts : set ℝ := { x | curve x = 0 }

-- y-intercept occurs when x = 0
def y_intercept : ℝ := curve 0

-- Base of the triangle is the distance between the x-intercepts
def base_of_triangle : ℝ := max (4 : ℝ) (-3) - min (4 : ℝ) (-3)

-- Height of the triangle is the y-intercept value
def height_of_triangle : ℝ := y_intercept

-- Area of the triangle
def area_of_triangle (b h : ℝ) : ℝ := 1 / 2 * b * h

theorem triangle_area : area_of_triangle base_of_triangle height_of_triangle = 168 := by
  -- Stating the problem requires definitions of x-intercepts and y-intercept
  have hx : x_intercepts = {4, -3} := by
    sorry -- The proof for finding x-intercepts

  have hy : y_intercept = 48 := by
    sorry -- The proof for finding y-intercept

  -- Setup base and height using the intercepts
  have b : base_of_triangle = 7 := by
    -- Calculate the base from x_intercepts
    rw [hx]
    exact calc
      4 - (-3) = 4 + 3 := by ring
      ... = 7 := rfl

  have h : height_of_triangle = 48 := by
    -- height_of_triangle should be y_intercept which is 48
    rw [hy]

  -- Finally calculate the area
  have A : area_of_triangle base_of_triangle height_of_triangle = 1 / 2 * 7 * 48 := by
    rw [b, h]

  -- Explicitly calculate the numerical value
  exact calc
    1 / 2 * 7 * 48 = 1 / 2 * 336 := by ring
    ... = 168 := by norm_num

end triangle_area_l434_434476


namespace verify_value_of_a_l434_434862

noncomputable def verify_a_value (a : ℝ) : Prop :=
  let A := {a^2, 2 - a, 4}
  in A.card = 3

theorem verify_value_of_a : ∃ (a : ℝ), verify_a_value a :=
begin
  use 6,
  sorry,
end

end verify_value_of_a_l434_434862


namespace largest_8_10_triple_l434_434800

def is_8_10_triple (N : ℕ) : Prop :=
  let N_base8 := Nat.digits 8 N; -- get the digits of N in base-8 representation
  let N_base8_to_N_base10 := Nat.ofDigits 10 N_base8; -- convert the base-8 digits to form a base-10 number
  3 * N = N_base8_to_N_base10 

theorem largest_8_10_triple : ∃ N : ℕ, is_8_10_triple N ∧ ∀ M : ℕ, is_8_10_triple M → M ≤ N :=
  ⟨273, -- specify the value
  begin
    sorry, -- proof for is_8_10_triple 273; omitted here
  end,
  begin
    sorry, -- proof for maximality; omitted here
  end⟩

end largest_8_10_triple_l434_434800


namespace find_a_b_solve_inequality_l434_434051

-- Defining the condition for part (Ⅰ)
def hasSolutionSet (a b : ℝ) : Prop :=
  ∀ x : ℝ, -3 < x ∧ x < 1 → ax^2 - bx + 3 > 0

-- Part (Ⅰ): Proving the values of a and b
theorem find_a_b (a b : ℝ) : 
  a = -1 ∧ b = 2 := 
  sorry

-- Defining the condition for part (Ⅱ)
def logInequalitySolution (b : ℝ) (x : ℝ) : Prop :=
  log b (2 * x - 1) ≤ 1 / (2 ^ a)

-- Part (Ⅱ): Proving the solution set of the inequality
theorem solve_inequality (x : ℝ) : 
  log 2 (2 * x - 1) ≤ 2 ↔ (1 / 2 < x ∧ x ≤ 5 / 2) :=
  sorry

end find_a_b_solve_inequality_l434_434051


namespace average_percentage_decrease_l434_434172

theorem average_percentage_decrease :
  ∃ (x : ℝ), (5000 * (1 - x / 100)^3 = 2560) ∧ x = 20 :=
by
  sorry

end average_percentage_decrease_l434_434172


namespace find_original_paycheck_l434_434234

theorem find_original_paycheck (P : ℝ) (h₀ : 0 < P) 
  (taxes_left : P * 0.20) 
  (saving_amount : P * 0.80 * 0.20 = 20) : 
  P = 125 :=
by
  sorry

end find_original_paycheck_l434_434234


namespace inequality_problem_l434_434585

variable (a b c : ℝ)

theorem inequality_problem (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
sorry

end inequality_problem_l434_434585


namespace work_ratio_of_man_to_boy_l434_434325

theorem work_ratio_of_man_to_boy 
  (M B : ℝ) 
  (work : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = work)
  (h2 : (13 * M + 24 * B) * 4 = work) :
  M / B = 2 :=
by 
  sorry

end work_ratio_of_man_to_boy_l434_434325


namespace binomial_variance_l434_434897

variables {p : ℝ} {ξ : ℝ}
variables (h1 : E ξ = 9) (h2: ∀ ξ, ξ ∼ B 18 p)

theorem binomial_variance : D ξ = 9 / 2 :=
by
  sorry

end binomial_variance_l434_434897


namespace circle_diameter_l434_434276

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l434_434276


namespace circle_diameter_l434_434260

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434260


namespace find_ratio_l434_434443

noncomputable def ratio (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a, b, c)

theorem find_ratio 
  (a b c : ℝ)
  (hpos_a : a > 0)
  (hpos_b : b > 0)
  (hpos_c : c > 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (harith : 2 * a = b + c)
  (hgeom : a ^ 2 = b * c) : 
  ratio a=b=c = (2:4:1) :=
sorry

end find_ratio_l434_434443


namespace cosine_expression_value_l434_434586

noncomputable def c : ℝ := 2 * Real.pi / 7

theorem cosine_expression_value :
  (Real.cos (3 * c) * Real.cos (5 * c) * Real.cos (6 * c)) / 
  (Real.cos c * Real.cos (2 * c) * Real.cos (3 * c)) = 1 :=
by
  sorry

end cosine_expression_value_l434_434586


namespace complement_intersection_l434_434922

/-- Given the universal set U={1,2,3,4,5},
    A={2,3,4}, and B={1,2,3}, 
    Prove the complement of (A ∩ B) in U is {1,4,5}. -/
theorem complement_intersection 
    (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) 
    (hU : U = {1, 2, 3, 4, 5})
    (hA : A = {2, 3, 4})
    (hB : B = {1, 2, 3}) :
    U \ (A ∩ B) = {1, 4, 5} :=
by
  -- proof goes here
  sorry

end complement_intersection_l434_434922


namespace hcf_of_numbers_l434_434690

theorem hcf_of_numbers (H : ∃ x, (4 * x) * (5 * x) = x * 80) : ∃ y, y = 2 :=
by
  obtain ⟨x, hx⟩ := H
  use 2
  sorry

end hcf_of_numbers_l434_434690


namespace triangle_angle_C_60_degrees_l434_434029

variables {A B C : ℝ}

theorem triangle_angle_C_60_degrees
  (h1 : ∀ (a b c : ℝ), a + b + c = sqrt 2 + 1 → ABC a b c)   -- Condition on the perimeter
  (h2 : ∀ (a b c : ℝ), sin A + sin B = sqrt 2 * sin C → ABC a b c) -- Condition on the sine sum
  (h3 : ∀ (a b c : ℝ), (1/2) * b * c * sin C = (1/6) * sin C → ABC a b c) -- Condition on the area
  : C = 60 := 
begin
  sorry
end

end triangle_angle_C_60_degrees_l434_434029


namespace one_percent_of_x_l434_434220

noncomputable def x := 89

theorem one_percent_of_x (h: 0.89 * 19 = 0.19 * x) : 0.01 * x = 0.89 := 
by 
  -- Sorry, proof is skipped
  sorry

end one_percent_of_x_l434_434220


namespace triangle_ratio_perimeter_l434_434964

theorem triangle_ratio_perimeter (AC BC : ℝ) (CD : ℝ) (AB : ℝ) (m n : ℕ) :
  AC = 15 → BC = 20 → AB = 25 → CD = 10 * Real.sqrt 3 →
  gcd m n = 1 → (2 * Real.sqrt ((AC * BC) / AB) + AB) / AB = m / n → m + n = 7 :=
by
  intros hAC hBC hAB hCD hmn hratio
  sorry

end triangle_ratio_perimeter_l434_434964


namespace max_regions_11_l434_434470

noncomputable def max_regions (n : ℕ) : ℕ :=
  1 + n * (n + 1) / 2

theorem max_regions_11 : max_regions 11 = 67 := by
  unfold max_regions
  norm_num

end max_regions_11_l434_434470


namespace elvin_fixed_charge_l434_434233

variable {F C : ℝ}

-- Conditions
def january_equation : Prop := F + C = 48
def february_equation : Prop := F + 2 * C = 90

-- Theorem statement: we want to prove F = 6 given the conditions
theorem elvin_fixed_charge : january_equation ∧ february_equation → F = 6 :=
by sorry

end elvin_fixed_charge_l434_434233


namespace arithmetic_sequence_a13_l434_434969

theorem arithmetic_sequence_a13 (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : a 5 = 3) (h2 : a 9 = 6) 
  (h3 : ∀ n, a n = a1 + (n - 1) * d) : 
  a 13 = 9 :=
sorry

end arithmetic_sequence_a13_l434_434969


namespace jason_probability_reroll_two_dice_l434_434120

-- Defining the problem conditions
def rolls : Type := list ℕ -- a representation of the three dice rolls, values between 1 and 6

-- Function to count favorable outcomes based on Jason's strategy
def count_favorable_outcomes (dice : rolls) : ℕ := sorry -- Detailed implementation omitted

-- Function to calculate probability (number of favorable outcomes / total outcomes)
def probability_of_rerolling_two_dice (dice : rolls) : ℚ := 
  (count_favorable_outcomes dice) / 216

-- Theorem: The probability that Jason chooses to reroll exactly two of the dice is 7/36
theorem jason_probability_reroll_two_dice :
  probability_of_rerolling_two_dice [1, 2, 3] = 7 / 36 :=
sorry

end jason_probability_reroll_two_dice_l434_434120


namespace sufficient_but_not_necessary_l434_434872

-- Definitions of propositions p and q
def p (a b m : ℝ) : Prop := a * m^2 < b * m^2
def q (a b : ℝ) : Prop := a < b

-- Problem statement as a Lean theorem
theorem sufficient_but_not_necessary (a b m : ℝ) : 
  (p a b m → q a b) ∧ (¬ (q a b → p a b m)) :=
by
  sorry

end sufficient_but_not_necessary_l434_434872


namespace fraction_computation_l434_434361

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l434_434361


namespace base_is_16_l434_434071

noncomputable def base_y_eq : Prop := ∃ base : ℕ, base ^ 8 = 4 ^ 16

theorem base_is_16 (base : ℕ) (h₁ : base ^ 8 = 4 ^ 16) : base = 16 :=
by
  sorry  -- Proof goes here

end base_is_16_l434_434071


namespace number_of_distinct_elements_l434_434035

noncomputable def f (n : ℕ) : ℂ := complex.I ^ n

theorem number_of_distinct_elements : 
  ∀ (S : finset ℂ), (S = finset.image f { n : ℕ | n > 0 }) → S.card = 4 :=
by
  sorry

end number_of_distinct_elements_l434_434035


namespace Alan_ate_1_fewer_pretzel_than_John_l434_434702

/-- Given that there are 95 pretzels in a bowl, John ate 28 pretzels, 
Marcus ate 12 more pretzels than John, and Marcus ate 40 pretzels,
prove that Alan ate 1 fewer pretzel than John. -/
theorem Alan_ate_1_fewer_pretzel_than_John 
  (h95 : 95 = 95)
  (John_ate : 28 = 28)
  (Marcus_ate_more : ∀ (x : ℕ), 40 = x + 12 → x = 28)
  (Marcus_ate : 40 = 40) :
  ∃ (Alan : ℕ), Alan = 27 ∧ 28 - Alan = 1 :=
by
  sorry

end Alan_ate_1_fewer_pretzel_than_John_l434_434702


namespace peter_read_more_books_l434_434155

/-
Given conditions:
  Peter has 20 books.
  Peter has read 40% of them.
  Peter's brother has read 10% of them.
We aim to prove that Peter has read 6 more books than his brother.
-/

def total_books : ℕ := 20
def peter_read_fraction : ℚ := 0.4
def brother_read_fraction : ℚ := 0.1

def books_read_by_peter := total_books * peter_read_fraction
def books_read_by_brother := total_books * brother_read_fraction

theorem peter_read_more_books :
  books_read_by_peter - books_read_by_brother = 6 := by
  sorry

end peter_read_more_books_l434_434155


namespace degree_odd_of_polynomials_l434_434837

theorem degree_odd_of_polynomials 
  (d : ℕ) 
  (P Q : Polynomial ℝ) 
  (hP_deg : P.degree = d) 
  (h_eq : P^2 + 1 = (X^2 + 1) * Q^2) 
  : Odd d :=
sorry

end degree_odd_of_polynomials_l434_434837


namespace value_b_100_l434_434144

def sequence_b : ℕ → ℕ
| 1       := 3
| (n + 1) := sequence_b n + 3 * n

theorem value_b_100 : sequence_b 100 = 14853 := 
by sorry

end value_b_100_l434_434144


namespace circle_diameter_l434_434275

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l434_434275


namespace sqrt_3_is_mian_l434_434551

-- Definitions based on the conditions
def is_mian (x : ℝ) : Prop :=
  ∀ n : ℤ, x ≠ n

-- The numbers given in the options
def sqrt_3 : ℝ := real.sqrt 3
def sqrt_4 : ℝ := real.sqrt 4
def sqrt_9 : ℝ := real.sqrt 9
def sqrt_16 : ℝ := real.sqrt 16

-- The statement that proves which number is "面"
theorem sqrt_3_is_mian : is_mian sqrt_3 :=
by sorry

end sqrt_3_is_mian_l434_434551


namespace fix_stick_two_nails_l434_434743

theorem fix_stick_two_nails (A B : Point) (lineAB : Line) 
  (h1 : A ∈ lineAB) (h2 : B ∈ lineAB): 
  ∀ (stick : Stick), stick ∩ lineAB = {A, B} → 
  stick.determined :
  two_points_determine_line :=
by 
  sorry

end fix_stick_two_nails_l434_434743


namespace required_run_rate_l434_434555

def remaining_run_rate (target total_overs first_overs first_run_rate remaining_overs : ℕ) := 
  float := (target - first_run_rate * first_overs) / remaining_overs

theorem required_run_rate :
  ∀ (target total_overs first_overs : ℕ) (first_run_rate : float), 
  first_overs = 10 →
  first_run_rate = 3.2 →
  target = 282 →
  total_overs = 40 →
  remaining_run_rate target total_overs first_overs first_run_rate (total_overs - first_overs) = 8.33 :=
by sorry

end required_run_rate_l434_434555


namespace t_value_l434_434069

def lcm (a b : ℕ) : ℕ := Nat.lcm a b
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem t_value :
  let a := 12^3
  let b := 16^2
  let c := 18^4
  let d := 24^5
  gcd (lcm a b) (lcm c d) = 6912 :=
by
  sorry

end t_value_l434_434069


namespace girls_doctors_percentage_l434_434091

-- Define the total number of students in the class
variables (total_students : ℕ)

-- Define the proportions given in the problem
def proportion_boys : ℚ := 3 / 5
def proportion_boys_who_want_to_be_doctors : ℚ := 1 / 3
def proportion_doctors_who_are_boys : ℚ := 2 / 5

-- Compute the proportion of boys in the class who want to be doctors
def proportion_boys_as_doctors := proportion_boys * proportion_boys_who_want_to_be_doctors

-- Compute the proportion of girls in the class
def proportion_girls := 1 - proportion_boys

-- Compute the number of girls who want to be doctors compared to boys
def proportion_girls_as_doctors := (1 - proportion_doctors_who_are_boys) / proportion_doctors_who_are_boys * proportion_boys_as_doctors

-- Compute the proportion of girls who want to be doctors
def proportion_girls_who_want_to_be_doctors := proportion_girls_as_doctors / proportion_girls

-- Define the expected percentage of girls who want to be doctors
def expected_percentage_girls_who_want_to_be_doctors : ℚ := 75 / 100

-- The theorem we need to prove
theorem girls_doctors_percentage : proportion_girls_who_want_to_be_doctors * 100 = expected_percentage_girls_who_want_to_be_doctors :=
sorry

end girls_doctors_percentage_l434_434091


namespace greatest_GCD_of_product_7200_l434_434217

theorem greatest_GCD_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧ ∀ d, (d ∣ a ∧ d ∣ b) → d ≤ 60 :=
by
  sorry

end greatest_GCD_of_product_7200_l434_434217


namespace find_certain_number_l434_434067

theorem find_certain_number (m : ℤ) (n : ℤ) (h1 : m = 6) (h2 : (-2)^(2*m) = 2^(n - m)) : n = 18 :=
by
  sorry

end find_certain_number_l434_434067


namespace gravitational_field_height_depth_equality_l434_434612

theorem gravitational_field_height_depth_equality
  (R G ρ : ℝ) (hR : R > 0) :
  ∃ x : ℝ, x = R * ((-1 + Real.sqrt 5) / 2) ∧
  (G * ρ * ((4 / 3) * Real.pi * R^3) / (R + x)^2 = G * ρ * ((4 / 3) * Real.pi * (R - x)^3) / (R - x)^2) :=
by
  sorry

end gravitational_field_height_depth_equality_l434_434612


namespace time_difference_after_race_l434_434146

noncomputable def malcolm_time (distance : ℕ) (speed : ℕ) : ℕ := distance * speed
noncomputable def joshua_time (distance : ℕ) (speed : ℕ) : ℕ := distance * speed

theorem time_difference_after_race :
  let malcolm_speed := 7 in
  let joshua_speed := 8 in
  let race_distance := 15 in
  let malcolm_finish_time := malcolm_time race_distance malcolm_speed in
  let joshua_finish_time := joshua_time race_distance joshua_speed in
  joshua_finish_time - malcolm_finish_time = 15 :=
by
  sorry

end time_difference_after_race_l434_434146


namespace find_N_l434_434832

theorem find_N (N : ℕ) (hN : N > 1) (h1 : 2019 ≡ 1743 [MOD N]) (h2 : 3008 ≡ 2019 [MOD N]) : N = 23 :=
by
  sorry

end find_N_l434_434832


namespace overall_support_is_68_l434_434779

-- Define the total population of men and women.
def men_count : ℕ := 200
def women_count : ℕ := 800

-- Define the percentage of men and women supporting the policy.
def men_support_percentage : ℝ := 0.60
def women_support_percentage : ℝ := 0.70

-- Calculate the number of supporters among men and women.
def men_support_count : ℕ := Nat.floor (men_support_percentage * men_count)
def women_support_count : ℕ := Nat.floor (women_support_percentage * women_count)

-- Define the total number of people surveyed and the total number of supporters.
def total_population : ℕ := men_count + women_count
def total_supporters : ℕ := men_support_count + women_support_count

-- Calculate the overall percentage of supporters in the population.
def overall_support_percentage : ℝ := (total_supporters.toReal / total_population.toReal) * 100

-- The theorem to be proved: The overall support percentage is 68%.
theorem overall_support_is_68 :
  overall_support_percentage = 68 :=
by
  sorry

end overall_support_is_68_l434_434779


namespace dot_product_correct_l434_434469

def cos45 : ℝ := Real.cos (Real.pi / 4) -- cos 45 degrees
def sin45 : ℝ := Real.sin (Real.pi / 4) -- sin 45 degrees
def cos15 : ℝ := Real.cos (Real.pi / 12) -- cos 15 degrees
def sin15 : ℝ := Real.sin (Real.pi / 12) -- sin 15 degrees

def vec_a : ℝ × ℝ := (cos45, sin45)
def vec_b : ℝ × ℝ := (cos15, sin15)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem dot_product_correct : dot_product vec_a vec_b = (Real.sqrt 3) / 2 := by
  sorry

end dot_product_correct_l434_434469


namespace find_x_l434_434735

theorem find_x :
  ∃ x : ℕ, (5 * 12) / (x / 3) + 80 = 81 ∧ x = 180 :=
by
  sorry

end find_x_l434_434735


namespace fraction_simplification_l434_434369

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l434_434369


namespace vector_projection_l434_434245

variables {a b : ℝ} (a_vec b_vec : ℝ → ℝ → ℝ)

def norm (v : ℝ → ℝ → ℝ) := sqrt ((v 0 0)^2 + (v 1 1)^2)
def dot (v1 v2 : ℝ → ℝ → ℝ) := (v1 0 0) * (v2 0 0) + (v1 1 1) * (v2 1 1)

theorem vector_projection (a b : ℝ) (a_vec b_vec : ℝ → ℝ → ℝ) (h1 : norm a_vec = 2)
  (h2 : dot a_vec (λ x y, a_vec x y + 2 * b_vec x y) = 0) :
  dot a_vec b_vec / norm a_vec = -1 :=
by sorry

end vector_projection_l434_434245


namespace problem_part_1_problem_part_2_problem_part_3_l434_434893

noncomputable def f : ℝ → ℝ := sorry

axiom domain_f : ∀ x, f x ∈ ℝ
axiom additivity_f : ∀ x y : ℝ, f (x + y) = f x + f y
axiom positivity_f : ∀ x : ℝ, x > 0 → f x > 0
axiom f_of_one : f 1 = 2

theorem problem_part_1 : f 0 = 0 ∧ f 3 = 6 := sorry

theorem problem_part_2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := sorry

theorem problem_part_3 : (∀ x : ℝ, f (4^x - a) + f (6 + 2^(x+1)) > 6) → a ≤ 3 := sorry

end problem_part_1_problem_part_2_problem_part_3_l434_434893


namespace domain_of_f_l434_434665
open Function

noncomputable def f (x : ℝ) : ℝ :=
  log (1 - 2 * x) / log 2 + 1 / (x + 1)

theorem domain_of_f : 
  ∀ x, (f x).is_defined ↔ (x ∈ (Set.Ioo (-(1 : ℝ)) ∞) ∪ Set.Ioo (-(1 : ℝ)) (1 / 2)) := 
by
  sorry

end domain_of_f_l434_434665


namespace meet_position_l434_434717

noncomputable def distance_A (t : ℕ) : ℕ := 4 * t

noncomputable def distance_B (t : ℕ) : ℕ := 3 * (2 ^ t - 1)

theorem meet_position (t : ℕ) : 4 * t + 3 * (2^t - 1) = 100 → 100 - 3 * (2^t - 1) - 4 * t = 7 :=
by
  intro h1
  calc
    100 - (3 * (2^t - 1)) - 4 * t
      = 100 - (3 * 2^t - 3) - 4 * t : by simp
  ... = 100 - 3 * 2^t + 3 - 4 * t : by rewrite [sub_eq_add_neg]; simp
  ... = 103 - 3 * 2^t - 4 * t : by sorry  /-
  ergodic steps to use the given h1 condition 
  - the actual calc steps to manipulate the equations  can be complex,
  but this serves the main purpose to initiate equivalence in Lean4
 -/
  ... = 7 : by rw [h1]; simp

#check  meet_position

end meet_position_l434_434717


namespace concurrent_diagonals_l434_434640

structure Hexagon (A B C D E F : Type) :=
  (inscribed_circle : Prop)
  (R : A)
  (Q : B)
  (T : C)
  (S : D)
  (P : E)
  (U : F)

theorem concurrent_diagonals 
  (A B C D E F R Q T S P U : Type) 
  (hex : Hexagon A B C D E F)
  (tangency_R : R ∈ set.range (λ x : ℝ, x * A))
  (tangency_Q : Q ∈ set.range (λ x : ℝ, x * B))
  (tangency_T : T ∈ set.range (λ x : ℝ, x * C))
  (tangency_S : S ∈ set.range (λ x : ℝ, x * D))
  (tangency_P : P ∈ set.range (λ x : ℝ, x * E))
  (tangency_U : U ∈ set.range (λ x : ℝ, x * F)) :
  ∃ X, ∃ AD BE CF : Type, X ∈ (set.range (λ x : ℝ, x * (AD))) ∧ X ∈ (set.range (λ x : ℝ, x * (BE))) ∧ X ∈ (set.range (λ x : ℝ, x * (CF))) := sorry

end concurrent_diagonals_l434_434640


namespace fare_from_midpoint_C_to_B_l434_434201

noncomputable def taxi_fare (d : ℝ) : ℝ :=
  if d <= 5 then 10.8 else 10.8 + 1.2 * (d - 5)

theorem fare_from_midpoint_C_to_B (x : ℝ) (h1 : taxi_fare x = 24)
    (h2 : taxi_fare (x - 0.46) = 24) :
    taxi_fare (x / 2) = 14.4 :=
by
  sorry

end fare_from_midpoint_C_to_B_l434_434201


namespace point_B_coordinates_sum_l434_434622

theorem point_B_coordinates_sum (x : ℚ) (h1 : ∃ (B : ℚ × ℚ), B = (x, 5))
    (h2 : (5 - 0) / (x - 0) = 3/4) :
    x + 5 = 35/3 :=
by
  sorry

end point_B_coordinates_sum_l434_434622


namespace triangle_area_is_168_l434_434493

def curve (x : ℝ) : ℝ :=
  (x - 4)^2 * (x + 3)

noncomputable def x_intercepts : set ℝ :=
  {x | curve x = 0}

noncomputable def y_intercept : ℝ :=
  curve 0

theorem triangle_area_is_168 :
  let base := 7 in
  let height := y_intercept in
  let area := (1 / 2) * base * height in
  area = 168 :=
by
  sorry

end triangle_area_is_168_l434_434493


namespace reciprocal_roots_sum_of_roots_minimum_c_l434_434524

-- Problem 1
theorem reciprocal_roots (m n : ℝ) (h : n ≠ 0) :
  quadratic_eq_roots m n ↔ quadratic_eq_reciprocal_roots m n :=
sorry

-- Problem 2
theorem sum_of_roots (a b : ℝ) (ha : a^2 - 15 * a - 5 = 0) (hb : b^2 - 15 * b - 5 = 0) :
  a + b = 15 :=
sorry

-- Problem 3
theorem minimum_c (a b c : ℝ) (h : a + b + c = 0) (h2 : a * b * c = 16) :
  c ≥ 4 :=
sorry

end reciprocal_roots_sum_of_roots_minimum_c_l434_434524


namespace necessary_but_not_sufficient_condition_of_p_for_q_l434_434867

def is_necessary_condition (p q : Prop) : Prop := q → p
def is_not_sufficient_condition (p q : Prop) : Prop := ¬ (p → q)

variable (m : ℝ)

def p : Prop := -1 < m ∧ m < 5
def q : Prop := ∀ x, (x^2 - 2 * m * x + m^2 - 1 = 0) → (-2 < x ∧ x < 4)

theorem necessary_but_not_sufficient_condition_of_p_for_q :
  is_necessary_condition p q ∧ is_not_sufficient_condition p q :=
sorry

end necessary_but_not_sufficient_condition_of_p_for_q_l434_434867


namespace difference_of_squares_example_l434_434796

theorem difference_of_squares_example : 503^2 - 497^2 = 6000 :=
by
  let a := 503
  let b := 497
  have h1 : a + b = 1000 := by norm_num
  have h2 : a - b = 6 := by norm_num
  have h3 : a^2 - b^2 = (a + b) * (a - b) := by rw [pow_two, pow_two, sub_mul, add_sub, sub_add_eq_add_sub]
  have h4 : (a + b) * (a - b) = 1000 * 6 := by rw [h1, h2]
  have h5 : 1000 * 6 = 6000 := by norm_num
  rw [h3, h4, h5]
  exact h5

end difference_of_squares_example_l434_434796


namespace right_triangle_area_l434_434558

variable (A B C M P Q : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (dist : B → C → ℝ) 

theorem right_triangle_area (right_triangle : A × B × C)
  (M : A) (angle_bisector : B)
  (BC_dist : dist M (BC : B) = 4)
  (AM_len : dist A M = 5) :
  ∃ S : ℝ, (S = (49 / 2)) ∧ (S = (1 / 2) * 7 * 7) :=
sorry

end right_triangle_area_l434_434558


namespace sum_of_coordinates_l434_434633

theorem sum_of_coordinates (x y : ℚ) (h₁ : y = 5) (h₂ : (y - 0) / (x - 0) = 3/4) : x + y = 35/3 :=
by sorry

end sum_of_coordinates_l434_434633


namespace avg_head_long_gt_1988_l434_434008

open Real

def is_long (a : ℕ → ℝ) (k l : ℕ) : Prop :=
  (1 ≤ l) ∧ (∑ i in range l, a (k + i)) / l > 1988

def is_head (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∃ l, is_long a k l

theorem avg_head_long_gt_1988 (a : ℕ → ℝ) (n : ℕ) (h : ∃ k l, k < n ∧ is_long a k l) :
  (∑ k in range n, if is_head a k then a k else 0) /
  (∑ k in range n, if is_head a k then 1 else 0) > 1988 :=
sorry

end avg_head_long_gt_1988_l434_434008


namespace min_a2_plus_b2_l434_434939

theorem min_a2_plus_b2 (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 :=
sorry

end min_a2_plus_b2_l434_434939


namespace dan_took_pencils_l434_434206

theorem dan_took_pencils (initial_pencils remaining_pencils : ℕ) (h_initial : initial_pencils = 34) (h_remaining : remaining_pencils = 12) : (initial_pencils - remaining_pencils) = 22 := 
by
  sorry

end dan_took_pencils_l434_434206


namespace part1_part2_l434_434459

noncomputable def f (x : ℝ) (λ ω : ℝ) : ℝ :=
  λ * Real.sin (ω * x) + Real.cos (ω * x)

noncomputable def g (x : ℝ) (λ : ℝ) : ℝ :=
  f x λ 2 + Real.cos (2 * x - π / 3)

theorem part1 (λ : ℝ) :
  (∀ x : ℝ, (f x λ 2 = f (x - π / 3) λ 2)) →
  f (π / 6) λ 2 = sqrt (λ ^ 2 + 1) →
  f (π / 6) λ 2 = sqrt (λ ^ 2 + 1) →
  λ = sqrt 3 :=
by
  sorry

theorem part2 :
  (λ = sqrt 3) →
  (∀ x ∈ Icc (-π / 2) (π / 12), g x (sqrt 3) ∈ Icc (- 3) (3 * sqrt 3 / 2)) :=
by
  sorry

end part1_part2_l434_434459


namespace area_OPTQ_l434_434866

-- Define the ellipse parameters
def ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 2 = 1

-- Definitions from the problem
def focus : ℝ × ℝ := (-2, 0)
def eccentricity : ℝ := sqrt(6) / 3
def a := sqrt(6)
def b := sqrt(2)

-- The point T on the line x = -3
variable {m : ℝ}
def T_point : ℝ × ℝ := (-3, m)

-- Points P and Q defined by a perpendicular from F intersecting the ellipse
variable {x1 y1 x2 y2 : ℝ}
def P : ℝ × ℝ := (x1, y1)
def Q : ℝ × ℝ := (x2, y2)

-- Statement that PQ is a perpendicular from F
def perpendicular_FPQ : Prop := (y1 - 0) / (x1 + 2) = -(x2 + 2) / (y2 - 0)

-- Quadrilateral OPTQ is a parallelogram
def parallelogram_OPTQ : Prop := 
  x1 + x2 = -3 ∧ 
  y1 + y2 = m ∧ 
  -3 - 0 = x2 - x1

-- Prove that the area of the quadrilateral OPTQ is 2√3
theorem area_OPTQ : 
  ellipse x1 y1 → ellipse x2 y2 → perpendicular_FPQ → parallelogram_OPTQ →
  2 * (sqrt ((4 * m / (m^2 + 3))^2 - 4 * (-2 / (m^2 + 3)))) = 2 * sqrt 3 :=
by
  sorry

end area_OPTQ_l434_434866


namespace distance_point_B_eq_2sqrt2_l434_434709

noncomputable def distance_point_B (side : ℝ) : ℝ :=
  let diagonal := side * real.sqrt 2
  in diagonal

theorem distance_point_B_eq_2sqrt2 :
  ∀ (side : ℝ), side = 2 → distance_point_B side = 2 * real.sqrt 2 :=
by
  intros side h_side
  simp [distance_point_B, h_side]
  sorry

end distance_point_B_eq_2sqrt2_l434_434709


namespace min_convex_cover_area_l434_434777

-- Define the dimensions of the box and the hole
def box_side := 5
def hole_side := 1

-- Define a function to represent the minimum area convex cover
def min_area_convex_cover (box_side hole_side : ℕ) : ℕ :=
  5 -- As given in the problem, the minimum area is concluded to be 5.

-- Theorem to state that the minimum area of the convex cover is 5
theorem min_convex_cover_area : min_area_convex_cover box_side hole_side = 5 :=
by
  -- Proof of the theorem
  sorry

end min_convex_cover_area_l434_434777


namespace triangle_area_l434_434497

-- Define the curve and points
noncomputable def curve (x : ℝ) : ℝ := (x-4)^2 * (x+3)

-- Define the x-intercepts
def x_intercept1 := 4
def x_intercept2 := -3

-- Define the y-intercept
def y_intercept := curve 0

-- Define the base and height of the triangle
def base : ℝ := x_intercept1 - x_intercept2
def height : ℝ := y_intercept

-- Statement of the problem: calculating the area of the triangle
theorem triangle_area : (1/2) * base * height = 168 := by
  sorry

end triangle_area_l434_434497


namespace percentage_of_students_who_speak_lies_l434_434090

theorem percentage_of_students_who_speak_lies
  (T : ℝ)    -- percentage of students who speak the truth
  (I : ℝ)    -- percentage of students who speak both truth and lies
  (U : ℝ)    -- probability of a randomly selected student speaking the truth or lies
  (H_T : T = 0.3)
  (H_I : I = 0.1)
  (H_U : U = 0.4) :
  ∃ (L : ℝ), L = 0.2 :=
by
  sorry

end percentage_of_students_who_speak_lies_l434_434090


namespace find_p_plus_q_of_n_squared_l434_434138

theorem find_p_plus_q_of_n_squared :
  let w1 := { center := (3 : ℝ, -4 : ℝ), radius := 6 }
  let w2 := { center := (-3 : ℝ, -4 : ℝ), radius := 2 }
  n = 8 / 5
  n^2 = 64 / 25
  p = 64
  q = 25
  p.gcd q = 1
  n > 0
  ∃ (n : ℝ) (p q : ℕ), (n^2 = p / q) ∧ (p.gcd q = 1) ∧ (n > 0) → p + q = 89 := 
begin
  -- Definitions for circles
  let w1_center := (3 : ℝ, -4 : ℝ),
  let w1_radius := 6,
  let w2_center := (-3 : ℝ, -4 : ℝ),
  let w2_radius := 2,
  -- Given values
  let n_val := 8 / 5,
  let n_val_sq := 64 / 25,
  let p_val := 64,
  let q_val := 25,
  have gcd_pq : nat.gcd p_val q_val = 1 := by sorry,
  have n_positive : n_val > 0 := by sorry,
  -- Conclusion
  existsi [n_val, p_val, q_val],
  exact ⟨by rfl, gcd_pq, n_positive⟩,
end

end find_p_plus_q_of_n_squared_l434_434138


namespace circle_diameter_l434_434258

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434258


namespace cone_surface_area_l434_434688

theorem cone_surface_area (r l: ℝ) (θ : ℝ) (h₁ : r = 3) (h₂ : θ = 2 * π / 3) (h₃: 2 * π * r = θ * l) :
  π * r * l + π * r ^ 2 = 36 * π :=
by
  sorry

end cone_surface_area_l434_434688


namespace part1_part2_part3_l434_434912

open Real Function

def f (x : ℝ) : ℝ := sin x ^ 2 - cos x ^ 2 - 2 * sqrt 3 * sin x * cos x

theorem part1 : f (2 * π / 3) = 2 :=
by
  sorry

theorem part2 : ∀ x : ℝ, f (x + π) = f x :=
by
  sorry

theorem part3 : ∀ k : ℤ, ∀ x : ℝ, k * π + (π / 6) ≤ x ∧ x ≤ k * π + (2 * π / 3) → 
  MonotoneOn f (Icc (k * π + (π / 6)) (k * π + (2 * π / 3))) :=
by
  sorry

end part1_part2_part3_l434_434912


namespace angle_PQR_parallel_PT_QR_l434_434970

theorem angle_PQR_parallel_PT_QR
  (PT QR : Prop)
  (h₁ : PT ∥ QR)
  (angle_TPQ : ℝ)
  (h₂ : 2 * angle_TPQ = 128) :
  ∠PQR = 116 :=
by
  -- Since PT and QR are parallel lines and we know 2 * ∠TPQ = 128
  rw [h₂]
  -- solving for ∠TPQ
  have h₃ : angle_TPQ = 64, by linarith
  -- Since ∠TPQ and ∠PQR are supplementary, ∠PQR + ∠TPQ = 180
  have h₄ : ∠PQR + angle_TPQ = 180, by sorry -- this step would be justified based on the properties of parallel lines and angles
  -- substituting ∠TPQ = 64 into ∠PQR + ∠TPQ = 180
  rw [h₃] at h₄
  linarith

end angle_PQR_parallel_PT_QR_l434_434970


namespace parallel_condition_l434_434577

theorem parallel_condition (a : ℝ) :
  (a^2 = 1 → a = 1) ∧ (a = 1 → a^2 = 1) ∧ (a = -1 → a^2 ≠ 1) → false ∧
  (a = 1 → y = a^2 * x + 1 ∧ y = a * x - 1) :=
begin
  sorry
end

end parallel_condition_l434_434577


namespace inequality_solution_range_l434_434943

theorem inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → x ^ 2 + a * x + 4 < 0) ↔ a < -4 :=
by 
  sorry

end inequality_solution_range_l434_434943


namespace set_roster_method_l434_434649

open Set

theorem set_roster_method :
  { m : ℤ | ∃ n : ℕ, 12 = n * (m + 1) } = {0, 1, 2, 3, 5, 11} :=
  sorry

end set_roster_method_l434_434649


namespace circle_diameter_l434_434264

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l434_434264


namespace simplify_expression_l434_434171

-- Define the given problem and conditions
def e : ℝ := Real.exp 1 -- Euler's number
def ln (x : ℝ) : ℝ := Real.log x -- Natural logarithm function
def lg (x : ℝ) : ℝ := Real.log10 x -- Base-10 logarithm function

-- Define the expressions involved
def expr1 := Real.exp (ln 2)
def expr2 := (lg 2) ^ 2
def expr3 := (lg 2) * (lg 5)
def expr4 := lg 5

-- The statement of the theorem
theorem simplify_expression : expr1 + expr2 + expr3 + expr4 = 3 + expr2 + expr4 := 
by
  -- Proof to be filled in
  sorry

end simplify_expression_l434_434171


namespace opposite_of_neg23_eq_23_reciprocal_of_neg23_eq_neg_1_div_23_abs_value_of_neg23_eq_23_l434_434187

theorem opposite_of_neg23_eq_23 : -(-23) = 23 := 
by sorry

theorem reciprocal_of_neg23_eq_neg_1_div_23 : (1 : ℚ) / (-23) = -(1 / 23 : ℚ) :=
by sorry

theorem abs_value_of_neg23_eq_23 : abs (-23) = 23 :=
by sorry

end opposite_of_neg23_eq_23_reciprocal_of_neg23_eq_neg_1_div_23_abs_value_of_neg23_eq_23_l434_434187


namespace bugs_ate_each_l434_434608

theorem bugs_ate_each : 
  ∀ (total_bugs total_flowers each_bug_flowers : ℕ), 
    total_bugs = 3 ∧ total_flowers = 6 ∧ each_bug_flowers = total_flowers / total_bugs -> each_bug_flowers = 2 := by
  sorry

end bugs_ate_each_l434_434608


namespace range_of_a_l434_434883

theorem range_of_a (f : ℝ → ℝ) 
  (h_decreasing : ∀ x y, x ≤ y → f(x) ≥ f(y)) 
  (h_addition : ∀ x y : ℝ, f(x) + f(y) = f(x + y) + 2) :
  {a : ℝ | ∃! (x_integers : ℤ × ℤ), f((x_integers.1 : ℝ)^2 - a * (x_integers.1 : ℝ)) + f((x_integers.1 : ℝ) - a) > 4} = 
  {a : ℝ | (-4 ≤ a ∧ a < -3) ∨ (1 < a ∧ a ≤ 2)} :=
by
  sorry

end range_of_a_l434_434883


namespace proposition_does_not_hold_for_4_l434_434861

variable (P : ℕ → Prop)
variable (h1 : ∀ k ∈ ℕ, 0 < k → P k → P (k + 1))
variable (h2 : ¬ P 5)

theorem proposition_does_not_hold_for_4 : ¬ P 4 := 
sorry

end proposition_does_not_hold_for_4_l434_434861


namespace kamal_weighted_average_67_755_l434_434088

noncomputable def kamal_weighted_average :
  (score_English : ℕ) (total_English : ℕ) (weight_English : ℕ)
  (score_Mathematics : ℕ) (total_Mathematics : ℕ) (weight_Mathematics : ℕ)
  (score_Physics : ℕ) (total_Physics : ℕ) (weight_Physics : ℕ)
  (score_Chemistry : ℕ) (total_Chemistry : ℕ) (weight_Chemistry : ℕ)
  (score_Biology : ℕ) (total_Biology : ℕ) (weight_Biology : ℕ)
  (score_History : ℕ) (total_History : ℕ) (weight_History : ℕ)
  (score_Geography : ℕ) (total_Geography : ℕ) (weight_Geography : ℕ)
  → (weighted_average : ℚ) :=
λ score_English total_English weight_English
  score_Mathematics total_Mathematics weight_Mathematics
  score_Physics total_Physics weight_Physics
  score_Chemistry total_Chemistry weight_Chemistry
  score_Biology total_Biology weight_Biology
  score_History total_History weight_History
  score_Geography total_Geography weight_Geography
  →
  (score_English.toRat / total_English * 100 * weight_English +
  score_Mathematics.toRat / total_Mathematics * 100 * weight_Mathematics +
  score_Physics.toRat / total_Physics * 100 * weight_Physics +
  score_Chemistry.toRat / total_Chemistry * 100 * weight_Chemistry +
  score_Biology.toRat / total_Biology * 100 * weight_Biology +
  score_History.toRat / total_History * 100 * weight_History +
  score_Geography.toRat / total_Geography * 100 * weight_Geography) /
  (weight_English + weight_Mathematics + weight_Physics + weight_Chemistry +
   weight_Biology + weight_History + weight_Geography) 

theorem kamal_weighted_average_67_755 :
  kamal_weighted_average 
    76 120 2
    65 150 3
    82 100 2
    67 80 1
    85 100 2
    92 150 1
    58 75 1 = 67.755 :=
by sorry

end kamal_weighted_average_67_755_l434_434088


namespace person_catch_up_time_l434_434660

theorem person_catch_up_time :
  let track_length := 400 
  let speed_A := 52 
  let speed_B := 46 
  let rest_distance := 100 
  let rest_time := 1
  let catch_up_time := 147 + (1 : ℝ) / 3
  in catch_up_time = (147 : ℝ) + (1 : ℝ)/ 3 :=
by
  sorry

end person_catch_up_time_l434_434660


namespace highest_score_l434_434333

-- Let a, b, c, d be real numbers representing the scores of students A, B, C, and D respectively.
variables {a b c d : ℝ}

-- Define the conditions given in the problem
def conditions : Prop :=
  (a + b = c + d) ∧ (b + c > a + d) ∧ (a > b + d)

-- State the theorem to be proved: "C has the highest score"
theorem highest_score (h : conditions) : c > a ∧ c > b ∧ c > d :=
sorry

end highest_score_l434_434333


namespace find_b_l434_434518

noncomputable def n : ℝ := 2 ^ 0.15
def b := 5 / 0.15 

theorem find_b (h1 : n = 2 ^ 0.15) (h2 : n ^ b = 32) : b ≈ 33.333 :=
by
  sorry

end find_b_l434_434518


namespace smallest_positive_a_integer_root_l434_434814

theorem smallest_positive_a_integer_root :
  ∀ x a : ℚ, (exists x : ℚ, (x > 0) ∧ (a > 0) ∧ 
    (
      ((x - a) / 2 + (x - 2 * a) / 3) / ((x + 4 * a) / 5 - (x + 3 * a) / 4) =
      ((x - 3 * a) / 4 + (x - 4 * a) / 5) / ((x + 2 * a) / 3 - (x + a) / 2)
    )
  ) → a = 419 / 421 :=
by sorry

end smallest_positive_a_integer_root_l434_434814


namespace not_p_and_q_is_false_l434_434873

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def p : Prop := is_even_function (λ x, Real.log ((1 - x) * (1 + x)))

def q : Prop := is_even_function (λ x, (Real.exp x - 1) / (Real.exp x + 1))

theorem not_p_and_q_is_false : ¬ p ∧ q → False := by
  sorry

end not_p_and_q_is_false_l434_434873


namespace circle_diameter_l434_434293

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l434_434293


namespace ellipse_conditions_l434_434431

theorem ellipse_conditions (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : 
  (∀ (x y : ℝ), ax^2 + by^2 = 1 → ((a ≠ b) → ellipse)) ∧
  ¬ (∀ (x y : ℝ), ax^2 + by^2 = 1 → ellipse) :=
sorry

end ellipse_conditions_l434_434431


namespace projections_are_concyclic_l434_434572

variables {A B C D A' B' C' D' : Point}

-- Define the concept of concyclic points
def concyclic (P Q R S : Point) : Prop := 
  ∃ (circle : Circle), P ∈ circle ∧ Q ∈ circle ∧ R ∈ circle ∧ S ∈ circle  

-- Define the orthogonal projection condition. Proj(P, line(l1, l2)) represents the orthogonal projection of P onto line l1l2.
def orthogonal_projection (P P' line_start line_end : Point) : Prop := 
  ∃ (foot : Point), foot = P' ∧ foot.is_perpendicular_to (line_start, line_end)

-- Inputs
variables (concyclic_ABCD : concyclic A B C D)
          (orth_proj_A_on_BD : orthogonal_projection A A' B D)
          (orth_proj_C_on_BD : orthogonal_projection C C' B D)
          (orth_proj_B_on_AC : orthogonal_projection B B' A C)
          (orth_proj_D_on_AC : orthogonal_projection D D' A C)

-- Output to show
theorem projections_are_concyclic : concyclic A' B' C' D' :=
sorry  -- Proof goes here

end projections_are_concyclic_l434_434572


namespace find_x0_l434_434858

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem find_x0 (x0 : ℝ) (h : deriv f x0 = 0) : x0 = Real.exp 1 :=
by 
  sorry

end find_x0_l434_434858


namespace S9_div_S6_eq_7div3_l434_434865

noncomputable def S : ℕ → ℚ := sorry  -- Define the sequence sum S to be of type natural numbers to rational numbers.

axiom S_n_property (n : ℕ) : S (2 * n) - S n : S n = 3

theorem S9_div_S6_eq_7div3 : S 9 / S 6 = 7 / 3 := by
  sorry

end S9_div_S6_eq_7div3_l434_434865


namespace BE_eq_CF_l434_434059

theorem BE_eq_CF (A B C P E F : Point)
  (hP : is_angle_bisector A B C P)
  (hCE_parallel_PB : is_parallel (line_through C E) (line_through P B))
  (hBF_parallel_PC : is_parallel (line_through B F) (line_through P C))
  (hE_on_ext_AB : online (line_through A B) E)
  (hF_on_ext_AC : online (line_through A C) F) : 
  length (segment B E) = length (segment C F) :=
sorry

end BE_eq_CF_l434_434059


namespace smallest_integer_remainder_l434_434726

theorem smallest_integer_remainder (b : ℕ) :
  (b ≡ 2 [MOD 3]) ∧ (b ≡ 3 [MOD 5]) → b = 8 := 
by
  sorry

end smallest_integer_remainder_l434_434726


namespace find_number_with_21_multiples_of_4_l434_434205

theorem find_number_with_21_multiples_of_4 (n : ℕ) (h₁ : ∀ k : ℕ, n + k * 4 ≤ 92 → k < 21) : n = 80 :=
sorry

end find_number_with_21_multiples_of_4_l434_434205


namespace compare_values_l434_434006

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := 3 ^ 0.2
noncomputable def c : ℝ := 0.3 ^ 2

theorem compare_values : a < c ∧ c < b :=
by
  -- This is where the proof would go.
  sorry

end compare_values_l434_434006


namespace triangle_inscribed_in_semicircle_l434_434510

theorem triangle_inscribed_in_semicircle (A B C : Point) (d : ℝ) (h : diameter_circle AB d) :
  ∠ACB = 90 → AC + BC ≤ AB * sqrt 2 := by
  sorry

end triangle_inscribed_in_semicircle_l434_434510


namespace find_n_l434_434670

theorem find_n : ∃ n : ℤ, 50 ≤ n ∧ n ≤ 120 ∧ n % 8 = 0 ∧ n % 9 = 5 ∧ n % 7 = 3 ∧ n = 104 :=
by {
  let n := 104,
  existsi n,
  repeat {
    split,
    any_goals { exact dec_trivial }
  },
  sorry
}

end find_n_l434_434670


namespace false_exterior_angle_l434_434227

theorem false_exterior_angle (triangle : Type) [triangle.exists_interior_angles]
  (ext_angle : ∀ (a b c : ℝ), ((triangle.interior a b c) → 
                              triangle.exterior (a + b = c)) → 
                              (a + b > c) → false  :=
begin
  sorry
end

end false_exterior_angle_l434_434227


namespace min_abs_sum_l434_434874

theorem min_abs_sum (a b c : ℝ) (h₁ : a + b + c = -2) (h₂ : a * b * c = -4) :
  ∃ (m : ℝ), m = min (abs a + abs b + abs c) 6 :=
sorry

end min_abs_sum_l434_434874


namespace bolzano_first_bolzano_second_l434_434750

open Set Filter TopologicalSpace

-- Definitions for the first problem
def seq (x : ℕ → ℝ) := ∀ n : ℕ, 0 ≤ x n ∧ x n < 1

def holds_infinitely_often (P : ℕ → Prop) := ∀ᶠ n in at_top, P n

-- Statement for the first problem
theorem bolzano_first (x : ℕ → ℝ) (h_seq : seq x) :
  holds_infinitely_often (λ n, x n ∈ Ico 0 (1/2)) ∨
  holds_infinitely_often (λ n, x n ∈ Ico (1/2) 1) :=
sorry

-- Definitions for the second problem
def in_interval (x : ℕ → ℝ) (ε : ℝ) := 
  ∃ α ∈ Icc 0 1, holds_infinitely_often (λ n, |x n - α| < ε)

-- Statement for the second problem
theorem bolzano_second (x : ℕ → ℝ) (h_seq : seq x) (ε : ℝ) (h_ε : 0 < ε ∧ ε < 1/2) :
  in_interval x ε :=
sorry

end bolzano_first_bolzano_second_l434_434750


namespace circle_with_radius_1_contains_at_least_11_points_in_large_circle_l434_434298

def point := (ℝ × ℝ)

def inside_circle (center : point) (radius : ℝ) (p : point) : Prop :=
  let (x, y) := p
  let (a, b) := center
  (x - a) * (x - a) + (y - b) * (y - b) ≤ radius * radius

def count_points_inside_circle (center : point) (radius : ℝ) (points : list point): ℕ :=
  points.count (inside_circle center radius)

theorem circle_with_radius_1_contains_at_least_11_points_in_large_circle 
  (points: list point)
  (h_points_card: points.card = 251)
  (R: ℝ)
  (hR: R = 4)
  (origin: point)
  (h_origin: origin = (0, 0)) :
  ∃ (center: point), count_points_inside_circle center 1 points ≥ 11 :=
sorry

end circle_with_radius_1_contains_at_least_11_points_in_large_circle_l434_434298


namespace inverse_proposition_false_l434_434613

-- Define a space and basic geometric concepts
def Point := ℝ × ℝ × ℝ

-- A predicate to determine if four points are coplanar
def coplanar (a b c d : Point) : Prop :=
∃ (u v : ℝ), d = (u • (b - a) + v • (c - a) + a)

-- A predicate to determine if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop :=
∃ (u : ℝ), p3 = (u • (p2 - p1) + p1)

-- Original proposition's condition: Four points are not coplanar
def not_coplanar (a b c d : Point) : Prop := 
  ¬ coplanar a b c d

-- Proving the inverse proposition's falsehood
theorem inverse_proposition_false (a b c d : Point) : 
  (¬ coplanar a b c d → (¬ collinear a b c ∧ ¬ collinear a b d ∧ ¬ collinear a c d ∧ ¬ collinear b c d)) → 
  (¬ (¬ collinear a b c ∨ ¬ collinear a b d ∨ ¬ collinear a c d ∨ ¬ collinear b c d → ¬ coplanar a b c d)) :=
by
  sorry

end inverse_proposition_false_l434_434613


namespace fraction_computation_l434_434377

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l434_434377


namespace optionA_is_quadratic_l434_434740

-- Definitions of the conditions for each option
def optionA : Prop := ∃ x : ℝ, x^2 = 0
def optionB : Prop := ∃ (x y : ℝ), x^2 - 2 * y + 1 = 0
def optionC (a b c : ℝ) : Prop := ∃ x : ℝ, ax^2 + bx + c = 0
def optionD : Prop := ∃ x : ℝ, (1 / x) - 5 * x^2 + 6 = 0

-- Proof statement that option A is definitely a quadratic equation
theorem optionA_is_quadratic : optionA := by
  sorry

end optionA_is_quadratic_l434_434740


namespace possible_to_divide_divisors_into_equal_groups_l434_434117

theorem possible_to_divide_divisors_into_equal_groups :
  ∃ (A B : Finset ℕ), (A ∪ B = (Nat.divisors (100.factorial : ℤ)).to_finset ∧ 
  A ∩ B = ∅ ∧ 
  A.card = B.card ∧ 
  (A.prod id) = (B.prod id)) :=
sorry

end possible_to_divide_divisors_into_equal_groups_l434_434117


namespace parallelogram_count_392_l434_434154

theorem parallelogram_count_392 (area : ℕ) (A B D C : ℕ × ℕ) (m n : ℤ) 
  (h_area : area = 500000) 
  (hA : A = (0, 0))
  (hB : ∃ b : ℕ, B = (b, m * b))
  (hD : ∃ d : ℕ, D = (d, n * d))
  (h_distinct : m ≠ n ∧ m > 1 ∧ n > 1) :
  ∑ (k : ℕ) in (divisors 500000), 
  (∑ (i j : ℕ) in fintype.of_finset (finset.filter (λ p, p.1 * p.2 = 500000 / k) _), 1) = 392 := 
sorry

end parallelogram_count_392_l434_434154


namespace value_at_2pi_over_3_minimum_positive_period_interval_of_monotonic_increase_l434_434911

-- Define the function f
def f (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

-- Prove that f(2π/3) = 2
theorem value_at_2pi_over_3 : f (2 * Real.pi / 3) = 2 := by sorry

-- Prove that the minimum positive period of f is π
theorem minimum_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := by sorry

-- Prove the interval of monotonic increase
theorem interval_of_monotonic_increase : 
  ∀ k : ℤ, ∀ x, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3 → 
  ∀ y, x < y ∧ y < k * Real.pi + Real.pi / 6 ∨ k * Real.pi + 2 * Real.pi / 3 < y → f x < f y := by sorry

end value_at_2pi_over_3_minimum_positive_period_interval_of_monotonic_increase_l434_434911


namespace circle_diameter_l434_434256

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434256


namespace avgPercentageSpentOnFoodCorrect_l434_434792

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

end avgPercentageSpentOnFoodCorrect_l434_434792


namespace exists_composite_sequence_l434_434651

theorem exists_composite_sequence (k : ℕ) : 
  ∃ n : ℤ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → Nat.composite (n + i) := 
sorry

end exists_composite_sequence_l434_434651


namespace percentage_difference_l434_434535

variable {x y z : ℝ}

def y_condition := y = 1.80 * x
def z_condition := z = 1.50 * y

theorem percentage_difference (hx : y_condition) (hy : z_condition) :
  (2.70 * x - x) / 2.70 * x * 100 = 62.96 := by
sory

end percentage_difference_l434_434535


namespace continuous_function_nondecreasing_l434_434568

open Set

variable {α : Type*} [LinearOrder ℝ] [Preorder ℝ]

theorem continuous_function_nondecreasing
  (f : (ℝ)→ ℝ) 
  (h_cont : ContinuousOn f (Ioi 0))
  (h_seq : ∀ x > 0, Monotone (fun n : ℕ => f (n*x))):
  ∀ x y, x ≤ y → f x ≤ f y := 
sorry

end continuous_function_nondecreasing_l434_434568


namespace maximize_S_n_at_24_l434_434102

noncomputable def a_n (n : ℕ) : ℝ := 142 + (n - 1) * (-2)
noncomputable def b_n (n : ℕ) : ℝ := 142 + (n - 1) * (-6)
noncomputable def S_n (n : ℕ) : ℝ := (n / 2.0) * (2 * 142 + (n - 1) * (-6))

theorem maximize_S_n_at_24 : ∀ (n : ℕ), S_n n ≤ S_n 24 :=
by sorry

end maximize_S_n_at_24_l434_434102


namespace cos_of_angle_C_l434_434537

theorem cos_of_angle_C (A B C : ℝ)
  (h1 : Real.sin (π - A) = 3 / 5)
  (h2 : Real.tan (π + B) = 12 / 5)
  (h_cos_A : Real.cos A = 4 / 5) :
  Real.cos C = 16 / 65 :=
sorry

end cos_of_angle_C_l434_434537


namespace more_stable_yield_A_l434_434756

theorem more_stable_yield_A (s_A s_B : ℝ) (hA : s_A * s_A = 794) (hB : s_B * s_B = 958) : s_A < s_B :=
by {
  sorry -- Details of the proof would go here
}

end more_stable_yield_A_l434_434756


namespace intersecting_functions_bound_a_l434_434907

theorem intersecting_functions_bound_a 
  (a : ℝ) (x₀ : ℝ) (y₀ : ℝ)
  (h₀ : a > 0) (h₁ : a ≠ 1)
  (h₂ : x₀ ≥ 2)
  (h₃ : y₀ = (1 / 2) ^ x₀)
  (h₄ : y₀ = log a x₀) : 
  a ≥ 16 :=
sorry

end intersecting_functions_bound_a_l434_434907


namespace six_trillion_scientific_notation_l434_434151

theorem six_trillion_scientific_notation : 
  (∃ n : ℕ, 6000000000000 = 6 * 10^n) → ∃ n : ℕ, n = 12 :=
by
  intro h
  cases h with n hn
  use 12
  sorry

end six_trillion_scientific_notation_l434_434151


namespace quadratic_min_attains_at_a_l434_434027

theorem quadratic_min_attains_at_a
  (r s : ℝ)
  (γ δ : ℝ)
  (hf : ∀ x : ℝ, f(x) = x^2 + r * x + s)
  (hg : ∀ x : ℝ, g(x) = x^2 - 9 * x + 6)
  (hγδ_sum : γ + δ = 9)
  (hγδ_prod : γ * δ = 6)
  (roots_sum_f : r = -(γ * δ))
  (roots_prod_f : s = γ + δ)
  (a : ℝ) :
  f(a) = a ↔ a = 3 :=
by
  sorry

end quadratic_min_attains_at_a_l434_434027


namespace number_of_mappings_l434_434921

theorem number_of_mappings (P Q : Type) (x y z : P) (f : P → Q)
  (P_def : P = {x, y, z}) (Q_def : Q = {1, 2, 3}) (hy : f y = 2) :
  (∃ g : P → Q, g y = 2 ∧ (g x ∈ Q ∧ g z ∈ Q)) ∧ (∀ h : P → Q, h y = 2 → (h x ∈ Q ∧ h z ∈ Q) → (P → Q).count = 9) :=
sorry

end number_of_mappings_l434_434921


namespace vector_calculation_l434_434925

theorem vector_calculation :
  let a := ⟨3, 1⟩
  let b := ⟨-2, 5⟩
  2 • a + b = ⟨4, 7⟩ :=
by
  intros
  have ha : 2 • a = ⟨6, 2⟩ := sorry
  have hb : 2 • a + b = ⟨6, 2⟩ + ⟨-2, 5⟩ := sorry
  show 2 • a + b = ⟨4, 7⟩ from sorry

end vector_calculation_l434_434925


namespace segment_length_EF_l434_434550

theorem segment_length_EF (AB BC : ℝ) (hAB : AB = 4) (hBC : BC = 12) :
  let EF := sqrt ((4 / 3) ^ 2 + 4 ^ 2) in
  EF = (4 * sqrt 10) / 3 :=
by
  -- The proof is left as an exercise.
  sorry

end segment_length_EF_l434_434550


namespace theater_loss_l434_434314

/-- 
A movie theater has a total capacity of 50 people and charges $8 per ticket.
On a Tuesday night, they only sold 24 tickets. 
Prove that the revenue lost by not selling out is $208.
-/
theorem theater_loss 
  (capacity : ℕ) 
  (price : ℕ) 
  (sold_tickets : ℕ) 
  (h_cap : capacity = 50) 
  (h_price : price = 8) 
  (h_sold : sold_tickets = 24) : 
  capacity * price - sold_tickets * price = 208 :=
by
  sorry

end theater_loss_l434_434314


namespace theta_values_l434_434892

theorem theta_values (θ : ℝ) (h1 : dist (cos θ, sin θ) (λ (x y : ℝ), x * sin θ + y * cos θ - 1) = 1/2)
  (h2 : 0 ≤ θ ∧ θ ≤ π/2) : θ = π/12 ∨ θ = 5 * π / 12 := by
  sorry

end theta_values_l434_434892


namespace sum_coordinates_eq_l434_434627

theorem sum_coordinates_eq : ∀ (A B : ℝ × ℝ), A = (0, 0) → (∃ x : ℝ, B = (x, 5)) → 
  (∀ x : ℝ, B = (x, 5) → (5 - 0) / (x - 0) = 3 / 4) → 
  let x := 20 / 3 in
  (x + 5 = 35 / 3) :=
by
  intros A B hA hB hslope
  have hx : ∃ x, B = (x, 5) := hB
  cases hx with x hx_def
  rw hx_def at hslope
  have : x = 20 / 3 := by
    rw hx_def at hslope
    field_simp at hslope
    linarith
  rw [this, hx_def]
  norm_num

end sum_coordinates_eq_l434_434627


namespace find_N_l434_434835

theorem find_N (N : ℕ) (hN : N > 1) (h1 : 2019 ≡ 1743 [MOD N]) (h2 : 3008 ≡ 2019 [MOD N]) : N = 23 :=
by
  sorry

end find_N_l434_434835


namespace sum_coords_B_l434_434626

def point (A B : ℝ × ℝ) : Prop :=
A = (0, 0) ∧ B.snd = 5 ∧ (B.snd - A.snd) / (B.fst - A.fst) = 3 / 4

theorem sum_coords_B (x : ℝ) :
  point (0, 0) (x, 5) → x + 5 = 35 / 3 :=
by
  intro h
  sorry

end sum_coords_B_l434_434626


namespace possible_values_of_f_l434_434581

def sequence (k : ℕ) : ℝ :=
  k * (-1) ^ k

noncomputable def f (n : ℕ) : ℝ :=
  (∑ k in Finset.range n, sequence (k + 1)) / n

theorem possible_values_of_f (n : ℕ) (hn : 0 < n) : 
  ∃ s ∈ ({1/2, -1/2} : Set ℝ), f n = s :=
sorry

end possible_values_of_f_l434_434581


namespace find_q_l434_434427

theorem find_q (d : ℝ) (q t : ℝ) (h1 : d ≠ 0) (h2 : 0 < q ∧ q < 1) 
  (h3 : a1 = d) 
  (h4 : b1 = d^2) 
  (h5 : (a1^2 + (a1+d)^2 + (a1+2*d)^2) / (b1 + b1*q + b1*q^2) = t) 
  (h6 : t ∈ ℤ) 
  (h7 : 0 < t) : q = 1/2 :=
by
  sorry

end find_q_l434_434427


namespace prism_cut_similarity_l434_434346

theorem prism_cut_similarity {a b c : ℝ} (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) :
  (∃ a' b' c' : ℝ, a' = a ∧ b' = b ∧ c' = c / 2 ∧ (a' / a) = 1 ∧ (b' / a) = real.sqrt 3^2 ∧ (c' / a) = real.sqrt 3^4) :=
  sorry

end prism_cut_similarity_l434_434346


namespace collinear_E_F_D_l434_434056

theorem collinear_E_F_D (A B C D P E F : Point) (h1 : collinear B C D)
  (h2 : distance C D = distance A C)
  (circumcircle_ACD : Circle)
  (circle_diameter_BC : Circle)
  (h3 : circumcircle_ACD ⊃ {A, C, D})
  (h4 : circle_diameter_BC ⊃ {B, C})
  (h5 : circumcircle_ACD ∩ circle_diameter_BC = {C, P})
  (h6 : line_through B P = line_through C E)
  (h7 : line_through C P = line_through B F)
  (h8 : line_through E A = line_through B F)
  : collinear E F D :=
sorry

end collinear_E_F_D_l434_434056


namespace sum_of_values_l434_434536

theorem sum_of_values (x : ℤ) (h : |x - 5| = 23) : ∃ x1 x2, x1 + x2 = 10 ∧ |x1 - 5| = 23 ∧ |x2 - 5| = 23 :=
by sorry

end sum_of_values_l434_434536


namespace triangle_ABI_ratio_l434_434962

theorem triangle_ABI_ratio:
  ∀ (AC BC : ℝ) (hAC : AC = 15) (hBC : BC = 20),
  let AB := Real.sqrt (AC^2 + BC^2) in
  let CD :=  (AC * BC) / AB in
  let r := CD / 2 in
  let x := Real.sqrt (r^2 + (Real.sqrt (r^2 + (AB/2)^2) - r)^2) in
  let P := 2 * x + AB in
  (P / AB = 177 / 100) ∧ (177 + 100 = 277) :=
by
  intros AC BC hAC hBC
  let AB := Real.sqrt (AC^2 + BC^2)
  let CD := (AC * BC) / AB
  let r := CD / 2
  let x := Real.sqrt (r^2 + (Real.sqrt (r^2 + (AB/2)^2) - r)^2)
  let P := 2 * x + AB
  have ratio : P / AB = 177 / 100 := sorry
  exact ⟨ratio, rfl⟩


end triangle_ABI_ratio_l434_434962


namespace player_B_has_winning_strategy_l434_434214

-- Define the initial condition: The initial pile of stones
def initial_pile : ℕ := 2003

-- Define a function to determine if a number is a divisor of another
def is_divisor (d n : ℕ) : Prop := d > 0 ∧ n % d = 0

-- Define the condition that removing the last stone means losing
def loses (stones : ℕ) (player: string) : Prop := stones = 0 ∧ player = "remover"

-- Define the condition that guarantees a winning strategy
def winning_strategy_for_B : Prop :=
  ∀ (n : ℕ), n > 0 → (∃ d, is_divisor d n ∧ 
  ((∃ m, n = 2 * m ∧ (∃ odd_d, is_divisor odd_d m ∧ m % 2 = 1)) ∨ 
  (d = n ∧ (∃ x, n = 2 * x + 1))))

-- The main statement: Player $B$ has a winning strategy
theorem player_B_has_winning_strategy (n: ℕ) (h_initial: n = initial_pile) : winning_strategy_for_B :=
by sorry

end player_B_has_winning_strategy_l434_434214


namespace function_property_l434_434937

theorem function_property (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f(x + 2012) = -f(x + 2011)) (h2 : f 2012 = -2012) : f (-1) = 2012 :=
sorry

end function_property_l434_434937


namespace opposite_of_neg_nine_is_nine_l434_434683

-- Define the predicate for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that needs to be proved
theorem opposite_of_neg_nine_is_nine : ∃ x, is_opposite (-9) x ∧ x = 9 := 
by {
  -- Use "sorry" to indicate the proof is not provided
  sorry
}

end opposite_of_neg_nine_is_nine_l434_434683


namespace trajectory_of_P_l434_434001

-- Define the points F1 and F2
def F1 := (-8, 3)
def F2 := (2, 3)

-- Define the Euclidean distance function
def dist (A B : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Define the condition that point P satisfies
def condition (P : (ℝ × ℝ)) : Prop :=
  abs (dist P F1 - dist P F2) = 10

-- Theorem stating the trajectory of point P
theorem trajectory_of_P (P : (ℝ × ℝ)) (h : condition P) : Prop :=
  P.1 = 5 -- Expressing that P follows a ray (simplified example; actually, this would need better definition to match "a ray")

end trajectory_of_P_l434_434001


namespace largest_digit_B_divisible_by_4_l434_434559

theorem largest_digit_B_divisible_by_4 :
  ∃ B : ℕ, B = 9 ∧ ∀ k : ℕ, (k ≤ 9 → (∃ n : ℕ, 4 * n = 10 * B + 792 % 100)) :=
by
  sorry

end largest_digit_B_divisible_by_4_l434_434559


namespace sum_of_coordinates_l434_434631

theorem sum_of_coordinates (x y : ℚ) (h₁ : y = 5) (h₂ : (y - 0) / (x - 0) = 3/4) : x + y = 35/3 :=
by sorry

end sum_of_coordinates_l434_434631


namespace collinear_vectors_mn_l434_434034

variables {α : Type*} [linear_ordered_field α]

/-- Given that vectors i and j are not collinear,
AB = i + m * j, AD = n * i + j with m ≠ 1.
If points A, B, and D are collinear, then mn = 1. -/
theorem collinear_vectors_mn (i j : α) (m n : α)
  (h1 : ¬ (∃ (k : α), i = k * j))
  (hAB : ∀ (A B : α), A = i + m * j)
  (hAD : ∀ (A D : α), A = n * i + j)
  (hm : m ≠ 1)
  (hcol : ∃ (k : α), i + m * j = k * (n * i + j)) :
  m * n = 1 :=
sorry

end collinear_vectors_mn_l434_434034


namespace count_positive_integers_l434_434408

theorem count_positive_integers : 
    ∃ (xs : Finset ℕ), (∀ x ∈ xs, x ≠ 0) ∧ (∀ x ∈ xs, x ≠ 9) ∧ (∀ x ∈ xs, log (3 : ℝ) ((x : ℝ ^ 2) / 3) / log (3 : ℝ)((x : ℝ) / 9) < 6 + log (3 : ℝ) (9 / (x : ℝ))) ∧ xs.card = 223 :=
by
  sorry

end count_positive_integers_l434_434408


namespace tan_ratio_of_triangle_l434_434863

theorem tan_ratio_of_triangle (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = (3 / 5) * c) : 
  Real.tan A / Real.tan B = 4 :=
sorry

end tan_ratio_of_triangle_l434_434863


namespace angle_B_max_area_l434_434085

variable (A B C a b c : ℝ) (S : ℝ)

-- Assume the condition in the problem
axiom tan_condition : tan A + tan C = sqrt(3) * (tan A * tan C - 1)
axiom angle_sum : A + B + C = Real.pi
axiom b_value : b = 2

-- Prove the angle B is π/3
theorem angle_B : B = Real.pi / 3 := by
  sorry

-- Prove the maximum area of the triangle ABC is sqrt(3)
theorem max_area : S = sqrt(3) :=
  have B_value : B = Real.pi / 3 := angle_B A B C a b c S tan_condition angle_sum
  -- Now we can state the area condition when b = 2
  by
    sorry

end angle_B_max_area_l434_434085


namespace largest_difference_is_A_l434_434574

-- Define the constants
def P : ℝ := 3 * 1003^1004
def Q : ℝ := 1003^1004
def R : ℝ := 1002 * 1003^1003
def S : ℝ := 3 * 1003^1003
def T : ℝ := 1003^1003
noncomputable def U : ℝ := 1003^1002 * (nat.factorial 1002)

-- Define the differences
def A : ℝ := P - Q
def B : ℝ := Q - R
def C : ℝ := R - S
def D : ℝ := S - T
noncomputable def E : ℝ := T - U

-- Problem statement
theorem largest_difference_is_A :
  max (max (max A B) (max C D)) E = A :=
by 
  sorry

end largest_difference_is_A_l434_434574


namespace annual_interest_rate_l434_434838

noncomputable def find_annual_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  let r := (A / P)^(1 / (n * t)) - 1
  r

theorem annual_interest_rate (P I A : ℝ) (n t : ℕ) (hP : P = 14800) (hI : I = 4265.73) (hA : A = P + I) (hn : n = 1) (ht : t = 2) :
  find_annual_interest_rate P A n t ≈ 0.13559 :=
by
  have h_total_amount : A = 19065.73 := by
    rw [hP, hI]; exact hA
  let r := find_annual_interest_rate P A n t
  have r_correct : r ≈ 0.13559 := by sorry
  exact r_correct

end annual_interest_rate_l434_434838


namespace point_B_coordinates_sum_l434_434619

theorem point_B_coordinates_sum (x : ℚ) (h1 : ∃ (B : ℚ × ℚ), B = (x, 5))
    (h2 : (5 - 0) / (x - 0) = 3/4) :
    x + 5 = 35/3 :=
by
  sorry

end point_B_coordinates_sum_l434_434619


namespace time_no_traffic_is_4_hours_l434_434125

-- Definitions and conditions
def distance : ℕ := 200
def time_traffic : ℕ := 5

axiom traffic_speed_relation : ∃ (speed_traffic : ℕ), distance = speed_traffic * time_traffic
axiom speed_difference : ∀ (speed_traffic speed_no_traffic : ℕ), speed_no_traffic = speed_traffic + 10

-- Prove that the time when there's no traffic is 4 hours
theorem time_no_traffic_is_4_hours : ∀ (speed_traffic speed_no_traffic : ℕ), 
  distance = speed_no_traffic * (distance / speed_no_traffic) -> (distance / speed_no_traffic) = 4 :=
by
  intros speed_traffic speed_no_traffic h
  sorry

end time_no_traffic_is_4_hours_l434_434125


namespace minimum_arg_z_l434_434039

open Complex Real

noncomputable def z_cond (z : ℂ) := abs (z + 3 - (complex.I * sqrt 3)) = sqrt 3

theorem minimum_arg_z : ∀ z : ℂ, z_cond z → arg z = 5 / 6 * π :=
by
  intros
  sorry

end minimum_arg_z_l434_434039


namespace bookcase_length_in_feet_l434_434607

theorem bookcase_length_in_feet (length_in_inches : ℕ) 
  (h : length_in_inches = 48) : 
  length_in_inches / 12 = 4 :=
by {
  have h_div := Nat.div_eq_of_eq_mul_left _ _ _ (by linarith) h,
  linarith,
}

end bookcase_length_in_feet_l434_434607


namespace max_value_of_function_l434_434511

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) : 
  ∃ y_max, y_max = (1 / 4) ∧ ∀ y, y = x * sqrt (1 - 4 * x ^ 2) → y ≤ y_max :=
sorry

end max_value_of_function_l434_434511


namespace find_f3_l434_434421

theorem find_f3 (a b : ℝ) (f : ℝ → ℝ)
  (h1 : f 1 = 4)
  (h2 : f 2 = 10)
  (h3 : ∀ x, f x = a * x^2 + b * x + 2) :
  f 3 = 20 :=
by
  sorry

end find_f3_l434_434421


namespace magnitude_of_b_l434_434033

-- Definitions of the vectors involved
def vec_a : ℝ × ℝ := (1, 2)

def vec_b (y : ℝ) : ℝ × ℝ := (-2, y)

-- Condition that vectors are parallel
def parallel_vectors (v₁ v₂ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v₁.1 = k * v₂.1 ∧ v₁.2 = k * v₂.2

-- The target theorem
theorem magnitude_of_b {y : ℝ} (h_parallel: parallel_vectors vec_a (vec_b y)) : 
  real.sqrt ((-2 : ℝ)^2 + y^2) = 2 * real.sqrt 5 :=
sorry

end magnitude_of_b_l434_434033


namespace min_arg_z_l434_434036

noncomputable def z (x y : ℝ) := x + y * Complex.I

def satisfies_condition (x y : ℝ) : Prop :=
  Complex.abs (z x y + 3 - Real.sqrt 3 * Complex.I) = Real.sqrt 3

theorem min_arg_z (x y : ℝ) (h : satisfies_condition x y) :
  Complex.arg (z x y) = 5 * Real.pi / 6 := 
sorry

end min_arg_z_l434_434036


namespace line_common_chord_eq_l434_434940

theorem line_common_chord_eq (a b : ℝ) :
  (∀ (x1 x2 y1 y2 : ℝ), x1^2 + y1^2 = 1 → (x2 - a)^2 + (y2 - b)^2 = 1 → 
    2 * a * x2 + 2 * b * y2 - 3 = 0) :=
sorry

end line_common_chord_eq_l434_434940


namespace athletes_meeting_time_and_overtakes_l434_434706

-- Define the constants for the problem
noncomputable def track_length : ℕ := 400
noncomputable def speed1 : ℕ := 155
noncomputable def speed2 : ℕ := 200
noncomputable def speed3 : ℕ := 275

-- The main theorem for the problem statement
theorem athletes_meeting_time_and_overtakes :
  ∃ (t : ℚ) (n_overtakes : ℕ), 
  (t = 80 / 3) ∧
  (n_overtakes = 13) ∧
  (∀ n : ℕ, n * (track_length / 45) = t) ∧
  (∀ k : ℕ, k * (track_length / 120) = t) ∧
  (∀ m : ℕ, m * (track_length / 75) = t) := 
sorry

end athletes_meeting_time_and_overtakes_l434_434706


namespace noah_ate_burgers_l434_434605

theorem noah_ate_burgers :
  ∀ (weight_hotdog weight_burger weight_pie : ℕ) 
    (mason_hotdog_weight : ℕ) 
    (jacob_pies noah_burgers mason_hotdogs : ℕ),
    weight_hotdog = 2 →
    weight_burger = 5 →
    weight_pie = 10 →
    (jacob_pies + 3 = noah_burgers) →
    (mason_hotdogs = 3 * jacob_pies) →
    (mason_hotdog_weight = 30) →
    (mason_hotdog_weight / weight_hotdog = mason_hotdogs) →
    noah_burgers = 8 :=
by
  intros weight_hotdog weight_burger weight_pie mason_hotdog_weight
         jacob_pies noah_burgers mason_hotdogs
         h1 h2 h3 h4 h5 h6 h7
  sorry

end noah_ate_burgers_l434_434605


namespace angle_JYH_parallel_lines_l434_434553

theorem angle_JYH_parallel_lines
  (AB CD EF GH : Line)
  (A B X F Y J H : Point)
  (h1 : parallel AB CD)
  (h2 : parallel EF GH)
  (h3 : angle AXF = 130)
  (h4 : angle FYJ = 50) :
  angle JYH = 130 := 
  sorry

end angle_JYH_parallel_lines_l434_434553


namespace contrapositive_roots_l434_434229

theorem contrapositive_roots {a b c : ℝ} (h : a ≠ 0) (hac : a * c ≤ 0) :
  ¬ (∀ x : ℝ, (a * x^2 - b * x + c = 0) → x > 0) :=
sorry

end contrapositive_roots_l434_434229


namespace standard_eq_of_ellipse_value_of_k_l434_434450

noncomputable def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∀ (x y : ℝ), (x, y) = (3, 16/5) → (x^2 / a^2 + y^2 / b^2 = 1) ∧ ((√(a^2 - b^2) / a) = 3/5)

noncomputable def find_k (k : ℝ) : Prop :=
  let P := (3 : ℝ, 16 / 5 : ℝ) in
  let x1 x2 : ℝ := sorry in  -- These are the x-coordinates of the points of intersection with the line
  let k1 := (k * x1 - k * 3 - 16/5) / (x1 - 3) in
  let k2 := (k * x2 - k * 3 - 16/5) / (x2 - 3) in
  k1 + k2 = 0

theorem standard_eq_of_ellipse : 
  ∀ x y : ℝ, let a := 5 in let b := 4 in
  ellipse_equation a b (by exact ⟨by norm_num, by norm_num⟩) :=
begin
  intros x y,
  unfold ellipse_equation,
  intro hyp,
  simp only [hyp],
  split,
  { sorry }, -- Skipping the actual proof steps
  { sorry }
end

theorem value_of_k (k : ℝ) :
  find_k (3 / 5) :=
begin
  unfold find_k,
  split,
  { sorry }, -- Skipping the actual proof steps
  { sorry }
end

end standard_eq_of_ellipse_value_of_k_l434_434450


namespace area_of_triangle_l434_434507

-- Define the given curve equation
def curve (x : ℝ) := (x - 4)^2 * (x + 3)

-- The x and y intercepts of the curve, (x, y at x = 0)
def x_intercepts : set ℝ := {x | curve x = 0}
def y_intercept := curve 0

-- Vertices of the triangle
def vertex_A := (-3 : ℝ, 0 : ℝ)
def vertex_B := (4 : ℝ, 0 : ℝ)
def vertex_C := (0 : ℝ, y_intercept)

-- Calculate the base and height of the triangle
def base := 7
def height := y_intercept

-- The area of the triangle
def triangle_area := 1/2 * base * height

-- The theorem to prove
theorem area_of_triangle : triangle_area = 168 := by
  sorry

end area_of_triangle_l434_434507


namespace rectangle_perimeter_l434_434648

theorem rectangle_perimeter : 
  ∃ (x y a b : ℝ), 
  (x * y = 2016) ∧ 
  (a * b = 2016) ∧ 
  (x^2 + y^2 = 4 * (a^2 - b^2)) → 
  2 * (x + y) = 8 * Real.sqrt 1008 :=
sorry

end rectangle_perimeter_l434_434648


namespace lines_parallel_l434_434663

-- Define the direction vectors as tuples for lines l1 and l2
def v1 : ℝ × ℝ × ℝ := (1, 0, -1)
def v2 : ℝ × ℝ × ℝ := (-2, 0, 2)

-- State that lines l1 and l2 are parallel
theorem lines_parallel : 
  ∃ k : ℝ, (v2.1 = k * v1.1) ∧ (v2.2 = k * v1.2) ∧ (v2.3 = k * v1.3) :=
by
  -- These conditions are derived from the definitions of v1 and v2
  use -2
  -- Placeholders for each element-wise equality in the direction vectors
  simp [v1, v2]
  sorry

end lines_parallel_l434_434663


namespace line_intersects_fixed_point_l434_434041

-- Define the given conditions
variable (a b c : ℝ)
variable (x y k m : ℝ)

-- Assume the conditions
axiom a_gt_b_gt_0 : a > b ∧ b > 0
axiom ellipse_eq : (x / a)^2 + (y / b)^2 = 1
axiom point_eq : x = 1 ∧ y = 3/2
axiom eccentricity : 1 / 2 = c / a

-- Define the ellipse equation
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x / 2)^2 + (y / 1.732) ^ 2 = 1

-- Define the fixed point
def fixed_point : ℝ × ℝ := (2 / 7, 0)

-- The proof statement to be shown
theorem line_intersects_fixed_point :
  (∀ (k m : ℝ), (3 + 4 * k^2 - m^2 > 0) → 
  ∃ (p : ℝ × ℝ), 
    (p = (k * (x - 2 / 7), k * y + m))) → 
    ∀ (k m : ℝ), 
      (y = k * x + m) → 
        p = fixed_point :=
begin
  sorry
end

end line_intersects_fixed_point_l434_434041


namespace find_consecutive_numbers_l434_434184

theorem find_consecutive_numbers (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c)
    (h_lcm : Nat.lcm a (Nat.lcm b c) = 660) : a = 10 ∧ b = 11 ∧ c = 12 := 
    sorry

end find_consecutive_numbers_l434_434184


namespace min_a_plus_b_l434_434028

-- Given conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Equation of line L passing through point (4,1) with intercepts a and b
def line_eq (a b : ℝ) : Prop := (4 / a) + (1 / b) = 1

-- Proof statement
theorem min_a_plus_b (h : line_eq a b) : a + b ≥ 9 :=
sorry

end min_a_plus_b_l434_434028


namespace sqrt_3920_is_28_sqrt_5_l434_434170

-- Define the term sqrt_3920 according to given conditions
def sqrt_3920 := Real.sqrt 3920

-- Define the simplified expression
def simplified := 28 * Real.sqrt 5

-- The theorem stating the equivalence to be proven
theorem sqrt_3920_is_28_sqrt_5 : sqrt_3920 = simplified :=
  sorry

end sqrt_3920_is_28_sqrt_5_l434_434170


namespace area_of_triangle_l434_434502

-- Define the given curve equation
def curve (x : ℝ) := (x - 4)^2 * (x + 3)

-- The x and y intercepts of the curve, (x, y at x = 0)
def x_intercepts : set ℝ := {x | curve x = 0}
def y_intercept := curve 0

-- Vertices of the triangle
def vertex_A := (-3 : ℝ, 0 : ℝ)
def vertex_B := (4 : ℝ, 0 : ℝ)
def vertex_C := (0 : ℝ, y_intercept)

-- Calculate the base and height of the triangle
def base := 7
def height := y_intercept

-- The area of the triangle
def triangle_area := 1/2 * base * height

-- The theorem to prove
theorem area_of_triangle : triangle_area = 168 := by
  sorry

end area_of_triangle_l434_434502


namespace area_of_triangle_l434_434483

theorem area_of_triangle :
  let f : ℝ → ℝ := λ x, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := (0, f 0)
  let base := (x_intercepts[0] - x_intercepts[1]).abs
  let height := y_intercept.2
  1 / 2 * base * height = 168 :=
by
  sorry

end area_of_triangle_l434_434483


namespace work_completion_l434_434748

theorem work_completion (d : ℝ) :
  (9 * (1 / d) + 8 * (1 / 20) = 1) ↔ (d = 15) :=
by
  sorry

end work_completion_l434_434748


namespace Kevin_crates_per_week_l434_434992

theorem Kevin_crates_per_week (a b c : ℕ) (h₁ : a = 13) (h₂ : b = 20) (h₃ : c = 17) :
  a + b + c = 50 :=
by 
  sorry

end Kevin_crates_per_week_l434_434992


namespace circle_diameter_l434_434268

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l434_434268


namespace num_correct_propositions_l434_434457

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

def Proposition1 (x1 x2 : ℝ) : Prop := f x1 = 0 ∧ f x2 = 0 → ∃ k : ℤ, x1 - x2 = k * Real.pi
def Proposition2 : Prop := ∀ x : ℝ, f x = 3 * Real.cos (2 * x - Real.pi / 4)
def Proposition3 : Prop := ∀ x : ℝ, -7 * Real.pi / 8 ≤ x ∧ x ≤ -3 * Real.pi / 8 → f x < f (x + 1e-3)
def Proposition4 : Prop := ∀ x : ℝ, f (x - Real.pi / 8) = -f x

theorem num_correct_propositions : num_correct := 
  let prop1 := ¬Proposition1
  let prop2 := Proposition2
  let prop3 := Proposition3
  let prop4 := Proposition4
  3 = (if prop1 then 0 else 1) + (if prop2 then 1 else 0) + (if prop3 then 1 else 0) + (if prop4 then 1 else 0) :=
sorry

end num_correct_propositions_l434_434457


namespace triangle_sides_squared_constant_l434_434639

variable {α β γ : ℝ}  -- angles of the triangle
variable {a b c : ℝ}  -- sides of the triangle

def d_1 := a^2 + b^2 - a * b * Real.cos γ
def d_2 := a^2 + c^2 - a * c * Real.cos β
def d_3 := c^2 + b^2 - c * b * Real.cos α

theorem triangle_sides_squared_constant :
  ∀ {α β γ : ℝ} {a b c : ℝ},
    d_1 = a^2 + b^2 + c^2 ∧
    d_2 = a^2 + b^2 + c^2 ∧
    d_3 = a^2 + b^2 + c^2 :=
by
  sorry

end triangle_sides_squared_constant_l434_434639


namespace max_of_three_numbers_l434_434208

theorem max_of_three_numbers : ∀ (a b c : ℕ), a = 10 → b = 11 → c = 12 → max (max a b) c = 12 :=
by
  intros a b c h1 h2 h3
  rw [h1, h2, h3]
  sorry

end max_of_three_numbers_l434_434208


namespace area_of_triangle_from_intercepts_l434_434486

theorem area_of_triangle_from_intercepts :
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := ({4, -3} : Set ℝ)
  let y_intercept := f 0
  let base := 4 - (-3)
  let height := y_intercept
  let area := 1 / 2 * base * height
  area = 168 := 
by
  -- Define the function f
  let f := λ x : ℝ, (x - 4) ^ 2 * (x + 3)
  -- Calculate x-intercepts
  have hx1 : f 4 = 0 := by simp [f, pow_two, mul_assoc]
  have hx2 : f (-3) = 0 := by simp [f, pow_two]
  let x_intercepts := ({4, -3} : Set ℝ)
  -- Calculate y-intercept
  have hy : f 0 = 48 := by simp [f, pow_two]
  let y_intercept := 48
  -- Define base and height
  let base := 4 - (-3)
  let height := y_intercept
  -- Compute the area
  let area := 1 / 2 * base * height
  -- Show that the area is 168
  show area = 168
  by
    simp [base, height, hy]
    norm_num
    sorry -- Skip the full proof

end area_of_triangle_from_intercepts_l434_434486


namespace small_tub_count_l434_434335

noncomputable def num_large_tubs := 3
noncomputable def total_cost := 48
noncomputable def cost_large_tub := 6
noncomputable def cost_small_tub := 5
noncomputable def correct_number_small_tubs := 6

theorem small_tub_count:
  18 + (cost_small_tub * correct_number_small_tubs) = total_cost :=
by
  have h1 : 18 = num_large_tubs * cost_large_tub,
    exact rfl,
  rw h1,
  have h2 : 30 = 48 - 18,
    exact rfl,
  rw [←h2, mul_comm],
  exact rfl

end small_tub_count_l434_434335


namespace distinct_collections_in_bag_l434_434152

def num_distinct_collections : ℕ := 288

theorem distinct_collections_in_bag :
  let letters := ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'E', 'O', 'N', 'S'],
      vowels := ['A', 'A', 'E', 'E', 'I'],
      consonants := ['C', 'C', 'N', 'N', 'T', 'T', 'M', 'M'],
      selected_vowels := 3,
      selected_consonants := 5
  in (∃ (n : ℕ), n = num_distinct_collections ∧ n = 288) :=
sorry

end distinct_collections_in_bag_l434_434152


namespace trajectory_and_fixed_point_l434_434416

-- Let's define the problem context first
variables {P M F: ℝ × ℝ} {k1 k2: ℝ}

-- Define given conditions
def condition1 : F = (1, 0) := rfl
def condition2 : M = (1, 2) := rfl
def condition_slope_sum : k1 + k2 = -1 := rfl

-- Define the trajectory equation derived
def trajectory_eqn (x y: ℝ): Prop := y^2 = 4 * x

-- Define the fixed point we need to prove
def fixed_point_passes (AB_x AB_y: ℝ): Prop := AB_x = 5 ∧ AB_y = -6

-- The Lean statement only
theorem trajectory_and_fixed_point :
  (∀ (P: ℝ × ℝ), let ⟨x, y⟩ := P in trajectory_eqn x y) →
  ((∀ (A B : ℝ × ℝ), lines_through M A k1 ∧ lines_through M B k2 → 
    (k1 + k2 = -1) → trajectory_eqn A.1 A.2 → trajectory_eqn B.1 B.2 →
    fixed_point_passes F M)
  sorry)

end trajectory_and_fixed_point_l434_434416


namespace area_of_triangle_l434_434504

-- Define the given curve equation
def curve (x : ℝ) := (x - 4)^2 * (x + 3)

-- The x and y intercepts of the curve, (x, y at x = 0)
def x_intercepts : set ℝ := {x | curve x = 0}
def y_intercept := curve 0

-- Vertices of the triangle
def vertex_A := (-3 : ℝ, 0 : ℝ)
def vertex_B := (4 : ℝ, 0 : ℝ)
def vertex_C := (0 : ℝ, y_intercept)

-- Calculate the base and height of the triangle
def base := 7
def height := y_intercept

-- The area of the triangle
def triangle_area := 1/2 * base * height

-- The theorem to prove
theorem area_of_triangle : triangle_area = 168 := by
  sorry

end area_of_triangle_l434_434504


namespace hyperbola_eccentricity_l434_434461

noncomputable def hyperbola_eccentricity_range (a b e : ℝ) (h_a_pos : 0 < a) (h_a_less_1 : a < 1) (h_b_pos : 0 < b) : Prop :=
  let c := Real.sqrt ((5 * a^2 - a^4) / (1 - a^2))
  let e := c / a
  e > Real.sqrt 5

theorem hyperbola_eccentricity (a b e : ℝ) (h_a_pos : 0 < a) (h_a_less_1 : a < 1) (h_b_pos : 0 < b) :
  hyperbola_eccentricity_range a b e h_a_pos h_a_less_1 h_b_pos := 
sorry

end hyperbola_eccentricity_l434_434461


namespace circle_diameter_l434_434267

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l434_434267


namespace determine_a_l434_434176

noncomputable def hyperbola_a_value : ℝ :=
  let a : ℝ := sorry,
      b : ℝ := a / real.sqrt 3,
      c : ℝ := real.sqrt (a ^ 2 + (a / real.sqrt 3) ^ 2) in
  if h : c = 1 then a else sorry

theorem determine_a : 
  ∃ a: ℝ, a > 0 ∧ (∃ b: ℝ, b = a / real.sqrt 3 ∧ b > 0 ∧ real.sqrt (a^2 + b^2) = 1) ∧ a = real.sqrt 3 / 2 :=
begin
  use (real.sqrt 3 / 2),
  split,
  { exact real.sqrt 3 / 2 > 0, },
  { use (real.sqrt 3 / 2 / real.sqrt 3),
    split,
    { refl, },
    split,
    { exact (real.sqrt 3 / 2 / real.sqrt 3) > 0, },
    { rw [real.sqrt_eq_rfl_iff, real.sqrt_eq_rfl_iff] at *,
      field_simp,
      norm_num, } },
end

end determine_a_l434_434176


namespace distinct_real_roots_l434_434407

-- Define the main equation
def main_equation (x : ℝ) : ℝ := x^2 + (9 * x^2) / (x + 3)^2 - 40

-- Statement for the problem
theorem distinct_real_roots : ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ main_equation r1 = 0 ∧ main_equation r2 = 0 ∧ ∀ x, main_equation x = 0 → x = r1 ∨ x = r2 :=
sorry

end distinct_real_roots_l434_434407


namespace log_base_49_of_7_l434_434395

theorem log_base_49_of_7 : log 49 7 = 1 / 2 :=
by {
    -- declaring the conditions properly
    have h : 49 = 7 ^ 2, by norm_num, -- or use exact ((7 ^ 2).symm),
     -- goal follows from the conditions
    exact sorry
}

end log_base_49_of_7_l434_434395


namespace parabola_focus_l434_434404

theorem parabola_focus (y : ℝ) : 
  let x := - (1/16 : ℝ) * y^2 + 2 in 
  (∃ h k p : ℝ, h = 2 ∧ k = 0 ∧ p = 4 ∧ x = - (1/4 : ℝ) / p * (y - k)^2 + h)
  → (h - p = -2 ∧ k = 0) :=
by
  intro h k p hyp
  sorry

end parabola_focus_l434_434404


namespace simplify_expression_l434_434165

variable (x y z : ℝ)

-- Statement of the problem to be proved.
theorem simplify_expression :
  (15 * x + 45 * y - 30 * z) + (20 * x - 10 * y + 5 * z) - (5 * x + 35 * y - 15 * z) = 
  (30 * x - 10 * z) :=
by
  -- Placeholder for the actual proof
  sorry

end simplify_expression_l434_434165


namespace final_population_l434_434188

theorem final_population : 
  let doubling_time := 5
  let initial_population := 1000
  let total_time := 44.82892142331043
  let number_of_doublings := total_time / doubling_time
  let final_population := initial_population * (2 ^ number_of_doublings)
  Math.round final_population = 495033 :=
by
  let doubling_time := 5
  let initial_population := 1000
  let total_time := 44.82892142331043
  let number_of_doublings := total_time / doubling_time
  let final_population := initial_population * (2 ^ number_of_doublings)
  have h : final_population ≈ 495033.0 := by sorry
  exact Math.round_eq h sorry

end final_population_l434_434188


namespace hyperbola_equation_l434_434597

def equation_of_hyperbola (x y a b : ℝ) : Prop :=
  (x^2 / a^2 - y^2 / b^2 = 1) ∧ a > 0 ∧ b > 0

def line_passing_through_focus (x y b: ℝ) : Prop :=
  let l := λ x, -b * (x - 1) in
  l(0) = b ∧ l(1) = 0

def asymptotes (x y a b : ℝ) : Prop :=
  (∃ m : ℝ, m = b / a ∧ y = m * x) ∨ (∃ m : ℝ, m = - b / a ∧ y = m * x)

theorem hyperbola_equation {x y a b : ℝ} (h1 : equation_of_hyperbola x y a b)
    (h2 : line_passing_through_focus x y b)
    (h3 : ∀ (a b : ℝ), asymptotes x y a b → ((b/a = -b) ∨ (b/a * -b = -1 → x^2 - y^2 = 1))) :
  a = 1 ∧ b = 1 → x^2 - y^2 = 1 := sorry

end hyperbola_equation_l434_434597


namespace circle_diameter_l434_434291

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l434_434291


namespace parents_age_when_Mark_was_born_l434_434603

-- Define the ages based on the problem conditions
constant Mark_age : ℕ := 18
constant John_age : ℕ := Mark_age - 10
constant Emma_age : ℕ := Mark_age - 4

constant parents_current_age : ℕ := 7 * John_age
constant mother_age_at_Emma_birth : ℕ := 25
constant mother_current_age : ℕ := mother_age_at_Emma_birth + Emma_age

-- Define the theorem we want to prove
theorem parents_age_when_Mark_was_born : parents_current_age - Mark_age = 38 :=
by
  -- We skip the proof details for now and assume it's provided correctly
  sorry

end parents_age_when_Mark_was_born_l434_434603


namespace root_is_neg_one_then_m_eq_neg_3_l434_434412

theorem root_is_neg_one_then_m_eq_neg_3 (m : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + m = 0 ∧ x = -1) : m = -3 :=
sorry

end root_is_neg_one_then_m_eq_neg_3_l434_434412


namespace alpha_value_f_positive_l434_434452

noncomputable def f (x : ℝ) (α : ℝ) := 2^(x + cos α) - 2^(-x + cos α)

theorem alpha_value (α : ℝ) (h1 : f 1 α = 3 * sqrt 2 / 4) (h2 : 0 ≤ α ∧ α ≤ π) : 
  α = 2 * π / 3 := 
sorry

theorem f_positive (m : ℝ) (θ : ℝ) (α : ℝ) 
  (h1 : f 1 α = 3 * sqrt 2 / 4) (h2 : 0 ≤ α ∧ α ≤ π) 
  (h3 : m < 1) : 
  f (m * abs (cos θ)) α + f (1 - m) α > 0 := 
sorry

end alpha_value_f_positive_l434_434452


namespace hyperbola_equation_l434_434594

theorem hyperbola_equation (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_hyperbola : ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) 
  (h_focus : ∃ (p : ℝ × ℝ), p = (1, 0))
  (h_line_passing_focus : ∀ y, ∃ (m c : ℝ), y = -b * y + c)
  (h_parallel : ∀ x y : ℝ, b/a = -b)
  (h_perpendicular : ∀ x y : ℝ, b/a * (-b) = -1) : 
  ∀ x y : ℝ, x^2 - y^2 = 1 :=
by
  sorry

end hyperbola_equation_l434_434594


namespace simplify_and_evaluate_l434_434655

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) : (x + 5)^2 - (x - 2) * (x + 2) = 9 :=
by
  rw [h]
  -- Continue with standard proof techniques here
  sorry

end simplify_and_evaluate_l434_434655


namespace fraction_simplification_l434_434372

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l434_434372


namespace problem1_problem2_problem3_l434_434653

-- Define the determinant for a 3x3 matrix
def det3x3 (a b c d e f g h i : ℚ) : ℚ :=
a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

-- Define the determinant for a 4x4 matrix
def det4x4 (a b c d e f g h i j k l m n o p : ℚ) : ℚ :=
a * (f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n)) - 
b * (e * (k * p - l * o) - g * (i * p - l * m) + h * (i * o - k * m)) + 
c * (e * (j * p - l * n) - f * (i * p - l * m) + h * (i * n - j * m)) - 
d * (e * (j * o - k * n) - f * (i * o - k * m) + g * (i * n - j * m))

-- Problem 1 statement
theorem problem1 (a b c : ℚ) : 
  det3x3 a b c (b+c) (c+a) (a+b) (b*c) (c*a) (a*b) = -(a + b + c) * (b - c) * (c - a) * (a - b) :=
sorry

-- Problem 2 statement
theorem problem2 (a b c : ℚ) : 
  det3x3 (b*c) a a^2 (a*c) b b^2 (a*b) c c^2 = (a - b) * (b - c) * (c - a) * (a * b + a * c + b * c) :=
sorry

-- Problem 3 statement
theorem problem3 (a b c d : ℚ) : 
  det4x4 (1 + a) 1 1 1 1 (1 + b) 1 1 1 1 (1 + c) 1 1 1 1 (1 + d) = 
  a * b * c * d * (1 + 1/a + 1/b + 1/c + 1/d) :=
sorry

end problem1_problem2_problem3_l434_434653


namespace square_root_domain_l434_434532

theorem square_root_domain (x : ℝ) : (∃ y, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end square_root_domain_l434_434532


namespace proof_1_proof_2_l434_434308

-- Given conditions
def P (a : ℝ) : ℝ × ℝ := (a, -2)
def parabolaC : set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 = 4 * p.2}
def is_tangent (p : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, (p.1 = k ∨ p.1 = -k) ∧ p ∈ parabolaC

def pointA (x1 y1 : ℝ) : Prop := is_tangent (x1, y1)
def pointB (x2 y2 : ℝ) : Prop := is_tangent (x2, y2)

-- Proof problems
theorem proof_1 (a x1 y1 x2 y2 : ℝ) (hA : pointA x1 y1) (hB : pointB x2 y2) :
  x1 * x2 + y1 * y2 = -4 := sorry

theorem proof_2 (a x1 y1 x2 y2 : ℝ) (hA : pointA x1 y1) (hB : pointB x2 y2) :
  let M : ℝ × ℝ := (3/2 * a, 1 + a^2 / 2)
  let F : ℝ × ℝ := (0, 1)
  ∀ x y : ℝ, (x, y) ∈ (set_of (λ p : ℝ × ℝ, (p.1 - a) * (p.1 - 3/2 * a) + (p.2 + 2) * (p.2 -1 - a^2 / 2) = 0)) →
  F ∈ (set_of (λ p : ℝ × ℝ, (p.1 - a) * (p.1 - 3/2 * a) + (p.2 + 2) * (p.2 -1 - a^2 / 2) = 0)) := sorry

end proof_1_proof_2_l434_434308


namespace find_r_in_arithmetic_sequence_l434_434967

-- Define an arithmetic sequence
def is_arithmetic_sequence (a b c d e f : ℤ) : Prop :=
  (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d) ∧ (e - d = f - e)

-- Define the given problem
theorem find_r_in_arithmetic_sequence :
  ∃ r : ℤ, ∀ p q s : ℤ, is_arithmetic_sequence 23 p q r s 59 → r = 41 :=
by
  sorry

end find_r_in_arithmetic_sequence_l434_434967


namespace solution_set_inequality_l434_434191

theorem solution_set_inequality (x : ℝ) : (| 2 * x - 1 | - | x - 2 | < 0) ↔ (-1 < x ∧ x < 1) := 
sorry

end solution_set_inequality_l434_434191


namespace simplify_series_l434_434656

theorem simplify_series : 
  (2 + 22 + 222 + ∙∙∙ + (2 * 10 ^ 2020 + 2)) = (2 * 10 ^ 2022 - 36398) / 81 :=
by 
  sorry

end simplify_series_l434_434656


namespace _l434_434974

noncomputable def circle_eq : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 1}

noncomputable def line_eq (t : ℝ) : Set (ℝ × ℝ) := {p | 3 * p.1 + 4 * p.2 + 8 = 0}

noncomputable theorem curves_disjoint : ∀ (t : ℝ), ∀ (x y : ℝ), ((x, y) ∈ circle_eq) → ((x, y) ∉ line_eq t) :=
by
  sorry

noncomputable def within_range (t : ℝ) : ℝ → ℝ → Prop :=
  λ (x y : ℝ), x + y = t

noncomputable theorem range_for_x_plus_y : ∀ (t : ℝ), (∃ (x y : ℝ), (x, y) ∈ circle_eq ∧ within_range t x y) → (1 - Real.sqrt 2 ≤ t ∧ t ≤ 1 + Real.sqrt 2) :=
by
  sorry

end _l434_434974


namespace triangle_DEF_DF_l434_434975

theorem triangle_DEF_DF {
  D E F M : Type 
  [IsTriangle D E F]
  (DE : length D E = 6)
  (EF : length E F = 10)
  (DM : isMedian D M F E)
  (DM_len : length D M = 5) :
  ∃ DF, length D F = 8 := 
sorry

end triangle_DEF_DF_l434_434975


namespace movie_theater_loss_l434_434310

theorem movie_theater_loss :
  let capacity := 50
  let ticket_price := 8.0
  let tickets_sold := 24
  (capacity * ticket_price - tickets_sold * ticket_price) = 208 := by
  sorry

end movie_theater_loss_l434_434310


namespace remainder_when_divided_by_5_l434_434391

theorem remainder_when_divided_by_5 : (1234 * 1987 * 2013 * 2021) % 5 = 4 :=
by
  sorry

end remainder_when_divided_by_5_l434_434391


namespace polynomial_no_xy_term_l434_434523

theorem polynomial_no_xy_term (m : ℝ) : 
  (∃ p : ℝ → ℝ → ℝ, p = λ x y, x^2 - m * x * y - y^2 + 6 * x * y - 1) →
  (∀ x y, (x^2 - y^2 + (-m + 6) * x * y - 1) ≠ 0 → m = 6) :=
by
  sorry

end polynomial_no_xy_term_l434_434523


namespace triangle_abe_area_l434_434212

-- Given definitions around the problem
variables (A B C D E F : Type) 
          
-- Conditions
variable [HasArea A B C 10] -- Triangle ABC has an area of 10
variable [OnSide D A B]     -- Point D is on side AB distinct from A and B
variable [OnSide E B C]     -- Point E is on side BC distinct from B and C
variable [OnSide F C A]     -- Point F is on side CA distinct from C and A
variable [Length D A = 2]   -- Length AD is 2
variable [Length D B = 3]   -- Length DB is 3
variable [AreaEqual A B E D B E F] -- Area of triangle ABE is equal to area of quadrilateral DBEF

-- Proof Goal
theorem triangle_abe_area : Area A B E = 6 := 
sorry

end triangle_abe_area_l434_434212


namespace problem1_problem2_problem3_l434_434965

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 9 = 0

def circle2 (x y m : ℝ) : Prop := (x + m)^2 + (y + m + 5)^2 = 2*m^2 + 8*m + 10

def circle1_center : (ℝ × ℝ) := (3, -2)
def circle2_center (m : ℝ) : (ℝ × ℝ) := (-m, -m - 5)

-- Problem (I)
theorem problem1 (h : circle2_center 5 = (cir︀cle_center_1).dist circle_center_2) : 2 :=
sorry

-- Problem (II)
theorem problem2 (x0 y0 : ℝ) (h : x0 + y0 + 1 = 0) : (x0, y0) = (0, -1) ∨ (x0, y0) = (-1, 0) :=
sorry

-- Problem (III)
theorem problem3 (k : ℝ) (line_bisects_circle1 : ∀ m ≠ -3, ∃ x : ℝ, ∃ y : ℝ, circle2 x y m ∧ (line_bisects_circle2 x y)) : k > 0 :=
sorry

end problem1_problem2_problem3_l434_434965


namespace problem_statement_l434_434546

variable (A B C D E P : Type) [euclidean_geometry E] (H1 : segment_parallel AE BC) :
  (isConvexPentagon A B C D E) 
  (H2 : angle ADE = angle BDC) (H3 : intersects AC BE P)

theorem problem_statement (H1 : segment_parallel AE BC) 
                          (H2 : angle ADE = angle BDC) 
                          (H3 : intersects AC BE P) :
  angle EAD = angle BDP ∧ angle CBD = angle ADP := 
  sorry

end problem_statement_l434_434546


namespace hyperbola_eccentricity_l434_434917

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (F : ℝ × ℝ) (hf : F = (sqrt (a^2 + b^2), 0))
  (h_asymp : ∀ (x : ℝ), (0, sqrt (a^2 + b^2)) = (b / a) * x) 
  (h_tangent : ∀ (r : ℝ), r = b) 
  (h_circle : ∀ (x : ℝ), (x = (sqrt (a^2 + b^2))) -> (y = (b^2 / a) = b))
  (h_perpendicular : ∀ (M : ℝ × ℝ), M.2 =  b * sqrt (a^2)) :
  ∃ e : ℝ, e = sqrt 2 := sorry

end hyperbola_eccentricity_l434_434917


namespace miles_left_to_reach_E_l434_434987

-- Given conditions as definitions
def total_journey : ℕ := 2500
def miles_driven : ℕ := 642
def miles_B_to_C : ℕ := 400
def miles_C_to_D : ℕ := 550
def detour_D_to_E : ℕ := 200

-- Proof statement
theorem miles_left_to_reach_E : 
  (miles_B_to_C + miles_C_to_D + detour_D_to_E) = 1150 :=
by
  sorry

end miles_left_to_reach_E_l434_434987


namespace find_N_l434_434826

theorem find_N :
  ∃ N : ℕ, N > 1 ∧
    (1743 % N = 2019 % N) ∧ (2019 % N = 3008 % N) ∧ N = 23 :=
by
  sorry

end find_N_l434_434826


namespace max_value_under_constraint_l434_434938

noncomputable def max_value_expression (a b c : ℝ) : ℝ :=
3 * a * b - 3 * b * c + 2 * c^2

theorem max_value_under_constraint
  (a b c : ℝ)
  (h : a^2 + b^2 + c^2 = 1) :
  max_value_expression a b c ≤ 3 :=
sorry

end max_value_under_constraint_l434_434938


namespace total_games_played_l434_434566

theorem total_games_played (jerry_wins dave_wins ken_wins : ℕ)
  (h1 : jerry_wins = 7)
  (h2 : dave_wins = jerry_wins + 3)
  (h3 : ken_wins = dave_wins + 5) :
  jerry_wins + dave_wins + ken_wins = 32 :=
by
  sorry

end total_games_played_l434_434566


namespace determinant_shift_l434_434573

variables {V : Type*} [AddCommGroup V] [Module ℝ V] [FiniteDimensional ℝ V]
variables (a b c d : V) (D : ℝ)

-- The definition of determinant D as an assumption
axiom det_original : a ⬝ (b × c) = D

-- The equivalence we need to prove
theorem determinant_shift (a b c d : V) (D : ℝ) (det_original : a ⬝ (b × c) = D) :
  (a + d) ⬝ ((b + d) × (c + d)) = D :=
sorry

end determinant_shift_l434_434573


namespace area_shaded_region_l434_434045

-- Define the points O, A, B, C, D, and E in terms of coordinates.
def O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (4, 0)
def B : (ℝ × ℝ) := (16, 0)
def C : (ℝ × ℝ) := (16, 12)
def D : (ℝ × ℝ) := (4, 12)
def E : (ℝ × ℝ) := (4, 12/3)

-- Define distances OA, OB, CB, and the triangle area conditions
def OA := dist O A
def OB := dist O B
def CB := dist C B
def DE := 12 - 3 -- DE computed from the problem
def DC := 12

noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  0.5 * abs((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)))

-- Proof statement: The area of triangle CDE is 54 square centimeters.
theorem area_shaded_region : area_triangle C D E = 54 := by
  -- This is where the proof would go, which we are not providing.
  sorry

end area_shaded_region_l434_434045


namespace cubes_closed_under_multiplication_l434_434145

theorem cubes_closed_under_multiplication :
  ∀ (a b : ℕ), ∃ c : ℕ, a > 0 ∧ b > 0 → a^3 * b^3 = c^3 :=
by {
  intros a b ha hb,
  sorry
}

end cubes_closed_under_multiplication_l434_434145


namespace bankers_discount_problem_l434_434745

theorem bankers_discount_problem
  (BD : ℚ) (TD : ℚ) (SD : ℚ)
  (h1 : BD = 36)
  (h2 : TD = 30)
  (h3 : BD = TD + TD^2 / SD) :
  SD = 150 := 
sorry

end bankers_discount_problem_l434_434745


namespace general_term_a_sum_Tn_l434_434031

section sequence_problem

variables {n : ℕ} (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Problem 1: General term formula for {a_n}
axiom Sn_def : ∀ n, S n = 1/4 * (a n + 1)^2
axiom a1_def : a 1 = 1
axiom an_diff : ∀ n, a (n+1) - a n = 2

theorem general_term_a : a n = 2 * n - 1 := sorry

-- Problem 2: Sum of the first n terms of sequence {b_n}
axiom an_formula : ∀ n, a n = 2 * n - 1
axiom bn_def : ∀ n, b n = 1 / (a n * a (n+1))

theorem sum_Tn : T n = n / (2 * n + 1) := sorry

end sequence_problem

end general_term_a_sum_Tn_l434_434031


namespace find_x_l434_434164

theorem find_x (x : ℝ) (h : x^2 + 75 = (x - 20)^2) : x = 8.125 :=
by
  sorry

end find_x_l434_434164


namespace midpoint_segment_on_segment_l434_434175

theorem midpoint_segment_on_segment 
  {A B C A1 C1 I K M : Type*} [IsTriangle A B C] 
  (h_right : angle ∠B = 90∘) 
  (h_bisectors : bisector A A1 I ∧ bisector C C1 I) 
  (h_perpendicular_1 : perpendicular (line_through C1) (line_through A A1)) 
  (h_perpendicular_2 : perpendicular (line_through A1) (line_through C C1))
  (h_intersect : intersects (line_through C1) (perpendicular_line A A1) (line_through A1) (perpendicular_line C C1) K)
  : midpoint_segment K I M = lies_on_segment M A C := 
sorry

end midpoint_segment_on_segment_l434_434175


namespace seating_arrangement_l434_434784

def valid_arrangements := 6

def Alice_refusal (A B C : Prop) := (¬ (A ∧ B)) ∧ (¬ (A ∧ C))
def Derek_refusal (D E C : Prop) := (¬ (D ∧ E)) ∧ (¬ (D ∧ C))

theorem seating_arrangement (A B C D E : Prop) : 
  Alice_refusal A B C ∧ Derek_refusal D E C → valid_arrangements = 6 := 
  sorry

end seating_arrangement_l434_434784


namespace fraction_identity_l434_434367

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l434_434367


namespace part1_f_inequality_part2_a_range_l434_434136

open Real

-- Proof Problem 1
theorem part1_f_inequality (x : ℝ) : 
    (|x - 1| + |x + 1| ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5) :=
sorry

-- Proof Problem 2
theorem part2_a_range (a : ℝ) : 
    (∀ x : ℝ, |x - 1| + |x - a| ≥ 2) ↔ (a = 3 ∨ a = -1) :=
sorry

end part1_f_inequality_part2_a_range_l434_434136


namespace area_of_triangle_l434_434479

theorem area_of_triangle :
  let f : ℝ → ℝ := λ x, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := (0, f 0)
  let base := (x_intercepts[0] - x_intercepts[1]).abs
  let height := y_intercept.2
  1 / 2 * base * height = 168 :=
by
  sorry

end area_of_triangle_l434_434479


namespace value_of_expression_l434_434190

theorem value_of_expression (x : ℝ) (h : x^2 - 5 * x + 6 < 0) : x^2 - 5 * x + 10 = 4 :=
sorry

end value_of_expression_l434_434190


namespace equation_of_locus_of_E_range_of_m_l434_434871

-- Definitions for the given conditions
def point := ℝ × ℝ

def A : point := (1, 0)

def circle_C (P : point) : Prop :=
  let (x, y) := P
  (x + 1)^2 + y^2 = 8

def line_CP (C P E : point) : Prop :=
  let (x1, y1) := C
  let (x2, y2) := P
  let (x3, y3) := E
  x3 * (y2 - y1) = x2 * (y3 - y1) + x1 * y2 - y1 * x2

def perpendicular_bisector_PA_intersect (P A E : point) : Prop :=
  let (x1, y1) := P
  let (x2, y2) := A
  let (x3, y3) := E
  2 * (x3 - (x1 + x2) / 2) = y2 - y1

-- Constants in the problem
noncomputable def C : point := (-1, 0)

-- Part (1): Prove the locus of E is an ellipse with the given equation
theorem equation_of_locus_of_E (P E : point) (h : circle_C P) :
  perpendicular_bisector_PA_intersect P A E → line_CP C P E →
  (∃ K : ℝ, 2 * K + K^2 - 2 = 0) :=
sorry

-- Part (2): Prove the range of m given the conditions
theorem range_of_m (k m : ℝ) :
  (∀ P Q : point, (∃ O : point, (circle_C P) ∧ (circle_C Q) ∧
  let (x1, y1) := P in let (x2, y2) := Q in
  (x1, k*x1 + m) = P ∧ (x2, k*x2 + m) = Q ∧
  let x1x2 := x1 * x2 in let y1y2 := (k*x1+m)*(k*x2+m) in
  x1x2 + y1y2 < 0)) →
  m^2 < 2/(2*k^2+1) → m^2 < 2/3 → 
  -√(2/3) < m ∧ m < √(2/3) :=
sorry

end equation_of_locus_of_E_range_of_m_l434_434871


namespace number_of_routes_l434_434805

-- Definitions based on the conditions in the original problem
def cities : Finset ℕ := {A, B, C, D, E, F}

def roads : Finset (ℕ × ℕ) :=
  {(A, B), (A, D), (A, E), (A, F), (B, C), (B, D), (C, D), (D, E), (D, F)}

def road_repeated : (ℕ × ℕ) := (A, F)

-- Statement: Total number of distinct routes from A to B using each road exactly once and repeating road_repeated is 3
theorem number_of_routes (trails : Finset (Finset (ℕ × ℕ))) :
  trails.card = 3 :=
  sorry

end number_of_routes_l434_434805


namespace total_rooms_l434_434348

-- Definitions for the problem conditions
variables (x y : ℕ)

-- Given conditions
def condition1 : Prop := x = 8
def condition2 : Prop := 2 * x + 3 * y = 31

-- The theorem to prove
theorem total_rooms (h1 : condition1 x) (h2 : condition2 x y) : x + y = 13 :=
by sorry

end total_rooms_l434_434348


namespace price_increase_l434_434768

-- Define the given conditions
def OriginalQuantity : ℕ := 71
def NewQuantity : ℕ := 63
def OriginalRevenue : ℕ := 568000
def NewRevenue : ℕ := 594000

-- Define variables for prices and increase in price
variable (P x : ℝ)

-- Define the equations derived from the conditions
def equation1 := P * OriginalQuantity = OriginalRevenue
def equation2 := (P + x) * NewQuantity = NewRevenue

-- State the theorem to prove the price increase
theorem price_increase : equation1 → equation2 → x = 1428.57 :=
by
  -- Placeholder for proof
  intros h1 h2
  sorry

end price_increase_l434_434768


namespace probability_of_53_sundays_in_leap_year_l434_434744

/--
Given:
1. A leap year has 366 days.
2. Since there are 7 days in a week, a year will have 52 weeks and 2 extra days.
3. For a leap year to have 53 Sundays, one of the two extra days must be a Sunday.
4. There are 49 possible combinations of the 2 extra days.
5. Out of these 49 combinations, there are 7 combinations where Sunday is one of the extra days.

Prove:
The probability that a leap year chosen at random will have 53 Sundays is 1/7.
-/
theorem probability_of_53_sundays_in_leap_year : ℚ :=
  let total_days : ℕ := 366
  let days_in_week : ℕ := 7
  let extra_days : ℕ := total_days % days_in_week
  let total_combinations : ℕ := days_in_week * days_in_week
  let favorable_combinations : ℕ := days_in_week
  favorable_combinations / total_combinations
  sorry

end probability_of_53_sundays_in_leap_year_l434_434744


namespace mass_percentage_Al_in_AlCl3_l434_434406

noncomputable def molar_mass_Al : ℝ := 26.98 
noncomputable def molar_mass_Cl : ℝ := 35.45 

noncomputable def molar_mass_AlCl3 : ℝ := molar_mass_Al + 3 * molar_mass_Cl

theorem mass_percentage_Al_in_AlCl3 : 
  (molar_mass_Al / molar_mass_AlCl3) * 100 ≈ 20.23 :=
by
  sorry

end mass_percentage_Al_in_AlCl3_l434_434406


namespace complex_dilation_image_l434_434764

def dilation_image (center scale_factor : ℂ) (z : ℂ):=
  scale_factor * (z - center) + center

theorem complex_dilation_image :
  dilation_image (0 + 6 * Complex.i) 3 (-1 + 2 * Complex.i) = -3 - 6 * Complex.i :=
by
  -- Proof would be here
  sorry

end complex_dilation_image_l434_434764


namespace probability_divisible_by_4_is_zero_l434_434695

def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

noncomputable def prob_of_divisibility_by_4 (n : ℕ) (h : n % 10 = 7) : ℚ :=
  if divisible_by_4 n then 1 else 0

theorem probability_divisible_by_4_is_zero :
  ∀ (M : ℕ), (100 ≤ M ∧ M < 1000) ∧ (M % 10 = 7) → prob_of_divisibility_by_4 M (by sorry) = 0 :=
sorry

end probability_divisible_by_4_is_zero_l434_434695


namespace circle_diameter_l434_434261

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434261


namespace cistern_height_l434_434760

theorem cistern_height (l w A : ℝ) (h : ℝ) (hl : l = 8) (hw : w = 6) (hA : 48 + 2 * (l * h) + 2 * (w * h) = 99.8) : h = 1.85 := by
  sorry

end cistern_height_l434_434760


namespace simplify_expression_l434_434166

theorem simplify_expression :
  (8 * 10^12) / (4 * 10^4) + 2 * 10^3 = 200002000 := 
by
  sorry

end simplify_expression_l434_434166


namespace mapping_image_l434_434859

theorem mapping_image (f : ℕ → ℕ) (h : ∀ x, f x = x + 1) : f 3 = 4 :=
by {
  sorry
}

end mapping_image_l434_434859


namespace exists_c_gt_zero_l434_434882

theorem exists_c_gt_zero (a b : ℝ) (h : a < b) : ∃ c > 0, a < b + c := 
sorry

end exists_c_gt_zero_l434_434882


namespace number_of_arrangements_l434_434099

theorem number_of_arrangements (P : Fin 5 → Type) (youngest : Fin 5) 
  (h_in_not_first_last : ∀ (i : Fin 5), i ≠ 0 → i ≠ 4 → i ≠ youngest) : 
  ∃ n, n = 72 := 
by
  sorry

end number_of_arrangements_l434_434099


namespace widest_opening_l434_434556

-- Definitions for the absolute values of coefficients
def A : ℝ := -10
def B : ℝ := 2
def C : ℝ := 1 / 100
def D : ℝ := -1

-- Theorem statement that the quadratic function with the smallest absolute value of the coefficient has the widest opening
theorem widest_opening :
  (|A| > |D| ∧ |D| > |B| ∧ |B| > |C|) →
  (|C| < |D| ∧ |D| < |B| ∧ |B| < |A|) :=
by {
  sorry
}

-- Constant to represent the answer
def correct_answer : ℝ := C

end widest_opening_l434_434556


namespace range_of_a_l434_434436

noncomputable def f (a x : ℝ) :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a) ∧
  (∀ x y : ℝ, x < y → f a x ≥ f a y) →
  1 / 7 ≤ a ∧ a < 1 / 3 := 
sorry

end range_of_a_l434_434436


namespace marbles_problem_l434_434344
open Nat

theorem marbles_problem : 
  ∃ m N : ℕ, 
    (let total_marbles := 7 + m in 
    m = 19 ∧ 
    N = choose (total_marbles - 1) 7 ∧
    (N % 1000 = 970)) :=
begin
  -- m = 19 Yellow marbles
  let m := 19,
  -- total_marbles = 7 Blue + 19 Yellow = 26
  let total_marbles := 7 + m,
  -- N = number of ways to arrange 26 marbles with the condition
  let N := choose (total_marbles - 1) 7,
  -- Proof statement
  use [m, N],
  split,
  { refl, },
  split,
  { refl, },
  {
    calc N % 1000 
        = (125970 % 1000) : by sorry-- Calculation will show as per combinatorial logic derived manually.
    ... = 970         : by sorry
  }
end

end marbles_problem_l434_434344


namespace perp_lines_implies_values_l434_434447

variable (a : ℝ)

def line1_perpendicular (a : ℝ) : Prop :=
  (1 - a) * (2 * a + 3) + a * (a - 1) = 0

theorem perp_lines_implies_values (h : line1_perpendicular a) :
  a = 1 ∨ a = -3 :=
by {
  sorry
}

end perp_lines_implies_values_l434_434447


namespace change_in_height_proof_l434_434303

-- Define the parameters
def field_length : ℝ := 90
def field_breadth : ℝ := 50
def tank_length : ℝ := 25
def tank_breadth : ℝ := 20
def tank_depth_shallow : ℝ := 2
def tank_depth_deep : ℝ := 6
def depth_ratio : ℝ := 1 / 2

-- Calculate the average depth
def average_depth : ℝ := (tank_depth_shallow + tank_depth_deep) / 2

-- Calculate the volume of the tank
def volume_tank : ℝ := tank_length * tank_breadth * average_depth

-- Calculate the area of the field and the tank
def area_field : ℝ := field_length * field_breadth
def area_tank : ℝ := tank_length * tank_breadth

-- Calculate the remaining area
def remaining_area : ℝ := area_field - area_tank

-- Define the change in height
def change_in_height : ℝ := volume_tank / remaining_area

-- Prove the change in height is 0.5 meters
theorem change_in_height_proof : change_in_height = 0.5 := 
by sorry

end change_in_height_proof_l434_434303


namespace probability_of_event_l434_434642

open Set Real

noncomputable def probability_event_interval (x : ℝ) : Prop :=
  1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 3

noncomputable def interval := Icc (0 : ℝ) (3 : ℝ)

noncomputable def event_probability := 1 / 3

theorem probability_of_event :
  ∀ x ∈ interval, probability_event_interval x → (event_probability) = 1 / 3 :=
by
  sorry

end probability_of_event_l434_434642


namespace no_angle_alpha_exists_in_0_pi_div_2_ap_sine_cosine_tangent_cotangent_l434_434818

theorem no_angle_alpha_exists_in_0_pi_div_2_ap_sine_cosine_tangent_cotangent :
  ¬ ∃ (α : ℝ), (0 < α ∧ α < (π / 2)) ∧
  ∃ (a b c d : ℝ) (h_order : {a, b, c, d} = {sin α, cos α, tan α, cot α}),
  (b - a = c - b ∧ c - b = d - c) :=
by
  sorry

end no_angle_alpha_exists_in_0_pi_div_2_ap_sine_cosine_tangent_cotangent_l434_434818


namespace evaluate_g_at_3_l434_434062

def g (x : ℝ) : ℝ := 9 * x^3 - 4 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g(3) = 209 := 
by 
  sorry

end evaluate_g_at_3_l434_434062


namespace solve_fractional_equation_l434_434196

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 4) : 
  (3 - x) / (x - 4) + 1 / (4 - x) = 1 → x = 3 :=
by {
  sorry
}

end solve_fractional_equation_l434_434196


namespace sqrt_x_minus_1_domain_l434_434529

theorem sqrt_x_minus_1_domain (x : ℝ) : (∃ y, y^2 = x - 1) → (x ≥ 1) := 
by 
  sorry

end sqrt_x_minus_1_domain_l434_434529


namespace problem_solution_l434_434877

def satisfies_conditions (x y : ℚ) : Prop :=
  (3 * x + y = 6) ∧ (x + 3 * y = 6)

theorem problem_solution :
  ∃ (x y : ℚ), satisfies_conditions x y ∧ 3 * x^2 + 5 * x * y + 3 * y^2 = 24.75 :=
by
  sorry

end problem_solution_l434_434877


namespace area_of_triangle_l434_434482

theorem area_of_triangle :
  let f : ℝ → ℝ := λ x, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := (0, f 0)
  let base := (x_intercepts[0] - x_intercepts[1]).abs
  let height := y_intercept.2
  1 / 2 * base * height = 168 :=
by
  sorry

end area_of_triangle_l434_434482


namespace point_D_coordinates_l434_434105

theorem point_D_coordinates 
  (F : (ℕ × ℕ)) 
  (coords_F : F = (5,5)) 
  (D : (ℕ × ℕ)) 
  (coords_D : D = (2,4)) :
  (D = (2,4)) :=
by 
  sorry

end point_D_coordinates_l434_434105


namespace problem_inequality_l434_434645

theorem problem_inequality (x y z : ℝ) (h1 : x + y + z = 0) (h2 : |x| + |y| + |z| ≤ 1) :
  x + (y / 2) + (z / 3) ≤ 1 / 3 :=
sorry

end problem_inequality_l434_434645


namespace smallest_total_squares_l434_434766

theorem smallest_total_squares (n : ℕ) (h : 4 * n - 4 = 2 * n) : n^2 = 4 :=
by
  sorry

end smallest_total_squares_l434_434766


namespace necessary_but_not_sufficient_condition_l434_434438

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > -2) :
  (∀ x, x^{2} < 4 → x > -2) ∧ (∃ x, x > -2 ∧ x^{2} ≥ 4) :=
sorry

end necessary_but_not_sufficient_condition_l434_434438


namespace fraction_computation_l434_434374

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l434_434374


namespace solution_in_positive_integers_l434_434400

theorem solution_in_positive_integers
  (a b c : ℕ)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (eq1 : a + b = Nat.gcd a b * Nat.gcd a b)
  (eq2 : b + c = Nat.gcd b c * Nat.gcd b c)
  (eq3 : c + a = Nat.gcd c a * Nat.gcd c a) :
  a = 2 ∧ b = 2 ∧ c = 2 :=
by {
  sorry,
}

end solution_in_positive_integers_l434_434400


namespace trains_clear_time_l434_434747

noncomputable def time_to_clear : ℕ :=
let length_train1 : ℕ := 120 in
let length_train2 : ℕ := 280 in
let speed_train1_kmph : ℕ := 42 in
let speed_train2_kmph : ℕ := 30 in
let relative_speed_kmph := speed_train1_kmph + speed_train2_kmph in
let relative_speed_mps : ℕ := relative_speed_kmph * 1000 / 3600 in
let total_length : ℕ := length_train1 + length_train2 in
total_length / relative_speed_mps

theorem trains_clear_time : time_to_clear = 20 := by
  sorry

end trains_clear_time_l434_434747


namespace find_f_107_l434_434894

-- Define the conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 3) = -f x

def piecewise_function (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x / 5

-- Main theorem to prove based on the conditions
theorem find_f_107 (f : ℝ → ℝ)
  (h_periodic : periodic_function f)
  (h_piece : piecewise_function f)
  (h_even : even_function f) : f 107 = 1 / 5 :=
sorry

end find_f_107_l434_434894


namespace games_attended_l434_434124

theorem games_attended (games_this_month games_last_month games_next_month total_games : ℕ) 
  (h1 : games_this_month = 11) 
  (h2 : games_last_month = 17) 
  (h3 : games_next_month = 16) : 
  total_games = games_this_month + games_last_month + games_next_month → 
  total_games = 44 :=
by
  sorry

end games_attended_l434_434124


namespace peter_reads_more_books_l434_434157

-- Definitions and conditions
def total_books : ℕ := 20
def peter_percentage_read : ℕ := 40
def brother_percentage_read : ℕ := 10

def percentage_to_count (percentage : ℕ) (total : ℕ) : ℕ := (percentage * total) / 100

-- Main statement to prove
theorem peter_reads_more_books :
  percentage_to_count peter_percentage_read total_books - percentage_to_count brother_percentage_read total_books = 6 :=
by
  sorry

end peter_reads_more_books_l434_434157


namespace measure_of_hypotenuse_l434_434213

theorem measure_of_hypotenuse (a : ℝ) : 
  (∃ a : ℝ, (1/2) * a^2 = 25 ∧ XY = a * Real.sqrt 2 ∧ XY > YZ ∧ XY = 10) :=
begin
  use a,
  sorry
end

end measure_of_hypotenuse_l434_434213


namespace average_speed_of_car_l434_434253

/-- The average speed of a car over four hours given specific distances covered each hour. -/
theorem average_speed_of_car
  (d1 d2 d3 d4 : ℝ)
  (t1 t2 t3 t4 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 40)
  (h3 : d3 = 60)
  (h4 : d4 = 100)
  (h5 : t1 = 1)
  (h6 : t2 = 1)
  (h7 : t3 = 1)
  (h8 : t4 = 1) :
  (d1 + d2 + d3 + d4) / (t1 + t2 + t3 + t4) = 55 :=
by sorry

end average_speed_of_car_l434_434253


namespace train_speed_correct_l434_434327

def length_of_train := 280 -- in meters
def time_to_pass_tree := 16 -- in seconds
def speed_of_train := 63 -- in km/hr

theorem train_speed_correct :
  (length_of_train / time_to_pass_tree) * (3600 / 1000) = speed_of_train :=
sorry

end train_speed_correct_l434_434327


namespace athletes_meet_time_number_of_overtakes_l434_434707

-- Define the speeds of the athletes
def speed1 := 155 -- m/min
def speed2 := 200 -- m/min
def speed3 := 275 -- m/min

-- Define the total length of the track
def track_length := 400 -- meters

-- Prove the minimum time for the athletes to meet again is 80/3 minutes
theorem athletes_meet_time (speed1 speed2 speed3 track_length : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) :
  ∃ t : ℚ, t = (80 / 3 : ℚ) :=
by
  sorry

-- Prove the number of overtakes during this time is 13
theorem number_of_overtakes (speed1 speed2 speed3 track_length t : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) (h5 : t = 80 / 3) :
  ∃ n : ℕ, n = 13 :=
by
  sorry

end athletes_meet_time_number_of_overtakes_l434_434707


namespace find_N_l434_434827

theorem find_N :
  ∃ N : ℕ, N > 1 ∧
    (1743 % N = 2019 % N) ∧ (2019 % N = 3008 % N) ∧ N = 23 :=
by
  sorry

end find_N_l434_434827


namespace line_pq_passes_through_orthocenter_l434_434570

theorem line_pq_passes_through_orthocenter
  (A B C : Point)
  (h₁ : Line.through A ⊥ Line.through B C)
  (ℓ : Line) 
  (h₂ : Line.through A ℓ)
  (ℓ' : Line := Reflection ℓ (Line.through A B)) 
  (ℓ'' : Line := Reflection ℓ (Line.through A C))
  (P : Point)
  (Q : Point)
  (h₃ : Intersection ℓ' (Line.perpendicular_through B (Line.through A B)) = P)
  (h₄ : Intersection ℓ'' (Line.perpendicular_through C (Line.through A C)) = Q)
  : PassesThrough (Line.through P Q) (Orthocenter A B C) :=
sorry

end line_pq_passes_through_orthocenter_l434_434570


namespace sequence_transformable_l434_434413

open Nat

theorem sequence_transformable (n : ℕ) (hn : n ≥ 3) :
  (∃ (σ : List ℕ), (σ = List.range' 1 n → (σ.tail.reverse = List.range' 1 n.tail.reverse))) ↔ (n % 4 = 0 ∨ n % 4 = 1) :=
sorry

end sequence_transformable_l434_434413


namespace smallest_positive_integer_remainder_l434_434733

theorem smallest_positive_integer_remainder : ∃ a : ℕ, 
  (a ≡ 2 [MOD 3]) ∧ (a ≡ 3 [MOD 5]) ∧ (a = 8) := 
by
  use 8
  split
  · exact Nat.modeq.symm (Nat.modeq.modeq_of_dvd _ (by norm_num))
  · split
    · exact Nat.modeq.symm (Nat.modeq.modeq_of_dvd _ (by norm_num))
    · rfl
  sorry  -- The detailed steps of the proof are omitted as per the instructions

end smallest_positive_integer_remainder_l434_434733


namespace range_m_n_l434_434047

noncomputable def f (m n x: ℝ) : ℝ := m * Real.exp x + x^2 + n * x

theorem range_m_n (m n: ℝ) :
  (∃ x, f m n x = 0) ∧ (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end range_m_n_l434_434047


namespace range_of_power_function_l434_434435

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^k

theorem range_of_power_function (k : ℝ) (h : k > 0) : 
  Set.range (λ x : ℝ, f x k) = set.Ici 1 := 
sorry

end range_of_power_function_l434_434435


namespace extreme_values_when_a_eq_e_zero_points_inequality_exponential_log_inequality_l434_434050

noncomputable theory

def f (x a : ℝ) := Real.exp x - a * Real.log x - a

theorem extreme_values_when_a_eq_e :
  is_min (f 1 Real.exp 1) ∧ ¬ ∃ M, ∀ x > 0, f x Real.exp ≤ M := sorry

theorem zero_points_inequality (a x1 x2 : ℝ) (h_a_pos : 0 < a) 
  (hx1x2 : 0 < x1 ∧ x1 < x2) 
  (hx1_gt_0: 0 < x1) 
  (hx2_gt_x1: x1 < x2) 
  (hx2_gt_0: 0 < x2) 
  (hfx1 : f x1 a = 0) 
  (hfx2 : f x2 a = 0) :
  (1 / a) < x1 ∧ x1 < 1 ∧ 1 < x2 ∧ x2 < a := sorry

theorem exponential_log_inequality (x : ℝ) (hx : 0 < x) :
  Real.exp (2 * x - 2) - Real.exp (x - 1) * Real.log x - x ≥ 0 := sorry

end extreme_values_when_a_eq_e_zero_points_inequality_exponential_log_inequality_l434_434050


namespace jason_probability_reroll_two_dice_l434_434121

-- Defining the problem conditions
def rolls : Type := list ℕ -- a representation of the three dice rolls, values between 1 and 6

-- Function to count favorable outcomes based on Jason's strategy
def count_favorable_outcomes (dice : rolls) : ℕ := sorry -- Detailed implementation omitted

-- Function to calculate probability (number of favorable outcomes / total outcomes)
def probability_of_rerolling_two_dice (dice : rolls) : ℚ := 
  (count_favorable_outcomes dice) / 216

-- Theorem: The probability that Jason chooses to reroll exactly two of the dice is 7/36
theorem jason_probability_reroll_two_dice :
  probability_of_rerolling_two_dice [1, 2, 3] = 7 / 36 :=
sorry

end jason_probability_reroll_two_dice_l434_434121


namespace random_events_count_is_five_l434_434451

-- Definitions of the events in the conditions
def event1 := "Classmate A successfully runs for class president"
def event2 := "Stronger team wins in a game between two teams"
def event3 := "A school has a total of 998 students, and at least three students share the same birthday"
def event4 := "If sets A, B, and C satisfy A ⊆ B and B ⊆ C, then A ⊆ C"
def event5 := "In ancient times, a king wanted to execute a painter. Secretly, he wrote 'death' on both slips of paper, then let the painter draw a 'life or death' slip. The painter drew a death slip"
def event6 := "It snows in July"
def event7 := "Choosing any two numbers from 1, 3, 9, and adding them together results in an even number"
def event8 := "Riding through 10 intersections, all lights encountered are red"

-- Tally up the number of random events
def is_random_event (event : String) : Bool :=
  event = event1 ∨
  event = event2 ∨
  event = event3 ∨
  event = event6 ∨
  event = event8

def count_random_events (events : List String) : Nat :=
  (events.map (λ event => if is_random_event event then 1 else 0)).sum

-- List of events
def events := [event1, event2, event3, event4, event5, event6, event7, event8]

-- Theorem statement
theorem random_events_count_is_five : count_random_events events = 5 :=
  by
    sorry

end random_events_count_is_five_l434_434451


namespace part1_part2_part3_l434_434053

namespace QuadraticProblem

def f (x m : ℝ) : ℝ :=
  x^2 - 2*(m-1)*x - 2*m + m^2

-- For part (1)
theorem part1 (m : ℝ) : f 0 m = 0 ↔ m = 0 ∨ m = 2 := by
  sorry

-- For part (2)
theorem part2 (m : ℝ) : 
  (∀ x, f x m = f (-x) m) → m = 1 := by
  sorry

noncomputable def f_sym : ℝ → ℝ :=
  λ x, x^2 - 1

-- For part (3)
theorem part3 (m : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x m ≥ 3) ↔ m ≤ 0 ∨ m ≥ 6 := by
  sorry

end QuadraticProblem

end part1_part2_part3_l434_434053


namespace probability_shots_result_l434_434347

open ProbabilityTheory

noncomputable def P_A := 3 / 4
noncomputable def P_B := 4 / 5
noncomputable def P_not_A := 1 - P_A
noncomputable def P_not_B := 1 - P_B

theorem probability_shots_result :
    (P_not_A * P_not_B * P_A) + (P_not_A * P_not_B * P_not_A * P_B) = 19 / 400 :=
    sorry

end probability_shots_result_l434_434347


namespace find_angle_C_max_area_triangle_l434_434946

-- Part I: Proving angle C
theorem find_angle_C (a b c : ℝ) (A B C : ℝ)
    (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) :
    C = Real.pi / 3 :=
sorry

-- Part II: Finding maximum area of triangle ABC
theorem max_area_triangle (a b : ℝ) (c : ℝ) (h_c : c = 2 * Real.sqrt 3) (A B C : ℝ)
    (h_A : A > 0) (h_B : B > 0) (h_C : C = Real.pi / 3)
    (h : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) :
    0.5 * a * b * Real.sin C ≤ 3 * Real.sqrt 3 :=
sorry

end find_angle_C_max_area_triangle_l434_434946


namespace solve_for_x_arcsin_arccos_l434_434173

theorem solve_for_x_arcsin_arccos (x : ℝ) (h : real.arcsin (3 * x) - real.arccos (2 * x) = real.pi / 6) : 
  x = -real.sqrt 7⁻¹ := 
sorry

end solve_for_x_arcsin_arccos_l434_434173


namespace workers_time_l434_434215

variables (x y: ℝ)

theorem workers_time (h1 : (x > 0) ∧ (y > 0)) 
                     (h2 : (3/x + 2/y = 11/20)) 
                     (h3 : (1/x + 1/y = 1/2)) :
                     (x = 10 ∧ y = 8) := 
by
  sorry

end workers_time_l434_434215


namespace probability_three_hits_out_of_four_l434_434775

noncomputable def hits_target (n : ℕ) : Prop := n ≥ 2 ∧ n ≤ 9

noncomputable def group_successful (nums : list ℕ) : Prop :=
(nums.filter hits_target).length ≥ 3

def groups : list (list ℕ) := 
[
  [7, 5, 2, 7], [0, 2, 9, 3], [7, 1, 4, 0], [9, 8, 5, 7], [0, 3, 4, 7], 
  [4, 3, 7, 3], [8, 6, 3, 6], [6, 9, 4, 7], [1, 4, 1, 7], [4, 6, 9, 8],
  [0, 3, 7, 1], [6, 2, 3, 3], [2, 6, 1, 6], [8, 0, 4, 5], [6, 0, 1, 1], 
  [3, 6, 6, 1], [9, 5, 9, 7], [7, 4, 2, 4], [7, 6, 1, 0], [4, 2, 8, 1]
]

theorem probability_three_hits_out_of_four (g_hit_prob : ℚ := 15 / 20) :
  (∑ g in groups, if (group_successful g) then 1 else 0) / groups.length = g_hit_prob :=
sorry

end probability_three_hits_out_of_four_l434_434775


namespace negation_of_existential_proposition_l434_434672

-- Define the propositions
def proposition (x : ℝ) := x^2 - 2 * x + 1 ≤ 0

-- Define the negation of the propositions
def negation_prop (x : ℝ) := x^2 - 2 * x + 1 > 0

-- Theorem to prove that the negation of the existential proposition is the universal proposition
theorem negation_of_existential_proposition
  (h : ¬ ∃ x : ℝ, proposition x) :
  ∀ x : ℝ, negation_prop x :=
by
  sorry

end negation_of_existential_proposition_l434_434672


namespace triangle_area_is_168_l434_434492

def curve (x : ℝ) : ℝ :=
  (x - 4)^2 * (x + 3)

noncomputable def x_intercepts : set ℝ :=
  {x | curve x = 0}

noncomputable def y_intercept : ℝ :=
  curve 0

theorem triangle_area_is_168 :
  let base := 7 in
  let height := y_intercept in
  let area := (1 / 2) * base * height in
  area = 168 :=
by
  sorry

end triangle_area_is_168_l434_434492


namespace area_of_triangle_DEF_l434_434972

-- Definitions and conditions
def square_area (s : ℝ) := s * s
def side_length_of_square (A : ℝ) := real.sqrt A

-- Given conditions
def PQRS_area : ℝ := 49
def small_square_side : ℝ := 2
def large_square_side : ℝ := side_length_of_square PQRS_area

def EF_length : ℝ := large_square_side - 2 * small_square_side

-- Altitude calculation based on the problem conditions
def DN_length : ℝ := (large_square_side / 2) + small_square_side + small_square_side

-- Function to calculate the area of an isosceles triangle
def triangle_area (base height : ℝ) := 0.5 * base * height

-- The proof statement
theorem area_of_triangle_DEF :
  triangle_area EF_length DN_length = 51 / 4 :=
by
  -- Here, we would normally complete the proof. In this case, we use 'sorry' to focus on the statement itself.
  sorry

end area_of_triangle_DEF_l434_434972


namespace total_shingles_for_all_roofs_l434_434126

def roof_A_length : ℕ := 20
def roof_A_width : ℕ := 40
def roof_A_shingles_per_sqft : ℕ := 8

def roof_B_length : ℕ := 25
def roof_B_width : ℕ := 35
def roof_B_shingles_per_sqft : ℕ := 10

def roof_C_length : ℕ := 30
def roof_C_width : ℕ := 30
def roof_C_shingles_per_sqft : ℕ := 12

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def total_area (length : ℕ) (width : ℕ) : ℕ :=
  2 * area length width

def total_shingles_needed (length : ℕ) (width : ℕ) (shingles_per_sqft : ℕ) : ℕ :=
  total_area length width * shingles_per_sqft

theorem total_shingles_for_all_roofs :
  total_shingles_needed roof_A_length roof_A_width roof_A_shingles_per_sqft +
  total_shingles_needed roof_B_length roof_B_width roof_B_shingles_per_sqft +
  total_shingles_needed roof_C_length roof_C_width roof_C_shingles_per_sqft = 51900 :=
by
  sorry

end total_shingles_for_all_roofs_l434_434126


namespace cindy_correct_answer_l434_434803

noncomputable def cindy_number (x : ℝ) : Prop :=
  (x - 10) / 5 = 40

theorem cindy_correct_answer (x : ℝ) (h : cindy_number x) : (x - 4) / 10 = 20.6 :=
by
  -- The proof is omitted as instructed
  sorry

end cindy_correct_answer_l434_434803


namespace determine_x_l434_434096

variable (A B C x : ℝ)
variable (hA : A = x)
variable (hB : B = 2 * x)
variable (hC : C = 45)
variable (hSum : A + B + C = 180)

theorem determine_x : x = 45 := 
by
  -- proof steps would go here
  sorry

end determine_x_l434_434096


namespace example_functions_for_behavior_finite_bounded_example_functions_for_behavior_unbounded_l434_434851

noncomputable def does_oscillate_finite (f : ℝ → ℝ) (p a b : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < abs (x - p) < δ → abs (f x) < ε ∧ (a ≤ f x ∧ f x ≤ b)

noncomputable def does_oscillate_unbounded (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ M > 0, ∃ δ > 0, ∀ x, 0 < abs (x - p) < δ → abs (f x) > M

theorem example_functions_for_behavior_finite_bounded:
  ∃ f g : ℝ → ℝ, ∃ (p a b : ℝ) (h : a < b),
    does_oscillate_finite (λ x, Real.sin (1 / (x - p))) p (-1) 1 ∧
    does_oscillate_finite (λ x, a + (b - a) * Real.sin (1 / (x - p))^2) p a b :=
by {
  sorry
}

theorem example_functions_for_behavior_unbounded:
  ∃ f g : ℝ → ℝ, ∃ p : ℝ,
    does_oscillate_unbounded (λ x, 1 / (x - p)) p ∧
    does_oscillate_unbounded (λ x, Real.sin (1 / (x - p))) p :=
by {
  sorry
}

end example_functions_for_behavior_finite_bounded_example_functions_for_behavior_unbounded_l434_434851


namespace least_m_plus_n_l434_434582

theorem least_m_plus_n (m n : ℕ) (hmn : Nat.gcd (m + n) 330 = 1) (hm_multiple : m^m % n^n = 0) (hm_not_multiple : ¬ (m % n = 0)) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  m + n = 119 :=
sorry

end least_m_plus_n_l434_434582


namespace fraction_simplification_l434_434371

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l434_434371


namespace factorization_exists_l434_434666

theorem factorization_exists (a b x y : ℝ) : 
  (a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1)) →
  ∃ (m n p : ℤ), (a^(m : ℕ) * x - a^(n : ℕ)) * (a^(p : ℕ) * y - a^3) = a^5 * b^5 ∧ m * n * p = 2 :=
by
  -- Provided equation is true
  intro h1,
  -- We need to show there exist integers m, n, p such that the given factorization holds and their product is 2
  sorry

end factorization_exists_l434_434666


namespace sum_of_c_n_l434_434692

noncomputable def a_n (n : ℕ) : ℝ := 2n - 1

noncomputable def b_n (n : ℕ) : ℝ := (2 / 3) * (1 / 3)^(n - 1)

noncomputable def c_n (n : ℕ) : ℝ := a_n n * b_n n

noncomputable def S_n (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, c_n (i + 1))

theorem sum_of_c_n (n : ℕ) : S_n n = 2 - (2n + 2) * (1 / 3)^n := sorry

end sum_of_c_n_l434_434692


namespace circle_diameter_l434_434295

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l434_434295


namespace product_implication_l434_434057

theorem product_implication (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab : a * b > 1) : a > 1 ∨ b > 1 :=
sorry

end product_implication_l434_434057


namespace circle_diameter_problem_circle_diameter_l434_434287

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l434_434287


namespace compare_abc_l434_434856

noncomputable def a : ℝ := Real.log_base (1 / 2) 3
noncomputable def b : ℝ := Real.log (1 / 2)
noncomputable def c : ℝ := (1 / 3)^((1 : ℝ) / 2)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l434_434856


namespace length_OS_correct_l434_434758

noncomputable def length_segment_OS 
  (O P : Type) [metric_space O] [metric_space P] 
  (rO rP : ℝ) (radius_O : rO = 12) (radius_P : rP = 4) 
  (externally_tangent : O ≠ P ∨ rO + rP = 16) 
  (segment_TS : Type) [has_point T segment_TS] [has_point S segment_TS]
  (tangent_O : ∀ t ∈ T, ∀ o ∈ O, dist t o = rO)
  (tangent_P : ∀ s ∈ S, ∀ p ∈ P, dist s p = rP) : ℝ :=
  sqrt (((12: ℝ) ^ 2) + ((8 * sqrt 3) ^ 2)) -- This is essentially sqrt(336)

theorem length_OS_correct 
  (O P : Type) [metric_space O] [metric_space P] 
  (rO rP : ℝ) (radius_O : rO = 12) (radius_P : rP = 4) 
  (externally_tangent : O ≠ P ∨ rO + rP = 16) 
  (segment_TS : Type) [has_point T segment_TS] [has_point S segment_TS]
  (tangent_O : ∀ t ∈ T, ∀ o ∈ O, dist t o = rO)
  (tangent_P : ∀ s ∈ S, ∀ p ∈ P, dist s p = rP) : 
  length_segment_OS O P rO rP radius_O radius_P externally_tangent segment_TS tangent_O tangent_P = 4 * sqrt 21 :=
  sorry

end length_OS_correct_l434_434758


namespace andy_cavities_l434_434343

def candy_canes_from_parents : ℕ := 2
def candy_canes_per_teacher : ℕ := 3
def number_of_teachers : ℕ := 4
def fraction_to_buy : ℚ := 1 / 7
def cavities_per_candies : ℕ := 4

theorem andy_cavities : (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers 
                         + (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers) * fraction_to_buy)
                         / cavities_per_candies = 4 := by
  sorry

end andy_cavities_l434_434343


namespace real_part_of_expression_l434_434589

noncomputable def complex_num (x y : ℝ) : ℂ :=
  x + y * complex.I

theorem real_part_of_expression (z : ℂ) (hz1 : z.im ≠ 0) (hz2 : complex.abs z = 2) :
  (complex.re (2 / (1 - z))) = 2/5 :=
sorry

end real_part_of_expression_l434_434589


namespace Geometry_l434_434356

-- Defining the problem conditions and main theorem statement in Lean 4
theorem Geometry.TangentsAndSegmentsSum (O A B C T1 T2 T3 : Type) 
  [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  [MetricSpace T1] [MetricSpace T2] [MetricSpace T3]
  (ω : O) (radius_O : ℝ) (center_O : O) 
  (radius_def : radius_O = 3) (OA_length : ℝ) (OA_def : OA_length = 10)
  (tangent_AT1_AT2 : ∀ x, x = sqrt 91)
  (BT_CT : ∀ x, x = 9/2)
  (BC_length : ℝ) (BC_def : BC_length = 9) : 
  (AB_length AC_length : ℝ) (AB_def AC_def : AB_length + AC_length = 2 * sqrt 91 + 9) :=
  sorry
  
end Geometry_l434_434356


namespace molecular_weight_is_171_35_l434_434222

def atomic_weight_ba : ℝ := 137.33
def atomic_weight_o : ℝ := 16.00
def atomic_weight_h : ℝ := 1.01

def molecular_weight : ℝ :=
  (1 * atomic_weight_ba) + (2 * atomic_weight_o) + (2 * atomic_weight_h)

-- The goal is to prove that the molecular weight is 171.35
theorem molecular_weight_is_171_35 : molecular_weight = 171.35 :=
by
  sorry

end molecular_weight_is_171_35_l434_434222


namespace distance_is_correct_l434_434251

noncomputable def distance_between_home_and_school : ℝ :=
  let t := (23 / 60 : ℝ) in
  let d := 3 * (t + 7 / 60) in
  d

theorem distance_is_correct : distance_between_home_and_school = 1.5 := by
  let t := (23 / 60 : ℝ)
  have h1 : d = 3 * (t + 7 / 60) := rfl
  have h2 : d = 6 * (t - 8 / 60) := by
    sorry -- Derive this from the conditions
  have eqn : 3 * (t + 7 / 60) = 6 * (t - 8 / 60) := by
    sorry -- Derived from the equations given in the problem
  calc
    d = 3 * (t + 7 / 60) : h1
    ... = 3 * (30 / 60)  : by sorry -- Fill in detailed steps here
    ... = 1.5            : by norm_num

end distance_is_correct_l434_434251


namespace focus_coordinates_l434_434519

theorem focus_coordinates (a : ℝ) (h : a - 4 ≠ 0 ∧ a + 5 ≠ 0) : 
  foci (a - 4) (a + 5) = (0, ±3) :=
sorry

end focus_coordinates_l434_434519


namespace cos_gamma_value_l434_434999

-- Define the conditions and variables
variables {Q : Type} [x : ℝ] [y : ℝ] [z : ℝ]
  (Q_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (cos_alpha : ℝ) (cos_beta : ℝ) (cos_gamma : ℝ)
  (alpha : ℝ) (beta : ℝ) (gamma : ℝ)
  (r1 : cos_alpha = 1 / 4)
  (r2 : cos_beta = 1 / 3)
  (gamma_obtuse : 90 < gamma ∧ gamma < 180)

-- main theorem to prove
theorem cos_gamma_value : cos_gamma = - (sqrt 119) / 12 :=
  sorry

end cos_gamma_value_l434_434999


namespace theater_loss_l434_434316

-- Define the conditions
def theater_capacity : Nat := 50
def ticket_price : Nat := 8
def tickets_sold : Nat := 24

-- Define the maximum revenue and actual revenue
def max_revenue : Nat := theater_capacity * ticket_price
def actual_revenue : Nat := tickets_sold * ticket_price

-- Define the money lost by not selling out
def money_lost : Nat := max_revenue - actual_revenue

-- Theorem statement to prove
theorem theater_loss : money_lost = 208 := by
  sorry

end theater_loss_l434_434316


namespace circumference_of_cone_base_l434_434323

-- Define the given data
def volume : ℝ := 24 * real.pi
def height : ℝ := 6

-- Define the formulas used
def volume_cone (r h : ℝ) : ℝ := (1/3) * real.pi * r^2 * h
def circumference (r : ℝ) : ℝ := 2 * real.pi * r

-- State the problem we want to prove
theorem circumference_of_cone_base (r : ℝ) (hr : r = real.sqrt 12) :
  volume_cone r height = volume → circumference r = 4 * real.sqrt 3 * real.pi :=
by
  sorry

end circumference_of_cone_base_l434_434323


namespace max_view_angle_dist_l434_434218

theorem max_view_angle_dist (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : ∃ (x : ℝ), x = Real.sqrt (b * (a + b)) := by
  sorry

end max_view_angle_dist_l434_434218


namespace round_fraction_to_3_decimal_places_l434_434933

theorem round_fraction_to_3_decimal_places : 
  (Real.round (8 / 11 * 1000) / 1000 = 0.727) :=
by 
  sorry

end round_fraction_to_3_decimal_places_l434_434933


namespace common_ratio_geometric_sequence_l434_434936

theorem common_ratio_geometric_sequence (a : ℝ) (h : a + real.log 2016 / real.log 3 = 0 → a + (real.log 2016 / (2 * real.log 3)) = 0 → a + (real.log 2016 / (3 * real.log 3)) = 0 → 
                       (∃ q : ℝ, a + real.log 2016 / real.log 3 + q = a + (real.log 2016 / (2 * real.log 3)) ∧
                                     a + (real.log 2016 / (2 * real.log 3)) + q = a + (real.log 2016 / (3 * real.log 3)))) :
  q = 1/3 :=
sorry

end common_ratio_geometric_sequence_l434_434936


namespace triangle_area_l434_434499

-- Define the curve and points
noncomputable def curve (x : ℝ) : ℝ := (x-4)^2 * (x+3)

-- Define the x-intercepts
def x_intercept1 := 4
def x_intercept2 := -3

-- Define the y-intercept
def y_intercept := curve 0

-- Define the base and height of the triangle
def base : ℝ := x_intercept1 - x_intercept2
def height : ℝ := y_intercept

-- Statement of the problem: calculating the area of the triangle
theorem triangle_area : (1/2) * base * height = 168 := by
  sorry

end triangle_area_l434_434499


namespace perpendicular_lines_slope_condition_l434_434896

theorem perpendicular_lines_slope_condition (k : ℝ) :
  (∀ x y : ℝ, y = k * x - 1 ↔ x + 2 * y + 3 = 0) → k = 2 :=
by
  sorry

end perpendicular_lines_slope_condition_l434_434896


namespace sin_alpha_minus_cos_alpha_l434_434448

theorem sin_alpha_minus_cos_alpha {α : ℝ} (h1 : ∃ x : ℝ, x < 0 ∧ ∃ y : ℝ, y = -3 * x ∧ angle α (x, y)) : 
  sin α - cos α = (2 * real.sqrt 10) / 5 :=
sorry

end sin_alpha_minus_cos_alpha_l434_434448


namespace solve_inequality_l434_434657

noncomputable def within_interval (x : ℝ) : Prop :=
  x > -3 ∧ x < 5

theorem solve_inequality (x : ℝ) :
  (x^3 - 125) / (x + 3) < 0 ↔ within_interval x :=
sorry

end solve_inequality_l434_434657


namespace bekah_days_left_l434_434791

theorem bekah_days_left 
  (total_pages : ℕ)
  (pages_read : ℕ)
  (pages_per_day : ℕ)
  (remaining_pages : ℕ := total_pages - pages_read)
  (days_left : ℕ := remaining_pages / pages_per_day) :
  total_pages = 408 →
  pages_read = 113 →
  pages_per_day = 59 →
  days_left = 5 :=
by {
  sorry
}

end bekah_days_left_l434_434791


namespace part_I_part_II_part_III_l434_434449

-- Conditions
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def line_through_origin (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (y = k * x)

def distinct_points_on_ellipse (A B P : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse P.1 P.2 ∧ A ≠ P ∧ B ≠ P

variables {A B P C D E F : ℝ × ℝ}
variables {k1 k2 : ℝ}

-- (I)
theorem part_I (h : distinct_points_on_ellipse A B P) (hA : A = (-2, 0)) (hB : B = (2, 0)) :
  ¬ (∃ P : ℝ × ℝ, ellipse P.1 P.2 ∧ P ≠ A ∧ P ≠ B ∧ (vector_angle (P.1 - A.1, P.2 - A.2) (P.1 - B.1, P.2 - B.2) = real.pi / 2)) :=
sorry

-- (II)
theorem part_II (h : distinct_points_on_ellipse A B P) (h₁ : k1 ≠ 0) (h₂ : k2 ≠ 0) :
  k1 * k2 = -1 / 4 :=
sorry

-- (III)
theorem part_III (h : distinct_points_on_ellipse A B P) (hC : line_through_origin C.1 C.2 ∧ ∀ y, ellipse y (k1 * y) → ellipse C.1 C.2) (hE : line_through_origin E.1 E.2 ∧ ∀ y, ellipse y (k2 * y) → ellipse E.1 E.2) :
  |(dist C D)| ^ 2 + |(dist E F)| ^ 2 = 20 :=
sorry

end part_I_part_II_part_III_l434_434449


namespace hex_B2F_to_dec_l434_434385

theorem hex_B2F_to_dec : 
  let A := 10
  let B := 11
  let C := 12
  let D := 13
  let E := 14
  let F := 15
  let base := 16
  let b2f := B * base^2 + 2 * base^1 + F * base^0
  b2f = 2863 :=
by {
  sorry
}

end hex_B2F_to_dec_l434_434385


namespace athletics_competition_races_l434_434544

theorem athletics_competition_races :
  ∀ (total_sprinters lanes qualifying_runners eliminated_runners : ℕ),
    total_sprinters = 275 →
    lanes = 8 →
    qualifying_runners = 2 →
    eliminated_runners = lanes - qualifying_runners →
    (∃ (rounds rounds1 rounds2 rounds3 rounds4 final_race : ℕ),
      rounds = 35 ∧
      rounds1 = 9 ∧
      rounds2 = 3 ∧
      rounds3 = 1 ∧
      final_race = 1 ∧
      rounds + rounds1 + rounds2 + rounds3 + final_race = 49) :=
by
  intros total_sprinters lanes qualifying_runners eliminated_runners
  intros h1 h2 h3 h4
  use [35, 9, 3, 1, 1]
  split ; try {exact dec_trivial}
  sorry

end athletics_competition_races_l434_434544


namespace apple_percentage_is_23_l434_434659

def total_responses := 70 + 80 + 50 + 30 + 70
def apple_responses := 70

theorem apple_percentage_is_23 :
  (apple_responses : ℝ) / (total_responses : ℝ) * 100 = 23 := 
by
  sorry

end apple_percentage_is_23_l434_434659


namespace sin_cos_relation_l434_434932

theorem sin_cos_relation 
  (α β : Real) 
  (h : 2 * Real.sin α - Real.cos β = 2) 
  : Real.sin α + 2 * Real.cos β = 1 ∨ Real.sin α + 2 * Real.cos β = -1 := 
sorry

end sin_cos_relation_l434_434932


namespace find_number_l434_434254

theorem find_number (a : ℤ) (h : a - a + 99 * (a - 99) = 19802) : a = 299 := 
by 
  sorry

end find_number_l434_434254


namespace part1_part2_l434_434445

noncomputable def f : ℝ → ℝ := sorry
def a : ℝ := sorry

-- Conditions
axiom f_odd : ∀ x : ℝ, x ≠ 0 → f(-x) = -f(x)
axiom f_increasing : ∀ x y : ℝ, 0 < x → x < y → f(x) < f(y)
axiom f_neg1_zero : f(-1) = 0
axiom f_a_minus_half_neg : f(a - 0.5) < 0

-- Questions
theorem part1 : f(1) = 0 := sorry
theorem part2 : (1 / 2 < a ∧ a < 3 / 2) ∨ (a < -1 / 2) := sorry

end part1_part2_l434_434445


namespace Equality_of_areas_l434_434180

variable {α : Type} [LinearOrderedCommRing α] {A B C D P M : EuclideanGeometry.Point α}

-- Define the conditions
axiom quadrilateral (A B C D : EuclideanGeometry.Point α)
axiom diagonals_intersect_at_P (A C B D P : EuclideanGeometry.Point α) : EuclideanGeometry.Intersect (A, C) (B, D) = P
axiom M_midpoint_AC (A C M : EuclideanGeometry.Point α) : EuclideanGeometry.Distance A M = EuclideanGeometry.Distance M C

-- Define the problem for area equality
theorem Equality_of_areas (A B C D P M : EuclideanGeometry.Point α) 
  (h_quad : quadrilateral A B C D)
  (h_diag : diagonals_intersect_at_P A C B D P)
  (h_mid_AM_MC : M_midpoint_AC A C M):
  EuclideanGeometry.Area (EuclideanGeometry.Triangle A B M) 
  + EuclideanGeometry.Area (EuclideanGeometry.Triangle A M D) 
  = EuclideanGeometry.Area (EuclideanGeometry.Triangle B M C) 
  + EuclideanGeometry.Area (EuclideanGeometry.Triangle C M D) := 
sorry

end Equality_of_areas_l434_434180


namespace math_problem_l434_434799

/-- Prove that the expression evaluates to 5/4 -/
theorem math_problem : (-1 : ℝ) ^ 2023 + (π - 1) ^ 0 * (2 / 3) ^ (-2) = 5 / 4 :=
by
  sorry

end math_problem_l434_434799


namespace tickets_used_for_clothes_l434_434350

theorem tickets_used_for_clothes (C : ℕ) (H1 : ∃ (C : ℕ), let T := C + 5 in C + T = 12) : C = 7 :=
by
suffices : ∀ (C T : ℕ), T = C + 5 → C + T = 12 → C = 7 by
  exact Exists.elim H1 this
sorry

end tickets_used_for_clothes_l434_434350


namespace lines_perpendicular_l434_434580

-- Defining the conditions and variables
variables (a b c : ℝ) (A B C : ℝ)
variables (h : b * sin A = a * sin B)

-- Defining the lines
def line1 : linear_ordered_field ℝ :=
  λ x y, x * sin A + a * y + c
def line2 : linear_ordered_field ℝ :=
  λ x y, b * x - y * sin B + sin C

-- The theorem stating the positional relationship
theorem lines_perpendicular : b * sin A = a * sin B →
  ∀ x y, (line1 x y) • (line2 x y) = 0 :=
by
  intro h
  sorry

end lines_perpendicular_l434_434580


namespace circle_diameter_l434_434272

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l434_434272


namespace circle_diameter_l434_434274

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l434_434274


namespace s_2_eq_14_l434_434235

def s (n : ℕ) : ℕ := 
  let perfect_squares := (List.range n).map (λ x => (x + 1) * (x + 1))
  perfect_squares.foldl (λ acc x => acc * 10^ (Nat.log10 (x) + 1) + x) 0

theorem s_2_eq_14 : s 2 = 14 := 
by sorry

end s_2_eq_14_l434_434235


namespace circle_diameter_l434_434297

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l434_434297


namespace circle_diameter_l434_434283

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434283


namespace area_of_triangle_l434_434481

theorem area_of_triangle :
  let f : ℝ → ℝ := λ x, (x - 4) ^ 2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := (0, f 0)
  let base := (x_intercepts[0] - x_intercepts[1]).abs
  let height := y_intercept.2
  1 / 2 * base * height = 168 :=
by
  sorry

end area_of_triangle_l434_434481


namespace range_of_g_l434_434934

noncomputable def g (c a b x : ℝ) := c * x^2 + a * x + b

theorem range_of_g (c a b : ℝ) (h : 0 < c) :
  set.range (g c a b) = set.Icc (b - a^2 / (4 * c)) (b + 2 * a + 4 * c) :=
sorry

end range_of_g_l434_434934


namespace triangle_area_l434_434472

-- Define the given curve
def curve (x : ℝ) : ℝ := (x - 4) ^ 2 * (x + 3)

-- x-intercepts occur when y = 0
def x_intercepts : set ℝ := { x | curve x = 0 }

-- y-intercept occurs when x = 0
def y_intercept : ℝ := curve 0

-- Base of the triangle is the distance between the x-intercepts
def base_of_triangle : ℝ := max (4 : ℝ) (-3) - min (4 : ℝ) (-3)

-- Height of the triangle is the y-intercept value
def height_of_triangle : ℝ := y_intercept

-- Area of the triangle
def area_of_triangle (b h : ℝ) : ℝ := 1 / 2 * b * h

theorem triangle_area : area_of_triangle base_of_triangle height_of_triangle = 168 := by
  -- Stating the problem requires definitions of x-intercepts and y-intercept
  have hx : x_intercepts = {4, -3} := by
    sorry -- The proof for finding x-intercepts

  have hy : y_intercept = 48 := by
    sorry -- The proof for finding y-intercept

  -- Setup base and height using the intercepts
  have b : base_of_triangle = 7 := by
    -- Calculate the base from x_intercepts
    rw [hx]
    exact calc
      4 - (-3) = 4 + 3 := by ring
      ... = 7 := rfl

  have h : height_of_triangle = 48 := by
    -- height_of_triangle should be y_intercept which is 48
    rw [hy]

  -- Finally calculate the area
  have A : area_of_triangle base_of_triangle height_of_triangle = 1 / 2 * 7 * 48 := by
    rw [b, h]

  -- Explicitly calculate the numerical value
  exact calc
    1 / 2 * 7 * 48 = 1 / 2 * 336 := by ring
    ... = 168 := by norm_num

end triangle_area_l434_434472


namespace correct_statements_l434_434425

-- Definitions from the problem
def closed_set (A : Set ℤ) : Prop :=
  ∀ a b ∈ A, a + b ∈ A ∧ a - b ∈ A

-- Closed set examples
def A1 : Set ℤ := {-4, -2, 0, 2, 4}
def A2 : Set ℤ := {n | ∃ k : ℤ, n = 3 * k}

-- Statements to verify
def statement1 := ¬ closed_set A1
def statement2 := closed_set A2
def statement3 := ∀ (A1 A2 : Set ℤ), closed_set A1 → closed_set A2 → ¬ closed_set (A1 ∪ A2)

-- Main theorem combining all results
theorem correct_statements : statement1 ∧ statement2 ∧ statement3 :=
by {
  -- Proof steps can be implemented here to show
  -- each of the statements are correct as described.
  sorry
}

end correct_statements_l434_434425


namespace minimal_overlap_facebook_instagram_l434_434538

variable (P : ℝ → Prop)
variable [Nonempty (Set.Icc 0 1)]

theorem minimal_overlap_facebook_instagram :
  ∀ (f i : ℝ), f = 0.85 → i = 0.75 → ∃ b : ℝ, 0 ≤ b ∧ b ≤ 1 ∧ b = 0.6 :=
by
  intros
  sorry

end minimal_overlap_facebook_instagram_l434_434538


namespace circle_diameter_l434_434273

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l434_434273


namespace largest_k_divide_Q_l434_434132

def factorial_divisibility (n m : ℕ) : ℕ :=
  (list.range n).sum (/ m)

theorem largest_k_divide_Q :
  let Q := (list.range 150).map (λ k, 2 * k + 1) -- The product of the first 150 positive odd integers
  in ∃ k, (∃ q, Q = q * (3 ^ k)) ∧ k = 76 := 
by
  let Q : ℕ := (list.range 150).map (λ k, 2 * k + 1) product;
  have hQ : Q = nat.factorial 300 / (2 ^ 150 * nat.factorial 150) := sorry;
  have h300 : factorial_divisibility 300 3 = 148 := sorry;
  have h150 : factorial_divisibility 150 3 = 72 := sorry;
  use 76;
  split;
  · use (Q / (3 ^ 76));
    have h : 148 - 72 = 76 := by norm_num;
    rw [h];
    sorry;
  · refl

end largest_k_divide_Q_l434_434132


namespace P_work_time_l434_434771

theorem P_work_time (T : ℝ) (hT : T > 0) : 
  (1 / T + 1 / 6 = 1 / 2.4) → T = 4 :=
by
  intros h
  sorry

end P_work_time_l434_434771


namespace fourth_square_has_37_dots_l434_434381

-- Define the sequence of the number of dots in each square.
def dots_in_square (n : ℕ) : ℕ :=
  if n = 1 then 1
  else (dots_in_square (n - 1)) + 4 * n

-- Define the goal to prove that the fourth square has 37 dots.
theorem fourth_square_has_37_dots : dots_in_square 4 = 37 :=
sorry

end fourth_square_has_37_dots_l434_434381


namespace smallest_AAAB_value_l434_434330

theorem smallest_AAAB_value : ∃ (A B : ℕ), A ≠ B ∧ A < 10 ∧ B < 10 ∧ 111 * A + B = 7 * (10 * A + B) ∧ 111 * A + B = 667 :=
by sorry

end smallest_AAAB_value_l434_434330


namespace how_many_pairs_of_shoes_l434_434822

theorem how_many_pairs_of_shoes (l k : ℕ) (h_l : l = 52) (h_k : k = 2) : l / k = 26 := by
  sorry

end how_many_pairs_of_shoes_l434_434822


namespace cost_per_ounce_of_water_l434_434617

theorem cost_per_ounce_of_water :
  ∃ cost_per_ounce : ℝ,
    (∀ (total_weight : ℝ) (cubes_weight : ℝ) (water_per_cube : ℝ) 
       (cubes_per_hour : ℕ) (hourly_cost : ℝ) (total_cost : ℝ),
        total_weight = 10 ∧
        cubes_weight = 1 / 16 ∧
        water_per_cube = 2 ∧
        cubes_per_hour = 10 ∧
        hourly_cost = 1.50 ∧
        total_cost = 56 →
        cost_per_ounce = 0.10) :=
begin
  sorry
end

end cost_per_ounce_of_water_l434_434617


namespace part1_part2_l434_434434

-- Given conditions
variables (a b c : ℝ)
variable h₁ : a > 0
variable h₂ : b > 0
variable h₃ : c > 0
variable h₄ : a + b + c = 1

-- Part (1): Proof of inequality
theorem part1 (h₁ h₂ h₃ h₄) : 
  ( (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ) >= 8 :=
sorry

-- Part (2): Minimum value
theorem part2 (h₁ h₂ h₃ h₄) :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 → (1 / a + 1 / b + 1 / c >= 9) :=
sorry

end part1_part2_l434_434434


namespace three_over_x_solution_l434_434935

theorem three_over_x_solution (x : ℝ) (h : 1 - 9 / x + 9 / (x^2) = 0) :
  3 / x = (3 - Real.sqrt 5) / 2 ∨ 3 / x = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end three_over_x_solution_l434_434935


namespace circle_diameter_problem_circle_diameter_l434_434284

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l434_434284


namespace prob_eliminated_after_semifinal_expected_value_ξ_l434_434949

-- Define the problem conditions
variables (Ω : Type*) [MeasureSpace Ω]
variables (A B C : Event Ω)
variables (P : MeasureTheory.Measure Ω)
variable hA : P(A) = 2/3
variable hB : P(B) = 1/2
variable hC : P(C) = 1/3
variable h_ind : pairwise (independent P)

-- Problem 1: Probability of elimination during semifinal stage is 1/3
theorem prob_eliminated_after_semifinal : P(A ∩ Bᶜ) = 1 / 3 :=
sorry

-- Define the random variable ξ
def ξ : Ω → ℕ
| ω := if ¬A ω then 1 else if ¬B ω then 2 else 3

-- Problem 2: Expected value of ξ is 2
theorem expected_value_ξ : MeasureTheory.ProbabilityTheory.ProbabilityMassFunction.expected_value (MeasureTheory.ProbabilityTheory.ProbabilityMassFunction.from_fun ξ) = 2 :=
sorry

end prob_eliminated_after_semifinal_expected_value_ξ_l434_434949


namespace total_games_played_l434_434567

theorem total_games_played (jerry_wins dave_wins ken_wins : ℕ)
  (h1 : jerry_wins = 7)
  (h2 : dave_wins = jerry_wins + 3)
  (h3 : ken_wins = dave_wins + 5) :
  jerry_wins + dave_wins + ken_wins = 32 :=
by
  sorry

end total_games_played_l434_434567


namespace range_of_m_l434_434453

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - (1/2) * x^2 - 2 * x + 5

-- Define the interval predicate
def interval (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2

-- Statement: For any real number m greater than 7, f(x) < m for all x in the interval [-1, 2].
theorem range_of_m : ∀ m : ℝ, (m > 7) → ∀ x : ℝ, (interval x) → (f x < m) :=
by {
  intro m hm,
  intro x,
  intro hx,
  -- Proof to be completed
  sorry
}

end range_of_m_l434_434453


namespace find_w_l434_434386

open Matrix

-- Define matrix B
def B : Matrix (Fin 2) (Fin 2) ℚ := !![2, 0; 0, 2]

-- Define w
def w := !![17 / 31; 1]

-- Define the expression B^4 + B^3 + B^2 + B + I
def expr := B^4 + B^3 + B^2 + B + 1

-- The theorem to prove
theorem find_w : expr.mul_vec w = !![17; 31] :=
by
  sorry

end find_w_l434_434386


namespace baron_munchausen_failed_l434_434351

theorem baron_munchausen_failed : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → ¬∃ (d1 d2 : ℕ), ∃ (k : ℕ), n * 100 + (d1 * 10 + d2) = k^2 := 
by
  intros n hn
  obtain ⟨h10, h99⟩ := hn
  sorry

end baron_munchausen_failed_l434_434351


namespace fraction_computation_l434_434360

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l434_434360


namespace inequality_l434_434419

theorem inequality {n : ℕ} 
  (x a : Fin n → ℝ) 
  (hx : ∑ i, (x i)^2 = 1) :
  ∑ i, (a i) * (x i) ≤ Real.sqrt (∑ i, (a i)^2) :=
sorry

end inequality_l434_434419


namespace travel_time_CA_l434_434703

variables (v_b v_r : ℝ) -- Boat's speed and river's speed

-- Distance from dock A to the confluence point is 1 km upstream
-- Distance from the confluence point to dock B is 2 km downstream
-- Distance from dock B to dock C is 1 km downstream
-- Distance from dock C to dock A is 3 km upstream

def time_upstream (d : ℝ) : ℝ := d / (v_b - v_r) -- Time to travel d km upstream
def time_downstream (d : ℝ) : ℝ := d / (v_b + v_r) -- Time to travel d km downstream

-- Given conditions
axiom travel_time_AB : 30 = time_upstream 1 + time_downstream 2
axiom travel_time_BC : 18 = time_downstream 1

-- Proposition to prove
theorem travel_time_CA : ∃ t : ℝ, t = 24 ∨ t = 72 :=
by {
  sorry
}

end travel_time_CA_l434_434703


namespace circle_diameter_l434_434294

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l434_434294


namespace find_N_l434_434831

theorem find_N (N : ℕ) (h1 : N > 1)
  (h2 : ∀ (a b c : ℕ), (a ≡ b [MOD N]) ∧ (b ≡ c [MOD N]) ∧ (a ≡ c [MOD N])
  → (a = 1743) ∧ (b = 2019) ∧ (c = 3008)) :
  N = 23 :=
by
  have h3 : Nat.gcd (2019 - 1743) (3008 - 2019) = 23 := sorry
  show N = 23 from sorry

end find_N_l434_434831


namespace central_angle_of_sector_l434_434081

def radius := 8
def slant_height := 15
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def central_angle (C : ℝ) (l : ℝ) : ℝ := (C * 180) / (l * Real.pi)

theorem central_angle_of_sector :
  central_angle (circumference radius) slant_height = 192 := 
by
  sorry

end central_angle_of_sector_l434_434081


namespace range_of_function_l434_434189

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - 4 * Real.sin x + 5

theorem range_of_function : Set.Icc 2 10 = set.range f := by
  sorry

end range_of_function_l434_434189


namespace prime_divides_sum_of_squares_implies_divides_both_l434_434516

theorem prime_divides_sum_of_squares_implies_divides_both (p n a b : ℤ) [hp : Fact (Nat.Prime p)] 
  (h1: p = 4 * n + 3) (h2: p ∣ (a^2 + b^2)) : p ∣ a ∧ p ∣ b :=
by
  sorry

end prime_divides_sum_of_squares_implies_divides_both_l434_434516


namespace josanna_minimum_score_l434_434990

-- Let josanna_scores be Josanna's test scores so far.
def josanna_scores : List ℕ := [95, 88, 82, 76, 91, 84]

-- Define the number of tests she has already taken.
def num_tests_so_far : ℕ := 6

-- Calculate the sum of Josanna's current test scores.
def sum_scores : ℕ := List.sum josanna_scores

-- Define the current average score.
def current_average : ℚ := sum_scores / num_tests_so_far.toRat

-- Define the desired increase in average.
def desired_increase : ℚ := 5

-- Calculate the desired average score.
def target_average : ℚ := current_average + desired_increase

-- Define the total number of tests after taking the seventh test.
def total_tests : ℕ := num_tests_so_far + 1

-- Define the total score needed to reach the target average.
def total_score_needed : ℚ := target_average * total_tests.toRat

-- Define the minimum score Josanna needs on her seventh test.
def min_seventh_test_score : ℚ := total_score_needed - sum_scores

theorem josanna_minimum_score : min_seventh_test_score = 121 := by
  have sum_eq : sum_scores = 516 := by
    simp [sum_scores, josanna_scores]
  have current_avg_eq : current_average = (516 : ℚ) / 6 := by
    rw [sum_eq]
    simp [current_average, num_tests_so_far]
  have target_avg_eq : target_average = (86 : ℚ) + 5 := by
    rw [current_avg_eq]
    simp [target_average, current_average, desired_increase]
  have total_score_eq : total_score_needed = (91 : ℚ) * 7 := by
    rw [target_avg_eq]
    simp [total_score_needed, target_average, total_tests]
  have min_score_eq : min_seventh_test_score = (637 : ℚ) - 516 := by
    rw [total_score_eq, sum_eq]
    simp [min_seventh_test_score, total_score_needed, sum_scores]
  norm_num at *
  exact min_score_eq

end josanna_minimum_score_l434_434990


namespace points_on_line_l434_434070

theorem points_on_line
  (a b c : ℝ)
  (M_on_line : a + (1 / b) = 1)
  (N_on_line : b + (1 / c) = 1)
  (l : ℝ → ℝ → Prop := λ x y, x + y = 1) :
  l c (1 / a) ∧ l (1 / c) b :=
by
  sorry

end points_on_line_l434_434070


namespace ellipse_range_l434_434042

theorem ellipse_range (t : ℝ) (x y : ℝ) :
  (10 - t > 0) → (t - 4 > 0) → (10 - t ≠ t - 4) →
  (t ∈ (Set.Ioo 4 7 ∪ Set.Ioo 7 10)) :=
by
  intros h1 h2 h3
  sorry

end ellipse_range_l434_434042


namespace range_of_a_decreasing_function_l434_434182

theorem range_of_a_decreasing_function (a : ℝ) 
  (h : ∀ x y : ℝ, x ≤ y → y ≤ 4 → f y ≤ f x) : a ≤ -3 :=
by
  let f := λ x, x^2 + 2 * (a - 1) * x + 2
  have : 1 - a ≤ 4 :=
    begin
      sorry -- The detailed proof is omitted
    end
  exact this

end range_of_a_decreasing_function_l434_434182


namespace problem_statement_l434_434397

theorem problem_statement : (4^4 / 4^3) * 2^8 = 1024 := by
  sorry

end problem_statement_l434_434397


namespace fraction_computation_l434_434359

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l434_434359


namespace sequence_2004_l434_434694

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else let prev := sequence (n - 1) in
       Nat.find (λ m, m > prev ∧ ∀ i j k, i, j, k ≤ n → sequence i + sequence j ≠ 3 * sequence k)

theorem sequence_2004 : sequence 2004 = 3006 :=
by
  sorry

end sequence_2004_l434_434694


namespace find_a6_a7_l434_434103

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- Given Conditions
axiom cond1 : arithmetic_sequence a d
axiom cond2 : a 2 + a 4 + a 9 + a 11 = 32

-- Proof Problem
theorem find_a6_a7 : a 6 + a 7 = 16 :=
  sorry

end find_a6_a7_l434_434103


namespace maximum_lanterns_l434_434249

-- Define the road length and lantern segment length in meters.
def road_length : ℕ := 1000
def segment_length : ℕ := 1

-- Define the maximum number of lanterns such that the road is fully illuminated 
-- but if any one lantern is turned off, the road will no longer be fully illuminated.
theorem maximum_lanterns : 
  ∃ n : ℕ, n = 1998 ∧ 
  ∀ (lanterns : fin n → ℕ), (∀ i, lanterns i < road_length) ∧ 
  (∀ i j, i ≠ j → abs (lanterns i - lanterns j) ≥ segment_length) → 
  (∀ k, ∃ (lantern : ℕ), lantern < road_length ∧ (k ≠ ∧ lanterns k = lantern) →
  (¬ ∃ m : ℕ, ∃ (lanterns' : fin m → ℕ), ∀ i, (lanterns' i < road_length) ∧ 
  (∀ i j, i ≠ j → abs(lanterns' i - lanterns' j) ≥ segment_length)) := 
by 
  sorry

end maximum_lanterns_l434_434249


namespace shoes_selection_l434_434204

theorem shoes_selection : 
  let num_pairs := 10 in 
  let total_ways := (num_pairs.choose 1) * (9.choose 2) * 4 in
  total_ways = 1440 :=
by
  sorry

end shoes_selection_l434_434204


namespace area_of_triangle_l434_434506

-- Define the given curve equation
def curve (x : ℝ) := (x - 4)^2 * (x + 3)

-- The x and y intercepts of the curve, (x, y at x = 0)
def x_intercepts : set ℝ := {x | curve x = 0}
def y_intercept := curve 0

-- Vertices of the triangle
def vertex_A := (-3 : ℝ, 0 : ℝ)
def vertex_B := (4 : ℝ, 0 : ℝ)
def vertex_C := (0 : ℝ, y_intercept)

-- Calculate the base and height of the triangle
def base := 7
def height := y_intercept

-- The area of the triangle
def triangle_area := 1/2 * base * height

-- The theorem to prove
theorem area_of_triangle : triangle_area = 168 := by
  sorry

end area_of_triangle_l434_434506


namespace linear_regression_equation_correct_l434_434899

theorem linear_regression_equation_correct {x y : ℝ} 
    (negatively_correlated : negatively_corr x y) 
    (mean_x : ℝ) (mean_y : ℝ) 
    (hx : mean_x = 4) 
    (hy : mean_y = 6.5) : 
    y = -2 * x + 14.5 :=
by
  sorry

end linear_regression_equation_correct_l434_434899


namespace parabola_trajectory_fixed_point_pq_minimum_area_triangle_l434_434424

-- Definitions for the conditions and objects in the problem

def point (α : Type) [LinearOrderedField α] := (α × α)

variable {α : Type} [LinearOrderedField α]

def distance (M F : point α) : α :=
  real.sqrt ((M.1 - F.1) ^ 2 + (M.2 - F.2) ^ 2)

def line (α : Type) [LinearOrderedField α] := (α × α) -> Prop

def parabola_trajectory_equation : Prop :=
  ∀ (M : point α), distance M (1, 0) < distance M (-2, M.2) + 1 ↔ M.2^2 = 4 * M.1

def line_through_fixed_point (M N A B : point α) : Prop :=
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let Q := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  ∀ F : point α, (P, Q).snd - F.1 = 3

def minimum_triangle_area (k : α) : Prop :=
  ∀ k ≠ 0, ∀ F : point α, triangle_area F (1 + 2 / k ^ 2, 2 / k) (1 + 2 * k ^ 2, -2 * k) >= 4

-- Lean 4 statements

theorem parabola_trajectory :
  parabola_trajectory_equation := by
  sorry

theorem fixed_point_pq :
  ∀ (F E M N A B : point α), line_through_fixed_point M N A B := by
  sorry

theorem minimum_area_triangle :
  minimum_triangle_area := by
  sorry

end parabola_trajectory_fixed_point_pq_minimum_area_triangle_l434_434424


namespace domain_and_period_of_f_max_min_of_g_on_interval_l434_434048

def domain_of_f (x : ℝ) : Prop :=
  ∃ k : ℤ, x = 2 * k * π + π / 2

def f (x : ℝ) := 2 * real.sqrt 3 * real.tan (x / 2 + π / 4) * (real.cos (x / 2 + π / 4))^2 - real.sin (x + π)

def g (x : ℝ) := 2 * real.sin (x + π / 6)

theorem domain_and_period_of_f :
  (∀ x, ¬ domain_of_f x → x ∈ set.univ) ∧
  (∃ T > 0, ∀ x, f (x + T) = f x) :=
  sorry

theorem max_min_of_g_on_interval :
  ∀ x ∈ set.Icc 0 π, 
  ∃! M m, g M = 2 ∧ g m = -1 :=
  sorry

end domain_and_period_of_f_max_min_of_g_on_interval_l434_434048


namespace parabola_points_on_circle_l434_434685

noncomputable def parabola := λ x : ℝ, x^2 - 2 * x - 3

def point_A := (-1 : ℝ, 0 : ℝ)
def point_B := (3 : ℝ, 0 : ℝ)
def point_C := (0 : ℝ, -3 : ℝ)
def circle_eq := λ x y : ℝ, (x - 1)^2 + (y + 1)^2 = 5

theorem parabola_points_on_circle :
  circle_eq point_A.1 point_A.2 ∧
  circle_eq point_B.1 point_B.2 ∧
  circle_eq point_C.1 point_C.2 :=
by
  sorry

end parabola_points_on_circle_l434_434685


namespace sum_binom_99_2k_l434_434852

theorem sum_binom_99_2k :
  (∑ k in Finset.range 50, (-1 : ℤ) ^ k * Nat.choose 99 (2 * k)) = -2 ^ 49 :=
by
  sorry

end sum_binom_99_2k_l434_434852


namespace prime_divides_sum_diff_l434_434130

theorem prime_divides_sum_diff
  (a b c p : ℕ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hp : p.Prime) 
  (h1 : p ∣ (100 * a + 10 * b + c)) 
  (h2 : p ∣ (100 * c + 10 * b + a)) 
  : p ∣ (a + b + c) ∨ p ∣ (a - b + c) ∨ p ∣ (a - c) :=
by
  sorry

end prime_divides_sum_diff_l434_434130


namespace collinear_H_M_N_l434_434159

variables {A B C : Point}
variable {H : Point}

/-- Constructing midpoints -/
noncomputable def A1 : Point := midpoint B C
noncomputable def B1 : Point := midpoint A C
noncomputable def M : Point := midpoint A1 B1

/-- Foot of the perpendicular from C onto AB -/
noncomputable def H : Point := foot_perpendicular C A B

/-- Defining the circles and their intersection points -/
noncomputable def circle_through_M_tangent_to_BC : Circle := circle M A1
noncomputable def circle_through_M_tangent_to_AC : Circle := circle M B1
noncomputable def N : Point := (circle_through_M_tangent_to_BC ∩ circle_through_M_tangent_to_AC).second 

/-- The theorem to be proved -/
theorem collinear_H_M_N 
  (hA1 : midpoint B C = A1)
  (hB1 : midpoint A C = B1)
  (hM : midpoint A1 B1 = M)
  (hH : foot_perpendicular C A B = H)
  (hN : (circle M A1) ∩ (circle M B1) = {M, N}) :
  collinear {H, M, N} :=
sorry

end collinear_H_M_N_l434_434159


namespace books_finished_when_reached_japan_l434_434149

-- Define the conditions as constants
def total_miles_traveled : ℝ := 6987.5
def miles_per_book : ℝ := 482.3

-- Define the statement that needs to be proved
theorem books_finished_when_reached_japan : 
  (total_miles_traveled / miles_per_book).floor = 14 :=
by {
  sorry
}

end books_finished_when_reached_japan_l434_434149


namespace sequence_a_general_formula_l434_434143

def a (n : ℕ) : ℂ := 
  if n = 0 then 1 
  else if n % 4 == 0 then 1 
  else if n % 4 == 2 then -1
  else 0

theorem sequence_a_general_formula (n : ℕ) (hn : n > 0) :
  1 + ∑ m in Finset.range(n+1), (Nat.choose n m * a(m)) = 2^(n / 2) * Complex.cos (n * Real.pi / 4) :=
sorry

end sequence_a_general_formula_l434_434143


namespace remarkable_number_is_golden_ratio_or_zero_l434_434248

def is_geometric_progression (x : ℝ) : Prop :=
  let int_part := x.floor
  let frac_part := x - int_part
  x = int_part * frac_part * frac_part

theorem remarkable_number_is_golden_ratio_or_zero (x : ℝ) :
  is_geometric_progression x → x = 0 ∨ x = (1 + Real.sqrt 5) / 2 :=
begin
  sorry
end

end remarkable_number_is_golden_ratio_or_zero_l434_434248


namespace area_of_new_face_l434_434762

-- Definitions based on conditions
def radius : ℝ := 8 -- Cylinder radius
def height : ℝ := 10 -- Cylinder height
def angle_subtended_degrees : ℝ := 90 -- Arc subtends an angle of 90 degrees

-- Auxiliary definitions
def arc_length (r : ℝ) (θ : ℝ) : ℝ := 2 * r * Real.sin(θ / 2)
def area_of_triangle (base height : ℝ) : ℝ := 0.5 * base * height

-- Problem statement
theorem area_of_new_face :
  let PQ := 2 * radius * Real.sin(angle_subtended_degrees.toRadians / 2),
    triangle_area := area_of_triangle PQ radius in
  triangle_area = 32 * Real.sqrt 2 →
  ∃ d e f : ℤ, d = 0 ∧ e = 32 ∧ f = 2 ∧ d + e + f = 34 :=
by
  sorry

end area_of_new_face_l434_434762


namespace find_N_l434_434830

theorem find_N (N : ℕ) (h1 : N > 1)
  (h2 : ∀ (a b c : ℕ), (a ≡ b [MOD N]) ∧ (b ≡ c [MOD N]) ∧ (a ≡ c [MOD N])
  → (a = 1743) ∧ (b = 2019) ∧ (c = 3008)) :
  N = 23 :=
by
  have h3 : Nat.gcd (2019 - 1743) (3008 - 2019) = 23 := sorry
  show N = 23 from sorry

end find_N_l434_434830


namespace smallest_integer_remainder_l434_434725

theorem smallest_integer_remainder (b : ℕ) :
  (b ≡ 2 [MOD 3]) ∧ (b ≡ 3 [MOD 5]) → b = 8 := 
by
  sorry

end smallest_integer_remainder_l434_434725


namespace problem1_l434_434752

theorem problem1 (α : ℝ) (h : sin (α - 3 * π) = 2 * cos (α - 3 * π)) :
  (sin (π - α) + 5 * cos (2 * π - α)) / (2 * sin (3 * π / 2 - α) - sin (-α)) = -3 / 4 :=
sorry

end problem1_l434_434752


namespace root_not_integer_then_not_fraction_l434_434637

theorem root_not_integer_then_not_fraction (A : ℤ) (k : ℤ) (hA : 0 < A) (hk : 0 < k) 
  (h_not_int : sqrt_real (A : ℝ) (k : ℝ) ∉ ℤ) : ∀ (n p : ℤ), gcd n p = 1 → p > 1 → sqrt_real (A : ℝ) (k : ℝ) ≠ n / p :=
by
  sorry

end root_not_integer_then_not_fraction_l434_434637


namespace set_complement_intersection_l434_434055

theorem set_complement_intersection
  (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)
  (hU : U = {0, 1, 2, 3, 4})
  (hM : M = {0, 1, 2})
  (hN : N = {2, 3}) :
  ((U \ M) ∩ N) = {3} :=
  by sorry

end set_complement_intersection_l434_434055


namespace problem_inequality_l434_434647

theorem problem_inequality (x y z : ℝ) (h1 : x + y + z = 0) (h2 : |x| + |y| + |z| ≤ 1) :
  x + (y / 2) + (z / 3) ≤ 1 / 3 :=
sorry

end problem_inequality_l434_434647


namespace hyperbola_sum_l434_434807

theorem hyperbola_sum
  (h k a b : ℝ)
  (center : h = 3 ∧ k = 1)
  (vertex : ∃ (v : ℝ), (v = 4 ∧ h = 3 ∧ a = |k - v|))
  (focus : ∃ (f : ℝ), (f = 10 ∧ h = 3 ∧ (f - k) = 9 ∧ ∃ (c : ℝ), c = |k - f|))
  (relationship : ∀ (c : ℝ), c = 9 → c^2 = a^2 + b^2): 
  h + k + a + b = 7 + 6 * Real.sqrt 2 :=
by 
  sorry

end hyperbola_sum_l434_434807


namespace ratio_sum_is_four_l434_434588

theorem ratio_sum_is_four
  (x y : ℝ)
  (hx : 0 < x) (hy : 0 < y)
  (θ : ℝ)
  (hθ_ne : ∀ n : ℤ, θ ≠ (n * (π / 2)))
  (h1 : (Real.sin θ) / x = (Real.cos θ) / y)
  (h2 : (Real.cos θ)^4 / x^4 + (Real.sin θ)^4 / y^4 = 97 * (Real.sin (2 * θ)) / (x^3 * y + y^3 * x)) :
  (x / y) + (y / x) = 4 := by
  sorry

end ratio_sum_is_four_l434_434588


namespace real_solution_eq_one_l434_434841

theorem real_solution_eq_one :
  ∃! x : ℝ, x ≠ 0 ∧ (x^1002 + 1) * (x^1000 + x^998 + x^996 + x^994 + ... + x^2 + 1) = 1002 * x^1001 :=
sorry

end real_solution_eq_one_l434_434841


namespace maximum_ab_l434_434579

open Real

theorem maximum_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 6 * a + 5 * b = 75) :
  ab ≤ 46.875 :=
by
  -- proof goes here
  sorry

end maximum_ab_l434_434579


namespace initial_notebooks_is_10_l434_434337

-- Define the conditions
def ordered_notebooks := 6
def lost_notebooks := 2
def current_notebooks := 14

-- Define the initial number of notebooks
def initial_notebooks (N : ℕ) :=
  N + ordered_notebooks - lost_notebooks = current_notebooks

-- The proof statement
theorem initial_notebooks_is_10 : initial_notebooks 10 :=
by
  sorry

end initial_notebooks_is_10_l434_434337


namespace solution_correct_l434_434691

theorem solution_correct :
  ∃ (x y z : ℝ), 2 * sqrt (x - 4) + 3 * sqrt (y - 9) + 4 * sqrt (z - 16) = 1/2 * (x + y + z)
  ∧ x = 8 ∧ y = 18 ∧ z = 32 := by
  use 8, 18, 32
  sorry

end solution_correct_l434_434691


namespace friend_spent_more_l434_434230

/-- Given that the total amount spent for lunch is $15 and your friend spent $8 on their lunch,
we need to prove that your friend spent $1 more than you did. -/
theorem friend_spent_more (total_spent friend_spent : ℤ) (h1 : total_spent = 15) (h2 : friend_spent = 8) :
  friend_spent - (total_spent - friend_spent) = 1 :=
by
  sorry

end friend_spent_more_l434_434230


namespace probability_second_roll_twice_first_l434_434763

theorem probability_second_roll_twice_first :
  let outcomes := [(1, 2), (2, 4), (3, 6)]
  let total_outcomes := 36
  let favorable_outcomes := 3
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 12 :=
by
  sorry

end probability_second_roll_twice_first_l434_434763


namespace start_A_can_give_C_l434_434951

theorem start_A_can_give_C :
  (∀ d, (d : ℝ) / 1000 * 1000 = d) → -- 1 km = 1000 meters
  (∀ a b d, (a / 1000) * 1000 = 1000 ∧ (b / 1000) * 950 = b) → -- A runs 1000 meters when B runs 950 meters
  (∀ b c d, (b / 1000) * 1000 = 1000 ∧ (c / 1000) * 947.3684210526316 = c) → -- B runs 1000 meters when C runs 947.3684210526316 meters
  (∀ a c, a / 1000 * (c / 1000) * 950 = 900) → -- Transformation property to correlate distances
  true := -- Conclusion
sorry

end start_A_can_give_C_l434_434951


namespace minimum_arg_z_l434_434038

open Complex Real

noncomputable def z_cond (z : ℂ) := abs (z + 3 - (complex.I * sqrt 3)) = sqrt 3

theorem minimum_arg_z : ∀ z : ℂ, z_cond z → arg z = 5 / 6 * π :=
by
  intros
  sorry

end minimum_arg_z_l434_434038


namespace range_of_a_l434_434000

open Set Real

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2x + a ≥ 0}

theorem range_of_a (a : ℝ) : (A ∩ B a).Nonempty ↔ a ≥ -8 :=
by
  sorry

end range_of_a_l434_434000


namespace tenth_number_in_sixteenth_group_is_257_l434_434054

-- Define the general term of the sequence a_n = 2n - 3.
def a_n (n : ℕ) : ℕ := 2 * n - 3

-- Define the first number of the n-th group.
def first_number_of_group (n : ℕ) : ℕ := n^2 - n - 1

-- Define the m-th number in the n-th group.
def group_n_m (n m : ℕ) : ℕ := first_number_of_group n + (m - 1) * 2

theorem tenth_number_in_sixteenth_group_is_257 : group_n_m 16 10 = 257 := by
  sorry

end tenth_number_in_sixteenth_group_is_257_l434_434054


namespace line_segment_intersection_range_l434_434870

theorem line_segment_intersection_range (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (2, 1)) :
  ∃ k : ℝ, (1 / 2 ≤ k ∧ k ≤ 1) ∧ ∀ k, (1 / 2 ≤ k ∧ k ≤ 1) → 
  let y := λ x : ℝ, k * x + 1 in
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (t * fst B + (1 - t) * fst A, t * snd B + (1 - t) * snd A) = (fst A, y (fst A)) ∨ (fst B, y (fst B))) :=
sorry

end line_segment_intersection_range_l434_434870


namespace largest_possible_sin_a_l434_434590

theorem largest_possible_sin_a (a b c : ℝ) (h1 : cos a = tan b) (h2 : cos b = tan c) (h3 : cos c = tan a) :
  sin a <= (√5 - 1) / 2 := sorry

end largest_possible_sin_a_l434_434590


namespace sum_of_diagonals_l434_434306

theorem sum_of_diagonals (A B C D E F : Point) 
  (h : hexagon_inscribed_in_circle A B C D E F)
  (hAB : length A B = 50)
  (hBC : length B C = 100)
  (hCD : length C D = 100)
  (hDE : length D E = 100)
  (hEF : length E F = 100)
  (hFA : length F A = 100)
  (x : length A C)
  (y : length A D)
  (z : length A E) :
  x + y + z = 485.43 := by
  -- Ptolemy's theorem application:
  have ptolemy1 : x * z = 100 * y + 5000 := sorry,
  have ptolemy2 : x * z + 10000 = y^2 := sorry,
  -- Solve for x, y, z
  sorry

end sum_of_diagonals_l434_434306


namespace sin_intersection_ratios_l434_434390

theorem sin_intersection_ratios :
  ∃ (p q : ℕ), gcd p q = 1 ∧ 
  ((set.range (λ k : ℕ, if k % 2 = 0 then 60 else 120) ∩ set.Icc 0 360).to_finset 
  = {60, 120, 240, 300}) ∧ p / gcd p q = 1 ∧ q / gcd p q = 2 :=
sorry

end sin_intersection_ratios_l434_434390


namespace triangle_area_l434_434501

-- Define the curve and points
noncomputable def curve (x : ℝ) : ℝ := (x-4)^2 * (x+3)

-- Define the x-intercepts
def x_intercept1 := 4
def x_intercept2 := -3

-- Define the y-intercept
def y_intercept := curve 0

-- Define the base and height of the triangle
def base : ℝ := x_intercept1 - x_intercept2
def height : ℝ := y_intercept

-- Statement of the problem: calculating the area of the triangle
theorem triangle_area : (1/2) * base * height = 168 := by
  sorry

end triangle_area_l434_434501


namespace work_problem_l434_434207

theorem work_problem (W : ℕ) (h1: ∀ w, w = W → (24 * w + 1 = 73)) : W = 3 :=
by {
  -- Insert proof here
  sorry
}

end work_problem_l434_434207


namespace construct_equal_parallel_segment_l434_434216

noncomputable def construct_segment (S1 S2 : Circle) (MN : LineSegment) : Prop :=
∃ (A B : Point), 
  A ∈ S1 ∧ B ∈ S2 ∧ 
  (LineSegment.mk A B).length = MN.length ∧ 
  (LineSegment.mk A B).is_parallel_to(MN)

-- The main goal is to prove or assert the existence of such points A and B
theorem construct_equal_parallel_segment {S1 S2 : Circle} {MN : LineSegment} :
  construct_segment S1 S2 MN :=
sorry

end construct_equal_parallel_segment_l434_434216


namespace circle_properties_l434_434442

-- Define the circle structure
structure Circle where
  center : Point
  radius : ℝ

-- Define a Point
structure Point where
  x : ℝ
  y : ℝ

-- Given conditions
def center_C : Point := {x := 1, y := 1}
def line_l (x y : ℝ) : Prop := x + y = 1
def chord_length : ℝ := Real.sqrt 2
def point_P : Point := {x := 2, y := 3}

-- Correct answers
def equation_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def tangent_line1 (x y : ℝ) : Prop := x = 2
def tangent_line2 (x y : ℝ) : Prop := 3 * x - 4 * y + 6 = 0

-- Proof problem
theorem circle_properties :
  (∃ (C : Circle), C.center = center_C ∧
                   (∀ (x y : ℝ), (line_l x y) → chord_length = Real.sqrt (2) →
                                 equation_circle x y) ∧
                   (∀ (P : Point), P = point_P →
                                   (∃ (l1 l2 : ℝ → ℝ), 
                                     (∀ (x y : ℝ), tangent_line1 x y ∨ tangent_line2 x y)))) :=
by
  sorry

end circle_properties_l434_434442


namespace age_of_youngest_boy_l434_434177

theorem age_of_youngest_boy (avg_age : ℝ) (ages_in_proportion : ℝ) (h_avg : avg_age = 150) (h_prop : ages_in_proportion = 3/28) : 
  (3 * (750 / 28)).round = 80 :=
by
  sorry

end age_of_youngest_boy_l434_434177


namespace markers_per_box_l434_434148

theorem markers_per_box (original_markers new_boxes total_markers : ℕ) 
    (h1 : original_markers = 32) (h2 : new_boxes = 6) (h3 : total_markers = 86) : 
    total_markers - original_markers = new_boxes * 9 :=
by sorry

end markers_per_box_l434_434148


namespace problem_solution_l434_434415

theorem problem_solution (n : ℕ) (x : Fin n → ℝ) 
    (h1 : ∑ i, x i = 3) 
    (h2 : ∑ i, (x i)⁻¹ = 3) : 
  n ≤ 3 := 
sorry

end problem_solution_l434_434415


namespace standard_deviation_of_data_l434_434880

theorem standard_deviation_of_data (a : ℝ) 
  (h_avg : (1 + 2 + a + 6) / 4 = 3) : 
  stddev [1, 2, a, 6] = real.sqrt (7 / 2) :=
by
  sorry

end standard_deviation_of_data_l434_434880


namespace triangle_area_l434_434496

-- Define the curve and points
noncomputable def curve (x : ℝ) : ℝ := (x-4)^2 * (x+3)

-- Define the x-intercepts
def x_intercept1 := 4
def x_intercept2 := -3

-- Define the y-intercept
def y_intercept := curve 0

-- Define the base and height of the triangle
def base : ℝ := x_intercept1 - x_intercept2
def height : ℝ := y_intercept

-- Statement of the problem: calculating the area of the triangle
theorem triangle_area : (1/2) * base * height = 168 := by
  sorry

end triangle_area_l434_434496


namespace vector_addition_proof_l434_434850

variables {Point : Type} [AddCommGroup Point]

variables (A B C D : Point)

theorem vector_addition_proof :
  (D - A) + (C - D) - (C - B) = B - A :=
by
  sorry

end vector_addition_proof_l434_434850


namespace greatest_divisor_l434_434405

theorem greatest_divisor (d : ℕ) (h₀ : 1657 % d = 6) (h₁ : 2037 % d = 5) : d = 127 :=
by
  -- Proof skipped
  sorry

end greatest_divisor_l434_434405


namespace circle_diameter_l434_434257

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434257


namespace theater_loss_l434_434317

-- Define the conditions
def theater_capacity : Nat := 50
def ticket_price : Nat := 8
def tickets_sold : Nat := 24

-- Define the maximum revenue and actual revenue
def max_revenue : Nat := theater_capacity * ticket_price
def actual_revenue : Nat := tickets_sold * ticket_price

-- Define the money lost by not selling out
def money_lost : Nat := max_revenue - actual_revenue

-- Theorem statement to prove
theorem theater_loss : money_lost = 208 := by
  sorry

end theater_loss_l434_434317


namespace sin_alpha_beta_value_l434_434019

theorem sin_alpha_beta_value (α β : ℝ) (h1 : 13 * Real.sin α + 5 * Real.cos β = 9) (h2 : 13 * Real.cos α + 5 * Real.sin β = 15) : 
  Real.sin (α + β) = 56 / 65 :=
by
  sorry

end sin_alpha_beta_value_l434_434019


namespace cat_toy_cost_l434_434563

-- Define the conditions
def cost_of_cage : ℝ := 11.73
def total_cost_of_purchases : ℝ := 21.95

-- Define the proof statement
theorem cat_toy_cost : (total_cost_of_purchases - cost_of_cage) = 10.22 := by
  sorry

end cat_toy_cost_l434_434563


namespace sum_of_consecutive_integers_l434_434840

theorem sum_of_consecutive_integers (a : ℤ) (h₁ : a = 18) (h₂ : a + 1 = 19) (h₃ : a + 2 = 20) : a + (a + 1) + (a + 2) = 57 :=
by
  -- Add a sorry to focus on creating the statement successfully
  sorry

end sum_of_consecutive_integers_l434_434840


namespace divide_triangle_into_two_equal_areas_l434_434955

-- Definitions
variable {A B C O H H₁ M_H N_H : Point} 

-- Assumptions
variable [triangle : IsAcuteTriangle A B C] 
variable [circumcenter : IsCircumcenter O A B C]
variable [footH₁ : FootFromAPerpendicularToBC H₁ A B C]
variable [projectionM_H : ProjectionOfH₁OnAC M_H H₁ A C]
variable [projectionN_H : ProjectionOfH₁OnAB N_H H₁ A B]

-- Theorem Statement
theorem divide_triangle_into_two_equal_areas :
  let polyline := [M_H, O, N_H] in
  dividesIntoTwoEqualAreas polyline A B C :=
sorry

end divide_triangle_into_two_equal_areas_l434_434955


namespace least_number_to_add_1055_to_div_by_23_l434_434236

theorem least_number_to_add_1055_to_div_by_23 : ∃ k : ℕ, (1055 + k) % 23 = 0 ∧ k = 3 :=
by
  sorry

end least_number_to_add_1055_to_div_by_23_l434_434236


namespace problem1_problem2_area_l434_434025

variables {A B C D E F : Type} [EuclideanGeometry A B C D E F]
variables {AB BC AC AD DE AF CF AE BE : ℝ}

-- The given conditions
variable (h1 : altitude AD (triangle A B C))
variable (h2 : perpendicular DE AB E)
variable (h3 : extension_point F ED)
variable (h4 : perpendicular AF CF)

-- Proof that \frac{AF^2}{CF^2} = \frac{AE}{BE}
theorem problem1 (h1 : altitude AD (triangle A B C))
               (h2 : perpendicular DE AB E)
               (h3 : extension_point F ED)
               (h4 : perpendicular AF CF) :
               \(\frac{AF^2}{CF^2} = \frac{AE}{BE} \) := 
sorry

-- Given values
variables (AB_13 BC_14 AC_15 : ℝ)
variable (h5 : AB = 13)
variable (h6 : BC = 14)
variable (h7 : AC = 15)

-- Proof that the area of quadrilateral \(ACFD\) is \(\frac{10206}{169}\)
theorem problem2_area (h1 : altitude AD (triangle A B C))
                      (h2 : perpendicular DE AB E)
                      (h3 : extension_point F ED)
                      (h4 : perpendicular AF CF)
                      (h5 : AB = 13)
                      (h6 : BC = 14)
                      (h7 : AC = 15) :
                      area_quadrilateral ACFD = \(\frac{10206}{169}\) := 
sorry

end problem1_problem2_area_l434_434025


namespace family_weight_gain_l434_434790

def orlando := 5
def jose := 2 * orlando + 2
def fernando := (1 / 2) * jose - 3
def maria := (orlando + jose + fernando) / 3 + 1
def laura := 0.75 * (jose + fernando)
def total_weight := orlando + jose + fernando + maria + laura

theorem family_weight_gain : total_weight = 38.92 :=
sorry

end family_weight_gain_l434_434790


namespace equidistant_trajectory_l434_434667

theorem equidistant_trajectory (x y : ℝ) (h : abs x = abs y) : y^2 = x^2 :=
by
  sorry

end equidistant_trajectory_l434_434667


namespace circle_diameter_l434_434269

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l434_434269


namespace initial_people_on_train_l434_434210

theorem initial_people_on_train (remaining after_stop getting_off : ℕ)
    (h₁ : remaining = 31) 
    (h₂ : getting_off = 17) : 
    remaining + getting_off = 48 :=
by
  rw [h₁, h₂]
  simp
  sorry

end initial_people_on_train_l434_434210


namespace value_of_xyz_l434_434432

variable (x y z : ℝ)

theorem value_of_xyz (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
                     (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 22) 
                     : x * y * z = 14 / 3 := 
sorry

end value_of_xyz_l434_434432


namespace solution_set_inequality_l434_434192

theorem solution_set_inequality (x : ℝ) : (| 2 * x - 1 | - | x - 2 | < 0) ↔ (-1 < x ∧ x < 1) := 
sorry

end solution_set_inequality_l434_434192


namespace sets_equal_find_a2003_plus_b2004_l434_434776

theorem sets_equal_find_a2003_plus_b2004 
  (a b : ℝ) 
  (h1 : a ≠ 0)
  (h2 : {a, b / a, 1} = {a^2, a + b, 0}) : 
  a^2003 + b^2004 = -1 := 
by 
  sorry

end sets_equal_find_a2003_plus_b2004_l434_434776


namespace museum_rid_paintings_l434_434769

def initial_paintings : ℕ := 1795
def leftover_paintings : ℕ := 1322

theorem museum_rid_paintings : initial_paintings - leftover_paintings = 473 := by
  sorry

end museum_rid_paintings_l434_434769


namespace shaded_area_fraction_l434_434554

/-- Define triangle ABC where AB = AC and AG is perpendicular to BC --/
def isosceles_triangle (A B C G D F E : Point) : Prop :=
  isosceles ABC ∧
  perpendicular AG BC ∧
  midpoint D AB ∧ midpoint F AC ∧
  intersects E DF AG

/-- Prove that the shaded area in the triangle ABC given the conditions is 1/8 the area of the triangle --/
theorem shaded_area_fraction
  {A B C G D F E : Point}
  (h_triangle : isosceles_triangle A B C G D F E) :
  ∃ (shaded_area_fraction : ℚ), shaded_area_fraction = 1/8 :=
sorry

end shaded_area_fraction_l434_434554


namespace graph_transformation_l434_434711

theorem graph_transformation : 
  ∀ (y : ℝ → ℝ), 
    (y = λ x, log 2 (x : ℝ)) → 
    ∀ (y' : ℝ → ℝ), 
      (y' = λ x, log 2 (sqrt (x - 1))) → 
      (∀ x, y' x = 1 / 2 * y (x - 1) + 0.5 * log 2 1) :=
sorry

end graph_transformation_l434_434711


namespace solution_set_of_inequality_l434_434023

variable {R : Type} [LinearOrderedField R]

def is_even (f : R → R) : Prop :=
  ∀ x, f x = f (-x)

def is_monotone_dec (f : R → R) : Prop :=
  ∀ x y, x ≤ y → f y ≤ f x

theorem solution_set_of_inequality {f : R → R}
  (h_even : is_even f)
  (h_mono_dec : is_monotone_dec f)
  (h_f_R : ∀ x, f x ∈ set.univ):
    {x : R | f ((λ x, real.log x) x) > f (-2)} = 
    set.union {x : R | 0 < x ∧ x < (1 : R) / 100} 
              {x : R | x > 100} := by
  sorry

end solution_set_of_inequality_l434_434023


namespace smallest_integer_remainder_l434_434727

theorem smallest_integer_remainder (b : ℕ) :
  (b ≡ 2 [MOD 3]) ∧ (b ≡ 3 [MOD 5]) → b = 8 := 
by
  sorry

end smallest_integer_remainder_l434_434727


namespace exists_hamiltonian_path_l434_434540

noncomputable def complete_directed_graph (V : Type*) (E : set (V × V)) :=
  ∀ u v : V, u ≠ v → (u, v) ∈ E ∨ (v, u) ∈ E

def hamiltonian_path (V : Type*) (E : set (V × V)) (p : list V) :=
  ∀ u, u ∈ p → u ∈ V ∧ ∀ i, i < list.length p - 1 → ((p.nth i).get_or_else u, (p.nth (i + 1)).get_or_else u) ∈ E

theorem exists_hamiltonian_path 
  (V : Type*)
  (E : set (V × V))
  [finite V]
  [inhabited V]
  (h : complete_directed_graph V E) :
  ∃ p : list V, hamiltonian_path V E p :=
sorry

end exists_hamiltonian_path_l434_434540


namespace theater_loss_l434_434312

/-- 
A movie theater has a total capacity of 50 people and charges $8 per ticket.
On a Tuesday night, they only sold 24 tickets. 
Prove that the revenue lost by not selling out is $208.
-/
theorem theater_loss 
  (capacity : ℕ) 
  (price : ℕ) 
  (sold_tickets : ℕ) 
  (h_cap : capacity = 50) 
  (h_price : price = 8) 
  (h_sold : sold_tickets = 24) : 
  capacity * price - sold_tickets * price = 208 :=
by
  sorry

end theater_loss_l434_434312


namespace derive_units_equivalent_to_velocity_l434_434739

-- Define the unit simplifications
def watt := 1 * (1 * (1 * (1 / 1)))
def newton := 1 * (1 * (1 / (1 * 1)))

-- Define the options
def option_A := watt / newton
def option_B := newton / watt
def option_C := watt / (newton * newton)
def option_D := (watt * watt) / newton
def option_E := (newton * newton) / (watt * watt)

-- Define what it means for a unit to be equivalent to velocity
def is_velocity (unit : ℚ) : Prop := unit = (1 * (1 / 1))

theorem derive_units_equivalent_to_velocity :
  is_velocity option_A ∧ 
  ¬ is_velocity option_B ∧ 
  ¬ is_velocity option_C ∧ 
  ¬ is_velocity option_D ∧ 
  ¬ is_velocity option_E := 
by sorry

end derive_units_equivalent_to_velocity_l434_434739


namespace probability_of_rerolling_two_dice_l434_434122

/-- Jason rolls three fair six-sided dice. Then he looks at the rolls and chooses a subset of the 
dice (possibly empty, possibly all three dice) to reroll. After rerolling, he wins if and only 
if the sum of the numbers face up on the three dice is exactly 7. Jason always plays to optimize 
his chances of winning. Prove that the probability he chooses to reroll exactly two of the dice 
is 7/36. -/
theorem probability_of_rerolling_two_dice :
  let win (dice : Fin 3 → ℕ) := (∑ i, dice i = 7) 
  let F := (Finset.finRange 7).val
  let ⦃optimize_strategy⦄ : Prop := sorry
  in (probability (reroll_exactly_two_and_win win F) = 7 / 36) :=
sorry

end probability_of_rerolling_two_dice_l434_434122


namespace line_equation_l434_434020

noncomputable def P (A B C x y : ℝ) := A * x + B * y + C

theorem line_equation {A B C x₁ y₁ x₂ y₂ : ℝ} (h1 : P A B C x₁ y₁ = 0) (h2 : P A B C x₂ y₂ ≠ 0) :
    ∀ (x y : ℝ), P A B C x y - P A B C x₁ y₁ - P A B C x₂ y₂ = 0 ↔ P A B 0 x y = -P A B 0 x₂ y₂ := by
  sorry

end line_equation_l434_434020


namespace min_abs_diff_l434_434420

theorem min_abs_diff (x y : ℝ) (h : log 4 (x + 2 * y) + log 4 (x - 2 * y) = 1) (h_pos1 : x + 2 * y > 0) (h_pos2 : x - 2 * y > 0) :
  ∃ (L : ℝ), L = abs x - abs y ∧ (∀ w, w = abs x - abs y → w ≥ L) ∧ L = sqrt 3 :=
sorry

end min_abs_diff_l434_434420


namespace distance_between_skew_lines_in_unit_cube_l434_434885

theorem distance_between_skew_lines_in_unit_cube :
  ∀ (A B C D A₁ B₁ C₁ D₁ : Point),
  is_unit_cube A B C D A₁ B₁ C₁ D₁ →
  distance (Line.mk A C) (Line.mk A₁ D) = sqrt 3 / 3 :=
by
  sorry

end distance_between_skew_lines_in_unit_cube_l434_434885


namespace number_of_correct_propositions_l434_434338

noncomputable def proposition1 (x : ℝ) : Prop := x^2 + 1/4 ≥ x

noncomputable def proposition2 (x : ℝ) (k : ℤ) : Prop := ¬(x = k * Real.pi) → sin x + 1 / sin x ≥ 2

noncomputable def proposition3 (x y : ℝ) : Prop := (x > 0) → (y > 0) → (∀ x y, (x + y) * (1/x + 4/y) = 8)

noncomputable def proposition4 (x : ℝ) : Prop := (x > 1) → (∀ x, x + 1 / (x - 1) = 3)

theorem number_of_correct_propositions (n : ℕ) :
  n = (if (∃ x, proposition1 x)
       then 1 else 0) +
      (if (∃ x k, proposition2 x k)
       then 1 else 0) +
      (if (∃ x y, proposition3 x y)
       then 1 else 0) +
      (if (∃ x, proposition4 x)
       then 1 else 0) :=
  sorry

end number_of_correct_propositions_l434_434338


namespace area_triangle_BOC_l434_434114

theorem area_triangle_BOC 
  (A B C K O : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace K] [MetricSpace O]
  [∀ {X}, MetricSpace (Set X)] 
  (triangle_ABC : ∀ {X}, Set (X × X) → Prop)
  (h1 : segment A C = 14)
  (h2 : segment A B = 6)
  (h3 : O = midpoint A C)
  (circle_O : Circle O (segment A C / 2))
  (h4 : K ∈ circle_O ∧ K ∈ segment B C)
  (h5 : ∠ B A K = ∠ A C B) :
  area (triangle B O C) = 21 := 
sorry

end area_triangle_BOC_l434_434114


namespace opposite_neg_9_l434_434679

theorem opposite_neg_9 : 
  ∃ y : Int, -9 + y = 0 ∧ y = 9 :=
by
  sorry

end opposite_neg_9_l434_434679


namespace circle_diameter_l434_434279

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434279


namespace monotonic_intervals_of_f_when_a_is_zero_range_of_a_when_ratio_inequality_holds_l434_434009

noncomputable def f (x a : ℝ) : ℝ := (a + 1) * x^2 + (x - 4) * |x - a| - x

-- Part (1)
theorem monotonic_intervals_of_f_when_a_is_zero :
  (∀ x : ℝ, x < 0 → (∃ k : ℝ, f x 0 = k * x ∧ k > 0)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < (5 / 4) → (∃ k : ℝ, f x 0 = k * x ∧ k < 0)) ∧
  (∀ x : ℝ, x > (5 / 4) → (∃ k : ℝ, f x 0 = k * x ∧ k > 0)) :=
  sorry

-- Part (2)
theorem range_of_a_when_ratio_inequality_holds :
  (∀ x1 x2 : ℝ, x1 ∈ [0, 5] → x2 ∈ [0, 5] → x1 ≠ x2 → 
    (f x1 0 - f x2 0) / (x1^2 - x2^2) > (a : ℝ)) →
    a ∈ (-∞, -5] ∪ [5 / 3, ∞) :=
  sorry

end monotonic_intervals_of_f_when_a_is_zero_range_of_a_when_ratio_inequality_holds_l434_434009


namespace sin_cos_inequality_l434_434534

theorem sin_cos_inequality (α : ℝ) (a : ℝ) (ha : a ≠ 0) (hP : ∃ (x y : ℝ), (x = 2 * a) ∧ (y = 3 * a) ∧ (x^2 + y^2 = 1)) :
  sin α * cos α > 0 := 
sorry

end sin_cos_inequality_l434_434534


namespace determinant_matrix_l434_434378

theorem determinant_matrix : 
  let M := Matrix.of ![![4, -5], ![3, 7]] in
  Matrix.det M = 43 := 
by 
  let M := Matrix.of ![![4, -5], ![3, 7]]
  have h: M.det = 4 * 7 - (-5) * 3, by simp [Matrix.det_fin_two]
  rw [h]
  norm_num
  sorry

end determinant_matrix_l434_434378


namespace eccentricity_of_ellipse_l434_434383

-- Definitions and conditions
variable (a b : ℝ) (e : ℝ) (F_1 F_2 P : ℝ × ℝ)
variable (h_a_gt_b : a > b) (h_b_gt_0 : b > 0)
variable (h_ellipse : ∀ {x y : ℝ}, x^2 / a^2 + y^2 / b^2 = 1)
variable (h_perpendicular : PF_2 ⊥ F_1F_2)
variable (h_angle : ∠ PF_1F_2 = 30°)

-- The proof statement
theorem eccentricity_of_ellipse :
  e = Real.sqrt 3 / 3 :=
sorry

end eccentricity_of_ellipse_l434_434383


namespace correct_statement_count_l434_434673

theorem correct_statement_count :
  let s1 := (∀ (R2 : ℝ), 0 ≤ R2 ∧ R2 ≤ 1 → R2 = 1 → true) -- condition for statement 1
  let s2 := (∀ (a b c d : ℤ), |a * d - b * c| > 0 → false) -- condition for statement 2
  let s3 := (∀ (x : ℝ), x^2 = 1 → (x = 1) → false) -- condition for statement 3
  let s4 := (∀ (a b : ℝ), a > b ↔ a * |a| > b * |b|) -- condition for statement 4
  (s1, s2, s3, s4) = (true, false, false, true) → 2 = 2 :=
  by
  intros s1 s2 s3 s4 h
  exact h

end correct_statement_count_l434_434673


namespace abs_inequality_solution_set_l434_434193

theorem abs_inequality_solution_set (x : ℝ) : -1 < x ∧ x < 1 ↔ |2*x - 1| - |x - 2| < 0 := by
  sorry

end abs_inequality_solution_set_l434_434193


namespace children_neither_happy_nor_sad_l434_434150

theorem children_neither_happy_nor_sad (total_children happy_children sad_children : ℕ)
  (total_boys total_girls happy_boys sad_girls boys_neither_happy_nor_sad : ℕ)
  (h₀ : total_children = 60)
  (h₁ : happy_children = 30)
  (h₂ : sad_children = 10)
  (h₃ : total_boys = 19)
  (h₄ : total_girls = 41)
  (h₅ : happy_boys = 6)
  (h₆ : sad_girls = 4)
  (h₇ : boys_neither_happy_nor_sad = 7) :
  total_children - happy_children - sad_children = 20 :=
by
  sorry

end children_neither_happy_nor_sad_l434_434150


namespace chord_length_condition_l434_434462

noncomputable def line_parametric (t : ℝ) (m : ℝ) : ℝ × ℝ :=
(t, m * t)

noncomputable def circle_parametric (α : ℝ) : ℝ × ℝ :=
(Math.cos α, 1 + Math.sin α)

theorem chord_length_condition (m : ℝ)
  (h : ∃ (t α : ℝ), let line := line_parametric t m in
                    let circle := circle_parametric α in
                    line = circle ∧
                    2 * Real.sqrt (1 - (1 / (m^2 + 1))) ≥ Real.sqrt 2) :
  m ≤ -1 ∨ m ≥ 1 :=
by
  sorry

end chord_length_condition_l434_434462


namespace stratified_sampling_l434_434761

noncomputable def employees := 500
noncomputable def under_35 := 125
noncomputable def between_35_and_49 := 280
noncomputable def over_50 := 95
noncomputable def sample_size := 100

theorem stratified_sampling : 
  under_35 * sample_size / employees = 25 := by
  sorry

end stratified_sampling_l434_434761


namespace james_work_hours_l434_434983

noncomputable def calculate_hours : ℕ := by
  let meat_cost := 20 * 5
  let fruits_vegetables_cost := 15 * 4
  let cheese_cost := 25 * 3.5
  let bread_products_cost := 60 * 1.5
  let milk_cost := 20 * 2
  let juice_cost := 5 * 6
  let total_food_cost := meat_cost + fruits_vegetables_cost + cheese_cost + bread_products_cost + milk_cost + juice_cost
  let cleaning_supplies := 15
  let janitorial_overtime_pay := 10 * 10 * 1.5
  let total_cost := total_food_cost + cleaning_supplies + janitorial_overtime_pay
  let interest := total_cost * 0.05
  let total_cost_with_interest := total_cost + interest
  let hours_to_work := total_cost_with_interest / 8
  let rounded_hours_to_work := hours_to_work.ceil
  exact rounded_hours_to_work

theorem james_work_hours : calculate_hours = 76 := by
  sorry

end james_work_hours_l434_434983


namespace sum_of_coordinates_l434_434632

theorem sum_of_coordinates (x y : ℚ) (h₁ : y = 5) (h₂ : (y - 0) / (x - 0) = 3/4) : x + y = 35/3 :=
by sorry

end sum_of_coordinates_l434_434632


namespace somu_present_age_l434_434174

def Somu_Age_Problem (S F : ℕ) : Prop := 
  S = F / 3 ∧ S - 6 = (F - 6) / 5

theorem somu_present_age (S F : ℕ) 
  (h : Somu_Age_Problem S F) : S = 12 := 
by
  sorry

end somu_present_age_l434_434174


namespace triangle_incircle_radius_incorrect_l434_434115

open Triangle -- Assuming we have a triangle geometry library

-- Define the sides of the triangle and the incorrect statement to be proven
def triangle_ABC_sides (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 4 ∧ c < 1

theorem triangle_incircle_radius_incorrect (a b c : ℝ) (h : triangle_ABC_sides a b c) :
  let r := 2 * (1/2 * a * b) / (a + b + c) in r ≥ 1 :=
by
  -- We need to calculate and prove the condition here.
  sorry

end triangle_incircle_radius_incorrect_l434_434115


namespace three_Z_five_l434_434078

def Z (a b : ℤ) : ℤ := b + 10 * a - 3 * a^2

theorem three_Z_five : Z 3 5 = 8 := sorry

end three_Z_five_l434_434078


namespace f_monotonically_increasing_l434_434456

def f (x : ℝ) : ℝ := 2 * real.sqrt 3 * real.sin (real.pi - x) * real.cos x - 1 + 2 * real.cos x ^ 2

theorem f_monotonically_increasing : monotone_on f (set.Icc (-real.pi / 3) (real.pi / 6)) :=
sorry

end f_monotonically_increasing_l434_434456


namespace sum_sequence_S17_S22_l434_434111

def sequence (n : ℕ) : ℤ := (List.range n).sum (λ k, (-1 : ℤ) ^ k * (4 * (k+1) - 3))

theorem sum_sequence_S17_S22 : sequence 17 + sequence 22 = -11 := by
  -- Proof here
  sorry

end sum_sequence_S17_S22_l434_434111


namespace parameter_values_exist_l434_434401

noncomputable def exists_solution (a : ℝ) : Prop :=
  ∃ t ∈ set.Icc (0 : ℝ) (π / 2), 
    (|Real.cos t - 0.5| + |Real.sin t| - a) / (√3 * Real.sin t - Real.cos t) = 0

theorem parameter_values_exist (a : ℝ) :
  (0.5 ≤ a ∧ a ≤ 1.5) ↔ exists_solution a := sorry

end parameter_values_exist_l434_434401


namespace veronica_yellow_balls_percentage_l434_434718

def percentage_of_yellow_balls (yellow_balls brown_balls : ℕ) : ℝ :=
  (yellow_balls : ℝ) / (yellow_balls + brown_balls) * 100

theorem veronica_yellow_balls_percentage :
  percentage_of_yellow_balls 27 33 = 45 := 
by
  sorry

end veronica_yellow_balls_percentage_l434_434718


namespace angle_ABC_l434_434557

noncomputable def quadrilateral (A B C D : Type) [InnerProductSpace ℝ A] :=
  (BAC CAD ACD ABC : Angle)
  (AB AC AD : ℝ)

axiom quadrilateral_properties (a b c d : Type) [InnerProductSpace ℝ a] :
  ∃ (BAC CAD ACD : Angle) (AB AC AD : ℝ),
  BAC = 60 ∧ CAD = 60 ∧ ACD = 23 ∧ AB + AD = AC

theorem angle_ABC (A B C D : Type) [InnerProductSpace ℝ A]
  (BAC CAD ACD ABC : Angle)
  (AB AC AD : ℝ)
  (h1 : BAC = 60)
  (h2 : CAD = 60)
  (h3 : ACD = 23)
  (h4 : AB + AD = AC) :
  ABC = 83 := 
sorry

end angle_ABC_l434_434557


namespace fraction_computation_l434_434358

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l434_434358


namespace area_of_triangle_l434_434505

-- Define the given curve equation
def curve (x : ℝ) := (x - 4)^2 * (x + 3)

-- The x and y intercepts of the curve, (x, y at x = 0)
def x_intercepts : set ℝ := {x | curve x = 0}
def y_intercept := curve 0

-- Vertices of the triangle
def vertex_A := (-3 : ℝ, 0 : ℝ)
def vertex_B := (4 : ℝ, 0 : ℝ)
def vertex_C := (0 : ℝ, y_intercept)

-- Calculate the base and height of the triangle
def base := 7
def height := y_intercept

-- The area of the triangle
def triangle_area := 1/2 * base * height

-- The theorem to prove
theorem area_of_triangle : triangle_area = 168 := by
  sorry

end area_of_triangle_l434_434505


namespace speech_competition_sequences_l434_434113

theorem speech_competition_sequences
    (contestants : Fin 5 → Prop)
    (girls boys : Fin 5 → Prop)
    (girl_A : Fin 5)
    (not_girl_A_first : ¬contestants 0)
    (no_consecutive_boys : ∀ i, boys i → ¬boys (i + 1))
    (count_girls : ∀ x, girls x → x = girl_A ∨ (contestants x ∧ ¬boys x))
    (count_boys : ∀ x, (boys x) → contestants x)
    (total_count : Fin 5 → Fin 5 → ℕ)
    (correct_answer : total_count = 276) : 
    ∃ seq_count, seq_count = 276 := 
sorry

end speech_competition_sequences_l434_434113


namespace equal_or_equal_exponents_l434_434242

theorem equal_or_equal_exponents
  (a b c p q r : ℕ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r)
  (h1 : a^p + b^q + c^r = a^q + b^r + c^p)
  (h2 : a^q + b^r + c^p = a^r + b^p + c^q) :
  a = b ∧ b = c ∧ c = a ∨ p = q ∧ q = r ∧ r = p :=
  sorry

end equal_or_equal_exponents_l434_434242


namespace part_a_part_b_l434_434241

noncomputable def is_equilateral {A B C : Point} (T : Triangle A B C) : Prop :=
  T.AB = T.BC ∧ T.BC = T.CA

structure Quadrilateral (A B C D : Point) :=
(O : Point)
(BOC_eq : ∠ B O C = 60°)
(AOD_eq : ∠ A O D = 60°)
(T : Point)
(T_reflection_O : reflect_point O (midpoint C D) T)

def area_ratio (Obj1 Obj2 : Polyhedron) : ℝ :=
  (Obj1.area / Obj2.area)

noncomputable def abc_d (AB CD : ℝ) [quadrilateral ABCD] : ℝ :=
  area_ratio (triangle ABCD.A ABCD.B ABCD.T) (quadrilateral ABCD)

theorem part_a {A B C D O T : Point}
  (h : Quadrilateral A B C D O)
  (h_equilateral_boc : is_equilateral (Triangle B O C))
  (h_equilateral_aod : is_equilateral (Triangle A O D))
  (h_reflection : reflect_point O (midpoint C D) T)
  : is_equilateral (Triangle A B T) :=
sorry

theorem part_b 
  {A B C D T : Point}
  (h_quad : Quadrilateral A B C D)
  (BC_eq_2 : dist B C = 2)
  (AD_eq_7 : dist A D = 7)
  : abc_d (2) (7) = 67/81 :=
sorry

end part_a_part_b_l434_434241


namespace flaw_in_major_premise_l434_434326

theorem flaw_in_major_premise 
  (major_premise : ∀ (VCR : Prop) (can_open : Prop), VCR → can_open)
  (minor_premise : ∃ (can_open : Prop), can_open)
  (conclusion : Prop)
  (H : conclusion ↔ ((∃ (VCR : Prop), VCR) ∧ (∃ (can_open : Prop), can_open))) :
  ∃ (flaw : Prop), flaw = major_premise := 
sorry

end flaw_in_major_premise_l434_434326


namespace sequence_type_l434_434022

def a_n (n : ℕ) : ℝ := Real.cos (n * Real.pi)

theorem sequence_type (h : ∀ n : ℕ, a_n n = Real.cos (n * Real.pi)) : 
  ∀ a, a ∈ (list.map a_n (list.range (2 * a)) ∪ list.map a_n (list.range (2 * a))) → 
  (a = 1) ∨ (a = -1) := 
by
  sorry

end sequence_type_l434_434022


namespace meaningful_square_root_l434_434527

theorem meaningful_square_root (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 := 
by {
  -- This part would contain the proof
  sorry
}

end meaningful_square_root_l434_434527


namespace hyperbola_eccentricity_l434_434888

-- Define the conditions
variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c = real.sqrt (a^2 + b^2))

-- Represent the hyperbola C and focuses
def hyperbola_C : Prop := ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1

-- Define points F, A, B, and the relations given
def right_focus_F : Prop := F = (c, 0)
def perpendicular_line_FA : Prop := 
  ∀ A B: ℝ, ∃ m n : ℝ, 
    A = (m, b/a * m) ∧ B = (n, -b/a * n) 
    ∧ 3 * (F.1 - m, -b/a * m) = (n - F.1, -b/a * n)

-- Formulate the final statement for the proof of eccentricity
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (c := real.sqrt (a^2 + b^2)) :
    (a^2 = 2 * b^2) → (c / a = real.sqrt 6 / 2) := 
by 
  sorry

-- Variables assignment for existential quantification in Lean 4
def exists_perpendicular_line_FA := 
  ∃ m n : ℝ, (m = 2 * c / 3) ∧ (n = 2 * c)

-- Final declaration with conditions
def equivalent_proof_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop := 
  ∃ (c : ℝ), c = real.sqrt (a^2 + b^2) ∧
  (a^2 = 2 * b^2) → (c / a = real.sqrt 6 / 2)

end hyperbola_eccentricity_l434_434888


namespace deductive_reasoning_error_l434_434704

theorem deductive_reasoning_error :
    (∃ (r : ℚ), r.is_proper_fraction) → 
    (∀ (z : ℤ), (z : ℚ)) → 
    ¬(∀ (z : ℤ), z.is_proper_fraction) :=
begin
  -- conditions
  intros h_rational_frac h_int_rational,
  -- aim to show the form of reasoning is wrong
  sorry,
end

end deductive_reasoning_error_l434_434704


namespace circle_diameter_l434_434296

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l434_434296


namespace jackson_has_1900_more_than_brandon_l434_434981

-- Conditions
def initial_investment : ℝ := 500
def jackson_multiplier : ℝ := 4
def brandon_multiplier : ℝ := 0.20

-- Final values
def jackson_final_value := jackson_multiplier * initial_investment
def brandon_final_value := brandon_multiplier * initial_investment

-- Statement to prove the difference
theorem jackson_has_1900_more_than_brandon : jackson_final_value - brandon_final_value = 1900 := 
    by sorry

end jackson_has_1900_more_than_brandon_l434_434981


namespace cost_per_first_30_kg_is_10_l434_434345

-- Definitions of the constants based on the conditions
def cost_per_33_kg (p q : ℝ) : Prop := 30 * p + 3 * q = 360
def cost_per_36_kg (p q : ℝ) : Prop := 30 * p + 6 * q = 420
def cost_per_25_kg (p : ℝ) : Prop := 25 * p = 250

-- The statement we want to prove
theorem cost_per_first_30_kg_is_10 (p q : ℝ) 
  (h1 : cost_per_33_kg p q)
  (h2 : cost_per_36_kg p q)
  (h3 : cost_per_25_kg p) : 
  p = 10 :=
sorry

end cost_per_first_30_kg_is_10_l434_434345


namespace baseball_card_count_l434_434237

-- Define initial conditions
def initial_cards := 15

-- Maria takes half of one more than the number of initial cards
def maria_takes := (initial_cards + 1) / 2

-- Remaining cards after Maria takes her share
def remaining_after_maria := initial_cards - maria_takes

-- You give Peter 1 card
def remaining_after_peter := remaining_after_maria - 1

-- Paul triples the remaining cards
def final_cards := remaining_after_peter * 3

-- Theorem statement to prove
theorem baseball_card_count :
  final_cards = 18 := by
sorry

end baseball_card_count_l434_434237


namespace theater_loss_l434_434315

-- Define the conditions
def theater_capacity : Nat := 50
def ticket_price : Nat := 8
def tickets_sold : Nat := 24

-- Define the maximum revenue and actual revenue
def max_revenue : Nat := theater_capacity * ticket_price
def actual_revenue : Nat := tickets_sold * ticket_price

-- Define the money lost by not selling out
def money_lost : Nat := max_revenue - actual_revenue

-- Theorem statement to prove
theorem theater_loss : money_lost = 208 := by
  sorry

end theater_loss_l434_434315


namespace cases_in_2010_l434_434087

noncomputable def decay_model (N0 : ℝ) (k : ℝ) : ℝ → ℝ :=
  λ t, N0 * Real.exp (-k * t)

theorem cases_in_2010 (N0 : ℝ) (N2020 : ℝ) (k : ℝ) 
  (hN0 : N0 = 60000) (hN2020 : N2020 = 300) 
  (hk : k = (1/20) * Real.log 200) : 
  decay_model N0 k 10 = 4243 :=
by
  sorry

end cases_in_2010_l434_434087


namespace shift_arrangements_l434_434819

theorem shift_arrangements (n : ℕ) (k : ℕ) (h_volunteers : n = 14) (h_shift_size : k = 4)
    (h_unique_shift : ∀ (volunteers : set (fin n)), #volunteers = 14 → ∀ (s : finset (fin n)), s.card = 4 → (volunteers ∩ s).card ≤ 1):
    (nat.choose 14 4) * (nat.choose 10 4) * (nat.choose 6 4) = 3153150 := 
by
  sorry

end shift_arrangements_l434_434819


namespace inv_three_mod_thirty_seven_l434_434398

theorem inv_three_mod_thirty_seven : (3 * 25) % 37 = 1 :=
by
  -- Explicit mention to skip the proof with sorry
  sorry

end inv_three_mod_thirty_seven_l434_434398


namespace age_of_youngest_child_l434_434198

theorem age_of_youngest_child (x : ℕ) :
  (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 55) → x = 5 :=
by
  intro h
  sorry

end age_of_youngest_child_l434_434198


namespace triangle_area_l434_434500

-- Define the curve and points
noncomputable def curve (x : ℝ) : ℝ := (x-4)^2 * (x+3)

-- Define the x-intercepts
def x_intercept1 := 4
def x_intercept2 := -3

-- Define the y-intercept
def y_intercept := curve 0

-- Define the base and height of the triangle
def base : ℝ := x_intercept1 - x_intercept2
def height : ℝ := y_intercept

-- Statement of the problem: calculating the area of the triangle
theorem triangle_area : (1/2) * base * height = 168 := by
  sorry

end triangle_area_l434_434500


namespace range_of_a_l434_434908

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 1 then a^x else (4 - a/2)*x + 2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, x1 < x2 → f a x1 ≤ f a x2) ↔ (4 ≤ a ∧ a < 8) :=
by
  sorry

end range_of_a_l434_434908


namespace greatest_triangle_perimeter_l434_434954

theorem greatest_triangle_perimeter :
  ∃ (x : ℕ), 3 < x ∧ x < 6 ∧ max (x + 4 * x + 17) (5 + 4 * 5 + 17) = 42 :=
by
  sorry

end greatest_triangle_perimeter_l434_434954


namespace find_real_roots_l434_434687

-- Definitions of the polynomials P and Q
def P (x : ℝ) := x^2 + x / 2 - 1 / 2
def Q (x : ℝ) := x^2 + x / 2

-- The condition that P(x) Q(x) = Q(P(x))
def Condition_Equality (x : ℝ): Prop := P x * Q x = Q (P x)

-- Theorem statement to find the real roots of P(Q(x)) = 0
theorem find_real_roots : ∀ x : ℝ, P (Q x) = 0 → x = -1 ∨ x = 1 / 2 :=
by
  sorry

end find_real_roots_l434_434687


namespace union_complement_eq_universal_l434_434246

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 5}

-- The proof problem
theorem union_complement_eq_universal :
  U = A ∪ (U \ B) :=
by
  sorry

end union_complement_eq_universal_l434_434246


namespace find_N_l434_434825

theorem find_N :
  ∃ N : ℕ, N > 1 ∧
    (1743 % N = 2019 % N) ∧ (2019 % N = 3008 % N) ∧ N = 23 :=
by
  sorry

end find_N_l434_434825


namespace elena_measurements_l434_434820

theorem elena_measurements :
  let inches_to_feet := 1.0 / 16.0
  let feet_to_cm := 33.0
  let chest_inch := 36.0
  let waist_inch := 28.0
  let chest_cm := ((chest_inch * inches_to_feet) * feet_to_cm)
  let waist_cm := ((waist_inch * inches_to_feet) * feet_to_cm)
  (Real.floor (chest_cm * 10.0 + 0.5) / 10.0 = 74.3 ∧ Real.floor (waist_cm * 10.0 + 0.5) / 10.0 = 57.8) :=
by
  sorry

end elena_measurements_l434_434820


namespace avg_one_sixth_one_fourth_l434_434185

theorem avg_one_sixth_one_fourth : (1 / 6 + 1 / 4) / 2 = 5 / 24 := by
  sorry

end avg_one_sixth_one_fourth_l434_434185


namespace triangle_area_l434_434473

-- Define the given curve
def curve (x : ℝ) : ℝ := (x - 4) ^ 2 * (x + 3)

-- x-intercepts occur when y = 0
def x_intercepts : set ℝ := { x | curve x = 0 }

-- y-intercept occurs when x = 0
def y_intercept : ℝ := curve 0

-- Base of the triangle is the distance between the x-intercepts
def base_of_triangle : ℝ := max (4 : ℝ) (-3) - min (4 : ℝ) (-3)

-- Height of the triangle is the y-intercept value
def height_of_triangle : ℝ := y_intercept

-- Area of the triangle
def area_of_triangle (b h : ℝ) : ℝ := 1 / 2 * b * h

theorem triangle_area : area_of_triangle base_of_triangle height_of_triangle = 168 := by
  -- Stating the problem requires definitions of x-intercepts and y-intercept
  have hx : x_intercepts = {4, -3} := by
    sorry -- The proof for finding x-intercepts

  have hy : y_intercept = 48 := by
    sorry -- The proof for finding y-intercept

  -- Setup base and height using the intercepts
  have b : base_of_triangle = 7 := by
    -- Calculate the base from x_intercepts
    rw [hx]
    exact calc
      4 - (-3) = 4 + 3 := by ring
      ... = 7 := rfl

  have h : height_of_triangle = 48 := by
    -- height_of_triangle should be y_intercept which is 48
    rw [hy]

  -- Finally calculate the area
  have A : area_of_triangle base_of_triangle height_of_triangle = 1 / 2 * 7 * 48 := by
    rw [b, h]

  -- Explicitly calculate the numerical value
  exact calc
    1 / 2 * 7 * 48 = 1 / 2 * 336 := by ring
    ... = 168 := by norm_num

end triangle_area_l434_434473


namespace problem_statement_l434_434895

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 1 then real.log (x - 1) / real.log a else sorry

theorem problem_statement
  (a : ℝ)
  (sym_f : ∀ x, f a (2 - x) = -f a x)
  (h3 : f a 3 = -1)
  (h_add : x₁ + x₂ < 2)
  (h_prod : (x₁ - 1) * (x₂ - 1) < 0) :
  f a x₁ + f a x₂ > 0 :=
sorry

end problem_statement_l434_434895


namespace directrix_of_parabola_l434_434444

theorem directrix_of_parabola (p : ℝ) (hp : p = 6 + 2 * Real.sqrt 2) : 
  ∃ d : ℝ, d = -3 - Real.sqrt 2 ∧ ∀ x y : ℝ, y^2 = 2 * p * x → x = d :=
by
  use -3 - Real.sqrt 2
  split
  . refl
  . intros x y h
    sorry

end directrix_of_parabola_l434_434444


namespace FM_perpendicular_ED_l434_434956

variables {A B C D E F G K L M : Type} 
variables [inner_product_space ℝ A]

-- Conditions: Defining points and their relations
variables (ABC : triangle)
variables (D E F G K L M : point)
variables (h1 : A = perpendicular_foot D B C)
variables (h2 : C = perpendicular_foot E A B)
variables (h3 : line E parallel line BC ∧ intersection_point E parallel_line BC AC F)
variables (h4 : line D parallel line AB ∧ intersection_point D parallel_line AB AC G)
variables (h5 : K = perpendicular_foot F D G)
variables (h6 : L = perpendicular_foot F G E)
variables (h7 : intersection_point K L ED M)

theorem FM_perpendicular_ED : angle F M E = 90 :=
by sorry

end FM_perpendicular_ED_l434_434956


namespace handshake_problem_l434_434789

theorem handshake_problem :
  let team_size := 6
  let teams := 2
  let referees := 3
  let handshakes_between_teams := team_size * team_size
  let handshakes_within_teams := teams * (team_size * (team_size - 1)) / 2
  let handshakes_with_referees := (teams * team_size) * referees
  handshakes_between_teams + handshakes_within_teams + handshakes_with_referees = 102 := by
  sorry

end handshake_problem_l434_434789


namespace opposite_of_neg_nine_is_nine_l434_434681

-- Define the predicate for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that needs to be proved
theorem opposite_of_neg_nine_is_nine : ∃ x, is_opposite (-9) x ∧ x = 9 := 
by {
  -- Use "sorry" to indicate the proof is not provided
  sorry
}

end opposite_of_neg_nine_is_nine_l434_434681


namespace hyperbola_equation_l434_434010

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ c : ℝ, c = sqrt (a^2 + b^2) ∧ c = 2)
  (h4 : ∃ k : ℝ, k = sqrt (a^2 + b^2) ∧ (2 * b / k = sqrt 3)) :
  ∃ a b : ℝ, (x^2 - (y^2 / 3) = 1) :=
by
  sorry

end hyperbola_equation_l434_434010


namespace fraction_identity_l434_434364

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l434_434364


namespace intersection_point_symmetric_to_A_with_respect_to_line_l434_434890

theorem intersection_point_symmetric_to_A_with_respect_to_line
  (A : ℝ × ℝ)
  (hA : A = (-2, 2))
  (center1 center2 : ℝ × ℝ)
  (h1 : center1.1 - center1.2 + 1 = 0)
  (h2 : center2.1 - center2.2 + 1 = 0)
  (intersects : (∃ (x y : ℝ), x = -2 ∧ y = 2 ∧ 
               (x - y + 1 = 0)))
  : ∃ B : ℝ × ℝ, B = (1, -1) :=
by
  use (1, -1)
  sorry


end intersection_point_symmetric_to_A_with_respect_to_line_l434_434890


namespace fuel_cost_per_liter_l434_434092

def service_cost_per_vehicle : ℝ := 2.20
def num_minivans : ℕ := 3
def num_trucks : ℕ := 2
def total_cost : ℝ := 347.7
def mini_van_tank_capacity : ℝ := 65
def truck_tank_increase : ℝ := 1.2
def truck_tank_capacity : ℝ := mini_van_tank_capacity * (1 + truck_tank_increase)

theorem fuel_cost_per_liter : 
  let total_service_cost := (num_minivans + num_trucks) * service_cost_per_vehicle
  let total_capacity_minivans := num_minivans * mini_van_tank_capacity
  let total_capacity_trucks := num_trucks * truck_tank_capacity
  let total_fuel_capacity := total_capacity_minivans + total_capacity_trucks
  let fuel_cost := total_cost - total_service_cost
  let cost_per_liter := fuel_cost / total_fuel_capacity
  cost_per_liter = 0.70 := 
  sorry

end fuel_cost_per_liter_l434_434092


namespace cost_increase_percentage_l434_434095

theorem cost_increase_percentage 
  (C S : ℝ) (X : ℝ)
  (h_proft : S = 2.6 * C)
  (h_new_profit : 1.6 * C - (X / 100) * C = 0.5692307692307692 * S) :
  X = 12 := 
by
  sorry

end cost_increase_percentage_l434_434095


namespace triangle_area_ABC_l434_434782

-- Define the three vertices of the triangle A, B, and C
def A : ℝ × ℝ := (3, -3)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (12, 4)

-- Define the lengths of the sides AB and BC
def AB : ℝ := dist A B
def BC : ℝ := dist B C

-- Prove that the area of the triangle ABC is 31.5 square units
theorem triangle_area_ABC : 
  (calc
    let base := BC
    let height := AB
    let area := 0.5 * base * height
    area := 31.5
   )
  sorry

end triangle_area_ABC_l434_434782


namespace probability_of_two_primary_schools_from_selected_l434_434751

-- Definitions
def num_primary_schools : ℕ := 21
def num_middle_schools : ℕ := 14
def num_universities : ℕ := 7

def num_selected_primary_schools : ℕ := 3
def num_selected_middle_schools : ℕ := 2
def num_selected_universities : ℕ := 1

def selected_schools : Finset ℕ := Finset.range 6

-- Total possible outcomes of selecting 2 schools from 6 selected schools
def total_possible_outcomes : Finset (Finset ℕ) := Finset.powersetLen 2 selected_schools

-- Outcomes where both selected schools are primary
def primary_school_outcomes : Finset (Finset ℕ) := {Finset.mk [0, 1], Finset.mk [0, 2], Finset.mk [1, 2]}

-- Total combination count
def total_outcome_count : ℕ := Finset.card total_possible_outcomes

-- Favorable outcome count
def favorable_outcome_count : ℕ := Finset.card primary_school_outcomes

-- Probability calculation
def selection_probability : ℚ := favorable_outcome_count / total_outcome_count

theorem probability_of_two_primary_schools_from_selected : selection_probability = 1 / 5 := by
  -- Sorry is used to skip the proof.
  sorry

end probability_of_two_primary_schools_from_selected_l434_434751


namespace mia_socks_problem_l434_434147

theorem mia_socks_problem (x y z w : ℕ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hw : 1 ≤ w)
  (h1 : x + y + z + w = 16) (h2 : x + 2*y + 3*z + 4*w = 36) : x = 3 :=
sorry

end mia_socks_problem_l434_434147


namespace math_problem_proof_l434_434101

-- Define curve C with parametric equations
def curve_C_param (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sin α)

-- Ordinary equation of curve C
def curve_C_equation (x y : ℝ) : Prop := (x^2/4 + y^2 = 1)

-- Polar equation of line l
def line_l_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - (Real.pi / 4)) = (Real.sqrt 2) / 2

-- Ordinary equation of line l
def line_l_equation (x y : ℝ) : Prop := x - y + 1 = 0

-- Point P coordinates
def point_P : ℝ × ℝ := (0, 1)

-- Given conditions, prove the required results
theorem math_problem_proof :
  (∀ α, curve_C_equation (2 * Real.cos α) (Real.sin α)) ∧
  (line_l_polar ick
  sorry

end math_problem_proof_l434_434101


namespace sequence_product_exceeds_million_l434_434920

theorem sequence_product_exceeds_million (n : ℕ) (h₁ : ∑ i in Finset.range(n + 1), i > 0) :
  (∏ i in Finset.range(n + 1), 5 ^ (i / 3 : ℚ)) > 1000000 ↔ n = 8 :=
by
  sorry

end sequence_product_exceeds_million_l434_434920


namespace determine_a_l434_434854

noncomputable def polynomial_factorization (a : ℝ) : Prop :=
  ∃ b : ℝ, (y^2 + 3 * y - a) = (y - 3) * (y + b)

theorem determine_a (a : ℝ) :
  polynomial_factorization a → a = 18 :=
by
  -- Need proof here
  sorry

end determine_a_l434_434854


namespace peter_reads_more_books_l434_434158

-- Definitions and conditions
def total_books : ℕ := 20
def peter_percentage_read : ℕ := 40
def brother_percentage_read : ℕ := 10

def percentage_to_count (percentage : ℕ) (total : ℕ) : ℕ := (percentage * total) / 100

-- Main statement to prove
theorem peter_reads_more_books :
  percentage_to_count peter_percentage_read total_books - percentage_to_count brother_percentage_read total_books = 6 :=
by
  sorry

end peter_reads_more_books_l434_434158


namespace trigonometry_problem_l434_434058

theorem trigonometry_problem
  (θ φ : ℝ)
  (h1 : θ ∈ set.Ioo 0 (Real.pi / 2))
  (h2 : φ ∈ set.Ioo 0 (Real.pi / 2))
  (h3 : (Real.sin θ, -2):ℝ × ℝ. orthogonal := (1, Real.cos θ))
  (h4 : 5 * Real.cos (θ - φ)=  3 * Real.sqrt 5 * Real.cos φ):
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧ 
  Real.cos θ = Real.sqrt 5 / 5 ∧ 
  φ= Real.pi / 4 := by sorry

end trigonometry_problem_l434_434058


namespace factorial_inequality_l434_434998

theorem factorial_inequality (m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_leq : n ≤ m) :
  2^n * nat.factorial n ≤ (nat.factorial (m + n)) / (nat.factorial (m - n))
  ∧ (nat.factorial (m + n)) / (nat.factorial (m - n)) ≤ (m^2 + m)^n := 
sorry

end factorial_inequality_l434_434998


namespace rachelle_meat_needed_l434_434161

-- Define the ratio of meat per hamburger
def meat_per_hamburger (pounds : ℕ) (hamburgers : ℕ) : ℚ :=
  pounds / hamburgers

-- Define the total meat needed for a given number of hamburgers
def total_meat (meat_per_hamburger : ℚ) (hamburgers : ℕ) : ℚ :=
  meat_per_hamburger * hamburgers

-- Prove that Rachelle needs 15 pounds of meat to make 36 hamburgers
theorem rachelle_meat_needed : total_meat (meat_per_hamburger 5 12) 36 = 15 := by
  sorry

end rachelle_meat_needed_l434_434161


namespace AB_passes_through_focus_l434_434423

-- Define the condition that a point is on the parabola
def on_parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define the length of segment AB
def length_AB (A B : ℝ × ℝ) : ℝ := 
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Definition of segment AB passing through the focus
def segment_through_focus (A B F : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, F = (A.1 + k * (B.1 - A.1), A.2 + k * (B.2 - A.2))

theorem AB_passes_through_focus {A B : ℝ × ℝ} (F : ℝ × ℝ) (p a : ℝ)
  (h_non_neg : 0 < 2 * p ≤ a) 
  (h_A_on_parabola : on_parabola p A.1 A.2)
  (h_B_on_parabola : on_parabola p B.1 B.2)
  (h_length_AB : length_AB A B = a) 
  (h_tangent_directrix : on_directrix (A, B) directrix) :
  segment_through_focus A B F := 
sorry

end AB_passes_through_focus_l434_434423


namespace sequence_n_equals_l434_434013

noncomputable def arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, n > 0 → a_n n = a_n 1 + (n - 1) * d

noncomputable def sum_first_n_terms (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) :=
  ∀ n : ℕ, S_n n = n * a_n 1 + (n * (n - 1) / 2) * (a_n 2 - a_n 1)

noncomputable def transformed_sequence_is_arithmetic (S_n : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, n > 0 → (sqrt (S_n n + n)) = (sqrt (S_n 1 + 1)) + (n - 1) * d

theorem sequence_n_equals :
  ∃ (a : ℕ → ℝ) (d : ℝ), arithmetic_sequence a d ∧ sum_first_n_terms a (λ n, n * a 1 + (n * (n - 1) / 2) * d) ∧ transformed_sequence_is_arithmetic (λ n, n * a 1 + (n * (n - 1) / 2) * d) d ∧
  ∀ n, a n = -1 ∨ a n = (1 / 2 : ℝ) * n - (5 / 4 : ℝ) :=
begin
  sorry
end

end sequence_n_equals_l434_434013


namespace votes_majority_proved_l434_434957

variable (V : ℝ)

def total_votes_condition (V : ℝ) :=
  V ≥ 7143 ∧ 0.65 * V ≥ (0.65 * V - 2500) + 0.35 * V

theorem votes_majority_proved (V : ℝ) (h : total_votes_condition V) : 
  V ≥ 7143 := 
begin
  sorry
end

end votes_majority_proved_l434_434957


namespace shuffleboard_total_games_l434_434565

theorem shuffleboard_total_games
    (jerry_wins : ℕ)
    (dave_wins : ℕ)
    (ken_wins : ℕ)
    (h1 : jerry_wins = 7)
    (h2 : dave_wins = jerry_wins + 3)
    (h3 : ken_wins = dave_wins + 5) :
    jerry_wins + dave_wins + ken_wins = 32 := 
by
  sorry

end shuffleboard_total_games_l434_434565


namespace kyoko_payment_l434_434993

noncomputable def total_cost (balls skipropes frisbees : ℕ) (ball_cost rope_cost frisbee_cost : ℝ) : ℝ :=
  (balls * ball_cost) + (skipropes * rope_cost) + (frisbees * frisbee_cost)

noncomputable def final_amount (total_cost discount_rate : ℝ) : ℝ :=
  total_cost - (discount_rate * total_cost)

theorem kyoko_payment :
  let balls := 3
  let skipropes := 2
  let frisbees := 4
  let ball_cost := 1.54
  let rope_cost := 3.78
  let frisbee_cost := 2.63
  let discount_rate := 0.07
  final_amount (total_cost balls skipropes frisbees ball_cost rope_cost frisbee_cost) discount_rate = 21.11 :=
by
  sorry

end kyoko_payment_l434_434993


namespace fraction_identity_l434_434365

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l434_434365


namespace hexagon_midpoints_intersect_at_single_point_l434_434684

-- Define the structure of convex hexagon
structure ConvexHexagon (V : Type) :=
(vertices : Fin 6 → V)
(convex : ∀ A B C : Fin 6, A ≠ B → B ≠ C → A ≠ C → ∃ P ∈ ↑(∧ {α : Finset (Fin 6) // α.card = 3}), segment V (vertices A) (vertices B) ∩ segment V (vertices B) (vertices C) ⊆ triangles P)
(opposite_parallel : ∀ i : Fin 3, parallel (segment V (vertices i)) (segment V (vertices (i + 3))))

-- Define what it means for lines to intersect at a single point
def lines_intersect_at_single_point {V : Type} [field V] 
  (l1 l2 l3 : affine_subspace V) : Prop :=
∀ P : V, P ∈ l1 → P ∈ l2 → P ∈ l3

-- Problem statement
theorem hexagon_midpoints_intersect_at_single_point {V : Type} [field V] 
  (hex : ConvexHexagon V) :
  ∃ P : V, ∀ i : Fin 3, let mid1 := midpoint V (hex.vertices i) (hex.vertices (i + 1)) 
                        in let mid2 := midpoint V (hex.vertices (i + 3)) (hex.vertices (i + 4))
                        in P ∈ line [mid1, mid2] := 
sorry

end hexagon_midpoints_intersect_at_single_point_l434_434684


namespace sin_alpha_plus_5pi_div_6_cos_alpha_minus_beta_l434_434753

-- Problem 1: Lean statement for proving sin(α + 5π/6) given the condition
theorem sin_alpha_plus_5pi_div_6 (α : ℝ) 
  (h : cos (α + π / 6) - sin α = 3 * real.sqrt 3 / 5) :
  sin (α + 5 * π / 6) = 3 / 5 :=
sorry

-- Problem 2: Lean statement for proving cos(α - β) given the conditions
theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : sin α + sin β = 1 / 2)
  (h2 : cos α + cos β = real.sqrt 2 / 2) :
  cos (α - β) = -5 / 8 :=
sorry

end sin_alpha_plus_5pi_div_6_cos_alpha_minus_beta_l434_434753


namespace circle_diameter_l434_434280

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434280


namespace fraction_identity_l434_434363

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l434_434363


namespace sum_coords_B_l434_434625

def point (A B : ℝ × ℝ) : Prop :=
A = (0, 0) ∧ B.snd = 5 ∧ (B.snd - A.snd) / (B.fst - A.fst) = 3 / 4

theorem sum_coords_B (x : ℝ) :
  point (0, 0) (x, 5) → x + 5 = 35 / 3 :=
by
  intro h
  sorry

end sum_coords_B_l434_434625


namespace fraction_identity_l434_434366

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l434_434366


namespace symmetric_projection_wrt_midline_l434_434995

-- Definitions for Triangles, Midpoints, Lines, and Orthogonal Projections
structure Point := (x : ℝ) (y : ℝ)
structure Line := (a : ℝ) (b : ℝ) (c : ℝ) -- representing ax + by + c = 0

-- Definitions for Triangle and Midpoints
structure Triangle := (A B C : Point)
def midpoint (P Q : Point) : Point := 
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

-- Definition of orthogonal projection (placeholder, assuming well-defined)
def orth_proj (P : Point) (l : Line) : Point := sorry

-- Example Symmetry with respect to Angle Bisector (simplified)
def symmetric_wrt_angle_bisector (l1 l2 : Line) (vertex : Point) : Prop := sorry

-- Main theorem statement
theorem symmetric_projection_wrt_midline (A B C : Point) 
  (C' := midpoint A B) (A' := midpoint B C)
  (g g' : Line) (h_symm : symmetric_wrt_angle_bisector g g' A)
  (Y := orth_proj B g) (Y' := orth_proj B g') :
  let CA' := Line.mk (C'.x - A'.x) (C'.y - A'.y) 0 in
  ∃ CA' : Line, 
  (∃ (Y Y' : Point), 
    orth_proj B g = Y ∧ orth_proj B g' = Y' ∧ 
    (Y.y - Y'.y) = 0 ∧ 
    symmetric_wrt_angle_bisector Y Y' CA') := 
sorry

end symmetric_projection_wrt_midline_l434_434995


namespace double_summation_eq_l434_434798

theorem double_summation_eq :
  (∑ i in Finset.range 50 + 1, ∑ j in Finset.range 150 + 1, (i^2 + j)) = 7005000 :=
by
  sorry

end double_summation_eq_l434_434798


namespace negation_of_exists_l434_434430

theorem negation_of_exists (p : Prop) :
  (∃ x : ℝ, x^2 + 2 * x < 0) ↔ ¬ (∀ x : ℝ, x^2 + 2 * x >= 0) :=
sorry

end negation_of_exists_l434_434430


namespace tan_eq_sqrt3_l434_434066

theorem tan_eq_sqrt3 
  (α : ℝ) (h0 : 0 < α) (h1 : α < π / 2)  
  (h2 : sin α ^ 2 + cos (2 * α) = 1 / 4) : 
  tan α = √3 :=
by
  sorry

end tan_eq_sqrt3_l434_434066


namespace circle_diameter_l434_434263

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l434_434263


namespace sum_first_n_terms_of_geometric_seq_l434_434077

variable {α : Type*} [LinearOrderedField α] (a r : α) (n : ℕ)

def geometric_sequence (a r : α) (n : ℕ) : α :=
  a * r ^ (n - 1)

def sum_geometric_sequence (a r : α) (n : ℕ) : α :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_first_n_terms_of_geometric_seq (h₁ : a * r + a * r^3 = 20) 
    (h₂ : a * r^2 + a * r^4 = 40) :
  sum_geometric_sequence a r n = 2^(n + 1) - 2 := 
sorry

end sum_first_n_terms_of_geometric_seq_l434_434077


namespace phung_more_than_chiu_l434_434606

theorem phung_more_than_chiu
  (C P H : ℕ)
  (h1 : C = 56)
  (h2 : H = P + 5)
  (h3 : C + P + H = 205) :
  P - C = 16 :=
by
  sorry

end phung_more_than_chiu_l434_434606


namespace max_sum_of_digits_time_l434_434392

def sum_of_digits (n : Nat) : Nat :=
  n / 10 + n % 10

def time_with_max_sum_digits : Nat × Nat :=
  (19, 59)

theorem max_sum_of_digits_time : ∃ (hh mm : Nat), (hh < 24 ∧ mm < 60) ∧ 
  (∀ (hh' mm' : Nat), hh' < 24 → mm' < 60 → 
  sum_of_digits hh + sum_of_digits mm ≥ sum_of_digits hh' + sum_of_digits mm') ∧
  (hh, mm) = (19, 59) :=
by {
  use (19, 59),
  split,
  { split; norm_num },
  { intros hh' mm' hh_range mm_range,
    sorry -- Proof would go here
  },
  norm_num
}

end max_sum_of_digits_time_l434_434392


namespace evaluate_expression_l434_434353

theorem evaluate_expression : (5 - Real.pi)^0 - abs (-1 / 8) + (-2 : ℝ)^(-3) = 3 / 4 :=
by
  sorry

end evaluate_expression_l434_434353


namespace find_g_h_l434_434411

theorem find_g_h (g h : ℚ) (polynomial : polynomial ℚ) :
  (8 * polynomial.X^2 - 5 * polynomial.X + polynomial.C g) *
  (2 * polynomial.X^2 + h * polynomial.X - 9) =
  (16 * polynomial.X^4 + 21 * polynomial.X^3 - 73 * polynomial.X^2 - 41 * polynomial.X + 45) →
  g + h = -82 / 25 :=
by
  sorry

end find_g_h_l434_434411


namespace opposite_neg_9_l434_434680

theorem opposite_neg_9 : 
  ∃ y : Int, -9 + y = 0 ∧ y = 9 :=
by
  sorry

end opposite_neg_9_l434_434680


namespace general_term_a_n_sum_first_n_b_n_l434_434864

-- Define the arithmetic sequence {a_n} with conditions a₃ = 5 and S₁₅ = 225
def is_arithmetic_sequence (a_n : ℕ → ℕ) : Prop :=
  a_n 3 = 5 ∧ ∑ i in finset.range 15, a_n (i + 1) = 225

-- Define the conditions
def condition_1 (a_n : ℕ → ℕ) : Prop :=
  a_n 3 = 5

def condition_2 (a_n : ℕ → ℕ) : Prop :=
  ∑ i in finset.range 15, a_n (i + 1) = 225

-- Define b_n
def b_n (a_n : ℕ → ℕ) (n : ℕ) : ℕ :=
  2 ^ (a_n n) + 2 * n

-- Prove the general term of a_n
theorem general_term_a_n (a_n : ℕ → ℕ) (h1 : condition_1 a_n) (h2 : condition_2 a_n) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- Prove the sum of the first n terms of b_n is T_n
theorem sum_first_n_b_n (a_n : ℕ → ℕ) (h1 : condition_1 a_n) (h2 : condition_2 a_n)
  (h : ∀ n, a_n n = 2 * n - 1) :
  ∀ n : ℕ, (∑ i in finset.range n, b_n a_n (i + 1)) = ((2 * (3^n - 1)) / 3) + n + n^2 :=
sorry

end general_term_a_n_sum_first_n_b_n_l434_434864


namespace card_selection_l434_434508

theorem card_selection :
  (Finset.card {S : Finset (Fin 52) | 
                 S.card = 4 ∧ 
                 ∃ s1 s2 s3 s4 : Fin 52, 
                   s1.val / 13 = s2.val / 13 ∧ s1.val / 13 ≠ s3.val / 13 ∧ s1.val / 13 ≠ s4.val / 13}) = 158004 := by
  sorry

end card_selection_l434_434508


namespace biased_coin_probability_l434_434250

theorem biased_coin_probability (p_h : ℚ) (n : ℕ) (k : ℕ) (prob : ℚ) : 
  p_h = 1/3 → n = 8 → k = 3 → prob = 1792/6561 →
  (nat.choose n k) * (p_h^k) * ((1 - p_h)^(n - k)) = prob :=
by
  sorry

end biased_coin_probability_l434_434250


namespace inequality_proof_l434_434927

theorem inequality_proof
  (a b λ : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hλ₁ : 0 < λ) 
  (hλ₂ : λ ≤ 1/2) :
  (a^λ + b^λ) * (1/(2*a + b)^λ + 1/(a + 2*b)^λ) ≤ 4/(3^λ) :=
by
  sorry

end inequality_proof_l434_434927


namespace sector_arc_length_l434_434012

theorem sector_arc_length (r : ℝ) (θ : ℝ) (deg_to_rad : ℝ) (pi_approx : ℝ) 
  (r_eq : r = 8) (θ_eq : θ = 45) (deg_to_rad_eq : deg_to_rad = π / 180) 
  (pi_approx_eq : pi_approx = π) : 
  let θ_rad := θ * deg_to_rad in
  θ_rad * r = 2 * pi_approx :=
by 
  have theta_rad_calc : θ * deg_to_rad = π / 4 := sorry
  rw [theta_rad_calc, r_eq, pi_approx_eq],
  sorry

end sector_arc_length_l434_434012


namespace net_loss_correct_l434_434772

-- Definitions based on conditions
def borrowed_amount : ℝ := 7000       -- Principal borrowed
def lent_amount : ℝ := 5000           -- Principal lent
def borrow_rate : ℝ := 0.06           -- Annual interest rate (borrowed)
def lend_rate : ℝ := 0.08             -- Annual interest rate (lent)
def borrow_n : ℝ := 1                 -- Compounding frequency (annually)
def lend_n : ℝ := 2                   -- Compounding frequency (semi-annually)
def years : ℝ := 3                    -- Time in years

-- Calculate amount owed after 3 years with compounded annually interest
def future_value_borrowed : ℝ := borrowed_amount * (1 + borrow_rate / borrow_n) ^ (borrow_n * years)

-- Calculate amount received after 3 years with compounded semi-annually interest
def future_value_lent : ℝ := lent_amount * (1 + lend_rate / lend_n) ^ (lend_n * years)

-- Net gain or loss
def net_gain_or_loss : ℝ := future_value_lent - future_value_borrowed

theorem net_loss_correct : net_gain_or_loss = -2010.517 :=
by
  sorry

end net_loss_correct_l434_434772


namespace number_of_valid_polynomials_l434_434930

-- Define the set S and the property of the polynomial.
def S : Set ℝ := {1, 2, 3, 4, 5, 6}

def isPermutation (f : ℝ → ℝ) : Prop :=
  ∃ (σ : Fin 6 → Fin 6), ∀ x ∈ S, f x = (σ ⟨x, by decide⟩).val

-- Polynomial of degree exactly 5 with real coefficients that permutes elements of S.
def isValidPolynomial (P : ℝ → ℝ) : Prop :=
  degree P = 5 ∧ ∀ x ∈ S, P.coeff x ∈ S ∧ isPermutation P

-- Number of such polynomials
theorem number_of_valid_polynomials : ∃ n, n = 718 ∧ 
  (∀ P : ℝ → ℝ, isValidPolynomial P → ∃ σ : Fin 6 → Fin 6, (P.coeff ∘ σ) = P.coeff) := 
sorry

end number_of_valid_polynomials_l434_434930


namespace circle_diameter_problem_circle_diameter_l434_434286

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l434_434286


namespace roots_difference_quadratic_eq_three_l434_434352

def quadratic_roots_difference {a b c : ℝ} (h : a ≠ 0) (h_eq : a = 1 ∧ b = -9 ∧ c = 18) : ℝ :=
  let r1 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a) in
  let r2 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a) in
  r1 - r2

theorem roots_difference_quadratic_eq_three : quadratic_roots_difference (by norm_num) (by norm_num) = 3 :=
sorry

end roots_difference_quadratic_eq_three_l434_434352


namespace solution_l434_434018

variable {x y z : ℝ}

def condition1 : Prop := log 2 (x * y * z - 6 + log 5 x) = 4
def condition2 : Prop := log 3 (x * y * z - 6 + log 5 y) = 3
def condition3 : Prop := log 4 (x * y * z - 6 + log 5 z) = 2

theorem solution (hx : condition1) (hy : condition2) (hz : condition3) :
  abs (log 5 x) + abs (log 5 y) + abs (log 5 z) = 14 :=
sorry

end solution_l434_434018


namespace response_rate_is_64_99_l434_434255

noncomputable def response_rate_percentage (responses : ℝ) (questionnaires : ℝ) : ℝ :=
  (responses / questionnaires) * 100

theorem response_rate_is_64_99 (h1 : responses = 300) (h2 : questionnaires = 461.54) :
  response_rate_percentage 300 461.54 = 64.99 :=
by
  have H : response_rate_percentage 300 461.54 = (300 / 461.54) * 100 := rfl
  rw [h1, h2] at H
  rw H
  sorry

end response_rate_is_64_99_l434_434255


namespace opposite_negative_nine_l434_434676

theorem opposite_negative_nine : 
  (∃ (y : ℤ), -9 + y = 0 ∧ y = 9) :=
by sorry

end opposite_negative_nine_l434_434676


namespace find_a3_l434_434968

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

noncomputable
def arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem find_a3 {a : ℕ → ℤ} (d : ℤ) (h6 : a 6 = 6) (h9 : a 9 = 9) :
  (∃ d : ℤ, arithmetic_sequence a d) →
  a 3 = 3 :=
by
  intro h_arith_seq
  sorry

end find_a3_l434_434968


namespace m_div_x_eq_4_div_5_l434_434746

variable (a b : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_ratio : a / b = 4 / 5)

def x := a * 1.25

def m := b * 0.80

theorem m_div_x_eq_4_div_5 : m / x = 4 / 5 :=
by
  sorry

end m_div_x_eq_4_div_5_l434_434746


namespace number_of_ordered_pairs_l434_434043

def num_even_pair_solutions : Nat :=
  let solutions := { (a, b) : Nat × Nat | a + b = 40 ∧ (∃ x y, a = 2 * x ∧ b = 2 * y ∧ x > 0 ∧ y > 0) }
  Finset.card solutions

theorem number_of_ordered_pairs : num_even_pair_solutions = 19 := sorry

end number_of_ordered_pairs_l434_434043


namespace sum_coordinates_eq_l434_434629

theorem sum_coordinates_eq : ∀ (A B : ℝ × ℝ), A = (0, 0) → (∃ x : ℝ, B = (x, 5)) → 
  (∀ x : ℝ, B = (x, 5) → (5 - 0) / (x - 0) = 3 / 4) → 
  let x := 20 / 3 in
  (x + 5 = 35 / 3) :=
by
  intros A B hA hB hslope
  have hx : ∃ x, B = (x, 5) := hB
  cases hx with x hx_def
  rw hx_def at hslope
  have : x = 20 / 3 := by
    rw hx_def at hslope
    field_simp at hslope
    linarith
  rw [this, hx_def]
  norm_num

end sum_coordinates_eq_l434_434629


namespace invariant_n_circle_is_constant_l434_434542

noncomputable def isosceles_triangle (A B C : Type) [Real]
  (h : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (AB AC : Real) (H : AB = AC) :=
  let altitude := -- Define the altitude calculation from A to BC (denoted as D)
  circle_with_radius := altitude

noncomputable def invariant_n (A B C : Type) [Real]
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (AB AC : Real) (H : AB = AC)
  (altitude := -- Define the altitude calculation from A to midpoint D on BC
  (circle_r := altitude, roll_along_AB)
  (intersect_AC_BC : variable_points M N)
  (theta := half_angle (angle_BAC := 50)) :=
  -- Prove the fixed number of degrees in arc:
  let n := 2 * theta
  n = 100

theorem invariant_n_circle_is_constant:
  ∀ (A B C : Type) [Real] 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (AB AC : Real) (H : AB = AC),
  ∃ (altitude := -- altitude calculation from A to midpoint D on BC),
  ∃ (circle_r := altitude).roll_along_AB,
  ∃ (variable_points M N),
  let θ := 50
  let arc_MTN := 2 * θ,
  arc_MTN = 100 := sorry


end invariant_n_circle_is_constant_l434_434542


namespace inequality_proof_l434_434243

theorem inequality_proof 
  (a : ℕ → ℝ) (n : ℕ) 
  (h1: 0 < n)
  (h2: ∀ i j, i < j → j ≤ 100 * n → a i ≥ a j) 
  (h3 : ∀ (b : Fin (2 * n + 1) → ℝ), 
         (∀ i j, i < j → b i ≥ b j) → 
         ∃ S1 S2 : ℝ, S1 = ∑ i in Finset.range n, b i ∧ S2 = ∑ i in Finset.range (2 * n + 1) \ Finset.range n, b i ∧ S1 > S2) :
  (n + 1) * ∑ i in Finset.range n, a i > ∑ i in Finset.Ico n (100 * n), a i := 
sorry

end inequality_proof_l434_434243


namespace find_m_l434_434465

def vec_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vec_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_m (m : ℝ) : dot_product (vec_a m) (vec_b m) = 0 ↔ m = -1/3 := by 
  sorry

end find_m_l434_434465


namespace trajectory_equation_F1M_equation_l434_434466

def F1 := (-(Real.sqrt 3), 0)
def F2 := (Real.sqrt 3, 0)

def trajectory_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  Real.sqrt ((x - Real.sqrt 3) ^ 2 + y ^ 2) + Real.sqrt ((x + Real.sqrt 3) ^ 2 + y ^ 2) = 4

theorem trajectory_equation :
  ∀ (P : ℝ × ℝ),
    trajectory_condition P →
    ∃ x y, P = (x, y) ∧ x^2 / 4 + y^2 = 1 := sorry

theorem F1M_equation : 
  ∃ m : ℝ, 
    let F1M := λ y, m * y - Real.sqrt 3 in
    circle_through_origin (center := (0, 2)) (diameter := F1M) ∧
    (m = Real.sqrt (4 * Real.sqrt 10 - 2) / 4 ∨ 
     m = -Real.sqrt (4 * Real.sqrt 10 - 2) / 4) := sorry

end trajectory_equation_F1M_equation_l434_434466


namespace value_of_R_l434_434061

theorem value_of_R (R : ℝ) (hR_pos : 0 < R)
  (h_line : ∀ x y : ℝ, x + y = 2 * R)
  (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = R) :
  R = (3 + Real.sqrt 5) / 4 ∨ R = (3 - Real.sqrt 5) / 4 :=
by
  sorry

end value_of_R_l434_434061


namespace smallest_number_mod_conditions_l434_434728

theorem smallest_number_mod_conditions :
  ∃ b : ℕ, b > 0 ∧ b % 3 = 2 ∧ b % 5 = 3 ∧ ∀ n : ℕ, (n > 0 ∧ n % 3 = 2 ∧ n % 5 = 3) → n ≥ b :=
begin
  use 8,
  split,
  { linarith },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  intros n h,
  cases h with n_pos h_cond,
  cases h_cond with h_mod3 h_mod5,
  have h := chinese_remainder_theorem 3 5 (by norm_num) (by norm_num_ge)
  (λ _, by norm_num) (λ _ _, by norm_num),
  specialize h 2 3,
  rcases h ⟨h_mod3, h_mod5⟩ with ⟨m, rfl⟩,
  linarith,
end

end smallest_number_mod_conditions_l434_434728


namespace eccentricity_of_ellipse_l434_434903

theorem eccentricity_of_ellipse (e : ℝ) (h_e : e = 1 / 3) (m : ℝ) :
    (∃ a b : ℝ, a^2 = m + 2 ∧ b^2 = 4 ∧ e = (real.sqrt (m - 2)) / a) ∨
    (∃ a b : ℝ, a^2 = 4 ∧ b^2 = m + 2 ∧ e = (real.sqrt (2 - m)) / 2) →
    m = 5 / 2 ∨ m = 14 / 9 :=
by
  sorry

end eccentricity_of_ellipse_l434_434903


namespace correct_calculation_result_l434_434060

theorem correct_calculation_result (n : ℤ) (h1 : n - 59 = 43) : n - 46 = 56 :=
by {
  sorry -- Proof is omitted
}

end correct_calculation_result_l434_434060


namespace seq_2012_is_neg_2012_l434_434610

def sequence (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n : ℤ) else -(n : ℤ)

theorem seq_2012_is_neg_2012 :
  sequence 2012 = -2012 :=
by
  sorry

end seq_2012_is_neg_2012_l434_434610


namespace part1_part2_part3_l434_434913

open Real Function

def f (x : ℝ) : ℝ := sin x ^ 2 - cos x ^ 2 - 2 * sqrt 3 * sin x * cos x

theorem part1 : f (2 * π / 3) = 2 :=
by
  sorry

theorem part2 : ∀ x : ℝ, f (x + π) = f x :=
by
  sorry

theorem part3 : ∀ k : ℤ, ∀ x : ℝ, k * π + (π / 6) ≤ x ∧ x ≤ k * π + (2 * π / 3) → 
  MonotoneOn f (Icc (k * π + (π / 6)) (k * π + (2 * π / 3))) :=
by
  sorry

end part1_part2_part3_l434_434913


namespace room_length_l434_434127

theorem room_length (w : ℕ) (h : w = 19) (l : ℕ) (hl : l = w + 1) : l = 20 :=
by 
  rw [h, hl]
  rfl

end room_length_l434_434127


namespace tangerines_left_l434_434928

theorem tangerines_left (total_tangerines : ℕ) (given_tangerines : ℕ) (tangerines_left : ℕ) :
  total_tangerines = 27 → given_tangerines = 18 → tangerines_left = 9 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end tangerines_left_l434_434928


namespace cos_diff_identity_l434_434021

theorem cos_diff_identity 
  (α β : ℝ)
  (h1 : sin α - sin β = 1 - (sqrt 3) / 2)
  (h2 : cos α - cos β = 1 / 2) :
  cos (α - β) = (sqrt 3) / 2 :=
by
  sorry

end cos_diff_identity_l434_434021


namespace smallest_positive_period_l434_434844

def f (x : ℝ) := (2 * Real.sin x + 1) / (3 * Real.sin x - 5)

theorem smallest_positive_period : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', (∀ x, f (x + T') = f x) → T ≤ T') := by
  use 2 * Real.pi
  sorry

end smallest_positive_period_l434_434844


namespace evaluate_expression_l434_434394

theorem evaluate_expression : (8^5 / 8^2) * 2^12 = 2^21 := by
  sorry

end evaluate_expression_l434_434394


namespace min_max_f_l434_434239

noncomputable def f (x y z : ℝ) : ℝ :=
  cos x * sin y + cos y * sin z + cos z * sin x

theorem min_max_f (x y z : ℝ) : -3/2 ≤ f x y z ∧ f x y z ≤ 3/2 :=
sorry

end min_max_f_l434_434239


namespace incorrect_intersections_l434_434384

theorem incorrect_intersections :
  (∃ x, (x = x ∧ x = Real.sqrt (x + 2)) ↔ x = 1 ∨ x = 2) →
  (∃ x, (x^2 - 3 * x + 2 = 2 ∧ x = 2) ↔ x = 1 ∨ x = 2) →
  (∃ x, (Real.sin x = 3 * x - 4 ∧ x = 2) ↔ x = 1 ∨ x = 2) → False :=
by {
  sorry
}

end incorrect_intersections_l434_434384


namespace circle_diameter_l434_434271

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l434_434271


namespace square_root_domain_l434_434533

theorem square_root_domain (x : ℝ) : (∃ y, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end square_root_domain_l434_434533


namespace round_to_nearest_hundredth_l434_434162

theorem round_to_nearest_hundredth : ∃ (r : ℝ), r = 0.0375 ∧ (to_nearest_hundredth r) = 0.04 :=
begin
  sorry
end

end round_to_nearest_hundredth_l434_434162


namespace simplify_fraction_l434_434168

-- Define the given variables and their assigned values.
variable (b : ℕ)
variable (b_eq : b = 2)

-- State the theorem we want to prove
theorem simplify_fraction (b : ℕ) (h : b = 2) : 
  15 * b ^ 4 / (75 * b ^ 3) = 2 / 5 :=
by
  -- sorry indicates where the proof would be written.
  sorry

end simplify_fraction_l434_434168


namespace circle_diameter_l434_434282

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434282


namespace eval_expression_l434_434849

theorem eval_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ((a + b + c + d)⁻¹ *
   (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) *
   (ab + bc + cd + da + ac + bd)⁻¹ *
   ((ab)⁻¹ + (bc)⁻¹ + (cd)⁻¹ + (da)⁻¹ + (ac)⁻¹ + (bd)⁻¹)) = (abcd)⁻² :=
sorry

end eval_expression_l434_434849


namespace intersection_of_M_and_N_l434_434600

open Set Nat

def M := {x : ℕ | x > 0 ∧ x ≤ 2}
def N := {2, 6}

theorem intersection_of_M_and_N : M ∩ N = {2} := by
  sorry

end intersection_of_M_and_N_l434_434600


namespace min_value_of_2a_plus_b_l434_434578

variable (a b : ℝ)

def condition := a > 0 ∧ b > 0 ∧ a - 2 * a * b + b = 0

-- Define what needs to be proved
theorem min_value_of_2a_plus_b (h : condition a b) : ∃ a b : ℝ, 2 * a + b = (3 / 2) + Real.sqrt 2 :=
sorry

end min_value_of_2a_plus_b_l434_434578


namespace numerator_denominator_zero_count_valid_solutions_problem_solution_l434_434471

theorem numerator_denominator_zero (x : ℕ) :
  (\prod k in Finset.range 120 (+ 1), (x - k)) = 0 ↔ x ∈ Finset.range 120 (+ 1) :=
sorry

theorem count_valid_solutions :
  (Finset.range 120).card - 11 = 109 :=
sorry

theorem problem_solution :
  ∃ x, numerator_denominator_zero x ∧ count_valid_solutions := 
sorry

end numerator_denominator_zero_count_valid_solutions_problem_solution_l434_434471


namespace intersect_tetrahedron_with_rhombus_l434_434560

noncomputable theory
open_locale classical

-- Define a tetrahedron and a plane in 3D space
structure Point3D (F : Type*) [Field F] :=
(x y z : F)

structure Tetrahedron (F : Type*) [Field F] :=
(A B C D : Point3D F)

structure Plane (F : Type*) [Field F] :=
(normal : Point3D F)
(constant : F)

-- Define what it means for a cross-section of a tetrahedron to be a rhombus
def is_rhombus {F : Type*} [Field F] (points : Finset (Point3D F)) : Prop :=
  points.card = 4 ∧ ∃ (a b c d : Point3D F), 
    points = {a, b, c, d} ∧
    ∀ (u v : Point3D F), u ∈ points ∧ v ∈ points ∧ u ≠ v →
      ∃ (length : F), 
        (u = v + length * a * b ∨ u = v + length * b * c ∨ u = v + length * c * d ∨ u = v + length * d * a)

-- The main theorem
theorem intersect_tetrahedron_with_rhombus {F : Type*} [Field F] 
  (T : Tetrahedron F) :
  ∃ P : Plane F, ∃ cross_section : Finset (Point3D F),
    is_rhombus cross_section ∧
    ∀ p ∈ cross_section, ∃ λ : F, 
      p = λ * P.normal :=
sorry

end intersect_tetrahedron_with_rhombus_l434_434560


namespace sandy_total_sums_l434_434163

theorem sandy_total_sums (C I : ℕ) (h1 : C = 22) (h2 : 3 * C - 2 * I = 50) :
  C + I = 30 :=
sorry

end sandy_total_sums_l434_434163


namespace opposite_negative_nine_l434_434675

theorem opposite_negative_nine : 
  (∃ (y : ℤ), -9 + y = 0 ∧ y = 9) :=
by sorry

end opposite_negative_nine_l434_434675


namespace cars_meet_distance_from_midpoint_l434_434716

theorem cars_meet_distance_from_midpoint (dist_AB : ℕ) (speed_A : ℕ) (speed_B : ℕ) (half_dist_C : ℕ) : 
  dist_AB = 220 → speed_A = 60 → speed_B = 80 → half_dist_C = 110 → 
  let relative_speed := speed_A + speed_B in
  let time_to_meet := (dist_AB : ℚ) / relative_speed in
  let dist_A_meet := (speed_A : ℚ) * time_to_meet in
  let dist_from_C := half_dist_C - dist_A_meet in
  dist_from_C ≈ 15.71 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  let relative_speed := 140
  let time_to_meet := (220 : ℚ) / relative_speed
  let dist_A_meet := (60 : ℚ) * time_to_meet
  let dist_from_C := 110 - dist_A_meet
  have : dist_from_C = 110 - 94.2857142857 := by norm_num
  norm_num at this
  exact this
  sorry

end cars_meet_distance_from_midpoint_l434_434716


namespace regular_tetrahedron_ratio_l434_434548

/-- In plane geometry, the ratio of the radius of the circumscribed circle to the 
inscribed circle of an equilateral triangle is 2:1, --/
def ratio_radii_equilateral_triangle : ℚ := 2 / 1

/-- In space geometry, we study the relationship between the radii of the circumscribed
sphere and the inscribed sphere of a regular tetrahedron. --/
def ratio_radii_regular_tetrahedron : ℚ := 3 / 1

/-- Prove the ratio of the radius of the circumscribed sphere to the inscribed sphere
of a regular tetrahedron is 3 : 1, given the ratio is 2 : 1 for the equilateral triangle. --/
theorem regular_tetrahedron_ratio : 
  ratio_radii_equilateral_triangle = 2 / 1 → 
  ratio_radii_regular_tetrahedron = 3 / 1 :=
by
  sorry

end regular_tetrahedron_ratio_l434_434548


namespace minimal_sum_in_triangle_l434_434086

def is_minimal_sum (triangle_BAC : Triangle) (A B C D E : Point) : Prop :=
∠ triangle_BAC.A triangle_BAC.B triangle_BAC.C = 60 ∧
triangle_BAC.side_AB = 8 ∧
triangle_BAC.side_AC = 12 ∧
lies_on_segment D triangle_BAC.A triangle_BAC.B ∧
lies_on_segment E triangle_BAC.A triangle_BAC.C ∧
BE + DE + CD = sqrt (304)

theorem minimal_sum_in_triangle (triangle_BAC : Triangle) (A B C D E : Point) (h : is_minimal_sum triangle_BAC A B C D E) : BE + DE + CD = sqrt (304) :=
sorry

end minimal_sum_in_triangle_l434_434086


namespace triangle_area_is_168_l434_434495

def curve (x : ℝ) : ℝ :=
  (x - 4)^2 * (x + 3)

noncomputable def x_intercepts : set ℝ :=
  {x | curve x = 0}

noncomputable def y_intercept : ℝ :=
  curve 0

theorem triangle_area_is_168 :
  let base := 7 in
  let height := y_intercept in
  let area := (1 / 2) * base * height in
  area = 168 :=
by
  sorry

end triangle_area_is_168_l434_434495


namespace f_of_7_is_neg2_l434_434884

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 2 then 2 * x^2
else if x + 4 = 7 then f (x - 4)
else if ¬(0 < x ∧ x < 2) then -f (-x)
else 0 -- Undefined cases of f are filled in arbitrarily

theorem f_of_7_is_neg2 (x : ℝ) (h1 : ∀ x, f(x+4) = f(x)) (h2 : ∀ x, f(-x) = -f(x)) (h3 : ∀ x, 0 < x ∧ x < 2 → f(x) = 2*x^2) : f(7) = -2 :=
by
  sorry

end f_of_7_is_neg2_l434_434884


namespace circle_diameter_problem_circle_diameter_l434_434290

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l434_434290


namespace evaluate_inequality_false_l434_434396

theorem evaluate_inequality_false : 
    12.37 * ((lim (λ x, ((sqrt (3 * x - 5) - 1) / (x - 2))) 2) - (lim (λ x, ((cbrt (x + 1) - 1) / x)) 0)) < (Real.cos (Real.pi / 10)) = False :=
by
  have first_limit: lim (λ x, ((sqrt (3 * x - 5) - 1) / (x - 2))) 2 = (3 / 2) := sorry
  have second_limit: lim (λ x, ((cbrt (x + 1) - 1) / x)) 0 = (1 / 3) := sorry
  calc 
    12.37 * ((3 / 2) - (1 / 3)) = 12.37 * (9 / 6 - 2 / 6) : by rw [first_limit, second_limit]
    ... = 12.37 * (7 / 6)       : by norm_num
    ... = 12.37 * 1.1667        : by norm_num
    ... = 14.4                  : by norm_num
    ... < Real.cos (Real.pi / 10)= False : by norm_num  -- since 14.4 > 0.951

end evaluate_inequality_false_l434_434396


namespace last_two_digits_of_sum_of_factorials_l434_434219

theorem last_two_digits_of_sum_of_factorials :
  (∑ n in Finset.range 1000, n.fact) % 100 = 13 :=
by
  sorry

end last_two_digits_of_sum_of_factorials_l434_434219


namespace point_has_zero_measure_measure_disjoint_intervals_measure_union_intervals_rationals_have_zero_measure_measure_even_integer_part_measure_translation_invariant_measure_no_digit_9_has_zero_measure_homothety_measure_l434_434754

open MeasureTheory

-- 1. Prove that a point has zero measure.
theorem point_has_zero_measure (x : ℝ) : measure_of_Ixx volume (Ioo x x) = 0 :=
sorry

-- 2. What is the measure of [0, 1[ ∩ ]3/2, 4]?
theorem measure_disjoint_intervals : measure_of_Ixx volume (Ico 0 1 ∩ Ioc (3/2) 4) = 0 :=
sorry

-- 3. What is the measure of ∪_{n=0}^∞ [2^{-2n}, 2^{-2n-1}]
theorem measure_union_intervals : measure_of_Ixx volume (⋃ n, Icc (Real.exp (-(2*n))) (Real.exp (-(2*n) - 1))) = 2/3 :=
sorry

-- 4. Prove that ℚ has zero measure.
theorem rationals_have_zero_measure : measure_of_Ixx volume (⋃ q ∈ ℚ, Ioo q q) = 0 :=
sorry

-- 5. What is the measure of the set of numbers whose integer part is even?
def even_integer_part (x : ℝ) : Prop := ∃ k : ℤ, x ∈ Ico (2*k : ℝ) (2*k + 1)
theorem measure_even_integer_part : measure_of_Ixx volume {x | even_integer_part x} = ∞ :=
sorry

-- 6. Prove that the measure is invariant under translation.
theorem measure_translation_invariant (A : Set ℝ) (t : ℝ) : measurable_set A → measure_of_Ixx volume (A + t) = measure_of_Ixx volume A :=
sorry

-- 7. Prove that the set of numbers whose decimal representation does not contain the digit 9 has zero measure.
theorem measure_no_digit_9_has_zero_measure : measure_of_Ixx volume {x : ℝ | ¬ 9 ∈ (to_digits (floor x))} = 0 :=
sorry

-- 8. Prove that if A is a measurable set and λ is a real number, then λA has measure λ times the measure of A.
theorem homothety_measure (A : Set ℝ) (λ : ℝ) : measurable_set A → measure_of_Ixx volume (λ • A) = λ * measure_of_Ixx volume A :=
sorry

end point_has_zero_measure_measure_disjoint_intervals_measure_union_intervals_rationals_have_zero_measure_measure_even_integer_part_measure_translation_invariant_measure_no_digit_9_has_zero_measure_homothety_measure_l434_434754


namespace circle_diameter_l434_434281

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l434_434281


namespace ram_weight_increase_percentage_l434_434698

noncomputable def ram_original_weight (k : ℝ) := 2 * k
noncomputable def shyam_original_weight (k : ℝ) := 5 * k
noncomputable def total_original_weight (k : ℝ) := ram_original_weight k + shyam_original_weight k
noncomputable def total_weight_after_increase : ℝ := 82.8
noncomputable def increase_percentage : ℝ := 0.15

-- Given conditions as definitions
axiom ram_shyam_ratio (k : ℝ) : ram_original_weight k / shyam_original_weight k = 2 / 5
axiom total_weight_equation (k : ℝ) : total_weight_after_increase = total_original_weight k * (1 + increase_percentage)
axiom shyam_new_weight (k : ℝ) : shyam_original_weight k * 1.17 = shyam_original_weight k * 1.17

-- We need to prove the percentage increase in Ram's weight is approximately 10%
theorem ram_weight_increase_percentage {k : ℝ} (h1 : ram_shyam_ratio k)
    (h2 : total_weight_equation k)
    (h3 : shyam_new_weight k) : 
    (82.8 - shyam_original_weight k * 1.17 - ram_original_weight k) / ram_original_weight k * 100 ≈ 10 := sorry

end ram_weight_increase_percentage_l434_434698


namespace hyperbola_focus_asymptote_distance_l434_434664

theorem hyperbola_focus_asymptote_distance :
  let a := 2
  let b := √12 / 2
  let c := √(a^2 + b^2)
  let d := c / √(b^2 / a^2 + 1)
  (d = 2 * √3) := by
  sorry

end hyperbola_focus_asymptote_distance_l434_434664


namespace least_positive_a_exists_l434_434569

noncomputable def f (x a : ℤ) : ℤ := 5 * x ^ 13 + 13 * x ^ 5 + 9 * a * x

theorem least_positive_a_exists :
  ∃ a : ℕ, (∀ x : ℤ, 65 ∣ f x a) ∧ ∀ b : ℕ, (∀ x : ℤ, 65 ∣ f x b) → a ≤ b :=
sorry

end least_positive_a_exists_l434_434569


namespace animals_percentage_monkeys_l434_434354

theorem animals_percentage_monkeys (initial_monkeys : ℕ) (initial_birds : ℕ) (birds_eaten : ℕ) (final_monkeys : ℕ) (final_birds : ℕ) : 
  initial_monkeys = 6 → 
  initial_birds = 6 → 
  birds_eaten = 2 → 
  final_monkeys = initial_monkeys → 
  final_birds = initial_birds - birds_eaten → 
  (final_monkeys * 100 / (final_monkeys + final_birds) = 60) := 
by intros
   sorry

end animals_percentage_monkeys_l434_434354


namespace area_inside_C_outside_A_B_l434_434804

noncomputable def circleArea (r : ℝ) : ℝ := π * r * r

def circles_A_B_C (rA rB rC d_ac d_bc : ℝ) : Prop :=
  (rA = 1) ∧
  (rB = 1) ∧
  (rC = 2) ∧
  (d_ac = 2) ∧
  (d_bc = 2)

theorem area_inside_C_outside_A_B (rA rB rC d_ac d_bc : ℝ) (h : circles_A_B_C rA rB rC d_ac d_bc) :
  circleArea rC - (circleArea rA + circleArea rB) = 2 * π := by
  sorry

end area_inside_C_outside_A_B_l434_434804


namespace collinear_mnL_l434_434889

variables {α : Type*} [euclidean_geometry α]

open_locale classical

theorem collinear_mnL
  (A B C : Point α)
  (Γ : Circle α)
  (hA : A ∉ Γ)
  (tangent_AB : Tangent_Line Γ A B)
  (tangent_AC : Tangent_Line Γ A C)
  (P : Γ.minor_arc B C → Point α)
  (tangent_ΓP : ∀ P ∈ Γ.minor_arc B C, Tangent_Line Γ P (tangent_points_Γ_ΓP P))
  (U V : Point α)
  (hU : online U B P)
  (hV : online V C P)
  (M : ∀ P, Point α)
  (hM : ∀ P, perp_line P A B M P ∧ online M P (bisector_line_A_DV A D V))
  (N : ∀ P, Point α)
  (hN : ∀ P, perp_line P A C N P ∧ online N P (bisector_line_A_EU A E U))
  (L : ∀ (ABC : Triangle α), circumcenter ABC) :
  ∃ (L : Point α), ∀ P, collinear M N L :=  -- Collinearity of M, N, L is independent of the choice of P
sorry

end collinear_mnL_l434_434889


namespace most_economical_route_length_l434_434240

noncomputable def cableRouteLength 
  (riverWidth : ℝ) (upstreamDistance : ℝ) 
  (bankCost : ℝ) (waterCost : ℝ) : ℝ :=
  let b := 75.0 -- Distance along the shore from the direct opposite point of the plant
  let d_water := (b^2 + riverWidth^2).sqrt
  let d_shore := upstreamDistance - b
  d_water + d_shore

theorem most_economical_route_length :
  ∀ (riverWidth upstreamDistance bankCost waterCost : ℝ),
  riverWidth = 100 ∧ upstreamDistance = 500 ∧ bankCost = 9 ∧ waterCost = 15 →
  cableRouteLength riverWidth upstreamDistance bankCost waterCost = 550 :=
by
  intros riverWidth upstreamDistance bankCost waterCost h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  have h_width : riverWidth = 100 := h1
  have h_distance : upstreamDistance = 500 := h2
  have h_bank : bankCost = 9 := h3
  have h_water : waterCost = 15 := h4
  sorry

end most_economical_route_length_l434_434240


namespace proportion_is_88_percent_l434_434817

-- Define frequencies for different intervals
def frequencies : List (ℕ × ℕ) := [(17, 1), (18, 0), (19, 1), (20, 0), (21, 3), (22, 0), 
                                    (23, 3), (24, 0), (25, 18), (26, 0), (27, 10), (28, 0), 
                                    (29, 8), (30, 0), (31, 6)]

-- Define the sample size
def sample_size : ℕ := 50

-- Calculate the number of data points less than or equal to 31
def points_le_31 : ℕ := frequencies.map Prod.snd.filter (λ x => x ≤ 31).sum

-- Define the proportion of data less than or equal to 31
def proportion_le_31 : ℚ := (points_le_31 : ℚ) / (sample_size : ℚ)

theorem proportion_is_88_percent : proportion_le_31 = 0.88 := sorry

end proportion_is_88_percent_l434_434817


namespace circle_diameter_l434_434266

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l434_434266


namespace evaluate_expression_l434_434806

theorem evaluate_expression :
  -(12 * 2) - (3 * 2) + ((-18 / 3) * -4) = -6 := 
by
  sorry

end evaluate_expression_l434_434806


namespace avg_prime_factors_of_multiples_of_10_l434_434402

theorem avg_prime_factors_of_multiples_of_10 : 
  (2 + 5) / 2 = 3.5 :=
by
  -- The prime factors of 10 are 2 and 5.
  -- Therefore, the average of these prime factors is (2 + 5) / 2.
  sorry

end avg_prime_factors_of_multiples_of_10_l434_434402


namespace statement_A_statement_B_statement_D_l434_434868

section
variables (x t a : ℝ)

def f (x : ℝ) : ℝ := exp x - x
def g (x : ℝ) : ℝ := x - log x

-- Statement A
theorem statement_A : ∀ x > 0, ∃ t, g (exp x) = t ∧ strict_mono_incr_on (g ∘ exp) (set.Ioi 0) := sorry

-- Statement B
theorem statement_B : ∀ x > 1, ∀ a > 0, (a = 2 / real.exp 1 → f (a*x) ≥ f (log (x^2))) := sorry

-- Statement D
theorem statement_D : ∀ t > 2, ∀ x1 x2 > 0, (f x1 = t ∧ g x2 = t ∧ x2 > x1 → ∃ k, k = 1/exp 1 ∧ (real.log t) / (x2 - x1) ≤ k) := sorry
end

end statement_A_statement_B_statement_D_l434_434868


namespace find_x_values_l434_434439

open Real

theorem find_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h₁ : x + 1/y = 5) (h₂ : y + 1/x = 7/4) : 
  x = 4/7 ∨ x = 5 := 
by sorry

end find_x_values_l434_434439


namespace translate_coordinates_l434_434966

theorem translate_coordinates :
  ∀ (A B A' : ℤ × ℤ), 
  A = (-4, -1) → 
  B = (1, 1) → 
  A' = (-2, 2) → 
  let translation := (A'.1 - A.1, A'.2 - A.2)
  let B' := (B.1 + translation.1, B.2 + translation.2)
  B' = (3, 4) :=
by
  intros A B A' hA hB hA'
  simp only [Prod.mk.inj_iff] at *
  obtain ⟨hAx, hAy⟩ := hA
  obtain ⟨hBx, hBy⟩ := hB
  obtain ⟨hA'x, hA'y⟩ := hA'
  have translation := (A'.1 - A.1, A'.2 - A.2)
  have B' := (B.1 + translation.1, B.2 + translation.2)
  rw [hAx, hAy, hA'x, hA'y, hBx, hBy]
  have translation_eq : translation = (2, 3) := by simp [translation]
  rw [translation_eq]
  simp [B']
  exact rfl

end translate_coordinates_l434_434966


namespace cardinality_prod_set_l434_434131

theorem cardinality_prod_set :
  let P := {0, 1, 2}
  let Q := {1, 2, 3, 4}
  P.prod Q.card = 12 := sorry

end cardinality_prod_set_l434_434131
