import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SmoothMonotone
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Matrix.Finite
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.VectorBasis
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.Matrix.Basis
import Mathlib.LinearAlgebra.VectorSpace
import Mathlib.Probability
import Mathlib.Probability.Independence
import Mathlib.Tactic

namespace magnitude_a_plus_2b_l256_256877

variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions
variables (a b : V)
variable (ha : ∥a∥ = 1)
variable (hb : ∥b∥ = 2)
variable (hab : a - b = (sqrt 2 : ℝ) • ![1/sqrt 2, sqrt 3/2])

-- Prove that |a + 2b| = sqrt 17
theorem magnitude_a_plus_2b : ∥a + 2 • b∥ = sqrt 17 :=
sorry

end magnitude_a_plus_2b_l256_256877


namespace table_count_is_19_l256_256143

noncomputable def numberOfTables (t : ℕ) (s : ℕ) : Prop :=
  s = 8 * t ∧ 4 * s + 5 * t = 724

theorem table_count_is_19 : ∃ t, t = 19 ∧ ∃ s, numberOfTables t s :=
by
  use 19
  use (8 * 19)
  unfold numberOfTables
  simp
  sorry

end table_count_is_19_l256_256143


namespace seats_filled_percentage_l256_256907

theorem seats_filled_percentage (total_seats vacant_seats : ℕ) (h1 : total_seats = 600) (h2 : vacant_seats = 228) :
  ((total_seats - vacant_seats) / total_seats * 100 : ℝ) = 62 := by
  sorry

end seats_filled_percentage_l256_256907


namespace count_not_divides_in_range_is_33_l256_256184

-- Definition of product of proper divisors of n
def g (n : ℕ) : ℕ := (Finset.filter (λ d, d ≠ n ∧ d ∣ n) (Finset.range (n + 1))).prod id

-- Condition 1: Defines the range for n
def in_range (n : ℕ) : Prop := 2 ≤ n ∧ n ≤ 100

-- Condition 2: n does not divide g(n)
def not_divides (n : ℕ) : Prop := ¬ (n ∣ g n)

-- Statement of the problem
theorem count_not_divides_in_range_is_33 : (Finset.filter (λ n, in_range n ∧ not_divides n) (Finset.range 101)).card = 33 := by
  sorry

end count_not_divides_in_range_is_33_l256_256184


namespace movie_tickets_distribution_l256_256827

theorem movie_tickets_distribution :
  (∃ (tickets : Finset ℕ) (people : Finset String) (A B : String),
    let P := "Person"
    let A_name := "A"
    let B_name := "B"
    let others := [ "C", "D", "E" ].toFinset
    tickets = ({1, 2, 3, 4, 5} : Finset ℕ) ∧
    people = (others ∪ {A_name, B_name}) ∧
    ∀ (p : String), p ∈ people → ∃ t ∈ tickets, (p = A_name ∨ p = B_name → t + 1 ∈ tickets ∨ t - 1 ∈ tickets)) 
→ 48 := sorry

end movie_tickets_distribution_l256_256827


namespace projection_a_onto_b_l256_256095

def vector_a : ℝ × ℝ × ℝ := (1, 3, 0)
def vector_b : ℝ × ℝ × ℝ := (2, 1, 1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def projection_vector (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let scale := dot_product u v / (magnitude v) ^ 2
  (scale * v.1, scale * v.2, scale * v.3)

theorem projection_a_onto_b :
  projection_vector vector_a vector_b = (5 / 3, 5 / 6, 5 / 6) :=
by
  sorry

end projection_a_onto_b_l256_256095


namespace no_possible_values_of_k_l256_256390

theorem no_possible_values_of_k :
  ¬(∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p + q = 65) :=
by
  sorry

end no_possible_values_of_k_l256_256390


namespace find_area_of_triangle_APQ_l256_256625

/-
  Given an equilateral triangle ABC with side length 10,
  and points P and Q on sides AB and AC such that PQ = 4 and
  PQ is tangent to the incircle of ABC,
  prove that the area of triangle APQ is equal to 5 * sqrt 3 / 3.
-/

def area_of_triangle_APQ  : ℝ :=
  let side_length := 10
  let PQ_length := 4
  let APQ_area := (5 * Real.sqrt 3) / 3
  APQ_area

theorem find_area_of_triangle_APQ :
  ∃ (P Q : (fin 2) → ℝ) (APQ_area: ℝ),
  (P 0).dist (P 1) = PQ_length ∧ (Q 0).dist (Q 1) = PQ_length ∧
  APQ_area = area_of_triangle_APQ ∧ 
  APQ_area = (5 * Real.sqrt 3) / 3 :=
by
  sorry

end find_area_of_triangle_APQ_l256_256625


namespace max_n_geom_sequence_l256_256055

theorem max_n_geom_sequence (a : ℕ → ℝ) 
  (h_exp : ∃ q : ℝ, ∀ n, a (n+1) = a n * q)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1 / 4) :
  ∃ n : ℕ, (∀ m, m ≤ n → ∑ i in finset.range (m + 1), a i * a (i + 1) ≤ 21 / 2) ∧ 
           (n = 3) :=
by
  sorry

end max_n_geom_sequence_l256_256055


namespace total_tickets_sold_l256_256690

theorem total_tickets_sold (A C : ℕ) (hC : C = 16) (h1 : 3 * C = 48) (h2 : 5 * A + 3 * C = 178) : 
  A + C = 42 :=
by
  sorry

end total_tickets_sold_l256_256690


namespace math_problem_proof_l256_256414

noncomputable def numerator : ℝ :=
  (Finset.range 2016).sum (λ k, (2021 - (k + 1)) / (k + 1))

noncomputable def denominator : ℝ :=
  (Finset.range 2016).sum (λ j, 1 / (j + 3))

theorem math_problem_proof : numerator / denominator = 2021 :=
  sorry

end math_problem_proof_l256_256414


namespace train_length_is_correct_l256_256372

noncomputable def length_of_train (time_in_seconds : ℝ) (relative_speed : ℝ) : ℝ :=
  relative_speed * time_in_seconds

noncomputable def relative_speed_in_mps (speed_of_train_kmph : ℝ) (speed_of_man_kmph : ℝ) : ℝ :=
  (speed_of_train_kmph + speed_of_man_kmph) * (1000 / 3600)

theorem train_length_is_correct :
  let speed_of_train_kmph := 65.99424046076315
  let speed_of_man_kmph := 6
  let time_in_seconds := 6
  length_of_train time_in_seconds (relative_speed_in_mps speed_of_train_kmph speed_of_man_kmph) = 119.9904 := by
  sorry

end train_length_is_correct_l256_256372


namespace marble_total_eq_l256_256905

-- Define the variables and conditions
variable (r b g y : ℝ)

def condition1 : Prop := r = 1.30 * b
def condition2 : Prop := g = 1.70 * r
def condition3 : Prop := y = b + 40

-- Define the statement to prove the answer
theorem marble_total_eq (r b g y : ℝ)
  (h1 : condition1 r b g y)
  (h2 : condition2 r b g y)
  (h3 : condition3 r b g y) :
  r + b + g + y = 3.84615 * r + 40 :=
sorry

end marble_total_eq_l256_256905


namespace third_vertex_y_coordinate_in_first_quadrant_l256_256295

theorem third_vertex_y_coordinate_in_first_quadrant :
  ∃ y : ℝ, 
    (∀ (x y₁ y₂ : ℝ), 
     (x, y₁) = (2, 7) ∧ (x + 12, y₁) = (14, 7) ∧ 
     (√((x + 12 - x)^2 + (y₂ - y₁)^2) = 12) → 
     (y₂ = y₁ + 6 * √3)) ∧
    (y = 7 + 6 * √3) := 
sorry

end third_vertex_y_coordinate_in_first_quadrant_l256_256295


namespace hyperbola_condition_l256_256676

theorem hyperbola_condition (m : ℝ) : 
  (∃ a b : ℝ, a = m + 4 ∧ b = m - 3 ∧ (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)) ↔ m > 3 :=
sorry

end hyperbola_condition_l256_256676


namespace distinct_positive_integer_roots_l256_256744

theorem distinct_positive_integer_roots (m a b : ℤ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) (h4 : a + b = -m) (h5 : a * b = -m + 1) : m = -5 := 
by
  sorry

end distinct_positive_integer_roots_l256_256744


namespace hyperbola_proof_l256_256491

noncomputable def hyperbola : Type :=
  { e : ℝ × ℝ // 16 * (e.1)^2 - 9 * (e.2)^2 = 144 }

def real_axis_length (h : hyperbola) : ℝ := 2 * 3
def imaginary_axis_length (h : hyperbola) : ℝ := 2 * 4
def eccentricity (h : hyperbola) : ℝ := 5 / 3

noncomputable def parabola_C_equation (h : hyperbola) : ℝ → ℝ := 
  λ x, -12 * x

theorem hyperbola_proof :
  ∀ (h : hyperbola),
    real_axis_length h = 6 ∧
    imaginary_axis_length h = 8 ∧
    eccentricity h = 5 / 3 ∧
    (parabola_C_equation h = λ y, y^2 + 12 * y) :=
by
  intro h
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . exact sorry -- parabola part


end hyperbola_proof_l256_256491


namespace largest_divisor_of_product_l256_256382

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Definition of P, the product of the visible numbers when an 8-sided die is rolled
def P (excluded: ℕ) : ℕ :=
  factorial 8 / excluded

-- The main theorem to prove
theorem largest_divisor_of_product (excluded: ℕ) (h₁: 1 ≤ excluded) (h₂: excluded ≤ 8): 
  ∃ n, n = 192 ∧ ∀ k, k > 192 → ¬k ∣ P excluded :=
sorry

end largest_divisor_of_product_l256_256382


namespace smallest_integer_x_l256_256736

theorem smallest_integer_x (x : ℕ) : 27 ^ x > 3 ^ 24 ↔ x ≥ 9 :=
by {
  sorry,
}

end smallest_integer_x_l256_256736


namespace mr_bodhi_adds_twenty_sheep_l256_256215

def cows : ℕ := 20
def foxes : ℕ := 15
def zebras : ℕ := 3 * foxes
def required_total : ℕ := 100

def sheep := required_total - (cows + foxes + zebras)

theorem mr_bodhi_adds_twenty_sheep : sheep = 20 :=
by
  -- Proof for the theorem is not required and is thus replaced with sorry.
  sorry

end mr_bodhi_adds_twenty_sheep_l256_256215


namespace i_pow_6_eq_neg_1_complex_multiplication_complex_series_sum_l256_256661

noncomputable def i := Complex.I

-- Question 1
theorem i_pow_6_eq_neg_1 : i^6 = -1 := by
  -- Lean's Complex number support and tactics will go here
  sorry

-- Question 2
theorem complex_multiplication : (1 + i) * (3 - 4 * i) + i^5 = 7 := by
  -- Lean's Complex number support and tactics will go here
  sorry

-- Question 3
theorem complex_series_sum : ∑ n in Finset.range 2024, i^n = -1 := by
  -- Sum from 0 to 2023 (2024 terms)
  -- Lean's Complex number support and tactics will go here
  sorry

end i_pow_6_eq_neg_1_complex_multiplication_complex_series_sum_l256_256661


namespace sin_half_angle_product_lt_quarter_l256_256978

theorem sin_half_angle_product_lt_quarter (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h_sum : A + B + C = Real.pi) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by
  sorry

end sin_half_angle_product_lt_quarter_l256_256978


namespace find_area_of_triangle_APQ_l256_256623

/-
  Given an equilateral triangle ABC with side length 10,
  and points P and Q on sides AB and AC such that PQ = 4 and
  PQ is tangent to the incircle of ABC,
  prove that the area of triangle APQ is equal to 5 * sqrt 3 / 3.
-/

def area_of_triangle_APQ  : ℝ :=
  let side_length := 10
  let PQ_length := 4
  let APQ_area := (5 * Real.sqrt 3) / 3
  APQ_area

theorem find_area_of_triangle_APQ :
  ∃ (P Q : (fin 2) → ℝ) (APQ_area: ℝ),
  (P 0).dist (P 1) = PQ_length ∧ (Q 0).dist (Q 1) = PQ_length ∧
  APQ_area = area_of_triangle_APQ ∧ 
  APQ_area = (5 * Real.sqrt 3) / 3 :=
by
  sorry

end find_area_of_triangle_APQ_l256_256623


namespace problem_lemma_l256_256000

noncomputable def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def quadratic_roots_diff_abs (a b c : ℝ) : ℝ :=
  let disc := b^2 - 4 * a * c
  real.sqrt (4 * a * c) / (2 * a)

theorem problem_lemma :
  (quadratic_eq 5 (-11) (-14) 0 = 0) ∧
  (quadratic_roots_diff_abs 5 (-11) (-14) = real.sqrt (401) / 5) ∧
  (406 = 401 + 5) :=
by
  sorry

end problem_lemma_l256_256000


namespace dave_elevator_problem_l256_256405

noncomputable def elevator (n : ℕ) : ℚ := 
  let k := (2^(n-1) + 2^n - 1) / (3 * 2^(n-1))
  in k

theorem dave_elevator_problem : (482 : ℕ) -> 
  ∃ m n : ℕ, 
  (elevator 482 = m / n) ∧
  (Nat.coprime m n) ∧
  (m + n) % 1000 = 803 :=
  by
    sorry

end dave_elevator_problem_l256_256405


namespace sec_750_degrees_cos_periodic_l256_256012

noncomputable def cos_30_degrees : ℝ := Real.sqrt 3 / 2

theorem sec_750_degrees : Real.sec 750 = 2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos 750 = Real.cos 30 := by
    rw [← Real.cos_periodic 750 (2 * 360)]
  have h2 : Real.cos 30 = Real.sqrt 3 / 2 := by
    exact cos_30_degrees
  rw [Real.sec_eq_inv_cos, h1, h2]
  norm_num
  rw [mul_div_assoc 2 _ 2, Real.sqrt_div, Real.sqrt_mul_self_eq_abs, abs_of_pos]
  norm_num
  rw [div_div_eq_div_mul, mul_comm]
  norm_num
  sorry

namespace Real
noncomputable def sqrt (x : ℝ) : ℝ := sorry

noncomputable def sec (x : ℝ) : ℝ := 1 / cos x

noncomputable def cos (x : ℝ) : ℝ := sorry

theorem cos_periodic (x y : ℝ) : cos (x + y) = cos x := sorry
end Real

end sec_750_degrees_cos_periodic_l256_256012


namespace area_ratio_l256_256221
-- Import the necessary library for mathematics

-- Define the problem conditions and theorem
theorem area_ratio (ABCD A1 B1 C1 D1 : Quadrilateral) (AB CD : Segment) (p : ℝ)
  (hA1_on_AB : A1 ∈ AB) (hB1_on_AB : B1 ∈ AB) 
  (hC1_on_CD : C1 ∈ CD) (hD1_on_CD : D1 ∈ CD)
  (hAA1 : length (Segment.mk A A1) = p * length AB)
  (hBB1 : length (Segment.mk B B1) = p * length AB)
  (hCC1 : length (Segment.mk C C1) = p * length CD)
  (hDD1 : length (Segment.mk D D1) = p * length CD)
  (h_p : p < 0.5) :
  area_ratio (A1 B1 C1 D1) (ABCD) = 1 - 2 * p := 
by 
  sorry

end area_ratio_l256_256221


namespace handshake_parity_l256_256544

theorem handshake_parity (n : ℕ) (A : Fin n → Fin n → ℤ) (h_symm : ∀ i j : Fin n, A i j = A j i) (h_self : ∀ i : Fin n, A i i = 0) :
  Even (Finset.card {i : Fin n | Odd (Finset.sum (Finset.univ.filter (λ j, i ≠ j) (λ j, A i j)))}) :=
begin
  sorry
end

end handshake_parity_l256_256544


namespace goods_train_length_l256_256752

noncomputable def length_of_goods_train
  (speed_train_kmph : ℕ) (speed_platform_kmph : ℕ) (platform_length_m : ℕ) (time_s : ℕ) : ℕ :=
let speed_train_ms := speed_train_kmph * 1000 / 3600 in
let speed_platform_ms := speed_platform_kmph * 1000 / 3600 in
let relative_speed_ms := speed_train_ms + speed_platform_ms in
let total_distance_m := relative_speed_ms * time_s in
total_distance_m - platform_length_m

theorem goods_train_length
  (speed_train_kmph : ℕ) (speed_platform_kmph : ℕ) (platform_length_m : ℕ) (time_s : ℕ) :
  length_of_goods_train speed_train_kmph speed_platform_kmph platform_length_m time_s = 400 :=
by
  -- Conditions from the problem
  have h_speed_train : speed_train_kmph = 72 := rfl,
  have h_speed_platform : speed_platform_kmph = 18 := rfl,
  have h_platform_length : platform_length_m = 250 := rfl,
  have h_time : time_s = 26 := rfl,
  
  -- Substituting the conditions
  rw [h_speed_train, h_speed_platform, h_platform_length, h_time],
  
  -- Speed conversions
  have conversion1 : 72 * 1000 / 3600 = 20 := by norm_num,
  have conversion2 : 18 * 1000 / 3600 = 5 := by norm_num,
  rw [conversion1, conversion2],
  
  -- Calculation of relative speed and distance
  have rel_speed : 20 + 5 = 25 := by norm_num,
  rw rel_speed,
  have distance_covered : 25 * 26 = 650 := by norm_num,

  -- Calculation of train length
  have length_train_calc : 650 - 250 = 400 := by norm_num,
  rw length_train_calc,
  
  -- Prove the theorem
  exact rfl

end goods_train_length_l256_256752


namespace max_distance_from_point_on_ellipse_to_focus_l256_256486

theorem max_distance_from_point_on_ellipse_to_focus :
  let a := 4
  let b := Real.sqrt 7
  let c := Real.sqrt (a^2 - b^2)
  let f1 := (-c, 0)
  let ellipse (x y : ℝ) := (x^2) / 16 + (y^2) / 7 = 1
  ∀ x y, ellipse x y → |((x + c)^2 + y^2)^0.5| ≤ 7 :=
begin
  sorry
end

end max_distance_from_point_on_ellipse_to_focus_l256_256486


namespace clothing_value_is_correct_l256_256999

-- Define the value of the clothing to be C and the correct answer
def value_of_clothing (C : ℝ) : Prop :=
  (C + 2) = (7 / 12) * (C + 10)

-- Statement of the problem
theorem clothing_value_is_correct :
  ∃ (C : ℝ), value_of_clothing C ∧ C = 46 / 5 :=
by {
  sorry
}

end clothing_value_is_correct_l256_256999


namespace max_intersections_five_points_l256_256537

noncomputable def max_perpendicular_intersections (n : ℕ) : ℕ :=
  let num_lines := (Nat.choose n 2)
  let num_perpendiculars := n * (Nat.choose (n - 1) 2)
  let max_intersections := Nat.choose num_perpendiculars 2
  let adjust_perpendiculars := n * (Nat.choose (n - 1) 2)
  let adjust_triangles := (Nat.choose n 3) * (Nat.choose (n - 1) 2 - 1)
  let adjust_points := n * (Nat.choose (Nat.choose (n - 1) 2) 2)
  (max_intersections - adjust_perpendiculars - adjust_triangles - adjust_points)

theorem max_intersections_five_points : max_perpendicular_intersections 5 = 310 := by
  sorry

end max_intersections_five_points_l256_256537


namespace probability_event_A_probability_event_B_l256_256354

-- Define the conditions in Lean
def fair_die_rolls : List (ℕ × ℕ) := [(m, n) | m ← [1, 2, 3, 4, 5, 6], n ← [1, 2, 3, 4, 5, 6]]

def dot_product (a b : (ℕ × ℕ)) : Int :=
  (a.1 * b.1 - a.2 * b.2)

def event_A : ((ℕ × ℕ) → Prop) := λ a, (dot_product a (2, -2) > 0)

def distance_square (a : ℕ × ℕ) : ℕ :=
  a.1 * a.1 + a.2 * a.2

def event_B : ((ℕ × ℕ) → Prop) := λ a, (distance_square a ≤ 16)

-- Prove that the probability of A is 5/12
theorem probability_event_A : ∃ (p : ℝ), p = 5 / 12 := by
  sorry

-- Prove that the probability of event B is 2/9
theorem probability_event_B : ∃ (p : ℝ), p = 2 / 9 := by
  sorry

end probability_event_A_probability_event_B_l256_256354


namespace daily_evaporation_l256_256346

variable (initial_water : ℝ) (percentage_evaporated : ℝ) (days : ℕ)
variable (evaporation_amount : ℝ)

-- Given conditions
def conditions_met : Prop :=
  initial_water = 10 ∧ percentage_evaporated = 0.4 ∧ days = 50

-- Question: Prove the amount of water evaporated each day is 0.08
theorem daily_evaporation (h : conditions_met initial_water percentage_evaporated days) :
  evaporation_amount = (initial_water * percentage_evaporated) / days :=
sorry

end daily_evaporation_l256_256346


namespace coefficient_m5n5_in_mn_pow10_l256_256310

-- Definition of the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- The main theorem statement
theorem coefficient_m5n5_in_mn_pow10 : 
  (∃ c, (m + n) ^ 10 = c * m^5 * n^5 + ∑ (k ≠ 5), (binomial_coeff 10 k) * m^(10 - k) * n^k) → 
  c = 252 := 
by 
  sorry

end coefficient_m5n5_in_mn_pow10_l256_256310


namespace yoga_to_exercise_ratio_is_one_l256_256211

theorem yoga_to_exercise_ratio_is_one :
  ∀ (gym_ratio bicycle_ratio bicycle_time yoga_time : ℕ),
  gym_ratio = 2 → bicycle_ratio = 3 →
  bicycle_time = 12 → yoga_time = 20 →
  let gym_time := (gym_ratio * bicycle_time) / bicycle_ratio in
  let total_exercise_time := gym_time + bicycle_time in
  yoga_time / total_exercise_time = 1 :=
by
  intros gym_ratio bicycle_ratio bicycle_time yoga_time h_gym_ratio h_bicycle_ratio h_bicycle_time h_yoga_time;
  let gym_time := (gym_ratio * bicycle_time) / bicycle_ratio;
  let total_exercise_time := gym_time + bicycle_time;
  calc
    yoga_time / total_exercise_time = 20 / 20 : by sorry
    ... = 1 : by sorry

end yoga_to_exercise_ratio_is_one_l256_256211


namespace slope_of_parallel_line_l256_256314

theorem slope_of_parallel_line (x y : ℝ) (h : 5 * x - 3 * y = 9) : 
  ∃ m : ℝ, (y = m * x - 3) ∧ m = 5 / 3 :=
by 
  have h1 : -3 * y = -5 * x + 9 := by linarith
  have h2 : y = (5 / 3) * x - 3 := by 
    -- Divide both sides by -3
    field_simp [h1]
  use 5 / 3
  exact ⟨ h2, rfl ⟩

end slope_of_parallel_line_l256_256314


namespace palindrome_count_on_24_hour_clock_l256_256134

theorem palindrome_count_on_24_hour_clock : 
  let three_digit_palindromes := 9 * 6,
      four_digit_palindromes := 2 * 6 + 1 * 4
  in three_digit_palindromes + four_digit_palindromes = 70 :=
by
  let three_digit_palindromes := 9 * 6
  let four_digit_palindromes := 2 * 6 + 1 * 4
  show three_digit_palindromes + four_digit_palindromes = 70
  sorry

end palindrome_count_on_24_hour_clock_l256_256134


namespace intersection_of_A_and_B_l256_256472

def A : set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : set ℝ := {-1, 0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 2} :=
by sorry

end intersection_of_A_and_B_l256_256472


namespace path_traveled_by_A_l256_256230

-- Define the initial conditions
def RectangleABCD (A B C D : ℝ × ℝ) :=
  dist A B = 3 ∧ dist C D = 3 ∧ dist B C = 5 ∧ dist D A = 5

-- Define the transformations
def rotated90Clockwise (D : ℝ × ℝ) (A : ℝ × ℝ) (A' : ℝ × ℝ) : Prop :=
  -- 90-degree clockwise rotation moves point A to A'
  A' = (D.1 + D.2 - A.2, D.2 - D.1 + A.1)

def translated3AlongDC (D C A' : ℝ × ℝ) (A'' : ℝ × ℝ) : Prop :=
  -- Translation by 3 units along line DC moves point A' to A''
  A'' = (A'.1 - 3, A'.2)

-- Define the total path traveled
noncomputable def totalPathTraveled (rotatedPath translatedPath : ℝ) : ℝ :=
  rotatedPath + translatedPath

-- Prove the total path is 2.5*pi + 3
theorem path_traveled_by_A (A B C D A' A'' : ℝ × ℝ) (hRect : RectangleABCD A B C D) (hRotate : rotated90Clockwise D A A') (hTranslate : translated3AlongDC D C A' A'') :
  totalPathTraveled (2.5 * Real.pi) 3 = (2.5 * Real.pi + 3) := by
  sorry

end path_traveled_by_A_l256_256230


namespace equilateral_triangle_area_l256_256251

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 3 * Real.sqrt 2) : 
  let s := 2 * (h / (Real.sqrt 3)) in 
  let area := (s * h) / 2 in 
  area = 9 * Real.sqrt 2 :=
by
  -- Provided condition
  have h_alt : h = 3 * Real.sqrt 2 := h_eq
  
  -- Side length calculation
  let bm := h / (Real.sqrt 3)
  let s := 2 * bm
  
  -- Area calculation
  let area := (s * h) / 2

  -- The statement to prove
  show area = 9 * Real.sqrt 2
  sorry

end equilateral_triangle_area_l256_256251


namespace tan_half_alpha_l256_256845

theorem tan_half_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 24 / 25) : Real.tan (α / 2) = 3 / 4 :=
by
  sorry

end tan_half_alpha_l256_256845


namespace remainder_1125_1127_1129_div_12_l256_256323

theorem remainder_1125_1127_1129_div_12 :
  (1125 * 1127 * 1129) % 12 = 3 :=
by
  -- Proof can be written here
  sorry

end remainder_1125_1127_1129_div_12_l256_256323


namespace football_shaped_area_approx_l256_256985

-- Define the problem's conditions and question in Lean 4.
theorem football_shaped_area_approx :
  ∀ (PQRS : Type) [IsSquare PQRS] (P Q R S : PQRS),
  distance P Q = 3 →
  is_circle_centered_at S arc P X C →
  is_circle_centered_at R arc P Y C →
  let area := 2 * ((9 / 4) * Real.pi - (9 / 2)) in
  Real.to_decimals (area.to_float) 1 = 5.1 :=
by
  intros PQRS pqrs_h [P Q R S]
  intro PQ_eq_3
  intro circle_P_X_C
  intro circle_P_Y_C

  sorry

end football_shaped_area_approx_l256_256985


namespace sale_price_of_trouser_l256_256928

theorem sale_price_of_trouser : (100 - 0.70 * 100) = 30 := by
  sorry

end sale_price_of_trouser_l256_256928


namespace arrival_time_difference_l256_256883

theorem arrival_time_difference
  (d : ℝ) (r_H : ℝ) (r_A : ℝ) (h₁ : d = 2) (h₂ : r_H = 12) (h₃ : r_A = 6) :
  (d / r_A * 60) - (d / r_H * 60) = 10 :=
by
  sorry

end arrival_time_difference_l256_256883


namespace kaleb_total_points_l256_256917

theorem kaleb_total_points (points_first_half points_second_half : ℕ) (h1 : points_first_half = 43) (h2 : points_second_half = 23) : points_first_half + points_second_half = 66 :=
by {
  rw [h1, h2],
  exact rfl
}

end kaleb_total_points_l256_256917


namespace right_triangle_perimeter_l256_256364

noncomputable def perimeter_of_right_triangle (a b c : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter : 
  ∀ (a b c : ℝ), 
    (1/2) * a * b = 180 ∧ a = 30 ∧ a^2 + b^2 = c^2 
    → perimeter_of_right_triangle a b c = 42 + 2 * real.sqrt 261 :=
by
  intros a b c h
  sorry

end right_triangle_perimeter_l256_256364


namespace max_xy_max_xy_value_l256_256847

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 12) : x * y ≤ 3 :=
sorry

theorem max_xy_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 12) : x * y = 3 → x = 3 / 2 ∧ y = 2 :=
sorry

end max_xy_max_xy_value_l256_256847


namespace x_sq_plus_3x_minus_2_ge_zero_neg_x_sq_plus_3x_minus_2_lt_zero_l256_256653

theorem x_sq_plus_3x_minus_2_ge_zero (x : ℝ) (h : x ≥ 1) : x^2 + 3 * x - 2 ≥ 0 :=
sorry

theorem neg_x_sq_plus_3x_minus_2_lt_zero (x : ℝ) (h : x < 1) : x^2 + 3 * x - 2 < 0 :=
sorry

end x_sq_plus_3x_minus_2_ge_zero_neg_x_sq_plus_3x_minus_2_lt_zero_l256_256653


namespace expression_equals_x_minus_2_l256_256415

theorem expression_equals_x_minus_2 {x : ℝ} (h : x ≠ 2) : 
  let y := (x^2 - 4*x + 4) / (x - 2) in y = x - 2 :=
by 
  let y := (x^2 - 4*x + 4) / (x - 2)
  sorry

end expression_equals_x_minus_2_l256_256415


namespace min_text_length_l256_256924

theorem min_text_length : ∃ (L : ℕ), (∀ x : ℕ, 0.105 * (L : ℝ) < (x : ℝ) ∧ (x : ℝ) < 0.11 * (L : ℝ)) → L = 19 :=
by
  sorry

end min_text_length_l256_256924


namespace angle_BAC_measure_l256_256348

theorem angle_BAC_measure 
(∠AOB ∠BOC : ℝ)
(hAOB : ∠AOB = 120)
(hBOC: ∠BOC = 95) 
: ∠BAC = 47.5 := 
by
  /- Conditions: 
    ∠AOB = 120°
    ∠BOC = 95° 
  -/
  sorry

end angle_BAC_measure_l256_256348


namespace find_x_when_y_is_72_l256_256195

theorem find_x_when_y_is_72 
  (x y : ℝ) (k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_const : ∀ x y, 0 < x → 0 < y → x^2 * y = k)
  (h_initial : 9 * 8 = k)
  (h_y_72 : y = 72)
  (h_x2_factor : x^2 = 4 * 9) :
  x = 1 :=
sorry

end find_x_when_y_is_72_l256_256195


namespace number_of_roots_of_f_eq_0_l256_256275

def f (x : ℝ) : ℝ := x - sin x

theorem number_of_roots_of_f_eq_0 : ∃! (x : ℝ), f x = 0 :=
sorry

end number_of_roots_of_f_eq_0_l256_256275


namespace equilateral_triangle_area_APQ_l256_256645

theorem equilateral_triangle_area_APQ (ABC : Triangle) 
  (h_eq : is_equilateral ABC)
  (h_side : ABC.sides = (10, 10, 10)) 
  (P Q : Point) 
  (hP : P ∈ segment ABC.A ABC.B) 
  (hQ : Q ∈ segment ABC.A ABC.C) 
  (h_tangent : is_tangent (segment P Q) ABC.incircle) 
  (hPQ : segment.length P Q = 4) : 
  area (triangle ABC.A P Q) = 5 / sqrt 3 :=
by 
  sorry

end equilateral_triangle_area_APQ_l256_256645


namespace complex_pure_imaginary_l256_256850

theorem complex_pure_imaginary (a : ℝ) : (↑a + Complex.I) / (1 - Complex.I) = 0 + b * Complex.I → a = 1 :=
by
  intro h
  -- Proof content here
  sorry

end complex_pure_imaginary_l256_256850


namespace nitin_rank_from_first_l256_256618

theorem nitin_rank_from_first (total_students : ℕ) (rank_from_last : ℕ) 
    (h1 : total_students = 58) (h2 : rank_from_last = 34) : 
    (total_students - rank_from_last + 1 = 25) :=
by 
  rw [h1, h2]
  -- The followed steps are skipped, fulfilling the theorem without proof.
  sorry

end nitin_rank_from_first_l256_256618


namespace smallest_n_for_sum_of_cubes_l256_256022

theorem smallest_n_for_sum_of_cubes :
  ∃ (n : ℕ), n > 0 ∧ (∃ (x : ℕ → ℤ), (∑ i in Finset.range n, (x i)^3) = 2002^2002) ∧ n = 4 :=
by
  sorry

end smallest_n_for_sum_of_cubes_l256_256022


namespace find_f_a_l256_256175

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 4 * Real.logb 2 (-x) else abs (x^2 + a * x)

theorem find_f_a (a : ℝ) (h : a ≠ 0) (h1 : f a (f a (-Real.sqrt 2)) = 4) : f a a = 8 :=
sorry

end find_f_a_l256_256175


namespace second_volume_pages_l256_256697

theorem second_volume_pages (total_digits : ℕ) (page_diff : ℕ) (pages_in_volume : ℕ) : 
  total_digits = 888 → page_diff = 8 → pages_in_volume = (fun x => x + 8) → 
  let x := (((total_digits - 189) / 3) - 8) / 2 in 
  pages_in_volume = x + page_diff :=
by sorry

end second_volume_pages_l256_256697


namespace cubes_mod_l256_256297

theorem cubes_mod (a : ℕ → ℕ) (n : ℕ) (h : ∑ i in finset.range n, a i = 1996):
  (∑ i in finset.range n, (a i) ^ 3) % 6 = 4 := 
by
  sorry

end cubes_mod_l256_256297


namespace amount_of_brown_paint_l256_256698

-- Definition of the conditions
def white_paint : ℕ := 20
def green_paint : ℕ := 15
def total_paint : ℕ := 69

-- Theorem statement for the amount of brown paint
theorem amount_of_brown_paint : (total_paint - (white_paint + green_paint)) = 34 :=
by
  sorry

end amount_of_brown_paint_l256_256698


namespace parabola_hyperbola_p_value_l256_256871

theorem parabola_hyperbola_p_value:
  (∃ (p : ℝ), p > 0 ∧ (∀ x y, y^2 = 2 * p * x → (exists directrix, 
    directrix.x = p/2 → (directrix.x = 2 → p = 4)))) :=
sorry

end parabola_hyperbola_p_value_l256_256871


namespace coordinates_of_P_60_l256_256836

-- Define the sequence of points P_n where P_n is represented as (x, y)
def sequence : Nat → (Nat × Nat)
| 0     => (0, 0)  -- Not used, just an initializer.
| 1     => (1, 1)
| n + 2 =>
  let k := Nat.find (fun k => ((k + 1) * k) / 2 ≥ n + 1) - 1
  let total := (k * (k + 1)) / 2
  (n - total, k - (n - total) + 1)

-- Define the coordinates of P_60
def P_60_coordinates : (Nat × Nat) := sequence 60

-- Proposition to be proved
theorem coordinates_of_P_60 : P_60_coordinates = (5, 7) :=
by
  sorry

end coordinates_of_P_60_l256_256836


namespace joshua_miles_ratio_l256_256613

-- Definitions corresponding to conditions
def mitch_macarons : ℕ := 20
def joshua_extra : ℕ := 6
def total_kids : ℕ := 68
def macarons_per_kid : ℕ := 2

-- Variables for unspecified amounts
variable (M : ℕ) -- number of macarons Miles made

-- Calculations for Joshua and Renz's macarons based on given conditions
def joshua_macarons := mitch_macarons + joshua_extra
def renz_macarons := (3 * M) / 4 - 1

-- Total macarons calculation
def total_macarons := mitch_macarons + joshua_macarons + renz_macarons + M

-- Proof statement: Showing the ratio of number of macarons Joshua made to the number of macarons Miles made
theorem joshua_miles_ratio : (total_macarons = total_kids * macarons_per_kid) → (joshua_macarons : ℚ) / (M : ℚ) = 1 / 2 :=
by
  sorry

end joshua_miles_ratio_l256_256613


namespace quadrilateral_problem_l256_256657

-- Define the conditions in the problem
variables (A B C D E : ℝ) (AB BC : ℝ)

-- Define quadrilateral ABCD with right angles at B and C
def quadrilateral_right_angles_at_B_and_C 
  (right_angle_B : (∠ A B C) = π / 2)
  (right_angle_C : (∠ B C D) = π / 2) : Prop := 
true

-- Define similarity of triangles: ABC ~ BCD and ABC ~ CEB
def triangles_similar (ABC BCD : Triangle)
  (similarityABC_BCD : ABC ∼ BCD)
  (similarityABC_CEB : ABC ∼ (C E B)) : Prop := 
true

-- Define the condition: AB > BC
def AB_greater_BC (h : AB > BC) : Prop := 
true

-- Define the condition: area of AED is 9 times the area of CEB
def area_condition 
  (area_AED : ℚ) (area_CEB : ℚ) 
  (h : area_AED = 9 * area_CEB) : Prop := 
true

-- Define the final statement for the problem
theorem quadrilateral_problem
  (right_angle_B : (∠ A B C) = π / 2)
  (right_angle_C : (∠ B C D) = π / 2)
  (similarityABC_BCD : ΔABC ∼ ΔBCD)
  (similarityABC_CEB : ΔABC ∼ ΔCEB)
  (h1 : AB > BC)
  (area_AED : ℚ)
  (area_CEB : ℚ)
  (h2 : area_AED = 9 * area_CEB) : 
  AB / BC = 1 + 2 * sqrt(3) := 
sorry

end quadrilateral_problem_l256_256657


namespace hillary_descending_rate_correct_l256_256507

-- Define the conditions in Lean
def base_to_summit := 5000 -- height from base camp to the summit
def departure_time := 6 -- departure time in hours after midnight (6:00)
def summit_time_hillary := 5 -- time taken by Hillary to reach 1000 ft short of the summit
def passing_time := 12 -- time when Hillary and Eddy pass each other (12:00)
def climb_rate_hillary := 800 -- Hillary's climbing rate in ft/hr
def climb_rate_eddy := 500 -- Eddy's climbing rate in ft/hr
def stop_short := 1000 -- distance short of the summit Hillary stops at

-- Define the correct answer based on the conditions
def descending_rate_hillary := 1000 -- Hillary's descending rate in ft/hr

-- Create the theorem to prove Hillary's descending rate
theorem hillary_descending_rate_correct (base_to_summit departure_time summit_time_hillary passing_time climb_rate_hillary climb_rate_eddy stop_short descending_rate_hillary : ℕ) :
  (descending_rate_hillary = 1000) :=
sorry

end hillary_descending_rate_correct_l256_256507


namespace circles_intersect_l256_256411

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def radius (r : ℝ) : ℝ := r

def center_of_circle (h k : ℝ) : ℝ × ℝ := (h, k)

def circle_intersect (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  let d := distance c1 c2 in
  d < r1 + r2 ∧ d > abs (r1 - r2)

theorem circles_intersect :
  let C1 := center_of_circle (-2) 2,
      R1 := radius 2,
      C2 := center_of_circle 2 5,
      R2 := radius 4 in
  circle_intersect C1 C2 R1 R2 :=
by {
  sorry
}

end circles_intersect_l256_256411


namespace cut_corner_results_l256_256296

noncomputable def Cube : Type := {
  faces : Fin 6,
  vertices : Fin 8,
  edges : Fin 12
}

noncomputable def cutCorner (c : Cube) : Type := {
  faces : Fin (c.faces + 3),
  vertices : Fin (c.vertices - 1 + 3)
}

theorem cut_corner_results (c : Cube) :
  let newShape := cutCorner c in
  newShape.faces = Fin 9 ∧ newShape.vertices = Fin 10 :=
by sorry

end cut_corner_results_l256_256296


namespace count_divisible_by_3_l256_256108

-- Define the sequence
def seq (n : ℕ) : ℤ :=
  10^n + 2

-- Define the property of being divisible by 3
def divisible_by_3 (n : ℕ) : Prop :=
  seq(n) % 3 = 0

-- Prove the equivalent problem
theorem count_divisible_by_3 : 
  (∀ n : ℕ, n < 1500 → divisible_by_3 n) ∧ (∑ n in finset.range 1500, if divisible_by_3 n then 1 else 0) = 1500 := 
by
  sorry

end count_divisible_by_3_l256_256108


namespace segment_length_E_E_l256_256293

-- Define the points D, E, F, and E'
structure Point where
  x : ℝ
  y : ℝ

def D : Point := {x := -6, y := 1}
def E : Point := {x := 2, y := 5}
def F : Point := {x := -4, y := 3}
def E' : Point := {x := 2, y := -5}

-- Define the distance formula for vertical distance
def distance (p1 p2 : Point) : ℝ :=
  real.abs (p2.y - p1.y)

-- State the theorem
theorem segment_length_E_E' : distance E E' = 10 :=
  sorry

end segment_length_E_E_l256_256293


namespace nonnegative_integers_in_form_l256_256510

theorem nonnegative_integers_in_form :
  ∃ (f : ℕ → ℕ), (∀ i : ℕ, 0 ≤ i ∧ i ≤ 8 → f i ∈ {0, 1, 2}) ∧
  ∃ n, n = ∑ i in finset.range 9, f i * (3 ^ i) ∧ n = 3 ^ 9 :=
by
  sorry

end nonnegative_integers_in_form_l256_256510


namespace quadratic_distinct_real_roots_no_opposite_roots_k_l256_256075

# First problem statement: finding the range of values for k for which the given quadratic equation has two distinct real roots
theorem quadratic_distinct_real_roots {k : ℝ} : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ k^2 * x1^2 + (2 * k - 1) * x1 + 1 = 0 ∧ k^2 * x2^2 +(2 * k - 1) * x2 + 1 = 0) ↔ (k < 1/4 ∧ k ≠ 0) :=
sorry

# Second problem statement: showing there is no k such that the roots are opposite numbers
theorem no_opposite_roots_k {k : ℝ} :
  ¬(∃ x1 x2 : ℝ, x1 ≠ x2 ∧ k^2 * x1^2 + (2 * k - 1) * x1 + 1 = 0 ∧ k^2 * x2^2 +(2 * k - 1) * x2 + 1 = 0 ∧ x1 + x2 = 0) :=
sorry

end quadratic_distinct_real_roots_no_opposite_roots_k_l256_256075


namespace pencil_eraser_cost_l256_256976

variable (p e : ℕ)

theorem pencil_eraser_cost
  (h1 : 15 * p + 5 * e = 125)
  (h2 : p > e)
  (h3 : p > 0)
  (h4 : e > 0) :
  p + e = 11 :=
sorry

end pencil_eraser_cost_l256_256976


namespace find_sum_coefficients_find_difference_of_squares_l256_256457

variables {a0 a1 a2 a3 a4 : ℤ}

def poly_expansion (x : ℤ) : ℤ := a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4

theorem find_sum_coefficients (h0 : poly_expansion 0 = (-3)^4) (h1 : poly_expansion 1 = (2*1 - 3)^4) :
  a1 + a2 + a3 + a4 = -80 := sorry

theorem find_difference_of_squares (h1 : poly_expansion 1 = (2*1 - 3)^4) (h_1 : poly_expansion (-1) = (2* (-1) - 3)^4) :
  (a0 + a2 + a4)^2 - (a1 + a3)^2 = 625 := sorry

end find_sum_coefficients_find_difference_of_squares_l256_256457


namespace carnival_masks_costumes_min_l256_256139

theorem carnival_masks_costumes_min : 
  ∀ n x : ℕ, (n = 42) → (3 * n / 7 = 18) → (5 * n / 6 = 35) → 
  (n = 3 * n / 7 + 5 * n / 6 - x) → x = 11 :=
by
  intros n x h₁ h₂ h₃ h₄
  have hn := h₁
  have h_masks := h₂
  have h_costumes := h₃
  have h_eq := h₄
  sorry

end carnival_masks_costumes_min_l256_256139


namespace trigonometric_identity_l256_256463

theorem trigonometric_identity
  (α β : ℝ)
  (h₁ : (sin α) ^ 4 / (sin β) ^ 2 + (cos α) ^ 4 / (cos β) ^ 2 = 1)
  (h₂ : sin α ≠ 0 ∧ cos α ≠ 0 ∧ sin β ≠ 0 ∧ cos β ≠ 0) :
  (sin β) ^ 4 / (sin α) ^ 2 + (cos β) ^ 4 / (cos α) ^ 2 = 1 := by
  sorry

end trigonometric_identity_l256_256463


namespace arithmetic_sequence_l256_256607

variable {α : Type} [LinearOrderedField α]

/-- Sum of the first n terms of an arithmetic sequence is given by S_n -/
noncomputable def arithmetic_sum (a d : α) (n : ℕ) : α :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Proving that given a₃ = 7 and S₃ = 12, the 10th term a₁₀ = 28
theorem arithmetic_sequence (a d : α) (a₃ S₃ : α) (h₃ : a₃ = 7) (hS₃ : S₃ = 12) :
    let a₁ := (S₃ * 2 - 3 * a₃) / 3 in
    a₁ + 9 * d = 28 :=
by
  let a₁ := (S₃ * 2 - 3 * a₃) / 3
  let d := (a₃ - a₁) / 2
  have ha₃ : a₁ + 2 * d = a₃ := by linarith
  have hS₃' : 3 * a₁ + 3 * d = S₃ := by linarith
  have hS₃'' : 3 * a₁ + (3 * (a₃ - a₁) / 2) = S₃ := by rw [d]
  have : a₁ = (S₃ * 2 - 3 * a₃) / 3 := by sorry
  exact sorry

end arithmetic_sequence_l256_256607


namespace find_length_of_rect_patch_l256_256109

-- Define the conditions
def width_of_rect_patch : ℝ := 300
def perimeter_of_square_patch : ℝ := 4 * 700
def perimeter_of_rect_patch : ℝ := perimeter_of_square_patch / 2

-- Define what we need to prove
theorem find_length_of_rect_patch : 
  ∃ length_of_rect_patch : ℝ, 2 * (length_of_rect_patch + width_of_rect_patch) = perimeter_of_rect_patch ∧ length_of_rect_patch = 400 :=
by
  sorry

end find_length_of_rect_patch_l256_256109


namespace f_2012_eq_sin_l256_256593

noncomputable def f (x : ℝ) : ℝ := Real.sin x

def f_seq : ℕ → (ℝ → ℝ)
| 0     := f
| (n+1) := (derivative (f_seq n))

theorem f_2012_eq_sin (x : ℝ) : (f_seq 2012) x = Real.sin x := 
by sorry

end f_2012_eq_sin_l256_256593


namespace train_speed_is_correct_l256_256764

-- Definitions of the problem conditions
def train_length : ℝ := 360  -- length of the train in meters
def platform_length : ℝ := 240  -- length of the platform in meters
def time_to_pass : ℝ := 48  -- time in seconds for the train to pass the platform

-- Definition to calculate the speed of the train in km/hr
def calculate_speed (train_length : ℝ) (platform_length : ℝ) (time_to_pass : ℝ) : ℝ :=
  let total_distance := train_length + platform_length  -- total distance to travel in meters
  let speed_m_s := total_distance / time_to_pass  -- speed in meters per second
  speed_m_s * 3.6  -- convert speed to km/hr

-- Theorem stating the correct speed is 45 km/hr
theorem train_speed_is_correct : calculate_speed train_length platform_length time_to_pass = 45 := by
  -- The detailed proof goes here; skipping with sorry for now
  sorry

end train_speed_is_correct_l256_256764


namespace f_increasing_a_for_odd_function_range_f_when_odd_l256_256863

section
variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

theorem f_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f x1 a < f x2 a :=
by
  intros x1 x2 h
  unfold f
  sorry

theorem a_for_odd_function : ∃ a : ℝ, (∀ x : ℝ, f (-x) a = - f x a) :=
by
  use 1/2
  intro x
  unfold f
  sorry

theorem range_f_when_odd : ∀ x : ℝ, -1 / 2 < f x (1 / 2) ∧ f x (1 / 2) < 1 / 2 :=
by
  intro x
  unfold f
  sorry

end

end f_increasing_a_for_odd_function_range_f_when_odd_l256_256863


namespace car_speed_second_hour_l256_256280

theorem car_speed_second_hour (x : ℕ) 
  (h1 : 65) 
  (h2 : (65 + x) / 2 = 55) : 
  x = 45 := 
by
  sorry

end car_speed_second_hour_l256_256280


namespace projection_a_onto_b_l256_256093

def vector_a : ℝ × ℝ × ℝ := (1, 3, 0)
def vector_b : ℝ × ℝ × ℝ := (2, 1, 1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def projection_vector (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let scale := dot_product u v / (magnitude v) ^ 2
  (scale * v.1, scale * v.2, scale * v.3)

theorem projection_a_onto_b :
  projection_vector vector_a vector_b = (5 / 3, 5 / 6, 5 / 6) :=
by
  sorry

end projection_a_onto_b_l256_256093


namespace calculate_interest_rates_l256_256927

def first_account_inv := 4000
def second_account_inv := 8200
def third_account_inv := 5000
def total_interest := 1282
def second_account_premium := 0.015
def first_account_rate := 1159 / 22200 -- Approximation of r

theorem calculate_interest_rates :
  let r := first_account_rate in
  let second_account_rate := r + second_account_premium in
  let third_account_rate := 2 * r in
  (first_account_inv * r + second_account_inv * second_account_rate + third_account_inv * third_account_rate) = total_interest :=
by
  let r := first_account_rate
  let second_account_rate := r + second_account_premium
  let third_account_rate := 2 * r
  have h : first_account_inv * r + second_account_inv * second_account_rate + third_account_inv * third_account_rate = 
           first_account_inv * r + second_account_inv * (r + second_account_premium) + third_account_inv * (2 * r) := by sorry -- Simplification
  have h1 : first_account_inv * r + second_account_inv * (r + second_account_premium) + third_account_inv * (2 * r) =
            (4000 + 8200 + 10000) * r + 8200 * 0.015 := by sorry -- Grouping and simplifying
  have h2 : (4000 + 8200 + 10000) * r + 8200 * 0.015 = 22200 * r + 123 := by sorry -- Simplifying arithmetic
  have h3 : 22200 * r + 123 = total_interest := by sorry -- Solving for r
  show (first_account_inv * r + second_account_inv * second_account_rate + third_account_inv * third_account_rate) = total_interest by 
  exact h3

end calculate_interest_rates_l256_256927


namespace power_function_value_at_3_l256_256865

theorem power_function_value_at_3 :
  (∃ a : ℝ, ∃ x y : ℝ, y = log a (2 * x - 3) + 4 ∧ (x = 2 ∧ y = 4) ∧ f x = y) →
  ∀ f : ℝ → ℝ, ∃ α : ℝ, f x = x ^ α → f 3 = 9 :=
by
  intro h
  sorry

end power_function_value_at_3_l256_256865


namespace problem_TX_perp_TF_l256_256265

open EuclideanGeometry

theorem problem_TX_perp_TF (A B C D E F X T : Point)
  (h1 : AB ≠ AC)
  (h2 : incircle_tangent_to BC D)
  (h3 : incircle_tangent_to CA E)
  (h4 : incircle_tangent_to AB F)
  (h5 : meets_perpendicular_from D EF AB X)
  (h6 : second_intersection_circumcircles A E F ABC T) :
  TX ⊥ TF :=
sorry

end problem_TX_perp_TF_l256_256265


namespace smallest_possible_fourth_number_l256_256145

theorem smallest_possible_fourth_number :
  ∃ (n : ℕ), 
    let d₁ := 23 in 
    let d₂ := 45 in 
    let d₃ := 36 in 
    let sd₁ := 2 + 3 in 
    let sd₂ := 4 + 5 in 
    let sd₃ := 3 + 6 in 
    let total_sum := d₁ + d₂ + d₃ + n in 
    let digits_sum := sd₁ + sd₂ + sd₃ + (n / 10) + (n % 10) in 
    total_sum = 4 * digits_sum ∧ 
    (n ≥ 70 ∧ n < 80 ∧ 7 < n / 10 ∨ n ≥ 80 ∧ 8 < n / 10)  := 
sorry


end smallest_possible_fourth_number_l256_256145


namespace correlation_implies_slope_positive_l256_256073

-- Definition of the regression line
def regression_line (x y : ℝ) (b a : ℝ) : Prop :=
  y = b * x + a

-- Given conditions
variables (x y : ℝ)
variables (b a r : ℝ)

-- The statement of the proof problem
theorem correlation_implies_slope_positive (h1 : r > 0) (h2 : regression_line x y b a) : b > 0 :=
sorry

end correlation_implies_slope_positive_l256_256073


namespace casey_correct_result_l256_256789

variable (x : ℕ)

def incorrect_divide (x : ℕ) := x / 7
def incorrect_subtract (x : ℕ) := x - 20
def incorrect_result := 19

def reverse_subtract (x : ℕ) := x + 20
def reverse_divide (x : ℕ) := x * 7

def correct_multiply (x : ℕ) := x * 7
def correct_add (x : ℕ) := x + 20

theorem casey_correct_result (x : ℕ) (h : reverse_divide (reverse_subtract incorrect_result) = x) : correct_add (correct_multiply x) = 1931 :=
by
  sorry

end casey_correct_result_l256_256789


namespace floor_inequality_factorial_ratio_is_integer_l256_256238

-- Part 1: Defining the floor inequality for non-negative reals x, y
theorem floor_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  nat.floor (5 * x) + nat.floor (5 * y) ≥ nat.floor (3 * x + y) + nat.floor (x + 3 * y) :=
sorry

-- Part 2: Using the above result to show the factorial ratio is an integer for positive integers a, b
theorem factorial_ratio_is_integer (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℕ, k * (a! * b! * (3 * a + b)! * (a + 3 * b)!) = (5 * a)! * (5 * b)! :=
sorry

end floor_inequality_factorial_ratio_is_integer_l256_256238


namespace angles_of_triangle_ODC_l256_256220

-- Definitions of the conditions
def is_equilateral (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def midpoint (A M D : Point) : Prop :=
  dist A D = dist D M

def median (B N M O : Point) :=
  exists D E, is_midpoint B N D ∧ is_midpoint N M E ∧ line B M O D E

variables (A B C M N O D : Point)

-- Given conditions
axiom equilateral_triangle : is_equilateral A B C
axiom point_M_on_extension : collinear B C M ∧ ∃ C', dist B C' < dist B C
axiom line_parallel_to_AC : parallel (line M N) (line A C)
axiom N_on_extension_of_AB : collinear A B N ∧ ∃ A', dist A B' < dist A B
axiom medians_intersect_O : median B N M O
axiom D_is_midpoint_AM : midpoint A M D

-- Goal
theorem angles_of_triangle_ODC : 
  ∃ α β γ, α = 30 ∧ β = 60 ∧ γ = 90 ∧ sum_of_angles α β γ = 180 :=
sorry

end angles_of_triangle_ODC_l256_256220


namespace area_triangle_APQ_l256_256634

/-
  ABC is an equilateral triangle with side length 10.
  Points P and Q are on sides AB and AC respectively.
  Segment PQ is tangent to the incircle of triangle ABC and has length 4.
  Let AP = x and AQ = y, such that x + y = 6 and x^2 + y^2 - xy = 16.
  Prove that the area of triangle APQ is 5 * sqrt(3) / 3.
-/

theorem area_triangle_APQ :
  ∀ (x y : ℝ),
  x + y = 6 ∧ x^2 + y^2 - x * y = 16 → 
  (∃ (S : ℝ), S = (1 / 2) * x * y * (sqrt 3 / 2) ∧ S = 5 * (sqrt 3) / 3) :=
by 
  intro x y,
  intro h,
  sorry

end area_triangle_APQ_l256_256634


namespace trajectory_curve_point_F_exists_l256_256489

noncomputable def curve_C := { p : ℝ × ℝ | (p.1 - 1/2)^2 + (p.2 - 1/2)^2 = 4 }

theorem trajectory_curve (M : ℝ × ℝ) (p : ℝ × ℝ) (q : ℝ × ℝ) :
    M = ((p.1 + q.1) / 2, (p.2 + q.2) / 2) → 
    p.1^2 + p.2^2 = 9 → 
    q.1^2 + q.2^2 = 9 →
    (p.1 - 1)^2 + (p.2 - 1)^2 > 0 → 
    (q.1 - 1)^2 + (q.2 - 1)^2 > 0 → 
    ((p.1 - 1) * (q.1 - 1) + (p.2 - 1) * (q.2 - 1) = 0) →
    (M.1 - 1/2)^2 + (M.2 - 1/2)^2 = 4 :=
sorry

theorem point_F_exists (E D : ℝ × ℝ) (F : ℝ × ℝ) (H : ℝ × ℝ) :
    E = (9/2, 1/2) → D = (1/2, 1/2) → F.2 = 1/2 → 
    (∃ t : ℝ, t ≠ 9/2 ∧ F.1 = t) →
    (H ∈ curve_C) →
    ((H.1 - 9/2)^2 + (H.2 - 1/2)^2) / ((H.1 - F.1)^2 + (H.2 - 1/2)^2) = 24 * (15 - 8 * H.1) / ((t^2 + 15/4) * (24)) :=
sorry

end trajectory_curve_point_F_exists_l256_256489


namespace find_a_b_c_and_arithmetic_sqrt_l256_256855

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_a_b_c_and_arithmetic_sqrt :
  ∃ a b c : ℤ, (1 - 2 * a) + (a + 4) = 0 ∧ real.cbrt (4 * a + 2 * b - 1) = 3 ∧ c = Int.floor (Real.sqrt 13) ∧
    a = 5 ∧ b = 4 ∧ c = 3 ∧ sqrt (↑a + 2 * ↑b + ↑c) = 4 :=
by
  sorry

end find_a_b_c_and_arithmetic_sqrt_l256_256855


namespace fuel_consumed_l256_256747

theorem fuel_consumed (fuel_efficiency : ℝ) (speed_miles_per_hour : ℝ) (time_hours : ℝ) 
  (mile_to_km : ℝ) (liter_to_gallon : ℝ) :
  (fuel_efficiency = 72) →
  (speed_miles_per_hour = 117) →
  (time_hours = 5.7) →
  (mile_to_km = 1.6) →
  (liter_to_gallon = 3.8) →
  let speed_km_per_hour := speed_miles_per_hour * mile_to_km in
  let distance_km := speed_km_per_hour * time_hours in
  let fuel_consumed_liters := distance_km / fuel_efficiency in
  let fuel_consumed_gallons := fuel_consumed_liters / liter_to_gallon in
  fuel_consumed_gallons ≈ 3.896 :=
by
  intros h1 h2 h3 h4 h5
  let speed_km_per_hour := speed_miles_per_hour * mile_to_km
  let distance_km := speed_km_per_hour * time_hours
  let fuel_consumed_liters := distance_km / fuel_efficiency
  let fuel_consumed_gallons := fuel_consumed_liters / liter_to_gallon
  have : fuel_consumed_gallons = 3.896 at sorry

end fuel_consumed_l256_256747


namespace find_area_of_triangle_APQ_l256_256626

/-
  Given an equilateral triangle ABC with side length 10,
  and points P and Q on sides AB and AC such that PQ = 4 and
  PQ is tangent to the incircle of ABC,
  prove that the area of triangle APQ is equal to 5 * sqrt 3 / 3.
-/

def area_of_triangle_APQ  : ℝ :=
  let side_length := 10
  let PQ_length := 4
  let APQ_area := (5 * Real.sqrt 3) / 3
  APQ_area

theorem find_area_of_triangle_APQ :
  ∃ (P Q : (fin 2) → ℝ) (APQ_area: ℝ),
  (P 0).dist (P 1) = PQ_length ∧ (Q 0).dist (Q 1) = PQ_length ∧
  APQ_area = area_of_triangle_APQ ∧ 
  APQ_area = (5 * Real.sqrt 3) / 3 :=
by
  sorry

end find_area_of_triangle_APQ_l256_256626


namespace axis_of_symmetry_l256_256255

variables (a : ℝ) (x : ℝ)

def parabola := a * (x + 1) * (x - 3)

theorem axis_of_symmetry (h : a ≠ 0) : x = 1 := 
sorry

end axis_of_symmetry_l256_256255


namespace triangle_abc_angles_l256_256914

theorem triangle_abc_angles (A B C D E F X : Point) :
  acute_triangle A B C ∧ angle A B C = 45 ∧ altitude A D B C ∧ altitude B E A C ∧
  altitude C F A B ∧ intersect_in_fe_extended_to_bc E F X B C ∧ parallel A X D E →
  angle A B C + angle B C A + angle C A B = 180 ∧
  angle A B C = 75 ∧ angle B C A = 60 ∧ angle C A B = 45 :=
by
  sorry

end triangle_abc_angles_l256_256914


namespace max_intersection_points_of_fifth_degree_polynomials_l256_256720

-- Definitions for the conditions
def is_fifth_degree_polynomial_with_leading_coeff (p : ℝ[X]) (lc : ℝ) : Prop :=
  degree p = 5 ∧ leading_coeff p = lc

-- The proof problem
theorem max_intersection_points_of_fifth_degree_polynomials (p q : ℝ[X])
  (hp : is_fifth_degree_polynomial_with_leading_coeff p 1)
  (hq : is_fifth_degree_polynomial_with_leading_coeff q 2) :
  ∃ n : ℕ, n = 5 ∧
  (∀ x, p.eval x = q.eval x ↔ x ∈ {x : ℝ | p x = q x}) →
  finset.card {x : ℝ | p.eval x = q.eval x}.to_finset ≤ n :=
sorry

end max_intersection_points_of_fifth_degree_polynomials_l256_256720


namespace problem_f_of_f_of_3_l256_256035

def f (x : ℝ) : ℝ :=
if x < 3 then 3 * Real.exp(x - 1) else Real.logb 3 (x^2 - 6)

theorem problem_f_of_f_of_3 : f (f 3) = 3 :=
by
  sorry

end problem_f_of_f_of_3_l256_256035


namespace not_divide_g_count_30_l256_256190

-- Define the proper positive divisors function
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range (n - 1)).filter (λ d, d > 0 ∧ n % d = 0)

-- Define the product of proper divisors function
def g (n : ℕ) : ℕ :=
  proper_divisors n |>.prod

-- Define the main theorem
theorem not_divide_g_count_30 : 
  (Finset.range 99).filter (λ n, 2 ≤ n + 1 ∧ n + 1 ≤ 100 ∧ ¬(n + 1) ∣ g (n + 1)).card = 30 := 
  by
  sorry

end not_divide_g_count_30_l256_256190


namespace count_not_divides_in_range_is_33_l256_256183

-- Definition of product of proper divisors of n
def g (n : ℕ) : ℕ := (Finset.filter (λ d, d ≠ n ∧ d ∣ n) (Finset.range (n + 1))).prod id

-- Condition 1: Defines the range for n
def in_range (n : ℕ) : Prop := 2 ≤ n ∧ n ≤ 100

-- Condition 2: n does not divide g(n)
def not_divides (n : ℕ) : Prop := ¬ (n ∣ g n)

-- Statement of the problem
theorem count_not_divides_in_range_is_33 : (Finset.filter (λ n, in_range n ∧ not_divides n) (Finset.range 101)).card = 33 := by
  sorry

end count_not_divides_in_range_is_33_l256_256183


namespace number_of_integers_satisfying_eq_l256_256020

theorem number_of_integers_satisfying_eq :
  {n : ℤ | 1 + (200 * n / 201).floor = (198 * n / 199).ceil }.to_finset.card = 39899 := 
sorry

end number_of_integers_satisfying_eq_l256_256020


namespace year_possibilities_l256_256353

theorem year_possibilities : 
  let digits := [1, 1, 1, 5, 8, 9]
  let odd_digits := [1, 5, 9]
  let perms1 := Nat.factorial 5 / Nat.factorial 2 -- First digit is 1
  let perms2 := Nat.factorial 5 / Nat.factorial 3 -- First digit is 5 or 9
  (List.mem 1 odd_digits ∧ List.mem 5 odd_digits ∧ List.mem 9 odd_digits) ∧
  List.length digits = 6 ∧
  sortable.permutations digits |>.length = (perms1 + perms2 + perms2) :=
by
  sorry

end year_possibilities_l256_256353


namespace total_length_approaches_pi_C_l256_256256

noncomputable def total_length_quarter_circles (C : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0 else (2 * n * (π * C) / (2 * n))

theorem total_length_approaches_pi_C (C : ℝ) :
  filter.tendsto (λ n, total_length_quarter_circles C n) filter.at_top (nhds (π * C)) :=
begin
  sorry
end

end total_length_approaches_pi_C_l256_256256


namespace compare_ab_l256_256460

section ComparisonProof

variables {a b : ℚ}

-- Conditions
def abs_a : Prop := |a| = 2 / 3
def abs_b : Prop := |b| = 3 / 5

-- Proof required
theorem compare_ab (ha : abs_a) (hb : abs_b) :
  (a = 2 / 3 ∨ a = -2 / 3) ∧ (b = 3 / 5 ∨ b = -3 / 5) ∧
  (a = 2 / 3 → a > b) ∧ (a = -2 / 3 → a < b) :=
sorry

end ComparisonProof

end compare_ab_l256_256460


namespace tetrahedron_cross_section_area_l256_256542

theorem tetrahedron_cross_section_area (A B C D : Type)
  (area_ABC : ℝ) (area_ABD : ℝ) (angle_ACB_ABD : ℝ) 
  (h1 : area_ABC = 4) 
  (h2 : area_ABD = 7) 
  (h3 : angle_ACB_ABD = 60) : 
  (cross_section_area A B C D) = 28 * Real.sqrt 3 / 11 := 
sorry

end tetrahedron_cross_section_area_l256_256542


namespace angle_AOB_in_parallelogram_is_40_degrees_l256_256913

variable {A B C D O : Type*}
variables [Parallelogram ABCD]
variables [has_angle DCB : 80°]
variables (O : EquivMidpoint AC BD)

theorem angle_AOB_in_parallelogram_is_40_degrees (h_parallelogram : Parallelogram ABCD)
  (h_angle : angle DCB = 80°) (h_midpoint : Midpoint O AC BD) : angle AOB = 40° :=
sorry

end angle_AOB_in_parallelogram_is_40_degrees_l256_256913


namespace probability_at_least_one_even_l256_256821

theorem probability_at_least_one_even :
  let S := {1, 2, 3, 4, 5}
  let P := (comb (finset.card S) 2).card
  let P_odd := (comb (finset.filter (λ x, x % 2 ≠ 0) S).card 2).card
  P ≠ 0 → (1 - (P_odd / P) = 7 / 10) :=
by sorry

end probability_at_least_one_even_l256_256821


namespace part_a_part_b_l256_256319

-- Part (a)
theorem part_a (O1 O2 A A1 A2 : Point)
  (H1 : midpoint O1 A A1)
  (H2 : midpoint O2 A1 A2) :
  translate A A2 (2 * (vector O2 - vector O1)) :=
sorry

-- Part (b)
theorem part_b (O1 O2 : Point) (T : Translation) (S_O1 S_O2 : Symmetry)
  (H1 : S_O2 ∘ S_O1 = T) :
  S_O2 ∘ T = S_O1 ∧ T ∘ S_O1 = S_O2 :=
sorry

end part_a_part_b_l256_256319


namespace area_triangle_APQ_l256_256637

/-
  ABC is an equilateral triangle with side length 10.
  Points P and Q are on sides AB and AC respectively.
  Segment PQ is tangent to the incircle of triangle ABC and has length 4.
  Let AP = x and AQ = y, such that x + y = 6 and x^2 + y^2 - xy = 16.
  Prove that the area of triangle APQ is 5 * sqrt(3) / 3.
-/

theorem area_triangle_APQ :
  ∀ (x y : ℝ),
  x + y = 6 ∧ x^2 + y^2 - x * y = 16 → 
  (∃ (S : ℝ), S = (1 / 2) * x * y * (sqrt 3 / 2) ∧ S = 5 * (sqrt 3) / 3) :=
by 
  intro x y,
  intro h,
  sorry

end area_triangle_APQ_l256_256637


namespace height_of_remaining_solid_l256_256352

theorem height_of_remaining_solid (side_length : ℝ) (sqrt2 : ℝ) (sqrt3 : ℝ)
  (h1 : side_length = 2)
  (h2 : sqrt2 = Real.sqrt 2)
  (h3 : sqrt3 = Real.sqrt 3) :
  ∃ h' : ℝ, h' = (6 - 2 * sqrt3) / 3 :=
by
  use (6 - 2 * sqrt3) / 3
  sorry

end height_of_remaining_solid_l256_256352


namespace pn_remainder_l256_256056

variables (n : ℕ) (P_n : ℕ) (S_n : Finset ℕ)
def S_n_def (n : ℕ) : Finset ℕ := {x | 1 ≤ x ∧ x < n ∧ (n ∣ (x^2 - 1))}

theorem pn_remainder (h1 : 1 < n) 
                     (hs : S_n = S_n_def n)
                     (hp : P_n = S_n.prod id) :
  P_n % n = if ∃ p : ℕ, p.prime ∧ ∃ k : ℕ, n = p^k then n - 1 else 1 := 
sorry

end pn_remainder_l256_256056


namespace area_triangle_APQ_l256_256632

def equilateral_triangle (A B C P Q : Type) [metric_space A] :=
  ∃ (a b c : A) (R T : A → A) (x y : ℝ), 
    dist a b = 10 ∧
    dist a c = 10 ∧
    dist b c = 10 ∧
    dist P Q = 4 ∧
    x = dist a P ∧
    y = dist a Q ∧
    (R a = P) ∧
    (T a = Q) ∧
    (x + y = 6) ∧
    (x^2 + y^2 - x * y = 16)

theorem area_triangle_APQ (A B C P Q : Type) [metric_space A] : 
  ∀ (x y : ℝ), 
    equilateral_triangle A B C P Q → 
    x = dist A P → 
    y = dist A Q → 
    (x * y = 20 / 3) → 
    let s := real.sqrt 3 / 2 in 
    (1 / 2 * x * y * s = 5 * real.sqrt 3 / 3) :=
by simp [equilateral_triangle]

end area_triangle_APQ_l256_256632


namespace inequality_holds_for_k_l256_256812

def largest_constant_k : ℝ := (Real.sqrt 6) / 2

theorem inequality_holds_for_k (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) :
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) ≤ largest_constant_k * Real.sqrt (x + y + z) :=
sorry

end inequality_holds_for_k_l256_256812


namespace range_of_a_l256_256867

def f (a x : ℝ) : ℝ := 2 * x - 2 - a * Real.log x

noncomputable def exists_m (a : ℝ) : Prop :=
  ∃ m > 1, ∀ x ∈ Set.Ioo 1 m, |f a x| > 2 * Real.log x

theorem range_of_a (a : ℝ) : (∃ m > 1, ∀ x ∈ Set.Ioo 1 m, |f a x| > 2 * Real.log x) ↔ a > 4 :=
by
  sorry

end range_of_a_l256_256867


namespace functional_equation_sum_l256_256119

theorem functional_equation_sum :
  (∀ a b : ℕ+, f (a + b) = f a * f b) →
  f 1 = 2 →
  (∑ i in (finset.range 1007).map (λ x => x + 1), f (2 * (i : ℕ)) / f (2 * (i - 1) + 1)) = 2014 :=
by
  -- Definition of f used in the theorem
  intros h1 h2
  sorry

end functional_equation_sum_l256_256119


namespace find_x_l256_256521

theorem find_x (x : ℕ) (h1 : 8^x = 2^9) (h2 : 8 = 2^3) : x = 3 := by
  sorry

end find_x_l256_256521


namespace distinct_possible_values_count_l256_256402

-- Define the set of positive odd integers less than 15.
def oddIntsLessThan15 : Set ℤ := {1, 3, 5, 7, 9, 11, 13}

-- Define the expression (p * q - (p + q)).
def expression (p q : ℤ) : ℤ := p * q - (p + q)

-- Prove there are exactly 28 different possible values for the given expression.
theorem distinct_possible_values_count :
  ∃ (S : Set ℤ), S = (λ (p q : ℤ), expression p q) '' {pq | pq.1 ∈ oddIntsLessThan15 ∧ pq.2 ∈ oddIntsLessThan15} ∧ S.card = 28 :=
  sorry

end distinct_possible_values_count_l256_256402


namespace transform_sequence_result_l256_256106

theorem transform_sequence_result :
  let a := (104 * 5 + 2) / 5 : ℚ,
      b := a * (8/3),
      c := b * (1/2),
      d := c + (29 / 2),
      e := d * (4 / 7),
      f := e - (59 / 28)
  in f = 86 := 
by 
  admit

end transform_sequence_result_l256_256106


namespace marble_probability_l256_256345

theorem marble_probability :
  let total_marbles := 13
  let red_marbles := 5
  let white_marbles := 8
  let first_red_prob := (red_marbles:ℚ) / total_marbles
  let second_white_given_first_red_prob := (white_marbles:ℚ) / (total_marbles - 1)
  let third_red_given_first_red_and_second_white_prob := (red_marbles - 1:ℚ) / (total_marbles - 2)
  first_red_prob * second_white_given_first_red_prob * third_red_given_first_red_and_second_white_prob = (40 : ℚ) / 429 :=
by
  let total_marbles := 13
  let red_marbles := 5
  let white_marbles := 8
  let first_red_prob := (red_marbles:ℚ) / total_marbles
  let second_white_given_first_red_prob := (white_marbles:ℚ) / (total_marbles - 1)
  let third_red_given_first_red_and_second_white_prob := (red_marbles - 1:ℚ) / (total_marbles - 2)
  -- Adding sorry to skip the proof
  sorry

end marble_probability_l256_256345


namespace max_value_f_in_interval_l256_256823

noncomputable def f (x : ℝ) : ℝ :=
  (1 - x + Real.sqrt(2 * x^2 - 2 * x + 1)) / (2 * x)

theorem max_value_f_in_interval :
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x ≤ (Real.sqrt 5 - 1) / 4 :=
by
  intros x hx
  sorry

end max_value_f_in_interval_l256_256823


namespace passing_students_this_year_l256_256893

constant initial_students : ℕ := 200 -- Initial number of students who passed three years ago
constant growth_rate : ℝ := 0.5      -- Growth rate of 50%

-- Function to calculate the number of students passing each year
def students_passing (n : ℕ) : ℕ :=
nat.rec_on n initial_students (λ n' ih, ih + (ih / 2))

-- Proposition stating the number of students passing the course this year
theorem passing_students_this_year : students_passing 3 = 675 := sorry

end passing_students_this_year_l256_256893


namespace intersection_product_is_15_l256_256313

-- Define the first circle equation as a predicate
def first_circle (x y : ℝ) : Prop :=
  x^2 - 4 * x + y^2 - 6 * y + 12 = 0

-- Define the second circle equation as a predicate
def second_circle (x y : ℝ) : Prop :=
  x^2 - 10 * x + y^2 - 6 * y + 34 = 0

-- The Lean statement for the proof problem
theorem intersection_product_is_15 :
  ∃ x y : ℝ, first_circle x y ∧ second_circle x y ∧ (x * y = 15) :=
by
  sorry

end intersection_product_is_15_l256_256313


namespace probability_forming_lot_l256_256267

theorem probability_forming_lot : 
  let letters := ['o', 'l', 't'] in
  let arrangements := permutations letters in
  let favorable_outcomes := list.filter (λ w, w = ['l', 'o', 't']) arrangements in
  (list.length favorable_outcomes) / (list.length arrangements) = 1 / 6 :=
  sorry

end probability_forming_lot_l256_256267


namespace smallest_possible_perimeter_l256_256795

-- Definitions for prime numbers and scalene triangles
def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions
def valid_sides (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ is_scalene_triangle a b c

def valid_perimeter (a b c : ℕ) : Prop :=
  is_prime (a + b + c)

-- The goal statement
theorem smallest_possible_perimeter : ∃ a b c : ℕ, valid_sides a b c ∧ valid_perimeter a b c ∧ (a + b + c) = 23 :=
by
  sorry

end smallest_possible_perimeter_l256_256795


namespace crayons_more_than_erasers_l256_256651

-- Definitions of the conditions
def initial_crayons := 531
def initial_erasers := 38
def final_crayons := 391
def final_erasers := initial_erasers -- no erasers lost

-- Theorem statement
theorem crayons_more_than_erasers :
  final_crayons - final_erasers = 102 :=
by
  -- Placeholder for the proof
  sorry

end crayons_more_than_erasers_l256_256651


namespace fraction_area_isosceles_triangle_outside_circle_l256_256911

noncomputable def isosceles_triangle_area_fraction_outside_circle : Real :=
  let α := 80 * Real.pi / 180 -- Convert 80 degrees to radians
  let β := 100 * Real.pi / 180 -- Convert 100 degrees to radians
  let S_triangle := ((Real.sqrt 3) / 4) * (2 * Real.sqrt (1 - (Real.cos β)))^2
  let S_sector := (100 / 360) * (Real.pi * r^2)
  let S_triangle_OBC := (1 / 2) * r^2 * Real.sin β
  let S_segment := S_sector - S_triangle_OBC
  let S_outside := S_triangle - 2 * S_segment
  S_outside / S_triangle

theorem fraction_area_isosceles_triangle_outside_circle
  (AB AC : ℝ) (BAC : ℝ) (r : ℝ)
  (h_iso : AB = AC)
  (h_angle : BAC = 80 * Real.pi / 180)
  (h_tangent : is_tangent_to_circle AB AC r) :
  isosceles_triangle_area_fraction_outside_circle = (4 * Real.sqrt 3 / 3) - (Real.sqrt 3 * Real.pi) :=
sorry

end fraction_area_isosceles_triangle_outside_circle_l256_256911


namespace sequence_sum_l256_256796

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ a 2 = 1/2 ∧ ∀ k ≥ 1, a (k+2) = a k + 1/2 * a (k+1) + 1/(4 * a k * a (k+1))

theorem sequence_sum (a : ℕ → ℚ) (h : sequence a) :
  ∑ k in Finset.range 99, 1 / (a k * a (k + 2)) < 4 :=
sorry

end sequence_sum_l256_256796


namespace intersection_M_N_l256_256503

def M : Set ℝ := { y | ∃ x : ℝ, y = 2^(|x|) }
def N : Set ℝ := { x | ∃ y : ℝ, y = log(3 - x) }

theorem intersection_M_N : M ∩ N = {[y | 1 ≤ y < 3]} :=
by
  sorry

end intersection_M_N_l256_256503


namespace total_legs_and_hands_on_ground_is_118_l256_256146

-- Definitions based on the conditions given
def total_dogs := 20
def dogs_on_two_legs := total_dogs / 2
def dogs_on_four_legs := total_dogs / 2

def total_cats := 10
def cats_on_two_legs := total_cats / 3
def cats_on_four_legs := total_cats - cats_on_two_legs

def total_horses := 5
def horses_on_two_legs := 2
def horses_on_four_legs := total_horses - horses_on_two_legs

def total_acrobats := 6
def acrobats_on_one_hand := 4
def acrobats_on_two_hands := 2

-- Functions to calculate the number of legs/paws/hands on the ground
def dogs_legs_on_ground := (dogs_on_two_legs * 2) + (dogs_on_four_legs * 4)
def cats_legs_on_ground := (cats_on_two_legs * 2) + (cats_on_four_legs * 4)
def horses_legs_on_ground := (horses_on_two_legs * 2) + (horses_on_four_legs * 4)
def acrobats_hands_on_ground := (acrobats_on_one_hand * 1) + (acrobats_on_two_hands * 2)

-- Total legs/paws/hands on the ground
def total_legs_on_ground := dogs_legs_on_ground + cats_legs_on_ground + horses_legs_on_ground + acrobats_hands_on_ground

-- The theorem to prove
theorem total_legs_and_hands_on_ground_is_118 : total_legs_on_ground = 118 :=
by sorry

end total_legs_and_hands_on_ground_is_118_l256_256146


namespace num_of_3_good_subsets_l256_256357

open Finset

def is_3_good (s : Finset ℕ) : Prop :=
  s.sum id % 3 = 0

theorem num_of_3_good_subsets : 
  (univ.filter is_3_good).card = 351 := 
sorry

end num_of_3_good_subsets_l256_256357


namespace no_flippy_numbers_divisible_by_11_and_6_l256_256016

def is_flippy (n : ℕ) : Prop :=
  let d1 := n / 10000
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  (d1 = d3 ∧ d3 = d5 ∧ d2 = d4 ∧ d1 ≠ d2) ∨ 
  (d2 = d4 ∧ d4 = d5 ∧ d1 = d3 ∧ d1 ≠ d2)

def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11) = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10000) + (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

def sum_divisible_by_6 (n : ℕ) : Prop :=
  (sum_of_digits n) % 6 = 0

theorem no_flippy_numbers_divisible_by_11_and_6 :
  ∀ n, (10000 ≤ n ∧ n < 100000) → is_flippy n → is_divisible_by_11 n → sum_divisible_by_6 n → false :=
by
  intros n h_range h_flippy h_div11 h_sum6
  sorry

end no_flippy_numbers_divisible_by_11_and_6_l256_256016


namespace six_digit_number_consecutive_evens_l256_256433

theorem six_digit_number_consecutive_evens :
  ∃ n : ℕ,
    287232 = (2 * n - 2) * (2 * n) * (2 * n + 2) ∧
    287232 / 100000 = 2 ∧
    287232 % 10 = 2 :=
by
  sorry

end six_digit_number_consecutive_evens_l256_256433


namespace minimum_distance_l256_256492

/-
  Define the parabola and the line equations.
  Define the function to compute the distance from a point to a vertical line (y-axis)
  Define the function to compute the distance from a point to a given line
-/

def parabola (A : ℝ × ℝ) : Prop := A.snd^2 = -4 * A.fst
def line_l (A : ℝ × ℝ) : Prop := 2 * A.fst + A.snd - 4 = 0

def distance_to_y_axis (A : ℝ × ℝ) : ℝ := abs A.1
def distance_to_line (A : ℝ × ℝ) : ℝ := abs (2 * A.fst + A.snd - 4) / sqrt (2^2 + 1^2)

theorem minimum_distance (A : ℝ × ℝ) (h1 : parabola A) :
  ∃ m n, m = distance_to_y_axis A ∧ n = distance_to_line A ∧ (m + n) = (6 * sqrt 5 / 5 - 1) := sorry

end minimum_distance_l256_256492


namespace at_least_five_friends_l256_256283

-- Define the main properties and conditions
variable (Delegate : Type) [Fintype Delegate]
variable [DecidableEq Delegate] (conference : Finset Delegate) (friends : Delegate → Finset Delegate)

-- Define the problem condition
def condition (conference : Finset Delegate) :=
  -- Conference has 12 delegates
  conference.card = 12 ∧
  -- Every two delegates share a common friend
  (∀ a b ∈ conference, ∃ c ∈ conference, c ≠ a ∧ c ≠ b ∧ c ∈ friends a ∧ c ∈ friends b)

-- Define the theorem statement
theorem at_least_five_friends (h : condition conference) :
  ∃ d ∈ conference, (friends d).card ≥ 5 :=
sorry

end at_least_five_friends_l256_256283


namespace areasEqualForHexagonAndOctagon_l256_256790

noncomputable def areaHexagon (s : ℝ) : ℝ :=
  let r := s / Real.sin (Real.pi / 6) -- Circumscribed radius
  let a := s / (2 * Real.tan (Real.pi / 6)) -- Inscribed radius
  Real.pi * (r^2 - a^2)

noncomputable def areaOctagon (s : ℝ) : ℝ :=
  let r := s / Real.sin (Real.pi / 8) -- Circumscribed radius
  let a := s / (2 * Real.tan (3 * Real.pi / 8)) -- Inscribed radius
  Real.pi * (r^2 - a^2)

theorem areasEqualForHexagonAndOctagon :
  let s := 3
  areaHexagon s = areaOctagon s := sorry

end areasEqualForHexagonAndOctagon_l256_256790


namespace simplify_expression1_simplify_expression2_simplify_expression3_simplify_expression4_l256_256995

-- Problem 1
theorem simplify_expression1 (a b : ℝ) : 
  4 * a^3 + 2 * b - 2 * a^3 + b = 2 * a^3 + 3 * b := 
sorry

-- Problem 2
theorem simplify_expression2 (x : ℝ) : 
  2 * x^2 + 6 * x - 6 - (-2 * x^2 + 4 * x + 1) = 4 * x^2 + 2 * x - 7 := 
sorry

-- Problem 3
theorem simplify_expression3 (a b : ℝ) : 
  3 * (3 * a^2 - 2 * a * b) - 2 * (4 * a^2 - a * b) = a^2 - 4 * a * b := 
sorry

-- Problem 4
theorem simplify_expression4 (x y : ℝ) : 
  6 * x * y^2 - (2 * x - (1 / 2) * (2 * x - 4 * x * y^2) - x * y^2) = 5 * x * y^2 - x := 
sorry

end simplify_expression1_simplify_expression2_simplify_expression3_simplify_expression4_l256_256995


namespace hyperbola_eccentricity_correct_l256_256868

def hyperbola_eccentricity (a b : ℝ) (h1 : b > 0)
    (h2 : ∀ x y : ℝ, x - 2 * y = 0) : ℝ :=
    let e := 1 / (Real.sqrt (1 - (1 / 2) ^ 2)) in e

theorem hyperbola_eccentricity_correct (a b : ℝ) (h1 : b > 0)
    (h2 : ∀ x y : ℝ, x - 2 * y = 0) : hyperbola_eccentricity a b h1 h2 = Real.sqrt (5) / 2 :=
by
    unfold hyperbola_eccentricity
    sorry

end hyperbola_eccentricity_correct_l256_256868


namespace mean_innovation_degree_two_l256_256202

def innovation_degree {α : Type*} [linear_order α] (A : list α) : ℕ :=
(A.map (λ k, A.take k).map max_fun).erase_dup.length

def permutations_with_innovation_degree_two {α : Type*} [linear_order α] (n : ℕ) :
  list (list ℕ) :=
(list.permutations (list.range n.succ)).filter (λ A, innovation_degree A = 2)

noncomputable def arithmetic_mean_innovation_degree_two (n : ℕ) : ℚ :=
let perms := permutations_with_innovation_degree_two n in
(list.sum (perms.map (list.sum ∘ list.map (coe : ℕ → ℚ))) / list.length perms)

theorem mean_innovation_degree_two (n : ℕ) (h : 2 ≤ n) :
  arithmetic_mean_innovation_degree_two n =
    n - (n - 1) / (1 + (list.range (n-1)).sum (λ m, 1 / (m.succ : ℚ))) :=
sorry

end mean_innovation_degree_two_l256_256202


namespace find_k_l256_256586

variable (S : ℕ → ℝ) (a : ℕ → ℝ)
variable (k : ℕ)
variable (q : ℝ) (a1 : ℝ)

-- Define the conditions as Lean assumptions
-- Condition: Geometric sequence with first term 1 and common ratio 2
axiom geometric_sequence (n : ℕ) : a n = a1 * q^n
axiom first_term : a1 = 1
axiom common_ratio : q = 2

-- Define sum of geometric sequence terms
axiom sum_of_terms (n : ℕ) : S n = ∑ i in range (n + 1), a i

-- Given condition: S_{k+2} - S_k = 48
axiom given_condition : S (k + 2) - S k = 48

-- The target proposition to prove
theorem find_k : k = 4 :=
by
  sorry

end find_k_l256_256586


namespace solution_set_inequality_l256_256479

noncomputable def f : ℝ → ℝ := sorry  -- f is a function ℝ -> ℝ
def f' (x : ℝ) := deriv f x -- f' denotes the derivative of f

-- Given conditions
axiom condition1 : f(1) = Real.exp 1
axiom condition2 : ∀ x, 2 * f x - deriv f x > 0

theorem solution_set_inequality : {x : ℝ | f(x) / Real.exp x < Real.exp (x - 1)} = Ioi 1 :=
sorry

end solution_set_inequality_l256_256479


namespace sorting_mailbox_probability_l256_256343
open_locale big_operators

theorem sorting_mailbox_probability :
  let total_ways := 3^4,
      at_least_one_empty := total_ways - (3 * (Nat.choose 4 2)),
      probability := at_least_one_empty / total_ways
  in probability = 5 / 9 :=
  sorry

end sorting_mailbox_probability_l256_256343


namespace inradius_of_right_triangle_l256_256760

-- Define the side lengths of the triangle
def a : ℕ := 9
def b : ℕ := 40
def c : ℕ := 41

-- Define the semiperimeter of the triangle
def s : ℕ := (a + b + c) / 2

-- Define the area of a right triangle
def A : ℕ := (a * b) / 2

-- Define the inradius of the triangle
def inradius : ℕ := A / s

theorem inradius_of_right_triangle : inradius = 4 :=
by
  -- The proof is omitted since only the statement is requested
  sorry

end inradius_of_right_triangle_l256_256760


namespace number_of_sets_l256_256926

theorem number_of_sets (weight_per_rep reps total_weight : ℕ) 
  (h_weight_per_rep : weight_per_rep = 15)
  (h_reps : reps = 10)
  (h_total_weight : total_weight = 450) :
  (total_weight / (weight_per_rep * reps)) = 3 :=
by
  sorry

end number_of_sets_l256_256926


namespace garden_area_correct_l256_256389

def property_width : ℝ := 1000
def property_length : ℝ := 2250
def shorter_side := property_width / 8
def longer_side := property_width / 6
def trapezoid_height := property_length / 10
def garden_area := 1 / 2 * (shorter_side + longer_side) * trapezoid_height

theorem garden_area_correct : garden_area = 32812.875 := by
  sorry

end garden_area_correct_l256_256389


namespace dot_product_self_l256_256114

variables {ω : Type*} {w : ω → ℝ}

theorem dot_product_self (norm_w : ∥w∥ = 7) : (w • w) = 49 :=
by
  sorry

end dot_product_self_l256_256114


namespace cake_pieces_per_sister_l256_256031

theorem cake_pieces_per_sister (total_pieces : ℕ) (percentage_eaten : ℕ) (sisters : ℕ)
  (h1 : total_pieces = 240) (h2 : percentage_eaten = 60) (h3 : sisters = 3) :
  (total_pieces * (1 - percentage_eaten / 100)) / sisters = 32 :=
by
  sorry

end cake_pieces_per_sister_l256_256031


namespace width_of_floor_X_eq_10_l256_256662

-- Define the conditions
def area_of_floor (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def equal_area_rectangles (length_X width_X : ℕ) (length_Y width_Y : ℕ) : Prop :=
  area_of_floor length_X width_X = area_of_floor length_Y width_Y

-- Constants given in the problem
constant length_X : ℕ := 18
constant length_Y : ℕ := 20
constant width_Y  : ℕ := 9

-- The main theorem we want to prove
theorem width_of_floor_X_eq_10 (h : equal_area_rectangles length_X width_X length_Y width_Y) : width_X = 10 := by
  -- Proof goes here
  sorry

end width_of_floor_X_eq_10_l256_256662


namespace domino_3x4_equal_dots_l256_256714

theorem domino_3x4_equal_dots : 
  ∃ (arrangement : list (fin 3 × fin 4 → ℕ)), 
    (∀ (row : fin 3),
      ∑ col, arrangement row col = 4) ∧ 
    (∀ (col : fin 4),
      ∑ row, arrangement row col = 3) :=
sorry

end domino_3x4_equal_dots_l256_256714


namespace max_value_prod_distances_l256_256036

noncomputable def max_product_of_distances (F₁ F₂ : ℝ × ℝ) (P : ℝ × ℝ) : ℝ :=
  let a := 5
  let b := 3
  let c := 4
  let |F₁ - P| := 5 + (4/5)*P.1
  let |F₂ - P| := 5 - (4/5)*P.1
  max |F₁ - P| * |F₂ - P|

theorem max_value_prod_distances (F₁ F₂ : ℝ × ℝ) (P : ℝ × ℝ) :
  ∀ (P : ℝ × ℝ) on ellipse (25, 9), 
  ∃ (max_value : ℝ), max_value_prod_distances F₁ F₂  = 25 :=
sorry

end max_value_prod_distances_l256_256036


namespace fixed_point_linear_l256_256041

-- Define the linear function y = kx + k + 2
def linear_function (k x : ℝ) : ℝ := k * x + k + 2

-- Prove that the point (-1, 2) lies on the graph of the function for any k
theorem fixed_point_linear (k : ℝ) : linear_function k (-1) = 2 := by
  sorry

end fixed_point_linear_l256_256041


namespace rational_neither_positive_nor_fraction_l256_256363

def is_rational (q : ℚ) : Prop :=
  q.floor = q

def is_integer (q : ℚ) : Prop :=
  ∃ n : ℤ, q = n

def is_fraction (q : ℚ) : Prop :=
  ∃ p q : ℤ, q ≠ 0 ∧ q = p / q

def is_positive (q : ℚ) : Prop :=
  q > 0

theorem rational_neither_positive_nor_fraction (q : ℚ) :
  (is_rational q) ∧ ¬(is_positive q) ∧ ¬(is_fraction q) ↔
  (is_integer q ∧ q ≤ 0) :=
sorry

end rational_neither_positive_nor_fraction_l256_256363


namespace equation_of_line_l256_256268

theorem equation_of_line
  (l : ℝ → ℝ)
  (h_intersects : ∃ (x1 y1 x2 y2 : ℝ), y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧ l x1 = y1 ∧ l x2 = y2)
  (h_midpoint : ∃ (x1 y1 x2 y2 : ℝ), l x1 = y1 ∧ l x2 = y2 ∧ (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 1) :
  l = λ x, 2 * x - 1 :=
sorry

end equation_of_line_l256_256268


namespace problem_B_problem_D_l256_256683

-- Problem B
def hyperbola (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

theorem problem_B (P Q F1 F2 : ℝ × ℝ) (a b : ℝ) (x y : ℝ)
  (h_hyperbola : hyperbola a b x y)
  (h_eq : a = 2 ∧ b = sqrt 6)
  (h_PF1 : 6 ≥ 0) -- Placeholder to identify point conditions
  (h_PF2 : 6 ≥ 0) : -- Placeholder to identify point conditions
  (area_of_triangle P F1 Q = 24) := sorry

-- Problem D
theorem problem_D (P Q F1 F2 : ℝ × ℝ) (a b : ℝ)
  (h_hyperbola : hyperbola a b (F2.1 + a) F2.2)
  (h_cond : 3 * distance P F2 = distance Q F2) : 
  (eccentricity a b = (sqrt 10) / 2) := sorry

end problem_B_problem_D_l256_256683


namespace complement_U_A_l256_256206

open Set

-- Definitions of the universal set U and the set A
def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}

-- Proof statement: the complement of A with respect to U is {3}
theorem complement_U_A : U \ A = {3} :=
by
  sorry

end complement_U_A_l256_256206


namespace ap_eq_aq_if_and_only_if_isosceles_l256_256204

variable {α : Type*} [linear_ordered_field α]

structure Triangle :=
(A B C : α × α)

structure Midpoint (α : Type*) :=
(M N : α × α)

structure Points (α : Type*) :=
(P Q : α × α)

def angle_bisector (T : Triangle) (P Q : Points α) : Prop :=
-- Placeholder: Define angle bisector relationship between angles ACB and MCP, and ABC and NBQ
sorry

def is_isosceles (T : Triangle) : Prop :=
-- Placeholder: Define isosceles property for the Triangle
sorry

theorem ap_eq_aq_if_and_only_if_isosceles
  (T : Triangle) (M N : Midpoint α) (P Q : Points α)
  (h1 : M.M = midpoint T.B T.C)
  (h2 : N.N = midpoint T.B T.C)
  (h3 : angle_bisector T P Q)
  (h4 : angle_bisector T P Q) :
  (distance T.A P.P = distance T.A Q.Q) ↔ is_isosceles T :=
sorry

end ap_eq_aq_if_and_only_if_isosceles_l256_256204


namespace find_x_value_l256_256444

variable (x : ℝ)
def product_term (n : ℝ) : ℝ := 1 + 1 / n
def product_P : ℝ := (product_term 4) * (product_term 5) * (product_term 6) * (product_term 7) * (product_term 8) * (product_term 9) * (product_term 10) * (product_term 11) * (product_term 12) * (product_term 13) * (product_term 14) * (product_term 15) * (product_term 16) * (product_term 17) * (product_term 18) * (product_term 19) * (product_term 20) * (product_term 21) * (product_term 22) * (product_term 23) * (product_term 24) * (product_term 25) * (product_term 26) * (product_term 27) * (product_term 28) * (product_term 29) * (product_term 30) * (product_term 31) * (product_term 32) * (product_term 33) * (product_term 34) * (product_term 35) * (product_term 36) * (product_term 37) * (product_term 38) * (product_term 39) * (product_term 40) * (product_term 41) * (product_term 42) * (product_term 43) * (product_term 44) * (product_term 45) * (product_term 46) * (product_term 47) * (product_term 48) * (product_term 49) * (product_term 50) * (product_term 51) * (product_term 52) * (product_term 53) * (product_term 54) * (product_term 55) * (product_term 56) * (product_term 57) * (product_term 58) * (product_term 59) * (product_term 60) * (product_term 61) * (product_term 62) * (product_term 63) * (product_term 64) * (product_term 65) * (product_term 66) * (product_term 67) * (product_term 68) * (product_term 69) * (product_term 70) * (product_term 71) * (product_term 72) * (product_term 73) * (product_term 74) * (product_term 75) * (product_term 76) * (product_term 77) * (product_term 78) * (product_term 79) * (product_term 80) * (product_term 81) * (product_term 82) * (product_term 83) * (product_term 84) * (product_term 85) * (product_term 86) * (product_term 87) * (product_term 88) * (product_term 89) * (product_term 90) * (product_term 91) * (product_term 92) * (product_term 93) * (product_term 94) * (product_term 95) * (product_term 96) * (product_term 97) * (product_term 98) * (product_term 99) * (product_term 100) * (product_term 101) * (product_term 102) * (product_term 103) * (product_term 104) * (product_term 105) * (product_term 106) * (product_term 107) * (product_term 108) * (product_term 109) * (product_term 110) * (product_term 111) * (product_term 112) * (product_term 113) * (product_term 114) * (product_term 115) * (product_term 116) * (product_term 117) * (product_term 118) * (product_term 119) * (product_term 120)

theorem find_x_value : 3 / 11 * (x * product_P) = 11 -> x = 11 := by
  intros
  sorry

end find_x_value_l256_256444


namespace expressions_not_equal_l256_256808

theorem expressions_not_equal (x : ℝ) (hx : x > 0) : 
  3 * x^x ≠ 2 * x^x + x^(2 * x) ∧ 
  x^(3 * x) ≠ 2 * x^x + x^(2 * x) ∧ 
  (3 * x)^x ≠ 2 * x^x + x^(2 * x) ∧ 
  (3 * x)^(3 * x) ≠ 2 * x^x + x^(2 * x) :=
by 
  sorry

end expressions_not_equal_l256_256808


namespace number_of_valid_ks_l256_256700

theorem number_of_valid_ks : 
  let N := (λ (n : ℤ), abs ((12 / n) + 5 * n)) in
  ∃ (n_count : ℤ), n_count = 78 ∧
  (∀ (k : ℚ), abs k < 200 → ∃ (x : ℤ), 5 * x^2 + k * x + 12 = 0 → 
  ∃ (n : ℤ), 1 ≤ abs n ∧ abs (12 / n + 5 * n) < 200) := 
sorry

end number_of_valid_ks_l256_256700


namespace niko_total_profit_l256_256216

noncomputable def calculate_total_profit : ℝ :=
  let pairs := 9
  let price_per_pair := 2
  let discount_rate := 0.10
  let shipping_cost := 5
  let profit_4_pairs := 0.25
  let profit_5_pairs := 0.20
  let tax_rate := 0.05
  let cost_socks := pairs * price_per_pair
  let discount := discount_rate * cost_socks
  let cost_after_discount := cost_socks - discount
  let total_cost := cost_after_discount + shipping_cost
  let resell_price_4_pairs := (price_per_pair * (1 + profit_4_pairs)) * 4
  let resell_price_5_pairs := (price_per_pair * (1 + profit_5_pairs)) * 5
  let total_resell_price := resell_price_4_pairs + resell_price_5_pairs
  let sales_tax := tax_rate * total_resell_price
  let total_resell_price_after_tax := total_resell_price + sales_tax
  let total_profit := total_resell_price_after_tax - total_cost
  total_profit

theorem niko_total_profit : calculate_total_profit = 0.85 :=
by
  sorry

end niko_total_profit_l256_256216


namespace units_digit_of_m_squared_plus_two_to_m_is_3_l256_256200

def m := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m_is_3 : (m^2 + 2^m) % 10 = 3 := 
by 
  sorry

end units_digit_of_m_squared_plus_two_to_m_is_3_l256_256200


namespace defective_units_percentage_l256_256554

variables (D : ℝ)

-- 4% of the defective units are shipped for sale
def percent_defective_shipped : ℝ := 0.04

-- 0.24% of the units produced are defective units that are shipped for sale
def percent_total_defective_shipped : ℝ := 0.0024

-- The theorem to prove: the percentage of the units produced that are defective is 0.06
theorem defective_units_percentage (h : percent_defective_shipped * D = percent_total_defective_shipped) : D = 0.06 :=
sorry

end defective_units_percentage_l256_256554


namespace min_income_wealthiest_500_l256_256679

theorem min_income_wealthiest_500 (x : ℝ) : 
  (∀ N, N = 2 * 10^9 * x^(-2) → N = 500) → x = 2000 :=
by
  sorry

end min_income_wealthiest_500_l256_256679


namespace trapezoid_EC_length_l256_256334

-- Define a trapezoid and its properties.
structure Trapezoid (A B C D : Type) :=
(base1 : ℝ) -- AB
(base2 : ℝ) -- CD
(diagonal_AC : ℝ) -- AC
(AB_eq_3CD : base1 = 3 * base2)
(AC_length : diagonal_AC = 15)
(E : Type) -- point of intersection of diagonals

-- Proof statement that length of EC is 15/4
theorem trapezoid_EC_length
  {A B C D E : Type}
  (t : Trapezoid A B C D)
  (E : Type)
  (intersection_E : E) :
  ∃ (EC : ℝ), EC = 15 / 4 :=
by
  have h1 : t.base1 = 3 * t.base2 := t.AB_eq_3CD
  have h2 : t.diagonal_AC = 15 := t.AC_length
  -- Use the given conditions to derive the length of EC
  sorry

end trapezoid_EC_length_l256_256334


namespace complement_of_beta_l256_256476

theorem complement_of_beta (α β : ℝ) (h₀ : α + β = 180) (h₁ : α > β) : 
  90 - β = 1/2 * (α - β) :=
by
  sorry

end complement_of_beta_l256_256476


namespace evaluate_S_l256_256244

variables {x : ℕ → ℝ} {S : ℝ}

-- setting up the conditions as hypothesis
def condition (x : ℕ → ℝ) := 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1004 → x n + (n + 2) = (∑ k in finset.range 1004, x k) + 1005

theorem evaluate_S (h : condition x) :
  ∥∑ n in finset.range 1004, x n∥.toNat = 498 := by
  sorry

end evaluate_S_l256_256244


namespace magnitude_of_sum_l256_256092

-- Definitions for the conditions
variables (a b : EuclideanSpace ℝ (Fin 2))
variables (h1 : a ∙ b = 0)  -- Dot product is zero, implying orthogonality
variables (h2 : ‖a‖ = 1)   -- Magnitude of a
variables (h3 : ‖b‖ = 2)   -- Magnitude of b

-- Proof statement to be filled in later
theorem magnitude_of_sum (h1 : a ∙ b = 0) (h2 : ‖a‖ = 1) (h3 : ‖b‖ = 2) : ‖a + b‖ = real.sqrt 5 :=
by sorry

end magnitude_of_sum_l256_256092


namespace average_contribution_required_l256_256731

def total_amount_needed (T : ℝ) : Prop :=
  T > 0

def donation_received (d : ℝ) : Prop :=
  d = 400

def percentage_received (r : ℝ) : Prop :=
  r = 0.6

def percentage_solicited (ps : ℝ) : Prop :=
  ps = 0.4

theorem average_contribution_required (T P : ℝ) (hT : total_amount_needed T)
    (hd : donation_received 400) (hr : percentage_received 0.6) (hps : percentage_solicited 0.4):
  let amount_received := hr * T,
      people_solicited := hps * P,
      total_collected := people_solicited * 400,
      remaining_amount := T - amount_received,
      remaining_people := 0.6 * P,
      required_contribution := remaining_amount / remaining_people
  in
  required_contribution = 444.44 :=
by
  sorry

end average_contribution_required_l256_256731


namespace value_of_a_l256_256316

theorem value_of_a (x a : ℝ) (h1 : 0 < x) (h2 : x < 1 / a) (h3 : ∀ x, x * (1 - a * x) ≤ 1 / 12) : a = 3 :=
sorry

end value_of_a_l256_256316


namespace find_lambda_l256_256102

-- Given vectors a and b
def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (1, -1)

-- Definition of dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Condition for perpendicularity
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Establishing a mathematical property that needs to be proven
theorem find_lambda (λ : ℝ) : perpendicular (a.1 + λ * b.1, a.2 + λ * b.2) b ↔ λ = -1 :=
by sorry

end find_lambda_l256_256102


namespace smallest_diameter_rope_l256_256332

noncomputable theory

-- Definitions
def mass : ℝ := 20 -- in tons
def number_of_slings : ℝ := 3
def angle_alpha : ℝ := 30 -- in degrees
def safety_factor : ℝ := 6
def max_load_per_thread : ℝ := 10^3 -- in N/mm²
def g : ℝ := 10 -- in m/s²

-- Conversion from tons to kilograms (since 1 ton = 1000 kg)
def mass_kg : ℝ := mass * 1000

-- Total weight in Newtons
def total_weight : ℝ := mass_kg * g

-- Load shared by each strop
def load_per_strop : ℝ := total_weight / number_of_slings

-- Angle in radians
def angle_alpha_rad : ℝ := angle_alpha * (Real.pi / 180)

-- Vertical component considering the angle α
def load_with_angle : ℝ := load_per_strop / Real.cos angle_alpha_rad

-- Required breaking strength considering the safety factor
def required_breaking_strength : ℝ := safety_factor * load_with_angle

-- Convert max load per thread to N/m²
def max_load_per_thread_N_m2 : ℝ := max_load_per_thread * 10^6

-- Necessary cross-sectional area
def required_area : ℝ := required_breaking_strength / max_load_per_thread_N_m2

-- Diameter calculation from cross-sectional area
def min_diameter : ℝ := Real.sqrt ((required_area * 4) / Real.pi)

-- Round up to the nearest whole millimeter
def min_diameter_rounded : ℕ := Int.ceil min_diameter.toReal -- convert to whole mm

-- Proof statement
theorem smallest_diameter_rope : min_diameter_rounded = 26 := 
by sorry

end smallest_diameter_rope_l256_256332


namespace coefficient_m5_n5_in_expansion_l256_256306

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Goal: prove the coefficient of m^5 n^5 in the expansion of (m+n)^{10} is 252
theorem coefficient_m5_n5_in_expansion : binomial 10 5 = 252 :=
by
  sorry

end coefficient_m5_n5_in_expansion_l256_256306


namespace mean_score_all_students_l256_256968

variables (M A : ℝ) (m a : ℕ)
variables (additional_group_score : ℝ) (ratio_of_students : ℝ)

-- Given conditions
def conditions : Prop :=
  M = 85 ∧
  A = 72 ∧
  ratio_of_students = 4 / 5 ∧
  additional_group_score = 68

-- ToProve: The mean score of all the students is 87
def mean_student_scores := 
  (M : ℝ) * (4 / 5) * (a : ℝ) + (A : ℝ) * (a : ℝ) + 68 * (1 / 4) * (a : ℝ)

def total_students := (4 / 5) * (a : ℝ) + (a : ℝ)

theorem mean_score_all_students (h : conditions): 
  mean_student_scores M A m a / total_students = 87 :=
by
  sorry

end mean_score_all_students_l256_256968


namespace coefficient_x90_is_neg1_l256_256439

-- Definition of the polynomial
def poly : Polynomial ℤ := (Polynomial.X - 1) *
                           (Polynomial.X^2 - 2) * 
                           (Polynomial.X^3 - 3) * 
                           (Polynomial.X^4 - 4) * 
                           (Polynomial.X^5 - 5) * 
                           (Polynomial.X^6 - 6) * 
                           (Polynomial.X^7 - 7) * 
                           (Polynomial.X^8 - 8) * 
                           (Polynomial.X^9 - 9) * 
                           (Polynomial.X^10 - 10) * 
                           (Polynomial.X^11 - 11) * 
                           (Polynomial.X^13 - 13)

-- Statement to prove that coefficient of x^90 is -1
theorem coefficient_x90_is_neg1 : poly.coeff 90 = -1 := 
by {
  -- Add your proof here
  sorry
}

end coefficient_x90_is_neg1_l256_256439


namespace find_desired_expression_l256_256518

variable (y : ℝ)

theorem find_desired_expression
  (h : y + Real.sqrt (y^2 - 4) + (1 / (y - Real.sqrt (y^2 - 4))) = 12) :
  y^2 + Real.sqrt (y^4 - 4) + (1 / (y^2 - Real.sqrt (y^4 - 4))) = 200 / 9 :=
sorry

end find_desired_expression_l256_256518


namespace incorrect_value_not_in_list_l256_256912

theorem incorrect_value_not_in_list (a b c : ℝ) :
  let f (x : ℕ) := a * x^2 + b * x + c in
  let values := [3844, 3969, 4096, 4227, 4356, 4489, 4624, 4761] in
  ∀ x : ℕ, x ∈ [3, 5, 7, 9, 11, 13, 15, 17] →
    f x = values.nth_le (x - 3) (by linarith) →
    ¬ (f 5 = 4096 ∨ f 9 = 4356 ∨ f 11 = 4489 ∨ f 17 = 4761) := sorry

end incorrect_value_not_in_list_l256_256912


namespace part_I_part_II_l256_256042

def sequence_def (x : ℕ → ℝ) (p : ℝ) : Prop :=
  x 1 = 1 ∧ ∀ n ∈ (Nat.succ <$> {n | n > 0}), x (n + 1) = 1 + x n / (p + x n)

theorem part_I (x : ℕ → ℝ) (p : ℝ) (h_seq : sequence_def x p) :
  p = 2 → ∀ n ∈ (Nat.succ <$> {n | n > 0}), x n < Real.sqrt 2 :=
sorry

theorem part_II (x : ℕ → ℝ) (p : ℝ) (h_seq : sequence_def x p) :
  (∀ n ∈ (Nat.succ <$> {n | n > 0}), x (n + 1) > x n) → ¬ ∃ M ∈ {n | n > 0}, ∀ n > 0, x M ≥ x n :=
sorry

end part_I_part_II_l256_256042


namespace circumcircle_tangent_to_BD_l256_256152

open EuclideanGeometry

variables {A B C D E H I : Point ℝ}

/-- Given a convex quadrilateral ABCD, with specific perpendicular conditions and given angle conditions, 
    prove that the circumcircle of triangle HIE is tangent to line BD. -/
theorem circumcircle_tangent_to_BD 
  (hAD_perp_CD : ⟪AD - A, CD - D⟫ = 0)
  (hAB_perp_CB : ⟪AB - A, CB - B⟫ = 0)
  (hAE_perp_BD : ⟪AE - A, BD - B⟫ = 0)
  (h_angle_CEH_CHD : ∠CEH - ∠CHD = 90)
  (h_angle_CEI_CIB : ∠CEI - ∠CIB = 90)
  (H_on_AD : ∃ p, p ∈ line AD)
  (I_on_AB : ∃ q, q ∈ line AB) :
  tangent (circumcircle H I E) (line BD) :=
begin
  sorry
end

end circumcircle_tangent_to_BD_l256_256152


namespace find_amplitude_l256_256388

noncomputable def oscillation_amplitude (max min : ℝ) : ℝ :=
  (max - min) / 2

theorem find_amplitude 
  (a b c d : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : 0 < d)
  (max_value : ℝ)
  (min_value : ℝ)
  (h_max : max_value = 5)
  (h_min : min_value = -3) :
  oscillation_amplitude max_value min_value = 4 :=
by {
  rw [h_max, h_min],
  simp [oscillation_amplitude],
  norm_num,
  sorry
}

end find_amplitude_l256_256388


namespace ellipse_equation_valid_l256_256054

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (eccentricity : a^2 * 8 / b^2 = 9) : Prop :=
  ∀ (A1 A2 B : ℝ × ℝ), 
    (A1 = (-3, 0)) ∧ 
    (A2 = (3, 0)) ∧ 
    (B = (0, 2 * Real.sqrt 2)) ∧ 
    ((B.1 - A1.1) * (B.1 - A2.1) + (B.2 - A1.2) * (B.2 - A2.2) = -1) →
    ( ∃ m : ℝ, m ≠ 0 ∧ (m^2 = 1) ∧ 
    (a^2 = 9 * m^2) ∧ (b^2 = 8 * m^2) ∧ 
    (C : ℝ × ℝ → Prop, ∀ x y, C (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1) → 
    ( ∀ x y, C (x, y) ↔ x^2 / 9 + y^2 / 8 = 1) )

theorem ellipse_equation_valid :
ellipse_equation 3 (2 * Real.sqrt 2) (by linarith) (by linarith) 
(3^2 * 8 / (2 * Real.sqrt 2)^2 = 9) := 
sorry

end ellipse_equation_valid_l256_256054


namespace problem_statement_l256_256325

-- Define the ellipse
def ellipse_eq (x y : ℝ) : Prop := 
  (x^2)/25 + (y^2)/9 = 1

-- Define the vertices
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define the slopes with given ratios
def slopes_condition (k1 k2 : ℝ) : Prop := 
  k1 / k2 = 1 / 9

-- Define the line intersection with the ellipse
def line_intersects (l : ℝ → ℝ) (x y : ℝ) : Prop :=
  ellipse_eq x y ∧ y = l x

-- Fixed point condition
def fixed_point : ℝ × ℝ := (4, 0)

-- Areas of triangles
def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  abs ((fst P * (snd Q - snd R) + fst Q * (snd R - snd P) + fst R * (snd P - snd Q)) / 2)

def AMN_area (M N : ℝ × ℝ) : ℝ :=
  area_triangle A M N

def BMN_area (M N : ℝ × ℝ) : ℝ :=
  area_triangle B M N

-- Maximum difference between areas
def max_diff (M N : ℝ × ℝ) : ℝ :=
  AMN_area M N - BMN_area M N

theorem problem_statement (k1 k2 : ℝ) (M N : ℝ × ℝ) (l : ℝ → ℝ) :
    slopes_condition k1 k2 →
    line_intersects l (fst M) (snd M) →
    line_intersects l (fst N) (snd N) →
    (∃ x y, line_intersects l x y ∧ (x, y) = fixed_point) ∧ 
    max_diff M N ≤ 15 := 
sorry

end problem_statement_l256_256325


namespace intersection_distance_l256_256150

noncomputable def cartesian_equation_curve_C : ℝ → ℝ × ℝ → Prop :=
  fun ρ θ =>
  let x := ρ * Math.cos θ
  let y := ρ * Math.sin θ
  x^2 + y^2 = 4 * x

def parametric_equation_line_l (t : ℝ) : ℝ × ℝ :=
  (1 + (Math.sqrt 3 / 2) * t, (1 / 2) * t)

theorem intersection_distance :
  ∃ A B : ℝ × ℝ, 
    cartesian_equation_curve_C (Math.sqrt ((A.fst)^2 + (A.snd)^2)) (Math.atan2 A.snd A.fst)
    ∧ cartesian_equation_curve_C (Math.sqrt ((B.fst)^2 + (B.snd)^2)) (Math.atan2 B.snd B.fst)
    ∧ A ≠ B
    ∧ ∃ t1 t2 : ℝ,
      parametric_equation_line_l t1 = A
      ∧ parametric_equation_line_l t2 = B
      ∧ |(B.fst - A.fst)^2 + (B.snd - A.snd)^2| = real.sqrt 15 := 
sorry

end intersection_distance_l256_256150


namespace problem_f_2012_eq_cos_l256_256594

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => Real.cos x
| n+1, x => (f n x)' x

theorem problem_f_2012_eq_cos : ∀ x : ℝ, f 2012 x = Real.cos x :=
begin
  intros,
  sorry,
end

end problem_f_2012_eq_cos_l256_256594


namespace find_p_q_l256_256932

variable (A B C D P P1 P2 Q Q1 Q2 ω : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P] [Inhabited P1] [Inhabited P2] [Inhabited Q] [Inhabited Q1] [Inhabited Q2] [Inhabited ω]
variable (AC DB CD AP1 P2B P2Q2 P1Q1 : ℕ)
variable (cyclic : ω)

-- Conditions
def AC := 5
def CD := 6
def DB := 7
def AP1 := 3
def P2B := 4
def P2Q2 := 2

-- Question: Find p+q where P1Q1 = p/q and p,q are relatively prime positive integers

theorem find_p_q (p q : ℕ) (pq_rel_prime : Nat.gcd p q = 1)
  (P1Q1 : ℚ := 106 / 47) :
  p + q = 153 := by
  sorry

end find_p_q_l256_256932


namespace range_of_f_l256_256498

noncomputable def f (x k : ℝ) : ℝ := x - k * real.sqrt (x^2 - 1)

theorem range_of_f (k : ℝ) (h : 0 < k ∧ k < 1) : 
  set.range (λ x : ℝ, if x >= 1 then f x k else 0) = set.Ici (real.sqrt (1 - k^2)) :=
sorry

end range_of_f_l256_256498


namespace distinct_permutations_of_MAMMA_l256_256409

theorem distinct_permutations_of_MAMMA : 
  let total_permutations := 5! / (3! * 2!) in
  total_permutations = 10 :=
by
  let total_permutations := 120 / (6 * 2);
  have perm_eq : total_permutations = 10 := by norm_num;
  exact perm_eq

end distinct_permutations_of_MAMMA_l256_256409


namespace train_length_is_250_meters_l256_256370

def train_speed_kmph := 58
def man_speed_kmph := 8
def passing_time_seconds := 17.998560115190788
def conversion_factor_kmph_to_mps := 1000.0 / 3600.0
def relative_speed_mps := (train_speed_kmph - man_speed_kmph) * conversion_factor_kmph_to_mps
def train_length_meters := relative_speed_mps * passing_time_seconds

theorem train_length_is_250_meters : train_length_meters = 250 :=
by
  sorry

end train_length_is_250_meters_l256_256370


namespace product_inequality_l256_256203

theorem product_inequality (n : ℕ) (x : Fin (n+2) → ℝ)
  (h1 : ∀ i, 0 < x i)
  (h2 : ∑ i, 1 / (1 + x i) = 1) :
  ∏ i, x i ≥ n^(n + 1) :=
by
  sorry

end product_inequality_l256_256203


namespace tangent_line_parallel_to_give_line_l256_256678

noncomputable def line_params (c : ℝ) := 2 * c

def is_parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∃ m : ℝ, ∀ x y, l1 x y = 2 * x - y → l2 x y = 2 * x - (y + m)

def is_tangent_to_circle (l : ℝ → ℝ → Prop) : Prop :=
  ∃ c : ℝ, l 2 c = 2 * 2 - 2 * c ∧ abs (c) = 5 ∧ (c * c) + (l c) = 5

theorem tangent_line_parallel_to_give_line : 
  ∀ (l : ℝ → ℝ → Prop), 
  is_parallel (λ x y, 2 * x - y) l ∧ is_tangent_to_circle l →
  ∃ c : ℝ, l = (λ x y, 2 * x - y + c) ∧ (c = 5 ∨ c = -5) :=
by sorry

end tangent_line_parallel_to_give_line_l256_256678


namespace remainder_at_point_is_correct_l256_256361

noncomputable def remainder_at_point
  (p : Polynomial ℚ)
  (h₁ : p.eval 3 = 2)
  (h₂ : p.eval 4 = 5)
  (h₃ : p.eval (-5) = -6) : ℚ :=
  let r : Polynomial ℚ := Polynomial.X^2 * (2 / 9) - Polynomial.X * (13 / 9) + (7 / 3)
  in r.eval 7

theorem remainder_at_point_is_correct
  (p : Polynomial ℚ)
  (h₁ : p.eval 3 = 2)
  (h₂ : p.eval 4 = 5)
  (h₃ : p.eval (-5) = -6) :
  remainder_at_point p h₁ h₂ h₃ = 28 / 9 :=
by
  sorry

end remainder_at_point_is_correct_l256_256361


namespace length_PS_l256_256560

-- Define the triangle PQS and the length PQ
variable (P Q S : Type)
variable (PQ : P → Q → ℝ) -- PQ is a function that maps P and Q to a real number (their distance)

-- Conditions based on the problem
variable (PQ_len : PQ P Q = 2)

-- Main statement to prove
theorem length_PS : ∃ (PS : ℝ), PS = 1.5 :=
  sorry

end length_PS_l256_256560


namespace coeff_m5n5_in_m_plus_n_pow_10_l256_256303

theorem coeff_m5n5_in_m_plus_n_pow_10 :
  binomial (10, 5) = 252 := by
sorry

end coeff_m5n5_in_m_plus_n_pow_10_l256_256303


namespace number_of_decompositions_l256_256169

theorem number_of_decompositions (M : ℕ) : 
  let b_range := {b : ℕ // 0 ≤ b ∧ b ≤ 199},
      count := λ b3 b2 b1 b0, 
                (4050 = b3 * 10^3 + b2 * 10^2 + b1 * 10 + b0) ∧ 
                b1 % 2 = 0 in
  M = (finset.univ.filter (λ b3 : b_range, finset.univ.filter (λ b2 : b_range, finset.univ.filter (λ b1 : b_range, count b3.val b2.val b1.val b0.val).card).card).card) → M = 359 :=
by
  sorry

end number_of_decompositions_l256_256169


namespace evaluate_expression_l256_256807

-- Definitions for conditions
def x := (1 / 4 : ℚ)
def y := (1 / 2 : ℚ)
def z := (3 : ℚ)

-- Statement of the problem
theorem evaluate_expression : 
  4 * (x^3 * y^2 * z^2) = 9 / 64 :=
by
  sorry

end evaluate_expression_l256_256807


namespace max_value_sqrt_expression_l256_256596

variable (x y z : ℝ)

theorem max_value_sqrt_expression :
  x + y + z = 2 ∧ x ≥ -1 ∧ y ≥ (-3 / 2) ∧ z ≥ -2 →
  ∃ max_val, max_val = 2 * sqrt 30 ∧ sqrt (4 * x + 2) + sqrt (4 * y + 6) + sqrt (4 * z + 8) ≤ max_val :=
by
  intro h
  sorry

end max_value_sqrt_expression_l256_256596


namespace circumcircles_tangent_l256_256738

variables {A B C O B' C' : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space O] [metric_space B'] [metric_space C']
variables [has_dist A B] [has_dist B C] [has_dist O A] [has_dist O B] [has_dist O C] [has_dist O B'] [has_dist O C']

-- Conditions
def point_inside_triangle (O : Type*) (A B C : Type*) := 
  sorry -- definition of point inside triangle

def is_midpoint_of_arc (x : Type*) (P Q R : Type*) := 
  sorry -- definition of midpoint of arc

def distance_relation (O A B C : Type*) :=
  sorry -- definition for distance relation OA = OB + OC

axiom midpoint_B'_arc_AOC : is_midpoint_of_arc B' A O C
axiom midpoint_C'_arc_AOB : is_midpoint_of_arc C' A O B
axiom point_O_inside : point_inside_triangle O A B C
axiom dist_relation : distance_relation O A B C

-- Theorem to prove
theorem circumcircles_tangent : 
  (circumcircle O C C') ∩ (circumcircle O B B') = {O} :=
sorry

end circumcircles_tangent_l256_256738


namespace four_pow_k_eq_5_l256_256123

theorem four_pow_k_eq_5 (k N : ℝ) (h1 : 4 ^ k = N) (h2 : 4 ^ (2 * k + 2) = 400) : N = 5 :=
by
  sorry

end four_pow_k_eq_5_l256_256123


namespace birds_per_site_average_l256_256931

def total_birds : ℕ := 5 * 7 + 5 * 5 + 10 * 8 + 7 * 10 + 3 * 6 + 8 * 12 + 4 * 9
def total_sites : ℕ := 5 + 5 + 10 + 7 + 3 + 8 + 4

theorem birds_per_site_average :
  (total_birds : ℚ) / total_sites = 360 / 42 :=
by
  -- Skip proof
  sorry

end birds_per_site_average_l256_256931


namespace identify_at_least_13_blondes_l256_256328

theorem identify_at_least_13_blondes :
  ∀ (women : Finset ℕ) (brunettes blondes : Finset ℕ), 
  women.card = 217 →
  brunettes.card = 17 →
  blondes.card = 200 →
  (∀ b ∈ brunettes, ∀ x ∈ women.erase b, x ∈ blondes ∧ women.erase b.card = 200) →
  (∀ b ∈ blondes, ∃ lst : Finset ℕ, lst ⊆ women.erase b ∧ lst.card = 200) →
  ∃ (identified_blondes : Finset ℕ), identified_blondes ⊆ blondes ∧ identified_blondes.card ≥ 13 := 
by 
  sorry

end identify_at_least_13_blondes_l256_256328


namespace am_minus_hm_lt_bound_l256_256089

theorem am_minus_hm_lt_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) :
  (x - y)^2 / (2 * (x + y)) < (x - y)^2 / (8 * x) := 
by
  sorry

end am_minus_hm_lt_bound_l256_256089


namespace area_triangle_APQ_l256_256630

def equilateral_triangle (A B C P Q : Type) [metric_space A] :=
  ∃ (a b c : A) (R T : A → A) (x y : ℝ), 
    dist a b = 10 ∧
    dist a c = 10 ∧
    dist b c = 10 ∧
    dist P Q = 4 ∧
    x = dist a P ∧
    y = dist a Q ∧
    (R a = P) ∧
    (T a = Q) ∧
    (x + y = 6) ∧
    (x^2 + y^2 - x * y = 16)

theorem area_triangle_APQ (A B C P Q : Type) [metric_space A] : 
  ∀ (x y : ℝ), 
    equilateral_triangle A B C P Q → 
    x = dist A P → 
    y = dist A Q → 
    (x * y = 20 / 3) → 
    let s := real.sqrt 3 / 2 in 
    (1 / 2 * x * y * s = 5 * real.sqrt 3 / 3) :=
by simp [equilateral_triangle]

end area_triangle_APQ_l256_256630


namespace num_convex_quadrilateral_angles_arith_prog_l256_256508

theorem num_convex_quadrilateral_angles_arith_prog :
  ∃ (S : Finset (Finset ℤ)), S.card = 29 ∧
    ∀ {a b c d : ℤ}, {a, b, c, d} ∈ S →
      a + b + c + d = 360 ∧
      a < b ∧ b < c ∧ c < d ∧
      ∃ (m d_diff : ℤ), 
        m - d_diff = a ∧
        m = b ∧
        m + d_diff = c ∧
        m + 2 * d_diff = d ∧
        a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
sorry

end num_convex_quadrilateral_angles_arith_prog_l256_256508


namespace surface_area_of_circumscribed_sphere_l256_256155

variables {A B C D : Type*}
variables [metric_space A]
variables (a b c d : A)

-- Conditions
variables (h1 : dist a b ^ 2 + dist c a ^ 2 + dist a d ^ 2 = 6)
variables (h2 : ∥a - b∥ * ∥a - c∥ = √2)
variables (h3 : ∥a - c∥ * ∥a - d∥ = √3)
variables (h4 : ∥a - b∥ * ∥a - d∥ = √6)

-- Question: Prove that the surface area of the circumscribed sphere of the triangular pyramid is 6π
theorem surface_area_of_circumscribed_sphere :
  4 * π * (sqrt 6 / 2) ^ 2 = 6 * π :=
sorry

end surface_area_of_circumscribed_sphere_l256_256155


namespace geometric_sequence_term_formula_l256_256696
noncomputable def general_term_formula (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  if q = 2 then 3 * 3^(a - 1)
  else if q = 3 then 2 * 2^(a - 1)
  else 0

theorem geometric_sequence_term_formula (a : ℕ → ℝ) (q : ℝ):
  (a 2 = 6) ∧ (6 * a 1 + a 3 = 30) →
  (a = general_term_formula a q) :=
sorry

end geometric_sequence_term_formula_l256_256696


namespace factorial_division_l256_256398

theorem factorial_division : (30! / 28!) = 870 :=
by sorry

end factorial_division_l256_256398


namespace min_value_fractions_l256_256949

theorem min_value_fractions (x y z : ℝ) (hx : 0 < x) (hy: 0 < y) (hz: 0 < z) (h : x + y + z = 3) : 
  ∃ m : ℝ, m = 16/9 ∧ (∀ w, w = (x + y) / (x * y * z) → w ≥ m) :=
begin
  sorry
end

end min_value_fractions_l256_256949


namespace proof_problem_l256_256548

noncomputable def parabola_y_squared_2px (p x : ℝ) : ℝ :=
  real.sqrt (2 * p * x)

theorem proof_problem (t p : ℝ) (ht : t ≠ 0) (hp : p > 0) :
  let M := (0, t),
      P := (t^2 / (2 * p), t),
      N := (t^2 / (2 * p), 0),
      ON := (λ x : ℝ, (p / t) * x),
      H := (2 * t^2 / p, 2 * t) in
  (|on_y H - 0| / |on_y N - 0| = 2) ∧
  (∀ (x : ℝ), N x ≠ H x → parabola_y_squared_2px p x ≠ parabola_y_squared_2px p (N x)) := 
begin
  let M := (0, t),
  let P := (t^2 / (2 * p), t),
  let N := (t^2 / (2 * p), 0),
  let ON := λ x : ℝ, (p / t) * x,
  let H := (2 * t^2 / p, 2 * t),
  sorry
end

end proof_problem_l256_256548


namespace cake_pieces_l256_256162

theorem cake_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) 
  (pan_dim : pan_length = 24 ∧ pan_width = 15) 
  (piece_dim : piece_length = 3 ∧ piece_width = 2) : 
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
sorry

end cake_pieces_l256_256162


namespace triangle_count_in_square_config_l256_256399

/-- 
  Given a square with vertices A, B, C, and D, 
  with diagonals AC and BD intersecting at the center,
  and midpoints M, N, P, Q of sides AB, BC, CD, and DA respectively,
  and an inscribed circle touching these midpoints,
  the total number of triangles of any size formed within this configuration is 16.
-/
theorem triangle_count_in_square_config (A B C D M N P Q O I : Point)
  (h_square : Square A B C D)
  (h_diagonals : Diagonal AC) (h_diagonals : Diagonal BD)
  (h_midpoints : Midpoint M A B) (h_midpoints : Midpoint N B C)
  (h_midpoints : Midpoint P C D) (h_midpoints : Midpoint Q D A)
  (h_circle : InscribedCircle I M N P Q O) :
  TotalTrianglesInSquareConfig A B C D M N P Q O I = 16 :=
sorry

end triangle_count_in_square_config_l256_256399


namespace area_triangle_APQ_l256_256629

def equilateral_triangle (A B C P Q : Type) [metric_space A] :=
  ∃ (a b c : A) (R T : A → A) (x y : ℝ), 
    dist a b = 10 ∧
    dist a c = 10 ∧
    dist b c = 10 ∧
    dist P Q = 4 ∧
    x = dist a P ∧
    y = dist a Q ∧
    (R a = P) ∧
    (T a = Q) ∧
    (x + y = 6) ∧
    (x^2 + y^2 - x * y = 16)

theorem area_triangle_APQ (A B C P Q : Type) [metric_space A] : 
  ∀ (x y : ℝ), 
    equilateral_triangle A B C P Q → 
    x = dist A P → 
    y = dist A Q → 
    (x * y = 20 / 3) → 
    let s := real.sqrt 3 / 2 in 
    (1 / 2 * x * y * s = 5 * real.sqrt 3 / 3) :=
by simp [equilateral_triangle]

end area_triangle_APQ_l256_256629


namespace unique_solution_m_l256_256015

theorem unique_solution_m:
  (∀ (x y : ℝ) (m : ℝ), (x^2 = 2^|x| + |x| - y - m) ∧ (1 - y^2 = 0) → 
  (x = 0 ∧ y = 1 ∧ m = 0) ∨ (x ≠ 0 ∨ y ≠ 1 ∨ m ≠ 0)) :=
begin
  intros x y m h,
  obtain ⟨h1, h2⟩ := h,
  have h3 : y = 1 ∨ y = -1,
  { rw ←eq_sub_iff_add_eq',
    linarith },
  cases h3,
  { subst h3,
    rw sub_zero at h2,
    have h4 : x = 0,
    { sorry },
    subst h4,
    simp [eq_comm] },
  { subst h3,
    rw sub_add_eq_zero at h2,
    have h5 : x ≠ 0 ∨ 2 ≠ 0 ∨ m ≠ 2,
    { sorry },
    exact h5 }
end

end unique_solution_m_l256_256015


namespace sqrt_14_range_l256_256420

theorem sqrt_14_range : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 :=
by
  -- We know that 9 < 14 < 16, so we can take the square root of all parts to get 3 < sqrt(14) < 4.
  sorry

end sqrt_14_range_l256_256420


namespace calculate_angles_and_sides_l256_256540

noncomputable def right_angled_triangle (c : ℝ) : Prop :=
  ∃ (a b : ℝ), (b^2 = a * c) ∧ (a^2 + b^2 = c^2) ∧ 
  (a = c * ((Real.sqrt 5 - 1) / 2)) ∧ 
  (Real.arcsin ((a) / c) * (180 / Real.pi) ≈ 38.1667) ∧ 
  (Real.arcsin ((b) / c) * (180 / Real.pi) ≈ 51.8333)

theorem calculate_angles_and_sides (c : ℝ) (hc : 0 < c) :
  right_angled_triangle c :=
begin
  sorry
end

end calculate_angles_and_sides_l256_256540


namespace A_intersection_B_nonempty_l256_256843

def A (x : ℝ) : Prop := log x / log 2 < 1
def B (x : ℝ) : Prop := -1 < x ∧ x < 1

theorem A_intersection_B_nonempty : ∃ x, A x ∧ B x := 
by
  sorry

end A_intersection_B_nonempty_l256_256843


namespace factorize_l256_256425

theorem factorize (m : ℝ) : m^3 - 4 * m = m * (m + 2) * (m - 2) :=
by
  sorry

end factorize_l256_256425


namespace magnitude_a_plus_2b_l256_256879

open Real

variable {a b : ℝ × ℝ} -- Define vectors 'a' and 'b' in ℝ x ℝ

-- Conditions
def condition1 : |a| = 1 := sorry
def condition2 : |b| = 2 := sorry
def condition3 : a - b = (sqrt 2, sqrt 3) := sorry

-- Goal
theorem magnitude_a_plus_2b (h₁ : |a| = 1) (h₂ : |b| = 2) (h₃ : a - b = (sqrt 2, sqrt 3)) : 
  |a + 2 • b| = sqrt 17 :=
sorry

end magnitude_a_plus_2b_l256_256879


namespace gcd_12a_18b_l256_256118

theorem gcd_12a_18b (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a.gcd b = 15) : (12 * a).gcd (18 * b) = 90 :=
by sorry

end gcd_12a_18b_l256_256118


namespace angle_CAB_in_regular_hexagon_l256_256538

theorem angle_CAB_in_regular_hexagon
  (ABCDEF : Type)
  [Hexagon : ∀ (i : ℕ), ABCDEF]         -- Define the regular hexagon type
  (regular : ∀ (i : ℕ), ∃ (angle : ℝ), angle = 120) -- Interior angles are 120 degrees
  (symmetry : ∀ (A B C : ABCDEF), ∃ (x : ℝ), x = ∠CAB = ∠BCA) -- Symmetry of the hexagon
: ∀ (AC : Diagonal ABCDEF), (∠CAB = 30) := sorry

end angle_CAB_in_regular_hexagon_l256_256538


namespace probability_one_pair_one_three_of_a_kind_l256_256803

def one_pair_one_three_of_a_kind_probability : ℚ :=
  25 / 1296

theorem probability_one_pair_one_three_of_a_kind :
  let total_rolls := 6^6
  let ways_to_choose_pair_and_three_of_a_kind := (Nat.choose 6 2) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 3 * Nat.factorial 1))
  let successful_outcomes := ways_to_choose_pair_and_three_of_a_kind
  let result := (successful_outcomes : ℚ) / (total_rolls : ℚ)
  result = one_pair_one_three_of_a_kind_probability :=
by {
  let total_rolls := 46656
  let ways_to_choose_pair_and_three_of_a_kind := 15 * 60
  let successful_outcomes := 900
  let result := (successful_outcomes : ℚ) / (total_rolls : ℚ)
  show result = one_pair_one_three_of_a_kind_probability,
    sorry
}

end probability_one_pair_one_three_of_a_kind_l256_256803


namespace solve_polynomial_relation_l256_256242

--Given Conditions
def polynomial_relation (x y : ℤ) : Prop := y^3 = x^3 + 8 * x^2 - 6 * x + 8 

--Proof Problem
theorem solve_polynomial_relation : ∃ (x y : ℤ), (polynomial_relation x y) ∧ 
  ((y = 11 ∧ x = 9) ∨ (y = 2 ∧ x = 0)) :=
by 
  sorry

end solve_polynomial_relation_l256_256242


namespace inscribed_circle_radius_centroid_incenter_parallel_angle_bisector_perpendicular_sum_of_distances_constant_incenter_and_midpoints_cyclic_l256_256981

noncomputable def triangle_arith_progression (a b c : ℝ) :=
  b = (a + c) / 2

variable {a b c h_b r G I O : ℝ}

theorem inscribed_circle_radius
  (h : triangle_arith_progression a b c)
  (area : ℝ)
  (semi_perimeter : (a + b + c) / 2)
  (altitude_to_b : h_b)
  (radius : r) :
  r = (1 / 3) * h_b := sorry

theorem centroid_incenter_parallel
  (h : triangle_arith_progression a b c)
  (centroid : G)
  (incenter : I)
  : line_parallel (connected_line G I) b:= sorry

theorem angle_bisector_perpendicular
  (h : triangle_arith_progression a b c)
  (incenter : I)
  (circumcenter : O) :
  angle_bisector b ⟂ line_through I O := sorry

theorem sum_of_distances_constant
  (h : triangle_arith_progression a b c)
  (bisector_point : ℝ) :
  ∀ pt on angle_bisector, sum_of_distances_to_sides pt a b c constant := sorry

theorem incenter_and_midpoints_cyclic
  (h : triangle_arith_progression a b c)
  (incenter : I)
  (midpoint_largest : ℝ)
  (midpoint_smallest : ℝ)
  (vertex : ℝ) :
  cyclic I midpoint_largest midpoint_smallest vertex := sorry

end inscribed_circle_radius_centroid_incenter_parallel_angle_bisector_perpendicular_sum_of_distances_constant_incenter_and_midpoints_cyclic_l256_256981


namespace surface_area_comparison_l256_256349

-- Definition of convex polyhedron (for the sake of translation, we define them abstractly)
structure ConvexPolyhedron :=
(surface_area : ℝ)

-- Assume we have inner and outer polyhedra with the given conditions
variables (P_inner P_outer : ConvexPolyhedron)
  (inside : P_inner)

-- Define the main theorem statement
theorem surface_area_comparison (h1 : P_inner.surface_area < P_outer.surface_area) : 
  P_outer.surface_area > P_inner.surface_area :=
by sorry

end surface_area_comparison_l256_256349


namespace beavers_fraction_l256_256355

theorem beavers_fraction (total_beavers : ℕ) (swim_percentage : ℕ) (work_percentage : ℕ) (fraction_working : ℕ) : 
total_beavers = 4 → 
swim_percentage = 75 → 
work_percentage = 100 - swim_percentage → 
fraction_working = 1 →
(work_percentage * total_beavers) / 100 = fraction_working → 
fraction_working / total_beavers = 1 / 4 :=
by 
  intros h1 h2 h3 h4 h5 
  sorry

end beavers_fraction_l256_256355


namespace suraya_picked_less_than_mia_l256_256208

def apples_picked_by_kayla : ℕ := 20
def apples_picked_by_caleb : ℕ := apples_picked_by_kayla - 5
def apples_picked_by_suraya : ℕ := apples_picked_by_caleb + 12
def apples_picked_by_mia : ℕ := apples_picked_by_caleb * 2

theorem suraya_picked_less_than_mia :
  abs (apples_picked_by_suraya - apples_picked_by_mia) = 3 :=
by
  sorry

end suraya_picked_less_than_mia_l256_256208


namespace prob_max_atleast_twice_chloe_l256_256394

section
variables {Ω : Type*} [ProbabilitySpace Ω]

noncomputable def probability_favor : ℝ :=
  let A := set.prod (set.Icc (0 : ℝ) 3000) (set.Icc (0 : ℝ) 4500) in
  let B := {xy : ℝ × ℝ | xy.2 >= 2 * xy.1} in
  (Prob (A ∩ B) / Prob A)

theorem prob_max_atleast_twice_chloe : probability_favor = (3 / 8) :=
sorry
end

end prob_max_atleast_twice_chloe_l256_256394


namespace equal_triangle_areas_l256_256288

-- Define points and triangles
variables (A B C P A1 B1 C1 A2 B2 C2 : Type*)

-- Define conditions about the construction: three segments through P parallel to sides of ABC
-- We assume the area function which calculates the area of a triangle
def area {α : Type*} [has_area α] : α → α → α → ℝ := sorry

-- The equivalent conditions given in the problem
-- Using A1, B1, C1 as the points on segments parallel to BC, CA, AB passing through P
-- Using A2, B2, C2 as the points on segments parallel to sides in another configuration

-- Assertion: The area of triangles A1B1C1 and A2B2C2 are equal
theorem equal_triangle_areas :
  area A1 B1 C1 = area A2 B2 C2 := sorry

end equal_triangle_areas_l256_256288


namespace none_takes_own_hat_probability_l256_256706

noncomputable def probability_none_takes_own_hat : ℚ :=
  have total_arrangements := 3.factorial
  have derangements : ℕ := 2
  have probability := (derangements : ℚ) / (total_arrangements : ℚ)
  probability

theorem none_takes_own_hat_probability : probability_none_takes_own_hat = 1 / 3 :=
by
  have total_arrangements := 3.factorial
  have derangements : ℕ := 2
  have probability := (derangements : ℚ) / (total_arrangements : ℚ)
  show probability_none_takes_own_hat = 1 / 3
  sorry

end none_takes_own_hat_probability_l256_256706


namespace soviet_olympiad_1973_l256_256228

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + x

theorem soviet_olympiad_1973 (a b : ℝ) :
  (∀ x : ℝ, f a b x = x → false) → (∀ y : ℝ, f a b (f a b y) = y → false) :=
by {
  intro h,
  intro y,
  intro H,
  apply h,
  sorry
}

end soviet_olympiad_1973_l256_256228


namespace find_number_l256_256358

theorem find_number (x : ℝ) (h : 97 * x - 89 * x = 4926) : x = 615.75 :=
by
  sorry

end find_number_l256_256358


namespace find_EC_l256_256336

variable (A B C D E: Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variable (AB: ℝ) (CD: ℝ) (AC: ℝ) (EC: ℝ)
variable [Parallel : ∀ A B, Prop] [Measure : ∀ A B, Real]

def is_trapezoid (AB: ℝ) (CD: ℝ) := AB = 3 * CD

theorem find_EC 
  (h1 : is_trapezoid AB CD)
  (h2 : AC = 15)
  : EC = 15 / 4 :=
by
  sorry

end find_EC_l256_256336


namespace max_distance_condition_l256_256365

variable (v g θ : ℝ)

def distance (v g θ : ℝ) : ℝ := 
  v^2 * sin (2 * θ) / g

theorem max_distance_condition :
  (θ = π / 4) → (sin θ * Real.log (Real.sec θ + Real.tan θ) = 1) :=
by
  sorry

end max_distance_condition_l256_256365


namespace range_of_k_l256_256523

noncomputable def quadratic_inequality (k : ℝ) := 
  ∀ x : ℝ, 2 * k * x^2 + k * x - (3 / 8) < 0

theorem range_of_k (k : ℝ) :
  (quadratic_inequality k) → -3 < k ∧ k < 0 := sorry

end range_of_k_l256_256523


namespace fill_cards_with_inequalities_l256_256284

theorem fill_cards_with_inequalities :
  ∀ (cards : Fin 101 → Option (Sum (Sum PNat PNat) (PNat → PNat → Prop))), 
    (∀ i : Fin 50, (cards ⟨2 * (i + 1), by linarith⟩).isNone = false) → 
    (∀ i : Fin 50, exists! x : Fin 101, cards x = some (Sum.inr i) → exists! y : Fin 101, x < y ∧ (cards y = some (Sum.inr i) → true → false)) → 
    (∃ f : Fin 51 → Fin 101, ∀ i : Fin 50, 
      (cards (f ⟨2 * (i + 1), by linarith⟩) = some (Sum.inr i)) ∧ 
      ∀ j k : Fin 51, j < k → (cards (f j) = some (Sum.inr i) → cards (f k) = some (Sum.inr i) → false)) :=
by
  sorry

end fill_cards_with_inequalities_l256_256284


namespace initial_roses_count_l256_256516

theorem initial_roses_count 
  (roses_to_mother : ℕ)
  (roses_to_grandmother : ℕ)
  (roses_to_sister : ℕ)
  (roses_kept : ℕ)
  (initial_roses : ℕ)
  (h_mother : roses_to_mother = 6)
  (h_grandmother : roses_to_grandmother = 9)
  (h_sister : roses_to_sister = 4)
  (h_kept : roses_kept = 1)
  (h_initial : initial_roses = roses_to_mother + roses_to_grandmother + roses_to_sister + roses_kept) :
  initial_roses = 20 :=
by
  rw [h_mother, h_grandmother, h_sister, h_kept] at h_initial
  exact h_initial

end initial_roses_count_l256_256516


namespace rope_diameter_is_26mm_l256_256329

noncomputable def smallest_rope_diameter (M : ℝ) (n : ℕ) (alpha : ℝ) (k : ℝ) (q : ℝ) (g : ℝ) : ℝ :=
let W := M * g in
let F_strop := W / n in
let N := F_strop / real.cos (alpha * real.pi / 180) in
let Q := k * N in
let S := Q / q in
let A := (real.pi * 4 * S) in
let D := real.sqrt (A / real.pi) * 1000 in
real.ceil D

theorem rope_diameter_is_26mm : smallest_rope_diameter 20 3 30 6 (10^3) 10 = 26 :=
by
  sorry

end rope_diameter_is_26mm_l256_256329


namespace find_ratio_pq_division_at_y_axis_l256_256874

-- Definition of the points P and Q
structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 4, y := -9 }
def Q : Point := { x := -2, y := 3 }

-- Definition of the section formula for dividing ratio
def sectionFormula (P Q : Point) (λ : ℝ) : (ℝ × ℝ) :=
  let x := (P.x * (1 - λ) + Q.x * λ) / (1 + λ)
  let y := (P.y * (1 - λ) + Q.y * λ) / (1 + λ)
  (x, y)

-- Proof statement to find λ such that P and Q are divided at the y-axis
theorem find_ratio_pq_division_at_y_axis (λ : ℝ) :
  (sectionFormula P Q λ).fst = 0 → λ = 2 :=
by sorry

end find_ratio_pq_division_at_y_axis_l256_256874


namespace max_value_of_b_l256_256828

-- Given integers a and b, if a + b is a root of the equation x^2 + ax + b = 0,
-- then prove the maximum possible value of b is 9.

theorem max_value_of_b (a b : ℤ) (h1 : a + b = root_of_quadratic a b) : b ≤ 9 :=
by
  sorry

def root_of_quadratic (a b : ℤ) : ℤ :=
  -- function to compute one root of the quadratic equation x^2 + ax + b = 0
  let delta := a*a - 4*b
  -a/2 ± sqrt(delta)/2

end max_value_of_b_l256_256828


namespace domain_of_g_l256_256130

theorem domain_of_g (f : ℝ → ℝ) (x : ℝ) (h1 : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x ≠ 0) :
  (0 ≤ 2*x ∧ 2*x ≤ 2) ∧ x ≠ 1 ↔ 0 ≤ x ∧ x < 1 :=
begin
  sorry
end

end domain_of_g_l256_256130


namespace determinant_problem_l256_256794

open Matrix

theorem determinant_problem
  (a b : ℝ) (k : ℝ) :
  det ![
    ![1, k * sin (a - b), sin a],
    ![k * sin (a - b), 1, k * sin b],
    ![sin a, k * sin b, 1]
  ] = 1 - sin a ^ 2 - k ^ 2 * sin b ^ 2 :=
by sorry

end determinant_problem_l256_256794


namespace monochromatic_triangle_exists_l256_256007

theorem monochromatic_triangle_exists (V : Finset ℕ) (E : Finset (Sym2 ℕ)) 
  (hV : V.card = 10) (hE : E = (V.pairs : Finset (Sym2 ℕ)))
  (color : Sym2 ℕ → Prop)
  (hcolor : ∀ e ∈ E, color e ∨ ¬color e) :
  ∃ (T : Finset (Sym2 ℕ)), T.card = 3 ∧ (∀ e ∈ T, color e) ∨ (∀ e ∈ T, ¬color e) :=
by
  sorry

end monochromatic_triangle_exists_l256_256007


namespace sum_of_x_coordinates_l256_256815

theorem sum_of_x_coordinates :
  let f (x : ℝ) := real.abs (x^2 - 4*x + 3)
  let g (x : ℝ) := (25 : ℝ)/4 - x 
  ∃ (x1 x2 : ℝ), (f x1 = g x1) ∧ (f x2 = g x2) ∧ ((x1 + x2) = 3) :=
by
  sorry

end sum_of_x_coordinates_l256_256815


namespace division_of_routes_two_airlines_no_cycles_l256_256350

variable (N : ℕ)
variable (G : SimpleGraph (Fin N))

-- Define the condition provided in the problem
def route_condition (G : SimpleGraph (Fin N)) : Prop :=
  ∀ (k : ℕ) (hk : 2 ≤ k ≤ N),
    ∀ (V_k : Finset (Fin N)), V_k.card = k → G.edge_finset.filter (λ e, e.fst ∈ V_k ∧ e.snd ∈ V_k).card ≤ 2 * k - 2

-- Define the problem hypothesis
theorem division_of_routes_two_airlines_no_cycles
  (hG : route_condition G) : 
  ∃ (f : G.edge_finset → Fin 2), 
    ∀ (c : Fin 2), ¬∃ (e : List G.edge_finset), e.Nodup ∧ (∀ x ∈ e, f x = c) ∧ (e.Last ∈ e.head.Support) :=
sorry

end division_of_routes_two_airlines_no_cycles_l256_256350


namespace number_of_n_not_dividing_g_in_range_l256_256188

def g (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ x, x ≠ n ∧ x ∣ n) (finset.range (n+1))), d

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem number_of_n_not_dividing_g_in_range :
  (finset.filter (λ n, n ∉ (finset.range (101)).filter (λ n, n ∣ g n))
  (finset.Icc 2 100)).card = 29 :=
by
  sorry

end number_of_n_not_dividing_g_in_range_l256_256188


namespace decreasing_interval_a_ge_3_l256_256076

-- Given function f(x) = x² - 2ax + 6
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 6

-- Theorem: If f(x) is decreasing on the interval (-∞, 3), then a ≥ 3.
theorem decreasing_interval_a_ge_3 (a : ℝ) :
  (∀ x : ℝ, x < 3 → (f x a)' < 0) → (a ≥ 3) := by
  -- skipping proof steps
  sorry

end decreasing_interval_a_ge_3_l256_256076


namespace domain_of_sqrt_and_fraction_l256_256260

noncomputable def domain_of_function : Set ℝ :=
  {x : ℝ | x > -1}

theorem domain_of_sqrt_and_fraction :
  ∀ x : ℝ, (∃ y : ℝ, y = sqrt (x + 1) + 1 / (x + 1)) ↔ x > -1 :=
by
  -- proof omitted
  sorry

end domain_of_sqrt_and_fraction_l256_256260


namespace pieces_per_sister_l256_256030

-- Defining the initial conditions
def initial_cake_pieces : ℕ := 240
def percentage_eaten : ℕ := 60
def number_of_sisters : ℕ := 3

-- Defining the statements to be proved
theorem pieces_per_sister (initial_cake_pieces : ℕ) (percentage_eaten : ℕ) (number_of_sisters : ℕ) :
  let pieces_eaten := (percentage_eaten * initial_cake_pieces) / 100
  let remaining_pieces := initial_cake_pieces - pieces_eaten
  let pieces_per_sister := remaining_pieces / number_of_sisters
  pieces_per_sister = 32 :=
by 
  sorry

end pieces_per_sister_l256_256030


namespace five_digit_numbers_count_l256_256509

/-
Problem:
How many five-digit numbers can be formed using the digits 0, 1, 3, 5, and 7 without repeating any digits and ensuring that 5 is not in the tens place?
-/

theorem five_digit_numbers_count : 
  ∃ n : ℕ, 
    (n = 78) ∧
    ∀ (digits : List ℕ), 
      digits.length = 5 ∧ 
      digits.nodup ∧ 
      digits.all (λ d, d ∈ [0, 1, 3, 5, 7]) ∧ 
      digits.head ≠ 0 ∧ 
      digits.nth 3 ≠ some 5 →
      (digits.permutations.length = n) :=
sorry

end five_digit_numbers_count_l256_256509


namespace largest_k_for_g_l256_256443

theorem largest_k_for_g (k : ℝ) (g : ℝ → ℝ) (h_def : ∀ x, g x = x^2 - 7 * x + k) :
  (4 ∈ set.range g) → k ≤ 65 / 4 := 
sorry

end largest_k_for_g_l256_256443


namespace incorrect_statement_l256_256087

noncomputable theory

def sequence_bn (n : ℕ) : ℤ := 2 * n - 1

def sequence_cn (n : ℕ) : ℤ := 3 * n - 2

def sequence_an (n : ℕ) (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : ℝ :=
  x * (sequence_bn n).toReal + y * (sequence_cn n).toReal

theorem incorrect_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1):
  ∃ n : ℕ, n ≥ 2 ∧ ¬ (sequence_bn n < (sequence_an n x y hx hy hxy).toInt ∧ (sequence_an n x y hx hy hxy).toInt < sequence_cn n) :=
sorry

end incorrect_statement_l256_256087


namespace random_event_count_is_3_l256_256770

def is_random_event (e : Type) : Prop := ∃ cond, e.may_or_may_not_occur cond

def event1 := is_random_event "Throwing two dice twice in a row, and both times getting 2 points"
def event2 := ¬is_random_event "On Earth, a pear falling from a tree will fall down if not caught"
def event3 := is_random_event "Someone winning the lottery"
def event4 := is_random_event "Already having one daughter, then having a boy the second time"
def event5 := ¬is_random_event "Under standard atmospheric pressure, water heated to 90°C will boil"

theorem random_event_count_is_3 : 
  (∃ e1 e2 e3 : Type, 
    event1 e1 ∧ event3 e2 ∧ event4 e3 ∧
    (event2 → ¬is_random_event e1) ∧
    (event5 → ¬is_random_event e2)) :=
sorry

end random_event_count_is_3_l256_256770


namespace max_unique_minions_height_of_dave_minions_per_chair_l256_256338

-- Problem 1: Max Unique Minions
theorem max_unique_minions : (2 * 4 * 3 = 24) :=
by
  sorry

-- Problem 2: Height of Dave
noncomputable def pi : Real := Real.pi

theorem height_of_dave :
  let r := 2 in
  let V_hemisphere := (2/3) * pi * (r^3) in
  let V_two_hemispheres := 2 * V_hemisphere in
  let V_cylinder (h : Real) := pi * (r^2) * h in
  (20 * pi = V_two_hemispheres + V_cylinder (7/3)) :=
by
  sorry

-- Problem 4: Minions per Chair
theorem minions_per_chair :
  let A_chair := pi * (r^2) in
  let A_minion := pi * ((r / 5) ^ 2) in
  (A_chair / A_minion = 25) :=
by
  sorry

end max_unique_minions_height_of_dave_minions_per_chair_l256_256338


namespace g_is_even_l256_256565

noncomputable def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end g_is_even_l256_256565


namespace probability_no_cowboys_picks_own_hat_l256_256707

def derangements (n : Nat) : Nat :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangements (n - 1) + derangements (n - 2))

theorem probability_no_cowboys_picks_own_hat : 
  let total_arrangements := Nat.factorial 3
  let favorable_arrangements := derangements 3
  let probability := (favorable_arrangements : ℚ) / (total_arrangements : ℚ)
  probability = 1 / 3 :=
by
  let total_arrangements := 6
  let favorable_arrangements := 2
  have h : probability = (2 : ℚ) / (6 : ℚ) := by rfl
  rw h
  calc (2 : ℚ) / (6 : ℚ)
          = 1 / 3 : by norm_num
  rw [h]
  sorry

end probability_no_cowboys_picks_own_hat_l256_256707


namespace cocktail_cost_per_litre_is_accurate_l256_256270

noncomputable def mixed_fruit_juice_cost_per_litre : ℝ := 262.85
noncomputable def acai_berry_juice_cost_per_litre : ℝ := 3104.35
noncomputable def mixed_fruit_juice_litres : ℝ := 35
noncomputable def acai_berry_juice_litres : ℝ := 23.333333333333336

noncomputable def cocktail_total_cost : ℝ := 
  (mixed_fruit_juice_cost_per_litre * mixed_fruit_juice_litres) +
  (acai_berry_juice_cost_per_litre * acai_berry_juice_litres)

noncomputable def cocktail_total_volume : ℝ := 
  mixed_fruit_juice_litres + acai_berry_juice_litres

noncomputable def cocktail_cost_per_litre : ℝ := 
  cocktail_total_cost / cocktail_total_volume

theorem cocktail_cost_per_litre_is_accurate : 
  abs (cocktail_cost_per_litre - 1399.99) < 0.01 := by
  sorry

end cocktail_cost_per_litre_is_accurate_l256_256270


namespace num_values_of_n_l256_256178

-- Definitions of proper positive integer divisors and function g(n)
def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d > 1 ∧ d < n ∧ n % d = 0) (List.range (n + 1))

def g (n : ℕ) : ℕ :=
  List.prod (proper_divisors n)

-- Condition: 2 ≤ n ≤ 100 and n does not divide g(n)
def n_does_not_divide_g (n : ℕ) : Prop :=
  2 ≤ n ∧ n ≤ 100 ∧ ¬ (n ∣ g n)

-- Main theorem statement
theorem num_values_of_n : 
  (Finset.card (Finset.filter n_does_not_divide_g (Finset.range 101))) = 31 :=
by
  sorry

end num_values_of_n_l256_256178


namespace var_X_is_86_over_225_l256_256223

/-- The probability of Person A hitting the target is 2/3. -/
def prob_A : ℚ := 2 / 3

/-- The probability of Person B hitting the target is 4/5. -/
def prob_B : ℚ := 4 / 5

/-- The events of A and B hitting or missing the target are independent. -/
def independent_events : Prop := true -- In Lean, independence would involve more complex definitions.

def prob_X (x : ℕ) : ℚ :=
  if x = 0 then (1 - prob_A) * (1 - prob_B)
  else if x = 1 then (1 - prob_A) * prob_B + (1 - prob_B) * prob_A
  else if x = 2 then prob_A * prob_B
  else 0

/-- Expected value of X -/
noncomputable def expect_X : ℚ :=
  0 * prob_X 0 + 1 * prob_X 1 + 2 * prob_X 2

/-- Variance of X -/
noncomputable def var_X : ℚ :=
  (0 - expect_X) ^ 2 * prob_X 0 +
  (1 - expect_X) ^ 2 * prob_X 1 +
  (2 - expect_X) ^ 2 * prob_X 2

theorem var_X_is_86_over_225 : var_X = 86 / 225 :=
by {
  sorry
}

end var_X_is_86_over_225_l256_256223


namespace heidi_zoe_paint_fraction_l256_256522

theorem heidi_zoe_paint_fraction (H_period : ℝ) (HZ_period : ℝ) :
  (H_period = 60 → HZ_period = 40 → (8 / 40) = (1 / 5)) :=
by intros H_period_eq HZ_period_eq
   sorry

end heidi_zoe_paint_fraction_l256_256522


namespace AB_tangent_to_circumcircle_l256_256159

open Real

variables {A B C D : Type*}
variables {angle_ACB angle_ADB : ℝ}
variables {AD DC : ℝ}

-- Given conditions
def angle_ACB := 45 * pi / 180
def angle_ADB := 60 * pi / 180
def AD_DC_ratio := 2 / 1

-- The statement to be proved
theorem AB_tangent_to_circumcircle (h1 : angle_ACB = 45 * pi / 180)
  (h2 : angle_ADB = 60 * pi / 180) (h3 : AD_DC_ratio = 2 / 1) :
  AB_is_tangent A B C D :=
sorry

end AB_tangent_to_circumcircle_l256_256159


namespace unit_vectors_have_equal_squares_l256_256473

variables {V : Type*} [inner_product_space ℝ V] 

def is_unit_vector (v : V) : Prop := ∥v∥ = 1

theorem unit_vectors_have_equal_squares 
  (a b : V) (ha : is_unit_vector a) (hb : is_unit_vector b) : 
  a • a = b • b :=
by
  sorry

end unit_vectors_have_equal_squares_l256_256473


namespace arithmetic_sequence_a_n_sum_b_n_l256_256588

/-- Define the arithmetic sequence as a function. -/
def arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ := 
  a1 + (n - 1) * d

/-- Define the conditions of the problem. -/
def satisfies_conditions (a1 d : ℕ) : Prop :=
  arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 5 = 7 ∧ 
  arithmetic_sequence a1 d 3 * arithmetic_sequence a1 d 4 = 12 ∧ 
  d > 0

/-- Define the sequence bn. -/
def b (a1 d : ℕ) (n : ℕ) : ℚ := 
  (arithmetic_sequence a1 d (n + 1) : ℚ) / (arithmetic_sequence a1 d (n + 2) : ℚ) + 
  (arithmetic_sequence a1 d (n + 2) : ℚ) / (arithmetic_sequence a1 d (n + 1) : ℚ) - 2

/-- Define the sum of the first n terms of the sequence bn. -/
def T (a1 d : ℕ) (n : ℕ) : ℚ :=
  (finset.range n).sum (b a1 d)

theorem arithmetic_sequence_a_n (a1 d : ℕ) (n : ℕ) (h : satisfies_conditions a1 d) :
  arithmetic_sequence a1 d 1 = 1 ∧ ∀ n, arithmetic_sequence a1 d n = n := sorry

theorem sum_b_n (a1 d : ℕ) (n : ℕ) (h : satisfies_conditions a1 d):
  T a1 d n = (n + 1 : ℚ) / (2 * n + 4) := sorry

end arithmetic_sequence_a_n_sum_b_n_l256_256588


namespace triangle_acute_of_tan_product_pos_l256_256846

theorem triangle_acute_of_tan_product_pos
  (A B C : ℝ)
  (h1 : A + B + C = π)
  (h2 : 0 < A)
  (h3 : 0 < B)
  (h4 : 0 < C)
  (h5 : tan A * tan B * tan C > 0) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 :=
by
  sorry

end triangle_acute_of_tan_product_pos_l256_256846


namespace non_seniors_play_instrument_count_l256_256531

def student_data := {num_students : ℕ, num_seniors : ℕ, num_non_seniors : ℕ, seniors_play_instrument : ℕ, non_seniors_not_play_instrument : ℕ}

theorem non_seniors_play_instrument_count (data : student_data)
  (h1 : data.num_students = 500)
  (h2 : data.seniors_play_instrument = 40 * data.num_seniors / 100)
  (h3 : data.non_seniors_not_play_instrument = 30 * data.num_non_seniors / 100)
  (h4 : data.num_non_seniors = data.num_students - data.num_seniors)
  (h5 : data.seniors_play_instrument + data.non_seniors_not_play_instrument = 234)
  : data.num_non_seniors * 70 / 100 = 154 :=
sorry

end non_seniors_play_instrument_count_l256_256531


namespace symmetric_points_product_l256_256151

theorem symmetric_points_product (a b : ℝ) 
    (h1 : a + 2 = -4) 
    (h2 : b = 2) : 
    a * b = -12 := 
sorry

end symmetric_points_product_l256_256151


namespace yn_minus_one_is_perfect_square_l256_256504

theorem yn_minus_one_is_perfect_square
  (x y : ℕ → ℕ)
  (H : ∀ n : ℕ, x n + real.sqrt 2 * y n = real.sqrt 2 * (3 + 2 * real.sqrt 2)^(2^n)) :
  ∀ n : ℕ, ∃ k : ℕ, y n - 1 = k * k :=
by
  sorry

end yn_minus_one_is_perfect_square_l256_256504


namespace tangerine_sales_difference_l256_256004

theorem tangerine_sales_difference :
    let sales := [300, -400, -200, 100, -600, 1200, 500]
    in (maxList sales - minList sales = 1800) :=
by
    let sales := [300, -400, -200, 100, -600, 1200, 500]
    have hMax : maxList sales = 1200 := by sorry
    have hMin : minList sales = -600 := by sorry
    sorry

end tangerine_sales_difference_l256_256004


namespace friends_belong_special_team_l256_256298

-- Define a type for students
universe u
variable {Student : Type u}

-- Assume a friendship relation among students
variable (friend : Student → Student → Prop)

-- Assume the conditions as given in the problem
variable (S : Student → Set (Set Student))
variable (students : Set Student)
variable (S_non_empty : ∀ v : Student, S v ≠ ∅)
variable (friendship_condition : 
  ∀ u v : Student, friend u v → 
    (∃ w : Student, S u ∩ S v ⊇ S w))
variable (special_team : ∀ (T : Set Student),
  (∃ v ∈ T, ∀ w : Student, w ∈ T → friend v w) ↔
  (∃ v ∈ T, ∀ w : Student, friend v w → w ∈ T))

-- Prove that any two friends belong to some special team
theorem friends_belong_special_team :
  ∀ u v : Student, friend u v → 
    (∃ T : Set Student, T ∈ S u ∩ S v ∧ 
      (∃ w ∈ T, ∀ x : Student, friend w x → x ∈ T)) :=
by
  sorry  -- Proof omitted


end friends_belong_special_team_l256_256298


namespace length_of_CB_l256_256135

-- Given conditions
variables (A B C D E : Type) [geometry A B C] [geometry D E]
variables (CD DA CE : ℝ)
variable h_parallel : parallel DE AB

-- Define the given lengths
def CD_length : ℝ := 8
def DA_length : ℝ := 12
def CE_length : ℝ := 9

--- The main statement to prove 
theorem length_of_CB :
  ∀ (CB : ℝ), parallel DE AB ∧ CD = CD_length ∧ DA = DA_length ∧ CE = CE_length → CB = 22.5 :=
begin
  sorry
end

end length_of_CB_l256_256135


namespace dima_always_wins_l256_256003

theorem dima_always_wins (n : ℕ) (P : Prop) : 
  (∀ (gosha dima : ℕ → Prop), 
    (∀ k : ℕ, k < n → (gosha k ∨ dima k))
    ∧ (∀ i : ℕ, i < 14 → (gosha i ∨ dima i))
    ∧ (∃ j : ℕ, j ≤ n ∧ (∃ k ≤ j + 7, dima k))
    ∧ (∃ l : ℕ, l ≤ 14 ∧ (∃ m ≤ l + 7, dima m))
    → P) → P := sorry

end dima_always_wins_l256_256003


namespace proof_problem_l256_256786

noncomputable def problem_expression : ℝ :=
  50 * 39.96 * 3.996 * 500

theorem proof_problem : problem_expression = (3996 : ℝ)^2 :=
by
  sorry

end proof_problem_l256_256786


namespace operation_correct_l256_256524

def operation (x y : ℝ) := x^2 + y^2 + 12

theorem operation_correct :
  operation (Real.sqrt 6) (Real.sqrt 6) = 23.999999999999996 :=
by
  -- proof omitted
  sorry

end operation_correct_l256_256524


namespace fair_coin_toss_is_random_event_l256_256263

-- Definition of a fair coin.
def is_fair_coin (C : Type) (event : C → Prop) :=
  ∃ (heads tails : C), event heads ∧ ¬ event tails

-- Definition stating that both outcomes are equally likely.
def equally_likely (C : Type) (prob : C → ℝ) (heads tails : C) :=
  prob heads = prob tails

-- The event is classified as a random event if both outcomes are equally likely and neither is guaranteed.
theorem fair_coin_toss_is_random_event (C : Type) (event : C → Prop) (prob : C → ℝ) :
  is_fair_coin C event ∧ (∃ heads tails, equally_likely C prob heads tails) → event ∨ (¬ event : C) :=
begin
  sorry
end

end fair_coin_toss_is_random_event_l256_256263


namespace single_elimination_tournament_games_l256_256969

theorem single_elimination_tournament_games (n : ℕ) :
  n = 32 →
  (∑ i in Finset.range n, 1) - 1 = 31 :=
by
  intros h
  rw h
  -- Proof is required here, but we'll skip it
  sorry

end single_elimination_tournament_games_l256_256969


namespace Benjamin_has_45_presents_l256_256805

-- Define the number of presents each person has
def Ethan_presents : ℝ := 31.5
def Alissa_presents : ℝ := Ethan_presents + 22
def Benjamin_presents : ℝ := Alissa_presents - 8.5

-- The statement we need to prove
theorem Benjamin_has_45_presents : Benjamin_presents = 45 :=
by
  -- on the last line, we type sorry to skip the actual proof
  sorry

end Benjamin_has_45_presents_l256_256805


namespace range_of_a_l256_256077

-- Definition for the given function f(x)
def f (x a : ℝ) : ℝ := (exp x - a) / x

-- Definition of the monotonicity condition on the interval [2, 4]
def f_monotonic (a : ℝ) : Prop :=
  ∀ x ∈ set.Icc (2:ℝ) 4, deriv (λ x, f x a) x ≥ 0

-- The theorem we want to prove
theorem range_of_a : ∀ a : ℝ, f_monotonic a ↔ a ≥ -exp 2 :=
begin
  sorry
end

end range_of_a_l256_256077


namespace mike_chocolate_squares_l256_256572

theorem mike_chocolate_squares (M : ℕ) (h1 : 65 = 3 * M + 5) : M = 20 :=
by {
  -- proof of the theorem (not included as per instructions)
  sorry
}

end mike_chocolate_squares_l256_256572


namespace boy_age_proof_l256_256455

theorem boy_age_proof (P X : ℕ) (hP : P = 16) (hcond : P - X = (P + 4) / 2) : X = 6 :=
by
  sorry

end boy_age_proof_l256_256455


namespace correct_average_marks_l256_256673

theorem correct_average_marks
  (n : ℕ) (avg_mks wrong_mk correct_mk correct_avg_mks : ℕ)
  (H1 : n = 10)
  (H2 : avg_mks = 100)
  (H3 : wrong_mk = 50)
  (H4 : correct_mk = 10)
  (H5 : correct_avg_mks = 96) :
  (n * avg_mks - wrong_mk + correct_mk) / n = correct_avg_mks :=
by
  sorry

end correct_average_marks_l256_256673


namespace number_of_small_pipes_needed_l256_256273

theorem number_of_small_pipes_needed :
  let diameter_large := 8
  let diameter_small := 1
  let radius_large := diameter_large / 2
  let radius_small := diameter_small / 2
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let num_small_pipes := area_large / area_small
  num_small_pipes = 64 :=
by
  sorry

end number_of_small_pipes_needed_l256_256273


namespace curve_equation_l256_256018

theorem curve_equation :
  (∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0 ∧ x = 3 ∧ y = 2) ∧
  (∃ (C : ℝ), 
    8 * 3 + 6 * 2 + C = 0 ∧
    8 * x + 6 * y + C = 0 ∧
    4 * x + 3 * y - 18 = 0 ∧
    ∀ x y, 6 * x - 8 * y + 3 = 0 → 
    4 * x + 3 * y - 18 = 0) ∧
  (∃ (a : ℝ), ∀ x y, (x + 1)^2 + 1 = (x - 1)^2 + 9 →
    ((x - 2)^2 + y^2 = 10 ∧ a = 2)) :=
sorry

end curve_equation_l256_256018


namespace min_phi_for_shifted_odd_l256_256891

noncomputable def original_function (x : ℝ) : ℝ := cos (2 * x + π / 6)

def shifted_function (φ : ℝ) (h : φ > 0) (x : ℝ) : ℝ := cos (2 * x + π / 6 - 2 * φ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

theorem min_phi_for_shifted_odd :
  ∃ φ : ℝ, φ > 0 ∧ is_odd (shifted_function φ) ∧ φ = π / 3 :=
by
  sorry

end min_phi_for_shifted_odd_l256_256891


namespace max_square_sum_leq_l256_256229

theorem max_square_sum_leq (n : ℕ) (a : Fin n → ℝ) (h_sum: ∑ i, a i = 0) :
  (∃ k, 1 ≤ k ∧ k ≤ n ∧ (a k)^2 = Finset.max (Finset.range n) (λ k, (a k)^2)) →
  (∑ i in Finset.range (n - 1), (a i - a (i + 1)) ^ 2) * (n / 3) ≥ (∃ k, (a k) ^ 2) :=
sorry

end max_square_sum_leq_l256_256229


namespace not_divide_g_count_30_l256_256191

-- Define the proper positive divisors function
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range (n - 1)).filter (λ d, d > 0 ∧ n % d = 0)

-- Define the product of proper divisors function
def g (n : ℕ) : ℕ :=
  proper_divisors n |>.prod

-- Define the main theorem
theorem not_divide_g_count_30 : 
  (Finset.range 99).filter (λ n, 2 ≤ n + 1 ∧ n + 1 ≤ 100 ∧ ¬(n + 1) ∣ g (n + 1)).card = 30 := 
  by
  sorry

end not_divide_g_count_30_l256_256191


namespace solution_set_abs_inequality_l256_256692

theorem solution_set_abs_inequality : {x : ℝ | |1 - 2 * x| < 3} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end solution_set_abs_inequality_l256_256692


namespace number_of_elements_in_S_10_l256_256944

def f (x k : ℝ) : ℝ := (x + k) / x

def f_seq (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then f x 10
  else f (f_seq (n - 1) x) 10

def S_k (k : ℝ) : Set ℝ :=
  {x : ℝ | ∃ n : ℕ, n > 0 ∧ f_seq n x = x}

def S_10 : Set ℝ := S_k 10

theorem number_of_elements_in_S_10 : ∃! x, x ∈ S_10 :=
  -- Outline:
  -- Two solutions:
  --   x = (1 + sqrt 41) / 2
  --   x = (1 - sqrt 41) / 2
  -- Conclusion:
  --   The set S_10 contains exactly 2 elements.
  sorry

end number_of_elements_in_S_10_l256_256944


namespace triangle_angle_bisector_proportionality_l256_256160

-- Main theorem statement
theorem triangle_angle_bisector_proportionality (ABC : Type) [euclidean_geometry ABC]
  (A B C D E F : ABC) :
  is_angle_bisector A B C D → 
  (line_through D E B) →
  (line_through D F C) →
  (line_through A B C) →
  (line_through A C B) →
  (point_on_line D B C) →
  (point_on_line E A B) →
  (point_on_line F A C) →
  (line_segment_ratio B C D E F : B C D E F) : 
  AB / AC = BE * AF / (CF * AE) :=
  sorry

end triangle_angle_bisector_proportionality_l256_256160


namespace ramu_loss_percent_correct_l256_256660

def cost_of_car : ℕ := 42000
def cost_of_repairs : ℕ := 13000
def usd_to_inr (usd : ℕ) : ℕ := usd * 75
def php_to_inr (php : ℕ) : ℕ := php * 3 / 2
def cost_of_taxes_in_inr : ℕ := usd_to_inr 500
def cost_of_insurance_in_inr : ℕ := php_to_inr 3000
def selling_price : ℕ := 83000
def total_cost : ℕ := cost_of_car + cost_of_repairs + cost_of_taxes_in_inr + cost_of_insurance_in_inr
def profit_or_loss : ℤ := selling_price - total_cost
def loss_percent : ℚ := (abs profit_or_loss / total_cost) * 100

theorem ramu_loss_percent_correct :
  loss_percent ≈ 14.43 := sorry

end ramu_loss_percent_correct_l256_256660


namespace find_m_min_value_l256_256864

theorem find_m (f : ℝ → ℝ) (h : ∀ x, f(x) = |(1/2) * x| - |(1/2) * x - m| ∧ ∀ x ∈ ℝ, f(x) ≤ 4) : 
  m = 4 := 
sorry

theorem min_value (m x : ℝ) (h1 : m > 0) (h2 : 0 < x ∧ x < m/2) : 
  ∃ minimum, minimum = 4 ∧ minimum = 2 * (1/|x| + 1/|x - 2|) := 
sorry

end find_m_min_value_l256_256864


namespace num_values_of_n_l256_256179

-- Definitions of proper positive integer divisors and function g(n)
def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d > 1 ∧ d < n ∧ n % d = 0) (List.range (n + 1))

def g (n : ℕ) : ℕ :=
  List.prod (proper_divisors n)

-- Condition: 2 ≤ n ≤ 100 and n does not divide g(n)
def n_does_not_divide_g (n : ℕ) : Prop :=
  2 ≤ n ∧ n ≤ 100 ∧ ¬ (n ∣ g n)

-- Main theorem statement
theorem num_values_of_n : 
  (Finset.card (Finset.filter n_does_not_divide_g (Finset.range 101))) = 31 :=
by
  sorry

end num_values_of_n_l256_256179


namespace monotonic_increasing_interval_l256_256685

def f (x : ℝ) : ℝ := abs (x + 1)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, x ∈ [-1, +∞) → ∀ y : ℝ, y > x → f x ≤ f y :=
by
  intros x hx y hy
  sorry

end monotonic_increasing_interval_l256_256685


namespace isosceles_right_triangle_length_eq_l256_256742

variables {A B C D E K L : Type*}
variables [IsoscelesRightTriangle ABC]
variables [OnLegs D E A C B AC BC]
variables [EqualLengths D E CD CE]
variables [Perpendiculars D C AE]
variables [IntersectionPoints A B K L]
variables [Centers A C B K L ACBC]

theorem isosceles_right_triangle_length_eq
  (h_iso : IsoscelesRightTriangle ABC)
  (h_on_legs : OnLegs D E A C B AC BC)
  (h_equal_lengths : EqualLengths D E CD CE)
  (h_perpendiculars : Perpendiculars D C AE)
  (h_intersection : IntersectionPoints A B K L)
  (h_centers : Centers A C B K L ACBC) :
  KL = BL :=
sorry

end isosceles_right_triangle_length_eq_l256_256742


namespace cost_of_camel_l256_256341

theorem cost_of_camel
  (C H O E : ℝ)
  (h1 : 10 * C = 24 * H)
  (h2 : 16 * H = 4 * O)
  (h3 : 6 * O = 4 * E)
  (h4 : 10 * E = 140000) :
  C = 5600 :=
by
  -- Skipping the proof steps
  sorry

end cost_of_camel_l256_256341


namespace inverse_sum_l256_256165

def f (x : ℝ) : ℝ := if x < 15 then x + 2 else 2 * x - 3

theorem inverse_sum : f⁻¹(5) + f⁻¹(40) = 24.5 :=
by
  sorry

end inverse_sum_l256_256165


namespace problem_1_problem_2_problem_3_l256_256080

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := sin x + a * cos x

theorem problem_1 (h : f (π / 3) a = 0) : a = -sqrt 3 :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (f x a) ^ 2 - 2

theorem problem_2 (h : ∀ x : ℝ, x ∈ Ioo (π / 4) (2 * π / 3) → (sin x - sqrt 3 * cos x) ^ 2 - 2 = g x (- sqrt 3)) : 
  set.image (g (- sqrt 3)) (Ioo (π / 4) (2 * π / 3)) = Ico (-2) 1 := 
sorry

theorem problem_3 
  (h : g (a / 2) a = -sqrt 3 / 4) 
  (h_interval : π / 6 < a ∧ a < 2 * π / 3) 
  : cos (α + 3 * π / 2) = (3 + sqrt 61) / 16 :=
sorry

end problem_1_problem_2_problem_3_l256_256080


namespace sqrt_difference_inequality_l256_256980

theorem sqrt_difference_inequality (x : ℝ) (hx : x ≥ 4) : 
  sqrt (x - 3) - sqrt (x - 1) > sqrt (x - 4) - sqrt (x - 2) :=
sorry

end sqrt_difference_inequality_l256_256980


namespace coeff_m5n5_in_m_plus_n_pow_10_l256_256302

theorem coeff_m5n5_in_m_plus_n_pow_10 :
  binomial (10, 5) = 252 := by
sorry

end coeff_m5n5_in_m_plus_n_pow_10_l256_256302


namespace katelyn_integer_mod_11_probability_l256_256163

theorem katelyn_integer_mod_11_probability :
  let p : ℚ := 11 / 20,
  ∃ a b : ℕ, Nat.coprime a b ∧ p = a / b ∧ a + b = 31 :=
by
  sorry

end katelyn_integer_mod_11_probability_l256_256163


namespace triangle_OH_intersection_and_ratio_l256_256942

theorem triangle_OH_intersection_and_ratio {A B C O H P Q : Point} (h_non_eq : ¬Equilateral △ABC)
  (h_acute : Acute △ABC) (h_angle_A : ∠BAC = 60°) (h_O : Circumcenter O △ABC)
  (h_H : Orthocenter H △ABC) (h_PQ_on_OH : Line_OH_intersects_OH △ABC O H P Q) :
  (Line_OH_intersects_AB_AC △ABC O H P Q) ∧ (∃ (s t : ℝ), Area_△APQ s ∧ Area_BQQC t ∧ (4/5 < s/t ∧ s/t < 1)) :=
sorry

end triangle_OH_intersection_and_ratio_l256_256942


namespace inequality_sum_l256_256880

theorem inequality_sum 
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≥ a2)
  (h2 : a2 ≥ a3)
  (h3 : a3 > 0)
  (h4 : b1 ≥ b2)
  (h5 : b2 ≥ b3)
  (h6 : b3 > 0)
  (h7 : a1 * a2 * a3 = b1 * b2 * b3)
  (h8 : a1 - a3 ≤ b1 - b3) :
  a1 + a2 + a3 ≤ 2 * (b1 + b2 + b3) := 
sorry

end inequality_sum_l256_256880


namespace line_equation_standard_form_l256_256677

def point : ℝ × ℝ := (2, 1)
def slope : ℝ := -2

theorem line_equation_standard_form : 
  ∃ A B C : ℤ, A * 2 + B * 1 + C = 0 ∧ 
  A * (2 : ℝ) + B * (1 : ℝ) + C = 0 ∧
  B * slope = -A ∧
  2 * (2 : ℝ) + (1 : ℝ) - (5 : ℝ) = 0 := 
sorry

end line_equation_standard_form_l256_256677


namespace mr_desmond_toys_l256_256614

theorem mr_desmond_toys (toys_for_elder : ℕ) (h1 : toys_for_elder = 60)
  (h2 : ∀ (toys_for_younger : ℕ), toys_for_younger = 3 * toys_for_elder) : 
  ∃ (total_toys : ℕ), total_toys = 240 :=
by {
  sorry
}

end mr_desmond_toys_l256_256614


namespace matrix_linear_combination_l256_256171

noncomputable section

open Matrix

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (M : Matrix (Fin 2) (Fin 2) α) (v w : α)

def mv_eq_v : M.mul_vec v = ![2, -3] := sorry
def mw_eq_w : M.mul_vec w = ![4, 1] := sorry

theorem matrix_linear_combination :
  M.mul_vec (3 • v - 2 • w) = ![-2, -11] :=
begin
  sorry
end

end matrix_linear_combination_l256_256171


namespace find_m_l256_256502

theorem find_m (m : ℝ) : 
  {0, 1, 2} ∩ {1, m} = {1, m} → (m = 0 ∨ m = 2) :=
by
  sorry

end find_m_l256_256502


namespace front_view_triangle_not_cylinder_l256_256680

theorem front_view_triangle_not_cylinder (G : Type) [geometric_body G] (front_view_is_triangle : is_triangle (front_view G)) : ¬ is_cylinder G :=
sorry

end front_view_triangle_not_cylinder_l256_256680


namespace tangents_perpendicular_l256_256247

theorem tangents_perpendicular (a : ℝ) (x₀ : ℝ) (h₀ : a ≠ 0) (h₁ : cos x₀ = a * tan x₀) : 
  (-(sin x₀)) * (a * (1 + (tan x₀)^2)) = -1 :=
by 
  sorry

end tangents_perpendicular_l256_256247


namespace consecutive_even_product_6digit_l256_256436

theorem consecutive_even_product_6digit :
  ∃ (a b c : ℕ), 
  (a % 2 = 0) ∧ (b = a + 2) ∧ (c = a + 4) ∧ 
  (Nat.digits 10 (a * b * c)).length = 6 ∧ 
  (Nat.digits 10 (a * b * c)).head! = 2 ∧ 
  (Nat.digits 10 (a * b * c)).getLast! = 2 ∧ 
  (a * b * c = 287232) :=
by
  sorry

end consecutive_even_product_6digit_l256_256436


namespace rohan_monthly_salary_expenses_l256_256663

theorem rohan_monthly_salary_expenses 
    (food_expense_pct : ℝ)
    (house_rent_expense_pct : ℝ)
    (entertainment_expense_pct : ℝ)
    (conveyance_expense_pct : ℝ)
    (utilities_expense_pct : ℝ)
    (misc_expense_pct : ℝ)
    (monthly_saved_amount : ℝ)
    (entertainment_expense_increase_after_6_months : ℝ)
    (conveyance_expense_decrease_after_6_months : ℝ)
    (monthly_salary : ℝ)
    (savings_pct : ℝ)
    (new_savings_pct : ℝ) : 
    (food_expense_pct + house_rent_expense_pct + entertainment_expense_pct + conveyance_expense_pct + utilities_expense_pct + misc_expense_pct = 90) → 
    (100 - (food_expense_pct + house_rent_expense_pct + entertainment_expense_pct + conveyance_expense_pct + utilities_expense_pct + misc_expense_pct) = savings_pct) → 
    (monthly_saved_amount = monthly_salary * savings_pct / 100) → 
    (entertainment_expense_pct + entertainment_expense_increase_after_6_months = 20) → 
    (conveyance_expense_pct - conveyance_expense_decrease_after_6_months = 7) → 
    (new_savings_pct = 100 - (30 + 25 + (entertainment_expense_pct + entertainment_expense_increase_after_6_months) + (conveyance_expense_pct - conveyance_expense_decrease_after_6_months) + 5 + 5)) → 
    monthly_salary = 15000 ∧ new_savings_pct = 8 := 
sorry

end rohan_monthly_salary_expenses_l256_256663


namespace minimum_shots_l256_256972

-- Define the 8x8 checkerboard and the conditions.
def checkerboard := fin 8 × fin 8

-- Define a ship as a set of 3 contiguous cells.
structure ship :=
  (cells : set checkerboard)
  (size : cells.card = 3)
  (contiguous : ∀ (x y z : checkerboard), x ∈ cells → y ∈ cells → z ∈ cells → (x.1 = y.1 ∧ y.1 = z.1 ∧ (x.2 < y.2 ∧ y.2 < z.2 ∨ z.2 < y.2 ∧ y.2 < x.2)) ∨ 
    (x.2 = y.2 ∧ y.2 = z.2 ∧ (x.1 < y.1 ∧ y.1 < z.1 ∨ z.1 < y.1 ∧ y.1 < x.1)))

-- Define a configuration of ships on the board.
structure configuration :=
  (ships : vector ship 8)
  (no_overlap : ∀ (s1 s2 : ship), s1 ∈ ships.to_list → s2 ∈ ships.to_list → s1 ≠ s2 → disjoint s1.cells s2.cells)

-- Define a shot that covers all cells in a row or column.
inductive shot
| row : fin 8 → shot
| column : fin 8 → shot

-- Define function to determine if a shot hits a ship.
def shot_hits (s : shot) (c : configuration) : bool :=
  match s with
  | shot.row r => c.ships.to_list.any (λ sh, sh.cells.any (λ cell, cell.1 = r))
  | shot.column col => c.ships.to_list.any (λ sh, sh.cells.any (λ cell, cell.2 = col))
  end

-- Define the proof problem: proving that a minimum of 2 shots are required to guarantee hitting at least one ship.
theorem minimum_shots (c : configuration) : ∃ s1 s2 : shot, s1 ≠ s2 ∧ (shot_hits s1 c = tt ∨ shot_hits s2 c = tt) :=
sorry

end minimum_shots_l256_256972


namespace line_intersects_ellipse_l256_256410

theorem line_intersects_ellipse (k : ℝ) : 
  (let y := k * 2 + 1 - 2 * k in y < sqrt ((1 - (2 ^ 2) / 9) * 4)) →
  ∃ x y : ℝ, (y = k * x + 1 - 2 * k) ∧ (x^2 / 9 + y^2 / 4 = 1) :=
by
  sorry

end line_intersects_ellipse_l256_256410


namespace max_participants_A_l256_256326

variables (a b c x y : ℕ)

-- Problem conditions
def condition1 := a = b + c
def condition2 := b = 2 * c
def condition3 := a + 3 * b - 5 = a

-- Total participants condition
def total_participants := a + b + c + (3 * b + a - 5) + 3*b + y = 39

-- Proof goal
theorem max_participants_A : ∃ A_max, A_max = 23 ∧ 
  a + (3 * b + a - 5) = A_max :=
begin
  unfold condition1 condition2 condition3 total_participants,
  sorry
end

end max_participants_A_l256_256326


namespace paths_A_to_D_through_B_and_C_l256_256105

-- Define points and paths in a grid
structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 3⟩
def C : Point := ⟨6, 4⟩
def D : Point := ⟨9, 6⟩

-- Calculate binomial coefficient
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.div (Nat.factorial n) ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Number of paths from one point to another in a grid
def numPaths (p1 p2 : Point) : ℕ :=
  let stepsRight := p2.x - p1.x
  let stepsDown := p2.y - p1.y
  choose (stepsRight + stepsDown) stepsRight

theorem paths_A_to_D_through_B_and_C : numPaths A B * numPaths B C * numPaths C D = 500 := by
  -- Using the conditions provided:
  -- numPaths A B = choose 5 2 = 10
  -- numPaths B C = choose 5 1 = 5
  -- numPaths C D = choose 5 2 = 10
  -- Therefore, numPaths A B * numPaths B C * numPaths C D = 10 * 5 * 10 = 500
  sorry

end paths_A_to_D_through_B_and_C_l256_256105


namespace b_2030_is_5_l256_256591

def seq (b : ℕ → ℚ) : Prop :=
  b 1 = 4 ∧ b 2 = 5 ∧ ∀ n ≥ 3, b (n + 1) = b n / b (n - 1)

theorem b_2030_is_5 (b : ℕ → ℚ) (h : seq b) : 
  b 2030 = 5 :=
sorry

end b_2030_is_5_l256_256591


namespace expression_one_expression_two_l256_256392

/-- 
Prove that the mathematical expression 
( (27 / 8)^(-2 / 3) - (49 / 9)^(1 / 2) + (0.2)^(-2) * (3 / 25) ) 
is equal to 10 / 9.
-/
theorem expression_one : 
  ( (27 / 8 : ℚ) ^ (-2 / 3 : ℚ) - (49 / 9 : ℚ) ^ (1 / 2 : ℚ) + (0.2) ^ (-2 : ℚ) * (3 / 25 : ℚ) = 10 / 9) :=
by sorry

/-- 
Prove that the mathematical expression 
( -5 * log 9 4 + log 3 (32 / 9) - 5 ^ (log 5 3) ) 
is equal to -5 * log 3 2 - 5.
-/
theorem expression_two : 
  ( -5 * Real.logBase 9 4 + Real.logBase 3 (32 / 9) - 5 ^ (Real.logBase 5 3) = -5 * Real.logBase 3 2 - 5) :=
by sorry

end expression_one_expression_two_l256_256392


namespace coefficient_m5_n5_in_expansion_l256_256307

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Goal: prove the coefficient of m^5 n^5 in the expansion of (m+n)^{10} is 252
theorem coefficient_m5_n5_in_expansion : binomial 10 5 = 252 :=
by
  sorry

end coefficient_m5_n5_in_expansion_l256_256307


namespace length_to_width_ratio_is_three_l256_256454

def rectangle_ratio (x : ℝ) : Prop :=
  let side_length_large_square := 4 * x
  let length_rectangle := 4 * x
  let width_rectangle := x
  length_rectangle / width_rectangle = 3

-- We state the theorem to be proved
theorem length_to_width_ratio_is_three (x : ℝ) (h : 0 < x) :
  rectangle_ratio x :=
sorry

end length_to_width_ratio_is_three_l256_256454


namespace inequality_problem_l256_256317

theorem inequality_problem (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
by
  sorry

end inequality_problem_l256_256317


namespace distance_between_centers_l256_256174

noncomputable def distance_centers_inc_exc (PQ PR QR: ℝ) (hPQ: PQ = 17) (hPR: PR = 15) (hQR: QR = 8) : ℝ :=
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  let r := area / s
  let r' := area / (s - QR)
  let PU := s - PQ
  let PV := s
  let PI := Real.sqrt ((PU)^2 + (r)^2)
  let PE := Real.sqrt ((PV)^2 + (r')^2)
  PE - PI

theorem distance_between_centers (PQ PR QR : ℝ) (hPQ: PQ = 17) (hPR: PR = 15) (hQR: QR = 8) :
  distance_centers_inc_exc PQ PR QR hPQ hPR hQR = 5 * Real.sqrt 17 - 3 * Real.sqrt 2 :=
by sorry

end distance_between_centers_l256_256174


namespace equilateral_triangle_area_APQ_l256_256646

theorem equilateral_triangle_area_APQ (ABC : Triangle) 
  (h_eq : is_equilateral ABC)
  (h_side : ABC.sides = (10, 10, 10)) 
  (P Q : Point) 
  (hP : P ∈ segment ABC.A ABC.B) 
  (hQ : Q ∈ segment ABC.A ABC.C) 
  (h_tangent : is_tangent (segment P Q) ABC.incircle) 
  (hPQ : segment.length P Q = 4) : 
  area (triangle ABC.A P Q) = 5 / sqrt 3 :=
by 
  sorry

end equilateral_triangle_area_APQ_l256_256646


namespace count_f_s_mod_5_l256_256945

-- Define the function f
def f (x : ℕ) : ℕ := x^2 + 3 * x + 2

-- Define the set S
def S := { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }

-- The theorem to prove
theorem count_f_s_mod_5 : 
  (S.filter (λ s, f s % 5 = 0)).card = 6 :=
sorry

end count_f_s_mod_5_l256_256945


namespace g_is_even_function_l256_256562

def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even_function : ∀ x : ℝ, g (-x) = g x :=
by
  intro x
  rw [g, g]
  sorry

end g_is_even_function_l256_256562


namespace base_seven_to_ten_l256_256716

theorem base_seven_to_ten : 
  (7 * 7^4 + 6 * 7^3 + 5 * 7^2 + 4 * 7^1 + 3 * 7^0) = 19141 := 
by 
  sorry

end base_seven_to_ten_l256_256716


namespace cos_B_is_sqrt_6_div_3_l256_256525

-- Definitions and assumptions based on the conditions given.
def a : ℝ := 15
def b : ℝ := 10
def A : ℝ := real.pi / 3 -- This represents 60 degrees in radians.

-- The theorem we want to prove, restating the problem in Lean's formal language.
theorem cos_B_is_sqrt_6_div_3 : 
  let B := real.arcsin ((b * real.sin A) / a) in
  real.cos B = real.sqrt 6 / 3 :=
by sorry

end cos_B_is_sqrt_6_div_3_l256_256525


namespace statement_E_not_true_l256_256406

def star (x y : ℝ) : ℝ := x^2 - y^2

theorem statement_E_not_true : ¬ (∀ x : ℝ, star x (-x) = 2 * x^2) :=
by
  intro h
  have h1 : star 1 (-1) = 0 := by
    unfold star
    exact calc
      (1:ℝ)^2 - (-1:ℝ)^2 = 1 - 1 : by ring
      ... = 0 : by norm_num
  have h2 : 2 * 1^2 = 2 := by norm_num
  rw h at h1
  rw h2 at h1
  norm_num at h1
  contradiction

end statement_E_not_true_l256_256406


namespace sumOfShadedCells_l256_256327

-- Definition of the 3x3 grid and the conditions
def isValidGrid (grid : Matrix Int 3 3) : Prop :=
  (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 9) ∧
  (∑ i in Finset.finRange 3, grid i i = 7) ∧
  (∑ i in Finset.finRange 3, grid i (2 - i) = 21)

-- Target sum in the shaded cells (position wise combination must be determined as per solution)
def shadedCellSum (grid : Matrix Int 3 3) : Int :=
  grid 0 0 + grid 0 2 + grid 1 1 + grid 2 0 + grid 2 2

-- The final proposition to prove
theorem sumOfShadedCells (grid : Matrix Int 3 3) (h : isValidGrid grid) : shadedCellSum grid = 25 :=
 by sorry

end sumOfShadedCells_l256_256327


namespace constant_PQ_length_l256_256033

-- Definitions and conditions
variables {R : ℝ} (O : EuclideanSpace ℝ 2) (M : EuclideanSpace ℝ 2)
variables {A B C D P Q : EuclideanSpace ℝ 2}
variable [Nontrivial ℝVectorSpace ℝ (EuclideanSpace ℝ 2)]

-- Assume the following:
-- 1. O is the center of the circle.
-- 2. The distance between O and any point on the circumference is R.
-- 3. A and B are endpoints of one diameter, C and D are endpoints of another diameter.
-- 4. MP and MQ are perpendiculars from M to the diameters AB and CD.

-- Objective: Prove PQ length is constant.
theorem constant_PQ_length (circ : Circle O R) (MA : IsDiameter circ A B) (MC : IsDiameter circ C D)
  (MP_perp : IsPerpendicular (M, P) (A, B)) (MQ_perp : IsPerpendicular (M, Q) (C, D)) :
  ∃ const_len : ℝ, ∀ M, (dist P Q) = const_len := 
sorry

end constant_PQ_length_l256_256033


namespace total_length_of_lines_in_S_l256_256597

-- Define the set S
def S : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in 
    abs (abs (abs x - 3) - 1) + abs (abs (abs y - 3) - 1) = 2 }

-- Statement to prove the total length of all the lines that make up S is 128
theorem total_length_of_lines_in_S : 
  (∃ L : ℝ, L = 128) := 
sorry

end total_length_of_lines_in_S_l256_256597


namespace distance_A_to_focus_l256_256840

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  ((b^2 - 4*a*c) / (4*a), 0)

theorem distance_A_to_focus 
  (P : ℝ × ℝ) (parabola : ℝ → ℝ → Prop)
  (A B : ℝ × ℝ)
  (hP : P = (-2, 0))
  (hPar : ∀ x y, parabola x y ↔ y^2 = 4 * x)
  (hLine : ∃ m b, ∀ x y, y = m * x + b ∧ y^2 = 4 * x → (x, y) = A ∨ (x, y) = B)
  (hDist : dist P A = (1 / 2) * dist A B)
  (hFocus : focus_of_parabola 1 0 (-1) = (1, 0)) :
  dist A (1, 0) = 5 / 3 :=
sorry

end distance_A_to_focus_l256_256840


namespace solve_for_y_l256_256462

theorem solve_for_y (a b c x : ℝ) (p q r y : ℝ) 
  (h1 : (log 10 a) / p = (log 10 b) / q)
  (h2 : (log 10 b) / q = (log 10 c) / r)
  (h3 : (log 10 c) / r = log 10 x)
  (h4 : x ≠ 1)
  (h5 : b^3 / (a^2 * c) = x^y) : 
  y = 3 * q - 2 * p - r := 
sorry

end solve_for_y_l256_256462


namespace infinite_product_of_sequence_l256_256797

theorem infinite_product_of_sequence :
  let a : ℕ → ℝ := 
    λ n, nat.rec_on n (2/3) (λ n' a_n', 1 + (a_n' - 1)^2) in
  (∀ n, a n > 0) →
  let P : ℝ := (a 0) * (a 1) * (a 2) * (a 3) * ... ƒ  in
  P = 1/2 :=
by
  sorry

end infinite_product_of_sequence_l256_256797


namespace problem_statement_l256_256470

def z1 := complex.mk 1 (-1)
def z2 := complex.mk 2 (-1)
def z3 := complex.mk 2 2

theorem problem_statement :
  (im (z1 + z2) ≠ -2) ∧
  (im (z2 - z3) ≠ 0) ∧
  ((1 - 1) * 2 + (1 * -1) * 2 = 0) ∧
  (complex.norm_sq z1 + complex.norm_sq z2 < complex.norm_sq z3) :=
begin
  sorry,
end

end problem_statement_l256_256470


namespace smallest_number_of_students_l256_256534

theorem smallest_number_of_students : 
  ∃ n : ℕ, (∀ k : ℕ, k < n → k ≥ 8) ∧ 
           (∀ i < 4, student_scores i = 80) ∧ 
           (∀ j ∈ (univ \ {0, 1, 2, 3}), student_scores j ≥ 50) ∧ 
           (∑ k in finset.range n, student_scores k) / n = 65 :=
by
  sorry

end smallest_number_of_students_l256_256534


namespace unique_positive_solution_exists_l256_256849

theorem unique_positive_solution_exists 
    (a b : ℝ) 
    (ha : a > 0) 
    (hb : b > 0)
    (hab : a ≠ b) : 
  ∃! x : ℝ, x > 0 ∧ 
    (a + b) * x + (a * b) = (16 * x + 4 * (a + b)) * (a⁻¹/3 + b⁻¹/3)^-3 := 
by
  sorry

end unique_positive_solution_exists_l256_256849


namespace sequence_periodic_l256_256043

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n

theorem sequence_periodic (a : ℕ → ℝ) (m_0 : ℕ) (h : sequence_condition a) :
  ∀ m ≥ m_0, a (m + 9) = a m := 
sorry

end sequence_periodic_l256_256043


namespace sum_cos_sin_ratios_2017_l256_256862

-- Definitions
def f (x : ℕ) : ℚ := 1 / (x + 1 : ℚ)

def A (n : ℕ) : ℚ × ℚ := (n, f n)

def cos_theta (n : ℕ) : ℚ :=
  let OA_n : ℚ × ℚ := A n
  let i : ℚ × ℚ := (0, 1) 
  OA_n.2 / real.sqrt (OA_n.1^2 + OA_n.2^2)

def sin_theta (n : ℕ) : ℚ := 
  real.sqrt (1 - (cos_theta n)^2)

def ratio (n : ℕ) : ℚ := cos_theta n / sin_theta n

noncomputable def sum_cos_sin_ratios : ℚ :=
  Finset.sum (Finset.range 2017) (λ n, ratio (n + 1))

-- Theorem to prove the given statement
theorem sum_cos_sin_ratios_2017 :
  sum_cos_sin_ratios = 2017 / 2018 :=
sorry

end sum_cos_sin_ratios_2017_l256_256862


namespace lock_combination_l256_256235

-- Define the digits as distinct
def distinct_digits (V E N U S I A R : ℕ) : Prop :=
  V ≠ E ∧ V ≠ N ∧ V ≠ U ∧ V ≠ S ∧ V ≠ I ∧ V ≠ A ∧ V ≠ R ∧
  E ≠ N ∧ E ≠ U ∧ E ≠ S ∧ E ≠ I ∧ E ≠ A ∧ E ≠ R ∧
  N ≠ U ∧ N ≠ S ∧ N ≠ I ∧ N ≠ A ∧ N ≠ R ∧
  U ≠ S ∧ U ≠ I ∧ U ≠ A ∧ U ≠ R ∧
  S ≠ I ∧ S ≠ A ∧ S ≠ R ∧
  I ≠ A ∧ I ≠ R ∧
  A ≠ R

-- Define the base 12 addition for the equation
def base12_addition (V E N U S I A R : ℕ) : Prop :=
  let VENUS := V * 12^4 + E * 12^3 + N * 12^2 + U * 12^1 + S
  let IS := I * 12^1 + S
  let NEAR := N * 12^3 + E * 12^2 + A * 12^1 + R
  let SUN := S * 12^2 + U * 12^1 + N
  VENUS + IS + NEAR = SUN

-- The theorem statement
theorem lock_combination :
  ∃ (V E N U S I A R : ℕ),
    distinct_digits V E N U S I A R ∧
    base12_addition V E N U S I A R ∧
    (S * 12^2 + U * 12^1 + N) = 655 := 
sorry

end lock_combination_l256_256235


namespace number_of_intersection_points_l256_256274

def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem number_of_intersection_points : ∃ (n : ℕ), n = 2 ∧ ∀ x : ℝ, f x = g x →
{
  sorry
}

end number_of_intersection_points_l256_256274


namespace pentagon_to_trapezoid_l256_256759

def is_regular_pentagon (P : set Point) : Prop :=
  ∃ (A B C D E : Point), 
    P = {A, B, C, D, E} ∧ 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ A ∧
    dist A B = dist B C ∧ dist B C = dist C D ∧ 
    dist C D = dist D E ∧ dist D E = dist E A ∧ ∀ X ∈ P, ∃ θ, ∠BOC = 108

def can_form_isosceles_trapezoid (S1 S2 S3 : set Point) : Prop :=
  ∃ (A B C D : Point), 
    S1 ⊆ {A, B, C, D} ∧
    S2 ⊆ {A, B, C, D} ∧ S3 ⊆ {A, B, C, D} ∧ is_isosceles_trapezoid {A, B, C, D}

theorem pentagon_to_trapezoid 
  (P : set Point) (hP : is_regular_pentagon P) : 
  ∃ (S1 S2 S3 : set Point), 
    P = S1 ∪ S2 ∪ S3 ∧ S1 ∩ S2 = ∅ ∧ S2 ∩ S3 = ∅ ∧ S1 ∩ S3 = ∅ ∧ can_form_isosceles_trapezoid S1 S2 S3 :=
by
  sorry

end pentagon_to_trapezoid_l256_256759


namespace sum_B_equals_35_over_8_l256_256401

def in_set_B (n : ℕ) : Prop :=
  ∀ p, p.prime ∧ p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7

def sum_B : ℝ :=
  ∑' (n : ℕ) in {n : ℕ | in_set_B n}, n⁻¹

theorem sum_B_equals_35_over_8 : sum_B = (35 / 8) := by
  sorry

end sum_B_equals_35_over_8_l256_256401


namespace number_of_n_not_dividing_g_in_range_l256_256186

def g (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ x, x ≠ n ∧ x ∣ n) (finset.range (n+1))), d

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem number_of_n_not_dividing_g_in_range :
  (finset.filter (λ n, n ∉ (finset.range (101)).filter (λ n, n ∣ g n))
  (finset.Icc 2 100)).card = 29 :=
by
  sorry

end number_of_n_not_dividing_g_in_range_l256_256186


namespace price_first_variety_is_126_l256_256248

variable (x : ℝ) -- price of the first variety per kg (unknown we need to solve for)
variable (p2 : ℝ := 135) -- price of the second variety per kg
variable (p3 : ℝ := 175.5) -- price of the third variety per kg
variable (mix_ratio : ℝ := 4) -- total weight ratio of the mixture
variable (mix_price : ℝ := 153) -- price of the mixture per kg
variable (w1 w2 w3 : ℝ := 1) -- weights of the first two varieties
variable (w4 : ℝ := 2) -- weight of the third variety

theorem price_first_variety_is_126:
  (w1 * x + w2 * p2 + w4 * p3) / mix_ratio = mix_price → x = 126 := by
  sorry

end price_first_variety_is_126_l256_256248


namespace eccentricity_correct_max_area_triangle_l256_256837

section ellipse_geometry

variables {a b c : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b)

-- Condition: Ellipse equation
def ellipse (x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Question 1: Eccentricity of the ellipse
def eccentricity_ellipse : ℝ := 
  sqrt (1 - (b^2 / a^2))

-- Proof of the eccentricity
theorem eccentricity_correct (h : a = sqrt 2 * b) : 
  eccentricity_ellipse a b = sqrt 2 / 2 :=
sorry

-- Variables and perimeter condition
variables {P Q : ℝ × ℝ} {M N : ℝ × ℝ} 
(variable LinePQ : ℝ → ℝ)

def perimeter (perim_val : ℝ) : Prop := 
  perim_val = 8

-- Question 2: Maximum area of triangle MPQ
def triangle_area (M P Q : ℝ × ℝ) : ℝ := 
  let (m1, m2) := M in
  let (p1, p2) := P in
  let (q1, q2) := Q in
  abs ((m1*(p2-q2) + p1*(q2-m2) + q1*(m2-p2)) / 2)

-- Proof of the maximum area
theorem max_area_triangle (hM : M = (c, 0)) (ha : a = 2) (hP : ellipse a b P.1 P.2) (hQ : ellipse a b Q.1 Q.2) : 
  ∃ P Q, perimeter (|M.1 - P.1| + |P.1 - Q.1| + |Q.1 - M.1|) ∧ 
  (∀ P Q, triangle_area M P Q ≤ 2 * sqrt 2) :=
sorry

end ellipse_geometry

end eccentricity_correct_max_area_triangle_l256_256837


namespace patty_weight_loss_l256_256987

/-- Definitions from the problem conditions -/
def robbie_weight : ℕ := 100
def patty_original_weight : ℕ := 4.5 * robbie_weight
def patty_current_weight : ℕ := robbie_weight + 115

/-- The problem rewritten as a proof problem -/
theorem patty_weight_loss :
  patty_original_weight - patty_current_weight = 235 := 
by
  sorry

end patty_weight_loss_l256_256987


namespace triangle_area_ratio_l256_256550

theorem triangle_area_ratio (A B C D E : Point)
  (h_angle_A : ∠A = 45)
  (h_angle_B : ∠B = 30)
  (h_angle_ADE : ∠ADE = 60)
  (h_equal_area : area (triangle ADE) = area (triangle ABC) / 2)
  : AD / AB = 1 / real.sqrt (real.sqrt 12) :=
sorry

end triangle_area_ratio_l256_256550


namespace equation_of_circle_l256_256881

theorem equation_of_circle (a : ℝ) :
    (a, -2 * a) = (1, -2) ∧
    (λ x y, (x - 2) ^ 2 + (y + 1) ^ 2 = 2) ∧
    (λ x y, |a + -2 - y| / sqrt (2 * -2) = abs (x + y - 1)) → 
    ∃ r : ℝ, (x - 1) ^ 2 + (y + 2) ^ 2 = 2 :=
by
  sorry

end equation_of_circle_l256_256881


namespace total_votes_l256_256545

variable (T S R F V : ℝ)

-- Conditions
axiom h1 : T = S + 0.15 * V
axiom h2 : S = R + 0.05 * V
axiom h3 : R = F + 0.07 * V
axiom h4 : T + S + R + F = V
axiom h5 : T - 2500 - 2000 = S + 2500
axiom h6 : S + 2500 = R + 2000 + 0.05 * V

theorem total_votes : V = 30000 :=
sorry

end total_votes_l256_256545


namespace chord_length_mnp_l256_256395

noncomputable def radius_c1 : ℝ := 6
noncomputable def radius_c2 : ℝ := 12
noncomputable def radius_c4 : ℝ := 20

def is_tangent (r1 r2 d : ℝ) : Prop :=
  d = r1 + r2

def centers_collinear (O1 O2 O3 : ℝ) : Prop :=
  O1 ≤ O2 ∧ O2 ≤ O3

theorem chord_length_mnp :
  ∃ (m n p : ℕ),
  is_tangent radius_c1 radius_c2 (radius_c1 + radius_c2) ∧
  is_tangent radius_c1 radius_c4 (radius_c1 + radius_c4) ∧
  is_tangent radius_c2 radius_c4 (radius_c2 + radius_c4) ∧
  centers_collinear 0 radius_c1 (radius_c1 + radius_c2) ∧
  gcd m p = 1 ∧
  n = 7 ∧
  ¬ ∃ q : ℕ, (q ^ 2) ∣ n ∧ q > 1 ∧
  (10 : ℝ) * real.sqrt 7 = 10 * real.sqrt 7 ∧
  m + n + p = 18 :=
sorry

end chord_length_mnp_l256_256395


namespace probability_exact_n_points_l256_256291

open Classical

noncomputable def probability_of_n_points (n : ℕ) : ℚ :=
  1/3 * (2 + (-1/2)^n)

theorem probability_exact_n_points (n : ℕ) :
  ∀ n : ℕ, probability_of_n_points n = 1/3 * (2 + (-1/2)^n) :=
sorry

end probability_exact_n_points_l256_256291


namespace g_is_even_l256_256564

noncomputable def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end g_is_even_l256_256564


namespace expressions_with_variables_count_l256_256378

def has_variable_in_denominator (expr : Expr) : Prop :=
  match expr with
  | Expr.frac x y => (isVariable y)
  | _ => false

def Expr :=
  | frac : Term -> Term -> Expr
  | term : Term -> Expr

def Term :=
  | var : String -> Term
  | const : Nat -> Term
  | pi : Term

def expressions : List Expr :=
  [ Expr.frac (Term.var "a" - Term.var "b") (Term.const 2),
    Expr.frac (Term.var "x" + Term.const 3) (Term.var "x"),
    Expr.frac (Term.const 5 + Term.var "y") Term.pi,
    Expr.frac (Term.var "x" * Term.var "x") (Term.const 4),
    Expr.frac (Term.var "a" + Term.var "b") (Term.var "a" - Term.var "b")
    Expr.frac ((Term.const 1) * (Term.var "x" - Term.var "y")) (Term.var "m"),
    Expr.frac (Term.var "x" * Term.var "x") (Term.var "x")]

theorem expressions_with_variables_count : (count (has_variable_in_denominator) expressions) = 4 :=
by sorry

end expressions_with_variables_count_l256_256378


namespace cos_alpha_plus_pi_over_6_cos_alpha_plus_beta_l256_256955

theorem cos_alpha_plus_pi_over_6 (α β : ℝ) (hα : α ∈ Ioo 0 (π / 3)) (hβ : β ∈ Ioo (π / 6) (π / 2))
  (h1 : 5 * sqrt 3 * sin α + 5 * cos α = 8) 
  (h2 : sqrt 2 * sin β + sqrt 6 * cos β = 2) :
  cos (α + π / 6) = 3 / 5 := 
sorry

theorem cos_alpha_plus_beta (α β : ℝ) (hα : α ∈ Ioo 0 (π / 3)) (hβ : β ∈ Ioo (π / 6) (π / 2))
  (h1 : 5 * sqrt 3 * sin α + 5 * cos α = 8) 
  (h2 : sqrt 2 * sin β + sqrt 6 * cos β = 2) :
  cos (α + β) = -sqrt 2 / 10 := 
sorry

end cos_alpha_plus_pi_over_6_cos_alpha_plus_beta_l256_256955


namespace harry_sandy_meeting_point_l256_256104

theorem harry_sandy_meeting_point : 
  let h := (12, 3)
  let s := (4, 9)
  let midpoint := ((h.1 + s.1) / 2, (h.2 + s.2) / 2)
  in midpoint = (8, 6) ∧ ¬(midpoint.2 = -midpoint.1 + 6) := 
by
  let h := (12, 3)
  let s := (4, 9)
  let midpoint := ((h.1 + s.1) / 2, (h.2 + s.2) / 2)
  show midpoint = (8, 6) ∧ ¬(midpoint.2 = -midpoint.1 + 6)
  from sorry

end harry_sandy_meeting_point_l256_256104


namespace students_after_join_l256_256734

theorem students_after_join (N : ℕ)
  (hN : N > 0)
  (average_age : Nat := 48)
  (new_students : Nat := 120)
  (avg_age_new_students : Nat := 32)
  (avg_age_decrease : Nat := 4)
  (new_avg_age : Nat := 44)
  (h1 : average_age * N + new_students * avg_age_new_students = new_avg_age * (N + new_students)) :
  N = 360
  → N + new_students = 480 :=
by
  intro hN_eq
  rw [hN_eq]
  rfl

end students_after_join_l256_256734


namespace find_lambda_l256_256112

open real

-- Definitions of vectors a and b
def a : ℝ × ℝ × ℝ := (0, 1, -1)
def b : ℝ × ℝ × ℝ := (1, 1, 0)

-- Dot product definition for 3D vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- The problem statement
theorem find_lambda (λ : ℝ) :
  dot_product (a.1 + λ * b.1, a.2 + λ * b.2, a.3 + λ * b.3) a = 0 -> λ = -2 :=
by
  sorry

end find_lambda_l256_256112


namespace cars_no_air_conditioning_l256_256141

variables {A R AR : Nat}

/-- Given a total of 100 cars, of which at least 51 have racing stripes,
and the greatest number of cars that could have air conditioning but not racing stripes is 49,
prove that the number of cars that do not have air conditioning is 49. -/
theorem cars_no_air_conditioning :
  ∀ (A R AR : ℕ), 
  (A = AR + 49) → 
  (R ≥ 51) → 
  (AR ≤ R) → 
  (AR ≤ 51) → 
  (100 - A = 49) :=
by
  intros A R AR h1 h2 h3 h4
  exact sorry

end cars_no_air_conditioning_l256_256141


namespace BE_eq_FD_l256_256546

variable {A B C D E F : Point}
variable {O : Circle}
variable [hTriangle : Triangle ABC]
variable [hIsosceles : IsoscelesTriangle ABC]
variable [hCircumscribed : CircumscribedCircle ABC O]
variable [hBisector : Bisector CD]
variable [hPerpendicular : PerpendicularToBisectorThroughCenter CD O E]
variable [hParallel : ParallelLineThrough E CD AB F]

-- The conjecture to prove BE = FD
theorem BE_eq_FD :
    IsoscelesTriangle ABC ∧
    Bisector CD ∧
    PerpendicularToBisectorThroughCenter CD O E ∧
    ParallelLineThrough E CD AB F → 
    SegmentLength BE = SegmentLength FD := 
sorry

end BE_eq_FD_l256_256546


namespace pencils_in_drawer_l256_256285

theorem pencils_in_drawer :
  let initial_pencils := 115
  let sara_pencils := 100
  let john_pencils := 75
  let ben_removed_pencils := 45
  initial_pencils + sara_pencils + john_pencils - ben_removed_pencils = 245 :=
by
  -- Setting up the conditions
  let initial_pencils := 115
  let sara_pencils := 100
  let john_pencils := 75
  let ben_removed_pencils := 45

  -- The final total calculation
  show initial_pencils + sara_pencils + john_pencils - ben_removed_pencils = 245,
    from sorry

end pencils_in_drawer_l256_256285


namespace arrange_integrals_l256_256166

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (1 + x)

noncomputable def I (n : ℕ) : ℝ :=
  ∫ x in 0..((Real.pi : ℝ) * n), f x

theorem arrange_integrals :
  I 2 < I 4 ∧ I 4 < I 3 ∧ I 3 < I 1 :=
by
  sorry

end arrange_integrals_l256_256166


namespace length_of_solution_set_l256_256266

variable {a b : ℝ}

theorem length_of_solution_set (h : ∀ x : ℝ, a ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ b → 12 = (b - a) / 3) : b - a = 36 :=
sorry

end length_of_solution_set_l256_256266


namespace problem_inequality_l256_256602

theorem problem_inequality (a b c : ℝ) (h₀ : a + b + c = 0) (d : ℝ) (h₁ : d = max (|a|) (max (|b|) (|c|))) : 
  |(1 + a) * (1 + b) * (1 + c)| ≥ 1 - d^2 :=
sorry

end problem_inequality_l256_256602


namespace equivalent_expression_l256_256799

variable {x y : ℝ}

def P := 2 * x + 3 * y
def Q := x - 2 * y

theorem equivalent_expression :
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (2 * x + 3 * y) / (2 * x + 10 * y) :=
by
  sorry

end equivalent_expression_l256_256799


namespace ratio_of_trout_l256_256234

-- Definition of the conditions
def trout_caught_by_Sara : Nat := 5
def trout_caught_by_Melanie : Nat := 10

-- Theorem stating the main claim to be proved
theorem ratio_of_trout : trout_caught_by_Melanie / trout_caught_by_Sara = 2 := by
  sorry

end ratio_of_trout_l256_256234


namespace min_score_10_l256_256366

-- Define the scores for the 6th, 7th, 8th, and 9th shots
def score_6 := 9.0
def score_7 := 8.4
def score_8 := 8.1
def score_9 := 9.3

-- Define conditions based on the problem statement
variable (scores_first_5 : List ℝ) (scores_rest : List ℝ)
variable (average_5 : ℝ) (average_9 : ℝ) (score10 : ℝ)

-- Assuming the average of the first 5 scores is less than the average of the first 9 scores which is less than 8.7
axiom avg_5_lt_avg_9 : average_5 < average_9
axiom avg_9_le_8_7 : average_9 < 8.7

-- Define the condition for the scores scale
def scale_accuracy := 0.1

-- The Lean statement to prove the minimum score required in the 10th shot to ensure average exceeds 8.8
theorem min_score_10 (scores : List ℝ) (total_first_9 : ℝ) :
  (scores = scores_first_5 ++ [score_6, score_7, score_8, score_9]) →
  total_first_9 = (∑ score in scores, score) →
  (average_5 = (total_first_9 - (score_6 + score_7 + score_8 + score_9)) / 5) →
  (average_9 = total_first_9 / 9) →
  avg_5_lt_avg_9 →
  avg_9_le_8_7 →
  score10 >= 9.9 →
  (total_first_9 + score10) / 10 > 8.8 :=
sorry

end min_score_10_l256_256366


namespace solve_equation_l256_256997

theorem solve_equation :
  ∀ (x : ℝ), 
    x^3 + (Real.log 25 + Real.log 32 + Real.log 53) * x = (Real.log 23 + Real.log 35 + Real.log 52) * x^2 + 1 ↔ 
    x = Real.log 23 ∨ x = Real.log 35 ∨ x = Real.log 52 :=
by
  sorry

end solve_equation_l256_256997


namespace edge_incircle_exists_l256_256227

variables {T : Type} [Tetrahedron T]
variables (a a' b b' c c' : Real)

open Tetrahedron

def opposite_edge_sum (T : Tetrahedron) (a a' b b' c c' : Real) : Prop :=
  a + a' = b + b' ∧ b + b' = c + c'

theorem edge_incircle_exists (T : Tetrahedron) (a a' b b' c c' : Real) :
  opposite_edge_sum T a a' b b' c c' → exists (S : Sphere T), (∀ e ∈ Edges T, touches S e) :=
by
  sorry

end edge_incircle_exists_l256_256227


namespace max_ratio_FA_OH_l256_256490

-- Define the ellipse equation and the conditions for a and b
variables {a b : ℝ}
def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Assume a > b > 0
variables (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)

-- Define the distance function for |FA| and |OH|
noncomputable def FA := a - a * (b^2 / a^2).sqrt
noncomputable def OH := a^2 / (a * (b^2 / a^2).sqrt)

-- Define the ratio |FA| / |OH|
noncomputable def ratio := FA a b / OH a b

-- State the theorem to be proved
theorem max_ratio_FA_OH : ∃ e : ℝ, 0 < e ∧ e < 1 ∧ ratio a b = 1/4 :=
sorry

end max_ratio_FA_OH_l256_256490


namespace route_one_speed_is_50_l256_256666

noncomputable def speed_route_one (x : ℝ) : Prop :=
  let time_route_one := 75 / x
  let time_route_two := 90 / (1.8 * x)
  time_route_one = time_route_two + 1/2

theorem route_one_speed_is_50 :
  ∃ x : ℝ, speed_route_one x ∧ x = 50 :=
by
  sorry

end route_one_speed_is_50_l256_256666


namespace speed_of_second_projectile_l256_256713

-- conditions
def initial_apart_distance : ℝ := 1386
def speed_projectile_1 : ℝ := 445
def meeting_time_hours : ℝ := 84 / 60

-- statement to prove
theorem speed_of_second_projectile (v: ℝ) (h1: v * meeting_time_hours = initial_apart_distance - (speed_projectile_1 * meeting_time_hours)) : 
(v = 545) :=
sorry

end speed_of_second_projectile_l256_256713


namespace expansion_correct_l256_256809

noncomputable def f(z : ℤ) : ℤ := 2 * z ^ 2 + 5 * z - 6
noncomputable def g(z : ℤ) : ℤ := 3 * z ^ 3 - 2 * z + 1
noncomputable def expanded(z : ℤ) : ℤ := 6 * z ^ 5 + 15 * z ^ 4 - 22 * z ^ 3 - 8 * z ^ 2 + 17 * z - 6

theorem expansion_correct (z : ℤ) : (f(z) * g(z)) = expanded(z) := by
  sorry

end expansion_correct_l256_256809


namespace cos_beta_value_sin_alpha_value_l256_256065

theorem cos_beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : π / 2 < β ∧ β < π) (h3 : cos (2 * β) = -7 / 9) 
  (h4 : sin (α + β) = 7 / 9) : 
  cos β = -1 / 3 := 
sorry

theorem sin_alpha_value (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : π / 2 < β ∧ β < π) (h3 : cos (2 * β) = -7 / 9) 
  (h4 : sin (α + β) = 7 / 9) : 
  sin α = 1 / 3 := 
sorry

end cos_beta_value_sin_alpha_value_l256_256065


namespace inscribed_exscribed_radii_l256_256021

theorem inscribed_exscribed_radii (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13)
  (h4 : a^2 + b^2 = c^2) : 
  let P := a + b + c,
      s := P / 2,
      A := (a * b) / 2 in
  let r := (a + b - c) / 2,
      r_a := A / (s - a),
      r_b := A / (s - b),
      r_c := A / (s - c) in
  r = 2 ∧ r_a = 3 ∧ r_b = 10 ∧ r_c = 15 :=
by 
  sorry

end inscribed_exscribed_radii_l256_256021


namespace intervals_of_monotonicity_range_of_a_l256_256081

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + a^2 * x - 1

-- Interval of monotonicity for a = 2
theorem intervals_of_monotonicity (x : ℝ) :
  let a := 2 in
  let f' := λ x : ℝ, 3 * x^2 + 4 * x - 4 in
  f' x = (3 * x - 2) * (x + 2) →
  -- Critical points
  (x = 2 / 3 ∨ x = -2) →
  -- Increasing and decreasing intervals
  (∀ x, x > 2 / 3 → f' x > 0) ∧ (∀ x, x < -2 → f' x > 0) ∧ (∀ x, -2 < x ∧ x < 2 / 3 → f' x < 0)
:= by
  -- Proof is skipped with 'sorry'
  sorry

-- Range of values for a if f(x) ≤ 0 has solutions on [1, ∞)
theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x → f x a ≤ 0) →
  (1 ≤ a ∨ a > 3)
:= by
  -- Proof is skipped with 'sorry'
  sorry

end intervals_of_monotonicity_range_of_a_l256_256081


namespace identify_conic_section_l256_256413

theorem identify_conic_section (x y : ℝ) :
  (x + 7)^2 = (5 * y - 6)^2 + 125 →
  ∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y + e * x * y + f = 0 ∧
  (a > 0) ∧ (b < 0) := sorry

end identify_conic_section_l256_256413


namespace mr_desmond_toys_l256_256615

theorem mr_desmond_toys (toys_for_elder : ℕ) (h1 : toys_for_elder = 60)
  (h2 : ∀ (toys_for_younger : ℕ), toys_for_younger = 3 * toys_for_elder) : 
  ∃ (total_toys : ℕ), total_toys = 240 :=
by {
  sorry
}

end mr_desmond_toys_l256_256615


namespace sean_total_cost_l256_256991

noncomputable def total_cost (soda_cost soup_cost sandwich_cost : ℕ) (num_soda num_soup num_sandwich : ℕ) : ℕ :=
  num_soda * soda_cost + num_soup * soup_cost + num_sandwich * sandwich_cost

theorem sean_total_cost :
  let soda_cost := 1
  let soup_cost := 3 * soda_cost
  let sandwich_cost := 3 * soup_cost
  let num_soda := 3
  let num_soup := 2
  let num_sandwich := 1
  total_cost soda_cost soup_cost sandwich_cost num_soda num_soup num_sandwich = 18 :=
by
  sorry

end sean_total_cost_l256_256991


namespace hyperbola_foci_distance_l256_256440

theorem hyperbola_foci_distance :
  ∀ (a b : ℝ), (a^2 = 25) ∧ (b^2 = 9) →
  2 * real.sqrt (a^2 + b^2) = 2 * real.sqrt 34 := by
  intros a b h,
  have ha : a^2 = 25 := h.1,
  have hb : b^2 = 9 := h.2,
  sorry

end hyperbola_foci_distance_l256_256440


namespace proj_vector_l256_256100

open Real

def projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_prod := (a.1 * b.1 + a.2 * b.2 + a.3 * b.3)
  let b_mag_sq := (b.1^2 + b.2^2 + b.3^2)
  let scalar := dot_prod / b_mag_sq
  (scalar * b.1, scalar * b.2, scalar * b.3)

theorem proj_vector (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (1, 3, 0))
  (hb : b = (2, 1, 1))
  (hc : c = (5/3, 5/6, 5/6)) : 
  projection a b = c :=
by
  sorry

end proj_vector_l256_256100


namespace simplify_fraction_l256_256239

/-- Simplify the fraction 120/180 to its simplest form, which is 2/3. -/
theorem simplify_fraction (num den : ℕ) (h1 : num = 120) (h2 : den = 180) : num / den = 2 / 3 :=
by
  rw [h1, h2]
  change 120 / 180 = 2 / 3
  simp
  sorry

end simplify_fraction_l256_256239


namespace cyclic_quad_angles_l256_256132

theorem cyclic_quad_angles (A B C D : ℝ) (x : ℝ)
  (h_ratio : A = 5 * x ∧ B = 6 * x ∧ C = 4 * x)
  (h_cyclic : A + D = 180 ∧ B + C = 180):
  (B = 108) ∧ (C = 72) :=
by
  sorry

end cyclic_quad_angles_l256_256132


namespace incorrect_statement_A_l256_256767

-- conditions as stated in the table
def spring_length (x : ℕ) : ℝ :=
  if x = 0 then 20
  else if x = 1 then 20.5
  else if x = 2 then 21
  else if x = 3 then 21.5
  else if x = 4 then 22
  else if x = 5 then 22.5
  else 0 -- assuming 0 for out of range for simplicity

-- questions with answers
-- Prove that statement A is incorrect
theorem incorrect_statement_A : ¬ (spring_length 0 = 20) := by
  sorry

end incorrect_statement_A_l256_256767


namespace coefficient_x_l256_256017

def expr := 4 * (x - 5) + 3 * (2 - 3 * x^2 + 6 * x) - 10 * (3 * x - 2)

theorem coefficient_x : (∀ x : ℝ, expr = 4 * x + 18 * x - 30 * x + other_terms) → coefficient x expr = -8 := by 
  sorry

end coefficient_x_l256_256017


namespace james_pays_per_episode_l256_256571

-- Conditions
def minor_characters : ℕ := 4
def major_characters : ℕ := 5
def pay_per_minor_character : ℕ := 15000
def multiplier_major_payment : ℕ := 3

-- Theorems and Definitions needed
def pay_per_major_character : ℕ := pay_per_minor_character * multiplier_major_payment
def total_pay_minor : ℕ := minor_characters * pay_per_minor_character
def total_pay_major : ℕ := major_characters * pay_per_major_character
def total_pay_per_episode : ℕ := total_pay_minor + total_pay_major

-- Main statement to prove
theorem james_pays_per_episode : total_pay_per_episode = 285000 := by
  sorry

end james_pays_per_episode_l256_256571


namespace solution_set_of_inequality_l256_256695

theorem solution_set_of_inequality :
  { x : ℝ | |1 - 2 * x| < 3 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_of_inequality_l256_256695


namespace max_reciprocal_sum_of_eccentricities_l256_256061

-- Definitions of the conditions
variables {F1 F2 : Point} -- common foci of the ellipse and hyperbola
variables {P : Point} -- common point of the ellipse and hyperbola
variable {angle_F1PF2 : ℝ} -- angle at point P between F1 and F2

-- Conditions
def is_common_foci_of_ellipse_and_hyperbola : Prop :=
  -- Placeholder for the definition that F1 and F2 are common foci of an ellipse and hyperbola
  sorry

def is_common_point (F1 F2 P : Point) : Prop :=
  -- Placeholder for the definition that P is a common point
  sorry

def angle_is_pi_over_3 (angle : ℝ) : Prop :=
  angle = π / 3

-- Question to answer: Maximum value of the sum of the reciprocals of the eccentricities
def max_sum_of_reciprocals_eccentricities (e1 e2 : ℝ) : ℝ :=
  1 / e1 + 1 / e2

-- Properties of the ellipse and hyperbola
variables {a a1 c r1 r2 : ℝ} -- semi-major axis of ellipse, real semi-axis of hyperbola, semi-focal distance, distances to P

def ellipse_properties (a c r1 r2 : ℝ) : Prop :=
  -- Placeholder for properties of the ellipse involving a, c, r1, r2
  sorry

def hyperbola_properties (a1 c r1 r2 : ℝ) : Prop :=
  -- Placeholder for properties of the hyperbola involving a1, c, r1, r2
  sorry

theorem max_reciprocal_sum_of_eccentricities
  (h_common_foci : is_common_foci_of_ellipse_and_hyperbola)
  (h_common_point : is_common_point F1 F2 P)
  (h_angle : angle_is_pi_over_3 angle_F1PF2)
  (h_ellipse_props : ellipse_properties a c r1 r2)
  (h_hyperbola_props : hyperbola_properties a1 c r1 r2) :
  max_sum_of_reciprocals_eccentricities e1 e2 = (4 * Real.sqrt 3) / 3 := 
sorry

end max_reciprocal_sum_of_eccentricities_l256_256061


namespace sin_of_cos_of_angle_l256_256558

-- We need to assume that A is an angle of a triangle, hence A is in the range (0, π).
theorem sin_of_cos_of_angle (A : ℝ) (hA : 0 < A ∧ A < π) (h_cos : Real.cos A = -3/5) : Real.sin A = 4/5 := by
  sorry

end sin_of_cos_of_angle_l256_256558


namespace find_ellipse_equation_l256_256046

noncomputable def ellipse_equation (a b : ℝ) (ecc : ℝ) (dot_product : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧
  ecc = 1 / 3 ∧
  dot_product = -1 →
  (9 : ℝ) = a ^ 2 ∧ (8 : ℝ) = b ^ 2 

theorem find_ellipse_equation :
  ∃ (a b : ℝ), ellipse_equation a b (1/3) (-1) :=
begin
  sorry,
end

end find_ellipse_equation_l256_256046


namespace range_of_x_l256_256059

variable (x : ℝ)

def p : Prop := real.log (x^2 - 2 * x - 2) ≥ 0
def q : Prop := 0 < x ∧ x < 4
def neg_p : Prop := ¬p
def neg_q : Prop := ¬q

theorem range_of_x (h_neg_p : neg_p x) (h_neg_q : neg_q x) (h_p_or_q : p x ∨ q x) :
  x ≤ -1 ∨ (0 < x ∧ x < 3) ∨ x ≥ 4 :=
by
  sorry

end range_of_x_l256_256059


namespace jasmine_paperclips_l256_256161

theorem jasmine_paperclips :
  ∃ k : ℕ, (4 * 3^k > 500) ∧ (∀ n < k, 4 * 3^n ≤ 500) ∧ k = 5 ∧ (n = 6) :=
by {
  sorry
}

end jasmine_paperclips_l256_256161


namespace find_n_l256_256818

noncomputable def S_n (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  ∑ k in Finset.range n + 1, Real.sqrt ((2 * k + 1)^2 + (a k)^2)

theorem find_n (n : ℕ) (a : ℕ → ℝ) (H1 : n ≠ 0)
  (H2 : ∀ k, (a k > 0)) (H3 : ∑ k in Finset.range (n + 1), a k = 17)
  (H4 : ∃ m : ℤ, S_n n a = m) : n = 12 :=
by
  sorry

end find_n_l256_256818


namespace calculate_expression_l256_256788

theorem calculate_expression :
  2^3 - (Real.tan (Real.pi / 3))^2 = 5 := by
  sorry

end calculate_expression_l256_256788


namespace distance_AB_area_triangle_AOB_l256_256153

open Real
open Float

section Geometry

variable (A B O : Type)
variable (OA OB : ℝ) (angAOB : ℝ)

-- Given conditions
def polar_coordinates_A : OA = 2 := sorry
def polar_coordinates_B : OB = 3 := sorry
def angle_AOB : angAOB = π / 3 := sorry

-- Question (1):
theorem distance_AB (h1 : OA = 2) (h2 : OB = 3) (h3 : angAOB = π / 3) : 
  dist AB = sqrt (7) := sorry

-- Question (2):
theorem area_triangle_AOB (h1 : OA = 2) (h2 : OB = 3) (h3 : angAOB = π / 3) :
  area AOB = 3 * sqrt 3 / 2 := sorry

end Geometry

end distance_AB_area_triangle_AOB_l256_256153


namespace probability_left_red_off_second_blue_on_right_blue_on_l256_256233

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def total_lamps : ℕ := num_red_lamps + num_blue_lamps
def num_on : ℕ := 4
def position := Fin total_lamps
def lamp_state := {state // state < (total_lamps.choose num_red_lamps) * (total_lamps.choose num_on)}

def valid_configuration (leftmost : position) (second_left : position) (rightmost : position) (s : lamp_state) : Prop :=
(leftmost.1 = 1 ∧ second_left.1 = 2 ∧ rightmost.1 = 8) ∧ (s.1 =  (((total_lamps - 3).choose 3) * ((total_lamps - 3).choose 2)))

theorem probability_left_red_off_second_blue_on_right_blue_on :
  ∀ (leftmost second_left rightmost : position) (s : lamp_state),
  valid_configuration leftmost second_left rightmost s ->
  ((total_lamps.choose num_red_lamps) * (total_lamps.choose num_on)) = 49 :=
sorry

end probability_left_red_off_second_blue_on_right_blue_on_l256_256233


namespace probability_distinct_real_roots_l256_256595

theorem probability_distinct_real_roots :
  let outcomes := [(4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (1, 4), (2, 4), (3, 4), (5, 4), (6, 4), (4, 4)] in
  let distinct_root_cases := [(4, 1), (4, 2), (4, 3), (5, 4), (6, 4)] in
  (list.length distinct_root_cases : ℚ) / (list.length outcomes : ℚ) = 5 / 11 :=
by
  have total_outcomes := 5 + 5 + 1
  have favorable_cases := 3 + 2
  show (5 : ℚ) / (11 : ℚ) = 5 / 11, sorry

end probability_distinct_real_roots_l256_256595


namespace ellipse_equation_valid_l256_256053

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (eccentricity : a^2 * 8 / b^2 = 9) : Prop :=
  ∀ (A1 A2 B : ℝ × ℝ), 
    (A1 = (-3, 0)) ∧ 
    (A2 = (3, 0)) ∧ 
    (B = (0, 2 * Real.sqrt 2)) ∧ 
    ((B.1 - A1.1) * (B.1 - A2.1) + (B.2 - A1.2) * (B.2 - A2.2) = -1) →
    ( ∃ m : ℝ, m ≠ 0 ∧ (m^2 = 1) ∧ 
    (a^2 = 9 * m^2) ∧ (b^2 = 8 * m^2) ∧ 
    (C : ℝ × ℝ → Prop, ∀ x y, C (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1) → 
    ( ∀ x y, C (x, y) ↔ x^2 / 9 + y^2 / 8 = 1) )

theorem ellipse_equation_valid :
ellipse_equation 3 (2 * Real.sqrt 2) (by linarith) (by linarith) 
(3^2 * 8 / (2 * Real.sqrt 2)^2 = 9) := 
sorry

end ellipse_equation_valid_l256_256053


namespace jenny_eggs_in_each_basket_l256_256573

theorem jenny_eggs_in_each_basket (n : ℕ) (h1 : 30 % n = 0) (h2 : 45 % n = 0) (h3 : n ≥ 5) : n = 15 :=
sorry

end jenny_eggs_in_each_basket_l256_256573


namespace particle_position_after_2023_minutes_l256_256756

def movement_pattern : ℕ → ℕ × ℕ
| 0 => (0, 0)
| n+1 =>
  let (x, y) := movement_pattern n
  if n % 2 == 0 then (x + 1, y) else (x, y + 1)

def enclosing_time : ℕ → ℕ
| 0 => 0
| n+1 => enclosing_time n + (2 * (n + 1) + 1) + 2

def total_time (n : ℕ) : ℕ :=
  (n + 1) * (n + 1) + 5 * (n + 1)

def final_position (t : ℕ) : ℕ × ℕ :=
  let n := (Nat.floor (Real.sqrt t)).pred
  let time_for_n := total_time n
  let excess_time = t - time_for_n
  let (x, y) := movement_pattern n
  if excess_time % 2 == 0 then (x + excess_time / 2, y) else (x, y + excess_time / 2)

theorem particle_position_after_2023_minutes :
  final_position 2023 = (43, 43) :=
sorry

end particle_position_after_2023_minutes_l256_256756


namespace minimum_value_exists_l256_256603

noncomputable def min_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : ℝ :=
  x^2 + 3 * y

theorem minimum_value_exists :
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ (1 / (x + 3) + 1 / (y + 3) = 1 / 4) ∧ min_value x y _ _ _ = 20 + 16 * Real.sqrt 3 :=
sorry

end minimum_value_exists_l256_256603


namespace trapezoid_EC_length_l256_256333

-- Define a trapezoid and its properties.
structure Trapezoid (A B C D : Type) :=
(base1 : ℝ) -- AB
(base2 : ℝ) -- CD
(diagonal_AC : ℝ) -- AC
(AB_eq_3CD : base1 = 3 * base2)
(AC_length : diagonal_AC = 15)
(E : Type) -- point of intersection of diagonals

-- Proof statement that length of EC is 15/4
theorem trapezoid_EC_length
  {A B C D E : Type}
  (t : Trapezoid A B C D)
  (E : Type)
  (intersection_E : E) :
  ∃ (EC : ℝ), EC = 15 / 4 :=
by
  have h1 : t.base1 = 3 * t.base2 := t.AB_eq_3CD
  have h2 : t.diagonal_AC = 15 := t.AC_length
  -- Use the given conditions to derive the length of EC
  sorry

end trapezoid_EC_length_l256_256333


namespace product_first_2015_terms_l256_256500

noncomputable def sequence_a : ℕ → ℚ
| 0       := 2
| (n + 1) := (1 + sequence_a n) / (1 - sequence_a n)

theorem product_first_2015_terms :
  (Finset.range 2015).prod (λ n, sequence_a n) = 3 := 
sorry

end product_first_2015_terms_l256_256500


namespace max_pens_for_student_l256_256286

theorem max_pens_for_student : 
  ∀ (a b c d : ℕ), a < b → b < c → c < d → (a + b + c + d = 20) → 
  1 ≤ a → 1 ≤ b → 1 ≤ c → 1 ≤ d → d = 14 :=
begin
  sorry
end

end max_pens_for_student_l256_256286


namespace six_digit_product_of_consecutive_even_integers_l256_256430

theorem six_digit_product_of_consecutive_even_integers :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ a * b * c = 287232 :=
sorry

end six_digit_product_of_consecutive_even_integers_l256_256430


namespace relationship_m_n_p_l256_256860

variable {a b : ℝ}
noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x

-- Definitions based on the given conditions
def m := f ((a + b) / 2)
def n := f (Real.sqrt (a * b))
def p := f ((2 * a * b) / (a + b))

-- The theorem statement
theorem relationship_m_n_p (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (f_decreasing : ∀ x y : ℝ, x < y → f x > f y) :
  m < n ∧ n < p := by
  sorry

end relationship_m_n_p_l256_256860


namespace sum_of_all_modified_palindromes_is_45_l256_256207

/-- Let a, b, c be digits such that:
    - a is non-zero
    - b ∈ {0, 1, 2, 3, 4, 5}
    - c ∈ {0, 1, 2, 3, 4, 5, 6, 7}
    
  Then the sum of the digits of the total sum of all five-digit palindromes of the form 
  abcba is 45. -/
theorem sum_of_all_modified_palindromes_is_45 : 
  (∑ (a : ℕ) in Finset.range 1 10, 
     ∑ (b : ℕ) in Finset.range 6, 
       ∑ (c : ℕ) in Finset.range 8,
         2 * a + 5 * (9 - a) + (9 - 7 - c) +
         2 * b + 5) = 45 :=
sorry

end sum_of_all_modified_palindromes_is_45_l256_256207


namespace sum_of_divisors_distinct_prime_factors_l256_256107

theorem sum_of_divisors_distinct_prime_factors (n : ℕ) (h : n = 2^5 * 5^3) :
  (Σ x in finset.range(n + 1), if x ∣ n then x else 0).nat_prime_factors.card = 4 :=
sorry

end sum_of_divisors_distinct_prime_factors_l256_256107


namespace area_expression_correct_l256_256765

-- Define bases and height of the trapezoid
def base1 (h : ℝ) := 4 * h
def base2 (h : ℝ) := 5 * h
def height (h : ℝ) := h

-- Define the area of the trapezoid
def area_of_trapezoid (h : ℝ) := 1/2 * (base1 h + base2 h) * height h

-- Statement to prove the area of the trapezoid
theorem area_expression_correct (h : ℝ) : area_of_trapezoid h = 9 * h^2 / 2 := 
by 
  -- A placeholder for the actual proof
  sorry

end area_expression_correct_l256_256765


namespace functional_eq_res_l256_256592

theorem functional_eq_res (
  f : ℝ → ℝ 
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 + y) + 4 * f x * y)
) : 
  let n := 1
  let s := 0 in
  n * s = 0 :=
by
  sorry

end functional_eq_res_l256_256592


namespace vikas_rank_among_boys_l256_256529

theorem vikas_rank_among_boys (class : Type) 
  (Vikas Tanvi : class) 
  (students : list class)
  (index_Vikas : index_of Vikas students = 8) 
  (index_Tanvi : index_of Tanvi students = 16)
  (num_girls_between : count_girls_between Vikas Tanvi students = 2)
  (rank_Vikas_boys : rank_among_boys Vikas students = 4) : 
  rank_among_boys Vikas students = 4 :=
sorry

end vikas_rank_among_boys_l256_256529


namespace max_f_on_interval_min_f_on_interval_t_range_l256_256494

-- Declare constants and the function definition.
def f (x : ℝ) : ℝ := x^2 / Real.log x

-- Maximum value of f(x) on the interval [e^(1/4), e]
theorem max_f_on_interval : ∀ x ∈ Set.Icc (Real.exp (1/4)) (Real.exp 1), f x ≤ f (Real.exp 1) := 
sorry

-- Minimum value of f(x) on the interval [e^(1/4), e]
theorem min_f_on_interval : ∀ x ∈ Set.Icc (Real.exp (1/4)) (Real.exp 1), (f (Real.sqrt (Real.exp 1))) ≤ f x := 
sorry

-- Range for the real number t given the equation g(x)=tf(x)-x has two roots in the given intervals
theorem t_range : ∀ (t : ℝ), (Set.Icc (2 / Real.exp 2) (1 / Real.exp 1)).indicator ((λ t, (∃ x₁ x₂ ∈ Set.Icc (1 / Real.exp 1) 1 ∪ Set.Ioo 1 (Real.exp 2), g x₁ = 0 ∧ g x₂ = 0)) t) = 1 :=
sorry


-- Utility function for equation g(x)
def g (t x : ℝ) : ℝ := t * f x - x

end max_f_on_interval_min_f_on_interval_t_range_l256_256494


namespace tangent_circle_circumference_l256_256400

-- Definitions for the terms used in the conditions
def triangle_equilateral (A B C : Point) :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def arc_length (A C : Point) (r : ℝ) (θ : ℝ) := θ * r
def equilateral_triangle_angle: ℝ := (60 : ℝ) * (π / 180)

-- Main theorem statement
theorem tangent_circle_circumference (A B C : Point) (rAB : ℝ) (rTangent : ℝ) (circumference : ℝ) :
  triangle_equilateral A B C →
  arc_length B C rAB equilateral_triangle_angle = 15 →
  rAB = 45 / π →
  (circumference = 2 * π * rTangent) →
  (rTangent = (45 * π - 2025) / (360 * π^2)) →
  circumference = 33.75 :=
by 
  assume hABC heq1 heq2 heq3 heq4
  sorry

end tangent_circle_circumference_l256_256400


namespace radius_of_sphere_l256_256702

-- Define the necessary parameters: radii and apex angles of the cones
def r1 : ℝ := 1
def r2 : ℝ := 12
def r3 : ℝ := 12

def alpha1 : ℝ := -4 * Real.arctan (1/3)
def alpha2 : ℝ := 4 * Real.arctan (2/3)
def alpha3 : ℝ := 4 * Real.arctan (2/3)

-- Define the proof problem stating the radius of the sphere is 40/21
theorem radius_of_sphere :
  ∃ R : ℝ, (R = 40 / 21) ∧ 
           (∀ t1 t2 t3 : Cone, t1.radius = r1 ∧ t1.apex_angle = alpha1 ∧
                                     t2.radius = r2 ∧ t2.apex_angle = alpha2 ∧
                                     t3.radius = r3 ∧ t3.apex_angle = alpha3 ∧
                                     t1.tangential_to t2.tangential_to t3 ∧
                                     tangential (Sphere R) [] hok,nil sorry.

end radius_of_sphere_l256_256702


namespace monotonicity_f_range_of_a_l256_256078

noncomputable theory

-- Definitions of the functions
def f (a : ℝ) (x : ℝ) := (2*x - 1) * Real.exp x - a * (x^2 + x)
def g (a : ℝ) (x : ℝ) := -a * (x^2 + 1)

-- Monotonicity proof statement
theorem monotonicity_f (a : ℝ) :
  (a ≤ 0 →
    (∀ x, x < -0.5 → deriv (f a) x < 0) ∧
    (∀ x, x > -0.5 → deriv (f a) x > 0)) ∧
  (0 < a ∧ a < Real.exp (-0.5) →
    (∀ x, x < Real.log a → deriv (f a) x > 0) ∧
    (∀ x, Real.log a < x ∧ x < -0.5 → deriv (f a) x < 0) ∧
    (∀ x, x > -0.5 → deriv (f a) x > 0)) ∧
  (a = Real.exp (-0.5) → ∀ x, deriv (f a) x ≥ 0) ∧
  (a > Real.exp (-0.5) →
    (∀ x, x < -0.5 → deriv (f a) x > 0) ∧
    (∀ x, -0.5 < x ∧ x < Real.log a → deriv (f a) x < 0) ∧
    (∀ x, x > Real.log a → deriv (f a) x > 0)) :=
sorry

-- Range of 'a' proof statement
theorem range_of_a (a : ℝ) :
  (∀ x, f a x ≥ g a x) → (1 ≤ a ∧ a ≤ 4 * Real.exp 1.5) :=
sorry

end monotonicity_f_range_of_a_l256_256078


namespace sequence_pairwise_coprime_and_series_limit_l256_256669

-- Definition of the sequence
def T : ℕ → ℕ
| 0       := 2
| (n + 1) := (T n)^2 - (T n) + 1

-- Proof problem in Lean statement
theorem sequence_pairwise_coprime_and_series_limit :
  (∀ n m : ℕ, n ≠ m → Nat.gcd (T n) (T m) = 1) ∧ 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (∑ k in finset.range (n + 1), 1 / (T k) - 1) < ε) :=
sorry

end sequence_pairwise_coprime_and_series_limit_l256_256669


namespace distinct_shell_placements_l256_256576

-- Definitions based on conditions
def hexagram_points : Finset ℕ := {i | i < 12}

def symmetries : Finset (Finset ℕ) := {σ | ∃ (ω : ℕ) (i : Finset ℕ), ω < 12 ∧ i in S / (FiniteGroup.dihedral 12)}

-- We state the final proof theorem.
theorem distinct_shell_placements :
  let total_arrangements := nat.factorial 12 in
  let symmetry_count := 12 in
  total_arrangements / symmetry_count = 479001600 := by
  sorry

end distinct_shell_placements_l256_256576


namespace max_elements_in_T_l256_256938

noncomputable def max_elements_T (S : Finset ℕ) : ℕ :=
  if h : (∀ (a b : ℕ), a ≠ b → a ∈ S → b ∈ S → (a + b) % 10 ≠ 0) then S.card else 0

theorem max_elements_in_T : max {T : Finset ℕ // (∀ (a b : ℕ), a ≠ b → a ∈ T → b ∈ T → (a + b) % 10 ≠ 0) ∧ T ⊆ (Finset.range 100).map (λ n, n + 1)}.card = 6 := sorry

end max_elements_in_T_l256_256938


namespace compatible_polynomial_count_l256_256362

theorem compatible_polynomial_count (n : ℕ) : 
  ∃ num_polynomials : ℕ, num_polynomials = (n / 2) + 1 :=
by
  sorry

end compatible_polynomial_count_l256_256362


namespace find_k_collinear_l256_256505

def vector_collinear {α : Type*} [Field α] (a b : α × α) : Prop :=
  ∃ λ : α, a = (λ * b.1, λ * b.2)

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)
def k := - (16 : ℝ) / 13

theorem find_k_collinear : 
  vector_collinear (a.1 + k * c.1, a.2 + k * c.2) (2 * b.1 - a.1, 2 * b.2 - a.2) :=
sorry

end find_k_collinear_l256_256505


namespace rectangle_to_square_l256_256979

theorem rectangle_to_square (r : ℝ) (c : ℕ) : 
  r = 10 ∧ c = 7 → 
  ∃ (parts : list (set (ℝ × ℝ))), (∀ p ∈ parts, is_rectangular_piece p (1, 10)) 
   ∧ (∀ q ∈ parts, is_square q) := 
sorry

end rectangle_to_square_l256_256979


namespace given_complex_in_fourth_quadrant_l256_256915

def complex_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem given_complex_in_fourth_quadrant : complex_in_fourth_quadrant (4 + 3 * Complex.i) / (1 + Complex.i) :=
by
  sorry 

end given_complex_in_fourth_quadrant_l256_256915


namespace min_remainders_unique_l256_256453

theorem min_remainders_unique (x a r : ℕ) (hx : 100 ≤ x ∧ x + 3 ≤ 999)
  (ha : 10 ≤ a ∧ a + 3 ≤ 99) :
  (∃ r : ℕ, ∀ i, i ∈ {0, 1, 2, 3} → (x + i) % (a + i) = r) :=
begin
  sorry
end

end min_remainders_unique_l256_256453


namespace log_2_16384_eq_14_l256_256010

theorem log_2_16384_eq_14 (h : 16384 = 2 ^ 14) : log 2 16384 = 14 :=
by sorry

end log_2_16384_eq_14_l256_256010


namespace domain_of_function_l256_256811

noncomputable def domain (f : ℝ → ℝ) : set ℝ := {x | ∃ y, f x = y}

def function := λ x : ℝ, (x^5 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_function :
  domain function = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < 3) ∨ x > 3} :=
by
  sorry

end domain_of_function_l256_256811


namespace part_i_part_ii_l256_256582

noncomputable def a : ℕ → ℕ :=
  λ n => ⌊real.sqrt (2 * (n + 1) ^ 2 + n ^ 2)⌋

theorem part_i : ∃ᶠ (m : ℕ) in at_top, a (m + 1) - a m > 1 :=
  sorry

theorem part_ii : ∃ᶠ (m : ℕ) in at_top, a (m + 1) - a m = 1 :=
  sorry

end part_i_part_ii_l256_256582


namespace minimum_area_of_triangle_l256_256585

noncomputable def area_of_triangle (p q : ℤ) : ℚ :=
  5 * | -p + 3 * q |

theorem minimum_area_of_triangle :
  ∃ (p q : ℤ), area_of_triangle p q = 5 / 2 :=
by
  sorry

end minimum_area_of_triangle_l256_256585


namespace exists_n_divisible_by_5_l256_256670

open Int

theorem exists_n_divisible_by_5 
  (a b c d m : ℤ) 
  (h1 : 5 ∣ (a * m^3 + b * m^2 + c * m + d)) 
  (h2 : ¬ (5 ∣ d)) :
  ∃ n : ℤ, 5 ∣ (d * n^3 + c * n^2 + b * n + a) :=
by
  sorry

end exists_n_divisible_by_5_l256_256670


namespace count_not_divides_in_range_is_33_l256_256182

-- Definition of product of proper divisors of n
def g (n : ℕ) : ℕ := (Finset.filter (λ d, d ≠ n ∧ d ∣ n) (Finset.range (n + 1))).prod id

-- Condition 1: Defines the range for n
def in_range (n : ℕ) : Prop := 2 ≤ n ∧ n ≤ 100

-- Condition 2: n does not divide g(n)
def not_divides (n : ℕ) : Prop := ¬ (n ∣ g n)

-- Statement of the problem
theorem count_not_divides_in_range_is_33 : (Finset.filter (λ n, in_range n ∧ not_divides n) (Finset.range 101)).card = 33 := by
  sorry

end count_not_divides_in_range_is_33_l256_256182


namespace probability_al_multiple_bill_and_bill_multiple_cal_proof_l256_256768

noncomputable def probability_al_multiple_bill_and_bill_multiple_cal : ℚ :=
  let numbers := Set.univ.filter (λ n: ℕ, n > 0 ∧ n <= 12)
  let even_numbers := Set.univ.filter (λ n: ℕ, n % 2 = 0 ∧ n > 0 ∧ n <= 12)
  let total_assignments := 6 * 11 * 10
  let valid_assignments := 4
  valid_assignments / total_assignments

theorem probability_al_multiple_bill_and_bill_multiple_cal_proof :
  probability_al_multiple_bill_and_bill_multiple_cal = 1 / 165 :=
  by
    sorry

end probability_al_multiple_bill_and_bill_multiple_cal_proof_l256_256768


namespace complement_M_intersect_N_l256_256959

def M : Set ℤ := {m | m ≤ -3 ∨ m ≥ 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}
def complement_M : Set ℤ := {m | -3 < m ∧ m < 2} 

theorem complement_M_intersect_N : (complement_M ∩ N) = {-1, 0, 1} := by
  sorry

end complement_M_intersect_N_l256_256959


namespace find_a_to_satisfy_divisibility_l256_256844

theorem find_a_to_satisfy_divisibility (a : ℕ) (h₀ : 0 ≤ a) (h₁ : a < 11) (h₂ : (2 * 10^10 + a) % 11 = 0) : a = 9 :=
sorry

end find_a_to_satisfy_divisibility_l256_256844


namespace other_train_length_l256_256344

noncomputable def length_of_other_train
  (l1 : ℝ) (v1_kmph : ℝ) (v2_kmph : ℝ) (t : ℝ) : ℝ :=
  let v1 := (v1_kmph * 1000) / 3600
  let v2 := (v2_kmph * 1000) / 3600
  let relative_speed := v1 + v2
  let total_distance := relative_speed * t
  total_distance - l1

theorem other_train_length
  (l1 : ℝ) (v1_kmph : ℝ) (v2_kmph : ℝ) (t : ℝ)
  (hl1 : l1 = 230)
  (hv1 : v1_kmph = 120)
  (hv2 : v2_kmph = 80)
  (ht : t = 9) :
  length_of_other_train l1 v1_kmph v2_kmph t = 269.95 :=
by
  rw [hl1, hv1, hv2, ht]
  -- Proof steps skipped
  sorry

end other_train_length_l256_256344


namespace arrange_numbers_in_grid_l256_256513

theorem arrange_numbers_in_grid :
  let nums := [1, 2, 3, 4, 5, 6, 7, 8]
  ∃ (m : matrix (fin 2) (fin 4) ℕ),
  (∀ i, ∑ j, m i j = 6 * (∑ k, m i (some k mod 2))) ∧
  (∀ j, ∑ i, m i j = 3 * (∑ l, m (some l mod 2) j)) ∧
  multiset.card (m.to_multiset) = 8 ∧
  (multiset.filter (λ x, x ∈ nums) (m.to_multiset) = nums.to_multiset)
  → fintype.card { m : matrix (fin 2) (fin 4) ℕ // 
      (∀ i, ((∑ j, m i j) % 6 = 0)) ∧ 
      (∀ j, ((∑ i, m i j) % 3 = 0)) } = 288 :=
by
  -- Your proof goes here
  sorry

end arrange_numbers_in_grid_l256_256513


namespace ribbons_left_l256_256664

theorem ribbons_left (initial_A : ℕ) (initial_B : ℕ) : 
  let used_A := 4 + 8 in  -- ribbons used for odd and even gifts
  let used_B := 8 + 4 in  -- ribbons used for odd and even gifts
  (initial_A = 10) →
  (initial_B = 12) →
  initial_A - used_A = -2 ∧ initial_B - used_B = 0 :=
by
  intros initial_A_eq initial_B_eq
  simp [initial_A_eq, initial_B_eq]
  sorry

end ribbons_left_l256_256664


namespace find_k_l256_256481

-- Define the normal vectors
def vec_n1 : Vector3 := ⟨1, 2, -3⟩
def vec_n2 (k : ℝ) : Vector3 := ⟨-2, -4, k⟩

-- Define the condition of parallelism
def parallel_planes (α β : ℝ) := α / β = vec_n1.z / vec_n2 β.z

theorem find_k (k : ℝ) (h₁ : vec_n2 k = ⟨-2, -4, k⟩) (h₂ : parallel_planes 1 (-2)) : k = 6 :=
by
  sorry

end find_k_l256_256481


namespace probability_at_least_one_coordinate_greater_l256_256600

theorem probability_at_least_one_coordinate_greater (p : ℝ) :
  (∃ (x y : ℝ), (0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ (x > p ∨ y > p))) ↔ p = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end probability_at_least_one_coordinate_greater_l256_256600


namespace helen_gas_needed_l256_256506

-- Defining constants for the problem
def largeLawnGasPerUsage (n : ℕ) : ℕ := (n / 3) * 2
def smallLawnGasPerUsage (n : ℕ) : ℕ := (n / 2) * 1

def monthsSpringFall : ℕ := 4
def monthsSummer : ℕ := 4

def largeLawnCutsSpringFall : ℕ := 1
def largeLawnCutsSummer : ℕ := 3

def smallLawnCutsSpringFall : ℕ := 2
def smallLawnCutsSummer : ℕ := 2

-- Number of times Helen cuts large lawn in March-April and September-October
def largeLawnSpringFallCuts : ℕ := monthsSpringFall * largeLawnCutsSpringFall

-- Number of times Helen cuts large lawn in May-August
def largeLawnSummerCuts : ℕ := monthsSummer * largeLawnCutsSummer

-- Total cuts for large lawn
def totalLargeLawnCuts : ℕ := largeLawnSpringFallCuts + largeLawnSummerCuts

-- Number of times Helen cuts small lawn in March-April and September-October
def smallLawnSpringFallCuts : ℕ := monthsSpringFall * smallLawnCutsSpringFall

-- Number of times Helen cuts small lawn in May-August
def smallLawnSummerCuts : ℕ := monthsSummer * smallLawnCutsSummer

-- Total cuts for small lawn
def totalSmallLawnCuts : ℕ := smallLawnSpringFallCuts + smallLawnSummerCuts

-- Total gas needed for both lawns
def totalGasNeeded : ℕ :=
  largeLawnGasPerUsage totalLargeLawnCuts + smallLawnGasPerUsage totalSmallLawnCuts

-- The statement to prove
theorem helen_gas_needed : totalGasNeeded = 18 := sorry

end helen_gas_needed_l256_256506


namespace angle_CAB_in_regular_hexagon_l256_256539

theorem angle_CAB_in_regular_hexagon
  (ABCDEF : Type)
  [Hexagon : ∀ (i : ℕ), ABCDEF]         -- Define the regular hexagon type
  (regular : ∀ (i : ℕ), ∃ (angle : ℝ), angle = 120) -- Interior angles are 120 degrees
  (symmetry : ∀ (A B C : ABCDEF), ∃ (x : ℝ), x = ∠CAB = ∠BCA) -- Symmetry of the hexagon
: ∀ (AC : Diagonal ABCDEF), (∠CAB = 30) := sorry

end angle_CAB_in_regular_hexagon_l256_256539


namespace find_angle_AHB_l256_256741

theorem find_angle_AHB
  (A B G : Point)
  (circle : Circle A B G)
  (angle_AGB : ∠AGB = 48) 
  (C D : Point)
  (trisect_chord : between_trisection A B C D) 
  (E F : Point)
  (trisect_arc : arc_trisection A B E F)
  (H: Point)
  (intersect_EC_FD : intersection E C F D H) : 
  ∠AHB = 32 :=
sorry

end find_angle_AHB_l256_256741


namespace simple_vs_compound_loss_l256_256923

theorem simple_vs_compound_loss :
  let P : ℝ := 1250
  let r1 : ℝ := 0.08
  let r2 : ℝ := 0.10
  let r3 : ℝ := 0.12
  let s1 : ℝ := 0.04
  let s2 : ℝ := 0.06
  let s3 : ℝ := 0.07
  let s4 : ℝ := 0.09
  let compound_amount : ℝ := 
        P * (1 + r1) * (1 + r2) * (1 + r3)
  let simple_interest : ℝ := 
        P * s1 + P * s2 + P * s3 + P * s4
  let simple_amount : ℝ := 
        P + simple_interest
  let loss : ℝ := 
        compound_amount - simple_amount
  in loss = 88.2 := sorry

end simple_vs_compound_loss_l256_256923


namespace intersection_of_lines_l256_256800

theorem intersection_of_lines :
  ∃ (x y : ℚ), 8 * x - 5 * y = 4 ∧ 6 * x + 2 * y = 18 ∧ x = 49 / 23 ∧ y = 60 / 23 :=
by {
  use [49 / 23, 60 / 23],
  simp,
  split,
  { norm_num, },
  { norm_num, split; refl }
}

end intersection_of_lines_l256_256800


namespace distance_traveled_l256_256691

-- Definitions for the problem
def radius : ℝ := 10 -- The length of the second hand in cm
def period : ℝ := 15 -- The period in minutes
def one_revolution : ℝ := 2 * Real.pi * radius -- Circumference of the circle

-- Theorem to prove the total distance traveled
theorem distance_traveled : 
  ∃ D : ℝ, (D = (period * one_revolution)) ∧ (D = 300 * Real.pi) := 
by
  -- Use sorry to skip the proof details
  sorry

end distance_traveled_l256_256691


namespace part1_part2_l256_256961

-- Define the sequence and its sum
def a (n : ℕ) : ℕ → ℝ 
def S (n : ℕ) : ℝ 

-- Conditions
axiom h1 : ∀ n, S (n + 1) = S n + a (n + 1) + 3^n
axiom h2 : a 1 ≠ 3
axiom h3 : ∀ n, n ≥ 1 → a (n + 1) = S n + 3^n

-- Prove that the sequence {S_n - 3^n} is geometric with common ratio 2
theorem part1 : ∃ r, ∀ n, S (n + 1) - 3^(n + 1) = r * (S n - 3^n) :=
by sorry

-- Given that {a_n} is an increasing sequence, find the range of values for a_1
theorem part2 (h4 : ∀ n, a (n + 1) > a n) : a 1 > -9 :=
by sorry

end part1_part2_l256_256961


namespace determinant_evaluation_l256_256421

variables (x y z : ℝ)

def matrix := !![
  [1, x, x + y + z],
  [1, y, x + z],
  [1, z, y + z]
]

theorem determinant_evaluation : matrix.det = x^2 - x * y * z := by
  sorry

end determinant_evaluation_l256_256421


namespace area_of_triangle_APQ_proof_l256_256640

noncomputable def area_of_triangle_APQ : ℝ :=
  let ABC := (A B C : Point) := {a : ℝ | a ≤ 10}
  let P := (P : Point) := {p : ℝ | 0 ≤ p ∧ p ≤ 10}
  let Q := (Q : Point) := {q : ℝ | 0 ≤ q ∧ q ≤ 10}
  let pq := (pq : ℝ) := 4
  let x := (x : ℝ) := AP
  let y := (y : ℝ) := AQ
  let XY := (S : ℝ) := xy
  XY = {S : ℝ | (x, y) ∈ ABC × P × Q, pq = 4, S = PQAP.area}
  let area_APQ := (S : ℝ) := 5 / sqrt 3

theorem area_of_triangle_APQ_proof :
  area_of_triangle_APQ = 5 / sqrt 3 := by sorry

end area_of_triangle_APQ_proof_l256_256640


namespace count_not_divides_in_range_is_33_l256_256185

-- Definition of product of proper divisors of n
def g (n : ℕ) : ℕ := (Finset.filter (λ d, d ≠ n ∧ d ∣ n) (Finset.range (n + 1))).prod id

-- Condition 1: Defines the range for n
def in_range (n : ℕ) : Prop := 2 ≤ n ∧ n ≤ 100

-- Condition 2: n does not divide g(n)
def not_divides (n : ℕ) : Prop := ¬ (n ∣ g n)

-- Statement of the problem
theorem count_not_divides_in_range_is_33 : (Finset.filter (λ n, in_range n ∧ not_divides n) (Finset.range 101)).card = 33 := by
  sorry

end count_not_divides_in_range_is_33_l256_256185


namespace distance_between_foci_of_hyperbola_l256_256441

theorem distance_between_foci_of_hyperbola (a b c : ℝ) :
  a^2 = 25 → b^2 = 4 → c^2 = a^2 + b^2 → 2 * c = 2 * real.sqrt 29 :=
begin
  intros ha hb hc,
  rw [ha, hb] at hc,
  have hc' : c = real.sqrt 29,
  { apply eq_of_pow_eq_pow two_ne_zero,
    rw hc },
  rw hc',
  ring
end

end distance_between_foci_of_hyperbola_l256_256441


namespace coeff_x2_binomial_expansion_l256_256957
open Real

def integral_value : ℝ := 3 * (∫ x in - (π / 2) .. (π / 2), cos x)

def binomial_expansion (x : ℝ) (n : ℕ) := (1 / 2 * x - sqrt 2) ^ n

theorem coeff_x2_binomial_expansion :
  integral_value = 6 →
  let f (x : ℝ) := binomial_expansion x 6 in
  ∃ c : ℝ, (∀ (g : ℕ → ℝ) (x : ℝ), (f x) = ∑ i in (finset.range 7), g i * x ^ i) →
  c = 15 ∧ (∀ i, g i = if i = 2 then c else 0) :=
begin
  intros h,
  obtain ⟨g, hg⟩ := exists_nat_cast (f x = ∑ i in (finset.range 7), g i * x ^ i),
  use 15,
  split,
  sorry,
  intro i,
  by_cases i = 2,
  exact h,
  exact ne.symm h
end

end coeff_x2_binomial_expansion_l256_256957


namespace increasing_interval_l256_256271

noncomputable def y (a x : ℝ) : ℝ := a^(x^2 - 3 * x + 2)

theorem increasing_interval (a : ℝ) (h : a > 1) : 
  ∃ (I : set ℝ), I = set.Ici (3 / 2) ∧ 
  ∀ x1 x2, x1 ∈ I → x2 ∈ I → x1 < x2 → y a x1 < y a x2 :=
sorry

end increasing_interval_l256_256271


namespace determine_function_f_l256_256681

-- Conditions
def translated_one_unit_left (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 1) = g x

def symmetric_about_y_axis (g h : ℝ → ℝ) : Prop :=
  ∀ x, g x = h (-x)

-- Question (to be proved)
theorem determine_function_f (f : ℝ → ℝ) :
  translated_one_unit_left f (λ x, (1/2) ^ x) ∧ symmetric_about_y_axis (λ x, (1/2) ^ x) (λ x, 2 ^ x) →
  ∀ x, f x = (1/2) ^ (x - 1) :=
by
  intros h,
  sorry

end determine_function_f_l256_256681


namespace average_age_of_two_women_is_30_l256_256735

-- Given definitions
def avg_age_before_replacement (A : ℝ) := 8 * A
def avg_age_after_increase (A : ℝ) := 8 * (A + 2)
def ages_of_men_replaced := 20 + 24

-- The theorem to prove: the average age of the two women is 30 years
theorem average_age_of_two_women_is_30 (A : ℝ) :
  (avg_age_after_increase A) - (avg_age_before_replacement A) = 16 →
  (ages_of_men_replaced + 16) / 2 = 30 :=
by
  sorry

end average_age_of_two_women_is_30_l256_256735


namespace figure_100_squares_l256_256427

theorem figure_100_squares : (∃ f : ℕ → ℕ, f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 ∧ f 100 = 30301) :=
  sorry

end figure_100_squares_l256_256427


namespace missing_score_and_variance_l256_256533

theorem missing_score_and_variance (score_A score_B score_D score_E : ℕ) (avg_score : ℕ)
  (h_scores : score_A = 81 ∧ score_B = 79 ∧ score_D = 80 ∧ score_E = 82)
  (h_avg : avg_score = 80):
  ∃ (score_C variance : ℕ), score_C = 78 ∧ variance = 2 := by
  sorry

end missing_score_and_variance_l256_256533


namespace extremum_iff_derivative_zero_l256_256121

variable (f : ℝ → ℝ)
variable (x₀ : ℝ)

theorem extremum_iff_derivative_zero :
  (∀ (x₀ : ℝ), has_deriv_at f x₀) → (∀ (x₀ : ℝ), x₀ is_extremum → deriv f x₀ = 0) ∧ ¬(∀ (x₀ : ℝ), deriv f x₀ = 0 → x₀ is_extremum) :=
by
  sorry

end extremum_iff_derivative_zero_l256_256121


namespace solution_set_for_inequality_l256_256070

theorem solution_set_for_inequality (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (hf_mono : Monotonic f)
  (h_inv : ∀ y, f (f_inv y) = y ∧ f_inv (f y) = y)
  (hA : f (-1) = 3)
  (hB : f (1) = 1) :
  {x : ℝ | |f_inv (2^x)| < 1} = {x : ℝ | 0 < x ∧ x < Real.log 3 / Real.log 2} :=
  sorry

end solution_set_for_inequality_l256_256070


namespace set_intersection_l256_256873

def R := set.univ : set ℝ
def A : set ℝ := { x | x^2 - 2 * x < 0 }
def B : set ℝ := { x | 1 ≤ x }
def complement (S : set ℝ) : set ℝ := { x | x ∉ S }

theorem set_intersection :
  A ∩ complement B = { x | 0 < x ∧ x < 1 } :=
by simp [A, B, complement, set.ext_iff]; intro x; split; intro h; cases h; 
   simp at h_left ⊢; linarith


end set_intersection_l256_256873


namespace machine_value_depletion_time_l256_256755

theorem machine_value_depletion_time :
  ∃ t : ℝ, 972 = 1200 * (0.90 ^ t) → t ≈ 2 :=
by
  sorry

end machine_value_depletion_time_l256_256755


namespace statement1_cannot_support_safety_l256_256986

variable (airAsiaCrashed : Bool) (deaths2014 : ℕ := if airAsiaCrashed then 1320 else 0)
variable (carAccidentsDeathsYearly : ℕ := 1240000)
variable (maxAviationDeathsYear : ℕ := 3346)
variable (accidentsPerMillionFlights : Float := 2.1)
variable (deathsPerMillionCarTraveling : Float := 100.0)

theorem statement1_cannot_support_safety :
  ¬(deaths2014 ≤ maxAviationDeathsYear ∧ (carAccidentsDeathsYearly / 1.24e6) > (deathsPerMillionCarTraveling / 1.0e3)).  

end statement1_cannot_support_safety_l256_256986


namespace diagonals_from_one_vertex_icosikaipentagon_l256_256701

-- Define an icosikaipentagon as a 25-sided polygon
def is_icosikaipentagon (n : ℕ) : Prop := n = 25

-- Define the theorem to prove the number of diagonals from one vertex
theorem diagonals_from_one_vertex_icosikaipentagon :
  ∀ {n : ℕ}, is_icosikaipentagon n → (n - 3) = 22 :=
by
  intros n hn
  rw [hn]
  norm_num
  sorry


end diagonals_from_one_vertex_icosikaipentagon_l256_256701


namespace polyhedron_edges_and_vertices_l256_256360

-- Define the conditions
def num_faces : ℕ := 20
def num_edges (X : ℕ) : ℕ := X / 2
def num_vertices (A F : ℕ) : ℕ := A - F + 2

-- State the main theorem
theorem polyhedron_edges_and_vertices 
  (F : ℕ := 20)
  (T : Type) [fintype T] (X : ℕ)
  (edge_face_incidence : ∀ t : T, fintype.card (finset.fintype (set_of (λ p : T × T, snd p = t))) = 3)
  : num_edges 60 = 30 ∧ num_vertices 30 F = 12 :=
by
  sorry

end polyhedron_edges_and_vertices_l256_256360


namespace x4_plus_inverse_x4_l256_256321

theorem x4_plus_inverse_x4 (x : ℝ) (hx : x ^ 2 + 1 / x ^ 2 = 2) : x ^ 4 + 1 / x ^ 4 = 2 := 
sorry

end x4_plus_inverse_x4_l256_256321


namespace cakes_sold_eq_l256_256781

variable {cakes_initial cakes_left cakes_sold : ℕ}

-- Definitions based on the conditions
def initial_cakes := 167
def cakes_left := 59

-- The statement we want to prove
theorem cakes_sold_eq : cakes_sold = initial_cakes - cakes_left → cakes_sold = 108 := 
by
  sorry

end cakes_sold_eq_l256_256781


namespace function_is_zero_l256_256429

theorem function_is_zero (f : ℝ → ℝ)
  (H : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end function_is_zero_l256_256429


namespace maximum_possible_number_of_digits_of_A_l256_256817

def P (n : ℕ) : ℕ := 
  n.digits.map (λ d, d.natAbs).prod

def S (n : ℕ) : ℕ := 
  n.digits.map (λ d, d.natAbs).sum

def A : Set ℕ := 
  { n | (P n) ≠ 0 ∧ (Nat.squareFree (P n)) ∧ (S n ∣ P n) ∧ (S n ≠ P n) }

def max_digits_of_A :=
  92

theorem maximum_possible_number_of_digits_of_A :
  Exists.elem_of A → max_digits_of_A = 92 :=
  sorry

end maximum_possible_number_of_digits_of_A_l256_256817


namespace initial_inventory_correct_l256_256404

-- Define the conditions as given in the problem
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_wed_to_sun : ℕ := 50
def days_wed_to_sun : ℕ := 5
def bottles_delivered_saturday : ℕ := 650
def final_inventory : ℕ := 1555

-- Define the total number of bottles sold during the week
def total_bottles_sold : ℕ :=
  bottles_sold_monday + bottles_sold_tuesday + (bottles_sold_per_day_wed_to_sun * days_wed_to_sun)

-- Define the initial inventory calculation
def initial_inventory : ℕ :=
  final_inventory + total_bottles_sold - bottles_delivered_saturday

-- The theorem we want to prove
theorem initial_inventory_correct :
  initial_inventory = 4500 :=
by
  sorry

end initial_inventory_correct_l256_256404


namespace line_tangent_slope_and_m_l256_256269

def line := {p : ℝ × ℝ | p.1 - 2 * p.2 - 1 = 0}
def circle (m : ℝ) := {p : ℝ × ℝ | p.1 ^ 2 + (p.2 - m) ^ 2 = 1}

theorem line_tangent_slope_and_m (m : ℝ) :
  (∀ p : ℝ × ℝ, p ∈ line → p.1 - 2 * p.2 - 1 = 0) →
  (∀ p : ℝ × ℝ, p ∈ circle m → p.1 ^ 2 + (p.2 - m) ^ 2 = 1) →
  (∃ s : ℝ, s = 1 / 2) ∧ (m = (-1 + Real.sqrt 5) / 2 ∨ m = (-1 - Real.sqrt 5) / 2) :=
by
  sorry

end line_tangent_slope_and_m_l256_256269


namespace polynomial_multiple_real_pos_coeffs_iff_no_real_pos_root_l256_256168

noncomputable def hasRealPosCoeffs (P Q : Polynomial ℂ → Polynomial ℝ → Prop) :=
  ∃ c : ℂ, c ≠ 0 ∧ Q = Polynomial.scale c P

theorem polynomial_multiple_real_pos_coeffs_iff_no_real_pos_root (P : Polynomial ℂ) (hP0 : P.eval 0 ≠ 0) :
  (∃ Q : Polynomial ℝ, hasRealPosCoeffs P Q) ↔ ¬ ∃ r : ℝ, r > 0 ∧ P.eval r = 0 :=
by sorry

end polynomial_multiple_real_pos_coeffs_iff_no_real_pos_root_l256_256168


namespace center_inscribed_circle_on_fixed_line_area_triangle_PAB_given_angle_l256_256324

-- Definitions and conditions
structure Point where
  x : ℝ
  y : ℝ

def line (m : ℝ) (p : Point) : ℝ → ℝ :=
  λ x => m * x + p.y - m * p.x

def ellipse_eq (a b x y : ℝ) : Prop := 
  (x^2)/(a^2) + (y^2)/(b^2) = 1

def A := Point -- Intersection of line with ellipse
def B := Point -- Intersection of line with ellipse
def P : Point := ⟨3 * sqrt 2, sqrt 2⟩

-- Given conditions
variables (l : ℝ → ℝ)
def slope_l : ℝ := 1 / 3
def ellipse : Point → Prop := ellipse_eq 6 2

axioms 
  (h1 : l = line slope_l P)
  (h2 : ellipse A)
  (h3 : ellipse B)

-- Questions as theorem statements
theorem center_inscribed_circle_on_fixed_line :
  ∃ k : ℝ, k = 3 * sqrt 2 :=
sorry

theorem area_triangle_PAB_given_angle :
  ∃ area : ℝ, 
    (angle := 60),
    sin (π / 3) = sqrt 3 / 2 →
    area = (sqrt 3 / 4) * ((3 * sqrt 2) ^ 2 / 2) :=
sorry

end center_inscribed_circle_on_fixed_line_area_triangle_PAB_given_angle_l256_256324


namespace cos_YXW_correct_l256_256157

noncomputable def cos_angle {α β γ : ℝ} (XY XZ YZ : ℝ) (W : ℝ) : ℝ :=
  let cos_YXZ := (XY^2 + XZ^2 - YZ^2) / (2 * XY * XZ)
  in Real.sqrt ((1 + cos_YXZ) / 2)

theorem cos_YXW_correct :
  cos_angle 5 7 9 = Real.sqrt 0.45 :=
by
  -- Definitions and assumptions
  let XY := 5
  let XZ := 7
  let YZ := 9
  let W := 0  -- W is on YZ, assuming a parametric representation
  let cos_YXZ := (XY^2 + XZ^2 - YZ^2) / (2 * XY * XZ)
  
  -- Calculate cos_YXW using the angle bisector property
  have cos_YXW : ℝ := Real.sqrt ((1 + cos_YXZ) / 2)
  
  -- Prove that the calculated value is equal to sqrt(0.45)
  have hx : cos_YXW = Real.sqrt 0.45, 
    from sorry,
  
  exact hx

end cos_YXW_correct_l256_256157


namespace paco_cookies_l256_256648

theorem paco_cookies :
  let initial_cookies := 25
  let ate_cookies := 5
  let remaining_cookies_after_eating := initial_cookies - ate_cookies
  let gave_away_cookies := 4
  let remaining_cookies_after_giving := remaining_cookies_after_eating - gave_away_cookies
  let bought_cookies := 3
  let final_cookies := remaining_cookies_after_giving + bought_cookies
  let combined_bought_and_gave_away := gave_away_cookies + bought_cookies
  (ate_cookies - combined_bought_and_gave_away) = -2 :=
by sorry

end paco_cookies_l256_256648


namespace find_N_value_l256_256960

-- Definitions based on given conditions
def M (n : ℕ) : ℕ := 4^n
def N (n : ℕ) : ℕ := 2^n
def condition (n : ℕ) : Prop := M n - N n = 240

-- Theorem statement to prove N == 16 given the conditions
theorem find_N_value (n : ℕ) (h : condition n) : N n = 16 := 
  sorry

end find_N_value_l256_256960


namespace find_angles_of_triangle_AMB_l256_256748

-- Define the geometric entities and conditions
structure Triangle :=
(A B C : Point)

noncomputable def EquilateralTriangle (t : Triangle) : Prop :=
angle t.A t.B t.C = 60 ∧ angle t.B t.C t.A = 60 ∧ angle t.C t.A t.B = 60

structure Circle :=
(center : Point)
(radius : ℝ)

structure Arc :=
(start end : Point)
(center : Point)

noncomputable def DividesArcInRatio (M : Point) (arc : Arc) (ratio1 : ℝ) (ratio2 : ℝ) : Prop :=
let total_degree := angle arc.start arc.center arc.end
in angle arc.start arc.center M = ratio1 / (ratio1 + ratio2) * total_degree ∧
   angle M arc.center arc.end = ratio2 / (ratio1 + ratio2) * total_degree

-- Proof problem statement
theorem find_angles_of_triangle_AMB (t : Triangle) (c : Circle) (M : Point) (arc : Arc)
    (h1 : EquilateralTriangle t)
    (h2 : c = Circle t.A (distance t.A t.B))
    (h3 : arc = Arc t.B t.C t.A)
    (h4 : DividesArcInRatio M arc 1 2) :
  (angle t.A M t.B = 40 ∧ angle t.B t.A M = 80 ∧ angle M t.B t.C = 60) ∨
  (angle t.A M t.B = 20 ∧ angle t.B t.A M = 100 ∧ angle M t.B t.C = 60) :=
sorry

end find_angles_of_triangle_AMB_l256_256748


namespace functional_equation_l256_256194

noncomputable def g (x : ℝ) : ℝ := x + 4

theorem functional_equation (x y : ℝ) :
  (g(x) * g(y) - g(x * y)) / 4 = x + y + 3 :=
by
  sorry

end functional_equation_l256_256194


namespace exists_x_bound_sum_l256_256461

theorem exists_x_bound_sum (n : ℕ) (p : Fin n → ℝ) 
  (h : ∀ i : Fin n, 0 ≤ p i ∧ p i ≤ 1) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ 
  (∑ i, 1 / |x - p i| ≤ 8 * n * (∑ k in Finset.range n, 1 / (2 * k + 1))) := 
sorry

end exists_x_bound_sum_l256_256461


namespace find_area_of_triangle_APQ_l256_256624

/-
  Given an equilateral triangle ABC with side length 10,
  and points P and Q on sides AB and AC such that PQ = 4 and
  PQ is tangent to the incircle of ABC,
  prove that the area of triangle APQ is equal to 5 * sqrt 3 / 3.
-/

def area_of_triangle_APQ  : ℝ :=
  let side_length := 10
  let PQ_length := 4
  let APQ_area := (5 * Real.sqrt 3) / 3
  APQ_area

theorem find_area_of_triangle_APQ :
  ∃ (P Q : (fin 2) → ℝ) (APQ_area: ℝ),
  (P 0).dist (P 1) = PQ_length ∧ (Q 0).dist (Q 1) = PQ_length ∧
  APQ_area = area_of_triangle_APQ ∧ 
  APQ_area = (5 * Real.sqrt 3) / 3 :=
by
  sorry

end find_area_of_triangle_APQ_l256_256624


namespace move_arrows_706_to_709_l256_256527

def position_in_cycle (n : Nat) : Nat := n % 6

theorem move_arrows_706_to_709 :
  position_in_cycle 706 = 4 →
  position_in_cycle 709 = 1 →
  ───────────────⟩ {a := 4, b := 5, c := 6, d := 1} :
  seq current_position 706 (next_moves 706 709)
sorry

end move_arrows_706_to_709_l256_256527


namespace max_min_values_l256_256813

namespace ProofPrimary

-- Define the polynomial function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 36 * x + 1

-- State the interval of interest
def interval : Set ℝ := Set.Icc 1 11

-- Main theorem asserting the minimum and maximum values
theorem max_min_values : 
  (∀ x ∈ interval, f x ≥ -43 ∧ f x ≤ 2630) ∧
  (∃ x ∈ interval, f x = -43) ∧
  (∃ x ∈ interval, f x = 2630) :=
by
  sorry

end ProofPrimary

end max_min_values_l256_256813


namespace george_reels_per_day_l256_256526

theorem george_reels_per_day
  (days : ℕ := 5)
  (jackson_per_day : ℕ := 6)
  (jonah_per_day : ℕ := 4)
  (total_fishes : ℕ := 90) :
  (∃ george_per_day : ℕ, george_per_day = 8) :=
by
  -- Calculation steps are skipped here; they would need to be filled in for a complete proof.
  sorry

end george_reels_per_day_l256_256526


namespace simplify_expression_l256_256241

variable (a : ℝ) (ha : a > 0)

theorem simplify_expression : 
  ( ( (a ^ (16 / 5)) ^ (1 / 4)) ^ 6 * ( (a ^ (16 / 4)) ^ (1 / 5)) ^ 6 ) = a ^ 9.6 :=
by
  sorry

end simplify_expression_l256_256241


namespace locus_point_C_l256_256833

open_locale real

variables {a x y t : ℝ} 

def fixed_point_A (a : ℝ) : (ℝ × ℝ) := (a, 0)

def line_l : set (ℝ × ℝ) := {p | p.1 = 1}

def point_B (t : ℝ) : (ℝ × ℝ) := (1, t)

def on_segment_AB (A B C : ℝ × ℝ) : Prop :=
  ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ C = (s * A.1 + (1 - s) * B.1, s * A.2 + (1 - s) * B.2)

theorem locus_point_C (a : ℝ) (ha : 0 < a) (t : ℝ) (C : ℝ × ℝ)
  (hC : on_segment_AB (fixed_point_A a) (point_B t) C) :
  (1 - a) * C.1^2 - 2 * a * C.1 + (1 + a) * C.2^2 = 0 :=
sorry

end locus_point_C_l256_256833


namespace g_at_pi_over_3_l256_256859

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

noncomputable def g (x : ℝ) (ω φ : ℝ) : ℝ := 3 * Real.sin (ω * x + φ) - 1

theorem g_at_pi_over_3 (ω φ : ℝ) :
  (∀ x : ℝ, f (π / 3 + x) ω φ = f (π / 3 - x) ω φ) →
  g (π / 3) ω φ = -1 :=
by sorry

end g_at_pi_over_3_l256_256859


namespace sum_of_g_45_l256_256177

def f (x : ℝ) := 5 * x^2 - 4
def g (y : ℝ) := 2 * (y / 5) + 3 * (sqrt (y / 5 + 0.8)) + 2

theorem sum_of_g_45 : (g 45 + g 45 = 43.2) :=
  sorry

end sum_of_g_45_l256_256177


namespace area_inequality_l256_256919

namespace Geometry

open Real

variables (a b c : ℝ) (A B C D K L O1 O2 : Point)

-- Define the right triangle ABC with a right angle at A
def right_triangle (A B C : Point) : Prop :=
  angle A B C = 90 ∧ angle C A B = 90 ∧ angle B C A = 90

-- Define the foot of the perpendicular from A to BC as D
def perpendicular_foot (A B C D : Point) : Prop :=
  is_perpendicular (line_through A D) (line_through B C)

-- Define the centers of the incircles of triangles ABD and ACD
def incircle_center (P Q R T : Point) : Point :=
  let center := (P + Q + R) / 3 in center

-- Define the points K and L
def intersects_at (line1 line2 : Line) (P Q : Point) : Prop :=
  P ∈ line1 ∧ Q ∈ line2 ∧ P = Q

-- Define the areas S and T
def triangle_area (A B C : Point) : ℝ :=
  (abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2))

-- Define the statement to prove
theorem area_inequality (h1 : right_triangle A B C) 
  (h2 : perpendicular_foot A B C D) 
  (h3 : incircle_center A B D O1) 
  (h4 : incircle_center A C D O2) 
  (h5 : intersects_at (line_through O1 O2) (line_through A B) K) 
  (h6 : intersects_at (line_through O1 O2) (line_through A C) L) :
  let S := triangle_area A B C in
  let T := triangle_area A K L in
  S ≥ 2 * T :=
sorry

end Geometry

end area_inequality_l256_256919


namespace part_I_part_II_l256_256497

-- Define the function f(x) = |x-1| + |x+1|
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Part (I): Prove the set where f(x) < 3 is (-3/2, 3/2)
theorem part_I : {x : ℝ | f x < 3} = set.Ioo (-(3/2): ℝ) (3/2: ℝ) :=
by
  sorry

-- Part (II): Prove the minimum value of (1/a + 2/b) under given conditions is (3/2 + sqrt(2))
theorem part_II {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 2) : 
  (∀ a b : ℝ, 0 < a → 0 < b → a + b = 2 → (1/a + 2/b) ≥ (3/2 + real.sqrt 2)) ∧ 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ (1/a + 2/b) = (3/2 + real.sqrt 2)) :=
by
  sorry

end part_I_part_II_l256_256497


namespace tickets_spent_l256_256780

theorem tickets_spent (initial_tickets : ℕ) (tickets_left : ℕ) (tickets_spent : ℕ) 
  (h1 : initial_tickets = 11) (h2 : tickets_left = 8) : tickets_spent = 3 :=
by
  sorry

end tickets_spent_l256_256780


namespace area_of_triangle_ABF_l256_256484

theorem area_of_triangle_ABF :
  let C : Set (ℝ × ℝ) := {p | (p.1 ^ 2 / 4) + (p.2 ^ 2 / 3) = 1}
  let line : Set (ℝ × ℝ) := {p | p.1 - p.2 - 1 = 0}
  let F : ℝ × ℝ := (-1, 0)
  let AB := C ∩ line
  ∃ A B : ℝ × ℝ, A ∈ AB ∧ B ∈ AB ∧ A ≠ B ∧ 
  (1/2) * (2 : ℝ) * (12 * Real.sqrt (2 : ℝ) / 7) = (12 * Real.sqrt (2 : ℝ) / 7) :=
sorry

end area_of_triangle_ABF_l256_256484


namespace maximum_area_OABC_l256_256058

open Real

noncomputable def max_area_OABC (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : ℝ :=
  let B_x := 2 * cos θ
  let B_y := sqrt 3 * sin θ
  let C_y := sqrt 3 * sin θ
  sqrt 3 * sin θ * (1 - cos θ) + (sqrt 3 / 2) * sin (2 * θ)

theorem maximum_area_OABC : 
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ max_area_OABC θ ⟨lt_trans zero_lt_sqrt $ sqrt_pos_iff.mp (by norm_num), half_pos $ by norm_num⟩ = 9 / 4 := 
sorry

end maximum_area_OABC_l256_256058


namespace total_tree_planting_tasks_l256_256775

variable (a T : ℕ)

-- Define the conditions
def ninth_grade_task := T / 2
def eighth_grade_task := (T - ninth_grade_task) * (2 / 3)
def seventh_grade_planted := a

-- Prove that given these conditions, T = 6a
theorem total_tree_planting_tasks (h1 : ninth_grade_task = T / 2)
  (h2 : eighth_grade_task = (T - ninth_grade_task) * (2 / 3))
  (h3 : seventh_grade_planted = T / 6) :
  T = 6 * a :=
by
  sorry

end total_tree_planting_tasks_l256_256775


namespace magnitude_a_plus_2b_l256_256876

variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions
variables (a b : V)
variable (ha : ∥a∥ = 1)
variable (hb : ∥b∥ = 2)
variable (hab : a - b = (sqrt 2 : ℝ) • ![1/sqrt 2, sqrt 3/2])

-- Prove that |a + 2b| = sqrt 17
theorem magnitude_a_plus_2b : ∥a + 2 • b∥ = sqrt 17 :=
sorry

end magnitude_a_plus_2b_l256_256876


namespace smallest_five_digit_multiple_of_18_l256_256026

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 0 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 0 → n ≤ m :=
  sorry

end smallest_five_digit_multiple_of_18_l256_256026


namespace largest_product_not_less_than_993_squared_l256_256276

-- This order is introduced to refer to permutations of natural numbers
open Finset

noncomputable def max_product_ge_993_squared (seq : Fin N) : Prop :=
  ∃ (a : ℕ → ℕ) (h_perm : { i | i ∈ range (N + 1) } = { i | i ∈ seq}),
    ∃ (k : ℕ) (h_finset : k ∈ range (N + 1)), (N = 1985) ∧ (seq ≠ range (N + 1)) → max (a k * k) seq ≥ 993 ^ 2

theorem largest_product_not_less_than_993_squared :
  ∀ (a : ℕ → ℕ), 
  (∀ k, 1 ≤ k → k ≤ 1985 → Exists (a = k)) →
  (∀ (k : ℕ), k ∈ range (1985 + 1)) →
    max { a k * k | a k ∈ a } ≥ 993 ^ 2 :=
begin
  -- proof will be provided here
  sorry
end

end largest_product_not_less_than_993_squared_l256_256276


namespace find_a_l256_256466

-- Definitions of the points
def P : (ℝ × ℝ × ℝ) := (2, 0, 0)
def A : (ℝ × ℝ × ℝ) := (1, -3, 2)
def B : (ℝ × ℝ × ℝ) := (8, -1, 4)
def C (a : ℝ) : (ℝ × ℝ × ℝ) := (2 * a + 1, a + 1, 2)

-- Vector operations
def vector_sub (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2, v.3 - w.3)

-- Defining vectors from P to other points
def PC (a : ℝ) : ℝ × ℝ × ℝ := vector_sub (C a) P
def PA : ℝ × ℝ × ℝ := vector_sub A P
def PB : ℝ × ℝ × ℝ := vector_sub B P

-- The statement to prove
theorem find_a (a : ℕ) (λ μ : ℝ) :
  PC a = (λ • PA) + (μ • PB) → a = 5/2 :=
by
  sorry

end find_a_l256_256466


namespace consecutive_even_product_6digit_l256_256438

theorem consecutive_even_product_6digit :
  ∃ (a b c : ℕ), 
  (a % 2 = 0) ∧ (b = a + 2) ∧ (c = a + 4) ∧ 
  (Nat.digits 10 (a * b * c)).length = 6 ∧ 
  (Nat.digits 10 (a * b * c)).head! = 2 ∧ 
  (Nat.digits 10 (a * b * c)).getLast! = 2 ∧ 
  (a * b * c = 287232) :=
by
  sorry

end consecutive_even_product_6digit_l256_256438


namespace area_of_congruent_squares_l256_256668

-- Definitions based on conditions
def square (side_length : ℝ) (x y : ℝ) := 
  ∃ a b : ℝ, a = side_length ∧ b = side_length ∧ x = y

def midpoint (a b : Point) (m : Point) := 
  m = (a + b) / 2

-- Assumptions and the final statement
theorem area_of_congruent_squares (side_length : ℝ) (P : Point)
  (h1 : side_length = 20)
  (h2 : midpoint (Point.mk 0 0) (Point.mk side_length 0) P) :
  area_of_covered_region_by_two_squares = 600 := sorry

end area_of_congruent_squares_l256_256668


namespace measure_of_angle_BAC_is_45_l256_256761

open EuclideanGeometry

-- Definitions reflecting the problem conditions
variables (square rectangle: Quadrilateral) 
variables (C: Point) 
variables (A B: Point)
variables (O: Point) -- Center of the circle

-- Conditions
def inscribed (Q: Quadrilateral) : Prop := Circle Q.O
def shared_vertex (square rectangle : Quadrilateral) (P : Point) : Prop := 
  P ∈ vertices square ∧ P ∈ vertices rectangle
def angle_at_shared_vertex_is_135 (rectangle : Quadrilateral) (P : Point) : Prop := 
  Q.angle P = 135
def isosceles_triangle (A B C : Point) : Prop := 
  A ≠ B ∧ distance A B = distance B C
def sides_of_square_and_rectangle (A B C: Point) (square rectangle: Quadrilateral) : Prop := 
  side A B ∈ sides square ∧ side B C ∈ sides rectangle

-- Theorem statement to be proved
theorem measure_of_angle_BAC_is_45 
  (hv: shared_vertex square rectangle P)
  (hinscribed_square: inscribed square)
  (hinscribed_rectangle: inscribed rectangle)
  (hangle: angle_at_shared_vertex_is_135 rectangle P)
  (hiso: isosceles_triangle A B C)
  (hsides: sides_of_square_and_rectangle A B C square rectangle) 
  : angle BAC = 45 :=
sorry

end measure_of_angle_BAC_is_45_l256_256761


namespace scientific_notation_826M_l256_256569

theorem scientific_notation_826M : 826000000 = 8.26 * 10^8 :=
by
  sorry

end scientific_notation_826M_l256_256569


namespace part1_part2_l256_256866

theorem part1 (a : ℝ) (h_a : 0 < a) (h_max : ∃ x ∈ set.Icc 1 (real.exp 1), ∀ y ∈ set.Icc 1 (real.exp 1), f y ≤ f x ∧ f x = -4) :
  f = (fun x => real.log x - 4 * x) :=
by
  let f := fun x => real.log x - a * x
  have max_x : ∃ x ∈ set.Icc 1 (real.exp 1), ∀ y ∈ set.Icc 1 (real.exp 1), f y ≤ f x ∧ f x = -4 := h_max
  have expr_1 : ∀ x, f x = (fun x => real.log x - 4 * x) x :=
    sorry -- This needs to establish that f(x) = ln(x) - 4x based on the given max constraint.
  exact expr_1

theorem part2 (a : ℝ) (h_a_range : 0 < a) :
  (∀ x, deriv g x ≤ 0) ↔ a ≥ 1 / 4 :=
by
  let f := fun x => real.log x - a * x
  let g := fun x => f x + deriv f x
  have deriv_g : ∀ x, deriv g x = (1/(4 * x) - a/x + -1/x^2) := by
    sorry -- This is the calculation of the derivative g'(x) for examining its monotonicity.
  split
  case mp =>
    intro h
    have a_ge : a ≥ 1 / 4 := by
      sorry -- Prove that a ≥ 1/4 if g(x) is monotonic using the given derivative.
    exact a_ge
  case mpr =>
    intro h
    have g_noninc : ∀ x, deriv g x ≤ 0 := by
      sorry -- Show that g(x) is non-increasing for a ≥ 1 / 4.
    exact g_noninc

end part1_part2_l256_256866


namespace length_LN_eq_NA_l256_256196

theorem length_LN_eq_NA
  (ABC : Triangle)
  (A B C L K M N : Point)
  (H1 : angle_bisector A C (Line.mk B C) = Line.mk A L)
  (H2 : angle_bisector B A (Line.mk C A) = Line.mk B K)
  (H3 : L ∈ segment B C)
  (H4 : K ∈ segment C A)
  (H5 : is_perpendicular_bisector (segment B K) (Line.mk A L) M)
  (H6 : N ∈ line.mk B K)
  (H7 : parallel (Line.mk L N) (Line.mk M K))
  : dist L N = dist N A :=
by
  sorry

end length_LN_eq_NA_l256_256196


namespace log_eq_solution_l256_256996

theorem log_eq_solution (x : ℝ) (h : Real.log 8 / Real.log x = Real.log 5 / Real.log 125) : x = 512 := by
  sorry

end log_eq_solution_l256_256996


namespace fewest_coach_handshakes_l256_256779

theorem fewest_coach_handshakes (n m1 m2 : ℕ) 
  (handshakes_total : (n * (n - 1)) / 2 + m1 + m2 = 465) 
  (m1_m2_eq_n : m1 + m2 = n) : 
  n * (n - 1) / 2 = 465 → m1 + m2 = 0 :=
by 
  sorry

end fewest_coach_handshakes_l256_256779


namespace pieces_on_black_squares_even_l256_256219

def is_piece_on (board : ℕ × ℕ → Prop) (i j : ℕ) : Prop := board (i, j)
def is_black_square (i j : ℕ) : Prop := (i + j) % 2 = 1

def odd_number_of_pieces_in_row (board : ℕ × ℕ → Prop) (i : ℕ) : Prop :=
  ∃ k, (k % 2 = 1) ∧ (k = (Finset.filter (λ j, is_piece_on board i j) (Finset.range 8)).card)

def odd_number_of_pieces_in_col (board : ℕ × ℕ → Prop) (j : ℕ) : Prop :=
  ∃ k, (k % 2 = 1) ∧ (k = (Finset.filter (λ i, is_piece_on board i j) (Finset.range 8)).card)

theorem pieces_on_black_squares_even (board : ℕ × ℕ → Prop)
  (rows_odd : ∀ i, odd_number_of_pieces_in_row board i)
  (cols_odd : ∀ j, odd_number_of_pieces_in_col board j) :
  ∃ k, k % 2 = 0 ∧ (k = (Finset.filter (λ (ij : ℕ × ℕ), (is_piece_on board ij.1 ij.2) 
    ∧ (is_black_square ij.1 ij.2)) (Finset.pi_finset Finset.univ Finset.univ)).card) :=
sorry

end pieces_on_black_squares_even_l256_256219


namespace exists_x_such_that_6_in_A_l256_256501

theorem exists_x_such_that_6_in_A :
  ∃ x : ℝ, 6 ∈ ({2, x, x^2 + x} : set ℝ) :=
sorry

end exists_x_such_that_6_in_A_l256_256501


namespace eq_total_area_of_sum_sqrt_areas_l256_256824

-- Define the triangle areas S and S_i
variables {S S1 S2 S3 : ℝ}

-- The main theorem to prove
theorem eq_total_area_of_sum_sqrt_areas (hS1 : S1 > 0) (hS2 : S2 > 0) (hS3 : S3 > 0) :
  S = (real.sqrt S1 + real.sqrt S2 + real.sqrt S3)^2 :=
sorry

end eq_total_area_of_sum_sqrt_areas_l256_256824


namespace profit_percentage_is_18_l256_256783

-- Define the conditions
def cost_per_kg_first_brand : ℝ := 200
def quantity_first_brand : ℝ := 2
def cost_per_kg_second_brand : ℝ := 116.67
def quantity_second_brand : ℝ := 3
def selling_price_per_kg : ℝ := 177

-- Calculate the total cost and total quantity of the mixture
def total_cost_mixture : ℝ := (quantity_first_brand * cost_per_kg_first_brand) + (quantity_second_brand * cost_per_kg_second_brand)
def total_quantity_mixture : ℝ := quantity_first_brand + quantity_second_brand

-- Calculate the cost price per kg of the mixture
def cost_price_per_kg_mixture : ℝ := total_cost_mixture / total_quantity_mixture

-- Calculate the profit per kg and the profit percentage
def profit_per_kg : ℝ := selling_price_per_kg - cost_price_per_kg_mixture
def profit_percentage : ℝ := (profit_per_kg / cost_price_per_kg_mixture) * 100

-- Statement of the problem to be proved
theorem profit_percentage_is_18 : profit_percentage = 18 := by
  sorry

end profit_percentage_is_18_l256_256783


namespace area_triangle_APQ_l256_256633

/-
  ABC is an equilateral triangle with side length 10.
  Points P and Q are on sides AB and AC respectively.
  Segment PQ is tangent to the incircle of triangle ABC and has length 4.
  Let AP = x and AQ = y, such that x + y = 6 and x^2 + y^2 - xy = 16.
  Prove that the area of triangle APQ is 5 * sqrt(3) / 3.
-/

theorem area_triangle_APQ :
  ∀ (x y : ℝ),
  x + y = 6 ∧ x^2 + y^2 - x * y = 16 → 
  (∃ (S : ℝ), S = (1 / 2) * x * y * (sqrt 3 / 2) ∧ S = 5 * (sqrt 3) / 3) :=
by 
  intro x y,
  intro h,
  sorry

end area_triangle_APQ_l256_256633


namespace log_proof_l256_256125

theorem log_proof (x : ℝ) (h : log 3 (x + 1) = 4) : log 9 x = 2 :=
sorry

end log_proof_l256_256125


namespace congruent_triangles_l256_256902

-- Define points A, B, C
variable (A B C : Point)
-- Define midpoints A', B', C'
variable (A' B' C' : Point)
-- Define that A', B', C' are midpoints
axiom midpoint_A' : midpoint A' B C
axiom midpoint_B' : midpoint B' C A
axiom midpoint_C' : midpoint C' A B

-- Define circumcenters O1, O2, O3
variable (O1 O2 O3 I1 I2 I3 : Point)
axiom circumcenter_O1 : circumcenter O1 A B' C'
axiom circumcenter_O2 : circumcenter O2 A' B C'
axiom circumcenter_O3 : circumcenter O3 A' B' C

-- Define incenters I1, I2, I3
axiom incenter_I1 : incenter I1 A B' C'
axiom incenter_I2 : incenter I2 A' B C'
axiom incenter_I3 : incenter I3 A' B' C

-- The theorem we want to prove
theorem congruent_triangles : congruent (triangle O1 O2 O3) (triangle I1 I2 I3) := 
sorry

end congruent_triangles_l256_256902


namespace lowest_temperature_l256_256254

theorem lowest_temperature 
  (temperatures : list ℤ)
  (h_length : temperatures.length = 5)
  (h_sum : temperatures.sum = 300)
  (h_range : temperatures.maximum - temperatures.minimum = 50) :
  temperatures.minimum = 20 :=
by
  sorry

end lowest_temperature_l256_256254


namespace math_proof_problem_l256_256495

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.log x - x^2 + x
def g (m x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x - 1

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := (-(2 * x + 1) * (x - 1)) / x

-- Define the function h
def h (m x : ℝ) : ℝ := f x - g m x

-- Define the derivative of h
noncomputable def h' (m x : ℝ) : ℝ := (1 / x - 2 * m * x + 1 - 2 * m)

-- Define the function φ
noncomputable def φ (m : ℝ) : ℝ := (1 / (4 * m)) - Real.log (2 * m)

-- The Lean statement of the math proof problem
theorem math_proof_problem (x m : ℝ) (hx: x > 0) :
  ((0 < x ∧ x < 1) → f' x > 0) ∧ 
  ((x > 1) → f' x < 0) ∧ 
  ((h' m x > 0 → 0 < x ∧ x < 1 / (2 * m)) ∧ 
  (h' m x < 0 → x > 1 / (2 * m))) ∧ 
  (φ 1 > 0 ∧ (m ≥ 1) → φ m < 0) := 
  sorry

end math_proof_problem_l256_256495


namespace AM_LT_BM_plus_CM_l256_256934

theorem AM_LT_BM_plus_CM
  (A B C M O : Type)
  [EquilateralTriangle A B C O]
  [PointOnCircleArc M A C B]
  [Lines AM BM CM] : AM < BM + CM := 
sorry

end AM_LT_BM_plus_CM_l256_256934


namespace john_bought_two_dozens_l256_256925

theorem john_bought_two_dozens (x : ℕ) (h₁ : 21 + 3 = x * 12) : x = 2 :=
by {
    -- Placeholder for skipping the proof since it's not required.
    sorry
}

end john_bought_two_dozens_l256_256925


namespace exists_smallest_n_cube_sum_l256_256024

theorem exists_smallest_n_cube_sum (n : ℕ) (x : ℕ → ℤ) :
  (∀ i ∈ finset.range n, ∃ (x_i : ℤ), x i = x_i) ∧ 
  (finset.range n).sum (λ i, (x i) ^ 3) = 2002 ^ 2002 ∧
  (∀ m, m < n → ¬(∃ y : ℕ → ℤ, (∀ i ∈ finset.range m, ∃ (y_i : ℤ), y i = y_i) ∧ (finset.range m).sum (λ i, (y i) ^ 3) = 2002 ^ 2002))
  ↔ n = 4 := by
  sorry

end exists_smallest_n_cube_sum_l256_256024


namespace height_of_rectangular_block_l256_256282

variable (V A h : ℕ)

theorem height_of_rectangular_block :
  V = 120 ∧ A = 24 ∧ V = A * h → h = 5 :=
by
  sorry

end height_of_rectangular_block_l256_256282


namespace n_exists_and_unique_l256_256117

theorem n_exists_and_unique:
  ∀ a b : ℤ,
  (a ≡ 24 [MOD 50]) →
  (b ≡ 95 [MOD 50]) →
  ∃ n : ℤ, 
  (150 ≤ n ∧ n ≤ 200) ∧
  (a - b ≡ n [MOD 50]) ∧
  (n ≡ 3 [MOD 4]) ∧
  (n = 179) :=
by
  sorry

end n_exists_and_unique_l256_256117


namespace campaign_funds_perc_raised_by_friends_l256_256249

variables (T F S E : ℝ) 

theorem campaign_funds_perc_raised_by_friends :
  T = 10000 ∧
  S = 4200 ∧
  E = 0.30 * (T - F) ∧
  T = F + E + S → 
  (F / T) * 100 = 40 :=
begin
  intros h,
  cases h with hT h1,
  cases h1 with hS h2,
  cases h2 with hE hT',
  sorry
end

end campaign_funds_perc_raised_by_friends_l256_256249


namespace find_a_2018_l256_256856

-- Define the conditions
-- S_9 is the sum of the first 9 terms of the arithmetic sequence {a_n}
def S_9 (a : Nat → Int) : Int := (∑ k in finset.range 9, a k) -- sum of first 9 terms
def a_10 (a : Nat → Int) : Int := a 9 -- the 10th term a_10 since Lean indices from 0

-- Assume the following values based on the given conditions:
axiom sum_first_nine (a : Nat → Int) : S_9 a = 27
axiom tenth_term (a : Nat → Int) : a_10 a = 8

-- Define the arithmetic sequence
-- Arithmetic sequence: a_n = a_1 + (n - 1) * d
def arithmetic_sequence (a_1 d : Int) (n : Nat) : Int := a_1 + Int.ofNat n * d

-- Define the statement to be proved
theorem find_a_2018 : ∃ (a : Nat → Int), (S_9 a = 27) ∧ (a_10 a = 8) ∧ (a 2017 = 2016) := by
  -- Here, S_9 a = 27 and a_10 a = 8 by assumptions
  -- a_{2018} is defined as a 2017 in Lean (due to zero-based indexing)
  existsi (arithmetic_sequence (-1) 1) -- using the derived a_1 = -1 and d = 1
  simp [S_9, a_10, arithmetic_sequence] -- simplification should resolve conditions
  exact ⟨sum_first_nine (arithmetic_sequence (-1) 1), tenth_term (arithmetic_sequence (-1) 1), by rfl⟩


end find_a_2018_l256_256856


namespace coefficient_m5_n5_in_expansion_l256_256304

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Goal: prove the coefficient of m^5 n^5 in the expansion of (m+n)^{10} is 252
theorem coefficient_m5_n5_in_expansion : binomial 10 5 = 252 :=
by
  sorry

end coefficient_m5_n5_in_expansion_l256_256304


namespace car_plate_number_count_l256_256528

/-- In a certain city, the car plate number consists of a six-digit number formed from the digits 0 to 9. 
Digits can be reused, and 0 can be the first digit. 
We prove that the number of car plate numbers such that the sum of the digits is a multiple of 9 
and the number contains at least three 9s is 1762. -/
theorem car_plate_number_count :
  let car_plate := { n : Fin 1000000 // 0 ≤ n.val ∧ n.val < 1000000 },
      valid_plate (n : Fin 1000000) : Prop :=
        let digits := (List.range 6).map (λ i => (n.val / 10^i % 10)) in
        let sum_d := digits.sum in
        let count_9s := digits.count 9 in
        sum_d % 9 = 0 ∧ count_9s ≥ 3 in
  (Finset.filter valid_plate (Finset.univ : Finset (Fin 1000000))).card = 1762 :=
by
  sorry

end car_plate_number_count_l256_256528


namespace fraction_simplification_l256_256422

theorem fraction_simplification (a b d : ℝ) (h : a^2 + d^2 - b^2 + 2 * a * d ≠ 0) :
  (a^2 + b^2 + d^2 + 2 * b * d) / (a^2 + d^2 - b^2 + 2 * a * d) = (a^2 + (b + d)^2) / ((a + d)^2 + a^2 - b^2) :=
sorry

end fraction_simplification_l256_256422


namespace similar_triangles_mapping_to_each_other_l256_256655

-- Define the triangles and necessary conditions
variables (A B C A' B' C' : Type) [plane_geometry A B C] [plane_geometry A' B' C']

def similar {A B C A' B' C'} {P Q R P' Q' R' : Type} [plane_geometry P Q R] [plane_geometry P' Q' R'] : Prop :=
∃ (f : P → P'), is_similar f ∧ same_orientation f

def not_congruent {A B C A' B' C'} {P Q R P' Q' R' : Type} [plane_geometry P Q R] [plane_geometry P' Q' R'] (f : P → P') : Prop :=
¬ is_congruent f

def not_homothetic {A B C A' B' C'} {P Q R P' Q' R' : Type} [plane_geometry P Q R] [plane_geometry P' Q' R'] (f : P → P') : Prop :=
¬ is_homothetic f

theorem similar_triangles_mapping_to_each_other (tri1 tri2 : triangle)
    (h1 : similar tri1 tri2) 
    (h2 : not_congruent tri1 tri2)
    (h3 : not_homothetic tri1 tri2) :
  ∃ (S : Type) (f : triangle → triangle), (is_rotation_with_center S f) ∧ (is_homothety_with_center S f)
:= sorry

end similar_triangles_mapping_to_each_other_l256_256655


namespace triangle_ABC_right_triangle_l256_256557

theorem triangle_ABC_right_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h_angle_sum : A + B + C = π)
  (h_cos_half_B : cos (B / 2) ^ 2 = (a + c) / (2 * c)) :
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 :=
by sorry

end triangle_ABC_right_triangle_l256_256557


namespace necessary_and_sufficient_conditions_convex_polyhedral_cone_intersection_l256_256299

/-- Necessary and sufficient conditions for a convex polyhedral angle to contain a cone of revolution -/
theorem necessary_and_sufficient_conditions_convex_polyhedral_cone_intersection
  (S A B C D : Point)
  (α' α'' β' β'' γ' γ'' δ' δ'' : ℝ)
  (angle_SAB angle_SCD angle_SBC angle_SDA : ℝ)
  (flat_angle_condition : angle_SAB + angle_SCD = angle_SBC + angle_SDA) :
  -- Necessary condition:
  (α'' = β' ∧ β'' = γ' ∧ γ'' = δ' ∧ δ'' = α') ∧
  -- Sufficient condition:
  (∃ (l : Line),
    -- l is an axis passing through S and equidistant from four faces 
    (∃ (points_lie_on_l : ∀ (P : Point), P ∈ l ↔ P = S ∨ P = A ∨ P = B ∨ P = C ∨ P = D),
        (∃ (axis_conical : l.is_axis_of_revolution S A B C D))) :=
sorry

end necessary_and_sufficient_conditions_convex_polyhedral_cone_intersection_l256_256299


namespace negation_of_proposition_l256_256686

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), (a = b → a^2 = a * b)) = ∀ (a b : ℝ), (a ≠ b → a^2 ≠ a * b) :=
sorry

end negation_of_proposition_l256_256686


namespace panda_bamboo_consumption_l256_256650

theorem panda_bamboo_consumption (x : ℝ) (h : 0.40 * x = 16) : x = 40 :=
  sorry

end panda_bamboo_consumption_l256_256650


namespace one_of_inequalities_true_l256_256199

theorem one_of_inequalities_true (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (| (a + real.sqrt (a^2 + 2 * b^2)) / (2 * b) | < 1)
  ∨ (| (a - real.sqrt (a^2 + 2 * b^2)) / (2 * b) | < 1) :=
sorry

end one_of_inequalities_true_l256_256199


namespace shara_shells_on_fourth_day_l256_256993

-- Given conditions
def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days := 3
def total_shells := 41

-- Prove that Shara found 6 shells on the fourth day
theorem shara_shells_on_fourth_day : 
  let shells_in_first_three_days := shells_per_day * days in
  let shells_after_three_days := initial_shells + shells_in_first_three_days in
  let shells_on_fourth_day := total_shells - shells_after_three_days in
  shells_on_fourth_day = 6 := 
by 
  sorry

end shara_shells_on_fourth_day_l256_256993


namespace sum_of_possible_values_on_olgas_card_l256_256612

noncomputable def possible_values_olgas_card (x : ℝ) : ℝ :=
  if 90 < x ∧ x < 180 then tan x else 0

theorem sum_of_possible_values_on_olgas_card (x : ℝ) (h1 : 90 < x ∧ x < 180) (h2 : tan x = 1) : ∑ v in {v | v = possible_values_olgas_card x}, v = 1 :=
by
  sorry

end sum_of_possible_values_on_olgas_card_l256_256612


namespace shaded_region_area_l256_256916

noncomputable def side_length := 1 -- Length of each side of the squares, in cm.

-- Conditions
def top_square_center_above_edge : Prop := 
  ∀ square1 square2 square3 : ℝ, square3 = (square1 + square2) / 2

-- Question: Area of the shaded region
def area_of_shaded_region := 1 -- area in cm^2

-- Lean 4 Statement
theorem shaded_region_area :
  top_square_center_above_edge → area_of_shaded_region = 1 := 
by
  sorry

end shaded_region_area_l256_256916


namespace quadratic_complex_root_l256_256890

open Complex

theorem quadratic_complex_root (b : ℝ) :
  (root_of_real_quadratic (3 + I) (x^2 - 6 * x + b = 0)) ↔ b = 10 := 
by 
  sorry

end quadratic_complex_root_l256_256890


namespace ellipse_equation_valid_l256_256052

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (eccentricity : a^2 * 8 / b^2 = 9) : Prop :=
  ∀ (A1 A2 B : ℝ × ℝ), 
    (A1 = (-3, 0)) ∧ 
    (A2 = (3, 0)) ∧ 
    (B = (0, 2 * Real.sqrt 2)) ∧ 
    ((B.1 - A1.1) * (B.1 - A2.1) + (B.2 - A1.2) * (B.2 - A2.2) = -1) →
    ( ∃ m : ℝ, m ≠ 0 ∧ (m^2 = 1) ∧ 
    (a^2 = 9 * m^2) ∧ (b^2 = 8 * m^2) ∧ 
    (C : ℝ × ℝ → Prop, ∀ x y, C (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1) → 
    ( ∀ x y, C (x, y) ↔ x^2 / 9 + y^2 / 8 = 1) )

theorem ellipse_equation_valid :
ellipse_equation 3 (2 * Real.sqrt 2) (by linarith) (by linarith) 
(3^2 * 8 / (2 * Real.sqrt 2)^2 = 9) := 
sorry

end ellipse_equation_valid_l256_256052


namespace sum_of_series_l256_256964

open Real

def v0 : ℝ × ℝ := (2, 1)
def w0 : ℝ × ℝ := (1, -1)

def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_ab := a.1 * b.1 + a.2 * b.2
  let norm_b_sq := b.1 * b.1 + b.2 * b.2
  (dot_ab / norm_b_sq * b.1, dot_ab / norm_b_sq * b.2)

def v (n : ℕ) : ℝ × ℝ :=
  if n = 0 then v0
  else (1/2) * (proj (w (n - 1)) v0)

def w (n : ℕ) : ℝ × ℝ :=
  if n = 0 then w0
  else 2 * (proj (v n) w0)

theorem sum_of_series :
  let series_sum := finset.sum (finset.range 1000) (λ n, (v n) + (w n))
  series_sum = (3, 0) :=
sorry

end sum_of_series_l256_256964


namespace good_walker_vs_bad_walker_l256_256549

theorem good_walker_vs_bad_walker :
  ∀ (x y : ℕ), (x - y = 100) ∧ (x = (100 * y / 60)) :=
by
  intro x y
  split
  . sorry
  . sorry

end good_walker_vs_bad_walker_l256_256549


namespace fraction_of_paint_first_week_l256_256574

-- Definitions based on conditions
def total_paint := 360
def fraction_first_week (f : ℚ) : ℚ := f * total_paint
def paint_remaining_first_week (f : ℚ) : ℚ := total_paint - fraction_first_week f
def fraction_second_week (f : ℚ) : ℚ := (1 / 5) * paint_remaining_first_week f
def total_paint_used (f : ℚ) : ℚ := fraction_first_week f + fraction_second_week f
def total_paint_used_value := 104

-- Proof problem statement
theorem fraction_of_paint_first_week (f : ℚ) (h : total_paint_used f = total_paint_used_value) : f = 1 / 9 := 
sorry

end fraction_of_paint_first_week_l256_256574


namespace not_divide_g_count_30_l256_256193

-- Define the proper positive divisors function
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range (n - 1)).filter (λ d, d > 0 ∧ n % d = 0)

-- Define the product of proper divisors function
def g (n : ℕ) : ℕ :=
  proper_divisors n |>.prod

-- Define the main theorem
theorem not_divide_g_count_30 : 
  (Finset.range 99).filter (λ n, 2 ≤ n + 1 ∧ n + 1 ≤ 100 ∧ ¬(n + 1) ∣ g (n + 1)).card = 30 := 
  by
  sorry

end not_divide_g_count_30_l256_256193


namespace prob_sum_seven_l256_256712

/-- Define the event that two dice are tossed and the sum is exactly 7. -/
def event_sum_seven : set (ℕ × ℕ) := {x | x.1 + x.2 = 7}

/-- Calculate the probability that the sum is exactly 7 when two dice are tossed. -/
theorem prob_sum_seven (faces : finset ℕ)
  (h_faces : faces = {1, 2, 3, 4, 5, 6}) :
  let outcomes := (faces.product faces) in
  let favorable_outcomes := (outcomes.filter (λ x, x.1 + x.2 = 7)) in
  (favorable_outcomes.card : ℚ) / outcomes.card = 1 / 6 := 
sorry

end prob_sum_seven_l256_256712


namespace identical_triangles_exist_after_any_number_of_steps_l256_256561

theorem identical_triangles_exist_after_any_number_of_steps
  (initial_triangles : ℕ)
  (h_initial : initial_triangles = 4)
  (cut_triangle : ∀ n : ℕ, n → n + 1)
  (right_angled_triangle : ∀ n : ℕ, n + 1 → n) :
  ∃ n : ℕ, ∃ m : ℕ, n ≠ m ∧ right_angled_triangle n = right_angled_triangle m :=
by
  sorry

end identical_triangles_exist_after_any_number_of_steps_l256_256561


namespace solution_set_of_xf_l256_256483

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

theorem solution_set_of_xf (f : ℝ → ℝ) (hf_odd : is_odd_function f) (hf_one : f 1 = 0)
    (h_derivative : ∀ x > 0, (x * (deriv f x) - f x) / (x^2) > 0) :
    {x : ℝ | x * f x > 0} = {x : ℝ | x < -1 ∨ x > 1} :=
by
  sorry

end solution_set_of_xf_l256_256483


namespace area_of_triangle_APQ_proof_l256_256641

noncomputable def area_of_triangle_APQ : ℝ :=
  let ABC := (A B C : Point) := {a : ℝ | a ≤ 10}
  let P := (P : Point) := {p : ℝ | 0 ≤ p ∧ p ≤ 10}
  let Q := (Q : Point) := {q : ℝ | 0 ≤ q ∧ q ≤ 10}
  let pq := (pq : ℝ) := 4
  let x := (x : ℝ) := AP
  let y := (y : ℝ) := AQ
  let XY := (S : ℝ) := xy
  XY = {S : ℝ | (x, y) ∈ ABC × P × Q, pq = 4, S = PQAP.area}
  let area_APQ := (S : ℝ) := 5 / sqrt 3

theorem area_of_triangle_APQ_proof :
  area_of_triangle_APQ = 5 / sqrt 3 := by sorry

end area_of_triangle_APQ_proof_l256_256641


namespace count_3_digit_numbers_satisfying_condition_l256_256884

def is_3_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def satisfies_condition (n : ℕ) : Prop :=
  let u := n % 10 in
  let t := (n / 10) % 10 in
  u >= 2 * t

theorem count_3_digit_numbers_satisfying_condition :
  (Finset.card (Finset.filter (λ n, is_3_digit_number n ∧ satisfies_condition n) (Finset.range 1000))) = 270 :=
by
  sorry

end count_3_digit_numbers_satisfying_condition_l256_256884


namespace students_passing_course_l256_256899

theorem students_passing_course :
  let students_three_years_ago := 200
  let increase_factor := 1.5
  let students_two_years_ago := students_three_years_ago * increase_factor
  let students_last_year := students_two_years_ago * increase_factor
  let students_this_year := students_last_year * increase_factor
  students_this_year = 675 :=
by
  sorry

end students_passing_course_l256_256899


namespace coefficient_m5_n5_in_expansion_l256_256305

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Goal: prove the coefficient of m^5 n^5 in the expansion of (m+n)^{10} is 252
theorem coefficient_m5_n5_in_expansion : binomial 10 5 = 252 :=
by
  sorry

end coefficient_m5_n5_in_expansion_l256_256305


namespace continuous_functional_equation_l256_256428

theorem continuous_functional_equation (f : ℝ → ℝ) (h_cont : continuous f)
  (h_eq : ∀ x y : ℝ, f(x + y) = f(x) * f(y)) :
  (f = (fun x => 0)) ∨ (∃ c : ℝ, f = (fun x => Real.exp(c * x))) :=
  sorry

end continuous_functional_equation_l256_256428


namespace find_CD_l256_256559

noncomputable def cd_length (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : ℝ :=
  let A : ℝ := a^2 / b in
  b - A

theorem find_CD {a b : ℝ} (A B C D O : ℝ) (H1 : A ≠ B) (H2 : A ≠ C) (H3 : B ≠ C) (H4 : B ≠ D) (H5 : C ≠ D) (H6 : A ≠ O) (H7 : B ≠ O) (H8 : C ≠ O):
  AB = a → AC = b → O is the center of the circumcircle → (BD ⊥ AO) → (BD intersects AC at D) → CD = b - a^2 / b := 
by
  intros h1 h2 h3 h4 h5
  sorry

end find_CD_l256_256559


namespace tables_capacity_l256_256749

theorem tables_capacity (invited attended : ℕ) (didn't_show_up : ℕ) (tables : ℕ) (capacity : ℕ) 
    (h1 : invited = 24) (h2 : didn't_show_up = 10) (h3 : attended = invited - didn't_show_up) 
    (h4 : attended = 14) (h5 : tables = 2) : capacity = attended / tables :=
by {
  -- Proof goes here
  sorry
}

end tables_capacity_l256_256749


namespace students_passed_this_year_l256_256898

theorem students_passed_this_year
  (initial_students : ℕ)
  (annual_increase_rate : ℝ)
  (years_lapsed : ℕ)
  (current_students : ℕ)
  (h_initial : initial_students = 200)
  (h_rate : annual_increase_rate = 1.5)
  (h_years : years_lapsed = 3)
  (h_calc : current_students = (λ n, initial_students * (annual_increase_rate ^ n)) years_lapsed) :
  current_students = 675 :=
begin
  sorry
end

end students_passed_this_year_l256_256898


namespace brocard_inequalities_l256_256654

theorem brocard_inequalities (α β γ φ: ℝ) (h1: φ > 0) (h2: φ < π / 6)
  (h3: α > 0) (h4: β > 0) (h5: γ > 0) (h6: α + β + γ = π) : 
  (φ^3 ≤ (α - φ) * (β - φ) * (γ - φ)) ∧ (8 * φ^3 ≤ α * β * γ) := 
by 
  sorry

end brocard_inequalities_l256_256654


namespace time_spent_on_aerobics_l256_256930

theorem time_spent_on_aerobics (A W : ℝ) 
  (h1 : A + W = 250) 
  (h2 : A / W = 3 / 2) : 
  A = 150 := 
sorry

end time_spent_on_aerobics_l256_256930


namespace probability_in_interval_l256_256072

noncomputable def normal := distribution.normal 3 sigma_sq

theorem probability_in_interval (X : ℝ → Prop) 
    (h1 : X follows normal 3 σ^2)
    (h2 : ∫ (x in IIQ X 5), pdf normal x = 0.8) : 
    ∫ (x in IIO 1 3), pdf normal x = 0.3 :=
sorry

end probability_in_interval_l256_256072


namespace num_values_of_n_l256_256180

-- Definitions of proper positive integer divisors and function g(n)
def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d > 1 ∧ d < n ∧ n % d = 0) (List.range (n + 1))

def g (n : ℕ) : ℕ :=
  List.prod (proper_divisors n)

-- Condition: 2 ≤ n ≤ 100 and n does not divide g(n)
def n_does_not_divide_g (n : ℕ) : Prop :=
  2 ≤ n ∧ n ≤ 100 ∧ ¬ (n ∣ g n)

-- Main theorem statement
theorem num_values_of_n : 
  (Finset.card (Finset.filter n_does_not_divide_g (Finset.range 101))) = 31 :=
by
  sorry

end num_values_of_n_l256_256180


namespace cake_pieces_per_sister_l256_256032

theorem cake_pieces_per_sister (total_pieces : ℕ) (percentage_eaten : ℕ) (sisters : ℕ)
  (h1 : total_pieces = 240) (h2 : percentage_eaten = 60) (h3 : sisters = 3) :
  (total_pieces * (1 - percentage_eaten / 100)) / sisters = 32 :=
by
  sorry

end cake_pieces_per_sister_l256_256032


namespace exists_three_members_playing_all_games_l256_256904

open Classical

section ClubGame

variables (n : ℕ) (members : Fin (3 * n + 1))

-- Conditions
variable (plays_tennis : members → members → Prop)
variable (plays_chess : members → members → Prop)
variable (plays_table_tennis : members → members → Prop)

variable (tennis_same : ∀ m : members, (univ.filter (plays_tennis m)).card = n)
variable (chess_same : ∀ m : members, (univ.filter (plays_chess m)).card = n)
variable (table_tennis_same : ∀ m : members, (univ.filter (plays_table_tennis m)).card = n)

-- Problem statement
theorem exists_three_members_playing_all_games :
  ∃ (a b c : members),
    plays_tennis a b ∧ plays_tennis b c ∧ plays_tennis a c ∧
    plays_chess a b ∧ plays_chess b c ∧ plays_chess a c ∧
    plays_table_tennis a b ∧ plays_table_tennis b c ∧ plays_table_tennis a c := sorry

end ClubGame

end exists_three_members_playing_all_games_l256_256904


namespace sum_of_solutions_mod_congruence_l256_256723

theorem sum_of_solutions_mod_congruence :
  (∑ x in Finset.filter (λ x, 1 ≤ x ∧ x ≤ 30) (Finset.Icc 1 30), 
    if 15 * (5 * x - 3) % 12 = 30 % 12 then x else 0) = 120 :=
by
  sorry

end sum_of_solutions_mod_congruence_l256_256723


namespace only_solution_l256_256814

noncomputable def real_solution : set (ℝ × ℝ) :=
  { (x, y) | (4^(-x) + 27^(-y) = 5 / 6) ∧
            (log 27 y - log 4 x ≥ 1 / 6) ∧
            (27^y - 4^x ≤ 1) }

theorem only_solution :
  real_solution = { (1 / 2, 1 / 3) } :=
by {
  sorry
}

end only_solution_l256_256814


namespace minimum_n_of_colored_balls_l256_256933

theorem minimum_n_of_colored_balls (n : ℕ) (h1 : n ≥ 3)
  (h2 : (n * (n + 1)) / 2 % 10 = 0) : n = 24 :=
sorry

end minimum_n_of_colored_balls_l256_256933


namespace value_of_a_value_of_cos_C_value_of_sin_2C_minus_pi_over_6_l256_256158

-- Given conditions
variables {A B C : ℝ} {a b c : ℝ}
axiom sin_ratio : sin A : sin B : sin C = 2 : 1 : sqrt 2
axiom b_value : b = sqrt 2

-- Proof problems
theorem value_of_a : a = 2 * sqrt 2 :=
by
  have h := sin_ratio,
  sorry

theorem value_of_cos_C : cos C = 3 / 4 :=
by
  have h := sin_ratio,
  sorry

theorem value_of_sin_2C_minus_pi_over_6 : sin (2 * C - π / 6) = (3 * sqrt 21 - 1) / 16 :=
by
  have h := sin_ratio,
  sorry

end value_of_a_value_of_cos_C_value_of_sin_2C_minus_pi_over_6_l256_256158


namespace town_population_l256_256975

/-- 
  Considering a town population that follows a sequence of mathematical properties:
  1. The original population is a triangular number.
  2. After an increase of 121, the population becomes a perfect square.
  3. After an additional increase of 144, the population also becomes a perfect square.
  Prove that the original population equals 2280.
-/
theorem town_population (n a b : ℕ)
  (h1 : T n = (n * (n + 1)) / 2)
  (h2 : T n + 121 = a ^ 2)
  (h3 : T n + 265 = b ^ 2) : T n = 2280 := sorry

-- Defining the triangular number function for reference
def T (n : ℕ) : ℕ := (n * (n + 1)) / 2

end town_population_l256_256975


namespace f_eq_n_l256_256589

def is_strictly_increasing_sequence (f : ℕ → ℕ) : Prop :=
∀ n m : ℕ, n < m → f(n) < f(m)

def takes_positive_integer_values (f : ℕ → ℕ) : Prop :=
∀ n : ℕ, f(n) > 0

def functional_property (f : ℕ → ℕ) : Prop :=
∀ m n : ℕ, Nat.coprime m n → f(m * n) = f(m) * f(n)

theorem f_eq_n (f : ℕ → ℕ) :
  is_strictly_increasing_sequence f →
  takes_positive_integer_values f →
  f 2 = 2 →
  functional_property f →
  ∀ n : ℕ, f(n) = n :=
by
  sorry

end f_eq_n_l256_256589


namespace eccentricity_of_ellipse_ratio_of_H_coordinates_l256_256468
-- Import the necessary math library

-- Conditions for Lean
variables (a b c : ℝ) (h : a > b > 0) (k : c > 0)

-- Define the ellipse
def ellipse (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

-- Coordinates for foci
def F1 := (-c, 0)
def F2 := (c, 0)

-- Ellipse endpoints of short axis
def A := (0, b)
def C := (0, -b)

-- Point E
def E := (3*c, 0)

-- Prove the eccentricity of the ellipse
theorem eccentricity_of_ellipse :
  let e := c / a in e = sqrt(3) / 3 :=
sorry

-- Coordinates for line F2B intersection point B
def B := (3*c / 2, b /2)

-- Prove the ratio of coordinates for point H
theorem ratio_of_H_coordinates :
  ∀ (m n : ℝ), (m ≠ 0) → 
  let H := (m, n) in 
  let circ_center := (c / 2, 0) in 
  let circumcircle := (x y : ℝ) := (x - circ_center.1)^2 + y^2 = (3 * c / 2)^2 in
  let line_F2B := (x y : ℝ) := y = sqrt(2) * (x - c) in
  circumcircle H.1 H.2 → line_F2B H.1 H.2 →
  n / m = (2 * sqrt(2)) / 5 ∨ n / m = -(2 * sqrt(2) / 5) :=
sorry

end eccentricity_of_ellipse_ratio_of_H_coordinates_l256_256468


namespace solution_set_of_inequality_l256_256694

theorem solution_set_of_inequality :
  { x : ℝ | |1 - 2 * x| < 3 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_of_inequality_l256_256694


namespace range_of_a_l256_256083

noncomputable def f (a x : ℝ) : ℝ := x * Real.exp x + (1 / 2) * a * x^2 + a * x

theorem range_of_a (a : ℝ) : 
    (∀ x : ℝ, 2 * Real.exp (f a x) + Real.exp 1 + 2 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
sorry

end range_of_a_l256_256083


namespace range_of_c_l256_256743

theorem range_of_c (c : ℝ) : (0 < c ∧ c < 1) ∨ (c > 2) ↔
  (function.decreasing_on (λ x : ℝ, c^x) set.univ ∨
   (∀ x ∈ set.Icc 0 2, x + c > 2)) ∧
  ¬ ((function.decreasing_on (λ x : ℝ, c^x) set.univ ∧
     (∀ x ∈ set.Icc 0 2, x + c > 2))) :=
sorry

end range_of_c_l256_256743


namespace I_value_l256_256197

open Finset

noncomputable def I (n : ℕ) : ℤ :=
∑ A in powerset (range (2 * n + 1)), 
  if A.nonempty then ∑ a in A, (-1)^a * a^2 else 0

theorem I_value (n : ℕ) : 
  I n = (2^(n-1) : ℤ) * n * (2 * n + 1) :=
sorry

end I_value_l256_256197


namespace coefficient_m5n5_in_mn_pow10_l256_256308

-- Definition of the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- The main theorem statement
theorem coefficient_m5n5_in_mn_pow10 : 
  (∃ c, (m + n) ^ 10 = c * m^5 * n^5 + ∑ (k ≠ 5), (binomial_coeff 10 k) * m^(10 - k) * n^k) → 
  c = 252 := 
by 
  sorry

end coefficient_m5n5_in_mn_pow10_l256_256308


namespace girls_total_distance_l256_256579

theorem girls_total_distance
  (boys_laps : ℕ)
  (girls_laps : ℕ)
  (boys_lap_distance : ℚ)
  (girls_lap_distance1 : ℚ)
  (girls_lap_distance2 : ℚ)
  (h1 : boys_laps = 27)
  (h2 : girls_laps = boys_laps + 9)
  (h3 : boys_lap_distance = 3 / 4)
  (h4 : girls_lap_distance1 = 3 / 4)
  (h5 : girls_lap_distance2 = 7 / 8)
  : girls_lap_distance1 * (girls_laps / 2) + girls_lap_distance2 * (girls_laps / 2) = 29.25 := 
begin
  sorry
end

end girls_total_distance_l256_256579


namespace g_is_even_function_l256_256563

def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even_function : ∀ x : ℝ, g (-x) = g x :=
by
  intro x
  rw [g, g]
  sorry

end g_is_even_function_l256_256563


namespace car_speed_problem_l256_256281

theorem car_speed_problem (S1 S2 : ℝ) (T : ℝ) (avg_speed : ℝ) (H1 : S1 = 70) (H2 : T = 2) (H3 : avg_speed = 80) :
  S2 = 90 :=
by
  have avg_speed_eq : avg_speed = (S1 + S2) / T := sorry
  have h : S2 = 90 := sorry
  exact h

end car_speed_problem_l256_256281


namespace area_DEF_eq_one_seventh_l256_256553

variables {A B C D E F : Type} [triangle : Triangle A B C]
-- Assume points D, E, F are midpoints of corresponding segments as given in the problem statement
variable (D_midpoint : Midpoint D (segment A E))
variable (E_midpoint : Midpoint E (segment B F))
variable (F_midpoint : Midpoint F (segment C D))
variable (area_ABC_eq_one : Area (triangle A B C) = 1)

theorem area_DEF_eq_one_seventh :
  Area (triangle D E F) = 1 / 7 :=
by
  sorry

end area_DEF_eq_one_seventh_l256_256553


namespace total_marbles_l256_256530

-- There are only red, blue, and yellow marbles
universe u
variable {α : Type u}

-- The ratio of red marbles to blue marbles to yellow marbles is \(2:3:4\)
variables {r b y T : ℕ}
variable (ratio_cond : 2 * y = 4 * r ∧ 3 * y = 4 * b)

-- There are 40 yellow marbles in the container
variable (yellow_cond : y = 40)

-- Prove the total number of marbles in the container is 90
theorem total_marbles (ratio_cond : 2 * y = 4 * r ∧ 3 * y = 4 * b) (yellow_cond : y = 40) :
  T = r + b + y → T = 90 :=
sorry

end total_marbles_l256_256530


namespace rhombus_area_and_perimeter_l256_256259

theorem rhombus_area_and_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 26) :
  let area := (d1 * d2) / 2
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let perimeter := 4 * s
  area = 234 ∧ perimeter = 20 * Real.sqrt 10 := by
  sorry

end rhombus_area_and_perimeter_l256_256259


namespace triangle_angle_B_90_l256_256921

theorem triangle_angle_B_90
  (B M W O H K P : Point)
  (triangle_BMW : Triangle B M W)
  (BM_lt_BW : B M < B W)
  (BW_lt_MW : B W < M W)
  (BO_altitude : Altitude B O)
  (BH_median : Median B H)
  (K_symmetric_M_O : Symmetric M K O)
  (PK_perp_MW : Perpendicular (Line P K) (Line M W))
  (P_on_BW : OnLineSegment P (LineSegment B W))
  (MP_perp_BH : Perpendicular (Line M P) (Line B H)) :
  IsRightAngle (Angle B M W) :=
sorry

end triangle_angle_B_90_l256_256921


namespace concyclic_points_l256_256584

-- Defining a structure for a triangle and its incenter
structure Triangle :=
(A B C : Point)

-- Defining the incenter of the triangle
def incenter (ABC : Triangle) : Point := sorry  -- Incenter finding definition goes here

-- Defining the angle bisector intersection point
def intersection_angle_bisector (ABC : Triangle) : Point := sorry  -- Definition of D

-- Definitions of M and N based on perpendicular bisector and angle bisectors
def point_M (ABC : Triangle) (D : Point) : Point := sorry -- Definition of M
def point_N (ABC : Triangle) (D : Point) : Point := sorry -- Definition of N

-- Statement of the theorem
theorem concyclic_points (ABC : Triangle) :
  let I := incenter ABC in
  let D := intersection_angle_bisector ABC in
  let M := point_M ABC D in
  let N := point_N ABC D in
  cyclic_points I A M N :=
sorry

end concyclic_points_l256_256584


namespace electricity_fee_and_usage_l256_256290

-- Problem definitions
def tiered_pricing (usage: ℕ) : ℝ :=
  if usage ≤ 200 then 0.5 * usage
  else if usage ≤ 450 then 0.5 * 200 + 0.7 * (usage - 200)
  else 0.5 * 200 + 0.7 * 250 + 1 * (usage - 450)

def july_fee := 300
def july_fee_expected := 170

axiom total_usage : ℕ := 500
axiom total_fee : ℝ := 290
axiom usage_may : ℕ
axiom usage_june : ℕ

axiom usage_condition1 : usage_may + usage_june = total_usage
axiom usage_condition2 : tiered_pricing usage_may + tiered_pricing usage_june = total_fee
axiom usage_condition3 : usage_june > usage_may
axiom usage_condition4 : usage_may < 450
axiom usage_condition5 : usage_june < 450

-- Lean statement
theorem electricity_fee_and_usage :
  tiered_pricing july_fee = july_fee_expected ∧ usage_may = 100 ∧ usage_june = 400 :=
by sorry

end electricity_fee_and_usage_l256_256290


namespace find_x_exponent_l256_256520

theorem find_x_exponent (x : ℝ) (h : 4^x = 32) : x = 5 / 2 :=
by sorry -- Placeholder for proof

end find_x_exponent_l256_256520


namespace prob_A_winning_l256_256224

variable (P_draw P_B : ℚ)

def P_A_winning := 1 - P_draw - P_B

theorem prob_A_winning (h1 : P_draw = 1 / 2) (h2 : P_B = 1 / 3) :
  P_A_winning P_draw P_B = 1 / 6 :=
by
  rw [P_A_winning, h1, h2]
  norm_num
  done

end prob_A_winning_l256_256224


namespace four_weighings_sufficient_three_weighings_insufficient_l256_256034

-- Conditions (definitions)
def numCans : ℕ := 80
def balanceScale (left right : List ℕ) : ℤ :=
  List.sum_left - List.sum_right  -- Assumes a function for the balance difference

-- Questions translated to Lean statements
theorem four_weighings_sufficient (weights : List ℕ) (h_weights_len : weights.length = numCans)
: ∃ (weighings : List (List ℕ × List ℕ)), 
    weighings.length = 4 ∧ -- four weighings
    (∀ can, ∃! (l r : List ℕ), (l, r) ∈ weighings ∧ can ∈ l ∨ can ∈ r) -- all cans can be classified 
  :=
sorry

theorem three_weighings_insufficient (weights : List ℕ) (h_weights_len : weights.length = numCans)
: ¬ ∃ (weighings : List (List ℕ × List ℕ)), 
    weighings.length = 3 ∧ -- three weighings
    (∀ can (l, r) ∈ weighings, can ∈ l ∨ can ∈ r) -- all cans can be classified
  :=
sorry

end four_weighings_sufficient_three_weighings_insufficient_l256_256034


namespace problem_solution_l256_256391

theorem problem_solution:
  (3⁻¹)^(Real.logBase 3 2) + (0.25) ^ -0.5 = (5/2) :=
  sorry

end problem_solution_l256_256391


namespace problem_solution_l256_256001

noncomputable def harmonic (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (1 : ℝ) / (i + 1)

theorem problem_solution :
  (2022 * harmonic 2023) / harmonic 2024 = 2022 := by
  sorry

end problem_solution_l256_256001


namespace smallest_n_for_sum_of_cubes_l256_256023

theorem smallest_n_for_sum_of_cubes :
  ∃ (n : ℕ), n > 0 ∧ (∃ (x : ℕ → ℤ), (∑ i in Finset.range n, (x i)^3) = 2002^2002) ∧ n = 4 :=
by
  sorry

end smallest_n_for_sum_of_cubes_l256_256023


namespace students_passing_course_l256_256900

theorem students_passing_course :
  let students_three_years_ago := 200
  let increase_factor := 1.5
  let students_two_years_ago := students_three_years_ago * increase_factor
  let students_last_year := students_two_years_ago * increase_factor
  let students_this_year := students_last_year * increase_factor
  students_this_year = 675 :=
by
  sorry

end students_passing_course_l256_256900


namespace proj_magnitude_eq_two_l256_256091

noncomputable def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def vector_dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def cos_angle_between_vectors (u v : ℝ × ℝ × ℝ) (θ : ℝ) : ℝ :=
  vector_dot_product u v / (vector_magnitude u * vector_magnitude v)

noncomputable def projection_magnitude (u v : ℝ × ℝ × ℝ) : ℝ :=
  Real.abs (vector_dot_product u v) / vector_magnitude v

theorem proj_magnitude_eq_two :
  ∀ (u z : ℝ × ℝ × ℝ),
    vector_magnitude u = 4 →
    vector_magnitude z = 5 →
    cos_angle_between_vectors u z (Real.pi / 3) = 1 / 2 →
    projection_magnitude u z = 2 :=
by
  intros u z hu hz hθ
  sorry

end proj_magnitude_eq_two_l256_256091


namespace equilateral_triangle_area_APQ_l256_256644

theorem equilateral_triangle_area_APQ (ABC : Triangle) 
  (h_eq : is_equilateral ABC)
  (h_side : ABC.sides = (10, 10, 10)) 
  (P Q : Point) 
  (hP : P ∈ segment ABC.A ABC.B) 
  (hQ : Q ∈ segment ABC.A ABC.C) 
  (h_tangent : is_tangent (segment P Q) ABC.incircle) 
  (hPQ : segment.length P Q = 4) : 
  area (triangle ABC.A P Q) = 5 / sqrt 3 :=
by 
  sorry

end equilateral_triangle_area_APQ_l256_256644


namespace compute_g_five_times_l256_256167

def g (x : ℤ) : ℤ :=
  if x ≥ 0 then -x^3 else x + 6

theorem compute_g_five_times :
  g (g (g (g (g 1)))) = -113 :=
  by sorry

end compute_g_five_times_l256_256167


namespace line_through_point_and_trisection_l256_256609

theorem line_through_point_and_trisection :
  ∃ P1 P2 P3 P4 : ℝ × ℝ,
  (P1 = (1, 2) ∧ P2 = (6, 0) ∧ P3 = (2, 3) ∧ 
   P4 = ((1 + 2*(6-1)/3), (2 + 2*(0-2)/3)) ∧ -- First trisection point
   (3 * P3.1 + P3.2 - 9 = 0) ∧ (3 * P4.1 + P4.2 - 9 = 0)) := 
begin
  sorry
end

end line_through_point_and_trisection_l256_256609


namespace positive_integers_count_is_66_l256_256008

theorem positive_integers_count_is_66 :
  (∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 
              ∀ i ∈ (List.ofDigits (Nat.digits 10 n)), 
              ((n ≥ 1000) → (n < 10000) ∧ consecutive_4_digits n ∧ sum_digits_divisible_by_3 n)) → 
  count_valid_numbers 1000 10000 = 66 := 
sorry

def consecutive_4_digits (n : ℕ) : Prop := 
  let digits := Nat.digits 10 n
  ∃ a b c d, digits = [a, b, c, d] ∧ 
             (∀ m, m ∈ [a, b, c, d] → m + 1 ∈ [a, b, c, d] ∨ m - 1 ∈ [a, b, c, d])

def sum_digits_divisible_by_3 (n : ℕ) : Prop := 
  List.sum (Nat.digits 10 n) % 3 = 0

def count_valid_numbers (low high : ℕ) : ℕ := 
  List.length (List.filter 
               (λ n, consecutive_4_digits n ∧ sum_digits_divisible_by_3 n) 
               (List.range (high - low + 1)))

end positive_integers_count_is_66_l256_256008


namespace min_value_of_f_l256_256887

def f (x a : ℝ) : ℝ := x^2 + x + a

theorem min_value_of_f (a : ℝ) (h : ∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), f x a ≤ 2) : 
  ∃ x ∈ Icc (-1 : ℝ) (1 : ℝ), f x a = - 1 / 4 :=
by
  sorry

end min_value_of_f_l256_256887


namespace segment_length_polar_coords_l256_256555

noncomputable def length_segment_cut_by_curve_from_line : ℝ :=
  let r : ℝ := 1
  let d : ℝ := |1 / (real.sqrt 2)|
  2 * real.sqrt (r^2 - d^2)

theorem segment_length_polar_coords : length_segment_cut_by_curve_from_line = real.sqrt 2 := by
  sorry

end segment_length_polar_coords_l256_256555


namespace magnitude_of_T_l256_256170

noncomputable def i : ℂ := Complex.I
noncomputable def T : ℂ := (1 + i)^19 - (1 - i)^19
noncomputable def magnitude_T : ℝ := Complex.abs T

theorem magnitude_of_T : magnitude_T = 512 * Real.sqrt 2 := 
by
-- sorry is used to skip the proof implementation
sorry

#check magnitude_of_T

end magnitude_of_T_l256_256170


namespace boundary_line_f_g_l256_256063

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

noncomputable def g (x : ℝ) : ℝ := 0.5 * (x - 1 / x)

theorem boundary_line_f_g :
  ∀ (x : ℝ), 1 ≤ x → (x - 1) ≤ f x ∧ (g x) ≤ (x - 1) :=
by
  intro x hx
  sorry

end boundary_line_f_g_l256_256063


namespace smallest_x_value_l256_256475

def floor (x : ℝ) : ℤ := Int.floor x

theorem smallest_x_value (x : ℝ) (h₀ : floor (x + 0.1) + floor (x + 0.2) + floor (x + 0.3) + 
                                      floor (x + 0.4) + floor (x + 0.5) + floor (x + 0.6) + 
                                      floor (x + 0.7) + floor (x + 0.8) + floor (x + 0.9) = 104) :
  x = 11.5 :=
sorry

end smallest_x_value_l256_256475


namespace trees_in_yard_l256_256138

theorem trees_in_yard :
  ∀ (yard_length tree_distance : ℕ), yard_length = 225 → tree_distance = 10 →
    ∃ (trees : ℕ), trees = 24 :=
by
  intros yard_length tree_distance h_yard_length h_tree_distance
  use 24
  sorry

end trees_in_yard_l256_256138


namespace exists_smallest_n_cube_sum_l256_256025

theorem exists_smallest_n_cube_sum (n : ℕ) (x : ℕ → ℤ) :
  (∀ i ∈ finset.range n, ∃ (x_i : ℤ), x i = x_i) ∧ 
  (finset.range n).sum (λ i, (x i) ^ 3) = 2002 ^ 2002 ∧
  (∀ m, m < n → ¬(∃ y : ℕ → ℤ, (∀ i ∈ finset.range m, ∃ (y_i : ℤ), y i = y_i) ∧ (finset.range m).sum (λ i, (y i) ^ 3) = 2002 ^ 2002))
  ↔ n = 4 := by
  sorry

end exists_smallest_n_cube_sum_l256_256025


namespace no_solution_to_equation_l256_256667

theorem no_solution_to_equation :
  ¬ ∃ x : ℝ, 8 / (x ^ 2 - 4) + 1 = x / (x - 2) :=
by
  sorry

end no_solution_to_equation_l256_256667


namespace none_takes_own_hat_probability_l256_256705

noncomputable def probability_none_takes_own_hat : ℚ :=
  have total_arrangements := 3.factorial
  have derangements : ℕ := 2
  have probability := (derangements : ℚ) / (total_arrangements : ℚ)
  probability

theorem none_takes_own_hat_probability : probability_none_takes_own_hat = 1 / 3 :=
by
  have total_arrangements := 3.factorial
  have derangements : ℕ := 2
  have probability := (derangements : ℚ) / (total_arrangements : ℚ)
  show probability_none_takes_own_hat = 1 / 3
  sorry

end none_takes_own_hat_probability_l256_256705


namespace complex_mul_conjugate_real_implies_t_value_l256_256074

/-- Given two complex numbers z1 and z2, if z1 * conjugate(z2) is real, then t must equal 3/4 -/
theorem complex_mul_conjugate_real_implies_t_value (t : ℝ) 
  (z1 : ℂ) (z2 : ℂ)
  (h1 : z1 = 3 + 4 * complex.I) 
  (h2 : z2 = t + complex.I) 
  (h_real : (z1 * conj(z2)).im = 0): 
  t = 3 / 4 :=
by 
  sorry

end complex_mul_conjugate_real_implies_t_value_l256_256074


namespace inequality_Cauchy_Schwarz_l256_256037

theorem inequality_Cauchy_Schwarz (a b : ℝ) : 
  (a^4 + b^4) * (a^2 + b^2) ≥ (a^3 + b^3)^2 :=
by
  sorry

end inequality_Cauchy_Schwarz_l256_256037


namespace six_digit_product_of_consecutive_even_integers_l256_256432

theorem six_digit_product_of_consecutive_even_integers :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ a * b * c = 287232 :=
sorry

end six_digit_product_of_consecutive_even_integers_l256_256432


namespace stool_height_l256_256376

theorem stool_height {h_ceiling h_bulb h_item : ℕ} 
    (Ceiling_height : h_ceiling = 300)
    (Bulb_below_ceiling : 15)
    (Item_below_ceiling : 20)
    {h_alice h_reach : ℕ}
    (Alice_height : h_alice = 160)
    (Reach_above_head : h_reach = 50) :
  h_bulb = h_ceiling - Bulb_below_ceiling → 
  h_item = h_ceiling - Item_below_ceiling → 
  h_bulb - (h_alice + h_reach) = 75 :=
by
  intros
  sorry

end stool_height_l256_256376


namespace inscribed_parallelogram_locus_l256_256835
-- Assuming required geometry and algebra imported from Mathlib

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)

variables {A B C D E F P Q R S : ℝ × ℝ}
variables (AB : line ℝ) (CD : line ℝ) (EF : segment ℝ) (PQ : line ℝ) (RS : line ℝ)
variables (side_dir : PQ.direction = RS.direction)

axiom points_on_lines :
  A ∈ AB ∧ B ∈ AB ∧ C ∈ CD ∧ D ∈ CD ∧ E ∈ AB ∧ F ∈ CD

axiom segment_EF :
  S ∈ EF

theorem inscribed_parallelogram_locus :
  ∀ (A B C D E F P Q R S : ℝ × ℝ),
  ∃ (PQRS : Parallelogram ℝ),
  PQRS.side_direction = side_dir →
  P ∈ AB ∧ Q ∈ line ℝ B C ∧ R ∈ CD ∧ S ∈ EF :=
sorry

end inscribed_parallelogram_locus_l256_256835


namespace chips_total_bags_l256_256977

theorem chips_total_bags (plain : ℕ) (bbq : ℕ) (p : ℚ) (h_plain : plain = 4) (h_bbq : bbq = 5) (h_p : p = 0.11904761904761904) :
  let T := plain + bbq in T = 9 :=
by
  sorry

end chips_total_bags_l256_256977


namespace simplify_expression_l256_256240

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem simplify_expression :
  (1/2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b) = 6 • b - 3 • a :=
by sorry

end simplify_expression_l256_256240


namespace total_pencils_l256_256699

theorem total_pencils (drawer_pencils desk_initial_pencils desk_added_pencils total_pencils : ℕ) 
    (h1 : drawer_pencils = 43) 
    (h2 : desk_initial_pencils = 19) 
    (h3 : desk_added_pencils = 16) 
    (h4 : total_pencils = drawer_pencils + desk_initial_pencils + desk_added_pencils) : 
    total_pencils = 78 := 
by
  rw [h1, h2, h3]
  simp [h4]
  sorry

end total_pencils_l256_256699


namespace eight_sided_die_divisible_by_48_l256_256380

/--
An eight-sided die is numbered from 1 to 8. When it is rolled, the product \( P \) of the seven numbers that are visible is always divisible by \( 48 \).
-/
theorem eight_sided_die_divisible_by_48 (f : Fin 8 → ℕ)
  (h : ∀ i, f i = i + 1) : 
  ∃ (P : ℕ), (∀ n, P = (∏ i in (Finset.univ.filter (λ j, j ≠ n)), f i)) ∧ (48 ∣ P) := 
by
  sorry

end eight_sided_die_divisible_by_48_l256_256380


namespace six_digit_number_consecutive_evens_l256_256435

theorem six_digit_number_consecutive_evens :
  ∃ n : ℕ,
    287232 = (2 * n - 2) * (2 * n) * (2 * n + 2) ∧
    287232 / 100000 = 2 ∧
    287232 % 10 = 2 :=
by
  sorry

end six_digit_number_consecutive_evens_l256_256435


namespace family_average_age_l256_256751

theorem family_average_age (A : ℝ) (number_of_family_members : ℕ) 
    (age_youngest : ℝ) (average_age_at_birth : ℝ)
    (number_of_family_members = 5) 
    (age_youngest = 10)
    (average_age_at_birth = 12.5) :
    A = 20 :=
  sorry

end family_average_age_l256_256751


namespace sum_distances_constant_l256_256982

variables {A B C O A_1 B_1 C_1 : Type}
variables (a h : ℝ)
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space A_1] [metric_space B_1] [metric_space C_1]
variables [normed_add_comm_group A] [normed_add_comm_group B] [normed_add_comm_group C]
variables [normed_add_comm_group A_1] [normed_add_comm_group B_1] [normed_add_comm_group C_1]

structure equilateral_triangle (A B C : Type) : Type :=
(side_length : ℝ)
(height : ℝ)
(ABC_eq : ∀ {O : Type} (O : O) (O_dist_A1 : ℝ) (O_dist_B1 : ℝ) (O_dist_C1 : ℝ), height = O_dist_A1 + O_dist_B1 + O_dist_C1)

theorem sum_distances_constant (T : equilateral_triangle A B C)
  (O : Type)
  (O_dist_A1 O_dist_B1 O_dist_C1 : ℝ)
  (H : T.height = O_dist_A1 + O_dist_B1 + O_dist_C1) :
  T.height = O_dist_A1 + O_dist_B1 + O_dist_C1 :=
begin
  sorry
end

end sum_distances_constant_l256_256982


namespace fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8_l256_256144

def visitors_enjoyed_understood_fraction (E U : ℕ) (total_visitors no_enjoy_no_understood : ℕ) : Prop :=
  E = U ∧
  no_enjoy_no_understood = 110 ∧
  total_visitors = 440 ∧
  E = (total_visitors - no_enjoy_no_understood) / 2 ∧
  E = 165 ∧
  (E / total_visitors) = 3 / 8

theorem fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8 :
  ∃ (E U : ℕ), visitors_enjoyed_understood_fraction E U 440 110 :=
by
  sorry

end fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8_l256_256144


namespace determine_sequence_parameters_l256_256909

variables {n : ℕ} {d q : ℝ} (h1 : 1 + (n-1) * d = 81) (h2 : 1 * q^(n-1) = 81) (h3 : q / d = 0.15)

theorem determine_sequence_parameters : n = 5 ∧ d = 20 ∧ q = 3 :=
by {
  -- Assumptions:
  -- h1: Arithmetic sequence, a1 = 1, an = 81
  -- h2: Geometric sequence, b1 = 1, bn = 81
  -- h3: q / d = 0.15
  -- Goal: n = 5, d = 20, q = 3
  sorry
}

end determine_sequence_parameters_l256_256909


namespace exists_triangle_sides_l256_256071

theorem exists_triangle_sides (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h1 : a * b * c ≤ 1 / 4)
  (h2 : 1 / (a^2) + 1 / (b^2) + 1 / (c^2) < 9) : 
  a + b > c ∧ b + c > a ∧ c + a > b := 
by
  sorry

end exists_triangle_sides_l256_256071


namespace max_distance_from_point_on_ellipse_to_focus_l256_256485

theorem max_distance_from_point_on_ellipse_to_focus :
  let a := 4
  let b := Real.sqrt 7
  let c := Real.sqrt (a^2 - b^2)
  let f1 := (-c, 0)
  let ellipse (x y : ℝ) := (x^2) / 16 + (y^2) / 7 = 1
  ∀ x y, ellipse x y → |((x + c)^2 + y^2)^0.5| ≤ 7 :=
begin
  sorry
end

end max_distance_from_point_on_ellipse_to_focus_l256_256485


namespace traffic_light_change_probability_l256_256369

theorem traffic_light_change_probability :
  let cycle_duration := 100
  let change_intervals := 15
  ∃ p : ℚ, p = (change_intervals : ℚ) / cycle_duration ∧ p = 3 / 20 :=
begin
  sorry
end

end traffic_light_change_probability_l256_256369


namespace b_not_six_iff_neg_two_not_in_range_l256_256451

def g (x b : ℝ) := x^3 + x^2 + b*x + 2

theorem b_not_six_iff_neg_two_not_in_range (b : ℝ) : 
  (∀ x : ℝ, g x b ≠ -2) ↔ b ≠ 6 :=
by
  sorry

end b_not_six_iff_neg_two_not_in_range_l256_256451


namespace factorize_expression_l256_256011

theorem factorize_expression (x y : ℝ) : 
  6 * x^3 * y^2 + 3 * x^2 * y^2 = 3 * x^2 * y^2 * (2 * x + 1) := 
by 
  sorry

end factorize_expression_l256_256011


namespace find_m_l256_256496

theorem find_m (m : ℝ) (h : ∀ x : ℝ, m - |x| ≥ 0 ↔ -1 ≤ x ∧ x ≤ 1) : m = 1 :=
sorry

end find_m_l256_256496


namespace natural_numbers_divide_power_of_two_l256_256014

open Nat

theorem natural_numbers_divide_power_of_two (n : ℕ) (h1 : n > 3) (h2 : 1 + binom n 1 + binom n 2 + binom n 3 ∣ 2^2000) : n = 7 ∨ n = 23 :=
  sorry

end natural_numbers_divide_power_of_two_l256_256014


namespace prank_combinations_zero_l256_256709

theorem prank_combinations_zero :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 0
  let saturday_choices := 2
  let sunday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices * saturday_choices * sunday_choices = 0 :=
by {
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 0
  let saturday_choices := 2
  let sunday_choices := 1
  calc
    1 * 2 * 6 * 5 * 0 * 2 * 1 = 0 : by sorry
}

end prank_combinations_zero_l256_256709


namespace pq_parallel_ac_l256_256920

variables {A B C P Q : Type} [Points A] [Points B] [Points C] [Points P] [Points Q]
variables {segment : Type → Type} [Segment segment]
variables {angle_bisector : Type → Type} [AngleBisector angle_bisector]
variables {perpendicular : Type → Type} 

noncomputable def is_parallel (a b : segment ABC) : Prop :=
sorry -- Definition of parallel segments

noncomputable def triangle (A B C : Type) [Points A] [Points B] [Points C] : Prop :=
sorry -- Definition of a triangle

noncomputable def is_angle_bisector (b : angle_bisector ABC) : Prop :=
sorry -- Definition of an angle bisector

noncomputable def is_perpendicular (p : perpendicular ABC) : Prop :=
sorry -- Definition of perpendicularity

axiom angle_bisectors_in_triangle :
  ∀ {A B C P Q : Type} [Points A] [Points B] [Points C] [Points P] [Points Q],
    triangle A B C → is_angle_bisector P → is_angle_bisector Q →
    (is_perpendicular B P) → (is_perpendicular B Q) →
    is_parallel (segment PQ) (segment AC)

theorem pq_parallel_ac
  (h₁ : triangle A B C)
  (h₂ : is_angle_bisector P)
  (h₃ : is_angle_bisector Q)
  (h₄ : is_perpendicular B P)
  (h₅ : is_perpendicular B Q) :
  is_parallel (segment PQ) (segment AC) :=
begin
  apply angle_bisectors_in_triangle,
  exact h₁,
  exact h₂,
  exact h₃,
  exact h₄,
  exact h₅,
end

end pq_parallel_ac_l256_256920


namespace average_of_measurements_l256_256711

def measurements : List ℝ := [79.4, 80.6, 80.8, 79.1, 80.0, 79.6, 80.5]

theorem average_of_measurements : (measurements.sum / measurements.length) = 80 := by sorry

end average_of_measurements_l256_256711


namespace symmetric_log_0_1_and_log_minus_x_l256_256839

def is_symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

theorem symmetric_log_0_1_and_log_minus_x :
  is_symmetric_about_origin (λ x, Real.log based 0.1 x) ∧
  is_symmetric_about_origin (λ x, Real.log based 10 (-x)) :=
sorry

end symmetric_log_0_1_and_log_minus_x_l256_256839


namespace ratio_bc_cd_l256_256320

-- Definitions based on given conditions.
variable (a b c d e : ℝ)
variable (h_ab : b - a = 5)
variable (h_ac : c - a = 11)
variable (h_de : e - d = 8)
variable (h_ae : e - a = 22)

-- The theorem to prove bc : cd = 2 : 1.
theorem ratio_bc_cd (h_ab : b - a = 5) (h_ac : c - a = 11) (h_de : e - d = 8) (h_ae : e - a = 22) :
  (c - b) / (d - c) = 2 :=
by
  sorry

end ratio_bc_cd_l256_256320


namespace fraction_evaluation_l256_256127

theorem fraction_evaluation (p q s u : ℚ) (hq : q ≠ 0) (hu : u ≠ 0) 
  (h1 : p / q = 5 / 4) (h2 : s / u = 7 / 8) : 
  (2 * p * s - 3 * q * u) / (5 * q * u - 4 * p * s) = -13 / 10 :=
by
  have hp : p = (5 / 4) * q, from sorry, -- Derived from h1
  have hs : s = (7 / 8) * u, from sorry, -- Derived from h2
  sorry -- Proof of the final result using above have statements

end fraction_evaluation_l256_256127


namespace coefficient_m5n5_in_mn_pow10_l256_256311

-- Definition of the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- The main theorem statement
theorem coefficient_m5n5_in_mn_pow10 : 
  (∃ c, (m + n) ^ 10 = c * m^5 * n^5 + ∑ (k ≠ 5), (binomial_coeff 10 k) * m^(10 - k) * n^k) → 
  c = 252 := 
by 
  sorry

end coefficient_m5n5_in_mn_pow10_l256_256311


namespace inequality_l256_256984

theorem inequality (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) :
  (Finset.univ.sum a) * (Finset.univ.sum (λ i, (a i)⁻¹)) ≥ n^2 :=
by
  sorry

end inequality_l256_256984


namespace exists_prime_pair_with_rational_roots_l256_256567

open Nat

theorem exists_prime_pair_with_rational_roots :
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ (∃ (x : ℚ), p * x^2 - q * x + p = 0) :=
by
  -- Conditions: p and q are prime numbers, the equation px^2 - qx + p = 0 has rational roots
  sorry

end exists_prime_pair_with_rational_roots_l256_256567


namespace alpha_plus_beta_is_pi_div_4_l256_256103

theorem alpha_plus_beta_is_pi_div_4 (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.sin α = sqrt 10 / 10) 
  (h4 : Real.cos β = 2 * sqrt 5 / 5) : 
  α + β = π / 4 :=
sorry

end alpha_plus_beta_is_pi_div_4_l256_256103


namespace largest_divisor_of_product_l256_256381

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Definition of P, the product of the visible numbers when an 8-sided die is rolled
def P (excluded: ℕ) : ℕ :=
  factorial 8 / excluded

-- The main theorem to prove
theorem largest_divisor_of_product (excluded: ℕ) (h₁: 1 ≤ excluded) (h₂: excluded ≤ 8): 
  ∃ n, n = 192 ∧ ∀ k, k > 192 → ¬k ∣ P excluded :=
sorry

end largest_divisor_of_product_l256_256381


namespace solve_system_l256_256998

theorem solve_system (x y z : ℝ) :
  (x^2 = 2 * real.sqrt (y^2 + 1) ∧
   y^2 = 2 * real.sqrt (z^2 - 1) - 2 ∧
   z^2 = 4 * real.sqrt (x^2 + 2) - 6) ↔
  ((x = real.sqrt 2 ∨ x = -real.sqrt 2) ∧ y = 0 ∧ (z = real.sqrt 2 ∨ z = -real.sqrt2)) :=
by
  sorry

end solve_system_l256_256998


namespace max_value_sqrt_function_l256_256474

theorem max_value_sqrt_function (x : ℝ) (h₁ : 2 < x) (h₂ : x < 5) :
  ∃ (x_max : ℝ), x_max = 4 ∧ f x_max = 4 * Real.sqrt 3 :=
begin
  let f : ℝ → ℝ := λ x, Real.sqrt (3 * x * (8 - x)),
  use 4,
  split,
  { refl }, -- This proves x_max = 4
  { sorry } -- Placeholder for the proof that f(4) = 4 * sqrt(3)
end

end max_value_sqrt_function_l256_256474


namespace find_EC_l256_256335

variable (A B C D E: Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variable (AB: ℝ) (CD: ℝ) (AC: ℝ) (EC: ℝ)
variable [Parallel : ∀ A B, Prop] [Measure : ∀ A B, Real]

def is_trapezoid (AB: ℝ) (CD: ℝ) := AB = 3 * CD

theorem find_EC 
  (h1 : is_trapezoid AB CD)
  (h2 : AC = 15)
  : EC = 15 / 4 :=
by
  sorry

end find_EC_l256_256335


namespace gcd_number_l256_256264

open Nat

theorem gcd_number : ∃ n : ℕ, 90 ≤ n ∧ n ≤ 100 ∧ gcd 35 n = 7 ∧ n = 98 :=
by
  use 98
  split
  · exact le_refl 98
  split
  · exact le_of_lt (lt_add_one 98)
  split
  · exact Nat.gcd_eq_right (Nat.dvd_of_mod_eq_zero rfl)
  · trivial

end gcd_number_l256_256264


namespace total_matches_in_group_l256_256149

theorem total_matches_in_group (n : ℕ) (hn : n = 6) : 2 * (n * (n - 1) / 2) = 30 :=
by
  sorry

end total_matches_in_group_l256_256149


namespace part1_1_part1_2_part2_l256_256740

theorem part1_1 (a b : ℝ) (h1 : a + b = 13) (h2 : a * b = 36) : (a - b)^2 = 25 := 
sorry

noncomputable def part1_2_solutions : set (ℝ × ℝ) := 
  {(8 / 3, 1 / 3), (-8 / 3, -1 / 3)}

theorem part1_2 (a b : ℝ) (h1 : a^2 + a * b = 8) (h2 : b^2 + b * a = 1) : (a, b) ∈ part1_2_solutions := 
sorry

theorem part2 (a b x y : ℝ) 
  (h1 : a * x + b * y = 3) 
  (h2 : a * x^2 + b * y^2 = 7) 
  (h3 : a * x^3 + b * y^3 = 16) 
  (h4 : a * x^4 + b * y^4 = 42) : 
  x + y = -14 := 
sorry

end part1_1_part1_2_part2_l256_256740


namespace last_three_digits_of_expression_l256_256787

theorem last_three_digits_of_expression : 
  let prod := 301 * 402 * 503 * 604 * 646 * 547 * 448 * 349
  (prod ^ 3) % 1000 = 976 :=
by
  sorry

end last_three_digits_of_expression_l256_256787


namespace pie_eating_contest_l256_256543

def pies_eaten (Adam Bill Sierra Taylor: ℕ) : ℕ :=
  Adam + Bill + Sierra + Taylor

theorem pie_eating_contest (Bill : ℕ) 
  (Adam_eq_Bill_plus_3 : ∀ B: ℕ, Adam = B + 3)
  (Sierra_eq_2times_Bill : ∀ B: ℕ, Sierra = 2 * B)
  (Sierra_eq_12 : Sierra = 12)
  (Taylor_eq_avg : ∀ A B S: ℕ, Taylor = (A + B + S) / 3)
  : pies_eaten Adam Bill Sierra Taylor = 36 := sorry

end pie_eating_contest_l256_256543


namespace count_integer_points_in_intersection_l256_256403

def sphere1 (x y z : ℤ) : Prop := x^2 + y^2 + (z - 10)^2 ≤ 64
def sphere2 (x y z : ℤ) : Prop := x^2 + y^2 + (z - 3)^2 ≤ 25

def inSphereIntersection (x y z : ℤ) : Prop := sphere1 x y z ∧ sphere2 x y z

theorem count_integer_points_in_intersection : 
  (λ S, ∃ n, n = 9 ∧ S = {p : ℤ × ℤ × ℤ | inSphereIntersection p.1 p.2 p.3}.card) :=
by 
  sorry

end count_integer_points_in_intersection_l256_256403


namespace equilateral_triangle_area_APQ_l256_256647

theorem equilateral_triangle_area_APQ (ABC : Triangle) 
  (h_eq : is_equilateral ABC)
  (h_side : ABC.sides = (10, 10, 10)) 
  (P Q : Point) 
  (hP : P ∈ segment ABC.A ABC.B) 
  (hQ : Q ∈ segment ABC.A ABC.C) 
  (h_tangent : is_tangent (segment P Q) ABC.incircle) 
  (hPQ : segment.length P Q = 4) : 
  area (triangle ABC.A P Q) = 5 / sqrt 3 :=
by 
  sorry

end equilateral_triangle_area_APQ_l256_256647


namespace sum_of_altitudes_l256_256754

theorem sum_of_altitudes (A B C : ℝ) (A_ne_zero : A ≠ 0) (B_ne_zero : B ≠ 0) 
  (h_line_eq : A * 4 + B * 10 = C)
  (h_sqrt_positive : sqrt ((A^2 + B^2)) > 0) :
    4 + 10 + (abs C) / (sqrt ((A^2) + (B^2))) =  14 + (40 / (sqrt 116)) :=
by
  sorry

end sum_of_altitudes_l256_256754


namespace sum_of_v_seq_l256_256407

open Real

noncomputable def v0 : ℝ³ := ⟨2, 1, 0⟩
noncomputable def w0 : ℝ³ := ⟨0, 0, 1⟩

noncomputable def v (n : ℕ) : ℝ³ :=
  if n = 0 then v0 else (0 : ℝ³)

noncomputable def w (n : ℕ) : ℝ³ :=
  if n = 0 then w0 else (0 : ℝ³)

lemma v_n_projection (n : ℕ) (hn : n ≥ 1) : v n = (0 : ℝ³) :=
begin
  -- proof goes here
  sorry
end

lemma w_n_projection (n : ℕ) (hn : n ≥ 1) : w n = (0 : ℝ³) :=
begin
  -- proof goes here
  sorry
end

theorem sum_of_v_seq : (Σ' n, v n) = (0 : ℝ³) :=
begin
  -- proof goes here
  sorry
end

end sum_of_v_seq_l256_256407


namespace two_digit_numbers_tens_greater_ones_l256_256511

theorem two_digit_numbers_tens_greater_ones : 
  ∃ (count : ℕ), count = 45 ∧ ∀ (n : ℕ), 10 ≤ n ∧ n < 100 → 
    let tens := n / 10;
    let ones := n % 10;
    tens > ones → count = 45 :=
by {
  sorry
}

end two_digit_numbers_tens_greater_ones_l256_256511


namespace possible_galina_numbers_l256_256822

def is_divisible_by (m n : ℕ) : Prop := n % m = 0

def conditions_for_galina_number (n : ℕ) : Prop :=
  let C1 := is_divisible_by 7 n
  let C2 := is_divisible_by 11 n
  let C3 := n < 13
  let C4 := is_divisible_by 77 n
  (C1 ∧ ¬C2 ∧ C3 ∧ ¬C4) ∨ (¬C1 ∧ C2 ∧ C3 ∧ ¬C4)

theorem possible_galina_numbers (n : ℕ) :
  conditions_for_galina_number n ↔ (n = 7 ∨ n = 11) :=
by
  -- Proof to be filled in
  sorry

end possible_galina_numbers_l256_256822


namespace skittles_left_l256_256222

theorem skittles_left (initial_skittles : ℕ) (skittles_given : ℕ) (final_skittles : ℕ) :
  initial_skittles = 50 → skittles_given = 7 → final_skittles = initial_skittles - skittles_given → final_skittles = 43 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end skittles_left_l256_256222


namespace ellipse_equation_l256_256049

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (e : ℝ) (h3 : e = 1 / 3) 
  (h4 : (\vector (-a) 0).dot (\vector a 0) = -1) : 
  (C : ℝ) : 
  (C = \vector (\frac x^2 9) + \vector (\frac y^2 8) = 1) := 
  sorry

end ellipse_equation_l256_256049


namespace point_not_on_graph_of_division_by_zero_l256_256318

noncomputable def pointNotOnGraph : Prop :=
  ¬ ∃ x : ℝ, y : ℝ, (y = (2 * x - 1) / (x + 2)) ∧ (x, y) = (-2, -5)

theorem point_not_on_graph_of_division_by_zero :
  pointNotOnGraph :=
by
  intro h
  apply exists.elim h
  intro x hx
  apply exists.elim hx
  intro y hy
  cases hy with hy_eq hy_coord
  cases hy_coord
  have : x + 2 = 0,
  sorry

end point_not_on_graph_of_division_by_zero_l256_256318


namespace projection_a_onto_b_l256_256094

def vector_a : ℝ × ℝ × ℝ := (1, 3, 0)
def vector_b : ℝ × ℝ × ℝ := (2, 1, 1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def projection_vector (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let scale := dot_product u v / (magnitude v) ^ 2
  (scale * v.1, scale * v.2, scale * v.3)

theorem projection_a_onto_b :
  projection_vector vector_a vector_b = (5 / 3, 5 / 6, 5 / 6) :=
by
  sorry

end projection_a_onto_b_l256_256094


namespace Ian_kept_1_rose_l256_256517

def initial_roses : ℕ := 20
def roses_given_to_mother : ℕ := 6
def roses_given_to_grandmother : ℕ := 9
def roses_given_to_sister : ℕ := 4
def total_roses_given : ℕ := roses_given_to_mother + roses_given_to_grandmother + roses_given_to_sister
def roses_kept (initial: ℕ) (given: ℕ) : ℕ := initial - given

theorem Ian_kept_1_rose :
  roses_kept initial_roses total_roses_given = 1 :=
by
  sorry

end Ian_kept_1_rose_l256_256517


namespace b_is_arithmetic_sum_first_n_terms_geometric_impossible_l256_256606

-- Define the sequence a_n
def a : ℕ → ℚ
| 0     := 0
| (n+1) := (4 * a n + 2 * real.sqrt (4 * a n + 1) + 1) / 4

-- Define b_n = sqrt(4 * a_n + 1)
def b (n : ℕ) : ℚ := real.sqrt (4 * (a n) + 1)

-- Question 1: Prove sequence {b_n} is arithmetic
theorem b_is_arithmetic : ∀ n : ℕ, b (n+1) = b n + 1 := 
sorry

-- Define c_n = 1 / a_{n+1}
def c (n : ℕ) : ℚ := 1 / (a (n+1))

-- Define S_n as the sum of the first n terms of {c_n}
def S (n : ℕ) : ℚ := ∑ i in finset.range n, c i

-- Question 2: Prove the sum of the first n terms of {c_n} is as expected
theorem sum_first_n_terms : ∀ n : ℕ, S n = 3 - 2 * (2 * n + 3) / ((n+1) * (n+2)) := 
sorry

-- Question 3: Prove there do not exist m, n such that 1, a_m, a_n form a geometric sequence
theorem geometric_impossible : ¬∃ (m n : ℕ), m ≠ n ∧ 1 < m ∧ 1 < n ∧ 1 * a m * a n = a m ^ 2 := 
sorry

end b_is_arithmetic_sum_first_n_terms_geometric_impossible_l256_256606


namespace tangents_perpendicular_l256_256246

theorem tangents_perpendicular (a : ℝ) (x₀ : ℝ) (h₀ : a ≠ 0) (h₁ : cos x₀ = a * tan x₀) : 
  (-(sin x₀)) * (a * (1 + (tan x₀)^2)) = -1 :=
by 
  sorry

end tangents_perpendicular_l256_256246


namespace complex_power_sum_l256_256604

open Complex

theorem complex_power_sum (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 + Complex.I * Real.sqrt 3 ∨ z^99 + z^100 + z^101 + z^102 + z^103 = 2 - Complex.I * Real.sqrt 3 :=
sorry

end complex_power_sum_l256_256604


namespace income_statistics_changes_l256_256674

open Real

-- Define the income data and their basic properties
variables (incomes : Fin 101 → ℝ)
variables (income_jack_ma : ℝ)
hypothesis (income_jack_ma_value : income_jack_ma ≈ 10^10)
hypothesis (max_income : ∀ i : Fin 100, incomes i ≤ 20000)

-- Define median, mean, and variance for initial 100 incomes
noncomputable def median_initial (incomes : Fin 100 → ℝ) : ℝ := (incomes 49 + incomes 50) / 2
noncomputable def mean_initial (incomes : Fin 100 → ℝ) : ℝ := (∑ i in Finset.range 100, incomes i) / 100
noncomputable def variance_initial (incomes : Fin 100 → ℝ) : ℝ := (∑ i in Finset.range 100, (incomes i - mean_initial incomes)^2) / 100

-- Define median, mean, and variance for all 101 incomes
noncomputable def median_all (incomes : Fin 101 → ℝ) : ℝ := incomes 50
noncomputable def mean_all (incomes : Fin 101 → ℝ) : ℝ := (∑ i in Finset.range 101, incomes i + income_jack_ma) / 101
noncomputable def variance_all (incomes : Fin 101 → ℝ) : ℝ := (∑ i in Finset.range 101, (incomes i - mean_all incomes)^2 + (income_jack_ma - mean_all incomes)^2) / 101

-- The main theorem to prove
theorem income_statistics_changes {incomes : Fin 100 → ℝ} {income_jack_ma : ℝ}
  (income_jack_ma_value : income_jack_ma ≈ 10^10)
  (max_income : ∀ i : Fin 100, incomes i ≤ 20000) :
  mean_all incomes income_jack_ma > mean_initial incomes ∧
  (median_all (λ i, if i = 100 then income_jack_ma else incomes i) = incomes 50 ∨
   median_all (λ i, if i = 100 then income_jack_ma else incomes i) ≠ incomes 50) ∧
  variance_all incomes income_jack_ma > variance_initial incomes :=
sorry

end income_statistics_changes_l256_256674


namespace chess_tournament_l256_256140

theorem chess_tournament (n : ℕ) (participants : fin n → ℕ → ℕ) (points : fin n → ℕ) :
  (∀ p : fin n, points p = n - 1) →
  (∀ p1 p2 : fin n, p1 ≠ p2 → (∃ games : fin n → ℕ, games p1 ≠ games p2)) →
  ∃ p1 p2 : fin n, p1 ≠ p2 ∧ (λ p, participants p 0 = participants p1 0 ∨ participants p 0 = participants p2 0) :=
by
  sorry

end chess_tournament_l256_256140


namespace problem1_problem2_problem3_l256_256994

theorem problem1 : 999 * 999 + 1999 = 1000000 := by
  sorry

theorem problem2 : 9 * 72 * 125 = 81000 := by
  sorry

theorem problem3 : 416 - 327 + 184 - 273 = 0 := by
  sorry

end problem1_problem2_problem3_l256_256994


namespace functional_eqn_l256_256013

noncomputable def f : ℤ → ℝ := fun x => (5/2)^x

theorem functional_eqn (f : ℤ → ℝ) 
  (h1 : ∀ x y : ℤ, f x * f y = f (x + y) + f (x - y))
  (h2 : f 1 = 5/2) : 
  ∀ x : ℤ, f x = (5/2)^x :=
begin
  sorry
end

end functional_eqn_l256_256013


namespace equilateral_triangle_sum_l256_256396

noncomputable def equilateral_triangle (a b c : Complex) (s : ℝ) : Prop :=
  Complex.abs (a - b) = s ∧ Complex.abs (b - c) = s ∧ Complex.abs (c - a) = s

theorem equilateral_triangle_sum (a b c : Complex):
  equilateral_triangle a b c 18 →
  Complex.abs (a + b + c) = 36 →
  Complex.abs (b * c + c * a + a * b) = 432 := by
  intros h_triangle h_sum
  sorry

end equilateral_triangle_sum_l256_256396


namespace claire_crafting_hours_l256_256792

theorem claire_crafting_hours (H1 : 24 = 24) (H2 : 8 = 8) (H3 : 4 = 4) (H4 : 2 = 2):
  let total_hours_per_day := 24
  let sleep_hours := 8
  let cleaning_hours := 4
  let cooking_hours := 2
  let working_hours := total_hours_per_day - sleep_hours
  let remaining_hours := working_hours - (cleaning_hours + cooking_hours)
  let crafting_hours := remaining_hours / 2
  crafting_hours = 5 :=
by
  sorry

end claire_crafting_hours_l256_256792


namespace question_I_question_II_l256_256956

def is_real_number (Z : ℂ) := ∀ x : ℝ, Z = x

def second_quadrant (Z : ℂ) := Z.re < 0 ∧ Z.im > 0

def Z (m : ℝ) : ℂ := complex.log (m^2 + 2*m - 14) + (m^2 - m - 6) * I

theorem question_I : m = 3 :=
by {
  have h : m^2 - m - 6 = 0, sorry,
  have h_m_3 : m = 3, sorry,
  exact h_m_3,
}

theorem question_II : -5 < m ∧ m < -1 - real.sqrt 15 :=
by {
  have h1 : 0 < m^2 + 2*m - 14 < 1, sorry,
  have h2 : m^2 - m - 6 > 0, sorry,
  split,
  exact h1.left,
  exact h1.right,
}

end question_I_question_II_l256_256956


namespace eric_sara_total_drink_l256_256009

theorem eric_sara_total_drink (x : ℝ) (h1 : Sara's_drink = 1.4 * Eric's_drink) 
  (h2 : Eric_left = x / 3) (h3 : Sara_left = 1.4 * x / 3) 
  (h4 : given_Amount = 0.7 * x / 3 + 3) : 
  let Eric_total := (2 / 3 * x + 0.7 * x / 3 + 3) in
  let Sara_total := (2.8 * x / 3 - 3) in
  Eric_total - Sara_total = 0 →
  2 * Eric_total = 46 :=
by
  -- substitution and calculations go here
  sorry

end eric_sara_total_drink_l256_256009


namespace isosceles_triangle_problem_l256_256252

theorem isosceles_triangle_problem 
  (a h b : ℝ) 
  (area_relation : (1/2) * a * h = (1/3) * a ^ 2) 
  (leg_relation : b = a - 1)
  (height_relation : h = (2/3) * a) 
  (pythagorean_theorem : h ^ 2 + (a / 2) ^ 2 = b ^ 2) : 
  a = 6 ∧ b = 5 ∧ h = 4 :=
sorry

end isosceles_triangle_problem_l256_256252


namespace radius_of_circle_l256_256480

noncomputable def point : Type := ℝ × ℝ

def line (a b c : ℝ) : point → Prop :=
λ ⟨x, y⟩, a * x + b * y + c = 0

def circle (center : point) (r : ℝ) : point → Prop :=
λ ⟨x, y⟩, (x - center.1) ^ 2 + (y - center.2) ^ 2 = r ^ 2

def distance (p1 p2 : point) : ℝ :=
((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2).sqrt

theorem radius_of_circle :
  ∃ (center : point) (r : ℝ),
    circle center r (-2, 0) ∧ 
    (∃ (B : point), B = (-1, 1) ∧ line 3 (-4) 7 B ∧ distance center B = r)
  → r = 5 :=
sorry

end radius_of_circle_l256_256480


namespace aaron_next_birthday_age_l256_256374

theorem aaron_next_birthday_age (years months weeks days : ℕ) 
(hy : years = 50) (hm : months = 50) (hw : weeks = 50) (hd : days = 50) : 
years + (months / 12) + ((weeks * 7) / 365.25) + (days / 365.25) = 56 := 
sorry

end aaron_next_birthday_age_l256_256374


namespace domain_of_f_value_at_pi_over_4_monotonic_increase_interval_l256_256493

noncomputable def f (x : ℝ) : ℝ := (sin (2 * x) + 2 * (cos x) ^ 2) / cos x 

theorem domain_of_f :
  ∀ (x : ℝ), (∃ k : ℤ, x = k * π + π / 2) ↔ (cos x = 0) := by
  sorry

theorem value_at_pi_over_4 :
  f (π / 4) = 2 * Real.sqrt 2 := by
  sorry

theorem monotonic_increase_interval :
  ∃ (a b : ℝ), 0 < a ∧ a < b ∧ b < π / 2 ∧ 
  (∀ (x y : ℝ), a < x ∧ x < b ∧ a < y ∧ y < b → x < y → f x < f y) := by
  use (0 : ℝ), (π / 4 : ℝ)
  sorry

end domain_of_f_value_at_pi_over_4_monotonic_increase_interval_l256_256493


namespace coeff_m5n5_in_m_plus_n_pow_10_l256_256300

theorem coeff_m5n5_in_m_plus_n_pow_10 :
  binomial (10, 5) = 252 := by
sorry

end coeff_m5n5_in_m_plus_n_pow_10_l256_256300


namespace trig_inequality_l256_256590

noncomputable def a : ℝ := Real.sin (31 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (58 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (32 * Real.pi / 180)

theorem trig_inequality : c > b ∧ b > a := by
  sorry

end trig_inequality_l256_256590


namespace train_length_solution_l256_256371

noncomputable def train_length_problem : Prop :=
  let speed_kph := 42
  let bridge_length_m := 200
  let time_sec := 60
  let speed_mps := (speed_kph * 1000) / 3600
  let total_distance_m := speed_mps * time_sec in
  total_distance_m - bridge_length_m = 500.2

theorem train_length_solution : train_length_problem :=
by
  sorry

end train_length_solution_l256_256371


namespace max_distance_focus_l256_256488

theorem max_distance_focus :
  let C := {P : ℝ × ℝ | ∃ x y : ℝ, P = (x, y) ∧ (x^2 / 16 + y^2 / 7 = 1)}
  let F₁ := (-3, 0)
  ∀ P ∈ C, |(fst P + 3)^2 + (snd P)^2| ≤ 7 :=
by
  -- definition of the ellipse C
  let C := {P : ℝ × ℝ | ∃ x y : ℝ, P = (x, y) ∧ (x^2 / 16 + y^2 / 7 = 1)}
  -- coordinates of the left focus F₁
  let F₁ := (-3, 0)
  -- prove that the maximum distance from any point P on the ellipse to F₁ is 7
  intros P hP
  sorry

end max_distance_focus_l256_256488


namespace sum_of_fifth_powers_divisibility_l256_256566

theorem sum_of_fifth_powers_divisibility (a b c d e : ℤ) :
  (a^5 + b^5 + c^5 + d^5 + e^5) % 25 = 0 → (a % 5 = 0) ∨ (b % 5 = 0) ∨ (c % 5 = 0) ∨ (d % 5 = 0) ∨ (e % 5 = 0) :=
by
  sorry

end sum_of_fifth_powers_divisibility_l256_256566


namespace original_cube_volume_l256_256351

theorem original_cube_volume :
  ∃ (s : ℝ), (8 * s^3 = 512) → (s^3 = 64) :=
by
  intros s h1
  have h2 : 8 * s^3 = 512 := h1
  sorry

end original_cube_volume_l256_256351


namespace minimum_value_fraction_l256_256869

-- Define the conditions in Lean
theorem minimum_value_fraction
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (line_through_center : ∀ x y, x = 1 ∧ y = -2 → a * x - b * y - 1 = 0) :
  (2 / a + 1 / b) = 8 := 
sorry

end minimum_value_fraction_l256_256869


namespace quadratic_point_value_l256_256682

theorem quadratic_point_value 
  (a b c : ℝ) 
  (h_min : ∀ x : ℝ, a * x^2 + b * x + c ≥ a * (-1)^2 + b * (-1) + c) 
  (h_at_min : a * (-1)^2 + b * (-1) + c = -3)
  (h_point : a * (1)^2 + b * (1) + c = 7) : 
  a * (3)^2 + b * (3) + c = 37 :=
sorry

end quadratic_point_value_l256_256682


namespace range_of_m_l256_256852

theorem range_of_m (x m : ℝ) (h1 : 2 * x - m ≤ 3) (h2 : -5 < x) (h3 : x < 4) :
  ∃ m, ∀ (x : ℝ), (-5 < x ∧ x < 4) → (2 * x - m ≤ 3) ↔ (m ≥ 5) :=
by sorry

end range_of_m_l256_256852


namespace factorize_l256_256426

theorem factorize (m : ℝ) : m^3 - 4 * m = m * (m + 2) * (m - 2) :=
by
  sorry

end factorize_l256_256426


namespace triangle_DBN_side_lengths_and_trig_values_l256_256918

-- Given conditions
def is_square (ABCD : ℝ) := ABCD = 1
def is_equilateral (BCE : ℝ) := BCE = 1
def midpoint (M CE : ℝ) := M = CE / 2
def perpendicular (DN BM CE : ℝ) := BM ⊥ CE ∧ DN ⊥ BM

theorem triangle_DBN_side_lengths_and_trig_values :
  (is_square 1) →
  (is_equilateral 1) →
  (midpoint 0.5 1) →
  (perpendicular 0 0 1) →
  (
    let DB := real.sqrt 2,
    let DN := (real.sqrt 3 + 1) / 2,
    let BN := (real.sqrt 3 - 1) / 2,
    let cos₇₅ := (real.sqrt 6 - real.sqrt 2) / 4,
    let sin₇₅ := (real.sqrt 6 + real.sqrt 2) / 4,
    let tan₇₅ := (real.sqrt 6 + real.sqrt 2) / (real.sqrt 6 - real.sqrt 2),
    let cos₁₅ := (real.sqrt 6 + real.sqrt 2) / 4,
    let sin₁₅ := (real.sqrt 6 - real.sqrt 2) / 4,
    let tan₁₅ := (real.sqrt 6 - real.sqrt 2) / (real.sqrt 6 + real.sqrt 2)
  ) →
  (
    DB = real.sqrt 2 ∧
    DN = (real.sqrt 3 + 1) / 2 ∧
    BN = (real.sqrt 3 - 1) / 2 ∧
    cos₇₅ = (real.sqrt 6 - real.sqrt 2) / 4 ∧
    sin₇₅ = (real.sqrt 6 + real.sqrt 2) / 4 ∧
    tan₇₅ = (real.sqrt 6 + real.sqrt 2) / (real.sqrt 6 - real.sqrt 2) ∧
    cos₁₅ = (real.sqrt 6 + real.sqrt 2) / 4 ∧
    sin₁₅ = (real.sqrt 6 - real.sqrt 2) / 4 ∧
    tan₁₅ = (real.sqrt 6 - real.sqrt 2) / (real.sqrt 6 + real.sqrt 2)
  ) :=
by sorry

end triangle_DBN_side_lengths_and_trig_values_l256_256918


namespace common_sum_42_l256_256776

theorem common_sum_42 : 
  let elements := list.range' (-18) 49;
  let n := 7;
  let total_sum := list.sum elements;
  (total_sum = 294) → (total_sum / n) = 42 :=
by
  sorry

end common_sum_42_l256_256776


namespace marks_in_physics_l256_256367

section
variables (P C M B CS : ℕ)

-- Given conditions
def condition_1 : Prop := P + C + M + B + CS = 375
def condition_2 : Prop := P + M + B = 255
def condition_3 : Prop := P + C + CS = 210

-- Prove that P = 90
theorem marks_in_physics : condition_1 P C M B CS → condition_2 P M B → condition_3 P C CS → P = 90 :=
by sorry
end

end marks_in_physics_l256_256367


namespace salad_dressing_vinegar_percentage_l256_256988

-- Define the initial conditions
def percentage_in_vinegar_in_Q : ℝ := 10
def percentage_of_vinegar_in_combined : ℝ := 12
def percentage_of_dressing_P_in_combined : ℝ := 0.10
def percentage_of_dressing_Q_in_combined : ℝ := 0.90
def percentage_of_vinegar_in_P (V : ℝ) : ℝ := V

-- The statement to prove
theorem salad_dressing_vinegar_percentage (V : ℝ) 
  (hQ : percentage_in_vinegar_in_Q = 10)
  (hCombined : percentage_of_vinegar_in_combined = 12)
  (hP_combined : percentage_of_dressing_P_in_combined = 0.10)
  (hQ_combined : percentage_of_dressing_Q_in_combined = 0.90)
  (hV_combined : 0.10 * percentage_of_vinegar_in_P V + 0.90 * percentage_in_vinegar_in_Q = 12) :
  V = 30 :=
by 
  sorry

end salad_dressing_vinegar_percentage_l256_256988


namespace probability_one_divisible_by_2_l256_256722

noncomputable def probability_exactly_one_divisible_by_2 (s : Finset ℕ) (d : ℕ) : ℚ :=
  let total_ways := (s.card.choose 3 : ℕ)
  let even_numbers := s.filter (λ x, x % 2 = 0)
  let odd_numbers := s.filter (λ x, x % 2 ≠ 0)
  let favorable_ways := even_numbers.card * odd_numbers.card.choose 2
  favorable_ways / total_ways

theorem probability_one_divisible_by_2 {s : Finset ℕ} (h : s = {1, 2, 3, 4, 5}) :
  probability_exactly_one_divisible_by_2 s 2 = 3 / 5 :=
by
  rw [probability_exactly_one_divisible_by_2, h]
  sorry

end probability_one_divisible_by_2_l256_256722


namespace number_of_n_not_dividing_g_in_range_l256_256187

def g (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ x, x ≠ n ∧ x ∣ n) (finset.range (n+1))), d

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem number_of_n_not_dividing_g_in_range :
  (finset.filter (λ n, n ∉ (finset.range (101)).filter (λ n, n ∣ g n))
  (finset.Icc 2 100)).card = 29 :=
by
  sorry

end number_of_n_not_dividing_g_in_range_l256_256187


namespace expected_miss_volleys_l256_256294

noncomputable def shooter_miss_prob_1 := 0.2
noncomputable def shooter_miss_prob_2 := 0.4
noncomputable def total_volleys := 25
noncomputable def combined_miss_prob := shooter_miss_prob_1 * shooter_miss_prob_2

theorem expected_miss_volleys : 
  total_volleys * combined_miss_prob = 2 := by
sorry

end expected_miss_volleys_l256_256294


namespace sufficient_but_not_necessary_condition_l256_256176

theorem sufficient_but_not_necessary_condition (a b : ℝ) : 
  (a ≥ 1 ∧ b ≥ 1) → (a + b ≥ 2) ∧ ¬((a + b ≥ 2) → (a ≥ 1 ∧ b ≥ 1)) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l256_256176


namespace intersection_point_l256_256581

noncomputable def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem intersection_point : ∃ a : ℝ, g a = a ∧ a = -3 :=
by
  use -3
  split
  · -- g(-3) = -3
    show g (-3) = -3
    sorry
  · -- a = -3 is tautology
    show -3 = -3
    exact rfl

end intersection_point_l256_256581


namespace number_of_planes_through_intersecting_lines_l256_256704

-- Define that three lines m, n, l intersect at a single point
def lines_intersect_at_single_point (m n l : Line) (P : Point) : Prop :=
  m.includes P ∧ n.includes P ∧ l.includes P

-- Define the problem stating that there is either one or zero planes passing through intersecting lines m, n, l
theorem number_of_planes_through_intersecting_lines (m n l : Line) (P : Point) :
  lines_intersect_at_single_point m n l P → 
  ∃ k, k = 0 ∨ k = 1 ∧ (number_of_planes_through m n l = k) :=
sorry

end number_of_planes_through_intersecting_lines_l256_256704


namespace required_raise_after_wage_cut_l256_256766

theorem required_raise_after_wage_cut (W : ℝ) : 
  let new_wage := 0.7 * W in
  let required_raise := (W / new_wage - 1) * 100 in
  required_raise ≈ 42.86 :=
by
  sorry

end required_raise_after_wage_cut_l256_256766


namespace planes_perpendicular_then_yz_l256_256853

variable (y z : ℝ)

def u : ℝ × ℝ × ℝ := (3, -1, z)
def v : ℝ × ℝ × ℝ := (-2, -y, 1)
def dot_product (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3

theorem planes_perpendicular_then_yz :
  dot_product (u y z) (v y z) = 0 →
  y + z = 6 :=
by
  sorry

end planes_perpendicular_then_yz_l256_256853


namespace min_value_144_l256_256943

noncomputable def min_expression (a b c d : ℝ) : ℝ :=
  (a + b + c) / (a * b * c * d)

theorem min_value_144 (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_pos_d : 0 < d) (h_sum : a + b + c + d = 2) : min_expression a b c d ≥ 144 :=
by
  sorry

end min_value_144_l256_256943


namespace correct_sum_of_first_four_cards_l256_256989

namespace CardStack

-- Definitions for red and blue cards
def red_cards : List ℕ := [2, 3, 5, 1]
def blue_cards : List ℕ := [2, 4, 6, 8, 9]

-- Condition: Red and Blue cards must alternate in the stack
def alternate_stack (stack : List ℕ) : Prop :=
  ∀ n, n < stack.length - 1 → (stack[n] ∈ red_cards ↔ stack[n+1] ∈ blue_cards)

-- Condition: Adjacency rule for divisibility
def divisible_adjacency (stack : List ℕ) : Prop :=
  ∀ n, n < stack.length - 1 →
    (stack[n] ∈ red_cards → stack[n+1] ∈ blue_cards ∧ stack[n+1] % stack[n] = 0) ∧
    (stack[n+1] ∈ red_cards → stack[n] ∈ blue_cards ∧ stack[n] % stack[n+1] = 0)

-- The sum of the first four cards in the stack
def sum_first_four (stack : List ℕ) : ℕ :=
  (stack.take 4).sum

-- The main theorem
theorem correct_sum_of_first_four_cards
  (stack : List ℕ)
  (h_alt : alternate_stack stack)
  (h_div : divisible_adjacency stack) :
  sum_first_four stack = 12 :=
sorry

end CardStack

end correct_sum_of_first_four_cards_l256_256989


namespace tetrahedron_inequality_l256_256154

/-- Given a tetrahedron ABCD where all plane angles at vertex A are 60 degrees,
    prove that AB + AC + AD <= BC + CD + BD. -/
theorem tetrahedron_inequality (A B C D : Point)
  (angle_BAC : ∠ B A C = 60°)
  (angle_BAD : ∠ B A D = 60°)
  (angle_CAD : ∠ C A D = 60°) :
  distance A B + distance A C + distance A D ≤ distance B C + distance C D + distance D B := 
sorry

end tetrahedron_inequality_l256_256154


namespace three_digit_difference_l256_256889

theorem three_digit_difference (x : ℕ) (a b c : ℕ)
  (h1 : a = x + 2)
  (h2 : b = x + 1)
  (h3 : c = x)
  (h4 : a > b)
  (h5 : b > c) :
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = 198 :=
by
  sorry

end three_digit_difference_l256_256889


namespace range_of_a_l256_256082

noncomputable def is_decreasing_on_interval (a : ℝ) : Prop :=
∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ x1 < x2 → 
  log a (2 - a * x2) < log a (2 - a * x1)

theorem range_of_a (a : ℝ) (h : a > 0) (h1 : a ≠ 1) (h2 : is_decreasing_on_interval a) : 
  1 < a ∧ a < 2 :=
sorry

end range_of_a_l256_256082


namespace intersection_A_B_l256_256471

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | -Real.sqrt 3 ≤ x ∧ x ≤ 1}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} :=
by
  sorry

end intersection_A_B_l256_256471


namespace largest_distinct_digit_number_with_product_2016_l256_256019

def digits (n : ℕ) : List ℕ := n.digits 10

def distinct_digits (n : ℕ) : Prop := (digits n).nodup

def digit_product (n : ℕ) : ℕ := List.prod (digits n)

theorem largest_distinct_digit_number_with_product_2016 : 
  ∃ n : ℕ, distinct_digits n ∧ digit_product n = 2016 ∧ ∀ m : ℕ, (distinct_digits m ∧ digit_product m = 2016) → n ≥ m :=
sorry

end largest_distinct_digit_number_with_product_2016_l256_256019


namespace combined_salaries_l256_256689

theorem combined_salaries (A B C D E : ℝ) 
  (hA : A = 9000) 
  (h_avg : (A + B + C + D + E) / 5 = 8200) :
  (B + C + D + E) = 32000 :=
by
  sorry

end combined_salaries_l256_256689


namespace valid_lists_12_l256_256950

noncomputable def valid_lists_count (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2 * valid_lists_count (n-1)

theorem valid_lists_12 :
  valid_lists_count 12 = 2048 :=
by
  let F : ℕ → ℕ
    | 0 => 0
    | 1 => 1
    | (n+1) => 2 * F n
  have h : F 12 = 2048,
  admit,
  exact h

end valid_lists_12_l256_256950


namespace maria_trip_distance_l256_256005

variable (D : ℕ) -- Defining the total distance D as a natural number

-- Defining the conditions given in the problem
def first_stop_distance := D / 2
def second_stop_distance := first_stop_distance - (1 / 3 * first_stop_distance)
def third_stop_distance := second_stop_distance - (2 / 5 * second_stop_distance)
def remaining_distance := 180

-- The statement to prove
theorem maria_trip_distance : third_stop_distance = remaining_distance → D = 900 :=
by
  sorry

end maria_trip_distance_l256_256005


namespace sequence_gcd_is_index_l256_256279

theorem sequence_gcd_is_index (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ i : ℕ, a i = i :=
by
  sorry

end sequence_gcd_is_index_l256_256279


namespace determinant_scaled_matrix_l256_256935

variables {V : Type} [inner_product_space ℝ V]
variables (a b c : V)
variable (D : ℝ)
variable (k : ℝ)

-- Given D is the determinant of the matrix with column vectors a, b, c.
axiom deter : D = a • (b × c)

-- Prove that the determinant of the matrix with column vectors k * a + b, b + k * c, k * c + a is (k^2 + 2k) * D.
theorem determinant_scaled_matrix :
  let v1 := k • a + b
      v2 := b + k • c
      v3 := k • c + a in
  (v1 • (v2 × v3)) = (k^2 + 2 * k) * D :=
sorry

end determinant_scaled_matrix_l256_256935


namespace passing_students_this_year_l256_256895

constant initial_students : ℕ := 200 -- Initial number of students who passed three years ago
constant growth_rate : ℝ := 0.5      -- Growth rate of 50%

-- Function to calculate the number of students passing each year
def students_passing (n : ℕ) : ℕ :=
nat.rec_on n initial_students (λ n' ih, ih + (ih / 2))

-- Proposition stating the number of students passing the course this year
theorem passing_students_this_year : students_passing 3 = 675 := sorry

end passing_students_this_year_l256_256895


namespace James_age_after_5_years_l256_256929

def Justin_age : ℕ := 26
def Jessica_age_when_Justin_born : ℕ := 6
def James_age_diff_with_Jessica : ℕ := 7
def years_later : ℕ := 5

theorem James_age_after_5_years : ∀ (Justin_age Jessica_age_when_Justin_born James_age_diff_with_Jessica years_later : ℕ), 
  Justin_age = 26 → 
  Jessica_age_when_Justin_born = 6 → 
  James_age_diff_with_Jessica = 7 → 
  years_later = 5 → 
  (Justin_age + Jessica_age_when_Justin_born + James_age_diff_with_Jessica + years_later - 1 = 44) :=
by
  intros Justin_age Jessica_age_when_Justin_born James_age_diff_with_Jessica years_later hJustin hJessica hJames_diff hYears
  have hJessica_age := Justin_age + Jessica_age_when_Justin_born
  have hJames_age := hJessica_age + James_age_diff_with_Jessica
  have hJames_age_after_5_years := hJames_age + years_later - 1
  exact eq.trans hJames_age_after_5_years
  sorry

end James_age_after_5_years_l256_256929


namespace ellipse_equation_l256_256051

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (e : ℝ) (h3 : e = 1 / 3) 
  (h4 : (\vector (-a) 0).dot (\vector a 0) = -1) : 
  (C : ℝ) : 
  (C = \vector (\frac x^2 9) + \vector (\frac y^2 8) = 1) := 
  sorry

end ellipse_equation_l256_256051


namespace contradiction_even_odd_l256_256226

theorem contradiction_even_odd (a b c : ℕ) (h : (Odd a ∧ Odd b ∧ Odd c) ∨ (Even a ∧ Even b) ∨ (Even a ∧ Even c) ∨ (Even b ∧ Even c)) : 
  ¬ (ExactlyOne (Even a) (Even b) (Even c)) :=
by 
  sorry

end contradiction_even_odd_l256_256226


namespace divisor_is_11_l256_256725

noncomputable def least_subtracted_divisor : Nat := 11

def problem_condition (D : Nat) (x : Nat) : Prop :=
  2000 - x = 1989 ∧ (2000 - x) % D = 0

theorem divisor_is_11 (D : Nat) (x : Nat) (h : problem_condition D x) : D = least_subtracted_divisor :=
by
  sorry

end divisor_is_11_l256_256725


namespace polynomial_irreducible_l256_256946

def f (n : ℕ) : Polynomial ℤ :=
  (Finset.range n).prod (λ i => Polynomial.X ^ 2 + (i + 1)^2) + 1

theorem polynomial_irreducible (n : ℕ) (hn : n > 0) : Irreducible (f n) :=
  sorry

end polynomial_irreducible_l256_256946


namespace geom_seq_m_equals_11_l256_256551

theorem geom_seq_m_equals_11 
  (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 1) 
  (h2 : |q| ≠ 1) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h4 : a m = (a 1) * (a 2) * (a 3) * (a 4) * (a 5)) : 
  m = 11 :=
sorry

end geom_seq_m_equals_11_l256_256551


namespace complex_square_one_plus_i_l256_256066

def complex_square (a b : ℝ) : ℂ := (a + b * complex.I) * (a + b * complex.I)

theorem complex_square_one_plus_i : complex_square 1 1 = 2 * complex.I :=
by
  sorry

end complex_square_one_plus_i_l256_256066


namespace airplane_seats_l256_256773

theorem airplane_seats (s : ℝ)
  (h1 : 0.30 * s = 0.30 * s)
  (h2 : (3 / 5) * s = (3 / 5) * s)
  (h3 : 36 + 0.30 * s + (3 / 5) * s = s) : s = 360 :=
by
  sorry

end airplane_seats_l256_256773


namespace b_n_is_geometric_sum_c_sequence_first_n_terms_l256_256086

variable {n : ℕ}

-- Definitions based on conditions
def a_sequence (n : ℕ) : ℕ 
| 1 := 1
| 2 := 2
| (k + 1) := 3 * a_sequence k - 2 * a_sequence (k - 1)

def b_sequence (n : ℕ) : ℕ := a_sequence (n + 1) - a_sequence n

def c_sequence (n : ℕ) : ℕ := b_sequence n / ((4 * n^2 - 1) * 2^n)

def partial_sum_c_sequence (n : ℕ) : ℕ := (finset.range n).sum c_sequence

-- Prove the sequence b_n is geometric with first term 1 and common ratio 2
theorem b_n_is_geometric : ∃ r : ℕ, b_sequence 1 = 1 ∧ ∀ n : ℕ, b_sequence (n + 1) = r * b_sequence n := 
sorry

-- Prove the sum S_n of the first n terms of c_sequence is equal to n/(4n+2)
theorem sum_c_sequence_first_n_terms (n : ℕ) : partial_sum_c_sequence n = n / (4 * n + 2) :=
sorry

end b_n_is_geometric_sum_c_sequence_first_n_terms_l256_256086


namespace quadratic_function_vertex_and_comparison_l256_256851

theorem quadratic_function_vertex_and_comparison
  (a b c : ℝ)
  (A_conds : 4 * a - 2 * b + c = 9)
  (B_conds : c = 3)
  (C_conds : 16 * a + 4 * b + c = 3) :
  (a = 1/2 ∧ b = -2 ∧ c = 3) ∧
  (∀ (m : ℝ) (y₁ y₂ : ℝ),
     y₁ = 1/2 * m^2 - 2 * m + 3 ∧
     y₂ = 1/2 * (m + 1)^2 - 2 * (m + 1) + 3 →
     (m < 3/2 → y₁ > y₂) ∧
     (m = 3/2 → y₁ = y₂) ∧
     (m > 3/2 → y₁ < y₂)) :=
by
  sorry

end quadratic_function_vertex_and_comparison_l256_256851


namespace students_passing_course_l256_256901

theorem students_passing_course :
  let students_three_years_ago := 200
  let increase_factor := 1.5
  let students_two_years_ago := students_three_years_ago * increase_factor
  let students_last_year := students_two_years_ago * increase_factor
  let students_this_year := students_last_year * increase_factor
  students_this_year = 675 :=
by
  sorry

end students_passing_course_l256_256901


namespace proposition_p_true_l256_256067

def f (x : ℝ) : ℝ := Real.exp x - x

def p : Prop := ∀ x : ℝ, f x > 0

theorem proposition_p_true : p :=
by sorry

end proposition_p_true_l256_256067


namespace greatest_value_of_a_b_c_l256_256446

variables {n : ℕ} (a b c : ℕ)
def A_n (n a: ℕ) : ℕ := a * ((10^n - 1) / 9)
def B_n (n b: ℕ) : ℕ := b * ((10^(2*n) - 1) / 9)
def C_n (n c: ℕ) : ℕ := c * ((10^(3*n) - 1) / 9)

theorem greatest_value_of_a_b_c :
  n > 0 → a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (∃ n1 n2 : ℕ, n1 ≠ n2 ∧ C_n n1 c - A_n n1 a = B_n n1 b * B_n n1 b 
                 ∧ C_n n2 c - A_n n2 a = B_n n2 b * B_n n2 b) →
  a + b + c = 13 :=
begin
  sorry
end

end greatest_value_of_a_b_c_l256_256446


namespace find_m_l256_256826

def z1 (m : ℝ) : ℂ := m + complex.I
def z2 : ℂ := 1 - 2 * complex.I
def ratio (z1 z2 : ℂ) : ℂ := z1 / z2

theorem find_m (m : ℝ) (h : ratio (z1 m) z2 = -1/2) : m = -1/2 :=
sorry

end find_m_l256_256826


namespace common_divisors_count_75_90_l256_256885

def is_divisor (a b : ℤ) : Prop := b % a = 0

theorem common_divisors_count_75_90 : 
  (finset.filter (λ x, is_divisor x 75 ∧ is_divisor x 90) (finset.range 76)).card = 8 :=
by sorry

end common_divisors_count_75_90_l256_256885


namespace molecular_weight_of_BaF2_l256_256721

theorem molecular_weight_of_BaF2 (mw_6_moles : ℕ → ℕ) (h : mw_6_moles 6 = 1050) : mw_6_moles 1 = 175 :=
by
  sorry

end molecular_weight_of_BaF2_l256_256721


namespace fractional_black_area_remains_l256_256774

theorem fractional_black_area_remains (A : ℕ) :
  let change (A : ℚ) := (8/9) * A in
  let initial_black_area := (A : ℚ) in
  let final_black_area := (change^[6]) initial_black_area in
  final_black_area = (262144 / 531441) :=
by
  sorry

end fractional_black_area_remains_l256_256774


namespace largest_number_with_digits_sum_12_l256_256719

def is_digit (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5

def digits_are_valid (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, is_digit d

def sum_of_digits (digits : List ℕ) : ℕ :=
  digits.foldr (· + ·) 0

theorem largest_number_with_digits_sum_12 (digits : List ℕ) :
  digits_are_valid digits → sum_of_digits digits = 12 → 552 ≤ list.to_digits digits :=
sorry

end largest_number_with_digits_sum_12_l256_256719


namespace right_triangle_properties_l256_256536

theorem right_triangle_properties (leg1 leg2 : ℝ) (h_leg1 : leg1 = 30) (h_leg2 : leg2 = 24) :
  let area := (1 / 2) * leg1 * leg2 in
  let hypotenuse := Real.sqrt (leg1^2 + leg2^2) in
  area = 360 ∧ hypotenuse = Real.sqrt 1476 := by
  sorry

end right_triangle_properties_l256_256536


namespace common_points_line_circle_l256_256085

theorem common_points_line_circle (a : ℝ) : 
  (∀ x y: ℝ, (x - 2*y + a = 0) → ((x - 2)^2 + y^2 = 1)) ↔ (-2 - Real.sqrt 5 ≤ a ∧ a ≤ -2 + Real.sqrt 5) :=
by sorry

end common_points_line_circle_l256_256085


namespace two_solutions_exists_l256_256737

theorem two_solutions_exists (a : ℝ) : 
  (a ∈ Iio (-1) ∨ a ∈ Ioi 5) -> 
  ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (a - 5 + |x1 + 1| = 0 ∨ a - x1^2 - 2 * x1 = 0) ∧ (a - 5 + |x2 + 1| = 0 ∨ a - x2^2 - 2 * x2 = 0) :=
by
  sorry

end two_solutions_exists_l256_256737


namespace math_problem_l256_256315

theorem math_problem : (4 + 6 + 7) * 2 - 2 + (3 / 3) = 33 := 
by
  sorry

end math_problem_l256_256315


namespace dot_product_self_l256_256113

variables {ω : Type*} {w : ω → ℝ}

theorem dot_product_self (norm_w : ∥w∥ = 7) : (w • w) = 49 :=
by
  sorry

end dot_product_self_l256_256113


namespace average_value_s7_squared_is_3680_l256_256621

def sum_of_digits_base (b n : ℕ) : ℕ := (n.digits b).sum

noncomputable def average_value_s7_squared : ℕ :=
  let N := 7^20 in
  (∑ n in Finset.range N, (sum_of_digits_base 7 n)^2) / N

theorem average_value_s7_squared_is_3680 :
  average_value_s7_squared = 3680 := 
by
  sorry

end average_value_s7_squared_is_3680_l256_256621


namespace necessary_but_not_sufficient_l256_256478

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (b < -1 → |a| + |b| > 1) ∧ (∃ a b : ℝ, |a| + |b| > 1 ∧ b >= -1) :=
by
  sorry

end necessary_but_not_sufficient_l256_256478


namespace binomial_coefficient_third_term_binomial_sum_coefficients_l256_256258

theorem binomial_coefficient_third_term :
  (binomial 5 2) * (2^2) = 40 :=
by
  sorry

theorem binomial_sum_coefficients :
  (∑ r in Finset.range (5 + 1), binomial 5 r) = 2^5 :=
by
  sorry

end binomial_coefficient_third_term_binomial_sum_coefficients_l256_256258


namespace six_digit_number_consecutive_evens_l256_256434

theorem six_digit_number_consecutive_evens :
  ∃ n : ℕ,
    287232 = (2 * n - 2) * (2 * n) * (2 * n + 2) ∧
    287232 / 100000 = 2 ∧
    287232 % 10 = 2 :=
by
  sorry

end six_digit_number_consecutive_evens_l256_256434


namespace sum_a1_to_a10_l256_256739

def f (x : ℝ) : ℝ := Real.log (2 * x / (1 - x)) / Real.log 2

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then
    1
  else
    ∑ i in Finset.range n \ {0}, f(i / n.toReal)

theorem sum_a1_to_a10 : (∑ i in Finset.range 11 \ {0}, a i) = 46 := sorry

end sum_a1_to_a10_l256_256739


namespace num_values_of_n_l256_256181

-- Definitions of proper positive integer divisors and function g(n)
def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d > 1 ∧ d < n ∧ n % d = 0) (List.range (n + 1))

def g (n : ℕ) : ℕ :=
  List.prod (proper_divisors n)

-- Condition: 2 ≤ n ≤ 100 and n does not divide g(n)
def n_does_not_divide_g (n : ℕ) : Prop :=
  2 ≤ n ∧ n ≤ 100 ∧ ¬ (n ∣ g n)

-- Main theorem statement
theorem num_values_of_n : 
  (Finset.card (Finset.filter n_does_not_divide_g (Finset.range 101))) = 31 :=
by
  sorry

end num_values_of_n_l256_256181


namespace total_texts_sent_is_97_l256_256620

def textsSentOnMondayAllison := 5
def textsSentOnMondayBrittney := 5
def textsSentOnMondayCarol := 5

def textsSentOnTuesdayAllison := 15
def textsSentOnTuesdayBrittney := 10
def textsSentOnTuesdayCarol := 12

def textsSentOnWednesdayAllison := 20
def textsSentOnWednesdayBrittney := 18
def textsSentOnWednesdayCarol := 7

def totalTextsAllison := textsSentOnMondayAllison + textsSentOnTuesdayAllison + textsSentOnWednesdayAllison
def totalTextsBrittney := textsSentOnMondayBrittney + textsSentOnTuesdayBrittney + textsSentOnWednesdayBrittney
def totalTextsCarol := textsSentOnMondayCarol + textsSentOnTuesdayCarol + textsSentOnWednesdayCarol

def totalTextsAllThree := totalTextsAllison + totalTextsBrittney + totalTextsCarol

theorem total_texts_sent_is_97 : totalTextsAllThree = 97 := by
  sorry

end total_texts_sent_is_97_l256_256620


namespace P_Q_H_collinear_l256_256951

variables {A B C H M N P Q : Type} 
(variable [is_trig ABC])
(variable [is_ortho_center H ABC])
(variable (M : segment AB))
(variable (N : segment AC))
(variable (P : circle_intersection (diameter_circle BN)))
(variable (Q : circle_intersection (diameter_circle CM)))

theorem P_Q_H_collinear : collinear P Q H :=
sorry

end P_Q_H_collinear_l256_256951


namespace lilith_additional_fund_l256_256209

theorem lilith_additional_fund
  (num_water_bottles : ℕ)
  (original_price : ℝ)
  (reduced_price : ℝ)
  (expected_difference : ℝ)
  (h1 : num_water_bottles = 5 * 12)
  (h2 : original_price = 2)
  (h3 : reduced_price = 1.85)
  (h4 : expected_difference = 9) :
  (num_water_bottles * original_price) - (num_water_bottles * reduced_price) = expected_difference :=
by
  sorry

end lilith_additional_fund_l256_256209


namespace question1_question2_question3_l256_256729

-- Question 1
theorem question1 (a b m n : ℤ) (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5)^2) :
  a = m^2 + 5 * n^2 ∧ b = 2 * m * n :=
sorry

-- Question 2
theorem question2 (x m n: ℕ) (h : x + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) :
  (m = 1 ∧ n = 2 ∧ x = 13) ∨ (m = 2 ∧ n = 1 ∧ x = 7) :=
sorry

-- Question 3
theorem question3 : Real.sqrt (5 + 2 * Real.sqrt 6) = Real.sqrt 2 + Real.sqrt 3 :=
sorry

end question1_question2_question3_l256_256729


namespace relationship_among_a_b_c_l256_256825

theorem relationship_among_a_b_c :
  let a := -2^(1 - Real.log 3 / Real.log 2)
  let b := 1 - Real.log 3 / Real.log 2
  let c := Real.cos (5 * Real.pi / 6)
  c < a ∧ a < b :=
by
  let a := -2^(1 - Real.log 3 / Real.log 2)
  let b := 1 - Real.log 3 / Real.log 2
  let c := Real.cos (5 * Real.pi / 6)
  sorry

end relationship_among_a_b_c_l256_256825


namespace ratio_a_over_3_to_b_over_2_l256_256122

theorem ratio_a_over_3_to_b_over_2 (a b c : ℝ) (h1 : 2 * a = 3 * b) (h2 : c ≠ 0) (h3 : 3 * a + 2 * b = c) :
  (a / 3) / (b / 2) = 1 :=
sorry

end ratio_a_over_3_to_b_over_2_l256_256122


namespace spherical_to_rectangular_and_cylindrical_l256_256278

theorem spherical_to_rectangular_and_cylindrical :
  let r := 2
  let theta := π / 6
  let phi := π / 3
  let x := r * sin(theta) * cos(phi)
  let y := r * sin(theta) * sin(phi)
  let z := r * cos(theta)
  (x, y, z) = (1/2 : ℝ, (sqrt 3)/2 : ℝ, sqrt 3 : ℝ) ∧ 
  let ρ := sqrt (x^2 + y^2)
  let t := arctan (y / x)
  (ρ, t, z) = (1, π / 3, sqrt 3 : ℝ) :=
by
  sorry

end spherical_to_rectangular_and_cylindrical_l256_256278


namespace max_value_of_A_l256_256829

theorem max_value_of_A (x y z: ℝ) (hx: 0 < x ∧ x ≤ 2) (hy: 0 < y ∧ y ≤ 2) (hz: 0 < z ∧ z ≤ 2) : 
  let A := (x^3 - 6) * (Real.cbrt (x + 6)) + (y^3 - 6) * (Real.cbrt (y + 6)) + (z^3 - 6) * (Real.cbrt (z + 6)) / (x^2 + y^2 + z^2)
  in A ≤ 1 := 
sorry

end max_value_of_A_l256_256829


namespace ball_rebound_travel_distance_l256_256974

theorem ball_rebound_travel_distance (H : ℝ) (n : ℕ) (D : ℝ) :
  H = 100 ∧ D = 250 ∧ (∀ i, 0 < i → i < n → D = H + 2 * (∑ k in range (i - 1), H / 2^k)) → n = 3 :=
by sorry

end ball_rebound_travel_distance_l256_256974


namespace problem1_problem2_l256_256038

-- Define the function f(x) = |x + 2| + |x - 1|
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- 1. Prove the solution set of f(x) > 5 is {x | x < -3 or x > 2}
theorem problem1 : {x : ℝ | f x > 5} = {x : ℝ | x < -3 ∨ x > 2} :=
by
  sorry

-- 2. Prove that if f(x) ≥ a^2 - 2a always holds, then -1 ≤ a ≤ 3
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, f x ≥ a^2 - 2 * a) : -1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end problem1_problem2_l256_256038


namespace area_of_triangle_APQ_proof_l256_256642

noncomputable def area_of_triangle_APQ : ℝ :=
  let ABC := (A B C : Point) := {a : ℝ | a ≤ 10}
  let P := (P : Point) := {p : ℝ | 0 ≤ p ∧ p ≤ 10}
  let Q := (Q : Point) := {q : ℝ | 0 ≤ q ∧ q ≤ 10}
  let pq := (pq : ℝ) := 4
  let x := (x : ℝ) := AP
  let y := (y : ℝ) := AQ
  let XY := (S : ℝ) := xy
  XY = {S : ℝ | (x, y) ∈ ABC × P × Q, pq = 4, S = PQAP.area}
  let area_APQ := (S : ℝ) := 5 / sqrt 3

theorem area_of_triangle_APQ_proof :
  area_of_triangle_APQ = 5 / sqrt 3 := by sorry

end area_of_triangle_APQ_proof_l256_256642


namespace sum_of_coefficients_l256_256459

theorem sum_of_coefficients (n : ℕ) 
  (h1 : ∀ r, (C n 2 = C n 8))
  (h2 : n = 10) :
  (2 + 1)^n = 3^10 :=
by
  sorry

end sum_of_coefficients_l256_256459


namespace count_odd_number_of_ones_l256_256772

/-- 
  There are 520 four-digit numbers formed by the six digits {0, 1, 2, 3, 4, 5},
  where the digits can be repeated, that contain an odd number of 1s.
-/
theorem count_odd_number_of_ones : 
  let digits := {0, 1, 2, 3, 4, 5}
  in let possible_numbers := {x : Fin 6 → Fin 6 // ∃ n : Fin 4, (x n = 1) % 2 = 1 } 
  in possible_numbers.card = 520 :=
by
  sorry   

end count_odd_number_of_ones_l256_256772


namespace area_of_triangle_APQ_proof_l256_256639

noncomputable def area_of_triangle_APQ : ℝ :=
  let ABC := (A B C : Point) := {a : ℝ | a ≤ 10}
  let P := (P : Point) := {p : ℝ | 0 ≤ p ∧ p ≤ 10}
  let Q := (Q : Point) := {q : ℝ | 0 ≤ q ∧ q ≤ 10}
  let pq := (pq : ℝ) := 4
  let x := (x : ℝ) := AP
  let y := (y : ℝ) := AQ
  let XY := (S : ℝ) := xy
  XY = {S : ℝ | (x, y) ∈ ABC × P × Q, pq = 4, S = PQAP.area}
  let area_APQ := (S : ℝ) := 5 / sqrt 3

theorem area_of_triangle_APQ_proof :
  area_of_triangle_APQ = 5 / sqrt 3 := by sorry

end area_of_triangle_APQ_proof_l256_256639


namespace largest_prime_factor_of_6370_l256_256312

theorem largest_prime_factor_of_6370 : ∃ p : ℕ, prime p ∧ (p ∣ 6370) ∧ ∀ q : ℕ, prime q ∧ (q ∣ 6370) → q ≤ p :=
by
  sorry

end largest_prime_factor_of_6370_l256_256312


namespace john_needs_392_tanks_l256_256575

/- Variables representing the conditions -/
def small_balloons : ℕ := 5000
def medium_balloons : ℕ := 5000
def large_balloons : ℕ := 5000

def small_balloon_volume : ℕ := 20
def medium_balloon_volume : ℕ := 30
def large_balloon_volume : ℕ := 50

def helium_tank_capacity : ℕ := 1000
def hydrogen_tank_capacity : ℕ := 1200
def mixture_tank_capacity : ℕ := 1500

/- Mathematical calculations -/
def helium_volume : ℕ := small_balloons * small_balloon_volume
def hydrogen_volume : ℕ := medium_balloons * medium_balloon_volume
def mixture_volume : ℕ := large_balloons * large_balloon_volume

def helium_tanks : ℕ := (helium_volume + helium_tank_capacity - 1) / helium_tank_capacity
def hydrogen_tanks : ℕ := (hydrogen_volume + hydrogen_tank_capacity - 1) / hydrogen_tank_capacity
def mixture_tanks : ℕ := (mixture_volume + mixture_tank_capacity - 1) / mixture_tank_capacity

def total_tanks : ℕ := helium_tanks + hydrogen_tanks + mixture_tanks

theorem john_needs_392_tanks : total_tanks = 392 :=
by {
  -- calculation proof goes here
  sorry
}

end john_needs_392_tanks_l256_256575


namespace smallest_diameter_rope_l256_256331

noncomputable theory

-- Definitions
def mass : ℝ := 20 -- in tons
def number_of_slings : ℝ := 3
def angle_alpha : ℝ := 30 -- in degrees
def safety_factor : ℝ := 6
def max_load_per_thread : ℝ := 10^3 -- in N/mm²
def g : ℝ := 10 -- in m/s²

-- Conversion from tons to kilograms (since 1 ton = 1000 kg)
def mass_kg : ℝ := mass * 1000

-- Total weight in Newtons
def total_weight : ℝ := mass_kg * g

-- Load shared by each strop
def load_per_strop : ℝ := total_weight / number_of_slings

-- Angle in radians
def angle_alpha_rad : ℝ := angle_alpha * (Real.pi / 180)

-- Vertical component considering the angle α
def load_with_angle : ℝ := load_per_strop / Real.cos angle_alpha_rad

-- Required breaking strength considering the safety factor
def required_breaking_strength : ℝ := safety_factor * load_with_angle

-- Convert max load per thread to N/m²
def max_load_per_thread_N_m2 : ℝ := max_load_per_thread * 10^6

-- Necessary cross-sectional area
def required_area : ℝ := required_breaking_strength / max_load_per_thread_N_m2

-- Diameter calculation from cross-sectional area
def min_diameter : ℝ := Real.sqrt ((required_area * 4) / Real.pi)

-- Round up to the nearest whole millimeter
def min_diameter_rounded : ℕ := Int.ceil min_diameter.toReal -- convert to whole mm

-- Proof statement
theorem smallest_diameter_rope : min_diameter_rounded = 26 := 
by sorry

end smallest_diameter_rope_l256_256331


namespace solutions_are__l256_256408

def satisfies_system (x y z : ℝ) : Prop :=
  x^2 * y + y^2 * z = 1040 ∧
  x^2 * z + z^2 * y = 260 ∧
  (x - y) * (y - z) * (z - x) = -540

theorem solutions_are_ (x y z : ℝ) :
  satisfies_system x y z ↔ (x = 16 ∧ y = 4 ∧ z = 1) ∨ (x = 1 ∧ y = 16 ∧ z = 4) :=
by
  sorry

end solutions_are__l256_256408


namespace g_half_eq_neg_one_l256_256131

noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem g_half_eq_neg_one : g (1/2) = -1 := by 
  sorry

end g_half_eq_neg_one_l256_256131


namespace prob_same_side_each_time_in_10_tosses_l256_256727

-- Defining a fair coin and its equal probability condition.
def prob_one_side : ℝ := 1 / 2
def num_tosses : ℕ := 10

-- Defining the probability of landing on the same side for 10 tosses.
theorem prob_same_side_each_time_in_10_tosses : (prob_one_side ^ num_tosses) = 1 / 1024 :=
by
  -- Proof of the theorem is skipped.
  sorry

end prob_same_side_each_time_in_10_tosses_l256_256727


namespace sum_perimeter_area_parallelogram_l256_256804

open Real EuclideanGeometry

theorem sum_perimeter_area_parallelogram :
  ∃ (A B C D : ℝ × ℝ),
    A = (6, 3) ∧
    B = (9, 7) ∧
    C = (2, 0) ∧
    ∥B - A∥ = 5 ∧
    ∥C - A∥ = 5 ∧
    (∃ D,
      (∥D - B∥ = 5 ∧
       ∥D - C∥ = 5 ∧
       (angle A B D = π/2 ∨
        angle A B D ≥ π/2 ∧
        angle A C D ≥ π/2)) ∧
     (let p := 4 * 5
          a := 7 * 4
      in p + a = 48)) sorry

end sum_perimeter_area_parallelogram_l256_256804


namespace circle_proof_l256_256469

noncomputable def circle_center (O : Type*) := O
noncomputable def circle_radius (R : ℝ) := R
noncomputable def inscribed_triangle
  (A B C : Type*) (circumcenter : Type*) := true

noncomputable def heights_of_triangle 
  (A B C H_A H_B H_C : Type*) := true

noncomputable def symmetric_point 
  (P Q : Type*) (_ : Type*) := true

noncomputable def intersection 
  (L1 L2 : Type*) := true

noncomputable def orthocenter 
  (A B C H : Type*) := true

theorem circle_proof 
  {O : Type*} {R : ℝ} {A B C H_A H_B H_C D E P H : Type*}
  (h_circle_center : circle_center O)
  (h_circle_radius : circle_radius R)
  (h_triangle : inscribed_triangle A B C O)
  (h_heights : heights_of_triangle A B C H_A H_B H_C)
  (h_symmetric_D : symmetric_point H_A D H_BH_C)
  (h_symmetric_E : symmetric_point H_B E H_AH_C)
  (h_intersection : intersection (line.from_points A D) (line.from_points B E) = P)
  (h_orthocenter : orthocenter A B C H) :
  dist O P * dist O H = R^2 :=
sorry

end circle_proof_l256_256469


namespace leak_empty_time_l256_256652

theorem leak_empty_time (A L : ℝ) (h1 : A = 1 / 8) (h2 : A - L = 1 / 12) : 1 / L = 24 :=
by
  -- The proof will be provided here
  sorry

end leak_empty_time_l256_256652


namespace find_k_l256_256482

theorem find_k (k : ℝ) :
  (∃ P Q : ℝ × ℝ, 
    (P.1^2 + P.2^2 - 4*P.1 - 4*P.2 + 7 = 0) ∧ 
    (Q.2 = k * Q.1) ∧ 
    (∃ C : ℝ × ℝ, 
      C = (2, 2) ∧ 
      radius(C, P) = 1 ∧ 
      distance(C, Q) = 2*sqrt(2) - 1)) → 
  k = -1 :=
sorry

end find_k_l256_256482


namespace algebra_expression_value_l256_256062

theorem algebra_expression_value (a : ℤ) (h : (2023 - a) ^ 2 + (a - 2022) ^ 2 = 7) :
  (2023 - a) * (a - 2022) = -3 := 
sorry

end algebra_expression_value_l256_256062


namespace median_computation_l256_256547

noncomputable def length_of_median (A B C A1 P Q R : ℝ) : Prop :=
  let AB := 10
  let AC := 6
  let BC := Real.sqrt (AB^2 - AC^2)
  let A1C := 24 / 7
  let A1B := 32 / 7
  let QR := Real.sqrt (A1B^2 - A1C^2)
  let median_length := QR / 2
  median_length = 4 * Real.sqrt 7 / 7

theorem median_computation (A B C A1 P Q R : ℝ) :
  length_of_median A B C A1 P Q R := by
  sorry

end median_computation_l256_256547


namespace collinear_centers_of_bicentric_quad_l256_256237

-- Definition of a bicentric quadrilateral
def BicentricQuad (A B C D I O Z : Point) :=
  InscribedQuad A B C D I ∧ CircumscribedQuad A B C D O ∧ DiagonalsIntersectAt A B C D Z

-- Condition stating that points I, O, Z are collinear
theorem collinear_centers_of_bicentric_quad 
  (A B C D I O Z : Point) 
  (h1 : BicentricQuad A B C D I O Z) 
  (h2 : Brianchon'sTheorem A B C D) :
  Collinear I O Z :=
sorry

end collinear_centers_of_bicentric_quad_l256_256237


namespace line_relationship_l256_256129

variable {α : Type*} [Plane : Set α]

def line_not_in_plane (a : Set α) (α : Set α) : Prop :=
  ¬ (a ⊆ α)

def line_parallel_or_intersects_plane (a : Set α) (α : Set α) : Prop :=
  (∃ l, l ∈ α ∧ ¬ (∃ p, p ∈ a ∧ p ∈ l)) ∨ (∃ p, p ∈ a ∧ p ∈ α)

theorem line_relationship (a α : Set α) (h : line_not_in_plane a α) :
  line_parallel_or_intersects_plane a α :=
  sorry

end line_relationship_l256_256129


namespace reasoning_classification_correct_l256_256820

def analogical_reasoning := "reasoning from specific to specific"
def inductive_reasoning := "reasoning from part to whole and from individual to general"
def deductive_reasoning := "reasoning from general to specific"

theorem reasoning_classification_correct : 
  (analogical_reasoning, inductive_reasoning, deductive_reasoning) =
  ("reasoning from specific to specific", "reasoning from part to whole and from individual to general", "reasoning from general to specific") := 
by 
  sorry

end reasoning_classification_correct_l256_256820


namespace range_of_a_l256_256499

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (x a : ℝ) : ℝ := 2 ^ x - a

theorem range_of_a (a : ℝ) : (∀ x₁ ∈ set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ set.Icc (0 : ℝ) 2, f x₁ > g x₂ a) → a > 1 :=
by
  sorry

end range_of_a_l256_256499


namespace volume_ratio_surface_area_ratio_l256_256232

/-
Given:
1. Rotate a regular hexagon around one of its diagonals passing through the center (Part α):
  - Volume of the resulting solid (K1) = a^3 * π
  - Surface area of the resulting solid (F1) = 3 * √3 * a^2 * π

2. Rotate a regular hexagon around one of its midlines (Part β):
  - Volume of the resulting solid (K2) = 7/6 * a^3 * π * √3
  - Surface area of the resulting solid (F2) = 11/2 * a^2 * π

Prove:
1. Volume ratio: K1 / K2 = (2 * √3) / 7
2. Surface area ratio: F1 / F2 = (6 * √3) / 11
-/

variables (a π : ℝ)

def K1 := a^3 * π
def F1 := 3 * real.sqrt 3 * a^2 * π

def K2 := (7 / 6) * a^3 * π * real.sqrt 3
def F2 := (11 / 2) * a^2 * π

theorem volume_ratio : K1 / K2 = (2 * real.sqrt 3) / 7 :=
by {
  sorry
}

theorem surface_area_ratio : F1 / F2 = (6 * real.sqrt 3) / 11 :=
by {
  sorry
}

end volume_ratio_surface_area_ratio_l256_256232


namespace f1_is_intelligent_quadratic_intelligent_for_any_b_minimum_b_for_symmetric_intelligent_points_l256_256028

section IntelligentFunction

-- Problem (1)
def is_intelligent_function (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, f m = m

def f1 (x : ℝ) : ℝ := 2 * x - 3

theorem f1_is_intelligent : is_intelligent_function f1 :=
sorry

-- Problem (2)
variables {a b c : ℝ} {x1 x2 : ℝ}

def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

def intersects_x_axis (x1 x2 : ℝ) (a : ℝ) : Prop :=
  x1 * x2 + x1 + x2 = -2 / a

theorem quadratic_intelligent_for_any_b (h : intersects_x_axis x1 x2 a) : 0 < a ∧ a ≤ 1 :=
sorry

-- Problem (3)
def symmetric_intelligent_points (C D : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, C.1 + D.1 = 0 ∧ C.2 + D.2 = 2 * k * C.1 + 2 * a / (2 * a^2 + a + 1)

theorem minimum_b_for_symmetric_intelligent_points (h : intersects_x_axis x1 x2 a) 
  (h_sym : symmetric_intelligent_points (3, 3) (3, 3)) : b = 3 / 4 :=
sorry

end IntelligentFunction

end f1_is_intelligent_quadratic_intelligent_for_any_b_minimum_b_for_symmetric_intelligent_points_l256_256028


namespace symmetric_point_reflection_l256_256225

theorem symmetric_point_reflection (x y : ℝ) : (2, -(-5)) = (2, 5) := by
  sorry

end symmetric_point_reflection_l256_256225


namespace dot_product_square_of_norm_l256_256116

variable (w : ℝ^n) (n : ℕ)

theorem dot_product_square_of_norm (h : ‖w‖ = 7) : w • w = 49 :=
by
  sorry

end dot_product_square_of_norm_l256_256116


namespace max_determinant_is_sqrt_233_l256_256941

noncomputable def max_det_of_matrix : ℝ :=
let v := ![3, 2, 0]
let w := ![1, -1, 4]
let u := ((1 : ℝ) / Real.sqrt 233) • ![8, -12, -5] in
  (Matrix.det ![
    u,
    v,
    w
  ])

theorem max_determinant_is_sqrt_233 :
  max_det_of_matrix = Real.sqrt 233 :=
sorry

end max_determinant_is_sqrt_233_l256_256941


namespace probability_of_divisible_by_3_before_not_divisible_l256_256750

noncomputable def probability_rolls (d : List ℕ) : ℚ :=
  let divisible_by_3 := [3, 6]
  let not_divisible_by_3 := [1, 2, 4, 5, 7, 8]
  -- Assuming independence and uniform probability distribution
  let p_div3 := 1 / 4
  let p_not_div3 := 3 / 4
  let p_valid_sequence := (n : ℕ) -> (p_div3 ^ (n-1)) * p_not_div3
  let p_complement := (n : ℕ) -> p_valid_sequence n * (2 / 2^(n-1))
  let p_total := ∑' n, if n >= 3 then p_valid_sequence n - p_complement n else 0
  p_total

theorem probability_of_divisible_by_3_before_not_divisible :
  probability_rolls [1, 2, 3, 4, 5, 6, 7, 8] = (3 / 128) :=
by
  sorry

end probability_of_divisible_by_3_before_not_divisible_l256_256750


namespace rectangle_perimeter_l256_256292

theorem rectangle_perimeter (a b c width : ℕ) (area : ℕ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) 
  (h5 : area = (a * b) / 2) 
  (h6 : width = 5) 
  (h7 : area = width * ((area * 2) / (a * b)))
  : 2 * (width + (area / width)) = 22 := 
by 
  sorry

end rectangle_perimeter_l256_256292


namespace find_t_l256_256124

theorem find_t (s t : ℝ) (h1 : 12 * s + 7 * t = 165) (h2 : s = t + 3) : t = 6.789 := 
by 
  sorry

end find_t_l256_256124


namespace a_plus_b_eq_zero_l256_256962

-- Define the universal set and the relevant sets
def U : Set ℝ := Set.univ
def M (a : ℝ) : Set ℝ := {x | x^2 + a * x ≤ 0}
def C_U_M (b : ℝ) : Set ℝ := {x | x > b ∨ x < 0}

-- Define the proof theorem
theorem a_plus_b_eq_zero (a b : ℝ) (h1 : ∀ x, x ∈ M a ↔ -a < x ∧ x < 0 ∨ 0 < x ∧ x < -a)
                         (h2 : ∀ x, x ∈ C_U_M b ↔ x > b ∨ x < 0) : a + b = 0 := 
sorry

end a_plus_b_eq_zero_l256_256962


namespace smallest_n_with_seven_proper_factors_l256_256577

open Nat

theorem smallest_n_with_seven_proper_factors (n : ℕ) (h1 : (∃ m : ℕ, m > 1 ∧ ∀ d : ℕ, d > 1 ∧ d < m → d ∣ n) ) :
  n = 180 :=
begin
  sorry
end

end smallest_n_with_seven_proper_factors_l256_256577


namespace clock_equiv_4_cubic_l256_256970

theorem clock_equiv_4_cubic :
  ∃ x : ℕ, x > 3 ∧ x % 12 = (x^3) % 12 ∧ (∀ y : ℕ, y > 3 ∧ y % 12 = (y^3) % 12 → x ≤ y) :=
by
  use 4
  sorry

end clock_equiv_4_cubic_l256_256970


namespace painted_cells_solutions_l256_256164

def painted_cells (k l : ℕ) : ℕ := (2 * k + 1) * (2 * l + 1) - 74

theorem painted_cells_solutions : ∃ k l : ℕ, k * l = 74 ∧ (painted_cells k l = 373 ∨ painted_cells k l = 301) :=
by
  sorry

end painted_cells_solutions_l256_256164


namespace f_divisible_by_64_l256_256656

theorem f_divisible_by_64 (n : ℕ) (h : n > 0) : 64 ∣ (3^(2*n + 2) - 8*n - 9) :=
sorry

end f_divisible_by_64_l256_256656


namespace triangle_angle_inequality_l256_256111

theorem triangle_angle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  4 / A + 1 / (B + C) ≥ 9 / Real.pi := by
  sorry

end triangle_angle_inequality_l256_256111


namespace area_between_lines_l256_256798

def line1 (x : ℝ) : ℝ := -1/5 * x + 3
def line2 (x : ℝ) : ℝ := -5/4 * x + 11.5

def integrand (x : ℝ) : ℝ := line2 x - line1 x

theorem area_between_lines :
  ∫ (x : ℝ) in 0..8, integrand x = 40.8 :=
by
  sorry

end area_between_lines_l256_256798


namespace trisha_walked_distance_l256_256622

theorem trisha_walked_distance :
  ∃ x : ℝ, (x + x + 0.67 = 0.89) ∧ (x = 0.11) :=
by sorry

end trisha_walked_distance_l256_256622


namespace sin_beta_l256_256044

open Real

theorem sin_beta {α β : ℝ} (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_cosα : cos α = 2 * sqrt 5 / 5)
  (h_sinαβ : sin (α - β) = -3 / 5) :
  sin β = 2 * sqrt 5 / 5 := 
sorry

end sin_beta_l256_256044


namespace millet_percentage_in_mix_l256_256356

def contribution_millet_brandA (percA mixA : ℝ) := percA * mixA
def contribution_millet_brandB (percB mixB : ℝ) := percB * mixB

theorem millet_percentage_in_mix
  (percA : ℝ) (percB : ℝ) (mixA : ℝ) (mixB : ℝ)
  (h1 : percA = 0.40) (h2 : percB = 0.65) (h3 : mixA = 0.60) (h4 : mixB = 0.40) :
  (contribution_millet_brandA percA mixA + contribution_millet_brandB percB mixB = 0.50) :=
by
  sorry

end millet_percentage_in_mix_l256_256356


namespace sin_cos_values_l256_256477

theorem sin_cos_values (x : ℝ) (h : sin x - 3 * cos x = 2) : sin x + 3 * cos x = 4 ∨ sin x + 3 * cos x = -2 :=
sorry

end sin_cos_values_l256_256477


namespace paint_left_l256_256416

theorem paint_left (dexter_usage_gallons : ℚ) (jay_usage_gallons : ℚ) (gallon_to_liters : ℚ) :
  dexter_usage_gallons = 3 / 8 → 
  jay_usage_gallons = 5 / 8 → 
  gallon_to_liters = 4 →
  let dexter_usage_liters := dexter_usage_gallons * gallon_to_liters in
  let jay_usage_liters := jay_usage_gallons * gallon_to_liters in
  8 - (dexter_usage_liters + jay_usage_liters) = 4 :=
by
  intros h_dexter h_jay h_gallon
  have dexter_usage_l := dexter_usage_gallons * gallon_to_liters
  have jay_usage_l := jay_usage_gallons * gallon_to_liters
  sorry

end paint_left_l256_256416


namespace ellipse_equation_l256_256050

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (e : ℝ) (h3 : e = 1 / 3) 
  (h4 : (\vector (-a) 0).dot (\vector a 0) = -1) : 
  (C : ℝ) : 
  (C = \vector (\frac x^2 9) + \vector (\frac y^2 8) = 1) := 
  sorry

end ellipse_equation_l256_256050


namespace lorelai_jellybeans_l256_256231

variable (Gigi Rory Luke Lane Lorelai : ℕ)
variable (h1 : Gigi = 15)
variable (h2 : Rory = Gigi + 30)
variable (h3 : Luke = 2 * Rory)
variable (h4 : Lane = Gigi + 10)
variable (h5 : Lorelai = 3 * (Gigi + Luke + Lane))

theorem lorelai_jellybeans : Lorelai = 390 := by
  sorry

end lorelai_jellybeans_l256_256231


namespace person_speed_l256_256757

theorem person_speed (distance time : ℝ) (h_distance : distance = 125) (h_time : time = 5) :
    let speed := distance / time in speed = 25 := by
  sorry

end person_speed_l256_256757


namespace trigonometric_relationship_l256_256060

theorem trigonometric_relationship :
  let a := [10, 9, 8, 7, 6, 4, 3, 2, 1]
  let sum_of_a := a.sum
  let x := Real.sin sum_of_a
  let y := Real.cos sum_of_a
  let z := Real.tan sum_of_a
  sum_of_a = 50 →
  z < x ∧ x < y :=
by
  sorry

end trigonometric_relationship_l256_256060


namespace divide_80_into_two_parts_l256_256418

theorem divide_80_into_two_parts :
  ∃ a b : ℕ, a + b = 80 ∧ b / 2 = a + 10 ∧ a = 20 ∧ b = 60 :=
by
  sorry

end divide_80_into_two_parts_l256_256418


namespace cyclist_first_part_distance_l256_256971

theorem cyclist_first_part_distance
  (T₁ T₂ T₃ : ℝ)
  (D : ℝ)
  (h1 : D = 9 * T₁)
  (h2 : T₂ = 12 / 10)
  (h3 : T₃ = (D + 12) / 7.5)
  (h4 : T₁ + T₂ + T₃ = 7.2) : D = 18 := by
  sorry

end cyclist_first_part_distance_l256_256971


namespace total_interest_after_four_years_l256_256383

noncomputable def compoundInterest (principal : ℝ) (rates : List ℝ) (years : ℕ) : ℝ :=
  (List.foldl (λ acc r, acc * (1 + r)) principal (rates.take years))

theorem total_interest_after_four_years :
  let principal := 2000
  let rates := [0.05, 0.06, 0.07, 0.08]
  compoundInterest principal rates 4 - principal = 572.36416 := by
  sorry

end total_interest_after_four_years_l256_256383


namespace sin_alpha_given_cos_alpha_sin_cos_ratio_given_tan_l256_256339

theorem sin_alpha_given_cos_alpha (α : Real) (h₁ : cos α = -4/5) (h₂ : π < α ∧ α < 3 * π / 2) :
  sin α = -3/5 :=
sorry

theorem sin_cos_ratio_given_tan (θ : Real) (h₁ : tan θ = 3) :
  (sin θ + cos θ) / (2 * sin θ + cos θ) = 4/7 :=
sorry

end sin_alpha_given_cos_alpha_sin_cos_ratio_given_tan_l256_256339


namespace find_digit_C_l256_256910

/-- Given deck of 60 distinct cards, number of distinct unordered hands that can be dealt
  is written as 192B000C3210. Identify digit C -/
def digit_C_in_combination : Nat := 3

theorem find_digit_C :
  (Nat.choose 60 12 = 19200003210 ∨ Nat.choose 60 12 = 1920000331210 ∨ Nat.choose 60 12 = 1920000341210 ∨ (·)) -- continue listing any probable matches here
  ∧ (Nat.choose 60 12 = 1920000331210 → digit_C_in_combination = 3) 
∧ (Nat.choose 60 12 = 1920000341210 → digit_C_in_combination = 4)
∧ (Nat.choose 60 12 = 1920000321210 → digit_C_in_combination = 2) 
∧ 
sorry

end find_digit_C_l256_256910


namespace television_total_percent_reduction_l256_256763

def total_percent_reduction (T : ℝ) : ℝ :=
  let first_discount := 0.75 * T
  let second_discount := 0.65 * first_discount
  let final_price := second_discount
  (1 - (final_price / T)) * 100

theorem television_total_percent_reduction :
  ∀ (T : ℝ), total_percent_reduction T = 51.25 := by
  intro T
  have h1 : 0.75 * T = 0.75 * T := by rfl
  have h2 : 0.65 * (0.75 * T) = 0.4875 * T := by norm_num
  have h3 : 1 - (0.4875 * T / T) = 0.5125 := by
    field_simp [ne_of_gt (by norm_num : (T ≠ 0))]
    ring
  calc
    total_percent_reduction T
        = (1 - (0.4875 * T / T)) * 100 : by sorry
    ... = 51.25 : by norm_num

end television_total_percent_reduction_l256_256763


namespace least_possible_perimeter_l256_256136

noncomputable def cos_d : ℝ := 3 / 5
noncomputable def cos_e : ℝ := 12 / 13
noncomputable def cos_f : ℝ := -3 / 5

theorem least_possible_perimeter (D E F : ℝ) (d e f : ℕ) 
  (h_cos_d : cos D = cos_d) (h_cos_e : cos E = cos_e) (h_cos_f : cos F = cos_f) 
  (h_sum_angles : D + E + F = π) 
  (triangle_inequality_1 : d + e > f)
  (triangle_inequality_2 : d + f > e)
  (triangle_inequality_3 : e + f > d) :
  d + e + f = 129 :=
sorry

end least_possible_perimeter_l256_256136


namespace vector_calculations_l256_256875

-- Given vectors a and b
def a : ℝ × ℝ × ℝ := (3, 5, -4)
def b : ℝ × ℝ × ℝ := (2, 1, 8)

-- Definitions of scalar multiplication and vector addition
def smul (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (c * v.1, c * v.2, c * v.3)
def add (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (u.1 + v.1, u.2 + v.2, u.3 + v.3)
def sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (u.1 - v.1, u.2 - v.2, u.3 - v.3)

-- Definition of dot product
def dot (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Proving the vector calculations
theorem vector_calculations :
  sub (smul 3 a) (smul 2 b) = (5, 13, -28) ∧ dot a b = -21 ∧ ∀ λ μ, -4 * λ + 8 * μ = 0 → (3 * λ + 2 * μ, 5 * λ + μ, -4 * λ + 8 * μ) ⋅ (0, 0, 1) = 0 :=
by
  sorry

end vector_calculations_l256_256875


namespace washing_time_per_cycle_l256_256375

theorem washing_time_per_cycle
    (shirts pants sweaters jeans : ℕ)
    (items_per_cycle total_hours : ℕ)
    (h1 : shirts = 18)
    (h2 : pants = 12)
    (h3 : sweaters = 17)
    (h4 : jeans = 13)
    (h5 : items_per_cycle = 15)
    (h6 : total_hours = 3) :
    ((shirts + pants + sweaters + jeans) / items_per_cycle) * (total_hours * 60) / ((shirts + pants + sweaters + jeans) / items_per_cycle) = 45 := 
by
  sorry

end washing_time_per_cycle_l256_256375


namespace find_a_l256_256842

theorem find_a (a : ℤ) (A : Set ℤ) (B : Set ℤ) :
  A = {-2, 3 * a - 1, a^2 - 3} ∧
  B = {a - 2, a - 1, a + 1} ∧
  A ∩ B = {-2} → a = -3 :=
by
  intro H
  sorry

end find_a_l256_256842


namespace point_within_d_units_of_lattice_point_l256_256758

theorem point_within_d_units_of_lattice_point (d : ℝ) :
  (∃ (d : ℝ), ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 1000 ∧ 0 ≤ y ∧ y ≤ 1000 →
  x.int_floor.mod 1 = 0 ∧ y.int_floor.mod 1 = 0 → (x - x.int_floor)^2 + (y - y.int_floor)^2 ≤ d^2 ∧ (x - y.int_floor)^2 + (y - x.int_floor)^2 ≤ d^2) →
  (π * d^2 = 1 / 3) →
  (|d - 0.3| < 0.1) :=
by sorry

end point_within_d_units_of_lattice_point_l256_256758


namespace line_intersects_parabola_at_one_point_l256_256002

theorem line_intersects_parabola_at_one_point (k : ℝ) : (∃ y : ℝ, -y^2 - 4 * y + 2 = k) ↔ k = 6 :=
by 
  sorry

end line_intersects_parabola_at_one_point_l256_256002


namespace angle_C_is_60_degrees_l256_256069

noncomputable def size_of_angle_C : ℚ :=
  let area := 3 * Real.sqrt 3
  let bc := (4 : ℚ)
  let ca := (3 : ℚ)
  let sinC := (Real.sqrt 3) / 2
  if (sinC = (3 * Real.sqrt 3) / (bc * ca)) ∧ (0 < 60) ∧ (60 < 90) then
    60
  else
    sorry

theorem angle_C_is_60_degrees (area : ℚ) (bc : ℚ) (ca : ℚ) :
  area = 3 * Real.sqrt 3 → bc = 4 → ca = 3 → size_of_angle_C = 60 :=
by
  intros area_id bc_id ca_id
  rw [area_id, bc_id, ca_id]
  sorry

end angle_C_is_60_degrees_l256_256069


namespace equilateral_triangle_area_APQ_l256_256643

theorem equilateral_triangle_area_APQ (ABC : Triangle) 
  (h_eq : is_equilateral ABC)
  (h_side : ABC.sides = (10, 10, 10)) 
  (P Q : Point) 
  (hP : P ∈ segment ABC.A ABC.B) 
  (hQ : Q ∈ segment ABC.A ABC.C) 
  (h_tangent : is_tangent (segment P Q) ABC.incircle) 
  (hPQ : segment.length P Q = 4) : 
  area (triangle ABC.A P Q) = 5 / sqrt 3 :=
by 
  sorry

end equilateral_triangle_area_APQ_l256_256643


namespace quadratic_function_n_neg_l256_256819

theorem quadratic_function_n_neg (n : ℝ) :
  (∀ x : ℝ, x^2 + 3 * x + n = 0 → x > 0) → n < 0 :=
by
  sorry

end quadratic_function_n_neg_l256_256819


namespace expected_rolls_in_a_year_l256_256782

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
def is_composite (n : ℕ) : Prop := n = 4 ∨ n = 6
def is_odd_non_prime (n : ℕ) : Prop := n = 1
def is_reroll (n : ℕ) : Prop := n = 8

theorem expected_rolls_in_a_year : 
  let p_prime := 4 / 8 in
  let p_composite := 2 / 8 in
  let p_odd_non_prime := 1 / 8 in
  let p_reroll := 1 / 8 in
  let E := p_prime * 1 + p_composite * 1 + p_odd_non_prime * 1 + p_reroll * (1 + E) in
  E = 1 → 365 * E = 365 := 
by
  intros
  sorry

end expected_rolls_in_a_year_l256_256782


namespace units_digit_of_square_l256_256583

theorem units_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 := 
by 
  sorry

end units_digit_of_square_l256_256583


namespace area_triangle_APQ_l256_256636

/-
  ABC is an equilateral triangle with side length 10.
  Points P and Q are on sides AB and AC respectively.
  Segment PQ is tangent to the incircle of triangle ABC and has length 4.
  Let AP = x and AQ = y, such that x + y = 6 and x^2 + y^2 - xy = 16.
  Prove that the area of triangle APQ is 5 * sqrt(3) / 3.
-/

theorem area_triangle_APQ :
  ∀ (x y : ℝ),
  x + y = 6 ∧ x^2 + y^2 - x * y = 16 → 
  (∃ (S : ℝ), S = (1 / 2) * x * y * (sqrt 3 / 2) ∧ S = 5 * (sqrt 3) / 3) :=
by 
  intro x y,
  intro h,
  sorry

end area_triangle_APQ_l256_256636


namespace quadrilateral_not_necessarily_square_l256_256568

theorem quadrilateral_not_necessarily_square
  (ABCD : Type)
  (A B C D : ABCD)
  (angleA angleB angleC angleD : ℝ)
  (iso_right_triangles : (ABCD → ABCD) → Prop)
  (h1 : angleA = 45)
  (h2 : angleB = 45)
  (h3 : angleC = 90)
  (h4 : angleD = 45)
  (h5 : iso_right_triangles (λabcd, (A, B, C, D))) :
  ∃ (ABCD : Type), ¬(ABCD = square) :=
sorry

end quadrilateral_not_necessarily_square_l256_256568


namespace mo_hot_chocolate_per_rainy_morning_l256_256973

theorem mo_hot_chocolate_per_rainy_morning 
  (total_cups : ℕ)
  (extra_tea_cups : ℕ)
  (rainy_days : ℕ)
  (non_rainy_tea_cups_per_day : ℕ)
  (total_days : ℕ := 7) 
  (non_rainy_days : ℕ := total_days - rainy_days) :
  total_cups = 22 
  → extra_tea_cups = 8
  → rainy_days = 4 
  → non_rainy_tea_cups_per_day = 5
  → (total_cups - 5 * non_rainy_days - extra_tea_cups) / rainy_days = 1.75 :=
by
  intros h_total h_extra h_rainy h_tea_per_day
  -- Lean proof steps become irrelevant as proof is not essential
  sorry

end mo_hot_chocolate_per_rainy_morning_l256_256973


namespace number_of_classes_l256_256801

theorem number_of_classes (sheets_per_class_per_day : ℕ) 
                          (total_sheets_per_week : ℕ) 
                          (school_days_per_week : ℕ) 
                          (h1 : sheets_per_class_per_day = 200) 
                          (h2 : total_sheets_per_week = 9000) 
                          (h3 : school_days_per_week = 5) : 
                          total_sheets_per_week / school_days_per_week / sheets_per_class_per_day = 9 :=
by {
  -- Proof steps would go here
  sorry
}

end number_of_classes_l256_256801


namespace total_hours_correct_skill_hours_correct_l256_256213

def weekly_schedule : List (String × List (String × Float)) :=
  [
    ("Monday", [("Diving and catching drills", 2.0), ("Strength and conditioning", 1.0)]),
    ("Tuesday", [("Goalkeeper-specific training", 2.0), ("Strength and conditioning", 2.0), ("Footwork drills", 1.0)]),
    ("Wednesday", [("Rest day", 0.0)]),
    ("Thursday", [("Footwork drills", 1.0), ("Reaction time exercises", 1.0), ("Aerial ball drills", 1.5)]),
    ("Friday", [("Shot-stopping", 1.5), ("Defensive communication", 1.5), ("Aerial ball drills", 2.0), ("Strength and conditioning", 1.0)]),
    ("Saturday", [("Game simulation", 3.0), ("Endurance training", 3.0)]),
    ("Sunday", [("Rest day", 0.0)])
  ]

def total_practice_hours (schedule : List (String × List (String × Float))) : Float :=
  schedule.foldl (λ acc day => acc + day.snd.foldl (λ acc practice => acc + practice.snd) 0.0) 0.0 * 3

def total_skill_hours (schedule : List (String × List (String × Float))) : List (String × Float) :=
  let skills := schedule.bind (λ day => day.snd)
  let grouped := skills.groupBy Prod.fst
  grouped.map (λ (skill, hours) => (skill, hours.foldl (λ acc practice => acc + practice.snd) 0.0 * 3))

theorem total_hours_correct : total_practice_hours weekly_schedule = 70.5 := by 
  sorry

theorem skill_hours_correct : 
  total_skill_hours weekly_schedule = 
    [("Diving and catching drills", 6.0),
     ("Strength and conditioning", 12.0),
     ("Goalkeeper-specific training", 6.0),
     ("Footwork drills", 6.0),
     ("Reaction time exercises", 3.0),
     ("Aerial ball drills", 10.5),
     ("Shot-stopping", 4.5),
     ("Defensive communication", 4.5),
     ("Game simulation", 9.0),
     ("Endurance training", 9.0)] := by 
  sorry

end total_hours_correct_skill_hours_correct_l256_256213


namespace consecutive_even_product_6digit_l256_256437

theorem consecutive_even_product_6digit :
  ∃ (a b c : ℕ), 
  (a % 2 = 0) ∧ (b = a + 2) ∧ (c = a + 4) ∧ 
  (Nat.digits 10 (a * b * c)).length = 6 ∧ 
  (Nat.digits 10 (a * b * c)).head! = 2 ∧ 
  (Nat.digits 10 (a * b * c)).getLast! = 2 ∧ 
  (a * b * c = 287232) :=
by
  sorry

end consecutive_even_product_6digit_l256_256437


namespace coefficient_x8_in_expansion_l256_256257

theorem coefficient_x8_in_expansion :
  let expr := (x^2 - 1)^2 * (x^3 + (1/x))^4
  ∃ c : ℤ, (c = 10) → coefficient_of_term expr x^8 = c := 
by
  let expr := ((x^2 - 1)^2) * ((x^3 + (1/x))^4)
  let c := 10
  sorry

end coefficient_x8_in_expansion_l256_256257


namespace find_ellipse_equation_l256_256048

noncomputable def ellipse_equation (a b : ℝ) (ecc : ℝ) (dot_product : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧
  ecc = 1 / 3 ∧
  dot_product = -1 →
  (9 : ℝ) = a ^ 2 ∧ (8 : ℝ) = b ^ 2 

theorem find_ellipse_equation :
  ∃ (a b : ℝ), ellipse_equation a b (1/3) (-1) :=
begin
  sorry,
end

end find_ellipse_equation_l256_256048


namespace impossible_to_make_all_numbers_divisible_by_3_l256_256386

theorem impossible_to_make_all_numbers_divisible_by_3
  (polygon: ℕ)
  (vertices: ℕ → ℤ)
  (h1 : polygon = 2018)
  (h2 : (∑ i in Finset.range 2018, if i = 0 then 1 else 0) = 1)
  (h3 : ∀ i, vertices i = if i = 0 then 1 else 0)
  (h4 : ∀ i, vertices (i % 2018) + vertices ((i + 1) % 2018) + k = vertices (i % 2018) + vertices ((i + 1) % 2018) + k)
  :
  ¬ (∀ i, vertices i % 3 = 0) := sorry

end impossible_to_make_all_numbers_divisible_by_3_l256_256386


namespace dot_product_value_l256_256068

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Condition: the magnitudes of the vectors
axiom mag_a : ‖a‖ = 2
axiom mag_b : ‖b‖ = Real.sqrt 3

-- Condition: the angle between vectors
axiom angle_ab : a.angle b = Real.pi * 5 / 6

-- Definition of the desired value
noncomputable def value := a ⬝ (2 • (b - a))

-- Statement of the theorem to prove
theorem dot_product_value : value a b = -14 :=
by
  sorry

end dot_product_value_l256_256068


namespace projection_vector_of_a_onto_b_l256_256096

open Real

noncomputable def a : ℝ × ℝ × ℝ := (1, 3, 0)
noncomputable def b : ℝ × ℝ × ℝ := (2, 1, 1)

theorem projection_vector_of_a_onto_b :
  let dot_product := (a.1 * b.1) + (a.2 * b.2) + (a.3 * b.3)
  let b_magnitude_sq := (b.1 ^ 2) + (b.2 ^ 2) + (b.3 ^ 2)
  let scalar := dot_product / b_magnitude_sq
  let c := (scalar * b.1, scalar * b.2, scalar * b.3)
  c = (5 / 3, 5 / 6, 5 / 6) :=
by
  sorry

end projection_vector_of_a_onto_b_l256_256096


namespace area_triangle_APQ_l256_256635

/-
  ABC is an equilateral triangle with side length 10.
  Points P and Q are on sides AB and AC respectively.
  Segment PQ is tangent to the incircle of triangle ABC and has length 4.
  Let AP = x and AQ = y, such that x + y = 6 and x^2 + y^2 - xy = 16.
  Prove that the area of triangle APQ is 5 * sqrt(3) / 3.
-/

theorem area_triangle_APQ :
  ∀ (x y : ℝ),
  x + y = 6 ∧ x^2 + y^2 - x * y = 16 → 
  (∃ (S : ℝ), S = (1 / 2) * x * y * (sqrt 3 / 2) ∧ S = 5 * (sqrt 3) / 3) :=
by 
  intro x y,
  intro h,
  sorry

end area_triangle_APQ_l256_256635


namespace average_distinct_t_l256_256854

theorem average_distinct_t (t : ℕ) :
  (∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 5 ∧ r₁ * r₂ = t) →
  ∀ t_set : set ℕ, t_set = {4, 6} → (∑ x in t_set, x) / 2 = 5 :=
by
  sorry

end average_distinct_t_l256_256854


namespace arithmetic_sequence_ratio_l256_256892

open Nat

-- Define the arithmetic sequence using a
variable {a d : ℕ} 
-- Define terms a_8 and a_3
def a_8 := a + 7 * d
def a_3 := a + 2 * d

-- The given condition
axiom h : a_8 = 2 * a_3

-- Define sum of first n terms
def S (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

-- Assert the goal
theorem arithmetic_sequence_ratio : S 15 / S 5 = 6 :=
by sorry

end arithmetic_sequence_ratio_l256_256892


namespace loraine_used_20_sticks_l256_256965

noncomputable def loraine_wax_usage (sticks_small: ℕ) (sticks_large: ℕ) (total_sticks: ℕ) : Prop :=
  sticks_small = 12 ∧ total_sticks = 20 ∧ 4 * ((total_sticks - sticks_small) / 4) + sticks_small = total_sticks

theorem loraine_used_20_sticks :
  loraine_wax_usage 12 8 20 :=
by
  have sticks_large := ((20 - 12) / 4) * 4
  split
  . exact rfl
  . split
    . exact rfl
    . rw [sticks_large]
      sorry

end loraine_used_20_sticks_l256_256965


namespace new_parabola_equation_l256_256262

theorem new_parabola_equation (x y : ℝ) : 
  (y = 3 * x ^ 2) → (∃ x', y = 3 * (x' - 2) ^ 2 + 5) :=
by
  assume h : y = 3 * x ^ 2,
  let x' := x + 2,
  use x',
  sorry

end new_parabola_equation_l256_256262


namespace final_water_percentage_l256_256967

theorem final_water_percentage (orig_milk_volume : ℝ) (orig_water_pct : ℝ) (added_pure_milk : ℝ) (final_water_pct : ℝ) : 
  orig_milk_volume = 20 →
  orig_water_pct = 10 →
  added_pure_milk = 20 →
  final_water_pct = (2 / (orig_milk_volume + added_pure_milk)) * 100 →
  final_water_pct = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end final_water_percentage_l256_256967


namespace max_hopping_school_vertices_l256_256541

def is_simple_graph (G : Type) [graph G] : Prop := sorry -- Definition of a simple graph

def is_hopping_school {G : Type} [graph G] (P : list G) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ P.length → degree (P.nth_le (i - 1) sorry) = i

theorem max_hopping_school_vertices {G : Type} [graph G] (n : ℕ) (hG : is_simple_graph G) :
  ∃ (P : list G), is_hopping_school P ∧ P.length = n - 3 := sorry

end max_hopping_school_vertices_l256_256541


namespace three_mathematicians_interact_l256_256703

section conference

variables (n : ℕ)
variables (Country : Type) (Mathematician : Country → Type)
variables [fintype (Mathematician Country)] [fintype Country]
variables [card : Π c, fintype.card (Mathematician c) = n] -- Each country has n mathematicians
variables [interacts : Π (c c' : Country) (m : Mathematician c), fintype (finset (Mathematician c'))]
variables (h_interacts : ∀ (c : Country) (m : Mathematician c), fintype.card (univ.filter (λ x, ¬ ∃ d, ↥(Mathematician d) = x ∧ d = c)) = n + 1)

theorem three_mathematicians_interact (h : ∀ c, 2 < fintype.card (Mathematician c)) :
  ∃ (a b c : Σ c, Mathematician c),
    a.2 ≠ b.2 ∧ b.2 ≠ c.2 ∧ a.2 ≠ c.2 ∧ 
    (a.2 ∈ (interacts a.1 b.1) ∧
    b.2 ∈ (interacts b.1 c.1) ∧
    c.2 ∈ (interacts c.1 a.1)) :=
sorry

end conference

end three_mathematicians_interact_l256_256703


namespace dot_product_square_of_norm_l256_256115

variable (w : ℝ^n) (n : ℕ)

theorem dot_product_square_of_norm (h : ‖w‖ = 7) : w • w = 49 :=
by
  sorry

end dot_product_square_of_norm_l256_256115


namespace cosine_values_count_l256_256512

theorem cosine_values_count (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 360) (h3 : Real.cos x = -0.65) : 
  ∃ (n : ℕ), n = 2 := by
  sorry

end cosine_values_count_l256_256512


namespace intersecting_points_of_curves_l256_256858

theorem intersecting_points_of_curves :
  (∀ x y, (y = 2 * x^3 + x^2 - 5 * x + 2) ∧ (y = 3 * x^2 + 6 * x - 4) → 
   (x = -1 ∧ y = -7) ∨ (x = 3 ∧ y = 41)) := sorry

end intersecting_points_of_curves_l256_256858


namespace problem_l256_256939

def f (x : ℝ) : ℝ := (3 * x + 5) / (x + 3)

theorem problem :
  (∃ P, ∀ x, x ≥ 1 → f x ≤ P) ∧ (∀ P, ∀ x, x ≥ 1 → f x < P ∨ P = 3) ∧
  (∃ q, ∀ x, x ≥ 1 → q ≤ f x ∧ f 1 = q) :=
by
  sorry

end problem_l256_256939


namespace fraction_product_l256_256393

theorem fraction_product :
  (7 / 4) * (8 / 14) * (28 / 16) * (24 / 36) * (49 / 35) * (40 / 25) * (63 / 42) * (32 / 48) = 56 / 25 :=
by sorry

end fraction_product_l256_256393


namespace find_sequence_l256_256872

def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → 
    a (n + 1) = (a n * a (n - 1)) / 
               Real.sqrt (a n^2 + a (n - 1)^2 + 1)

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 5

def sequence_property (F : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = Real.sqrt (1 / (Real.exp (F n * Real.log 10) - 1))

theorem find_sequence (a : ℕ → ℝ) (F : ℕ → ℝ) :
  initial_conditions a →
  recurrence_relation a →
  (∀ n : ℕ, n ≥ 2 →
    F (n + 1) = F n + F (n - 1)) →
  sequence_property F a :=
by
  intros h_initial h_recur h_F
  sorry

end find_sequence_l256_256872


namespace first_part_eq_19_l256_256746

theorem first_part_eq_19 (x y : ℕ) (h1 : x + y = 36) (h2 : 8 * x + 3 * y = 203) : x = 19 :=
by sorry

end first_part_eq_19_l256_256746


namespace algebraic_simplification_l256_256519

theorem algebraic_simplification (m x : ℝ) (h₀ : 0 < m) (h₁ : m < 10) (h₂ : m ≤ x) (h₃ : x ≤ 10) : 
  |x - m| + |x - 10| + |x - m - 10| = 20 - x :=
by
  sorry

end algebraic_simplification_l256_256519


namespace min_max_sum_of_distances_l256_256218

open Real

theorem min_max_sum_of_distances :
  ∃ (A B C A0 B0 C0 A1 B1 C1 A2 B2 C2 A3 B3 C3 : ℝ×ℝ), 
  let AB0 := dist A0 B0 in 
  let BC0 := dist B0 C0 in 
  let CA0 := dist C0 A0 in 
  A0 != B0 ∧ B0 != C0 ∧ C0 != A0 ∧
  AB0 + BC0 + CA0 = 1 ∧
  let AB1 := dist A1 B1 in 
  let BC1 := dist B1 C1 in 
  AB1 = AB0 ∧ BC1 = BC0 ∧
  let AB2 := dist A2 B2 in 
  let BC2 := dist B2 C2 in 
  let AC2 := dist C2 A2 in 
  AB2 = AB1 ∧ BC2 = BC1 ∧ AC2 = AB1 ∧
  let AB3 := dist A3 B3 in 
  let BC3 := dist B3 C3 in 
  let CA3 := dist C3 A3 in 
  AB3 = AB2 ∧ BC3 = BC2 ∧
  (1/3 : ℝ) ≤ AB3 + BC3 + CA3 ∧ AB3 + BC3 + CA3 ≤ 3 :=
begin
  sorry
end

end min_max_sum_of_distances_l256_256218


namespace no_valid_n_l256_256448

theorem no_valid_n : ¬ ∃ (n : ℕ), (n > 0) ∧ (100 ≤ n / 4) ∧ (n / 4 ≤ 999) ∧ (100 ≤ 4 * n) ∧ (4 * n ≤ 999) :=
by {
  sorry
}

end no_valid_n_l256_256448


namespace determine_numbers_l256_256201

theorem determine_numbers (n : ℕ) (m : ℕ) (x y z u v : ℕ) (h₁ : 10000 <= n ∧ n < 100000)
(h₂ : n = 10000 * x + 1000 * y + 100 * z + 10 * u + v)
(h₃ : m = 1000 * x + 100 * y + 10 * u + v)
(h₄ : x ≠ 0)
(h₅ : n % m = 0) :
∃ a : ℕ, (10 <= a ∧ a <= 99 ∧ n = a * 1000) :=
sorry

end determine_numbers_l256_256201


namespace my_op_comm_my_op_not_assoc_my_op_comm_but_not_assoc_l256_256449

-- Define the operation for the given k
def my_op (k : ℝ) (x y : ℝ) : ℝ := (k * x * y) / (x + y)

-- Commutativity
theorem my_op_comm (k : ℝ) (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  my_op k x y = my_op k y x :=
by sorry

-- Non-associativity
theorem my_op_not_assoc (k : ℝ) (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0):
  my_op k (my_op k x y) z ≠ my_op k x (my_op k y z) :=
by sorry

-- Combined statement of commutativity and non-associativity
theorem my_op_comm_but_not_assoc (k : ℝ) (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (my_op k x y = my_op k y x) ∧ (my_op k (my_op k x y) z ≠ my_op k x (my_op k y z)) :=
by
  exact ⟨my_op_comm k x y hx hy, my_op_not_assoc k x y z hx hy hz⟩

end my_op_comm_my_op_not_assoc_my_op_comm_but_not_assoc_l256_256449


namespace coefficient_m5n5_in_mn_pow10_l256_256309

-- Definition of the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- The main theorem statement
theorem coefficient_m5n5_in_mn_pow10 : 
  (∃ c, (m + n) ^ 10 = c * m^5 * n^5 + ∑ (k ≠ 5), (binomial_coeff 10 k) * m^(10 - k) * n^k) → 
  c = 252 := 
by 
  sorry

end coefficient_m5n5_in_mn_pow10_l256_256309


namespace gena_hits_target_l256_256456

-- Definitions from the problem conditions
def initial_shots : ℕ := 5
def total_shots : ℕ := 17
def shots_per_hit : ℕ := 2

-- Mathematical equivalent proof statement
theorem gena_hits_target (G : ℕ) (H : G * shots_per_hit + initial_shots = total_shots) : G = 6 :=
by
  sorry

end gena_hits_target_l256_256456


namespace sin_diff_identity_l256_256322

theorem sin_diff_identity : sin(57 * real.pi / 180) * cos(27 * real.pi / 180) - cos(57 * real.pi / 180) * sin(27 * real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_diff_identity_l256_256322


namespace find_ellipse_equation_l256_256047

noncomputable def ellipse_equation (a b : ℝ) (ecc : ℝ) (dot_product : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧
  ecc = 1 / 3 ∧
  dot_product = -1 →
  (9 : ℝ) = a ^ 2 ∧ (8 : ℝ) = b ^ 2 

theorem find_ellipse_equation :
  ∃ (a b : ℝ), ellipse_equation a b (1/3) (-1) :=
begin
  sorry,
end

end find_ellipse_equation_l256_256047


namespace remainder_of_sum_of_squares_l256_256039

  open Nat

  def N (primes : Fin 98 → ℕ) : ℕ :=
    ∑ i, (primes i) ^ 2

  theorem remainder_of_sum_of_squares (primes : Fin 98 → ℕ) (h : ∀ i, Prime (primes i)) :
    (N primes) % 3 = 1 ∨ (N primes) % 3 = 2 :=
  by
    sorry
  
end remainder_of_sum_of_squares_l256_256039


namespace has_only_one_minimum_point_and_no_maximum_point_l256_256205

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3

theorem has_only_one_minimum_point_and_no_maximum_point :
  ∃! c : ℝ, (deriv f c = 0 ∧ ∀ x < c, deriv f x < 0 ∧ ∀ x > c, deriv f x > 0) ∧
  ∀ x, f x ≥ f c ∧ (∀ x, deriv f x > 0 ∨ deriv f x < 0) := sorry

end has_only_one_minimum_point_and_no_maximum_point_l256_256205


namespace a_minus_b_l256_256464

theorem a_minus_b (a b : ℝ) 
  (h1 : real.cbrt a - real.cbrt b = 12)
  (h2 : a * b = ( (a + b + 8) / 6 ) ^ 3) : 
  a - b = 468 := 
sorry

end a_minus_b_l256_256464


namespace find_y_l256_256243

-- Define the variables and their properties
variables {A B C : ℝ} {y : ℝ}

-- Given conditions
def condition1 := A = B + C
def condition2 := B > C
def condition3 := C > 0
def condition4 := B = C + (y / 100) * C

-- Theorem to prove
theorem find_y (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : y = 100 * (B - C) / C :=
by { sorry }

end find_y_l256_256243


namespace product_number_and_sum_possible_values_f_2_eq_8_l256_256599

noncomputable def f : ℝ → ℝ := sorry -- Exact function definition so as not found within conditions.

theorem product_number_and_sum_possible_values_f_2_eq_8
    (k : ℝ)
    (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + k * f x * y) :
    let values := {v | ∃ x, f x = v},
        n := values.to_finset.card,
        s := values.to_finset.sum (λ x, x)
    in n * s = 8 :=
sorry

end product_number_and_sum_possible_values_f_2_eq_8_l256_256599


namespace addition_base10_to_base5_l256_256724

theorem addition_base10_to_base5 (a b sum_in_base10 sum_in_base5 : ℕ) :
  a = 34 → b = 47 → sum_in_base10 = 81 → sum_in_base5 = 311 →
  a + b = sum_in_base10 ∧ sum_in_base10_base5 = sum_in_base5 :=
by {
  intros,
  have h1 : a + b = 81 := by simp [*],
  have h2 : sum_in_base10_base5 = 311 := by sorry, -- Need to apply base conversion
  exact ⟨h1, h2⟩
}

end addition_base10_to_base5_l256_256724


namespace problem_solution_l256_256810

noncomputable def solve_system : List (ℝ × ℝ × ℝ) :=
[(0, 1, -2), (-3/2, 5/2, -1/2)]

theorem problem_solution (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h1 : x^2 + y^2 = -x + 3*y + z)
  (h2 : y^2 + z^2 = x + 3*y - z)
  (h3 : z^2 + x^2 = 2*x + 2*y - z) :
  (x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2) :=
sorry

end problem_solution_l256_256810


namespace k_values_for_unit_vector_cross_product_l256_256940

theorem k_values_for_unit_vector_cross_product (a b c : ℝ^3)
  (unit_a : ‖a‖ = 1)
  (unit_b : ‖b‖ = 1)
  (unit_c : ‖c‖ = 1)
  (orth_a_b : a ⋅ b = 0)
  (orth_a_c : a ⋅ c = 0)
  (angle_bc : real.angle b c = π / 3) :
  ∃ (k : ℝ), a = k • (b × c) ∧ (k = 2 / real.sqrt 3 ∨ k = -2 / real.sqrt 3) :=
begin
  sorry
end

end k_values_for_unit_vector_cross_product_l256_256940


namespace find_M_l256_256953

theorem find_M (A M C : ℕ) (h1 : (100 * A + 10 * M + C) * (A + M + C) = 2040)
(h2 : (A + M + C) % 2 = 0)
(h3 : A ≤ 9) (h4 : M ≤ 9) (h5 : C ≤ 9) :
  M = 7 := 
sorry

end find_M_l256_256953


namespace correct_conclusion_numbers_l256_256771

theorem correct_conclusion_numbers:
  (is_subset : ∅ ⊆ ∅) ∧ ¬(0 ∈ ∅) ∧ (∅ ⊆ {0}) ∧ ¬({0} = ∅) ∧ (is_superset : ∀ (x : Prop), x ∈ ∅ → x ∈ {0}) := 
by sorry

end correct_conclusion_numbers_l256_256771


namespace initial_cards_collected_l256_256659

  -- Ralph collects some cards.
  variable (initial_cards: ℕ)

  -- Ralph's father gives Ralph 8 more cards.
  variable (added_cards: ℕ := 8)

  -- Now Ralph has 12 cards.
  variable (total_cards: ℕ := 12)

  -- Proof statement: Prove that the initial number of cards Ralph collected plus 8 equals 12.
  theorem initial_cards_collected: initial_cards + added_cards = total_cards := by
    sorry
  
end initial_cards_collected_l256_256659


namespace flour_already_in_is_2_l256_256212

-- Define the total amount of flour required by the recipe
def required_flour : ℕ := 7
-- Define the amount of additional flour Mary needs to add
def additional_flour_needed : ℕ := 5
-- Define the amount of flour Mary already put in
def flour_already_in : ℕ := required_flour - additional_flour_needed

-- Prove that the amount of flour Mary already put in is 2
theorem flour_already_in_is_2 : flour_already_in = 2 := 
by
  rw [flour_already_in]
  rw [required_flour, additional_flour_needed]
  norm_num
  sorry

end flour_already_in_is_2_l256_256212


namespace pieces_per_sister_l256_256029

-- Defining the initial conditions
def initial_cake_pieces : ℕ := 240
def percentage_eaten : ℕ := 60
def number_of_sisters : ℕ := 3

-- Defining the statements to be proved
theorem pieces_per_sister (initial_cake_pieces : ℕ) (percentage_eaten : ℕ) (number_of_sisters : ℕ) :
  let pieces_eaten := (percentage_eaten * initial_cake_pieces) / 100
  let remaining_pieces := initial_cake_pieces - pieces_eaten
  let pieces_per_sister := remaining_pieces / number_of_sisters
  pieces_per_sister = 32 :=
by 
  sorry

end pieces_per_sister_l256_256029


namespace find_range_k_l256_256936

-- Definition of ellipse and parabola with given conditions
variable (a b : ℝ)
variable h₁ : a > b > 0
variable h₂ : a = 2 ∧ e = (Real.sqrt 3) / 2 ∧ b^2 = a^2 - (Real.sqrt 3)^2

-- Define equations of ellipse and parabola
def ellipse_eq := ∀ x y : ℝ, (x^2) / (a^2) + y^2 = 1 ↔ (x^2) / 4 + y^2 = 1
def parabola_eq := ∀ x y : ℝ, x^2 = -a * y ↔ x^2 = -2 * y

-- Define the meeting conditions for the line with slope k
variable (k : ℝ)
variable line_eq : (x^2 / 4 + (k * x + 2)^2 = 1)

-- We need to prove the range of k given the conditions
theorem find_range_k (h : a > b > 0 ∧ e = (Real.sqrt 3) / 2 ∧ a = 2 ∧ y = -2 * x ∧ line_eq) :
(−2 < k ∧ k < −(Real.sqrt 3) / 2) ∨ (k > (Real.sqrt 3) / 2 ∧ k < 2) :=
sorry

end find_range_k_l256_256936


namespace tan_22_5_decomposition_l256_256688

theorem tan_22_5_decomposition :
  ∃ (a b c d : ℕ),
    a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
    tan (22.5 * Real.pi / 180) = Real.sqrt a - Real.sqrt b - Real.sqrt c + d ∧
    a + b + c + d = 5 := by
  sorry

end tan_22_5_decomposition_l256_256688


namespace part1_part2_l256_256841

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x^2 ≤ 5 * x - 4
def q (x a : ℝ) : Prop := x^2 - (a + 2) * x + 2 * a ≤ 0

-- Theorem statement for part (1)
theorem part1 (x : ℝ) (h : p x) : 1 ≤ x ∧ x ≤ 4 := 
by sorry

-- Theorem statement for part (2)
theorem part2 (a : ℝ) : 
  (∀ x, p x → q x a) ∧ (∃ x, p x) ∧ ¬ (∀ x, q x a → p x) → 1 ≤ a ∧ a ≤ 4 := 
by sorry

end part1_part2_l256_256841


namespace leo_trousers_count_l256_256580

theorem leo_trousers_count (S T : ℕ) (h1 : 5 * S + 9 * T = 140) (h2 : S = 10) : T = 10 :=
by
  sorry

end leo_trousers_count_l256_256580


namespace calculate_star_l256_256785

def star (a b : ℝ) : ℝ := (a + b) / (a - b)

theorem calculate_star :
  star (star (star 3 5) 2) 7 = -11 / 10 :=
by
  -- Here you would provide the proof steps
  sorry

end calculate_star_l256_256785


namespace find_X_l256_256027

noncomputable def X : Real := 2.5

theorem find_X : 1.5 * ((3.6 * 0.48 * X) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002 :=
by
  -- Place the steps that simplify the proof here (if needed)
  , sorry

end find_X_l256_256027


namespace exist_polynomials_l256_256601

noncomputable def alpha : ℂ := sorry -- represent the complex (2n+1)th root of unity

theorem exist_polynomials (n : ℕ) :
  ∃ p q : polynomial ℤ, 
  (p (alpha^(2 * n + 1)))^2 + (q (alpha^(2 * n + 1)))^2 = -1 :=
sorry

end exist_polynomials_l256_256601


namespace b_n_formula_c_n_sum_formula_l256_256467

def a_seq (n : ℕ) : ℕ
| 1     := 1
| 2     := 2
| (n+3) := a_seq n + 2 * 3^n

def b_seq (n : ℕ) : ℕ := a_seq n + a_seq (n + 1)

theorem b_n_formula (n : ℕ) : b_seq n = 3^n := 
sorry

def c_seq (n : ℕ) : ℝ := (4 * (n + 1)) / ((4 * n^2 - 1) * 3^n)

theorem c_n_sum_formula (n : ℕ) : 
  (finset.range n).sum c_seq = 1 - 1 / ((2 * n + 1) * 3^n) :=
sorry

end b_n_formula_c_n_sum_formula_l256_256467


namespace base_seven_to_ten_l256_256717

theorem base_seven_to_ten : 
  (7 * 7^4 + 6 * 7^3 + 5 * 7^2 + 4 * 7^1 + 3 * 7^0) = 19141 := 
by 
  sorry

end base_seven_to_ten_l256_256717


namespace problem1_problem2_l256_256340

theorem problem1 (x : ℚ) (h : x - 2/11 = -1/3) : x = -5/33 :=
sorry

theorem problem2 : -2 - (-1/3 + 1/2) = -13/6 :=
sorry

end problem1_problem2_l256_256340


namespace time_difference_l256_256137

-- Definitions of speeds and distance
def distance : Nat := 12
def alice_speed : Nat := 7
def bob_speed : Nat := 9

-- Calculations of total times based on speeds and distance
def alice_time : Nat := alice_speed * distance
def bob_time : Nat := bob_speed * distance

-- Statement of the problem
theorem time_difference : bob_time - alice_time = 24 := by
  sorry

end time_difference_l256_256137


namespace rope_diameter_is_26mm_l256_256330

noncomputable def smallest_rope_diameter (M : ℝ) (n : ℕ) (alpha : ℝ) (k : ℝ) (q : ℝ) (g : ℝ) : ℝ :=
let W := M * g in
let F_strop := W / n in
let N := F_strop / real.cos (alpha * real.pi / 180) in
let Q := k * N in
let S := Q / q in
let A := (real.pi * 4 * S) in
let D := real.sqrt (A / real.pi) * 1000 in
real.ceil D

theorem rope_diameter_is_26mm : smallest_rope_diameter 20 3 30 6 (10^3) 10 = 26 :=
by
  sorry

end rope_diameter_is_26mm_l256_256330


namespace Erika_chalk_original_l256_256419

theorem Erika_chalk_original (C : ℕ) :
  let num_siblings := 3,
      num_friends := 7,
      num_total := 1 + num_siblings + num_friends, -- Erika is 1
      lost_chalk := 3,
      added_chalk := 20,
      pieces_per_person := 5 in
  (C - lost_chalk + added_chalk = num_total * pieces_per_person) → C = 38 :=
by
  intro h
  sorry

end Erika_chalk_original_l256_256419


namespace shirts_sold_l256_256992

theorem shirts_sold (pants shorts shirts jackets credit_remaining : ℕ) 
  (price_shirt1 price_shirt2 price_pants : ℕ) 
  (discount tax : ℝ) :
  (pants = 3) →
  (shorts = 5) →
  (jackets = 2) →
  (price_shirt1 = 10) →
  (price_shirt2 = 12) →
  (price_pants = 15) →
  (discount = 0.10) →
  (tax = 0.05) →
  (credit_remaining = 25) →
  (store_credit : ℕ) →
  (store_credit = pants * 5 + shorts * 3 + jackets * 7 + shirts * 4) →
  (total_cost : ℝ) →
  (total_cost = (price_shirt1 + price_shirt2 + price_pants) * (1 - discount) * (1 + tax)) →
  (total_store_credit_used : ℝ) →
  (total_store_credit_used = total_cost - credit_remaining) →
  (initial_credit : ℝ) →
  (initial_credit = total_store_credit_used + (pants * 5 + shorts * 3 + jackets * 7)) →
  shirts = 2 :=
by
  intros
  sorry

end shirts_sold_l256_256992


namespace max_value_of_A_l256_256830

theorem max_value_of_A (x y z: ℝ) (hx: 0 < x ∧ x ≤ 2) (hy: 0 < y ∧ y ≤ 2) (hz: 0 < z ∧ z ≤ 2) : 
  let A := (x^3 - 6) * (Real.cbrt (x + 6)) + (y^3 - 6) * (Real.cbrt (y + 6)) + (z^3 - 6) * (Real.cbrt (z + 6)) / (x^2 + y^2 + z^2)
  in A ≤ 1 := 
sorry

end max_value_of_A_l256_256830


namespace train_length_l256_256732

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_m : ℝ) : 
  speed_kmph = 60 → time_sec = 9 → length_m = ((60 * (1000 / 3600)) * 9) → length_m = 150.03 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end train_length_l256_256732


namespace exists_n_pretty_polynomial_min_degree_n_pretty_l256_256834

-- Define n-pretty polynomial
def is_n_pretty (P : ℝ → ℝ) (n : ℕ) : Prop :=
  (n ∈ ℕ) ∧ (∃ x : ℝ, P (⌊x⌋) = ⌊P(x)⌋) ∧ (Fintype.card {x : ℝ | P (⌊x⌋) = ⌊P x⌋}.toFinite ⟨n⟩)

theorem exists_n_pretty_polynomial (n : ℕ) (hn : 0 < n) : 
  ∃ P : ℝ → ℝ, is_n_pretty P n :=
by sorry

theorem min_degree_n_pretty (P : ℝ → ℝ) (n : ℕ) (hn : 0 < n) (hp : is_n_pretty P n) : 
  ∃ d : ℕ, polynomial.degree P ≥ (2 * n + 1) / 3 :=
by sorry

end exists_n_pretty_polynomial_min_degree_n_pretty_l256_256834


namespace number_of_n_not_dividing_g_in_range_l256_256189

def g (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ x, x ≠ n ∧ x ∣ n) (finset.range (n+1))), d

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem number_of_n_not_dividing_g_in_range :
  (finset.filter (λ n, n ∉ (finset.range (101)).filter (λ n, n ∣ g n))
  (finset.Icc 2 100)).card = 29 :=
by
  sorry

end number_of_n_not_dividing_g_in_range_l256_256189


namespace find_m_value_l256_256261

theorem find_m_value (m : ℤ) (h1 : m - 2 ≠ 0) (h2 : |m| = 2) : m = -2 :=
by {
  sorry
}

end find_m_value_l256_256261


namespace area_triangle_APQ_l256_256628

def equilateral_triangle (A B C P Q : Type) [metric_space A] :=
  ∃ (a b c : A) (R T : A → A) (x y : ℝ), 
    dist a b = 10 ∧
    dist a c = 10 ∧
    dist b c = 10 ∧
    dist P Q = 4 ∧
    x = dist a P ∧
    y = dist a Q ∧
    (R a = P) ∧
    (T a = Q) ∧
    (x + y = 6) ∧
    (x^2 + y^2 - x * y = 16)

theorem area_triangle_APQ (A B C P Q : Type) [metric_space A] : 
  ∀ (x y : ℝ), 
    equilateral_triangle A B C P Q → 
    x = dist A P → 
    y = dist A Q → 
    (x * y = 20 / 3) → 
    let s := real.sqrt 3 / 2 in 
    (1 / 2 * x * y * s = 5 * real.sqrt 3 / 3) :=
by simp [equilateral_triangle]

end area_triangle_APQ_l256_256628


namespace magnitude_a_plus_2b_l256_256878

open Real

variable {a b : ℝ × ℝ} -- Define vectors 'a' and 'b' in ℝ x ℝ

-- Conditions
def condition1 : |a| = 1 := sorry
def condition2 : |b| = 2 := sorry
def condition3 : a - b = (sqrt 2, sqrt 3) := sorry

-- Goal
theorem magnitude_a_plus_2b (h₁ : |a| = 1) (h₂ : |b| = 2) (h₃ : a - b = (sqrt 2, sqrt 3)) : 
  |a + 2 • b| = sqrt 17 :=
sorry

end magnitude_a_plus_2b_l256_256878


namespace Ma_Xiaohu_speed_l256_256966

theorem Ma_Xiaohu_speed
  (distance_home_school : ℕ := 1800)
  (distance_to_school : ℕ := 1600)
  (father_speed_factor : ℕ := 2)
  (time_difference : ℕ := 10)
  (x : ℕ)
  (hx : distance_home_school - distance_to_school = 200)
  (hspeed : father_speed_factor * x = 2 * x)
  :
  (distance_to_school / x) - (distance_to_school / (2 * x)) = time_difference ↔ x = 80 :=
by
  sorry

end Ma_Xiaohu_speed_l256_256966


namespace student_ticket_price_l256_256289

theorem student_ticket_price :
  ∃ (x : ℝ), (850 * 9 + 1100 * x = 10500) ∧ x = 2.59 :=
begin
  use 2.59,
  split,
  { norm_num, },
  { norm_num, }
end

end student_ticket_price_l256_256289


namespace max_gcd_value_l256_256384

theorem max_gcd_value (n : ℕ) (hn : 0 < n) : ∃ k, k = gcd (13 * n + 4) (8 * n + 3) ∧ k <= 7 := sorry

end max_gcd_value_l256_256384


namespace period_f_max_m_range_k_l256_256079

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x / 2) * cos (x / 2) + 2 * sqrt 3 * cos (x / 2) ^ 2 - sqrt 3
noncomputable def g (x : ℝ) : ℝ := f (π / 2 - x)
noncomputable def h (x : ℝ) : ℝ := g (x / 2 - π / 12)

theorem period_f :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
begin
  use 2 * π,
  intros x,
  sorry
end

theorem max_m :
  ∃ m ∈ ℤ, ∀ x ∈ Icc (-π / 6) (π / 3), abs (f x - m) ≤ 3 :=
begin
  use 4,
  intros x hx,
  sorry
end

theorem range_k :
  ∃ k : ℝ, ∀ x ∈ Icc (-π / 12) (5 * π / 12), (1 / 2) * h x = k * (sin x + cos x) :=
begin
  use Icc (- sqrt 2 / 2) (sqrt 2 / 2),
  intros x hx,
  sorry
end

end period_f_max_m_range_k_l256_256079


namespace students_passed_this_year_l256_256896

theorem students_passed_this_year
  (initial_students : ℕ)
  (annual_increase_rate : ℝ)
  (years_lapsed : ℕ)
  (current_students : ℕ)
  (h_initial : initial_students = 200)
  (h_rate : annual_increase_rate = 1.5)
  (h_years : years_lapsed = 3)
  (h_calc : current_students = (λ n, initial_students * (annual_increase_rate ^ n)) years_lapsed) :
  current_students = 675 :=
begin
  sorry
end

end students_passed_this_year_l256_256896


namespace valid_configuration_exists_l256_256552

noncomputable def unique_digits (digits: List ℕ) := (digits.length = List.length (List.eraseDup digits)) ∧ ∀ (d : ℕ), d ∈ digits ↔ d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem valid_configuration_exists :
  ∃ a b c d e f g h i j : ℕ,
  unique_digits [a, b, c, d, e, f, g, h, i, j] ∧
  a * (100 * b + 10 * c + d) * (100 * e + 10 * f + g) = 1000 * h + 100 * i + 10 * 9 + 71 := 
by
  sorry

end valid_configuration_exists_l256_256552


namespace gcd_of_three_digit_palindromes_l256_256718

def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 101 * a + 10 * b

theorem gcd_of_three_digit_palindromes :
  ∀ n, is_palindrome n → Nat.gcd n 1 = 1 := by
  sorry

end gcd_of_three_digit_palindromes_l256_256718


namespace find_eighth_number_l256_256253

theorem find_eighth_number (x : ℕ) (h1 : (1 + 2 + 4 + 5 + 6 + 9 + 9 + x + 12) / 9 = 7) : x = 27 :=
sorry

end find_eighth_number_l256_256253


namespace can_derive_one_fifth_all_rationals_between_zero_and_one_l256_256217

-- Define the initial set of numbers as {0, 1}
def initial_set : Set ℚ := {0, 1}

-- Define the rule for deriving new numbers based on the arithmetic mean
def arithmetic_mean_rule (s : Set ℚ) (x y : ℚ) : ℚ := (x + y) / 2

-- Define the derivable set of numbers based on initial set and rules
def derivable_numbers : Set ℚ := 
  {x | 
    ∃ (s : Set ℚ), s ⊆ initial_set ∧ 
      (∀ y z ∈ s, arithmetic_mean_rule s y z ∉ s → s ∪ {arithmetic_mean_rule s y z} = s ∪ {x})}

-- Proof statements
theorem can_derive_one_fifth : (1 / 5) ∈ derivable_numbers :=
sorry

theorem all_rationals_between_zero_and_one : ∀ (x : ℚ), (0 < x ∧ x < 1) → x ∈ derivable_numbers :=
sorry

end can_derive_one_fifth_all_rationals_between_zero_and_one_l256_256217


namespace line_through_fixed_point_circle_center_coordinates_no_real_k_for_tangent_chord_length_not_two_l256_256084

theorem line_through_fixed_point (k : ℝ) :
  ∀ x : ℝ, ∀ y : ℝ, (kx - y - k = 0) → (x = 1 → y = 0) :=
by sorry

theorem circle_center_coordinates :
  ∃ (x : ℝ) (y : ℝ),
  (x^2 + y^2 - 4 * x - 2 * y + 1 = 0) ↔ (x = 2 ∧ y = 1) :=
by sorry

theorem no_real_k_for_tangent :
  ¬∃ k : ℝ, ∀ x : ℝ, ∀ y : ℝ,
  (kx - y - k = 0) ∧ (x^2 + y^2 - 4 * x - 2 * y + 1 = 0) →
  (| kx + ky + k | / sqrt(k^2 + 1) = 2) :=
by sorry

theorem chord_length_not_two (k : ℝ) :
  ∀ x : ℝ, ∀ y : ℝ, (kx - y - k = 0) ∧ (x^2 + y^2 - 4 * x - 2 * y + 1 = 0) →  (k = 1) → (2 ≠ 4) :=
by sorry

end line_through_fixed_point_circle_center_coordinates_no_real_k_for_tangent_chord_length_not_two_l256_256084


namespace combine_figures_to_symmetric_l256_256793

-- Defining the basic properties and types
structure GridFigure where
  fig : Set (ℕ × ℕ)
  no_axi_of_sym : ¬(∃ l : ℕ, ∀ (x, y) ∈ fig, (l - x, y) ∈ fig)

-- A function to check vertical symmetry
def has_vertical_symmetry (s : Set (ℕ × ℕ)) : Prop :=
  ∃ l : ℕ, ∀ (x, y) ∈ s, (l - x, y) ∈ s

-- Defining the problem as a theorem
theorem combine_figures_to_symmetric :
  ∀ (fig : GridFigure), has_vertical_symmetry (combine_three_figures fig fig fig) :=
by {
  sorry
}

end combine_figures_to_symmetric_l256_256793


namespace parallel_chords_arcs_equal_l256_256133

open EuclideanGeometry

-- Definitions for the conditions in the problem
def chord (O : Point) (A B : Point) (is_circle : Circle O) : Prop :=
  is_circle A ∧ is_circle B ∧ A ≠ B

def parallel_chords (O : Point) (A B C D : Point) (is_circle : Circle O) : Prop :=
  chord O A B is_circle ∧ chord O C D is_circle ∧ (A - B) ∥ (C - D)

-- The statement representing the mathematical proof problem
theorem parallel_chords_arcs_equal 
  (O : Point) (A B C D : Point) (is_circle : Circle O)
  (h_parallel : parallel_chords O A B C D is_circle) : 
  arc_measure (A, C) is_circle = arc_measure (B, D) is_circle := 
sorry

end parallel_chords_arcs_equal_l256_256133


namespace max_A_value_l256_256831

noncomputable def max_A (x y z : ℝ) : ℝ :=
  (x^3 - 6) * Real.cbrt (x + 6) + (y^3 - 6) * Real.cbrt (y + 6) + (z^3 - 6) * Real.cbrt (z + 6) / (x^2 + y^2 + z^2)

theorem max_A_value (x y z : ℝ) (hx : 0 < x ∧ x ≤ 2) (hy : 0 < y ∧ y ≤ 2) (hz : 0 < z ∧ z ≤ 2) : 
  max_A x y z ≤ 1 := sorry

end max_A_value_l256_256831


namespace permutations_of_red_l256_256514

theorem permutations_of_red : 
  let n := 3 
  in let r := 3 
  in let factorial := Nat.factorial 
  in (factorial n) / (factorial (n - r)) = 6 :=
by
  sorry

end permutations_of_red_l256_256514


namespace not_divide_g_count_30_l256_256192

-- Define the proper positive divisors function
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range (n - 1)).filter (λ d, d > 0 ∧ n % d = 0)

-- Define the product of proper divisors function
def g (n : ℕ) : ℕ :=
  proper_divisors n |>.prod

-- Define the main theorem
theorem not_divide_g_count_30 : 
  (Finset.range 99).filter (λ n, 2 ≤ n + 1 ∧ n + 1 ≤ 100 ∧ ¬(n + 1) ∣ g (n + 1)).card = 30 := 
  by
  sorry

end not_divide_g_count_30_l256_256192


namespace sum_sqrt_series_eq_l256_256198

theorem sum_sqrt_series_eq (T : ℝ) (p q r : ℕ) (hp : p = 100) (hq : q = 50) (hr : r = 2)
    (hT : T = ∑ n in Finset.range 10000 + 1, (1 : ℝ) / Real.sqrt (n + Real.sqrt (n^2 + 1))) :
  T = p + q * Real.sqrt r ∧ p + q + r = 152 := by
  sorry

end sum_sqrt_series_eq_l256_256198


namespace function_range_l256_256861

def f (x : ℤ) : ℤ := abs (x - 1) - 1

theorem function_range : set.range f = {-1, 0, 1} := by
  sorry

end function_range_l256_256861


namespace numberOfWaysToChooseLeadershipStructure_correct_l256_256373

noncomputable def numberOfWaysToChooseLeadershipStructure : ℕ :=
  12 * 11 * 10 * Nat.choose 9 3 * Nat.choose 6 3

theorem numberOfWaysToChooseLeadershipStructure_correct :
  numberOfWaysToChooseLeadershipStructure = 221760 :=
by
  simp [numberOfWaysToChooseLeadershipStructure]
  -- Add detailed simplification/proof steps here if required
  sorry

end numberOfWaysToChooseLeadershipStructure_correct_l256_256373


namespace find_angle_between_vectors_l256_256848

variables {a b : EuclideanSpace ℝ (Fin 2)}

def norm_a : ℝ := ∥a∥ -- norm of vector a
def norm_b : ℝ := ∥b∥ -- norm of vector b

theorem find_angle_between_vectors
  (ha : norm_a = sqrt 2)
  (hb : norm_b = 2)
  (orthog : (a - b) ⬝ a = 0) : 
  real.angle a b = π / 4 :=
sorry

end find_angle_between_vectors_l256_256848


namespace max_A_value_l256_256832

noncomputable def max_A (x y z : ℝ) : ℝ :=
  (x^3 - 6) * Real.cbrt (x + 6) + (y^3 - 6) * Real.cbrt (y + 6) + (z^3 - 6) * Real.cbrt (z + 6) / (x^2 + y^2 + z^2)

theorem max_A_value (x y z : ℝ) (hx : 0 < x ∧ x ≤ 2) (hy : 0 < y ∧ y ≤ 2) (hz : 0 < z ∧ z ≤ 2) : 
  max_A x y z ≤ 1 := sorry

end max_A_value_l256_256832


namespace base6_divisibility_13_l256_256450

theorem base6_divisibility_13 (d : ℕ) (h : 0 ≤ d ∧ d ≤ 5) : (435 + 42 * d) % 13 = 0 ↔ d = 5 :=
by sorry

end base6_divisibility_13_l256_256450


namespace area_triangle_APQ_l256_256631

def equilateral_triangle (A B C P Q : Type) [metric_space A] :=
  ∃ (a b c : A) (R T : A → A) (x y : ℝ), 
    dist a b = 10 ∧
    dist a c = 10 ∧
    dist b c = 10 ∧
    dist P Q = 4 ∧
    x = dist a P ∧
    y = dist a Q ∧
    (R a = P) ∧
    (T a = Q) ∧
    (x + y = 6) ∧
    (x^2 + y^2 - x * y = 16)

theorem area_triangle_APQ (A B C P Q : Type) [metric_space A] : 
  ∀ (x y : ℝ), 
    equilateral_triangle A B C P Q → 
    x = dist A P → 
    y = dist A Q → 
    (x * y = 20 / 3) → 
    let s := real.sqrt 3 / 2 in 
    (1 / 2 * x * y * s = 5 * real.sqrt 3 / 3) :=
by simp [equilateral_triangle]

end area_triangle_APQ_l256_256631


namespace split_enthusiasts_into_100_sections_l256_256385

theorem split_enthusiasts_into_100_sections :
  ∃ (sections : Fin 100 → Set ℕ),
    (∀ i, sections i ≠ ∅) ∧
    (∀ i j, i ≠ j → sections i ∩ sections j = ∅) ∧
    (⋃ i, sections i) = {n : ℕ | n < 5000} :=
sorry

end split_enthusiasts_into_100_sections_l256_256385


namespace two_digit_numbers_div_quotient_remainder_l256_256412

theorem two_digit_numbers_div_quotient_remainder (x y : ℕ) (N : ℕ) (h1 : N = 10 * x + y) (h2 : N = 7 * (x + y) + 6) (hx_range : 1 ≤ x ∧ x ≤ 9) (hy_range : 0 ≤ y ∧ y ≤ 9) :
  N = 62 ∨ N = 83 := sorry

end two_digit_numbers_div_quotient_remainder_l256_256412


namespace perpendicular_lines_a_l256_256870

theorem perpendicular_lines_a : 
  set (a : ℝ) where 
    (2 * a + a * (2 * a - 1) = 0) :=
begin
  simp,
  intro h,
  split,
  {
    intro ha,
    rw [← ha],
    exact ha
  },
  {
    intro hb,
    rw [← hb],
    exact hb
  }
end

end perpendicular_lines_a_l256_256870


namespace average_matches_played_rounding_l256_256532

theorem average_matches_played_rounding :
  let total_matches := 4 * 1 + 3 * 2 + 2 * 4 + 2 * 6 + 4 * 8 in
  let total_players := 4 + 3 + 2 + 2 + 4 in
  let average_matches := total_matches / total_players in
  Int.round average_matches = 4 :=
by
  let total_matches := 4 * 1 + 3 * 2 + 2 * 4 + 2 * 6 + 4 * 8
  let total_players := 4 + 3 + 2 + 2 + 4
  let average_matches := total_matches / total_players
  have : average_matches = 62 / 15 := sorry
  have : Int.round average_matches = 4 := sorry
  exact this

end average_matches_played_rounding_l256_256532


namespace constant_value_l256_256838

def f (x : ℝ) : ℝ := 3 * x - 5

theorem constant_value : ∀ x, x = 3 → (2 * f x - f (x - 2) = 10) := by
  intro x h
  rw h
  dsimp [f]
  norm_num
  sorry

end constant_value_l256_256838


namespace toys_total_is_240_l256_256616

def number_of_toys_elder : Nat := 60
def number_of_toys_younger (toys_elder : Nat) : Nat := 3 * toys_elder
def total_number_of_toys (toys_elder toys_younger : Nat) : Nat := toys_elder + toys_younger

theorem toys_total_is_240 : total_number_of_toys number_of_toys_elder (number_of_toys_younger number_of_toys_elder) = 240 :=
by
  sorry

end toys_total_is_240_l256_256616


namespace snow_volume_l256_256578

-- Define the dimensions of the sidewalk and the snow depth
def length : ℝ := 20
def width : ℝ := 2
def depth : ℝ := 0.5

-- Define the volume calculation
def volume (l w d : ℝ) : ℝ := l * w * d

-- The theorem to prove
theorem snow_volume : volume length width depth = 20 := 
by
  sorry

end snow_volume_l256_256578


namespace problem_solution_l256_256397

theorem problem_solution : 
  (1 / (2 ^ 1980) * (∑ n in Finset.range 991, (-3:ℝ) ^ n * Nat.choose 1980 (2 * n))) = -1 / 2 :=
by
  sorry

end problem_solution_l256_256397


namespace toys_total_is_240_l256_256617

def number_of_toys_elder : Nat := 60
def number_of_toys_younger (toys_elder : Nat) : Nat := 3 * toys_elder
def total_number_of_toys (toys_elder toys_younger : Nat) : Nat := toys_elder + toys_younger

theorem toys_total_is_240 : total_number_of_toys number_of_toys_elder (number_of_toys_younger number_of_toys_elder) = 240 :=
by
  sorry

end toys_total_is_240_l256_256617


namespace geometric_sequence_nth_term_l256_256908

variables {m n : ℕ+} {b_m q : ℝ} -- ℕ+ to ensure m and n are positive naturals

theorem geometric_sequence_nth_term :
  b_n = b_m * q^(n - m) :=
by
  sorry  -- Skip proof as specified

end geometric_sequence_nth_term_l256_256908


namespace required_distilled_water_l256_256903

variable (distilled_water nutrient_concentrate total_medium final_medium : ℝ)
variable (fraction_of_distilled_water : distilled_water / total_medium = 3 / 8)
variable (final_distilled_water : final_medium * (3 / 8) = 0.24)

theorem required_distilled_water :
  distilled_water = 0.03 ∧ nutrient_concentrate = 0.05 ∧ total_medium = 0.08 ∧ final_medium = 0.64 →
  ∃ distilled_water_needed : ℝ, distilled_water_needed = 0.24 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  use 0.24
  exact h6.symm

end required_distilled_water_l256_256903


namespace largest_r_for_sequence_property_l256_256236

theorem largest_r_for_sequence_property :
  ∀ (a : ℕ → ℕ), (∃ r > 0, ∀ n, a n ≤ a (n + 2) ∧ a (n + 2) ≤ real.sqrt (a n ^ 2 + r * a (n + 1))) →
  (∃ M, ∀ n ≥ M, a (n + 2) = a n) → 
  r = 2 :=
sorry

end largest_r_for_sequence_property_l256_256236


namespace symmetric_graph_l256_256958

variable (f : ℝ → ℝ)
variable (c : ℝ)
variable (h_nonzero : c ≠ 0)
variable (h_fx_plus_y : ∀ (x y : ℝ), f (x + y) + f (x - y) = 2 * f x * f y)
variable (h_f_half_c : f (c / 2) = 0)
variable (h_f_zero : f 0 ≠ 0)

theorem symmetric_graph (k : ℤ) : 
  ∀ (x : ℝ), f (x) = f (2*k*c - x) :=
sorry

end symmetric_graph_l256_256958


namespace part1_part2_part3_l256_256556

variables {A B C A1 B1 C1 E F G : Type}
variables (AA1_perpendicular_to_ABC : AA_1 ⟂ ABC)
variables (AB_perpendicular_to_AC : AB ⟂ AC)
variables (AC_eq_AB_eq_AA1 : AC = AB ∧ AB = AA_1 ∧ AA_1 = AC)
variables (E_midpoint_BC : E = midpoint(B, C))
variables (F_midpoint_A1A : F = midpoint(A1, A))
variables (G_on_CC1_parallel_C1F_AEG : G ∈ CC1 ∧ C1F ∥ plane(A, E, G))

theorem part1 : (CG / CC1) = 1 / 2 :=
sorry

theorem part2 : EG ⟂ A1C :=
sorry

theorem part3 : (cosine_dihedral_angle(A1, AG, E)) = - (sqrt(6) / 6) :=
sorry

end part1_part2_part3_l256_256556


namespace future_ratio_l256_256287

variable (j e : ℕ)

-- Conditions
axiom condition1 : j - 3 = 4 * (e - 3)
axiom condition2 : j - 5 = 5 * (e - 5)

-- Theorem to be proved
theorem future_ratio : ∃ x : ℕ, x = 1 ∧ ((j + x) / (e + x) = 3) := by
  sorry

end future_ratio_l256_256287


namespace number_of_observations_l256_256684

variable {n : ℕ}
variable (mean_initial mean_corrected : ℚ) (error magnitude : ℚ)

-- Conditions
def initial_mean_conditions : Prop :=
  mean_initial = 36

def corrected_error_condition : Prop :=
  magnitude = 46 - 23

def corrected_mean_condition : Prop :=
  mean_corrected = 36.5

-- Statement to prove
theorem number_of_observations :
  initial_mean_conditions ∧ corrected_error_condition ∧ corrected_mean_condition → n = 46 :=
by sorry

end number_of_observations_l256_256684


namespace sues_answer_l256_256368

theorem sues_answer (n : ℕ) (h : n = 8) : 
  let ben_result := (n + 2) * 3 in
  let sue_result := (ben_result - 2) * 3 in
  sue_result = 84 := 
by {
  have ben_computation : ben_result = (n + 2) * 3 := rfl,
  have sue_computation : sue_result = (ben_result - 2) * 3 := rfl,
  simp [h, ben_computation, sue_computation],
  norm_num,
  sorry
}

end sues_answer_l256_256368


namespace find_z_l256_256090

variables {i z : ℂ} (A B : Set ℂ)
def imaginary_unit := (i * i = -1)
def A_definition := A = {1, 3, z * i}
def B_definition := B = {4}
def union_condition := A ∪ B = A

theorem find_z (h_imag : imaginary_unit) (hA : A_definition) (hB : B_definition) (h_union : union_condition) : z = -4 * i :=
sorry

end find_z_l256_256090


namespace range_of_a_l256_256947

open Set

theorem range_of_a (a : ℝ) (h1 : (∃ x, a^x > 1 ∧ x < 0) ∨ (∀ x, ax^2 - x + a ≥ 0))
  (h2 : ¬((∃ x, a^x > 1 ∧ x < 0) ∧ (∀ x, ax^2 - x + a ≥ 0))) :
  a ∈ (Ioo 0 (1/2)) ∪ (Ici 1) :=
by {
  sorry
}

end range_of_a_l256_256947


namespace angle_BFP_half_B_l256_256922

variable {A B C I F P : Type*}
variable [triangle : Triangle A B C] 
variable [incenter I : Incenter A B C]
variable (line_parallel_AC : ∃ (l : Line), parallel l (Line.mk A C) ∧ point_on_line I l)
variable (F_on_AB : point_on_line F (Line.mk A B))
variable (P_on_BC : point_on_line P (Line.mk B C)) 
variable (three_BP_eq_BC : 3 * (B.to P) = B.to C)
variable (A_eq_60 : ∠A = 60)

-- The proof problem statement:
theorem angle_BFP_half_B (triangle_ABC : Triangle A B C) 
  (incenter_I : Incenter A B C)
  (parallel_line : parallel (Line.mk I A) (Line.mk A C))
  (F_intersects_AB : intersect_point F (Line.mk A B))
  (P_on_BC : 3 * (distance B P) = distance B C)
  (angle_A_60 : ∠A = 60) :
  ∠BFP = ∠B / 2 :=
sorry  -- Proof goes here

end angle_BFP_half_B_l256_256922


namespace find_x_to_be_2_l256_256088

variable (x : ℝ)

def a := (2, x)
def b := (3, x + 1)

theorem find_x_to_be_2 (h : a x = b x) : x = 2 := by
  sorry

end find_x_to_be_2_l256_256088


namespace probability_team_A_3_points_probability_team_A_1_point_probability_combined_l256_256710

namespace TeamProbabilities

noncomputable def P_team_A_3_points : ℚ :=
  (1 / 3) * (1 / 3) * (1 / 3)

noncomputable def P_team_A_1_point : ℚ :=
  (1 / 3) * (2 / 3) * (2 / 3) + (2 / 3) * (1 / 3) * (2 / 3) + (2 / 3) * (2 / 3) * (1 / 3)

noncomputable def P_team_A_2_points : ℚ :=
  (1 / 3) * (1 / 3) * (2 / 3) + (1 / 3) * (2 / 3) * (1 / 3) + (2 / 3) * (1 / 3) * (1 / 3)

noncomputable def P_team_B_1_point : ℚ :=
  (1 / 2) * (2 / 3) * (3 / 4) + (1 / 2) * (1 / 3) * (3 / 4) + (1 / 2) * (2 / 3) * (1 / 4) + (1 / 2) * (2 / 3) * (1 / 4) +
  (1 / 2) * (1 / 3) * (1 / 4) + (1 / 2) * (1 / 3) * (3 / 4) + (2 / 3) * (2 / 3) * (1 / 4) + (2 / 3) * (1 / 3) * (1 / 4)

noncomputable def combined_probability : ℚ :=
  P_team_A_2_points * P_team_B_1_point

theorem probability_team_A_3_points :
  P_team_A_3_points = 1 / 27 := by
  sorry

theorem probability_team_A_1_point :
  P_team_A_1_point = 4 / 9 := by
  sorry

theorem probability_combined :
  combined_probability = 11 / 108 := by
  sorry

end TeamProbabilities

end probability_team_A_3_points_probability_team_A_1_point_probability_combined_l256_256710


namespace jack_initial_weight_proof_l256_256570

def initial_weight (current_weight: ℝ) (future_weight: ℝ) (months_future: ℝ) (months_past: ℝ) : ℝ :=
  let weight_loss_future := current_weight - future_weight
  let monthly_loss_rate := weight_loss_future / months_future
  let weight_loss_past := monthly_loss_rate * months_past
  current_weight + weight_loss_past

theorem jack_initial_weight_proof :
  let current_weight := 198 : ℝ
  let future_weight := 180 : ℝ
  let months_future := 45 : ℝ
  let months_past := 6 : ℝ
  initial_weight current_weight future_weight months_future months_past = 200.4 :=
by
  sorry

end jack_initial_weight_proof_l256_256570


namespace width_of_carton_is_25_l256_256753

-- Definitions for the given problem
def carton_width := 25
def carton_length := 60
def width_or_height := min carton_width carton_length

theorem width_of_carton_is_25 : width_or_height = 25 := by
  sorry

end width_of_carton_is_25_l256_256753


namespace coeff_m5n5_in_m_plus_n_pow_10_l256_256301

theorem coeff_m5n5_in_m_plus_n_pow_10 :
  binomial (10, 5) = 252 := by
sorry

end coeff_m5n5_in_m_plus_n_pow_10_l256_256301


namespace starting_positions_P0_P1024_l256_256172

noncomputable def sequence_fn (x : ℝ) : ℝ := 4 * x / (x^2 + 1)

def find_starting_positions (n : ℕ) : ℕ := 2^n - 2

theorem starting_positions_P0_P1024 :
  ∃ P0 : ℝ, ∀ n : ℕ, P0 = sequence_fn^[n] P0 → P0 = sequence_fn^[1024] P0 ↔ find_starting_positions 1024 = 2^1024 - 2 :=
sorry

end starting_positions_P0_P1024_l256_256172


namespace amanda_tickets_l256_256769

-- Mathematically equivalent proof problem based on the conditions and question.

theorem amanda_tickets:
  let goal := 150 in
  let day1 := 8 * 4 in
  let day2 := 45 in
  let day3 := 25 in
  let total_sold := day1 + day2 + day3 in
  goal - total_sold = 48 :=
by
  sorry

end amanda_tickets_l256_256769


namespace combined_average_age_l256_256671

theorem combined_average_age (avg_age_5th_graders : ℝ) (num_5th_graders : ℝ) 
                             (avg_age_parents : ℝ) (num_parents : ℝ) :
  avg_age_5th_graders = 10 → 
  num_5th_graders = 40 → 
  avg_age_parents = 35 → 
  num_parents = 50 → 
  ( (avg_age_5th_graders * num_5th_graders) + (avg_age_parents * num_parents) ) 
  / (num_5th_graders + num_parents) = 215 / 9 :=
by {
  intros h1 h2 h3 h4,
  sorry -- Proof goes here
}

end combined_average_age_l256_256671


namespace cube_edge_in_pyramid_l256_256442

theorem cube_edge_in_pyramid (a h : ℝ) (x : ℝ) 
  (h_pos : 0 < h) (a_pos : 0 < a) :
  (x * ((sqrt 3 + 2) * h + 3 * a) = 3 * a) → 
  x = 3 * a / ((sqrt 3 + 2) * h + 3 * a) :=
by
  sorry

end cube_edge_in_pyramid_l256_256442


namespace max_area_equilateral_triangle_l256_256377

/-- For any triangle with a fixed perimeter, the triangle that maximizes the area is an equilateral triangle. -/
theorem max_area_equilateral_triangle (a b c : ℝ) (h : a + b + c = 2 * p) : 
  ∃ (a b c : ℝ), (a = b ∧ b = c) :=
begin
  sorry
end

end max_area_equilateral_triangle_l256_256377


namespace product_of_values_l256_256886

theorem product_of_values (x : ℝ) (h : abs (20 / x + 1) = 4) : 
  ∏ x in {(20 / 3), -4}, x = -80 / 3 :=
by
  sorry

end product_of_values_l256_256886


namespace four_digits_sum_l256_256608

theorem four_digits_sum (A B C D : ℕ) 
  (A_neq_B : A ≠ B) (A_neq_C : A ≠ C) (A_neq_D : A ≠ D) 
  (B_neq_C : B ≠ C) (B_neq_D : B ≠ D) 
  (C_neq_D : C ≠ D)
  (digits_A : A ≤ 9) (digits_B : B ≤ 9) (digits_C : C ≤ 9) (digits_D : D ≤ 9)
  (A_lt_B : A < B) 
  (minimize_fraction : ∃ k : ℕ, (A + B) = k ∧ k ≤ (A + B) ∧ (C + D) ≥ (C + D)) :
  C + D = 17 := 
by
  sorry

end four_digits_sum_l256_256608


namespace marked_percentage_above_cost_l256_256210

theorem marked_percentage_above_cost (CP SP : ℝ) (discount_percentage MP : ℝ) 
  (h1 : CP = 540) 
  (h2 : SP = 457) 
  (h3 : discount_percentage = 26.40901771336554) 
  (h4 : SP = MP * (1 - discount_percentage / 100)) : 
  ((MP - CP) / CP) * 100 = 15 :=
by
  sorry

end marked_percentage_above_cost_l256_256210


namespace problem_statement_l256_256447

noncomputable def isRelativelyPrime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def sumHarmonic (n : ℕ) : ℚ :=
  (∑ k in Finset.range (n + 1) \ {0}, (1 : ℚ) / k)

def quotientDenominator (n : ℕ) :=
  (sumHarmonic n).den

def notDivisibleBy5 (q : ℕ) : Prop := ¬ (5 ∣ q)

theorem problem_statement :
  ∀ n : ℕ, n > 0 →
    (notDivisibleBy5 (quotientDenominator n) ↔ 
      n ∈ {1, 2, 3, 4, 20, 21, 22, 23, 24, 100, 101, 102, 103, 104, 120, 121, 122, 123, 124}) :=
sorry

end problem_statement_l256_256447


namespace values_of_k_for_exactly_one_real_solution_l256_256452

variable {k : ℝ}

def quadratic_eq (k : ℝ) : Prop := 3 * k^2 + 42 * k - 573 = 0

theorem values_of_k_for_exactly_one_real_solution :
  quadratic_eq k ↔ k = 8 ∨ k = -22 := by
  sorry

end values_of_k_for_exactly_one_real_solution_l256_256452


namespace desargues_theorem_l256_256778

-- Define the points and collinearity condition
noncomputable def points_collinear (p1 p2 p3: Point) : Prop :=
  ∃ (a b c: ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧ a * p1.1 + b * p1.2 + c = 0 ∧
                                a * p2.1 + b * p2.2 + c = 0 ∧
                                a * p3.1 + b * p3.2 + c = 0

-- Define the problem
theorem desargues_theorem 
  (O A A' B B' C C' X Y Z : Point)
  (h_O : A * A' + B * B' + C * C' = O)
  (h_X : BC ∩ B'C' = X)
  (h_Y : CA ∩ C'A' = Y)
  (h_Z : AB ∩ A'B' = Z)
  : points_collinear X Y Z := 
sorry

end desargues_theorem_l256_256778


namespace find_positive_integer_l256_256726

theorem find_positive_integer (n : ℕ) (h1 : 100 % n = 3) (h2 : 197 % n = 3) : n = 97 := 
sorry

end find_positive_integer_l256_256726


namespace polyhedron_volume_correct_l256_256535

-- Definitions of geometric shapes and their properties
def is_isosceles_right_triangle (A : Type) (a b c : ℝ) := 
  a = b ∧ c = a * Real.sqrt 2

def is_square (B : Type) (side : ℝ) := 
  side = 2

def is_equilateral_triangle (G : Type) (side : ℝ) := 
  side = Real.sqrt 8

noncomputable def polyhedron_volume (A E F B C D G : Type) (a b c d e f g : ℝ) := 
  let cube_volume := 8
  let tetrahedron_volume := 2 * Real.sqrt 2 / 3
  cube_volume - tetrahedron_volume

theorem polyhedron_volume_correct (A E F B C D G : Type) (a b c d e f g : ℝ) :
  (is_isosceles_right_triangle A a b c) →
  (is_isosceles_right_triangle E a b c) →
  (is_isosceles_right_triangle F a b c) →
  (is_square B d) →
  (is_square C e) →
  (is_square D f) →
  (is_equilateral_triangle G g) →
  a = 2 → d = 2 → e = 2 → f = 2 → g = Real.sqrt 8 →
  polyhedron_volume A E F B C D G a b c d e f g =
    8 - (2 * Real.sqrt 2 / 3) :=
by
  intros hA hE hF hB hC hD hG ha hd he hf hg
  sorry

end polyhedron_volume_correct_l256_256535


namespace smallest_abs_difference_l256_256272

theorem smallest_abs_difference (a b m n : ℕ) 
(h₁ : 2021 = (list.prod (list.map (Nat.factorial) (list.finRange m)))
             / (list.prod (list.map (Nat.factorial) (list.finRange n))))
(h₂ : list.sorted (λ x y => x ≥ y) (list.map Nat.factorial (list.finRange m)))
(h₃ : list.sorted (λ x y => x ≥ y) (list.map Nat.factorial (list.finRange n)))
(h₄ : a + b = 90 ∧ a = 47 ∧ b = 43) :
  |a - b| = 4 :=
  sorry

end smallest_abs_difference_l256_256272


namespace book_arrangement_l256_256110

/-- Given 6 books, 3 of which are identical, the number of ways to arrange them is 120. -/
theorem book_arrangement : finset.prod (finset.range 7) (λ n, n!) / finset.prod (finset.range 4) (λ n, n!) = 120 := by
    sorry

end book_arrangement_l256_256110


namespace area_of_triangle_APQ_proof_l256_256638

noncomputable def area_of_triangle_APQ : ℝ :=
  let ABC := (A B C : Point) := {a : ℝ | a ≤ 10}
  let P := (P : Point) := {p : ℝ | 0 ≤ p ∧ p ≤ 10}
  let Q := (Q : Point) := {q : ℝ | 0 ≤ q ∧ q ≤ 10}
  let pq := (pq : ℝ) := 4
  let x := (x : ℝ) := AP
  let y := (y : ℝ) := AQ
  let XY := (S : ℝ) := xy
  XY = {S : ℝ | (x, y) ∈ ABC × P × Q, pq = 4, S = PQAP.area}
  let area_APQ := (S : ℝ) := 5 / sqrt 3

theorem area_of_triangle_APQ_proof :
  area_of_triangle_APQ = 5 / sqrt 3 := by sorry

end area_of_triangle_APQ_proof_l256_256638


namespace distance_inequality_l256_256148

variables {α : Type*} [linear_ordered_field α]
variables (A B C D M : Type*) -- Points
variables (a b c : α) -- Lengths
variables {S : α} -- Sum of distances

-- Mutual perpendicular edges of tetrahedron ABCD
def mutually_perpendicular (AD BD CD : Type*) : Prop :=
  -- Definition ensuring AD, BD, and CD are mutually perpendicular
  sorry

-- Length definitions
def length (X Y : Type*) : α := 
  -- Length between points X and Y
  sorry

-- Condition: M lies on one side of triangle ABC
def M_on_triangle_ABC (M A B C : Type*) : Prop :=
  -- Definition ensuring M lies on one side of triangle ABC
  sorry

-- Definition of distances from vertices to line DM
def dist_to_DM (PT Line : Type*) : α :=
  -- Distance from a point to a line
  sorry

noncomputable def sum_distances (A B C : Type*) (DM : Type*) : α :=
  dist_to_DM A DM + dist_to_DM B DM + dist_to_DM C DM

-- Theorem statement
theorem distance_inequality {A B C D M : Type*} {a b c : α} (h_perpendicular: mutually_perpendicular A B C D) 
  (h_lengths : length A D = a ∧ length B D = b ∧ length C D = c)
  (h_M_on_triangle_ABC : M_on_triangle_ABC M A B C) :
  sum_distances A B C D M ≤ sqrt (2 * (a^2 + b^2 + c^2)) ∧ 
  (sum_distances A B C D M = sqrt (2 * (a^2 + b^2 + c^2)) ↔ -- condition for equality
   ⟪ condition for equality from the problem ⟫) :=
sorry

end distance_inequality_l256_256148


namespace geometric_sequence_problem_l256_256040

noncomputable def geometric_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_problem
  (a1 q : ℝ) (a2 : ℝ := a1 * q) (a5 : ℝ := a1 * q^4)
  (S2 : ℝ := geometric_sum a1 q 2) (S4 : ℝ := geometric_sum a1 q 4)
  (h1 : 8 * a2 + a5 = 0) :
  S4 / S2 = 5 :=
by
  sorry

end geometric_sequence_problem_l256_256040


namespace six_digit_product_of_consecutive_even_integers_l256_256431

theorem six_digit_product_of_consecutive_even_integers :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ a * b * c = 287232 :=
sorry

end six_digit_product_of_consecutive_even_integers_l256_256431


namespace pq_even_impossible_l256_256120

theorem pq_even_impossible {p q : ℤ} (h : (p^2 + q^2 + p*q) % 2 = 1) : ¬(p % 2 = 0 ∧ q % 2 = 0) :=
by
  sorry

end pq_even_impossible_l256_256120


namespace max_distance_focus_l256_256487

theorem max_distance_focus :
  let C := {P : ℝ × ℝ | ∃ x y : ℝ, P = (x, y) ∧ (x^2 / 16 + y^2 / 7 = 1)}
  let F₁ := (-3, 0)
  ∀ P ∈ C, |(fst P + 3)^2 + (snd P)^2| ≤ 7 :=
by
  -- definition of the ellipse C
  let C := {P : ℝ × ℝ | ∃ x y : ℝ, P = (x, y) ∧ (x^2 / 16 + y^2 / 7 = 1)}
  -- coordinates of the left focus F₁
  let F₁ := (-3, 0)
  -- prove that the maximum distance from any point P on the ellipse to F₁ is 7
  intros P hP
  sorry

end max_distance_focus_l256_256487


namespace vertex_h_is_3_l256_256277

open Real

theorem vertex_h_is_3 (a b c : ℝ) (h : ℝ)
    (h_cond : 3 * (a * 3^2 + b * 3 + c) + 6 = 3) : 
    4 * (a * x^2 + b * x + c) = 12 * (x - 3)^2 + 24 → 
    h = 3 := 
by 
sorry

end vertex_h_is_3_l256_256277


namespace students_passed_this_year_l256_256897

theorem students_passed_this_year
  (initial_students : ℕ)
  (annual_increase_rate : ℝ)
  (years_lapsed : ℕ)
  (current_students : ℕ)
  (h_initial : initial_students = 200)
  (h_rate : annual_increase_rate = 1.5)
  (h_years : years_lapsed = 3)
  (h_calc : current_students = (λ n, initial_students * (annual_increase_rate ^ n)) years_lapsed) :
  current_students = 675 :=
begin
  sorry
end

end students_passed_this_year_l256_256897


namespace average_speed_increased_pace_l256_256347

theorem average_speed_increased_pace 
  (speed_constant : ℝ) (time_constant : ℝ) (distance_increased : ℝ) (total_time : ℝ) 
  (h1 : speed_constant = 15) 
  (h2 : time_constant = 3) 
  (h3 : distance_increased = 190) 
  (h4 : total_time = 13) :
  (distance_increased / (total_time - time_constant)) = 19 :=
by
  sorry

end average_speed_increased_pace_l256_256347


namespace ratio_OBE_BAC_l256_256147

noncomputable def circle_O := sorry -- Define the circle O
noncomputable def triangle_ABC := sorry -- Define the inscribed triangle ABC 
def arc_AB := 140 -- Measure of arc AB is 140 degrees
def arc_BC := 72 -- Measure of arc BC is 72 degrees
noncomputable def point_E_on_minor_arc_AC := sorry -- Define point E on the minor arc AC with OE perpendicular to AC
noncomputable def angle_OBE := 71 / 2 -- \(\angle OBE = 35.5\)
noncomputable def angle_BAC := 72 / 2 -- \(\angle BAC = 36\)

theorem ratio_OBE_BAC : 
  ∀ (circle_O : Type) (triangle_ABC : Type) (E : Type) (OE_per_AC : Prop), 
    arc_AB = 140 ∧ arc_BC = 72 ∧ point_E_on_minor_arc_AC ∧ OE_per_AC → 
    angle_OBE / angle_BAC = 71 / 72 :=
by
  intro circle_O triangle_ABC point_E_on_minor_arc_AC OE_per_AC
  assume h1 : arc_AB = 140
  assume h2 : arc_BC = 72
  assume h3 : point_E_on_minor_arc_AC
  assume h4 : OE_per_AC
  exact sorry

end ratio_OBE_BAC_l256_256147


namespace llama_roaming_area_l256_256791

/-- 
Given:
- a right-angled triangle with sides of lengths 2m and 3m
- a llama tied to the right-angle vertex with a 4m leash
Prove that the total area that Chuck can roam around the outside of the shed is 6.5π m².
--/
theorem llama_roaming_area
  (A B C : Point)
  (side1 : A.dist B = 2)
  (side2 : A.dist C = 3)
  (right_angle : ∠BAC = 90°)
  (leash_length : 4) :
  total_area Chuck_can_roam = 6.5 * π :=
sorry

end llama_roaming_area_l256_256791


namespace integral_point_of_f_l256_256465

def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)
def a : ℝ := 0
def b : ℝ := Real.pi / 2
noncomputable def integral_f : ℝ := ∫ x : ℝ in a..b, f x
def x0 : ℝ := Real.pi / 4

theorem integral_point_of_f :
  f x0 = integral_f := sorry

end integral_point_of_f_l256_256465


namespace elevator_time_correct_l256_256611

def elevator_floors : ℕ := 50
def time_first_segment : ℚ := 35 / 60 -- in hours
def time_second_segment : ℚ := 8 * 7 / 60 -- in hours
def time_third_segment : ℚ := 4.5 * 13 / 60 -- in hours
def time_fourth_segment : ℚ := 75 / 60 -- in hours
def time_fifth_segment_six_floors : ℚ := 18 * 3 / 60 -- in hours
def time_fifth_segment_remaining_floors : ℚ := 18 * 1.5 / (2 * 60) -- in hours

def total_time : ℚ :=
  time_first_segment +
  time_second_segment +
  time_third_segment +
  time_fourth_segment +
  time_fifth_segment_six_floors +
  time_fifth_segment_remaining_floors

theorem elevator_time_correct :
  total_time ≈ 4.87 := by
  sorry

end elevator_time_correct_l256_256611


namespace square_side_length_l256_256952

theorem square_side_length (A B C : ℝ) (h1 : AB = 13) (h2 : BC = 14) (h3 : CA = 15)
  (P QR S : ℝ → ℝ)
  (h4 : P QR ∥ BC)
  (h5 : Q ∈ CA) (h6 : S ∈ AB) : 
  side_length PQRS = 42 * (√2) := 
sorry

end square_side_length_l256_256952


namespace sin_neg_1290_l256_256784

theorem sin_neg_1290 : Real.sin (-(1290 : ℝ) * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_neg_1290_l256_256784


namespace proj_vector_l256_256099

open Real

def projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_prod := (a.1 * b.1 + a.2 * b.2 + a.3 * b.3)
  let b_mag_sq := (b.1^2 + b.2^2 + b.3^2)
  let scalar := dot_prod / b_mag_sq
  (scalar * b.1, scalar * b.2, scalar * b.3)

theorem proj_vector (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (1, 3, 0))
  (hb : b = (2, 1, 1))
  (hc : c = (5/3, 5/6, 5/6)) : 
  projection a b = c :=
by
  sorry

end proj_vector_l256_256099


namespace replace_asterisk_l256_256728

theorem replace_asterisk (star : ℝ) : ((36 / 18) * (star / 72) = 1) → star = 36 :=
by
  intro h
  sorry

end replace_asterisk_l256_256728


namespace triangle_trapezoids_distance_l256_256156

theorem triangle_trapezoids_distance
  (XY YZ XZ : ℝ)
  (hXY : XY = 15)
  (hYZ : YZ = 18)
  (hXZ : XZ = 21)
  (QZ : ℝ)
  (hQZ : QZ = 12)
  (FG : ℝ)
  (hXYFZ : XYFZ trapezoid)
  (hXYGZ : XYGZ trapezoid) :
  FG = 6 := 
sorry

end triangle_trapezoids_distance_l256_256156


namespace coronavirus_particle_diameter_scientific_notation_l256_256424

theorem coronavirus_particle_diameter_scientific_notation :
  (0.00000012 : ℝ) = 1.2 * 10^(-7) :=
sorry

end coronavirus_particle_diameter_scientific_notation_l256_256424


namespace projection_vector_of_a_onto_b_l256_256098

open Real

noncomputable def a : ℝ × ℝ × ℝ := (1, 3, 0)
noncomputable def b : ℝ × ℝ × ℝ := (2, 1, 1)

theorem projection_vector_of_a_onto_b :
  let dot_product := (a.1 * b.1) + (a.2 * b.2) + (a.3 * b.3)
  let b_magnitude_sq := (b.1 ^ 2) + (b.2 ^ 2) + (b.3 ^ 2)
  let scalar := dot_product / b_magnitude_sq
  let c := (scalar * b.1, scalar * b.2, scalar * b.3)
  c = (5 / 3, 5 / 6, 5 / 6) :=
by
  sorry

end projection_vector_of_a_onto_b_l256_256098


namespace proj_vector_l256_256101

open Real

def projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_prod := (a.1 * b.1 + a.2 * b.2 + a.3 * b.3)
  let b_mag_sq := (b.1^2 + b.2^2 + b.3^2)
  let scalar := dot_prod / b_mag_sq
  (scalar * b.1, scalar * b.2, scalar * b.3)

theorem proj_vector (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (1, 3, 0))
  (hb : b = (2, 1, 1))
  (hc : c = (5/3, 5/6, 5/6)) : 
  projection a b = c :=
by
  sorry

end proj_vector_l256_256101


namespace max_a_plus_b_l256_256598

theorem max_a_plus_b (a b : ℝ) (h1 : 4 * a + 3 * b ≤ 10) (h2 : 3 * a + 6 * b ≤ 12) : a + b ≤ 14 / 5 := 
sorry

end max_a_plus_b_l256_256598


namespace find_f_expression_l256_256214

noncomputable def f (x : ℝ) := -2 * Real.cos (2 * x)

noncomputable def g (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 6)

theorem find_f_expression :
    (∀ x : ℝ, f (x) = g (x - Real.pi / 3)) → (f = -2 * Real.cos ∘ (λ x, 2 * x)) :=
by
  intro h
  funext x
  specialize h x
  simp only [f, g] at h
  rw [← Real.sin_add_pi_div_two] at h
  exact h
  sorry

end find_f_expression_l256_256214


namespace mode_is_crucial_l256_256906

-- Defining the statistical measures in the pre-promotion survey context
def variance : Type := sorry  -- Placeholder definition for variance
def mode : Type := sorry      -- Placeholder definition for mode
def median : Type := sorry    -- Placeholder definition for median
def mean : Type := sorry      -- Placeholder definition for mean

-- Condition: Inventory decision context for a shoe store
def inventory_decision : variance → mode → median → mean → Prop := sorry

-- Theorem statement
theorem mode_is_crucial (v : variance) (mo : mode) (me : median) (mn : mean) : inventory_decision v mo me mn → mo = "Mode" := 
by
  sorry

end mode_is_crucial_l256_256906


namespace pantry_proof_l256_256649

noncomputable def stew_serving_per_can := 2
noncomputable def baked_beans_serving_per_can := 4
noncomputable def soup_serving_per_can := 3
noncomputable def initial_people := 40
noncomputable def reduction_percentage := 0.30
noncomputable def proportion_stews := 0.60
noncomputable def proportion_baked_beans := 0.25
noncomputable def proportion_soups := 0.15

noncomputable def reduced_people := initial_people * (1 - reduction_percentage)

noncomputable def servings_needed_stews := reduced_people * proportion_stews
noncomputable def servings_needed_baked_beans := reduced_people * proportion_baked_beans
noncomputable def servings_needed_soups := reduced_people * proportion_soups

noncomputable def cans_needed_stews := nat.ceil (servings_needed_stews / stew_serving_per_can)
noncomputable def cans_needed_baked_beans := nat.ceil (servings_needed_baked_beans / baked_beans_serving_per_can)
noncomputable def cans_needed_soups := nat.ceil (servings_needed_soups / soup_serving_per_can)

theorem pantry_proof :
  cans_needed_stews = 9 ∧ 
  cans_needed_baked_beans = 2 ∧ 
  cans_needed_soups = 2 := 
by
  sorry

end pantry_proof_l256_256649


namespace total_arrangements_l256_256417

-- Given conditions
def teachers := 2
def students := 6
def group_size := 4  -- because each group consists of 1 teacher and 3 students

-- Required to prove
theorem total_arrangements : ((Finset.card (Finset.powersetLen 1 (Finset.range teachers))).card * 
                              (Finset.card (Finset.powersetLen 3 (Finset.range students))).card * 
                              1) = 40 := by
  sorry

end total_arrangements_l256_256417


namespace find_older_pennies_l256_256515

-- Define the total number of pennies Iain has
def total_pennies : ℕ := 200

-- Define the number of pennies Iain has after getting rid of the older ones and throwing out 20% of the remainder
def remaining_pennies (P : ℕ) : ℕ := total_pennies - P - (0.20 * (total_pennies - P)).natAbs

-- The condition that Iain is left with 136 pennies afterwards
theorem find_older_pennies : ∃ P : ℕ, remaining_pennies P = 136 :=
by
  -- Existence proof skipped
  sorry

end find_older_pennies_l256_256515


namespace conics_concurrent_lines_l256_256173

/-
Let three conics \mathcal{E}_{1}, \mathcal{E}_{2}, \mathcal{E}_{3} share two common points A and B.
  For each pair of conics \mathcal{E}_i and \mathcal{E}_j, besides A and B,
  they intersect at two other points, and let \ell_{i,j} denote the line
  through these two points. Show that the three lines \ell_{1,2}, \ell_{1,3},
  and \ell_{2,3} obtained this way are concurrent.
-/
theorem conics_concurrent_lines
  (E1 E2 E3 : Conic)
  (A B : Point)
  (h_common : ∀ {i j : ℕ}, i ≠ j → ∀ x ∈ {A, B}, x ∈ (E1 i) ∩ (E1 j))
  (I₁₂ I₁₃ I₂₃ : set Point) -- Intersection sets, besides A and B
  (h_I₁₂ : ∀ x, x ∈ I₁₂ → x ∈ E1 ∩ E2 ∧ x ≠ A ∧ x ≠ B)
  (h_I₁₃ : ∀ x, x ∈ I₁₃ → x ∈ E1 ∩ E3 ∧ x ≠ A ∧ x ≠ B)
  (h_I₂₃ : ∀ x, x ∈ I₂₃ → x ∈ E2 ∩ E3 ∧ x ≠ A ∧ x ≠ B)
  (l₁₂ l₁₃ l₂₃ : Line)
  (h_l₁₂ : l₁₂ = line_through_points E1 E2 I₁₂)
  (h_l₁₃ : l₁₃ = line_through_points E1 E3 I₁₃)
  (h_l₂₃ : l₂₃ = line_through_points E2 E3 I₂₃) :
  ∃ P : Point, P ∈ l₁₂ ∧ P ∈ l₁₃ ∧ P ∈ l₂₃ :=
sorry

end conics_concurrent_lines_l256_256173


namespace laundry_loads_l256_256342

-- Definitions based on conditions
def num_families : ℕ := 3
def people_per_family : ℕ := 4
def num_people : ℕ := num_families * people_per_family

def days : ℕ := 7
def towels_per_person_per_day : ℕ := 1
def total_towels : ℕ := num_people * days * towels_per_person_per_day

def washing_machine_capacity : ℕ := 14

-- Statement to prove
theorem laundry_loads : total_towels / washing_machine_capacity = 6 := 
by
  sorry

end laundry_loads_l256_256342


namespace solution_l256_256587

noncomputable def problem := 
  let AB := 80
  let AC := 150
  let BC := 170
  let s := (AB + AC + BC) / 2
  let A := (AB * AC) / 2
  let r1 := A / s
  let BD x := AC - x
  let CF y := AB - y
  let r2 x := (r1 * BD x) / AB
  let r3 y := (r1 * CF y) / AC
  let O2 x := (AC - x, r2 x)
  let O3 y := (y, r3 y)
  let distance_between_centers x y := 
    (O2 x).dist (O3 y)
  forall (x : ℝ) (y : ℝ), 
    x = 70 → y = 50 → 
    distance_between_centers x y = real.sqrt (10 * 1057.6)
  
theorem solution : problem := 
  sorry

end solution_l256_256587


namespace simplify_and_evaluate_l256_256665

-- Define the expression as a function of a and b
def expr (a b : ℚ) : ℚ := 5 * a * b - 2 * (3 * a * b - (4 * a * b^2 + (1/2) * a * b)) - 5 * a * b^2

-- State the condition and the target result
theorem simplify_and_evaluate : 
  let a : ℚ := -1
  let b : ℚ := 1 / 2
  expr a b = -3 / 4 :=
by
  -- Proof goes here
  sorry

end simplify_and_evaluate_l256_256665


namespace oldest_child_age_l256_256672

def arithmeticProgression (a d : ℕ) (n : ℕ) : ℕ := 
  a + (n - 1) * d

theorem oldest_child_age (a : ℕ) (d : ℕ) (n : ℕ) 
  (average : (arithmeticProgression a d 1 + arithmeticProgression a d 2 + arithmeticProgression a d 3 + arithmeticProgression a d 4 + arithmeticProgression a d 5) / 5 = 10)
  (distinct : ∀ i j, i ≠ j → arithmeticProgression a d i ≠ arithmeticProgression a d j)
  (constant_difference : d = 3) :
  arithmeticProgression a d 5 = 16 :=
by
  sorry

end oldest_child_age_l256_256672


namespace sarah_total_pencils_l256_256990

-- Define the number of pencils Sarah buys on each day
def pencils_monday : ℕ := 35
def pencils_tuesday : ℕ := 42
def pencils_wednesday : ℕ := 3 * pencils_tuesday
def pencils_thursday : ℕ := pencils_wednesday / 2
def pencils_friday : ℕ := 2 * pencils_monday

-- Define the total number of pencils
def total_pencils : ℕ :=
  pencils_monday + pencils_tuesday + pencils_wednesday + pencils_thursday + pencils_friday

-- Theorem statement to prove the total number of pencils equals 336
theorem sarah_total_pencils : total_pencils = 336 :=
by
  -- here goes the proof, but it is not required
  sorry

end sarah_total_pencils_l256_256990


namespace rachel_pizza_eaten_l256_256006

theorem rachel_pizza_eaten (pizza_total : ℕ) (pizza_bella : ℕ) (pizza_rachel : ℕ) :
  pizza_total = pizza_bella + pizza_rachel → pizza_bella = 354 → pizza_total = 952 → pizza_rachel = 598 :=
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  sorry

end rachel_pizza_eaten_l256_256006


namespace perimeter_decreases_to_convex_hull_outer_polygon_perimeter_ge_inner_polygon_l256_256733

open_locale classical

-- Define what it means to be a convex polygon and its perimeter
structure Polygon :=
  (vertices : list (ℝ × ℝ))
  (is_convex : Prop)
  (perimeter : ℝ)

-- Define a non-convex polygon and its convex hull
def convex_hull (p : Polygon) : Polygon := {
  vertices := sorry,  -- to be replaced with actual convex hull computation
  is_convex := true,
  perimeter := sorry  -- compute the perimeter of the convex hull
}

-- Part (a): transitioning from a non-convex polygon to its convex hull, the perimeter decreases
theorem perimeter_decreases_to_convex_hull (p : Polygon) (h : ¬p.is_convex) :
  convex_hull(p).perimeter < p.perimeter := sorry

-- Part (b): perimeter of the outer convex polygon is not less than the perimeter of the inner polygon
theorem outer_polygon_perimeter_ge_inner_polygon (A B : Polygon) (hA : A.is_convex) (hB : B.is_convex) (h_contains : ∀ v ∈ A.vertices, v ∈ B.vertices) :
  B.perimeter ≥ A.perimeter := sorry

end perimeter_decreases_to_convex_hull_outer_polygon_perimeter_ge_inner_polygon_l256_256733


namespace alpha_solutions_l256_256888

theorem alpha_solutions (k : ℤ) (α : ℝ) (h : cos α = 1 / 2) : ∃ k : ℤ, α = 2 * π * k :=
by
  sorry

end alpha_solutions_l256_256888


namespace fifth_pillow_price_l256_256658

theorem fifth_pillow_price:
  ∀ (a b c x: ℕ), 
    a = 5 ∧ b = 4 ∧ c = 6 ∧ 
    x = (5 * c) - (b * a) → 
    x = 10 :=
by
  intros a b c x h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  rw [h1, h3, h5] at h6
  exact h6
  sorry

end fifth_pillow_price_l256_256658


namespace correct_propositions_l256_256057

structure Line := (name : String)
structure Plane := (name : String)

def perpendicular_to (m : Line) (α : Plane) : Prop := ∃ p, p ∈ α ∧ m ⊥ p
def parallel_to (m : Line) (α : Plane) : Prop := ∀ p1 p2, p1 ∈ α ∧ p2 ∈ α → m ∥ p1 ∧ m ∥ p2
def intersects (α β : Plane) (l : Line) : Prop := l ∈ α ∧ l ∈ β

variables (m n : Line) (α β : Plane)

def is_correct (prop : String) : Prop :=
  match prop with
  | "1" := perpendicular_to m α ∧ perpendicular_to n β ∧ m ⊥ n → α ⊥ β
  | "2" := parallel_to m α ∧ intersects α β n → m ∥ n
  | "3" := perpendicular_to m α ∧ parallel_to n β ∧ m ⊥ n → α ⊥ β
  | "4" := perpendicular_to m α ∧ parallel_to n β ∧ m ∥ n → α ⊥ β
  | _   := False

theorem correct_propositions:
  is_correct m n α β "1" ∧ is_correct m n α β "4" :=
sorry

end correct_propositions_l256_256057


namespace determine_a_l256_256605

noncomputable def area_ratio (x y : ℝ) : ℝ :=
  (sqrt 3 / 4 * x^2) / (3 * sqrt 3 / 2 * y^2)

theorem determine_a (x y a : ℝ) (h₁ : 3 * x = 6 * y) (h₂ : area_ratio x y = 2 / a) :
  a = 3 := 
sorry

end determine_a_l256_256605


namespace parts_outside_range_l256_256745

noncomputable def number_of_parts_outside_range (μ σ : ℝ) (n : ℕ) : ℕ :=
  let proportion_outside := 1 - 0.9973
  let expected_parts_outside := (n : ℝ) * proportion_outside
  nat_ceil expected_parts_outside

theorem parts_outside_range (μ σ : ℝ) (n : ℕ) (hn : n = 1000):
  number_of_parts_outside_range μ σ n = 3 :=
by 
  have proportion_outside : ℝ := 1 - 0.9973
  have expected_parts_outside : ℝ := (n : ℝ) * proportion_outside
  have h1 : expected_parts_outside ≈ 2.7
  -- Since expected parts not in range is a fraction, we need to round appropriately.
  sorry

end parts_outside_range_l256_256745


namespace trig_sum_identity_l256_256445

theorem trig_sum_identity :
  Real.sin (20 * Real.pi / 180) + Real.sin (40 * Real.pi / 180) +
  Real.sin (60 * Real.pi / 180) - Real.sin (80 * Real.pi / 180) = Real.sqrt 3 / 2 := 
sorry

end trig_sum_identity_l256_256445


namespace projection_vector_of_a_onto_b_l256_256097

open Real

noncomputable def a : ℝ × ℝ × ℝ := (1, 3, 0)
noncomputable def b : ℝ × ℝ × ℝ := (2, 1, 1)

theorem projection_vector_of_a_onto_b :
  let dot_product := (a.1 * b.1) + (a.2 * b.2) + (a.3 * b.3)
  let b_magnitude_sq := (b.1 ^ 2) + (b.2 ^ 2) + (b.3 ^ 2)
  let scalar := dot_product / b_magnitude_sq
  let c := (scalar * b.1, scalar * b.2, scalar * b.3)
  c = (5 / 3, 5 / 6, 5 / 6) :=
by
  sorry

end projection_vector_of_a_onto_b_l256_256097


namespace integrand_eval_l256_256337

theorem integrand_eval :
  ∫ (x : ℝ) in 0..1, (λ x, e^(-2 * x) / (1 + e^(-x))) = -1 - e^(-x) + log(1 + e^(-x)) + C :=
sorry

end integrand_eval_l256_256337


namespace probability_no_cowboys_picks_own_hat_l256_256708

def derangements (n : Nat) : Nat :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangements (n - 1) + derangements (n - 2))

theorem probability_no_cowboys_picks_own_hat : 
  let total_arrangements := Nat.factorial 3
  let favorable_arrangements := derangements 3
  let probability := (favorable_arrangements : ℚ) / (total_arrangements : ℚ)
  probability = 1 / 3 :=
by
  let total_arrangements := 6
  let favorable_arrangements := 2
  have h : probability = (2 : ℚ) / (6 : ℚ) := by rfl
  rw h
  calc (2 : ℚ) / (6 : ℚ)
          = 1 / 3 : by norm_num
  rw [h]
  sorry

end probability_no_cowboys_picks_own_hat_l256_256708


namespace passing_students_this_year_l256_256894

constant initial_students : ℕ := 200 -- Initial number of students who passed three years ago
constant growth_rate : ℝ := 0.5      -- Growth rate of 50%

-- Function to calculate the number of students passing each year
def students_passing (n : ℕ) : ℕ :=
nat.rec_on n initial_students (λ n' ih, ih + (ih / 2))

-- Proposition stating the number of students passing the course this year
theorem passing_students_this_year : students_passing 3 = 675 := sorry

end passing_students_this_year_l256_256894


namespace compare_xy_l256_256064

theorem compare_xy (m n : ℝ) (h : m ≠ n) :
  let x := m^4 - m^3 * n,
      y := n^3 * m - n^4
  in x > y :=
by
  sorry

end compare_xy_l256_256064


namespace sums_solved_correctly_l256_256762

theorem sums_solved_correctly (x : ℕ) (h : x + 2 * x = 48) : x = 16 := by
  sorry

end sums_solved_correctly_l256_256762


namespace general_term_formula_no_pos_int_for_S_n_gt_40n_plus_600_exists_pos_int_for_S_n_gt_40n_plus_600_l256_256045

noncomputable def arith_seq (n : ℕ) (d : ℝ) :=
  2 + (n - 1) * d

theorem general_term_formula :
  ∃ d, ∀ n, arith_seq n d = 2 ∨ arith_seq n d = 4 * n - 2 :=
by sorry

theorem no_pos_int_for_S_n_gt_40n_plus_600 :
  ∀ n, (arith_seq n 0) * n ≤ 40 * n + 600 :=
by sorry

theorem exists_pos_int_for_S_n_gt_40n_plus_600 :
  ∃ n, (arith_seq n 4) * n > 40 * n + 600 ∧ n = 31 :=
by sorry

end general_term_formula_no_pos_int_for_S_n_gt_40n_plus_600_exists_pos_int_for_S_n_gt_40n_plus_600_l256_256045


namespace scientific_notation_equivalent_l256_256423

theorem scientific_notation_equivalent : ∃ a n, (3120000 : ℝ) = a * 10^n ∧ a = 3.12 ∧ n = 6 :=
by
  exists 3.12
  exists 6
  sorry

end scientific_notation_equivalent_l256_256423


namespace number_of_classes_l256_256802

theorem number_of_classes (sheets_per_class_per_day : ℕ) 
                          (total_sheets_per_week : ℕ) 
                          (school_days_per_week : ℕ) 
                          (h1 : sheets_per_class_per_day = 200) 
                          (h2 : total_sheets_per_week = 9000) 
                          (h3 : school_days_per_week = 5) : 
                          total_sheets_per_week / school_days_per_week / sheets_per_class_per_day = 9 :=
by {
  -- Proof steps would go here
  sorry
}

end number_of_classes_l256_256802


namespace max_soap_bars_purchase_l256_256610

/--
Max has $10 and each soap bar costs $0.95, tax included.
Prove that the maximum number of soap bars Max can purchase is 10.
-/
theorem max_soap_bars_purchase (budget : ℝ) (price_per_bar : ℝ) (h : budget = 10 ∧ price_per_bar = 0.95) :
  ∀ n : ℕ, (n : ℝ) ≤ budget / price_per_bar → n ≤ 10 :=
by
  intro n h
  suffices : (n : ℝ) ≤ 10.5263157894736
  linarith
  sorry

end max_soap_bars_purchase_l256_256610


namespace area_square_II_l256_256675

theorem area_square_II (
  a b : ℝ
  (h_diag : ∀ d₁ : ℝ, d₁ = (a + b * Real.sqrt 2)) :
  ∃ A₂ : ℝ, A₂ = 3 * (a^2 + 2 * a * b * Real.sqrt 2 + 2 * b^2)) :=
by sorry

end area_square_II_l256_256675


namespace parallel_k_eq_neg_third_perpendicular_k_eq_106_third_l256_256458

noncomputable def k_parallel (a b : ℝ × ℝ × ℝ) (k : ℝ) : Prop :=
  ∃ m : ℝ, (k * a.1 + b.1, k * a.2 + b.2, k * a.3 + b.3) = m • (a.1 - 3 * b.1, a.2 - 3 * b.2, a.3 - 3 * b.3)

noncomputable def k_perpendicular (a b : ℝ × ℝ × ℝ) (k : ℝ) : Prop :=
  (k * a.1 + b.1) * (a.1 - 3 * b.1) + (k * a.2 + b.2) * (a.2 - 3 * b.2) + (k * a.3 + b.3) * (a.3 - 3 * b.3) = 0

theorem parallel_k_eq_neg_third :
  k_parallel (1, 5, -1) (-2, 3, 5) (-1 / 3) := sorry

theorem perpendicular_k_eq_106_third :
  k_perpendicular (1, 5, -1) (-2, 3, 5) (106 / 3) := sorry

end parallel_k_eq_neg_third_perpendicular_k_eq_106_third_l256_256458


namespace total_tagged_numbers_l256_256387

theorem total_tagged_numbers:
  let W := 200
  let X := W / 2
  let Y := X + W
  let Z := 400
  W + X + Y + Z = 1000 := by 
    sorry

end total_tagged_numbers_l256_256387


namespace tank_empty_time_l256_256359

def tank_capacity : ℝ := 6480
def leak_time : ℝ := 6
def inlet_rate_per_minute : ℝ := 4.5
def inlet_rate_per_hour : ℝ := inlet_rate_per_minute * 60

theorem tank_empty_time : tank_capacity / (tank_capacity / leak_time - inlet_rate_per_hour) = 8 := 
by
  sorry

end tank_empty_time_l256_256359


namespace value_of_d_is_one_l256_256128

theorem value_of_d_is_one (a b c d w x y z : ℕ) (hw : nat.prime w) (hx : nat.prime x) (hy : nat.prime y) (hz : nat.prime z) (h_order : w < x ∧ x < y ∧ y < z) (h_eq : (w ^ a) * (x ^ b) * (y ^ c) * (z ^ d) = 660) (h_cond : (a + b) - (c + d) = 1) : d = 1 :=
sorry

end value_of_d_is_one_l256_256128


namespace solution_set_abs_inequality_l256_256693

theorem solution_set_abs_inequality : {x : ℝ | |1 - 2 * x| < 3} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end solution_set_abs_inequality_l256_256693


namespace find_area_of_triangle_APQ_l256_256627

/-
  Given an equilateral triangle ABC with side length 10,
  and points P and Q on sides AB and AC such that PQ = 4 and
  PQ is tangent to the incircle of ABC,
  prove that the area of triangle APQ is equal to 5 * sqrt 3 / 3.
-/

def area_of_triangle_APQ  : ℝ :=
  let side_length := 10
  let PQ_length := 4
  let APQ_area := (5 * Real.sqrt 3) / 3
  APQ_area

theorem find_area_of_triangle_APQ :
  ∃ (P Q : (fin 2) → ℝ) (APQ_area: ℝ),
  (P 0).dist (P 1) = PQ_length ∧ (Q 0).dist (Q 1) = PQ_length ∧
  APQ_area = area_of_triangle_APQ ∧ 
  APQ_area = (5 * Real.sqrt 3) / 3 :=
by
  sorry

end find_area_of_triangle_APQ_l256_256627


namespace average_weight_increase_l256_256142

theorem average_weight_increase :
  let n := 5 in
  let initial_weight := 65 in 
  let new_weight := 72.5 in
  let weight_difference := new_weight - initial_weight in
  let avg_increase := weight_difference / n in
  avg_increase = 1.5 := 
by
  sorry

end average_weight_increase_l256_256142


namespace exists_k_satisfying_inequality_l256_256954

theorem exists_k_satisfying_inequality (n : ℕ) (a b : Fin n → ℂ) : 
  ∃ k : Fin n, (∑ i in Finset.range n, complex.abs (a i - a k) ≤ ∑ i in Finset.range n, complex.abs (b i - a k)) 
            ∨ (∑ i in Finset.range n, complex.abs (b i - b k) ≤ ∑ i in Finset.range n, complex.abs (a i - b k)) :=
sorry

end exists_k_satisfying_inequality_l256_256954


namespace find_a_b_l256_256857

theorem find_a_b (a b : ℝ)
  (h1 : (0 - a)^2 + (-12 - b)^2 = 36)
  (h2 : (0 - a)^2 + (0 - b)^2 = 36) :
  a = 0 ∧ b = -6 :=
by
  sorry

end find_a_b_l256_256857


namespace inverse_function_result_l256_256126

def g (x : ℝ) : ℝ := 25 / (4 + 5 * x)

theorem inverse_function_result : (g⁻¹ 5) ^ (-3) = 125 :=
  sorry

end inverse_function_result_l256_256126


namespace fraction_subtraction_l256_256715

theorem fraction_subtraction :
  ((2 + 4 + 6 : ℚ) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12) :=
by
  sorry

end fraction_subtraction_l256_256715


namespace largest_option_l256_256948

-- Define x with 2499 zeros before 235.
def x : ℝ := 235 * 10 ^ (-2502)

-- Define the expressions.
def optionA : ℝ := 4 + x
def optionB : ℝ := 4 - x
def optionC : ℝ := x / 2
def optionD : ℝ := 5 / x
def optionE : ℝ := 5 * x

-- Statement to prove that optionD is the largest.
theorem largest_option : optionD > optionA ∧ optionD > optionB ∧ optionD > optionC ∧ optionD > optionE := by
  sorry

end largest_option_l256_256948


namespace hana_stamp_collection_value_l256_256882

theorem hana_stamp_collection_value : 
  let V : ℚ := 1 in
  let collection_sold := (4/7) + (1/3 * (3/7)) + (1/5 * (2/7)) in
  let fraction_sold := 27/35 in
  28 = (fraction_sold * V) → V = 980 / 27 := by
  sorry

end hana_stamp_collection_value_l256_256882


namespace dots_not_visible_l256_256816

theorem dots_not_visible (total_dice : ℕ) (sum_per_die : ℕ) (visible_dots : ℕ) :
  total_dice = 5 → sum_per_die = 21 → visible_dots = 33 → 
  (total_dice * sum_per_die - visible_dots) = 72 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end dots_not_visible_l256_256816


namespace find_length_of_other_train_l256_256730

def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

theorem find_length_of_other_train :
  ∀ (length_of_first_train speed_of_first_train speed_of_second_train time_cross : ℝ),
  length_of_first_train = 280 →
  speed_of_first_train = 120 →
  speed_of_second_train = 80 →
  time_cross = 9 →
  let relative_speed := kmph_to_mps speed_of_first_train + kmph_to_mps speed_of_second_train in
  let total_distance := relative_speed * time_cross in
  let length_of_other_train := total_distance - length_of_first_train in
  length_of_other_train = 219.95 :=
begin
  intros,
  sorry
end

end find_length_of_other_train_l256_256730


namespace eight_sided_die_divisible_by_48_l256_256379

/--
An eight-sided die is numbered from 1 to 8. When it is rolled, the product \( P \) of the seven numbers that are visible is always divisible by \( 48 \).
-/
theorem eight_sided_die_divisible_by_48 (f : Fin 8 → ℕ)
  (h : ∀ i, f i = i + 1) : 
  ∃ (P : ℕ), (∀ n, P = (∏ i in (Finset.univ.filter (λ j, j ≠ n)), f i)) ∧ (48 ∣ P) := 
by
  sorry

end eight_sided_die_divisible_by_48_l256_256379


namespace initial_money_l256_256806

theorem initial_money (M : ℝ) (h1 : M - (1/4 * M) - (1/3 * (M - (1/4 * M))) = 1600) : M = 3200 :=
sorry

end initial_money_l256_256806


namespace infinitely_many_m_not_prime_l256_256983

theorem infinitely_many_m_not_prime (n : ℕ) : ∃ (m : ℕ), ∃ (p : ℕ), prime p ∧ p % 4 = 3 ∧ m = p^2 ∧ ¬ prime (n^4 + m) :=
by sorry

end infinitely_many_m_not_prime_l256_256983


namespace negation_of_forall_exists_l256_256687

open Classical

theorem negation_of_forall_exists :
  (¬ ∀ x : ℝ, x^2 + sin x + 1 < 0) ↔ ∃ x : ℝ, x^2 + sin x + 1 ≥ 0 := by
  sorry

end negation_of_forall_exists_l256_256687


namespace arthur_walked_distance_in_miles_l256_256777

def blocks_west : ℕ := 8
def blocks_south : ℕ := 10
def block_length_in_miles : ℚ := 1 / 4

theorem arthur_walked_distance_in_miles : 
  (blocks_west + blocks_south) * block_length_in_miles = 4.5 := by
sorry

end arthur_walked_distance_in_miles_l256_256777


namespace evaluate_expression_l256_256245

noncomputable def inverse (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  classical.some (classical.some_spec (classical.some_spec (function.surjective_has_right_inverse 
    (function.bijective_of_finite (classical.some_spec (exists_bijective_finite $ (∀ y, ∃ x, f x = y))) x))))

theorem evaluate_expression 
  (f : ℝ → ℝ) 
  (hf_inv : ∀ x, f (inverse f x) = x ∧ inverse f (f x) = x)
  (h1 : f 3 = 4)
  (h2 : f 5 = 1)
  (h3 : f 2 = 5) : 
  inverse f (inverse f 5 + inverse f 4) = 2 :=
by {
  sorry
}

end evaluate_expression_l256_256245


namespace min_additional_games_l256_256250

def num_initial_games : ℕ := 4
def num_lions_won : ℕ := 3
def num_eagles_won : ℕ := 1
def win_threshold : ℝ := 0.90

theorem min_additional_games (M : ℕ) : (num_eagles_won + M) / (num_initial_games + M) ≥ win_threshold ↔ M ≥ 26 :=
by
  sorry

end min_additional_games_l256_256250


namespace square_difference_l256_256619

variable (n : ℕ)

theorem square_difference (n : ℕ) : (n + 1)^2 - n^2 = 2 * n + 1 :=
sorry

end square_difference_l256_256619


namespace infinite_intersection_l256_256937

noncomputable def S (x : ℝ) : Set ℤ := { n | ∃ (k : ℕ+), n = ⌊↑k * x⌋ }

theorem infinite_intersection (α β : ℝ) (hα : α ∈ Ioo 1 2) (hβ : β ∈ Ioo 2 3)
  (h_roots : ∀ x, (x^3 - 10 * x^2 + 29 * x - 25) = 0 →
    (x = α ∨ x = β ∨ x ∈ Ioo 5 6))
  : Set.Infinite (S α ∩ S β) :=
  sorry

end infinite_intersection_l256_256937


namespace find_k_l256_256963

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def non_collinear (e1 e2 : V) : Prop :=
  ¬ ∃ (a : ℝ), e1 = a • e2

def collinear (v1 v2 : V) : Prop :=
  ∃ (λ : ℝ), v1 = λ • v2

theorem find_k (e1 e2 : V) (k : ℝ) (h1 : e1 ≠ 0) (h2 : e2 ≠ 0) (h3 : non_collinear e1 e2)
  (h4 : collinear (k • e1 + e2) (e1 + k • e2)) : k = 1 ∨ k = -1 := by
  sorry

end find_k_l256_256963
