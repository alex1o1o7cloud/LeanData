import Mathlib
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trig
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Polynomial
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.NumberTheory.Pi
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import RealFloor

namespace toothpicks_needed_base_1001_l640_640281

-- Define the number of small triangles at the base of the larger triangle
def base_triangle_count := 1001

-- Define the total number of small triangles using the sum of the first 'n' natural numbers
def total_small_triangles (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Calculate the total number of sides for all triangles if there was no sharing
def total_sides (n : ℕ) : ℕ :=
  3 * total_small_triangles n

-- Calculate the number of shared toothpicks
def shared_toothpicks (n : ℕ) : ℕ :=
  total_sides n / 2

-- Calculate the number of unshared perimeter toothpicks
def unshared_perimeter_toothpicks (n : ℕ) : ℕ :=
  3 * n

-- Calculate the total number of toothpicks required
def total_toothpicks (n : ℕ) : ℕ :=
  shared_toothpicks n + unshared_perimeter_toothpicks n

-- Prove that the total toothpicks required for the base of 1001 small triangles is 755255
theorem toothpicks_needed_base_1001 : total_toothpicks base_triangle_count = 755255 :=
by {
  sorry
}

end toothpicks_needed_base_1001_l640_640281


namespace sum_of_4_inclusive_numbers_l640_640261

def is_4_inclusive (n : ℕ) : Prop :=
  (n % 4 = 0) ∨ (n.digits 10).contains 4

theorem sum_of_4_inclusive_numbers : 
  (Finset.filter is_4_inclusive (Finset.range 101)).sum id = 1883 :=
by
  sorry

end sum_of_4_inclusive_numbers_l640_640261


namespace equilateral_triangle_l640_640643

variables {P_0 P_1986 : Point} {A_1 A_2 A_3 : Point} {P : ℕ → Point}

-- Define the cyclic points A_s
def A (s : ℕ) : Point := if s % 3 = 1 then A_1 else if s % 3 = 2 then A_2 else A_3

-- Define the sequence of points P_k that rotate around A_{k+1}
def P_sequence (k : ℕ) : Point := 
  if k = 0 then P_0 
  else rotate (A k) 120 (P_sequence (k - 1))

-- Condition that P_1986 is the same as P_0
axiom P_1986_eq_P_0 : P_sequence 1986 = P_0

-- The theorem proving the main question
theorem equilateral_triangle {A_1 A_2 A_3 : Point} (P_sequence) (P_1986_eq_P_0 : P_sequence 1986 = P_0) :
  is_equilateral_triangle A_1 A_2 A_3 :=
sorry

end equilateral_triangle_l640_640643


namespace convex_quadrilateral_inequality_l640_640642

variable (a b c d x y : ℝ)
variable (ABCD : ConvexQuadrilateral)
variable (h1 : ABCD.side a b c d)
variable (h2 : ABCD.diagonal x y)
variable (h3 : a ≤ b)
variable (h4 : a ≤ c)
variable (h5 : a ≤ d)
variable (h6 : a ≤ x)
variable (h7 : a ≤ y)

theorem convex_quadrilateral_inequality (ABCD : ConvexQuadrilateral)
  (h1 : ABCD.side a b c d)
  (h2 : ABCD.diagonal x y)
  (h3 : a ≤ b)
  (h4 : a ≤ c)
  (h5 : a ≤ d)
  (h6 : a ≤ x)
  (h7 : a ≤ y) : 
  x + y ≥ (1 + Real.sqrt 3) * a := 
by
  sorry

end convex_quadrilateral_inequality_l640_640642


namespace geometric_log_sum_l640_640758

theorem geometric_log_sum (a₁ a₂ : ℝ) (q : ℝ) (h₁ : a₁ + a₂ = 11) (h₂ : a₁ * a₂ = 10)
  (h₃ : ∀ n, (aₙ = a₁ * q^(n-1))) :
  ∑ i in Finset.range 10, log (a₁ * q ^ i) = -35 :=
by sorry

end geometric_log_sum_l640_640758


namespace intersect_rectangular_eqn_range_of_m_l640_640359

-- Definitions of the parametric and polar equations
def curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Proving the rectangular equation and range of m for the intersection of l and C
theorem intersect_rectangular_eqn (m t : ℝ) :
  (∃ ρ θ, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  ∃ x y, (x, y) = curve_C t ∧ √3 * x + y + 2 * m = 0 :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ ρ θ t, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end intersect_rectangular_eqn_range_of_m_l640_640359


namespace rectangular_eq_of_line_l_range_of_m_l640_640348

noncomputable def curve_C_param_eq (t : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l_polar_eq (θ ρ m : ℝ) : Prop :=
  ρ * Real.sin(θ + Real.pi / 3) + m = 0

theorem rectangular_eq_of_line_l (x y m : ℝ) (h : line_l_polar_eq (Real.atan2 y x) (Real.sqrt (x^2 + y^2)) m) :
  sqrt 3 * x + y + 2 * m = 0 :=
  sorry

theorem range_of_m (m : ℝ) (t : ℝ) (h : ∃ (t : ℝ), curve_C_param_eq t ∈ set_of (λ p, p.2 = -√3 * p.1 - 2 * m))
  : -19/12 ≤ m ∧ m ≤ 5/2 :=
  sorry

end rectangular_eq_of_line_l_range_of_m_l640_640348


namespace intersect_rectangular_eqn_range_of_m_l640_640356

-- Definitions of the parametric and polar equations
def curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Proving the rectangular equation and range of m for the intersection of l and C
theorem intersect_rectangular_eqn (m t : ℝ) :
  (∃ ρ θ, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  ∃ x y, (x, y) = curve_C t ∧ √3 * x + y + 2 * m = 0 :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ ρ θ t, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end intersect_rectangular_eqn_range_of_m_l640_640356


namespace average_price_of_pen_l640_640863

theorem average_price_of_pen :
  (let num_pens := 30
       num_pencils := 75
       total_cost := 510
       price_per_pencil := 2
       total_cost_pencils := num_pencils * price_per_pencil
       total_cost_pens := total_cost - total_cost_pencils
       price_per_pen := total_cost_pens / num_pens
   in price_per_pen = 12) := 
by
  sorry

end average_price_of_pen_l640_640863


namespace range_of_func_l640_640803

def func (x : Int) : Int :=
  abs x - 1

theorem range_of_func : Set (func '' {-1, 0, 1, 2, 3}) = {-1, 0, 1, 2} :=
by
  sorry

end range_of_func_l640_640803


namespace convert_base_10_to_base_8_l640_640918

theorem convert_base_10_to_base_8 (n : ℕ) (n_eq : n = 3275) : 
  n = 3275 → ∃ (a b c d : ℕ), (a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0 = 6323) :=
by 
  sorry

end convert_base_10_to_base_8_l640_640918


namespace find_smallest_d_l640_640190

-- Given conditions: The known digits sum to 26
def sum_known_digits : ℕ := 5 + 2 + 4 + 7 + 8 

-- Define the smallest digit d such that 52,d47,8 is divisible by 9
def smallest_d (d : ℕ) (sum_digits_with_d : ℕ) : Prop :=
  sum_digits_with_d = sum_known_digits + d ∧ (sum_digits_with_d % 9 = 0)

theorem find_smallest_d : ∃ d : ℕ, smallest_d d 27 :=
sorry

end find_smallest_d_l640_640190


namespace eccentricity_of_ellipse_l640_640671

theorem eccentricity_of_ellipse (F₁ F₂ : Point) (P : Point) (C : Ellipse)
  (h1 : F₁ ≠ F₂)
  (h2 : P ∈ C)
  (h3 : ∀ P ∈ C, max (dist P F₁) / min (dist P F₂) = 3) :
  eccentricity C = 1 / 2 := 
sorry

end eccentricity_of_ellipse_l640_640671


namespace no_integer_roots_of_polynomial_l640_640934

theorem no_integer_roots_of_polynomial :
  ¬ ∃ x : ℤ, x^3 - 4 * x^2 - 14 * x + 28 = 0 :=
by
  sorry

end no_integer_roots_of_polynomial_l640_640934


namespace congruence_a_b_mod_1008_l640_640001

theorem congruence_a_b_mod_1008
  (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : a ^ b - b ^ a = 1008) : a ≡ b [MOD 1008] :=
sorry

end congruence_a_b_mod_1008_l640_640001


namespace BAGAAnswerVariation_l640_640605

-- Definition of BAGA problems and their analogy
def BAGAToLogAnalogy (problem: Type) := 
  ∀ p : problem, (transform_bagel_to_log p) = true

-- The statement to prove
theorem BAGAAnswerVariation (problem : Type) (P : problem → Prop)
  (H : BAGAToLogAnalogy problem) : 
  ∀ p1 p2 : problem, (P p1 ≠ P p2) := 
sorry

end BAGAAnswerVariation_l640_640605


namespace relationship_among_abc_l640_640217

def a : ℝ := 2 ^ 1.3
def b : ℝ := (1 / 4) ^ 0.7
def c : ℝ := Real.log 8 / Real.log 3

theorem relationship_among_abc : b < c ∧ c < a := by
  sorry

end relationship_among_abc_l640_640217


namespace polar_to_rectangular_range_of_m_l640_640294

-- Definition of the parametric equations of curve C
def curve (t : ℝ) : ℝ × ℝ :=
  (sqrt(3) * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line (rho theta m : ℝ) : Prop :=
  rho * sin (theta + π / 3) + m = 0

-- Definition of the rectangular equation of line l
def rectangular_line (x y m : ℝ) : Prop :=
  sqrt(3) * x + y + 2 * m = 0

-- Convert polar line equation to rectangular form
theorem polar_to_rectangular (rho theta m x y : ℝ)
  (h1 : polar_line rho theta m)
  (h2 : rho = sqrt (x^2 + y^2))
  (h3 : cos theta * rho = x ∧ sin theta * rho = y) :
  rectangular_line x y m :=
sorry

-- Range of values for m for which l intersects C
theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, (sqrt(3) * cos (2 * t), 2 * sin t) ∈ {p : ℝ × ℝ | rectangular_line p.1 p.2 m}) ↔
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end polar_to_rectangular_range_of_m_l640_640294


namespace dice_probability_l640_640088

theorem dice_probability :
  let primes := {2, 3, 5, 7, 11}
  let non_primes := {1, 4, 6, 8, 9, 10, 12}
  let favorable_permutations := (Fact (Nat.factorial 5)) * (Fact (Nat.factorial 7))
  let total_permutations := Fact (Nat.factorial 12)
  let prob_fraction := favorable_permutations / total_permutations in
  let reduced_fraction := Rational.mk 1 792 in
  (prob_fraction.num + prob_fraction.denom) = (reduced_fraction.num + reduced_fraction.denom)
:=
  sorry

end dice_probability_l640_640088


namespace rect_eq_line_range_of_m_l640_640332

-- Definitions based on the conditions
def parametric_x (t : ℝ) : ℝ := sqrt 3 * cos (2 * t)
def parametric_y (t : ℝ) : ℝ := 2 * sin t

def polar_rho (θ m : ℝ) : ℝ := -(m / (sin (θ + π / 3)))

-- Rectangular equation of the line l
theorem rect_eq_line (x y m : ℝ) : polar_rho (atan2 y x) m * sin (atan2 y x + π / 3) + m = 0 → sqrt 3 * x + y + 2 * m = 0 :=
by
  sorry

-- Range of values for m
theorem range_of_m (C : ℝ → ℝ × ℝ)
  (hC : ∀ t, C t = (parametric_x t, parametric_y t)) (m : ℝ) :
  (∃ t, sqrt 3 * fst (C t) + snd (C t) + 2 * m = 0) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
by
  sorry

end rect_eq_line_range_of_m_l640_640332


namespace complete_collection_prob_l640_640050

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end complete_collection_prob_l640_640050


namespace evaluate_g_of_neg_one_l640_640919

def g (x : ℤ) : ℤ :=
  x^2 - 2*x + 1

theorem evaluate_g_of_neg_one :
  g (g (g (g (g (g (-1 : ℤ)))))) = 15738504 := by
  sorry

end evaluate_g_of_neg_one_l640_640919


namespace solve_for_x_l640_640941

theorem solve_for_x : ∃ x : ℝ, 5 * x + 15 = 225 ∧ x = 42 :=
by {
  have h : 5 * 42 + 15 = 225,
  calc
    5 * 42 + 15 
        = 210 + 15 : by rw [mul_comm 5 42]
    ... = 225 : by norm_num,
  use 42,
  split,
  exact h,
  refl,
  }

end solve_for_x_l640_640941


namespace ab_ne_9_of_roots_on_circle_l640_640748

theorem ab_ne_9_of_roots_on_circle (a b : ℂ) (P : Polynomial ℂ) (hP : P = Polynomial.C a * X^3 + Polynomial.C b * X^2 + X) :
  (∃ z : ℂ, ∀ w ∈ {0..3}, P.eval w = 0) → ab ≠ 9 :=
by sorry

end ab_ne_9_of_roots_on_circle_l640_640748


namespace mom_approach_is_sampling_survey_l640_640528

def is_sampling_survey (action : String) : Prop :=
  action = "tasting a little bit"

def is_census (action : String) : Prop :=
  action = "tasting the entire dish"

theorem mom_approach_is_sampling_survey :
  is_sampling_survey "tasting a little bit" :=
by {
  -- This follows from the given conditions directly.
  sorry
}

end mom_approach_is_sampling_survey_l640_640528


namespace spinner_probability_divisible_by_5_l640_640930

theorem spinner_probability_divisible_by_5 :
  let outcomes := {2, 3, 5, 0}
  ∃ (count : ℕ) (spinner : ℕ → ℕ → ℕ),
    count = 4^3 ∧
    (∀ a b c, a ∈ outcomes ∧ b ∈ outcomes ∧ c ∈ outcomes →
     (spinner a b c = a * 100 + b * 10 + c)) →
    (let favorable := (finset.filter (λ n, n % 5 = 0) (finset.univ.image (λ ⟨a, b, c⟩, spinner a b c)))
     in favorable.card / count = 1 / 2) := sorry

end spinner_probability_divisible_by_5_l640_640930


namespace pipe_q_fill_time_l640_640505

theorem pipe_q_fill_time :
  ∀ (T : ℝ), (2 * (1 / 10 + 1 / T) + 10 * (1 / T) = 1) → T = 15 :=
by
  intro T
  intro h
  sorry

end pipe_q_fill_time_l640_640505


namespace edge_count_bound_of_no_shared_edge_triangle_l640_640752

variables {V : Type} [Fintype V]

-- Definition of a graph without two triangles sharing an edge
def no_shared_edge_triangle_graph (G : SimpleGraph V) : Prop :=
  ¬∃ u v w x : V, G.Adj u v ∧ G.Adj v w ∧ G.Adj w u ∧ G.Adj u x ∧ G.Adj v x

-- Proposition
theorem edge_count_bound_of_no_shared_edge_triangle (G : SimpleGraph V) (n : ℕ) 
  (hV : Fintype.card V = 2 * n) (hG : no_shared_edge_triangle_graph G) : 
  G.edge_finset.card ≤ n^2 + 1 :=
sorry

end edge_count_bound_of_no_shared_edge_triangle_l640_640752


namespace exp_expression_l640_640712

theorem exp_expression (x y E : ℝ) (h : x * y = 1) (h1 : (5^E)^2 / (5^(x - y)^2) = 625) : E = 2 :=
sorry

end exp_expression_l640_640712


namespace intersection_is_A_l640_640992

-- Define the set M based on the given condition
def M : Set ℝ := {x | x / (x - 1) ≥ 0}

-- Define the set N based on the given condition
def N : Set ℝ := {x | ∃ y, y = 3 * x^2 + 1}

-- Define the set A as the intersection of M and N
def A : Set ℝ := {x | x > 1}

-- Prove that the intersection of M and N is equal to the set A
theorem intersection_is_A : (M ∩ N = A) :=
by {
  sorry
}

end intersection_is_A_l640_640992


namespace remainder_x3_minus_2x2_plus_4x_minus_1_div_x_minus_2_l640_640188

/--
  Problem: Find the remainder when \( x^3 - 2x^2 + 4x - 1 \) is divided by \( x - 2 \).
  Solution: By the Remainder Theorem, substituting \( x = 2 \) into the polynomial gives \( 7 \).
-/
theorem remainder_x3_minus_2x2_plus_4x_minus_1_div_x_minus_2 :
  let f := λ x : ℝ, x^3 - 2 * x^2 + 4 * x - 1
  in f 2 = 7 :=
by
  let f := λ x : ℝ, x^3 - 2 * x^2 + 4 * x - 1
  show f 2 = 7
  sorry

end remainder_x3_minus_2x2_plus_4x_minus_1_div_x_minus_2_l640_640188


namespace find_b_l640_640715

noncomputable def angle_B : ℝ := 60
noncomputable def c : ℝ := 8
noncomputable def diff_b_a (b a : ℝ) : Prop := b - a = 4

theorem find_b (b a : ℝ) (h₁ : angle_B = 60) (h₂ : c = 8) (h₃ : diff_b_a b a) :
  b = 7 :=
sorry

end find_b_l640_640715


namespace find_integer_n_l640_640701

theorem find_integer_n (n : ℤ) : (⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋ ^ 2 = 3) → n = 7 :=
by sorry

end find_integer_n_l640_640701


namespace det_dilation_matrix_l640_640757

-- Let E be the 3 x 3 matrix corresponding to the dilation with scale factor 5 centered at the origin
def E : Matrix (Fin 3) (Fin 3) ℝ := ![![5, 0, 0], ![0, 5, 0], ![0, 0, 5]]

-- Prove that the determinant of E is 125
theorem det_dilation_matrix :
  det E = 125 :=
by 
  sorry

end det_dilation_matrix_l640_640757


namespace domain_of_f_l640_640792

open Set

noncomputable def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f y = x}

noncomputable def f (x : ℝ) : ℝ := (real.sqrt (x - 1)) / (x - 3) + (x - 1) ^ 0

theorem domain_of_f :
  domain f = {x | 1 < x ∧ x ≠ 3} :=
by {
  sorry
}

end domain_of_f_l640_640792


namespace total_surface_area_of_rectangular_solid_l640_640842

theorem total_surface_area_of_rectangular_solid (L W D : ℝ) 
  (hL : L = 5) (hW : W = 4) (hD : D = 1) : 
  2 * (L * W + W * D + L * D) = 58 :=
by
  -- Given conditions
  rw [hL, hW, hD]
  -- Calculation based on the conditions and definition of total surface area
  calc
    2 * (L * W + W * D + L * D)
    = 2 * (5 * 4 + 4 * 1 + 5 * 1) : by rw [hL, hW, hD]
... = 2 * (20 + 4 + 5) : by norm_num
... = 2 * 29 : by norm_num
... = 58 : by norm_num

end total_surface_area_of_rectangular_solid_l640_640842


namespace amount_after_two_years_l640_640604

/-- Defining given conditions. -/
def initial_value : ℤ := 65000
def first_year_increase : ℚ := 12 / 100
def second_year_increase : ℚ := 8 / 100

/-- The main statement that needs to be proved. -/
theorem amount_after_two_years : 
  let first_year_amount := initial_value + (initial_value * first_year_increase)
  let second_year_amount := first_year_amount + (first_year_amount * second_year_increase)
  second_year_amount = 78624 := 
by 
  sorry

end amount_after_two_years_l640_640604


namespace probability_at_least_three_of_five_operational_l640_640802

open_locale big_operators -- Enable Big-Operators localization for sums/prod

theorem probability_at_least_three_of_five_operational (p : ℝ) (q : ℝ) (X : ℕ → ℝ) :
  p = 0.2 →
  q = 0.8 →
  (∀ n k, X k = (nat.choose n k : ℝ) * p^k * q^(n - k)) →
  (X 5 3 + X 5 4 + X 5 5 = 0.06) :=
begin
  intros hp hq hX,
  sorry -- Proof goes here
end

end probability_at_least_three_of_five_operational_l640_640802


namespace find_numbers_l640_640826

-- Define the conditions
def geometric_mean_condition (a b : ℝ) : Prop :=
  a * b = 3

def harmonic_mean_condition (a b : ℝ) : Prop :=
  2 / (1 / a + 1 / b) = 3 / 2

-- State the theorem to be proven
theorem find_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  geometric_mean_condition a b ∧ harmonic_mean_condition a b → (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 1) := 
by 
  sorry

end find_numbers_l640_640826


namespace range_g_on_interval_l640_640595

def g (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem range_g_on_interval : 
  (set.image g (set.Icc 0 4) = set.Icc (-5 : ℝ) 4) :=
by
  sorry

end range_g_on_interval_l640_640595


namespace promote_type_A_l640_640869

variable (yields_A : List ℕ) (yields_B : List ℕ)
variable (season_count : ℕ) 

def average (yields : List ℕ) : ℚ :=
  (yields.foldl (λ acc x => acc + (x : ℚ)) 0) / season_count

def variance (yields : List ℕ) (mean : ℚ) : ℚ :=
  (yields.foldl (λ acc x => acc + ((x : ℚ) - mean) ^ 2) 0) / season_count

theorem promote_type_A
  (yields_A : List ℕ) 
  (yields_B : List ℕ)
  (season_count : ℕ) 
  (h_size_A : yields_A.length = season_count)
  (h_size_B : yields_B.length = season_count)
  : yields_A = [550, 580, 570, 570, 550, 600] →
    yields_B = [540, 590, 560, 580, 590, 560] →
    average yields_A = 570 →
    average yields_B = 570 →
    let var_A := variance yields_A 570
    let var_B := variance yields_B 570
    var_A < var_B →
    "type A rice"
:=
by
  intros
  sorry

end promote_type_A_l640_640869


namespace train_length_is_correct_l640_640091

noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_s

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 :=
by 
  -- Here, a proof would be provided, eventually using the definitions and conditions given
  sorry

end train_length_is_correct_l640_640091


namespace plane_equation_through_point_normal_vector_l640_640795

theorem plane_equation_through_point_normal_vector
  (A B C D : ℤ)
  (point : ℝ × ℝ × ℝ)
  (normal_vector : ℝ × ℝ × ℝ)
  (h1 : point = (12, -4, 3))
  (h2 : normal_vector = (12, -4, 3))
  (h3 : A = 12)
  (h4 : B = -4)
  (h5 : C = 3)
  (h6 : D = -169)
  (h7 : gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1)
  (h8 : A > 0) :
  A * point.1 + B * point.2 + C * point.3 + D = 0 :=
  sorry

end plane_equation_through_point_normal_vector_l640_640795


namespace surface_area_pyramid_l640_640402

noncomputable def triangle (A B C : Type) := sorry
noncomputable def pyramid (A B C D : Type) := sorry

variables {A B C D : Type}

-- Assume the conditions as definitions
def is_triangle (ABC : triangle A B C) : Prop := sorry
def is_pyramid (DABC : pyramid A B C D) : Prop := sorry
def valid_edges (ABC : triangle A B C) (lengths : set ℕ) : Prop := lengths = {13, 34}
def no_equilateral (ABC : triangle A B C) : Prop := sorry

-- Prove the surface area statement given the conditions
theorem surface_area_pyramid 
  (ABC : triangle A B C) (DABC : pyramid A B C D)
  (h1 : is_triangle ABC) 
  (h2 : is_pyramid DABC) 
  (h3 : valid_edges ABC {13, 34}) 
  (h4 : no_equilateral ABC) : 
  (calculate_surface_area DABC = 867.88) :=
sorry

end surface_area_pyramid_l640_640402


namespace jelly_beans_total_l640_640424

-- Definitions from the conditions
def vanilla : Nat := 120
def grape : Nat := 5 * vanilla + 50
def total : Nat := vanilla + grape

-- Statement to prove
theorem jelly_beans_total :
  total = 770 := 
by 
  sorry

end jelly_beans_total_l640_640424


namespace train_passes_man_in_time_l640_640090

noncomputable def time_to_pass (train_length : ℝ) (train_speed_kmh : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh + man_speed_kmh
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  train_length / relative_speed_ms

theorem train_passes_man_in_time :
  time_to_pass 110 80 8 ≈ 4.5 := by
  -- Usually, you would provide a detailed proof here.
  -- For the purpose of this task, we simply provide a justification placeholder.
  sorry

end train_passes_man_in_time_l640_640090


namespace calculate_power_of_fractions_l640_640579

-- Defining the fractions
def a : ℚ := 5 / 6
def b : ℚ := 3 / 5

-- The main statement to prove the given question
theorem calculate_power_of_fractions : a^3 + b^3 = (21457 : ℚ) / 27000 := by 
  sorry

end calculate_power_of_fractions_l640_640579


namespace retain_exactly_five_coins_l640_640194

-- Define the problem setup
structure GameSetup :=
  (players : Finset (String)) -- Five friends: "Abby", "Bernardo", "Carl", "Debra", "Elina"
  (initial_coins : ℕ := 5)    -- Each player starts with 5 coins
  (rounds : ℕ := 3)           -- Number of rounds
  (urn : Finset (String) := {"green", "red", "white", "white", "white"}) -- Balls in the urn

-- The game conditions
def game_conditions (setup : GameSetup) : Prop :=
  setup.players = {"Abby", "Bernardo", "Carl", "Debra", "Elina"} ∧
  setup.initial_coins = 5 ∧
  setup.rounds = 3 ∧
  setup.urn = {"green", "red", "white", "white", "white"}

-- The statement to prove: Probability that everyone still has exactly 5 coins at the end of 3 rounds
theorem retain_exactly_five_coins (setup : GameSetup) (h : game_conditions setup) :
  (1 / 10 : ℚ) ^ 3 = 1 / 1000 :=
by
  sorry

end retain_exactly_five_coins_l640_640194


namespace problem_final_value_l640_640255

theorem problem_final_value (x y z : ℝ) (hz : z ≠ 0) 
  (h1 : 3 * x - 2 * y - 2 * z = 0) 
  (h2 : x - 4 * y + 8 * z = 0) :
  (3 * x^2 - 2 * x * y) / (y^2 + 4 * z^2) = 120 / 269 := 
by 
  sorry

end problem_final_value_l640_640255


namespace area_of_trapezium_is_105_l640_640184

-- Define points in the coordinate plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨14, 3⟩
def C : Point := ⟨18, 10⟩
def D : Point := ⟨0, 10⟩

noncomputable def length (p1 p2 : Point) : ℝ := abs (p2.x - p1.x)
noncomputable def height (p1 p2 : Point) : ℝ := abs (p2.y - p1.y)

-- Calculate lengths of parallel sides AB and CD, and height
noncomputable def AB := length A B
noncomputable def CD := length C D
noncomputable def heightAC := height A C

-- Define the area of trapezium
noncomputable def area_trapezium (AB CD height : ℝ) : ℝ := (1/2) * (AB + CD) * height

-- The proof problem statement
theorem area_of_trapezium_is_105 :
  area_trapezium AB CD heightAC = 105 := by
  sorry

end area_of_trapezium_is_105_l640_640184


namespace min_students_with_same_score_l640_640279

theorem min_students_with_same_score
  (highest_score lowest_score : ℤ) (total_students : ℕ)
  (h1 : highest_score = 83) (h2 : lowest_score = 30) (h3 : total_students = 8000) :
  ∃ (n : ℕ), n ≥ 149 ∧ ∀ score : ℤ, lowest_score ≤ score ∧ score ≤ highest_score → n = (8000 / 54).ceil :=
sorry

end min_students_with_same_score_l640_640279


namespace Julie_work_hours_per_week_l640_640740

theorem Julie_work_hours_per_week 
  (hours_summer_per_week : ℕ)
  (weeks_summer : ℕ)
  (total_earnings_summer : ℕ)
  (planned_weeks_school_year : ℕ)
  (needed_income_school_year : ℕ)
  (hourly_wage : ℝ := total_earnings_summer / (hours_summer_per_week * weeks_summer))
  (total_hours_needed_school_year : ℝ := needed_income_school_year / hourly_wage)
  (hours_per_week_needed : ℝ := total_hours_needed_school_year / planned_weeks_school_year) :
  hours_summer_per_week = 60 →
  weeks_summer = 8 →
  total_earnings_summer = 6000 →
  planned_weeks_school_year = 40 →
  needed_income_school_year = 10000 →
  hours_per_week_needed = 20 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end Julie_work_hours_per_week_l640_640740


namespace total_jelly_beans_l640_640428

theorem total_jelly_beans (vanilla_jelly_beans : ℕ) (h1 : vanilla_jelly_beans = 120) 
                           (grape_jelly_beans : ℕ) (h2 : grape_jelly_beans = 5 * vanilla_jelly_beans + 50) :
                           vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  sorry

end total_jelly_beans_l640_640428


namespace triangle_inequality_l640_640968

theorem triangle_inequality
  (a b c A B C : ℝ)
  (h1 : a + b + c = 1)
  (h2 : A + B + C = Real.pi)
  (h3 : ∀ {a' b' c' A' B' C' : ℝ}, A' + B' + C' = Real.pi → a' + b' + c' = 1 → 
    A' = acos ((b' * b' + c' * c' - a' * a') / (2 * b' * c')) →
    B' = acos ((a' * a' + c' * c' - b' * b') / (2 * a' * c')) →
    C' = acos ((a' * a' + b' * b' - c' * c') / (2 * a' * b'))) :
  (a + b) * Real.sin (C / 2) + (b + c) * Real.sin (A / 2) 
  + (c + a) * Real.sin (B / 2) ≤ a + b + c := 
sorry

end triangle_inequality_l640_640968


namespace frank_steps_forward_l640_640623

theorem frank_steps_forward :
  let initial_pos := 0 in
  let final_pos := initial_pos - 5 + 10 - 2 + 2 * 2 in
  final_pos = 7 :=
by
  sorry

end frank_steps_forward_l640_640623


namespace work_days_B_l640_640109

theorem work_days_B (A_days B_days : ℕ) (hA : A_days = 12) (hTogether : (1/12 + 1/A_days) = (1/8)) : B_days = 24 := 
by
  revert hTogether -- reversing to tackle proof
  sorry

end work_days_B_l640_640109


namespace num_goldfish_is_three_l640_640037

/-
There are some goldfish and ten platyfish in a fish tank. Each goldfish plays with ten red balls,
while each platyfish plays with five white balls. There are a total of 80 balls in the fish tank.
Prove that the number of goldfish is 3.
-/

def num_goldfish_condition (G : ℕ) : Prop :=
  let num_platyfish := 10 in
  let red_balls_per_goldfish := 10 in
  let white_balls_per_platyfish := 5 in
  let total_balls := 80 in
  red_balls_per_goldfish * G + white_balls_per_platyfish * num_platyfish = total_balls

theorem num_goldfish_is_three : num_goldfish_condition 3 :=
sorry -- proof omitted

end num_goldfish_is_three_l640_640037


namespace total_birds_and_storks_l640_640818

-- Definitions of initial counts and changes
def initial_sparrows : ℕ := 3
def initial_storks : ℕ := 2
def initial_pigeons : ℕ := 4

def joined_swallows : ℕ := 5
def joined_sparrows : ℕ := 3
def flew_away_pigeons : ℕ := 2

-- Definition of current counts
def total_sparrows : ℕ := initial_sparrows + joined_sparrows
def total_storks : ℕ := initial_storks
def total_pigeons : ℕ := initial_pigeons - flew_away_pigeons
def total_swallows : ℕ := joined_swallows

-- Total number of birds including storks
def total_birds : ℕ := total_sparrows + total_storks + total_pigeons + total_swallows

-- Lean statement to prove the total number of birds and storks currently on the fence
theorem total_birds_and_storks : total_birds = 15 :=
by
  -- Definitions and assumptions based on conditions
  have h1 : initial_sparrows = 3 := rfl
  have h2 : initial_storks = 2 := rfl
  have h3 : initial_pigeons = 4 := rfl
  have h4 : joined_swallows = 5 := rfl
  have h5 : joined_sparrows = 3 := rfl
  have h6 : flew_away_pigeons = 2 := rfl
  
  -- Calculations based on solution steps
  have sparrows := 3 + 3
  have storks := 2
  have pigeons := 4 - 2
  have swallows := 5
  
  -- Assert the total number of birds
  have total := sparrows + storks + pigeons + swallows
  rw [sparrows, storks, pigeons, swallows] at total
  change total with 6 + 2 + 2 + 5
  norm_num
  exact rfl

end total_birds_and_storks_l640_640818


namespace even_number_three_colored_vertices_l640_640148

-- Definitions based on given conditions
def is_vertex_of_polyhedron (P : Type) [convex_polyhedron P] (v : vertex P) : Prop := true

def three_faces_meet_at (P : Type) [convex_polyhedron P] (v : vertex P) (c1 c2 c3 : Color) : Prop :=
  v.is_vertex_of_polyhedron P ∧ (∃ f1 f2 f3 : face P, 
  f1.contains_vertex v ∧ f2.contains_vertex v ∧ f3.contains_vertex v ∧ 
  f1.color = c1 ∧ f2.color = c2 ∧ f3.color = c3)

def num_three_colored_vertices (P : Type) [convex_polyhedron P] : Nat := sorry

-- Problem Statement
theorem even_number_three_colored_vertices (P : Type) [convex_polyhedron P]
  (colors : face P → Color):
  is_polyhedron_convex P →
  (∀ v : vertex P, three_faces_meet_at P v colors.red colors.yellow colors.blue ∨
                  three_faces_meet_at P v colors.red colors.blue colors.yellow ∨
                  three_faces_meet_at P v colors.blue colors.red colors.yellow
  ) →
  ∃ n : ℕ, num_three_colored_vertices P % 2 = 0 :=
by
  intro h1 h2
  sorry

end even_number_three_colored_vertices_l640_640148


namespace tan_sum_correct_l640_640977

noncomputable def tan_sum (α : ℝ) (h0 : 0 < α) (h1 : α < π / 2) (h2 : cos (2 * α) + cos α ^ 2 = 0) : ℝ :=
tan (α + π / 4)

theorem tan_sum_correct (α : ℝ) (h0 : 0 < α) (h1 : α < π / 2) (h2 : cos (2 * α) + cos α ^ 2 = 0) :
  tan_sum α h0 h1 h2 = -3 - 2 * sqrt 2 :=
sorry

end tan_sum_correct_l640_640977


namespace rooms_with_people_after_one_hour_l640_640812

theorem rooms_with_people_after_one_hour :
  ∃ nums : list ℕ, length nums = 1000 ∧ nums.head = 940 ∧ (forall i : ℕ, 1 ≤ i ∧ i ≤ 31 → nums.nth i = some 2) ∧ (forall i : ℕ, 32 ≤ i → nums.nth i = some 0) := sorry

end rooms_with_people_after_one_hour_l640_640812


namespace sum_of_positive_differences_l640_640395

noncomputable def S : Finset ℕ := (Finset.range 11).image (fun x => 3^x)

def N : ℕ := ∑ i in S.attach, ∑ j in S.attach, if (i.val > j.val) then (i.val - j.val) else 0

theorem sum_of_positive_differences :
  N = 779409 :=
sorry

end sum_of_positive_differences_l640_640395


namespace smallest_b_l640_640072

theorem smallest_b (b: ℕ) (h1: b > 3) (h2: ∃ n: ℕ, n^3 = 2 * b + 3) : b = 12 :=
sorry

end smallest_b_l640_640072


namespace matrix_not_invertible_y_l640_640167

theorem matrix_not_invertible_y :
  ∃ y : ℚ, det ![![2 * y, 5], ![4 - y, 9]] = 0 ↔ y = 20 / 23 := 
by
  sorry

end matrix_not_invertible_y_l640_640167


namespace smallest_prime_dividing_sum_l640_640524

theorem smallest_prime_dividing_sum (a b : ℕ) (h_a : a = 7) (h_b : b = 11) :
  ∃ p : ℕ, prime p ∧ p ∣ (a^13 + b^15) ∧ (∀ q : ℕ, prime q ∧ q ∣ (a^13 + b^15) → p ≤ q) :=
sorry

end smallest_prime_dividing_sum_l640_640524


namespace dice_probability_l640_640864

-- Definitions for the problem setup
def num_20_sided_dice : ℕ := 6
def num_one_digit : ℕ := 9
def num_two_digit : ℕ := 11
def p_one_digit : ℚ := 9 / 20
def p_two_digit : ℚ := 11 / 20

-- Binomial coefficient for choosing 3 out of 6 dice
noncomputable def binom_6_3 : ℕ := combinatorics.binom 6 3

-- Calculation for the probability
noncomputable def probability : ℚ := binom_6_3 * (p_one_digit^3) * (p_two_digit^3)

-- The theorem to prove
theorem dice_probability:
  probability = 969969 / 32000000 := 
sorry

end dice_probability_l640_640864


namespace induction_step_term_l640_640832

theorem induction_step_term (k : ℕ) (h : 0 < k) :
  let lhs_k := (finset.range k).sum (λ i, (1 : ℝ) / (k + 1 + i))
  let lhs_k1 := (finset.range (k + 1)).sum (λ i, (1 : ℝ) / (k + 2 + i))
  (lhs_k1 - lhs_k) = (1 / (2 * k + 1) + 1 / (2 * k + 2) - 1 / (k + 1)) :=
by {
    -- Here would be the proof steps if required
    sorry
}

end induction_step_term_l640_640832


namespace rect_eq_and_range_of_m_l640_640310

noncomputable def rect_eq_of_polar (m : ℝ) : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ * sin (θ + π / 3) + m = 0 → sqrt 3 * (ρ * cos θ) + (ρ * sin θ) + 2 * m = 0

noncomputable def range_of_m_for_intersection : set ℝ := 
  {m : ℝ | -19 / 12 ≤ m ∧ m ≤ 5 / 2}

theorem rect_eq_and_range_of_m (m : ℝ) : 
  rect_eq_of_polar m ∧ (∀ t : ℝ, ∃ x y : ℝ, x = sqrt 3 * cos (2 * t) ∧ y = 2 * sin t ∧ 
    (sqrt 3 * x + y + 2 * m = 0) → m ∈ range_of_m_for_intersection) :=
by sorry

end rect_eq_and_range_of_m_l640_640310


namespace total_cookies_dropped_throughout_entire_baking_process_l640_640436

def initially_baked_by_alice := 74 + 45 + 15
def initially_baked_by_bob := 7 + 32 + 18

def initially_dropped_by_alice := 5 + 8
def initially_dropped_by_bob := 10 + 6

def additional_baked_by_alice := 5 + 4 + 12
def additional_baked_by_bob := 22 + 36 + 14

def edible_cookies := 145

theorem total_cookies_dropped_throughout_entire_baking_process :
  initially_baked_by_alice + initially_baked_by_bob +
  additional_baked_by_alice + additional_baked_by_bob -
  edible_cookies = 139 := by
  sorry

end total_cookies_dropped_throughout_entire_baking_process_l640_640436


namespace water_bottle_shape_l640_640933

-- Conditions: Declaration of the height H, and the total volume V0 at full bottle height
variables (H V0 : ℝ)

-- Definition expressing the condition at h = H / 2
def volume_condition_at_half_height (V : ℝ → ℝ) : Prop :=
  V (H / 2) > V0 / 2

-- Proof goal: There exists a function (representing the volume function of a conical bottle)
-- satisfying the given conditions.
theorem water_bottle_shape (V : ℝ → ℝ) 
  (h_conical_shape : ∀ h, V h = V0 * (h / H) ^ 2) 
  (hV_condition : volume_condition_at_half_height V) :
  V (H / 2) > V0 / 2 :=
by {
  -- Open the definition of volume_condition_at_half_height
  unfold volume_condition_at_half_height,
  -- Apply the given function and the condition at half height
  exact hV_condition,
}

end water_bottle_shape_l640_640933


namespace linear_eq_solution_l640_640637

theorem linear_eq_solution (a : ℝ) (h : (a - 2) = 0) : ∃ x : ℝ, ax^2 + 5x + 14 = 2x^2 - 2x + 3a ∧ x = -(8 / 7) :=
by
  have ha : a = 2 := by linarith
  use -(8 / 7)
  split
  {
    calc
      ax^2 + 5x + 14
          = 2x^2 - 2x + 3a
          : by sorry
  }
  {
    exact rfl
  }

end linear_eq_solution_l640_640637


namespace cost_per_kg_additional_l640_640146

-- Define the variables
variables (l m : ℕ)

-- Conditions
def condition1 : Prop := 25 * l = 250
def condition2 : Prop := 30 * l + 3 * m = 360
def condition3 : Prop := 30 * l + 6 * m = 420

-- The theorem to prove
theorem cost_per_kg_additional (h1 : condition1) (h2 : condition2) (h3 : condition3) : m = 20 := 
sorry

end cost_per_kg_additional_l640_640146


namespace combined_plot_area_l640_640128

theorem combined_plot_area (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 972) (h3 : ∃ h : ℝ, h = 7) :
  (972 + (1 / 2 * b * 7) = 1035) :=
by
  have h4 : b = Real.sqrt (972 / 3) := by sorry
  have h5 : b = 18 := by sorry
  have h6 : 1 / 2 * b * 7 = 63 := by sorry
  have h7 : 972 + 63 = 1035 := by sorry
  exact h7

end combined_plot_area_l640_640128


namespace peopleFeelWaveType_leastImpactedSphere_earthquakeSourceLayer_l640_640576

-- Definitions for wave propagation conditions
def LongitudinalWavePropagatesThrough : Prop := 
  ∀ (s : Substance), PropagatesThrough LongitudinalWave s

def TransverseWavePropagatesThroughSolidsOnly : Prop := 
  PropagatesThrough TransverseWave Solid ∧ 
  ¬PropagatesThrough TransverseWave Liquid ∧ 
  ¬PropagatesThrough TransverseWave Gas

-- Proof goal for wave propagation
theorem peopleFeelWaveType (cond_longitudinal : LongitudinalWavePropagatesThrough) 
  (cond_transverse : TransverseWavePropagatesThroughSolidsOnly) : 
  PeopleFeel = Longitudinal := by
  sorry

-- Definitions for the least impacted sphere conditions
def EarthquakeImpactsLithosphereDirectly : Prop := 
  Affects Lithosphere Earthquake

def TsunamiImpactsHydrosphereBiosphere : Prop := 
  Affects Hydrosphere Tsunami ∧ Affects Biosphere Tsunami

-- Proof goal for least impacted sphere
theorem leastImpactedSphere (cond_earthquake : EarthquakeImpactsLithosphereDirectly) 
  (cond_tsunami : TsunamiImpactsHydrosphereBiosphere) : 
  LeastImpactedSphere = Atmosphere := by
  sorry

-- Definitions for earthquake source layer conditions
def EarthLayer (layer : string) : Prop := 
  ∃ d : Nat, EarthLayerAtDepth d = layer

def FocalDepth : Nat := 20

-- Proof goal for source layer of the earthquake
theorem earthquakeSourceLayer (layer_mantle : EarthLayer "mantle") 
  (cond_depth : FocalDepth = 20) : 
  SourceLayer = Mantle := by
  sorry

end peopleFeelWaveType_leastImpactedSphere_earthquakeSourceLayer_l640_640576


namespace original_numbers_l640_640036

theorem original_numbers (a b c d : ℕ) (x : ℕ)
  (h1 : a + b + c + d = 45)
  (h2 : a + 2 = x)
  (h3 : b - 2 = x)
  (h4 : 2 * c = x)
  (h5 : d / 2 = x) : 
  (a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20) :=
sorry

end original_numbers_l640_640036


namespace remainder_of_2n_div_10_l640_640094

theorem remainder_of_2n_div_10 (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end remainder_of_2n_div_10_l640_640094


namespace arithmetic_sequence_count_l640_640937

noncomputable def count_arithmetic_triplets : ℕ := 17

theorem arithmetic_sequence_count :
  ∃ S : Finset (Finset ℕ), 
    (∀ s ∈ S, s.card = 3 ∧ (∃ d, ∀ x ∈ s, ∀ y ∈ s, ∀ z ∈ s, (x ≠ y ∧ y ≠ z ∧ x ≠ z) → ((x = y + d ∨ x = z + d ∨ y = z + d) ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9))) ∧ 
    S.card = count_arithmetic_triplets :=
by
  -- placeholder for proof
  sorry

end arithmetic_sequence_count_l640_640937


namespace simplest_fractions_are_ACD_l640_640140

noncomputable def fraction_A := (x^2 - 1) / (x^2 + 1)
noncomputable def fraction_B := (x + 1) / (x^2 - 1)
noncomputable def fraction_C := (x^2 - 1) / x
noncomputable def fraction_D := (x - 1) / (x + 1)

def is_simplified (fraction : ℚ) : Prop := 
  -- Definition of simplified can vary; for simplicity, let's assume it checks coprime numerator and denominator.
  sorry

theorem simplest_fractions_are_ACD (x : ℚ) :
  is_simplified (fraction_A x) ∧ 
  ¬ is_simplified (fraction_B x) ∧ 
  is_simplified (fraction_C x) ∧ 
  is_simplified (fraction_D x) :=
sorry

end simplest_fractions_are_ACD_l640_640140


namespace tiles_needed_l640_640850

theorem tiles_needed (S : ℕ) (n : ℕ) (k : ℕ) (N : ℕ) (H1 : S = 18144) 
  (H2 : n * k^2 = S) (H3 : n = (N * (N + 1)) / 2) : n = 2016 := 
sorry

end tiles_needed_l640_640850


namespace average_weight_BC_l640_640468

-- Define the weights as variables
variables (A B C : ℝ)

-- Define the conditions
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (A + B) / 2 = 40
def condition3 : Prop := B = 31

-- The theorem to prove
theorem average_weight_BC (h1 : condition1) (h2 : condition2) (h3 : condition3) : (B + C) / 2 = 43 :=
sorry

end average_weight_BC_l640_640468


namespace people_born_in_country_l640_640743

-- Define the conditions
def people_immigrated : ℕ := 16320
def new_people_total : ℕ := 106491

-- Define the statement to be proven
theorem people_born_in_country (people_born : ℕ) (h : people_born = new_people_total - people_immigrated) : 
    people_born = 90171 :=
  by
    -- This is where we would provide the proof, but we use sorry to skip the proof.
    sorry

end people_born_in_country_l640_640743


namespace rectangular_equation_common_points_l640_640317

-- Define parametric equations of curve C
def x (t : ℝ) := (sqrt 3) * cos (2 * t)
def y (t : ℝ) := 2 * sin t

-- Define polar equation of line l
def polar_line (ρ θ m : ℝ) := ρ * sin (θ + π / 3) + m

-- Rectangular equation of l
theorem rectangular_equation (ρ θ x y m : ℝ) (h1 : ρ = sqrt (x^2 + y^2)) (h2 : θ = atan2 y x) :
  polar_line ρ θ m = 0 → (sqrt 3 * x + y + 2 * m = 0) :=
by 
  sorry

-- Range of values for m where line intersects curve
theorem common_points (h: ∃ t, sqrt 3 * cos (2 * t) = x ∧ 2 * sin t = y) :
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by 
  sorry

end rectangular_equation_common_points_l640_640317


namespace probability_penny_dime_heads_l640_640002

-- Define the probabilities for individual coin flips
def coin_flip_outcomes : ℕ := 2
def total_outcomes (n : ℕ) : ℕ := coin_flip_outcomes ^ n
def successful_outcomes : ℕ := coin_flip_outcomes ^ 3

-- Statement of the problem
theorem probability_penny_dime_heads : 
  (successful_outcomes : ℚ) / (total_outcomes 5 : ℚ) = 1 / 4 :=
by
  -- Proof omitted
  sorry

end probability_penny_dime_heads_l640_640002


namespace average_of_data_set_l640_640580

theorem average_of_data_set :
  let data_set := [9.8, 9.9, 10, 10.1, 10.2] in
  let n := data_set.length in
  let sum := data_set.foldl (· + ·) 0 in
  (sum / n) = 10 :=
by
  let data_set := [9.8, 9.9, 10, 10.1, 10.2]
  let n := data_set.length
  let sum := data_set.foldl (· + ·) 0
  have h : (sum / n) = 10 := sorry
  exact h

end average_of_data_set_l640_640580


namespace product_of_intersection_coordinates_l640_640521

noncomputable def circle1 := {P : ℝ×ℝ | (P.1^2 - 4*P.1 + P.2^2 - 8*P.2 + 20) = 0}
noncomputable def circle2 := {P : ℝ×ℝ | (P.1^2 - 6*P.1 + P.2^2 - 8*P.2 + 25) = 0}

theorem product_of_intersection_coordinates :
  ∀ P ∈ circle1 ∩ circle2, P = (2, 4) → (P.1 * P.2) = 8 :=
by
  sorry

end product_of_intersection_coordinates_l640_640521


namespace simple_interest_rate_l640_640149

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) 
  (hT : T = 10) (hSI : (P * R * T) / 100 = (1 / 5) * P) : R = 2 :=
by
  sorry

end simple_interest_rate_l640_640149


namespace sum_of_digits_square_nines_twos_l640_640080

def sum_of_digits (n : ℕ) : ℕ := 
  (222222222 : ℕ) ^ 2 D.sum

theorem sum_of_digits_square_nines_twos : 
  sum_of_digits (9) = 162 :=
sorry

end sum_of_digits_square_nines_twos_l640_640080


namespace translate_inverse_proportion_l640_640823

theorem translate_inverse_proportion (k : ℝ) (h : k ≠ 0) :
  ((-3) + 1 ≠ 0) →
  (∃ k, (k : ℝ) = -6 ↔ y = (k / (x + 1)) - 2 : ℝ) →
  y = 1 ∧ x = -3 :=
begin
  sorry
end

end translate_inverse_proportion_l640_640823


namespace prove_a_plus_b_eq_zero_l640_640634

noncomputable def imaginary_unit : ℂ := complex.I

def condition (a b : ℝ) : Prop := (a + complex.I) / complex.I = 1 + b * complex.I

theorem prove_a_plus_b_eq_zero (a b : ℝ) (h : condition a b) : a + b = 0 :=
begin
  sorry
end

end prove_a_plus_b_eq_zero_l640_640634


namespace valid_prime_expression_l640_640825

open Nat

def is_prime_and_within_bounds (p : ℕ) : Prop := prime p ∧ 4 < p ∧ p < 18

theorem valid_prime_expression : ∃ p q : ℕ, is_prime_and_within_bounds p ∧ is_prime_and_within_bounds q ∧ p ≠ q ∧ (p * q - (p + q) = 119) :=
by
  sorry

end valid_prime_expression_l640_640825


namespace sum_values_q_l640_640787

noncomputable def q : ℤ → ℤ := sorry

theorem sum_values_q :
  q(1) = 5 ∧ q(6) = 20 ∧ q(14) = 12 ∧ q(19) = 30 →
  (∑ x in finset.range 21, q x) = 357 :=
sorry

end sum_values_q_l640_640787


namespace leak_empties_tank_in_10_hours_l640_640855

theorem leak_empties_tank_in_10_hours :
  let rate_pipeA := 1 / 2 -- Pipe A's rate without the leak
  let rate_combined := 1 / 2.5 -- Combined rate of Pipe A and the leak
  let rate_leak := rate_pipeA - rate_combined -- Leak's rate
  rate_leak ≠ 0 → -- Ensuring leak's rate is non-zero to avoid division by zero
  1 / rate_leak = 10 :=
by 
  let rate_pipeA := 1 / 2
  let rate_combined := 1 / 2.5
  let rate_leak := rate_pipeA - rate_combined
  have h : rate_leak = 1 / 10 := by
    rw [div_eq_mul_inv, div_eq_mul_inv, inv_div, inv_div, inv_eq_one_div]
    norm_num
  have h' : 1 / rate_leak = 10 := by
    rw [h, one_div, div_mul_cancel]
    norm_num
  
  exact h'

end leak_empties_tank_in_10_hours_l640_640855


namespace unfriendly_subsets_card_l640_640913

def is_unfriendly (s : Finset ℕ) : Prop := 
  (∀ (x y ∈ s), abs (x - y) ≠ 1)

def unfriendly_subsets_count (n k : ℕ) : ℕ :=
  Finset.card {s : Finset ℕ | s ⊆ Finset.range(n+1) ∧ s.card = k ∧ is_unfriendly s}

theorem unfriendly_subsets_card (n k : ℕ) :
  unfriendly_subsets_count n k = Nat.choose (n - k + 1) k :=
by
  sorry

end unfriendly_subsets_card_l640_640913


namespace total_goals_by_other_players_l640_640889

theorem total_goals_by_other_players (total_players goals_season games_played : ℕ)
  (third_players_goals avg_goals_per_third : ℕ)
  (h1 : total_players = 24)
  (h2 : goals_season = 150)
  (h3 : games_played = 15)
  (h4 : third_players_goals = total_players / 3)
  (h5 : avg_goals_per_third = 1)
  : (goals_season - (third_players_goals * avg_goals_per_third * games_played)) = 30 :=
by
  sorry

end total_goals_by_other_players_l640_640889


namespace rectangular_eq_of_line_l_range_of_m_l640_640343

noncomputable def curve_C_param_eq (t : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l_polar_eq (θ ρ m : ℝ) : Prop :=
  ρ * Real.sin(θ + Real.pi / 3) + m = 0

theorem rectangular_eq_of_line_l (x y m : ℝ) (h : line_l_polar_eq (Real.atan2 y x) (Real.sqrt (x^2 + y^2)) m) :
  sqrt 3 * x + y + 2 * m = 0 :=
  sorry

theorem range_of_m (m : ℝ) (t : ℝ) (h : ∃ (t : ℝ), curve_C_param_eq t ∈ set_of (λ p, p.2 = -√3 * p.1 - 2 * m))
  : -19/12 ≤ m ∧ m ≤ 5/2 :=
  sorry

end rectangular_eq_of_line_l_range_of_m_l640_640343


namespace min_distance_between_vertices_l640_640827

/- Define a structure for a triangle in the Euclidean space -/
structure Triangle (α : Type*) :=
(A B C : α)

variables {α : Type*} [add_comm_group α] [vector_space ℝ α] [finite_dimensional ℝ α]

/- Define the condition for side length -/
def min_side_length (T : Triangle α) (a : ℝ) : Prop :=
(dist T.A T.B ≥ a) ∧ (dist T.B T.C ≥ a) ∧ (dist T.C T.A ≥ a)

/- Define the main theorem statement -/
theorem min_distance_between_vertices
  (T1 T2 : Triangle α)
  (a a' : ℝ)
  (hT1 : min_side_length T1 a)
  (hT2 : min_side_length T2 a') :
  ∃ (p ∈ {T1.A, T1.B, T1.C}) (p' ∈ {T2.A, T2.B, T2.C}), dist p p' ≥ sqrt ((a^2 + a'^2) / 3) :=
sorry

end min_distance_between_vertices_l640_640827


namespace paint_faces_valid_pairs_l640_640587

def isValidPair (x y : ℕ) : Prop := 
  (x + y ≠ 9) ∧ (|x - y| ≠ 1) ∧ (x ≠ y)

def validPairs : List (ℕ × ℕ) :=
  [(1, 4), (1, 6), (2, 5), (4, 6), (5, 7), (6, 8)]

theorem paint_faces_valid_pairs : 
  (list.filter (λ pair : ℕ × ℕ => 
    let (x, y) := pair 
    x ∈ [1, 2, 3, 4, 5, 6, 7, 8] ∧ 
    y ∈ [1, 2, 3, 4, 5, 6, 7, 8] ∧ 
    isValidPair x y) validPairs).length = 6 := 
by
  sorry

end paint_faces_valid_pairs_l640_640587


namespace sqrt_difference_inequality_l640_640828

theorem sqrt_difference_inequality (x : ℝ) (h : x ≥ 5) : 
  sqrt (x - 2) - sqrt (x - 3) < sqrt (x - 4) - sqrt (x - 5) :=
sorry

end sqrt_difference_inequality_l640_640828


namespace sum_of_positive_differences_l640_640396

noncomputable def S : Finset ℕ := (Finset.range 11).image (fun x => 3^x)

def N : ℕ := ∑ i in S.attach, ∑ j in S.attach, if (i.val > j.val) then (i.val - j.val) else 0

theorem sum_of_positive_differences :
  N = 779409 :=
sorry

end sum_of_positive_differences_l640_640396


namespace sticker_probability_l640_640062

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end sticker_probability_l640_640062


namespace sample_size_drawn_l640_640954

theorem sample_size_drawn (sample_size : ℕ) (probability : ℚ) (N : ℚ) 
  (h1 : sample_size = 30) 
  (h2 : probability = 0.25) 
  (h3 : probability = sample_size / N) : 
  N = 120 := by
  sorry

end sample_size_drawn_l640_640954


namespace p_nonnegative_iff_equal_l640_640440

def p (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem p_nonnegative_iff_equal (a b c : ℝ) : (∀ x : ℝ, p a b c x ≥ 0) ↔ a = b ∧ b = c :=
by
  sorry

end p_nonnegative_iff_equal_l640_640440


namespace cookies_with_three_cups_l640_640389

theorem cookies_with_three_cups (cookies_per_two_cups : ℕ) (h : cookies_per_two_cups = 18) : 
  ∃ x : ℕ, x = 27 :=
by
  use 27
  -- Here we would provide the proof that based on the conditions, x = 27.
  sorry

end cookies_with_three_cups_l640_640389


namespace imaginary_part_of_z2_l640_640226

def z1 : ℂ := 1 - 2 * complex.I

def z2 : ℂ := (z1 + 1) / (z1 - 1)

theorem imaginary_part_of_z2 :
  z2.im = 1 :=
by
  sorry

end imaginary_part_of_z2_l640_640226


namespace find_N_l640_640706

-- Definitions and conditions directly appearing in the problem
variable (X Y Z N : ℝ)

axiom condition1 : 0.15 * X = 0.25 * N + Y
axiom condition2 : X + Y = Z

-- The theorem to prove
theorem find_N : N = 4.6 * X - 4 * Z := by
  sorry

end find_N_l640_640706


namespace final_number_less_than_one_l640_640433

theorem final_number_less_than_one :
  let S := { n | 1000 ≤ n ∧ n ≤ 2999 } in
  ∀ (initial_set : set ℕ) (hS : initial_set = S) (erase_two_write_one : set ℕ → set ℕ),
  (∀ a b, a ∈ initial_set → b ∈ initial_set → a ≤ b → erase_two_write_one initial_set = (initial_set \ {a, b}) ∪ {a / 2}) →
  ∃ remaining_number, erase_two_write_one^[1999] initial_set = {remaining_number} ∧ remaining_number < 1 :=
begin
  sorry
end

end final_number_less_than_one_l640_640433


namespace product_units_digit_mod_10_l640_640522

theorem product_units_digit_mod_10 (a b c d : ℕ) (ha : a = 1723) (hb : b = 5497) (hc : c = 80605) (hd : d = 93) :
  ((a * b * c * d) % 10) = 5 :=
by
  rw [ha, hb, hc, hd]
  -- Now evaluate the expression manually with the given numbers to reach the final statement.
  have h1 : 1723 % 10 = 3 := by norm_num,
  have h2 : 5497 % 10 = 7 := by norm_num,
  have h3 : 80605 % 10 = 5 := by norm_num,
  have h4 : 93 % 10 = 3 := by norm_num,
  rw [h1, h2, h3, h4],
  norm_num [mul_mod, add_mod, mul_add_mod, add_mul_mod]

-- The statement has been formulated; proof steps (tactics) include norm_num for numeric reductions and rewrites for substituting modular operations.

end product_units_digit_mod_10_l640_640522


namespace probability_sum_odd_l640_640497

theorem probability_sum_odd {d1 d2 d3 : ℕ}
  (h1 : d1 ∈ {1, 2, 3, 4, 5, 6})
  (h2 : d2 ∈ {1, 2, 3, 4, 5, 6})
  (h3 : d3 ∈ {1, 2, 3, 4, 5, 6}) :
  (∃ (p : ℚ), p = 1 / 2 ∧ 
  (∃ (odd_count : ℕ), odd_count = (if d1 % 2 = 1 then 1 else 0) + (if d2 % 2 = 1 then 1 else 0) + (if d3 % 2 = 1 then 1 else 0) ∧ 
  (odd_count % 2 = 1))) :=
by 
  sorry

end probability_sum_odd_l640_640497


namespace can_cut_rectangle_with_area_300_cannot_cut_rectangle_with_ratio_3_2_l640_640418

-- Question and conditions
def side_length_of_square (A : ℝ) := A = 400
def area_of_rect (A : ℝ) := A = 300
def ratio_of_rect (length width : ℝ) := 3 * width = 2 * length

-- Prove that Li can cut a rectangle with area 300 from the square with area 400
theorem can_cut_rectangle_with_area_300 
  (a : ℝ) (h1 : side_length_of_square a)
  (length width : ℝ)
  (ha : a ^ 2 = 400) (har : length * width = 300) :
  length ≤ a ∧ width ≤ a :=
by
  sorry

-- Prove that Li cannot cut a rectangle with ratio 3:2 from the square
theorem cannot_cut_rectangle_with_ratio_3_2 (a : ℝ)
  (h1 : side_length_of_square a)
  (length width : ℝ)
  (har : area_of_rect (length * width))
  (hratio : ratio_of_rect length width)
  (ha : a ^ 2 = 400) :
  ¬(length ≤ a ∧ width ≤ a) :=
by
  sorry

end can_cut_rectangle_with_area_300_cannot_cut_rectangle_with_ratio_3_2_l640_640418


namespace max_red_points_critical_coloring_l640_640549

def S : Set (ℕ × ℕ) := { (i, j) | i ∈ {0, 1, ..., 99} ∧ j ∈ {0, 1, ..., 99} }

def is_critical_coloring (red_points : Set (ℕ × ℕ)) : Prop :=
  ∀ i j ∈ {0, 1, ..., 99}, 
    (⟨i, j⟩ ∈ red_points ∨ 
    ⟨(i + 1) % 100, j⟩ ∈ red_points ∨ 
    ⟨i, (j + 1) % 100⟩ ∈ red_points ∨ 
    ⟨(i + 1) % 100, (j + 1) % 100⟩ ∈ red_points)

theorem max_red_points_critical_coloring :
  ∃ red_points : Set (ℕ × ℕ), 
  is_critical_coloring red_points ∧ red_points.card = 5000 :=
sorry

end max_red_points_critical_coloring_l640_640549


namespace solve_system_l640_640413

variable {R : Type*} [CommRing R] {a b c x y z : R}

theorem solve_system (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h₁ : z + a*y + a^2*x + a^3 = 0) 
  (h₂ : z + b*y + b^2*x + b^3 = 0) 
  (h₃ : z + c*y + c^2*x + c^3 = 0) :
  x = -(a + b + c) ∧ y = (a * b + a * c + b * c) ∧ z = -(a * b * c) := 
sorry

end solve_system_l640_640413


namespace probability_complete_collection_l640_640044

def combination (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem probability_complete_collection : 
  combination 6 6 * combination 12 4 / combination 18 10 = 5 / 442 :=
by
  have h1 : combination 6 6 = 1 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h2 : combination 12 4 = 495 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h3 : combination 18 10 = 43758 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  rw [h1, h2, h3]
  norm_num
  sorry

end probability_complete_collection_l640_640044


namespace frank_final_steps_l640_640630

def final_position : ℤ :=
  let initial_back := -5
  let first_forward := 10
  let second_back := -2
  let second_forward := 2 * 2
  initial_back + first_forward + second_back + second_forward

theorem frank_final_steps : final_position = 7 := by
  simp
  sorry

end frank_final_steps_l640_640630


namespace second_player_cannot_prevent_l640_640065

-- Define the structure of the game
inductive Dot : Type
| Red : Dot
| Blue : Dot

structure Move :=
(dot_type : Dot)
(position : ℝ × ℝ)  -- Assuming a 2D plane

-- Define the condition of forming an equilateral triangle
def is_equilateral_triangle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let d := λ (a b : ℝ × ℝ), real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) in
  d p1 p2 = d p2 p3 ∧ d p2 p3 = d p3 p1

-- Define the winning condition
def first_player_wins (moves : list Move) : Prop :=
  ∃ (p1 p2 p3 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    (∃ m1 m2 m3, 
      m1.dot_type = Dot.Red ∧ m2.dot_type = Dot.Red ∧ m3.dot_type = Dot.Red ∧ 
      m1.position = p1 ∧ m2.position = p2 ∧ m3.position = p3 ∧
      is_equilateral_triangle p1 p2 p3)

-- Define the main theorem
theorem second_player_cannot_prevent :
  ∀ (moves : list Move),
  (∀ n, n ≤ 12 → (∃ b : ℕ, b = 10 ∧ moves.length = (12 * b))) →
  first_player_wins moves := sorry

end second_player_cannot_prevent_l640_640065


namespace fixed_point_for_all_a_find_a_for_tangent_find_a_for_tangent_neg_l640_640691

-- Question 1
def passes_through_fixed_point (a : ℝ) : Prop :=
  let c : ℝ := 4
  let d : ℝ := -2
  (c ^ 2) + (d ^ 2) - 4 * a * c + 2 * a * d + 20 * a - 20 = 0

theorem fixed_point_for_all_a (a : ℝ) : (passes_through_fixed_point a) :=
  sorry

-- Question 2
def tangent_to_circle (a : ℝ) : Prop :=
  let r := 2
  let (cx, cy) := (0 : ℝ, 0 : ℝ)
  ∀ (x y : ℝ), (x ^ 2) + (y ^ 2) - 4 * a * x + 2 * a * y + 20 * a - 20 = 0 → 
    let distance := (x - cx)^2 + (y - cy)^2 in
    abs (distance - r^2) < 1e-6

theorem find_a_for_tangent : { a : ℝ // tangent_to_circle a } :=
  ⟨ 1 + (real.sqrt 5 / 5), sorry ⟩

theorem find_a_for_tangent_neg : { a : ℝ // tangent_to_circle a } :=
  ⟨ 1 - (real.sqrt 5 / 5), sorry ⟩

end fixed_point_for_all_a_find_a_for_tangent_find_a_for_tangent_neg_l640_640691


namespace sum_of_angles_l640_640007

theorem sum_of_angles (O : Type) (circle : Circle O) (arcs : Fin 16 → SubCircle circle) 
  (x_arc_count : Fin 16 := 3) (y_arc_count : Fin 16 := 5) :
  let arc_angle := (360 : ℝ) / 16
  let x_central_angle := x_arc_count.val * arc_angle
  let y_central_angle := y_arc_count.val * arc_angle
  let x := x_central_angle / 2
  let y := y_central_angle / 2
  (x + y = 90 : ℝ) :=
by
  sorry

end sum_of_angles_l640_640007


namespace arithmetic_sequence_general_formula_l640_640945

noncomputable theory

variables {a_1 d n : ℤ} 

def arithmetic_sequence (a_1 d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d

def sum_first_n_terms (a_1 d n : ℤ) : ℤ := n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_general_formula
  (a_1 d : ℤ) 
  (h1 : a_1 ≠ 0)
  (h2 : a_1 ^ 2 + (a_1 + 6 * d) ^ 2 = (a_1 + 2 * d) ^ 2 + (a_1 + 8 * d) ^ 2)
  (h3 : sum_first_n_terms a_1 d 8 = 8) :
  arithmetic_sequence a_1 d n = 10 - 2 * n :=
begin
  sorry
end

end arithmetic_sequence_general_formula_l640_640945


namespace dance_steps_equiv_l640_640626

theorem dance_steps_equiv
  (back1 : ℕ)
  (forth1 : ℕ)
  (back2 : ℕ)
  (forth2 : ℕ)
  (back3 : ℕ := 2 * back2) : 
  back1 = 5 ∧ forth1 = 10 ∧ back2 = 2 → 
  (0 - back1 + forth1 - back2 + forth2 = 7) :=
by
  intros h
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw [h1, h3, h4],
  sorry

end dance_steps_equiv_l640_640626


namespace median_of_first_twelve_positive_integers_l640_640512

theorem median_of_first_twelve_positive_integers : 
  let first_twelve_positive_integers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (first_twelve_positive_integers.length = 12) →
  (first_twelve_positive_integers.nth (first_twelve_positive_integers.length / 2 - 1) = some 6) →
  (first_twelve_positive_integers.nth (first_twelve_positive_integers.length / 2) = some 7) →
  (6 + 7) / 2 = 6.5 :=
by
  sorry

end median_of_first_twelve_positive_integers_l640_640512


namespace pyramid_volume_calculation_l640_640554

noncomputable def square_base_area := 256
noncomputable def triangle_jlp_area := 128
noncomputable def triangle_mnp_area := 112

theorem pyramid_volume_calculation
  (base_area : ℝ := square_base_area)
  (jlp_area : ℝ := triangle_jlp_area)
  (mnp_area: ℝ := triangle_mnp_area)
  (side_length : ℝ := real.sqrt base_area)
  (height_jlp : ℝ := 2 * jlp_area / side_length)
  (height_mnp : ℝ := 2 * mnp_area / side_length)
  (b : ℝ := (side_length^2 - 42) / (2 * side_length))
  (height: ℝ := real.sqrt (side_length^2 - b^2))
  (volume : ℝ := (1 / 3 : ℝ) * base_area * height) :
  volume ≈ 1237.28 :=
sorry

end pyramid_volume_calculation_l640_640554


namespace parabola_focus_distance_l640_640988

noncomputable def focus_of_parabola (a : ℝ) : ℝ × ℝ := (a, 0)

noncomputable def point_on_parabola (x y : ℝ) (a : ℝ) : Prop := y^2 = 4 * a * x

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem parabola_focus_distance :
  ∀ m : ℝ, 
  point_on_parabola (-4) m (-3) →
  distance (-4, m) (-3, 0) = 7 :=
by
  intro m hp,
  sorry

end parabola_focus_distance_l640_640988


namespace expression_evaluation_l640_640526

theorem expression_evaluation : (50 + 12) ^ 2 - (12 ^ 2 + 50 ^ 2) = 1200 := 
by
  sorry

end expression_evaluation_l640_640526


namespace find_x_l640_640696

theorem find_x (x : ℝ) : 
  let a := (4, 2)
  let b := (x, 3)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -3 / 2 :=
by
  intros a b h
  sorry

end find_x_l640_640696


namespace arithmetic_proof_l640_640101

theorem arithmetic_proof :
  (3652 * 2487) + (979 - 45 * 13) = 9085008 :=
by {
  have h1 : 3652 * 2487 = 9084614 := by norm_num,
  have h2 : 979 - 45 * 13 = 394 := by norm_num,
  rw [h1, h2],
  norm_num,
  }

end arithmetic_proof_l640_640101


namespace solve_y_equation_l640_640455

theorem solve_y_equation :
  ∃ y : ℚ, 4 * (5 * y + 3) - 3 = -3 * (2 - 8 * y) ∧ y = 15 / 4 :=
by
  sorry

end solve_y_equation_l640_640455


namespace max_leap_years_l640_640575

theorem max_leap_years (years : ℕ) (leap_interval : ℕ) (total_years : ℕ) (leap_years : ℕ)
  (h1 : leap_interval = 5)
  (h2 : total_years = 200)
  (h3 : years = total_years / leap_interval) :
  leap_years = 40 :=
by
  sorry

end max_leap_years_l640_640575


namespace rectangular_equation_common_points_l640_640321

-- Define parametric equations of curve C
def x (t : ℝ) := (sqrt 3) * cos (2 * t)
def y (t : ℝ) := 2 * sin t

-- Define polar equation of line l
def polar_line (ρ θ m : ℝ) := ρ * sin (θ + π / 3) + m

-- Rectangular equation of l
theorem rectangular_equation (ρ θ x y m : ℝ) (h1 : ρ = sqrt (x^2 + y^2)) (h2 : θ = atan2 y x) :
  polar_line ρ θ m = 0 → (sqrt 3 * x + y + 2 * m = 0) :=
by 
  sorry

-- Range of values for m where line intersects curve
theorem common_points (h: ∃ t, sqrt 3 * cos (2 * t) = x ∧ 2 * sin t = y) :
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by 
  sorry

end rectangular_equation_common_points_l640_640321


namespace problem1_simplification_problem2_expression_l640_640104

-- Problem 1
theorem problem1_simplification: 
  (Real.sqrt (6 + 1/4) + 38^2 + (0.027^(-2/3)) * ((-1/3)^(-2))) = 24746.5 :=
sorry

-- Problem 2
theorem problem2_expression (a : ℝ) (h : a^(1/2) + a^(-1/2) = 3) :
  a^2 + a^(-2) = 47 :=
sorry

end problem1_simplification_problem2_expression_l640_640104


namespace simplest_quadratic_radical_l640_640530

theorem simplest_quadratic_radical :
  ∀ (sqrt6 sqrt12 sqrt1over3 sqrt03 : ℝ),
  sqrt6 = Real.sqrt 6 →
  sqrt12 = Real.sqrt 12 →
  sqrt1over3 = Real.sqrt (1 / 3) →
  sqrt03 = Real.sqrt 0.3 →
  sqrt6 = Real.sqrt 6 ∧ 
  (√12 = 2 * √3) ∧ 
  (√(1/3) = √3 / 3) ∧ 
  (√0.3 = √30 / 10) →
  sqrt6 is the simplest among sqrt6, sqrt12, sqrt1over3, sqrt03 := sorry

end simplest_quadratic_radical_l640_640530


namespace product_ineq_l640_640750

-- Define the relevant elements and conditions
variables (a b : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ)

-- Assumptions based on the conditions provided
variables (h₀ : a > 0) (h₁ : b > 0)
variables (h₂ : a + b = 1)
variables (h₃ : x₁ > 0) (h₄ : x₂ > 0) (h₅ : x₃ > 0) (h₆ : x₄ > 0) (h₇ : x₅ > 0)
variables (h₈ : x₁ * x₂ * x₃ * x₄ * x₅ = 1)

-- The theorem statement to be proved
theorem product_ineq : (a * x₁ + b) * (a * x₂ + b) * (a * x₃ + b) * (a * x₄ + b) * (a * x₅ + b) ≥ 1 :=
sorry

end product_ineq_l640_640750


namespace probability_exactly_r_white_balls_sum_binomial_coeff_eq_l640_640532

-- Assume combinatorial functions (binomial coefficients) are defined
noncomputable def comb (n k : ℕ) : ℕ := sorry

variables (n m k r : ℕ)

-- Statement for part (a)
theorem probability_exactly_r_white_balls :
  (comb n r * comb m (k - r)) / comb (n + m) k = 
  (comb n r * comb m (k - r)) / comb (n + m) k :=
sorry

-- Statement for part (b)
theorem sum_binomial_coeff_eq :
  ∑ i in finset.range (k + 1), comb n i * comb m (k - i) = comb (n + m) k :=
sorry

end probability_exactly_r_white_balls_sum_binomial_coeff_eq_l640_640532


namespace find_a_and_period_find_max_m_l640_640234

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * Real.sin x^2 + a

theorem find_a_and_period (a : ℝ) :
  let f (x : ℝ) := 2 * Real.sin (2 * x - π / 6) + 1 + a,
  (f (π / 3) = 3) → a = 0 ∧ ∀ x, f x = 2 * Real.sin (2 * x - π / 6) + 1 :=
by
  intros
  sorry

theorem find_max_m (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 2 * Real.sin (2 * x - π / 6) + 1 + a) →
  f (π/3) = 3 →
  (∀ x ∈ Icc 0 (2 * π / 3), f x ≥ 0) → 
  ∃ m : ℝ, m = 2 * π / 3 :=
by
  intros
  sorry

end find_a_and_period_find_max_m_l640_640234


namespace midpoint_of_XY_is_foot_of_altitude_to_AB_l640_640574

open EuclideanGeometry

variables {A B C D E X Y : Point}
variables (ABC : Triangle A B C) (h_acute : ∀ {α}, α ∈ angles ABC → α < π / 2)
variables (hD : D ∈ lineSegment A C) (hE : E ∈ lineSegment B C)
variables (circle1 : Circle A B D E) (circle2 : Circle D E C)
variables (hXY : ∃ X Y, X ∈ (circle2 ∩ lineSegment A B) ∧ Y ∈ (circle2 ∩ lineSegment A B))

theorem midpoint_of_XY_is_foot_of_altitude_to_AB :
  let M := midpoint X Y,
  let H := foot A B C,
  M = H :=
sorry

end midpoint_of_XY_is_foot_of_altitude_to_AB_l640_640574


namespace binomial_thm_approx_l640_640829

noncomputable def binomial_approx : ℝ :=
  (10 - 0.02)^5

theorem binomial_thm_approx :
  ∃ n : ℤ, n = 99004 ∧ abs (binomial_approx - n.to_real) < 1 :=
by
  use 99004
  sorry

end binomial_thm_approx_l640_640829


namespace find_divisor_nearest_to_3105_l640_640186

def nearest_divisible_number (n : ℕ) (d : ℕ) : ℕ :=
  if n % d = 0 then n else n + d - (n % d)

theorem find_divisor_nearest_to_3105 (d : ℕ) (h : nearest_divisible_number 3105 d = 3108) : d = 3 :=
by
  sorry

end find_divisor_nearest_to_3105_l640_640186


namespace A_intersection_complement_B_eq_l640_640991

def A (x : ℝ) : Prop := ∃ y : ℝ, y = sqrt(6 / (x + 1) - 1)
def B (x : ℝ) : Prop := ∃ y : ℝ, y = log (-x^2 + 2*x + 3)
def R_complement_B (x : ℝ) : Prop := x ≥ 3 ∨ x ≤ -1
def intersection_A_complement_B (x : ℝ) : Prop := A x ∧ R_complement_B x

theorem A_intersection_complement_B_eq (x : ℝ) : 
  intersection_A_complement_B x = (3 ≤ x ∧ x ≤ 5) :=
sorry

end A_intersection_complement_B_eq_l640_640991


namespace sum_eighth_row_l640_640733

-- Definitions based on the conditions
def sum_of_interior_numbers (n : ℕ) : ℕ := 2^(n-1) - 2

axiom sum_fifth_row : sum_of_interior_numbers 5 = 14
axiom sum_sixth_row : sum_of_interior_numbers 6 = 30

-- The proof problem statement
theorem sum_eighth_row : sum_of_interior_numbers 8 = 126 :=
by {
  sorry
}

end sum_eighth_row_l640_640733


namespace count_valid_n_l640_640922

-- Define that n is a positive integer satisfying both conditions
def valid_n (n : ℕ) : Prop :=
  (65 ^ 40 * n ^ 40 > n ^ 80) ∧ (n ^ 80 > 4 ^ 160)

-- Count the number of integers satisfying the given conditions
theorem count_valid_n :
  (finset.Icc 17 64).card = 48 :=
by sorry

end count_valid_n_l640_640922


namespace weight_of_new_student_l640_640095

theorem weight_of_new_student (avg_weight_29 : ℝ) (total_students_29 : ℕ)
                             (new_avg_weight_30 : ℝ) (total_students_30 : ℕ)
                             (w29 : avg_weight_29 = 28) 
                             (n29 : total_students_29 = 29) 
                             (w30 : new_avg_weight_30 = 27.4) 
                             (n30 : total_students_30 = 30) 
                             (new_total_weight_30 : (total_students_30 : ℝ) * new_avg_weight_30)
                             (old_total_weight_29 : (total_students_29 : ℝ) * avg_weight_29)
                             (weight_new_student: new_total_weight_30 - old_total_weight_29 = 10) : 
                             ∃ s, weight_new_student = 10 :=
by 
  sorry

end weight_of_new_student_l640_640095


namespace total_cans_collected_l640_640016

theorem total_cans_collected (total_students : ℕ) (half_students_collecting_12 : ℕ) 
 (students_collecting_0 : ℕ) (remaining_students_collecting_4 : ℕ) 
 (cans_collected_by_half : ℕ) (cans_collected_by_remaining : ℕ) :
 total_students = 30 →
 half_students_collecting_12 = 15 →
 students_collecting_0 = 2 →
 remaining_students_collecting_4 = 13 →
 cans_collected_by_half = half_students_collecting_12 * 12 →
 cans_collected_by_remaining = remaining_students_collecting_4 * 4 →
 (cans_collected_by_half + students_collecting_0 * 0 + cans_collected_by_remaining) = 232 :=
 by
  intros h1 h2 h3 h4 h5 h6
  obtain rfl : 15 * 12 = 180 := rfl
  obtain rfl : 13 * 4 = 52 := rfl
  rw [h5, h6]
  simp
  sorry

end total_cans_collected_l640_640016


namespace sum_minimized_n_eq_7_or_8_l640_640370

noncomputable def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem sum_minimized_n_eq_7_or_8 :
  ∃ n : ℕ, (n = 7 ∨ n = 8) →
  let S_n := Σ i in finset.range n, arithmetic_sequence (-28) 4 (i + 1) in
  is_minimum (λ n, Σ i in finset.range n, arithmetic_sequence (-28) 4 (i + 1)) S_n :=
sorry

end sum_minimized_n_eq_7_or_8_l640_640370


namespace matrix_reflection_l640_640927

theorem matrix_reflection (a b : ℝ) 
  (hR : (Matrix.vecCons (Matrix.vecCons a b Matrix.vecEmpty) 
                         (Matrix.vecCons (-1 / 2 : ℝ) (sqrt 3 / 2 : ℝ) Matrix.vecEmpty)) ^ 2
        = (Matrix.vecCons (Matrix.vecCons 1 0 Matrix.vecEmpty) 
                          (Matrix.vecCons 0 1 Matrix.vecEmpty))) :
  a = -sqrt(3) / 2 ∧ b = -1 / 2 := 
sorry

end matrix_reflection_l640_640927


namespace rooms_with_people_after_one_hour_l640_640813

theorem rooms_with_people_after_one_hour :
  ∃ nums : list ℕ, length nums = 1000 ∧ nums.head = 940 ∧ (forall i : ℕ, 1 ≤ i ∧ i ≤ 31 → nums.nth i = some 2) ∧ (forall i : ℕ, 32 ≤ i → nums.nth i = some 0) := sorry

end rooms_with_people_after_one_hour_l640_640813


namespace cesaro_sum_100_term_seq_l640_640195

def cesaro_sum (B : List ℝ) (n : ℝ) : ℝ := 
  (List.sum (List.scanl (+) 0 B.tail)) / n

theorem cesaro_sum_100_term_seq (B : List ℝ) (hB : List.length B = 99) 
(hCesaroSum : cesaro_sum B 99 = 800) : 
  cesaro_sum (10 :: B) 100 = 802 := sorry

end cesaro_sum_100_term_seq_l640_640195


namespace flies_eaten_per_day_l640_640171

variable (flies_per_frog per_day frogs_per_fish per_day fish_per_gharial per_day gharials: ℕ)

-- Each frog needs to eat 30 flies per day to live.
def fliesPerFrog: ℕ := 30

-- Each fish needs to eat 8 frogs per day to live.
def frogsPerFish: ℕ := 8

-- Each gharial needs to eat 15 fish per day to live.
def fishPerGharial: ℕ := 15

-- The swamp has 9 gharials.
def gharials: ℕ := 9

theorem flies_eaten_per_day 
  (fliesPerFrog: ℕ) (frogsPerFish: ℕ) (fishPerGharial: ℕ) (gharials: ℕ)
  (h1: fliesPerFrog = 30)
  (h2: frogsPerFish = 8)
  (h3: fishPerGharial = 15)
  (h4: gharials = 9)
  : 9 * (15 * 8 * 30) = 32400 := by
  sorry

end flies_eaten_per_day_l640_640171


namespace sum_of_integer_solutions_l640_640779

theorem sum_of_integer_solutions : 
  (∑ i in finset.filter (λ x : ℤ, -(5 : ℤ) / 2 < (x : ℝ) ∧ (x : ℝ) ≤ 5) (finset.Icc (-100) 100), i) = 12 :=
by
  sorry

end sum_of_integer_solutions_l640_640779


namespace perpendicular_collinear_condition_l640_640903

open_locale classical

variables {P : Type*} [metric_space P]

/-- Definitions for required elements -/
variables {O O1 O2 M N S T : P}

/-- Given conditions -/
variables (r : ℝ) 
variables (intersection_condition : M = N)
variables (tangent_to_O1_at_S : dist O S = r) 
variables (tangent_to_O2_at_T : dist O T = r)
variables (collinear_O_O1_S : collinear P ({O, O1, S} : set P))
variables (collinear_O_O2_T : collinear P ({O, O2, T} : set P))


/-- To prove: Necessary and sufficient condition for OM ⊥ MN is that S, N, T are collinear -/
theorem perpendicular_collinear_condition :
  (orthogonal (line_through O M) (line_through M N)) ↔ 
  collinear P ({S, N, T} : set P) :=
sorry

end perpendicular_collinear_condition_l640_640903


namespace number_of_points_on_circle_at_distance_2_from_line_l640_640209

-- Definitions from the given problem
def circle (x y : ℝ) : Prop := x^2 + y^2 = 16
def line (x y : ℝ) : Prop := x + y = 2
def distance (x y : ℝ) : ℝ := abs ((x + y - 2) / (real.sqrt (1^2 + 1^2)))

-- Goal is to prove the number of points on the circle at a distance of 2 from the line
theorem number_of_points_on_circle_at_distance_2_from_line :
  ∃ (points : ℕ), points = 4 ∧ (∀ (x y : ℝ), circle x y → distance x y = 2 → true) :=
sorry

end number_of_points_on_circle_at_distance_2_from_line_l640_640209


namespace bracelet_ways_l640_640722

-- Definition of the factorial function
def factorial (n : ℕ) : ℕ := if h : n = 0 then 1 else n * factorial (n - 1)

-- Noncomputable definition to avoid excess computation in proofs
noncomputable def bracelet_arrangements (n : ℕ) := factorial n / n

-- Main theorem to be proven
theorem bracelet_ways : bracelet_arrangements 8 = 5040 :=
by
  unfold bracelet_arrangements
  unfold factorial
  -- Calculation of 8!
  have fact8 : factorial 8 = 40320 := by norm_num [factorial]
  -- Calculation of 8 rotations
  have rotations8 : 40320 / 8 = 5040 := by norm_num
  -- Substituting the calculations in the main expression
  rw [fact8, rotations8]
  sorry

end bracelet_ways_l640_640722


namespace systematic_sampling_l640_640816

theorem systematic_sampling : ∃ (S : Set ℕ), S = {3, 13, 23, 33, 43, 53} ∧
  set.subset S (set.range (λ i, i + 1)) ∧
  set.card S = 6 ∧
  (∀ a b ∈ S, a ≠ b → ∃ k : ℕ, b = a + k * 10)
:=
sorry

end systematic_sampling_l640_640816


namespace sum_factors_of_18_l640_640525

theorem sum_factors_of_18 : (1 + 18 + 2 + 9 + 3 + 6) = 39 := by
  sorry

end sum_factors_of_18_l640_640525


namespace polar_to_rectangular_range_of_m_l640_640297

-- Definition of the parametric equations of curve C
def curve (t : ℝ) : ℝ × ℝ :=
  (sqrt(3) * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line (rho theta m : ℝ) : Prop :=
  rho * sin (theta + π / 3) + m = 0

-- Definition of the rectangular equation of line l
def rectangular_line (x y m : ℝ) : Prop :=
  sqrt(3) * x + y + 2 * m = 0

-- Convert polar line equation to rectangular form
theorem polar_to_rectangular (rho theta m x y : ℝ)
  (h1 : polar_line rho theta m)
  (h2 : rho = sqrt (x^2 + y^2))
  (h3 : cos theta * rho = x ∧ sin theta * rho = y) :
  rectangular_line x y m :=
sorry

-- Range of values for m for which l intersects C
theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, (sqrt(3) * cos (2 * t), 2 * sin t) ∈ {p : ℝ × ℝ | rectangular_line p.1 p.2 m}) ↔
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end polar_to_rectangular_range_of_m_l640_640297


namespace sum_of_distinct_natural_numbers_with_seven_zeros_and_seventy_two_divisors_l640_640504

theorem sum_of_distinct_natural_numbers_with_seven_zeros_and_seventy_two_divisors :
  ∃ (N1 N2 : ℕ), N1 ≠ N2 ∧ (N1 % 10^7 = 0) ∧ (N2 % 10^7 = 0) ∧ (num_divisors N1 = 72) ∧ (num_divisors N2 = 72) ∧ (N1 + N2 = 70000000) :=
sorry

end sum_of_distinct_natural_numbers_with_seven_zeros_and_seventy_two_divisors_l640_640504


namespace range_a_for_monotonicity_l640_640266

noncomputable def f : ℝ → ℝ → ℝ
| a, x := if (x < 1) then (a + 3) * x - 3 else 1 - a / (x + 1)

theorem range_a_for_monotonicity (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (0 < a ∧ a ≤ (2 / 3)) :=
by
  sorry

end range_a_for_monotonicity_l640_640266


namespace constant_distance_QR_l640_640409

open EuclideanGeometry

variable {A B C M P Q R : Point}
variable (triangleABC : Triangle A B C)
variable (midpointM : M = midpoint B C)
variable (insideP : (insideTriangle P A B C))
variable (angleCondition : ∠C P M = ∠P A B)
variable (circumCircleGamma : CircleGamma P A B)
variable (secondIntersectionQ : secondIntersectionPointLineCircle (lineThrough P M) circumCircleGamma Q)
variable (reflectionR : reflection P (tangentToCircle circumCircleGamma B) R)

theorem constant_distance_QR :
  ∀ P, Q R -- with necessary conditions above being held true
  (midpointM : M = midpoint B C)
  (insideP : (insideTriangle P A B C))
  (angleCondition : ∠C P M = ∠P A B)
  (circumCircleGamma : CircleGamma P A B)
  (secondIntersectionQ : secondIntersectionPointLineCircle (lineThrough P M) circumCircleGamma Q)
  (reflectionR : reflection P (tangentToCircle circumCircleGamma B) R),
  distance Q R = constant :=
sorry

end constant_distance_QR_l640_640409


namespace winning_candidate_percentage_l640_640105

theorem winning_candidate_percentage (votes1 votes2 votes3 : ℕ) (total_votes : ℕ)
(h1 : votes1 = 1036) (h2 : votes2 = 4636) (h3 : votes3 = 11628) (h_total : total_votes = votes1 + votes2 + votes3) :
  (votes3 : ℝ) / (total_votes : ℝ) * 100 ≈ 67.23 :=
by {
  sorry
}

end winning_candidate_percentage_l640_640105


namespace triangle_angle_conditions_l640_640805

theorem triangle_angle_conditions
  (a b c : ℝ)
  (α β γ : ℝ)
  (h_triangle : c^2 = a^2 + 2 * b^2 * Real.cos β)
  (h_tri_angles : α + β + γ = 180):
  (γ = β / 2 + 90 ∧ α = 90 - 3 * β / 2 ∧ 0 < β ∧ β < 60) ∨ 
  (α = β / 2 ∧ γ = 180 - 3 * β / 2 ∧ 0 < β ∧ β < 120) :=
sorry

end triangle_angle_conditions_l640_640805


namespace inequality_proof_l640_640764

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) >= 9 * (a * b + b * c + c * a) :=
begin
  sorry,
end

end inequality_proof_l640_640764


namespace smallest_x_for_multiple_of_720_l640_640075

theorem smallest_x_for_multiple_of_720 (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 720 = 2^4 * 3^2 * 5^1) : x = 8 ↔ (450 * x) % 720 = 0 :=
by
  sorry

end smallest_x_for_multiple_of_720_l640_640075


namespace min_value_of_function_l640_640936

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem min_value_of_function :
  ∃ x ∈ set.Icc (-1 : ℝ) 2, f x = -2 ∧ (∀ y ∈ set.Icc (-1 : ℝ) 2, f y ≥ -2) :=
by
  use 1
  split
  { -- Check if 1 is within the interval [-1, 2]
    norm_num
  }
  split
  { -- Prove that f(1) = -2
    norm_num
  }
  { -- Prove that -2 is the minimum value in the given interval
    intro y
    intro hy
    -- Remaining proof omitted
    sorry
  }

end min_value_of_function_l640_640936


namespace incorrect_statement_is_C_l640_640432

theorem incorrect_statement_is_C (b h s a x : ℝ) (hb : b > 0) (hh : h > 0) (hs : s > 0) (hx : x < 0) :
  ¬ (9 * s^2 = 4 * (3 * s)^2) :=
by
  sorry

end incorrect_statement_is_C_l640_640432


namespace EF_divides_BC_l640_640124

-- Define the points and geometric relations as given in the conditions.
variables {A B C D E F K : Point}
variables (ABCD : parallelogram A B C D)
variables (P1 : dropped_perpendicular C D F)
variables (P2 : dropped_perpendicular A (diagonal B D) F)
variables (P3 : dropped_perpendicular B A E)
variables (P4 : bisector_AC E)

-- Definition of ratio division
def ratio_divides (p q r : Point) (α β : ℕ) : Prop :=
  (dist p q) * β = (dist q r) * α

-- Statement: Prove that the segment EF divides side BC in the ratio 1:2.
theorem EF_divides_BC : ratio_divides E F B 1 2 :=
sorry

end EF_divides_BC_l640_640124


namespace rectangular_eq_of_line_l_range_of_m_l640_640350

noncomputable def curve_C_param_eq (t : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l_polar_eq (θ ρ m : ℝ) : Prop :=
  ρ * Real.sin(θ + Real.pi / 3) + m = 0

theorem rectangular_eq_of_line_l (x y m : ℝ) (h : line_l_polar_eq (Real.atan2 y x) (Real.sqrt (x^2 + y^2)) m) :
  sqrt 3 * x + y + 2 * m = 0 :=
  sorry

theorem range_of_m (m : ℝ) (t : ℝ) (h : ∃ (t : ℝ), curve_C_param_eq t ∈ set_of (λ p, p.2 = -√3 * p.1 - 2 * m))
  : -19/12 ≤ m ∧ m ≤ 5/2 :=
  sorry

end rectangular_eq_of_line_l_range_of_m_l640_640350


namespace factorial_expression_l640_640583

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_expression (N : ℕ) (h : N > 0) :
  (factorial (N + 1) + factorial (N - 1)) / factorial (N + 2) = 
  (N^2 + N + 1) / (N^3 + 3 * N^2 + 2 * N) :=
by
  sorry

end factorial_expression_l640_640583


namespace probability_log3_three_digit_integer_l640_640125

theorem probability_log3_three_digit_integer :
  let N := {n // 100 ≤ n ∧ n ≤ 999}
  let count_3_pow := (100 ≤ 3^5 ∧ 3^5 ≤ 999) + (100 ≤ 3^6 ∧ 3^6 ≤ 999)
  (count_3_pow : ℕ) = 2 → 
  (∀ n : N, ∃ k : ℕ, n = 3^k ↔ k = 5 ∨ k = 6) → -- All \(N\) in range can only be \(3^5\) or \(3^6\)
  (prob := (2 : ℕ) / (900 : ℕ)) = (1 / 450 : ℕ) :=
by sorry

end probability_log3_three_digit_integer_l640_640125


namespace rectangular_equation_of_l_range_of_m_intersection_l640_640364

-- Definitions based on the problem conditions
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  let x := sqrt 3 * cos (2 * t)
  let y := 2 * sin t
  (x, y)

def polar_line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- The rectangular equation converted from polar form
theorem rectangular_equation_of_l (x y m : ℝ) :
  (polar_line_l (sqrt (x^2 + y^2)) (atan2 y x) m) ↔ (√3 * x + y + 2 * m = 0) :=
sorry

-- Range of values for m ensuring intersection of line l with curve C
theorem range_of_m_intersection (m : ℝ) :
  ((-19/12 : ℝ) ≤ m ∧ m ≤ (5/2 : ℝ)) ↔
  ∃ t : ℝ, let x := sqrt 3 * cos (2 * t)
           let y := 2 * sin t
           (sqrt 3 * x + y + 2 * m = 0 ∧ (-2 ≤ y ∧ y ≤ 2)) :=
sorry

end rectangular_equation_of_l_range_of_m_intersection_l640_640364


namespace volume_of_one_piece_l640_640132

def cake_thickness := 1 / 2
def cake_diameter := 16
def num_pieces := 8

theorem volume_of_one_piece :
  let r := cake_diameter / 2 in
  let V := π * r^2 * cake_thickness in
  V / num_pieces = 4 * π :=
by
  let r := cake_diameter / 2
  let V := π * r^2 * cake_thickness
  show V / num_pieces = 4 * π
  sorry

end volume_of_one_piece_l640_640132


namespace exists_s_th_power_between_n_and_2n_l640_640773

theorem exists_s_th_power_between_n_and_2n (s : ℕ) (h : s > 1) :
  (∃ ms : ℕ, (∀ n : ℕ, n ≥ ms → ∃ k : ℕ, n < k^s ∧ k^s < 2*n)) ∧
  ((s = 2 → 5) ∧ (s = 3 → 33)) :=
begin
  sorry
end

end exists_s_th_power_between_n_and_2n_l640_640773


namespace max_x_add_2y_l640_640216

theorem max_x_add_2y (x y : ℝ) (h : 2^x + 4^y = 1) : x + 2*y ≤ -2 :=
sorry

end max_x_add_2y_l640_640216


namespace solve_for_x_l640_640456

theorem solve_for_x (x : ℝ) (h : 64^(3*x) = 16^(4*x - 5)) : x = -10 :=
by
  sorry

end solve_for_x_l640_640456


namespace triangle_area_le_two_in_six_points_l640_640631

-- Definitions of the problem scenario
def is_point (pt : (ℕ × ℕ)) : Prop :=
  0 ≤ pt.1 ∧ pt.1 ≤ 4 ∧ 0 ≤ pt.2 ∧ pt.2 ≤ 4

def no_three_collinear (pts : Finset (ℕ × ℕ)) : Prop :=
  ∀ (p1 p2 p3 : (ℕ × ℕ)),
    p1 ∈ pts →
    p2 ∈ pts →
    p3 ∈ pts →
    (p2.2 - p1.2) * (p3.1 - p1.1) ≠ (p3.2 - p1.2) * (p2.1 - p1.1)

def area (p1 p2 p3 : (ℕ × ℕ)) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

def has_triangle_with_area_le_two (pts : Finset (ℕ × ℕ)) : Prop :=
  ∃ (p1 p2 p3 : (ℕ × ℕ)),
    p1 ∈ pts ∧
    p2 ∈ pts ∧
    p3 ∈ pts ∧
    area p1 p2 p3 ≤ 2

-- Proof statement
theorem triangle_area_le_two_in_six_points (pts : Finset (ℕ × ℕ)) :
  pts.card = 6 →
  (∀ pt ∈ pts, is_point pt) →
  no_three_collinear pts →
  has_triangle_with_area_le_two pts :=
sorry -- proof omitted

end triangle_area_le_two_in_six_points_l640_640631


namespace arithmetic_sequence_a2_a6_l640_640726

theorem arithmetic_sequence_a2_a6 (a : ℕ → ℕ) (d : ℕ) (h_arith_seq : ∀ n, a (n+1) = a n + d)
  (h_a4 : a 4 = 4) : a 2 + a 6 = 8 :=
by sorry

end arithmetic_sequence_a2_a6_l640_640726


namespace average_first_18_even_l640_640537

theorem average_first_18_even :
  (∑ i in Finset.range 18, (2 * (i + 1))) / 18 = 19 :=
by
  sorry

end average_first_18_even_l640_640537


namespace summation_equivalence_l640_640506

theorem summation_equivalence : 
  ∀ n : ℕ, (∑ i in finset.range n, (-1) ^ i * (1 / (i + 1))) = (∑ i in finset.range n, 1 / (n + 1 + i)) := 
by {
  sorry
}

end summation_equivalence_l640_640506


namespace circle_and_cosine_solution_l640_640871

noncomputable def circle_and_cosine_problem : Prop :=
  ∀ (h k r : ℝ), ∃ max_intersections : ℕ,
  max_intersections = 8 ∧
  ∀ x y : ℝ, -2 * Real.pi ≤ x ∧ x ≤ 2 * Real.pi →
  ((x - h) ^ 2 + (y - k) ^ 2 = r ^ 2) →
  y = Real.cos x →
  max_intersections is the maximum number of intersection points.

theorem circle_and_cosine_solution : circle_and_cosine_problem :=
by sorry

end circle_and_cosine_solution_l640_640871


namespace coefficient_x3_correct_coefficient_x4_correct_l640_640594

def expression := 4 * (X^2 - 2 * X^3 + X^4 + X) + 2 * (X + 3 * X^3 - 2 * X^2 + 4 * X^5 - X^4) - 6 * (1 + X - 3 * X^3 + X^2 + 2 * X^4)

noncomputable def coefficient_x3 : Int := 
(x^3 term coefficient) -- Simplified manually as needed

noncomputable def coefficient_x4 : Int := 
(x^4 term coefficient) -- Simplified manually as needed

theorem coefficient_x3_correct : coefficient_x3 expression = 16 := by
  sorry

theorem coefficient_x4_correct : coefficient_x4 expression = -10 := by
  sorry

end coefficient_x3_correct_coefficient_x4_correct_l640_640594


namespace average_root_cross_sectional_area_average_volume_sample_correlation_coefficient_total_volume_estimation_l640_640565

open Real

-- Declarations about sums and average calculation
def sum_x := 0.6
def sum_y := 3.9
def n := 10
def avg_x := sum_x / n -- 0.06
def avg_y := sum_y / n -- 0.39

-- Declarations for the second part
def sum_x_squared := 0.038
def sum_y_squared := 1.6158
def sum_xy := 0.2474

-- Declaration for the third part
def total_root_area := 186.0

-- Theorems to prove
theorem average_root_cross_sectional_area :
  avg_x = 0.06 := by
  sorry

theorem average_volume :
  avg_y = 0.39 := by
  sorry

noncomputable def correlation_coefficient := 
  (sum_xy - n * avg_x * avg_y) / (sqrt ((sum_x_squared - n * avg_x^2) * (sum_y_squared - n * avg_y^2)))

theorem sample_correlation_coefficient :
  correlation_coefficient ≈ 0.97 := by
  sorry

noncomputable def estimated_total_volume := (avg_y / avg_x) * total_root_area

theorem total_volume_estimation :
  estimated_total_volume ≈ 1209 := by
  sorry

end average_root_cross_sectional_area_average_volume_sample_correlation_coefficient_total_volume_estimation_l640_640565


namespace john_money_left_l640_640737

theorem john_money_left 
  (start_amount : ℝ := 100) 
  (price_roast : ℝ := 17)
  (price_vegetables : ℝ := 11)
  (price_wine : ℝ := 12)
  (price_dessert : ℝ := 8)
  (price_bread : ℝ := 4)
  (price_milk : ℝ := 2)
  (discount_rate : ℝ := 0.15)
  (tax_rate : ℝ := 0.05)
  (total_cost := price_roast + price_vegetables + price_wine + price_dessert + price_bread + price_milk)
  (discount_amount := discount_rate * total_cost)
  (discounted_total := total_cost - discount_amount)
  (tax_amount := tax_rate * discounted_total)
  (final_amount := discounted_total + tax_amount)
  : start_amount - final_amount = 51.80 := sorry

end john_money_left_l640_640737


namespace median_of_first_twelve_positive_integers_l640_640515

theorem median_of_first_twelve_positive_integers :
  let S := (set.range 12).image (λ x, x + 1) in  -- Set of first twelve positive integers
  median S = 6.5 :=
by
  sorry

end median_of_first_twelve_positive_integers_l640_640515


namespace central_angle_of_chord_l640_640607

noncomputable def central_angle (l : LinearMap ℝ (ℝ × ℝ) ℝ) (k : ℝ) (r : ℝ) : ℝ :=
  2 * Real.arccos ((Real.sqrt (r^2 - (|k| / Real.sqrt (l 3 4))) / r))

theorem central_angle_of_chord (l : LinearMap ℝ (ℝ × ℝ) ℝ)
  (h_l : l (3, 4) = 5) (r : ℝ) (h_circle : r = 2) :
  central_angle l 5 r = 2 * Real.arccos (Real.sqrt 3 / 2) :=
sorry

end central_angle_of_chord_l640_640607


namespace log_f_2_eq_neg_one_l640_640690

noncomputable def power_function (α : ℝ) : (ℝ → ℝ) := λ x, x ^ α

theorem log_f_2_eq_neg_one {α : ℝ} (h : power_function α 3 = 1/3) : Real.logBase 2 (power_function α 2) = -1 := 
by
  have : (3 : ℝ) ^ α = 1 / 3 := h
  have α_eq_neg_one : α = -1 := sorry  -- This will be derived from the previous step
  have f_eq_inv : power_function α 2 = 2 ^ α := rfl
  rw [α_eq_neg_one] at f_eq_inv
  simp [f_eq_inv]
  sorry

end log_f_2_eq_neg_one_l640_640690


namespace fold_points_area_l640_640158

-- We define the given conditions in Lean.
def AB : ℝ := 45
def AC : ℝ := 90
def angle_B : ℝ := real.pi / 2 -- 90 degrees in radians

-- The main theorem that encodes the problem statement and solution.
theorem fold_points_area : 
  let Q : ℝ := 379
  let R : ℝ := 0
  let S : ℝ := 1
  q + r + s = 380 :=
by
  let q := 379
  let r := 0
  let s := 1
  have h1 : q = 379 := rfl
  have h2 : r = 0 := rfl
  have h3 : s = 1 := rfl
  rw [h1, h2, h3]
  exact rfl

end fold_points_area_l640_640158


namespace possible_license_plates_count_eq_1008_l640_640466

def Rotokas := {'A', 'E', 'G', 'I', 'K', 'O', 'R', 'T', 'U', 'V'}

def valid_license_plate (plate : list Char) : Prop :=
  plate.length = 5 ∧
  (plate.head = 'A' ∨ plate.head = 'E') ∧
  plate.get_last = 'V' ∧
  ∀ l, l ∈ {'S', 'P'} → l ∉ plate ∧
  plate.nodup

theorem possible_license_plates_count_eq_1008 :
  (finset.univ.filter (λ plate, valid_license_plate plate)).card = 1008 :=
sorry

end possible_license_plates_count_eq_1008_l640_640466


namespace polar_to_rectangular_intersection_range_l640_640336

-- Definitions of parametric equations for curve C
def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line_l (ρ θ m: ℝ) : Prop := ρ * sin (θ + π / 3) + m = 0

-- Rectangular equation of line l
def rectangular_line_l (x y m: ℝ) : Prop := sqrt 3 * x + y + 2 * m = 0

-- Prove that polar equation of line l can be transformed to rectangular form
theorem polar_to_rectangular (ρ θ m : ℝ) :
  (polar_line_l ρ θ m) → (∃ x y, ρ = sqrt (x^2 + y^2) ∧ θ = arctan (y / x) ∧ rectangular_line_l x y m) :=
  sorry

-- Prove that the range of values for m ensuring line l intersects curve C is [-19/12, 5/2]
theorem intersection_range (m : ℝ) :
  (∃ t : ℝ, rectangular_line_l (sqrt 3 * cos (2 * t)) (2 * sin t) m) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
  sorry

end polar_to_rectangular_intersection_range_l640_640336


namespace problem1a_problem1b_problem2_l640_640984

-- Definition of the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - 4*x + (2 - a)*Real.log x

-- Problem 1a: Interval where f(x) is increasing when a = 8
theorem problem1a (x : ℝ) : (3 < x ∧ 8 = 8) → deriv (λ x : ℝ, x^2 - 4*x - 6*Real.log x) x > 0 :=
by sorry

-- Problem 1b: Equation of the tangent line at (1, -3) for a = 8
theorem problem1b (x y : ℝ) : (x = 1 ∧ y = -3 ∧ 8 = 8) → 8*x + y - 5 = 0 :=
by sorry

-- Problem 2: Minimum value of f(x) on [e, e^2]
theorem problem2 (a : ℝ) (x : ℝ) : 
  (x ∈ Set.Icc Real.exp (Real.exp ^ 2)) → 
  (if a ≥ 2*(Real.exp^2 - 1)^2 then f x a = Real.exp^4 - 4*Real.exp^2 + 4 - 2*a
   else if 2*(Real.exp - 1)^2 < a ∧ a < 2*(Real.exp^2 - 1)^2 then f x a = (a/2) - Real.sqrt 2 * a - 3 + (2 - a)*Real.log (1 + Real.sqrt 2 * a / 2)
   else f x a = Real.exp^2 - 4*Real.exp + 2 - a) :=
by sorry

end problem1a_problem1b_problem2_l640_640984


namespace triangle_tangent_identity_l640_640732

theorem triangle_tangent_identity (A B C : ℝ) (a b c : ℝ) (G : ℝ × ℝ × ℝ)
(h1 : ∀ v₁ v₂ v₃ : ℝ × ℝ × ℝ, G = (v₁.1, v₂.2, v₃.3) → 
  (v₁.1 + v₁.2, v₂.1 + v₂.2, v₃.1 + v₃.2) = (0, 0, 0))
(h2 : ∀ v₁ v₂ : ℝ × ℝ, v1 = (a, b) → 
  v2 = (b, c) → v₁.1 * v₂.1 + v₁.2 * v₂.2 = 0)
(h3 : a^2 + b^2 = 5 * c^2) :
(\tan A + \tan B) * \tan C = (1 / 2) * \tan A * \tan B :=
sorry

end triangle_tangent_identity_l640_640732


namespace lemons_for_lemonade_l640_640099

theorem lemons_for_lemonade (lemons_gallons_ratio : 30 / 25 = x / 10) : x = 12 :=
by
  sorry

end lemons_for_lemonade_l640_640099


namespace odd_function_phi_l640_640020

theorem odd_function_phi (φ : ℝ) (hφ1 : 0 ≤ φ) (hφ2 : φ ≤ π) (h : ∀ x : ℝ, cos (x + φ) = - cos (-x + φ)) : φ = π / 2 :=
by
  sorry

end odd_function_phi_l640_640020


namespace points_opposite_sides_l640_640695

theorem points_opposite_sides (a : ℝ) :
  let A := (1, 3)
  let B := (-1, -4)
  let L := λ (x y : ℝ), a * x + 3 * y + 1
  L 1 3 * L (-1) (-4) < 0 → a < -11 ∨ a > -10 :=
by
  dsimp [L, A, B]
  sorry

end points_opposite_sides_l640_640695


namespace total_amount_sold_l640_640885

theorem total_amount_sold (metres_sold : ℕ) (loss_per_metre cost_price_per_metre : ℕ) 
  (h1 : metres_sold = 600) (h2 : loss_per_metre = 5) (h3 : cost_price_per_metre = 35) :
  (cost_price_per_metre - loss_per_metre) * metres_sold = 18000 :=
by
  sorry

end total_amount_sold_l640_640885


namespace complex_expression_calculation_l640_640472

noncomputable def complex_i := Complex.I -- Define the imaginary unit i

theorem complex_expression_calculation : complex_i * (1 - complex_i)^2 = 2 := by
  sorry

end complex_expression_calculation_l640_640472


namespace no_integers_solution_l640_640444

theorem no_integers_solution (k : ℕ) (x y z : ℤ) (hx1 : 0 < x) (hx2 : x < k) (hy1 : 0 < y) (hy2 : y < k) (hz : z > 0) :
  x^k + y^k ≠ z^k :=
sorry

end no_integers_solution_l640_640444


namespace point_A_equidistant_l640_640608

/-
This statement defines the problem of finding the coordinates of point A that is equidistant from points B and C.
-/
theorem point_A_equidistant (x : ℝ) :
  (dist (x, 0, 0) (3, 5, 6)) = (dist (x, 0, 0) (1, 2, 3)) ↔ x = 14 :=
by {
  sorry
}

end point_A_equidistant_l640_640608


namespace range_of_a_l640_640960

variable (a : ℝ)

def p : Prop := a ≤ -4 ∨ a ≥ 4
def q : Prop := a ≥ -12

-- Proposition (p ∨ q) is true
axiom hpq_or : p a ∨ q a
-- Proposition (p ∧ q) is false
axiom hpq_and_not : ¬ (p a ∧ q a)

theorem range_of_a : a ∈ Set.Ioo (-∞) (-12) ∪ Set.Ioo (-4) 4 :=
by
  sorry

end range_of_a_l640_640960


namespace line_equation_max_value_x_plus_y_l640_640692

-- Define the polar equation conditions
def polar_equation (ρ θ : ℝ) : Prop := (ρ^2 - 2 * sqrt(2) * ρ * cos (θ + π / 4) - 2 = 0)

-- Define Cartesian coordinates connection to polar coordinates
def to_cartesian (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * cos θ ∧ y = ρ * sin θ

-- Define the curve C in Cartesian form
def curve_c (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 4

-- Lean statement for the first proof problem
theorem line_equation (O C: ℝ × ℝ) : 
  (O = (0, 0)) → (C = (1, -1)) → 
  (∀ P : ℝ × ℝ, (x' - 1)^2 + (y' + 1)^2 = 4 → P ≠ C → (P - O) • (P - C) = 0) →
  ∃ m : ℝ, m = 1 ∧ ∀ P : ℝ × ℝ, (y' = m * x') :=
sorry

-- Define expression for x + y and find its maximum value
def expression_x_y_max (θ: ℝ): ℝ := 1 + 2 * cos(θ) + (-1 + 2 * sin(θ))

theorem max_value_x_plus_y (θ : ℝ) : 
  (M : ℝ × ℝ) →
  (∀ θ : ℝ, (1 + 2 * cos θ, -1 + 2 * sin θ) = M) →
  ∃ max : ℝ, max = 2 * sqrt(2) ∧ ∀ θ, (1 + 2 * cos θ + -1 + 2 * sin θ) ≤ max
sorry

end line_equation_max_value_x_plus_y_l640_640692


namespace min_percentage_reduction_l640_640724

-- Definitions and conditions
def P2004 : ℝ := 1

def P2005 : ℝ := 0.85 * P2004

def P2006 (x : ℝ) : ℝ := P2005 * (1 - x / 100)

-- Theorem statement
theorem min_percentage_reduction (x : ℝ) : P2006 x ≤ 0.75 * P2004 → x ≥ 11.8 :=
by
  sorry

end min_percentage_reduction_l640_640724


namespace area_T_prime_l640_640399

-- Define the matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, -1], ![8, 4]]

-- Define the area of the initial region T
def area_T : ℝ := 15

-- State the theorem
theorem area_T_prime (det_A : ℝ) (area_T' : ℝ) : det_A = |Matrix.det A| → area_T' = det_A * area_T → area_T' = 300 :=
by
  assume h1 : det_A = |Matrix.det A|
  assume h2 : area_T' = det_A * area_T
  sorry

end area_T_prime_l640_640399


namespace percentage_of_integers_divisible_by_7_in_range_101_to_200_is_14_l640_640527

theorem percentage_of_integers_divisible_by_7_in_range_101_to_200_is_14 :
  let total_numbers := 200 - 101 + 1,
      divisors_of_7 := (200 / 7).nat_ceil - (101 / 7).nat_floor,
      percentage := (divisors_of_7 * 100) / total_numbers
  in percentage = 14 :=
by
  sorry

end percentage_of_integers_divisible_by_7_in_range_101_to_200_is_14_l640_640527


namespace tetrahedron_sphere_properties_l640_640471

variables {V : Type*} [EuclideanSpace V]

def circumcenter (tetra : V × V × V × V) : V := sorry -- definition skipped
def centroid (tetra : V × V × V × V) : V := sorry -- definition skipped
def reflect_about (v1 v2 : V) : V := sorry -- definition skipped
def sphere_center_of_centroids (tetra : V × V × V × V) : V := sorry -- definition skipped
def circumradius (tetra : V × V × V × V) : ℝ := sorry -- definition skipped
def sphere_radius_of_centroids (tetra : V × V × V × V) : ℝ := sorry -- definition skipped
def point_divides_segment_closer (p1 p2 : V) (r : ℕ) : V := sorry -- definition skipped
def perpendicular_projection (p : V) (face : V × V × V) : V := sorry -- definition skipped

theorem tetrahedron_sphere_properties 
  (tetra : V × V × V × V)
  (O := circumcenter tetra)
  (S := centroid tetra)
  (T := reflect_about O S)
  (F := sphere_center_of_centroids tetra) :
  point_divides_segment_closer O T 2 = F ∧
  sphere_radius_of_centroids tetra = circumradius tetra / 3 ∧
  ∀ (v : V), v ∈ vertices tetra → 
    point_divides_segment_closer T v 2 ∈ sphere_passes_through G ∧ 
    perpendicular_projection (point_divides_segment_closer T v 2) (opposite_face v tetra) ∈ sphere_passes_through G := 
sorry

end tetrahedron_sphere_properties_l640_640471


namespace root_interval_a_range_l640_640267

def f (a x : ℝ) : ℝ := x^2 - x * log (a * x) - exp (x - 1) / a

theorem root_interval_a_range (a : ℝ) (h : a > 0) : (∃ x : ℝ, 0 < x ∧ f a x = 0) ↔ a ∈ {a | 1 ≤ a} :=
by {
  sorry -- Proof goes here
}

end root_interval_a_range_l640_640267


namespace fraction_comparison_l640_640635

theorem fraction_comparison (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a / b) > (a + 1) / (b + 1) :=
by
  sorry

end fraction_comparison_l640_640635


namespace paint_cans_used_l640_640439

theorem paint_cans_used (initial_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) 
    (h1 : initial_rooms = 50) (h2 : lost_cans = 5) (h3 : remaining_rooms = 40) : 
    (remaining_rooms / (initial_rooms - remaining_rooms) / lost_cans) = 20 :=
by
  sorry

end paint_cans_used_l640_640439


namespace determine_triangle_shape_and_size_l640_640845

theorem determine_triangle_shape_and_size :
  (∀ {A B C : Type} (angle_A angle_B : ℝ) (side_AB : ℝ), 
    angle_A = 50 ∧ angle_B = 50 ∧ side_AB = 5 → 
    ∃ (ΔABC : Triangle), ΔABC.angle_A = 50 ∧ ΔABC.angle_B = 50 ∧ ΔABC.side_AB = 5 
    ∧ (∀ (ΔABC' : Triangle), ΔABC'.angle_A = 50 ∧ ΔABC'.angle_B = 50 ∧ ΔABC'.side_AB = 5 → ΔABC = ΔABC')) := 
sorry

end determine_triangle_shape_and_size_l640_640845


namespace range_of_a_l640_640235

noncomputable def f (x a : ℝ) : ℝ := x * |x^2 - a|

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f x a < 2) → -1 < a ∧ a < 5 :=
by
  intros
  cases ‹∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f x a < 2› with x hx
  have h1 := hx.1
  have h2 := hx.2.1
  have h3 := hx.2.2
  sorry

end range_of_a_l640_640235


namespace jelly_beans_total_l640_640423

-- Definitions from the conditions
def vanilla : Nat := 120
def grape : Nat := 5 * vanilla + 50
def total : Nat := vanilla + grape

-- Statement to prove
theorem jelly_beans_total :
  total = 770 := 
by 
  sorry

end jelly_beans_total_l640_640423


namespace complete_collection_prob_l640_640052

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end complete_collection_prob_l640_640052


namespace fraction_order_l640_640846

theorem fraction_order :
  (24 : ℚ) / 19 < 23 / 17 ∧ 23 / 17 < 11 / 8 :=
by 
   -- Conditions given
  have h1 : (11 : ℚ) / 8 = 11 / 8 := rfl,
  have h2 : (23 : ℚ) / 17 = 23 / 17 := rfl,
  have h3 : (24 : ℚ) / 19 = 24 / 19 := rfl,
  -- Define the proven inequalities
  have h4 : 24 * 17 < 23 * 19 := 
    by { norm_num, exact nat.lt_of_sub_pos (by norm_num) },
  have h5 : 23 * 8 < 11 * 17 := 
    by { norm_num, exact nat.lt_of_sub_pos (by norm_num) },
  -- Prove the order using the inequalities.
  exact ⟨by { field_simp, assumption }, by { field_simp, assumption }⟩,
  sorry -- placeholder to complete the proof 

end fraction_order_l640_640846


namespace order_of_magnitude_l640_640669

theorem order_of_magnitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let m := a / Real.sqrt b + b / Real.sqrt a
  let n := Real.sqrt a + Real.sqrt b
  let p := Real.sqrt (a + b)
  m ≥ n ∧ n > p := 
sorry

end order_of_magnitude_l640_640669


namespace no_real_solution_l640_640181

theorem no_real_solution (x : ℝ) : x + 64 / (x + 3) ≠ -13 :=
by {
  -- Proof is not required, so we mark it as sorry.
  sorry
}

end no_real_solution_l640_640181


namespace water_volume_to_sea_per_minute_l640_640555

def depth : ℝ := 4  -- depth of the river in meters
def width : ℝ := 40  -- width of the river in meters
def flow_rate_kmph : ℝ := 4  -- flow rate of the river in kilometers per hour

-- Convert flow rate to meters per minute
def flow_rate_m_per_min : ℝ := flow_rate_kmph * 1000 / 60

-- Calculate cross-sectional area of the river
def area : ℝ := depth * width

-- Calculate volume of water running into the sea per minute
def volume_per_minute : ℝ := area * flow_rate_m_per_min

theorem water_volume_to_sea_per_minute :
  volume_per_minute = 10666.67 := by
    sorry

end water_volume_to_sea_per_minute_l640_640555


namespace seq_theorem_l640_640648

-- Definitions and conditions
def seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  (∀ n : ℕ, S (n + 1) = S n + 2 * n) ∧
  (a 1, a 2, a 3).geometric_sequence

-- Problem theorem
theorem seq_theorem (a S : ℕ → ℕ) (h : seq a S) : 
  ∃ c : ℕ, c = 2 ∧ (∀ n : ℕ, S n = n^2 - n + 1) :=
begin
  sorry
end

end seq_theorem_l640_640648


namespace height_of_flagpole_l640_640069

theorem height_of_flagpole 
  (house_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ) 
  (flagpole_shadow : ℝ) (house_height : ℝ)
  (h1 : house_shadow = 70)
  (h2 : tree_height = 28)
  (h3 : tree_shadow = 40)
  (h4 : flagpole_shadow = 25)
  (h5 : house_height = (tree_height * house_shadow) / tree_shadow) :
  round ((house_height * flagpole_shadow / house_shadow) : ℝ) = 18 := 
by
  sorry

end height_of_flagpole_l640_640069


namespace complex_division_l640_640785

-- Define the imaginary unit 'i'
def i := Complex.I

-- Define the complex numbers as described in the problem
def num := Complex.mk 3 (-1)
def denom := Complex.mk 1 (-1)
def expected := Complex.mk 2 1

-- State the theorem to prove that the complex division is as expected
theorem complex_division : (num / denom) = expected :=
by
  sorry

end complex_division_l640_640785


namespace accurate_location_determination_l640_640562

-- Define conditions for each of the given options
def optionA := "Hall 3, Row 2, Beiguo Cinema"
def optionB := "Middle section of Shoujing Road"
def optionC := (116 : ℝ, 42 : ℝ)  -- Coordinates (Longitude, Latitude)
def optionD := "Southward 40° eastward"

-- Define a predicate that checks if an option determines an accurate location
def determines_accurate_location (option : Type) : Prop :=
  match option with
  | (lng, lat) : ℝ × ℝ => true
  | _ => false

-- The theorem to prove
theorem accurate_location_determination : 
  determines_accurate_location optionC = true ∧
  determines_accurate_location optionA = false ∧
  determines_accurate_location optionB = false ∧
  determines_accurate_location optionD = false :=
by 
  sorry

end accurate_location_determination_l640_640562


namespace nine_numbers_sum_multiple_of_9_l640_640636

theorem nine_numbers_sum_multiple_of_9
  (n : ℕ)
  (h : n ≥ 13)
  (a : Fin n → ℕ)
  : ∃ (i : Fin 9 → Fin n) (b : Fin 9 → ℕ), 
    (∀ j : Fin 9, b j ∈ ({4, 7} : Set ℕ)) ∧ 
    ((∑ j : Fin 9, b j * a (i j)) % 9 = 0) :=
sorry

end nine_numbers_sum_multiple_of_9_l640_640636


namespace factorial_division_l640_640660

theorem factorial_division : ∀ (a b : ℕ), a = 10! ∧ b = 4! → a / b = 151200 :=
by
  intros a b h
  cases h with ha hb
  rw [ha, Nat.factorial, Nat.factorial] at hb,
  norm_num at hb,
  exact sorry

end factorial_division_l640_640660


namespace maximize_profit_l640_640474

def fixed_cost := 20

def C (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 100 * x
  else if x ≥ 40 then 501 * x + 10000 / x - 4500
  else 0  -- Cost isn't defined for x <= 0, considering it be 0 here

def revenue (x : ℝ) : ℝ := 500 * x

def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then -10 * x^2 + 400 * x - 2000
  else if x ≥ 40 then - x - 10000 / x + 2500
  else 0

theorem maximize_profit :
  L 100 = 2300 := sorry

end maximize_profit_l640_640474


namespace total_jelly_beans_l640_640422

-- Given conditions:
def vanilla_jelly_beans : ℕ := 120
def grape_jelly_beans := 5 * vanilla_jelly_beans + 50

-- The proof problem:
theorem total_jelly_beans :
  vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  -- Proof steps would go here
  sorry

end total_jelly_beans_l640_640422


namespace rectangular_equation_common_points_l640_640322

-- Define parametric equations of curve C
def x (t : ℝ) := (sqrt 3) * cos (2 * t)
def y (t : ℝ) := 2 * sin t

-- Define polar equation of line l
def polar_line (ρ θ m : ℝ) := ρ * sin (θ + π / 3) + m

-- Rectangular equation of l
theorem rectangular_equation (ρ θ x y m : ℝ) (h1 : ρ = sqrt (x^2 + y^2)) (h2 : θ = atan2 y x) :
  polar_line ρ θ m = 0 → (sqrt 3 * x + y + 2 * m = 0) :=
by 
  sorry

-- Range of values for m where line intersects curve
theorem common_points (h: ∃ t, sqrt 3 * cos (2 * t) = x ∧ 2 * sin t = y) :
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by 
  sorry

end rectangular_equation_common_points_l640_640322


namespace range_of_a_if_functions_decreasing_l640_640674

-- Given conditions
def f (a b x : ℝ) : ℝ := (2 * a + 1) * x + b
def g (a x : ℝ) : ℝ := x^2 - 2 * (1 - a) * x + 2

-- Prove the range of values for a
theorem range_of_a_if_functions_decreasing (a b : ℝ) :
  (∀ x ∈ Iic 4, deriv (f a b) x < 0) ∧ (∀ x ∈ Iic 4, deriv (g a) x < 0) → a ≤ -3 :=
by
  -- The detailed proof steps would go here
  sorry

end range_of_a_if_functions_decreasing_l640_640674


namespace frank_final_steps_l640_640629

def final_position : ℤ :=
  let initial_back := -5
  let first_forward := 10
  let second_back := -2
  let second_forward := 2 * 2
  initial_back + first_forward + second_back + second_forward

theorem frank_final_steps : final_position = 7 := by
  simp
  sorry

end frank_final_steps_l640_640629


namespace largest_prime_factor_11236_l640_640610

theorem largest_prime_factor_11236 :
  ∃ p : ℕ, prime p ∧ p ∣ 11236 ∧ (∀ q : ℕ, prime q ∧ q ∣ 11236 → q ≤ p) :=
  sorry

end largest_prime_factor_11236_l640_640610


namespace ilya_defeats_dragon_l640_640274

noncomputable def prob_defeat : ℝ := 1 / 4 * 2 + 1 / 3 * 1 + 5 / 12 * 0

theorem ilya_defeats_dragon : prob_defeat = 1 := sorry

end ilya_defeats_dragon_l640_640274


namespace proof_problem_l640_640257

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
def is_odd_function := ∀ x ∈ Set.Icc (-3 : ℝ) 3, f (-x) = -f x
def condition_1 := is_odd_function f
def condition_2 := f 3 = -3

-- Statement we want to prove
theorem proof_problem (h1 : condition_1) (h2 : condition_2) : f (-3) + f 0 = 3 :=
by {
  sorry,
}

end proof_problem_l640_640257


namespace total_cans_collected_l640_640017

theorem total_cans_collected (total_students : ℕ) (half_students_collecting_12 : ℕ) 
 (students_collecting_0 : ℕ) (remaining_students_collecting_4 : ℕ) 
 (cans_collected_by_half : ℕ) (cans_collected_by_remaining : ℕ) :
 total_students = 30 →
 half_students_collecting_12 = 15 →
 students_collecting_0 = 2 →
 remaining_students_collecting_4 = 13 →
 cans_collected_by_half = half_students_collecting_12 * 12 →
 cans_collected_by_remaining = remaining_students_collecting_4 * 4 →
 (cans_collected_by_half + students_collecting_0 * 0 + cans_collected_by_remaining) = 232 :=
 by
  intros h1 h2 h3 h4 h5 h6
  obtain rfl : 15 * 12 = 180 := rfl
  obtain rfl : 13 * 4 = 52 := rfl
  rw [h5, h6]
  simp
  sorry

end total_cans_collected_l640_640017


namespace jelly_beans_total_l640_640425

-- Definitions from the conditions
def vanilla : Nat := 120
def grape : Nat := 5 * vanilla + 50
def total : Nat := vanilla + grape

-- Statement to prove
theorem jelly_beans_total :
  total = 770 := 
by 
  sorry

end jelly_beans_total_l640_640425


namespace frank_final_steps_l640_640628

def final_position : ℤ :=
  let initial_back := -5
  let first_forward := 10
  let second_back := -2
  let second_forward := 2 * 2
  initial_back + first_forward + second_back + second_forward

theorem frank_final_steps : final_position = 7 := by
  simp
  sorry

end frank_final_steps_l640_640628


namespace hyperbola_asymptotes_l640_640794

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 - 3 * y^2 = 1) → (y = sqrt 3 / 3 * x ∨ y = -sqrt 3 / 3 * x) :=
by
  intro h
  sorry

end hyperbola_asymptotes_l640_640794


namespace distance_between_first_and_last_trees_l640_640599

theorem distance_between_first_and_last_trees : 
  ∀ (n : ℕ) (d : ℕ), 
  n = 8 → d = 100 → 
  distance_from_first_to_last_tree n d = 175 := 
by
  intro n d h1 h2
  sorry

-- Definitions used in Lean 4 statement

def distance_from_first_to_last_tree (num_trees : ℕ) (distance_first_to_fifth : ℕ) : ℕ := 
  let intervals_between_first_and_fifth := 4
  let distance_per_interval := distance_first_to_fifth / intervals_between_first_and_fifth
  let total_intervals := num_trees - 1
  total_intervals * distance_per_interval

end distance_between_first_and_last_trees_l640_640599


namespace find_median_of_first_twelve_positive_integers_l640_640518

def median_of_first_twelve_positive_integers : ℚ :=
  let A := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (A[5] + A[6]) / 2

theorem find_median_of_first_twelve_positive_integers :
  median_of_first_twelve_positive_integers = 6.5 :=
by
  sorry

end find_median_of_first_twelve_positive_integers_l640_640518


namespace largest_integer_condition_l640_640836

theorem largest_integer_condition (x : ℤ) : (x/3 + 3/4 : ℚ) < 7/3 → x ≤ 4 :=
by
  sorry

end largest_integer_condition_l640_640836


namespace rectangular_eq_of_line_l_range_of_m_l640_640346

noncomputable def curve_C_param_eq (t : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l_polar_eq (θ ρ m : ℝ) : Prop :=
  ρ * Real.sin(θ + Real.pi / 3) + m = 0

theorem rectangular_eq_of_line_l (x y m : ℝ) (h : line_l_polar_eq (Real.atan2 y x) (Real.sqrt (x^2 + y^2)) m) :
  sqrt 3 * x + y + 2 * m = 0 :=
  sorry

theorem range_of_m (m : ℝ) (t : ℝ) (h : ∃ (t : ℝ), curve_C_param_eq t ∈ set_of (λ p, p.2 = -√3 * p.1 - 2 * m))
  : -19/12 ≤ m ∧ m ≤ 5/2 :=
  sorry

end rectangular_eq_of_line_l_range_of_m_l640_640346


namespace find_median_of_first_twelve_positive_integers_l640_640519

def median_of_first_twelve_positive_integers : ℚ :=
  let A := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (A[5] + A[6]) / 2

theorem find_median_of_first_twelve_positive_integers :
  median_of_first_twelve_positive_integers = 6.5 :=
by
  sorry

end find_median_of_first_twelve_positive_integers_l640_640519


namespace max_value_of_f_l640_640025

noncomputable def f (x : ℝ) : ℝ := (-2 * x^2 + x - 3) / x

theorem max_value_of_f : 
  ∀ x > 0, f(x) ≤ 1 - 2 * Real.sqrt 6 :=
begin
  sorry
end

end max_value_of_f_l640_640025


namespace difference_in_peaches_l640_640906

-- Define the number of peaches Audrey has
def audrey_peaches : ℕ := 26

-- Define the number of peaches Paul has
def paul_peaches : ℕ := 48

-- Define the expected difference
def expected_difference : ℕ := 22

-- The theorem stating the problem
theorem difference_in_peaches : (paul_peaches - audrey_peaches = expected_difference) :=
by
  sorry

end difference_in_peaches_l640_640906


namespace find_t_value_l640_640995

theorem find_t_value (t : ℝ) (a b : ℝ × ℝ) (h₁ : a = (t, 1)) (h₂ : b = (1, 2)) 
  (h₃ : (a.1 + b.1)^2 + (a.2 + b.2)^2 = a.1^2 + a.2^2 + b.1^2 + b.2^2) : 
  t = -2 :=
by 
  sorry

end find_t_value_l640_640995


namespace correct_statements_l640_640956

variable (α β : Plane)
variable (l : Line)
variable (P : Point)

-- Conditions 
axiom perp_planes : Perpendicular α β
axiom intersect_planes : PlaneIntersect α β l
axiom point_in_plane : InPlane P α
axiom point_not_in_line : ¬ OnLine P l

-- Questions
theorem correct_statements :
  (∀ (π : Plane), perp_to_line P l α π → Perpendicular π β) ∧
  (¬ (∀ (line : Line), perp_to_line P l line → Perpendicular line β)) ∧
  (∀ (line : Line), perp_to_plane P α line → Parallel line β) ∧
  (∀ (line : Line), perp_to_plane P β line → InPlane line α) :=
by
  sorry

end correct_statements_l640_640956


namespace locus_of_M_is_circle_l640_640092

theorem locus_of_M_is_circle
  {A B A1 B1 M : Type}
  (on_circle : A → B → A1 → B1 → Prop)
  (arc_length_constant : ∀ A1 B1, constant (arc_length A1 B1))
  (M_def : ∀ A1 B1, M ∈ intersection (AA1_line A A1) (BB1_line B B1)) :
  locus M A B :=
sorry

end locus_of_M_is_circle_l640_640092


namespace NumberDivisibleBy9_l640_640463

-- Define 63,477 as a specific case and verify its divisibility by 9.
theorem NumberDivisibleBy9 : (6 + 3 + 4 + 7 + 7) % 9 = 0 :=
by
  -- Sum the digits first
  have sum_digits : 6 + 3 + 4 + 7 + 7 = 27 := by norm_num
  -- Verify divisibility by 9
  show 27 % 9 = 0 from by norm_num استعمال_

end NumberDivisibleBy9_l640_640463


namespace factorial_division_l640_640667

theorem factorial_division : (10! / 4! = 151200) :=
by
  have fact_10 : 10! = 3628800 := by sorry
  rw [fact_10]
  -- Proceeding with calculations and assumptions that follow directly from the conditions
  have fact_4 : 4! = 24 := by sorry
  rw [fact_4]
  exact (by norm_num : 3628800 / 24 = 151200)

end factorial_division_l640_640667


namespace intersect_rectangular_eqn_range_of_m_l640_640352

-- Definitions of the parametric and polar equations
def curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Proving the rectangular equation and range of m for the intersection of l and C
theorem intersect_rectangular_eqn (m t : ℝ) :
  (∃ ρ θ, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  ∃ x y, (x, y) = curve_C t ∧ √3 * x + y + 2 * m = 0 :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ ρ θ t, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end intersect_rectangular_eqn_range_of_m_l640_640352


namespace perfect_cube_probability_l640_640900

theorem perfect_cube_probability (m n : ℕ) 
  (hrel_primes : Nat.gcd m n = 1)
  (hprob : ∀ (rolls : Fin 8 → ℕ), 
            (∑ x in Finset.univ.image (λ i, rolls i), x) = 5 → 
            (∏ x in Finset.univ.image (λ i, rolls i), x) % 512 = 7⁻³) :
  m + n = 525 :=
sorry

end perfect_cube_probability_l640_640900


namespace interval_length_l640_640161

theorem interval_length (c d : ℝ) (h : ∀ x : ℝ, c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d)
    (length_condition : ∀ x : ℝ, (x ≤ (d-5)/3) - (x ≥ (c-5)/3) = 12) : d - c = 36 := sorry

end interval_length_l640_640161


namespace half_dollar_difference_l640_640769

theorem half_dollar_difference (n d h : ℕ) 
  (h1 : n + d + h = 150) 
  (h2 : 5 * n + 10 * d + 50 * h = 1500) : 
  ∃ h_max h_min, (h_max - h_min = 16) :=
by sorry

end half_dollar_difference_l640_640769


namespace find_P_Q_sum_l640_640704

theorem find_P_Q_sum (P Q : ℤ) 
  (h : ∃ b c : ℤ, x^2 + 3 * x + 2 ∣ x^4 + P * x^2 + Q 
    ∧ b + 3 = 0 
    ∧ c + 3 * b + 6 = P 
    ∧ 3 * c + 2 * b = 0 
    ∧ 2 * c = Q): 
  P + Q = 3 := 
sorry

end find_P_Q_sum_l640_640704


namespace vertex_disjoint_cycles_l640_640394

variable (G : SimpleGraph V) (n : ℕ)

theorem vertex_disjoint_cycles (hn : n ≥ 5) (he : G.edge_count ≥ n + 4) : 
  ∃ C₁ C₂ : Finset (Finset V), G.is_cycle C₁ ∧ G.is_cycle C₂ ∧ C₁ ∩ C₂ = ∅ := 
sorry

end vertex_disjoint_cycles_l640_640394


namespace jar_lasts_20_days_l640_640382

def serving_size : ℝ := 0.5
def daily_servings : ℕ := 3
def container_size : ℝ := 32 - 2

def daily_usage (serving_size : ℝ) (daily_servings : ℕ) : ℝ :=
  serving_size * daily_servings

def days_to_finish (container_size daily_usage : ℝ) : ℝ :=
  container_size / daily_usage

theorem jar_lasts_20_days :
  days_to_finish container_size (daily_usage serving_size daily_servings) = 20 :=
by
  sorry

end jar_lasts_20_days_l640_640382


namespace father_of_pi_l640_640798

def chinese_mathematician (name : String) (dynasties : String) (achievements : String) : Prop :=
  name = "Zu Chongzhi" ∧ dynasties = "Southern and Northern Dynasties" ∧ achievements = "calculated the value of π up to the 9th decimal place"

theorem father_of_pi : ∃ (name : String),
  chinese_mathematician name "Southern and Northern Dynasties" "calculated the value of π up to the 9th decimal place" ∧
  name = "Zu Chongzhi" :=
by
  exists "Zu Chongzhi"
  unfold chinese_mathematician
  constructor
  repeat { split }
  · exact rfl
  · exact rfl
  · exact rfl

end father_of_pi_l640_640798


namespace inequality_sqrt_am_gm_l640_640772

theorem inequality_sqrt_am_gm (a : ℝ) (h : a > 0) : 
  sqrt (a^2 + 1/a^2) + 2 ≥ a + 1/a + sqrt 2 :=
sorry

end inequality_sqrt_am_gm_l640_640772


namespace line_parallel_l640_640203

theorem line_parallel (α β γ : Plane) (l₁ l₂ l₃ : Line)
  (hαβ : α ∩ β = l₁)
  (hαγ : α ∩ γ = l₂)
  (hβγ : β ∩ γ = l₃)
  (hl₁l₂_par : l₁ ∥ l₂) :
  l₁ ∥ l₃ :=
sorry

end line_parallel_l640_640203


namespace project_selection_l640_640872

theorem project_selection :
  let num_key_projects := 4
  let num_general_projects := 6
  let ways_select_key := Nat.choose 4 2
  let ways_select_general := Nat.choose 6 2
  let total_ways := ways_select_key * ways_select_general
  let ways_excluding_A := Nat.choose 3 2
  let ways_excluding_B := Nat.choose 5 2
  let ways_excluding_A_and_B := ways_excluding_A * ways_excluding_B
  total_ways - ways_excluding_A_and_B = 60
:= by
  sorry

end project_selection_l640_640872


namespace factorial_division_l640_640657

theorem factorial_division : ∀ (a b : ℕ), a = 10! ∧ b = 4! → a / b = 151200 :=
by
  intros a b h
  cases h with ha hb
  rw [ha, Nat.factorial, Nat.factorial] at hb,
  norm_num at hb,
  exact sorry

end factorial_division_l640_640657


namespace union_M_N_l640_640694

open Set

variable {α : Type*} [LinearOrder α]

-- Sets M and N
def M : Set α := {x | -3 < x ∧ x ≤ 5}
def N : Set α := {x | x < -5 ∨ 5 < x}

-- Theorem statement
theorem union_M_N : M ∪ N = {x | x < -5 ∨ -3 < x} :=
by
  sorry

end union_M_N_l640_640694


namespace rate_per_sq_meter_is_900_l640_640484

/-- The length of the room L is 7 (meters). -/
def L : ℝ := 7

/-- The width of the room W is 4.75 (meters). -/
def W : ℝ := 4.75

/-- The total cost of paving the floor is Rs. 29,925. -/
def total_cost : ℝ := 29925

/-- The rate per square meter for the slabs is Rs. 900. -/
theorem rate_per_sq_meter_is_900 :
  total_cost / (L * W) = 900 :=
by
  sorry

end rate_per_sq_meter_is_900_l640_640484


namespace cube_section_volume_l640_640112

theorem cube_section_volume : 
  ∀ (A P Q : ℝ × ℝ × ℝ) (l : ℝ),
  A = (0, 0, 0) →
  P = (2, 1, 0) →
  Q = (0, 0, 1) →
  l = 2 →
  let total_volume := l^3 in
  let base_area := 1 in
  let height := 2 in
  let small_pyramid_volume := (1 / 3) * base_area * height in
  total_volume - small_pyramid_volume = 22 / 3 :=
begin
  sorry
end

end cube_section_volume_l640_640112


namespace angle_D_is_60_l640_640729

variables (A B C D E : Point)
variables (AB BC CD CE : LineSegment)
variable (angle : Triangle -> ℝ)

-- Conditions
axiom intersect (BD AE : LineSegment) : BD ∩ AE = C
axiom segments_equal : AB = BC ∧ BC = CD ∧ CD = CE
axiom equilateral_ABC : ∀ (Δ : Triangle), Δ = ABC → angle Δ = 60
axiom equilateral_CDE : ∀ (Δ : Triangle), Δ = CDE → angle Δ = 60
axiom angle_relation : angle A = 3 * angle B

-- Question: Determine the measure of ∠D
theorem angle_D_is_60 : angle D = 60 := by
  sorry

end angle_D_is_60_l640_640729


namespace remainder_sum_mod_13_l640_640084

theorem remainder_sum_mod_13 (a b c d : ℕ) 
(h₁ : a % 13 = 3) (h₂ : b % 13 = 5) (h₃ : c % 13 = 7) (h₄ : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 :=
by sorry

end remainder_sum_mod_13_l640_640084


namespace y_increase_by_41_8_units_l640_640716

theorem y_increase_by_41_8_units :
  ∀ (x y : ℝ),
    (∀ k : ℝ, y = 2 + k * 11 / 5 → x = 1 + k * 5) →
    x = 20 → y = 41.8 :=
by
  sorry

end y_increase_by_41_8_units_l640_640716


namespace find_ratio_l640_640375

-- Define the points and conditions
variables {A B C D E F : Type*} [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] [vector_space ℝ D] [vector_space ℝ E] [vector_space ℝ F]
variables (a b c : A)
variables (d : B)
variables (e : C)
variables (ad db : ℝ) (be ec : ℝ)
variables (DE EF : ℝ)

-- Hypotheses based on given conditions
hypothesis h1 : ad = 2 * db
hypothesis h2 : be = 2 * ec
hypothesis h3 : ∃ F, F = intersection D E A C

-- The goal to prove
theorem find_ratio (h1 : ad = 2 * db) (h2 : be = 2 * ec) (h3 : ∃ F, F = intersection D E A C) : DE / EF = 1 / 2 :=
sorry

end find_ratio_l640_640375


namespace count_true_propositions_l640_640200

open Real

def f (x : ℝ) := x^3 - 3 * x^2

def f_prime (x : ℝ) := 3 * x^2 - 6 * x

theorem count_true_propositions :
  (  ¬∀ x, f_prime x ≥ 0 ∧ ¬ ( ∃ x, f_prime x = 0 ∧ f x ≠ 0) ) ∧
  (  ¬∀ x, f_prime x ≤ 0 ∧ ( ∃ x, f_prime x = 0 ∧ f x ≠ 0) ) ∧
  (  (f_prime 0 > 0 ∧ f_prime 2 > 0 ∧ ∀ x ∈ Icc (-∞ : Icc 0), f_prime x > 0 ∧ ∀ x ∈ Icc (2 : ℝ) (∞ : Icc), f_prime x > 0) ) ∧
  (  f 0 = 0 ∧ f 2 = -4 ) → 2
:= by  sorry

end count_true_propositions_l640_640200


namespace absolute_sum_of_coefficients_l640_640632

theorem absolute_sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) :
  (2 - x)^6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_0 = 2^6 →
  a_0 > 0 ∧ a_2 > 0 ∧ a_4 > 0 ∧ a_6 > 0 ∧
  a_1 < 0 ∧ a_3 < 0 ∧ a_5 < 0 → 
  |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 665 :=
by sorry

end absolute_sum_of_coefficients_l640_640632


namespace rectangular_equation_common_points_l640_640318

-- Define parametric equations of curve C
def x (t : ℝ) := (sqrt 3) * cos (2 * t)
def y (t : ℝ) := 2 * sin t

-- Define polar equation of line l
def polar_line (ρ θ m : ℝ) := ρ * sin (θ + π / 3) + m

-- Rectangular equation of l
theorem rectangular_equation (ρ θ x y m : ℝ) (h1 : ρ = sqrt (x^2 + y^2)) (h2 : θ = atan2 y x) :
  polar_line ρ θ m = 0 → (sqrt 3 * x + y + 2 * m = 0) :=
by 
  sorry

-- Range of values for m where line intersects curve
theorem common_points (h: ∃ t, sqrt 3 * cos (2 * t) = x ∧ 2 * sin t = y) :
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by 
  sorry

end rectangular_equation_common_points_l640_640318


namespace rectangular_equation_common_points_l640_640320

-- Define parametric equations of curve C
def x (t : ℝ) := (sqrt 3) * cos (2 * t)
def y (t : ℝ) := 2 * sin t

-- Define polar equation of line l
def polar_line (ρ θ m : ℝ) := ρ * sin (θ + π / 3) + m

-- Rectangular equation of l
theorem rectangular_equation (ρ θ x y m : ℝ) (h1 : ρ = sqrt (x^2 + y^2)) (h2 : θ = atan2 y x) :
  polar_line ρ θ m = 0 → (sqrt 3 * x + y + 2 * m = 0) :=
by 
  sorry

-- Range of values for m where line intersects curve
theorem common_points (h: ∃ t, sqrt 3 * cos (2 * t) = x ∧ 2 * sin t = y) :
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by 
  sorry

end rectangular_equation_common_points_l640_640320


namespace azure_valley_skirts_l640_640446

variables (P S A : ℕ)

theorem azure_valley_skirts (h1 : P = 10) 
                           (h2 : P = S / 4) 
                           (h3 : S = 2 * A / 3) : 
  A = 60 :=
by sorry

end azure_valley_skirts_l640_640446


namespace quadratic_inequality_solution_l640_640710

theorem quadratic_inequality_solution (a b c : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ x < -2 ∨ x > 4)
  (h₂ : a > 0) (h₃ : ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h₄ : ∀ x : ℝ, f (-1) = f 3)
  (h₅ : ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → x₁ < x₂ → f x₁ < f x₂) :
  f 2 < f (-1) ∧ f (-1) < f 5 :=
begin
  sorry,
end

end quadratic_inequality_solution_l640_640710


namespace polarCurves_areaMPQ_l640_640227

-- Definitions based on conditions
def curveC1 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9
def polarEqC1 (ρ θ : ℝ) : Prop := ρ = 6 * Real.sin θ
def pointA (ρ θ : ℝ) : Prop := ∀ x y, curveC1 x y → (ρ = Real.sqrt (x^2 + y^2) ∧ θ = Real.atan2 y x)
def pointB (ρ θ ρ0 θ0 : ℝ) (hA : pointA ρ0 θ0) : Prop := ρ = ρ0 ∧ θ = θ0 + Real.pi / 2 ∧ pointA ρ0 θ0
def polarEqC2 (ρ θ : ℝ) : Prop := ρ = -6 * Real.cos θ

-- Definition based on second part conditions
def rayIntersection (θ ρ : ℝ) : Prop := θ = 5 * Real.pi / 6 ∧ ρ > 0
def M : ℝ × ℝ := (-4, 0)

-- Proof problems
theorem polarCurves : 
  (∀ ρ θ, polarEqC1 ρ θ) ∧ 
  (∀ ρ θ, polarEqC2 ρ θ := sorry

theorem areaMPQ : 
  (∀ θ, rayIntersection θ 3 ∧ rayIntersection (θ-Real.pi/2) (3 * Real.sqrt 3)) →
  let d := 4 * Real.sin (5 * Real.pi / 6) in 
  (1 / 2) * (3 * Real.sqrt 3 - 3) * d = 3 * Real.sqrt 3 - 3 := sorry

end polarCurves_areaMPQ_l640_640227


namespace sum_first_five_odds_equals_25_smallest_in_cube_decomposition_eq_21_l640_640951

-- Problem 1: Define the sum of the first n odd numbers and prove it equals n^2 when n = 5.
theorem sum_first_five_odds_equals_25 : (1 + 3 + 5 + 7 + 9 = 5^2) := 
sorry

-- Problem 2: Prove that if the smallest number in the decomposition of m^3 is 21, then m = 5.
theorem smallest_in_cube_decomposition_eq_21 : 
  (∃ m : ℕ, m > 0 ∧ 21 = 2 * m - 1 ∧ m = 5) := 
sorry

end sum_first_five_odds_equals_25_smallest_in_cube_decomposition_eq_21_l640_640951


namespace most_cost_effective_payment_l640_640531

theorem most_cost_effective_payment :
  let worker_days := 5 * 10
  let hourly_rate_per_worker := 8 * 10 * 4
  let paint_cost := 4800
  let area_painted := 150
  let cost_option_1 := worker_days * 30
  let cost_option_2 := paint_cost * 0.30
  let cost_option_3 := area_painted * 12
  let cost_option_4 := 5 * hourly_rate_per_worker
  (cost_option_2 < cost_option_1) ∧ (cost_option_2 < cost_option_3) ∧ (cost_option_2 < cost_option_4) :=
by
  sorry

end most_cost_effective_payment_l640_640531


namespace trajectory_is_ellipse_l640_640214

-- Define the fixed points F1 and F2 with their coordinates.
structure Point := (x : ℝ) (y : ℝ)

def F1 : Point := ⟨0, -3⟩
def F2 : Point := ⟨0, 3⟩

-- Define a predicate to express the condition on the moving point P.
def satisfies_condition (P : Point) (m : ℝ) : Prop :=
  m > 0 ∧ (real.sqrt ((P.x - F1.x)^2 + (P.y - F1.y)^2) + real.sqrt ((P.x - F2.x)^2 + (P.y - F2.y)^2)) = m + 16 / m

-- Define the theorem stating that the trajectory of point P is an ellipse when it satisfies the condition.
theorem trajectory_is_ellipse (P : Point) (m : ℝ) (h : satisfies_condition P m) : 
  ∃ a b : ℝ, (a > 0 ∧ b > 0 ∧ (P.x / a)^2 + (P.y / b)^2 = 1) :=
sorry

end trajectory_is_ellipse_l640_640214


namespace median_of_first_twelve_positive_integers_l640_640511

theorem median_of_first_twelve_positive_integers : 
  let first_twelve_positive_integers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (first_twelve_positive_integers.length = 12) →
  (first_twelve_positive_integers.nth (first_twelve_positive_integers.length / 2 - 1) = some 6) →
  (first_twelve_positive_integers.nth (first_twelve_positive_integers.length / 2) = some 7) →
  (6 + 7) / 2 = 6.5 :=
by
  sorry

end median_of_first_twelve_positive_integers_l640_640511


namespace min_distance_between_curves_l640_640609

noncomputable def curve1 := λ x : ℝ, Real.exp (5 * x + 7)
noncomputable def curve2 := λ y : ℝ, (Real.log y - 7) / 5

theorem min_distance_between_curves :
  let x0 := (-(Real.log 5) - 7) / 5 in
  let y0 := Real.exp (5 * x0 + 7) in
  let distance := Real.sqrt 2 * (y0 - x0) in
  distance = Real.sqrt 2 * ((8 + Real.log 5) / 5) :=
by
  sorry

end min_distance_between_curves_l640_640609


namespace find_range_of_a_l640_640655

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * a * x + 4 > 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 4 - 2 * a > 0 ∧ 4 - 2 * a < 1

noncomputable def problem_statement (a : ℝ) : Prop :=
  let p := proposition_p a
  let q := proposition_q a
  (p ∨ q) ∧ ¬(p ∧ q)

theorem find_range_of_a (a : ℝ) :
  problem_statement a → -2 < a ∧ a ≤ 3/2 :=
sorry

end find_range_of_a_l640_640655


namespace mary_gives_one_cup_kibbles_l640_640429

theorem mary_gives_one_cup (x : ℕ) (h1 : 2 * x = 2) :
  5 - 7 - (2 * x + 3) = 12 :=
begin
  sorry
end

theorem kibbles : ∃ (x: ℕ), 2 * x = 2 :=
begin
  use 1,
  exact (sorry : 2 * 1 = 2),
end

end mary_gives_one_cup_kibbles_l640_640429


namespace john_strength_decrease_l640_640738

-- Conditions and variables defining John's peak lift amounts and strength decrease percentages
def peak_squat : ℕ := 700
def peak_bench : ℕ := 400
def peak_deadlift : ℕ := 800
def peak_overhead_press : ℕ := 500

def squat_loss_percentage : ℚ := 0.30
def deadlift_loss : ℕ := 200
def overhead_press_loss_percentage : ℚ := 0.15

-- New totals calculations
def new_squat : ℕ := peak_squat - (squat_loss_percentage * peak_squat).to_nat
def new_bench : ℕ := peak_bench
def new_deadlift : ℕ := peak_deadlift - deadlift_loss
def new_overhead_press : ℕ := peak_overhead_press - (overhead_press_loss_percentage * peak_overhead_press).to_nat

-- Calculating original and new total weight lifted
def total_peak_lifted : ℕ := peak_squat + peak_bench + peak_deadlift + peak_overhead_press
def total_new_lifted : ℕ := new_squat + new_bench + new_deadlift + new_overhead_press

-- Decrease in total weight lifted
def total_decrease : ℕ := total_peak_lifted - total_new_lifted

-- Overall percentage decrease calculation
def percentage_decrease : ℚ := (total_decrease.to_rat / total_peak_lifted.to_rat) * 100

-- Theorem to be proven
theorem john_strength_decrease :
  new_squat = 490 ∧ new_bench = 400 ∧ new_deadlift = 600 ∧ new_overhead_press = 425
  ∧ percentage_decrease ≈ 20.21 :=
by
  sorry

end john_strength_decrease_l640_640738


namespace angle_between_hands_l640_640593

theorem angle_between_hands (t : Time) : t = Time.mk 13 20 → angle_between_hour_and_minute_hands t = 80 :=
begin
  intro h,
  sorry
end

end angle_between_hands_l640_640593


namespace max_value_F_l640_640638

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x^2
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x

noncomputable def F (x : ℝ) : ℝ :=
if f x ≥ g x then f x else g x

theorem max_value_F : ∃ x : ℝ, ∀ y : ℝ, F y ≤ F x ∧ F x = 7 / 9 := 
sorry

end max_value_F_l640_640638


namespace has_incircle_symmetric_about_diagonal_l640_640788

variables (A B C D O : Type) (AB AO BO CO DO : ℝ)
variables (AB CD AD BC : ℝ) -- sides of quadrilateral
variables (s_aob r_aob s_cod r_cod s_boc r_boc s_doa r_doa : ℝ)

-- Assumptions
axiom perp_diagonals : AO * BO + CO * DO = 0 
axiom inradii_cond : r_aob + r_cod = r_boc + r_doa
axiom semi_perimeters_sum : s_aob + s_cod + s_boc + s_doa = s_abcd -- reformulation
axiom r_aob_def : r_aob = s_aob - AB
axiom r_cod_def : r_cod = s_cod - CD
axiom r_boc_def : r_boc = s_boc - BC
axiom r_doa_def : r_doa = s_doa - AD

-- Goal a: ABO + BCD = ABC + BOD
theorem has_incircle (h1 : AB + CD = AD + BC) : has_incircle A B C D := sorry

-- Goal b: Quadrilateral is symmetric about one of its diagonals
theorem symmetric_about_diagonal (h1 : AB + CD = AD + BC)
                                  (h2 : AB * CD = AD * BC) : 
                                  (AB = AD ∧ CD = BC) ∨ (AB = BC ∧ AD = CD) := sorry

end has_incircle_symmetric_about_diagonal_l640_640788


namespace grace_wait_time_l640_640999

variable (hose1_rate : ℕ) (hose2_rate : ℕ) (pool_capacity : ℕ) (time_after_second_hose : ℕ)
variable (h : ℕ)

theorem grace_wait_time 
  (h1 : hose1_rate = 50)
  (h2 : hose2_rate = 70)
  (h3 : pool_capacity = 390)
  (h4 : time_after_second_hose = 2) : 
  50 * h + (50 + 70) * 2 = 390 → h = 3 :=
by
  sorry

end grace_wait_time_l640_640999


namespace soccer_team_goals_l640_640887

theorem soccer_team_goals (total_players total_goals games_played : ℕ)
(one_third_players_goals : ℕ) :
  total_players = 24 →
  total_goals = 150 →
  games_played = 15 →
  one_third_players_goals = (total_players / 3) * games_played →
  (total_goals - one_third_players_goals) = 30 :=
by
  intros h1 h2 h3 h4
  rw h1 at h4
  rw h3 at h4
  sorry

end soccer_team_goals_l640_640887


namespace find_ratio_DE_EF_l640_640731

noncomputable def ratio_DE_EF (A B C D E F : Point) :=
  let AD_DB := 2 / 3
  let BE_EC := 4 / 1
  ∃ F : Point, ∃ DE : Segment, ∃ EF : Segment, 
    SegmentRatio AD_DB BE_EC F DE EF ∧
    Segment.intersection AC DE = F

theorem find_ratio_DE_EF (A B C D E F : Point)
  (h1 : D ∈ LineSegment A B)
  (h2 : E ∈ LineSegment B C)
  (h3 : ∃ r1 r2, r1 = 2 ∧ r2 = 3 ∧
              rSegmentRatio (LineSegment A D) 
              (LineSegment D B) = r1 / r2)
  (h4 : ∃ r3 r4, r3 = 4 ∧ r4 = 1 ∧
              rSegmentRatio (LineSegment B E) 
              (LineSegment E C) = r3 / r4)
  (h5 : ∃ G : Point, isIntersection F AC DE)
  : SegmentRatio D E E F = 1 / 4 := 
begin
  sorry
end

end find_ratio_DE_EF_l640_640731


namespace amount_of_water_to_add_l640_640042

theorem amount_of_water_to_add (m x : ℝ) (h₀ : m > 50) (h₁ : x ≥ 0)
    (initial_solution : m * (2 * m / 100))
    (final_solution : (m + x) * (m / 100)) :
(m * (2 * m / 100) = (m + x) * (m / 100)) → x = m :=
by 
    intros h
    sorry

end amount_of_water_to_add_l640_640042


namespace C_rent_share_is_1907_lcu_l640_640136

noncomputable def A_ox_months := 10 * 7
noncomputable def B_ox_months := 12 * 5
noncomputable def C_ox_months := 15 * 3
noncomputable def D_ox_months := 20 * 6
noncomputable def total_ox_months := A_ox_months + B_ox_months + C_ox_months + D_ox_months
noncomputable def total_rent_usd := 250
noncomputable def conversion_rate_lcu_per_usd := 50

noncomputable def C_share_lcu :=
  (C_ox_months / total_ox_months) * total_rent_usd * conversion_rate_lcu_per_usd

theorem C_rent_share_is_1907_lcu :
  C_share_lcu ≈ 1907 := by
    sorry

end C_rent_share_is_1907_lcu_l640_640136


namespace rect_eq_and_range_of_m_l640_640308

noncomputable def rect_eq_of_polar (m : ℝ) : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ * sin (θ + π / 3) + m = 0 → sqrt 3 * (ρ * cos θ) + (ρ * sin θ) + 2 * m = 0

noncomputable def range_of_m_for_intersection : set ℝ := 
  {m : ℝ | -19 / 12 ≤ m ∧ m ≤ 5 / 2}

theorem rect_eq_and_range_of_m (m : ℝ) : 
  rect_eq_of_polar m ∧ (∀ t : ℝ, ∃ x y : ℝ, x = sqrt 3 * cos (2 * t) ∧ y = 2 * sin t ∧ 
    (sqrt 3 * x + y + 2 * m = 0) → m ∈ range_of_m_for_intersection) :=
by sorry

end rect_eq_and_range_of_m_l640_640308


namespace count_non_carrying_pairs_in_range_l640_640198

-- Define the range of numbers from 1500 to 2500
def range_set : List ℕ := [1500..2500]

-- Define a function to check if the addition of two numbers causes no carry at any digit
def no_carry_pair (a b : ℕ) : Prop :=
  let digits := Nat.digits 10
  ∀ n, n < digits a.length →
  let x := digits a.get n
  let y := digits b.get n
  x + y < 10

-- Define a function to count all non-carrying pairs of consecutive integers in the given range
def count_non_carrying_pairs (s : List ℕ) : ℕ :=
  s.enum.filter (λ i, match i with
                     | (i, n) => n < (List.length s - 1) ∧ no_carry_pair n (n + 1))

theorem count_non_carrying_pairs_in_range :
  count_non_carrying_pairs range_set = 1980 :=
sorry

end count_non_carrying_pairs_in_range_l640_640198


namespace count_random_events_l640_640899

/-- Define what makes an event random -/
def is_random_event (description : String) : Prop :=
  match description with
  | "Throwing two dice twice in a row, and both times getting 2 points" => True
  | "On Earth, a pear falling from a tree will fall down if not caught" => False
  | "Someone winning the lottery" => True
  | "Having one daughter already, then having a boy the second time" => True
  | "Under standard atmospheric pressure, water heated to 90°C will boil" => False
  | _ => False

/-- Prove that the number of random events is exactly 3 -/
theorem count_random_events : 
  let conditions := [
    "Throwing two dice twice in a row, and both times getting 2 points",
    "On Earth, a pear falling from a tree will fall down if not caught",
    "Someone winning the lottery",
    "Having one daughter already, then having a boy the second time",
    "Under standard atmospheric pressure, water heated to 90°C will boil"
  ]
  in (conditions.filter is_random_event).length = 3 :=
by {
  sorry
}

end count_random_events_l640_640899


namespace find_measure_of_A_find_range_of_b_plus_c_l640_640978

noncomputable def conditions := ∀ (a b c : ℝ) (A B C : ℝ),
  a = sqrt 3 ∧
  A ∈ (0, π) ∧
  B ∈ (0, π) ∧
  C ∈ (0, π) ∧
  A + B + C = π ∧
  a / sin A = b / sin B ∧
  a / sin A = c / sin C ∧
  (sin B - sin A) / sin C = (b - c) / (a + b) ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A + B > C ∧ B + C > A ∧ C + A > B ∧ -- Triangle Inequality
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2    -- Acute Triangle

theorem find_measure_of_A : conditions → A = π / 3 :=
by { sorry }

theorem find_range_of_b_plus_c : conditions → b + c ∈ (3, 2 * sqrt 3] :=
by { sorry }

end find_measure_of_A_find_range_of_b_plus_c_l640_640978


namespace rectangular_equation_of_l_range_of_m_intersection_l640_640361

-- Definitions based on the problem conditions
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  let x := sqrt 3 * cos (2 * t)
  let y := 2 * sin t
  (x, y)

def polar_line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- The rectangular equation converted from polar form
theorem rectangular_equation_of_l (x y m : ℝ) :
  (polar_line_l (sqrt (x^2 + y^2)) (atan2 y x) m) ↔ (√3 * x + y + 2 * m = 0) :=
sorry

-- Range of values for m ensuring intersection of line l with curve C
theorem range_of_m_intersection (m : ℝ) :
  ((-19/12 : ℝ) ≤ m ∧ m ≤ (5/2 : ℝ)) ↔
  ∃ t : ℝ, let x := sqrt 3 * cos (2 * t)
           let y := 2 * sin t
           (sqrt 3 * x + y + 2 * m = 0 ∧ (-2 ≤ y ∧ y ≤ 2)) :=
sorry

end rectangular_equation_of_l_range_of_m_intersection_l640_640361


namespace sum_floors_l640_640390

theorem sum_floors (n : ℕ) (hn : n ≥ 2) (x : ℕ → ℝ)
  (hx : ∀ i, 1 ≤ i ∧ i ≤ n → (i : ℝ) ≤ x i ∧ x i < (i + 1 : ℝ)) :
  ∃ S, (0 ≤ S ∧ S ≤ n-1) ∧ S = (∑ i in Finset.range (n-1), ⌊ x (i+2) - x (i+1) ⌋) :=
sorry

end sum_floors_l640_640390


namespace remainder_division_x_squared_minus_one_l640_640223

variable (f g h : ℝ → ℝ)

noncomputable def remainder_when_divided_by_x_squared_minus_one (x : ℝ) : ℝ :=
-7 * x - 9

theorem remainder_division_x_squared_minus_one (h1 : ∀ x, f x = g x * (x - 1) + 8) (h2 : ∀ x, f x = h x * (x + 1) + 1) :
  ∀ x, f x % (x^2 - 1) = -7 * x - 9 :=
sorry

end remainder_division_x_squared_minus_one_l640_640223


namespace average_is_207_l640_640093

variable (x : ℕ)

theorem average_is_207 (h : (201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + 212 + x) / 10 = 207) :
  x = 212 :=
sorry

end average_is_207_l640_640093


namespace negation_statement_l640_640847

variable {α : Type} (S : Set α)

theorem negation_statement (P : α → Prop) :
  (∀ x ∈ S, ¬ P x) ↔ (∃ x ∈ S, P x) :=
by
  sorry

end negation_statement_l640_640847


namespace mary_remaining_money_l640_640430

variable (p : ℝ) -- p is the price per drink in dollars

def drinks_cost : ℝ := 3 * p
def medium_pizzas_cost : ℝ := 2 * (2 * p)
def large_pizza_cost : ℝ := 3 * p

def total_cost : ℝ := drinks_cost p + medium_pizzas_cost p + large_pizza_cost p

theorem mary_remaining_money : 
  30 - total_cost p = 30 - 10 * p := 
by
  sorry

end mary_remaining_money_l640_640430


namespace area_decrease_by_4_percent_l640_640833

variable (L W : ℝ)

def initial_area (L W : ℝ) := L * W
def new_length (L : ℝ) := 1.20 * L
def new_width (W : ℝ) := 0.80 * W
def new_area (L W : ℝ) := (new_length L) * (new_width W)
def percentage_change_in_area (A_initial A_new : ℝ) := ((A_new - A_initial) / A_initial) * 100

theorem area_decrease_by_4_percent (L W : ℝ) :
  percentage_change_in_area (initial_area L W) (new_area L W) = -4 := by
  sorry

end area_decrease_by_4_percent_l640_640833


namespace area_of_triangle_bounded_by_lines_l640_640068

def line1 (x : ℝ) : ℝ := 2 * x
def line2 (x : ℝ) : ℝ := -2 * x
def line3 (y : ℝ) : Prop := y = 8

theorem area_of_triangle_bounded_by_lines 
  (x1 x2 y1 y2 : ℝ)
  (h1 : line1 x1 = y2)
  (h2 : line2 x2 = y2)
  (h3 : line1 x1 = line3 y2)
  (h4 : line2 x2 = line3 y2)
  (h5 : line1 0 = line2 0) :
  let base := abs (x1 - x2)
  let height := abs y2
  let area := (1 / 2 : ℝ) * base * height
  area = 32 :=
by
  sorry

end area_of_triangle_bounded_by_lines_l640_640068


namespace rectangular_equation_of_l_range_of_m_intersection_l640_640366

-- Definitions based on the problem conditions
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  let x := sqrt 3 * cos (2 * t)
  let y := 2 * sin t
  (x, y)

def polar_line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- The rectangular equation converted from polar form
theorem rectangular_equation_of_l (x y m : ℝ) :
  (polar_line_l (sqrt (x^2 + y^2)) (atan2 y x) m) ↔ (√3 * x + y + 2 * m = 0) :=
sorry

-- Range of values for m ensuring intersection of line l with curve C
theorem range_of_m_intersection (m : ℝ) :
  ((-19/12 : ℝ) ≤ m ∧ m ≤ (5/2 : ℝ)) ↔
  ∃ t : ℝ, let x := sqrt 3 * cos (2 * t)
           let y := 2 * sin t
           (sqrt 3 * x + y + 2 * m = 0 ∧ (-2 ≤ y ∧ y ≤ 2)) :=
sorry

end rectangular_equation_of_l_range_of_m_intersection_l640_640366


namespace least_pos_int_k_l640_640744

def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2022}

def has_cardinality_three (s : Set ℕ) : Prop := s.card = 3

def subsets_of_M_with_card_three : Set (Set ℕ) := {s | s ⊆ M ∧ has_cardinality_three s}

def k (k : ℕ) : Prop :=
  ∀ (subsets : Finset (Set ℕ)), (subsets.card = k ∧ subsets ⊆ subsets_of_M_with_card_three) →
    ∃ (A B : Set ℕ), A ∈ subsets ∧ B ∈ subsets ∧ (A ≠ B ∧ A ∩ B ≠ ∅ ∧ (A ∩ B).card = 1)

theorem least_pos_int_k : ∃ k : ℕ, k = 2020 ∧ k 2020 :=
begin
  sorry
end

end least_pos_int_k_l640_640744


namespace probability_interval_l640_640277

noncomputable def normal (μ σ : ℝ) (hσ : σ > 0) : measure ℝ := sorry

theorem probability_interval (σ : ℝ) (hσ : σ > 0)
  (h : ⁇(normal 1 σ hσ) (set.Ioo 0 1) = 0.4) : ⁇(normal 1 σ hσ) (set.Ioo 0 2) = 0.8 :=
sorry

end probability_interval_l640_640277


namespace roots_of_polynomial_l640_640778

theorem roots_of_polynomial :
  (∃ x1 x2 x3 : ℝ, (x1 = -x2) ∧ (2 * x1^5 - x1^4 - 2 * x1^3 + x1^2 - 4 * x1 + 2 = 0)
    ∧ (2 * x2^5 - x2^4 - 2 * x2^3 + x2^2 - 4 * x2 + 2 = 0)
    ∧ (2 * x3^5 - x3^4 - 2 * x3^3 + x3^2 - 4 * x3 + 2 = 0)
    ∧ (x1 = sqrt 2 ∨ x1 = -sqrt 2)
    ∧ (x3 = 1/2)) := sorry

end roots_of_polynomial_l640_640778


namespace average_speed_of_train_l640_640134

theorem average_speed_of_train (x : ℝ) (h : x > 0) :
  let distanceA := x
      speedA := 75
      distanceB := 2 * x
      speedB := 25
      distanceC := 1.5 * x
      speedC := 50
      distanceD := x
      speedD := 60
      distanceE := 3 * x
      speedE := 40 * 0.9
      total_distance := distanceA + distanceB + distanceC + distanceD + distanceE
      timeA := distanceA / speedA
      timeB := distanceB / speedB
      timeC := distanceC / speedC
      timeD := distanceD / speedD
      timeE := distanceE / speedE
      total_time := timeA + timeB + timeC + timeD + timeE
      average_speed := total_distance / total_time
  in abs (average_speed - 33.582) < 0.001 :=
by
  sorry

end average_speed_of_train_l640_640134


namespace tan_alpha_second_quadrant_l640_640676

-- Define the conditions and prove the assertion

theorem tan_alpha_second_quadrant (
  α : ℝ,
  m : ℝ,
  h1 : m^2 + (sqrt 3 / 2)^2 = 1,
  h2 : m < 0
) : Real.tan α = -sqrt 3 :=
by
  /- Here we prove that the point (m, sqrt 3 / 2) lies on the unit circle and is in the second quadrant,
     then derive the value of tan α -/
  sorry

end tan_alpha_second_quadrant_l640_640676


namespace prove_market_demand_prove_tax_revenue_prove_per_unit_tax_rate_prove_tax_revenue_specified_l640_640098

noncomputable def market_supply_function (P : ℝ) : ℝ := 6 * P - 312

noncomputable def market_demand_function (a b P : ℝ) : ℝ := a - b * P

noncomputable def price_elasticity_supply (P_e Q_e : ℝ) : ℝ := 6 * (P_e / Q_e)

noncomputable def price_elasticity_demand (b P_e Q_e : ℝ) : ℝ := -b * (P_e / Q_e)

noncomputable def tax_rate := 30

noncomputable def consumer_price_after_tax := 118

theorem prove_market_demand (a P_e Q_e : ℝ) :
  1.5 * |price_elasticity_demand 4 P_e Q_e| = price_elasticity_supply P_e Q_e →
  market_demand_function a 4 P_e = a - 4 * P_e := sorry

theorem prove_tax_revenue (Q_d : ℝ) :
  Q_d = 216 →
  Q_d * tax_rate = 6480 := sorry

theorem prove_per_unit_tax_rate (t : ℝ) :
  t = 60 → 4 * t = 240 := sorry

theorem prove_tax_revenue_specified (t : ℝ) :
  t = 60 →
  (288 * t - 2.4 * t^2) = 8640 := sorry

end prove_market_demand_prove_tax_revenue_prove_per_unit_tax_rate_prove_tax_revenue_specified_l640_640098


namespace floor_euler_number_l640_640175

theorem floor_euler_number :
  ⌊real.exp 1⌋ = 2 :=
sorry

end floor_euler_number_l640_640175


namespace circle_center_and_radius_l640_640681

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 5 = 0

-- Statement of the center and radius of the circle
theorem circle_center_and_radius :
  (∀ x y : ℝ, circle_equation x y) →
  (∃ (h k r : ℝ), (∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ h = 3 ∧ k = 0 ∧ r = 2) :=
by
  sorry

end circle_center_and_radius_l640_640681


namespace find_12th_missing_integer_l640_640589

def sequence_an (n : ℕ) : ℕ := ⌊ n + Real.sqrt (2 * n) + 1/2 ⌋

-- Define a function to compute the missing integers in the sequence
def find_missing_integers (limit : ℕ) : List ℕ :=
  let s := List.range (limit + 1)
  let seq_vals := s.map sequence_an
  s.filter (λ k => k ∉ seq_vals)

theorem find_12th_missing_integer :
  (find_missing_integers 1000).nth 11 = some 21 :=
by
  sorry

end find_12th_missing_integer_l640_640589


namespace parallel_CD_EF_l640_640756

-- Definitions of points and circles
variables (Γ1 Γ2 : Type) [circle Γ1] [circle Γ2]
variables (A B C D E F X : Type) [point A] [point B] [point C] [point D] [point E] [point F] [point X]

-- Assumptions of the given conditions
axiom circles_intersect_at : Γ1.intersect Γ2 A B
axiom point_outside : X ∉ Γ1 ∧ X ∉ Γ2
axiom line_XA_intersects : (line X A).intersects Γ1 C ∧ (line X A).intersects Γ2 E
axiom line_XB_intersects : (line X B).intersects Γ1 D ∧ (line X B).intersects Γ2 F

-- The theorem to be proved:
theorem parallel_CD_EF : parallel (line C D) (line E F) :=
by {
  -- Detailed proof steps go here
  sorry
}

end parallel_CD_EF_l640_640756


namespace area_of_triangle_l640_640224

/-- Given point A(1, 2) on the circle O : x^2 + y^2 = 5, 
the area of the triangle formed by the tangent line passing through A and the coordinate axes is 25/4. -/
theorem area_of_triangle (A : ℝ × ℝ) (hA : A = (1, 2))
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 = 5 → (x, y) = A) :
  let line := (x + 2 * y - 5 = 0) in
  let x_intercept := 5 in
  let y_intercept := 5 / 2 in
  let area := (1 / 2) * x_intercept * y_intercept in
  area = 25 / 4 :=
by
  sorry

end area_of_triangle_l640_640224


namespace exists_circumcircle_with_three_points_and_no_other_points_inside_l640_640494

open Set

theorem exists_circumcircle_with_three_points_and_no_other_points_inside 
  (points : Set (EuclideanSpace ℝ (Fin 2))) 
  (h_card : points.card = 2003)
  (h_non_collinear : ∀ (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)), p1 ∈ points → p2 ∈ points → p3 ∈ points → (¬Collinear ℝ (λ _, (Fin 2).val) {p1, p2, p3})) :
  ∃ (circ : circle ℝ (EuclideanSpace ℝ (Fin 2))), 
    (∃ (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ {p1, p2, p3} ⊆ circ.points ∧ ∀ p ∈ points, p ∉ circ.interior) := 
by sorry

end exists_circumcircle_with_three_points_and_no_other_points_inside_l640_640494


namespace part_a_part_b_constant_determination_l640_640749

-- Given sequence definition
def sequence (u : ℕ → ℝ) := ∀ n : ℕ, u (n + 2) = u n - u (n + 1)

-- Part (a)
theorem part_a (u : ℕ → ℝ) (h : sequence u) : 
  ∃ α β a b : ℝ, a = (sqrt 5 - 1) / 2 ∧ b = (-sqrt 5 - 1) / 2 ∧ 
                  (∀ n, u n = α * a ^ n + β * b ^ n) :=
sorry

-- Part (b)
theorem part_b (u : ℕ → ℝ) (h : sequence u) (u_0 u_1 : ℝ)
  (S : ℕ → ℝ) (u_init : u 0 = u_0 ∧ u 1 = u_1 ∧ ∀ n, S n = (finset.range n).sum u) :
  ∃ k : ℝ, ∀ n, S n + u (n - 1) = k :=
sorry

-- Constant determination for part (b)
theorem constant_determination (u : ℕ → ℝ) (h : sequence u) (u_0 u_1 : ℝ)
  (u_init : u 0 = u_0 ∧ u 1 = u_1) (S : ℕ → ℝ)
  (sum_def : ∀ n, S n = (finset.range n).sum u) :
  ∃ k : ℝ, k = 2 * u_0 + u_1 ∧ ∀ n, S n + u (n - 1) = k :=
sorry

end part_a_part_b_constant_determination_l640_640749


namespace find_speed_of_first_train_l640_640067

noncomputable def relative_speed (length1 length2 : ℕ) (time_seconds : ℝ) : ℝ :=
  let total_length_km := (length1 + length2) / 1000
  let time_hours := time_seconds / 3600
  total_length_km / time_hours

theorem find_speed_of_first_train
  (length1 : ℕ)   -- Length of the first train in meters
  (length2 : ℕ)   -- Length of the second train in meters
  (speed2 : ℝ)    -- Speed of the second train in km/h
  (time_seconds : ℝ)  -- Time in seconds to be clear from each other
  (correct_speed1 : ℝ)  -- Correct speed of the first train in km/h
  (h_length1 : length1 = 160)
  (h_length2 : length2 = 280)
  (h_speed2 : speed2 = 30)
  (h_time_seconds : time_seconds = 21.998240140788738)
  (h_correct_speed1 : correct_speed1 = 41.98) :
  relative_speed length1 length2 time_seconds = speed2 + correct_speed1 :=
by
  sorry

end find_speed_of_first_train_l640_640067


namespace sum_of_primes_between_30_and_50_l640_640081

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- List of prime numbers between 30 and 50
def prime_numbers_between_30_and_50 : List ℕ := [31, 37, 41, 43, 47]

-- Sum of prime numbers between 30 and 50
def sum_prime_numbers_between_30_and_50 : ℕ :=
  prime_numbers_between_30_and_50.sum

-- Theorem: The sum of prime numbers between 30 and 50 is 199
theorem sum_of_primes_between_30_and_50 :
  sum_prime_numbers_between_30_and_50 = 199 := by
    sorry

end sum_of_primes_between_30_and_50_l640_640081


namespace general_term_of_sequence_l640_640647

theorem general_term_of_sequence (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = n^2 - 4 * n) : 
  a n = 2 * n - 5 :=
by
  -- Proof can be completed here
  sorry

end general_term_of_sequence_l640_640647


namespace carlos_gold_quarters_l640_640582

theorem carlos_gold_quarters 
  (face_value_per_quarter : ℝ)
  (melting_multiplier : ℝ)
  (weight_per_quarter_in_ounces : ℝ)
  (face_value_per_quarter = 0.25)
  (melting_multiplier = 80)
  (weight_per_quarter_in_ounces = 1/5)
: (melting_multiplier * face_value_per_quarter * (1 / weight_per_quarter_in_ounces)) = 100 := by
  sorry

end carlos_gold_quarters_l640_640582


namespace michael_lap_time_is_approx_41_point_14_l640_640597

def donovan_lap_time := 48

def michael_pass_laps := 6.000000000000002

noncomputable def michael_lap_time : ℝ :=
  donovan_lap_time * michael_pass_laps / (michael_pass_laps + 1)

theorem michael_lap_time_is_approx_41_point_14 :
  |michael_lap_time - 41.14285714285716| < 0.01 := by
  sorry

end michael_lap_time_is_approx_41_point_14_l640_640597


namespace median_of_first_twelve_positive_integers_l640_640516

theorem median_of_first_twelve_positive_integers :
  let S := (set.range 12).image (λ x, x + 1) in  -- Set of first twelve positive integers
  median S = 6.5 :=
by
  sorry

end median_of_first_twelve_positive_integers_l640_640516


namespace smallest_natural_number_l640_640939

open Nat

theorem smallest_natural_number (n : ℕ) :
  (n + 1) % 4 = 0 ∧ (n + 1) % 6 = 0 ∧ (n + 1) % 10 = 0 ∧ (n + 1) % 12 = 0 →
  n = 59 :=
by
  sorry

end smallest_natural_number_l640_640939


namespace problem_part1_problem_part2_l640_640304

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l_rect_eq (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (sqrt 3 * x + y + 2 * m = 0)

def line_l_range_m (m : ℝ) : Prop :=
  -19/12 ≤ m ∧ m ≤ 5/2

theorem problem_part1 (m : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * sin (θ + π / 3) + m = 0) ↔ line_l_rect_eq m := 
sorry

theorem problem_part2 :
  ∀ m : ℝ, 
  (∃ t : ℝ, line_l_rect_eq m (curve_C t).1 (curve_C t).2) ↔ line_l_range_m m := 
sorry

end problem_part1_problem_part2_l640_640304


namespace trajectory_of_N_is_ellipse_line_AB_l640_640213

/-- 
Let F1 be the point (-√2, 0) 
Let F2 be the circle (x - √2)² + y² = 16 
Let M be a moving point on the circle F2 
Let N be the point of intersection of the line NF1, which is perpendicular bisector of MF1, and the segment MF2 
Let P be the point (0, 1) 
-/
def point_F1 := (-Real.sqrt 2, 0)
def circle_F2 := { p | (p.1 - Real.sqrt 2)^2 + p.2^2 = 16 }

def M (p : ℝ × ℝ) : Prop := circle_F2 p

def N (M : ℝ × ℝ) : ℝ × ℝ :=
-- Intersection point calculation definition
sorry

def P := (0, 1)

theorem trajectory_of_N_is_ellipse :
    ∀ N, (N.1^2 / 4 + N.2^2 / 2 = 1) :=
sorry

def B_symm_about_y_axis (B : ℝ × ℝ) : ℝ × ℝ :=
(-B.1, B.2)

theorem line_AB'_passes_fixed_point_and_max_area {k : ℝ} (hk : k ≠ 0) (A B : ℝ × ℝ) 
    (hA : A.2 = k * A.1 + 1) (hB : B.2 = k * B.1 + 1) :
    let B' := B_symm_about_y_axis B in
    let AB' := λ x, ((A.2 - B'.2) / (A.1 + B'.1)) * (x - A.1) + A.2 in 
    (AB' 0 = 2 ∧ δeq ((1 / 2) * abs (2 * k / (1 + 2 * k^2))) (Real.sqrt 2 / 2)) :=
sorry

end trajectory_of_N_is_ellipse_line_AB_l640_640213


namespace percentage_of_knives_l640_640915

def initial_knives : Nat := 6
def initial_forks : Nat := 12
def initial_spoons : Nat := 3 * initial_knives
def traded_knives : Nat := 10
def traded_spoons : Nat := 6

theorem percentage_of_knives :
  100 * (initial_knives + traded_knives) / (initial_knives + initial_forks + initial_spoons - traded_spoons + traded_knives) = 40 := by
  sorry

end percentage_of_knives_l640_640915


namespace rectangular_eq_of_line_l_range_of_m_l640_640349

noncomputable def curve_C_param_eq (t : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l_polar_eq (θ ρ m : ℝ) : Prop :=
  ρ * Real.sin(θ + Real.pi / 3) + m = 0

theorem rectangular_eq_of_line_l (x y m : ℝ) (h : line_l_polar_eq (Real.atan2 y x) (Real.sqrt (x^2 + y^2)) m) :
  sqrt 3 * x + y + 2 * m = 0 :=
  sorry

theorem range_of_m (m : ℝ) (t : ℝ) (h : ∃ (t : ℝ), curve_C_param_eq t ∈ set_of (λ p, p.2 = -√3 * p.1 - 2 * m))
  : -19/12 ≤ m ∧ m ≤ 5/2 :=
  sorry

end rectangular_eq_of_line_l_range_of_m_l640_640349


namespace automobile_distance_in_yards_l640_640572

theorem automobile_distance_in_yards
  (b t : ℝ) (h : t ≠ 0) : 
  let rate := (b / 4) / t,
      distance_in_2_hours := rate * 2,
      distance_in_yards := distance_in_2_hours * 1760 in
  distance_in_yards = 880 * b / t :=
by {
  let rate := (b / 4) / t,
  let distance_in_2_hours := rate * 2,
  let distance_in_yards := distance_in_2_hours * 1760,
  show distance_in_yards = 880 * b / t,
  -- Proof is omitted, as stated in the instructions
  sorry
}

end automobile_distance_in_yards_l640_640572


namespace tshirt_costs_more_than_jersey_l640_640465

-- Definitions based on the conditions
def cost_of_tshirt : ℕ := 192
def cost_of_jersey : ℕ := 34

-- Theorem statement
theorem tshirt_costs_more_than_jersey : cost_of_tshirt - cost_of_jersey = 158 := by
  sorry

end tshirt_costs_more_than_jersey_l640_640465


namespace min_n_exists_20_subsets_l640_640761

open Set

-- Definitions corresponding to the conditions
def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def B (n : ℕ) : Set ℕ := { x | 7 ≤ x ∧ x ≤ n }

-- The main theorem statement
theorem min_n_exists_20_subsets (n : ℕ) (Ai : Fin 20 → Set ℕ)
  (hAi1 : ∀ i, Ai i ⊆ A ∪ B n)
  (hAi2 : ∀ i, (Ai i) ∩ A ∈ powerset A)
  (hAi3 : ∀ i, (Ai i).card = 5)
  (hAi4 : ∀ i, (Ai i).card + (Ai i ∩ A).card = 3 + 2)
  (hAi5 : ∀ i j, 1 ≤ i → i < j → j ≤ 20 → (Ai i ∩ Ai j).card ≤ 2):

  ∃ n ≥ 16, ∀ i, i < 20 → Ai i ⊆ A ∪ B n := 
begin
  sorry
end

end min_n_exists_20_subsets_l640_640761


namespace distance_focus_to_asymptote_l640_640790

open Real

def parabola : set (ℝ × ℝ) := {p | p.2 ^ 2 = 8 * p.1}

def hyperbola : set (ℝ × ℝ) := {p | p.1 ^ 2 - p.2 ^ 2 / 3 = 1}

def focus : ℝ × ℝ := (2, 0)

def asymptote1 (x y : ℝ) : Prop := x + (sqrt 3 / 3) * y = 0
def asymptote2 (x y : ℝ) : Prop := x - (sqrt 3 / 3) * y = 0

theorem distance_focus_to_asymptote : 
  min (abs (2 / sqrt (1 + (sqrt 3 / 3) ^ 2))) (abs (2 / sqrt (1 + (sqrt 3 / 3) ^ 2))) = sqrt 3 :=
by
  sorry

end distance_focus_to_asymptote_l640_640790


namespace total_jelly_beans_l640_640427

theorem total_jelly_beans (vanilla_jelly_beans : ℕ) (h1 : vanilla_jelly_beans = 120) 
                           (grape_jelly_beans : ℕ) (h2 : grape_jelly_beans = 5 * vanilla_jelly_beans + 50) :
                           vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  sorry

end total_jelly_beans_l640_640427


namespace smallest_b_l640_640071

theorem smallest_b (b: ℕ) (h1: b > 3) (h2: ∃ n: ℕ, n^3 = 2 * b + 3) : b = 12 :=
sorry

end smallest_b_l640_640071


namespace derivative_y_l640_640207

noncomputable def y (x : ℝ) : ℝ := exp (3 * x) - cos x

theorem derivative_y (x : ℝ) : deriv y x = (3 * exp (3 * x) + sin x * exp (3 * x) - sin x) :=
by sorry

end derivative_y_l640_640207


namespace positional_relationship_l640_640831

-- Definitions of the lines l1 and l2
def l1 (m x y : ℝ) : Prop := (m + 3) * x + 5 * y = 5 - 3 * m
def l2 (m x y : ℝ) : Prop := 2 * x + (m + 6) * y = 8

theorem positional_relationship (m : ℝ) :
  (∃ x y : ℝ, l1 m x y ∧ l2 m x y) ∨ (∀ x y : ℝ, l1 m x y ↔ l2 m x y) ∨
  ¬(∃ x y : ℝ, l1 m x y ∨ l2 m x y) :=
sorry

end positional_relationship_l640_640831


namespace hockey_cards_count_l640_640865

-- Define integer variables for the number of hockey, football and baseball cards
variables (H F B : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := F = 4 * H
def condition2 : Prop := B = F - 50
def condition3 : Prop := H > 0
def condition4 : Prop := H + F + B = 1750

-- The theorem to prove
theorem hockey_cards_count 
  (h1 : condition1 H F)
  (h2 : condition2 F B)
  (h3 : condition3 H)
  (h4 : condition4 H F B) : 
  H = 200 := by
sorry

end hockey_cards_count_l640_640865


namespace probability_complete_collection_l640_640045

def combination (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem probability_complete_collection : 
  combination 6 6 * combination 12 4 / combination 18 10 = 5 / 442 :=
by
  have h1 : combination 6 6 = 1 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h2 : combination 12 4 = 495 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h3 : combination 18 10 = 43758 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  rw [h1, h2, h3]
  norm_num
  sorry

end probability_complete_collection_l640_640045


namespace distinct_students_AT_Pythagoras_Academy_l640_640147

theorem distinct_students_AT_Pythagoras_Academy Taking_AMC8 :
  let students_Archimedes := 15
  let students_Euler := 10
  let students_Gauss := 12
  let double_counted_students := 3 in
  students_Archimedes + students_Euler + students_Gauss - double_counted_students = 34 :=
  by {
    let students_Archimedes := 15
    let students_Euler := 10
    let students_Gauss := 12
    let double_counted_students := 3
    exact eq.refl _
  }

end distinct_students_AT_Pythagoras_Academy_l640_640147


namespace unique_arrangement_l640_640953

def valid_grid (arrangement : Matrix (Fin 4) (Fin 4) Char) : Prop :=
  (∀ i : Fin 4, (∃ j1 j2 j3 : Fin 4,
    j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3 ∧
    arrangement i j1 = 'A' ∧
    arrangement i j2 = 'B' ∧
    arrangement i j3 = 'C')) ∧
  (∀ j : Fin 4, (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 j = 'A' ∧
    arrangement i2 j = 'B' ∧
    arrangement i3 j = 'C')) ∧
  (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 i1 = 'A' ∧
    arrangement i2 i2 = 'B' ∧
    arrangement i3 i3 = 'C') ∧
  (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 (Fin.mk (3 - i1.val) sorry) = 'A' ∧
    arrangement i2 (Fin.mk (3 - i2.val) sorry) = 'B' ∧
    arrangement i3 (Fin.mk (3 - i3.val) sorry) = 'C')

def fixed_upper_left (arrangement : Matrix (Fin 4) (Fin 4) Char) : Prop :=
  arrangement 0 0 = 'A'

theorem unique_arrangement : ∃! arrangement : Matrix (Fin 4) (Fin 4) Char,
  valid_grid arrangement ∧ fixed_upper_left arrangement :=
sorry

end unique_arrangement_l640_640953


namespace smallest_base_10_integer_l640_640165

theorem smallest_base_10_integer (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b) (h₃ : 2 * a - 3 * b = -1) : 2 * a + 3 = 3 * b + 2 :=
by
  sorry

end smallest_base_10_integer_l640_640165


namespace red_tetrahedron_exists_l640_640966

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem red_tetrahedron_exists
  {X : Type*} [HasCardinality X n]
  (n_gt_4 : n > 4)
  (no_four_coplanar : ∀ x1 x2 x3 x4 : X, ¬Plane x1 x2 x3 x4)
  (red_triangles : Finset (Finset X))
  (m : ℕ)
  (hM : m > (2 * n - 3) / 9 * binom n 2)
  (red_triangle_count : red_triangles.card = m) :
  ∃ (t : Finset (Finset X)), t.card = 4 ∧ all_faces_red t red_triangles := 
sorry

end red_tetrahedron_exists_l640_640966


namespace length_of_bridge_l640_640892

noncomputable def speed_in_mps (speed_in_kmph : ℝ) : ℝ :=
  speed_in_kmph * (1000 / 3600)

noncomputable def total_distance_covered (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def bridge_length (total_distance : ℝ) (train_length : ℝ) : ℝ :=
  total_distance - train_length

theorem length_of_bridge :
  let train_length := 165,
      train_speed_in_kmph := 90,
      time_to_cross_bridge := 32.99736021118311,
      train_speed_in_mps := speed_in_mps train_speed_in_kmph,
      total_distance := total_distance_covered train_speed_in_mps time_to_cross_bridge
  in abs ((bridge_length total_distance train_length) - 659.9340052795778) < 0.0001 :=
by
  let train_length := 165
  let train_speed_in_kmph := 90
  let time_to_cross_bridge := 32.99736021118311
  let train_speed_in_mps := speed_in_mps train_speed_in_kmph
  let total_distance := total_distance_covered train_speed_in_mps time_to_cross_bridge
  let bridge := bridge_length total_distance train_length
  have h : abs (bridge - 659.9340052795778) < 0.0001 := sorry
  exact h

end length_of_bridge_l640_640892


namespace count_positive_bases_for_log_1024_l640_640253

-- Define the conditions 
def is_positive_integer_log_base (b n : ℕ) : Prop := b^n = 1024 ∧ n > 0

-- State that there are exactly 4 positive integers b that satisfy the condition
theorem count_positive_bases_for_log_1024 :
  (∃ b1 b2 b3 b4 : ℕ, b1 ≠ b2 ∧ b1 ≠ b3 ∧ b1 ≠ b4 ∧ b2 ≠ b3 ∧ b2 ≠ b4 ∧ b3 ≠ b4 ∧
    (∀ b, is_positive_integer_log_base b 1 ∨ is_positive_integer_log_base b 2 ∨ is_positive_integer_log_base b 5 ∨ is_positive_integer_log_base b 10) ∧
    (is_positive_integer_log_base b1 1 ∨ is_positive_integer_log_base b1 2 ∨ is_positive_integer_log_base b1 5 ∨ is_positive_integer_log_base b1 10) ∧
    (is_positive_integer_log_base b2 1 ∨ is_positive_integer_log_base b2 2 ∨ is_positive_integer_log_base b2 5 ∨ is_positive_integer_log_base b2 10) ∧
    (is_positive_integer_log_base b3 1 ∨ is_positive_integer_log_base b3 2 ∨ is_positive_integer_log_base b3 5 ∨ is_positive_integer_log_base b3 10) ∧
    (is_positive_integer_log_base b4 1 ∨ is_positive_integer_log_base b4 2 ∨ is_positive_integer_log_base b4 5 ∨ is_positive_integer_log_base b4 10)) :=
sorry

end count_positive_bases_for_log_1024_l640_640253


namespace jack_and_jill_passing_distance_l640_640381

theorem jack_and_jill_passing_distance :
  ∀ (t_jack_start : ℝ)
    (speed_jack_uphill speed_jack_downhill : ℝ)
    (speed_jill_uphill speed_jill_downhill : ℝ),
    t_jack_start = (1 / 6) ∧
    speed_jack_uphill = 15 ∧
    speed_jack_downhill = 20 ∧
    speed_jill_uphill = 16 ∧
    speed_jill_downhill = 22 →
  let time_jack_top := 5 / speed_jack_uphill,
      time_jill_top := 5 / speed_jill_uphill in
  let expr1 := 5 - speed_jack_downhill * (time_jack_top + t_jack_start),
      expr2 := speed_jill_uphill * (time_jill_top - t_jack_start) in
  (5 - (36/108) * expr1) = (5 - expr2) → 
  (5 - (100 / 27)) = (35 / 27) := sorry

end jack_and_jill_passing_distance_l640_640381


namespace num_same_family_functions_l640_640698

-- Conditions
def y_eq_x_squared : ℝ → ℝ := λ x, x^2
def original_domain : Set ℝ := {-3, 3}
def same_family (f g : ℝ → ℝ) (dom_f dom_g : Set ℝ) : Prop :=
  (∀ x ∈ dom_f, f x = y_eq_x_squared x) ∧
  (∀ x' ∈ dom_g, g x' = y_eq_x_squared x') ∧
  (Set.image f dom_f = Set.image y_eq_x_squared original_domain) ∧ 
  (dom_f ≠ original_domain) ∧
  (dom_f ⊆ original_domain)

-- To prove
theorem num_same_family_functions : ∃ n, n = 2 :=
by
  have h1 : same_family (λ x, (if x = -3 then x else -3)^2) y_eq_x_squared {-3} original_domain,
  have h2 : same_family (λ x, (if x = 3 then x else 3)^2) y_eq_x_squared {3} original_domain,
  use 2,
  sorry

end num_same_family_functions_l640_640698


namespace space_diagonal_length_of_cube_l640_640287

-- Define the conditions as provided in the problem statement
def base_diagonal (a : ℝ) : ℝ := a * Real.sqrt 2
def space_diagonal (a : ℝ) : ℝ := a * Real.sqrt 3

-- State the main theorem to be proved
theorem space_diagonal_length_of_cube (h : base_diagonal a = 5) : space_diagonal a = 5 * Real.sqrt 6 / 2 := 
by 
-- We are not providing the proof, just indicating the theorem statement based on given conditions.
sorry

end space_diagonal_length_of_cube_l640_640287


namespace crease_length_of_folded_isosceles_triangle_l640_640145

/-- Given an isosceles triangle with sides 5, 5, and 6 inches, 
    and folding the vertex opposite the 6 inch side to the midpoint of that side, 
    the length of the resulting crease is sqrt(34) inches. -/
theorem crease_length_of_folded_isosceles_triangle :
  ∀ (A B C M : Type) (AB AC : ℝ), -- Define types for points and lengths
    is_isosceles_triangle A B C 5 5 6 ∧ -- Triangle ABC is isosceles with AB = AC = 5, BC = 6
    midpoint B C M ∧ -- M is the midpoint of BC
    folded_vertices_to_midpoint A M -> -- Vertex A is folded to M
    distance A M = real.sqrt 34 :=  -- Length of crease AM is sqrt(34)
begin
  sorry
end

end crease_length_of_folded_isosceles_triangle_l640_640145


namespace number_of_intersections_l640_640185

def line_parametric (t : ℝ) : ℝ × ℝ :=
  (t - 1, 2 - t)

def curve_parametric (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ, 2 * Real.sin θ)

def line_general (x y : ℝ) : Prop :=
  x + y - 1 = 0

def curve_general (x y : ℝ) : Prop :=
  (x^2) / 9 + (y^2) / 4 = 1

theorem number_of_intersections : (∃ x y, line_general x y ∧ curve_general x y) → 2 :=
sorry

end number_of_intersections_l640_640185


namespace union_and_complement_intersection_find_a_l640_640861

-- Define the sets using their descriptions
def A_3_7 : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B_2_10 : Set ℝ := { x | 2 < x ∧ x < 10 }

-- Statement for union and intersection complement
theorem union_and_complement_intersection :
  (A_3_7 ∪ B_2_10 = {x | 2 < x ∧ x < 10}) ∧
  ((A_3_7ᶜ ∩ B_2_10) = ({x | 2 < x ∧ x < 3} ∪ {x | 7 ≤ x ∧ x < 10})) :=
by
  sorry

-- Define sets A, B, and C using their quadratic descriptions
def A : ℝ → Set ℝ := λ a, { x | x^2 - a*x + a^2 - 19 = 0 }
def B_5_6 : Set ℝ := { x | x^2 - 5*x + 6 = 0 }
def C_2_8 : Set ℝ := { x | x^2 + 2*x - 8 = 0 }

-- Validating the specific requirement on a
theorem find_a (a : ℝ) :
  ( ∃ x, x ∈ A a ∧ x ∈ B_5_6 ) ∧
  ( ∀ x, x ∈ A a → x ∉ C_2_8 ) →
  a = -2 :=
by
  sorry


end union_and_complement_intersection_find_a_l640_640861


namespace find_third_side_l640_640882

def vol_of_cube (side : ℝ) : ℝ := side ^ 3

def vol_of_box (length width height : ℝ) : ℝ := length * width * height

theorem find_third_side (n : ℝ) (vol_cube : ℝ) (num_cubes : ℝ) (l w : ℝ) (vol_box : ℝ) :
  num_cubes = 24 →
  vol_cube = 27 →
  l = 8 →
  w = 12 →
  vol_box = num_cubes * vol_cube →
  vol_box = vol_of_box l w n →
  n = 6.75 :=
by
  intros hcubes hc_vol hl hw hvbox1 hvbox2
  -- The proof goes here
  sorry

end find_third_side_l640_640882


namespace determine_omega_l640_640703

noncomputable def f (ω x : ℝ) : ℝ :=
  2 * Real.sin (ω * x)

theorem determine_omega (ω : ℝ) :
  (0 < ω ∧ ω < 1) →
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 3 → f ω x ≤ sqrt 2) →
  (∃! ω, f ω (Real.pi / 3) = sqrt 2) →
  ω = 3 / 4 :=
by
  intros hω hfx hmax
  sorry

end determine_omega_l640_640703


namespace probability_at_least_one_blown_l640_640435

theorem probability_at_least_one_blown (P_A P_B P_AB : ℝ)  
  (hP_A : P_A = 0.085) 
  (hP_B : P_B = 0.074) 
  (hP_AB : P_AB = 0.063) : 
  P_A + P_B - P_AB = 0.096 :=
by
  sorry

end probability_at_least_one_blown_l640_640435


namespace midpoint_correct_l640_640070

-- Define the points (10, -8) and (-5, 6)
def P1  := (10 : ℝ, -8 : ℝ)
def P2  := (-5 : ℝ, 6 : ℝ)

-- Define the midpoint formula for two points in 2D space
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the expected midpoint
def expectedMidpoint := (2.5 : ℝ, -1 : ℝ)

-- The theorem to be proven: the midpoint of P1 and P2 is expectedMidpoint
theorem midpoint_correct : midpoint P1 P2 = expectedMidpoint :=
by
  -- This is how you would begin the proof; the actual steps are not needed
  sorry

end midpoint_correct_l640_640070


namespace regular_decagon_side_length_l640_640776

noncomputable def cos_18 : ℝ := (Real.sqrt 5 + 1) / 4

theorem regular_decagon_side_length (r x : ℝ)
  (h1: ∠OAB = 36)
  (h2: x^2 + r * x - r^2 = 0)
  (h3: cos_18 = (Real.sqrt 5 + 1) / 4) :
  x = r / 2 * (Real.sqrt 5 - 1) := 
sorry

end regular_decagon_side_length_l640_640776


namespace minimal_perimeter_triangle_l640_640601

theorem minimal_perimeter_triangle
  (A B C O D E F : Type*)
  [triangle ABC] 
  [ortho_center O ABC]
  (def_height_A : height A D)
  (def_height_B : height B E)
  (def_height_C : height C F)
  (perpendicular_A : perpendicular D O A)
  (perpendicular_B : perpendicular E O B)
  (perpendicular_C : perpendicular F O C) :
  (∀ DEF∈ triangles_inscribed_in ABC, perimeter DEF ≥ perimeter D E F) := 
sorry

end minimal_perimeter_triangle_l640_640601


namespace solve_for_f_e_l640_640264

noncomputable def f (x : ℝ) := 2 * (f' 1) * Real.log x + x
noncomputable def f' (x : ℝ) := (∂ / ∂ x)[f x]

theorem solve_for_f_e (h : f' 1 = -1) : f (Real.exp 1) = -2 + Real.exp 1 := by
  sorry

end solve_for_f_e_l640_640264


namespace decrease_percent_revenue_l640_640539

theorem decrease_percent_revenue (T C : ℝ) :
  let original_revenue := T * C in
  let new_tax := 0.68 * T in
  let new_consumption := 1.12 * C in
  let new_revenue := new_tax * new_consumption in
  let decrease := original_revenue - new_revenue in
  let decrease_percent := (decrease / original_revenue) * 100 in
  decrease_percent = 23.84 :=
by
  sorry

end decrease_percent_revenue_l640_640539


namespace percentage_of_remainder_left_village_l640_640544

theorem percentage_of_remainder_left_village (initial_population current_population : ℕ)
  (died_percentage : ℝ)
  (died_population rounded_died_population left_population : ℤ)
  (percent_left : ℝ) :
  initial_population = 3161 →
  died_percentage = 0.05 →
  died_population = ⌊0.05 * 3161⌋ →
  rounded_died_population = 158 →
  current_population = 2553 →
  left_population = 3161 - 158 - 2553 →
  percent_left = ((3161 - 158 - 2553 : ℤ) : ℝ) / (3161 - 158 : ℤ) * 100 →
  percent_left ≈ 14.99 :=
by
  sorry

end percentage_of_remainder_left_village_l640_640544


namespace rect_eq_line_range_of_m_l640_640327

-- Definitions based on the conditions
def parametric_x (t : ℝ) : ℝ := sqrt 3 * cos (2 * t)
def parametric_y (t : ℝ) : ℝ := 2 * sin t

def polar_rho (θ m : ℝ) : ℝ := -(m / (sin (θ + π / 3)))

-- Rectangular equation of the line l
theorem rect_eq_line (x y m : ℝ) : polar_rho (atan2 y x) m * sin (atan2 y x + π / 3) + m = 0 → sqrt 3 * x + y + 2 * m = 0 :=
by
  sorry

-- Range of values for m
theorem range_of_m (C : ℝ → ℝ × ℝ)
  (hC : ∀ t, C t = (parametric_x t, parametric_y t)) (m : ℝ) :
  (∃ t, sqrt 3 * fst (C t) + snd (C t) + 2 * m = 0) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
by
  sorry

end rect_eq_line_range_of_m_l640_640327


namespace total_toothpicks_correct_l640_640822

-- Define the number of vertical lines and toothpicks in them
def num_vertical_lines : ℕ := 41
def num_toothpicks_per_vertical_line : ℕ := 20
def vertical_toothpicks : ℕ := num_vertical_lines * num_toothpicks_per_vertical_line

-- Define the number of horizontal lines and toothpicks in them
def num_horizontal_lines : ℕ := 21
def num_toothpicks_per_horizontal_line : ℕ := 40
def horizontal_toothpicks : ℕ := num_horizontal_lines * num_toothpicks_per_horizontal_line

-- Define the dimensions of the triangle
def triangle_base : ℕ := 20
def triangle_height : ℕ := 20
def triangle_hypotenuse : ℕ := 29 -- approximated

-- Total toothpicks in the triangle
def triangle_toothpicks : ℕ := triangle_height + triangle_hypotenuse

-- Total toothpicks used in the structure
def total_toothpicks : ℕ := vertical_toothpicks + horizontal_toothpicks + triangle_toothpicks

-- Theorem to prove the total number of toothpicks used is 1709
theorem total_toothpicks_correct : total_toothpicks = 1709 := by
  sorry

end total_toothpicks_correct_l640_640822


namespace circumcircle_trisection_point_l640_640723

theorem circumcircle_trisection_point (A B C D : Point) :
  parallelogram A B C D →
  2 * (BD.length)^2 = (BA.length)^2 + (BC.length)^2 →
  ∃ H : Point, H = 1 / 3 • (C - A) + A ∧ H ∈ circumcircle B C D := sorry

end circumcircle_trisection_point_l640_640723


namespace combined_contingency_funds_l640_640708

theorem combined_contingency_funds 
  (donation1 : ℕ := 360)
  (distribution1 : ℕ → ℕ := λ x, if x = 1 then 35 else if x = 2 then 40 else if x = 3 then 10 else if x = 4 then 5 else 10)
  (donation2 : ℕ := 180)
  (conversion_rate : ℚ := 1.20)
  (distribution2 : ℕ → ℕ := λ x, if x = 1 then 30 else if x = 2 then 25 else if x = 3 then 25 else 20) :
  (contingency_funds : ℚ) :=
  let contingency1 := donation1 * (distribution1 5) / 100 in
  let total2 := donation2 * conversion_rate in
  let contingency2 := total2 * (distribution2 4) / 100 in
  contingency1 + contingency2 = 79.20

end combined_contingency_funds_l640_640708


namespace ratio_AH_HD_l640_640730

-- Defining the triangle ABC with given sides and angle
variables {A B C H D : Type}
variables (BC AC : ℝ) (angle_C : ℝ) (is_orthocenter : H exists)
variables (AD BE CF : ℝ)

-- Given conditions
# In triangle ABC
axiom h_BC : BC = 6
axiom h_AC : AC = 6 * Real.sqrt 3
axiom h_angle_C : angle_C = 30

-- Altitudes are intersecting at the orthocenter H
axiom h_orthocenter : is_orthocenter = true

-- Proving ratio AH:HD
theorem ratio_AH_HD (AD_value : AD = 12 - 6 * Real.sqrt 3) (AH : ℝ) (HD : ℝ) : AH / HD = 2 :=
sorry

end ratio_AH_HD_l640_640730


namespace girls_in_class_l640_640034

theorem girls_in_class (B G : ℕ) 
  (h1 : G = B + 3) 
  (h2 : G + B = 41) : 
  G = 22 := 
sorry

end girls_in_class_l640_640034


namespace fraction_sum_to_decimal_l640_640911

theorem fraction_sum_to_decimal :
  (3 / 20 : ℝ) + (5 / 200 : ℝ) + (7 / 2000 : ℝ) = 0.1785 :=
by 
  sorry

end fraction_sum_to_decimal_l640_640911


namespace verify_fraction_property_verify_angle_l640_640917

noncomputable def collinear_points (B C D : Point) : Prop :=
  collinear B C D ∧ between C B D

noncomputable def point_not_on_line (A B D : Point) : Prop :=
  ¬ collinear A B D

noncomputable def distances_eq (A B C D : Point) : Prop :=
  dist A B = dist A C ∧ dist A C = dist C D

theorem verify_fraction_property (A B C D : Point) :
  collinear_points B C D →
  point_not_on_line A B D →
  distances_eq A B C D →
  angle A B C = 36 :=
  ∀ (A B C D : Point), (dist C D)⁻¹ - (dist B D)⁻¹ = (dist C D + dist B D)⁻¹ := sorry

theorem verify_angle (A B C D : Point) :
  collinear_points B C D →
  point_not_on_line A B D →
  distances_eq A B C D →
  (dist C D)⁻¹ - (dist B D)⁻¹ = (dist C D + dist B D)⁻¹ →
  angle A B C = 36 := sorry

end verify_fraction_property_verify_angle_l640_640917


namespace frank_steps_forward_l640_640624

theorem frank_steps_forward :
  let initial_pos := 0 in
  let final_pos := initial_pos - 5 + 10 - 2 + 2 * 2 in
  final_pos = 7 :=
by
  sorry

end frank_steps_forward_l640_640624


namespace line_through_point_circle_l640_640012

theorem line_through_point_circle {k : ℝ} :
    let P := (1, 1)
    let circ_eq := ∀ (x y : ℝ), x^2 + y^2 - 4 * x - 4 * y + 4 = 0
    let chord_length := 2 * real.sqrt 2
    ∃ (l : ℝ × ℝ → Prop), 
      (l P.1 P.2 = 0) ∧
      (∃ x y, circ_eq x y ∧ l x y = 0) ∧
      (∃ d : ℝ, d = real.sqrt ((2 - 0)^2 - (0.5 * chord_length)^2) ∧ d = real.sqrt 2) ∧
      ∀ x y, l x y = 0 ↔ x + y - 2 = 0 := 
by
  sorry

end line_through_point_circle_l640_640012


namespace simplify_sqrt_27_minus_sqrt_3_simplify_expr_l640_640907

-- Problem 1
theorem simplify_sqrt_27_minus_sqrt_3 : 
  sqrt 27 - sqrt 3 = 2 * sqrt 3 := by
  sorry

-- Problem 2
theorem simplify_expr : 
  (sqrt 10 - sqrt 2)^2 + (-1)^0 = 13 - 4 * sqrt 5 := by
  sorry

end simplify_sqrt_27_minus_sqrt_3_simplify_expr_l640_640907


namespace maximal_volume_of_solid_l640_640734

noncomputable def AC_length (P : ℝ) := (3/8) * P
noncomputable def BC_length (P : ℝ) := (1/4) * P

theorem maximal_volume_of_solid (P : ℝ) : 
    let AC := AC_length P in
    let BC := BC_length P in
    ∀ (x : ℝ), x = AC →
    (BC = P - 2 * x) →
    (AK : ℝ) (h1 : AK^2 = x^2 - ((1/2) * P - x)^2) →
    (V : ℝ) (h2 : V = (2/3) * Math.pi * (P * x - (1/4) * P^2) * ((1/2) * P - x)) →
    (x = (3/8) * P) ∧ (BC = (1/4) * P) :=
    sorry

end maximal_volume_of_solid_l640_640734


namespace round_time_of_A_l640_640089

theorem round_time_of_A (T_a T_b : ℝ) 
  (h1 : 4 * T_b = 5 * T_a) 
  (h2 : 4 * T_b = 4 * T_a + 10) : T_a = 10 :=
by
  sorry

end round_time_of_A_l640_640089


namespace base_length_of_parallelogram_l640_640004

theorem base_length_of_parallelogram 
  (area : ℝ) (base altitude : ℝ) 
  (h_area : area = 242)
  (h_altitude : altitude = 2 * base) :
  base = 11 :=
by
  sorry

end base_length_of_parallelogram_l640_640004


namespace range_distance_between_parallel_lines_l640_640682

noncomputable def distance (a b : ℝ) : ℝ :=
  (|a - b|) / real.sqrt 2

theorem range_distance_between_parallel_lines (a b c : ℝ) (h1: a + b = -1) (h2: ab = c) (h3: 0 ≤ c) (h4: c ≤ 1/8) :
  1/2 ≤ distance a b ∧ distance a b ≤ real.sqrt 2 / 2 :=
by
  sorry

end range_distance_between_parallel_lines_l640_640682


namespace proportional_x_y2_y_z2_l640_640087

variable {x y z k m c : ℝ}

theorem proportional_x_y2_y_z2 (h1 : x = k * y^2) (h2 : y = m / z^2) (h3 : x = 2) (hz4 : z = 4) (hz16 : z = 16):
  x = 1/128 :=
by
  sorry

end proportional_x_y2_y_z2_l640_640087


namespace power_mod_l640_640838

theorem power_mod (x n m : ℕ) : (x^n) % m = x % m := by 
  sorry

example : 5^2023 % 150 = 5 % 150 :=
by exact power_mod 5 2023 150

end power_mod_l640_640838


namespace proof1_proof2_proof3_l640_640541

variables (x m n : ℝ)

theorem proof1 (x : ℝ) : (-3 * x - 5) * (5 - 3 * x) = 9 * x^2 - 25 :=
sorry

theorem proof2 (x : ℝ) : (-3 * x - 5) * (5 + 3 * x) = - (3 * x + 5) ^ 2 :=
sorry

theorem proof3 (m n : ℝ) : (2 * m - 3 * n + 1) * (2 * m + 1 + 3 * n) = (2 * m + 1) ^ 2 - (3 * n) ^ 2 :=
sorry

end proof1_proof2_proof3_l640_640541


namespace solve_equation_l640_640451

theorem solve_equation (x : ℝ) (h : ((x^2 + 3*x + 4) / (x + 5)) = x + 6) : x = -13 / 4 :=
by sorry

end solve_equation_l640_640451


namespace chips_circle_arrangement_l640_640640

theorem chips_circle_arrangement 
  (n : ℕ)
  (colors : Finset {c : ℕ // c < n}) 
  (chips : Multiset {c : ℕ // c < n})
  (h_chips_len : chips.card = n)
  (h_color_count : ∀ c ∈ colors, (Multiset.count c chips ≤ n / 2)) :
  ∃ (arrangement : Multiset {c : ℕ // c < n}), 
    (∀ i : ℕ, ∀c ∈ arrangement, c ≠ arrangement.get (i % n)) := 
sorry

end chips_circle_arrangement_l640_640640


namespace disprove_toms_claim_l640_640139

variable (C : Type) -- Type of cards
variable (is_even : C → Prop) -- Predicate for even number
variable (is_consonant : C → Prop) -- Predicate for consonant
variable (even_card : C) -- Card representing the number 4

-- Tom's claim as a predicate
def toms_claim (c : C) : Prop := is_consonant c → ¬ is_even c

-- The goal is to show that if the card with number 4 has a consonant, it disproves Tom's claim
theorem disprove_toms_claim (h_even : is_even even_card) (h_contra : is_consonant even_card) : 
  ¬ toms_claim even_card :=
by
  sorry

end disprove_toms_claim_l640_640139


namespace range_a_l640_640711

noncomputable def f (x a : ℝ) : ℝ := 1 / (a * log (x + 1))
noncomputable def g (x : ℝ) : ℝ := x^2 * (x + 1)^2
noncomputable def h (x : ℝ) : ℝ := x^2 / log x

theorem range_a (x_1 a : ℝ) (h₁ : e^1/4 - 1 < x_1) (h₂ : x_1 < e - 1) (h₃ : a = (x_1 + 1)^2 / log (x_1 + 1)) :
  2 * exp 1 ≤ a ∧ a < exp 1 ^ 2 :=
sorry

end range_a_l640_640711


namespace triangle_area_correct_l640_640275

variables (DE EF DF : ℝ)

def s := (DE + EF + DF) / 2

noncomputable def area (DE EF DF : ℝ) : ℝ :=
  real.sqrt (s DE EF DF * (s DE EF DF - DE) * (s DE EF DF - EF) * (s DE EF DF - DF))

theorem triangle_area_correct :
  DE = 30 → EF = 24 → DF = 18 → area DE EF DF = 216 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp only [s]
  have : s 30 24 18 = 36 := by unfold s; linarith
  rw this
  unfold area
  have : real.sqrt (36 * (36 - 30) * (36 - 24) * (36 - 18)) = 216 := by
    norm_num
  rw this
  apply congr_arg
  sorry

end triangle_area_correct_l640_640275


namespace find_y_l640_640258

theorem find_y (y : ℤ) (h : 3^(y - 2) = 9^3) : y = 8 :=
by
  sorry

end find_y_l640_640258


namespace sum_of_divisors_and_totient_inequality_equality_condition_l640_640641

open Nat

def sigma (n : ℕ) : ℕ := 
  ∑ d in divisors n, d

def phi (n : ℕ) : ℕ := 
  (finset.range n).filter (λ m, gcd m n = 1).card

theorem sum_of_divisors_and_totient_inequality (n : ℕ) (h: n > 0) : 
  (1 / sigma n : ℚ) + (1 / phi n : ℚ) ≥ (2 / n : ℚ) :=
sorry

theorem equality_condition (n : ℕ) (h: n > 0) :
  (1 / sigma n : ℚ) + (1 / phi n : ℚ) = (2 / n : ℚ) ↔ n = 1 :=
sorry

end sum_of_divisors_and_totient_inequality_equality_condition_l640_640641


namespace f_le_g_l640_640958

noncomputable def f (n : ℕ) : ℝ := 
  ∑ i in finset.range (n + 1), if i = 0 then 1 else (1 : ℝ) / i ^ 3

noncomputable def g (n : ℕ) : ℝ := 
  (3 / 2) - (1 / (2 * n ^ 2))

theorem f_le_g (n : ℕ) (h : 0 < n) : f n ≤ g n := by
  sorry

end f_le_g_l640_640958


namespace polar_to_rectangular_intersection_range_l640_640339

-- Definitions of parametric equations for curve C
def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line_l (ρ θ m: ℝ) : Prop := ρ * sin (θ + π / 3) + m = 0

-- Rectangular equation of line l
def rectangular_line_l (x y m: ℝ) : Prop := sqrt 3 * x + y + 2 * m = 0

-- Prove that polar equation of line l can be transformed to rectangular form
theorem polar_to_rectangular (ρ θ m : ℝ) :
  (polar_line_l ρ θ m) → (∃ x y, ρ = sqrt (x^2 + y^2) ∧ θ = arctan (y / x) ∧ rectangular_line_l x y m) :=
  sorry

-- Prove that the range of values for m ensuring line l intersects curve C is [-19/12, 5/2]
theorem intersection_range (m : ℝ) :
  (∃ t : ℝ, rectangular_line_l (sqrt 3 * cos (2 * t)) (2 * sin t) m) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
  sorry

end polar_to_rectangular_intersection_range_l640_640339


namespace pears_picked_total_l640_640138

theorem pears_picked_total : 
  let alyssa_pears := 42
  let nancy_pears := 17
  alyssa_pears + nancy_pears = 59 := 
by
  let alyssa_pears := 42
  let nancy_pears := 17
  show alyssa_pears + nancy_pears = 59 from sorry

end pears_picked_total_l640_640138


namespace perpendicular_bisector_of_circles_l640_640480

theorem perpendicular_bisector_of_circles
  (circle1 : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0)
  (circle2 : ∀ x y : ℝ, x^2 + y^2 - 6 * x = 0) :
  ∃ x y : ℝ, (3 * x - y - 9 = 0) :=
by
  sorry

end perpendicular_bisector_of_circles_l640_640480


namespace complete_collection_prob_l640_640049

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end complete_collection_prob_l640_640049


namespace find_x_and_C_l640_640248

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2 - x}

theorem find_x_and_C (x : ℝ) (C : Set ℝ) :
  B x ⊆ A x → B (-2) ∪ C = A (-2) → x = -2 ∧ C = {3} :=
by
  sorry

end find_x_and_C_l640_640248


namespace company_start_to_incur_losses_l640_640873

noncomputable def P (n : ℕ) : ℤ := 2000000 - (n - 1) * 200000

theorem company_start_to_incur_losses : 
  ∃ n, P n ≤ 0 ∧ ∀ m, m < n → P m > 0 :=
begin
  use 12,
  split,
  {
    -- Show that P 12 ≤ 0
    unfold P,
    norm_num,
  },
  {
    -- Show that for all m < 12, P m > 0
    intros m h,
    unfold P,
    have h1 : m < 12 := h,
    norm_num at h1,
    sorry, -- This 'sorry' indicates unfinished parts, as proofs are not required.
  }
end

end company_start_to_incur_losses_l640_640873


namespace solve_quadratic_abs_l640_640490

theorem solve_quadratic_abs (x : ℝ) :
  x^2 - |x| - 1 = 0 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 ∨ 
                   x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
sorry

end solve_quadratic_abs_l640_640490


namespace construct_triangle_l640_640163

-- Define the necessary elements such as points and lengths
variables {A B C S D E : Type} [AffineSpace ℝ A] [MetricSpace B] [MetricSpace C]

-- Given conditions
variables (c : ℝ) 
noncomputable def centroid {A B C : AffineSpace ℝ A} : AffineSpace ℝ A := sorry
def segment_through_centroid_parallel_BC {A B C S : AffineSpace ℝ A} : Line ℝ A := sorry
def distance_from_A_to_line {A : MetricSpace B} (line : Line ℝ A) : ℝ := sorry

-- Define the properties of the given line and centroid requirement
axiom line_through_centroid (S : centroid) (line : segment_through_centroid_parallel_BC) : Prop := sorry
axiom distance_A (A : MetricSpace B) : distance_from_A_to_line (segment_through_centroid_parallel_BC) (A) : Prop := sorry

-- Prove the constructibility of triangle ABC
theorem construct_triangle (c : ℝ) :
  ∃ (A B C D E : A), 
    (distance_A A = sorry) ∧ 
    (distance_from_A_to_line (segment_through_centroid_parallel_BC) A = sorry) ∧
    sorry := 
sorry

end construct_triangle_l640_640163


namespace triangle_ABC_BC_length_l640_640728

theorem triangle_ABC_BC_length 
  (A B C D : ℝ)
  (AB AD DC AC BD BC : ℝ)
  (h1 : BD = 20)
  (h2 : AC = 69)
  (h3 : AB = 29)
  (h4 : BD^2 + DC^2 = BC^2)
  (h5 : AD^2 + BD^2 = AB^2)
  (h6 : AC = AD + DC) : 
  BC = 52 := 
by
  sorry

end triangle_ABC_BC_length_l640_640728


namespace data_transformation_stddev_l640_640646

variable (n : ℕ)
variable (x y : Fin n → ℝ)

noncomputable def stddev (x : Fin n → ℝ) : ℝ :=
  Real.sqrt (Finset.univ.sum (fun i => (x i - (Finset.univ.sum x) / n) ^ 2) / n)

theorem data_transformation_stddev
  (n : ℕ)
  (x : Fin n → ℝ)
  (h : stddev x = 4) :
  let y := fun i => 2 * (x i) - 1 in
  stddev y = 8 := sorry

end data_transformation_stddev_l640_640646


namespace xy_composite_l640_640391

theorem xy_composite (x y : ℕ) (hx : 1 < x) (hy : 1 < y) 
  (h : (x^2 + y^2 - 1) % (x + y - 1) = 0) : ¬(x + y - 1).prime :=
by
  sorry

end xy_composite_l640_640391


namespace tens_digit_of_2013_pow_2018_minus_2019_l640_640841

theorem tens_digit_of_2013_pow_2018_minus_2019 :
  (2013 ^ 2018 - 2019) % 100 / 10 % 10 = 5 := sorry

end tens_digit_of_2013_pow_2018_minus_2019_l640_640841


namespace sticker_probability_l640_640060

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end sticker_probability_l640_640060


namespace total_heads_eq_50_l640_640122

theorem total_heads_eq_50 (H C : ℕ) (hH : H = 28) (feet_eq : 2 * H + 4 * C = 144) :
    H + C = 50 :=
by
  have h1 : 2 * 28 + 4 * C = 144, from feet_eq
  have h2 : 56 + 4 * C = 144, from congr_arg (λ x, 2 * 28 + x) (add_zero (4 * C))
  have h3 : 4 * C = 144 - 56, by
    rw h2
    exact sub_eq_of_eq_add rfl
  have h4 : C = (144 - 56) / 4, by
    rw [←nat.mul_left_inj (lt_of_lt_of_le (nat.zero_lt_succ 3) (le_of_eq _))]
    assumption
  have h5 : C = 22, from nat.div_eq_of_eq_mul_left (dec_trivial : 0 < 4) (by simp [*, add_comm])
  rw [←h5, hH]
  exact add_comm 28 22

end total_heads_eq_50_l640_640122


namespace proof_problem_l640_640649

def problem_conditions : Prop :=
  -- P1: Every pib is a collection of maas (implicitly given by existence of pibs and maas)
  True 

def P2 : Prop :=
  ∀ (pi1 pi2 : pib) (h_pi1_ne_pi2 : pi1 ≠ pi2), 
  ∃ (ma1 ma2 : maa), 
  ma1 ∈ pi1 ∧ ma1 ∈ pi2 ∧ ma2 ∈ pi1 ∧ ma2 ∈ pi2 ∧ ma1 ≠ ma2

def P3 : Prop :=
  ∀ (m : maa), ∃ (pi1 pi2 : pib),
  pi1 ≠ pi2 ∧ m ∈ pi1 ∧ m ∈ pi2

def P4 : Prop := 
  ∃ (pibs : Finset pib), pibs.card = 5

def num_maas : Prop :=
  ∃ (maas : Finset maa), maas.card = 10

def num_maas_per_pib : Prop :=
  ∀ (pi : pib) (h_pi : pi ∈ pibs), 
  (pibs.bind (λ pi, pi.to_finset)).card = 4

def unique_maas_not_shared : Prop :=
  ∀ (m1 m2 : maa) (pi1 pi2 : pib) (h_m1_in_pi1 : m1 ∈ pi1) (h_m2_in_pi2 : m2 ∈ pi2),
  m1 ≠ m2 → pi1 ≠ pi2 →
  ∃! (m : maa), m ∉ pi1 ∧ m ∉ pi2

theorem proof_problem : problem_conditions ∧ P2 ∧ P3 ∧ P4 → num_maas ∧ num_maas_per_pib ∧ unique_maas_not_shared := 
by sorry

end proof_problem_l640_640649


namespace find_pairs_l640_640948

def sum_of_digits (n : ℕ) : ℕ := -- Definition of sum of digits
  n.digits 10 |>.sum

theorem find_pairs (a b : ℕ) (h1 : a > b) 
  (h2 : a ∣ b + sum_of_digits a)
  (h3 : b ∣ a + sum_of_digits b) : 
  (a, b) = (18, 9) ∨ (a, b) = (27, 18) :=
begin
  sorry
end

end find_pairs_l640_640948


namespace relationship_among_abc_l640_640986

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then 1 + Real.log (2 - x) / Real.log 2
else 2^(x - 1)

def a : ℝ := f (-2)
def b : ℝ := f (2)
def c : ℝ := f (Real.log 12 / Real.log 2)

theorem relationship_among_abc : b < a ∧ a < c :=
by 
  -- sorry to skip the proof
  sorry

end relationship_among_abc_l640_640986


namespace customers_remaining_in_evening_l640_640548

theorem customers_remaining_in_evening
  (m_0 : ℕ) (m_1 : ℕ) (a : ℕ) (p : ℚ) (d : ℚ)
  (h_m0 : m_0 = 33)
  (h_m1 : m_1 = 31)
  (h_a : a = 26)
  (h_p : p = 0.25)
  (h_d : d = 0.40)
  : (m_0 - m_1 + a) * (1 + p) * (1 - d) = 25 := 
by 
  -- Definitions and assumptions according to described conditions
  let morning_shift_after_departure := m_0 - m_1
  let total_after_morning_shift := morning_shift_after_departure + a
  let evening_shift := (nat.floor ((total_after_morning_shift : ℚ) * (1 - d)) : ℕ)
  let evening_customers := nat.floor (m_0 * (1 + p))
  exact sorry


end customers_remaining_in_evening_l640_640548


namespace find_z_l640_640415

variable (x y z w : ℝ)
variable (f : ℝ → ℝ)
variable (C : ℝ)

noncomputable def problem_statement : Prop :=
  (x = 200) ∧
  (y = 2 * z) ∧
  (x - z = 0.5 * y) ∧
  (x + y + z + w = 500) ∧
  (w = ∫ (t : ℝ) in 0..1, f t) ∧
  (∀ t, f' t = 2 * t) ∧
  (f 0 = 2) ∧
  (f 1 = 50)

theorem find_z (h : problem_statement x y z w f C) : z = 100 :=
sorry

end find_z_l640_640415


namespace rect_eq_line_range_of_m_l640_640326

-- Definitions based on the conditions
def parametric_x (t : ℝ) : ℝ := sqrt 3 * cos (2 * t)
def parametric_y (t : ℝ) : ℝ := 2 * sin t

def polar_rho (θ m : ℝ) : ℝ := -(m / (sin (θ + π / 3)))

-- Rectangular equation of the line l
theorem rect_eq_line (x y m : ℝ) : polar_rho (atan2 y x) m * sin (atan2 y x + π / 3) + m = 0 → sqrt 3 * x + y + 2 * m = 0 :=
by
  sorry

-- Range of values for m
theorem range_of_m (C : ℝ → ℝ × ℝ)
  (hC : ∀ t, C t = (parametric_x t, parametric_y t)) (m : ℝ) :
  (∃ t, sqrt 3 * fst (C t) + snd (C t) + 2 * m = 0) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
by
  sorry

end rect_eq_line_range_of_m_l640_640326


namespace frank_steps_forward_l640_640622

theorem frank_steps_forward :
  let initial_pos := 0 in
  let final_pos := initial_pos - 5 + 10 - 2 + 2 * 2 in
  final_pos = 7 :=
by
  sorry

end frank_steps_forward_l640_640622


namespace arithmetic_sequence_sum_inequality_l640_640247

open BigOperators
open Nat

-- Given sequences {a_n} and {b_n} with non-zero terms and the relation
axiom seq_relation (a b : ℕ → ℝ) (n : ℕ) : ℕ → ℝ → Prop :=
∀ n, 0 < a n ∧ 0 < b n ∧ a n * b (n+1) = a (n+1) * (2 * a n + b n)

-- Define c_n = b_n / a_n
def c (a b : ℕ → ℝ) (n : ℕ) := b n / a n

-- Problem (I): Prove that {c_n} is an arithmetic sequence
theorem arithmetic_sequence (a b : ℕ → ℝ) (n : ℕ)
  (h : seq_relation a b n) : ∀ n, c a b (n+1) - c a b n = 2 :=
sorry

-- Problem (II): Given specific initial conditions, prove S_n < 1/2
def S (b : ℕ → ℝ) (n : ℕ) : ℝ := ∑ k in finset.range n, 1 / b (k + 1)

theorem sum_inequality (a b : ℕ → ℝ)
  (h1 : seq_relation a b)
  (h2 : b 1 = 4)
  (h3 : b 2 = 12)
  (h4 : a 1 = 2)
  (h_arith : ∀ n, a (n + 1) = a n + 1) :
  ∀ n, S b n < 1/2 :=
sorry

end arithmetic_sequence_sum_inequality_l640_640247


namespace polar_to_rectangular_range_of_m_l640_640290

-- Definition of the parametric equations of curve C
def curve (t : ℝ) : ℝ × ℝ :=
  (sqrt(3) * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line (rho theta m : ℝ) : Prop :=
  rho * sin (theta + π / 3) + m = 0

-- Definition of the rectangular equation of line l
def rectangular_line (x y m : ℝ) : Prop :=
  sqrt(3) * x + y + 2 * m = 0

-- Convert polar line equation to rectangular form
theorem polar_to_rectangular (rho theta m x y : ℝ)
  (h1 : polar_line rho theta m)
  (h2 : rho = sqrt (x^2 + y^2))
  (h3 : cos theta * rho = x ∧ sin theta * rho = y) :
  rectangular_line x y m :=
sorry

-- Range of values for m for which l intersects C
theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, (sqrt(3) * cos (2 * t), 2 * sin t) ∈ {p : ℝ × ℝ | rectangular_line p.1 p.2 m}) ↔
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end polar_to_rectangular_range_of_m_l640_640290


namespace rect_eq_line_range_of_m_l640_640329

-- Definitions based on the conditions
def parametric_x (t : ℝ) : ℝ := sqrt 3 * cos (2 * t)
def parametric_y (t : ℝ) : ℝ := 2 * sin t

def polar_rho (θ m : ℝ) : ℝ := -(m / (sin (θ + π / 3)))

-- Rectangular equation of the line l
theorem rect_eq_line (x y m : ℝ) : polar_rho (atan2 y x) m * sin (atan2 y x + π / 3) + m = 0 → sqrt 3 * x + y + 2 * m = 0 :=
by
  sorry

-- Range of values for m
theorem range_of_m (C : ℝ → ℝ × ℝ)
  (hC : ∀ t, C t = (parametric_x t, parametric_y t)) (m : ℝ) :
  (∃ t, sqrt 3 * fst (C t) + snd (C t) + 2 * m = 0) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
by
  sorry

end rect_eq_line_range_of_m_l640_640329


namespace figure_total_area_l640_640018

theorem figure_total_area (a : ℝ) (h : a^2 - (3/2 * a^2) = 0.6) : 
  5 * a^2 = 6 :=
by
  sorry

end figure_total_area_l640_640018


namespace weekly_earnings_l640_640110

theorem weekly_earnings (computers_per_day : ℕ) (price_per_computer : ℕ) (days_in_week : ℕ) 
  (h1 : computers_per_day = 1500) 
  (h2 : price_per_computer = 150) 
  (h3 : days_in_week = 7) 
  : computers_per_day * days_in_week * price_per_computer = 1575000 := 
by
  rw [h1, h2, h3]
  simp
  sorry

end weekly_earnings_l640_640110


namespace volume_of_truncated_cone_l640_640894

-- Define the initial conditions 
variables (R1 R2 h : ℝ)

-- Set the given values
def radius_large_base := (R1 = 12)
def radius_small_base := (R2 = 6)
def height_truncated_cone := (h = 10)

-- Define the expression of the volume of the truncated cone using given values
noncomputable def volume_truncated_cone := (1 / 3) * π * h * (R1^2 + R2^2 + R1 * R2)

-- The theorem statement to be proved
theorem volume_of_truncated_cone
  (R1 R2 h : ℝ)
  (h_pos : h > 0)
  (radius_large_base : R1 = 12)
  (radius_small_base : R2 = 6)
  (height_truncated_cone : h = 10):
  volume_truncated_cone R1 R2 h = 840 * π :=
by 
  -- sorry is used here since we are not proving
  sorry

end volume_of_truncated_cone_l640_640894


namespace product_of_all_t_values_l640_640187

theorem product_of_all_t_values :
  let pairs := [(a, b) | (a, b) ∈ List.product (List.range 43).map (λ a, a*(-1))],
      valid_pairs := List.filter (λ (a, b), a * b = -42) pairs,
      t_values := List.map (λ (a, b), a + b) valid_pairs,
      product := t_values.foldr (*) 1
  in product = 73407281 := sorry

end product_of_all_t_values_l640_640187


namespace sum_of_final_numbers_l640_640032

theorem sum_of_final_numbers (x y : ℝ) (S : ℝ) (h : x + y = S) : 
  3 * (x + 4) + 3 * (y + 4) = 3 * S + 24 := by
  sorry

end sum_of_final_numbers_l640_640032


namespace find_first_class_tickets_l640_640884

-- Definitions corresponding to the conditions
variables {x y : ℕ}  -- The number of first-class and second-class tickets respectively

-- Condition 1: The total number of tickets is 45
def total_tickets := x + y = 45

-- Condition 2: The total cost of tickets bought is 400 yuan
def total_cost := 10 * x + 8 * y = 400

-- The theorem that needs to be proved
theorem find_first_class_tickets (h1 : total_tickets) (h2 : total_cost) : x = 20 :=
by
  sorry

end find_first_class_tickets_l640_640884


namespace find_minimum_abs_sum_l640_640412

noncomputable def minimum_abs_sum (α β γ : ℝ) : ℝ :=
|α| + |β| + |γ|

theorem find_minimum_abs_sum :
  ∃ α β γ : ℝ, α + β + γ = 2 ∧ α * β * γ = 4 ∧
  minimum_abs_sum α β γ = 6 := by
  sorry

end find_minimum_abs_sum_l640_640412


namespace package_cost_l640_640914

theorem package_cost
  (total_cost : ℝ)
  (cost_per_letter : ℝ)
  (num_letters : ℕ)
  (num_packages : ℕ)
  (cost_per_package : ℝ) :
  total_cost = 4.49 →
  cost_per_letter = 0.37 →
  num_letters = 5 →
  num_letters = num_packages + 2 →
  total_cost = (real.of_nat num_letters * cost_per_letter) + (real.of_nat num_packages * cost_per_package) →
  cost_per_package = 0.88 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end package_cost_l640_640914


namespace positive_integers_divisors_of_2_to_the_n_plus_1_l640_640606

theorem positive_integers_divisors_of_2_to_the_n_plus_1:
  ∀ n : ℕ, 0 < n → (n^2 ∣ 2^n + 1) ↔ (n = 1 ∨ n = 3) :=
by
  sorry

end positive_integers_divisors_of_2_to_the_n_plus_1_l640_640606


namespace probability_of_passing_l640_640721

theorem probability_of_passing :
  let total_combinations := (comb 10 3)
  let success_two_correct_one_incorrect := (comb 4 1) * (comb 6 2)
  let success_all_correct := (comb 6 3)
  let total_successful_outcomes := success_two_correct_one_incorrect + success_all_correct
  total_successful_outcomes / total_combinations = 2 / 3 :=
by
  sorry

end probability_of_passing_l640_640721


namespace general_admission_ticket_cost_l640_640459

theorem general_admission_ticket_cost
    (student_ticket_cost : ℕ := 4)
    (total_tickets_sold : ℕ := 525)
    (total_revenue : ℕ := 2876)
    (general_admission_tickets_sold : ℕ := 388)
    (x : ℕ) :
  total_revenue = (student_ticket_cost * (total_tickets_sold - general_admission_tickets_sold)) + (general_admission_tickets_sold * x) →
  x = 6 :=
by
  intros h
  have student_tickets_sold : ℕ := total_tickets_sold - general_admission_tickets_sold
  have revenue_student_tickets : ℕ := student_ticket_cost * student_tickets_sold
  have revenue_general_admission_tickets : ℕ := general_admission_tickets_sold * x
  have h1 : total_revenue = revenue_student_tickets + revenue_general_admission_tickets := h
  sorry

end general_admission_ticket_cost_l640_640459


namespace charging_time_l640_640880

theorem charging_time (S T L : ℕ → ℕ) 
  (HS : ∀ t, S t = 15 * t) 
  (HT : ∀ t, T t = 8 * t) 
  (HL : ∀ t, L t = 5 * t)
  (smartphone_capacity tablet_capacity laptop_capacity : ℕ)
  (smartphone_percentage tablet_percentage laptop_percentage : ℕ)
  (h_smartphone : smartphone_capacity = 4500)
  (h_tablet : tablet_capacity = 10000)
  (h_laptop : laptop_capacity = 20000)
  (p_smartphone : smartphone_percentage = 75)
  (p_tablet : tablet_percentage = 25)
  (p_laptop : laptop_percentage = 50)
  (required_charge_s required_charge_t required_charge_l : ℕ)
  (h_rq_s : required_charge_s = smartphone_capacity * smartphone_percentage / 100)
  (h_rq_t : required_charge_t = tablet_capacity * tablet_percentage / 100)
  (h_rq_l : required_charge_l = laptop_capacity * laptop_percentage / 100)
  (time_s time_t time_l : ℕ)
  (h_time_s : time_s = required_charge_s / 15)
  (h_time_t : time_t = required_charge_t / 8)
  (h_time_l : time_l = required_charge_l / 5) : 
  max time_s (max time_t time_l) = 2000 := 
by 
  -- This theorem states that the maximum time taken for charging is 2000 minutes
  sorry

end charging_time_l640_640880


namespace pencils_per_box_l640_640419

theorem pencils_per_box:
  ∀ (red_pencils blue_pencils yellow_pencils green_pencils total_pencils num_boxes : ℕ),
  red_pencils = 20 →
  blue_pencils = 2 * red_pencils →
  yellow_pencils = 40 →
  green_pencils = red_pencils + blue_pencils →
  total_pencils = red_pencils + blue_pencils + yellow_pencils + green_pencils →
  num_boxes = 8 →
  total_pencils / num_boxes = 20 :=
by
  intros red_pencils blue_pencils yellow_pencils green_pencils total_pencils num_boxes
  intros h1 h2 h3 h4 h5 h6
  sorry

end pencils_per_box_l640_640419


namespace sum_fibonacci_harmonic_l640_640943

noncomputable def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

noncomputable def harmonic (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), (1 / (k + 1 : ℚ))

theorem sum_fibonacci_harmonic :
  ∑' (n : ℕ), (fibonacci n / ((n + 1 : ℚ) * harmonic n * harmonic (n + 1))) = 1 := 
by 
  sorry

end sum_fibonacci_harmonic_l640_640943


namespace number_of_SUVs_washed_l640_640388

theorem number_of_SUVs_washed (charge_car charge_truck charge_SUV total_raised : ℕ) (num_trucks num_cars S : ℕ) :
  charge_car = 5 →
  charge_truck = 6 →
  charge_SUV = 7 →
  total_raised = 100 →
  num_trucks = 5 →
  num_cars = 7 →
  total_raised = num_cars * charge_car + num_trucks * charge_truck + S * charge_SUV →
  S = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end number_of_SUVs_washed_l640_640388


namespace sin_double_angle_l640_640670

variable (θ : ℝ)

theorem sin_double_angle (h : sin (θ - π / 4) = 1 / 3) : sin (2 * θ) = 7 / 9 :=
begin
  sorry
end

end sin_double_angle_l640_640670


namespace find_a_if_point_on_line_l640_640270

theorem find_a_if_point_on_line (a : ℝ) (z : ℂ) (h₁ : z = (a-1) + 3 * complex.I) (h₂ : ∃ p : ℂ, p.im = 3 ∧ p.re = a - 1 ∧ (p.im = p.re + 2)) : 
a = 2 := sorry

end find_a_if_point_on_line_l640_640270


namespace value_of_p_l640_640265

theorem value_of_p (p : ℝ) (h : p > 0) (focus : (1, 0)) :
  (∀ (x y : ℝ), y^2 = 2 * p * x → (x, y) = (1, 0)) → p = 2 :=
by 
  intro h_f
  have : ∀ (x y : ℝ), y^2 = 2 * p * x ↔ (x, y) = (1, 0) := λ x y, Iff.intro
    (λ h_xy, ((eq.recOn (congr_arg (λ z, x = z) h_f (x, y))) h_xy : (x, y) = (1, 0)))
    (λ h_1, eq.recOn (congr_arg (λ z, (x, y) = z) h_f (1, 0)) h_1)
  sorry -- proof is omitted

end value_of_p_l640_640265


namespace probability_of_picking_grain_buds_l640_640086

theorem probability_of_picking_grain_buds :
  let num_stamps := 3
  let num_grain_buds := 1
  let probability := num_grain_buds / num_stamps
  probability = 1 / 3 :=
by
  sorry

end probability_of_picking_grain_buds_l640_640086


namespace circumscribed_sphere_surface_area_l640_640135

theorem circumscribed_sphere_surface_area
  (l1 l2 l3 : ℝ) 
  (h1 : l1 = 1)
  (h2 : l2 = Real.sqrt 2)
  (h3 : l3 = Real.sqrt 3)
  (h4 : l1*l1 + l2*l2 + l3*l3 = 6) :
  4 * Real.pi * (Real.sqrt(6) / 2) ^ 2 = 6 * Real.pi :=
by {
  sorry
}

end circumscribed_sphere_surface_area_l640_640135


namespace proof_ellipse_proof_OP_range_l640_640981
noncomputable def ellipse_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) : Prop :=
  (∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ x = 1 ∧ y = 3/2)

noncomputable def ellipse_eccentricity (a b : ℝ) : Prop :=
  (a^2 - b^2) / a^2 = 1/4

noncomputable def equations (a b : ℝ) (M : ℝ × ℝ) (ecc : ℝ) : Prop :=
  (a^2 = 4 ∧ b^2 = 3)

noncomputable def ellipse_equation_result : Prop :=
  (∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1)

noncomputable def range_OP (a b k m : ℝ) (k_range: 0 < |k| ∧ |k| ≤ 1/2) : Prop :=
  (∃ P : ℝ × ℝ, sqrt(3) < |sqrt(P.1^2 + P.2^2)| ∧ |sqrt(P.1^2 + P.2^2)| ≤ sqrt(13) / 2)

theorem proof_ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
                      (ecc : ellipse_eccentricity a b) (M : ℝ × ℝ) :
  ellipse_equation a b a_pos b_pos a_gt_b ∧ equations a b M ecc → ellipse_equation_result :=
by 
  sorry

theorem proof_OP_range (a b k m : ℝ) (k_range: 0 < |k| ∧ |k| ≤ 1/2) :
  range_OP a b k m k_range :=
by 
  sorry

end proof_ellipse_proof_OP_range_l640_640981


namespace determine_a_l640_640926

noncomputable def G₁₇ : ℤ := 1597
noncomputable def G₁₈ : ℤ := 2584
noncomputable def G₁₉ : ℤ := 4181

def satisfies_fibonacci_relation (a b : ℤ) : Prop := 
  G₁₉ * a + G₁₈ * b = 0 ∧ G₁₈ * a + G₁₇ * b = -1

theorem determine_a (a b : ℤ) (h_int : a ∈ ℤ ∧ b ∈ ℤ)
    (h_factor: satisfies_fibonacci_relation a b) : a = 2584 :=
by
  sorry

end determine_a_l640_640926


namespace solve_for_x_l640_640452

-- Define the equation
def equation (x : ℝ) : Prop := (x^2 + 3 * x + 4) / (x + 5) = x + 6

-- Prove that x = -13 / 4 satisfies the equation
theorem solve_for_x : ∃ x : ℝ, equation x ∧ x = -13 / 4 :=
by
  sorry

end solve_for_x_l640_640452


namespace length_of_train_is_correct_l640_640133

-- Define the conditions
def time_crossing_bridge : ℝ := 21.998240140788738
def bridge_length : ℝ := 120
def train_speed_kmph : ℝ := 36
def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600) -- Convert speed from kmph to m/s
def distance_covered : ℝ := train_speed_mps * time_crossing_bridge

-- Define the problem statement in Lean
theorem length_of_train_is_correct : (distance_covered - bridge_length) = 99.98240140788738 := by
  sorry

end length_of_train_is_correct_l640_640133


namespace irrational_sqrt_3_l640_640568

theorem irrational_sqrt_3 : 
  (irr : ∀ x : ℝ, x^2 ≠ 3 → ¬ ∃ q : ℚ, x = q) → 
  irr (sqrt 3) :=
sorry

end irrational_sqrt_3_l640_640568


namespace factorial_division_l640_640662

theorem factorial_division (h : 10! = 3628800) : 10! / 4! = 151200 := by
  sorry

end factorial_division_l640_640662


namespace complex_in_third_quadrant_l640_640225

open Complex

noncomputable def quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else 4

theorem complex_in_third_quadrant (z : ℂ) (h : (2 + I) * z = -I) : quadrant z = 3 := by
  sorry

end complex_in_third_quadrant_l640_640225


namespace rhombus_area_l640_640859

theorem rhombus_area (side d1 : ℝ) (h_side : side = 28) (h_d1 : d1 = 12) : 
  (side = 28 ∧ d1 = 12) →
  ∃ area : ℝ, area = 328.32 := 
by 
  sorry

end rhombus_area_l640_640859


namespace smallest_non_moderate_prime_l640_640126

-- Define a moderate prime number
def is_moderate (p : ℕ) : Prop :=
  ∀ k m : ℕ, k > 1 → ∃ n : fin k → ℕ, (∑ i, (n i)^2) = p^(k+m)

-- Define the smallest moderate prime number q
def smallest_moderate_prime : ℕ :=
  if h : ∃ p, p.prime ∧ is_moderate p ∧ ∀ q, q.prime ∧ is_moderate q → p ≤ q
  then classical.some h
  else 0

-- Define the smallest non-moderate prime r given q
def smallest_non_moderate_prime_given (q : ℕ) : ℕ :=
  if h : ∃ p, p.prime ∧ ¬ is_moderate p ∧ q < p ∧ ∀ r, r.prime ∧ ¬ is_moderate r ∧ q < r → p ≤ r
  then classical.some h
  else 0

-- The main theorem statement
theorem smallest_non_moderate_prime (q r : ℕ) (h1 : q = smallest_moderate_prime) (h2 : r = smallest_non_moderate_prime_given q) : r = 7 := 
by sorry

end smallest_non_moderate_prime_l640_640126


namespace total_jelly_beans_l640_640426

theorem total_jelly_beans (vanilla_jelly_beans : ℕ) (h1 : vanilla_jelly_beans = 120) 
                           (grape_jelly_beans : ℕ) (h2 : grape_jelly_beans = 5 * vanilla_jelly_beans + 50) :
                           vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  sorry

end total_jelly_beans_l640_640426


namespace sum_of_roots_of_quadratic_l640_640166

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h : a ≠ 0) (h_eq : a = 1 ∧ b = -16 ∧ c = 12) :
  (sum_of_roots : ℝ) = 16 := by
  have h_quadratic := (λ (a b c : ℝ), ∀ x : ℝ, a * x^2 + b * x + c = 0) -- Definition of quadratic equation
  sorry

end sum_of_roots_of_quadratic_l640_640166


namespace value_two_stddev_below_mean_l640_640856

def mean : ℝ := 16.2
def standard_deviation : ℝ := 2.3

theorem value_two_stddev_below_mean : mean - 2 * standard_deviation = 11.6 :=
by
  sorry

end value_two_stddev_below_mean_l640_640856


namespace vector_rotation_l640_640244

theorem vector_rotation
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = ((1 - Real.sqrt 3) / 2, (1 + Real.sqrt 3) / 2)) :
  ∃ θ : ℝ, θ = Real.pi / 3 ∧ (∃ c : ℝ, c > 0 ∧ b = (c * (cos θ * a.1 - sin θ * a.2), c * (sin θ * a.1 + cos θ * a.2))) :=
by
  sorry

end vector_rotation_l640_640244


namespace proof_intervals_length_l640_640921

-- Define the necessary functions
def floor (x : ℝ) : ℤ := Int.floor x
def frac (x : ℝ) : ℝ := x - (floor x)

def f (x : ℝ) : ℝ := (floor x) * (frac x)
def g (x : ℝ) : ℝ := x - 1

theorem proof_intervals_length :
  let d1 := 1
  let d2 := 1
  let d3 := 2009
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ 2011) →
    ((f x > g x → (1 - 0 = d1)) ∧
    (f x = g x → (2 - 1 = d2)) ∧
    (f x < g x → (2011 - 2 = d3))) :=
begin
  intros x hx,
  sorry
end

end proof_intervals_length_l640_640921


namespace count_quadratic_functions_satisfying_condition_l640_640201

theorem count_quadratic_functions_satisfying_condition :
  ∃ (s : Finset ℕ), (s = Finset.range 1 10) ∧
  (∃ a b c ∈ s, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c) % 2 = 0) →
  ∃ (n : ℕ), n = 264 :=
by
  sorry

end count_quadratic_functions_satisfying_condition_l640_640201


namespace hexagon_inequality_l640_640024

theorem hexagon_inequality
  (A B C D E F : Point)
  (h1 : dist A B = dist B C)
  (h2 : dist C D = dist D E)
  (h3 : dist E F = dist F A) :
  (dist B C / dist B E) + (dist D E / dist D A) + (dist F A / dist F C) ≥ 3 / 2 :=
by
  sorry

end hexagon_inequality_l640_640024


namespace part_I_part_II_l640_640239

theorem part_I (x y: ℝ) (x0 : ℝ) (y0 : ℝ) (b : ℝ) (A B : ℝ×ℝ) (Q : ℝ×ℝ):
  (∀ (x1 y1 x2 y2: ℝ),
  A = (x1, y1) ∧ B = (x2, y2) ∧
  (y = - (1 / 2) * x + b)  ∧ (y^2 = 4 * x)  ∧ (y0 = -4) ∧ 
  (b = -8 / 5) ∧ (x0 = (x1 + x2) / 2) ∧ 
  (x0 = 2*b + 8) ∧ (Q = ((24 / 5), -4))) 
  → let x_center := 24 / 5,
        y_center := -4,
        radius_sq := 16
    in  (x - x_center)^2 + (y - y_center)^2 = radius_sq
  :=
sorry


theorem part_II (b : ℝ) (area : ℝ):
  (-2 < b ∧ b < 0 ∧
   let g := λ b, b^3 + 2*b^2 in
   let g' := λ b, 3*b^2 + 4*b in 
   let A := (- 4 / 3 : ℝ) in
   area = 4 * sqrt (2 * g A) ∧
   g A = 32/27 ∧
   area = 4 * sqrt (2) * sqrt ((32/27)))
   → area = (32 * sqrt 3 / 9)
  :=
sorry

end part_I_part_II_l640_640239


namespace variance_3X_minus_1_l640_640680

-- Define a random variable X with a given distribution
noncomputable def X_distribution : Distribution :=
  { p := λ x,
      if x = -1 then 1/2 else
      if x = 0  then 1/6 else
      if x = 1  then 1/6 else
      if x = 2  then 1/6 else 0,
    support := [-1, 0, 1, 2] }

-- Define the expected value of X
def E_X : ℝ := -1 * (1/2) + 0 * (1/6) + 1 * (1/6) + 2 * (1/6)

-- Given that the expected value of X is 0
lemma E_X_zero : E_X = 0 := by sorry

-- Variance calculation
def D_X : ℝ := E_X^2 - (E_X)^2

-- Given the distribution of X, we prove the variance of 3X - 1
theorem variance_3X_minus_1 : D(3 * X_distribution - 1) = 12 := by
  -- Proof steps are omitted
  sorry

end variance_3X_minus_1_l640_640680


namespace new_average_doubled_l640_640858

theorem new_average_doubled
  (average : ℕ)
  (num_students : ℕ)
  (h_avg : average = 45)
  (h_num_students : num_students = 30)
  : (2 * average * num_students / num_students) = 90 := by
  sorry

end new_average_doubled_l640_640858


namespace total_prairie_area_correct_percentage_untouched_correct_l640_640821

/-- Define the areas covered by the three storms and the untouched area: -/
def storm1_area : ℕ := 75000
def storm2_area : ℕ := 120000
def storm3_area : ℕ := 170000
def untouched_area : ℕ := 5000

/-- Calculate the total size of the prairie: -/
def total_prairie_area : ℕ := storm1_area + storm2_area + storm3_area + untouched_area

theorem total_prairie_area_correct :
  total_prairie_area = 370000 :=
by
  -- The proof is omitted
  sorry

/-- Calculate the percentage of the untouched prairie: -/
def percentage_untouched : ℚ := (untouched_area : ℚ) / total_prairie_area * 100

theorem percentage_untouched_correct :
  percentage_untouched ≈ 1.35 :=
by
  -- The proof is omitted
  sorry

end total_prairie_area_correct_percentage_untouched_correct_l640_640821


namespace find_sides_of_quadrilateral_l640_640489

noncomputable def quadrilateral_sides (r : ℝ) (m : ℝ) : Prop :=
  let AB := 2 * r
  let CD := 2 * r
  let O₁O₂ := 2 * r
  let area := m * r^2
  let BC := r * sqrt(6 + 2 * m)
  let AD := r * sqrt(6 - 2 * m)
  AB * CD / 2 = area → (BC, AD) = (r * sqrt(6 + 2 * m), r * sqrt(6 - 2 * m))

theorem find_sides_of_quadrilateral (r m : ℝ) (h : quadrilateral_sides r m) :
    ∃ BC AD, (BC, AD) = (r * sqrt(6 + 2 * m), r * sqrt(6 - 2 * m)) := sorry

end find_sides_of_quadrilateral_l640_640489


namespace speed_of_faster_train_l640_640066

-- Definitions for the given conditions
def speed_slower_train_kmph : ℝ := 36
def faster_train_length_m : ℝ := 70
def time_to_cross_s : ℝ := 7
def kmph_to_mps_factor : ℝ := 1000 / 3600 

-- Definitions for translation between units
def speed_slower_train_mps : ℝ := speed_slower_train_kmph * kmph_to_mps_factor
def relative_speed_mps := faster_train_length_m / time_to_cross_s
def relative_speed_kmph := relative_speed_mps / kmph_to_mps_factor
def speed_faster_train_kmph := speed_slower_train_kmph + relative_speed_kmph

-- Theorem statement
theorem speed_of_faster_train : speed_faster_train_kmph = 72 := 
by 
  have speed_slower_train_mps_calc : speed_slower_train_mps = 10, by sorry
  have relative_speed_mps_calc : relative_speed_mps = 10, by sorry
  have relative_speed_kmph_calc : relative_speed_kmph = 36, by sorry
  sorry

end speed_of_faster_train_l640_640066


namespace rect_eq_line_range_of_m_l640_640328

-- Definitions based on the conditions
def parametric_x (t : ℝ) : ℝ := sqrt 3 * cos (2 * t)
def parametric_y (t : ℝ) : ℝ := 2 * sin t

def polar_rho (θ m : ℝ) : ℝ := -(m / (sin (θ + π / 3)))

-- Rectangular equation of the line l
theorem rect_eq_line (x y m : ℝ) : polar_rho (atan2 y x) m * sin (atan2 y x + π / 3) + m = 0 → sqrt 3 * x + y + 2 * m = 0 :=
by
  sorry

-- Range of values for m
theorem range_of_m (C : ℝ → ℝ × ℝ)
  (hC : ∀ t, C t = (parametric_x t, parametric_y t)) (m : ℝ) :
  (∃ t, sqrt 3 * fst (C t) + snd (C t) + 2 * m = 0) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
by
  sorry

end rect_eq_line_range_of_m_l640_640328


namespace length_of_ship_l640_640602

-- Variables and conditions
variables (E L S : ℝ)
variables (W : ℝ := 0.9) -- Wind reducing factor

-- Conditions as equations
def condition1 : Prop := 150 * E = L + 150 * S
def condition2 : Prop := 70 * E = L - 63 * S

-- Theorem to prove
theorem length_of_ship (hc1 : condition1 E L S) (hc2 : condition2 E L S) : L = (19950 / 213) * E :=
sorry

end length_of_ship_l640_640602


namespace integer_not_in_range_of_g_l640_640760

def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉ else ⌊2 / (x + 3)⌋

theorem integer_not_in_range_of_g : ¬ ∃ x : ℝ, g x = 0 :=
by
  sorry

end integer_not_in_range_of_g_l640_640760


namespace problem_part1_problem_part2_l640_640976

theorem problem_part1 (a b : ℤ) (h : |a - 2| + (b + 1)^2 = 0) : b^a = 1 := sorry

theorem problem_part2 (a b : ℤ) (h : |a - 2| + (b + 1)^2 = 0) : a^3 + b^15 = 7 := sorry

end problem_part1_problem_part2_l640_640976


namespace solve_equation_l640_640450

theorem solve_equation (x : ℝ) (h : ((x^2 + 3*x + 4) / (x + 5)) = x + 6) : x = -13 / 4 :=
by sorry

end solve_equation_l640_640450


namespace rectangular_equation_common_points_l640_640316

-- Define parametric equations of curve C
def x (t : ℝ) := (sqrt 3) * cos (2 * t)
def y (t : ℝ) := 2 * sin t

-- Define polar equation of line l
def polar_line (ρ θ m : ℝ) := ρ * sin (θ + π / 3) + m

-- Rectangular equation of l
theorem rectangular_equation (ρ θ x y m : ℝ) (h1 : ρ = sqrt (x^2 + y^2)) (h2 : θ = atan2 y x) :
  polar_line ρ θ m = 0 → (sqrt 3 * x + y + 2 * m = 0) :=
by 
  sorry

-- Range of values for m where line intersects curve
theorem common_points (h: ∃ t, sqrt 3 * cos (2 * t) = x ∧ 2 * sin t = y) :
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by 
  sorry

end rectangular_equation_common_points_l640_640316


namespace ways_to_distribute_balls_l640_640254

theorem ways_to_distribute_balls (num_balls : ℕ) (num_boxes : ℕ) (balls_distinguishable : Prop) (boxes_indistinguishable : Prop) : 
  num_balls = 5 → 
  num_boxes = 3 → 
  balls_distinguishable → 
  boxes_indistinguishable → 
  (∃ count : ℕ, count = 41) :=
by
  intro h1 h2 h3 h4 
  use 41
  sorry

end ways_to_distribute_balls_l640_640254


namespace domain_of_f_l640_640834

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x - 5) + real.cbrt (x + 2) + real.sqrt (2 * x - 8)

theorem domain_of_f : {x : ℝ | x ≥ 5} = {x : ℝ | x ∈ domain_of f} := by
  sorry

end domain_of_f_l640_640834


namespace binomial_log_expression_l640_640982

theorem binomial_log_expression : 
  (∀ C : ℕ → ℕ → ℕ, C 20 0 * (Real.log 2)^20 +
  C 20 1 * (Real.log 2)^19 * Real.log 5 +
  ... +
  C 20 (20-1) * (Real.log 2)^(21-20) * (Real.log 5)^(20-1) +
  C 20 20 * (Real.log 5)^20 = 1) := sorry

end binomial_log_expression_l640_640982


namespace inequality_and_equality_condition_l640_640947

theorem inequality_and_equality_condition
  (a1 a2 a3 b1 b2 b3 : ℕ)
  (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3)
  (h4 : 0 < b1) (h5 : 0 < b2) (h6 : 0 < b3) :
  (a1 * b2 + a1 * b3 + a2 * b1 + a2 * b3 + a3 * b1 + a3 * b2) ^ 2 ≥
    4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) ∧
  ((a1 / b1 = a2 / b2) = (a2 / b2 = a3 / b3)) ↔
  (a1 / b1 = a2 / b2 = a3 / b3) := 
sorry

end inequality_and_equality_condition_l640_640947


namespace num_solutions_eq_2_l640_640478

-- Define the problem-specific conditions
def equation (x: ℝ) := sin (π / 2 * cos x) = cos (π / 2 * sin x)

def interval (x: ℝ) := 0 ≤ x ∧ x ≤ π

-- State the theorem
theorem num_solutions_eq_2 : (∃ (count: ℕ), count = 2 ∧ 
  (∀ x, equation x → interval x)) :=
begin
  sorry
end

end num_solutions_eq_2_l640_640478


namespace selection_count_arrangement_count_l640_640118

-- Define the group of boys and girls
def boys := {1, 2, 3, 4}
def girls := {1, 2, 3}

-- Problem 1: Prove the number of different selections of one boy and one girl is 12
theorem selection_count : boys.to_finset.card = 4 → girls.to_finset.card = 3 → (4 * 3 = 12) :=
by { intros, sorry }

-- Problem 2: Prove the number of different arrangements of two boys and two girls with the condition is 216
theorem arrangement_count : 
  boys.to_finset.card = 4 →
  girls.to_finset.card = 3 →
  (nat.choose 4 2) * (nat.choose 3 2) * 2! * nat.choose 3 2 = 216 :=
by { intros, sorry }

end selection_count_arrangement_count_l640_640118


namespace sphere_radius_in_tetrahedron_l640_640793

theorem sphere_radius_in_tetrahedron (a r : ℝ) :
  let P := {p : ℝ^3 | ∃ B M N : ℝ^3, p = B ∧ is_midpoint M (A, C) ∧ is_midpoint N (A, D) ∧ ∃ plane_P : plane, (B ∈ plane_P) ∧ (M ∈ plane_P) ∧ (N ∈ plane_P)} in
  (∀ (A B C D : ℝ^3), regular_tetrahedron A B C D a ∧ sphere_touches_lines A B C D r ∧ sphere_touches_plane P r) →
  (r = a * sqrt 2 / (5 + sqrt 11) ∨ r = a * sqrt 2 / (5 - sqrt 11)) :=
sorry

end sphere_radius_in_tetrahedron_l640_640793


namespace problem_N_calculation_l640_640398

def S : Finset ℕ := Finset.range 11

noncomputable def calculateN : ℕ :=
  ∑ x in S, (2 * x - 10) * 3^x

theorem problem_N_calculation : calculateN = 754152 := by
  sorry

end problem_N_calculation_l640_640398


namespace sum_product_coordinates_l640_640404

-- Given data points for the function f
def f : ℝ → ℝ
| 1 => 7
| 2 => 5
| 3 => 7
| 4 => 1 
| _ => 0 -- default value for all other inputs, for completeness

open Classical -- to enable classical logic, if necessary

noncomputable def g : ℝ → ℝ := f (f 4)

theorem sum_product_coordinates : 
  (4 * g 4) = 28 :=
by
  -- Since g 4 == f(f(4)) == f(1) == 7, we prove the sum of the product of coordinates
  -- Given only the guaranteed points
  sorry

end sum_product_coordinates_l640_640404


namespace ripe_oranges_after_73_days_l640_640250

theorem ripe_oranges_after_73_days (ripe_per_day : ℕ) (days : ℕ) : ripe_per_day = 5 → days = 73 → (ripe_per_day * days) = 365 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end ripe_oranges_after_73_days_l640_640250


namespace domain_of_f_l640_640685

-- Define the function f
def f (x : ℝ) : ℝ := real.sqrt (2 * x - 1)

-- Define the condition for the domain
theorem domain_of_f :
  ∀ x : ℝ, 2 * x - 1 ≥ 0 ↔ x ∈ set.Ici (1 / 2) :=
by sorry

end domain_of_f_l640_640685


namespace factorize_expression_l640_640179

variable (x y : ℝ)

theorem factorize_expression : 
  (y - 2 * x * y + x^2 * y) = y * (1 - x)^2 := 
by
  sorry

end factorize_expression_l640_640179


namespace simplification_l640_640777

theorem simplification (b : ℝ) : 3 * b * (3 * b^3 + 2 * b) - 2 * b^2 = 9 * b^4 + 4 * b^2 :=
by
  sorry

end simplification_l640_640777


namespace cos_sum_to_product_l640_640178

theorem cos_sum_to_product (a b : ℝ) : 
  cos (a + b) + cos (a - b) = 2 * cos a * cos b := 
  sorry

end cos_sum_to_product_l640_640178


namespace value_of_a_probability_ge_3_over_5_probability_in_interval_l640_640781

noncomputable def prob_dist (k : ℕ) : ℝ := match k with
  | 1 => a
  | 2 => 2 * a
  | 3 => 3 * a
  | 4 => 4 * a
  | 5 => 5 * a
  | _ => 0

axiom sum_of_probabilities_is_one : a * (1 + 2 + 3 + 4 + 5) = 1

theorem value_of_a : a = 1 / 15 := by
  sorry

theorem probability_ge_3_over_5 (a : ℝ) (h : a = 1 / 15) :
  prob_dist 3 + prob_dist 4 + prob_dist 5 = 4 / 5 := by
  sorry

theorem probability_in_interval (a : ℝ) (h : a = 1 / 15) :
  prob_dist 2 + prob_dist 3 = 1 / 3 := by
  sorry

end value_of_a_probability_ge_3_over_5_probability_in_interval_l640_640781


namespace plane_through_A_perpendicular_to_BC_l640_640085

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_between_points (P Q : Point3D) : Point3D :=
  { x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z }

def plane_eq (n : Point3D) (P : Point3D) (x y z : ℝ) : ℝ :=
  n.x * (x - P.x) + n.y * (y - P.y) + n.z * (z - P.z)

def A := Point3D.mk 0 (-2) 8
def B := Point3D.mk 4 3 2
def C := Point3D.mk 1 4 3

def n := vector_between_points B C
def plane := plane_eq n A

theorem plane_through_A_perpendicular_to_BC :
  ∀ x y z : ℝ, plane x y z = 0 ↔ -3 * x + y + z - 6 = 0 :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l640_640085


namespace correct_statements_l640_640963

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
hypothesis geometric_sequence : ∀ n, a (n + 1) = q * a n
hypothesis a1_gt_one : a 1 > 1
hypothesis a99_a100_gt_one : a 99 * a 100 - 1 > 0
hypothesis a99_a100_frac_lt_zero : (a 99 - 1) / (a 100 - 1) < 0

-- Definitions
def T (n : ℕ) := ∏ i in Finset.range n, a (i + 1)

-- Statements to be proven
theorem correct_statements :
  (0 < q ∧ q < 1) ∧
  (a 99 * a 101 - 1 < 0) ∧
  ¬ (∀ n, T n ≤ T 100) ∧
  ∃ n, n = 198 ∧ T n > 1 ∧ T (n + 1) < 1 :=
by
  sorry

end correct_statements_l640_640963


namespace problem1_problem2_l640_640994

-- Definitions and conditions
def A (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 1 }
def B : Set ℝ := { x | x < -6 ∨ x > 1 }

-- (Ⅰ) Problem statement: Prove that if A ∩ B = ∅, then -6 ≤ m ≤ 0.
theorem problem1 (m : ℝ) : A m ∩ B = ∅ ↔ -6 ≤ m ∧ m ≤ 0 := 
by
  sorry

-- (Ⅱ) Problem statement: Prove that if A ⊆ B, then m < -7 or m > 1.
theorem problem2 (m : ℝ) : A m ⊆ B ↔ m < -7 ∨ m > 1 := 
by
  sorry

end problem1_problem2_l640_640994


namespace polar_to_rectangular_range_of_m_l640_640291

-- Definition of the parametric equations of curve C
def curve (t : ℝ) : ℝ × ℝ :=
  (sqrt(3) * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line (rho theta m : ℝ) : Prop :=
  rho * sin (theta + π / 3) + m = 0

-- Definition of the rectangular equation of line l
def rectangular_line (x y m : ℝ) : Prop :=
  sqrt(3) * x + y + 2 * m = 0

-- Convert polar line equation to rectangular form
theorem polar_to_rectangular (rho theta m x y : ℝ)
  (h1 : polar_line rho theta m)
  (h2 : rho = sqrt (x^2 + y^2))
  (h3 : cos theta * rho = x ∧ sin theta * rho = y) :
  rectangular_line x y m :=
sorry

-- Range of values for m for which l intersects C
theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, (sqrt(3) * cos (2 * t), 2 * sin t) ∈ {p : ℝ × ℝ | rectangular_line p.1 p.2 m}) ↔
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end polar_to_rectangular_range_of_m_l640_640291


namespace average_weight_BC_l640_640467

-- Define the weights as variables
variables (A B C : ℝ)

-- Define the conditions
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (A + B) / 2 = 40
def condition3 : Prop := B = 31

-- The theorem to prove
theorem average_weight_BC (h1 : condition1) (h2 : condition2) (h3 : condition3) : (B + C) / 2 = 43 :=
sorry

end average_weight_BC_l640_640467


namespace tennis_tournament_matches_l640_640719

noncomputable def total_matches (players: ℕ) : ℕ :=
  players - 1

theorem tennis_tournament_matches :
  total_matches 104 = 103 :=
by
  sorry

end tennis_tournament_matches_l640_640719


namespace sum_of_odd_powers_l640_640970

variable (x y z a : ℝ) (k : ℕ)

theorem sum_of_odd_powers (h1 : x + y + z = a) (h2 : x^3 + y^3 + z^3 = a^3) (hk : k % 2 = 1) : 
  x^k + y^k + z^k = a^k :=
sorry

end sum_of_odd_powers_l640_640970


namespace max_constant_lambda_l640_640654

variable (n : ℕ) (a : ℕ → ℝ)
variable (n_ge_2 : 2 ≤ n)
variable (a_seq : ∀ i, 0 = a 0 ∧ a 0 ≤ a 1 ∧ ∀ j : ℕ, j < i → a j ≤ a (j+1))
variable (a_ineq : ∀ i, 1 ≤ i ∧ i ≤ n → 2 * a i ≥ a (i+1) + a (i-1))

theorem max_constant_lambda :
  ∃ λ, λ = n * (n + 1)^2 / 4 ∧ 
    ( (∑ i in Finset.range (n + 1), i * a i) ^ 2 
    ≥ λ * (∑ i in Finset.range (n + 1), (a i)^2) ) :=
  sorry

end max_constant_lambda_l640_640654


namespace part1_geometric_seq_part1_general_formula_part2_first_term_part2_recurrence_part2_general_formula_l640_640102

namespace SeqProblem

def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * a n + 1

def S : ℕ → ℕ
| 0       := 0
| (n + 1) := S n + a (n + 1)

theorem part1_geometric_seq (n : ℕ) :
  ∃ q : ℕ, a n + 1 = (a 1 + 1) * q ^ (n - 1) := 
sorry

theorem part1_general_formula (n : ℕ) :
  a n = 2 ^ n - 1 := 
sorry

theorem part2_first_term :
  a 1 = 6 :=
sorry

theorem part2_recurrence (n : ℕ) (h : n ≥ 2) :
  a n = 3 * a (n - 1) :=
sorry

theorem part2_general_formula (n : ℕ) :
  a n = 6 * 3^(n - 1) :=
sorry

end SeqProblem

end part1_geometric_seq_part1_general_formula_part2_first_term_part2_recurrence_part2_general_formula_l640_640102


namespace problem1_problem2_l640_640211

-- We define a point P(x, y) on the circle x^2 + y^2 = 2y.
variables {x y a : ℝ}

-- Condition for the point P to be on the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

-- Definition for 2x + y range
def range_2x_plus_y (x y : ℝ) : Prop := - Real.sqrt 5 + 1 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 5 + 1

-- Definition for the range of a given x + y + a ≥ 0
def range_a (x y a : ℝ) : Prop := x + y + a ≥ 0 → a ≥ Real.sqrt 2 - 1

-- Main statements to prove
theorem problem1 (hx : on_circle x y) : range_2x_plus_y x y := sorry

theorem problem2 (hx : on_circle x y) (h : ∀ θ, x = Real.cos θ ∧ y = 1 + Real.sin θ) : range_a x y a := sorry

end problem1_problem2_l640_640211


namespace sum_of_valid_n_l640_640920

def is_tasty_crossword (crossword : ℕ → ℕ → bool) : Prop :=
  ∀ i j, (i < 14 ∧ j < 14 ∧ 
  (crossword i j ∧ crossword (i+1) j ∧ crossword i (j+1) ∧ crossword (i+1) (j+1)))

def valid_n (n : ℕ) : Prop :=
  n % 4 = 0 ∧ n ≤ 196

theorem sum_of_valid_n :
  (∑ n in (finset.Icc 4 196).filter valid_n, n) = 4900 := by
  sorry

end sum_of_valid_n_l640_640920


namespace part1_part2_l640_640702

theorem part1 (a : ℝ) (h : a = log 4 3) : 2 ^ a + 2 ^ -a = 4 * real.sqrt 3 / 3 :=
by
  sorry

theorem part2 (x : ℝ) : (∃ x, log 2 (9 ^ (x - 1) - 5) = log 2 (3 ^ (x - 1) - 2) + 2) → x = 2 :=
by
  sorry

end part1_part2_l640_640702


namespace sum_T_values_l640_640755

-- Define S_n based on the parity of n
def S (n : ℕ) : ℤ := 
  if n % 2 = 0 then - n / 2
  else (n + 1) / 2

-- Define T_n using S_n and floor of the square root of n
def T (n : ℕ) : ℤ := 
  S n + Int.ofNat (Nat.floor (Real.sqrt n))

-- The theorem to prove that T_19 + T_21 + T_40 = 15
theorem sum_T_values : T 19 + T 21 + T 40 = 15 :=
by
  -- The proof is omitted as per the instructions
  sorry

end sum_T_values_l640_640755


namespace like_terms_exponents_l640_640700

theorem like_terms_exponents (n m : ℕ) (H : 2 * x ^ 2 * y ^ (m + 1) = -2 * x ^ n * y ^ 5) : -|n - m| = -2 :=
by
  have exponents_equal_x : n = 2 := by sorry -- From comparison of x exponents
  have exponents_equal_y : m + 1 = 5 := by sorry -- From comparison of y exponents
  have m_equals : m = 4 := by linarith -- Solving m + 1 = 5
  have abs_difference : |n - m| = 2 := by sorry -- Calculating |2 - 4|
  show -|n - m| = -2 from by {
    rw [abs_difference],
    refl,
  }
  sorry


end like_terms_exponents_l640_640700


namespace polar_to_rectangular_intersection_range_l640_640337

-- Definitions of parametric equations for curve C
def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line_l (ρ θ m: ℝ) : Prop := ρ * sin (θ + π / 3) + m = 0

-- Rectangular equation of line l
def rectangular_line_l (x y m: ℝ) : Prop := sqrt 3 * x + y + 2 * m = 0

-- Prove that polar equation of line l can be transformed to rectangular form
theorem polar_to_rectangular (ρ θ m : ℝ) :
  (polar_line_l ρ θ m) → (∃ x y, ρ = sqrt (x^2 + y^2) ∧ θ = arctan (y / x) ∧ rectangular_line_l x y m) :=
  sorry

-- Prove that the range of values for m ensuring line l intersects curve C is [-19/12, 5/2]
theorem intersection_range (m : ℝ) :
  (∃ t : ℝ, rectangular_line_l (sqrt 3 * cos (2 * t)) (2 * sin t) m) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
  sorry

end polar_to_rectangular_intersection_range_l640_640337


namespace cos_alpha_minus_beta_eq_sin_alpha_eq_l640_640997

-- First proof problem: Prove that cos(α - β) = 3/5.
theorem cos_alpha_minus_beta_eq (α β : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (Real.cos α, Real.sin α))
  (hb : b = (Real.cos β, Real.sin β))
  (hab : Real.norm (a.1 - b.1, a.2 - b.2) = (2/5) * Real.sqrt 5) :
  Real.cos (α - β) = 3/5 :=
sorry

-- Second proof problem: Prove that sin α = 33/65.
theorem sin_alpha_eq (α β : ℝ)
  (hα : 0 < α ∧ α < π/2)
  (hβ : -π/2 < β ∧ β < 0)
  (sin_β : Real.sin β = -5/13) 
  (cos_alpha_minus_beta : Real.cos (α - β) = 3/5) :
  Real.sin α = 33/65 :=
sorry

end cos_alpha_minus_beta_eq_sin_alpha_eq_l640_640997


namespace janet_earnings_per_hour_l640_640736

theorem janet_earnings_per_hour :
  let text_posts := 150
  let image_posts := 80
  let video_posts := 20
  let rate_text := 0.25
  let rate_image := 0.30
  let rate_video := 0.40
  text_posts * rate_text + image_posts * rate_image + video_posts * rate_video = 69.50 :=
by
  sorry

end janet_earnings_per_hour_l640_640736


namespace find_h_inv_k_of_12_l640_640904

variables {α β : Type} [real_point α] [real_point β]

def h : α → α
def k : α → α
noncomputable def h_inv : α → α
noncomputable def k_inv : α → α

axiom inverse_property : ∀ z : α, k_inv (h z) = 7 * z - 4

theorem find_h_inv_k_of_12 : h_inv (k 12) = (16 : α) / 7 :=
by
  sorry

end find_h_inv_k_of_12_l640_640904


namespace square_pentagon_area_l640_640447

theorem square_pentagon_area
  (A B C D F : Point)
  (AF_perp_FD : is_perpendicular (segment A F) (segment F D))
  (AF_length : distance A F = 12)
  (FD_length : distance F D = 9)
  (ABCD_square : is_square A B C D) :
  area_of_pentagon A F D C B = 171 := 
sorry

end square_pentagon_area_l640_640447


namespace sum_of_valid_x_l640_640078

theorem sum_of_valid_x :
  let mean := (3 + 5 + 7 + 15 + x) / 5,
      sorted_list := (List.qsort (.≤.) [3, 5, 7, 15, x]),
      median := sorted_list.nth (sorted_list.length / 2),
      is_integer := ∃ (x : ℤ), median = mean → True
  mean = 5 :=
sorry

end sum_of_valid_x_l640_640078


namespace tetrahedron_OABC_volume_zero_l640_640824

theorem tetrahedron_OABC_volume_zero :
  ∃ (a b c : ℝ), (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧
  (a^2 + b^2 = 36) ∧ (b^2 + c^2 = 64) ∧ (c^2 + a^2 = 100) ∧
  let V := (1 / 6) * a * b * c in V = 0 :=
sorry

end tetrahedron_OABC_volume_zero_l640_640824


namespace sally_cut_red_orchids_l640_640496

-- Definitions and conditions
def initial_red_orchids := 9
def orchids_in_vase_after_cutting := 15

-- Problem statement
theorem sally_cut_red_orchids : (orchids_in_vase_after_cutting - initial_red_orchids) = 6 := by
  sorry

end sally_cut_red_orchids_l640_640496


namespace workers_needed_l640_640262

/-- 
Given that 7 workers can build 7 cars in 9 days,
prove that the number of workers required to build 9 cars in 9 days is 9.
-/
theorem workers_needed (w : ℕ) : 
  (7 * 9) / 9 = 7 → (9 * 9) / 9 = w → w = 9 :=
by
  intro h1 h2
  rw [mul_div_cancel_left] at h1
  rw [mul_div_cancel_left] at h2
  have h1' : 7 = 7 := eq.symm h1
  have h2' : w = 9 := eq.trans h2 rfl
  exact h2'

end workers_needed_l640_640262


namespace factorial_division_l640_640663

theorem factorial_division (h : 10! = 3628800) : 10! / 4! = 151200 := by
  sorry

end factorial_division_l640_640663


namespace red_higher_than_blue_l640_640129

noncomputable def toss_probability_dist (k : ℕ) : ℝ :=
  if k = 1 then 1 / 2 else 1 / 2 ^ k

theorem red_higher_than_blue :
  let higher_than := ∀ (k1 k2 : ℕ), toss_probability_dist k1 > toss_probability_dist k2 → k1 > k2,
  probability higher_than = 5 / 16 :=
sorry

end red_higher_than_blue_l640_640129


namespace D_working_alone_completion_time_l640_640097

variable (A_rate D_rate : ℝ)
variable (A_job_hours D_job_hours : ℝ)

-- Conditions
def A_can_complete_in_15_hours : Prop := (A_job_hours = 15)
def A_and_D_together_complete_in_10_hours : Prop := (1/A_rate + 1/D_rate = 10)

-- Proof statement
theorem D_working_alone_completion_time
  (hA : A_job_hours = 15)
  (hAD : 1/A_rate + 1/D_rate = 10) :
  D_job_hours = 30 := sorry

end D_working_alone_completion_time_l640_640097


namespace chess_tournament_solution_l640_640717

theorem chess_tournament_solution (n : ℕ) 
  (w_count : n > 0) 
  (m_count : 2 * n > 0)
  (total_games : (1/2) * n * (n - 1) + 2 * n^2 + n * (2 * n - 1)) 
  (ratio_75 : ∃ N F : ℕ, N + F = 2 * n^2 ∧ 5 * ((1/2) * n * (n - 1) + N) = 7 * (n * (2 * n - 1) + F)) 
  : n = 3 :=
by
  sorry

end chess_tournament_solution_l640_640717


namespace right_angle_OQP_l640_640392

variable {A B C D O P Q : Type}
variable [quadrilateral_center (ABCD : quadrilateral) O]
variable [diagonals_intersect (AC BD : diagonal) P]
variable [circumscribed_circles_intersect (ABP CDP : triangle) Q]

theorem right_angle_OQP 
    (h1 : quadrilateral_center ABCD O)
    (h2 : diagonals_intersect AC BD P)
    (h3 : circumscribed_circles_intersect ABP CDP Q) :
    angle O Q P = 90 := 
sorry

end right_angle_OQP_l640_640392


namespace factorial_division_l640_640658

theorem factorial_division : ∀ (a b : ℕ), a = 10! ∧ b = 4! → a / b = 151200 :=
by
  intros a b h
  cases h with ha hb
  rw [ha, Nat.factorial, Nat.factorial] at hb,
  norm_num at hb,
  exact sorry

end factorial_division_l640_640658


namespace value_added_after_doubling_l640_640713

theorem value_added_after_doubling (x v : ℝ) (h1 : x = 4) (h2 : 2 * x + v = x / 2 + 20) : v = 14 :=
by
  sorry

end value_added_after_doubling_l640_640713


namespace extended_lattice_num_equilateral_triangles_l640_640286

-- Definitions based on the conditions
def Lattice := { pts : Finset (ℤ × ℤ) // 
  ∀ p ∈ pts, ∃ d : ℤ × ℤ, 
    let dist := fun (x y : ℤ × ℤ) => ((x.1 - y.1)^2 + (x.2 - y.2)^2) in 
    dist p d = 1}
 
def InnerHexagon := { pts : Finset (ℤ × ℤ) // 
  ∀ p ∈ pts, ∃ q ∈ pts, ∃ r ∈ pts,
    let dist := fun (x y : ℤ × ℤ) => ((x.1 - y.1)^2 + (x.2 - y.2)^2) in 
    dist p q = 1 ∧ dist q r = 1 ∧ dist r p = 1}

def OuterHexagon := { pts : Finset (ℤ × ℤ) // 
  ∀ p ∈ pts, ∃ q ∈ pts, ∃ r ∈ pts,
    let dist := fun (x y : ℤ × ℤ) => ((x.1 - y.1)^2 + (x.2 - y.2)^2) in 
    dist p q = 4 ∧ dist q r = 4 ∧ dist r p = 4}

-- Main proof statement
theorem extended_lattice_num_equilateral_triangles 
  (lattice : Lattice) 
  (innerHex : InnerHexagon) 
  (outerHex : OuterHexagon) : 
  ∃ n : ℕ, n = 20 :=
by sorry

end extended_lattice_num_equilateral_triangles_l640_640286


namespace unique_solution_l640_640592

def s (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem unique_solution (m n : ℕ) (h : n * (n + 1) = 3 ^ m + s n + 1182) : (m, n) = (0, 34) :=
by
  sorry

end unique_solution_l640_640592


namespace find_certain_number_l640_640862

-- Definition of the conditions
def a := 3 * 13
def b := 3 * 14
def c := 3 * 17
def sum := a + b + c
def target := 143

-- The statement of the proof problem
theorem find_certain_number (x : ℕ) : sum + x = target → x = 11 :=
by
  unfold a b c sum target
  sorry

end find_certain_number_l640_640862


namespace total_points_correct_l640_640386

def num_white_mallows_dad := 21
def num_pink_mallows_joe := 4 * num_white_mallows_dad
def num_blue_mallows_mom := 3 * num_pink_mallows_joe
def num_white_mallows_sis := num_white_mallows_dad / 2

def roasted_white_mallows_dad := num_white_mallows_dad / 3
def roasted_pink_mallows_joe := num_pink_mallows_joe / 2
def roasted_blue_mallows_mom := num_blue_mallows_mom / 4
def roasted_white_mallows_sis := num_white_mallows_sis * (2 / 3)

def points_dad := roasted_white_mallows_dad * 1
def points_joe := roasted_pink_mallows_joe * 2
def points_mom := roasted_blue_mallows_mom * 3
def points_sis := roasted_white_mallows_sis * 1

def total_roasted_points := points_dad + points_joe + points_mom + points_sis

theorem total_points_correct : total_roasted_points = 286 := by
    unfold total_roasted_points
    unfold points_dad points_joe points_mom points_sis
    unfold roasted_white_mallows_dad roasted_pink_mallows_joe roasted_blue_mallows_mom roasted_white_mallows_sis
    unfold num_white_mallows_dad num_pink_mallows_joe num_blue_mallows_mom num_white_mallows_sis
    calc
      7 + 84 + 189 + 6 = 286
    sorry

end total_points_correct_l640_640386


namespace total_area_is_infinite_l640_640285

noncomputable def circle_radius (n : ℕ) : ℝ :=
  if n = 0 then 2
  else 2 / 2 ^ n

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  π * r ^ 2

def main_areas_sum : ℝ :=
  ∑' n, area_of_circle (circle_radius n)

noncomputable def extra_circle_area : ℝ :=
  area_of_circle (1 / 3)

noncomputable def total_extra_areas_sum : ℝ :=
  ∑' n, extra_circle_area

theorem total_area_is_infinite : main_areas_sum + total_extra_areas_sum = ∞ := by
  sorry

end total_area_is_infinite_l640_640285


namespace behavior_on_1_3_l640_640218

open Real

noncomputable def f : ℝ → ℝ := sorry -- Assume f(x) is an even function which satisfies the given conditions

lemma even_function (x : ℝ) : f x = f (-x) := sorry

lemma periodic_function (x : ℝ) : f (x + 1) = -f x := sorry

lemma decreasing_on_interval : ∀ (x y : ℝ), -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x > f y := sorry

theorem behavior_on_1_3 :
  (∀ (x y : ℝ), 1 ≤ x ∧ x < y ∧ y ≤ 2 → f x > f y) ∧
  (∀ (x y : ℝ), 2 ≤ x ∧ x < y ∧ y ≤ 3 → f x < f y) :=
begin
  sorry
end

end behavior_on_1_3_l640_640218


namespace tangent_line_at_x_2_range_of_S_l640_640481

section
variable {a x y m n S : ℝ}
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x - a / x - 2 * log x

-- Prove that the tangent line equation to the curve y = f(x, 2) at x = 2 is 3x - 2y - 4ln 2 = 0
theorem tangent_line_at_x_2 (h_a : a = 2) : 
  tangent_line_eq (λ x. f x 2) 2 = "3x - 2y - 4ln 2 = 0" := sorry

-- Prove the range of S, where S is the difference between the maximum and minimum values of f(x), given a > 2e/(e^2+1)
theorem range_of_S (h_a : a > 2 * exp(1) / (exp(1)^2 + 1)) :
  ∃ (m n : ℝ), S = m - n ∧ 0 < S ∧ S < 8 / (exp(1)^2 + 1) := sorry
end

end tangent_line_at_x_2_range_of_S_l640_640481


namespace addition_and_rounding_l640_640563

-- Define the two numbers
def num1 : ℝ := 56.238
def num2 : ℝ := 75.914

-- Define their sum
def sum := num1 + num2

-- Define the rounding functionality to the nearest thousandth
def round_to_thousandth (x : ℝ) : ℝ :=
  let factor := 1000
  (Real.round (x * factor)) / factor

-- Theorem statement
theorem addition_and_rounding : round_to_thousandth sum = 132.152 :=
by
  -- Insert proof here, currently omitted.
  sorry

end addition_and_rounding_l640_640563


namespace estimated_probability_is_2_div_9_l640_640866

def groups : List (List ℕ) :=
  [[3, 4, 3], [4, 3, 2], [3, 4, 1], [3, 4, 2], [2, 3, 4], [1, 4, 2], [2, 4, 3], [3, 3, 1], [1, 1, 2],
   [3, 4, 2], [2, 4, 1], [2, 4, 4], [4, 3, 1], [2, 3, 3], [2, 1, 4], [3, 4, 4], [1, 4, 2], [1, 3, 4]]

def count_desired_groups (gs : List (List ℕ)) : Nat :=
  gs.foldl (fun acc g =>
    if g.contains 1 ∧ g.contains 2 ∧ g.length ≥ 3 then acc + 1 else acc) 0

theorem estimated_probability_is_2_div_9 :
  (count_desired_groups groups) = 4 →
  4 / 18 = 2 / 9 :=
by
  intro h
  sorry

end estimated_probability_is_2_div_9_l640_640866


namespace sum_s_k_l640_640416

-- Definition of sk
def s_k (k : ℕ) : ℚ :=
  if k > 0 then 1 / 2 * abs (1 / k - 1 / (k + 1))
  else 0

-- The main theorem statement
theorem sum_s_k :
  (\sum k in Finset.range 2006, s_k (k + 1)) = 1003 / 2007 :=
by
  sorry

end sum_s_k_l640_640416


namespace maple_trees_remaining_l640_640495

-- Define the initial number of maple trees in the park
def initial_maple_trees : ℝ := 9.0

-- Define the number of maple trees that will be cut down
def cut_down_maple_trees : ℝ := 2.0

-- Define the expected number of maple trees left after cutting down
def remaining_maple_trees : ℝ := 7.0

-- Theorem to prove the remaining number of maple trees is correct
theorem maple_trees_remaining :
  initial_maple_trees - cut_down_maple_trees = remaining_maple_trees := by
  admit -- sorry can be used alternatively

end maple_trees_remaining_l640_640495


namespace tan_alpha_value_trigonometric_expression_value_l640_640955

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.sin α = (2 * Real.sqrt 5) / 5) : 
  Real.tan α = 2 :=
sorry

theorem trigonometric_expression_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.sin α = (2 * Real.sqrt 5) / 5) : 
  (4 * Real.sin (π - α) + 2 * Real.cos (2 * π - α)) / (Real.sin (π / 2 - α) + Real.sin (-α)) = -10 := 
sorry

end tan_alpha_value_trigonometric_expression_value_l640_640955


namespace rect_eq_and_range_of_m_l640_640309

noncomputable def rect_eq_of_polar (m : ℝ) : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ * sin (θ + π / 3) + m = 0 → sqrt 3 * (ρ * cos θ) + (ρ * sin θ) + 2 * m = 0

noncomputable def range_of_m_for_intersection : set ℝ := 
  {m : ℝ | -19 / 12 ≤ m ∧ m ≤ 5 / 2}

theorem rect_eq_and_range_of_m (m : ℝ) : 
  rect_eq_of_polar m ∧ (∀ t : ℝ, ∃ x y : ℝ, x = sqrt 3 * cos (2 * t) ∧ y = 2 * sin t ∧ 
    (sqrt 3 * x + y + 2 * m = 0) → m ∈ range_of_m_for_intersection) :=
by sorry

end rect_eq_and_range_of_m_l640_640309


namespace find_m_value_l640_640709

/-- 
If the function y = (m + 1)x^(m^2 + 3m + 4) is a quadratic function, 
then the value of m is -2.
--/
theorem find_m_value 
  (m : ℝ)
  (h1 : m^2 + 3 * m + 4 = 2)
  (h2 : m + 1 ≠ 0) : 
  m = -2 := 
sorry

end find_m_value_l640_640709


namespace relationship_radii_distance_l640_640645

theorem relationship_radii_distance (R r d : ℝ) (h : 0 < r) (h1 : 0 < R) (h2 : 
    ∀ A B C D : ℝ, 
    inscribed_in_circle A B C D S ∧ circumscribed_around_circle A B C D s ∧
    centers_distance_eq S s d → 
    (1 / (R + d)^2 + 1 / (R - d)^2 = 1 / r^2)) : 
    1 / (R + d)^2 + 1 / (R - d)^2 = 1 / r^2 :=
sorry

end relationship_radii_distance_l640_640645


namespace domain_when_a_is_four_range_of_a_if_f_greater_equals_2_l640_640684

-- Definition of the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb 2 (abs (2 * x - 1) + abs (x + 2) - a)

-- Proving the domain when a=4
theorem domain_when_a_is_four : 
  ∀ x, (4 < abs (2 * x - 1) + abs (x + 2)) ↔ (x ∈ (-∞, -1) ∪ (1, ∞)) :=
sorry

-- Definition for part 2
noncomputable def g (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x + 2) - 4

-- Proving the range of 'a' provided f(x) ≥ 2 for all x in ℝ
theorem range_of_a_if_f_greater_equals_2 :
  (∀ x : ℝ, f x a ≥ 2) ↔ (a ∈ (-∞, -3/2]) :=
sorry

end domain_when_a_is_four_range_of_a_if_f_greater_equals_2_l640_640684


namespace second_guldins_theorem_l640_640445

-- Define conditions given in the problem
variables (S : ℝ) (V : ℝ) (z : ℝ) 
variables (S_n : ℕ → ℝ) (V_n : ℕ → ℝ) (z_n : ℕ → ℝ)

-- The statement representing the proof problem
theorem second_guldins_theorem 
  (h1 : ∀ n, V_n n = 2 * Real.pi * z_n n * S_n n)
  (h2 : Tendsto S_n atTop (𝓝 S))
  (h3 : Tendsto V_n atTop (𝓝 V))
  (h4 : Tendsto z_n atTop (𝓝 z)) :
  V = 2 * Real.pi * z * S :=
by
  sorry  -- Proof is omitted

end second_guldins_theorem_l640_640445


namespace solve_for_x_l640_640453

-- Define the equation
def equation (x : ℝ) : Prop := (x^2 + 3 * x + 4) / (x + 5) = x + 6

-- Prove that x = -13 / 4 satisfies the equation
theorem solve_for_x : ∃ x : ℝ, equation x ∧ x = -13 / 4 :=
by
  sorry

end solve_for_x_l640_640453


namespace smallest_x_for_multiple_of_720_l640_640076

theorem smallest_x_for_multiple_of_720 (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 720 = 2^4 * 3^2 * 5^1) : x = 8 ↔ (450 * x) % 720 = 0 :=
by
  sorry

end smallest_x_for_multiple_of_720_l640_640076


namespace election_votes_l640_640039

noncomputable def third_candidate_votes (total_votes first_candidate_votes second_candidate_votes : ℕ) (winning_fraction : ℚ) : ℕ :=
  total_votes - (first_candidate_votes + second_candidate_votes)

theorem election_votes :
  ∃ total_votes : ℕ, 
  ∃ first_candidate_votes : ℕ,
  ∃ second_candidate_votes : ℕ,
  ∃ winning_fraction : ℚ,
  first_candidate_votes = 5000 ∧ 
  second_candidate_votes = 15000 ∧ 
  winning_fraction = 2/3 ∧ 
  total_votes = 60000 ∧ 
  third_candidate_votes total_votes first_candidate_votes second_candidate_votes winning_fraction = 40000 :=
    sorry

end election_votes_l640_640039


namespace polar_to_rectangular_range_of_m_l640_640293

-- Definition of the parametric equations of curve C
def curve (t : ℝ) : ℝ × ℝ :=
  (sqrt(3) * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line (rho theta m : ℝ) : Prop :=
  rho * sin (theta + π / 3) + m = 0

-- Definition of the rectangular equation of line l
def rectangular_line (x y m : ℝ) : Prop :=
  sqrt(3) * x + y + 2 * m = 0

-- Convert polar line equation to rectangular form
theorem polar_to_rectangular (rho theta m x y : ℝ)
  (h1 : polar_line rho theta m)
  (h2 : rho = sqrt (x^2 + y^2))
  (h3 : cos theta * rho = x ∧ sin theta * rho = y) :
  rectangular_line x y m :=
sorry

-- Range of values for m for which l intersects C
theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, (sqrt(3) * cos (2 * t), 2 * sin t) ∈ {p : ℝ × ℝ | rectangular_line p.1 p.2 m}) ↔
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end polar_to_rectangular_range_of_m_l640_640293


namespace problem_part1_problem_part2_l640_640300

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l_rect_eq (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (sqrt 3 * x + y + 2 * m = 0)

def line_l_range_m (m : ℝ) : Prop :=
  -19/12 ≤ m ∧ m ≤ 5/2

theorem problem_part1 (m : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * sin (θ + π / 3) + m = 0) ↔ line_l_rect_eq m := 
sorry

theorem problem_part2 :
  ∀ m : ℝ, 
  (∃ t : ℝ, line_l_rect_eq m (curve_C t).1 (curve_C t).2) ↔ line_l_range_m m := 
sorry

end problem_part1_problem_part2_l640_640300


namespace common_difference_is_one_l640_640571

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Conditions given in the problem
axiom h1 : a 1 ^ 2 + a 10 ^ 2 = 101
axiom h2 : a 5 + a 6 = 11
axiom h3 : ∀ n m, n < m → a n < a m
noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n+1) = a n + d

-- Theorem stating the common difference d is 1
theorem common_difference_is_one : is_arithmetic_sequence a d → d = 1 := 
by
  sorry

end common_difference_is_one_l640_640571


namespace average_first_15_nat_l640_640534

-- Define the sequence and necessary conditions
def sum_first_n_nat (n : ℕ) : ℕ := n * (n + 1) / 2

theorem average_first_15_nat : (sum_first_n_nat 15) / 15 = 8 := 
by 
  -- Here we shall place the proof to show the above statement holds true
  sorry

end average_first_15_nat_l640_640534


namespace hexagon_perimeter_l640_640837

-- Define a regular hexagon and its properties
def is_regular_hexagon (hexagon : Type) (side_length : ℕ) : Prop :=
  ∃ sides : list ℕ, sides.length = 6 ∧ ∀ s ∈ sides, s = side_length

-- Define the perimeter function for a hexagon
def perimeter_of_hexagon (hexagon : Type) (side_length : ℕ) : ℕ :=
  6 * side_length

-- State the theorem
theorem hexagon_perimeter (hexagon : Type) (side_length : ℕ) (h : is_regular_hexagon hexagon 5) : perimeter_of_hexagon hexagon 5 = 30 :=
by
  sorry

end hexagon_perimeter_l640_640837


namespace rateA_is_40_l640_640770

-- Definitions of the conditions
def rateB : ℝ := 30 -- Pipe B's rate in liters per minute
def rateC : ℝ := -20 -- Pipe C's rate (negative because it drains water) in liters per minute
def tankCapacity : ℝ := 900 -- Tank capacity in liters
def totalTime : ℝ := 54 -- Total time in minutes
def cycleTime : ℝ := 3 -- Time for one complete cycle in minutes
def cycles : ℕ := (totalTime / cycleTime).toNat -- Number of cycles in totalTime

-- Definition of the goal
def rateA (A : ℝ) : Prop :=
  cycles * (A + rateB + rateC) = tankCapacity

-- The theorem to be proved
theorem rateA_is_40 : rateA 40 :=
by
  unfold rateA
  simp [rateB, rateC, tankCapacity, totalTime, cycleTime, cycles]
  sorry

end rateA_is_40_l640_640770


namespace trapezoid_area_l640_640475

theorem trapezoid_area 
  (diagonals_perpendicular : ∀ A B C D : ℝ, (A ≠ B → C ≠ D → A * C + B * D = 0)) 
  (diagonal_length : ∀ B D : ℝ, B ≠ D → (B - D) = 17) 
  (height_of_trapezoid : ∀ (height : ℝ), height = 15) : 
  ∃ (area : ℝ), area = 4335 / 16 := 
sorry

end trapezoid_area_l640_640475


namespace cans_collected_is_232_l640_640014

-- Definitions of the conditions
def total_students : ℕ := 30
def half_students : ℕ := total_students / 2
def cans_per_half_student : ℕ := 12
def remaining_students : ℕ := 13
def cans_per_remaining_student : ℕ := 4

-- Calculate total cans collected
def total_cans_collected : ℕ := (half_students * cans_per_half_student) + (remaining_students * cans_per_remaining_student)

-- The theorem to be proved
theorem cans_collected_is_232 : total_cans_collected = 232 := by
  -- Proof would go here
  sorry

end cans_collected_is_232_l640_640014


namespace suv_max_distance_l640_640895

theorem suv_max_distance:
  ∀ (highway_mpg city_mpg gallons: ℝ), 
  highway_mpg = 12.2 → city_mpg = 7.6 → gallons = 22 → 
  highway_mpg * gallons = 268.4 :=
by
  intros highway_mpg city_mpg gallons h_mpgh h_mpgc h_gal
  rw [h_mpgh, h_gal]
  norm_num
  sorry

end suv_max_distance_l640_640895


namespace smallest_n_no_real_roots_l640_640925

theorem smallest_n_no_real_roots :
  ∃ n : ℤ, (∀ x : ℝ, (3 * n - 2) * x^2 + 9 * x - 9 ≠ 0) ∧ ∀ m, (3 * m - 2 < 3 * n - 2 → m < n) :=
begin
  sorry
end

end smallest_n_no_real_roots_l640_640925


namespace combin_sum_l640_640152

def combin (n m : ℕ) : ℕ := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem combin_sum (n : ℕ) (h₁ : n = 99) : combin n 2 + combin n 3 = 161700 := by
  sorry

end combin_sum_l640_640152


namespace total_cats_in_academy_l640_640577

theorem total_cats_in_academy (cats_jump cats_jump_fetch cats_fetch cats_fetch_spin cats_spin cats_jump_spin cats_all_three cats_none: ℕ)
  (h_jump: cats_jump = 60)
  (h_jump_fetch: cats_jump_fetch = 20)
  (h_fetch: cats_fetch = 35)
  (h_fetch_spin: cats_fetch_spin = 15)
  (h_spin: cats_spin = 40)
  (h_jump_spin: cats_jump_spin = 22)
  (h_all_three: cats_all_three = 11)
  (h_none: cats_none = 10) :
  cats_all_three + (cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + (cats_jump_spin - cats_all_three) +
  (cats_jump - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_fetch - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_spin - ((cats_jump_spin - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) + cats_none = 99 :=
by
  calc 
  cats_all_three + (cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + (cats_jump_spin - cats_all_three) +
  (cats_jump - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_fetch - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_spin - ((cats_jump_spin - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) + cats_none 
  = 11 + (20 - 11) + (15 - 11) + (22 - 11) + (60 - (9 + 11 + 11)) + (35 - (9 + 4 + 11)) + (40 - (11 + 4 + 11)) + 10 
  := by sorry
  _ = 99 := by sorry

end total_cats_in_academy_l640_640577


namespace matrix_determinant_l640_640157

variable (a b c d : ℤ)

theorem matrix_determinant : 
  det (Matrix ![[8, 4], [-2, 3]]) = 32 :=
by
  sorry

end matrix_determinant_l640_640157


namespace scientific_notation_of_0_0000025_l640_640100

theorem scientific_notation_of_0_0000025 :
  0.0000025 = 2.5 * 10^(-6) :=
by
  sorry

end scientific_notation_of_0_0000025_l640_640100


namespace inclination_angle_of_line_l640_640260

theorem inclination_angle_of_line (p1 p2 : Real × Real)
  (h1 : p1 = (1, 2))
  (h2 : p2 = (4, 2 + Real.sqrt 3)) :
  ∃ θ : ℝ, θ = 30 ∧
  ∀ (k : ℝ), k = (p2.2 - p1.2) / (p2.1 - p1.1) -> θ = Real.atan k * 180 / Real.pi :=
by
  sorry

end inclination_angle_of_line_l640_640260


namespace rectangular_eq_of_line_l_range_of_m_l640_640347

noncomputable def curve_C_param_eq (t : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l_polar_eq (θ ρ m : ℝ) : Prop :=
  ρ * Real.sin(θ + Real.pi / 3) + m = 0

theorem rectangular_eq_of_line_l (x y m : ℝ) (h : line_l_polar_eq (Real.atan2 y x) (Real.sqrt (x^2 + y^2)) m) :
  sqrt 3 * x + y + 2 * m = 0 :=
  sorry

theorem range_of_m (m : ℝ) (t : ℝ) (h : ∃ (t : ℝ), curve_C_param_eq t ∈ set_of (λ p, p.2 = -√3 * p.1 - 2 * m))
  : -19/12 ≤ m ∧ m ≤ 5/2 :=
  sorry

end rectangular_eq_of_line_l_range_of_m_l640_640347


namespace intersect_rectangular_eqn_range_of_m_l640_640358

-- Definitions of the parametric and polar equations
def curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Proving the rectangular equation and range of m for the intersection of l and C
theorem intersect_rectangular_eqn (m t : ℝ) :
  (∃ ρ θ, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  ∃ x y, (x, y) = curve_C t ∧ √3 * x + y + 2 * m = 0 :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ ρ θ t, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end intersect_rectangular_eqn_range_of_m_l640_640358


namespace total_money_in_euros_correct_l640_640251

noncomputable def total_money_in_euros : ℝ := 
  let henry_initial := 5.50
  let henry_earning := 2.75
  let simon_initial := 13.30
  let simon_spent := 0.25 * simon_initial
  let olivia_mult := 2
  let david_mult := 1 / 3
  let conversion_rate := 0.85
  let henry_total := henry_initial + henry_earning
  let simon_remaining := simon_initial - simon_spent
  let olivia_total := olivia_mult * henry_total
  let david_total := olivia_total - (david_mult * olivia_total)
  let total_dollars := henry_total + simon_remaining + olivia_total + david_total
  total_dollars * conversion_rate

theorem total_money_in_euros_correct : total_money_in_euros = 38.87 :=
by
  rw [total_money_in_euros]
  let henry_initial := 5.50
  let henry_earning := 2.75
  let simon_initial := 13.30
  let simon_spent := 0.25 * simon_initial
  let olivia_mult := 2
  let david_mult := 1 / 3
  let conversion_rate := 0.85
  let henry_total := henry_initial + henry_earning
  let simon_remaining := simon_initial - simon_spent
  let olivia_total := olivia_mult * henry_total
  let david_total := olivia_total - (david_mult * olivia_total)
  let total_dollars := henry_total + simon_remaining + olivia_total + david_total
  have : total_dollars = 45.73 := by
    sorry -- individual calculations are omitted for this example
  show total_dollars * conversion_rate = 38.87
  calc
    total_dollars * conversion_rate = 45.73 * conversion_rate : by rw [this]
    ... = 45.73 * 0.85 : rfl
    ... = 38.8705 : by norm_num
    ... = 38.87 : by norm_num

end total_money_in_euros_correct_l640_640251


namespace probability_two_out_of_four_germinate_l640_640585

theorem probability_two_out_of_four_germinate :
  let p := 4 / 5
  let C (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  let P (n k : ℕ) (p : ℚ) := C n k * p^k * (1 - p)^(n - k)
  P 4 2 p = 96 / 625 :=
by
  let p : ℚ := 4 / 5
  let C := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial (4 - 2))
  let P := C * p^2 * (1 - p)^2
  have hC : C = 6 := by -- combination step
    simp [Nat.factorial]
    linarith
  have hP : P = 6 * (4/5)^2 * (1 - 4/5)^2 := by
    rw hC
    norm_num
  rw hP
  norm_num
  exact (show 96 / 625 = 96 / 625 by rfl)

end probability_two_out_of_four_germinate_l640_640585


namespace fraction_of_men_collected_dues_l640_640883

theorem fraction_of_men_collected_dues
  (M W : ℕ)
  (x : ℚ)
  (h1 : 45 * x * M + 5 * W = 17760)
  (h2 : M + W = 3552)
  (h3 : 1 / 12 * W = W / 12) :
  x = 1 / 9 :=
by
  -- Proof steps would go here
  sorry

end fraction_of_men_collected_dues_l640_640883


namespace simplify_floor_expression_l640_640221

theorem simplify_floor_expression (n : ℕ) (h_pos : n > 0) :
  1 + Real.floor ((7 + 4 * Real.sqrt 3) ^ n) - (7 + 4 * Real.sqrt 3) ^ n = (7 - 4 * Real.sqrt 3) ^ n :=
sorry

end simplify_floor_expression_l640_640221


namespace count_a_satisfying_conditions_l640_640220

theorem count_a_satisfying_conditions : 
  (∃ a : ℕ, (101 ∣ a) ∧ 
    (∃ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ 99 ∧ a = 10^j - 10^i)) = 1200 :=
sorry

end count_a_satisfying_conditions_l640_640220


namespace increment_M0_to_M1_increment_M0_to_M2_increment_M0_to_M3_l640_640923

-- Define the function z = x * y
def z (x y : ℝ) : ℝ := x * y

-- Initial point M0
def M0 : ℝ × ℝ := (1, 2)

-- Points to which we move
def M1 : ℝ × ℝ := (1.1, 2)
def M2 : ℝ × ℝ := (1, 1.9)
def M3 : ℝ × ℝ := (1.1, 2.2)

-- Proofs for the increments
theorem increment_M0_to_M1 : z M1.1 M1.2 - z M0.1 M0.2 = 0.2 :=
by sorry

theorem increment_M0_to_M2 : z M2.1 M2.2 - z M0.1 M0.2 = -0.1 :=
by sorry

theorem increment_M0_to_M3 : z M3.1 M3.2 - z M0.1 M0.2 = 0.42 :=
by sorry

end increment_M0_to_M1_increment_M0_to_M2_increment_M0_to_M3_l640_640923


namespace min_square_side_length_l640_640656

theorem min_square_side_length {a b : ℝ} (h1 : a > b) (h2 : b > 0) :
  let min_side := if a < (Real.sqrt 2 + 1) * b then a else (Real.sqrt 2 / 2) * (a + b)
  in ∀ s : ℝ, side_length := if a < (Real.sqrt 2 + 1) * b then a else (Real.sqrt 2 / 2) * (a + b)
  in squared_side := 
  if h : a < (Real.sqrt 2 + 1) * b 
  then s == a ∨ s == (Real.sqrt 2 / 2) * (a + b)

#eval min_square_side_length sorry

end min_square_side_length_l640_640656


namespace calculate_a_minus_b_l640_640259

theorem calculate_a_minus_b
 (a b : ℚ)
 (h : ∀ x : ℚ, x > 0 → (a / (10^x - 1) + b / (10^x + 2)) = (3 * 10^x + 5) / ((10^x - 1) * (10^x + 2))) :
 a - b = 7 / 3 :=
sorry

end calculate_a_minus_b_l640_640259


namespace polar_to_rectangular_range_of_m_l640_640296

-- Definition of the parametric equations of curve C
def curve (t : ℝ) : ℝ × ℝ :=
  (sqrt(3) * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line (rho theta m : ℝ) : Prop :=
  rho * sin (theta + π / 3) + m = 0

-- Definition of the rectangular equation of line l
def rectangular_line (x y m : ℝ) : Prop :=
  sqrt(3) * x + y + 2 * m = 0

-- Convert polar line equation to rectangular form
theorem polar_to_rectangular (rho theta m x y : ℝ)
  (h1 : polar_line rho theta m)
  (h2 : rho = sqrt (x^2 + y^2))
  (h3 : cos theta * rho = x ∧ sin theta * rho = y) :
  rectangular_line x y m :=
sorry

-- Range of values for m for which l intersects C
theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, (sqrt(3) * cos (2 * t), 2 * sin t) ∈ {p : ℝ × ℝ | rectangular_line p.1 p.2 m}) ↔
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end polar_to_rectangular_range_of_m_l640_640296


namespace find_expression_l640_640206

theorem find_expression (f : ℝ → ℝ) (h : ∀ x, f (real.sqrt x + 1) = x + 3) :
  ∀ y, y ≥ 0 → f(y + 1) = y^2 + 3 :=
by
  sorry

end find_expression_l640_640206


namespace probability_complete_collection_l640_640047

def combination (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem probability_complete_collection : 
  combination 6 6 * combination 12 4 / combination 18 10 = 5 / 442 :=
by
  have h1 : combination 6 6 = 1 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h2 : combination 12 4 = 495 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h3 : combination 18 10 = 43758 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  rw [h1, h2, h3]
  norm_num
  sorry

end probability_complete_collection_l640_640047


namespace function_monotonic_increasing_l640_640686

def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x - a / 2
          else log a x

theorem function_monotonic_increasing (a : ℝ) (h1 : a < 2) (h2 : 1 < a) (h3 : a * 3 / 2 ≤ 2) :
  ∃ (a : ℝ), a ∈ set.Icc (4 / 3) 2 :=
sorry

end function_monotonic_increasing_l640_640686


namespace volume_of_given_tetrahedron_l640_640817

noncomputable def volume_tetrahedron (a b c : ℝ) : ℝ :=
  (1 / 3) * a * b * c

theorem volume_of_given_tetrahedron :
  ∀ (A B C D : Type) (a b c : ℝ),
    a = 5 → b = 3 → c = 7 →
    (a * a + b * b = 34) → (b * b + c * c = 58) → (c * c + a * a = 74) →
    volume_tetrahedron a b c = 35 :=
by
  intros A B C D a b c ha hb hc hab hac hca
  simp [ha, hb, hc, volume_tetrahedron]
  norm_num
  sorry

end volume_of_given_tetrahedron_l640_640817


namespace smallest_integer_sqrt_difference_l640_640523

theorem smallest_integer_sqrt_difference :
  ∃ n : ℕ, (0 < n) ∧ (∀ m : ℕ, (0 < m) ∧ (m < n) → ¬ (sqrt m - sqrt (m - 1) < 0.1)) ∧ (sqrt n - sqrt (n - 1)) < 0.1 ∧ n = 26 :=
by
  sorry

end smallest_integer_sqrt_difference_l640_640523


namespace symmetric_center_f_div_x_sum_expression_l640_640973

-- Define the necessary functions and properties for f and f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
-- Conditions from the problem
-- f’((x / 2) + 1) is an even function
axiom even_f'_half_x_plus_one : ∀ x, f'((x / 2) + 1) = f'((-x / 2) + 1)
-- f(x) - x is an even function
axiom even_function_f_minus_x : ∀ x, f(x) - x = f(-x) + x

-- Theorem 1: Prove that the symmetric center for the function f(x) / x is (0, 1).
theorem symmetric_center_f_div_x : 
  ∃ c : ℝ, ∀ x : ℝ, f x / x + f (-x) / -x = 2 :=
begin
  use 1,
  sorry
end

-- Theorem 2: Prove the given expression simplifies to 0
theorem sum_expression : 
  (∑ k in (finset.range 2024).filter (λ n, n % 2 = 1), 
  (f' k - 1) * (f' (k + 1) + 1)) = 0 :=
begin
  sorry
end

end symmetric_center_f_div_x_sum_expression_l640_640973


namespace problem_solution_l640_640400

noncomputable def omega_nonreal_root (ω : ℂ) : ω ^ 4 = 1 ∧ ω.im ≠ 0 := sorry
noncomputable def conditions (b : ℕ → ℝ) (ω : ℂ) : ℂ := ∑ k in Finset.range n, (1 / (b k + ω)) = 3 - 4 * I

theorem problem_solution (b : ℕ → ℝ) (ω : ℂ) (n : ℕ)
  (h1 : omega_nonreal_root ω)
  (h2 : conditions b ω n):
  ∑ k in Finset.range n, ((2 * (b k) - 1) / ((b k) ^ 2 - (b k) + 1)) = 6 := sorry

end problem_solution_l640_640400


namespace correct_statement_A_l640_640492

-- Declare Avogadro's constant
def Avogadro_constant : ℝ := 6.022e23

-- Given conditions
def gas_mass_ethene : ℝ := 5.6 -- grams of ethylene
def gas_mass_cyclopropane : ℝ := 5.6 -- grams of cyclopropane
def gas_combined_carbon_atoms : ℝ := 0.4 * Avogadro_constant

-- Assertion to prove
theorem correct_statement_A :
    gas_combined_carbon_atoms = 0.4 * Avogadro_constant :=
by
  sorry

end correct_statement_A_l640_640492


namespace rectangular_equation_of_l_range_of_m_intersection_l640_640362

-- Definitions based on the problem conditions
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  let x := sqrt 3 * cos (2 * t)
  let y := 2 * sin t
  (x, y)

def polar_line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- The rectangular equation converted from polar form
theorem rectangular_equation_of_l (x y m : ℝ) :
  (polar_line_l (sqrt (x^2 + y^2)) (atan2 y x) m) ↔ (√3 * x + y + 2 * m = 0) :=
sorry

-- Range of values for m ensuring intersection of line l with curve C
theorem range_of_m_intersection (m : ℝ) :
  ((-19/12 : ℝ) ≤ m ∧ m ≤ (5/2 : ℝ)) ↔
  ∃ t : ℝ, let x := sqrt 3 * cos (2 * t)
           let y := 2 * sin t
           (sqrt 3 * x + y + 2 * m = 0 ∧ (-2 ≤ y ∧ y ≤ 2)) :=
sorry

end rectangular_equation_of_l_range_of_m_intersection_l640_640362


namespace seventh_term_of_arithmetic_sequence_l640_640808

theorem seventh_term_of_arithmetic_sequence 
  (a d : ℤ)
  (h1 : 5 * a + 10 * d = 35)
  (h2 : a + 5 * d = 10) :
  a + 6 * d = 11 :=
by
  sorry

end seventh_term_of_arithmetic_sequence_l640_640808


namespace hyperbola_foci_distance_l640_640009

theorem hyperbola_foci_distance (c : ℝ) (h : c = Real.sqrt 2) : 
  let f1 := (c * Real.sqrt 2, c * Real.sqrt 2)
  let f2 := (-c * Real.sqrt 2, -c * Real.sqrt 2)
  Real.sqrt ((f2.1 - f1.1) ^ 2 + (f2.2 - f1.2) ^ 2) = 4 * Real.sqrt 2 := 
by
  sorry

end hyperbola_foci_distance_l640_640009


namespace rectangular_equation_common_points_l640_640323

-- Define parametric equations of curve C
def x (t : ℝ) := (sqrt 3) * cos (2 * t)
def y (t : ℝ) := 2 * sin t

-- Define polar equation of line l
def polar_line (ρ θ m : ℝ) := ρ * sin (θ + π / 3) + m

-- Rectangular equation of l
theorem rectangular_equation (ρ θ x y m : ℝ) (h1 : ρ = sqrt (x^2 + y^2)) (h2 : θ = atan2 y x) :
  polar_line ρ θ m = 0 → (sqrt 3 * x + y + 2 * m = 0) :=
by 
  sorry

-- Range of values for m where line intersects curve
theorem common_points (h: ∃ t, sqrt 3 * cos (2 * t) = x ∧ 2 * sin t = y) :
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by 
  sorry

end rectangular_equation_common_points_l640_640323


namespace find_m_l640_640288

-- Definitions based on conditions
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def are_roots_of_quadratic (b c m : ℝ) : Prop :=
  b * c = 6 - m ∧ b + c = -(m + 2)

-- The theorem statement
theorem find_m {a b c m : ℝ} (h₁ : a = 5) (h₂ : is_isosceles_triangle a b c) (h₃ : are_roots_of_quadratic b c m) : m = -10 :=
sorry

end find_m_l640_640288


namespace rectangular_equation_common_points_l640_640319

-- Define parametric equations of curve C
def x (t : ℝ) := (sqrt 3) * cos (2 * t)
def y (t : ℝ) := 2 * sin t

-- Define polar equation of line l
def polar_line (ρ θ m : ℝ) := ρ * sin (θ + π / 3) + m

-- Rectangular equation of l
theorem rectangular_equation (ρ θ x y m : ℝ) (h1 : ρ = sqrt (x^2 + y^2)) (h2 : θ = atan2 y x) :
  polar_line ρ θ m = 0 → (sqrt 3 * x + y + 2 * m = 0) :=
by 
  sorry

-- Range of values for m where line intersects curve
theorem common_points (h: ∃ t, sqrt 3 * cos (2 * t) = x ∧ 2 * sin t = y) :
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by 
  sorry

end rectangular_equation_common_points_l640_640319


namespace min_dot_product_l640_640401

variables {α : Type*} [inner_product_space ℝ α] {a b c : α}

-- Definition for unit vectors
def is_unit_vector (v : α) : Prop := ∥v∥ = 1

-- Lean statement for the problem
theorem min_dot_product 
  (ha : is_unit_vector a)
  (hb : is_unit_vector b)
  (hc : is_unit_vector c)
  (h_ab : ⟪a, b⟫ = 1/2) : 
  ∃ c : α, is_unit_vector c → 
  real.min ((2 • a + c) ⬝ (b - c)) = -real.sqrt 3 :=
sorry

end min_dot_product_l640_640401


namespace evaluate_expression_l640_640177

-- Define the expression and the expected result
def expression := -(14 / 2 * 9 - 60 + 3 * 9)
def expectedResult := -30

-- The theorem that states the equivalence
theorem evaluate_expression : expression = expectedResult := by
  sorry

end evaluate_expression_l640_640177


namespace trigonometric_identity_l640_640542

theorem trigonometric_identity : 2 * sin (π / 12) * cos (π / 12) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l640_640542


namespace area_of_PQRSUV_proof_l640_640183

noncomputable def PQRSW_area (PQ QR RS SW : ℝ) : ℝ :=
  (1 / 2) * PQ * QR + (1 / 2) * (RS + SW) * 5

noncomputable def WUV_area (WU UV : ℝ) : ℝ :=
  WU * UV

theorem area_of_PQRSUV_proof 
  (PQ QR RS SW WU UV : ℝ)
  (hPQ : PQ = 8) (hQR : QR = 5) (hRS : RS = 7) (hSW : SW = 10)
  (hWU : WU = 6) (hUV : UV = 7) :
  PQRSW_area PQ QR RS SW + WUV_area WU UV = 147 :=
by
  simp only [PQRSW_area, WUV_area, hPQ, hQR, hRS, hSW, hWU, hUV]
  norm_num
  sorry

end area_of_PQRSUV_proof_l640_640183


namespace count_bad_arrangements_is_three_l640_640027

-- Define the problem of arranging numbers in a circle with the given conditions
def is_bad_arrangement (arr : List ℕ) : Prop :=
  ∃ (n : ℕ), n ∈ List.range 1 17 ∧ 
             ∀ (subseq : List ℕ), (subseq ≠ [] → subseq ⊆ arr → List.sum subseq ≠ n ∧ List.cyclic_subseq subseq arr)

-- Count the number of unique bad arrangements considering rotations and reflections
def count_bad_arrangements : ℕ :=
  if nbad = 3 then nbad else sorry
  where
    n1 := [1, 2, 3, 4, 6],
    n2 :: rest := List.permutations n1,
    reflect : List ℕ → List ℕ := id,
    rotations : List (List ℕ) := List.map (List.cyclic_rotations ∘ reflect) n2 :: rest,
    bads := List.filter is_bad_arrangement rotations,
    nbad := List.length (List.nub rotations)

-- Statement to prove the count of bad arrangements is exactly 3
theorem count_bad_arrangements_is_three : count_bad_arrangements = 3 := by
  sorry

end count_bad_arrangements_is_three_l640_640027


namespace rect_eq_and_range_of_m_l640_640307

noncomputable def rect_eq_of_polar (m : ℝ) : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ * sin (θ + π / 3) + m = 0 → sqrt 3 * (ρ * cos θ) + (ρ * sin θ) + 2 * m = 0

noncomputable def range_of_m_for_intersection : set ℝ := 
  {m : ℝ | -19 / 12 ≤ m ∧ m ≤ 5 / 2}

theorem rect_eq_and_range_of_m (m : ℝ) : 
  rect_eq_of_polar m ∧ (∀ t : ℝ, ∃ x y : ℝ, x = sqrt 3 * cos (2 * t) ∧ y = 2 * sin t ∧ 
    (sqrt 3 * x + y + 2 * m = 0) → m ∈ range_of_m_for_intersection) :=
by sorry

end rect_eq_and_range_of_m_l640_640307


namespace train_pass_time_approx_l640_640558

noncomputable def time_for_train_to_pass_man (train_length : ℝ) (train_speed_kmph : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph
  let relative_speed_mps := relative_speed_kmph * (5/18)
  train_length / relative_speed_mps

theorem train_pass_time_approx (h1 : train_length = 240) (h2 : train_speed_kmph = 60) (h3 : man_speed_kmph = 6) :
  time_for_train_to_pass_man train_length train_speed_kmph man_speed_kmph ≈ 13.09 :=
sorry

end train_pass_time_approx_l640_640558


namespace total_staff_left_l640_640006

variable (initial_chefs : ℕ) (initial_waiters : ℕ) (initial_busboys : ℕ) (initial_hostesses : ℕ)
variable (leaving_chefs : ℕ) (leaving_waiters : ℕ) (leaving_busboys : ℕ) (leaving_hostesses : ℕ)

def remaining_chefs := initial_chefs - leaving_chefs
def remaining_waiters := initial_waiters - leaving_waiters
def remaining_busboys := initial_busboys - leaving_busboys
def remaining_hostesses := initial_hostesses - leaving_hostesses

theorem total_staff_left (h_chefs : initial_chefs = 16)
                        (h_waiters : initial_waiters = 16)
                        (h_busboys : initial_busboys = 10)
                        (h_hostesses : initial_hostesses = 5)
                        (h_leave_chefs : leaving_chefs = 6)
                        (h_leave_waiters : leaving_waiters = 3)
                        (h_leave_busboys : leaving_busboys = 4)
                        (h_leave_hostesses : leaving_hostesses = 2)
                        :
                        remaining_chefs initial_chefs leaving_chefs +
                        remaining_waiters initial_waiters leaving_waiters +
                        remaining_busboys initial_busboys leaving_busboys +
                        remaining_hostesses initial_hostesses leaving_hostesses = 32 :=
by {
  rw [h_chefs, h_waiters, h_busboys, h_hostesses, h_leave_chefs, h_leave_waiters, h_leave_busboys, h_leave_hostesses],
  simp [remaining_chefs, remaining_waiters, remaining_busboys, remaining_hostesses],
  norm_num,
  exact rfl
}

end total_staff_left_l640_640006


namespace range_of_y_l640_640705

theorem range_of_y (y: ℝ) (hy: y > 0) (h_eq: ⌈y⌉ * ⌊y⌋ = 72) : 8 < y ∧ y < 9 :=
by
  sorry

end range_of_y_l640_640705


namespace selling_price_of_book_l640_640547

variable (cost_price profit_percentage : ℝ)
variable (h_cost_price : cost_price = 32)
variable (h_profit_percentage : profit_percentage = 0.75)

theorem selling_price_of_book :
  let profit_amount := profit_percentage * cost_price in
  let selling_price := cost_price + profit_amount in
  selling_price = 56 :=
by
  sorry

end selling_price_of_book_l640_640547


namespace shooting_game_probability_l640_640106

theorem shooting_game_probability :
  let P_A := 3 / 5 in
  let P_A_complement := 1 - P_A in
  let P_B_complement (p : ℝ) := 1 - p in
  let event_equation (p : ℝ) := P_A * (P_B_complement p) + P_A_complement * p = 9 / 20 in
  (∃ p : ℝ, event_equation p) ↔ p = 3 / 4 :=
by
  sorry

end shooting_game_probability_l640_640106


namespace paintings_not_both_l640_640573

theorem paintings_not_both (A J' I : ℕ) (hA : A = 25) (hJ' : J' = 8) (hI : I = 15) :
  A - I + J' = 18 :=
by
  rw [hA, hJ', hI]
  norm_num

end paintings_not_both_l640_640573


namespace quadratic_inequality_solution_l640_640644

theorem quadratic_inequality_solution 
  (a : ℝ) 
  (h : ∀ x : ℝ, -1 < x ∧ x < a → -x^2 + 2 * a * x + a + 1 > a + 1) : -1 < a ∧ a ≤ -1/2 :=
sorry

end quadratic_inequality_solution_l640_640644


namespace total_goals_by_other_players_l640_640888

theorem total_goals_by_other_players (total_players goals_season games_played : ℕ)
  (third_players_goals avg_goals_per_third : ℕ)
  (h1 : total_players = 24)
  (h2 : goals_season = 150)
  (h3 : games_played = 15)
  (h4 : third_players_goals = total_players / 3)
  (h5 : avg_goals_per_third = 1)
  : (goals_season - (third_players_goals * avg_goals_per_third * games_played)) = 30 :=
by
  sorry

end total_goals_by_other_players_l640_640888


namespace jar_lasts_20_days_l640_640383

def serving_size : ℝ := 0.5
def daily_servings : ℕ := 3
def container_size : ℝ := 32 - 2

def daily_usage (serving_size : ℝ) (daily_servings : ℕ) : ℝ :=
  serving_size * daily_servings

def days_to_finish (container_size daily_usage : ℝ) : ℝ :=
  container_size / daily_usage

theorem jar_lasts_20_days :
  days_to_finish container_size (daily_usage serving_size daily_servings) = 20 :=
by
  sorry

end jar_lasts_20_days_l640_640383


namespace hyperbola_eccentricity_l640_640229

theorem hyperbola_eccentricity (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
  (h_eq : b ^ 2 = (5 / 4) * a ^ 2) 
  (h_c : c ^ 2 = a ^ 2 + b ^ 2) : 
  (3 / 2) = c / a :=
by sorry

end hyperbola_eccentricity_l640_640229


namespace union_of_sets_l640_640972

def set_A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }

def set_B : Set ℝ := { x | sqrt (x - 1) < 1 }

theorem union_of_sets : (set_A ∪ set_B) = { x : ℝ | -1 ≤ x ∧ x ≤ 2 } := by
  sorry

end union_of_sets_l640_640972


namespace cube_midpoints_five_or_four_distinct_values_l640_640742

theorem cube_midpoints_five_or_four_distinct_values :
  ∃ (V : Finset ℕ) (f : ℕ → ℕ) (s : Finset ℕ),
    (V = {1, 2, 3, 4, 5, 6, 7, 8}) ∧
    ((∀ x y ∈ V, (Finset.member x /\ Finset.member y  -> s member (f x + f y))) -> 
    (∃ s , s.card =5)) ∧
      (∀ x y ∈ V , x ≠ y→ s member (f x + f y )) -> ¬(∃ s, s.card = 4)
       :=
begin
  sorry
end

end cube_midpoints_five_or_four_distinct_values_l640_640742


namespace complete_collection_prob_l640_640051

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end complete_collection_prob_l640_640051


namespace batsman_average_after_17th_inning_l640_640853

theorem batsman_average_after_17th_inning
  (A : ℝ) -- average before 17th inning
  (h1 : (16 * A + 50) / 17 = A + 2) : 
  (A + 2) = 18 :=
by
  -- Proof goes here
  sorry

end batsman_average_after_17th_inning_l640_640853


namespace total_balloons_after_giving_l640_640449

def sam_initial_balloons : Float := 46.0
def fred_balloons : Float := 10.0
def dan_balloons : Float := 16.0

theorem total_balloons_after_giving (sam_initial_balloons fred_balloons dan_balloons : Float) : 
  sam_initial_balloons = 46.0 →
  fred_balloons = 10.0 →
  dan_balloons = 16.0 →
  let sam_remaining_balloons := sam_initial_balloons - fred_balloons in 
  (sam_remaining_balloons + dan_balloons) = 52.0 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  let sam_remaining_balloons := 46.0 - 10.0
  show (sam_remaining_balloons + 16.0) = 52.0
  sorry

end total_balloons_after_giving_l640_640449


namespace set_intersection_l640_640242

-- Defining sets A and B as per the given problem conditions
def A : Set ℤ := {-1, 0, 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 1}

-- Stating the proof problem, that A ∩ B = {-1, 0}
theorem set_intersection (h : ∀ x, x ∈ ({x : ℤ | x ∈ A ∧ (x : ℝ) ∈ B}) → x = -1 ∨ x = 0) : A ∩ B = {-1, 0} := sorry

end set_intersection_l640_640242


namespace assign_volunteers_to_schools_l640_640908

theorem assign_volunteers_to_schools :
  let volunteers := 4
  let schools := 3
  (∀ (v : Fin (volunteers + 1) → Fin (schools + 1)), 
    (∀ s, 1 ≤ (v.val.filter (λ x, x = s)).length) →
    (v.val.length = volunteers) → 
    (schools = 3)) → 
  ∑ i : Fin (schools + 1), choose volunteers i * i.factorial = 36 :=
by
  let volunteers := 4
  let schools := 3
  intros v h1 h2
  sorry

end assign_volunteers_to_schools_l640_640908


namespace shoes_lost_eq_ten_l640_640767

-- Definitions derived from the conditions
def original_pairs : ℕ := 23
def pairs_left : ℕ := 18
def original_individuals : ℕ := original_pairs * 2
def individuals_left : ℕ := pairs_left * 2

-- Theorem statement to be proved
theorem shoes_lost_eq_ten : original_individuals - individuals_left = 10 := 
begin
  -- We are not providing the solution steps here, so we markdown the proof.
  sorry
end

end shoes_lost_eq_ten_l640_640767


namespace parallel_vectors_perpendicular_vectors_l640_640245

/-- Given vectors a and b where a = (1, 2) and b = (x, 1),
    let u = a + b and v = a - b.
    Prove that if u is parallel to v, then x = 1/2. 
    Also, prove that if u is perpendicular to v, then x = 2 or x = -2. --/

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)
noncomputable def vector_u (x : ℝ) : ℝ × ℝ := (1 + x, 3)
noncomputable def vector_v (x : ℝ) : ℝ × ℝ := (1 - x, 1)

theorem parallel_vectors (x : ℝ) :
  (vector_u x).fst / (vector_v x).fst = (vector_u x).snd / (vector_v x).snd ↔ x = 1 / 2 :=
by
  sorry

theorem perpendicular_vectors (x : ℝ) :
  (vector_u x).fst * (vector_v x).fst + (vector_u x).snd * (vector_v x).snd = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end parallel_vectors_perpendicular_vectors_l640_640245


namespace first_expression_second_expression_l640_640983

open Real 

theorem first_expression (a x y : ℝ) : 
    (-2 * a)^6 * (-3 * a^3) + [2 * a^2]^3 / (1 / (1 / ((-2)^2 * 3^2 * (x * y)^3))) = 
    192 * a^9 + 288 * a^6 * (x * y)^3 := 
sorry

theorem second_expression : 
    abs (-1/8) + π^3 + (- (1/2)^3 - (1/3)^2) = 
    π^3 - 1 / 72 := 
sorry

end first_expression_second_expression_l640_640983


namespace seventy_fifth_percentile_expected_value_X_variance_X_l640_640499

def ratings : List ℝ := [7.8, 8.9, 8.6, 7.4, 8.5, 8.5, 9.5, 9.9, 8.3, 9.1]

theorem seventy_fifth_percentile : List.percentile ratings 0.75 = 9.1 := 
sorry

noncomputable def X : ℕ → ℕ → ProbabilityTheory.BinomialDistribution :=
λ n p, ProbabilityTheory.BinomialDistribution.mk n p

theorem expected_value_X :
  ProbabilityTheory.expectedValue (X 3 0.3) = 0.9 :=
sorry 

theorem variance_X :
  ProbabilityTheory.variance (X 3 0.3) = 0.63 :=
sorry

end seventy_fifth_percentile_expected_value_X_variance_X_l640_640499


namespace part_I_possible_lineups_part_II_outcome_3_games_part_II_outcome_4_or_5_games_l640_640870

theorem part_I_possible_lineups : 
  ∃ n : ℕ, n = 60 ∧ 
  (let total_athletes := 5 in 
   let games := 3 in 
   let lineups := Nat.Perm total_athletes games in 
   lineups = n) :=
begin
  sorry
end

theorem part_II_outcome_3_games (A B C : Type) : 
  ∃ n : ℕ, n = 6 ∧ 
  (let lineups := List.permutations [A, B, C] in 
   lineups.length = n) :=
begin
  sorry
end

theorem part_II_outcome_4_or_5_games : 
  ∃ n : ℕ, n = 24 ∧ 
  (let first_3_games_lineups := 6 in 
   let subsequent_games_factor := 2 in 
   let total_lineups := first_3_games_lineups * subsequent_games_factor * 2 in 
   total_lineups = n) :=
begin
  sorry
end

end part_I_possible_lineups_part_II_outcome_3_games_part_II_outcome_4_or_5_games_l640_640870


namespace intersection_complement_U_A_B_l640_640993

namespace SetIntersection

variable (U A B : Set ℤ)
variable (complementA : Set ℤ) [decidable_pred (λ x, x ∈ U)]

axiom U_def : U = {-2, -1, 0, 1, 2}
axiom A_def : A = {-1, 2}
axiom B_def : B = {-1, 0, 1}
noncomputable def complement_U_A : Set ℤ := U \ A

theorem intersection_complement_U_A_B : (complement_U_A U A) ∩ B = {0, 1} := by
  rw [U_def, A_def, B_def, Set.diff_eq_compl_inter, Set.compl_eq_univ_diff, Set.univ_diff, Set.inter_assoc,
      Set.inter_univ, Set.of_eq]
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff, Set.mem_diff, Set.mem_compl_iff, Set.mem_inter_iff,
             Int.coe_nat_eq, true_and, not_or_distrib, Set.mem_set_of_eq, Set.inter_eq, SetDiff, Set.singleton]

  sorry

end SetIntersection

end intersection_complement_U_A_B_l640_640993


namespace exists_red_1_or_green_congruent_triangle_l640_640745

theorem exists_red_1_or_green_congruent_triangle 
  (A B C : ℝ × ℝ)
  (distance : (ℝ × ℝ) → (ℝ × ℝ) → ℝ := λ p q, real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) 
  (paint : ℝ × ℝ → Prop) -- paint is true for red points and false for green points
  (H : ∀ p q : ℝ × ℝ, paint p = true ∧ paint q = true → distance p q ≠ 1) -- No two red points are 1 unit apart
  (H2 : ∀ p q r : ℝ × ℝ, paint p = false ∧ paint q = false ∧ paint r = false → ¬(∃ (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ), congruent [p, q, r] [A, B, C])) -- No three green points form a triangle congruent to ABC
  : false :=
sorry

end exists_red_1_or_green_congruent_triangle_l640_640745


namespace find_median_of_first_twelve_positive_integers_l640_640517

def median_of_first_twelve_positive_integers : ℚ :=
  let A := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (A[5] + A[6]) / 2

theorem find_median_of_first_twelve_positive_integers :
  median_of_first_twelve_positive_integers = 6.5 :=
by
  sorry

end find_median_of_first_twelve_positive_integers_l640_640517


namespace problem_part1_problem_part2_l640_640302

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l_rect_eq (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (sqrt 3 * x + y + 2 * m = 0)

def line_l_range_m (m : ℝ) : Prop :=
  -19/12 ≤ m ∧ m ≤ 5/2

theorem problem_part1 (m : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * sin (θ + π / 3) + m = 0) ↔ line_l_rect_eq m := 
sorry

theorem problem_part2 :
  ∀ m : ℝ, 
  (∃ t : ℝ, line_l_rect_eq m (curve_C t).1 (curve_C t).2) ↔ line_l_range_m m := 
sorry

end problem_part1_problem_part2_l640_640302


namespace rectangular_equation_of_l_range_of_m_intersection_l640_640363

-- Definitions based on the problem conditions
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  let x := sqrt 3 * cos (2 * t)
  let y := 2 * sin t
  (x, y)

def polar_line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- The rectangular equation converted from polar form
theorem rectangular_equation_of_l (x y m : ℝ) :
  (polar_line_l (sqrt (x^2 + y^2)) (atan2 y x) m) ↔ (√3 * x + y + 2 * m = 0) :=
sorry

-- Range of values for m ensuring intersection of line l with curve C
theorem range_of_m_intersection (m : ℝ) :
  ((-19/12 : ℝ) ≤ m ∧ m ≤ (5/2 : ℝ)) ↔
  ∃ t : ℝ, let x := sqrt 3 * cos (2 * t)
           let y := 2 * sin t
           (sqrt 3 * x + y + 2 * m = 0 ∧ (-2 ≤ y ∧ y ≤ 2)) :=
sorry

end rectangular_equation_of_l_range_of_m_intersection_l640_640363


namespace rectangular_solid_diagonal_l640_640033

theorem rectangular_solid_diagonal {a b c : ℝ} 
  (h_surface : 2 * (a * b + b * c + c * a) = 30)
  (h_edges : 4 * (a + b + c) = 28) : 
  (a^2 + b^2 + c^2 = 19) ∧ (sqrt (a^2 + b^2 + c^2) = sqrt 19) :=
by 
  sorry

end rectangular_solid_diagonal_l640_640033


namespace consecutive_numbers_product_diff_by_54_l640_640581

def product_of_nonzero_digits (n : ℕ) : ℕ :=
  (to_digits 10 n).filter (λ d, d ≠ 0).prod

theorem consecutive_numbers_product_diff_by_54 :
  let n := 299 in
  product_of_nonzero_digits n = 54 * product_of_nonzero_digits (n + 1) :=
by
  let n := 299
  -- 299 has non-zero digits 2, 9, 9 with product 162
  have h1 : product_of_nonzero_digits n = 2 * 9 * 9 := sorry
  -- 300 has non-zero digit 3 with product 3
  have h2 : product_of_nonzero_digits (n + 1) = 3 := sorry
  -- Confirm that 162 = 54 * 3
  show product_of_nonzero_digits n = 54 * product_of_nonzero_digits (n + 1)
  rw [h1, h2]
  simp
  norm_num
  sorry

end consecutive_numbers_product_diff_by_54_l640_640581


namespace rect_eq_and_range_of_m_l640_640315

noncomputable def rect_eq_of_polar (m : ℝ) : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ * sin (θ + π / 3) + m = 0 → sqrt 3 * (ρ * cos θ) + (ρ * sin θ) + 2 * m = 0

noncomputable def range_of_m_for_intersection : set ℝ := 
  {m : ℝ | -19 / 12 ≤ m ∧ m ≤ 5 / 2}

theorem rect_eq_and_range_of_m (m : ℝ) : 
  rect_eq_of_polar m ∧ (∀ t : ℝ, ∃ x y : ℝ, x = sqrt 3 * cos (2 * t) ∧ y = 2 * sin t ∧ 
    (sqrt 3 * x + y + 2 * m = 0) → m ∈ range_of_m_for_intersection) :=
by sorry

end rect_eq_and_range_of_m_l640_640315


namespace exists_unique_k_2k_when_a_is_3_no_k_2k_when_a_gt_3_l640_640588

noncomputable def sequence (a : ℕ) : ℕ → ℕ 
| 1       := a
| (n + 1) := match n with
             | 0     => a
             | n' + 1 => sequence (n' + 1) + Nat.gcd (n' + 2) (sequence (n' + 1))
             end

theorem exists_unique_k_2k_when_a_is_3 :
  ∃! (k : ℕ), sequence 3 k = 2 * k :=
by
  sorry

theorem no_k_2k_when_a_gt_3 (a : ℕ) (h : a > 3) :
  ¬∃ (k : ℕ), sequence a k = 2 * k :=
by
  sorry

end exists_unique_k_2k_when_a_is_3_no_k_2k_when_a_gt_3_l640_640588


namespace no_intersection_l640_640620

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def inside_parabola (x₀ y₀ : ℝ) : Prop := y₀^2 < 4 * x₀
def line (x x₀ y y₀ : ℝ) : Prop := y₀ * y = 2 * (x + x₀)
def delta (x₀ y₀ x : ℝ) : ℝ := (2 * x₀ - y₀^2)^2 - 4 * x₀^2

theorem no_intersection (x₀ y₀ : ℝ) (h : inside_parabola x₀ y₀) :
  ∀ x y : ℝ, ¬ (parabola x y ∧ line x x₀ y y₀) :=
by
  intro x y
  assume hP hL
  have hQuad: x^2 + (2 * x₀ - y₀^2) * x + x₀^2 = 0,
    sorry -- Follows from substitution into parabola equation and simplification.
  have discriminant: delta x₀ y₀ x < 0 := by
    calc
      delta x₀ y₀ x = y₀^4 - 4 * x₀ * y₀^2 := by sorry -- Definition expansion and simplification
      ... < 0 : by sorry -- Because of inside_parabola condition h
  exact (hQuad.disc_lt_zero discriminant hP hL)

end no_intersection_l640_640620


namespace total_penalty_kicks_l640_640464

/-- The Benton Youth Soccer Team has 25 players, including 4 goalies. Each player (except for the goalie in goal) takes a shot on goal against each of the 4 goalies. The total number of penalty kicks must be equal to 96. -/
theorem total_penalty_kicks (total_players : ℕ) (goalies : ℕ) (shots_per_goalie : ℕ) 
    (h1 : total_players = 25) 
    (h2 : goalies = 4) 
    (h3 : shots_per_goalie = total_players - 1) : 
    goalies * shots_per_goalie = 96 := 
by 
  -- Using the conditions:
  -- total_players = 25
  -- goalies = 4
  -- shots_per_goalie = 24 = total_players - 1
  -- We need to show: 4 * 24 = 96
  rw [h1, h2, h3]
  calc
  4 * 24 = 96 : by norm_num
  sorry

end total_penalty_kicks_l640_640464


namespace number_of_polynomials_with_given_roots_l640_640916

-- Define the polynomial form
def polynomial (b : Fin 10 → Bool) : Fintype ℤ :=
{ x | (∃ n, x + n = 0) ∧ (∃ m, x - m = 0) ∧ b n = true ∧ b m = true }

-- State the problem
theorem number_of_polynomials_with_given_roots : 
  (∑ b in (Fin 10 → Bool), if (0, -1, 1) ∈ polynomial b then 1 else 0) = 65 := sorry

end number_of_polynomials_with_given_roots_l640_640916


namespace determine_a_l640_640019

theorem determine_a (x y a : ℝ) 
  (h1 : x + y = a)
  (h2 : x^3 + y^3 = a)
  (h3 : x^5 + y^5 = a) : 
  a = 0 := 
sorry

end determine_a_l640_640019


namespace range_of_h_l640_640403

def f (x : ℝ) : ℝ := 4 * x - 3
def h (x : ℝ) : ℝ := f (f (f x))

theorem range_of_h : 
  (∀ x, -1 ≤ x ∧ x ≤ 3 → -127 ≤ h x ∧ h x ≤ 129) :=
by
  sorry

end range_of_h_l640_640403


namespace sum_first_five_terms_arithmetic_seq_l640_640371

def arithmetic_seq (a d : ℕ) : ℕ → ℕ
| 0 => a
| (n + 1) => (arithmetic_seq n) + d

def sum_first_n (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem sum_first_five_terms_arithmetic_seq :
  ∀ (a d : ℕ),
    arithmetic_seq a d 6 = 19 → (arithmetic_seq a d 1) + (arithmetic_seq a d 7) = 26 → sum_first_n a d 5 = 35 :=
by
  intros a d h1 h2
  sorry

end sum_first_five_terms_arithmetic_seq_l640_640371


namespace slope_of_intersection_points_l640_640199

theorem slope_of_intersection_points (s : ℝ) :
  ∃ m : ℝ, m = 221 / 429 ∧ 
  ∀ (x y : ℝ), (2 * x + 3 * y = 9 * s + 4) ∧ (3 * x - 2 * y = 5 * s - 3) →
  ∃ (c : ℝ), y = c + m * x :=
begin
  sorry
end

end slope_of_intersection_points_l640_640199


namespace contractor_fine_amount_l640_640111

def total_days := 30
def daily_earning := 25
def total_earnings := 360
def days_absent := 12
def days_worked := total_days - days_absent
def fine_per_absent_day (x : ℝ) : Prop :=
  (daily_earning * days_worked) - (x * days_absent) = total_earnings

theorem contractor_fine_amount : ∃ x : ℝ, fine_per_absent_day x := by
  use 7.5
  sorry

end contractor_fine_amount_l640_640111


namespace ellipse_equation_l640_640143

noncomputable def standard_equation_ellipse (a b x y : ℝ) : Prop :=
  (x^2 / b^2) + (y^2 / a^2) = 1

theorem ellipse_equation :
  ∃ (a b : ℝ), a > b ∧ (∃ (x y : ℝ), (x, y) ≠ (0, 0) ∧ x^2 / b^2 + y^2 / a^2 = 1) ∧ 
  (∃ (c : ℝ), c = sqrt 50 ∧ a^2 - b^2 = c^2) ∧
  (let x0 := 1 / 2 in let y0 := 3 * x0 - 2 in (y0 = -1 / 2) → a^2 = 3 * b^2) →
  standard_equation_ellipse 75 25 x y :=
sorry

end ellipse_equation_l640_640143


namespace find_BC_length_l640_640714

theorem find_BC_length (A B C X : Type) (AB AC : ℕ) (h1 : AB = 70) (h2 : AC = 90)
  (h_circle : ∃ (circle : Type), circle = {center := A, radius := AC}) 
  (h_intersections : ∀ (BC : Type), BC = {B, X} ∈ h_circle)
  (h_BX_CX_integers : ∀ (BX CX : ℕ), BX ≠ 0 ∧ CX ≠ 0) : 
  (BC = 100) := sorry

end find_BC_length_l640_640714


namespace max_2xy_eq_one_fourth_min_4x2_y2_eq_one_half_min_one_over_x_plus_one_over_y_eq_three_plus_two_sqrt_two_l640_640974

variable (x y : ℝ)
hypothesis pos_x : x > 0
hypothesis pos_y : y > 0
hypothesis sum_eq : 2 * x + y = 1

theorem max_2xy_eq_one_fourth : ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 2 * x * y = 1 / 4 :=
sorry

theorem min_4x2_y2_eq_one_half: ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 4 * x^2 + y^2 = 1 / 2 :=
sorry

theorem min_one_over_x_plus_one_over_y_eq_three_plus_two_sqrt_two: ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ (1 / x) + (1 / y) = 3 + 2 * Real.sqrt 2 :=
sorry

end max_2xy_eq_one_fourth_min_4x2_y2_eq_one_half_min_one_over_x_plus_one_over_y_eq_three_plus_two_sqrt_two_l640_640974


namespace mike_telephone_numbers_l640_640566

theorem mike_telephone_numbers : 
  let digits := {0, 2, 3, 4, 5, 6, 7, 8}
  let non_zero_digits := digits \ {0}
  let length := 7
  ∃ (numbers : Finset (Finset ℕ)), 
    (∀ x ∈ numbers, x.card = length ∧ x ⊆ non_zero_digits ∧ x.pairwise (≤)) ∧
    numbers.card = 1 := 
by
  sorry

end mike_telephone_numbers_l640_640566


namespace intersect_rectangular_eqn_range_of_m_l640_640354

-- Definitions of the parametric and polar equations
def curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Proving the rectangular equation and range of m for the intersection of l and C
theorem intersect_rectangular_eqn (m t : ℝ) :
  (∃ ρ θ, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  ∃ x y, (x, y) = curve_C t ∧ √3 * x + y + 2 * m = 0 :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ ρ θ t, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end intersect_rectangular_eqn_range_of_m_l640_640354


namespace roots_form_parallelogram_l640_640182

noncomputable def polynomial := λ (a : ℝ), 
  polynomial.C 1 * polynomial.X^4 - 
  polynomial.C 8 * polynomial.X^3 + 
  polynomial.C (17 * a) * polynomial.X^2 - 
  polynomial.C (5 * (2 * a^2 + 5 * a - 4)) * polynomial.X + 
  polynomial.C 4

theorem roots_form_parallelogram (a : ℝ) :
  (a = 2 ∨ a = -5 / 2) ↔
  (∀ z : ℂ, polynomial a).eval z = 0 → 
  (∃ w : ℂ, (z = w + 2) ∨ (z = -w + 2)) := 
sorry

end roots_form_parallelogram_l640_640182


namespace distance_from_origin_to_line_l640_640791

-- Lean statement representing the given mathematical proof problem
theorem distance_from_origin_to_line : ∀ (x y : ℝ), (3 * x + 2 * y - 13 = 0) → (sqrt (x^2 + y^2) = sqrt 13) :=
by 
  intro x y
  intro h_line
  sorry

end distance_from_origin_to_line_l640_640791


namespace factorial_division_l640_640584

theorem factorial_division : 12! / 10! = 132 := by sorry

end factorial_division_l640_640584


namespace family_of_lines_fixed_point_l640_640441

theorem family_of_lines_fixed_point (m : ℝ) : 
  ∃ (x y : ℝ), 
    (3m - 2) * x - (m - 2) * y - (m - 5) = 0 ∧ 
    x = -3/4 ∧ 
    y = -13/4 := 
by
  use (-3/4), (-13/4)
  sorry

end family_of_lines_fixed_point_l640_640441


namespace balanced_tokens_parity_l640_640751

-- Define what it means for a token to be balanced
def is_balanced (n : ℕ) (tokens : list bool) (idx : ℕ) : bool :=
  let count_whites_left := tokens.take idx |>.filter (λ t, t = ff) |>.length
  let count_blacks_right := tokens.drop (idx + 1) |>.filter (λ t, t = tt) |>.length
  count_whites_left + count_blacks_right = n

-- Check the parity of the number of balanced tokens
theorem balanced_tokens_parity (n : ℕ) (h : 0 < n) (tokens : list bool) 
  (h_len : tokens.length = 2 * n + 1) : 
  (list.range tokens.length |>.filter (is_balanced n tokens)).length % 2 = 1 :=
sorry

end balanced_tokens_parity_l640_640751


namespace systematic_sampling_fourth_group_l640_640718

theorem systematic_sampling_fourth_group (n m k g2 g4 : ℕ) (h_class_size : n = 72)
  (h_sample_size : m = 6) (h_k : k = n / m) (h_group2 : g2 = 16) (h_group4 : g4 = g2 + 2 * k) :
  g4 = 40 := by
  sorry

end systematic_sampling_fourth_group_l640_640718


namespace phil_winning_strategy_12_ellie_winning_strategy_2012_l640_640617

noncomputable def philWins_12 (n : ℕ) (initial_board : matrix (fin n) (fin n) ℕ) : Prop :=
  n = 12 ∧ ∃ (S : ℕ) (modS : S % 3 ≠ 0), S = initial_board.valsum (λ i j, initial_board i j)

noncomputable def ellieWins_2012 (n : ℕ) (initial_board : matrix (fin n) (fin n) ℕ) : Prop :=
  n = 2012 ∧ ∃ (strategy : list (fin n × fin n × fin n)),
    (λ board : matrix (fin n) (fin n) ℕ, ∀ (i j : fin n), board i j = 0)

theorem phil_winning_strategy_12 (n : ℕ) (initial_board : matrix (fin n) (fin n) ℕ) :
  philWins_12 n initial_board := sorry

theorem ellie_winning_strategy_2012 (n : ℕ) (initial_board : matrix (fin n) (fin n) ℕ) :
  ellieWins_2012 n initial_board := sorry

end phil_winning_strategy_12_ellie_winning_strategy_2012_l640_640617


namespace opposite_pair_A_l640_640848

theorem opposite_pair_A : ∃ x y : ℝ, x = -2 ∧ y = Real.sqrt ((-2) ^ 2) ∧ x = -y := by
  use [-2, Real.sqrt ((-2) ^ 2)]
  sorry

end opposite_pair_A_l640_640848


namespace rectangular_equation_of_l_range_of_m_intersection_l640_640365

-- Definitions based on the problem conditions
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  let x := sqrt 3 * cos (2 * t)
  let y := 2 * sin t
  (x, y)

def polar_line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- The rectangular equation converted from polar form
theorem rectangular_equation_of_l (x y m : ℝ) :
  (polar_line_l (sqrt (x^2 + y^2)) (atan2 y x) m) ↔ (√3 * x + y + 2 * m = 0) :=
sorry

-- Range of values for m ensuring intersection of line l with curve C
theorem range_of_m_intersection (m : ℝ) :
  ((-19/12 : ℝ) ≤ m ∧ m ≤ (5/2 : ℝ)) ↔
  ∃ t : ℝ, let x := sqrt 3 * cos (2 * t)
           let y := 2 * sin t
           (sqrt 3 * x + y + 2 * m = 0 ∧ (-2 ≤ y ∧ y ≤ 2)) :=
sorry

end rectangular_equation_of_l_range_of_m_intersection_l640_640365


namespace problem_part1_problem_part2_l640_640298

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l_rect_eq (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (sqrt 3 * x + y + 2 * m = 0)

def line_l_range_m (m : ℝ) : Prop :=
  -19/12 ≤ m ∧ m ≤ 5/2

theorem problem_part1 (m : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * sin (θ + π / 3) + m = 0) ↔ line_l_rect_eq m := 
sorry

theorem problem_part2 :
  ∀ m : ℝ, 
  (∃ t : ℝ, line_l_rect_eq m (curve_C t).1 (curve_C t).2) ↔ line_l_range_m m := 
sorry

end problem_part1_problem_part2_l640_640298


namespace inverse_of_exponential_is_logarithm_l640_640483

theorem inverse_of_exponential_is_logarithm :
  ∀ (y : ℝ), ∃ (x : ℝ), y = 2^x ↔ x = log y / log 2 := by
sorry

end inverse_of_exponential_is_logarithm_l640_640483


namespace triangle_intersection_l640_640376

theorem triangle_intersection
  (ABC : Triangle)
  (I : Point) (hI : incenter I ABC)
  (I' : Point) (hI' : excenter I' ABC (opposite C) (tangent AB) (extension CB) (extension CA))
  (L : Point) (hL : touches I' Circle (AB))
  (L' : Point) (hL' : touches I Circle (AB))
  (CH : Line) (hCH : altitude CH ABC) :
  collinear IL' I'L CH :=
sorry

end triangle_intersection_l640_640376


namespace problem_part1_problem_part2_l640_640306

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l_rect_eq (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (sqrt 3 * x + y + 2 * m = 0)

def line_l_range_m (m : ℝ) : Prop :=
  -19/12 ≤ m ∧ m ≤ 5/2

theorem problem_part1 (m : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * sin (θ + π / 3) + m = 0) ↔ line_l_rect_eq m := 
sorry

theorem problem_part2 :
  ∀ m : ℝ, 
  (∃ t : ℝ, line_l_rect_eq m (curve_C t).1 (curve_C t).2) ↔ line_l_range_m m := 
sorry

end problem_part1_problem_part2_l640_640306


namespace sticker_probability_l640_640058

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end sticker_probability_l640_640058


namespace polar_to_rectangular_intersection_range_l640_640335

-- Definitions of parametric equations for curve C
def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line_l (ρ θ m: ℝ) : Prop := ρ * sin (θ + π / 3) + m = 0

-- Rectangular equation of line l
def rectangular_line_l (x y m: ℝ) : Prop := sqrt 3 * x + y + 2 * m = 0

-- Prove that polar equation of line l can be transformed to rectangular form
theorem polar_to_rectangular (ρ θ m : ℝ) :
  (polar_line_l ρ θ m) → (∃ x y, ρ = sqrt (x^2 + y^2) ∧ θ = arctan (y / x) ∧ rectangular_line_l x y m) :=
  sorry

-- Prove that the range of values for m ensuring line l intersects curve C is [-19/12, 5/2]
theorem intersection_range (m : ℝ) :
  (∃ t : ℝ, rectangular_line_l (sqrt 3 * cos (2 * t)) (2 * sin t) m) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
  sorry

end polar_to_rectangular_intersection_range_l640_640335


namespace calc_sqrt_pow_l640_640153

theorem calc_sqrt_pow:
  (\(\sqrt{(\sqrt{5})^5}\right)^6 = 125 \cdot \sqrt[4]{125} :=
by sorry

end calc_sqrt_pow_l640_640153


namespace train_crossing_time_l640_640559

/-- 
Given a train with speed 90 km/hr and length 125 meters, 
prove that it takes the train 5 seconds to cross a pole. 
-/
theorem train_crossing_time :
  ∀ (speed_kmph : ℕ) (length_m : ℕ), 
  speed_kmph = 90 → 
  length_m = 125 → 
  (length_m * 3600 / (speed_kmph * 1000) = 5) :=
by
  intros speed_kmph length_m h_speed h_length
  rw [h_speed, h_length]
  -- convert speed to m/s
  have speed_mps : ℕ := 25  -- 90 * 1000 / 3600
  -- compute time
  have time : ℕ := 125 / speed_mps  -- 125 / 25
  exact rfl

end train_crossing_time_l640_640559


namespace flies_eaten_per_day_l640_640169

def frogs_needed (fish : ℕ) : ℕ := fish * 8
def flies_needed (frogs : ℕ) : ℕ := frogs * 30
def fish_needed (gharials : ℕ) : ℕ := gharials * 15

theorem flies_eaten_per_day (gharials : ℕ) : flies_needed (frogs_needed (fish_needed gharials)) = 32400 :=
by
  have h1 : fish_needed 9 = 135 := rfl
  have h2 : frogs_needed 135 = 1080 := rfl
  have h3 : flies_needed 1080 = 32400 := rfl
  exact Eq.trans (Eq.trans h1.symm (Eq.trans h2.symm h3.symm)) sorry

end flies_eaten_per_day_l640_640169


namespace quadrilateral_to_cyclic_l640_640378

-- Assumptions that the segments a, b, c, and d can form a quadrilateral
variables {a b c d : ℝ}

-- Definition of a valid quadrilateral (triangle inequalities must be satisfied)
def valid_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

-- The statement we need to prove
theorem quadrilateral_to_cyclic (h : valid_quadrilateral a b c d) : 
  ∃ e : ℝ, e > 0 ∧
  ((e^2 = a^2 + d^2 - 2 * a * d * cos (angle a b c d)) ∧
  (e^2 = b^2 + c^2 + 2 * b * c * cos (angle a b c d))) :=
sorry

end quadrilateral_to_cyclic_l640_640378


namespace l_2003_value_l640_640228

noncomputable def x (n : ℕ) : ℝ := sorry  -- Define x_n based on the recursive relationship
noncomputable def y (n : ℕ) : ℝ := sorry  -- Define y_n based on the recursive relationship
noncomputable def l (n : ℕ) : ℝ := sorry  -- Define l_n based on the geometric progression

theorem l_2003_value :
  let l_2003 := l 2003 in
  l_2003 = 1 / 3^1001 := 
sorry

end l_2003_value_l640_640228


namespace smallest_pos_mult_of_31_mod_97_l640_640077

theorem smallest_pos_mult_of_31_mod_97 {k : ℕ} (h : 31 * k % 97 = 6) : 31 * k = 2015 :=
sorry

end smallest_pos_mult_of_31_mod_97_l640_640077


namespace largest_possible_value_of_s_l640_640407

theorem largest_possible_value_of_s (p q r s : ℝ)
  (h₁ : p + q + r + s = 12)
  (h₂ : pq + pr + ps + qr + qs + rs = 24) : 
  s ≤ 3 + 3 * Real.sqrt 5 :=
sorry

end largest_possible_value_of_s_l640_640407


namespace construct_triangle_from_side_and_altitudes_l640_640830

/-- Given a side BC of a triangle and the altitudes from vertices B and C to the other two sides,
   construct the triangle ABC using ruler and compass -/
theorem construct_triangle_from_side_and_altitudes 
  (B C A B1 C1 : Point)
  (h_BC : LineSegment BC)
  (h_BB1 : Altitude BB1)
  (h_CC1 : Altitude CC1) :
  ∃ (A : Point), Triangle ABC := 
sorry

end construct_triangle_from_side_and_altitudes_l640_640830


namespace count_valid_three_digit_numbers_l640_640613

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, n = 36 ∧ 
    (∀ (a b c : ℕ), a ≠ 0 ∧ c ≠ 0 → 
    ((10 * b + c) % 4 = 0 ∧ (10 * b + a) % 4 = 0) → 
    n = 36) :=
sorry

end count_valid_three_digit_numbers_l640_640613


namespace jar_last_days_l640_640384

theorem jar_last_days :
  let serving_size := 0.5 -- each serving is 0.5 ounces
  let daily_servings := 3  -- James uses 3 servings every day
  let quart_ounces := 32   -- 1 quart = 32 ounces
  let jar_size := quart_ounces - 2 -- container is 2 ounces less than 1 quart
  let daily_consumption := daily_servings * serving_size
  let number_of_days := jar_size / daily_consumption
  number_of_days = 20 := by
  sorry

end jar_last_days_l640_640384


namespace smallest_product_is_neg20_l640_640030

def smallest_product_set : set ℤ := {-10, -4, 0, 1, 2}

theorem smallest_product_is_neg20 :
  ∃ x y ∈ smallest_product_set, x ≠ y ∧ x * y = -20 ∧
  ∀ a b ∈ smallest_product_set, a ≠ b → a * b ≥ -20 :=
by
  sorry

end smallest_product_is_neg20_l640_640030


namespace general_term_formula_l640_640677

-- Define the arithmetic sequence conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 2) = a (n + 1) + d

-- Define the conditions of the problem
def problem_conditions (a : ℕ → ℕ) : Prop :=
  arithmetic_sequence a ∧ a 2 = 8 ∧ (∑ i in Finset.range 10, a (i + 1)) = 185

-- Define the statement of the problem
theorem general_term_formula (a : ℕ → ℕ) (h : problem_conditions a) : 
  ∀ n : ℕ, a n = 3 * n + 2 :=
sorry

end general_term_formula_l640_640677


namespace problem_inequality_solution_l640_640238

noncomputable def find_b_and_c (x : ℝ) (b c : ℝ) : Prop :=
  ∀ x, (x > 2 ∨ x < 1) ↔ x^2 + b*x + c > 0

theorem problem_inequality_solution (x : ℝ) :
  find_b_and_c x (-3) 2 ∧ (2*x^2 - 3*x + 1 ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end problem_inequality_solution_l640_640238


namespace weight_of_4_moles_H2O_is_correct_l640_640520

def atomic_weight_H : ℝ := 1.008 -- g/mol
def atomic_weight_O : ℝ := 16.00 -- g/mol

def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + atomic_weight_O -- g/mol
def moles_H2O : ℝ := 4 -- moles

def weight_of_4_moles_H2O : ℝ := moles_H2O * molecular_weight_H2O -- g

theorem weight_of_4_moles_H2O_is_correct :
  weight_of_4_moles_H2O = 72.064 :=
  by
    rw [weight_of_4_moles_H2O, moles_H2O, molecular_weight_H2O, atomic_weight_H, atomic_weight_O]
    norm_num
    sorry

end weight_of_4_moles_H2O_is_correct_l640_640520


namespace integers_between_1000_and_3000_with_factors_30_45_60_l640_640252

theorem integers_between_1000_and_3000_with_factors_30_45_60 :
  let multiples_180 := { x : ℕ | x % 180 = 0 }
  in count (multiples_180 ∩ { x | 1000 ≤ x ∧ x ≤ 3000 }) = 11 :=
begin
  sorry
end

end integers_between_1000_and_3000_with_factors_30_45_60_l640_640252


namespace Zixuan_amount_l640_640851

noncomputable def amounts (X Y Z : ℕ) : Prop := 
  (X + Y + Z = 50) ∧
  (X = 3 * (Y + Z) / 2) ∧
  (Y = Z + 4)

theorem Zixuan_amount : ∃ Z : ℕ, ∃ X Y : ℕ, amounts X Y Z ∧ Z = 8 :=
by
  sorry

end Zixuan_amount_l640_640851


namespace count_ordered_triples_l640_640807

theorem count_ordered_triples :
  ∀ (x y z : ℕ), 
  x + y + z = 120 → 
  20 < x → x < 60 → 
  20 < y → y < 60 → 
  20 < z → z < 60 → 
  ∃! (x y z : ℕ), 
  x + y + z = 120 ∧ 
  20 < x ∧ x < 60 ∧ 
  20 < y ∧ y < 60 ∧ 
  20 < z ∧ z < 60 :=
⟨λ x y z _, ⟨_, ⟨⟨_, rfl⟩, by linarith⟩⟩⟩

end count_ordered_triples_l640_640807


namespace driver_speed_l640_640116

theorem driver_speed (v : ℝ) : 
  (∀ t > 0, ∀ d > 0, (v + 10) * (t * (3 / 4)) = d → v * t = d) → 
  v = 30 := 
by
  intro h
  have eq1 : (3 / 4) * (v + 10) = v by sorry
  exact sorry

end driver_speed_l640_640116


namespace anna_partition_l640_640765

theorem anna_partition (m n : ℕ) (h1 : 1 < m) :
  ∃ P : (fin (2*m) → fin m), ∀ chosen : (fin m → fin (2*m)),
  ∑ i : fin (2*m), if chosen (P i) = i then (i + 1) else 0 ≠ n := 
sorry

end anna_partition_l640_640765


namespace lcm_inequality_l640_640159

open Nat

theorem lcm_inequality (n : ℕ) (h : n ≥ 5 ∧ Odd n) :
  Nat.lcm n (n + 1) * (n + 2) > Nat.lcm (n + 1) (n + 2) * (n + 3) := by
  sorry

end lcm_inequality_l640_640159


namespace P_vector_space_basis_set_polynomial_expansion_l640_640591

open Polynomial

-- Definition: P is the set of all polynomials of degree <= 4 with rational coefficients
def P : Polynomial ℚ := {p : Polynomial ℚ | degree p ≤ 4}

-- Part (a): Prove P has a vector space structure over ℚ
theorem P_vector_space : ∀ (p q : P), ∃ (r : P), p + q = r ∧ ∀ (a : ℚ) (p : P), ∃ (q : P), a • p = q := sorry

-- Part (b): Prove the given set forms a basis
theorem basis_set : LinearIndependent ℚ ![1, X - 2, (X - 2)^2, (X - 2)^3, (X - 2)^4] ∧
                    span ℚ (Set.range ![1, X - 2, (X - 2)^2, (X - 2)^3, (X - 2)^4]) = ⊤ := sorry

-- Part (c): Express polynomial in the given basis
theorem polynomial_expansion : 
  (7 + 2 * X - 45 * X^2 + 3 * X^4 : Polynomial ℚ) = 
  3 * (X - 2)^4 + 24 * (X - 2)^3 + 27 * (X - 2)^2 - 82 * (X - 2) - 121 := sorry

end P_vector_space_basis_set_polynomial_expansion_l640_640591


namespace dance_steps_equiv_l640_640625

theorem dance_steps_equiv
  (back1 : ℕ)
  (forth1 : ℕ)
  (back2 : ℕ)
  (forth2 : ℕ)
  (back3 : ℕ := 2 * back2) : 
  back1 = 5 ∧ forth1 = 10 ∧ back2 = 2 → 
  (0 - back1 + forth1 - back2 + forth2 = 7) :=
by
  intros h
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw [h1, h3, h4],
  sorry

end dance_steps_equiv_l640_640625


namespace jar_last_days_l640_640385

theorem jar_last_days :
  let serving_size := 0.5 -- each serving is 0.5 ounces
  let daily_servings := 3  -- James uses 3 servings every day
  let quart_ounces := 32   -- 1 quart = 32 ounces
  let jar_size := quart_ounces - 2 -- container is 2 ounces less than 1 quart
  let daily_consumption := daily_servings * serving_size
  let number_of_days := jar_size / daily_consumption
  number_of_days = 20 := by
  sorry

end jar_last_days_l640_640385


namespace sequence_infinite_coprime_l640_640408

theorem sequence_infinite_coprime (a : ℤ) (h : a > 1) :
  ∃ (S : ℕ → ℕ), (∀ n m : ℕ, n ≠ m → Int.gcd (a^(S n + 1) + a^S n - 1) (a^(S m + 1) + a^S m - 1) = 1) :=
sorry

end sequence_infinite_coprime_l640_640408


namespace part1_part2_l640_640990

noncomputable def f (x m : ℝ) : ℝ :=
  (m^2 - 2 * m - 2) * x^m

noncomputable def g (x : ℝ) : ℝ := 
  x^3 + 2 * x - 3

theorem part1 (m : ℝ) (h : ∀ x : ℝ, x > 0 → (f x m)' > 0) : m = 3 := 
  sorry

theorem part2 : set.range (λ x, g x) (set.Icc (-1 : ℝ) 3) = set.Icc (-6 : ℝ) 30 :=
  sorry

end part1_part2_l640_640990


namespace line_intersects_y_axis_at_0_4_l640_640905

theorem line_intersects_y_axis_at_0_4 :
  ∃ m b, (m = (14 - 8) / (5 - 2) ∧ b = 8 - m * 2 ∧ (0, b) = (0, 4)) :=
begin
  let m := (14 - 8) / (5 - 2),
  let b := 8 - m * 2,
  use [m, b],
  split,
  { norm_num, },
  split,
  { norm_num, 
    unfold m,
    norm_num,
   },
  { exact rfl }
end

end line_intersects_y_axis_at_0_4_l640_640905


namespace second_child_birth_year_l640_640739

theorem second_child_birth_year (first_child_birth : ℕ)
  (second_child_birth : ℕ)
  (third_child_birth : ℕ)
  (fourth_child_birth : ℕ)
  (first_child_years_ago : first_child_birth = 15)
  (third_child_on_second_child_fourth_birthday : third_child_birth = second_child_birth + 4)
  (fourth_child_two_years_after_third : fourth_child_birth = third_child_birth + 2)
  (fourth_child_age : fourth_child_birth = 8) :
  second_child_birth = first_child_birth - 14 := 
by
  sorry

end second_child_birth_year_l640_640739


namespace tangent_line_slope_through_origin_perpendicular_tangent_lines_l640_640679

-- Problem (1)
theorem tangent_line_slope_through_origin (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - x) : 
  ∃ k, (∃ m, f' m = k ∧ (0, 0) ∈ set_of (λ p : ℝ × ℝ, p.snd = k * (p.fst - m) + f m)) ∧ k = -1 :=
by sorry

-- Problem (2)
theorem perpendicular_tangent_lines (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - x) :
  ∃ (x0 y0 : ℝ), f' x0 = -1/2 ∧ (y0 = f x0) ∧ (x0, y0) ∈ {(x, y) : x - 2*y - sqrt 2 = 0, x - 2*y + sqrt 2 = 0} :=
by sorry

end tangent_line_slope_through_origin_perpendicular_tangent_lines_l640_640679


namespace ellipse_trajectory_lambda_range_l640_640379

def hyperbola_eq (x y : ℝ) := x^2 - y^2 / 3 = 1
def foci := [(-2, 0), (2, 0)]
def sum_dist_gt_4 (x y : ℝ) := dist (x, y) (-2, 0) + dist (x, y) (2, 0) > 4
def max_prod_dist_eq_9 (x y : ℝ) := (dist (x, y) (-2, 0) * dist (x, y) (2, 0)) ≤ 9
def ellipse_eq (x y : ℝ) := x^2 / 9 + y^2 / 5 = 1

theorem ellipse_trajectory :
  ∀ (P : ℝ × ℝ), 
    hyperbola_eq P.1 P.2 → 
    sum_dist_gt_4 P.1 P.2 → 
    max_prod_dist_eq_9 P.1 P.2 → 
    ellipse_eq P.1 P.2 :=
by
  sorry

def collinear_points (A M B: ℝ × ℝ) 
  (λ : ℝ) := (M.1 = 0 ∧ M.2 = 2) ∧ 
             (A.1 + λ * B.1 = 0 ∧ 
              -2 - A.2 = λ * (B.2 + 2))

theorem lambda_range :
  ∀ (A B : ℝ × ℝ) (λ : ℝ), 
    ellipse_eq A.1 A.2 →
    ellipse_eq B.1 B.2 → 
    collinear_points A (0, 2) B λ → 
    9 - 4 * real.sqrt 5 ≤ λ ∧ 
    λ ≤ 9 + 4 * real.sqrt 5 :=
by
  sorry

end ellipse_trajectory_lambda_range_l640_640379


namespace min_moves_to_balance_stacks_l640_640035

theorem min_moves_to_balance_stacks :
  let stack1 := 9
  let stack2 := 7
  let stack3 := 5
  let stack4 := 10
  let target := 8
  let total_coins := stack1 + stack2 + stack3 + stack4
  total_coins = 31 →
  ∃ moves, moves = 11 ∧
    (stack1 + 3 * moves = target) ∧
    (stack2 + 3 * moves = target) ∧
    (stack3 + 3 * moves = target) ∧
    (stack4 + 3 * moves = target) :=
sorry

end min_moves_to_balance_stacks_l640_640035


namespace chromosome_stability_due_to_meiosis_and_fertilization_l640_640725

/-- Definition of reducing chromosome number during meiosis -/
def meiosis_reduces_chromosome_number (n : ℕ) : ℕ := n / 2

/-- Definition of restoring chromosome number during fertilization -/
def fertilization_restores_chromosome_number (n : ℕ) : ℕ := n * 2

/-- Axiom: Sexual reproduction involves meiosis and fertilization to maintain chromosome stability -/
axiom chromosome_stability (n m : ℕ) (h1 : meiosis_reduces_chromosome_number n = m) 
  (h2 : fertilization_restores_chromosome_number m = n) : n = n

/-- Theorem statement in Lean 4: The chromosome number stability in sexually reproducing organisms is maintained due to meiosis and fertilization -/
theorem chromosome_stability_due_to_meiosis_and_fertilization 
  (n : ℕ) (h_meiosis: meiosis_reduces_chromosome_number n = n / 2) 
  (h_fertilization: fertilization_restores_chromosome_number (n / 2) = n) : 
  n = n := 
by
  apply chromosome_stability
  exact h_meiosis
  exact h_fertilization

end chromosome_stability_due_to_meiosis_and_fertilization_l640_640725


namespace analytical_expression_l640_640117

section
variable (a : ℝ) (α : ℝ)

/-- The function defined with specific parameters, ensuring A > 0 and ω > 0. -/
def f (x : ℝ) : ℝ := 2 * sin (2 * x - π / 6) + 1

/-- The minimum value condition for the function f(x) -/
def min_value : Prop := ∀ x, f x ≥ -1

/-- Condition that the distance between adjacent high points on the graph of f(x) is π -/
def period_condition : Prop := ∃ T, T = π ∧ ∀ x, f(x + T) = f x

/-- Given a value within specified range and solving for α -/
def alpha_condition : Prop := a ∈ (0, π / 2) ∧ f (α / 2) = 2

theorem analytical_expression :
  (∀ x, f x = 2 * sin (2 * x - π / 6) + 1) ∧
  (min_value ∧ period_condition) →
  ∃ α, alpha_condition ∧ α = π / 3 :=
sorry

end

end analytical_expression_l640_640117


namespace solve_for_a_b_l640_640961

open Complex

theorem solve_for_a_b :
  ∃ a b : ℝ, let z : ℂ := 1 + I in
  let lhs := a * z + 2 * b * conj z in
  let rhs := (a + 2 * z) ^ 2 in
  lhs = rhs ∧ ((a = -2 ∧ b = -1) ∨ (a = -4 ∧ b = 2)) :=
begin
  sorry
end

end solve_for_a_b_l640_640961


namespace math_proof_l640_640510

noncomputable def a := 0.70 * 120
noncomputable def b := (6 / 9) * 150
noncomputable def c := 0.80 * 250
noncomputable def d := 0.18 * 180
noncomputable def e := (5 / 7) * 210

theorem math_proof :
  (a - (b / c) - (d * e) = 4776.5) :=
by
  -- Use sorry to skip the proof
  sorry

end math_proof_l640_640510


namespace euler_lines_intersection_and_segment_relation_l640_640967

open EuclideanGeometry

-- Define the prerequisites and construct essential geometric points and properties
variables {A B C A₁ B₁ C₁ P : Point}

-- Assume the existence of the triangle and the altitudes
variables (h_triangle : IsTriangle ABC)
variables (h_altitudes : Altitude A A₁ ∧ Altitude B B₁ ∧ Altitude C C₁)

-- Assume the Euler lines of the three specified triangles
variables (h_euler_lines : IntersectEulerLines (Triangle.mk A B₁ C₁) (Triangle.mk A₁ B C₁) (Triangle.mk A₁ B₁ C) P)

-- Assume P lies on the nine-point circle
variables (h_on_nine_point_circle : OnNinePointCircle P (Triangle.mk A B C))

-- Define the relationship between segments
def segment_relation : Prop :=
  (PA_1 = PB_1 + PC_1) ∨ (PB_1 = PA_1 + PC_1) ∨ (PC_1 = PA_1 + PB_1)

-- The final theorem statement
theorem euler_lines_intersection_and_segment_relation :
  ∃ P, IntersectEulerLines (Triangle.mk A B₁ C₁) (Triangle.mk A₁ B C₁) (Triangle.mk A₁ B₁ C) P ∧
  OnNinePointCircle P (Triangle.mk A B C) ∧
  segment_relation :=
by
  sorry

end euler_lines_intersection_and_segment_relation_l640_640967


namespace toys_in_shipment_l640_640891

theorem toys_in_shipment :
  ∃ T : ℕ, (0.7 * T - 50 = 150) ∧ T = 286 :=
by
  use 286
  rw [algebra.mul_comm (0.7 : ℝ) (286 : ℝ)]
  have h : 0.7 * 286 = 200.2,
  calc
    0.7 * 286 = 200.2 : by norm_num
  calc
    200.2 - 50 = 150.2 : by norm_num, sorry

end toys_in_shipment_l640_640891


namespace count_valid_n_in_range_l640_640197

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def is_multiple_of_3 (x : ℕ) : Prop :=
  x % 3 = 0

theorem count_valid_n_in_range :
  ∃ k, k = 120 ∧ ∀ n ∈ finset.range 201, is_multiple_of_3 (tens_digit (n * n)) →
  (∃ huntynn, huntynn = finset.filter (fun n => is_multiple_of_3 (tens_digit (n * n))) (finset.range 201) ∧ finset.card huntynn = k) :=
sorry

end count_valid_n_in_range_l640_640197


namespace find_m_l640_640653

def arithmetic_seq (a : ℕ) (b : ℕ) (n : ℕ) : ℕ := a + (n - 1) * b
def geometric_seq (b : ℕ) (a : ℕ) (n : ℕ) : ℕ := b * a^(n - 1)
def c_seq (a_seq : ℕ → ℕ) (b_seq : ℕ → ℕ) (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | k + 1 => if k = 1 then 4 else 2 * b_seq 1 -- simplified insertion logic for the context

theorem find_m (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_seq : a = b ∧ a + b = b * a) :
  ∀ n, (c_seq (arithmetic_seq a b) (geometric_seq b a) 1) + 
       (c_seq (arithmetic_seq a b) (geometric_seq b a) 2) + 
       (c_seq (arithmetic_seq a b) (geometric_seq b a) n) = 
       2 * (c_seq (arithmetic_seq a b) (geometric_seq b a) (n + 1)) → n = 2 :=
by
  sorry

end find_m_l640_640653


namespace rooms_containing_people_after_one_hour_l640_640814

theorem rooms_containing_people_after_one_hour :
  let movement : ∀ (r : ℕ → ℕ), ℕ → ℕ → ℕ := 
    λ t r -> if t <= 1 then 0 else (r - 1) + t else r,
    num_rooms := 1000,
    initial_state : Finₓ num_rooms → ℕ := 
    λ i => if i = 0 then 1000 else 0,
    people_after_one_hour : ℕ := 60 in
  let final_state : Finₓ num_rooms → ℕ :=
    λ i => initial_state i fun r t i =>
    if t = 0 then initial_state i
    else movement (final_state (i - 1)) else final_state i
  in ∃ cnt, cnt = finset.count (λ i => final_state i > 0) (finset.range num_rooms) = 31 :=
sorry

end rooms_containing_people_after_one_hour_l640_640814


namespace rectangular_equation_of_l_range_of_m_intersection_l640_640369

-- Definitions based on the problem conditions
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  let x := sqrt 3 * cos (2 * t)
  let y := 2 * sin t
  (x, y)

def polar_line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- The rectangular equation converted from polar form
theorem rectangular_equation_of_l (x y m : ℝ) :
  (polar_line_l (sqrt (x^2 + y^2)) (atan2 y x) m) ↔ (√3 * x + y + 2 * m = 0) :=
sorry

-- Range of values for m ensuring intersection of line l with curve C
theorem range_of_m_intersection (m : ℝ) :
  ((-19/12 : ℝ) ≤ m ∧ m ≤ (5/2 : ℝ)) ↔
  ∃ t : ℝ, let x := sqrt 3 * cos (2 * t)
           let y := 2 * sin t
           (sqrt 3 * x + y + 2 * m = 0 ∧ (-2 ≤ y ∧ y ≤ 2)) :=
sorry

end rectangular_equation_of_l_range_of_m_intersection_l640_640369


namespace problem_statement_l640_640942

def Omega (n : ℕ) : ℕ := 
  -- Number of prime factors of n, counting multiplicity
  sorry

def f1 (n : ℕ) : ℕ :=
  -- Sum of positive divisors d|n where Omega(d) ≡ 1 (mod 4)
  sorry

def f3 (n : ℕ) : ℕ :=
  -- Sum of positive divisors d|n where Omega(d) ≡ 3 (mod 4)
  sorry

theorem problem_statement : 
  f3 (6 ^ 2020) - f1 (6 ^ 2020) = (1 / 10 : ℚ) * (6 ^ 2021 - 3 ^ 2021 - 2 ^ 2021 - 1) :=
sorry

end problem_statement_l640_640942


namespace minimum_buses_for_second_group_l640_640868

theorem minimum_buses_for_second_group (max_students_per_bus : ℕ) (total_students : ℕ) (max_buses_first_group : ℕ) (capacity_first_group : ℕ) :
  max_students_per_bus = 45 →
  total_students = 550 →
  max_buses_first_group = 8 →
  capacity_first_group = 8 * 45 →
  ∃ min_buses_second_group : ℕ, min_buses_second_group = 5 :=
by
  intros _ _ _ _
  use 5
  sorry

end minimum_buses_for_second_group_l640_640868


namespace equal_pair_only_C_l640_640567

theorem equal_pair_only_C :
  let p1 := 2^3
  let p2 := (-3)^2
  let p3 := -3^2
  let p4 := (-3)^2
  let p5 := -3^3
  let p6 := (-3)^3
  let p7 := -3 * 2^3
  let p8 := (-3 * 2)^3
  (p1 ≠ p2) ∧ (p3 ≠ p4) ∧ (p5 = p6) ∧ (p7 ≠ p8) :=
by
  let p1 := 2^3
  let p2 := (-3)^2
  let p3 := -3^2
  let p4 := (-3)^2
  let p5 := -3^3
  let p6 := (-3)^3
  let p7 := -3 * 2^3
  let p8 := (-3 * 2)^3
  have h1 : p1 = 8 := rfl
  have h2 : p2 = 9 := rfl
  have h3 : p3 = -9 := rfl
  have h4 : p4 = 9 := rfl
  have h5 : p5 = -27 := rfl
  have h6 : p6 = -27 := rfl
  have h7 : p7 = -24 := rfl
  have h8 : p8 = -216 := rfl
  exact ⟨h1.symm ▸ h2.symm ▸ ne_of_lt (by norm_num : 8 < 9),
         h3.symm ▸ h4.symm ▸ ne_of_lt (by norm_num : -9 < 9),
         h5.symm ▸ h6.symm ▸ eq.refl (-27),
         h7.symm ▸ h8.symm ▸ ne_of_lt (by norm_num : -24 < -216)⟩

end equal_pair_only_C_l640_640567


namespace find_first_number_l640_640271

theorem find_first_number
  (x y : ℝ)
  (h1 : y = 3.0)
  (h2 : x * y + 4 = 19) : x = 5 := by
  sorry

end find_first_number_l640_640271


namespace rect_eq_and_range_of_m_l640_640312

noncomputable def rect_eq_of_polar (m : ℝ) : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ * sin (θ + π / 3) + m = 0 → sqrt 3 * (ρ * cos θ) + (ρ * sin θ) + 2 * m = 0

noncomputable def range_of_m_for_intersection : set ℝ := 
  {m : ℝ | -19 / 12 ≤ m ∧ m ≤ 5 / 2}

theorem rect_eq_and_range_of_m (m : ℝ) : 
  rect_eq_of_polar m ∧ (∀ t : ℝ, ∃ x y : ℝ, x = sqrt 3 * cos (2 * t) ∧ y = 2 * sin t ∧ 
    (sqrt 3 * x + y + 2 * m = 0) → m ∈ range_of_m_for_intersection) :=
by sorry

end rect_eq_and_range_of_m_l640_640312


namespace find_y_l640_640273

noncomputable def x : ℝ := 3.87

def y (x : ℝ) : ℝ := 2 * (Real.log x)^3 - 5 / 3

theorem find_y :
  y x ≈ -1.2613 := by
  sorry

end find_y_l640_640273


namespace complete_collection_probability_l640_640056

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end complete_collection_probability_l640_640056


namespace no_groups_of_six_l640_640121

theorem no_groups_of_six (x y z : ℕ) 
  (h1 : (2 * x + 6 * y + 10 * z) / (x + y + z) = 5)
  (h2 : (2 * x + 30 * y + 90 * z) / (2 * x + 6 * y + 10 * z) = 7) : 
  y = 0 := 
sorry

end no_groups_of_six_l640_640121


namespace no_solution_eq_l640_640621

theorem no_solution_eq (k : ℝ) :
  (¬ ∃ x : ℝ, x ≠ 3 ∧ x ≠ 7 ∧ (x + 2) / (x - 3) = (x - k) / (x - 7)) ↔ k = 2 :=
by
  sorry

end no_solution_eq_l640_640621


namespace sum_of_product_of_consecutive_numbers_divisible_by_12_l640_640810

theorem sum_of_product_of_consecutive_numbers_divisible_by_12 (a : ℤ) : 
  (a * (a + 1) + (a + 1) * (a + 2) + (a + 2) * (a + 3) + a * (a + 3) + 1) % 12 = 0 :=
by sorry

end sum_of_product_of_consecutive_numbers_divisible_by_12_l640_640810


namespace factorial_division_l640_640659

theorem factorial_division : ∀ (a b : ℕ), a = 10! ∧ b = 4! → a / b = 151200 :=
by
  intros a b h
  cases h with ha hb
  rw [ha, Nat.factorial, Nat.factorial] at hb,
  norm_num at hb,
  exact sorry

end factorial_division_l640_640659


namespace polar_to_rectangular_range_of_m_l640_640292

-- Definition of the parametric equations of curve C
def curve (t : ℝ) : ℝ × ℝ :=
  (sqrt(3) * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line (rho theta m : ℝ) : Prop :=
  rho * sin (theta + π / 3) + m = 0

-- Definition of the rectangular equation of line l
def rectangular_line (x y m : ℝ) : Prop :=
  sqrt(3) * x + y + 2 * m = 0

-- Convert polar line equation to rectangular form
theorem polar_to_rectangular (rho theta m x y : ℝ)
  (h1 : polar_line rho theta m)
  (h2 : rho = sqrt (x^2 + y^2))
  (h3 : cos theta * rho = x ∧ sin theta * rho = y) :
  rectangular_line x y m :=
sorry

-- Range of values for m for which l intersects C
theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, (sqrt(3) * cos (2 * t), 2 * sin t) ∈ {p : ℝ × ℝ | rectangular_line p.1 p.2 m}) ↔
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end polar_to_rectangular_range_of_m_l640_640292


namespace factorial_division_l640_640666

theorem factorial_division : (10! / 4! = 151200) :=
by
  have fact_10 : 10! = 3628800 := by sorry
  rw [fact_10]
  -- Proceeding with calculations and assumptions that follow directly from the conditions
  have fact_4 : 4! = 24 := by sorry
  rw [fact_4]
  exact (by norm_num : 3628800 / 24 = 151200)

end factorial_division_l640_640666


namespace smallest_three_digit_palindrome_l640_640191

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let n_str := toString n
  n_str = n_str.reverse

-- Define the conditions stated in a)
def is_three_digit_palindrome (x : ℕ) : Prop :=
  (100 ≤ x) ∧ (x < 1000) ∧ is_palindrome x

def is_five_digit_not_palindrome (n : ℕ) : Prop :=
  (10000 ≤ n) ∧ (n < 100000) ∧ ¬ is_palindrome n

-- The statement we want to prove
theorem smallest_three_digit_palindrome :
  ∃ x : ℕ, is_three_digit_palindrome x ∧ is_five_digit_not_palindrome (102 * x) ∧ ∀ y : ℕ, 
    is_three_digit_palindrome y ∧ is_five_digit_not_palindrome (102 * y) → x ≤ y := 
by
  sorry

end smallest_three_digit_palindrome_l640_640191


namespace domain_log_sqrt_l640_640011

noncomputable def domain_function : Set ℝ :=
  {x : ℝ | x > -1 ∧ -x^2 - 3*x + 4 > 0}

theorem domain_log_sqrt {x : ℝ} : x > -1 ∧ -x^2 - 3*x + 4 > 0 ↔ x ∈ Ioo (-1) 1 :=
by
  split
  {
    intro h,
    cases h with h1 h2,
    split,
    {
      linarith,
    },
    {
      have : x^2 + 3*x - 4 < 0 := by linarith,
      contrapose! this,
      simp only [not_lt, sub_nonneg],
      ring,
    },
  },
  {
    intro h,
    cases h with h1 h2,
    split,
    {
      exact h1,
    },
    {
      linarith,
    },
  }

end domain_log_sqrt_l640_640011


namespace scientific_notation_correct_l640_640150

theorem scientific_notation_correct :
  0.00000032 = 3.2 * (10 : ℝ) ^ (-7) := sorry

end scientific_notation_correct_l640_640150


namespace find_S2019_l640_640727

noncomputable theory

variables (a1 d : ℤ) (S : ℕ → ℤ)

-- Define the initial conditions
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) := a1 + (n - 1) * d
def sum_of_terms (a1 d : ℤ) (S : ℕ → ℤ) (n : ℕ) := n / 2 * (2 * a1 + (n - 1) * d)

-- Given conditions
axiom initial_term : a1 = -2018
axiom diff_condition : (S 15) / 15 - (S 10) / 10 = 5

-- Prove that S_2019 = 0
theorem find_S2019 : (S 2019) = 0 :=
sorry

end find_S2019_l640_640727


namespace find_constant_c_l640_640797

theorem find_constant_c (c : ℝ) :
  (∀ x y : ℝ, x + y = c ∧ y - (2 + 5) / 2 = x - (8 + 11) / 2) →
  (c = 13) :=
by
  sorry

end find_constant_c_l640_640797


namespace pistachio_shells_percentage_l640_640107

theorem pistachio_shells_percentage (total_pistachios : ℕ) (opened_shelled_pistachios : ℕ) (P : ℝ) :
  total_pistachios = 80 →
  opened_shelled_pistachios = 57 →
  (0.75 : ℝ) * (P / 100) * (total_pistachios : ℝ) = (opened_shelled_pistachios : ℝ) →
  P = 95 :=
by
  intros h_total h_opened h_equation
  sorry

end pistachio_shells_percentage_l640_640107


namespace dot_product_parallel_vectors_l640_640249

variable (x : ℝ)
def a : ℝ × ℝ := (x, x - 1)
def b : ℝ × ℝ := (1, 2)
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

theorem dot_product_parallel_vectors
  (h_parallel : are_parallel (a x) b)
  (h_x : x = -1) :
  (a x).1 * (b).1 + (a x).2 * (b).2 = -5 :=
by
  sorry

end dot_product_parallel_vectors_l640_640249


namespace find_u_l640_640461

theorem find_u 
    (a b c p q u : ℝ) 
    (H₁: (∀ x, x^3 + 2*x^2 + 5*x - 8 = 0 → x = a ∨ x = b ∨ x = c))
    (H₂: (∀ x, x^3 + p*x^2 + q*x + u = 0 → x = a+b ∨ x = b+c ∨ x = c+a)) :
    u = 18 :=
by 
    sorry

end find_u_l640_640461


namespace smallest_abcde_l640_640875

noncomputable def digit_valid (n : ℕ) : Prop :=
  n < 100000 ∧ n > 9999

noncomputable def multiple_of (num divisor : ℕ) : Prop :=
  num % divisor = 0

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (List.range n).count (fun x => x > 0 ∧ n % x = 0)

theorem smallest_abcde (ABCDE : ℕ) :
  digit_valid ABCDE ∧
  multiple_of ABCDE 2014 ∧
  count_divisors (ABCDE % 1000) = 16 →
  ABCDE = 24168 :=
by
  sorry

end smallest_abcde_l640_640875


namespace area_under_arcsin_cos_l640_640935

open Real

def f (x : ℝ) : ℝ := arcsin (cos x)

theorem area_under_arcsin_cos (a b : ℝ) (h : a = π / 4 ∧ b = 9 * π / 4) :
  ∫ x in a..b, f x = 2 * π^2 :=
by
  sorry

end area_under_arcsin_cos_l640_640935


namespace kanul_cash_percentage_l640_640387

theorem kanul_cash_percentage (raw_materials : ℕ) (machinery : ℕ) (total_amount : ℕ) (cash_percentage : ℕ)
  (H1 : raw_materials = 80000)
  (H2 : machinery = 30000)
  (H3 : total_amount = 137500)
  (H4 : cash_percentage = 20) :
  ((total_amount - (raw_materials + machinery)) * 100 / total_amount) = cash_percentage := by
    sorry

end kanul_cash_percentage_l640_640387


namespace taller_is_mother_l640_640064

theorem taller_is_mother (taller shorter : Type) 
  (son : shorter is_son_of taller) 
  (not_father : ¬ taller is_father_of shorter) : 
  taller is_mother_of shorter := 
sorry

end taller_is_mother_l640_640064


namespace integral_cos_expression_l640_640678

-- Given the binomial (x - 1 / sqrt x) ^ 6 and its expansion constant term
def binomial_constant_term : ℕ := 6
def constant_term_in_expansion : ℚ := 15

-- The proof problem: prove the integral equals the given value
theorem integral_cos_expression :
  ∫ x in 0..(Real.pi / 2), Real.cos ((constant_term_in_expansion * x) / 5) = -1 / 3 :=
by
  sorry

end integral_cos_expression_l640_640678


namespace smallest_natural_number_ends_with_6_l640_640938

theorem smallest_natural_number_ends_with_6 
  (n : ℕ) 
  (h : n % 10 = 6)
  (h_condition : 6 * 10 ^ (nat.log10 (n / 10) + 1) + (n / 10) = 4 * n) : 
  n = 153846 :=
by
  sorry

end smallest_natural_number_ends_with_6_l640_640938


namespace union_of_sets_l640_640971

theorem union_of_sets (A B : Set ℤ) (hA : A = {-1, 3}) (hB : B = {2, 3}) : A ∪ B = {-1, 2, 3} := 
by
  sorry

end union_of_sets_l640_640971


namespace prove_inequality_l640_640763

noncomputable def inequality (a b : ℝ) : Prop :=
  0 ≤ a → 0 ≤ b → (1 / 2 * (a + b)^2 + 1 / 4 * (a + b) ≥ a * real.sqrt b + b * real.sqrt a)

theorem prove_inequality (a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) : inequality a b :=
by sorry

end prove_inequality_l640_640763


namespace circle_standard_equation_l640_640192

theorem circle_standard_equation {a : ℝ} :
  (∃ a : ℝ, a ≠ 0 ∧ (a = 2 * a - 3 ∨ a = 3 - 2 * a) ∧ 
  (((x - a)^2 + (y - (2 * a - 3))^2 = a^2) ∧ 
   ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1))) :=
sorry

end circle_standard_equation_l640_640192


namespace cans_collected_is_232_l640_640015

-- Definitions of the conditions
def total_students : ℕ := 30
def half_students : ℕ := total_students / 2
def cans_per_half_student : ℕ := 12
def remaining_students : ℕ := 13
def cans_per_remaining_student : ℕ := 4

-- Calculate total cans collected
def total_cans_collected : ℕ := (half_students * cans_per_half_student) + (remaining_students * cans_per_remaining_student)

-- The theorem to be proved
theorem cans_collected_is_232 : total_cans_collected = 232 := by
  -- Proof would go here
  sorry

end cans_collected_is_232_l640_640015


namespace range_of_a_l640_640987

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x = 1 → ¬ ((x + 1) / (x + a) < 2))) ↔ -1 ≤ a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l640_640987


namespace a_needed_b_not_sufficient_l640_640263

def purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

noncomputable def a_zero_is_nec_but_not_suf (a b : ℝ) : Prop :=
  ∀ (z : ℂ), (z = a + b * complex.i) → (¬ purely_imaginary z) → a = 0

noncomputable def a_zero_is_not_suf (b : ℝ) : Prop :=
  (∀ (a : ℝ), ¬purely_imaginary (a + b * complex.i)) → b = 0

theorem a_needed_b_not_sufficient (a b : ℝ) : 
  a_zero_is_nec_but_not_suf a b ∧ ¬a_zero_is_not_suf b :=
sorry

end a_needed_b_not_sufficient_l640_640263


namespace polar_to_rectangular_intersection_range_l640_640342

-- Definitions of parametric equations for curve C
def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line_l (ρ θ m: ℝ) : Prop := ρ * sin (θ + π / 3) + m = 0

-- Rectangular equation of line l
def rectangular_line_l (x y m: ℝ) : Prop := sqrt 3 * x + y + 2 * m = 0

-- Prove that polar equation of line l can be transformed to rectangular form
theorem polar_to_rectangular (ρ θ m : ℝ) :
  (polar_line_l ρ θ m) → (∃ x y, ρ = sqrt (x^2 + y^2) ∧ θ = arctan (y / x) ∧ rectangular_line_l x y m) :=
  sorry

-- Prove that the range of values for m ensuring line l intersects curve C is [-19/12, 5/2]
theorem intersection_range (m : ℝ) :
  (∃ t : ℝ, rectangular_line_l (sqrt 3 * cos (2 * t)) (2 * sin t) m) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
  sorry

end polar_to_rectangular_intersection_range_l640_640342


namespace airplane_altitude_l640_640137

open Real

noncomputable def altitude_airplane (AB : ℝ) (θA : ℝ) (θB : ℝ) : ℝ :=
  let AC := AB / (cos θB * cot θA + sin θB)
  AC * sin θA

theorem airplane_altitude (
  h : ℝ :=
  let AB := 12
  let AC := 12 / (cos (π / 4) * cot (π / 6) + sin (π / 4))
  AC * sin (π / 6)) :
  h = 6 * sqrt 2 := 
sorry

end airplane_altitude_l640_640137


namespace multiplication_problem_l640_640431

noncomputable def problem_statement (x : ℂ) : Prop :=
  (x^4 + 30 * x^2 + 225) * (x^2 - 15) = x^6 - 3375

theorem multiplication_problem (x : ℂ) : 
  problem_statement x :=
sorry

end multiplication_problem_l640_640431


namespace sequence_sum_property_l640_640373

noncomputable def a : ℕ → ℤ
| 1       := 2
| (n + 1) := 1 - a n

def S (n : ℕ) : ℤ := ∑ i in Finset.range n, a (i + 1)

theorem sequence_sum_property :
  S 2017 - 2 * S 2018 + S 2019 = 3 :=
sorry

end sequence_sum_property_l640_640373


namespace main_l640_640820

open set

structure Proposition (α β γ : Type*) :=
(perpendicular : α → β → Prop)
(parallel : α → β → Prop)
(subset : α → β → Prop)

-- Definitions for each condition
def Prop1 {α β γ : Type*} [Proposition α β γ]
  (H1 : Proposition.perpendicular α β)
  (H2 : Proposition.perpendicular β γ) :
  Proposition.parallel α β := sorry

def Prop2 {a b c : Type*} [Proposition a b c]
  (H1 : Proposition.perpendicular a b)
  (H2 : Proposition.perpendicular b c) :
  Proposition.parallel a c ∨ Proposition.perpendicular a c := sorry

def Prop3 {a b c α β : Type*} [Proposition a α β] [Proposition b β β]
  (H1 : Proposition.subset a α)
  (H2 : Proposition.subset b β)
  (H3 : Proposition.perpendicular a b)
  (H4 : Proposition.perpendicular a c) :
  Proposition.perpendicular α β := sorry

def Prop4 {a α β : Type*} [Proposition a α β]
  (H1 : Proposition.perpendicular a α)
  (H2 : Proposition.subset b β)
  (H3 : Proposition.parallel a b) :
  Proposition.perpendicular α β := sorry

-- Main theorem: Exactly one of the propositions is true.
theorem main : 
  (Prop4 ∧ ¬ Prop1 ∧ ¬ Prop2 ∧ ¬ Prop3) ∨
  (Prop1 ∧ ¬ Prop2 ∧ ¬ Prop3 ∧ ¬ Prop4) ∨
  (Prop2 ∧ ¬ Prop1 ∧ ¬ Prop3 ∧ ¬ Prop4) ∨
  (Prop3 ∧ ¬ Prop1 ∧ ¬ Prop2 ∧ ¬ Prop4) :=
sorry

end main_l640_640820


namespace measure_of_angle_Q_l640_640282

theorem measure_of_angle_Q (Q R : ℝ) 
  (h1 : Q = 2 * R)
  (h2 : 130 + 90 + 110 + 115 + Q + R = 540) :
  Q = 63.33 :=
by
  sorry

end measure_of_angle_Q_l640_640282


namespace cubes_painted_identically_after_rotation_probability_l640_640041

def num_ways_to_paint_cube : ℕ := 30

def prob_identical_rotated_cubes : Real := 1/45

theorem cubes_painted_identically_after_rotation_probability :
  (num_ways_to_paint_cube^2) * prob_identical_rotated_cubes = 20 := 
sorry

end cubes_painted_identically_after_rotation_probability_l640_640041


namespace rect_eq_line_range_of_m_l640_640330

-- Definitions based on the conditions
def parametric_x (t : ℝ) : ℝ := sqrt 3 * cos (2 * t)
def parametric_y (t : ℝ) : ℝ := 2 * sin t

def polar_rho (θ m : ℝ) : ℝ := -(m / (sin (θ + π / 3)))

-- Rectangular equation of the line l
theorem rect_eq_line (x y m : ℝ) : polar_rho (atan2 y x) m * sin (atan2 y x + π / 3) + m = 0 → sqrt 3 * x + y + 2 * m = 0 :=
by
  sorry

-- Range of values for m
theorem range_of_m (C : ℝ → ℝ × ℝ)
  (hC : ∀ t, C t = (parametric_x t, parametric_y t)) (m : ℝ) :
  (∃ t, sqrt 3 * fst (C t) + snd (C t) + 2 * m = 0) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
by
  sorry

end rect_eq_line_range_of_m_l640_640330


namespace initial_ribbon_tape_length_l640_640735

namespace RibbonTapeProblem

def ribbonTapeUsedPerRibbon : ℝ := 0.84
def ribbonsMade : ℝ := 10
def remainingTapeInMeters : ℝ := 0.50
def initialTape : ℝ := 8.9

theorem initial_ribbon_tape_length (h_used : ribbonTapeUsedPerRibbon * ribbonsMade = 8.4) 
                                   (h_remaining : remainingTapeInMeters = 0.50) :
    initialTape = ribbonTapeUsedPerRibbon * ribbonsMade + remainingTapeInMeters := by
  calc
    initialTape = 8.9 := by rfl
    _ = 8.4 + 0.50 := by congr; assumption; assumption
    _ = ribbonTapeUsedPerRibbon * ribbonsMade + remainingTapeInMeters := by congr; assumption; assumption

end RibbonTapeProblem

end initial_ribbon_tape_length_l640_640735


namespace problem_l640_640210

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2^x - 1 else Real.log 2 (abs x) + 1

theorem problem (x₀ x₁ : ℝ) (hx₀ : x₀ = 1) (hx₁ : x₁ = f 1) : f x₁ = 2 :=
by 
  sorry

end problem_l640_640210


namespace min_modulus_of_complex_m_l640_640477

noncomputable def m_min_modulus (m : ℂ) : ℝ :=
  complex.abs m

theorem min_modulus_of_complex_m (m : ℂ)
  (h : ∃ x : ℝ, (x^2 + m.re * x + 1 = 0 ∧ m.im * x + 2 = 0 )) :
  m_min_modulus m = real.sqrt (2 + 2 * real.sqrt 5) :=
sorry

end min_modulus_of_complex_m_l640_640477


namespace metallurgist_alloy_l640_640552

theorem metallurgist_alloy (x : ℝ) : 
  (6.2 * x + 6.2 * 0.4) / 12.4 = 0.5 → 
  x = 0.6 :=
by 
  intro h
  linarith [mul_eq_mul_right_iff.mp (h.mul_right 12.4)]

sorry

end metallurgist_alloy_l640_640552


namespace point_outside_circle_l640_640801

theorem point_outside_circle {P : ℝ × ℝ} (hP : P = (2, 5))
  {O : ℝ × ℝ} (hO : O = (0, 0))
  {r : ℝ} (hr : r = 2 * Real.sqrt 6)
  (h_eq : ∀ (x y : ℝ), x^2 + y^2 = 24 ↔ x = 0 ∧ y = 0 ∨ (x, y) ≠ (0, 0))
  : Real.dist P O > r := 
by
  sorry

end point_outside_circle_l640_640801


namespace intervals_where_f_is_increasing_find_a_l640_640688

noncomputable def f (x : ℝ) : ℝ :=
  sin (2 * x - π / 6) + 2 * cos x ^ 2 - 1

theorem intervals_where_f_is_increasing (k : ℤ) :
  ∀ x ∈ set.Icc (k * π - π / 3) (k * π + π / 6), deriv f x ≥ 0 :=
sorry

structure triangle :=
  (a b c : ℝ)
  (angle_A : ℝ)

variables (t : triangle)
  (hArithmetic : 2 * t.a = t.b + t.c)
  (hDotProduct : t.b * t.c * cos t.angle_A = 9)
  (hAngle_A : f t.angle_A = 1 / 2)

theorem find_a : t.a = 3 * sqrt 2 :=
sorry

end intervals_where_f_is_increasing_find_a_l640_640688


namespace solve_inequality_l640_640780

-- We will define the conditions and corresponding solution sets
def solution_set (a x : ℝ) : Prop :=
  (a < -1 ∧ (x > -a ∨ x < 1)) ∨
  (a = -1 ∧ x ≠ 1) ∨
  (a > -1 ∧ (x < -a ∨ x > 1))

theorem solve_inequality (a x : ℝ) :
  (x - 1) * (x + a) > 0 ↔ solution_set a x :=
by
  sorry

end solve_inequality_l640_640780


namespace complete_collection_probability_l640_640055

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end complete_collection_probability_l640_640055


namespace transform_cos_to_sin_l640_640980

theorem transform_cos_to_sin (x : ℝ) :
  let C1 := λ x, Real.cos x in
  let C2 := λ x, Real.sin (2 * x + (2 * Real.pi / 3)) in
  C2 x = C1 (x / 2 + Real.pi / 12) :=
by
  sorry

end transform_cos_to_sin_l640_640980


namespace parallel_vectors_m_value_l640_640996

theorem parallel_vectors_m_value :
  ∀ (m : ℝ), (∃ (λ : ℝ), (2 * m + 1, 3, m - 1) = (2 * λ, λ * m, -λ * m)) ↔ m = -2 :=
by
  sorry

end parallel_vectors_m_value_l640_640996


namespace minimize_total_distance_l640_640783

theorem minimize_total_distance (h1 h2 h3 h4 h5 : ℕ) 
    (h1_orders : h1 = 1)
    (h2_orders : h2 = 2)
    (h3_orders : h3 = 3)
    (h4_orders : h4 = 4)
    (h5_orders : h5 = 10) : 
  ∃ house, house = 5 :=
by
  use 5
  sorry

end minimize_total_distance_l640_640783


namespace rooms_containing_people_after_one_hour_l640_640815

theorem rooms_containing_people_after_one_hour :
  let movement : ∀ (r : ℕ → ℕ), ℕ → ℕ → ℕ := 
    λ t r -> if t <= 1 then 0 else (r - 1) + t else r,
    num_rooms := 1000,
    initial_state : Finₓ num_rooms → ℕ := 
    λ i => if i = 0 then 1000 else 0,
    people_after_one_hour : ℕ := 60 in
  let final_state : Finₓ num_rooms → ℕ :=
    λ i => initial_state i fun r t i =>
    if t = 0 then initial_state i
    else movement (final_state (i - 1)) else final_state i
  in ∃ cnt, cnt = finset.count (λ i => final_state i > 0) (finset.range num_rooms) = 31 :=
sorry

end rooms_containing_people_after_one_hour_l640_640815


namespace two_initial_values_finite_seq_l640_640405

noncomputable def g (x : ℝ) : ℝ := 3 * x - x^2

def sequence (y0 : ℝ) : ℕ → ℝ
| 0       := y0
| (n + 1) := g (sequence n)

def is_finite_values (ys : ℕ → ℝ) : Prop :=
∃ (s : set ℝ), finite s ∧ (∀ n, ys n ∈ s)

theorem two_initial_values_finite_seq : {y0 : ℝ | is_finite_values (sequence y0)} = {0, 2} :=
sorry

end two_initial_values_finite_seq_l640_640405


namespace problem_equivalence_l640_640912

theorem problem_equivalence : 
  sqrt 12 + (2014 - 2015)^0 + (1/4)^(-1) - 6 * tan (real.pi / 6) = 5 :=
by
  have h1: sqrt 12 = 2 * sqrt 3 := by sorry
  have h2: (2014 - 2015)^0 = 1 := by sorry
  have h3: (1 / 4)^(-1) = 4 := by sorry
  have h4: tan (real.pi / 6) = sqrt 3 / 3 := by sorry
  sorry

end problem_equivalence_l640_640912


namespace max_distinct_values_l640_640021

theorem max_distinct_values (f : ℝ^3 → ℝ)
  (h : ∀ (u v : ℝ^3) (α β : ℝ), f (α • u + β • v) ≤ max (f u) (f v)) :
  ∃ T : ℕ, T ≤ 4 ∧ ∀ x y : set ℝ, x.countable → y.countable → f '' x ∪ f '' y ⊆ {0, 1, 2, 3} := sorry

end max_distinct_values_l640_640021


namespace sum_of_integers_with_product_32_and_difference_4_l640_640028

theorem sum_of_integers_with_product_32_and_difference_4 :
  ∃ (a b : ℕ), a * b = 32 ∧ |a - b| = 4 ∧ a + b = 12 :=
by
  sorry

end sum_of_integers_with_product_32_and_difference_4_l640_640028


namespace s_t_inequality_l640_640619

noncomputable def s : ℝ → ℝ := sorry
noncomputable def t : ℝ → ℝ := sorry

axiom s_0_eq_t_0 (h : s(0) = t(0)) : 0 < s(0)
axiom s'_sqrt_t' (x : ℝ) (h : x ∈ set.Icc 0 1) : (deriv s x) * (real.sqrt (deriv t x)) = 5

theorem s_t_inequality (x : ℝ) (hx : x ∈ set.Icc 0 1) : 2 * s x + 5 * t x > 15 * x := by
  have h : s(0) = t(0) := sorry
  have h0 : 0 < s(0) := s_0_eq_t_0 h
  have h1 : (deriv s x) * (real.sqrt (deriv t x)) = 5 := s'_sqrt_t' x hx
  sorry

end s_t_inequality_l640_640619


namespace diff_of_cubes_is_sum_of_squares_l640_640442

theorem diff_of_cubes_is_sum_of_squares (n : ℤ) : 
  (n+2)^3 - n^3 = n^2 + (n+2)^2 + (2*n+2)^2 := 
by sorry

end diff_of_cubes_is_sum_of_squares_l640_640442


namespace find_x_if_alpha_beta_eq_4_l640_640411

def alpha (x : ℝ) : ℝ := 4 * x + 9
def beta (x : ℝ) : ℝ := 9 * x + 6

theorem find_x_if_alpha_beta_eq_4 :
  (∃ x : ℝ, alpha (beta x) = 4 ∧ x = -29 / 36) :=
by
  sorry

end find_x_if_alpha_beta_eq_4_l640_640411


namespace generalTerm_bSumFormula_l640_640969

-- Arithmetic sequence with non-zero common difference
def arithmeticSeq (a₁ d : ℕ) := λ n : ℕ, a₁ + (n - 1) * d

-- Sum of the first 4 terms equals 20
def sumFirstFourTerms (a₁ d : ℕ) := 4 * a₁ + 6 * d = 20

-- The first, second and fourth terms form a geometric sequence
def geomSeqCondition (a₁ d : ℕ) := (a₁ + d)^2 = a₁ * (a₁ + 3 * d)

-- Prove the general term of the sequence
theorem generalTerm (a₁ d : ℕ) (h₁ : sumFirstFourTerms a₁ d) (h₂ : geomSeqCondition a₁ d) :
  ∀ n, arithmeticSeq a₁ d n = 2 * n :=
sorry

-- Define b_n and the sum of first n terms
def bSeq (a : ℕ → ℕ) := λ n : ℕ, n * 2^(a n)
def bSum (a : ℕ → ℕ) := λ n : ℕ, ∑ i in finset.range n, bSeq a i

-- Prove the sum of the first n terms of b_n
theorem bSumFormula (a₁ d : ℕ) (h₁ : sumFirstFourTerms a₁ d) (h₂ : geomSeqCondition a₁ d)
  (h₃ : ∀ n, arithmeticSeq a₁ d n = 2 * n) :
  ∀ n, bSum (arithmeticSeq a₁ d) n = (3 * n - 1) * 4^(n + 1) + 4 / 9 :=
sorry

end generalTerm_bSumFormula_l640_640969


namespace shaded_area_is_30_l640_640901

theorem shaded_area_is_30 :
  ∃ (large_area small_area shaded_area : ℝ), 
    (large_area = 1 / 2 * 10 * 10) ∧
    (small_area = large_area / 25) ∧
    (shaded_area = 15 * small_area) ∧
    (shaded_area = 30) :=
by
  use [50, 2, 30]
  simp
  sorry

end shaded_area_is_30_l640_640901


namespace rect_eq_and_range_of_m_l640_640313

noncomputable def rect_eq_of_polar (m : ℝ) : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ * sin (θ + π / 3) + m = 0 → sqrt 3 * (ρ * cos θ) + (ρ * sin θ) + 2 * m = 0

noncomputable def range_of_m_for_intersection : set ℝ := 
  {m : ℝ | -19 / 12 ≤ m ∧ m ≤ 5 / 2}

theorem rect_eq_and_range_of_m (m : ℝ) : 
  rect_eq_of_polar m ∧ (∀ t : ℝ, ∃ x y : ℝ, x = sqrt 3 * cos (2 * t) ∧ y = 2 * sin t ∧ 
    (sqrt 3 * x + y + 2 * m = 0) → m ∈ range_of_m_for_intersection) :=
by sorry

end rect_eq_and_range_of_m_l640_640313


namespace rectangle_lengths_correct_l640_640010

-- Definitions of the parameters and their relationships
noncomputable def AB := 1200
noncomputable def BC := 150
noncomputable def AB_ext := AB
noncomputable def BC_ext := BC + 350
noncomputable def CD := AB
noncomputable def DA := BC

-- Definitions of the calculated distances using the conditions
noncomputable def AP := Real.sqrt (AB^2 + BC_ext^2)
noncomputable def PD := Real.sqrt (BC_ext^2 + AB^2)

-- Using similarity of triangles for PQ and CQ
noncomputable def PQ := (350 / 500) * AP
noncomputable def CQ := (350 / 500) * AB

-- The theorem to prove the final results
theorem rectangle_lengths_correct :
    AP = 1300 ∧
    PD = 1250 ∧
    PQ = 910 ∧
    CQ = 840 :=
    by
    sorry

end rectangle_lengths_correct_l640_640010


namespace area_HAKBO_eq_l640_640603

noncomputable def area_HAKBO (h : ℝ) (k : ℝ) (o : EuclideanGeometry.Point ℝ) 
  (a b ab d g ad bg k : EuclideanGeometry.Point ℝ) 
  (octagon : EuclideanGeometry.ConvexHull {a, b, c, d, e, f, g, h} ) : ℝ :=
  sorry

theorem area_HAKBO_eq (O : EuclideanGeometry.Point ℝ) (A B C D E F G H K : EuclideanGeometry.Point ℝ)
  (h1 : EuclideanGeometry.EquiangularOctagon {A, B, C, D, E, F, G, H})
  (h2 : EuclideanGeometry.inscribedInCircleCenteredAt {A, B, C, D, E, F, G, H} O)
  (h3 : EuclideanGeometry.intersect (AD) (BG) = K)
  (h4 : EuclideanGeometry.len (AB) = 2)
  (h5 : EuclideanGeometry.area {A, B, C, D, E, F, G, H} = 15) :
  area_HAKBO H A K B O = 11/4 :=
sorry

end area_HAKBO_eq_l640_640603


namespace find_remainder_l640_640189

noncomputable def remainder_expr_division (β : ℂ) (hβ : β^4 + β^3 + β^2 + β + 1 = 0) : ℂ :=
  1 - β

theorem find_remainder (β : ℂ) (hβ : β^4 + β^3 + β^2 + β + 1 = 0) :
  ∃ r, (x^45 + x^34 + x^23 + x^12 + 1) % (x^4 + x^3 + x^2 + x + 1) = r ∧ r = remainder_expr_division β hβ :=
sorry

end find_remainder_l640_640189


namespace incorrect_relations_count_l640_640569

noncomputable def count_incorrect_relations : Nat := 
  let condition1 := 0 ∈ ({0} : Set ℕ)
  let condition2 := (∅ : Set ℕ) ⊆ {0}
  let condition3 := ¬(0.3 ∈ ℚ)
  let condition4 := 0 ∈ (Nat.lift n)
  let condition5 := ({a, b} : Set ℕ) ⊆ {b, a}
  let condition6 := ({ x ∈ Int | x ^ 2 = 2 }) = ∅
  [condition1, condition2, condition3, condition4, condition5, condition6].count (λ rel => ¬rel)

theorem incorrect_relations_count :
  count_incorrect_relations = 1 := 
sorry

end incorrect_relations_count_l640_640569


namespace solve_for_x_l640_640454

theorem solve_for_x (x : ℝ) (h : 6 * x ^ (1 / 3) - 3 * (x / x ^ (2 / 3)) = -1 + 2 * x ^ (1 / 3) + 4) :
  x = 27 :=
by 
  sorry

end solve_for_x_l640_640454


namespace sum_of_integers_between_negative15_and_5_l640_640079

theorem sum_of_integers_between_negative15_and_5 : 
  ∑ i in Finset.Icc (-15) 5, i = -105 :=
by
  sorry

end sum_of_integers_between_negative15_and_5_l640_640079


namespace unique_solution_l640_640180

def is_solution (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ) (hx : 0 < x), 
    ∃! (y : ℝ) (hy : 0 < y), 
      x * f y + y * f x ≤ 2

theorem unique_solution : ∀ (f : ℝ → ℝ), 
  is_solution f ↔ (∀ x, 0 < x → f x = 1 / x) :=
by
  intros
  sorry

end unique_solution_l640_640180


namespace perpendicular_OE_CD_l640_640753

variable (O A B C D E : Point) (AB AC : ℝ)

-- Definitions based on the conditions
def is_circumcenter (O A B C : Point) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

def is_midpoint (D A B : Point) : Prop :=
  2 * (dist A D) = dist A B ∧ 2 * (dist B D) = dist A B

def is_centroid (E A C D : Point) : Prop :=
  dist E (midpoint A C) = dist E (midpoint C D)

def is_isosceles_triangle (A B C : Point) : Prop :=
  dist A B = dist A C

-- Theorem to prove
theorem perpendicular_OE_CD (hO : is_circumcenter O A B C)
  (hD : is_midpoint D A B) (hE : is_centroid E A C D)
  (hIso : is_isosceles_triangle A B C) : 
  dist (dot_product (vector_from_points O E) (vector_from_points C D)) = 0 := 
sorry

end perpendicular_OE_CD_l640_640753


namespace coefficient_x3_l640_640509

def poly1 := 2 * x^4 + 3 * x^3 - 4 * x^2 + 2
def poly2 := x^3 - 8 * x + 3

theorem coefficient_x3 (x : ℤ) : 
  (let prod := poly1 * poly2 
   in prod.coeff 3) = 41 :=
by
  unfold poly1 poly2
  sorry

end coefficient_x3_l640_640509


namespace probability_complete_collection_l640_640046

def combination (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem probability_complete_collection : 
  combination 6 6 * combination 12 4 / combination 18 10 = 5 / 442 :=
by
  have h1 : combination 6 6 = 1 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h2 : combination 12 4 = 495 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h3 : combination 18 10 = 43758 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  rw [h1, h2, h3]
  norm_num
  sorry

end probability_complete_collection_l640_640046


namespace intersection_of_M_and_N_l640_640202

noncomputable def M : Set ℝ := { y : ℝ | ∃ x : ℝ, y = x^2 }
noncomputable def N : Set ℝ := { y : ℝ | ∃ x : ℝ, x^2 + y^2 = 1 }

theorem intersection_of_M_and_N : M ∩ N = { y : ℝ | 0 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_of_M_and_N_l640_640202


namespace chips_calories_l640_640507

-- Define the conditions
def calories_from_breakfast : ℕ := 560
def calories_from_lunch : ℕ := 780
def calories_from_cake : ℕ := 110
def calories_from_coke : ℕ := 215
def daily_calorie_limit : ℕ := 2500
def remaining_calories : ℕ := 525

-- Define the total calories consumed so far
def total_consumed : ℕ := calories_from_breakfast + calories_from_lunch + calories_from_cake + calories_from_coke

-- Define the total allowable calories without exceeding the limit
def total_allowed : ℕ := daily_calorie_limit - remaining_calories

-- Define the calories in the chips
def calories_in_chips : ℕ := total_allowed - total_consumed

-- Prove that the number of calories in the chips is 310
theorem chips_calories :
  calories_in_chips = 310 :=
by
  sorry

end chips_calories_l640_640507


namespace rectangular_eq_of_line_l_range_of_m_l640_640351

noncomputable def curve_C_param_eq (t : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l_polar_eq (θ ρ m : ℝ) : Prop :=
  ρ * Real.sin(θ + Real.pi / 3) + m = 0

theorem rectangular_eq_of_line_l (x y m : ℝ) (h : line_l_polar_eq (Real.atan2 y x) (Real.sqrt (x^2 + y^2)) m) :
  sqrt 3 * x + y + 2 * m = 0 :=
  sorry

theorem range_of_m (m : ℝ) (t : ℝ) (h : ∃ (t : ℝ), curve_C_param_eq t ∈ set_of (λ p, p.2 = -√3 * p.1 - 2 * m))
  : -19/12 ≤ m ∧ m ≤ 5/2 :=
  sorry

end rectangular_eq_of_line_l_range_of_m_l640_640351


namespace soccer_ball_weight_l640_640616

theorem soccer_ball_weight :
  (∀ (x : ℝ), (5 * x = 2 * 15) → (x = 6)) :=
by
  intro x
  intro h
  have h1 : (5 : ℝ) * x = (2 : ℝ) * 15 := h
  rw [mul_comm 5 x, mul_comm 2 15] at h1
  rw [mul_assoc, mul_assoc] at h1
  simp only [mul_one] at h1
  exact eq_div_of_mul_eq h1

variables x : ℝ -- Denote the variable x represents the weight of one soccer ball
variable b : ℝ := 15 -- One bicycle weighs 15 pounds
axiom eqn : (5 * x = 2 * b) -- Five soccer balls weigh the same as two bicycles

example : x = 6 := 
by
  sorry

end soccer_ball_weight_l640_640616


namespace simplify_complex_fraction_l640_640219

theorem simplify_complex_fraction :
  (3 - (Complex.i)) / (2 + (Complex.i)) = 1 - (Complex.i) :=
by
  -- Using definitions and conditions
  have h_i_square := Complex.i_mul_I,
  -- Our goal is to simplify the complex fraction
  sorry

end simplify_complex_fraction_l640_640219


namespace sticker_probability_l640_640059

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end sticker_probability_l640_640059


namespace find_common_ratio_l640_640784

-- Define the geometric sequence with the given conditions
variable (a_n : ℕ → ℝ)
variable (q : ℝ)

axiom a2_eq : a_n 2 = 1
axiom a4_eq : a_n 4 = 4
axiom q_pos : q > 0

-- Define the nature of the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The specific problem statement to prove
theorem find_common_ratio (h: is_geometric_sequence a_n q) : q = 2 :=
by
  sorry

end find_common_ratio_l640_640784


namespace rect_eq_and_range_of_m_l640_640314

noncomputable def rect_eq_of_polar (m : ℝ) : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ * sin (θ + π / 3) + m = 0 → sqrt 3 * (ρ * cos θ) + (ρ * sin θ) + 2 * m = 0

noncomputable def range_of_m_for_intersection : set ℝ := 
  {m : ℝ | -19 / 12 ≤ m ∧ m ≤ 5 / 2}

theorem rect_eq_and_range_of_m (m : ℝ) : 
  rect_eq_of_polar m ∧ (∀ t : ℝ, ∃ x y : ℝ, x = sqrt 3 * cos (2 * t) ∧ y = 2 * sin t ∧ 
    (sqrt 3 * x + y + 2 * m = 0) → m ∈ range_of_m_for_intersection) :=
by sorry

end rect_eq_and_range_of_m_l640_640314


namespace probability_of_intersection_is_7_over_12_l640_640874

noncomputable def probability_line_intersects_circle : ℚ :=
  let outcomes : Finset (ℕ × ℕ) := { (a, b) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} }.to_finset
  let favorable_outcomes : Finset (ℕ × ℕ) := { (a, b) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ a ≤ b }.to_finset
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_of_intersection_is_7_over_12 : probability_line_intersects_circle = 7 / 12 :=
by
  sorry

end probability_of_intersection_is_7_over_12_l640_640874


namespace initial_volume_of_mixture_l640_640879

variable (V : ℝ)
variable (H1 : 0.2 * V + 12 = 0.25 * (V + 12))

theorem initial_volume_of_mixture (H : 0.2 * V + 12 = 0.25 * (V + 12)) : V = 180 := by
  sorry

end initial_volume_of_mixture_l640_640879


namespace trapezoid_BC_length_l640_640651

/-- Given a trapezoid ABCD with parallel sides BC and AD, point H on AB such
that ∠DHA = 90°, CH = CD = 13, and AD = 19, prove that the length of BC = 9.5. -/
theorem trapezoid_BC_length 
  (A B C D H : Type)
  [is_trapezoid : ∀ (A B C D : Type), parallel BC AD] 
  (is_right_angle : ∠ (D H A) = π / 2)
  (length_CH : CH = 13)
  (length_CD : CD = 13)
  (length_AD : AD = 19) :
  BC = 9.5 :=
sorry

end trapezoid_BC_length_l640_640651


namespace train_length_140_meters_l640_640893

theorem train_length_140_meters
    (time_to_cross : ℕ)
    (speed_kmph : ℕ) :
  time_to_cross = 6 →
  speed_kmph = 84 →
  let speed_mps := speed_kmph * (5/18 : ℚ) in
  let length_of_train := speed_mps * time_to_cross in
  length_of_train = 140 := 
by
  intros h_time h_speed
  sorry

end train_length_140_meters_l640_640893


namespace rect_eq_line_range_of_m_l640_640325

-- Definitions based on the conditions
def parametric_x (t : ℝ) : ℝ := sqrt 3 * cos (2 * t)
def parametric_y (t : ℝ) : ℝ := 2 * sin t

def polar_rho (θ m : ℝ) : ℝ := -(m / (sin (θ + π / 3)))

-- Rectangular equation of the line l
theorem rect_eq_line (x y m : ℝ) : polar_rho (atan2 y x) m * sin (atan2 y x + π / 3) + m = 0 → sqrt 3 * x + y + 2 * m = 0 :=
by
  sorry

-- Range of values for m
theorem range_of_m (C : ℝ → ℝ × ℝ)
  (hC : ∀ t, C t = (parametric_x t, parametric_y t)) (m : ℝ) :
  (∃ t, sqrt 3 * fst (C t) + snd (C t) + 2 * m = 0) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
by
  sorry

end rect_eq_line_range_of_m_l640_640325


namespace length_of_bridge_l640_640485

theorem length_of_bridge
  (length_of_train : ℕ)
  (speed_km_hr : ℝ)
  (time_sec : ℝ)
  (h_train_length : length_of_train = 155)
  (h_train_speed : speed_km_hr = 45)
  (h_time : time_sec = 30) :
  ∃ (length_of_bridge : ℝ),
    length_of_bridge = 220 :=
by
  sorry

end length_of_bridge_l640_640485


namespace shaded_region_area_eq_108_l640_640789

/-- There are two concentric circles, where the outer circle has twice the radius of the inner circle,
and the total boundary length of the shaded region is 36π. Prove that the area of the shaded region
is nπ, where n = 108. -/
theorem shaded_region_area_eq_108 (r : ℝ) (h_outer : ∀ (c₁ c₂ : ℝ), c₁ = 2 * c₂) 
  (h_boundary : 2 * Real.pi * r + 2 * Real.pi * (2 * r) = 36 * Real.pi) : 
  ∃ (n : ℕ), n = 108 ∧ (Real.pi * (2 * r)^2 - Real.pi * r^2) = n * Real.pi := 
sorry

end shaded_region_area_eq_108_l640_640789


namespace cole_drive_time_l640_640533

theorem cole_drive_time (D : ℝ) (T_work T_home : ℝ) 
  (h1 : T_work = D / 75) 
  (h2 : T_home = D / 105)
  (h3 : T_work + T_home = 4) : 
  T_work * 60 = 140 := 
by sorry

end cole_drive_time_l640_640533


namespace serving_size_is_six_l640_640561

-- Define the conditions
def concentrate_to_water (c : ℕ) (w : ℕ) : ℕ := c + w
def total_cans (concentrate : ℕ) (cans_per_concentrate : ℕ) : ℕ := concentrate * cans_per_concentrate
def total_volume (total_cans : ℕ) (ounces_per_can : ℕ) : ℕ := total_cans * ounces_per_can
def serving_size (total_volume : ℕ) (servings : ℕ) : ℕ := total_volume / servings

-- Define the problem statement
theorem serving_size_is_six :
  let c := 1,
      w := 4,
      concentrate := 12,
      servings := 120,
      ounces_per_can := 12,
      cans_per_concentrate := concentrate_to_water c w,
      total_cans := total_cans concentrate cans_per_concentrate,
      total_volume := total_volume total_cans ounces_per_can
  in serving_size total_volume servings = 6 :=
by
  sorry

end serving_size_is_six_l640_640561


namespace part1_part2_l640_640998

open Real

-- Definitions based on conditions
def m (x : ℝ) := (sqrt 3 * sin (x / 4), 1)
def n (x : ℝ) := (cos (x / 4), cos (x / 4) ^ 2)
def f (x : ℝ) := (m x).1 * (n x).1 + (m x).2 * (n x).2
def a (x y z : ℝ) := x
def b (x y z : ℝ) := y
def c (x y z : ℝ) := z

-- Part 1
theorem part1 (x : ℝ) (h : f x = 1) : cos (2 * π / 3 - x) = 1 / 2 := 
  sorry

-- Part 2
theorem part2 (a b c : ℝ) (h : a * cos (C a b c) + 1 / 2 * c = b) : ∀ B : ℝ, 1 < f B ∧ f B < 3 / 2 := 
  sorry

end part1_part2_l640_640998


namespace complete_collection_probability_l640_640057

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end complete_collection_probability_l640_640057


namespace smallest_whole_number_larger_than_perimeter_l640_640839

theorem smallest_whole_number_larger_than_perimeter :
  let a := 7
  let b := 24
  let perimeter (c : ℝ) := a + b + c
  ∀ (c : ℝ), 17 < c ∧ c < 31 →
  62 =
  ceil (perimeter c)
:=
by
  intro a b h p c hc
  sorry


end smallest_whole_number_larger_than_perimeter_l640_640839


namespace operation_star_test_l640_640173

theorem operation_star_test:
  (∀ (a b : ℕ), a * b = a \star b) →
  (∀ (c d : ℕ), c - d = c * d) →
  (16 * 4 / (8 - 2) = 4) →
  (9 * 3 / (18 - 6) = 9 / 4) := 
sorry

end operation_star_test_l640_640173


namespace number_of_sets_satisfying_union_l640_640241

open Set

theorem number_of_sets_satisfying_union (M : Set ℕ) (N : Set ℕ) 
  (hM : M = {0, 1}) : 
  {N : Set ℕ | M ∪ N = {0, 1, 2}}.card = 4 := by
  sorry

end number_of_sets_satisfying_union_l640_640241


namespace find_b2_l640_640488

theorem find_b2 (b : ℕ → ℝ) (h1 : b 1 = 25) (h2 : b 10 = 125)
  (h3 : ∀ n, n ≥ 4 → b n = (1 / (n - 1)) * (finset.range (n-1)).sum (λ i, b (i + 1))) :
  b 2 = 162.5 :=
sorry

end find_b2_l640_640488


namespace sum_contains_even_digit_l640_640775

-- Define the five-digit integer and its reversed form
def reversed_digits (n : ℕ) : ℕ := 
  let a := n % 10
  let b := (n / 10) % 10
  let c := (n / 100) % 10
  let d := (n / 1000) % 10
  let e := (n / 10000) % 10
  a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem sum_contains_even_digit (n m : ℕ) (h1 : n >= 10000) (h2 : n < 100000) (h3 : m = reversed_digits n) : 
  ∃ d : ℕ, d < 10 ∧ d % 2 = 0 ∧ (n + m) % 10 = d ∨ (n + m) / 10 % 10 = d ∨ (n + m) / 100 % 10 = d ∨ (n + m) / 1000 % 10 = d ∨ (n + m) / 10000 % 10 = d := 
sorry

end sum_contains_even_digit_l640_640775


namespace animal_arrangements_l640_640782

-- Given conditions
def chickens := 3
def dogs := 3
def cats := 5
def rabbits := 2
def cages := 13

-- Proving the total number of ways
theorem animal_arrangements :
  (fact 4) * (fact chickens) * (fact dogs) * (fact cats) * (fact rabbits) = 207360 :=
by {
  -- The proof itself is not required to be written as per the instruction, so we use sorry.
  sorry
}

end animal_arrangements_l640_640782


namespace problem_1_problem_2_l640_640154

theorem problem_1 :
  ((2 ^ (1/3)) * (3 ^ (1/2))) ^ 6 + ((2 * (2 ^ (1/2))) ^ (1/2)) ^ (4/3) - 
    4 * (16/49) ^ (-1/2) - ((2 ^ (1/4)) * 8 ^ (1/4)) - ((-2009) ^ 0) = 100 := 
  sorry

theorem problem_2 :
  2 * (Real.log10 (2 ^ (1/2))) ^ 2 + (Real.log10 (2 ^ (1/2))) + 
    (Real.log10 5) + Real.sqrt ((Real.log10 (2 ^ (1/2))) ^ 2 - (Real.log10 2) + 1) = 1 := 
  sorry

end problem_1_problem_2_l640_640154


namespace determine_moles_l640_640924

variables (HCl AgNO3 NH4NO3 NaCl HNO3 AgCl NH4Cl NaNO3 : ℝ)

-- Define the initial conditions
variables (initial_HCl : ℝ := 4)
variables (initial_AgNO3 : ℝ := 3)
variables (initial_NH4NO3 : ℝ := 2)
variables (initial_NaCl : ℝ := 4)

-- Define the reactions
def reaction1 (x₁ x₂ : ℝ) := x₁ + x₂ = min initial_AgNO3 initial_HCl
def reaction2 (x₃ x₄ : ℝ) := x₃ + x₄ = min initial_NH4NO3 initial_NaCl

-- Number of moles of products formed
def moles_HNO3 := min initial_AgNO3 initial_HCl
def remaining_HCl := initial_HCl - min initial_AgNO3 initial_HCl
def remaining_NaCl := initial_NaCl - min initial_NH4NO3 initial_NaCl

-- Theorem to prove
theorem determine_moles : 
  moles_HNO3 = 3 ∧
  remaining_HCl = 1 ∧
  remaining_NaCl = 2 :=
begin
  unfold moles_HNO3 remaining_HCl remaining_NaCl,
  split,
  {
    -- Proof for moles_HNO3
    sorry,
  },
  split,
  {
    -- Proof for remaining_HCl
    sorry,
  },
  {
    -- Proof for remaining_NaCl
    sorry,
  }
end

end determine_moles_l640_640924


namespace ratio_gluten_free_l640_640553

theorem ratio_gluten_free (total_cupcakes vegan_cupcakes non_vegan_gluten cupcakes_gluten_free : ℕ)
    (H1 : total_cupcakes = 80)
    (H2 : vegan_cupcakes = 24)
    (H3 : non_vegan_gluten = 28)
    (H4 : cupcakes_gluten_free = vegan_cupcakes / 2) :
    (cupcakes_gluten_free : ℚ) / (total_cupcakes : ℚ) = 3 / 20 :=
by 
  -- Proof goes here
  sorry

end ratio_gluten_free_l640_640553


namespace minimize_second_order_moment_l640_640443

noncomputable def pdf (f : ℝ → ℝ) := ∀ x, 0 ≤ f x ∧ ∫ t in Set.Ioc (-∞ : ℝ) (∞ : ℝ), f t = 1

noncomputable def mean (f : ℝ → ℝ) := ∫ t in Set.Ioc (-∞ : ℝ) (∞ : ℝ), t * f t

noncomputable def second_order_moment (f : ℝ → ℝ) (c : ℝ) := ∫ t in Set.Ioc (-∞ : ℝ) (∞ : ℝ), (t - c) ^ 2 * f t

theorem minimize_second_order_moment {f : ℝ → ℝ} (h_pdf : pdf f) :
  second_order_moment f (mean f) ≤ second_order_moment f c := 
sorry

end minimize_second_order_moment_l640_640443


namespace percentage_proof_l640_640083

/-- Lean 4 statement proving the percentage -/
theorem percentage_proof :
  ∃ P : ℝ, (800 - (P / 100) * 8000) = 796 ∧ P = 0.05 :=
by
  use 0.05
  sorry

end percentage_proof_l640_640083


namespace radius_of_inscribed_circle_l640_640498

theorem radius_of_inscribed_circle
  (r r_y t : ℝ)
  (ω_x ω_y ω_z : ℝ → ℝ → Prop)
  (p q : ℝ → ℝ → Prop)
  (tangent_points : ω_x → ω_y → ω_z → ℝ × ℝ × ℝ)
  (midpoint_Y : ∃ Y : ℝ, ∀ X Z : ℝ, (X + Z) / 2 = Y)
  (tangent_to_line : ∀ (c : ℝ), ω_x c → ω_y c → ω_z c → (ℝ → Prop))
  (radii_conditions : r_x = r ∧ r_z = r ∧ r_y > r)
  (isosceles_triangle : ∃ (A B C : ℝ), A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ p A B ∧ q B C ∧ t B) :
  ∃ (r_insc : ℝ), r_insc = r := sorry

end radius_of_inscribed_circle_l640_640498


namespace total_four_digit_numbers_multiple_of_9_l640_640570

-- First, we state the conditions as definitions

def digits := {1, 2, 3, 4, 5, 6}
def sum_digits := 1 + 2 + 3 + 4 + 5 + 6
def remaining_digits := {1, 2, 4, 5, 6}
def sum_remaining_digits := 1 + 2 + 4 + 5 + 6

-- Then, we state the problem and the theorem

theorem total_four_digit_numbers_multiple_of_9 :
  (∃ numbers : Finset ℕ, numbers.card = 4 ∧ numbers ⊆ digits ∧ (numbers.sum id) % 9 = 0) →
  (numbers : Finset ℕ) → (numbers.card = 4 ∧ numbers ⊆ digits ∧ (numbers.sum id) % 9 = 0) →
  numbers.count = 24 := sorry

end total_four_digit_numbers_multiple_of_9_l640_640570


namespace rectangular_equation_of_l_range_of_m_intersection_l640_640368

-- Definitions based on the problem conditions
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  let x := sqrt 3 * cos (2 * t)
  let y := 2 * sin t
  (x, y)

def polar_line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- The rectangular equation converted from polar form
theorem rectangular_equation_of_l (x y m : ℝ) :
  (polar_line_l (sqrt (x^2 + y^2)) (atan2 y x) m) ↔ (√3 * x + y + 2 * m = 0) :=
sorry

-- Range of values for m ensuring intersection of line l with curve C
theorem range_of_m_intersection (m : ℝ) :
  ((-19/12 : ℝ) ≤ m ∧ m ≤ (5/2 : ℝ)) ↔
  ∃ t : ℝ, let x := sqrt 3 * cos (2 * t)
           let y := 2 * sin t
           (sqrt 3 * x + y + 2 * m = 0 ∧ (-2 ≤ y ∧ y ≤ 2)) :=
sorry

end rectangular_equation_of_l_range_of_m_intersection_l640_640368


namespace price_of_second_container_l640_640115

-- Conditions given in the problem
def radius_1 : ℝ := 2
def height_1 : ℝ := 5
def price_1 : ℝ := 1.50

def radius_2 : ℝ := 2
def height_2 : ℝ := 10

-- Given that the price is proportional to the volume, we need to prove that the price of the second container is \$3.00
theorem price_of_second_container :
  let volume_1 := π * radius_1^2 * height_1 in
  let volume_2 := π * radius_2^2 * height_2 in
  let volume_ratio := volume_2 / volume_1 in
  let price_2 := price_1 * volume_ratio in
  price_2 = 3 :=
by
  sorry

end price_of_second_container_l640_640115


namespace four_g_users_scientific_notation_l640_640560

-- Condition for scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

-- Given problem in scientific notation form
theorem four_g_users_scientific_notation :
  ∃ a n, is_scientific_notation a n 1030000000 ∧ a = 1.03 ∧ n = 9 :=
sorry

end four_g_users_scientific_notation_l640_640560


namespace percentage_of_germinated_seeds_l640_640944

def plot1_seeds := 300
def plot1_rate := 0.25
def plot2_seeds := 200
def plot2_rate := 0.35
def plot3_seeds := 400
def plot3_rate := 0.45
def plot4_seeds := 350
def plot4_rate := 0.15
def plot5_seeds := 150
def plot5_rate := 0.50

def total_germinated_seeds : ℕ :=
  (plot1_seeds * plot1_rate + plot2_seeds * plot2_rate + plot3_seeds * plot3_rate +
   plot4_seeds * plot4_rate + plot5_seeds * plot5_rate).toNat

def total_planted_seeds := plot1_seeds + plot2_seeds + plot3_seeds + plot4_seeds + plot5_seeds

def percentage_germinated := (total_germinated_seeds / total_planted_seeds.toNat * 100).toRat

theorem percentage_of_germinated_seeds :
  percentage_germinated = 32.3 := by
  sorry

end percentage_of_germinated_seeds_l640_640944


namespace subtract_base8_l640_640909

theorem subtract_base8 (a b : ℕ) (h₁ : a = 0o2101) (h₂ : b = 0o1245) :
  a - b = 0o634 := sorry

end subtract_base8_l640_640909


namespace value_of_x_l640_640707

theorem value_of_x (x : ℝ) (h : 0.75 * 600 = 0.50 * x) : x = 900 :=
by
  sorry

end value_of_x_l640_640707


namespace intersect_rectangular_eqn_range_of_m_l640_640357

-- Definitions of the parametric and polar equations
def curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Proving the rectangular equation and range of m for the intersection of l and C
theorem intersect_rectangular_eqn (m t : ℝ) :
  (∃ ρ θ, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  ∃ x y, (x, y) = curve_C t ∧ √3 * x + y + 2 * m = 0 :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ ρ θ t, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end intersect_rectangular_eqn_range_of_m_l640_640357


namespace probability_same_color_dice_l640_640699

theorem probability_same_color_dice :
  let total_sides := 12
  let red_sides := 3
  let green_sides := 4
  let blue_sides := 2
  let yellow_sides := 3
  let prob_red := (red_sides / total_sides) ^ 2
  let prob_green := (green_sides / total_sides) ^ 2
  let prob_blue := (blue_sides / total_sides) ^ 2
  let prob_yellow := (yellow_sides / total_sides) ^ 2
  prob_red + prob_green + prob_blue + prob_yellow = 19 / 72 := 
by
  -- The proof goes here
  sorry

end probability_same_color_dice_l640_640699


namespace complement_intersection_is_empty_l640_640243

open Set

variable (U A B : Set ℕ)
variable (x: U)

def U := {1, 2, 3, 4, 5, 6, 7, 8}
def A := {1, 2, 3, 4}
def B := {3, 4, 5, 6, 7, 8}

theorem complement_intersection_is_empty :
  (U \ A) ∩ (U \ B) = ∅ := by
  sorry

end complement_intersection_is_empty_l640_640243


namespace problem_part1_problem_part2_l640_640299

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l_rect_eq (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (sqrt 3 * x + y + 2 * m = 0)

def line_l_range_m (m : ℝ) : Prop :=
  -19/12 ≤ m ∧ m ≤ 5/2

theorem problem_part1 (m : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * sin (θ + π / 3) + m = 0) ↔ line_l_rect_eq m := 
sorry

theorem problem_part2 :
  ∀ m : ℝ, 
  (∃ t : ℝ, line_l_rect_eq m (curve_C t).1 (curve_C t).2) ↔ line_l_range_m m := 
sorry

end problem_part1_problem_part2_l640_640299


namespace correct_proposition_l640_640141

-- Definitions corresponding to the conditions
def propositionA (P L : Prop) : Prop := 
  ∃! l : Prop, (P ∧ l) ∧ (L ∧ l) ∧ (l ≠ L)

def propositionB : Prop := 
  ∀ α β : Prop, (α = β) → (α = vertical_angle β)

def propositionC (L1 L2 T : Prop) : Prop := 
  intersect (L1 ∧ L2 ∧ T) → supplementary (interior_angles L1 L2 T)

def propositionD (L1 L2 T : Prop) : Prop := 
  (same_plane L1 L2) ∧ perpendicular L1 T ∧ perpendicular L2 T → parallel L1 L2

-- The theorem to prove Proposition D is the correct one given all propositions
theorem correct_proposition (P L1 L2 T : Prop) :
  (¬ propositionA P L1) ∧ 
  (¬ propositionB) ∧ 
  (¬ propositionC L1 L2 T) ∧ 
  propositionD L1 L2 T :=
sorry

end correct_proposition_l640_640141


namespace quadrilateral_concurrency_l640_640746

open EuclideanGeometry

-- Let ABCD be a convex quadrilateral with pairwise distinct side lengths
noncomputable def is_pairwise_distinct (a b c d : ℝ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem quadrilateral_concurrency (A B C D O1 O2: Point) (ABC_circumcenter ABD_circumcenter : Set Point):
  convex A B C D →
  is_pairwise_distinct (dist A B) (dist B C) (dist C D) (dist D A) →
  ⟪A - C, B - D⟫ = 0 →
  circumcenter ⟨A, B, D⟩ = O1 →
  circumcenter ⟨C, B, D⟩ = O2 →
  ∃ P: Point, Euler_line ⟨A, B, C⟩ P ∧ Euler_line ⟨A, D, C⟩ P ∧ collinear A O2 P ∧ collinear C O1 P :=
sorry

end quadrilateral_concurrency_l640_640746


namespace polar_to_rectangular_intersection_range_l640_640338

-- Definitions of parametric equations for curve C
def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line_l (ρ θ m: ℝ) : Prop := ρ * sin (θ + π / 3) + m = 0

-- Rectangular equation of line l
def rectangular_line_l (x y m: ℝ) : Prop := sqrt 3 * x + y + 2 * m = 0

-- Prove that polar equation of line l can be transformed to rectangular form
theorem polar_to_rectangular (ρ θ m : ℝ) :
  (polar_line_l ρ θ m) → (∃ x y, ρ = sqrt (x^2 + y^2) ∧ θ = arctan (y / x) ∧ rectangular_line_l x y m) :=
  sorry

-- Prove that the range of values for m ensuring line l intersects curve C is [-19/12, 5/2]
theorem intersection_range (m : ℝ) :
  (∃ t : ℝ, rectangular_line_l (sqrt 3 * cos (2 * t)) (2 * sin t) m) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
  sorry

end polar_to_rectangular_intersection_range_l640_640338


namespace profit_percentage_l640_640536

theorem profit_percentage (SP : ℝ) (h1 : SP > 0) (h2 : CP = 0.99 * SP) : (SP - CP) / CP * 100 = 1.01 :=
by
  sorry

end profit_percentage_l640_640536


namespace min_value_proof_l640_640689

def min_value_exp (a b c : ℝ) : ℝ := (3 * a + 2 * b + c) / (2 * b - 3 * a)

theorem min_value_proof (a b c : ℝ) (h1 : a < (2 / 3) * b) (h2 : a > 0) (h3 : 4 * b^2 - 12 * a * c ≤ 0) : min_value_exp a b c = 4 :=
by
  sorry

end min_value_proof_l640_640689


namespace distinct_solutions_abs_equation_l640_640611

theorem distinct_solutions_abs_equation : ∃! (x : ℝ), |x - |3*x - 2|| = 4 :=
by {
  sorry  -- Proof to be completed
}

end distinct_solutions_abs_equation_l640_640611


namespace find_max_r_squared_l640_640503

noncomputable def cone (base_radius height : ℝ) : Prop :=
base_radius = 3 ∧ height = 8

noncomputable def sphere_in_cones (r : ℝ) (interior_point_distance : ℝ) : Prop :=
interior_point_distance = 3 ∧ (sphere_radius := r, sphere_radius^2 = 225 / 73)

theorem find_max_r_squared : ∃ m n : ℕ, m + n = 298 ∧ (∀ r : ℝ, 
  cone 3 8 ∧ sphere_in_cones r 3 → r^2 = 225 / 73) :=
begin
  use [225, 73],
  split,
  { exact (225 + 73), },
  { intros r hr,
    sorry
  }
end

end find_max_r_squared_l640_640503


namespace polar_to_rectangular_intersection_range_l640_640334

-- Definitions of parametric equations for curve C
def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line_l (ρ θ m: ℝ) : Prop := ρ * sin (θ + π / 3) + m = 0

-- Rectangular equation of line l
def rectangular_line_l (x y m: ℝ) : Prop := sqrt 3 * x + y + 2 * m = 0

-- Prove that polar equation of line l can be transformed to rectangular form
theorem polar_to_rectangular (ρ θ m : ℝ) :
  (polar_line_l ρ θ m) → (∃ x y, ρ = sqrt (x^2 + y^2) ∧ θ = arctan (y / x) ∧ rectangular_line_l x y m) :=
  sorry

-- Prove that the range of values for m ensuring line l intersects curve C is [-19/12, 5/2]
theorem intersection_range (m : ℝ) :
  (∃ t : ℝ, rectangular_line_l (sqrt 3 * cos (2 * t)) (2 * sin t) m) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
  sorry

end polar_to_rectangular_intersection_range_l640_640334


namespace perimeter_ABCDE_l640_640280

-- Define the vertices of the polygon
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (4, 8)
def C : ℝ × ℝ := (4, 4)
def D : ℝ × ℝ := (8, 0)
def E : ℝ × ℝ := (0, 0)

-- Define a function to compute the distance between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Perimeter calculation proof statement
theorem perimeter_ABCDE : distance A B + distance B C + distance C D + distance D E + distance E A = 12 + 4 * real.sqrt 5 :=
by
  sorry

end perimeter_ABCDE_l640_640280


namespace math_problem_l640_640687

noncomputable def f (x a : ℝ) : ℝ := -4 * (Real.cos x) ^ 2 + 4 * Real.sqrt 3 * a * (Real.sin x) * (Real.cos x) + 2

theorem math_problem (a : ℝ) :
  (∃ a, ∀ x, f x a = f (π/6 - x) a) →    -- Symmetry condition
  (a = 1 ∧
  ∀ k : ℤ, ∀ x, (x ∈ Set.Icc (π/3 + k * π) (5 * π / 6 + k * π) → 
    x ∈ Set.Icc (π/3 + k * π) (5 * π / 6 + k * π)) ∧  -- Decreasing intervals
  (∀ x, 2 * x - π / 6 ∈ Set.Icc (-2 * π / 3) (π / 6) → 
    f x a ∈ Set.Icc (-4 : ℝ) 2)) := -- Range on given interval
sorry

end math_problem_l640_640687


namespace range_of_k_l640_640236

noncomputable def f (x : ℝ) := x^2 + 4 / x^2 - 3
noncomputable def g (k x : ℝ) := k * x + 2

theorem range_of_k :
  (∀ x₁ ∈ set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ set.Icc (1 : ℝ) (Real.sqrt 3), g k x₁ > f x₂) ↔ k ∈ set.Ioo (-1 / 2 : ℝ) 1 :=
by sorry

end range_of_k_l640_640236


namespace number_of_unique_outfits_l640_640460

-- Define the given conditions
def num_shirts : ℕ := 8
def num_ties : ℕ := 6
def special_shirt_ties : ℕ := 3
def remaining_shirts := num_shirts - 1
def remaining_ties := num_ties

-- Define the proof problem
theorem number_of_unique_outfits : num_shirts * num_ties - remaining_shirts * remaining_ties + special_shirt_ties = 45 :=
by
  sorry

end number_of_unique_outfits_l640_640460


namespace plane_divided_into_8_regions_l640_640590

noncomputable def number_of_regions (lines : List (ℝ → ℝ)) : ℕ :=
  -- Placeholder function to return the number of regions split by given lines
  -- The actual implementation of this computation is non-trivial and not required here
  8

theorem plane_divided_into_8_regions :
  number_of_regions [
    (λ x, 2 * x),
    (λ x, (1 / 2) * x),
    (λ x, 3 * x),
    (λ x, (1 / 3) * x)
  ] = 8 := 
  by
  sorry

end plane_divided_into_8_regions_l640_640590


namespace problem_l640_640766

def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 1 then x^2 - x else if 0 ≤ -x ∧ -x ≤ 1 then -(x^2 + x) else 
  if x % 2 = 0 then f(x - 2) else if x % 2 = 1 then f(x - 1) else x

axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom periodic_function : ∀ x : ℝ, f(x + 2) = f(x)
axiom interval_function : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = x^2 - x

theorem problem (f : ℝ → ℝ) (odd_function : ∀ x : ℝ, f(-x) = -f(x))
  (periodic_function : ∀ x : ℝ, f(x + 2) = f(x))
  (interval_function : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = x^2 - x) :
  f(-5 / 2) = 1 / 4 := by
  sorry

end problem_l640_640766


namespace increasing_exponential_iff_l640_640013

-- Define the conditions
def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Define the function
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, (a - 1) ^ x

-- The Lean proof statement
theorem increasing_exponential_iff (a : ℝ) : 
  is_increasing_function (f a) ↔ a > 2 :=
sorry

end increasing_exponential_iff_l640_640013


namespace depth_of_channel_l640_640096

theorem depth_of_channel (top_width bottom_width : ℝ) (area : ℝ) (h : ℝ) 
  (h_top : top_width = 14) (h_bottom : bottom_width = 8) (h_area : area = 770) :
  (1 / 2) * (top_width + bottom_width) * h = area → h = 70 :=
by
  intros h_trapezoid
  sorry

end depth_of_channel_l640_640096


namespace cylinder_volume_multiplication_factor_l640_640844

theorem cylinder_volume_multiplication_factor (r h : ℝ) (h_r_positive : r > 0) (h_h_positive : h > 0) :
  let V := π * r^2 * h
  let V' := π * (2.5 * r)^2 * (3 * h)
  let X := V' / V
  X = 18.75 :=
by
  -- Proceed with the proof here
  sorry

end cylinder_volume_multiplication_factor_l640_640844


namespace hook_all_circles_with_7_hooks_l640_640811

open Real

-- Define the condition "each pair of circles has a common point"
def intersects (circle1 circle2 : ℝ × ℝ × ℝ) : Prop := 
  let ⟨x1, y1, r1⟩ := circle1
  let ⟨x2, y2, r2⟩ := circle2
  (x1 - x2)^2 + (y1 - y2)^2 ≤ (r1 + r2)^2

-- Define the problem's main condition: "There are n circular paper pieces"
def circles_have_common_point (circles : Fin n.succ → ℝ × ℝ × ℝ) : Prop :=
  ∀ i j, i ≠ j → intersects (circles i) (circles j)

-- State the theorem
theorem hook_all_circles_with_7_hooks (n : ℕ) (circles : Fin n.succ → ℝ × ℝ × ℝ) 
    (h : circles_have_common_point circles) : 
    ∃ hooks : Fin 7 → ℝ × ℝ, 
      ∀ (i : Fin n.succ), ∃ (j : Fin 7), (squared_dist (circles i).fst (hooks j)) ≤ (circles i).snd^2 :=
  sorry

end hook_all_circles_with_7_hooks_l640_640811


namespace distance_PF_l640_640965

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the point P on the parabola with x-coordinate 4
def point_on_parabola (y : ℝ) : ℝ × ℝ := (4, y)

-- Prove the distance |PF| for given conditions
theorem distance_PF
  (hP : ∃ y : ℝ, parabola 4 y)
  (hF : focus = (2, 0)) :
  ∃ y : ℝ, y^2 = 8 * 4 ∧ abs (4 - 2) + abs y = 6 := 
by
  sorry

end distance_PF_l640_640965


namespace graph_symmetric_point_l640_640639

open Real

def f (x : ℝ) : ℝ := sin (0.5 * x + (π / 3))

theorem graph_symmetric_point : 
  (∀ x, f(- x - 2 * π / 3) = - f(x) + f(-2 * π / 3)) :=
sorry

end graph_symmetric_point_l640_640639


namespace median_of_first_twelve_positive_integers_l640_640513

theorem median_of_first_twelve_positive_integers : 
  let first_twelve_positive_integers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (first_twelve_positive_integers.length = 12) →
  (first_twelve_positive_integers.nth (first_twelve_positive_integers.length / 2 - 1) = some 6) →
  (first_twelve_positive_integers.nth (first_twelve_positive_integers.length / 2) = some 7) →
  (6 + 7) / 2 = 6.5 :=
by
  sorry

end median_of_first_twelve_positive_integers_l640_640513


namespace new_person_weight_l640_640538

theorem new_person_weight {n : ℕ} (old_weight avg_increase : ℝ) (h_n : n = 10) (h_old_weight : old_weight = 50) (h_avg_increase : avg_increase = 2.5) :
  let total_increase := n * avg_increase in
  let new_weight := old_weight + total_increase in
  new_weight = 75 := 
begin
  sorry
end

end new_person_weight_l640_640538


namespace distance_travelled_72_minutes_l640_640877

def minutes_to_hours (minutes : ℕ) : ℝ :=
  minutes / 60.0

def distance_walked (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem distance_travelled_72_minutes :
  let speed := 10.0
  let time_in_minutes := 72
  let time_in_hours := minutes_to_hours time_in_minutes
  distance_walked speed time_in_hours = 12 := by
  sorry

end distance_travelled_72_minutes_l640_640877


namespace prove_logarithm_identity_l640_640204

-- Define the conditions
variables {a : ℝ} (ha : a > 0)
variables {x : ℝ} (hxeq : 10^x = (10 * a).log10 + (a⁻¹).log10)

-- State the theorem to be proven
theorem prove_logarithm_identity : x = 0 :=
by
  -- Skipping the proof with sorry
  sorry

end prove_logarithm_identity_l640_640204


namespace bruces_son_age_l640_640151

variable (Bruce_age : ℕ) (son_age : ℕ)
variable (h1 : Bruce_age = 36)
variable (h2 : Bruce_age + 6 = 3 * (son_age + 6))

theorem bruces_son_age :
  son_age = 8 :=
by {
  sorry
}

end bruces_son_age_l640_640151


namespace find_y_l640_640372

-- Assuming angles are measured in degrees
variables (A B C D : Type) [dec_string_map_β.has_zero A]
  (angle_ABC : B = C)
  (angle_ABD : Angle B A D = 150)
  (angle_CBD : Angle C B D = 70)
  (y : Angle B C D)

-- Define math proof problem
theorem find_y : y = 80 :=
by sorry

end find_y_l640_640372


namespace Simson_line_B_perpendicular_Euler_line_ACD_l640_640747

variables {A B C D : Type} [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space D]

-- Define cyclic quadrilateral
def is_cyclic_quadrilateral (A B C D : Type) [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space D] :=
  ∃ O : euclidean_space ℝ, ∀ P ∈ {A, B, C, D}, euclidean_distance O P = euclidean_distance O A 

-- Define Simson line relation
def Simson_line_perpendicular_to_Euler_line (P : Type) (Δ : triangle (euclidean_space ℝ)) [euclidean_space P] :=
  ∃ L₁ L₂ : line (euclidean_space ℝ), Simson_line P Δ L₁ ∧ Euler_line Δ L₂ ∧ L₁ ⊥ L₂

variables {Δ₁ Δ₂ : triangle (euclidean_space ℝ)}

-- Conditions
variables 
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_triangle_not_eq1 : ¬is_equilateral_triangle B C D)
  (h_triangle_not_eq2 : ¬is_equilateral_triangle C D A)
  (h_perpendicular : Simson_line_perpendicular_to_Euler_line A Δ₁)
 
-- Conclusion
theorem Simson_line_B_perpendicular_Euler_line_ACD :
  Simson_line_perpendicular_to_Euler_line B Δ₂ :=
sorry

end Simson_line_B_perpendicular_Euler_line_ACD_l640_640747


namespace trapezoid_BC_length_l640_640650

/-- Given a trapezoid ABCD with parallel sides BC and AD, point H on AB such
that ∠DHA = 90°, CH = CD = 13, and AD = 19, prove that the length of BC = 9.5. -/
theorem trapezoid_BC_length 
  (A B C D H : Type)
  [is_trapezoid : ∀ (A B C D : Type), parallel BC AD] 
  (is_right_angle : ∠ (D H A) = π / 2)
  (length_CH : CH = 13)
  (length_CD : CD = 13)
  (length_AD : AD = 19) :
  BC = 9.5 :=
sorry

end trapezoid_BC_length_l640_640650


namespace f_g_g_f_l640_640205

noncomputable def f (x: ℝ) := 1 - 2 * x
noncomputable def g (x: ℝ) := x^2 + 3

theorem f_g (x : ℝ) : f (g x) = -2 * x^2 - 5 :=
by
  sorry

theorem g_f (x : ℝ) : g (f x) = 4 * x^2 - 4 * x + 4 :=
by
  sorry

end f_g_g_f_l640_640205


namespace necessary_but_not_sufficient_l640_640786

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x > 2 → |x| ≥ 1) ∧ (∃ x : ℝ, |x| ≥ 1 ∧ ¬ (x > 2)) :=
by
  sorry

end necessary_but_not_sufficient_l640_640786


namespace combined_leak_amount_l640_640843

def largest_hole_rate : ℝ := 3 -- ounces per minute
def medium_hole_rate : ℝ := largest_hole_rate / 2
def smallest_hole_rate : ℝ := medium_hole_rate / 3
def time_period : ℝ := 120 -- minutes (2 hours)

def largest_hole_leak : ℝ := largest_hole_rate * time_period
def medium_hole_leak : ℝ := medium_hole_rate * time_period
def smallest_hole_leak : ℝ := smallest_hole_rate * time_period

def total_leak : ℝ := largest_hole_leak + medium_hole_leak + smallest_hole_leak

theorem combined_leak_amount : total_leak = 600 := by
  -- The proof details go here
  sorry

end combined_leak_amount_l640_640843


namespace cube_intersection_area_l640_640130

theorem cube_intersection_area :
  ∀ s : ℝ, s = 2 → ∀ plane_parallel : Prop, plane_parallel → 
  ∀ at_middle: Prop, at_middle → ∃ area : ℝ, area = 4 :=
by
  intro s hs plane_parallel h_plane at_middle h_middle
  use 4
  sorry

end cube_intersection_area_l640_640130


namespace two_point_distribution_properties_l640_640272

open ProbabilityTheory

noncomputable def X : ℕ → ℝ := λ n, if n = 0 then 0 else if n = 1 then 1 else 0

theorem two_point_distribution_properties :
  (PMF.toOuterMeasure (PMF.ofMultiset [0, 1])).volume {1} = (1 / 2) ∧
  PMF.mean (PMF.ofMultiset [0, 1]) = (1 / 2) ∧
  PMF.mean (PMF.map (λ x, 2 * x) (PMF.ofMultiset [0, 1])) ≠ (1 / 2) ∧
  PMF.variance (PMF.ofMultiset [0, 1]) = (1 / 4) :=
by
  sorry

end two_point_distribution_properties_l640_640272


namespace annual_cost_l640_640500

def monday_miles : ℕ := 50
def wednesday_miles : ℕ := 50
def friday_miles : ℕ := 50
def sunday_miles : ℕ := 50

def tuesday_miles : ℕ := 100
def thursday_miles : ℕ := 100
def saturday_miles : ℕ := 100

def cost_per_mile : ℝ := 0.1
def weekly_fee : ℝ := 100
def weeks_in_year : ℕ := 52

noncomputable def total_weekly_miles : ℕ := 
  (monday_miles + wednesday_miles + friday_miles + sunday_miles) * 1 +
  (tuesday_miles + thursday_miles + saturday_miles) * 1

noncomputable def weekly_mileage_cost : ℝ := total_weekly_miles * cost_per_mile

noncomputable def weekly_total_cost : ℝ := weekly_fee + weekly_mileage_cost

noncomputable def annual_total_cost : ℝ := weekly_total_cost * weeks_in_year

theorem annual_cost (monday_miles wednesday_miles friday_miles sunday_miles
                     tuesday_miles thursday_miles saturday_miles : ℕ)
                     (cost_per_mile weekly_fee : ℝ) 
                     (weeks_in_year : ℕ) :
  monday_miles = 50 → wednesday_miles = 50 → friday_miles = 50 → sunday_miles = 50 →
  tuesday_miles = 100 → thursday_miles = 100 → saturday_miles = 100 →
  cost_per_mile = 0.1 → weekly_fee = 100 → weeks_in_year = 52 →
  annual_total_cost = 7800 :=
by
  intros
  sorry

end annual_cost_l640_640500


namespace expression_parity_l640_640256

theorem expression_parity (a b c : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_a_odd : a % 2 = 1) (h_b_odd : b % 2 = 1) : (3^a + (b + 1)^2 * c) % 2 = 1 :=
by sorry

end expression_parity_l640_640256


namespace floor_sqrt_product_l640_640155

theorem floor_sqrt_product :
  (∏ i in Finset.range 18 | odd (2*i + 1), ⌊real.sqrt (2*i + 1)⌋ : ℤ) /
  (∏ i in Finset.range 18 | even (2*i + 2), ⌊real.sqrt (2*i + 2)⌋ : ℤ) = 1 / 6 := by
  sorry

end floor_sqrt_product_l640_640155


namespace relationship_ab_c_l640_640957
open Real

noncomputable def a : ℝ := (1 / 3) ^ (log 3 / log (1 / 3))
noncomputable def b : ℝ := (1 / 3) ^ (log 4 / log (1 / 3))
noncomputable def c : ℝ := 3 ^ log 3

theorem relationship_ab_c : c > b ∧ b > a := by
  sorry

end relationship_ab_c_l640_640957


namespace intersection_problem_complement_subset_problem_l640_640414

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def set_B (m : ℝ) : Set ℝ := {x | m - 3 ≤ x ∧ x ≤ m}

theorem intersection_problem (m : ℝ) :
  (set_A ∩ set_B m = {x : ℝ | 2 ≤ x ∧ x ≤ 4}) → m = 5 :=
by
  sorry

theorem complement_subset_problem (m : ℝ) :
  (set_A ⊆ {x : ℝ | x > m ∨ x < m - 3}) → m ∈ (-∞, -2) ∪ (7, ∞) :=
by
  sorry

end intersection_problem_complement_subset_problem_l640_640414


namespace solve_inequality_l640_640164

open Real

theorem solve_inequality (x : ℝ) : (x ≠ 3) ∧ (x * (x + 1) / (x - 3) ^ 2 ≥ 9) ↔ (2.13696 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4.73804) :=
by
  sorry

end solve_inequality_l640_640164


namespace a_n_nine_l640_640675

open Nat

-- Definitions as specified in the problem statement.
def is_arithmetic_sequence (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → 2 * S n = S (n + 1) + S (n + 2)

-- Given conditions.
variables {a : ℕ → ℤ} {S : ℕ → ℤ}
hypothesis (h1 : a 2 = 4)
hypothesis (h2 : ∀ n, S (n + 1) = S n + a (n + 1))
hypothesis (h3 : is_arithmetic_sequence S)

-- Lean 4 statement to prove.
theorem a_n_nine : a 9 = -512 :=
sorry

end a_n_nine_l640_640675


namespace polar_to_rectangular_range_of_m_l640_640295

-- Definition of the parametric equations of curve C
def curve (t : ℝ) : ℝ × ℝ :=
  (sqrt(3) * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line (rho theta m : ℝ) : Prop :=
  rho * sin (theta + π / 3) + m = 0

-- Definition of the rectangular equation of line l
def rectangular_line (x y m : ℝ) : Prop :=
  sqrt(3) * x + y + 2 * m = 0

-- Convert polar line equation to rectangular form
theorem polar_to_rectangular (rho theta m x y : ℝ)
  (h1 : polar_line rho theta m)
  (h2 : rho = sqrt (x^2 + y^2))
  (h3 : cos theta * rho = x ∧ sin theta * rho = y) :
  rectangular_line x y m :=
sorry

-- Range of values for m for which l intersects C
theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, (sqrt(3) * cos (2 * t), 2 * sin t) ∈ {p : ℝ × ℝ | rectangular_line p.1 p.2 m}) ↔
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end polar_to_rectangular_range_of_m_l640_640295


namespace arithmetic_operators_identification_l640_640540

-- Given conditions:
-- Replaced letters: A, B, C, D, E
-- Original equations:
-- 4 A 2 = 2
-- 8 = 4 C 2
-- 2 D 3 = 5
-- 4 = 5 E 1

theorem arithmetic_operators_identification :
  (A = (/) ∧ B = (=) ∧ C = (*) ∧ D = (+) ∧ E = (-)) :=
by
  sorry

end arithmetic_operators_identification_l640_640540


namespace least_n_condition_l640_640406

theorem least_n_condition (n : ℕ) (h : n ≠ 0) :
  (1 / 3 + 1 / 4 + 1 / 8 + 1 / n : ℚ).denom = 1 → n = 24 :=
by
  sorry

end least_n_condition_l640_640406


namespace sum_of_real_solutions_l640_640615

noncomputable def sum_of_solutions := (4527 : ℝ) / 1296

theorem sum_of_real_solutions :
  let f (x : ℝ) := (sqrt (2 * x) + sqrt (9 / x) + 2 * sqrt (2 * x + 9 / x)) in
  ∀ x y z : ℝ, f x = 9 ∧ f y = 9 ∧ f z = 9 →
  (x + y + z = sum_of_solutions) :=
by
  sorry

end sum_of_real_solutions_l640_640615


namespace smallest_x_for_multiple_l640_640073

theorem smallest_x_for_multiple (x : ℕ) : (450 * x) % 720 = 0 ↔ x = 8 := 
by {
  sorry
}

end smallest_x_for_multiple_l640_640073


namespace unique_strictly_increasing_sequence_l640_640038

theorem unique_strictly_increasing_sequence :
  ∃! (k : ℕ),
  ∃ (a : ℕ → ℕ) (strictly_increasing : ∀ i j, i < j → a i < a j) (nonnegative: ∀ i, 0 ≤ a i)
  (sum_equals : (∑ i in finset.range k, 2 ^ (a i)) = (\(2^225 + 1\) / \(2^15 + 1\))),
  k = 113 :=
sorry

end unique_strictly_increasing_sequence_l640_640038


namespace group_size_l640_640005

theorem group_size (n : ℕ) (h : 2.5 * n = 20) : n = 8 :=
sorry

end group_size_l640_640005


namespace rect_eq_line_range_of_m_l640_640333

-- Definitions based on the conditions
def parametric_x (t : ℝ) : ℝ := sqrt 3 * cos (2 * t)
def parametric_y (t : ℝ) : ℝ := 2 * sin t

def polar_rho (θ m : ℝ) : ℝ := -(m / (sin (θ + π / 3)))

-- Rectangular equation of the line l
theorem rect_eq_line (x y m : ℝ) : polar_rho (atan2 y x) m * sin (atan2 y x + π / 3) + m = 0 → sqrt 3 * x + y + 2 * m = 0 :=
by
  sorry

-- Range of values for m
theorem range_of_m (C : ℝ → ℝ × ℝ)
  (hC : ∀ t, C t = (parametric_x t, parametric_y t)) (m : ℝ) :
  (∃ t, sqrt 3 * fst (C t) + snd (C t) + 2 * m = 0) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
by
  sorry

end rect_eq_line_range_of_m_l640_640333


namespace modulus_product_l640_640176

open Complex -- to open the complex namespace

-- Define the complex numbers
def z1 : ℂ := 10 - 5 * Complex.I
def z2 : ℂ := 7 + 24 * Complex.I

-- State the theorem to prove
theorem modulus_product : abs (z1 * z2) = 125 * Real.sqrt 5 := by
  sorry

end modulus_product_l640_640176


namespace problem_part1_problem_part2_l640_640305

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l_rect_eq (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (sqrt 3 * x + y + 2 * m = 0)

def line_l_range_m (m : ℝ) : Prop :=
  -19/12 ≤ m ∧ m ≤ 5/2

theorem problem_part1 (m : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * sin (θ + π / 3) + m = 0) ↔ line_l_rect_eq m := 
sorry

theorem problem_part2 :
  ∀ m : ℝ, 
  (∃ t : ℝ, line_l_rect_eq m (curve_C t).1 (curve_C t).2) ↔ line_l_range_m m := 
sorry

end problem_part1_problem_part2_l640_640305


namespace problem_part1_problem_part2_l640_640301

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l_rect_eq (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (sqrt 3 * x + y + 2 * m = 0)

def line_l_range_m (m : ℝ) : Prop :=
  -19/12 ≤ m ∧ m ≤ 5/2

theorem problem_part1 (m : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * sin (θ + π / 3) + m = 0) ↔ line_l_rect_eq m := 
sorry

theorem problem_part2 :
  ∀ m : ℝ, 
  (∃ t : ℝ, line_l_rect_eq m (curve_C t).1 (curve_C t).2) ↔ line_l_range_m m := 
sorry

end problem_part1_problem_part2_l640_640301


namespace batsman_average_increase_l640_640545

theorem batsman_average_increase:
  ∀ (A : ℝ),
  let total_runs_18 := 18 * A in
  let total_runs_19 := total_runs_18 + 100 in
  total_runs_19 / 19 = 64 → (64 - A = 2) :=
by
  intros A total_runs_18 total_runs_19 h
  sorry

end batsman_average_increase_l640_640545


namespace pentagonal_number_theorem_l640_640448

theorem pentagonal_number_theorem (z : ℂ) :
    (∑ n : ℤ, (-1)^n * z^((3 * n * n - n) / 2)) = ∏ n in (Finset.range (nat.succ ℕ)).filter (λ n, n > 0), (1 - z^n) :=
by
  sorry

end pentagonal_number_theorem_l640_640448


namespace parallel_or_concurrent_l640_640410

theorem parallel_or_concurrent
  (ABC : Triangle)
  (I : Point)
  (omega : Circle)
  (A : Point) (B : Point) (C : Point)
  (D E : Point)
  (F G : Point)
  (P K : Point)
  (h1 : IsIncenter(ABC, I))
  (h2 : IsCircumcircle(omega, ABC))
  (h3 : SecondIntersection(omega, I, A, D))
  (h4 : SecondIntersection(omega, I, B, E))
  (h5 : ChordIntersection(DE, AC, F))
  (h6 : ChordIntersection(DE, BC, G))
  (h7 : Parallel(LineThrough(F, AD), LineThrough(G, BE), LineThrough(P)))
  (h8 : TangentsMeetAtPoint(omega, A, B, K)) :
  ConcurrentOrParallel(Lines(AE, BD, KP)) :=
sorry

end parallel_or_concurrent_l640_640410


namespace rect_eq_line_range_of_m_l640_640331

-- Definitions based on the conditions
def parametric_x (t : ℝ) : ℝ := sqrt 3 * cos (2 * t)
def parametric_y (t : ℝ) : ℝ := 2 * sin t

def polar_rho (θ m : ℝ) : ℝ := -(m / (sin (θ + π / 3)))

-- Rectangular equation of the line l
theorem rect_eq_line (x y m : ℝ) : polar_rho (atan2 y x) m * sin (atan2 y x + π / 3) + m = 0 → sqrt 3 * x + y + 2 * m = 0 :=
by
  sorry

-- Range of values for m
theorem range_of_m (C : ℝ → ℝ × ℝ)
  (hC : ∀ t, C t = (parametric_x t, parametric_y t)) (m : ℝ) :
  (∃ t, sqrt 3 * fst (C t) + snd (C t) + 2 * m = 0) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
by
  sorry

end rect_eq_line_range_of_m_l640_640331


namespace n95_masks_quality_inspection_l640_640491

theorem n95_masks_quality_inspection
  (num_masks_tested : Fin 6 → ℕ)
  (num_qualified_masks : Fin 6 → ℕ)
  (qualified_rate : Fin 6 → ℚ)
  (num_masks_tested_0 : num_masks_tested 0 = 500)
  (num_masks_tested_1 : num_masks_tested 1 = 1000)
  (num_masks_tested_2 : num_masks_tested 2 = 1500)
  (num_masks_tested_3 : num_masks_tested 3 = 2000)
  (num_masks_tested_4 : num_masks_tested 4 = 3000)
  (num_masks_tested_5 : num_masks_tested 5 = 4000)
  (num_qualified_masks_0 : num_qualified_masks 0 = 471)
  (num_qualified_masks_1 : num_qualified_masks 1 = 946)
  (num_qualified_masks_2 : num_qualified_masks 2 = 1425)
  (num_qualified_masks_3 : num_qualified_masks 3 = 1898)
  (num_qualified_masks_4 : num_qualified_masks 4 = 2853)
  (num_qualified_masks_5 : num_qualified_masks 5 = 3812)
  (qualified_rate_0 : qualified_rate 0 = 471 / 500)
  (qualified_rate_1 : qualified_rate 1 = 946 / 1000)
  (qualified_rate_2 : qualified_rate 2 = 1425 / 1500)
  (qualified_rate_5 : qualified_rate 5 = 3812 / 4000) :
  let a := qualified_rate 3 in
  let b := qualified_rate 4 in
  let estimated_probability := (qualified_rate 0 +
                               qualified_rate 1 +
                               qualified_rate 2 +
                               a +
                               b +
                               qualified_rate 5) / 6 in
  let total_masks_needed := 285000 / estimated_probability in
  a = 1898 / 2000 ∧
  b = 2853 / 3000 ∧
  estimated_probability ≈ 0.95 ∧
  total_masks_needed = 300000 :=
by
  sorry

end n95_masks_quality_inspection_l640_640491


namespace frog_vertical_boundary_probability_l640_640550

def grid := {p : ℕ × ℕ // p.1 ≤ 6 ∧ p.2 ≤ 6}

def is_boundary (p : ℕ × ℕ) : Prop :=
  p.1 = 0 ∨ p.1 = 6 ∨ p.2 = 0 ∨ p.2 = 6

def P (p : grid) : ℚ :=
  if is_boundary p.val then
    if p.val.1 = 0 ∨ p.val.1 = 6 then 1 else 0
  else 1/4 * (P ⟨(p.val.1 - 1, p.val.2), sorry⟩ + P ⟨(p.val.1 + 1, p.val.2), sorry⟩ + 
              P ⟨(p.val.1, p.val.2 - 1), sorry⟩ + P ⟨(p.val.1, p.val.2 + 1), sorry⟩)

theorem frog_vertical_boundary_probability : P ⟨(2, 3), sorry⟩ = 17/32 := sorry

end frog_vertical_boundary_probability_l640_640550


namespace rectangular_equation_of_l_range_of_m_intersection_l640_640367

-- Definitions based on the problem conditions
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  let x := sqrt 3 * cos (2 * t)
  let y := 2 * sin t
  (x, y)

def polar_line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- The rectangular equation converted from polar form
theorem rectangular_equation_of_l (x y m : ℝ) :
  (polar_line_l (sqrt (x^2 + y^2)) (atan2 y x) m) ↔ (√3 * x + y + 2 * m = 0) :=
sorry

-- Range of values for m ensuring intersection of line l with curve C
theorem range_of_m_intersection (m : ℝ) :
  ((-19/12 : ℝ) ≤ m ∧ m ≤ (5/2 : ℝ)) ↔
  ∃ t : ℝ, let x := sqrt 3 * cos (2 * t)
           let y := 2 * sin t
           (sqrt 3 * x + y + 2 * m = 0 ∧ (-2 ≤ y ∧ y ≤ 2)) :=
sorry

end rectangular_equation_of_l_range_of_m_intersection_l640_640367


namespace L_is_vector_subspace_R_is_equivalence_relation_vectors_same_equivalence_class_l640_640508

open Real

def R3 := (ℝ × ℝ × ℝ)

def L : Set R3 := { x | x.1 + x.2 + x.3 = 0 }

def vector_add (u v : R3) : R3 := (u.1 + v.1, u.2 + v.2, u.3 + v.3)
def scalar_mul (c : ℝ) (u : R3) : R3 := (c * u.1, c * u.2, c * u.3)

def is_vector_subspace (L : Set R3) : Prop :=
  (0, 0, 0) ∈ L ∧
  ∀ (u v : R3), u ∈ L → v ∈ L → vector_add u v ∈ L ∧
  ∀ (c : ℝ) (u : R3), u ∈ L → scalar_mul c u ∈ L

def equivalence_relation (R : R3 → R3 → Prop) : Prop :=
  (∀ (x : R3), R x x) ∧
  (∀ (x y : R3), R x y → R y x) ∧
  (∀ (x y z : R3), R x y → R y z → R x z)

def same_equivalence_class (x y : R3) (L : Set R3) : Prop :=
  x - y ∈ L

theorem L_is_vector_subspace : is_vector_subspace L := 
sorry

theorem R_is_equivalence_relation : equivalence_relation (λ x y, x - y ∈ L) := 
sorry

theorem vectors_same_equivalence_class : same_equivalence_class (-1, 3, 2) (0, 0, 4) L ∧
  same_equivalence_class (-1, 3, 2) (1, 2, 0) L :=
sorry

end L_is_vector_subspace_R_is_equivalence_relation_vectors_same_equivalence_class_l640_640508


namespace f_is_even_f_is_monotonic_l640_640232

noncomputable def f (x : ℝ) : ℝ := |x| + 1

-- Theorem 1: Prove that f(x) is even
theorem f_is_even : ∀ x : ℝ, f x = f (-x) :=
by
  -- Proof omitted
  sorry

-- Theorem 2: Prove the monotonic intervals of f(x)
theorem f_is_monotonic : (∀ x1 x2 : ℝ, x1 < x2 < 0 → f x2 < f x1) ∧ (∀ x1 x2 : ℝ, 0 < x1 < x2 → f x1 < f x2) :=
by
  -- Proof omitted
  sorry

end f_is_even_f_is_monotonic_l640_640232


namespace probability_expression_equals_one_over_n_minus_one_l640_640946

-- Definitions and conditions
variable (n : ℕ)
variable (even_n : Even n)
variable (p : ℕ → ℝ)

-- The expression we want to prove is equal to 1/(n-1)
noncomputable def special_expression (n : ℕ) (p : ℕ → ℝ) : ℝ :=
(p (n-1) - p (n-2) + ... + p 1) / (p (n-1) + p (n-2) + ... + p 1)

theorem probability_expression_equals_one_over_n_minus_one
  (n_pos : 0 < n)
  (even_n : Even n)
  (p_def : ∀ k, (0 < k ∧ k < n) → p k = 1 / (n-1)) :
  special_expression n p = 1 / (n - 1) :=
by
  -- Proof is omitted
  sorry

end probability_expression_equals_one_over_n_minus_one_l640_640946


namespace johns_donation_l640_640535

theorem johns_donation (avg_after : ℝ) (contributions_before : ℕ) (avg_increase : ℝ) : 
  avg_after = 75 → 
  contributions_before = 3 → 
  avg_increase = 1.5 → 
  let avg_before := avg_after / avg_increase in
  let total_before_john := contributions_before * avg_before in
  let total_after_john := total_before_john + 150 in
  let new_contributions := contributions_before + 1 in
  (avg_after = total_after_john / new_contributions) → 
  150 = 150 :=
by sorry

end johns_donation_l640_640535


namespace complete_collection_probability_l640_640053

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end complete_collection_probability_l640_640053


namespace train_passes_jogger_in_32_seconds_l640_640119

theorem train_passes_jogger_in_32_seconds :
  ∀ (speed_jogger speed_train relative_speed distance_ahead train_length total_distance time_to_pass : ℝ),
    speed_jogger = 9 * (1000 / 3600) →
    speed_train = 45 * (1000 / 3600) →
    relative_speed = speed_train - speed_jogger →
    distance_ahead = 200 →
    train_length = 120 →
    total_distance = distance_ahead + train_length →
    time_to_pass = total_distance / relative_speed →
    time_to_pass = 32 :=
by
  intros speed_jogger speed_train relative_speed distance_ahead train_length total_distance time_to_pass
  assume h1 : speed_jogger = 9 * (1000 / 3600)
  assume h2 : speed_train = 45 * (1000 / 3600)
  assume h3 : relative_speed = speed_train - speed_jogger
  assume h4 : distance_ahead = 200
  assume h5 : train_length = 120
  assume h6 : total_distance = distance_ahead + train_length
  assume h7 : time_to_pass = total_distance / relative_speed
  sorry

end train_passes_jogger_in_32_seconds_l640_640119


namespace tess_distance_proof_l640_640003

noncomputable def distance_from_point_A (t : ℕ) : ℝ := 
  if t ≤ 2 then t -- First half of the journey, 0 to maximum
  else 4 - t     -- Second half of the journey, maximum to 0

theorem tess_distance_proof : 
    tess_path : ∀ t, 0 ≤ t ∧ t ≤ 4 → distance_from_point_A t = (if t ≤ 2 then t else 4 - t) ∧ distance_from_point_A 0 = 0 ∧ distance_from_point_A 2 = 2 ∧ distance_from_point_A 4 = 0 := sorry

end tess_distance_proof_l640_640003


namespace arithmetic_progression_primes_l640_640103

theorem arithmetic_progression_primes (p₁ p₂ p₃ : ℕ) (d : ℕ) 
  (hp₁ : Prime p₁) (hp₁_cond : 3 < p₁) 
  (hp₂ : Prime p₂) (hp₂_cond : 3 < p₂) 
  (hp₃ : Prime p₃) (hp₃_cond : 3 < p₃) 
  (h_prog_1 : p₂ = p₁ + d) (h_prog_2 : p₃ = p₁ + 2 * d) : 
  d % 6 = 0 :=
sorry

end arithmetic_progression_primes_l640_640103


namespace merchant_profit_percentage_l640_640878

def unit_cost_price_A : ℝ := 10
def unit_cost_price_B : ℝ := 18

def marked_price_A : ℝ := unit_cost_price_A * (1 + 0.6)
def marked_price_B : ℝ := unit_cost_price_B * (1 + 0.8)

def selling_price_A : ℝ := marked_price_A * (1 - 0.2)
def selling_price_B : ℝ := marked_price_B * (1 - 0.1)

def units_A : ℝ := 30
def units_B : ℝ := 20

def total_revenue_A : ℝ := units_A * selling_price_A
def total_revenue_B : ℝ := units_B * selling_price_B

def total_revenue : ℝ := total_revenue_A + total_revenue_B

def total_cost_A : ℝ := units_A * unit_cost_price_A
def total_cost_B : ℝ := units_B * unit_cost_price_B

def total_cost : ℝ := total_cost_A + total_cost_B

def profit : ℝ := total_revenue - total_cost

def profit_percentage : ℝ := (profit / total_cost) * 100

theorem merchant_profit_percentage : profit_percentage ≈ 46.55 := by
  -- The proof would go here
  sorry

end merchant_profit_percentage_l640_640878


namespace order_of_function_values_l640_640222

noncomputable def f : ℝ → ℝ := sorry

theorem order_of_function_values (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (x + 4))
  (h2 : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → x1 < x2 → x2 ≤ 2 → f x1 < f x2)
  (h3 : ∀ x, f(x + 2) = f(-x - 2))
  : f 7 < f 4.5 ∧ f 4.5 < f 6.5 := sorry

end order_of_function_values_l640_640222


namespace eight_bees_have_48_legs_l640_640860

  def legs_per_bee : ℕ := 6
  def number_of_bees : ℕ := 8
  def total_legs : ℕ := 48

  theorem eight_bees_have_48_legs :
    number_of_bees * legs_per_bee = total_legs :=
  by
    sorry
  
end eight_bees_have_48_legs_l640_640860


namespace soccer_team_goals_l640_640886

theorem soccer_team_goals (total_players total_goals games_played : ℕ)
(one_third_players_goals : ℕ) :
  total_players = 24 →
  total_goals = 150 →
  games_played = 15 →
  one_third_players_goals = (total_players / 3) * games_played →
  (total_goals - one_third_players_goals) = 30 :=
by
  intros h1 h2 h3 h4
  rw h1 at h4
  rw h3 at h4
  sorry

end soccer_team_goals_l640_640886


namespace domain_of_f_l640_640476

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log (x + 2) + Real.sqrt (2 - 2 * x)

theorem domain_of_f : { x : ℝ | x > -2 ∧ x ≠ -1 ∧ x ≤ 1 } = (-2, -1) ∪ (-1, 1] := by
  sorry

end domain_of_f_l640_640476


namespace plane_line_perpendicular_converse_l640_640633

variables {α β : Type} [Plane α] [Plane β] (m : Line) [InPlane m α]

-- Define the perpendicular relations
def Plane.perpendicular (p₁ p₂ : Plane) : Prop := 
  -- Assume a definition in Mathlib for planes being perpendicular
  sorry

def Line.perpendicular (l : Line) (p : Plane) : Prop := 
  -- Assume a definition in Mathlib for lines being perpendicular to planes
  sorry

theorem plane_line_perpendicular_converse
  (hα : α ≠ β)
  (hm_in_α : InPlane m α) :
  (Plane.perpendicular α β) → (Line.perpendicular m β) 
  ∧ (¬(Plane.perpendicular α β) → (¬(Line.perpendicular m β))) :=
sorry

end plane_line_perpendicular_converse_l640_640633


namespace line_through_P_parallel_to_AB_is_correct_circumscribed_circle_of_OAB_is_correct_l640_640246

-- Definitions based on conditions
def A := (4 : ℝ, 0 : ℝ)
def B := (0 : ℝ, 2 : ℝ)
def P := (2 : ℝ, 3 : ℝ)
def O := (0 : ℝ, 0 : ℝ)

-- Question 1: Prove the equation of the line passing through P and parallel to AB
theorem line_through_P_parallel_to_AB_is_correct :
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧ 
                 a * (B.1 - A.1) + b * (B.2 - A.2) = 0 ∧ 
                 a = 1 ∧ b = 2 ∧ c = -8 :=
sorry

-- Question 2: Prove the equation of the circumscribed circle of ΔOAB
theorem circumscribed_circle_of_OAB_is_correct :
  ∃ (h k r : ℝ), (h = 2) ∧ (k = 1) ∧ 
                 (r = sqrt 5) ∧ 
                 ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end line_through_P_parallel_to_AB_is_correct_circumscribed_circle_of_OAB_is_correct_l640_640246


namespace probability_calculation_l640_640754

-- Define the coordinates for the diagonal endpoints of square S
def point1 := (1/5, -1/5)
def point2 := (-1/5, 1/5)

-- Define the random point v within the given range of x and y
def random_point (x y : ℝ) : Prop :=
  -100 <= x ∧ x <= 100 ∧ -100 <= y ∧ y <= 100

-- Define the circle C(v) with a radius of 1 centered at point v
def circle (v : ℝ × ℝ) (radius : ℝ) : Prop :=
  radius = 1

-- Probability calculation
def probability_of_integer_point_not_overlapped (v : ℝ × ℝ) : ℝ :=
  1/16

-- The theorem that needs to be proven
theorem probability_calculation :
  ∀ v : ℝ × ℝ, random_point v.1 v.2 → circle v 1 → probability_of_integer_point_not_overlapped v = 1/16 :=
by sorry

end probability_calculation_l640_640754


namespace cubic_polynomial_roots_l640_640975

theorem cubic_polynomial_roots (x1 x2 x3 s t u : ℝ) :
  x1 + x2 + x3 = s →
  x1 * x2 + x2 * x3 + x3 * x1 = t →
  x1 * x2 * x3 = u →
  ∃ p : Polynomial ℝ, p = Polynomial.X^3 - s * Polynomial.X^2 + t * Polynomial.X - u ∧
                      Polynomial.eval x1 p = 0 ∧
                      Polynomial.eval x2 p = 0 ∧
                      Polynomial.eval x3 p = 0 :=
by
  sorry

end cubic_polynomial_roots_l640_640975


namespace point_not_in_second_quadrant_l640_640989

-- Define the point P and the condition
def point_is_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def point (m : ℝ) : ℝ × ℝ :=
  (m + 1, m)

-- The main theorem stating that P cannot be in the second quadrant
theorem point_not_in_second_quadrant (m : ℝ) : ¬ point_is_in_second_quadrant (point m) :=
by
  sorry

end point_not_in_second_quadrant_l640_640989


namespace integral_solution_l640_640156

noncomputable def definite_integral : ℝ :=
  ∫ x in (0 : ℝ)..(5 : ℝ), x ^ 2 * sqrt (25 - x ^ 2)

theorem integral_solution :
  definite_integral = 625 * Real.pi / 16 :=
by
  -- The proof would typically go here, but for now we place a placeholder
  sorry

end integral_solution_l640_640156


namespace remainder_even_nearest_to_S_div_5_l640_640910

-- Given conditions
def sum_S : ℤ :=
  ∑ k in Finset.range 502, 2013 / ((2 + 4 * k) * (6 + 4 * k))

-- Statement to prove
theorem remainder_even_nearest_to_S_div_5 : 
  let S := sum_S; 
  let nearest_even_S := 2 * (S / 2).round;
  nearest_even_S % 5 = 2 := 
by 
  sorry

end remainder_even_nearest_to_S_div_5_l640_640910


namespace no_statement_implies_neg_p_or_q_l640_640162

def statement1 (p q : Prop) : Prop := p ∨ q
def statement2 (p q : Prop) : Prop := p ∨ ¬ q
def statement3 (p q : Prop) : Prop := ¬ p ∨ q
def statement4 (p q : Prop) : Prop := ¬ p ∧ q
def neg_p_or_q (p q : Prop) : Prop := ¬ (p ∨ q)

theorem no_statement_implies_neg_p_or_q (p q : Prop) :
  ¬ (statement1 p q → neg_p_or_q p q) ∧
  ¬ (statement2 p q → neg_p_or_q p q) ∧
  ¬ (statement3 p q → neg_p_or_q p q) ∧
  ¬ (statement4 p q → neg_p_or_q p q)
:= by
  sorry

end no_statement_implies_neg_p_or_q_l640_640162


namespace smallest_n_satisfies_conditions_l640_640618

def digitSum (n : ℕ) : ℕ :=
  n.digits.sum

theorem smallest_n_satisfies_conditions :
  ∀ n : ℕ, 0 < n → digitSum n = 20 → digitSum (n + 864) = 20 → n = 695 :=
  by
  intros n hn hsn hsn_add
  sorry

end smallest_n_satisfies_conditions_l640_640618


namespace initial_girls_count_l640_640284

theorem initial_girls_count (initial_boys : ℕ) (additional_girls : ℕ) (girls_more_than_boys : ℕ) :
  initial_boys = 410 → additional_girls = 465 → girls_more_than_boys = 687 →
  ∃ G, G + additional_girls = initial_boys + girls_more_than_boys ∧ G = 632 :=
begin
  intros h1 h2 h3,
  use 632,
  split,
  { calc
      632 + 465 = 1097 : by norm_num
      ... = 410 + 687 : by rw [h1, h3] },
  { refl }
end

end initial_girls_count_l640_640284


namespace number_of_x_satisfying_f_l640_640931

def f : ℕ × ℕ → ℕ
| (0, 0) := 0
| (2 * x, 2 * y) := f (x, y)
| (2 * x + 1, 2 * y + 1) := f (x, y)
| (2 * x + 1, 2 * y) := f (x, y) + 1
| (2 * x, 2 * y + 1) := f (x, y) + 1

theorem number_of_x_satisfying_f (a b n : ℕ) (h : f (a, b) = n) :
  ∃ (count : ℕ), (∀ x, f (a, x) + f (b, x) = n ↔ x < count) ∧ count = 2^n :=
sorry

end number_of_x_satisfying_f_l640_640931


namespace non_invited_students_l640_640278

variables (Class : Type) [Fintype Class]
variable (Mia : Class) (Friends : Class → Prop)
variable (FriendsOfFriends : Class → Prop)
variables (isolatedGroup : Finset Class)
variables (isolatedIndividuals : Finset Class)

-- Assumptions based on the conditions
axiom class_size : Fintype.card Class = 25
axiom Mia_friends : (Fintype.card {x // Friends x}) = 4
axiom friends_of_friends : ∀ x : {x // Friends x}, (Fintype.card {y // FriendsOfFriends y ∧ ¬ Friends y}) = 3
axiom distinct_groups : ∀ x y: {x // Friends x}, x ≠ y → Disjoint {z // FriendsOfFriends z ∧ ¬ Friends z}
axiom isolated_group_size : isolatedGroup.card = 3
axiom isolated_individuals_size : isolatedIndividuals.card = 2
axiom no_intersection: Disjoint isolatedGroup {x // Friends x} ∧ Disjoint isolatedIndividuals {x // Friends x}
axiom total_non_invited: Fintype.card isolatedGroup + isolatedIndividuals.card = 5

theorem non_invited_students : 
  let invited := 1 + (Fintype.card {x // Friends x}) + 4*3 in
  25 - invited = 8 := sorry

end non_invited_students_l640_640278


namespace relationship_among_neg_a_neg_a3_a2_l640_640759

theorem relationship_among_neg_a_neg_a3_a2 (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 :=
by sorry

end relationship_among_neg_a_neg_a3_a2_l640_640759


namespace distinct_real_roots_l640_640479

theorem distinct_real_roots (p : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * |x1| - p = 0) ∧ (x2^2 - 2 * |x2| - p = 0)) → p > -1 :=
by
  intro h
  sorry

end distinct_real_roots_l640_640479


namespace mateo_grape_soda_l640_640741

theorem mateo_grape_soda :
  ∀ (G : ℕ),
    let julio_orange := 4 * 2,
    let julio_grape := 7 * 2,
    let mateo_orange := 1 * 2,
    let mateo_total := (2 + 2 * G),
    let julio_total := julio_orange + julio_grape,
    let diff := julio_total - mateo_total
    in diff = 14 → G = 3 :=
by
  sorry

end mateo_grape_soda_l640_640741


namespace distinct_constructions_l640_640113

def num_cube_constructions (white_cubes : Nat) (blue_cubes : Nat) : Nat :=
  if white_cubes = 5 ∧ blue_cubes = 3 then 5 else 0

theorem distinct_constructions : num_cube_constructions 5 3 = 5 :=
by
  sorry

end distinct_constructions_l640_640113


namespace rectangular_eq_of_line_l_range_of_m_l640_640345

noncomputable def curve_C_param_eq (t : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l_polar_eq (θ ρ m : ℝ) : Prop :=
  ρ * Real.sin(θ + Real.pi / 3) + m = 0

theorem rectangular_eq_of_line_l (x y m : ℝ) (h : line_l_polar_eq (Real.atan2 y x) (Real.sqrt (x^2 + y^2)) m) :
  sqrt 3 * x + y + 2 * m = 0 :=
  sorry

theorem range_of_m (m : ℝ) (t : ℝ) (h : ∃ (t : ℝ), curve_C_param_eq t ∈ set_of (λ p, p.2 = -√3 * p.1 - 2 * m))
  : -19/12 ≤ m ∧ m ≤ 5/2 :=
  sorry

end rectangular_eq_of_line_l_range_of_m_l640_640345


namespace michelle_savings_in_local_currency_l640_640897

theorem michelle_savings_in_local_currency (exchange_rate : ℝ) (exchange_fee_rate : ℝ) (usd_bills : ℝ) :
  usd_bills / (1 - exchange_fee_rate) * exchange_rate ≈ 701.03 :=
by
  -- Conditions extracted from the problem
  let exchange_rate := 0.85
  let exchange_fee_rate := 0.03
  let usd_bills := 800

  -- Proof to be completed
  sorry

end michelle_savings_in_local_currency_l640_640897


namespace log_minus_one_has_one_zero_l640_640026

theorem log_minus_one_has_one_zero : ∃! x : ℝ, x > 0 ∧ (Real.log x - 1 = 0) :=
sorry

end log_minus_one_has_one_zero_l640_640026


namespace factorial_division_l640_640661

theorem factorial_division (h : 10! = 3628800) : 10! / 4! = 151200 := by
  sorry

end factorial_division_l640_640661


namespace rectangular_eq_of_line_l_range_of_m_l640_640344

noncomputable def curve_C_param_eq (t : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l_polar_eq (θ ρ m : ℝ) : Prop :=
  ρ * Real.sin(θ + Real.pi / 3) + m = 0

theorem rectangular_eq_of_line_l (x y m : ℝ) (h : line_l_polar_eq (Real.atan2 y x) (Real.sqrt (x^2 + y^2)) m) :
  sqrt 3 * x + y + 2 * m = 0 :=
  sorry

theorem range_of_m (m : ℝ) (t : ℝ) (h : ∃ (t : ℝ), curve_C_param_eq t ∈ set_of (λ p, p.2 = -√3 * p.1 - 2 * m))
  : -19/12 ≤ m ∧ m ≤ 5/2 :=
  sorry

end rectangular_eq_of_line_l_range_of_m_l640_640344


namespace range_of_m_l640_640230

theorem range_of_m (x m : ℝ) (h1 : (x ≥ 0) ∧ (x ≠ 1) ∧ (x = (6 - m) / 4)) :
    m ≤ 6 ∧ m ≠ 2 :=
by
  sorry

end range_of_m_l640_640230


namespace find_T_l640_640962

-- We assume the existence of a geometric progression with terms less than 100
def geom_progression (b s : Nat) : Prop :=
  b + b * s + b * s ^ 2 + b * s ^ 3 + b * s ^ 4 = 186 ∧ b > 0 ∧ s > 0 ∧ 
  ∀ k : Nat, k < 5 → (b * s ^ k) < 100

-- We prove that the sum of terms in the progression that are squares of integers is 180
theorem find_T (b s : Nat) (h : geom_progression b s) : 
  let terms := [b, b * s, b * s ^ 2, b * s ^ 3, b * s ^ 4] in
  let squares := terms.filter (λ x, ∃ (n : Nat), n ^ 2 = x) in
  T = squares.sum → T = 180 :=
by 
  simp [geom_progression, T]; 
  sorry

end find_T_l640_640962


namespace ratio_of_first_to_third_l640_640438

variables {A B C k : ℤ}

def condition1 := A = 2 * B
def condition2 := ∃ k : ℤ, A = k * C
def condition3 := (A + B + C) / 3 = 88
def condition4 := A - C = 96

theorem ratio_of_first_to_third (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : A / C = 15 / 7 :=
sorry

end ratio_of_first_to_third_l640_640438


namespace factorial_division_l640_640668

theorem factorial_division : (10! / 4! = 151200) :=
by
  have fact_10 : 10! = 3628800 := by sorry
  rw [fact_10]
  -- Proceeding with calculations and assumptions that follow directly from the conditions
  have fact_4 : 4! = 24 := by sorry
  rw [fact_4]
  exact (by norm_num : 3628800 / 24 = 151200)

end factorial_division_l640_640668


namespace power_function_x_value_l640_640482

theorem power_function_x_value (f : ℝ → ℝ) (α : ℝ) (h₁ : ∀ x, f(x) = x^α) (h₂ : f(2) = 8) (h₃ : f(x) = 64) :
  x = 4 :=
sorry

end power_function_x_value_l640_640482


namespace find_smallest_m_l640_640762

def is_in_S (z : ℂ) : Prop :=
  ∃ (x y : ℝ), ((1 / 2 : ℝ) ≤ x) ∧ (x ≤ Real.sqrt 2 / 2) ∧ (z = (x : ℂ) + (y : ℂ) * Complex.I)

def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def smallest_m (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, is_in_S z ∧ is_nth_root_of_unity z n

theorem find_smallest_m : smallest_m 24 :=
  sorry

end find_smallest_m_l640_640762


namespace number_of_trees_in_park_l640_640800

theorem number_of_trees_in_park :
  ∃ a : ℤ, 80 < a ∧ a < 150 ∧ 
  a % 4 = 2 ∧ 
  a % 5 = 3 ∧ 
  a % 6 = 4 ∧ 
  a = 98 :=
begin
  -- Proof will be constructed here
  sorry
end

end number_of_trees_in_park_l640_640800


namespace probability_two_points_one_unit_apart_l640_640131

theorem probability_two_points_one_unit_apart (H : Type)
  [hexagon : regular_hexagon H] :
  (probability (exists (p1 p2 : H), distance p1 p2 = 1)) = 2 / 5 := sorry

end probability_two_points_one_unit_apart_l640_640131


namespace quadratic_rewrite_l640_640029

theorem quadratic_rewrite (x : ℝ) (b c : ℝ) : 
  (x^2 + 1560 * x + 2400 = (x + b)^2 + c) → 
  c / b = -300 :=
by
  sorry

end quadratic_rewrite_l640_640029


namespace clients_number_l640_640854

theorem clients_number (C : ℕ) (total_cars : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ)
  (h1 : total_cars = 12)
  (h2 : cars_per_client = 4)
  (h3 : selections_per_car = 3)
  (h4 : C * cars_per_client = total_cars * selections_per_car) : C = 9 :=
by sorry

end clients_number_l640_640854


namespace angle_ACB_eq_120_NM_squared_plus_NL_squared_eq_ML_squared_l640_640902

-- Definition of the setup of the triangle and angle bisectors
variables (A B C L M N : Type*) [IsTriangle A B C]
          [IsAngleBisector L A C] [IsAngleBisector M B C] [IsAngleBisector N C A]

-- Given conditions
variables (h1 : ∠ ANM = ∠ ALC)

-- To be proved (1)
theorem angle_ACB_eq_120 : ∠ ACB = 120 := 
by {
  sorry 
}

-- To be proved (2)
theorem NM_squared_plus_NL_squared_eq_ML_squared (NM NL ML : ℝ) : NM^2 + NL^2 = ML^2 :=
by {
  sorry
}

end angle_ACB_eq_120_NM_squared_plus_NL_squared_eq_ML_squared_l640_640902


namespace sum_reciprocals_of_transformed_roots_l640_640160

noncomputable def cubic_poly : Polynomial ℝ := 40 * X^3 - 60 * X^2 + 26 * X - 1

-- Declare the roots from the polynomial
variables {a b c : ℝ}

-- Assume a, b, c are distinct roots of the polynomial and all lie within (0, 1)
axiom roots_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c
axiom roots_interval : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1
axiom roots_of_poly : cubic_poly.eval a = 0 ∧ cubic_poly.eval b = 0 ∧ cubic_poly.eval c = 0

-- Prove that the sum of the reciprocals of (1 - the roots) equals 1.5
theorem sum_reciprocals_of_transformed_roots : 
  \(\frac{1}{1-a} + \frac{1}{1-b} + \frac{1}{1-c} = 1.5\) :=
by
  sorry

end sum_reciprocals_of_transformed_roots_l640_640160


namespace range_of_f_l640_640614

def f (x : ℝ) : ℝ := (x^3 + 4 * x^2 + 5 * x + 2) / (x + 1)

theorem range_of_f : Set.range f = {y | 0 < y} := 
by sorry

end range_of_f_l640_640614


namespace common_area_of_rectangle_and_circle_l640_640557

-- Definition (Conditions)
def square_side_length : ℝ := 8
def rectangle_width : ℝ := 10
def rectangle_height : ℝ := 4

-- Derived data from conditions that would have been calculated in the proof
def circle_diameter : ℝ := square_side_length * real.sqrt 2
def circle_radius : ℝ := circle_diameter / 2

-- Statement (Question and expected answer)
theorem common_area_of_rectangle_and_circle : 
  ∀ (square_side_length rectangle_width rectangle_height: ℝ),
    square_side_length = 8 →
    rectangle_width = 10 →
    rectangle_height = 4 →
    (π * (circle_radius ^ 2) = 32 * π) → 
    ((rectangle_width * rectangle_height) = 32) := 
begin
  intros,
  sorry

end common_area_of_rectangle_and_circle_l640_640557


namespace tangent_line_at_e_extreme_values_range_of_a_l640_640233

-- Definition of the function f
def f (x a : ℝ) : ℝ := ((1 / 2) * x^2 + ax) * real.log x - (1 / 4) * x^2 - ax

-- Question 1: Prove that if a = 0, the equation of the tangent line at x = e is y = ex - (3/4)e^2
theorem tangent_line_at_e (x e : ℝ) (h_e : e > 0) : 
  ( ∃ (y : ℝ), y = exp x - (3 / 4) * e^2 ) ↔ (x = e ∧ a = 0) := sorry

-- Question 2: Prove that if a < 0, find the extreme values of the function
theorem extreme_values (a : ℝ) (h_neg : a < 0) : 
  ( ∀ (x : ℝ), (-1 < a ∧ a < 0 ∧ f(-a a) = (3 / 4) * a^2 - (1 / 2) * a^2 * real.log (-a))
             ∧ (f(1 a) = - (1 / 4) - a)
             ∧ (a = -1 → ¬ ∃ y, f y = 0)
             ∧ (a < -1 ∧ f(1 a) = - (1 / 4) - a ∧ f(-a a) = (3 / 4) * a^2 - (1 / 2) * a^2 * real.log (-a) ) ) := sorry

-- Question 3: Prove the range of values of a for which f(x) > 0 always holds
theorem range_of_a (a : ℝ) : 
  (∀ x, f x a > 0) ↔ a ∈ Set.Ioo (-real.exp (3 / 2) ) (-1 / 4) := sorry

end tangent_line_at_e_extreme_values_range_of_a_l640_640233


namespace DE_plus_FG_eq_3_l640_640063

-- Definitions: Equilateral triangle, side length, points on sides, parallel lines, equal perimeters
def is_equilateral_triangle (A B C : Point) (s : ℝ) : Prop :=
  (dist A B = s) ∧ (dist B C = s) ∧ (dist C A = s)

def points_on_side (A B : Point) (D E : Point) : Prop :=
  collinear A B D ∧ collinear A B E

def parallel_lines (DE FG BC : Line) : Prop :=
  parallel DE BC ∧ parallel FG BC

def equal_perimeters (ADE : Triangle) (DFGE : Trapezoid) (FBC : Triangle) : Prop :=
  perimeter ADE = perimeter DFGE ∧ perimeter DFGE = perimeter FBC

-- Proof statement
theorem DE_plus_FG_eq_3 (A B C D E F G : Point) (s : ℝ)
  (hABC : is_equilateral_triangle A B C s) (hAB_eq_2 : dist A B = 2)
  (hDE_on_AC : points_on_side A C D E) (hFG_on_AB : points_on_side A B F G)
  (hParallel : parallel_lines (line_through D E) (line_through F G) (line_through B C))
  (hEqualPerimeters : equal_perimeters (triangle A D E) (trapezoid D F G E) (triangle F B C)) :
  dist D E + dist F G = 3 :=
sorry

end DE_plus_FG_eq_3_l640_640063


namespace a_beats_b_by_26_meters_l640_640283

theorem a_beats_b_by_26_meters :
  (distance_A : ℕ) (distance_B : ℕ) (time_A : ℕ) (time_B : ℕ) 
  (distance: ℕ) 
  (h1 : distance_A = 130)
  (h2 : time_A = 20)
  (h3 : time_B = 25)
  (h4 : distance_B = (130 * time_A / time_B)) :
  distance_A - distance_B = 26 :=
begin
  sorry
end

end a_beats_b_by_26_meters_l640_640283


namespace triangle_XYZ_XWT_x_sum_l640_640502

def Point := (ℝ × ℝ)

noncomputable def area (A B C : Point) : ℝ :=
  0.5 * |(B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)|

noncomputable def line_distance (P : Point) (A B : Point) : ℝ :=
  (|(B.2 - A.2) * P.1 - (B.1 - A.1) * P.2 + (B.1 * A.2 - B.2 * A.1)|) / 
  (Math.sqrt ((B.2 - A.2)^2 + (B.1 - A.1)^2))

theorem triangle_XYZ_XWT_x_sum :
  ∃ X : Point, 
    area (0, 0) (169, 0) X = 1302 ∧
    area X (800, 400) (811, 411) = 5208 →
    let x_coords := [X.1 | x ∈  points satisfying above conditions] 
    List.sum x_coords = 1556 :=
by
  sorry

end triangle_XYZ_XWT_x_sum_l640_640502


namespace num_tangents_to_circles_l640_640771

noncomputable def point (α : Type) := (α × α)
noncomputable def distance {α : Type} [metric_space α] (a b : α) : ℝ := dist a b

noncomputable def num_tangents {α : Type} [metric_space α] (A B : point α) (rA rB : ℝ) : ℕ :=
  if distance A B = rA + rB then 3 else sorry

theorem num_tangents_to_circles
  {α : Type} [metric_space α] {A B : point α} : distance A B = 7 → num_tangents A B 3 4 = 3 :=
by sorry

end num_tangents_to_circles_l640_640771


namespace find_AC_l640_640276

open Real

noncomputable def AC (AB BC : ℝ) (B : ℝ) : ℝ :=
  sqrt (AB^2 + BC^2 - 2 * AB * BC * cos B)

theorem find_AC {A B C : ℝ} (B_angle : B = 120 * pi / 180)
  (BC_length : BC = 1)
  (area_ABC : (1 / 2) * A * BC * sin B = sqrt(3) / 2) :
  AC A BC B = sqrt(7) :=
by
  sorry

end find_AC_l640_640276


namespace find_range_of_m_l640_640215

variables (p q : Prop) (m : ℝ)

def prop_p (m : ℝ) : Prop :=
  (∃ (f : ℝ → ℝ), f = λ x, (1/3) * x ^ 3 + x ^ 2 + m * x + 1 ∧
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (f' x1 = 0 ∧ f' x2 = 0)))

def prop_q (m : ℝ) : Prop :=
  ∀ x ∈ set.Icc (-1 : ℝ) (2 : ℝ), 
  ∀ y ∈ set.Icc (-1 : ℝ) (2 : ℝ), x ≤ y → (f x ≥ f y)

theorem find_range_of_m (h_p : prop_p m) (h_not_q : ¬ prop_q m) : m < 1 :=
by {
  sorry
}

end find_range_of_m_l640_640215


namespace intersect_rectangular_eqn_range_of_m_l640_640355

-- Definitions of the parametric and polar equations
def curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Proving the rectangular equation and range of m for the intersection of l and C
theorem intersect_rectangular_eqn (m t : ℝ) :
  (∃ ρ θ, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  ∃ x y, (x, y) = curve_C t ∧ √3 * x + y + 2 * m = 0 :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ ρ θ t, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end intersect_rectangular_eqn_range_of_m_l640_640355


namespace find_a_l640_640979

def f (x : ℝ) : ℝ := Real.logBase (Real.sin 1) (x^2 - 6 * x + 5)

def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y, x ∈ I → y ∈ I → x < y → f x ≥ f y

theorem find_a (a : ℝ) :
  is_decreasing_on f (set.Ioi a) ↔ a ≥ 5 :=
sorry

end find_a_l640_640979


namespace soccer_league_games_l640_640493

theorem soccer_league_games (n_teams games_played : ℕ) (h1 : n_teams = 10) (h2 : games_played = 45) :
  ∃ k : ℕ, (n_teams * (n_teams - 1)) / 2 = games_played ∧ k = 1 :=
by
  sorry

end soccer_league_games_l640_640493


namespace find_median_of_set_l640_640799

variable (x y : ℕ)

def numbers : List ℕ := [88, 86, 81, 84, 85, x, y]

theorem find_median_of_set :
  (x + y = 171) →
  (List.mean [88, 86, 81, 84, 85, x, y] = 85) →
  List.median [88, 86, 81, 84, 85, x, y] = 86 :=
by
  sorry

end find_median_of_set_l640_640799


namespace flies_eaten_per_day_l640_640172

variable (flies_per_frog per_day frogs_per_fish per_day fish_per_gharial per_day gharials: ℕ)

-- Each frog needs to eat 30 flies per day to live.
def fliesPerFrog: ℕ := 30

-- Each fish needs to eat 8 frogs per day to live.
def frogsPerFish: ℕ := 8

-- Each gharial needs to eat 15 fish per day to live.
def fishPerGharial: ℕ := 15

-- The swamp has 9 gharials.
def gharials: ℕ := 9

theorem flies_eaten_per_day 
  (fliesPerFrog: ℕ) (frogsPerFish: ℕ) (fishPerGharial: ℕ) (gharials: ℕ)
  (h1: fliesPerFrog = 30)
  (h2: frogsPerFish = 8)
  (h3: fishPerGharial = 15)
  (h4: gharials = 9)
  : 9 * (15 * 8 * 30) = 32400 := by
  sorry

end flies_eaten_per_day_l640_640172


namespace problem_part1_problem_part2_l640_640303

noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l_rect_eq (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (sqrt 3 * x + y + 2 * m = 0)

def line_l_range_m (m : ℝ) : Prop :=
  -19/12 ≤ m ∧ m ≤ 5/2

theorem problem_part1 (m : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * sin (θ + π / 3) + m = 0) ↔ line_l_rect_eq m := 
sorry

theorem problem_part2 :
  ∀ m : ℝ, 
  (∃ t : ℝ, line_l_rect_eq m (curve_C t).1 (curve_C t).2) ↔ line_l_range_m m := 
sorry

end problem_part1_problem_part2_l640_640303


namespace max_elephants_l640_640437

def union_members : ℕ := 28
def non_union_members : ℕ := 37

/-- Given 28 union members and 37 non-union members, where elephants are distributed equally among
each group and each person initially receives at least one elephant, and considering 
the unique distribution constraint, the maximum number of elephants is 2072. -/
theorem max_elephants (n : ℕ) 
  (h1 : n % union_members = 0)
  (h2 : n % non_union_members = 0)
  (h3 : n ≥ union_members * non_union_members) :
  n = 2072 :=
by sorry

end max_elephants_l640_640437


namespace quadratic_inequality_solution_l640_640806

theorem quadratic_inequality_solution (x : ℝ) : 
  ((x - 1) * x ≥ 2) ↔ (x ≤ -1 ∨ x ≥ 2) := 
sorry

end quadratic_inequality_solution_l640_640806


namespace greatest_k_value_l640_640804

theorem greatest_k_value 
  (k : ℝ)
  (h : ∀ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = sqrt 85 ∨ x2 - x1 = sqrt 85) → x1^2 + k * x1 + 7 = 0 ∧ x2^2 + k * x2 + 7 = 0)
  : k = sqrt 113 :=
by
sorr

end greatest_k_value_l640_640804


namespace complete_collection_prob_l640_640048

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end complete_collection_prob_l640_640048


namespace diameter_of_larger_circle_approx_equal_l640_640598

-- Definitions of the given conditions
def small_circle_radius : ℝ := 2
def num_small_circles : ℕ := 8
def side_length_of_octagon : ℝ := 2 * small_circle_radius

-- Function to calculate the circumradius of a regular polygon
noncomputable def circumradius (s : ℝ) (n : ℕ) : ℝ := s / (2 * real.sin (real.pi / n))

-- Circumradius calculation for the octagon
noncomputable def radius_of_larger_circle : ℝ :=
  circumradius side_length_of_octagon num_small_circles + small_circle_radius

-- Calculation of the diameter of the larger circle
noncomputable def diameter_of_larger_circle : ℝ := 2 * radius_of_larger_circle

-- Problem statement in Lean 4
theorem diameter_of_larger_circle_approx_equal :
  diameter_of_larger_circle ≈ 14.4526 := sorry

end diameter_of_larger_circle_approx_equal_l640_640598


namespace volume_filled_space_l640_640586

-- Definitions using the given conditions
def radius : ℝ := 3
def height_cone : ℝ := 10

-- Define the volume of the cone using the given formula
def volume_cone : ℝ := (1/3) * Real.pi * (radius^2) * height_cone

-- Define the volume of the hemisphere using the given formula
def volume_hemisphere : ℝ := (2/3) * Real.pi * (radius^3)

-- Define the total volume as the sum of the volumes of the cone and the hemisphere
def total_volume : ℝ := volume_cone + volume_hemisphere

-- The theorem to be proven
theorem volume_filled_space : total_volume = 48 * Real.pi := by
  sorry

end volume_filled_space_l640_640586


namespace new_machine_rate_is_150_l640_640123

-- Defining constants and variables for the problem
def old_machine_rate : ℝ := 100
def time_in_minutes : ℝ := 108
def total_bolts : ℝ := 450
def new_machine_rate : ℝ := 150

-- Converting time from minutes to hours
def time_in_hours : ℝ := time_in_minutes / 60

-- Total bolts produced by both machines working together
def total_bolts_by_both_machines : ℝ := ((old_machine_rate + new_machine_rate) * time_in_hours)

-- Proof statement
theorem new_machine_rate_is_150 :
  (old_machine_rate * time_in_hours + new_machine_rate * time_in_hours = total_bolts)
  → (new_machine_rate = 150) :=
by
  sorry

end new_machine_rate_is_150_l640_640123


namespace round_fraction_to_two_decimal_places_l640_640774

theorem round_fraction_to_two_decimal_places :
  let frac := (7 : ℚ) / 9 in
  (Real.roundTo (frac : ℝ) 2 = 0.78) :=
by
  sorry

end round_fraction_to_two_decimal_places_l640_640774


namespace polar_to_rectangular_intersection_range_l640_640341

-- Definitions of parametric equations for curve C
def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line_l (ρ θ m: ℝ) : Prop := ρ * sin (θ + π / 3) + m = 0

-- Rectangular equation of line l
def rectangular_line_l (x y m: ℝ) : Prop := sqrt 3 * x + y + 2 * m = 0

-- Prove that polar equation of line l can be transformed to rectangular form
theorem polar_to_rectangular (ρ θ m : ℝ) :
  (polar_line_l ρ θ m) → (∃ x y, ρ = sqrt (x^2 + y^2) ∧ θ = arctan (y / x) ∧ rectangular_line_l x y m) :=
  sorry

-- Prove that the range of values for m ensuring line l intersects curve C is [-19/12, 5/2]
theorem intersection_range (m : ℝ) :
  (∃ t : ℝ, rectangular_line_l (sqrt 3 * cos (2 * t)) (2 * sin t) m) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
  sorry

end polar_to_rectangular_intersection_range_l640_640341


namespace parabola_equation_l640_640673

theorem parabola_equation (p x0 : ℝ) (h_p : p > 0) (h_dist_focus : x0 + p / 2 = 10) (h_parabola : 2 * p * x0 = 36) :
  (2 * p = 4) ∨ (2 * p = 36) :=
by sorry

end parabola_equation_l640_640673


namespace regular_polygon_identity_l640_640008

noncomputable def chord_length (n : ℕ) (i : ℕ) : ℝ :=
2 * real.sin (i * real.pi / n)

theorem regular_polygon_identity :
  let d1 := chord_length 15 1
  let d2 := chord_length 15 2
  let d4 := chord_length 15 4
  let d7 := chord_length 15 7
  in (1 / d2) + (1 / d4) + (1 / d7) = (1 / d1) := 
by
  sorry

end regular_polygon_identity_l640_640008


namespace solve_inequality_l640_640952

noncomputable def inequality_holds (a b : ℝ) :=
  ∀ (n : ℕ) (h : n > 2) (x : Fin n → ℝ), (∀ i, x i > 0) →
  ∑ i in (Finset.range n), (x i) * (x (Fin.rotate 1 i)) ≥
  ∑ i in (Finset.range n), (x i) ^ a * (x (Fin.rotate 1 i)) ^ b * (x (Fin.rotate 2 i)) ^ a

theorem solve_inequality :
  inequality_holds (1 / 2) 1 :=
by
  sorry

end solve_inequality_l640_640952


namespace interval_contains_fractions_l640_640486

theorem interval_contains_fractions
  (p q : ℕ)
  (h_coprime : Nat.coprime p q)
  (h_positive_p : 0 < p)
  (h_positive_q : 0 < q) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ (p + q - 2) →
  ∃ (i : ℕ), (i < p ∧ (k * (p + q) / i = 1 ∨ k * i / (p + q) = 1)) ∨ 
  ∃ (j : ℕ), (j < q ∧ (k * (p + q) / j = 1 ∨ k * j / (p + q) = 1)) :=
by
  sorry

end interval_contains_fractions_l640_640486


namespace T_12_mod_5_l640_640949

def T : ℕ → ℕ
| 0       := 1
| 1       := 2
| 2       := 4
| (n + 3) := T (n + 2) + T (n + 1) + T n

theorem T_12_mod_5 : T 12 % 5 = 4 := 
by
  sorry

end T_12_mod_5_l640_640949


namespace factorial_division_l640_640664

theorem factorial_division (h : 10! = 3628800) : 10! / 4! = 151200 := by
  sorry

end factorial_division_l640_640664


namespace diane_needs_more_money_l640_640168

-- Define the given conditions
def cost_of_cookies : ℕ := 65
def amount_diane_has : ℕ := 27

-- Define the question with the expected answer as a goal
theorem diane_needs_more_money : (cost_of_cookies - amount_diane_has) = 38 :=
by
  rw [cost_of_cookies, amount_diane_has]
  norm_num

end diane_needs_more_money_l640_640168


namespace initial_knives_l640_640578

theorem initial_knives (K T : ℕ)
  (h1 : T = 2 * K)
  (h2 : K + T + (1 / 3 : ℚ) * K + (2 / 3 : ℚ) * T = 112) : 
  K = 24 :=
by
  sorry

end initial_knives_l640_640578


namespace total_jelly_beans_l640_640420

-- Given conditions:
def vanilla_jelly_beans : ℕ := 120
def grape_jelly_beans := 5 * vanilla_jelly_beans + 50

-- The proof problem:
theorem total_jelly_beans :
  vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  -- Proof steps would go here
  sorry

end total_jelly_beans_l640_640420


namespace ellipse_constant_sum_l640_640212

/-- Given an ellipse ∂(x²/4 + y² = 1) with eccentricity sqrt(3)/2, point P=(-1,0), 
 and point A on the circle x² + y² = 1 and line through P perpendicular to PA
 intersects circles at B and C, prove that |BC|² + |CA|² + |AB|² is a constant value 26. -/
theorem ellipse_constant_sum {x y : ℝ} (h₁ : x^2 / 4 + y^2 = 1) (h₂ : sqrt (3) / 2 = 1)
  (P : ℝ × ℝ) (hP : P = (-1, 0)) (A B C : ℝ × ℝ) (hA : A.1^2 + A.2^2 = 1) 
  (hB : B.1^2 + B.2^2 = 4) (hC : C.1^2 + C.2^2 = 4)
  (h_perpendicular : (A.1 + 1) * (B.1 + 1) + A.2 * B.2 = 0) :
  (dist B C)^2 + (dist C A)^2 + (dist A B)^2 = 26 :=
by
  sorry

end ellipse_constant_sum_l640_640212


namespace radius_of_new_sphere_l640_640462

def clay_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

def drilled_hole (R : ℝ) (r : ℝ) : ℝ := clay_sphere R - clay_sphere r

def remaining_volume (R : ℝ) (r : ℝ) : ℝ := drilled_hole R r

def new_sphere_radius (V_remaining : ℝ) : ℝ :=
  let V_sphere_radius := λ r : ℝ, clay_sphere r
  Classical.some (exists_unique_of_exists_real (λ r, V_sphere_radius r = V_remaining))

theorem radius_of_new_sphere (R r : ℝ) (h: R = 13) (hr : r = 5) : new_sphere_radius (remaining_volume R r) = 12 := 
by
  have := remaining_volume 13 5
  -- Additional required mathematical assertions and relevant properties go here
  sorry

end radius_of_new_sphere_l640_640462


namespace avg_weight_BC_l640_640470

variable (A B C : ℝ)

def totalWeight_ABC := 3 * 45
def totalWeight_AB := 2 * 40
def weight_B := 31

theorem avg_weight_BC : ((B + C) / 2) = 43 :=
  by
    have totalWeight_ABC_eq : A + B + C = totalWeight_ABC := by sorry
    have totalWeight_AB_eq : A + B = totalWeight_AB := by sorry
    have weight_B_eq : B = weight_B := by sorry
    sorry

end avg_weight_BC_l640_640470


namespace reconstruct_triangle_from_altitudes_feet_l640_640434

theorem reconstruct_triangle_from_altitudes_feet :
  ∀ (A B C A₁ B₁ C₁ : Point) (hA₁ : altitude_foot A B C = A₁) (hB₁ : altitude_foot B A C = B₁) (hC₁ : altitude_foot C A B = C₁),
    is_acute_triangle A B C →
    ∃ (A' B' C' : Point),
      triangle A B C' ∧
      altitude_foot A' B' C' = A₁ ∧
      altitude_foot B' A' C' = B₁ ∧
      altitude_foot C' A' B' = C₁ :=
by simp [reconstruct_triangle]

end reconstruct_triangle_from_altitudes_feet_l640_640434


namespace complete_collection_probability_l640_640054

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end complete_collection_probability_l640_640054


namespace rectangular_equation_common_points_l640_640324

-- Define parametric equations of curve C
def x (t : ℝ) := (sqrt 3) * cos (2 * t)
def y (t : ℝ) := 2 * sin t

-- Define polar equation of line l
def polar_line (ρ θ m : ℝ) := ρ * sin (θ + π / 3) + m

-- Rectangular equation of l
theorem rectangular_equation (ρ θ x y m : ℝ) (h1 : ρ = sqrt (x^2 + y^2)) (h2 : θ = atan2 y x) :
  polar_line ρ θ m = 0 → (sqrt 3 * x + y + 2 * m = 0) :=
by 
  sorry

-- Range of values for m where line intersects curve
theorem common_points (h: ∃ t, sqrt 3 * cos (2 * t) = x ∧ 2 * sin t = y) :
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by 
  sorry

end rectangular_equation_common_points_l640_640324


namespace weight_of_second_block_l640_640108

variables {d : Type} [uniform_density d]
variables (t : ℝ) (s1 w1 s2 : ℝ)

def side_length1 : ℝ := 4
def thickness : ℝ := 0.5
def weight1 : ℝ := 20
def side_length2 : ℝ := 6

theorem weight_of_second_block :
  ∃ (w2 : ℝ), w2 = 45 :=
by {
  sorry
}

end weight_of_second_block_l640_640108


namespace hyperbola_equation_theorem_l640_640237

noncomputable def hyperbola_equation (a b : ℝ) (eq : (a > 0) ∧ (b > 0) ∧ ((a = b) ∧ (4^2 = a^2 + b^2)) ) : 
  (real) := ( (a^2) = 8 ) ∧ ( (b^2) = 8 ) 

theorem hyperbola_equation_theorem {a b : ℝ} (h : (a > 0) ∧ (b > 0))
  (h_focus : (4^2 = a^2 + b^2)) (h_asymptotes : (a = b)) :
  ((a^2 = 8) ∧ (b^2 = 8)) :=
begin
  sorry,
end

end hyperbola_equation_theorem_l640_640237


namespace batch_of_pizza_dough_makes_three_pizzas_l640_640380

theorem batch_of_pizza_dough_makes_three_pizzas
  (pizza_dough_time : ℕ)
  (baking_time : ℕ)
  (total_time_minutes : ℕ)
  (oven_capacity : ℕ)
  (total_pizzas : ℕ) 
  (number_of_batches : ℕ)
  (one_batch_pizzas : ℕ) :
  pizza_dough_time = 30 →
  baking_time = 30 →
  total_time_minutes = 300 →
  oven_capacity = 2 →
  total_pizzas = 12 →
  total_time_minutes = total_pizzas / oven_capacity * baking_time + number_of_batches * pizza_dough_time →
  number_of_batches = total_time_minutes / 30 →
  one_batch_pizzas = total_pizzas / number_of_batches →
  one_batch_pizzas = 3 :=
by
  intros
  sorry

end batch_of_pizza_dough_makes_three_pizzas_l640_640380


namespace part_I_b1_part_I_b2_part_I_general_part_II_exists_l640_640417

open Nat

def a (p q n : ℕ) : ℕ := p * n + q

def b (p q m : ℕ) : ℕ := Inf {n : ℕ | a p q n ≥ m}

-- Part (I)
theorem part_I_b1 {p : ℕ} (h : p = 2) {q : ℤ} (h2 : q = -1) : b 2 (-1) 1 = 1 := 
by sorry

theorem part_I_b2 {p : ℕ} (h : p = 2) {q : ℤ} (h2 : q = -1) : b 2 (-1) 2 = 2 := 
by sorry

theorem part_I_general (h_p : p = 2) (h_q : q = -1) (m k : ℕ) (h : m = 2 * k ∨ m = 2 * k - 1) :
  b 2 (-1) m = if m % 2 = 1 then (m + 1) / 2 else m / 2 := 
by sorry

-- Part (II)
theorem part_II_exists :
  ∃ (p q : ℕ), p > 0 ∧ q < 5 ∧ ∀ m : ℕ, b p q m = 3 * m + 2 := 
by sorry

end part_I_b1_part_I_b2_part_I_general_part_II_exists_l640_640417


namespace median_of_first_twelve_positive_integers_l640_640514

theorem median_of_first_twelve_positive_integers :
  let S := (set.range 12).image (λ x, x + 1) in  -- Set of first twelve positive integers
  median S = 6.5 :=
by
  sorry

end median_of_first_twelve_positive_integers_l640_640514


namespace mary_cut_roses_l640_640819

-- Definitions from conditions
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- The theorem to prove
theorem mary_cut_roses : (final_roses - initial_roses) = 10 :=
by
  sorry

end mary_cut_roses_l640_640819


namespace slope_of_line_passing_through_MN_l640_640596

theorem slope_of_line_passing_through_MN :
  let M := (-2, 1)
  let N := (1, 4)
  ∃ m : ℝ, m = (N.2 - M.2) / (N.1 - M.1) ∧ m = 1 :=
by
  sorry

end slope_of_line_passing_through_MN_l640_640596


namespace third_largest_three_digit_number_with_tens_digit_5_l640_640269

theorem third_largest_three_digit_number_with_tens_digit_5 
  (digits : Set ℕ)
  (tens_digit : ℕ) 
  (third_largest : ℕ) : 
  digits = {1, 5, 9, 4} →
  tens_digit = 5 →
  ∃ hundreds units : ℕ, 
    hundreds ∈ digits \ {tens_digit} ∧ 
    units ∈ digits \ {tens_digit, hundreds} ∧ 
    third_largest = 100 * hundreds + 10 * tens_digit + units ∧ 
    (∃ n1 n2 n3 : ℕ, 
      (n1 > n2 ∧ n2 > n3 ∧ n3 = third_largest) ∧ 
      (n1 = 100 * (digits \ {tens_digit, units}).max! + 10 * tens_digit + units.max!) ∧ 
      (n2 = 100 * (digits \ {tens_digit, units \ {units.max!}}).max! + 10 * tens_digit + (units \ {units.max!}).max!)) := 
begin
  sorry
end

end third_largest_three_digit_number_with_tens_digit_5_l640_640269


namespace proof_standard_eq_and_P_PA_PB_value_l640_640600

noncomputable def standard_eq_curve_C (x y : ℝ) : Prop :=
  (x^2 / 5 + y^2 = 1)

noncomputable def cartesian_eq_line_l (x y : ℝ) : Prop :=
  (y = x - 2)

noncomputable def PA_PB_value (A B : ℝ) : ℝ :=
  abs A + abs B

open Real

theorem proof_standard_eq_and_P_PA_PB_value (x y : ℝ) (α : ℝ) (P : ℝ × ℝ) (A B : ℝ) :
  (x = sqrt 5 * cos α) ∧ (y = sin α) ∧ (P = (0, -2)) ∧ (sqrt (x^2 + y^2) * cos (atan2 y x + π/4) = sqrt 2) →
  (standard_eq_curve_C x y) ∧ (cartesian_eq_line_l x y) ∧
  PA_PB_value A B = (10 * sqrt 2 / 3) :=
by
  intros
  sorry

end proof_standard_eq_and_P_PA_PB_value_l640_640600


namespace intersect_rectangular_eqn_range_of_m_l640_640353

-- Definitions of the parametric and polar equations
def curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Proving the rectangular equation and range of m for the intersection of l and C
theorem intersect_rectangular_eqn (m t : ℝ) :
  (∃ ρ θ, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  ∃ x y, (x, y) = curve_C t ∧ √3 * x + y + 2 * m = 0 :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ ρ θ t, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end intersect_rectangular_eqn_range_of_m_l640_640353


namespace volume_fraction_of_remaining_solid_l640_640881

noncomputable def original_pyramid_base_edge := 20 -- cm
noncomputable def original_pyramid_height := 40 -- cm
noncomputable def smaller_pyramid_factor1 := 1 / 3
noncomputable def smaller_pyramid_factor2 := 1 / 5

theorem volume_fraction_of_remaining_solid (V : ℝ) :
  let original_volume := (original_pyramid_base_edge ^ 2 * original_pyramid_height) / 3
  let smaller_volume1 := (smaller_pyramid_factor1 ^ 3) * original_volume
  let smaller_volume2 := (smaller_pyramid_factor2 ^ 3) * original_volume
  let total_cut_volume := smaller_volume1 + smaller_volume2
  let remaining_volume_fraction := (original_volume - total_cut_volume) / original_volume
  in remaining_volume_fraction = (3223 / 3375) :=
by
  sorry

end volume_fraction_of_remaining_solid_l640_640881


namespace f_log2_20_l640_640551

noncomputable def f : ℝ → ℝ :=
  sorry -- A definition of f would fit here with respect to given properties

theorem f_log2_20 : f (Real.log2 20) = -(4 / 5) :=
by
  -- Assuming f is odd: for all x in ℝ, f(-x) = -f(x)
  sorry
  -- Given condition f(x + 2) = -f(x)
  sorry
  -- Given f(x) = 2^x for x in [-1, 0)
  sorry
  -- Using the log properties and the definition to prove the equality
  sorry

end f_log2_20_l640_640551


namespace triangle_inequality_l640_640393

variables {A B C I P : Type}
variables [Geometry A] [IsIncenter I A B C] [IsPointInside P A B C]

noncomputable def angles_sum_condition (P : Point) : Prop :=
  ∠PBC + ∠PCB = ∠PCA + ∠PBA

theorem triangle_inequality (h₁ : IsTriangle A B C) 
                           (h₂ : IsIncenter I A B C)
                           (h₃ : IsPointInside P A B C)
                           (h₄ : angles_sum_condition P):
  length (segment AP) ≥ length (segment AI) := 
sorry

end triangle_inequality_l640_640393


namespace largest_n_divisible_by_every_integer_less_than_cube_root_l640_640835

theorem largest_n_divisible_by_every_integer_less_than_cube_root :
  ∃ n : ℕ, (∀ m : ℕ, (m < (n ^ (1/3 : ℝ)).ceil) → m ∣ n) ∧ (n = 420) :=
by
  sorry

end largest_n_divisible_by_every_integer_less_than_cube_root_l640_640835


namespace beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l640_640950

def beautiful_association_number (x y a t : ℚ) : Prop :=
  |x - a| + |y - a| = t

theorem beautiful_association_number_part1 :
  beautiful_association_number (-3) 5 2 8 :=
by sorry

theorem beautiful_association_number_part2 (x : ℚ) :
  beautiful_association_number x 2 3 4 ↔ x = 6 ∨ x = 0 :=
by sorry

theorem beautiful_association_number_part3 (x0 x1 x2 x3 x4 : ℚ) :
  beautiful_association_number x0 x1 1 1 ∧ 
  beautiful_association_number x1 x2 2 1 ∧ 
  beautiful_association_number x2 x3 3 1 ∧ 
  beautiful_association_number x3 x4 4 1 →
  x1 + x2 + x3 + x4 = 10 :=
by sorry

end beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l640_640950


namespace Eldora_total_cost_l640_640174

-- Conditions
def paper_clip_cost : ℝ := 1.85
def index_card_cost : ℝ := 3.95 -- from Finn's purchase calculation
def total_cost (clips : ℝ) (cards : ℝ) (clip_price : ℝ) (card_price : ℝ) : ℝ :=
  (clips * clip_price) + (cards * card_price)

theorem Eldora_total_cost :
  total_cost 15 7 paper_clip_cost index_card_cost = 55.40 :=
by
  sorry

end Eldora_total_cost_l640_640174


namespace expand_expression_l640_640932

variable (x y : ℝ)

theorem expand_expression :
  12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 :=
by
  sorry

end expand_expression_l640_640932


namespace area_of_EFCD_l640_640720

-- Define the trapezoid and its properties
structure Trapezoid where
  AB : ℝ
  CD : ℝ 
  altitude : ℝ
  (parallel : AB > 0 ∧ CD > 0 ∧ altitude > 0)

-- Define points E and F and their properties of dividing the sides in the ratio 2:3
structure DivisionPoints where
  AD_ratio : ℝ
  BC_ratio : ℝ 
  (ratios : AD_ratio = 2 / 5 ∧ BC_ratio = 3 / 5)

-- Define the problem conditions
def trapezoid_ABCD : Trapezoid := 
  { AB := 10, CD := 24, altitude := 15, parallel := ⟨by linarith, by linarith, by linarith⟩ }

def division_points_EF : DivisionPoints :=
  { AD_ratio := 2 / 5, BC_ratio := 3 / 5, ratios := ⟨rfl, rfl⟩ }

-- Goal: Prove the area of quadrilateral EFCD
theorem area_of_EFCD : 
  let EF := ((2 / 5) * trapezoid_ABCD.AB + (3 / 5) * trapezoid_ABCD.CD)
  let new_altitude := (3 / 5) * trapezoid_ABCD.altitude
  let area := new_altitude * ((EF + trapezoid_ABCD.CD) / 2)
  area = 190.8 :=
by sorry

end area_of_EFCD_l640_640720


namespace find_a5_l640_640693

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 0 then 2
  else if n > 0 
  then sequence (n - 1) - 1 / sequence (n - 1)
  else 0

theorem find_a5 : sequence 5 = 1/2 := 
sorry

end find_a5_l640_640693


namespace annual_average_growth_rate_l640_640022

theorem annual_average_growth_rate (p q : ℝ) (hp : 0 ≤ p) (hq : 0 ≤ q) : 
  let x := real.sqrt ((1 + p) * (1 + q)) - 1 in
  (1 + x)^2 = (1 + p) * (1 + q) := by
  let x := real.sqrt ((1 + p) * (1 + q)) - 1
  sorry

end annual_average_growth_rate_l640_640022


namespace appropriate_expression_l640_640142

theorem appropriate_expression :
  let sentence_A := "At the end of the letter, the company's HR director respectfully wrote: 'For the position of manager, I humbly ask for your consideration.'"
  let sentence_B := "My wife and Professor Guo's wife have been close friends for many years; they often go shopping and travel together, seemingly having endless conversations."
  let sentence_C := "A student said at the class's study method exchange meeting: 'I earnestly hope that students with ineffective methods adjust their mindset, improve their methods, and achieve excellent results.'"
  let sentence_D := "The Ministry of Foreign Affairs stated in a newspaper article today: 'The Chinese government has always advocated for resolving regional conflicts peacefully, rather than resorting to violence at the slightest disagreement.'"
  (is_expressed_appropriately sentence_A) ∧
  (¬ is_expressed_appropriately sentence_B) ∧
  (¬ is_expressed_appropriately sentence_C) ∧
  (¬ is_expressed_appropriately sentence_D) :=
by
  sorry

end appropriate_expression_l640_640142


namespace necessary_but_not_sufficient_condition_for_tangency_l640_640031

-- Define the concepts of a line, hyperbola, and tangency in a mathematical structure
structure Point := (x : ℝ) (y : ℝ)

structure Line := (p1 p2 : Point) -- A line is defined by two points

structure Hyperbola := (center : Point) (a b : ℝ) -- Define hyperbola parameters

-- Definition of a line intersecting a hyperbola at one and only one point.
def line_has_one_point_in_common_with_hyperbola (l : Line) (h : Hyperbola) : Prop :=
  ∃! p : Point, (p lies on l) ∧ (p lies on h)

-- Definition of a line being tangent to a hyperbola.
def line_is_tangent_to_hyperbola (l : Line) (h : Hyperbola) : Prop :=
  ∃! p : Point, (p lies on l) ∧ (p lies on h) ∧ ∀ q, q ≠ p → ¬ (q lies on l ∧ q lies on h)

-- The proof problem: Prove that having one point in common is a necessary but not sufficient condition for tangency.
theorem necessary_but_not_sufficient_condition_for_tangency (l : Line) (h : Hyperbola) :
  (line_has_one_point_in_common_with_hyperbola l h) ↔ (line_is_tangent_to_hyperbola l h) :=
sorry

end necessary_but_not_sufficient_condition_for_tangency_l640_640031


namespace product_not_odd_in_17_gon_l640_640374

theorem product_not_odd_in_17_gon (a : Fin 17 → ℤ) (h_distinct : Function.Injective a) :
  let b (i : Fin 17) := a (i + 1) - a (i + 2)
  ¬Odd (∏ i, b i) := by
  sorry

end product_not_odd_in_17_gon_l640_640374


namespace total_jelly_beans_l640_640421

-- Given conditions:
def vanilla_jelly_beans : ℕ := 120
def grape_jelly_beans := 5 * vanilla_jelly_beans + 50

-- The proof problem:
theorem total_jelly_beans :
  vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  -- Proof steps would go here
  sorry

end total_jelly_beans_l640_640421


namespace correct_factorization_l640_640529

theorem correct_factorization (x m n a : ℝ) : 
  (¬ (x^2 + 2 * x + 1 = x * (x + 2) + 1)) ∧
  (¬ (m^2 - 2 * m * n + n^2 = (m + n)^2)) ∧
  (¬ (-a^4 + 16 = -(a^2 + 4) * (a^2 - 4))) ∧
  (x^3 - 4 * x = x * (x + 2) * (x - 2)) :=
by
  sorry

end correct_factorization_l640_640529


namespace flour_already_put_in_l640_640768

-- Defining the conditions given in the problem
def recipe_flour : ℕ := 7
def recipe_sugar : ℕ := 3
def flour_needed_to_add : ℕ := 5

-- The proof problem statement
theorem flour_already_put_in : ∃ (x : ℕ), x + flour_needed_to_add = recipe_flour ∧ x = 2 :=
by
  use 2
  constructor
  sorry

end flour_already_put_in_l640_640768


namespace range_of_a_monotonic_l640_640985

theorem range_of_a_monotonic (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 + 2 * a * x - 1 ≤ 0) ↔ (-real.sqrt 3 ≤ a ∧ a ≤ real.sqrt 3) :=
by 
  sorry

end range_of_a_monotonic_l640_640985


namespace polynomial_value_at_minus_2_l640_640082

variable (a b : ℝ)

def polynomial (x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem polynomial_value_at_minus_2 :
  (polynomial a b (-2) = -21) :=
  sorry

end polynomial_value_at_minus_2_l640_640082


namespace factorial_division_l640_640665

theorem factorial_division : (10! / 4! = 151200) :=
by
  have fact_10 : 10! = 3628800 := by sorry
  rw [fact_10]
  -- Proceeding with calculations and assumptions that follow directly from the conditions
  have fact_4 : 4! = 24 := by sorry
  rw [fact_4]
  exact (by norm_num : 3628800 / 24 = 151200)

end factorial_division_l640_640665


namespace arithmetic_seq_common_diff_l640_640652

theorem arithmetic_seq_common_diff
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_d_nonzero : d ≠ 0)
  (h_a1 : a 1 = 1)
  (h_geomet : (a 3) ^ 2 = a 1 * a 13) :
  d = 2 :=
by
  sorry

end arithmetic_seq_common_diff_l640_640652


namespace right_triangle_with_inscribed_circle_l640_640023

-- Define the problem conditions and the result
theorem right_triangle_with_inscribed_circle (k : ℝ) (h : k > 0) :
  (∀ α : ℝ, 
    α = (π / 4 - arcsin ((√2 * (k - 1)) / (2 * (k + 1)))) ∨
    α = (π / 4 + arcsin ((√2 * (k - 1)) / (2 * (k + 1)))))
  :=
sorry

end right_triangle_with_inscribed_circle_l640_640023


namespace number_of_ants_in_section_correct_l640_640127

noncomputable def ants_in_section := 
  let width_feet : ℝ := 600
  let length_feet : ℝ := 800
  let ants_per_square_inch : ℝ := 5
  let side_feet : ℝ := 200
  let feet_to_inches : ℝ := 12
  let side_inches := side_feet * feet_to_inches
  let area_section_square_inches := side_inches^2
  ants_per_square_inch * area_section_square_inches

theorem number_of_ants_in_section_correct :
  ants_in_section = 28800000 := 
by 
  unfold ants_in_section 
  sorry

end number_of_ants_in_section_correct_l640_640127


namespace gain_percent_l640_640114

noncomputable def original_price : ℝ := 1285
noncomputable def discount_rate : ℝ := 0.18
noncomputable def refurbishing_cost : ℝ := 365
noncomputable def selling_price : ℝ := 2175

theorem gain_percent :
  let discount_amount := discount_rate * original_price in
  let discounted_price := original_price - discount_amount in
  let total_cost_price := discounted_price + refurbishing_cost in
  let gain := selling_price - total_cost_price in
  let gain_percent := (gain / total_cost_price) * 100 in
  gain_percent = 53.3 :=
by {
  -- The actual proof will go here, we'll skip it for now as only the statement is needed.
  sorry
}

end gain_percent_l640_640114


namespace sin_cos_quotient_l640_640959

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f_prime (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem sin_cos_quotient 
  (x : ℝ)
  (h : f_prime x = 3 * f x) 
  : (Real.sin x ^ 2 - 3) / (Real.cos x ^ 2 + 1) = -14 / 9 := 
by 
  sorry

end sin_cos_quotient_l640_640959


namespace terminating_decimal_values_l640_640196

theorem terminating_decimal_values : 
  let terminates (n : ℕ) : Prop :=
    ∃ k : ℕ, n = 77 * k ∧ 1 ≤ n ∧ n ≤ 539
  (count n, 1 ≤ n ∧ n ≤ 539 ∧ terminates n) = 7 :=
by
  sorry

end terminating_decimal_values_l640_640196


namespace inequality_sin_values_l640_640928

theorem inequality_sin_values :
  let a := Real.sin (-5)
  let b := Real.sin 3
  let c := Real.sin 5
  a > b ∧ b > c :=
by
  sorry

end inequality_sin_values_l640_640928


namespace polar_to_rectangular_intersection_range_l640_640340

-- Definitions of parametric equations for curve C
def curve_C (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line_l (ρ θ m: ℝ) : Prop := ρ * sin (θ + π / 3) + m = 0

-- Rectangular equation of line l
def rectangular_line_l (x y m: ℝ) : Prop := sqrt 3 * x + y + 2 * m = 0

-- Prove that polar equation of line l can be transformed to rectangular form
theorem polar_to_rectangular (ρ θ m : ℝ) :
  (polar_line_l ρ θ m) → (∃ x y, ρ = sqrt (x^2 + y^2) ∧ θ = arctan (y / x) ∧ rectangular_line_l x y m) :=
  sorry

-- Prove that the range of values for m ensuring line l intersects curve C is [-19/12, 5/2]
theorem intersection_range (m : ℝ) :
  (∃ t : ℝ, rectangular_line_l (sqrt 3 * cos (2 * t)) (2 * sin t) m) ↔ (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
  sorry

end polar_to_rectangular_intersection_range_l640_640340


namespace area_T_is_34_l640_640120

/-- Define the dimensions of the large rectangle -/
def width_rect : ℕ := 10
def height_rect : ℕ := 4
/-- Define the dimensions of the removed section -/
def width_removed : ℕ := 6
def height_removed : ℕ := 1

/-- Calculate the area of the large rectangle -/
def area_rect : ℕ := width_rect * height_rect

/-- Calculate the area of the removed section -/
def area_removed : ℕ := width_removed * height_removed

/-- Calculate the area of the "T" shape -/
def area_T : ℕ := area_rect - area_removed

/-- To prove that the area of the T-shape is 34 square units -/
theorem area_T_is_34 : area_T = 34 := 
by {
  sorry
}

end area_T_is_34_l640_640120


namespace cone_volume_l640_640876

theorem cone_volume (V_f : ℝ) (A1 A2 : ℝ) (V : ℝ)
  (h1 : V_f = 78)
  (h2 : A1 = 9 * A2) :
  V = 81 :=
sorry

end cone_volume_l640_640876


namespace flies_eaten_per_day_l640_640170

def frogs_needed (fish : ℕ) : ℕ := fish * 8
def flies_needed (frogs : ℕ) : ℕ := frogs * 30
def fish_needed (gharials : ℕ) : ℕ := gharials * 15

theorem flies_eaten_per_day (gharials : ℕ) : flies_needed (frogs_needed (fish_needed gharials)) = 32400 :=
by
  have h1 : fish_needed 9 = 135 := rfl
  have h2 : frogs_needed 135 = 1080 := rfl
  have h3 : flies_needed 1080 = 32400 := rfl
  exact Eq.trans (Eq.trans h1.symm (Eq.trans h2.symm h3.symm)) sorry

end flies_eaten_per_day_l640_640170


namespace odd_function_neg_value_l640_640487

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x + 3 else -(x^2 - 2 * x + 3)

theorem odd_function_neg_value :
  f(-3) = -6 :=
by
  -- The actual proof goes here, but we'll use sorry to skip it.
  sorry

end odd_function_neg_value_l640_640487


namespace sum_of_valid_h_values_l640_640940

theorem sum_of_valid_h_values :
  let f r h := |r + h| - 2 * |r| - 3 * r - 7 * |r - 1|
  in (∀ h : ℤ, f 1 h = 0 → (f 0 h ≠ 0 ∧ ∀ r : ℝ, r ≠ 1 → f r h ≠ 0)) →
     ∑ k in Icc (-6) 4, k = -11 :=
by
  -- Define the function
  let f (r h : ℝ) := |r + h| - 2 * |r| - 3 * r - 7 * |r - 1|

  -- Set up the conditions for the equation to have at most one solution
  assume H : ∀ h : ℤ, f 1 h = 0 → (f 0 h ≠ 0 ∧ ∀ r : ℝ, r ≠ 1 → f r h ≠ 0)

  -- Calculate the sum of the valid integer values of h
  have h_values : Icc (-6) 4 = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4},
  from by { reflexivity }

  -- Compute the sum of these values
  let S := ∑ k in h_values, k

  -- Check the expected value
  have sum_check : S = -11,
  by { sorry }

  -- Complete the proof
  exact sum_check

end sum_of_valid_h_values_l640_640940


namespace rearrange_digits_perfect_square_l640_640040

theorem rearrange_digits_perfect_square :
  ∃ (n : ℕ), (nat.sqrt (2 * 1000 + 0 * 100 + 1 * 10 + 4) = nat.sqrt 2401) ∧
  (2401 = 49 * 49) ∧
  0 ≤ 2 ∧ 2 < 10 ∧
  0 ≤ 0 ∧ 0 < 10 ∧
  0 ≤ 1 ∧ 1 < 10 ∧
  0 ≤ 4 ∧ 4 < 10 :=
sorry

end rearrange_digits_perfect_square_l640_640040


namespace exterior_angle_measure_l640_640809

theorem exterior_angle_measure (h : ∑ i in (finset.range n), interior_angle i = 720) : 
  ∃ k : ℕ, measure_exterior_angle k = 60 :=
by
  sorry

end exterior_angle_measure_l640_640809


namespace titu_andreescu_quadrilateral_inequality_l640_640543

theorem titu_andreescu_quadrilateral_inequality
  (AB AD BC CD: ℝ)
  (circumscribed : circumscribed_quadrilateral AB AD BC CD)
  (angle_condition : angle_constraints AB AD BC CD) : 
  (1/3) * abs(AB^3 - AD^3) ≤ abs(BC^3 - CD^3) ∧ abs(BC^3 - CD^3) ≤ 3 * abs(AB^3 - AD^3) := 
sorry

-- These definitions would be elaborated based on the exact mathematical considerations of circumscribed quadrilateral and angle conditions.
-- For example:
def circumscribed_quadrilateral (AB AD BC CD: ℝ) : Prop := 
  AB + CD = BC + AD

def angle_constraints (AB AD BC CD: ℝ) : Prop := 
  ∀ θ, (60 ≤ θ ∧ θ ≤ 120)

end titu_andreescu_quadrilateral_inequality_l640_640543


namespace sticker_probability_l640_640061

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end sticker_probability_l640_640061


namespace tori_needs_5_more_problems_l640_640501

noncomputable def num_arithmetic_problems := 10
noncomputable def num_algebra_problems := 30
noncomputable def num_geometry_problems := 35
noncomputable def total_problems := num_arithmetic_problems + num_algebra_problems + num_geometry_problems

noncomputable def percent_pass := 0.6
noncomputable def num_correct_to_pass := total_problems * percent_pass

noncomputable def percent_arithmetic_correct := 0.7
noncomputable def num_correct_arithmetic := percent_arithmetic_correct * num_arithmetic_problems

noncomputable def percent_algebra_correct := 0.4
noncomputable def num_correct_algebra := percent_algebra_correct * num_algebra_problems

noncomputable def percent_geometry_correct := 0.6
noncomputable def num_correct_geometry := percent_geometry_correct * num_geometry_problems

noncomputable def total_correct := num_correct_arithmetic + num_correct_algebra + num_correct_geometry

noncomputable def num_more_correct_needed := num_correct_to_pass - total_correct

theorem tori_needs_5_more_problems :
  num_more_correct_needed = 5 := by
  sorry

end tori_needs_5_more_problems_l640_640501


namespace xiao_hong_spent_l640_640852

theorem xiao_hong_spent (original_price : ℝ) (discount_rate : ℝ) (result : ℝ) : 
  original_price = 260 ∧ discount_rate = 0.30 → result = 260 * (1 - discount_rate) → result = 182 :=
by
  intros h1 h2
  cases h1 with h_op h_dr
  rw [h_op, h_dr] at h2
  linarith

end xiao_hong_spent_l640_640852


namespace part_i_part_ii_part_iii_l640_640231

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  e^x / (a * x^2 + b * x + 1)

theorem part_i (a b : ℝ) (h₁ : a = 1) (h₂ : b = 1) (x : ℝ) :
  (f x 1 1) isIncreasingOn Ioo (-∞ : ℝ) 0 ∧ (f x 1 1) isIncreasingOn Ioo 1 ∞ ∧
  f x 1 1 isDecreasingOn Ioo 0 1 := sorry

theorem part_ii (a : ℝ) (h₁ : a = 0) (h₂ : ∀ x, x ≥ 0 → f x 0 b ≥ 1) :
  (0 ≤ b) ∧ (b ≤ 1) := sorry

theorem part_iii (a : ℝ) (b : ℝ) (h₁ : a > 0) (h₂ : b = 0) (x₁ x₂ : ℝ) (h₃ : isExtremal x₁ ∧ isExtremal x₂) :
  (f x₁ a 0) + (f x₂ a 0) < e := sorry

end part_i_part_ii_part_iii_l640_640231


namespace min_f_value_l640_640208

-- Define the function f
def f (x y z : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + 256 * y * z) / (y^2 + z^2))) + 
  (Real.sqrt ((y^2 + 256 * z * x) / (z^2 + x^2))) + 
  (Real.sqrt ((z^2 + 256 * x * y) / (x^2 + y^2)))

theorem min_f_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_zero_count : (if x = 0 then 1 else 0) + (if y = 0 then 1 else 0) + (if z = 0 then 1 else 0) ≤ 1) : 
  (∃ M : ℝ, M = 12 ∧ ∀ x y z, f x y z ≥ M) :=
sorry

end min_f_value_l640_640208


namespace depth_of_lost_ship_l640_640556

theorem depth_of_lost_ship (rate : ℕ) (time : ℕ) (desc_rate : rate = 30) (desc_time : time = 80) : rate * time = 2400 :=
by {
  rw [desc_rate, desc_time],
  exact Nat.mul_comm 30 80
}

end depth_of_lost_ship_l640_640556


namespace driver_end_position_fuel_consumed_l640_640890

/-- Define the distances traveled in each batch -/
def distances : List Int := [5, 2, -4, -3, 10]

/-- Define the fuel consumption rate per kilometer -/
def fuelConsumptionRate : Float := 0.2

/-- Calculate the total distance traveled -/
def totalDistance : Int := distances.sum

/-- Calculate the total fuel consumed based on the absolute distance traveled -/
def totalFuelConsumed : Float := (distances.map Int.natAbs).sum * fuelConsumptionRate

/-- Prove that the driver ends up 10 kilometers south of the company -/
theorem driver_end_position :
  totalDistance = 10 :=
by
  /- All conditions have been defined in terms of lean functions. So 
     simply prove the theorem by using the provided distances and their sum.-/
  sorry

/-- Prove that the total fuel consumed is 4.8 liters -/
theorem fuel_consumed :
  totalFuelConsumed = 4.8 :=
by
  /- All conditions have been defined in terms of lean functions. So
     simply prove the theorem by using the provided distances and absolute value sum. -/
  sorry

end driver_end_position_fuel_consumed_l640_640890


namespace dance_steps_equiv_l640_640627

theorem dance_steps_equiv
  (back1 : ℕ)
  (forth1 : ℕ)
  (back2 : ℕ)
  (forth2 : ℕ)
  (back3 : ℕ := 2 * back2) : 
  back1 = 5 ∧ forth1 = 10 ∧ back2 = 2 → 
  (0 - back1 + forth1 - back2 + forth2 = 7) :=
by
  intros h
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw [h1, h3, h4],
  sorry

end dance_steps_equiv_l640_640627


namespace temperature_constant_zero_l640_640896

theorem temperature_constant_zero
  (delta_f : ℝ) (delta_c : ℝ) (k : ℝ)
  (h_delta_f : delta_f = 26)
  (h_delta_c : delta_c = 14.444444444444445)
  (h_formula : ∀ c : ℝ, f : ℝ, f = (9 / 5) * c + k) :
  k = 0 :=
by
  sorry

end temperature_constant_zero_l640_640896


namespace non_empty_A_l640_640240

noncomputable def A (a : ℝ) : set ℝ := {x | (sqrt x)^2 ≠ a}

theorem non_empty_A (a : ℝ) : (∃ x : ℝ, x ∈ A a) ↔ a ∈ Iio 0 := 
sorry

end non_empty_A_l640_640240


namespace avg_weight_BC_l640_640469

variable (A B C : ℝ)

def totalWeight_ABC := 3 * 45
def totalWeight_AB := 2 * 40
def weight_B := 31

theorem avg_weight_BC : ((B + C) / 2) = 43 :=
  by
    have totalWeight_ABC_eq : A + B + C = totalWeight_ABC := by sorry
    have totalWeight_AB_eq : A + B = totalWeight_AB := by sorry
    have weight_B_eq : B = weight_B := by sorry
    sorry

end avg_weight_BC_l640_640469


namespace total_weight_kg_l640_640144

def envelope_weight_grams : ℝ := 8.5
def num_envelopes : ℝ := 800

theorem total_weight_kg : (envelope_weight_grams * num_envelopes) / 1000 = 6.8 :=
by
  sorry

end total_weight_kg_l640_640144


namespace urea_formation_l640_640612

theorem urea_formation (CO2 NH3 : ℕ) (OCN2 H2O : ℕ) (h1 : CO2 = 3) (h2 : NH3 = 6) :
  (∀ x, CO2 * 1 + NH3 * 2 = x + (2 * x) + x) →
  OCN2 = 3 :=
by
  sorry

end urea_formation_l640_640612


namespace monotonically_increasing_range_of_a_l640_640268

noncomputable def f (a x : ℝ) : ℝ :=
  x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotonically_increasing_range_of_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) :=
sorry

end monotonically_increasing_range_of_a_l640_640268


namespace solve_fra_eq_l640_640458

theorem solve_fra_eq : ∀ x : ℝ, (x - 2) / (x + 2) + 4 / (x^2 - 4) = 1 → x = 3 :=
by 
  -- Proof steps go here
  sorry

end solve_fra_eq_l640_640458


namespace monochromatic_grid_in_2n_minus_2_rounds_l640_640377

-- Define the colors
inductive Color
| Red
| Yellow
| Blue

open Color

-- Function to represent the color change
def color_change (grid : Matrix (Fin n) (Fin n) Color) : Matrix (Fin n) (Fin n) Color :=
  λ i j,
  match grid i j with
  | Red    => if i > 0 ∧ grid (i - 1) j = Yellow ∨ i < n-1 ∧ grid (i + 1) j = Yellow ∨ j > 0 ∧ grid i (j - 1) = Yellow ∨ j < n-1 ∧ grid i (j + 1) = Yellow then Yellow else Red
  | Yellow => if i > 0 ∧ grid (i - 1) j = Blue ∨ i < n-1 ∧ grid (i + 1) j = Blue ∨ j > 0 ∧ grid i (j - 1) = Blue ∨ j < n-1 ∧ grid i (j + 1) = Blue then Blue else Yellow
  | Blue   => if i > 0 ∧ grid (i - 1) j = Red ∨ i < n-1 ∧ grid (i + 1) j = Red ∨ j > 0 ∧ grid i (j - 1) = Red ∨ j < n-1 ∧ grid i (j + 1) = Red then Red else Blue

-- Main theorem
theorem monochromatic_grid_in_2n_minus_2_rounds (initial_grid : Matrix (Fin n) (Fin n) Color) :
  ∃ k ≤ 2 * n - 2, ∀ i j, (iterate color_change k initial_grid) i j = (iterate color_change k initial_grid) 0 0 :=
sorry

end monochromatic_grid_in_2n_minus_2_rounds_l640_640377


namespace volume_of_regular_octahedron_l640_640193

theorem volume_of_regular_octahedron (a : ℝ) : 
  let height := sqrt (a^2 - (a * sqrt 2 / 2)^2)
  let volume_pyramid := (1 / 3) * (a^2) * height
  let volume_octahedron := 2 * volume_pyramid
  volume_octahedron = (a^3 * sqrt 2) / 3 := by
  sorry

end volume_of_regular_octahedron_l640_640193


namespace minimum_OP_value_l640_640964

-- Definition of points A and B and their properties
structure Point where
  x : ℝ
  y : ℝ

noncomputable def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Midpoint definition
noncomputable def midpoint (A B : Point) : Point :=
  ⟨ (A.x + B.x) / 2, (A.y + B.y) / 2 ⟩

-- Main statement
theorem minimum_OP_value (A B P O : Point) 
    (h_midpoint : O = midpoint A B)
    (h_AB : dist A B = 4)
    (h_dist_diff : dist P A - dist P B = 3) : 
    ∃ P, dist O P = 3 / 2 :=
sorry

end minimum_OP_value_l640_640964


namespace quadratic_equation_correct_form_l640_640564

theorem quadratic_equation_correct_form :
  ∀ (a b c x : ℝ), a = 3 → b = -6 → c = 1 → a * x^2 + c = b * x :=
by
  intros a b c x ha hb hc
  rw [ha, hb, hc]
  sorry

end quadratic_equation_correct_form_l640_640564


namespace sum_of_values_l640_640840

theorem sum_of_values (x : ℝ) (h : x^2 = 15 * x - 10) : x = 10 ∨ x = 5 → (10 + 5) = 15 := 
by 
    intros 
    rw [← add_self_eq_double 5] 
    simp

end sum_of_values_l640_640840


namespace probability_complete_collection_l640_640043

def combination (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem probability_complete_collection : 
  combination 6 6 * combination 12 4 / combination 18 10 = 5 / 442 :=
by
  have h1 : combination 6 6 = 1 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h2 : combination 12 4 = 495 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h3 : combination 18 10 = 43758 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  rw [h1, h2, h3]
  norm_num
  sorry

end probability_complete_collection_l640_640043


namespace max_white_cells_l640_640929

-- Problem entities and conditions
def cell := ℕ × ℕ
inductive color
| red | blue | white

def condition1 (table : cell → color) : Prop :=
∀ c : cell, table c = color.red ∨ table c = color.blue ∨ table c = color.white

def has_pawn (c : cell) (table : cell → color) : bool :=
if table c = color.red then tt
else if table c = color.blue then tt
else ff

def condition2 (table : cell → color) : Prop :=
∀ c : cell, (table c = color.white) ->
((∃ (dv dh : ℕ), dv * dv + dh * dh ≤ 2 ∧ table (c.1 + dv, c.2 + dh) = color.blue) ∨
 (∃ (dv dh : ℕ), dv * dv + dh * dh ≤ 2 ∧ table (c.1 - dv, c.2 - dh) = color.blue))

def condition3 (table : cell → color) : Prop :=
∀ (i j : ℕ), i < 2023 → j < 2023 → (table (i,j) = color.red ∧ table (i+1,j) = color.blue ∧
 table (i,j+1) = color.blue ∧ table (i+1,j+1) = color.red) ∨
(table (i,j) = color.blue ∧ table (i+1,j) = color.red ∧ 
table (i,j+1) = color.red ∧ table (i+1,j+1) = color.blue) ∨
(table (i,j) = color.red ∧ table (i+1,j) = color.red ∧ 
table (i,j+1) = color.blue ∧ table (i+1,j+1) = color.blue) ∨
(table (i,j) = color.blue ∧ table (i+1,j) = color.blue ∧ 
table (i,j+1) = color.red ∧ table (i+1,j+1) = color.red)

-- Theorem statement
theorem max_white_cells : ∃ n : ℕ, 4 * n = 2024 ∧ ((n * n * 10) = 2560360) ∧ condition1 table ∧ condition2 table ∧ condition3 table :=
sorry

end max_white_cells_l640_640929


namespace problem_N_calculation_l640_640397

def S : Finset ℕ := Finset.range 11

noncomputable def calculateN : ℕ :=
  ∑ x in S, (2 * x - 10) * 3^x

theorem problem_N_calculation : calculateN = 754152 := by
  sorry

end problem_N_calculation_l640_640397


namespace smallest_x_for_multiple_l640_640074

theorem smallest_x_for_multiple (x : ℕ) : (450 * x) % 720 = 0 ↔ x = 8 := 
by {
  sorry
}

end smallest_x_for_multiple_l640_640074


namespace sum_values_of_cubes_eq_l640_640672

theorem sum_values_of_cubes_eq :
  ∀ (a b : ℝ), a^3 + b^3 + 3 * a * b = 1 → a + b = 1 ∨ a + b = -2 :=
by
  intros a b h
  sorry

end sum_values_of_cubes_eq_l640_640672


namespace proof_options_correct_l640_640849

noncomputable theory

def option_A (n : ℕ) (p : ℝ) (X : ℕ → ℝ) [BinomialDist n p X] : Prop :=
  ∀ E_X D_X, E_X = 40 ∧ D_X = 30 → p = 1/4

def option_B (σ² : ℝ) : Prop :=
  ∀ σ²', σ²' = 4 * σ²

def option_C (ξ : ℝ → ℝ) (p : ℝ) [NormalDist 0 4 ξ] : Prop :=
  P(ξ > 2) = p → P(-2 ≤ ξ < 0) = (1 / 2) - p

def option_D (X : ℕ → ℝ) [BinomialDist 10 0.8 X] : Prop :=
  ∀ k, maximized_prob k → k = 8

theorem proof_options_correct : 
  (option_A n p X) ∧ (option_B σ²) ∧ (option_C ξ p) ∧ (option_D X) :=
sorry


end proof_options_correct_l640_640849


namespace manager_salary_l640_640857

theorem manager_salary (avg_salary_20 : ℕ) (num_employees : ℕ) (salary_increase : ℕ) (new_avg_salary : ℕ)
    (old_total_salary : ℕ) (new_total_salary : ℕ) (manager_salary : ℕ) :
  avg_salary_20 = 1500 →
  num_employees = 20 →
  salary_increase = 1000 →
  new_avg_salary = avg_salary_20 + salary_increase →
  old_total_salary = num_employees * avg_salary_20 →
  new_total_salary = (num_employees + 1) * new_avg_salary →
  manager_salary = new_total_salary - old_total_salary →
  manager_salary = 22500 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  -- Proof steps could be added here, but we use sorry for now.
  sorry,
end

end manager_salary_l640_640857


namespace beetle_paths_count_l640_640546

-- Defining points in the lattice
inductive Point
| A | B | D | E | F | G | H | C

-- Defining possible moves according to the problem conditions
def move : Point → Point → Prop
| Point.A Point.B := true
| Point.B Point.D := true
| Point.B Point.E := true
| Point.D Point.F := true
| Point.D Point.G := true
| Point.E Point.G := true
| Point.E Point.H := true
| Point.F Point.C := true
| Point.G Point.C := true
| Point.H Point.C := true
| _ _ := false

-- Define what a path is: a sequence of points where each pair is a valid move
def Path : List Point → Prop
| [] := false
| (p :: []) := p = Point.A
| (p1 :: p2 :: ps) := move p1 p2 ∧ Path (p2 :: ps)

-- Counting the number of valid paths from A to C
def count_paths : Nat :=
let paths := [
    [Point.A, Point.B, Point.D, Point.F, Point.C],
    [Point.A, Point.B, Point.D, Point.G, Point.C],
    [Point.A, Point.B, Point.E, Point.G, Point.C],
    [Point.A, Point.B, Point.E, Point.H, Point.C],
    [Point.A, Point.B, Point.E, Point.G, Point.C]
]
List.length paths

theorem beetle_paths_count : count_paths = 5 := by
  sorry

end beetle_paths_count_l640_640546


namespace conditional_statement_C_correct_l640_640796

def conditional_statement_C_represents_not_met_block : Prop :=
  ∀ (A B C : Prop), (if A then B else C) = C → C = "The content executed when the condition is not met"

theorem conditional_statement_C_correct :
  conditional_statement_C_represents_not_met_block :=
by
  sorry

end conditional_statement_C_correct_l640_640796


namespace intersect_rectangular_eqn_range_of_m_l640_640360

-- Definitions of the parametric and polar equations
def curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def line_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Proving the rectangular equation and range of m for the intersection of l and C
theorem intersect_rectangular_eqn (m t : ℝ) :
  (∃ ρ θ, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  ∃ x y, (x, y) = curve_C t ∧ √3 * x + y + 2 * m = 0 :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ ρ θ t, line_l ρ θ m ∧ (ρ * cos θ, ρ * sin θ) = curve_C t) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end intersect_rectangular_eqn_range_of_m_l640_640360


namespace Carol_optimal_choice_l640_640898

noncomputable def Alice_choices := Set.Icc 0 (1 : ℝ)
noncomputable def Bob_choices := Set.Icc (1 / 3) (3 / 4 : ℝ)

theorem Carol_optimal_choice : 
  ∀ (c : ℝ), c ∈ Set.Icc 0 1 → 
  (∃! c, c = 7 / 12) := 
sorry

end Carol_optimal_choice_l640_640898


namespace rect_eq_and_range_of_m_l640_640311

noncomputable def rect_eq_of_polar (m : ℝ) : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ * sin (θ + π / 3) + m = 0 → sqrt 3 * (ρ * cos θ) + (ρ * sin θ) + 2 * m = 0

noncomputable def range_of_m_for_intersection : set ℝ := 
  {m : ℝ | -19 / 12 ≤ m ∧ m ≤ 5 / 2}

theorem rect_eq_and_range_of_m (m : ℝ) : 
  rect_eq_of_polar m ∧ (∀ t : ℝ, ∃ x y : ℝ, x = sqrt 3 * cos (2 * t) ∧ y = 2 * sin t ∧ 
    (sqrt 3 * x + y + 2 * m = 0) → m ∈ range_of_m_for_intersection) :=
by sorry

end rect_eq_and_range_of_m_l640_640311


namespace conjugate_quadrant_l640_640473

def main : IO Unit := 
  IO.println s!"Hello, world!"

noncomputable def z : ℂ := 1 / (1 + complex.I)

theorem conjugate_quadrant (h : complex.I^2 = -1) : 
  let z_conj := complex.conj z 
  ∃ q : Nat, (q = 1) ∧ (z_conj.re > 0) ∧ (z_conj.im > 0) :=
by
  have : z = (1 - complex.I) / 2 := by
    sorry
  let z_conj := complex.conj z
  use 1
  split
  · sorry
  split
  · sorry
  · sorry

end conjugate_quadrant_l640_640473


namespace solve_equation_l640_640457

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) ↔ x = -4 ∨ x = -2 :=
by
  sorry

end solve_equation_l640_640457


namespace avg_value_T_l640_640000

noncomputable def avg_value_of_set (S : Finset ℕ) (H : S.Nonempty) : ℚ := 
  (S.sum id : ℚ) / S.card

theorem avg_value_T 
  (T : Finset ℕ) 
  (Hm1 : T.card > 2) -- T is a finite set and its cardinality is greater than 2
  (H_remove_max_avg_43 : avg_value_of_set (T.erase (T.max' Hm1)) (by sorry) = 43)
  (H_remove_min_max_avg_47 : avg_value_of_set (T.erase (T.min') \ (T.max' Hm1)) (by sorry) = 47)
  (H_add_max_back_avg_52_no_min : avg_value_of_set ((T.erase (T.min')).insert (T.max' Hm1)) (by sorry) = 52)
  (H_max_min_rel : T.max' Hm1 = T.min' + 65) : 
  avg_value_of_set T (by sorry) = 47.8 := 
sorry

end avg_value_T_l640_640000


namespace probability_other_side_red_l640_640867

def card_black_black := 4
def card_black_red := 2
def card_red_red := 2

def total_cards := card_black_black + card_black_red + card_red_red

-- Calculate the total number of red faces
def total_red_faces := (card_red_red * 2) + card_black_red

-- Number of red faces that have the other side also red
def red_faces_with_other_red := card_red_red * 2

-- Target probability to prove
theorem probability_other_side_red (h : total_cards = 8) : 
  (red_faces_with_other_red / total_red_faces) = 2 / 3 := 
  sorry

end probability_other_side_red_l640_640867


namespace four_letter_arrangements_l640_640697

-- Definitions based on the problem conditions
def first_letter_fixed : Prop := true
def second_letter_fixed : Prop := true
def is_in_third_or_fourth_position (letter : Char) : Prop := letter = 'B'
def remaining_letters := ['A', 'E', 'F', 'G']

-- The math proof problem in Lean 4 statement
theorem four_letter_arrangements : 
  first_letter_fixed → 
  second_letter_fixed → 
  is_in_third_or_fourth_position 'B' → 
  ∃ (num_arrangements : ℕ), num_arrangements = 10 :=
by
  intros _ _ _
  use 10
  sorry

end four_letter_arrangements_l640_640697


namespace polar_to_rectangular_range_of_m_l640_640289

-- Definition of the parametric equations of curve C
def curve (t : ℝ) : ℝ × ℝ :=
  (sqrt(3) * cos (2 * t), 2 * sin t)

-- Definition of the polar equation of line l
def polar_line (rho theta m : ℝ) : Prop :=
  rho * sin (theta + π / 3) + m = 0

-- Definition of the rectangular equation of line l
def rectangular_line (x y m : ℝ) : Prop :=
  sqrt(3) * x + y + 2 * m = 0

-- Convert polar line equation to rectangular form
theorem polar_to_rectangular (rho theta m x y : ℝ)
  (h1 : polar_line rho theta m)
  (h2 : rho = sqrt (x^2 + y^2))
  (h3 : cos theta * rho = x ∧ sin theta * rho = y) :
  rectangular_line x y m :=
sorry

-- Range of values for m for which l intersects C
theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, (sqrt(3) * cos (2 * t), 2 * sin t) ∈ {p : ℝ × ℝ | rectangular_line p.1 p.2 m}) ↔
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
sorry

end polar_to_rectangular_range_of_m_l640_640289


namespace correct_propositions_l640_640683

-- Define the original propositions
def prop1 (a b c : ℝ) (h : a ≠ 0): Prop := (b^2 - 4 * a * c < 0 → ¬ has_real_roots a b c)
def prop2 (A B C : Triangle): Prop := (A.side_eq B ∧ B.side_eq C ∧ C.side_eq A → A.is_equilateral)
def prop3 (a b : ℝ) : Prop := (a > b ∧ b > 0 ∧ 3 * a > 3 * b ∧ 3 * b > 0) 
def prop4 (m : ℝ) : Prop := (m > 1 → solution_set m = ℝ)

-- Define the transformed propositions
def neg_prop1 (a b c : ℝ) (h : a ≠ 0): Prop := (b^2 - 4 * a * c ≥ 0 → has_real_roots a b c)
def conv_prop2 (A B C : Triangle) : Prop := (A.is_equilateral → A.side_eq B ∧ B.side_eq C ∧ C.side_eq A)
def inv_neg_prop3 (a b : ℝ) : Prop := (a ≤ b ∨ b ≤ 0 ∨ 3 * a ≤ 3 * b ∨ 3 * b ≤ 0)
def conv_prop4 (m : ℝ) : Prop := (solution_set m = ℝ → m > 1)

-- Define if each proposition is correct
def isCorrect_prop1 : Prop := neg_prop1 a b c h
def isCorrect_prop2 : Prop := conv_prop2 A B C
def isCorrect_prop3 : Prop := inv_neg_prop3 a b
def isCorrect_prop4 : Prop := ¬ conv_prop4 m

-- Define the main theorem
theorem correct_propositions : isCorrect_prop1 ∧ isCorrect_prop2 ∧ isCorrect_prop3 ∧ ¬ isCorrect_prop4 :=
by
  sorry

end correct_propositions_l640_640683
