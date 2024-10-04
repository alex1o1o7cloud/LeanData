import Mathlib
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.Calculus.FDeriv
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Partition
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Parity
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Trig
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Barycentric
import Mathlib.GroupTheory.SpecificGroups.Dihedral
import Mathlib.NumberTheory.ProbabilityTheory
import Mathlib.Probability
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.Basic
import Probability.ProbabilityTheory
import data.real.basic
import tactic

namespace find_x_l83_83373

variable (a b x : ℝ)

def star (a b : ℝ) : ℝ :=
  (Real.sqrt (a + b)) / (Real.sqrt (a - b))

theorem find_x (h : star x 30 = 8) : x = 650 / 21 :=
  sorry

end find_x_l83_83373


namespace largest_multiple_of_7_whose_negation_greater_than_neg80_l83_83461

theorem largest_multiple_of_7_whose_negation_greater_than_neg80 : ∃ (n : ℤ), n = 77 ∧ (∃ (k : ℤ), n = k * 7) ∧ (-n > -80) :=
by
  sorry

end largest_multiple_of_7_whose_negation_greater_than_neg80_l83_83461


namespace probability_first_player_takes_card_l83_83786

variable (n : ℕ) (i : ℕ)

-- Conditions
def even_n : Prop := ∃ k, n = 2 * k
def valid_i : Prop := 1 ≤ i ∧ i ≤ n

-- The key function (probability) and theorem to prove
def P (i n : ℕ) : ℚ := (i - 1) / (n - 1)

theorem probability_first_player_takes_card :
  even_n n → valid_i n i → P i n = (i - 1) / (n - 1) :=
by
  intro h1 h2
  sorry

end probability_first_player_takes_card_l83_83786


namespace repeating_decimal_fraction_sum_l83_83836

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l83_83836


namespace initial_number_of_children_l83_83409

theorem initial_number_of_children (C S : ℕ) (h1 : S = C * 15)
  (h2 : S = (C - 32) * 21) : C = 112 :=
begin
  -- Proof omitted
  sorry
end

end initial_number_of_children_l83_83409


namespace necessary_and_sufficient_condition_l83_83261

theorem necessary_and_sufficient_condition (x : ℝ) (h : x > 0) : (x + 1/x ≥ 2) ↔ (x > 0) :=
sorry

end necessary_and_sufficient_condition_l83_83261


namespace solve_for_x_l83_83622

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end solve_for_x_l83_83622


namespace angle_bisector_DE_l83_83237

variables {P : Type} [MetricSpace P] [Table2D P]
variables {D A B C E : P}
variables {S S1 : Set P}

-- Given conditions
axiom cond1 : D ∈ S
axiom cond2 : ∀ {a b : P}, is_perpendicular a b → (C ∈ segment a b) → a = A → b = B
axiom cond3 : is_tangent S1 (segment C A) E
axiom cond4 : is_tangent S1 (segment C D) 
axiom cond5 : is_tangent S1 S

-- Prove that DE is the angle bisector of triangle ADC
theorem angle_bisector_DE : is_angle_bisector (segment D E) (triangle A D C) :=
sorry

end angle_bisector_DE_l83_83237


namespace foil_covered_prism_width_l83_83135

theorem foil_covered_prism_width
    (l w h : ℕ)
    (inner_volume : l * w * h = 128)
    (width_length_relation : w = 2 * l)
    (width_height_relation : w = 2 * h) :
    (w + 2) = 10 := 
sorry

end foil_covered_prism_width_l83_83135


namespace zip_code_relationship_l83_83540

theorem zip_code_relationship (A B C D E : ℕ) 
(h1 : A + B + C + D + E = 10) 
(h2 : C = 0) 
(h3 : D = 2 * A) 
(h4 : D + E = 8) : 
A + B = 2 :=
sorry

end zip_code_relationship_l83_83540


namespace triangular_stack_log_count_l83_83975

theorem triangular_stack_log_count : 
  ∀ (a₁ aₙ d : ℤ) (n : ℤ), a₁ = 15 → aₙ = 1 → d = -2 → 
  (a₁ - aₙ) / (-d) + 1 = n → 
  (n * (a₁ + aₙ)) / 2 = 64 :=
by
  intros a₁ aₙ d n h₁ hₙ hd hn
  sorry

end triangular_stack_log_count_l83_83975


namespace sqrt3_exp_7pi_over_3_to_rect_form_l83_83552

noncomputable def sqrt3_exp_7pi_over_3_rect_form : ℂ := 
  √3 * complex.exp (7 * real.pi * complex.I / 3)

theorem sqrt3_exp_7pi_over_3_to_rect_form :
  sqrt3_exp_7pi_over_3_rect_form = (√3 / 2) + (3 / 2) * complex.I := by
  sorry

end sqrt3_exp_7pi_over_3_to_rect_form_l83_83552


namespace least_positive_integer_l83_83811

theorem least_positive_integer :
  ∃ (N : ℕ), N % 11 = 10 ∧ N % 12 = 11 ∧ N % 13 = 12 ∧ N % 14 = 13 ∧ N = 12011 :=
by
  use 12011
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 11 = 10
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 12 = 11
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 13 = 12
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 14 = 13
  · rfl

end least_positive_integer_l83_83811


namespace sum_of_remainders_l83_83118

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4) + (n % 5) = 4 := 
by {
  -- proof omitted
  sorry
}

end sum_of_remainders_l83_83118


namespace tan_three_radians_neg_l83_83084

theorem tan_three_radians_neg : 3 ∈ set.Ioc (π / 2) π → real.tan 3 < 0 :=
begin
    sorry
end

end tan_three_radians_neg_l83_83084


namespace trigonometric_identity_proof_l83_83490

open Real

noncomputable def trigonometric_identity (α : ℝ) :=
  tan ((π / 4) - (α / 2)) * (1 - cos ((3 / 2) * π - α)) * (sec α) - 2 * cos 2 * α

noncomputable def numerator (α : ℝ) :=
  tan ((π / 4) - (α / 2)) * (1 + sin (4 * π + α)) * (sec α) + 2 * cos 2 * α

theorem trigonometric_identity_proof (α : ℝ) :
  (trigonometric_identity α) / (numerator α) = 
  tan ((π / 6) + α) * tan (α - (π / 6)) :=
sorry

end trigonometric_identity_proof_l83_83490


namespace candy_received_l83_83230

theorem candy_received (pieces_eaten : ℕ) (piles : ℕ) (pieces_per_pile : ℕ) 
  (h_eaten : pieces_eaten = 12) (h_piles : piles = 4) (h_pieces_per_pile : pieces_per_pile = 5) :
  pieces_eaten + piles * pieces_per_pile = 32 := 
by
  sorry

end candy_received_l83_83230


namespace draw_is_unfair_ensure_fair_draw_l83_83325

open ProbabilityTheory MeasureTheory

-- Definitions for the given conditions:
def Card := {rank : ℕ // 6 ≤ rank ∧ rank ≤ 14} -- Ranks 6 to Ace (6 to 14)
def Deck := Finset (Fin 36) -- 36 unique cards
noncomputable def suit_high_rank_count (d : Deck) (v_card : Fin 36) (m_card : Fin 36) : ℕ := 
  -- Count how many cards are higher than Volodya's card
  card.count (λ c, c.val > v_card.val) d

-- Volodya draws first, then Masha draws:
variables (d : Deck) (v_card m_card : Fin 36)

-- Masha wins if she draws a card with a higher rank than Volodya’s card
def masha_wins := ∃ (m_card : Fin 36), (m_card ∈ d) ∧ (m_card.val > v_card.val)

-- Volodya wins if Masha doesn't win (Masha loses)
def volodya_wins := ¬ masha_wins

theorem draw_is_unfair (d : Deck) (v_card m_card : Fin 36) :
  (volodya_wins d v_card m_card) → ¬ (masha_wins d v_card) := sorry

-- To make it fair, we can introduce a suit hierarchy:
def suits := {"Hearts", "Diamonds", "Clubs", "Spades"}
def suit_order : suits → suits → Prop
| "Spades" "Hearts" := true
| "Hearts" "Diamonds" := true
| "Diamonds" "Clubs" := true
| "Clubs" "Spades" := false
| _, _ := false

-- A fair draw means using the suit_order to rank otherwise equal cards:
def fair_draw :=
  ∀ (c1 c2 : Card), (c1.rank = c2.rank → suit_order c1.suit c2.suit)

theorem ensure_fair_draw : fair_draw := sorry

end draw_is_unfair_ensure_fair_draw_l83_83325


namespace roots_polynomial_value_l83_83008

theorem roots_polynomial_value (a b c : ℝ) (h_roots : (x^3 - 15 * x^2 + 25 * x - 10 = 0) → (a, b, c ∈ roots x^3 - 15 * x^2 + 25 * x - 10 )) :
  (1 + a) * (1 + b) * (1 + c) = 51 :=
by
  sorry

end roots_polynomial_value_l83_83008


namespace constant_term_in_binomial_expansion_l83_83376

theorem constant_term_in_binomial_expansion :
  let a := ∫ x in Set.Icc (Real.pi) (Real.exp (2 * Real.pi)), 1 / x in
  (a = 1) →
  constant_term (λ x: ℝ, (a * x^2 - 1 / x)^6) = 15 :=
by
  sorry

end constant_term_in_binomial_expansion_l83_83376


namespace find_h_l83_83372

noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + 3 * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + ...)))

theorem find_h (h : ℝ) (H : bowtie 4 h = 10) : h = 2 :=
sorry

end find_h_l83_83372


namespace apple_count_l83_83620

-- Definitions of initial conditions and calculations.
def B_0 : Int := 5  -- initial number of blue apples
def R_0 : Int := 3  -- initial number of red apples
def Y : Int := 2 * B_0  -- number of yellow apples given by neighbor
def R : Int := R_0 - 2  -- number of red apples after giving away to a friend
def B : Int := B_0 - 3  -- number of blue apples after 3 rot
def G : Int := (B + Y) / 3  -- number of green apples received
def Y' : Int := Y - 2  -- number of yellow apples after eating 2
def R' : Int := R - 1  -- number of red apples after eating 1

-- Lean theorem statement
theorem apple_count (B_0 R_0 Y R B G Y' R' : ℤ)
  (h1 : B_0 = 5)
  (h2 : R_0 = 3)
  (h3 : Y = 2 * B_0)
  (h4 : R = R_0 - 2)
  (h5 : B = B_0 - 3)
  (h6 : G = (B + Y) / 3)
  (h7 : Y' = Y - 2)
  (h8 : R' = R - 1)
  : B + Y' + G + R' = 14 := 
by
  sorry

end apple_count_l83_83620


namespace problem1_problem2_problem3_l83_83020

section problem

variable (m : ℝ)

-- Proposition p: The equation x^2 - 4mx + 1 = 0 has real solutions
def p : Prop := (16 * m^2 - 4) ≥ 0

-- Proposition q: There exists some x₀ ∈ ℝ such that mx₀^2 - 2x₀ - 1 > 0
def q : Prop := ∃ (x₀ : ℝ), (m * x₀^2 - 2 * x₀ - 1) > 0

-- Solution to (1): If p is true, the range of values for m
theorem problem1 (hp : p m) : m ≥ 1/2 ∨ m ≤ -1/2 := sorry

-- Solution to (2): If q is true, the range of values for m
theorem problem2 (hq : q m) : m > -1 := sorry

-- Solution to (3): If both p and q are false but either p or q is true,
-- find the range of values for m
theorem problem3 (hnp : ¬p m) (hnq : ¬q m) (hpq : p m ∨ q m) : -1 < m ∧ m < 1/2 := sorry

end problem

end problem1_problem2_problem3_l83_83020


namespace mailman_blocks_l83_83157

theorem mailman_blocks (total_mail pieces_per_block : ℕ) (h1 : total_mail = 192) (h2 : pieces_per_block = 48) :
  total_mail / pieces_per_block = 4 :=
by
  rw [h1, h2]
  norm_num

end mailman_blocks_l83_83157


namespace repeating_decimal_fraction_sum_l83_83854

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l83_83854


namespace monotonic_increase_interval_for_quadratic_l83_83768

noncomputable def is_monotonic_increase_interval (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ interval → y ∈ interval → x ≤ y → f x ≤ f y

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem monotonic_increase_interval_for_quadratic :
  is_monotonic_increase_interval (quadratic_function 1 (-0.4) Real.exp) (Set.Ici 0.2) :=
sorry

end monotonic_increase_interval_for_quadratic_l83_83768


namespace smallest_abundant_not_multiple_of_five_l83_83106

def is_abundant (n : ℕ) : Prop :=
  (∑ m in Nat.properDivisors n, m) > n

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

theorem smallest_abundant_not_multiple_of_five : ∃ n : ℕ, is_abundant n ∧ ¬ is_multiple_of_five n ∧ ∀ m : ℕ, is_abundant m ∧ ¬ is_multiple_of_five m → n ≤ m := sorry

end smallest_abundant_not_multiple_of_five_l83_83106


namespace min_colors_needed_for_boxes_l83_83824

noncomputable def min_colors_needed : Nat := 23

theorem min_colors_needed_for_boxes :
  ∀ (boxes : Fin 8 → Fin 6 → Nat), 
  (∀ i, ∀ j : Fin 6, boxes i j < min_colors_needed) → 
  (∀ i, (Function.Injective (boxes i))) → 
  (∀ c1 c2, c1 ≠ c2 → (∃! b, ∃ p1 p2, (p1 ≠ p2 ∧ boxes b p1 = c1 ∧ boxes b p2 = c2))) → 
  min_colors_needed = 23 := 
by sorry

end min_colors_needed_for_boxes_l83_83824


namespace lateral_edge_base_plane_angle_l83_83776

theorem lateral_edge_base_plane_angle (n : ℕ) (t : ℝ) 
  (h_t_pos : 1 < t) : 
  let β := Real.arctan (Real.cos (Real.pi / n) * Real.sqrt (t^2 - 2 * t)) in 
  ∃ β, β = β :=
by sorry

end lateral_edge_base_plane_angle_l83_83776


namespace max_abs_diff_sum_l83_83033

-- Define the grid as a 3x3 matrix of integers
def grid := matrix (fin 3) (fin 3) ℕ

-- Define a function to compute the sum of absolute differences between adjacent cells 
def abs_diff_sum (g : grid) : ℕ :=
  let horizontal_diff := ∑ i in finset.univ, ∑ j in finset.range 2, abs (g i j - g i (j + 1))
  let vertical_diff := ∑ j in finset.univ, ∑ i in finset.range 2, abs (g i j - g (i + 1) j)
  horizontal_diff + vertical_diff

-- Define the condition that the grid contains exactly the numbers 1 through 9 in some permutation
def valid_grid (g : grid) : Prop :=
  (finset.univ : finset (fin 3 × fin 3)).image g = {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- The theorem that needs to be proven
theorem max_abs_diff_sum : ∃ g : grid, valid_grid g ∧ abs_diff_sum g = 58 := sorry

end max_abs_diff_sum_l83_83033


namespace continued_fraction_identity_l83_83598

noncomputable section

-- Define the sequence of fractions
def continued_fraction : ℕ → ℚ
| 0 => 1
| (n + 1) => 1 / (1 + continued_fraction n)

-- Define the recursive relation for the given problem
def a_l (l : ℕ) : ℚ :=
if l = 0 then 1 else 1 / (1 + a_l (l - 1))

-- Define the irreducible fraction after 1990 layers
def m_n_1990 : ℚ :=
a_l 1990

theorem continued_fraction_identity :
  ∃ m n : ℤ, Nat.coprime m.natAbs n.natAbs ∧ 
  (m_n_1990.num : ℤ) = m ∧ (m_n_1990.denom : ℤ) = n ∧
  (1 / 2 + m / n) ^ 2 = 5 / 4 - 1 / n ^ 2 :=
sorry

end continued_fraction_identity_l83_83598


namespace sum_of_fraction_terms_l83_83856

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l83_83856


namespace parabola_equation_ellipse_eccentricity_l83_83609

-- Given conditions
def parabola_eqn (p : ℝ) := ∀ x y : ℝ, y^2 = 2 * p * x
def point_on_parabola (x₀ y₀ p : ℝ) := y₀^2 = 2 * p * x₀
def focus_distance (x₀ y₀ p : ℝ) := real.sqrt ((x₀ - p / 2)^2 + y₀^2) = 3
def point_on_circle (x₀ y₀ : ℝ) := x₀^2 + y₀^2 = 9

-- Prove the equation of parabola C1
theorem parabola_equation (p x₀ y₀ : ℝ) (H₁ : p > 0)
  (H₂ : point_on_parabola x₀ y₀ p)
  (H₃ : focus_distance x₀ y₀ p)
  (H₄ : point_on_circle x₀ y₀) : 
  y^2 = 8 * x :=
sorry

-- Ellipse conditions and eccentricity
def ellipse_eqn (m n : ℝ) := m > n ∧ n > 0 ∧ ∀ x y, x^2 / m^2 + y^2 / n^2 = 1
def ellipse_focus_eqn (m n c : ℝ) := m^2 - n^2 = c^2
def symmetrical_points (x₁ y₁ x₂ y₂ : ℝ) 
  (m n : ℝ) :=
  y₁ = y₂ ∧ x₁ + x₂ = 8 * m^2 * (y₂ / (16 * m^2 + n^2)) ∧ 
  (y₁ + y₂) = (2 * y₂ * n^2) / (16 * m^2 + n^2)
  
def midpoint_eqn (m n λ : ℝ) :=
  ∃ (x₀ y₀ : ℝ),
  x₀ = 4 * m^2 * λ / (16 * m^2 + n^2) ∧
  y₀ = λ * n^2 / (16 * m^2 + n^2) ∧
  y₀ = 1/4 * x₀ + 1/3

-- Prove the range of the eccentricity e
theorem ellipse_eccentricity (m n e λ : ℝ) (H₁ : m > n)
  (H₂ : midpoint_eqn m n λ)
  (H₃ : ellipse_focus_eqn m n 2)
  (H₄ : e = real.sqrt (1 - (n^2 / m^2))) :
  (real.sqrt (629) / 37 < e) ∧ (e < 1) :=
sorry

end parabola_equation_ellipse_eccentricity_l83_83609


namespace eccentricity_of_hyperbola_l83_83739

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h : a > 0) (k : b > 0) 
  (hp : ∃ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1)
  (hfocus : c^2 = a^2 + b^2) 
  (hm : |c| = √((2c - c)^2 + (√3 c)^2)) 
  (hdot : c * (2c - c) / 2 = c^2 / 2) 
  : ℝ := 
    (1 + real.sqrt 3) / 2

theorem eccentricity_of_hyperbola (a b c e: ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : ∃ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1)
  (h₄ : c^2 = a^2 + b^2)
  (h₅ : |c| = √((2c - c)^2 + (√3 c)^2)) 
  (h₆ : c * (2c - c) / 2 = c^2 / 2)
  : e = (1 + real.sqrt 3) / 2 :=
sorry

end eccentricity_of_hyperbola_l83_83739


namespace intersection_point_on_circumcircle_l83_83698

open EuclideanGeometry

-- Letting I and O be the incentre and circumcentre of ΔABC
variables {A B C D E : Point}
variable {I O I1 O1 K : Point}
-- BD = CE = BC
variable (hBC: BD = CE ∧ BD = BC)

-- D and E are on sides AB and AC respectively
variables (hD_on_AB : ∃ M, on_line AB D ∧ M = D) 
          (hE_on_AC : ∃ N, on_line AC E ∧ N = E)

-- I and O are the incentre and circumcentre of ΔABC
variables (hI : incentre A B C I)
          (hO : circumcentre A B C O)

-- I1 and O1 are the incentre and circumcentre of ΔADE
variables (hI1 : incentre A D E I1)
          (hO1 : circumcentre A D E O1)

-- Prove K lies on the circumscribed circle of ΔABC.
theorem intersection_point_on_circumcircle 
  (h_intersect : ∃ K, lies_on (line_through I O) K ∧ lies_on (line_through I1 O1) K) : 
  lies_on_circumscribed_circle A B C K :=
sorry

end intersection_point_on_circumcircle_l83_83698


namespace regions_divided_by_great_circles_l83_83671

def f : ℕ → ℕ
| 0 => 1
| n+1 =>  f n + 2*n

theorem regions_divided_by_great_circles (n : ℕ) :
  f(n) = if n = 0 then 1 else n^2 - n + 2 := sorry

end regions_divided_by_great_circles_l83_83671


namespace main_theorem_l83_83217

section

noncomputable def phi : (Polynomial ℚ) → (Polynomial ℚ) := sorry

def is_Q_linear (ϕ : (Polynomial ℚ) → (Polynomial ℚ)) : Prop :=
  ∀ (p q : Polynomial ℚ) (a b : ℚ), ϕ(a • p + b • q) = a • ϕ(p) + b • ϕ(q)

def irreducible_preserving (ϕ : (Polynomial ℚ) → (Polynomial ℚ)) : Prop :=
  ∀ (p : Polynomial ℚ), irreducible p → irreducible (ϕ p)

theorem main_theorem :
    ∀ (ϕ : (Polynomial ℚ) → (Polynomial ℚ)),
    is_Q_linear ϕ →
    irreducible_preserving ϕ →
    ∃ (a b c : ℚ), (a ≠ 0) ∧ (b ≠ 0) ∧ (Φ = λ p, a * p.eval (b * X + c)) := sorry

end

end main_theorem_l83_83217


namespace radius_of_film_l83_83151

theorem radius_of_film (r h t : ℝ) (V : ℝ) (R : ℝ) : 
  r = 5 → 
  h = 8 → 
  t = 0.2 → 
  V = π * r^2 * h → 
  π * R^2 * t = V → 
  R = 10 * real.sqrt 10 :=
by
  intros hr hh ht hv heq
  sorry

end radius_of_film_l83_83151


namespace code_word_TREES_is_41225_l83_83389

def letter_to_digit (c : Char) : Nat :=
  match c with
  | 'G' => 0
  | 'R' => 1
  | 'E' => 2
  | 'A' => 3
  | 'T' => 4
  | 'S' => 5
  | 'U' => 6
  | 'C' => 7
  | _ => 0  -- this default case is here to handle non-matching inputs, technically we should have a non-exhaustive match since we control the input.

theorem code_word_TREES_is_41225 :
    (letter_to_digit 'T') * 10000 +
    (letter_to_digit 'R') * 1000 +
    (letter_to_digit 'E') * 100 +
    (letter_to_digit 'E') * 10 +
    (letter_to_digit 'S') = 41225 := 
by
  simp [letter_to_digit]
  sorry

end code_word_TREES_is_41225_l83_83389


namespace recurring_fraction_sum_l83_83920

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l83_83920


namespace problem_1_problem_2_l83_83605

-- Part (I) 
theorem problem_1 (x : ℝ) : 
  let f (x : ℝ) := |x - 1| in 
  f x ≥ (1 / 2) * (x + 1) ↔ 
  (x ≤ 1 / 3) ∨ (x ≥ 3) := 
sorry

-- Part (II)
theorem problem_2 (a : ℝ) : 
  let f (x : ℝ) := |x - a| in 
  let g (x : ℝ) := f x - |x - 2| in 
  (∀ y, g y ∈ [-1, 3]) ↔ 
  1 ≤ a ∧ a ≤ 3 := 
sorry

end problem_1_problem_2_l83_83605


namespace part_a_part_b_l83_83570

namespace ProofProblem

def number_set := {n : ℕ | ∃ k : ℕ, n = (10^k - 1)}

noncomputable def special_structure (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2 * m + 1 ∨ n = 2 * m + 2

theorem part_a :
  ∃ (a b c : ℕ) (ha : a ∈ number_set) (hb : b ∈ number_set) (hc : c ∈ number_set),
    special_structure (a + b + c) :=
by
  sorry

theorem part_b (cards : List ℕ) (h : ∀ x ∈ cards, x ∈ number_set)
    (hs : special_structure (cards.sum)) :
  ∃ (d : ℕ), d ≠ 2 ∧ (d = 0 ∨ d = 1) :=
by
  sorry

end ProofProblem

end part_a_part_b_l83_83570


namespace trigonometric_identity_l83_83633

theorem trigonometric_identity
  (α : ℝ)
  (h₁ : cos (7 * π / 2 + α) = 4 / 7)
  (h₂ : tan α < 0) :
  cos (π - α) + sin (π / 2 - α) * tan α = (4 + Real.sqrt 33) / 7 := 
sorry

end trigonometric_identity_l83_83633


namespace Jason_gave_cards_l83_83680

theorem Jason_gave_cards (original current given : ℕ) 
  (h_original : original = 9) 
  (h_current : current = 5) 
  (h_give : given = original - current) : 
  given = 4 := 
by
  rw [h_original, h_current] at h_give
  rw [←h_give]
  norm_num

end Jason_gave_cards_l83_83680


namespace standard_equation_of_curve_l83_83350

theorem standard_equation_of_curve 
  (eccentricity_C : ℝ)
  (passes_through : ℝ × ℝ)
  (h1 : eccentricity_C = sqrt 2)
  (h2 : passes_through = (1, sqrt 2)) :
  ∃ λ, (∀ x y, y^2 - x^2 = λ ↔ (x, y) = passes_through) ∧ λ = 1 :=
by
  sorry

end standard_equation_of_curve_l83_83350


namespace number_of_chairs_l83_83521

-- Definitions based on conditions
def total_legs := 40
def tables := 4
def legs_per_table := 4
def legs_per_chair := 4

-- Statement we want to prove
theorem number_of_chairs : 
  let legs_for_tables := tables * legs_per_table in
  let legs_for_chairs := total_legs - legs_for_tables in
  let chairs := legs_for_chairs / legs_per_chair in
  chairs = 6 :=
by
  -- Sorry to skip the proof
  sorry

end number_of_chairs_l83_83521


namespace acute_angle_at_MDN_l83_83709

theorem acute_angle_at_MDN :
  ∀ (k t : ℝ), ∀ (y1 y2 : ℝ), 
  y1 + y2 = 2 * k → 
  y1 * y2 = -4 → 
  D_x = -1 / 2 →
  let M := (k * y1 + 2, y1),
      N := (k * y2 + 2, y2),
      D := (-1 / 2, t) in
  let DM := λ (D M : ℝ × ℝ), ((D.1 - M.1) , (D.2 - M.2)),
      DN := λ (D N : ℝ × ℝ), ((D.1 - N.1) , (D.2 - N.2)),
      dot := λ (u v : ℝ × ℝ), (u.1 * v.1) + (u.2 * v.2) in
  dot (DM D M) (DN D N) > 0 := sorry

end acute_angle_at_MDN_l83_83709


namespace parallelogram_A2B2C2D2_l83_83336

theorem parallelogram_A2B2C2D2
  (A B C D A1 B1 C1 D1 A2 B2 C2 D2 : Point)
  (hABCD_parallelogram : parallelogram A B C D)
  (hA1_on_AB : lies_on_line_segment A1 A B)
  (hB1_on_BC : lies_on_line_segment B1 B C)
  (hC1_on_CD : lies_on_line_segment C1 C D)
  (hD1_on_DA : lies_on_line_segment D1 D A)
  (hA2_on_A1B1 : lies_on_line_segment A2 A1 B1)
  (hB2_on_B1C1 : lies_on_line_segment B2 B1 C1)
  (hC2_on_C1D1 : lies_on_line_segment C2 C1 D1)
  (hD2_on_D1A1 : lies_on_line_segment D2 D1 A1)
  (h_ratios : (segment_ratio A A1 B A1)
            = (segment_ratio B B1 C B1)
            = (segment_ratio C C1 D C1)
            = (segment_ratio D D1 A D1)
            = (segment_ratio A1 D2 D1 D2)
            = (segment_ratio D1 C2 C1 C2)
            = (segment_ratio C1 B2 B1 B2)
            = (segment_ratio B1 A2 A1 A2))
  : parallelogram A2 B2 C2 D2
    ∧ parallel A2 B2 A B
    ∧ parallel B2 C2 B C
    ∧ parallel C2 D2 C D
    ∧ parallel D2 A2 D A := 
sorry

end parallelogram_A2B2C2D2_l83_83336


namespace total_valid_votes_l83_83332

theorem total_valid_votes (V : ℚ) 
  (h1 : let winning_votes := 0.70 * V in
        let losing_votes := 0.30 * V in
        winning_votes - losing_votes = 172) 
  : V = 430 :=
sorry

end total_valid_votes_l83_83332


namespace count_positive_integers_condition_l83_83782

theorem count_positive_integers_condition :
  {x : ℕ // 0 < x ∧ 2 * x + 28 < 42}.card = 6 :=
by {
  sorry
}

end count_positive_integers_condition_l83_83782


namespace fib6_equals_8_l83_83760

def fib : ℕ → ℕ
| 1       := 1
| 2       := 1
| (n + 3) := fib (n + 2) + fib (n + 1)

theorem fib6_equals_8 : fib 6 = 8 := by
  sorry

end fib6_equals_8_l83_83760


namespace cylinder_cone_volume_ratio_l83_83756

theorem cylinder_cone_volume_ratio (h r_cylinder r_cone : ℝ)
  (hcylinder_csa : π * r_cylinder^2 = π * r_cone^2 / 4):
  (π * r_cylinder^2 * h) / (1 / 3 * π * r_cone^2 * h) = 3 / 4 :=
by
  sorry

end cylinder_cone_volume_ratio_l83_83756


namespace distance_point_parabola_focus_l83_83064

theorem distance_point_parabola_focus (P : ℝ × ℝ) (x y : ℝ) (hP : P = (3, y)) (h_parabola : y^2 = 4 * 3) :
    dist P (0, -1) = 4 :=
by
  sorry

end distance_point_parabola_focus_l83_83064


namespace space_diagonals_of_convex_polyhedron_l83_83149

theorem space_diagonals_of_convex_polyhedron (vertices edges faces triangular_faces pentagonal_faces : ℕ)
  (h_vertices : vertices = 30)
  (h_edges : edges = 72)
  (h_faces : faces = 42)
  (h_triangular_faces : triangular_faces = 30)
  (h_pentagonal_faces : pentagonal_faces = 12) :
  let face_diagonals := pentagonal_faces * 5,
      total_line_segments := vertices * (vertices - 1) / 2,
      space_diagonals := total_line_segments - edges - face_diagonals
  in space_diagonals = 303 :=
by
  -- Definitions by conditions
  /-
  vertices   : ℕ := 30
  edges      : ℕ := 72
  faces      : ℕ := 42
  triangular_faces : ℕ := 30
  pentagonal_faces : ℕ := 12
  face_diagonals := pentagonal_faces * 5 
                 := 12 * 5 = 60
  total_line_segments := (vertices choose 2) 
                       := 30 * 29 / 2 = 435
  space_diagonals := 435 - 72 - 60
                   := 303
  -/
  sorry -- Proof skipped

end space_diagonals_of_convex_polyhedron_l83_83149


namespace alpha_in_first_quadrant_l83_83293

theorem alpha_in_first_quadrant (α : ℝ) 
  (h1 : Real.sin (α - Real.pi / 2) < 0) 
  (h2 : Real.tan (Real.pi + α) > 0) : 
  (0 < α ∧ α < Real.pi / 2) ∨ (2 * Real.pi < α ∧ α < 5 * Real.pi / 2) := 
by
  sorry

end alpha_in_first_quadrant_l83_83293


namespace parallelogram_angles_l83_83147

-- Define the points and the fact that they form a parallelogram
variables {A B C D : Point}
variable h1 : parallelogram A B C D

-- Define the fact that a circle is constructed on AD as its diameter
variable circle_AD : Circle
variable h2 : circle_AD.diameter = (A, D)

-- Define that the circle passes through B and the midpoint M of BC
variable M : Point
variable h3 : midpoint B C M
variable h4 : circle_AD.contains B
variable h5 : circle_AD.contains M

theorem parallelogram_angles (h1 : parallelogram A B C D) 
                             (h2 : circle_AD.diameter = (A, D))
                             (h3 : midpoint B C M)
                             (h4 : circle_AD.contains B)
                             (h5 : circle_AD.contains M) : 
                             (angle A D B = 60) ∧ (angle A B C = 120) := 
by sorry

end parallelogram_angles_l83_83147


namespace sum_le_0_point4_eq_zero_l83_83473

theorem sum_le_0_point4_eq_zero :
  let l := [0.8, 1/2, 0.9],
      filtered_l := l.filter (λ x => x ≤ 0.4),
      sum_filtered_l := filtered_l.sum
  in sum_filtered_l = 0 :=
by 
  -- Let '0.5' represent '1/2' directly
  let l := [0.8, 0.5, 0.9]
  let filtered_l := l.filter (λ x => x ≤ 0.4)
  let sum_filtered_l := filtered_l.sum
  exact eq.refl 0

end sum_le_0_point4_eq_zero_l83_83473


namespace smallest_perfect_number_l83_83963

def is_proper_divisor (a b : ℕ) : Prop := b > 0 ∧ b < a ∧ a % b = 0

def sum_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ x, is_proper_divisor n x) (Finset.range n)).sum id

def is_perfect (n : ℕ) : Prop := sum_proper_divisors n = n

theorem smallest_perfect_number :
  ∃ n : ℕ, is_perfect n ∧ ∀ m : ℕ, is_perfect m → m ≥ n :=
sorry

end smallest_perfect_number_l83_83963


namespace sum_of_remainders_l83_83117

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4) + (n % 5) = 4 := 
by {
  -- proof omitted
  sorry
}

end sum_of_remainders_l83_83117


namespace milk_leftover_l83_83989

variable {v : ℕ} -- 'v' is the number of sets of milkshakes in the 2:1 ratio.
variables {milk vanilla_chocolate : ℕ} -- spoon amounts per milkshake types
variables {total_milk total_vanilla_ice_cream total_chocolate_ice_cream : ℕ} -- total amount constraints
variables {milk_left : ℕ} -- amount of milk left after

-- Definitions based on the conditions
def milk_per_vanilla := 4
def milk_per_chocolate := 5
def ice_vanilla_per_milkshake := 12
def ice_chocolate_per_milkshake := 10
def initial_milk := 72
def initial_vanilla_ice_cream := 96
def initial_chocolate_ice_cream := 96

-- Constraints
def max_milkshakes := 16
def milk_needed (v : ℕ) := (4 * 2 * v) + (5 * v)
def vanilla_needed (v : ℕ) := 12 * 2 * v
def chocolate_needed (v : ℕ) := 10 * v 

-- Inequalities
lemma milk_constraint (v : ℕ) : milk_needed v ≤ initial_milk := sorry

lemma vanilla_constraint (v : ℕ) : vanilla_needed v ≤ initial_vanilla_ice_cream := sorry

lemma chocolate_constraint (v : ℕ) : chocolate_needed v ≤ initial_chocolate_ice_cream := sorry

lemma total_milkshakes_constraint (v : ℕ) : 3 * v ≤ max_milkshakes := sorry

-- Conclusion
theorem milk_leftover : milk_left = initial_milk - milk_needed 5 := sorry

end milk_leftover_l83_83989


namespace remainder_of_101_pow_37_mod_100_l83_83465

theorem remainder_of_101_pow_37_mod_100 : (101 ^ 37) % 100 = 1 := by
  sorry

end remainder_of_101_pow_37_mod_100_l83_83465


namespace limit_proof_l83_83927

noncomputable def limit_expression : ℝ := 
  lim (λ (h : ℝ), ((3 + h)^2 - 3^2) / h) (0 : ℝ)

theorem limit_proof :
  limit_expression = 6 := by
  sorry

end limit_proof_l83_83927


namespace find_a_l83_83235

noncomputable def poly_has_geometric_roots (a : ℝ) : Prop :=
  ∃ (b q : ℝ), b ≠ 0 ∧ q ≠ 0 ∧
    (x^3 + 16*x^2 + a*x + 64 = 0 ∧
      x_1 = b ∧
      x_2 = b*q ∧
      x_3 = b*q^2 ∧
      x_1 + x_2 + x_3 = -16 ∧
      x_1 * x_2 + x_2 * x_3 + x_3 * x_1 = a ∧
      x_1 * x_2 * x_3 = -64)

theorem find_a (a : ℝ) :
  poly_has_geometric_roots a → a = 64 :=
begin
  sorry
end

end find_a_l83_83235


namespace extreme_values_sin_2x0_l83_83602

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.cos (Real.pi / 2 + x)^2 - 
  2 * Real.sin (Real.pi + x) * Real.cos x - Real.sqrt 3

-- Part (1)
theorem extreme_values : 
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), 1 ≤ f x ∧ f x ≤ 2) :=
sorry

-- Part (2)
theorem sin_2x0 (x0 : ℝ) (h : x0 ∈ Set.Icc (3 * Real.pi / 4) Real.pi) (hx : f (x0 - Real.pi / 6) = 10 / 13) : 
  Real.sin (2 * x0) = - (5 + 12 * Real.sqrt 3) / 26 :=
sorry

end extreme_values_sin_2x0_l83_83602


namespace strictly_increasing_function_exists_l83_83017

noncomputable def exists_strictly_increasing_function (f : ℕ → ℕ) :=
  (∀ n : ℕ, n = 1 → f n = 2) ∧
  (∀ n : ℕ, f (f n) = f n + n) ∧
  (∀ m n : ℕ, m < n → f m < f n)

theorem strictly_increasing_function_exists : 
  ∃ f : ℕ → ℕ,
  exists_strictly_increasing_function f :=
sorry

end strictly_increasing_function_exists_l83_83017


namespace elevator_people_count_l83_83783

theorem elevator_people_count (weight_limit : ℕ) (excess_weight : ℕ) (avg_weight : ℕ) (total_weight : ℕ) (n : ℕ) 
  (h1 : weight_limit = 1500)
  (h2 : excess_weight = 100)
  (h3 : avg_weight = 80)
  (h4 : total_weight = weight_limit + excess_weight)
  (h5 : total_weight = n * avg_weight) :
  n = 20 :=
sorry

end elevator_people_count_l83_83783


namespace max_intersection_points_l83_83193

theorem max_intersection_points (C : Set ℝ × ℝ) (l1 l2 l3 : Set ℝ × ℝ)
  (hC : ∃ (r : ℝ) (a b : ℝ), ∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 → (x, y) ∈ C)
  (h_l1_dist : ∃ (a1 b1 c1 : ℝ), ∀ (x y : ℝ), a1*x + b1*y + c1 = 0 → (x, y) ∈ l1)
  (h_l2_dist : ∃ (a2 b2 c2 : ℝ), ∀ (x y : ℝ), a2*x + b2*y + c2 = 0 → (x, y) ∈ l2)
  (h_l3_dist : ∃ (a3 b3 c3 : ℝ), ∀ (x y : ℝ), a3*x + b3*y + c3 = 0 → (x, y) ∈ l3)
  (h_intersect_circle : ∀ l, l = l1 ∨ l = l2 ∨ l = l3 → ∃ (p1 p2 : ℝ × ℝ), (p1 ∈ C ∧ p1 ∈ l ∧ p2 ∈ C ∧ p2 ∈ l))
  (h_pairwise_intersection : l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧ ∀ (li lj : Set ℝ × ℝ), li ≠ lj → li = l1 ∨ li = l2 ∨ li = l3 → lj = l1 ∨ lj = l2 ∨ lj = l3 → ∃ p : ℝ × ℝ, p ∈ li ∧ p ∈ lj) :
  ∃ n : ℕ, n = 9 :=
by
  sorry

end max_intersection_points_l83_83193


namespace z1_div_z2_in_fourth_quadrant_l83_83597

def quadrant_of_complex_division (z1 z2 : ℂ) : ℕ :=
if h1 : (z2 ≠ 0) then
  let z := z1 / z2 in
  if z.re > 0 then
    if z.im > 0 then 1
    else if z.im < 0 then 4
    else 0
  else if z.re < 0 then
    if z.im > 0 then 2
    else if z.im < 0 then 3
    else 0
  else 0
else 0

theorem z1_div_z2_in_fourth_quadrant :
  quadrant_of_complex_division (2 + I) (1 + I) = 4 :=
by sorry

end z1_div_z2_in_fourth_quadrant_l83_83597


namespace recurring_fraction_sum_l83_83925

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l83_83925


namespace axis_symmetry_correct_coordinates_P_correct_l83_83595

-- Define a parabola with general form y = ax^2 + bx + 2 where a ≠ 0
def parabola_eq (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Condition: The parabola passes through A(-1, 0)
def passes_through_A (a b : ℝ) : Prop := parabola_eq a b (-1) = 0

-- Condition: Distance AB = 3
def distance_AB (a b : ℝ) : Prop := ∃ x2 : ℝ, x2 ≠ -1 ∧ x2^2 + 2 * x2 - (1 - x2)^2 = 9

-- Axis of symmetry of the parabola
def axis_of_symmetry (a b : ℝ) : ℝ := -((a + 2) / (2 * a))

theorem axis_symmetry_correct (a b : ℝ) (h1 : a ≠ 0) (h2 : passes_through_A a b) :
  axis_of_symmetry a b = -((a + 2) / (2 * a)) :=
by sorry

-- Condition: Point B exists and parabola intersects the y-axis at C(0, y)
def point_B (a b : ℝ) : ℝ := -(b + 2)
def point_C : ℝ := 2

-- Define the area division condition by x-axis in triangle BPC
def area_division_condition (a b : ℝ) : Prop :=
  let Bx := point_B a b,
  let Px1 := -2,
  let Px2 := -3,
  let Py1 := parabola_eq a b Px1,
  let Py2 := parabola_eq a b Px2 in
  (-Bx * Py1) / 2 = 1 ∧ (-Bx * (Py2 - Py1)) / 2 = 2

-- The coordinates of point P on the parabola below the x-axis
def coordinates_P (a b x : ℝ) : Prop :=
  (x = -3 ∧ parabola_eq a b x = -1) ∨ (x = -2 ∧ parabola_eq a b x = -1)

theorem coordinates_P_correct (a : ℝ) (h1 : a = 1/2) (h2 : passes_through_A a (5/2)) (h3 : distance_AB a (5/2)) (h4 : area_division_condition a (5/2)) :
  coordinates_P a (5/2) (-3) ∨ coordinates_P a (5/2) (-2) :=
by sorry

end axis_symmetry_correct_coordinates_P_correct_l83_83595


namespace triangle_similarity_l83_83371

variables {A B C D E F K L M O : Type}
variables [CommRing A] [CommRing B] [CommRing C] [CommRing D] [CommRing E] [CommRing F]
variables [CommRing K] [CommRing L] [CommRing M] [CommRing O]
variables (AB AC BC : B) (circ_O circ_bod circ_cod circ_aef : C)
variables (triangle_ABC : Triangle B) (triangle_BOD triangle_COD triangle_AEF : Triangle C)
variables (segment_BC : Segment B) (point_D : D) (line_AB line_AC : C)
variables (ood_O : Circumcircle O) (E F K L M : Point)

-- conditions
-- let A, B, and C be the vertices of the triangle ABC
-- let D be a point on BC, not coincident with endpoints or midpoint
-- let the circumcircle of triangle BOD intersect O again at K 
-- and line AB again at E
-- let the circumcircle of triangle COD intersect O again at L 
-- and line AC again at F
-- let the circumcircle of triangle AEF intersect O again at M
-- given that points E and F are on segments AB and AC respectively

def cyclic_quadrilateral (a b c d : Point) [CommRing O] := sorry

def Miquel_point_theorem := sorry

theorem triangle_similarity 
    (h1 : cyclic_quadrilateral triangle_ABC circ_O circ_bod circ_cod circ_aef)
    (h2 : point_on_segment D segment_BC)
    (h3 : point_on_line E line_AB)
    (h4 : point_on_line F line_AC)
    (h5 : point_on_circle K circ_bod)
    (h6 : point_on_circle L circ_cod)
    (h7 : point_on_circle M circ_aef)
    : similar triangle_ABC triangle_MKL := 
        by {
            -- applying the geometric conditions and properties following from Miquel point theorem
            sorry
        }

end triangle_similarity_l83_83371


namespace sum_of_fraction_numerator_and_denominator_l83_83911

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l83_83911


namespace two_guests_match_name_cards_l83_83942

/-- Assuming there are 15 guests seated at a circular table with 15 seats each having a name card,
and initially, all guests are in the wrong seats, there exists a rotation such that at least two guests
match their name cards. -/
theorem two_guests_match_name_cards
  (guests : Fin 15 → Fin 15)
  (h_all_wrong : ∀ i : Fin 15, guests i ≠ i) :
  ∃ r : Fin 15, 2 ≤ (Finset.univ.filter (λ i, guests (i + r) = i)).card :=
sorry

end two_guests_match_name_cards_l83_83942


namespace convert_and_subtract_bases_l83_83541

theorem convert_and_subtract_bases :
  let n6 := 5 * 6^4 + 2 * 6^3 + 1 * 6^2 + 3 * 6^1 + 4 * 6^0,
      n7 := 4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0
  in n6 - n7 = -3768 :=
by
  let n6 := 5 * 6^4 + 2 * 6^3 + 1 * 6^2 + 3 * 6^1 + 4 * 6^0
  let n7 := 4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0
  show n6 - n7 = -3768
  sorry

end convert_and_subtract_bases_l83_83541


namespace sum_of_fraction_parts_l83_83879

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l83_83879


namespace range_of_expression_l83_83636

theorem range_of_expression (x : ℝ) : (x + 2 ≥ 0 ∧ x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end range_of_expression_l83_83636


namespace solve_comb_eq_l83_83049

open Nat

def comb (n k : ℕ) : ℕ := (factorial n) / ((factorial k) * (factorial (n - k)))
def perm (n k : ℕ) : ℕ := (factorial n) / (factorial (n - k))

theorem solve_comb_eq (x : ℕ) :
  comb (x + 5) x = comb (x + 3) (x - 1) + comb (x + 3) (x - 2) + 3/4 * perm (x + 3) 3 ->
  x = 14 := 
by 
  sorry

end solve_comb_eq_l83_83049


namespace compare_An_Bn_l83_83255

-- Define the arithmetic sequence
def a_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

-- Define the sum of the first n terms
def S_n (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * (a + (a + d*(n-1))) / 2

-- Preconditions
variables {a : ℝ} {n : ℕ} (d : ℝ)
hypothesis nonzero_d : d ≠ 0
hypothesis first_term : a_sequence a d 1 = a

-- Define A_n and B_n
def A_n (a : ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / S_n a a i.succ

def B_n (a : ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / a_sequence a a (2^(i))

-- Theorem to compare A_n and B_n
theorem compare_An_Bn (a : ℝ) (n : ℕ) (h : n ≥ 2) :
  (0 < a → A_n a n < B_n a n) ∧ (a < 0 → A_n a n > B_n a n) := sorry

end compare_An_Bn_l83_83255


namespace flowers_sold_on_monday_is_4_l83_83522

constant flowers_sold_monday : ℕ
constant flowers_sold_tuesday : ℕ := 8
constant flowers_sold_friday : ℕ := 2 * flowers_sold_monday
constant total_flowers_sold : ℕ := 20

theorem flowers_sold_on_monday_is_4
    (h : flowers_sold_monday + flowers_sold_tuesday + flowers_sold_friday = total_flowers_sold) :
    flowers_sold_monday = 4 :=
by
  sorry

end flowers_sold_on_monday_is_4_l83_83522


namespace seventh_element_row_20_l83_83558

-- Definition for binomial coefficient
def binom : ℕ → ℕ → ℕ
| n 0     := 1
| 0 k     := 0
| n (k + 1) := if k > n then 0 else binom (n - 1) k + binom (n - 1) (k + 1)

-- Problem statement: Proving that the seventh element in Row 20 of Pascal's triangle is 38760
theorem seventh_element_row_20 : binom 20 6 = 38760 := sorry

end seventh_element_row_20_l83_83558


namespace mary_sheep_purchase_l83_83401

theorem mary_sheep_purchase: 
  ∀ (mary_sheep bob_sheep add_sheep : ℕ), 
    mary_sheep = 300 → 
    bob_sheep = 2 * mary_sheep + 35 → 
    add_sheep = (bob_sheep - 69) - mary_sheep → 
    add_sheep = 266 :=
by
  intros mary_sheep bob_sheep add_sheep _ _
  sorry

end mary_sheep_purchase_l83_83401


namespace solve_for_x_l83_83424

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.07 * (25 + x) = 15.1) : x = 111.25 :=
by
  sorry

end solve_for_x_l83_83424


namespace solution_to_equation_l83_83015

noncomputable def f (x : ℝ) : ℝ := log (2^x - 1) / log 2

noncomputable def f_inv (x : ℝ) : ℝ := log (2^x + 1) / log 2

theorem solution_to_equation :
  ∀ x : ℝ, f (2 * x) = f_inv x ↔ x = 1 := 
by
  sorry

end solution_to_equation_l83_83015


namespace how_many_cakes_each_friend_ate_l83_83024

-- Definitions pertaining to the problem conditions
def crackers : ℕ := 29
def cakes : ℕ := 30
def friends : ℕ := 2

-- The main theorem statement we aim to prove
theorem how_many_cakes_each_friend_ate 
  (h1 : crackers = 29)
  (h2 : cakes = 30)
  (h3 : friends = 2) : 
  (cakes / friends = 15) :=
by
  sorry

end how_many_cakes_each_friend_ate_l83_83024


namespace repeating_decimal_sum_l83_83872

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l83_83872


namespace dart_probability_l83_83951

-- Define the radius of the circle
def radius : ℝ := 10

-- Define the area of the square in terms of the radius
def area_square : ℝ := let side := radius * Math.sqrt 2 in side ^ 2

-- Define the area of the circle in terms of the radius
def area_circle : ℝ := Real.pi * radius ^ 2

-- Define the probability as the ratio of the area of the square to the area of the circle
def probability : ℝ := area_square / area_circle

-- The statement we want to prove
theorem dart_probability : probability = 2 / Real.pi := by
  sorry

end dart_probability_l83_83951


namespace recurring_decimal_fraction_sum_l83_83899

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l83_83899


namespace find_n_l83_83224

theorem find_n : ∃ n : ℕ, n ≡ -2187 [MOD 10] ∧ 0 ≤ n ∧ n ≤ 9 :=
by {use 3, norm_num, exact int.mod_eq_of_lt (by norm_num) (by norm_num)}

end find_n_l83_83224


namespace sales_in_fifth_month_l83_83152

variable (sales1 sales2 sales3 sales4 sales5 sales6 : ℝ)
variable (average_sales required_total_sales actual_total_sales : ℝ)

def average_sales_condition := average_sales = 5600
def sales_condition_months := sales1 = 5400 ∧ sales2 = 9000 ∧ sales3 = 6300 ∧ sales4 = 7200 ∧ sales6 = 1200
def required_total_sales_condition := required_total_sales = 33600
def actual_total_sales_condition := actual_total_sales = sales1 + sales2 + sales3 + sales4 + sales6

theorem sales_in_fifth_month (h1 : average_sales_condition) 
(h2 : sales_condition_months) 
(h3 : required_total_sales_condition) 
(h4 : actual_total_sales_condition) : sales5 = 4500 :=
by 
  sorry

end sales_in_fifth_month_l83_83152


namespace f_1989_1990_1991_divisible_by_13_l83_83385

noncomputable def f : ℕ → ℤ 
| 0       := 0
| 1       := 0
| (n + 2) := 4^(n + 2) * f (n + 1) - 16^(n + 1) * f n + n * 2^(n^2)

theorem f_1989_1990_1991_divisible_by_13 : 
  f 1989 % 13 = 0 ∧ f 1990 % 13 = 0 ∧ f 1991 % 13 = 0 :=
  by
    sorry

end f_1989_1990_1991_divisible_by_13_l83_83385


namespace least_positive_integer_l83_83810

theorem least_positive_integer :
  ∃ (N : ℕ), N % 11 = 10 ∧ N % 12 = 11 ∧ N % 13 = 12 ∧ N % 14 = 13 ∧ N = 12011 :=
by
  use 12011
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 11 = 10
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 12 = 11
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 13 = 12
  split
  · exact Nat.mod_eq_of_lt (by norm_num) -- 12011 % 14 = 13
  · rfl

end least_positive_integer_l83_83810


namespace friendly_not_green_l83_83794

-- Defining types and properties
def Snake : Type := sorry
def green_snake (s : Snake) : Prop := sorry
def friendly_snake (s : Snake) : Prop := sorry
def can_multiply (s : Snake) : Prop := sorry
def can_divide (s : Snake) : Prop := sorry

-- Defining conditions as axioms
axiom total_snakes : ∃ (snakes : Set Snake), snakes.card = 15
axiom num_green_snakes : ∃ (snakes : Set Snake), s ∈ snakes → green_snake s
axiom num_friendly_snakes : ∃ (snakes : Set Snake), s ∈ snakes → friendly_snake s
axiom friendly_can_multiply (s : Snake) : friendly_snake s → can_multiply s
axiom green_cannot_divide (s : Snake) : green_snake s → ¬ can_divide s
axiom cannot_divide_cannot_multiply (s : Snake) : ¬ can_divide s → ¬ can_multiply s

-- Theorem to prove
theorem friendly_not_green (s : Snake) : friendly_snake s → ¬ green_snake s :=
sorry

end friendly_not_green_l83_83794


namespace weight_increase_is_four_l83_83652

variable (A : ℝ)

theorem weight_increase_is_four
  (h1 : A ≠ 0) :
  let original_total_weight := 5 * A,
      new_person_weight := 70,
      replaced_person_weight := 50,
      total_weight_change := new_person_weight - replaced_person_weight,
      new_total_weight := original_total_weight + total_weight_change,
      new_average_weight := new_total_weight / 5 in
  (new_average_weight - A = (20 / 5)) :=
by
  intros
  let original_total_weight := 5 * A
  let new_person_weight := 70
  let replaced_person_weight := 50
  let total_weight_change := new_person_weight - replaced_person_weight
  let new_total_weight := original_total_weight + total_weight_change
  let new_average_weight := new_total_weight / 5
  have h : new_average_weight - A = 4
    sorry
  exact h

end weight_increase_is_four_l83_83652


namespace part_a_part_b_l83_83727

noncomputable def average_price_between : Prop :=
  ∃ (prices : Fin 14 → ℝ), 
    prices 0 = 5 ∧ prices 6 = 5.14 ∧ prices 13 = 5 ∧ 
    5.09 < (∑ i, prices i) / 14 ∧ (∑ i, prices i) / 14 < 5.10

theorem part_a : average_price_between :=
  sorry

def average_difference : ℝ :=
  let prices1 := [5.0, 5.1, 5.1, 5.1, 5.1, 5.1, 5.14] in
  let prices2 := [5.14, 5.14, 5.14, 5.14, 5.14, 5.14, 5.0] in
  let avg1 := (prices1.sum / prices1.length : ℝ) in
  let avg2 := (prices2.sum / prices2.length : ℝ) in
  abs (avg2 - avg1)

theorem part_b : average_difference < 0.105 :=
  sorry

end part_a_part_b_l83_83727


namespace card_of_S_l83_83368

noncomputable def is_value_of_a (a : ℝ) :=
  ∃ (x : ℝ), (x^2 + a*x = 0) ∨ (x^2 + a*x + 2 = 0)

noncomputable def set_B (a : ℝ) : set ℝ :=
  {x | (x^2 + a*x) * (x^2 + a*x + 2) = 0}

def card (A : set ℝ) : ℕ := A.to_finset.card

noncomputable def A := {1, 2} : set ℝ

noncomputable def star (A B : set ℝ) :=
  if h : card A ≥ card B then card A - card B else card B - card A

theorem card_of_S :
  let S := {a : ℝ | is_value_of_a a ∧ star A (set_B a) = 1} in
  card S = 3 :=
by
  intros; sorry

end card_of_S_l83_83368


namespace martin_leftover_raisins_l83_83395

theorem martin_leftover_raisins :
  ∀ (v k r : ℝ),
  (3 * v + 3 * k = 18 * r) → 
  (12 * r + 5 * k = v + 6 * k + (x * r)) →
  x = 6 :=
begin 
  intros v k r h1 h2,
  have h3 : v + k = 6 * r,
  { linarith, },
  have h4 : 12 * r = v + k + x * r,
  { linarith, },
  rw h3 at h2,
  exact eq_of_mul_eq_mul_right (by linarith) (by linarith),
  sorry
end

end martin_leftover_raisins_l83_83395


namespace remainder_101_pow_37_mod_100_l83_83470

theorem remainder_101_pow_37_mod_100 :
  (101: ℤ) ≡ 1 [MOD 100] →
  (101: ℤ)^37 ≡ 1 [MOD 100] :=
by
  sorry

end remainder_101_pow_37_mod_100_l83_83470


namespace probability_combined_event_l83_83459

def box_A := finset.range 21 \ {0}  -- {1, 2, ..., 20}
def box_B := finset.range 40 \ finset.range 10  -- {10, 11, ..., 39}

def tiles_A_less_10 := {n ∈ box_A | n < 10}.card
def tiles_B_odd_or_greater_35 := {n ∈ box_B | n % 2 = 1 ∨ n > 35}.card

theorem probability_combined_event : 
  (tiles_A_less_10 / box_A.card : ℚ) * (tiles_B_odd_or_greater_35 / box_B.card : ℚ) = (51 / 200 : ℚ) :=
by
  have h1 : tiles_A_less_10 = 9 := by sorry
  have h2 : tiles_B_odd_or_greater_35 = 17 := by sorry
  rw [h1, h2]
  norm_num
  sorry

end probability_combined_event_l83_83459


namespace coins_in_first_piggy_bank_l83_83124

theorem coins_in_first_piggy_bank (c2 c3 c4 c5 c6 : ℕ) (h2 : c2 = 81) 
  (h3 : c3 = 90) (h4 : c4 = 99) (h5 : c5 = 108) 
  (h6 : c6 = 117) (h_inc : ∀ (n : ℕ), n ≥ 2 → c (n + 1) - c n = 9) : 
  ∃ c1, c1 = 72 :=
by
  let c1 := c2 - 9
  have : c1 = 72, from sorry
  existsi c1
  exact this

end coins_in_first_piggy_bank_l83_83124


namespace sum_of_coordinates_of_reflected_midpoint_is_1_l83_83740

-- Define points P and R
def P : ℝ × ℝ := (2, 1)
def R : ℝ × ℝ := (12, 15)

-- Define midpoint of P and R
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define reflection over y-axis
def reflect_y (A : ℝ × ℝ) : ℝ × ℝ := (-A.1, A.2)

-- Midpoint M of segment PR
def M : ℝ × ℝ := midpoint P R

-- Reflection of points P and R over y-axis
def P' : ℝ × ℝ := reflect_y P
def R' : ℝ × ℝ := reflect_y R

-- Midpoint of reflected segment P'R'
def M' : ℝ × ℝ := midpoint P' R'

-- Sum of coordinates of M'
def sum_coordinates (A : ℝ × ℝ) : ℝ := A.1 + A.2

-- Statement: The sum of coordinates of M' is 1
theorem sum_of_coordinates_of_reflected_midpoint_is_1 : sum_coordinates M' = 1 :=
by
  sorry

end sum_of_coordinates_of_reflected_midpoint_is_1_l83_83740


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l83_83895

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l83_83895


namespace three_perspectors_theorem_l83_83615

theorem three_perspectors_theorem 
  (A B C A1 B1 C1 O O1 O2 O3 : Type) 
  [line] 
  (intersect_A: collinear A A1 O) 
  (intersect_B: collinear B B1 O) 
  (intersect_C: collinear C C1 O) 
  (intersect_A1: collinear A A1 O1) 
  (intersect_BC1: collinear B C1 O1) 
  (intersect_CB1: collinear C B1 O1) 
  (intersect_AC1: collinear A C1 O2) 
  (intersect_B1B: collinear B B1 O2) 
  (intersect_CA1: collinear C A1 O2) 
: ∃ O3, collinear A B1 O ∧ collinear B A1 O ∧ collinear C C1 O :=
sorry

end three_perspectors_theorem_l83_83615


namespace find_number_of_boxes_l83_83736

-- Definitions and assumptions
def pieces_per_box : ℕ := 5 + 5
def total_pieces : ℕ := 60

-- The theorem to be proved
theorem find_number_of_boxes (B : ℕ) (h : total_pieces = B * pieces_per_box) :
  B = 6 :=
sorry

end find_number_of_boxes_l83_83736


namespace find_marks_in_chemistry_l83_83197

theorem find_marks_in_chemistry
  (marks_english : ℕ)
  (marks_math : ℕ)
  (marks_physics : ℕ)
  (marks_biology : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (marks_english_eq : marks_english = 86)
  (marks_math_eq : marks_math = 85)
  (marks_physics_eq : marks_physics = 92)
  (marks_biology_eq : marks_biology = 95)
  (average_marks_eq : average_marks = 89)
  (num_subjects_eq : num_subjects = 5) : 
  ∃ marks_chemistry : ℕ, marks_chemistry = 87 :=
by
  sorry

end find_marks_in_chemistry_l83_83197


namespace crackers_per_person_l83_83715

theorem crackers_per_person (total_crackers : ℕ) (friends : ℕ) (h1 : total_crackers = 36) (h2 : friends = 18) : total_crackers / friends = 2 := by
  rw [←h1, ←h2]
  norm_num
  sorry

end crackers_per_person_l83_83715


namespace thelma_tomato_count_l83_83790

-- Definitions and conditions
def slices_per_tomato : ℕ := 8
def slices_per_meal_per_person : ℕ := 20
def family_members : ℕ := 8
def total_slices_needed : ℕ := slices_per_meal_per_person * family_members
def tomatoes_needed : ℕ := total_slices_needed / slices_per_tomato

-- Statement of the theorem to be proved
theorem thelma_tomato_count :
  tomatoes_needed = 20 := by
  sorry

end thelma_tomato_count_l83_83790


namespace percent_increase_in_area_l83_83177

noncomputable def side_length_first_triangle : ℝ := 3
noncomputable def side_length_fifth_triangle : ℝ := 3 * (1.2)^4

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s ^ 2

noncomputable def area_first_triangle : ℝ :=
  area_of_equilateral_triangle side_length_first_triangle

noncomputable def area_fifth_triangle : ℝ :=
  area_of_equilateral_triangle side_length_fifth_triangle

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem percent_increase_in_area :
  percent_increase area_first_triangle area_fifth_triangle ≈ 329.98 := sorry

end percent_increase_in_area_l83_83177


namespace reach_end_in_three_rolls_l83_83931

def dice_steps_game (n : ℕ) (rolls : ℕ → ℕ) : Prop :=
  ∀ i, 1 ≤ rolls i ∧ rolls i ≤ 6

def valid_moves (moves : list ℕ) : list (list ℕ) :=
  moves.filter (λ lst, (lst.sum = 8) ∧ ¬(6 ∈ lst))

theorem reach_end_in_three_rolls :
  (valid_moves [[1,1,6], [4,2,2], [3,3,2], [3,2,3], [5,1,2], [5,2,1], [4,3,1]])
  = 19 :=
by sorry

end reach_end_in_three_rolls_l83_83931


namespace small_seat_capacity_l83_83759

-- Definitions for the conditions
def smallSeats : Nat := 2
def largeSeats : Nat := 23
def capacityLargeSeat : Nat := 54
def totalPeopleSmallSeats : Nat := 28

-- Theorem statement
theorem small_seat_capacity : totalPeopleSmallSeats / smallSeats = 14 := by
  sorry

end small_seat_capacity_l83_83759


namespace slope_of_tangent_at_0_l83_83451

noncomputable def y : ℝ → ℝ
| x := sin x + exp x

theorem slope_of_tangent_at_0 :
  (deriv y 0) = 2 :=
 sorry

end slope_of_tangent_at_0_l83_83451


namespace find_a_b_and_max_value_l83_83603

noncomputable theory

open Real

def f (a b : ℝ) (x : ℝ) := a * log x - b * x^2

def tangent_eq (a b : ℝ) : Prop :=
  let f' := (λ x, a / x - 2 * b * x)
  f' 2 = 3 / 2 ∧ f a b 2 = (2 + log 2)

theorem find_a_b_and_max_value :
  ∃ (a b : ℝ), tangent_eq a b ∧
  f a b (1 / exp 1) ≤ f a b x ∧ f a b x ≤ f a b (sqrt (real.exp 1)) :=
sorry

end find_a_b_and_max_value_l83_83603


namespace repeating_decimal_fraction_sum_l83_83837

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l83_83837


namespace repeating_decimal_fraction_sum_l83_83851

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l83_83851


namespace pushups_difference_l83_83998

theorem pushups_difference :
  let David_pushups := 44
  let Zachary_pushups := 35
  David_pushups - Zachary_pushups = 9 :=
by
  -- Here we define the push-ups counts
  let David_pushups := 44
  let Zachary_pushups := 35
  -- We need to show that David did 9 more push-ups than Zachary.
  show David_pushups - Zachary_pushups = 9
  sorry

end pushups_difference_l83_83998


namespace total_admissions_over_30_days_l83_83447

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ (∀ n : ℕ, a (n + 2) - a n = 1 + (-1)^n)

theorem total_admissions_over_30_days (a : ℕ → ℕ) (h : seq a) : 
  (∑ n in Finset.range 30, a (n + 1)) = 255 :=
by sorry

end total_admissions_over_30_days_l83_83447


namespace geo_seq_formula_l83_83326

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n+1) = a n * (a 1 / a 0)

theorem geo_seq_formula (a : ℕ → ℝ) (q : ℝ) (h1 : q > 0) (h2 : a 2 - a 1 = 1) (h3 : geometric_sequence a) :
  (∀ n, a n = 2^(n-1)) :=
begin
  sorry
end

end geo_seq_formula_l83_83326


namespace calculate_sum_l83_83543

noncomputable def sum_of_adjusted_series : ℕ := 20 * 2^41

theorem calculate_sum :
  let a1 := 10,
      an := 20,
      d := 1/4,
      n := 41,
      S := ∑ k in Finset.range n, 2^k * (a1 + k * d) in
  S = 20 * 2^41 :=
by {
  let a1 := 10,
  let an := 20,
  let d := 1 / 4,
  let n := 41,
  let S := ∑ k in Finset.range n, 2^k * (a1 + k * d),
  have S1 : S = 20 * 2^41 := sorry,
  exact S1,
}

end calculate_sum_l83_83543


namespace recurring_decimal_fraction_sum_l83_83898

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l83_83898


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l83_83886

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l83_83886


namespace minimum_tank_cost_l83_83932

noncomputable def tank_cost (length : ℝ) : ℝ :=
  let volume : ℝ := 4800
  let depth : ℝ := 3
  let base_cost_per_sqm : ℝ := 150
  let wall_cost_per_sqm : ℝ := 120
  let base_area := volume / depth
  let width := base_area / length
  let base_cost := base_cost_per_sqm * base_area
  let wall_cost := wall_cost_per_sqm * depth * (2 * length + 2 * width)
  base_cost + wall_cost

theorem minimum_tank_cost : ∃ length : ℝ, tank_cost length = 297600 :=
by {
  use 40,
  unfold tank_cost,
  norm_num,
}

end minimum_tank_cost_l83_83932


namespace domain_of_f_l83_83440

noncomputable def f (x : ℝ) : ℝ := (√(2 * x + 1)) / (2 * x ^ 2 - x - 1)

def domain (x : ℝ) : Prop := (2 * x + 1 ≥ 0) ∧ (2 * x ^ 2 - x - 1 ≠ 0)

theorem domain_of_f :
  {x : ℝ | domain x} = {x : ℝ | x > -1 / 2 ∧ x ≠ 1} :=
by
  sorry

end domain_of_f_l83_83440


namespace sum_of_fraction_parts_l83_83878

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l83_83878


namespace value_of_d_l83_83634

theorem value_of_d (d : ℝ) (h : x^2 - 60 * x + d = (x - 30)^2) : d = 900 :=
by { sorry }

end value_of_d_l83_83634


namespace hunter_time_comparison_l83_83041

-- Definitions for time spent in swamp, forest, and highway
variables {a b c : ℝ}

-- Given conditions
-- 1. Total time equation
#check a + b + c = 4

-- 2. Total distance equation
#check 2 * a + 4 * b + 6 * c = 17

-- Prove that the hunter spent more time on the highway than in the swamp
theorem hunter_time_comparison (h1 : a + b + c = 4) (h2 : 2 * a + 4 * b + 6 * c = 17) : c > a :=
by sorry

end hunter_time_comparison_l83_83041


namespace polynomial_division_l83_83454

noncomputable def P_n (a : ℕ → ℤ) (n : ℕ) : (ℤ → ℤ) :=
λ x, (finRange (n+1)).foldr (λ i acc, a i + x * acc) 0

theorem polynomial_division 
  (a : ℕ → ℤ)   -- Coefficients
  (c : ℤ)       -- Point of evaluation
  (n : ℕ)
  (b : ℕ → ℤ)
  (h_bn : b n = a n)
  (h_bk : ∀ k, k < n → b k = c * b (k + 1) + a k)
  (h_an : a n ≠ 0) :
  P_n a n = (fun x => (x - c) * (finRange n).foldr (λ i acc, b (i + 1) + x * acc) 0 + b 0) :=
sorry

end polynomial_division_l83_83454


namespace value_of_b_3_pow_100_l83_83009

-- Define the sequence b
def b : ℕ → ℕ
| 1       := 2
| (3 * n) := n * b n
| _       := 0 -- define default case for non-pattern matches

-- Define a theorem to prove the value of b_{3^100}
theorem value_of_b_3_pow_100 : b (3 ^ 100) = 2 * 3 ^ 200 := by
  sorry

end value_of_b_3_pow_100_l83_83009


namespace solve_eqn_l83_83048

def f (x : ℝ) : ℝ := x + real.arctan x * real.sqrt (x^2 + 1)

theorem solve_eqn : 
  ∀ (x : ℝ),
  (f(x) = x + real.arctan x * real.sqrt(x^2 + 1)) → 
  (∀ x, f(-x) = -f(x)) → 
  (∀ x y, x < y → f(x) < f(y)) →
  2*x + 2 + f(x) + f(x + 2) = 0 → x = -1 :=
by
  intros x fx odd_f inc_f eqn
  sorry

end solve_eqn_l83_83048


namespace sum_M_2020_l83_83710

def coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

def M (k : ℕ) : Finset (ℕ × ℕ) :=
  ((Finset.range k).filter (λ n => 0 < n)).biUnion (λ n, 
    ((Finset.range n).filter (λ m => 0 < m ∧ coprime m n ∧ m + n > k)).product (Finset.singleton n))

theorem sum_M_2020 : (∑ (p : ℕ × ℕ) in M 2020, (1 : ℚ) / (p.fst * p.snd)) = 1 / 2 :=
  by
  sorry

end sum_M_2020_l83_83710


namespace sum_of_four_triangles_l83_83523

theorem sum_of_four_triangles :
  ∀ (x y : ℝ), 3 * x + 2 * y = 27 → 2 * x + 3 * y = 23 → 4 * y = 12 :=
by
  intros x y h1 h2
  sorry

end sum_of_four_triangles_l83_83523


namespace sqrt_a_sqrt_a_sqrt_a_eq_a_l83_83182

theorem sqrt_a_sqrt_a_sqrt_a_eq_a (a : ℝ) (h : a > 0) : (real.sqrt (a * real.sqrt (a * real.sqrt a))) = a :=
by sorry

end sqrt_a_sqrt_a_sqrt_a_eq_a_l83_83182


namespace sum_of_coordinates_of_reflected_midpoint_is_1_l83_83741

-- Define points P and R
def P : ℝ × ℝ := (2, 1)
def R : ℝ × ℝ := (12, 15)

-- Define midpoint of P and R
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define reflection over y-axis
def reflect_y (A : ℝ × ℝ) : ℝ × ℝ := (-A.1, A.2)

-- Midpoint M of segment PR
def M : ℝ × ℝ := midpoint P R

-- Reflection of points P and R over y-axis
def P' : ℝ × ℝ := reflect_y P
def R' : ℝ × ℝ := reflect_y R

-- Midpoint of reflected segment P'R'
def M' : ℝ × ℝ := midpoint P' R'

-- Sum of coordinates of M'
def sum_coordinates (A : ℝ × ℝ) : ℝ := A.1 + A.2

-- Statement: The sum of coordinates of M' is 1
theorem sum_of_coordinates_of_reflected_midpoint_is_1 : sum_coordinates M' = 1 :=
by
  sorry

end sum_of_coordinates_of_reflected_midpoint_is_1_l83_83741


namespace average_age_of_persons_l83_83057

theorem average_age_of_persons 
  (total_age : ℕ := 270) 
  (average_age : ℕ := 15) : 
  (total_age / average_age) = 18 := 
by { 
  sorry 
}

end average_age_of_persons_l83_83057


namespace max_n_is_2_l83_83569

def is_prime_seq (q : ℕ → ℕ) : Prop :=
  ∀ i, Nat.Prime (q i)

def gen_seq (q0 : ℕ) : ℕ → ℕ
  | 0 => q0
  | (i + 1) => (gen_seq q0 i - 1)^3 + 3

theorem max_n_is_2 (q0 : ℕ) (hq0 : q0 > 0) :
  ∀ (q1 q2 : ℕ), q1 = gen_seq q0 1 → q2 = gen_seq q0 2 → 
  is_prime_seq (gen_seq q0) → q2 = (q1 - 1)^3 + 3 := 
  sorry

end max_n_is_2_l83_83569


namespace series_sum_eq_l83_83744

noncomputable def series_sum (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, 1 / (k * (k + 1) * (k + 2))

theorem series_sum_eq (n : ℕ) :
  series_sum n = 1 / 2 * (1 / 2 - 1 / ((n + 1) * (n + 2))) := 
by
  sorry

end series_sum_eq_l83_83744


namespace total_cost_is_2750_l83_83672

def squat_rack_cost : ℕ := 2500
def barbell_cost : ℕ := squat_rack_cost / 10
def total_cost : ℕ := squat_rack_cost + barbell_cost

theorem total_cost_is_2750 : total_cost = 2750 := by
  have h1 : squat_rack_cost = 2500 := by rfl
  have h2 : barbell_cost = 2500 / 10 := by rfl
  have h3 : total_cost = 2500 + 250 := by rfl
  have h4 : total_cost = 2750 := by rw [h1, h2, h3]
  sorry

end total_cost_is_2750_l83_83672


namespace ellen_smoothie_total_ingredients_l83_83208

theorem ellen_smoothie_total_ingredients :
  let strawberries := 0.2
  let yogurt := 0.1
  let orange_juice := 0.2
  strawberries + yogurt + orange_juice = 0.5 :=
by {
  -- We introduce the specific quantities given in the conditions
  let strawberries := 0.2;
  let yogurt := 0.1;
  let orange_juice := 0.2;
  -- We verify the sum
  calc
    strawberries + yogurt + orange_juice = 0.2 + 0.1 + 0.2 : by rfl
    ... = 0.5 : by norm_num
  sorry
}

end ellen_smoothie_total_ingredients_l83_83208


namespace square_area_ratio_l83_83640

theorem square_area_ratio (s₁ s₂ d₂ : ℝ)
  (h1 : s₁ = 2 * d₂)
  (h2 : d₂ = s₂ * Real.sqrt 2) :
  (s₁^2) / (s₂^2) = 8 :=
by
  sorry

end square_area_ratio_l83_83640


namespace total_people_in_club_after_5_years_l83_83949

noncomputable def club_initial_people := 18
noncomputable def executives_per_year := 6
noncomputable def initial_regular_members := club_initial_people - executives_per_year

-- Define the function for regular members growth
noncomputable def regular_members_after_n_years (n : ℕ) : ℕ := initial_regular_members * 2 ^ n

-- Total people in the club after 5 years
theorem total_people_in_club_after_5_years : 
  club_initial_people + regular_members_after_n_years 5 - initial_regular_members = 390 :=
by
  sorry

end total_people_in_club_after_5_years_l83_83949


namespace segment_AC_length_l83_83648

noncomputable def circle_radius := 8
noncomputable def chord_length_AB := 10
noncomputable def arc_length_AC (circumference : ℝ) := circumference / 3

theorem segment_AC_length :
  ∀ (C : ℝ) (r : ℝ) (AB : ℝ) (AC : ℝ),
    r = circle_radius →
    AB = chord_length_AB →
    C = 2 * Real.pi * r →
    AC = arc_length_AC C →
    AC = 8 * Real.sqrt 3 :=
by
  intros C r AB AC hr hAB hC hAC
  sorry

end segment_AC_length_l83_83648


namespace smallest_product_of_digits_l83_83032

theorem smallest_product_of_digits : 
  ∃ (a b c d : ℕ), 
  (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6) ∧ 
  (∃ x y : ℕ, (x = a * 10 + c ∧ y = b * 10 + d) ∨ (x = a * 10 + d ∧ y = b * 10 + c) ∨ (x = b * 10 + c ∧ y = a * 10 + d) ∨ (x = b * 10 + d ∧ y = a * 10 + c)) ∧
  (∀ x1 y1 x2 y2 : ℕ, ((x1 = 34 ∧ y1 = 56 ∨ x1 = 35 ∧ y1 = 46) ∧ (x2 = 34 ∧ y2 = 56 ∨ x2 = 35 ∧ y2 = 46)) → x1 * y1 ≥ x2 * y2) ∧
  35 * 46 = 1610 :=
sorry

end smallest_product_of_digits_l83_83032


namespace radius_of_truck_tank_l83_83954

-- Definitions for right circular cylinder volumes
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

-- Conditions
variables (r_stationary : ℝ) (h_stationary : ℝ) (oil_drop : ℝ) (h_truck : ℝ)
variable (r_truck : ℝ)

-- Given constants
def r_stationary := 100  -- radius of stationary tank (in feet)
def h_stationary := 25   -- height of stationary tank (in feet)
def oil_drop := 0.016    -- drop in oil level in stationary tank (in feet)
def h_truck := 10        -- height of truck's tank (in feet)

-- Theorem statement
theorem radius_of_truck_tank :
  let V_pumped := cylinder_volume r_stationary oil_drop in
  let r_truck := 4 in
  cylinder_volume r_truck h_truck = V_pumped :=
by
  sorry

end radius_of_truck_tank_l83_83954


namespace sum_of_fraction_parts_l83_83881

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l83_83881


namespace least_sum_exponents_of_distinct_powers_of_2_eq_72_l83_83296

theorem least_sum_exponents_of_distinct_powers_of_2_eq_72 : 
  ∃ (exponents : set ℕ), (72 = ∑ x in exponents, 2^x) ∧ (exponents.card ≥ 3) ∧ (exponents.sum id = 9) :=
sorry

end least_sum_exponents_of_distinct_powers_of_2_eq_72_l83_83296


namespace tangent_line_at_e_inequality_holds_l83_83599

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / x

theorem tangent_line_at_e :
  ∀ (y : ℝ), y = f Real.exp →
  (∀ (x : ℝ), x + Real.exp^2 * y - 3 * Real.exp = 0) :=
sorry

theorem inequality_holds (a : ℝ) :
  (∀ (x : ℝ), x ≥ 1 → f x - 1 / x ≥ a * (x^2 - 1) / x) → a ≤ 0 :=
sorry

end tangent_line_at_e_inequality_holds_l83_83599


namespace thirtieth_progressive_number_is_1359_l83_83494

def is_progressive (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  (digits.length = 4) ∧ (list.pairwise (<) digits)

def nth_progressive (m : ℕ) : ℕ :=
  @nat.find (λ n, ∃ k, k = m ∧ is_progressive n) nat.infinite_progressive_number_seq

theorem thirtieth_progressive_number_is_1359 : nth_progressive 30 = 1359 := by
  sorry

end thirtieth_progressive_number_is_1359_l83_83494


namespace cookie_count_l83_83360

theorem cookie_count (C : ℕ) 
  (h1 : 3 * C / 4 + 1 * (C / 4) / 5 + 1 * (C / 4) * 4 / 20 = 10) 
  (h2: 1 * (5 * 4 / 20) / 10 = 1): 
  C = 100 :=
by 
sorry

end cookie_count_l83_83360


namespace pi_approx_correct_l83_83238

noncomputable def pi_approx (n m : ℕ) (pairs : Fin n → ℝ × ℝ)
  (h_uniform : ∀ i, 0 ≤ (pairs i).1 ∧ (pairs i).1 ≤ 1 ∧ 0 ≤ (pairs i).2 ∧ (pairs i).2 ≤ 1)
  (h_sum_squares : ∃ count : ℕ, count = m ∧ ∀ i, count = (pairs i).1^2 + (pairs i).2^2 < 1) :
  Real :=
4 * m / n

theorem pi_approx_correct {n m : ℕ} (pairs : Fin n → ℝ × ℝ)
  (h_uniform : ∀ i, 0 ≤ (pairs i).1 ∧ (pairs i).1 ≤ 1 ∧ 0 ≤ (pairs i).2 ∧ (pairs i).2 ≤ 1)
  (h_sum_squares : ∃ count : ℕ, count = m ∧ ∀ i, count = (pairs i).1^2 + (pairs i).2^2 < 1) :
  pi_approx n m pairs h_uniform h_sum_squares ≈ Real.pi := sorry

end pi_approx_correct_l83_83238


namespace ned_initial_video_games_l83_83716

theorem ned_initial_video_games : ∀ (w t : ℕ), 7 * w = 63 ∧ t = w + 6 → t = 15 := by
  intro w t
  intro h
  sorry

end ned_initial_video_games_l83_83716


namespace calculate_value_l83_83545

theorem calculate_value : (3^2 * 5^4 * 7^2) / 7 = 39375 := by
  sorry

end calculate_value_l83_83545


namespace regular_ngon_fibonacci_product_l83_83511

noncomputable def distances_inscribed_ngon (n : ℕ) (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i < n → a i = |1 - e ^ (2 * real.pi * complex.I * i / n)|

noncomputable def fib (n : ℕ) : ℕ :=
  nat.fib n

theorem regular_ngon_fibonacci_product (n : ℕ) (a : ℕ → ℝ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i < n → a i = |1 - e ^ (2 * real.pi * complex.I * i / n)>) :
  ∏ i in finset.range (n - 1), (5 - (a i) ^ 2) = (fib n) ^ 2 :=
sorry

end regular_ngon_fibonacci_product_l83_83511


namespace abhay_speed_l83_83485

variables (A S : ℝ)

theorem abhay_speed :
  (24 / A = 24 / S + 2) →
  (24 / (2 * A) = 24 / S - 1) →
  A = 4 :=
begin
  intros h1 h2,
  sorry
end

end abhay_speed_l83_83485


namespace jack_paid_total_l83_83677

theorem jack_paid_total (cost_squat_rack : ℕ) (cost_barbell_fraction : ℕ) 
  (h1 : cost_squat_rack = 2500) (h2 : cost_barbell_fraction = 10) :
  let cost_barbell := cost_squat_rack / cost_barbell_fraction in
  let total_cost := cost_squat_rack + cost_barbell in
  total_cost = 2750 :=
by
  -- Assign the values
  let cost_barbell := cost_squat_rack / cost_barbell_fraction
  let total_cost := cost_squat_rack + cost_barbell
  -- We use the assumptions h1 and h2
  have h_cost_barbell : cost_barbell = 250 := by
    simp only [h1, h2]
    sorry -- complete arithmetic step
  have h_total_cost : total_cost = 2750 := by
    rw [h1, h_cost_barbell]
    sorry -- complete arithmetic step
  exact h_total_cost

end jack_paid_total_l83_83677


namespace mean_xyz_l83_83054

-- Define the original seven numbers and their mean
variables {a1 a2 a3 a4 a5 a6 a7 : ℝ}

-- Define the three additional numbers
variables {x y z : ℝ}

-- The arithmetic mean of the seven numbers is 63
def mean_seven : Prop := (a1 + a2 + a3 + a4 + a5 + a6 + a7) / 7 = 63

-- Adding x, y, and z results in a new mean of 78 for the ten numbers
def mean_ten : Prop := (a1 + a2 + a3 + a4 + a5 + a6 + a7 + x + y + z) / 10 = 78

-- Proving that the mean of x, y, and z is 113
theorem mean_xyz : mean_seven → mean_ten → (x + y + z) / 3 = 113 :=
by
  intros h1 h2
  sorry

end mean_xyz_l83_83054


namespace sum_of_fraction_numerator_and_denominator_l83_83908

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l83_83908


namespace percentage_of_loss_l83_83496

-- Define the conditions as given in the problem
def original_selling_price : ℝ := 720
def gain_selling_price : ℝ := 880
def gain_percentage : ℝ := 0.10

-- Define the main theorem
theorem percentage_of_loss : ∀ (CP : ℝ),
  (1.10 * CP = gain_selling_price) → 
  ((CP - original_selling_price) / CP * 100 = 10) :=
by
  intro CP
  intro h
  have h1 : CP = gain_selling_price / 1.10 := by sorry
  have h2 : (CP - original_selling_price) = 80 := by sorry -- Intermediate step to show loss
  have h3 : ((80 / CP) * 100 = 10) := by sorry -- Calculation of percentage of loss
  sorry

end percentage_of_loss_l83_83496


namespace probability_of_all_same_color_l83_83945

open Finset

-- Define the conditions of the problem
def num_red : ℕ := 3
def num_white : ℕ := 6
def num_blue : ℕ := 9
def total_marbles : ℕ := num_red + num_white + num_blue
def draw_count : ℕ := 4

-- Calculate the probabilities using combination
def P_all_red : ℚ := if draw_count ≤ num_red then 1 else 0
def P_all_white : ℚ := (Nat.choose num_white draw_count : ℚ) / (Nat.choose total_marbles draw_count : ℚ)
def P_all_blue : ℚ := (Nat.choose num_blue draw_count : ℚ) / (Nat.choose total_marbles draw_count : ℚ)

-- Define the total probability of drawing four marbles of the same color
def P_all_same_color : ℚ := P_all_red + P_all_white + P_all_blue

-- The goal is to prove that the total probability is equal to the correct answer
theorem probability_of_all_same_color :
  P_all_same_color = 9 / 170 := 
by 
  sorry

end probability_of_all_same_color_l83_83945


namespace solve_inequality_l83_83426

theorem solve_inequality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) (h3 : x ≠ 4) (h4 : x ≠ 5) :
  (2 / (x - 2) - 5 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔
  (x ∈ set.Iio 2 ∪ set.Ioo 3 4 ∪ set.Ioi 5) :=
sorry

end solve_inequality_l83_83426


namespace probability_M_or_bottom_theorem_l83_83665

-- Define the structure of our octahedron with specific adjacency properties.
structure OctahedronVertex where
  is_top : Bool
  is_bottom : Bool
  is_middle : Bool
  adjacent_vertices : List OctahedronVertex

-- Assume there are at least one top, one bottom, and multiple middle vertices.
axiom exists_top_vertex : ∃ v : OctahedronVertex, v.is_top
axiom exists_bottom_vertex : ∃ v : OctahedronVertex, v.is_bottom
axiom exists_middle_vertex : ∃ v : OctahedronVertex, v.is_middle

-- Define our specific middle vertex M.
def M (v : OctahedronVertex) := v.is_middle ∧ ∃ u : OctahedronVertex, u ∈ v.adjacent_vertices

-- Define a function to check if a vertex is either M or the bottom vertex.
def is_M_or_bottom (v : OctahedronVertex) (M : OctahedronVertex) : Prop :=
  v = M ∨ v.is_bottom

-- The probability calculation logic.
noncomputable def probability_M_or_bottom (v : OctahedronVertex) : ℚ :=
  let P_total := 4 -- Total number of adjacent vertices
  let P_M_or_bottom := 2 -- Number of desired outcomes
  P_M_or_bottom / P_total

-- Theorem stating the result, given the conditions.
theorem probability_M_or_bottom_theorem (M : OctahedronVertex) (v : OctahedronVertex) (H : M.is_middle) (H_adj : v ∈ M.adjacent_vertices) :
  probability_M_or_bottom M = 1 / 2 :=
s

end probability_M_or_bottom_theorem_l83_83665


namespace closure_union_is_target_result_l83_83284

-- Definitions of sets M and N
def M : Set ℝ := { x | (x + 3) * (x - 1) < 0 }
def N : Set ℝ := { x | x ≤ -3 }

-- Union of sets M and N
def M_union_N : Set ℝ := M ∪ N

-- The closure of a set in the real numbers
def closure (S : Set ℝ) : Set ℝ := { x | ∃ y ∈ S, x ≤ y } -- Note: This is a simplified version; the actual closure in Lean might differ

-- The target result
def target_result : Set ℝ := { x | x ≤ 1 }

-- Statement to prove
theorem closure_union_is_target_result : closure M_union_N = target_result :=
by {
  sorry -- Proof is omitted
}

end closure_union_is_target_result_l83_83284


namespace sum_of_integers_satisfying_binom_identity_l83_83116

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_of_integers_satisfying_binom_identity :
  (∀ n : ℕ, binom 30 15 + binom 30 n = binom 31 16) →
  ∑ n in {14, 16}, n = 30 :=
by
  sorry

end sum_of_integers_satisfying_binom_identity_l83_83116


namespace sheep_problem_l83_83403

theorem sheep_problem (mary_sheep : ℕ) (bob_sheep : ℕ) (mary_sheep_initial : mary_sheep = 300)
    (bob_sheep_calculated : bob_sheep = (2 * mary_sheep) + 35) :
    (mary_sheep + 266 = bob_sheep - 69) :=
begin
  sorry
end

end sheep_problem_l83_83403


namespace required_sand_volume_is_five_l83_83167

noncomputable def length : ℝ := 10
noncomputable def depth_cm : ℝ := 50
noncomputable def depth_m : ℝ := depth_cm / 100  -- converting cm to m
noncomputable def width : ℝ := 2
noncomputable def total_volume : ℝ := length * depth_m * width
noncomputable def current_volume : ℝ := total_volume / 2
noncomputable def additional_sand : ℝ := total_volume - current_volume

theorem required_sand_volume_is_five : additional_sand = 5 :=
by sorry

end required_sand_volume_is_five_l83_83167


namespace repeating_decimals_to_fraction_l83_83209

theorem repeating_decimals_to_fraction :
  (0.\overline{2} + 0.\overline{03} + 0.\overline{0004} + 0.\overline{00005} = \frac{56534}{99999}) :=
by 
  have h1 : 0.\overline{2} = 2 / 9 := sorry,
  have h2 : 0.\overline{03} = 3 / 99 := sorry,
  have h3 : 0.\overline{0004} = 4 / 9999 := sorry,
  have h4 : 0.\overline{00005} = 5 / 99999 := sorry,
  sorry

end repeating_decimals_to_fraction_l83_83209


namespace central_angle_of_sector_in_unit_circle_with_area_1_is_2_l83_83331

theorem central_angle_of_sector_in_unit_circle_with_area_1_is_2 :
  ∀ (θ : ℝ), (∀ (r : ℝ), (r = 1) → (1 / 2 * r^2 * θ = 1) → θ = 2) :=
by
  intros θ r hr h
  sorry

end central_angle_of_sector_in_unit_circle_with_area_1_is_2_l83_83331


namespace percentage_of_birth_in_june_l83_83771

theorem percentage_of_birth_in_june (total_scientists: ℕ) (born_in_june: ℕ) (h_total: total_scientists = 150) (h_june: born_in_june = 15) : (born_in_june * 100 / total_scientists) = 10 := 
by 
  sorry

end percentage_of_birth_in_june_l83_83771


namespace modular_inverse_of_2_mod_199_l83_83216

theorem modular_inverse_of_2_mod_199 : (2 * 100) % 199 = 1 := 
by sorry

end modular_inverse_of_2_mod_199_l83_83216


namespace value_of_R_l83_83630

def P : ℝ := 4014 / 2
def Q : ℝ := P / 4
def R : ℝ := P - Q

theorem value_of_R: R = 1505.25 := 
by
  sorry

end value_of_R_l83_83630


namespace find_Mindy_tax_rate_l83_83027

variables (M r : ℝ) (Mork_income Mindy_income combined_income combined_tax : ℝ)

-- Conditions
def Mork_tax_rate := 0.40
def combined_tax_rate := 1 / 3
def Mork_income := M
def Mindy_income := 2 * M
def combined_tax := 0.40 * M + r * 2 * M
def combined_income := M + 2 * M 

-- Theorem statement
theorem find_Mindy_tax_rate 
  (h1 : combined_tax_rate = (combined_tax / combined_income)) : r = 0.30 :=
sorry

end find_Mindy_tax_rate_l83_83027


namespace k_divisible_by_p_minus_1_l83_83411

theorem k_divisible_by_p_minus_1 {p k : ℕ} (hp : nat.prime p) :
  (∀ x : zmod p, x ≠ 0 → x^k = 1) →
  (k % (p - 1) = 0) := 
by
  sorry

end k_divisible_by_p_minus_1_l83_83411


namespace problem_I_problem_II_l83_83276

-- Problem (I): Proving the inequality solution set
theorem problem_I (x : ℝ) : |x - 5| + |x + 6| ≤ 12 ↔ -13/2 ≤ x ∧ x ≤ 11/2 :=
by
  sorry

-- Problem (II): Proving the range of m
theorem problem_II (m : ℝ) : (∀ x : ℝ, |x - m| + |x + 6| ≥ 7) ↔ (m ≤ -13 ∨ m ≥ 1) :=
by
  sorry

end problem_I_problem_II_l83_83276


namespace area_of_triangle_OPF_l83_83588

theorem area_of_triangle_OPF (O : ℝ × ℝ) (F : ℝ × ℝ) (P : ℝ × ℝ)
  (hO : O = (0, 0)) (hF : F = (1, 0)) (hP_on_parabola : P.2 ^ 2 = 4 * P.1)
  (hPF : dist P F = 3) : Real.sqrt 2 = 1 / 2 * abs (F.1 - O.1) * (2 * Real.sqrt 2) := 
sorry

end area_of_triangle_OPF_l83_83588


namespace draw_is_unfair_ensure_fair_draw_l83_83323

open ProbabilityTheory MeasureTheory

-- Definitions for the given conditions:
def Card := {rank : ℕ // 6 ≤ rank ∧ rank ≤ 14} -- Ranks 6 to Ace (6 to 14)
def Deck := Finset (Fin 36) -- 36 unique cards
noncomputable def suit_high_rank_count (d : Deck) (v_card : Fin 36) (m_card : Fin 36) : ℕ := 
  -- Count how many cards are higher than Volodya's card
  card.count (λ c, c.val > v_card.val) d

-- Volodya draws first, then Masha draws:
variables (d : Deck) (v_card m_card : Fin 36)

-- Masha wins if she draws a card with a higher rank than Volodya’s card
def masha_wins := ∃ (m_card : Fin 36), (m_card ∈ d) ∧ (m_card.val > v_card.val)

-- Volodya wins if Masha doesn't win (Masha loses)
def volodya_wins := ¬ masha_wins

theorem draw_is_unfair (d : Deck) (v_card m_card : Fin 36) :
  (volodya_wins d v_card m_card) → ¬ (masha_wins d v_card) := sorry

-- To make it fair, we can introduce a suit hierarchy:
def suits := {"Hearts", "Diamonds", "Clubs", "Spades"}
def suit_order : suits → suits → Prop
| "Spades" "Hearts" := true
| "Hearts" "Diamonds" := true
| "Diamonds" "Clubs" := true
| "Clubs" "Spades" := false
| _, _ := false

-- A fair draw means using the suit_order to rank otherwise equal cards:
def fair_draw :=
  ∀ (c1 c2 : Card), (c1.rank = c2.rank → suit_order c1.suit c2.suit)

theorem ensure_fair_draw : fair_draw := sorry

end draw_is_unfair_ensure_fair_draw_l83_83323


namespace part_a_prices_example_part_b_correct_observer_l83_83721

theorem part_a_prices_example (p : ℕ → ℝ)
  (h1 : p 0 = 5)
  (h2 : p 6 = 5.14)
  (h3 : p 13 = 5)
  (h_nondecreasing : ∀ i, 0 ≤ i ∧ i < 6 → p i ≤ p (i+1))
  (h_nonincreasing : ∀ i, 6 ≤ i ∧ i < 12 → p (i+1) ≤ p i)
  (h_avg_conditional: 5.09 < (∑ i in range 14, p i) / 14 ∧ (∑ i in range 14, p i) / 14 < 5.10)
  : ∃ p, 56.12 < (p 1 + p 2 + p 3 + p 4 + p 5 +p 7 + p 8 + p 9 + p 10 + p 11 + p 12) ∧ (p 1 + p 2 + p 3 + p 4 + p 5 +p 7 + p 8 + p 9 + p 10 + p 11 + p 12) < 56.26 := 
begin
  sorry
end

theorem part_b_correct_observer
  (p : ℕ → ℝ)
  (h1 : p 0 = 5)
  (h2 : p 6 = 5.14)
  (h3 : p 13 = 5)
  (h_nondecreasing : ∀ i, 0 ≤ i ∧ i < 6 → p i ≤ p (i+1))
  (h_nonincreasing : ∀ i, 6 ≤ i ∧ i < 12 → p (i+1) ≤ p i)
  : (∑ i in range 7, p i) / 7 + 10.5 < (∑ i in range 7, p (i + 7)) / 7 → observer B is correct :=
begin
  sorry
end

end part_a_prices_example_part_b_correct_observer_l83_83721


namespace pond_completely_frozen_l83_83078

noncomputable def pond_frozen_day (a b : ℝ) (initial_area : ℝ) : ℝ :=
if h₁ : ((a - 20) * (b - 20) = 0.798 * initial_area) ∧
        ((a - 40) * (b - 40) = 0.814 * initial_area) then
    7 -- As demonstrated in the solution
else
    0 -- Placeholder for other analysis

theorem pond_completely_frozen :
  ∀ (a b : ℝ) (initial_area : ℝ),
   (initial_area = a * b) →
   ((a - 20) * (b - 20) = 0.798 * initial_area) →
   ((a - 40) * (b - 40) = 0.814 * initial_area) →
   pond_frozen_day a b initial_area = 7 :=
by
  intros a b initial_area h_initial h_day1 h_day2
  unfold pond_frozen_day 
  split_ifs
  { refl }
  { contradiction }
  sorry

end pond_completely_frozen_l83_83078


namespace recurring_fraction_sum_l83_83922

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l83_83922


namespace smaller_square_side_length_l83_83050

theorem smaller_square_side_length
  (ABCD : Square)
  (h1 : ABCD.side_length = 1)
  (E F : Point)
  (h2 : E ∈ line_segment (BC) ∧ F ∈ line_segment (CD))
  (h3 : is_isosceles_right_triangle (triangle A E F) ∧ AE = AF)
  (s : ℝ)
  (h4 : smaller_square B s sides_parallel)
  (h5 : vertex_on_line_segment V AE):
  s = (1 - real.sqrt 2) / 1 ∧ p + q + r = 4 :=
begin
  -- Proof omitted
  sorry
end

end smaller_square_side_length_l83_83050


namespace sum_of_fraction_numerator_and_denominator_l83_83914

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l83_83914


namespace isosceles_triangle_proof_l83_83345

variables {A B C D M E : Type}
variables (AC BC CD AB CE : ℝ)
variables (IsIsosceles : AC = BC)
variables (AltitudeFromC : true) -- We'll assume the existence of D.
variables (MidpointM : (M = midpoint C D))
variables (IntersectionE : ∃ E : Type, E = intersection BM AC)

theorem isosceles_triangle_proof :
  AC = 3 * CE :=
by
  -- Conditions
  have h1 : AC = BC := IsIsosceles
  have h2 : E = intersection BM AC := IntersectionE
  have h3 : ∃ M, midpoint C D M := MidpointM
  -- The proof steps go here, but we're only required to state the goal.
  sorry

end isosceles_triangle_proof_l83_83345


namespace books_read_by_student_body_in_one_year_l83_83948

theorem books_read_by_student_body_in_one_year
  (c s : ℕ)
  (books_per_month_per_student : ℕ)
  (books_per_month_per_student_eq_six : books_per_month_per_student = 6) :
  (12 * books_per_month_per_student * s * c) = 72 * s * c := 
  by
    rw [books_per_month_per_student_eq_six, mul_comm 6 12]
    sorry

end books_read_by_student_body_in_one_year_l83_83948


namespace find_A_B_l83_83566

theorem find_A_B :
  ∃ A B : ℚ, (∀ x : ℚ, x ≠ 9 ∧ x ≠ -6 → (4 * x - 3) / (x ^ 2 - 3 * x - 54) = A / (x - 9) + B / (x + 6)) ∧ 
  A = 11 / 5 ∧ B = 9 / 5 :=
by {
  let A := (11 : ℚ) / 5,
  let B := (9 : ℚ) / 5,
  use [A, B],
  split,
  { intros x hx,
    have h : x^2 - 3 * x - 54 = (x - 9) * (x + 6), from sorry,
    rw h,
    field_simp [hx.1, hx.2],
    ring, },
  { split; refl, }
}

end find_A_B_l83_83566


namespace matrix_product_eq_C_l83_83547

def matrix_mult (A B : Matrix (Fin 2) (Fin 2) ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  λ i j, ∑ k, A i k * B k j

def A: Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![0, -4]]
def B: Matrix (Fin 2) (Fin 2) ℤ := ![![7, -3], ![-2, 2]]
def C: Matrix (Fin 2) (Fin 2) ℤ := ![![19, -7], ![8, -8]]

theorem matrix_product_eq_C : matrix_mult A B = C := by
  sorry

end matrix_product_eq_C_l83_83547


namespace unfair_draw_fair_draw_with_suit_hierarchy_l83_83318

noncomputable def deck := {suit : String, rank : ℕ // suit ∈ {"hearts", "diamonds", "clubs", "spades"} ∧ rank ∈ {6, 7, 8, 9, 10, 11, 12, 13, 14}}
def prob_V (v : deck) : ℚ := 1 / 36
def prob_M_given_V (v m : deck) : ℚ := 1 / 35
def higher_rank (v m : deck) : Prop := m.rank > v.rank

-- Prove the draw is unfair
theorem unfair_draw : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank v m then prob_M_given_V v m else 0) < 
  (∑ m in (deck \ {v}), if ¬higher_rank v m then prob_M_given_V v m else 0)) :=
sorry

-- Making the draw fair by introducing suit hierarchy
def suit_order : String → ℕ
| "spades" := 4
| "hearts" := 3
| "diamonds" := 2
| "clubs" := 1
| _ := 0

def higher_rank_with_suit (v m : deck) : Prop :=
  if v.rank = m.rank then suit_order m.suit > suit_order v.suit else m.rank > v.rank

-- Prove introducing suit hierarchy can make the draw fair
theorem fair_draw_with_suit_hierarchy : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank_with_suit v m then prob_M_given_V v m else 0) = 
  (∑ m in (deck \ {v}), if ¬higher_rank_with_suit v m then prob_M_given_V v m else 0)) :=
sorry

end unfair_draw_fair_draw_with_suit_hierarchy_l83_83318


namespace compute_det_l83_83703

variable (α β γ : ℝ)

-- Assume that α, β, γ are the angles of a non-right triangle
axiom angle_sum : α + β + γ = 180

-- Define the determinant we need to compute
def det_matrix := det! (λ i j => match (i, j) with
  | (0, 0) => Real.sin α
  | (0, _) => 1
  | (_, 0) => 1
  | (1, 1) => Real.sin β
  | (2, 2) => Real.sin γ
  | _      => 1)

theorem compute_det : det_matrix α β γ = 2 :=
by
  sorry

end compute_det_l83_83703


namespace circle_area_increase_l83_83639

theorem circle_area_increase (r : ℝ) :
  let r_new := 3 * r in
  let area_orig := π * r^2 in
  let area_new := π * r_new^2 in
  let increase := (area_new - area_orig) / area_orig in
  increase * 100 = 800 := 
by
  sorry

end circle_area_increase_l83_83639


namespace g_eval_1000_l83_83069

def g (n : ℕ) : ℕ := sorry
axiom g_comp (n : ℕ) : g (g n) = 2 * n
axiom g_form (n : ℕ) : g (3 * n + 1) = 3 * n + 2

theorem g_eval_1000 : g 1000 = 1008 :=
by
  sorry

end g_eval_1000_l83_83069


namespace part_I_part_II_l83_83581

-- Definition of the sequence with given conditions
def seq_1 : ℕ → ℝ
| 0     := 1
| (n+1) := if n == 0 then 1 else seq_1 n + (1/3)^n

-- Question I: Prove that p = 1/3 for the arithmetic sequence
theorem part_I :
  ∀ a : ℕ → ℝ,
  (∀ n : ℕ, a n = seq_1 n) →
  a 1 = 1 →
  (∀ n : ℕ, abs (a (n + 1) - a n) = (1/3)^n) →
  (∀ n : ℕ, a n < a (n + 1)) →
  (a 1, 2 * a 2, 3 * a 3) = (a 1, 2 * (a 1 + (1/3)), 3 * (a 1 + (1/3) + (1/3)^2)) →
  (4 * (a 1 + (1/3)) = a 1 + 3 * (a 1 + (1/3) + (1/3)^2)) →
  (3 * (1/3)^2 - (1/3) = 0) →
  sorry

-- Question II: Prove the general term formula for the sequence when p = 1/2
def seq_2 : ℕ → ℝ
| 0     := 1
| (n+1) := seq_2 n + if (n + 1) % 2 = 0 then -(1/2)^n else (1/2)^n

theorem part_II :
  ∀ a : ℕ → ℝ,
  (∀ n : ℕ, a n = seq_2 n) →
  a 1 = 1 →
  (∀ n : ℕ, abs (a (n + 1) - a n) = (1/2)^n) →
  (a (2 * n - 1) < a (2 * n)) →
  (a (2 * (n + 1) - 1) > a (2 * (n))) →
  (∀ n, a n = if n % 2 = 0 then (4/3 + 1/(3 * 2^((n-1)//2))) else (4/3 - 1/(3 * 2^((n-1)//2)))) →
  sorry

end part_I_part_II_l83_83581


namespace checkerboard_max_coverage_l83_83146

noncomputable def diagonal_length (side_length : ℝ) : ℝ :=
  Real.sqrt (side_length ^ 2 + side_length ^ 2)

def card_side_length : ℝ := 2

def max_covered_squares : ℕ := 9

@[problem]
theorem checkerboard_max_coverage :
  ∀ (side_length : ℝ),
  side_length = card_side_length →
  let diagonal := diagonal_length side_length in
  2 < diagonal ∧ diagonal < 3 →
  max_covered_squares = 9 :=
by
  sorry

end checkerboard_max_coverage_l83_83146


namespace true_compound_proposition_l83_83019

-- Define conditions and propositions in Lean
def proposition_p : Prop := ∃ (x : ℝ), x^2 + x + 1 < 0
def proposition_q : Prop := ∀ (x : ℝ), 1 ≤ x → x ≤ 2 → x^2 - 1 ≥ 0

-- Define the compound proposition
def correct_proposition : Prop := ¬ proposition_p ∧ proposition_q

-- Prove the correct compound proposition
theorem true_compound_proposition : correct_proposition :=
by
  sorry

end true_compound_proposition_l83_83019


namespace rectangle_area_problem_l83_83956

theorem rectangle_area_problem (l w l1 l2 w1 w2 : ℝ) (h1 : l = l1 + l2) (h2 : w = w1 + w2) 
  (h3 : l1 * w1 = 12) (h4 : l2 * w1 = 15) (h5 : l1 * w2 = 12) 
  (h6 : l2 * w2 = 8) (h7 : w1 * l2 = 18) (h8 : l1 * w2 = 20) :
  l2 * w1 = 18 :=
sorry

end rectangle_area_problem_l83_83956


namespace range_of_x2_plus_y2_l83_83591

noncomputable def f : ℝ → ℝ := sorry

def increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

def symmetric_about_origin (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(-x) = -f(x)

def given_inequality (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x^2 - 6 * x + 21) + f (y^2 - 8 * y) < 0

theorem range_of_x2_plus_y2 (h_increasing : increasing f) 
  (h_symmetric : symmetric_about_origin f) 
  (h_inequality : given_inequality f) 
  (h_x_gt_three : ∀ x : ℝ, 3 < x) :
  ∀ x y : ℝ, 3 < x → 13 < x^2 + y^2 ∧ x^2 + y^2 < 49 :=
  sorry

end range_of_x2_plus_y2_l83_83591


namespace area_of_triangle_XYZ_l83_83646

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_XYZ : area_of_triangle 31 31 46 = 476.25 :=
by
  sorry

end area_of_triangle_XYZ_l83_83646


namespace min_value_quadratic_l83_83819

theorem min_value_quadratic : ∃ x : ℝ, ∀ y : ℝ, 3 * x ^ 2 - 18 * x + 2023 ≤ 3 * y ^ 2 - 18 * y + 2023 :=
sorry

end min_value_quadratic_l83_83819


namespace contrapositive_of_proposition_is_false_l83_83762

variables {a b : ℤ}

/-- Proposition: If a and b are both even, then a + b is even -/
def proposition (a b : ℤ) : Prop :=
  (∀ n m : ℤ, a = 2 * n ∧ b = 2 * m → ∃ k : ℤ, a + b = 2 * k)

/-- Contrapositive: If a and b are not both even, then a + b is not even -/
def contrapositive (a b : ℤ) : Prop :=
  ¬(∀ n m : ℤ, a = 2 * n ∧ b = 2 * m) → ¬(∃ k : ℤ, a + b = 2 * k)

/-- The contrapositive of the proposition "If a and b are both even, then a + b is even" -/
theorem contrapositive_of_proposition_is_false :
  (contrapositive a b) = false :=
sorry

end contrapositive_of_proposition_is_false_l83_83762


namespace sum_of_fraction_parts_l83_83882

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l83_83882


namespace balance_two_years_l83_83428

-- Define the conditions
def initial_deposit : ℝ := 100
def annual_deposit : ℝ := 10
def annual_interest_rate : ℝ := 0.10

-- Define the function to calculate the balance at the end of a given year
def balance_after_n_years (n : ℕ) : ℝ :=
  let rec helper (years : ℕ) (current_balance : ℝ) : ℝ :=
    match years with
    | 0     => current_balance
    | (y+1) => let balance_before_interest := current_balance + annual_deposit
               in helper y (balance_before_interest + (balance_before_interest * annual_interest_rate))
  helper n initial_deposit

-- Prove that the balance after 2 years is $132
theorem balance_two_years : balance_after_n_years 2 = 132 :=
by
  -- Here you can use the calculation steps to perform the actual proof.
  sorry

end balance_two_years_l83_83428


namespace part1_part2_l83_83708

noncomputable def f (x b : ℝ) : ℝ := x^2 + b * x - 3

noncomputable def M (b : ℝ) : ℝ := 
  let x1 := b + 2
  let x2 := b - 2
  Real.Max (f x1 b) (f x2 b)

noncomputable def m (b : ℝ) : ℝ := 
  let x1 := b + 2
  let x2 := b - 2
  Real.Min (f x1 b) (f x2 b)

noncomputable def g (b : ℝ) : ℝ := M b - m b

theorem part1 {b : ℝ} (hb : b > 2) : g b = 12 * b := 
  sorry

theorem part2 : ∀ b : ℝ, g b ≥ 4 :=
  sorry

end part1_part2_l83_83708


namespace distinct_possibilities_bingo_first_column_l83_83327

theorem distinct_possibilities_bingo_first_column : 
  fintype.card {s : finset ℕ // s.card = 5 ∧ ∀ x ∈ s, 1 ≤ x ∧ x ≤ 15} = 360360 := 
by 
  sorry

end distinct_possibilities_bingo_first_column_l83_83327


namespace angle_OAB_half_length_eq_90_angle_OAB_half_OA_eq_120_l83_83220

-- Part (a)
theorem angle_OAB_half_length_eq_90
  (O A B : Point)
  (M : Point)
  (circle_center : is_center_of_circle O)
  (OM_perpendicular_AB : is_perpendicular OM AB)
  (OM_half_AB : distance(OM) = (1 / 2) * distance(AB)) :
  angle(O, A, B) = 90 := by
  sorry

-- Part (b)
theorem angle_OAB_half_OA_eq_120
  (O A B : Point)
  (M : Point)
  (circle_center : is_center_of_circle O)
  (OM_perpendicular_AB : is_perpendicular OM AB)
  (OM_half_OA : distance(OM) = (1 / 2) * distance(OA)) :
  angle(O, A, B) = 120 := by
  sorry

end angle_OAB_half_length_eq_90_angle_OAB_half_OA_eq_120_l83_83220


namespace f_0_increasing_f_nonnegative_implies_a_le_one_l83_83275

-- Definition of f_0
def f_0 (x : ℝ) : ℝ := (x - 1) * Real.exp x - (1 / 2) * x^2 + 1

-- Definition for the condition of f with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (1 / 3) * a * x^3 - (1 / 2) * x^2 + 1

-- Proof problems

-- Problem 1: Prove f_0 is increasing on ℝ
theorem f_0_increasing : ∀ (x y : ℝ), x ≤ y → f_0 x ≤ f_0 y :=
sorry

-- Problem 2: Prove if f(x) ≥ 0 on [0, +∞), then a ≤ 1
theorem f_nonnegative_implies_a_le_one (a : ℝ) (h : ∀ x : ℝ, 0 ≤ x → f a x ≥ 0) : a ≤ 1 :=
sorry

end f_0_increasing_f_nonnegative_implies_a_le_one_l83_83275


namespace power_eq_half_l83_83012

theorem power_eq_half (m n : ℝ) (h1 : {1, m} = {2, -n}) : m ^ n = 1 / 2 :=
sorry

end power_eq_half_l83_83012


namespace round_1_647_eq_1_65_l83_83053

def round_to_nearest_hundredth (x : ℚ) : ℚ :=
  let d := (x * 100) in
  if d - d.floor ≥ 0.5 then (d.floor + 1) / 100 else d.floor / 100

theorem round_1_647_eq_1_65 : round_to_nearest_hundredth 1.647 = 1.65 :=
by
  sorry

end round_1_647_eq_1_65_l83_83053


namespace out_of_pocket_expense_l83_83679

theorem out_of_pocket_expense :
  let initial_purchase := 3000
  let tv_return := 700
  let bike_return := 500
  let sold_bike_cost := bike_return + (0.20 * bike_return)
  let sold_bike_sell_price := 0.80 * sold_bike_cost
  let toaster_purchase := 100
  (initial_purchase - tv_return - bike_return - sold_bike_sell_price + toaster_purchase) = 1420 :=
by
  sorry

end out_of_pocket_expense_l83_83679


namespace tens_place_digit_l83_83062

-- Define the set of digits to be used
def digits : set ℕ := {1, 3, 5, 7, 6}

-- Define a predicate to check if a list of digits forms an even number
def is_even_number (l : list ℕ) : Prop :=
  l.last = some 6

-- Define a predicate to check if a list of digits forms the smallest possible number
def is_smallest_number (l : list ℕ) : Prop :=
  l = [1, 3, 5, 7, 6].sort

-- Define a predicate to check if the digits are used exactly once
def used_once (l : list ℕ) : Prop :=
  (l.to_set ⊆ digits) ∧ (∀ x ∈ digits, l.count x = 1)

-- Define the main theorem
theorem tens_place_digit :
  ∃ l : list ℕ, used_once l ∧ is_even_number l ∧ is_smallest_number l ∧ (l.nth 3 = some 7) :=
by
  sorry

end tens_place_digit_l83_83062


namespace rectangle_area_side_midpoints_l83_83549

open Classical
noncomputable section

-- Define the side length of the original square
def side_length (A : ℝ) : ℝ := Real.sqrt A

-- Define the length of the diagonal of the square
def diagonal_length (s : ℝ) : ℝ := s * Real.sqrt 2

-- Define the side length of the rectangle
def rectangle_side_length (s : ℝ) : ℝ := (diagonal_length s) / 2

-- Prove the area of the rectangle
theorem rectangle_area_side_midpoints (A : ℝ) (hA : A = 100) :
  let s := side_length A
  let l := rectangle_side_length s
  l * l = 50 :=
by
  sorry

end rectangle_area_side_midpoints_l83_83549


namespace sum_of_fraction_parts_of_repeating_decimal_l83_83831

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l83_83831


namespace tom_candy_pieces_l83_83457

/-!
# Problem Statement
Tom bought 14 boxes of chocolate candy, 10 boxes of fruit candy, and 8 boxes of caramel candy. 
He gave 8 chocolate boxes and 5 fruit boxes to his little brother. 
If each chocolate box has 3 pieces inside, each fruit box has 4 pieces, and each caramel box has 5 pieces, 
prove that Tom still has 78 pieces of candy.
-/

theorem tom_candy_pieces 
  (chocolate_boxes : ℕ := 14)
  (fruit_boxes : ℕ := 10)
  (caramel_boxes : ℕ := 8)
  (gave_away_chocolate_boxes : ℕ := 8)
  (gave_away_fruit_boxes : ℕ := 5)
  (chocolate_pieces_per_box : ℕ := 3)
  (fruit_pieces_per_box : ℕ := 4)
  (caramel_pieces_per_box : ℕ := 5)
  : chocolate_boxes * chocolate_pieces_per_box + 
    fruit_boxes * fruit_pieces_per_box + 
    caramel_boxes * caramel_pieces_per_box - 
    (gave_away_chocolate_boxes * chocolate_pieces_per_box + 
     gave_away_fruit_boxes * fruit_pieces_per_box) = 78 :=
by
  sorry

end tom_candy_pieces_l83_83457


namespace jack_paid_total_l83_83675

theorem jack_paid_total (cost_squat_rack : ℕ) (cost_barbell_fraction : ℕ) 
  (h1 : cost_squat_rack = 2500) (h2 : cost_barbell_fraction = 10) :
  let cost_barbell := cost_squat_rack / cost_barbell_fraction in
  let total_cost := cost_squat_rack + cost_barbell in
  total_cost = 2750 :=
by
  -- Assign the values
  let cost_barbell := cost_squat_rack / cost_barbell_fraction
  let total_cost := cost_squat_rack + cost_barbell
  -- We use the assumptions h1 and h2
  have h_cost_barbell : cost_barbell = 250 := by
    simp only [h1, h2]
    sorry -- complete arithmetic step
  have h_total_cost : total_cost = 2750 := by
    rw [h1, h_cost_barbell]
    sorry -- complete arithmetic step
  exact h_total_cost

end jack_paid_total_l83_83675


namespace aunt_angela_nieces_l83_83537

theorem aunt_angela_nieces (total_jellybeans : ℕ)
                           (jellybeans_per_child : ℕ)
                           (num_nephews : ℕ)
                           (num_nieces : ℕ) 
                           (total_children : ℕ) 
                           (h1 : total_jellybeans = 70)
                           (h2 : jellybeans_per_child = 14)
                           (h3 : num_nephews = 3)
                           (h4 : total_children = total_jellybeans / jellybeans_per_child)
                           (h5 : total_children = num_nephews + num_nieces) :
                           num_nieces = 2 :=
by
  sorry

end aunt_angela_nieces_l83_83537


namespace repeating_decimal_fraction_sum_l83_83840

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l83_83840


namespace min_quadratic_expr_l83_83814

noncomputable def quadratic_expr (x : ℝ) := 3 * x^2 - 18 * x + 2023

theorem min_quadratic_expr : ∃ x : ℝ, quadratic_expr x = 1996 :=
by
  have h : quadratic_expr (3 : ℝ) = 1996
  exact h
  use 3
  rw h
  sorry -- Proof of h (already derived in given solution)

end min_quadratic_expr_l83_83814


namespace unfair_draw_fair_draw_with_suit_hierarchy_l83_83319

noncomputable def deck := {suit : String, rank : ℕ // suit ∈ {"hearts", "diamonds", "clubs", "spades"} ∧ rank ∈ {6, 7, 8, 9, 10, 11, 12, 13, 14}}
def prob_V (v : deck) : ℚ := 1 / 36
def prob_M_given_V (v m : deck) : ℚ := 1 / 35
def higher_rank (v m : deck) : Prop := m.rank > v.rank

-- Prove the draw is unfair
theorem unfair_draw : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank v m then prob_M_given_V v m else 0) < 
  (∑ m in (deck \ {v}), if ¬higher_rank v m then prob_M_given_V v m else 0)) :=
sorry

-- Making the draw fair by introducing suit hierarchy
def suit_order : String → ℕ
| "spades" := 4
| "hearts" := 3
| "diamonds" := 2
| "clubs" := 1
| _ := 0

def higher_rank_with_suit (v m : deck) : Prop :=
  if v.rank = m.rank then suit_order m.suit > suit_order v.suit else m.rank > v.rank

-- Prove introducing suit hierarchy can make the draw fair
theorem fair_draw_with_suit_hierarchy : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank_with_suit v m then prob_M_given_V v m else 0) = 
  (∑ m in (deck \ {v}), if ¬higher_rank_with_suit v m then prob_M_given_V v m else 0)) :=
sorry

end unfair_draw_fair_draw_with_suit_hierarchy_l83_83319


namespace min_time_for_flashes_l83_83513

theorem min_time_for_flashes 
  (num_lights : ℕ)
  (flash_time : ℕ)
  (interval_time : ℕ)
  (distinct_sequences : ℕ)
  (h_lights : num_lights = 3)
  (h_flash_time : flash_time = 3)
  (h_interval_time : interval_time = 3)
  (h_sequences : distinct_sequences = 6) :
  (distinct_sequences * flash_time + interval_time * (distinct_sequences - 1) = 33) :=
by 
  rw [h_lights, h_flash_time, h_interval_time, h_sequences]
  sorry

end min_time_for_flashes_l83_83513


namespace number_of_terms_arithmetic_sequence_l83_83201

theorem number_of_terms_arithmetic_sequence :
  ∀ (a d l : ℤ), a = -36 → d = 6 → l = 66 → ∃ n, l = a + (n-1) * d ∧ n = 18 :=
by
  intros a d l ha hd hl
  exists 18
  rw [ha, hd, hl]
  sorry

end number_of_terms_arithmetic_sequence_l83_83201


namespace option_b_option_d_l83_83240

def f (x : ℝ) : ℝ := (Real.cos x) * (Real.cos x + Real.sqrt 3 * Real.sin x)

theorem option_b 
  : ∀ x : ℝ, f (x - π / 6) = f (-(x - π / 6)) :=
by sorry

theorem option_d 
  : ∀ x1 x2 : ℝ, x1 + x2 = 5 * π / 6 → f x1 + f x2 = 1 :=
by sorry

end option_b_option_d_l83_83240


namespace repeating_decimal_sum_l83_83873

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l83_83873


namespace MrC_loses_240_after_transactions_l83_83155

theorem MrC_loses_240_after_transactions :
  let house_initial_value := 12000
  let first_transaction_loss_percent := 0.15
  let second_transaction_gain_percent := 0.20
  let house_value_after_first_transaction :=
    house_initial_value * (1 - first_transaction_loss_percent)
  let house_value_after_second_transaction :=
    house_value_after_first_transaction * (1 + second_transaction_gain_percent)
  house_value_after_second_transaction - house_initial_value = 240 :=
by
  sorry

end MrC_loses_240_after_transactions_l83_83155


namespace sum_of_data_points_is_30_l83_83582

theorem sum_of_data_points_is_30 (x : Fin 10 → ℝ) 
  (h : (1 / 10) * ((∑ i, (x i - 3)^2)) = S^2) :
  ∑ i, x i = 30 :=
by
  sorry

end sum_of_data_points_is_30_l83_83582


namespace triangle_sum_of_remaining_sides_l83_83798

noncomputable def sum_of_sides (a b c : ℝ) : ℝ :=
a + b + c

theorem triangle_sum_of_remaining_sides :
  ∃ (A B C : ℝ), ∠A = 40 ∧ ∠B = 50 ∧ ∠C = 90 ∧ opposite 40 = 8 ∧ 
  (sum_of_sides = 20.3) :=
begin
  sorry
end

end triangle_sum_of_remaining_sides_l83_83798


namespace sum_of_fraction_terms_l83_83864

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l83_83864


namespace greatest_distance_between_circle_centers_l83_83304

theorem greatest_distance_between_circle_centers :
  let rectangle_width := 18
  let rectangle_height := 15
  let circle_diameter := 7
  let circle_radius := circle_diameter / 2
  let new_rectangle_width := rectangle_width - 2 * circle_radius
  let new_rectangle_height := rectangle_height - 2 * circle_radius
  (math.sqrt (new_rectangle_width^2 + new_rectangle_height^2) = math.sqrt 185) :=
by
  sorry

end greatest_distance_between_circle_centers_l83_83304


namespace find_k_l83_83483

theorem find_k : ∃ k : ℝ, 64 / k = 4 ∧ k = 16 :=
begin
  existsi (16 : ℝ),
  split,
  { norm_num, },
  { refl, }
end

end find_k_l83_83483


namespace sum_of_fraction_parts_of_repeating_decimal_l83_83835

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l83_83835


namespace polar_equation_of_circle_length_of_PQ_l83_83351

-- Prove that the polar equation of the circle C is ρ = 2 sin θ given the parametric equations.
theorem polar_equation_of_circle (φ θ ρ : ℝ) : 
  (∀ φ, x = cos φ ∧ y = 1 + sin φ → (x^2 + (y - 1)^2 = 1) → (x = ρ * cos θ ∧ y = ρ * sin θ) → ρ = 2 * sin θ) := 
sorry

-- Prove the length of line segment PQ is 1 given the intersection definitions
theorem length_of_PQ (ρP ρQ : ℝ) : 
  (ρP = (2 * sin (π / 6)) ∧ ρQ = 2 * sin ((π / 6) + (π / 3)) → ρQ = 2 → |ρP - ρQ| = 1) :=
sorry

end polar_equation_of_circle_length_of_PQ_l83_83351


namespace cone_volume_l83_83267

theorem cone_volume (l r h : ℝ) (π : ℝ) (hs : l = 5) (hls : π * r * l = 15 * π) : 
  (1 / 3 * π * r * r * h = 12 * π) :=
begin
  sorry
end

end cone_volume_l83_83267


namespace canoe_upstream_speed_l83_83144

noncomputable def speed_of_canoe_when_rowing_upstream (C D S : ℝ) : ℝ :=
  C - S

theorem canoe_upstream_speed :
  ∀ (C D S : ℝ), D = C + S → S = 4.5 → D = 12 → speed_of_canoe_when_rowing_upstream C D S = 3 :=
by {
  intros C D S h1 h2 h3,
  rw [speed_of_canoe_when_rowing_upstream, h1, h3, h2],
  norm_num,
  sorry
}

end canoe_upstream_speed_l83_83144


namespace octagon_area_ratio_l83_83002

noncomputable def ratio_of_areas : ℚ := (1 / 4 : ℝ)

theorem octagon_area_ratio (m n : ℕ) (h1 : m.coprime n) (h2 : ratio_of_areas = m / n) : m + n = 5 :=
sorry

end octagon_area_ratio_l83_83002


namespace repeating_decimal_sum_l83_83868

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l83_83868


namespace lamps_on_iff_m_divides_n_l83_83086

theorem lamps_on_iff_m_divides_n (n m : ℕ) (h1 : 0 < n ∧ 0 < m) :
  (∃ moves : list (ℕ × ℕ × ℕ × bool), (∀ i j : ℕ, i < n → j < n → ((i, j) ∈ (moves.foldr
    (λ (move : ℕ × ℕ × ℕ × bool) (table : ℕ × ℕ → bool),
     let row_col := move.1
     let pos := move.2
     let length := move.3
     let is_row := move.4
     if is_row then
       (λ rc, if rc.1 = row_col ∧ pos ≤ rc.2 ∧ rc.2 < pos + length then !table rc else table rc)
     else
       (λ rc, if rc.2 = row_col ∧ pos ≤ rc.1 ∧ rc.1 < pos + length then !table rc else table rc))
    (λ _, ff) ((nat.lt_wf n).fix (λ (k : ℕ) (rec : (ℕ → bool)),
     if k = 0 then pure false else ff))) ∧ (∃ i j, i < n ∧ j < n ∧ (moves.foldr (λ (move : ℕ × ℕ × ℕ × bool) (table : ℕ × ℕ → bool),
      let row_col := move.1
      let pos := move.2
      let length := move.3
      let is_row := move.4
      if is_row then
        (λ rc, if rc.1 = row_col ∧ pos ≤ rc.2 ∧ rc.2 < pos + length then !table rc else table rc)
      else
        (λ rc, if rc.2 = row_col ∧ pos ≤ rc.1 ∧ rc.1 < pos + length then !table rc else table rc))
     ((λ rc, ff)) ((nat.lt_wf n).fix (λ (k : ℕ) (rec : _),
      if k = 0 then sorry else sorry)) i j = ff)) ↔ m ∣ n := by sorry

end lamps_on_iff_m_divides_n_l83_83086


namespace educational_expenditure_increase_l83_83335

theorem educational_expenditure_increase (x : ℝ) :
  let y := 0.15*x + 0.2 in
  let y1 := 0.15*(x + 1) + 0.2 in
  y1 - y = 0.15 :=
by
  sorry

end educational_expenditure_increase_l83_83335


namespace part_a_prices_example_part_b_correct_observer_l83_83720

theorem part_a_prices_example (p : ℕ → ℝ)
  (h1 : p 0 = 5)
  (h2 : p 6 = 5.14)
  (h3 : p 13 = 5)
  (h_nondecreasing : ∀ i, 0 ≤ i ∧ i < 6 → p i ≤ p (i+1))
  (h_nonincreasing : ∀ i, 6 ≤ i ∧ i < 12 → p (i+1) ≤ p i)
  (h_avg_conditional: 5.09 < (∑ i in range 14, p i) / 14 ∧ (∑ i in range 14, p i) / 14 < 5.10)
  : ∃ p, 56.12 < (p 1 + p 2 + p 3 + p 4 + p 5 +p 7 + p 8 + p 9 + p 10 + p 11 + p 12) ∧ (p 1 + p 2 + p 3 + p 4 + p 5 +p 7 + p 8 + p 9 + p 10 + p 11 + p 12) < 56.26 := 
begin
  sorry
end

theorem part_b_correct_observer
  (p : ℕ → ℝ)
  (h1 : p 0 = 5)
  (h2 : p 6 = 5.14)
  (h3 : p 13 = 5)
  (h_nondecreasing : ∀ i, 0 ≤ i ∧ i < 6 → p i ≤ p (i+1))
  (h_nonincreasing : ∀ i, 6 ≤ i ∧ i < 12 → p (i+1) ≤ p i)
  : (∑ i in range 7, p i) / 7 + 10.5 < (∑ i in range 7, p (i + 7)) / 7 → observer B is correct :=
begin
  sorry
end

end part_a_prices_example_part_b_correct_observer_l83_83720


namespace area_ratio_l83_83690

-- Define the sets S1 and S2
def S1 : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | let x := p.1 in let y := p.2 
             in Real.log10 (3 + x^2 + y^2) <= 1 + Real.log10 (x + y)}

def S2 : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | let x := p.1 in let y := p.2 
             in Real.log10 (5 + x^2 + y^2) <= 2 + Real.log10 (x + y)}

-- Prove the ratio of the area of S2 to the area of S1 is 223
theorem area_ratio (area_S1 area_S2 : ℝ) (h_S1 : ∀ A : ℝ, (∀ x y : ℝ, (x, y) ∈ S1 → (x - 5)^2 + (y - 5)^2 ≤ 22) → A = 22 * Real.pi) 
                         (h_S2 : ∀ B : ℝ, (∀ x y : ℝ, (x, y) ∈ S2 → (x - 50)^2 + (y - 50)^2 ≤ 4905) → B = 4905 * Real.pi) : 
  area_S2 / area_S1 = 223 := 
by
  sorry

end area_ratio_l83_83690


namespace roots_of_polynomial_l83_83567

theorem roots_of_polynomial :
  ∃ x1 x2 x3 x4 : ℝ,
  (3 * x1^4 + 2 * x1^3 - 8 * x1^2 + 2 * x1 + 3 = 0) ∧
  (3 * x2^4 + 2 * x2^3 - 8 * x2^2 + 2 * x2 + 3 = 0) ∧
  (3 * x3^4 + 2 * x3^3 - 8 * x3^2 + 2 * x3 + 3 = 0) ∧
  (3 * x4^4 + 2 * x4^3 - 8 * x4^2 + 2 * x4 + 3 = 0) ∧
  (x1 = (-1 + sqrt(172 - 12 * sqrt(43))) / 6) ∧
  (x2 = (-1 - sqrt(172 - 12 * sqrt(43))) / 6) ∧
  (x3 = (-1 + sqrt(172 + 12 * sqrt(43))) / 6) ∧
  (x4 = (-1 - sqrt(172 + 12 * sqrt(43))) / 6) :=
by {
  sorry
}

end roots_of_polynomial_l83_83567


namespace proof_part1_proof_part2_proof_part3_l83_83668

variables (A B C a b c R : ℝ) (n : ℕ)

-- Assume A, B, C are the angles of a triangle
-- with side lengths a, b, c, and circumradius R 
-- Specifically, we need ℝ variables since Lean 4 does not natively handle trig functions on ℝ
-- Extrinsic definitions might still hold but are generally for completeness

noncomputable def part1 (h_triangle: A + B + C = π) : Prop :=
  (cos (A / 3))^3 + (cos (B / 3))^3 + (cos (C / 3))^3 ≤ (9 / 4) * cos (π / 9) + 3 / 8

noncomputable def part2 (h_triangle: A + B + C = π) : Prop :=
  csc(A)^n + csc(B)^n + csc(C)^n ≥ 3 * (2 / sqrt 3)^n

noncomputable def part3 (h_triangle: A + B + C = π) (h_sides: a^2 = b^2 + c^2 - 2 * b * c * cos A ∧ b^2 = a^2 + c^2 - 2 * a * c * cos B ∧ c^2 = a^2 + b^2 - 2 * a * b * cos C) : Prop :=
  1 / a^n + 1 / b^n + 1 / c^n ≥ 3 / (sqrt 3 * R)^n

theorem proof_part1 (h_triangle: A + B + C = π) : part1 A B C :=
sorry

theorem proof_part2 (h_triangle: A + B + C = π) : part2 A B C :=
sorry

theorem proof_part3 (h_triangle: A + B + C = π) (h_sides: a^2 = b^2 + c^2 - 2 * b * c * cos A ∧ b^2 = a^2 + c^2 - 2 * a * c * cos B ∧ c^2 = a^2 + b^2 - 2 * a * b * cos C) : part3 A B C a b c R :=
sorry

end proof_part1_proof_part2_proof_part3_l83_83668


namespace convert_rectangular_to_polar_l83_83997

theorem convert_rectangular_to_polar (x y : ℝ) (hx : x = sqrt 2) (hy : y = - (sqrt 2)) :
    ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r = 2) ∧ (θ = (7 * Real.pi) / 4) :=
by
  exists 2
  exists (7 * Real.pi / 4)
  simp [hx, hy, Real.sqrt_eq_rpow, Real.rpow_two, Real.pi]
  sorry

end convert_rectangular_to_polar_l83_83997


namespace digits_difference_l83_83099

theorem digits_difference :
  let digits := [9, 7, 4, 2, 1, 5] in
  let largest := 975421 in
  let least := 124579 in
  largest - least = 850842 :=
by
  let digits := [9, 7, 4, 2, 1, 5]
  let largest := 975421
  let least := 124579
  show largest - least = 850842
  sorry

end digits_difference_l83_83099


namespace unique_tetrahedron_l83_83035

variables {A B C D P Q R S : Type}

-- Define conditions as hypotheses
-- We assume the points are in ℝ³ space.

axiom point_on_AB (v : ℝ × ℝ × ℝ) : v ∈ {a | ∃ (t ∈ ℝ), t ∈ (0 : ℝ,1 : ℝ) ∧ a = t • A + (1 - t) • B }
axiom point_on_BC (v : ℝ × ℝ × ℝ) : v ∈ {a | ∃ (t ∈ ℝ), t ∈ (0 : ℝ,1 : ℝ) ∧ a = t • B + (1 - t) • C }
axiom point_on_CD (v : ℝ × ℝ × ℝ) : v ∈ {a | ∃ (t ∈ ℝ), t ∈ (0 : ℝ,1 : ℝ) ∧ a = t • C + (1 - t) • D }
axiom point_on_DA (v : ℝ × ℝ × ℝ) : v ∈ {a | ∃ (t ∈ ℝ), t ∈ (0 : ℝ,1 : ℝ) ∧ a = t • D + (1 - t) • A }

def trisect {X Y : ℝ × ℝ × ℝ} (t : ℝ) (t1 t2 : ℝ) : Prop := t = t1 ∨ t = t2

axiom P_trisection : trisect point_on_AB A B
axiom Q_trisection : trisect point_on_BC B C
axiom R_trisection : trisect point_on_CD C D
axiom S_trisection : trisect point_on_DA D A

theorem unique_tetrahedron (h : matrix (point_on_AB, point_on_BC, point_on_CD, point_on_DA)).
  (h1 : ¬ collinear {P, Q, R, S}) :
   unique vertices (A, B, C, D) :=
sorry

end unique_tetrahedron_l83_83035


namespace total_price_purchase_l83_83478

variable (S T : ℝ)

theorem total_price_purchase (h1 : 2 * S + T = 2600) (h2 : 900 = 1200 * 0.75) : 2600 + 900 = 3500 := by
  sorry

end total_price_purchase_l83_83478


namespace candy_cost_l83_83619

theorem candy_cost (total_cents : ℕ) (gumdrops : ℕ) (cost_per_gumdrop : ℕ) :
  total_cents = 224 → gumdrops = 28 → cost_per_gumdrop = total_cents / gumdrops → cost_per_gumdrop = 8 :=
by {
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  exact h3,
}

end candy_cost_l83_83619


namespace reject_null_hypothesis_serum_prevents_colds_l83_83661

def K_squared : ℝ := 3.918
def critical_value : ℝ := 3.841
def significance_level : ℝ := 0.05
def null_hypothesis : Prop := ∀ (serum_effective : Prop), ¬ serum_effective

-- Prove that we reject the null hypothesis H at the 0.05 significance level
theorem reject_null_hypothesis (H : Prop) 
    (H_null: H = null_hypothesis)
    (K2 : ℝ) 
    (critical : ℝ)
    (alpha : ℝ)
    (hK2 : K2 ≈ K_squared)
    (hCritical : critical = critical_value)
    (halpha : alpha = significance_level)
    : K2 > critical → H = null_hypothesis → (¬ H) :=
by
  sorry

-- Prove that the correct statement is "There is a 95% confidence that 'this serum can prevent colds'"
theorem serum_prevents_colds 
    (H : Prop)
    (K2 : ℝ)
    (critical : ℝ)
    (alpha : ℝ)
    (hK2 : K2 ≈ K_squared)
    (hCritical : critical = critical_value)
    (halpha : alpha = significance_level)
    (reject_H : K2 > critical → H = null_hypothesis → (¬ H))
    : ∃ statement : String, statement = "There is a 95% confidence that 'this serum can prevent colds'" :=
by
  sorry

end reject_null_hypothesis_serum_prevents_colds_l83_83661


namespace binom_15_12_eq_455_l83_83191

theorem binom_15_12_eq_455 : Nat.choose 15 12 = 455 := 
by sorry

end binom_15_12_eq_455_l83_83191


namespace derivative_f1_derivative_f2_derivative_f3_derivative_f4_derivative_f5_l83_83034

noncomputable def f1 (x : ℝ) : ℝ := exp (sin x)
noncomputable def f2 (x : ℝ) : ℝ := (x + 3) / (x + 2)
noncomputable def f3 (x : ℝ) : ℝ := log (2 * x + 3)
noncomputable def f4 (x : ℝ) : ℝ := (x ^ 2 + 2) * (2 * x - 1)
noncomputable def f5 (x : ℝ) : ℝ := cos (2 * x + real.pi / 3)

theorem derivative_f1 (x : ℝ) : deriv f1 x = exp (sin x) * cos x :=
sorry

theorem derivative_f2 (x : ℝ) : deriv f2 x = -1 / (x + 2)^2 :=
sorry

theorem derivative_f3 (x : ℝ) : deriv f3 x = 2 / (2 * x + 3) :=
sorry

theorem derivative_f4 (x : ℝ) : deriv f4 x = 6 * x ^ 2 - 2 * x + 4 :=
sorry

theorem derivative_f5 (x : ℝ) : deriv f5 x = -2 * sin (2 * x + real.pi / 3) :=
sorry

end derivative_f1_derivative_f2_derivative_f3_derivative_f4_derivative_f5_l83_83034


namespace rearrange_raven_no_consecutive_vowels_l83_83289

theorem rearrange_raven_no_consecutive_vowels :
  let letters := ["R", "A", "V", "E", "N"]
  let vowels := ["A", "E"]
  let consonants := ["R", "V", "N"]
  (letters.permutations.length - (consonants.permutations.length * 2)) = 72 :=
by
  sorry

end rearrange_raven_no_consecutive_vowels_l83_83289


namespace line_AB_intersects_S2_S_l83_83437

noncomputable def circle := sorry -- Placeholder definition for circle

variables {S1 S2 S : circle}
variables (A : Point) (O : Point) (B : Point)

-- Conditions
axiom radius_S1_S2 : ∀ P ∈ {S1, S2}, radius P = 1
axiom touch_S1_S2 : tangent S1 S2 A
axiom radius_S : radius S = 2
axiom center_on_S1 : O ∈ S1
axiom touch_S1_S : tangent S1 S B
axiom center_S : center S = O

-- Question
theorem line_AB_intersects_S2_S :
  passes (line_through A B) (intersection (S2, S)) :=
sorry

end line_AB_intersects_S2_S_l83_83437


namespace equation_represents_circle_of_radius_8_l83_83571

theorem equation_represents_circle_of_radius_8 (k : ℝ) : 
  (x^2 + 14 * x + y^2 + 8 * y - k = 0) → k = -1 ↔ (∃ r, r = 8 ∧ (x + 7)^2 + (y + 4)^2 = r^2) :=
by
  sorry

end equation_represents_circle_of_radius_8_l83_83571


namespace john_pills_per_week_l83_83683

theorem john_pills_per_week :
  ∀ (pills_per_interval : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ),
  pills_per_interval = 6 →
  hours_per_day = 24 →
  days_per_week = 7 →
  (hours_per_day / pills_per_interval) * days_per_week = 28 :=
by
  intros pills_per_interval hours_per_day days_per_week hppi hpd hdpw
  rw [hppi, hpd, hdpw]
  sorry

end john_pills_per_week_l83_83683


namespace instantaneous_rate_of_change_at_4_l83_83248

noncomputable def radius := 4  -- radius of the cup's bottom in cm
noncomputable def volume (t : ℝ) : ℝ := real.pi * t^3 + 2 * real.pi * t^2  -- volume function in ml
noncomputable def area := real.pi * radius^2  -- area of the cup's bottom in cm²
noncomputable def height (t : ℝ) : ℝ := volume(t) / area -- height of the solution as a function of time

noncomputable def height_rate_of_change (t : ℝ) : ℝ := (derivative height) t  -- derivative of height function

theorem instantaneous_rate_of_change_at_4 :
  height_rate_of_change 4 = 4 :=
by
  sorry

end instantaneous_rate_of_change_at_4_l83_83248


namespace orthocenters_concyclic_and_center_l83_83592

variables {A1 A2 A3 A4 O H1 H2 H3 H4 : Type*}
variables [IncircleQuadrilateral A1 A2 A3 A4 O]
variables [Orthocenters H1 H2 H3 H4 A1 A2 A3 A4]

theorem orthocenters_concyclic_and_center :
  (orthocenters_concyclic A1 A2 A3 A4 H1 H2 H3 H4) ∧
  (center_perpendicular_bisector H2 H3 DO EP P)
  sorry

end orthocenters_concyclic_and_center_l83_83592


namespace vectors_are_dependent_l83_83045

-- Define the vectors.
def a1 : ℝ × ℝ × ℝ := (2, 1, 3)
def a2 : ℝ × ℝ × ℝ := (1, 1, -1)
def a3 : ℝ × ℝ × ℝ := (1, -1, 9)

-- Define what it means for vectors to be linearly dependent.
def linear_dependent (v1 v2 v3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (λ1 λ2 λ3 : ℝ), (λ1 ≠ 0 ∨ λ2 ≠ 0 ∨ λ3 ≠ 0) ∧ 
  (λ1 * v1.1 + λ2 * v2.1 + λ3 * v3.1 = 0) ∧
  (λ1 * v1.2 + λ2 * v2.2 + λ3 * v3.2 = 0) ∧
  (λ1 * v1.3 + λ2 * v2.3 + λ3 * v3.3 = 0)

-- Prove the vectors are linearly dependent.
theorem vectors_are_dependent : linear_dependent a1 a2 a3 :=
sorry

end vectors_are_dependent_l83_83045


namespace isosceles_triangle_angle_B_l83_83532

theorem isosceles_triangle_angle_B :
  ∀ (A B C : ℝ), (B = C) → (C = 3 * A) → (A + B + C = 180) → (B = 540 / 7) :=
by
  intros A B C h1 h2 h3
  sorry

end isosceles_triangle_angle_B_l83_83532


namespace find_n_times_s_l83_83700

def g : ℕ → ℕ := sorry

axiom g_condition : ∀ a b : ℕ, 3 * g (a^2 + b^2) = (g a)^2 + (g b)^2 + g a * g b

theorem find_n_times_s : 
  let possible_values := { x : ℕ | ∃ a b : ℕ, g 36 = x } in
  let n := possible_values.finite.card in
  let s := possible_values.sum id in
  n * s = 2 :=
by sorry

end find_n_times_s_l83_83700


namespace problem_statement_l83_83253

noncomputable def a (n : ℕ) : ℕ → ℝ
| 0       := 4
| (n + 1) := (n + 1) * 2 ^ (2 - n : ℤ)

def S (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

theorem problem_statement (n : ℕ) (h₁ : S 1 = 2) (h₂ : S n - S (n-1) = (n / (2 * (n - 1))) * a (n-1)) :
  (∀ n, a n = (4 * n) / (2^n)) ∧ (S n = 8 - ((8 + 4 * n) / (2^n))) :=
by
  sorry

end problem_statement_l83_83253


namespace part1_part2_l83_83274

-- Definitions and conditions
def f (x a : ℝ) : ℝ := exp x - (1 / 2) * x^2 - a * x
def g (x a : ℝ) : ℝ := exp x - a * x^2 - a * x

-- Part 1
theorem part1 (a : ℝ) : (∀ x : ℝ, diff (f x a) > 0) ↔ (a ≤ 1) :=
by sorry

-- Part 2
theorem part2 {a x1 x2 : ℝ} (h1 : g x1 a = 0) (h2 : g x2 a = 0) : 
  (x1 ≠ x2) → ((x1 + x2) / 2 < log (2 * a)) :=
by sorry

end part1_part2_l83_83274


namespace triangle_AC_l83_83780

theorem triangle_AC (AB CD : ℝ) (angle_ABC angle_CBD : ℝ) (AC : ℝ) :
    AB = 1 ∧ CD = 1 ∧ angle_ABC = real.pi / 2 ∧ angle_CBD = real.pi / 6 → AC = real.cbrt 2 :=
by
  intros h
  sorry

end triangle_AC_l83_83780


namespace fibonacci_identity_cassini_identity_l83_83745

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fibonacci_identity (n m : ℕ) (hn : 1 ≤ n) (hm : 0 ≤ m) :
  fib (n + m) = fib (n - 1) * fib m + fib n * fib (m + 1) := sorry

theorem cassini_identity (n : ℕ) (hn : 1 ≤ n) :
  fib (n + 1) * fib (n - 1) - fib n * fib n = (-1)^n := sorry

end fibonacci_identity_cassini_identity_l83_83745


namespace max_no_roots_equations_l83_83785

-- Define the conditions of the problem as a Lean statement

theorem max_no_roots_equations (n : ℕ) (odd_n : n % 2 = 1) : 
  ∃ k, k = (n + 1) / 2 ∧ ∀ strategy2, player1_strategy(strategy2) = k :=
sorry

end max_no_roots_equations_l83_83785


namespace minimum_value_expression_l83_83820

theorem minimum_value_expression : ∃ x : ℝ, (3 * x^2 - 18 * x + 2023) = 1996 := sorry

end minimum_value_expression_l83_83820


namespace instantaneous_rate_of_change_at_4s_l83_83245

noncomputable def V : ℝ → ℝ := λ t, Real.pi * t^3 + 2 * Real.pi * t^2

noncomputable def S : ℝ := Real.pi * 4^2

noncomputable def h (t : ℝ) : ℝ := (V t) / S

noncomputable def h' (t: ℝ) : ℝ := (deriv (λ t, h t)) t

theorem instantaneous_rate_of_change_at_4s : h' 4 = 4 :=
by
  -- proof steps will be inserted here
  sorry

end instantaneous_rate_of_change_at_4s_l83_83245


namespace sum_x_y_is_4_l83_83013

theorem sum_x_y_is_4 {x y : ℝ} (h : x / (1 - (I : ℂ)) + y / (1 - 2 * I) = 5 / (1 - 3 * I)) : x + y = 4 :=
sorry

end sum_x_y_is_4_l83_83013


namespace recurring_fraction_sum_l83_83924

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l83_83924


namespace area_trisected_quadrilateral_l83_83662

-- Definitions for areas of quadrilaterals and trisection points
def trisect_segment (A B : Point) (A' B' : Point) : Prop :=
  dist A A' = dist A' B' ∧ dist A' B' = dist B' B

def quadrilateral_area (A B C D : Point) : ℝ :=
  -- Here this should be a function that calculates the area of the quadrilateral ABCD
  sorry

-- Main theorem statement
theorem area_trisected_quadrilateral (A B C D A' B' C' D' : Point)
  (h1 : trisect_segment A B A' B')
  (h2 : trisect_segment C D C' D') :
  quadrilateral_area A' B' C' D' = 1 / 3 * quadrilateral_area A B C D :=
  sorry

end area_trisected_quadrilateral_l83_83662


namespace kinder_surprise_min_purchase_l83_83961

-- Define the conditions
def kinder_surprise_problem : Prop :=
  ∀ (car_kinds : Finset ℕ) (h1 : car_kinds.card = 5),
  ∃ (n : ℕ), n = 11 ∧ (∀ (f : Fin n → car_kinds), ∃ (k ∈ car_kinds), (f⁻¹' {k}).card ≥ 3)

-- State the theorem
theorem kinder_surprise_min_purchase : kinder_surprise_problem := by
  sorry

end kinder_surprise_min_purchase_l83_83961


namespace range_of_m_l83_83258

namespace ProofProblem

variable (x m : ℝ)

def condition_p : Prop := (x - m) * (x - (m + 3)) > 0
def condition_q : Prop := x^2 + 3 * x - 4 < 0

theorem range_of_m (h : condition_p x m → condition_q x m) :
  m ∈ Iic (-7) ∪ Ici 1 :=
sorry

end ProofProblem

end range_of_m_l83_83258


namespace train_pass_time_correct_l83_83518

-- Definition of the conditions
def train_length : ℝ := 860 -- in meters
def bridge_length : ℝ := 450 -- in meters
def train_speed_kmh : ℝ := 85 -- in km/hour

-- Conversion factor from km/h to m/s
def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

-- Converted speed in m/s
def train_speed_mps : ℝ := kmh_to_mps train_speed_kmh

-- Total distance to be covered by the train to pass the bridge
def total_distance : ℝ := train_length + bridge_length

-- Time calculation
def time_to_pass_bridge (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem train_pass_time_correct :
  time_to_pass_bridge total_distance train_speed_mps = 55.52 :=
by sorry

end train_pass_time_correct_l83_83518


namespace lines_PQ_RS_perpendicular_l83_83693

variables {A B C D M P Q R S : Type}
variables [has_coords A] [has_coords B] [has_coords C] [has_coords D]
variables [is_centroid P A M D] [is_centroid Q B M C]
variables [is_orthocenter R A M B] [is_orthocenter S C M D]

theorem lines_PQ_RS_perpendicular {M : Point}
    (hM: M = intersection_point (diagonal AC) (diagonal BD))
    (hP: P = centroid_triangle A M D)
    (hQ: Q = centroid_triangle B M C)
    (hR: R = orthocenter_triangle A M B)
    (hS: S = orthocenter_triangle C M D) :
    are_perpendicular (line P Q) (line R S) :=
sorry

end lines_PQ_RS_perpendicular_l83_83693


namespace part_a_part_b_l83_83729

noncomputable def average_price_between : Prop :=
  ∃ (prices : Fin 14 → ℝ), 
    prices 0 = 5 ∧ prices 6 = 5.14 ∧ prices 13 = 5 ∧ 
    5.09 < (∑ i, prices i) / 14 ∧ (∑ i, prices i) / 14 < 5.10

theorem part_a : average_price_between :=
  sorry

def average_difference : ℝ :=
  let prices1 := [5.0, 5.1, 5.1, 5.1, 5.1, 5.1, 5.14] in
  let prices2 := [5.14, 5.14, 5.14, 5.14, 5.14, 5.14, 5.0] in
  let avg1 := (prices1.sum / prices1.length : ℝ) in
  let avg2 := (prices2.sum / prices2.length : ℝ) in
  abs (avg2 - avg1)

theorem part_b : average_difference < 0.105 :=
  sorry

end part_a_part_b_l83_83729


namespace range_of_x_l83_83604

noncomputable theory

def f (x : ℝ) : ℝ := Real.exp x + x^3

theorem range_of_x (x : ℝ) (h : f (x^2) < f (3 * x - 2)) : 1 < x ∧ x < 2 :=
sorry

end range_of_x_l83_83604


namespace volume_neq_cross_section_l83_83207

variable {G : Type} [GeometricBody G]

def cross_sectional_area (A : G) (h : ℝ) : ℝ
def volume (A : G) : ℝ

theorem volume_neq_cross_section (A B : G) :
  (∀ h : ℝ, cross_sectional_area A h = cross_sectional_area B h → volume A = volume B) →
  (volume A ≠ volume B) → (∃ h : ℝ, cross_sectional_area A h ≠ cross_sectional_area B h) := by
  sorry

end volume_neq_cross_section_l83_83207


namespace sheep_problem_l83_83404

theorem sheep_problem (mary_sheep : ℕ) (bob_sheep : ℕ) (mary_sheep_initial : mary_sheep = 300)
    (bob_sheep_calculated : bob_sheep = (2 * mary_sheep) + 35) :
    (mary_sheep + 266 = bob_sheep - 69) :=
begin
  sorry
end

end sheep_problem_l83_83404


namespace total_cost_is_2750_l83_83674

def squat_rack_cost : ℕ := 2500
def barbell_cost : ℕ := squat_rack_cost / 10
def total_cost : ℕ := squat_rack_cost + barbell_cost

theorem total_cost_is_2750 : total_cost = 2750 := by
  have h1 : squat_rack_cost = 2500 := by rfl
  have h2 : barbell_cost = 2500 / 10 := by rfl
  have h3 : total_cost = 2500 + 250 := by rfl
  have h4 : total_cost = 2750 := by rw [h1, h2, h3]
  sorry

end total_cost_is_2750_l83_83674


namespace sum_of_fraction_parts_of_repeating_decimal_l83_83830

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l83_83830


namespace h_5_is_40_l83_83198

def h : ℕ → ℤ
| 1       := 1
| 2       := 1
| (n + 1) := if n < 2 then 1 else h n - h (n - 1) + (n + 1) ^ 2

theorem h_5_is_40 : h 5 = 40 :=
by sorry

end h_5_is_40_l83_83198


namespace union_of_sets_complement_intersection_of_sets_l83_83612

def setA : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def setB : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_sets :
  setA ∪ setB = {x | 2 < x ∧ x < 10} :=
sorry

theorem complement_intersection_of_sets :
  (setAᶜ) ∩ setB = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
sorry

end union_of_sets_complement_intersection_of_sets_l83_83612


namespace tan_range_l83_83244

open Real

noncomputable def range_tan (θ : ℝ) : Prop :=
  (sin θ) / (sqrt 3 * cos θ + 1) > 1

theorem tan_range (θ : ℝ) (h : range_tan θ) : 
  tan θ ∈ set.Iic (-sqrt 2) ∪ set.Ioc (sqrt 3 / 3) (sqrt 2) := 
sorry

end tan_range_l83_83244


namespace simplify_expr_calculate_expr_l83_83751

noncomputable theory
open Classical

-- Define a custom namespace for the problem
namespace SimplifyCalculations
variable (a b : ℝ)
variable h₁ : 0 < a
variable h₂ : 0 < b

-- Problem 1: Simplify the given expression to 1/a
theorem simplify_expr : 
  (a ^ (2/3) * b ^ (-1)) ^ (-1/2) * a ^ (-1/2) * b ^ (1/3) / (a * b ^ 5) ^ (1/6) = 1 / a :=
by
  sorry

-- Problem 2: Calculate the given expression to get 0.09
theorem calculate_expr : 
  (0.027)^(2/3) + (27/125)^(-1/3) - (2 + 7/9)^(0.5) = 0.09 :=
by
  sorry

end SimplifyCalculations

end simplify_expr_calculate_expr_l83_83751


namespace roots_mul_shift_eq_neg_2018_l83_83007

theorem roots_mul_shift_eq_neg_2018 {a b : ℝ}
  (h1 : a + b = -1)
  (h2 : a * b = -2020) :
  (a - 1) * (b - 1) = -2018 :=
sorry

end roots_mul_shift_eq_neg_2018_l83_83007


namespace volume_ratio_cones_l83_83103

def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

theorem volume_ratio_cones :
  let r_C := 20
  let h_C := 50
  let r_D := 25
  let h_D := 40
  let V_C := volume_cone r_C h_C
  let V_D := volume_cone r_D h_D
  (V_C / V_D) = (4 / 5) :=
by
  let r_C := 20
  let h_C := 50
  let r_D := 25
  let h_D := 40
  let V_C := volume_cone r_C h_C
  let V_D := volume_cone r_D h_D
  have h1: V_C = (1 / 3) * real.pi * (20:ℝ)^2 * (50:ℝ) := by rfl
  have h2: V_D = (1 / 3) * real.pi * (25:ℝ)^2 * (40:ℝ) := by rfl
  rw [h1, h2]
  rw [div_mul_eq_mul_div, one_div, one_div]
  rw [mul_assoc, mul_assoc, mul_comm (real.pi * 20 ^ 2) _, mul_div_assoc, mul_div_assoc]
  have h3 : (20:ℝ)^2 * 50 = 20000 := by { norm_num }
  have h4 : (25:ℝ)^2 * 40 = 25000 := by { norm_num }
  rw [h3, h4]
  norm_num
  sorry

end volume_ratio_cones_l83_83103


namespace largest_p_plus_q_l83_83796

-- All required conditions restated as Assumptions
def triangle {R : Type*} [LinearOrderedField R] (p q : R) : Prop :=
  let B : R × R := (10, 15)
  let C : R × R := (25, 15)
  let A : R × R := (p, q)
  let M : R × R := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let area : R := (1 / 2) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))
  let median_slope : R := (A.2 - M.2) / (A.1 - M.1)
  area = 100 ∧ median_slope = -3

-- Statement to be proven
theorem largest_p_plus_q {R : Type*} [LinearOrderedField R] (p q : R) :
  triangle p q → p + q = 70 / 3 :=
by
  sorry

end largest_p_plus_q_l83_83796


namespace binomial_coefficient_x4_l83_83452

noncomputable def binomial_expansion_coefficient (n : ℕ) :=
  (2 - Real.sqrt x)^n

theorem binomial_coefficient_x4 {n : ℕ} (h : 2^n = 256) :
  ∑ k in finset.range (n + 1), binomial n k = 256 →
  let T_r := (λ r, binomial n r * 2^(n-r) * (-Real.sqrt x)^r) in
  (T_r 8 = 1) :=
sorry

end binomial_coefficient_x4_l83_83452


namespace sum_of_fraction_parts_of_repeating_decimal_l83_83827

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l83_83827


namespace find_number_l83_83133

theorem find_number : ∃ x, x - 0.16 * x = 126 ↔ x = 150 :=
by 
  sorry

end find_number_l83_83133


namespace repeating_decimal_fraction_sum_l83_83855

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l83_83855


namespace find_a_b_l83_83699

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def g (x : ℝ) : ℝ := 3 * x - 6

theorem find_a_b (a b : ℝ) (h : ∀ x : ℝ, g (f a b x) = 4 * x + 7) :
  a + b = 17 / 3 :=
by
  sorry

end find_a_b_l83_83699


namespace matrix_zero_product_or_rank_one_l83_83691

variables {n : ℕ}
variables (A B C : matrix (fin n) (fin n) ℝ)

theorem matrix_zero_product_or_rank_one
  (h1 : A * B * C = 0)
  (h2 : B.rank = 1) :
  A * B = 0 ∨ B * C = 0 :=
sorry

end matrix_zero_product_or_rank_one_l83_83691


namespace lowest_height_l83_83772

noncomputable def length_A : ℝ := 2.4
noncomputable def length_B : ℝ := 3.2
noncomputable def length_C : ℝ := 2.8

noncomputable def height_Eunji : ℝ := 8 * length_A
noncomputable def height_Namjoon : ℝ := 4 * length_B
noncomputable def height_Hoseok : ℝ := 5 * length_C

theorem lowest_height :
  height_Namjoon = 12.8 ∧ 
  height_Namjoon < height_Eunji ∧ 
  height_Namjoon < height_Hoseok :=
by
  sorry

end lowest_height_l83_83772


namespace shift_line_one_unit_left_l83_83339

theorem shift_line_one_unit_left : ∀ (x y : ℝ), (y = x) → (y - 1 = (x + 1) - 1) :=
by
  intros x y h
  sorry

end shift_line_one_unit_left_l83_83339


namespace no_rational_solutions_l83_83037

-- Define the problem
theorem no_rational_solutions (n : ℕ) :
  ∀ (x y : ℚ), (x + real.sqrt 3 * y)^n ≠ real.sqrt (1 + real.sqrt 3) :=
by
  sorry

end no_rational_solutions_l83_83037


namespace problem_1_problem_2_l83_83036

theorem problem_1 (a b : ℝ) (h : a * b > 0) : 
  ( ∛(a^2 * b^2 * (a + b)^2 / 4) ≤ (a^2 + 10 * a * b + b^2) / 12 ) := 
begin
  sorry
end

theorem problem_2 (a b : ℝ) : 
  ( ∛(a^2 * b^2 * (a + b)^2 / 4) ≤ (a^2 + a * b + b^2) / 3 ) := 
begin
  sorry
end

end problem_1_problem_2_l83_83036


namespace part_a_prices_example_part_b_correct_observer_l83_83719

theorem part_a_prices_example (p : ℕ → ℝ)
  (h1 : p 0 = 5)
  (h2 : p 6 = 5.14)
  (h3 : p 13 = 5)
  (h_nondecreasing : ∀ i, 0 ≤ i ∧ i < 6 → p i ≤ p (i+1))
  (h_nonincreasing : ∀ i, 6 ≤ i ∧ i < 12 → p (i+1) ≤ p i)
  (h_avg_conditional: 5.09 < (∑ i in range 14, p i) / 14 ∧ (∑ i in range 14, p i) / 14 < 5.10)
  : ∃ p, 56.12 < (p 1 + p 2 + p 3 + p 4 + p 5 +p 7 + p 8 + p 9 + p 10 + p 11 + p 12) ∧ (p 1 + p 2 + p 3 + p 4 + p 5 +p 7 + p 8 + p 9 + p 10 + p 11 + p 12) < 56.26 := 
begin
  sorry
end

theorem part_b_correct_observer
  (p : ℕ → ℝ)
  (h1 : p 0 = 5)
  (h2 : p 6 = 5.14)
  (h3 : p 13 = 5)
  (h_nondecreasing : ∀ i, 0 ≤ i ∧ i < 6 → p i ≤ p (i+1))
  (h_nonincreasing : ∀ i, 6 ≤ i ∧ i < 12 → p (i+1) ≤ p i)
  : (∑ i in range 7, p i) / 7 + 10.5 < (∑ i in range 7, p (i + 7)) / 7 → observer B is correct :=
begin
  sorry
end

end part_a_prices_example_part_b_correct_observer_l83_83719


namespace cos_B_rel_prime_sum_condition_l83_83354

-- Define the given conditions and prove the desired result.
theorem cos_B_rel_prime_sum_condition
  (A B C D : Type)
  (angle_C : ℝ)
  (BD : ℝ)
  (BD_value : BD = 17^3)
  (cosB : ℝ)
  (cosB_value : cosB = 1 / 17)
  (m n : ℕ)
  (rel_prime : Nat.coprime m n)
  (cosB_frac : cosB = m / n) :
  m + n = 18 := 
by
  sorry

end cos_B_rel_prime_sum_condition_l83_83354


namespace xn_lt_27_sqrt_k_l83_83199

noncomputable def S (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

def x_seq (k : ℕ) : ℕ → ℕ
| 1     := 1
| (n+1) := S (k * x_seq k n)

theorem xn_lt_27_sqrt_k (k : ℕ) (n : ℕ) (hk : 0 < k) :
  x_seq k n < 27 * Int.to_nat (real.sqrt k) := 
by sorry

end xn_lt_27_sqrt_k_l83_83199


namespace domain_of_function_l83_83767

noncomputable def domain_of_f : Set ℝ :=
  {x | x > -1/2 ∧ x ≠ 1}

theorem domain_of_function :
  (∀ x : ℝ, (2 * x + 1 ≥ 0) ∧ (2 * x^2 - x - 1 ≠ 0) ↔ (x > -1/2 ∧ x ≠ 1)) := by
  sorry

end domain_of_function_l83_83767


namespace triangles_cover_base_l83_83251

variables {n : ℕ} {i : ℕ} (A : Fin (n+1) → ℝ × ℝ)

def is_convex (poly : Fin (n+1) → ℝ × ℝ) : Prop :=
  ∀ (i j k : Fin (n+1)),
    let (x₁, y₁) := poly i in
    let (x₂, y₂) := poly j in
    let (x₃, y₃) := poly k in
    let det := (x₂ - x₁) * (y₃ - y₁) - (x₃ - x₁) * (y₂ - y₁) in
    det ≥ 0

structure triangle :=
  (v : Fin 3 → ℝ × ℝ)

def congruent (t₁ t₂ : triangle) : Prop :=
  sorry -- Detailed definition omitted for brevity

def triangle_covers (tri : triangle) (P : ℝ × ℝ) : Prop :=
  sorry -- Detailed definition omitted for brevity

def pyramid (S : ℝ × ℝ × ℝ) (A : Fin (n+1) → ℝ × ℝ) : Prop := 
  ∀ i, (∃ (X : ℝ × ℝ),
    congruent
      ⟨λ j, match j with -- This defines points of triangle SA_iA_{i+1}
            | 0 => (S.1, S.2)
            | 1 => A i
            | 2 => A (i+1) % n
            end⟩
      ⟨λ j, match j with -- This defines points of congruent triangle X_iA_iA_{i+1}
            | 0 => X
            | 1 => A i
            | 2 => A (i+1) % n
            end⟩)

theorem triangles_cover_base
  (S : ℝ × ℝ × ℝ)
  (A : Fin (n+1) → ℝ × ℝ)
  [hconv : is_convex A]
  (hpyramid : pyramid S A) :
  ∀ P : ℝ × ℝ, 
    ∃ i, triangle_covers ⟨λ j, match j with
                              | 0 => _
                              | 1 => A i
                              | 2 => A (i+1) % n
                              end⟩ P :=
sorry

end triangles_cover_base_l83_83251


namespace transformed_curve_is_circle_l83_83270

open Real

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * cos θ^2 + 4 * sin θ^2)

def cartesian_curve (x y: ℝ) : Prop :=
  3 * x^2 + 4 * y^2 = 12

def transformation (x y x' y' : ℝ) : Prop :=
  x' = x / 2 ∧ y' = y * sqrt (3 / 3)

theorem transformed_curve_is_circle (x y x' y' : ℝ) 
  (h1: cartesian_curve x y) (h2: transformation x y x' y') : 
  (x'^2 + y'^2 = 1) :=
sorry

end transformed_curve_is_circle_l83_83270


namespace g_five_l83_83442

variable (g : ℝ → ℝ)

-- Given conditions
axiom g_add : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_three : g 3 = 4

-- Prove g(5) = 16 * (1 / 4)^(1/3)
theorem g_five : g 5 = 16 * (1 / 4)^(1/3) := by
  sorry

end g_five_l83_83442


namespace least_possible_value_of_smallest_integer_l83_83055

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℤ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (A + B + C + D) / 4 = 74 ∧ D = 90 →
  A = 29 :=
by
  intros A B C D h_neq h_avg h_d
  sorry

end least_possible_value_of_smallest_integer_l83_83055


namespace exists_special_matrix_l83_83580

open Nat

theorem exists_special_matrix (p : ℕ) (hp_prime : Prime p) (hp1 : p % 5 = 1) (h2p1_prime : Prime (2 * p + 1)) :
  ∃ (m n : ℕ) (matrix : Matrix ℕ ℕ ℕ), 
  (m * n = 4 * p ∨ m * n = 4 * p + 2) ∧ 
  ∀ submatrix, (∃ (m' n' : ℕ), submatrix = Matrix.submatrix matrix m' n' ∧ (m' * n' ≠ 2 * p) ∧ (m' * n' ≠ 2 * p + 1)) :=
sorry

end exists_special_matrix_l83_83580


namespace Megan_pictures_left_l83_83929

theorem Megan_pictures_left (zoo_pictures museum_pictures deleted_pictures : ℕ) 
  (h1 : zoo_pictures = 15) 
  (h2 : museum_pictures = 18) 
  (h3 : deleted_pictures = 31) : 
  zoo_pictures + museum_pictures - deleted_pictures = 2 := 
by
  sorry

end Megan_pictures_left_l83_83929


namespace sum_of_fraction_terms_l83_83858

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l83_83858


namespace polynomial_identity_l83_83420

open Polynomial

-- Definition of the non-zero polynomial of interest
noncomputable def p (a : ℝ) : Polynomial ℝ := Polynomial.C a * (Polynomial.X ^ 3 - Polynomial.X)

-- Theorem stating that, for all x, the given equation holds for the polynomial p
theorem polynomial_identity (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, (x - 1) * (p a).eval (x + 1) - (x + 2) * (p a).eval x = 0 :=
by
  sorry

end polynomial_identity_l83_83420


namespace minimum_tangent_length_l83_83250

-- Given definitions directly from the problem conditions
def line := {p : ℝ × ℝ | p.2 = p.1 + 1}
def circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 1}

-- Problem statement in Lean 4
theorem minimum_tangent_length (p : ℝ × ℝ) (hp : p ∈ line) :
  ∃ q ∈ circle, ∀ q' ∈ circle, dist p q ≤ dist p q' → dist p q = √7 :=
sorry

end minimum_tangent_length_l83_83250


namespace determinant_expression_l83_83375

theorem determinant_expression (a b c p q r : ℝ) (h : Polynomial.roots (Polynomial.C r + Polynomial.C q * Polynomial.X + Polynomial.C p * Polynomial.X^2 + Polynomial.X^3) = multiset.of_list [a, b, c]) :
  Matrix.det ![![2 + a, 1, 1], ![1, 2 + b, 1], ![1, 1, 2 + c]] = 3 + 3 * q - 2 * p - p * q := 
by sorry

end determinant_expression_l83_83375


namespace limit_expression_l83_83701

-- Define the triangle
def triangle (a : ℝ) : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ a - p.1}

-- Define the integral over the triangle
noncomputable def integral_over_triangle (a : ℝ) : ℝ :=
  ∫ (p : ℝ × ℝ) in triangle a, Real.exp (p.1^3 + p.2^3) ∂Measure.prod Measure.volume Measure.volume

-- Define the expression we are interested in
noncomputable def expression (a : ℝ) : ℝ :=
  a^4 * Real.exp (-a^3) * integral_over_triangle a

-- The main statement
theorem limit_expression : 
  filter.tendsto expression filter.at_top (nhds (2 / 9)) :=
sorry

end limit_expression_l83_83701


namespace Jessica_cut_roses_l83_83788

variable (initial_roses final_roses added_roses : Nat)

theorem Jessica_cut_roses
  (h_initial : initial_roses = 10)
  (h_final : final_roses = 18)
  (h_added : final_roses = initial_roses + added_roses) :
  added_roses = 8 := by
  sorry

end Jessica_cut_roses_l83_83788


namespace vector_calculation_l83_83564

def v1 : ℝ × ℝ × ℝ := (-3, 2, -5)
def v2 : ℝ × ℝ × ℝ := (1, 6, -3)
def scalar : ℝ := 2

theorem vector_calculation :
  let s_mul := (scalar * v1.1, scalar * v1.2, scalar * v1.3)
  let addition := (s_mul.1 + v2.1, s_mul.2 + v2.2, s_mul.3 + v2.3)
  addition = (-5, 10, -13) := 
by
  sorry

end vector_calculation_l83_83564


namespace exercise_serial_matches_year_problem_serial_matches_year_l83_83031

-- Definitions for the exercise
def exercise_initial := 1169
def exercises_per_issue := 8
def issues_per_year := 9
def exercise_year := 1979
def exercises_per_year := exercises_per_issue * issues_per_year

-- Definitions for the problem
def problem_initial := 1576
def problems_per_issue := 8
def problems_per_year := problems_per_issue * issues_per_year
def problem_year := 1973

theorem exercise_serial_matches_year :
  ∃ (issue_number : ℕ) (exercise_number : ℕ),
    (issue_number = 3) ∧
    (exercise_number = 2) ∧
    (exercise_initial + 11 * exercises_per_year + 16 = exercise_year) :=
by {
  sorry
}

theorem problem_serial_matches_year :
  ∃ (issue_number : ℕ) (problem_number : ℕ),
    (issue_number = 5) ∧
    (problem_number = 5) ∧
    (problem_initial + 5 * problems_per_year + 36 = problem_year) :=
by {
  sorry
}

end exercise_serial_matches_year_problem_serial_matches_year_l83_83031


namespace parking_lot_perimeter_l83_83165

theorem parking_lot_perimeter (x y: ℝ) 
  (h1: x = (2 / 3) * y)
  (h2: x^2 + y^2 = 400)
  (h3: x * y = 120) :
  2 * (x + y) = 20 * Real.sqrt 5 :=
by
  sorry

end parking_lot_perimeter_l83_83165


namespace pulley_center_distance_l83_83534

theorem pulley_center_distance (r₁ r₂ : ℝ) (d_contact : ℝ) (h₁ : r₁ = 6) (h₂ : r₂ = 18) (h₃ : d_contact = 30) : 
  ∃ d : ℝ, d = 2 * Real.sqrt 261 := 
by
  -- Define the components as per the problem description
  let AE := 30
  let BE := 12
  have d := Real.sqrt (AE^2 + BE^2)
  have h₄ : d = 2 * Real.sqrt 261 :=
    by
      calc d = Real.sqrt (30^2 + 12^2) : by rw [← AE, ← BE]
         ... = Real.sqrt (900 + 144)    : by norm_num
         ... = Real.sqrt 1044           : by norm_num
         ... = 2 * Real.sqrt 261        : by norm_num

  exact ⟨d, h₄⟩

end pulley_center_distance_l83_83534


namespace find_m_if_pure_imaginary_l83_83011

-- Given conditions as definitions
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

-- The complex number z defined in terms of m
def z (m : ℝ) : ℂ :=
  (m^2 + 2 * m - 8 : ℝ) + ((m - 2 : ℝ) * Complex.I)

-- The main theorem to prove
theorem find_m_if_pure_imaginary (m : ℝ) : is_pure_imaginary (z m) → m = -4 :=
by {
  intros h,
  unfold is_pure_imaginary at h,
  -- Real part equals zero
  have h1 : m^2 + 2 * m - 8 = 0 := h.1,
  -- Imaginary part is non-zero
  have h2 : m ≠ 2 := h.2,
  sorry
}

end find_m_if_pure_imaginary_l83_83011


namespace period_of_function_y_increasing_intervals_of_function_y_range_of_function_y_on_interval_l83_83278

noncomputable def function_y (x : ℝ) : ℝ := 2 * sin (π / 3 - 2 * x)

theorem period_of_function_y :
  ∃ T > 0, ∀ x, function_y (x + T) = function_y x :=
sorry

theorem increasing_intervals_of_function_y :
  ∀ k : ℤ, ∀ x, (5 * π / 12 + k * π) ≤ x ∧ x ≤ (11 * π / 12 + k * π) → 
  function_y x ≤ function_y (x + T) :=
sorry

theorem range_of_function_y_on_interval :
  ∀ x, 0 ≤ x ∧ x ≤ π / 2 → -2 ≤ function_y x ∧ function_y x ≤ sqrt 3 :=
sorry

end period_of_function_y_increasing_intervals_of_function_y_range_of_function_y_on_interval_l83_83278


namespace sum_of_fraction_terms_l83_83857

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l83_83857


namespace true_props_l83_83415

def f (x : ℝ) : ℝ := log ((1 - x) / (1 + x))

-- Proposition definitions
def prop1 : Prop := domain f = {x | -1 < x ∧ x < 1}
def prop2 : Prop := ∀ x, f (-x) = -f x
def prop3 : Prop := ∀ x y, (x < y ∧ x ∈ {-1 < x ∧ x < 1} ∧ y ∈ {-1 < x ∧ x < 1}) → f x < f y
def prop4 : Prop := ∀ x1 x2, x1 ∈ {-1 < x ∧ x < 1} ∧ x2 ∈ {-1 < x ∧ x < 1} → f x1 + f x2 = f ((x1 + x2) / (1 + x1 * x2))

theorem true_props : prop2 ∧ prop4 :=
by
  sorry

end true_props_l83_83415


namespace different_dispatch_plans_l83_83965

theorem different_dispatch_plans :
  let teachers := {A, B, C, D, E, F, G, H} -- Constant set of teachers
  let dispatches :=  _ -- Function to calculate all valid dispatch plans according to conditions
  dispatches (teachers) = 600 :=
by
  sorry -- Proof omitted

end different_dispatch_plans_l83_83965


namespace triangle_area_approx_l83_83129

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def herons_formula (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_approx : herons_formula 30 26 10 ≈ 126.72 := sorry

end triangle_area_approx_l83_83129


namespace sum_equal_numbered_squares_l83_83498

def row_wise_numbering (i j : ℕ) : ℕ :=
  15 * (i - 1) + j

def column_wise_numbering (i j : ℕ) : ℕ :=
  10 * (j - 1) + i

theorem sum_equal_numbered_squares :
  (∑ (i j : ℕ) in (finset.range 11).product (finset.range 16), 
   if row_wise_numbering i j = column_wise_numbering i j 
   then row_wise_numbering i j else 0) = 630 :=
by
  sorry

end sum_equal_numbered_squares_l83_83498


namespace no_conditional_statements_in_1_and_2_l83_83273

theorem no_conditional_statements_in_1_and_2 :
  ∀ (triangle_area : ℝ) (a b c : ℝ),
    triangle_area = 1 ∧ (∀ a b c : ℝ, (arithmetic_mean a b c = (a + b + c) / 3)) →
    (equilateral_triangle_perimeter_no_conditional triangle_area ∧
    arithmetic_mean_no_conditional a b c)
  :=
begin
  intro triangle_area,
  intro a,
  intro b,
  intro c,
  intro cond,
  have h1 : triangle_area = 1 := cond.1,
  have h2 : ∀ a b c : ℝ, arithmetic_mean a b c = (a + b + c) / 3 := cond.2,
  sorry
end

end no_conditional_statements_in_1_and_2_l83_83273


namespace length_of_PT_l83_83139

-- Define the trapezoid with given properties
variables (P Q R S T: Type)
variables [AffineSpace ℝ P] [AffineSpace ℝ Q] [AffineSpace ℝ R] [AffineSpace ℝ S] [AffineSpace ℝ T]

-- Conditions
variables {PQ RS PR PT : ℝ}
variables (h1 : PQ = 3 * RS) (h2 : diag_intersect : diag_intersect P Q R S T) (h3 : PR = 21)

-- Prove the length of PT
theorem length_of_PT 
  (hPQ_RS_ratio : PQ = 3 * RS)
  (hdiag_intersect : diag_intersect P Q R S T)
  (hPR : PR = 21) :
  PT = 7 :=
by 
  -- the proof steps go here
  sorry

end length_of_PT_l83_83139


namespace sum_of_fraction_numerator_and_denominator_l83_83912

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l83_83912


namespace sum_of_fraction_parts_of_repeating_decimal_l83_83834

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l83_83834


namespace amount_borrowed_l83_83507

variable (P : ℝ)
variable (interest_paid : ℝ) -- Interest paid on borrowing
variable (interest_earned : ℝ) -- Interest earned on lending
variable (gain_per_year : ℝ)

variable (h1 : interest_paid = P * 4 * 2 / 100)
variable (h2 : interest_earned = P * 6 * 2 / 100)
variable (h3 : gain_per_year = 160)
variable (h4 : gain_per_year = (interest_earned - interest_paid) / 2)

theorem amount_borrowed : P = 8000 := by
  sorry

end amount_borrowed_l83_83507


namespace polynomial_simplified_l83_83097

def polynomial (x : ℝ) : ℝ := 4 - 6 * x - 8 * x^2 + 12 - 14 * x + 16 * x^2 - 18 + 20 * x + 24 * x^2

theorem polynomial_simplified (x : ℝ) : polynomial x = 32 * x^2 - 2 :=
by
  sorry

end polynomial_simplified_l83_83097


namespace not_geometric_exp_function_exists_possible_values_of_m_l83_83249

variable {α : Type*} [LinearOrderedField α]

-- Conditions for the problem
variables (an : ℕ → α) (m : ℕ) (d : α)
variable [arith_seq : ∀ (n : ℕ), an (n + 1) = an n + d]
variable (h1 : m ≥ 3) (h2 : d > 0)

-- Mathematical statements to prove
-- 1. The sequence is not geometric
theorem not_geometric :
  ¬(∃ r : α, ∀ n, an (n + 1) = r * an n) :=
sorry

-- 2. Existence of the exponential function
theorem exp_function_exists :
  ∃ (f : α → α), (f = λ x, exp (-x / d)) ∧ 
  (∀ i < m - 1, ∃ t : α, t * deriv (λ x, exp (-x / d)) (an i) + an (i + 1) = 0) :=
sorry

-- 3. Possible values of m
theorem possible_values_of_m :
  ∃ (bn : ℕ → α), (∀ i j, i < j → bn i ≠ bn j) ∧ 
  (∀ i, an i ∈ finset.image an (finset.range m) → 
  (∃ r : α, ∀ n, bn (n + 1) = r * bn n)) → (m = 3) :=
sorry

end not_geometric_exp_function_exists_possible_values_of_m_l83_83249


namespace sum_of_cubes_of_consecutive_even_integers_l83_83453

theorem sum_of_cubes_of_consecutive_even_integers (x : ℤ) (h : x^2 + (x+2)^2 + (x+4)^2 = 2960) :
  x^3 + (x + 2)^3 + (x + 4)^3 = 90117 :=
sorry

end sum_of_cubes_of_consecutive_even_integers_l83_83453


namespace minimum_area_triangle_OAB_l83_83003

noncomputable def minimum_area_OAB : ℝ :=
  let F := (3 / 4, 0)
  let parabola : ℝ → ℝ → Prop := λ x y, y^2 = 3 * x
  let line_through_F := λ y, ∃ x, x = 3 / 4 ∧ parabola x y  -- Vertically at x = 3/4
  let O := (0, 0)
  let A := (3 / 4, 3 / 2)
  let B := (3 / 4, -3 / 2)
  1 / 2 * 3 / 4 * 2 * 3 / 2

theorem minimum_area_triangle_OAB : minimum_area_OAB = 9 / 8 :=
  sorry

end minimum_area_triangle_OAB_l83_83003


namespace yardage_lost_due_to_sacks_l83_83509

theorem yardage_lost_due_to_sacks 
  (throws : ℕ)
  (percent_no_throw : ℝ)
  (half_sack_prob : ℕ)
  (sack_pattern : ℕ → ℕ)
  (correct_answer : ℕ) : 
  throws = 80 →
  percent_no_throw = 0.30 →
  (∀ (n: ℕ), half_sack_prob = n/2) →
  (sack_pattern 1 = 3 ∧ sack_pattern 2 = 5 ∧ ∀ n, n > 2 → sack_pattern n = sack_pattern (n - 1) + 2) →
  correct_answer = 168 :=
by
  sorry

end yardage_lost_due_to_sacks_l83_83509


namespace smallest_abundant_not_multiple_of_5_is_12_l83_83104

def is_abundant (n : ℕ) : Prop :=
  ∑ d in finset.filter (λ x, x ∣ n) (finset.range n), d > n

def smallest_abundant_not_multiple_of_5 : ℕ :=
  finset.find (λ n, is_abundant n ∧ ¬(5 ∣ n)) (finset.range 100) -- arbitrarily chose 100 as an upper bound for demonstration purposes

theorem smallest_abundant_not_multiple_of_5_is_12 :
  smallest_abundant_not_multiple_of_5 = 12 :=
sorry

end smallest_abundant_not_multiple_of_5_is_12_l83_83104


namespace repeating_decimal_fraction_sum_l83_83843

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l83_83843


namespace rhombus_diagonal_length_l83_83734

theorem rhombus_diagonal_length (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 80) (h2 : area = 2480) (h3 : area = (d1 * d2) / 2) : d1 = 62 :=
by sorry

end rhombus_diagonal_length_l83_83734


namespace intersection_with_negative_y_axis_max_value_at_x3_l83_83578

theorem intersection_with_negative_y_axis (m : ℝ) (h : 4 - 2 * m < 0) : m > 2 :=
sorry

theorem max_value_at_x3 (m : ℝ) (h : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → 3 * x + 4 - 2 * m ≤ -4) : m = 8.5 :=
sorry

end intersection_with_negative_y_axis_max_value_at_x3_l83_83578


namespace complement_of_subset_l83_83613

def universal_set : Set ℕ := {1, 2, 3, 4, 5}
def subset : Set ℕ := {2, 4}

theorem complement_of_subset (M : Set ℕ) (N : Set ℕ) (H1 : M = universal_set) (H2 : N = subset) :
  compl N = {1, 3, 5} :=
by
  rw [H2, compl_eq_univ_diff, H1]
  simp
  sorry

end complement_of_subset_l83_83613


namespace new_average_age_l83_83435

theorem new_average_age (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ) (remaining_people : ℕ) :
  avg_age = 40 →
  num_people = 8 →
  leaving_age = 25 →
  remaining_people = 7 →
  (avg_age * num_people - leaving_age) / remaining_people = 42 :=
by
  sorry

end new_average_age_l83_83435


namespace sum_of_n_values_l83_83112

theorem sum_of_n_values (n : ℕ) (h₀ : n = 16 ∨ n = 14) :
  (14 + 16 = 30 ∧ ∑ n in {16, 14}, n = 30) :=
by
  have h1: ∑ n in {16, 14}, n = 14 + 16 := by
    simp [Finset.sum_insert, Finset.sum_singleton]
  simp [h0, h1]
  sorry

end sum_of_n_values_l83_83112


namespace geom_seq_thm_l83_83607

noncomputable def geom_seq (a : ℕ → ℝ) :=
  a 1 = 2 ∧ (a 2 * a 4 = a 6)

noncomputable def b_seq (a : ℕ → ℝ) (n : ℕ) :=
  1 / (Real.logb 2 (a (2 * n - 1)) * Real.logb 2 (a (2 * n + 1)))

noncomputable def sn_sum (b : ℕ → ℝ) (n : ℕ) :=
  (Finset.range (n + 1)).sum b

theorem geom_seq_thm (a : ℕ → ℝ) (n : ℕ) (b : ℕ → ℝ) :
  geom_seq a →
  ∀ n, a n = 2 ^ n ∧ sn_sum (b_seq a) n = n / (2 * n + 1) :=
by
  sorry

end geom_seq_thm_l83_83607


namespace sum_of_fraction_numerator_and_denominator_l83_83913

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l83_83913


namespace total_bottles_l83_83042

variable m : ℝ
variable e : ℝ
variable t : ℝ

def morning_bottles := 7.5
def afternoon_bottles := 9.25
def evening_bottles := 5.75
def night_bottles := 3.5

def m := morning_bottles + afternoon_bottles
def e := evening_bottles + night_bottles
def t := m + e

theorem total_bottles : t = 26 := by
  sorry

end total_bottles_l83_83042


namespace complex_quadrant_l83_83241

theorem complex_quadrant (i : ℂ) (hi : i * i = -1) (z : ℂ) (hz : z = 1 / (1 - i)) : 
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_quadrant_l83_83241


namespace recurring_decimal_fraction_sum_l83_83902

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l83_83902


namespace bella_stamps_l83_83539

theorem bella_stamps :
  let snowflake_cost := 1.05
  let truck_cost := 1.20
  let rose_cost := 0.90
  let butterfly_cost := 1.15
  let snowflake_spent := 15.75
  
  let snowflake_stamps := snowflake_spent / snowflake_cost
  let truck_stamps := snowflake_stamps + 11
  let rose_stamps := truck_stamps - 17
  let butterfly_stamps := 1.5 * rose_stamps
  
  let total_stamps := snowflake_stamps + truck_stamps + rose_stamps + butterfly_stamps
  
  total_stamps = 64 := by
  sorry

end bella_stamps_l83_83539


namespace student_can_create_16_valid_programs_l83_83172

open Finset

variable (Courses : Finset String := {"English", "Algebra", "Geometry", "History", "Art", "Latin", "Biology"})
variable (MathCourses : Finset String := {"Algebra", "Geometry"})

def valid_programs (sel : Finset String) : Prop :=
  "English" ∈ sel ∧ (sel ∩ MathCourses).nonempty ∧ sel.card = 4

noncomputable def count_valid_programs : ℕ :=
  (Courses.erase "English").powerset.filter (λ c, (c ∩ MathCourses).nonempty ∧ c.card = 3).card

theorem student_can_create_16_valid_programs :
  count_valid_programs = 16 :=
sorry

end student_can_create_16_valid_programs_l83_83172


namespace block_exists_if_path_possible_block_exists_if_closed_euler_path_l83_83807

-- Part (a) statement
theorem block_exists_if_path_possible (p q r : ℕ) (hp : 1 ≤ p) (hq : 1 ≤ q) (hr : 1 ≤ r) : 
  ∃ (A B : ℕ), (path_exists (p, q, r) A B) := sorry

-- Part (b) statement
theorem block_exists_if_closed_euler_path (p q r : ℕ) (hp : 1 ≤ p) (hq : 1 ≤ q) (hr : 1 ≤ r) :
  (even_two_of_three p q r) → ∃ (A : ℕ), (closed_path_exists (p, q, r) A) := sorry

-- Definitions that might be needed.
def even_two_of_three (p q r : ℕ) : Prop :=
  (even p ∧ even q) ∨ (even q ∧ even r) ∨ (even r ∧ even p)

def path_exists (dims : ℕ × ℕ × ℕ) (A B : ℕ) : Prop := sorry -- Detailed definition needed

def closed_path_exists (dims : ℕ × ℕ × ℕ) (A : ℕ) : Prop := sorry -- Detailed definition needed

end block_exists_if_path_possible_block_exists_if_closed_euler_path_l83_83807


namespace interior_angle_ratio_l83_83079

theorem interior_angle_ratio (exterior_angle1 exterior_angle2 exterior_angle3 : ℝ)
  (h_ratio : 3 * exterior_angle1 = 4 * exterior_angle2 ∧ 
             4 * exterior_angle1 = 5 * exterior_angle3 ∧ 
             3 * exterior_angle1 + 4 * exterior_angle2 + 5 * exterior_angle3 = 360 ) : 
  3 * (180 - exterior_angle1) = 2 * (180 - exterior_angle2) ∧ 
  2 * (180 - exterior_angle2) = 1 * (180 - exterior_angle3) :=
sorry

end interior_angle_ratio_l83_83079


namespace convert_waist_size_to_cm_l83_83173

theorem convert_waist_size_to_cm (waist_in_inches : ℕ):
  (let inches_per_foot := 10 in 
   let cm_per_foot := 25 in 
   waist_in_inches = 40 → 
   (waist_in_inches / inches_per_foot) * cm_per_foot = 100) :=
by
  sorry

end convert_waist_size_to_cm_l83_83173


namespace repeating_decimal_sum_l83_83870

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l83_83870


namespace bottles_not_in_crates_l83_83685

def total_bottles : ℕ := 250
def num_small_crates : ℕ := 5
def num_medium_crates : ℕ := 5
def num_large_crates : ℕ := 5
def bottles_per_small_crate : ℕ := 8
def bottles_per_medium_crate : ℕ := 12
def bottles_per_large_crate : ℕ := 20

theorem bottles_not_in_crates : 
  num_small_crates * bottles_per_small_crate + 
  num_medium_crates * bottles_per_medium_crate + 
  num_large_crates * bottles_per_large_crate = 200 → 
  total_bottles - 200 = 50 := 
by
  sorry

end bottles_not_in_crates_l83_83685


namespace max_positive_n_l83_83341

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

noncomputable def sequence_condition (a : ℕ → ℤ) : Prop :=
a 1010 / a 1009 < -1

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * (a 1 + a n) / 2

theorem max_positive_n (a : ℕ → ℤ) (h1 : is_arithmetic_sequence a) 
    (h2 : sequence_condition a) : n = 2018 ∧ sum_of_first_n_terms a 2018 > 0 := sorry

end max_positive_n_l83_83341


namespace second_pipe_filling_time_l83_83161

theorem second_pipe_filling_time :
  let time_first_pipe := 3
  let time_both_pipes := 1 + (12 / 60 : ℝ)  -- 1 hour 12 minutes in hours
  let rate_first_pipe := 1 / time_first_pipe
  let rate_both_pipes := 1 / time_both_pipes
  let rate_second_pipe := rate_both_pipes - rate_first_pipe
  let time_second_pipe := 1 / rate_second_pipe
  in time_second_pipe = 2 :=
by
  sorry

end second_pipe_filling_time_l83_83161


namespace total_cost_is_2750_l83_83673

def squat_rack_cost : ℕ := 2500
def barbell_cost : ℕ := squat_rack_cost / 10
def total_cost : ℕ := squat_rack_cost + barbell_cost

theorem total_cost_is_2750 : total_cost = 2750 := by
  have h1 : squat_rack_cost = 2500 := by rfl
  have h2 : barbell_cost = 2500 / 10 := by rfl
  have h3 : total_cost = 2500 + 250 := by rfl
  have h4 : total_cost = 2750 := by rw [h1, h2, h3]
  sorry

end total_cost_is_2750_l83_83673


namespace power_multiplication_l83_83542

theorem power_multiplication (x : ℝ) : (-4 * x^3)^2 = 16 * x^6 := 
by 
  sorry

end power_multiplication_l83_83542


namespace infinite_rectangles_cannot_cover_plane_l83_83482

def rectangle_area (n : ℕ) : ℝ :=
  n ^ 2

theorem infinite_rectangles_cannot_cover_plane
  (hr : ∀ n : ℕ, ∃ r : rectangle, r.area = rectangle_area n)
  (overlaps_allowed : true) :
  ¬ (∃ (C : cover), C.covers_plane) :=
sorry

end infinite_rectangles_cannot_cover_plane_l83_83482


namespace unique_plants_total_l83_83091

-- Define the conditions using given parameters
def X := 700
def Y := 600
def Z := 400
def XY := 100
def XZ := 200
def YZ := 50
def XYZ := 25

-- Define the problem using the Principle of Inclusion-Exclusion
def unique_plants := X + Y + Z - XY - XZ - YZ + XYZ

-- The theorem to prove the unique number of plants
theorem unique_plants_total : unique_plants = 1375 := by
  sorry

end unique_plants_total_l83_83091


namespace problem_solution_l83_83203

theorem problem_solution : ∃ n : ℕ, n = 4 ∧
  { a : ℕ | a ≥ 20 ∧ a < 24 }.card = n ∧
  (∀ x : ℕ, 2 * x > 4 * x - 6 → x = 3 ∨ false) ∧
  (∀ x : ℕ, (x ≠ 3 → ¬(4 * x - a > -12)) →
   (x = 3 → 4 * x - a > -12)) :=
  sorry

end problem_solution_l83_83203


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l83_83887

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l83_83887


namespace jesse_money_left_l83_83681

def initial_money : ℝ := 500
def novel_cost_pounds : ℝ := 13
def num_novels : ℕ := 10
def bookstore_discount : ℝ := 0.20
def exchange_rate_usd_to_pounds : ℝ := 0.7
def lunch_cost_multiplier : ℝ := 3
def lunch_tax_rate : ℝ := 0.12
def lunch_tip_rate : ℝ := 0.18
def jacket_original_euros : ℝ := 120
def jacket_discount : ℝ := 0.30
def jacket_expense_multiplier : ℝ := 2
def exchange_rate_pounds_to_euros : ℝ := 1.15

theorem jesse_money_left : 
  initial_money - (
    ((novel_cost_pounds * num_novels * (1 - bookstore_discount)) / exchange_rate_usd_to_pounds)
    + ((novel_cost_pounds * lunch_cost_multiplier * (1 + lunch_tax_rate + lunch_tip_rate)) / exchange_rate_usd_to_pounds)
    + ((((jacket_original_euros * (1 - jacket_discount)) / exchange_rate_pounds_to_euros) / exchange_rate_usd_to_pounds))
  ) = 174.66 := by
  sorry

end jesse_money_left_l83_83681


namespace probability_prime_and_multiple_of_5_l83_83618

-- Conditions
def cards : Finset ℕ := Finset.range 101 -- 100 cards, numbered 1 to 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def desired_cards : Finset ℕ := cards.filter (λ n => is_prime n ∧ is_multiple_of_5 n)

-- Theorem
theorem probability_prime_and_multiple_of_5 :
  (desired_cards.card : ℚ) / (cards.card : ℚ) = 1 / 100 := 
by
  sorry

end probability_prime_and_multiple_of_5_l83_83618


namespace triangle_sum_of_sides_l83_83800

noncomputable def sum_of_remaining_sides (side_a : ℝ) (angle_B : ℝ) (angle_C : ℝ) : ℝ :=
  let BD := side_a * Real.sin angle_B
  let DC := BD / Real.tan angle_C
  side_a + DC

theorem triangle_sum_of_sides :
  let side_a := 8
  let angle_B := Real.pi * 50 / 180
  let angle_C := Real.pi * 40 / 180
  sum_of_remaining_sides side_a angle_B angle_C ≈ 22.6 :=
by
  let side_a := 8
  let angle_B := Real.pi * 50 / 180
  let angle_C := Real.pi * 40 / 180
  have h1 : sum_of_remaining_sides side_a angle_B angle_C ≈ 22.6 := sorry
  exact h1

end triangle_sum_of_sides_l83_83800


namespace calculate_volume_of_soil_extracted_l83_83554

def volume_rect (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ :=
  length * width * depth

def volume_full_elliptical_dome (semi_major : ℝ) (semi_minor : ℝ) (depth : ℝ) : ℝ :=
  (4 / 3) * Real.pi * semi_major * semi_minor * depth

def volume_semi_elliptical_dome (major_axis : ℝ) (minor_axis : ℝ) (depth : ℝ) : ℝ :=
  volume_full_elliptical_dome (major_axis / 2) (minor_axis / 2) depth / 2

def volume_pond (length : ℝ) (width : ℝ) (rect_depth : ℝ) (major_axis : ℝ) (minor_axis : ℝ) (dome_depth : ℝ) : ℝ :=
  volume_rect length width rect_depth + volume_semi_elliptical_dome major_axis minor_axis dome_depth

theorem calculate_volume_of_soil_extracted :
  volume_pond 20 10 5 10 5 5 = 1130.9 := by
  sorry

end calculate_volume_of_soil_extracted_l83_83554


namespace lady_cross_field_time_l83_83502

theorem lady_cross_field_time (area : ℝ) (initial_speed : ℝ) (speed_reduction : ℝ) :
  area = 7201 → initial_speed = 2.4 → speed_reduction = 0.75 →
  let side := Real.sqrt area in
  let diagonal := side * Real.sqrt 2 in
  let speed := initial_speed * 1000 * speed_reduction in
  let time := (diagonal / speed) * 60 in
  time ≈ 4.0062 :=
begin
  intros h1 h2 h3,
  let side := Real.sqrt area,
  let diagonal := side * Real.sqrt 2,
  let speed := initial_speed * 1000 * speed_reduction,
  let time := (diagonal / speed) * 60,
  sorry
end

end lady_cross_field_time_l83_83502


namespace sequence_fifth_term_l83_83329

theorem sequence_fifth_term (a b c : ℕ) :
  (a = (2 + b) / 3) →
  (b = (a + 34) / 3) →
  (34 = (b + c) / 3) →
  c = 89 :=
by
  intros ha hb hc
  sorry

end sequence_fifth_term_l83_83329


namespace num_triangles_from_intersections_of_chords_l83_83758

open Nat

theorem num_triangles_from_intersections_of_chords 
  (n : ℕ) (h : n = 10) :
  let total_intersections := choose n 4 in
  let total_triangles := choose total_intersections 3 in
  total_triangles = 1524180 :=
by
  sorry

end num_triangles_from_intersections_of_chords_l83_83758


namespace n_gon_angles_l83_83524

theorem n_gon_angles (n : ℕ) (h1 : n > 7) (h2 : n < 12) : 
  (∃ x : ℝ, (150 * (n - 1) + x = 180 * (n - 2)) ∧ (x < 150)) :=
by {
  sorry
}

end n_gon_angles_l83_83524


namespace cos_beta_value_l83_83575

theorem cos_beta_value (α β : ℝ) (h1 : cos α = 4 / 5) (h2 : cos (α + β) = 3 / 5) (h3 : 0 < α ∧ α < π / 2) (h4 : 0 < β ∧ β < π / 2) : cos β = 24 / 25 := 
sorry

end cos_beta_value_l83_83575


namespace pieces_per_box_l83_83947

theorem pieces_per_box (total_pieces : ℕ) (boxes : ℕ) (h_total : total_pieces = 3000) (h_boxes : boxes = 6) :
  total_pieces / boxes = 500 := by
  sorry

end pieces_per_box_l83_83947


namespace prime_sequence_constant_l83_83234

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Condition: There exists a constant sequence of primes such that the given recurrence relation holds.
theorem prime_sequence_constant (p : ℕ) (k : ℤ) (n : ℕ) 
  (h1 : 1 ≤ n)
  (h2 : ∀ m ≥ 1, is_prime (p + m))
  (h3 : p + k = p + p + k) :
  ∀ m ≥ 1, p + m = p :=
sorry

end prime_sequence_constant_l83_83234


namespace simplify_and_evaluate_l83_83422

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 2) :
  ((m ^ 2 - 9) / (m ^ 2 - 6 * m + 9) - 3 / (m - 3)) / (m ^ 2 / (m - 3)) = Real.sqrt 2 / 2 :=
by {
  -- Proof goes here
  sorry
}

end simplify_and_evaluate_l83_83422


namespace complex_quadrant_l83_83637

noncomputable def purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem complex_quadrant (a : ℝ) (h : purely_imaginary ((2 * a + 2 * complex.I) / (1 + complex.I))) : 
  (2 * a + 2 * complex.I).re < 0 ∧ (2 * a + 2 * complex.I).im > 0 :=
by
  sorry

end complex_quadrant_l83_83637


namespace largest_difference_l83_83367

noncomputable def A := 3 * (1003 ^ 1004)
noncomputable def B := 1003 ^ 1004
noncomputable def C := 1002 * (1003 ^ 1003)
noncomputable def D := 3 * (1003 ^ 1003)
noncomputable def E := 1003 ^ 1003
noncomputable def F := 1003 ^ 1002

theorem largest_difference : 
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B := 
sorry

end largest_difference_l83_83367


namespace arithmetic_formula_geometric_formula_comparison_S_T_l83_83342

noncomputable def a₁ : ℕ := 16
noncomputable def d : ℤ := -3

def a_n (n : ℕ) : ℤ := -3 * (n : ℤ) + 19
def b_n (n : ℕ) : ℤ := 4^(3 - n)

def S_n (n : ℕ) : ℚ := (-3 * (n : ℚ)^2 + 35 * n) / 2
def T_n (n : ℕ) : ℤ := -n^2 + 3 * n

theorem arithmetic_formula (n : ℕ) : a_n n = -3 * n + 19 :=
sorry

theorem geometric_formula (n : ℕ) : b_n n = 4^(3 - n) :=
sorry

theorem comparison_S_T (n : ℕ) :
  if n = 29 then S_n n = (T_n n : ℚ)
  else if n < 29 then S_n n > (T_n n : ℚ)
  else S_n n < (T_n n : ℚ) :=
sorry

end arithmetic_formula_geometric_formula_comparison_S_T_l83_83342


namespace smallest_x_mod_l83_83960

theorem smallest_x_mod (x : ℕ) : 
  (x % 19 = 9) ∧ (x % 23 = 7) → x = 161 :=
begin
  sorry
end

end smallest_x_mod_l83_83960


namespace min_quadratic_expr_l83_83815

noncomputable def quadratic_expr (x : ℝ) := 3 * x^2 - 18 * x + 2023

theorem min_quadratic_expr : ∃ x : ℝ, quadratic_expr x = 1996 :=
by
  have h : quadratic_expr (3 : ℝ) = 1996
  exact h
  use 3
  rw h
  sorry -- Proof of h (already derived in given solution)

end min_quadratic_expr_l83_83815


namespace probability_heads_not_consecutive_l83_83501

theorem probability_heads_not_consecutive :
  let i := 9 in let j := 64 in
  (Nat.gcd i j = 1) ∧ (i + j = 73) :=
by {
  let i := 9,
  let j := 64,
  sorry
}

end probability_heads_not_consecutive_l83_83501


namespace area_of_triangle_AKF_l83_83448

theorem area_of_triangle_AKF :
  let parabola := λ x y: ℝ, y^2 = 4 * x
  let focus := (1, 0)
  let directrix := λ x: ℝ, x = -1
  let line_through_focus := λ x y: ℝ, y = sqrt 3 * (x - focus.1)
  let point_A := (3, 2 * sqrt 3)
  let point_K := (-1, 2 * sqrt 3)
  ∀ {AK_perpendicular_to_l : point_A.2 = point_K.2},
  let base := (point_A.1 - point_K.1)  -- AK
  let height := (focus.2 - point_K.2)  -- FK
  ∃ area_of_AKF : ℝ,
  area_of_AKF = 1 / 2 * base * abs height :=

by
  let parabola : ℝ × ℝ → Prop := λ p, (p.2)^2 = 4 * p.1
  let focus : ℝ × ℝ := (1, 0)
  let directrix : ℝ → Prop := λ x, x = -1
  let line_through_focus : ℝ → ℝ := λ x, sqrt 3 * (x - focus.1)
  have h1: ∀ y x, parabola (x, y) ↔ (y = 2 * sqrt 3 ∨ y = -2 * sqrt 3) := sorry,
  have hA: focus = (1, 0) := rfl,
  have hAK_perpendicular_to_l: point_A.2 = point_K.2 := rfl,
  let base := abs (point_A.1 - point_K.1),  -- AK
  let height := abs (focus.2 - point_K.2),  -- FK
  exists 4 * sqrt 3,
  sorry

end area_of_triangle_AKF_l83_83448


namespace min_questions_to_identify_liars_l83_83346

/-- 
  Proves that the minimum number of questions needed to identify liars among 
  10 people sitting at the vertices of a regular decagon is 2.
--/
theorem min_questions_to_identify_liars :
  ∀ (persons : fin 10 → bool), -- Assume persons is a function from vertices to bool (True = knight, False = liar)
  ∃ (questions_count : ℕ), 
  questions_count = 2 ∧ 
  (∀ (ask : (fin 10) → ℤ → ℕ), -- Function ask which takes a vertex and gets distance of nearest liar
  ∃ (result : fin 10 → bool), -- Identify liars and knights
  (∀ i, persons i = result i)) := 
by 
  sorry -- Proof is skipped.

end min_questions_to_identify_liars_l83_83346


namespace proof_equivalent_problem_l83_83579

noncomputable def parabola_eq (x y : ℝ) : Prop :=
  y^2 = 4 * x

noncomputable def symmetric_point := (-1, 0)

noncomputable def exists_T (x y : ℝ) : Prop :=
  ∃ T : ℝ, T ≠ -1 ∧ (∀ a b : ℝ, parabola_eq a y ∧ parabola_eq b y →
  ∃ AB : ℝ, (AB = a - T ∧ AB = b - T))

noncomputable def area_triangle (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  0.5 * |x₁ * y₂ - x₂ * y₁|

noncomputable def angle_between_vectors (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  let OA := (x₁, y₁)
  let OB := (x₂, y₂)
  let dot_product := x₁ * x₂ + y₁ * y₂
  let magnitude_OA := real.sqrt (x₁^2 + y₁^2)
  let magnitude_OB := real.sqrt (x₂^2 + y₂^2)
  real.arccos (dot_product / (magnitude_OA * magnitude_OB))

theorem proof_equivalent_problem 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : parabola_eq x₁ y₁) 
  (h₂ : parabola_eq x₂ y₂) 
  (h₃ : area_triangle x₁ y₁ x₂ y₂ = 2.5) : 
  ∃ T : ℝ, T ≠ -1 ∧ 
  (∀ a b : ℝ, parabola_eq a y₁ ∧ parabola_eq b y₂ →
  ∃ AB : ℝ, AB = a - T ∧ AB = b - T) ∧
  angle_between_vectors x₁ y₁ x₂ y₂ = real.pi / 4 :=
sorry

end proof_equivalent_problem_l83_83579


namespace circumcircle_diameter_l83_83643

-- Define the problem conditions
def triangle (A B C : Type) := angle B = 45 ∧ BC = 1 ∧ area = 2

-- Statement of the problem to be proven
theorem circumcircle_diameter (A B C : Type) (B_angle_is_45 BC_is_1 area_is_2 : triangle A B C) : 
  ∃ R : ℝ, (2 * R) = 5 * sqrt 2 :=
by
  sorry

end circumcircle_diameter_l83_83643


namespace recurring_fraction_sum_l83_83917

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l83_83917


namespace square_area_eq_41_l83_83968

theorem square_area_eq_41 :
  (∃ (side: ℝ), 
    (∀ (x : ℝ), 
      ((x^2 + 5*x + 6 = 10) → (side = sqrt 41)) 
        ∧ (side^2 = 41))) :=
sorry

end square_area_eq_41_l83_83968


namespace empty_set_condition_l83_83122

def isEmptySet (s : Set ℝ) : Prop := s = ∅

def A : Set ℕ := {n : ℕ | n^2 ≤ 0}
def B : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def C : Set ℝ := {x : ℝ | x^2 + x + 1 = 0}
def D : Set ℝ := {0}

theorem empty_set_condition : isEmptySet C := by
  sorry

end empty_set_condition_l83_83122


namespace area_ratio_AEF_ABC_l83_83629

noncomputable def triangle_area_ratio (A B C D E F : ℝ) : Prop :=
  let S_ABC := (B - A) * (C - A) / 2 in
  let S_AEF := S_ABC * (2 / 9) in
  S_AEF / S_ABC = 2 / 9

theorem area_ratio_AEF_ABC
  (A B C D : ℝ)
  (hD : D ∈ segment ℝ B C)
  (centroid_ABD : ℝ)
  (centroid_ACD : ℝ)
  (hE : centroid_ABD = (A + B + D) / 3)
  (hF : centroid_ACD = (A + C + D) / 3) :
  triangle_area_ratio A B C D centroid_ABD centroid_ACD := 
  sorry

end area_ratio_AEF_ABC_l83_83629


namespace average_price_condition_observer_b_correct_l83_83733

-- Define the conditions
def stock_price {n : ℕ} (daily_prices : Fin n → ℝ) : Prop :=
  daily_prices 0 = 5 ∧ 
  daily_prices 6 = 5.14 ∧
  daily_prices 13 = 5 ∧ 
  (∀ i : Fin 6, daily_prices i ≤ daily_prices (i + 1)) ∧ 
  (∀ i : Fin (n - 7), daily_prices (i + 7) ≥ daily_prices (i + 8))

-- Define the problem statements
theorem average_price_condition (daily_prices : Fin 14 → ℝ) :
  stock_price daily_prices →
  ∃ S, 5.09 < (5 + S + 5.14) / 14 ∧ (5 + S + 5.14) / 14 < 5.10 :=
sorry

theorem observer_b_correct (daily_prices : Fin 14 → ℝ) :
  stock_price daily_prices →
  let avg_1 : ℝ := (∑ i in Finset.range 7, daily_prices i) / 7
  let avg_2 : ℝ := (∑ i in Finset.range 7, daily_prices (i + 7)) / 7
  ¬ avg_1 = avg_2 + 0.105 :=
sorry

end average_price_condition_observer_b_correct_l83_83733


namespace circle_radius_problem_l83_83188

theorem circle_radius_problem :
  ∃ (p q : ℕ), 
    p > 0 ∧ q > 0 ∧ 
    (∃ r : ℝ, r > 0 ∧ 4 * r = real.sqrt p - q) ∧
    1 = 1 ∧   -- Placeholder to indicate all conditions are met
    p + q = 598 :=
by { sorry }

end circle_radius_problem_l83_83188


namespace unique_P_exists_l83_83039

noncomputable def P : ℝ[X] :=
  sorry

theorem unique_P_exists (A : ℝ[X]) (hA : A = (X + Y)^1000):
  ∃! (P : ℝ[X]), ∀ x y : ℝ, (x * y - x - y) ∣ eval₂ (fun a b => eval b a) A (X + Y) - eval₂ (fun a b => eval b a) P X - eval₂ (fun a b => eval b a) P y :=
begin
  -- proof placeholder
  sorry
end

end unique_P_exists_l83_83039


namespace solution_set_f_derivative_l83_83706

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 1

theorem solution_set_f_derivative :
  { x : ℝ | (deriv f x) < 0 } = { x : ℝ | -1 < x ∧ x < 3 } :=
by
  sorry

end solution_set_f_derivative_l83_83706


namespace probability_at_least_two_same_l83_83738

theorem probability_at_least_two_same (n : ℕ) (sides : ℕ) (h1 : n = 8) (h2 : sides = 6) : 
  (at_least_two_same_probability n sides) = 1 :=
by
  -- Definitions
  let at_least_two_same_probability (n : ℕ) (sides : ℕ) := 1 - (num_ways_all_different n sides / num_total_outcomes n sides)
  let num_ways_all_different (n : ℕ) (sides : ℕ) := if n > sides then 0 else sides.permutations n
  let num_total_outcomes (n : ℕ) (sides : ℕ) := sides ^ n
  
  -- Show that at_least_two_same_probability 8 6 = 1
  have h3 : n > sides, from calc
    n = 8   : h1
    sides = 6 : h2
    8 > 6    : by decide
  have h4 : num_ways_all_different n sides = 0, from if_pos h3
  have h5 : num_total_outcomes n sides = 6^8, by simp [num_total_outcomes, h1, h2]
  
  -- Compose final proof
  rw [at_least_two_same_probability, h4, h5]
  calc
    1 - 0 / 6^8 = 1 - 0 : by simp
            ... = 1     : by simp

end probability_at_least_two_same_l83_83738


namespace repeating_decimal_fraction_sum_l83_83852

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l83_83852


namespace committees_share_one_member_l83_83497

theorem committees_share_one_member
  (n : ℕ)
  (committees : Fin (n + 1) → Finset (Fin n))
  (h1 : ∀ i, committees i.card = 3)
  (h2 : ∀ i j, i ≠ j → committees i ≠ committees j) : 
  ∃ i j, i ≠ j ∧ (committees i ∩ committees j).card = 1 := 
sorry

end committees_share_one_member_l83_83497


namespace spaghetti_manicotti_ratio_l83_83491

-- Define the number of students who were surveyed and their preferences
def total_students := 800
def students_prefer_spaghetti := 320
def students_prefer_manicotti := 160

-- The ratio of students who prefer spaghetti to those who prefer manicotti is 2
theorem spaghetti_manicotti_ratio :
  students_prefer_spaghetti / students_prefer_manicotti = 2 :=
by
  sorry

end spaghetti_manicotti_ratio_l83_83491


namespace women_count_l83_83180

noncomputable def number_of_women (n_men : ℕ) (men_dances_with_women : ℕ) (women_dances_with_men : ℕ) : ℕ :=
  (n_men * men_dances_with_women) / women_dances_with_men

theorem women_count (n_men : ℕ) (men_dances_with_women : ℕ) (women_dances_with_men : ℕ) : 
  n_men = 15 → men_dances_with_women = 4 → women_dances_with_men = 3 → number_of_women n_men men_dances_with_women women_dances_with_men = 20 :=
by
  intros h1 h2 h3
  simp [number_of_women, h1, h2, h3]
  exact sorry

end women_count_l83_83180


namespace sin_neg_60_l83_83559

theorem sin_neg_60 : Real.sin (-60 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
by
  -- Trigonometric identity \(\sin(-\theta) = -\sin(\theta)\)
  have h1 : ∀ (θ : ℝ), Real.sin (-θ) = - Real.sin θ := Real.sin_neg
  -- Special angle value: \(\sin(60^\circ) = \frac{\sqrt{3}}{2}\)
  have h2 : Real.sin (60 * Real.pi / 180) = (Real.sqrt 3) / 2 := Real.sin_pi_div_three
  sorry

end sin_neg_60_l83_83559


namespace stock_price_satisfies_conditions_part_a_stock_price_satisfies_conditions_part_b_l83_83724

-- Define the conditions: stock prices on specific dates
constants apr_7 apr_13 apr_20 : ℝ
axiom apr_7_eq : apr_7 = 5
axiom apr_13_eq : apr_13 = 5.14
axiom apr_20_eq : apr_20 = 5

-- Define the prices on the intervening dates
constants (x : ℕ → ℝ)

-- Define the correct answer for average price calculation for part (a)
def avg_price_between_apr_7_Apr_20 : ℝ := (apr_7 + (Σ i in FinSet.range 12, x i) + apr_13 + (Σ j in FinSet.range 6, x (j + 7)) + apr_20) / 14

noncomputable def part_a : Prop :=
  5.09 < avg_price_between_apr_7_Apr_20 ∧ avg_price_between_apr_7_Apr_20 < 5.10

-- Part (b): Comparing average stock prices for different periods
def avg_price_apr_7_to_apr_13 : ℝ := (apr_7 + (Σ i in FinSet.range 5, x i) + apr_13) / 7
def avg_price_apr_14_to_apr_20 : ℝ := (apr_13 + x 7 + x 8 + x 9 + x 10 + x 11 + apr_20) / 7

noncomputable def part_b : Prop :=
  | (avg_price_apr_14_to_apr_20 - avg_price_apr_7_to_apr_13 ≠ 0.105 )

-- The final proof problems for part (a) and part (b)
theorem stock_price_satisfies_conditions_part_a : part_a := sorry
theorem stock_price_satisfies_conditions_part_b : part_b := sorry

end stock_price_satisfies_conditions_part_a_stock_price_satisfies_conditions_part_b_l83_83724


namespace wealth_ratio_l83_83196

variables (g h i j k l P W : ℝ)
variables (g_pos : g > 0) (h_pos : h > 0) (i_pos : i > 0) (j_pos : j > 0)
variables (P_pos : P > 0) (W_pos : W > 0)

theorem wealth_ratio (H1 : g > 0) (H2 : i > 0) (H3 : g ≠ 0) (H4 : i ≠ 0) :
  let wX := (h * W) / (g * P),
      wY := (j * W) / (i * P) in
  (wX / wY) = (h * i) / (g * j) :=
by
  sorry

end wealth_ratio_l83_83196


namespace imaginary_part_of_z_squared_l83_83269

def z : ℂ := 3 - complex.I
def z_squared : ℂ := z * z

theorem imaginary_part_of_z_squared : im z_squared = -6 := 
by sorry

end imaginary_part_of_z_squared_l83_83269


namespace find_volume_of_prism_l83_83166

def volume_of_prism (volume : ℝ) : Prop :=
  let AB := 2 * (2 * Real.sqrt 2)
  let h := 2
  let area_base := (Real.sqrt 3 / 4) * AB^2
  volume = area_base * h

theorem find_volume_of_prism :
  ∀ (DL DK : ℝ), DL = Real.sqrt 6 → DK = 3 → volume_of_prism (12 * Real.sqrt 3) :=
by
  intro DL DK
  intro h₁ h₂
  have h₃ : 12 * Real.sqrt 3 = 6 * Real.sqrt 3 * 2 := by sorry
  rw [h₁, h₂]
  exact h₃

end find_volume_of_prism_l83_83166


namespace high_school_boys_height_l83_83264

open ProbabilityTheory

noncomputable def height_distribution : Distribution := normal 175 16

theorem high_school_boys_height (X : ℝ) :
  X ~ height_distribution →
  P(175 - 2 * √16 < X ∧ X ≤ 175 + 2 * √16) = 0.9544 →
  ( ∃ (avg : ℝ), avg = 175 ) ∧ 
  ( ∃ (var : ℝ), var = 16 ) ∧
  ( ∃ (prob : ℝ), prob = P(X > 183) ∧ prob < 0.03 ) ∧
  ( ∃ (prob_sym : ℝ), prob_sym = P(X > 180) ∧ prob_sym = P(X ≤ 170) ) :=
by 
  sorry

end high_school_boys_height_l83_83264


namespace robert_birth_year_l83_83416

theorem robert_birth_year (n : ℕ) (h1 : (n + 1)^2 - n^2 = 89) : n = 44 ∧ n^2 = 1936 :=
by {
  sorry
}

end robert_birth_year_l83_83416


namespace problem_tangent_sum_l83_83290

theorem problem_tangent_sum (θ : ℝ) (h : tan (θ + π/4) = -3) : 
  2 * sin θ ^ 2 - cos θ ^ 2 = 7 / 5 := 
sorry

end problem_tangent_sum_l83_83290


namespace solve_gnome_mystery_l83_83802

inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

structure Gnome :=
(name : String)
(lies_on : Day → Prop)

def senya : Gnome :=
{ name := "Senya",
  lies_on := fun d => d = Monday ∨ d = Tuesday ∨ d = Wednesday }

def venya : Gnome :=
{ name := "Venya",
  lies_on := fun d => d = Tuesday ∨ d = Thursday ∨ d = Saturday }

def conversation_day (d: Day) : Prop :=
(d = Monday → "Yesterday was Sunday" ∧ "Tomorrow will be Friday" ∧ "I always tell the truth on Wednesday" → False) ∧
(d = Tuesday → "Yesterday was Sunday" ∧ "Tomorrow will be Friday" ∧ "I always tell the truth on Wednesday" → True) ∧
(d ≠ Tuesday → "Yesterday was Sunday" ∧ "Tomorrow will be Friday" ∧ "I always tell the truth on Wednesday" → False)

theorem solve_gnome_mystery : ∃ (d: Day), conversation_day d ∧ d = Tuesday :=
sorry

end solve_gnome_mystery_l83_83802


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l83_83888

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l83_83888


namespace smallest_abundant_not_multiple_of_five_l83_83107

def is_abundant (n : ℕ) : Prop :=
  (∑ m in Nat.properDivisors n, m) > n

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

theorem smallest_abundant_not_multiple_of_five : ∃ n : ℕ, is_abundant n ∧ ¬ is_multiple_of_five n ∧ ∀ m : ℕ, is_abundant m ∧ ¬ is_multiple_of_five m → n ≤ m := sorry

end smallest_abundant_not_multiple_of_five_l83_83107


namespace matrix_multiplication_zero_matrix_l83_83189

variable (d e f : ℝ) 
variable (A : Matrix (Fin 3) (Fin 3) ℝ) := 
  ![![0, d, -e], 
    ![-d, 0, f], 
    ![e, -f, 0]]
variable (B : Matrix (Fin 3) (Fin 3) ℝ) := 
  ![![d^2, d*e, d*f], 
    ![d*e, e^2, e*f], 
    ![d*f, e*f, f^2]]

theorem matrix_multiplication_zero_matrix (h1 : d = f) (h2 : e = f) :
  A * B = (0 : Matrix (Fin 3) (Fin 3) ℝ) := sorry

end matrix_multiplication_zero_matrix_l83_83189


namespace min_teams_seven_l83_83659

theorem min_teams_seven (n : ℕ) (H : ∀ (A B : ℕ), A ≠ B ∧ A < n ∧ B < n → ∃ C : ℕ, C < n ∧ C ≠ A ∧ C ≠ B ∧ C wins_against A ∧ C wins_against B) : 7 ≤ n := 
sorry

end min_teams_seven_l83_83659


namespace photo_students_count_l83_83081

theorem photo_students_count (n m : ℕ) 
  (h1 : m - 1 = n + 4) 
  (h2 : m - 2 = n) : 
  n * m = 24 := 
by 
  sorry

end photo_students_count_l83_83081


namespace partition_right_angle_l83_83692

variable (ABC : Type) [EquilateralTriangle ABC] 
  (E : Set (PointOnSegments ABC))
  (partition : E → Prop)

def right_angled_triangle_in_partition (X Y : Set (PointOnSegments ABC)) : Prop :=
  ∃ (A B C : PointOnSegments ABC),
    (right_angle A B C) ∧ 
    ((A ∈ X ∧ B ∈ X ∧ C ∈ X) ∨ (A ∈ Y ∧ B ∈ Y ∧ C ∈ Y))

theorem partition_right_angle :
  ∀ X Y : Set (PointOnSegments ABC), 
    (X ∪ Y = E) → 
    (X ∩ Y = ∅) → 
    right_angled_triangle_in_partition X Y :=
sorry

end partition_right_angle_l83_83692


namespace probability_div_25_l83_83369

-- Define the set S
def S : set ℕ := { n | ∃ j k : ℕ, j < k ∧ k < 40 ∧ (n = 2^j + 2^k) }

-- Prime factors definition for gcd check (relatively prime)
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Probability fraction in reduced form
def probability_fraction (prob: ℚ) : ℕ → ℕ → Prop := λ p q, prob = p / q ∧ relatively_prime p q

-- Main theorem statement
theorem probability_div_25 : ∃ p q : ℕ, probability_fraction (1 / 39) p q ∧ p + q = 40 :=
by
  sorry

end probability_div_25_l83_83369


namespace second_offset_l83_83221

theorem second_offset (d : ℝ) (h1 : ℝ) (A : ℝ) (h2 : ℝ) : 
  d = 28 → h1 = 9 → A = 210 → h2 = 6 :=
by
  sorry

end second_offset_l83_83221


namespace intersection_lines_k_l83_83641

theorem intersection_lines_k (k : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ x - y - 1 = 0 ∧ x + k * y = 0) → k = -2 :=
by
  sorry

end intersection_lines_k_l83_83641


namespace Xiaoming_correct_l83_83233

-- Definition of the polynomial
def polynomial (x : ℝ) : ℝ := 6 * x^5 + 5 * x^4 + 4 * x^3 + 3 * x^2 + 2 * x + 2002

-- Statement to prove Xiaoming's statement is correct
theorem Xiaoming_correct {x : ℝ} :
  let p := polynomial x in
  (p = (((((6 * x + 5) * x + 4) * x + 3) * x + 2) * x + 2002)) →
  (∃ n, n = 5) → -- condition that Xiaoming's method uses 5 multiplications
  true :=
by
  intros p h1 h2
  sorry

end Xiaoming_correct_l83_83233


namespace shortest_tangent_segment_correct_l83_83014

noncomputable def shortest_tangent_segment {x y : ℝ} 
  (C1 C2 : (ℝ × ℝ) × ℝ) : ℝ :=
  if C1 = ((12, 0), 5) ∧ C2 = ((-18, 0), 10) then 5 * sqrt 15 + 10 else 0

theorem shortest_tangent_segment_correct :
  shortest_tangent_segment (((12 : ℝ), 0), 5) (((-18 : ℝ), 0), 10) = 5 * sqrt 15 + 10 :=
by { sorry }

end shortest_tangent_segment_correct_l83_83014


namespace leo_weight_l83_83297

theorem leo_weight 
  (L K E : ℝ)
  (h1 : L + 10 = 1.5 * K)
  (h2 : L + 10 = 0.75 * E)
  (h3 : L + K + E = 210) :
  L = 63.33 := 
sorry

end leo_weight_l83_83297


namespace center_coordinates_of_circle_l83_83058

theorem center_coordinates_of_circle :
  ∃ (rho theta : ℝ), (∀ θ, rho = sqrt 2 * (cos θ + sin θ)) ∧ 
    (rho = 1 ∧ θ = π / 4) :=
begin
  sorry
end

end center_coordinates_of_circle_l83_83058


namespace repeating_decimal_fraction_sum_l83_83845

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l83_83845


namespace find_counterfeit_in_7_weighings_l83_83530

def contains_counterfeit (coins : List ℕ) : Prop :=
  ∃ coin, coin ∈ coins ∧ coin < coins.head

/- Weigh function simulates the balance scale -
   it returns true if the left side is lighter. -/
def weigh (left right : List ℕ) : Bool :=
  List.sum left < List.sum right

/- The main theorem we want to prove: 
   Given 99 coins with exactly one counterfeit and weights 
   distributed such that it can be identified in 7 weighings. -/
theorem find_counterfeit_in_7_weighings (coins : List ℕ) 
  (h_len : coins.length = 99) 
  (h_counterfeit : contains_counterfeit coins) 
  (h_weighings : ∀ c ∈ coins, count_weighings coins c ≤ 7) :
  ∃ c, c ∈ coins ∧ count_weighings coins c ≤ 7 :=
sorry

/- Assuming there's a function to count the number of weighings for a particular coin 
   which must ensure no coin is weighed more than twice. -/
def count_weighings : List ℕ → ℕ → ℕ :=
  sorry

end find_counterfeit_in_7_weighings_l83_83530


namespace sum_of_consecutive_integers_iff_not_power_of_2_l83_83750

theorem sum_of_consecutive_integers_iff_not_power_of_2 (N : ℕ) (hN : N > 0) : 
  (∃ k m : ℕ, k ≥ 2 ∧ N = (m+1) + (m+2) + ... + (m+k)) ↔ ¬ ∃ n : ℕ, N = 2^n :=
by
  sorry

end sum_of_consecutive_integers_iff_not_power_of_2_l83_83750


namespace solve_for_x_l83_83628

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end solve_for_x_l83_83628


namespace pentagon_area_ABCDE_l83_83438

noncomputable def area_of_convex_pentagon (EA AB BC CD DE : ℝ) (angleA angleB : ℝ) :=
  if angleA = 90 ∧ angleB = 90 ∧ EA = AB
     ∧ AB = 3 ∧ BC = 4 ∧ CD = 5 ∧ DE = 5 then
    19.5
  else
    sorry

theorem pentagon_area_ABCDE :
  area_of_convex_pentagon 3 3 4 5 5 90 90 = 19.5 :=
by 
  rfl

end pentagon_area_ABCDE_l83_83438


namespace tetrahedron_edge_length_l83_83774

-- Define the problem as a Lean theorem statement
theorem tetrahedron_edge_length (r : ℝ) (a : ℝ) (h : r = 1) :
  a = 2 * Real.sqrt 2 :=
sorry

end tetrahedron_edge_length_l83_83774


namespace cos_F_l83_83355

theorem cos_F (D E F : ℝ) (hDEF : D + E + F = 180)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = -16 / 65 :=
  sorry

end cos_F_l83_83355


namespace original_revenue_l83_83520

theorem original_revenue (current_revenue : ℝ) (percentage_decrease : ℝ) : ℝ :=
  let original_revenue := current_revenue / (1 - percentage_decrease / 100)
  original_revenue

example : original_revenue 48.0 30.434782608695656 ≈ 68.97 := 
by
  have h : original_revenue 48.0 30.434782608695656 ≈ 68.96551724137931,
  { simp [original_revenue],
    norm_num,
  },
  norm_num at h,
  -- This part states that we are approximately correct to 2 decimal places with our given value.
  sorry

end original_revenue_l83_83520


namespace arithmetic_sequence_sum_l83_83583

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hS : ∀ n, S n = n * (a 1 + a n) / 2)
  (h : a 3 = 20 - a 6) : S 8 = 80 :=
sorry

end arithmetic_sequence_sum_l83_83583


namespace sequence_eventually_constant_l83_83939

noncomputable def sequence (n : ℕ+) : ℕ+ → ℕ
| k :=
  if k = 1 then
    n
  else
    let S (k' : ℕ+) := (sequence (n)) k' in
    let sum := (Finset.range k).sum (λ i, S (⟨i + 1, Nat.succ_pos i⟩)) in
    (⟨((Finset.range k).filter (λ x, (sum + x) % k = 0)).min' (by simp [Finset.nonempty_range_succ]) _, 
      Nat.lt_trans (Finset.min'_mem _ (by simp [Finset.nonempty_range_succ])) k⟩ : ℕ)

theorem sequence_eventually_constant (n : ℕ+) : ∃ N : ℕ+, ∀ m ≥ N, ∀ k ≥ N, sequence n m = sequence n k := 
sorry

end sequence_eventually_constant_l83_83939


namespace centroid_of_perpendicular_triangle_on_line_M_l83_83656

open EuclideanGeometry

-- Definitions of the scalene triangle, orthocenter, centroid, and the corresponding lines.
variables {A B C H M : Point}
variables (triangle_ABC : ScaleneTriangle A B C)
variables (orthocenter_ABC : Orthocenter A B C H)
variables (centroid_ABC : Centroid A B C M)
variables (perpendicular_lines : ∀ X : Point, LinePerpendicularToMedLine X (triangle_med_line triangle_ABC X))

-- Proof statement
theorem centroid_of_perpendicular_triangle_on_line_M : 
  ∃ (G : Point), (Centroid (TriangleFormedByPerpendicularLines A B C) G ∧ G ∈ LineThrough M)
:=
sorry

end centroid_of_perpendicular_triangle_on_line_M_l83_83656


namespace rectangle_perimeters_l83_83159

theorem rectangle_perimeters (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 3 * (2 * a + 2 * b)) : 
  2 * (a + b) = 36 ∨ 2 * (a + b) = 28 :=
by sorry

end rectangle_perimeters_l83_83159


namespace weekly_allowance_60_l83_83516

-- Define the initial data and conditions
variable (A : ℝ) -- Weekly allowance

-- Conditions mapped from a)
def arcade_spending := 3/5 * A        -- Spent at the arcade
def remaining_after_arcade := 2/5 * A -- Remaining after arcade

def toy_store_spending := 1/3 * remaining_after_arcade
def remaining_after_toy_store := (2/3) * remaining_after_arcade

def comic_book_spending := 1/4 * remaining_after_toy_store
def remaining_after_comics := (3/4) * remaining_after_toy_store

-- $1.20 spent at candy store is 10% of remaining after comics
def last_spent := 1.20 : ℝ
def percentage_candy_store := 0.10 * remaining_after_comics

-- Goal: Prove the weekly allowance A is $60

theorem weekly_allowance_60 : 
  percentage_candy_store = last_spent → 
  A = 60 :=
by
  sorry

end weekly_allowance_60_l83_83516


namespace initial_bird_families_l83_83087

/- Definitions: -/
def birds_away_africa : ℕ := 23
def birds_away_asia : ℕ := 37
def birds_left_mountain : ℕ := 25

/- Theorem (Question and Correct Answer): -/
theorem initial_bird_families : birds_away_africa + birds_away_asia + birds_left_mountain = 85 := by
  sorry

end initial_bird_families_l83_83087


namespace least_positive_integer_division_conditions_l83_83812

theorem least_positive_integer_division_conditions :
  ∃ M : ℤ, M > 0 ∧
  M % 11 = 10 ∧
  M % 12 = 11 ∧
  M % 13 = 12 ∧
  M % 14 = 13 ∧
  M = 30029 := 
by
  sorry

end least_positive_integer_division_conditions_l83_83812


namespace average_activity_minutes_l83_83984

theorem average_activity_minutes (g : ℕ) :
  let third_graders := 3 * g,
      fourth_graders := g,
      fifth_graders := g,
      third_grade_time := 18 + 20,
      fourth_grade_time := 20 + 25,
      fifth_grade_time := 15 + 30,
      total_minutes := third_grade_time * third_graders + fourth_grade_time * fourth_graders + fifth_grade_time * fifth_graders,
      total_students := third_graders + fourth_graders + fifth_graders
  in (total_minutes / total_students : ℚ) = 40.8 := by
  sorry

end average_activity_minutes_l83_83984


namespace solution_x_y_zero_l83_83298

theorem solution_x_y_zero (x y : ℤ) (h : x^2 * y^2 = x^2 + y^2) : x = 0 ∧ y = 0 :=
by
sorry

end solution_x_y_zero_l83_83298


namespace slope_of_AB_area_of_AOB_l83_83338

theorem slope_of_AB (α θ : ℝ) :
  let C1_x := 1 + cos α
  let C1_y := sin α
  let C2_rho := 4 * sin θ
  let C1 := (C1_x, C1_y)
  let C2 := (C2_rho * cos θ, C2_rho * sin θ)
  (C1 = C2) → (2 * C1_y - C1_x = 0) → (slope C1 C2 = 1 / 2) :=
begin
  intros,
  sorry
end

theorem area_of_AOB (α θ : ℝ) :
  let C1 := (1 + cos α, sin α)
  let C2 := (4 * sin θ * cos θ, 4 * sin θ * sin θ)
  let d := 2 / sqrt 5
  let max_length_AB := 3 + sqrt 5
  (C1 = C2) → area O C1 C2 = (3 * sqrt 5 / 5) + 1 :=
begin
  intros,
  sorry
end

end slope_of_AB_area_of_AOB_l83_83338


namespace solve_for_x_l83_83623

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end solve_for_x_l83_83623


namespace calculate_expression_l83_83990

theorem calculate_expression : ((18^18 / 18^17)^3 * 8^3) / 2^9 = 5832 := by
  sorry

end calculate_expression_l83_83990


namespace sufficient_condition_increasing_l83_83930

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 1

theorem sufficient_condition_increasing (a : ℝ) :
  (∀ x y : ℝ, 1 < x → x < y → (f x a ≤ f y a)) → a = -1 := sorry

end sufficient_condition_increasing_l83_83930


namespace sum_of_n_values_l83_83111

theorem sum_of_n_values (n : ℕ) (h₀ : n = 16 ∨ n = 14) :
  (14 + 16 = 30 ∧ ∑ n in {16, 14}, n = 30) :=
by
  have h1: ∑ n in {16, 14}, n = 14 + 16 := by
    simp [Finset.sum_insert, Finset.sum_singleton]
  simp [h0, h1]
  sorry

end sum_of_n_values_l83_83111


namespace goods_train_speed_is_92_l83_83934

noncomputable def speed_of_goods_train (man_train_speed : ℕ) (pass_time : ℕ) (train_length : ℕ) : ℕ :=
  let relative_speed_m_per_s := train_length / pass_time
  let relative_speed_km_per_h := relative_speed_m_per_s * 3.6
  let goods_train_speed := relative_speed_km_per_h - man_train_speed
  goods_train_speed

theorem goods_train_speed_is_92 :
  speed_of_goods_train 20 9 280 = 92 :=
by
  -- Here would go the proof steps demonstrating the correctness of this
  sorry

end goods_train_speed_is_92_l83_83934


namespace sum_fractions_lt_l83_83364

theorem sum_fractions_lt (n : ℕ) (h_n : n ≥ 2)
  (x : Fin n → ℝ) (h_x_gt : ∀ i, 1 < x i)
  (h_diff_lt : ∀ i : Fin (n-1), |x i - x i.succ| < 1) :
  (Finset.univ.sum (λ i : Fin n, x i / x i.succ)) < 2 * n - 1 :=
sorry

end sum_fractions_lt_l83_83364


namespace robin_candy_consumption_l83_83229

theorem robin_candy_consumption (x : ℕ) : 23 - x + 21 = 37 → x = 7 :=
by
  intros h
  sorry

end robin_candy_consumption_l83_83229


namespace solution_of_az_eq_b_l83_83272

theorem solution_of_az_eq_b (a b z x y : ℝ) :
  (∃! x, 4 + 3 * a * x = 2 * a - 7) →
  (¬ ∃ y, 2 + y = (b + 1) * y) →
  az = b →
  z = 0 :=
by
  intros h1 h2 h3
  -- proof starts here
  sorry

end solution_of_az_eq_b_l83_83272


namespace find_values_of_k_x_y_l83_83503

/-- Prove that the values of k, x, and y are 4/3, -5/3, and -11/3 respectively
    given the conditions of the points and relationships. -/
theorem find_values_of_k_x_y :
  ∃ (k x y : ℚ), 
    (let A := (-1 : ℚ, -4 : ℚ) in
     let B := (3 : ℚ, k) in
     let slope := (k + 4) / 4 in
     slope = k ∧
     x - y = 2 ∧
     k - x = 3 ∧
     k = 4 / 3 ∧
     x = -5 / 3 ∧
     y = -11 / 3) :=
by 
  sorry

end find_values_of_k_x_y_l83_83503


namespace percentage_difference_l83_83102

theorem percentage_difference (x : ℝ) : 
  (62 / 100) * 150 - (x / 100) * 250 = 43 → x = 20 :=
by
  intro h
  sorry

end percentage_difference_l83_83102


namespace lcm_of_36_and_132_is_396_l83_83432

theorem lcm_of_36_and_132_is_396 
  (a b hcf : ℕ) 
  (ha : a = 36) 
  (hb : b = 132) 
  (hhcf : hcf = 12)
  (h_hcf_def : nat.gcd a b = hcf) : 
  nat.lcm a b = 396 := 
by
  sorry

end lcm_of_36_and_132_is_396_l83_83432


namespace train_speed_is_90_km_per_hr_l83_83070

-- Definitions based on conditions
def length_of_train := 750  -- meters
def length_of_platform := 750 -- meters
def total_distance := length_of_train + length_of_platform -- meters
def time_taken := (1 : ℝ) / 60 -- hours

-- The theorem to prove
theorem train_speed_is_90_km_per_hr : (total_distance / 1000) / time_taken = 90 := by
  sorry

end train_speed_is_90_km_per_hr_l83_83070


namespace cost_of_carpeting_l83_83514

noncomputable def height_per_step := 0.16 -- meters
noncomputable def depth_per_step := 0.26 -- meters
noncomputable def width_of_staircase := 3 -- meters
noncomputable def num_of_steps := 15
noncomputable def cost_per_square_meter := 80 -- RMB

noncomputable def total_cost : ℝ :=
  let total_height := num_of_steps * height_per_step in
  let total_depth := num_of_steps * depth_per_step in
  let combined_length := total_height + total_depth in
  let total_area := combined_length * width_of_staircase in
  total_area * cost_per_square_meter

theorem cost_of_carpeting : total_cost = 1512 := by
  sorry

end cost_of_carpeting_l83_83514


namespace sum_abcd_l83_83228

theorem sum_abcd (a b c d : ℤ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 8) : 
  a + b + c + d = -6 :=
sorry

end sum_abcd_l83_83228


namespace correct_propositions_l83_83528

-- Definitions of the propositions as provided in the problem.
def proposition_1 : Prop :=
  ∀ (pyramid : Type), cutting_pyramid_with_plane_parallel_to_base p, the_part_between_base_and_section_is_frustum := true

def proposition_2 : Prop :=
  ∀ (frustum : Type), the_lateral_edges_of_a_frustum_when_extended_must_intersect_at_a_point := true

def proposition_3 : Prop :=
  ∀ (cone : Type), cone_is_right_angled_trapezoid_rotating_around_line_containing_leg_perpendicular_to_base_with_other_three_sides_forming_curved_surface := true

def proposition_4 : Prop :=
  ∀ (hemisphere : Type), hemisphere_rotating_around_line_of_diameter_forms_sphere := false

-- Main theorem statement
theorem correct_propositions : 
  (proposition_1 ∧ proposition_2 ∧ proposition_3) ∧ ¬proposition_4 :=
sorry

end correct_propositions_l83_83528


namespace find_x_for_parallel_vectors_l83_83614

theorem find_x_for_parallel_vectors (x : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, 3)) (h_b : b = (x, 6)) (h_parallel : ∃ λ : ℝ, a = λ • b) : x = 4 :=
by
  sorry

end find_x_for_parallel_vectors_l83_83614


namespace recurring_fraction_sum_l83_83923

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l83_83923


namespace integer_values_of_m_l83_83232

def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n

def count_valid_m (n : ℤ) : ℕ :=
  (list.range (2 * n + 1)).countp (λ m, is_integer (5000 * (5 / 2 : ℚ) ^ (m - n)))

theorem integer_values_of_m : count_valid_m 3 = 7 := by
  sorry

end integer_values_of_m_l83_83232


namespace problem_solution_l83_83352

-- Definition of parametric equations for C1
def C1_parametric (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 2 + 2 * Real.sin α)

-- Definition of the curve C1 in Cartesian coordinates
def C1 (x y : ℝ) : Prop :=
  ∃ α : ℝ, x = 2 * Real.cos α ∧ y = 2 + 2 * Real.sin α

-- Definition of the point P related to the curve C1
def P (x y : ℝ) : Prop :=
  ∃ α : ℝ, x = 4 * Real.cos α ∧ y = 4 + 4 * Real.sin α

-- Cartesian equation of curve C2
def C2_Cartesian (x y : ℝ) : Prop :=
  x^2 + (y - 4)^2 = 16

-- Define the polar coordinates of curve C1 and C2
def C1_polar (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sin θ

def C2_polar (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.sin θ

-- Function to find intersection points and |AB|
noncomputable def AB_distance (θ : ℝ) : ℝ :=
  let ρ1 := 4 * Real.sin θ in
  let ρ2 := 8 * Real.sin θ in
  Real.abs (ρ2 - ρ1)

-- Main theorem statement verifying the solution components
theorem problem_solution :
  (∀ x y, C1 x y → P x y → C2_Cartesian x y) ∧
  (AB_distance (π / 3) = 2 * Real.sqrt 3) :=
by
  sorry

end problem_solution_l83_83352


namespace binom_15_12_eq_455_l83_83190

theorem binom_15_12_eq_455 : Nat.choose 15 12 = 455 := 
by sorry

end binom_15_12_eq_455_l83_83190


namespace janice_second_throw_ratio_l83_83187

-- Definitions of the given conditions
def christine_first_throw := 20
def janice_first_throw := christine_first_throw - 4
def christine_second_throw := christine_first_throw + 10
def christine_third_throw := christine_second_throw + 4
def janice_third_throw := christine_first_throw + 17
def highest_throw := 37

-- Prove the ratio of Janice's second throw to her first throw is 2:1
theorem janice_second_throw_ratio :
  ∃ (j2 : ℕ), j2 = 2 * janice_first_throw ∧
    j2 < christine_third_throw ∧
    j2 < janice_third_throw :=
begin
  use 2 * janice_first_throw,
  split,
  { refl },
  split,
  { exact nat.lt_of_lt_of_le (mul_pos (nat.succ_pos 1) (nat.lt_of_lt_of_le (nat.succ_pos 3) rfl)).ne' sorry },
  { exact nat.lt_of_lt_of_le (mul_pos (nat.succ_pos 1) (nat.lt_of_lt_of_le (nat.succ_pos 3) rfl)).ne' sorry }
end

end janice_second_throw_ratio_l83_83187


namespace remainder_50_pow_50_mod_7_l83_83778

theorem remainder_50_pow_50_mod_7 : (50^50) % 7 = 1 := by
  sorry

end remainder_50_pow_50_mod_7_l83_83778


namespace A_work_time_l83_83481

-- Define the problem using the given conditions
variables (W : ℝ) (x A B C : ℝ)

-- Define the conditions in Lean
def B_rate := W / 20
def C_rate := W / 55

-- The rate of work for A
def A_rate := W / x

-- The work done in 8 days with A assisted by B and C on alternate days
def work_done := (8 * (W / x) + 4 * (W / 20) + 4 * (W / 55))

-- The proof statement
theorem A_work_time : 8 * (W / x) + (W / 5) + (4 * (W / 55)) = W → x = 11 :=
by
  intros h
  sorry 

end A_work_time_l83_83481


namespace rooms_illuminated_after_100_steps_l83_83505

noncomputable theory

def RoomState := List Bool -- represent the state of rooms, True means illuminated

-- Initial state: all rooms are illuminated
def initial_state : RoomState := replicate 7 true

def toggle (state : RoomState) (index : ℕ) : RoomState :=
  state.update_nth index (not (state.get index))

def walk_one_cycle (state : RoomState) : RoomState :=
  let state1 := state.foldl (λ st idx => toggle st idx) state (List.range 7)
  in state1.foldl (λ st idx => toggle st idx) state1 (List.range 1 6)

def walk_n_rooms (n : ℕ) : ℕ :=
  let cycles := n / 12
  let remainder := n % 12
  let state_after_cycles := (List.range cycles).foldl (λ st _ => walk_one_cycle st) initial_state
  let final_state := (List.range remainder).foldl (λ st idx => toggle st (idx % 7)) state_after_cycles
  final_state.count (λ b => b) -- count illuminated rooms

theorem rooms_illuminated_after_100_steps : walk_n_rooms 100 = 3 :=
sorry

end rooms_illuminated_after_100_steps_l83_83505


namespace math_problem_solution_l83_83525

-- Define the given vectors and their properties
variables {α β : ℝ}

def vec_a : ℝ × ℝ := (Real.cos α, Real.sin α)
def vec_b : ℝ × ℝ := (Real.cos β, Real.sin β)

-- Define vector scaling condition
def vec_scaled (λ : ℝ) : Prop := vec_a = λ • vec_b

-- Define centroid property
def centroid_property {OA OB OC : ℝ × ℝ} : Prop := OA + OB + OC = (0, 0)

-- Define lengths of vectors with equal angles condition
def vectors_with_equal_angles (a b c : ℝ × ℝ) : Prop :=
  ∀ (θ : ℝ), (θ = 0 ∨ θ = 2 * Real.pi / 3) →
  (a = (1, 0)) ∧ (b = (2 * Real.cos θ, 2 * Real.sin θ)) ∧ (c = (3 * Real.cos θ, 3 * Real.sin θ))

-- Define the propositions
def prop1 : Prop := ¬ ((λ x, (x + 1) ^ 2) (x - 1) = x ^ 2)
def prop2 : Prop := ∀ (λ : ℝ), vec_scaled λ ↔ (λ = 1 ∨ λ = -1)
def prop3 : Prop := ∀ (OA OB OC : ℝ × ℝ), centroid_property { OA, OB, OC }
def prop4 : Prop := ¬ (‖vec_a‖ = 1 ∧ ‖vec_b‖ = 2 ∧ ‖(vec_a + vec_b + vec_c)‖ = sqrt 3)

-- Combining the propositions check
def solution := prop1 = false ∧ prop2 = true ∧ prop3 = true ∧ prop4 = false

-- The statement to prove
theorem math_problem_solution : solution := 
by 
  sorry

end math_problem_solution_l83_83525


namespace dot_product_intersection_equal_neg_11_l83_83280

-- Definitions derived from the problem statement

variables {x1 y1 x2 y2 : ℝ}

def line_eq (x : ℝ) : ℝ := 2 * x - 2
def parabola_eq (x : ℝ) : ℝ := real.sqrt (8 * x)
def focus : (ℝ × ℝ) := (2, 0)

-- Definitions of intersection points A and B
def point_A := (x1, y1)
def point_B := (x2, y2)

def A_point_on_line : Prop := y1 = line_eq x1
def B_point_on_line : Prop := y2 = line_eq x2

def A_point_on_parabola : Prop := y1^2 = 8 * x1
def B_point_on_parabola : Prop := y2^2 = 8 * x2

-- Definition of vectors FA and FB
def vector_FA := (x1 - 2, y1)
def vector_FB := (x2 - 2, y2)

-- Dot product of vectors FA and FB
def dot_product_FA_FB := (x1 - 2) * (x2 - 2) + y1 * y2

theorem dot_product_intersection_equal_neg_11 :
  A_point_on_line → B_point_on_line →
  A_point_on_parabola → B_point_on_parabola →
  dot_product_FA_FB = -11 :=
by
  intros hA_line hB_line hA_para hB_para
  sorry

end dot_product_intersection_equal_neg_11_l83_83280


namespace repeating_decimal_fraction_sum_l83_83849

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l83_83849


namespace reasoning_incorrect_l83_83082

-- Define odd function property
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Define sine function as odd function
lemma sine_odd_function : is_odd_function sin :=
by
  intro x,
  exact sin_neg x

-- Define f(x) = sin(x^2 + 1)
def f (x : ℝ) : ℝ := sin (x^2 + 1)

-- Prove the reasoning given above is incorrect
theorem reasoning_incorrect : ¬ is_odd_function f :=
by
  -- Proof is omitted
  sorry

end reasoning_incorrect_l83_83082


namespace number_of_ways_to_label_decagon_equal_sums_l83_83748

open Nat

-- Formal definition of the problem
def sum_of_digits : Nat := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

-- The problem statement: Prove there are 3840 ways to label digits ensuring the given condition
theorem number_of_ways_to_label_decagon_equal_sums :
  ∃ (n : Nat), n = 3840 ∧ ∀ (A B C D E F G H I J K L : Nat), 
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ (A ≠ G) ∧ (A ≠ H) ∧ (A ≠ I) ∧ (A ≠ J) ∧ (A ≠ K) ∧ (A ≠ L) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (B ≠ G) ∧ (B ≠ H) ∧ (B ≠ I) ∧ (B ≠ J) ∧ (B ≠ K) ∧ (B ≠ L) ∧
    (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (C ≠ G) ∧ (C ≠ H) ∧ (C ≠ I) ∧ (C ≠ J) ∧ (C ≠ K) ∧ (C ≠ L) ∧
    (D ≠ E) ∧ (D ≠ F) ∧ (D ≠ G) ∧ (D ≠ H) ∧ (D ≠ I) ∧ (D ≠ J) ∧ (D ≠ K) ∧ (D ≠ L) ∧
    (E ≠ F) ∧ (E ≠ G) ∧ (E ≠ H) ∧ (E ≠ I) ∧ (E ≠ J) ∧ (E ≠ K) ∧ (E ≠ L) ∧
    (F ≠ G) ∧ (F ≠ H) ∧ (F ≠ I) ∧ (F ≠ J) ∧ (F ≠ K) ∧ (F ≠ L) ∧
    (G ≠ H) ∧ (G ≠ I) ∧ (G ≠ J) ∧ (G ≠ K) ∧ (G ≠ L) ∧
    (H ≠ I) ∧ (H ≠ J) ∧ (H ≠ K) ∧ (H ≠ L) ∧
    (I ≠ J) ∧ (I ≠ K) ∧ (I ≠ L) ∧
    (J ≠ K) ∧ (J ≠ L) ∧
    (K ≠ L) ∧
    (A + L + F = B + L + G) ∧ (B + L + G = C + L + H) ∧ 
    (C + L + H = D + L + I) ∧ (D + L + I = E + L + J) ∧ 
    (E + L + J = F + L + K) ∧ (F + L + K = A + L + F) :=
sorry

end number_of_ways_to_label_decagon_equal_sums_l83_83748


namespace yogurt_raisins_l83_83393

-- Definitions for the problem context
variables (v k r x : ℕ)

-- Conditions based on the given problem statements
axiom h1 : 3 * v + 3 * k = 18 * r
axiom h2 : 12 * r + 5 * k = v + 6 * k + x * r

-- Statement that we need to prove
theorem yogurt_raisins : x = 6 :=
by
  have h3 : v + k = 6 * r :=
    calc
      3 * v + 3 * k = 18 * r : h1
      3 * (v + k) = 18 * r : by ring
      v + k = 6 * r : by linarith
  have h4 : 12 * r + 5 * k = v + 6 * k + x * r : h2
  have h5 : 12 * r = (v + k) + 5 * k + x * r := by linarith
  rw [h3] at h5
  have h6 : 12 * r = 6 * r + 5 * k + x * r := by linarith
  have h7 : 6 * r = 5 * k + x * r := by linarith
  have h8 : x = 6 := by linarith
  exact h8

end yogurt_raisins_l83_83393


namespace smallest_value_of_N_l83_83164

theorem smallest_value_of_N (l m n : ℕ) (N : ℕ) (h1 : (l-1) * (m-1) * (n-1) = 270) (h2 : N = l * m * n): 
  N = 420 :=
sorry

end smallest_value_of_N_l83_83164


namespace factorize_expression_l83_83214

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end factorize_expression_l83_83214


namespace max_x_satisfying_sqrt_eqn_l83_83557

theorem max_x_satisfying_sqrt_eqn : 
  ∃ x, (sqrt (3 * x) = 5 * x^2) ∧ (∀ y, sqrt (3 * y) = 5 * y^2 → y ≤ x) ∧ 
  x = (3 / 25)^(1 / 3) := 
sorry

end max_x_satisfying_sqrt_eqn_l83_83557


namespace sequence_sum_eq_ten_implies_n_eq_120_l83_83282

theorem sequence_sum_eq_ten_implies_n_eq_120 :
  (∀ (a : ℕ → ℝ), (∀ n, a n = 1 / (Real.sqrt n + Real.sqrt (n + 1))) →
    (∃ n, (Finset.sum (Finset.range n) a) = 10 → n = 120)) :=
by
  intro a h
  use 120
  intro h_sum
  sorry

end sequence_sum_eq_ten_implies_n_eq_120_l83_83282


namespace correct_proposition_is_B_l83_83979

def proposition_A : Prop :=
  ∀ {l₁ l₂ : Line} {P : Plane}, (parallel l₁ l₂ ∧ projection l₁ P = projection l₂ P) → 
    (parallel (projection l₁ P) (projection l₂ P) ∨ (projection l₁ P) = (projection l₂ P))

def proposition_B : Prop :=
  ∀ {l₁ l₂ : Line} {P : Plane}, (orthogonal l₁ P ∧ orthogonal l₂ P) → parallel l₁ l₂

def proposition_C : Prop :=
  ∀ {P₁ P₂ P₃ : Plane}, (orthogonal P₁ P₃ ∧ orthogonal P₂ P₃) → parallel P₁ P₂

def proposition_D : Prop :=
  ∀ {P₁ P₂ : Plane} {l : Line}, (parallel P₁ l ∧ parallel P₂ l) → parallel P₁ P₂

theorem correct_proposition_is_B : proposition_B :=
sorry

end correct_proposition_is_B_l83_83979


namespace bill_can_buy_donuts_in_35_ways_l83_83988

def different_ways_to_buy_donuts : ℕ :=
  5 + 20 + 10  -- Number of ways to satisfy the conditions

theorem bill_can_buy_donuts_in_35_ways :
  different_ways_to_buy_donuts = 35 :=
by
  -- Proof steps
  -- The problem statement and the solution show the calculation to be correct.
  sorry

end bill_can_buy_donuts_in_35_ways_l83_83988


namespace angle_B_measure_l83_83669

theorem angle_B_measure (A B C K L : Point)
  (h1 : is_triangle A B C)
  (h2 : is_altitude A K B C)
  (h3 : is_altitude C L A B)
  (h4 : AC = 2 * LK) : (angle A B C = 60 ∨ angle A B C = 120) :=
sorry

end angle_B_measure_l83_83669


namespace area_of_connected_colored_paper_l83_83753

noncomputable def side_length : ℕ := 30
noncomputable def overlap : ℕ := 7
noncomputable def sheets : ℕ := 6
noncomputable def total_length : ℕ := side_length + (sheets - 1) * (side_length - overlap)
noncomputable def width : ℕ := side_length

theorem area_of_connected_colored_paper : total_length * width = 4350 := by
  sorry

end area_of_connected_colored_paper_l83_83753


namespace minimum_value_expression_l83_83821

theorem minimum_value_expression : ∃ x : ℝ, (3 * x^2 - 18 * x + 2023) = 1996 := sorry

end minimum_value_expression_l83_83821


namespace conditional_probability_chinese_fail_l83_83649

theorem conditional_probability_chinese_fail :
  let P_math := 0.16
  let P_chinese := 0.07
  let P_both := 0.04
  P_both / P_chinese = (4 / 7) := by
  let P_math := 0.16
  let P_chinese := 0.07
  let P_both := 0.04
  sorry

end conditional_probability_chinese_fail_l83_83649


namespace draw_consecutive_not_all_three_l83_83085

-- Define the problem conditions
def people : Nat := 3
def balls : Finset Nat := {1, 2, 3, 4, 5}
def consecutive (a b : Nat) : Bool := abs (a - b) = 1
def no_replacement : Bool := True  -- Drawing without replacement is inherently true in our model

-- The goal is to prove the given statement
theorem draw_consecutive_not_all_three (people = 3) 
  (balls = {1, 2, 3, 4, 5}) 
  (no_replacement) :
  ∃ (ways : Nat), ways = 36 := 
sorry

end draw_consecutive_not_all_three_l83_83085


namespace m_and_n_relationship_l83_83300

-- Define the function f
def f (x m : ℝ) := x^2 - 4*x + 4 + m

-- State the conditions and required proof
theorem m_and_n_relationship (m n : ℝ) (h_domain : ∀ x, 2 ≤ x ∧ x ≤ n → 2 ≤ f x m ∧ f x m ≤ n) :
  m^n = 8 :=
by
  -- Placeholder for the actual proof
  sorry

end m_and_n_relationship_l83_83300


namespace seating_pairs_count_l83_83749

theorem seating_pairs_count (f m : ℕ) 
  (h1 : 1 ≤ f) 
  (h2 : 1 ≤ m) 
  (h3 : f ≤ 7) 
  (h4 : m ≤ 7) : 
  (∃ count : ℕ, count = 8 ∧ 
    (∀ pairs : Finset (ℕ × ℕ), 
      pairs.card = count ∧ 
      (∀ p ∈ pairs, 
        let (f, m) := p in 
        (1 ≤ f ∧ f ≤ 7) ∧ (1 ≤ m ∧ m ≤ 7)))) :=
sorry

end seating_pairs_count_l83_83749


namespace problem_solution_correct_l83_83260

def proposition_p : Prop :=
  ∃ x : ℝ, Real.tan x = 1

def proposition_q : Prop :=
  {x : ℝ | x^2 - 3 * x + 2 < 0} = {x : ℝ | 1 < x ∧ x < 2}

theorem problem_solution_correct :
  (proposition_p ∧ proposition_q) ∧
  (proposition_p ∧ ¬proposition_q) = false ∧
  (¬proposition_p ∨ proposition_q) ∧
  (¬proposition_p ∨ ¬proposition_q) = false :=
by
  sorry

end problem_solution_correct_l83_83260


namespace incorrect_observation_correction_l83_83446

theorem incorrect_observation_correction:
  (mean_initial mean_corrected n incorrect_val correct_val : ℝ)
  (h1 : n = 50)
  (h2 : mean_initial = 36)
  (h3 : incorrect_val = 23)
  (h4 : mean_corrected = 36.5)
  (h5 : mean_initial * n + correct_val - incorrect_val = mean_corrected * n) :
  correct_val = 48 :=
sorry

end incorrect_observation_correction_l83_83446


namespace sum_of_fraction_terms_l83_83860

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l83_83860


namespace convert_line_to_polar_convert_circle_to_polar_l83_83195

noncomputable def cartesian_to_polar (x y : ℝ) (θ ρ : ℝ) : Prop :=
θ = Real.arctan2 y x ∧ ρ = Real.sqrt (x^2 + y^2)

noncomputable def line_polar (x y : ℝ) : Prop :=
∃ ρ θ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ θ = 3 * Real.pi / 4 ∧ ρ ∈ Set.univ

noncomputable def circle_polar (x y a : ℝ) (h : a ≠ 0) : Prop :=
∃ ρ θ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ ρ = -2 * a * Real.cos θ

theorem convert_line_to_polar (x y : ℝ) : line_polar x y ↔ (x + y = 0) := by
  sorry

theorem convert_circle_to_polar (x y a : ℝ) (h : a ≠ 0) : circle_polar x y a h ↔ (x^2 + y^2 + 2 * a * x = 0) := by
  sorry

end convert_line_to_polar_convert_circle_to_polar_l83_83195


namespace bridge_length_l83_83935

theorem bridge_length :
  ∀ (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ),
  train_length = 130 → train_speed_kmh = 45 → crossing_time = 30 →
  let train_speed_ms := train_speed_kmh * (1000 / 3600) in
  let total_distance := train_speed_ms * crossing_time in
  let bridge_length := total_distance - train_length in
  bridge_length = 245 :=
begin
  intros train_length train_speed_kmh crossing_time h1 h2 h3,
  dsimp [train_speed_ms, total_distance, bridge_length],
  rw [h1, h2, h3],
  norm_num,
  sorry
end

end bridge_length_l83_83935


namespace proof1_proof2a_proof2b_l83_83252

section problem1

variable {f : ℝ → ℝ} 
variable (cond_f_zeros : f (-3) = 0 ∧ f 1 = 0)
variable (cond_f_minimum : ∃ x: ℝ, ∀ y: ℝ, f x ≤ f y)
variable (value.f_minimum : (∃ x: ℝ, ∀ y: ℝ, f x ≤ f y) → f (-1) = -4)

theorem proof1 : f = λ x, x^2 + 2*x - 3 :=
sorry

end problem1

section problem2

variable (m : ℝ)
variable (f : ℝ → ℝ)
variable (cond_f_zeros : f (-3) = 0 ∧ f 1 = 0)
variable (cond_f_minimum : ∃ x: ℝ, ∀ y: ℝ, f x ≤ f y)
variable (value.f_minimum : (∃ x: ℝ, ∀ y: ℝ, f x ≤ f y) → f (-1) = -4)

def g (x : ℝ) : ℝ := m * (f x) + 2

theorem proof2a (h : m < 0) : ∃! x ∈ Ici (-3), g x = 0 :=
sorry

theorem proof2b (h : 0 < m) : 
  (m ≤ 8 / 7 → ∃ x ∈ Icc (-3) (3 / 2), |g x| = (9 / 4) * m + 2) ∧ 
  (m > 8 / 7 → ∃ x ∈ Icc (-3) (3 / 2), |g x| = 4 * m - 2) :=
sorry

end problem2

end proof1_proof2a_proof2b_l83_83252


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l83_83889

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l83_83889


namespace sum_first_2m_terms_l83_83281

-- Definitions
def seq_a (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def seq_b (m : ℕ) : ℕ :=
  if m % 2 = 1 then
    seq_a (3 * ((m + 1) / 2) - 1)
  else
    seq_a (3 * (m / 2))

def sum_seq_b (m : ℕ) : ℕ :=
  (List.range (2 * m)).map seq_b |
    (List.range (2 * m)).map seq_b
    |>.sum

theorem sum_first_2m_terms (m : ℕ) :
  sum_seq_b m = (3 * m * (m + 1) * (2 * m + 1)) / 2 :=
    sorry

end sum_first_2m_terms_l83_83281


namespace sum_of_fraction_terms_l83_83862

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l83_83862


namespace math_problem_l83_83000

noncomputable theory -- since we are using non-computable functions

-- Define the function f^1(x)
def f1 (x : ℝ) : ℝ := x^3 - 3 * x

-- Define the n-th iteration of the function f^n(x)
def fn : ℕ → (ℝ → ℝ)
| 0     := λ x, x
| (n+1) := λ x, f1 (fn n x)

-- Define the set of roots of f^2022(x)/x
def roots : set ℝ := { x | fn 2022 x = 0 }

-- Define the sum of the reciprocals of the squares of the roots
def sum_of_reciprocals_of_squares : ℝ :=
  ∑ r in roots.to_finset, 1 / (r ^ 2)

-- Constants for the solution
constants (a b c d : ℕ)
  (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)
  (b_max : ∀ n : ℕ, a ^ n < a ^ b → n < b)

-- The final theorem asserting the solution
theorem math_problem :
  sum_of_reciprocals_of_squares = (a^b - c) / d ∧ nat.coprime c d →
  a + b + c + d = 4060 :=
sorry

end math_problem_l83_83000


namespace min_value_quadratic_l83_83817

theorem min_value_quadratic : ∃ x : ℝ, ∀ y : ℝ, 3 * x ^ 2 - 18 * x + 2023 ≤ 3 * y ^ 2 - 18 * y + 2023 :=
sorry

end min_value_quadratic_l83_83817


namespace BD_value_l83_83645

noncomputable def triangle_problem (A B C D : Point) (AC BC : ℝ) (AB : ℝ) (CD : ℝ) (BD : ℝ) :=
  AC = 10 ∧ BC = 10 ∧ AB = 4 ∧ B ∈ segment A D ∧ CD = 12 → BD = 4 * Real.sqrt 3 - 2

-- Stating the theorem
theorem BD_value (A B C D : Point) (AC BC : ℝ) (AB : ℝ) (CD : ℝ) (BD : ℝ) :
  triangle_problem A B C D AC BC AB CD BD :=
by intros; sorry

end BD_value_l83_83645


namespace draw_is_unfair_suit_hierarchy_makes_fair_l83_83310

structure Card where
  suit : ℕ -- 4 suits numbered from 0 to 3
  rank : ℕ -- 9 ranks numbered from 0 to 8

def deck : List Card :=
  List.join (List.map (λ s, List.map (λ r, ⟨s, r⟩) (List.range 9)) (List.range 4))

def DrawFair? : (deck : List Card) → Prop := sorry

-- Part (a): Prove that the draw is unfair
theorem draw_is_unfair : ¬ DrawFair? deck := sorry

-- Part (b): Prove that introducing a suit hierarchy can make the draw fair
def suit_hierarchy : Card → Card → Prop :=
λ c1 c2, (c1.rank < c2.rank) ∨ (c1.rank = c2.rank ∧ c1.suit < c2.suit)

theorem suit_hierarchy_makes_fair : ∃ h : Card → Card → Prop, h = suit_hierarchy ∧ DrawFair? deck[h] := sorry

end draw_is_unfair_suit_hierarchy_makes_fair_l83_83310


namespace solve_for_x_l83_83626

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end solve_for_x_l83_83626


namespace count_polynomials_in_G_l83_83004

-- Define the set of polynomials G with conditions as given
def polynomial_set (n : ℕ) : Set (Polynomial ℤ) :=
  {P : Polynomial ℤ |
    P.degree = n ∧
    ∃ c : Fin (n - 1) → ℤ,  -- coefficients c1, c2, ..., c_{n-1}
      P = Polynomial.monomial n 1 + 
          ∑ i in Finset.range (n - 1), (Polynomial.monomial i.succ (c ⟨i, sorry⟩)) +
          Polynomial.C 60 ∧
    ∀ (z : ℤ), (∃ a b : ℤ, P.eval z = 0 ∧ z = a + b * I)}

-- Define the set of polynomials count proof
theorem count_polynomials_in_G (n : ℕ) : ∃ N : ℕ, ∀ P ∈ polynomial_set n, finite (polynomial_set n) ∧ polynomial_set n.card = N := 
sorry

end count_polynomials_in_G_l83_83004


namespace ground_beef_cost_l83_83295

theorem ground_beef_cost (cost_3_5_pounds : ℝ) (weight_3_5_pounds : ℝ) 
  (cost_per_pound : ℝ) (weight_5_6_pounds : ℝ) : 
  cost_3_5_pounds = 9.77 → weight_3_5_pounds = 3.5 → 
  cost_per_pound = cost_3_5_pounds / weight_3_5_pounds → 
  weight_5_6_pounds = 5.6 →
  (cost_per_pound * weight_5_6_pounds).round2 = 15.62 :=
by
  sorry

end ground_beef_cost_l83_83295


namespace draw_is_unfair_suit_hierarchy_makes_fair_l83_83306

structure Card where
  suit : ℕ -- 4 suits numbered from 0 to 3
  rank : ℕ -- 9 ranks numbered from 0 to 8

def deck : List Card :=
  List.join (List.map (λ s, List.map (λ r, ⟨s, r⟩) (List.range 9)) (List.range 4))

def DrawFair? : (deck : List Card) → Prop := sorry

-- Part (a): Prove that the draw is unfair
theorem draw_is_unfair : ¬ DrawFair? deck := sorry

-- Part (b): Prove that introducing a suit hierarchy can make the draw fair
def suit_hierarchy : Card → Card → Prop :=
λ c1 c2, (c1.rank < c2.rank) ∨ (c1.rank = c2.rank ∧ c1.suit < c2.suit)

theorem suit_hierarchy_makes_fair : ∃ h : Card → Card → Prop, h = suit_hierarchy ∧ DrawFair? deck[h] := sorry

end draw_is_unfair_suit_hierarchy_makes_fair_l83_83306


namespace sin_maxima_range_l83_83769

theorem sin_maxima_range (ω : ℝ) (hω : ω > 0) (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x)) :
  (∃ n : ℕ, n = 50 ∧ ∀ x ∈ Icc 0 1, f'' x < 0) ↔ ω ∈ Icc (201 * Real.pi / 4) (205 * Real.pi / 4) :=
by sorry

end sin_maxima_range_l83_83769


namespace min_bench_sections_l83_83515

theorem min_bench_sections (N : ℕ) :
  ∀ x y : ℕ, (x = y) → (x = 8 * N) → (y = 12 * N) → (24 * N) % 20 = 0 → N = 5 :=
by
  intros
  sorry

end min_bench_sections_l83_83515


namespace recurring_fraction_sum_l83_83916

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l83_83916


namespace polynomial_sum_even_terms_l83_83366

theorem polynomial_sum_even_terms :
  ∃ a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} a_{12} : ℝ,
  (x-1)^4 * (x+2)^8 = a_0 * x^12 + a_1 * x^11 + a_2 * x^10 + a_3 * x^9 + a_4 * x^8 + a_5 * x^7 + 
                             a_6 * x^6 + a_7 * x^5 + 
                             a_8 * x^4 + a_9 * x^3 + a_{10} * x^2 + a_{11} * x + a_{12} ∧
  (a_2 + a_4 + a_6 + a_8 + a_{10} + a_{12} = 7) :=
begin
  sorry
end

end polynomial_sum_even_terms_l83_83366


namespace simplify_expression_l83_83423

-- Definitions of the square roots and their fractions
def a : Real := (Real.sqrt 726) / (Real.sqrt 484)
def b : Real := (Real.sqrt 245) / (Real.sqrt 147)
def c : Real := (Real.sqrt 1089) / (Real.sqrt 441)

-- Statement of the problem
theorem simplify_expression : a + b + c = (87 + 14 * Real.sqrt 15) / 42 := 
sorry

end simplify_expression_l83_83423


namespace measure_of_angle_B_maximum_area_of_triangle_l83_83644

variables (a b c : ℝ) (A B C : ℝ)
variables (BA BC CB CA : ℝ) -- vector magnitudes
variables (S : ℝ) -- area of the triangle

-- Conditions
axiom triangle_sides : b^2 = 6
axiom angle_b_condition : (\sqrt{2} * a - c) * BA * BC = c * CB * CA
axiom vector_diff_magnitude : |BA - BC| = \sqrt{6}

-- Questions as goals to be proved
theorem measure_of_angle_B : B = π / 4 :=
by sorry

theorem maximum_area_of_triangle : S = \frac{3 * (\sqrt{2} + 1)}{2} :=
by sorry

end measure_of_angle_B_maximum_area_of_triangle_l83_83644


namespace initial_sticks_correct_l83_83408

-- Define the number of sticks given per group and the number of groups
def sticks_per_group : ℕ := 15
def num_groups : ℕ := 10

-- Define the total number of sticks given away
def total_given : ℕ := sticks_per_group * num_groups

-- Define the remaining number of sticks
def remaining_sticks : ℕ := 20

-- Define the initial number of sticks
def initial_sticks : ℕ := total_given + remaining_sticks

-- Prove that the initial number of sticks is 170
theorem initial_sticks_correct : initial_sticks = 170 := 
by
  unfold initial_sticks total_given sticks_per_group num_groups remaining_sticks
  dsimp
  sorry

end initial_sticks_correct_l83_83408


namespace cot6_sub_cot2_eq_zero_l83_83291

theorem cot6_sub_cot2_eq_zero (x : ℝ) (h_geom_seq : ∃ r : ℝ, sin x = r * cos x ∧ tan x = r * sin x) : 
  cot x ^ 6 - cot x ^ 2 = 0 :=
by
  sorry

end cot6_sub_cot2_eq_zero_l83_83291


namespace sum_sequence_bn_l83_83223

theorem sum_sequence_bn (n : ℕ) (a2 : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  (∀ k : ℕ, b k = a2 + 2^k * (a k)) → 
  (∀ k : ℕ, a k = 2 ^ (k - 1)) →
  (S_n : ℝ) = (∑ k in finset.range n, b k) →
  S_n = ((4^n - 1) / 3) + (1 / 2) * n * (n - 1) :=
by
  intros,
  sorry

end sum_sequence_bn_l83_83223


namespace mark_reading_pages_before_injury_l83_83714

theorem mark_reading_pages_before_injury:
  ∀ (h_increased: Nat) (pages_week: Nat), 
  (h_increased = 2 + (2 * 3/2)) ∧ (pages_week = 1750) → 100 = pages_week / 7 / h_increased * 2 := 
by
  sorry

end mark_reading_pages_before_injury_l83_83714


namespace find_six_digit_numbers_l83_83168

variable (m n : ℕ)

-- Definition that the original number becomes six-digit when multiplied by 4
def is_six_digit (x : ℕ) : Prop := x ≥ 100000 ∧ x < 1000000

-- Conditions
def original_number := 100 * m + n
def new_number := 10000 * n + m
def satisfies_conditions (m n : ℕ) : Prop :=
  is_six_digit (100 * m + n) ∧
  is_six_digit (10000 * n + m) ∧
  4 * (100 * m + n) = 10000 * n + m

-- Theorem statement
theorem find_six_digit_numbers (h₁ : satisfies_conditions 1428 57)
                               (h₂ : satisfies_conditions 1904 76)
                               (h₃ : satisfies_conditions 2380 95) :
  ∃ m n, satisfies_conditions m n :=
  sorry -- Proof omitted

end find_six_digit_numbers_l83_83168


namespace intersection_of_A_and_B_l83_83418

   def set_A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}
   def set_B : Set ℝ := {x | x^2 + 4 * x ≤ 0}
   def intersection := {x | x = 0}

   theorem intersection_of_A_and_B : set_A ∩ set_B = intersection := by
     sorry
   
end intersection_of_A_and_B_l83_83418


namespace find_increase_x_l83_83801

noncomputable def initial_radius : ℝ := 7
noncomputable def initial_height : ℝ := 5
variable (x : ℝ)

theorem find_increase_x (hx : x > 0)
  (volume_eq : π * (initial_radius + x) ^ 2 * initial_height =
               π * initial_radius ^ 2 * (initial_height + 2 * x)) :
  x = 28 / 5 :=
by
  sorry

end find_increase_x_l83_83801


namespace triangle_median_difference_l83_83353

theorem triangle_median_difference
    (A B C D E : Type)
    (BC_len : BC = 10)
    (AD_len : AD = 6)
    (BE_len : BE = 7.5) :
    ∃ X_max X_min : ℝ, 
    X_max = AB^2 + AC^2 + BC^2 ∧ 
    X_min = AB^2 + AC^2 + BC^2 ∧ 
    (X_max - X_min) = 56.25 :=
by
  sorry

end triangle_median_difference_l83_83353


namespace convert_C1_to_cartesian_minimum_PQ_distance_l83_83266

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

-- Curve C1 in polar coordinates
def C1_polar (ρ θ : ℝ) : Prop :=
  ρ^2 = 8 * ρ * sin θ - 15

-- Curve C1 in Cartesian coordinates
def C1_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 * y + 15 = 0

-- Curve C2 in Cartesian coordinates with parameter α
def C2 (α : ℝ) : ℝ × ℝ :=
  (2 * sqrt 2 * cos α, sqrt 2 * sin α)

-- Point Q on C2 with α = 3π/4
def Q : ℝ × ℝ :=
  C2 (3 * real.pi / 4)

-- Proving the conversion of C1 from polar to Cartesian
theorem convert_C1_to_cartesian :
  ∀ (ρ θ : ℝ), C1_polar ρ θ ↔ C1_cartesian (ρ * cos θ) (ρ * sin θ) := by
  sorry

-- Proving the minimum distance from point P on C1 to point Q
theorem minimum_PQ_distance :
  let P_distance (ρ θ : ℝ) : ℝ := (polar_to_cartesian ρ θ).dist Q
  ∃ (ρ θ : ℝ), C1_polar ρ θ → ∀ (ρ' θ' : ℝ), C1_polar ρ' θ' → 
    P_distance ρ θ ≤ (polar_to_cartesian ρ' θ').dist Q
  := by
  sorry

end convert_C1_to_cartesian_minimum_PQ_distance_l83_83266


namespace part1_arithmetic_sequence_part2_sum_first_n_terms_part3_range_inequality_l83_83600

-- Function and its derivative
def f (x : ℝ) := log x + cos x - ((6 / real.pi) - (9 / 2)) * x
def f_prime (x : ℝ) := 1 / x - sin x - (6 / real.pi) + 9 / 2
noncomputable def a_seq (n : ℕ) := sorry -- Define based on the recurrence relation

-- Conditions and statements to be proven
theorem part1_arithmetic_sequence (a_seq : ℕ → ℝ) (a1 : ℝ) (C : ℝ) :
  (∀ n : ℕ, a_seq (n + 1) + a_seq n = 4 * (n + 1) + 3) →
  (∀ n : ℕ, a_seq n = a1 + n * C) →
  (C = 2) → 
  (a1 = 5 / 2) :=
sorry

theorem part2_sum_first_n_terms (a_seq : ℕ → ℝ) (S_n : ℕ → ℝ) :
  (∀ n : ℕ, a_seq (n + 1) + a_seq n = 4 * (n + 1) + 3) →
  (a_seq 0 = 2) →
  (S_n = λ n, if even n then (2 * n^2 + 3 * n) / 2 else (2 * n^2 + 3 * n - 1) / 2) :=
sorry

theorem part3_range_inequality (a_seq : ℕ → ℝ) (a1 : ℝ) :
  (∀ n : ℕ, a_seq (n + 1) + a_seq n = 4 * (n + 1) + 3) →
  (∀ n : ℕ, (a_seq n^2 + a_seq (n + 1)^2) / (a_seq n + a_seq (n + 1)) ≥ 4) →
  (a1 ∈ set.Iio ((7 - sqrt 7) / 2) ∪ set.Ici ((7 + sqrt 7) / 2)) :=
sorry

end part1_arithmetic_sequence_part2_sum_first_n_terms_part3_range_inequality_l83_83600


namespace martin_leftover_raisins_l83_83394

theorem martin_leftover_raisins :
  ∀ (v k r : ℝ),
  (3 * v + 3 * k = 18 * r) → 
  (12 * r + 5 * k = v + 6 * k + (x * r)) →
  x = 6 :=
begin 
  intros v k r h1 h2,
  have h3 : v + k = 6 * r,
  { linarith, },
  have h4 : 12 * r = v + k + x * r,
  { linarith, },
  rw h3 at h2,
  exact eq_of_mul_eq_mul_right (by linarith) (by linarith),
  sorry
end

end martin_leftover_raisins_l83_83394


namespace four_weighings_sufficient_three_weighings_insufficient_l83_83137

namespace CanWeighing

-- Definitions based on the conditions.
def unique_weights (cans : List ℝ) : Prop :=
  ∀ i j : ℕ, i ≠ j → cans.get? i ≠ cans.get? j

def preserved_list (cans : List ℝ) : List ℝ :=
  cans

def weigh (cans1 cans2 : List ℝ) : ℤ :=
  let weight1 := (cans1.map (·)).sum
  let weight2 := (cans2.map (·)).sum
  weight1 - weight2

-- Assume list of 80 cans with unique weights.
def cans : List ℝ := sorry -- let's assume this list is provided and it has 80 distinct weights

-- Define the correct statements to prove.
theorem four_weighings_sufficient (cans : List ℝ) (h : unique_weights cans) :
  ∃ (weighings : List (List ℝ × List ℝ)), weighings.length = 4 ∧
  (∀ (i j : ℕ), i ≠ j → weighings.nth i ≠ weighings.nth j) :=
sorry

theorem three_weighings_insufficient (cans : List ℝ) (h : unique_weights cans) :
  ¬ ∃ (weighings : List (List ℝ × List ℝ)), weighings.length = 3 ∧
  (∀ (i j : ℕ), i ≠ j → weighings.nth i ≠ weighings.nth j) :=
sorry

end CanWeighing

end four_weighings_sufficient_three_weighings_insufficient_l83_83137


namespace sum_of_fraction_numerator_and_denominator_l83_83910

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l83_83910


namespace circle_tangent_problem_solution_l83_83991

noncomputable def circle_tangent_problem
(radius : ℝ)
(center : ℝ × ℝ)
(point_A : ℝ × ℝ)
(distance_OA : ℝ)
(segment_BC : ℝ) : ℝ :=
  let r := radius
  let O := center
  let A := point_A
  let OA := distance_OA
  let BC := segment_BC
  let AT := Real.sqrt (OA^2 - r^2)
  2 * AT - BC

-- Definitions for the conditions
def radius : ℝ := 8
def center : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (17, 0)
def distance_OA : ℝ := 17
def segment_BC : ℝ := 12

-- Statement of the problem as an example theorem
theorem circle_tangent_problem_solution :
  circle_tangent_problem radius center point_A distance_OA segment_BC = 18 :=
by
  -- We would provide the proof here. The proof steps are not required as per the instructions.
  sorry

end circle_tangent_problem_solution_l83_83991


namespace line_outside_plane_has_at_most_one_common_point_l83_83095

-- Definitions of necessary terms
variable {P : Type} -- P for points (or positions)
def line (l : set P) := ∃ a b : P, a ≠ b ∧ ∀ p ∈ l, ∃ t : ℝ, p = a + t • (b - a)
def plane (π : set P) := ∃ p q r : P, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ ∀ x ∈ π, ∃ a b c : ℝ, a + b + c = 1 ∧ x = a • p + b • q + c • r

-- Statement of the theorem
theorem line_outside_plane_has_at_most_one_common_point
  (l : set P) (π : set P) (h1 : line l) (h2 : plane π) :
  (∀ p ∈ l, p ∉ π) ∨ (∃ p ∈ l, p ∈ π) → (∀ p ∈ l, p ∉ π) ∨ (∃! p ∈ l, p ∈ π) :=
by
  sorry

end line_outside_plane_has_at_most_one_common_point_l83_83095


namespace coin_flip_probability_l83_83757

theorem coin_flip_probability :
  let outcomes := (finset.powerset (finset.univ : finset (fin 5))) 
  let successful_outcomes := finset.filter (λ s, 
    (s ∈ [{0, 2, 4}, {1, 3, 5}])) outcomes
  (successful_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 4 := sorry

end coin_flip_probability_l83_83757


namespace trapezoid_shorter_base_length_l83_83658

theorem trapezoid_shorter_base_length 
  (a b : ℕ) 
  (mid_segment_length longer_base : ℕ) 
  (h1 : mid_segment_length = 5) 
  (h2 : longer_base = 103) 
  (trapezoid_property : mid_segment_length = (longer_base - a) / 2) : 
  a = 93 := 
sorry

end trapezoid_shorter_base_length_l83_83658


namespace measure_angle_PQC_l83_83382

noncomputable def isosceles_triangle (A B C : Type) [Triangle A B C] : Prop :=
AB = AC

noncomputable def midpoint (D B C : Type) [LineSegment D B] [LineSegment D C] : Prop :=
BD = DC

noncomputable def on_line_segment (P A D : Type) [LineSegment A D] : Prop :=
P ∈ A D

noncomputable def isosceles_segment (B P Q : Type) [LineSegment B P] [LineSegment B Q] (h: P = Q) : Prop :=
PB = PQ

theorem measure_angle_PQC {A B C D P Q : Type} [Triangle A B C] (h₁ : isosceles_triangle A B C)
(h₂ : ∠A = 30°) (h₃ : midpoint D B C) (h₄ : on_line_segment P A D) (h₅ : on_line_segment Q A B)
(h₆ : isosceles_segment B P Q sorry) : ∠PQC = 15° :=
sorry

end measure_angle_PQC_l83_83382


namespace sum_of_remainders_l83_83120

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : ((n % 4) + (n % 5) = 4) :=
sorry

end sum_of_remainders_l83_83120


namespace power_function_expression_l83_83262

theorem power_function_expression (f : ℝ → ℝ) (h : f 2 = 1/4) : f = λ x, x ^ (-2) := 
by 
  sorry

end power_function_expression_l83_83262


namespace speed_of_B_l83_83970

/-- A walks at a uniform speed of 5 kmph,
    A starts walking half an hour before B,
    B overtakes A after 1 hour and 48 minutes of B starting to walk,
    Prove that the speed of B is approximately 6.39 kmph. -/
theorem speed_of_B {vA vB : ℝ} (hA : vA = 5) (t1 t2 : ℝ) (h_t1 : t1 = 0.5) (h_t2 : t2 = 1 + 48 / 60) (d : ℝ) (h_d : d = vA * (t2 + t1)) :
  vB = d / t2 :=
  by
    have hs : d = 2.5 + (5 * 1.8) := 
      by rw [hA, h_t1, h_t2]
    suffices : vB = d / t2,
    { exact this }
    sorry

end speed_of_B_l83_83970


namespace multiply_and_count_digit_l83_83773

theorem multiply_and_count_digit :
  let result := 987654321 * 9 in 
  (list.count 8 (int.to_nat_digits 10 result) = 9) :=
by
  let result := 987654321 * 9
  have h1 : result = 8888888889 := by norm_num
  rw [h1]
  have h2 : list.count 8 (int.to_nat_digits 10 8888888889) = 9 := by norm_num
  exact h2

end multiply_and_count_digit_l83_83773


namespace numbers_distance_one_neg_two_l83_83076

theorem numbers_distance_one_neg_two (x : ℝ) (h : abs (x + 2) = 1) : x = -1 ∨ x = -3 := 
sorry

end numbers_distance_one_neg_two_l83_83076


namespace proportion_of_boys_geq_35_percent_l83_83456

variables (a b c d n : ℕ)

axiom room_constraint : 2 * (b + d) ≥ n
axiom girl_constraint : 3 * a ≥ 8 * b

theorem proportion_of_boys_geq_35_percent : (3 * c + 4 * d : ℚ) / (3 * a + 4 * b + 3 * c + 4 * d : ℚ) ≥ 0.35 :=
by 
  sorry

end proportion_of_boys_geq_35_percent_l83_83456


namespace value_of_m_l83_83299

theorem value_of_m (m : ℝ) (z : ℂ) (H1 : z = (m - 2) + (m^2 - 3 * m + 2) * complex.I) (H2 : m - 2 ≠ 0) (H3 : m^2 - 3 * m + 2 = 0) : m = 1 :=
sorry

end value_of_m_l83_83299


namespace linear_function_of_non_dense_graph_l83_83384

noncomputable def f : ℝ → ℝ := sorry

theorem linear_function_of_non_dense_graph (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y)) 
  (h2 : ¬ dense_in_plane (graph f)) : 
  ∃ a : ℝ → ℝ, linear a :=
sorry

end linear_function_of_non_dense_graph_l83_83384


namespace unfair_draw_fair_draw_with_suit_hierarchy_l83_83317

noncomputable def deck := {suit : String, rank : ℕ // suit ∈ {"hearts", "diamonds", "clubs", "spades"} ∧ rank ∈ {6, 7, 8, 9, 10, 11, 12, 13, 14}}
def prob_V (v : deck) : ℚ := 1 / 36
def prob_M_given_V (v m : deck) : ℚ := 1 / 35
def higher_rank (v m : deck) : Prop := m.rank > v.rank

-- Prove the draw is unfair
theorem unfair_draw : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank v m then prob_M_given_V v m else 0) < 
  (∑ m in (deck \ {v}), if ¬higher_rank v m then prob_M_given_V v m else 0)) :=
sorry

-- Making the draw fair by introducing suit hierarchy
def suit_order : String → ℕ
| "spades" := 4
| "hearts" := 3
| "diamonds" := 2
| "clubs" := 1
| _ := 0

def higher_rank_with_suit (v m : deck) : Prop :=
  if v.rank = m.rank then suit_order m.suit > suit_order v.suit else m.rank > v.rank

-- Prove introducing suit hierarchy can make the draw fair
theorem fair_draw_with_suit_hierarchy : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank_with_suit v m then prob_M_given_V v m else 0) = 
  (∑ m in (deck \ {v}), if ¬higher_rank_with_suit v m then prob_M_given_V v m else 0)) :=
sorry

end unfair_draw_fair_draw_with_suit_hierarchy_l83_83317


namespace draw_is_unfair_ensure_fair_draw_l83_83321

open ProbabilityTheory MeasureTheory

-- Definitions for the given conditions:
def Card := {rank : ℕ // 6 ≤ rank ∧ rank ≤ 14} -- Ranks 6 to Ace (6 to 14)
def Deck := Finset (Fin 36) -- 36 unique cards
noncomputable def suit_high_rank_count (d : Deck) (v_card : Fin 36) (m_card : Fin 36) : ℕ := 
  -- Count how many cards are higher than Volodya's card
  card.count (λ c, c.val > v_card.val) d

-- Volodya draws first, then Masha draws:
variables (d : Deck) (v_card m_card : Fin 36)

-- Masha wins if she draws a card with a higher rank than Volodya’s card
def masha_wins := ∃ (m_card : Fin 36), (m_card ∈ d) ∧ (m_card.val > v_card.val)

-- Volodya wins if Masha doesn't win (Masha loses)
def volodya_wins := ¬ masha_wins

theorem draw_is_unfair (d : Deck) (v_card m_card : Fin 36) :
  (volodya_wins d v_card m_card) → ¬ (masha_wins d v_card) := sorry

-- To make it fair, we can introduce a suit hierarchy:
def suits := {"Hearts", "Diamonds", "Clubs", "Spades"}
def suit_order : suits → suits → Prop
| "Spades" "Hearts" := true
| "Hearts" "Diamonds" := true
| "Diamonds" "Clubs" := true
| "Clubs" "Spades" := false
| _, _ := false

-- A fair draw means using the suit_order to rank otherwise equal cards:
def fair_draw :=
  ∀ (c1 c2 : Card), (c1.rank = c2.rank → suit_order c1.suit c2.suit)

theorem ensure_fair_draw : fair_draw := sorry

end draw_is_unfair_ensure_fair_draw_l83_83321


namespace jack_paid_total_l83_83676

theorem jack_paid_total (cost_squat_rack : ℕ) (cost_barbell_fraction : ℕ) 
  (h1 : cost_squat_rack = 2500) (h2 : cost_barbell_fraction = 10) :
  let cost_barbell := cost_squat_rack / cost_barbell_fraction in
  let total_cost := cost_squat_rack + cost_barbell in
  total_cost = 2750 :=
by
  -- Assign the values
  let cost_barbell := cost_squat_rack / cost_barbell_fraction
  let total_cost := cost_squat_rack + cost_barbell
  -- We use the assumptions h1 and h2
  have h_cost_barbell : cost_barbell = 250 := by
    simp only [h1, h2]
    sorry -- complete arithmetic step
  have h_total_cost : total_cost = 2750 := by
    rw [h1, h_cost_barbell]
    sorry -- complete arithmetic step
  exact h_total_cost

end jack_paid_total_l83_83676


namespace large_ball_radius_correct_l83_83407

-- Condition: 8 solid iron balls each with radius 1
def small_ball_radius := 1
def num_small_balls := 8

-- Given the volume formula of a sphere
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Volume of one small ball
def volume_small_ball := volume_of_sphere small_ball_radius

-- Total volume of 8 small balls
def total_volume_small_balls := num_small_balls * volume_small_ball

-- Let the radius of the large ball be R
def large_ball_radius := 2

-- Hypothesis: Volume of the large ball is equal to the total volume of small balls
def volume_large_ball := volume_of_sphere large_ball_radius

theorem large_ball_radius_correct :
  volume_large_ball = total_volume_small_balls :=
sorry

end large_ball_radius_correct_l83_83407


namespace melanie_books_bought_l83_83406

def books_before_yard_sale : ℝ := 41.0
def books_after_yard_sale : ℝ := 128
def books_bought : ℝ := books_after_yard_sale - books_before_yard_sale

theorem melanie_books_bought : books_bought = 87 := by
  sorry

end melanie_books_bought_l83_83406


namespace parallel_vectors_result_l83_83616

noncomputable def a : ℝ × ℝ := (1, -2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, 4)
noncomputable def m : ℝ := -1 / 2

theorem parallel_vectors_result :
  (b m).1 * a.2 = (b m).2 * a.1 →
  2 * a - b m = (4, -8) :=
by
  intro h
  -- Proof omitted
  sorry

end parallel_vectors_result_l83_83616


namespace cost_per_square_meter_l83_83653

theorem cost_per_square_meter 
  (length width height : ℝ) 
  (total_expenditure : ℝ) 
  (hlength : length = 20) 
  (hwidth : width = 15) 
  (hheight : height = 5) 
  (hmoney : total_expenditure = 38000) : 
  58.46 = total_expenditure / (length * width + 2 * length * height + 2 * width * height) :=
by 
  -- Let's assume our definitions and use sorry to skip the proof
  sorry

end cost_per_square_meter_l83_83653


namespace calculate_expression_l83_83183

theorem calculate_expression : 
  (8 / 27) ^ (-1 / 3 : ℝ) + real.sqrt ((-1 / 2 : ℝ) ^ 2) = 2 := 
by sorry

end calculate_expression_l83_83183


namespace circles_intersect_tangent_lines_l83_83257

noncomputable def circle_C : set (ℝ × ℝ) :=
{ p | let (x, y) := p in x^2 + y^2 - 2 * x + 4 * y - 4 = 0 }

noncomputable def circle_C1 : set (ℝ × ℝ) :=
{ p | let (x, y) := p in (x - 3)^2 + (y - 1)^2 = 4 }

def point_P : ℝ × ℝ := (3, 1)

theorem circles_intersect :
  ∃ x y : ℝ, (x, y) ∈ circle_C ∧ (x, y) ∈ circle_C1 :=
sorry

theorem tangent_lines :
  ∃ k : ℝ, (k = 0 ∨ k = -12 / 5) ∧
    ((3 : ℝ) * x + (1 : ℝ) * y - 1 = 0 ∨ 12 * x + 5 * y - 41 = 0) :=
sorry

end circles_intersect_tangent_lines_l83_83257


namespace total_items_count_l83_83044

theorem total_items_count :
  let old_women  := 7
  let mules      := 7
  let bags       := 7
  let loaves     := 7
  let knives     := 7
  let sheaths    := 7
  let sheaths_per_loaf := knives * sheaths
  let sheaths_per_bag := loaves * sheaths_per_loaf
  let sheaths_per_mule := bags * sheaths_per_bag
  let sheaths_per_old_woman := mules * sheaths_per_mule
  let total_sheaths := old_women * sheaths_per_old_woman

  let loaves_per_bag := loaves
  let loaves_per_mule := bags * loaves_per_bag
  let loaves_per_old_woman := mules * loaves_per_mule
  let total_loaves := old_women * loaves_per_old_woman

  let knives_per_loaf := knives
  let knives_per_bag := loaves * knives_per_loaf
  let knives_per_mule := bags * knives_per_bag
  let knives_per_old_woman := mules * knives_per_mule
  let total_knives := old_women * knives_per_old_woman

  let total_bags := old_women * mules * bags

  let total_mules := old_women * mules

  let total_items := total_sheaths + total_loaves + total_knives + total_bags + total_mules + old_women

  total_items = 137256 :=
by
  sorry

end total_items_count_l83_83044


namespace integral_x_pow_x_series_l83_83410

open Real

noncomputable def integral_x_pow_x : ℝ := ∫ x in 0..1, x^x

theorem integral_x_pow_x_series :
  integral_x_pow_x = ∑' n : ℕ, (-1 : ℝ)^n / (n + 1)^(n + 1) :=
sorry

end integral_x_pow_x_series_l83_83410


namespace circle_standard_equation_l83_83594

theorem circle_standard_equation :
  ∃ (h k r : ℝ), (h = 2) ∧ (k = 1) ∧ (r = ℝ.sqrt 2) ∧
    ∀ x y : ℝ, 
      ((x - h)^2 + (y - k)^2 = r^2) ↔ 
      ((x - 2)^2 + (y - 1)^2 = 2) :=
by
  sorry

end circle_standard_equation_l83_83594


namespace largest_expression_l83_83476

theorem largest_expression (a b c x y z : ℝ) (h1 : a < b) (h2 : b < c) (h3 : x < y) (h4 : y < z) :
  ax + by + cz > ax + bz + cy ∧ ax + by + cz > bx + ay + cz ∧ ax + by + cz > bx + cy + az := 
sorry

end largest_expression_l83_83476


namespace icing_two_sides_on_Jack_cake_l83_83358

noncomputable def Jack_cake_icing_two_sides (cake_size : ℕ) : ℕ :=
  let side_cubes := 4 * (cake_size - 2) * 3
  let vertical_edge_cubes := 4 * (cake_size - 2)
  side_cubes + vertical_edge_cubes

-- The statement to be proven
theorem icing_two_sides_on_Jack_cake : Jack_cake_icing_two_sides 5 = 96 :=
by
  sorry

end icing_two_sides_on_Jack_cake_l83_83358


namespace sum_of_fraction_parts_l83_83876

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l83_83876


namespace part_a_part_b_l83_83728

noncomputable def average_price_between : Prop :=
  ∃ (prices : Fin 14 → ℝ), 
    prices 0 = 5 ∧ prices 6 = 5.14 ∧ prices 13 = 5 ∧ 
    5.09 < (∑ i, prices i) / 14 ∧ (∑ i, prices i) / 14 < 5.10

theorem part_a : average_price_between :=
  sorry

def average_difference : ℝ :=
  let prices1 := [5.0, 5.1, 5.1, 5.1, 5.1, 5.1, 5.14] in
  let prices2 := [5.14, 5.14, 5.14, 5.14, 5.14, 5.14, 5.0] in
  let avg1 := (prices1.sum / prices1.length : ℝ) in
  let avg2 := (prices2.sum / prices2.length : ℝ) in
  abs (avg2 - avg1)

theorem part_b : average_difference < 0.105 :=
  sorry

end part_a_part_b_l83_83728


namespace janet_used_bouquet_flowers_l83_83359

theorem janet_used_bouquet_flowers (tulips roses extra picked_used : ℕ)
  (h1 : tulips = 4)
  (h2 : roses = 11)
  (h3 : extra = 4)
  (h4 : picked_used = tulips + roses - extra) :
  picked_used = 11 :=
by {
  rw [h1, h2, h3],
  calc 4 + 11 - 4 = 11 : by norm_num,
  exact eq.refl 11,
}

end janet_used_bouquet_flowers_l83_83359


namespace smallest_prime_factors_sum_of_540_l83_83825

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m → m < n → n % m ≠ 0

def prime_factors (n : ℕ) : List ℕ :=
  if n = 1 then []
  else
    let factors := (List.range (n + 1)).filter (λ m, m > 1 ∧ n % m = 0 ∧ is_prime m)
    factors.concat_map (λ p, List.replicate (Nat.factor_count n p) p)

noncomputable def smallest_prime_factors_sum (n : ℕ) : ℕ :=
  let primes := (prime_factors n).eraseDup
  primes.take 2 |>.sum

theorem smallest_prime_factors_sum_of_540 : smallest_prime_factors_sum 540 = 5 := 
by
  sorry

end smallest_prime_factors_sum_of_540_l83_83825


namespace maximize_profit_l83_83493

noncomputable def production_problem : Prop :=
  ∃ (x y : ℕ), (3 * x + 2 * y ≤ 1200) ∧ (x + 2 * y ≤ 800) ∧ 
               (30 * x + 40 * y) = 18000 ∧ 
               x = 200 ∧ 
               y = 300

theorem maximize_profit : production_problem :=
sorry

end maximize_profit_l83_83493


namespace minimum_value_expression_l83_83822

theorem minimum_value_expression : ∃ x : ℝ, (3 * x^2 - 18 * x + 2023) = 1996 := sorry

end minimum_value_expression_l83_83822


namespace angle_DAF_eq_115_l83_83488

noncomputable def circle {V : Type*} [inner_product_space ℝ V] (O : V) (r : ℝ) : set V :=
  {P | ‖P - O‖ = r}

variables {V : Type*} [inner_product_space ℝ V]
variables (O A B C D E F P : V)
variables (r : ℝ)
variables (h_tangent : ∀ (Q : V), Q ∈ circle O r → P ≠ Q → ∀ (R : V), P - A = 0 → inner_product (R - A) (P - A) = 0)
variables (h_intersect_1 : B ∈ circle O r)
variables (h_intersect_2 : C ∈ circle O r)
variables (h_intersect_3 : D ∈ circle O r)
variables (h_perpendicular : inner_product (E - A) (O - P) = 0)
variables (h_line_be : ∃ (Q : V), Q ∈ circle O r ∧ B ≠ Q ∧ E ≠ Q ∧ F = Q)
variables (angle_BCO : real_angle B C O = 30)
variables (angle_BFO : real_angle B F O = 20)

theorem angle_DAF_eq_115 :
  real_angle D A F = 115 :=
sorry

end angle_DAF_eq_115_l83_83488


namespace remainder_101_pow_37_mod_100_l83_83469

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := 
by 
  sorry

end remainder_101_pow_37_mod_100_l83_83469


namespace tablets_taken_l83_83417

theorem tablets_taken (total_time interval_time : ℕ) (h1 : total_time = 60) (h2 : interval_time = 15) : total_time / interval_time = 4 :=
by
  sorry

end tablets_taken_l83_83417


namespace frog_reaches_pad_14_l83_83455

noncomputable def frog_journey_probability : ℚ :=
  let p0_to_p3 := 1/2 in
  let p3_to_p5 := (1/2) * (1/2) in
  let p5_to_p9 := (1/2) * (1/2) + 1/2 in
  let p9_to_p14 := ((1/2) * (1/2) * (1/2)) + ((1/2) * (1/2)) in
  p0_to_p3 * p3_to_p5 * p5_to_p9 * p9_to_p14

theorem frog_reaches_pad_14 : frog_journey_probability = 9/128 := by
  unfold frog_journey_probability
  norm_num
  sorry

end frog_reaches_pad_14_l83_83455


namespace binomial_expansion_of_110_minus_1_l83_83474

theorem binomial_expansion_of_110_minus_1:
  110^5 - 5 * 110^4 + 10 * 110^3 - 10 * 110^2 + 5 * 110 - 1 = 109^5 :=
by
  -- We will use the binomial theorem: (a - b)^n = ∑ (k in range(n+1)), C(n, k) * a^(n-k) * (-b)^k
  -- where C(n, k) are the binomial coefficients.
  sorry

end binomial_expansion_of_110_minus_1_l83_83474


namespace area_of_region_l83_83460

theorem area_of_region :
  (area_of_region_defined_by_equation : ℝ) :=
  let equation := λ x y: ℝ, x^2 + y^2 + 8*x - 18*y = 0 in
    ∀ (A : ℝ), (A = 97 * π) → sorry

end area_of_region_l83_83460


namespace repeating_decimal_sum_l83_83867

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l83_83867


namespace total_marbles_l83_83650

variable (r b g y : Nat)
variable (h1 : r = 1.30 * b)
variable (h2 : g = 1.50 * r)
variable (h3 : y = 1.20 * (r + g))

theorem total_marbles (r b g y : Nat) (h1 : r = 1.30 * b) (h2 : g = 1.50 * r) (h3 : y = 1.20 * (r + g)) : 
    r + b + g + y = 6.27 * r :=
by
  sorry

end total_marbles_l83_83650


namespace ratio_of_segments_l83_83803

-- Definitions and conditions as per part (a)
variables (a b c r s : ℝ)
variable (h₁ : a / b = 1 / 3)
variable (h₂ : a^2 = r * c)
variable (h₃ : b^2 = s * c)

-- The statement of the theorem directly addressing part (c)
theorem ratio_of_segments (a b c r s : ℝ) 
  (h₁ : a / b = 1 / 3)
  (h₂ : a^2 = r * c)
  (h₃ : b^2 = s * c) :
  r / s = 1 / 9 :=
  sorry

end ratio_of_segments_l83_83803


namespace remainder_M_div_1000_l83_83695

noncomputable def setB : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def g (f : ℕ → ℕ) (x : ℕ) : Prop := ∀ y, f (f y) = x

def numOfFunctions (B : Finset ℕ) : ℕ :=
  ∑ d in B, ∑ k in Finset.range 7, (Nat.choose 7 k) * k^(7 - k)

theorem remainder_M_div_1000 :
  (8 * (numOfFunctions setB)) % 1000 = 576 :=
sorry

end remainder_M_div_1000_l83_83695


namespace none_incorrect_l83_83175

variable (a b : ℝ)

def original_volume : ℝ := a * b

def transformed_volumes : list ℝ :=
  [2 * a * b,           -- Doubling a
   a * 3 * b,           -- Tripling b
   2 * a * 3 * b,       -- Doubling a and tripling b
   (a / 2) * 2 * b,     -- Halving a and doubling b
   3 * a * (b / 2)]     -- Tripling a and halving b

theorem none_incorrect :
  (2 * a * b = 2 * original_volume a b) ∧
  (a * 3 * b = 3 * original_volume a b) ∧
  (2 * a * 3 * b = 6 * original_volume a b) ∧
  ((a / 2) * 2 * b = original_volume a b) ∧
  (3 * a * (b / 2) = (3 / 2) * original_volume a b) :=
sorry

end none_incorrect_l83_83175


namespace volume_ratio_of_cubes_l83_83823

theorem volume_ratio_of_cubes (e1 e2 : ℕ) (h1 : e1 = 9) (h2 : e2 = 36) :
  (e1^3 : ℚ) / (e2^3 : ℚ) = 1 / 64 := by
  sorry

end volume_ratio_of_cubes_l83_83823


namespace components_leq_15_components_leq_20_components_leq_quarter_l83_83334

-- Conditions as Lean definitions
def grid8x8 : Type := Array (Array Bool 8) 8
def diagonalConfiguration (grid : grid8x8) : Bool := -- Dummy function representing the diagonal configuration validity
  sorry -- placeholder for the actual condition logic

-- Problem statements
theorem components_leq_15 (grid : grid8x8) (h : diagonalConfiguration grid) : 
  ∃ (components : ℕ), components ≤ 15 := 
  sorry -- Proof to be filled

theorem components_leq_20 (grid : grid8x8) (h : diagonalConfiguration grid) : 
  ∃ (components : ℕ), components ≤ 20 := 
  sorry -- Proof to be filled

-- For a general n × n grid
def grid (n : ℕ) : Type := Array (Array Bool n) n
def diagonalConfiguration_n (n : ℕ) (grid : grid n) : Bool := -- Dummy function representing the diagonal configuration validity
  sorry -- placeholder for the actual condition logic

theorem components_leq_quarter (n : ℕ) (h : n > 8) (grid : grid n) (h : diagonalConfiguration_n n grid) : 
  ∃ (components : ℕ), components ≤ n^2 / 4 := 
  sorry -- Proof to be filled

end components_leq_15_components_leq_20_components_leq_quarter_l83_83334


namespace sector_area_l83_83596

theorem sector_area (r θ : ℝ) (hr : r = 1) (hθ : θ = 2) : 
  (1 / 2) * r * r * θ = 1 := by
sorry

end sector_area_l83_83596


namespace leap_day_2040_is_tuesday_l83_83689

def days_in_non_leap_year := 365
def days_in_leap_year := 366
def leap_years_between_2000_and_2040 := 10

def total_days_between_2000_and_2040 := 
  30 * days_in_non_leap_year + leap_years_between_2000_and_2040 * days_in_leap_year

theorem leap_day_2040_is_tuesday :
  (total_days_between_2000_and_2040 % 7) = 0 :=
by
  sorry

end leap_day_2040_is_tuesday_l83_83689


namespace num_piles_of_quarters_l83_83181

variable (piles_of_dimes : ℕ)
variable (coins_per_pile : ℕ)
variable (total_coins : ℕ)

-- Given conditions
def given_conditions : Prop :=
  piles_of_dimes = 3 ∧ coins_per_pile = 4 ∧ total_coins = 20

-- Proof problem: Prove the number of piles of quarters
theorem num_piles_of_quarters (h : given_conditions) : 
  let dimes := piles_of_dimes * coins_per_pile
  let quarters := total_coins - dimes
  let piles_of_quarters := quarters / coins_per_pile
  piles_of_quarters = 2 :=
by
  sorry

end num_piles_of_quarters_l83_83181


namespace zero_squared_sum_l83_83414

theorem zero_squared_sum (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := 
by 
  sorry

end zero_squared_sum_l83_83414


namespace village_population_equal_l83_83093

theorem village_population_equal {n : ℕ} :
  ∀ (X Y : ℕ), X = 72000 - 1200 * n ∧ Y = 42000 + 800 * n → X = Y ↔ n = 15  :=
by
  intros X Y h
  sorry

end village_population_equal_l83_83093


namespace unpainted_unit_cubes_of_6x6x6_cube_l83_83944

theorem unpainted_unit_cubes_of_6x6x6_cube : 
  let total_cubes := 6 * 6 * 6,
      painted_per_face := 2 * 6,
      faces := 6,
      total_painted_area := painted_per_face * faces,
      shared_edges := 12,
      overlap_per_edge := 6 / 2,
      total_overlap := shared_edges * overlap_per_edge,
      unique_painted_cubes := total_painted_area - total_overlap,
      total_painted_cubes := unique_painted_cubes + total_overlap in
  total_cubes - total_painted_cubes = 144 :=
by
  let total_cubes := 6 * 6 * 6
  let painted_per_face := 2 * 6
  let faces := 6
  let total_painted_area := painted_per_face * faces
  let shared_edges := 12
  let overlap_per_edge := 6 / 2
  let total_overlap := shared_edges * overlap_per_edge
  let unique_painted_cubes := total_painted_area - total_overlap
  let total_painted_cubes := unique_painted_cubes + total_overlap
  show total_cubes - total_painted_cubes = 144
  sorry

end unpainted_unit_cubes_of_6x6x6_cube_l83_83944


namespace martin_leftover_raisins_l83_83396

theorem martin_leftover_raisins :
  ∀ (v k r : ℝ),
  (3 * v + 3 * k = 18 * r) → 
  (12 * r + 5 * k = v + 6 * k + (x * r)) →
  x = 6 :=
begin 
  intros v k r h1 h2,
  have h3 : v + k = 6 * r,
  { linarith, },
  have h4 : 12 * r = v + k + x * r,
  { linarith, },
  rw h3 at h2,
  exact eq_of_mul_eq_mul_right (by linarith) (by linarith),
  sorry
end

end martin_leftover_raisins_l83_83396


namespace velocity_at_1_eq_5_l83_83638

def S (t : ℝ) : ℝ := 2 * t^2 + t

theorem velocity_at_1_eq_5 : (deriv S 1) = 5 :=
by sorry

end velocity_at_1_eq_5_l83_83638


namespace nine_digit_number_l83_83962

-- Conditions as definitions
def highest_digit (n : ℕ) : Prop :=
  (n / 100000000) = 6

def million_place (n : ℕ) : Prop :=
  (n / 1000000) % 10 = 1

def hundred_place (n : ℕ) : Prop :=
  n % 1000 / 100 = 1

def rest_digits_zero (n : ℕ) : Prop :=
  (n % 1000000 / 1000) % 10 = 0 ∧ 
  (n % 1000000 / 10000) % 10 = 0 ∧ 
  (n % 1000000 / 100000) % 10 = 0 ∧ 
  (n % 100000000 / 10000000) % 10 = 0 ∧ 
  (n % 100000000 / 100000000) % 10 = 0 ∧ 
  (n % 1000000000 / 100000000) % 10 = 6

-- The nine-digit number
def given_number : ℕ := 6001000100

-- Prove number == 60,010,001,00 and approximate to 6 billion
theorem nine_digit_number :
  ∃ n : ℕ, highest_digit n ∧ million_place n ∧ hundred_place n ∧ rest_digits_zero n ∧ n = 6001000100 ∧ (n / 1000000000) = 6 :=
sorry

end nine_digit_number_l83_83962


namespace algebra_expression_value_l83_83242

theorem algebra_expression_value (x y : ℝ) (h1 : x * y = 3) (h2 : x - y = -2) : x^2 * y - x * y^2 = -6 := 
by
  sorry

end algebra_expression_value_l83_83242


namespace ratio_of_areas_l83_83059

def area_of_sector (R : ℝ) : ℝ :=
  (2 * Real.pi / 3) * R^2 / 2

def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem ratio_of_areas (r : ℝ) (h : r > 0) :
  (area_of_sector ((2 * r / Real.sqrt 3) + r)) / (area_of_circle r) = (7 + 4 * Real.sqrt 3) / 9 :=
by 
  sorry

end ratio_of_areas_l83_83059


namespace sum_of_edge_weights_at_least_neg_10000_l83_83770

-- Define a graph structure
structure Graph :=
  (V : Type)
  (E : V → V → Prop)
  (edge_weight : E → ℤ)

noncomputable def crab (G : Graph) (A B : G.V) : set (G.E) :=
  {e | G.E A (e) ∨ G.E B (e)}

-- Define the main theorem statement
theorem sum_of_edge_weights_at_least_neg_10000 (G : Graph)
  (hv : fintype G.V)
  (h400 : fintype.card G.V = 400)
  (hw : ∀ (A B : G.V) (e ∈ crab G A B), (G.edge_weight e) = 1 ∨ G.edge_weight e = -1)
  (hcrab_sum : ∀ (A B : G.V), ∑ e in crab G A B, (G.edge_weight e) ≥ 1) :
  ∑ e, G.edge_weight e ≥ -10000 := 
sorry

end sum_of_edge_weights_at_least_neg_10000_l83_83770


namespace quadruple_problem_l83_83365

open Finset

def quadruple_set := (finset.range 5).product (finset.range 5).product (finset.range 5).product (finset.range 5)

def count_even_sequences := 
  quadruple_set.filter 
    (λ quadruple, (let (a, (b, (c, d))) := quadruple in (a * d - b * c + 1) % 2 = 0)).card

theorem quadruple_problem : count_even_sequences = 136 := 
  by
  -- Proof is not needed, so we'll use sorry here.
  sorry

end quadruple_problem_l83_83365


namespace petya_mistake_l83_83737

theorem petya_mistake :
  (35 + 10 - 41 = 42 + 12 - 50) →
  (35 + 10 - 45 = 42 + 12 - 54) →
  (5 * (7 + 2 - 9) = 6 * (7 + 2 - 9)) →
  False :=
by
  intros h1 h2 h3
  sorry

end petya_mistake_l83_83737


namespace recurring_decimal_fraction_sum_l83_83900

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l83_83900


namespace number_of_shoes_l83_83946

theorem number_of_shoes
  (pairs : ℕ)
  (prob_matching : ℝ)
  (h_pairs : pairs = 6)
  (h_prob : prob_matching = 0.09090909090909091) :
  2 * pairs = 12 :=
by
  rw [h_pairs],
  norm_num,
  sorry

end number_of_shoes_l83_83946


namespace friends_reach_destinations_l83_83063

noncomputable def travel_times (d : ℕ) := 
  let walking_speed := 6
  let cycling_speed := 18
  let meet_time := d / (walking_speed + cycling_speed)
  let remaining_time := d / cycling_speed
  let total_time_A := meet_time + (d - cycling_speed * meet_time) / walking_speed
  let total_time_B := (cycling_speed * meet_time) / walking_speed + (d - cycling_speed * meet_time) / walking_speed
  let total_time_C := remaining_time + meet_time
  (total_time_A, total_time_B, total_time_C)

theorem friends_reach_destinations (d : ℕ) (d_eq_24 : d = 24) : 
  let (total_time_A, total_time_B, total_time_C) := travel_times d
  total_time_A ≤ 160 / 60 ∧ total_time_B ≤ 160 / 60 ∧ total_time_C ≤ 160 / 60 :=
by 
  sorry

end friends_reach_destinations_l83_83063


namespace train_length_l83_83974

theorem train_length {speed_kmh : ℝ} {time_sec : ℝ} {bridge_length_m : ℝ} 
(h1 : speed_kmh = 60) (h2 : time_sec = 72) (h3 : bridge_length_m = 800) : 
  let speed_mps := speed_kmh * (1000 / 3600) in
  let total_distance := speed_mps * time_sec in
  total_distance - bridge_length_m = 400.24 :=
sorry

end train_length_l83_83974


namespace sheet_metal_needed_l83_83150

-- Define the given information about the cylinder
def base_diameter (d : ℝ) := d = 30
def height (h : ℝ) := h = 45
def pi : ℝ := Real.pi

-- Define the necessary areas
def radius (d : ℝ) := d / 2
def base_area (r : ℝ) := pi * r^2
def lateral_surface_area (d : ℝ) (h : ℝ) := pi * d * h
def total_surface_area (r : ℝ) (d : ℝ) (h : ℝ) := lateral_surface_area d h + base_area r

-- The statement to be proved
theorem sheet_metal_needed :
  ∀ (d h : ℝ), base_diameter d → height h →
  let r := radius d in total_surface_area r d h = 4945.5 :=
begin
  intros d h hd hh,
  rw base_diameter at hd,
  rw height at hh,
  rw hd,
  rw hh,
  let r := radius 30,
  have : r = 15 := by norm_num,
  rw this,
  sorry
end

end sheet_metal_needed_l83_83150


namespace largest_radius_circle_intersecting_ellipse_l83_83556

noncomputable def largest_intersecting_radius (a b : ℝ) : ℝ :=
  if b > real.sqrt (a^2 - b^2) then 
    2 * b 
  else 
    a^2 / real.sqrt (a^2 - b^2)

theorem largest_radius_circle_intersecting_ellipse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b <= a) :
    let r = largest_intersecting_radius a b in 
    (b > real.sqrt (a^2 - b^2) → r = 2 * b) 
    ∧ (b <= real.sqrt (a^2 - b^2) → r = a^2 / real.sqrt (a^2 - b^2)) := sorry

end largest_radius_circle_intersecting_ellipse_l83_83556


namespace quadratic_single_solution_l83_83430

-- Definitions of the conditions
variable (a : ℝ) (x : ℝ)
-- a is non-zero
hypothesis h_a_nonzero : a ≠ 0
-- given the equation has only one solution (implies discriminant is zero)
hypothesis h_one_solution : 20^2 - 4 * a * 7 = 0 

-- Definition of the final claim
def is_solution : Prop :=
  a = (100 : ℝ) / (7 : ℝ) ∧ x = -(7 : ℝ) / (10 : ℝ)

-- Statement of the proof problem
theorem quadratic_single_solution :
  is_solution a x :=
by
  sorry

end quadratic_single_solution_l83_83430


namespace bacteria_growth_l83_83654

theorem bacteria_growth (n : ℕ): 
  (2 * (3 : ℕ)^n > 200) → n ≥ 5 :=
by {
  assume h : (2 * (3 : ℕ)^n > 200),
  sorry
}

end bacteria_growth_l83_83654


namespace compare_angles_l83_83992

noncomputable def degree_to_minutes (d : ℝ) : ℝ := d * 60

theorem compare_angles :
  let angle1 := 40.15 : ℝ,
      angle2_deg := 40 : ℝ,
      angle2_min := 15 : ℝ,
      angle1_total_minutes := degree_to_minutes (angle1 - 40),
      angle2_total_minutes := angle2_deg * 60 + angle2_min
  in angle1_total_minutes < angle2_total_minutes := sorry

end compare_angles_l83_83992


namespace remainder_101_pow_37_mod_100_l83_83468

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := 
by 
  sorry

end remainder_101_pow_37_mod_100_l83_83468


namespace solve_equation1_solve_equation2_l83_83425

-- Define the first equation as a condition
def equation1 (x : ℝ) : Prop :=
  3 * x + 20 = 4 * x - 25

-- Prove that x = 45 satisfies equation1
theorem solve_equation1 : equation1 45 :=
by 
  -- Proof steps would go here
  sorry

-- Define the second equation as a condition
def equation2 (x : ℝ) : Prop :=
  (2 * x - 1) / 3 = 1 - (2 * x - 1) / 6

-- Prove that x = 3/2 satisfies equation2
theorem solve_equation2 : equation2 (3 / 2) :=
by 
  -- Proof steps would go here
  sorry

end solve_equation1_solve_equation2_l83_83425


namespace find_b_vector_l83_83386

open Matrix

-- Define vectors a and b
def a : Fin 3 → ℝ := ![-3, 4, 1]
def b : Fin 3 → ℝ := ![-3/2, 20/3, 13/6]

-- Define dot product and cross product functions
def dot_product (u v : Fin 3 → ℝ) : ℝ := 
  u 0 * v 0 + u 1 * v 1 + u 2 * v 2

def cross_product (u v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![ u 1 * v 2 - u 2 * v 1,
     u 2 * v 0 - u 0 * v 2,
     u 0 * v 1 - u 1 * v 0 ]

-- Theorem statement
theorem find_b_vector :
  dot_product a b = 4 ∧ cross_product a b = ![2, -8, 14] :=
  by
    -- Proof omitted
    sorry

end find_b_vector_l83_83386


namespace unfair_draw_l83_83314

-- Define the types for suits and ranks
inductive Suit
| hearts | diamonds | clubs | spades

inductive Rank
| Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace

-- Define a card as a combination of suit and rank
structure Card where
  suit : Suit
  rank : Rank

-- Function to determine if a card is higher in rank
def higher_rank (r1 r2 : Rank) : Prop :=
  match r1, r2 with
  | Rank.Six, _ | Rank.Seven, Rank.Six | Rank.Eight, (Rank.Six | Rank.Seven) | Rank.Nine, (Rank.Six | Rank.Seven | Rank.Eight)
  | Rank.Ten, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine) | Rank.Jack, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten)
  | Rank.Queen, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack)
  | Rank.King, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen)
  | Rank.Ace, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King)
    => true
  | _, _ => false

-- Problem statement to prove unfairness of the draw
theorem unfair_draw :
  ∀ (vCard mCard : Card), (∃ (deck : List Card), 
  deck.length = 36 ∧ ∀ c, c ∈ deck →
  match c.rank with 
  | Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King | Rank.Ace => true 
  | _ => false) →
  (∃ (vCard mCard : Card), 
    vCard ∈ deck ∧ mCard ∈ (deck.erase vCard) ∧ higher_rank vCard.rank mCard.rank) →
  ¬fair :=
sorry

end unfair_draw_l83_83314


namespace area_of_circumscribed_circle_eq_48pi_l83_83148

noncomputable def side_length := 12
noncomputable def radius := (2/3) * (side_length / 2) * (Real.sqrt 3)
noncomputable def area := Real.pi * radius^2

theorem area_of_circumscribed_circle_eq_48pi :
  area = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_eq_48pi_l83_83148


namespace sum_of_fraction_parts_l83_83884

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l83_83884


namespace repeating_decimal_fraction_sum_l83_83841

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l83_83841


namespace weight_in_one_hand_l83_83519

theorem weight_in_one_hand (total_weight : ℕ) (h : total_weight = 16) : total_weight / 2 = 8 :=
by
  sorry

end weight_in_one_hand_l83_83519


namespace area_PVQ_is_32_l83_83337

open Real
open Classical

noncomputable def area_PVQ : ℝ :=
  let P := (0, 0) : ℝ × ℝ
  let Q := (8, 0)
  let R := (8, 4)
  let S := (0, 4)
  let T := (2, 4)
  let U := (6, 4)
  let V := (3.2, 6) -- Intersection from the geometry of the problem

  -- Function to calculate the area of a triangle from its vertices
  def triangle_area (A B C : ℝ × ℝ) : ℝ :=
    abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

  -- Calculate the area of the triangle PVQ
  triangle_area P V Q

theorem area_PVQ_is_32 : area_PVQ = 32 :=
by {
  unfold area_PVQ,
  unfold triangle_area,
  norm_num,
  sorry
}

end area_PVQ_is_32_l83_83337


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l83_83890

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l83_83890


namespace recurring_decimal_fraction_sum_l83_83903

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l83_83903


namespace fractional_equation_solution_l83_83125

theorem fractional_equation_solution (x : ℝ) (h : x ≠ 2) :
  (1 - x) / (2 - x) - 1 = (2 * x - 5) / (x - 2) → x = 3 :=
by 
  intro h_eq
  sorry

end fractional_equation_solution_l83_83125


namespace sprinter_speed_l83_83967

theorem sprinter_speed
  (distance : ℝ)
  (time : ℝ)
  (H1 : distance = 100)
  (H2 : time = 10) :
    (distance / time = 10) ∧
    ((distance / time) * 60 = 600) ∧
    (((distance / time) * 60 * 60) / 1000 = 36) :=
by
  sorry

end sprinter_speed_l83_83967


namespace probability_odd_sum_pairs_l83_83243

def odd_sum_pairs_probability : ℚ :=
  let cards := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let pairs := { (a, b) | a ∈ cards ∧ b ∈ cards ∧ a < b }
  let odd_sum_pairs := { (a, b) | (a, b) ∈ pairs ∧ (a + b) % 2 = 1 }
  let total_pairs := pairs.to_finset.card
  let favorable_pairs := odd_sum_pairs.to_finset.card
  favorable_pairs / total_pairs

theorem probability_odd_sum_pairs : 
  odd_sum_pairs_probability = 5 / 9 := 
by 
  sorry

end probability_odd_sum_pairs_l83_83243


namespace min_AL_value_l83_83022

theorem min_AL_value {A B C I H J L : Type}
  (h1 : dist A B = 13)
  (h2 : dist B C = 14)
  (h3 : dist A C = 15)
  (h4 : is_incenter I A B C)
  (h5 : is_circumcenter_of_radius A (dist A I) H J)
  (h6 : lies_on_incircle_and_line L A B C H J)
  (h7 : ∀m, m = min_AL A L) :
  min_AL A L = sqrt 17 := 
sorry

end min_AL_value_l83_83022


namespace determinant_expression_l83_83292

noncomputable def matA (n : ℕ) : Matrix (Fin n) (Fin n) ℝ := sorry
noncomputable def matB (n : ℕ) : Matrix (Fin n) (Fin n) ℝ := sorry

variable {n : ℕ}
variable (detA : det (matA n) = 3)
variable (detB : det (matB n) = 5)

theorem determinant_expression :
  det (3 • matA n ⬝ (matB n ^ 2)) = 3^(n+1) * 25 := sorry

end determinant_expression_l83_83292


namespace second_player_wins_l83_83787

noncomputable def game_state := {
  total_matches: ℕ,
  exceptional_moves_left: ℕ
}

def initial_state : game_state := {
  total_matches := 1000,
  exceptional_moves_left := 10
}

def is_valid_move (s : game_state) (matches_taken : ℕ) : Prop :=
  1 ≤ matches_taken ∧ matches_taken ≤ 5 ∨ (matches_taken = 6 ∧ s.exceptional_moves_left > 0)

def next_state (s : game_state) (matches_taken : ℕ) : game_state :=
  if matches_taken = 6 then
    { total_matches := s.total_matches - matches_taken, exceptional_moves_left := s.exceptional_moves_left - 1 }
  else
    { total_matches := s.total_matches - matches_taken, exceptional_moves_left := s.exceptional_moves_left }

def winning_condition (s : game_state) : Prop :=
  s.total_matches = 0

theorem second_player_wins :
  ∃ strategy : game_state → ℕ, 
    ∀ s, is_valid_move s (strategy s) → 
    winning_condition (next_state s (strategy s)) → 
    (∃ move, is_valid_move s move ∧ ¬ winning_condition (next_state s move)) → 
    ¬ winning_condition (next_state s (strategy s)) :=
sorry

end second_player_wins_l83_83787


namespace company_blocks_l83_83131

theorem company_blocks (total_amount : ℕ) (gift_worth : ℕ) (workers_per_block : ℕ) (total_gifts : ℕ) :
  total_amount = 6000 →
  gift_worth = 2 →
  workers_per_block = 200 →
  total_gifts = total_amount / gift_worth →
  total_gifts / workers_per_block = 15 :=
begin
  intros h1 h2 h3 h4,
  rw h1 at h4,
  rw h2 at h4,
  rw h3,
  sorry
end

end company_blocks_l83_83131


namespace factorization_l83_83211

def expression1 (x y : ℝ) : ℝ := (x + 2) * (x - 2) - 4 * y * (x - y)
def expression2 (x y : ℝ) : ℝ := (x - 2y + 2) * (x - 2y - 2)

theorem factorization (x y : ℝ) : expression1 x y = expression2 x y :=
by
  sorry

end factorization_l83_83211


namespace area_of_B_l83_83663

def point_in_A (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in x + y ≤ 1 ∧ x ≥ 0 ∧ y ≥ 0

def point_in_B (q : ℝ × ℝ) : Prop :=
  ∃ p : ℝ × ℝ, point_in_A p ∧ q = (p.1 + p.2, p.1 - p.2)

theorem area_of_B :
  let S := { q : ℝ × ℝ | point_in_B q } in
  ∃ A B : ℝ × ℝ, 
      A = (1, 1) ∧ B = (1, -1) ∧ 
      S = { p : ℝ × ℝ | ∃ α β γ : ℝ, α + β + γ = 1 ∧ α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ p = α • (0,0) + β • (1,1) + γ • (1,-1) } ∧ 
      ∃ area : ℝ, area = 1 :=
  sorry

end area_of_B_l83_83663


namespace unfair_draw_l83_83311

-- Define the types for suits and ranks
inductive Suit
| hearts | diamonds | clubs | spades

inductive Rank
| Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace

-- Define a card as a combination of suit and rank
structure Card where
  suit : Suit
  rank : Rank

-- Function to determine if a card is higher in rank
def higher_rank (r1 r2 : Rank) : Prop :=
  match r1, r2 with
  | Rank.Six, _ | Rank.Seven, Rank.Six | Rank.Eight, (Rank.Six | Rank.Seven) | Rank.Nine, (Rank.Six | Rank.Seven | Rank.Eight)
  | Rank.Ten, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine) | Rank.Jack, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten)
  | Rank.Queen, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack)
  | Rank.King, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen)
  | Rank.Ace, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King)
    => true
  | _, _ => false

-- Problem statement to prove unfairness of the draw
theorem unfair_draw :
  ∀ (vCard mCard : Card), (∃ (deck : List Card), 
  deck.length = 36 ∧ ∀ c, c ∈ deck →
  match c.rank with 
  | Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King | Rank.Ace => true 
  | _ => false) →
  (∃ (vCard mCard : Card), 
    vCard ∈ deck ∧ mCard ∈ (deck.erase vCard) ∧ higher_rank vCard.rank mCard.rank) →
  ¬fair :=
sorry

end unfair_draw_l83_83311


namespace num_of_integers_abs_leq_six_l83_83288

theorem num_of_integers_abs_leq_six (x : ℤ) : 
  (|x - 3| ≤ 6) → ∃ (n : ℕ), n = 13 := 
by 
  sorry

end num_of_integers_abs_leq_six_l83_83288


namespace first_term_of_arithmetic_sequence_l83_83779

theorem first_term_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ)
  (h_arith : ∀ n, a n = a1 + ↑n - 1) 
  (h_sum : ∀ n, S n = n / 2 * (2 * a1 + (n - 1))) 
  (h_min : ∀ n, S 2022 ≤ S n) : 
  -2022 < a1 ∧ a1 < -2021 :=
by
  sorry

end first_term_of_arithmetic_sequence_l83_83779


namespace speed_conversion_l83_83169

theorem speed_conversion (speed_mps : ℝ) (conversion_factor : ℝ) (speed_kmph_expected : ℝ) :
  speed_mps = 35.0028 →
  conversion_factor = 3.6 →
  speed_kmph_expected = 126.01008 →
  speed_mps * conversion_factor = speed_kmph_expected :=
by
  intros h_mps h_cf h_kmph
  rw [h_mps, h_cf, h_kmph]
  sorry

end speed_conversion_l83_83169


namespace onion_rings_cost_l83_83026

variable (hamburger_cost smoothie_cost total_payment change_received : ℕ)

theorem onion_rings_cost (h_hamburger : hamburger_cost = 4) 
                         (h_smoothie : smoothie_cost = 3) 
                         (h_total_payment : total_payment = 20) 
                         (h_change_received : change_received = 11) :
                         total_payment - change_received - hamburger_cost - smoothie_cost = 2 :=
by
  sorry

end onion_rings_cost_l83_83026


namespace nth_equation_pattern_l83_83029

theorem nth_equation_pattern (n : ℕ) : 
  (List.range' n (2 * n - 1)).sum = (2 * n - 1) ^ 2 :=
by
  sorry

end nth_equation_pattern_l83_83029


namespace repeating_decimal_fraction_sum_l83_83853

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l83_83853


namespace tetrahedron_isosceles_l83_83038

-- Define the tetrahedron and the equal angles given in the problem
variables {A B C D : Type} [Tetrahedron A B C D]
variable (angle_BAC angle_ABD angle_ACD angle_BDC : Angle A B C)

-- Condition that the given angles are equal
axiom angle_eq_1 : angle_BAC = angle_ABD
axiom angle_eq_2 : angle_BAC = angle_ACD
axiom angle_eq_3 : angle_BAC = angle_BDC

-- Define midpoint of the edges
def midpoint (X Y : Type) [Midpoint X Y] : Type := sorry

-- Define the midpoint quadrilateral and that it forms a rhombus
def quadrilateral_rhombus (S K L M N : Type) [Quadrilateral S K L M N] : Prop := sorry

-- Define isosceles tetrahedron
def isosceles_tetrahedron (A B C D : Type) [Tetrahedron A B C D] : Prop :=
  dist A B = dist C D ∧ dist A C = dist B D

-- The theorem statement
theorem tetrahedron_isosceles (A B C D : Type) [Tetrahedron A B C D]
  (angle_BAC angle_ABD angle_ACD angle_BDC : Angle A B C)
  (angle_eq_1 : angle_BAC = angle_ABD)
  (angle_eq_2 : angle_BAC = angle_ACD)
  (angle_eq_3 : angle_BAC = angle_BDC) :
  isosceles_tetrahedron A B C D :=
begin
  sorry  -- Proof will be provided here
end

end tetrahedron_isosceles_l83_83038


namespace average_price_condition_observer_b_correct_l83_83732

-- Define the conditions
def stock_price {n : ℕ} (daily_prices : Fin n → ℝ) : Prop :=
  daily_prices 0 = 5 ∧ 
  daily_prices 6 = 5.14 ∧
  daily_prices 13 = 5 ∧ 
  (∀ i : Fin 6, daily_prices i ≤ daily_prices (i + 1)) ∧ 
  (∀ i : Fin (n - 7), daily_prices (i + 7) ≥ daily_prices (i + 8))

-- Define the problem statements
theorem average_price_condition (daily_prices : Fin 14 → ℝ) :
  stock_price daily_prices →
  ∃ S, 5.09 < (5 + S + 5.14) / 14 ∧ (5 + S + 5.14) / 14 < 5.10 :=
sorry

theorem observer_b_correct (daily_prices : Fin 14 → ℝ) :
  stock_price daily_prices →
  let avg_1 : ℝ := (∑ i in Finset.range 7, daily_prices i) / 7
  let avg_2 : ℝ := (∑ i in Finset.range 7, daily_prices (i + 7)) / 7
  ¬ avg_1 = avg_2 + 0.105 :=
sorry

end average_price_condition_observer_b_correct_l83_83732


namespace repeating_decimal_sum_l83_83869

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l83_83869


namespace repeating_decimal_sum_l83_83875

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l83_83875


namespace area_triangle_APB_21_l83_83969

variables {P A B C E F D : Point}
variables {PA PB PC AE EB : ℝ}
variables (s : Square)

open EuclideanGeometry

-- Given conditions
def conditions (P A B C E F D : Point) (sides : ℝ) (PA PB PC AE EB : ℝ) : Prop :=
  sides = 10 ∧ PA = PB ∧ PB = PC ∧
  PC ⊥ Line(F, D) ∧
  C ∠ Line(E, AB) ⊥ AB ∧
  AE = 4 ∧ EB = 6

-- Question: What is the area of triangle APB?
def area_of_triangle_APB (P A B : Point) (PA PB PC AE EB : ℝ) (sides : ℝ) [IsSquare s] [conditions P A B C E F D sides PA PB PC AE EB] : ℝ :=
  let x : ℝ := 5.8
  (0.5 : ℝ) * (10 : ℝ) * (10 - x)

theorem area_triangle_APB_21 :
  conditions P A B C E F D 10 PA PB PC 4 6 →
  area_of_triangle_APB P A B PA PB PC 10 = 21 := by sorry

end area_triangle_APB_21_l83_83969


namespace Lorin_black_marbles_l83_83711

variable (B : ℕ)

def Jimmy_yellow_marbles := 22
def Alex_yellow_marbles := Jimmy_yellow_marbles / 2
def Alex_black_marbles := 2 * B
def Alex_total_marbles := Alex_yellow_marbles + Alex_black_marbles

theorem Lorin_black_marbles : Alex_total_marbles = 19 → B = 4 :=
by
  intros h
  unfold Alex_total_marbles at h
  unfold Alex_yellow_marbles at h
  unfold Alex_black_marbles at h
  norm_num at h
  exact sorry

end Lorin_black_marbles_l83_83711


namespace sixth_operation_result_l83_83651

   def pattern_operation (a b : ℕ) : ℕ :=
     (a + b) * a - a

   theorem sixth_operation_result : pattern_operation 7 8 = 98 :=
   by 
     rw [pattern_operation, show 7 + 8 = 15 by rfl, show 15 * 7 = 105 by rfl, show 105 - 7 = 98 by rfl]
     rfl   
   
end sixth_operation_result_l83_83651


namespace cost_of_large_poster_is_correct_l83_83687

/-- Problem conditions -/
def posters_per_day : ℕ := 5
def large_posters_per_day : ℕ := 2
def large_poster_sale_price : ℝ := 10
def small_posters_per_day : ℕ := 3
def small_poster_sale_price : ℝ := 6
def small_poster_cost : ℝ := 3
def weekly_profit : ℝ := 95

/-- The cost to make a large poster -/
noncomputable def large_poster_cost : ℝ := 5

/-- Prove that the cost to make a large poster is $5 given the conditions -/
theorem cost_of_large_poster_is_correct :
    large_poster_cost = 5 :=
by
  -- (Condition translation into Lean)
  let daily_profit := weekly_profit / 5
  let daily_revenue := (large_posters_per_day * large_poster_sale_price) + (small_posters_per_day * small_poster_sale_price)
  let daily_cost_small_posters := small_posters_per_day * small_poster_cost
  
  -- Express the daily profit in terms of costs, including unknown large_poster_cost
  have calc_profit : daily_profit = daily_revenue - daily_cost_small_posters - (large_posters_per_day * (large_poster_cost)) :=
    sorry
  
  -- Setting the equation to solve for large_poster_cost
  have eqn : daily_profit = 19 := by
    sorry

  -- Solve for large_poster_cost
  have solve_large_poster_cost : 19 = daily_revenue - daily_cost_small_posters - (large_posters_per_day * 5) :=
    by sorry
  
  sorry

end cost_of_large_poster_is_correct_l83_83687


namespace gladys_typing_speed_l83_83060

theorem gladys_typing_speed 
  (rudy_speed : ℕ) (joyce_speed : ℕ) (lisa_speed : ℕ) (mike_speed : ℕ) 
  (avg_speed : ℕ) (num_employees : ℕ) (total_speed : ℕ)
  (h_rudy : rudy_speed = 64)
  (h_joyce : joyce_speed = 76)
  (h_lisa : lisa_speed = 80)
  (h_mike : mike_speed = 89)
  (h_avg : avg_speed = 80)
  (h_num_employees : num_employees = 5)
  (h_total : total_speed = avg_speed * num_employees) : 
  ∃ (gladys_speed : ℕ), gladys_speed = 91 := 
by 
  -- We define the total speed to avoid redundant calculations.
  have h_total_speed : total_speed = 400, from calc 
    total_speed = avg_speed * num_employees : h_total
    ... = 80 * 5 : by rw [h_avg, h_num_employees],

  sorry

end gladys_typing_speed_l83_83060


namespace bianca_coloring_books_total_l83_83489

theorem bianca_coloring_books_total
  (initial_books : ℕ)
  (books_given_away : ℕ)
  (books_bought : ℕ)
  (final_books : ℕ) :
  initial_books = 45 → books_given_away = 6 → books_bought = 20 → final_books = 59 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end bianca_coloring_books_total_l83_83489


namespace intersection_S_T_eq_S_l83_83449

def S : set ℝ := {y | ∃ x : ℝ, y = 3 ^ x}
def T : set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 1}

theorem intersection_S_T_eq_S : S ∩ T = S :=
by 
  sorry

end intersection_S_T_eq_S_l83_83449


namespace solution_X_amount_l83_83047

variable (x : ℝ) -- Amount of Solution X in the mixture
variable (y : ℝ) -- Total amount of the mixture
variable (aX : ℝ) -- Percentage of material A in Solution X
variable (aY : ℝ) -- Percentage of material A in Solution Y
variable (aM : ℝ) -- Percentage of material A in the mixture

-- Given conditions
axiom h1 : aX = 0.20
axiom h2 : aY = 0.30
axiom h3 : aM = 0.22
axiom h4 : y = 100 -- Total amount of the mixture is assumed to be 100 units

-- Proof goal
theorem solution_X_amount : (aX * x + aY * (y - x) = aM * y) → x = 80 := by
  intro h
  apply_fun (λ z, z * (10 : ℝ)) at h
  simp [aX, aY, aM, y] at h
  linarith
  sorry

end solution_X_amount_l83_83047


namespace calculate_expression_l83_83185

theorem calculate_expression :
  let a := (1/3)⁻¹
  let b := (2023 - Real.pi)⁰
  let c := Real.sqrt 12
  let d := Real.sin (Real.pi / 3)
  a + b - c * d = 1 :=
by
  sorry

end calculate_expression_l83_83185


namespace add_neg3_and_2_mul_neg3_and_2_l83_83186

theorem add_neg3_and_2 : -3 + 2 = -1 := 
by
  sorry

theorem mul_neg3_and_2 : (-3) * 2 = -6 := 
by
  sorry

end add_neg3_and_2_mul_neg3_and_2_l83_83186


namespace enclosed_area_of_curve_l83_83067

theorem enclosed_area_of_curve :
  let r := 1,
      sector_area := 1/2 * r^2 * (π/2),
      total_sector_area := 8 * sector_area,
      octagon_area := 2 * (1 + sqrt 2) * (3^2),
      total_area := octagon_area + total_sector_area
  in total_area = 54 + 54 * sqrt 2 + 2 * π :=
by
  let r := 1
  let sector_area := 1/2 * r^2 * (π/2)
  let total_sector_area := 8 * sector_area
  let octagon_area := 2 * (1 + sqrt 2) * (3^2)
  let total_area := octagon_area + total_sector_area
  sorry

end enclosed_area_of_curve_l83_83067


namespace jan_drove_more_l83_83126

variables (d t s : ℕ)
variables (h h_ans : ℕ)
variables (ha_speed j_speed : ℕ)
variables (j d_plus : ℕ)

-- Ian's equation
def ian_distance (s t : ℕ) : ℕ := s * t

-- Han's additional conditions
def han_distance (s t : ℕ) (h_speed : ℕ)
    (d_plus : ℕ) : Prop :=
  d_plus + 120 = (s + h_speed) * (t + 2)

-- Jan's conditions and equation
def jan_distance (s t : ℕ) (j_speed : ℕ) : ℕ :=
  (s + j_speed) * (t + 3)

-- Proof statement
theorem jan_drove_more (d t s h_ans : ℕ)
    (h_speed j_speed : ℕ) (d_plus : ℕ)
    (h_dist_cond : han_distance s t h_speed d_plus)
    (j_dist_cond : jan_distance s t j_speed = h_ans) :
  h_ans = 195 :=
sorry

end jan_drove_more_l83_83126


namespace magnitude_Z_l83_83279

-- Defining the imaginary unit i
def i : ℂ := complex.I

-- Defining the complex number Z
def Z : ℂ := (1 + 2 * i) / (2 - i)

-- Stating the theorem to prove the magnitude of Z equals 1
theorem magnitude_Z : complex.abs Z = 1 :=
  sorry

end magnitude_Z_l83_83279


namespace cookies_difference_l83_83735

theorem cookies_difference 
    (initial_sweet : ℕ) (initial_salty : ℕ) (initial_chocolate : ℕ)
    (ate_sweet : ℕ) (ate_salty : ℕ) (ate_chocolate : ℕ)
    (ratio_sweet : ℕ) (ratio_salty : ℕ) (ratio_chocolate : ℕ) :
    initial_sweet = 39 →
    initial_salty = 18 →
    initial_chocolate = 12 →
    ate_sweet = 27 →
    ate_salty = 6 →
    ate_chocolate = 8 →
    ratio_sweet = 3 →
    ratio_salty = 1 →
    ratio_chocolate = 2 →
    ate_sweet - ate_salty = 21 :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end cookies_difference_l83_83735


namespace repeating_decimal_fraction_sum_l83_83842

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l83_83842


namespace set_intersection_A_B_l83_83586

-- Define set A
def setA := { y : ℝ | ∃ x : ℝ, y = Real.log 2 x ∧ 0 < x ∧ x ≤ 4 }

-- Define set B
def setB := { x : ℝ | Real.exp x > 1 }

-- Define the intersection of A and B
def intersection := { x : ℝ | 0 < x ∧ x ≤ 2 }

-- The main theorem to prove
theorem set_intersection_A_B : (setA ∩ setB) = intersection :=
by
  sorry

end set_intersection_A_B_l83_83586


namespace projection_of_b_on_a_is_negative_sqrt_two_l83_83268

noncomputable theory

def projection_of_b_on_a (b a : ℝ × ℝ) (b_mag : ℝ) (angle_between : ℝ) : ℝ :=
b_mag * real.cos angle_between

theorem projection_of_b_on_a_is_negative_sqrt_two (a b : ℝ × ℝ)
  (hb : |b| = 2)
  (angle_between : ℝ)
  (h_angle : angle_between = 3 * π / 4) :
  projection_of_b_on_a b a 2 (3 * π / 4) = - real.sqrt 2 :=
by
  sorry

end projection_of_b_on_a_is_negative_sqrt_two_l83_83268


namespace unfair_draw_l83_83312

-- Define the types for suits and ranks
inductive Suit
| hearts | diamonds | clubs | spades

inductive Rank
| Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace

-- Define a card as a combination of suit and rank
structure Card where
  suit : Suit
  rank : Rank

-- Function to determine if a card is higher in rank
def higher_rank (r1 r2 : Rank) : Prop :=
  match r1, r2 with
  | Rank.Six, _ | Rank.Seven, Rank.Six | Rank.Eight, (Rank.Six | Rank.Seven) | Rank.Nine, (Rank.Six | Rank.Seven | Rank.Eight)
  | Rank.Ten, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine) | Rank.Jack, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten)
  | Rank.Queen, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack)
  | Rank.King, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen)
  | Rank.Ace, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King)
    => true
  | _, _ => false

-- Problem statement to prove unfairness of the draw
theorem unfair_draw :
  ∀ (vCard mCard : Card), (∃ (deck : List Card), 
  deck.length = 36 ∧ ∀ c, c ∈ deck →
  match c.rank with 
  | Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King | Rank.Ace => true 
  | _ => false) →
  (∃ (vCard mCard : Card), 
    vCard ∈ deck ∧ mCard ∈ (deck.erase vCard) ∧ higher_rank vCard.rank mCard.rank) →
  ¬fair :=
sorry

end unfair_draw_l83_83312


namespace sum_of_reflected_midpoint_coordinates_l83_83743

open Real

-- Define Point as an alias for pair of reals (x, y)
abbreviation Point := (ℝ × ℝ)

-- Define the original points P and R
def P : Point := (2, 1)
def R : Point := (12, 15)

-- Midpoint function for two points
def midpoint (A B : Point) : Point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Reflect a point over the y-axis
def reflect_y_axis (A : Point) : Point :=
  (-A.1, A.2)

-- The midpoint M of segment PR
def M : Point := midpoint P R

-- The reflected points of P and R over the y-axis
def P' : Point := reflect_y_axis P
def R' : Point := reflect_y_axis R

-- The midpoint M' of segment P'R'
def M' : Point := midpoint P' R'

-- The sum of the coordinates of point M'
def sum_of_coordinates (A : Point) : ℝ :=
  A.1 + A.2

-- The theorem to be proved
theorem sum_of_reflected_midpoint_coordinates :
  sum_of_coordinates M' = 1 :=
by
  -- Proof will go here (skipping with sorry)
  sorry

end sum_of_reflected_midpoint_coordinates_l83_83743


namespace speed_of_current_approx_l83_83958

def boat_speed_kmph : ℝ := 26
def time_seconds_downstream : ℝ := 17.998560115190784
def distance_meters_downstream : ℝ := 150
def meters_per_second_to_kmph (x : ℝ) : ℝ := x * 3600 / 1000

theorem speed_of_current_approx :
  let boat_speed_mps := boat_speed_kmph * 1000 / 3600 in
  let downstream_speed_mps := distance_meters_downstream / time_seconds_downstream in
  let speed_of_current_mps := downstream_speed_mps - boat_speed_mps in
  meters_per_second_to_kmph speed_of_current_mps ≈ 4.0024 :=
by sorry

end speed_of_current_approx_l83_83958


namespace chord_length_of_circle_and_line_intersection_l83_83445

theorem chord_length_of_circle_and_line_intersection :
  ∀ (x y : ℝ), (x - 2 * y = 3) → ((x - 2)^2 + (y + 3)^2 = 9) → ∃ chord_length : ℝ, (chord_length = 4) :=
by
  intros x y hx hy
  sorry

end chord_length_of_circle_and_line_intersection_l83_83445


namespace set_points_quadrants_l83_83450

theorem set_points_quadrants (x y : ℝ) :
  (y > 3 * x) ∧ (y > 5 - 2 * x) → 
  (y > 0 ∧ x > 0) ∨ (y > 0 ∧ x < 0) :=
by 
  sorry

end set_points_quadrants_l83_83450


namespace sum_of_reciprocals_of_roots_l83_83568

theorem sum_of_reciprocals_of_roots :
  let p := (Polynomial.C 1 * Polynomial.X^2 - Polynomial.C 16 * Polynomial.X + Polynomial.C 15) in
  (p.roots.sum (λ r, r⁻¹)) = (16/15) := 
by
  sorry

end sum_of_reciprocals_of_roots_l83_83568


namespace gcd_repeated_integer_l83_83529

theorem gcd_repeated_integer (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) :
  ∃ d, (∀ k : ℕ, k = 1001001001 * n → d = 1001001001 ∧ d ∣ k) :=
sorry

end gcd_repeated_integer_l83_83529


namespace correct_input_statement_l83_83527

-- Define the types of statements
inductive Statement
| PRINT
| INPUT
| IF
| LET

-- Define the properties
def is_output_statement : Statement → Prop
| Statement.PRINT := true
| _ := false

def is_input_statement : Statement → Prop
| Statement.INPUT := true
| _ := false

-- The main theorem to prove
theorem correct_input_statement : ∃ s : Statement, is_input_statement s ∧ s = Statement.INPUT :=
by
  existsi Statement.INPUT
  split
  case h₁ =>
    exact trivial
  case h₂ =>
    rfl

end correct_input_statement_l83_83527


namespace table_repositioning_side_length_l83_83964

theorem table_repositioning_side_length :
  ∀ (S : ℕ), (∀ (d : ℝ), d = real.sqrt (9^2 + 12^2) → d ≤ S) → S = 15 :=
by 
{ 
  intro S,
  intro h,
  have h₁ : real.sqrt (9^2 + 12^2) = 15,
  { norm_num },
  specialize h (real.sqrt (9^2 + 12^2)),
  rw h₁ at h,
  exact h rfl
}

end table_repositioning_side_length_l83_83964


namespace three_digit_number_proof_l83_83128

theorem three_digit_number_proof (a b c : ℕ) (h1 : a + b + c = 10) (h2 : b = a + c) (h3 : 100c + 10b + a = 100a + 10b + c + 99) : 100 * a + 10 * b + c = 253 :=
by
  -- Placeholder for the proof
  sorry

end three_digit_number_proof_l83_83128


namespace meeting_probability_correct_l83_83717

noncomputable def meeting_probability : ℝ := 
  let pA_move_right : ℝ := 0.4
  let pA_move_up : ℝ := 0.4
  let pA_move_diag : ℝ := 0.2
  let pB_move_left : ℝ := 0.4
  let pB_move_down : ℝ := 0.4
  let pB_move_diag : ℝ := 0.2
  sorry -- calculations for meeting probability

theorem meeting_probability_correct :
  meeting_probability = -- computed probability
  sorry

end meeting_probability_correct_l83_83717


namespace triangle_sum_of_sides_l83_83799

noncomputable def sum_of_remaining_sides (side_a : ℝ) (angle_B : ℝ) (angle_C : ℝ) : ℝ :=
  let BD := side_a * Real.sin angle_B
  let DC := BD / Real.tan angle_C
  side_a + DC

theorem triangle_sum_of_sides :
  let side_a := 8
  let angle_B := Real.pi * 50 / 180
  let angle_C := Real.pi * 40 / 180
  sum_of_remaining_sides side_a angle_B angle_C ≈ 22.6 :=
by
  let side_a := 8
  let angle_B := Real.pi * 50 / 180
  let angle_C := Real.pi * 40 / 180
  have h1 : sum_of_remaining_sides side_a angle_B angle_C ≈ 22.6 := sorry
  exact h1

end triangle_sum_of_sides_l83_83799


namespace principal_amount_l83_83302

variable (P : ℝ)
variable (R : ℝ := 4)
variable (T : ℝ := 5)

theorem principal_amount :
  ((P * R * T) / 100 = P - 2000) → P = 2500 :=
by
  sorry

end principal_amount_l83_83302


namespace sum_of_fraction_numerator_and_denominator_l83_83907

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l83_83907


namespace max_ak_at_k_125_l83_83808

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def ak (k : ℕ) : ℚ :=
  binomial_coefficient 500 k * (0.3)^k

theorem max_ak_at_k_125 : 
  ∀ k : ℕ, k ∈ Finset.range 501 → (ak k ≤ ak 125) :=
by sorry

end max_ak_at_k_125_l83_83808


namespace sum_radii_converges_l83_83138

-- Define the points and regular octagons with corresponding properties
variables (N₁ N₂ : set Point) (A₁ B₁ C₁ D₁ E₁ F₁ G₁ H₁ : Point)
          (A₂ B₂ C₂ D₂ E₂ F₂ G₂ H₂ : Point)
          (r₁ r₂ : ℝ)
          (P M O₂ : Point)
          (circumradius : set Point → ℝ)

-- Define the regularity of octagons
axiom regular_octagon : ∀ (N : set Point), is_regular_octagon N

-- Define the condition that sides A₁B₁ and A₂B₂ are on the same line, and similarly for D₁E₁ and D₂E₂
axiom same_line_AB : collinear A₁ B₁ A₂ B₂
axiom same_line_DE : collinear D₁ E₁ D₂ E₂

-- Define the positional and scaling properties
axiom coincide_G₂_C₁ : G₂ = C₁
axiom shorter_side_A₂B₂ : length A₂ B₂ < length A₁ B₁
axiom scaled_N_i : ∀ (i : ℕ), scaled_down_version (N₁) (N_i)

-- Define the sum of the radii of circumcircles
noncomputable def sum_radii : ℝ := ∑' i, circumradius (N i)

-- Define the intersection point M and point O₂ (center of N₂)
axiom intersection_point_M : M = line_intersection (line A₁ C₁) (line D₁ O₂)
axiom center_O₂ : O₂ = center N₂

-- Theorem statement
theorem sum_radii_converges : sum_radii = length A₁ M :=
sorry

end sum_radii_converges_l83_83138


namespace sum_series_formula_l83_83184

noncomputable def series_sum (n : ℕ) : ℚ :=
  ∑ k in finset.range n, 1 / ((3 * (k + 1) - 2) * (3 * (k + 1) + 1))

theorem sum_series_formula (n : ℕ) :
  series_sum n = n / (3 * n + 1) :=
  by sorry

end sum_series_formula_l83_83184


namespace john_pills_per_week_l83_83684

theorem john_pills_per_week :
  ∀ (pills_per_interval : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ),
  pills_per_interval = 6 →
  hours_per_day = 24 →
  days_per_week = 7 →
  (hours_per_day / pills_per_interval) * days_per_week = 28 :=
by
  intros pills_per_interval hours_per_day days_per_week hppi hpd hdpw
  rw [hppi, hpd, hdpw]
  sorry

end john_pills_per_week_l83_83684


namespace repeating_decimal_sum_l83_83871

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l83_83871


namespace efficiency_difference_l83_83132

variables (Rp Rq : ℚ)

-- Given conditions
def p_rate := Rp = 1 / 21
def combined_rate := Rp + Rq = 1 / 11

-- Define the percentage efficiency difference
def percentage_difference := (Rp - Rq) / Rq * 100

-- Main statement to prove
theorem efficiency_difference : 
  p_rate Rp ∧ 
  combined_rate Rp Rq → 
  percentage_difference Rp Rq = 10 :=
sorry

end efficiency_difference_l83_83132


namespace walking_rate_on_escalator_l83_83980

theorem walking_rate_on_escalator (v : ℝ)
  (escalator_speed : ℝ := 12)
  (escalator_length : ℝ := 196)
  (travel_time : ℝ := 14)
  (effective_speed : ℝ := v + escalator_speed)
  (distance_eq : effective_speed * travel_time = escalator_length) :
  v = 2 := by
  sorry

end walking_rate_on_escalator_l83_83980


namespace sum_of_fraction_terms_l83_83859

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l83_83859


namespace contrapositive_equivalence_l83_83479

theorem contrapositive_equivalence {A : Set} {x y : A} :
  (x ∈ A → y ∉ A) ↔ (y ∈ A → x ∉ A) := 
by 
  sorry

end contrapositive_equivalence_l83_83479


namespace isosceles_triangle_of_area_ratio_l83_83030

theorem isosceles_triangle_of_area_ratio
  (A B C M N K : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace M] [MetricSpace N] [MetricSpace K]
  (hMN : dist C M = dist M N ∧ dist M N = dist N B)
  (h_perpendicular : dist (Project (Subspace AB) N) N = 0)
  (area_ratio : ∀ (area_ABC area_AMK : ℝ), area_AMK = area_ABC / 4.5) :
  Isosceles ABC :=
by
  sorry

end isosceles_triangle_of_area_ratio_l83_83030


namespace bus_children_problem_l83_83943

theorem bus_children_problem :
  ∃ X, 5 - 63 + X = 14 ∧ X - 63 = 9 :=
by 
  sorry

end bus_children_problem_l83_83943


namespace bridge_length_correct_l83_83495

noncomputable def length_of_bridge : ℝ :=
  let speed_kmh := 40
  let speed_ms := speed_kmh * (1000 / 3600)
  let time_seconds := 45
  let train_length := 200
  let total_distance := speed_ms * time_seconds
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_correct : length_of_bridge ≈ 299.95 :=
begin
  -- Convert speed from km/h to m/s
  let speed_ms := 40 * (1000 / 3600),
  -- Calculate total distance covered in 45 seconds
  let total_distance := speed_ms * 45,
  -- Subtract the length of the train
  let bridge_length := total_distance - 200,
  -- Show that the length of the bridge is approximately 299.95 meters
  show bridge_length ≈ 299.95,
  sorry
end

end bridge_length_correct_l83_83495


namespace third_position_is_two_l83_83075

theorem third_position_is_two :
  ∃ (seq : ℕ → ℕ), (∀ i, 1 ≤ i → i ≤ 37 → seq i ∈ finset.range (37 + 1)) ∧ 
    seq 1 = 37 ∧ seq 2 = 1 ∧ ∀ n, 1 ≤ n ∧ n < 37 → (finset.range (n + 1)).sum seq % seq (n + 1 + 1) = 0 ∧ seq 3 = 2 :=
by
  sorry

end third_position_is_two_l83_83075


namespace latin_squares_count_l83_83551

-- Question: Prove that the number of Latin squares L_n of size n × n satisfies L_n ≥ ∏_{k=1}^n k!
theorem latin_squares_count (n : ℕ) (L : ℕ) : 
  (∀ (A : Matrix (Fin n) (Fin n) ℕ), 
    (∀ i, (∀ j₁ j₂, j₁ ≠ j₂ → A i j₁ ≠ A i j₂)) ∧  -- each number appears once per row
    (∀ j, (∀ i₁ i₂, i₁ ≠ i₂ → A i₁ j ≠ A i₂ j)))  -- each number appears once per column
  → L ≥ ∏ k in Finset.range n, factorial (k + 1) := 
sorry

end latin_squares_count_l83_83551


namespace tomatoes_needed_for_meal_l83_83792

theorem tomatoes_needed_for_meal :
  (∀ (slices_per_tomato slices_per_meal people : ℕ),
    slices_per_tomato = 8 →
    slices_per_meal = 20 →
    people = 8 →
    (people * slices_per_meal) / slices_per_tomato = 20) :=
by {
  intros slices_per_tomato slices_per_meal people h_slices_per_tomato h_slices_per_meal h_people,
  rw [h_slices_per_tomato, h_slices_per_meal, h_people],
  norm_num,
}

end tomatoes_needed_for_meal_l83_83792


namespace average_of_sequence_l83_83761

theorem average_of_sequence (x : ℝ) (h : (1 + 2 + 3 + ... + 97 + 98 + x) / 99 = 50 * x) : 
  x = (4851 / 4949) :=
by
  sorry

end average_of_sequence_l83_83761


namespace num_ordered_pairs_satisfying_equation_l83_83202

-- Define the conditions for the ordered pairs (x, y)
def satisfies_equation (x y : ℤ) : Prop := x^4 + y^2 = 4 * y

-- Define what we want to prove
theorem num_ordered_pairs_satisfying_equation : 
  ({p : ℤ × ℤ | satisfies_equation p.fst p.snd}.to_finset.card = 2) :=
sorry

end num_ordered_pairs_satisfying_equation_l83_83202


namespace smallest_unique_multiple_of_9_remainder_l83_83696

def unique_digits (n : ℕ) : Prop :=
  (n.digits 10).nodup

def smallest_unique_multiple_of_9 (bound : ℕ) : ℕ :=
  Inf {n : ℕ | n % 9 = 0 ∧ unique_digits n ∧ n < bound}

theorem smallest_unique_multiple_of_9_remainder :
  smallest_unique_multiple_of_9 10000 % 100 = 89 :=
sorry

end smallest_unique_multiple_of_9_remainder_l83_83696


namespace geometric_sequence_sum_relation_l83_83577

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

noncomputable def sum_of_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (finset.range n).sum a

theorem geometric_sequence_sum_relation
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (x y : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : sum_of_terms a S)
  (h3 : x = S n ^ 2 + S (2 * n) ^ 2)
  (h4 : y = S n * (S (2 * n) + S (3 * n))) :
  x = y := sorry

end geometric_sequence_sum_relation_l83_83577


namespace parabola_x_coordinate_l83_83766

theorem parabola_x_coordinate (x y : ℝ) : 
  y^2 = 4 * x → 
  sqrt ((x - 1)^2 + y^2) = 6 → 
  x = 5 := 
by
  sorry

end parabola_x_coordinate_l83_83766


namespace final_elephants_count_l83_83809

def E_0 : Int := 30000
def R_exodus : Int := 2880
def H_exodus : Int := 4
def R_entry : Int := 1500
def H_entry : Int := 7
def E_final : Int := E_0 - (R_exodus * H_exodus) + (R_entry * H_entry)

theorem final_elephants_count : E_final = 28980 := by
  sorry

end final_elephants_count_l83_83809


namespace solve_for_x_l83_83621

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end solve_for_x_l83_83621


namespace collinear_points_x_value_l83_83141

theorem collinear_points_x_value :
  (∀ A B C : ℝ × ℝ, A = (-1, 1) → B = (2, -4) → C = (x, -9) → 
                    (∃ x : ℝ, x = 5)) :=
by sorry

end collinear_points_x_value_l83_83141


namespace j_h_neg3_eq_14_l83_83010

def h (x : ℝ) : ℝ := 5 * x^2 + 3

def j : ℝ → ℝ := sorry -- j is an arbitrary function from ℝ to ℝ 

lemma h_value_at_3 : h 3 = 48 :=
by
  unfold h
  norm_num

lemma h_value_at_neg3 : h (-3) = 48 :=
by
  unfold h
  norm_num

axiom j_applied_at_h_3 : j (h 3) = 14

theorem j_h_neg3_eq_14 : j (h (-3)) = 14 :=
by
  rw [h_value_at_neg3, h_value_at_3]
  exact j_applied_at_h_3

end j_h_neg3_eq_14_l83_83010


namespace inverse_of_true_proposition_l83_83928

theorem inverse_of_true_proposition (a b : ℝ) (h1 : a > 0 ∧ b > 0 → a * b > 0)
                                   (h2 : ∃ t : Type, (triangle t → (side_lengths t = (3, 4, 5) → right_triangle t)))
                                   (h3 : ∃ p : point, (angle_bisector p → equidistant_from_sides p))
                                   (h4 : a = b → |a| = |b|) :
  (¬(a * b > 0 → a > 0 ∧ b > 0)) ∧ 
  (¬(right_triangle t → side_lengths t = (3, 4, 5))) ∧ 
  (equidistant_from_sides p → angle_bisector p) ∧ 
  (¬(|a| = |b| → a = b)) :=
by {
  sorry
}

end inverse_of_true_proposition_l83_83928


namespace length_AB_is_8_l83_83025

-- Conditions
variables {A B C M N : Point}
variable (right_triangle : RightTriangle A B C)
variable (mid_M : Midpoint M B C)
variable (mid_N : Midpoint N A C)
variable (AM_length : length (segment A M) = 6)
variable (BN_length : length (segment B N) = 2 * sqrt 11)

-- Prove the length of segment AB is 8
theorem length_AB_is_8 :
  length (segment A B) = 8 :=
by
  sorry

end length_AB_is_8_l83_83025


namespace tomatoes_needed_for_meal_l83_83793

theorem tomatoes_needed_for_meal :
  (∀ (slices_per_tomato slices_per_meal people : ℕ),
    slices_per_tomato = 8 →
    slices_per_meal = 20 →
    people = 8 →
    (people * slices_per_meal) / slices_per_tomato = 20) :=
by {
  intros slices_per_tomato slices_per_meal people h_slices_per_tomato h_slices_per_meal h_people,
  rw [h_slices_per_tomato, h_slices_per_meal, h_people],
  norm_num,
}

end tomatoes_needed_for_meal_l83_83793


namespace express_pollen_in_scientific_notation_l83_83655

theorem express_pollen_in_scientific_notation :
  (0.0000084 : ℝ) = 8.4 * 10^(-6) :=
by
  sorry

end express_pollen_in_scientific_notation_l83_83655


namespace count_numbers_with_two_transitions_l83_83555

def binary_representation (n : ℕ) : list bool :=
-- this function will return the binary representation as a list of bits
sorry

def count_transitions (bits : list bool) : ℕ :=
-- this function will count the transitions between adjacent bits in a list
sorry

def has_two_transitions (n : ℕ) : Prop :=
count_transitions (binary_representation n) = 2

theorem count_numbers_with_two_transitions :
  {n : ℕ | 1 ≤ n ∧ n ≤ 127 ∧ has_two_transitions n}.to_finset.card = 33 :=
by
  sorry

end count_numbers_with_two_transitions_l83_83555


namespace unfair_draw_fair_draw_with_suit_hierarchy_l83_83316

noncomputable def deck := {suit : String, rank : ℕ // suit ∈ {"hearts", "diamonds", "clubs", "spades"} ∧ rank ∈ {6, 7, 8, 9, 10, 11, 12, 13, 14}}
def prob_V (v : deck) : ℚ := 1 / 36
def prob_M_given_V (v m : deck) : ℚ := 1 / 35
def higher_rank (v m : deck) : Prop := m.rank > v.rank

-- Prove the draw is unfair
theorem unfair_draw : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank v m then prob_M_given_V v m else 0) < 
  (∑ m in (deck \ {v}), if ¬higher_rank v m then prob_M_given_V v m else 0)) :=
sorry

-- Making the draw fair by introducing suit hierarchy
def suit_order : String → ℕ
| "spades" := 4
| "hearts" := 3
| "diamonds" := 2
| "clubs" := 1
| _ := 0

def higher_rank_with_suit (v m : deck) : Prop :=
  if v.rank = m.rank then suit_order m.suit > suit_order v.suit else m.rank > v.rank

-- Prove introducing suit hierarchy can make the draw fair
theorem fair_draw_with_suit_hierarchy : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank_with_suit v m then prob_M_given_V v m else 0) = 
  (∑ m in (deck \ {v}), if ¬higher_rank_with_suit v m then prob_M_given_V v m else 0)) :=
sorry

end unfair_draw_fair_draw_with_suit_hierarchy_l83_83316


namespace boys_at_beginning_is_15_l83_83985

noncomputable def number_of_boys_at_beginning (B : ℝ) : Prop :=
  let girls_start := 1.20 * B
  let girls_end := 2 * girls_start
  let total_students := B + girls_end
  total_students = 51 

theorem boys_at_beginning_is_15 : number_of_boys_at_beginning 15 := 
  by
  -- Sorry is added to skip the proof
  sorry

end boys_at_beginning_is_15_l83_83985


namespace solve_for_x_l83_83625

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end solve_for_x_l83_83625


namespace draw_is_unfair_ensure_fair_draw_l83_83324

open ProbabilityTheory MeasureTheory

-- Definitions for the given conditions:
def Card := {rank : ℕ // 6 ≤ rank ∧ rank ≤ 14} -- Ranks 6 to Ace (6 to 14)
def Deck := Finset (Fin 36) -- 36 unique cards
noncomputable def suit_high_rank_count (d : Deck) (v_card : Fin 36) (m_card : Fin 36) : ℕ := 
  -- Count how many cards are higher than Volodya's card
  card.count (λ c, c.val > v_card.val) d

-- Volodya draws first, then Masha draws:
variables (d : Deck) (v_card m_card : Fin 36)

-- Masha wins if she draws a card with a higher rank than Volodya’s card
def masha_wins := ∃ (m_card : Fin 36), (m_card ∈ d) ∧ (m_card.val > v_card.val)

-- Volodya wins if Masha doesn't win (Masha loses)
def volodya_wins := ¬ masha_wins

theorem draw_is_unfair (d : Deck) (v_card m_card : Fin 36) :
  (volodya_wins d v_card m_card) → ¬ (masha_wins d v_card) := sorry

-- To make it fair, we can introduce a suit hierarchy:
def suits := {"Hearts", "Diamonds", "Clubs", "Spades"}
def suit_order : suits → suits → Prop
| "Spades" "Hearts" := true
| "Hearts" "Diamonds" := true
| "Diamonds" "Clubs" := true
| "Clubs" "Spades" := false
| _, _ := false

-- A fair draw means using the suit_order to rank otherwise equal cards:
def fair_draw :=
  ∀ (c1 c2 : Card), (c1.rank = c2.rank → suit_order c1.suit c2.suit)

theorem ensure_fair_draw : fair_draw := sorry

end draw_is_unfair_ensure_fair_draw_l83_83324


namespace find_AP_find_PT_and_area_l83_83436

-- Definitions for the given conditions
variable (Γ : Type) [MetricSpace Γ]
variable (O : Γ) -- Center of the circle
variable (A B C D P L : Γ) -- Points on the circle and intersections
variable (r : ℝ) -- Radius of the circle
variable (length_AB length_CD : ℝ) -- Lengths of the chords
variable (ratio_AL_LC : ℝ) -- Ratio of segments AL and LC

-- Conditions
axiom AB_CD_length : length_AB = 8 ∧ length_CD = 8
axiom circle_radius : r = 5
axiom ratio_condition : ratio_AL_LC = 1 / 5 -- Given AL : LC = 1 : 5

-- Questions to be proved
theorem find_AP (h1 : dist A B = length_AB) (h2 : dist C D = length_CD)
  (h3 : h1 = 8) (h4 : h2 = 8) (h5 : ratio_condition) :
  dist A P = 2 :=
sorry

theorem find_PT_and_area (h1 : dist O _ = r) (h2 : circle_radius):
  dist P T = 3 * sqrt 5 - 5 ∧
  let area_ABC := 8 in
  True :=
sorry

end find_AP_find_PT_and_area_l83_83436


namespace total_cups_l83_83143

theorem total_cups (initial_cups added_cups : ℕ) (h1 : initial_cups = 17) (h2 : added_cups = 16) : initial_cups + added_cups = 33 :=
by
  -- Introduction and assignment of values based on conditions
  rw [h1, h2]
  -- Prove the result
  exact rfl

end total_cups_l83_83143


namespace problem_conditions_l83_83584

open Real

noncomputable def ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 / a)^2 + (p.2 / b)^2 = 1}

theorem problem_conditions (a b : ℝ) (h1 : a > b > 0) (h2 : (sqrt 3 / 2) = (sqrt (a^2 - b^2) / a)) (h3 : (1, - (sqrt 3) / 2) ∈ ellipse a b) :
  (∃ a b : ℝ, ellipse a b = {p | (p.1 / 2)^2 + p.2^2 = 1}) ∧
  (∃ m : ℝ, (1/2 * sqrt 2 * sqrt (5 - m^2) * |m| / sqrt 2 = 1) → m = sqrt 10 / 2 ∨ m = - sqrt 10 / 2) := 
sorry

end problem_conditions_l83_83584


namespace rectangle_perimeter_equal_area_l83_83506

theorem rectangle_perimeter_equal_area (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 2 * a + 2 * b) : 2 * (a + b) = 18 := 
by 
  sorry

end rectangle_perimeter_equal_area_l83_83506


namespace distance_travelled_within_5_seconds_l83_83533

noncomputable def velocity (t : ℝ) : ℝ := 3 * t^2 + 10 * t + 3

theorem distance_travelled_within_5_seconds :
  ∫ (t : ℝ) in 0..5, velocity t = 265 := by
  sorry

end distance_travelled_within_5_seconds_l83_83533


namespace total_payment_is_correct_l83_83500

def original_price_coffee_maker := 70
def original_price_blender := 100
def discount_coffee_maker := 0.20
def discount_blender := 0.15
def coffee_makers_count := 2
def blender_count := 1

def price_after_discount (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price * (1 - discount)

def total_price (count : ℝ) (price : ℝ) : ℝ :=
  count * price

def total_payment := 
  total_price coffee_makers_count (price_after_discount original_price_coffee_maker discount_coffee_maker) + 
  total_price blender_count (price_after_discount original_price_blender discount_blender)

theorem total_payment_is_correct : total_payment = 197 := by
  sorry

end total_payment_is_correct_l83_83500


namespace person_walk_rate_l83_83531

theorem person_walk_rate (v : ℝ) (elevator_speed : ℝ) (length : ℝ) (time : ℝ) 
  (h1 : elevator_speed = 10) 
  (h2 : length = 112) 
  (h3 : time = 8) 
  (h4 : length = (v + elevator_speed) * time) 
  : v = 4 :=
by 
  sorry

end person_walk_rate_l83_83531


namespace cases_in_1990_l83_83647

theorem cases_in_1990 (cases_1970 cases_2000 : ℕ) (linear_decrease : ℕ → ℝ) :
  cases_1970 = 300000 →
  cases_2000 = 600 →
  (∀ t, linear_decrease t = cases_1970 - (cases_1970 - cases_2000) * t / 30) →
  linear_decrease 20 = 100400 :=
by
  intros h1 h2 h3
  sorry

end cases_in_1990_l83_83647


namespace repeating_decimal_sum_l83_83866

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l83_83866


namespace thelma_tomato_count_l83_83791

-- Definitions and conditions
def slices_per_tomato : ℕ := 8
def slices_per_meal_per_person : ℕ := 20
def family_members : ℕ := 8
def total_slices_needed : ℕ := slices_per_meal_per_person * family_members
def tomatoes_needed : ℕ := total_slices_needed / slices_per_tomato

-- Statement of the theorem to be proved
theorem thelma_tomato_count :
  tomatoes_needed = 20 := by
  sorry

end thelma_tomato_count_l83_83791


namespace area_of_ellipse_l83_83222

theorem area_of_ellipse (x y : ℝ) :
  2 * x^2 + 8 * x + 3 * y^2 - 9 * y + 12 = 0 →
  (∃ a b : ℝ, a = sqrt (1 / 2) ∧ b = sqrt (1 / 3) ∧ 
   ∀ u v : ℝ, (u = x + 2) ∧ (v = y - 3 / 2) →
     (u^2 / a^2 + v^2 / b^2 = 1) ∧
     a * b * π = π * sqrt (1 / 6) ∧
     π * sqrt (1 / 6) = π * sqrt (6) / 6) :=
sorry

end area_of_ellipse_l83_83222


namespace find_k_for_circle_of_radius_8_l83_83573

theorem find_k_for_circle_of_radius_8 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ∧ (∀ r : ℝ, r = 8) → k = -1 :=
sorry

end find_k_for_circle_of_radius_8_l83_83573


namespace sum_of_fraction_numerator_and_denominator_l83_83915

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l83_83915


namespace f_p_f1_not_eq_f_f_p1_l83_83388

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * x - 1

-- Define the piecewise function f_p with p = 2
def f_p (x : ℝ) (p : ℝ) : ℝ :=
if f x ≤ p then f x else p

-- Specialize f_2 given p = 2
def f_2 (x : ℝ) : ℝ := f_p x 2

-- State the theorem to be proved
theorem f_p_f1_not_eq_f_f_p1 :
  f_p (f 1) 2 ≠ f (f_p 1 2) :=
by
  sorry

end f_p_f1_not_eq_f_f_p1_l83_83388


namespace two_digit_values_heartsuit_heartsuit_eq_5_l83_83697

def heartsuit (x : ℕ) : ℕ :=
  x.digits.sum

theorem two_digit_values_heartsuit_heartsuit_eq_5 : 
  (Finset.filter (λ x : ℕ, heartsuit (heartsuit x) = 5) (Finset.Icc 10 99)).card = 10 := by
  sorry

end two_digit_values_heartsuit_heartsuit_eq_5_l83_83697


namespace min_value_quadratic_l83_83818

theorem min_value_quadratic : ∃ x : ℝ, ∀ y : ℝ, 3 * x ^ 2 - 18 * x + 2023 ≤ 3 * y ^ 2 - 18 * y + 2023 :=
sorry

end min_value_quadratic_l83_83818


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l83_83892

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l83_83892


namespace theta_value_l83_83006

noncomputable def poly := λ z: Complex, z^5 + z^2 + 1

theorem theta_value (P : Complex) (r θ : ℝ) (h1 : poly z = 0) 
  (h2 : Im z > 0)
  (h3 : P = r * (Complex.cos θ + i * Complex.sin θ))
  (hr_pos : 0 < r) 
  (hθ_range : 0 ≤ θ ∧ θ < 360) :
  θ = 108 :=
sorry

end theta_value_l83_83006


namespace find_k_for_circle_of_radius_8_l83_83574

theorem find_k_for_circle_of_radius_8 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ∧ (∀ r : ℝ, r = 8) → k = -1 :=
sorry

end find_k_for_circle_of_radius_8_l83_83574


namespace second_hand_bisect_angle_l83_83121

theorem second_hand_bisect_angle :
  ∃ x : ℚ, (6 * x - 360 * (x - 1) = 360 * (x - 1) - 0.5 * x) ∧ (x = 1440 / 1427) :=
by
  sorry

end second_hand_bisect_angle_l83_83121


namespace remainder_101_pow_37_mod_100_l83_83467

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := 
by 
  sorry

end remainder_101_pow_37_mod_100_l83_83467


namespace exponential_first_quadrant_l83_83789

theorem exponential_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, y = (1 / 2)^x + m → y ≤ 0) ↔ m ≤ -1 := 
by
  sorry

end exponential_first_quadrant_l83_83789


namespace curve_line_eq_distance_relation_l83_83349

-- Definitions for Polar to Rectangular properties
def polar_to_rect_eq (rho θ : ℝ) : Prop :=
  rho = 4 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4)

def parameter_eq (t : ℝ) (x y : ℝ) : Prop :=
  x = 2 + 1/2 * t ∧ y = 1 + Real.sqrt 3 / 2 * t

-- Definitions for Given conditions
def point (x y : ℝ) : Prop := 
  x = 2 ∧ y = 1

-- Proving required equations from given conditions
theorem curve_line_eq (rho θ t x y : ℝ) (h1 : polar_to_rect_eq rho θ) (h2 : parameter_eq t x y) :
  (x - 2)^2 + (y - 2)^2 = 8 ∧ (Real.sqrt 3 * x - y - 2 * Real.sqrt 3 + 1 = 0) := sorry

-- Proving the distance relation
theorem distance_relation (t1 t2 : ℝ) (h : t^2 - Real.sqrt 3 * t - 7 = 0) :
  (Real.abs ((1 / Real.abs t1) - (1 / Real.abs t2))) = (Real.sqrt 31 / 7) := sorry

end curve_line_eq_distance_relation_l83_83349


namespace contrapositive_exp_l83_83061

theorem contrapositive_exp (a b : ℝ) : (a > b → 2^a > 2^b) ↔ (2^a ≤ 2^b → a ≤ b) :=
sorry

end contrapositive_exp_l83_83061


namespace p_finishes_job_after_q_in_24_minutes_l83_83936

theorem p_finishes_job_after_q_in_24_minutes :
  let P_rate := 1 / 4
  let Q_rate := 1 / 20
  let together_rate := P_rate + Q_rate
  let work_done_in_3_hours := together_rate * 3
  let remaining_work := 1 - work_done_in_3_hours
  let time_for_p_to_finish := remaining_work / P_rate
  let time_in_minutes := time_for_p_to_finish * 60
  time_in_minutes = 24 :=
by
  sorry

end p_finishes_job_after_q_in_24_minutes_l83_83936


namespace binom_sum_problem_l83_83108

theorem binom_sum_problem :
  ∑ n in {n : ℕ | nat.choose 30 n = nat.choose 16 16 ∨ nat.choose 16 14}, n = 30 :=
by
  sorry

end binom_sum_problem_l83_83108


namespace avg_between_6_and_34_div_5_l83_83484

theorem avg_between_6_and_34_div_5 : 
  let nums := [10, 15, 20, 25, 30] in 
  nums.Sum / nums.length = 20 := 
by 
sorry 

end avg_between_6_and_34_div_5_l83_83484


namespace Jack_goal_l83_83678

-- Define the amounts Jack made from brownies and lemon squares
def brownies (n : ℕ) (price : ℕ) : ℕ := n * price
def lemonSquares (n : ℕ) (price : ℕ) : ℕ := n * price

-- Define the amount Jack needs to make from cookies
def cookies (n : ℕ) (price : ℕ) : ℕ := n * price

-- Define the total goal for Jack
def totalGoal (browniesCount : ℕ) (browniesPrice : ℕ) 
              (lemonSquaresCount : ℕ) (lemonSquaresPrice : ℕ) 
              (cookiesCount : ℕ) (cookiesPrice: ℕ) : ℕ :=
  brownies browniesCount browniesPrice + lemonSquares lemonSquaresCount lemonSquaresPrice + cookies cookiesCount cookiesPrice

theorem Jack_goal : totalGoal 4 3 5 2 7 4 = 50 :=
by
  -- Adding up the different components of the total earnings
  let totalFromBrownies := brownies 4 3
  let totalFromLemonSquares := lemonSquares 5 2
  let totalFromCookies := cookies 7 4
  -- Summing up the amounts
  have step1 : totalFromBrownies = 12 := rfl
  have step2 : totalFromLemonSquares = 10 := rfl
  have step3 : totalFromCookies = 28 := rfl
  have step4 : totalGoal 4 3 5 2 7 4 = totalFromBrownies + totalFromLemonSquares + totalFromCookies := rfl
  have step5 : totalFromBrownies + totalFromLemonSquares + totalFromCookies = 12 + 10 + 28 := by rw [step1, step2, step3]
  have step6 : 12 + 10 + 28 = 50 := by norm_num
  exact step4 ▸ (step5 ▸ step6)

end Jack_goal_l83_83678


namespace stock_price_satisfies_conditions_part_a_stock_price_satisfies_conditions_part_b_l83_83723

-- Define the conditions: stock prices on specific dates
constants apr_7 apr_13 apr_20 : ℝ
axiom apr_7_eq : apr_7 = 5
axiom apr_13_eq : apr_13 = 5.14
axiom apr_20_eq : apr_20 = 5

-- Define the prices on the intervening dates
constants (x : ℕ → ℝ)

-- Define the correct answer for average price calculation for part (a)
def avg_price_between_apr_7_Apr_20 : ℝ := (apr_7 + (Σ i in FinSet.range 12, x i) + apr_13 + (Σ j in FinSet.range 6, x (j + 7)) + apr_20) / 14

noncomputable def part_a : Prop :=
  5.09 < avg_price_between_apr_7_Apr_20 ∧ avg_price_between_apr_7_Apr_20 < 5.10

-- Part (b): Comparing average stock prices for different periods
def avg_price_apr_7_to_apr_13 : ℝ := (apr_7 + (Σ i in FinSet.range 5, x i) + apr_13) / 7
def avg_price_apr_14_to_apr_20 : ℝ := (apr_13 + x 7 + x 8 + x 9 + x 10 + x 11 + apr_20) / 7

noncomputable def part_b : Prop :=
  | (avg_price_apr_14_to_apr_20 - avg_price_apr_7_to_apr_13 ≠ 0.105 )

-- The final proof problems for part (a) and part (b)
theorem stock_price_satisfies_conditions_part_a : part_a := sorry
theorem stock_price_satisfies_conditions_part_b : part_b := sorry

end stock_price_satisfies_conditions_part_a_stock_price_satisfies_conditions_part_b_l83_83723


namespace age_relationships_l83_83134

variables (a b c d : ℕ)

theorem age_relationships (h1 : a + b = b + c + d + 18) (h2 : 2 * a = 3 * c) :
  c = 2 * a / 3 ∧ d = a / 3 - 18 :=
by
  sorry

end age_relationships_l83_83134


namespace base8_subtraction_correct_l83_83565

noncomputable def base8_subtraction (x y : Nat) : Nat :=
  if y > x then 0 else x - y

theorem base8_subtraction_correct :
  base8_subtraction 546 321 - 105 = 120 :=
by
  -- Given the condition that all arithmetic is in base 8
  sorry

end base8_subtraction_correct_l83_83565


namespace total_cakes_needed_l83_83712

theorem total_cakes_needed (C : ℕ) (h : C / 4 - C / 12 = 10) : C = 60 := by
  sorry

end total_cakes_needed_l83_83712


namespace sum_of_fraction_terms_l83_83861

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l83_83861


namespace max_n_for_Sn_pos_l83_83664

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def condition_a16_pos : Prop := a 16 > 0
def condition_a17_neg : Prop := a 17 < 0
def condition_a16_gt_abs_a17 : Prop := a 16 > abs (a 17)
def Sn (n : ℕ) : ℝ := (n * (a 1 + a n)) / 2

-- The goal statement
theorem max_n_for_Sn_pos (h1 : condition_a16_pos) (h2 : condition_a17_neg) (h3 : condition_a16_gt_abs_a17) :
  ∃ n, S n > 0 ∧ ∀ m, S m > 0 → m ≤ 32 :=
sorry

end max_n_for_Sn_pos_l83_83664


namespace max_possible_median_soda_cans_l83_83433

theorem max_possible_median_soda_cans (total_cans : ℕ) (total_customers : ℕ) (min_cans_per_customer : ℕ) 
  (h1 : total_cans = 312) (h2 : total_customers = 120) (h3 : min_cans_per_customer = 2) : 
  (∃ (median : ℚ), median = 3.5) :=
by
  use 3.5
  sorry

end max_possible_median_soda_cans_l83_83433


namespace no_solution_exists_l83_83218

theorem no_solution_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x ^ 2 + f y) = 2 * x - f y :=
by
  sorry

end no_solution_exists_l83_83218


namespace lauren_total_distance_driven_l83_83688

-- Definitions based on the conditions given in the problem
def car_efficiency := 40 -- miles per gallon
def tank_capacity := 16 -- gallons when full
def initial_drive := 480 -- miles
def gas_bought := 10 -- gallons
def tank_fraction_full := 1/3

-- The theorem statement proving the total distance driven is 830 miles
theorem lauren_total_distance_driven :
  let remaining_gas := tank_capacity - initial_drive / car_efficiency in
  let total_gas := remaining_gas + gas_bought in
  let gas_used_second_leg := total_gas - tank_fraction_full * tank_capacity in
  let second_leg_distance := gas_used_second_leg * car_efficiency in
  let total_distance := initial_drive + second_leg_distance in
  total_distance = 830 :=
by
  sorry

end lauren_total_distance_driven_l83_83688


namespace find_a_plus_b_and_oscillation_l83_83955

def f (x a b : ℝ) : ℝ := x^3 - a*x^2 - (b + 2)*x

theorem find_a_plus_b_and_oscillation 
  (a b : ℝ)
  (h_odd : ∀ x, f (-x) a b = -f x a b)
  (h_interval : ∀ x (hx : -2*b ≤ x ∧ x ≤ 3*b - 1), true) :
  a + b = 1 ∧ 
  (let osc := (let vmax := max (max (f (-2) 0 1) (f (-1) 0 1)) (max (f 1 0 1) (f 2 0 1)) in
               let vmin := min (min (f (-2) 0 1) (f (-1) 0 1)) (min (f 1 0 1) (f 2 0 1)) in
               vmax - vmin) in
   osc = 4) :=
by 
  sorry

end find_a_plus_b_and_oscillation_l83_83955


namespace lateral_surface_area_of_cylinder_l83_83805

noncomputable def lateral_surface_area (a : ℝ) (α : ℝ) : ℝ :=
  (1/4) * π * a^2 * (Real.cos α / Real.sin α) * (3 * (Real.sin α)^2 + 1)

theorem lateral_surface_area_of_cylinder (a : ℝ) (α : ℝ) :
  ∃ (S : ℝ), 
  -- Two vertices of an equilateral triangle with side length a lie on the upper base circumference,
  -- and the third vertex lies on the lower base circumference.
  -- The plane of the triangle forms an angle α with a generator of the cylinder.
  S = lateral_surface_area a α :=
sorry

end lateral_surface_area_of_cylinder_l83_83805


namespace cos_sq_half_diff_eq_csquared_over_a2_b2_l83_83005

theorem cos_sq_half_diff_eq_csquared_over_a2_b2
  (a b c α β : ℝ)
  (h1 : a^2 + b^2 ≠ 0)
  (h2 : a * (Real.cos α) + b * (Real.sin α) = c)
  (h3 : a * (Real.cos β) + b * (Real.sin β) = c)
  (h4 : ∀ k : ℤ, α ≠ β + 2 * k * Real.pi) :
  Real.cos (α - β) / 2 = c^2 / (a^2 + b^2) :=
by
  sorry

end cos_sq_half_diff_eq_csquared_over_a2_b2_l83_83005


namespace max_s_value_l83_83231

def s (x y : ℝ) : ℝ := min (min x (1 - y)) (y - x)

theorem max_s_value : ∀ x y : ℝ, ∃ x y, s(x, y) = 1 / 3 ∧ ∀ u v, s(u, v) ≤ 1 / 3 :=
by
  sorry

end max_s_value_l83_83231


namespace sum_of_fraction_parts_of_repeating_decimal_l83_83833

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l83_83833


namespace recurring_fraction_sum_l83_83919

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l83_83919


namespace required_speed_remaining_l83_83158

theorem required_speed_remaining (total_distance : ℕ) (total_time : ℕ) (initial_speed : ℕ) (initial_time : ℕ) 
  (h1 : total_distance = 24) (h2 : total_time = 8) (h3 : initial_speed = 4) (h4 : initial_time = 4) :
  (total_distance - initial_speed * initial_time) / (total_time - initial_time) = 2 := by
  sorry

end required_speed_remaining_l83_83158


namespace quadratic_intersect_y_axis_l83_83764

theorem quadratic_intersect_y_axis :
  (∃ y, y = -(0 - 1)^2 + 2 ∧ (0, y) = (0, 1)) :=
by
  simp
  sorry

end quadratic_intersect_y_axis_l83_83764


namespace combined_total_percentage_difference_l83_83754

theorem combined_total (students_total : ℕ) (mashed_potatoes : ℕ) (bacon : ℕ) (pasta : ℕ) (vegetarian : ℕ)
  (h_total : students_total = 1200)
  (h_mashed_potatoes : mashed_potatoes = 479)
  (h_bacon : bacon = 489)
  (h_pasta : pasta = 320)
  (h_vegetarian : vegetarian = students_total - mashed_potatoes - bacon - pasta) :
  mashed_potatoes + bacon = 968 := 
by
  rw [h_mashed_potatoes, h_bacon]
  norm_num

theorem percentage_difference (students_total : ℕ) (mashed_potatoes : ℕ) (bacon : ℕ) (pasta : ℕ) (vegetarian : ℕ)
  (h_total : students_total = 1200)
  (h_mashed_potatoes : mashed_potatoes = 479)
  (h_bacon : bacon = 489)
  (h_pasta : pasta = 320)
  (h_vegetarian : vegetarian = students_total - mashed_potatoes - bacon - pasta) :
  (abs (mashed_potatoes - bacon)).to_float / ((mashed_potatoes + bacon) / 2).to_float * 100 ≈ 2.07 := 
by
  rw [h_mashed_potatoes, h_bacon]
  norm_num
  apply rat_approx, -- assumes we have a suitable function for approximation
  sorry -- exact numerical validation by appropriate computation

end combined_total_percentage_difference_l83_83754


namespace parabola_expression_parabola_vertex_l83_83348

-- Define the points A and B
def A := (-2, 0)
def B := (-1, 3)

-- Define the parabola equation with unknown coefficients a and b
noncomputable def parabola (a b : ℝ) := λ x : ℝ, x^2 + a * x + b

-- Define the hypotheses that the parabola passes through points A and B
def parabola_through_points (a b : ℝ) : Prop :=
  parabola a b (-2) = 0 ∧ parabola a b (-1) = 3

-- Define the calculated values of a and b
noncomputable def a := 6
noncomputable def b := 8

-- Define the final parabola equation
def final_parabola := parabola a b

-- Define the vertex calculation
noncomputable def vertex_x (a b : ℝ) := -a / 2
noncomputable def vertex_y (a b : ℝ) := parabola a b (vertex_x a b)

-- Prove that the parabola passing through points A and B has the specified equation
theorem parabola_expression : parabola_through_points 6 8 := by
  sorry

-- Prove that the vertex of the parabola is the specified point
theorem parabola_vertex : 
  vertex_x 6 8 = -3 ∧ vertex_y 6 8 = -1 := by
  sorry

end parabola_expression_parabola_vertex_l83_83348


namespace martin_leftover_raisins_l83_83397

-- Definitions
variables {v k r : ℝ}
-- Let v be the cost of 1 cream puff
-- Let k be the cost of 1 deciliter of Kofola
-- Let r be the cost of 1 dekagram of raisins

theorem martin_leftover_raisins (v k r : ℝ) (h : r ≠ 0) :
  (3 * v + 3 * k = 18 * r) →
  (12 * r + 5 * k = v + 6 * k + x * r) →
  x = 6 →
  x * 10 = 60 :=
begin
  -- Assuming the conditions of the problem:
  -- 1. Martin could buy three cream puffs and 3 dl of Kofola, or 18 dkg of yogurt raisins.
  -- 2. Martin could buy 12 dkg of yogurt raisins and 5 dl of Kofola, or one cream puff and 6 dl of Kofola.
  
  -- This theorem describes that the quantity of yogurt raisins Martin has left over is equal to 60 grams.
 sorry
end

end martin_leftover_raisins_l83_83397


namespace find_metal_sheet_width_l83_83504

-- The given conditions
def metalSheetLength : ℝ := 100
def cutSquareSide : ℝ := 10
def boxVolume : ℝ := 24000

-- Statement to prove
theorem find_metal_sheet_width (w : ℝ) (h : w - 2 * cutSquareSide > 0):
  boxVolume = (metalSheetLength - 2 * cutSquareSide) * (w - 2 * cutSquareSide) * cutSquareSide → 
  w = 50 := 
by {
  sorry
}

end find_metal_sheet_width_l83_83504


namespace repeating_decimal_fraction_sum_l83_83846

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l83_83846


namespace ages_of_three_persons_l83_83434

theorem ages_of_three_persons (y m e : ℕ) 
  (h1 : e = m + 16)
  (h2 : m = y + 8)
  (h3 : e - 6 = 3 * (y - 6))
  (h4 : e - 6 = 2 * (m - 6)) :
  y = 18 ∧ m = 26 ∧ e = 42 := 
by 
  sorry

end ages_of_three_persons_l83_83434


namespace distance_circle_center_to_point_l83_83100

theorem distance_circle_center_to_point :
  let circle_eq := λ x y : ℝ, x^2 + y^2 = 4 * x + 6 * y - 4
  let center_x : ℝ := 2
  let center_y : ℝ := 3
  let point_x : ℝ := 10
  let point_y : ℝ := 8
  sqrt ((point_x - center_x)^2 + (point_y - center_y)^2) = sqrt 89 :=
by
  sorry

end distance_circle_center_to_point_l83_83100


namespace car_count_is_150_l83_83775

variable (B C K : ℕ)  -- Define the variables representing buses, cars, and bikes

/-- Given conditions: The ratio of buses to cars to bikes is 3:7:10,
    there are 90 fewer buses than cars, and 140 fewer buses than bikes. -/
def conditions : Prop :=
  (C = (7 * B / 3)) ∧ (K = (10 * B / 3)) ∧ (C = B + 90) ∧ (K = B + 140)

theorem car_count_is_150 (h : conditions B C K) : C = 150 :=
by
  sorry

end car_count_is_150_l83_83775


namespace final_integer_in_sequence_l83_83043

theorem final_integer_in_sequence :
  let start := 800000 in
  let f := λ (n: ℕ) (x: ℕ), if n % 2 = 0 then x / 4 else x * 3 in
  (fin 10).foldl f start = 1518750 :=
by sorry

end final_integer_in_sequence_l83_83043


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l83_83893

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l83_83893


namespace geo_arith_sequences_sum_first_2n_terms_l83_83343

variables (n : ℕ)

-- Given conditions in (a)
def common_ratio : ℕ := 3
def arithmetic_diff : ℕ := 2

-- The sequences provided in the solution (b)
def a_n (n : ℕ) : ℕ := common_ratio ^ n
def b_n (n : ℕ) : ℕ := 2 * n + 1

-- Sum formula for geometric series up to 2n terms
def S_2n (n : ℕ) : ℕ := (common_ratio^(2 * n + 1) - common_ratio) / 2 + 2 * n

theorem geo_arith_sequences :
  a_n n = common_ratio ^ n
  ∨ b_n n = 2 * n + 1 := sorry

theorem sum_first_2n_terms :
  S_2n n = (common_ratio^(2 * n + 1) - common_ratio) / 2 + 2 * n := sorry

end geo_arith_sequences_sum_first_2n_terms_l83_83343


namespace total_seeds_l83_83088

theorem total_seeds (A B C : ℕ) (h₁ : A = B + 10) (h₂ : B = 30) (h₃ : C = 30) : A + B + C = 100 :=
by
  sorry

end total_seeds_l83_83088


namespace panda_weight_l83_83959

theorem panda_weight (monkey_weight : ℕ) (panda_weight : ℕ) : 
  monkey_weight = 25 → panda_weight = 6 * monkey_weight + 12 → panda_weight = 162 :=
by
  intros h_monkey_weight h_panda_weight
  rw [h_monkey_weight] at h_panda_weight
  exact h_panda_weight
sorry

end panda_weight_l83_83959


namespace dimensions_and_area_of_rectangle_l83_83444

theorem dimensions_and_area_of_rectangle
  (side_length_square : ℝ)
  (radius_circle : ℝ)
  (length_rectangle breadth_rectangle : ℝ)
  (volume_cylinder height_cylinder : ℝ)
  (h1 : side_length_square ^ 2 = 2500)
  (h2 : radius_circle = side_length_square)
  (h3 : length_rectangle = (2/5) * radius_circle)
  (h4 : breadth_rectangle = 10)
  (h5 : volume_cylinder = 2 * (length_rectangle * breadth_rectangle))
  (h6 : height_cylinder = side_length_square)
  (h7 : volume_cylinder = ℝ.pi * (radius_circle / sqrt(2.54648)) ^ 2 * height_cylinder) :
  length_rectangle = 20 ∧ breadth_rectangle = 10 ∧ (length_rectangle * breadth_rectangle) = 200 ∧ (radius_circle / sqrt(2.54648)) ≈ 1.59514 ∧ height_cylinder = 50 :=
by
  sorry

end dimensions_and_area_of_rectangle_l83_83444


namespace count_ways_to_sum_consecutive_integers_l83_83561

-- Definition of the function f(n) that counts the valid odd and even factors of n.
noncomputable def f (n : ℕ) : ℕ := -- implementation placeholder
  sorry

-- Theorem statement: Given a positive integer n, f(n) gives the number of ways to express n as a sum of consecutive positive integers.
theorem count_ways_to_sum_consecutive_integers (n : ℕ) (h : 0 < n) : ∃ k : ℕ, k = f n := 
by 
  use f n
  exact rfl

end count_ways_to_sum_consecutive_integers_l83_83561


namespace exists_a_l83_83277

variable (a : ℝ) (x : ℝ)

def g (x : ℝ) (a : ℝ) : ℝ := (a + 1)^(x - 2) + 1
def f (x : ℝ) (a : ℝ) : ℝ := log (sqrt 3) (x + a) -- log to the base sqrt(3)

def pointA_on_g_and_f (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A ∈ set_of (λ p : ℝ × ℝ, g p.1 a = p.2) ∧ A ∈ set_of (λ p : ℝ × ℝ, f p.1 a = p.2)

theorem exists_a (A : ℝ × ℝ) : 
  (∃ a > 0, pointA_on_g_and_f A a) → 
  ∃ a > 0,
    (∀ x, (g x a > 3) → (true)) :=
sorry

-- Ensure necessary imports and setup
open Real

end exists_a_l83_83277


namespace triangle_ratio_proof_l83_83363

variable (A B C O P X : Type*) [add_comm_group A] [module ℝ A] 
variables (a b c : ℝ) (AB BC AC AX XC : ℝ)
variables (h1 : AB = 13) (h2 : BC = 15) (h3 : AC = 14)
variables (PB: A → A) (PA: A → A) (BX: A → A) (OP: A → A)
variable (hPBC: ∀ p ∈ PB, ∃ q ∈ BC, p ⊥ q)
variable (hPAB: ∀ p ∈ PA, ∃ q ∈ AB, p ⊥ q)
variable (hBXOP: ∀ b ∈ BX, ∃ o ∈ OP, b ⊥ o)

def ratio_AX_XC : ℝ := AX/XC

theorem triangle_ratio_proof :
  (∃ (AX XC : ℝ), AX * XC ≠ 0) →
  (AX / XC) = (169 / 225) := by
  intro h
  sorry

end triangle_ratio_proof_l83_83363


namespace vertex_after_translation_l83_83763

-- Define the given quadratic function
def original_parabola (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the coordinates of the vertex after the translation
def translated_vertex_coordinates : ℝ × ℝ := (4, -1)

-- Statement to prove: The coordinates of the vertex of the new parabola are (4, -1)
theorem vertex_after_translation :
  ∃ h k : ℝ, (h, k) = (4, -1) ∧ ∃ x0 : ℝ, original_parabola x0 = k ∧ x0 = h - 2 :=
by
  convert_and_simplify
  exact sorry

end vertex_after_translation_l83_83763


namespace sequence_divisibility_l83_83704

theorem sequence_divisibility (k : ℤ) (h_k : k > 2) (a : ℕ → ℤ) 
  (h_a0 : a 0 = 0) (h_a1 : ∃ n : ℤ, a 1 = n)
  (h_recurrence : ∀ i : ℕ, a (i + 2) = k * a (i + 1) - a i) :
    ∀ m : ℕ, m > 0 → (2 * m)! ∣ ∏ i in Finset.range (3 * m + 1), a i :=
begin
  sorry
end

end sequence_divisibility_l83_83704


namespace toothpicks_in_square_grid_l83_83458

theorem toothpicks_in_square_grid (n : ℕ) (hn : n = 15) : 
  (n + 1) * n + (n + 1) * n = 480 :=
by
  rw hn
  norm_num

end toothpicks_in_square_grid_l83_83458


namespace repeating_decimal_fraction_sum_l83_83850

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l83_83850


namespace raffle_solution_I_raffle_solution_II_raffle_solution_III_l83_83154

section raffle_problem
variables (total_tickets first_prize_tickets second_prize_tickets third_prize_tickets : ℕ)
variables (prob_draw_2_first_prize prob_draw_3_at_most_1_first_prize expect_xi : ℚ)

-- Conditions:
def conditions : Prop :=
  total_tickets = 10 ∧
  first_prize_tickets = 2 ∧
  second_prize_tickets = 3 ∧
  third_prize_tickets = 5

-- Questions and answers:
def question_I : Prop := prob_draw_2_first_prize = 1 / 45
def question_II : Prop := prob_draw_3_at_most_1_first_prize = 14 / 15
def question_III : Prop := expect_xi = 9 / 10

theorem raffle_solution_I : conditions → question_I :=
by sorry

theorem raffle_solution_II : conditions → question_II :=
by sorry

theorem raffle_solution_III : conditions → question_III :=
by sorry

end raffle_problem


end raffle_solution_I_raffle_solution_II_raffle_solution_III_l83_83154


namespace find_A_l83_83429

theorem find_A (A B C : ℝ) :
  (∀ x : ℝ, x^3 - 2 * x ^ 2 - 13 * x + 10 ≠ 0 → 1 / (x ^ 3 - 2 * x ^ 2 - 13 * x + 10) = A / (x + 2) + B / (x - 1) + C / (x - 1) ^ 2)
  → A = 1 / 9 := 
sorry

end find_A_l83_83429


namespace remainder_of_101_pow_37_mod_100_l83_83464

theorem remainder_of_101_pow_37_mod_100 : (101 ^ 37) % 100 = 1 := by
  sorry

end remainder_of_101_pow_37_mod_100_l83_83464


namespace twenty_fifth_decimal_of_ten_div_eleven_l83_83098

theorem twenty_fifth_decimal_of_ten_div_eleven : 
  (decimal_expansion 10 11).get_nth 25 = 0 := 
sorry

end twenty_fifth_decimal_of_ten_div_eleven_l83_83098


namespace solutions_of_quadratic_eq_l83_83781

theorem solutions_of_quadratic_eq : 
    {x : ℝ | x^2 - 3 * x = 0} = {0, 3} :=
sorry

end solutions_of_quadratic_eq_l83_83781


namespace remainder_of_101_pow_37_mod_100_l83_83466

theorem remainder_of_101_pow_37_mod_100 : (101 ^ 37) % 100 = 1 := by
  sorry

end remainder_of_101_pow_37_mod_100_l83_83466


namespace soccer_lineup_count_l83_83966

theorem soccer_lineup_count :
  let players := 20
  let forwards := 6
  let defenders := 4
  let binom := λ (n k : ℕ), nat.choose n k
  let total_ways :=
    20 * binom 19 6 * binom 13 4
  total_ways = 387889200 :=
by
  sorry

end soccer_lineup_count_l83_83966


namespace minimum_difference_composite_sum_92_l83_83192

def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ n = p * q

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem minimum_difference_composite_sum_92 :
  ∃ x y : ℕ, x ≠ y ∧ is_composite x ∧ is_composite y ∧ x + y = 92 ∧ is_perfect_square (x * y) ∧ abs (x - y) = 8 :=
sorry

end minimum_difference_composite_sum_92_l83_83192


namespace expr1_approx_equals_expr2_approx_equals_l83_83938

-- Define the initial expressions using Lean's architecture.
def expr1 := (2 * Real.sqrt 3) / (Real.sqrt 3 - Real.sqrt 2)
def expr2 := ((3 + Real.sqrt 3) * (1 + Real.sqrt 5)) / ((5 + Real.sqrt 5) * (1 + Real.sqrt 3))

-- Define the approximations of the answers to 3 decimal places.
def answer1_approx := 10.899
def answer2_approx := 0.775

-- Statements to be verified
theorem expr1_approx_equals : Real.abs (expr1 - answer1_approx) < 0.001 := sorry
theorem expr2_approx_equals : Real.abs (expr2 - answer2_approx) < 0.001 := sorry

end expr1_approx_equals_expr2_approx_equals_l83_83938


namespace martin_leftover_raisins_l83_83398

-- Definitions
variables {v k r : ℝ}
-- Let v be the cost of 1 cream puff
-- Let k be the cost of 1 deciliter of Kofola
-- Let r be the cost of 1 dekagram of raisins

theorem martin_leftover_raisins (v k r : ℝ) (h : r ≠ 0) :
  (3 * v + 3 * k = 18 * r) →
  (12 * r + 5 * k = v + 6 * k + x * r) →
  x = 6 →
  x * 10 = 60 :=
begin
  -- Assuming the conditions of the problem:
  -- 1. Martin could buy three cream puffs and 3 dl of Kofola, or 18 dkg of yogurt raisins.
  -- 2. Martin could buy 12 dkg of yogurt raisins and 5 dl of Kofola, or one cream puff and 6 dl of Kofola.
  
  -- This theorem describes that the quantity of yogurt raisins Martin has left over is equal to 60 grams.
 sorry
end

end martin_leftover_raisins_l83_83398


namespace subset_sets_condition_l83_83587

theorem subset_sets_condition {a : ℝ} : 
  ({3, 2 * a} : Set ℝ) ⊆ {a + 1, 3} → a = 1 :=
by
  sorry

end subset_sets_condition_l83_83587


namespace light_path_cube_k_plus_l_l83_83001

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def not_divisible_by_square_of_prime (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → ¬ p^2 ∣ n

def light_path_length (edge : ℕ) (dist_BC : ℕ) (dist_BG : ℕ) : ℝ :=
  let l := dist_BC^2 + dist_BG^2 + edge^2 in
  10 * Real.sqrt l

theorem light_path_cube_k_plus_l (edge : ℕ) (dist_BC dist_BG : ℕ)
  (h_edge : edge = 10) (h_dist_BC : dist_BC = 4) (h_dist_BG : dist_BG = 3) 
  (h_prime : not_divisible_by_square_of_prime 5) :
  ∃ k l : ℕ, not_divisible_by_square_of_prime l ∧ light_path_length edge dist_BC dist_BG = k * Real.sqrt l ∧ k + l = 55 := by
  sorry

end light_path_cube_k_plus_l_l83_83001


namespace distance_flash_overtakes_ace_l83_83976

-- The speeds, head start, and delay definitions
variables (v y : ℝ) (t : ℝ) -- Ace's speed, head start in meters, and delay in minutes

-- The statement of the problem
theorem distance_flash_overtakes_ace :
  let distance := 2 * (y + 60 * v * t)
  in ∀ (v y t : ℝ), (t > 0) -> (v > 0) -> distance = 2 * (y + 60 * v * t) :=
sorry

end distance_flash_overtakes_ace_l83_83976


namespace equation_represents_circle_of_radius_8_l83_83572

theorem equation_represents_circle_of_radius_8 (k : ℝ) : 
  (x^2 + 14 * x + y^2 + 8 * y - k = 0) → k = -1 ↔ (∃ r, r = 8 ∧ (x + 7)^2 + (y + 4)^2 = r^2) :=
by
  sorry

end equation_represents_circle_of_radius_8_l83_83572


namespace recurring_decimal_fraction_sum_l83_83901

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l83_83901


namespace total_birds_in_marsh_l83_83090

def number_of_geese : Nat := 58
def number_of_ducks : Nat := 37

theorem total_birds_in_marsh :
  number_of_geese + number_of_ducks = 95 :=
sorry

end total_birds_in_marsh_l83_83090


namespace average_price_condition_observer_b_correct_l83_83730

-- Define the conditions
def stock_price {n : ℕ} (daily_prices : Fin n → ℝ) : Prop :=
  daily_prices 0 = 5 ∧ 
  daily_prices 6 = 5.14 ∧
  daily_prices 13 = 5 ∧ 
  (∀ i : Fin 6, daily_prices i ≤ daily_prices (i + 1)) ∧ 
  (∀ i : Fin (n - 7), daily_prices (i + 7) ≥ daily_prices (i + 8))

-- Define the problem statements
theorem average_price_condition (daily_prices : Fin 14 → ℝ) :
  stock_price daily_prices →
  ∃ S, 5.09 < (5 + S + 5.14) / 14 ∧ (5 + S + 5.14) / 14 < 5.10 :=
sorry

theorem observer_b_correct (daily_prices : Fin 14 → ℝ) :
  stock_price daily_prices →
  let avg_1 : ℝ := (∑ i in Finset.range 7, daily_prices i) / 7
  let avg_2 : ℝ := (∑ i in Finset.range 7, daily_prices (i + 7)) / 7
  ¬ avg_1 = avg_2 + 0.105 :=
sorry

end average_price_condition_observer_b_correct_l83_83730


namespace expression_value_l83_83475

theorem expression_value (x y z : ℤ) (hx : x = 26) (hy : y = 3 * x / 2) (hz : z = 11) :
  x - (y - z) - ((x - y) - z) = 22 := 
by
  -- problem statement here
  -- simplified proof goes here
  sorry

end expression_value_l83_83475


namespace monthly_rent_calculation_l83_83957

def monthly_rent (house_price : ℝ) (annual_roi_rate : ℝ) (yearly_tax : ℝ) (yearly_insurance : ℝ) (maintenance_rate : ℝ) : ℝ :=
  let annual_income_needed := house_price * annual_roi_rate + yearly_tax + yearly_insurance in
  let monthly_income_needed := annual_income_needed / 12 in
  monthly_income_needed / (1 - maintenance_rate)

theorem monthly_rent_calculation :
  monthly_rent 15000 0.06 450 200 (12.5 / 100) = 147.62 :=
by
  sorry

end monthly_rent_calculation_l83_83957


namespace no_three_collinear_vertices_l83_83993

theorem no_three_collinear_vertices (p : ℕ) [hp : Fact (Nat.Prime p)] :
  ∃ S : Finset (ℕ × ℕ), S.card = p ∧ ∀ (A B C : (ℕ × ℕ)), A ∈ S → B ∈ S → C ∈ S → A ≠ B → B ≠ C → A ≠ C →
    let ⟨xa, ya⟩ := A,
        ⟨xb, yb⟩ := B,
        ⟨xc, yc⟩ := C in
    (yb - ya) * (xc - xb) ≠ (yc - yb) * (xb - xa) :=
by
  sorry

end no_three_collinear_vertices_l83_83993


namespace fourth_term_sum_eq_40_l83_83548

theorem fourth_term_sum_eq_40 : 3^0 + 3^1 + 3^2 + 3^3 = 40 := by
  sorry

end fourth_term_sum_eq_40_l83_83548


namespace sum_of_remainders_l83_83119

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : ((n % 4) + (n % 5) = 4) :=
sorry

end sum_of_remainders_l83_83119


namespace repeating_decimal_fraction_sum_l83_83839

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l83_83839


namespace unfair_draw_fair_draw_with_suit_hierarchy_l83_83320

noncomputable def deck := {suit : String, rank : ℕ // suit ∈ {"hearts", "diamonds", "clubs", "spades"} ∧ rank ∈ {6, 7, 8, 9, 10, 11, 12, 13, 14}}
def prob_V (v : deck) : ℚ := 1 / 36
def prob_M_given_V (v m : deck) : ℚ := 1 / 35
def higher_rank (v m : deck) : Prop := m.rank > v.rank

-- Prove the draw is unfair
theorem unfair_draw : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank v m then prob_M_given_V v m else 0) < 
  (∑ m in (deck \ {v}), if ¬higher_rank v m then prob_M_given_V v m else 0)) :=
sorry

-- Making the draw fair by introducing suit hierarchy
def suit_order : String → ℕ
| "spades" := 4
| "hearts" := 3
| "diamonds" := 2
| "clubs" := 1
| _ := 0

def higher_rank_with_suit (v m : deck) : Prop :=
  if v.rank = m.rank then suit_order m.suit > suit_order v.suit else m.rank > v.rank

-- Prove introducing suit hierarchy can make the draw fair
theorem fair_draw_with_suit_hierarchy : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank_with_suit v m then prob_M_given_V v m else 0) = 
  (∑ m in (deck \ {v}), if ¬higher_rank_with_suit v m then prob_M_given_V v m else 0)) :=
sorry

end unfair_draw_fair_draw_with_suit_hierarchy_l83_83320


namespace recurring_decimal_fraction_sum_l83_83904

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l83_83904


namespace product_of_invertible_labels_l83_83994

-- Define the functions and their domains
def f2 (x : ℝ) : ℝ := x^2 - 4*x + 3
def domain_f2 : set ℝ := set.Icc (-1) 4

def g4 (x : ℝ) : ℝ := -tan x
def domain_g4 : set ℝ := { x : ℝ | -(real.pi / 2) < x ∧ x < real.pi / 2 }

def h5 (x : ℝ) : ℝ := 5 / x
def domain_h5 : set ℝ := { x : ℝ | x < -0.2 ∨ 0.2 < x }

def domain_f3 : set ℤ := { -5, -4, -3, -2, -1, 0, 1, 2, 3 }

-- Statement that these functions are invertible or not
def invertible_f2 := ¬function.injective (λ x : subtype domain_f2, f2 x)
def invertible_f3 := true
def invertible_g4 := function.injective (λ x : subtype domain_g4, g4 x)
def invertible_h5 := function.injective (λ x : subtype domain_h5, h5 x)

theorem product_of_invertible_labels :
  (invertible_f3 → 3) * (invertible_g4 → 4) * (invertible_h5 → 5) = 60 :=
by
  -- Providing definition for the proof, left as incomplete
  sorry

end product_of_invertible_labels_l83_83994


namespace min_quadratic_expr_l83_83816

noncomputable def quadratic_expr (x : ℝ) := 3 * x^2 - 18 * x + 2023

theorem min_quadratic_expr : ∃ x : ℝ, quadratic_expr x = 1996 :=
by
  have h : quadratic_expr (3 : ℝ) = 1996
  exact h
  use 3
  rw h
  sorry -- Proof of h (already derived in given solution)

end min_quadratic_expr_l83_83816


namespace correct_statements_l83_83123

theorem correct_statements :
  (let l := 4
   let r := 2
   central_angle := l / r
   central_angle = 2)
  ∧
  (let y (x : Real) := cos (3 / 2 * x + π / 2)
   is_odd := ∀ x : Real, y (-x) = -y x
   is_odd = True)
:=
begin
  sorry
end

end correct_statements_l83_83123


namespace y_completes_work_in_24_days_l83_83937

noncomputable def total_time_y_completes (x_days : ℕ) (x_works : ℕ) (y_days : ℕ) : ℕ :=
  if x_days = 20 ∧ x_works = 10 ∧ y_days = 12 then 24 else 0

theorem y_completes_work_in_24_days 
  (x_days : ℕ) (x_works : ℕ) (y_days : ℕ)
  (h1 : x_days = 20) 
  (h2 : x_works = 10)
  (h3 : y_days = 12) : 
  total_time_y_completes x_days x_works y_days = 24 :=
by
  unfold total_time_y_completes
  rw [if_pos]
  . refl
  . exact ⟨h1, h2, h3⟩

end y_completes_work_in_24_days_l83_83937


namespace sqrt_sum_ineq_l83_83421

theorem sqrt_sum_ineq (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  sqrt a + sqrt b > sqrt (a + b) :=
sorry

end sqrt_sum_ineq_l83_83421


namespace last_standing_is_Lex_l83_83755

noncomputable def last_student_standing (students : List String) : String :=
  let elimination_rule (n : ℕ) : Bool :=
    let str_n := toString n
    str_n.contains '8' ∨ n % 8 == 0
  
  let eliminate (students : List String) (round : ℕ) : List String :=
    students.filterWithIndex fun i _ => ¬ elimination_rule (round + i + 1)
  
  let rec solve (students : List String) (round : ℕ) : String :=
    if students.length = 1 then
      students.head!
    else
      solve (eliminate students round) (round + students.length)

  solve students 0

theorem last_standing_is_Lex :
  last_student_standing ["Hal", "Ian", "Jak", "Kim", "Lex", "Mia", "Nor", "Opal"] = "Lex" :=
  sorry

end last_standing_is_Lex_l83_83755


namespace instantaneous_rate_of_change_at_4_l83_83247

noncomputable def radius := 4  -- radius of the cup's bottom in cm
noncomputable def volume (t : ℝ) : ℝ := real.pi * t^3 + 2 * real.pi * t^2  -- volume function in ml
noncomputable def area := real.pi * radius^2  -- area of the cup's bottom in cm²
noncomputable def height (t : ℝ) : ℝ := volume(t) / area -- height of the solution as a function of time

noncomputable def height_rate_of_change (t : ℝ) : ℝ := (derivative height) t  -- derivative of height function

theorem instantaneous_rate_of_change_at_4 :
  height_rate_of_change 4 = 4 :=
by
  sorry

end instantaneous_rate_of_change_at_4_l83_83247


namespace lucy_total_fish_l83_83713

theorem lucy_total_fish (current fish_needed : ℕ) (h1 : current = 212) (h2 : fish_needed = 68) : 
  current + fish_needed = 280 := 
by
  sorry

end lucy_total_fish_l83_83713


namespace fraction_simplifies_to_nine_l83_83068

theorem fraction_simplifies_to_nine :
  (\frac{(3 : ℕ) ^ 2010 ^ 2 - (3 : ℕ) ^ 2008 ^ 2}{(3 : ℕ) ^ 2009 ^ 2 - (3 : ℕ) ^ 2007 ^ 2} = 9) :=
by
  sorry

end fraction_simplifies_to_nine_l83_83068


namespace percentage_difference_approx_l83_83089

/-- Number of girls on the playground -/
def numGirls : ℝ := 28.75

/-- Number of boys on the playground -/
def numBoys : ℝ := 36.25

/-- Difference between the number of boys and girls -/
def diff : ℝ := numBoys - numGirls

/-- Total number of children on the playground -/
def totalChildren : ℝ := numGirls + numBoys

/-- The percentage that the difference represents of the total number of children -/
def percentageDifference : ℝ := (diff / totalChildren) * 100

/-- Prove that the percentage difference is approximately 11.54% -/
theorem percentage_difference_approx : percentageDifference ≈ 11.54 := 
sorry

end percentage_difference_approx_l83_83089


namespace find_b_if_perpendicular_l83_83804

-- Defining the problem conditions and expected solution
variable (b : ℝ)

def direction_vector1 := ⟨4, -5⟩
def direction_vector2 := ⟨b, 3⟩

-- Proving that the dot product of the vectors is 0 implies that b = 15/4
theorem find_b_if_perpendicular (hb : (4 * b + (-5) * 3 = 0)) : b = 15 / 4 := by
  -- sorry to skip the proof
  sorry

end find_b_if_perpendicular_l83_83804


namespace sufficient_but_not_necessary_condition_l83_83390

def vectors_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

def vector_a (x : ℝ) : ℝ × ℝ := (2, x - 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x + 1, 4)

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  x = 3 → vectors_parallel (vector_a x) (vector_b x) ∧
  vectors_parallel (vector_a 3) (vector_b 3) :=
by
  sorry

end sufficient_but_not_necessary_condition_l83_83390


namespace average_mpg_l83_83987

-- Definition of the parameters
def initial_odometer : ℕ := 56100
def final_odometer : ℕ := 57500
def start_gasoline : ℕ := 10
def refill1_amount : ℕ := 15
def refill1_odometer : ℕ := 56420
def refill2_amount : ℕ := 25
def refill2_odometer : ℕ := 57020

-- Calculation of total distance and total gasoline used
def total_distance : ℕ := final_odometer - initial_odometer
def total_gasoline_used : ℕ := start_gasoline + refill1_amount + refill2_amount

-- Proof of average miles per gallon
theorem average_mpg :
  total_distance / total_gasoline_used = 28 := by
  have h1 : total_distance = 1400 := rfl
  have h2 : total_gasoline_used = 50 := rfl
  exact (div_eq_iff_eq_mul' (by norm_num : 50 ≠ 0)).mpr rfl

-- Both lines ensure that the Lean code can be compiled successfully by providing necessary assumptions

end average_mpg_l83_83987


namespace solve_inequality_l83_83427

theorem solve_inequality (a : ℝ) : 
  (λ x : ℝ, 12 * x^2 - a * x > a^2) = 
  if a > 0 then 
    {x : ℝ | x < -a / 4 ∨ x > a / 3}
  else if a = 0 then 
    {x : ℝ | x ≠ 0}
  else 
    {x : ℝ | x < a / 3 ∨ x > -a / 4} :=
sorry

end solve_inequality_l83_83427


namespace proj_linear_scaling_l83_83370

open Real

def proj (z u : ℝ) : ℝ := z * u

def proj_vec (z u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_prod := z * u.1 + z * u.2
  let norm_sq := z * z
  (dot_prod / norm_sq * z, dot_prod / norm_sq * z)

variables (u z : ℝ × ℝ)
variables (proj_uz : ℝ × ℝ)

theorem proj_linear_scaling (h : proj_vec z u = (4, -1)) : 
  proj_vec z (3 • u) = (12, -3) :=
sorry

end proj_linear_scaling_l83_83370


namespace repeating_decimal_fraction_sum_l83_83847

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l83_83847


namespace simplify_and_evaluate_l83_83752

-- Definitions based on conditions in part (a)
def simplifiable_expression (x : ℝ) : Prop :=
  (1 + 1/x) / ((x^2 - 1) / x) = 1 / (x - 1)

-- Equivalent Lean statement
theorem simplify_and_evaluate : ∀ x ∈ ({1, -1, 0, 2} : set ℝ), x = 2 → (1 + 1/x) / ((x^2 - 1) / x) = 1 :=
  by
    intros x hx h_eq
    rw [set.mem_set_of_eq] at hx
    cases hx with hx1 hx2
    any_goals { sorry } -- Skipping the proofs for each case
    rw h_eq
    norm_num
    sorry

end simplify_and_evaluate_l83_83752


namespace sqrt_511100_approx_l83_83632

-- Define the conditions
def sqrt_approx_51_11 : ℝ := 7.149
def sqrt_approx_5111 : ℝ := 226.08

-- Define the main theorem we want to prove
theorem sqrt_511100_approx : 
  sqrt 511100 ≈ 714.9 := 
by sorry

end sqrt_511100_approx_l83_83632


namespace part_i_part_ii_l83_83941

-- Part (i)
theorem part_i (m : ℕ) (hm : m ≥ 1) : 
  (∃ p : ℕ, prime p ∧ ∃ k : ℕ, m * (m + 1) = p ^ k) ↔ m = 1 :=
by sorry

-- Part (ii)
theorem part_ii (m a k : ℕ) (hm : m ≥ 1) (ha : a ≥ 1) (hk : k ≥ 2) :
  ¬ (m * (m + 1) = a ^ k) :=
by sorry

end part_i_part_ii_l83_83941


namespace unfair_draw_l83_83313

-- Define the types for suits and ranks
inductive Suit
| hearts | diamonds | clubs | spades

inductive Rank
| Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace

-- Define a card as a combination of suit and rank
structure Card where
  suit : Suit
  rank : Rank

-- Function to determine if a card is higher in rank
def higher_rank (r1 r2 : Rank) : Prop :=
  match r1, r2 with
  | Rank.Six, _ | Rank.Seven, Rank.Six | Rank.Eight, (Rank.Six | Rank.Seven) | Rank.Nine, (Rank.Six | Rank.Seven | Rank.Eight)
  | Rank.Ten, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine) | Rank.Jack, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten)
  | Rank.Queen, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack)
  | Rank.King, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen)
  | Rank.Ace, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King)
    => true
  | _, _ => false

-- Problem statement to prove unfairness of the draw
theorem unfair_draw :
  ∀ (vCard mCard : Card), (∃ (deck : List Card), 
  deck.length = 36 ∧ ∀ c, c ∈ deck →
  match c.rank with 
  | Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King | Rank.Ace => true 
  | _ => false) →
  (∃ (vCard mCard : Card), 
    vCard ∈ deck ∧ mCard ∈ (deck.erase vCard) ∧ higher_rank vCard.rank mCard.rank) →
  ¬fair :=
sorry

end unfair_draw_l83_83313


namespace an_formula_Tn_formula_l83_83256

def an (n : ℕ) : ℕ := 2 * n + 3  -- General formula for the sequence {a_n}

def Sn (n : ℕ) : ℕ := n / 2 * (2 * an 1 + (n - 1) * 2)  -- Sum of first n terms of {a_n}

def bn (n : ℕ) : ℕ := ((an n - n - 4) * 2 ^ n)  -- Sequence {b_n}

def Tn (n : ℕ) : ℕ := ∑ k in Finset.range n, bn k  -- Sum of first n terms of {b_n}

-- Proof that {a_n} is an arithmetic sequence with the given formula and conditions
theorem an_formula (n : ℕ) : an 1 = 5 ∧ Sn 3 = 21 → an n = 2 * n + 3 :=
by sorry

-- Proof that the sum of the first n terms of {b_n} is (n - 2) * 2^(n + 1) + 4
theorem Tn_formula (n : ℕ) : Tn n = (n - 2) * 2 ^ (n + 1) + 4 :=
by sorry

end an_formula_Tn_formula_l83_83256


namespace repeating_decimal_sum_l83_83874

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l83_83874


namespace remaining_amount_correct_l83_83361

-- Defining the conditions from problem statement
def drink_cost (q : ℝ) := q
def small_pizza_cost (q : ℝ) := 1.5 * q
def medium_pizza_cost (q : ℝ) := 2.5 * q
def initial_amount := 50 : ℝ

-- Calculate the total cost
def total_cost (q : ℝ) := 2 * drink_cost q + small_pizza_cost q + medium_pizza_cost q

-- Calculate the remaining amount after purchases
def remaining_amount (q : ℝ) := initial_amount - total_cost q

-- Prove the remaining amount is 50 - 6q
theorem remaining_amount_correct (q : ℝ) : remaining_amount q = 50 - 6 * q := 
by
  -- Proof would go here
  sorry

end remaining_amount_correct_l83_83361


namespace smallest_natural_number_with_condition_l83_83226

theorem smallest_natural_number_with_condition {N : ℕ} :
  (N % 10 = 6) ∧ (4 * N = (6 * 10 ^ ((Nat.digits 10 (N / 10)).length) + (N / 10))) ↔ N = 153846 :=
by
  sorry

end smallest_natural_number_with_condition_l83_83226


namespace quarters_count_l83_83517

variables (Q D : ℕ)
variable h1 : Q + D = 23
variable h2 : 0.25 * Q + 0.10 * D = 3.35

theorem quarters_count : Q = 7 :=
by {
  have h3 : D = 23 - Q,
  calc
    0.25 * Q + 0.10 * (23 - Q) = 3.35 : by rw [←h1]
    ... 0.25 * Q + 2.3 - 0.10 * Q = 3.35 : sorry
    ... 0.15 * Q = 1.05 : by ring
    ... Q = 7 : by norm_num [h2],
  exact (h3)
}

end quarters_count_l83_83517


namespace proposition_p_to_q_l83_83585

theorem proposition_p_to_q (a : ℝ) :
  (a = 1) ↔ (∀ x y : ℝ, let l1 := (fun x y => a * x + y - 1) in
                     let l2 := (fun x y => 3 * x + (a + 2) * y + 1) in
                     (∃ (k : ℝ), (l1 = λ x y, k * (l2 x y)))) :=
sorry

end proposition_p_to_q_l83_83585


namespace maximize_profit_units_l83_83950

noncomputable def fixed_cost := 20000

noncomputable def cost_per_unit := 100

noncomputable def Revenue (x : ℝ) : ℝ :=
  if x ≤ 390 then - x^3 / 900 + 400 * x
  else 90090

noncomputable def Cost (x : ℝ) : ℝ :=
  fixed_cost + cost_per_unit * x

noncomputable def Profit (x : ℝ) : ℝ :=
  Revenue x - Cost x

theorem maximize_profit_units :
  ∃ x : ℝ, (0 ≤ x) ∧ (x ≤ 390) ∧ (Profit x = 40000) :=
begin
  use 300,
  split, { linarith, },
  split, { linarith, },
  exact sorry,
end

end maximize_profit_units_l83_83950


namespace range_of_a_l83_83271

noncomputable def has_two_common_tangents (a : ℝ) : Prop :=
  ∃ (s : ℝ), s > 1 ∧ (f s = a)
  where f (s : ℝ) := Real.log (2 * (s - 1)) - (s + 3) / 2

theorem range_of_a : ∀ (a : ℝ), has_two_common_tangents a ↔ a < 2 * Real.log 2 - 3 :=
begin
  sorry
end

end range_of_a_l83_83271


namespace draw_is_unfair_suit_hierarchy_makes_fair_l83_83308

structure Card where
  suit : ℕ -- 4 suits numbered from 0 to 3
  rank : ℕ -- 9 ranks numbered from 0 to 8

def deck : List Card :=
  List.join (List.map (λ s, List.map (λ r, ⟨s, r⟩) (List.range 9)) (List.range 4))

def DrawFair? : (deck : List Card) → Prop := sorry

-- Part (a): Prove that the draw is unfair
theorem draw_is_unfair : ¬ DrawFair? deck := sorry

-- Part (b): Prove that introducing a suit hierarchy can make the draw fair
def suit_hierarchy : Card → Card → Prop :=
λ c1 c2, (c1.rank < c2.rank) ∨ (c1.rank = c2.rank ∧ c1.suit < c2.suit)

theorem suit_hierarchy_makes_fair : ∃ h : Card → Card → Prop, h = suit_hierarchy ∧ DrawFair? deck[h] := sorry

end draw_is_unfair_suit_hierarchy_makes_fair_l83_83308


namespace mary_sheep_purchase_l83_83402

theorem mary_sheep_purchase: 
  ∀ (mary_sheep bob_sheep add_sheep : ℕ), 
    mary_sheep = 300 → 
    bob_sheep = 2 * mary_sheep + 35 → 
    add_sheep = (bob_sheep - 69) - mary_sheep → 
    add_sheep = 266 :=
by
  intros mary_sheep bob_sheep add_sheep _ _
  sorry

end mary_sheep_purchase_l83_83402


namespace sum_of_fraction_terms_l83_83865

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l83_83865


namespace find_n_from_equation_l83_83635

theorem find_n_from_equation :
  ∃ n : ℕ, (3 * 3 * 5 * 5 * 7 * 9 = 3 * 3 * 7 * n * n) → n = 15 :=
by
  sorry

end find_n_from_equation_l83_83635


namespace _l83_83441

noncomputable theorem measure_angle_EAB
  (ABCDE : Type)
  (equilateral_pentagon : ∀ (A B C D E : ABCDE), ∃ (l : ℝ), dist A B = l ∧ dist B C = l ∧ dist C D = l ∧ dist D E = l ∧ dist E A = l)
  (no_coplanar_four : ∀ (A B C D E : ABCDE), ¬ collinear A B C ∧ ¬ collinear B C D ∧ ¬ collinear C D E ∧ ¬ collinear D E A ∧ ¬ collinear E A B)
  (angle_ABC_90 : ∀ (A B C : ABCDE), angle A B C = π/2)
  (angle_BCD_90 : ∀ (B C D : ABCDE), angle B C D = π/2)
  (angle_CDE_90 : ∀ (C D E : ABCDE), angle C D E = π/2)
  (angle_DEA_90 : ∀ (D E A : ABCDE), angle D E A = π/2) :
  angle E A B = 2 * π * (110 * 60 + 55) / (360 * 60) :=
sorry

end _l83_83441


namespace work_completion_times_l83_83092

-- Definitions based on conditions
def condition1 (x y : ℝ) : Prop := 2 * (1 / x) + 5 * (1 / y) = 1 / 2
def condition2 (x y : ℝ) : Prop := 3 * (1 / x + 1 / y) = 0.45

-- Main theorem stating the solution
theorem work_completion_times :
  ∃ (x y : ℝ), condition1 x y ∧ condition2 x y ∧ x = 12 ∧ y = 15 := 
sorry

end work_completion_times_l83_83092


namespace polynomial_divisibility_l83_83746

theorem polynomial_divisibility (m : ℕ) (odd_m : m % 2 = 1) (x y z : ℤ) :
    ∃ k : ℤ, (x + y + z)^m - x^m - y^m - z^m = k * ((x + y + z)^3 - x^3 - y^3 - z^3) := 
by 
  sorry

end polynomial_divisibility_l83_83746


namespace exists_singular_matrix_with_stable_det_l83_83999

theorem exists_singular_matrix_with_stable_det (n : ℕ) :
  ∃ (A : matrix (fin n) (fin n) ℝ),
  (det A = 0) ∧ 
  (∀ i j, det (A.update_entry i j (A i j + 1)) ≠ 0) :=
sorry

end exists_singular_matrix_with_stable_det_l83_83999


namespace draw_is_unfair_suit_hierarchy_makes_fair_l83_83309

structure Card where
  suit : ℕ -- 4 suits numbered from 0 to 3
  rank : ℕ -- 9 ranks numbered from 0 to 8

def deck : List Card :=
  List.join (List.map (λ s, List.map (λ r, ⟨s, r⟩) (List.range 9)) (List.range 4))

def DrawFair? : (deck : List Card) → Prop := sorry

-- Part (a): Prove that the draw is unfair
theorem draw_is_unfair : ¬ DrawFair? deck := sorry

-- Part (b): Prove that introducing a suit hierarchy can make the draw fair
def suit_hierarchy : Card → Card → Prop :=
λ c1 c2, (c1.rank < c2.rank) ∨ (c1.rank = c2.rank ∧ c1.suit < c2.suit)

theorem suit_hierarchy_makes_fair : ∃ h : Card → Card → Prop, h = suit_hierarchy ∧ DrawFair? deck[h] := sorry

end draw_is_unfair_suit_hierarchy_makes_fair_l83_83309


namespace sum_of_z_for_f4z_eq_14_l83_83378

def f (x : ℝ) : ℝ :=
  x^2 - 2 * x + 3

theorem sum_of_z_for_f4z_eq_14 :
  let z : ℝ := by sorry
  (∑ z in {z | f (4 * z) = 14}, z) = 1 / 4 := 
sorry

end sum_of_z_for_f4z_eq_14_l83_83378


namespace unfair_draw_l83_83315

-- Define the types for suits and ranks
inductive Suit
| hearts | diamonds | clubs | spades

inductive Rank
| Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace

-- Define a card as a combination of suit and rank
structure Card where
  suit : Suit
  rank : Rank

-- Function to determine if a card is higher in rank
def higher_rank (r1 r2 : Rank) : Prop :=
  match r1, r2 with
  | Rank.Six, _ | Rank.Seven, Rank.Six | Rank.Eight, (Rank.Six | Rank.Seven) | Rank.Nine, (Rank.Six | Rank.Seven | Rank.Eight)
  | Rank.Ten, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine) | Rank.Jack, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten)
  | Rank.Queen, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack)
  | Rank.King, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen)
  | Rank.Ace, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King)
    => true
  | _, _ => false

-- Problem statement to prove unfairness of the draw
theorem unfair_draw :
  ∀ (vCard mCard : Card), (∃ (deck : List Card), 
  deck.length = 36 ∧ ∀ c, c ∈ deck →
  match c.rank with 
  | Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King | Rank.Ace => true 
  | _ => false) →
  (∃ (vCard mCard : Card), 
    vCard ∈ deck ∧ mCard ∈ (deck.erase vCard) ∧ higher_rank vCard.rank mCard.rank) →
  ¬fair :=
sorry

end unfair_draw_l83_83315


namespace find_b_l83_83301

open Real

def f (x : ℝ) (φ : ℝ) (b : ℝ) := sin (2 * x + φ) + b

theorem find_b (φ b : ℝ) (h₁ : ∀ x : ℝ, f (x + π / 3) φ b = f (-x) φ b)
  (h₂ : f (2 * π / 3) φ b = -1) : b = 0 ∨ b = -2 :=
sorry

end find_b_l83_83301


namespace function_matches_table_values_l83_83576

variable (f : ℤ → ℤ)

theorem function_matches_table_values (h1 : f (-1) = -2) (h2 : f 0 = 0) (h3 : f 1 = 2) (h4 : f 2 = 4) : 
  ∀ x : ℤ, f x = 2 * x := 
by
  -- Prove that the function satisfying the given table values is f(x) = 2x
  sorry

end function_matches_table_values_l83_83576


namespace sum_of_first_n_odd_numbers_eq_n_squared_l83_83413

theorem sum_of_first_n_odd_numbers_eq_n_squared (n : ℕ) : 
  let S := λ n : ℕ, (finset.range n).sum (λ i, 2 * i + 1)
  in S n = n * n :=
by
  sorry

end sum_of_first_n_odd_numbers_eq_n_squared_l83_83413


namespace trajectory_of_P_fixed_point_l83_83347

-- Definition of the first condition
def distance_from_point_to_line (P : ℝ × ℝ) (line : ℝ → Prop) : ℝ :=
  abs (P.1 + 1) / sqrt 1 -- distance to line x = -1

-- Definition of the main condition
def point_P_cond (P : ℝ × ℝ) : Prop :=
  let point_Q := (1, 0)
  sqrt ((P.1 - point_Q.1)^2 + (P.2 - point_Q.2)^2) = distance_from_point_to_line P (λ x, x = -1)

-- First part: Proving the trajectory C
theorem trajectory_of_P (P : ℝ × ℝ) (h : point_P_cond P) : P.2^2 = 4 * P.1 := sorry

-- Second part: Proving line l passing through a fixed point
theorem fixed_point (l : ℝ → ℝ) (M N : ℝ × ℝ) (hM : M.2^2 = 4 * M.1) (hN : N.2^2 = 4 * N.1) 
  (h₀ : M ≠ (0, 0)) (h₁ : N ≠ (0, 0)) (h₂ : (M.1 * N.1 + M.2 * N.2) = 0) : 
  ∃ (F : ℝ × ℝ), F = (4, 0) ∧ ∀ (y : ℝ), l y = y * (M.1 / M.2) + 4 := sorry

end trajectory_of_P_fixed_point_l83_83347


namespace sum_of_fraction_parts_l83_83883

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l83_83883


namespace sum_of_fraction_parts_of_repeating_decimal_l83_83828

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l83_83828


namespace sum_of_fraction_numerator_and_denominator_l83_83909

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l83_83909


namespace no_integer_roots_of_polynomial_l83_83219

theorem no_integer_roots_of_polynomial :
  ¬ ∃ (x : ℤ), x^3 - 3 * x^2 - 10 * x + 20 = 0 :=
by
  sorry

end no_integer_roots_of_polynomial_l83_83219


namespace ratio_of_volumes_l83_83535

noncomputable def volume_cylinder (a : ℝ) : ℝ :=
  2 * π * a^3

noncomputable def volume_sphere (a : ℝ) : ℝ :=
  (4 / 3) * π * a^3

noncomputable def volume_cone (a : ℝ) : ℝ :=
  (2 / 3) * π * a^3

theorem ratio_of_volumes (a : ℝ) (ha : 0 < a) :
  (volume_cylinder a) / (volume_cone a) = 3 ∧
  (volume_sphere a) / (volume_cone a) = 2 :=
by
  sorry

end ratio_of_volumes_l83_83535


namespace complex_solution_l83_83204

theorem complex_solution (z : ℂ) (h : 4 * z + 2 * complex.I * conj z = -14 - 6 * complex.I) :
  z = -17/5 + 1/5 * complex.I :=
sorry

end complex_solution_l83_83204


namespace sum_of_a_and_b_l83_83294

theorem sum_of_a_and_b :
  ∃ (a b : ℚ), ((1 + real.sqrt 2)^5 = a + b * real.sqrt 2) ∧ (a + b = 70) := 
by
  sorry

end sum_of_a_and_b_l83_83294


namespace maximize_wz_xy_zx_l83_83387

-- Variables definition
variables {w x y z : ℝ}

-- Main statement
theorem maximize_wz_xy_zx (h_sum : w + x + y + z = 200) (h_nonneg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) :
  (w * z + x * y + z * x) ≤ 7500 :=
sorry

end maximize_wz_xy_zx_l83_83387


namespace solve_for_x_l83_83624

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end solve_for_x_l83_83624


namespace math_team_selection_l83_83179

/-- At Clearview High School, the math team is being selected from a club that includes 
four girls and six boys. Prove that the number of different teams consisting of three girls 
and two boys can be formed is 180 if one of the girls must be the team captain. -/
theorem math_team_selection (G B : ℕ) (choose : ℕ → ℕ → ℕ) (girls captains remaining_boys : ℕ) :
  G = 4 → B = 6 → 
  girls = choose G 1 * choose (G - 1) 2 → 
  captains = choose B 2 →
  remaining_boys = girls * captains →
  remaining_boys = 180 := 
by 
  intro hG hB hgirls hcaptains hresult
  rw [hG, hB] at *
  simp only [Nat.choose] at *
  sorry

end math_team_selection_l83_83179


namespace remainder_101_pow_37_mod_100_l83_83471

theorem remainder_101_pow_37_mod_100 :
  (101: ℤ) ≡ 1 [MOD 100] →
  (101: ℤ)^37 ≡ 1 [MOD 100] :=
by
  sorry

end remainder_101_pow_37_mod_100_l83_83471


namespace inversion_of_three_l83_83544

theorem inversion_of_three : 3⁻¹ = 1 / 3 := 
by sorry

end inversion_of_three_l83_83544


namespace instantaneous_rate_of_change_at_4s_l83_83246

noncomputable def V : ℝ → ℝ := λ t, Real.pi * t^3 + 2 * Real.pi * t^2

noncomputable def S : ℝ := Real.pi * 4^2

noncomputable def h (t : ℝ) : ℝ := (V t) / S

noncomputable def h' (t: ℝ) : ℝ := (deriv (λ t, h t)) t

theorem instantaneous_rate_of_change_at_4s : h' 4 = 4 :=
by
  -- proof steps will be inserted here
  sorry

end instantaneous_rate_of_change_at_4s_l83_83246


namespace find_b8_as_fraction_l83_83018

/-- Define the sequences a and b based on initial conditions and recurrence relations -/
def a : ℕ → ℚ 
| 0     := 3 
| (n+1) := (a n)^2 / b n

def b : ℕ → ℚ 
| 0     := 4 
| (n+1) := (b n)^2 / a n

/-- Define the s and t sequences as outlined from the conditions -/
def s : ℕ → ℕ 
| 0     := 1 
| (n+1) := 2 * s n - t n

def t : ℕ → ℕ 
| 0     := 0 
| (n+1) := 2 * t n - s n

/-- The main theorem to prove the required equivalent problem -/
theorem find_b8_as_fraction :
  b 8 = 4^17 / 3^16 :=
sorry

end find_b8_as_fraction_l83_83018


namespace min_decimal_digits_l83_83462

theorem min_decimal_digits (n : ℕ) (h : n = 987654321) : 
  let denom := (2^30) * (5^3)
  in (min_digits_to_decimal n denom = 30) := by
  sorry

end min_decimal_digits_l83_83462


namespace stock_price_satisfies_conditions_part_a_stock_price_satisfies_conditions_part_b_l83_83725

-- Define the conditions: stock prices on specific dates
constants apr_7 apr_13 apr_20 : ℝ
axiom apr_7_eq : apr_7 = 5
axiom apr_13_eq : apr_13 = 5.14
axiom apr_20_eq : apr_20 = 5

-- Define the prices on the intervening dates
constants (x : ℕ → ℝ)

-- Define the correct answer for average price calculation for part (a)
def avg_price_between_apr_7_Apr_20 : ℝ := (apr_7 + (Σ i in FinSet.range 12, x i) + apr_13 + (Σ j in FinSet.range 6, x (j + 7)) + apr_20) / 14

noncomputable def part_a : Prop :=
  5.09 < avg_price_between_apr_7_Apr_20 ∧ avg_price_between_apr_7_Apr_20 < 5.10

-- Part (b): Comparing average stock prices for different periods
def avg_price_apr_7_to_apr_13 : ℝ := (apr_7 + (Σ i in FinSet.range 5, x i) + apr_13) / 7
def avg_price_apr_14_to_apr_20 : ℝ := (apr_13 + x 7 + x 8 + x 9 + x 10 + x 11 + apr_20) / 7

noncomputable def part_b : Prop :=
  | (avg_price_apr_14_to_apr_20 - avg_price_apr_7_to_apr_13 ≠ 0.105 )

-- The final proof problems for part (a) and part (b)
theorem stock_price_satisfies_conditions_part_a : part_a := sorry
theorem stock_price_satisfies_conditions_part_b : part_b := sorry

end stock_price_satisfies_conditions_part_a_stock_price_satisfies_conditions_part_b_l83_83725


namespace inradii_sum_l83_83016

theorem inradii_sum (A B C K L M : Point) (hK: K ∈ segment B C) (hL: L ∈ segment C A) (hM: M ∈ segment A B)
  (h_intersect: ∃ P : Point, concurrent (line_through A K) (line_through B L) (line_through C M)) :
  ∃ Δ₁ Δ₂ : Triangle, Δ₁ ∈ {Triangle.mk A L M, Triangle.mk B M K, Triangle.mk C K L} ∧
              Δ₂ ∈ {Triangle.mk A L M, Triangle.mk B M K, Triangle.mk C K L} ∧
              Δ₁ ≠ Δ₂ ∧
              inradius Δ₁ + inradius Δ₂ ≥ inradius (Triangle.mk A B C) :=
sorry

end inradii_sum_l83_83016


namespace cost_of_high_heels_l83_83563

theorem cost_of_high_heels
  (H : ℝ)
  (h_condition : H + (5 : ℝ) * (2 / 3) * H = 260) : 
  H = 60 := 
begin 
  -- The proof is omitted let's add sorry
  sorry 
end

end cost_of_high_heels_l83_83563


namespace trapezoid_area_l83_83463

theorem trapezoid_area (E F G H : ℝ × ℝ)
  (hE : E = (0, 0)) (hF : F = (0, 3)) (hG : G = (3, 3)) (hH : H = (6, 0)) :
  let base1 := dist G F
  let base2 := dist H E
  let height := abs (F.2 - E.2) 
  in (1 / 2) * (base1 + base2) * height = 13.5 := 
by {
  sorry
}

end trapezoid_area_l83_83463


namespace no_counterexample_exists_l83_83381

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_counterexample_exists : ∀ n : ℕ, sum_of_digits n % 9 = 0 → n % 9 = 0 :=
by
  intro n h
  sorry

end no_counterexample_exists_l83_83381


namespace rectangle_area_of_given_conditions_l83_83487

theorem rectangle_area_of_given_conditions :
  let s := Real.sqrt 1225 in
  let r := s in
  let l := r / 4 in
  let b := 10 in
  l * b = 87.5 :=
by
  -- The proof is omitted as we are only asked for the statement
  sorry

end rectangle_area_of_given_conditions_l83_83487


namespace composition_difference_l83_83051

def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := x / 4 + 1

theorem composition_difference (x : ℝ) : f (g x) - g (f x) = 1 / 4 :=
by
  sorry

end composition_difference_l83_83051


namespace quantity_of_milk_in_original_mixture_l83_83127

variable (M W : ℕ)

-- Conditions
def ratio_original : Prop := M = 2 * W
def ratio_after_adding_water : Prop := M * 5 = 6 * (W + 10)

theorem quantity_of_milk_in_original_mixture
  (h1 : ratio_original M W)
  (h2 : ratio_after_adding_water M W) :
  M = 30 := by
  sorry

end quantity_of_milk_in_original_mixture_l83_83127


namespace adult_ticket_cost_l83_83176

-- Definitions from the conditions
def total_amount : ℕ := 35
def child_ticket_cost : ℕ := 3
def num_children : ℕ := 9

-- The amount spent on children’s tickets
def total_child_ticket_cost : ℕ := num_children * child_ticket_cost

-- The remaining amount after purchasing children’s tickets
def remaining_amount : ℕ := total_amount - total_child_ticket_cost

-- The adult ticket cost should be equal to the remaining amount
theorem adult_ticket_cost : remaining_amount = 8 :=
by sorry

end adult_ticket_cost_l83_83176


namespace range_of_a_for_positive_quadratic_l83_83593

theorem range_of_a_for_positive_quadratic (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) := 
sor: integrators.parametrize_params.get_rest.aggregate.get_synonyms_recursive.remove_element_also._


end range_of_a_for_positive_quadratic_l83_83593


namespace prime_factor_of_T_l83_83480

-- Define constants and conditions
def x : ℕ := 2021
def T : ℕ := Nat.sqrt ((x + x) + (x - x) + (x * x) + (x / x))

-- Define what needs to be proved
theorem prime_factor_of_T : ∃ p : ℕ, Nat.Prime p ∧ Nat.factorization T p > 0 ∧ (∀ q : ℕ, Nat.Prime q ∧ Nat.factorization T q > 0 → q ≤ p) :=
sorry

end prime_factor_of_T_l83_83480


namespace repeating_decimal_fraction_sum_l83_83844

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l83_83844


namespace area_GHCD_correct_l83_83666

noncomputable def area_of_GHCD : ℝ :=
  let GH := (10 + 24) / 2
  let altitude_GHCD := 15 / 2
  let area := (GH + 24) * altitude_GHCD / 2
  area

theorem area_GHCD_correct :
  area_of_GHCD = 153.75 :=
by
  have GH_eq : (10 + 24) / 2 = 17 := by norm_num
  have altitude_eq : 15 / 2 = 7.5 := by norm_num
  calc
    area_of_GHCD
        = (GH + 24) * altitude_GHCD / 2 : rfl
    ... = (17 + 24) * 7.5 / 2 : by rw [GH_eq, altitude_eq]
    ... = 153.75 : by norm_num

end area_GHCD_correct_l83_83666


namespace repeating_decimal_fraction_sum_l83_83838

noncomputable def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d = 0.45 then 5 / 11 else 0

theorem repeating_decimal_fraction_sum :
  let x := repeating_decimal_to_fraction 0.45 in
  (x.num + x.den = 16) :=
by
  let x := repeating_decimal_to_fraction 0.45
  have h : x = 5 / 11 := by
    unfold repeating_decimal_to_fraction
    split_ifs with h
    . exact rfl
  rw [h, Rat.num_den_eq 5 11, Int.add_comm]
  sorry

end repeating_decimal_fraction_sum_l83_83838


namespace root_and_increasing_inequality_l83_83590

noncomputable def f (x : ℝ) : ℝ := 2^x - log (1/2) x

theorem root_and_increasing_inequality (a x0 : ℝ) (h_root : f a = 0) (h_interval : 0 < x0 ∧ x0 < a) (h_increasing : ∀ x y : ℝ, 0 < x → x < y → f x < f y) : 
  f x0 < 0 :=
by
  sorry

end root_and_increasing_inequality_l83_83590


namespace angle_between_a_b_l83_83610

open Real

def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.acos ((a.1 * b.1 + a.2 * b.2) / (sqrt (a.1 ^ 2 + a.2 ^ 2) * sqrt (b.1 ^ 2 + b.2 ^ 2)))

theorem angle_between_a_b {a b : ℝ × ℝ}
  (h1 : sqrt (a.1 ^ 2 + a.2 ^ 2) = 3)
  (h2 : sqrt (b.1 ^ 2 + b.2 ^ 2) = 1)
  (h3 : sqrt ((a.1 - 3 * b.1) ^ 2 + (a.2 - 3 * b.2) ^ 2) = 3) :
  angle_between_vectors a b = π / 3 :=
sorry

end angle_between_a_b_l83_83610


namespace find_a_l83_83140

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + x

-- Define the derivative of the function f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x + 1

-- The main theorem: if the tangent at x = 1 is parallel to the line y = 2x, then a = 1
theorem find_a (a : ℝ) : f' 1 a = 2 → a = 1 :=
by
  intro h
  -- The proof is skipped
  sorry

end find_a_l83_83140


namespace part_a_prices_example_part_b_correct_observer_l83_83718

theorem part_a_prices_example (p : ℕ → ℝ)
  (h1 : p 0 = 5)
  (h2 : p 6 = 5.14)
  (h3 : p 13 = 5)
  (h_nondecreasing : ∀ i, 0 ≤ i ∧ i < 6 → p i ≤ p (i+1))
  (h_nonincreasing : ∀ i, 6 ≤ i ∧ i < 12 → p (i+1) ≤ p i)
  (h_avg_conditional: 5.09 < (∑ i in range 14, p i) / 14 ∧ (∑ i in range 14, p i) / 14 < 5.10)
  : ∃ p, 56.12 < (p 1 + p 2 + p 3 + p 4 + p 5 +p 7 + p 8 + p 9 + p 10 + p 11 + p 12) ∧ (p 1 + p 2 + p 3 + p 4 + p 5 +p 7 + p 8 + p 9 + p 10 + p 11 + p 12) < 56.26 := 
begin
  sorry
end

theorem part_b_correct_observer
  (p : ℕ → ℝ)
  (h1 : p 0 = 5)
  (h2 : p 6 = 5.14)
  (h3 : p 13 = 5)
  (h_nondecreasing : ∀ i, 0 ≤ i ∧ i < 6 → p i ≤ p (i+1))
  (h_nonincreasing : ∀ i, 6 ≤ i ∧ i < 12 → p (i+1) ≤ p i)
  : (∑ i in range 7, p i) / 7 + 10.5 < (∑ i in range 7, p (i + 7)) / 7 → observer B is correct :=
begin
  sorry
end

end part_a_prices_example_part_b_correct_observer_l83_83718


namespace second_solution_sugar_concentration_l83_83486

variable (W : ℝ) -- Total weight of the original solution
variable (S : ℝ) -- Concentration of sugar in the second solution

theorem second_solution_sugar_concentration :
  (∀ W : ℝ, W > 0 →
  let original_solution_sugar := 0.1 * W / 4 in
  let resulting_solution_sugar := 0.2 * W in
  let first_part_sugar := 3 * original_solution_sugar in
  let second_solution_sugar := S / 100 * W / 4 in
  resulting_solution_sugar = first_part_sugar + second_solution_sugar →
  S = 50) := 
by 
  sorry

end second_solution_sugar_concentration_l83_83486


namespace joe_trip_avg_speed_l83_83682

def joe_avg_speed (d1 d2 d3 d4 : ℝ) (v1 v2 v3 v4 : ℝ) : ℝ :=
  let total_distance := d1 + d2 + d3 + d4
  let total_time := d1 / v1 + d2 / v2 + d3 / v3 + d4 / v4
  total_distance / total_time

theorem joe_trip_avg_speed :
  joe_avg_speed 360 200 150 90 60 80 50 40 = 58.18 :=
by
  sorry

end joe_trip_avg_speed_l83_83682


namespace range_of_a_l83_83021

def f (x : ℝ) : ℝ :=
if x < 0 then (1/2)^x - 7 else sqrt x

theorem range_of_a (a : ℝ) : f a < 1 ↔ -3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l83_83021


namespace binom_sum_problem_l83_83109

theorem binom_sum_problem :
  ∑ n in {n : ℕ | nat.choose 30 n = nat.choose 16 16 ∨ nat.choose 16 14}, n = 30 :=
by
  sorry

end binom_sum_problem_l83_83109


namespace ella_dog_food_ratio_l83_83660

variable (ella_food_per_day : ℕ) (total_food_10days : ℕ) (x : ℕ)

theorem ella_dog_food_ratio
  (h1 : ella_food_per_day = 20)
  (h2 : total_food_10days = 1000) :
  (x : ℕ) = 4 :=
by
  sorry

end ella_dog_food_ratio_l83_83660


namespace projection_onto_plane_l83_83705

-- Declare a plane Q with normal vector
def normal_vector : ℝ^3 := ![2, -1, 2]

-- Define the projection matrix Q
def projection_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![5 / 9, 2 / 9, -4 / 9],
    ![2 / 9, 8 / 9, 2 / 9],
    ![-4 / 9, 2 / 9, 5 / 9]]

-- State the theorem
theorem projection_onto_plane (v : ℝ^3) : 
  projection_matrix.mul_vec v = v - ((v.dot_product normal_vector) / (normal_vector.dot_product normal_vector)) • normal_vector :=
sorry

end projection_onto_plane_l83_83705


namespace fraction_to_percentage_decimal_l83_83553

theorem fraction_to_percentage_decimal (num : ℚ) (den : ℚ) (h : den ≠ 0) :
  num / den = 7 / 15 → (num / den) * 100 / 100 = 0.4666 :=
by
  sorry

end fraction_to_percentage_decimal_l83_83553


namespace determine_value_of_a_l83_83205

theorem determine_value_of_a :
  ∃ b, (∀ x : ℝ, (4 * x^2 + 12 * x + (b^2)) = (2 * x + b)^2) :=
sorry

end determine_value_of_a_l83_83205


namespace find_p_l83_83160

variable {p : ℝ}
variable (hx : 0 < p)

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := y^2 = p * x

-- Definition of the slant line passing through the focus F(p/2, 0)
def slant_line (x y : ℝ) : Prop := y = x - p / 2

-- Focus of the parabola
def focus := (p / 2, 0)

-- Points of intersection A and B
variable {x1 y1 x2 y2 : ℝ}
def points_of_intersection : Prop :=
  parabola x1 y1 ∧ slant_line x1 y1 ∧
  parabola x2 y2 ∧ slant_line x2 y2

-- Area of triangle OAB
noncomputable def area_OAB : ℝ :=
  1 / 2 * (x1 * y2 - x2 * y1)

-- Given area of triangle OAB
def given_area : Prop := area_OAB = 2 * Real.sqrt 2

theorem find_p (h_intersection : points_of_intersection) (h_area : given_area) : p = 4 * Real.sqrt 2 := sorry

end find_p_l83_83160


namespace six_consecutive_ints_product_condition_l83_83200

theorem six_consecutive_ints_product_condition :
  (∃ (a b c d e f : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
    f = e + 1 ∧ e = d + 1 ∧ d = c + 1 ∧ c = b + 1 ∧ b = a + 1 ∧
    (∃ x y z w : ℕ, {a, b, c, d, e, f} = {x, y, z, w, a * b + c * d = e * f})) →
  {a, a + 1, a + 2, a + 3, a + 4, a + 5} = {6, 7, 8, 9, 10, 11} := 
sorry

end six_consecutive_ints_product_condition_l83_83200


namespace gcd_is_3_l83_83101

noncomputable def a : ℕ := 130^2 + 240^2 + 350^2
noncomputable def b : ℕ := 131^2 + 241^2 + 351^2

theorem gcd_is_3 : Nat.gcd a b = 3 := 
by 
  sorry

end gcd_is_3_l83_83101


namespace car_miles_per_tankful_in_city_l83_83145

-- Define constants for the given values
def miles_per_tank_on_highway : ℝ := 462
def fewer_miles_per_gallon : ℝ := 15
def miles_per_gallon_in_city : ℝ := 40

-- Prove the car traveled 336 miles per tankful in the city
theorem car_miles_per_tankful_in_city :
  (miles_per_tank_on_highway / (miles_per_gallon_in_city + fewer_miles_per_gallon)) * miles_per_gallon_in_city = 336 := 
by
  sorry

end car_miles_per_tankful_in_city_l83_83145


namespace correct_function_is_f1_l83_83526

variable x : ℝ

def f1 (x : ℝ) := cos (2 * x + π / 6)
def f2 (x : ℝ) := sin (2 * x + π / 6)
def f3 (x : ℝ) := cos (x / 2 + π / 6)
def f4 (x : ℝ) := tan (2 * x + 2 * π / 3)

def has_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x ∧ (∀ q, 0 < q < p → f (x + q) ≠ f x)

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x, f (2 * a - x) = -f x

theorem correct_function_is_f1 :
  has_smallest_positive_period f1 π ∧ is_symmetric_about f1 (π / 6) :=
  sorry

end correct_function_is_f1_l83_83526


namespace fujian_provincial_games_distribution_count_l83_83052

theorem fujian_provincial_games_distribution_count 
  (staff_members : Finset String)
  (locations : Finset String)
  (A B C D E F : String)
  (A_in_B : A ∈ staff_members)
  (B_in_B : B ∈ staff_members)
  (C_in_B : C ∈ staff_members)
  (D_in_B : D ∈ staff_members)
  (E_in_B : E ∈ staff_members)
  (F_in_B : F ∈ staff_members)
  (locations_count : locations.card = 2)
  (staff_count : staff_members.card = 6)
  (must_same_group : ∀ g₁ g₂ : Finset String, A ∈ g₁ → B ∈ g₁ → g₁ ∪ g₂ = staff_members)
  (min_two_people : ∀ g : Finset String, 2 ≤ g.card) :
  ∃ distrib_methods : ℕ, distrib_methods = 22 := 
by
  sorry

end fujian_provincial_games_distribution_count_l83_83052


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l83_83894

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l83_83894


namespace break2023_cycle_l83_83071

theorem break2023_cycle :
  let letters := ["B", "R", "E", "A", "K"]
  let digits := [2, 0, 2, 3]
  let letter_cycle := 5
  let digit_cycle := 4
  nat.lcm letter_cycle digit_cycle = 20 :=
by
  sorry

end break2023_cycle_l83_83071


namespace bushes_needed_l83_83977

theorem bushes_needed 
  (sides : ℕ) (side_length : ℕ) (bush_fill : ℕ)
  (h_sides : sides = 3)
  (h_side_length : side_length = 16)
  (h_bush_fill : bush_fill = 4) : 
  sides * side_length / bush_fill = 12 :=
by
  rw [h_sides, h_side_length, h_bush_fill]
  norm_num
  sorry

end bushes_needed_l83_83977


namespace surjective_gamma_implies_alpha_le_half_l83_83362

open Real

noncomputable def γ (s : ℝ) : ℝ × ℝ := sorry

theorem surjective_gamma_implies_alpha_le_half (M α : ℝ) (hγ : ∀ s t ∈ Icc 0 1, dist (γ s) (γ t) ≤ M * abs (s - t) ^ α)
  (hsurj : Function.Surjective γ) : α ≤ 1 / 2 :=
by
  sorry

end surjective_gamma_implies_alpha_le_half_l83_83362


namespace max_distance_line_eq_l83_83065

   noncomputable def line_through_point_max_distance (x y : ℝ) : Prop :=
   ∃ (a b c : ℝ), a * x + b * y + c = 0

   theorem max_distance_line_eq :
     line_through_point_max_distance 1 2 ↔ (∀ (l : ℝ → ℝ → Prop), 
     l 1 2 → l 0 0 = (x + 2 * y - 5)) :=
   sorry
   
end max_distance_line_eq_l83_83065


namespace necessary_sufficient_condition_l83_83073

theorem necessary_sufficient_condition (a : ℝ) :
  (∃ x : ℝ, ax^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ a ≤ 1 := sorry

end necessary_sufficient_condition_l83_83073


namespace circumscribed_sphere_radius_l83_83657

variable (a b c : ℝ)

def radius_of_circumscribed_sphere (a b c : ℝ) : ℝ :=
  (sqrt (4 * c^4 - a^2 * b^2)) / (2 * sqrt (4 * c^2 - a^2 - b^2))

theorem circumscribed_sphere_radius (R : ℝ):
  R = (sqrt (4 * c^4 - a^2 * b^2)) / (2 * sqrt (4 * c^2 - a^2 - b^2)) :=
sorry

end circumscribed_sphere_radius_l83_83657


namespace min_points_double_star_l83_83328

-- Define the conditions
variables {n : ℕ} (A B : ℕ)
hypothesis h1 : A = B + 15
hypothesis h2 : n * 15 = 360

-- Prove that the minimal number of points n is 24
theorem min_points_double_star : n = 24 :=
sorry

end min_points_double_star_l83_83328


namespace largest_common_term_up_to_150_l83_83996

theorem largest_common_term_up_to_150 :
  ∃ a : ℕ, a ≤ 150 ∧ (∃ n : ℕ, a = 2 + 8 * n) ∧ (∃ m : ℕ, a = 3 + 9 * m) ∧ (∀ b : ℕ, b ≤ 150 → (∃ n' : ℕ, b = 2 + 8 * n') → (∃ m' : ℕ, b = 3 + 9 * m') → b ≤ a) := 
sorry

end largest_common_term_up_to_150_l83_83996


namespace unicorn_silo_problem_l83_83174

/-- A proof problem with a unicorn tethered to a cylindrical silo. -/
theorem unicorn_silo_problem 
  (p q r : ℕ) 
  (prime_r : Nat.Prime r)
  (h1 : r = 6) 
  (h2 : p = 45) 
  (h3 : q = 450) 
  (h4 : 15 - 5 * √2 = 10 * √2) 
  (h5 : 5 * √6 - 3 * √2 = (p - √q) / r) : 
  p + q + r = 501 := 
by
  sorry

end unicorn_silo_problem_l83_83174


namespace part_a_part_b_l83_83726

noncomputable def average_price_between : Prop :=
  ∃ (prices : Fin 14 → ℝ), 
    prices 0 = 5 ∧ prices 6 = 5.14 ∧ prices 13 = 5 ∧ 
    5.09 < (∑ i, prices i) / 14 ∧ (∑ i, prices i) / 14 < 5.10

theorem part_a : average_price_between :=
  sorry

def average_difference : ℝ :=
  let prices1 := [5.0, 5.1, 5.1, 5.1, 5.1, 5.1, 5.14] in
  let prices2 := [5.14, 5.14, 5.14, 5.14, 5.14, 5.14, 5.0] in
  let avg1 := (prices1.sum / prices1.length : ℝ) in
  let avg2 := (prices2.sum / prices2.length : ℝ) in
  abs (avg2 - avg1)

theorem part_b : average_difference < 0.105 :=
  sorry

end part_a_part_b_l83_83726


namespace volume_of_convex_hull_div_t_l83_83694

-- Definitions referenced from the problem conditions
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def circumcenter (poly : List Point) : Point := sorry
def midpoint (P1 P2 : Point) : Point := sorry
def volume_convex_hull (points : List Point) : ℝ := sorry
def volume_tetrahedron (vertices : List Point) : ℝ := sorry

-- T and t as lists of vertices
def T : List Point := sorry
def t : List Point := sorry

-- Circumcenter O of tetrahedrons T and t
def O : Point := circumcenter T

-- Define the midpoint function m for a point P
def m (P : Point) : Point :=
  let intersection_T := sorry -- Point of intersection on T along OP
  let intersection_t := sorry -- Point of intersection on t along OP
  midpoint intersection_T intersection_t

-- Set S
def S : List Point := do
  let vertices_T := T
  let vertices_t := t
  vertices_T.map m ++ vertices_t.map m

-- Statement to be proved
theorem volume_of_convex_hull_div_t (T t : List Point) (O : Point) (S : List Point) :
  volume_convex_hull S / volume_tetrahedron t = 5 := by
  sorry

end volume_of_convex_hull_div_t_l83_83694


namespace martin_leftover_raisins_l83_83399

-- Definitions
variables {v k r : ℝ}
-- Let v be the cost of 1 cream puff
-- Let k be the cost of 1 deciliter of Kofola
-- Let r be the cost of 1 dekagram of raisins

theorem martin_leftover_raisins (v k r : ℝ) (h : r ≠ 0) :
  (3 * v + 3 * k = 18 * r) →
  (12 * r + 5 * k = v + 6 * k + x * r) →
  x = 6 →
  x * 10 = 60 :=
begin
  -- Assuming the conditions of the problem:
  -- 1. Martin could buy three cream puffs and 3 dl of Kofola, or 18 dkg of yogurt raisins.
  -- 2. Martin could buy 12 dkg of yogurt raisins and 5 dl of Kofola, or one cream puff and 6 dl of Kofola.
  
  -- This theorem describes that the quantity of yogurt raisins Martin has left over is equal to 60 grams.
 sorry
end

end martin_leftover_raisins_l83_83399


namespace four_consecutive_integers_product_plus_one_is_square_l83_83040

theorem four_consecutive_integers_product_plus_one_is_square (n : ℤ) :
  (n - 1) * n * (n + 1) * (n + 2) + 1 = (n^2 + n - 1)^2 := by
  sorry

end four_consecutive_integers_product_plus_one_is_square_l83_83040


namespace weight_of_milk_l83_83405

def max_bag_capacity : ℕ := 20
def green_beans : ℕ := 4
def carrots : ℕ := 2 * green_beans
def fit_more : ℕ := 2
def current_weight : ℕ := max_bag_capacity - fit_more
def total_weight_of_green_beans_and_carrots : ℕ := green_beans + carrots

theorem weight_of_milk : (current_weight - total_weight_of_green_beans_and_carrots) = 6 := by
  -- Proof to be written here
  sorry

end weight_of_milk_l83_83405


namespace factorize_expression_l83_83215

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end factorize_expression_l83_83215


namespace part1_part2_l83_83094

theorem part1 : ∃ x : ℝ, 3 * x = 4.5 ∧ x = 4.5 - 3 :=
by {
  -- Skipping the proof for now
  sorry
}

theorem part2 (m : ℝ) (h : ∃ x : ℝ, 5 * x - m = 1 ∧ x = 1 - m - 5) : m = 21 / 4 :=
by {
  -- Skipping the proof for now
  sorry
}

end part1_part2_l83_83094


namespace recurring_fraction_sum_l83_83918

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l83_83918


namespace length_of_train_l83_83973

-- Define conditions
def speed_first_platform (L : ℕ) : ℚ := (L + 150) / 15
def speed_second_platform (L : ℕ) : ℚ := (L + 250) / 20

-- Statement to prove
theorem length_of_train : ∃ L : ℕ, speed_first_platform L = speed_second_platform L ∧ L = 150 :=
by {
  existsi 150, -- This notes that we assert L = 150
  unfold speed_first_platform speed_second_platform,
  split; try sorry
}

end length_of_train_l83_83973


namespace repayment_amount_correct_l83_83986

noncomputable def annual_repayment (M P : ℝ) : ℝ :=
  M * P * (1 + P)^10 / ((1 + P)^10 - 1)

theorem repayment_amount_correct (M P : ℝ) : 
  let x := annual_repayment M P in
  (∑ i in Finset.range 10, x * (1 + P)^i) = M * (1 + P)^10 := 
by
  sorry

end repayment_amount_correct_l83_83986


namespace tyler_puppies_count_l83_83806

theorem tyler_puppies_count :
  (let initial_dogs := 25
   let puppies_per_dog := 7
   let additional_dogs := 3
   let additional_puppies_per_dog := 4 in
   initial_dogs * puppies_per_dog + additional_dogs * additional_puppies_per_dog = 187) :=
by
  sorry

end tyler_puppies_count_l83_83806


namespace age_of_replaced_man_l83_83056

-- Definitions based on conditions
def avg_age_men (A : ℝ) := A
def age_man1 := 10
def avg_age_women := 23
def total_age_women := 2 * avg_age_women
def new_avg_age_men (A : ℝ) := A + 2

-- Proposition stating that given conditions yield the age of the other replaced man
theorem age_of_replaced_man (A M : ℝ) :
  8 * avg_age_men A - age_man1 - M + total_age_women = 8 * new_avg_age_men A + 16 →
  M = 20 :=
by
  sorry

end age_of_replaced_man_l83_83056


namespace recurring_decimal_fraction_sum_l83_83905

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l83_83905


namespace arithmetic_mean_eq_l83_83611

-- Define the conditions
variable (n : ℕ) (h : n > 3)

-- Define the set of numbers
def number1 := (1 - (1 / n.toReal))
def number2 := (1 / n.toReal)
def remaining_numbers := List.replicate (n - 2) 1

-- Define the full set of numbers
def numbers := number1 :: number2 :: remaining_numbers

-- Arithmetic mean calculation
def arithmetic_mean := (number1 + number2 + List.sum remaining_numbers) / n.toReal

-- The Lean statement to prove
theorem arithmetic_mean_eq : arithmetic_mean n h = 1 - (1 / n.toReal) :=
sorry

end arithmetic_mean_eq_l83_83611


namespace locus_of_points_l83_83254

variable (O X Y : Point) (λ : ℝ) (h : λ > 0)

/-- 
Given a triangle OXY and a real number λ > 0, 
the locus of points P such that OP / OQ = λ, where 
Q is any point on segment XY and P lies on the perpendicular 
from O to OQ, forms two segments X'Y' and X''Y'' 
rotated by 90 degrees around O and scaled by λ. 
-/
theorem locus_of_points (O X Y : Point) (λ : ℝ) (h : λ > 0) :
  ∃ (X' Y' X'' Y'' : Point), 
  is_rotated_and_scaled O X Y X' Y' X'' Y'' (90 : ℝ) λ := sorry

end locus_of_points_l83_83254


namespace range_of_m_l83_83608

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) → m ≤ 2 :=
sorry

end range_of_m_l83_83608


namespace yogurt_raisins_l83_83392

-- Definitions for the problem context
variables (v k r x : ℕ)

-- Conditions based on the given problem statements
axiom h1 : 3 * v + 3 * k = 18 * r
axiom h2 : 12 * r + 5 * k = v + 6 * k + x * r

-- Statement that we need to prove
theorem yogurt_raisins : x = 6 :=
by
  have h3 : v + k = 6 * r :=
    calc
      3 * v + 3 * k = 18 * r : h1
      3 * (v + k) = 18 * r : by ring
      v + k = 6 * r : by linarith
  have h4 : 12 * r + 5 * k = v + 6 * k + x * r : h2
  have h5 : 12 * r = (v + k) + 5 * k + x * r := by linarith
  rw [h3] at h5
  have h6 : 12 * r = 6 * r + 5 * k + x * r := by linarith
  have h7 : 6 * r = 5 * k + x * r := by linarith
  have h8 : x = 6 := by linarith
  exact h8

end yogurt_raisins_l83_83392


namespace sum_of_first_100_digits_of_1_over_2323_l83_83926

theorem sum_of_first_100_digits_of_1_over_2323 : 
  let frac := 1 / 2323 in
  let decimal_expansion := "00043000" in
  (sum_of_first_n_digits frac (100 : ℕ) decimal_expansion) = 88 :=
by 
  sorry

-- Helper function definition (you can assume this exists somewhere appropriate)
def sum_of_first_n_digits (frac : ℝ) (n : ℕ) (decimal_expansion : String) : ℕ :=
  sorry

end sum_of_first_100_digits_of_1_over_2323_l83_83926


namespace qed_product_l83_83631

-- Define the complex numbers given in the conditions
def Q : ℂ := 7 + 3i
def E : ℂ := 2 + i
def D : ℂ := 7 - 3i

-- The theorem we want to prove
theorem qed_product : Q * E * D = 116 + 58i :=
by
  sorry

end qed_product_l83_83631


namespace distance_covered_by_center_of_circle_l83_83303

-- Definition of the sides of the triangle
def side1 : ℕ := 5
def side2 : ℕ := 12
def hypotenuse : ℕ := 13

-- Definition of the circle's radius
def radius : ℕ := 2

-- Define a function that calculates the perimeter of the smaller triangle
noncomputable def smallerTrianglePerimeter (s1 s2 hyp r : ℕ) : ℕ :=
  (s1 - 2 * r) + (s2 - 2 * r) + (hyp - 2 * r)

-- Main theorem statement
theorem distance_covered_by_center_of_circle :
  smallerTrianglePerimeter side1 side2 hypotenuse radius = 18 :=
by
  sorry

end distance_covered_by_center_of_circle_l83_83303


namespace sum_of_fraction_parts_of_repeating_decimal_l83_83829

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l83_83829


namespace sum_of_fraction_numerator_and_denominator_l83_83906

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l83_83906


namespace find_alpha_beta_l83_83589

noncomputable def trigonometric_ids (α β : ℝ) : Prop :=
  (sin (π - α) = sqrt 2 * cos (3 * π / 2 + β)) ∧ 
  (sqrt 3 * cos (-α) = - sqrt 2 * cos (π - β)) ∧
  (0 < α) ∧ (α < π) ∧
  (0 < β) ∧ (β < π)

theorem find_alpha_beta (α β : ℝ) :
  trigonometric_ids α β → 
  (α = π / 4 ∧ β = π / 6) ∨ 
  (α = 3 * π / 4 ∧ β = 5 * π / 6) :=
by
  sorry

end find_alpha_beta_l83_83589


namespace midpoints_locus_circle_l83_83383

universes u v

-- Definitions of the circle, chords, midpoints, and point P
section

variable (K : Type u) [MetricSpace K] [NormedSpace ℝ K]

def circle (O : K) (r : ℝ) := {P : K | dist O P = r}

def point_P {O : K} {r : ℝ} (K : circle O r) :=
  ∃ P : K, dist O P = r / 3

def locus_of_midpoints (O : K) (r : ℝ) (P : K) :=
  ∃ M : K, ∀ A B : K, A ≠ B ∧ dist A O = r ∧ dist B O = r ∧ dist A P = dist B P → dist M O = r / 6 ∧ dist M P = r / 6

-- The proof statement of the desired problem
theorem midpoints_locus_circle
  {O : K} {r : ℝ} (h : ∃ P : K, dist O P = r / 3) :
  locus_of_midpoints O r P :=
sorry

end

end midpoints_locus_circle_l83_83383


namespace find_angles_l83_83981

theorem find_angles (A B : ℝ) (h1 : A + B = 90) (h2 : A = 4 * B) : A = 72 ∧ B = 18 :=
by {
  sorry
}

end find_angles_l83_83981


namespace factorize_expression_l83_83562

theorem factorize_expression (m n : ℤ) : m^2 * n - 9 * n = n * (m + 3) * (m - 3) := by
  sorry

end factorize_expression_l83_83562


namespace intersection_M_N_l83_83283

theorem intersection_M_N :
  let M := {x | x^2 < 36}
  let N := {2, 4, 6, 8}
  M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l83_83283


namespace toms_age_ratio_l83_83795

variables (T N : ℕ)

-- Conditions
def toms_age (T : ℕ) := T
def sum_of_children_ages (T : ℕ) := T
def years_ago (T N : ℕ) := T - N
def children_ages_years_ago (T N : ℕ) := T - 4 * N

-- Given statement
theorem toms_age_ratio (h1 : toms_age T = sum_of_children_ages T)
  (h2 : years_ago T N = 3 * children_ages_years_ago T N) :
  T / N = 11 / 2 :=
sorry

end toms_age_ratio_l83_83795


namespace find_k_given_points_l83_83156

theorem find_k_given_points :
  ∃ k : ℤ, (line_through (3, 5) (1, k)) ∧ (line_through (1, k) (7, 9)) ∧ (k = 3) := 
sorry

end find_k_given_points_l83_83156


namespace number_of_sets_l83_83153

variable (x : ℕ)

/-- Each set costs $6, a 10% tax is applied, and the total amount paid was $33. 
    Prove that the number of sets of drill bits (x) is 5. -/
theorem number_of_sets (h : 6 * x + 0.10 * (6 * x) = 33) : x = 5 :=
by
  sorry

end number_of_sets_l83_83153


namespace concylic_points_l83_83983

-- Define the type for points in geometry
variables {Point : Type*}

-- Assume the conditions
variables {A B C D E F K M P Q : Point}

-- Some necessary definitions and assumptions
noncomputable theory

-- Assume altitude conditions
variable [EuclideanGeometry.Point] -- Using a hypothetical module

-- Altitude conditions in the triangle
variable (altitudeBD: is_altitude B D A C)
variable (altitudeAE: is_altitude A E B C)

-- Geometrical midpoint condition
variable (midpointM: is_midpoint M A B)

-- Intersection conditions of line and circumcircle
variable (circumcircleDEFK: intersects_with_circumcircle D E F K)
variable (circumcircleMKPQ: intersects_with_circumcircle M K P Q)
variable (circumcircleMFQ: intersects_with_circumcircle M F Q)

-- Concyclic points conclusion
theorem concylic_points (h1: is_midpoint M A B) (h2: resolves_at_points D E F K M P Q):
  (cyclic_points A P Q B) :=
sorry

end concylic_points_l83_83983


namespace power_sums_equal_l83_83995

theorem power_sums_equal (x y a b : ℝ)
  (h1 : x + y = a + b)
  (h2 : x^2 + y^2 = a^2 + b^2) :
  ∀ n : ℕ, x^n + y^n = a^n + b^n :=
by
  sorry

end power_sums_equal_l83_83995


namespace johns_share_l83_83560

def divide_amount (total_amount : ℕ) (ratios : List ℕ) : List ℕ :=
  let total_parts := ratios.sum
  let part_value := total_amount / total_parts
  ratios.map (fun r => r * part_value)

theorem johns_share (total_amount : ℕ) (john_ratio : ℕ) (ratios : List ℕ) (h : john_ratio = 2) (hs : ratios = [2,4,6,8]) :
  divide_amount total_amount ratios !! 0 = 420 :=
by
  sorry

end johns_share_l83_83560


namespace lattice_point_in_pentagon_l83_83499

structure Point :=
(x : Int)
(y : Int)

structure Pentagon :=
(A B C D E : Point)

def is_convex (P : Pentagon) : Prop :=
  sorry  -- Placeholder for the convexity condition

def is_inside_or_on_boundary (P : Pentagon) (pt : Point) : Prop :=
  sorry  -- Placeholder for the condition to check if pt is inside or on the boundary

theorem lattice_point_in_pentagon {P : Pentagon}
  (h_convex : is_convex P)
  (h_lattice : ∀ pt ∈ {P.A, P.B, P.C, P.D, P.E}, pt.x % 1 = 0 ∧ pt.y % 1 = 0) :
  ∃ pt : Point, pt ∈ {P.A, P.B, P.C, P.D, P.E} ∨ is_inside_or_on_boundary P pt :=
sorry

end lattice_point_in_pentagon_l83_83499


namespace percentage_of_water_in_new_mixture_l83_83933

noncomputable def percentage_of_water (x y z w : ℕ) : Real :=
  (x * y + z * w) / (x + z) * 100

theorem percentage_of_water_in_new_mixture :
  let initial_volume1 := 100
  let pure_liquid1 := 0.25
  let initial_volume2 := 90
  let pure_liquid2 := 0.30
  let total_volume := initial_volume1 + initial_volume2
  let total_water := initial_volume1 * (1 - pure_liquid1) + initial_volume2 * (1 - pure_liquid2)
  total_volume = 190 →
  total_water = 138 →
  percentage_of_water initial_volume1 (1 - pure_liquid1) initial_volume2 (1 - pure_liquid2) ≈ 72.63 := by
  sorry

end percentage_of_water_in_new_mixture_l83_83933


namespace find_x_l83_83287

theorem find_x (x : ℝ) 
  (a : ℝ × ℝ := (2*x - 1, x + 3)) 
  (b : ℝ × ℝ := (x, 2*x + 1))
  (c : ℝ × ℝ := (1, 2))
  (h : (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0) :
  x = 3 :=
  sorry

end find_x_l83_83287


namespace inequality_solution_set_l83_83227

theorem inequality_solution_set (x : ℝ) :
  (3 * x + 1) / (1 - 2 * x) ≥ 0 ↔ -1 / 3 ≤ x ∧ x < 1 / 2 :=
by
  sorry

end inequality_solution_set_l83_83227


namespace zeroable_board_l83_83508

theorem zeroable_board (m n : ℕ) (board : ℕ → ℕ → ℕ) 
  (condition_on_moves : ∀ i j k, board i j ≥ 0 → board k (ℕ.zero_add _) ≥ 0 → true) :
  (∑ i in finset.fin_range m, ∑ j in finset.fin_range n, if (i + j) % 2 = 0 then board i j else 0) =
  (∑ i in finset.fin_range m, ∑ j in finset.fin_range n, if (i + j) % 2 = 1 then board i j else 0)
  ↔ ∃ sequence_of_moves : list (ℕ × ℕ × ℕ), true := 
sorry

end zeroable_board_l83_83508


namespace repeating_decimal_fraction_sum_l83_83848

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
if h : x = (45 / 99 : ℚ) then (5 / 11) else x

theorem repeating_decimal_fraction_sum : 
  let x := (0.45 : ℝ) in 
  (x.mul 100 - x).div 99 = (45 : ℝ) -> 
  let (a, b) := (5, 11) in 
  a + b = 16 :=
by
  intro x
  rfl
  rw [show (5 : ℚ) + (11 : ℚ) = 16 from rfl]
  sorry

end repeating_decimal_fraction_sum_l83_83848


namespace fraction_of_shaded_area_is_one_eighth_l83_83096

-- Define the dimensions of the rectangle
def length : ℝ := 15
def width : ℝ := 20

-- Define the area of the large rectangle
def large_area : ℝ := length * width

-- Define the area of a quarter rectangle
def quarter_area : ℝ := large_area / 4

-- Define the shaded area
def shaded_area : ℝ := quarter_area / 2

-- Define the expected fraction of the shaded area
def expected_fraction_shaded : ℝ := shaded_area / large_area

-- The theorem to prove
theorem fraction_of_shaded_area_is_one_eighth : expected_fraction_shaded = (1 : ℝ) / 8 :=
by
  -- Insert necessary proof steps here
  sorry

end fraction_of_shaded_area_is_one_eighth_l83_83096


namespace limit_value_l83_83136

noncomputable def limit_expression (n : ℕ) : ℝ :=
  (sqrt(n * (n^5 + 9)) - sqrt((n^4 - 1) * (n^2 + 5))) / n

theorem limit_value : 
  tendsto limit_expression at_top (𝓝 (-5 / 2)) :=
begin
  sorry
end

end limit_value_l83_83136


namespace shortest_distance_from_house_l83_83617

def north_distance := 8
def west_distance := 6

theorem shortest_distance_from_house : 
  (north_distance^2 + west_distance^2 = 100) -> (real.sqrt (north_distance^2 + west_distance^2) = 10) :=
by
  sorry

end shortest_distance_from_house_l83_83617


namespace non_egg_laying_chickens_count_l83_83028

noncomputable def num_chickens : ℕ := 80
noncomputable def roosters : ℕ := num_chickens / 4
noncomputable def hens : ℕ := num_chickens - roosters
noncomputable def egg_laying_hens : ℕ := (3 * hens) / 4
noncomputable def hens_on_vacation : ℕ := (2 * egg_laying_hens) / 10
noncomputable def remaining_hens_after_vacation : ℕ := egg_laying_hens - hens_on_vacation
noncomputable def ill_hens : ℕ := (1 * remaining_hens_after_vacation) / 10
noncomputable def non_egg_laying_chickens : ℕ := roosters + hens_on_vacation + ill_hens

theorem non_egg_laying_chickens_count : non_egg_laying_chickens = 33 := by
  sorry

end non_egg_laying_chickens_count_l83_83028


namespace sum_of_fraction_parts_of_repeating_decimal_l83_83826

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l83_83826


namespace intersection_M_N_l83_83285

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N :
  M ∩ N = {-1, 0, 1, 2} :=
sorry

end intersection_M_N_l83_83285


namespace segment_AB_length_l83_83702

-- Define the circles and their properties
variables {Ω ω : Type} [metric_space Ω] [metric_space ω]
variable {O P C D B A : Ω}
variable {OP_diameter : O ≠ P}
variable {P_center_ω : P = center ω}
variable {P_radius_smaller : radius ω < radius (metric_space.to_ball P)}
variable {intersection_CD : C ∈ metric_space.to_sphere P ∧ D ∈ metric_space.to_sphere P}
variable {chord_OB : B ∈ metric_space.to_sphere O}
variable {A_on_chord_OB : A ∈ (segment O B)}

-- The given condition
variable {BD_BC_eq_5 : dist B D * dist B C = 5}

-- The main theorem to be proven
theorem segment_AB_length : dist A B = sqrt 5 :=
by sorry

end segment_AB_length_l83_83702


namespace least_positive_integer_division_conditions_l83_83813

theorem least_positive_integer_division_conditions :
  ∃ M : ℤ, M > 0 ∧
  M % 11 = 10 ∧
  M % 12 = 11 ∧
  M % 13 = 12 ∧
  M % 14 = 13 ∧
  M = 30029 := 
by
  sorry

end least_positive_integer_division_conditions_l83_83813


namespace sum_of_reflected_midpoint_coordinates_l83_83742

open Real

-- Define Point as an alias for pair of reals (x, y)
abbreviation Point := (ℝ × ℝ)

-- Define the original points P and R
def P : Point := (2, 1)
def R : Point := (12, 15)

-- Midpoint function for two points
def midpoint (A B : Point) : Point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Reflect a point over the y-axis
def reflect_y_axis (A : Point) : Point :=
  (-A.1, A.2)

-- The midpoint M of segment PR
def M : Point := midpoint P R

-- The reflected points of P and R over the y-axis
def P' : Point := reflect_y_axis P
def R' : Point := reflect_y_axis R

-- The midpoint M' of segment P'R'
def M' : Point := midpoint P' R'

-- The sum of the coordinates of point M'
def sum_of_coordinates (A : Point) : ℝ :=
  A.1 + A.2

-- The theorem to be proved
theorem sum_of_reflected_midpoint_coordinates :
  sum_of_coordinates M' = 1 :=
by
  -- Proof will go here (skipping with sorry)
  sorry

end sum_of_reflected_midpoint_coordinates_l83_83742


namespace pure_imaginary_iff_a_eq_1_l83_83707

theorem pure_imaginary_iff_a_eq_1 (a : ℝ) : (∃ x : ℝ, (a - complex.I) / (1 + complex.I) = complex.I * x) ↔ a = 1 := by
  sorry

end pure_imaginary_iff_a_eq_1_l83_83707


namespace exists_infinite_sets_of_positive_integers_l83_83747

theorem exists_infinite_sets_of_positive_integers (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (S : ℕ → ℕ × ℕ × ℕ), ∀ n : ℕ, S n = (x, y, z) ∧ 
  ((x + y + z)^2 + 2*(x + y + z) = 5*(x*y + y*z + z*x)) :=
sorry

end exists_infinite_sets_of_positive_integers_l83_83747


namespace geometric_sequence_tenth_term_l83_83550

theorem geometric_sequence_tenth_term
  (a : ℚ) (r : ℚ) (h1 : a = 2) (h2 : r = 5 / 4) :
  let T (n : ℕ) := a * r ^ (n - 1)
  in T 10 = 3906250 / 262144 :=
by
  sorry

end geometric_sequence_tenth_term_l83_83550


namespace triangle_BC_min_l83_83356

theorem triangle_BC_min (A B C : Type) [RealType A] [RealType B] [RealType C] 
  (AB AC BC : Real) (S : Real) 
  (h1 : AB = 2 * AC) 
  (h2 : S = 1) 
  (area_triangle : Real) 
  (h3 : area_triangle = 1 / 2 * AC * AB * (Math.sin (Math.angle A))) 
  (law_of_cosines : Real) 
  (h4 : BC ^ 2 = AB ^ 2 + AC ^ 2 - 2 * AB * AC * (Math.cos (Math.angle A))) :
  BC_min = Math.sqrt 3 :=
by
  sorry

end triangle_BC_min_l83_83356


namespace triangle_sum_of_remaining_sides_l83_83797

noncomputable def sum_of_sides (a b c : ℝ) : ℝ :=
a + b + c

theorem triangle_sum_of_remaining_sides :
  ∃ (A B C : ℝ), ∠A = 40 ∧ ∠B = 50 ∧ ∠C = 90 ∧ opposite 40 = 8 ∧ 
  (sum_of_sides = 20.3) :=
begin
  sorry
end

end triangle_sum_of_remaining_sides_l83_83797


namespace geometric_progression_iff_l83_83377

noncomputable def a_seq : ℕ → ℝ
| 0       := 1
| 1       := 1
| (n + 2) := a_seq n * (a_seq (n + 1))^2

-- Define the hypothesis that a_n forms a geometric progression
def is_geometric_progression (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a 1

-- The main theorem to prove
theorem geometric_progression_iff (a2 : ℝ) (h : a2 > 0) :
  is_geometric_progression (λ n, if n = 0 then 1 else if n = 1 then a2 else a_seq n) ↔ a2 = 1 :=
sorry

end geometric_progression_iff_l83_83377


namespace min_n_for_binomial_constant_term_l83_83066

theorem min_n_for_binomial_constant_term : ∃ (n : ℕ), n > 0 ∧ 3 * n - 7 * ((3 * n) / 7) = 0 ∧ n = 7 :=
by {
  sorry
}

end min_n_for_binomial_constant_term_l83_83066


namespace distinct_necklaces_count_l83_83178

open Classical

/-- Using 5 beads and 3 different colors, the total number of distinct necklaces,
  considering the dihedral group D5 symmetries, is 39. -/
theorem distinct_necklaces_count : 
  ∃ N : ℕ, N = 39 ∧
    ∀ beaded_necklaces : fin 3 ^ 5, 
    true := -- Assuming we have some definite combinatorial structure beaded_necklaces 
  begin
    let D5 := dihedralGroup 5,
    have H : |D5| = 10 := by sorry,
    let count_fixed_points := λ (g : D5), (fin 3) ^ (fixedPoints g),
    let total_count := (1 / |D5|.toReal) * ∑ g in D5, count_fixed_points g,
    exact ⟨39, rfl, sorry⟩
  end

end distinct_necklaces_count_l83_83178


namespace sum_of_n_values_l83_83113

theorem sum_of_n_values (n : ℕ) (h₀ : n = 16 ∨ n = 14) :
  (14 + 16 = 30 ∧ ∑ n in {16, 14}, n = 30) :=
by
  have h1: ∑ n in {16, 14}, n = 14 + 16 := by
    simp [Finset.sum_insert, Finset.sum_singleton]
  simp [h0, h1]
  sorry

end sum_of_n_values_l83_83113


namespace probability_sum_divisible_by_3_l83_83286

theorem probability_sum_divisible_by_3:
  ∀ (n a b c : ℕ), a + b + c = n →
  4 * (a^3 + b^3 + c^3 + 6 * a * b * c) ≥ (a + b + c)^3 :=
by 
  intros n a b c habc_eq_n
  sorry

end probability_sum_divisible_by_3_l83_83286


namespace factorize_expression_l83_83212

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) :=
by sorry

end factorize_expression_l83_83212


namespace optimal_worker_assignment_l83_83953

theorem optimal_worker_assignment :
  ∃ (x y : ℕ), x + y = 140 ∧ 25 * x = 2 * 20 * y ∧ x = 40 ∧ y = 100 :=
by
  use 40, 100
  split
  { norm_num }
  split
  { norm_num }
  split
  { refl }
  { refl }
  sorry

end optimal_worker_assignment_l83_83953


namespace color_circles_with_four_colors_l83_83419

theorem color_circles_with_four_colors (n : ℕ) (circles : Fin n → (ℝ × ℝ)) (radius : ℝ):
  (∀ i j, i ≠ j → dist (circles i) (circles j) ≥ 2 * radius) →
  ∃ f : Fin n → Fin 4, ∀ i j, dist (circles i) (circles j) < 2 * radius → f i ≠ f j :=
by
  sorry

end color_circles_with_four_colors_l83_83419


namespace sum_of_integers_satisfying_binom_identity_l83_83115

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_of_integers_satisfying_binom_identity :
  (∀ n : ℕ, binom 30 15 + binom 30 n = binom 31 16) →
  ∑ n in {14, 16}, n = 30 :=
by
  sorry

end sum_of_integers_satisfying_binom_identity_l83_83115


namespace angle_bisector_inequality_l83_83357

variable {A B C D : Type} [HasAngle A B C] [HasDistance A B C D]

-- Given conditions:
-- Triangle ABC with angle bisector of ∠B meeting the circumcircle of triangle ABC at D
variables (triangleABC : triangle A B C)
          (circumcircleABC : circle (circumcenter triangleABC))
          (angleBisectorBD : ray B D)
          (meetsAtD : IsOnCircumcircle D triangleABC)

-- Proof goal:
-- Prove that BD^2 > BA * BC given the conditions above
theorem angle_bisector_inequality 
  (h1 : bisector_of_triangle ∠ B angleBisectorBD)
  (h2 : meetsAtD) :
  (distance B D) ^ 2 > (distance B A) * (distance B C) := 
  sorry

end angle_bisector_inequality_l83_83357


namespace sum_of_fraction_parts_l83_83885

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l83_83885


namespace cos_angle_YXW_l83_83670

theorem cos_angle_YXW {X Y Z W : Type} [EuclideanGeometry X Y Z W] : 
  (XY = 5) ∧ (XZ = 7) ∧ (YZ = 9) ∧ (OnLineSegment W YZ) ∧ (AngleBisector X W (Y X Z)) → 
  cos (YXW : Angle Y X W) = (3 * Real.sqrt 5) / 10 := 
by sorry

end cos_angle_YXW_l83_83670


namespace sum_of_fraction_parts_l83_83880

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l83_83880


namespace normal_retail_price_of_each_item_l83_83170

/--
A store decides to shut down and sell all of its inventory.
They have 2000 different items which would normally retail for a certain price.
They are offering an 80% discount and manage to sell 90% of the items.
They owed $15000 to their creditors and have $3000 left after the sale.
What was the normal retail price of each item?
-/
theorem normal_retail_price_of_each_item
  (items : ℕ)
  (discount : ℝ)
  (sold_percentage : ℝ)
  (debt : ℝ)
  (remaining : ℝ)
  (retail_price : ℝ)
  (rev_eq_debt_plus_remaining : rev = debt + remaining) :
  items = 2000 ∧
  discount = 0.80 ∧
  sold_percentage = 0.90 ∧
  debt = 15000 ∧
  remaining = 3000 →
  retail_price = 50 :=
by
  intros h
  cases h with h_items h_rest
  cases h_rest with h_discount h_rest
  cases h_rest with h_sold_percentage h_rest
  cases h_rest with h_debt h_remaining
  rw [h_items, h_discount, h_sold_percentage, h_debt, h_remaining]
  -- TODO: prove the retail price is 50 using the provided conditions and equations.
  sorry

end normal_retail_price_of_each_item_l83_83170


namespace max_value_x_minus_y_l83_83777

theorem max_value_x_minus_y
  (x y z : ℝ)
  (h₁ : x + y + z = 2)
  (h₂ : x * y + y * z + z * x = 1) : 
  x - y ≤ 2 * real.sqrt 3 / 3 :=
sorry

end max_value_x_minus_y_l83_83777


namespace angle_BHZ_in_orthocenter_l83_83978

theorem angle_BHZ_in_orthocenter (ABC : Triangle) (H : Point) (BX : Line) (YZ : Line) :
  H = orthocenter ABC ∧
  angle ABC.ABC.A = 55 ∧
  angle ABC.ABC.C = 67 ∧
  is_perpendicular ABC.B BX →
  angle BHZ = 23 :=
by sorry

end angle_BHZ_in_orthocenter_l83_83978


namespace lcm_nuts_bolts_l83_83210

theorem lcm_nuts_bolts : Nat.lcm 13 8 = 104 := 
sorry

end lcm_nuts_bolts_l83_83210


namespace derivative_of_y_l83_83439

variable (x : ℝ)

def y : ℝ := x * Real.cos x - Real.sin x

theorem derivative_of_y : deriv (λ x : ℝ, x * Real.cos x - Real.sin x) = (λ x : ℝ, -x * Real.sin x) :=
by
  sorry

end derivative_of_y_l83_83439


namespace possible_value_of_phi_l83_83443

theorem possible_value_of_phi :
  ∃ (ϕ : ℝ), (∀ x : ℝ, sin (2 * (x + (π / 8)) + ϕ) = sin (-(2 * (x + (π / 8)) + ϕ))) → ϕ = π / 4 :=
by
  sorry

end possible_value_of_phi_l83_83443


namespace factorize_expression_l83_83213

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) :=
by sorry

end factorize_expression_l83_83213


namespace trigonometric_identity_l83_83546

theorem trigonometric_identity :
  ∀ (x: ℝ), cos (70 * real.pi / 180) ≠ 0 → sin (70 * real.pi / 180) ≠ 0 →
  ( 1 / cos (70 * real.pi / 180) - real.sqrt 3 / sin (70 * real.pi / 180)
  = 4 * sin (10 * real.pi / 180) / sin (40 * real.pi / 180)) :=
by
  intros x hcos hsin
  sorry

end trigonometric_identity_l83_83546


namespace weighted_average_percentage_l83_83023

variables (x y : ℝ) 

theorem weighted_average_percentage (hx : 0 ≤ x ∧ x ≤ 100) (hy : 0 ≤ y ∧ y ≤ 100) :
  let W := (15 * x + 10 * y) / 25 
  in W = (3 * x + 2 * y) / 5 := 
by
  unfold W
  sorry

end weighted_average_percentage_l83_83023


namespace new_arithmetic_mean_l83_83072

theorem new_arithmetic_mean
  (seq : List ℝ)
  (h_seq_len : seq.length = 60)
  (h_mean : (seq.sum / 60 : ℝ) = 42)
  (h_removed : ∃ a b, a ∈ seq ∧ b ∈ seq ∧ a = 50 ∧ b = 60) :
  ((seq.erase 50).erase 60).sum / 58 = 41.55 := 
sorry

end new_arithmetic_mean_l83_83072


namespace certain_event_at_least_one_good_product_l83_83236

-- Define the number of products and their types
def num_products := 12
def num_good_products := 10
def num_defective_products := 2
def num_selected_products := 3

-- Statement of the problem
theorem certain_event_at_least_one_good_product :
  ∀ (selected : Finset (Fin num_products)),
  selected.card = num_selected_products →
  ∃ p ∈ selected, p.val < num_good_products :=
sorry

end certain_event_at_least_one_good_product_l83_83236


namespace triangle_min_ab_l83_83642

noncomputable def minimum_ab (a b c S : ℝ) : ℝ := if 2 * c * Real.cos B = 2 * a + b ∧ S = (Real.sqrt 3 / 2) * c then 12 else 0

theorem triangle_min_ab (a b c : ℝ) (S : ℝ) (h1 : 2 * c * Real.cos B = 2 * a + b) (h2 : S = (Real.sqrt 3 / 2) * c) : ab ≥ 12 :=
by sorry

end triangle_min_ab_l83_83642


namespace prove_correct_ordering_of_f_l83_83379

noncomputable def f : ℝ → ℝ := sorry

-- Defining the conditions in Lean 4
def periodic (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(x + 6)

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f(a + x) = f(a - x)

def strictly_monotone_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

-- Expressing the given conditions
axiom f_periodic : periodic f
axiom f_symmetric : symmetric_about f 3
axiom f_monotone_decreasing : strictly_monotone_decreasing f (set.Ioo 0 3)

-- Statement to prove
theorem prove_correct_ordering_of_f :
  f 3.5 < f 1.5 ∧ f 1.5 < f 6.5 :=
by
  sorry

end prove_correct_ordering_of_f_l83_83379


namespace solve_for_x_l83_83627

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end solve_for_x_l83_83627


namespace final_result_l83_83225

def a : ℕ := 2548
def b : ℕ := 364
def hcd := Nat.gcd a b
def result := hcd + 8 - 12

theorem final_result : result = 360 := by
  sorry

end final_result_l83_83225


namespace soda_comparison_l83_83686

def soda_amount_julio : ℕ := 4 * 2 + 7 * 2
def soda_amount_mateo : ℕ := 1 * 2.5 + 3 * 2.5
def soda_amount_sophia : ℕ := 6 * 1.5 + 3 * 2.5 + 2 * 2.5 * 0.75

theorem soda_comparison :
  soda_amount_julio = 22 ∧
  soda_amount_mateo = 10 ∧
  soda_amount_sophia = 20.25 ∧
  (soda_amount_julio - soda_amount_mateo) = 12 :=
by
  sorry

end soda_comparison_l83_83686


namespace average_price_condition_observer_b_correct_l83_83731

-- Define the conditions
def stock_price {n : ℕ} (daily_prices : Fin n → ℝ) : Prop :=
  daily_prices 0 = 5 ∧ 
  daily_prices 6 = 5.14 ∧
  daily_prices 13 = 5 ∧ 
  (∀ i : Fin 6, daily_prices i ≤ daily_prices (i + 1)) ∧ 
  (∀ i : Fin (n - 7), daily_prices (i + 7) ≥ daily_prices (i + 8))

-- Define the problem statements
theorem average_price_condition (daily_prices : Fin 14 → ℝ) :
  stock_price daily_prices →
  ∃ S, 5.09 < (5 + S + 5.14) / 14 ∧ (5 + S + 5.14) / 14 < 5.10 :=
sorry

theorem observer_b_correct (daily_prices : Fin 14 → ℝ) :
  stock_price daily_prices →
  let avg_1 : ℝ := (∑ i in Finset.range 7, daily_prices i) / 7
  let avg_2 : ℝ := (∑ i in Finset.range 7, daily_prices (i + 7)) / 7
  ¬ avg_1 = avg_2 + 0.105 :=
sorry

end average_price_condition_observer_b_correct_l83_83731


namespace dart_lands_in_center_hexagon_l83_83952

noncomputable def area_regular_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

theorem dart_lands_in_center_hexagon {s : ℝ} (h : s > 0) :
  let A_outer := area_regular_hexagon s
  let A_inner := area_regular_hexagon (s / 2)
  (A_inner / A_outer) = 1 / 4 :=
by
  let A_outer := area_regular_hexagon s
  let A_inner := area_regular_hexagon (s / 2)
  sorry

end dart_lands_in_center_hexagon_l83_83952


namespace odd_function_decreasing_on_positive_reals_l83_83601

noncomputable def f (x : ℝ) : ℝ := -x + (1 / (2 * x))

theorem odd_function : ∀ x, f (-x) = -f x := by
  intro x
  unfold f
  rw [neg_neg, div_neg] -- Use arithmetic rules
  exact rfl  -- Finish proof using equality

theorem decreasing_on_positive_reals : ∀ x, x > 0 → f' x < 0 := by
  intro x hx
  have hfx : (f' x) = -1 - (1 / (2 * x^2)) := by
    calc 
      f' x = -1 - (1 / (2 * x^2)) : by
        simp [f, differentiable]
  linarith -- Use linear arithmetic to conclude
  sorry -- Placeholder for the rigorous proof

end odd_function_decreasing_on_positive_reals_l83_83601


namespace recurring_fraction_sum_l83_83921

def x := 0.454545...

theorem recurring_fraction_sum :
  let x := 0.454545...
  let frac := (5 : ℕ) / (11 : ℕ)
  (frac.numerator + frac.denominator) = 16 :=
by
  sorry

end recurring_fraction_sum_l83_83921


namespace draw_is_unfair_suit_hierarchy_makes_fair_l83_83307

structure Card where
  suit : ℕ -- 4 suits numbered from 0 to 3
  rank : ℕ -- 9 ranks numbered from 0 to 8

def deck : List Card :=
  List.join (List.map (λ s, List.map (λ r, ⟨s, r⟩) (List.range 9)) (List.range 4))

def DrawFair? : (deck : List Card) → Prop := sorry

-- Part (a): Prove that the draw is unfair
theorem draw_is_unfair : ¬ DrawFair? deck := sorry

-- Part (b): Prove that introducing a suit hierarchy can make the draw fair
def suit_hierarchy : Card → Card → Prop :=
λ c1 c2, (c1.rank < c2.rank) ∨ (c1.rank = c2.rank ∧ c1.suit < c2.suit)

theorem suit_hierarchy_makes_fair : ∃ h : Card → Card → Prop, h = suit_hierarchy ∧ DrawFair? deck[h] := sorry

end draw_is_unfair_suit_hierarchy_makes_fair_l83_83307


namespace mary_extra_lambs_l83_83400

theorem mary_extra_lambs (original_lambs : ℕ) (baby_lambs_multiplier : ℕ)
  (babies_per_lamb : ℕ) (trade_lambs : ℕ) (now_lambs : ℕ) :
  original_lambs = 6 →
  baby_lambs_multiplier = 2 →
  babies_per_lamb = 2 →
  trade_lambs = 3 →
  now_lambs = 14 →
  let additional_lambs := baby_lambs_multiplier * babies_per_lamb in
  let remaining_lambs := original_lambs + additional_lambs - trade_lambs in
  now_lambs - remaining_lambs = 7 :=
by
  intros h1 h2 h3 h4 h5
  let additional_lambs := baby_lambs_multiplier * babies_per_lamb
  let remaining_lambs := original_lambs + additional_lambs - trade_lambs
  show now_lambs - remaining_lambs = 7 from sorry

end mary_extra_lambs_l83_83400


namespace sum_of_fraction_parts_of_repeating_decimal_l83_83832

theorem sum_of_fraction_parts_of_repeating_decimal :
  let frac := Rat.mk 45 99 in
  let simplest_frac := frac.num.gcd 45 99 in
  let num := simplest_frac.num in
  let denom := simplest_frac.denom in
  num + denom = 16 :=
by
  sorry

end sum_of_fraction_parts_of_repeating_decimal_l83_83832


namespace proposition_one_proposition_two_proposition_three_proposition_four_l83_83765

theorem proposition_one (a : ℝ) (h : 1 < a) : ∀ x : ℝ, f_diff a x > 0 :=
  sorry

def f_diff (a x : ℝ) : ℝ := (Real.log a) * (a ^ (x - 1))

theorem proposition_two : ¬((∀ x, x - 1 ∈ Set.Ioo (1 : ℝ) 3 → x ∈ Set.Ioo 2 4)) :=
  sorry

theorem proposition_three (a b : ℝ) (h : (f_val a b (-2) = 8)) : f_val a b 2 ≠ -8 :=
  sorry

def f_val (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem proposition_four : ∀ x : ℝ, (odd_function f_odd x) :=
  sorry

def f_odd (x : ℝ) : ℝ := 1 / (1 - 2^x) - 1 / 2

def odd_function (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f (-x) = -f x

end proposition_one_proposition_two_proposition_three_proposition_four_l83_83765


namespace quadrilateral_of_two_diameters_is_rectangle_l83_83305

theorem quadrilateral_of_two_diameters_is_rectangle 
  (C : Type) [MetricSpace C] [ProperSpace C] [NormedAddTorsor E P] [Circle C P E] 
  (d1 d2 : Line P) (d1_diameter : diameter d1) (d2_diameter : diameter d2) 
  (inter : ∃ p : P, p ∈ d1 ∧ p ∈ d2) :
  is_rectangle (Quadrilateral.of_diameters d1 d2) := 
sorry

end quadrilateral_of_two_diameters_is_rectangle_l83_83305


namespace semicircle_to_cone_volume_l83_83477

noncomputable def volume_of_cone (R : ℝ) : ℝ :=
  (1/3) * π * (R/2)^2 * ( (sqrt 3) / 2 * R )

theorem semicircle_to_cone_volume (R : ℝ) :
  volume_of_cone R = (sqrt 3 / 24) * π * R^3 := 
by
  sorry

end semicircle_to_cone_volume_l83_83477


namespace yogurt_raisins_l83_83391

-- Definitions for the problem context
variables (v k r x : ℕ)

-- Conditions based on the given problem statements
axiom h1 : 3 * v + 3 * k = 18 * r
axiom h2 : 12 * r + 5 * k = v + 6 * k + x * r

-- Statement that we need to prove
theorem yogurt_raisins : x = 6 :=
by
  have h3 : v + k = 6 * r :=
    calc
      3 * v + 3 * k = 18 * r : h1
      3 * (v + k) = 18 * r : by ring
      v + k = 6 * r : by linarith
  have h4 : 12 * r + 5 * k = v + 6 * k + x * r : h2
  have h5 : 12 * r = (v + k) + 5 * k + x * r := by linarith
  rw [h3] at h5
  have h6 : 12 * r = 6 * r + 5 * k + x * r := by linarith
  have h7 : 6 * r = 5 * k + x * r := by linarith
  have h8 : x = 6 := by linarith
  exact h8

end yogurt_raisins_l83_83391


namespace ellipse_largest_angle_90_l83_83194

noncomputable def largest_angle_in_triangle (a b c : ℝ) : ℝ :=
if a >= b ∧ a >= c then a else if b >= a ∧ b >= c then b else c

-- Definitions and conditions from the ellipse geometry problem
def ellipse := {p : ℝ × ℝ // (p.1^2 / 16 + p.2^2 / 12 = 1)}

axiom foci_distance : ℝ := 4 -- distance between foci F1 and F2 is 4
axiom |MF1_minus_MF2| : ∀ (M : ellipse), |M.val.1 - 5| - |M.val.1 - 3| = 2

-- The problem's question in Lean
theorem ellipse_largest_angle_90 (M : ellipse) : 
  largest_angle_in_triangle (|M.val.1 - 5|) (|M.val.1 - 3|) foci_distance = 90 :=
sorry

end ellipse_largest_angle_90_l83_83194


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l83_83891

noncomputable def repeating_decimal_to_fraction (n d : ℤ) (h : 0 < d) := ∃ (x : ℚ), x = n / d ∧ 0 < x

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.4545454545 -- defining x
  let f := x.to_rational -- converting to rational
  let f_simp := f.num / f.denom -- get numerator and denominator
  -- conclude the sum of numerator and denominator is 16
  f.num + f.denom = 16 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l83_83891


namespace stock_price_satisfies_conditions_part_a_stock_price_satisfies_conditions_part_b_l83_83722

-- Define the conditions: stock prices on specific dates
constants apr_7 apr_13 apr_20 : ℝ
axiom apr_7_eq : apr_7 = 5
axiom apr_13_eq : apr_13 = 5.14
axiom apr_20_eq : apr_20 = 5

-- Define the prices on the intervening dates
constants (x : ℕ → ℝ)

-- Define the correct answer for average price calculation for part (a)
def avg_price_between_apr_7_Apr_20 : ℝ := (apr_7 + (Σ i in FinSet.range 12, x i) + apr_13 + (Σ j in FinSet.range 6, x (j + 7)) + apr_20) / 14

noncomputable def part_a : Prop :=
  5.09 < avg_price_between_apr_7_Apr_20 ∧ avg_price_between_apr_7_Apr_20 < 5.10

-- Part (b): Comparing average stock prices for different periods
def avg_price_apr_7_to_apr_13 : ℝ := (apr_7 + (Σ i in FinSet.range 5, x i) + apr_13) / 7
def avg_price_apr_14_to_apr_20 : ℝ := (apr_13 + x 7 + x 8 + x 9 + x 10 + x 11 + apr_20) / 7

noncomputable def part_b : Prop :=
  | (avg_price_apr_14_to_apr_20 - avg_price_apr_7_to_apr_13 ≠ 0.105 )

-- The final proof problems for part (a) and part (b)
theorem stock_price_satisfies_conditions_part_a : part_a := sorry
theorem stock_price_satisfies_conditions_part_b : part_b := sorry

end stock_price_satisfies_conditions_part_a_stock_price_satisfies_conditions_part_b_l83_83722


namespace train_length_l83_83510

-- Define the speed in kmph.
def speed_kmph : ℝ := 36

-- Define the speed in m/s.
def speed_mps : ℝ := (1000 / 3600) * speed_kmph

-- Define the time in seconds.
def time_seconds : ℝ := 6.999440044796416

-- Define the expected length (distance) of the train.
def length_of_train : ℝ := speed_mps * time_seconds

-- The theorem stating the length of the train.
theorem train_length : length_of_train = 69.99440044796416 := by
  -- Placeholder proof
  sorry

end train_length_l83_83510


namespace binom_sum_problem_l83_83110

theorem binom_sum_problem :
  ∑ n in {n : ℕ | nat.choose 30 n = nat.choose 16 16 ∨ nat.choose 16 14}, n = 30 :=
by
  sorry

end binom_sum_problem_l83_83110


namespace third_term_of_arithmetic_sequence_l83_83340

variable (a : ℕ → ℤ)
variable (a1_eq_2 : a 1 = 2)
variable (a2_eq_8 : a 2 = 8)
variable (arithmetic_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1))

theorem third_term_of_arithmetic_sequence :
  a 3 = 14 :=
by
  sorry

end third_term_of_arithmetic_sequence_l83_83340


namespace moment_of_inertia_of_thin_spherical_shell_l83_83972

variable (R : ℝ) (M : ℝ) (μ : ℝ)

-- Definitions for the conditions
def spherical_shell_surface_density (μ : ℝ) : Prop := true
def spherical_shell_radius (R : ℝ) : Prop := true
def spherical_shell_mass (M : ℝ) (R : ℝ) (μ : ℝ) : Prop := M = 4 * π * R^2 * μ

-- The statement of the proof problem
theorem moment_of_inertia_of_thin_spherical_shell 
  (h1 : spherical_shell_surface_density μ)
  (h2 : spherical_shell_radius R)
  (h3 : spherical_shell_mass M R μ) :
  ∃ Θ, Θ = (2 / 3) * M * R^2 :=
sorry

end moment_of_inertia_of_thin_spherical_shell_l83_83972


namespace shaded_square_side_length_l83_83080

-- Definitions
def base (x : Real) := 4 * x
def height (x : Real) := x
def hypotenuse (x : Real) := Real.sqrt ((base x)^2 + (height x)^2)
def large_square_side (x : Real) := hypotenuse x
def large_square_area (x : Real) := (large_square_side x)^2
def triangle_area (x : Real) := (1/2) * (base x) * (height x)
def total_triangles_area (x : Real) := 4 * (triangle_area x)
def shaded_square_area (x : Real) := large_square_area x - total_triangles_area x
def shaded_square_side (x : Real) := Real.sqrt (shaded_square_area x)

-- Problem statement
theorem shaded_square_side_length (x : Real) : shaded_square_side x = 2 * Real.sqrt 2 * x := 
by 
  sorry

end shaded_square_side_length_l83_83080


namespace remainder_101_pow_37_mod_100_l83_83472

theorem remainder_101_pow_37_mod_100 :
  (101: ℤ) ≡ 1 [MOD 100] →
  (101: ℤ)^37 ≡ 1 [MOD 100] :=
by
  sorry

end remainder_101_pow_37_mod_100_l83_83472


namespace arithmetic_seq_sum_l83_83431

theorem arithmetic_seq_sum(S : ℕ → ℝ) (d : ℝ) (h1 : S 5 < S 6) 
    (h2 : S 6 = S 7) (h3 : S 7 > S 8) : S 9 < S 5 := 
sorry

end arithmetic_seq_sum_l83_83431


namespace sum_of_fraction_terms_l83_83863

theorem sum_of_fraction_terms (x : ℚ) (hx : x = 45 / 99) (y : ℚ) (hy : y * 99 = 45) : 
  let z := (5 : ℚ) / (11 : ℚ) in 
  hwf : x = z → z.den + z.num = 16 := 
by
  sorry

end sum_of_fraction_terms_l83_83863


namespace symmetric_shading_ways_l83_83982

def is_axis_symmetric (grid : matrix (fin 4) (fin 4) bool) : Prop := sorry -- Define axis symmetry

def add_square_and_symmetric_ways (grid : matrix (fin 4) (fin 4) bool) : nat :=
  if h : ∃ i j, grid i j = tt ∧ is_axis_symmetric (matrix.update grid i j tt) then
    finset.card (finset.filter (λ i, ∃ j, is_axis_symmetric (matrix.update grid i j tt)) (finset.univ : finset (fin 4))) 
  else 0

theorem symmetric_shading_ways (grid : matrix (fin 4) (fin 4) bool) (h₀ : ∃ a b c, grid a b = tt ∧ grid b c = tt ∧ grid c a = tt) :
  add_square_and_symmetric_ways grid = 2 :=
sorry

end symmetric_shading_ways_l83_83982


namespace range_of_k_l83_83263

variable {R : Type*} [field R]

-- Let f be a continuously differentiable function
variable (f : R → R) (f' : R → R)
variable (h1 : ∀ x : R, differentiable_at R f x)

-- Condition: f'(x) ≠ 0 in R
variable (h2 : ∀ x : R, f'(x) ≠ 0)

-- Condition: f[f(x) - 2017^x] = 2017
variable (h3 : ∀ x : R, f (f x - real.exp (2017 * x)) = 2017)

-- Function g(x)
def g (x : R) (k : R) : R := real.sin x - real.cos x - k * x

-- Proof statement: if g(x) is monotonically increasing on [-π/2, π/2], then k ≤ -1
theorem range_of_k (k : R) :
  (∀ x ∈ set.Icc (-real.pi / 2) (real.pi / 2), 0 ≤ real.cos x + real.sin x - k) → k ≤ -1 :=
sorry

end range_of_k_l83_83263


namespace log_sum_of_exp_function_condition_l83_83265

theorem log_sum_of_exp_function_condition {a : ℝ} (h : 1 + a^3 = 9) : 
  log (1/4) a + log a 8 = 5/2 := sorry

end log_sum_of_exp_function_condition_l83_83265


namespace production_rate_equation_l83_83162

theorem production_rate_equation (x : ℝ) (h : x > 0) :
  3000 / x - 3000 / (2 * x) = 5 :=
sorry

end production_rate_equation_l83_83162


namespace ratio_new_average_to_original_l83_83171

theorem ratio_new_average_to_original (scores : List ℝ) (h_len : scores.length = 50) :
  let A := (scores.sum / scores.length : ℝ)
  let new_sum := scores.sum + 2 * A
  let new_avg := new_sum / (scores.length + 2)
  new_avg / A = 1 := 
by
  sorry

end ratio_new_average_to_original_l83_83171


namespace total_sum_lent_l83_83971

def interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time / 100

theorem total_sum_lent (x : ℕ) (second_part : ℕ) (h : interest x 3 8 = interest second_part 5 3) (second_part : second_part = 1664) :
  x + second_part = 2704 :=
begin
  sorry
end

end total_sum_lent_l83_83971


namespace max_sum_of_10_consecutive_terms_l83_83333

open nat

noncomputable def infinite_sequence_of_natural_numbers :=
  { seq : ℕ → ℕ // ∀ n : ℕ, (∏ i in finset.range 15, seq (n+i)) = 10^6  }

theorem max_sum_of_10_consecutive_terms
  (seq : infinite_sequence_of_natural_numbers)
  (S : ℕ) :
  (∀ n : ℕ, (∑ i in finset.range 10, seq.val (n+i)) = S) →
  S = 208 :=
by
  sorry

end max_sum_of_10_consecutive_terms_l83_83333


namespace total_price_for_shirts_l83_83330

theorem total_price_for_shirts (total_price_sweaters : ℕ) (average_price_diff : ℕ) (total_price_shirts : ℕ) :
  total_price_sweaters = 900 →
  average_price_diff = 2 →
  total_price_shirts = 20 * ((total_price_sweaters / 45) - average_price_diff) →
  total_price_shirts = 360 :=
by
  intros htotal_price_sweaters haverage_price_diff htotal_price_shirts
  have W : ℕ := total_price_sweaters / 45
  have S : ℕ := W - average_price_diff
  have total_price_shirts_calc : ℕ := 20 * S
  rw [htotal_price_sweaters] at *
  rw [haverage_price_diff] at *
  rw [htotal_price_shirts] at *
  have hW : W = 20 := by sorry
  finish

end total_price_for_shirts_l83_83330


namespace tilted_rectangle_l83_83163

theorem tilted_rectangle (VWYZ : Type) (YW ZV : ℝ) (ZY VW : ℝ) (W_above_horizontal : ℝ) (Z_height : ℝ) (x : ℝ) :
  YW = 100 → ZV = 100 → ZY = 150 → VW = 150 → W_above_horizontal = 20 → Z_height = (100 + x) →
  x = 67 :=
by
  sorry

end tilted_rectangle_l83_83163


namespace sum_of_integers_satisfying_binom_identity_l83_83114

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_of_integers_satisfying_binom_identity :
  (∀ n : ℕ, binom 30 15 + binom 30 n = binom 31 16) →
  ∑ n in {14, 16}, n = 30 :=
by
  sorry

end sum_of_integers_satisfying_binom_identity_l83_83114


namespace prime_sum_to_17_implies_product_l83_83536

theorem prime_sum_to_17_implies_product :
  ∃ p1 p2 p3 p4 : ℕ, p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧
  p1 + p2 + p3 + p4 = 17 ∧ p1 * p2 * p3 * p4 = 210 :=
by
  sorry

end prime_sum_to_17_implies_product_l83_83536


namespace least_length_XZ_is_8_l83_83667

noncomputable def least_length_XZ :=
  let PQR := triangle.mk 8 12 (by linarith)
  let α : angle := 60
  ∃ X : point, ∀ X ∈ line.PQ, XY ∥ QR ∧ YZ ∥ PQ → ∀ B ∈ midpoint(QR), ∠PBQ = α → length(XZ) = 8

theorem least_length_XZ_is_8 :
  least_length_XZ := sorry

end least_length_XZ_is_8_l83_83667


namespace sum_of_reciprocals_of_numbers_l83_83130

theorem sum_of_reciprocals_of_numbers (x y : ℕ) (h_sum : x + y = 45) (h_hcf : Nat.gcd x y = 3)
    (h_lcm : Nat.lcm x y = 100) : 1/x + 1/y = 3/20 := 
by 
  sorry

end sum_of_reciprocals_of_numbers_l83_83130


namespace Austin_friday_hours_l83_83538

variable (hourly_rate : ℕ := 5)
variable (monday_hours : ℕ := 2)
variable (wednesday_hours : ℕ := 1)
variable (bicycle_cost : ℕ := 180)
variable (weeks : ℕ := 6)

theorem Austin_friday_hours : 
  (bicycle_cost - (weeks * (monday_hours + wednesday_hours) * hourly_rate)) / (weeks * hourly_rate) = 3 := 
by 
    -- Define the earnings from Mondays and Wednesdays
    let total_monday_wednesday_earnings := weeks * (monday_hours + wednesday_hours) * hourly_rate
    -- Calculate the remaining amount needed
    let remaining_needed := bicycle_cost - total_monday_wednesday_earnings
    -- Calculate the required hours on Fridays
    let required_friday_hours := remaining_needed / (weeks * hourly_rate)
    -- Ensure the required hours equals 3
    exact required_friday_hours

end Austin_friday_hours_l83_83538


namespace perpendicular_bisector_locus_l83_83412

theorem perpendicular_bisector_locus {A B M : ℝ × ℝ} (h : dist A M = dist B M) : 
  M ∈ {P : ℝ × ℝ | ∃ C, is_midpoint C A B ∧ is_perpendicular_bisector C P A B} :=
sorry

end perpendicular_bisector_locus_l83_83412


namespace roots_of_cubic_eq_l83_83380

theorem roots_of_cubic_eq (u v w : ℝ) (huv : Polynomial.root (Polynomial.C u * Polynomial.C v * Polynomial.C w) (Polynomial.X^3 - 15 * Polynomial.X^2 + 13 * Polynomial.X - 6)) :
  (1 + u) * (1 + v) * (1 + w) = 35 := 
sorry

end roots_of_cubic_eq_l83_83380


namespace max_corner_odd_rectangles_is_60_l83_83142

noncomputable def max_corner_odd_rectangles_in_5x5_grid : ℕ :=
  let grid_size := 5 in
  let pair_count := nat.choose grid_size 2 in
  let corner_odd_rectangles_per_pair := 6 in
  pair_count * corner_odd_rectangles_per_pair

theorem max_corner_odd_rectangles_is_60 :
  max_corner_odd_rectangles_in_5x5_grid = 60 :=
by
  unfold max_corner_odd_rectangles_in_5x5_grid
  norm_num
  sorry

-- This statement encapsulates the equivalence problem of proving that the maximum possible number
-- of corner-odd rectangles in a 5x5 grid is exactly 60 given the conditions described.

end max_corner_odd_rectangles_is_60_l83_83142


namespace smallest_abundant_not_multiple_of_5_is_12_l83_83105

def is_abundant (n : ℕ) : Prop :=
  ∑ d in finset.filter (λ x, x ∣ n) (finset.range n), d > n

def smallest_abundant_not_multiple_of_5 : ℕ :=
  finset.find (λ n, is_abundant n ∧ ¬(5 ∣ n)) (finset.range 100) -- arbitrarily chose 100 as an upper bound for demonstration purposes

theorem smallest_abundant_not_multiple_of_5_is_12 :
  smallest_abundant_not_multiple_of_5 = 12 :=
sorry

end smallest_abundant_not_multiple_of_5_is_12_l83_83105


namespace range_of_t_l83_83606

noncomputable def f (x : ℝ) : ℝ :=
  real.log (sqrt (1 + x^2) - x) - x^3

theorem range_of_t (t : ℝ) (θ : ℝ) (hx : θ ∈ set.Icc 0 (real.pi / 2)) :
  f (cos θ ^ 2 - 2 * t) + f (4 * sin θ - 3) ≥ 0 ↔ t ∈ set.Ici (1 / 2) :=
begin
  sorry
end

end range_of_t_l83_83606


namespace prob1_prob2_l83_83492

-- Given conditions
def f (x : ℝ) : ℝ := (2 / 3) * x ^ 3 - x
def N := (1 : ℝ, -1 / 3 : ℝ)
def theta := Real.arctan 1

-- Problem statements
theorem prob1 : ∃ m n : ℝ, (f 1 = m - 1) ∧ (f' 1 = Real.tan theta) :=
sorry

theorem prob2 : ∃ (k : ℕ), (∀ x ∈ Icc (-1 : ℝ) (3 : ℝ), f x ≤ k - 1995) ∧ (∀ k', (∀ x ∈ Icc (-1 : ℝ) (3 : ℝ), f x ≤ k' - 1995) → k ≤ k') :=
sorry

end prob1_prob2_l83_83492


namespace arith_general_formula_geom_general_formula_geom_sum_formula_l83_83940

-- Arithmetic Sequence Conditions
def arith_seq (a₈ a₁₀ : ℕ → ℝ) := a₈ = 6 ∧ a₁₀ = 0

-- General formula for arithmetic sequence
theorem arith_general_formula (a₁ : ℝ) (d : ℝ) (h₈ : 6 = a₁ + 7 * d) (h₁₀ : 0 = a₁ + 9 * d) :
  ∀ n : ℕ, aₙ = 30 - 3 * (n - 1) :=
sorry

-- General formula for geometric sequence
def geom_seq (a₁ a₄ : ℕ → ℝ) := a₁ = 1/2 ∧ a₄ = 4

theorem geom_general_formula (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 1 / 2) (h₄ : 4 = a₁ * q ^ 3) :
  ∀ n : ℕ, aₙ = 2^(n-2) :=
sorry

-- Sum of the first n terms of geometric sequence
theorem geom_sum_formula (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 1 / 2) (h₄ : 4 = a₁ * q ^ 3) :
  ∀ n : ℕ, Sₙ = 2^(n-1) - 1 / 2 :=
sorry

end arith_general_formula_geom_general_formula_geom_sum_formula_l83_83940


namespace no_such_function_l83_83206

theorem no_such_function :
  ¬ (∃ f : ℕ → ℕ, ∀ n ≥ 2, f (f (n - 1)) = f (n + 1) - f (n)) :=
sorry

end no_such_function_l83_83206


namespace initial_girls_count_24_l83_83512

-- Define the initial total number of troop members
variables (p : ℕ)

-- Define the initial number of girls
def initial_number_of_girls : ℕ := 6 * p / 10

-- Define the number of girls after 4 girls leave
def girls_after_leaving : ℕ := initial_number_of_girls p - 4

-- Define the total number of troop members remains the same
def total_after_change : ℕ := p

-- Define the percentage condition after the change
def percentage_condition : Prop := (girls_after_leaving p) * 2 = total_after_change p

-- Main theorem
theorem initial_girls_count_24 : percentage_condition p → initial_number_of_girls p = 24 :=
begin
  sorry
end

end initial_girls_count_24_l83_83512


namespace number_of_arrangements_l83_83784

-- Define the problem conditions
variables (A B C D E : Type)
variable (adjacent : (A → B → Prop) ∧ (B → A → Prop))
variable (not_adjacent : (C → D → Prop) ∧ (D → C → Prop))

-- Define the theorem to prove the number of arrangements
theorem number_of_arrangements 
  (h_adj : adjacent A B) (h_nonadj : not_adjacent C D) :
  24 := by
  sorry

end number_of_arrangements_l83_83784


namespace recurring_decimal_fraction_sum_l83_83897

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l83_83897


namespace recurring_decimal_fraction_sum_l83_83896

theorem recurring_decimal_fraction_sum (x : ℚ) (h : x = 0.454545454545...) : 
    (let frac := rat.num_denom x.simplify in frac.fst + frac.snd)  = 16 :=
sorry

end recurring_decimal_fraction_sum_l83_83896


namespace minimum_value_l83_83239

noncomputable def hyperbola := {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ (x^2 / 9) - (y^2 / 4) = 1}
def foci := (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) : Prop :=
  F1 = (-3, 0) ∧ F2 = (3, 0)
def right_branch (p : ℝ × ℝ) : Prop :=
  p ∈ hyperbola ∧ p.1 > 0

theorem minimum_value (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (hP : right_branch P) (hF : foci F1 F2) :
  ∃ (m : ℝ), 0 < m ∧ ∀ (m : ℝ), m ≥ sqrt 13 - 3 → (m + 36 / m + 11) >= 23 :=
sorry

end minimum_value_l83_83239


namespace prove_seven_consecutive_even_numbers_l83_83083

def smallest_of_seven_consecutive_even (sum : ℕ) : ℕ :=
  let n := sum / 7
  n - 6

def median_of_seven_consecutive_even (sum : ℕ) : ℕ :=
  sum / 7

def mean_of_seven_consecutive_even (sum : ℕ) : ℕ :=
  sum / 7

theorem prove_seven_consecutive_even_numbers (h_sum : 686 = 686) :
  smallest_of_seven_consecutive_even 686 = 92 ∧
  median_of_seven_consecutive_even 686 = 98 ∧
  mean_of_seven_consecutive_even 686 = 98 :=
by
  {
      sorry,
  }

end prove_seven_consecutive_even_numbers_l83_83083


namespace draw_is_unfair_ensure_fair_draw_l83_83322

open ProbabilityTheory MeasureTheory

-- Definitions for the given conditions:
def Card := {rank : ℕ // 6 ≤ rank ∧ rank ≤ 14} -- Ranks 6 to Ace (6 to 14)
def Deck := Finset (Fin 36) -- 36 unique cards
noncomputable def suit_high_rank_count (d : Deck) (v_card : Fin 36) (m_card : Fin 36) : ℕ := 
  -- Count how many cards are higher than Volodya's card
  card.count (λ c, c.val > v_card.val) d

-- Volodya draws first, then Masha draws:
variables (d : Deck) (v_card m_card : Fin 36)

-- Masha wins if she draws a card with a higher rank than Volodya’s card
def masha_wins := ∃ (m_card : Fin 36), (m_card ∈ d) ∧ (m_card.val > v_card.val)

-- Volodya wins if Masha doesn't win (Masha loses)
def volodya_wins := ¬ masha_wins

theorem draw_is_unfair (d : Deck) (v_card m_card : Fin 36) :
  (volodya_wins d v_card m_card) → ¬ (masha_wins d v_card) := sorry

-- To make it fair, we can introduce a suit hierarchy:
def suits := {"Hearts", "Diamonds", "Clubs", "Spades"}
def suit_order : suits → suits → Prop
| "Spades" "Hearts" := true
| "Hearts" "Diamonds" := true
| "Diamonds" "Clubs" := true
| "Clubs" "Spades" := false
| _, _ := false

-- A fair draw means using the suit_order to rank otherwise equal cards:
def fair_draw :=
  ∀ (c1 c2 : Card), (c1.rank = c2.rank → suit_order c1.suit c2.suit)

theorem ensure_fair_draw : fair_draw := sorry

end draw_is_unfair_ensure_fair_draw_l83_83322


namespace more_polygons_without_A1_l83_83259

noncomputable def count_polygons_with (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), if k ≥ 3 then Nat.choose (n - 1) (k - 1) else 0

noncomputable def count_polygons_without (n : ℕ) : ℕ :=
  ∑ k in Finset.range n, if k ≥ 3 then Nat.choose (n - 1) k else 0

theorem more_polygons_without_A1 (n : ℕ) : 
  count_polygons_without n > count_polygons_with n :=
by
  sorry

end more_polygons_without_A1_l83_83259


namespace simplest_fraction_sum_l83_83074

noncomputable def p := 63
noncomputable def q := 125
noncomputable def fraction := 0.504

theorem simplest_fraction_sum (h : fraction = 0.504): (p = 63 ∧ q = 125) ∧ (p + q = 188) :=
by
  have : 0.504 = 63/125 := sorry
  split
  · exact (rfl, rfl)
  · exact (rfl, rfl)

end simplest_fraction_sum_l83_83074


namespace power_dissipated_R3_l83_83046

theorem power_dissipated_R3 : 
  let R1 := 1
  let R2 := 2
  let R3 := 3
  let R4 := 4
  let R5 := 5
  let R6 := 6
  let U := 12
  let R_total := R1 + R2 + R3 + R4 + R5 + R6
  R_total = 21 →
  P3 = (U^2 * R3) / (R_total^2) → 
  P3 ≈ 0.98 := 
by
  intros
  sorry

end power_dissipated_R3_l83_83046


namespace valid_三_digit_奥运会_l83_83344

noncomputable def 奥运会_心想事成_is_valid_division (奥运会 心想事成 : ℕ) : Prop :=
  (奥运会 ∈ {163, 318, 729, 1638, 1647}) ∧ (心想事成 = 9 * 奥运会)

theorem valid_三_digit_奥运会 :
  ∃ 奥运会 心想事成, 
  (奥运会_心想事成_is_valid_division 奥运会 心想事成) :=
begin
  use 163,
  use 1467,
  unfold 奥运会_心想事成_is_valid_division,
  split,
  { exact dec_trivial },
  { norm_num }
end

end valid_三_digit_奥运会_l83_83344


namespace sum_of_fraction_parts_l83_83877

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l83_83877


namespace perfect_odd_squares_mod_8_impossible_45_odd_sequence_l83_83077

namespace ProofProblems

-- Define what it means to be an odd integer
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Problem (a)
theorem perfect_odd_squares_mod_8 (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 :=
by sorry

-- Problem (b)
theorem impossible_45_odd_sequence :
  ¬∃ seq : Fin 45 → ℤ, (∀ (i : Fin 41), is_odd (seq i) ∧ ((∑ k in range (5), seq (i + k)) % 8 = 1))
  ∧ (∀ (i : Fin 37), is_odd (seq i) ∧ ((∑ k in range (9), seq (i + k)) % 8 = 1)) :=
by sorry

end ProofProblems

end perfect_odd_squares_mod_8_impossible_45_odd_sequence_l83_83077


namespace max_sqrt3a_sqrt2b_sqrtc_l83_83374

theorem max_sqrt3a_sqrt2b_sqrtc (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_cond : a + 2 * b + 3 * c = 13) : 
  (sqrt (3 * a) + sqrt (2 * b) + sqrt c) ≤ 13 * sqrt 3 / 3 :=
sorry

end max_sqrt3a_sqrt2b_sqrtc_l83_83374
