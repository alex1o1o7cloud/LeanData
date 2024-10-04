import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.DifferentialEquations.Linear
import Mathlib.Analysis.Polynomial.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Comb
import Mathlib.Combinatorics.Perm
import Mathlib.Data.Fin
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Perm
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.EquivFintype
import Mathlib.Tactic

namespace condition_for_m_l760_760664

theorem condition_for_m (m x : ℝ) (h1 : (2 - x) ≠ 0) : 
  (m + x) / (2 - x) - 3 = 0 → m ≠ -2 := 
by
  intro h_eq
  have h_mul : (m + x) - 3 * (2 - x) = 0,
  {
    calc (m + x) - 3 * (2 - x) = 0 : sorry,  -- multiply out and simplify
  }
  have h_rearrange : m + 4x - 6 = 0,
  {
    calc m + 4x - 6 = 0 : sorry,  -- combine like terms
  }
  have h_solve_x : x = (6 - m) / 4,
  {
    calc x = (6 - m) / 4 : sorry,  -- solve for x
  }
  have h_condition : 6 - m ≠ 8,
  {
    calc 6 - m ≠ 8 : sorry,  -- derive the condition from x ≠ 2
  }
  have h_notm : m ≠ -2,
  {
    calc m ≠ -2 : sorry,  -- simplify the inequality
  }
  exact h_notm

end condition_for_m_l760_760664


namespace marthas_bedroom_size_l760_760885

theorem marthas_bedroom_size (M J : ℕ) 
  (h1 : M + J = 300)
  (h2 : J = M + 60) :
  M = 120 := 
sorry

end marthas_bedroom_size_l760_760885


namespace binom_16_9_l760_760209

open Nat

theorem binom_16_9 :
  (Nat.choose 15 7 = 6435) ∧
  (Nat.choose 15 8 = 6435) ∧
  (Nat.choose 17 9 = 24310) →
  Nat.choose 16 9 = 11440 :=
by
  intros h
  cases h with h1 h23
  cases h23 with h2 h3
  sorry

end binom_16_9_l760_760209


namespace greatest_two_digit_multiple_of_17_l760_760447

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760447


namespace minimum_length_of_intersection_l760_760161

noncomputable def A (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 1 }
noncomputable def B (b : ℝ) : Set ℝ := { x | b - (1 / 2) ≤ x ∧ x ≤ b + 1 }
noncomputable def U : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
noncomputable def length (S : Set ℝ) : ℝ := Sup S - Inf S
  
theorem minimum_length_of_intersection :
  ∀ (a b : ℝ),
  (A a ⊆ U) → (B b ⊆ U) → (1 ≤ a) → (a ≤ 2) → (3 / 2 ≤ b) → (b ≤ 2) →
  length (A a ∩ B b) = 1 / 2 :=
by
  sorry

end minimum_length_of_intersection_l760_760161


namespace find_a_c_l760_760600

theorem find_a_c (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_neg : c < 0)
    (h_max : c + a = 3) (h_min : c - a = -5) :
  a = 4 ∧ c = -1 := 
sorry

end find_a_c_l760_760600


namespace sum_of_primes_less_than_20_l760_760003

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760003


namespace sum_of_arithmetic_terms_l760_760878

theorem sum_of_arithmetic_terms (a₁ a₂ a₃ c d a₆ : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₂ = 10)
  (h₃ : a₃ = 17)
  (h₄ : a₆ = 32)
  (h_arith : ∀ n, (a₁ + n * (a₂ - a₁)) = seq)
  : c + d = 55 :=
by
  have d := a₂ - a₁
  have c := a₃ + d
  have d := c + d
  have h_seq := list.map (λ n, (a₁ + n * d)) (list.range 6) -- Making use of the arithmetic property
  have h_seq_eq := h_seq = [3, 10, 17, c, d, 32]
  sorry

end sum_of_arithmetic_terms_l760_760878


namespace conscript_from_western_village_l760_760777

/--
Given:
- The population of the northern village is 8758
- The population of the western village is 7236
- The population of the southern village is 8356
- The total number of conscripts needed is 378

Prove that the number of people to be conscripted from the western village is 112.
-/
theorem conscript_from_western_village (hnorth : ℕ) (hwest : ℕ) (hsouth : ℕ) (hconscripts : ℕ)
    (htotal : hnorth + hwest + hsouth = 24350) :
    let prop := (hwest / (hnorth + hwest + hsouth)) * hconscripts
    hnorth = 8758 → hwest = 7236 → hsouth = 8356 → hconscripts = 378 → prop = 112 :=
by
  intros
  simp_all
  sorry

end conscript_from_western_village_l760_760777


namespace max_possible_value_of_a_l760_760797

theorem max_possible_value_of_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 150) : 
  a ≤ 8924 :=
by {
  sorry
}

end max_possible_value_of_a_l760_760797


namespace sam_total_pennies_l760_760342

def a : ℕ := 98
def b : ℕ := 93

theorem sam_total_pennies : a + b = 191 :=
by
  sorry

end sam_total_pennies_l760_760342


namespace packet_b_average_height_l760_760320

theorem packet_b_average_height (x y R_A R_B H_A H_B : ℝ)
  (h_RA : R_A = 2 * x + y)
  (h_RB : R_B = 3 * x - y)
  (h_x : x = 10)
  (h_y : y = 6)
  (h_HA : H_A = 192)
  (h_20percent : H_A = H_B + 0.20 * H_B) :
  H_B = 160 := 
sorry

end packet_b_average_height_l760_760320


namespace probability_area_ratio_l760_760705

theorem probability_area_ratio (A B C P : ℝ) (S : ℝ) (h d : ℝ) 
  (hP: P ∈ segment A B)
  (S_ABC : ℝ) (S_PBC : ℝ) 
  (hPBC_le: S_PBC ≤ (1/3) * S_ABC) 
  (hS_ABC: S_ABC = S):
  (∃ p : ℝ, p = (1/3)) := 
by
  sorry

end probability_area_ratio_l760_760705


namespace geometric_sequence_increasing_iff_condition_l760_760683

variable (a_n : ℕ → ℝ) (q : ℝ)

noncomputable def geometric_sequence (a_n q : ℝ) : Prop := 
  ∀ n : ℕ, a_n > 0

theorem geometric_sequence_increasing_iff_condition (h : geometric_sequence a_n q) : 
  (a_n 1 < a_n 3 ∧ a_n 3 < a_n 5) ↔ (∀ m n : ℕ, m < n → a_n m < a_n n) :=
sorry

end geometric_sequence_increasing_iff_condition_l760_760683


namespace greatest_two_digit_multiple_of_17_l760_760495

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760495


namespace mean_height_of_players_l760_760890

def heights_50s : List ℕ := [57, 59]
def heights_60s : List ℕ := [62, 64, 64, 65, 65, 68, 69]
def heights_70s : List ℕ := [70, 71, 73, 75, 75, 77, 78]

def all_heights : List ℕ := heights_50s ++ heights_60s ++ heights_70s

def mean_height (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / (l.length : ℚ)

theorem mean_height_of_players :
  mean_height all_heights = 68.25 :=
by
  sorry

end mean_height_of_players_l760_760890


namespace greatest_two_digit_multiple_of_17_l760_760488

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760488


namespace students_voted_both_policies_l760_760598

theorem students_voted_both_policies (total students_A students_B voted_against abstained: ℕ):
  total = 185 →
  students_A = 140 →
  students_B = 110 →
  voted_against = 22 →
  abstained = 15 →
  (students_A + students_B - total + voted_against + abstained) = 102 := 
by
  intros h_total h_A h_B h_against h_abstained
  calc
    students_A + students_B - total + voted_against + abstained = 140 + 110 - 185 + 22 + 15 : by rw [h_A, h_B, h_total, h_against, h_abstained]
    ... = 102 : sorry

end students_voted_both_policies_l760_760598


namespace reflection_sum_l760_760865

-- Definitions based on conditions
def point1 : ℝ × ℝ := (2, 3)
def point2 : ℝ × ℝ := (8, 7)
def line_of_reflection (m b : ℝ) : ℝ × ℝ → ℝ := λ p, -p.1 * m + b

-- The Lean 4 statement with conditions and the proof goal
theorem reflection_sum (m b : ℝ) 
  (line_reflects : ∀ (p : ℝ × ℝ), p = point1 → line_of_reflection m b p = point2) :
  m + b = 11 :=
sorry

end reflection_sum_l760_760865


namespace length_of_line_segment_A_B_l760_760720

/-- Let L be a line defined by the parametric equations x = 2 + t and y = 1 + t,
    and let C be a curve defined by the polar equation ρ^2 - 4ρcosθ + 3 = 0.
    The line L intersects the curve C at points A and B.
    Find the length of the line segment |AB|. -/
theorem length_of_line_segment_A_B 
  (L : ℝ → ℝ × ℝ := λ t, (2 + t, 1 + t))
  (C : ℝ → ℝ × ℝ := λ θ, let ρ := quadratic_solutions 1 (-4 * Math.cos θ) 3 
                         in (ρ * Math.cos θ, ρ * Math.sin θ))
  (A B : ℝ × ℝ) 
  (H_A : ∃ t, L t = A)
  (H_B : ∃ t, L t = B)
  (H_A_C : ∃ θ, C θ = A)
  (H_B_C : ∃ θ, C θ = B) :
  dist A B = sqrt 2 :=
sorry

end length_of_line_segment_A_B_l760_760720


namespace molecular_weight_correct_l760_760523

-- Define atomic weights
def atomic_weight_aluminium : Float := 26.98
def atomic_weight_oxygen : Float := 16.00
def atomic_weight_hydrogen : Float := 1.01
def atomic_weight_silicon : Float := 28.09
def atomic_weight_nitrogen : Float := 14.01

-- Define the number of each atom in the compound
def num_aluminium : Nat := 2
def num_oxygen : Nat := 6
def num_hydrogen : Nat := 3
def num_silicon : Nat := 2
def num_nitrogen : Nat := 4

-- Calculate the expected molecular weight
def expected_molecular_weight : Float :=
  (2 * atomic_weight_aluminium) + 
  (6 * atomic_weight_oxygen) + 
  (3 * atomic_weight_hydrogen) + 
  (2 * atomic_weight_silicon) + 
  (4 * atomic_weight_nitrogen)

-- Prove that the expected molecular weight is 265.21 amu
theorem molecular_weight_correct : expected_molecular_weight = 265.21 :=
by
  sorry

end molecular_weight_correct_l760_760523


namespace exists_coloring_for_K6_complete_l760_760631

def k6 : Type := fin 6

def edges (V : Type) := {e : V × V // e.1 ≠ e.2}

def colored_edges (V : Type) (colors : Type) :=
  edges V → colors

inductive color : Type
| C1 | C2 | C3 | C4 | C5

open color

noncomputable def color_K6 (c : colored_edges k6 color) : Prop :=
  ∀ (v : k6), ∃ (c1 c2 c3 c4 c5 : color),
  ∀ (u : k6), u ≠ v → (c ⟨(u, v), by simp [fin.veq_of_eq]⟩ = c1 ∨
                        c ⟨(u, v), by simp [fin.veq_of_eq]⟩ = c2 ∨
                        c ⟨(u, v), by simp [fin.veq_of_eq]⟩ = c3 ∨
                        c ⟨(u, v), by simp [fin.veq_of_eq]⟩ = c4 ∨
                        c ⟨(u, v), by simp [fin.veq_of_eq]⟩ = c5)

theorem exists_coloring_for_K6_complete : ∃ (c : colored_edges k6 color), color_K6 c :=
begin
  sorry
end

end exists_coloring_for_K6_complete_l760_760631


namespace solve_arithmetic_sequence_sum_l760_760875

noncomputable def arithmetic_sequence_sum : ℕ :=
  let a : ℕ := 3
  let b : ℕ := 10
  let c : ℕ := 17
  let e : ℕ := 32
  let d := b - a
  let c_term := c + d
  let d_term := c_term + d
  c_term + d_term

theorem solve_arithmetic_sequence_sum : arithmetic_sequence_sum = 55 :=
by
  sorry

end solve_arithmetic_sequence_sum_l760_760875


namespace part_a_part_b_l760_760329

open Matrix

-- Definition of the board size
def board_size : ℕ := 7

-- A configuration of Xs on a board is a matrix of size 7x7 with boolean entries
def Board := Matrix (Fin board_size) (Fin board_size) Bool

-- Part (a): A specific configuration where no 4 cells form a rectangle
def no_castles_config : Board :=
 λ r c, 
   (r.val, c.val) ∈ [
     (0, 0), (0, 1), (0, 3), 
     (1, 1), (1, 2), (1, 4), 
     (2, 2), (2, 3), (2, 5), 
     (3, 3), (3, 4), (3, 6), 
     (4, 0), (4, 4), (4, 5), 
     (5, 1), (5, 5), (5, 6), 
     (6, 0), (6, 2), (6, 6)]

-- Definition to check if four given points form a rectangle
def forms_rectangle (a b c d : (Fin board_size) × (Fin board_size)) : Prop :=
  (a.1 = b.1 ∧ c.1 = d.1 ∧ a.2 = c.2 ∧ b.2 = d.2) ∨
  (a.1 = c.1 ∧ b.1 = d.1 ∧ a.2 = b.2 ∧ c.2 = d.2) ∨
  (a.1 = d.1 ∧ b.1 = c.1 ∧ a.2 = b.2 ∧ d.2 = c.2)

-- Part (a) theorem
theorem part_a : 
  ¬ ∃ (a b c d : (Fin board_size) × (Fin board_size)), 
    no_castles_config a.1 a.2 ∧ no_castles_config b.1 b.2 ∧ 
    no_castles_config c.1 c.2 ∧ no_castles_config d.1 d.2 ∧ 
    forms_rectangle a b c d := 
by
  sorry

-- Part (b): For any configuration of 22 marked cells, there is at least one rectangle
theorem part_b (marked_cells : Fin 22 → (Fin board_size) × (Fin board_size)) :
  ∃ (a b c d : (Fin board_size) × (Fin board_size)),
    (∃ i, marked_cells i = a) ∧ (∃ i, marked_cells i = b) ∧
    (∃ i, marked_cells i = c) ∧ (∃ i, marked_cells i = d) ∧
    forms_rectangle a b c d := 
by
  sorry

end part_a_part_b_l760_760329


namespace combination_15_6_l760_760119

theorem combination_15_6 : nat.choose 15 6 = 5005 := by
  sorry

end combination_15_6_l760_760119


namespace sum_divisible_by_105_l760_760338

theorem sum_divisible_by_105 : 
  let S := ∑ i in (Finset.range 12), 2^(2*i + 1) in 
  S % 105 = 0 := by
  sorry

end sum_divisible_by_105_l760_760338


namespace find_abc_sum_l760_760902

def floor_part (x : ℝ) : ℝ := ⌊x⌋
def frac_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem find_abc_sum (x a b c : ℕ) (h1 : x = (a + real.sqrt b) / c)
  (h2 : 1 + real.log x (floor_part x) = 2 * real.log x (real.sqrt 3 * frac_part x))
  (hb_prop : nat.coprime b (nat.min_fac b) ∧ ∀ p, nat.prime p → p^2 ∣ b → false) : 
  a + b + c = 26 := 
  sorry

end find_abc_sum_l760_760902


namespace sum_of_primes_less_than_twenty_is_77_l760_760054

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760054


namespace root_of_quadratic_probability_l760_760908

noncomputable def probability_is_root_of_quadratic (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) 
    (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) : ℚ :=
  if (a = 1 ∧ (c = b * b + 1 ∧ (b = 1 ∨ b = 2))) then (1 : ℚ) / 108 else 0

theorem root_of_quadratic_probability :
  (true -- replace with proper condition if necessary
  → 
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 
  (a + b * complex.I).root_of (λ x, x^2 - 2*x + c) ∧ probability_is_root_of_quadratic a b c _ _ _ = 1/108) :=
sorry

end root_of_quadratic_probability_l760_760908


namespace octahedron_side_length_l760_760582

noncomputable def side_length_of_octahedron : ℝ :=
let Q1 := (0, 0, 0) in
let Q1' := (1, 1, 1) in
let v1 := (3/4, 0, 0) in
let v2 := (0, 3/4, 0) in
let v3 := (0, 0, 3/4) in
let v4 := (1, 1/4, 1) in
let v5 := (1, 1, 1/4) in
let v6 := (1/4, 1, 1) in
(real.sqrt ((3/4 : ℝ) ^ 2 + (3/4 : ℝ) ^ 2))

theorem octahedron_side_length (Q1 Q1' : ℝ × ℝ × ℝ)
  (v1 v2 v3 v4 v5 v6 : ℝ × ℝ × ℝ)
  (h1 : Q1 = (0, 0, 0))
  (h2 : Q1' = (1, 1, 1))
  (h3 : v1 = (3/4, 0, 0))
  (h4 : v2 = (0, 3/4, 0))
  (h5 : v3 = (0, 0, 3/4))
  (h6 : v4 = (1, 1/4, 1))
  (h7 : v5 = (1, 1, 1/4))
  (h8 : v6 = (1/4, 1, 1)) :
  side_length_of_octahedron = 3 * real.sqrt 2 / 4 := sorry

end octahedron_side_length_l760_760582


namespace best_fit_model_l760_760768

theorem best_fit_model
  (R2_M1 R2_M2 R2_M3 R2_M4 : ℝ)
  (h1 : R2_M1 = 0.78)
  (h2 : R2_M2 = 0.85)
  (h3 : R2_M3 = 0.61)
  (h4 : R2_M4 = 0.31) :
  ∀ i, (i = 2 ∧ R2_M2 ≥ R2_M1 ∧ R2_M2 ≥ R2_M3 ∧ R2_M2 ≥ R2_M4) := 
sorry

end best_fit_model_l760_760768


namespace measure_angle_BAO_l760_760776

-- Definitions corresponding to conditions in the problem
variables (CD AO AF : ℝ)
variables (O : Type) [EuclideanSpace O]  -- O as a EuclideanSpace
variables (A B C D F : O)
variable (r : ℝ)  -- radius

-- Given conditions
def is_diameter (CD : circle O r) : Prop := true -- Assuming CD is the diameter
def on_semicircle (P : O) : Prop := true -- Placeholder for definition of P on the semi-circle
def length_eq (P Q : O) (L : ℝ) : Prop := dist P Q = L

-- Additional given measures
def angle_FOD_eq_60 (bo do af : O) : Prop := angle do af bo = 60

-- Main proof statement
theorem measure_angle_BAO (h1 : is_diameter CD) (h2 : on_semicircle F)
  (h3 : length_eq A B (dist O D)) 
  (h4 : angle_FOD_eq_60 B O F) : ∠ B A O = 20 := 
begin
  sorry
end

end measure_angle_BAO_l760_760776


namespace find_Q_l760_760274

theorem find_Q (m n Q p : ℝ) (h1 : m = 6 * n + 5)
    (h2 : p = 0.3333333333333333)
    (h3 : m + Q = 6 * (n + p) + 5) : Q = 2 := 
by
  sorry

end find_Q_l760_760274


namespace earliest_year_for_mismatched_pairs_l760_760585

def num_pairs (year : ℕ) : ℕ := 2 ^ (year - 2013)

def mismatched_pairs (pairs : ℕ) : ℕ := pairs * (pairs - 1)

theorem earliest_year_for_mismatched_pairs (year : ℕ) (h : year ≥ 2013) :
  (∃ pairs, (num_pairs year = pairs) ∧ (mismatched_pairs pairs ≥ 500)) → year = 2018 :=
by
  sorry

end earliest_year_for_mismatched_pairs_l760_760585


namespace two_solutions_exist_l760_760779

-- Definitions for the letters and their digit assignments
def letter_to_digit := 
  (R A T M D U L O H Y : ℕ)
  
variable (R A T M D U L O H Y : ℕ)

-- Conditions to ensure different letters represent different digits
def distinct_letters : Prop := 
  R ≠ A ∧ R ≠ T ∧ R ≠ M ∧ R ≠ D ∧
  R ≠ U ∧ R ≠ L ∧ R ≠ O ∧ R ≠ H ∧ R ≠ Y ∧
  A ≠ T ∧ A ≠ M ∧ A ≠ D ∧
  A ≠ U ∧ A ≠ L ∧ A ≠ O ∧ A ≠ H ∧ A ≠ Y ∧
  T ≠ M ∧ T ≠ D ∧
  T ≠ U ∧ T ≠ L ∧ T ≠ O ∧ T ≠ H ∧ T ≠ Y ∧
  M ≠ D ∧
  M ≠ U ∧ M ≠ L ∧ M ≠ O ∧ M ≠ H ∧ M ≠ Y ∧
  D ≠ U ∧ D ≠ L ∧ D ≠ O ∧ D ≠ H ∧ D ≠ Y ∧
  U ≠ L ∧ U ≠ O ∧ U ≠ H ∧ U ≠ Y ∧
  L ≠ O ∧ L ≠ H ∧ L ≠ Y ∧
  O ≠ H ∧ O ≠ Y ∧
  H ≠ Y

-- Conversion of letters to their digit values
def to_digit (R A T M D : ℕ) : ℕ := 
  10000 * R + 1000 * A + 100 * T + 10 * A + M

-- The addition problem
def add_letters (R A T M D U L O H Y : ℕ) : Prop := 
  to_digit R A T M D + (100 * R + 10 * A + D) = 10000 * U + 1000 * L + 100 * O + 10 * H + Y

-- Proving the existence of two different valid solutions
theorem two_solutions_exist : ∃ (R A T M D U L O H Y : ℕ), 
  distinct_letters R A T M D U L O H Y ∧ add_letters R A T M D U L O H Y ∧
  ∃ (R2 A2 T2 M2 D2 U2 L2 O2 H2 Y2 : ℕ), 
  distinct_letters R2 A2 T2 M2 D2 U2 L2 O2 H2 Y2 ∧ add_letters R2 A2 T2 M2 D2 U2 L2 O2 H2 Y2 ∧
  (R ≠ R2 ∨ A ≠ A2 ∨ T ≠ T2 ∨ M ≠ M2 ∨ D ≠ D2 ∨ U ≠ U2 ∨ L ≠ L2 ∨ O ≠ O2 ∨ H ≠ H2 ∨ Y ≠ Y2) :=
sorry

end two_solutions_exist_l760_760779


namespace sum_primes_less_than_20_l760_760067

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760067


namespace four_equilateral_triangles_cover_delta_l760_760251

theorem four_equilateral_triangles_cover_delta
  (a b : ℝ)
  (Delta : Triangle)
  (hb : ∀ (t : Triangle), t ∈ cover_5b(Delta) → side_length(t) = b)
  (ha : side_length(Delta) = a)
  (cover_5b : Triangle -> set Triangle) :
  ∃ (cover_4b : Triangle -> set Triangle), (∀ (t : Triangle), t ∈ cover_4b(Delta) → side_length(t) = b) ∧ Delta ⊆ ⋃ (t : Triangle) (ht : t ∈ cover_4b(Delta)), t  :=
sorry

end four_equilateral_triangles_cover_delta_l760_760251


namespace union_of_sets_l760_760727

def set_M : Set ℕ := {0, 1, 3}
def set_N : Set ℕ := {x | ∃ (a : ℕ), a ∈ set_M ∧ x = 3 * a}

theorem union_of_sets :
  set_M ∪ set_N = {0, 1, 3, 9} :=
by
  sorry

end union_of_sets_l760_760727


namespace six_points_seventeen_triangles_l760_760170

theorem six_points_seventeen_triangles : 
  ∃ (points : Finset (ℝ × ℝ)), points.card = 6 ∧ 
  (∑ S in points.powerset.filter (λ S, S.card = 3), (if collinear S then 0 else 1)) = 17 :=
sorry

end six_points_seventeen_triangles_l760_760170


namespace range_of_a_l760_760723

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x^2 + 2*a*x + a) > 0) → (0 < a ∧ a < 1) :=
sorry

end range_of_a_l760_760723


namespace max_value_fraction_l760_760255

theorem max_value_fraction (a b : ℝ)
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  (a ≠ 0) → (b ≠ 0) →
  ∃ m, m = (a + 2 * b) / (2 * a + b) ∧ m ≤ 7 / 5 :=
by
  sorry

end max_value_fraction_l760_760255


namespace abc_is_perfect_cube_l760_760541

-- Define the main problem statement for part (b):
theorem abc_is_perfect_cube (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a : ℚ) / b + b / c + c / a ∈ ℤ) : ∃ n : ℤ, abc = n^3 :=
by
  sorry

-- Examples for part (a):
example : (1, 2, 4, 5) := by
  let a := 1
  let b := 2
  let c := 4
  let n := 5
  have h : (a : ℚ) / b + b / c + c / a = n := by
    norm_num
  exact ⟨a, b, c, n, h⟩

example : (9, 2, 12, 6) := by
  let a := 9
  let b := 2
  let c := 12
  let n := 6
  have h : (a : ℚ) / b + b / c + c / a = n := by
    norm_num
  exact ⟨a, b, c, n, h⟩

end abc_is_perfect_cube_l760_760541


namespace remaining_bricks_below_250_l760_760553

theorem remaining_bricks_below_250 :
  let initial_bricks := 2020 ^ 2 in
  ∃ (k : ℕ), let t : ℕ → ℕ := λ n, (2020 - n) ^ 2 - (2020 - n) in
  t k < 250 ∧ t k = 240 :=
begin
  sorry
end

end remaining_bricks_below_250_l760_760553


namespace percentage_girls_l760_760579

theorem percentage_girls (x y : ℕ) (S₁ S₂ : ℕ)
  (h1 : S₁ = 22 * x)
  (h2 : S₂ = 47 * y)
  (h3 : (S₁ + S₂) / (x + y) = 41) :
  (x : ℝ) / (x + y) = 0.24 :=
sorry

end percentage_girls_l760_760579


namespace distance_between_foci_of_ellipse_l760_760990

theorem distance_between_foci_of_ellipse :
  ∃ (a b c : ℝ),
  -- Condition: axes are parallel to the coordinate axes (implicitly given by tangency points).
  a = 3 ∧
  b = 2 ∧
  c = Real.sqrt (a^2 - b^2) ∧
  2 * c = 2 * Real.sqrt 5 :=
sorry

end distance_between_foci_of_ellipse_l760_760990


namespace max_integer_m_solution_max_integer_m_l760_760753

theorem max_integer_m (m : ℝ) (h : ∃ x : ℝ, x = 3 ∧ (x / 3 + 2 * m < -3)) : m ≤ -2 :=
by
  sorry

theorem solution_max_integer_m (m : ℝ) (h : max_integer_m m) : m = -3 :=
by
  sorry

end max_integer_m_solution_max_integer_m_l760_760753


namespace spiral_2018_position_l760_760597

def T100_spiral : Matrix ℕ ℕ ℕ := sorry -- Definition of T100 as a spiral matrix

def pos_2018 := (34, 95) -- The given position we need to prove

theorem spiral_2018_position (i j : ℕ) (h₁ : T100_spiral 34 95 = 2018) : (i, j) = pos_2018 := by  
  sorry

end spiral_2018_position_l760_760597


namespace permutations_count_l760_760656

open Finset

-- Define the set of permutations
def perms := univ.permutations (erase_univ 6)

-- Define the condition predicate
def condition (b : Fin 6 → Fin 6) : Prop :=
  ((b 0 + 1) / 3) * ((b 1 + 2) / 3) * ((b 2 + 3) / 3) * ((b 3 + 4) / 3) * ((b 4 + 5) / 3) * ((b 5 + 6) / 3) > fact 6

-- Define the final problem statement
theorem permutations_count :
  (univ.permutations (erase_univ 6)).filter (λ b, condition b) = 719 :=
by
  sorry

end permutations_count_l760_760656


namespace min_product_coprime_l760_760693

theorem min_product_coprime (n : ℕ) (h_n : n ≥ 2) (a : Fin n → ℕ) 
  (h_pos : ∀ i, a i > 0) (h_coprime : ∀ i j, i ≠ j → gcd (a i) (a j) = 1) :
  let A := ∑ i, a i,
      d := λ i, gcd A (a i),
      D := λ i, gcd ( (Finset.univ.erase i).sum a ) 
  in
  (∏ i, (A - a i) / (d i * D i)) = (n - 1)^n := 
sorry

end min_product_coprime_l760_760693


namespace cos_B_plus_C_value_of_c_l760_760758

variable {A B C a b c : ℝ}

-- Given conditions
axiom a_eq_2b : a = 2 * b
axiom sine_arithmetic_sequence : 2 * Real.sin C = Real.sin A + Real.sin B

-- First proof
theorem cos_B_plus_C (h : a = 2 * b) (h_seq : 2 * Real.sin C = Real.sin A + Real.sin B) :
  Real.cos (B + C) = 1 / 4 := 
sorry

-- Given additional condition for the area
axiom area_eq : (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 15) / 3

-- Second proof
theorem value_of_c (h : a = 2 * b) (h_seq : 2 * Real.sin C = Real.sin A + Real.sin B) (h_area : (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 15) / 3) :
  c = 4 * Real.sqrt 2 :=
sorry

end cos_B_plus_C_value_of_c_l760_760758


namespace greatest_two_digit_multiple_of_17_l760_760511

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760511


namespace eldest_child_age_l760_760964

variable (y m e : Nat)

theorem eldest_child_age :
  (m - y = 3) →
  (e = 3 * y) →
  (e = y + m + 2) →
  (e = 15) :=
by
  intros h1 h2 h3
  sorry

end eldest_child_age_l760_760964


namespace part_a_part_b_l760_760606

-- Define the regions Omega_a and Omega_b
def region_a (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ (3 - x)

def region_b (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ x

noncomputable def integral_a : ℝ :=
  ∫ x in 0..3, ∫ y in 0..(3-x), (x + y)

noncomputable def integral_b : ℝ :=
  ∫ x in 0..3, ∫ y in 0..x, (x + y)

theorem part_a : integral_a = 9 := by
  sorry

theorem part_b : integral_b = 27 / 2 := by
  sorry

end part_a_part_b_l760_760606


namespace equilateral_triangle_l760_760840

theorem equilateral_triangle (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + bc + ca) : a = b ∧ b = c := 
by sorry

end equilateral_triangle_l760_760840


namespace crop_planting_problem_l760_760558

theorem crop_planting_problem : 
  ∃ (n : ℕ), 
    (let sections := [0, 1, 2] in
    let crops := {lettuce, carrot, radish} in
    -- define the condition that lettuce and radishes cannot be adjacent
    (∀ (a b : ℕ), a ≠ b → ¬((sections[a] = lettuce ∧ sections[succ a] = radish) ∨ (sections[a] = radish ∧ sections[succ a] = lettuce))) →
    -- prove the number of valid ways to plant the crops is 15
    n = 15)
:= sorry

end crop_planting_problem_l760_760558


namespace range_of_a_l760_760738

theorem range_of_a (a : ℚ) (h_pos : 0 < a) (h_int_count : ∀ n : ℕ, 2 * n + 1 = 2007 -> ∃ k : ℤ, -a < ↑k ∧ ↑k < a) : 1003 < a ∧ a ≤ 1004 :=
sorry

end range_of_a_l760_760738


namespace sum_of_three_digit_even_naturals_correct_l760_760660

noncomputable def sum_of_three_digit_even_naturals : ℕ := 
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  n / 2 * (a + l)

theorem sum_of_three_digit_even_naturals_correct : 
  sum_of_three_digit_even_naturals = 247050 := by 
  sorry

end sum_of_three_digit_even_naturals_correct_l760_760660


namespace fraction_order_l760_760622

def frac_21_16 := 21 / 16
def frac_25_19 := 25 / 19
def frac_23_17 := 23 / 17
def frac_27_20 := 27 / 20

theorem fraction_order : frac_21_16 < frac_25_19 ∧ frac_25_19 < frac_27_20 ∧ frac_27_20 < frac_23_17 := by sorry

end fraction_order_l760_760622


namespace teal_more_blue_proof_l760_760950

theorem teal_more_blue_proof (P G B N : ℕ) (hP : P = 150) (hG : G = 90) (hB : B = 40) (hN : N = 25) : 
  (∃ (x : ℕ), x = 75) :=
by
  sorry

end teal_more_blue_proof_l760_760950


namespace candy_store_revenue_l760_760552

/-- A candy store sold 20 pounds of fudge for $2.50 per pound,
    5 dozen chocolate truffles for $1.50 each, 
    and 3 dozen chocolate-covered pretzels at $2.00 each.
    Prove that the total money made by the candy store is $212.00. --/
theorem candy_store_revenue :
  let fudge_pounds := 20
  let fudge_price_per_pound := 2.50
  let truffle_dozen := 5
  let truffle_price_each := 1.50
  let pretzel_dozen := 3
  let pretzel_price_each := 2.00
  (fudge_pounds * fudge_price_per_pound) + 
  (truffle_dozen * 12 * truffle_price_each) + 
  (pretzel_dozen * 12 * pretzel_price_each) = 212 :=
by
  sorry

end candy_store_revenue_l760_760552


namespace sum_of_primes_less_than_20_eq_77_l760_760026

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760026


namespace greatest_two_digit_multiple_of_17_l760_760431

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760431


namespace sum_prime_numbers_less_than_twenty_l760_760031

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760031


namespace greatest_two_digit_multiple_of_17_is_85_l760_760462

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760462


namespace red_balls_probability_l760_760954

noncomputable def number_of_red_balls : ℕ := 5

theorem red_balls_probability (total_balls : ℕ) (prob : ℚ) :
  (total_balls = 15) →
  (prob = (1/21 : ℚ)) →
  ∃ (r : ℕ), (r * (r - 1) / (15 * 14 : ℕ) : ℚ) = prob ∧ r = number_of_red_balls :=
by
  intros h_total_balls h_prob
  use number_of_red_balls
  split
  . rw [h_total_balls, h_prob]
    norm_num -- simplifies the arithmetic
  . refl -- states that the solution red balls is 5


end red_balls_probability_l760_760954


namespace smallest_integer_solution_l760_760915

def star_op (x y : ℝ) : ℝ := (x * y) / 3 - 2 * y

theorem smallest_integer_solution :
  ∃ (a : ℤ), star_op 2 a ≤ 2 ∧ ∀ (b : ℤ), star_op 2 b ≤ 2 → a ≤ b :=
sorry

end smallest_integer_solution_l760_760915


namespace predict_waste_in_2019_l760_760992

variable (a b : ℝ)

/-- Predict the amount of municipal solid waste in 2019 -/
theorem predict_waste_in_2019 (a b : ℝ) : 
  let waste_2019 := a * (1 + b) ^ 10 in
  waste_2019 = a * (1 + b) ^ 10 :=
by
  sorry

end predict_waste_in_2019_l760_760992


namespace function_increasing_l760_760623

theorem function_increasing (a : ℝ) (h : a > 1) :
  ∀ x ≥ (3 / 2), ∃ y, y = a^(x^2 - 3*x + 2) ∧
                (∀ x1 x2, (3 / 2) ≤ x1 → x1 < x2 → y1 = a^(x1^2 - 3*x1 + 2) → y2 = a^(x2^2 - 3*x2 + 2) → y1 < y2) :=
begin
  sorry
end

end function_increasing_l760_760623


namespace hyperbola_eccentricity_l760_760755

-- Define the hyperbola and its components given in the conditions.
variable (a b c : ℝ) (e : ℝ)
variable (hyp : a^2 + b^2 = c^2)
variable (asymptote1 asymptote2 : ℝ → Prop)
variable (Focus : ℝ × ℝ → Prop)
variable (Point : ℝ × ℝ → Prop)

-- Define the equations of the hyperbola and asymptotes.
def hyperbola_eq : Prop := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1
def asymptote_eq : Prop := ∀ x y : ℝ, (y = (b / a) * x) ∨ (y = -(b / a) * x)
def focus_eq : Prop := Focus(c, 0)

-- Conditions that the symmetric point P lies on the other asymptote.
def symmetric_point : Prop := ∃ m : ℝ, Point(m, -(b / a) * m) ∧ ∀ x y : ℝ, Point((m + c) / 2, -(b / a) * m / 2) ∧ (x = (m + c) / 2) ∧ (y = (-(b / a) * m / 2))

-- Define the eccentricity based on the provided conditions.
def eccentricity := c / a

theorem hyperbola_eccentricity : symmetric_point a b c → e = 2 := by
  sorry

end hyperbola_eccentricity_l760_760755


namespace least_clock_equiv_square_l760_760328

def clock_equiv (h k : ℕ) : Prop := (h - k) % 24 = 0

theorem least_clock_equiv_square : ∃ (h : ℕ), h > 6 ∧ (h^2) % 24 = h % 24 ∧ (∀ (k : ℕ), k > 6 ∧ clock_equiv k (k^2) → h ≤ k) :=
sorry

end least_clock_equiv_square_l760_760328


namespace greatest_two_digit_multiple_of_17_l760_760500

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760500


namespace height_of_lot_l760_760772

variable (Length Width Volume Height : ℝ)

-- Given conditions
axiom H1 : Length = 40
axiom H2 : Width = 20
axiom H3 : Volume = 1600

-- Rectangular prism volume formula
axiom volume_formula (h l w : ℝ) : Volume = l * w * h

-- Theorem that states the height of the lot
theorem height_of_lot : Height = 2 :=
by
  have H : Height = Volume / (Length * Width) := sorry
  exact H

end height_of_lot_l760_760772


namespace total_number_of_people_l760_760263

theorem total_number_of_people (L F LF N T : ℕ) (hL : L = 13) (hF : F = 15) (hLF : LF = 9) (hN : N = 6) : 
  T = (L + F - LF) + N → T = 25 :=
by
  intros h
  rw [hL, hF, hLF, hN] at h
  exact h

end total_number_of_people_l760_760263


namespace find_m_l760_760236

noncomputable def m_eq_sqrt3 (a b : ℝ × ℝ) (m : ℝ) : Prop :=
  a + b = (m, 2) ∧ b = (0, 1) ∧
  (real.angle (a.fst * a.fst + a.snd * a.snd) (b.fst * b.fst + b.snd * b.snd) = real.pi / 3) →
  m = real.sqrt 3 ∨ m = -real.sqrt 3

theorem find_m (a b : ℝ × ℝ) (m : ℝ) : m_eq_sqrt3 a b m := by
  sorry

end find_m_l760_760236


namespace find_b_c_l760_760395

theorem find_b_c : 
  ∃ (b c : ℝ), 
    let v1 := (4, 1) in
    let v2 := (b, -8) in
    let v3 := (5, c) in
    (v1.1 * v3.1 + v1.2 * v3.2 = 0) ∧ (v2.1 * v1.1 + v2.2 * v1.2 = 0) ∧ 
    b = 2 ∧ c = -20 :=
by
  sorry

end find_b_c_l760_760395


namespace angle_through_point_l760_760756

theorem angle_through_point : 
  (∃ θ : ℝ, ∃ k : ℤ, θ = 2 * k * Real.pi + 5 * Real.pi / 6 ∧ 
                      ∃ x y : ℝ, x = -Real.sqrt 3 / 2 ∧ y = 1 / 2 ∧ 
                                    y / x = Real.tan θ) := 
sorry

end angle_through_point_l760_760756


namespace greatest_two_digit_multiple_of_17_l760_760414

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760414


namespace community_service_selection_l760_760113

theorem community_service_selection:
  let boys := 4
  let girls := 2
  ∀ (selection : Finset (Fin (boys + girls))), 
    selection.card = 4 ∧ ∃ g ∈ selection, g.1 ≥ boys → 
    selection.card = (14: ℕ) →
   sorry

end community_service_selection_l760_760113


namespace ab_range_l760_760752

-- Define the conditions of the problem
def Line (a b : ℝ) := { p : ℝ × ℝ // p.1 * a - p.2 * b + 1 = 0 }
def Circle := { p : ℝ × ℝ // p.1^2 + p.2^2 + 2 * p.1 - 4 * p.2 + 1 = 0 }

-- State that the line ax - by + 1 = 0 passes through the center of the circle
theorem ab_range (a b : ℝ) (h1 : Line a b (-1, 2)) : ab ≤ 1/8 :=
by {
  have h_center : (-1) * a - 2 * b + 1 = 0 := h1,
  -- Proof steps (omitted)
  sorry
}

end ab_range_l760_760752


namespace weekend_minutes_l760_760816

theorem weekend_minutes (episodes_per_week : ℕ) (minutes_per_episode : ℕ) (monday_minutes : ℕ)
  (tuesday_minutes : ℕ) (wednesday_minutes : ℕ) (thursday_minutes : ℕ) (friday_episodes : ℕ) :
  episodes_per_week = 8 →
  minutes_per_episode = 44 →
  monday_minutes = 138 →
  tuesday_minutes = 0 →
  wednesday_minutes = 0 →
  thursday_minutes = 21 →
  friday_episodes = 2 →
  let total_minutes := episodes_per_week * minutes_per_episode,
      week_minutes := monday_minutes + tuesday_minutes + wednesday_minutes + thursday_minutes + friday_episodes * minutes_per_episode in
  total_minutes - week_minutes = 105 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  have total_minutes : ℕ := episodes_per_week * minutes_per_episode,
  have week_minutes : ℕ := monday_minutes + tuesday_minutes + wednesday_minutes + thursday_minutes + friday_episodes * minutes_per_episode,
  rw [h1, h2, h3, h4, h5, h6, h7],
  repeat {simp},
  sorry

end weekend_minutes_l760_760816


namespace factorial_division_problem_statement_l760_760604

-- Definition of factorial (fact)
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- The conditions statement
theorem factorial_division (n : ℕ) : ∀ 4!, 4! = 24 := 
by {
  have h : fact 4 = 24 := by norm_num [fact];
  exact h;
}

-- The mathematically equivalent proof problem
theorem problem_statement : ((fact (fact 4)) / (fact 4) = fact 23) := 
by {
  have h1 : fact 4 = 24 := by norm_num [fact];
  rw [h1],
  norm_num [fact, h1],
  have h2 := Nat.fact_div_fact_div,
  exact h2,
}

end factorial_division_problem_statement_l760_760604


namespace exist_n_l760_760940

theorem exist_n : ∃ n : ℕ, n > 1 ∧ ¬(Nat.Prime n) ∧ ∀ a : ℤ, (a^n - a) % n = 0 :=
by
  sorry

end exist_n_l760_760940


namespace sum_prime_numbers_less_than_twenty_l760_760040

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760040


namespace greatest_two_digit_multiple_of_17_l760_760481

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760481


namespace reduce_entanglement_l760_760969

/- 
Define a graph structure and required operations as per the given conditions. 
-/
structure Graph (V : Type) :=
  (E : V -> V -> Prop)

def remove_odd_degree_verts (G : Graph V) : Graph V :=
  sorry -- Placeholder for graph reduction logic

def duplicate_graph (G : Graph V) : Graph V :=
  sorry -- Placeholder for graph duplication logic

/--
  Prove that any graph where each vertex can be part of multiple entanglements 
  can be reduced to a state where no two vertices are connected using the given operations.
-/
theorem reduce_entanglement (G : Graph V) : ∃ G', 
  G' = remove_odd_degree_verts (duplicate_graph G) ∧
  (∀ (v1 v2 : V), ¬ G'.E v1 v2) :=
  by
  sorry

end reduce_entanglement_l760_760969


namespace seq_geom_exists_m_l760_760712

noncomputable def S (n : ℕ) (a : ℕ → ℕ) : ℕ := (3 / 2 : ℚ) * a n + n - 3

-- 1. Prove that the sequence {a_n - 1} is a geometric sequence
theorem seq_geom (a : ℕ → ℕ) (h : ∀ n : ℕ, S n a = S (n + 1) a - a (n + 1) + 3) :
  ∃ r : ℕ, ∀ n : ℕ, a (n + 1) - 1 = r * (a n - 1) := 
sorry

-- 2. Prove that there exists a positive integer m such that 
-- (1 / c₁ + 1 / c₂ + ... + 1 / cₙ ≥ m / 3) for m = 1, 2, or 3,
-- where cₙ = log₃(a₁ - 1) + log₃(a₂ - 1) + ... + log₃(aₙ - 1)
theorem exists_m (a : ℕ → ℕ) (h_geom : ∀ n : ℕ, a (n + 1) - 1 = 3 * (a n - 1)) :
  ∃ m : ℕ, m = 1 ∨ m = 2 ∨ m = 3 ∧
  ∀ n : ℕ, (∑ i in finset.range(n + 1), (1 / ∑ j in finset.range(i + 1), log 3 (a j - 1))) ≥ m / 3 :=
sorry

end seq_geom_exists_m_l760_760712


namespace sum_prime_numbers_less_than_twenty_l760_760033

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760033


namespace probability_x2_plus_y2_less_than_one_quarter_l760_760930

theorem probability_x2_plus_y2_less_than_one_quarter :
  let measure_space := MeasureTheory.MeasureSpace
  let interval := Set.Icc (-1 : ℝ) 1
  let measure := measure_space.volume;
  
  ∫ x in interval, 
  ∫ y in interval, 
  if (x^2 + y^2 < 1/4) then 1 else 0 ∂measure ∂measure = 
  (π / 16 : ℝ) := 
by
  sorry

end probability_x2_plus_y2_less_than_one_quarter_l760_760930


namespace Euler_theorem_l760_760633

theorem Euler_theorem {m a : ℕ} (hm : m ≥ 1) (h_gcd : Nat.gcd a m = 1) : a ^ Nat.totient m ≡ 1 [MOD m] :=
by
  sorry

end Euler_theorem_l760_760633


namespace cyclic_iff_side_relation_l760_760293

noncomputable def is_cyclic (A B C D : Point) : Prop :=
  ∃ (circ : Circle), A ∈ circ ∧ B ∈ circ ∧ C ∈ circ ∧ D ∈ circ

noncomputable theory

variables 
  (A B C D H O' N : Point)
  (a b c R : ℝ)
  (h_orthocenter : is_orthocenter H A B C)
  (h_circumcenter : is_circumcenter O' B H C)
  (h_midpoint : midpoint N A O')
  (h_reflection : reflection D N (line_through B C))
  (h_a_eq_BC : a = dist B C)
  (h_b_eq_CA : b = dist C A)
  (h_c_eq_AB : c = dist A B)
  (h_circumradius : R = circumradius A B C)

theorem cyclic_iff_side_relation :
  is_cyclic A B D C ↔ b^2 + c^2 - a^2 = 3 * R^2 :=
sorry

end cyclic_iff_side_relation_l760_760293


namespace base12_addition_correct_l760_760586

noncomputable theory

def base12_add (a b : string) : string := sorry -- Assume this function adds two base12 numbers correctly

def A43_base12 := "A43"
def 2B7_base12 := "2B7"
def 189_base12 := "189"
def result_base12 := "1C07"

theorem base12_addition_correct :
  base12_add (base12_add A43_base12 2B7_base12) 189_base12 = result_base12 := 
sorry

end base12_addition_correct_l760_760586


namespace sum_primes_less_than_20_l760_760085

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760085


namespace greatest_two_digit_multiple_of_17_l760_760478

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760478


namespace avg_of_eleven_numbers_l760_760853

variable (S1 : ℕ)
variable (S2 : ℕ)
variable (sixth_num : ℕ)
variable (total_sum : ℕ)
variable (avg_eleven : ℕ)

def condition1 := S1 = 6 * 58
def condition2 := S2 = 6 * 65
def condition3 := sixth_num = 188
def condition4 := total_sum = S1 + S2 - sixth_num
def condition5 := avg_eleven = total_sum / 11

theorem avg_of_eleven_numbers : (S1 = 6 * 58) →
                                (S2 = 6 * 65) →
                                (sixth_num = 188) →
                                (total_sum = S1 + S2 - sixth_num) →
                                (avg_eleven = total_sum / 11) →
                                avg_eleven = 50 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end avg_of_eleven_numbers_l760_760853


namespace slope_of_line_l760_760377

theorem slope_of_line (a b : ℝ) (h : b ≠ 0) (x y : ℝ) 
  (f : x = y * a + b) : a = 3 :=
by
  have h_line : ∀ x : ℝ, (∃ y : ℝ, y = 3 * x + 1) := 
  λ x, ⟨x, rfl⟩
  sorry

end slope_of_line_l760_760377


namespace factorize_3x2_minus_3y2_l760_760644

theorem factorize_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factorize_3x2_minus_3y2_l760_760644


namespace marthas_bedroom_size_l760_760886

theorem marthas_bedroom_size (M J : ℕ) 
  (h1 : M + J = 300)
  (h2 : J = M + 60) :
  M = 120 := 
sorry

end marthas_bedroom_size_l760_760886


namespace larger_fraction_l760_760624

theorem larger_fraction :
  (22222222221 : ℚ) / 22222222223 > (33333333331 : ℚ) / 33333333334 := by sorry

end larger_fraction_l760_760624


namespace meteorite_weight_possibilities_l760_760963

def valid_meteorite_weight_combinations : ℕ :=
  (2 * (Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2))) + (Nat.factorial 5)

theorem meteorite_weight_possibilities :
  valid_meteorite_weight_combinations = 180 :=
by
  -- Sorry added to skip the proof.
  sorry

end meteorite_weight_possibilities_l760_760963


namespace greatest_two_digit_multiple_of_17_l760_760503

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760503


namespace jumps_correct_l760_760837

def R : ℕ := 157
def X : ℕ := 86
def total_jumps (R X : ℕ) : ℕ := R + (R + X)

theorem jumps_correct : total_jumps R X = 400 := by
  sorry

end jumps_correct_l760_760837


namespace no_consecutive_squares_l760_760190

open Nat

-- Define a function to get the n-th prime number
def prime (n : ℕ) : ℕ := sorry -- Use an actual function or sequence that generates prime numbers, this is a placeholder.

-- Define the sequence S_n, the sum of the first n prime numbers
def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + prime (n + 1)

-- Define a predicate to check if a number is a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- The theorem that no two consecutive terms S_{n-1} and S_n can both be perfect squares
theorem no_consecutive_squares (n : ℕ) : ¬ (is_square (S n) ∧ is_square (S (n + 1))) :=
by
  sorry

end no_consecutive_squares_l760_760190


namespace find_T_b_plus_T_neg_b_l760_760187

noncomputable def T (r : ℝ) : ℝ := 15 / (1 - r)

theorem find_T_b_plus_T_neg_b (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 3600) :
  T b + T (-b) = 480 :=
sorry

end find_T_b_plus_T_neg_b_l760_760187


namespace average_speed_entire_trip_l760_760934

noncomputable def total_distance : ℝ := 40 + 180
noncomputable def total_time : ℝ := (40 / 20) + (180 / 60)
noncomputable def average_speed : ℝ := total_distance / total_time

theorem average_speed_entire_trip : average_speed = 44 := 
by
  -- conditions from the problem
  let local_road_distance := 40
  let local_road_speed := 20
  let highway_distance := 180
  let highway_speed := 60
    
  -- computation according to the conditions
  let total_distance := local_road_distance + highway_distance
  let time_local_roads := local_road_distance / local_road_speed
  let time_highway := highway_distance / highway_speed
  let total_time := time_local_roads + time_highway
  let computed_average_speed := total_distance / total_time

  -- desired check
  show average_speed = 44,
  from sorry

end average_speed_entire_trip_l760_760934


namespace base7_divisible_by_19_l760_760856

theorem base7_divisible_by_19 (x : ℕ) (h : x < 7) :
  let n := 5 * 7^3 + 2 * 7^2 + x * 7 + 3 in
  n % 19 = 0 ↔ x = 3 :=
by
  sorry

end base7_divisible_by_19_l760_760856


namespace greatest_two_digit_multiple_of_17_l760_760520

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760520


namespace pista_advantage_first_game_sanyi_advantage_second_game_overall_advantage_pista_l760_760828

noncomputable def total_possible_outcomes : ℕ := 81

def is_first_digit_seven_eight_nine (n : ℕ) : Prop :=
  let first_digit := n / 10^(nat.floor (real.log10 (n : ℝ)))
  first_digit = 7 ∨ first_digit = 8 ∨ first_digit = 9

def is_first_digit_one_two (n : ℕ) : Prop :=
  let first_digit := n / 10^(nat.floor (real.log10 (n : ℝ)))
  first_digit = 1 ∨ first_digit = 2

def is_first_digit_five_or_more (n : ℕ) : Prop :=
  let first_digit := n / 10^(nat.floor (real.log10 (n : ℝ)))
  first_digit ≥ 5

def is_first_digit_three_or_less (n : ℕ) : Prop :=
  let first_digit := n / 10^(nat.floor (real.log10 (n : ℝ)))
  first_digit ≤ 3

theorem pista_advantage_first_game :
  (2/9 : ℝ) > (1/9 : ℝ) → total_possible_outcomes = 81 → (Pista_wins_first_game : ℝ) > (Sanyi_wins_first_game : ℝ)
  ∧ long_term_advantage Pista
:= sorry

theorem sanyi_advantage_second_game :
  (5/9 : ℝ) > (4/9 : ℝ) → total_possible_outcomes = 81 → (Sanyi_wins_second_game : ℝ) > (Pista_wins_second_game : ℝ)
:= sorry

theorem overall_advantage_pista :
  total_possible_outcomes = 81 → (long_term_advantage Pista : ℝ) > (long_term_advantage Sanyi : ℝ)
:= sorry

end pista_advantage_first_game_sanyi_advantage_second_game_overall_advantage_pista_l760_760828


namespace fifth_term_geom_progression_l760_760864

-- Let the first three terms of the geometric progression be defined as follows
def a1 := 3^(-1 / 4 : ℝ)
def a2 := 3^(-1 / 5 : ℝ)
def a3 := 3^(-1 / 6 : ℝ)

-- Define the general form of the n-th term in the geometric sequence
def geom_term (n : ℕ) : ℝ := 3^(-1 / (n + 3) : ℝ)

-- The fifth term corresponds to n = 5
def a5 := geom_term 5

-- Prove that the fifth term is equal to 3^(-1 / 8)
theorem fifth_term_geom_progression : a5 = 3^(-1 / 8 : ℝ) := by
  sorry

end fifth_term_geom_progression_l760_760864


namespace find_rate_percent_l760_760533

theorem find_rate_percent (P : ℝ) (r : ℝ) (A1 A2 : ℝ) (t1 t2 : ℕ)
  (h1 : A1 = P * (1 + r)^t1) (h2 : A2 = P * (1 + r)^t2) (hA1 : A1 = 2420) (hA2 : A2 = 3146) (ht1 : t1 = 2) (ht2 : t2 = 3) :
  r = 0.2992 :=
by
  sorry

end find_rate_percent_l760_760533


namespace minimum_selling_price_per_unit_l760_760554

theorem minimum_selling_price_per_unit (units_produced : ℕ) (cost_per_unit : ℕ) (desired_profit : ℕ) :
  units_produced = 400 →
  cost_per_unit = 40 →
  desired_profit = 40000 →
  ∃ (selling_price : ℕ), selling_price = 140 :=
by
  assume h_units_produced h_cost_per_unit h_desired_profit
  let total_production_cost := cost_per_unit * units_produced
  let total_revenue_needed := total_production_cost + desired_profit
  let min_selling_price := total_revenue_needed / units_produced
  have h1: total_production_cost = 16000 := by rw [h_units_produced, h_cost_per_unit]; exact rfl
  have h2: total_revenue_needed = 56000 := by rw [h1, h_desired_profit]; exact rfl
  have h3: min_selling_price = 140 := by rw [h2, h_units_produced]; exact rfl
  use 140
  exact h3

end minimum_selling_price_per_unit_l760_760554


namespace no_real_solution_ratio_l760_760870

theorem no_real_solution_ratio (x : ℝ) : (x + 3) / (2 * x + 5) = (5 * x + 4) / (8 * x + 5) → false :=
by {
  sorry
}

end no_real_solution_ratio_l760_760870


namespace sum_of_primes_less_than_twenty_is_77_l760_760051

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760051


namespace max_schools_with_odd_players_l760_760595

-- Definitions based on conditions
variables {n : ℕ} {B G : ℕ} {B_i G_i : Fin n → ℕ}
variable {sum_B_i : ℕ}
variable {sum_G_i : ℕ}

-- Conditions
def condition1 : Prop :=
  ∀ i j : Fin n, i ≠ j → ((B_i i = 0 ∨ G_i i = 0) ∧ (B_i j = 0 ∨ G_i j = 0))

def condition2 : Prop :=
  ∀ i j : Fin n, i ≠ j → (B_i i > 0 ∨ G_i i > 0) ∧ (B_i j > 0 ∨ G_i j > 0)

def condition3 (i : Fin n) : Prop :=
  B_i i + G_i i % 2 = 0

def total_boys_girls_diff : Prop :=
  |B - G| ≤ 1

def total_singles_diff_mixed_singles : Prop :=
  let singles := (1 / 2 : ℚ) * (B ^ 2 - (B_i ∘ Fin.val).sum^2 + G ^ 2 - (G_i ∘ Fin.val).sum^2)
  let mixed_singles := B * G - (B_i ∘ Fin.val).sum * (G_i ∘ Fin.val).sum
  (singles - mixed_singles).nat_abs ≤ 1

-- Proof statement for Lean 4
theorem max_schools_with_odd_players 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : total_boys_girls_diff) 
  (h4 : total_singles_diff_mixed_singles) 
  : ∃ (k : ℕ), k ≤ 3
:= sorry

end max_schools_with_odd_players_l760_760595


namespace length_A_l760_760308

noncomputable def coord := (ℝ × ℝ)

def A : coord := (0, 6)
def B : coord := (0, 15)
def C : coord := (3, 9)
def lineYX (p : coord) : Prop := p.1 = p.2
def intersectsC (p : coord) : Prop := 
  ∃ m : ℝ, (C.2 = m * C.1 + p.2 - m * p.1)

theorem length_A'B' (A' B' : coord) 
  (hA' : lineYX A') 
  (hB' : lineYX B') 
  (hCA' : intersectsC A') 
  (hCB' : intersectsC B') 
  : dist A' B' = Real.sqrt 2 := sorry

end length_A_l760_760308


namespace greatest_two_digit_multiple_of_17_l760_760443

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760443


namespace cannot_win_l760_760384

-- Define the properties and the game rules
def sandwiches (N : ℕ) := 100 * N

-- Define the turns and rules for moves
structure Game (N : ℕ) :=
(Uf_moves : ℕ)
(M_moves : ℕ)
(Uf_move_first : Bool)

-- Define the winning condition for Uncle Fyodor
def Uf_wins (N : ℕ) (g : Game N) := ∀ matroskin_strategy : (fin (sandwiches N)) → Prop, ¬ matroskin_strategy (fin.last (sandwiches N))

-- Main theorem statement
theorem cannot_win (N : ℕ) : ∃ g : Game N, Uf_wins N g := 
sorry

end cannot_win_l760_760384


namespace sum_prime_numbers_less_than_twenty_l760_760042

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760042


namespace charlie_wins_probability_l760_760984

theorem charlie_wins_probability :
  let cards := {1, 2, 3, 4, 5, 6} in
  let players := {Alice, Bob, Charlie} in
  let distributions := (cards.to_list.permutations.filter (λ p, 
    p.take 2 ≠ p.drop 2.take 2 ∧ p.drop 4 ≠ p.take 2 ∧ p.drop 4 ≠ p.drop 2.take 2)) 
  in 
  let favorable_distributions := distributions.filter (λ p, 
    (median (p.take 2).head! (p.drop 2.take 2).head! (p.drop 4).head!) = 
    (p.drop 4).head!) in
  (favorable_distributions.length / distributions.length : ℚ) = 2 / 15 :=
sorry

end charlie_wins_probability_l760_760984


namespace average_difference_l760_760577

theorem average_difference (num_data_points wrong_value correct_value : ℕ) 
  (h_num_data_points : num_data_points = 30) (h_wrong_value : wrong_value = 15) (h_correct_value : correct_value = 75) :
  (correct_value - wrong_value) / num_data_points = -2 := 
by 
  sorry

end average_difference_l760_760577


namespace Paco_ate_cookies_l760_760826

def Paco_initial_cookies : ℕ := 40
def Paco_bought_cookies : ℕ := 37
def Paco_left_cookies : ℕ := 75

theorem Paco_ate_cookies :
  let total_cookies := Paco_initial_cookies + Paco_bought_cookies in
  let cookies_ate := total_cookies - Paco_left_cookies in
  cookies_ate = 2 :=
by
  let total_cookies := Paco_initial_cookies + Paco_bought_cookies
  let cookies_ate := total_cookies - Paco_left_cookies
  show cookies_ate = 2
  sorry

end Paco_ate_cookies_l760_760826


namespace Brenda_mice_left_l760_760998

theorem Brenda_mice_left :
  ∀ (total_litters total_each sixth factor remaining : ℕ),
    total_litters = 3 → 
    total_each = 8 →
    sixth = total_litters * total_each / 6 →
    factor = 3 * (total_litters * total_each / 6) →
    remaining = total_litters * total_each - sixth - factor →
    remaining / 2 = ((total_litters * total_each - sixth - factor) / 2) →
    total_litters * total_each - sixth - factor - ((total_litters * total_each - sixth - factor) / 2) = 4 :=
by
  intros total_litters total_each sixth factor remaining h_litters h_each h_sixth h_factor h_remaining h_half
  sorry

end Brenda_mice_left_l760_760998


namespace grayson_speed_first_hour_l760_760731

-- Definitions of the conditions
def grayson_first_hour_speed := ℝ
def grayson_first_hour_distance (S: grayson_first_hour_speed) := S
def grayson_second_half_hour_distance := 20 * 0.5
def grayson_total_distance (S: grayson_first_hour_speed) := grayson_first_hour_distance S + grayson_second_half_hour_distance
def rudy_total_distance := 10 * 3
def distance_difference (S: grayson_first_hour_speed) := grayson_total_distance S - rudy_total_distance

-- Theorem to prove
theorem grayson_speed_first_hour (S : ℝ) (h : distance_difference S = 5) : S = 25 :=
by
  -- Placeholder for the proof
  sorry

end grayson_speed_first_hour_l760_760731


namespace problem1_problem2_problem3_l760_760218

-- Definitions up front
def S (n : ℕ) : ℚ := (1/2 : ℚ) * n^2 + (11/2 : ℚ) * n
def a (n : ℕ+) : ℚ := if n = 1 then 6 else S n - S (n - 1)
def c (n : ℕ+) : ℚ := 1 / ((2 * a n - 11) * (2 * a n - 9))
def T (n : ℕ+) : ℚ := (Finset.range n).sum (λ k, c (k + 1))
def f (n : ℕ+) : ℚ := if Odd n then a n else 3 * a n - 13

-- Problem 1: General term formula of the sequence {a_n}
theorem problem1 (n : ℕ+) : a n = n + 5 := sorry

-- Problem 2: Maximum integer k such that T_n > k / 2014 holds for all n
theorem problem2 (k : ℕ) (h : ∀ n : ℕ+, T n > k / 2014) : k ≤ 670 := sorry

-- Problem 3: Existence of m such that f(m + 15) = 5f(m)
theorem problem3 : ∃ m : ℕ+, f (m + 15) = 5 * f m := sorry

end problem1_problem2_problem3_l760_760218


namespace candy_count_after_giving_l760_760588

def numKitKats : ℕ := 5
def numHersheyKisses : ℕ := 3 * numKitKats
def numNerds : ℕ := 8
def numLollipops : ℕ := 11
def numBabyRuths : ℕ := 10
def numReeseCups : ℕ := numBabyRuths / 2
def numLollipopsGivenAway : ℕ := 5

def totalCandyBefore : ℕ := numKitKats + numHersheyKisses + numNerds + numLollipops + numBabyRuths + numReeseCups
def totalCandyAfter : ℕ := totalCandyBefore - numLollipopsGivenAway

theorem candy_count_after_giving : totalCandyAfter = 49 := by
  sorry

end candy_count_after_giving_l760_760588


namespace cookies_recipes_count_l760_760994

theorem cookies_recipes_count 
  (total_students : ℕ)
  (attending_percentage : ℚ)
  (cookies_per_student : ℕ)
  (cookies_per_batch : ℕ) : 
  (total_students = 150) →
  (attending_percentage = 0.60) →
  (cookies_per_student = 3) →
  (cookies_per_batch = 18) →
  (total_students * attending_percentage * cookies_per_student / cookies_per_batch = 15) :=
by
  intros h1 h2 h3 h4
  sorry

end cookies_recipes_count_l760_760994


namespace a_n_eq_2_pow_n_b_n_eq_1_div_2n_minus_1_T_n_eq_sum_lambda_range_l760_760219

def sequence_a (n : ℕ) : ℕ :=
  if h : n > 0 then 2^n else 0

noncomputable def sequence_b (n : ℕ) : ℚ :=
  if h : n > 0 then 1 / (2 * n - 1) else 0

def sequence_c (n : ℕ) : ℚ :=
  if h : n > 0 then (sequence_a n) / (sequence_b n) else 0

noncomputable def sum_c (n : ℕ) : ℚ :=
  if h : n > 0 then 6 + (2 * n - 3) * 2^(n + 1) else 0

noncomputable def sequence_h (n : ℕ) : ℚ :=
  if h : n > 0 then (2 * n - 1) * (2 / 3)^n else 0

theorem a_n_eq_2_pow_n (n : ℕ) (h : n > 0) : 
  sequence_a n = 2^n := sorry

theorem b_n_eq_1_div_2n_minus_1 (n : ℕ) (h : n > 0) : 
  sequence_b n = 1 / (2 * n - 1) := sorry

theorem T_n_eq_sum (n : ℕ) (h : n > 0) : 
  (finset.range n).sum (λ i, sequence_c (i + 1)) = sum_c n := sorry

theorem lambda_range (λ : ℚ) (hn_ge_3 : ∀ n > 0, λ > sequence_h n) : 
  λ > 40 / 27 := sorry

end a_n_eq_2_pow_n_b_n_eq_1_div_2n_minus_1_T_n_eq_sum_lambda_range_l760_760219


namespace hexagon_angle_Q_zero_l760_760841

noncomputable def hexagon_angle : ℕ := 720
noncomputable def angles_of_hexagon (n : ℕ) : ℕ := hexagon_angle / n
noncomputable def total_angle (angle_count : ℕ → ℕ) (n : ℕ) : ℕ := angle_count n * n

/-- To prove that the degree measure of angle Q is zero, given that ABCDEF is a regular hexagon
    and AF and CD are extended to meet at point Q. -/
theorem hexagon_angle_Q_zero
  (hexagon_angle : ℕ := 720)
  (n : ℕ := 6)
  (regular_hexagon : ∀ n, n = 6)
  (angles_of_hexagon : ℕ := hexagon_angle / n)
  (total_angle : ℕ := angles_of_hexagon * n) :
  ∀ (point_Q : Type), ∃ (Q: point_Q), angles_of_hexagon = 120 → 
  angle_at_Q point_Q = 0
  :=
sorry

end hexagon_angle_Q_zero_l760_760841


namespace number_of_babies_in_quadruplets_l760_760629

-- Definitions for the given conditions
def setsOfQuadruplets := ℝ
def setsOfTriplets := 5 * setsOfQuadruplets
def setsOfTwins := 3 * setsOfTriplets
def totalBabies := 2 * setsOfTwins + 3 * setsOfTriplets + 4 * setsOfQuadruplets

-- Theorem statement: proving the equivalence given the conditions
theorem number_of_babies_in_quadruplets (c : ℝ) (h1 : setsOfTriplets = 5 * c) 
(h2 : setsOfTwins = 15 * c) (h3 : 2 * setsOfTwins + 3 * setsOfTriplets + 4 * c = 1250) 
: 4 * c = 5000 / 49 :=
by
  sorry

end number_of_babies_in_quadruplets_l760_760629


namespace parabola_zero_difference_l760_760370

theorem parabola_zero_difference (a b c : ℝ) (m n : ℝ) (h₁ : m > n)
  (h₂ : ∀ x, y = a * x^2 + b * x + c)
  (h₃ : y = 4 * (x - 3)^2 - 2)
  (h₄ : (∀ x y, (x, y) = (3, -2)) → (x = 5, y = 14))
  (h₅ : ∀ x, (x = 3 + sqrt 2 / 2) ∨ (x = 3 - sqrt 2 / 2)) :
  m - n = sqrt 2 := 
sorry

end parabola_zero_difference_l760_760370


namespace sin_pi_div_six_eq_one_half_l760_760545

theorem sin_pi_div_six_eq_one_half : sin (π / 6) = 1 / 2 :=
sorry

end sin_pi_div_six_eq_one_half_l760_760545


namespace sum_of_primes_less_than_twenty_is_77_l760_760046

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760046


namespace find_phase_shift_l760_760366

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)

-- Define the shifted function g
def g (x ϕ : ℝ) : ℝ := 2 * Real.cos (2 * x - 2 * ϕ)

-- The main theorem to prove
theorem find_phase_shift (ϕ : ℝ) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π / 2) :
  (∃ x1 x2 : ℝ, |f x1 - g x2 ϕ| = 4 ∧ |x1 - x2| = π / 6) →
  ϕ = π / 3 :=
by
  sorry

end find_phase_shift_l760_760366


namespace seats_taken_correct_l760_760667

-- Define the conditions
def rows := 40
def chairs_per_row := 20
def unoccupied_seats := 10

-- Define the total number of seats
def total_seats := rows * chairs_per_row

-- Define the number of seats taken
def seats_taken := total_seats - unoccupied_seats

-- Statement of our math proof problem
theorem seats_taken_correct : seats_taken = 790 := by
  sorry

end seats_taken_correct_l760_760667


namespace plastic_bag_co2_release_l760_760818

def total_co2_canvas_bag_lb : ℕ := 600
def total_co2_canvas_bag_oz : ℕ := 9600
def plastic_bags_per_trip : ℕ := 8
def shopping_trips : ℕ := 300

theorem plastic_bag_co2_release :
  total_co2_canvas_bag_oz = 2400 * 4 :=
by
  sorry

end plastic_bag_co2_release_l760_760818


namespace arc_length_of_sector_l760_760707

theorem arc_length_of_sector (r : ℝ) (h_area : (1/2) * 2 * r^2 = 4) (h_angle : 2): (2 * r = 4) :=
by sorry

end arc_length_of_sector_l760_760707


namespace problem_part1_problem_part2_l760_760714

noncomputable def f : ℝ → ℝ := λ x, 2 * cos(2 * x + π / 3) - 2 * cos x + 1

theorem problem_part1 :
  (∃ A ω φ B, f = (λ x, A * sin (ω * x + φ) + B)) ∧
  (∃ k : ℤ, ∀ x, f x = -2 * sin (2 * x + π / 6) + 1 → (x = -π / 12 + k * π / 2) → (1 = 1)) :=
by sorry

theorem problem_part2 (A B C a b c : ℝ) (hA : f A = 0) (h_acute_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sides : a = sin A ∧ b = sin B ∧ c = sin C) : 
  (1 / 2) < (b / c) ∧ (b / c) < 2 :=
by sorry

end problem_part1_problem_part2_l760_760714


namespace solve_system_l760_760845

open Real

-- Define the system of equations as hypotheses
def eqn1 (x y z : ℝ) : Prop := x + y + 2 - 4 * x * y = 0
def eqn2 (x y z : ℝ) : Prop := y + z + 2 - 4 * y * z = 0
def eqn3 (x y z : ℝ) : Prop := z + x + 2 - 4 * z * x = 0

-- State the theorem
theorem solve_system (x y z : ℝ) :
  (eqn1 x y z ∧ eqn2 x y z ∧ eqn3 x y z) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by 
  sorry

end solve_system_l760_760845


namespace greatest_two_digit_multiple_of_17_l760_760486

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760486


namespace perpendicular_vectors_lambda_l760_760234

variables (λ : ℝ)

/-- Given vectors a = (1,3) and b = (3,λ), if a is perpendicular to b, then λ = -1. -/
theorem perpendicular_vectors_lambda :
  let a := (1 : ℝ, 3), b := (3 : ℝ, λ) in
  a.1 * b.1 + a.2 * b.2 = 0 -> λ = -1 :=
by
  intros
  rw [←add_eq_zero_iff_eq_neg, ←mul_eq_zero] at ‹1 * 3 + 3 * λ = 0› -- process the input condition
  sorry

end perpendicular_vectors_lambda_l760_760234


namespace train_speed_is_36_0036_kmph_l760_760581

noncomputable def train_length : ℝ := 130
noncomputable def bridge_length : ℝ := 150
noncomputable def crossing_time : ℝ := 27.997760179185665
noncomputable def speed_in_kmph : ℝ := (train_length + bridge_length) / crossing_time * 3.6

theorem train_speed_is_36_0036_kmph :
  abs (speed_in_kmph - 36.0036) < 0.001 :=
by
  sorry

end train_speed_is_36_0036_kmph_l760_760581


namespace unique_circle_arrangement_l760_760268

def is_divisor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def valid_circle_arrangement (arrangement : List ℕ) (n : ℕ) : Prop :=
  (arrangement.length = 2 * n) ∧
  (arrangement.nodup) ∧
  (∀ i : ℕ, is_divisor (arrangement.get i (nat.zero_lt_two_mul n)) 
                   (arrangement.get (i + 1) % (2 * n) + arrangement.get (i - 1 + 2 * n) % (2 * n)))

theorem unique_circle_arrangement (n : ℕ) :
  (card (set_of (λ arrangement : List ℕ, valid_circle_arrangement arrangement n)) = 1) :=
sorry

end unique_circle_arrangement_l760_760268


namespace marthas_bedroom_size_l760_760884

theorem marthas_bedroom_size (M J : ℕ) 
  (h1 : M + J = 300)
  (h2 : J = M + 60) :
  M = 120 := 
sorry

end marthas_bedroom_size_l760_760884


namespace greatest_two_digit_multiple_of_17_l760_760403

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760403


namespace max_value_func_l760_760947

-- Define the function and the conditions
noncomputable def func (a b x : ℝ) : ℝ := a * real.sin x + b

theorem max_value_func (a b : ℝ) (h : a < 0) :
  (∃ x : ℝ, func a b x = -a + b) :=
begin
  sorry
end

end max_value_func_l760_760947


namespace greatest_two_digit_multiple_of_17_l760_760470

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760470


namespace cross_section_area_of_cone_l760_760368

theorem cross_section_area_of_cone (h : ℝ) (r : ℝ) (d : ℝ) :
  h = 20 → r = 25 → d = 12 → 
  let α := real.arcsin (d / h) in
  let OM := r * real.cos α in
  let AM := h * real.cos α in
  let BM := real.sqrt (r^2 - OM^2) in
  let BC := 2 * BM in
  let S := 0.5 * BC * AM in
  S = 500 :=
by
  intros h_eq r_eq d_eq α OM AM BM BC S
  sorry

end cross_section_area_of_cone_l760_760368


namespace sum_prime_numbers_less_than_twenty_l760_760027

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760027


namespace train_length_correct_l760_760136

namespace TrainLengthProof

-- Define the given quantities
def train_speed_kmph : ℝ := 62
def man_speed_kmph : ℝ := 8
def passing_time_seconds : ℝ := 9.99920006399488
def relative_speed_mps := (train_speed_kmph - man_speed_kmph) * (1000 / 3600)
def expected_train_length : ℝ := 149.99

-- The theorem to prove
theorem train_length_correct :
  (relative_speed_mps * passing_time_seconds) = expected_train_length :=
sorry

end TrainLengthProof

end train_length_correct_l760_760136


namespace smallest_possible_n_l760_760340

theorem smallest_possible_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : n > 20) : n = 52 := 
sorry

end smallest_possible_n_l760_760340


namespace market_trips_l760_760381

theorem market_trips (d_school_round: ℝ) (d_market_round: ℝ) (num_school_trips_per_day: ℕ) (num_school_days_per_week: ℕ) (total_week_mileage: ℝ) :
  d_school_round = 5 →
  d_market_round = 4 →
  num_school_trips_per_day = 2 →
  num_school_days_per_week = 4 →
  total_week_mileage = 44 →
  (total_week_mileage - (d_school_round * num_school_trips_per_day * num_school_days_per_week)) / d_market_round = 1 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end market_trips_l760_760381


namespace grown_ups_in_milburg_l760_760900

def number_of_children : ℕ := 2987
def total_population : ℕ := 8243

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 := 
by 
  sorry

end grown_ups_in_milburg_l760_760900


namespace max_sin_angle_l760_760392

noncomputable def sin_angle_max_value (P : ℝ → ℝ × ℝ) (A B : ℝ × ℝ) : ℝ :=
  let P' := P 0
  sin (angle A P' B)

theorem max_sin_angle (A B : ℝ × ℝ) (P : ℝ → ℝ × ℝ)
  (hA : A = (0, 2)) (hB : B = (0, 4))
  (hP : ∀ x, P x = (x, 0)) :
  ∃ x, sin_angle_max_value P A B = 1/3 :=
by
  sorry

end max_sin_angle_l760_760392


namespace vehicle_speeds_l760_760910

theorem vehicle_speeds (d t: ℕ) (b_speed c_speed : ℕ) (h1 : d = 80) (h2 : c_speed = 3 * b_speed) (h3 : t = 3) (arrival_difference : ℕ) (h4 : arrival_difference = 1 / 3):
  b_speed = 20 ∧ c_speed = 60 :=
by
  sorry

end vehicle_speeds_l760_760910


namespace greatest_two_digit_multiple_of_17_l760_760515

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760515


namespace area_of_extended_quadrilateral_l760_760331

-- Define a quadrilateral that is designated convex
structure Quadrilateral :=
(A B C D : ℝ)

-- Define the areas of the quadrilaterals
def area (Q : Quadrilateral) : ℝ := sorry

-- Define that sides are extended as given in the problem
structure ExtendedQuadrilateral extends Quadrilateral :=
(B1 C1 D1 A1 : ℝ)
(BB1_eq_AB : B1 = A + (B - A))
(CC1_eq_BC : C1 = B + (C - B))
(DD1_eq_CD : D1 = C + (D - C))
(AA1_eq_AD : A1 = D + (A - D))

-- Define the problem statement in Lean
theorem area_of_extended_quadrilateral (Q : Quadrilateral) (EQ : ExtendedQuadrilateral Q) :
  area (ExtendedQuadrilateral.toQuadrilateral EQ) = 5 * area Q :=
sorry

end area_of_extended_quadrilateral_l760_760331


namespace houses_with_neither_l760_760100

def total_houses : ℕ := 90
def two_car_garage : ℕ := 50
def swimming_pool : ℕ := 40
def both_amenities : ℕ := 35

theorem houses_with_neither :
  total_houses - (two_car_garage + swimming_pool - both_amenities) = 35 :=
by
  let houses_with_either := two_car_garage + swimming_pool - both_amenities
  calc
    total_houses - houses_with_either
        = 90 - (50 + 40 - 35) : by sorry
    ... = 35 : by sorry

end houses_with_neither_l760_760100


namespace inequality_product_geq_two_power_n_equality_condition_l760_760310

open Real BigOperators

noncomputable def is_solution (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i ∧ a i = 1

theorem inequality_product_geq_two_power_n (a : ℕ → ℝ) (n : ℕ)
  (h1 : ( ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i))
  (h2 : ∑ i in Finset.range n, a (i + 1) = n) :
  (∏ i in Finset.range n, (1 + 1 / a (i + 1))) ≥ 2 ^ n :=
sorry

theorem equality_condition (a : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i)
  (h2 : ∑ i in Finset.range n, a (i + 1) = n):
  (∏ i in Finset.range n, (1 + 1 / a (i + 1))) = 2 ^ n ↔ is_solution a n :=
sorry

end inequality_product_geq_two_power_n_equality_condition_l760_760310


namespace greatest_two_digit_multiple_of_17_l760_760507

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760507


namespace greatest_two_digit_multiple_of_17_l760_760435

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760435


namespace find_scores_l760_760126

theorem find_scores (A B : ℕ) (S : Finset ℕ) (hApos : A > 0) (hBpos : B > 0) (hA_neq_B : A ≠ B) (hS_card : S.card = 35) (h58_in_S : 58 ∈ S) :
    ∀ (a b : ℕ), a ≠ b → Finset.card (Finset.filter (λ n => ∀ x y : ℕ, n ≠ (x * A + y * B)) (Finset.range (A * B)) ) = S.card → A = 11 ∧ B = 8 :=
by
  sorry

end find_scores_l760_760126


namespace emily_jumps_75_seconds_l760_760530

/-- Emily jumps 52 times in 60 seconds, maintaining the same rate.
    Prove that she jumps 65 times in 75 seconds. -/
theorem emily_jumps_75_seconds (
    jumps_per_60_seconds : ℚ := 52 / 60
) : (75 * jumps_per_60_seconds = 65) :=
by
  -- Simplify the jumps per 60 seconds rate
  have rate : ℚ := 13 / 15

  -- Calculate the jumps in 75 seconds
  have jumps_75 : ℚ := 75 * rate

  -- Conclude that the number of jumps in 75 seconds is 65
  show jumps_75 = 65, by
    calc
      jumps_75 = 75 * (13 / 15) : by rw [rate]
      ... = (75 / 15) * 13 : by rw [mul_div_assoc]
      ... = 5 * 13 : by norm_num
      ... = 65 : by norm_num

end emily_jumps_75_seconds_l760_760530


namespace sum_primes_less_than_20_l760_760068

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760068


namespace sum_of_primes_less_than_20_l760_760007

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760007


namespace dihedral_angle_sum_equality_l760_760834

-- Define the points A1, A2, A3, A4 representing the vertices of the tetrahedron
variables (A1 A2 A3 A4 : Point)

-- Define a tetrahedron with vertices A1, A2, A3, A4
def tetrahedron := {A1A2, A1A3, A1A4, A2A3, A2A4, A3A4}

-- Given condition: the sum of two opposite edges is equal to the sum of another two opposite edges
axiom edge_sum_condition : (distance A1 A3) + (distance A2 A4) = (distance A1 A4) + (distance A3 A2)

-- Define the dihedral angles at edges (Note: additional definitions/notations might be needed here, 
-- depending on Mathlib's approach to geometry and dihedral angles)
def dihedral_angle (e1 e2 : edge) : Angle := sorry

-- Define the theorem to prove the equality of sums of corresponding dihedral angles given the condition
theorem dihedral_angle_sum_equality :
  (dihedral_angle A1 A2 A3 + dihedral_angle A4 A2 A3) = (dihedral_angle A1 A2 A4 + dihedral_angle A3 A2 A4) :=
sorry

end dihedral_angle_sum_equality_l760_760834


namespace find_other_number_l760_760866

theorem find_other_number (b : ℕ) (lcm_val gcd_val : ℕ)
  (h_lcm : Nat.lcm 240 b = 2520)
  (h_gcd : Nat.gcd 240 b = 24) :
  b = 252 :=
sorry

end find_other_number_l760_760866


namespace three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l760_760927

theorem three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693 :
  ∃ (n : ℕ), n = 693 ∧ 
    (100 * 6 + 10 * (n / 10 % 10) + 3) = n ∧
    (n % 10 = 3) ∧
    (n / 100 = 6) ∧
    n % 11 = 0 :=
by
  sorry

end three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l760_760927


namespace girls_boys_ratio_l760_760820

-- Let g be the number of girls and b be the number of boys.
-- From the conditions, we have:
-- 1. Total students: g + b = 32
-- 2. More girls than boys: g = b + 6

theorem girls_boys_ratio
  (g b : ℕ) -- Declare number of girls and boys as natural numbers
  (h1 : g + b = 32) -- Total number of students
  (h2 : g = b + 6)  -- 6 more girls than boys
  : g = 19 ∧ b = 13 := 
sorry

end girls_boys_ratio_l760_760820


namespace sum_prime_numbers_less_than_twenty_l760_760037

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760037


namespace maggie_sold_2_subscriptions_to_neighbor_l760_760317

-- Definition of the problem conditions
def maggie_pays_per_subscription : Int := 5
def maggie_subscriptions_to_parents : Int := 4
def maggie_subscriptions_to_grandfather : Int := 1
def maggie_earned_total : Int := 55

-- Define the function to be proven
def subscriptions_sold_to_neighbor (x : Int) : Prop :=
  maggie_pays_per_subscription * (maggie_subscriptions_to_parents + maggie_subscriptions_to_grandfather + x + 2*x) = maggie_earned_total

-- The statement we need to prove
theorem maggie_sold_2_subscriptions_to_neighbor :
  subscriptions_sold_to_neighbor 2 :=
sorry

end maggie_sold_2_subscriptions_to_neighbor_l760_760317


namespace factor_1024_into_three_factors_l760_760269

theorem factor_1024_into_three_factors : 
  (∃ (a b c : ℕ), a + b + c = 10 ∧ a ≥ b ∧ b ≥ c) → ∑ (h : ∃ (a b c : ℕ), a + b + c = 10 ∧ a ≥ b ∧ b ≥ c), 1 = 14 :=
by
  sorry

end factor_1024_into_three_factors_l760_760269


namespace sum_primes_less_than_20_l760_760061

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760061


namespace find_parabola_equation_area_of_triangle_eq_24_sqrt_5_l760_760203

-- Definitions of the conditions
def parabola (p : ℝ) : Set (ℝ × ℝ) := { point | point.2^2 = 2 * p * point.1 }
def focus (p : ℝ) : (ℝ × ℝ) := (p, 0)
def line_l1 : (ℝ × ℝ) → Prop := λ point, point.2 = -point.1
def intersection_condition : (ℝ × ℝ) := (8, -8)

-- Problems (Parts I and II)
theorem find_parabola_equation :
  ∃ p : ℝ, parabola p intersection_condition ∧ parabola p = { point | point.2^2 = 8 * point.1 } :=
sorry

-- Definitions for Part II
def line_l2 : Set (ℝ × ℝ) := { point | point.1 = point.2 + 8 }
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola 4 A ∧ parabola 4 B ∧ line_l2 A ∧ line_l2 B

theorem area_of_triangle_eq_24_sqrt_5 (A B : ℝ × ℝ) (F : ℝ × ℝ) :
  intersection_condition ∧
  intersection_points A B ∧
  (A.1 + B.1) / 2 = 4 ∧
  (A.2 + B.2) / 2 = 0 ∧
  F = (4, 0) →
  area_of_triangle F A B = 24 * real.sqrt 5 :=
sorry

-- Auxiliary functions and definitions
noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

end find_parabola_equation_area_of_triangle_eq_24_sqrt_5_l760_760203


namespace greatest_two_digit_multiple_of_17_l760_760439

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760439


namespace range_of_g_l760_760180

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : Set.Icc (-1.1071) 1.1071 = Set.image g (Set.Icc (-1:ℝ) 1) := by
  sorry

end range_of_g_l760_760180


namespace total_simple_interest_l760_760578

-- Define the principal (P), the rate (R) in percent, and the time (T) in years.
def principal : ℝ := 5737.5
def rate : ℝ := 14
def time : ℝ := 5

-- Define the formula for calculating simple interest.
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- State the theorem that proves the computed simple interest is 802.25.
theorem total_simple_interest : simple_interest principal rate time = 802.25 :=
by
  sorry

end total_simple_interest_l760_760578


namespace Bob_spends_135_dollars_on_this_shopping_trip_l760_760174

def price_hammer := 15
def quantity_hammer := 4

def price_nails := 3
def quantity_nails := 6

def price_saw := 12
def quantity_saw := 3
def discount_saw := 0.75

def price_paint := 20
def quantity_paint := 2

def order_discount_threshold := 50
def coupon_discount := 10

-- Proof statement
theorem Bob_spends_135_dollars_on_this_shopping_trip :
  let total_hammer := price_hammer * quantity_hammer in
  let total_nails := price_nails * quantity_nails in
  let total_saw := price_saw * discount_saw * quantity_saw in
  let total_paint := price_paint * quantity_paint in
  let total_before_coupon := total_hammer + total_nails + total_saw + total_paint in
  total_before_coupon >= order_discount_threshold →
  total_before_coupon - coupon_discount = 135 :=
by
  sorry

end Bob_spends_135_dollars_on_this_shopping_trip_l760_760174


namespace avg_fixed_points_is_one_l760_760566

-- Define the set of permutations of {1, 2, ..., n}
noncomputable def permutations (n : ℕ) : List (List ℕ) :=
  List.permutations (List.range n)

-- Define the number of fixed points of a permutation
def fixedPoints (σ : List ℕ) : ℕ :=
  σ.enum.filter (λ (ix : ℕ × ℕ), ix.1 = ix.2).length

-- Averaging function over list
def avg {α : Type*} [Add α] [Zero α] [Div α ℕ] (xs : List α) : α :=
  if h : xs ≠ [] then xs.sum / xs.length else 0

-- Define the average number of fixed points
noncomputable def avg_fixed_points_permutation (n : ℕ) : ℕ :=
  avg (permutations n).map fixedPoints

-- The theorem: the average number of fixed points in a random permutation
theorem avg_fixed_points_is_one (n : ℕ) : avg_fixed_points_permutation n = 1 :=
by
  sorry

end avg_fixed_points_is_one_l760_760566


namespace plane_total_seats_l760_760986

theorem plane_total_seats (s : ℕ) : 24 + 0.30 * s + (2 / 3) * s = s → s = 720 :=
by
  sorry

end plane_total_seats_l760_760986


namespace smallest_possible_value_l760_760296

theorem smallest_possible_value (a b c : ℤ)
  (B : Matrix (Fin 2) (Fin 2) ℝ := 1 / 4 *! Matrix.of ![-4, a; b, c])
  (cond : B * B = -Matrix.identity (Fin 2)) :
  a + b + c = 0 :=
sorry

end smallest_possible_value_l760_760296


namespace part_one_part_two_l760_760811

variables {n : ℕ}
variables {a : Fin n → ℝ}
variables {x : Fin n → ℝ}

/-- Define d_i as per the given specification -/
def d_i (i : Fin n) :=
  let a_max_j := Finset.sup (Finset.filter (λ j, j ≤ i) Finset.univ) (λ j, a j)
  let a_min_j := Finset.inf (Finset.filter (λ j, i ≤ j) Finset.univ) (λ j, a j)
  (a_max_j - a_min_j)

/-- Define d as per the given specification -/
def d : ℝ :=
  Finset.sup (Finset.univ : Finset (Fin n)) d_i

/-- The first part of the problem: For any non-decreasing sequence x_i, 
     the inequality \max \left\{ \left| x_i - a_i \right| \mid 1 \leqslant i \leqslant n \right\} \geqslant \frac{d}{2} holds. -/
theorem part_one (hx : ∀ i j : Fin n, i ≤ j → x i ≤ x j) :
  Finset.sup (Finset.image (λ i : Fin n, | x i - a i |) Finset.univ) id ≥ d / 2 :=
sorry

/-- There exist non-decreasing sequences such that equality in the previous inequality holds -/
theorem part_two :
  ∃ x : Fin n → ℝ,
    (∀ i j : Fin n, i ≤ j → x i ≤ x j) ∧ 
    Finset.sup (Finset.image (λ i : Fin n, |x i - a i|) Finset.univ) id = d / 2 :=
sorry

end part_one_part_two_l760_760811


namespace factorize_3x2_minus_3y2_l760_760645

theorem factorize_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factorize_3x2_minus_3y2_l760_760645


namespace cost_of_trees_l760_760734

theorem cost_of_trees (fence_length_yds : ℕ) (tree_width_ft : ℝ) (tree_cost : ℝ) (yard_to_feet : ℝ) :
  fence_length_yds = 25 →
  tree_width_ft = 1.5 →
  tree_cost = 8.0 →
  yard_to_feet = 3 →
  50 * tree_cost = 400 :=
by
  intros h1 h2 h3 h4
  have h5 : fence_length_yds * yard_to_feet = 75 := by sorry -- Convert fence length to feet
  have h6 : fence_length_yds * yard_to_feet / tree_width_ft = 50 := by sorry -- Calculate number of trees
  rw [h3]
  have h7 : 50 * 8 = 400 := by sorry -- Calculate total cost
  exact h7

end cost_of_trees_l760_760734


namespace min_value_a_plus_8b_min_value_a_plus_8b_min_l760_760199

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  a + 8 * b ≥ 9 :=
by sorry

-- The minimum value is 9 (achievable at specific values of a and b)
theorem min_value_a_plus_8b_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  ∃ a b, a > 0 ∧ b > 0 ∧ 2 * a * b = a + 2 * b ∧ a + 8 * b = 9 :=
by sorry

end min_value_a_plus_8b_min_value_a_plus_8b_min_l760_760199


namespace find_x_value_l760_760382

theorem find_x_value :
  (∃ x : ℝ, ((0.02)^2 + (0.52)^2 + x^2) / ((0.002)^2 + (0.052)^2 + (0.0035)^2) = 100 ∧ x = 0.035) := 
sorry

end find_x_value_l760_760382


namespace distance_between_foci_l760_760988

-- Define the properties of the ellipse
def ellipse_center := (3 : ℝ, 2 : ℝ)
def ellipse_tangent_x_axis := (3 : ℝ, 0 : ℝ)
def ellipse_tangent_y_axis := (0 : ℝ, 2 : ℝ)

-- Semi-major and semi-minor axes
def a : ℝ := 3
def b : ℝ := 2

-- Formula for the distance between the foci
theorem distance_between_foci : 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 5 := by
  sorry

end distance_between_foci_l760_760988


namespace sum_primes_less_than_20_l760_760084

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760084


namespace sequence_general_term_l760_760782

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a (n + 1) = 2^n * a n) : 
  ∀ n, a n = 2^((n-1)*n / 2) := sorry

end sequence_general_term_l760_760782


namespace sum_of_primes_less_than_twenty_is_77_l760_760057

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760057


namespace martha_bedroom_size_l760_760883

theorem martha_bedroom_size (x jenny_size total_size : ℤ) (h₁ : jenny_size = x + 60) (h₂ : total_size = x + jenny_size) (h_total : total_size = 300) : x = 120 :=
by
  -- Adding conditions and the ultimate goal
  sorry


end martha_bedroom_size_l760_760883


namespace additional_track_length_l760_760973

theorem additional_track_length (rise : ℝ) (grade1 grade2 : ℝ) (h1 : grade1 = 0.04) (h2 : grade2 = 0.02) (h3 : rise = 800) :
  ∃ (additional_length : ℝ), additional_length = (rise / grade2 - rise / grade1) ∧ additional_length = 20000 :=
by
  sorry

end additional_track_length_l760_760973


namespace find_ab_l760_760698

theorem find_ab (A B : Set ℝ) (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = Set.univ) → 
  (A ∩ B = {x | 3 < x ∧ x ≤ 4}) →
  a + b = -7 :=
by
  intros
  sorry

end find_ab_l760_760698


namespace backpack_swap_meet_price_l760_760974

theorem backpack_swap_meet_price
  (cost_per_case : ℕ := 576)
  (total_backpacks : ℕ := 48)
  (sold_at_swap_meet : ℕ := 17)
  (sold_to_department_store : ℕ := 10)
  (price_department_store : ℕ := 25)
  (remainder_price : ℕ := 22)
  (profit : ℕ := 442)
  (x : ℕ) :
  let department_store_revenue := sold_to_department_store * price_department_store,
      remainder := total_backpacks - sold_at_swap_meet - sold_to_department_store,
      remainder_revenue := remainder * remainder_price,
      total_revenue := department_store_revenue + remainder_revenue in
  (17 * x + department_store_revenue + remainder_revenue = cost_per_case + profit) →
  x = 18 :=
by
  sorry

end backpack_swap_meet_price_l760_760974


namespace fish_population_estimate_l760_760537

theorem fish_population_estimate
  (N : ℕ) 
  (tagged_initial : ℕ)
  (caught_again : ℕ)
  (tagged_again : ℕ)
  (h1 : tagged_initial = 60)
  (h2 : caught_again = 60)
  (h3 : tagged_again = 2)
  (h4 : (tagged_initial : ℚ) / N = (tagged_again : ℚ) / caught_again) :
  N = 1800 :=
by
  sorry

end fish_population_estimate_l760_760537


namespace sum_of_primes_less_than_20_l760_760005

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760005


namespace abs_sum_lt_abs_l760_760200

theorem abs_sum_lt_abs (a b : ℝ) (h : a * b < 0) : |a + b| < |a| + |b| :=
sorry

end abs_sum_lt_abs_l760_760200


namespace hyperbola_foci_difference_l760_760214

noncomputable def hyperbola_foci_distance (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (a : ℝ) : ℝ :=
  |dist P F₁ - dist P F₂|

theorem hyperbola_foci_difference (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : 
  (P.1 ^ 2 - P.2 ^ 2 = 4) ∧ (P.1 < 0) → (hyperbola_foci_distance P F₁ F₂ 2 = -4) :=
by
  intros h
  sorry

end hyperbola_foci_difference_l760_760214


namespace circle_radius_10_l760_760192

theorem circle_radius_10 (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0 → (x + 7) ^ 2 + (y + 4) ^ 2 = 100) ↔ (k = 35) :=
begin
  sorry
end

end circle_radius_10_l760_760192


namespace coeffs_equal_implies_a_plus_b_eq_4_l760_760778

theorem coeffs_equal_implies_a_plus_b_eq_4 (a b : ℕ) (h_rel_prime : Nat.gcd a b = 1) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq_coeffs : (Nat.choose 2000 1998) * (a ^ 2) * (b ^ 1998) = (Nat.choose 2000 1997) * (a ^ 3) * (b ^ 1997)) :
  a + b = 4 := 
sorry

end coeffs_equal_implies_a_plus_b_eq_4_l760_760778


namespace greatest_two_digit_multiple_of_17_l760_760477

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760477


namespace grown_ups_in_milburg_l760_760899

def number_of_children : ℕ := 2987
def total_population : ℕ := 8243

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 := 
by 
  sorry

end grown_ups_in_milburg_l760_760899


namespace three_digit_number_divisible_by_11_l760_760925

theorem three_digit_number_divisible_by_11 : 
  (∀ x : ℕ, x < 10 → (600 + 10 * x + 3) % 11 = 0 → 600 + 10 * x + 3 = 693) :=
by 
  intros x x_lt_10 h 
  have h1 : 600 % 11 = 7 := by norm_num
  have h2 : (10 * x + 3) % 11 = (10 * x + 3) % 11 := by norm_num
  rw Nat.add_mod at h 
  rw [h1, h2] at h 
  have h3 : (7 + (10 * x + 3) % 11) % 11 = 0 := by rw ← h 
  rw Nat.add_mod at h3 
  cases x 
  case h_0 => rw zero_mul at * 
             simp at h3 
             norm_num at h3
  cases x 
  case h_0 => sorry -- Assume this case has been proved
  case h_succ x_1 => sorry -- Assume this case has been proved
  sorry

end three_digit_number_divisible_by_11_l760_760925


namespace greatest_two_digit_multiple_of_17_l760_760445

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760445


namespace min_distance_origin_l760_760205

theorem min_distance_origin (a b : ℝ) (h: 3 * a + 4 * b = 15) : sqrt (a^2 + b^2) = 3 :=
sorry

end min_distance_origin_l760_760205


namespace parallelogram_D_l760_760271

noncomputable def A : ℂ := 2 + 1 * complex.i
noncomputable def B : ℂ := 4 + 3 * complex.i
noncomputable def C : ℂ := 3 + 5 * complex.i

theorem parallelogram_D (D : ℂ) :
  (D = 1 + 3 * complex.i ∨ D = 5 + 7 * complex.i ∨ D = 3 - complex.i) ↔
  (A + C = B + D ∨ A + D = B + C ∨ A + B = D + C) := sorry

end parallelogram_D_l760_760271


namespace find_original_price_l760_760790

def original_price (P : ℝ) : Prop :=
  let first_discount_price := P - 0.25 * P
  let final_price := first_discount_price - 0.25 * first_discount_price
  final_price = 15

theorem find_original_price : ∃ P : ℝ, original_price P ∧ P = 26.67 :=
begin
  use 26.67,
  unfold original_price,
  sorry -- Skip the proof
end

end find_original_price_l760_760790


namespace exactly_one_true_proposition_l760_760728

variable (α β γ : Plane)
variable (m n l : Line)

-- Definitions of the propositions
def prop1 : Prop := perpendicular m l ∧ perpendicular n l → parallel m n
def prop2 : Prop := perpendicular α γ ∧ perpendicular β γ → parallel α β
def prop3 : Prop := perpendicular m α ∧ parallel m n ∧ lies_in n β → perpendicular α β
def prop4 : Prop := parallel m α ∧ intersection α β = n → parallel m n

-- Proof that exactly one proposition is true
theorem exactly_one_true_proposition : (prop1 α β γ m n l → False) ∧
                                      (prop2 α β γ → False) ∧
                                      prop3 α β γ m n l ∧
                                      (prop4 α β m n l → False) :=
by sorry

end exactly_one_true_proposition_l760_760728


namespace greatest_two_digit_multiple_of_17_l760_760496

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760496


namespace exists_convex_ngon_with_side_lengths_tangent_l760_760669

theorem exists_convex_ngon_with_side_lengths_tangent
  (n : ℕ) (hn : n ≥ 4) :
  (∃ (a : ℕ → ℕ) (hperm : ∀ i, a i ∈ {1, 2, ..., n} ∧ 
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∃ (x : ℕ → ℕ), (∀ i, a i = x i + x ((i+1) % n)) ∧ 
    (∃ (r : ℝ), 0 < r ∧ 
      2 * ∑ i in finset.range n, real.arctan (x i / r) = 2 * real.pi))
  )) ↔ ¬ ∃ k : ℕ, n = 4 * k + 2
 :=
sorry

end exists_convex_ngon_with_side_lengths_tangent_l760_760669


namespace proof_problem_l760_760092

variable (W H I S : ℕ)
-- Condition 1: When whosis is is and so and so is is
def condition1 : Prop := H = I ∧ S = S

-- Condition 2: When whosis is so and so and so is so - so, again is is 2
def condition2 : Prop := H = S ∧ S - I = 2

theorem proof_problem (h1 : condition1) (h2 : condition2) : (S + S) = 2 * S :=
by
  sorry

end proof_problem_l760_760092


namespace cost_to_feed_turtles_l760_760354

theorem cost_to_feed_turtles :
  (∀ (weight : ℚ), weight > 0 → (food_per_half_pound_ounces = 1) →
    (total_weight_pounds = 30) →
    (jar_contents_ounces = 15 ∧ jar_cost = 2) →
    (total_cost_dollars = 8)) :=
by
  sorry

variables
  (food_per_half_pound_ounces : ℚ := 1)
  (total_weight_pounds : ℚ := 30)
  (jar_contents_ounces : ℚ := 15)
  (jar_cost : ℚ := 2)
  (total_cost_dollars : ℚ := 8)

-- Assuming all variables needed to state the theorem exist and are meaningful
/-!
  Given:
  - Each turtle needs 1 ounce of food per 1/2 pound of body weight
  - Total turtles' weight is 30 pounds
  - Each jar of food contains 15 ounces and costs $2

  Prove:
  - The total cost to feed Sylvie's turtles is $8
-/

end cost_to_feed_turtles_l760_760354


namespace geometric_sequence_a7_eq_64_l760_760316

open Nat

theorem geometric_sequence_a7_eq_64 (a : ℕ → ℕ) (h1 : a 1 = 1) (hrec : ∀ n : ℕ, a (n + 1) = 2 * a n) : a 7 = 64 := by
  sorry

end geometric_sequence_a7_eq_64_l760_760316


namespace peter_ate_total_l760_760333

-- Define the conditions
def total_slices : ℕ := 12
def peter_ate_alone : ℝ := 1 / 12
def peter_shared : ℝ := (1/2) * (1/12)

-- Define the theorem to prove
theorem peter_ate_total : peter_ate_alone + peter_shared = 1 / 8 := 
by sorry

end peter_ate_total_l760_760333


namespace lines_intersect_at_l760_760396

def Line1 (t : ℝ) : ℝ × ℝ :=
  let x := 1 + 3 * t
  let y := 2 - t
  (x, y)

def Line2 (u : ℝ) : ℝ × ℝ :=
  let x := -1 + 4 * u
  let y := 4 + 3 * u
  (x, y)

theorem lines_intersect_at :
  ∃ t u : ℝ, Line1 t = Line2 u ∧
             Line1 t = (-53 / 17, 56 / 17) :=
by
  sorry

end lines_intersect_at_l760_760396


namespace geometric_series_sum_example_l760_760155

def geometric_series_sum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r ^ n - 1) / (r - 1))

theorem geometric_series_sum_example : geometric_series_sum 2 (-3) 8 = -3280 :=
by
  sorry

end geometric_series_sum_example_l760_760155


namespace factorize_expr_solve_inequality_solve_equation_simplify_expr_l760_760610

-- Problem 1
theorem factorize_expr (x y m n : ℝ) : x^2 * (3 * m - 2 * n) + y^2 * (2 * n - 3 * m) = (3 * m - 2 * n) * (x + y) * (x - y) := 
sorry

-- Problem 2
theorem solve_inequality (x : ℝ) : 
  (∃ x, (x - 3) / 2 + 3 > x + 1 ∧ 1 - 3 * (x - 1) < 8 - x) → -2 < x ∧ x < 1 :=
sorry

-- Problem 3
theorem solve_equation (x : ℝ) : 
  (∃ x, (3 - x) / (x - 4) + 1 / (4 - x) = 1) → x = 3 :=
sorry

-- Problem 4
theorem simplify_expr (a : ℝ) (h : a = 3) : 
  (2 / (a + 1) + (a + 2) / (a^2 - 1)) / (a / (a - 1)) = 3 / 4 :=
sorry

end factorize_expr_solve_inequality_solve_equation_simplify_expr_l760_760610


namespace greatest_two_digit_multiple_of_17_l760_760407

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760407


namespace fraction_of_smart_integers_divisible_by_5_is_one_third_l760_760618

-- Condition definitions
def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_of_digits (n : ℕ) : ℕ := (Nat.digits 10 n).sum

def is_smart_integer (n : ℕ) : Prop :=
  is_even n ∧ 30 < n ∧ n < 150 ∧ sum_of_digits n = 10

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Function to filter smart integers
def smart_integers := List.filter is_smart_integer (List.range 150).filter (fun n => n > 30)

-- Count occurrences
def count_pred (l : List ℕ) (p : ℕ -> Prop) : ℕ :=
  l.filter p |>.length

noncomputable def fraction_smart_integers_divisible_by_5 : ℚ :=
  (count_pred smart_integers is_divisible_by_5 : ℚ) / (smart_integers.length : ℚ)

-- The theorem to prove
theorem fraction_of_smart_integers_divisible_by_5_is_one_third :
  fraction_smart_integers_divisible_by_5 = 1 / 3 := by
  sorry

end fraction_of_smart_integers_divisible_by_5_is_one_third_l760_760618


namespace greatest_two_digit_multiple_of_17_l760_760420

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760420


namespace solution_set_f_positive_l760_760213

variable (f : ℝ → ℝ)

-- Defining that f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f(x)

-- Given conditions
axiom f_odd : odd_function f
axiom f_at_1 : f 1 = 0
axiom derivative_condition : ∀ x : ℝ, 0 < x → (x * (f' x) - f x) / x^2 > 0

-- Define the statement to prove
theorem solution_set_f_positive : { x : ℝ | f x > 0 } = { x | (-1 < x ∧ x < 0) ∨ (1 < x) } :=
by
  sorry

end solution_set_f_positive_l760_760213


namespace normal_length_half_diameter_conjugate_l760_760546

variables {a b : ℝ} -- semi-major and semi-minor axes
variables {ϕ : ℝ} -- angular parameter

def ellipse_point (a b ϕ : ℝ) : ℝ × ℝ :=
  (a * Real.cos ϕ, b * Real.sin ϕ)

def length_square (p : ℝ × ℝ) : ℝ :=
  p.1^2 + p.2^2

def normal_length_square (a b ϕ : ℝ) : ℝ :=
  b^2 * Real.cos ϕ^2 + a^2 * Real.sin ϕ^2

theorem normal_length_half_diameter_conjugate (a b ϕ : ℝ) :
  (normal_length_square a b ϕ) = (a^2 + b^2) / 2 :=
by
  sorry

end normal_length_half_diameter_conjugate_l760_760546


namespace sum_of_primes_less_than_twenty_is_77_l760_760048

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760048


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l760_760346

theorem solve_eq1 (x : ℝ) : (3 - x)^2 + x^2 = 5 ↔ x = 1 ∨ x = 2 := 
sorry

theorem solve_eq2 (x : ℝ) : x^2 + 2 * (sqrt 3) * x + 3 = 0 ↔ x = -sqrt 3 := 
sorry

theorem solve_eq3 (x : ℝ) : 3 * x * (x - 1) = x * (x + 5) ↔ x = 0 ∨ x = 4 := 
sorry

theorem solve_eq4 (x : ℝ) : 2 * x^2 - sqrt 2 * x - 30 = 0 ↔ x = 3 * sqrt 2 ∨ x = -((5 * sqrt 2) / 2) := 
sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l760_760346


namespace actual_average_height_correct_l760_760538

noncomputable def average_height_corrected :=
  let num_boys := 40
  let avg_height_incorrect := 184
  let total_height_incorrect := num_boys * avg_height_incorrect
  let height_wrong1 := 166
  let height_actual1 := 106
  let height_wrong2 := 190
  let height_actual2 := 180
  let error1 := height_wrong1 - height_actual1
  let error2 := height_wrong2 - height_actual2
  let total_error := error1 + error2
  let total_height_correct := total_height_incorrect - total_error
  let avg_height_correct := total_height_correct / num_boys
  Float.round avg_height_correct 2

theorem actual_average_height_correct :
  average_height_corrected = 182.25 :=
by
  sorry

end actual_average_height_correct_l760_760538


namespace geometric_sequence_alternating_sum_l760_760781

noncomputable def geometric_sum (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - r ^ n) / (1 - r)

noncomputable def geometric_sum_squares (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁^2 * (1 - r^(2*n)) / (1 - r^2)

theorem geometric_sequence_alternating_sum (a₁ q : ℝ) (h₁ : q ≠ 1)
    (h₂ : geometric_sum a₁ q 5 = 15)
    (h₃ : geometric_sum_squares a₁ q 5 = 30) :
  ∑ i in finset.range 5, if i % 2 = 0 then a₁ * q^i else -a₁ * q^i = 2 :=
sorry

end geometric_sequence_alternating_sum_l760_760781


namespace hyperbola_eccentricity_l760_760270

theorem hyperbola_eccentricity (m : ℝ) 
  (hyperbola_eq : ∀ x y : ℝ, x ^ 2 / m - y ^ 2 / (m ^ 2 + 4) = 1)
  (eccentricity : ∀ a c : ℝ, c / a = sqrt 5) : m = 2 :=
by
  sorry

end hyperbola_eccentricity_l760_760270


namespace greatest_two_digit_multiple_of_17_l760_760441

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760441


namespace parabola_problem_l760_760204

noncomputable def parabola {p x y : ℝ} (hp : p > 0) : Prop :=
  y^2 = 2 * p * x

noncomputable def line {x y : ℝ} (m b : ℝ) : Prop :=
  y = m * (x - b)

noncomputable def distance {x1 y1 x2 y2 : ℝ} : ℝ :=
  real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem parabola_problem 
  (p : ℝ) (hp : p = 4)
  (x1 x2 : ℝ) (hx1 : x1 = 1) (hx2 : x2 = 4)
  (y1 y2 : ℝ) (hy1 : y1 = -2 * real.sqrt 2) (hy2 : y2 = 4 * real.sqrt 2)
  (h_dist : distance x1 y1 x2 y2 = 9)
  : parabola (show p = 4, by { rw hp, sorry }) 
    ∧ 
    (x1 = 1 ∧ y1 = -2 * real.sqrt 2) ∧ (x2 = 4 ∧ y2 = 4 * real.sqrt 2) := 
begin 
  sorry 
end

end parabola_problem_l760_760204


namespace sqrt_inequality_l760_760306

variable {a b c : ℝ}

theorem sqrt_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  sqrt (a * b * (a + b)) + sqrt (b * c * (b + c)) + sqrt (c * a * (c + a)) > 
  sqrt ((a + b) * (b + c) * (c + a)) :=
by
  sorry

end sqrt_inequality_l760_760306


namespace donuts_eaten_on_monday_l760_760141

theorem donuts_eaten_on_monday (D : ℕ) (h1 : D + D / 2 + 4 * D = 49) : 
  D = 9 :=
sorry

end donuts_eaten_on_monday_l760_760141


namespace correct_answer_is_A_l760_760094

-- Defining the events
def event1 : Prop := "Tossing a coin twice in a row and getting heads both times"
def event2 : Prop := "Opposite charges attract each other"
def event3 : Prop := "Water freezes at 1°C under standard atmospheric pressure"

-- Randomness properties
def is_random_event (e : Prop) : Prop := 
  match e with
  | "Tossing a coin twice in a row and getting heads both times" => true
  | "Opposite charges attract each other" => false
  | "Water freezes at 1°C under standard atmospheric pressure" => false
  | _ => false

theorem correct_answer_is_A : is_random_event event1 ∧ ¬ is_random_event event2 ∧ ¬ is_random_event event3 :=
by {
  split,
  all_goals { sorry }
}

end correct_answer_is_A_l760_760094


namespace greatest_two_digit_multiple_of_17_l760_760509

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760509


namespace composition_of_reflections_l760_760332

-- Definitions for the conditions
def perpendicular_lines (l1 l2 : ℝ^3 → ℝ^3) : Prop :=
  ∃ O : ℝ^3, (l1 O = O) ∧ (l2 O = O) ∧ (l1 l2 = λ P, P)

def reflection (l : ℝ^3 → ℝ^3) : ℝ^3 → ℝ^3 :=
  λ P, (λ (x, y, z), (x, -y, -z)) P

def composition_reflection (l1 l2 : ℝ^3 → ℝ^3) : ℝ^3 → ℝ^3 :=
  λ P, (reflection l2 (reflection l1 P))

def is_reflection (f : ℝ^3 → ℝ^3) (axis : ℝ^3 → ℝ^3) : Prop :=
  ∀ P : ℝ^3, f P = axis P

-- Theorem statement
theorem composition_of_reflections (l1 l2 : ℝ^3 → ℝ^3) (h1 : perpendicular_lines l1 l2) :
  ∃ l3 : ℝ^3 → ℝ^3, ⟪l1, l3⟫ = 0 ∧ ⟪l2, l3⟫ = 0
  ∧ is_reflection (composition_reflection l1 l2) l3 :=
sorry

end composition_of_reflections_l760_760332


namespace kabulek_numbers_four_digits_l760_760156

-- Definition of a Kabulek number
def is_kabulek (n : ℕ) : Prop :=
  let x := n / 100 in  -- first two digits
  let y := n % 100 in  -- last two digits
  (x + y)^2 = n

-- Set of all four-digit numbers
def four_digit_numbers := {n : ℕ | 1000 ≤ n ∧ n < 10000}

-- The statement proving that the four-digit Kabulek numbers are 2025, 3025, and 9801
theorem kabulek_numbers_four_digits :
  {n : ℕ | is_kabulek n ∧ n ∈ four_digit_numbers} = {2025, 3025, 9801} :=
by {
  sorry
}

end kabulek_numbers_four_digits_l760_760156


namespace find_f_2016_l760_760708

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x <= 2 then log x / log 4 else 0

theorem find_f_2016 : f 2016 = 1 / 2 := by
  sorry

end find_f_2016_l760_760708


namespace greatest_two_digit_multiple_of_17_l760_760494

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760494


namespace greatest_two_digit_multiple_of_17_l760_760498

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760498


namespace left_handed_fiction_readers_l760_760390

noncomputable section

def total_members (club : Type) [Fintype club] : ℕ := Fintype.card club
def fiction_readers (club : Type) : ℕ := 15  -- 15 members prefer reading fiction.
def left_handed (club : Type) : ℕ := 12  -- 12 members are left-handed.
def right_handed_non_fiction (club : Type) : ℕ := 3  -- 3 members are right-handed and do not prefer reading fiction.

-- Define the set type for our book club members
inductive ClubMember
| member : ClubMember

open ClubMember

theorem left_handed_fiction_readers : 
  total_members ClubMember = 25 →    -- Total number of members in the book club
  fiction_readers ClubMember = 15 →  -- Number of members who prefer reading fiction
  left_handed ClubMember = 12 →      -- Number of left-handed members
  right_handed_non_fiction ClubMember = 3 →  -- Number of right-handed non-fiction members
  ∃ x : ℕ, x = 5 :=           -- Prove that the number of left-handed fiction readers is 5
by
  sorry

end left_handed_fiction_readers_l760_760390


namespace brick_width_is_correct_l760_760111

-- Defining conditions
def wall_length : ℝ := 200 -- wall length in cm
def wall_width : ℝ := 300 -- wall width in cm
def wall_height : ℝ := 2   -- wall height in cm
def brick_length : ℝ := 25 -- brick length in cm
def brick_height : ℝ := 6  -- brick height in cm
def num_bricks : ℝ := 72.72727272727273

-- Total volume of wall
def vol_wall : ℝ := wall_length * wall_width * wall_height

-- Volume of one brick
def vol_brick (width : ℝ) : ℝ := brick_length * width * brick_height

-- Proof statement
theorem brick_width_is_correct : ∃ width : ℝ, vol_wall = vol_brick width * num_bricks ∧ width = 11 :=
by
  sorry

end brick_width_is_correct_l760_760111


namespace sum_of_primes_less_than_20_eq_77_l760_760023

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760023


namespace neg_three_is_monomial_l760_760095

def is_monomial (x : ℤ) : Prop := ∃ c : ℤ, ∃ n : ℕ, x = c * (b ^ n) ∧ n = 0

theorem neg_three_is_monomial : is_monomial (-3) :=
by
  sorry

end neg_three_is_monomial_l760_760095


namespace factorize_polynomial_l760_760642

theorem factorize_polynomial (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := 
by sorry

end factorize_polynomial_l760_760642


namespace sum_primes_less_than_20_l760_760076

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760076


namespace uncle_fyodor_wins_l760_760386

theorem uncle_fyodor_wins (N : ℕ) : ∀ sandwiches_with_sausage : fin (100 * N) → bool,
  (∃ sequence_of_moves : fin 101 → fin (100 * N), 
    (∀ i, (sequence_of_moves i) ∈ {0, 100 * N - 1}) ∧
    sandwiches_with_sausage (sequence_of_moves 100) = true) :=
by sorry

end uncle_fyodor_wins_l760_760386


namespace min_cardinality_union_intersection_l760_760848

theorem min_cardinality_union_intersection {X : Type} (A : fin 15 → set X)
  (hX : fintype.card X = 56) :
  (∃ (n : ℕ), 
    (∀ (S : finset (fin 15)), S.card = 7 → 
     fintype.card (⋃ i ∈ S, A i) ≥ n) → 
    (∃ T : finset (fin 15), T.card = 3 ∧ 
     (∩ i ∈ T, A i).nonempty)) ↔ n = 41 :=
sorry

end min_cardinality_union_intersection_l760_760848


namespace four_digit_integers_divisible_by_12_and_20_not_16_l760_760736

theorem four_digit_integers_divisible_by_12_and_20_not_16 : 
  (finset.filter (λ n => (60 ∣ n) ∧ (¬ 240 ∣ n)) (finset.range 10000 \ finset.range 1000)).card = 113 :=
by
  sorry

end four_digit_integers_divisible_by_12_and_20_not_16_l760_760736


namespace valid_cone_from_sector_l760_760529

-- Given conditions
def sector_angle : ℝ := 300
def circle_radius : ℝ := 15

-- Definition of correct option E
def base_radius_E : ℝ := 12
def slant_height_E : ℝ := 15

theorem valid_cone_from_sector :
  ( (sector_angle / 360) * (2 * Real.pi * circle_radius) = 25 * Real.pi ) ∧
  (slant_height_E = circle_radius) ∧
  (base_radius_E = 12) ∧
  (15^2 = 12^2 + 9^2) :=
by
  -- This theorem states that given sector angle and circle radius, the valid option is E
  sorry

end valid_cone_from_sector_l760_760529


namespace negation_equivalence_l760_760724

-- Define the original proposition p
def original_proposition (p : ℕ → Prop) : Prop :=
  ∀ x : ℝ, x > 0 → ln (x + 1) < x

-- Define the negation of the original proposition
def negation_of_proposition (p : ℕ → Prop) : Prop :=
  ∃ x : ℝ, x > 0 ∧ ln (x + 1) ≥ x

-- Proposition stating the equivalence
theorem negation_equivalence : 
  (original_proposition p) → (negation_of_proposition p) := 
by
  sorry

end negation_equivalence_l760_760724


namespace sum_digits_base8_to_base4_l760_760957

theorem sum_digits_base8_to_base4 :
  ∀ n : ℕ, (n ≥ 512 ∧ n ≤ 4095) →
  (∃ d : ℕ, (4^d > n ∧ n ≥ 4^(d-1))) →
  (d = 6) :=
by {
  sorry
}

end sum_digits_base8_to_base4_l760_760957


namespace inscribed_squares_ratio_l760_760976

theorem inscribed_squares_ratio (a b : ℝ) (h_triangle : 5^2 + 12^2 = 13^2)
    (h_square1 : a = 25 / 37) (h_square2 : b = 10) :
    a / b = 25 / 370 :=
by 
  sorry

end inscribed_squares_ratio_l760_760976


namespace sum_of_primes_less_than_20_eq_77_l760_760022

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760022


namespace sum_primes_less_than_20_l760_760074

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760074


namespace soccer_ball_price_l760_760280

theorem soccer_ball_price 
  (B S V : ℕ) 
  (h1 : (B + S + V) / 3 = 36)
  (h2 : B = V + 10)
  (h3 : S = V + 8) : 
  S = 38 := 
by 
  sorry

end soccer_ball_price_l760_760280


namespace greatest_two_digit_multiple_of_17_l760_760401

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760401


namespace distance_inequality_solution_l760_760830

theorem distance_inequality_solution (x : ℝ) (h : |x| > |x + 1|) : x < -1 / 2 :=
sorry

end distance_inequality_solution_l760_760830


namespace probability_of_prime_sum_is_three_fourths_l760_760913

def is_prime (n : ℕ) : Prop := nat.prime n

def S1 := {2, 4, 6}
def S2 := {1, 3, 5, 7}

def possible_sums (S1 S2 : set ℕ) : set ℕ :=
  {x + y | x ∈ S1, y ∈ S2}

def prime_sums (sums : set ℕ) : set ℕ :=
  {n ∈ sums | is_prime n}

def probability_prime_sum (S1 S2 : set ℕ) : ℚ :=
  (finset.card (prime_sums (possible_sums S1 S2)) : ℚ) / (finset.card (finset.product S1.to_finset S2.to_finset) : ℚ)

theorem probability_of_prime_sum_is_three_fourths :
  probability_prime_sum S1 S2 = 3 / 4 :=
by
  sorry

end probability_of_prime_sum_is_three_fourths_l760_760913


namespace g_property_l760_760300

theorem g_property (g : ℝ → ℝ)
  (h : ∀ x y z : ℝ, g (x^2 + y * g z) = x * g x + y * g z) :
  let m := 2 in -- Number of possible values of g(4) is 2 (g(4) can be 0 or 4)
  let t := 0 + 4 in -- Sum of all possible values of g(4) = 0 + 4 = 4
  m * t = 8 :=
by
  sorry

end g_property_l760_760300


namespace greatest_two_digit_multiple_of_17_l760_760411

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760411


namespace train_pass_bridge_in_36_seconds_l760_760534

def train_length : ℝ := 360 -- meters
def bridge_length : ℝ := 140 -- meters
def train_speed_kmh : ℝ := 50 -- km/h

noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600) -- m/s
noncomputable def total_distance : ℝ := train_length + bridge_length -- meters
noncomputable def passing_time : ℝ := total_distance / train_speed_ms -- seconds

theorem train_pass_bridge_in_36_seconds :
  passing_time = 36 := 
sorry

end train_pass_bridge_in_36_seconds_l760_760534


namespace find_d_l760_760904

-- Define vectors and conditions
def c : ℝ → ℝ × ℝ × ℝ := λ t, (t, 2*t, t)
def d (t : ℝ) : ℝ × ℝ × ℝ := (8 - t, -4 - 2*t, -8 - t)
def v : ℝ × ℝ × ℝ := (1, 2, 1)
def result_d : ℝ × ℝ × ℝ := (10, 0, -6)

-- dot product function
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Main theorem
theorem find_d : 
  ∃ (t : ℝ), (c t).1 + (d t).1 = 8 ∧ (c t).2 + (d t).2 = -4 ∧ (c t).3 + (d t).3 = -8 ∧ 
    dot_product (d t) v = 0 ∧ d t = result_d :=
by
  -- Placeholders for the final proof
  sorry

end find_d_l760_760904


namespace price_after_two_reductions_l760_760869

variable (orig_price : ℝ) (m : ℝ)

def current_price (orig_price : ℝ) (m : ℝ) : ℝ :=
  orig_price * (1 - m) * (1 - m)

theorem price_after_two_reductions (h1 : orig_price = 100) (h2 : 0 ≤ m ∧ m ≤ 1) :
  current_price orig_price m = 100 * (1 - m) ^ 2 := by
    sorry

end price_after_two_reductions_l760_760869


namespace max_angle_l760_760745

theorem max_angle {A B C : ℝ} (h₁ : 0 < A) (h₂ : A < π) (h₃ : 0 < B) (h₄ : B < π) (h₅ : 0 < C) (h₆ : C < π)
    (h_sum : A + B + C = π)
    (h_ratio : sin A / sin B = 3 / 5)
    (h_ratio' : sin B / sin C = 5 / 7) :
    max A (max B C) = 2 * π / 3 :=
  sorry

end max_angle_l760_760745


namespace find_weight_of_A_l760_760855

variables (a b c d e : ℕ)

-- Conditions
def condition1 := a + b + c = 210
def condition2 := a + b + c + d = 280
def condition3 := e = d + 3
def condition4 := b + c + d + e = 272

-- Theorem stating the problem
theorem find_weight_of_A (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : a = 81 := 
sorry

end find_weight_of_A_l760_760855


namespace apple_growth_rate_and_production_estimate_l760_760544

theorem apple_growth_rate_and_production_estimate (yield_2021 : ℝ) (yield_2023 : ℝ) : 
  yield_2021 = 1.5 →
  yield_2023 = 2.16 →
  ∃ (x : ℝ), (1.5 * (1 + x) ^ 2 = 2.16) ∧ x = 0.2 ∧ (2.16 * (1 + x) = 2.592) :=
by
  intros h1 h2
  use 0.2
  split
  · rw [← h1, ← h2]; linarith
  split
  · linarith
  · linarith

end apple_growth_rate_and_production_estimate_l760_760544


namespace voltmeter_readings_l760_760862

-- Definitions based on conditions
def U1 : ℝ := 4 -- Voltage read by V1
def U2 : ℝ := 6 -- Voltage read by V2
def U3 : ℝ := 2 -- Voltage read by V3
def U4 : ℝ := 4 / 3 -- Voltage read by V4

-- Goal to prove the sum of all readings is 15V
theorem voltmeter_readings : U3 + U4 + U1 + U2 = 15 :=
by
  sorry

end voltmeter_readings_l760_760862


namespace proof_problem_l760_760305

noncomputable def question (a b c : ℝ) : ℝ := 
  (a ^ 2 * b ^ 2) / ((a ^ 2 + b * c) * (b ^ 2 + a * c)) +
  (a ^ 2 * c ^ 2) / ((a ^ 2 + b * c) * (c ^ 2 + a * b)) +
  (b ^ 2 * c ^ 2) / ((b ^ 2 + a * c) * (c ^ 2 + a * b))

theorem proof_problem (a b c : ℝ) (h : a ≠ 0) (h1 : b ≠ 0) (h2 : c ≠ 0) 
  (h3 : a ^ 2 + b ^ 2 + c ^ 2 = a * b + b * c + c * a ) : 
  question a b c = 1 := 
by 
  sorry

end proof_problem_l760_760305


namespace greatest_two_digit_multiple_of_17_l760_760514

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760514


namespace Brenda_mice_left_l760_760999

theorem Brenda_mice_left :
  ∀ (total_litters total_each sixth factor remaining : ℕ),
    total_litters = 3 → 
    total_each = 8 →
    sixth = total_litters * total_each / 6 →
    factor = 3 * (total_litters * total_each / 6) →
    remaining = total_litters * total_each - sixth - factor →
    remaining / 2 = ((total_litters * total_each - sixth - factor) / 2) →
    total_litters * total_each - sixth - factor - ((total_litters * total_each - sixth - factor) / 2) = 4 :=
by
  intros total_litters total_each sixth factor remaining h_litters h_each h_sixth h_factor h_remaining h_half
  sorry

end Brenda_mice_left_l760_760999


namespace range_of_g_l760_760181

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : Set.Icc (-1.1071) 1.1071 = Set.image g (Set.Icc (-1:ℝ) 1) := by
  sorry

end range_of_g_l760_760181


namespace circle_center_construction_l760_760678

noncomputable def find_circle_center (circle : Set Point) (parallelogram : Set Point) : Point :=
  sorry

theorem circle_center_construction 
(circle center_exists : ∃ c : Point, is_center c circle) 
(parallelogram : Set Point) 
(vertices : ∀ v ∈ parallelogram, is_vertex v parallelogram) :
  ∃ c : Point, construction_with_straightedge c circle parallelogram :=
  sorry

end circle_center_construction_l760_760678


namespace exists_perpendicular_line_in_plane_l760_760188

-- Given a line l and a plane α
variables {point : Type} [MetricSpace point] (l : Set point) (α : Set point)

-- l is a line
def is_line (l : Set point) : Prop := 
  ∃ (p₁ p₂ : point), p₁ ≠ p₂ ∧ l = {x : point | ∃ (λ : ℝ), x = p₁ + λ • (p₂ - p₁)}

-- α is a plane
def is_plane (α : Set point) : Prop :=
  ∃ (p₁ p₂ p₃ : point), p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧ α = {x : point | ∃ (a b : ℝ), x = p₁ + a • (p₂ - p₁) + b • (p₃ - p₁)}

-- m is in the plane α
variables (m : Set point)
def is_in_plane (m α : Set point) : Prop := 
  ∃ (p₁ p₂ : point), p₁ ∈ α ∧ p₂ ∈ α ∧ m = {x : point | ∃ (λ : ℝ), x = p₁ + λ • (p₂ - p₁)}

-- lines l and m are perpendicular
def is_perpendicular (l m : Set point) : Prop :=
  ∃ (u v : point), u ∈ l ∧ v ∈ m ∧ inner u v = 0

theorem exists_perpendicular_line_in_plane (l : Set point) (α : Set point) (h₁ : is_line l) (h₂ : is_plane α) :
  ∃ (m : Set point), is_in_plane m α ∧ is_perpendicular l m :=
sorry

end exists_perpendicular_line_in_plane_l760_760188


namespace sum_prime_numbers_less_than_twenty_l760_760032

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760032


namespace remainder_p_x_minus_2_l760_760659

def p (x : ℝ) := x^5 + 2 * x^2 + 3

theorem remainder_p_x_minus_2 : p 2 = 43 := 
by
  sorry

end remainder_p_x_minus_2_l760_760659


namespace total_length_BC_DE_l760_760831

open_locale classical

variables (A B C D E : Type) [AddGroup₀ A] [MulGroup A]

-- Conditions based on the problem statement
variables
  (AB BD AC CD AD DE BC : A)
  (h_AB_BD : AB = 3 * BD)
  (h_AC_CD : AC = 7 * CD)
  (h_AD_AB_BD : AD = AB + BD)
  (h_AD_AC_CD : AD = AC + CD)
  (h_DE_mid_AD : DE = 1/2 * AD)
  (h_BC_eq : BC = AC - AB)

-- Theorem to prove
theorem total_length_BC_DE (A B C D E : Type) [AddGroup₀ A] [MulGroup A]
  (AB BD AC CD AD DE BC : A)
  (h_AB_BD : AB = 3 * BD)
  (h_AC_CD : AC = 7 * CD)
  (h_AD_AB_BD : AD = AB + BD)
  (h_AD_AC_CD : AD = AC + CD)
  (h_DE_mid_AD : DE = 1/2 * AD)
  (h_BC_eq : BC = AC - AB) :
  BC + DE = 5/8 * AD :=
by { sorry }

end total_length_BC_DE_l760_760831


namespace triangle_construction_and_angle_sum_l760_760289

theorem triangle_construction_and_angle_sum
  (A B C D E : Point)
  (h_isosceles : dist A B = dist A C)
  (h_points_on_side : D ∈ Segment B E ∧ E ∈ Segment D C)
  (h_angle_condition : 2 * angle D A E = angle B A C) :
  ∃ (X Y Z : Point), dist X Y = dist B D ∧ dist Y Z = dist D E ∧ dist Z X = dist E C ∧ angle B A C + angle Y X Z = 180 :=
begin
  sorry
end

end triangle_construction_and_angle_sum_l760_760289


namespace simplify_and_evaluate_l760_760842

theorem simplify_and_evaluate (x : Real) (h : x = Real.sqrt 2 - 1) :
  ( (1 / (x - 1) - 1 / (x + 1)) / (2 / (x - 1) ^ 2) ) = 1 - Real.sqrt 2 :=
by
  subst h
  sorry

end simplify_and_evaluate_l760_760842


namespace max_modulus_u_l760_760211

open Complex

theorem max_modulus_u (z : ℂ) (hz : abs z = 1) :
  let u := z^4 - z^3 - 3 * z^2 * Complex.i - z + 1 in
  abs u ≤ 5 :=
by
  let u := z^4 - z^3 - 3 * z^2 * Complex.i - z + 1
  have : |z| = 1 := hz
  sorry

end max_modulus_u_l760_760211


namespace secant_length_l760_760196

theorem secant_length
  (A B C D E : ℝ)
  (AB : A - B = 7)
  (BC : B - C = 7)
  (AD : A - D = 10)
  (pos : A > E ∧ D > E):
  E - D = 0.2 :=
by
  sorry

end secant_length_l760_760196


namespace cos_values_of_cot_plus_sec_eq_three_l760_760247

theorem cos_values_of_cot_plus_sec_eq_three (A : ℝ) (h₀ : ∀ A, tan A ≠ 0 ∧ cos A ≠ 0) (h : Real.cot A + Real.sec A = 3) :
  cos A = 2 / Real.sqrt 5 ∨ cos A = -2 / Real.sqrt 5 :=
by
  sorry

end cos_values_of_cot_plus_sec_eq_three_l760_760247


namespace turtle_feeding_cost_l760_760349

theorem turtle_feeding_cost (total_weight_pounds : ℝ) (food_per_half_pound : ℝ)
  (jar_food_ounces : ℝ) (jar_cost_dollars : ℝ) (total_cost : ℝ) : 
  total_weight_pounds = 30 →
  food_per_half_pound = 1 →
  jar_food_ounces = 15 →
  jar_cost_dollars = 2 →
  total_cost = 8 :=
by
  intros h_weight h_food h_jar_ounces h_jar_cost
  sorry

end turtle_feeding_cost_l760_760349


namespace minimum_value_inequality_l760_760248

open Function

theorem minimum_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 2) :
  (1 / a + 1 / (2 * b) + 4 * a * b) ≥ 4 :=
begin
  sorry, -- Proof steps will be filled here.
end

end minimum_value_inequality_l760_760248


namespace person_B_days_work_alone_l760_760827

-- Conditions
def person_A_work_rate := (1 : ℝ) / 30
def person_B_work_rate (B : ℝ) := 1 / B
def combined_work_rate (B : ℝ) := person_A_work_rate + person_B_work_rate B

-- Statement of the problem
theorem person_B_days_work_alone (B : ℝ) : 
  4 * combined_work_rate B = 2 / 9 → B = 45 :=
by
  sorry

end person_B_days_work_alone_l760_760827


namespace positiveDifference_l760_760179

noncomputable def positiveDifferenceOfSolutions : ℝ :=
  let x := 4 in
  2 * Real.sqrt 29

theorem positiveDifference (x : ℝ) (h : Real.cbrt (2 - x^2 / 4) = -3) : 
  positiveDifferenceOfSolutions = 4 * Real.sqrt 29 := by
  sorry

end positiveDifference_l760_760179


namespace increasing_function_on_positives_l760_760137

theorem increasing_function_on_positives :
  ∃ (f : ℝ → ℝ),
  (f = fun x => 3^x) ∧ 
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 < f x2) :=
begin
  use (fun x => 3^x),
  split,
  { refl, },
  { intros x1 x2 h0 hlt,
    simp [←lt_iff_le_and_ne] at *,
    sorry }
end

end increasing_function_on_positives_l760_760137


namespace find_a_of_tangent_asymptotes_l760_760226

noncomputable def circle_a_value (a : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - a)^2 + y^2 = 4 ∧
     (∀ (x₁ y₁ y₂ : ℝ), (x₁^2 - (y₁^2 / 4) = 1) ∧ 
      ((y₂ = 2 * x₁ + y₁) → ((2 * a) / (real.sqrt (2^2 + 1^2)) = 2)) ∧ (a > 0))

theorem find_a_of_tangent_asymptotes : circle_a_value (real.sqrt 5) :=
sorry

end find_a_of_tangent_asymptotes_l760_760226


namespace floor_ineq_l760_760949

theorem floor_ineq (α β : ℝ) : ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ :=
sorry

end floor_ineq_l760_760949


namespace greatest_two_digit_multiple_of_17_l760_760475

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760475


namespace factorize_polynomial_l760_760641

theorem factorize_polynomial (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := 
by sorry

end factorize_polynomial_l760_760641


namespace last_popsicle_melts_32_times_faster_l760_760110

theorem last_popsicle_melts_32_times_faster (t : ℕ) : 
  let time_first := t
  let time_sixth := t / 2^5
  (time_first / time_sixth) = 32 :=
by
  sorry

end last_popsicle_melts_32_times_faster_l760_760110


namespace find_divisors_l760_760647

noncomputable def is_solution (n : ℕ) (a : ℕ → ℕ) : Prop :=
  2 < a 1 ∧ (∀ i j, i ≤ n ∧ j ≤ n ∧ i ≤ j → a i ≤ a j) ∧
  (∀ i, 1 ≤ i ∧ i ≤ n → (15^25 + 1) % a i = 0) ∧
  2 - 2 / (15^25 + 1) = ∑ i in (finset.range n).image (λ i, 1 - 2 / a (i + 1))

theorem find_divisors :
  ∃ (a : ℕ → ℕ), is_solution 3 a ∧ a 1 = 4 ∧ a 2 = 4 ∧ a 3 = (15 ^ 25 + 1) :=
begin
  sorry
end

end find_divisors_l760_760647


namespace sum_of_odd_terms_geometric_sequence_l760_760684

theorem sum_of_odd_terms_geometric_sequence (n : ℕ) : 
  let a := λ n, 2 * 3^(n - 1)
  let new_sequence := λ k, a (2 * k - 1)
  nat.sum_range (λ k, new_sequence (k + 1) - 1) n = (9^n - 1) / 4 :=
  sorry

end sum_of_odd_terms_geometric_sequence_l760_760684


namespace greatest_two_digit_multiple_of_17_l760_760505

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760505


namespace greatest_two_digit_multiple_of_17_l760_760417

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760417


namespace greatest_two_digit_multiple_of_17_l760_760469

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760469


namespace sum_binom_mod_l760_760613

-- Define complex fifth roots of unity (assumed roots are given as omega, zeta, xi, eta)

noncomputable def omega := Complex.exp (2 * π * I / 5)
noncomputable def zeta := Complex.exp (4 * π * I / 5)
noncomputable def xi := Complex.exp (6 * π * I / 5)
noncomputable def eta := Complex.exp (8 * π * I / 5)

-- Define S as the sum using binomial coefficients and fifth roots of unity
noncomputable def S : ℂ := ∑ i in range (2022), (Complex.ofReal (binom 2021 i)) * (omega ^ i + zeta ^ i + xi ^ i + eta ^ i + 1)

theorem sum_binom_mod :
  (∑ i in range 405, (binom 2021 (5 * i)) * 5) % 500 = 29 :=
by
  sorry

end sum_binom_mod_l760_760613


namespace seats_taken_l760_760665

variable (num_rows : ℕ) (chairs_per_row : ℕ) (unoccupied_chairs : ℕ)

theorem seats_taken (h1 : num_rows = 40) (h2 : chairs_per_row = 20) (h3 : unoccupied_chairs = 10) :
  num_rows * chairs_per_row - unoccupied_chairs = 790 :=
sorry

end seats_taken_l760_760665


namespace rectangle_area_l760_760127

theorem rectangle_area (x : ℝ) (w : ℝ) (h_diag : (3 * w) ^ 2 + w ^ 2 = x ^ 2) : 
  3 * w ^ 2 = (3 / 10) * x ^ 2 :=
by
  sorry

end rectangle_area_l760_760127


namespace multiplication_division_l760_760535

theorem multiplication_division:
  (213 * 16 = 3408) → (1.6 * 2.13 = 3.408) :=
by
  sorry

end multiplication_division_l760_760535


namespace families_received_boxes_l760_760670

theorem families_received_boxes (F : ℕ) (box_decorations total_decorations : ℕ)
  (h_box_decorations : box_decorations = 10)
  (h_total_decorations : total_decorations = 120)
  (h_eq : box_decorations * (F + 1) = total_decorations) :
  F = 11 :=
by
  sorry

end families_received_boxes_l760_760670


namespace grown_ups_in_milburg_l760_760901

def number_of_children : ℕ := 2987
def total_population : ℕ := 8243

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 := 
by 
  sorry

end grown_ups_in_milburg_l760_760901


namespace P_is_reducible_l760_760821

noncomputable def P (x : ℤ) : ℤ := x ^ 1981 + x ^ 1980 + 12 * x ^ 2 + 24 * x + 1983

theorem P_is_reducible : ∃ (f g : polynomial ℤ), polynomial.is_reducible P := sorry

end P_is_reducible_l760_760821


namespace inequality_one_over_a_plus_one_over_b_geq_4_l760_760702

theorem inequality_one_over_a_plus_one_over_b_geq_4 
    (a b : ℕ) (hapos : 0 < a) (hbpos : 0 < b) (h : a + b = 1) : 
    (1 : ℚ) / a + (1 : ℚ) / b ≥ 4 := 
  sorry

end inequality_one_over_a_plus_one_over_b_geq_4_l760_760702


namespace greatest_two_digit_multiple_of_17_l760_760510

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760510


namespace problem_solution_l760_760298

noncomputable def f (x : ℝ) : ℝ := sin x + 2 * x * (deriv (λ t, f t) x)
def a : ℝ := -1
def b : ℝ := real.log 2 / real.log 3

theorem problem_solution : f a > f b := sorry

end problem_solution_l760_760298


namespace range_of_s_l760_760164

noncomputable def f : ℝ → ℝ :=
sorry

def is_decreasing (f : ℝ → ℝ) : Prop :=
∀ (x y : ℝ), x < y → f x > f y

def centrally_symmetric (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
∀ (x : ℝ), f (c.1 + (c.1 - x)) = 2 * c.2 - f x

theorem range_of_s
  (f : ℝ → ℝ)
  (decreasing_f : is_decreasing f)
  (symmetric_f : centrally_symmetric (λ x, f (x - 1)) (1, 0))
  (s : ℝ) :
  f (s^2 - 2 * s) + f (2 - s) ≤ 0 →
  s ∈ set.Iic 1 ∪ set.Ici 2 :=
sorry

end range_of_s_l760_760164


namespace solve_for_m_l760_760168

-- Definition of the equation with a parameter m
def equation (x m : ℝ) : Prop := (3 * x + 4) * (x - 8) = -95 + m * x

-- The condition for the equation to have exactly one real solution
def discriminant_zero (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

-- Main statement
theorem solve_for_m : ∀ (m : ℝ),
  (∀ (x : ℝ), equation x m) ↔
  discriminant_zero 3 (-(20 + m)) 63 →
  m = -20 + 2 * Real.sqrt 189 ∨ m = -20 - 2 * Real.sqrt 189 :=
begin
  -- Translate the conditions in the problem directly as definitions in Lean
  assume m,
  split,
  { intro heq,
    have hquad : ∀ x, 3 * x^2 - (20 + m) * x + 63 = 0,
    { intro x,
      simp [equation, heq] },
    have hdisc : (20 + m)^2 - 4 * 3 * 63 = 0,
    from discriminant_zero 3 (-(20 + m)) 63,
    sorry
  },
  { intro hdisc,
    have hsolutions : m = -20 + 2 * Real.sqrt 189 ∨ m = -20 - 2 * Real.sqrt 189,
    sorry
  }
end

end solve_for_m_l760_760168


namespace greatest_two_digit_multiple_of_17_l760_760516

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760516


namespace two_hundred_twenty_second_digit_of_fraction_l760_760398

theorem two_hundred_twenty_second_digit_of_fraction :
  (222nd_digit_after_decimal_point (47 / 777) = 5) := 
by
  sorry

end two_hundred_twenty_second_digit_of_fraction_l760_760398


namespace knight_liar_grouping_l760_760102

noncomputable def can_be_partitioned_into_knight_liar_groups (n m : ℕ) (h1 : n ≥ 2) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k < n) : Prop :=
  ∃ t : ℕ, n = (m + 1) * t

-- Show that if the company has n people, where n ≥ 2, and there exists at least one knight,
-- then n can be partitioned into groups where each group contains 1 knight and m liars.
theorem knight_liar_grouping (n m : ℕ) (h1 : n ≥ 2) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k < n) : can_be_partitioned_into_knight_liar_groups n m h1 h2 :=
sorry

end knight_liar_grouping_l760_760102


namespace no_possible_numbering_for_equal_sidesum_l760_760334

theorem no_possible_numbering_for_equal_sidesum (O : Point) (A : Fin 10 → Point) 
  (side_numbers : (Fin 10) → ℕ) (segment_numbers : (Fin 10) → ℕ) : 
  ¬ ∃ (side_segment_sum_equal : Fin 10 → ℕ) (sum_equal : ℕ),
    (∀ i, side_segment_sum_equal i = side_numbers i + segment_numbers i) ∧ 
    (∀ i, side_segment_sum_equal i = sum_equal) := 
sorry

end no_possible_numbering_for_equal_sidesum_l760_760334


namespace coprime_sum_product_infinite_set_l760_760648

def coprime (a b : ℕ) : Prop := Int.gcd a b = 1

theorem coprime_sum_product_infinite_set (n : ℕ) (A : Set ℕ) :
  (∀ a ∈ A, 0 < a) →    -- A is a set of positive integers
  Set.Infinite A →      -- A is an infinite set
  (∀ a1 a2 ... an ∈ A, 
     a1 ≠ a2 → a1 ≠ ... → a2 ≠ ...  → 
     a1 + a2 + ... + an = a1 + a2 + ... + an ∧ 
     coprime (a1 + a2 + ... + an) (a1 * a2 * ... * an)) → -- coprime property
  n ≠ 1 := 
by
  -- Proof can be added here.
  sorry

end coprime_sum_product_infinite_set_l760_760648


namespace average_attendees_per_day_is_l760_760547

theorem average_attendees_per_day_is {Monday Tuesday Wednesday Thursday Friday Saturday Sunday : ℕ} 
  (hM : Monday = 10)
  (hT : Tuesday = 15)
  (hW : Wednesday = 13)
  (hTh : Thursday = 10)
  (hF : Friday = 10)
  (hSa : Saturday = 8)
  (hSu : Sunday = 12) :
  (Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday) / 7 = 11.14 := by
  sorry

end average_attendees_per_day_is_l760_760547


namespace greatest_two_digit_multiple_of_17_l760_760425

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760425


namespace marthas_bedroom_size_l760_760887

-- Define the variables and conditions
def total_square_footage := 300
def additional_square_footage := 60
def Martha := 120
def Jenny := Martha + additional_square_footage

-- The main theorem stating the requirement 
theorem marthas_bedroom_size : (Martha + (Martha + additional_square_footage) = total_square_footage) -> Martha = 120 :=
by 
  sorry

end marthas_bedroom_size_l760_760887


namespace greatest_two_digit_multiple_of_17_l760_760406

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760406


namespace line_intersects_circle_l760_760719

theorem line_intersects_circle :
  let l := λ x, x + 1 in
  let C := λ x y, x^2 + y^2 = 1 in
  ∃ x y, y = l x ∧ C x y :=
by { sorry }

end line_intersects_circle_l760_760719


namespace total_number_of_arrangements_l760_760762

-- Definitions based on the conditions stated in part a)
def procedures : list string := ["A", "B", "C", "D", "E", "F"]

def is_first_or_last (seq : list string) : Prop :=
  seq.head? = some "A" ∨ seq.getLast? = some "A"

def are_consecutive (seq : list string) : Prop :=
  list.inits seq |> list.any (λ xs, "B"::"C"::xs ∈ list.tails seq) 
  ∨ list.inits seq |> list.any (λ xs, "C"::"B"::xs ∈ list.tails seq)

-- Lean statement proving the total number of arrangements
theorem total_number_of_arrangements : 
  ∃ seqs : list (list string),
    (∀ seq ∈ seqs, is_first_or_last seq ∧ are_consecutive seq ∧ 
    (list.perm seq ["A", "B", "C", "D", "E", "F"])) ∧
    seqs.length = 96 :=
by
  sorry

end total_number_of_arrangements_l760_760762


namespace correct_calculation_is_A_l760_760527

theorem correct_calculation_is_A : (1 + (-2)) = -1 :=
by 
  sorry

end correct_calculation_is_A_l760_760527


namespace binom_n_n_l760_760611

theorem binom_n_n (n : ℤ) (h : n ≥ 0) : Nat.choose n.to_nat n.to_nat = 1 := by
  sorry

end binom_n_n_l760_760611


namespace uv_square_l760_760249

theorem uv_square (u v : ℝ) (h1 : u * (u + v) = 50) (h2 : v * (u + v) = 100) : (u + v)^2 = 150 := by
  sorry

end uv_square_l760_760249


namespace determine_omega_phi_l760_760314

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem determine_omega_phi (ω φ : ℝ) (x : ℝ)
  (h₁ : 0 < ω) (h₂ : |φ| < Real.pi)
  (h₃ : f ω φ (5 * Real.pi / 8) = 2)
  (h₄ : f ω φ (11 * Real.pi / 8) = 0)
  (h₅ : (2 * Real.pi / ω) > 2 * Real.pi) :
  ω = 2 / 3 ∧ φ = Real.pi / 12 :=
sorry

end determine_omega_phi_l760_760314


namespace sum_of_primes_less_than_20_l760_760006

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760006


namespace cans_to_feed_adults_l760_760135

theorem cans_to_feed_adults (initial_cans : ℕ) (children_fed : ℕ) (children_per_can : ℕ) (adults_per_can : ℕ) :
  let cans_for_children := children_fed / children_per_can in
  let remaining_cans := initial_cans - cans_for_children in
  let adults_fed := remaining_cans * adults_per_can in
  initial_cans = 10 ->
  children_fed = 40 ->
  children_per_can = 8 ->
  adults_per_can = 4 ->
  adults_fed = 20 :=
by {
  intros,
  have h1 : cans_for_children = 5, { rw [children_fed, children_per_can], exact 40 / 8 },
  have h2 : remaining_cans = 5, { rw [initial_cans, h1], exact 10 - 5 },
  have h3 : adults_fed = 20, { rw [remaining_cans, adults_per_can], exact 5 * 4 },
  exact h3
}

end cans_to_feed_adults_l760_760135


namespace sum_primes_less_than_20_l760_760088

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760088


namespace greatest_two_digit_multiple_of_17_is_85_l760_760465

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760465


namespace aqua_park_earnings_l760_760139

def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def meal_fee : ℕ := 10
def souvenir_fee : ℕ := 8

def group1_admission_count : ℕ := 10
def group1_tour_count : ℕ := 10
def group1_meal_count : ℕ := 10
def group1_souvenir_count : ℕ := 10
def group1_discount : ℚ := 0.10

def group2_admission_count : ℕ := 15
def group2_meal_count : ℕ := 15
def group2_meal_discount : ℚ := 0.05

def group3_admission_count : ℕ := 8
def group3_tour_count : ℕ := 8
def group3_souvenir_count : ℕ := 8

-- total cost for group 1 before discount
def group1_total_before_discount : ℕ := 
  (group1_admission_count * admission_fee) +
  (group1_tour_count * tour_fee) +
  (group1_meal_count * meal_fee) +
  (group1_souvenir_count * souvenir_fee)

-- group 1 total cost after discount
def group1_total_after_discount : ℚ :=
  group1_total_before_discount * (1 - group1_discount)

-- total cost for group 2 before discount
def group2_admission_total_before_discount : ℕ := 
  group2_admission_count * admission_fee
def group2_meal_total_before_discount : ℕ := 
  group2_meal_count * meal_fee

-- group 2 total cost after discount
def group2_meal_total_after_discount : ℚ :=
  group2_meal_total_before_discount * (1 - group2_meal_discount)
def group2_total_after_discount : ℚ :=
  group2_admission_total_before_discount + group2_meal_total_after_discount

-- total cost for group 3 before discount
def group3_total_before_discount : ℕ := 
  (group3_admission_count * admission_fee) +
  (group3_tour_count * tour_fee) +
  (group3_souvenir_count * souvenir_fee)

-- group 3 total cost after discount (no discount applied)
def group3_total_after_discount : ℕ := group3_total_before_discount

-- total earnings from all groups
def total_earnings : ℚ :=
  group1_total_after_discount +
  group2_total_after_discount +
  group3_total_after_discount

theorem aqua_park_earnings : total_earnings = 854.50 := by
  sorry

end aqua_park_earnings_l760_760139


namespace greatest_two_digit_multiple_of_17_l760_760485

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760485


namespace most_balloons_are_blue_l760_760903

theorem most_balloons_are_blue
  (total_balloons : ℕ)
  (blue_minus_red : ℕ)
  (red_ratio : ℚ)
  (yellow_balloons : ℕ)
  (h1 : total_balloons = 24)
  (h2 : blue_minus_red = 6)
  (h3 : red_ratio = 1/4)
  (h4 : yellow_balloons = total_balloons - (red_ratio * total_balloons).nat_abs - (blue_minus_red + (red_ratio * total_balloons).nat_abs))
  : ∃ blue_balloons : ℕ, blue_balloons = (red_ratio * total_balloons).nat_abs + blue_minus_red ∧ blue_balloons > (red_ratio * total_balloons).nat_abs ∧ blue_balloons > yellow_balloons :=
by
  sorry

end most_balloons_are_blue_l760_760903


namespace common_non_integer_root_l760_760337

theorem common_non_integer_root (p1 p2 q1 q2 : ℤ)
  (h1 : ∃ r : ℝ, r ∉ ℤ ∧ (r^2 + (p1:ℝ) * r + (q1:ℝ) = 0) ∧ (r^2 + (p2:ℝ) * r + (q2:ℝ) = 0)) :
  p1 = p2 ∧ q1 = q2 := sorry

end common_non_integer_root_l760_760337


namespace expected_value_transformation_l760_760315

variable (ξ : ℝ)
variable (E : (ℝ → ℝ) → ℝ)
variable (D : (ℝ → ℝ) → ℝ)

axiom E_ξ : E (λ x, ξ) = -1
axiom D_ξ : D (λ x, ξ) = 3
axiom var_def : D (λ x, ξ) = E (λ x, ξ^2) - (E (λ x, ξ))^2

theorem expected_value_transformation : E (λ x, 3 * (ξ^2 - 2)) = 6 := by
  sorry

end expected_value_transformation_l760_760315


namespace sum_of_arithmetic_terms_l760_760879

theorem sum_of_arithmetic_terms (a₁ a₂ a₃ c d a₆ : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₂ = 10)
  (h₃ : a₃ = 17)
  (h₄ : a₆ = 32)
  (h_arith : ∀ n, (a₁ + n * (a₂ - a₁)) = seq)
  : c + d = 55 :=
by
  have d := a₂ - a₁
  have c := a₃ + d
  have d := c + d
  have h_seq := list.map (λ n, (a₁ + n * d)) (list.range 6) -- Making use of the arithmetic property
  have h_seq_eq := h_seq = [3, 10, 17, c, d, 32]
  sorry

end sum_of_arithmetic_terms_l760_760879


namespace cricket_match_count_l760_760955

theorem cricket_match_count (x : ℕ) (h_avg_1 : ℕ → ℕ) (h_avg_2 : ℕ) (h_avg_all : ℕ) (h_eq : 50 * x + 26 * 15 = 42 * (x + 15)) : x = 30 :=
by
  sorry

end cricket_match_count_l760_760955


namespace sum_of_products_non_positive_l760_760673

theorem sum_of_products_non_positive (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end sum_of_products_non_positive_l760_760673


namespace sum_of_primes_less_than_twenty_is_77_l760_760047

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760047


namespace sum_of_primes_less_than_20_l760_760004

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760004


namespace greatest_two_digit_multiple_of_17_l760_760413

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760413


namespace num_ways_enter_exit_l760_760956

-- Conditions
def num_entrances_exits : ℕ := 4

-- Statement
theorem num_ways_enter_exit (n : ℕ) (h : n = 4) : (4 * (4 - 1)) = 12 :=
by 
  rw h
  sorry

end num_ways_enter_exit_l760_760956


namespace infinite_series_sum_l760_760635

-- Define the infinite series sum.
def infinite_series : ℕ → ℝ :=
  λ k => k / 3^k

-- State the theorem.
theorem infinite_series_sum :
  (∑' k, infinite_series (k + 1)) = 1 / 4 :=
begin
  sorry
end

end infinite_series_sum_l760_760635


namespace greatest_two_digit_multiple_of_17_l760_760416

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760416


namespace num_bijections_l760_760795

theorem num_bijections (n : ℕ) (h : n % 4 = 0) (f : {p : ℕ // p ∈ finset.range n.succ} → {p : ℕ // p ∈ finset.range n.succ}) 
  (hf : bijective f) (hfn : ∀ j : {p : ℕ // p ∈ finset.range n.succ}, f j + (f⁻¹ j).val = n + 1) : 
  ∃ k : ℕ, k = (nat.factorial (n / 2)) / (nat.factorial (n / 4)) :=
sorry

end num_bijections_l760_760795


namespace fraction_covered_by_pepperoni_l760_760125

-- Define the conditions
def pizza_diameter : ℝ := 16
def number_of_pepperoni_across_diameter : ℕ := 8
def number_of_pepperoni_total : ℕ := 32
def pepperoni_radius : ℝ := (pizza_diameter / number_of_pepperoni_across_diameter) / 2

-- Calculate the area of the pizza and pepperoni
def area_of_pizza : ℝ := π * (pizza_diameter / 2) ^ 2
def area_of_one_pepperoni : ℝ := π * pepperoni_radius ^ 2
def total_area_of_pepperoni : ℝ := number_of_pepperoni_total * area_of_one_pepperoni

-- Prove the fraction of the pizza covered by pepperoni
theorem fraction_covered_by_pepperoni : total_area_of_pepperoni / area_of_pizza = 1 / 2 :=
by
  sorry

end fraction_covered_by_pepperoni_l760_760125


namespace total_fruit_count_l760_760857

-- Define the conditions as variables and equations
def apples := 4 -- based on the final deduction from the solution
def pears := 6 -- calculated from the condition of bananas
def bananas := 9 -- given in the problem

-- State the conditions
axiom h1 : pears = apples + 2
axiom h2 : bananas = pears + 3
axiom h3 : bananas = 9

-- State the proof objective
theorem total_fruit_count : apples + pears + bananas = 19 :=
by
  sorry

end total_fruit_count_l760_760857


namespace cube_volume_l760_760747

theorem cube_volume (A : ℝ) (V : ℝ) (h : A = 64) : V = 512 :=
by
  sorry

end cube_volume_l760_760747


namespace probability_three_digit_number_perfect_square_l760_760970

theorem probability_three_digit_number_perfect_square :
  (finset.card (finset.filter (λ n : ℕ, ∃ k : ℕ, k^2 = n) (finset.Icc 100 999))) / (finset.card (finset.Icc 100 999)) = 11 / 450 :=
sorry

end probability_three_digit_number_perfect_square_l760_760970


namespace unique_elements_condition_l760_760689

theorem unique_elements_condition (x : ℝ) : 
  (1 ≠ x ∧ x ≠ x^2 ∧ 1 ≠ x^2) ↔ (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :=
by 
  sorry

end unique_elements_condition_l760_760689


namespace cousin_brought_correct_l760_760789

noncomputable def jim_brought : ℝ := 100.0
noncomputable def cost_surf_turf : ℝ := 38.0
noncomputable def cost_wine : ℝ := 14.0
noncomputable def cost_dessert : ℝ := 18.0
noncomputable def discount_rate : ℝ := 0.20
noncomputable def tax_rate : ℝ := 0.10
noncomputable def rounded_total : ℝ := 120.0
noncomputable def spent_percentage : ℝ := 0.85

noncomputable def expected_cousin_brought : ℝ := 35.0 / 0.85

theorem cousin_brought_correct (C : ℝ) :
  jim_brought + C > 0 ∧
  let total_food_cost := 2 * cost_surf_turf + cost_dessert in
  let discounted_food_cost := total_food_cost * (1 - discount_rate) in
  let total_drink_cost := 2 * cost_wine in
  let subtotal := discounted_food_cost + total_drink_cost in
  let total_before_tip := subtotal * (1 + tax_rate) in
  let rounded_total := (⌊total_before_tip / 10⌋ + 1) * 10 in
  let total_combined_money := jim_brought + C in
  (spent_percentage * total_combined_money = rounded_total) →
  C = expected_cousin_brought := by
  sorry

end cousin_brought_correct_l760_760789


namespace find_q_l760_760574

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def square_pyramid_edge_length : ℝ := 6

def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨6, 0, 0⟩
def C : Point3D := ⟨6, 6, 0⟩
def D : Point3D := ⟨0, 6, 0⟩
def E : Point3D := ⟨3, 3, 3 * Real.sqrt 3⟩

def midpoint (P Q : Point3D) : Point3D :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2, (P.z + Q.z) / 2⟩

def M : Point3D := midpoint A E
def N : Point3D := midpoint B C
def P : Point3D := midpoint C D

noncomputable def plane_through_midpoints (M N P : Point3D) : Point3D → Prop :=
  λ Q, Q.x + Q.y + 2 * Real.sqrt 3 * Q.z = 9

noncomputable def area_of_intersection_polygon (M N P : Point3D) : ℝ :=
  18 * Real.sqrt 3 -- Derived in solution

theorem find_q :
  ∃ (q : ℝ), q = 972 ∧ area_of_intersection_polygon M N P = Real.sqrt q := by
  use 972
  sorry

end find_q_l760_760574


namespace sum_of_primes_less_than_twenty_is_77_l760_760049

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760049


namespace greatest_two_digit_multiple_of_17_l760_760493

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760493


namespace sum_of_primes_less_than_20_l760_760001

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760001


namespace evaluate_complex_modulus_l760_760634

namespace ComplexProblem

open Complex

theorem evaluate_complex_modulus : 
  abs ((1 / 2 : ℂ) - (3 / 8) * Complex.I) = 5 / 8 :=
by
  sorry

end ComplexProblem

end evaluate_complex_modulus_l760_760634


namespace total_assignments_to_earn_30_points_l760_760576

theorem total_assignments_to_earn_30_points : 
  let assignments_per_point (n : ℕ) := if n < 4 then 2 else 2 + (n / 4)
  let total_assignments := 
    (List.sum (List.range (30 - 2) |
    map (λ n, assignments_per_point n)) + 
    2 * assignments_per_point 29
  total_assignments = 156 :=
  sorry

end total_assignments_to_earn_30_points_l760_760576


namespace largest_of_four_integers_l760_760195

theorem largest_of_four_integers (n : ℤ) (h1 : n % 2 = 0) (h2 : (n+2) % 2 = 0) (h3 : (n+4) % 2 = 0) (h4 : (n+6) % 2 = 0) (h : n * (n+2) * (n+4) * (n+6) = 6720) : max (max (max n (n+2)) (n+4)) (n+6) = 14 := 
sorry

end largest_of_four_integers_l760_760195


namespace prob_factor_90_lt_8_l760_760524

theorem prob_factor_90_lt_8 : 
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90] →
  let factors_lt_8 := [1, 2, 3, 5, 6] →
  length factors_lt_8 / length factors = 5 / 12 :=
by sorry

end prob_factor_90_lt_8_l760_760524


namespace centroid_integer_coordinates_l760_760672

def Point : Type := ℤ × ℤ

theorem centroid_integer_coordinates (points : List Point) (h_length : points.length = 19) (h_no_collinear : ∀ (p1 p2 p3 : Point), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → ¬ collinear p1 p2 p3) :
  ∃ (p1 p2 p3 : Point), (p1 ∈ points) ∧ (p2 ∈ points) ∧ (p3 ∈ points) ∧ 
  (↑((p1.1 + p2.1 + p3.1) / 3) - ((p1.1 + p2.1 + p3.1) / 3) = 0) ∧ 
  (↑((p1.2 + p2.2 + p3.2) / 3) - ((p1.2 + p2.2 + p3.2) / 3) = 0) :=
by
  sorry

end centroid_integer_coordinates_l760_760672


namespace total_inflation_time_l760_760570

theorem total_inflation_time (time_per_ball : ℕ) (alexia_balls : ℕ) (extra_balls : ℕ) : 
  time_per_ball = 20 → alexia_balls = 20 → extra_balls = 5 →
  (alexia_balls * time_per_ball) + ((alexia_balls + extra_balls) * time_per_ball) = 900 :=
by 
  intros h1 h2 h3
  sorry

end total_inflation_time_l760_760570


namespace distinct_numbers_in_S_l760_760290

def seq1 : Fin 1000 → ℕ := λ k, 5 * k + 1
def seq2 : Fin 1000 → ℕ := λ l, 8 * l + 10
def seq3 : Fin 1000 → ℕ := λ m, 10 * m + 5

def A : Finset ℕ := Finset.image seq1 (Finset.univ : Finset (Fin 1000))
def B : Finset ℕ := Finset.image seq2 (Finset.univ : Finset (Fin 1000))
def C : Finset ℕ := Finset.image seq3 (Finset.univ : Finset (Fin 1000))

def S : Finset ℕ := A ∪ B ∪ C

theorem distinct_numbers_in_S : S.card = 2325 :=
sorry

end distinct_numbers_in_S_l760_760290


namespace turtle_feeding_cost_l760_760351

def cost_to_feed_turtles (turtle_weight: ℝ) (food_per_half_pound: ℝ) (jar_capacity: ℝ) (jar_cost: ℝ) : ℝ :=
  let total_food := turtle_weight * (food_per_half_pound / 0.5)
  let total_jars := total_food / jar_capacity
  total_jars * jar_cost

theorem turtle_feeding_cost :
  cost_to_feed_turtles 30 1 15 2 = 8 :=
by
  sorry

end turtle_feeding_cost_l760_760351


namespace farmer_potatoes_initial_l760_760557

theorem farmer_potatoes_initial (P : ℕ) (h1 : 175 + P - 172 = 80) : P = 77 :=
by {
  sorry
}

end farmer_potatoes_initial_l760_760557


namespace find_b_minus_a_l760_760677

theorem find_b_minus_a (a b : ℝ) (h : ∀ x : ℝ, 0 ≤ x → 
  0 ≤ x^4 - x^3 + a * x + b ∧ x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2) : 
  b - a = 2 :=
sorry

end find_b_minus_a_l760_760677


namespace number_of_valid_arrangements_is_144_l760_760394

-- Definitions for the problem
def children := {a1, a2, b1, b2, c1, c2}

def is_sibling : children → children → Prop
| a1, a2 := true
| a2, a1 := true
| b1, b2 := true
| b2, b1 := true
| _, _ := false

def is_cousin : children → children → Prop
| c1, c2 := true
| c2, c1 := true
| _, _ := false

def not_adjacent_in_row (x y : children) : Prop :=
  ¬is_sibling x y ∧ ¬is_cousin x y

def seating_arrangement := list (list children)

-- Conditions for seating
def valid_arrangement (arr : seating_arrangement) : Prop :=
  arr.length = 2 ∧
  (∀ row, row.length = 3) ∧
  (∀ row, not_adjacent_in_row row[0] row[1]) ∧
  (∀ row, not_adjacent_in_row row[1] row[2]) ∧
  (¬is_sibling arr[0][1] arr[1][1] ∧ ¬is_cousin arr[0][1] arr[1][1]) ∧
  (∃ (x1 x2 x3 : children), (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
                            [x1, x2, x3] = arr[0]) ∧
                            (¬is_sibling x1 x2 ∧ ¬is_sibling x2 x3 ∧ ¬is_sibling x1 x3) ∧
                            (¬is_cousin x1 x2 ∧ ¬is_cousin x2 x3 ∧ ¬is_cousin x1 x3))

-- Main statement to prove
theorem number_of_valid_arrangements_is_144 :
  ∃ (arrangements : finset seating_arrangement), arrangements.card = 144 ∧
  (∀ arr ∈ arrangements, valid_arrangement arr) :=
sorry

end number_of_valid_arrangements_is_144_l760_760394


namespace greatest_two_digit_multiple_of_17_l760_760480

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760480


namespace greatest_two_digit_multiple_of_17_l760_760484

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760484


namespace digit_2_appears_20_times_l760_760551

theorem digit_2_appears_20_times : 
  let num_pages := 100
  let page_numbers := list.range (num_pages + 1) -- 1 to 100 inclusive.
  (page_numbers.filter (λ n, n.digits 10).count 2) = 20 :=
by
  sorry

end digit_2_appears_20_times_l760_760551


namespace expression_value_eq_3084_l760_760803

theorem expression_value_eq_3084 (x : ℤ) (hx : x = -3007) :
  (abs (abs (Real.sqrt (abs x - x) - x) - x) - Real.sqrt (abs (x - x^2)) = 3084) :=
by
  sorry

end expression_value_eq_3084_l760_760803


namespace comparison_of_a_b_c_l760_760297

theorem comparison_of_a_b_c : 
  let a := (1/3)^(2/5)
  let b := 2^(4/3)
  let c := Real.logb 2 (1/3)
  c < a ∧ a < b :=
by
  sorry

end comparison_of_a_b_c_l760_760297


namespace average_gas_mileage_round_trip_l760_760134

theorem average_gas_mileage_round_trip
  (d : ℝ) (ms mr : ℝ)
  (h1 : d = 150)
  (h2 : ms = 35)
  (h3 : mr = 15) :
  (2 * d) / ((d / ms) + (d / mr)) = 21 :=
by
  sorry

end average_gas_mileage_round_trip_l760_760134


namespace sum_prime_numbers_less_than_twenty_l760_760030

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760030


namespace An_statement_l760_760304

/-- Define An and prove that A2 is true, but for n > 2, An is false. -/
theorem An_statement (n : ℕ) : 
    (∃ (k : ℕ) (a b : ℕ), (a + b = 2 * k * (nat.sqrt (a * b))) → a = b) ↔ n = 2 
    ∨ (n > 2 → ¬ ∀ (a b c: ℕ), (a + b + c = 3 * k * (nat.sqrt (a * b * c)/n)) → a = b = c ) :=
begin 
    sorry
end

end An_statement_l760_760304


namespace porter_l760_760832

def previous_sale_amount : ℕ := 9000

def recent_sale_price (previous_sale_amount : ℕ) : ℕ :=
  5 * previous_sale_amount - 1000

theorem porter's_recent_sale : recent_sale_price previous_sale_amount = 44000 :=
by
  sorry

end porter_l760_760832


namespace find_xyz_value_l760_760744

noncomputable def xyz_satisfying_conditions (x y z : ℝ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧
  (x + 1/y = 5) ∧
  (y + 1/z = 2) ∧
  (z + 1/x = 3)

theorem find_xyz_value (x y z : ℝ) (h : xyz_satisfying_conditions x y z) : x * y * z = 1 :=
by
  sorry

end find_xyz_value_l760_760744


namespace min_value_16x_4y_l760_760233

theorem min_value_16x_4y (x y : ℝ) (h : 4 * (x - 1) + 2 * y = 0) : 16^x + 4^y = 8 :=
begin
  -- The proof is omitted as the problem requires only the statement.
  sorry
end

end min_value_16x_4y_l760_760233


namespace ratio_of_areas_of_triangles_l760_760594

-- Definitions based on the conditions from step a)

-- Definition of an isosceles trapezoid inscribed in a circle
structure IsoscelesTrapezoid :=
  (A B C D : Type)
  (AD : A → D → ℝ)
  (BC : B → C → ℝ)
  (inscribed : ∀ (A B C D O : Type), O → A → D = 15)
  (parallel : ∀ (A B C D : Type), B → C = 5)

-- The definition of the triangles involved
structure Circle :=
  (center : Type)
  (radius : Type)

structure Triangle :=
  (A : Type)
  (B : Type)
  (C : Type)

-- Given Triangle properties
def RatioOfAreas (A D I B C : Type) [HasSmul A B] [HasSmul C D] : ℝ :=
  (AD.smul 15)^2 / (BC.smul 5)^2

-- The problem statement as a Lean theorem:
theorem ratio_of_areas_of_triangles :
  ∀ (A D I B C : Type) [Trapezoid : IsoscelesTrapezoid A B C D AD BC inscribed parallel],
    RatioOfAreas A D I B C = 9 :=
sorry

end ratio_of_areas_of_triangles_l760_760594


namespace find_P_7_l760_760312

-- This definition represents the polynomial P as stated in the problem
def P (x : ℝ) : ℝ :=
  (3 * x^4 - 30 * x^3 + a * x^2 + b * x + c) *
  (4 * x^4 - 84 * x^3 + d * x^2 + e * x + f)

-- The given conditions and question is to find P(7)
theorem find_P_7 (a b c d e f : ℝ)
  (h_roots : (polynomial.map polynomial.algebra_map (3 * polynomial.X^4 - 30 * polynomial.X^3 + a * polynomial.X^2 + b * polynomial.X + c) *
              polynomial.map polynomial.algebra_map (4 * polynomial.X^4 - 84 * polynomial.X^3 + d * polynomial.X^2 + e * polynomial.X + f)).roots = {2, 3, 4, 5, 5}) :
  P 7 = 86400 := 
sorry

end find_P_7_l760_760312


namespace initial_digit_is_2_or_9_l760_760564

theorem initial_digit_is_2_or_9 
  (x : ℕ) 
  (H1 : x < 10) 
  (H2 : ∃ n : ℕ, ─-(multiplier: x -> multiplier * 8 concat add: x + 14, x - 14)- = 777772) : 
  x = 2 ∨ x = 9 := 
sorry

end initial_digit_is_2_or_9_l760_760564


namespace find_n_l760_760692

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable (n : ℕ)

-- Given conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ m n, a_n (m + n) = a_n m + a_n n - a_n 0

def sum_first_n_terms (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  S n = (n / 2) * (a_n 1 + a_n n)

def condition1 := S 4 = 40
def condition2 := S n = 210
def condition3 := S (n - 4) = 130
def condition4 := sum_first_n_terms S a_n

-- The main theorem to prove
theorem find_n (S : ℕ → ℝ) (a_n : ℕ → ℝ) [arithmetic_sequence a_n] [sum_first_n_terms S a_n] :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 → n = 14 :=
sorry

end find_n_l760_760692


namespace greatest_two_digit_multiple_of_17_l760_760405

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760405


namespace grown_ups_in_milburg_l760_760898

def total_population : ℕ := 8243
def number_of_children : ℕ := 2987

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 :=
by {
  sorry
}

end grown_ups_in_milburg_l760_760898


namespace problem_statement_l760_760675

def g (a b : ℝ) : ℝ := a * Real.sqrt b - (1 / 4) * b

theorem problem_statement : ∃ (a : ℝ), ∀ (b : ℝ), b > 0 → g(a, 4) ≥ g(a, b) :=
by {
    sorry
}

end problem_statement_l760_760675


namespace constants_satisfy_equation_l760_760651

theorem constants_satisfy_equation :
  ∃ P Q R : ℚ, P = - 8 / 15 ∧ Q = - 7 / 6 ∧ R = 27 / 10 ∧
  (∀ x, x ≠ 1 → x ≠ 4 → x ≠ 6 → 
    (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) = 
    P / (x - 1) + Q / (x - 4) + R / (x - 6)) :=
by
  use [- 8 / 15, - 7 / 6, 27 / 10]
  split
  { refl },
  split
  { refl },
  split
  { norm_num },
  intros x h₁ h₄ h₆,
  sorry

end constants_satisfy_equation_l760_760651


namespace find_x_l760_760780

theorem find_x :
  ∀ (B C E F G A D : Type)
  [is_midpoint E B C]
  [area_triangle F E C = 7]
  [area_quadrilateral D B E G = 27]
  (x : ℕ)
  [area_triangle A D G = x]
  [area_triangle G E F = x],
  x = 8 := by
  sorry

end find_x_l760_760780


namespace sum_of_coefficients_not_x3_l760_760892

theorem sum_of_coefficients_not_x3 :
  let f := λ x : ℝ, (√x - 3 / x)^9
  in (f 1) - (Coeff.of_term (λ x, x^3) (f 1)) = -485 :=
by
  sorry

end sum_of_coefficients_not_x3_l760_760892


namespace sum_of_primes_less_than_20_eq_77_l760_760025

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760025


namespace sum_primes_less_than_20_l760_760078

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760078


namespace distance_between_foci_of_ellipse_l760_760991

theorem distance_between_foci_of_ellipse :
  ∃ (a b c : ℝ),
  -- Condition: axes are parallel to the coordinate axes (implicitly given by tangency points).
  a = 3 ∧
  b = 2 ∧
  c = Real.sqrt (a^2 - b^2) ∧
  2 * c = 2 * Real.sqrt 5 :=
sorry

end distance_between_foci_of_ellipse_l760_760991


namespace martha_initial_blocks_l760_760341

theorem martha_initial_blocks (final_blocks : ℕ) (found_blocks : ℕ) (initial_blocks : ℕ) : 
  final_blocks = initial_blocks + found_blocks → 
  final_blocks = 84 →
  found_blocks = 80 → 
  initial_blocks = 4 :=
by
  intros h1 h2 h3
  sorry

end martha_initial_blocks_l760_760341


namespace power_function_equation_l760_760254

-- Define the power function and its properties
def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Define the condition that the power function passes through the point (33, 3)
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f 33 = 3

-- Define the mathematical problem as a Lean theorem statement
theorem power_function_equation :
  ∃ α, (power_function α 33 = 3) → (∀ x, power_function α x = x ^ 3) :=
begin
  use 3,
  intro h,
  funext,
  have : α = 3 := sorry,
  rw this,
  reflexivity
end

end power_function_equation_l760_760254


namespace great_wall_scientific_notation_l760_760895

def length_of_great_wall : ℝ := 6700010
def scientific_notation (x : ℝ) (sig_figs : ℝ) : ℝ := (Real.floor (x * 10 ^ (-Real.floor(Log.log10(x) + 1 - sig_figs))) : ℝ) * 10 ^ (Real.floor(Log.log10(x) + 1) : ℝ)

theorem great_wall_scientific_notation :
  scientific_notation length_of_great_wall 2 = 6.7 * 10^6 :=
sorry

end great_wall_scientific_notation_l760_760895


namespace greatest_two_digit_multiple_of_17_l760_760521

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760521


namespace range_of_f_l760_760184

def g (x : ℝ) : ℝ := (cos (6 * x) + 2 * sin (3 * x) ^ 2) / (2 - 2 * cos (3 * x))

def f (x : ℝ) : ℝ := sqrt (1 - g x ^ 2)

theorem range_of_f :
  set.range f = set.Icc 0 (sqrt 15 / 4) :=
sorry

end range_of_f_l760_760184


namespace sum_primes_less_than_20_l760_760077

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760077


namespace sum_primes_less_than_20_l760_760071

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760071


namespace greatest_two_digit_multiple_of_17_l760_760476

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760476


namespace percent_students_scored_70_to_79_l760_760961

theorem percent_students_scored_70_to_79 :
  let tally_100 := 3
  let tally_90_99 := 6
  let tally_80_89 := 8
  let tally_70_79 := 10
  let tally_60_69 := 4
  let tally_50_59 := 3
  let tally_below_50 := 2
  let tally_no_attempt := 1
  let total_students := tally_100 + tally_90_99 + tally_80_89 + tally_70_79 + tally_60_69 + tally_50_59 + tally_below_50 + tally_no_attempt
  let percent_70_79 := (tally_70_79 / total_students.to_float) * 100
  percent_70_79 = 27 :=
by
  -- Definitions to use the conditions
  let tally_100 := 3
  let tally_90_99 := 6
  let tally_80_89 := 8
  let tally_70_79 := 10
  let tally_60_69 := 4
  let tally_50_59 := 3
  let tally_below_50 := 2
  let tally_no_attempt := 1
  let total_students := tally_100 + tally_90_99 + tally_80_89 + tally_70_79 + tally_60_69 + tally_50_59 + tally_below_50 + tally_no_attempt
  let percent_70_79 := (tally_70_79 / total_students.to_float) * 100
  -- Placeholder to skip the proof
  sorry

end percent_students_scored_70_to_79_l760_760961


namespace sum_of_primes_less_than_twenty_is_77_l760_760045

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760045


namespace solve_arithmetic_sequence_sum_l760_760876

noncomputable def arithmetic_sequence_sum : ℕ :=
  let a : ℕ := 3
  let b : ℕ := 10
  let c : ℕ := 17
  let e : ℕ := 32
  let d := b - a
  let c_term := c + d
  let d_term := c_term + d
  c_term + d_term

theorem solve_arithmetic_sequence_sum : arithmetic_sequence_sum = 55 :=
by
  sorry

end solve_arithmetic_sequence_sum_l760_760876


namespace center_of_rectangle_l760_760972

structure Point :=
(x : ℝ)
(y : ℝ)

def is_center_of_rectangle (O : Point) (A B C D : Point) : Prop :=
  let M := Point.mk ((A.x + C.x) / 2) ((A.y + C.y) / 2) in
  let M_eq := (M.x = O.x ∧ M.y = O.y) in
  (A.x = B.x) → (C.x = D.x) → (A.y = D.y) → (B.y = C.y) → M_eq

theorem center_of_rectangle
(O A B C D : Point)
(hA : A.x = 0 ∧ A.y = 0)
(hB : B.x = 55 ∧ B.y = 0)
(hC : C.x = 55 ∧ C.y = 40)
(hD : D.x = 0 ∧ D.y = 40) :
is_center_of_rectangle O A B C D :=
by {
  sorry
}

end center_of_rectangle_l760_760972


namespace sum_of_cos_squares_l760_760809

theorem sum_of_cos_squares (α β γ : ℝ) (h1 : α = real.angle.arbitrary_line_with_perpendicular_lines β γ) : 
  cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 1 :=
sorry

end sum_of_cos_squares_l760_760809


namespace series_inequality_l760_760339

open BigOperators

theorem series_inequality :
  (∑ k in Finset.range 2012, (1 / (((k + 1) * Real.sqrt k) + (k * Real.sqrt (k + 1))))) > 0.97 :=
sorry

end series_inequality_l760_760339


namespace number_of_valid_lines_l760_760774

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def lines_passing_through_point (x_int : ℕ) (y_int : ℕ) (p : ℕ × ℕ) : Prop :=
  p.1 * y_int + p.2 * x_int = x_int * y_int

theorem number_of_valid_lines (p : ℕ × ℕ) : 
  ∃! l : ℕ × ℕ, is_prime (l.1) ∧ is_power_of_two (l.2) ∧ lines_passing_through_point l.1 l.2 p :=
sorry

end number_of_valid_lines_l760_760774


namespace cri_du_chat_is_chromosomal_variation_l760_760138

-- Given definitions of genetic diseases
def Albinism : Prop := ∃ (m : Type), m ≠ "Chromosomal variation" ∧ m = "gene mutation"
def Hemophilia : Prop := ∃ (m : Type), m ≠ "Chromosomal variation" ∧ m = "gene mutation"
def Cri_du_chat_syndrome : Prop := ∃ (m : Type), m = "Chromosomal variation"
def Sickle_cell_anemia : Prop := ∃ (m : Type), m ≠ "Chromosomal variation" ∧ m = "gene mutation"

-- The proof statement
theorem cri_du_chat_is_chromosomal_variation : 
  Cri_du_chat_syndrome :=
sorry

end cri_du_chat_is_chromosomal_variation_l760_760138


namespace robot_visits_all_cells_l760_760121

-- Definitions for the problem
structure Cell :=
(x : Nat) (y : Nat)
(hx : x < 10)
(hy : y < 10)

inductive Command
| L | R | U | D

structure Labyrinth :=
(is_wall : Cell → Cell → Bool)
(connected : ∀ {a b : Cell}, is_wall a b = ff → (a.x = b.x ∧ (a.y = b.y + 1 ∨ a.y + 1 = b.y)) ∨ (a.y = b.y ∧ (a.x = b.x + 1 ∨ a.x + 1 = b.x)))
(accessible : ∀ a b, ∃ path : List Command, (∀ c ∈ path, ¬is_wall c (move c b)) → visit_all_cells a b path)

structure RobotState :=
(current : Cell)
(path : List Command)

-- Function to move the robot based on a command
def move : Command → Cell → Cell
| Command.L, ⟨x, y, hx, hy⟩ => if x > 0 then ⟨x - 1, y, Nat.sub_lt hx (Nat.zero_lt_succ y.sizeof), hy⟩ else ⟨x, y, hx, hy⟩
| Command.R, ⟨x, y, hx, hy⟩ => if x < 9 then ⟨x + 1, y, Nat.lt_succ_of_lt (Nat.lt_of_succ_lt_succ hx), hy⟩ else ⟨x, y, hx, hy⟩
| Command.U, ⟨x, y, hx, hy⟩ => if y > 0 then ⟨x, y - 1, hx, Nat.sub_lt hy (Nat.zero_lt_succ x.sizeof)⟩ else ⟨x, y, hx, hy⟩
| Command.D, ⟨x, y, hx, hy⟩ => if y < 9 then ⟨x, y + 1, hx, Nat.lt_succ_of_lt (Nat.lt_of_succ_lt_succ hy)⟩ else ⟨x, y, hx, hy⟩

-- Final proof statement
theorem robot_visits_all_cells (L : Labyrinth) (initial : Cell) :
  ∃ Pi : List Command, ∀ (pos : Cell), accessible initial pos Pi :=
sorry

end robot_visits_all_cells_l760_760121


namespace cost_to_feed_turtles_l760_760353

theorem cost_to_feed_turtles :
  (∀ (weight : ℚ), weight > 0 → (food_per_half_pound_ounces = 1) →
    (total_weight_pounds = 30) →
    (jar_contents_ounces = 15 ∧ jar_cost = 2) →
    (total_cost_dollars = 8)) :=
by
  sorry

variables
  (food_per_half_pound_ounces : ℚ := 1)
  (total_weight_pounds : ℚ := 30)
  (jar_contents_ounces : ℚ := 15)
  (jar_cost : ℚ := 2)
  (total_cost_dollars : ℚ := 8)

-- Assuming all variables needed to state the theorem exist and are meaningful
/-!
  Given:
  - Each turtle needs 1 ounce of food per 1/2 pound of body weight
  - Total turtles' weight is 30 pounds
  - Each jar of food contains 15 ounces and costs $2

  Prove:
  - The total cost to feed Sylvie's turtles is $8
-/

end cost_to_feed_turtles_l760_760353


namespace trigonometric_identity_l760_760212

variable (a b c θ α β γ : ℝ)

-- Given condition
def cond : Prop := 
  (tan (θ + α) / a = tan (θ + β) / b) ∧ 
  (tan (θ + β) / b = tan (θ + γ) / c)

-- Theorem statement
theorem trigonometric_identity (h : cond a b c θ α β γ) : 
  (a + b) / (a - b) * (sin (α - β)) ^ 2 +
  (b + c) / (b - c) * (sin (β - γ)) ^ 2 +
  (c + a) / (c - a) * (sin (γ - α)) ^ 2 = 0 := by
  sorry

end trigonometric_identity_l760_760212


namespace max_reflections_l760_760108

-- Conditions
def law_of_reflection (θi θr : ℝ) : Prop :=
  θi = θr -- angle of incidence equals angle of reflection

def incidence_angle (n : ℕ) : ℝ :=
  6 * n -- incidence angle increases by 6 degrees per reflection

-- Theorem statement
theorem max_reflections (A B : Point) (PQ RS : Line) (θ : ℝ) (n : ℕ)
  (h1 : PQ.contains A) (h2 : RS.contains B) (h3 : law_of_reflection θ θ)
  (h4 : θ = 6) : n ≤ 15 :=
by
  -- Define the incidence angles pattern
  have inc_angle : ℝ := incidence_angle n

  -- Prove the maximum number of reflections n before the angle reaches 90 degrees
  have prop : 6 * n ≤ 90 := by
    calc 6 * n ≤ 90 : sorry -- placeholder

  -- Derive the maximum integer n such that 6n ≤ 90
  exact 15

end max_reflections_l760_760108


namespace chord_length_l760_760763

theorem chord_length {r : ℝ} {θ : Real} (h_r : r = 15) (h_theta : θ = Real.pi / 6) :
  let c := 2 * r * Real.sin(θ / 2) in
  c = 15 :=
by
  -- Placeholder for the mathematical proof
  sorry

end chord_length_l760_760763


namespace quadratic_function_solution_exists_inequality_solution_range_l760_760687

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x - 5

theorem quadratic_function_solution_exists :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, f(x+1) - f(x) = 2 * x - 3)
  ∧ f 1 = -8
  ∧ ∀ x, f x = x^2 - 4 * x - 5) := sorry

theorem inequality_solution_range (m : ℝ) :
  (∀ x ∈ Icc (-2 : ℝ) 4, f x > 2 * x + m) ↔ m < -14 := sorry

end quadratic_function_solution_exists_inequality_solution_range_l760_760687


namespace greatest_two_digit_multiple_of_17_l760_760415

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760415


namespace cannot_win_l760_760385

-- Define the properties and the game rules
def sandwiches (N : ℕ) := 100 * N

-- Define the turns and rules for moves
structure Game (N : ℕ) :=
(Uf_moves : ℕ)
(M_moves : ℕ)
(Uf_move_first : Bool)

-- Define the winning condition for Uncle Fyodor
def Uf_wins (N : ℕ) (g : Game N) := ∀ matroskin_strategy : (fin (sandwiches N)) → Prop, ¬ matroskin_strategy (fin.last (sandwiches N))

-- Main theorem statement
theorem cannot_win (N : ℕ) : ∃ g : Game N, Uf_wins N g := 
sorry

end cannot_win_l760_760385


namespace arithmetic_sequence_sum_l760_760872

theorem arithmetic_sequence_sum (c d : ℤ) (h1 : c = 24) (h2 : d = 31) :
  c + d = 55 :=
by
  rw [h1, h2]
  exact rfl

end arithmetic_sequence_sum_l760_760872


namespace sequence_inequality_l760_760941

open_locale big_operators

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 0 ∧ (∀ n : ℕ, 0 ≤ a n) ∧ 
  (∀ i : ℕ, 1 ≤ i → i < n → (a (i-1) + a (i+1)) / 2 ≤ a i)

theorem sequence_inequality (a : ℕ → ℝ) (n : ℕ) (h : sequence a n):
  (∑ k in finset.range n.succ, a k) ^ 2 ≥ 3 * (n - 1) / 4 * ∑ k in finset.range n.succ, (a k) ^ 2 :=
sorry

end sequence_inequality_l760_760941


namespace find_angle_B_find_cos_C_l760_760757

open Real

variables (A B C a b c : ℝ)

noncomputable def condition1 := ∀ A B C a b c, b ≠ 0 → c ≠ 0 → cos C / cos B = (2 * a - c) / b

theorem find_angle_B (A B C a b c : ℝ) (h1 : condition1 A B C a b c) : B = π / 3 :=
sorry

variables (tan_pi_over_four_A : ℝ)

noncomputable def condition2 := tan (A + π / 4) = 7

theorem find_cos_C (A B C a b c : ℝ) (h1 : condition1 A B C a b c) (h2 : condition2 A) : cos C = (-4 + 3 * sqrt 3) / 10 :=
sorry

end find_angle_B_find_cos_C_l760_760757


namespace track_length_l760_760935

theorem track_length (L : ℝ)
  (h_brenda_first_meeting : ∃ (brenda_run1: ℝ), brenda_run1 = 100)
  (h_sally_first_meeting : ∃ (sally_run1: ℝ), sally_run1 = L/2 - 100)
  (h_brenda_second_meeting : ∃ (brenda_run2: ℝ), brenda_run2 = L - 100)
  (h_sally_second_meeting : ∃ (sally_run2: ℝ), sally_run2 = sally_run1 + 100)
  (h_meeting_total : brenda_run2 + sally_run2 = L) :
  L = 200 :=
by
  sorry

end track_length_l760_760935


namespace sufficient_but_not_necessary_l760_760944

theorem sufficient_but_not_necessary (a : ℝ) (h : a = 1/4) : 
  (∀ x : ℝ, x > 0 → x + a / x ≥ 1) ∧ ¬(∀ x : ℝ, x > 0 → x + a / x ≥ 1 ↔ a = 1/4) :=
by
  sorry

end sufficient_but_not_necessary_l760_760944


namespace find_m_from_root_l760_760250

theorem find_m_from_root :
  ∀ (m : ℕ), (x^2 - m*x + 2 = 0) -> (m = 3) := 
by
  intro m
  have h1 : (4 - 2*m + 2 = 0) := by sorry
  have h2 : (6 - 2*m = 0) := by sorry
  have h3 : (2*m = 6) := by sorry
  have h4 : (m = 3) := by sorry
  exact h4

end find_m_from_root_l760_760250


namespace new_bill_cost_l760_760243

def original_order_cost : ℝ := 25.00
def tomato_cost_old : ℝ := 0.99
def tomato_cost_new : ℝ := 2.20
def lettuce_cost_old : ℝ := 1.00
def lettuce_cost_new : ℝ := 1.75
def celery_cost_old : ℝ := 1.96
def celery_cost_new : ℝ := 2.00
def delivery_and_tip_cost : ℝ := 8.00

theorem new_bill_cost :
  original_order_cost +
  (tomato_cost_new - tomato_cost_old) +
  (lettuce_cost_new - lettuce_cost_old) +
  (celery_cost_new - celery_cost_old) +
  delivery_and_tip_cost = 35.00 :=
by
  simp [original_order_cost, tomato_cost_old, tomato_cost_new,
        lettuce_cost_old, lettuce_cost_new,
        celery_cost_old, celery_cost_new,
        delivery_and_tip_cost]
  norm_num
  -- Expected result of the simplification and normalization
  -- should be 35.00, thus leading to a successful proof.
  sorry

end new_bill_cost_l760_760243


namespace cube_roots_necessary_conditions_l760_760658

theorem cube_roots_necessary_conditions (x : ℝ) : x = x^(1/3) → (x = 0 ∨ x = 1 ∨ x = -1) :=
begin
  -- Proof to be provided
  sorry
end

end cube_roots_necessary_conditions_l760_760658


namespace second_player_wins_with_perfect_play_l760_760388

/-- There are 11 boxes, and if two players take turns placing coins in the boxes such that in each turn
one coin is placed in each of 10 different boxes, the winner is the player who first gets 21 coins
in any single box. Prove that the second player wins with perfect play. -/
theorem second_player_wins_with_perfect_play :
  ∀ (boxes : Fin 11 → Nat), ∃ (first_player_wins second_player_wins : Prop),
    (first_player_wins → second_player_wins) ∧ (is_optimal_play : second_player_wins) :=
sorry

end second_player_wins_with_perfect_play_l760_760388


namespace range_of_a_in_circle_l760_760217

theorem range_of_a_in_circle (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_a_in_circle_l760_760217


namespace brenda_mice_left_l760_760997

theorem brenda_mice_left (litters : ℕ) (mice_per_litter : ℕ) (fraction_to_robbie : ℚ) 
                          (mult_to_pet_store : ℕ) (fraction_to_feeder : ℚ) 
                          (total_mice : ℕ) (to_robbie : ℕ) (to_pet_store : ℕ) 
                          (remaining_after_first_sales : ℕ) (to_feeder : ℕ) (left_after_feeder : ℕ) :
  litters = 3 →
  mice_per_litter = 8 →
  fraction_to_robbie = 1/6 →
  mult_to_pet_store = 3 →
  fraction_to_feeder = 1/2 →
  total_mice = litters * mice_per_litter →
  to_robbie = total_mice * fraction_to_robbie →
  to_pet_store = mult_to_pet_store * to_robbie →
  remaining_after_first_sales = total_mice - to_robbie - to_pet_store →
  to_feeder = remaining_after_first_sales * fraction_to_feeder →
  left_after_feeder = remaining_after_first_sales - to_feeder →
  left_after_feeder = 4 := sorry

end brenda_mice_left_l760_760997


namespace probability_non_defective_pens_l760_760536

theorem probability_non_defective_pens :
  let total_pens := 12
  let defective_pens := 6
  let non_defective_pens := total_pens - defective_pens
  let probability_first_non_defective := non_defective_pens / total_pens
  let probability_second_non_defective := (non_defective_pens - 1) / (total_pens - 1)
  (probability_first_non_defective * probability_second_non_defective = 5 / 22) :=
by
  rfl

end probability_non_defective_pens_l760_760536


namespace center_axes_of_symmetry_l760_760713

def ellipse_eq (x y : ℝ) : Prop :=
  17 * x^2 - 16 * x * y + 4 * y^2 - 34 * x + 16 * y + 13 = 0

theorem center_axes_of_symmetry :
  (∀ x y : ℝ, ellipse_eq x y ↔ ellipse_eq (2 * 1 - x) (2 * 0 - y)) ∧
  (∀ x : ℝ, ∃ k : ℝ,
     y = (13 + 5*sqrt(17))/16 * (x - 1) ∨ y = (13 - 5*sqrt(17))/16 * (x - 1) ) := 
by 
  sorry

end center_axes_of_symmetry_l760_760713


namespace greatest_two_digit_multiple_of_17_l760_760483

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760483


namespace greatest_two_digit_multiple_of_17_l760_760472

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760472


namespace determinant_is_zero_l760_760172

-- Definitions of trigonometric functions and matrix
def matrix := 
  ![
    ![0, real.cos α, real.sin α],
    ![-real.cos α, 0, real.cos β],
    ![-real.sin α, -real.cos β, 0]
  ]

-- Statement of the lean proof problem
theorem determinant_is_zero (α β : ℝ) : matrix.det = 0 := 
by
  sorry

end determinant_is_zero_l760_760172


namespace solution_system_inequalities_l760_760846

theorem solution_system_inequalities (x : ℝ) : 
  (x - 4 ≤ 0 ∧ 2 * (x + 1) < 3 * x) ↔ (2 < x ∧ x ≤ 4) := 
sorry

end solution_system_inequalities_l760_760846


namespace cubic_root_monotonicity_contradiction_l760_760335

theorem cubic_root_monotonicity_contradiction (a b : ℝ) (h : a ≤ b) : ¬ (∃ c, c = ∛a ∧ c > ∛b) :=
by
  sorry

end cubic_root_monotonicity_contradiction_l760_760335


namespace cyclist_A_catches_up_B_l760_760107

-- Defining the speeds of cyclists A and B
def speed_A (D : ℝ) : ℝ := D / 30
def speed_B (D : ℝ) : ℝ := D / 40

-- Defining the time after which A catches up B
def time_to_catch_up (D : ℝ) : ℝ :=
  let v_A := speed_A D
  let v_B := speed_B D
  15

theorem cyclist_A_catches_up_B (D : ℝ) (hA : D / 30 > 0) (hB : D / 40 > 0) :
  let v_A := speed_A D
  let v_B := speed_B D
  v_A * 15 = v_B * (15 + 5) :=
by
  sorry

end cyclist_A_catches_up_B_l760_760107


namespace range_of_a_l760_760715

open Real Trigonometric

noncomputable def f (x : ℝ) : ℝ := (sin x + sqrt 3 * cos x)^2 - 2

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Ico (-π / 12) a, ∀ y ∈ Ico (-π / 12) a, f y ≤ f x) ↔ a ∈ (π / 6, ∞) := 
by
  sorry

end range_of_a_l760_760715


namespace total_cards_in_box_l760_760096

-- Definitions based on conditions
def xiaoMingCountsFaster (m h : ℕ) := 6 * h = 4 * m
def xiaoHuaForgets (h1 h2 : ℕ) := h1 + h2 = 112
def finalCardLeft (t : ℕ) := t - 1 = 112

-- Main theorem stating that the total number of cards is 353
theorem total_cards_in_box : ∃ N : ℕ, 
    (∃ m h1 h2 : ℕ,
        xiaoMingCountsFaster m h1 ∧
        xiaoHuaForgets h1 h2 ∧
        finalCardLeft N) ∧
    N = 353 :=
sorry

end total_cards_in_box_l760_760096


namespace fraction_identity_l760_760162

def at_op (a b : ℤ) : ℤ := a * b - 3 * b ^ 2
def hash_op (a b : ℤ) : ℤ := a + 2 * b - 2 * a * b ^ 2

theorem fraction_identity : (at_op 8 3) / (hash_op 8 3) = 3 / 130 := by
  sorry

end fraction_identity_l760_760162


namespace marshmallow_ratio_l760_760732

variables (M : ℕ) -- Number of marshmallows Michael can hold
variables (B : ℕ := M / 2) -- Number of marshmallows Brandon can hold
variables (H : ℕ := 8) -- Number of marshmallows Haley can hold
variables (total : ℕ := H + M + B) -- Total number of marshmallows held by all three

theorem marshmallow_ratio :
  total = 44 → (M / H) = 3 :=  
begin
  sorry  -- proof omitted
end

end marshmallow_ratio_l760_760732


namespace sum_of_distances_l760_760294

def parabola (x : ℝ) : ℝ := x^2

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem sum_of_distances :
  let focus := (0, 0),
      p1 := (-15, 225),
      p2 := (-3, 9),
      p3 := (8, 64),
      p4 := (10, 100) in
  distance focus p1 + distance focus p2 + distance focus p3 + distance focus p4 = 400.981 :=
begin
  sorry
end

end sum_of_distances_l760_760294


namespace sum_prime_numbers_less_than_twenty_l760_760034

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760034


namespace greatest_two_digit_multiple_of_17_l760_760421

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760421


namespace sum_primes_less_than_20_l760_760072

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760072


namespace sum_primes_less_than_20_l760_760070

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760070


namespace sum_of_primes_less_than_20_eq_77_l760_760018

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760018


namespace sum_primes_less_than_20_l760_760080

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760080


namespace sharks_win_more_than_90_percent_l760_760852

theorem sharks_win_more_than_90_percent (N : ℕ) : 
  (2 + N) / (5 + N) > 0.9 → N ≥ 26 :=
by
  sorry

end sharks_win_more_than_90_percent_l760_760852


namespace arithmetic_progression_num_terms_l760_760987

theorem arithmetic_progression_num_terms (a d n : ℕ) (h_even : n % 2 = 0) 
    (h_sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 30)
    (h_sum_even : (n / 2) * (2 * a + 2 * d + (n - 2) * d) = 36)
    (h_diff_last_first : (n - 1) * d = 12) :
    n = 8 := 
sorry

end arithmetic_progression_num_terms_l760_760987


namespace orthographic_projection_sphere_orthographic_projection_not_independent_cube_orthographic_projection_not_independent_regular_tetrahedron_orthographic_projection_not_independent_regular_triangular_pyramid_l760_760932

def orthographic_projection_independent (obj : Type) : Prop :=
  ∀ (orientation : obj → obj), 
  projection_top orientation = projection_top obj ∧
  projection_front orientation = projection_front obj ∧
  projection_side orientation = projection_side obj 

def is_sphere (obj : Type) : Prop := sorry -- Assume proper predicate for identification of the sphere
def is_cube (obj : Type) : Prop := sorry -- Assume proper predicate for identification of the cube
def is_regular_tetrahedron (obj : Type) : Prop := sorry -- Assume proper predicate for identification of the regular tetrahedron
def is_regular_triangular_pyramid (obj : Type) : Prop := sorry -- Assume proper predicate for identification of the regular triangular pyramid

theorem orthographic_projection_sphere :
  ∀ (obj : Type), is_sphere obj → orthographic_projection_independent obj :=
sorry -- The proof will be inserted here

theorem orthographic_projection_not_independent_cube :
  ∀ (obj : Type), is_cube obj → ¬orthographic_projection_independent obj :=
sorry -- The proof will be inserted here
  
theorem orthographic_projection_not_independent_regular_tetrahedron :
  ∀ (obj : Type), is_regular_tetrahedron obj → ¬orthographic_projection_independent obj :=
sorry -- The proof will be inserted here
  
theorem orthographic_projection_not_independent_regular_triangular_pyramid :
  ∀ (obj : Type), is_regular_triangular_pyramid obj → ¬orthographic_projection_independent obj :=
sorry -- The proof will be inserted here

end orthographic_projection_sphere_orthographic_projection_not_independent_cube_orthographic_projection_not_independent_regular_tetrahedron_orthographic_projection_not_independent_regular_triangular_pyramid_l760_760932


namespace percentage_fullness_before_storms_l760_760587

def capacity : ℕ := 200 -- capacity in billion gallons
def water_added_by_storms : ℕ := 15 + 30 + 75 -- total water added by storms in billion gallons
def percentage_after : ℕ := 80 -- percentage of fullness after storms
def amount_of_water_after_storms : ℕ := capacity * percentage_after / 100

theorem percentage_fullness_before_storms :
  (amount_of_water_after_storms - water_added_by_storms) * 100 / capacity = 20 := by
  sorry

end percentage_fullness_before_storms_l760_760587


namespace range_of_a_l760_760814

def f (x : ℝ) : ℝ :=
if x < 0 then (1 / 2)^x - 7 else sqrt x

theorem range_of_a (a : ℝ) : f a < 1 ↔ -3 < a ∧ a < 1 := by
  sorry

end range_of_a_l760_760814


namespace london_rome_distance_correct_l760_760365

noncomputable def dms_to_decimal (deg: ℕ) (min: ℕ) (sec: ℕ) : ℝ := 
  deg + min / 60.0 + sec / 3600.0

def latitude_london_dms := (51, 20, 49)
def longitude_london_dms := (17, 34, 15)
def latitude_rome_dms := (41, 53, 54)
def longitude_rome_dms := (30, 8, 48)

noncomputable def latitude_london := dms_to_decimal 51 20 49
noncomputable def longitude_london := dms_to_decimal 17 34 15
noncomputable def latitude_rome := dms_to_decimal 41 53 54
noncomputable def longitude_rome := dms_to_decimal 30 8 48

noncomputable def deg_to_rad (deg: ℝ) : ℝ := 
  deg * Real.pi / 180.0

noncomputable def lat_london_rad := deg_to_rad latitude_london
noncomputable def long_london_rad := deg_to_rad longitude_london
noncomputable def lat_rome_rad := deg_to_rad latitude_rome
noncomputable def long_rome_rad := deg_to_rad longitude_rome

noncomputable def haversine (phi1 phi2 delta_phi delta_lambda : ℝ) (R : ℝ) : ℝ :=
  let a := Real.sin (delta_phi / 2)^2 + Real.cos phi1 * Real.cos phi2 * Real.sin (delta_lambda / 2)^2
  let c := 2 * Real.arctan2 (Real.sqrt a) (Real.sqrt (1 - a))
  R * c

noncomputable def earth_radius_km : ℝ := 6371.0
noncomputable def km_to_nautical_miles (km: ℝ) : ℝ := km * 0.53996
noncomputable def nautical_miles_to_statute_miles (nautical_miles: ℝ) : ℝ := nautical_miles * 1.15078

noncomputable def distance_london_rome_km : ℝ := 
  let delta_phi := lat_london_rad - lat_rome_rad
  let delta_lambda := long_rome_rad - long_london_rad
  haversine lat_london_rad lat_rome_rad delta_phi delta_lambda earth_radius_km

noncomputable def distance_london_rome_nautical_miles : ℝ :=
  km_to_nautical_miles distance_london_rome_km

noncomputable def distance_london_rome_statute_miles : ℝ :=
  nautical_miles_to_statute_miles distance_london_rome_nautical_miles

theorem london_rome_distance_correct : distance_london_rome_statute_miles = 884 := 
  sorry

end london_rome_distance_correct_l760_760365


namespace sum_of_primes_less_than_20_l760_760009

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760009


namespace tangent_line_at_point_l760_760225

def f (x : ℝ) : ℝ := x^3 + x - 16

def f' (x : ℝ) : ℝ := 3*x^2 + 1

def tangent_line (x : ℝ) (f'val : ℝ) (p_x p_y : ℝ) : ℝ := f'val * (x - p_x) + p_y

theorem tangent_line_at_point (x y : ℝ) (h : x = 2 ∧ y = -6 ∧ f 2 = -6) : 
  ∃ a b c : ℝ, a*x + b*y + c = 0 ∧ a = 13 ∧ b = -1 ∧ c = -32 :=
by
  use 13, -1, -32
  sorry

end tangent_line_at_point_l760_760225


namespace range_of_g_l760_760182

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), 
  ∃ r ∈ set.Icc (Real.pi / 2 - Real.arctan 2) (Real.pi / 2 + Real.arctan 2), g x = r :=
by
  sorry

end range_of_g_l760_760182


namespace tangent_line_eqn_l760_760655

-- Define the function y = 4 / (e^x + 1)
def curve (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

-- State that the point of tangency is (0, 2)
def tangent_point : ℝ × ℝ := (0, 2)

-- Prove that the equation of the tangent line to the curve at the given point is x + y - 2 = 0
theorem tangent_line_eqn :
  ∀ x y : ℝ, (curve 0 = 2) → ((y - curve 0) = -1 * (x - 0) → x + y - 2 = 0) :=
by
  -- skip the proof
  sorry

end tangent_line_eqn_l760_760655


namespace place_two_after_three_digit_number_l760_760252

theorem place_two_after_three_digit_number (h t u : ℕ) 
  (Hh : h < 10) (Ht : t < 10) (Hu : u < 10) : 
  (100 * h + 10 * t + u) * 10 + 2 = 1000 * h + 100 * t + 10 * u + 2 := 
by
  sorry

end place_two_after_three_digit_number_l760_760252


namespace greatest_two_digit_multiple_of_17_l760_760450

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760450


namespace overlapping_squares_proof_l760_760825

noncomputable def overlapping_squares_area (s : ℝ) : ℝ :=
  let AB := s
  let MN := s
  let areaMN := s^2
  let intersection_area := areaMN / 4
  intersection_area

theorem overlapping_squares_proof (s : ℝ) :
  overlapping_squares_area s = s^2 / 4 := by
    -- proof would go here
    sorry

end overlapping_squares_proof_l760_760825


namespace find_ab_l760_760697

theorem find_ab (A B : Set ℝ) (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = Set.univ) → 
  (A ∩ B = {x | 3 < x ∧ x ≤ 4}) →
  a + b = -7 :=
by
  intros
  sorry

end find_ab_l760_760697


namespace probability_palindrome_divisible_by_11_is_zero_l760_760580

-- Define the three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b + a

-- Define the divisibility condition
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Prove that the probability is zero
theorem probability_palindrome_divisible_by_11_is_zero :
  (∃ n, is_palindrome n ∧ is_divisible_by_11 n) →
  (0 : ℕ) = 0 := by
  sorry

end probability_palindrome_divisible_by_11_is_zero_l760_760580


namespace bloggers_tiktok_l760_760550

-- Definitions of the problem conditions
def bloggers := Fin 9
def day := Fin 3
def group := List bloggers
def adjacent (a b : bloggers) : Prop := 
  match a, b with
  | 0, 1 | 1, 0 | 1, 2 | 2, 1 | 3, 4 | 4, 3 |
    4, 5 | 5, 4 | 6, 7 | 7, 6 | 7, 8 | 8, 7 | 
    0, 3 | 3, 0 | 1, 4 | 4, 1 | 2, 5 | 5, 2 | 
    3, 6 | 6, 3 | 4, 7 | 7, 4 | 5, 8 | 8, 5 => true
  | _, _ => false

noncomputable def exists_unrecorded_pair : Prop := 
  ∃ (a b : bloggers), 
  ¬ ∃ (d : day),
  (adjacent a b)

-- The final theorem statement
theorem bloggers_tiktok : exists_unrecorded_pair :=
  sorry

end bloggers_tiktok_l760_760550


namespace primes_between_70_and_80_l760_760737

theorem primes_between_70_and_80 : (finset.filter nat.prime (finset.Icc 70 80)).card = 3 :=
by
  sorry

end primes_between_70_and_80_l760_760737


namespace consecutive_green_balls_l760_760938

theorem consecutive_green_balls : ∃ (fill_ways : ℕ), fill_ways = 21 ∧ 
  (∃ (boxes : Fin 6 → Bool), 
    (∀ i, boxes i = true → 
      (∀ j, boxes j = true → (i ≤ j ∨ j ≤ i)) ∧ 
      ∃ k, boxes k = true)) :=
by
  sorry

end consecutive_green_balls_l760_760938


namespace number_of_girls_not_playing_soccer_l760_760272

namespace ParkwayElementary

theorem number_of_girls_not_playing_soccer :
  ∀ (total_students boys students_playing_soccer : ℕ)
    (percent_boys_playing_soccer : ℝ),
    total_students = 470 →
    boys = 300 →
    students_playing_soccer = 250 →
    percent_boys_playing_soccer = 0.86 →
    let boys_playing_soccer := (percent_boys_playing_soccer * students_playing_soccer).to_nat in
    let total_girls := total_students - boys in
    let girls_playing_soccer := students_playing_soccer - boys_playing_soccer in
    total_girls - girls_playing_soccer = 135 :=
by
  intros total_students boys students_playing_soccer percent_boys_playing_soccer
  sorry

end ParkwayElementary

end number_of_girls_not_playing_soccer_l760_760272


namespace milk_needed_for_cookies_l760_760284

theorem milk_needed_for_cookies : 
  ∀ (num_half_gallons : ℕ) (num_cookies : ℕ),
  num_half_gallons = 50 →
  num_cookies = 200 * 12 →
  (1 : ℕ ≤ 40 / (num_cookies / num_half_gallons)) →
  1 = 1 := 
by
  intros num_half_gallons num_cookies h1 h2 h3
  sorry

end milk_needed_for_cookies_l760_760284


namespace playground_area_l760_760371

open Real

theorem playground_area (l w : ℝ) (h1 : 2*l + 2*w = 100) (h2 : l = 2*w) : l * w = 5000 / 9 :=
by
  sorry

end playground_area_l760_760371


namespace range_of_a_l760_760709

noncomputable theory

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f(x1) = a ∧ f(x2) = a ∧ f(x3) = a) ↔ a ∈ Ioo (-2 : ℝ) (2 : ℝ) :=
by
  sorry

end range_of_a_l760_760709


namespace total_bags_l760_760592

theorem total_bags (people : ℕ) (bags_per_person : ℕ) (h_people : people = 4) (h_bags_per_person : bags_per_person = 8) : people * bags_per_person = 32 := by
  sorry

end total_bags_l760_760592


namespace binary_to_base5_l760_760616

theorem binary_to_base5 : Nat.digits 5 (Nat.ofDigits 2 [1, 0, 1, 1, 0, 0, 1]) = [4, 2, 3] :=
by
  sorry

end binary_to_base5_l760_760616


namespace variance_comparison_l760_760264

variable (A B : Type) [HasVariance A] [HasVariance B]

def average_scores_equal (μ_A μ_B : ℝ) : Prop :=
  μ_A = μ_B

def more_uniform_scores (S1 S2 : ℝ) : Prop :=
  S1 < S2

theorem variance_comparison
  (μ_A μ_B S_1 S_2 : ℝ)
  (h_avg_eq : average_scores_equal μ_A μ_B)
  (h_more_uniform : more_uniform_scores S_1 S_2) :
  S_1 < S_2 := 
by
  sorry

end variance_comparison_l760_760264


namespace greatest_two_digit_multiple_of_17_l760_760490

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760490


namespace greatest_two_digit_multiple_of_17_l760_760482

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760482


namespace greatest_two_digit_multiple_of_17_l760_760426

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760426


namespace unique_painted_cube_l760_760958

/-- Determine the number of distinct ways to paint a cube where:
  - One side is yellow,
  - Two sides are purple,
  - Three sides are orange.
  Taking into account that two cubes are considered identical if they can be rotated to match. -/
theorem unique_painted_cube :
  ∃ unique n : ℕ, n = 1 ∧
    (∃ (c : Fin 6 → Fin 3), 
      (∃ (i : Fin 6), c i = 0) ∧ 
      (∃ (j k : Fin 6), j ≠ k ∧ c j = 1 ∧ c k = 1) ∧ 
      (∃ (m p q : Fin 6), m ≠ p ∧ m ≠ q ∧ p ≠ q ∧ c m = 2 ∧ c p = 2 ∧ c q = 2)
    ) :=
sorry

end unique_painted_cube_l760_760958


namespace seats_taken_correct_l760_760668

-- Define the conditions
def rows := 40
def chairs_per_row := 20
def unoccupied_seats := 10

-- Define the total number of seats
def total_seats := rows * chairs_per_row

-- Define the number of seats taken
def seats_taken := total_seats - unoccupied_seats

-- Statement of our math proof problem
theorem seats_taken_correct : seats_taken = 790 := by
  sorry

end seats_taken_correct_l760_760668


namespace triangle_inequality_l760_760806

theorem triangle_inequality 
  (a b c : ℝ)
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
sorry

end triangle_inequality_l760_760806


namespace transformed_function_symmetry_l760_760847

-- Stretch the original function by a factor of 2 and shift it to the left by π/4
theorem transformed_function_symmetry :
  ∃ x ∈ ℝ, sin (8 * x - π / 6 - π / 2) = sin (2 * x + π / 3) → x = π / 12 :=
sorry

end transformed_function_symmetry_l760_760847


namespace max_distinct_letters_in_5x5_table_l760_760771

noncomputable def maxDistinctLetters (table : Matrix (Fin 5) (Fin 5) Char) : Nat :=
  (⟨table, sorry⟩ : Exists (λ table : Matrix (Fin 5) (Fin 5) Char, 
    ∀ i : Fin 5, (table i).toFinset.card ≤ 3 ∧ 
    ∀ j : Fin 5, (fun ij => table (ij, j)).toFinset.card ≤ 3
  )).val.map (fun x => x.toList).toList.toFinset.card

theorem max_distinct_letters_in_5x5_table : 
  ∀ (table : Matrix (Fin 5) (Fin 5) Char), 
  (∀ i : Fin 5, (table i).toFinset.card ≤ 3) → 
  (∀ j : Fin 5, (fun ij => table (ij, j)).toFinset.card ≤ 3) →
  maxDistinctLetters table = 11 := sorry

end max_distinct_letters_in_5x5_table_l760_760771


namespace lines_intersection_l760_760543

theorem lines_intersection (n : ℕ) (h1 : n = 5)
  (h2 : ∀ i j : fin n, i ≠ j → ¬ parallel (line i) (line j))
  (h3 : ∀ i j k : fin n, i ≠ j → j ≠ k → k ≠ i → ¬ concurrent (line i) (line j) (line k)) :
  (∑ i in (finset.range n).choose 2, 1) = 10 :=
by
  simp [h1]
  sorry

end lines_intersection_l760_760543


namespace radio_loss_percentage_l760_760539

theorem radio_loss_percentage (cost_price selling_price : ℕ) (h1 : cost_price = 1500) (h2 : selling_price = 1305) : 
  (cost_price - selling_price) * 100 / cost_price = 13 := by
  sorry

end radio_loss_percentage_l760_760539


namespace sum_primes_less_than_20_l760_760073

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760073


namespace chess_player_exactly_21_games_l760_760112

theorem chess_player_exactly_21_games :
  ∀ (a : ℕ → ℕ), (∀ n, 1 ≤ a n) ∧ (∑ i in finset.range 78, a i ≤ 132) →
    ∃ (i j : ℕ), i < j ∧ (∑ k in finset.Ico i j, a k = 21) :=
by
  intro a h
  sorry

end chess_player_exactly_21_games_l760_760112


namespace sum_prime_numbers_less_than_twenty_l760_760038

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760038


namespace turtle_feeding_cost_l760_760348

theorem turtle_feeding_cost (total_weight_pounds : ℝ) (food_per_half_pound : ℝ)
  (jar_food_ounces : ℝ) (jar_cost_dollars : ℝ) (total_cost : ℝ) : 
  total_weight_pounds = 30 →
  food_per_half_pound = 1 →
  jar_food_ounces = 15 →
  jar_cost_dollars = 2 →
  total_cost = 8 :=
by
  intros h_weight h_food h_jar_ounces h_jar_cost
  sorry

end turtle_feeding_cost_l760_760348


namespace relationship_f_neg1_f_1_l760_760682

variable {ℝ : Type*} [Real ℝ]

theorem relationship_f_neg1_f_1 (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_eq : ∀ x : ℝ, f x = x^2 * (f' 2) - 3 * x) :
  f (-1) > f 1 := 
sorry

end relationship_f_neg1_f_1_l760_760682


namespace problem_l760_760754

theorem problem (a b : ℝ)
  (h : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ (x < -1/2 ∨ x > 1/3)) : 
  a + b = -14 :=
sorry

end problem_l760_760754


namespace greatest_two_digit_multiple_of_17_l760_760501

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760501


namespace six_points_seventeen_triangles_l760_760169

theorem six_points_seventeen_triangles : 
  ∃ (points : Finset (ℝ × ℝ)), points.card = 6 ∧ 
  (∑ S in points.powerset.filter (λ S, S.card = 3), (if collinear S then 0 else 1)) = 17 :=
sorry

end six_points_seventeen_triangles_l760_760169


namespace sum_alternating_series_l760_760154

theorem sum_alternating_series :
  (Finset.sum (Finset.range 2023) (λ k => (-1)^(k + 1))) = -1 := 
by
  sorry

end sum_alternating_series_l760_760154


namespace alice_can_always_win_l760_760591

open Function Set

-- Define the elements on the checkered strip
inductive Cell
| X: Cell  -- Cross represents by 'X'
| O: Cell  -- Zero represents by 'O'
| Empty: Cell  -- Empty cell

-- Game state
structure GameState :=
  (board : ℤ → Cell)
  (turn : ℕ)

-- Alice's win condition (4-term arithmetic progression)
def Alice_win_condition (board : ℤ → Cell) : Prop :=
  ∃ a d : ℤ, d ≠ 0 ∧
    board a = Cell.X ∧ board (a + d) = Cell.X ∧
    board (a + 2 * d) = Cell.X ∧ board (a + 3 * d) = Cell.X

-- Constraint for Bob's moves
def Bob_move_constraint : ℕ → Prop := λ n, n = 2020

-- Initial condition: Alice moves first
def initial_turn : GameState → Prop := λ state, state.turn % 2 = 0

-- Main theorem statement
theorem alice_can_always_win :
  ∀ (state : GameState),
  initial_turn state →
  (∀ n, Bob_move_constraint n) →
  ∃ (strategy : GameState → ℤ), -- Alice's strategy mapping states to board positions
  ∀ state',
    strategy state' ≠ 0 →
    Alice_win_condition (λ i, if i = strategy state' then Cell.X else state'.board i) :=
by
  sorry

end alice_can_always_win_l760_760591


namespace factorize_polynomial_l760_760639

theorem factorize_polynomial (x y : ℝ) : (3 * x^2 - 3 * y^2) = 3 * (x + y) * (x - y) := 
by
  sorry

end factorize_polynomial_l760_760639


namespace subset_condition_l760_760726

theorem subset_condition (m : ℝ) (A : set ℝ) (B : set ℝ) (hA : A = {-1, 3, m^2}) (hB : B = {3, 4}) (h : B ⊆ A) : m = 2 ∨ m = -2 :=
by
  sorry

end subset_condition_l760_760726


namespace volume_of_red_tetrahedron_l760_760116

noncomputable def cube_side_length : ℝ := 10

def cube_volume (s : ℝ) : ℝ := s^3

def tetrahedron_base_area (s : ℝ) : ℝ := (1 / 2) * s * s

def tetrahedron_volume (base_area : ℝ) (height : ℝ) : ℝ := (1 / 3) * base_area * height

def clear_tetrahedra_volume (s : ℝ) : ℝ :=
  let base_area := tetrahedron_base_area s
  let vol_one_tetrahedron := tetrahedron_volume base_area s
  4 * vol_one_tetrahedron

def red_tetrahedron_volume (s : ℝ) : ℝ :=
  let cube_vol := cube_volume s
  let total_clear_vol := clear_tetrahedra_volume s
  cube_vol - total_clear_vol

theorem volume_of_red_tetrahedron :
  red_tetrahedron_volume cube_side_length = 333.33 := by
  sorry

end volume_of_red_tetrahedron_l760_760116


namespace M_equals_3_or_9_l760_760704

-- Definitions and conditions derived from the problem
variable (M : ℕ)

-- Conditions
def divisibility_rule_independent (M : ℕ) : Prop :=
  ∀ (n : ℕ) (a : Fin n → Fin 10),
    M ∣ ∑ i in Finset.univ, (10 ^ i) * (a i) ↔
    M ∣ ∑ i in Finset.univ, (a i)

theorem M_equals_3_or_9 :
  M ≠ 1 →
  divisibility_rule_independent M →
  M = 3 ∨ M = 9 :=
by
  sorry

end M_equals_3_or_9_l760_760704


namespace calculate_g_inv_l760_760301

noncomputable def g : ℤ → ℤ := sorry
noncomputable def g_inv : ℤ → ℤ := sorry

axiom g_inv_eq : ∀ x, g (g_inv x) = x

axiom cond1 : g (-1) = 2
axiom cond2 : g (0) = 3
axiom cond3 : g (1) = 6

theorem calculate_g_inv : 
  g_inv (g_inv 6 - g_inv 2) = -1 := 
by
  -- The proof goes here
  sorry

end calculate_g_inv_l760_760301


namespace green_leaves_count_l760_760389

def green_leaves_left (total_plants : ℕ) (leaves_per_plant : ℕ) (fraction_yellow : ℚ) : ℕ :=
  let total_leaves := total_plants * leaves_per_plant
  let yellow_leaves := fraction_yellow * total_leaves
  total_leaves - yellow_leaves.to_nat

theorem green_leaves_count (h : green_leaves_left 5 24 (2 / 5) = 72) : true := 
  by
  sorry

end green_leaves_count_l760_760389


namespace difference_of_fractions_l760_760101

theorem difference_of_fractions (a b c : ℝ) (h1 : a = 8000 * (1/2000)) (h2 : b = 8000 * (1/10)) (h3 : c = b - a) : c = 796 := 
sorry

end difference_of_fractions_l760_760101


namespace bounded_above_unbounded_below_solutions_l760_760097

noncomputable def differential_eq := λ (y : ℝ → ℝ), ∀ x, has_deriv_at (deriv y) ((x ^ 3 + x * k) * y x) x

theorem bounded_above_unbounded_below_solutions (k : ℝ) (y : ℝ → ℝ)
  (h_satisfies_eq : differential_eq y)
  (h_init_conditions : (y 0 = 1) ∧ (deriv y 0 = 0)) :
  (∃ M, ∀ x > M, y x ≠ 0) ∧ ¬ (∃ N, ∀ x < N, y x ≠ 0) :=
sorry

end bounded_above_unbounded_below_solutions_l760_760097


namespace quadratic_factored_value_m_l760_760686

theorem quadratic_factored_value_m :
  ∃ (a b : ℝ), a + b = 7 ∧ -8 * a + 3 * b = 43 ∧ a * b = -18 :=
by
  use -2, 9
  split; linarith
  split; linarith
  exact rfl

end quadratic_factored_value_m_l760_760686


namespace diagonals_equal_if_area_equals_product_of_medians_l760_760833

-- Definition of Convex Quadrilateral
structure ConvexQuadrilateral where
  A B C D : Point
  is_convex : convex_quadrilateral A B C D

-- Definition of Midpoint
def midpoint (P Q : Point) : Point := sorry

-- Definition of Median Lines
def median_line (Q1 Q2 : Point) (A B C D : Point) := 
  line_segment (midpoint A B) (midpoint C D)

-- Definition of Area
def area (Q : ConvexQuadrilateral) : Real := sorry

-- Definition of Diagonals
def diagonals (Q : ConvexQuadrilateral) : (Real × Real) := sorry

-- Theorem Statement
theorem diagonals_equal_if_area_equals_product_of_medians (Q : ConvexQuadrilateral):
  area Q = (line_length Q.median_line_1) * (line_length Q.median_line_2) →
  (Q.diagonals.1 = Q.diagonals.2) :=
by
  sorry

end diagonals_equal_if_area_equals_product_of_medians_l760_760833


namespace factorize_3x2_minus_3y2_l760_760643

theorem factorize_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factorize_3x2_minus_3y2_l760_760643


namespace frog_jumps_further_l760_760367

-- Given conditions
def grasshopper_jump : ℕ := 9 -- The grasshopper jumped 9 inches
def frog_jump : ℕ := 12 -- The frog jumped 12 inches

-- Proof statement
theorem frog_jumps_further : frog_jump - grasshopper_jump = 3 := by
  sorry

end frog_jumps_further_l760_760367


namespace sqrt_eq_self_l760_760357

theorem sqrt_eq_self (x : ℝ) : (sqrt x = x) ↔ (x = 0 ∨ x = 1) :=
by sorry

end sqrt_eq_self_l760_760357


namespace sum_of_primes_less_than_twenty_is_77_l760_760050

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760050


namespace find_smaller_number_l760_760373

noncomputable def smallest_number (n1 n2 LCM : ℕ) (ratio1 ratio2: ℕ) : ℕ :=
  if h : ratio2 * n1 = ratio1 * n2 then
    if h_LCM : nat.lcm n1 n2 = LCM then
      n1
    else
      0
  else
    0

theorem find_smaller_number : smallest_number 80 120 120 2 3 = 80 :=
by {
  have h1 : 3 * 80 = 2 * 120,
  {
    norm_num,
  },
  have h2 : nat.lcm 80 120 = 120,
  {
    rw nat.lcm, norm_num,
  },
  exact if_pos h1 (if_pos h2 rfl),
}

end find_smaller_number_l760_760373


namespace rational_t_rational_s1_irrational_s1_l760_760696

-- Part (a)
theorem rational_t (x y : ℝ) (s2 s3 s4 t : ℝ) (s2_rat : s2 = x^2 + y^2) 
                   (s3_rat : s3 = x^3 + y^3) (s4_rat : s4 = x^4 + y^4) (r_s2 : s2 ∈ ℚ) 
                   (r_s3 : s3 ∈ ℚ) (r_s4 : s4 ∈ ℚ) : t ∈ ℚ :=
sorry

-- Part (b)
theorem rational_s1 (x y : ℝ) (s1 s2 s3 s4 t : ℝ) (s2_rat : s2 = x^2 + y^2) 
                    (s3_rat : s3 = x^3 + y^3) (s4_rat : s4 = x^4 + y^4) (t_rat : t = x * y) 
                    (r_s2 : s2 ∈ ℚ) (r_s3 : s3 ∈ ℚ) (r_s4 : s4 ∈ ℚ) : s1 ∈ ℚ :=
sorry

-- Part (c)
theorem irrational_s1 (x y : ℝ) (s1 s2 s3 : ℝ) (s2_rat : s2 = x^2 + y^2) 
                      (s3_rat : s3 = x^3 + y^3) (r_s2 : s2 ∈ ℚ) (r_s3 : s3 ∈ ℚ) : ¬ (s1 ∈ ℚ) → false :=
sorry

end rational_t_rational_s1_irrational_s1_l760_760696


namespace train_length_is_100_l760_760979

-- Given conditions:
def train_speed_km_h : ℝ := 60
def time_to_cross_bridge_seconds : ℝ := 10.799136069114471
def bridge_length_meters : ℝ := 80

-- Conversion from km/h to m/s
def kmh_to_mps (kmh: ℝ) : ℝ := kmh * (1000 / 3600)
noncomputable def train_speed_m_s := kmh_to_mps train_speed_km_h

-- Calculation of total distance (distance = speed * time)
noncomputable def total_distance_covered := train_speed_m_s * time_to_cross_bridge_seconds

-- Length of the train
noncomputable def train_length := total_distance_covered - bridge_length_meters

-- Theorem: Prove the length of the train is approximately 100 meters.
theorem train_length_is_100 : train_length ≈ 100 := 
by
  sorry

end train_length_is_100_l760_760979


namespace red_area_percentage_l760_760133

def red_area_percentage_flag (k : ℝ) (total_cross_area_percent : ℝ) : ℝ :=
  let y := 0.35 in 4 * y^2

theorem red_area_percentage (k : ℝ) (total_cross_area_percent : ℝ) : 
  total_cross_area_percent = 0.49 →
  red_area_percentage_flag k total_cross_area_percent = 0.49 :=
by
  intro h
  rw [red_area_percentage_flag]
  norm_num
  exact h

end red_area_percentage_l760_760133


namespace number_of_badminton_players_l760_760267

/-
Conditions:
  Total number of members: 30
  Members playing tennis: T = 19
  Members playing both sports: B∩T = 8
  Members not playing either sport: 2

Prove:
  Members playing badminton B = 17
-/

theorem number_of_badminton_players (total_members : ℕ) (T : ℕ) (B_and_T : ℕ) (neither : ℕ)
  (h_total : total_members = 30) (h_T : T = 19) (h_B_and_T : B_and_T = 8) (h_neither : neither = 2) :
  ∃ B : ℕ, B = 17 :=
by {
  have total_either_or_both := total_members - neither,
  have number_playing_either_or_both := 28,
  have B := 28 - T + B_and_T,
  existsi B,
  sorry
}

end number_of_badminton_players_l760_760267


namespace midpoint_of_CN_l760_760202

noncomputable def circumscribed_trapezium :=
  {A B C D P E K N M : Type}
  [CircumscribedTrapezium A B C D]
  [AD_parallel_BC A D B C]
  [BC_less_AD B C A D]
  [Circumcircle A B C D]
  [TangentAtC_meets_AD_at_P C A D P]
  [Tangent_PE_at_E P E]
  [Line_BP_meets_circle_at_K B P K A B C D]
  [Line_through_C_parallel_AB_meets_AE_at_N C A B E N]
  [Line_through_C_parallel_AB_meets_AK_at_M C A B K M]

theorem midpoint_of_CN (h : circumscribed_trapezium) : Midpoint M C N :=
  sorry

end midpoint_of_CN_l760_760202


namespace evaluate_F_2_f_3_l760_760158

def f (a : ℤ) : ℤ := a^2 - 1

def F (a b : ℤ) : ℤ := b^3 - a

theorem evaluate_F_2_f_3 : F 2 (f 3) = 510 := by
  sorry

end evaluate_F_2_f_3_l760_760158


namespace arithmetic_square_root_l760_760730

theorem arithmetic_square_root (x y : ℝ) (h1 : sqrt(x - 2) = 2 ∨ sqrt(x - 2) = -2) (h2 : real.cbrt(2 * x + y + 7) = 3) :
  sqrt(x ^ 2 + y ^ 2) = 10 :=
sorry

end arithmetic_square_root_l760_760730


namespace greatest_two_digit_multiple_of_17_l760_760438

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760438


namespace greatest_two_digit_multiple_of_17_l760_760430

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760430


namespace first_complete_row_cover_l760_760132

def is_shaded_square (n : ℕ) : ℕ := n ^ 2

def row_number (square_number : ℕ) : ℕ :=
  (square_number + 9) / 10 -- ceiling of square_number / 10

theorem first_complete_row_cover : ∃ n, ∀ r : ℕ, 1 ≤ r ∧ r ≤ 10 → ∃ k : ℕ, is_shaded_square k ≤ n ∧ row_number (is_shaded_square k) = r :=
by
  use 100
  intros r h
  sorry

end first_complete_row_cover_l760_760132


namespace greatest_two_digit_multiple_of_17_is_85_l760_760461

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760461


namespace trig_identity1_trig_identity2_l760_760608

theorem trig_identity1 : 
  sin (-1395 * 3.141592653589793 / 180) * cos (1140 * 3.141592653589793 / 180) +
  cos (-1020 * 3.141592653589793 / 180) * sin (750 * 3.141592653589793 / 180) 
  = (Real.sqrt 2 + 1) / 4 := 
by sorry

theorem trig_identity2 : 
  sin (-11 * 3.141592653589793 / 6) + 
  cos (3 * 3.141592653589793 / 4) * tan (4 * 3.141592653589793) 
  = 1 / 2 := 
by sorry

end trig_identity1_trig_identity2_l760_760608


namespace martha_bedroom_size_l760_760881

theorem martha_bedroom_size (x jenny_size total_size : ℤ) (h₁ : jenny_size = x + 60) (h₂ : total_size = x + jenny_size) (h_total : total_size = 300) : x = 120 :=
by
  -- Adding conditions and the ultimate goal
  sorry


end martha_bedroom_size_l760_760881


namespace Pierre_correct_response_probability_l760_760822

-- Define the probabilities and events based on the conditions given in the problem
noncomputable def Probability := ℝ
noncomputable def A : Probability := 1/2  -- Aunt is at home
noncomputable def B_given_A : Probability := 3/5  -- Aunt answers phone if home
noncomputable def B_not_given_A : Probability := 1 - B_given_A  -- Aunt doesn't answer phone if home
noncomputable def P_A_and_B : Probability := A * B_given_A  -- Probability aunt is home and answers phone
noncomputable def P_A_and_B_not : Probability := A * B_not_given_A  -- Probability aunt is home and doesn't answer phone

noncomputable def P_A_and_B_or_B_not : Probability := P_A_and_B + P_A_and_B_not  -- Combined scenario probabilities

-- Statement to prove
theorem Pierre_correct_response_probability : P_A_and_B_or_B_not = 3 / 16 := by
  sorry

end Pierre_correct_response_probability_l760_760822


namespace coin_toss_sequences_l760_760237

theorem coin_toss_sequences (s : list char) :
  s.length = 17 ∧
  (count_subseq s ['H', 'H'] = 3) ∧
  (count_subseq s ['H', 'T'] = 2) ∧
  (count_subseq s ['T', 'H'] = 5) ∧
  (count_subseq s ['T', 'T'] = 6) →
  (number_of_sequences s = 2940) :=
sorry

end coin_toss_sequences_l760_760237


namespace find_savings_l760_760369

noncomputable def savings (income expenditure : ℕ) : ℕ :=
  income - expenditure

theorem find_savings (I E : ℕ) (h_ratio : I = 9 * E) (h_income : I = 18000) : savings I E = 2000 :=
by
  sorry

end find_savings_l760_760369


namespace maximum_m_value_l760_760206

theorem maximum_m_value (a : ℕ → ℤ) (m : ℕ) :
  (∀ n, a (n + 1) - a n = 3) →
  a 3 = -2 →
  (∀ k : ℕ, k ≥ 4 → (3 * k - 8) * (3 * k - 5) / (3 * k - 11) ≥ 3 * m - 11) →
  m ≤ 9 :=
by
  sorry

end maximum_m_value_l760_760206


namespace hyperbola_eccentricity_l760_760966

-- Define the problem conditions
variable {b : ℝ}
noncomputable def hyperbola (x y : ℝ) := x^2 - (y^2 / b^2) = 1
variable {A B C : ℝ × ℝ}

-- The line with slope 1 passing through point A
noncomputable def line (P : ℝ × ℝ) := { Q : ℝ × ℝ // Q.2 = Q.1 - 1 }

-- The eccentricity of the hyperbola
noncomputable def eccentricity (a b : ℝ) := (a^2 + b^2).sqrt / a

-- Line intersects the asymptotes of the hyperbola at points B and C
noncomputable def asymptotes (P : ℝ × ℝ) := 
  Set (ℝ × ℝ) := { Q | Q.2 =  b * Q.1 ∨ Q.2 = -b * Q.1 }

-- Main theorem to prove
theorem hyperbola_eccentricity : 
  (∃ (A : ℝ × ℝ), line(A)) ∧ 
  (∃ B C ∈ asymptotes, True) ∧
  (2 * (B.1 - A.1, B.2 - A.2) = (C.1 - B.1, C.2 - B.2)) → 
  eccentricity 1 b = √5 :=
  by 
    sorry

end hyperbola_eccentricity_l760_760966


namespace total_inflation_time_l760_760571

theorem total_inflation_time (time_per_ball : ℕ) (alexia_balls : ℕ) (extra_balls : ℕ) : 
  time_per_ball = 20 → alexia_balls = 20 → extra_balls = 5 →
  (alexia_balls * time_per_ball) + ((alexia_balls + extra_balls) * time_per_ball) = 900 :=
by 
  intros h1 h2 h3
  sorry

end total_inflation_time_l760_760571


namespace minimum_operations_sisyphus_minimum_moves_sisyphus_impassible_l760_760685

open Nat

theorem minimum_operations (n : ℕ) (hn : n > 0) :
  ∀ k : ℕ, k ∈ {1, 2, ..., n} →
  (n : ℝ) / (k : ℝ) ≤ n / k :=
  sorry

theorem sisyphus_minimum_moves (n : ℕ) (hn : n > 0) :
  ∀ k : ℕ, k ∈ {0, ..., n} →
  ∃m, m ≥ ceil ((n : ℝ) / (k : ℝ))  :=
  sorry

theorem sisyphus_impassible (n : ℕ) (hn : n > 0) :
  ∑ i in (range (n+1)), 1 ≤ 
  (ceil ((n : ℝ) / (i : ℝ))) :=
begin
  sorry
end

end minimum_operations_sisyphus_minimum_moves_sisyphus_impassible_l760_760685


namespace sum_of_primes_less_than_twenty_is_77_l760_760058

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760058


namespace simplify_expression_l760_760344

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b - 4) - 2 * b^2 = 9 * b^3 + 4 * b^2 - 12 * b :=
by sorry

end simplify_expression_l760_760344


namespace greatest_two_digit_multiple_of_17_l760_760422

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760422


namespace hyperbola_eccentricity_l760_760722

-- Conditions
def parabola (p : ℝ) : (ℝ × ℝ) → Prop := λ (x y : ℝ), y^2 = 2 * p * x

def hyperbola (a b : ℝ) : (ℝ × ℝ) → Prop := λ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def focus (p : ℝ) : (ℝ × ℝ) := 
  let f := (p / 2, p)
in f

-- Proof Problem
theorem hyperbola_eccentricity (p a b e : ℝ) (h1 : p > 0)
  (h2 : ∀ (x y : ℝ), hyperbola a b (x, y) ∧ parabola p (x, y) → (x, y) = (p / 2, p))
  (h3 : ∀ (x y : ℝ), hyperbola a b (x, y) ∧ parabola p (x, y) → (x = p / 2))
  (h4 : focus p = (p / 2, p)) : e = sqrt 2 + 1 := by
  sorry

end hyperbola_eccentricity_l760_760722


namespace greatest_two_digit_multiple_of_17_l760_760504

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760504


namespace triangle_inradius_l760_760372

theorem triangle_inradius (p A : ℝ) (h1 : p = 24) (h2 : A = 30) : ∃ r : ℝ, r = 2.5 :=
by
  let s := p / 2
  have hs : s = 12 := by
    simp [h1]
  have hr : r := A / s
  have hr_eq : r = 2.5 := by
    simp [h2, hs]
  use r
  exact hr_eq

end triangle_inradius_l760_760372


namespace total_inflation_time_l760_760573

/-- 
  Assume a soccer ball takes 20 minutes to inflate.
  Alexia inflates 20 soccer balls.
  Ermias inflates 5 more balls than Alexia.
  Prove that the total time in minutes taken to inflate all the balls is 900 minutes.
-/
theorem total_inflation_time 
  (alexia_balls : ℕ) (ermias_balls : ℕ) (each_ball_time : ℕ)
  (h1 : alexia_balls = 20)
  (h2 : ermias_balls = alexia_balls + 5)
  (h3 : each_ball_time = 20) :
  (alexia_balls + ermias_balls) * each_ball_time = 900 :=
by
  sorry

end total_inflation_time_l760_760573


namespace sum_first_five_terms_l760_760894

def a (n : ℕ) : ℚ := 1 / (n * (n + 1))

def S (n : ℕ) : ℚ := ∑ k in finset.range (n + 1), a k.succ

theorem sum_first_five_terms : S 5 = 5 / 6 :=
by
  sorry

end sum_first_five_terms_l760_760894


namespace stratified_sampling_l760_760261

/-
In a class of 50 students, including 30 male students, a stratified sampling method is used to select 5 students to participate in a community service activity.
Prove that:
1. The number of male students selected is 3 and the number of female students selected is 2.
2. The probability that exactly one of the two selected students is a male student is 3/5.
-/

theorem stratified_sampling (total_students : ℕ) (male_students : ℕ) (selected_students : ℕ) 
  (selected_male_students : ℕ) (selected_female_students : ℕ)
  (probability_one_male : ℚ) 
  (h1 : total_students = 50) 
  (h2 : male_students = 30) 
  (h3 : selected_students = 5) 
  (h4 : selected_male_students = 3) 
  (h5 : selected_female_students = 2) 
  (h6 : probability_one_male = 3 / 5) :
  selected_male_students = 5 * male_students / total_students ∧
  selected_female_students = selected_students - selected_male_students ∧
  probability_one_male = 6 / 10 := 
begin
  sorry
end

end stratified_sampling_l760_760261


namespace min_norm_b_l760_760691

variable (a b : ℝ^n) -- Declare variables a and b as real vectors
variable [InnerProductSpace ℝ ℝ^n] -- Specify that we're working in an inner product space over ℝ

theorem min_norm_b (ha : ∥a∥ = 1) (hab : ⟪a, b⟫ = 1) : ∃ b, ∥b∥ = 1 := by
  sorry

end min_norm_b_l760_760691


namespace cookies_baked_l760_760784

noncomputable def total_cookies (irin ingrid nell : ℚ) (percentage_ingrid : ℚ) : ℚ :=
  let total_ratio := irin + ingrid + nell
  let proportion_ingrid := ingrid / total_ratio
  let total_cookies := ingrid / (percentage_ingrid / 100)
  total_cookies

theorem cookies_baked (h_ratio: 9.18 + 5.17 + 2.05 = 16.4)
                      (h_percentage : 31.524390243902438 = 31.524390243902438) : 
  total_cookies 9.18 5.17 2.05 31.524390243902438 = 52 :=
by
  -- Placeholder for the proof.
  sorry

end cookies_baked_l760_760784


namespace average_temperature_monday_to_thursday_l760_760854

variables (M T W Th F Sa Su : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (M + T + W + Th) / 4 = 48
def condition2 : Prop := (T + W + Th + F) / 4 = 46
def condition3 : Prop := M = 39
def condition4 : Prop := F = 31

-- Define the main theorem to be proven
theorem average_temperature_monday_to_thursday : condition1 → condition2 → condition3 → condition4 → (M + T + W + Th) / 4 = 48 :=
by
  intros
  sorry

end average_temperature_monday_to_thursday_l760_760854


namespace circle_radius_10_l760_760191

theorem circle_radius_10 (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0 → (x + 7) ^ 2 + (y + 4) ^ 2 = 100) ↔ (k = 35) :=
begin
  sorry
end

end circle_radius_10_l760_760191


namespace circle_intersection_l760_760189

theorem circle_intersection (m : ℝ) :
  (x^2 + y^2 - 2*m*x + m^2 - 4 = 0 ∧ x^2 + y^2 + 2*x - 4*m*y + 4*m^2 - 8 = 0) →
  (-12/5 < m ∧ m < -2/5) ∨ (0 < m ∧ m < 2) :=
by sorry

end circle_intersection_l760_760189


namespace last_integer_not_one_l760_760919

theorem last_integer_not_one : 
  (∃ f : ℕ → ℤ, (∀ n, (1 ≤ n + 2 → 1 ≤ f n ≤ 2011))
     → (∀ n, (f (n + 1) = (f n) - (f n)))
     → (∀ k, 0 ≤ k ≤ 2010 → ∃ first_second_diff (f : ℕ → ℤ), 
       f (k + 1) = f k - f first_second_diff))
     → (f 2010 ≠ 1)) :=
begin
  sorry
end

end last_integer_not_one_l760_760919


namespace value_of_expression_l760_760607

noncomputable def expression_value : ℝ :=
  (3^1 + 3^0 + 3^(-1)) / (3^(-2) + 3^(-3) + 3^(-4))

theorem value_of_expression :
  expression_value = 27 :=
by
  sorry

end value_of_expression_l760_760607


namespace distance_between_intersection_points_l760_760718

theorem distance_between_intersection_points :
  let l := λ x y : Real, 3 * x + y - 6 = 0
  let C := λ x y : Real, x^2 + y^2 - 2 * y - 4 = 0
  let A := (2, 0)
  let B := (1, 3)
  let distance := λ (p1 p2 : Real × Real), Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  l 2 0 ∧ l 1 3 ∧ C 2 0 ∧ C 1 3 →
  distance A B = Real.sqrt 10 :=
by
  sorry

end distance_between_intersection_points_l760_760718


namespace similar_triangles_perimeters_and_area_ratios_l760_760867

theorem similar_triangles_perimeters_and_area_ratios
  (m1 m2 : ℝ) (p_sum : ℝ) (ratio_p : ℝ) (ratio_a : ℝ) :
  m1 = 10 →
  m2 = 4 →
  p_sum = 140 →
  ratio_p = 5 / 2 →
  ratio_a = 25 / 4 →
  (∃ (p1 p2 : ℝ), p1 + p2 = p_sum ∧ p1 = (5 / 7) * p_sum ∧ p2 = (2 / 7) * p_sum ∧ ratio_a = (ratio_p)^2) :=
by
  sorry

end similar_triangles_perimeters_and_area_ratios_l760_760867


namespace find_constant_a_l760_760793

def f (a x : ℝ) : ℝ := (a * x) / (x + 2)

theorem find_constant_a (a : ℝ) : (∀ x : ℝ, x ≠ -2 → f a (f a x) = x) → a = -1 :=
by
  sorry

end find_constant_a_l760_760793


namespace greatest_two_digit_multiple_of_17_l760_760467

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760467


namespace number_of_8_step_paths_l760_760962

-- Given conditions
def is_white_square (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

def is_adjoining_white_square (i j i' j' : ℕ) : Prop :=
  i' = i + 1 ∧ (j' = j ∨ j' = j + 1 ∨ j' = j - 1) ∧ is_white_square i' j'

-- Predicate to determine if a path is valid
def valid_path (path : list (ℕ × ℕ)) : Prop :=
  path.length = 9 ∧
  is_white_square path.head.fst path.head.snd ∧  -- starting at a white square
  is_white_square path.last.fst path.last.snd ∧ -- ending at a white square
  ∀ (k : ℕ), k < 8 → is_adjoining_white_square (path.nth k).fst (path.nth k).snd (path.nth (k + 1)).fst (path.nth (k + 1)).snd

-- Theorem statement
theorem number_of_8_step_paths (P Q : ℕ × ℕ) :
  P = (0, 0) ∧ Q = (8, 8) → is_white_square P.fst P.snd ∧ is_white_square Q.fst Q.snd →
  ∃! (paths : list (list (ℕ × ℕ))), (∀ path ∈ paths, valid_path path ∧ path.head = P ∧ path.last = Q) ∧ paths.length = 182 :=
by
  sorry

end number_of_8_step_paths_l760_760962


namespace rhys_reyn_ratio_l760_760321

noncomputable def puzzle_problem : Prop :=
  ∃ (Reyn_pieces Rhys_pieces Rory_pieces : ℕ),
  let total_pieces := 300,
      pieces_per_son := total_pieces / 3,
      reyn_placed := 25,
      rory_placed := 3 * reyn_placed,
      total_left := 150,
      reyn_left := pieces_per_son - reyn_placed,
      rory_left := pieces_per_son - rory_placed in
  Reyn_pieces = reyn_placed ∧
  Rory_pieces = rory_placed ∧
  Reyn_pieces + Rory_pieces + Rhys_pieces = total_pieces - total_left ∧
  Rhys_pieces / Reyn_pieces = 2

theorem rhys_reyn_ratio : puzzle_problem := by
  sorry

end rhys_reyn_ratio_l760_760321


namespace tree_height_l760_760981

theorem tree_height (B h : ℕ) (H : ℕ) (h_eq : h = 16) (B_eq : B = 12) (L : ℕ) (L_def : L ^ 2 = B ^ 2 + h ^ 2) (H_def : H = h + L) :
    H = 36 := by
  -- We do not need to provide the proof steps as per the instructions
  sorry

end tree_height_l760_760981


namespace probability_event_A_probability_event_B_probability_event_C_l760_760959

-- Define the sample space for a die roll
def sample_space := {1, 2, 3, 4, 5, 6}

-- Event A: Getting an even number (2, 4, 6)
def event_A := {2, 4, 6}

-- Event B: Getting a number divisible by 3 (3, 6)
def event_B := {3, 6}

-- Event C: Getting any number except 5 (1, 2, 3, 4, 6)
def event_C := {1, 2, 3, 4, 6}

-- Probability of an event in a uniform probability space
def probability (event : Set ℕ) : ℚ :=
  event.to_finset.card / sample_space.to_finset.card

-- Theorem: Probabilities for events A, B, and C
theorem probability_event_A : probability event_A = 1 / 2 := sorry

theorem probability_event_B : probability event_B = 1 / 3 := sorry

theorem probability_event_C : probability event_C = 5 / 6 := sorry

end probability_event_A_probability_event_B_probability_event_C_l760_760959


namespace mixture_replacement_l760_760561

theorem mixture_replacement:
  ∀ (A B x : ℝ),
    A = 64 →
    B = A / 4 →
    (A - (4/5) * x) / (B + (4/5) * x) = 2 / 3 →
    x = 40 :=
by
  intros A B x hA hB hRatio
  sorry

end mixture_replacement_l760_760561


namespace c_divides_n_l760_760798

theorem c_divides_n (a b c n : ℤ) (h : a * n^2 + b * n + c = 0) : c ∣ n :=
sorry

end c_divides_n_l760_760798


namespace arithmetic_sequence_sum_l760_760873

theorem arithmetic_sequence_sum (c d : ℤ) (h1 : c = 24) (h2 : d = 31) :
  c + d = 55 :=
by
  rw [h1, h2]
  exact rfl

end arithmetic_sequence_sum_l760_760873


namespace distance_Atlanta_NewYork_l760_760823

-- Definition of coordinates given in the problem
def NewYork : ℂ := 0
def Miami : ℂ := 0 + 3000 * Complex.I
def Atlanta : ℂ := 900 + 1200 * Complex.I

-- The theorem statement we are trying to prove
theorem distance_Atlanta_NewYork : Complex.abs (Atlanta - NewYork) = 1500 :=
by
  -- Skip the proof for now by using sorry
  sorry

end distance_Atlanta_NewYork_l760_760823


namespace factorial_sum_mod_30_l760_760165

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map factorial |>.sum

def remainder_when_divided_by (m k : Nat) : Nat :=
  m % k

theorem factorial_sum_mod_30 : remainder_when_divided_by (sum_of_factorials 100) 30 = 3 :=
by
  sorry

end factorial_sum_mod_30_l760_760165


namespace inequalities_not_equivalent_l760_760144

theorem inequalities_not_equivalent :
  let domain1 := {x : ℝ | x ≠ 2 ∧ x ≠ 3}
  let ineq1 := λ x : ℝ, x ∈ domain1 → (x-3)/(x^2 - 5*x + 6) < 2
  let ineq2 := λ x : ℝ, 2*x^2 - 11*x + 15 > 0
  ¬ ∀ x : ℝ, ineq1 x ↔ ineq2 x :=
sorry

end inequalities_not_equivalent_l760_760144


namespace depth_of_melted_ice_cream_l760_760131

noncomputable def sphereVolume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (r ^ 3)

noncomputable def cylinderVolume (r h : ℝ) : ℝ :=
  Real.pi * (r ^ 2) * h

theorem depth_of_melted_ice_cream :
  let r1 := 3 -- radius of the sphere
      r2 := 12 -- radius of the cylinder
      V1 := sphereVolume r1 -- volume of the sphere
      h := (1 / 4 : ℝ) -- depth of the melted ice cream
      V2 := cylinderVolume r2 h -- volume of the cylinder
  in V1 = V2 :=
by
  sorry

end depth_of_melted_ice_cream_l760_760131


namespace solve_for_x_l760_760186

theorem solve_for_x (x : ℕ) (h : x + 1 = 4) : x = 3 :=
by
  sorry

end solve_for_x_l760_760186


namespace correlation_function_of_Z_l760_760653

noncomputable def correlation_function_kz (Kx : ℝ → ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
  Kx t1 t2 +
  (deriv (λ t2, deriv (λ t1, Kx t1 t2) t1) t2) +
  (deriv (λ t2, Kx t1 t2) t2) +
  (deriv (λ t1, Kx t1 t2) t1)

theorem correlation_function_of_Z (Kx : ℝ → ℝ → ℝ) :
  ∀ t1 t2, correlation_function_kz Kx t1 t2 = 
  Kx t1 t2 + 
  (deriv (λ t2, deriv (λ t1, Kx t1 t2) t1) t2) +
  (deriv (λ t2, Kx t1 t2) t2) +
  (deriv (λ t1, Kx t1 t2) t1) :=
sorry

end correlation_function_of_Z_l760_760653


namespace interior_diagonals_sum_l760_760567

-- Define the variables representing the side lengths of the box.
variable (a b c : ℝ)

-- Condition 1: Total edge length is given.
def edge_sum (a b c : ℝ) : Prop := a + b + c = 15

-- Condition 2: Total surface area is given.
def surface_area (a b c : ℝ) : Prop := ab + bc + ca = 65  -- Note: 130 / 2 = 65

-- Theorem: The sum of the lengths of all interior diagonals.
theorem interior_diagonals_sum (a b c : ℝ) (h1 : edge_sum a b c) (h2 : surface_area a b c) : 4 * Real.sqrt (95) = 4 * Real.sqrt (a^2 + b^2 + c^2) :=
by
  -- Sorry for skipping the proof as requested.
  sorry

end interior_diagonals_sum_l760_760567


namespace sum_primes_less_than_20_l760_760079

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760079


namespace sum_of_primes_less_than_twenty_is_77_l760_760055

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760055


namespace greatest_two_digit_multiple_of_17_is_85_l760_760460

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760460


namespace value_of_a_plus_b_l760_760700

open Set Real

def setA : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def setB (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}
def universalSet : Set ℝ := univ

theorem value_of_a_plus_b (a b : ℝ) :
  (setA ∪ setB a b = universalSet) ∧ (setA ∩ setB a b = {x : ℝ | 3 < x ∧ x ≤ 4}) → a + b = -7 :=
by
  sorry

end value_of_a_plus_b_l760_760700


namespace roots_sum_ln_abs_l760_760157

noncomputable def roots_sum (m : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, ln (abs (x1 - 2)) = m ∧ ln (abs (x2 - 2)) = m → x1 + x2 = 4

-- Now let's state the theorem to be proved
theorem roots_sum_ln_abs (m : ℝ) : roots_sum m :=
sorry

end roots_sum_ln_abs_l760_760157


namespace final_selling_price_l760_760569

theorem final_selling_price (CP_A : ℕ) (P_A P_B : ℕ) (SP_C : ℕ) 
  (h1 : CP_A = 120) 
  (h2 : P_A = 25) 
  (h3 : P_B = 50) 
  (h4 : SP_C = 225) :
  let Profit_A := P_A * CP_A / 100,
      SP_A := CP_A + Profit_A,
      Profit_B := P_B * SP_A / 100,
      SP_B := SP_A + Profit_B
  in 
  SP_B = SP_C :=
by
  sorry

end final_selling_price_l760_760569


namespace grown_ups_in_milburg_l760_760897

def total_population : ℕ := 8243
def number_of_children : ℕ := 2987

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 :=
by {
  sorry
}

end grown_ups_in_milburg_l760_760897


namespace machine_a_sprockets_per_hour_l760_760936

theorem machine_a_sprockets_per_hour (A B : ℝ) (T : ℝ) 
    (h1 : B = 1.10 * A)
    (h2 : 220 = A * (T + 10)) 
    (h3 : 220 = B * T) :
    A = 2 :=
begin
  sorry
end

end machine_a_sprockets_per_hour_l760_760936


namespace sqrt_abc_sums_l760_760309

-- Defining the variables and conditions
variable (a b c : ℝ)
axiom h1 : b + c = 17
axiom h2 : c + a = 18
axiom h3 : a + b = 19

-- Defining the goal
theorem sqrt_abc_sums :
  sqrt (a * b * c * (a + b + c)) = 60 * sqrt 10 :=
sorry

end sqrt_abc_sums_l760_760309


namespace find_a_100_l760_760688

noncomputable def a_n : ℕ → ℕ
| 1 => 1
| n+1 => 4 * ∑ i in Finset.range (n+1), a_n i - (a_n n * a_n (n-1)) / 4

theorem find_a_100 :
  ∃ a_n : ℕ → ℕ, (∀ n : ℕ, a_n n ≠ 0)
  ∧ (∀ n : ℕ, ∑ i in Finset.range (n+1), a_n i = 4 * a_n (n + 1) * a_n n + 1)
  ∧ (a_n 1 = 1)
  ∧ (a_n 100 = 199) :=
by {
  sorry
}

end find_a_100_l760_760688


namespace total_players_l760_760262

/-- In a group of players, some play outdoor games, some play indoor games, and some play both.
- 350 players play outdoor games.
- 110 players play indoor games.
- 60 players play both outdoor and indoor games.
Prove the total number of players in the group is 400.
-/
theorem total_players (outdoor_players : ℕ) (indoor_players : ℕ) (both_players : ℕ) :
  outdoor_players = 350 →
  indoor_players = 110 →
  both_players = 60 →
  (outdoor_players + indoor_players - both_players) = 400 :=
by
  intros h_outdoor h_indoor h_both
  rw [h_outdoor, h_indoor, h_both]
  norm_num

end total_players_l760_760262


namespace find_d_from_factor_condition_l760_760741

theorem find_d_from_factor_condition (d : ℚ) : (∀ x, x = 5 → d * x^4 + 13 * x^3 - 2 * d * x^2 - 58 * x + 65 = 0) → d = -28 / 23 :=
by
  intro h
  sorry

end find_d_from_factor_condition_l760_760741


namespace probability_of_three_black_balls_l760_760109

def total_ball_count : ℕ := 4 + 8

def white_ball_count : ℕ := 4

def black_ball_count : ℕ := 8

def total_combinations : ℕ := Nat.choose total_ball_count 3

def black_combinations : ℕ := Nat.choose black_ball_count 3

def probability_three_black : ℚ := black_combinations / total_combinations

theorem probability_of_three_black_balls : 
  probability_three_black = 14 / 55 := 
sorry

end probability_of_three_black_balls_l760_760109


namespace tangent_line_eqn_monotonic_intervals_range_of_a_l760_760815

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x + 1
noncomputable def F (a x : ℝ) : ℝ := f x - g a x

theorem tangent_line_eqn :
  tangent_line f (e : ℝ) = (fun x => (1 / e) * x) := sorry

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, F a x > 0 ∧ monotone_on (F a) (Set.Ioi 0)) ∧ 
  (a > 0 → 
    (∀ x > 0, x < 1 / a → F a x > 0) ∧ 
    (∀ x > 1 / a, F a x < 0)) := sorry

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x > 0, F a x ≠ 0) → (a > Real.exp (-2)) := sorry

end tangent_line_eqn_monotonic_intervals_range_of_a_l760_760815


namespace sum_of_primes_less_than_twenty_is_77_l760_760053

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760053


namespace divide_7_students_into_groups_and_assign_l760_760625

theorem divide_7_students_into_groups_and_assign (students : Finset (Fin 7)) :
  ∃ (group1 group2 group3 : Finset (Fin 7)),
  group1.card = 3 ∧ group2.card = 2 ∧ group3.card = 2 ∧
  group1 ∪ group2 ∪ group3 = students ∧
  group1 ∩ group2 = ∅ ∧ group2 ∩ group3 = ∅ ∧ group1 ∩ group3 = ∅ ∧
  (∃ (places : Finset (Fin 3)), places.card = 3) →
  (choose 7 3) * (choose 4 2) * (choose 2 2) * (factorial 3) / (factorial 2) = 630 :=
by
  sorry

end divide_7_students_into_groups_and_assign_l760_760625


namespace difference_between_3_and_2_painted_faces_is_12_l760_760146

-- Definitions for conditions
variable (model : Type) [Finite model] [HasVolume model] [PaintedRedSurface model]

-- Define small cubes and painting properties
variable (cubes : model → ℕ) (painted_faces : ∀ (c : model), ℕ)

-- Define the counts of cubes with 2 and 3 faces painted red
def count_cubes_with_painted_faces (num_faces : ℕ) (m : model) :=
  cubes m |>.filter (λ c, painted_faces c = num_faces) |>.sum

noncomputable def count_3_faces_painted (m : model) := count_cubes_with_painted_faces 3 m
noncomputable def count_2_faces_painted (m : model) := count_cubes_with_painted_faces 2 m

-- The question to be proved is that the difference is 12
theorem difference_between_3_and_2_painted_faces_is_12 (m : model) :
  count_3_faces_painted m - count_2_faces_painted m = 12 :=
by sorry

end difference_between_3_and_2_painted_faces_is_12_l760_760146


namespace part_a_part_b_l760_760721

-- Define the parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Prove that (1, 1) lies on the parabola
theorem part_a : parabola 1 = 1 := by
  sorry

-- Prove that for any t, (t, t^2) lies on the parabola
theorem part_b (t : ℝ) : parabola t = t^2 := by
  sorry

end part_a_part_b_l760_760721


namespace garden_path_width_l760_760117

theorem garden_path_width (R r : ℝ) (h : 2 * Real.pi * R - 2 * Real.pi * r = 20 * Real.pi) : R - r = 10 :=
by
  sorry

end garden_path_width_l760_760117


namespace area_inside_S_but_outside_R_l760_760824

noncomputable def side_length_hexagon := 1
noncomputable def side_length_triangle := 1
noncomputable def hexagon_area := (3 * Real.sqrt 3) / 2 * side_length_hexagon ^ 2
noncomputable def triangle_area := (Real.sqrt 3) / 4 * side_length_triangle ^ 2
noncomputable def total_triangle_area := 18 * triangle_area
noncomputable def area_R := hexagon_area + total_triangle_area
noncomputable def side_length_S := side_length_hexagon + side_length_triangle
noncomputable def area_S := (3 * Real.sqrt 3) / 2 * side_length_S ^ 2

theorem area_inside_S_but_outside_R : (area_S - area_R) = 0 :=
by
  have hexagon_area_def : hexagon_area = (3 * Real.sqrt 3) / 2 * 1 ^ 2 := rfl
  have triangle_area_def : triangle_area = (Real.sqrt 3) / 4 * 1 ^ 2 := rfl
  have total_triangle_area_def : total_triangle_area = 18 * triangle_area := rfl
  have area_R_def : area_R = hexagon_area + total_triangle_area := rfl
  have side_length_S_def : side_length_S = side_length_hexagon + side_length_triangle := rfl
  have area_S_def : area_S = (3 * Real.sqrt 3) / 2 * 2 ^ 2 := rfl
  sorry

end area_inside_S_but_outside_R_l760_760824


namespace percent_gain_proof_l760_760960

variable (cost_per_sheep price_sold_first750 price_sold_remaining50 total_revenue profit percent_gain : ℝ)

-- Define the initial conditions
def initial_conditions : Prop := 
  cost_per_sheep > 0 ∧
  price_sold_first750 = (800 * cost_per_sheep) / 750 ∧
  price_sold_remaining50 = 1.1 * price_sold_first750 ∧
  total_revenue = (750 * price_sold_first750) + (50 * price_sold_remaining50) ∧
  profit = total_revenue - (800 * cost_per_sheep) ∧
  percent_gain = (profit / (800 * cost_per_sheep)) * 100

-- Prove the percentage gain
theorem percent_gain_proof (h : initial_conditions) : percent_gain = 7.334 :=
  sorry

end percent_gain_proof_l760_760960


namespace greatest_two_digit_multiple_of_17_l760_760452

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760452


namespace sum_primes_less_than_20_l760_760062

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760062


namespace marthas_bedroom_size_l760_760889

-- Define the variables and conditions
def total_square_footage := 300
def additional_square_footage := 60
def Martha := 120
def Jenny := Martha + additional_square_footage

-- The main theorem stating the requirement 
theorem marthas_bedroom_size : (Martha + (Martha + additional_square_footage) = total_square_footage) -> Martha = 120 :=
by 
  sorry

end marthas_bedroom_size_l760_760889


namespace geometric_S_sum_inequality_l760_760147

-- Define the arithmetic sequence with a common difference d.
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- S_n is the sum of 2^(n-1) consecutive terms starting from the (2^(n-1))th term.
def S (a d : ℕ) (n : ℕ) : ℕ :=
  ∑ k in finset.range (2^(n-1)), arithmetic_sequence a d (2^(n-1) + k.val)

-- Conditions: a_1 = 15/4 and d > 0
def a1 : ℚ := 15 / 4
axiom d_pos : ℚ → Prop

-- Goal 1: Prove sequence {S_n} is geometric if {S_1, S_2, S_3} is geometric
theorem geometric_S (d : ℕ) (h1 : d_pos d) : (S a1 d 1) * (S a1 d 3) = (S a1 d 2) * (S a1 d 2) → 
  ∀ n : ℕ, (S a1 d (n+1)) / (S a1 d n) = 4 := sorry

-- Goal 2: Prove the sum inequality
theorem sum_inequality (d : ℕ) (h1 : d > 0) (n : ℕ) : 
  ∑ k in finset.range n, 1 / (S a1 d (k+1)) ≤ (8 / (9 * d)) * (1 / 2 - 1 / (4^n + 1)) := sorry

end geometric_S_sum_inequality_l760_760147


namespace greatest_two_digit_multiple_of_17_l760_760448

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760448


namespace greatest_two_digit_multiple_of_17_l760_760404

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760404


namespace permutations_count_l760_760590

-- Define the set of digits
def digits : Set ℕ := {5, 3, 1}

-- Define the condition that the size of the set is 3
def digits_size (s : Set ℕ) : Prop := s.size = 3

-- Lean statement to prove the number of permutations
theorem permutations_count : digits_size digits → ∃ n, n = 6 :=
by
  intro h
  use 6
  -- Proof steps would go here 
  sorry

end permutations_count_l760_760590


namespace find_multiple_of_johns_age_l760_760761

theorem find_multiple_of_johns_age : ∃ m : ℕ, 21 - 11 = 10 ∧ 21 + 9 = 30 ∧ m * 10 = 30 :=
by
  use 3
  split
  · exact rfl
  split
  · exact rfl
  · exact rfl

end find_multiple_of_johns_age_l760_760761


namespace seats_taken_l760_760666

variable (num_rows : ℕ) (chairs_per_row : ℕ) (unoccupied_chairs : ℕ)

theorem seats_taken (h1 : num_rows = 40) (h2 : chairs_per_row = 20) (h3 : unoccupied_chairs = 10) :
  num_rows * chairs_per_row - unoccupied_chairs = 790 :=
sorry

end seats_taken_l760_760666


namespace heptagon_perpendicular_l760_760858

noncomputable theory

-- Define the convex regular heptagon and its properties
def is_regular_heptagon (C : Fin 7 → ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  ∀ i j, dist (C i) (C j) = dist (C (i + 1) % 7) (C ((j + 1) % 7)) ∧ dist O (C i) = dist O (C j)

-- Define the condition for diagonals intersection
def intersect_diagonals (C : Fin 7 → ℝ × ℝ) (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  ∃ M : ℝ × ℝ, (∃ k : ℝ, M = k • (p2 - p1) + p1) ∧ (∃ l : ℝ, M = l • (p4 - p3) + p3)

-- Define perpendicular lines
def are_perpendicular (A B C D : ℝ × ℝ) : Prop :=
  let v1 := B - A in
  let v2 := D - C in
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Main theorem statement
theorem heptagon_perpendicular
  (C : Fin 7 → ℝ × ℝ)
  (O M N : ℝ × ℝ)
  (h1 : is_regular_heptagon C O)
  (h2 : intersect_diagonals C (C 0) (C 2) (C 1) (C 4) M)
  (h3 : intersect_diagonals C (C 0) (C 3) (C 1) (C 5) N) :
  are_perpendicular O (C 0) M N :=
  sorry  -- Proof is omitted

end heptagon_perpendicular_l760_760858


namespace sum_is_eighteen_or_twentyseven_l760_760103

theorem sum_is_eighteen_or_twentyseven :
  ∀ (A B C D E I J K L M : ℕ),
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ I ∧ A ≠ J ∧ A ≠ K ∧ A ≠ L ∧ A ≠ M ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ I ∧ B ≠ J ∧ B ≠ K ∧ B ≠ L ∧ B ≠ M ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ I ∧ C ≠ J ∧ C ≠ K ∧ C ≠ L ∧ C ≠ M ∧
  D ≠ E ∧ D ≠ I ∧ D ≠ J ∧ D ≠ K ∧ D ≠ L ∧ D ≠ M ∧
  E ≠ I ∧ E ≠ J ∧ E ≠ K ∧ E ≠ L ∧ E ≠ M ∧
  I ≠ J ∧ I ≠ K ∧ I ≠ L ∧ I ≠ M ∧
  J ≠ K ∧ J ≠ L ∧ J ≠ M ∧
  K ≠ L ∧ K ≠ M ∧
  L ≠ M ∧
  (0 < I) ∧ (0 < J) ∧ (0 < K) ∧ (0 < L) ∧ (0 < M) ∧
  A + B + C + D + E + I + J + K + L + M = 45 ∧
  (I + J + K + L + M) % 10 = 0 →
  A + B + C + D + E + (I + J + K + L + M) / 10 = 18 ∨
  A + B + C + D + E + (I + J + K + L + M) / 10 = 27 :=
by
  intros
  sorry

end sum_is_eighteen_or_twentyseven_l760_760103


namespace greatest_two_digit_multiple_of_17_l760_760444

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760444


namespace greatest_two_digit_multiple_of_17_l760_760468

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760468


namespace max_f_value_l760_760674

def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := (1 / 2) * x^2 + b / x + c
def g (x : ℝ) : ℝ := (1 / 4) * x + 1 / x
def M : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

theorem max_f_value (b c : ℝ) (h : ∀ x ∈ M, ∃ x₀ ∈ M, f x b c ≥ f x₀ b c ∧ g x ≥ g x₀ ∧ f x₀ b c = g x₀) :
  ∃ xₘ ∈ M, ∀ x ∈ M, f x b c ≤ 5 :=
sorry

end max_f_value_l760_760674


namespace greatest_two_digit_multiple_of_17_l760_760427

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760427


namespace air_conditioners_total_consumption_l760_760907

-- Definitions of power consumers per hour
def power_A := 7.2 / 8
def power_B := 9.6 / 10
def power_C := 12 / 12

-- Usage durations
def hours_A := 6 * 5
def hours_B := 4 * 7
def hours_C := 3 * 10

-- Total consumptions
def total_consumption_A := power_A * hours_A
def total_consumption_B := power_B * hours_B
def total_consumption_C := power_C * hours_C

-- Total consumption of all air conditioners
def total_consumption := total_consumption_A + total_consumption_B + total_consumption_C

-- Theorem statement
theorem air_conditioners_total_consumption : total_consumption = 83.88 :=
by
  sorry

end air_conditioners_total_consumption_l760_760907


namespace change_in_energy_from_A_to_B_l760_760361

noncomputable def electric_field (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (-x, -y, -z)

def charge : ℝ := sorry  -- assume q is given

def change_in_energy (q : ℝ) (A B : ℝ × ℝ × ℝ) : ℝ :=
  q * (1 / 2) * ((B.1^2 + B.2^2 + B.3^2) - (A.1^2 + A.2^2 + A.3^2))

theorem change_in_energy_from_A_to_B :
  ∀ q : ℝ, let A := (0, 0, 0 : ℝ × ℝ × ℝ),
               B := (1, 1, 1 : ℝ × ℝ × ℝ) in
  change_in_energy q A B = (3 * q) / 2 :=
by sorry

end change_in_energy_from_A_to_B_l760_760361


namespace train_length_l760_760980

-- Define speed in km/hr
def speed_kmph : ℝ := 120

-- Define time in seconds
def time_sec : ℝ := 15

-- Conversion factor from km/hr to m/s
def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

-- Calculate speed in m/s
def speed_mps : ℝ := kmph_to_mps speed_kmph

-- Calculate distance using the formula: distance = speed * time
def distance (v : ℝ) (t : ℝ) : ℝ := v * t

-- Given conditions and expected answer
theorem train_length : distance speed_mps time_sec ≈ 500 := 
by
  -- Here, you would provide the formal proof steps.
  sorry

end train_length_l760_760980


namespace problem_AP_eq_PL_l760_760946

-- Define the conditions and the corresponding intersection point
variables {A B C L K P : Type*}

-- Define the geometrical configuration and conditions
variables (ABC : Triangle A B C)
variables (AL_bisects_ABC : AngleBisector A L B C)
variables (K_on_AC : K ∈ Side A C)
variables (CK_eq_CL : Length C K = Length C L)
variables (P_intersection : ∃ P, Intersection (Line K L) (AngleBisector B A C) = P)

-- State the final goal
theorem problem_AP_eq_PL (ABC : Triangle A B C)
  (AL_bisects_ABC : AngleBisector A L B C)
  (K_on_AC : K ∈ Side A C)
  (CK_eq_CL : Length C K = Length C L)
  (P_intersection : ∃ P, Intersection (Line K L) (AngleBisector B A C) = P) :
  Length A P = Length P L :=
by
  sorry

end problem_AP_eq_PL_l760_760946


namespace mary_needs_to_add_l760_760319

-- Define the conditions
def total_flour_required : ℕ := 7
def flour_already_added : ℕ := 2

-- Define the statement that corresponds to the mathematical equivalent proof problem
theorem mary_needs_to_add :
  total_flour_required - flour_already_added = 5 :=
by
  sorry

end mary_needs_to_add_l760_760319


namespace solve_arithmetic_sequence_sum_l760_760874

noncomputable def arithmetic_sequence_sum : ℕ :=
  let a : ℕ := 3
  let b : ℕ := 10
  let c : ℕ := 17
  let e : ℕ := 32
  let d := b - a
  let c_term := c + d
  let d_term := c_term + d
  c_term + d_term

theorem solve_arithmetic_sequence_sum : arithmetic_sequence_sum = 55 :=
by
  sorry

end solve_arithmetic_sequence_sum_l760_760874


namespace permutation_triple_count_correct_l760_760153

open Function

def permutation := equiv.perm (fin 5)

noncomputable def count_valid_triples : ℕ :=
  sorry -- here would be the actual computation of the count which assumes the techniques in the solution are implemented

theorem permutation_triple_count_correct :
  count_valid_triples = 146 :=
sorry

end permutation_triple_count_correct_l760_760153


namespace baron_munchausen_is_bragging_l760_760599

noncomputable def a_sequence : ℕ → ℕ := sorry
noncomputable def b_sequence (a : ℕ → ℕ) : ℕ → ℕ
| 0       := 1
| (n + 1) := a (b_sequence n)

theorem baron_munchausen_is_bragging (a : ℕ → ℕ) :
  (∀ n, a n ≠ 1) →
  (∀ m n, m ≠ n → a m ≠ a n) →
  (∃ N, ∀ n > N, a n > n) :=
sorry

end baron_munchausen_is_bragging_l760_760599


namespace solve_eq1_solve_eq2_l760_760844

noncomputable def eq1_solution1 := -2 + Real.sqrt 5
noncomputable def eq1_solution2 := -2 - Real.sqrt 5

noncomputable def eq2_solution1 := 3
noncomputable def eq2_solution2 := 1

theorem solve_eq1 (x : ℝ) :
  x^2 + 4 * x - 1 = 0 → (x = eq1_solution1 ∨ x = eq1_solution2) :=
by
  sorry

theorem solve_eq2 (x : ℝ) :
  (x - 3)^2 + 2 * x * (x - 3) = 0 → (x = eq2_solution1 ∨ x = eq2_solution2) :=
by 
  sorry

end solve_eq1_solve_eq2_l760_760844


namespace sum_primes_less_than_20_l760_760060

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760060


namespace greatest_two_digit_multiple_of_17_l760_760519

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760519


namespace total_worksheets_to_grade_l760_760978

def initial_worksheets : ℕ := 6
def graded_worksheets : ℕ := 4
def additional_worksheets : ℕ := 18

theorem total_worksheets_to_grade :
  let remaining_worksheets := initial_worksheets - graded_worksheets in
  let after_additional := remaining_worksheets + additional_worksheets in
  let twice_remaining := 2 * after_additional in
  after_additional + twice_remaining = 60 :=
by
  sorry

end total_worksheets_to_grade_l760_760978


namespace sum_primes_less_than_20_l760_760063

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760063


namespace flowers_to_embroider_l760_760150

-- Defining constants based on the problem conditions
def stitches_per_minute : ℕ := 4
def stitches_per_flower : ℕ := 60
def stitches_per_unicorn : ℕ := 180
def stitches_per_godzilla : ℕ := 800
def num_unicorns : ℕ := 3
def num_godzillas : ℕ := 1
def total_minutes : ℕ := 1085

-- Theorem statement to prove the number of flowers Carolyn wants to embroider
theorem flowers_to_embroider : 
  (total_minutes * stitches_per_minute - (num_godzillas * stitches_per_godzilla + num_unicorns * stitches_per_unicorn)) / stitches_per_flower = 50 :=
by
  sorry

end flowers_to_embroider_l760_760150


namespace sum_of_primes_less_than_20_l760_760002

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760002


namespace infinite_primes_dividing_polynomial_l760_760805

theorem infinite_primes_dividing_polynomial (P : Polynomial ℤ) (hP_nonconstant: P.degree ≥ 1) : 
  ∃ᶠ p in Filter.atTop, ∃ x : ℕ, p ∣ P.eval x := sorry

end infinite_primes_dividing_polynomial_l760_760805


namespace analyze_expression_l760_760802

-- Definitions of the conditions
variables {x y z : ℝ} (r : ℝ := real.sqrt (x^2 + y^2 + z^2))
def s := y / r
def c := x / r
def z_r := z / r

-- The theorem we want to prove
theorem analyze_expression : s^2 - c^2 + z_r^2 = 1 :=
sorry

end analyze_expression_l760_760802


namespace all_rational_segments_are_rational_l760_760145

def AB : ℚ := sorry -- length of AB
def AD : ℚ := sorry -- length of AD
def BD : ℚ := AB - AD -- length of BD inferred as rational
def CD : ℚ := sorry -- length of CD
def O : Point := sorry -- Center of the circle
def midpoint (A B : Point) := (A + B) / 2
def C := midpoint A B -- C is the midpoint
def OC : ℚ := AD / 2 -- OC is half of AB, thus rational
def OD : ℚ := real.sqrt(OC^2 - CD^2) -- OD calculated, thus rational
def OE : ℚ := OD^2 / OC -- OE calculated, thus rational
def DE : ℚ := real.sqrt(CD^2 - (OC - OE)^2) -- DE calculated, thus rational
def CE : ℚ := OC - OE -- CE calculated, thus rational

theorem all_rational_segments_are_rational (AD BD CD: ℚ): 
  let OA := AB / 2,
      OB := AB / 2,
      OC := OA in
  rational OE ∧ rational DE ∧ 
  (rational AD ∧ rational BD ∧ rational CD ∧ 
  rational AB ∧ rational OA ∧ rational OB ∧ 
  rational OC ∧ rational OD ∧ rational OE ∧ 
  rational DE ∧ rational CE) :=
by
  sorry

end all_rational_segments_are_rational_l760_760145


namespace sampling_method_is_stratified_l760_760120

/-- There are 500 boys and 400 girls in the high school senior year.
The total population consists of 900 students.
A random sample of 25 boys and 20 girls was taken.
Prove that the sampling method used is stratified sampling method. -/
theorem sampling_method_is_stratified :
    let boys := 500
    let girls := 400
    let total_students := 900
    let sample_boys := 25
    let sample_girls := 20
    let sampling_method := "Stratified sampling"
    sample_boys < boys ∧ sample_girls < girls → sampling_method = "Stratified sampling"
:=
sorry

end sampling_method_is_stratified_l760_760120


namespace inequality_proof_l760_760836

variable {n : ℕ} (x : Fin n → ℝ)

theorem inequality_proof (h : ∀ i, 0 < x i) :
  x 0 + ∑ i in Finset.range (n - 1), (i + 2) * x i.succ ≤
    ∑ i in Finset.range (n - 1), x (i.succ).succ ^ (i + 2) + ∑ i in Finset.range (n - 1), (i + 2) +
    x 0 :=
by
  sorry

end inequality_proof_l760_760836


namespace new_bill_amount_l760_760245

def original_order : ℝ := 25
def tomato_cost_old : ℝ := 0.99
def tomato_cost_new : ℝ := 2.20
def lettuce_cost_old : ℝ := 1.00
def lettuce_cost_new : ℝ := 1.75
def celery_cost_old : ℝ := 1.96
def celery_cost_new : ℝ := 2.00
def delivery_and_tip : ℝ := 8.00

theorem new_bill_amount :
  original_order + (tomato_cost_new - tomato_cost_old) + 
  (lettuce_cost_new - lettuce_cost_old) + 
  (celery_cost_new - celery_cost_old) + delivery_and_tip = 35 := 
by
  calc
    original_order + (tomato_cost_new - tomato_cost_old) + 
    (lettuce_cost_new - lettuce_cost_old) + 
    (celery_cost_new - celery_cost_old) + delivery_and_tip
    = 25 + (2.20 - 0.99) + (1.75 - 1.00) + (2.00 - 1.96) + 8.00 : by sorry -- intermediate step for clarity
    ... = 25 + 1.21 + 0.75 + 0.04 + 8.00 : by sorry
    ... = 25 + 2.00 + 8.00 : by sorry
    ... = 25 + 10.00 : by sorry
    ... = 35 : by sorry

end new_bill_amount_l760_760245


namespace silver_nitrate_sodium_hydroxide_reaction_l760_760239

theorem silver_nitrate_sodium_hydroxide_reaction :
  ∀ (AgNO₃ NaOH AgOH NaNO₃ : ℕ),
  (AgNO₃ + NaOH = AgOH + NaNO₃) →
  (AgOH = 3) →
  (NaNO₃ = 3) →
  (AgNO₃ = 3) ∧ (NaOH = 3) :=
by
  intros AgNO₃ NaOH AgOH NaNO₃ h_eq h_AgOH h_NaNO₃
  split
  all_goals {
    sorry
  }

end silver_nitrate_sodium_hydroxide_reaction_l760_760239


namespace units_place_3_pow_34_l760_760929

theorem units_place_3_pow_34 : (3^34 % 10) = 9 :=
by
  sorry

end units_place_3_pow_34_l760_760929


namespace sum_primes_less_than_20_l760_760065

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760065


namespace add_to_perfect_square_l760_760091

theorem add_to_perfect_square (n : ℕ) (h : n = 1296) : ∃ k : ℕ, k^2 - n = 73 :=
by {
    use 37,                -- We claim that the next perfect square's root is 37
    rw h,                  -- Substitute n with 1296
    norm_num,              -- Carry out the calculation
    sorry,                 -- Placeholder for proof
}

end add_to_perfect_square_l760_760091


namespace student_correct_answers_l760_760770

theorem student_correct_answers (C W : ℕ) (h₁ : C + W = 50) (h₂ : 4 * C - W = 130) : C = 36 := 
by
  sorry

end student_correct_answers_l760_760770


namespace derivative_y_at_1_l760_760859

-- Define the function y = x^2 + 2
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the proposition that the derivative at x=1 is 2
theorem derivative_y_at_1 : deriv f 1 = 2 :=
by sorry

end derivative_y_at_1_l760_760859


namespace cos_product_equals_one_eighth_l760_760603

noncomputable def cos_pi_over_9 := Real.cos (Real.pi / 9)
noncomputable def cos_2pi_over_9 := Real.cos (2 * Real.pi / 9)
noncomputable def cos_4pi_over_9 := Real.cos (4 * Real.pi / 9)

theorem cos_product_equals_one_eighth :
  cos_pi_over_9 * cos_2pi_over_9 * cos_4pi_over_9 = 1 / 8 := 
sorry

end cos_product_equals_one_eighth_l760_760603


namespace smallest_class_size_l760_760766

theorem smallest_class_size (n : ℕ) (x : ℕ) (h1 : n > 50) (h2 : n = 4 * x + 2) : n = 54 :=
by
  sorry

end smallest_class_size_l760_760766


namespace digit_subtraction_l760_760231

theorem digit_subtraction (a b c : ℕ) (h1 : c - a = 7) (h2 : b = b) (h3 : a = 2) (h4 : c = 9) : 
  let minuend := 100 * a + 10 * b + c,
      subtrahend := 100 * c + 10 * b + a,
      result := minuend - subtrahend 
  in result = 307 :=
by
  let minuend := 100 * 2 + 10 * b + 9
  let subtrahend := 100 * 9 + 10 * b + 2
  let result := minuend - subtrahend
  have : result = 307 := by
    simp [minuend, subtrahend, result]
    sorry
  exact this

end digit_subtraction_l760_760231


namespace least_k_is_576_l760_760122

noncomputable def find_least_k : ℕ :=
  Inf { k : ℕ | ∃ (q r : ℤ), k = 23 * q ∧ k = 48 * r ∧ q = r + 13 }

theorem least_k_is_576 : find_least_k = 576 :=
by
  sorry

end least_k_is_576_l760_760122


namespace log_sum_equality_l760_760167

open Real (log)

def factorial (n : ℕ) : ℝ :=
  if n = 0 then 1 else (finset.range n).prod (λ i, (i+1 : ℝ))

theorem log_sum_equality :
  (∑ b in finset.range (99), 1 / log (factorial 100)) = 1 :=
sorry

end log_sum_equality_l760_760167


namespace area_square_hypotenuse_l760_760142

theorem area_square_hypotenuse 
(a : ℝ) 
(h1 : ∀ a: ℝ,  ∃ YZ: ℝ, YZ = a + 3) 
(h2: ∀ XY: ℝ, ∃ total_area: ℝ, XY^2 + XY * (XY + 3) + (2 * XY^2 + 6 * XY + 9) = 450) :
  ∃ XZ: ℝ, (2 * a^2 + 6 * a + 9 = XZ) → XZ = 201 := by
  sorry

end area_square_hypotenuse_l760_760142


namespace sum_of_primes_less_than_twenty_is_77_l760_760044

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760044


namespace sum_of_primes_less_than_20_eq_77_l760_760017

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760017


namespace number_of_common_tangents_l760_760178

-- Define the circles using given conditions
def circle1 := { p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1 }
def circle2 := { p : ℝ × ℝ | p.1^2 + (p.2 + 2)^2 = 4 }

-- The question to prove, including conditions
theorem number_of_common_tangents :
  ∃ n : ℕ, n = 2 ∧ 
  (number_of_common_tangents circle1 circle2 = n) :=
sorry

end number_of_common_tangents_l760_760178


namespace greatest_two_digit_multiple_of_17_l760_760434

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760434


namespace stock_decrease_2008_l760_760575

theorem stock_decrease_2008 (x : ℝ) (h: 1.43 * x * (1 - p / 100) = 1.10 * x) : p ≈ 23.08 :=
by 
  have h1 : 1.43 * (1 - p / 100) = 1.10 := by sorry
  have h2 : 1 - p / 100 = 1.10 / 1.43 := by sorry
  have h3 : p / 100 = 1 - 1.10 / 1.43 := by sorry
  have h4 : p / 100 ≈ 0.23077 := by sorry
  exact sorry

end stock_decrease_2008_l760_760575


namespace bags_of_macintosh_l760_760628

-- Definitions based on conditions
def bags_of_golden_delicious : ℝ := 0.17
def bags_of_cortland : ℝ := 0.33
def total_bags : ℝ := 0.67

-- The proof problem
theorem bags_of_macintosh :
  ∃ (bags_of_macintosh : ℝ),
    total_bags = bags_of_golden_delicious + bags_of_macintosh + bags_of_cortland ∧
    bags_of_macintosh = 0.17 :=
begin
  sorry
end

end bags_of_macintosh_l760_760628


namespace total_weight_of_13_gold_bars_l760_760374

theorem total_weight_of_13_gold_bars
    (C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 : ℝ)
    (w12 w13 w23 w45 w67 w89 w1011 w1213 : ℝ)
    (h1 : w12 = C1 + C2)
    (h2 : w13 = C1 + C3)
    (h3 : w23 = C2 + C3)
    (h4 : w45 = C4 + C5)
    (h5 : w67 = C6 + C7)
    (h6 : w89 = C8 + C9)
    (h7 : w1011 = C10 + C11)
    (h8 : w1213 = C12 + C13) :
    C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12 + C13 = 
    (C1 + C2 + C3) + (C4 + C5) + (C6 + C7) + (C8 + C9) + (C10 + C11) + (C12 + C13) := 
  by
  sorry

end total_weight_of_13_gold_bars_l760_760374


namespace greatest_two_digit_multiple_of_17_l760_760506

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760506


namespace number_of_teachers_l760_760286

theorem number_of_teachers (students : ℕ) (classes_per_student : ℕ) (students_per_class : ℕ) (classes_per_teacher : ℕ)
  (h_students : students = 1500)
  (h_classes_per_student : classes_per_student = 5)
  (h_students_per_class : students_per_class = 25)
  (h_classes_per_teacher : classes_per_teacher = 5) :
  ∃ teachers : ℕ, teachers = 60 :=
by {
  let total_classes := students * classes_per_student,
  have h1: total_classes = 7500, by rw [h_students, h_classes_per_student]; norm_num,

  let unique_classes := total_classes / students_per_class,
  have h2: unique_classes = 300, by rw [h1, h_students_per_class]; norm_num,

  let teachers := unique_classes / classes_per_teacher,
  have h3: teachers = 60, by rw [h2, h_classes_per_teacher]; norm_num,

  use teachers,
  exact h3,
}

end number_of_teachers_l760_760286


namespace sin_tan_gt_x_square_l760_760951

theorem sin_tan_gt_x_square (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) : sin x * tan x > x^2 := 
sorry

end sin_tan_gt_x_square_l760_760951


namespace sum_prime_numbers_less_than_twenty_l760_760036

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760036


namespace sequence_of_consecutive_integers_with_primes_l760_760627

theorem sequence_of_consecutive_integers_with_primes (N : ℕ) (hN : N = 2011^2011) :
  ∃ seq : list ℕ, (seq.length = N) ∧ (count (λ n, prime n) seq = 2011) :=
sorry

end sequence_of_consecutive_integers_with_primes_l760_760627


namespace complex_problem_l760_760681

open Complex

theorem complex_problem (b : ℝ) (z : ℂ) (h1 : z = 3 + b * I) (h2 : (1 + 3 * I) * z = 0 + (Im ((1 + 3 * I) * z) * I)) :
  z = 3 + I ∧ (∃ w : ℂ, w = z / (2 + I) ∧ abs w = Real.sqrt 2) :=
by {
  sorry,
}

end complex_problem_l760_760681


namespace range_of_g_l760_760183

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), 
  ∃ r ∈ set.Icc (Real.pi / 2 - Real.arctan 2) (Real.pi / 2 + Real.arctan 2), g x = r :=
by
  sorry

end range_of_g_l760_760183


namespace number_of_boys_girls_l760_760977

-- Define the initial conditions.
def group_size : ℕ := 8
def total_ways : ℕ := 90

-- Define the actual proof problem.
theorem number_of_boys_girls 
  (n m : ℕ) 
  (h1 : n + m = group_size) 
  (h2 : Nat.choose n 2 * Nat.choose m 1 * Nat.factorial 3 = total_ways) 
  : n = 3 ∧ m = 5 :=
sorry

end number_of_boys_girls_l760_760977


namespace f_is_odd_range_of_x_l760_760163

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂
axiom f_3 : f 3 = 1
axiom f_increase_nonneg : ∀ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ x₂) → f x₁ ≤ f x₂
axiom f_lt_2 : ∀ x : ℝ, f (x - 1) < 2

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem range_of_x : {x : ℝ | f (x - 1) < 2} =
{s : ℝ | sorry } :=
sorry

end f_is_odd_range_of_x_l760_760163


namespace sum_of_primes_less_than_twenty_is_77_l760_760056

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760056


namespace my_age_is_18_l760_760241

theorem my_age_is_18 :
  ∃ d u : ℕ, 
  (x = 10 * d + u) ∧
  (alfred_age = 10 * u + d) ∧
  (|((10 * d + u) / (d + u)) - ((10 * u + d) / (d + u))| = |d - u|) ∧
  (d + u = 9) ∧
  (x = 18) :=
by
  let x := 18
  let alfred_age := 81
  have h₁ : d + u = 9 := sorry
  have h₂ : |(10 * d + u) / (d + u) - (10 * u + d) / (d + u)| = |d - u| := sorry
  have h₃ : (10 * d + u) / (d + u) * (10 * u + d) / (d + u) = (10 * d + u) := sorry
  use 1, 8
  simp [h₁, h₂, h₃]

end my_age_is_18_l760_760241


namespace greatest_two_digit_multiple_of_17_l760_760424

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760424


namespace scientific_notation_of_nanometer_l760_760323

theorem scientific_notation_of_nanometer : (0.000000001 : ℝ) = 1 * 10^(-9) :=
by
  sorry

end scientific_notation_of_nanometer_l760_760323


namespace greatest_two_digit_multiple_of_17_l760_760512

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760512


namespace reachable_region_l760_760565

open Real

def Point := (ℝ × ℝ)

def within_triangle (P O A B : Point) : Prop :=
  let (p1, p2) := P in
  let (o1, o2) := O in
  let (a1, a2) := A in
  let (b1, b2) := B in
  (b1 * a2 - b2 * a1) * p1 + ((a1 - o1) * (a2 - p2) - (a1 - p1) * (a2 - o2)) * (o2 - b2) + ((a2 - o2) * (o2 - b2) - (o1 - b1) * (o2 - p2)) * a1 ≥ 0

def distance (P Q : Point) : ℝ :=
  let (p1, p2) := P in
  let (q1, q2) := Q in
  (p1 - q1) ^ 2 + (p2 - q2) ^ 2

def within_sector (P O A C : Point) : Prop :=
  let (p1, p2) := P in
  let (o1, o2) := O in
  let (c1, c2) := C in
  ((p2 - o2) / distance(P, O)) <= sqrt(3) * (p1 / distance(P, O))

theorem reachable_region (P : Point) (O : Point := (0, 0)) (A : Point := (1 / 2, sqrt(3) / 2)) (B : Point := (2, 0)) (C : Point := (1, 0)) :
  within_triangle P O A B ∨ within_sector P O A C :=
sorry

end reachable_region_l760_760565


namespace tan_alpha_minus_pi_over_3_sin_double_alpha_l760_760198

variable (α : ℝ)

def tan_val (α : ℝ) := Real.tan α
def given_tan_alpha := tan_val α = 2

def tan_subtract (u v : ℝ) := (tan_val u - tan_val v) / (1 + tan_val u * tan_val v)
def sin_double (u : ℝ) := 2 * tan_val u / (1 + (tan_val u)^2)

theorem tan_alpha_minus_pi_over_3 : given_tan_alpha α → tan_subtract α (Real.pi / 3) = (5 * Real.sqrt 3 - 8) / 11 := by
  sorry

theorem sin_double_alpha : given_tan_alpha α → sin_double α = 4 / 5 := by
  sorry

end tan_alpha_minus_pi_over_3_sin_double_alpha_l760_760198


namespace greatest_two_digit_multiple_of_17_l760_760408

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760408


namespace greatest_two_digit_multiple_of_17_l760_760437

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760437


namespace volume_of_tetrahedron_l760_760275

-- Define the setup of tetrahedron D-ABC
def tetrahedron_volume (V : ℝ) : Prop :=
  ∃ (DA : ℝ) (A B C D : ℝ × ℝ × ℝ), 
  A = (0, 0, 0) ∧ 
  B = (2, 0, 0) ∧ 
  C = (1, Real.sqrt 3, 0) ∧
  D = (1, Real.sqrt 3/3, DA) ∧
  DA = 2 * Real.sqrt 3 ∧
  ∃ tan_dihedral : ℝ, tan_dihedral = 2 ∧
  V = 2

-- The statement to prove the volume is indeed 2 given the conditions.
theorem volume_of_tetrahedron : ∃ V, tetrahedron_volume V :=
by 
  sorry

end volume_of_tetrahedron_l760_760275


namespace sum_of_arithmetic_sequence_l760_760893

-- Define an arithmetic sequence and the sum of the first n terms
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (a₁ + arithmetic_sequence a₁ d n) / 2

-- Rewrite the math proof problem
theorem sum_of_arithmetic_sequence {a₁ d : ℝ} :
  (a₁ + arithmetic_sequence a₁ d 9 + arithmetic_sequence a₁ d 11 = 30) →
  arithmetic_sum a₁ d 13 = 130 :=
by
  intro h
  sorry

end sum_of_arithmetic_sequence_l760_760893


namespace vehicle_value_last_year_l760_760383

theorem vehicle_value_last_year (value_this_year : ℝ) (ratio : ℝ) (value_this_year_cond : value_this_year = 16000) (ratio_cond : ratio = 0.8) :
  ∃ (value_last_year : ℝ), value_this_year = ratio * value_last_year ∧ value_last_year = 20000 :=
by
  use 20000
  sorry

end vehicle_value_last_year_l760_760383


namespace new_bill_amount_l760_760244

def original_order : ℝ := 25
def tomato_cost_old : ℝ := 0.99
def tomato_cost_new : ℝ := 2.20
def lettuce_cost_old : ℝ := 1.00
def lettuce_cost_new : ℝ := 1.75
def celery_cost_old : ℝ := 1.96
def celery_cost_new : ℝ := 2.00
def delivery_and_tip : ℝ := 8.00

theorem new_bill_amount :
  original_order + (tomato_cost_new - tomato_cost_old) + 
  (lettuce_cost_new - lettuce_cost_old) + 
  (celery_cost_new - celery_cost_old) + delivery_and_tip = 35 := 
by
  calc
    original_order + (tomato_cost_new - tomato_cost_old) + 
    (lettuce_cost_new - lettuce_cost_old) + 
    (celery_cost_new - celery_cost_old) + delivery_and_tip
    = 25 + (2.20 - 0.99) + (1.75 - 1.00) + (2.00 - 1.96) + 8.00 : by sorry -- intermediate step for clarity
    ... = 25 + 1.21 + 0.75 + 0.04 + 8.00 : by sorry
    ... = 25 + 2.00 + 8.00 : by sorry
    ... = 25 + 10.00 : by sorry
    ... = 35 : by sorry

end new_bill_amount_l760_760244


namespace P_inter_Q_eq_set12_l760_760230

-- Define the sets P and Q as given in the conditions
def P := {x : ℕ | 1 ≤ x ∧ x ≤ 10}
def Q := {x : ℝ | x^2 + x - 6 ≤ 0}

-- The main theorem to prove
theorem P_inter_Q_eq_set12 : {x : ℝ | x ∈ P ∧ x ∈ Q} = {1.0, 2.0} :=
sorry

end P_inter_Q_eq_set12_l760_760230


namespace probability_double_interval_greater_l760_760819

theorem probability_double_interval_greater
  (x_dist : MeasureTheory.Measure (Set.Icc (0 : ℝ) 1000))
  (y_dist : MeasureTheory.Measure (Set.Icc (0 : ℝ) 2000)) :
  MeasureTheory.MeasureTheory.prob3P (λ(x y : ℝ), y > 2 * x) x_dist y_dist = 1 / 2 :=
sorry

end probability_double_interval_greater_l760_760819


namespace marthas_bedroom_size_l760_760888

-- Define the variables and conditions
def total_square_footage := 300
def additional_square_footage := 60
def Martha := 120
def Jenny := Martha + additional_square_footage

-- The main theorem stating the requirement 
theorem marthas_bedroom_size : (Martha + (Martha + additional_square_footage) = total_square_footage) -> Martha = 120 :=
by 
  sorry

end marthas_bedroom_size_l760_760888


namespace manufacturing_sector_angle_l760_760851

theorem manufacturing_sector_angle (h1 : 50 ≤ 100) (h2 : 360 = 4 * 90) : 0.50 * 360 = 180 := 
by
  sorry

end manufacturing_sector_angle_l760_760851


namespace p_minus_q_value_l760_760808

theorem p_minus_q_value (p q : ℝ) (h1 : (x - 4) * (x + 4) = 24 * x - 96) (h2 : x^2 - 24 * x + 80 = 0) (h3 : p = 20) (h4 : q = 4) : p - q = 16 :=
by
  sorry

end p_minus_q_value_l760_760808


namespace division_result_l760_760330

theorem division_result (quotient divisor remainder : ℕ) (h_quotient : quotient = 5) (h_divisor : divisor = 4) (h_remainder : remainder = 3) : 
  quotient * divisor + remainder = 23 := by
  rw [h_quotient, h_divisor, h_remainder]
  sorry

end division_result_l760_760330


namespace wheel_distance_traveled_l760_760583

-- Define the problem
def wheel_radius : ℝ := 1

-- Define the circumference of the wheel
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- Distance traveled by the center of the wheel after one complete revolution
theorem wheel_distance_traveled :
  circumference wheel_radius = 2 * Real.pi :=
by
  sorry

end wheel_distance_traveled_l760_760583


namespace system_of_linear_eq_with_two_variables_l760_760093

-- Definitions of individual equations
def eqA (x : ℝ) : Prop := 3 * x - 2 = 5
def eqB (x : ℝ) : Prop := 6 * x^2 - 2 = 0
def eqC (x y : ℝ) : Prop := 1 / x + y = 3
def eqD (x y : ℝ) : Prop := 5 * x + y = 2

-- The main theorem to prove that D is a system of linear equations with two variables
theorem system_of_linear_eq_with_two_variables :
    (∃ x y : ℝ, eqD x y) ∧ (¬∃ x : ℝ, eqA x) ∧ (¬∃ x : ℝ, eqB x) ∧ (¬∃ x y : ℝ, eqC x y) :=
by
  sorry

end system_of_linear_eq_with_two_variables_l760_760093


namespace value_of_F_at_neg1_l760_760149

-- Define the function F(x)
noncomputable def F (x : ℝ) : ℝ := (abs (x - 2)).sqrt + (8 / Real.pi) * Real.arctan (abs (x + 1)).sqrt

-- The proof statement
theorem value_of_F_at_neg1 : F (-1) = 4 :=
begin
    sorry
end

end value_of_F_at_neg1_l760_760149


namespace greatest_two_digit_multiple_of_17_l760_760489

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760489


namespace total_rainfall_2004_l760_760259

-- Given conditions
def avg_rainfall_2003 := 41.5 -- in mm
def additional_rain_2004 := 2 -- in mm
def months_in_year := 12

-- Define the average rainfall in 2004
def avg_rainfall_2004 := avg_rainfall_2003 + additional_rain_2004

-- The statement of the total rainfall in 2004
theorem total_rainfall_2004 : avg_rainfall_2004 * months_in_year = 522 := 
by
  sorry

end total_rainfall_2004_l760_760259


namespace sum_of_primes_less_than_20_eq_77_l760_760021

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760021


namespace greatest_two_digit_multiple_of_17_is_85_l760_760464

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760464


namespace greatest_two_digit_multiple_of_17_l760_760508

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760508


namespace g_neg4_g_3_l760_760812

def g (x : ℝ) : ℝ :=
  if x < 2 then 2 * x - 1 else 10 - 3 * x

theorem g_neg4 : g (-4) = -9 := sorry

theorem g_3 : g 3 = 1 := sorry

end g_neg4_g_3_l760_760812


namespace part_one_part_two_l760_760210

-- Definition for Part (1)
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (1, 1)
def k_vec_a_sub_vec_b (k : ℝ) : ℝ × ℝ := (k - 1, 2 * k - 1)
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Part (1) Proof
theorem part_one (k : ℝ) (h : dot_product (k_vec_a_sub_vec_b k) vector_a = 0) : k = 3 / 5 :=
sorry

-- Definition for Part (2)
def parallel (v₁ v₂ : ℝ × ℝ) : Prop := ∃ (λ : ℝ), v₂ = (λ * v₁.1, λ * v₁.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Part (2) Proof
theorem part_two (vec_b : ℝ × ℝ) 
  (h1 : parallel vector_a vec_b)
  (h2 : magnitude vec_b = 2 * Real.sqrt 5) : vec_b = (2, 4) ∨ vec_b = (-2, -4) :=
sorry

end part_one_part_two_l760_760210


namespace greatest_two_digit_multiple_of_17_l760_760429

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760429


namespace sum_of_primes_less_than_twenty_is_77_l760_760052

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760052


namespace trees_cleaning_time_l760_760785

noncomputable def time_per_tree_without_help (trees_x : ℕ) (trees_y : ℕ) (total_time_with_help : ℕ) (time_factor : ℚ) : ℚ :=
  let total_trees := trees_x * trees_y in
  let time_per_tree_with_help := total_time_with_help / total_trees in
  time_per_tree_with_help * time_factor

theorem trees_cleaning_time 
  (hx : trees_x = 4) 
  (hy : trees_y = 5) 
  (ht : total_time_with_help = 60) 
  (hf : time_factor = 2) : 
  time_per_tree_without_help trees_x trees_y total_time_with_help time_factor = 6 :=
by
  sorry

end trees_cleaning_time_l760_760785


namespace probability_two_tails_after_two_heads_l760_760617

noncomputable def fair_coin_probability : ℚ :=
  -- Given conditions:
  let p_head := (1 : ℚ) / 2
  let p_tail := (1 : ℚ) / 2

  -- Define the probability Q as stated in the problem
  let Q := ((1 : ℚ) / 4) / (1 - (1 : ℚ) / 4)

  -- Calculate the probability of starting with sequence "HTH"
  let p_HTH := p_head * p_tail * p_head

  -- Calculate the final probability
  p_HTH * Q

theorem probability_two_tails_after_two_heads :
  fair_coin_probability = (1 : ℚ) / 24 :=
by
  sorry

end probability_two_tails_after_two_heads_l760_760617


namespace minimize_average_cost_l760_760909

def cost_function (x : ℝ) : ℝ := 
  if 1 ≤ x ∧ x < 30 then
    (1 / 25) * x^3 - 640
  else if 30 ≤ x ∧ x ≤ 50 then
    x^2 - 10 * x + 1600
  else 
    0 -- Outside of the given interval, we can define the cost as 0 or undefined.

def average_cost (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 30 then
    ((1 / 25) * x^2 + 640 / x)
  else if 30 ≤ x ∧ x ≤ 50 then
    (x - 10 + 1600 / x)
  else 
    0 -- Similarly, undefined or 0 outside given ranges.

theorem minimize_average_cost : 
  (∀ x ∈ Icc 1 50, 
    average_cost x ≥ average_cost 40) ∧
  (average_cost 40 < average_cost 20) := by
  sorry

end minimize_average_cost_l760_760909


namespace brenda_mice_left_l760_760996

theorem brenda_mice_left (litters : ℕ) (mice_per_litter : ℕ) (fraction_to_robbie : ℚ) 
                          (mult_to_pet_store : ℕ) (fraction_to_feeder : ℚ) 
                          (total_mice : ℕ) (to_robbie : ℕ) (to_pet_store : ℕ) 
                          (remaining_after_first_sales : ℕ) (to_feeder : ℕ) (left_after_feeder : ℕ) :
  litters = 3 →
  mice_per_litter = 8 →
  fraction_to_robbie = 1/6 →
  mult_to_pet_store = 3 →
  fraction_to_feeder = 1/2 →
  total_mice = litters * mice_per_litter →
  to_robbie = total_mice * fraction_to_robbie →
  to_pet_store = mult_to_pet_store * to_robbie →
  remaining_after_first_sales = total_mice - to_robbie - to_pet_store →
  to_feeder = remaining_after_first_sales * fraction_to_feeder →
  left_after_feeder = remaining_after_first_sales - to_feeder →
  left_after_feeder = 4 := sorry

end brenda_mice_left_l760_760996


namespace sum_of_valid_z_l760_760166

theorem sum_of_valid_z : 
  ∑ z in {z | z ∈ (Finset.range 10) ∧ (17 + z) % 3 = 0}, z = 12 :=
by
  sorry

end sum_of_valid_z_l760_760166


namespace non_zero_real_x_solution_l760_760923

theorem non_zero_real_x_solution (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 → x = 2 / 9 := by
  sorry

end non_zero_real_x_solution_l760_760923


namespace part1_part2_part3_l760_760222

namespace Problem

-- Definitions and conditions for problem 1
def f (m x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + (m - 1)

theorem part1 (m : ℝ) :
  (∀ x : ℝ, f m x < 0) ↔ m < -5/3 := sorry

-- Definitions and conditions for problem 2
theorem part2 (m : ℝ) (h : m < 0) :
  ((-1 < m ∧ m < 0) → ∀ x : ℝ, x ≤ 1 ∨ x ≥ 1 / (m + 1)) ∧
  (m = -1 → ∀ x : ℝ, x ≤ 1) ∧
  (m < -1 → ∀ x : ℝ, 1 / (m + 1) ≤ x ∧ x ≤ 1) := sorry

-- Definitions and conditions for problem 3
theorem part3 (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f m x ≥ x^2 + 2 * x) ↔ m ≥ (2 * Real.sqrt 3) / 3 + 1 := sorry

end Problem

end part1_part2_part3_l760_760222


namespace percentage_greater_l760_760258

theorem percentage_greater (x y : ℝ) (h : x = 0.7142857142857143 * y) : 
  ∃ (P : ℝ), P = 28.57142857142857 ∧ y = (1 + P / 100) * x :=
by
  use 28.57142857142857
  split
  {
    exact rfl
  }
  {
    -- Skipping the proof steps here; we only need the statement
    sorry
  }

end percentage_greater_l760_760258


namespace length_of_remaining_rectangle_l760_760971

theorem length_of_remaining_rectangle (length width : ℕ) (h_length : length = 20) (h_width : width = 15) : 
  (remaining_length length width) = 10 := 
by 
  sorry

-- Define the behavior of cutting the largest square
def remaining_length (l w : ℕ) :=
  let first_cut_length := min l w in
  let new_length := max l w - first_cut_length in
  let new_width := first_cut_length in
  let second_cut_length := min new_length new_width in
  new_length - second_cut_length


end length_of_remaining_rectangle_l760_760971


namespace sum_of_primes_less_than_20_eq_77_l760_760013

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760013


namespace circle_radius_k_l760_760194

theorem circle_radius_k (k : ℝ) : (∃ x y : ℝ, (x^2 + 14*x + y^2 + 8*y - k = 0) ∧ ((x + 7)^2 + (y + 4)^2 = 100)) → k = 35 :=
by
  sorry

end circle_radius_k_l760_760194


namespace walking_area_calculation_l760_760767

noncomputable def walking_area_of_park (park_length park_width fountain_radius : ℝ) : ℝ :=
  let park_area := park_length * park_width
  let fountain_area := Real.pi * fountain_radius^2
  park_area - fountain_area

theorem walking_area_calculation :
  walking_area_of_park 50 30 5 = 1500 - 25 * Real.pi :=
by
  sorry

end walking_area_calculation_l760_760767


namespace shampoo_usage_per_wash_l760_760602

theorem shampoo_usage_per_wash :
  ∀ (shampoos_per_bottle : ℕ) (bottles_per_year : ℕ) (days_in_leap_year : ℕ),
  (days_in_leap_year % 2 = 0) →
  (shampoos_per_bottle = 14) →
  (bottles_per_year = 4) →
  (days_in_leap_year = 366) →
  let washes_per_year := days_in_leap_year / 2 in
  let total_shampoo_used := bottles_per_year * shampoos_per_bottle in
  let shampoo_per_wash := (total_shampoo_used : ℝ) / washes_per_year in
  shampoo_per_wash ≈ 0.306 := 
begin
  intros,
  have washes_per_year_eq : washes_per_year = 183,
  { simp only [washes_per_year],
    exact (nat.div_eq_of_eq_mul (by simp)).symm },
  have total_shampoo_used_eq : total_shampoo_used = 4 * 14,
  { simp },
  have shampoo_per_wash_eq : shampoo_per_wash = (56 : ℝ) / 183,
  { simp [shampoo_per_wash, total_shampoo_used_eq, washes_per_year_eq] },
  show (56 : ℝ) / 183 ≈ 0.306,
  sorry
end

end shampoo_usage_per_wash_l760_760602


namespace greatest_two_digit_multiple_of_17_l760_760455

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760455


namespace total_inflation_time_l760_760572

/-- 
  Assume a soccer ball takes 20 minutes to inflate.
  Alexia inflates 20 soccer balls.
  Ermias inflates 5 more balls than Alexia.
  Prove that the total time in minutes taken to inflate all the balls is 900 minutes.
-/
theorem total_inflation_time 
  (alexia_balls : ℕ) (ermias_balls : ℕ) (each_ball_time : ℕ)
  (h1 : alexia_balls = 20)
  (h2 : ermias_balls = alexia_balls + 5)
  (h3 : each_ball_time = 20) :
  (alexia_balls + ermias_balls) * each_ball_time = 900 :=
by
  sorry

end total_inflation_time_l760_760572


namespace sum_of_primes_less_than_twenty_is_77_l760_760043

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l760_760043


namespace greatest_two_digit_multiple_of_17_l760_760423

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760423


namespace num_students_l760_760379

theorem num_students (n : ℕ) (h1 : n < 60) (h2 : (60 - n) % 2 = 0): 
  (60 - n) / 2 = n → n = 20 :=
by
  intro h
  linarith

end num_students_l760_760379


namespace largest_possible_n_l760_760596

open Nat

-- Define arithmetic sequences a_n and b_n with given initial conditions
def arithmetic_seq (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) :=
  a_n 1 = 1 ∧ b_n 1 = 1 ∧ 
  a_n 2 ≤ b_n 2 ∧
  (∃n : ℕ, a_n n * b_n n = 1764)

-- Given the arithmetic sequences defined above, prove that the largest possible value of n is 44
theorem largest_possible_n : 
  ∀ (a_n b_n : ℕ → ℕ), arithmetic_seq a_n b_n →
  ∀ (n : ℕ), (a_n n * b_n n = 1764) → n ≤ 44 :=
sorry

end largest_possible_n_l760_760596


namespace cone_surface_area_calc_l760_760711

noncomputable def surface_area_of_cone (l: ℝ) (θ: ℝ) :=
  let r := l * Real.cos θ in
  Real.pi * r * (r + l)

theorem cone_surface_area_calc :
  surface_area_of_cone 8 (Real.pi / 3) = 48 * Real.pi  :=
by
  sorry

end cone_surface_area_calc_l760_760711


namespace at_least_one_divisible_by_10_l760_760729

theorem at_least_one_divisible_by_10
  (a b c : ℕ)
  (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  (∃ x, x ∈ {a^3 * b - a * b^3, b^3 * c - b * c^3, c^3 * a - c * a^3} ∧ 10 ∣ x) :=
sorry

end at_least_one_divisible_by_10_l760_760729


namespace sum_of_primes_less_than_20_l760_760010

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760010


namespace sum_of_primes_less_than_20_eq_77_l760_760020

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760020


namespace greatest_two_digit_multiple_of_17_l760_760433

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760433


namespace three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l760_760926

theorem three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693 :
  ∃ (n : ℕ), n = 693 ∧ 
    (100 * 6 + 10 * (n / 10 % 10) + 3) = n ∧
    (n % 10 = 3) ∧
    (n / 100 = 6) ∧
    n % 11 = 0 :=
by
  sorry

end three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l760_760926


namespace geometric_series_first_term_l760_760140

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_ratio : r = 1/4) (h_sum : S = 80) (h_series : S = a / (1 - r)) :
  a = 60 :=
by
  sorry

end geometric_series_first_term_l760_760140


namespace break_even_price_is_9_l760_760279

-- Definitions corresponding to our conditions
def fixed_cost : ℝ := 50000
def variable_cost_per_book : ℝ := 4
def number_of_books : ℝ := 10000

-- Definition for total cost based on conditions
def total_cost : ℝ := fixed_cost + (variable_cost_per_book * number_of_books)

-- Target statement: the break-even price per book
def break_even_price_per_book : ℝ := total_cost / number_of_books

theorem break_even_price_is_9 : break_even_price_per_book = 9 := by
  -- Proof details go here
  sorry

end break_even_price_is_9_l760_760279


namespace greatest_two_digit_multiple_of_17_is_85_l760_760463

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760463


namespace sum_primes_less_than_20_l760_760081

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760081


namespace area_of_triangle_l760_760717

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 1
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 1
def tangent_line_at_point (x y: ℝ) (pt: ℝ × ℝ) : Prop := x + y - 1 = 0

theorem area_of_triangle : 
  let x := (0, 1 : ℝ × ℝ) in
  let t := tangent_line_at_point in
  let y_intercept := 1 in
  let x_intercept := 1 in
  let area := (1/2 : ℝ) * y_intercept * x_intercept in
  ∀ f f' t x y_intercept x_intercept area, f = λ x, x^3 - x + 1
  ∧ f' = λ x, 3 * x^2 - 1
  ∧ t x y = x + y - 1 = 0
  ∧ (0, 1 : ℝ × ℝ)
  ∧ y_intercept = 1
  ∧ x_intercept = 1
  ⊢ area = (1/2 : ℝ) := sorry

end area_of_triangle_l760_760717


namespace greatest_two_digit_multiple_of_17_l760_760502

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760502


namespace molecular_weight_CaOH2_correct_l760_760605

/-- Molecular weight of Calcium hydroxide -/
def molecular_weight_CaOH2 (Ca O H : ℝ) : ℝ :=
  Ca + 2 * (O + H)

theorem molecular_weight_CaOH2_correct :
  molecular_weight_CaOH2 40.08 16.00 1.01 = 74.10 :=
by 
  -- This statement requires a proof that would likely involve arithmetic on real numbers
  sorry

end molecular_weight_CaOH2_correct_l760_760605


namespace sin_60_eq_sqrt3_div_2_l760_760661

theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
by
  sorry

end sin_60_eq_sqrt3_div_2_l760_760661


namespace grown_ups_in_milburg_l760_760896

def total_population : ℕ := 8243
def number_of_children : ℕ := 2987

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 :=
by {
  sorry
}

end grown_ups_in_milburg_l760_760896


namespace decimal_representation_ends_with_1994_l760_760835

def is_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def euler_totient (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else Fintype.card (Finset.filter (is_coprime n) (Finset.range n))

noncomputable def phi_10000 : ℕ := euler_totient 10000

theorem decimal_representation_ends_with_1994 (n : ℕ) (hn : n = 4000)
  (h1 : is_coprime 1993 10000)
  (h2 : 1993^phi_10000 ≡ 1 [MOD 10000]) :
  1994 * 1993^n ≡ 1994 [MOD 10000] :=
by
  sorry

end decimal_representation_ends_with_1994_l760_760835


namespace sum_primes_less_than_20_l760_760083

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760083


namespace trig_identity_problem_l760_760302

theorem trig_identity_problem (x y : ℝ) 
  (h1 : sin x / sin y = 2) 
  (h2 : cos x / cos y = 3) : 
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 6 + (9 * (cos y)^2 - 2) / (4 * (cos y)^2 - 2) :=
by sorry

end trig_identity_problem_l760_760302


namespace k_cubed_divisible_l760_760746

theorem k_cubed_divisible (k : ℕ) (h : k = 84) : ∃ n : ℕ, k ^ 3 = 592704 * n :=
by
  sorry

end k_cubed_divisible_l760_760746


namespace college_application_ways_correct_l760_760560

def college_application_ways : ℕ :=
  -- Scenario 1: Student does not apply to either of the two conflicting colleges
  (Nat.choose 4 3) +
  -- Scenario 2: Student applies to one of the two conflicting colleges
  ((Nat.choose 2 1) * (Nat.choose 4 2))

theorem college_application_ways_correct : college_application_ways = 16 := by
  -- We can skip the proof
  sorry

end college_application_ways_correct_l760_760560


namespace number_of_incorrect_props_is_3_l760_760985

-- Definition of the propositions
def prop1 : Prop := ∀ {α β : ℝ}, (sin α = sin β) ∧ (cos α = cos β) → α = β
def prop2 : Prop := ∀ {α : ℝ}, (sin α > 0) → (0 < α ∧ α < π)
def prop3 : Prop := ∀ {α β : ℝ}, (α ≠ β) → (sin α ≠ sin β) ∧ (cos α ≠ cos β)
def prop4 : Prop := ∀ {f : ℝ → ℝ} {y : ℝ}, (∃ α, f α = y) → (∃! α, f α = y)

-- Indicating incorrect propositions
def incorrect_props_count : Nat := [prop1, prop2, prop3, prop4].count (λ p, ¬ p = true)

-- The lemma to prove that the number of incorrect propositions is 3 
theorem number_of_incorrect_props_is_3 : incorrect_props_count = 3 := by
  sorry

end number_of_incorrect_props_is_3_l760_760985


namespace circumference_of_circle_of_given_area_l760_760399

theorem circumference_of_circle_of_given_area (A : ℝ) (h : A = 225 * Real.pi) : 
  ∃ C : ℝ, C = 2 * Real.pi * 15 :=
by
  let r := 15
  let C := 2 * Real.pi * r
  use C
  sorry

end circumference_of_circle_of_given_area_l760_760399


namespace sphere_radii_l760_760880

theorem sphere_radii (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  (∃ x y z : ℝ, x = b * c / (2 * a) ∧ y = a * c / (2 * b) ∧ z = a * b / (2 * c)) :=
by
  use [b * c / (2 * a), a * c / (2 * b), a * b / (2 * c)]
  split
  · exact rfl
  split
  · exact rfl
  · exact rfl

end sphere_radii_l760_760880


namespace triangle_ratios_l760_760783

theorem triangle_ratios
  (XY XZ : ℝ) (YZ E N Q : Point) (hXY : XY = 25) (hXZ : XZ = 14)
  (angle_bisector_X : AngleBisector (∠ XYZ XZ E))
  (N_midpoint : Midpoint X E N)
  (Q_intersection : Intersection (Line Q YN) (Line XZ))
  (ratio_ZQ_QX : ∃ p q : ℕ, gcd p q = 1 ∧ ZQ / QX = p / q) :
  ∃ p q : ℕ, gcd p q = 1 ∧ p + q = 39 :=
begin
  sorry
end

end triangle_ratios_l760_760783


namespace area_of_right_triangle_l760_760129

noncomputable def area_of_right_triangle_inscribed_in_circle 
  (α : ℝ) (r : ℝ) 
  (h_non_neg : 0 < α ∧ α < π/2) : ℝ :=
  let sin_2α := Real.sin (2 * α) in
  let area := (2 * r^2 * sin_2α) / (1 + sin_2α^2) in
  area

-- Main theorem statement
theorem area_of_right_triangle 
  (α : ℝ) (r : ℝ) 
  (h_non_neg : 0 < α ∧ α < π/2) 
  (h_triangle_right : ∃ (A B C : ℝ×ℝ), (A, B, C form a right triangle 
                                            with right angle at C and hypotenuse AB )) 
  (h_inscribed_in_circle : ∃ (O : ℝ×ℝ), (r = distance from O to vertices of ABC) ∧ (AB is a chord of the circle)) :
  ∃ (area : ℝ), area = area_of_right_triangle_inscribed_in_circle α r h_non_neg :=
by sorry

end area_of_right_triangle_l760_760129


namespace grid_game_winner_l760_760128

/-- Statement of the problem -/
theorem grid_game_winner (m n : ℕ) : 
  (if (m + n) % 2 = 0 then "second player wins" else "first player wins") :=
sorry

end grid_game_winner_l760_760128


namespace solution_l760_760663

def r_7 (n : ℕ) : ℕ := n % 7

theorem solution (n : ℕ) :
  let ns := (List.range (15 * 7)).filter (λ n, r_7 (3 * n) ≤ 3)
  List.nth ns 14 = some 22 :=
by 
  sorry

end solution_l760_760663


namespace range_of_a_in_third_quadrant_l760_760679

theorem range_of_a_in_third_quadrant (a : ℝ) :
  let Z_re := a^2 - 2*a
  let Z_im := a^2 - a - 2
  (Z_re < 0 ∧ Z_im < 0) → 0 < a ∧ a < 2 :=
by
  sorry

end range_of_a_in_third_quadrant_l760_760679


namespace sum_prime_numbers_less_than_twenty_l760_760035

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760035


namespace find_x_l760_760207

-- Define the points A and B
structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨1, 2⟩
def B : Point := ⟨3, 5⟩

-- Define the vector a and AB
structure Vector :=
(x : ℝ)
(y : ℝ)

def vector_a (x : ℝ) : Vector := ⟨x, 6⟩
def vector_AB : Vector := ⟨B.x - A.x, B.y - A.y⟩

-- Define the condition for parallel vectors
def parallel_vectors (v1 v2 : Vector) : Prop :=
v1.x * v2.y = v1.y * v2.x

-- State the theorem
theorem find_x (x : ℝ) (h : parallel_vectors (vector_a x) vector_AB) : x = 4 :=
sorry

end find_x_l760_760207


namespace sum_primes_less_than_20_l760_760082

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760082


namespace range_of_b_l760_760749

-- Define the function f as -1/2*x^2 + b*ln x
def f (x b : ℝ) : ℝ :=
  - (1 / 2) * x^2 + b * Real.log x

-- Define the derivative of f
def f' (x b : ℝ) : ℝ :=
  -x + b / x

-- State the theorem
theorem range_of_b (b : ℝ) (h : ∀ x ∈ Icc 1 2, f' x b ≥ 0) : 4 ≤ b :=
by
  -- Skipped proof
  sorry

end range_of_b_l760_760749


namespace stratified_sampling_correct_l760_760115

variables (total_employees senior_employees mid_level_employees junior_employees sample_size : ℕ)
          (sampling_ratio : ℚ)
          (senior_sample mid_sample junior_sample : ℕ)

-- Conditions
def company_conditions := 
  total_employees = 450 ∧ 
  senior_employees = 45 ∧ 
  mid_level_employees = 135 ∧ 
  junior_employees = 270 ∧ 
  sample_size = 30 ∧ 
  sampling_ratio = 1 / 15

-- Proof goal
theorem stratified_sampling_correct : 
  company_conditions total_employees senior_employees mid_level_employees junior_employees sample_size sampling_ratio →
  senior_sample = senior_employees * sampling_ratio ∧ 
  mid_sample = mid_level_employees * sampling_ratio ∧ 
  junior_sample = junior_employees * sampling_ratio ∧
  senior_sample + mid_sample + junior_sample = sample_size :=
by sorry

end stratified_sampling_correct_l760_760115


namespace sum_of_primes_less_than_20_eq_77_l760_760019

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760019


namespace equation_of_circle_O2_equation_of_tangent_line_l760_760695

-- Define circle O1
def circle_O1 (x y : ℝ) : Prop :=
  x^2 + (y + 1)^2 = 4

-- Define the center and radius of circle O2 given that they are externally tangent
def center_O2 : ℝ × ℝ := (3, 3)
def radius_O2 : ℝ := 3

-- Prove the equation of circle O2
theorem equation_of_circle_O2 :
  ∀ (x y : ℝ), (x - 3)^2 + (y - 3)^2 = 9 := by
  intro x y
  sorry

-- Prove the equation of the common internal tangent line to circles O1 and O2
theorem equation_of_tangent_line :
  ∀ (x y : ℝ), 3 * x + 4 * y - 21 = 0 := by
  intro x y
  sorry

end equation_of_circle_O2_equation_of_tangent_line_l760_760695


namespace quadrilateral_area_correct_l760_760773

noncomputable def area_quadrilateral_ABCD :
  {AB BC CD : ℝ} → {m∠B m∠C : ℝ} →
    (AB = 5) →
    (BC = 6) →
    (CD = 7) →
    (m∠B = 120) →
    (m∠C = 100) → ℝ
| AB BC CD m∠B m∠C, hAB, hBC, hCD, hAngleB, hAngleC => 
  let A_ABC := 0.5 * AB * BC * Real.sin (120 * Real.pi / 180)
  let A_BCD := 0.5 * BC * CD * Real.sin (100 * Real.pi / 180)
  A_ABC + A_BCD

theorem quadrilateral_area_correct :
  {AB BC CD : ℝ} → {m∠B m∠C : ℝ} →
    (AB = 5) →
    (BC = 6) →
    (CD = 7) →
    (m∠B = 120) →
    (m∠C = 100) →
    area_quadrilateral_ABCD AB BC CD m∠B m∠C
    = (15 * Real.sqrt 3) / 2 + 20.69 := 
by
  -- Definitions and conditions
  sorry

end quadrilateral_area_correct_l760_760773


namespace measure_of_angle_A_range_of_perimeter_l760_760690

-- Conditions for the problem
variables (a b c : ℝ) (A B C S : ℝ)
-- Definitions
def triangle_area (b c : ℝ) (A : ℝ) := 0.5 * b * c * Real.sin A
def given_condition1 (a S b c : ℝ) := 3 * a^2 - 4 * Real.sqrt 3 * S = 3 * b^2 + 3 * c^2
def given_condition2 := a = 3

-- Part 1: Prove measure of angle A
theorem measure_of_angle_A (h1 : given_condition1 a S b c) (hS : S = triangle_area b c A) : A = 2*Real.pi / 3 :=
by
  sorry

-- Part 2: Determining the range of perimeter
theorem range_of_perimeter (h1 : given_condition1 3 S b c) (h2 : given_condition2) : 
  ∃ y, y ∈ Ioo 6 (3 + 2 * Real.sqrt 3) :=
by
  sorry

end measure_of_angle_A_range_of_perimeter_l760_760690


namespace sum_of_arithmetic_terms_l760_760877

theorem sum_of_arithmetic_terms (a₁ a₂ a₃ c d a₆ : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₂ = 10)
  (h₃ : a₃ = 17)
  (h₄ : a₆ = 32)
  (h_arith : ∀ n, (a₁ + n * (a₂ - a₁)) = seq)
  : c + d = 55 :=
by
  have d := a₂ - a₁
  have c := a₃ + d
  have d := c + d
  have h_seq := list.map (λ n, (a₁ + n * d)) (list.range 6) -- Making use of the arithmetic property
  have h_seq_eq := h_seq = [3, 10, 17, c, d, 32]
  sorry

end sum_of_arithmetic_terms_l760_760877


namespace greatest_two_digit_multiple_of_17_l760_760492

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760492


namespace product_of_numerator_denominator_of_073_l760_760920

theorem product_of_numerator_denominator_of_073 : (let num := 73 and den := 999 in num * den) = 72827 :=
by
  sorry

end product_of_numerator_denominator_of_073_l760_760920


namespace distance_between_foci_l760_760989

-- Define the properties of the ellipse
def ellipse_center := (3 : ℝ, 2 : ℝ)
def ellipse_tangent_x_axis := (3 : ℝ, 0 : ℝ)
def ellipse_tangent_y_axis := (0 : ℝ, 2 : ℝ)

-- Semi-major and semi-minor axes
def a : ℝ := 3
def b : ℝ := 2

-- Formula for the distance between the foci
theorem distance_between_foci : 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 5 := by
  sorry

end distance_between_foci_l760_760989


namespace action_figures_added_l760_760788

-- Definitions according to conditions
def initial_action_figures : ℕ := 4
def books_on_shelf : ℕ := 22 -- This information is not necessary for proving the action figures added
def total_action_figures_after_adding : ℕ := 10

-- Theorem to prove given the conditions
theorem action_figures_added : (total_action_figures_after_adding - initial_action_figures) = 6 := by
  sorry

end action_figures_added_l760_760788


namespace hyperbola_equation_l760_760654

theorem hyperbola_equation :
  let C := 5 in
  let B := 3 in
  let a_sq := 10 in
  let b_sq := 6 in
  let directrix := (C : ℝ) / 2 in
  ( (b_sq / a_sq = (B / C) * (C : ℝ)) ∧ 
    ((a_sq / (real.sqrt (a_sq + b_sq + 0))) = directrix) ) →
  ( ∃ a b : ℝ, ( (a ^ 2 = a_sq ) ∧ (b ^ 2 = b_sq) ∧ (∀ x y : ℝ, (x^2/(a: ℝ) ^ 2 - y^2/(b: ℝ) ^ 2 = 1 ) ) ) ) 
:= 
begin
  intros,
  sorry
end

end hyperbola_equation_l760_760654


namespace problem1_problem2_l760_760235

variables {a b c : ℝ}
variables {A B C : ℝ} -- representing the angles

-- Vector m and n definitions
def m := (a + c, b)
def n := (a - c, b - a)

-- Given conditions
variables (h1 : (a + c) * (a - c) + b * (b - a) = 0)
variables (h2 : A + B + C = π)
variables (h3 : sin A * a = sin B * b = sin C * c)

-- The theorem statement
theorem problem1 (h1 : (a + c) * (a - c) + b * (b - a) = 0) : 
  C = π / 3 :=
sorry

theorem problem2 (h1 : (a + c) * (a - c) + b * (b - a) = 0) : 
  (A + B = 2 * π / 3) → 
  (dfrac{sqrt 3} 2 < sin A + sin B ≤ sqrt 3) :=
sorry

end problem1_problem2_l760_760235


namespace trapezoid_area_l760_760393

noncomputable def area_shaded_trapezoid : ℝ :=
  let base1 := 1.4
  let base2 := 3.73
  let height := 5
  (base1 + base2) * height / 2

theorem trapezoid_area :
  let squares := (3, 5, 7)
  let total_length := squares.1 + squares.2 + squares.3
  let max_height := squares.3
  let base_length := total_length
  (base1 := 1.4) (base2 := 3.73) (height := 5)
  area_shaded_trapezoid = 12.825 :=
by
  simp [area_shaded_trapezoid]
  norm_num
  sorry

end trapezoid_area_l760_760393


namespace factorize_polynomial_l760_760640

theorem factorize_polynomial (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := 
by sorry

end factorize_polynomial_l760_760640


namespace fish_caught_by_twentieth_fisherman_l760_760906

theorem fish_caught_by_twentieth_fisherman :
  ∀ (total_fishermen total_fish fish_per_fisherman nineten_fishermen : ℕ),
  total_fishermen = 20 →
  total_fish = 10000 →
  fish_per_fisherman = 400 →
  nineten_fishermen = 19 →
  (total_fishermen * fish_per_fisherman) - (nineten_fishermen * fish_per_fisherman) = 2400 :=
by
  intros
  sorry

end fish_caught_by_twentieth_fisherman_l760_760906


namespace incongruent_triangles_exist_with_properties_l760_760650

noncomputable theory

def is_relatively_prime (x y : ℕ) := Nat.gcd x y = 1

theorem incongruent_triangles_exist_with_properties :
  ∃ (r : ℤ) (a b c : ℕ), 
    (a = r^2 + 4) ∧ (b = r^4 + 3 * r^2 + 1) ∧ (c = r^4 + 4 * r^2 + 3) 
    ∧ (¬(r % 5 = 2 ∨ r % 5 = -2)) 
    ∧ is_relatively_prime a b ∧ is_relatively_prime a c ∧ is_relatively_prime b c 
    ∧ (∃ area : ℕ, ∃ altitude : ℝ, (area : ℝ) = 0.5 * (a + b - c) * altitude) 
    ∧ ¬(∃ h : ℕ,  altitude = h) :=
begin 
  sorry 
end

end incongruent_triangles_exist_with_properties_l760_760650


namespace exists_lambda_smallest_possible_lambda_l760_760810

open Real

variable {a b c : ℝ}
variable (hab : 0 < a ∧ 0 < b ∧ 0 < c)
variable (d := min ((a-b)^2) (min ((b-c)^2) ((c-a)^2)))

theorem exists_lambda 
  (h: ∃ (λ : ℝ), 0 < λ ∧ λ < 1 ∧ d ≤ λ * (a^2 + b^2 + c^2)) : 
  ∃ (λ : ℝ), λ = (2/3) := 
by
  sorry

theorem smallest_possible_lambda
  (h: ∀ (λ : ℝ), 0 < λ ∧ λ < 1 → d ≤ λ * (a^2 + b^2 + c^2)) : 
  ∃ (λ : ℝ), λ = (1/5) := 
by
  sorry

end exists_lambda_smallest_possible_lambda_l760_760810


namespace sum_prime_numbers_less_than_twenty_l760_760028

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760028


namespace sum_of_inverses_l760_760614

def f (x : ℝ) : ℝ :=
if x ≤ 2 then 3 * x + 1 else x^2

def g_inv (y : ℝ) : ℝ := (y - 1) / 3

def h_inv (y : ℝ) : ℝ := real.sqrt y

def f_inv (x : ℝ) : ℝ :=
if x ≤ 7 then g_inv x else h_inv x

theorem sum_of_inverses :
  (∑ x in {-5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : finset ℝ, f_inv x) = 22 + 2 * real.sqrt 2 :=
by sorry

end sum_of_inverses_l760_760614


namespace train_crossing_platform_time_l760_760953

theorem train_crossing_platform_time
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_signal_pole : ℝ)
  (speed : ℝ)
  (time_platform_cross : ℝ)
  (v := length_train / time_signal_pole)
  (d := length_train + length_platform)
  (t := d / v) :
  length_train = 300 →
  length_platform = 250 →
  time_signal_pole = 18 →
  time_platform_cross = 33 →
  t = time_platform_cross := by
  sorry

end train_crossing_platform_time_l760_760953


namespace step_difference_propagation_l760_760917

noncomputable def sum_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem step_difference_propagation (n : ℕ) (hn : n = 2011) :
  let initial_sum := sum_natural_numbers n in
  even initial_sum →
  ∀ last_integer, (∀ (k : ℕ), (k < n → (0 ≤ k ∧ k ≤ n ∧ k ≠ last_integer))) →
  last_integer ≠ 1 :=
by
  let initial_sum := sum_natural_numbers 2011
  have h_initial_sum_even : even initial_sum
  {
    rw sum_natural_numbers
    dsimp [initial_sum]
    norm_num
  }
  intro h
  intro last_integer last_integer_constraint
  have h_last_integer_even : even last_integer
  {
    apply even_of_sum even initial_sum -- Need an appropriate lemma/application
  }
  exact ne_of_even_of_odd h_last_integer_even one_even_false
  sorry

end step_difference_propagation_l760_760917


namespace greatest_two_digit_multiple_of_17_l760_760473

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760473


namespace american_literature_marks_l760_760322

variable (History HomeEconomics PhysicalEducation Art AverageMarks NumberOfSubjects TotalMarks KnownMarks : ℕ)
variable (A : ℕ)

axiom marks_history : History = 75
axiom marks_home_economics : HomeEconomics = 52
axiom marks_physical_education : PhysicalEducation = 68
axiom marks_art : Art = 89
axiom average_marks : AverageMarks = 70
axiom number_of_subjects : NumberOfSubjects = 5

def total_marks (AverageMarks NumberOfSubjects : ℕ) : ℕ := AverageMarks * NumberOfSubjects

def known_marks (History HomeEconomics PhysicalEducation Art : ℕ) : ℕ := History + HomeEconomics + PhysicalEducation + Art

axiom total_marks_eq : TotalMarks = total_marks AverageMarks NumberOfSubjects
axiom known_marks_eq : KnownMarks = known_marks History HomeEconomics PhysicalEducation Art

theorem american_literature_marks :
  A = TotalMarks - KnownMarks := by
  sorry

end american_literature_marks_l760_760322


namespace juwella_read_more_last_night_l760_760636

-- Definitions of the conditions
def pages_three_nights_ago : ℕ := 15
def book_pages : ℕ := 100
def pages_tonight : ℕ := 20
def pages_two_nights_ago : ℕ := 2 * pages_three_nights_ago
def total_pages_before_tonight : ℕ := book_pages - pages_tonight
def pages_last_night : ℕ := total_pages_before_tonight - pages_three_nights_ago - pages_two_nights_ago

theorem juwella_read_more_last_night :
  pages_last_night - pages_two_nights_ago = 5 :=
by
  sorry

end juwella_read_more_last_night_l760_760636


namespace sum_series_binomial_l760_760525

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

noncomputable def S_n (n : ℕ) : ℕ :=
  ∑ k in finset.range (n + 1), (k * (k + 1) * binomial_coefficient n k)

theorem sum_series_binomial (n : ℕ) (hn : n ≥ 4) :
  S_n n = n * (n + 3) * 2^(n - 2) :=
by {
  sorry
}

end sum_series_binomial_l760_760525


namespace eight_valid_squares_exist_l760_760105

open Matrix

def is_valid_square (A : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  (A 0 0 + A 0 1 + A 0 2 + A 1 0 + A 1 1 + A 1 2) = (A 2 0 + A 2 1 + A 2 2)

def valid_squares (squares : Fin 8 → Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i : Fin 8, is_valid_square (squares i) 

theorem eight_valid_squares_exist : 
  ∃ squares : Fin 8 → Matrix (Fin 3) (Fin 3) ℕ,
  valid_squares squares ∧ 
  (∀ i : Fin 8, ∀ (i1 i2 : Fin 3), ∑ j : Fin 3, (squares i) i1 j ≠ 162) ∧
  ∀ i : Fin 8, 
  ∀ (i1 i2 : Fin 3), 
  (i1 ≠ i2 → ∑ j : Fin 3, (squares i) (i1) j = ∑ j : Fin 3, (squares i) (i2) j) ∧
  (squares i).to_finset = (Finset.univ : Finset (Fin 3 → Fin 3)) :=
sorry

end eight_valid_squares_exist_l760_760105


namespace circle_radius_k_l760_760193

theorem circle_radius_k (k : ℝ) : (∃ x y : ℝ, (x^2 + 14*x + y^2 + 8*y - k = 0) ∧ ((x + 7)^2 + (y + 4)^2 = 100)) → k = 35 :=
by
  sorry

end circle_radius_k_l760_760193


namespace seating_arrangement_l760_760265

-- Definitions according to conditions
def num_players : ℕ := 10
def num_teams : ℕ := 4
def team_counts : List ℕ := [3, 3, 2, 2]        -- Number of players per team (Celtics, Lakers, Warriors, Nuggets)
def specific_warrior_pos : ℕ := 1               -- Specific Warrior must sit at the position 1 (left end)

-- Proof problem statement based on the mathematically equivalent proof problem
theorem seating_arrangement :
  (3! * 1 * 3! * 3! * 2!) = 432 :=
by
  sorry

end seating_arrangement_l760_760265


namespace arrangement_possible_l760_760278

theorem arrangement_possible :
  ∃ (perm : List ℕ), (∀ i, i < 99 → 50 ≤ abs ((perm.get i) - (perm.get (i+1))))
                       ∧ (perm ~ List.range' 1 100) :=
by
  sorry

end arrangement_possible_l760_760278


namespace turtle_feeding_cost_l760_760347

theorem turtle_feeding_cost (total_weight_pounds : ℝ) (food_per_half_pound : ℝ)
  (jar_food_ounces : ℝ) (jar_cost_dollars : ℝ) (total_cost : ℝ) : 
  total_weight_pounds = 30 →
  food_per_half_pound = 1 →
  jar_food_ounces = 15 →
  jar_cost_dollars = 2 →
  total_cost = 8 :=
by
  intros h_weight h_food h_jar_ounces h_jar_cost
  sorry

end turtle_feeding_cost_l760_760347


namespace martha_bedroom_size_l760_760882

theorem martha_bedroom_size (x jenny_size total_size : ℤ) (h₁ : jenny_size = x + 60) (h₂ : total_size = x + jenny_size) (h_total : total_size = 300) : x = 120 :=
by
  -- Adding conditions and the ultimate goal
  sorry


end martha_bedroom_size_l760_760882


namespace no_number_equals_sum_of_others_l760_760324

noncomputable def repeating_digit_number (i k : ℕ) : ℕ := i * (10^k - 1) / 9

theorem no_number_equals_sum_of_others :
  ∀ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ),
  let n₁ := repeating_digit_number 1 a₁,
      n₂ := repeating_digit_number 2 a₂,
      n₃ := repeating_digit_number 3 a₃,
      n₄ := repeating_digit_number 4 a₄,
      n₅ := repeating_digit_number 5 a₅,
      n₆ := repeating_digit_number 6 a₆,
      n₇ := repeating_digit_number 7 a₇,
      n₈ := repeating_digit_number 8 a₈,
      n₉ := repeating_digit_number 9 a₉,
      s := n₁ + n₂ + n₃ + n₄ + n₅ + n₆ + n₇ + n₈ + n₉
  in (∀ n ∈ [n₁, n₂, n₃, n₄, n₅, n₆, n₇, n₈, n₉], n ≠ s - n) :=
by sorry

end no_number_equals_sum_of_others_l760_760324


namespace last_integer_not_one_l760_760918

theorem last_integer_not_one : 
  (∃ f : ℕ → ℤ, (∀ n, (1 ≤ n + 2 → 1 ≤ f n ≤ 2011))
     → (∀ n, (f (n + 1) = (f n) - (f n)))
     → (∀ k, 0 ≤ k ≤ 2010 → ∃ first_second_diff (f : ℕ → ℤ), 
       f (k + 1) = f k - f first_second_diff))
     → (f 2010 ≠ 1)) :=
begin
  sorry
end

end last_integer_not_one_l760_760918


namespace sum_coordinates_eq_14_l760_760710

def f (x : ℝ) : ℝ

theorem sum_coordinates_eq_14 (h : f 4 = 7) : ∃ (x y : ℝ), (2 * y = 3 * f (4 * x) + 5) ∧ (x = 1) ∧ (y = 13) ∧ (x + y = 14) :=
by {
  use [1, 13],
  split,
  { rw [mul_comm 4 1, mul_comm 3 7, ←h, mul_add],
    norm_num },
  split,
  { refl },
  split,
  { refl },
  norm_num 
}

end sum_coordinates_eq_14_l760_760710


namespace find_shorter_segment_length_l760_760276

-- Definitions from conditions
variables {X Y Z M : Type}
variables {XY YZ ZX : ℝ}
variable {XM_bisects_largest_angle_at_X : Prop}
variable {sides_ratio : Prop}
variable {length_YZ : ℝ}
variable {MY MZ : ℝ}

-- Conditions
def triangle_sides_ratio: Prop := XY / YZ = 3 / 4 ∧ YZ / ZX = 4 / 5
def XM_angle_bisector : Prop := XM_bisects_largest_angle_at_X
def length_side_YZ : Prop := length_YZ = 12

-- Prove the length of the shorter segment MY
theorem find_shorter_segment_length (h1 : triangle_sides_ratio)
                                    (h2 : XM_angle_bisector)
                                    (h3 : length_side_YZ) :
                                    ∃ MY MZ, MY + MZ = length_YZ ∧ MY = 9 / 2 :=
begin
  -- Proof would go here
  sorry
end

end find_shorter_segment_length_l760_760276


namespace sum_primes_less_than_20_l760_760064

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760064


namespace greatest_two_digit_multiple_of_17_is_85_l760_760458

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760458


namespace daily_rate_first_week_l760_760359

-- Definitions from given conditions
variable (x : ℝ) (h1 : ∀ y : ℝ, 0 ≤ y)
def cost_first_week := 7 * x
def additional_days_cost := 16 * 14
def total_cost := cost_first_week + additional_days_cost

-- Theorem to solve the problem
theorem daily_rate_first_week (h : total_cost = 350) : x = 18 :=
sorry

end daily_rate_first_week_l760_760359


namespace complement_of_A_in_U_l760_760220

theorem complement_of_A_in_U :
  let U := set.univ
  let A := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
  ∀ x : ℝ, (x ∈ set.univ \ A ↔ x < -1 ∨ x > 3) :=
by
  intro x
  let U := set.univ
  let A := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
  calc
  x ∈ set.univ \ A ↔ ¬(x ∈ A) : Iff.rfl
               ... ↔ ¬(x^2 - 2 * x - 3 ≤ 0) : Iff.rfl
               ... ↔ x < -1 ∨ x > 3 : sorry

end complement_of_A_in_U_l760_760220


namespace sum_prime_numbers_less_than_twenty_l760_760039

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760039


namespace center_of_sphere_diameter_l760_760362

theorem center_of_sphere_diameter (A B : ℝ × ℝ × ℝ) (hA : A = (2, -3, 4)) (hB : B = (-6, 5, 10)) :
  let C : ℝ × ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)
  in C = (-2, 1, 7) :=
by
  sorry

end center_of_sphere_diameter_l760_760362


namespace integral_cosine_l760_760775

noncomputable def a : ℝ := 2 * Real.pi / 3

theorem integral_cosine (ha : a = 2 * Real.pi / 3) :
  ∫ x in -a..a, Real.cos x = Real.sqrt 3 := 
sorry

end integral_cosine_l760_760775


namespace isosceles_base_length_l760_760769

theorem isosceles_base_length (x b : ℕ) (h1 : 2 * x + b = 40) (h2 : x = 15) : b = 10 :=
by
  sorry

end isosceles_base_length_l760_760769


namespace greatest_two_digit_multiple_of_17_is_85_l760_760456

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760456


namespace part_b_l760_760260

-- Definitions for part (a)
variable {α : Type*} [Preorder α] {φ : α → ℂ} (cond1 : φ 0 = 1)
def non_negative_def (φ : α → ℂ) : Prop := ∀ t, 0 ≤ φ t

-- Proof statement for part (a)
lemma part_a {s t : α} (h : non_negative_def φ) :
  |φ t - φ s|^2 ≤ 2 * (1 - ℜ (φ (t - s))) :=
sorry

-- Definitions for part (b)
def lebesgue_measurable (f : ℝ → ℂ) : Prop := -- Add precise Lean definition for Lebesgue measurability
def positively_defined (φ : ℝ → ℂ) : Prop := -- Add precise Lean definition for positively defined
def characteristic_function (φ : ℝ → ℂ) : Prop := -- Add precise Lean definition for characteristic function

-- Proof statement for part (b)
theorem part_b {φ : ℝ → ℂ} (h : lebesgue_measurable φ) (h0 : φ 0 = 1) :
  positively_defined φ ↔ ∃ χ, characteristic_function χ ∧ φ = χ :=
sorry

end part_b_l760_760260


namespace range_of_x_plus_y_l760_760676

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y - (x + y) = 1) : 
  x + y ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end range_of_x_plus_y_l760_760676


namespace integral_identity_l760_760839

open Real

noncomputable def integral_equiv (a : ℝ) (x : ℝ) : Prop :=
  ∫ (1 / sqrt (x^2 + a)) dx = ln (abs (x + sqrt (x^2 + a))) + C

theorem integral_identity (a : ℝ) (h : 0 < a) : 
  integral_equiv a x := 
by
  sorry

end integral_identity_l760_760839


namespace sum_prime_numbers_less_than_twenty_l760_760029

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760029


namespace modulus_of_transformed_product_l760_760813

noncomputable def modulus_property (z w : ℂ) : Prop :=
  |z| = 3 ∧ (z + conj w) * (conj z - w) = 7 + 4 * I

theorem modulus_of_transformed_product (z w : ℂ) (h : modulus_property z w) :
  |(z + 2 * conj w) * (conj z - 2 * w)| = Real.sqrt 65 :=
by
  sorry

end modulus_of_transformed_product_l760_760813


namespace books_in_bin_after_transactions_l760_760943

def initial_books : ℕ := 4
def sold_books : ℕ := 3
def added_books : ℕ := 10

def final_books (initial_books sold_books added_books : ℕ) : ℕ :=
  initial_books - sold_books + added_books

theorem books_in_bin_after_transactions :
  final_books initial_books sold_books added_books = 11 := by
  sorry

end books_in_bin_after_transactions_l760_760943


namespace purely_imaginary_z_l760_760221

theorem purely_imaginary_z (a : ℝ) :
  (let z := (a + complex.I) / (1 - complex.I) in z.im = z ∧ z.re = 0) → a = 1 :=
by
  intro h
  sorry

end purely_imaginary_z_l760_760221


namespace combinations_b_not_eq_a_l760_760838

-- Definitions of sets a and b
def set_a := {1, 2, 3, 4, 5}
def set_b := {1, 2, 3}

-- Lean statement of the equivalent proof problem
theorem combinations_b_not_eq_a :
  (∃! n : ℕ, n = (set_a.to_finset.sum (λ a, set_b.to_finset.card - ite (a ∈ set_b.to_finset) 1 0)) ∧ n = 12) :=
by sorry

end combinations_b_not_eq_a_l760_760838


namespace julie_hours_per_week_l760_760285

variables (w_s : ℕ) (d_s : ℕ) (e_s : ℕ) (e_y : ℕ) (d_y : ℕ)

theorem julie_hours_per_week (h1 : w_s = 60) (h2 : d_s = 10) (h3 : e_s = 7500) (h4 : e_y = 9000) (h5 : d_y = 40) :
  (e_s / (w_s * d_s)) = (e_y / (w_y * d_y)) → w_y = 18 :=
by
sorrry

end julie_hours_per_week_l760_760285


namespace greatest_two_digit_multiple_of_17_l760_760410

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760410


namespace sum_primes_less_than_20_l760_760059

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760059


namespace third_house_price_l760_760601

variable (house1_price : ℝ) (house2_price : ℝ) -- prices of the first two houses
variable (total_commission : ℝ) -- total commission Brian earned
variable (commission_rate : ℝ) -- commission rate

-- Define the values based on the problem statement
def house1_price := 157000
def house2_price := 499000
def total_commission := 15620
def commission_rate := 0.02

-- Calculate the individual commissions
def commission1 := house1_price * commission_rate
def commission2 := house2_price * commission_rate

-- Total commission from the first two houses
def total_commission_first_two := commission1 + commission2

-- Commission from the third house
def commission3 := total_commission - total_commission_first_two

-- Selling price of the third house
def selling_price3 := commission3 / commission_rate

-- The theorem we need to prove
theorem third_house_price :
  selling_price3 = 125000 :=
by
  sorry

end third_house_price_l760_760601


namespace fraction_of_passengers_using_kennedy_airport_l760_760760

theorem fraction_of_passengers_using_kennedy_airport :
  let K := 12.433333333333332
  let total_passengers := 37.3
  (K / total_passengers) = (1 / 3) :=
by
  let K := 12.433333333333332
  let total_passengers := 37.3
  have h : (K / total_passengers) = 1 / 3, from sorry  -- Proof calculation goes here
  exact h

end fraction_of_passengers_using_kennedy_airport_l760_760760


namespace sqrt_equation_l760_760325

theorem sqrt_equation (n : ℕ) (h : 0 < n) : 
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / (n * (n + 1) : ℝ) :=
sorry

end sqrt_equation_l760_760325


namespace max_value_of_m_l760_760804

open Set Nat

theorem max_value_of_m {A B : ℕ → Set ℕ}
  (hA : ∀ i : ℕ, finite (A i) ∧ Card (A i) = 2012)
  (hB : ∀ i : ℕ, finite (B i) ∧ Card (B i) = 2013)
  (hDisjoint : ∀ i j : ℕ, A i ∩ B j = ∅ ↔ i = j)
  : ∃ m : ℕ, ∀ i : ℕ, i < m → Card (Finset.univ.filter (λ i, A i ∪ B i).toFinset) = binom 4025 2012 :=
by
  sorry

end max_value_of_m_l760_760804


namespace dot_product_of_PA_PB_l760_760216

theorem dot_product_of_PA_PB
  (A B P: ℝ × ℝ)
  (h_circle : ∀ (x y : ℝ), x ^ 2 + y ^ 2 + 4 * x - 5 = 0 → (x, y) = A ∨ (x, y) = B)
  (h_midpoint : (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 1)
  (h_x_axis_intersect : P.2 = 0 ∧ (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = -5) :
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = -5 :=
sorry

end dot_product_of_PA_PB_l760_760216


namespace path_to_tile_ratio_l760_760562

theorem path_to_tile_ratio
  (t p : ℝ) 
  (tiles : ℕ := 400)
  (grid_size : ℕ := 20)
  (total_tile_area : ℝ := (tiles : ℝ) * t^2)
  (total_courtyard_area : ℝ := (grid_size * (t + 2 * p))^2) 
  (tile_area_fraction : ℝ := total_tile_area / total_courtyard_area) : 
  tile_area_fraction = 0.25 → 
  p / t = 0.5 :=
by
  intro h
  sorry

end path_to_tile_ratio_l760_760562


namespace number_of_speaking_orders_l760_760555

-- The conditions of the problem:
variables (A B C D E F G H : Type) (select : Finset (A ⊕ B ⊕ C ⊕ D ⊕ E ⊕ F ⊕ G ⊕ H)) 
  (at_least_one_of_AB : (A ∈ select ∨ B ∈ select))
  (AB_exactly_one_between : (A ∈ select ∧ B ∈ select) → ∃ x ∈ select, (ensure_x_between_A_and_B : true))  -- Placeholder, higher-order logic needed

-- The goal, which is the mathematically equivalent proof problem:
theorem number_of_speaking_orders : Finset.card select = 1080 := 
sorry

end number_of_speaking_orders_l760_760555


namespace median_category_a_median_category_b_l760_760891

-- Conditions for Category A
def category_a_times_in_seconds : List ℕ := [
  25, 30, 45, 50, 65, 70, 75, 100, 125, 135, 140, 150
]

-- Conditions for Category B
def category_b_times_in_seconds : List ℕ := [
  115, 118, 120, 122, 135, 150, 200, 205, 288, 250, 255
]

-- Definitions for medians
def median (l : List ℕ) : ℕ :=
  if h : l.length % 2 = 0 then
    (l.nthLe (l.length / 2 - 1) (by linarith) + l.nthLe (l.length / 2) (by linarith)) / 2
  else
    l.nthLe (l.length / 2) (by linarith)

-- Proof Statements
theorem median_category_a : median category_a_times_in_seconds = 137 := by
  sorry

theorem median_category_b : median category_b_times_in_seconds = 150 := by
  sorry

end median_category_a_median_category_b_l760_760891


namespace principal_amount_l760_760256

theorem principal_amount (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (h1 : R = 4) 
  (h2 : T = 5) 
  (h3 : SI = P - 1920) 
  (h4 : SI = (P * R * T) / 100) : 
  P = 2400 := 
by 
  sorry

end principal_amount_l760_760256


namespace F_is_subspace_of_R3_l760_760159

def F : Set (ℝ × ℝ × ℝ) := {p | p.1 + p.2 - 2 * p.3 = 0}

theorem F_is_subspace_of_R3 : 
  ∃ (F : Set (ℝ × ℝ × ℝ)), 
    (∀ (u v : ℝ × ℝ × ℝ), u ∈ F ∧ v ∈ F → u + v ∈ F) ∧
    (∀ (u : ℝ × ℝ × ℝ) (c : ℝ), u ∈ F → c • u ∈ F) ∧
    (0, 0, 0) ∈ F :=
by
  use F
  split
  sorry
  split
  sorry
  sorry

end F_is_subspace_of_R3_l760_760159


namespace probability_multiple_of_45_l760_760099

def multiples_of_3 := [3, 6, 9]
def primes_less_than_20 := [2, 3, 5, 7, 11, 13, 17, 19]

def favorable_outcomes := (9, 5)
def total_outcomes := (multiples_of_3.length * primes_less_than_20.length)

theorem probability_multiple_of_45 : (multiples_of_3.length = 3 ∧ primes_less_than_20.length = 8) → 
  ∃ w : ℚ, w = 1 / 24 :=
by {
  sorry
}

end probability_multiple_of_45_l760_760099


namespace olivia_accident_count_l760_760326

theorem olivia_accident_count
  (initial_premium : ℕ) (ticket_increase : ℕ) (accident_increase_rate : ℝ) (num_tickets : ℕ) (final_premium : ℕ) (A : ℕ) :
  initial_premium = 50 →
  ticket_increase = 5 →
  accident_increase_rate = 0.10 →
  num_tickets = 3 →
  final_premium = 70 →
  final_premium = initial_premium + num_tickets * ticket_increase + A * (accident_increase_rate * initial_premium).toNat →
  A = 1 :=
by
  intros h_initial h_ticket h_accident_rate h_tickets h_final h_eq
  sorry

end olivia_accident_count_l760_760326


namespace complex_number_in_second_quadrant_l760_760556

theorem complex_number_in_second_quadrant (z : ℂ) (h : (2 - complex.I) * z = complex.I) :
  z.re < 0 ∧ 0 < z.im :=
sorry

end complex_number_in_second_quadrant_l760_760556


namespace greatest_two_digit_multiple_of_17_l760_760479

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760479


namespace greatest_two_digit_multiple_of_17_l760_760454

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760454


namespace true_speed_of_train_l760_760549

-- Definitions based on conditions
def train_length : ℝ := 200
def crossing_time : ℝ := 20
def wind_speed : ℝ := 5

-- Proof goal
theorem true_speed_of_train : 
  let apparent_speed := train_length / crossing_time in
  let true_speed := apparent_speed + wind_speed in
  true_speed = 15 := by
  sorry

end true_speed_of_train_l760_760549


namespace identical_monochromatic_sequences_l760_760106

theorem identical_monochromatic_sequences
  (n : ℕ) (h_even : even (2 * n)) 
  (red_indices blue_indices : finset ℕ)
  (h_disjoint : disjoint red_indices blue_indices)
  (h_card_red : red_indices.card = n) 
  (h_card_blue : blue_indices.card = n)
  (polygon : list ℝ) 
  (h_polygon_len : polygon.length = 2 * n)
  (h_polygon_regular : ∀ i j, polygon.nth i = polygon.nth j → i ≡ j [MOD (2 * n)])
  : (let red_segments := ((red_indices.product red_indices).filter (λ p, p.1 < p.2)).image (λ p, dist (polygon.nth p.1) (polygon.nth p.2)),
         blue_segments := ((blue_indices.product blue_indices).filter (λ p, p.1 < p.2)).image (λ p, dist (polygon.nth p.1) (polygon.nth p.2)) in
     multiset.sort (≤) red_segments = multiset.sort (≤) blue_segments) :=
sorry

end identical_monochromatic_sequences_l760_760106


namespace correct_differentiation_operation_l760_760528

theorem correct_differentiation_operation:
  (∀ x : ℝ, (x + 1/x)' = 1 - 1/x^2) = False ∧
  (∀ x : ℝ, (x^2 * cos x)' = 2 * x * cos x - x^2 * sin x) = False ∧
  (∀ x : ℝ, (3^x)' = 3^x * log 3) = False ∧
  (∀ x : ℝ, (log 2 x)' = 1 / (x * log (2 : ℝ))) = True := by
sorry

end correct_differentiation_operation_l760_760528


namespace factorization_correct_inequality_proof_l760_760948

theorem factorization_correct :
  ∀ x : ℂ, x^12 + x^9 + x^6 + x^3 + 1 =
  ((x^2 - (sqrt 5 - 1)/2 * x + 1) *
   (x^2 + (sqrt 5 + 1)/2 * x + 1) *
   (x^2 - (sqrt 5 + 1 + sqrt(30 - 6 * sqrt 5))/4 * x + 1) *
   (x^2 - (sqrt 5 + 1 - sqrt(30 - 6 * sqrt 5))/4 * x + 1) *
   (x^2 + (sqrt 5 - 1 + sqrt(30 + 6 * sqrt 5))/4 * x + 1) *
   (x^2 + (sqrt 5 - 1 - sqrt(30 + 6 * sqrt 5))/4 * x + 1)) :=
sorry

theorem inequality_proof (θ : ℝ) : 
  5 + 8 * cos θ + 4 * cos (2 * θ) + cos (3 * θ) ≥ 0 :=
sorry

end factorization_correct_inequality_proof_l760_760948


namespace target_annual_revenue_l760_760130

-- Given conditions as definitions
def monthly_sales : ℕ := 4000
def additional_sales : ℕ := 1000

-- The proof problem in Lean statement form
theorem target_annual_revenue : (monthly_sales + additional_sales) * 12 = 60000 := by
  sorry

end target_annual_revenue_l760_760130


namespace percentage_difference_l760_760257

variable (x y z : ℝ)

theorem percentage_difference (h1 : y = 1.75 * x) (h2 : z = 0.60 * y) :
  (1 - x / z) * 100 = 4.76 :=
by
  sorry

end percentage_difference_l760_760257


namespace volume_of_parallelepiped_l760_760232

theorem volume_of_parallelepiped (a : ℝ) 
    (h1 : 0 < a)
    (h2 : ∀ p1 p2 p3 : ℝ^3, orthogonal p1 p2 ∧ orthogonal p2 p3 ∧ dist p1 p2 = a ∧ dist p2 p3 = a ∧ dist p3 p1 = a)
    : volume_of_parallelepiped a = 9 * a ^ 3 :=
sorry

end volume_of_parallelepiped_l760_760232


namespace domain_f_eq_l760_760360

def f (x : ℝ) : ℝ := real.sqrt (-x^2 + 2 * x + 3)

theorem domain_f_eq : 
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0} = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } := 
by
  sorry

end domain_f_eq_l760_760360


namespace smallest_base_is_11_l760_760922

noncomputable def smallest_base := 
  Nat.find (λ b => b > 3 ∧ ∃ n, 2 * b + 3 = n^2)

theorem smallest_base_is_11 : smallest_base = 11 := 
by
    sorry

end smallest_base_is_11_l760_760922


namespace fourth_root_of_25000000_eq_70_7_l760_760152

theorem fourth_root_of_25000000_eq_70_7 :
  Real.sqrt (Real.sqrt 25000000) = 70.7 :=
sorry

end fourth_root_of_25000000_eq_70_7_l760_760152


namespace sum_of_coordinates_l760_760829

theorem sum_of_coordinates (x : ℝ) : 
  let A := (x, 7)
  let B := (x, -7)
  (A.1 + A.2 + B.1 + B.2) = 2 * x := 
by
  let A := (x, 7)
  let B := (x, -7)
  calc
    A.1 + A.2 + B.1 + B.2 
        = x + 7 + x - 7 : by rw [add_assoc, add_right_neg, add_zero]
    ... = 2 * x : by ring

end sum_of_coordinates_l760_760829


namespace greatest_two_digit_multiple_of_17_l760_760451

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760451


namespace greatest_two_digit_multiple_of_17_l760_760440

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760440


namespace curveC_eq_rect_parametric_line_eqn_area_triangle_GAB_l760_760849

noncomputable def curveC_rect_eqn := y ^ 2 = 8 * x

def line_param_eqn (α : ℝ) :=
  (λ t : ℝ, (2 + t * Real.cos α, t * Real.sin α))

def point_G := (-2 : ℝ, 0 : ℝ)

def point_Q := (0 : ℝ, -2 : ℝ)

theorem curveC_eq_rect :
  ∀ θ ρ, ρ * Real.sin θ ^ 2 - 8 * Real.cos θ = 0 ↔ y ^ 2 = 8 * x := sorry

theorem parametric_line_eqn :
  line_param_eqn (Real.pi / 4) = (λ t, (2 + t / Real.sqrt 2, t / Real.sqrt 2)) := sorry

theorem area_triangle_GAB (α : ℝ) :
  α = Real.pi / 4 →
  let l := line_param_eqn α
  let Q_ab := (λ t, l t) -- intersection points between line and curve
  let AB := Q_ab (t₂) - Q_ab (t₁)
  let d := (abs (-2 * 0 - 2) / (Real.sqrt 2))
  abs(t₂ - t₁) = 16 →
  (1 / 2) * AB * (2 * d) = 16 * Real.sqrt 2 := sorry

end curveC_eq_rect_parametric_line_eqn_area_triangle_GAB_l760_760849


namespace distance_inequality_l760_760942

variable {α : Type*} [InnerProductSpace ℝ α]

variables (P A B C : α) (pa pb pc : ℝ)

theorem distance_inequality
  (hPA : dist P A = pa)
  (hPB : dist P B = pb)
  (hPC : dist P C = pc) :
  pa * pb + pb * pc + pc * pa ≥ (pb + pc) * (pa + pc) + (pb + pc) * (pa + pb) + (pa + pc) * (pa + pb) :=
by sorry

end distance_inequality_l760_760942


namespace exponential_generating_function_formula_l760_760343

noncomputable def exponential_generating_function (P : ℕ → ℝ) (x : ℝ) : ℝ :=
  ∑' (N : ℕ), P N * x^N / (N.factorial)

def P_N (N : ℕ) : ℕ → ℝ :=
  λ n, ∑ i in Finset.range(N+1), 1 / i.factorial

theorem exponential_generating_function_formula :
  ∀ (P : ℕ → (ℕ → ℝ)) (x : ℝ), 
    (∀ N, P N = λ n, ∑ i in Finset.range(N+1), 1 / i.factorial) → 
    exponential_generating_function (λ N, P N N) x = (Real.exp x) / (1 - x) :=
by
  intros P x h
  sorry

end exponential_generating_function_formula_l760_760343


namespace cd_ef_df_ae_eq_bd_af_l760_760313

noncomputable theory
open_locale classical

variables {A B C D E F : Type*}
variables [noncomputable_field A B C D E F]

structure IsoscelesTriangle (A B C : Type*) :=
(base : B = C)
(vertex : A)
(isosceles : ∀ pt1 pt2 : Type*, (pt1 = A ∧ pt2 ≠ A) ∨ (pt2 = A ∧ pt1 ≠ A))

def on_arc (A D C F : Type*) : Prop := -- definition for point F on arc passing through A, D, and C
-- Assume circuference properties or angle relationships needed for proof
sorry

def circle_passing_through (pts : set (Type*)) : Prop := 
-- Assume all properties about circles passing through given points as needed for proof
sorry

def intersects (circle side : Type*) : Prop := -- definition for intersection
-- Assume properties of intersection as needed for proof
sorry

theorem cd_ef_df_ae_eq_bd_af 
  (ht : IsoscelesTriangle A B C)
  (hD : D = ht.base) 
  (hF : on_arc A D C F) 
  (circleBDF : circle_passing_through {B, D, F}) 
  (hE : intersects circleBDF AB) : 
  CD * EF + DF * AE = BD * AF :=
sorry

end cd_ef_df_ae_eq_bd_af_l760_760313


namespace greatest_two_digit_multiple_of_17_l760_760474

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760474


namespace sin_over_sin3_plus_cos3_l760_760197

variable {θ : ℝ}

-- Given condition
def tan_theta : Prop := Real.tan θ = 2

-- Assertion to prove
theorem sin_over_sin3_plus_cos3 (h : tan_theta) : (Real.sin θ / (Real.sin θ ^ 3 + Real.cos θ ^ 3)) = 10 / 9 :=
by
  sorry

end sin_over_sin3_plus_cos3_l760_760197


namespace algebraic_expression_l760_760175

-- Definition for the problem expressed in Lean
def number_one_less_than_three_times (a : ℝ) : ℝ :=
  3 * a - 1

-- Theorem stating the proof problem
theorem algebraic_expression (a : ℝ) : number_one_less_than_three_times a = 3 * a - 1 :=
by
  -- Proof steps would go here; omitted as per instructions
  sorry

end algebraic_expression_l760_760175


namespace uncle_fyodor_wins_l760_760387

theorem uncle_fyodor_wins (N : ℕ) : ∀ sandwiches_with_sausage : fin (100 * N) → bool,
  (∃ sequence_of_moves : fin 101 → fin (100 * N), 
    (∀ i, (sequence_of_moves i) ∈ {0, 100 * N - 1}) ∧
    sandwiches_with_sausage (sequence_of_moves 100) = true) :=
by sorry

end uncle_fyodor_wins_l760_760387


namespace sum_of_squares_l760_760740

theorem sum_of_squares (m n : ℝ) (h1 : m + n = 10) (h2 : m * n = 24) : m^2 + n^2 = 52 :=
by
  sorry

end sum_of_squares_l760_760740


namespace sum_of_digits_of_n_eq_4_l760_760905

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem sum_of_digits_of_n_eq_4 (n : ℕ) (h1 : 0 < n) (h2: (factorial (n+1)) + 2 * factorial (n+2) = factorial n * 456)  : n = 13 ∧ (1 + 3 = 4) :=
by 
  sorry

end sum_of_digits_of_n_eq_4_l760_760905


namespace find_valid_n_l760_760177

theorem find_valid_n : 
  ∀ (n : ℕ), 
    n ≥ 2 ∧ 
    (∃ (a : Fin n → ℝ), ∃ (r : ℝ), r > 0 ∧ 
      (∀ i j, (1 ≤ i) → (i < j) → (j ≤ n) → ∃ k, (0 < k) ∧ (k ≤ (n*(n-1))/2) ∧ (a j - a i = r^k))) ↔ 
    n = 2 ∨ n = 3 ∨ n = 4 :=
by
  sorry

end find_valid_n_l760_760177


namespace max_value_at_x_neg_2_l760_760928

def polynomial (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 16

theorem max_value_at_x_neg_2 : ∀ x : ℝ, polynomial x ≤ polynomial (-2) :=
by
  -- Let's denote the polynomial function as f(x)
  let f := polynomial
  -- We know from the problem statement that f(x) = -2 (x+2)^2 + 24
  have h₁ : ∀ x : ℝ, f x = -2 * (x + 2)^2 + 24 := sorry
  -- We also know that the maximum value of -2 (x+2)^2 + 24 is 24 and it occurs when (x+2)^2 = 0 -> x = -2
  have h₂ : ∀ x : ℝ, -2 * (x + 2)^2 ≤ 0 := by { intro x, nlinarith }
  have h₃ : ∀ x : ℝ, f x ≤ 24 := by { intro x, rw h₁, linarith [h₂ x] }
  -- Therefore, we need to prove that f(x) achieves its maximum value at x = -2
  intro x
  exact h₃ x
  sorry

end max_value_at_x_neg_2_l760_760928


namespace sequence_property_l760_760376

def seq_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 9
  | (n + 1) => (2 * (n + 1) - 1) * 2 * 3^(n + 1)

theorem sequence_property (n : ℕ) :
  (Finset.range (n + 1)).sum (λ k, seq_a k / (2 * (k + 1) - 1)) = 3^(n + 2) := 
sorry

end sequence_property_l760_760376


namespace yura_picture_dimensions_l760_760933

theorem yura_picture_dimensions (a b : ℕ) (h : (a + 2) * (b + 2) - a * b = a * b) :
  (a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) :=
by
  -- Place your proof here
  sorry

end yura_picture_dimensions_l760_760933


namespace dan_picked_l760_760995

-- Definitions:
def benny_picked : Nat := 2
def total_picked : Nat := 11

-- Problem statement:
theorem dan_picked (b : Nat) (t : Nat) (d : Nat) (h1 : b = benny_picked) (h2 : t = total_picked) (h3 : t = b + d) : d = 9 := by
  sorry

end dan_picked_l760_760995


namespace total_granola_bars_l760_760671

-- Problem conditions
def oatmeal_raisin_bars : ℕ := 6
def peanut_bars : ℕ := 8

-- Statement to prove
theorem total_granola_bars : oatmeal_raisin_bars + peanut_bars = 14 := 
by 
  sorry

end total_granola_bars_l760_760671


namespace total_lines_drawn_l760_760982

theorem total_lines_drawn 
  (triangles : ℕ) (squares : ℕ) (pentagons : ℕ) 
  (sides_triangle : ℕ) (sides_square : ℕ) (sides_pentagon : ℕ) :
  triangles = 12 → squares = 8 → pentagons = 4 → 
  sides_triangle = 3 → sides_square = 4 → sides_ppentagonion = 5 → 
  (triangles * sides_triangle + squares * sides_square + pentagons * sides_pentagon) = 88 :=
by 
  intro h_triangles h_squares h_pentagons h_sides_triangle h_sides_square h_sides_pentagon,
  have h1 : triangles * sides_triangle = 12 * 3, by rw [h_triangles, h_sides_triangle],
  have h2 : squares * sides_square = 8 * 4, by rw [h_squares, h_sides_square],
  have h3 : pentagons * sides_pentagon = 4 * 5, by rw [h_pentagons, h_sides_pentagon],
  have h4 : 12 * 3 = 36 := by norm_num,
  have h5 : 8 * 4 = 32 := by norm_num,
  have h6 : 4 * 5 = 20 := by norm_num,
  have h7 : 36 + 32 + 20 = 88 := by norm_num,
  rw [h1, h2, h3, h4, h5, h6],
  exact h7.

end total_lines_drawn_l760_760982


namespace community_members_after_five_years_l760_760114

theorem community_members_after_five_years:
  ∀ (a : ℕ → ℕ),
  a 0 = 20 →
  (∀ k : ℕ, a (k + 1) = 4 * a k - 15) →
  a 5 = 15365 :=
by
  intros a h₀ h₁
  sorry

end community_members_after_five_years_l760_760114


namespace turtle_feeding_cost_l760_760350

def cost_to_feed_turtles (turtle_weight: ℝ) (food_per_half_pound: ℝ) (jar_capacity: ℝ) (jar_cost: ℝ) : ℝ :=
  let total_food := turtle_weight * (food_per_half_pound / 0.5)
  let total_jars := total_food / jar_capacity
  total_jars * jar_cost

theorem turtle_feeding_cost :
  cost_to_feed_turtles 30 1 15 2 = 8 :=
by
  sorry

end turtle_feeding_cost_l760_760350


namespace greatest_two_digit_multiple_of_17_l760_760471

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l760_760471


namespace find_a_if_f_is_even_l760_760748

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * (Real.exp x - a / Real.exp x)

theorem find_a_if_f_is_even
  (h : ∀ x : ℝ, f x a = f (-x) a) : a = 1 :=
sorry

end find_a_if_f_is_even_l760_760748


namespace inequality_proofs_l760_760742

def sinSumInequality (A B C ε : ℝ) : Prop :=
  ε * (Real.sin A + Real.sin B + Real.sin C) ≤ Real.sin A * Real.sin B * Real.sin C + 1 + ε^3

def sinProductInequality (A B C ε : ℝ) : Prop :=
  (1 + ε + Real.sin A) * (1 + ε + Real.sin B) * (1 + ε + Real.sin C) ≥ 9 * ε * (Real.sin A + Real.sin B + Real.sin C)

theorem inequality_proofs (A B C ε : ℝ) (hA : 0 ≤ A ∧ A ≤ Real.pi) (hB : 0 ≤ B ∧ B ≤ Real.pi) 
  (hC : 0 ≤ C ∧ C ≤ Real.pi) (hε : ε ≥ 1) :
  sinSumInequality A B C ε ∧ sinProductInequality A B C ε :=
by
  sorry

end inequality_proofs_l760_760742


namespace correct_order_l760_760201

noncomputable def a : ℝ := Real.logBase 0.6 2
noncomputable def b : ℝ := Real.logBase 2 0.6
noncomputable def c : ℝ := 0.6 ^ 2

theorem correct_order (a_def : a = Real.logBase 0.6 2)
                      (b_def : b = Real.logBase 2 0.6)
                      (c_def : c = 0.6 ^ 2) :
  c > b ∧ b > a :=
by sorry

end correct_order_l760_760201


namespace find_number_plus_mod_divisible_l760_760540

theorem find_number_plus_mod_divisible (n : ℤ) : (n + 859722) % 456 = 0 ↔ n = 54 := 
begin 
  sorry
end

end find_number_plus_mod_divisible_l760_760540


namespace probability_ratio_l760_760843

-- Defining the total number of cards and each number's frequency
def total_cards := 60
def each_number_frequency := 4
def distinct_numbers := 15

-- Defining probability p' and q'
def p' := (15: ℕ) * (Nat.choose 4 4) / (Nat.choose 60 4)
def q' := 210 * (Nat.choose 4 3) * (Nat.choose 4 1) / (Nat.choose 60 4)

-- Prove the value of q'/p'
theorem probability_ratio : (q' / p') = 224 := by
  sorry

end probability_ratio_l760_760843


namespace greatest_two_digit_multiple_of_17_is_85_l760_760459

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760459


namespace three_digit_even_numbers_count_l760_760238

theorem three_digit_even_numbers_count : 
  let digits := {1, 2, 3, 4, 5}
  let even_digits := {2, 4}
  ∃ count : ℕ, 
  count = 24 ∧ 
  ∀ (num : ℕ), num ∈ digits → (num % 2 = 0) → 
  (2 ≤ count ∧ count ≤ 5 ∧ count = 24) :=
by
  sorry

end three_digit_even_numbers_count_l760_760238


namespace factorize_polynomial_l760_760637

theorem factorize_polynomial (x y : ℝ) : (3 * x^2 - 3 * y^2) = 3 * (x + y) * (x - y) := 
by
  sorry

end factorize_polynomial_l760_760637


namespace cyclic_sum_inequality_l760_760694

-- Defining the problem
theorem cyclic_sum_inequality (n : ℕ) (h : n > 3) (x : Fin n → ℝ) 
  (pos : ∀ i, 0 < x i) : 
  2 ≤ ∑ i, (x i) / (x (Fin.cast_add_one n.pred i) + x (Fin.cast_add_one n.succPred i)) :=
sorry

end cyclic_sum_inequality_l760_760694


namespace greatest_two_digit_multiple_of_17_l760_760499

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760499


namespace derivative_of_f_is_l760_760860

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2

theorem derivative_of_f_is (x : ℝ) : deriv f x = 2 * x + 2 :=
by
  sorry

end derivative_of_f_is_l760_760860


namespace area_triangle_inequality_l760_760914

variables (n k : ℕ)

-- condition: general position (no three points are collinear)
-- condition: k triangles of area 1 among n points
-- goal: Prove 3k ≤ 2n(n-1)

theorem area_triangle_inequality (h1 : ∀ (points : set (ℝ × ℝ)), points.card = n → points.general_position)
                                  (h2 : ∃ (triangles : set (set (ℝ × ℝ))), triangles.card = k ∧ ∀ t ∈ triangles, t.area = 1) :
  3 * k ≤ 2 * n * (n - 1) :=
sorry

end area_triangle_inequality_l760_760914


namespace frequency_of_blurred_pages_l760_760786

def crumpled_frequency : ℚ := 1 / 7
def total_pages : ℕ := 42
def neither_crumpled_nor_blurred : ℕ := 24

def blurred_frequency : ℚ :=
  let crumpled_pages := total_pages / 7
  let either_crumpled_or_blurred := total_pages - neither_crumpled_nor_blurred
  let blurred_pages := either_crumpled_or_blurred - crumpled_pages
  blurred_pages / total_pages

theorem frequency_of_blurred_pages :
  blurred_frequency = 2 / 7 := by
  sorry

end frequency_of_blurred_pages_l760_760786


namespace largest_three_digit_number_l760_760522

theorem largest_three_digit_number :
  ∃ (n : ℕ), (n < 1000) ∧ (n % 7 = 1) ∧ (n % 8 = 4) ∧ (∀ (m : ℕ), (m < 1000) ∧ (m % 7 = 1) ∧ (m % 8 = 4) → m ≤ n) :=
sorry

end largest_three_digit_number_l760_760522


namespace divide_into_parts_l760_760626

theorem divide_into_parts (x y : ℚ) (h_sum : x + y = 10) (h_diff : y - x = 5) : 
  x = 5 / 2 ∧ y = 15 / 2 := 
sorry

end divide_into_parts_l760_760626


namespace greatest_two_digit_multiple_of_17_l760_760409

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760409


namespace remainder_of_division_l760_760921

noncomputable def polynomial_remainder (p q : polynomial ℝ) : polynomial ℝ :=
  p % q

theorem remainder_of_division :
  polynomial_remainder (3 * X ^ 2 - 16 * X + 38) (X - 4) = 22 :=
sorry

end remainder_of_division_l760_760921


namespace min_value_expression_l760_760208

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  3 ≤ (1 / (x + y)) + ((x + y) / z) :=
by
  sorry

end min_value_expression_l760_760208


namespace albert_horses_l760_760983

variable {H C : ℝ}

theorem albert_horses :
  (2000 * H + 9 * C = 13400) ∧ (200 * H + 0.20 * 9 * C = 1880) ∧ (∀ x : ℝ, x = 2000) → H = 4 := 
by
  sorry

end albert_horses_l760_760983


namespace three_digit_number_divisible_by_11_l760_760924

theorem three_digit_number_divisible_by_11 : 
  (∀ x : ℕ, x < 10 → (600 + 10 * x + 3) % 11 = 0 → 600 + 10 * x + 3 = 693) :=
by 
  intros x x_lt_10 h 
  have h1 : 600 % 11 = 7 := by norm_num
  have h2 : (10 * x + 3) % 11 = (10 * x + 3) % 11 := by norm_num
  rw Nat.add_mod at h 
  rw [h1, h2] at h 
  have h3 : (7 + (10 * x + 3) % 11) % 11 = 0 := by rw ← h 
  rw Nat.add_mod at h3 
  cases x 
  case h_0 => rw zero_mul at * 
             simp at h3 
             norm_num at h3
  cases x 
  case h_0 => sorry -- Assume this case has been proved
  case h_succ x_1 => sorry -- Assume this case has been proved
  sorry

end three_digit_number_divisible_by_11_l760_760924


namespace turtle_feeding_cost_l760_760352

def cost_to_feed_turtles (turtle_weight: ℝ) (food_per_half_pound: ℝ) (jar_capacity: ℝ) (jar_cost: ℝ) : ℝ :=
  let total_food := turtle_weight * (food_per_half_pound / 0.5)
  let total_jars := total_food / jar_capacity
  total_jars * jar_cost

theorem turtle_feeding_cost :
  cost_to_feed_turtles 30 1 15 2 = 8 :=
by
  sorry

end turtle_feeding_cost_l760_760352


namespace positive_integer_solutions_xyz_leq_10_l760_760657

theorem positive_integer_solutions_xyz_leq_10 : 
  ∃ (n : ℕ), n = 120 ∧ (∀ (x y z : ℕ), (1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ x + y + z ≤ 10) ↔ n := 120) :=
by
  sorry

end positive_integer_solutions_xyz_leq_10_l760_760657


namespace new_bill_cost_l760_760242

def original_order_cost : ℝ := 25.00
def tomato_cost_old : ℝ := 0.99
def tomato_cost_new : ℝ := 2.20
def lettuce_cost_old : ℝ := 1.00
def lettuce_cost_new : ℝ := 1.75
def celery_cost_old : ℝ := 1.96
def celery_cost_new : ℝ := 2.00
def delivery_and_tip_cost : ℝ := 8.00

theorem new_bill_cost :
  original_order_cost +
  (tomato_cost_new - tomato_cost_old) +
  (lettuce_cost_new - lettuce_cost_old) +
  (celery_cost_new - celery_cost_old) +
  delivery_and_tip_cost = 35.00 :=
by
  simp [original_order_cost, tomato_cost_old, tomato_cost_new,
        lettuce_cost_old, lettuce_cost_new,
        celery_cost_old, celery_cost_new,
        delivery_and_tip_cost]
  norm_num
  -- Expected result of the simplification and normalization
  -- should be 35.00, thus leading to a successful proof.
  sorry

end new_bill_cost_l760_760242


namespace sister_age_ratio_l760_760791

-- Define the current ages of John and his sister
def john_age_current : ℕ := 10
def sister_age_current (x : ℕ) : ℕ := x * john_age_current

-- Define their ages when John is 50
def john_age_future : ℕ := 50
def sister_age_future (x : ℕ) : ℕ := sister_age_current(x) + (john_age_future - john_age_current)

-- The condition given
def condition (x : ℕ) : Prop := sister_age_future(x) = 60

-- The ratio to be proved
def ratio (x : ℕ) : Prop := sister_age_current(x) / john_age_current = 2

theorem sister_age_ratio : ∃ x : ℕ, condition(x) ∧ ratio(x) :=
by
  sorry

end sister_age_ratio_l760_760791


namespace product_sequence_equals_neg_one_l760_760609

-- Define the sequence and the product
def product_sequence : ℤ := ∏ i in (finset.range 2013), (i+1 - i - 1)

-- State the theorem
theorem product_sequence_equals_neg_one : product_sequence = -1 :=
by
  sorry

end product_sequence_equals_neg_one_l760_760609


namespace concyclic_points_l760_760291

/-- Given points A, B, C, D on a circle in that order, let S be the midpoint of the arc AB that 
    does not contain C and D. If SD intersects AB at E and SC intersects AB at F,
    then C, D, E, and F are concyclic. -/
theorem concyclic_points 
  (A B C D : Point) 
  (circle : Circle) 
  (hA : OnCircle A circle)
  (hB : OnCircle B circle)
  (hC : OnCircle C circle)
  (hD : OnCircle D circle)
  (S : Point)
  (hS : IsMidpointArc S A B circle (nContains C D))
  (E F : Point)
  (hE : IntersectsAt E (Line S D) (Line A B))
  (hF : IntersectsAt F (Line S C) (Line A B)) :
  Concyclic C D E F := 
sorry

end concyclic_points_l760_760291


namespace measure_of_angle_E_l760_760911

-- Define the triangle and its properties
def Triangle (A B C : Type) := sorry
def equilateral (T : Triangle) := sorry
def original_angle (T : Triangle) := 60
def angle_decrease (e : ℝ) := 15

-- Define the problem statement
theorem measure_of_angle_E
  (T : Triangle)
  (h1 : equilateral T)
  (E : ℝ)
  (h2 : E = original_angle T - angle_decrease E) :
  E = 45 :=
  sorry

end measure_of_angle_E_l760_760911


namespace part_a_part_b_complete_disorder_l760_760542

open BigOperators

def perm_probability (n m : ℕ) : ℚ :=
  1 / m! * ∑ k in finset.range (n - m + 1), (-1 : ℚ) ^ k / (k + m)!

def at_least_one_probability (n : ℕ) : ℚ :=
  1 - ∑ k in finset.range (n - 1), (-1 : ℚ) ^ (k + 1) / (k + 2)!

def complete_disorder_probability (n : ℕ) : ℚ :=
  ∑ k in finset.range (n + 1), (-1 : ℚ) ^ k / k!

theorem part_a (n m : ℕ) (hm : m ≤ n) :
  perm_probability n m = 1 / m! * ∑ k in finset.range (n - m + 1), (-1 : ℚ) ^ k / (k + m)! :=
by
  sorry

theorem part_b (n : ℕ) :
  at_least_one_probability n = 1 - ∑ k in finset.range (n - 1), (-1 : ℚ) ^ (k + 1) / (k + 2)! :=
by
  sorry

theorem complete_disorder (n : ℕ) :
  complete_disorder_probability n = ∑ k in finset.range (n + 1), (-1 : ℚ) ^ k / k! :=
by
  sorry

end part_a_part_b_complete_disorder_l760_760542


namespace angle_PBD_eq_abs_angle_BCA_minus_angle_PCA_l760_760993

variable (A B C D P : Type*)
variable [angle_space A B C D P]

axiom angle_BPC : angle B P C = 2 * angle B A C
axiom angle_PCA_PAD : angle P C A = angle P A D
axiom angle_PDA_PAC : angle P D A = angle P A C

theorem angle_PBD_eq_abs_angle_BCA_minus_angle_PCA :
  angle P B D = abs (angle B C A - angle P C A) :=
sorry

end angle_PBD_eq_abs_angle_BCA_minus_angle_PCA_l760_760993


namespace merchant_total_gross_profit_correct_l760_760563

noncomputable def total_gross_profit 
  (purchase_price_A purchase_price_B purchase_price_C : ℝ)
  (markup_percentage_A markup_percentage_B markup_percentage_C : ℝ)
  (discount_sequence_A discount_sequence_B : list ℝ)
  (flat_discount_C : ℝ) : ℝ :=
let selling_price_A := purchase_price_A / (1 - markup_percentage_A) in
let selling_price_B := purchase_price_B / (1 - markup_percentage_B) in
let selling_price_C := purchase_price_C / (1 - markup_percentage_C) in
let final_price_A := list.foldl (λ price discount, price * (1 - discount)) selling_price_A discount_sequence_A in
let final_price_B := list.foldl (λ price discount, price * (1 - discount)) selling_price_B discount_sequence_B in
let final_price_C := selling_price_C * (1 - flat_discount_C) in
(final_price_A - purchase_price_A) + (final_price_B - purchase_price_B) + (final_price_C - purchase_price_C)

theorem merchant_total_gross_profit_correct :
  total_gross_profit 56 80 120
                      0.30 0.35 0.40
                      [0.20, 0.15] [0.25, 0.10]
                      0.30 = 235.23 :=
by sorry

end merchant_total_gross_profit_correct_l760_760563


namespace trains_cross_time_l760_760939

noncomputable def time_to_cross (length_train : ℝ) (speed_train_kmph : ℝ) : ℝ :=
  let relative_speed_kmph := speed_train_kmph + speed_train_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  let total_distance := length_train + length_train
  total_distance / relative_speed_mps

theorem trains_cross_time :
  time_to_cross 180 80 = 8.1 := 
by
  sorry

end trains_cross_time_l760_760939


namespace sequence_a_formula_l760_760228

noncomputable def sequence_a : ℕ+ → ℝ
| 1       := 1
| (n + 1) := (n : ℝ) / (n + 1 : ℝ) * sequence_a n

theorem sequence_a_formula (n : ℕ+) : sequence_a n = 1 / (n : ℝ) :=
by sorry

end sequence_a_formula_l760_760228


namespace arithmetic_sequence_constant_l760_760807

theorem arithmetic_sequence_constant (n : ℕ) (h_n : n ≥ 2018)
    (a b : ℕ → ℕ) (ha : ∀ i j, i ≠ j → a i ≠ a j) (hb : ∀ i j, i ≠ j → b i ≠ b j)
    (h_bound_a : ∀ i, a i ≤ 5 * n) (h_bound_b : ∀ i, b i ≤ 5 * n)
    (h_ar : ∀ i, (i < n - 1) → (a (i+1) * b i - a i * b (i+1)) * b (i+2) = (a (i+2) * b (i+1) - a (i+1) * b (i+2)) * b i) :
  ∀ i j, i < n → j < n → a i * b j = a j * b i := 
begin
  sorry -- proof omitted
end

end arithmetic_sequence_constant_l760_760807


namespace sum_primes_less_than_20_l760_760069

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760069


namespace jasmine_commute_time_l760_760787

def time_get_off_work : ℕ := 16 * 60 -- 4:00 pm in minutes
def time_eat_dinner : ℕ := 19 * 60 -- 7:00 pm in minutes
def time_grocery_shop : ℕ := 30
def time_dry_cleaning : ℕ := 10
def time_dog_groomers : ℕ := 20
def time_cook_dinner : ℕ := 90

theorem jasmine_commute_time :
  let total_time := time_eat_dinner - time_get_off_work in
  let task_time := time_grocery_shop + time_dry_cleaning + time_dog_groomers + time_cook_dinner in
  total_time - task_time = 30 :=
by
  sorry

end jasmine_commute_time_l760_760787


namespace jill_savings_percentage_l760_760630

def jill_net_monthly_salary := 3300
def jill_discretionary_income := (1 / 5 : ℝ) * jill_net_monthly_salary
def jill_vacation_fund := (30 / 100 : ℝ) * jill_discretionary_income
def jill_eating_out_and_socializing := (35 / 100 : ℝ) * jill_discretionary_income
def jill_left_for_gifts_and_charity := 99

theorem jill_savings_percentage :
  let total_spent := jill_vacation_fund + jill_eating_out_and_socializing + jill_left_for_gifts_and_charity
  let amount_put_into_savings := jill_discretionary_income - total_spent
  let percentage_put_into_savings := (amount_put_into_savings / jill_discretionary_income) * 100
  percentage_put_into_savings = 20 :=
by
  sorry

end jill_savings_percentage_l760_760630


namespace eval_expression_l760_760173

theorem eval_expression :
  (2^2003 * 3^2005) / (6^2004) = 3 / 2 := by
  sorry

end eval_expression_l760_760173


namespace original_amount_l760_760652

theorem original_amount (x : ℝ) (h : 0.25 * x = 200) : x = 800 := 
by
  sorry

end original_amount_l760_760652


namespace greatest_two_digit_multiple_of_17_l760_760442

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760442


namespace range_of_a_l760_760253

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ [-4, a] → (x^2 - 4*x) ∈ [-4, 32]) →
  (2 ≤ a ∧ a ≤ 8) :=
begin
  sorry
end

end range_of_a_l760_760253


namespace greatest_two_digit_multiple_of_17_l760_760412

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760412


namespace f_values_f_relationship_f_summation_l760_760224

-- Define the function f
def f (x : ℝ) : ℝ := x / (1 + x)

-- Prove the specified values of f
theorem f_values :
  f 2 = 2 / 3 ∧
  f (1 / 2) = 1 / 3 ∧
  f 3 = 3 / 4 ∧
  f (1 / 3) = 1 / 4 :=
by sorry

-- Prove the relationship f(x) + f(1/x) = 1 for all x ≠ 0
theorem f_relationship (x : ℝ) (hx : x ≠ 0) :
  f x + f (1 / x) = 1 :=
by sorry

-- Prove the summation result
theorem f_summation :
  (∑ k in Finset.range 2016, f (↑k + 1)) + (∑ k in Finset.range 2015, f (1 / (↑k + 2))) = 4031 / 2 :=
by sorry

end f_values_f_relationship_f_summation_l760_760224


namespace factorize_polynomial_l760_760638

theorem factorize_polynomial (x y : ℝ) : (3 * x^2 - 3 * y^2) = 3 * (x + y) * (x - y) := 
by
  sorry

end factorize_polynomial_l760_760638


namespace greatest_two_digit_multiple_of_17_l760_760453

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760453


namespace marcy_needs_6_tubs_of_lip_gloss_l760_760817

theorem marcy_needs_6_tubs_of_lip_gloss (people tubes_per_person tubes_per_tub : ℕ) 
  (h1 : people = 36) (h2 : tubes_per_person = 3) (h3 : tubes_per_tub = 2) :
  (people / tubes_per_person) / tubes_per_tub = 6 :=
by
  -- The proof goes here
  sorry

end marcy_needs_6_tubs_of_lip_gloss_l760_760817


namespace distance_between_lines_l760_760801

noncomputable def f (x : ℝ) := Real.exp x + x

def line2 (A B C x y : ℝ) := A * x + B * y + C = 0

theorem distance_between_lines (h_tangent : ∀ x, 
    ∃ y, is_tangent_line (f x) y (2 : ℝ)) :
  let l2_eq : (λ (x y : ℝ), 2 * x - y + 3 = 0) :=
  let dist := (2 * Real.sqrt 5) / 5 in
  ∀ x1 y1,
  line2 2 (-1) 3 x1 y1 →
  ∃ x0 y0, f x0 = y0 ∧
  dist = abs (2 * x1 - y1 + 3) / Real.sqrt (2^2 + (-1)^2) :=
sorry

end distance_between_lines_l760_760801


namespace angle_same_terminal_side_315_l760_760356

theorem angle_same_terminal_side_315 (k : ℤ) : ∃ α, α = k * 360 + 315 ∧ α = -45 :=
by
  use -45
  sorry

end angle_same_terminal_side_315_l760_760356


namespace prove_scientific_notation_l760_760532

def scientific_notation_correct : Prop :=
  340000 = 3.4 * (10 ^ 5)

theorem prove_scientific_notation : scientific_notation_correct :=
  by
    sorry

end prove_scientific_notation_l760_760532


namespace women_in_third_group_l760_760548

variables (m w : ℝ)

theorem women_in_third_group (h1 : 3 * m + 8 * w = 6 * m + 2 * w) (x : ℝ) (h2 : 2 * m + x * w = 0.5 * (3 * m + 8 * w)) :
  x = 4 :=
sorry

end women_in_third_group_l760_760548


namespace exactly_two_knaves_telling_truth_l760_760240

-- Definitions from conditions
inductive Knave
| H | C | D | S 
open Knave

-- Predicate for whether a Knave is telling the truth
def is_telling_truth : Knave → Prop

-- Main theorem statement to prove
theorem exactly_two_knaves_telling_truth  (h_c_eq : C = ¬ H ∧ D = H ∧ S = ¬ H ∨ C = H ∧ D = ¬ H ∧ S = H)
    (knave_bool : Knave → Prop) :
    ((knave_bool H ∧ ¬ knave_bool C ∧ knave_bool D ∧ ¬ knave_bool S) ∨ 
    (¬ knave_bool H ∧ knave_bool C ∧ ¬ knave_bool D ∧ knave_bool S)) → 
    (∃ hn n : nat, hn + n = 2) :=
by
  sorry

end exactly_two_knaves_telling_truth_l760_760240


namespace bicycle_count_l760_760531

theorem bicycle_count (B T : ℕ) (hT : T = 20) (h_wheels : 2 * B + 3 * T = 160) : B = 50 :=
by
  sorry

end bicycle_count_l760_760531


namespace tetrahedron_cube_volume_ratio_l760_760764

theorem tetrahedron_cube_volume_ratio (s : ℝ) (h_s : s > 0):
    let V_cube := s ^ 3
    let a := s * Real.sqrt 3
    let V_tetrahedron := (Real.sqrt 2 / 12) * a ^ 3
    (V_tetrahedron / V_cube) = (Real.sqrt 6 / 4) := by
    sorry

end tetrahedron_cube_volume_ratio_l760_760764


namespace menelaus_theorem_l760_760345

variable (A B C P A₁ B₁ C₁ : Type) [MetricSpace A] [MetricSpace B]
[MetricSpace C] [MetricSpace P] [MetricSpace A₁] [MetricSpace B₁]
[MetricSpace C₁]
variable (circ : Circle A B C) (on_arc : P ∈ circ.arc B C)

theorem menelaus_theorem (transversal : collinear A₁ B₁ C₁) :
  (B.seg_dist A₁ / C.seg_dist A₁) * (C.seg_dist B₁ / A.seg_dist B₁) * 
  (A.seg_dist C₁ / B.seg_dist C₁) = 1 := sorry

end menelaus_theorem_l760_760345


namespace greatest_two_digit_multiple_of_17_l760_760402

theorem greatest_two_digit_multiple_of_17 : ∃ x : ℕ, x < 100 ∧ x ≥ 10 ∧ x % 17 = 0 ∧ ∀ y : ℕ, y < 100 ∧ y ≥ 10 ∧ y % 17 = 0 → y ≤ x :=
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760402


namespace greatest_two_digit_multiple_of_17_l760_760436

theorem greatest_two_digit_multiple_of_17 : ∃ n, (n < 100) ∧ (n % 17 = 0) ∧ (∀ m, (m < 100) ∧ (m % 17 = 0) → m ≤ n) := by
  let multiples := [17, 34, 51, 68, 85]
  have ht : ∀ m, m ∈ multiples → (m < 100) ∧ (m % 17 = 0) := by
    intros m hm
    cases hm with
    | inl h => exact ⟨by norm_num, by norm_num⟩
    | inr hm' =>
      cases hm' with
      | inl h => exact ⟨by norm_num, by norm_num⟩
      | inr hm'' =>
        cases hm'' with
        | inl h => exact ⟨by norm_num, by norm_num⟩
        | inr hm''' =>
          cases hm''' with
          | inl h => exact ⟨by norm_num, by norm_num⟩
          | inr hm'''' =>
            cases hm'''' with
            | inl h => exact ⟨by norm_num, by norm_num⟩
            | inr hm => cases hm
  
  use 85
  split
  { norm_num },
  split
  { exact (by norm_num : 85 % 17 = 0) },
  intro m
  intro h
  exact list.minimum_mem multiples h

end greatest_two_digit_multiple_of_17_l760_760436


namespace f_2011_is_zero_l760_760703

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f (x) + f (1)

-- Theorem stating the mathematically equivalent proof problem
theorem f_2011_is_zero : f (2011) = 0 :=
sorry

end f_2011_is_zero_l760_760703


namespace integral_compute_l760_760612

noncomputable def integral_function (x : ℝ) : ℝ :=
  (x^2 + 5 * x + 6) * Real.cos (2 * x)

theorem integral_compute :
  ∫ x in -2..0, integral_function x = (5 - Real.cos 4 - Real.sin 4) / 4 :=
by
  -- proof omitted
  sorry

end integral_compute_l760_760612


namespace range_of_f_2x_minus_x_squared_l760_760215

def g : ℝ → ℝ := λ x, (1 / 2) ^ x

def f : ℝ → ℝ := λ x, -Real.log x / Real.log 2

theorem range_of_f_2x_minus_x_squared :
  (∀ x, ∃ y, g y = x ↔ f x = y) →
  set.Ici 0 = {y | ∃ x : ℝ, f (2 * x - x^2) = y } :=
  by
    sorry

end range_of_f_2x_minus_x_squared_l760_760215


namespace greatest_integer_y_l760_760400

theorem greatest_integer_y :
  ∃ y : ℤ, y ≤ 10 ∧ ∀ z : ℤ, (z > y → \(frac{8}{11}\ > \frac{z}{15}\)) ∧ (y = 10) := 
begin
  sorry
end

end greatest_integer_y_l760_760400


namespace find_polynomial_P_l760_760799

theorem find_polynomial_P :
  ∃ P : ℂ → ℂ, 
    (∀ (a b c : ℂ), 
      (a + b + c = -3) → 
      (a * b + b * c + c * a = 5) →
      (a * b * c = -7) →
      (P a = b + c) →
      (P b = a + c) →
      (P c = a + b) →
      (P (-3) = -16) →
      P = (λ x : ℂ, 2 * x^3 + 6 * x^2 + 9 * x + 11)) :=
sorry

end find_polynomial_P_l760_760799


namespace students_8th_day_over_3280_l760_760327

-- Definitions based on the conditions.
def initial_day : ℕ := 1
def students_initial := 4
def secret_expansion (n : ℕ) : ℕ := 1 + (sum (λ k, 3^k) (range (n + 1)))

-- The theorem to be proven.
theorem students_8th_day_over_3280 : secret_expansion 7 ≥ 3280 := by
  -- The transformation steps from mathematical problem conditions to Lean definition.
  let S := 1 + (3^8 - 1) / 2
  have h1 : (3^8 : ℕ) = 6561 := by norm_num
  have h2 : 1 + (6561 - 1) / 2 = 3281 := by norm_num
  show 3281 ≥ 3280
  norm_num

end students_8th_day_over_3280_l760_760327


namespace sequence_expression_l760_760725

noncomputable def seq (n : ℕ) : ℝ := 
  match n with
  | 0 => 1  -- note: indexing from 1 means a_1 corresponds to seq 0 in Lean
  | m+1 => seq m / (3 * seq m + 1)

theorem sequence_expression (n : ℕ) : 
  ∀ n, seq (n + 1) = 1 / (3 * (n + 1) - 2) := 
sorry

end sequence_expression_l760_760725


namespace proof_problem_l760_760706

variables (a b : ℝ^3) (theta : ℝ) (sqrt3 : ℝ) (one : ℝ)

-- Definition of the conditions in a):
def angle_between_a_b : theta = real.pi / 6 := sorry
def norm_a : ∥a∥ = real.sqrt 3 := sorry
def norm_b : ∥b∥ = 1 := sorry
def dot_product_a_b : a ⬝ b = real.sqrt 3 / 2 := sorry

-- Problem 1:
def problem1 := ∥a - 2 • b∥ = 1

-- Problem 2:
def p := a + 2 • b
def q := a - 2 • b
def projection_p_q := -1

-- Final Statement
theorem proof_problem : 
  (∥a - 2 • b∥ = 1) ∧ ((p ⬝ q) / (∥p∥ * ∥q∥) = -1) :=
by {
  have h1 : ∥a - 2 • b∥ = 1 := sorry,
  have h2 : (p ⬝ q) / (∥p∥ * ∥q∥) = -1 := sorry,
  exact ⟨h1, h2⟩,
}

end proof_problem_l760_760706


namespace greatest_two_digit_multiple_of_17_l760_760446

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760446


namespace cost_to_feed_turtles_l760_760355

theorem cost_to_feed_turtles :
  (∀ (weight : ℚ), weight > 0 → (food_per_half_pound_ounces = 1) →
    (total_weight_pounds = 30) →
    (jar_contents_ounces = 15 ∧ jar_cost = 2) →
    (total_cost_dollars = 8)) :=
by
  sorry

variables
  (food_per_half_pound_ounces : ℚ := 1)
  (total_weight_pounds : ℚ := 30)
  (jar_contents_ounces : ℚ := 15)
  (jar_cost : ℚ := 2)
  (total_cost_dollars : ℚ := 8)

-- Assuming all variables needed to state the theorem exist and are meaningful
/-!
  Given:
  - Each turtle needs 1 ounce of food per 1/2 pound of body weight
  - Total turtles' weight is 30 pounds
  - Each jar of food contains 15 ounces and costs $2

  Prove:
  - The total cost to feed Sylvie's turtles is $8
-/

end cost_to_feed_turtles_l760_760355


namespace Jeff_pays_when_picking_up_l760_760282

-- Definition of the conditions
def deposit_rate : ℝ := 0.10
def increase_rate : ℝ := 0.40
def last_year_cost : ℝ := 250
def this_year_cost : ℝ := last_year_cost * (1 + increase_rate)
def deposit : ℝ := this_year_cost * deposit_rate

-- Lean statement of the proof
theorem Jeff_pays_when_picking_up : this_year_cost - deposit = 315 := by
  sorry

end Jeff_pays_when_picking_up_l760_760282


namespace calculate_gross_profit_l760_760375

theorem calculate_gross_profit :
  let sales_price : ℝ := 81
  let sales_tax : ℝ := 0.07
  let discount : ℝ := 0.15
  let gross_profit_margin : ℝ := 1.7
  let P_bt := sales_price / (1 + sales_tax)
  let P_o := P_bt / (1 - discount)
  let C := P_o / (1 + gross_profit_margin)
  let gross_profit := P_o - C
  gross_profit ≈ 56.07 := by
sorry


end calculate_gross_profit_l760_760375


namespace greatest_two_digit_multiple_of_17_l760_760518

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760518


namespace main_theorem_l760_760160

noncomputable def phase_trajectories_are_spirals
  (x : ℝ → ℝ)
  (v : ℝ → ℝ := λ t, deriv x t)
  (d2x_dt2 : ℝ → ℝ := λ t, deriv (deriv x) t) :
  Prop :=
  ∀ t : ℝ,
    d2x_dt2 t - v t + x t = 0 →
    ∃ k : ℝ, v t = x t / (1 - k)

theorem main_theorem {x : ℝ → ℝ}
  (h : phase_trajectories_are_spirals x) :
  ∀ t : ℝ, ∃ k : ℝ, (deriv x t) = x t / (1 - k) :=
sorry

end main_theorem_l760_760160


namespace sum_of_primes_less_than_20_eq_77_l760_760015

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760015


namespace degree_le_three_l760_760176

theorem degree_le_three
  (d : ℕ)
  (P : Polynomial ℤ)
  (hdeg : P.degree = d)
  (hP : ∃ (S : Finset ℤ), (S.card ≥ d + 1) ∧ ∀ m ∈ S, |P.eval m| = 1) :
  d ≤ 3 := 
sorry

end degree_le_three_l760_760176


namespace no_superabundant_numbers_l760_760800

-- Define the function h(n) which generates the product of all divisors of a positive integer n.
noncomputable def h (n : ℕ) : ℕ :=
  if n = 0 then 0 else ∏ d in (finset.Icc 1 n).filter (λ d, n % d = 0), d

-- Prove there are no positive integers n such that h(h(n)) = n^2 + 2n.
theorem no_superabundant_numbers :
  ∀ n : ℕ, h (h n) ≠ n^2 + 2*n := by
  intros n
  sorry

end no_superabundant_numbers_l760_760800


namespace equal_incircles_ratios_l760_760277

theorem equal_incircles_ratios (AB BC AC : ℕ) (h1 : AB = 14) (h2 : BC = 15) (h3 : AC = 17)
(M : Point → Point → Point) (incircle_equal : ∀ P Q R : Point, inradius_equal (triangle P Q R) (triangle P Q R) M) :
∃ AM CM : ℕ, AM + CM = AC ∧ AM / (AC - AM) = 36 / 83 → 36 + 83 = 119 :=
by
  sorry 

end equal_incircles_ratios_l760_760277


namespace max_a_plus_b_l760_760751

theorem max_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ A B : ℝ × ℝ, (A.1 * a + A.2 * b = 1 ∧ B.1 * a + B.2 * b = 1) ∧ 
  (A.1 ^ 2 + A.2 ^ 2 = 1 ∧ B.1 ^ 2 + B.2 ^ 2 = 1) ∧ 
  A ≠ B ∧ (det₂ ![A.1, A.2, B.1, B.2] ≠ 0) ∧ 
  ((A.1, A.2) ≠ (0, 0) ∧ (B.1, B.2) ≠ (0, 0)) ∧ 
  ∀ (P : ℝ × ℝ) (Q : ℝ × ℝ), 
  (P.1 * a + P.2 * b = 1 ∧ Q.1 * a + Q.2 * b = 1 ∧ 
  P.1 ^ 2 + P.2 ^ 2 = 1 ∧ Q.1 ^ 2 + Q.2 ^ 2 = 1 ∧ 
  P ≠ Q ∧ det₂ ![P.1, P.2, Q.1, Q.2] ≠ 0) → 
  (abs ((0, 0), P, Q))) ≤ (abs ((0, 0), A, B)))
→ a + b = 2 := begin
  sorry
end

end max_a_plus_b_l760_760751


namespace intersect_curves_l760_760227

-- Definitions of the parametric equations of the curves
def curve1 (θ : ℝ) : ℝ × ℝ :=
  (real.sqrt 5 * real.cos θ, real.sin θ)

def curve2 (t : ℝ) : ℝ × ℝ :=
  (5 / 4 * t, t)

-- Define the Cartesian equations derived from the parametric curves
def curve1_cartesian (p : ℝ × ℝ) : Prop :=
  p.2 ≥ 0 ∧ (p.1 ^ 2) / 5 + p.2 ^ 2 = 1

def curve2_cartesian (p : ℝ × ℝ) : Prop :=
  p.2 = 4 / 5 * p.1

theorem intersect_curves :
  ∃ (p : ℝ × ℝ), curve1_cartesian p ∧ curve2_cartesian p ∧ p = (5 / 6, 2 / 3) :=
by
  sorry

end intersect_curves_l760_760227


namespace five_digit_arithmetic_sequence_count_l760_760735

-- Definitions needed for the conditions
def digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Main statement translating the math problem to Lean 4
theorem five_digit_arithmetic_sequence_count :
  ∃! (count : ℕ), count = 744 ∧
  (∀ (numbers : list ℕ), 
    (∀ x ∈ numbers, x ∈ digits) →
    (∀ x ∈ numbers, list.nodup numbers) →
    (numbers.length = 5) →
    (∃ a b c d e : ℕ, numbers = [a, b, c, d, e] ∧ is_arithmetic_sequence b c d ∧
      (list.mem a digits ∧ list.mem b digits ∧ list.mem c digits ∧ list.mem d digits ∧ list.mem e digits) ∧
      (list.nodup [a, b, c, d, e]))) :=
sorry

end five_digit_arithmetic_sequence_count_l760_760735


namespace max_value_of_expression_l760_760295

theorem max_value_of_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + 2 * b + 3 * c = 1) :
    (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) ≤ 7) :=
sorry

end max_value_of_expression_l760_760295


namespace distinct_values_f_l760_760311

def f (x : ℝ) : ℝ := ∑ k in (finset.range 10).map (λ n, n + 3), (k * x).floor - k * x.floor

theorem distinct_values_f : ∀ x, x ≥ 0 → set.finite {y | ∃ x ∈ set.Ici (0 : ℝ), y = f x} ∧ set.card {y | ∃ x ∈ set.Ici (0 : ℝ), y = f x} = 45 :=
by
  intro x hx
  sorry -- proof to be provided

end distinct_values_f_l760_760311


namespace find_f_600_l760_760794

variable (f : ℝ → ℝ)
variable (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y)
variable (h2 : f 500 = 3)

theorem find_f_600 : f 600 = 5 / 2 :=
by
  sorry

end find_f_600_l760_760794


namespace int_to_fourth_power_l760_760246

theorem int_to_fourth_power:
  3^4 * 9^8 = 243^4 :=
by 
  sorry

end int_to_fourth_power_l760_760246


namespace p_value_correct_years_trees_cut_correct_additional_years_trees_cut_max_l760_760868

variable (a : ℝ) (p : ℝ := 100 * (1 - (1 / 2)^(1 / 10)))

theorem p_value_correct :
  (1 - p / 100) ^ 10 = 1 / 2 := by
  sorry

variable (m : ℝ := 5)

theorem years_trees_cut_correct :
  (1 - p / 100) ^ m = (sqrt 2) / 2 := by
  sorry

variable (n : ℝ := 15)

theorem additional_years_trees_cut_max :
  (1 - p / 100) ^ n ≥ sqrt 2 / 4 := by
  sorry

end p_value_correct_years_trees_cut_correct_additional_years_trees_cut_max_l760_760868


namespace pharmaceutical_royalties_decrease_l760_760968

noncomputable def ratio_decrease_percent (R₁ R₂ S₁ S₂ : ℝ) : ℝ :=
  let ratio₁ := R₁ / S₁
  let ratio₂ := R₂ / S₂
  let decrease := ratio₁ - ratio₂
  decrease * 100

theorem pharmaceutical_royalties_decrease :
  ratio_decrease_percent 8 9 20 108 ≈ 31.67 := 
sorry

end pharmaceutical_royalties_decrease_l760_760968


namespace largest_possible_area_of_G1G2G3_l760_760303

variables (A1 B1 C1 A2 B2 C2 A3 B3 C3 : Type)
variables [triangle A1 B1 C1] [triangle A2 B2 C2] [triangle A3 B3 C3]
variables (D1 E1 F1 D2 E2 F2 D3 E3 F3 G1 G2 G3 : Type)
variables [is_midpoint D1 B1 C1][is_midpoint E1 A1 C1][is_midpoint F1 A1 B1]
variables [is_midpoint D2 B2 C2][is_midpoint E2 A2 C2][is_midpoint F2 A2 B2]
variables [is_midpoint D3 B3 C3][is_midpoint E3 A3 C3][is_midpoint F3 A3 B3]
variables [is_centroid G1 A1 B1 C1][is_centroid G2 A2 B2 C2][is_centroid G3 A3 B3 C3]
variables (area_A1A2A3 : ℝ) (area_B1B2B3 : ℝ) (area_C1C2C3 : ℝ)
          (area_D1D2D3 : ℝ) (area_E1E2E3 : ℝ) (area_F1F2F3 : ℝ)

-- Conditions for the areas of the triangles
axiom areas_known : area_A1A2A3 = 2 ∧ area_B1B2B3 = 3 ∧ area_C1C2C3 = 4 ∧
                    area_D1D2D3 = 20 ∧ area_E1E2E3 = 21 ∧ area_F1F2F3 = 2020

-- Statement to prove
theorem largest_possible_area_of_G1G2G3 : area_of G1 G2 G3 = 917 :=
by sorry

end largest_possible_area_of_G1G2G3_l760_760303


namespace circumcircle_incircle_ineq_circumcircle_incircle_ineq_equilateral_exists_triangle_radii_l760_760098

theorem circumcircle_incircle_ineq (R r : ℝ) (hRr : R = 2 * r) : 
  ∃ Δ : Type, (is_triangle Δ) ∧ (circumradius Δ = R) ∧ (inradius Δ = r) :=
sorry

theorem circumcircle_incircle_ineq_equilateral (R r : ℝ) (hRr_eq : R = 2 * r) :
  ∀ Δ : Type, (is_triangle Δ) ∧ (circumradius Δ = R) ∧ (inradius Δ = r) → (is_equilateral Δ) :=
sorry

theorem exists_triangle_radii (R r : ℝ) (hIneq : R ≥ 2 * r) : 
  ∃ Δ : Type, (is_triangle Δ) ∧ (circumradius Δ = R) ∧ (inradius Δ = r) :=
sorry

end circumcircle_incircle_ineq_circumcircle_incircle_ineq_equilateral_exists_triangle_radii_l760_760098


namespace sum_primes_less_than_20_l760_760090

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760090


namespace cos_75_eq_sqrt6_sub_sqrt2_div_4_l760_760151

theorem cos_75_eq_sqrt6_sub_sqrt2_div_4 :
  Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := sorry

end cos_75_eq_sqrt6_sub_sqrt2_div_4_l760_760151


namespace largest_coeff_expansion_term_l760_760380

-- Define the expansion term
def expansion_term (n r : ℕ) : ℤ := (choose n r) * (-1) ^ r

-- Statement of the problem
theorem largest_coeff_expansion_term :
  (∀ n = 2009, ∃ r = 1004, (expansion_term n r) = (choose 2009 1004) * (-1) ^ 1004) ∧
  ∀ k, k ≠ 1004 → (choose 2009 k) ≤ (choose 2009 1004) :=
sorry

end largest_coeff_expansion_term_l760_760380


namespace number_of_paths_is_100_l760_760123

-- Define the movement rules and no right-angle turns condition
def valid_moves (p q : ℕ × ℕ) : Prop :=
  match p, q with
  | (a, b), (a', b') =>
    (a' = a + 1 ∧ b' = b) ∨     -- Move right
    (a' = a ∧ b' = b + 1) ∨     -- Move up
    (a' = a + 1 ∧ b' = b + 1)   -- Move diagonally right-up

def no_right_angle_turns (path : List (ℕ × ℕ)) : Prop :=
  path.pairwise valid_moves

-- Define the starting and ending points
def start : ℕ × ℕ := (0, 0)
def end : ℕ × ℕ := (6, 6)

-- Define the problem statement
theorem number_of_paths_is_100 :
  {path : List (ℕ × ℕ) // no_right_angle_turns path ∧ path.head = start ∧ path.last = end}.card = 100 :=
sorry

end number_of_paths_is_100_l760_760123


namespace sum_of_primes_less_than_20_eq_77_l760_760014

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760014


namespace impurities_removed_correct_l760_760397

def initial_gold_dust_mass : ℝ := 12
def pure_gold_fraction : ℝ := 17 / 24

def method_a_removal_fraction : ℝ := 1 / 16
def method_b_removal_fraction : ℝ := 1 / 8
def method_c_removal_fraction : ℝ := 1 / 4

noncomputable def total_impurities_removed : ℝ :=
  let pure_gold_mass := pure_gold_fraction * initial_gold_dust_mass
  let initial_impurities := initial_gold_dust_mass - pure_gold_mass
  let impurities_after_a := initial_impurities * (1 - method_a_removal_fraction)
  let impurities_after_b := impurities_after_a * (1 - method_b_removal_fraction)
  let impurities_after_c := impurities_after_b * (1 - method_c_removal_fraction)
  initial_impurities - impurities_after_c

theorem impurities_removed_correct :
  total_impurities_removed = 1.3466796875 :=
by
  sorry

end impurities_removed_correct_l760_760397


namespace find_digits_l760_760358

-- Define the factorial function
noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

-- Define the structure of the number that represents 15!
structure FactorialRepresentation :=
  (X Y Z : ℕ)
  (number_repr : ℕ := 1 * 10^12 + 307 * 10^9 + 674 * 10^6 + 3 * 10^5 + X * 10^4 + 0 * 10^3 + 0 * 10^2 + Y * 10 + 0)
  (actual_value : ℕ := factorial 15)

-- The theorem that states what we need to prove
theorem find_digits (repr: FactorialRepresentation) : repr.X + repr.Y + repr.Z = 0 :=
  sorry

end find_digits_l760_760358


namespace degree_of_g_is_4_l760_760739

-- Define a polynomial f with specified terms
def f (x : ℝ) := -7 * x^4 + 3 * x^3 + x - 5

-- Define g as an unknown polynomial 
variable (g : ℝ → ℝ)

-- Assume degree of f + g is 1
axiom h_deg : ∀ x, degree (f x + g x) = 1

-- Prove that the degree of g is 4
theorem degree_of_g_is_4 : ∀ x, degree (g x) = 4 := 
sorry

end degree_of_g_is_4_l760_760739


namespace value_of_a_plus_b_l760_760699

open Set Real

def setA : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def setB (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}
def universalSet : Set ℝ := univ

theorem value_of_a_plus_b (a b : ℝ) :
  (setA ∪ setB a b = universalSet) ∧ (setA ∩ setB a b = {x : ℝ | 3 < x ∧ x ≤ 4}) → a + b = -7 :=
by
  sorry

end value_of_a_plus_b_l760_760699


namespace function_c_is_even_l760_760931

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def f1 (x : ℝ) : ℝ := Real.cos (x + Real.pi / 2)
def f2 (x : ℝ) : ℝ := Real.sin x * Real.cos x
def f3 (x : ℝ) : ℝ := x^2 * Real.cos x
def f4 (x : ℝ) : ℝ := x^2 * Real.sin x

theorem function_c_is_even : is_even_function f3 :=
sorry

end function_c_is_even_l760_760931


namespace sum_of_primes_less_than_20_eq_77_l760_760011

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760011


namespace sum_of_primes_less_than_20_eq_77_l760_760024

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760024


namespace numWaysElectOfficers_l760_760965

-- Definitions and conditions from part (a)
def numMembers : Nat := 30
def numPositions : Nat := 5
def members := ["Alice", "Bob", "Carol", "Dave"]
def allOrNoneCondition (S : List String) : Bool := 
  S.all (members.contains)

-- Function to count the number of ways to choose the officers
def countWays (n : Nat) (k : Nat) (allOrNone : Bool) : Nat :=
if allOrNone then
  -- All four members are positioned
  Nat.factorial k * (n - k)
else
  -- None of the four members are positioned
  let remaining := n - members.length
  remaining * (remaining - 1) * (remaining - 2) * (remaining - 3) * (remaining - 4)

theorem numWaysElectOfficers :
  let casesWithNone := countWays numMembers numPositions false
  let casesWithAll := countWays numMembers numPositions true
  (casesWithNone + casesWithAll) = 6378720 :=
by
  sorry

end numWaysElectOfficers_l760_760965


namespace sum_f_2018_mod_1000_l760_760619

-- Definitions based on the conditions
def permutation (n : ℕ) : Type := { l : List ℕ // l.nodup ∧ (∀ i ∈ l, i ∈ List.range (n + 1)) }

def sortable (n : ℕ) (p : permutation n) : Prop :=
  ∃ k, ∀ (i j : ℕ), (i < j → i < k → (list.erase p.val k).nthElem i ≤ (list.erase p.val k).nthElem j)

noncomputable def f : ℕ → ℕ 
| 0     := 1
| (n+1) := f n + n

-- Calculate the required sum
noncomputable def sum_f (n : ℕ) : ℤ :=
  (Finset.range n).sum (λ i, (-1)^(i+1) * f (i+1))

-- Prove the final result
theorem sum_f_2018_mod_1000 : sum_f 2018 % 1000 = 153 :=
by
  sorry

end sum_f_2018_mod_1000_l760_760619


namespace solution_set_of_inequality_l760_760378

theorem solution_set_of_inequality (x : ℝ) : (1 / |x - 1| ≥ 1) ↔ (0 ≤ x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) :=
by
  sorry

end solution_set_of_inequality_l760_760378


namespace marcella_lost_shoes_l760_760318

theorem marcella_lost_shoes (total_shoes pairs_left : ℕ) (H1 : total_shoes = 54) (H2 : pairs_left = 44) :
  total_shoes - pairs_left = 10 :=
by
  rw [H1, H2]
  simp
  rfl

end marcella_lost_shoes_l760_760318


namespace bones_received_on_sunday_l760_760287

-- Definitions based on the conditions
def initial_bones : ℕ := 50
def bones_eaten : ℕ := initial_bones / 2
def bones_left_after_saturday : ℕ := initial_bones - bones_eaten
def total_bones_after_sunday : ℕ := 35

-- The theorem to prove how many bones received on Sunday
theorem bones_received_on_sunday : 
  (total_bones_after_sunday - bones_left_after_saturday = 10) :=
by
  -- proof will be filled in here
  sorry

end bones_received_on_sunday_l760_760287


namespace shark_teeth_problem_l760_760975

theorem shark_teeth_problem :
  ∃ (S : ℕ), 
  let hammerhead_teeth := S / 6,
      great_white_teeth := 2 * (S + hammerhead_teeth) in
  great_white_teeth = 420 ∧ S = 180 :=
begin
  sorry
end

end shark_teeth_problem_l760_760975


namespace chocolate_bar_cost_l760_760143

def anna_mom_gave_money : ℝ := 10.0
def cost_per_gum_pack : ℝ := 1.0
def num_gum_packs : ℕ := 3
def cost_per_candy_cane : ℝ := 0.5
def num_candy_canes : ℕ := 2
def num_chocolate_bars : ℕ := 5
def anna_money_left : ℝ := 1.0

theorem chocolate_bar_cost :
  let total_gum_cost := num_gum_packs * cost_per_gum_pack in
  let total_candy_cane_cost := num_candy_canes * cost_per_candy_cane in
  let total_spent := anna_mom_gave_money - anna_money_left in
  let total_known_cost := total_gum_cost + total_candy_cane_cost in
  let total_chocolate_cost := total_spent - total_known_cost in
  total_chocolate_cost / num_chocolate_bars = 1.0 :=
by
  sorry

end chocolate_bar_cost_l760_760143


namespace intersection_on_circumcircle_of_ABC_l760_760148

open EuclideanGeometry

variables {A B C P E F B' C' B'' C'' : Point}

theorem intersection_on_circumcircle_of_ABC 
  (ABC_circ : Circumcircle A B C)
  (P_in_triangle : InTriangle P A B C)
  (E_on_AC : OnLine E A C)
  (F_on_AB : OnLine F A B)
  (BP_intersect_AC : IntersectAt BP AC E)
  (CP_intersect_AB : IntersectAt CP AB F)
  (EF_intersect_circumcircle : Intersects EF ABC_circ B' C')
  (E_between_F_B' : IsBetween E F B')
  (B'P_intersect_BC : IntersectAt (LineOf B' P) BC C'')
  (C'P_intersect_BC : IntersectAt (LineOf C' P) BC B'') :
  ∃ K : Point, OnCircumcircle K ABC_circ ∧ IntersectAt (LineOf B' B'') (LineOf C' C'') K := sorry

end intersection_on_circumcircle_of_ABC_l760_760148


namespace parallelogram_area_twice_quadrilateral_area_l760_760336

variables {K : Type*} [field K] [vector_space K (euclidean_space 2)]

noncomputable def area_parallelogram_twice_quadrilateral (A B C D : euclidean_space 2) : Prop :=
  let Q := quadrilateral.mk A B C D in
  let AC := diagonal.mk A C in
  let BD := diagonal.mk B D in
  let KLMN := parallelogram.mk_through_vertices_parallel_to_diagonals Q AC BD in
  KLMN.area = 2 * Q.area

theorem parallelogram_area_twice_quadrilateral_area 
  (A B C D : euclidean_space 2) 
  (Q := quadrilateral.mk A B C D) 
  (AC := diagonal.mk A C) 
  (BD := diagonal.mk B D) 
  (KLMN := parallelogram.mk_through_vertices_parallel_to_diagonals Q AC BD) :
  KLMN.area = 2 * Q.area :=
sorry

end parallelogram_area_twice_quadrilateral_area_l760_760336


namespace sum_prime_numbers_less_than_twenty_l760_760041

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l760_760041


namespace combined_dolls_l760_760589

def dolls_combination (Vera_dolls Lisa_dolls Sophie_dolls Aida_dolls : ℕ) : Prop :=
  Vera_dolls = 15 ∧
  Lisa_dolls = Vera_dolls + 10 ∧
  Sophie_dolls = 2 * Vera_dolls ∧
  Aida_dolls = 3 * Sophie_dolls ∧
  (Aida_dolls + Sophie_dolls + Vera_dolls + Lisa_dolls = 160)

theorem combined_dolls : 
  ∃ Vera_dolls Lisa_dolls Sophie_dolls Aida_dolls : ℕ, 
  dolls_combination Vera_dolls Lisa_dolls Sophie_dolls Aida_dolls := 
begin
  sorry
end

end combined_dolls_l760_760589


namespace greatest_two_digit_multiple_of_17_l760_760513

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760513


namespace points_within_triangle_l760_760104

noncomputable theory
open_locale classical

open Set

-- Definitions and assumptions
variables {α : Type*} [linear_ordered_field α] {x y : Fin n → α}

def max_area {points : Fin n → (α × α)} : α :=
  finset.univ.image (λ (t : Finset (Fin n)), 
    ∆ t.sum) points).sup

def triangle_area (A B C : (α × α)) : α :=
  abs ((A.1 - C.1) * (B.2 - A.2) - (A.1 - B.1) * (C.2 - A.2)) / 2

-- Problem statement
theorem points_within_triangle (points : Fin n → (α × α))
    (h : ∀ (i j k : Fin n), triangle_area (points i) (points j) (points k) ≤ 1) :
  ∃ (T : Finset (α × α)), (T.card = 3) ∧ (triangle_area (T.sum) = 4) ∧
    ∀ i, (points i) ∈ convex_hull T :=
sorry


end points_within_triangle_l760_760104


namespace ratio_diff_l760_760952

theorem ratio_diff (x : ℕ) (h1 : 7 * x = 56) : 56 - 3 * x = 32 :=
by
  sorry

end ratio_diff_l760_760952


namespace problem_statement_l760_760850

def hall := Fin 100
def knows : hall → hall → Prop
def at_least_66_known (h : hall) : Prop := ∑ x, if knows h x then 1 else 0 ≥ 66

theorem problem_statement :
  (∀ h : hall, at_least_66_known h) →
  ∀ (s : Finset hall), s.card = 4 → ∃ (x y : hall), x ∈ s ∧ y ∈ s ∧ ¬knows x y :=
begin
  sorry
end

end problem_statement_l760_760850


namespace greatest_two_digit_multiple_of_17_l760_760491

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760491


namespace face_opposite_to_D_is_A_l760_760559

-- Definitions to contain the conditions of the problem
structure CubeFaces where
  A B C D E F : Type

structure CubeCondition (faces : CubeFaces) where
  adjacent : faces.C → faces.D → Prop
  adjacent_CF : faces.C → faces.F → Prop
  adjacent_CE : faces.C → faces.E → Prop
  adjacent_CB : faces.C → faces.B → Prop
  vertex_shared_CB : faces.C → faces.B → Prop

-- Theorem statement to prove the face opposite to D is A
theorem face_opposite_to_D_is_A
  (faces : CubeFaces)
  (cond : CubeCondition faces)
  (adj_CD : cond.adjacent faces.C faces.D)
  (adj_CF : cond.adjacent_CF faces.C faces.F)
  (adj_CE : cond.adjacent_CE faces.C faces.E)
  (vertex_CB : cond.vertex_shared_CB faces.C faces.B) :
  faces.A ≠ faces.D :=
sorry

end face_opposite_to_D_is_A_l760_760559


namespace ratio_of_segments_cotangent_l760_760288

variables {A B C D P : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P]
variables (ABCD : ConvexQuadrilateral A B C D)
variables (AC BD : Diagonal A C B D)
variables {AP PC : Segment A P P C}
variables {α β γ δ : Angle A B C D P}

theorem ratio_of_segments_cotangent (h : ∃ P, AC ∩ BD = some P) :
  (AP.to_real / PC.to_real) = 
  ((cot α + cot β) / (cot γ + cot δ)) := 
sorry

end ratio_of_segments_cotangent_l760_760288


namespace distance_to_school_l760_760912

theorem distance_to_school (A_initial_speed B_initial_speed A_new_speed B_new_speed D : ℕ) (h_A_initial_speed : A_initial_speed = 40)
  (h_B_initial_speed : B_initial_speed = 60) (h_A_new_speed : A_new_speed = 60)
  (h_B_new_speed : B_new_speed = 40)
  (h_A_arrives_late : let t := D / B_initial_speed in let t' := (D / 2) / A_initial_speed + (D / 2) / A_new_speed in t' = t + 2) :
  D = 960 := sorry

end distance_to_school_l760_760912


namespace sequence_a_nine_l760_760229

noncomputable def a_seq (n : ℕ) : ℝ :=
if n = 1 then 3 else a_seq (n - 1) + real.logb 3 (1 + 1 / (n - 1))

theorem sequence_a_nine : a_seq 9 = 5 :=
sorry

end sequence_a_nine_l760_760229


namespace eccentricity_of_ellipse_equation_of_ellipse_l760_760593

-- Given conditions
variable (a b : ℝ) (hb : 0 < b) (ha : b < a)
variable (h_eq_ellipse : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, p ∈ E))
variable (M O : ℝ × ℝ) (hM : |(fst B - fst M, snd B - snd M)| = 2 * |(fst A - fst M, snd A - snd M)|) (hOM : slope O M = sqrt 5 / 10)

-- Solution statements
def find_eccentricity_proof : Prop :=
  e = 2 * sqrt 5 / 5

def find_equation_proof : Prop :=
  E = set_of (λ p : ℝ × ℝ, p = ((x^2 / 45 + y^2 / 9) = 1))

/-- Prove the eccentricity -/
theorem eccentricity_of_ellipse (hM : |(fst B - fst M, snd B - snd M)| = 2 * |(fst A - fst M, snd A - snd M)|) : find_eccentricity_proof := sorry

/-- Prove the equation of ellipse -/
theorem equation_of_ellipse (hN : midpoint C B = N) (h_reflection : reflect (|N|) A B = y = 13 / 2) : find_equation_proof := sorry

end eccentricity_of_ellipse_equation_of_ellipse_l760_760593


namespace greatest_two_digit_multiple_of_17_l760_760517

noncomputable def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem greatest_two_digit_multiple_of_17 : 
    ∃ n : ℕ, is_two_digit n ∧ (∃ k : ℕ, n = 17 * k) ∧
    ∀ m : ℕ, is_two_digit m → (∃ k : ℕ, m = 17 * k) → m ≤ n := 
begin
  sorry
end

end greatest_two_digit_multiple_of_17_l760_760517


namespace baseball_games_l760_760765

theorem baseball_games (n_teams : ℕ) (games_per_pair : ℕ) 
    (hne : n_teams = 10) 
    (gpp : games_per_pair = 4) : 
    let unique_games := n_teams * (n_teams - 1) / 2 in 
    games_per_pair * unique_games = 180 := by
  sorry

end baseball_games_l760_760765


namespace production_process_arrangements_l760_760266

theorem production_process_arrangements (workers : Finset ℕ) 
  (A B C : ℕ) (h : {A, B, C} ⊆ workers) (stages : Finset ℕ) (h_stages : stages = {1, 2, 3, 4}) :
  (∃ f : ℕ → ℕ, (f 1 = A ∨ f 1 = B) ∧ (f 4 = A ∨ f 4 = C) ∧
    ∀ i ∈ stages, i ≠ 1 → i ≠ 4 → f i ∈ workers \ {A, B, C} ∧ f i ≠ f 1 ∧ f i ≠ f 4 ∧
    ∀ j k ∈ stages, j ≠ k → f j ≠ f k) → 
  ∃ n : ℕ, n = 36 :=
by
  sorry

end production_process_arrangements_l760_760266


namespace modulus_of_z_range_of_m_l760_760680

noncomputable def modulus_of_complex (z : ℂ) : ℝ :=
  Complex.abs z

theorem modulus_of_z (x : ℝ) (hx : 2 - x = 0 ∧ x + 2 ≠ 0) :
  modulus_of_complex (Complex.mk 2 x * Complex.i) = 2 * Real.sqrt 2 :=
by
  have z : ℂ := Complex.mk 2 x * Complex.i
  have h1 : 2 - x = 0 := hx.left
  have h2 : z = Complex.mk 2 (2 * Complex.i) := by sorry
  have h3 : Complex.abs z = 2 * Real.sqrt 2 := by sorry
  exact h3

theorem range_of_m (z : ℂ) (z1 : ℂ) (hx : 2 - x = 0 ∧ x + 2 ≠ 0) (hm : ℝ) :
  z = Complex.mk 2 x * Complex.i → z1 = (Complex.mk hm (-1) / z) →
  (0 < hm - 1) ∧ (-(hm + 1) < 0) :=
by
  intro hz hz1
  have h1 : z = Complex.mk 2 (2 * Complex.i) := by sorry
  have hz_eq : (Complex.mk hm (-1) / z) = Complex.mk ((hm - 1)/2) ((-hm - 1)/2) := by sorry
  exact ⟨by linarith, by linarith⟩

end modulus_of_z_range_of_m_l760_760680


namespace total_cost_of_shirts_l760_760662

def initial_cost_shirt_group_1 := 15
def discount_first_shirt_group_1 := 0.10
def discount_second_shirt_group_1 := 0.05
def number_of_shirts_group_1 := 3

def initial_cost_shirt_group_2 := 20
def tax_first_shirt_group_2 := 0.05
def tax_second_shirt_group_2 := 0.03
def number_of_shirts_group_2 := 2

noncomputable def final_cost_shirt_group_1 := 
  let first_discounted_price := initial_cost_shirt_group_1 * (1 - discount_first_shirt_group_1)
  let second_discounted_price := first_discounted_price * (1 - discount_second_shirt_group_1)
  number_of_shirts_group_1 * second_discounted_price

noncomputable def final_cost_shirt_group_2 := 
  let first_taxed_price := initial_cost_shirt_group_2 * (1 + tax_first_shirt_group_2)
  let second_taxed_price := first_taxed_price * (1 + tax_second_shirt_group_2)
  number_of_shirts_group_2 * second_taxed_price

noncomputable def total_cost := final_cost_shirt_group_1 + final_cost_shirt_group_2

theorem total_cost_of_shirts : total_cost = 81.735 := 
  by
    sorry

end total_cost_of_shirts_l760_760662


namespace solution_set_of_inequality_l760_760185

theorem solution_set_of_inequality : { x : ℝ | 0 < x ∧ x < 2 } = { x : ℝ | 0 < x ∧ x < 2 } :=
by
  sorry

end solution_set_of_inequality_l760_760185


namespace step_difference_propagation_l760_760916

noncomputable def sum_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem step_difference_propagation (n : ℕ) (hn : n = 2011) :
  let initial_sum := sum_natural_numbers n in
  even initial_sum →
  ∀ last_integer, (∀ (k : ℕ), (k < n → (0 ≤ k ∧ k ≤ n ∧ k ≠ last_integer))) →
  last_integer ≠ 1 :=
by
  let initial_sum := sum_natural_numbers 2011
  have h_initial_sum_even : even initial_sum
  {
    rw sum_natural_numbers
    dsimp [initial_sum]
    norm_num
  }
  intro h
  intro last_integer last_integer_constraint
  have h_last_integer_even : even last_integer
  {
    apply even_of_sum even initial_sum -- Need an appropriate lemma/application
  }
  exact ne_of_even_of_odd h_last_integer_even one_even_false
  sorry

end step_difference_propagation_l760_760916


namespace categorize_numbers_l760_760646

-- Definitions of categorization
def isNegativeFraction (x : ℚ) : Prop := x < 0
def isNonNegativeInteger (x : ℤ) : Prop := x ≥ 0
def isIrrational (x : Real) : Prop := ¬∃ (q : ℚ), x = q

-- Simplifications of the expressions
def eval₁ := -|-3.5| : Real -- evaluates to 3.5
def eval₂ := - 4 / 7 : ℚ   -- evaluates to -4/7
def eval₃ := 0 : ℤ        -- evaluates to 0
def eval₄ := 1 : ℤ        -- evaluates to 1
def eval₅ := -4.012345 : Real  -- evaluates to -4.012345
def eval₆ := - 7 / 100 : ℚ -- evaluates to -7/100
def eval₇ := - 0.1 ^ 3 : Real -- evaluates to -0.001
def eval₈ := 3 : ℤ        -- evaluates to 3
def eval₉ := - Real.pi / 7    -- evaluates to -π/7
def eval₁₀ := 13 / 3 : ℚ  -- evaluates to 13/3

theorem categorize_numbers :
  isNegativeFraction eval₂ ∧ isNegativeFraction (eval₅.toRat) ∧
  isNegativeFraction eval₆ ∧ isNegativeFraction (eval₇.toRat) ∧
  isNonNegativeInteger eval₃ ∧ isNonNegativeInteger eval₄ ∧
  isNonNegativeInteger eval₈ ∧ isIrrational eval₉ :=
by
  sorry

end categorize_numbers_l760_760646


namespace ruler_cost_l760_760967

variable {s c r : ℕ}

theorem ruler_cost (h1 : s > 18) (h2 : r > 1) (h3 : c > r) (h4 : s * c * r = 1729) : c = 13 :=
by
  sorry

end ruler_cost_l760_760967


namespace area_of_sector_of_circle_l760_760937

noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * Real.pi * (r ^ 2)

theorem area_of_sector_of_circle :
  sector_area 12 38 ≈ 47.746 :=
by
  sorry

end area_of_sector_of_circle_l760_760937


namespace greatest_two_digit_multiple_of_17_is_85_l760_760457

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760457


namespace max_soap_boxes_l760_760118

theorem max_soap_boxes 
  (base_width base_length top_width top_length height soap_width soap_length soap_height max_weight soap_weight : ℝ)
  (h_base_dims : base_width = 25)
  (h_base_len : base_length = 42)
  (h_top_width : top_width = 20)
  (h_top_length : top_length = 35)
  (h_height : height = 60)
  (h_soap_width : soap_width = 7)
  (h_soap_length : soap_length = 6)
  (h_soap_height : soap_height = 10)
  (h_max_weight : max_weight = 150)
  (h_soap_weight : soap_weight = 3) :
  (50 = 
    min 
      (⌊top_width / soap_width⌋ * ⌊top_length / soap_length⌋ * ⌊height / soap_height⌋)
      (⌊max_weight / soap_weight⌋)) := by sorry

end max_soap_boxes_l760_760118


namespace kyoto_inequality_l760_760796

variables {n : ℕ} (a : ℕ → ℝ)
hypothesis hn : n ≥ 2
hypothesis ha : ∀ j, 1 ≤ j ∧ j ≤ n → 1/2 < a j ∧ a j < 1

theorem kyoto_inequality (hn : n ≥ 2)
  (ha : ∀ j, 1 ≤ j ∧ j ≤ n → 1/2 < a j ∧ a j < 1) :
  (∏ k in finset.range n, (1 - a (k + 1))) > 
    1 - (a 1 + ∑ k in finset.range (n - 1), (a (k + 2)) / (2 ^ k)) :=
begin
  sorry
end

end kyoto_inequality_l760_760796


namespace sum_primes_less_than_20_l760_760066

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l760_760066


namespace sum_xi_eq_one_l760_760307

variables {n : ℕ} (x : Fin n → ℝ)

def condition1 := ∀ i, 0 ≤ x i

def condition2 := (∑ i, (x i) ^ 2) + 2 * (∑ (k j : Fin n), if k < j then sqrt (k.val / j.val) * x k * x j else 0) = 1

theorem sum_xi_eq_one (h1 : condition1 x) (h2 : condition2 x) :
  ∑ i, x i = 1 :=
sorry

end sum_xi_eq_one_l760_760307


namespace greatest_two_digit_multiple_of_17_l760_760418

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760418


namespace find_Y_value_l760_760632

noncomputable def arithmetic_sequence (a1 a4 : ℕ) : list ℕ :=
  let d := (a4 - a1) / 3 in
  [a1, a1 + d, a1 + 2 * d, a1 + 3 * d]

theorem find_Y_value :
  (arithmetic_sequence 3 18).nth 2 = some 13 →
  (arithmetic_sequence 11 50).nth 2 = some 37 →
  ∃ Y : ℕ, Y = 21 ∧
    match (arithmetic_sequence 3 18),
          (arithmetic_sequence 11 50),
          (arithmetic_sequence 13 (arithmetic_sequence 11 50).nth 3) with
    | [3, 8, 13, 18], [11, 24, 37, 50], [13, 21, 29, 37] => Y = 21
    | _, _, _ => false
    end := by
  sorry

end find_Y_value_l760_760632


namespace rotated_line_equation_l760_760863

-- Define the original equation of the line
def original_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define the rotated line equation we want to prove
def rotated_line (x y : ℝ) : Prop := -x + 2 * y + 4 = 0

-- Proof problem statement in Lean 4
theorem rotated_line_equation :
  ∀ (x y : ℝ), original_line x y → rotated_line x y :=
by
  sorry

end rotated_line_equation_l760_760863


namespace proof_problem_l760_760759

noncomputable def problem : ℕ :=
let AB := 12 in
let BC := 13 in
let AC := 15 in
let AM := 22 / 3 in
let CM := 15 - AM in
let p := 22 in
let q := 23 in
p + q

-- Theorem statement
theorem proof_problem : problem = 45 := 
sorry

end proof_problem_l760_760759


namespace greatest_two_digit_multiple_of_17_l760_760497

theorem greatest_two_digit_multiple_of_17 : ∀ n, n ∈ finset.range 100 → n % 17 = 0 → n ≤ 85 →
  (∀ m, m ∈ finset.range 100 → m % 17 = 0 → m > n → m ≥ 100) →
  n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760497


namespace sum_of_primes_less_than_20_l760_760008

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760008


namespace count_80_points_l760_760733

def heejin_scores : List ℕ := [80, 90, 80, 80, 100, 70]

theorem count_80_points : List.count heejin_scores 80 = 3 := sorry

end count_80_points_l760_760733


namespace textbook_distribution_l760_760568

theorem textbook_distribution (total_books : ℕ) (classroom : ℕ) (library : ℕ) 
  (h1 : total_books = 8) 
  (h2 : classroom > 0) 
  (h3 : classroom < total_books) : 
  ∃ (ways : ℕ), ways = 7 :=
by {
  have library := total_books - classroom,
  have classroom_range := (1 <= classroom) ∧ (classroom <= 7),
  -- The valid distributions are when classroom is 1, 2, 3, 4, 5, 6, 7,
  -- therefore the number of ways is exactly 7
  use 7,
  -- normally, you would put the proof here, but we will use sorry for placeholder
  sorry,
}

end textbook_distribution_l760_760568


namespace convert_to_rectangular_form_l760_760615

theorem convert_to_rectangular_form :
  3 * complex.exp(9 * real.pi * complex.I / 4) = (3 * real.sqrt 2 / 2) + (3 * real.sqrt 2 / 2) * complex.I := 
sorry

end convert_to_rectangular_form_l760_760615


namespace sin_cos_difference_l760_760701

theorem sin_cos_difference (α : ℝ) (h₁ : sin α * cos α = 1 / 8) (h₂ : 0 < α ∧ α < π / 4) :
  sin α - cos α = -√(3 / 4) :=
sorry

end sin_cos_difference_l760_760701


namespace solve_inequality_l760_760649

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x ^ 3 - 3 * x ^ 2 + 2 * x) / (x ^ 2 - 3 * x + 2) ≤ 0 ∧
  x ≠ 1 ∧ x ≠ 2

theorem solve_inequality :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | x ≤ 0 ∧ x ≠ 1 ∧ x ≠ 2} :=
  sorry

end solve_inequality_l760_760649


namespace greatest_two_digit_multiple_of_17_is_85_l760_760466

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l760_760466


namespace shorten_card_area_l760_760792

theorem shorten_card_area (initial_length initial_width : ℕ) 
  (h_length : initial_length = 5) (h_width : initial_width = 7) : 
  let new_length := initial_length - 1,
      new_width := initial_width - 1
  in new_length * new_width = 24 := 
by
  rw [h_length, h_width]
  dsimp [new_length, new_width]
  norm_num

end shorten_card_area_l760_760792


namespace sum_primes_less_than_20_l760_760087

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760087


namespace ratio_area_smaller_to_larger_correct_sum_m_n_l760_760292

noncomputable def regular_octagon := 
  sorry  -- definition of regular octagon

def is_midpoint (a b m : Point) :=
  sorry  -- definition of midpoint

def inside_smaller_octagon (a b c d e f g h p q r s t u v w : Point) :=
  sorry  -- definition to check if points form the described smaller octagon inside the larger one

theorem ratio_area_smaller_to_larger (A B C D E F G H P Q R S T U V W : Point)
  (h1 : regular_octagon A B C D E F G H)
  (h2 : is_midpoint A B P) (h3 : is_midpoint B C Q) (h4 : is_midpoint C D R)
  (h5 : is_midpoint D E S) (h6 : is_midpoint E F T) (h7 : is_midpoint F G U)
  (h8 : is_midpoint G H V) (h9 : is_midpoint H A W)
  (h10 : inside_smaller_octagon A B C D E F G H P Q R S T U V W) :
  let small_area_ratio := 1 / 4 in
  small_area_ratio = real.to_rat (1 / 4) :=
sorry

theorem correct_sum_m_n :
  let m := 1 in
  let n := 4 in
  m + n = 5 :=
sorry

end ratio_area_smaller_to_larger_correct_sum_m_n_l760_760292


namespace ounces_per_gallon_l760_760171

theorem ounces_per_gallon (bowls_per_minute : ℕ) (ounces_per_bowl : ℕ) 
  (gallons : ℕ) (serving_time : ℕ) 
  (H1 : bowls_per_minute = 5) (H2 : ounces_per_bowl = 10)
  (H3 : gallons = 6) (H4 : serving_time = 15) : 
  (bowls_per_minute * ounces_per_bowl * serving_time) / gallons = 125 :=
by
  rw [H1, H2, H3, H4]
  norm_num
  sorry

end ounces_per_gallon_l760_760171


namespace minimum_a_l760_760750

theorem minimum_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → 
  a ≥ -5/2 :=
sorry

end minimum_a_l760_760750


namespace unique_single_digit_A_l760_760363

noncomputable def quadratic_has_positive_integer_solutions (A : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ x * y = 3*A ∧ x + y = 2*A + 1

theorem unique_single_digit_A :
  {A : ℕ | A > 0 ∧ A < 10 ∧ quadratic_has_positive_integer_solutions A}.card = 1 := 
sorry

end unique_single_digit_A_l760_760363


namespace bead_when_count_to_100_l760_760273

variable (n : ℕ) -- Natural numbers
def contains_digit_seven (n : ℕ) : Prop :=
  let digits := n.digits 10
  7 ∈ digits

def is_multiple_of_seven (n : ℕ) : Prop :=
  n % 7 = 0

def should_skip (n : ℕ) : Prop :=
  contains_digit_seven n ∨ is_multiple_of_seven n

-- An auxiliary function to count to the kth valid number considering the skipping rules.
def next_valid_number (k : ℕ) : ℕ :=
  (List.range (2 * k + 1)).filter (λ n, ¬ should_skip n) !! (k - 1)

-- Circular sequence of beads where the count starts with bead 1
def bead_on_count (k : ℕ) : ℕ := ((next_valid_number k) % 22) + 1

-- The main theorem stating the problem and solution.
theorem bead_when_count_to_100 : bead_on_count 100 = 4 :=
  sorry

end bead_when_count_to_100_l760_760273


namespace volume_regular_octahedron_l760_760621

theorem volume_regular_octahedron (R : ℝ) : volume (regular_octahedron_circumscribed_around_sphere R) = 4 * R^3 * real.sqrt 3 := 
sorry

end volume_regular_octahedron_l760_760621


namespace sum_primes_less_than_20_l760_760075

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760075


namespace abigail_spent_in_store_l760_760584

theorem abigail_spent_in_store (initial_amount : ℕ) (amount_left : ℕ) (amount_lost : ℕ) (spent : ℕ) 
  (h1 : initial_amount = 11) 
  (h2 : amount_left = 3)
  (h3 : amount_lost = 6) :
  spent = initial_amount - (amount_left + amount_lost) :=
by
  sorry

end abigail_spent_in_store_l760_760584


namespace greatest_two_digit_multiple_of_17_l760_760449

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate for being a multiple of 17
def is_multiple_of_17 (n : ℕ) := ∃ k : ℕ, n = 17 * k

-- The specific problem to prove
theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ∈ two_digit_numbers ∧ is_multiple_of_17(n) ∧ 
  (∀ m : ℕ, m ∈ two_digit_numbers ∧ is_multiple_of_17(m) → n ≥ m) ∧ n = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l760_760449


namespace shift_and_expand_l760_760526

theorem shift_and_expand (x : ℝ) : 
  let y := 3*(x+6)^2 + 5*(x+6) + 9
  let a := 3
  let b := 41
  let c := 147
  a + b + c = 191 :=
by
  let y := 3*(x+6)^2 + 5*(x+6) + 9
  let a := 3
  let b := 41
  let c := 147
  show a + b + c = 191,
  from sorry

end shift_and_expand_l760_760526


namespace sum_of_primes_less_than_20_l760_760000

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l760_760000


namespace johns_beef_order_l760_760283

theorem johns_beef_order (B : ℕ)
  (h1 : 8 * B + 6 * (2 * B) = 14000) :
  B = 1000 :=
by
  sorry

end johns_beef_order_l760_760283


namespace ratio_mara_janet_l760_760281

variables {B J M : ℕ}

/-- Janet has 9 cards more than Brenda --/
def janet_cards (B : ℕ) : ℕ := B + 9

/-- Mara has 40 cards less than 150 --/
def mara_cards : ℕ := 150 - 40

/-- They have a total of 211 cards --/
axiom total_cards_eq (B : ℕ) : B + janet_cards B + mara_cards = 211

/-- Mara has a multiple of Janet's number of cards --/
axiom multiples_cards (J M : ℕ) : J * 2 = M

theorem ratio_mara_janet (B J M : ℕ) (h1 : janet_cards B = J)
  (h2 : mara_cards = M) (h3 : J * 2 = M) :
  (M / J : ℕ) = 2 :=
sorry

end ratio_mara_janet_l760_760281


namespace find_tan_2x_l760_760223

variable (x : ℝ)

def f (x : ℝ) := sin x + cos x

theorem find_tan_2x (h : deriv f x = 3 * f x) : tan (2 * x) = -4 / 3 := by
  have h1 : f x = sin x + cos x := rfl
  sorry

end find_tan_2x_l760_760223


namespace tangent_line_passing_origin_l760_760364

noncomputable def f (x : ℝ) := x^2 - x * Real.log x + 2

theorem tangent_line_passing_origin :
  ∃ (k : ℝ), ∀ x, (f x = k * x) ↔ (k = 3 - Real.log 2) :=
begin
  sorry
end

end tangent_line_passing_origin_l760_760364


namespace part1_part2_l760_760299

open Real

def f (t x : ℝ) : ℝ := x^2 - (t + 1) * x + t

theorem part1 : 
  ∃ sol_set : Set ℝ, 
    ({x : ℝ | f 3 x > 0} = sol_set) ∧ 
    (sol_set = {x : ℝ | x < 1} ∪ {x : ℝ | x > 3}) :=
by
  use {x : ℝ | x < 1} ∪ {x : ℝ | x > 3}
  sorry

theorem part2 :
  (∀ x : ℝ, f t x ≥ 0) ↔ (t = 1) :=
by
  split
  sorry
  sorry

end part1_part2_l760_760299


namespace expected_heads_value_in_cents_l760_760124

open ProbabilityTheory

-- Define the coins and their respective values
def penny_value := 1
def nickel_value := 5
def half_dollar_value := 50
def dollar_value := 100

-- Define the probability of landing heads for each coin
def heads_prob := 1 / 2

-- Define the expected value function
noncomputable def expected_value_of_heads : ℝ :=
  heads_prob * (penny_value + nickel_value + half_dollar_value + dollar_value)

theorem expected_heads_value_in_cents : expected_value_of_heads = 78 := by
  sorry

end expected_heads_value_in_cents_l760_760124


namespace varphi_proof_l760_760945

noncomputable def varphi_solution (x λ : ℝ) : ℝ :=
if λ = 2 / π then
  λ' * cos x + cos (3 * x)  -- where λ' is some constant related to λ
else if λ = -2 / π then
  λ' * sin x + cos (3 * x)  -- where λ' is some constant related to λ
else
  cos (3 * x)

theorem varphi_proof (x λ : ℝ) (C : ℝ) :
  varphi_solution x λ = cos (3 * x) - λ * ∫ (t : ℝ) in 0..π, cos (x + t) * varphi_solution t λ :=
begin
  sorry,
end

end varphi_proof_l760_760945


namespace value_of_y_l760_760743

-- Problem: Prove that given the conditions \( x - y = 8 \) and \( x + y = 16 \),
-- the value of \( y \) is 4.
theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := 
sorry

end value_of_y_l760_760743


namespace greatest_two_digit_multiple_of_17_l760_760419

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (17 ∣ n) ∧ n = 85 :=
by {
    use 85,
    split,
    { exact ⟨le_of_eq rfl, lt_of_le_of_ne (nat.le_of_eq rfl) (ne.symm (dec_trivial))⟩ },
    split,
    { exact dvd.intro 5 rfl },
    { refl }
}

end greatest_two_digit_multiple_of_17_l760_760419


namespace min_value_a_squared_plus_2b_squared_l760_760716

theorem min_value_a_squared_plus_2b_squared (a b : ℝ) (h₀ : f(x) = x * (x - a) * (x - b)) 
    (h₁ : f'(0) = 4) : a^2 + 2 * b^2 = 8 * sqrt 2 :=
by {
    sorry
}

end min_value_a_squared_plus_2b_squared_l760_760716


namespace grapefruit_touch_points_coplanar_l760_760391

-- Given a hemispherical vase (with a flat lid),
-- four identical oranges (all touching the vase),
-- and one grapefruit (touching all four oranges),
-- prove that all four points where the grapefruit touches the oranges necessarily lie in one plane.
theorem grapefruit_touch_points_coplanar
  (V G : Point)   -- Centers of the vase and grapefruit
  (A1 A2 A3 A4 : Point)  -- Centers of the 4 oranges
  (K1 K2 K3 K4 : Point)  -- Points where grapefruit touches the oranges
  (P1 P2 P3 P4 : Point)  -- Points where oranges touch the vase
  (v g a : ℝ)  -- Radii of vase, grapefruit, and oranges
  (hv : ∃ (u : ℝ), u ≠ 0 ∧ V = (0, 0, -u))  -- Hemispheric vase with flat top
  (h_va_eq : ∀ i, i ∈ {1, 2, 3, 4} → dist V Ai = v - a)  -- Distances from V to oranges
  (h_ga_eq : ∀ i, i ∈ {1, 2, 3, 4} → dist G Ai = g + a)  -- Distances from G to oranges
  (h_kg_eq : ∀ i, i ∈ {1, 2, 3, 4} → dist G Ki = g)  -- Distances from G to grapefruit contact points
  (h_ka_eq : ∀ i, i ∈ {1, 2, 3, 4} → dist Ki Ai = a)  -- Distances from oranges to grapefruit contact points
  : ∃ Π, ∀ i j, i ∈ {1, 2, 3, 4} → j ∈ {1, 2, 3, 4} → i ≠ j → ∃ ℓ, ℓ i = ℓ j ∧ ∀ k ≠ i j, ℓ i ≠ ℓ k :=
sorry

end grapefruit_touch_points_coplanar_l760_760391


namespace sum_of_primes_less_than_20_eq_77_l760_760016

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760016


namespace greatest_two_digit_multiple_of_17_l760_760432

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760432


namespace greatest_two_digit_multiple_of_17_l760_760428

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l760_760428


namespace sum_primes_less_than_20_l760_760089

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760089


namespace sum_of_primes_less_than_20_eq_77_l760_760012

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l760_760012


namespace sum_primes_less_than_20_l760_760086

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l760_760086


namespace prime_gt_5_divides_gn_gn1_l760_760620

-- Define the function g(n)
def g : ℕ → ℕ
| 1     := 0
| 2     := 1
| (n+2) := g n + g (n+1) + 1

theorem prime_gt_5_divides_gn_gn1 (n : ℕ) (h_prime : Nat.Prime n) (h_gt : n > 5) : 
  n ∣ g n * (g n + 1) :=
sorry

end prime_gt_5_divides_gn_gn1_l760_760620


namespace arithmetic_sequence_sum_l760_760871

theorem arithmetic_sequence_sum (c d : ℤ) (h1 : c = 24) (h2 : d = 31) :
  c + d = 55 :=
by
  rw [h1, h2]
  exact rfl

end arithmetic_sequence_sum_l760_760871


namespace greatest_two_digit_multiple_of_17_l760_760487

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n < 100 ∧ 10 ≤ n ∧ 17 ∣ n ∧ ∀ m : ℕ, m < 100 → 10 ≤ m → 17 ∣ m → m ≤ n :=
begin
  use 85,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    by_contradiction hmn,
    have h : m > 85 := not_le.mp hmn,
    linear_combination h * 17,
    sorry,
  },
end

end greatest_two_digit_multiple_of_17_l760_760487


namespace remainder_is_20_l760_760861

theorem remainder_is_20 :
  ∀ (larger smaller quotient remainder : ℕ),
    (larger = 1634) →
    (larger - smaller = 1365) →
    (larger = quotient * smaller + remainder) →
    (quotient = 6) →
    remainder = 20 :=
by
  intros larger smaller quotient remainder h_larger h_difference h_division h_quotient
  sorry

end remainder_is_20_l760_760861
