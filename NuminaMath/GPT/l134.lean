import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.GroupPower.Lemmas
import Mathlib.Analysis.Geometry.Angle
import Mathlib.Analysis.NormalDistribution
import Mathlib.Analysis.SpecialFunctions.Abs
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.GCD
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Irrational
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Intervals.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.Probability.Kolmogorov
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Pigeonhole

namespace simplify_trig_expression_l134_134081

theorem simplify_trig_expression :
  (cos (40 * Real.pi / 180) / (cos (25 * Real.pi / 180) * sqrt (1 - sin (40 * Real.pi / 180)))) = sqrt 2 :=
by sorry

end simplify_trig_expression_l134_134081


namespace distinct_natural_sum_ineq_l134_134326

theorem distinct_natural_sum_ineq (a : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) (n : ℕ) (h_pos : 0 < n) :
  (∑ k in finset.range n, (a k) / (k+1)^2 : ℚ) ≥ (∑ k in finset.range n, 1 / (k+1) : ℚ) :=
sorry

end distinct_natural_sum_ineq_l134_134326


namespace imaginary_unit_cube_l134_134189

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i :=
by
  sorry

end imaginary_unit_cube_l134_134189


namespace volume_ratio_l134_134493

noncomputable def probability_point_in_g
  (G : Set ℝ^3) (g : Set ℝ^3) (pointIn : ℝ^3 → Prop) : ℝ :=
  let ellipsoid := {p : ℝ^3 | (p.1^2 / 16) + (p.2^2 / 9) + (p.3^2 / 4) = 1}
  let sphere := {p : ℝ^3 | (p.1^2 + p.2^2 + p.3^2) = 4}

  let volume_G := (4 / 3) * Real.pi * 4 * 3 * 2
  let volume_sphere := (4 / 3) * Real.pi * (2^3)
  let volume_g := volume_G - volume_sphere

  volume_g / volume_G

theorem volume_ratio 
  (hG : {p : ℝ^3 | (p.1^2 / 16) + (p.2^2 / 9) + (p.3^2 / 4) = 1})
  (hg : {p : ℝ^3 | (p.1^2 / 16) + (p.2^2 / 9) + (p.3^2 / 4) = 1 ∧ (p.1^2 + p.2^2 + p.3^2) = 4}) :
  probability_point_in_g hG hg (λ p, true) = 2 / 3 :=
by
  sorry

end volume_ratio_l134_134493


namespace equal_distances_l134_134328

-- Define the trianglular context and the mentioned points
variables (A B C A' B' C' K L : Type*)
variable [plane_geom A B C A' B' C' K L]

-- Define incircle touching points
axiom incircle_touches (h1 : touches_incircle A B C A' B' C')

-- Define the angle condition
axiom angle_cond (h2 : ∠AKB' + ∠BKA' = 180° ∧ ∠ALB' + ∠BLA' = 180°)

theorem equal_distances (h1 : touches_incircle A B C A' B' C')
                         (h2 : ∠AKB' + ∠BKA' = 180° ∧ ∠ALB' + ∠BLA' = 180°) : 
  distance_to_line A' K L = distance_to_line B' K L ∧ 
  distance_to_line A' K L = distance_to_line C' K L :=
sorry

end equal_distances_l134_134328


namespace fraction_is_integer_iff_l134_134841

def is_integer (x : ℚ) : Prop := ∃ (z : ℤ), ↑z = x

theorem fraction_is_integer_iff 
  (n k : ℕ) : is_integer (↑((∏ i in finset.range (k+1), n + i) / n : ℚ)) ↔ n ∣ k! ∨ n = k! :=
by sorry

end fraction_is_integer_iff_l134_134841


namespace bulbs_probability_l134_134900

/-
There are 100 light bulbs initially all turned on.
Every second bulb is toggled after one second.
Every third bulb is toggled after two seconds, and so on.
This process continues up to 100 seconds.
We aim to prove the probability that a randomly selected bulb is on after 100 seconds is 0.1.
-/

theorem bulbs_probability : 
  let bulbs := {1..100}
  let perfect_squares := {n ∈ bulbs | ∃ k, n = k * k}
  let total_bulbs := 100
  let bulbs_on := Set.card perfect_squares
  (bulbs_on : ℕ) / total_bulbs = 0.1 :=
by
  -- We use the solution steps to take perfect squares directly
  have h : Set.card perfect_squares = 10 := sorry
  have t : total_bulbs = 100 := rfl
  rw [h]
  norm_num


end bulbs_probability_l134_134900


namespace solve_for_x_l134_134460

theorem solve_for_x : ∃ x : ℝ, 2^(x - 3) = 4^(x + 1) ∧ x = -5 :=
by {
  use -5,
  split,
  sorry,
  refl
}

end solve_for_x_l134_134460


namespace at_least_one_greater_than_one_l134_134840

open Classical

variable (x y : ℝ)

theorem at_least_one_greater_than_one (h : x + y > 2) : x > 1 ∨ y > 1 :=
by
  sorry

end at_least_one_greater_than_one_l134_134840


namespace problem1_problem2_l134_134571

theorem problem1 (x : ℝ) : 
  (sqrt(x^2 - 3 * x + 2) > x + 5) → 
  x < -23 / 13 :=
by sorry

theorem problem2 (k : ℝ) : 
  (∀ x : ℝ, (2 * x^2 + 2 * k * x + k) / (4 * x^2 + 6 * x + 3) < 1) → 
  1 < k ∧ k < 3 :=
by sorry

end problem1_problem2_l134_134571


namespace tangent_and_normal_lines_l134_134983

theorem tangent_and_normal_lines (x y : ℝ → ℝ) (t : ℝ) (t₀ : ℝ) 
  (h0 : t₀ = 0) 
  (h1 : ∀ t, x t = (1/2) * t^2 - (1/4) * t^4) 
  (h2 : ∀ t, y t = (1/2) * t^2 + (1/3) * t^3) :
  (∃ m : ℝ, y (x t₀) = m * (x t₀) ∧ m = 1) ∧
  (∃ n : ℝ, y (x t₀) = n * (x t₀) ∧ n = -1) :=
by 
  sorry

end tangent_and_normal_lines_l134_134983


namespace Sarah_shampoo_conditioner_usage_l134_134076

theorem Sarah_shampoo_conditioner_usage (daily_shampoo : ℝ) (daily_conditioner : ℝ) (days_in_week : ℝ) (weeks : ℝ) (total_days : ℝ) (daily_total : ℝ) (total_usage : ℝ) :
  daily_shampoo = 1 → 
  daily_conditioner = daily_shampoo / 2 → 
  days_in_week = 7 → 
  weeks = 2 → 
  total_days = days_in_week * weeks → 
  daily_total = daily_shampoo + daily_conditioner → 
  total_usage = daily_total * total_days → 
  total_usage = 21 := by
  sorry

end Sarah_shampoo_conditioner_usage_l134_134076


namespace base_10_sum_to_base_5_correct_l134_134626

def convert_to_base_5 (n : Nat) : List Nat :=
  if n < 5 then [n]
  else (n % 5) :: convert_to_base_5 (n / 5)

def base_5_add (a b : List Nat) : List Nat :=
  let rec add_aux (a b : List Nat) (carry : Nat) : List Nat :=
    match a, b, carry with
    | [], [], 0 => []
    | [], [], carry => [carry]
    | a, [], carry => add_aux a [0] carry
    | [], b, carry => add_aux [0] b carry
    | (ha :: ta), (hb :: tb), carry =>
      let sum := ha + hb + carry
      sum % 5 :: add_aux ta tb (sum / 5)
  add_aux a b 0

def base_10_to_base_5_sum (a b : Nat) : List Nat :=
  base_5_add (convert_to_base_5 a) (convert_to_base_5 b)

theorem base_10_sum_to_base_5_correct :
  base_10_to_base_5_sum 37 29 = [1, 3, 2] := 
by
  sorry

end base_10_sum_to_base_5_correct_l134_134626


namespace find_m_value_l134_134694

theorem find_m_value (α : ℝ) (m : ℝ) (h1 : m > 0)
  (h2 : ∀ α, m * real.sin α - 2 * real.sin (α + real.pi / 6) ≤ 2) :
  m = 2 * real.sqrt 3 :=
by
  sorry

end find_m_value_l134_134694


namespace proof_of_correct_props_l134_134624

/- Definitions for the specific propositions -/

def prop1 (A B P : EuclideanSpace ℝ 2) (k : ℝ) : Prop := abs (dist A P - dist B P) = k → ∃ c d, c ≠ 0 ∧ d ≠ 0 ∧ ∀ (P : EuclideanSpace ℝ 2), abs (dist P A - dist P B) = c - d

def prop2 (k : ℝ) : Prop := ∀ (n : ℕ), let s_n := 2^n + k in k = -1

def prop3 : Prop := ∀ (x : ℝ), 0 < x → 2^x + 2^(-x) = 2

def prop4 : Prop := let foci_h := (sqrt 34, 0) in let foci_e := (sqrt 34, 0) in foci_h = foci_e

def prop5 : Prop := ∀ (P : EuclideanSpace ℝ 2), P = (3, -1) → ∃ l : AffineSubspace ℝ (EuclideanSpace ℝ 2), ∀ Q, dist Q P = dist Q l

/- The main theorem that combines all propositions and proves only (2) and (4) are correct -/

theorem proof_of_correct_props :
  (prop2 -1) ∧ prop4 :=
by
  sorry

end proof_of_correct_props_l134_134624


namespace circle_add_center_and_radius_eq_l134_134418

noncomputable def circle_center_and_radius (x y : ℝ) : ℝ × ℝ × ℝ :=
let a : ℝ := 12
let b : ℝ := 2
let r : ℝ := 4 * Real.sqrt 7
in (a, b, r)

theorem circle_add_center_and_radius_eq :
  (let ⟨a, b, r⟩ := circle_center_and_radius x y
   in a + b + r) = 14 + 4 * Real.sqrt 7 :=
by
  sorry

end circle_add_center_and_radius_eq_l134_134418


namespace intersection_proof_1_square_proof_2_l134_134886

section Geometry

variables (A B C D E F G H K L M : Type)
variables [point A] [point B] [point C] [point D] [point E] [point F] [point G] [point H] [point K] [point L] [point M]

-- Assumptions
def polyhedron_tetrahedron : Prop := ∀ (B C D E : Type), regular_tetrahedron B C D E ∧ base_square B C D E

def segment_A_to_F : Prop := ∀ (A F : Type), on_segment A F ∧ A_distance_equivalent_to A F
def segment_A_to_G : Prop := ∀ (A G : Type), on_segment A G ∧ A_distance_equivalent_to A G
def segment_A_to_H : Prop := ∀ (A H : Type), on_segment A H ∧ A_distance_equivalent_to A H

-- Proof Problem 1
theorem intersection_proof_1 (polyhedron_tetrahedron A B C D E F G H) (segment_A_to_F A F) (segment_A_to_G A G) (segment_A_to_H A H) (EF_in_plane : Prop) (DG_in_plane : Prop):
  EF ∩ DG = K ∧ BG ∩ EH = L :=
sorry

-- Proof Problem 2
theorem square_proof_2 (intersection_proof_1: EF ∩ DG = K ∧ BG ∩ EH = L) (EG_in_plane : Prop) (plane_AKL : Prop) : AKML_square := 
sorry

end Geometry

end intersection_proof_1_square_proof_2_l134_134886


namespace total_wheels_in_garage_l134_134919

def bicycles: Nat := 3
def tricycles: Nat := 4
def unicycles: Nat := 7

def wheels_per_bicycle: Nat := 2
def wheels_per_tricycle: Nat := 3
def wheels_per_unicycle: Nat := 1

theorem total_wheels_in_garage (bicycles tricycles unicycles wheels_per_bicycle wheels_per_tricycle wheels_per_unicycle : Nat) :
  bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle = 25 := by
  sorry

end total_wheels_in_garage_l134_134919


namespace tiling_rect_divisible_by_4_l134_134211

theorem tiling_rect_divisible_by_4 (m n : ℕ) (h : ∃ k l : ℕ, m = 4 * k ∧ n = 4 * l) : 
  (∃ a : ℕ, m = 4 * a) ∧ (∃ b : ℕ, n = 4 * b) :=
by 
  sorry

end tiling_rect_divisible_by_4_l134_134211


namespace cost_price_computer_table_l134_134978

def cost_price (p : ℝ) (markup : ℝ → ℝ) : ℝ := p / markup 1.20

theorem cost_price_computer_table :
  cost_price 8337 id = 6947.5 :=
begin
  sorry,
end

end cost_price_computer_table_l134_134978


namespace correct_relationship_l134_134103

-- Define the given points
def points : List (ℕ × ℕ) := [(0, 80), (1, 70), (2, 60), (3, 50), (4, 40)]

-- Define the candidate functions
def f_A (x : ℕ) : ℕ := 80 - 10 * x
def f_B (x : ℕ) : ℕ := 80 - 5 * x ^ 3
def f_C (x : ℕ) : ℕ := 80 - 5 * x - 5 * x ^ 2
def f_D (x : ℕ) : ℕ := 80 - x - x ^ 2

-- Lean theorem statement
theorem correct_relationship :
  ∀ (p : ℕ × ℕ), p ∈ points → p.snd = f_A p.fst :=
by
  intros p hp
  cases hp with
  | inl h => simp [f_A, h]
  | inr hp1 =>
    cases hp1 with
    | inl h => simp [f_A, h]
    | inr hp2 =>
      cases hp2 with
      | inl h => simp [f_A, h]
      | inr hp3 =>
        cases hp3 with
        | inl h => simp [f_A, h]
        | inr hp4 => simp [f_A, hp4]

end correct_relationship_l134_134103


namespace custom_op_evaluation_l134_134742

def custom_op (x y : ℕ) : ℕ := x * y - 3 * x

theorem custom_op_evaluation : (custom_op 8 5) - (custom_op 5 8) = -9 := by
    have h1 : custom_op 8 5 = 8 * 5 - 3 * 8 := rfl
    have h2 : custom_op 5 8 = 5 * 8 - 3 * 5 := rfl
    calc
        (custom_op 8 5) - (custom_op 5 8)
            = (8 * 5 - 3 * 8) - (5 * 8 - 3 * 5) : by rw [h1, h2]
        ... = (40 - 24) - (40 - 15) : rfl
        ... = 16 - 25 : rfl
        ... = -9 : rfl

end custom_op_evaluation_l134_134742


namespace product_of_two_numbers_l134_134475

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 153) : x * y = 36 :=
by
  sorry

end product_of_two_numbers_l134_134475


namespace number_of_special_four_digit_numbers_l134_134267

theorem number_of_special_four_digit_numbers : 
  ∃ (n : ℕ), n = 20 ∧ 
  (∀ (num : ℕ), 1000 ≤ num ∧ num ≤ 9999 → 
     (∀ k < 4, even (num / 10^k % 10)) → 
     (num % 10 = 0) → 
     (num / 100 % 10 = 0) → 
     true) :=
by
  sorry

end number_of_special_four_digit_numbers_l134_134267


namespace polar_to_rectangular_transformation_l134_134208

theorem polar_to_rectangular_transformation : 
  ∀ (x y : ℝ), 
  x = 5 ∧ y = 12 → 
  (r θ : ℝ) (hr : r = Real.sqrt (x^2 + y^2)) (ht : θ = Real.atan2 y x) (k : ℝ) (hk : k = 2),
  let r' := k * r,
      θ' := θ + Real.pi / 4,
      x' := r' * Real.cos θ',
      y' := r' * Real.sin θ' in
  (x' = -7 * Real.sqrt 2 ∧ y' = 17 * Real.sqrt 2) :=
by
  intros x y h (r θ : ℝ) hr ht k hk,
  sorry

end polar_to_rectangular_transformation_l134_134208


namespace monotonicity_of_f_f_has_exactly_two_zeros_extremum_point_and_zero_l134_134340

section Part1

variable (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
variable (h_f_def : ∀ x, f x = ln (x + 1) - a * x * exp (x + 1))
variable (h_a_le_0 : a ≤ 0)
variable (h_f'_def : ∀ x, f' x = 1 / (x + 1) - a * (x + 1) * exp (x + 1))

theorem monotonicity_of_f : ∀ x, f' x > 0 :=
by
  sorry

end Part1

section Part2

variable (a : ℝ) (f : ℝ → ℝ) (x0 x1 : ℝ)
variable (h_f_def : ∀ x, f x = ln (x + 1) - a * x * exp (x + 1))
variable (h_a_bound : 0 < a ∧ a < 1 / exp 1)

theorem f_has_exactly_two_zeros : ∃ x y, f x = 0 ∧ f y = 0 ∧ x ≠ y :=
by
  sorry

theorem extremum_point_and_zero (h_extremum : ∀ t, x0 = t ↔ f' t = 0) 
(h_zero : f x1 = 0) (h_ineq : x1 > x0) : 3 * x0 > x1 :=
by
  sorry

end Part2

end monotonicity_of_f_f_has_exactly_two_zeros_extremum_point_and_zero_l134_134340


namespace area_of_triangle_ACE_l134_134141

-- Definitions for the given problem
variables {A B C D E : Type} 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ E]

-- Given conditions
noncomputable def AB : ℝ := 8
noncomputable def AC : ℝ := 12
noncomputable def BD : ℝ := 8
noncomputable def right_triangle_ABC := is_right_triangle A B C
noncomputable def right_triangle_ABD := is_right_triangle A B D

-- Problem statement
theorem area_of_triangle_ACE : 
  ∀ (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ E],
  AB = 8 → AC = 12 → BD = 8 → right_triangle_ABC → right_triangle_ABD →
  area (triangle A C E) = 28.8 :=
by
  intros A B C D E _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
  sorry

end area_of_triangle_ACE_l134_134141


namespace B_work_days_l134_134576

noncomputable def A_work_rate := 1 / 20  -- A's work rate
noncomputable def B_work_rate (x : ℝ) := 1 / x  -- B's work rate, to be determined
noncomputable def C_work_rate := 1 / 50  -- C's work rate

-- Total work summation in 11 days by A, B, and C.
theorem B_work_days : 
  ∃ x : ℝ, 
    (6 * (A_work_rate + B_work_rate x) + 5 * (A_work_rate + C_work_rate) = 1) ∧ 
    (x ≈ 17.14) := 
sorry

end B_work_days_l134_134576


namespace triangle_is_right_triangle_l134_134830

variable (A B C : ℂ)
variable hA : A = 1
variable hB : B = 2 * Complex.i
variable hC : C = 5 + 2 * Complex.i

theorem triangle_is_right_triangle (hA : A = 1) (hB : B = 2 * Complex.i) (hC : C = 5 + 2 * Complex.i) :
  let AB := B - A,
      BC := C - B,
      CA := A - C in
  Complex.abs(AB)^2 + Complex.abs(CA)^2 = Complex.abs(BC)^2 :=
by
  sorry

end triangle_is_right_triangle_l134_134830


namespace cos_sum_to_value_l134_134887

theorem cos_sum_to_value :
  cos (13 * real.pi / 180) * cos (32 * real.pi / 180) 
  - sin (13 * real.pi / 180) * cos (32 * real.pi / 180) 
  = real.sqrt 2 / 2 := 
sorry

end cos_sum_to_value_l134_134887


namespace calculate_expression_l134_134240

theorem calculate_expression : abs (-1/2) + (2023 - Real.pi)^0 - real.cbrt 27 = -3/2 :=
by sorry

end calculate_expression_l134_134240


namespace polynomial_identity_l134_134548

theorem polynomial_identity (p : ℕ → ℕ) :
  (∀ x, p(x) * p(x + 1) = p(x + p(x))) →
  (p = (λ x, 0) ∨ p = (λ x, 1) ∨ (∃ b c, ∀ x, p(x) = x^2 + b*x + c)) :=
by
  sorry

end polynomial_identity_l134_134548


namespace five_digit_numbers_arithmetic_sequence_count_l134_134225

open Finset

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b ∧ a < b ∧ b < c

def five_digit_numbers_count : ℕ :=
  card (filter (λ n : ℕ,
    let d1 := n / 10000, d2 := n / 1000 % 10, d3 := n / 100 % 10, d4 := n / 10 % 10, d5 := n % 10 in
    n >= 10000 ∧ n < 100000 ∧
    nodup [d1, d2, d3, d4, d5] ∧
    is_arithmetic_sequence d2 d3 d4)
  (range 100000))

theorem five_digit_numbers_arithmetic_sequence_count :
  five_digit_numbers_count = 744 :=
  sorry

end five_digit_numbers_arithmetic_sequence_count_l134_134225


namespace cube_faces_sum_l134_134203

theorem cube_faces_sum (x : ℤ) (h1 : Pairwise (λ i j, abs (i - j) = 1) [x, x+1, x+2, x+3, x+4, x+5])
  (h2 : (x + x + 5) = (x + 1 + x + 4) ∧ (x + 1 + x + 4) = (x + 2 + x + 3))
  (h3 : Prime (x + 5 - x)) : (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) = 87) :=
by
  -- Placeholder for the proof, not required
  sorry

end cube_faces_sum_l134_134203


namespace simplify_fraction_l134_134458

theorem simplify_fraction (h1 : 222 = 2 * 3 * 37) (h2 : 8888 = 8 * 11 * 101) :
  (222 / 8888) * 22 = 1 / 2 :=
by
  sorry

end simplify_fraction_l134_134458


namespace perpendicular_bisector_MN_through_P_l134_134837

/-- A quadrilateral ABCD with points M and N being the midpoints of sides AD and BC respectively.
    Perpendicular bisectors of sides AB and CD intersect at point P. Prove that the perpendicular 
    bisector of segment MN passes through point P. -/
theorem perpendicular_bisector_MN_through_P
  (A B C D M N P : EuclideanSpace ℝ (Fin 2)) 
  (hM : M = (A + D) / 2) 
  (hN : N = (B + C) / 2) 
  (hAD_BC : ∥A - D∥ = ∥B - C∥) 
  (hP_AB_perp : is_perpendicular_bisector P A B)
  (hP_CD_perp : is_perpendicular_bisector P C D) : 
  is_perpendicular_bisector P M N :=
sorry

end perpendicular_bisector_MN_through_P_l134_134837


namespace coefficient_of_m3n5_in_expansion_l134_134525

theorem coefficient_of_m3n5_in_expansion :
  ∑ k in finset.range (9), (nat.choose 8 k) * (m ^ (8 - k)) * (n ^ k) = 56 :=
by
  -- We will write out the specifically concerned term
  have t_eq : (nat.choose 8 5) * (m ^ 3) * (n ^ 5) = 56 := by
  -- The critical calculation step, manually verified
  rw [nat.choose_symm_diff 8 5, nat.choose_self, mul_comm, mul_assoc],
  ring,
  exact 56,
  sorry
-- The theorem states exactly what needs to be proved

end coefficient_of_m3n5_in_expansion_l134_134525


namespace students_month_l134_134927

theorem students_month (students : ℕ) (months : ℕ) (h_students : students = 38) (h_months : months = 12) :
  ∃ month_students : ℕ, month_students ≥ 4 ∧ ∃ m : fin months, month_birthdays m = month_students :=
by
  sorry

end students_month_l134_134927


namespace ratio_of_hexagon_areas_l134_134416

open Real

/-- 
  Given a regular hexagon ABCDEF with side length 4, 
  and G, H, I, J, K, L are the midpoints of the sides AB, BC, CD, DE, EF, and AF
  respectively, the segments AH, BI, CJ, DK, EL, FG form a smaller regular hexagon 
  inside the original. Prove that the ratio of the area of the smaller hexagon to 
  the area of ABCDEF is 49/36.
-/
theorem ratio_of_hexagon_areas :
  ∀ (ABCDEF : Type) [RegularHexagon ABCDEF] (side_len : ℝ) 
  [h_side : side_len = 4] 
  (G H I J K L : Point)
  [midpoints G H I J K L ABCDEF],
  let smaller_hex_area := areaOfRegularHexagon (midpoints_hexagon G H I J K L ABCDEF),
      original_hex_area := areaOfRegularHexagon ABCDEF in
  (smaller_hex_area / original_hex_area) = 49 / 36 := 
  sorry

end ratio_of_hexagon_areas_l134_134416


namespace charlie_steps_l134_134249

theorem charlie_steps (steps_per_run : ℕ) (runs : ℝ) (expected_steps : ℕ) :
  steps_per_run = 5350 →
  runs = 2.5 →
  expected_steps = 13375 →
  runs * steps_per_run = expected_steps :=
by intros; linarith; sorry

end charlie_steps_l134_134249


namespace laurent_greater_chloe_probability_l134_134615

noncomputable def probability_laurent_greater_chloe : ℝ :=
  let x_dist := Uniform [0, 1000]
  let y_dist := Uniform [0, 2000]
  3 / 4

theorem laurent_greater_chloe_probability (x y : ℝ) (hx : x ∈ Icc 0 1000) (hy : y ∈ Icc 0 2000) : 
  ∫x in 0..1000, ∫y in 0..2000, if y > x then 1 else 0 = 3 / 4 := by
  sorry

end laurent_greater_chloe_probability_l134_134615


namespace boundary_polygon_sides_eq_6_l134_134812

variable (a : ℝ) (h : 0 < a)
def S : set (ℝ × ℝ) := {p | let x := p.1, y := p.2 in (a / 3 ≤ x ∧ x ≤ 3 * a) ∧ (a / 3 ≤ y ∧ y ≤ 3 * a) ∧ (x + y ≤ 3 * a) ∧ (x + a ≥ y) ∧ (y + a ≥ x)}

theorem boundary_polygon_sides_eq_6 : ∀ (a : ℝ) (h : 0 < a), ∃ (polygon : set (ℝ × ℝ)), 
  polygon = (∂S a h) ∧ cardinal.mk ((boundary_of_set_polygon polygon).vertices) = 6 :=
by
  sorry

end boundary_polygon_sides_eq_6_l134_134812


namespace remainder_div_x_minus_4_l134_134529

def f (x : ℕ) : ℕ := x^5 - 8 * x^4 + 16 * x^3 + 25 * x^2 - 50 * x + 24

theorem remainder_div_x_minus_4 : 
  (f 4) = 224 := 
by 
  -- Proof goes here
  sorry

end remainder_div_x_minus_4_l134_134529


namespace book_arrangement_count_l134_134356

theorem book_arrangement_count :
  let total_books := 6
  let identical_books := 3
  (Nat.factorial total_books) / (Nat.factorial identical_books) = 120 := 
by 
  let total_books := 6
  let identical_books := 3
  calc
    (Nat.factorial total_books) / (Nat.factorial identical_books) = 720 / 6 : by simp [Nat.factorial]
    ... = 120 : by norm_num 

end book_arrangement_count_l134_134356


namespace greatest_odd_factors_l134_134823

theorem greatest_odd_factors (n : ℕ) (h1 : n < 1000) (h2 : ∀ k : ℕ, k * k = n → (k < 32)) :
  n = 31 * 31 :=
by
  sorry

end greatest_odd_factors_l134_134823


namespace proof_of_value_of_6y_plus_3_l134_134359

theorem proof_of_value_of_6y_plus_3 (y : ℤ) (h : 3 * y + 2 = 11) : 6 * y + 3 = 21 :=
by
  sorry

end proof_of_value_of_6y_plus_3_l134_134359


namespace sum_of_perimeters_l134_134975

theorem sum_of_perimeters 
  (t1_side_length : ℝ) 
  (h_t1_side_length : t1_side_length = 45) :
  let Pn (n : ℕ) := 3 * (t1_side_length / (2^(n-1))) in
  (∑' n, Pn n) = 270 :=
sorry

end sum_of_perimeters_l134_134975


namespace ellipse_equation_line_max_area_l134_134705

-- Step 1: Conditions of the ellipse (E)
variables (a b c : ℝ)
axiom ellipse_conditions :
  (c = sqrt 5) ∧
  (a^2 = b^2 + c^2) ∧
  (a * b = 6)

-- Question I: Equation of the ellipse
theorem ellipse_equation : ∃ (a b : ℝ), 
  (c = sqrt 5) ∧
  (a^2 = b^2 + c^2) ∧
  (a * b = 6) ∧
  (a^2 = 9) ∧
  (b^2 = 4) ∧
  (∀ x y : ℝ, (x / a)^2 + (y / b)^2 = 1 → (a = 3) ∧ (b = 2)) :=
begin
  sorry
end

-- Step 2: Conditions for the line (l) and triangle ABC
variables (m : ℝ) (C : ℝ × ℝ)
axiom line_triangle_conditions :
  (C = (4/3, 2)) ∧
  (∀ m : ℝ, m^2 < 18) ∧
  (∀ (x1 y1 x2 y2 : ℝ), 
    y1 = (3/2)*x1 + m ∧ 
    y2 = (3/2)*x2 + m → 
    ((x1 + x2 = - 2/3*m) ∧ 
    (x1 * x2 = (2*m^2 - 18) / 9) ∧ 
    (abs (sqrt (13/4 * (4*m^2 / 9 - 8*(m^2 - 9) / 9)) * 2/ sqrt 13)) ≤ 6))

-- Question II: Equation for the line (l)
theorem line_max_area : ∃ (m : ℝ), 
  (m^2 = 9) ∧
  (∀ (x1 y1 x2 y2 : ℝ), 
    y1 = (3/2)*x1 + m ∧ 
    y2 = (3/2)*x2 + m ∧ 
    (C = (4/3, 2)) →
    (area_triangle C (x1, y1) (x2, y2) = 6)) :=
begin
  sorry
end

end ellipse_equation_line_max_area_l134_134705


namespace largest_in_eight_consecutive_integers_l134_134123

theorem largest_in_eight_consecutive_integers (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) = 4304) :
  n + 7 = 544 :=
by
  sorry

end largest_in_eight_consecutive_integers_l134_134123


namespace range_of_a_l134_134725

noncomputable def A : Set ℝ := {x | x^2 - 3 * x < 0}
def B (a : ℝ) : Set ℝ := {1, a}
def intersection_has_4_subsets (a : ℝ) := ∃ s : Set ℝ, s = A ∩ B a ∧ s.subsets.card = 4

theorem range_of_a (a : ℝ) (h: intersection_has_4_subsets a) : 
    (a ∈ Set.Ioo 0 1 ∨ a ∈ Set.Ioo 1 3) :=
sorry

end range_of_a_l134_134725


namespace circle_radius_l134_134468

theorem circle_radius (A : ℝ) (h1 : A = 49 * real.pi) : ∃ r : ℝ, r = 7 :=
by
  -- introduce the area formula
  let area_formula := λ r, real.pi * r^2
  -- given area A is 49*pi
  have h2 : 49 * real.pi = real.pi * 49,
    rw [mul_comm],
  -- solve for radius r
  use (real.sqrt 49),
  -- verify the radius is 7
  rw [real.sqrt_eq_rpow, real.rpow_nat_cast, real.pow_two, real.eq_iff_iff],
  { exact rfl }

end circle_radius_l134_134468


namespace sum_first_1234_terms_l134_134892

def sequence : ℕ → ℕ
| 0 := 1
| (n+1) :=
  let block := n // 2
  in if n % 2 = 0 then block + 1 else 3

def sum_sequence_up_to : ℕ → ℕ
| 0 := sequence 0
| (n+1) := sum_sequence_up_to n + sequence (n+1)

theorem sum_first_1234_terms : sum_sequence_up_to 1233 = 4927 :=
by
  sorry

end sum_first_1234_terms_l134_134892


namespace multiply_106_94_l134_134253

theorem multiply_106_94 : 106 * 94 = 9964 := by
  have h1 : 106 = 100 + 6 := rfl
  have h2 : 94 = 100 - 6 := rfl
  rw [h1, h2]
  rw [mul_add, add_mul, sub_mul, mul_sub]
  norm_num
  sorry

end multiply_106_94_l134_134253


namespace charlie_steps_proof_l134_134247

-- Define the conditions
def Steps_Charlie_3km : ℕ := 5350
def Laps : ℚ := 2.5

-- Define the total steps Charlie can make in 2.5 laps
def Steps_Charlie_total : ℕ := 13375

-- The statement to prove
theorem charlie_steps_proof : Laps * Steps_Charlie_3km = Steps_Charlie_total :=
by
  sorry

end charlie_steps_proof_l134_134247


namespace min_possible_product_l134_134528

theorem min_possible_product : 
  ∃ a b c, 
    {a, b, c} ⊆ {-10, -7, -3, 0, 4, 6, 9} ∧ 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    ∀ x y z, 
      {x, y, z} ⊆ {-10, -7, -3, 0, 4, 6, 9} → 
      x ≠ y → y ≠ z → z ≠ x → 
      a * b * c ≤ x * y * z ∧ 
    a * b * c = -540 :=
sorry

end min_possible_product_l134_134528


namespace increasing_function_l134_134541

def fA (x : ℝ) : ℝ := -x
def fB (x : ℝ) : ℝ := (2 / 3) ^ x
def fC (x : ℝ) : ℝ := x ^ 2
def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function (x y : ℝ) (h : x < y) : fD x < fD y := sorry

end increasing_function_l134_134541


namespace find_m_l134_134655

-- Definitions corresponding to the conditions
def ray_from_pole_intersects_line (O M : Point) (θ : ℝ) : Prop :=
  (∃ ρ, M = Point.polar ρ θ) ∧ (ρ * cos θ = 3)

def point_on_ray_OM (P M : Point) (|OM| |OP| : ℝ) : Prop :=
  |OM| * |OP| = 12

def point_P_on_line (P : Point) (m : ℝ) : Prop :=
  P.y - P.x = m

-- Main theorem statement
theorem find_m (O M P : Point) (θ : ℝ) (|OM| |OP| : ℝ) 
  (h1 : ray_from_pole_intersects_line O M θ) 
  (h2 : point_on_ray_OM P M |OM| |OP|)
  (h3 : ∃! P, point_P_on_line P m) :
  m = -2 + 2 * Real.sqrt 2 ∨ m = -2 - 2 * Real.sqrt 2 ∨ m = 0 :=
by
  sorry

end find_m_l134_134655


namespace tangent_line_equation_l134_134186

def curve : ℝ → ℝ := λ x, (1/3) * x^3 + (4/3)

def point : ℝ × ℝ := (2, 4)

theorem tangent_line_equation : 
  ∃ (m : ℝ), ∃ (b : ℝ), 
  (∀ x : ℝ, curve x = m * x + b → (4 * x - y - 4 = 0)) :=
sorry

end tangent_line_equation_l134_134186


namespace cube_of_composite_as_diff_of_squares_l134_134451

theorem cube_of_composite_as_diff_of_squares (n : ℕ) (h : ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b) :
  ∃ (A₁ B₁ A₂ B₂ A₃ B₃ : ℕ), 
    n^3 = A₁^2 - B₁^2 ∧ 
    n^3 = A₂^2 - B₂^2 ∧ 
    n^3 = A₃^2 - B₃^2 ∧ 
    (A₁, B₁) ≠ (A₂, B₂) ∧ 
    (A₁, B₁) ≠ (A₃, B₃) ∧ 
    (A₂, B₂) ≠ (A₃, B₃) := sorry

end cube_of_composite_as_diff_of_squares_l134_134451


namespace min_tetrahedron_height_l134_134675

-- Define the radius of the spheres
def radius : ℝ := 1

-- Define the side length of the tetrahedron
def side_length (r : ℝ) : ℝ := 2 * r

-- Define the height of a regular tetrahedron given the side length a
def height_tetrahedron (a : ℝ) : ℝ :=
  real.sqrt ((sqrt 3 / 2 * a)^2 - (sqrt 3 / 6 * a)^2)

-- Define the minimum height of the tetrahedron that encloses four spheres
def min_height (r : ℝ) : ℝ :=
  2 + height_tetrahedron (side_length r)

theorem min_tetrahedron_height : min_height radius = 2 + (2 * sqrt 6 / 3) :=
by sorry

end min_tetrahedron_height_l134_134675


namespace categorize_numbers_l134_134567

def numbers : List Rat := [1, -3/5, 3.2, 0, 1/3, -6.5, 108, -4, -6 - 4/7]

def is_integer (x : Rat) : Prop :=
  ∃ (n : Int), x = n

def is_negative_fraction (x : Rat) : Prop :=
  x < 0 ∧ ∀ (n : Int), x ≠ n

def is_nonneg_integer (x : Rat) : Prop :=
  0 ≤ x ∧ ∃ (n : Int), x = n

theorem categorize_numbers :
  ({x ∈ numbers | is_integer x} = [1, 0, 108, -4]) ∧
  ({x ∈ numbers | is_negative_fraction x} = [-3/5, -6.5, -6 - 4/7]) ∧
  ({x ∈ numbers | is_nonneg_integer x} = [1, 0, 108]) :=
by
  sorry

end categorize_numbers_l134_134567


namespace difference_in_cost_l134_134092

noncomputable def price_jersey : ℝ := 115
def discount_jersey : ℝ := 0.1
def price_tshirt : ℝ := 25
def discount_tshirt : ℝ := 0.15 
def sales_tax : ℝ := 0.08
def shipping_fee_jersey : ℝ := 5
def shipping_fee_tshirt : ℝ := 3

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price - (price * discount)

def price_with_tax (discounted_price : ℝ) (tax_rate : ℝ) : ℝ :=
  discounted_price + (discounted_price * tax_rate)

def final_cost (price_tax : ℝ) (shipping_fee : ℝ) : ℝ :=
  price_tax + shipping_fee

def final_cost_jersey : ℝ :=
  final_cost (price_with_tax (discounted_price price_jersey discount_jersey) sales_tax) shipping_fee_jersey

def final_cost_tshirt : ℝ :=
  final_cost (price_with_tax (discounted_price price_tshirt discount_tshirt) sales_tax) shipping_fee_tshirt

theorem difference_in_cost : final_cost_jersey - final_cost_tshirt = 90.83 :=
by
  sorry

end difference_in_cost_l134_134092


namespace isosceles_right_triangle_area_l134_134939

theorem isosceles_right_triangle_area (a : ℝ) (h : ℝ) (p : ℝ) 
  (h_triangle : h = a * Real.sqrt 2) 
  (hypotenuse_is_16 : h = 16) :
  (1 / 2) * a * a = 64 := 
by
  -- Skip the proof as per guidelines
  sorry

end isosceles_right_triangle_area_l134_134939


namespace range_of_a_l134_134430

open Set

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

noncomputable def M (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+2)*x + 2a ≤ 0}

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ M a → x ∈ A) → (1 ≤ a ∧ a ≤ 4) := by
  sorry

end range_of_a_l134_134430


namespace charlie_steps_proof_l134_134248

-- Define the conditions
def Steps_Charlie_3km : ℕ := 5350
def Laps : ℚ := 2.5

-- Define the total steps Charlie can make in 2.5 laps
def Steps_Charlie_total : ℕ := 13375

-- The statement to prove
theorem charlie_steps_proof : Laps * Steps_Charlie_3km = Steps_Charlie_total :=
by
  sorry

end charlie_steps_proof_l134_134248


namespace apples_left_is_ten_l134_134509

noncomputable def appleCost : ℝ := 0.80
noncomputable def orangeCost : ℝ := 0.50
def initialApples : ℕ := 50
def initialOranges : ℕ := 40
def totalEarnings : ℝ := 49
def orangesLeft : ℕ := 6

theorem apples_left_is_ten (A : ℕ) :
  (50 - A) * appleCost + (40 - orangesLeft) * orangeCost = 49 → A = 10 :=
by
  sorry

end apples_left_is_ten_l134_134509


namespace smallest_sum_inequality_l134_134104

-- Definition of the problem
def problem_condition (p q r s : ℕ) := p * q * r * s = nat.factorial 12

-- The statement to be proved
theorem smallest_sum_inequality (p q r s : ℕ) (h : problem_condition p q r s) :
  p + q + r + s ≥ 683 :=
sorry

end smallest_sum_inequality_l134_134104


namespace triangle_area_l134_134582

open Real

noncomputable def point (x y : ℝ) := (x, y)

theorem triangle_area (P Q R : ℝ × ℝ)
  (hP : P = point 2 5)
  (hQ : Q = point 7 0)
  (hR : R = point (1 / 3) 0)
  (line1_slope : ∀ (x y : ℝ), y - 5 = -1 * (x - 2))
  (line2_slope : ∀ (x y : ℝ), y - 5 = 3 * (x - 2)) :
  let base := Q.1 - R.1,
      height := P.2 in
  1 / 2 * base * height = 50 / 3 :=
by
  sorry

end triangle_area_l134_134582


namespace sum_of_three_numbers_l134_134485

theorem sum_of_three_numbers : ∃ (a b c : ℝ), a ≤ b ∧ b ≤ c ∧ b = 8 ∧ 
  (a + b + c) / 3 = a + 8 ∧ (a + b + c) / 3 = c - 20 ∧ a + b + c = 60 :=
sorry

end sum_of_three_numbers_l134_134485


namespace tangent_line_through_origin_eqn_circle_second_quadrant_l134_134745

theorem tangent_line_through_origin_eqn_circle_second_quadrant :
  ∀ (k : ℝ), 
  (∀ x y : ℝ, (y = k * x) ↔ (x^2 + (y - 4)^2 = 4) ∧ y = k * x ∧ y ≤ 0 ∧ x ≤ 0 ∧ y = -sqrt(3) * x) :=
begin
  sorry
end

end tangent_line_through_origin_eqn_circle_second_quadrant_l134_134745


namespace new_job_larger_than_original_l134_134196

theorem new_job_larger_than_original (original_workers original_days new_workers new_days : ℕ) 
  (h_original_workers : original_workers = 250)
  (h_original_days : original_days = 16)
  (h_new_workers : new_workers = 600)
  (h_new_days : new_days = 20) :
  (new_workers * new_days) / (original_workers * original_days) = 3 := by
  sorry

end new_job_larger_than_original_l134_134196


namespace correct_statement_about_K_l134_134096

-- Defining the possible statements about the chemical equilibrium constant K
def K (n : ℕ) : String :=
  match n with
  | 1 => "The larger the K, the smaller the conversion rate of the reactants."
  | 2 => "K is related to the concentration of the reactants."
  | 3 => "K is related to the concentration of the products."
  | 4 => "K is related to temperature."
  | _ => "Invalid statement"

-- Given that the correct answer is that K is related to temperature
theorem correct_statement_about_K : K 4 = "K is related to temperature." :=
by
  rfl

end correct_statement_about_K_l134_134096


namespace area_enclosed_by_region_l134_134145

theorem area_enclosed_by_region :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6*x + 8*y = -9) →
  let radius := 4 in
  let area := real.pi * radius^2 in
  area = 16 * real.pi :=
sorry

end area_enclosed_by_region_l134_134145


namespace parabola_line_slope_l134_134206

theorem parabola_line_slope :
  let E := { p : ℝ × ℝ | p.2^2 = 4 * p.1 }
  let F := (1, 0)
  (A B C : ℝ × ℝ) (l : ℝ × ℝ → Prop),
  (l F) ∧ (E A) ∧ (E B) ∧ (l A) ∧ (l B) ∧ (l C) ∧
  (B = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) ∧
  (C.1 = -1) ∧ (A.1 > 0) ∧ (A.2 > 0) →
  (∃ k : ℝ, ∀ x : ℝ, l (x, k * (x - 1)) → k = 2 * real.sqrt 2) :=
sorry

end parabola_line_slope_l134_134206


namespace cartons_in_case_l134_134993

theorem cartons_in_case (b : ℕ) (hb : b ≥ 1) (h : 2 * c * b * 500 = 1000) : c = 1 :=
by
  -- sorry is used to indicate where the proof would go
  sorry

end cartons_in_case_l134_134993


namespace polynomial_simplification_l134_134457

theorem polynomial_simplification :
  (λ x : ℝ, x^3 + 4*x^2 - 7*x + 11 + (-4*x^4 - x^3 + x^2 + 7*x + 3)) = (λ x : ℝ, -4*x^4 + 5*x^2 + 14) :=
by 
  sorry 

end polynomial_simplification_l134_134457


namespace main_problem_l134_134714

   noncomputable def f: ℝ → ℝ :=
   λ x, if x ∈ Set.Ioo (-π/2) (π/2) then x + Real.tan x else f (π - x)

   def a := f 1
   def b := f 2
   def c := f 3

   theorem main_problem (f_def: ∀ x, f(x) = f(π - x)):
     (c < a ∧ a < b) :=
   by
     -- Using the conditions given in the problem
     have fc := f_def 1
     have fb := f_def 2
     -- additional steps skipped
     sorry
   
end main_problem_l134_134714


namespace translate_sine_function_l134_134878

theorem translate_sine_function (x : ℝ) :
  (translate_right (λ x, sin (1/2 * x + π / 3)) (π / 3)) x = sin (1/2 * x + π / 6) :=
by
  sorry

end translate_sine_function_l134_134878


namespace maximum_radius_in_quadrilateral_l134_134791

noncomputable def maximum_inscribed_circle_radius (a b c d : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 5) (hd : d = 4) : ℝ :=
  let s := (a + b + c + d) / 2 in
  let K := Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) in
  K / s

theorem maximum_radius_in_quadrilateral :
  maximum_inscribed_circle_radius 2 3 5 4 (by rfl) (by rfl) (by rfl) (by rfl) = 2 * Real.sqrt 30 / 7 :=
by {
  -- Insert detailed proof steps here later
  sorry
}

end maximum_radius_in_quadrilateral_l134_134791


namespace general_term_an_sum_first_n_bn_l134_134688

noncomputable def an (n : ℕ) : ℕ := 3 * n - 2

theorem general_term_an (a : ℕ → ℕ) (h1 : a 4 = 10) (h2 : (a 1, a 2, a 6) ∈ ℕ) :
  ∀ n, a n = 3 * n - 2 :=
sorry

noncomputable def bn (n : ℕ) : ℕ := 2 ^ (an n) + 2 * n

theorem sum_first_n_bn (n : ℕ) (S : ℕ → ℕ) :
  S n = ∑ i in range n, bn i ∧
  S n = (2 / 7) * (8 ^ n - 1) + n * (n + 1) :=
sorry

end general_term_an_sum_first_n_bn_l134_134688


namespace dice_probability_even_sum_one_six_l134_134941

theorem dice_probability_even_sum_one_six :
  let outcomes := [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                   (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
                   (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
                   (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                   (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
                   (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)] in
  let favorable := [(6, 2), (6, 4), (6, 6), (2, 6), (4, 6)] in
  (favorable.length : ℚ) / (outcomes.length : ℚ) = 5 / 36 :=
by
  sorry

end dice_probability_even_sum_one_six_l134_134941


namespace log_100_eq_2_l134_134188

theorem log_100_eq_2 : log 10 100 = 2 := 
by
  sorry

end log_100_eq_2_l134_134188


namespace angle_x_degrees_l134_134041

noncomputable def center_of_circle (O : Point) := true
noncomputable def radii (O B C A : Point) := (dist O B = dist O C) ∧ (dist O A = dist O C)
noncomputable def isosceles_triangle (O B C : Point) := (∠ B C O = 32) ∧ (∠ O B C = 32)
noncomputable def isosceles_base_angle (O A C : Point) := ∠ O A C = ∠ A O C = 23
noncomputable def angle_sum_property (O A C : Point) := ∠ A O C = 90

theorem angle_x_degrees
  (O B C A : Point) (h_radii : radii O B C A)
  (h_isosceles_triangle : isosceles_triangle O B C)
  (h_isosceles_base : isosceles_base_angle O A C)
  (h_sum_property : angle_sum_property O A C) :
  ∠ B C O - ∠ O A C = 9 := by sorry

end angle_x_degrees_l134_134041


namespace grandma_contribution_l134_134973

def trip_cost : ℝ := 485
def candy_bar_profit : ℝ := 1.25
def candy_bars_sold : ℕ := 188
def amount_earned_from_selling_candy_bars : ℝ := candy_bars_sold * candy_bar_profit
def amount_grandma_gave : ℝ := trip_cost - amount_earned_from_selling_candy_bars

theorem grandma_contribution :
  amount_grandma_gave = 250 := by
  sorry

end grandma_contribution_l134_134973


namespace probability_of_B_l134_134868

variable (P : (Set → ℝ))
variable (A B : Set)

-- Given conditions
axiom P_A_eq : P A = 0.01
axiom P_B_given_A_eq : P (B ∩ A) / P A = 0.99
axiom P_B_given_not_A_eq : P (B ∩ -A) / P (-A) = 0.1
axiom P_not_A_eq : P (-A) = 0.99

-- Prove P(B) = 0.1089
theorem probability_of_B : P B = 0.1089 :=
by 
  sorry

end probability_of_B_l134_134868


namespace floor_equation_solution_l134_134357

theorem floor_equation_solution (a b : ℝ) :
  (∀ x y : ℝ, ⌊a * x + b * y⌋ + ⌊b * x + a * y⌋ = (a + b) * ⌊x + y⌋) → (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) := by
  sorry

end floor_equation_solution_l134_134357


namespace sarah_total_volume_in_two_weeks_l134_134068

def shampoo_daily : ℝ := 1

def conditioner_daily : ℝ := 1 / 2 * shampoo_daily

def days : ℕ := 14

def total_volume : ℝ := (shampoo_daily * days) + (conditioner_daily * days)

theorem sarah_total_volume_in_two_weeks : total_volume = 21 := by
  sorry

end sarah_total_volume_in_two_weeks_l134_134068


namespace christmas_bulbs_on_probability_l134_134906

noncomputable def toggle_algorithm (n : ℕ) : ℕ → Bool
| k => (List.range n).foldl (λ b i => if (k + 1) % (i + 1) == 0 then !b else b) true

def is_perfect_square (n : ℕ) : Bool :=
  let root := (n + 1).natAbs.sqrt
  root * root = n + 1

def probability_bulb_on_after_toggling (n : ℕ) : ℚ :=
  let on_count := finset.filter (λ k => toggle_algorithm n k) (finset.range n) 
  on_count.card / n

theorem christmas_bulbs_on_probability : 
  probability_bulb_on_after_toggling 100 = 0.1 :=
sorry

end christmas_bulbs_on_probability_l134_134906


namespace pure_imaginary_solution_l134_134282

noncomputable def pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_solution (x : ℝ) :
  pure_imaginary ((x + 1 + complex.i) * (x + 2 + complex.i) * (x + 3 + complex.i)) ↔ x = -1 :=
by
  sorry

end pure_imaginary_solution_l134_134282


namespace locus_of_C_l134_134767

def point : Type := (ℝ × ℝ)

variables (A B C : point)
variables (x y : ℝ)

def distance (p1 p2 : point) : ℝ :=
real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def isosceles (A B C : point) : Prop :=
distance A B = distance A C

def locus_eq (x y : ℝ) : Prop :=
x^2 + y^2 - 6 * x + 4 * y - 5 = 0

theorem locus_of_C :
  A = (3, -2) ∧ B = (0, 1) ∧ isosceles A B C → locus_eq C.1 C.2 := by
  sorry

end locus_of_C_l134_134767


namespace relationship_between_first_and_third_numbers_l134_134128

variable (A B C : ℕ)

theorem relationship_between_first_and_third_numbers
  (h1 : A + B + C = 660)
  (h2 : A = 2 * B)
  (h3 : B = 180) :
  C = A - 240 :=
by
  sorry

end relationship_between_first_and_third_numbers_l134_134128


namespace PS_pass_through_fixed_point_l134_134400

-- Assuming the points A, B, and C and the midway point M, and the constructions of the squares
variable (A B C : Point)
variable (M : Point) -- This is the midpoint of BC

def is_square_outward (a b c d : Point) : Prop := sorry -- Define outward square property

theorem PS_pass_through_fixed_point
  (fixed_B : fixed_point B)
  (fixed_C : fixed_point C)
  (midpoint_M : midpoint M B C)
  (square_ACPQ : ∀ A, is_square_outward A C P Q)
  (square_BARS : ∀ A, is_square_outward B A R S)
  (half_plane : ∀ A, A ∈ one_half_plane_determined_by BC) :
  ∀ A, ∃ P S, line_through P S ∧ passes_through M (line_through P S) := 
sorry

end PS_pass_through_fixed_point_l134_134400


namespace find_increasing_function_l134_134535

-- Define each function
def fA (x : ℝ) := -x
def fB (x : ℝ) := (2 / 3) ^ x
def fC (x : ℝ) := x ^ 2
def fD (x : ℝ) := x ^ (1 / 3)

-- Define the statement that fD is the only increasing function among the options
theorem find_increasing_function (f : ℝ → ℝ) (hf : f = fD) :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fA x < fA y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fB x < fB y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fC x < fC y) :=
by {
  sorry
}

end find_increasing_function_l134_134535


namespace isosceles_right_triangle_exists_l134_134727

-- Definitions for non-congruent isosceles right triangles
structure Triangle :=
  (A B C : Point)
  (is_isosceles_right : is_isosceles_right_triangle A B C)

def non_congruent (Δ1 Δ2 : Triangle) : Prop :=
  (not_congruent Δ1 Δ2)

-- Main theorem
theorem isosceles_right_triangle_exists (ΔABC ΔADE : Triangle)
  (h_non_congruent : non_congruent ΔABC ΔADE)
  (h_ABC_fixed : ΔABC.A = FixedPoint)
  (h_ADE_rotates : rotates_around ΔADE ΔABC.A) :
∃ M : Point, on_segment M ΔADE.E ΔABC.C ∧ is_isosceles_right_triangle ΔABC.B M ΔADE.D :=
by sorry

end isosceles_right_triangle_exists_l134_134727


namespace quarter_circle_sum_greater_than_original_l134_134097

noncomputable def radius_of_quarter_circle (r : ℝ) (n : ℕ) : ℝ := (2 * π * r) / n

noncomputable def circumference_of_quarter_circle (r : ℝ) (n : ℕ) : ℝ :=
  (π^2 * r) / n

theorem quarter_circle_sum_greater_than_original
  (r : ℝ) (hn : ℕ) (hpos : 0 < hn) :
  let n := hn in
  n * circumference_of_quarter_circle r n > 2 * π * r :=
by
  sorry

end quarter_circle_sum_greater_than_original_l134_134097


namespace smallest_three_digit_not_divisible_l134_134531

theorem smallest_three_digit_not_divisible :
  ∃ n : ℕ, 100 ≤ n ∧ 
  (let S_n := (n * (n + 1) * (2 * n + 1)) / 6 in
   let P_n := nat.factorial n in 
   (P_n % S_n ≠ 0) ∧ (∀ k : ℕ, 100 ≤ k ∧ k < n → 
   (let S_k := (k * (k + 1) * (2 * k + 1)) / 6 in
    let P_k := nat.factorial k in
    P_k % S_k = 0))) :=
begin
  let n := 101,
  use n,
  split,
  { exact nat.le_refl 101 },
  { unfold S_n P_n,
    sorry, 
  }
end

end smallest_three_digit_not_divisible_l134_134531


namespace tangent_normal_lines_l134_134985

noncomputable def x (t : ℝ) : ℝ := (1 / 2) * t^2 - (1 / 4) * t^4
noncomputable def y (t : ℝ) : ℝ := (1 / 2) * t^2 + (1 / 3) * t^3
def t0 : ℝ := 0

theorem tangent_normal_lines :
  (∃ m : ℝ, ∀ t : ℝ, t = t0 → y t = m * x t) ∧
  (∃ n : ℝ, ∀ t : ℝ, t = t0 → y t = n * x t ∧ n = -1 / m) :=
sorry

end tangent_normal_lines_l134_134985


namespace exists_star_like_curve_l134_134870

-- Definitions of the necessary geometric entities and axioms.
def is_closed_curve (γ : ℝ → ℝ × ℝ) : Prop :=
∀ t₁ t₂ : ℝ, γ t₁ = γ t₂ → t₁ = t₂

def non_self_intersecting (γ : ℝ → ℝ × ℝ) : Prop :=
∀ t₁ t₂ : ℝ, t₁ ≠ t₂ → γ t₁ ≠ γ t₂

def can_trace_triangle (γ : ℝ → ℝ × ℝ) (Δ : ℝ × ℝ → Prop) : Prop :=
∃ (v₁ v₂ v₃ : ℝ), Δ v₁ ∧ Δ v₂ ∧ Δ v₃ ∧
  ∀ t : ℝ, ∃ (u₁ u₂ u₃ : ℝ),
    γ u₁ = (v₁.1 + t * (v₂.1 - v₁.1), v₁.2 + t * (v₂.2 - v₁.2)) ∧
    γ u₂ = (v₂.1 + t * (v₃.1 - v₂.1), v₂.2 + t * (v₃.2 - v₂.2)) ∧
    γ u₃ = (v₃.1 + t * (v₁.1 - v₃.1), v₃.2 + t * (v₁.2 - v₃.2))

noncomputable theory
def star_like_curve (curve : ℝ → ℝ × ℝ) : Prop :=
is_closed_curve curve ∧ non_self_intersecting curve ∧
can_trace_triangle curve (λ v, ∃ t : ℝ, curve t = v)

theorem exists_star_like_curve :
  ∃ (curve : ℝ → ℝ × ℝ), star_like_curve curve ∧
  ∃ (v₁ v₂ v₃ : ℝ), 
    |((v₁ - v₂), (v₂ - v₃), (v₃ - v₁))| = (equilateral_triangle : ℝ × ℝ × ℝ) ∧ 
    can_trace_triangle curve (λ v, ∃ t : ℝ, curve t = v) :=
by sorry

end exists_star_like_curve_l134_134870


namespace angle_x_is_58_l134_134017

-- Definitions and conditions based on the problem statement
variables {O A C D : Type} (x : ℝ)
variable [circular_center : O] -- O is the center of the circle
variable [radius_equal : AO = BO ∧ BO = CO] -- AO, BO, and CO are radii and hence equal
variable (angle_ACD_right : ∠ A C D = 90) -- ∠ACD is a right angle (subtended by diameter)
variable (angle_CDA : ∠ C D A = 42) -- ∠CDA is 42 degrees

-- Statement to be proven
theorem angle_x_is_58 (x : ℝ) : x = 58 := 
by sorry

end angle_x_is_58_l134_134017


namespace probability_interval_l134_134116

noncomputable def X_pdf (k : ℕ) : ℝ :=
  if 1 ≤ k ∧ k ≤ 4 then (5 / 4) / (k * (k + 1)) else 0

theorem probability_interval:
  (∑ k in {1, 2}, X_pdf k) = 5 / 6 :=
by
  sorry

end probability_interval_l134_134116


namespace calculate_expression_l134_134613

noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def tan_45 := Real.tan (Real.pi / 4)
noncomputable def cot_30 := Real.cot (Real.pi / 6)
noncomputable def cos_45 := Real.cos (Real.pi / 4)

theorem calculate_expression :
  2 * |1 - sin_60| + tan_45 / (cot_30 - 2 * cos_45) = 2 + Real.sqrt 2 :=
by
  sorry

end calculate_expression_l134_134613


namespace find_time_period_approx_l134_134284

def compound_interest_time_period (P A r : ℝ) (n : ℕ) : ℝ :=
  -- Solving for t given P, A, r, and n
  (Real.log (A / P)) / (Real.log (1 + r / n))

theorem find_time_period_approx :
  ∃ t : ℕ, 
    let P := 12000 in
    let CI := 4663.5 in
    let A := P + CI in
    let r := 0.15 in
    let n := 1 in
    (compound_interest_time_period P A r n).round = t ∧
    t = 2 :=
by
  sorry

end find_time_period_approx_l134_134284


namespace point_coordinates_in_second_quadrant_l134_134003

theorem point_coordinates_in_second_quadrant (P : ℝ × ℝ)
  (hx : P.1 ≤ 0)
  (hy : P.2 ≥ 0)
  (dist_x_axis : abs P.2 = 3)
  (dist_y_axis : abs P.1 = 10) :
  P = (-10, 3) :=
by
  sorry

end point_coordinates_in_second_quadrant_l134_134003


namespace orlan_rope_problem_l134_134450

theorem orlan_rope_problem:
  (initial_rope : ℕ) (given_to_allan : ℚ) (given_to_jack : ℚ) 
  (h1 : initial_rope = 20) 
  (h2 : given_to_allan = 1/4) 
  (h3 : given_to_jack = 2/3):
  let remaining_after_allan := initial_rope - initial_rope * given_to_allan in 
  let remaining_after_jack := remaining_after_allan - remaining_after_allan * given_to_jack in
  remaining_after_jack = 5 := 
by 
  sorry

end orlan_rope_problem_l134_134450


namespace length_of_ln_l134_134087

theorem length_of_ln (sin_N_eq : Real.sin angle_N = 3 / 5) (LM_eq : length_LM = 15) :
  length_LN = 25 :=
sorry

end length_of_ln_l134_134087


namespace find_certain_number_l134_134366

theorem find_certain_number (x certain_number : ℕ) (h1 : certain_number + x = 13200) (h2 : x = 3327) : certain_number = 9873 :=
by
  sorry

end find_certain_number_l134_134366


namespace multiplier_is_three_l134_134847

theorem multiplier_is_three (n m : ℝ) (h₁ : n = 3) (h₂ : 7 * n = m * n + 12) : m = 3 := 
by
  -- Skipping the proof using sorry
  sorry 

end multiplier_is_three_l134_134847


namespace soccer_stars_teams_l134_134393

theorem soccer_stars_teams (M : ℕ) (hM : M = 28) :
  ∃ n : ℕ, (n * (n - 1)) / 2 = M ∧ n = 8 :=
by
  have h : M = (8 * (8 - 1)) / 2 := by norm_num
  use 8
  constructor
  . exact h
  . rfl

end soccer_stars_teams_l134_134393


namespace solve_for_y_l134_134853

theorem solve_for_y (y : ℕ) (h : 5 * (2 ^ y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l134_134853


namespace increasing_function_l134_134540

def fA (x : ℝ) : ℝ := -x
def fB (x : ℝ) : ℝ := (2 / 3) ^ x
def fC (x : ℝ) : ℝ := x ^ 2
def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function (x y : ℝ) (h : x < y) : fD x < fD y := sorry

end increasing_function_l134_134540


namespace find_side_length_AC_l134_134998

theorem find_side_length_AC
  (A B C K T : Type)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ K] [InnerProductSpace ℝ T]
  (circ : circle A C)
  (intersects_AB : circle_intersects circe AB K)
  (intersects_BC : circle_intersects circe BC T)
  (AK_ratio : ∀ x, line_split_ratio AK K = 3/2)
  (BT_ratio : ∀ y, line_split_ratio BT T = 1/2)
  (length_KT : line_length KT = sqrt 6):
  line_length AC = 3 * sqrt 5 :=
sorry

end find_side_length_AC_l134_134998


namespace tangent_parallel_monotonic_intervals_l134_134336

noncomputable def f (a : ℝ) (x : ℝ) := log (x + 1) - a * x + (1 - a) / (x + 1)

theorem tangent_parallel (a : ℝ) (ha : a ≥ 2) : 
  ((deriv (f a) 1) = -2) ↔ a = 3 :=
sorry

theorem monotonic_intervals (a : ℝ) (ha : a ≥ 2) :
  (if a = 1/2 then ∀ x, x > -1 → deriv (f a) x ≤ 0 else
    if 1/2 < a ∧ a < 1 then ∀ x, x > -1 → 
      (x < 1/a - 2 ∨ x > 0 → deriv (f a) x < 0) ∧ (1/a - 2 < x ∧ x < 0 → deriv (f a) x > 0)
    else ∀ x, x > -1 →
      (x < 0 → deriv (f a) x > 0) ∧ (x > 0 → deriv (f a) x < 0)) :=
sorry

end tangent_parallel_monotonic_intervals_l134_134336


namespace solve_for_x_l134_134361

theorem solve_for_x (x : ℝ) (h : (x / 3) / 3 = 9 / (x / 3)) : x = 3 ^ (5 / 2) ∨ x = -3 ^ (5 / 2) :=
by
  sorry

end solve_for_x_l134_134361


namespace find_a_eigenvalues_and_eigenvectors_l134_134722

noncomputable def matrix_A (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![1, -1], ![a, 1]]

theorem find_a_eigenvalues_and_eigenvectors (a : ℝ) :
  (matrix_A a ⬝ ![1, 1] = ![0, -3] → a = -4) ∧
  let A := matrix_A (-4) in
  (A.det_char_poly.eval (-1) = 0 ∧ A.det_char_poly.eval 3 = 0) ∧
  (A.vec_eigenvector (-1) = ![1, 2]) ∧ 
  (A.vec_eigenvector 3 = ![1, -2]) :=
by sorry

end find_a_eigenvalues_and_eigenvectors_l134_134722


namespace solution_correct_l134_134484

def mascot_options := ["A Xiang", "A He", "A Ru", "A Yi", "Le Yangyang"]

def volunteer_options := ["A", "B", "C", "D", "E"]

noncomputable def count_valid_assignments (mascots : List String) (volunteers : List String) : Nat :=
  let all_assignments := mascots.permutations
  let valid_assignments := all_assignments.filter (λ p =>
    (p.get! 0 = "A Xiang" ∨ p.get! 1 = "A Xiang") ∧ p.get! 2 ≠ "Le Yangyang")
  valid_assignments.length

theorem solution_correct :
  count_valid_assignments mascot_options volunteer_options = 36 :=
by
  sorry

end solution_correct_l134_134484


namespace triangle_shortest_side_l134_134389

noncomputable def area (s a b c : ℝ) : ℝ :=
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_shortest_side (b c : ℝ) (hb : b ≠ real.sqrt (b*b)) {hn1b : ∃ n1 : ℤ, b = ↑n1} 
(hc : c ≠ real.sqrt (c*c)) {hn1c : ∃ n1 : ℤ, c = ↑n1}
(h1 : 17 + b + c = 50)
(h2 : ∃ A : ℝ, A = area (50/2) 17 b c ∧ A ∈ ℤ) :
b = 13 ∨ c = 13 := sorry

end triangle_shortest_side_l134_134389


namespace range_of_a_variance_Y_when_a_half_options_a_and_d_are_correct_l134_134712

open ProbabilityTheory

-- Definitions for the random variables X and Y
def X_pmf (a : ℝ) : ProbabilityMassFunction ℝ :=
  ⟨[
      (1/3, -1),
      ((2 - a) / 3, 0),
      (a / 3, 1)
    ], by norm_num [add_assoc]⟩

def Y_pmf (a : ℝ) : ProbabilityMassFunction ℝ :=
  ⟨[
     (1/2, 0),
     ((1 - a) / 2, 1),
     (a / 2, 2)
   ], by norm_num [add_assoc]⟩

-- Definition to check the range of a
theorem range_of_a (a : ℝ) : 0 ≤ a ∧ a ≤ 1 :=
begin
  split,
  { norm_num },
  { norm_num },
  sorry
end

-- Definition to check the variance of Y when a = 1/2
theorem variance_Y_when_a_half : (D(Y) = 11 / 16) ∧ (a = 1/2) :=
begin
  sorry
end

-- Stating the main question as a theorem  
theorem options_a_and_d_are_correct (a : ℝ) : 
  (0 ≤ a ∧ a ≤ 1) ∧ 
  (a = 1/2 → D(Y) = 11 / 16) :=
begin
  refine ⟨range_of_a a, _⟩,
  intro ha,
  rw ha,
  exact variance_Y_when_a_half
end

end range_of_a_variance_Y_when_a_half_options_a_and_d_are_correct_l134_134712


namespace perimeter_of_fenced_square_field_l134_134933

-- Definitions for conditions
def num_posts : ℕ := 36
def spacing_between_posts : ℝ := 6 -- in feet
def post_width : ℝ := 1 / 2 -- 6 inches in feet

-- The statement to be proven
theorem perimeter_of_fenced_square_field :
  (4 * ((9 * spacing_between_posts) + (10 * post_width))) = 236 :=
by
  sorry

end perimeter_of_fenced_square_field_l134_134933


namespace correct_option_is_B_l134_134971

-- Define the operations as hypotheses
def option_A (a : ℤ) : Prop := (a^2 + a^3 = a^5)
def option_B (a : ℤ) : Prop := ((a^2)^3 = a^6)
def option_C (a : ℤ) : Prop := (a^2 * a^3 = a^6)
def option_D (a : ℤ) : Prop := (6 * a^6 - 2 * a^3 = 3 * a^3)

-- Prove that option B is correct
theorem correct_option_is_B (a : ℤ) : option_B a :=
by
  unfold option_B
  sorry

end correct_option_is_B_l134_134971


namespace sixth_term_geometric_mean_l134_134061

variable (a d : ℝ)

-- Define the arithmetic progression terms
def a_n (n : ℕ) := a + (n - 1) * d

-- Provided condition: second term is the geometric mean of the 1st and 4th terms
def condition (a d : ℝ) := a_n a d 2 = Real.sqrt (a_n a d 1 * a_n a d 4)

-- The goal to be proved: sixth term is the geometric mean of the 4th and 9th terms
theorem sixth_term_geometric_mean (a d : ℝ) (h : condition a d) : 
  a_n a d 6 = Real.sqrt (a_n a d 4 * a_n a d 9) :=
sorry

end sixth_term_geometric_mean_l134_134061


namespace propositionA_to_propositionB_as_l134_134058

variable {p q : Prop}

-- Conditions from the problem
def propositionA : Prop := p → q
def propositionB : Prop := (p ↔ q)

-- Math proof problem statement
theorem propositionA_to_propositionB_as :
  (propositionA → propositionB) →
  (¬(propositionA) → ¬(propositionB)) →
  True := by
  sorry

end propositionA_to_propositionB_as_l134_134058


namespace sequence_after_10_operations_l134_134234

theorem sequence_after_10_operations : 
  let initial := 8^7
  let seq := λ n, if n % 2 = 0 then n / 4 else n * 3
  let final := (seq^[10] initial)
  final = 2^11 * 3^5 :=
by
  let initial := 2^21
  have cycle : ∀ n, seq^[2 * n] initial = 2^(21 - 2 * n) * 3^n := sorry
  exact (cycle 5)

end sequence_after_10_operations_l134_134234


namespace correct_operation_l134_134969

theorem correct_operation (a : ℝ) : 
  ((a^2)^3 = a^6) ∧ ¬(a^2 + a^3 = a^5) ∧ ¬(a^2 * a^3 = a^6) ∧ ¬(6 * a^6 - 2 * a^3 = 3 * a^3) :=
by
  -- Provide the four required conditions separately
  -- Option B is correct:
  show (a^2)^3 = a^6, from sorry,
  
  -- Option A is incorrect:
  show ¬(a^2 + a^3 = a^5), from sorry,
  
  -- Option C is incorrect:
  show ¬(a^2 * a^3 = a^6), from sorry,
  
  -- Option D is incorrect:
  show ¬(6 * a^6 - 2 * a^3 = 3 * a^3), from sorry


end correct_operation_l134_134969


namespace find_e_l134_134426

theorem find_e (a b c d e : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : a + b = 32)
  (h6 : a + c = 36)
  (h7 : b + c = 37)
  (h8 : c + e = 48)
  (h9 : d + e = 51) : e = 55 / 2 :=
  sorry

end find_e_l134_134426


namespace total_wheels_in_garage_l134_134925

theorem total_wheels_in_garage : 
  let bicycles := 3
  let tricycles := 4
  let unicycles := 7
  let wheels_per_bicycle := 2
  let wheels_per_tricycle := 3
  let wheels_per_unicycle := 1
  in (bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle) = 25 :=
by
  let bicycles := 3
  let tricycles := 4
  let unicycles := 7
  let wheels_per_bicycle := 2
  let wheels_per_tricycle := 3
  let wheels_per_unicycle := 1
  have h_bicycles := bicycles * wheels_per_bicycle
  have h_tricycles := tricycles * wheels_per_tricycle
  have h_unicycles := unicycles * wheels_per_unicycle
  show (h_bicycles + h_tricycles + h_unicycles) = 25
  sorry

end total_wheels_in_garage_l134_134925


namespace _l134_134169

def triangle (A B C : Type) : Prop :=
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def angles_not_equal_sides_not_equal (A B C : Type) (angleB angleC : ℝ) (sideAC sideAB : ℝ) : Prop :=
  triangle A B C →
  (angleB ≠ angleC → sideAC ≠ sideAB)
  
lemma xiaoming_theorem {A B C : Type} 
  (hTriangle : triangle A B C)
  (angleB angleC : ℝ)
  (sideAC sideAB : ℝ) :
  angleB ≠ angleC → sideAC ≠ sideAB := 
sorry

end _l134_134169


namespace general_term_sum_of_first_n_terms_l134_134776

noncomputable def a (n : ℕ) : ℕ := 2 * 2^(n-1)

theorem general_term (n : ℕ) : a n = 2^n := 
by sorry

theorem sum_of_first_n_terms (n : ℕ) : (finset.range n).sum a = 2^(n+1) - 2 :=
by sorry

end general_term_sum_of_first_n_terms_l134_134776


namespace T1_T2_T3_all_theorems_deducible_from_postulates_l134_134315

/-- Postulates definition -/
class Postulates (S : Type) :=
  (pib : S → Prop)
  (maa : S → Prop)
  (P1 : ∀ x, pib x → ∃ y, maa y → S)
  (P2 : ∀ x y, pib x ∧ pib y ∧ x ≠ y → ∃! z, maa z → (z ∈ x ∧ z ∈ y))
  (P3 : ∀ z, maa z → ∃ x y, pib x ∧ pib y ∧ x ≠ y ∧ (z ∈ x ∧ z ∈ y))
  (P4 : ∃! x1 x2 x3 x4, ∀ x, pib x ↔ (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4))

/-- Theorems -/
theorem T1 {S : Type} [Postulates S] : ∃ n, maa n ∧ n = 6 :=
sorry

theorem T2 {S : Type} [Postulates S] : ∃ n, maa n ∧ n = 3 :=
sorry

theorem T3 {S : Type} [Postulates S] : ∀ z, maa z → ∃! w, maa w ∧ w ≠ z → ∀ x, pib x → ¬(z ∈ x ∧ w ∈ x) :=
sorry

/-- Proof that all theorems follow from the postulates -/
theorem all_theorems_deducible_from_postulates {S : Type} [Postulates S] : T1 ∧ T2 ∧ T3 :=
sorry

end T1_T2_T3_all_theorems_deducible_from_postulates_l134_134315


namespace greatest_k_divisor_4_16_factorial_l134_134174

theorem greatest_k_divisor_4_16_factorial : 
  ∃ k : ℕ, (∀ m : ℕ, (m > k) → ¬(4 ^ m ∣ 16!)) ∧ (4 ^ k ∣ 16!) ∧ k = 7 :=
sorry

end greatest_k_divisor_4_16_factorial_l134_134174


namespace fruit_vendor_sold_fruits_l134_134443

def total_dozen_fruits_sold (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ) : ℝ :=
  (lemons_dozen * dozen) + (avocados_dozen * dozen)

theorem fruit_vendor_sold_fruits (hl : ∀ (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ), lemons_dozen = 2.5 ∧ avocados_dozen = 5 ∧ dozen = 12) :
  total_dozen_fruits_sold 2.5 5 12 = 90 :=
by
  sorry

end fruit_vendor_sold_fruits_l134_134443


namespace james_spends_90_dollars_per_week_l134_134781

structure PistachioPurchasing where
  can_cost : ℕ  -- cost in dollars per can
  can_weight : ℕ -- weight in ounces per can
  consumption_oz_per_5days : ℕ -- consumption in ounces per 5 days

def cost_per_week (p : PistachioPurchasing) : ℕ :=
  let daily_consumption := p.consumption_oz_per_5days / 5
  let weekly_consumption := daily_consumption * 7
  let cans_needed := (weekly_consumption + p.can_weight - 1) / p.can_weight -- round up
  cans_needed * p.can_cost

theorem james_spends_90_dollars_per_week :
  cost_per_week ⟨10, 5, 30⟩ = 90 :=
by
  sorry

end james_spends_90_dollars_per_week_l134_134781


namespace correct_operation_l134_134968

theorem correct_operation (a : ℝ) : 
  ((a^2)^3 = a^6) ∧ ¬(a^2 + a^3 = a^5) ∧ ¬(a^2 * a^3 = a^6) ∧ ¬(6 * a^6 - 2 * a^3 = 3 * a^3) :=
by
  -- Provide the four required conditions separately
  -- Option B is correct:
  show (a^2)^3 = a^6, from sorry,
  
  -- Option A is incorrect:
  show ¬(a^2 + a^3 = a^5), from sorry,
  
  -- Option C is incorrect:
  show ¬(a^2 * a^3 = a^6), from sorry,
  
  -- Option D is incorrect:
  show ¬(6 * a^6 - 2 * a^3 = 3 * a^3), from sorry


end correct_operation_l134_134968


namespace flour_already_put_in_l134_134818

theorem flour_already_put_in (total_flour : ℕ) (flour_to_add : ℕ) (h_total_flour : total_flour = 10) (h_flour_to_add : flour_to_add = 4) : (total_flour - flour_to_add) = 6 :=
by 
  rw [h_total_flour, h_flour_to_add]
  -- Perform the actual proof steps here
  sorry

end flour_already_put_in_l134_134818


namespace christmas_bulbs_on_probability_l134_134905

noncomputable def toggle_algorithm (n : ℕ) : ℕ → Bool
| k => (List.range n).foldl (λ b i => if (k + 1) % (i + 1) == 0 then !b else b) true

def is_perfect_square (n : ℕ) : Bool :=
  let root := (n + 1).natAbs.sqrt
  root * root = n + 1

def probability_bulb_on_after_toggling (n : ℕ) : ℚ :=
  let on_count := finset.filter (λ k => toggle_algorithm n k) (finset.range n) 
  on_count.card / n

theorem christmas_bulbs_on_probability : 
  probability_bulb_on_after_toggling 100 = 0.1 :=
sorry

end christmas_bulbs_on_probability_l134_134905


namespace angle_x_is_9_degrees_l134_134050

theorem angle_x_is_9_degrees
  (O B C A D : Type)
  (hO_center : is_center O)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (hOB_eq_OC : distance O B = distance O C)
  (hAngle_BCO : angle B C O = 32)
  (hOA_eq_OC : distance O A = distance O C)
  (hAngle_CAD : angle C A D = 67)
  : angle x = 9 :=
sorry

end angle_x_is_9_degrees_l134_134050


namespace birches_planted_l134_134504

variable 
  (G B X : ℕ) -- G: number of girls, B: number of boys, X: number of birches

-- Conditions:
variable
  (h1 : G + B = 24) -- Total number of students
  (h2 : 3 * G + X = 24) -- Total number of plants
  (h3 : X = B / 3) -- Birches planted by boys

-- Proof statement:
theorem birches_planted : X = 6 :=
by 
  sorry

end birches_planted_l134_134504


namespace probability_of_s25_at_s35_after_one_pass_l134_134861

def bubble_sort_pass_probability (s : List ℕ) (k₁ k₂ : ℕ) (h₁ : (25 ≤ k₁ ∧ k₁ ≤ 50) ∧ (25 ≤ k₂ ∧ k₂ ≤ 50)) : ℚ := 
  if (s.length = 50) then (1 / 1260) else 0

theorem probability_of_s25_at_s35_after_one_pass (s : List ℕ) (h_distinct : s.nodup) (h_length : s.length = 50) :
  (bubble_sort_pass_probability s 25 35 (_, _) = 1 / 1260) := sorry

end probability_of_s25_at_s35_after_one_pass_l134_134861


namespace find_ABC_sum_eq_neg24_l134_134879

theorem find_ABC_sum_eq_neg24
  (A B C : ℤ)
  (h1 : ∀ x > 5, (x^2 : ℝ) / (A * x^2 + B * x + C) > 0.5)
  (h2 : ∃ (D : ℂ), D = A * (x + 3) * (x - 4) ∧ (A * x^2 + B * x + C) = D)
  (h3 : 1 / A < 1 ∧ 1 / A > 0.5) :
  A + B + C = -24 := 
  sorry

end find_ABC_sum_eq_neg24_l134_134879


namespace solve_for_y_l134_134855

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l134_134855


namespace maximum_number_of_cards_per_box_l134_134408

theorem maximum_number_of_cards_per_box (total_cards : ℕ) (boxes_filled : ℕ) (h1 : total_cards = 94) (h2 : boxes_filled = 11) : (total_cards / boxes_filled == 8) :=
by {
  sorry,
}

end maximum_number_of_cards_per_box_l134_134408


namespace sum_first_20_terms_l134_134685

def sequence (n : Nat) : Nat :=
  if n = 1 then 1
  else if n = 2 then 1
  else if n % 2 = 1 then sequence (n - 2) + 2
  else 2 * sequence (n - 2)

def sum_sequence (num_terms : Nat) : Nat :=
  (List.range num_terms).map sequence |>.sum

theorem sum_first_20_terms :
  sum_sequence 20 = 1123 := by
  sorry

end sum_first_20_terms_l134_134685


namespace count_prime_digit_sum_divisible_by_3_l134_134354

-- Define the condition of being a prime single-digit number
def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

-- Define the condition for a number being a two-digit number
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Define the condition that the sum of digits is divisible by 3
def sum_of_digits_divisible_by_3 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  (d1 + d2) % 3 = 0

-- Define the condition that each digit of the number is prime
def digits_are_prime (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  is_prime_digit d1 ∧ is_prime_digit d2

-- State the theorem
theorem count_prime_digit_sum_divisible_by_3 :
  { n : ℕ | is_two_digit_number n ∧ digits_are_prime n ∧ sum_of_digits_divisible_by_3 n }.to_finset.card = 4 :=
sorry

end count_prime_digit_sum_divisible_by_3_l134_134354


namespace similar_rectangles_width_l134_134944

theorem similar_rectangles_width (a b : ℕ) (h : a^2 = b) (w : ℕ) (hw : w = 2) : 
  ∃ w' : ℕ, w' = 6 :=
by
  use 6
  sorry

end similar_rectangles_width_l134_134944


namespace find_value_of_x_l134_134010

-- Definitions used in conditions
variable (O B C A : Point) -- points on the circle
variable (OB OC OA : ℝ) -- Radii of the circle
variable (OB_eq_OC : OB = OC)
variable (angle_BCO : ℝ) -- Given angle ∠BCO
variable (angle_DAC : ℝ) -- Given angle ∠DAC
variable (right_angled_ACD : angle_DAC + 90 = 157) -- Triangle ACD is right-angled & 180 - 23 = 157

variable (x : ℝ) -- angle x

-- Setting specific values for given angles in degrees
noncomputable def angle_BCO_value : angle_BCO = 32
noncomputable def angle_DAC_value : angle_DAC = 67

-- What we want to prove
theorem find_value_of_x : x = 9 :=
sorry

end find_value_of_x_l134_134010


namespace mans_speed_against_current_l134_134173

/-- Given the man's speed with the current and the speed of the current, prove the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ) (speed_of_current : ℝ)
  (h1 : speed_with_current = 16)
  (h2 : speed_of_current = 3.2) :
  speed_with_current - 2 * speed_of_current = 9.6 :=
sorry

end mans_speed_against_current_l134_134173


namespace four_digit_numbers_with_digit_sum_20_count_l134_134352

theorem four_digit_numbers_with_digit_sum_20_count :
  {n : ℕ // n ≥ 1000 ∧ n < 10000 ∧ (nat.digits 10 n).sum = 20 ∧ (n / 100 % 10) >= 20 ∧ (n / 100 % 10) ≤ 22}.card = 6 :=
by
  sorry

end four_digit_numbers_with_digit_sum_20_count_l134_134352


namespace fence_perimeter_l134_134932

-- Definitions for the given conditions
def num_posts : ℕ := 36
def gap_between_posts : ℕ := 6

-- Defining the proof problem to show the perimeter equals 192 feet
theorem fence_perimeter (n : ℕ) (g : ℕ) (sq_field : ℕ → ℕ → Prop) : n = 36 ∧ g = 6 ∧ sq_field n g → 4 * ((n / 4 - 1) * g) = 192 :=
by
  intro h
  sorry

end fence_perimeter_l134_134932


namespace min_groups_required_l134_134676

-- Set definition and basic properties
noncomputable def S : Finset ℤ := Finset.Icc 1 2020 \ (Finset.image (λ n, 5 * n) (Finset.Icc 1 404))

-- Prime condition check
def prime_diff (a b : ℤ) : Prop := ∃ p, Nat.Prime p ∧ abs (a - b) = p

-- Group condition
def group_condition (G : Finset (Finset ℤ)) : Prop :=
  ∀ g ∈ G, ∀ a b ∈ g, a ≠ b → prime_diff a b

-- Minimum number of groups required
theorem min_groups_required : ∃ G : Finset (Finset ℤ), group_condition G ∧ G.card = 404 :=
begin
  sorry
end

end min_groups_required_l134_134676


namespace vectors_parallel_iff_m_l134_134729

theorem vectors_parallel_iff_m (m : ℝ) :
  (let a := (1, m + 1) in let b := (m, 2) in a.1 * b.2 = a.2 * b.1) ↔ (m = -2 ∨ m = 1) :=
by trivial

end vectors_parallel_iff_m_l134_134729


namespace probability_vowel_consonant_initials_l134_134753

def unique_initials (students : ℕ) : Prop :=
  students = 26

def consecutive_initials (initials : List (Char × Char)) : Prop :=
  initials = [
    ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'K'), 
    ('K', 'L'), ('L', 'M'), ('M', 'N'), ('N', 'O'), ('O', 'P'), ('P', 'Q'), ('Q', 'R'), ('R', 'S'), ('S', 'T'), ('T', 'U'),
    ('U', 'V'), ('V', 'W'), ('W', 'X'), ('X', 'Y'), ('Y', 'Z'), ('Z', 'A')
  ]

def is_consonant (c : Char) : Prop :=
  c ≠ 'A' ∧ c ≠ 'E' ∧ c ≠ 'I' ∧ c ≠ 'O' ∧ c ≠ 'U'

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def valid_pairs (initials : List (Char × Char)) : List (Char × Char) :=
  initials.filter (λ (p : Char × Char), is_vowel p.1 ∧ is_consonant p.2)

theorem probability_vowel_consonant_initials :
  unique_initials 26 →
  consecutive_initials [
    ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'K'), 
    ('K', 'L'), ('L', 'M'), ('M', 'N'), ('N', 'O'), ('O', 'P'), ('P', 'Q'), ('Q', 'R'), ('R', 'S'), ('S', 'T'), ('T', 'U'),
    ('U', 'V'), ('V', 'W'), ('W', 'X'), ('X', 'Y'), ('Y', 'Z'), ('Z', 'A')
  ] →
  (valid_pairs [
    ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'K'), 
    ('K', 'L'), ('L', 'M'), ('M', 'N'), ('N', 'O'), ('O', 'P'), ('P', 'Q'), ('Q', 'R'), ('R', 'S'), ('S', 'T'), ('T', 'U'),
    ('U', 'V'), ('V', 'W'), ('W', 'X'), ('X', 'Y'), ('Y', 'Z'), ('Z', 'A')
  ]).length = 4 →
  \((valid_pairs [
    ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'K'), 
    ('K', 'L'), ('L', 'M'), ('M', 'N'), ('N', 'O'), ('O', 'P'), ('P', 'Q'), ('Q', 'R'), ('R', 'S'), ('S', 'T'), ('T', 'U'),
    ('U', 'V'), ('V', 'W'), ('W', 'X'), ('X', 'Y'), ('Y', 'Z'), ('Z', 'A')
  ]).length / 26) = 4 / 26 := sorry

end probability_vowel_consonant_initials_l134_134753


namespace angle_x_is_58_l134_134019

-- Definitions and conditions based on the problem statement
variables {O A C D : Type} (x : ℝ)
variable [circular_center : O] -- O is the center of the circle
variable [radius_equal : AO = BO ∧ BO = CO] -- AO, BO, and CO are radii and hence equal
variable (angle_ACD_right : ∠ A C D = 90) -- ∠ACD is a right angle (subtended by diameter)
variable (angle_CDA : ∠ C D A = 42) -- ∠CDA is 42 degrees

-- Statement to be proven
theorem angle_x_is_58 (x : ℝ) : x = 58 := 
by sorry

end angle_x_is_58_l134_134019


namespace triangle_leg_ratio_l134_134600

theorem triangle_leg_ratio :
  ∀ (a b : ℝ) (h₁ : a = 4) (h₂ : b = 2 * Real.sqrt 5),
    ((a / b) = (2 * Real.sqrt 5) / 5) :=
by
  intros a b h₁ h₂
  sorry

end triangle_leg_ratio_l134_134600


namespace inscribed_circle_radius_approx_l134_134237

noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let term1 := 1/a + 1/b + 1/c
  let term2 := 2 * Real.sqrt (1/(a * b) + 1/(a * c) + 1/(b * c))
  1 / (term1 + term2)

theorem inscribed_circle_radius_approx :
  inscribed_circle_radius 3 5 7 ≈ 0.698 :=
by
  sorry

end inscribed_circle_radius_approx_l134_134237


namespace part1_part2_part3_l134_134683

-- Part (1)
theorem part1 (a : ℝ) (h1 : ∀ x ∈ Ioo 2 5, a * x^2 - (a + 1) * x + 2 > 0) : 
  3 - 2 * Real.sqrt 2 < a ∨ a > 1 / 3 := sorry

-- Part (2)
theorem part2 (x : ℝ) (h2 : ∀ a ∈ Icc (-2) (-1), a * x^2 - (a + 1) * x + 2 > 0) :
  (1 - Real.sqrt 17) / 4 < x ∧ x < (1 + Real.sqrt 17) / 4 := sorry

-- Part (3)
theorem part3 (b : ℝ) (h3 : b > 0)
  (h4 : ∀ x : ℝ, (a : ℝ) → a > 0 → x * b^2 ≤ 8 * a) :
  ∀ a : ℝ, a ≥ b^2 / 8 → (a + 2) / b ≥ 1 := sorry

end part1_part2_part3_l134_134683


namespace minimal_fencing_dimensions_l134_134817

noncomputable def width := 14.625
noncomputable def length := 34.25

theorem minimal_fencing_dimensions 
  (w l : ℝ)
  (h1 : l = 2 * w + 5)
  (h2 : w * l ≥ 500) : 
  w = width ∧ l = length := by
  sorry

end minimal_fencing_dimensions_l134_134817


namespace metro_earnings_in_6_minutes_l134_134836

theorem metro_earnings_in_6_minutes 
  (ticket_cost : ℕ) 
  (tickets_per_minute : ℕ) 
  (duration_minutes : ℕ) 
  (earnings_in_one_minute : ℕ) 
  (earnings_in_six_minutes : ℕ) 
  (h1 : ticket_cost = 3) 
  (h2 : tickets_per_minute = 5) 
  (h3 : duration_minutes = 6) 
  (h4 : earnings_in_one_minute = tickets_per_minute * ticket_cost) 
  (h5 : earnings_in_six_minutes = earnings_in_one_minute * duration_minutes) 
  : earnings_in_six_minutes = 90 := 
by 
  -- Proof goes here
  sorry

end metro_earnings_in_6_minutes_l134_134836


namespace christmas_tree_bulbs_l134_134914

def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

def number_of_bulbs_on (N : Nat) : Nat :=
  (Finset.range (N+1)).filter isPerfectSquare |>.card

def probability_bulb_on (total_bulbs bulbs_on : Nat) : Float :=
  (bulbs_on.toFloat) / (total_bulbs.toFloat)

theorem christmas_tree_bulbs :
  let N := 100
  let bulbs_on := number_of_bulbs_on N
  probability_bulb_on N bulbs_on = 0.1 :=
by
  sorry

end christmas_tree_bulbs_l134_134914


namespace return_trip_time_l134_134464

noncomputable def distance (v u : ℝ) : ℝ := 60 * (v - u)
noncomputable def time_still_air (d v : ℝ) : ℝ := d / v
noncomputable def time_against_wind (d v u : ℝ) : ℝ := d / (v - u)
noncomputable def time_with_wind (d v u : ℝ) : ℝ := d / (v + u)

theorem return_trip_time (v u : ℝ) (h_positive_speed: v > 0) (h_positive_wind: 0 < u) : 
  let d := distance v u in 
  let t_still := time_still_air d v in
  time_with_wind d v u = t_still - 10 → time_with_wind d v u = 20 :=
by
  sorry

end return_trip_time_l134_134464


namespace hyperbola_eccentricity_l134_134344

variables (a b e : ℝ) (F1 F2 P : ℝ × ℝ)

-- The hyperbola assumption
def hyperbola : Prop := ∃ (x y : ℝ), (x, y) = P ∧ x^2 / a^2 - y^2 / b^2 = 1
-- a > 0 and b > 0
def positive_a_b : Prop := a > 0 ∧ b > 0
-- Distance between foci
def distance_foci : Prop := dist F1 F2 = 12
-- Distance PF2
def distance_p_f2 : Prop := dist P F2 = 5
-- To be proven, eccentricity of the hyperbola
def eccentricity : Prop := e = 3 / 2

theorem hyperbola_eccentricity : hyperbola a b P ∧ positive_a_b a b ∧ distance_foci F1 F2 ∧ distance_p_f2 P F2 → eccentricity e :=
by
  sorry

end hyperbola_eccentricity_l134_134344


namespace simplify_expression_l134_134238

theorem simplify_expression : 
  (2020 : ℕ) ^ 4 - 3 * (2020 : ℕ) ^ 3 * (2020 + 1 : ℕ) + 4 * (2020 : ℕ) ^ 2 * (2020 + 1 : ℕ) ^ 2 - (2020 + 1 : ℕ) ^ 4 + 1) / (2020 * (2020 + 1 : ℕ)) = (2020 : ℕ) ^ 2 - 2 := 
by
  sorry

end simplify_expression_l134_134238


namespace simplify_f_find_tan_of_f_l134_134324

variable (α : Real) (h₁ : α ∈ set.Icc (π) (3 * π / 2))
variable (f : Real → Real) (hf : ∀ α, f(α) = (sin (α - π / 2) * cos (3 * π / 2 + α) * tan (π - α)) / (tan (-α - π) * sin (-α - π)))

theorem simplify_f : f α = cos α := sorry

theorem find_tan_of_f (h₂ : f α = 4 / 5) : tan α = 3 / 4 := sorry

end simplify_f_find_tan_of_f_l134_134324


namespace money_left_is_correct_l134_134928

-- Define the conditions
def initial_amount : ℕ := 78
def spent_amount : ℕ := 15
def remaining_amount : ℕ := initial_amount - spent_amount

-- Theorem stating the proof problem
theorem money_left_is_correct : remaining_amount = 63 := by
  calculate_data := rfl sorry 

end money_left_is_correct_l134_134928


namespace no_positive_integral_solution_l134_134289

theorem no_positive_integral_solution :
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ p : ℕ, Prime p ∧ n^2 - 45 * n + 520 = p :=
by {
  -- Since we only need the statement, we'll introduce the necessary steps without the full proof
  sorry
}

end no_positive_integral_solution_l134_134289


namespace angle_x_l134_134029

-- Define the conditions
variables (O A C D B : Point)
variable h : circle_center O
variable h1 : on_circumference A C D O
variable h2 : diameter AC
variable h3 : ∠ CDA = 42

-- The problem statement we want to prove
theorem angle_x (x : ℝ) : x = 58 :=
begin
  -- This is where the solution steps would go, but for now, we leave it as sorry
  sorry
end

end angle_x_l134_134029


namespace polynomial_divisor_specific_polynomial_divisors_l134_134792

open BigOperators
noncomputable theory

-- Statement for Part (a)
theorem polynomial_divisor (n : ℕ) (h_n : 3 ≤ n) : 
  ∃ p : ℕ, p > 0 ∧ (x^p + 1 ∣ (x^(2^n - 1).polynomial.indeterminate_sub 1)/(x - 1) - x^n) :=
sorry

-- Statement for Part (b)
theorem specific_polynomial_divisors :
  let f : polynomial ℝ := (x^127 - 1)/(x - 1) - x^7 in
  (x + 1 ∣ f) ∧ (x^2 + 1 ∣ f) ∧ (x^4 + 1 ∣ f) :=
sorry

end polynomial_divisor_specific_polynomial_divisors_l134_134792


namespace circumcircle_BOC_tangent_CD_l134_134474

-- Given definitions and conditions
variables {A B C D O : Point} [parallelogram : Parallelogram A B C D]
variables (H1 : DiagonalsIntersectAt A B C D O)
variables (H2 : TangentCircumcircle A O B BC)

-- Statement to prove
theorem circumcircle_BOC_tangent_CD :
  TangentCircumcircle B O C CD :=
sorry

end circumcircle_BOC_tangent_CD_l134_134474


namespace pentagon_cyclic_radius_l134_134691

-- Define the convex pentagon and its properties
variables {A B C D E : Point}

-- Given conditions as angles
axiom angle_A : ∠ A B C = 60
axiom angle_B : ∠ B C D = 100
axiom angle_C : ∠ C D E = 140

-- Desired proof that the pentagon can be inscribed in a circle with radius \( \frac{2}{3} \) of AD
theorem pentagon_cyclic_radius :
  ∃ (O : Point) (r : Real), IsCyclicPentagon A B C D E ∧ r = (2 / 3) * distance A D :=
sorry

end pentagon_cyclic_radius_l134_134691


namespace total_carriages_in_towns_l134_134167

noncomputable def total_carriages (euston norfolk norwich flyingScotsman victoria waterloo : ℕ) : ℕ :=
  euston + norfolk + norwich + flyingScotsman + victoria + waterloo

theorem total_carriages_in_towns :
  let euston := 130
  let norfolk := euston - (20 * euston / 100)
  let norwich := 100
  let flyingScotsman := 3 * norwich / 2
  let victoria := euston - (15 * euston / 100)
  let waterloo := 2 * norwich
  total_carriages euston norfolk norwich flyingScotsman victoria waterloo = 794 :=
by
  sorry

end total_carriages_in_towns_l134_134167


namespace find_increasing_function_l134_134537

-- Define each function
def fA (x : ℝ) := -x
def fB (x : ℝ) := (2 / 3) ^ x
def fC (x : ℝ) := x ^ 2
def fD (x : ℝ) := x ^ (1 / 3)

-- Define the statement that fD is the only increasing function among the options
theorem find_increasing_function (f : ℝ → ℝ) (hf : f = fD) :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fA x < fA y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fB x < fB y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fC x < fC y) :=
by {
  sorry
}

end find_increasing_function_l134_134537


namespace bottle_caps_calculation_l134_134916

theorem bottle_caps_calculation (initial_caps : ℕ) (marvins_taken_caps : ℕ) (additional_caps : ℕ) :
  initial_caps = 144 → marvins_taken_caps = 63 → additional_caps = 56 →
  let remaining_caps := initial_caps - marvins_taken_caps in
  let final_caps := remaining_caps + additional_caps in
  final_caps = 137 :=
by
  intros h_initial h_taken h_additional
  let remaining_caps := initial_caps - marvins_taken_caps
  let final_caps := remaining_caps + additional_caps
  have h1 : initial_caps = 144 := h_initial
  have h2 : marvins_taken_caps = 63 := h_taken
  have h3 : additional_caps = 56 := h_additional
  have h4 : remaining_caps = 144 - 63 := by rw [h1, h2]
  have h5 : final_caps = (144 - 63) + 56 := by rw [h4, h3]
  have h6 : final_caps = 137 := by norm_num at h5
  exact h6

end bottle_caps_calculation_l134_134916


namespace length_of_EF_in_folded_rectangle_l134_134391

noncomputable def midpoint (a b : ℝ) := (a + b) / 2

theorem length_of_EF_in_folded_rectangle :
  ∀ (A B C D M : ℝ) (AB BC : ℝ), 
  AB = 5 → 
  BC = 7 → 
  A = 0 → 
  B = AB → 
  C = complex.mk 7 0 → 
  D = complex.mk 7 5 → 
  M = midpoint 0 7 →
  EF = 3.5 :=
by
  intros A B C D M AB BC HAB HBC HA HB HC HD HM
  sorry

end length_of_EF_in_folded_rectangle_l134_134391


namespace ming_belief_contradiction_l134_134170

theorem ming_belief_contradiction
  (A B C : Type)
  [Plane A B C]
  (angle_B angle_C : Angle)
  (side_AC side_AB : Length) :
  (angle_B ≠ angle_C) → (AC = AB) → False :=
begin
  sorry
end

end ming_belief_contradiction_l134_134170


namespace enclosed_region_area_l134_134152

theorem enclosed_region_area :
  (∃ x y : ℝ, x ^ 2 + y ^ 2 - 6 * x + 8 * y = -9) →
  ∃ (r : ℝ), r ^ 2 = 16 ∧ ∀ (area : ℝ), area = π * 4 ^ 2 :=
by
  sorry

end enclosed_region_area_l134_134152


namespace number_of_envelopes_requiring_extra_postage_is_2_l134_134466

def envelope_requires_extra_postage (length height : ℕ) : Prop :=
length / height < 3 ∧ length / height > 1.5

def envelope_A_length : ℕ := 8
def envelope_A_height : ℕ := 5

def envelope_B_length : ℕ := 10
def envelope_B_height : ℕ := 2

def envelope_C_length : ℕ := 8
def envelope_C_height : ℕ := 8

def envelope_D_length : ℕ := 14
def envelope_D_height : ℕ := 5

def count_extra_postage_envelopes : ℕ :=
(list.length (list.filter (λ (length height : ℕ), length / height < 1.5 ∨ length / height > 3.0)
  [(envelope_A_length, envelope_A_height), (envelope_B_length, envelope_B_height),
    (envelope_C_length, envelope_C_height), (envelope_D_length, envelope_D_height)]))

theorem number_of_envelopes_requiring_extra_postage_is_2 :
  count_extra_postage_envelopes = 2 :=
sorry

end number_of_envelopes_requiring_extra_postage_is_2_l134_134466


namespace f_0_plus_f_neg2_eq_neg4_l134_134682

def f : ℝ → ℝ 
  := sorry

axiom additivity : ∀ x y : ℝ, f(x) + f(y) = f(x + y)

axiom f_of_2 : f (2) = 4 

theorem f_0_plus_f_neg2_eq_neg4 : f(0) + f(-2) = -4 :=
sorry

end f_0_plus_f_neg2_eq_neg4_l134_134682


namespace point_ratio_combination_l134_134795

noncomputable def vec (T : Type) [AddCommGroup T] [Module ℝ T] := T

variables {T : Type} [AddCommGroup T] [Module ℝ T]

theorem point_ratio_combination (C D Q : vec T) (h : (C -ᵥ Q : vec T) = (3 : ℝ) • (D -ᵥ Q) / 4) :
  Q = (3 / 7 : ℝ) • C + (4 / 7 : ℝ) • D :=
sorry

end point_ratio_combination_l134_134795


namespace maximum_black_squares_l134_134654

theorem maximum_black_squares (n : ℕ) (h : n = 1000) :
  let total_squares := 6 * n^2 in
  let strips := 500 in
  let max_black_squares_by_strip := λ k, 3*k - 2 in
  let total_black_squares := (n * n * n / 2) - (2*n - 1) in
  total_black_squares = 2998000 :=
by
  sorry

end maximum_black_squares_l134_134654


namespace differentiable_implies_continuous_l134_134844

-- Theorem: If a function f is differentiable at x0, then it is continuous at x0.
theorem differentiable_implies_continuous {f : ℝ → ℝ} {x₀ : ℝ} (h : DifferentiableAt ℝ f x₀) : 
  ContinuousAt f x₀ :=
sorry

end differentiable_implies_continuous_l134_134844


namespace water_tank_length_l134_134573

theorem water_tank_length (n : ℕ) (avg_displacement : ℝ) (breadth : ℝ) (rise : ℝ) : 
  n = 50 → 
  avg_displacement = 4 → 
  breadth = 20 → 
  rise = 0.25 → 
  let volume := n * avg_displacement in
  let length := volume / (breadth * rise) in
  length = 40 :=
by 
  intros h1 h2 h3 h4 
  let volume := n * avg_displacement
  let length := volume / (breadth * rise)
  show length = 40 from sorry

end water_tank_length_l134_134573


namespace solve_quadratic_l134_134084

   theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 5 * x^2 + 8 * x - 24 = 0) : x = 6 / 5 :=
   sorry
   
end solve_quadratic_l134_134084


namespace intersection_M_N_l134_134501

open Set

-- Definitions of sets M and N
def M : Set (ℝ × ℝ) := { a | ∃ x : ℝ, a = (-1, 1) + x • (1, 2) }
def N : Set (ℝ × ℝ) := { a | ∃ x : ℝ, a = (1, -2) + x • (2, 3) }

-- Theorem stating that the intersection of M and N is exactly {(-13, -23)}
theorem intersection_M_N : M ∩ N = {(-13, -23)} := by
  sorry

end intersection_M_N_l134_134501


namespace total_weekly_pay_l134_134942

theorem total_weekly_pay (Y_pay: ℝ) (X_pay: ℝ) (Y_weekly: Y_pay = 150) (X_weekly: X_pay = 1.2 * Y_pay) : 
  X_pay + Y_pay = 330 :=
by sorry

end total_weekly_pay_l134_134942


namespace pictures_at_the_museum_l134_134550

/-- Megan took 15 pictures at the zoo, deleted 31 pictures, and still has 2 pictures after deletion.
    Prove that the number of pictures Megan took at the museum is 18. -/
theorem pictures_at_the_museum (M : ℕ) (hz : 15) (hd : 31) (hr : 2) :
  2 = (15 + M) - 31 → M = 18 := by
  sorry

end pictures_at_the_museum_l134_134550


namespace increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l134_134546

noncomputable def fA (x : ℝ) : ℝ := -x
noncomputable def fB (x : ℝ) : ℝ := (2/3)^x
noncomputable def fC (x : ℝ) : ℝ := x^2
noncomputable def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function_fA : ¬∀ x y : ℝ, x < y → fA x < fA y := sorry
theorem increasing_function_fB : ¬∀ x y : ℝ, x < y → fB x < fB y := sorry
theorem increasing_function_fC : ¬∀ x y : ℝ, x < y → fC x < fC y := sorry
theorem increasing_function_fD : ∀ x y : ℝ, x < y → fD x < fD y := sorry

end increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l134_134546


namespace ming_belief_contradiction_l134_134171

theorem ming_belief_contradiction
  (A B C : Type)
  [Plane A B C]
  (angle_B angle_C : Angle)
  (side_AC side_AB : Length) :
  (angle_B ≠ angle_C) → (AC = AB) → False :=
begin
  sorry
end

end ming_belief_contradiction_l134_134171


namespace insurance_percentage_l134_134135

noncomputable def total_pills_per_year : ℕ := 2 * 365

noncomputable def cost_per_pill : ℕ := 5

noncomputable def total_medication_cost_per_year : ℕ := total_pills_per_year * cost_per_pill

noncomputable def doctor_visits_per_year : ℕ := 2

noncomputable def cost_per_doctor_visit : ℕ := 400

noncomputable def total_doctor_cost_per_year : ℕ := doctor_visits_per_year * cost_per_doctor_visit

noncomputable def total_yearly_cost_without_insurance : ℕ := total_medication_cost_per_year + total_doctor_cost_per_year

noncomputable def total_payment_per_year : ℕ := 1530

noncomputable def insurance_coverage_per_year : ℕ := total_yearly_cost_without_insurance - total_payment_per_year

theorem insurance_percentage:
  (insurance_coverage_per_year * 100) / total_medication_cost_per_year = 80 :=
by sorry

end insurance_percentage_l134_134135


namespace vector_BE_l134_134406

open EuclideanGeometry -- Assume we are working in the Euclidean geometry context

-- Define the points A, B, C, D and E
variables {A B C D E : Point}

-- Define the vector operations
variables (vBA vBC vBE vAD : Vect)

-- Given conditions
-- D is the midpoint of BC
axiom midpoint_D : D = midpoint B C

-- E is the midpoint of AD
axiom midpoint_E : E = midpoint A D

-- The statement to prove
theorem vector_BE :
  vBE = 1/2 * vBA + 1/4 * vBC :=
by
  sorry

end vector_BE_l134_134406


namespace percentage_vanilla_orders_l134_134077

theorem percentage_vanilla_orders 
  (V C : ℕ) 
  (h1 : V = 2 * C) 
  (h2 : V + C = 220) 
  (h3 : C = 22) : 
  (V * 100) / 220 = 20 := 
by 
  sorry

end percentage_vanilla_orders_l134_134077


namespace ramsey_theorem_l134_134294

theorem ramsey_theorem (r : ℕ) : ∃ (n : ℕ), ∀ (G : SimpleGraph ℕ), G.vertex_count ≥ n → (∃ (H : SimpleGraph ℕ), H.vertices ⊆ G.vertices ∧ (H = complete_graph r ∨ H = complement (complete_graph r))) :=
sorry

end ramsey_theorem_l134_134294


namespace eighth_term_of_geometric_sequence_l134_134874

noncomputable def geometric_sequence_8th_term (a r : ℝ) (h₁ : a * r^4 = 11) (h₂ : a * r^10 = 5) : ℝ :=
11 * real.sqrt (5 / 11)

theorem eighth_term_of_geometric_sequence
  (a r : ℝ) (h₁ : a * r^4 = 11) (h₂ : a * r^10 = 5) :
  geometric_sequence_8th_term a r h₁ h₂ = real.sqrt 55 := 
sorry

end eighth_term_of_geometric_sequence_l134_134874


namespace order_of_numbers_l134_134190

noncomputable def a : ℝ := 60.7
noncomputable def b : ℝ := 0.76
noncomputable def c : ℝ := Real.log 0.76

theorem order_of_numbers : (c < b) ∧ (b < a) :=
by
  have h1 : c = Real.log 0.76 := rfl
  have h2 : b = 0.76 := rfl
  have h3 : a = 60.7 := rfl
  have hc : c < 0 := sorry
  have hb : 0 < b := sorry
  have ha : 1 < a := sorry
  sorry 

end order_of_numbers_l134_134190


namespace percent_area_contained_l134_134217

-- Define the conditions as Lean definitions
def side_length_square (s : ℝ) : ℝ := s
def width_rectangle (s : ℝ) : ℝ := 2 * s
def length_rectangle (s : ℝ) : ℝ := 3 * (width_rectangle s)

-- Define areas based on definitions
def area_square (s : ℝ) : ℝ := (side_length_square s) ^ 2
def area_rectangle (s : ℝ) : ℝ := (length_rectangle s) * (width_rectangle s)

-- The main theorem stating the percentage of the rectangle's area contained within the square
theorem percent_area_contained (s : ℝ) (h : s ≠ 0) :
  (area_square s / area_rectangle s) * 100 = 8.33 := by
  sorry

end percent_area_contained_l134_134217


namespace eccentricity_of_ellipse_l134_134107

theorem eccentricity_of_ellipse 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) 
  : let C := λ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1,
    left_focus := (λ c : ℝ, (-c, 0)),
    symmetric_point := λ (F : ℝ × ℝ) (m n : ℝ), 
                      (d_eq_0 : (n / (m + F.1)) * (-sqrt 3) = -1) ∧ 
                      (sqrt 3 * (m - F.1) / 2 + n / 2 = 0),
    c := sqrt (a^2 - b^2),
    A := let F := left_focus c in 
           let (m, n) := (c / 2, sqrt 3 * c / 2) in (m, n)
  in (sqrt (a^2 - b^2) / a) = sqrt 3 - 1 :=
by {
  sorry
}

end eccentricity_of_ellipse_l134_134107


namespace interest_rate_first_year_l134_134659

theorem interest_rate_first_year (R : ℚ)
  (principal : ℚ := 7000)
  (final_amount : ℚ := 7644)
  (time_period_first_year : ℚ := 1)
  (time_period_second_year : ℚ := 1)
  (rate_second_year : ℚ := 5) :
  principal + (principal * R * time_period_first_year / 100) + 
  ((principal + (principal * R * time_period_first_year / 100)) * rate_second_year * time_period_second_year / 100) = final_amount →
  R = 4 := 
by {
  sorry
}

end interest_rate_first_year_l134_134659


namespace angle_x_is_9_degrees_l134_134052

theorem angle_x_is_9_degrees
  (O B C A D : Type)
  (hO_center : is_center O)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (hOB_eq_OC : distance O B = distance O C)
  (hAngle_BCO : angle B C O = 32)
  (hOA_eq_OC : distance O A = distance O C)
  (hAngle_CAD : angle C A D = 67)
  : angle x = 9 :=
sorry

end angle_x_is_9_degrees_l134_134052


namespace last_triangle_perimeter_l134_134796

theorem last_triangle_perimeter (T : ℕ → triangle) (a₁ a₂ a₃ : ℕ)
  (hT1 : T 1 = ⟨2011, 2012, 2013⟩)
  (h_seq : ∀ n ≥ 1, let ⟨a, b, c⟩ := T n in 
    let AD := (b + c - a) / 2 in
    let BE := (a + c - b) / 2 in
    let CF := (a + b - c) / 2 in
    T (n + 1) = ⟨AD, BE, CF⟩) :
  ∃ n, T (n + 1) does_not_exist ∧ (T n).perimeter = 1509 / 128 := 
sorry

end last_triangle_perimeter_l134_134796


namespace committee_selection_l134_134766

theorem committee_selection:
  (∃ president ∈ finset.range 10, 
   ∃ committee ∈ finset.powersetLen 2 (finset.filter (λ x, x > 30) (finset.range 10)), 
   ∀ c ∈ committee, c ≠ president) →
  120 := 
sorry

end committee_selection_l134_134766


namespace count_valid_rearrangements_wxyz_l134_134737

def is_adjacency_violated (a b : Char) : Prop :=
  abs (Char.toNat a - Char.toNat b) = 1

def valid_rearrangement (s : List Char) : Prop :=
  s.length = 4 ∧
  (∀ i, i < 3 → ¬is_adjacency_violated (s.get ⟨i, by linarith⟩) (s.get ⟨i + 1, by linarith⟩)) ∧
  (∀ i, i = 1 ∨ i = 3 → 
    ¬is_adjacency_violated (s.get ⟨i, by linarith⟩) (s.get ⟨i + 1, by linarith⟩)
    ∧ ¬is_adjacency_violated (s.get ⟨i, by linarith⟩) (s.get ⟨i - 1, by linarith⟩))

theorem count_valid_rearrangements_wxyz : 
  ∃! n : ℕ, n = 2 ∧ ∃ s : List (List Char), 
  s.length = n ∧ 
  (∀ t ∈ s, valid_rearrangement t) ∧ 
  (∀ t1 t2, t1 ∈ s ∧ t2 ∈ s → t1 ≠ t2) := 
sorry

end count_valid_rearrangements_wxyz_l134_134737


namespace solve_for_x_l134_134461

theorem solve_for_x : 
  ∃ (x : ℝ), (∛ (15 * x + ∛ (15 * x + 17)) = 18) := 
by 
  use 387
  have h1 : (15 * (387 : ℝ) + 17) = 5812 := by norm_num
  have h2 : ∛ (5812) = 18 := by norm_num -- This line should handle the computation of cube roots, can be replaced by calculator or verified values.
  show ∛ (15 * (387 : ℝ) + ∛ (15 * (387 : ℝ) + 17)) = 18
  sorry

end solve_for_x_l134_134461


namespace DanielAgeIs30_l134_134523

def UncleBobAge : ℕ := 60
def ElizabethAge := (2 * UncleBobAge) / 3
def DanielAge := ElizabethAge - 10

theorem DanielAgeIs30 : DanielAge = 30 :=
by
  unfold UncleBobAge ElizabethAge DanielAge
  simp
  rfl

end DanielAgeIs30_l134_134523


namespace num_solutions_eq_4_l134_134637

theorem num_solutions_eq_4 (θ : ℝ) (h : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  ∃ n : ℕ, n = 4 ∧ (2 + 4 * Real.cos θ - 6 * Real.sin (2 * θ) + 3 * Real.tan θ = 0) :=
sorry

end num_solutions_eq_4_l134_134637


namespace segments_satisfy_conditions_l134_134863

-- Define the lengths of the pencil and the eraser
variables {AB AD AH HB : ℝ}

-- Define the conditions of the problem
def conditions (AB AD AH HB : ℝ) : Prop :=
  AH + HB = AB ∧ ∃ x, x = AH * HB ∧ sqrt x = AD

-- Define the question we want to prove
theorem segments_satisfy_conditions (AB AD AH HB : ℝ) (h : conditions AB AD AH HB) : 
  AH + HB = AB ∧ sqrt (AH * HB) = AD :=
by
  -- The proof would go here
  sorry

end segments_satisfy_conditions_l134_134863


namespace average_value_of_T_l134_134089

theorem average_value_of_T :
  ∃ (T : Finset ℕ) (b_1 b_m : ℕ), 
  T.nonempty ∧ T.card = 12 ∧ 
  (∀ t ∈ T, t > 0) ∧ 
  b_1 = T.min' (by sorry) ∧ b_m = T.max' (by sorry) ∧ 
  (∑ t in T \ {b_1}, t) = 50 * (T.card - 1) ∧  
  (∑ t in (T \ {b_1, b_m}).toFinset, t) = 45 * (T.card - 2) ∧ 
  (∑ t in (insert b_1 ((T \ {b_m}).toFinset)), t) = 55 * (T.card - 1) ∧ 
  b_m = b_1 + 100 ∧ 
  (∑ t in T, t) / T.card = 50.83 :=
sorry

end average_value_of_T_l134_134089


namespace sum_of_solutions_l134_134888

theorem sum_of_solutions : 
  let a := 1
  let b := -7
  let c := -30
  (a * x^2 + b * x + c = 0) → ((-b / a) = 7) :=
by
  sorry

end sum_of_solutions_l134_134888


namespace Elle_in_seat_2_given_conditions_l134_134276

theorem Elle_in_seat_2_given_conditions
    (seats : Fin 4 → Type) -- Representation of the seating arrangement.
    (Garry Elle Fiona Hank : Type)
    (seat_of : Type → Fin 4)
    (h1 : seat_of Garry = 0) -- Garry is in seat #1 (index 0)
    (h2 : ¬ (seat_of Elle = seat_of Hank + 1 ∨ seat_of Elle = seat_of Hank - 1)) -- Elle is not next to Hank
    (h3 : ¬ (seat_of Fiona > seat_of Garry ∧ seat_of Fiona < seat_of Hank) ∧ ¬ (seat_of Fiona < seat_of Garry ∧ seat_of Fiona > seat_of Hank)) -- Fiona is not between Garry and Hank
    : seat_of Elle = 1 :=  -- Conclusion: Elle is in seat #2 (index 1)
    sorry

end Elle_in_seat_2_given_conditions_l134_134276


namespace representation_bound_l134_134293

-- Define the function f(n) which represents the number of ways to represent n as a sum of non-negative powers of 2
def f (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (List.range n).count (λ x, 2^x = n)

theorem representation_bound (n : ℕ) (hn : n ≥ 3) :
  2^((n^2) / 4) < f(2^n) ∧ f(2^n) < 2^((n^2) / 2) :=
by
  sorry -- Placeholder for the actual proof

end representation_bound_l134_134293


namespace earnings_pool_cleaning_correct_l134_134557

-- Definitions of the conditions
variable (Z : ℕ) -- Number of times Zoe babysat Zachary
variable (earnings_total : ℝ := 8000) 
variable (earnings_Zachary : ℝ := 600)
variable (earnings_per_session : ℝ := earnings_Zachary / Z)
variable (sessions_Julie : ℕ := 3 * Z)
variable (sessions_Chloe : ℕ := 5 * Z)

-- Calculation of earnings from babysitting
def earnings_Julie : ℝ := sessions_Julie * earnings_per_session
def earnings_Chloe : ℝ := sessions_Chloe * earnings_per_session
def earnings_babysitting_total : ℝ := earnings_Zachary + earnings_Julie + earnings_Chloe

-- Calculation of earnings from pool cleaning
def earnings_pool_cleaning : ℝ := earnings_total - earnings_babysitting_total

-- The theorem we are interested in
theorem earnings_pool_cleaning_correct :
  earnings_pool_cleaning Z = 2600 := by
  sorry

end earnings_pool_cleaning_correct_l134_134557


namespace primes_in_range_l134_134669

theorem primes_in_range (n : ℕ) (h1 : n > 2) : 
  let lower_bound := (n+1)! + 1
  let upper_bound := (n+1)! + (n+1)
  let is_composite (k : ℕ) := ∃ m, m > 1 ∧ m < k ∧ (k % m = 0) 
  ∀ a, lower_bound < a ∧ a < upper_bound → is_composite a :=
begin
  sorry
end

end primes_in_range_l134_134669


namespace probability_bulb_on_l134_134908

def toggle_process_result : ℕ → Bool
| n := (count_divisors n) % 2 = 1

def count_divisors (n : ℕ) : ℕ :=
  (finset.Icc 1 n).count (λ d : ℕ, n % d = 0)

theorem probability_bulb_on :
  let total_bulbs := 100
  let remaining_bulbs := finset.range 101 |>.filter toggle_process_result |>.card
  remaining_bulbs / total_bulbs = 0.1 :=
sorry

end probability_bulb_on_l134_134908


namespace num_int_vals_n_cube_neg100_lt_pos100_l134_134265

theorem num_int_vals_n_cube_neg100_lt_pos100 : 
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := 
by 
  sorry

end num_int_vals_n_cube_neg100_lt_pos100_l134_134265


namespace find_range_a_l134_134255

noncomputable def f (x : ℝ) (a : ℝ) (c : ℝ) : ℝ := exp x - a * log x + c
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := exp x - a * log x
noncomputable def h (x : ℝ) : ℝ := exp x / log x

theorem find_range_a (a : ℝ) (c : ℝ) (h1 : a > 0) (h2 : c ≠ 0) 
  (h3 : g (1, a) = exp 1) (h4 : ∃! x, (2 < x ∧ x < 3 ∧ g (x, a) = 0)) :
  (exp 2 / log 2) < a ∧ a < (exp 3 / log 3) :=
sorry

end find_range_a_l134_134255


namespace problem1_l134_134187

variable (P : Real × Real) (a b : Real)

def hyperbola_eq (a b : Real) (P : Real × Real) : Prop :=
  (P.2 / a) ^ 2 - (P.1 / b) ^ 2 = 1

theorem problem1
  (a b : Real)
  (h1 : 3 * a = 2 * b)
  (h2 : P = (sqrt 6, 2))
  (ha_pos : a > 0) 
  (hb_pos : b > 0) :
  hyperbola_eq a b P ↔ (a = sqrt 3 ∧ b = sqrt 9) := 
  sorry

end problem1_l134_134187


namespace bulbs_probability_l134_134901

/-
There are 100 light bulbs initially all turned on.
Every second bulb is toggled after one second.
Every third bulb is toggled after two seconds, and so on.
This process continues up to 100 seconds.
We aim to prove the probability that a randomly selected bulb is on after 100 seconds is 0.1.
-/

theorem bulbs_probability : 
  let bulbs := {1..100}
  let perfect_squares := {n ∈ bulbs | ∃ k, n = k * k}
  let total_bulbs := 100
  let bulbs_on := Set.card perfect_squares
  (bulbs_on : ℕ) / total_bulbs = 0.1 :=
by
  -- We use the solution steps to take perfect squares directly
  have h : Set.card perfect_squares = 10 := sorry
  have t : total_bulbs = 100 := rfl
  rw [h]
  norm_num


end bulbs_probability_l134_134901


namespace minimum_value_of_f_l134_134954

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + |x + 8| + |x - 5|

-- State the theorem
theorem minimum_value_of_f : ∃ x, f x = -25 :=
by
  sorry

end minimum_value_of_f_l134_134954


namespace find_m_of_slope_45_l134_134109

theorem find_m_of_slope_45 (m : ℝ) :
    let M := (-2, m)
    let N := (1, 4)
    let slope := (4 - m)/(1 + 2)
    slope = 1 → m = 1 :=
by
  intros M N slope h
  have h1 : (4 - m) / 3 = 1 := h
  sorry

end find_m_of_slope_45_l134_134109


namespace number_of_valid_k_l134_134112

theorem number_of_valid_k : 
  (∃ k : ℕ, (0 < k ∧ (∃ x : ℤ, k * x - 18 = 3 * k))) → 
  (number_of_divisors 18 = 6) :=
by
  sorry

end number_of_valid_k_l134_134112


namespace parallel_heater_time_l134_134943

theorem parallel_heater_time (t1 t2 : ℕ) (R1 R2 : ℝ) (t : ℕ) (I : ℝ) (Q : ℝ) (h₁ : t1 = 3) 
  (h₂ : t2 = 6) (hq1 : Q = I^2 * R1 * t1) (hq2 : Q = I^2 * R2 * t2) :
  t = (t1 * t2) / (t1 + t2) := by
  sorry

end parallel_heater_time_l134_134943


namespace reflection_eq_l134_134479

theorem reflection_eq (x y : ℝ) : 
    let line_eq (x y : ℝ) := 2 * x + 3 * y - 5 = 0 
    let reflection_eq (x y : ℝ) := 3 * x + 2 * y - 5 = 0 
    (∀ (x y : ℝ), line_eq x y ↔ reflection_eq y x) →
    reflection_eq x y :=
by
    sorry

end reflection_eq_l134_134479


namespace thirtieth_triangular_number_l134_134949

theorem thirtieth_triangular_number : 
  let T (n : ℕ) := n * (n + 1) / 2 in
  T 30 = 465 :=
by
  let T := λ n : ℕ, n * (n + 1) / 2
  sorry

end thirtieth_triangular_number_l134_134949


namespace width_of_room_l134_134483

theorem width_of_room (length cost_total cost_per_sqm : ℝ) 
  (h_length: length = 6) 
  (h_cost_total: cost_total = 25650) 
  (h_cost_per_sqm: cost_per_sqm = 900) : 
  (cost_total / cost_per_sqm) / length = 4.75 :=
by {
  rw [h_length, h_cost_total, h_cost_per_sqm],
  norm_num,
}

end width_of_room_l134_134483


namespace uneaten_pizza_pieces_l134_134607

-- Condition Definitions
def total_pizzas : ℕ := 4
def pieces_per_pizza : ℕ := 4

def b_d_eaten_fraction : ℝ := 0.5
def a_c_eaten_fraction : ℝ := 0.75

-- Mathematically Equivalent Proof Problem
theorem uneaten_pizza_pieces :
  let total_pieces : ℕ := total_pizzas * pieces_per_pizza,
      b_d_eaten_pieces : ℝ := 2 * pieces_per_pizza * b_d_eaten_fraction,
      a_c_eaten_pieces : ℝ := 2 * pieces_per_pizza * a_c_eaten_fraction,
      eaten_pieces : ℝ := b_d_eaten_pieces + a_c_eaten_pieces,
      uneaten_pieces : ℝ := total_pieces - eaten_pieces 
  in uneaten_pieces = 6 := 
by { sorry }

end uneaten_pizza_pieces_l134_134607


namespace triangle_angles_angle_DAC_l134_134411

variable (A B C D : Type) [IsoscelesTriangle A B C]
variable (angle_DBC : Angle D B C = 30)
variable (angle_DBA : Angle D B A = 50)
variable (angle_DCB : Angle D C B = 55)

theorem triangle_angles (A B C D : Type)
  [IsoscelesTriangle A B C]
  (angle_DBC : Angle D B C = 30)
  (angle_DBA : Angle D B A = 50)
  (angle_DCB : Angle D C B = 55) :
  Angle B = 80 ∧ Angle C = 80 := 
sorry

theorem angle_DAC (A B C D : Type)
  [IsoscelesTriangle A B C]
  (angle_DBC : Angle D B C = 30)
  (angle_DBA : Angle D B A = 50)
  (angle_DCB : Angle D C B = 55) :
  Angle D A C = 5 := 
sorry

end triangle_angles_angle_DAC_l134_134411


namespace number_of_arrangements_BANANAS_l134_134640

theorem number_of_arrangements_BANANAS : 
  let total_permutations := 7!
      a_repeats := 3!
      n_repeats := 2!
  in total_permutations / (a_repeats * n_repeats) = 420 :=
by
  sorry

end number_of_arrangements_BANANAS_l134_134640


namespace find_increasing_function_l134_134538

-- Define each function
def fA (x : ℝ) := -x
def fB (x : ℝ) := (2 / 3) ^ x
def fC (x : ℝ) := x ^ 2
def fD (x : ℝ) := x ^ (1 / 3)

-- Define the statement that fD is the only increasing function among the options
theorem find_increasing_function (f : ℝ → ℝ) (hf : f = fD) :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fA x < fA y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fB x < fB y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fC x < fC y) :=
by {
  sorry
}

end find_increasing_function_l134_134538


namespace arrangement_of_bananas_l134_134642

-- Define the constants for the number of letters and repetitions in the word BANANAS.
def num_letters : ℕ := 7
def count_A : ℕ := 3
def count_N : ℕ := 2
def factorial (n : ℕ) := nat.factorial n

-- The number of ways to arrange the letters of the word BANANAS.
noncomputable def num_ways_to_arrange := 
  (factorial num_letters) / (factorial count_A * factorial count_N)

theorem arrangement_of_bananas : 
  num_ways_to_arrange = 420 :=
sorry

end arrangement_of_bananas_l134_134642


namespace least_number_to_subtract_l134_134161

theorem least_number_to_subtract 
  (n : ℤ) 
  (h1 : 7 ∣ (90210 - n + 12)) 
  (h2 : 11 ∣ (90210 - n + 12)) 
  (h3 : 13 ∣ (90210 - n + 12)) 
  (h4 : 17 ∣ (90210 - n + 12)) 
  (h5 : 19 ∣ (90210 - n + 12)) : 
  n = 90198 :=
sorry

end least_number_to_subtract_l134_134161


namespace correct_operation_l134_134547

variable {a b : ℝ}

theorem correct_operation : (3 * a^2 * b - 3 * b * a^2 = 0) :=
by sorry

end correct_operation_l134_134547


namespace astronaut_total_days_l134_134449

-- Definitions of the regular and leap seasons.
def regular_season_days := 49
def leap_season_days := 51

-- Definition of the number of days in different types of years.
def days_in_regular_year := 2 * regular_season_days + 3 * leap_season_days
def days_in_first_3_years := 2 * regular_season_days + 3 * (leap_season_days + 1)
def days_in_years_7_to_9 := 2 * regular_season_days + 3 * (leap_season_days + 2)

-- Calculation for visits.
def first_visit := regular_season_days
def second_visit := 2 * regular_season_days + 3 * (leap_season_days + 1)
def third_visit := 3 * (2 * regular_season_days + 3 * (leap_season_days + 1))
def fourth_visit := 4 * days_in_regular_year + 3 * days_in_first_3_years + 3 * days_in_years_7_to_9

-- Total days spent.
def total_days := first_visit + second_visit + third_visit + fourth_visit

-- The proof statement.
theorem astronaut_total_days : total_days = 3578 :=
by
  -- We place a sorry here to skip the proof.
  sorry

end astronaut_total_days_l134_134449


namespace min_k_for_grid_sum_l134_134755

theorem min_k_for_grid_sum :
  ∀ (x : (Fin 100) → (Fin 25) → ℝ),
    (∀ i j, 0 ≤ x i j) →
    (∀ i, (∑ j, x i j) ≤ 1) →
    ∃ k, k = 97 ∧ ∀ i, i ≥ k → (∑ j, (rearrange x) i j) ≤ 1 := sorry

noncomputable def rearrange (x : (Fin 100) → (Fin 25) → ℝ) : (Fin 100) → (Fin 25) → ℝ :=
  sorry -- Function to rearrange each column in descending order

end min_k_for_grid_sum_l134_134755


namespace max_workers_l134_134980

-- Each worker produces 10 bricks a day and steals as many bricks per day as there are workers at the factory.
def worker_bricks_produced_per_day : ℕ := 10
def worker_bricks_stolen_per_day (n : ℕ) : ℕ := n

-- The factory must have at least 13 more bricks at the end of the day.
def factory_brick_surplus_requirement : ℕ := 13

-- Prove the maximum number of workers that can be hired so that the factory has at least 13 more bricks than at the beginning:
theorem max_workers
  (n : ℕ) -- Let \( n \) be the number of workers at the brick factory.
  (h : worker_bricks_produced_per_day * n - worker_bricks_stolen_per_day n + 13 ≥ factory_brick_surplus_requirement): 
  n = 8 := 
sorry

end max_workers_l134_134980


namespace sarah_shampoo_and_conditioner_usage_l134_134073

-- Condition Definitions
def shampoo_daily_oz := 1
def conditioner_daily_oz := shampoo_daily_oz / 2
def total_daily_usage := shampoo_daily_oz + conditioner_daily_oz

def days_in_two_weeks := 14

-- Assertion: Total volume used in two weeks.
theorem sarah_shampoo_and_conditioner_usage :
  (days_in_two_weeks * total_daily_usage) = 21 := by
  sorry

end sarah_shampoo_and_conditioner_usage_l134_134073


namespace total_books_borrowed_lunchtime_correct_l134_134349

def shelf_A_borrowed (X : ℕ) : Prop :=
  110 - X = 60 ∧ X = 50

def shelf_B_borrowed (Y : ℕ) : Prop :=
  150 - 50 + 20 - Y = 80 ∧ Y = 80

def shelf_C_borrowed (Z : ℕ) : Prop :=
  210 - 45 = 165 ∧ 165 - 130 = Z ∧ Z = 35

theorem total_books_borrowed_lunchtime_correct :
  ∃ (X Y Z : ℕ),
    shelf_A_borrowed X ∧
    shelf_B_borrowed Y ∧
    shelf_C_borrowed Z ∧
    X + Y + Z = 165 :=
by
  sorry

end total_books_borrowed_lunchtime_correct_l134_134349


namespace sufficient_not_necessary_condition_l134_134671

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ) (h₁ : -1 ≤ x ∧ x ≤ 2) : a > 4 → x^2 - a ≤ 0 :=
by
  sorry

end sufficient_not_necessary_condition_l134_134671


namespace total_wheels_in_garage_l134_134923

def total_wheels (bicycles tricycles unicycles : ℕ) (bicycle_wheels tricycle_wheels unicycle_wheels : ℕ) :=
  bicycles * bicycle_wheels + tricycles * tricycle_wheels + unicycles * unicycle_wheels

theorem total_wheels_in_garage :
  total_wheels 3 4 7 2 3 1 = 25 := by
  -- Calculation shows:
  -- (3 * 2) + (4 * 3) + (7 * 1) = 6 + 12 + 7 = 25
  sorry

end total_wheels_in_garage_l134_134923


namespace train_passing_platform_time_l134_134221

-- Conditions
variable (l t : ℝ) -- Length of the train and time to pass the pole
variable (v : ℝ) -- Velocity of the train
variable (n : ℝ) -- Multiple of t seconds to pass the platform
variable (d_platform : ℝ) -- Length of the platform

-- Theorem statement
theorem train_passing_platform_time (h1 : d_platform = 3 * l) (h2 : v = l / t) (h3 : n = (l + d_platform) / l) :
  n = 4 := by
  sorry

end train_passing_platform_time_l134_134221


namespace paper_plate_cup_cost_l134_134898

variables (P C : ℝ)

theorem paper_plate_cup_cost (h : 100 * P + 200 * C = 6) : 20 * P + 40 * C = 1.20 :=
by sorry

end paper_plate_cup_cost_l134_134898


namespace total_books_l134_134433

theorem total_books (D Loris Lamont : ℕ) 
  (h1 : Loris + 3 = Lamont)
  (h2 : Lamont = 2 * D)
  (h3 : D = 20) : D + Loris + Lamont = 97 := 
by 
  sorry

end total_books_l134_134433


namespace solve_for_y_l134_134856

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l134_134856


namespace area_of_rectangle_l134_134516

-- Definitions and conditions
variable (X Y Z : Type) [metric_space X] [metric_space Y] [metric_space Z]
variable (EFGH: Type) [metric_space EFGH]
variable (circle : X → Y → Z → ℝ) -- Defining the circle in terms of centers X, Y, Z and radius as real numbers
variable diameter : ℝ
variable height width : ℝ

-- Conditions
axiom circle_tangent_to_sides_of_rectangle : ∀ (r : ℝ), circle X Y Z = r / 2
axiom circle_diameter : diameter = 8
axiom circle_through_XY_and_Z : circle X Y Z = 8

-- Proof
theorem area_of_rectangle : ∃ (area : ℝ), area = 128 :=
by
  let height := 8
  let width := 16
  let area := height * width 
  exact ⟨area, rfl⟩

end area_of_rectangle_l134_134516


namespace midpoint_of_translated_segment_l134_134454

-- Define the endpoints of segment s₁
def endpoint1 := (3, -2)
def endpoint2 := (-9, 4)

-- Define the translation vector
def translation := (-3, -2)

-- Define the midpoint function for a segment given its endpoints
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Translate a point by a given translation vector
def translate (p t: ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + t.1, p.2 + t.2)

-- Prove the midpoint of the translated segment s₂ given conditions
theorem midpoint_of_translated_segment :
  translate (midpoint endpoint1 endpoint2) translation = (-6, -1) :=
by
  sorry

end midpoint_of_translated_segment_l134_134454


namespace calculate_expression_l134_134236

theorem calculate_expression : (3.75 - 1.267 + 0.48 = 2.963) :=
by
  sorry

end calculate_expression_l134_134236


namespace identify_quadratic_equation_l134_134534

/-- Proving which equation is a quadratic equation from given options -/
def is_quadratic_equation (eq : String) : Prop :=
  eq = "sqrt(x^2)=2" ∨ eq = "x^2 - x - 2" ∨ eq = "1/x^2 - 2=0" ∨ eq = "x^2=0"

theorem identify_quadratic_equation :
  ∀ (eq : String), is_quadratic_equation eq → eq = "x^2=0" :=
by
  intro eq h
  -- add proof steps here
  sorry

end identify_quadratic_equation_l134_134534


namespace min_value_2x_y_l134_134710

noncomputable def min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (heq : Real.log (x + 2 * y) = Real.log x + Real.log y) : ℝ :=
  2 * x + y

theorem min_value_2x_y : ∀ (x y : ℝ), 0 < x → 0 < y → Real.log (x + 2 * y) = Real.log x + Real.log y → 2 * x + y ≥ 9 :=
by
  intros x y hx hy heq
  sorry

end min_value_2x_y_l134_134710


namespace least_positive_linear_combination_24_18_l134_134560

theorem least_positive_linear_combination_24_18 (x y : ℤ) :
  ∃ (a : ℤ) (b : ℤ), 24 * a + 18 * b = 6 :=
by
  use 1
  use -1
  sorry

end least_positive_linear_combination_24_18_l134_134560


namespace solve_for_y_l134_134857

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 :=
by sorry

end solve_for_y_l134_134857


namespace range_of_m_l134_134670

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 4^x - m * 2^x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l134_134670


namespace total_wheels_in_garage_l134_134924

theorem total_wheels_in_garage : 
  let bicycles := 3
  let tricycles := 4
  let unicycles := 7
  let wheels_per_bicycle := 2
  let wheels_per_tricycle := 3
  let wheels_per_unicycle := 1
  in (bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle) = 25 :=
by
  let bicycles := 3
  let tricycles := 4
  let unicycles := 7
  let wheels_per_bicycle := 2
  let wheels_per_tricycle := 3
  let wheels_per_unicycle := 1
  have h_bicycles := bicycles * wheels_per_bicycle
  have h_tricycles := tricycles * wheels_per_tricycle
  have h_unicycles := unicycles * wheels_per_unicycle
  show (h_bicycles + h_tricycles + h_unicycles) = 25
  sorry

end total_wheels_in_garage_l134_134924


namespace H_function_is_f_x_abs_x_l134_134368

-- Definition: A function f is odd if ∀ x ∈ ℝ, f(-x) = -f(x)
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition: A function f is strictly increasing if ∀ x1, x2 ∈ ℝ, x1 < x2 implies f(x1) < f(x2)
def is_strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

-- Define the function f(x) = x * |x|
def f (x : ℝ) : ℝ := x * abs x

-- The main theorem which states that f(x) = x * |x| is an "H function"
theorem H_function_is_f_x_abs_x : is_odd f ∧ is_strictly_increasing f :=
  sorry

end H_function_is_f_x_abs_x_l134_134368


namespace probability_of_B_l134_134867

variable (P : (Set → ℝ))
variable (A B : Set)

-- Given conditions
axiom P_A_eq : P A = 0.01
axiom P_B_given_A_eq : P (B ∩ A) / P A = 0.99
axiom P_B_given_not_A_eq : P (B ∩ -A) / P (-A) = 0.1
axiom P_not_A_eq : P (-A) = 0.99

-- Prove P(B) = 0.1089
theorem probability_of_B : P B = 0.1089 :=
by 
  sorry

end probability_of_B_l134_134867


namespace fruit_vendor_total_l134_134446

theorem fruit_vendor_total (lemons_dozen avocados_dozen : ℝ) (dozen_size : ℝ) 
  (lemons : ℝ) (avocados : ℝ) (total_fruits : ℝ) 
  (h1 : lemons_dozen = 2.5) (h2 : avocados_dozen = 5) 
  (h3 : dozen_size = 12) (h4 : lemons = lemons_dozen * dozen_size) 
  (h5 : avocados = avocados_dozen * dozen_size) 
  (h6 : total_fruits = lemons + avocados) : 
  total_fruits = 90 := 
sorry

end fruit_vendor_total_l134_134446


namespace exterior_angle_value_l134_134394

theorem exterior_angle_value
  (P Q R T : Type)
  (y : ℝ)
  (PQR_straight : line P Q R)
  (T_above_PQ : above_point T P Q)
  (angle_PQT : ∠ PQ T = 150)
  (angle_QRT : ∠ QR T = 58) :
  y = 92 :=
by
  sorry

end exterior_angle_value_l134_134394


namespace constant_expression_l134_134064

theorem constant_expression (n : ℕ) (h_n : 0 < n) :
  ( ∑ k in finset.range (n^2 + 1), real.sqrt (n + real.sqrt k) ) /
  ( ∑ k in finset.range (n^2 + 1), real.sqrt (n - real.sqrt k) ) = 1 + real.sqrt 2 :=
sorry

end constant_expression_l134_134064


namespace train_passes_jogger_in_time_l134_134205

def jogger_speed_kmh : ℝ := 8
def train_speed_kmh : ℝ := 60
def initial_distance_m : ℝ := 360
def train_length_m : ℝ := 200

noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
noncomputable def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_m : ℝ := initial_distance_m + train_length_m
noncomputable def passing_time_s : ℝ := total_distance_m / relative_speed_ms

theorem train_passes_jogger_in_time :
  passing_time_s = 38.75 := by
  sorry

end train_passes_jogger_in_time_l134_134205


namespace compare_areas_l134_134417

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def area_triangle1 : ℝ :=
  heron_area 30 30 45

noncomputable def area_triangle2 : ℝ :=
  heron_area 30 30 55

theorem compare_areas : area_triangle1 > area_triangle2 := by
  -- The proof will be provided here
  sorry

end compare_areas_l134_134417


namespace lattice_points_on_hyperbola_l134_134735

theorem lattice_points_on_hyperbola : ∃ (n : ℕ), n = 70 ∧ ∀ (x y : ℤ), x^2 - y^2 = 2500^2 → n = 70 :=
by
  use 70
  intros x y h
  sorry

end lattice_points_on_hyperbola_l134_134735


namespace find_primes_l134_134183

theorem find_primes (A B C : ℕ) (hA : A < 20) (hB : B < 20) (hC : C < 20)
  (hA_prime : Prime A) (hB_prime : Prime B) (hC_prime : Prime C)
  (h_sum : A + B + C = 30) : 
  (A = 2 ∧ B = 11 ∧ C = 17) ∨ (A = 2 ∧ B = 17 ∧ C = 11) ∨ 
  (A = 11 ∧ B = 2 ∧ C = 17) ∨ (A = 11 ∧ B = 17 ∧ C = 2) ∨ 
  (A = 17 ∧ B = 2 ∧ C = 11) ∨ (A = 17 ∧ B = 11 ∧ C = 2) :=
sorry

end find_primes_l134_134183


namespace find_k_l134_134747

theorem find_k (α : ℝ) (hα : α ≠ 0) (h_root1 : α + (-1 / α) = -10)
  (h_root2 : α * (-1 / α) = -1) :
  let k := α * (-1 / α) in k = -1 :=
by
  have : α * (-1 / α) = -1 :=
    by sorry
  exact this

end find_k_l134_134747


namespace ratio_w_to_y_l134_134724

variable (w x y z : ℚ)
variable (h1 : w / x = 5 / 4)
variable (h2 : y / z = 5 / 3)
variable (h3 : z / x = 1 / 5)

theorem ratio_w_to_y : w / y = 15 / 4 := sorry

end ratio_w_to_y_l134_134724


namespace coefficient_x_squared_l134_134471

theorem coefficient_x_squared : 
  let expansion := (x + 2) ^ 6 in
  binomial_theorem (expansion) ∃ (coefficient : ℕ), coefficient = 240 -> coefficient of x^2 in expansion.

end coefficient_x_squared_l134_134471


namespace fruit_vendor_total_l134_134445

theorem fruit_vendor_total (lemons_dozen avocados_dozen : ℝ) (dozen_size : ℝ) 
  (lemons : ℝ) (avocados : ℝ) (total_fruits : ℝ) 
  (h1 : lemons_dozen = 2.5) (h2 : avocados_dozen = 5) 
  (h3 : dozen_size = 12) (h4 : lemons = lemons_dozen * dozen_size) 
  (h5 : avocados = avocados_dozen * dozen_size) 
  (h6 : total_fruits = lemons + avocados) : 
  total_fruits = 90 := 
sorry

end fruit_vendor_total_l134_134445


namespace part1_solution_part2_solution_l134_134719

section Part1

variable (a : ℝ)

def f (x : ℝ) : ℝ := 4 ^ x - 2 ^ (x + 1) + a

theorem part1_solution (x : ℝ) : f a x < a ↔ x < 1 :=
by sorry

end Part1

section Part2

variable (a : ℝ)

def f (x : ℝ) : ℝ := 4 ^ x - 2 ^ (x + 1) + a

def g (x : ℝ) : ℝ := f a x * (f a (x + 1) - 2 ^ (2 * x + 1)) + a ^ 2

theorem part2_solution : g a a = 2 * a ↔ a = 0 ∨ a = 2 :=
by sorry

end Part2

end part1_solution_part2_solution_l134_134719


namespace current_algae_plants_l134_134438

def original_algae_plants : ℕ := 809
def additional_algae_plants : ℕ := 2454

theorem current_algae_plants :
  original_algae_plants + additional_algae_plants = 3263 := by
  sorry

end current_algae_plants_l134_134438


namespace max_angle_MPN_is_pi_over_2_l134_134252

open Real

noncomputable def max_angle_MPN (θ : ℝ) (P : ℝ × ℝ) (hP : (P.1 - cos θ)^2 + (P.2 - sin θ)^2 = 1/25) : ℝ :=
  sorry

theorem max_angle_MPN_is_pi_over_2 (θ : ℝ) (P : ℝ × ℝ) (hP : (P.1 - cos θ)^2 + (P.2 - sin θ)^2 = 1/25) : 
  max_angle_MPN θ P hP = π / 2 :=
sorry

end max_angle_MPN_is_pi_over_2_l134_134252


namespace range_values_of_a_minimum_value_of_M_l134_134343

-- Define the functions f, F, and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x - log x
def F (a : ℝ) (x : ℝ) : ℝ := exp x + a * x
def g (a : ℝ) (x : ℝ) : ℝ := x * exp (a * x - 1) - 2 * a * x + f a x

-- Define the condition on the parameter 'a' and the domain of 'x'
variable (a : ℝ) (x : ℝ)
variable ha : a < 0
variable hx : 0 < x

-- Prove the first part of the problem
theorem range_values_of_a {x : ℝ} (hx : 0 < x ∧ x < log 3) : 
  (∀ (a : ℝ), (a < 0 → range_values (a ≤ -3))) :=
sorry

-- Prove the second part of the problem
theorem minimum_value_of_M {a : ℝ} (ha : a ∈ Iic (-1 / exp 2)) :
  ∃ M : ℝ, (∀ x > 0, g a x ≥ M) ∧ M = 0 :=
sorry

end range_values_of_a_minimum_value_of_M_l134_134343


namespace fastest_route_l134_134086

/-- Define the conditions for all routes -/
structure Route where
  total_distance : ℕ
  average_speed : ℕ
  traffic_delay : ℕ
  rest_stops : ℕ
  rest_duration : ℕ

/-- Define the total time calculation for a route -/
def total_time (route : Route) : ℚ :=
  let driving_time : ℚ := route.total_distance / route.average_speed
  let rest_time : ℚ := route.rest_stops * route.rest_duration / 60
  driving_time + route.traffic_delay + rest_time

/-- Route A details -/
def RouteA : Route := 
  { total_distance := 1500, average_speed := 75, traffic_delay := 2, rest_stops := 3, rest_duration := 30 }

/-- Route B details -/
def RouteB : Route := 
  { total_distance := 1300, average_speed := 70, traffic_delay := 0, rest_stops := 2, rest_duration := 45 }

/-- Route C details -/
def RouteC : Route := 
  { total_distance := 1800, average_speed := 80, traffic_delay := 2.5, rest_stops := 4, rest_duration := 20 }

/-- Route D details -/
def RouteD : Route := 
  { total_distance := 750, average_speed := 25, traffic_delay := 0, rest_stops := 1, rest_duration := 60 }

/-- Prove that Route B has the least total time -/
theorem fastest_route : total_time RouteB < total_time RouteA ∧ total_time RouteB < total_time RouteC ∧ total_time RouteB < total_time RouteD := by
  sorry

end fastest_route_l134_134086


namespace committee_count_l134_134763

theorem committee_count :
  let total_owners := 30
  let not_willing := 3
  let eligible_owners := total_owners - not_willing
  let committee_size := 5
  eligible_owners.choose committee_size = 65780 := by
  let total_owners := 30
  let not_willing := 3
  let eligible_owners := total_owners - not_willing
  let committee_size := 5
  have lean_theorem : eligible_owners.choose committee_size = 65780 := sorry
  exact lean_theorem

end committee_count_l134_134763


namespace largest_possible_median_l134_134311

theorem largest_possible_median
  (l : List ℕ)
  (h_len : l.length = 12)
  (h_positive : ∀ n ∈ l, 0 < n)
  (h_known : {5, 9, 3, 7, 10, 6} ⊆ l.toFinset) :
  median l = 6.5 :=
sorry

end largest_possible_median_l134_134311


namespace jump_difference_l134_134106

-- Definitions based on conditions
def grasshopper_jump : ℕ := 13
def frog_jump : ℕ := 11

-- Proof statement
theorem jump_difference : grasshopper_jump - frog_jump = 2 := by
  sorry

end jump_difference_l134_134106


namespace sarah_total_volume_in_two_weeks_l134_134070

def shampoo_daily : ℝ := 1

def conditioner_daily : ℝ := 1 / 2 * shampoo_daily

def days : ℕ := 14

def total_volume : ℝ := (shampoo_daily * days) + (conditioner_daily * days)

theorem sarah_total_volume_in_two_weeks : total_volume = 21 := by
  sorry

end sarah_total_volume_in_two_weeks_l134_134070


namespace total_wheels_in_garage_l134_134922

def total_wheels (bicycles tricycles unicycles : ℕ) (bicycle_wheels tricycle_wheels unicycle_wheels : ℕ) :=
  bicycles * bicycle_wheels + tricycles * tricycle_wheels + unicycles * unicycle_wheels

theorem total_wheels_in_garage :
  total_wheels 3 4 7 2 3 1 = 25 := by
  -- Calculation shows:
  -- (3 * 2) + (4 * 3) + (7 * 1) = 6 + 12 + 7 = 25
  sorry

end total_wheels_in_garage_l134_134922


namespace convert_to_cylindrical_l134_134628

theorem convert_to_cylindrical (x y z : ℝ) (r θ : ℝ) 
  (h₀ : x = 3) 
  (h₁ : y = -3 * Real.sqrt 3) 
  (h₂ : z = 2) 
  (h₃ : r > 0) 
  (h₄ : 0 ≤ θ) 
  (h₅ : θ < 2 * Real.pi) 
  (h₆ : r = Real.sqrt (x^2 + y^2)) 
  (h₇ : x = r * Real.cos θ) 
  (h₈ : y = r * Real.sin θ) : 
  (r, θ, z) = (6, 5 * Real.pi / 3, 2) :=
by
  -- Proof goes here
  sorry

end convert_to_cylindrical_l134_134628


namespace range_of_a_l134_134839

theorem range_of_a (a : ℝ) : (a < 0 → (∀ x : ℝ, 3 * a < x ∧ x < a → x^2 - 4 * a * x + 3 * a^2 < 0)) ∧ 
                              (∀ x : ℝ, (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0) ↔ (x < -4 ∨ x ≥ -2)) ∧ 
                              ((¬(∀ x : ℝ, 3 * a < x ∧ x < a → x^2 - 4 * a * x + 3 * a^2 < 0)) 
                                → (¬(∀ x : ℝ, (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0))))
                            → (a ≤ -4 ∨ (a < 0 ∧ 3 * a >= -2)) :=
by
  intros h
  sorry

end range_of_a_l134_134839


namespace number_of_yellow_parrots_l134_134442

variable (total_parrots red_fraction : ℕ)
variable (total_parrots = 119) (red_fraction = 5/7)

theorem number_of_yellow_parrots : 
  let yellow_fraction := 1 - red_fraction in
  let yellow_parrots := yellow_fraction * total_parrots in
  yellow_parrots = 34 :=
by
  sorry

end number_of_yellow_parrots_l134_134442


namespace possible_values_2a_b_l134_134700

theorem possible_values_2a_b (a b x y z : ℕ) (h1: a^x = 1994^z) (h2: b^y = 1994^z) (h3: 1/x + 1/y = 1/z) : 
  (2 * a + b = 1001) ∨ (2 * a + b = 1996) :=
by
  sorry

end possible_values_2a_b_l134_134700


namespace compare_abc_l134_134423

noncomputable def a := (0.3 : ℝ)^0.4
noncomputable def b := (0.4 : ℝ)^0.3
noncomputable def c := Real.log 0.3 / Real.log 0.4

theorem compare_abc : c < a ∧ a < b := 
by
  -- Prove the statement using the definitions and properties described.
  sorry

end compare_abc_l134_134423


namespace limit_tangent_eq_inv_pi_l134_134981

open Real

theorem limit_tangent_eq_inv_pi :
  tendsto (fun x => (2 * x) / tan(2 * π * (x + 1 / 2))) (nhds 0) (𝓝 (1 / π)) :=
sorry

end limit_tangent_eq_inv_pi_l134_134981


namespace playground_area_l134_134489

theorem playground_area :
  ∃ (l w : ℝ), 2 * l + 2 * w = 84 ∧ l = 3 * w ∧ l * w = 330.75 :=
by
  sorry

end playground_area_l134_134489


namespace sandwiches_count_l134_134407

theorem sandwiches_count (s c : ℕ) (h1 : s + c = 7)
  (h2 : 60 * s + 90 * c ∈ set.of_list [500, 600, 700]) :
  s = 11 :=
sorry

end sandwiches_count_l134_134407


namespace lattice_points_on_hyperbola_l134_134736

theorem lattice_points_on_hyperbola : ∃ (n : ℕ), n = 70 ∧ ∀ (x y : ℤ), x^2 - y^2 = 2500^2 → n = 70 :=
by
  use 70
  intros x y h
  sorry

end lattice_points_on_hyperbola_l134_134736


namespace sqrt_product_equals_l134_134620

noncomputable def sqrt128 : ℝ := Real.sqrt 128
noncomputable def sqrt50 : ℝ := Real.sqrt 50
noncomputable def sqrt18 : ℝ := Real.sqrt 18

theorem sqrt_product_equals : sqrt128 * sqrt50 * sqrt18 = 240 * Real.sqrt 2 := 
by
  sorry

end sqrt_product_equals_l134_134620


namespace find_value_of_x_l134_134006

-- Definitions used in conditions
variable (O B C A : Point) -- points on the circle
variable (OB OC OA : ℝ) -- Radii of the circle
variable (OB_eq_OC : OB = OC)
variable (angle_BCO : ℝ) -- Given angle ∠BCO
variable (angle_DAC : ℝ) -- Given angle ∠DAC
variable (right_angled_ACD : angle_DAC + 90 = 157) -- Triangle ACD is right-angled & 180 - 23 = 157

variable (x : ℝ) -- angle x

-- Setting specific values for given angles in degrees
noncomputable def angle_BCO_value : angle_BCO = 32
noncomputable def angle_DAC_value : angle_DAC = 67

-- What we want to prove
theorem find_value_of_x : x = 9 :=
sorry

end find_value_of_x_l134_134006


namespace magnitude_of_sum_of_squares_of_root_conjugates_l134_134370

theorem magnitude_of_sum_of_squares_of_root_conjugates 
  (z : ℂ) (h : z^2 + 2 * z + 2 = 0) : 
  (|z^2 + (conj z)^2| = 0) :=
sorry

end magnitude_of_sum_of_squares_of_root_conjugates_l134_134370


namespace root_exists_l134_134702

variable {R : Type} [LinearOrderedField R]
variables (a b c : R)

def f (x : R) : R := a * x^2 + b * x + c

theorem root_exists (h : f a b c ((a - b - c) / (2 * a)) = 0) : f a b c (-1) = 0 ∨ f a b c 1 = 0 := by
  sorry

end root_exists_l134_134702


namespace probability_of_consonant_initials_l134_134754

theorem probability_of_consonant_initials (students : Finset String) :
  students.card = 25 →
  (∀ s ∈ students, s.length = 2 ∧ s[0] = s[1] ∧ s[0] ≠ 'Y') →
  ∀ (vowel_initials : Finset String), vowel_initials = { "AA", "EE", "II", "OO", "UU" } →
  (students ∩ vowel_initials).card = 5 →
  (students.filter (λ s, s[0] ∉ { 'A', 'E', 'I', 'O', 'U' })).card = 20 →
  (students.filter (λ s, s[0] ∉ { 'A', 'E', 'I', 'O', 'U' })).card / students.card = 4 / 5 :=
by
  intros h_card h_initials h_vowel_initials h_vowel_card h_consonant_card
  have h_consonant_frac := h_consonant_card / h_card
  norm_num at h_consonant_frac
  exact h_consonant_frac


end probability_of_consonant_initials_l134_134754


namespace solve_fraction_equation_l134_134860

theorem solve_fraction_equation (x : ℝ) (h1 : 3*x - 2 ≠ 0) (h2 : x + 3 ≠ 0) :
  (8*x + 3) / (3*x^2 + 8*x - 6) = (3*x) / (3*x - 2) ↔ x = 1 ∨ x = -1 :=
by
  have h3 : 3*x^2 + 8*x - 6 = (3*x - 2)*(x + 3), sorry
  sorry

end solve_fraction_equation_l134_134860


namespace perimeter_of_fenced_square_field_l134_134935

-- Definitions for conditions
def num_posts : ℕ := 36
def spacing_between_posts : ℝ := 6 -- in feet
def post_width : ℝ := 1 / 2 -- 6 inches in feet

-- The statement to be proven
theorem perimeter_of_fenced_square_field :
  (4 * ((9 * spacing_between_posts) + (10 * post_width))) = 236 :=
by
  sorry

end perimeter_of_fenced_square_field_l134_134935


namespace area_of_quadrilateral_ABFG_l134_134210

/-- 
Given conditions:
1. Rectangle with dimensions AC = 40 and AE = 24.
2. Points B and F are midpoints of sides AC and AE, respectively.
3. G is the midpoint of DE.
Prove that the area of quadrilateral ABFG is 600 square units.
-/
theorem area_of_quadrilateral_ABFG (AC AE : ℝ) (B F G : ℤ) 
  (hAC : AC = 40) (hAE : AE = 24) (hB : B = 1/2 * AC) (hF : F = 1/2 * AE) (hG : G = 1/2 * AE):
  area_of_ABFG = 600 :=
by
  sorry

end area_of_quadrilateral_ABFG_l134_134210


namespace figure_B_has_largest_shaded_area_l134_134937

def rectangle_area (length width : ℝ) : ℝ :=
  length * width

def circle_area (radius : ℝ) : ℝ :=
  π * radius^2

def shaded_area_A : ℝ :=
  let rect_area := rectangle_area 4 3
  let circ_area := circle_area 1.5
  rect_area - circ_area

def shaded_area_B : ℝ :=
  let rect_area := rectangle_area 4 3
  let circ_area := 2 * circle_area 1
  rect_area - circ_area

def shaded_area_C : ℝ :=
  let rect_area := rectangle_area (Real.sqrt 8) (Real.sqrt 2)
  let circ_area := circle_area (Real.sqrt 2 / 2)
  rect_area - circ_area

theorem figure_B_has_largest_shaded_area :
  shaded_area_B > shaded_area_A ∧ shaded_area_B > shaded_area_C :=
by {
  -- proof goes here
  sorry
}

end figure_B_has_largest_shaded_area_l134_134937


namespace initial_monkey_count_l134_134241

theorem initial_monkey_count (B : ℕ) (B' : ℕ) (total_animals : ℕ) (total_monkeys : ℕ) (M : ℕ) 
(h1 : B = 6) 
(h2 : B' = B - 2) 
(h3 : 0.6 * total_animals = total_monkeys) 
(h4 : total_animals = B' + total_monkeys) 
(h5 : M = total_monkeys) : 
M = 6 := 
sorry

end initial_monkey_count_l134_134241


namespace christmas_tree_bulbs_l134_134915

def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

def number_of_bulbs_on (N : Nat) : Nat :=
  (Finset.range (N+1)).filter isPerfectSquare |>.card

def probability_bulb_on (total_bulbs bulbs_on : Nat) : Float :=
  (bulbs_on.toFloat) / (total_bulbs.toFloat)

theorem christmas_tree_bulbs :
  let N := 100
  let bulbs_on := number_of_bulbs_on N
  probability_bulb_on N bulbs_on = 0.1 :=
by
  sorry

end christmas_tree_bulbs_l134_134915


namespace positive_difference_is_50_l134_134157

open Nat
open List

-- Define sum of squares of the first 6 positive integers
def sum_of_squares (n : ℕ) := (range (n + 1)).map (λ x => x * x).sum

-- Define sum of primes between 1 and n^2
def sum_of_primes (m : ℕ) := (range (m + 1)).filter Prime.prime.sum

theorem positive_difference_is_50 : 
  sum_of_squares 6 - sum_of_primes 49 = 50 :=
by
  -- Steps not required, directly sorries to state the theorem.
  sorry

end positive_difference_is_50_l134_134157


namespace smallest_t_for_circle_l134_134876

theorem smallest_t_for_circle (t : ℝ) : (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → 2 * sin θ = r) → 
  (∃ t : ℝ, ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ↔ 0 ≤ θ ∧ θ < π) :=
by
  sorry

end smallest_t_for_circle_l134_134876


namespace simplify_polynomial_l134_134459

/-- Simplification of the polynomial expression -/
theorem simplify_polynomial (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^2 + 15) - (x^6 + 4 * x^5 - 2 * x^3 + 20) = x^6 - x^5 + 2 * x^3 - 5 :=
by {
  sorry
}

end simplify_polynomial_l134_134459


namespace probability_bulb_on_l134_134910

def toggle_process_result : ℕ → Bool
| n := (count_divisors n) % 2 = 1

def count_divisors (n : ℕ) : ℕ :=
  (finset.Icc 1 n).count (λ d : ℕ, n % d = 0)

theorem probability_bulb_on :
  let total_bulbs := 100
  let remaining_bulbs := finset.range 101 |>.filter toggle_process_result |>.card
  remaining_bulbs / total_bulbs = 0.1 :=
sorry

end probability_bulb_on_l134_134910


namespace angle_x_is_58_l134_134021

-- Definitions and conditions based on the problem statement
variables {O A C D : Type} (x : ℝ)
variable [circular_center : O] -- O is the center of the circle
variable [radius_equal : AO = BO ∧ BO = CO] -- AO, BO, and CO are radii and hence equal
variable (angle_ACD_right : ∠ A C D = 90) -- ∠ACD is a right angle (subtended by diameter)
variable (angle_CDA : ∠ C D A = 42) -- ∠CDA is 42 degrees

-- Statement to be proven
theorem angle_x_is_58 (x : ℝ) : x = 58 := 
by sorry

end angle_x_is_58_l134_134021


namespace points_on_single_line_l134_134136

theorem points_on_single_line 
  (O : Point) 
  (A B C : Point) 
  (circle : Circle O) 
  (X P : Point) 
  (phi : Real)
  (triangle_inscribed : InscribedTriangle A B C circle)
  (X_condition1 : ∠ X A B = phi)
  (X_condition2 : ∠ X B C = phi)
  (P_condition1 : Perpendicular (PX) (OX))
  (P_condition2 : ∠ X O P = phi)
  (angle_orientation : SameOrientation (∠ X O P) (∠ X A B)) : 
  ∃ l : Line, ∀ P, (PX ⊥ OX) → ∠ X O P = phi → SameOrientation (∠ X O P) (∠ X A B) → P ∈ l :=
sorry

end points_on_single_line_l134_134136


namespace sin_minus_cos_value_l134_134698

-- Definition of the conditions
def alpha_in_interval (α : ℝ) : Prop := 0 < α ∧ α < π
def sin_plus_cos (α : ℝ) : Prop := sin α + cos α = sqrt 2 / 2

-- Statement of the problem
theorem sin_minus_cos_value (α : ℝ) (hα : alpha_in_interval α) (h : sin_plus_cos α) :
  sin α - cos α = sqrt 6 / 2 :=
sorry

end sin_minus_cos_value_l134_134698


namespace sequence_properties_l134_134946

-- Define the sequences x and y using the given recurrence relations.
noncomputable def x : ℤ → ℚ
noncomputable def y : ℤ → ℚ

-- State the conditions given in the problem.
axiom sequence_conditions (k : ℤ) :
  3 * x k + 2 * y k = x (k + 1) ∧ 4 * x k + 3 * y k = y (k + 1)

-- State the properties to be proven about the sequences.
theorem sequence_properties (k r : ℤ) :
  x (-k) = -x k ∧ y (-k) = y k ∧
  ∀ k r, (relationship_1008_property k r) :=
sorry

end sequence_properties_l134_134946


namespace find_largest_divisor_l134_134697

def f (n : ℕ) : ℕ := (2 * n + 7) * 3 ^ n + 9

theorem find_largest_divisor :
  ∃ m : ℕ, (∀ n : ℕ, f n % m = 0) ∧ m = 36 :=
sorry

end find_largest_divisor_l134_134697


namespace no_element_divides_other_l134_134568

noncomputable def M (M₀ : Set ℕ) (b : ℕ → ℕ) : ℕ → Set ℕ
| 0     => M₀
| (n+1) => {b n * m + 1 | m ∈ M n}

theorem no_element_divides_other (M₀ : Set ℕ) (hM₀ : M₀.Nonempty ∧ M₀.Finite):
  ∃ k, ∀ a b ∈ M M₀ (λ n, some (M M₀ n)) k, ¬ (a ≠ b ∧ a ∣ b) :=
sorry

end no_element_divides_other_l134_134568


namespace y_intercept_tangent_line_l134_134300

noncomputable def f (a : ℝ) (x : ℝ) := a * x - Real.log x

theorem y_intercept_tangent_line (a : ℝ) :
  let f' x := a - 1 / x in
  let tangent_eq := λ y x, y - a = (a - 1) * (x - 1) in
  tangent_eq y 0 → y = 1 := by
  sorry

end y_intercept_tangent_line_l134_134300


namespace sum_of_fractions_is_correct_l134_134611

noncomputable def sumFractions : ℝ :=
  let fractions := [2, 4, 6, 8, 10, 12, 14, 16, 18, 32].map (λ n, n / 10 : ℝ)
  fractions.sum

theorem sum_of_fractions_is_correct : sumFractions = 12.2 := by
  sorry

end sum_of_fractions_is_correct_l134_134611


namespace diagonal_length_of_square_l134_134392

-- Definitions and conditions as per the problem statement
variables (E F G H M N O : Type) 
variable [metric_space E] [metric_space F] [metric_space G] [metric_space H] 
variable [metric_space M] [metric_space N] [metric_space O]
variables (EFGH : set (metric_space))
variables (MF : segment) (FN : segment)
variable (s : ℝ)  -- Side length of the square EFGH

-- Conditions
hypothesis EFGH_square : is_square EFGH
hypothesis FMN_right_angle : ∠FMN = π / 2 
hypothesis FO_length : dist F O = 8
hypothesis MO_length : dist M O = 9

-- Hypothesis about positions of M and N on EH and EF respectively
hypothesis M_on_EH : M ∈ segment EH
hypothesis N_on_EF : N ∈ segment EF

-- Proof theorems
theorem diagonal_length_of_square (EFGH_square : is_square EFGH)
  (FMN_right_angle : ∠FMN = π / 2) 
  (FO_length : dist F O = 8) 
  (MO_length : dist M O = 9)
  (M_on_EH : M ∈ segment EH) 
  (N_on_EF : N ∈ segment EF) :
  dist E H = 17 * sqrt 2 :=
sorry

end diagonal_length_of_square_l134_134392


namespace complex_modulus_squared_complex_squared_neq_l134_134813
noncomputable def complex_number : Type := ℂ

theorem complex_modulus_squared (z : ℂ) : complex.abs(z^2) = complex.abs(z)^2 :=
begin
  sorry
end

theorem complex_squared_neq (z : ℂ) : complex.abs(z)^2 ≠ z^2 :=
begin
  sorry
end

end complex_modulus_squared_complex_squared_neq_l134_134813


namespace number_of_students_above_90_l134_134761

noncomputable def num_students_above_90 (students : ℕ) (mean : ℝ) (variance : ℝ) : ℕ :=
  let sigma : ℝ := Real.sqrt variance
  let prob_above_90 : ℝ := 0.5 * (1 - 0.9544)
  let expected_num : ℝ := students * prob_above_90
  Real.to_nat (Real.round expected_num)

theorem number_of_students_above_90 :
  num_students_above_90 1000 80 25 = 23 :=
sorry

end number_of_students_above_90_l134_134761


namespace chord_triangle_count_l134_134305

theorem chord_triangle_count (n : ℕ) (h : n ≥ 6) :
  let S := (Nat.choose n 3) + 4 * (Nat.choose n 4) + 5 * (Nat.choose n 5) + (Nat.choose n 6) in
  ∃ (points : finset (ℤ × ℤ)), 
    points.card = n ∧
    (∀ (p1 p2 p3 : (ℤ × ℤ)), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points → (chord_intersection p1 p2 p3)) ∧
    (∀ (c1 c2 c3 : chord), ¬three_intersect c1 c2 c3) →
    (∃ (triangles : finset (finset (ℤ × ℤ))), triangles.card = S) := sorry

end chord_triangle_count_l134_134305


namespace square_radius_of_circumscribed_circle_l134_134578

-- Definitions as per conditions
variables {AP PB CQ QD AB CD r : Real}
def P := Point -- Tangent point on AB
def Q := Point -- Tangent point on CD
def A := Point -- Vertex of quadrilateral
def B := Point -- Vertex of quadrilateral
def C := Point -- Vertex of quadrilateral
def D := Point -- Vertex of quadrilateral

-- Conditions given in the problem
def circumscribed_around_ABCD : Prop := 
  (P ∈ Circle.r) ∧ (Q ∈ Circle.r) ∧
  (P ∈ Line(A, B)) ∧ (Q ∈ Line(C, D)) ∧
  (AP = 15) ∧ (PB = 31) ∧
  (CQ = 43) ∧ (QD = 17) ∧
  (AB = AP + PB) ∧ (CD = CQ + QD)

-- Statement to prove
theorem square_radius_of_circumscribed_circle (h: circumscribed_around_ABCD) : 
  r^2 = 127.3 := 
  sorry

end square_radius_of_circumscribed_circle_l134_134578


namespace correct_option_is_B_l134_134970

-- Define the operations as hypotheses
def option_A (a : ℤ) : Prop := (a^2 + a^3 = a^5)
def option_B (a : ℤ) : Prop := ((a^2)^3 = a^6)
def option_C (a : ℤ) : Prop := (a^2 * a^3 = a^6)
def option_D (a : ℤ) : Prop := (6 * a^6 - 2 * a^3 = 3 * a^3)

-- Prove that option B is correct
theorem correct_option_is_B (a : ℤ) : option_B a :=
by
  unfold option_B
  sorry

end correct_option_is_B_l134_134970


namespace candidate2_is_correct_l134_134226
-- Importing the necessary libraries from Mathlib

-- Question and conditions as Lean definitions and assumptions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ I → y ∈ I → x < y → f x > f y

def candidate1 (x : ℝ) : ℝ := real.sqrt x
def candidate2 (x : ℝ) : ℝ := -x^3
def candidate3 (x : ℝ) : ℝ := log (real.exp 1 / 2) x
def candidate4 (x : ℝ) : ℝ := x + 1/x

-- Mathematical problem statement in Lean 4 (without proof)
theorem candidate2_is_correct :
  is_odd candidate2 ∧ is_monotonically_decreasing candidate2 {x : ℝ | x > 0} :=
sorry

end candidate2_is_correct_l134_134226


namespace cyclic_shifts_l134_134990

open List

-- Definitions for the fields and movements.
def Fields := Fin 12

-- We represent the initial state as four adjacent fields containing pieces.
structure State :=
  (pieces : List Fields)
  (adjacent : ∀ i : Fin 4, (1 + ((pieces.nth_le i sorry).val + 1)%12).val == (pieces.nth_le ((i + 1) % 4) sorry).val)

-- Movement rule: a piece can move over four other fields to a fifth field if it is free.
def move (s : State) (i : Fin 4) (direction : Bool) : State :=
  let move_idx := if direction then (s.pieces.nth_le i sorry).val + 5 else (s.pieces.nth_le i sorry).val + 7
  { pieces := s.pieces.map (λ x, if x == s.pieces.nth_le i sorry then Fin.mk move_idx sorry else x), adjacent := sorry }

-- The goal is to prove that the pieces can only be rearranged into cyclic shifts of their initial positions.
theorem cyclic_shifts (s : State) : ∃ s', (∀ i j : Fin 4, (1 + ((s'.pieces.nth_le i sorry).val + 1)%12).val == (s'.pieces.nth_le ((i + 1) % 4) sorry).val) ∧
  ((s'.pieces == s.pieces) ∨ (s'.pieces == s.pieces.rotate 1) ∨ (s'.pieces == s.pieces.rotate 2) ∨ (s'.pieces == s.pieces.rotate 3)) :=
sorry

end cyclic_shifts_l134_134990


namespace blocks_fit_into_box_l134_134200

theorem blocks_fit_into_box :
  let box_height := 8
  let box_width := 10
  let box_length := 12
  let block_height := 3
  let block_width := 2
  let block_length := 4
  let box_volume := box_height * box_width * box_length
  let block_volume := block_height * block_width * block_length
  let num_blocks := box_volume / block_volume
  num_blocks = 40 :=
by
  sorry

end blocks_fit_into_box_l134_134200


namespace simplified_evaluation_l134_134456

noncomputable def poly1 := 2 * x^5 - 3 * x^4 + 5 * x^3 - 9 * x^2 + 8 * x - 15
noncomputable def poly2 := 5 * x^4 - 2 * x^3 + 3 * x^2 - 4 * x + 9
noncomputable def simplified_poly := 2 * x^5 + 2 * x^4 + 3 * x^3 - 6 * x^2 + 4 * x - 6
noncomputable def eval_poly_at_2 := 98

theorem simplified_evaluation :
  (poly1 + poly2) = simplified_poly ∧ simplified_poly.eval 2 = eval_poly_at_2 := by
  sorry

end simplified_evaluation_l134_134456


namespace convert_to_cylindrical_l134_134629

theorem convert_to_cylindrical (x y z : ℝ) (r θ : ℝ) 
  (h₀ : x = 3) 
  (h₁ : y = -3 * Real.sqrt 3) 
  (h₂ : z = 2) 
  (h₃ : r > 0) 
  (h₄ : 0 ≤ θ) 
  (h₅ : θ < 2 * Real.pi) 
  (h₆ : r = Real.sqrt (x^2 + y^2)) 
  (h₇ : x = r * Real.cos θ) 
  (h₈ : y = r * Real.sin θ) : 
  (r, θ, z) = (6, 5 * Real.pi / 3, 2) :=
by
  -- Proof goes here
  sorry

end convert_to_cylindrical_l134_134629


namespace derivative_y_l134_134285

noncomputable def y (x : ℝ) : ℝ := 
  Real.arcsin (1 / (2 * x + 3)) + 2 * Real.sqrt (x^2 + 3 * x + 2)

variable {x : ℝ}

theorem derivative_y :
  2 * x + 3 > 0 → 
  HasDerivAt y (4 * Real.sqrt (x^2 + 3 * x + 2) / (2 * x + 3)) x :=
by 
  sorry

end derivative_y_l134_134285


namespace rhombus_of_equal_inscribed_radii_l134_134872

theorem rhombus_of_equal_inscribed_radii
  (ABCD : ConvexQuadrilateral)
  (O : Point)
  (h_intersect : ABCD.diagonals_intersect_at O)
  (h_equal_radii : ABCD.inscribed_radii_equal) : ABCD.is_rhombus := 
sorry

end rhombus_of_equal_inscribed_radii_l134_134872


namespace negation_of_universal_l134_134488

theorem negation_of_universal :
  ¬ (∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
by
  sorry

end negation_of_universal_l134_134488


namespace angle_x_degrees_l134_134033

theorem angle_x_degrees 
  (O : Point)
  (A B C D : Point)
  (hC : Circle O)
  (hOB : O ∈ Segment O B)
  (hOC : O ∈ Segment O C)
  (hBCO : ∠ B C O = 32)
  (hACD : ∠ A C D = 90)
  (hCAD : ∠ C A D = 67) :
  ∠ B C A = 9 := 
  sorry

end angle_x_degrees_l134_134033


namespace simplify_sqrt_expression_l134_134268

-- Define the mathematical expression
def expression (x : ℝ) : ℝ :=
  sqrt (4 + ((x^3 - 2) / x)^2)

-- Define the expected simplified form
def simplified_form (x : ℝ) : ℝ :=
  (sqrt (x^6 - 4*x^3 + 4*x^2 + 4)) / x

-- The statement to be proved
theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  expression x = simplified_form x :=
by 
  -- Proof is required here, but we use sorry to skip it in this task
  sorry

end simplify_sqrt_expression_l134_134268


namespace line_eq_and_min_dist_l134_134709

noncomputable def polar_to_rectangular (ρ θ : ℝ) := (ρ * cos θ, ρ * sin θ)

theorem line_eq_and_min_dist :
  (∀ (ρ θ : ℝ), ρ * sin (θ - π / 4) = 3 * sqrt 2 → let (x, y) := polar_to_rectangular ρ θ in x - y + 6 = 0) ∧
  (∀ (α : ℝ), let P := (4 * cos α, 3 * sin α) in
    P ∈ {p : ℝ × ℝ | p.1^2 / 16 + p.2^2 / 9 = 1} →
    ∃ d : ℝ, d = sqrt 2 / 2 ∧ ∀ (l : ℝ × ℝ → Prop), l = λ P, P.1 - P.2 + 6 = 0 →
      ∀ (dist : ℝ), dist = |P.1 - P.2 + 6| / sqrt 2 → dist = d) := by
  sorry

end line_eq_and_min_dist_l134_134709


namespace bulbs_probability_l134_134902

/-
There are 100 light bulbs initially all turned on.
Every second bulb is toggled after one second.
Every third bulb is toggled after two seconds, and so on.
This process continues up to 100 seconds.
We aim to prove the probability that a randomly selected bulb is on after 100 seconds is 0.1.
-/

theorem bulbs_probability : 
  let bulbs := {1..100}
  let perfect_squares := {n ∈ bulbs | ∃ k, n = k * k}
  let total_bulbs := 100
  let bulbs_on := Set.card perfect_squares
  (bulbs_on : ℕ) / total_bulbs = 0.1 :=
by
  -- We use the solution steps to take perfect squares directly
  have h : Set.card perfect_squares = 10 := sorry
  have t : total_bulbs = 100 := rfl
  rw [h]
  norm_num


end bulbs_probability_l134_134902


namespace fruit_vendor_sold_fruits_l134_134444

def total_dozen_fruits_sold (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ) : ℝ :=
  (lemons_dozen * dozen) + (avocados_dozen * dozen)

theorem fruit_vendor_sold_fruits (hl : ∀ (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ), lemons_dozen = 2.5 ∧ avocados_dozen = 5 ∧ dozen = 12) :
  total_dozen_fruits_sold 2.5 5 12 = 90 :=
by
  sorry

end fruit_vendor_sold_fruits_l134_134444


namespace wheel_distance_covered_l134_134579

theorem wheel_distance_covered (radius : ℝ) (revolutions : ℝ) (h_radius : radius = 1.75) (h_revolutions : revolutions = 1000.4024994347707) : 
  (2 * Real.pi * radius * revolutions) ≈ 11004.427494347707 :=
by
  rw [h_radius, h_revolutions]
  sorry

end wheel_distance_covered_l134_134579


namespace sum_of_possible_values_x_l134_134533

def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)

def mode (l : List ℝ) : ℝ := 
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) (l.headI)

def median (l : List ℝ) : ℝ := 
  match l.length with
  | 0     => 0
  | n + 1 => let sorted_l := l.qsort (≤)
             if (n + 1) % 2 = 0 then 
               (sorted_l.get! (n / 2) + sorted_l.get! (n / 2 + 1)) / 2
             else 
               sorted_l.get! ((n + 1) / 2)

noncomputable def possible_values_of_x : Set ℝ := 
  { x | 
    let l := [12, 3, 6, 3, 7, 3, 3, x], 
        mean_val := mean l, 
        mode_val := mode l, 
        median_val := median l in 
    (mode_val < median_val ∧ median_val < mean_val ∨
     mode_val > median_val ∧ median_val > mean_val) ∧
    (mean_val = median_val ∨ median_val = mode_val ∨ mean_val = mode_val) = false }

theorem sum_of_possible_values_x : 
  Set.sum possible_values_of_x = 74.83 :=
sorry

end sum_of_possible_values_x_l134_134533


namespace pie_eating_contest_l134_134945

def a : ℚ := 7 / 8
def b : ℚ := 5 / 6
def difference : ℚ := 1 / 24

theorem pie_eating_contest : a - b = difference := 
sorry

end pie_eating_contest_l134_134945


namespace bulbs_probability_l134_134903

/-
There are 100 light bulbs initially all turned on.
Every second bulb is toggled after one second.
Every third bulb is toggled after two seconds, and so on.
This process continues up to 100 seconds.
We aim to prove the probability that a randomly selected bulb is on after 100 seconds is 0.1.
-/

theorem bulbs_probability : 
  let bulbs := {1..100}
  let perfect_squares := {n ∈ bulbs | ∃ k, n = k * k}
  let total_bulbs := 100
  let bulbs_on := Set.card perfect_squares
  (bulbs_on : ℕ) / total_bulbs = 0.1 :=
by
  -- We use the solution steps to take perfect squares directly
  have h : Set.card perfect_squares = 10 := sorry
  have t : total_bulbs = 100 := rfl
  rw [h]
  norm_num


end bulbs_probability_l134_134903


namespace average_price_of_pen_is_correct_l134_134572

-- Define conditions as variables
variable (price_of_pencil_before_tax : ℕ)
variable (number_of_pens : ℕ)
variable (number_of_pencils : ℕ)
variable (total_cost : ℕ)
variable (discount_on_pens : ℕ)
variable (tax_on_pencils : ℕ)

-- Assign given values to variables
def price_of_pencil_before_tax_val : ℕ := 2
def number_of_pens_val : ℕ := 30
def number_of_pencils_val : ℕ := 75
def total_cost_val : ℕ := 570
def discount_on_pens_val : ℕ := 10
def tax_on_pencils_val : ℕ := 5

-- Define the function to calculate the average price of a pen before discount
def average_price_of_pen_before_discount : ℕ :=
  let total_cost_of_pencils_before_tax := number_of_pencils * price_of_pencil_before_tax
  let total_cost_of_pencils_after_tax := total_cost_of_pencils_before_tax + (total_cost_of_pencils_before_tax * tax_on_pencils / 100)
  let total_cost_of_pens_after_discount := total_cost - total_cost_of_pencils_after_tax
  let total_cost_of_pens_before_discount := total_cost_of_pens_after_discount * 100 / (100 - discount_on_pens)
  total_cost_of_pens_before_discount / number_of_pens

-- Define the statement we need to prove
theorem average_price_of_pen_is_correct :
  average_price_of_pen_before_discount price_of_pencil_before_tax_val number_of_pens_val number_of_pencils_val total_cost_val discount_on_pens_val tax_on_pencils_val = 15.28 :=
  sorry

end average_price_of_pen_is_correct_l134_134572


namespace triangle_XYZ_l134_134402

theorem triangle_XYZ
  (X Y Z : Type)
  (angle_X : Degree)
  (YZ : ℝ)
  (tan_Z : ℝ)
  (cos_Y : ℝ)
  (XY : ℝ) :
  angle_X = 90 ∧ YZ = 20 ∧ tan_Z = 3 * cos_Y ∧ cos_Y = XY / 20 ∧ tan_Z = XY / (√(YZ^2 - XY^2)) →
    XY = 40 * sqrt 2 / 3 :=
begin
  sorry
end

end triangle_XYZ_l134_134402


namespace sarah_shampoo_and_conditioner_usage_l134_134071

-- Condition Definitions
def shampoo_daily_oz := 1
def conditioner_daily_oz := shampoo_daily_oz / 2
def total_daily_usage := shampoo_daily_oz + conditioner_daily_oz

def days_in_two_weeks := 14

-- Assertion: Total volume used in two weeks.
theorem sarah_shampoo_and_conditioner_usage :
  (days_in_two_weeks * total_daily_usage) = 21 := by
  sorry

end sarah_shampoo_and_conditioner_usage_l134_134071


namespace first_train_speed_l134_134197

noncomputable def speed_of_first_train (length_train1 : ℕ) (speed_train2 : ℕ) (length_train2 : ℕ) (time_cross : ℕ) : ℕ :=
  let relative_speed_m_s := (500 : ℕ) / time_cross
  let relative_speed_km_h := relative_speed_m_s * 18 / 5
  relative_speed_km_h - speed_train2

theorem first_train_speed :
  speed_of_first_train 270 80 230 9 = 920 := by
  sorry

end first_train_speed_l134_134197


namespace monotonically_decreasing_condition_l134_134696

variable (a : ℝ)

-- Assume initial conditions
axiom a_gt_zero : a > 0
axiom a_ne_one : a ≠ 1

-- The theorem statement 
theorem monotonically_decreasing_condition : 
  1 < a < 3 → 
  ∀ x ∈ Ioo 1 2, (λ x, log a (6 - a * x)) x ≤ (λ x, log a (6 - a * (x + 1))) x :=
sorry

end monotonically_decreasing_condition_l134_134696


namespace best_choice_to_calculate_89_8_sq_l134_134095

theorem best_choice_to_calculate_89_8_sq 
  (a b c d : ℚ) 
  (h1 : (89 + 0.8)^2 = a) 
  (h2 : (80 + 9.8)^2 = b) 
  (h3 : (90 - 0.2)^2 = c) 
  (h4 : (100 - 10.2)^2 = d) : 
  c = 89.8^2 := by
  sorry

end best_choice_to_calculate_89_8_sq_l134_134095


namespace angle_x_degrees_l134_134045

noncomputable def center_of_circle (O : Point) := true
noncomputable def radii (O B C A : Point) := (dist O B = dist O C) ∧ (dist O A = dist O C)
noncomputable def isosceles_triangle (O B C : Point) := (∠ B C O = 32) ∧ (∠ O B C = 32)
noncomputable def isosceles_base_angle (O A C : Point) := ∠ O A C = ∠ A O C = 23
noncomputable def angle_sum_property (O A C : Point) := ∠ A O C = 90

theorem angle_x_degrees
  (O B C A : Point) (h_radii : radii O B C A)
  (h_isosceles_triangle : isosceles_triangle O B C)
  (h_isosceles_base : isosceles_base_angle O A C)
  (h_sum_property : angle_sum_property O A C) :
  ∠ B C O - ∠ O A C = 9 := by sorry

end angle_x_degrees_l134_134045


namespace trig_inequality_sin_cos_l134_134063

theorem trig_inequality_sin_cos :
  Real.sin 2 + Real.cos 2 + 2 * (Real.sin 1 - Real.cos 1) ≥ 1 :=
by
  sorry

end trig_inequality_sin_cos_l134_134063


namespace rectangular_to_cylindrical_l134_134631

theorem rectangular_to_cylindrical (x y z r θ : ℝ) (h₁ : x = 3) (h₂ : y = -3 * Real.sqrt 3) (h₃ : z = 2)
  (h₄ : r = Real.sqrt (3^2 + (-3 * Real.sqrt 3)^2)) 
  (h₅ : r > 0) 
  (h₆ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₇ : θ = Float.pi * 5 / 3) : 
  (r, θ, z) = (6, 5 * Float.pi / 3, 2) :=
sorry

end rectangular_to_cylindrical_l134_134631


namespace scientific_notation_l134_134827

theorem scientific_notation : ∃ (a : ℝ) (n : ℤ), 274000000 = a * (10 ^ n) ∧ a = 2.74 ∧ n = 8 :=
by
  use [2.74, 8]
  sorry

end scientific_notation_l134_134827


namespace percentage_increase_pay_rate_l134_134222

theorem percentage_increase_pay_rate (r t c e : ℕ) (h_reg_rate : r = 10) (h_total_surveys : t = 100) (h_cellphone_surveys : c = 60) (h_total_earnings : e = 1180) : 
  (13 - 10) / 10 * 100 = 30 :=
by
  sorry

end percentage_increase_pay_rate_l134_134222


namespace average_stoppage_time_per_hour_l134_134512

theorem average_stoppage_time_per_hour :
    ∀ (v1_excl v1_incl v2_excl v2_incl v3_excl v3_incl : ℝ),
    v1_excl = 54 → v1_incl = 36 →
    v2_excl = 72 → v2_incl = 48 →
    v3_excl = 90 → v3_incl = 60 →
    ( ((54 / v1_excl - 54 / v1_incl) + (72 / v2_excl - 72 / v2_incl) + (90 / v3_excl - 90 / v3_incl)) / 3 = 0.5 ) := 
by
    intros v1_excl v1_incl v2_excl v2_incl v3_excl v3_incl
    sorry

end average_stoppage_time_per_hour_l134_134512


namespace second_pipe_fills_in_15_minutes_l134_134520

theorem second_pipe_fills_in_15_minutes :
  ∀ (x : ℝ),
  (∀ (x : ℝ), (1 / 2 + (7.5 / x)) = 1 → x = 15) :=
by
  intros
  sorry

end second_pipe_fills_in_15_minutes_l134_134520


namespace combined_area_ratio_l134_134894

def square (a : ℕ) : ℕ := a * a

theorem combined_area_ratio (sA sB sC : ℕ)
  (hA : sA = 36)
  (hB : sB = 42)
  (hC : sC = 54) :
  let areaA := square sA,
      areaB := square sB,
      areaC := square sC,
      combinedAreaAB := areaA + areaB,
      expectedRatioNumer := 255,
      expectedRatioDenom := 243 in 
  (combinedAreaAB * expectedRatioDenom) = (areaC * expectedRatioNumer) :=
by {
  sorry
}

end combined_area_ratio_l134_134894


namespace solution_set_of_f_gt_7_minimum_value_of_m_n_l134_134715

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem solution_set_of_f_gt_7 :
  { x : ℝ | f x > 7 } = { x | x > 4 ∨ x < -3 } :=
by
  ext x
  sorry

theorem minimum_value_of_m_n (m n : ℝ) (h : 0 < m ∧ 0 < n) (hfmin : ∀ x : ℝ, f x ≥ m + n) :
  m = n ∧ m = 3 / 2 ∧ m^2 + n^2 = 9 / 2 :=
by
  sorry

end solution_set_of_f_gt_7_minimum_value_of_m_n_l134_134715


namespace area_remaining_l134_134552

variable (d : ℝ)
variable (r : ℝ ≥ 0)
variable (A B M : Point)
variable (h_perpendicular : Chord M d (2 * sqrt 7))
variable (AB_semi : Semicircle A B)
variable (AM_semi MB_semi : Semicircle)

theorem area_remaining (h1 : Semicircle.diameter_len AB_semi d)
                       (h2 : Point.on_diameter M AB_semi)
                       (h3 : Semicircle.on_diameter AM_semi d/2)
                       (h4 : Semicircle.on_diameter MB_semi d/2)
                       (h5 : Chord.length h_perpendicular = 2 * sqrt 7) :
           remaining_area AB_semi AM_semi MB_semi = 21.99 := sorry

end area_remaining_l134_134552


namespace rob_has_12_pennies_l134_134452

def total_value_in_dollars (quarters dimes nickels pennies : ℕ) : ℚ :=
  (quarters * 25 + dimes * 10 + nickels * 5 + pennies) / 100

theorem rob_has_12_pennies
  (quarters : ℕ) (dimes : ℕ) (nickels : ℕ) (pennies : ℕ)
  (h1 : quarters = 7) (h2 : dimes = 3) (h3 : nickels = 5) 
  (h4 : total_value_in_dollars quarters dimes nickels pennies = 2.42) :
  pennies = 12 :=
by
  sorry

end rob_has_12_pennies_l134_134452


namespace intersection_proof_distance_proof_l134_134396

noncomputable def intersection_point (x y : ℝ) : Prop :=
  (y + x^2 = 1) ∧ (x + y + 1 = 0)

noncomputable def maximum_distance (x y : ℝ) : Prop :=
  (x + y + 1 = 0) → (x^2 + (y - 1)^2 = 1) → |dist (x, y) (0, 1)| ≤ sqrt 2 + 1

theorem intersection_proof : intersection_point (-1) 0 :=
by {
  have h1 : 0 + (-1)^2 = 1 := by simp,
  have h2 : -1 + 0 + 1 = 0 := by simp,
  exact ⟨h1, h2⟩
}

theorem distance_proof : ∀ x y : ℝ, maximum_distance x y :=
by {
  intros x y h1 h2,
  let d := abs ((0 + 1 + 1) / sqrt 2),
  have h3 : sqrt 2 ≤ sqrt 2 := le_sqrt (by norm_num) (by norm_num),
  have d_max : ∀ s := (sqrt 2 + 1), d ≤ (sqrt 2 + 1), {by norm_num},
  exact ⟨sub_le_iff_le_add'.mp d_max⟩
}

-- Including sorry to skip detailed calculations
sorry

end intersection_proof_distance_proof_l134_134396


namespace acute_triangle_area_and_ratio_l134_134616

theorem acute_triangle_area_and_ratio (a b c A B C : ℝ)
  (h_triangle : ∀ (A B C : ℝ), 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 ∧ A + B + C = π)
  (h_a : a = sqrt 3)
  (h_cos_diff : cos (B - C) = 9 / 10) :
  (let area_ABC := (1/2) * b * c * sin A in area_ABC = (7 * sqrt 3) / 10) ∧
  (let ratio_bc_a := (b + c) / a in sqrt 3 < ratio_bc_a ∧ ratio_bc_a <= 2) :=
by
  sorry

end acute_triangle_area_and_ratio_l134_134616


namespace number_of_paths_l134_134353

theorem number_of_paths (right_steps : ℕ) (upward_steps : ℕ) (total_steps : ℕ) 
  (h1 : right_steps = 7) (h2 : upward_steps = 9) (h3 : total_steps = right_steps + upward_steps) : 
  nat.choose total_steps upward_steps = 11440 :=
by
  -- Proof goes here
  sorry

end number_of_paths_l134_134353


namespace Joey_age_when_Beth_was_Joey_age_now_l134_134379

def Joey_age_now : ℕ := 9
def years_in_future : ℕ := 5
def Beth_age_now : ℕ := Joey_age_now + years_in_future

theorem Joey_age_when_Beth_was_Joey_age_now
  (Joey_age_now : ℕ)
  (years_in_future : ℕ)
  (Beth_age_now : ℕ)
  (h_Beth_current_age : Beth_age_now = Joey_age_now + years_in_future)
  (Joey_age_when_Beth_9 : ℕ := Joey_age_now - (Beth_age_now - Joey_age_now)) :
  Joey_age_when_Beth_9 = 4 :=
by
  rw [h_Beth_current_age]
  simp [Joey_age_now]
  sorry

end Joey_age_when_Beth_was_Joey_age_now_l134_134379


namespace shaded_region_area_is_correct_l134_134768

structure Point where
  x : ℝ
  y : ℝ
  
structure Parallelogram (A B C D E : Point) where
  base : ℝ
  height : ℝ
  CE : ℝ
  ED : ℝ
  condition1 : base = 12
  condition2 : height = 10
  condition3 : CE = 7
  condition4 : ED = 5

def area_of_shaded_region (A B C D E : Point) (parallelogram : Parallelogram A B C D E) : ℝ :=
  (parallelogram.base * parallelogram.height) - (1 / 2 * (parallelogram.base - parallelogram.CE) * parallelogram.height)

theorem shaded_region_area_is_correct (A B C D E : Point) (parallelogram : Parallelogram A B C D E) :
  area_of_shaded_region A B C D E parallelogram = 95 := by
  sorry

end shaded_region_area_is_correct_l134_134768


namespace inverse_proposition_l134_134880

theorem inverse_proposition (x : ℝ) : 
  (¬ (x > 2) → ¬ (x > 1)) ↔ ((x > 1) → (x > 2)) := 
by 
  sorry

end inverse_proposition_l134_134880


namespace sum_of_1980_vectors_is_zero_l134_134304

theorem sum_of_1980_vectors_is_zero :
  ∃ (v : ℕ → ℝ × ℝ), (∀ i j k, (i ≠ j ∧ j ≠ k ∧ i ≠ k) → ¬ collinear {v i, v j, v k}) ∧
                    (∀ i, i ≤ 1980 → ∃ u, collinear {u, ∑ j in finset.range 1979, v j}) ∧
                    (∑ i in finset.range 1980, v i = (0, 0)) := sorry

end sum_of_1980_vectors_is_zero_l134_134304


namespace correlation_coefficient_and_regression_equation_l134_134577

def selling_prices : List ℝ := [52, 50, 48, 45, 44, 43]
def sales_volumes : List ℝ := [5, 6, 7, 8, 10, 12]

noncomputable def correlation_coefficient (xs ys : List ℝ) : ℝ :=
  let n := xs.length
  let x̄ := (xs.sum / n)
  let ȳ := (ys.sum / n)
  let numerator := (List.zip xs ys).sum (λ ⟨x, y⟩ => (x - x̄) * (y - ȳ))
  let x_denom := Real.sqrt ((xs.sum_map (λ x => (x - x̄) ^ 2)))
  let y_denom := Real.sqrt ((ys.sum_map (λ y => (y - ȳ) ^ 2)))
  numerator / (x_denom * y_denom)

noncomputable def regression_equation (xs ys : List ℝ) : (ℝ × ℝ) :=
  let n := xs.length
  let x̄ := (xs.sum / n)
  let ȳ := (ys.sum / n)
  let slope := (List.zip xs ys).sum (λ ⟨x, y⟩ => (x - x̄) * (y - ȳ)) / ((xs.sum_map (λ x => (x - x̄) ^ 2)))
  let intercept := ȳ - slope * x̄
  (slope, intercept)

def estimated_sales_volume (slope intercept : ℝ) (selling_price : ℝ) : ℝ :=
  slope * selling_price + intercept

theorem correlation_coefficient_and_regression_equation :
  correlation_coefficient selling_prices sales_volumes = -0.94 ∧
  regression_equation selling_prices sales_volumes = (-11 / 16, 645 / 16) ∧
  estimated_sales_volume (-11 / 16) (645 / 16) 55 = 25 :=
  sorry

end correlation_coefficient_and_regression_equation_l134_134577


namespace percent_alcohol_in_new_solution_l134_134974

-- Define the volumes and initial percentages
variables {initial_vol : ℝ} {initial_percent_alcohol : ℝ}
variables {added_alcohol : ℝ} {added_water : ℝ} 

-- Given conditions
def initial_vol := 40 -- initial solution volume in liters
def initial_percent_alcohol := 0.05 -- initial percentage of alcohol
def added_alcohol := 6.5 -- additional alcohol in liters
def added_water := 3.5 -- additional water in liters

-- Calculate initial alcohol and final solution details
def initial_alcohol := initial_vol * initial_percent_alcohol
def final_alcohol := initial_alcohol + added_alcohol
def final_total_vol := initial_vol + added_alcohol + added_water

-- Theorem to prove the final percentage of alcohol
theorem percent_alcohol_in_new_solution :
  (final_alcohol / final_total_vol) * 100 = 17 :=
by sorry

end percent_alcohol_in_new_solution_l134_134974


namespace angle_x_l134_134022

-- Define the conditions
variables (O A C D B : Point)
variable h : circle_center O
variable h1 : on_circumference A C D O
variable h2 : diameter AC
variable h3 : ∠ CDA = 42

-- The problem statement we want to prove
theorem angle_x (x : ℝ) : x = 58 :=
begin
  -- This is where the solution steps would go, but for now, we leave it as sorry
  sorry
end

end angle_x_l134_134022


namespace arrangement_of_bananas_l134_134644

-- Define the constants for the number of letters and repetitions in the word BANANAS.
def num_letters : ℕ := 7
def count_A : ℕ := 3
def count_N : ℕ := 2
def factorial (n : ℕ) := nat.factorial n

-- The number of ways to arrange the letters of the word BANANAS.
noncomputable def num_ways_to_arrange := 
  (factorial num_letters) / (factorial count_A * factorial count_N)

theorem arrangement_of_bananas : 
  num_ways_to_arrange = 420 :=
sorry

end arrangement_of_bananas_l134_134644


namespace spending_record_l134_134262

-- Definitions based on conditions
def deposit_record (x : ℤ) : ℤ := x
def spend_record (x : ℤ) : ℤ := -x

-- Theorem statement
theorem spending_record (x : ℤ) (hx : x = 500) : spend_record x = -500 := by
  sorry

end spending_record_l134_134262


namespace total_books_97_l134_134435

variable (nDarryl nLamont nLoris : ℕ)

-- Conditions
def condition1 (nLoris nLamont : ℕ) : Prop := nLoris + 3 = nLamont
def condition2 (nLamont nDarryl : ℕ) : Prop := nLamont = 2 * nDarryl
def condition3 (nDarryl : ℕ) : Prop := nDarryl = 20

-- Theorem stating the total number of books is 97
theorem total_books_97 : nLoris + nLamont + nDarryl = 97 :=
by
  have h1 : nDarryl = 20 := condition3 nDarryl
  have h2 : nLamont = 2 * nDarryl := condition2 nLamont nDarryl
  have h3 : nLoris + 3 = nLamont := condition1 nLoris nLamont
  sorry

end total_books_97_l134_134435


namespace maddie_episodes_friday_l134_134436

theorem maddie_episodes_friday :
  let total_episodes : ℕ := 8
  let episode_duration : ℕ := 44
  let monday_time : ℕ := 138
  let thursday_time : ℕ := 21
  let weekend_time : ℕ := 105
  let total_time : ℕ := total_episodes * episode_duration
  let non_friday_time : ℕ := monday_time + thursday_time + weekend_time
  let friday_time : ℕ := total_time - non_friday_time
  let friday_episodes : ℕ := friday_time / episode_duration
  friday_episodes = 2 :=
by
  sorry

end maddie_episodes_friday_l134_134436


namespace probability_green_ball_l134_134625

noncomputable def P_g : ℚ := 7 / 15

def balls_in_X : ℕ × ℕ := (3, 7)  -- (red balls, green balls)
def balls_in_Y : ℕ × ℕ := (8, 2)  -- (red balls, green balls)
def balls_in_Z : ℕ × ℕ := (5, 5)  -- (red balls, green balls)

def containers : List (ℕ × ℕ) := [balls_in_X, balls_in_Y, balls_in_Z]

theorem probability_green_ball : 
  (1/3) * (7/10) + (1/3) * (1/5) + (1/3) * (1/2) = 7 / 15 := 
by sorry

end probability_green_ball_l134_134625


namespace peter_makes_fewest_pies_l134_134185

-- Define the areas based on given pie dimensions
def area_alex_pie : ℝ := 24
def area_ron_pie : ℝ := 16
noncomputable def area_peter_pie : ℝ := 9 * Real.pi
def area_tara_pie : ℝ := 24

-- Define the total dough amount D
variable (D : ℝ) (hD: D > 0)

-- Calculate the number of pies each friend can make
def pies_alex (D : ℝ) : ℝ := D / area_alex_pie
def pies_ron (D : ℝ) : ℝ := D / area_ron_pie
def pies_peter (D : ℝ) : ℝ := D / area_peter_pie
def pies_tara (D : ℝ) : ℝ := D / area_tara_pie

-- Prove that Peter makes the fewest pies
theorem peter_makes_fewest_pies (hD: D > 0) : 
  pies_peter D < min (pies_alex D) (min (pies_ron D) (pies_tara D)) := by
  sorry

end peter_makes_fewest_pies_l134_134185


namespace radius_of_ball_l134_134180

theorem radius_of_ball
  (R : ℝ)
  (h₁ : ∀ (b: ball), b.floats_on_surface_of_lake)
  (h₂: ∀ (hole: hole), hole.top_diameter = 24 ∧ hole.depth = 8): 
  R = 13 :=
sorry

end radius_of_ball_l134_134180


namespace roots_sum_and_product_l134_134622

theorem roots_sum_and_product (x : ℝ) (hx : |x|^2 - 5 * |x| + 6 = 0) :
  let y := |x| in
  (y = 2 ∨ y = 3) →
  (∀ x1 x2 x3 x4 : ℝ, {x1, x2, x3, x4} = {2, -2, 3, -3} → x1 + x2 + x3 + x4 = 0 ∧ x1 * x2 * x3 * x4 = 36) := 
begin
  intros hy,
  obtain ⟨y1, y2, hy_eq⟩ : {y1, y2} = {2, 3},
  {  sorry },
  intros x1 x2 x3 x4 h,
  subst_vars,
  split; sorry
end

end roots_sum_and_product_l134_134622


namespace remaining_demerits_correct_l134_134228

def total_demerits_for_lateness : ℕ := 2 * 6
def demerits_for_joke : ℕ := 15
def total_accumulated_demerits : ℕ := total_demerits_for_lateness + demerits_for_joke
def max_demerits : ℕ := 50
def remaining_demerits : ℕ := max_demerits - total_accumulated_demerits

theorem remaining_demerits_correct : remaining_demerits = 23 := by
  -- Using definitions to calculate
  have h1 : total_demerits_for_lateness = 12 := by
    simp [total_demerits_for_lateness]
  have h2 : total_accumulated_demerits = 12 + demerits_for_joke := by
    simp [total_accumulated_demerits, h1]
  have h3 : total_accumulated_demerits = 27 := by
    simp [total_accumulated_demerits, h1]
  show remaining_demerits = 23 from
    calc remaining_demerits = max_demerits - 27 : by simp [remaining_demerits, h3]
                       ... = 23 : by simp
  sorry

end remaining_demerits_correct_l134_134228


namespace possible_denominators_count_l134_134088

theorem possible_denominators_count (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 1 ≤ b ∧ b ≤ 9) (h3 : 1 ≤ c ∧ c ≤ 9) (h4 : ¬(a = b ∧ b = c)) :
  6 = finset.card { d ∈ {1, 3, 9, 27, 37, 111, 333, 999} | ∃ k, (k = 100*a + 10*b + c ∧ nat.gcd k 999 = d) }  :=
by sorry

end possible_denominators_count_l134_134088


namespace problem_statement_l134_134335

noncomputable def f (x : ℝ) : ℝ :=
if h : x ≥ 4 then 4 / x + 1 else log 2 x

theorem problem_statement (a b c : ℝ) (h1 : f a = c) (h2 : f b = c) (h3 : f' b < 0) : b > a ∧ a > c := by
  sorry

end problem_statement_l134_134335


namespace trapezoid_sides_l134_134999

theorem trapezoid_sides {R P Q K M O AB BC CD AD : ℝ}
  (hR : R = 5) 
  (hPQ : PQ = 8) 
  (hTangent1 : tangent O P C D) 
  (hTangent2 : tangent O Q A B)
  (hParallel : tangent_parallel AB PQ CD) :
  AB = 12.5 ∧ BC = 5 ∧ CD = 12.5 ∧ AD = 20 :=
by
  sorry

end trapezoid_sides_l134_134999


namespace max_tuesdays_in_first_63_days_l134_134953

theorem max_tuesdays_in_first_63_days : ∀ (days : ℕ), days = 63 → nat.div days 7 = 9 → ∃ tuesdays : ℕ, tuesdays = 9 := 
by
  intros 
  sorry 

end max_tuesdays_in_first_63_days_l134_134953


namespace find_value_of_x_l134_134005

-- Definitions used in conditions
variable (O B C A : Point) -- points on the circle
variable (OB OC OA : ℝ) -- Radii of the circle
variable (OB_eq_OC : OB = OC)
variable (angle_BCO : ℝ) -- Given angle ∠BCO
variable (angle_DAC : ℝ) -- Given angle ∠DAC
variable (right_angled_ACD : angle_DAC + 90 = 157) -- Triangle ACD is right-angled & 180 - 23 = 157

variable (x : ℝ) -- angle x

-- Setting specific values for given angles in degrees
noncomputable def angle_BCO_value : angle_BCO = 32
noncomputable def angle_DAC_value : angle_DAC = 67

-- What we want to prove
theorem find_value_of_x : x = 9 :=
sorry

end find_value_of_x_l134_134005


namespace cos_arith_prog_l134_134711

theorem cos_arith_prog (
  {a_n : ℕ → ℝ} 
  (h_arith : ∀ n m, a_n = a_m + (n - m) * d) -- arithmetic progression condition for any terms
  (h_sum : a_1 + a_7 + a_{13} = 4 * π)
) : cos (a_2 + a_{12}) = - (1 / 2) :=
begin
  -- The description of our theorem
  sorry
end

end cos_arith_prog_l134_134711


namespace angle_x_degrees_l134_134038

theorem angle_x_degrees 
  (O : Point)
  (A B C D : Point)
  (hC : Circle O)
  (hOB : O ∈ Segment O B)
  (hOC : O ∈ Segment O C)
  (hBCO : ∠ B C O = 32)
  (hACD : ∠ A C D = 90)
  (hCAD : ∠ C A D = 67) :
  ∠ B C A = 9 := 
  sorry

end angle_x_degrees_l134_134038


namespace triangle_XYZ_l134_134401

theorem triangle_XYZ
  (X Y Z : Type)
  (angle_X : Degree)
  (YZ : ℝ)
  (tan_Z : ℝ)
  (cos_Y : ℝ)
  (XY : ℝ) :
  angle_X = 90 ∧ YZ = 20 ∧ tan_Z = 3 * cos_Y ∧ cos_Y = XY / 20 ∧ tan_Z = XY / (√(YZ^2 - XY^2)) →
    XY = 40 * sqrt 2 / 3 :=
begin
  sorry
end

end triangle_XYZ_l134_134401


namespace cauchy_schwarz_equality_l134_134842

theorem cauchy_schwarz_equality (n : ℕ) (x y : Finₓ n → ℝ) :
  (∑ i, x i * y i) ^ 2 = (∑ i, x i ^ 2) * (∑ i, y i ^ 2) ↔ ∃ λ : ℝ, ∀ i, x i = λ * y i := sorry

end cauchy_schwarz_equality_l134_134842


namespace total_wheels_in_garage_l134_134921

def total_wheels (bicycles tricycles unicycles : ℕ) (bicycle_wheels tricycle_wheels unicycle_wheels : ℕ) :=
  bicycles * bicycle_wheels + tricycles * tricycle_wheels + unicycles * unicycle_wheels

theorem total_wheels_in_garage :
  total_wheels 3 4 7 2 3 1 = 25 := by
  -- Calculation shows:
  -- (3 * 2) + (4 * 3) + (7 * 1) = 6 + 12 + 7 = 25
  sorry

end total_wheels_in_garage_l134_134921


namespace number_of_fiction_books_l134_134828

theorem number_of_fiction_books (F NF : ℕ) (h1 : F + NF = 52) (h2 : NF = 7 * F / 6) : F = 24 := 
by
  sorry

end number_of_fiction_books_l134_134828


namespace no_nonzero_abc_polynomial_with_n_integer_roots_l134_134272

theorem no_nonzero_abc_polynomial_with_n_integer_roots (a b c : ℤ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) :
  ¬ ∀ n > 3, ∃ (P_n : ℤ[X]), (P_n.degree = n) ∧ (∀ x, P_n.eval x = 0 → x ∈ ℤ) ∧ (P_n.nroots = n) :=
by
  sorry

end no_nonzero_abc_polynomial_with_n_integer_roots_l134_134272


namespace yann_camille_meal_combinations_l134_134463

theorem yann_camille_meal_combinations :
  let main_dishes := 12
  let side_dishes := 5
  let yann_choices := main_dishes
  let camille_choices := main_dishes
  in (yann_choices * camille_choices * side_dishes) = 720 := by
  sorry

end yann_camille_meal_combinations_l134_134463


namespace solve_for_x_l134_134738

theorem solve_for_x :
  ∃ x : ℝ, 9^(x + 2) = 240 + 9^x ∧ x = 0.5 :=
begin
  use 0.5,
  split,
  { -- Prove that 9^(0.5 + 2) = 240 + 9^0.5
    calc 
      9^(0.5 + 2) = 9^2 * 9^0.5 : by rw [← pow_add]
              ... = 81 * 3      : by { rw [pow_one_half_eq_root, pow_two_eq], refl }
              ... = 240 + 9^0.5 : by { norm_num, rw pow_one_half_eq_root, norm_num }
  },
  { -- Prove that x = 0.5
    refl
  },
end

end solve_for_x_l134_134738


namespace complex_number_on_ray_is_specific_l134_134306

open Complex

theorem complex_number_on_ray_is_specific (a b : ℝ) (z : ℂ) (h₁ : z = a + b * I) 
  (h₂ : a = b) (h₃ : abs z = 1) : 
  z = (Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * I :=
by
  sorry

end complex_number_on_ray_is_specific_l134_134306


namespace M_is_infinite_l134_134948

variable (M : Set ℝ)

def has_properties (M : Set ℝ) : Prop :=
  (∃ x y : ℝ, x ∈ M ∧ y ∈ M ∧ x ≠ y) ∧ ∀ x ∈ M, (3*x - 2 ∈ M ∨ -4*x + 5 ∈ M)

theorem M_is_infinite (M : Set ℝ) (h : has_properties M) : ¬Finite M := by
  sorry

end M_is_infinite_l134_134948


namespace charlie_steps_proof_l134_134246

-- Define the conditions
def Steps_Charlie_3km : ℕ := 5350
def Laps : ℚ := 2.5

-- Define the total steps Charlie can make in 2.5 laps
def Steps_Charlie_total : ℕ := 13375

-- The statement to prove
theorem charlie_steps_proof : Laps * Steps_Charlie_3km = Steps_Charlie_total :=
by
  sorry

end charlie_steps_proof_l134_134246


namespace number_of_incorrect_statements_l134_134604

-- Definitions and conditions
def statement1 : Prop :=
  (¬ ∀ x : ℝ, x^2 - x + 1 ≤ 0) = (∃ x : ℝ, x^2 - x + 1 > 0)

def statement2 (p q : Prop) : Prop :=
  ¬ (p ∨ q) → (¬ p ∧ ¬ q)

def isEllipseCondition (m n : ℝ) : Prop :=
  (m > 0 ∧ n > 0 ∧ m ≠ n)

def statement3 (m n : ℝ) : Prop :=
  (mn > 0) → isEllipseCondition m n

-- The proof problem
theorem number_of_incorrect_statements :
  let s1 := statement1,
      s2 := statement2,
      s3 := statement3 in
  (¬s1 → ¬s2 → ¬s3 → 1 = 1) :=
sorry

end number_of_incorrect_statements_l134_134604


namespace PR_plus_PS_eq_BF_l134_134184

variable {A B C D P S R Q F : Type*}
variables [Geometry A B C D] [Rectangle A B C D] (P : Point (A, C))
variable [Perpendicular PS AD] [Perpendicular PR BC] [Perpendicular PQ BF]
variable [Perpendicular BF AD]

theorem PR_plus_PS_eq_BF (h1 : Perpendicular PS AD)
                            (h2 : Perpendicular PR BC)
                            (h3 : Perpendicular BF AD)
                            (h4 : Perpendicular PQ BF) :
  length (PR) + length (PS) = length (BF) :=
sorry

end PR_plus_PS_eq_BF_l134_134184


namespace correct_survey_order_l134_134601

theorem correct_survey_order :
  ∀ (steps : Finset ℤ),
  steps = {1, 2, 3, 4} →
  (∃ order : List ℤ, order = [2, 4, 3, 1] ∧
    (∀ step, step ∈ steps ↔ step ∈ order)) :=
by
  intros steps h_steps
  use [2, 4, 3, 1]
  split
  . rfl
  . intros step
    simp [h_steps]

end correct_survey_order_l134_134601


namespace soap_parts_needed_l134_134387

theorem soap_parts_needed (scraps soaps: ℕ) (H1: scraps = 251) (H2: soaps = 25) : ℕ :=
let P := (scraps / soaps).ceil in
P

end soap_parts_needed_l134_134387


namespace cream_strawberry_prices_l134_134260

noncomputable def price_flavor_B : ℝ := 30
noncomputable def price_flavor_A : ℝ := 40

theorem cream_strawberry_prices (x y : ℝ) 
  (h1 : y = x + 10) 
  (h2 : 800 / y = 600 / x) : 
  x = price_flavor_B ∧ y = price_flavor_A :=
by 
  sorry

end cream_strawberry_prices_l134_134260


namespace probability_bulb_on_l134_134911

def toggle_process_result : ℕ → Bool
| n := (count_divisors n) % 2 = 1

def count_divisors (n : ℕ) : ℕ :=
  (finset.Icc 1 n).count (λ d : ℕ, n % d = 0)

theorem probability_bulb_on :
  let total_bulbs := 100
  let remaining_bulbs := finset.range 101 |>.filter toggle_process_result |>.card
  remaining_bulbs / total_bulbs = 0.1 :=
sorry

end probability_bulb_on_l134_134911


namespace periodic_function_with_period_l134_134476

noncomputable theory
open Classical

variables {f : ℝ → ℝ} (c : ℝ) (h1 : ∀ a b : ℝ, f(a + b) + f(a - b) = 2 * f(a) * f(b))
  (h2 : c > 0) (h3 : f(c / 2) = 0)

theorem periodic_function_with_period : ∃ T > 0, ∀ x : ℝ, f(x + T) = f(x) :=
begin
  use 2 * c,
  split,
  { linarith [h2] },
  intros,
  sorry
end

end periodic_function_with_period_l134_134476


namespace solve_for_y_l134_134859

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 :=
by sorry

end solve_for_y_l134_134859


namespace binomial_coefficient_ratio_l134_134254

theorem binomial_coefficient_ratio :
  let C1 := (-(1/2))
  let n := 1007
  let k := 2014
  let binom1 := ((C1 * (C1 - 1) * (C1 - 2) * ... * (C1 - (n-1))) / (nat.factorial n))
  let binom2 := (nat.factorial k / (nat.factorial n * nat.factorial (k - n)))
  (binom1 * 4^n / binom2) = -(1 / (n + 1)) :=
by
  sorry

end binomial_coefficient_ratio_l134_134254


namespace sum_three_lowest_scores_l134_134846

theorem sum_three_lowest_scores
  (scores : list ℝ)
  (h_len : scores.length = 7)
  (h_mean : (scores.sum / 7) = 85)
  (h_median : ∃ l r, list.split_at 3 (scores.sorted) = (l, 88 :: r))
  (h_mode : ∃ t u, list.split_at 3 (scores.sorted.reverse) = (90 :: 90 :: 90 :: t, u)) :
  (list.take 3 (scores.sorted)).sum = 237 := 
  sorry

end sum_three_lowest_scores_l134_134846


namespace parallelogram_area_scaled_vectors_l134_134469

open RealEuclideanSpace

section ParallelogramArea

variables (u v : ℝ^3)
variable (area_uv : ℝ)

def parallelogram_area (a b : ℝ^3) : ℝ := ‖a × b‖

theorem parallelogram_area_scaled_vectors
    (h : parallelogram_area u v = 12) :
    parallelogram_area (3 • u - 4 • v) (2 • u + v) = 132 :=
by
  sorry

end ParallelogramArea

end parallelogram_area_scaled_vectors_l134_134469


namespace smaller_angle_between_hands_at_3_40_pm_larger_angle_between_hands_at_3_40_pm_l134_134155

def degree_movement_per_minute_of_minute_hand : ℝ := 6
def degree_movement_per_hour_of_hour_hand : ℝ := 30
def degree_movement_per_minute_of_hour_hand : ℝ := 0.5

def minute_position_at_3_40_pm : ℝ := 40 * degree_movement_per_minute_of_minute_hand
def hour_position_at_3_40_pm : ℝ := 3 * degree_movement_per_hour_of_hour_hand + 40 * degree_movement_per_minute_of_hour_hand

def clockwise_angle_between_hands_at_3_40_pm : ℝ := minute_position_at_3_40_pm - hour_position_at_3_40_pm
def counterclockwise_angle_between_hands_at_3_40_pm : ℝ := 360 - clockwise_angle_between_hands_at_3_40_pm

theorem smaller_angle_between_hands_at_3_40_pm : clockwise_angle_between_hands_at_3_40_pm = 130.0 := 
by
  sorry

theorem larger_angle_between_hands_at_3_40_pm : counterclockwise_angle_between_hands_at_3_40_pm = 230.0 := 
by
  sorry

end smaller_angle_between_hands_at_3_40_pm_larger_angle_between_hands_at_3_40_pm_l134_134155


namespace angle_x_degrees_l134_134031

theorem angle_x_degrees 
  (O : Point)
  (A B C D : Point)
  (hC : Circle O)
  (hOB : O ∈ Segment O B)
  (hOC : O ∈ Segment O C)
  (hBCO : ∠ B C O = 32)
  (hACD : ∠ A C D = 90)
  (hCAD : ∠ C A D = 67) :
  ∠ B C A = 9 := 
  sorry

end angle_x_degrees_l134_134031


namespace score_90_standard_devs_above_mean_l134_134667

-- Definitions for the conditions
def mean : ℝ := 88.8
def std_dev : ℝ := (mean - 86) / 7

-- Lean statement to prove the equivalence
theorem score_90_standard_devs_above_mean : 90 = mean + 3 * std_dev := by
  -- We add 'sorry' to skip the proof
  sorry

end score_90_standard_devs_above_mean_l134_134667


namespace intersection_M_N_l134_134500

open Set

-- Definitions of sets M and N
def M : Set (ℝ × ℝ) := { a | ∃ x : ℝ, a = (-1, 1) + x • (1, 2) }
def N : Set (ℝ × ℝ) := { a | ∃ x : ℝ, a = (1, -2) + x • (2, 3) }

-- Theorem stating that the intersection of M and N is exactly {(-13, -23)}
theorem intersection_M_N : M ∩ N = {(-13, -23)} := by
  sorry

end intersection_M_N_l134_134500


namespace john_days_present_contradiction_l134_134976

noncomputable theory

-- Definitions based on the conditions
def days_present (P A : ℕ) : Prop := P + A = 60
def total_pay (P A : ℕ) : Prop := 7 * P + 3 * A = 170

theorem john_days_present_contradiction : ∀ P A : ℕ, days_present P A → total_pay P A → False :=
by {
  intros P A h_days_present h_total_pay,
  -- Since the goal is to show that the given conditions lead to a contradiction
  sorry
}

end john_days_present_contradiction_l134_134976


namespace find_value_of_x_l134_134012

-- Definitions used in conditions
variable (O B C A : Point) -- points on the circle
variable (OB OC OA : ℝ) -- Radii of the circle
variable (OB_eq_OC : OB = OC)
variable (angle_BCO : ℝ) -- Given angle ∠BCO
variable (angle_DAC : ℝ) -- Given angle ∠DAC
variable (right_angled_ACD : angle_DAC + 90 = 157) -- Triangle ACD is right-angled & 180 - 23 = 157

variable (x : ℝ) -- angle x

-- Setting specific values for given angles in degrees
noncomputable def angle_BCO_value : angle_BCO = 32
noncomputable def angle_DAC_value : angle_DAC = 67

-- What we want to prove
theorem find_value_of_x : x = 9 :=
sorry

end find_value_of_x_l134_134012


namespace ratio_of_x_intercepts_l134_134139

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v : ℝ)
  (hu : u = -b / 5) (hv : v = -b / 3) : u / v = 3 / 5 := by
  sorry

end ratio_of_x_intercepts_l134_134139


namespace man_l134_134585

theorem man's_age_twice_son (S M Y : ℕ) (h1 : M = S + 26) (h2 : S = 24) (h3 : M + Y = 2 * (S + Y)) : Y = 2 :=
by
  sorry

end man_l134_134585


namespace passing_time_correct_l134_134522

-- Definitions based on the given conditions
def speed_first_train := 80 -- in kmph
def speed_second_train := 70 -- in kmph
def length_first_train := 150 -- in meters
def length_second_train := 100 -- in meters

-- Conversion constants
def km_to_m : Float := 1000.0
def hour_to_s : Float := 3600.0

-- Relative speed calculation in m/s
def relative_speed_kmph := speed_first_train + speed_second_train
def relative_speed_mps := (relative_speed_kmph * km_to_m) / hour_to_s

-- Combined length of trains in meters
def combined_length := length_first_train + length_second_train

-- Time calculation in seconds
def passing_time := combined_length / relative_speed_mps

theorem passing_time_correct : passing_time ≈ 6 := by
  sorry

end passing_time_correct_l134_134522


namespace T_n_less_than_one_fourth_l134_134301

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.sin (π / 2 * x)

-- Define the sequence a_n
def a (n : ℕ) : ℝ := 2 * n - 1

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 1 / (a (n + 1))^2

-- Define the sum of the first n terms of b, denoted as T_n
def T (n : ℕ) : ℝ := ∑ k in Finset.range n, b k

-- Prove that T_n < 1 / 4
theorem T_n_less_than_one_fourth (n : ℕ) : T n < 1 / 4 :=
by
  sorry

end T_n_less_than_one_fourth_l134_134301


namespace inverse_variation_a_squared_of_b_squared_l134_134465

theorem inverse_variation_a_squared_of_b_squared (a b k : ℝ) (h1: a⁻² * b² = k) (h2: a = 5) (h3: b = 2) : (∀ b : ℝ, b = 8 → a² = 25/16) :=
by
  have h4 : (25 : ℝ) * 2⁴ = 100 := by norm_num
  have h5 : (8 : ℝ)^2 = 64 := by norm_num
  have h6 : k = 25*4 := by norm_num
  have h7 : k = 100 := by rw [h4, h6]
  sorry -- Complete the final parts

end inverse_variation_a_squared_of_b_squared_l134_134465


namespace probability_of_positive_test_l134_134865

theorem probability_of_positive_test :
  let P_A := 0.01
  let P_B_given_A := 0.99
  let P_B_given_not_A := 0.1
  let P_not_A := 0.99
  let P_B := P_B_given_A * P_A + P_B_given_not_A * P_not_A
  in P_B = 0.1089 :=
by
  let P_A := 0.01
  let P_B_given_A := 0.99
  let P_B_given_not_A := 0.1
  let P_not_A := 0.99
  let P_B := P_B_given_A * P_A + P_B_given_not_A * P_not_A
  have : P_B = 0.1089, from by norm_num [P_B_given_A, P_A, P_B_given_not_A, P_not_A, P_B]
  exact this

end probability_of_positive_test_l134_134865


namespace sum_two_digit_integers_meeting_condition_l134_134963

def sum_of_valid_two_digit_integers : Nat :=
  let valid_pairs := [(1, 2), (2, 4), (3, 6)]
  let valid_numbers := valid_pairs.map (λ (a, b), 10 * a + b)
  valid_numbers.sum

theorem sum_two_digit_integers_meeting_condition : 
  sum_of_valid_two_digit_integers = 72 := by
  -- proof will follow from the definition of valid_pairs, valid_numbers, and the sum 
  sorry

end sum_two_digit_integers_meeting_condition_l134_134963


namespace sarah_shampoo_and_conditioner_usage_l134_134072

-- Condition Definitions
def shampoo_daily_oz := 1
def conditioner_daily_oz := shampoo_daily_oz / 2
def total_daily_usage := shampoo_daily_oz + conditioner_daily_oz

def days_in_two_weeks := 14

-- Assertion: Total volume used in two weeks.
theorem sarah_shampoo_and_conditioner_usage :
  (days_in_two_weeks * total_daily_usage) = 21 := by
  sorry

end sarah_shampoo_and_conditioner_usage_l134_134072


namespace two_a_minus_two_d_eq_zero_l134_134800

noncomputable def g (a b c d x : ℝ) : ℝ := (2 * a * x - b) / (c * x - 2 * d)

theorem two_a_minus_two_d_eq_zero (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : ∀ x : ℝ, (g a a c d (g a b c d x)) = x) : 2 * a - 2 * d = 0 :=
sorry

end two_a_minus_two_d_eq_zero_l134_134800


namespace tangent_line_equation_l134_134480

noncomputable def curve (x : ℝ) : ℝ := -x^3 + 3*x^2

def derivative_curve (x : ℝ) : ℝ := -3*x^2 + 6*x

def point_of_tangency : ℝ × ℝ := (1, 2)

theorem tangent_line_equation :
  let slope := derivative_curve 1 in
  let y_intercept := 2 - slope * 1 in
  ∀ x : ℝ, curve x = (slope * x + y_intercept) →
  curve 1 = 2 ∧ derivative_curve 1 = 3 ∧
  (∀ x : ℝ, slope * x + y_intercept = 3*x - 1) :=
by
  sorry

end tangent_line_equation_l134_134480


namespace new_average_age_l134_134094

/--
The average age of 7 people in a room is 28 years.
A 22-year-old person leaves the room, and a 30-year-old person enters the room.
Prove that the new average age of the people in the room is \( 29 \frac{1}{7} \).
-/
theorem new_average_age (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ) (entering_age : ℕ)
  (H1 : avg_age = 28)
  (H2 : num_people = 7)
  (H3 : leaving_age = 22)
  (H4 : entering_age = 30) :
  (avg_age * num_people - leaving_age + entering_age) / num_people = 29 + 1 / 7 := 
by
  sorry

end new_average_age_l134_134094


namespace tangent_line_circle_l134_134100

theorem tangent_line_circle (a b x y : ℝ) (h_circle : a^2 + b^2 = 1) : 
  a * x + b * y - 1 = 0 ↔ ((a, b) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} ∧ 
    ∀ (t : ℝ), (t * a, t * b) ∉ {p : ℝ × ℝ | p.1 * (x - a) + p.2 * (y - b) = 0}) := 
by 
  intro h
  sorry

end tangent_line_circle_l134_134100


namespace coin_flip_problem_solution_l134_134588

/-
Define the problem conditions and proof goal.
-/
theorem coin_flip_problem_solution (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) (h_ineq : a > b)
  (h_unattainable : number_of_unattainable_scores a b = 35) :
  a = 11 ∧ b = 8 :=
sorry

/-
Helper function that calculates the number of unattainable scores given a and b.
-/
noncomputable def number_of_unattainable_scores (a b : ℕ) : ℕ :=
  if Nat.gcd a b = 1 then
    (a - 1) * (b - 1) / 2
  else
    0

end coin_flip_problem_solution_l134_134588


namespace number_of_cartons_of_yoghurt_l134_134091

-- Define the given conditions
def number_of_ice_cream_cartons : ℕ := 19
def cost_per_ice_cream_carton : ℕ := 7
def extra_spent_on_ice_cream : ℕ := 129
def cost_of_ice_cream := number_of_ice_cream_cartons * cost_per_ice_cream_carton

-- Define the variable for cartons of yoghurt
variable (Y : ℕ)

-- State the problem conditions
axiom yoghurt_condition : cost_of_ice_cream = Y + extra_spent_on_ice_cream

-- Final proof statement
theorem number_of_cartons_of_yoghurt : Y = 4 :=
by
  -- Use the conditions given to state the exact proof requirement
  have : number_of_ice_cream_cartons * cost_per_ice_cream_carton = 133 := rfl
  have : extra_spent_on_ice_cream = 129 := rfl
  have : cost_of_ice_cream = 133 := rfl
  have : yogurt_condition : 133 = Y + 129 := yoghurt_condition
  calc
    Y = 133 - 129 := by ring
    ... = 4 := rfl

end number_of_cartons_of_yoghurt_l134_134091


namespace helen_gas_usage_l134_134350

def lawn_cuts_per_month (month : ℕ) : ℕ :=
  if month = 3 ∨ month = 4 ∨ month = 9 ∨ month = 10 then 2 else 4

theorem helen_gas_usage : ∑ m in finset.range(4) (2 + 4) * (lawn_cuts_per_month m) / 4 * 2 = 12 := by
  sorry

end helen_gas_usage_l134_134350


namespace centroid_plane_l134_134317

-- Definitions for the problem
structure TrihedralAngle (α β γ : ℝ) :=
  (vertex : ℝ)
  (point : ℝ)
  (edges_intersect_sphere : ℝ → ℝ → ℝ → Prop)

-- Conditions
variables {α β γ : ℝ}
variable (T : TrihedralAngle α β γ)
variable (N : ℝ)
variable (A B C : ℝ)

-- Condition: sphere passing through points S and N intersects the edges at A, B, C
axiom sphere_intersects_edges : T.edges_intersect_sphere A B C

-- Proof goal
theorem centroid_plane (h : T.edges_intersect_sphere A B C) : 
  ∃ (plane : ℝ → Prop), ∀ (T₁ T₂ T₃ : ℝ), T.edges_intersect_sphere T₁ T₂ T₃ → 
  centroid_of_triangle T₁ T₂ T₃ ∈ plane :=
sorry

def centroid_of_triangle (a b c : ℝ) : ℝ := (a + b + c) / 3

end centroid_plane_l134_134317


namespace part_a_exists_X_l134_134559

variables {P : Type} [EuclideanGeometry P]
variables {MN : Line P} {A B : P} (h : SameSide MN A B)

theorem part_a_exists_X (MN : Line P) (A B : P) (h : SameSide MN A B) :
  ∃ X : P, OnLine MN X ∧ Angle (A : P) X (MN : Line) = Angle (B : P) X (MN : Line) := sorry

end part_a_exists_X_l134_134559


namespace trig_simplification_l134_134082

theorem trig_simplification :
  (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180)) = 4 :=
sorry

end trig_simplification_l134_134082


namespace negation_equivalence_l134_134111

variable (x : ℝ)

def original_proposition := ∃ x : ℝ, x^2 - 3*x + 3 < 0

def negation_proposition := ∀ x : ℝ, x^2 - 3*x + 3 ≥ 0

theorem negation_equivalence : ¬ original_proposition ↔ negation_proposition :=
by 
  -- Lean doesn’t require the actual proof here
  sorry

end negation_equivalence_l134_134111


namespace probability_infection_l134_134093

theorem probability_infection :
  let P_A : ℝ := 0.5 in
  let P_B : ℝ := 0.3 in
  let P_C : ℝ := 0.2 in
  let P_D_given_A : ℝ := 0.95 in
  let P_D_given_B : ℝ := 0.90 in
  let P_D_given_C : ℝ := 0.85 in
  (P_D_given_A * P_A + P_D_given_B * P_B + P_D_given_C * P_C = 0.915) :=
by
  sorry

end probability_infection_l134_134093


namespace taller_pot_shadow_length_l134_134979

theorem taller_pot_shadow_length
  (height1 shadow1 height2 : ℝ)
  (h1 : height1 = 20)
  (h2 : shadow1 = 10)
  (h3 : height2 = 40) :
  ∃ shadow2 : ℝ, height2 / shadow2 = height1 / shadow1 ∧ shadow2 = 20 :=
by
  -- Since Lean requires proofs for existential statements,
  -- we add "sorry" to skip the proof.
  sorry

end taller_pot_shadow_length_l134_134979


namespace hyperbola_eccentricity_l134_134720

-- Define the hyperbola and its parameters
variables {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0)

-- Define the focus
def F := (2 * real.sqrt 2, 0)

-- Define the conditions for the perpendicular foot and the area
def perpendicular_foot (A : ℝ × ℝ) := A = (a, b)

def area_OAF : ℝ := 1 / 2 * a * b

-- Main statement to prove
theorem hyperbola_eccentricity :
  (area_OAF = 2) →
  (2 * real.sqrt 2) ^ 2 = a ^ 2 + b ^ 2 →
  ∃ e, e = real.sqrt 2 :=
by
  intros h_area h_focus
  -- Proof steps would go here...
  -- But for now, we use sorry to indicate the theorem can be proved given the conditions.
  sorry

end hyperbola_eccentricity_l134_134720


namespace count_n_digit_numbers_with_all_digits_l134_134296

theorem count_n_digit_numbers_with_all_digits (n : ℕ) : 
  let I := 3^n in
  let A1 := 2^n in
  let A2 := 2^n in
  let A3 := 2^n in
  let A1_inter_A2 := 1 in
  let A2_inter_A3 := 1 in
  let A3_inter_A1 := 1 in
  let A1_inter_A2_inter_A3 := 0 in
  I - A1 - A2 - A3 + A1_inter_A2 + A2_inter_A3 + A3_inter_A1 - A1_inter_A2_inter_A3 = 3^n - 3 * 2^n + 3 :=
by
  sorry

end count_n_digit_numbers_with_all_digits_l134_134296


namespace find_angle_A_max_perimeter_and_shape_l134_134752

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions from the given problem
axiom sides_opposite : b ∈ Ico 0 π ∧ c ∈ Ico 0 π ∧ a = opposite A ∧ b = opposite B ∧ c = opposite C
axiom tan_equation : (a^2 + c^2 - b^2) * tan B = √3 * (b^2 + c^2 - a^2)

-- Questions as Lean statements
theorem find_angle_A 
  (h1 : (a^2 + c^2 - b^2) * tan B = √3 * (b^2 + c^2 - a^2)) : A = π / 3 :=
sorry

theorem max_perimeter_and_shape 
  (h1 : (a^2 + c^2 - b^2) * tan B = √3 * (b^2 + c^2 - a^2)) 
  (h2 : a = 2) : ¬ (a = 2 ∧ b = 2 ∧ c = 2) ∧ L = 6 :=
sorry

end find_angle_A_max_perimeter_and_shape_l134_134752


namespace trail_length_is_20_km_l134_134784

-- Define the conditions and the question
def length_of_trail (L : ℝ) (hiked_percentage remaining_distance : ℝ) : Prop :=
  hiked_percentage = 0.60 ∧ remaining_distance = 8 ∧ 0.40 * L = remaining_distance

-- The statement: given the conditions, prove that length of trail is 20 km
theorem trail_length_is_20_km : ∃ L : ℝ, length_of_trail L 0.60 8 ∧ L = 20 := by
  -- Proof goes here
  sorry

end trail_length_is_20_km_l134_134784


namespace heavier_boxes_weight_l134_134762

theorem heavier_boxes_weight
  (x y : ℤ)
  (h1 : x ≥ 0)
  (h2 : x ≤ 30)
  (h3 : 10 * x + (30 - x) * y = 540)
  (h4 : 10 * x + (15 - x) * y = 240) :
  y = 20 :=
by
  sorry

end heavier_boxes_weight_l134_134762


namespace plane_perpendicular_l134_134191

variables {L : Type} [AffineSpace ℝ L]
variables (m n : line L) (α β : plane L)

-- Conditions from the problem
variables (m_diff_n : m ≠ n) (alpha_diff_beta : α ≠ β)

-- Condition D from the solution
variables (m_perp_alpha : m ⊆ α) (m_parallel_n : m ∥ n) (n_parallel_beta : n ∥ β)

-- Statement to be proven
theorem plane_perpendicular (h1 : m.perpendicular α) (h2 : m.parallel n) (h3 : n.parallel β) :
  α.perpendicular β := sorry

end plane_perpendicular_l134_134191


namespace problem_lean_version_l134_134182

theorem problem_lean_version (n : ℕ) : 
  (n > 0) ∧ (6^n - 1 ∣ 7^n - 1) ↔ ∃ k : ℕ, n = 4 * k :=
by
  sorry

end problem_lean_version_l134_134182


namespace find_a_from_cosine_l134_134323

theorem find_a_from_cosine
  (α : ℝ) (a : ℝ)
  (h1 : ∃ α, π/2 < α ∧ α < π)
  (h2 : (a, 1) ∈ ({p : ℝ × ℝ | ∃ α, π/2 < α ∧ α < π ∧ cos α * r = p.1 ∧ r = sqrt (p.1^2 + p.2^2)})  
  (h3 : cos α = (sqrt 2 / 4) * a) : a = -sqrt 7 :=
sorry

end find_a_from_cosine_l134_134323


namespace number_of_lattice_points_on_hyperbola_l134_134733

theorem number_of_lattice_points_on_hyperbola :
  {p : ℤ × ℤ | (p.1^2 - p.2^2 = 2500^2)}.finite.toFinset.card = 98 :=
by
  sorry

end number_of_lattice_points_on_hyperbola_l134_134733


namespace find_B_in_right_triangle_poly_l134_134125

noncomputable def right_triangle_poly_B: ℝ :=
  let u v w B : ℝ in
  ∃ (u v w: ℝ),  -- roots of the polynomial
    u + v + w = 14 ∧
    u * v * w = 84 ∧
    w^2 = u^2 + v^2 ∧
    B = u * v + u * w + v * w ∧
    B = 62

theorem find_B_in_right_triangle_poly : right_triangle_poly_B := 
  sorry

end find_B_in_right_triangle_poly_l134_134125


namespace sum_of_good_numbers_upto_2023_l134_134314

def a (n : ℕ) : ℝ := Real.log (n + 2) / Real.log 2 - Real.log (n + 1) / Real.log 2

def S (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i

def is_good (k : ℕ) : Prop := S k % 1 = 0

def sum_of_good_numbers (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), if is_good k then k else 0

theorem sum_of_good_numbers_upto_2023 : sum_of_good_numbers 2023 = 2026 := by
  sorry

end sum_of_good_numbers_upto_2023_l134_134314


namespace circle_line_tangent_l134_134373

theorem circle_line_tangent (m : ℝ) :
  ∃ (m : ℝ), (x^2 + y^2 = m^2) ∧ (x + 2y = √(3*m)) → m = 3/5 :=
by
  sorry

end circle_line_tangent_l134_134373


namespace Isabella_trip_theorem_l134_134448

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def Isabella_trip_exchange (d : ℕ) : Prop :=
  ∀ (exchange_rate_num exchange_rate_den spend : ℕ),
    exchange_rate_num = 11 → 
    exchange_rate_den = 8 → 
    spend = 80 →
    (exchange_rate_num * d / exchange_rate_den) - spend = d →
    sum_of_digits d = 6

theorem Isabella_trip_theorem : ∃ d : ℕ, Isabella_trip_exchange d :=
begin
  sorry
end

end Isabella_trip_theorem_l134_134448


namespace angle_x_is_58_l134_134018

-- Definitions and conditions based on the problem statement
variables {O A C D : Type} (x : ℝ)
variable [circular_center : O] -- O is the center of the circle
variable [radius_equal : AO = BO ∧ BO = CO] -- AO, BO, and CO are radii and hence equal
variable (angle_ACD_right : ∠ A C D = 90) -- ∠ACD is a right angle (subtended by diameter)
variable (angle_CDA : ∠ C D A = 42) -- ∠CDA is 42 degrees

-- Statement to be proven
theorem angle_x_is_58 (x : ℝ) : x = 58 := 
by sorry

end angle_x_is_58_l134_134018


namespace star_op_2004_equals_6012_l134_134261

-- Define the operation ※ for positive integers
def star_op (n : ℕ) : ℕ :=
  if n = 1 then 3 else 3 + star_op (n - 1)

-- Define the conditions
def one_star_one : Prop :=
  star_op 1 = 3

def recursive_star (n : ℕ) (h : n > 0) : Prop :=
  star_op (n + 1) = 3 + star_op n

-- The main theorem to prove
theorem star_op_2004_equals_6012 (h1 : one_star_one) (h2 : ∀ n > 0, recursive_star n) : 
  star_op 2004 = 6012 :=
sorry  -- Proof of this claim is not required


end star_op_2004_equals_6012_l134_134261


namespace angle_of_inclination_l134_134950

noncomputable def line_slope (a b : ℝ) : ℝ := 1  -- The slope of the line y = x + 1 is 1
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m -- angle of inclination is arctan of the slope

theorem angle_of_inclination (θ : ℝ) : 
  inclination_angle (line_slope 1 1) = 45 :=
by
  sorry

end angle_of_inclination_l134_134950


namespace total_wheels_in_garage_l134_134920

def bicycles: Nat := 3
def tricycles: Nat := 4
def unicycles: Nat := 7

def wheels_per_bicycle: Nat := 2
def wheels_per_tricycle: Nat := 3
def wheels_per_unicycle: Nat := 1

theorem total_wheels_in_garage (bicycles tricycles unicycles wheels_per_bicycle wheels_per_tricycle wheels_per_unicycle : Nat) :
  bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle = 25 := by
  sorry

end total_wheels_in_garage_l134_134920


namespace perimeter_of_fenced_square_field_l134_134936

-- Definitions for conditions
def num_posts : ℕ := 36
def spacing_between_posts : ℝ := 6 -- in feet
def post_width : ℝ := 1 / 2 -- 6 inches in feet

-- The statement to be proven
theorem perimeter_of_fenced_square_field :
  (4 * ((9 * spacing_between_posts) + (10 * post_width))) = 236 :=
by
  sorry

end perimeter_of_fenced_square_field_l134_134936


namespace charlie_steps_l134_134251

theorem charlie_steps (steps_per_run : ℕ) (runs : ℝ) (expected_steps : ℕ) :
  steps_per_run = 5350 →
  runs = 2.5 →
  expected_steps = 13375 →
  runs * steps_per_run = expected_steps :=
by intros; linarith; sorry

end charlie_steps_l134_134251


namespace find_a_smallest_period_max_value_l134_134713

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (a / 2) * sin (2 * x) - cos (2 * x)

-- The given point (π / 8, 0)
def given_point : ℝ × ℝ := (Real.pi / 8, 0)

-- Prove that the graph passes through the point (π / 8, 0), then a = √2
theorem find_a (a : ℝ) : f (Real.pi / 8) a = 0 → a = Real.sqrt 2 :=
by
  sorry

-- New function with a = √2
def new_f (x : ℝ) : ℝ := Real.sqrt 2 * sin (2 * x - Real.pi / 4)

-- Prove the smallest positive period is π
theorem smallest_period : ∃ T : ℝ, T > 0 ∧ ∀ x, new_f (x + T) = new_f x :=
by 
  use Real.pi
  sorry

-- Prove the maximum value of the function is √2
theorem max_value : ∃ M : ℝ, ∀ x, new_f x ≤ M ∧ (∃ y, new_f y = M) :=
by
  use Real.sqrt 2
  sorry

end find_a_smallest_period_max_value_l134_134713


namespace decimal_palindrome_multiple_l134_134591

def is_decimal_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem decimal_palindrome_multiple (n : ℕ) (h : ¬ (10 ∣ n)) : 
  ∃ m : ℕ, is_decimal_palindrome m ∧ m % n = 0 :=
by sorry

end decimal_palindrome_multiple_l134_134591


namespace tangent_line_eq_l134_134286

theorem tangent_line_eq (x y : ℝ) :
  (∀ x, y = sin x + exp (2*x)) ∧ (x = 0 ∧ y = 1) → 3*x - y + 1 = 0 :=
by
  sorry

end tangent_line_eq_l134_134286


namespace solution_set_correct_l134_134497

def inequality_solution_set : set ℝ := {x : ℝ | (3 + x) * (2 - x) < 0}

theorem solution_set_correct : 
  inequality_solution_set = {x : ℝ | x < -3 ∨ x > 2} :=
sorry

end solution_set_correct_l134_134497


namespace angle_x_is_9_degrees_l134_134049

theorem angle_x_is_9_degrees
  (O B C A D : Type)
  (hO_center : is_center O)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (hOB_eq_OC : distance O B = distance O C)
  (hAngle_BCO : angle B C O = 32)
  (hOA_eq_OC : distance O A = distance O C)
  (hAngle_CAD : angle C A D = 67)
  : angle x = 9 :=
sorry

end angle_x_is_9_degrees_l134_134049


namespace g_g_2_equals_226_l134_134364

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 4

theorem g_g_2_equals_226 : g (g 2) = 226 := by
  sorry

end g_g_2_equals_226_l134_134364


namespace third_side_not_one_l134_134377

theorem third_side_not_one (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c ≠ 1) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end third_side_not_one_l134_134377


namespace domain_of_f_l134_134527

def f (x : ℝ) : ℝ := (2 * x - 3) / (x^2 + 3 * x - 10)

theorem domain_of_f : 
  {x : ℝ | x ≠ -5 ∧ x ≠ 2} = {x : ℝ | x ∈ (Set.Ioo (-∞) (-5)) ∪ (Set.Ioo (-5) 2) ∪ (Set.Ioo 2 ∞)} := 
  sorry

end domain_of_f_l134_134527


namespace problem1_problem2_l134_134728

def vector (x y : ℝ) := (x, y : ℝ)

def a := vector 1 (-2)
def b := vector 3 4

-- Problem 1: Proving k = -1/3
theorem problem1 (k : ℝ) 
  (h1 : (3 * a - b) ∥ (a + k * b)) : 
  k = -(1 / 3) := 
sorry

-- Problem 2: Proving m = -1
theorem problem2 (m : ℝ) 
  (h2 : a ⊥ (m * a - b)) : 
  m = -1 := 
sorry

end problem1_problem2_l134_134728


namespace filming_time_for_4_weeks_episodes_l134_134518

theorem filming_time_for_4_weeks_episodes :
  (∀ (episode_length : ℕ) (percentage_longer : ℕ) (episodes_per_week : ℕ), 
  episode_length = 20 →
  percentage_longer = 50 →
  episodes_per_week = 5 →
  let filming_time_per_episode := episode_length + (percentage_longer * episode_length / 100) in
  let total_episodes := episodes_per_week * 4 in
  let total_film_time := total_episodes * filming_time_per_episode in
  total_film_time / 60 = 10) :=
by
  intros episode_length percentage_longer episodes_per_week h1 h2 h3,
  let filming_time_per_episode := episode_length + (percentage_longer * episode_length / 100),
  let total_episodes := episodes_per_week * 4,
  let total_film_time := total_episodes * filming_time_per_episode,
  have H : total_film_time = 600 := calc
    total_film_time = total_episodes * filming_time_per_episode : by rfl
    ... = (episodes_per_week * 4) * (episode_length + (percentage_longer * episode_length / 100)) : by rfl
    ... = (5 * 4) * (20 + (50 * 20 / 100)) : by rw [h1, h2, h3]
    ... = 20 * 30 : by norm_num
    ... = 600 : by norm_num,
  show 600 / 60 = 10 by norm_num,
  sorry

end filming_time_for_4_weeks_episodes_l134_134518


namespace income_mean_difference_l134_134467

theorem income_mean_difference {S' : ℝ} : 
  (S' + 1_200_000) / 500 - (S' + 120_000) / 500 = 2_160 := 
by 
  sorry

end income_mean_difference_l134_134467


namespace trains_time_to_clear_each_other_l134_134566

noncomputable def relative_speed (v1 v2 : ℝ) : ℝ :=
  v1 + v2

noncomputable def speed_to_m_s (v_kmph : ℝ) : ℝ :=
  v_kmph * 1000 / 3600

noncomputable def total_length (l1 l2 : ℝ) : ℝ :=
  l1 + l2

theorem trains_time_to_clear_each_other :
  ∀ (l1 l2 : ℝ) (v1_kmph v2_kmph : ℝ),
    l1 = 100 → l2 = 280 →
    v1_kmph = 42 → v2_kmph = 30 →
    (total_length l1 l2) / (speed_to_m_s (relative_speed v1_kmph v2_kmph)) = 19 :=
by
  intros l1 l2 v1_kmph v2_kmph h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end trains_time_to_clear_each_other_l134_134566


namespace williams_farm_tax_l134_134280

variables (T : ℝ)
variables (tax_collected : ℝ := 3840)
variables (percentage_williams_land : ℝ := 0.5)
variables (percentage_taxable_land : ℝ := 0.25)

theorem williams_farm_tax : (percentage_williams_land * tax_collected) = 1920 := by
  sorry

end williams_farm_tax_l134_134280


namespace coefficients_sum_is_neg_one_l134_134358

theorem coefficients_sum_is_neg_one :
  (∃ (a : Fin 2015 → ℤ), 
    (∀ x : ℝ, (1 - 2 * x) ^ 2014 = ∑ i in Finset.range 2015, a i * x ^ i) ∧
    1 + (∑ i in Finset.range 1 (λ i, a (i + 1) / 2 ^ (i + 1))) = 0) → 
  (∑ i in Finset.range 1 (λ i, a (i + 1) / 2 ^ (i + 1))) = -1 :=
sorry

end coefficients_sum_is_neg_one_l134_134358


namespace transformation_correct_l134_134772

-- Define the initial quadratic function
def initial_function (x : ℝ) : ℝ := (x + 2)^2 - 3

-- Define the transformation: shift 1 unit to the left
def shift_left (f : ℝ → ℝ) (h : ℝ) := λ x, f (x + h)

-- Define the transformation: shift 2 units upwards
def shift_up (f : ℝ → ℝ) (k : ℝ) := λ x, f x + k

-- Apply the transformations to the initial function
def transformed_function := shift_up (shift_left initial_function 1) 2

-- The final expected function
def expected_function (x : ℝ) : ℝ := (x + 3)^2 - 1

-- State the theorem
theorem transformation_correct :
  ∀ x, transformed_function x = expected_function x :=
by 
  -- Placeholder for proof
  sorry

end transformation_correct_l134_134772


namespace perpendicular_lines_c_value_l134_134270

theorem perpendicular_lines_c_value :
  ∀ (c : ℝ), (3 * 4 * c = -1) → c = -1 / 12 :=
begin
  assume c,
  assume h,
  sorry
end

end perpendicular_lines_c_value_l134_134270


namespace lattice_point_in_pentagon_l134_134307

theorem lattice_point_in_pentagon 
  (A B C D E : ℤ × ℤ) 
  (Hconvex : convex_hull ℝ {A, B, C, D, E}.val = set.univ)
  (Hdistinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E) 
  (A1 B1 C1 D1 E1 : ℚ × ℚ)
  (Hdiagonals : A1 ∈ line_intersection (line A C) (line B D) ∧
                B1 ∈ line_intersection (line A D) (line B E) ∧
                C1 ∈ line_intersection (line A E) (line B C) ∧
                D1 ∈ line_intersection (line B E) (line C D) ∧
                E1 ∈ line_intersection (line C E) (line D A)) : 
  ∃ (P : ℤ × ℤ), P ∈ convex_hull ℝ {A1, B1, C1, D1, E1}.val :=
  sorry

end lattice_point_in_pentagon_l134_134307


namespace area_ratio_greater_than_two_ninths_l134_134490

variable {α : Type*} [LinearOrder α] [LinearOrderedField α]

def area_triangle (A B C : α) : α := sorry -- Placeholder for the area function
noncomputable def triangle_division (A B C P Q R : α) : Prop :=
  -- Placeholder for division condition
  -- Here you would check that P, Q, and R divide the perimeter of triangle ABC into three equal parts
  sorry

theorem area_ratio_greater_than_two_ninths (A B C P Q R : α) :
  triangle_division A B C P Q R → area_triangle P Q R > (2 / 9) * area_triangle A B C :=
by
  sorry -- The proof goes here

end area_ratio_greater_than_two_ninths_l134_134490


namespace points_on_line_l134_134114

-- Define the slope function
def slope (p1 p2 : ℝ × ℝ) := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define whether a point lies on the line defined by two points
def on_line (p1 p2 p : ℝ × ℝ) := p.2 = (slope p1 p2) * (p.1 - p1.1) + p1.2

-- Given two points that define the line
def p1 := (8, 2) : ℝ × ℝ
def p2 := (2, -10) : ℝ × ℝ

-- Define the points to check
def point1 := (5, -4) : ℝ × ℝ
def point2 := (4, -6) : ℝ × ℝ
def point3 := (10, 6) : ℝ × ℝ
def point4 := (0, -14) : ℝ × ℝ
def point5 := (1, -12) : ℝ × ℝ

-- Define the theorem statement
theorem points_on_line :
  on_line p1 p2 point1 ∧
  on_line p1 p2 point2 ∧
  on_line p1 p2 point3 ∧
  on_line p1 p2 point4 ∧
  on_line p1 p2 point5 :=
by sorry -- Proof omitted

end points_on_line_l134_134114


namespace parts_from_blanks_l134_134587

theorem parts_from_blanks (initial_blanks : ℕ) (parts_per_blank : ℕ) (new_blank_from_parts : ℕ)
  (h1 : initial_blanks = 64) (h2 : parts_per_blank = 1) (h3 : new_blank_from_parts = 8) :
  ∑ i in (finset.range 3), 
    (if i = 0 then 64
    else if i = 1 then 8
    else if i = 2 then 1
    else 0) * 
    parts_per_blank = 73 :=
by
  sorry

end parts_from_blanks_l134_134587


namespace length_of_BC_l134_134399

theorem length_of_BC (AB AC AM : ℝ) (hAB : AB = 5) (hAC : AC = 8) (hAM : AM = 4.5) : 
  ∃ BC, BC = Real.sqrt 97 :=
by
  sorry

end length_of_BC_l134_134399


namespace data_relationship_in_leap_year_l134_134820

theorem data_relationship_in_leap_year 
  (total_values : ℕ)
  (occurrences1_to_29 occurrences30 occurrences31 : ℕ)
  (values : list ℕ)
  (h_total_values : total_values = 366)
  (h_occurrences1_to_29 : occurrences1_to_29 = 12)
  (h_occurrences30 : occurrences30 = 11)
  (h_occurrences31 : occurrences31 = 7)
  (h_values : values = list.repeat 1 12 ++ list.repeat 2 12 ++ list.repeat 3 12 ++ list.repeat 4 12 
             ++ list.repeat 5 12 ++ list.repeat 6 12 ++ list.repeat 7 12 ++ list.repeat 8 12 
             ++ list.repeat 9 12 ++ list.repeat 10 12 ++ list.repeat 11 12 ++ list.repeat 12 12 
             ++ list.repeat 13 12 ++ list.repeat 14 12 ++ list.repeat 15 12 ++ list.repeat 16 12 
             ++ list.repeat 17 12 ++ list.repeat 18 12 ++ list.repeat 19 12 ++ list.repeat 20 12 
             ++ list.repeat 21 12 ++ list.repeat 22 12 ++ list.repeat 23 12 ++ list.repeat 24 12 
             ++ list.repeat 25 12 ++ list.repeat 26 12 ++ list.repeat 27 12 ++ list.repeat 28 12 
             ++ list.repeat 29 12 ++ list.repeat 30 11 ++ list.repeat 31 7):
  let μ := (12 * 435 + 330 + 217) / 366,
      M := (values.nth_le 182 sorry + values.nth_le 183 sorry) / 2,
      modes := list.range' 1 29,
      d := (modes.nth_le 13 sorry + modes.nth_le 14 sorry) / 2
  in  d < μ ∧ μ < M :=
by
  sorry

end data_relationship_in_leap_year_l134_134820


namespace sequence_sum_square_l134_134825

theorem sequence_sum_square (n : ℕ) (h : n > 0) : 
  (∑ k in Finset.range n, k + 1) + (∑ k in Finset.range (n-1), n - 1 - k + 1) = n^2 := by 
  sorry

end sequence_sum_square_l134_134825


namespace coefficient_of_x3y2_in_binomial_expansion_l134_134098

theorem coefficient_of_x3y2_in_binomial_expansion :
  let f : ℕ → ℕ → ℚ := λ n k, Mathlib.Mathlib.Function.iterate (@polynomial.C _ CommRing.toRing (Polynomial Ring) _ Polynomial.Ring.polynomialRing Polynomial.C) (Polynomial.C (Bitrangle R))) (Binomial.expansion kFunny_coeff_of_x3y2_ _).val
  f 5 3 + Polynomial.C (neg_snd_part x3_coeff_poly_commute 3) - 10.0] sorry 

end coefficient_of_x3y2_in_binomial_expansion_l134_134098


namespace charlie_steps_in_running_session_l134_134244

variables (m_steps_3km : ℕ) (times_field : ℕ)
variables (distance_1_field : ℕ) (steps_per_km : ℕ)

-- Conditions
def charlie_steps : ℕ := 5350
def field_distance : ℕ := 3
def run_times : ℚ := 2.5

-- Statement we need to prove
theorem charlie_steps_in_running_session : 
  let distance_ran := run_times * field_distance in
  let total_steps := (charlie_steps * distance_ran) / field_distance in
  total_steps = 13375 := 
by simp [charlie_steps, field_distance, run_times]; sorry

end charlie_steps_in_running_session_l134_134244


namespace parallelogram_area_l134_134283

-- Define the two vectors u and v
def u : ℝ × ℝ × ℝ := (2, 4, -1)
def v : ℝ × ℝ × ℝ := (3, -2, 5)

-- Statement of the theorem
theorem parallelogram_area :
  let cross_product := (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)
  let magnitude := real.sqrt (cross_product.1 ^ 2 + cross_product.2 ^ 2 + cross_product.3 ^ 2)
  magnitude = real.sqrt 749 :=
sorry

end parallelogram_area_l134_134283


namespace even_sine_function_phi_eq_pi_div_2_l134_134875
open Real

theorem even_sine_function_phi_eq_pi_div_2 (φ : ℝ) (h : 0 ≤ φ ∧ φ ≤ π)
    (even_f : ∀ x : ℝ, sin (x + φ) = sin (-x + φ)) : φ = π / 2 :=
sorry

end even_sine_function_phi_eq_pi_div_2_l134_134875


namespace angle_x_is_58_l134_134014

-- Definitions and conditions based on the problem statement
variables {O A C D : Type} (x : ℝ)
variable [circular_center : O] -- O is the center of the circle
variable [radius_equal : AO = BO ∧ BO = CO] -- AO, BO, and CO are radii and hence equal
variable (angle_ACD_right : ∠ A C D = 90) -- ∠ACD is a right angle (subtended by diameter)
variable (angle_CDA : ∠ C D A = 42) -- ∠CDA is 42 degrees

-- Statement to be proven
theorem angle_x_is_58 (x : ℝ) : x = 58 := 
by sorry

end angle_x_is_58_l134_134014


namespace systematic_sampling_interval_l134_134215

theorem systematic_sampling_interval {S : ℕ} (hS : S = 883) {n : ℕ} {k : ℕ} 
  (hSampleSize : 80) (hEquation : S = hSampleSize * k + n) : 
  (k = 11 ∧ n = 3) :=
sorry

end systematic_sampling_interval_l134_134215


namespace right_trapezoid_area_l134_134214

def area_of_trapezoid (a b h : ℕ) : ℕ :=
  (a + b) * h / 2

theorem right_trapezoid_area :
  ∀ (b h : ℕ), b = 10 → h = 10 → area_of_trapezoid 25 b h = 175 :=
by {
  intros b h b_eq h_eq,
  rw [b_eq, h_eq],
  norm_num,
  sorry
}

end right_trapezoid_area_l134_134214


namespace increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l134_134545

noncomputable def fA (x : ℝ) : ℝ := -x
noncomputable def fB (x : ℝ) : ℝ := (2/3)^x
noncomputable def fC (x : ℝ) : ℝ := x^2
noncomputable def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function_fA : ¬∀ x y : ℝ, x < y → fA x < fA y := sorry
theorem increasing_function_fB : ¬∀ x y : ℝ, x < y → fB x < fB y := sorry
theorem increasing_function_fC : ¬∀ x y : ℝ, x < y → fC x < fC y := sorry
theorem increasing_function_fD : ∀ x y : ℝ, x < y → fD x < fD y := sorry

end increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l134_134545


namespace triangle_equilateral_l134_134001

open Triangle

variable {ABC : Type} [Triangle ABC]
variables {A B C D E F G : Point} 
variables {AFGE BDGF CEGD : Quadrilateral}

/-- Given points D, E, F on sides BC, CA, AB of triangle ABC respectively, and lines AD, BE, CF 
concur at point G, and circles can be inscribed in quadrilaterals AFGE, BDGF, CEGD such that 
each two of them have a common point -/
theorem triangle_equilateral (h_D_on_BC : PointOnLine D B C)
                            (h_E_on_CA : PointOnLine E C A)
                            (h_F_on_AB : PointOnLine F A B)
                            (h_concurrent : Concurrent A D G B E G C F G)
                            (h_circles : ExistsCommonIncenter AFGE BDGF CEGD) :
                            EquilateralTriangle ABC :=
                                sorry

end triangle_equilateral_l134_134001


namespace complement_of_angle_l134_134360

def angle := ℝ

def degrees_to_angle (deg: ℕ) (min: ℕ) : angle := deg + min / 60.0

theorem complement_of_angle (α : angle) (h : α = degrees_to_angle 29 45) :
  degrees_to_angle 60 15 = 90 - α :=
by
  sorry

end complement_of_angle_l134_134360


namespace find_A_l134_134890

theorem find_A (A B : ℕ) (hcfAB lcmAB : ℕ)
  (hcf_cond : Nat.gcd A B = hcfAB)
  (lcm_cond : Nat.lcm A B = lcmAB)
  (B_val : B = 169)
  (hcf_val : hcfAB = 13)
  (lcm_val : lcmAB = 312) :
  A = 24 :=
by 
  sorry

end find_A_l134_134890


namespace calculation_is_correct_l134_134530

-- Define the numbers involved in the calculation
def a : ℝ := 12.05
def b : ℝ := 5.4
def c : ℝ := 0.6

-- Expected result of the calculation
def expected_result : ℝ := 65.67

-- Prove that the calculation is correct
theorem calculation_is_correct : (a * b + c) = expected_result :=
by
  sorry

end calculation_is_correct_l134_134530


namespace _l134_134586

variables (u v w : ℝ^3)
variables (h₁ : v = 2 * (AB : ℝ^3))
variables (h₂ : w = AD)
variables (h₃ : u = (1/2) * (AE : ℝ^3))
variables (h₄ : dot v w = 0) -- v is orthogonal to w

lemma parallelepiped_theorem 
  (AG BH CE DF AB AD AE : ℝ^3)
  (AG_sq BH_sq CE_sq DF_sq : ℝ)
  (AB_sq AD_sq AE_sq : ℝ):
  (AG_sq = ∥ u + v + w ∥^2) →
  (BH_sq = ∥ u - v + w ∥^2) →
  (CE_sq = ∥ -u + v + w ∥^2) →
  (DF_sq = ∥ u + v - w ∥^2) →
  (AB_sq = ∥ v ∥^2) →
  (AD_sq = ∥ w ∥^2) →
  (AE_sq = ∥ u ∥^2) →
  (AG_sq + BH_sq + CE_sq + DF_sq) / (2 * AB_sq + AD_sq + (1/2) * AE_sq) = 4 :=
by {
  assume hAGsq hBHsq hCEsq hDFsq hABsq hADsq hAEsq,
  sorry
}

end _l134_134586


namespace solve_lambda_positive_int_l134_134673

noncomputable def exists_solution_for_lambda (x y z v λ : ℕ) : Prop :=
x > 0 ∧ y > 0 ∧ z > 0 ∧ v > 0 ∧ x^2 + y^2 + z^2 + v^2 = λ * x * y * z * v

theorem solve_lambda_positive_int (λ : ℕ) :
  (∃ x y z v : ℕ, exists_solution_for_lambda x y z v λ) ↔ λ = 1 ∨ λ = 4 :=
by
  sorry

end solve_lambda_positive_int_l134_134673


namespace second_largest_minus_smallest_l134_134505

theorem second_largest_minus_smallest : let s := {35, 68, 57, 95}
  let second_largest := 68
  let smallest := 35
  second_largest - smallest = 33 := by
  let s := {35, 68, 57, 95}
  let second_largest := max 57 68
  let smallest := min 35 57
  exact rfl

end second_largest_minus_smallest_l134_134505


namespace repeating_decimals_subtraction_l134_134160

theorem repeating_decimals_subtraction :
  let x := (246/999 : ℚ)
  let y := (135/999 : ℚ)
  let z := (9753/9999 : ℚ)
  (x - y - z) = (-8647897/9989001 : ℚ) := 
by
  sorry

end repeating_decimals_subtraction_l134_134160


namespace arrange_supervillains_l134_134864

-- Define a supervillain as an index within the range of 1 to 2n
def supervillain (n : ℕ) := fin (2 * n)

-- Predicate that represents the enemy relationship between two supervillains
def enemies (n : ℕ) (a b : supervillain n) : Prop

-- Condition: Each supervillain has at most (n-1) enemies 
def at_most_n_minus_1_enemies (n : ℕ) (v : supervillain n) : Prop :=
  ∃ (enemies_set : set (supervillain n)), 
    finset.card enemies_set ≤ n - 1 ∧ 
    ∀ (e ∈ enemies_set), enemies n v e

-- Predicate to check if a given arrangement is valid
def valid_arrangement (n : ℕ) (arr : list (supervillain n)) : Prop :=
  ∀ i, (i < 2 * n) → ¬ enemies n (arr.nth_le i (by linarith)) (arr.nth_le ((i + 1) % (2 * n)) (by linarith))

-- The main statement to be proved: existence of a valid arrangement
theorem arrange_supervillains (n : ℕ) (cond : ∀ v : supervillain n, at_most_n_minus_1_enemies n v) :
  ∃ arr : list (supervillain n), valid_arrangement n arr :=
sorry

end arrange_supervillains_l134_134864


namespace group_B_population_calculation_l134_134732

variable {total_population : ℕ}
variable {sample_size : ℕ}
variable {sample_A : ℕ}
variable {total_B : ℕ}

theorem group_B_population_calculation 
  (h_total : total_population = 200)
  (h_sample_size : sample_size = 40)
  (h_sample_A : sample_A = 16)
  (h_sample_B : sample_size - sample_A = 24) :
  total_B = 120 :=
sorry

end group_B_population_calculation_l134_134732


namespace minimum_distance_from_circle_to_line_l134_134704

-- Distance formula for a point to a line
def point_to_line_dist (x₀ y₀ A B C : ℝ) : ℝ :=
  (abs (A * x₀ + B * y₀ + C)) / (sqrt (A^2 + B^2))

-- Given circle equation definition
def circle (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 2

-- Given line equation definition
def line (x y : ℝ) : Prop :=
  x + y - 5 = 0

-- The mathematically equivalent proof problem in Lean 4
theorem minimum_distance_from_circle_to_line :
  ∃ P : ℝ × ℝ, circle P.1 P.2 ∧ line P.1 P.2 →
  (∀ x y : ℝ, circle x y → point_to_line_dist x y 1 1 (-5) - sqrt 2 ≥ 0)
∧ (∃ x y : ℝ, circle x y ∧ point_to_line_dist x y 1 1 (-5) - sqrt 2 = 2 * sqrt 2)
:= by
  sorry

end minimum_distance_from_circle_to_line_l134_134704


namespace garden_area_increase_l134_134212

theorem garden_area_increase :
  let length := 80
  let width := 20
  let additional_fence := 60
  let original_area := length * width
  let original_perimeter := 2 * (length + width)
  let total_fence := original_perimeter + additional_fence
  let side_of_square := total_fence / 4
  let square_area := side_of_square * side_of_square
  square_area - original_area = 2625 :=
by
  sorry

end garden_area_increase_l134_134212


namespace altitude_BD_median_BE_l134_134332

noncomputable theory

def point := ℝ × ℝ

-- Definition for the three points A, B, and C
def A : point := (3, -4)
def B : point := (6, 0)
def C : point := (-5, 2)

-- Predicate representing the equation of a line in the form ax + by + c = 0
def equation_of_line (a b c : ℝ) (P : point) : Prop :=
  let (x,y) := P in
  a * x + b * y + c = 0

-- The two theorem statements
theorem altitude_BD : equation_of_line 4 (-3) (-24) B := by
  sorry

theorem median_BE : equation_of_line 1 (-7) (-6) B := by
  sorry

end altitude_BD_median_BE_l134_134332


namespace problem_statement_l134_134807

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  (x^(n+1) - x^(-n-1)) / (x - x^(-1))

theorem problem_statement (n : ℕ) (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  let y := x + x⁻¹ in
  (n > 1 → f (n + 1) x = y * f n x - f (n - 1) x) ∧
  f n x =
  if even n then
    y^n + ∑ i in finset.range (n / 2 + 1), (-1) ^ i * (nat.choose (n-2*i) n) * y^(n - 2 * i)
  else
    y^n - ∑ i in finset.range (n / 2 + 1), (-1) ^ i * (nat.choose (n-i) n) * y^(n - 2 * i) :=
sorry

end problem_statement_l134_134807


namespace bamboo_pole_height_l134_134575

noncomputable def bamboo_height (L : ℝ) := ∃ (h : ℝ),
  (∀ a b : ℝ, a = b) → L = 24 ∧
  (∀ c d : ℝ, c = d) → a = 7 + 7 * Real.sqrt 2 ∧
  (∀ e f : ℝ, e = f) → h = 16 ± 4 * Real.sqrt 2

theorem bamboo_pole_height :
  bamboo_height 24 :=
sorry

end bamboo_pole_height_l134_134575


namespace complex_fraction_sum_zero_l134_134303

section complex_proof
open Complex

theorem complex_fraction_sum_zero (z1 z2 : ℂ) (hz1 : z1 = 1 + I) (hz2 : z2 = 1 - I) :
  (z1 / z2) + (z2 / z1) = 0 := by
  sorry
end complex_proof

end complex_fraction_sum_zero_l134_134303


namespace angle_x_degrees_l134_134032

theorem angle_x_degrees 
  (O : Point)
  (A B C D : Point)
  (hC : Circle O)
  (hOB : O ∈ Segment O B)
  (hOC : O ∈ Segment O C)
  (hBCO : ∠ B C O = 32)
  (hACD : ∠ A C D = 90)
  (hCAD : ∠ C A D = 67) :
  ∠ B C A = 9 := 
  sorry

end angle_x_degrees_l134_134032


namespace perimeter_of_fenced_square_field_l134_134934

-- Definitions for conditions
def num_posts : ℕ := 36
def spacing_between_posts : ℝ := 6 -- in feet
def post_width : ℝ := 1 / 2 -- 6 inches in feet

-- The statement to be proven
theorem perimeter_of_fenced_square_field :
  (4 * ((9 * spacing_between_posts) + (10 * post_width))) = 236 :=
by
  sorry

end perimeter_of_fenced_square_field_l134_134934


namespace altitude_BD_eqn_median_BE_eqn_l134_134330

variables {Point : Type}
variables {x y : Point → ℝ}

def A : Point := ⟨3, -4⟩
def B : Point := ⟨6, 0⟩
def C : Point := ⟨-5, 2⟩

-- Proving part 1: Equation of the line containing the altitude BD on side AC
theorem altitude_BD_eqn (A B C : Point) : 
  let D := (∃ p : ℝ × ℝ, ∃ k : ℝ, y(B) - y(C) = x(B) - (-1 / k) ∧ y(B) - (k * (x(B) - 6)) = p.2)
  in ∃ k : ℝ, k = 4 / 3 ∧ (4 * x(B) - 3 * y(B) - 24 = 0) :=
  sorry

-- Proving part 2: Equation of the line containing the median BE on side AC
theorem median_BE_eqn (A B C : Point) : 
  let E := ((x(A) + x(C)) / 2, (y(A) + y(C)) / 2)
  in E = (-1, -1) ∧ (x(B) - 1/7 * (x(B) - 6) = y(E)) :=
  sorry

end altitude_BD_eqn_median_BE_eqn_l134_134330


namespace max_volume_of_box_l134_134163

theorem max_volume_of_box (sheetside : ℝ) (cutside : ℝ) (volume : ℝ) 
  (h1 : sheetside = 6) 
  (h2 : ∀ (x : ℝ), 0 < x ∧ x < (sheetside / 2) → volume = x * (sheetside - 2 * x)^2) : 
  cutside = 1 :=
by
  sorry

end max_volume_of_box_l134_134163


namespace angle_x_degrees_l134_134046

noncomputable def center_of_circle (O : Point) := true
noncomputable def radii (O B C A : Point) := (dist O B = dist O C) ∧ (dist O A = dist O C)
noncomputable def isosceles_triangle (O B C : Point) := (∠ B C O = 32) ∧ (∠ O B C = 32)
noncomputable def isosceles_base_angle (O A C : Point) := ∠ O A C = ∠ A O C = 23
noncomputable def angle_sum_property (O A C : Point) := ∠ A O C = 90

theorem angle_x_degrees
  (O B C A : Point) (h_radii : radii O B C A)
  (h_isosceles_triangle : isosceles_triangle O B C)
  (h_isosceles_base : isosceles_base_angle O A C)
  (h_sum_property : angle_sum_property O A C) :
  ∠ B C O - ∠ O A C = 9 := by sorry

end angle_x_degrees_l134_134046


namespace rectangular_to_cylindrical_l134_134630

theorem rectangular_to_cylindrical (x y z r θ : ℝ) (h₁ : x = 3) (h₂ : y = -3 * Real.sqrt 3) (h₃ : z = 2)
  (h₄ : r = Real.sqrt (3^2 + (-3 * Real.sqrt 3)^2)) 
  (h₅ : r > 0) 
  (h₆ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₇ : θ = Float.pi * 5 / 3) : 
  (r, θ, z) = (6, 5 * Float.pi / 3, 2) :=
sorry

end rectangular_to_cylindrical_l134_134630


namespace random_variable_is_zero_with_prob_one_l134_134805

universe u
open ProbabilityTheory

-- Define the Lean 4 statement
theorem random_variable_is_zero_with_prob_one {Ω : Type u} {X Y : Ω → ℝ}
(HX1 : integrable X) (HY1 : integrable Y) (HYX : ae_eq_fun (E[Y | X]) (fun ω => 0))
(HYX_Y : ae_eq_fun (E[Y | fun ω => X ω + Y ω]) (fun ω => 0)) :
  ae_eq_fun (fun ω => Y ω) (fun ω => 0) :=
sorry

end random_variable_is_zero_with_prob_one_l134_134805


namespace f_strictly_increasing_on_intervals_tan_alpha_minus_pi_over_4_l134_134338

noncomputable def f (x : ℝ) : ℝ :=
  sin (2 * x + π / 3) + cos (2 * x - π / 6)

theorem f_strictly_increasing_on_intervals :
  ∀ k : ℤ, ∀ x : ℝ, -5 * π / 12 + k * π ≤ x ∧ x ≤ π / 12 + k * π → 
  f (x) is strictly increasing on [(-5 * π / 12 + k * π), (π / 12 + k * π)] :=
sorry

theorem tan_alpha_minus_pi_over_4 :
  ∀ α : ℝ, α ∈ Ioo (π / 2) π → 
  (2 * sin (α - π / 6) = 6 / 5) → 
  tan (α - π / 4) = 31 / 17 :=
sorry

end f_strictly_increasing_on_intervals_tan_alpha_minus_pi_over_4_l134_134338


namespace number_of_lattice_points_on_hyperbola_l134_134734

theorem number_of_lattice_points_on_hyperbola :
  {p : ℤ × ℤ | (p.1^2 - p.2^2 = 2500^2)}.finite.toFinset.card = 98 :=
by
  sorry

end number_of_lattice_points_on_hyperbola_l134_134734


namespace intersection_M_N_l134_134502

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (-1, 1) + x • (1, 2)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ := (1, -2) + y • (2, 3)

def set_M := {a : ℝ × ℝ | ∃ x : ℝ, a = vector_a x}
def set_N := {a : ℝ × ℝ | ∃ y : ℝ, a = vector_b y}

theorem intersection_M_N : M = {(ℝ, ℝ) | (ℝ, ℝ) == (-13, -23)} := sorry

end intersection_M_N_l134_134502


namespace find_a_l134_134320

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (x * x - 4 <= 0) → (2 * x + a <= 0)) ↔ (a = -4) := by
  sorry

end find_a_l134_134320


namespace rhombus_perimeter_l134_134473

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let s := (d1 / 2)^2 + (d2 / 2)^2 in
  4 * Real.sqrt s = 52 :=
by
  sorry

end rhombus_perimeter_l134_134473


namespace orthocenter_circumcenter_coincide_triangle_similarity_l134_134831

universe u
variable {α : Real} {A B C A1 B1 C1 A' B' C' : Type u}

-- Defining the points and the properties
def is_triangle (T : Type u) := ∃ (p1 p2 p3 : T), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1
def on_side_or_extension (p q r : Type u): Prop := sorry -- Placeholder predicate

def angle_eq_alpha (angle1 angle2 angle3 : Real) (α : Real) : Prop :=
  angle1 = α ∧ angle2 = α ∧ angle3 = α

variable [triangle : is_triangle ABC] -- Assume ABC is a triangle
variable [on_side_or_extension A B C A1]
variable [on_side_or_extension B C A B1]
variable [on_side_or_extension C A B C1]
variable [angle_eq_alpha (angle (C, C1, A, B) (A, A1, B, C) (B, B1, C, A)) α]
variable [intersect (A, A1) (B, B1) B']
variable [intersect (B, B1) (C, C1) C']
variable [intersect (C, C1) (A, A1) A']

-- The proofs to be provided
theorem orthocenter_circumcenter_coincide : orthocenter ABC = circumcenter A'B'C' := sorry

theorem triangle_similarity :
  similar_triangles A'B'C' ABC ∧ similarity_coefficient A'B'C' ABC = 2 * Real.cos α := sorry

end orthocenter_circumcenter_coincide_triangle_similarity_l134_134831


namespace function_inequality_l134_134708

variable (f : ℝ → ℝ)
variable [Differentiable ℝ f]

theorem function_inequality (h : ∀ x : ℝ, (2 - x) * f x + x * (deriv f x) < 0) : ∀ x : ℝ, f x < 0 :=
by
  sorry

end function_inequality_l134_134708


namespace largest_number_by_changing_digit_to_9_l134_134165

theorem largest_number_by_changing_digit_to_9 :
  let original_number : ℝ := 0.12345678 in
  let changed_numbers := [
    0.92345678, -- Change the first digit to 9
    0.19345678, -- Change the second digit to 9
    0.12945678, -- Change the third digit to 9
    0.12395678, -- Change the fourth digit to 9
    0.12349678, -- Change the fifth digit to 9
    0.12345978, -- Change the sixth digit to 9
    0.12345698, -- Change the seventh digit to 9
    0.12345679  -- Change the eighth digit to 9
  ] in
  (∀ n ∈ changed_numbers, n ≤ 0.92345678) :=
begin
  intros original_number changed_numbers n hn,
  fin_cases hn,
  iterate 1 {refl},
  repeat {norm_num, exact dec_trivial}
end

#eval largest_number_by_changing_digit_to_9

end largest_number_by_changing_digit_to_9_l134_134165


namespace fraction_unit_fraction_decomposition_l134_134060

theorem fraction_unit_fraction_decomposition 
  {m n : ℕ} (h1 : 0 < m) (h2 : m < n) :
  ∃ (q : ℕ → ℕ) (r : ℕ), 
    (∀ k, 1 ≤ k → k ≤ r → ∃ j, j = k - 1 → q(k) > q(j)) ∧
    (∀ k, 2 ≤ k → k ≤ r → q(k) % q(k-1) = 0) ∧ 
    (frac m n = ∑ i in range 1 (r+1), 1 / q(i)) :=
sorry

end fraction_unit_fraction_decomposition_l134_134060


namespace angle_x_degrees_l134_134047

noncomputable def center_of_circle (O : Point) := true
noncomputable def radii (O B C A : Point) := (dist O B = dist O C) ∧ (dist O A = dist O C)
noncomputable def isosceles_triangle (O B C : Point) := (∠ B C O = 32) ∧ (∠ O B C = 32)
noncomputable def isosceles_base_angle (O A C : Point) := ∠ O A C = ∠ A O C = 23
noncomputable def angle_sum_property (O A C : Point) := ∠ A O C = 90

theorem angle_x_degrees
  (O B C A : Point) (h_radii : radii O B C A)
  (h_isosceles_triangle : isosceles_triangle O B C)
  (h_isosceles_base : isosceles_base_angle O A C)
  (h_sum_property : angle_sum_property O A C) :
  ∠ B C O - ∠ O A C = 9 := by sorry

end angle_x_degrees_l134_134047


namespace convert_to_cylindrical_l134_134627

theorem convert_to_cylindrical (x y z : ℝ) (r θ : ℝ) 
  (h₀ : x = 3) 
  (h₁ : y = -3 * Real.sqrt 3) 
  (h₂ : z = 2) 
  (h₃ : r > 0) 
  (h₄ : 0 ≤ θ) 
  (h₅ : θ < 2 * Real.pi) 
  (h₆ : r = Real.sqrt (x^2 + y^2)) 
  (h₇ : x = r * Real.cos θ) 
  (h₈ : y = r * Real.sin θ) : 
  (r, θ, z) = (6, 5 * Real.pi / 3, 2) :=
by
  -- Proof goes here
  sorry

end convert_to_cylindrical_l134_134627


namespace circle_with_PQ_passes_through_F2_l134_134699

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1 ^ 2) / 2 + p.2 ^ 2 = 1 }

def is_tangent (k m : ℝ) :=
  let line : ℝ → ℝ := λ x, k * x + m
  ∃ P : ℝ × ℝ, P ∈ ellipse ∧ line P.1 = P.2 ∧ ∀ Q ∈ ellipse, Q ≠ P → line Q.1 ≠ Q.2

def intersects_at_2 (k m : ℝ) : ℝ × ℝ := (2, k * 2 + m)

theorem circle_with_PQ_passes_through_F2
  (k m : ℝ)
  (h_tangent : is_tangent k m) :
  let P := classical.some h_tangent
  let Q := intersects_at_2 k m
  let F2 := (1, 0)
  F2 ∈ {(p : ℝ × ℝ) | p.1 = (P.1 + Q.1) / 2 ∧ p.2 = (P.2 + Q.2) / 2}
  :=
sorry

end circle_with_PQ_passes_through_F2_l134_134699


namespace sabrina_sales_min_requirement_l134_134066

noncomputable def commission_per_sale : ℝ := 0.15 * 1500
noncomputable def salary_difference : ℝ := 90000 - 45000
noncomputable def required_sales : ℕ := nat_ceil (salary_difference / commission_per_sale)

theorem sabrina_sales_min_requirement:
  required_sales = 200 :=
by
  sorry

end sabrina_sales_min_requirement_l134_134066


namespace total_crayons_is_12_l134_134510

-- Definitions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Goal to prove
theorem total_crayons_is_12 : initial_crayons + added_crayons = 12 :=
by
  sorry

end total_crayons_is_12_l134_134510


namespace truth_probability_l134_134599

variables (P_A P_B P_AB : ℝ)

theorem truth_probability (h1 : P_B = 0.60) (h2 : P_AB = 0.48) : P_A = 0.80 :=
by
  have h3 : P_AB = P_A * P_B := sorry  -- Placeholder for the rule: P(A and B) = P(A) * P(B)
  rw [h2, h1] at h3
  sorry

end truth_probability_l134_134599


namespace probability_of_two_negative_roots_eq_l134_134845

theorem probability_of_two_negative_roots_eq (p : ℝ) (h : p ∈ set.Icc 0 5) :
  let quadratic_has_two_negative_roots := 
    (4 * p^2 - 4 * (3 * p - 2) ≥ 0) ∧
    (-2 * p < 0) ∧
    (3 * p - 2 > 0)
  in (set.Icc 0 5).measure
       {p | quadratic_has_two_negative_roots} / 
       (set.Icc 0 5).measure (set.Icc 0 5) = 2/3 :=
by admit

end probability_of_two_negative_roots_eq_l134_134845


namespace area_of_ABCD_l134_134770

variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Conditions
def m_angle_BC : Real := 110
def m_angle_CB : Real := 110
def AB : Real := 4
def BC : Real := 6
def CD : Real := 7

-- Main theorem: The area of quadrilateral ABCD is 33 * sin(70 degrees)
theorem area_of_ABCD : 
  ∃ AB BC CD, 
  (m_angle_BC = 110 ∧ m_angle_CB = 110 ∧ 
  AB = 4 ∧ BC = 6 ∧ CD = 7) → 
  (area_of_quadrilateral A B C D = 33 * real.sin (70 * pi / 180)) :=
by sorry

end area_of_ABCD_l134_134770


namespace point_coordinates_l134_134773

def point_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0 

theorem point_coordinates (m : ℝ) 
  (h1 : point_in_second_quadrant (-m-1) (2*m+1))
  (h2 : |2*m + 1| = 5) : (-m-1, 2*m+1) = (-3, 5) :=
sorry

end point_coordinates_l134_134773


namespace sum_of_segments_l134_134829

theorem sum_of_segments 
  (A B C D E F G : Type) 
  (dist_AG : ℝ) (dist_BF : ℝ) (dist_CE : ℝ)
  (h1 : dist_AG = 23)
  (h2 : dist_BF = 17)
  (h3 : dist_CE = 9) :
  let total_length := 6 * dist_AG + 4 * dist_BF + 2 * dist_CE in
  total_length = 224 :=
by
  unfold total_length
  sorry

end sum_of_segments_l134_134829


namespace sum_of_squares_mod_13_l134_134955

theorem sum_of_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 1 := sorry

end sum_of_squares_mod_13_l134_134955


namespace odd_expression_proof_l134_134741

theorem odd_expression_proof (n : ℤ) : Odd (n^2 + n + 5) :=
by 
  sorry

end odd_expression_proof_l134_134741


namespace output_correct_l134_134113

-- Define the initial values and assignments
def initial_a : ℕ := 1
def initial_b : ℕ := 2
def initial_c : ℕ := 3

-- Perform the assignments in sequence
def after_c_assignment : ℕ := initial_b
def after_b_assignment : ℕ := initial_a
def after_a_assignment : ℕ := after_c_assignment

-- Final values after all assignments
def final_a := after_a_assignment
def final_b := after_b_assignment
def final_c := after_c_assignment

-- Theorem statement
theorem output_correct :
  final_a = 2 ∧ final_b = 1 ∧ final_c = 2 :=
by {
  -- Proof is omitted
  sorry
}

end output_correct_l134_134113


namespace quadrilateral_non_acute_angles_l134_134687

variables {ABC : Type*} [triangle: simplex ABC] 
variable {point : Type*}
variables {S : point} [interior_point : interior_point S triangle]
variables {A1 B1 C1 : point}
variables [intersect_AS_BC: intersects AS BC A1]
variables [intersect_BS_CA: intersects BS CA B1]
variables [intersect_CS_AB: intersects CS AB C1]

theorem quadrilateral_non_acute_angles :
  (Some_quad_with_two_non_acute: ∃ quadrilateral : Type*, 
    (quadrilateral = AB1SC1 ∨ quadrilateral = C1SA1B ∨ quadrilateral = A1SB1C) ∧ 
    two_non_acute_angles quadrilateral) :=
begin
  sorry
end

end quadrilateral_non_acute_angles_l134_134687


namespace graph_transformation_identity_l134_134134

theorem graph_transformation_identity :
  ∀ x : ℝ, (λ x, sqrt 2 * cos x) x = (λ x, sqrt 2 * sin (2 * (x / 2 - π/4) + π/4)) x :=
by
  intro x
  sorry

end graph_transformation_identity_l134_134134


namespace find_value_of_x_l134_134004

-- Definitions used in conditions
variable (O B C A : Point) -- points on the circle
variable (OB OC OA : ℝ) -- Radii of the circle
variable (OB_eq_OC : OB = OC)
variable (angle_BCO : ℝ) -- Given angle ∠BCO
variable (angle_DAC : ℝ) -- Given angle ∠DAC
variable (right_angled_ACD : angle_DAC + 90 = 157) -- Triangle ACD is right-angled & 180 - 23 = 157

variable (x : ℝ) -- angle x

-- Setting specific values for given angles in degrees
noncomputable def angle_BCO_value : angle_BCO = 32
noncomputable def angle_DAC_value : angle_DAC = 67

-- What we want to prove
theorem find_value_of_x : x = 9 :=
sorry

end find_value_of_x_l134_134004


namespace least_palindrome_divisible_by_25_l134_134952

theorem least_palindrome_divisible_by_25 : ∃ (n : ℕ), 
  (10^4 ≤ n ∧ n < 10^5) ∧
  (∀ (a b c : ℕ), n = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a) ∧
  n % 25 = 0 ∧
  n = 10201 :=
by
  sorry

end least_palindrome_divisible_by_25_l134_134952


namespace num_valid_pairs_l134_134288

theorem num_valid_pairs : 
  (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 120 ∧ (a.toReal + 1 / b.toReal) / (1 / a.toReal + b.toReal) = 17) → 
  ((∃ p : Finset (ℕ × ℕ), p.card = 6 ∧ ∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ 
  a + b ≤ 120 ∧ (a.toReal + 1 / b.toReal) / (1 / a.toReal + b.toReal) = 17 ↔ (a, b) ∈ p)) := 
sorry

end num_valid_pairs_l134_134288


namespace warehouse_length_l134_134242

theorem warehouse_length (L W : ℕ) (times supposed_times : ℕ) (total_distance : ℕ)
  (h1 : W = 400)
  (h2 : supposed_times = 10)
  (h3 : times = supposed_times - 2)
  (h4 : total_distance = times * (2 * L + 2 * W))
  (h5 : total_distance = 16000) :
  L = 600 := by
  sorry

end warehouse_length_l134_134242


namespace increasing_function_l134_134542

def fA (x : ℝ) : ℝ := -x
def fB (x : ℝ) : ℝ := (2 / 3) ^ x
def fC (x : ℝ) : ℝ := x ^ 2
def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function (x y : ℝ) (h : x < y) : fD x < fD y := sorry

end increasing_function_l134_134542


namespace neg_universal_to_existential_l134_134885

theorem neg_universal_to_existential :
  (¬ (∀ x : ℝ, 2 * x^4 - x^2 + 1 < 0)) ↔ (∃ x : ℝ, 2 * x^4 - x^2 + 1 ≥ 0) :=
by 
  sorry

end neg_universal_to_existential_l134_134885


namespace expected_balls_original_positions_l134_134455

/-- The expected number of balls that occupy their original positions after three successive transpositions -/
theorem expected_balls_original_positions :
  let num_balls := 7
  let transpositions := 3
  let prob_original_position := 127 / 343
  (num_balls * prob_original_position) = 2.6 :=
by
  let num_balls := 7
  let transpositions := 3
  let prob_original_position := 127 / 343
  have h : num_balls * prob_original_position = 889 / 343 := by
    sorry
  have h_correct : (889 / 343 : ℝ) ≈ 2.6 := by
    sorry
  exact h_correct

end expected_balls_original_positions_l134_134455


namespace sheila_attends_picnic_l134_134078

theorem sheila_attends_picnic :
  let P_rain := 0.5
  let P_sunny := 0.5
  let P_attends_if_rain := 0.3
  let P_attends_if_sunny := 0.9
  let P_remembers := 0.9
  (P_rain * P_attends_if_rain * P_remembers + P_sunny * P_attends_if_sunny * P_remembers) = 0.54 :=
by
  -- Calculating the probability for rain scenario
  let P_rain_attends := P_rain * P_attends_if_rain * P_remembers

  -- Calculating the probability for sunny scenario
  let P_sunny_attends := P_sunny * P_attends_if_sunny * P_remembers

  -- Summing both probabilities to get the total probability Sheila attends
  let P_sheila_attends := P_rain_attends + P_sunny_attends

  -- Proving the final probability is 54%
  sorry

end sheila_attends_picnic_l134_134078


namespace possible_values_of_a_l134_134723

theorem possible_values_of_a (a : ℚ) : 
  (a^2 = 9 * 16) ∨ (16 * a = 81) ∨ (9 * a = 256) → 
  a = 12 ∨ a = -12 ∨ a = 81 / 16 ∨ a = 256 / 9 :=
by
  intros h
  sorry

end possible_values_of_a_l134_134723


namespace ball_rebound_percentage_l134_134832

theorem ball_rebound_percentage (P : ℝ) 
  (h₁ : 100 + 2 * 100 * P + 2 * 100 * P^2 = 250) : P = 0.5 := 
by 
  sorry

end ball_rebound_percentage_l134_134832


namespace isolating_and_counting_bacteria_process_l134_134491

theorem isolating_and_counting_bacteria_process
  (soil_sampling : Prop)
  (spreading_dilution_on_culture_medium : Prop)
  (decompose_urea : Prop) :
  (soil_sampling ∧ spreading_dilution_on_culture_medium ∧ decompose_urea) →
  (Sample_dilution ∧ Selecting_colonies_that_can_grow ∧ Identification) :=
sorry

end isolating_and_counting_bacteria_process_l134_134491


namespace no_scalene_triangle_similar_to_IHO_l134_134780

theorem no_scalene_triangle_similar_to_IHO (ABC : Triangle) (h_scalene : ABC.is_scalene) 
  (I : Point) (H : Point) (O : Point) 
  (hI : I = incenter ABC) 
  (hH : H = orthocenter ABC) 
  (hO : O = circumcenter ABC) :
  ¬ (ABC ∼ (Triangle.mk I H O)) :=
begin
  sorry
end

end no_scalene_triangle_similar_to_IHO_l134_134780


namespace CalculateValue_l134_134239

noncomputable def sumNumerator (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (2024 - k) / (k + 1)

noncomputable def sumDenominator (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, 1 / (k + 2)

theorem CalculateValue : 
  (sumNumerator 2023) / (sumDenominator 2023) = 2024 :=
by
  sorry

end CalculateValue_l134_134239


namespace angle_x_is_9_degrees_l134_134054

theorem angle_x_is_9_degrees
  (O B C A D : Type)
  (hO_center : is_center O)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (hOB_eq_OC : distance O B = distance O C)
  (hAngle_BCO : angle B C O = 32)
  (hOA_eq_OC : distance O A = distance O C)
  (hAngle_CAD : angle C A D = 67)
  : angle x = 9 :=
sorry

end angle_x_is_9_degrees_l134_134054


namespace arithmetic_sequence_n_value_l134_134689

theorem arithmetic_sequence_n_value (a_1 d a_nm1 n : ℤ) (h1 : a_1 = -1) (h2 : d = 2) (h3 : a_nm1 = 15) :
    a_nm1 = a_1 + (n - 2) * d → n = 10 :=
by
  intros h
  sorry

end arithmetic_sequence_n_value_l134_134689


namespace angle_x_l134_134023

-- Define the conditions
variables (O A C D B : Point)
variable h : circle_center O
variable h1 : on_circumference A C D O
variable h2 : diameter AC
variable h3 : ∠ CDA = 42

-- The problem statement we want to prove
theorem angle_x (x : ℝ) : x = 58 :=
begin
  -- This is where the solution steps would go, but for now, we leave it as sorry
  sorry
end

end angle_x_l134_134023


namespace solve_for_y_l134_134851

theorem solve_for_y (y : ℕ) (h : 5 * (2 ^ y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l134_134851


namespace angle_BAD_obtuse_l134_134899

-- Define parallelogram, midpoints, and segments
variables {A B C D N M : Type} [EuclideanGeometry α] 
variables [parallelogram A B C D]
variables (N_midpoint : midpoint N B C) (M_midpoint : midpoint M C D)
variable (H : segment_length A N = 2 * segment_length A M)

-- Statement of the proof problem
theorem angle_BAD_obtuse 
  (parallelogram ABCD : parallelogram A B C D) 
  (N_midpoint : midpoint N B C) 
  (M_midpoint : midpoint M C D) 
  (H : segment_length A N = 2 * segment_length A M) : 
  angle_obtuse (angle B A D) :=
sorry

end angle_BAD_obtuse_l134_134899


namespace average_pages_per_day_l134_134551

variable (total_pages : ℕ := 160)
variable (pages_read : ℕ := 60)
variable (days_left : ℕ := 5)

theorem average_pages_per_day : (total_pages - pages_read) / days_left = 20 := by
  sorry

end average_pages_per_day_l134_134551


namespace probability_at_least_twice_as_many_daughters_as_sons_l134_134439

-- Definitions extracted from the conditions
def num_children : ℕ := 8
def prob_each_child_is_female : ℝ := 0.5
def prob_each_child_is_male : ℝ := 0.5

-- The main proof problem statement
theorem probability_at_least_twice_as_many_daughters_as_sons :
  let total_gender_combinations := Nat.pow 2 num_children,
      favorable_combinations := (Nat.choose num_children 0) + (Nat.choose num_children 1) + (Nat.choose num_children 2)
  in
  favorable_combinations / total_gender_combinations = 37 / 256 :=
by
  sorry

end probability_at_least_twice_as_many_daughters_as_sons_l134_134439


namespace carolyn_sum_correct_l134_134462

def initial_sequence := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def carolyn_removes : List ℕ := [4, 8, 10, 9]

theorem carolyn_sum_correct : carolyn_removes.sum = 31 :=
by
  sorry

end carolyn_sum_correct_l134_134462


namespace find_value_of_x_l134_134007

-- Definitions used in conditions
variable (O B C A : Point) -- points on the circle
variable (OB OC OA : ℝ) -- Radii of the circle
variable (OB_eq_OC : OB = OC)
variable (angle_BCO : ℝ) -- Given angle ∠BCO
variable (angle_DAC : ℝ) -- Given angle ∠DAC
variable (right_angled_ACD : angle_DAC + 90 = 157) -- Triangle ACD is right-angled & 180 - 23 = 157

variable (x : ℝ) -- angle x

-- Setting specific values for given angles in degrees
noncomputable def angle_BCO_value : angle_BCO = 32
noncomputable def angle_DAC_value : angle_DAC = 67

-- What we want to prove
theorem find_value_of_x : x = 9 :=
sorry

end find_value_of_x_l134_134007


namespace f_of_f_neg_one_l134_134716

noncomputable def f : ℝ → ℝ :=
  λ x, if x < 0 then x + 2 else 3 * x - 1

theorem f_of_f_neg_one : f (f (-1)) = 2 := by
  sorry

end f_of_f_neg_one_l134_134716


namespace BANANAS_arrangement_l134_134646

open Nat

theorem BANANAS_arrangement :
  let total_letters := 7
  let freq_A := 3
  let freq_N := 2
  fact total_letters / (fact freq_A * fact freq_N) = 420 :=
by 
  sorry

end BANANAS_arrangement_l134_134646


namespace sin_square_sum_arcsin_l134_134680

open Real Trigonometric

theorem sin_square_sum_arcsin
  (α β : ℝ)
  (h : arcsin (sin α + sin β) + arcsin (sin α - sin β) = π / 2) :
  sin α ^ 2 + sin β ^ 2 = 1 / 2 := 
by
  sorry

end sin_square_sum_arcsin_l134_134680


namespace number_of_arrangements_BANANAS_l134_134639

theorem number_of_arrangements_BANANAS : 
  let total_permutations := 7!
      a_repeats := 3!
      n_repeats := 2!
  in total_permutations / (a_repeats * n_repeats) = 420 :=
by
  sorry

end number_of_arrangements_BANANAS_l134_134639


namespace erin_tv_marathon_hours_l134_134656

noncomputable def erin_tv_marathon_time {e1 e2 e3 b e4 e5 e6 b2 e7 e8 e9 : ℕ} : ℕ :=
    let pride_prejudice_time := (e1 * e2) + (e1 - 1) * b
    let breaking_bad_time := (e3 * e4) + (e3 - 1) * e5 + (e3/3).ceil * e6
    let stranger_things_time := (e7 * e8) + (e7 - 1) * e9
    let total_time := pride_prejudice_time + breaking_bad_time + stranger_things_time + b2 * e7/3 * 120
    total_time

theorem erin_tv_marathon_hours :
  let pride_prejudice_episodes : ℕ := 6
  let pride_prejudice_episode_length : ℕ := 50
  let pride_prejudice_break_time : ℕ := 10
  let breaking_bad_episodes : ℕ := 62
  let breaking_bad_episode_length : ℕ := 47
  let breaking_bad_short_break_time : ℕ := 10
  let breaking_bad_long_break_time : ℕ := 60
  let stranger_things_episodes : ℕ := 33
  let stranger_things_episode_length : ℕ := 51
  let stranger_things_break_time : ℕ := 15
  let longer_break_between_series : ℕ := 2 * 120 -- 2 hours
  total_time := erin_tv_marathon_time pride_prejudice_episodes pride_prejudice_episode_length pride_prejudice_break_time
                                     breaking_bad_episodes breaking_bad_episode_length breaking_bad_short_break_time
                                     breaking_bad_long_break_time stranger_things_episodes stranger_things_episode_length
                                     stranger_things_break_time longer_break_between_series in
  total_time / 60 = 125.62 :=
sorry

end erin_tv_marathon_hours_l134_134656


namespace proof_l134_134321

noncomputable def proof_problem : ℕ :=
  let M := (((20! : ℤ) / (2! * 18! : ℤ)) + (20! / (3! * 17!)) + (20! / (4! * 16!)) + (20! / (5! * 15!)) + 
            (20! / (6! * 14!)) + (20! / (7! * 13!)) + (20! / (8! * 12!)) + (20! / (9! * 11!)) + 
            (20! / (10! * 10!))).div 20
  let M_div_100 : ℚ := M / 100
  int.floor M_div_100

theorem proof : proof_problem = 262 :=
  sorry

end proof_l134_134321


namespace three_teams_not_played_each_other_l134_134382

theorem three_teams_not_played_each_other (teams : Finset ℕ) (rounds : List (Finset (ℕ × ℕ))) :
  teams.card = 18 →
  rounds.length = 8 →
  ∀ (r ∈ rounds), (∀ (p ∈ r), p.1 ∈ teams ∧ p.2 ∈ teams) →
  (∀ (r s : ℕ × ℕ), r ≠ s → ¬(s ∈ rounds)) →
  (∃ (A B C : ℕ), A ∈ teams ∧ B ∈ teams ∧ C ∈ teams ∧ 
  (¬ ∃ r ∈ rounds, (A, B) ∈ r ∨ (A, C) ∈ r ∨ (B, C) ∈ r)) :=
sorry

end three_teams_not_played_each_other_l134_134382


namespace BANANAS_arrangement_l134_134648

open Nat

theorem BANANAS_arrangement :
  let total_letters := 7
  let freq_A := 3
  let freq_N := 2
  fact total_letters / (fact freq_A * fact freq_N) = 420 :=
by 
  sorry

end BANANAS_arrangement_l134_134648


namespace consecutive_numbers_difference_l134_134498

theorem consecutive_numbers_difference :
  ∃ (n : ℕ), (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → (n + 5 - n = 5) :=
by {
  sorry
}

end consecutive_numbers_difference_l134_134498


namespace third_side_triangle_l134_134764

theorem third_side_triangle (a : ℝ) :
  (5 < a ∧ a < 13) → (a = 8) :=
sorry

end third_side_triangle_l134_134764


namespace angle_x_is_9_degrees_l134_134056

theorem angle_x_is_9_degrees
  (O B C A D : Type)
  (hO_center : is_center O)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (hOB_eq_OC : distance O B = distance O C)
  (hAngle_BCO : angle B C O = 32)
  (hOA_eq_OC : distance O A = distance O C)
  (hAngle_CAD : angle C A D = 67)
  : angle x = 9 :=
sorry

end angle_x_is_9_degrees_l134_134056


namespace blueberries_in_blue_box_l134_134177

theorem blueberries_in_blue_box (B S : ℕ) (h1 : S - B = 12) (h2 : S + B = 76) : B = 32 :=
sorry

end blueberries_in_blue_box_l134_134177


namespace a1_value_recursive_relation_general_formula_sum_of_first_n_terms_l134_134429

noncomputable def a_seq : ℕ → ℝ
| 0 := 3  -- Using a_1 = 3
| (n+1) := 2 * a_seq n + 3

def S (n : ℕ) := 2 * (a_seq n) - 3 * (n + 1)

theorem a1_value : a_seq 0 = 3 :=
by simp [a_seq]

theorem recursive_relation (n : ℕ) : a_seq (n+1) = 2 * a_seq n + 3 :=
by simp [a_seq]

/-- General formula for the sequence -/
theorem general_formula (n : ℕ) : a_seq (n+1) = 6 * 2^n - 3 :=
sorry

/-- Sum of the first n terms -/
theorem sum_of_first_n_terms (n : ℕ) : S n = 6 * 2^(n+1) - 3 * (n + 1) - 6 :=
sorry

end a1_value_recursive_relation_general_formula_sum_of_first_n_terms_l134_134429


namespace molecular_weight_bleach_l134_134487

theorem molecular_weight_bleach :
  let Na := 22.99
  let O := 16.00
  let Cl := 35.45
  let molecular_weight := Na + O + Cl
  molecular_weight = 74.44
:=
by
  let Na := 22.99
  let O := 16.00
  let Cl := 35.45
  let molecular_weight := Na + O + Cl
  sorry

end molecular_weight_bleach_l134_134487


namespace find_n_over_100_l134_134424

theorem find_n_over_100 : 
  let n := Nat.choose 48 2
  (n : ℚ) / 100 = 11.28 := sorry

end find_n_over_100_l134_134424


namespace total_books_l134_134432

theorem total_books (D Loris Lamont : ℕ) 
  (h1 : Loris + 3 = Lamont)
  (h2 : Lamont = 2 * D)
  (h3 : D = 20) : D + Loris + Lamont = 97 := 
by 
  sorry

end total_books_l134_134432


namespace sum_of_squares_mod_13_l134_134956

theorem sum_of_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 1 := sorry

end sum_of_squares_mod_13_l134_134956


namespace correct_option_l134_134972

theorem correct_option :
  (log 7 6 < log 6 7) ∧ ¬(1.01 ^ 3.4 > 1.01 ^ 3.5) ∧ ¬(3.5 ^ 0.3 < 3.4 ^ 0.3) ∧ ¬(log 0.4 4 < log 0.4 6) :=
by
  sorry

end correct_option_l134_134972


namespace cos_A_value_l134_134316

-- Definitions of triangle and altitudes
variables {a b c : ℝ}
variable (h₁ : ∃ (A B C : Type) (abc : A × B × C), true)
variable (altitude_a : ℝ := 1 / 2)
variable (altitude_b : ℝ := Real.sqrt 2 / 2)
variable (altitude_c : ℝ := 1)

-- Area expressions in terms of sides and altitudes
def area₁ (a : ℝ) := (1 / 2) * a * altitude_a
def area₂ (b : ℝ) := (1 / 2) * b * altitude_b
def area₃ (c : ℝ) := (1 / 2) * c * altitude_c

-- Relation between sides deduced from altitudes
def side_relation (a b c : ℝ) : Prop := (a = Real.sqrt 2 * b) ∧ (c = (1 / 2) * a) ∧ (b = (Real.sqrt 2 / 2) * a)

-- Law of Cosines for angle A in triangle ABC
def cos_A (a b c : ℝ) : ℝ := (b^2 + c^2 - a^2) / (2 * b * c)

-- Proof statement
theorem cos_A_value (a b c : ℝ) (h₁ : side_relation a b c) : cos_A a b c = - (Real.sqrt 2 / 4) :=
sorry

end cos_A_value_l134_134316


namespace deepak_meet_time_l134_134881

theorem deepak_meet_time :
  let C := 594 -- circumference in meters
  let v1 := 4.5 * 1000 / 60 -- Deepak's speed in m/min
  let v2 := 3.75 * 1000 / 60 -- His wife's speed in m/min
  let v_r := v1 + v2 -- relative speed
  v_r ≠ 0 → C / v_r = 4.32 :=
by
  intro h
  unfold C v1 v2 v_r
  norm_num at h ⊢
  exact h

end deepak_meet_time_l134_134881


namespace triangle_no_solution_l134_134730

def angleSumOfTriangle : ℝ := 180

def hasNoSolution (a b A : ℝ) : Prop :=
  A >= angleSumOfTriangle

theorem triangle_no_solution {a b A : ℝ} (ha : a = 181) (hb : b = 209) (hA : A = 121) :
  hasNoSolution a b A := sorry

end triangle_no_solution_l134_134730


namespace angle_x_is_58_l134_134013

-- Definitions and conditions based on the problem statement
variables {O A C D : Type} (x : ℝ)
variable [circular_center : O] -- O is the center of the circle
variable [radius_equal : AO = BO ∧ BO = CO] -- AO, BO, and CO are radii and hence equal
variable (angle_ACD_right : ∠ A C D = 90) -- ∠ACD is a right angle (subtended by diameter)
variable (angle_CDA : ∠ C D A = 42) -- ∠CDA is 42 degrees

-- Statement to be proven
theorem angle_x_is_58 (x : ℝ) : x = 58 := 
by sorry

end angle_x_is_58_l134_134013


namespace probability_of_s_in_statistics_l134_134397

theorem probability_of_s_in_statistics :
  let totalLetters := 10
  let count_s := 3
  (count_s / totalLetters : ℚ) = 3 / 10 := by
  sorry

end probability_of_s_in_statistics_l134_134397


namespace positive_difference_of_squares_l134_134897

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end positive_difference_of_squares_l134_134897


namespace arrangement_of_bananas_l134_134643

-- Define the constants for the number of letters and repetitions in the word BANANAS.
def num_letters : ℕ := 7
def count_A : ℕ := 3
def count_N : ℕ := 2
def factorial (n : ℕ) := nat.factorial n

-- The number of ways to arrange the letters of the word BANANAS.
noncomputable def num_ways_to_arrange := 
  (factorial num_letters) / (factorial count_A * factorial count_N)

theorem arrangement_of_bananas : 
  num_ways_to_arrange = 420 :=
sorry

end arrangement_of_bananas_l134_134643


namespace calculation_order_l134_134609

-- Define the problem statement and conditions
theorem calculation_order :
  ∀ a b c : ℕ, c * (a + b) = c * a + c * b → 
  (a = 20) → (b = 30) → (c = 4) → 
  c * (a + b) = 200 :=
by
  intros a b c h ha hb hc 
  rw [ha, hb, hc]
  simp
  exact h
  sorry

end calculation_order_l134_134609


namespace angle_eq_l134_134333

-- Definitions
def ellipse (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity (a : ℝ) : Prop :=
  ∃ c, c = a / 2 ∧ c^2 = a^2 - 3 ∧ (c / a = 1 / 2)

def points (a : ℝ) : Prop :=
  ∃ F, F = (1, 0) ∧ A = (-a, 0) ∧ abs (-a - 1) = 3

def midpoint (A P : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + P.1) / 2, (A.2 + P.2) / 2)

def intersects (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = 4 ∧ y₂ = (4 * y₁) / (x₂ - 2)

def parallel (x₁ y₁ x₂ y₂: ℝ) : Prop :=
  x₁ = 4 ∧ y₂ = (4 * y₁) / (x₂ + 2)

-- Problem Statement
theorem angle_eq :
  ∀ a b x₁ y₁ A P F, ellipse a b x₁ y₁ →
  eccentricity a →
  points a →
  let D := (4, (4 * y₁) / (x₁ - 2)) in
  let E := (4, (4 * y₁) / (x₁ + 2)) in
  ∃ D E : ℝ × ℝ, intersects x₁ y₁ D.1 D.2 →
  parallel x₁ y₁ E.1 E.2 →
  ∃ M : ℝ × ℝ, midpoint A P = M →
  ∠D O F = ∠E O F := sorry

end angle_eq_l134_134333


namespace fish_price_relation_l134_134235

variables (b_c m_c b_v m_v : ℝ)

axiom cond1 : 3 * b_c + m_c = 5 * b_v
axiom cond2 : 2 * b_c + m_c = 3 * b_v + m_v

theorem fish_price_relation : 5 * m_v = b_c + 2 * m_c :=
by
  sorry

end fish_price_relation_l134_134235


namespace gcd_assoc_gcd_three_eq_gcd_assoc_l134_134062

open Int

theorem gcd_assoc {a b c : ℕ} : Nat.gcd a (Nat.gcd b c) = Nat.gcd (Nat.gcd a b) c := by
  sorry

theorem gcd_three_eq_gcd_assoc {a b c : ℕ} : Nat.gcd3 a b c = Nat.gcd (Nat.gcd a b) c := by
  sorry

end gcd_assoc_gcd_three_eq_gcd_assoc_l134_134062


namespace greatest_odd_factors_l134_134824

theorem greatest_odd_factors (n : ℕ) (h1 : n < 1000) (h2 : ∀ k : ℕ, k * k = n → (k < 32)) :
  n = 31 * 31 :=
by
  sorry

end greatest_odd_factors_l134_134824


namespace angle_x_degrees_l134_134044

noncomputable def center_of_circle (O : Point) := true
noncomputable def radii (O B C A : Point) := (dist O B = dist O C) ∧ (dist O A = dist O C)
noncomputable def isosceles_triangle (O B C : Point) := (∠ B C O = 32) ∧ (∠ O B C = 32)
noncomputable def isosceles_base_angle (O A C : Point) := ∠ O A C = ∠ A O C = 23
noncomputable def angle_sum_property (O A C : Point) := ∠ A O C = 90

theorem angle_x_degrees
  (O B C A : Point) (h_radii : radii O B C A)
  (h_isosceles_triangle : isosceles_triangle O B C)
  (h_isosceles_base : isosceles_base_angle O A C)
  (h_sum_property : angle_sum_property O A C) :
  ∠ B C O - ∠ O A C = 9 := by sorry

end angle_x_degrees_l134_134044


namespace exists_distinct_multiples_l134_134410

theorem exists_distinct_multiples (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 3) : 
  ∃ (a : Fin m → ℕ), 
    (∀ i, a i ∣ n - 1) ∧ 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    1 / n = (Finset.univ.sum (λ i, (-1) ^ (i.val) / a i)) :=
sorry

end exists_distinct_multiples_l134_134410


namespace smallest_two_digit_multiple_of_17_l134_134960

theorem smallest_two_digit_multiple_of_17 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ 17 ∣ n ∧ ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ 17 ∣ m → n ≤ m :=
begin
  use 17,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { use 1,
    norm_num },
  intros m h1 h2 h3,
  rw ← nat.dvd_iff_mod_eq_zero at *,
  have h4 := nat.mod_eq_zero_of_dvd h3,
  cases (nat.le_of_mod_eq_zero h4),
  { linarith [nat.le_of_dvd (dec_trivial) this] },
  { exfalso,
    linarith }
end

end smallest_two_digit_multiple_of_17_l134_134960


namespace ned_now_owns_6_games_l134_134441

theorem ned_now_owns_6_games (initial_games : ℕ) (games_given_away : ℕ) 
  (h_initial : initial_games = 19) 
  (h_given_away : games_given_away = 13) : 
  initial_games - games_given_away = 6 := 
  by 
    rw [h_initial, h_given_away]
    exact rfl

end ned_now_owns_6_games_l134_134441


namespace minimum_value_of_f_l134_134663

noncomputable def f (x : ℝ) : ℝ :=
  x + (1 / x) + (1 / (x + (1 / x))) + (1 / (x^2))

theorem minimum_value_of_f : ∃ x > 0, f x = 3.5 :=
begin
  sorry
end

end minimum_value_of_f_l134_134663


namespace find_remaining_area_l134_134555

-- Given conditions
def semicircle (diameter : ℝ) : Prop := true

def point_on_diameter (M AB : ℝ) : Prop := true

def two_semicircles_cut_out (AM MB : ℝ) (AB : ℝ) : Prop := AM + MB = AB

def chord_length_perpendicular (M : ℝ) (chord_length : ℝ) : Prop := chord_length = 2 * real.sqrt 7

-- Remaining area calculation
def remaining_area (AB : ℝ) : ℝ :=
  let radius := AB / 2
  let chord_height := real.sqrt 7
  let r_squared := radius * radius + 7
  let total_area := (1 / 2) * real.pi * r_squared
  let smaller_semicircles_area := real.pi * (r_squared / 4)
  total_area - 2 * (smaller_semicircles_area / 2)

-- Proof problem
theorem find_remaining_area (AB M AM MB : ℝ)
  (h1 : semicircle AB)
  (h2 : point_on_diameter M AB)
  (h3 : two_semicircles_cut_out AM MB AB)
  (h4 : chord_length_perpendicular M (2 * real.sqrt 7)) :
  21.99 ≈ remaining_area AB :=
by sorry

end find_remaining_area_l134_134555


namespace bob_spends_more_time_l134_134838

def pages := 760
def time_per_page_bob := 45
def time_per_page_chandra := 30
def total_time_bob := pages * time_per_page_bob
def total_time_chandra := pages * time_per_page_chandra
def time_difference := total_time_bob - total_time_chandra

theorem bob_spends_more_time : time_difference = 11400 :=
by
  sorry

end bob_spends_more_time_l134_134838


namespace find_y_l134_134862

theorem find_y (k : ℝ) (h1 : 4^3 * real.sqrt 4 = k) : 
  ∃ y : ℝ, 16^3 * real.sqrt 16 = k → y = 128 := by
  use 128
  intro h2
  simp only [real.sqrt_eq_rpow, complex.cpow_nat_cast] at h1 h2
  norm_num at h1 h2
  have h3 : k = 128 := by linarith
  rw h3 at h2
  simp only [complex.cpow_nat_cast, real.sqrt_eq_rpow] at h2
  norm_num at h2
  sorry

end find_y_l134_134862


namespace partition_cities_l134_134126

-- Define the type for cities and airlines.
variable (City : Type) (Airline : Type)

-- Define the number of cities and airlines
variable (n k : ℕ)

-- Define a relation to represent bidirectional direct flights
variable (flight : Airline → City → City → Prop)

-- Define the condition: Some pairs of cities are connected by exactly one direct flight operated by one of the airline companies
-- or there are no such flights between them.
axiom unique_flight : ∀ (a : Airline) (c1 c2 : City), flight a c1 c2 → ¬ (∃ (a' : Airline), flight a' c1 c2 ∧ a' ≠ a)

-- Define the condition: Any two direct flights operated by the same company share a common endpoint
axiom shared_endpoint :
  ∀ (a : Airline) (c1 c2 c3 c4 : City), flight a c1 c2 → flight a c3 c4 → (c1 = c3 ∨ c1 = c4 ∨ c2 = c3 ∨ c2 = c4)

-- The main theorem to prove
theorem partition_cities :
  ∃ (partition : City → Fin (k + 2)), ∀ (c1 c2 : City) (a : Airline), flight a c1 c2 → partition c1 ≠ partition c2 :=
sorry

end partition_cities_l134_134126


namespace slope_of_line_l134_134895

-- Define the condition that the line equation is given.
def line_equation (x y : ℝ) := y - 3 = 4 * (x + 1)

-- Prove that the slope of the line is 4 given the line equation.
theorem slope_of_line : ∀ x y : ℝ, line_equation x y → 4 := 
sorry

end slope_of_line_l134_134895


namespace cos_gamma_l134_134419

theorem cos_gamma (Q : ℝ × ℝ × ℝ) (h : 0 < Q.1 ∧ 0 < Q.2 ∧ 0 < Q.3)
  (α β γ : ℝ) (cos_alpha_eq : real.cos α = 2 / 5) (cos_beta_eq : real.cos β = 1 / 4) :
  real.cos γ = real.sqrt 311 / 20 :=
by
  sorry

end cos_gamma_l134_134419


namespace bucket_fill_problem_l134_134176

theorem bucket_fill_problem (C : ℝ) (H : C > 0) : 
  let total_capacity := 25 * C in
  let new_bucket_capacity := (2 / 5) * C in
  let num_new_buckets := total_capacity / new_bucket_capacity in
  num_new_buckets.toNat = 63 :=
by
  let total_capacity := 25 * C
  let new_bucket_capacity := (2 / 5) * C
  let num_new_buckets := total_capacity / new_bucket_capacity
  have h : num_new_buckets = 125 / 2 := by
    calc
      total_capacity / new_bucket_capacity
        = (25 * C) / ((2 / 5) * C) : by rfl
        ... = 25 / (2 / 5) : by field_simp [C, H]
        ... = 25 * 5 / 2 : by rw div_eq_mul_inv
        ... = 125 / 2 : by norm_num
  show num_new_buckets.toNat = 63 from 
    sorry

end bucket_fill_problem_l134_134176


namespace math_problem_proof_l134_134365

noncomputable def X : ℝ :=
  (213 * 16) + (1.6 * 2.13)

theorem math_problem_proof : 
  213 * 16 = 3408 → 
  X = (213 * 16 + (1.6 * 2.13)) → 
  X - ((5 / 2) * 1.25) + (3 * Real.log 8 / Real.log 2) + Real.sin (Real.pi / 2) = 3418.283 :=
by
  intros h1 h2
  rw [h1] at h2
  have hX : X = 3411.408 := by sorry
  rw [hX, Real.log, Real.sin]
  sorry

end math_problem_proof_l134_134365


namespace coefficient_x3_in_expansion_l134_134775

theorem coefficient_x3_in_expansion : 
  ∀ (x : ℕ), (∃ c : ℕ, (x + 2) ^ 30 = (c * x^3 + ...) ↔ c = 4060 * 2^27) :=
sorry

end coefficient_x3_in_expansion_l134_134775


namespace number_of_arrangements_BANANAS_l134_134638

theorem number_of_arrangements_BANANAS : 
  let total_permutations := 7!
      a_repeats := 3!
      n_repeats := 2!
  in total_permutations / (a_repeats * n_repeats) = 420 :=
by
  sorry

end number_of_arrangements_BANANAS_l134_134638


namespace coefficient_of_x_cubed_l134_134661

theorem coefficient_of_x_cubed : 
  let coeff := (coeff x^3 in (polynomial.expand (x^2 - x - 2)^4)) in
  coeff = -40 :=
by
  sorry

end coefficient_of_x_cubed_l134_134661


namespace count_two_digit_numbers_with_odd_factors_eq_six_l134_134355

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def two_digit_numbers_with_odd_factors : ℕ :=
  finset.card { n | n ≥ 10 ∧ n < 100 ∧ is_perfect_square n}

theorem count_two_digit_numbers_with_odd_factors_eq_six : 
  two_digit_numbers_with_odd_factors = 6 :=
sorry

end count_two_digit_numbers_with_odd_factors_eq_six_l134_134355


namespace binomial_congruence_mod_p_l134_134808

open Nat

theorem binomial_congruence_mod_p (p : ℕ) [hp_prime : Prime p]
(n : ℕ) : 
  ∃ q : ℕ, q > 0 ∧ (n = q * (p - 1)) → binom n (p - 1) % p = 1
  ∨ (∀ q : ℕ, q > 0 ∧ (n ≠ q * (p - 1))) → binom n (p - 1) % p = 0 := 
by
  sorry

end binomial_congruence_mod_p_l134_134808


namespace point_on_angle_l134_134369

theorem point_on_angle (y : ℝ) : 
  (∀ P : ℝ × ℝ, P = (-1, y) → P_on_terminal_side_angle (-4 / 3 * real.pi) P) → y = real.sqrt 3 :=
by
  -- Assuming the point P is on the terminal side of the given angle
  sorry

end point_on_angle_l134_134369


namespace sum_of_fractions_equals_l134_134612

-- Define the fractions
def f1 := 3 / 10
def f2 := 2 / 100
def f3 := 8 / 1000
def f4 := 8 / 10000

-- State the theorem to prove that the sum of these fractions equals 0.3288
theorem sum_of_fractions_equals : f1 + f2 + f3 + f4 = 0.3288 := by
  sorry

end sum_of_fractions_equals_l134_134612


namespace pretty_number_characterization_l134_134291

def is_pretty (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ k ℓ : ℕ, k < n → ℓ < n → k > 0 → ℓ > 0 → 
    (n ∣ 2*k - ℓ ∨ n ∣ 2*ℓ - k)

theorem pretty_number_characterization :
  ∀ n : ℕ, is_pretty n ↔ (Prime n ∨ n = 6 ∨ n = 9 ∨ n = 15) :=
by
  sorry

end pretty_number_characterization_l134_134291


namespace max_five_digit_multiple_of_5_greatest_five_digit_multiple_of_5_using_1_3_5_7_9_l134_134524

open Nat

theorem max_five_digit_multiple_of_5 (n : ℕ) 
  (h1 : n >= 10000 ∧ n < 100000) 
  (h2 : (multiset.of_nat_digits n) = {1, 3, 5, 7, 9}) 
  (h3 : n % 5 = 0) : n ≤ 97315 :=
sorry

theorem greatest_five_digit_multiple_of_5_using_1_3_5_7_9 : 
   ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (multiset.of_nat_digits n = {1, 3, 5, 7, 9}) ∧ (n % 5 = 0) ∧ (n = 97315) := 
begin
  use 97315,
  repeat { split },
  { -- 97315 is a five-digit number
    exact dec_trivial },
  { -- Digits of 97315 are 1, 3, 5, 7, 9
    exact dec_trivial },
  { -- 97315 is a multiple of 5
    exact dec_trivial },
  { -- 97315 is the number itself
    refl }
end

end max_five_digit_multiple_of_5_greatest_five_digit_multiple_of_5_using_1_3_5_7_9_l134_134524


namespace remainder_when_divided_by_5_l134_134375

theorem remainder_when_divided_by_5 (n : ℕ) (k : ℤ) (h : n = 10 * k.to_nat + 7) : n % 5 = 2 := 
by 
  sorry

end remainder_when_divided_by_5_l134_134375


namespace sequence_unbounded_l134_134803

/-- 
Let \( A_n \) denote the \((n-1) \times (n-1)\) matrix \( (a_{ij}) \) with 
\( a_{ij} = i + 2 \) for \( i = j \), and 1 otherwise.
Prove that the sequence \(\frac{\det(A_n)}{n!}\) is unbounded.
-/ 
theorem sequence_unbounded (n : ℕ) (h : n ≥ 1) : 
  ¬∃ C : ℝ, ∀ (n : ℕ), n ≥ 1 → ( (matrix.det (matrix.of (λ i j, if i = j then i+2 else 1) : matrix (fin (n-1)) (fin (n-1)) ℝ)) / n ! ) ≤ C :=
sorry

end sequence_unbounded_l134_134803


namespace slope_angle_and_min_distance_l134_134345

/-- Given the parametric equations of a line and a curve, prove the slope angle of the line and the minimum distance from a point on the curve to the line. -/
theorem slope_angle_and_min_distance (t θ : ℝ) : 
  (let x_l := (1 / 2) * t,
       y_l := (sqrt 3 / 2) * t + 1,
       x_c := 2 + cos θ,
       y_c := sin θ in
   let α := pi / 3,
       d_min := (2 * sqrt 3 - 1) / 2 in
   (∀ t, (y_l = sqrt 3 * x_l + 1) → α = pi / 3) ∧
   (∀ θ, (x_c - 2)^2 + y_c^2 = 1 → d_min = ((2 * sqrt 3 + 1) / 2) - 1)) := 
by
  sorry

end slope_angle_and_min_distance_l134_134345


namespace Q_after_move_up_4_units_l134_134871

-- Define the initial coordinates.
def Q_initial : (ℤ × ℤ) := (-4, -6)

-- Define the transformation - moving up 4 units.
def move_up (P : ℤ × ℤ) (units : ℤ) : (ℤ × ℤ) := (P.1, P.2 + units)

-- State the theorem to be proved.
theorem Q_after_move_up_4_units : move_up Q_initial 4 = (-4, -2) :=
by 
  sorry

end Q_after_move_up_4_units_l134_134871


namespace count_geo_seqs_in_set_eq_four_l134_134193

-- Define the set of numbers {1, 2, ... , 10}
def mySet : Set ℕ := {n | n ∈ Finset.range 10 ∧ n > 0}

-- Define the property of being a geometric sequence
def is_geometric_sequence (a b c : ℕ) : Prop :=
  b^2 = a * c

-- Define a function that counts how many triples (a, b, c) satisfy the geometric sequence property
def count_geometric_sequences (s : Set ℕ) : ℕ :=
  Finset.card 
    (Finset.filter 
      (λ (t : ℕ × ℕ × ℕ), t.1 < t.2 ∧ t.2 < t.3 ∧ is_geometric_sequence t.1 t.2 t.3) 
      (Finset.product (Finset.product (s.to_finset) (s.to_finset)) (s.to_finset)))

-- The main theorem stating the number of geometric sequences in the set {1, 2, ..., 10}
theorem count_geo_seqs_in_set_eq_four : count_geometric_sequences mySet = 4 :=
  sorry

end count_geo_seqs_in_set_eq_four_l134_134193


namespace distance_between_homes_l134_134819

theorem distance_between_homes (Maxwell_speed : ℝ) (Brad_speed : ℝ) (M_time : ℝ) (B_delay : ℝ) (D : ℝ) 
  (h1 : Maxwell_speed = 4) 
  (h2 : Brad_speed = 6)
  (h3 : M_time = 8)
  (h4 : B_delay = 1) :
  D = 74 :=
by
  sorry

end distance_between_homes_l134_134819


namespace decreasing_condition_l134_134569

-- The function f(x)
def f (x m : ℝ) : ℝ := x^2 - 6 * m * x + 6

-- Definition of a decreasing function on an interval
def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x ≤ y → f x ≥ f y

-- The interval (-∞, 3]
def interval : set ℝ := {x | x ≤ 3}

-- Problem statement
theorem decreasing_condition (m : ℝ) :
  is_decreasing_on (λ x, f x m) interval ↔ 1 ≤ m :=
sorry

end decreasing_condition_l134_134569


namespace isosceles_triangle_base_length_l134_134472

theorem isosceles_triangle_base_length
  (a : ℕ) (b : ℕ)
  (ha : a = 7) 
  (p : ℕ)
  (hp : p = a + a + b) 
  (hp_perimeter : p = 21) : b = 7 :=
by 
  -- The actual proof will go here, using the provided conditions
  sorry

end isosceles_triangle_base_length_l134_134472


namespace five_year_salary_increase_l134_134440

noncomputable def salary_growth (S : ℝ) := S * (1.08)^5

theorem five_year_salary_increase (S : ℝ) : 
  salary_growth S = S * 1.4693 := 
sorry

end five_year_salary_increase_l134_134440


namespace find_value_of_x_l134_134011

-- Definitions used in conditions
variable (O B C A : Point) -- points on the circle
variable (OB OC OA : ℝ) -- Radii of the circle
variable (OB_eq_OC : OB = OC)
variable (angle_BCO : ℝ) -- Given angle ∠BCO
variable (angle_DAC : ℝ) -- Given angle ∠DAC
variable (right_angled_ACD : angle_DAC + 90 = 157) -- Triangle ACD is right-angled & 180 - 23 = 157

variable (x : ℝ) -- angle x

-- Setting specific values for given angles in degrees
noncomputable def angle_BCO_value : angle_BCO = 32
noncomputable def angle_DAC_value : angle_DAC = 67

-- What we want to prove
theorem find_value_of_x : x = 9 :=
sorry

end find_value_of_x_l134_134011


namespace sandwiches_sold_out_l134_134130

-- Define the parameters as constant values
def original : ℕ := 9
def available : ℕ := 4

-- The theorem stating the problem and the expected result
theorem sandwiches_sold_out : (original - available) = 5 :=
by
  -- This is the placeholder for the proof
  sorry

end sandwiches_sold_out_l134_134130


namespace plane_equation_perpendicular_l134_134662

-- Define the points and plane
def point1 : ℝ × ℝ × ℝ := (0, 2, 2)
def point2 : ℝ × ℝ × ℝ := (2, 0, 2)
def plane_normal : ℝ × ℝ × ℝ := (2, -1, 3)

-- The target theorem stating the desired plane equation
theorem plane_equation_perpendicular (A B C D : ℝ) (hA_pos: A > 0) 
(hgcd: Int.gcd (Int.ofReal A) (Int.gcd (Int.ofReal B) (Int.gcd (Int.ofReal C) (Int.ofReal D))) = 1)
(hpoint1 : A * 0 + B * 2 + C * 2 + D = 0) 
(hpoint2 : A * 2 + B * 0 + C * 2 + D = 0)
(hperpendicular : A * 2 + B * (-1) + C * 3 = 0) : 
A = 1 ∧ B = 1 ∧ C = -1 ∧ D = 0 := 
sorry

end plane_equation_perpendicular_l134_134662


namespace parallelogram_problem_l134_134769

open Real

/-- A proof problem based on the specified conditions of a parallelogram. -/
theorem parallelogram_problem
  (ABCD : Type)
  [parallelogram ABCD]
  (l1 l2 m1 m2 : Line ABCD)
  (l1_bisects_A : bisects l1 ∠A)
  (l2_bisects_C : bisects l2 ∠C)
  (m1_bisects_B : bisects m1 ∠B)
  (m2_bisects_D : bisects m2 ∠D)
  (d_l1_l2 : d l1 l2 = d m1 m2 / √3)
  (AC : length (diagonal ABCD) = √(22/3))
  (BD : length (other_diagonal ABCD) = 2) :
  ∠BAD = 60 ∧ inradius (triangle ABD) = √3 / 12 :=
by
  sorry

end parallelogram_problem_l134_134769


namespace incircle_radius_l134_134779

theorem incircle_radius (r1 r2 r3 r : ℝ) (h1 : r1 = 1) (h2 : r2 = 4) (h3 : r3 = 9)
  (h4 : ∃ (r : ℝ), r = 11) :
  ∀ (r : ℝ), r = 11 :=
by
  intro r,
  cases h4 with r hr,
  exact hr

#align incircle_radius

end incircle_radius_l134_134779


namespace exists_perfect_cube_subset_l134_134804

open Nat

theorem exists_perfect_cube_subset (S : Finset ℕ) (hS : S.card = 9)
  (h_prime_factors : ∀ n ∈ S, ∀ p, p.prime → p ∣ n → p ≤ 3) :
  ∃ a b c ∈ S, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ∃ k, a * b * c = k^3 :=
by
  sorry

end exists_perfect_cube_subset_l134_134804


namespace sally_pears_to_plums_l134_134067

-- Definitions based on the problem conditions
def weight_equivalence (pears plums : ℕ) : Prop :=
  4 * pears = 3 * plums

-- The problem statement to prove
theorem sally_pears_to_plums (p w : ℕ) (h1 : weight_equivalence 1 0.75) (h2 : p = 20) : w = 15 :=
by
  -- Proof goes here
  sorry

end sally_pears_to_plums_l134_134067


namespace sufficient_but_not_necessary_condition_l134_134814

theorem sufficient_but_not_necessary_condition (A B : Set ℝ) :
  (A = {x : ℝ | 1 < x ∧ x < 3}) →
  (B = {x : ℝ | x > -1}) →
  (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A) :=
by
  sorry

end sufficient_but_not_necessary_condition_l134_134814


namespace no_such_spatial_pentagon_exists_l134_134274

/--
A pentagon in 3-dimensional space is defined by five vertices A, B, C, D, E.
The condition is that the segment connecting any two non-adjacent vertices intersects
the plane of the triangle formed by the remaining three vertices at an interior point
of that triangle.
We want to prove that no such pentagon exists.
-/
theorem no_such_spatial_pentagon_exists :
  ¬ (∃ (A B C D E : ℝ × ℝ × ℝ), 
    (∀ (P Q R S T : ℝ × ℝ × ℝ), 
      P = A ∧ Q = B ∧ R = C ∧ S = D ∧ T = E →
      ((A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E) ∧
      (∀ (X Y Z U V : ℝ × ℝ × ℝ),
        (X = A ∧ Y = B ∧ Z = E ∧ U = C ∧ V = D ∨
         X = B ∧ Y = C ∧ Z = A ∧ U = D ∧ V = E ∨
         X = C ∧ Y = D ∧ Z = B ∧ U = E ∧ V = A ∨
         X = D ∧ Y = E ∧ Z = C ∧ U = A ∧ V = B ∨
         X = E ∧ Y = A ∧ Z = D ∧ U = B ∧ V = C) →
      ¬ (∃ (intPoint : ℝ × ℝ × ℝ), intPoint ∈ interior (triangle X Y Z) ∧
        intPoint ∈ segment U V))))) :=
sorry

end no_such_spatial_pentagon_exists_l134_134274


namespace part1_area_of_triangle_part2_range_of_ac_l134_134778

namespace TriangleProof

def condition1 (B C : Real) (b : Real) : Prop := 
  sqrt 3 * (Real.cos B) = b * (Real.sin C)

def condition2 (a c b : Real) (C : Real) : Prop := 
  2 * a - c = 2 * b * (Real.cos C)

theorem part1_area_of_triangle 
  (a b : Real) (B C : Real)
  (ha : a = 2)
  (hb : b = 2 * sqrt 3)
  (hB : B = Real.pi / 3)
  (hc1: condition1 B C b)
  (hc2: condition2 a 4 b C)
  : ∃ S : Real, S = 2 * sqrt 3 := 
sorry

theorem part2_range_of_ac 
  (b B : Real)
  (hB : B = Real.pi / 3)
  (hb : b = 2 * sqrt 3)
  (A C : Real)
  : 2 * sqrt 3 < a + c ∧ a + c ≤ 4 * sqrt 3 :=
sorry

end TriangleProof

end part1_area_of_triangle_part2_range_of_ac_l134_134778


namespace johns_initial_income_l134_134202

theorem johns_initial_income (I : ℝ) (h1 : 0.20 ≤ 0.30) (h2 : 1_500_000 > 0) 
  (h3 : 0.30 * 1_500_000 - 0.20 * I = 250_000) : 
  I = 1_000_000 :=
by
  sorry

end johns_initial_income_l134_134202


namespace length_PR_l134_134378

-- Definitions for the conditions in the problem
variables {P Q R M N : Type} [DecidableEq P] [DecidableEq Q] [DecidableEq R]
variables {length : P → P → ℝ}
variables {PM MR NQ : ℝ} (PR : ℝ)
variables (h1 : PM = 6)
variables (h2 : MR = 9)
variables (h3 : NQ = 8)
variables (h4 : length M N = length P Q)

-- The theorem to prove
theorem length_PR : PR = 20 :=
by sorry

end length_PR_l134_134378


namespace trajectory_centroid_l134_134319

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-4, 4)

-- Define the moving point C on the circle
def onCircle (C : ℝ × ℝ) : Prop :=
  let (x, y) := C
  (x - 3) ^ 2 + (y + 6) ^ 2 = 9

-- Define the coordinates of the centroid G of triangle ABC
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₀, y₀) := C
  ((x₁ + x₂ + x₀) / 3, (y₁ + y₂ + y₀) / 3)

-- The theorem stating the trajectory equation of the centroid G of triangle ABC
theorem trajectory_centroid (C : ℝ × ℝ) 
  (hC : onCircle C) : 
  let G := centroid A B C in 
  let (x, y) := G in 
  x^2 + y^2 = 1 :=
by
  sorry

end trajectory_centroid_l134_134319


namespace smallest_divisor_after_391_l134_134785

theorem smallest_divisor_after_391 (m : ℕ) (h₁ : 1000 ≤ m ∧ m < 10000) (h₂ : Even m) (h₃ : 391 ∣ m) : 
  ∃ d, d > 391 ∧ d ∣ m ∧ ∀ e, 391 < e ∧ e ∣ m → e ≥ d :=
by
  use 441
  sorry

end smallest_divisor_after_391_l134_134785


namespace find_eleventh_number_l134_134977

theorem find_eleventh_number (nums : List ℝ) (h1 : nums.length = 21) 
(h2 : (nums.take 11).sum = 11 * 48) (h3 : (nums.drop 10).sum = 11 * 41) 
(h4 : nums.sum = 21 * 44) : nums.nthLe 10 h1 = 55 := 
by
  sorry

end find_eleventh_number_l134_134977


namespace possible_values_of_n_l134_134437

theorem possible_values_of_n (n : ℕ) (h_pos : 0 < n) (h_prime_n : Nat.Prime n) (h_prime_double_sub1 : Nat.Prime (2 * n - 1)) (h_prime_quad_sub1 : Nat.Prime (4 * n - 1)) :
  n = 2 ∨ n = 3 :=
by
  sorry

end possible_values_of_n_l134_134437


namespace student_count_l134_134633

theorem student_count (cupcakes : ℕ) (leftover : ℕ) (teacher : ℕ) (aid : ℕ) (sick_students : ℕ) 
  (h1 : cupcakes = 30) 
  (h2 : leftover = 4) 
  (h3 : teacher = 1) 
  (h4 : aid = 1) 
  (h5 : sick_students = 3) : 
  let total_people := cupcakes - leftover in
  let total_students := total_people - teacher - aid in
  total_students + sick_students = 27 :=
by 
  sorry

end student_count_l134_134633


namespace transformed_curve_eq_l134_134703

section curve_transformation

variables (a : ℝ) (x y x' y' : ℝ)
def M : matrix (fin 2) (fin 2) ℝ := ![![a, 0], ![2, 1]]

-- Assume M has an eigenvalue of 2
axiom eigenvalue_condition : M.has_eigenvalue 2

-- Assume the original equation of curve C
axiom curve_original : x^2 + y^2 = 1

-- Define the transformation
noncomputable def transform_curve : ℝ × ℝ → ℝ × ℝ := 
  λ (v : ℝ × ℝ), (2 * v.1, 2 * v.1 + v.2)

theorem transformed_curve_eq :
  8 * x^2 + 4 * x * y + y^2 = 1 :=
sorry

end curve_transformation

end transformed_curve_eq_l134_134703


namespace min_cost_example_l134_134605

-- Define the numbers given in the problem
def num_students : Nat := 25
def num_vampire : Nat := 11
def num_pumpkin : Nat := 14
def pack_cost : Nat := 3
def individual_cost : Nat := 1
def pack_size : Nat := 5

-- Define the cost calculation function
def min_cost (num_v: Nat) (num_p: Nat) : Nat :=
  let num_v_packs := num_v / pack_size  -- number of packs needed for vampire bags
  let num_v_individual := num_v % pack_size  -- remaining vampire bags needed
  let num_v_cost := (num_v_packs * pack_cost) + (num_v_individual * individual_cost)
  let num_p_packs := num_p / pack_size  -- number of packs needed for pumpkin bags
  let num_p_individual := num_p % pack_size  -- remaining pumpkin bags needed
  let num_p_cost := (num_p_packs * pack_cost) + (num_p_individual * individual_cost)
  num_v_cost + num_p_cost

-- The statement to prove
theorem min_cost_example : min_cost num_vampire num_pumpkin = 17 :=
  by
  sorry

end min_cost_example_l134_134605


namespace shortest_path_l134_134652

noncomputable def diameter : ℝ := 18
noncomputable def radius : ℝ := diameter / 2
noncomputable def AC : ℝ := 7
noncomputable def BD : ℝ := 7
noncomputable def CD : ℝ := diameter - AC - BD
noncomputable def CP : ℝ := Real.sqrt (radius ^ 2 - (CD / 2) ^ 2)
noncomputable def DP : ℝ := CP

theorem shortest_path (C P D : ℝ) :
  (C - 7) ^ 2 + (D - 7) ^ 2 = CD ^ 2 →
  (C = AC) ∧ (D = BD) →
  2 * CP = 2 * Real.sqrt 77 :=
by
  intros h1 h2
  sorry

end shortest_path_l134_134652


namespace retirement_fund_increment_l134_134219

theorem retirement_fund_increment (k y : ℝ) (h1 : k * Real.sqrt (y + 3) = k * Real.sqrt y + 15)
  (h2 : k * Real.sqrt (y + 5) = k * Real.sqrt y + 27) : k * Real.sqrt y = 810 := by
  sorry

end retirement_fund_increment_l134_134219


namespace count_pairs_divisible_by_nine_l134_134701

open Nat

theorem count_pairs_divisible_by_nine (n : ℕ) (h : n = 528) :
  ∃ (count : ℕ), count = n ∧
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 100 ∧ (a^2 + b^2 + a * b) % 9 = 0 ↔
  count = 528 :=
by
  sorry

end count_pairs_divisible_by_nine_l134_134701


namespace angle_x_degrees_l134_134035

theorem angle_x_degrees 
  (O : Point)
  (A B C D : Point)
  (hC : Circle O)
  (hOB : O ∈ Segment O B)
  (hOC : O ∈ Segment O C)
  (hBCO : ∠ B C O = 32)
  (hACD : ∠ A C D = 90)
  (hCAD : ∠ C A D = 67) :
  ∠ B C A = 9 := 
  sorry

end angle_x_degrees_l134_134035


namespace metro_earnings_in_6_minutes_l134_134835

theorem metro_earnings_in_6_minutes 
  (ticket_cost : ℕ) 
  (tickets_per_minute : ℕ) 
  (duration_minutes : ℕ) 
  (earnings_in_one_minute : ℕ) 
  (earnings_in_six_minutes : ℕ) 
  (h1 : ticket_cost = 3) 
  (h2 : tickets_per_minute = 5) 
  (h3 : duration_minutes = 6) 
  (h4 : earnings_in_one_minute = tickets_per_minute * ticket_cost) 
  (h5 : earnings_in_six_minutes = earnings_in_one_minute * duration_minutes) 
  : earnings_in_six_minutes = 90 := 
by 
  -- Proof goes here
  sorry

end metro_earnings_in_6_minutes_l134_134835


namespace triangle_XYZ_proof_l134_134403

-- Define the parameters for the triangle XYZ
variables (X Y Z : ℝ)
variables (angleX := 90)
variables (length_YZ : ℝ := 20)
variables (tanZ : ℝ := 3 * (XY / length_YZ))
variables (XY := X - Y)
variables (XZ := Z - X)

noncomputable def triangle_XYZ (angleX YZ tanZ XY : ℝ) : ℝ :=
  if angleX = 90 ∧ YZ = length_YZ ∧ tanZ = 3 * (XY / YZ) then
    XY
  else 0

-- Statement to be proven
theorem triangle_XYZ_proof :
  triangle_XYZ 90 20 (3 * (XY / 20)) = (40 * (Real.sqrt 2)) / 3 :=
sorry

end triangle_XYZ_proof_l134_134403


namespace probability_bulb_on_l134_134909

def toggle_process_result : ℕ → Bool
| n := (count_divisors n) % 2 = 1

def count_divisors (n : ℕ) : ℕ :=
  (finset.Icc 1 n).count (λ d : ℕ, n % d = 0)

theorem probability_bulb_on :
  let total_bulbs := 100
  let remaining_bulbs := finset.range 101 |>.filter toggle_process_result |>.card
  remaining_bulbs / total_bulbs = 0.1 :=
sorry

end probability_bulb_on_l134_134909


namespace tangent_normal_lines_l134_134984

noncomputable def x (t : ℝ) : ℝ := (1 / 2) * t^2 - (1 / 4) * t^4
noncomputable def y (t : ℝ) : ℝ := (1 / 2) * t^2 + (1 / 3) * t^3
def t0 : ℝ := 0

theorem tangent_normal_lines :
  (∃ m : ℝ, ∀ t : ℝ, t = t0 → y t = m * x t) ∧
  (∃ n : ℝ, ∀ t : ℝ, t = t0 → y t = n * x t ∧ n = -1 / m) :=
sorry

end tangent_normal_lines_l134_134984


namespace sum_of_x_values_eq_4_l134_134159

theorem sum_of_x_values_eq_4 :
  (∀ x : ℝ, x ≠ -2 → 2 = (x^3 - 3*x^2 - 4*x) / (x + 2) → x = 4) :=
begin
  intro x,
  intro h1,
  intro h2,
  have h3 : 2 = (x^3 - 3*x^2 - 4*x) / (x + 2), from h2,
  sorry
end

end sum_of_x_values_eq_4_l134_134159


namespace christmas_tree_bulbs_l134_134913

def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

def number_of_bulbs_on (N : Nat) : Nat :=
  (Finset.range (N+1)).filter isPerfectSquare |>.card

def probability_bulb_on (total_bulbs bulbs_on : Nat) : Float :=
  (bulbs_on.toFloat) / (total_bulbs.toFloat)

theorem christmas_tree_bulbs :
  let N := 100
  let bulbs_on := number_of_bulbs_on N
  probability_bulb_on N bulbs_on = 0.1 :=
by
  sorry

end christmas_tree_bulbs_l134_134913


namespace part1_test_part2_test_l134_134322

variables {α β : ℝ}

-- Condition: α and β are acute angles
def acute_angle (x : ℝ) : Prop := 0 < x ∧ x < π / 2

-- Given conditions
axiom tan_alpha_eq_2 : tan α = 2
axiom sin_alpha_minus_beta_eq : sin (α - β) = sqrt 10 / 10
axiom alpha_acute : acute_angle α
axiom beta_acute : acute_angle β

-- Part (1)
theorem part1_test : sin α ^ 2 - 2 * sin α * cos α + 1 = 1 :=
by 
  sorry

-- Part (2)
theorem part2_test : β = π / 4 :=
by 
  sorry

end part1_test_part2_test_l134_134322


namespace minimize_MA_dot_MB_cosine_AMB_l134_134318

open Real

def coord_O := (0, 0)
def vector_OA := (1, 7)
def vector_OB := (5, 1)
def vector_OP := (2, 1)
def OM (M : ℝ × ℝ) : Prop := ∃ k : ℝ, M = (2 * k, k)

-- Condition (I): \overrightarrow{OM} = (4, 2)
def condition_OM := (4, 2)

theorem minimize_MA_dot_MB (M : ℝ × ℝ) (h : OM M ∧ M = condition_OM) : 
  let MA := (1 - (M.1), 7 - (M.2))
  let MB := (5 - (M.1), 1 - (M.2))
  (MA.1 * MB.1 + MA.2 * MB.2) = -8 := 
by 
  sorry

theorem cosine_AMB (M : ℝ × ℝ) (h : OM M ∧ M = condition_OM) :
  let MA := (1 - (M.1), 7 - (M.2))
  let MB := (5 - (M.1), 1 - (M.2))
  cosangle MA MB = - (4 * sqrt 17) / 17 :=
by 
  sorry

end minimize_MA_dot_MB_cosine_AMB_l134_134318


namespace Sarah_shampoo_conditioner_usage_l134_134074

theorem Sarah_shampoo_conditioner_usage (daily_shampoo : ℝ) (daily_conditioner : ℝ) (days_in_week : ℝ) (weeks : ℝ) (total_days : ℝ) (daily_total : ℝ) (total_usage : ℝ) :
  daily_shampoo = 1 → 
  daily_conditioner = daily_shampoo / 2 → 
  days_in_week = 7 → 
  weeks = 2 → 
  total_days = days_in_week * weeks → 
  daily_total = daily_shampoo + daily_conditioner → 
  total_usage = daily_total * total_days → 
  total_usage = 21 := by
  sorry

end Sarah_shampoo_conditioner_usage_l134_134074


namespace simplify_fraction_l134_134849

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : 
  (x / (x - y) - y / (x + y)) = (x^2 + y^2) / (x^2 - y^2) :=
sorry

end simplify_fraction_l134_134849


namespace debby_pictures_l134_134549

theorem debby_pictures : 
  let zoo_pics := 24
  let museum_pics := 12
  let pics_deleted := 14
  zoo_pics + museum_pics - pics_deleted = 22 := 
by
  sorry

end debby_pictures_l134_134549


namespace boundary_length_of_divided_rectangle_l134_134592

/-- Suppose a rectangle is divided into three equal parts along its length and two equal parts along its width, 
creating semicircle arcs connecting points on adjacent sides. Given the rectangle has an area of 72 square units, 
we aim to prove that the total length of the boundary of the resulting figure is 36.0. -/
theorem boundary_length_of_divided_rectangle 
(area_of_rectangle : ℝ)
(length_divisions : ℕ)
(width_divisions : ℕ)
(semicircle_arcs_length : ℝ)
(straight_segments_length : ℝ) :
  area_of_rectangle = 72 →
  length_divisions = 3 →
  width_divisions = 2 →
  semicircle_arcs_length = 7 * Real.pi →
  straight_segments_length = 14 →
  semicircle_arcs_length + straight_segments_length = 36 :=
by
  intros h_area h_length_div h_width_div h_arc_length h_straight_length
  sorry

end boundary_length_of_divided_rectangle_l134_134592


namespace rob_quarters_l134_134065

theorem rob_quarters :
  (∃ (quarters : ℕ), 
    let dimes := 3 in
    let nickels := 5 in
    let pennies := 12 in
    let total_value := 2.42 in
    let dime_value := 0.10 * dimes in
    let nickel_value := 0.05 * nickels in
    let penny_value := 0.01 * pennies in
    let other_coins_value := dime_value + nickel_value + penny_value in
    let quarter_value := 0.25 * quarters in
    total_value = other_coins_value + quarter_value) →
  (∃ (quarters : ℕ), quarters = 7) := 
sorry

end rob_quarters_l134_134065


namespace smaller_angle_in_parallelogram_l134_134380

theorem smaller_angle_in_parallelogram (x : ℝ) 
  (hx : ∃ y, y = x + 70) 
  (supplementary : ∀ a b, a + b = 180) 
  (hopp : ∀ a b, a = b) :
  x = 55 :=
by
  sorry

end smaller_angle_in_parallelogram_l134_134380


namespace sum_of_integers_not_zero_l134_134117

theorem sum_of_integers_not_zero {a : Fin 22 → ℤ} 
  (h : ∏ i, a i = 1) : ∑ i, a i ≠ 0 := 
sorry

end sum_of_integers_not_zero_l134_134117


namespace area_of_shape_by_Q_l134_134179

noncomputable def area_of_shape_formed_by_Q : ℝ :=
  let shape_vertices : set (ℝ × ℝ) := 
    { (u, v) | 0 ≤ u - v ∧ u - v ≤ 1 ∧ 0 ≤ u ∧ u ≤ 2 }
  in sorry -- Proof omitted

theorem area_of_shape_by_Q :
  area_of_shape_formed_by_Q = 2 :=
sorry -- Proof omitted

end area_of_shape_by_Q_l134_134179


namespace function_relationship_start_generating_profit_first_option_is_more_reasonable_l134_134583

-- Define the conditions
def purchase_price : ℝ := 980000
def first_year_maintenance : ℝ := 120000
def annual_income : ℝ := 500000
def second_year_onwards_maintenance_increase : ℝ := 40000

-- Define the profit function
def profit (x : ℝ) : ℝ := -2 * x^2 + 40 * x - 98

-- Statement 1: Function relationship between y and x
theorem function_relationship : ∀ x : ℝ, x > 0 → profit(x) = -2 * x^2 + 40 * x - 98 := by
  sorry

-- Statement 2: The machine starts generating profit from the third year
theorem start_generating_profit : ∃ N : ℝ, N = 3 ∧ ∀ x : ℝ, x > 3 → profit(x) > 0 := by
  sorry

-- Statement 3: First disposal option is more reasonable
theorem first_option_is_more_reasonable : ∃ max_profit_time : ℝ, max_profit_time = 7 ∧ 
  profit(7) + 300000 = profit(7) + 120000 :=
  sorry

end function_relationship_start_generating_profit_first_option_is_more_reasonable_l134_134583


namespace factorial_mod_sum_l134_134610

open Nat

theorem factorial_mod_sum :
  (fact 1 + fact 2 + fact 3 + fact 4 + fact 5 + fact 6 + fact 7 + fact 8 + fact 9 + fact 10) % 7 = 6 :=
by
  sorry

end factorial_mod_sum_l134_134610


namespace count_valid_n_l134_134312

theorem count_valid_n : 
  finset.card {n : ℕ | 3 ≤ n ∧ n ≤ 15 ∧ (180 * (n - 2)) % n = 0 ∧ 360 % n = 0} = 9 := 
by
  sorry

end count_valid_n_l134_134312


namespace polynomial_value_at_9_l134_134428

theorem polynomial_value_at_9 :
  ∃ p : Polynomial ℝ, p.Monic
  ∧ degree p = 8
  ∧ p.eval 1 = 1
  ∧ p.eval 2 = 2
  ∧ p.eval 3 = 3
  ∧ p.eval 4 = 4
  ∧ p.eval 5 = 5
  ∧ p.eval 6 = 6
  ∧ p.eval 7 = 7
  ∧ p.eval 8 = 8
  ∧ p.eval 9 = 40329 :=
by {
  sorry
}

end polynomial_value_at_9_l134_134428


namespace no_such_a_and_sequence_exists_l134_134273

theorem no_such_a_and_sequence_exists :
  ¬∃ (a : ℝ) (a_pos : 0 < a ∧ a < 1) (a_seq : ℕ → ℝ), (∀ n : ℕ, 0 < a_seq n) ∧ (∀ n : ℕ, 1 + a_seq (n + 1) ≤ a_seq n + (a / (n + 1)) * a_seq n) :=
by
  sorry

end no_such_a_and_sequence_exists_l134_134273


namespace largest_of_consecutive_odds_l134_134492

-- Defining the six consecutive odd numbers
def consecutive_odd_numbers (a b c d e f : ℕ) : Prop :=
  (a = b + 2) ∧ (b = c + 2) ∧ (c = d + 2) ∧ (d = e + 2) ∧ (e = f + 2)

-- Defining the product condition
def product_of_odds (a b c d e f : ℕ) : Prop :=
  a * b * c * d * e * f = 135135

-- Defining the odd numbers greater than zero
def positive_odds (a b c d e f : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0) ∧
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1)

-- Theorem
theorem largest_of_consecutive_odds (a b c d e f : ℕ) 
  (h1 : consecutive_odd_numbers a b c d e f)
  (h2 : product_of_odds a b c d e f)
  (h3 : positive_odds a b c d e f) : 
  a = 13 :=
sorry

end largest_of_consecutive_odds_l134_134492


namespace required_run_rate_l134_134562

theorem required_run_rate (run_rate_first_10_overs : ℝ) (target_runs total_overs first_overs : ℕ) :
  run_rate_first_10_overs = 4.2 ∧ target_runs = 282 ∧ total_overs = 50 ∧ first_overs = 10 →
  (target_runs - run_rate_first_10_overs * first_overs) / (total_overs - first_overs) = 6 :=
by
  sorry

end required_run_rate_l134_134562


namespace table_height_l134_134138

-- Definitions
def height_of_table (h l x: ℕ): ℕ := h 
def length_of_block (l: ℕ): ℕ := l 
def width_of_block (w x: ℕ): ℕ := x + 6
def overlap_in_first_arrangement (x : ℕ) : ℕ := x 

-- Conditions
axiom h_conditions (h l x: ℕ): 
  (l + h - x = 42) ∧ (x + 6 + h - l = 36)

-- Proof statement
theorem table_height (h l x : ℕ) (h_conditions : (l + h - x = 42) ∧ (x + 6 + h - l = 36)) :
  height_of_table h l x = 36 := sorry

end table_height_l134_134138


namespace tens_digit_of_3_pow_100_l134_134526

-- Definition: The cyclic behavior of the last two digits of 3^n.
def last_two_digits_cycle : List ℕ := [03, 09, 27, 81, 43, 29, 87, 61, 83, 49, 47, 41, 23, 69, 07, 21, 63, 89, 67, 01]

-- Condition: The length of the cycle of the last two digits of 3^n.
def cycle_length : ℕ := 20

-- Assertion: The last two digits of 3^20 is 01.
def last_two_digits_3_pow_20 : ℕ := 1

-- Given n = 100, the tens digit of 3^n when n is expressed in decimal notation
theorem tens_digit_of_3_pow_100 : (3 ^ 100 / 10) % 10 = 0 := by
  let n := 100
  let position_in_cycle := (n % cycle_length)
  have cycle_repeat : (n % cycle_length = 0) := rfl
  have digits_3_pow_20 : (3^20 % 100 = 1) := by sorry
  show (3 ^ 100 / 10) % 10 = 0
  sorry

end tens_digit_of_3_pow_100_l134_134526


namespace begonia_poetry_society_arrangement_l134_134194

-- Define the members of the poetry society
inductive Member
| LinDaiyu
| XueBaochai
| ShiXiangyun
| JiaYingchun
| JiaTanchun
| JiaXichun
| JiaBaoyu
| LiWan

open Member

-- Define the number of ways to arrange the remaining 5 members
def A_5_5 := Nat.factorial 5

-- Define the number of ways to arrange the 3 specific individuals in available spaces
def A_6_3 := Nat.factorial 6 / (Nat.factorial (6 - 3))

-- The proof statement
theorem begonia_poetry_society_arrangement :
  A_5_5 * A_6_3 = 14400 :=
by
  -- Calculation
  have fact_5 : Nat.factorial 5 = 120 := rfl
  have fact_6 : Nat.factorial 6 = 720 := rfl
  have spaces_arrangement : A_6_3 = 720 / 6 := rfl
  rw [fact_5, spaces_arrangement]
  norm_num
  sorry

end begonia_poetry_society_arrangement_l134_134194


namespace hexadecagon_area_l134_134595

/-- A regular hexadecagon is inscribed in a circle of radius r. -/
theorem hexadecagon_area (r : ℝ) : 
  let θ := 22.5 in 
  let triangle_area := (1 / 2) * r^2 * real.sin (θ * real.pi / 180) in
  let hexadecagon_area := 16 * triangle_area in
  hexadecagon_area = 4 * r^2 * real.sqrt(2 - real.sqrt 2) :=
by
  sorry

end hexadecagon_area_l134_134595


namespace BANANAS_arrangement_l134_134649

open Nat

theorem BANANAS_arrangement :
  let total_letters := 7
  let freq_A := 3
  let freq_N := 2
  fact total_letters / (fact freq_A * fact freq_N) = 420 :=
by 
  sorry

end BANANAS_arrangement_l134_134649


namespace altitude_BD_eqn_median_BE_eqn_l134_134329

variables {Point : Type}
variables {x y : Point → ℝ}

def A : Point := ⟨3, -4⟩
def B : Point := ⟨6, 0⟩
def C : Point := ⟨-5, 2⟩

-- Proving part 1: Equation of the line containing the altitude BD on side AC
theorem altitude_BD_eqn (A B C : Point) : 
  let D := (∃ p : ℝ × ℝ, ∃ k : ℝ, y(B) - y(C) = x(B) - (-1 / k) ∧ y(B) - (k * (x(B) - 6)) = p.2)
  in ∃ k : ℝ, k = 4 / 3 ∧ (4 * x(B) - 3 * y(B) - 24 = 0) :=
  sorry

-- Proving part 2: Equation of the line containing the median BE on side AC
theorem median_BE_eqn (A B C : Point) : 
  let E := ((x(A) + x(C)) / 2, (y(A) + y(C)) / 2)
  in E = (-1, -1) ∧ (x(B) - 1/7 * (x(B) - 6) = y(E)) :=
  sorry

end altitude_BD_eqn_median_BE_eqn_l134_134329


namespace sample_size_product_A_l134_134757

theorem sample_size_product_A 
  (ratio_A : ℕ)
  (ratio_B : ℕ)
  (ratio_C : ℕ)
  (total_ratio : ℕ)
  (sample_size : ℕ) 
  (h_ratio : ratio_A = 2 ∧ ratio_B = 3 ∧ ratio_C = 5)
  (h_total_ratio : total_ratio = ratio_A + ratio_B + ratio_C)
  (h_sample_size : sample_size = 80) :
  (80 * (ratio_A : ℚ) / total_ratio) = 16 :=
by
  sorry

end sample_size_product_A_l134_134757


namespace sales_tax_per_tire_l134_134519

def cost_per_tire : ℝ := 7
def number_of_tires : ℕ := 4
def final_total_cost : ℝ := 30

theorem sales_tax_per_tire :
  (final_total_cost - number_of_tires * cost_per_tire) / number_of_tires = 0.5 :=
sorry

end sales_tax_per_tire_l134_134519


namespace fraction_walk_home_l134_134232

theorem fraction_walk_home :
  let bus := 1/3
  let auto := 1/6
  let bicycle := 1/15
  let total_fraction := bus + auto + bicycle
  let walk := 1 - total_fraction
  walk = 13/30 :=
by
  -- Definitions of the fractions for students going home by bus, automobile, and bicycle:
  let bus := 1/3
  let auto := 1/6
  let bicycle := 1/15

  -- Compute the total fraction of students not walking:
  have total_fraction := bus + auto + bicycle

  -- Subtract from the whole to get the fraction of students who walk:
  have walk := 1 - total_fraction

  -- Assertion: the fraction walking home is 13/30.
  show walk = 13/30
  from sorry

end fraction_walk_home_l134_134232


namespace angle_x_l134_134026

-- Define the conditions
variables (O A C D B : Point)
variable h : circle_center O
variable h1 : on_circumference A C D O
variable h2 : diameter AC
variable h3 : ∠ CDA = 42

-- The problem statement we want to prove
theorem angle_x (x : ℝ) : x = 58 :=
begin
  -- This is where the solution steps would go, but for now, we leave it as sorry
  sorry
end

end angle_x_l134_134026


namespace correct_conclusions_l134_134794

-- Define the concept of "approximating functions"
def approximating_functions (C₁ C₂ : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x, a ≤ x ∧ x ≤ b → -1 ≤ C₁ x - C₂ x ∧ C₁ x - C₂ x ≤ 1

-- Define the four conclusions as propositions
def conclusion_1 : Prop :=
approximating_functions (λ x, x - 5) (λ x, 3 * x + 2) 1 2

def conclusion_2 : Prop :=
approximating_functions (λ x, x - 5) (λ x, x^2 - 4 * x) 3 4

def conclusion_3 : Prop :=
approximating_functions (λ x, x^2 - 1) (λ x, 2 * x^2 - x) 0 1

def conclusion_4 : Prop :=
approximating_functions (λ x, x - 5) (λ x, x^2 - 4 * x) 2 3

-- The main theorem statement to be proven
theorem correct_conclusions : conclusion_2 ∧ conclusion_3 ∧ ¬conclusion_1 ∧ ¬conclusion_4 :=
by sorry

end correct_conclusions_l134_134794


namespace maximum_n_for_sequence_l134_134684

theorem maximum_n_for_sequence :
  ∃ (n : ℕ), 
  (∀ a S : ℕ → ℝ, 
    a 1 = 1 → 
    (∀ n : ℕ, n > 0 → 2 * a (n + 1) + S n = 2) → 
    (1001 / 1000 < S (2 * n) / S n ∧ S (2 * n) / S n < 11 / 10)) →
  n = 9 :=
sorry

end maximum_n_for_sequence_l134_134684


namespace solve_problem_statement_l134_134178

noncomputable def problem_statement : Prop :=
  (∏ n in (Finset.range (2016 - 1755 + 1)).map (Finset.natCast 1755), (1 + (1 : ℚ) / n)) >
  real.sqrt (8 / 7)

theorem solve_problem_statement : problem_statement :=
begin
  sorry
end

end solve_problem_statement_l134_134178


namespace area_remaining_l134_134553

variable (d : ℝ)
variable (r : ℝ ≥ 0)
variable (A B M : Point)
variable (h_perpendicular : Chord M d (2 * sqrt 7))
variable (AB_semi : Semicircle A B)
variable (AM_semi MB_semi : Semicircle)

theorem area_remaining (h1 : Semicircle.diameter_len AB_semi d)
                       (h2 : Point.on_diameter M AB_semi)
                       (h3 : Semicircle.on_diameter AM_semi d/2)
                       (h4 : Semicircle.on_diameter MB_semi d/2)
                       (h5 : Chord.length h_perpendicular = 2 * sqrt 7) :
           remaining_area AB_semi AM_semi MB_semi = 21.99 := sorry

end area_remaining_l134_134553


namespace sum_of_roots_l134_134889

noncomputable theory

def quadratic_has_distinct_real_roots (p : ℝ) : Prop :=
  p^2 > 60

theorem sum_of_roots (p : ℝ) (r1 r2 : ℝ) (h : r1 + r2 = -p) (h2 : r1 * r2 = 15)
  (h3 : quadratic_has_distinct_real_roots p) : 
  |r1 + r2| > 2 * Real.sqrt 15 := 
by
  sorry

end sum_of_roots_l134_134889


namespace probability_all_correct_l134_134558

noncomputable def probability_mcq : ℚ := 1 / 3
noncomputable def probability_true_false : ℚ := 1 / 2

theorem probability_all_correct :
  (probability_mcq * probability_true_false * probability_true_false) = (1 / 12) :=
by
  sorry

end probability_all_correct_l134_134558


namespace incorrect_conclusion_l134_134740

theorem incorrect_conclusion
  (a b : ℝ) 
  (h₁ : 1/a < 1/b) 
  (h₂ : 1/b < 0) 
  (h₃ : a < 0) 
  (h₄ : b < 0) 
  (h₅ : a > b) : ¬ (|a| + |b| > |a + b|) := 
sorry

end incorrect_conclusion_l134_134740


namespace find_m_l134_134707

def point := (4, 0)
def line (m : ℝ) : ℝ → ℝ := λ x, 4 / 3 * x + m / 3

def distance_from_point_to_line (p : ℝ × ℝ) (L : ℝ → ℝ) : ℝ :=
  abs (4 * p.1 - 3 * p.2 + (L 0)) / sqrt (4^2 + 3^2)

theorem find_m (m : ℝ) : distance_from_point_to_line point (line m) = 3 ↔ (m = -1 ∨ m = -31) :=
by
  sorry

end find_m_l134_134707


namespace concurrency_of_ceva_circles_l134_134996

theorem concurrency_of_ceva_circles
  (A B C D1 D2 E1 E2 F1 F2 L M N : Point)
  (h1 : Circle.Intersects (Line.mk B C) = {D1, D2})
  (h2 : Circle.Intersects (Line.mk C A) = {E1, E2})
  (h3 : Circle.Intersects (Line.mk A B) = {F1, F2})
  (hL : Segment.mk D1 E1 ∩ Segment.mk D2 F2 = L)
  (hM : Segment.mk E1 F1 ∩ Segment.mk E2 D2 = M)
  (hN : Segment.mk F1 D1 ∩ Segment.mk F2 E2 = N) : 
  Concurrent (Line.mk A L) (Line.mk B M) (Line.mk C N) :=
sorry

end concurrency_of_ceva_circles_l134_134996


namespace avg_growth_rate_equation_l134_134603

/-- This theorem formalizes the problem of finding the equation for the average growth rate of working hours.
    Given that the average working hours in the first week are 40 hours and in the third week are 48.4 hours,
    we need to show that the equation for the growth rate \(x\) satisfies \( 40(1 + x)^2 = 48.4 \). -/
theorem avg_growth_rate_equation (x : ℝ) (first_week_hours third_week_hours : ℝ) 
  (h1: first_week_hours = 40) (h2: third_week_hours = 48.4) :
  40 * (1 + x) ^ 2 = 48.4 :=
sorry

end avg_growth_rate_equation_l134_134603


namespace angle_x_degrees_l134_134034

theorem angle_x_degrees 
  (O : Point)
  (A B C D : Point)
  (hC : Circle O)
  (hOB : O ∈ Segment O B)
  (hOC : O ∈ Segment O C)
  (hBCO : ∠ B C O = 32)
  (hACD : ∠ A C D = 90)
  (hCAD : ∠ C A D = 67) :
  ∠ B C A = 9 := 
  sorry

end angle_x_degrees_l134_134034


namespace scalar_positions_l134_134802

theorem scalar_positions {A C F H L S : ℕ} (h1 : A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
                         (h2 : C ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
                         (h3 : F ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
                         (h4 : H ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
                         (h5 : L ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
                         (h6 : S ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
                         (h7 : S ≠ 0) 
                         (h8 : F ≠ 0) : 
    (271 ∣ (S * 10^5 + C * 10^4 + H * 10^3 + L * 10^2 + A * 10 + F - F * 10^5 - L * 10^4 - A * 10^3 - C * 10^2 - H * 10 - S)) ↔ 
    (C = L ∧ H = A) := 
by 
  sorry

end scalar_positions_l134_134802


namespace find_remaining_area_l134_134554

-- Given conditions
def semicircle (diameter : ℝ) : Prop := true

def point_on_diameter (M AB : ℝ) : Prop := true

def two_semicircles_cut_out (AM MB : ℝ) (AB : ℝ) : Prop := AM + MB = AB

def chord_length_perpendicular (M : ℝ) (chord_length : ℝ) : Prop := chord_length = 2 * real.sqrt 7

-- Remaining area calculation
def remaining_area (AB : ℝ) : ℝ :=
  let radius := AB / 2
  let chord_height := real.sqrt 7
  let r_squared := radius * radius + 7
  let total_area := (1 / 2) * real.pi * r_squared
  let smaller_semicircles_area := real.pi * (r_squared / 4)
  total_area - 2 * (smaller_semicircles_area / 2)

-- Proof problem
theorem find_remaining_area (AB M AM MB : ℝ)
  (h1 : semicircle AB)
  (h2 : point_on_diameter M AB)
  (h3 : two_semicircles_cut_out AM MB AB)
  (h4 : chord_length_perpendicular M (2 * real.sqrt 7)) :
  21.99 ≈ remaining_area AB :=
by sorry

end find_remaining_area_l134_134554


namespace angle_x_is_58_l134_134020

-- Definitions and conditions based on the problem statement
variables {O A C D : Type} (x : ℝ)
variable [circular_center : O] -- O is the center of the circle
variable [radius_equal : AO = BO ∧ BO = CO] -- AO, BO, and CO are radii and hence equal
variable (angle_ACD_right : ∠ A C D = 90) -- ∠ACD is a right angle (subtended by diameter)
variable (angle_CDA : ∠ C D A = 42) -- ∠CDA is 42 degrees

-- Statement to be proven
theorem angle_x_is_58 (x : ℝ) : x = 58 := 
by sorry

end angle_x_is_58_l134_134020


namespace min_sum_floor_half_plus_one_l134_134884

theorem min_sum_floor_half_plus_one (n : ℕ) (a : ℕ → ℕ) :
  ∑ i in finset.range(n), a i ≥ (n / 2) + 1 :=
sorry

end min_sum_floor_half_plus_one_l134_134884


namespace sequence_multiple_middle_term_l134_134216

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 1 ∧
  a 3 = 1 ∧
  ∀ n, a n * a (n + 3) - a (n + 1) * a (n + 2) = 1

theorem sequence_multiple_middle_term (a : ℕ → ℕ) 
  (h : sequence a) :
  ∀ n, (a n + a (n + 2)) % a (n + 1) = 0 :=
begin
  sorry
end

end sequence_multiple_middle_term_l134_134216


namespace cube_function_increasing_l134_134883

noncomputable def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2

theorem cube_function_increasing : 
  is_increasing (λ x : ℝ, x^3) :=
sorry

end cube_function_increasing_l134_134883


namespace find_b_l134_134231

noncomputable def curve (x : ℝ) : ℝ := x^3 - 3 * x^2
noncomputable def tangent_line (x b : ℝ) : ℝ := -3 * x + b

theorem find_b
  (b : ℝ)
  (h : ∃ x : ℝ, curve x = tangent_line x b ∧ deriv curve x = -3) :
  b = 1 :=
by
  sorry

end find_b_l134_134231


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_formula_l134_134989

-- Definitions based on given conditions
variables (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions
axiom a_4 : a 4 = 6
axiom a_6 : a 6 = 10
axiom all_positive_b : ∀ n, 0 < b n
axiom b_3 : b 3 = a 3
axiom T_2 : T 2 = 3

-- Required to prove
theorem arithmetic_sequence_general_formula : ∀ n, a n = 2 * n - 2 :=
sorry

theorem geometric_sequence_sum_formula : ∀ n, T n = 2^n - 1 :=
sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_formula_l134_134989


namespace area_enclosed_by_region_l134_134146

theorem area_enclosed_by_region :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6*x + 8*y = -9) →
  let radius := 4 in
  let area := real.pi * radius^2 in
  area = 16 * real.pi :=
sorry

end area_enclosed_by_region_l134_134146


namespace problem_statement_l134_134427

noncomputable def recurrence_sequence (n : ℕ) : ℕ → ℝ × ℝ
| 0 => (-3, 2)
| (n + 1) => let (c, d) := recurrence_sequence n
             in (2 * c + d + Real.sqrt (c^2 + d^2), 2 * c + d - Real.sqrt (c^2 + d^2))

theorem problem_statement : 
  let c := (recurrence_sequence 10).1 
  let d := (recurrence_sequence 10).2 
  (1 / c) + (1 / d) = -512 / 3 :=
by
  sorry

end problem_statement_l134_134427


namespace cubic_polynomial_value_l134_134793

theorem cubic_polynomial_value :
  let f : ℝ → ℝ := λ x, x^3 - 3 * x + 2
  ∃ g : ℝ → ℝ,
    (∃ A : ℝ, g = λ x, A * (x - 1)^2 * (x - 4)) ∧
    g 0 = -2 ∧
    (g 16 = -1350) :=
begin
  sorry,
end

end cubic_polynomial_value_l134_134793


namespace area_of_triangle_ABC_is_63_l134_134608

noncomputable def area_of_triangle_ABC (A B C D E F G: Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space G] (S_BDG S_CDG S_AEG : ℝ) : Prop :=
  ∃ (S_ABC : ℝ), 
    S_BDG = 8 ∧ S_CDG = 6 ∧ S_AEG = 14 ∧ S_ABC = 63

theorem area_of_triangle_ABC_is_63 
  {A B C D E F G : Type*} 
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space G] 
  (h1 : ∃ (S_BDG S_CDG S_AEG : ℝ), S_BDG = 8 ∧ S_CDG = 6 ∧ S_AEG = 14) :
  ∃ (S_ABC : ℝ), S_ABC = 63 :=
sorry

end area_of_triangle_ABC_is_63_l134_134608


namespace area_of_region_l134_134660

def fractional_part (x : ℝ) : ℝ :=
  x - real.floor x

theorem area_of_region :
  (set.univ : set ℝ).indicator (λ x, (set.univ : set ℝ).indicator (λ y, 1) y) 
    (λ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 200 * fractional_part x ≥ real.floor x + real.floor y) = 101 :=
by
  sorry

end area_of_region_l134_134660


namespace eccentricity_hyperbola_l134_134721

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  let e := sqrt (1 + (b^2 / a^2))
  e

theorem eccentricity_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_line : ∀ x, y x = sqrt 3 * x)
  (h_b2 : b^2 = (3 + 2 * sqrt 3) * a^2) :
  hyperbola_eccentricity a b ha hb = (1 + sqrt 3) := 
sorry

end eccentricity_hyperbola_l134_134721


namespace christmas_bulbs_on_probability_l134_134904

noncomputable def toggle_algorithm (n : ℕ) : ℕ → Bool
| k => (List.range n).foldl (λ b i => if (k + 1) % (i + 1) == 0 then !b else b) true

def is_perfect_square (n : ℕ) : Bool :=
  let root := (n + 1).natAbs.sqrt
  root * root = n + 1

def probability_bulb_on_after_toggling (n : ℕ) : ℚ :=
  let on_count := finset.filter (λ k => toggle_algorithm n k) (finset.range n) 
  on_count.card / n

theorem christmas_bulbs_on_probability : 
  probability_bulb_on_after_toggling 100 = 0.1 :=
sorry

end christmas_bulbs_on_probability_l134_134904


namespace derek_points_l134_134760

theorem derek_points (total_points : ℕ) (num_other_players : ℕ) (average_points_other_players : ℕ) (points_other_players : ℕ) (derek_points : ℕ) :
  total_points = 65 →
  num_other_players = 8 →
  average_points_other_players = 5 →
  points_other_players = num_other_players * average_points_other_players →
  derek_points = total_points - points_other_players →
  derek_points = 25 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw h4 at h5
  exact h5

end derek_points_l134_134760


namespace no_nat_solution_for_exp_eq_l134_134843

theorem no_nat_solution_for_exp_eq (n x y z : ℕ) (hn : n > 1) (hx : x ≤ n) (hy : y ≤ n) :
  ¬ (x^n + y^n = z^n) :=
by
  sorry

end no_nat_solution_for_exp_eq_l134_134843


namespace power_sum_zero_l134_134891

theorem power_sum_zero (n : ℕ) (h : 0 < n) : (-1:ℤ)^(2*n) + (-1:ℤ)^(2*n+1) = 0 := 
by 
  sorry

end power_sum_zero_l134_134891


namespace closest_d_to_nearest_tenth_l134_134590

noncomputable def probability_within_d_units_of_lattice_point (d : ℝ) : ℝ :=
  let unit_square_area := 1
  let lattice_influenced_area := π * d^2
  lattice_influenced_area / unit_square_area -- This should be equivalent to the probability

-- The theorem stating the problem and the expected solution:
theorem closest_d_to_nearest_tenth :
  (∃ d : ℝ, probability_within_d_units_of_lattice_point d = 0.5) → d = 0.4 := 
by
  sorry

end closest_d_to_nearest_tenth_l134_134590


namespace _l134_134168

def triangle (A B C : Type) : Prop :=
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def angles_not_equal_sides_not_equal (A B C : Type) (angleB angleC : ℝ) (sideAC sideAB : ℝ) : Prop :=
  triangle A B C →
  (angleB ≠ angleC → sideAC ≠ sideAB)
  
lemma xiaoming_theorem {A B C : Type} 
  (hTriangle : triangle A B C)
  (angleB angleC : ℝ)
  (sideAC sideAB : ℝ) :
  angleB ≠ angleC → sideAC ≠ sideAB := 
sorry

end _l134_134168


namespace A_beats_B_by_80_meters_l134_134383

-- Define the constants and conditions
def distance : ℝ := 1000  -- in meters, distance of the race
def time_A : ℝ := 115  -- in seconds, time taken by A
def time_difference : ℝ := 10  -- in seconds, time by which A beats B

-- Calculate speeds
def speed_A : ℝ := distance / time_A  -- speed of A
def time_B : ℝ := time_A + time_difference  -- time taken by B to complete the race
def speed_B : ℝ := distance / time_B  -- speed of B

-- Calculate the distance covered by B in the time A finishes the race
def distance_B_in_time_A : ℝ := speed_B * time_A

-- The distance by which A beats B
def distance_by_which_A_beats_B : ℝ := distance - distance_B_in_time_A

-- The theorem that needs to be proven
theorem A_beats_B_by_80_meters : distance_by_which_A_beats_B = 80 := by
  sorry

end A_beats_B_by_80_meters_l134_134383


namespace tan_cond_sufficient_not_necessary_l134_134988

noncomputable def specific_value := (1 : ℝ)

theorem tan_cond_sufficient_not_necessary : 
  (∀ x : ℝ, tan x = 1 → ∃ a ∈ {specific_value}, tan a = 1) ∧ 
  (∃ x : ℝ, tan x = 1 ∧ x ∉ {specific_value}) :=
by {
  sorry
}

end tan_cond_sufficient_not_necessary_l134_134988


namespace rectangle_ratio_eq_4sqrt3_l134_134131

theorem rectangle_ratio_eq_4sqrt3 (t y x : ℝ) 
  (h_eq_triangle_area : (∃ t : ℝ, (sqrt 3 / 4) * t^2 = 3 * (x * y)))
  (h_eq_long_side : x = t) :
  x / y = 4 * sqrt 3 := 
sorry

end rectangle_ratio_eq_4sqrt3_l134_134131


namespace triangle_ABC_is_right_angled_l134_134692

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 2, y := 1 }
def B : Point := { x := 3, y := 2 }
def C : Point := { x := -1, y := 4 }

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

def AB := distance A B
def BC := distance B C
def AC := distance A C

theorem triangle_ABC_is_right_angled :
  (AB ^ 2 + AC ^ 2 = BC ^ 2) :=
sorry

end triangle_ABC_is_right_angled_l134_134692


namespace expression_evaluation_l134_134278

noncomputable def expression (b : ℝ) : ℝ := 
  (1 / 25 * b ^ 0) + ((1 / (25 * b)) ^ 0) - (125 ^ (-1 / 3)) - ((-81) ^ (-1 / 4))

theorem expression_evaluation (b : ℝ) (hb : b ≠ 0) : 
  expression b = 1 + 13 / 75 := by
  sorry

end expression_evaluation_l134_134278


namespace calculate_remainder_l134_134420

def is_valid_number (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000 ∧ (∀ (d : ℕ), d ∈ digits 10 n → d % 2 = 0) ∧ (list.nodup (digits 10 n))

def sum_of_valid_numbers : ℕ :=
  ∑ n in (finset.filter is_valid_number (finset.range 10000)), n

theorem calculate_remainder : sum_of_valid_numbers % 1000 = 560 := 
sorry

end calculate_remainder_l134_134420


namespace four_lines_intersect_at_single_point_l134_134129

-- We define the vertices of the pentagons as points in the plane.
variables (O A B C D A' B' C' D' : Point)

-- Definitions of regular pentagons, and condition that they share a common vertex O.
def regular_pentagon (O A B C D : Point) : Prop :=
  distance O A = distance A B ∧ distance A B = distance B C ∧ 
  distance B C = distance C D ∧ distance C D = distance D O

-- Regular pentagons with common vertex:
def regular_pentagons_with_common_vertex (O A B C D A' B' C' D' : Point) :=
  regular_pentagon O A B C D ∧ regular_pentagon O A' B' C' D' ∧ O = O

-- Lines connecting corresponding vertices:
def lines_connecting_vertices (O A B C D A' B' C' D' : Point) : Prop :=
  Line O A ∧ Line A A' ∧ Line B B' ∧ Line C C' ∧ Line D D'

-- The main theorem stating that these lines intersect at a single point.
theorem four_lines_intersect_at_single_point (O A B C D A' B' C' D' : Point) :
  regular_pentagons_with_common_vertex O A B C D A' B' C' D' → 
  lines_connecting_vertices O A B C D A' B' C' D' →
  ∃ (P : Point), (Line A A').contains P ∧ (Line B B').contains P ∧ 
  (Line C C').contains P ∧ (Line D D').contains P :=
by
  sorry  -- Proof is omitted as per the instructions.

end four_lines_intersect_at_single_point_l134_134129


namespace factorial_identity_l134_134964

theorem factorial_identity : (10.factorial * 4.factorial * 3.factorial) / (9.factorial * 7.factorial) = 2 / 7 := 
by {
  sorry
}

end factorial_identity_l134_134964


namespace percentage_y_of_x_l134_134743

variable {x y : ℝ}

theorem percentage_y_of_x 
  (h : 0.15 * x = 0.20 * y) : y = 0.75 * x := 
sorry

end percentage_y_of_x_l134_134743


namespace equal_plan_cost_l134_134995

theorem equal_plan_cost (x : ℕ) (h₁ : 9 < x) 
  (h₂ : 0.60 + (x - 9) * 0.06 = x * 0.08) : x = 12 := sorry

end equal_plan_cost_l134_134995


namespace enclosed_region_area_l134_134151

theorem enclosed_region_area :
  (∃ x y : ℝ, x ^ 2 + y ^ 2 - 6 * x + 8 * y = -9) →
  ∃ (r : ℝ), r ^ 2 = 16 ∧ ∀ (area : ℝ), area = π * 4 ^ 2 :=
by
  sorry

end enclosed_region_area_l134_134151


namespace vector_dot_product_l134_134751

variable {V : Type*} [inner_product_space ℝ V]

theorem vector_dot_product (a b : V) 
  (h1 : ∥a∥ = 3) 
  (h2 : ∥b∥ = 1) 
  (h3 : ∥a - b∥ = 2) : inner_product_space.is_R_or_C.inner a b = 3 := 
  sorry

end vector_dot_product_l134_134751


namespace no_integer_solution_mod_13_l134_134162

theorem no_integer_solution_mod_13 (x y : ℤ) :
  ¬ ((∃ x, x^4 % 13 + 6 = y^3 % 13)) :=
by
  intro h_exist
  cases h_exist with x hx
  sorry

end no_integer_solution_mod_13_l134_134162


namespace increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l134_134543

noncomputable def fA (x : ℝ) : ℝ := -x
noncomputable def fB (x : ℝ) : ℝ := (2/3)^x
noncomputable def fC (x : ℝ) : ℝ := x^2
noncomputable def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function_fA : ¬∀ x y : ℝ, x < y → fA x < fA y := sorry
theorem increasing_function_fB : ¬∀ x y : ℝ, x < y → fB x < fB y := sorry
theorem increasing_function_fC : ¬∀ x y : ℝ, x < y → fC x < fC y := sorry
theorem increasing_function_fD : ∀ x y : ℝ, x < y → fD x < fD y := sorry

end increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l134_134543


namespace problem_statement_l134_134801

noncomputable def w1 : set (ℝ × ℝ) := {p | (p.1)^2 + (p.2)^2 + 12 * p.1 - 20 * p.2 - 100 = 0}
noncomputable def w2 : set (ℝ × ℝ) := {p | (p.1)^2 + (p.2)^2 - 12 * p.1 - 20 * p.2 + 200 = 0}
noncomputable def line_b (b : ℝ) : set (ℝ × ℝ) := {p | p.2 = b * p.1}

theorem problem_statement : ∃ n : ℝ, (∀ b : ℝ, (line_b b).interior ∈ w1 → (line_b b).exterior ∈ w2 → b = n) ∧ 
  n^2 = 25 / 36 ∧ ∃ p q : ℕ, (n / (n+1) = p / q) ∧ gcd p q = 1 → (p + q = 61) :=
by
  -- "sorry" is used here to indicate that the proof has been omitted.
  sorry

end problem_statement_l134_134801


namespace range_of_a_l134_134337

noncomputable def f (x a : ℝ) := Real.log (x^2 - a * x + 3 * a) / Real.log (Real.sin 1)

theorem range_of_a (
  h_decreasing : ∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ ≤ x₂ → f x₁ a ≥ f x₂ a
) : -4 < a ∧ a ≤ 4 :=
begin
  sorry -- proof omitted
end

end range_of_a_l134_134337


namespace max_difference_proof_l134_134192

-- Define the revenue function R(x)
def R (x : ℕ+) : ℝ := 3000 * (x : ℝ) - 20 * (x : ℝ) ^ 2

-- Define the cost function C(x)
def C (x : ℕ+) : ℝ := 500 * (x : ℝ) + 4000

-- Define the profit function P(x) as revenue minus cost
def P (x : ℕ+) : ℝ := R x - C x

-- Define the marginal function M
def M (f : ℕ+ → ℝ) (x : ℕ+) : ℝ := f (⟨x + 1, Nat.succ_pos x⟩) - f x

-- Define the marginal profit function MP(x)
def MP (x : ℕ+) : ℝ := M P x

-- Statement of the proof
theorem max_difference_proof : 
  (∃ x_max : ℕ+, ∀ x : ℕ+, x ≤ 100 → P x ≤ P x_max) → -- P achieves its maximum at some x_max within constraints
  (∃ x_max : ℕ+, ∀ x : ℕ+, x ≤ 100 → MP x ≤ MP x_max) → -- MP achieves its maximum at some x_max within constraints
  (P x_max - MP x_max = 71680) := 
sorry -- proof omitted

end max_difference_proof_l134_134192


namespace angle_x_l134_134030

-- Define the conditions
variables (O A C D B : Point)
variable h : circle_center O
variable h1 : on_circumference A C D O
variable h2 : diameter AC
variable h3 : ∠ CDA = 42

-- The problem statement we want to prove
theorem angle_x (x : ℝ) : x = 58 :=
begin
  -- This is where the solution steps would go, but for now, we leave it as sorry
  sorry
end

end angle_x_l134_134030


namespace harmonic_point_P_3_m_harmonic_point_hyperbola_l134_134636

-- Part (1)
theorem harmonic_point_P_3_m (t : ℝ) (m : ℝ) (P : ℝ × ℝ → Prop)
  (h₁ : P ⟨ 3, m ⟩)
  (h₂ : ∀ x y, P ⟨ x, y ⟩ ↔ (x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y)) :
  m = -7 :=
by sorry

-- Part (2)
theorem harmonic_point_hyperbola (k : ℝ) (P : ℝ × ℝ → Prop)
  (h_hb : ∀ x, -3 < x ∧ x < -1 → P ⟨ x, k / x ⟩)
  (h₂ : ∀ x y, P ⟨ x, y ⟩ ↔ (x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y)) :
  3 < k ∧ k < 4 :=
by sorry

end harmonic_point_P_3_m_harmonic_point_hyperbola_l134_134636


namespace original_food_price_l134_134580

theorem original_food_price (P : ℝ) : 
  let total_paid := 165
  let discount := 0.15
  let tax := 0.10
  let service_fee := 0.05
  let tip := 0.20
  let discounted_price := P * (1 - discount)
  let price_with_tax_and_fee := discounted_price * (1 + tax + service_fee)
  let final_price := price_with_tax_and_fee * (1 + tip)
  final_price = total_paid → 
  P ≈ 119.83 :=
begin
  sorry
end

end original_food_price_l134_134580


namespace solve_for_x_l134_134083

theorem solve_for_x : ∃ x : ℚ, 24 - 4 = 3 * (1 + x) ∧ x = 17 / 3 :=
by
  sorry

end solve_for_x_l134_134083


namespace largest_binomial_coefficient_term_l134_134376

theorem largest_binomial_coefficient_term 
  (x : ℝ) (n : ℕ) 
  (h : (λ k : ℕ, (∑ k in finset.range (n + 1), (choose n k : ℝ))) n = 128) :
  ((\exists r : ℕ, r = 3 ∧ T (r + 1) = - (choose 7 3 : ℝ) * x) ∧
  (\exists r : ℕ, r = 4 ∧ T (r + 1) = (choose 7 4 : ℝ) * x^(1/6)) ∧
  (\exists r : ℕ, r = 4 ∧ T (r + 1) = (choose 7 4 : ℝ) * x^(1/6))) :=
sorry

-- Define the general term T in terms of r
noncomputable def T (r : ℕ) : ℝ := 
  choose 7 r * (-1)^r * x^((7 / 2) - (5 * r / 6))

end largest_binomial_coefficient_term_l134_134376


namespace angle_x_l134_134024

-- Define the conditions
variables (O A C D B : Point)
variable h : circle_center O
variable h1 : on_circumference A C D O
variable h2 : diameter AC
variable h3 : ∠ CDA = 42

-- The problem statement we want to prove
theorem angle_x (x : ℝ) : x = 58 :=
begin
  -- This is where the solution steps would go, but for now, we leave it as sorry
  sorry
end

end angle_x_l134_134024


namespace decimal_to_base5_of_89_l134_134259

theorem decimal_to_base5_of_89 : (324 : ℕ) = nat.to_digits 5 89 :=
by
  sorry

end decimal_to_base5_of_89_l134_134259


namespace outward_angle_l134_134121

theorem outward_angle {n : ℕ} (h : n > 4) : (zigzag_pattern_outward_angle n) = 720 / n :=
sorry

def zigzag_pattern_outward_angle (n : ℕ) : ℕ :=
2 * (360 / n)

end outward_angle_l134_134121


namespace closest_d_to_nearest_tenth_l134_134589

noncomputable def probability_within_d_units_of_lattice_point (d : ℝ) : ℝ :=
  let unit_square_area := 1
  let lattice_influenced_area := π * d^2
  lattice_influenced_area / unit_square_area -- This should be equivalent to the probability

-- The theorem stating the problem and the expected solution:
theorem closest_d_to_nearest_tenth :
  (∃ d : ℝ, probability_within_d_units_of_lattice_point d = 0.5) → d = 0.4 := 
by
  sorry

end closest_d_to_nearest_tenth_l134_134589


namespace circle_center_radius_sum_l134_134811

theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), (∀ x y : ℝ, (x^2 + 2 * x - 8 * y - 7 = - y^2 - 6 * x) →
  (a, b) = (-4, 4) ∧ r = Real.sqrt 39 ∧ a + b + r = Real.sqrt 39) :=
by 
  exists (-4 : ℝ) (4 : ℝ) (Real.sqrt 39)
  intros x y h
  split; try { refl }
  sorry

end circle_center_radius_sum_l134_134811


namespace majority_is_correct_l134_134765

noncomputable def winning_votes (total_votes : ℕ) (winning_percentage : ℝ) : ℕ :=
  (total_votes : ℝ) * winning_percentage / 1

noncomputable def losing_votes (total_votes : ℕ) (winning_percentage : ℝ) : ℕ :=
  total_votes - winning_votes total_votes winning_percentage

theorem majority_is_correct (total_votes : ℕ) (winning_percentage : ℝ)
  (h_total_votes : total_votes = 440)
  (h_winning_percentage : winning_percentage = 0.70) :
  (winning_votes total_votes winning_percentage - losing_votes total_votes winning_percentage) = 176 :=
by
  sorry

end majority_is_correct_l134_134765


namespace next_simultaneous_departure_time_l134_134869

-- Define the departure intervals for the bus routes
def Route1_interval : Nat := 4
def Route3_interval : Nat := 6
def Route7_interval : Nat := 9

-- Define the least common multiple (LCM) function
noncomputable def lcm (a b : Nat) : Nat := Nat.lcm a b

theorem next_simultaneous_departure_time :
  lcm Route1_interval (lcm Route3_interval Route7_interval) = 36 := 
by
  -- This is a placeholder. The actual proof should demonstrate that the LCM of 4, 6, and 9 is 36.
  sorry

end next_simultaneous_departure_time_l134_134869


namespace geometric_series_inequality_l134_134679

variables {x y : ℝ}

theorem geometric_series_inequality 
  (hx : |x| < 1) 
  (hy : |y| < 1) :
  (1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x * y)) :=
sorry

end geometric_series_inequality_l134_134679


namespace oranges_remaining_l134_134132

theorem oranges_remaining :
  ∀ (Michaela_oranges : ℕ) (Cassandra_oranges : ℕ) (total_oranges : ℕ),
    Michaela_oranges = 30 →
    Cassandra_oranges = 3 * Michaela_oranges →
    total_oranges = 200 →
    total_oranges - (Michaela_oranges + Cassandra_oranges) = 80 :=
  by
    intros Michaela_oranges Cassandra_oranges total_oranges
    intros hMichaela hCassandra hTotal
    rw [hMichaela, hCassandra, hTotal]
    exact calc
      200 - (30 + 3 * 30)
        = 200 - 120 : by rfl
        = 80       : by rfl

end oranges_remaining_l134_134132


namespace probability_of_winning_l134_134511

variable (P_A P_B P_C P_M_given_A P_M_given_B P_M_given_C : ℝ)

theorem probability_of_winning :
  P_A = 0.6 →
  P_B = 0.3 →
  P_C = 0.1 →
  P_M_given_A = 0.1 →
  P_M_given_B = 0.2 →
  P_M_given_C = 0.3 →
  (P_A * P_M_given_A + P_B * P_M_given_B + P_C * P_M_given_C) = 0.15 :=
by sorry

end probability_of_winning_l134_134511


namespace parallel_lines_k_l134_134327

theorem parallel_lines_k (k : ℝ) :
  (∃ (x y : ℝ), (k-3) * x + (4-k) * y + 1 = 0 ∧ 2 * (k-3) * x - 2 * y + 3 = 0) →
  (k = 3 ∨ k = 5) :=
by
  sorry

end parallel_lines_k_l134_134327


namespace average_of_three_students_l134_134938

def average_score (scores : List ℚ) : ℚ :=
  (List.sum scores) / (scores.length)

theorem average_of_three_students : 
  let scores := [92, 75, 98]
  average_score scores = 88. (3⁻¹ : ℚ)  -- This represents 88.\overline{3}
:= by
  let scores := [92, 75, 98]
  have h1 : List.sum scores = 265 := by sorry
  have h2 : scores.length = 3 := by sorry
  calc
    average_score scores = (List.sum scores) / (scores.length) : by rfl
                   ...   = 265 / 3                       : by rw [h1, h2]
                   ...   = 88. (3⁻¹ : ℚ) ⋆ -- Here we manually input the fractional value

end average_of_three_students_l134_134938


namespace angles_on_line_y_eq_x_l134_134495

-- Define a predicate representing that an angle has its terminal side on the line y = x
def angle_on_line_y_eq_x (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 4

-- The goal is to prove that the set of all such angles is as stated
theorem angles_on_line_y_eq_x :
  { α : ℝ | ∃ k : ℤ, α = k * Real.pi + Real.pi / 4 } = { α : ℝ | angle_on_line_y_eq_x α } :=
sorry

end angles_on_line_y_eq_x_l134_134495


namespace fence_perimeter_l134_134931

-- Definitions for the given conditions
def num_posts : ℕ := 36
def gap_between_posts : ℕ := 6

-- Defining the proof problem to show the perimeter equals 192 feet
theorem fence_perimeter (n : ℕ) (g : ℕ) (sq_field : ℕ → ℕ → Prop) : n = 36 ∧ g = 6 ∧ sq_field n g → 4 * ((n / 4 - 1) * g) = 192 :=
by
  intro h
  sorry

end fence_perimeter_l134_134931


namespace simplify_and_evaluate_expression_l134_134080

variable (a : ℚ)

theorem simplify_and_evaluate_expression (h : a = -1/3) : 
  (a + 1) * (a - 1) - a * (a + 3) = 0 := 
by
  sorry

end simplify_and_evaluate_expression_l134_134080


namespace average_books_per_student_l134_134756

theorem average_books_per_student :
  let total_students := 25
  let no_books := 2
  let one_book := 12
  let two_books := 4
  let at_least_three_books := total_students - no_books - one_book - two_books
  let total_books := (no_books * 0) + (one_book * 1) + (two_books * 2) + (at_least_three_books * 3)
  (total_books : ℚ) / total_students = 1.64 :=
by
  let total_students := 25
  let no_books := 2
  let one_book := 12
  let two_books := 4
  let at_least_three_books := total_students - no_books - one_book - two_books
  let total_books := (no_books * 0) + (one_book * 1) + (two_books * 2) + (at_least_three_books * 3)
  have h : (total_books : ℚ) / total_students = 1.64 := sorry
  exact h

end average_books_per_student_l134_134756


namespace find_a_values_l134_134658

noncomputable def is_root (a b c d : ℂ) (p : ℂ → ℂ) : Prop :=
p a = 0 ∧ p b = 0 ∧ p c = 0 ∧ p d = 0

noncomputable def are_opposite_vertices (z₁ z₂ z₃ z₄ : ℂ) : Prop :=
(z₁ + z₂) / 2 = (z₃ + z₄) / 2

theorem find_a_values :
  ∀ a : ℝ, 
  ∃ (z₁ z₂ z₃ z₄ : ℂ), 
  is_root z₁ z₂ z₃ z₄ (λ z, z^4 - 8*z^3 + 13*a*z^2 - 3*(3*a^2 - 2*a - 5)*z + 4) ∧ 
  (are_opposite_vertices z₁ z₂ z₃ z₄ ∨ are_opposite_vertices z₁ z₃ z₂ z₄) →
  a = (1 + Complex.sqrt 31) / 3 ∨ a = (1 - Complex.sqrt 31) / 3 :=
begin
  sorry
end

end find_a_values_l134_134658


namespace range_of_a_l134_134731

-- Definitions for propositions and conditions
def is_decreasing (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 ≥ f x2

def curve_intersects_two_points (a : ℝ) : Prop :=
  (2*a - 3)^2 - 4 > 0

-- Main theorem statement matching the mathematical proof problem
theorem range_of_a (a : ℝ) (p q : Prop) 
  (h0 : 0 < a) 
  (h1 : a ≠ 1)
  (hp : p ↔ is_decreasing a (λ x, Real.log (x + 3) / Real.log a))
  (hq : q ↔ curve_intersects_two_points a)
  (h_p_or_q : p ∨ q)
  (h_p_and_q : ¬ (p ∧ q)) :
  (1 / 2 ≤ a ∧ a < 1) ∨ (a > 5 / 2) :=
by
  sorry

end range_of_a_l134_134731


namespace supremum_is_zero_l134_134672

noncomputable def supremum_expression (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) : ℝ :=
  xyz * (x + y + z) / (x + y + z)^3

theorem supremum_is_zero : ∀ (x y z : ℝ),
  (0 < x ∧ 0 < y ∧ 0 < z) → 
  ∀ ε > 0, 
  ∃ (a b c : ℝ), (0 < a ∧ 0 < b ∧ 0 < c) ∧ supremum_expression a b c < ε := 
by
  sorry

end supremum_is_zero_l134_134672


namespace tan_phi_eq_neg2_l134_134363

def f (x φ : ℝ) : ℝ := sin (x + φ) + 2 * cos (x + φ)

theorem tan_phi_eq_neg2 {φ : ℝ} 
  (h₁ : ∀ x : ℝ, f x φ = sin (x + φ) + 2 * cos (x + φ))
  (h₂ : ∀ x : ℝ, f (-x) φ = -f x φ) : 
  tan φ = -2 :=
sorry

end tan_phi_eq_neg2_l134_134363


namespace symmetric_point_origin_l134_134388

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_point (P : Point) : Point :=
  ⟨-P.x, -P.y, -P.z⟩

theorem symmetric_point_origin :
  let P := ⟨1, -2, 1⟩
  let Q := symmetric_point P
  Q = ⟨-1, 2, -1⟩ :=
by
  let P := ⟨1, -2, 1⟩
  let Q := symmetric_point P
  show Q = ⟨-1, 2, -1⟩
  sorry

end symmetric_point_origin_l134_134388


namespace posts_needed_l134_134594

-- Define the main properties
def length_of_side_W_stone_wall := 80
def short_side := 50
def intervals (metres: ℕ) := metres / 10 + 1 

-- Define the conditions
def posts_along_w_stone_wall := intervals length_of_side_W_stone_wall
def posts_along_short_sides := 2 * (intervals short_side - 1)

-- Calculate total posts
def total_posts := posts_along_w_stone_wall + posts_along_short_sides

-- Define the theorem
theorem posts_needed : total_posts = 19 := 
by
  sorry

end posts_needed_l134_134594


namespace group_size_l134_134584

theorem group_size (boxes_per_man total_boxes : ℕ) (h1 : boxes_per_man = 2) (h2 : total_boxes = 14) :
  total_boxes / boxes_per_man = 7 := by
  -- Definitions and conditions from the problem
  have man_can_carry_2_boxes : boxes_per_man = 2 := h1
  have group_can_hold_14_boxes : total_boxes = 14 := h2
  -- Proof follows from these conditions
  sorry

end group_size_l134_134584


namespace count_true_propositions_l134_134693

open Set

-- Definitions of propositions
def p : Prop := ∅ ⊆ {0}
def q : Prop := {1} ∈ ({1, 2} : Set (Set ℕ))

-- Main statement to prove
theorem count_true_propositions :
  (if p ∨ q then 1 else 0) + (if p ∧ q then 1 else 0) + (if ¬p then 1 else 0) = 1 :=
  sorry

end count_true_propositions_l134_134693


namespace Sarah_shampoo_conditioner_usage_l134_134075

theorem Sarah_shampoo_conditioner_usage (daily_shampoo : ℝ) (daily_conditioner : ℝ) (days_in_week : ℝ) (weeks : ℝ) (total_days : ℝ) (daily_total : ℝ) (total_usage : ℝ) :
  daily_shampoo = 1 → 
  daily_conditioner = daily_shampoo / 2 → 
  days_in_week = 7 → 
  weeks = 2 → 
  total_days = days_in_week * weeks → 
  daily_total = daily_shampoo + daily_conditioner → 
  total_usage = daily_total * total_days → 
  total_usage = 21 := by
  sorry

end Sarah_shampoo_conditioner_usage_l134_134075


namespace angle_x_degrees_l134_134042

noncomputable def center_of_circle (O : Point) := true
noncomputable def radii (O B C A : Point) := (dist O B = dist O C) ∧ (dist O A = dist O C)
noncomputable def isosceles_triangle (O B C : Point) := (∠ B C O = 32) ∧ (∠ O B C = 32)
noncomputable def isosceles_base_angle (O A C : Point) := ∠ O A C = ∠ A O C = 23
noncomputable def angle_sum_property (O A C : Point) := ∠ A O C = 90

theorem angle_x_degrees
  (O B C A : Point) (h_radii : radii O B C A)
  (h_isosceles_triangle : isosceles_triangle O B C)
  (h_isosceles_base : isosceles_base_angle O A C)
  (h_sum_property : angle_sum_property O A C) :
  ∠ B C O - ∠ O A C = 9 := by sorry

end angle_x_degrees_l134_134042


namespace cube_inequality_l134_134744

theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_l134_134744


namespace triangle_angle_l134_134405

theorem triangle_angle (A B C : ℝ) (AB AC : ℝ) (B_angled : ℝ) (angle_sum : A + B + C = 180) :
  AB = 2 →
  AC = sqrt 2 →
  B_angled = 30 →
  (A = 105 ∨ A = 15) :=
by
  intros
  sorry

end triangle_angle_l134_134405


namespace equilateral_triangle_angle_60_deg_l134_134986

open EuclideanGeometry

theorem equilateral_triangle_angle_60_deg
  (A B C D E F : Point)
  (h_equilateral : Triangle.is_equilateral A B C)
  (hD_on_AB : D ∈ LineSegment AB)
  (hE_on_BC : E ∈ LineSegment BC)
  (hF_on_CA : F ∈ LineSegment CA)
  (hDE_parallel_AC : Line.parallel (Line.mk D E) (Line.mk A C))
  (hDF_parallel_BC : Line.parallel (Line.mk D F) (Line.mk B C)) :
  ∠ (Line.mk A E) (Line.mk B F) = 60 := sorry

end equilateral_triangle_angle_60_deg_l134_134986


namespace part_I_part_II_part_III_l134_134341

-- Definition of functions and conditions
def f (a x : ℝ) : ℝ := a * x - a / x - 2 * log x
def g (x : ℝ) : ℝ := 2 * real.exp 1 / x

-- Conditions
variable (a : ℝ) (h₁ : a > 0)

-- (I) Proof that f(x) has exactly one zero when a = 2.
theorem part_I : ∃! x, f 2 x = 0 :=
sorry

-- (II) Proof of monotonicity
theorem part_II (h₂ : a ≠ 1) :
  (∀ x y, 0 < x → x < y → f a x < f a y) ∨
  (∀ x y, 0 < x → x < y → f a x < f a y ∨ f a x > f a y ∧ f a x > f a y) :=
sorry

-- (III) Proof that if ∃ x₀ ∈ [1, e] such that f(x₀) > g(x₀), then a > 4e / (e^2 - 1)
theorem part_III (h₃ : ∃ x₀ ∈ Icc 1 (real.exp 1), f a x₀ > g x₀) :
  a > 4 * real.exp 1 / (real.exp 1 ^ 2 - 1) :=
sorry

end part_I_part_II_part_III_l134_134341


namespace min_period_and_parity_of_f_l134_134486

noncomputable def f : ℝ → ℝ := λ x, sin (2 * x) - 4 * (sin x) ^ 3 * cos x

def min_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x, f (x + T) = f x ∧ (∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x)

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem min_period_and_parity_of_f :
  min_positive_period f (π / 2) ∧ is_odd f :=
sorry

end min_period_and_parity_of_f_l134_134486


namespace equal_real_roots_implies_m_l134_134748

theorem equal_real_roots_implies_m (m : ℝ) : (∃ (x : ℝ), x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) → m = 1/4 :=
by
  sorry

end equal_real_roots_implies_m_l134_134748


namespace angle_x_is_9_degrees_l134_134053

theorem angle_x_is_9_degrees
  (O B C A D : Type)
  (hO_center : is_center O)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (hOB_eq_OC : distance O B = distance O C)
  (hAngle_BCO : angle B C O = 32)
  (hOA_eq_OC : distance O A = distance O C)
  (hAngle_CAD : angle C A D = 67)
  : angle x = 9 :=
sorry

end angle_x_is_9_degrees_l134_134053


namespace length_of_DB_l134_134774

variables (A B C D : Type)
variables [EuclideanGeometry A B C D]
open EuclideanGeometry

theorem length_of_DB
  (AB AC BC AD : ℝ)
  (ABC : RightAngle ∠ BAC)
  (AB : AB = 120)
  (AC : AC = 180)
  (D : Point BC)
  (perpendicular : Perpendicular AD BC) :
  Distance D B = 20 * sqrt 11 := 
by sorry

end length_of_DB_l134_134774


namespace cannot_form_cube_l134_134623

-- Define the pattern with the conditions provided
structure SquarePattern :=
(central : Prop)
(left : Prop)
(right : Prop)
(top : Prop)
(bottom_missing : Prop)

-- Hypothesis based on the conditions described
def pattern := SquarePattern
  { central := true,
    left := true,
    right := true,
    top := true,
    bottom_missing := true }

-- Prove that this pattern cannot form a cube
theorem cannot_form_cube (p : SquarePattern) : not (p.central ∧ p.left ∧ p.right ∧ p.top ∧ ¬p.bottom_missing → forms_cube) :=
by {
  sorry
}

end cannot_form_cube_l134_134623


namespace tangent_point_coordinates_l134_134374

theorem tangent_point_coordinates : 
  ∃ P : ℝ × ℝ, 
    (∃ m : ℝ, P = (m, m^3) ∧ 
    ∀ x : ℝ, deriv (λ x, x^3) m = 3 ∧ 3 * m^2 = 3) → 
  P = (1, 1) :=
by
  sorry

end tangent_point_coordinates_l134_134374


namespace christmas_bulbs_on_probability_l134_134907

noncomputable def toggle_algorithm (n : ℕ) : ℕ → Bool
| k => (List.range n).foldl (λ b i => if (k + 1) % (i + 1) == 0 then !b else b) true

def is_perfect_square (n : ℕ) : Bool :=
  let root := (n + 1).natAbs.sqrt
  root * root = n + 1

def probability_bulb_on_after_toggling (n : ℕ) : ℚ :=
  let on_count := finset.filter (λ k => toggle_algorithm n k) (finset.range n) 
  on_count.card / n

theorem christmas_bulbs_on_probability : 
  probability_bulb_on_after_toggling 100 = 0.1 :=
sorry

end christmas_bulbs_on_probability_l134_134907


namespace lemons_count_l134_134090

def total_fruits (num_baskets : ℕ) (total : ℕ) : Prop := num_baskets = 5 ∧ total = 58
def basket_contents (basket : ℕ → ℕ) : Prop := 
  basket 1 = 18 ∧ -- mangoes
  basket 2 = 10 ∧ -- pears
  basket 3 = 12 ∧ -- pawpaws
  (∀ i, (i = 4 ∨ i = 5) → basket i = (basket 4 + basket 5) / 2)

theorem lemons_count (num_baskets : ℕ) (total : ℕ) (basket : ℕ → ℕ) : 
  total_fruits num_baskets total ∧ basket_contents basket → basket 5 = 9 :=
by
  sorry

end lemons_count_l134_134090


namespace smallest_two_digit_multiple_of_17_l134_134962

theorem smallest_two_digit_multiple_of_17 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ n % 17 = 0 ∧ ∀ m, (10 ≤ m ∧ m < n ∧ m % 17 = 0) → false := sorry

end smallest_two_digit_multiple_of_17_l134_134962


namespace sarah_total_volume_in_two_weeks_l134_134069

def shampoo_daily : ℝ := 1

def conditioner_daily : ℝ := 1 / 2 * shampoo_daily

def days : ℕ := 14

def total_volume : ℝ := (shampoo_daily * days) + (conditioner_daily * days)

theorem sarah_total_volume_in_two_weeks : total_volume = 21 := by
  sorry

end sarah_total_volume_in_two_weeks_l134_134069


namespace shaded_region_area_l134_134574

theorem shaded_region_area : 
  let side_length := 10 in
  let midpoint_length := side_length / 2 in
  let base := midpoint_length in
  let height := side_length in
  let triangle_area := (1 / 2) * base * height in
  2 * triangle_area = 50 :=
by
  sorry

end shaded_region_area_l134_134574


namespace enclosed_region_area_l134_134150

theorem enclosed_region_area :
  (∃ x y : ℝ, x ^ 2 + y ^ 2 - 6 * x + 8 * y = -9) →
  ∃ (r : ℝ), r ^ 2 = 16 ∧ ∀ (area : ℝ), area = π * 4 ^ 2 :=
by
  sorry

end enclosed_region_area_l134_134150


namespace probability_at_least_one_die_shows_three_l134_134137

theorem probability_at_least_one_die_shows_three : 
  let outcomes := 36
  let not_three_outcomes := 25
  (outcomes - not_three_outcomes) / outcomes = 11 / 36 := sorry

end probability_at_least_one_die_shows_three_l134_134137


namespace solve_for_y_l134_134085

theorem solve_for_y (y : ℝ) (h : log 5 ((4 * y + 16) / (3 * y - 8)) + log 5 ((3 * y - 8) / (2 * y - 5)) = 3) : 
  y = 203 / 82 :=
sorry

end solve_for_y_l134_134085


namespace angle_x_degrees_l134_134037

theorem angle_x_degrees 
  (O : Point)
  (A B C D : Point)
  (hC : Circle O)
  (hOB : O ∈ Segment O B)
  (hOC : O ∈ Segment O C)
  (hBCO : ∠ B C O = 32)
  (hACD : ∠ A C D = 90)
  (hCAD : ∠ C A D = 67) :
  ∠ B C A = 9 := 
  sorry

end angle_x_degrees_l134_134037


namespace f_p_arbitrarily_large_l134_134412

theorem f_p_arbitrarily_large :
  ∀ n : ℕ, ∃ p : ℕ, Prime p ∧ 
  (∃ g : fin p → fin p, 
    (∀ x : fin p, ∃ y : fin p, y = (x^2 + 1) % p) ∧ 
    ∃ cycle_length : ℕ, cycle_length ≥ n ∧ 
    ∃ cycle : list (fin p), is_cycle cycle_length cycle g) :=
begin
  sorry
end

-- Definition and properties of is_cycle should be assumed to have previous definitions related in the imported Mathlib.

end f_p_arbitrarily_large_l134_134412


namespace angle_x_degrees_l134_134040

noncomputable def center_of_circle (O : Point) := true
noncomputable def radii (O B C A : Point) := (dist O B = dist O C) ∧ (dist O A = dist O C)
noncomputable def isosceles_triangle (O B C : Point) := (∠ B C O = 32) ∧ (∠ O B C = 32)
noncomputable def isosceles_base_angle (O A C : Point) := ∠ O A C = ∠ A O C = 23
noncomputable def angle_sum_property (O A C : Point) := ∠ A O C = 90

theorem angle_x_degrees
  (O B C A : Point) (h_radii : radii O B C A)
  (h_isosceles_triangle : isosceles_triangle O B C)
  (h_isosceles_base : isosceles_base_angle O A C)
  (h_sum_property : angle_sum_property O A C) :
  ∠ B C O - ∠ O A C = 9 := by sorry

end angle_x_degrees_l134_134040


namespace infinite_integral_terms_l134_134120

def satisfies_sequence (a : ℕ → ℤ) (α β : ℤ) : Prop :=
  a 1 = α ∧ a 2 = β ∧ ∀ n, a (n + 2) = (a n * a (n + 1)) / (2 * a n - a (n + 1))

theorem infinite_integral_terms (α β : ℤ) :
  (∃ a : ℕ → ℤ, satisfies_sequence a α β ∧ (∀ n, is_integral a n)) ↔ (α = β ∧ is_integral α) :=
sorry

end infinite_integral_terms_l134_134120


namespace domain_of_function_l134_134873

def function_domain : Set ℝ := { x : ℝ | x + 1 ≥ 0 ∧ 2 - x ≠ 0 }

theorem domain_of_function :
  function_domain = { x : ℝ | x ≥ -1 ∧ x ≠ 2 } :=
sorry

end domain_of_function_l134_134873


namespace A_is_infinite_m_seq_not_eventually_periodic_l134_134810

axiom pos_int (n : ℕ) : n > 0

-- (a, b) are pairs of positive integers.
def game_state : Type := ℕ × ℕ 

-- Definition of a winning strategy for player B
def B_winning_strategy : game_state → Prop := sorry

-- Set A as defined in the problem
def A : set ℕ := {a | ∃ (b : ℕ), pos_int b ∧ b < a ∧ B_winning_strategy (a, b)}

-- Elements of A denoted as a_1 < a_2 < a_3 < ...
noncomputable def a_seq (n : ℕ) : ℕ := sorry 

axiom increasing_a_seq : ∀ n, a_seq n < a_seq (n + 1)

-- Sequence m_k = a_{k+1} - a_k
noncomputable def m_seq (k : ℕ) : ℕ := a_seq (k + 1) - a_seq k

-- Statement 1: A is infinite
theorem A_is_infinite : ∀ n : ℕ, ∃ a : ℕ, a ∈ A ∧ a > n := 
sorry

-- Statement 2: The sequence m_k is not eventually periodic
theorem m_seq_not_eventually_periodic (N P : ℕ) : 
  ¬ (∃ N P : ℕ, ∀ k ≥ N, m_seq (k + P) = m_seq k) :=
sorry

end A_is_infinite_m_seq_not_eventually_periodic_l134_134810


namespace find_unique_number_l134_134290

def is_three_digit_number (N : ℕ) : Prop := 100 ≤ N ∧ N < 1000

def nonzero_digits (A B C : ℕ) : Prop := A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0

def digits_of_number (N A B C : ℕ) : Prop := N = 100 * A + 10 * B + C

def product (N A B : ℕ) := N * (10 * A + B) * A

def divides (n m : ℕ) := ∃ k, n * k = m

theorem find_unique_number (N A B C : ℕ) (h1 : is_three_digit_number N)
    (h2 : nonzero_digits A B C) (h3 : digits_of_number N A B C)
    (h4 : divides 1000 (product N A B)) : N = 875 :=
sorry

end find_unique_number_l134_134290


namespace geo_seq_prod_l134_134758

theorem geo_seq_prod {a : ℕ → ℝ} (h_pos : ∀ n, 0 < a n)
  (h_geo : ∃ r > 0, ∀ n, a (n+1) = a n * r) 
  (h_log : real.log (a 2 * a 3 * a 5 * a 7 * a 8) / real.log 2 = 5) : 
  a 1 * a 9 = 4 := 
by {
  sorry
}

end geo_seq_prod_l134_134758


namespace find_theta_l134_134726

theorem find_theta
  (θ : ℝ)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (ha : ∃ k, (2 * Real.cos θ, 2 * Real.sin θ) = (k * 3, k * Real.sqrt 3)) :
  θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6 :=
by
  sorry

end find_theta_l134_134726


namespace jeff_makes_donuts_for_days_l134_134782

variable (d : ℕ) (boxes donuts_per_box : ℕ) (donuts_per_day eaten_per_day : ℕ) (chris_eaten total_donuts : ℕ)

theorem jeff_makes_donuts_for_days :
  (donuts_per_day = 10) →
  (eaten_per_day = 1) →
  (chris_eaten = 8) →
  (boxes = 10) →
  (donuts_per_box = 10) →
  (total_donuts = boxes * donuts_per_box) →
  (9 * d - chris_eaten = total_donuts) →
  d = 12 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end jeff_makes_donuts_for_days_l134_134782


namespace greatest_integer_not_exceeding_100y_l134_134809

def sum_cos_n_deg (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1) \ {0}, real.cos (i : ℝ * real.pi / 180)
def sum_sin_n_deg (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1) \ {0}, real.sin (i : ℝ * real.pi / 180)

theorem greatest_integer_not_exceeding_100y : 
  ∀ (y : ℝ), y = (sum_cos_n_deg 50) / (sum_sin_n_deg 50) → ⌊100 * y⌋ = 100 :=
by {
  sorry
}

end greatest_integer_not_exceeding_100y_l134_134809


namespace natives_cannot_obtain_910_rupees_with_50_coins_l134_134142

theorem natives_cannot_obtain_910_rupees_with_50_coins (x y z : ℤ) : 
  x + y + z = 50 → 
  10 * x + 34 * y + 62 * z = 910 → 
  false :=
by
  sorry

end natives_cannot_obtain_910_rupees_with_50_coins_l134_134142


namespace combination_16_5_l134_134619

theorem combination_16_5 : nat.choose 16 5 = 4368 := 
by
  sorry

end combination_16_5_l134_134619


namespace sum_of_fourth_powers_mod_5_l134_134508

theorem sum_of_fourth_powers_mod_5 (A : Fin 40 → ℤ) (hA : ∀ i, 5 ∣ A i → False) :
  5 ∣ (∑ i, (A i) ^ 4) :=
by
  sorry

end sum_of_fourth_powers_mod_5_l134_134508


namespace four_digit_numbers_distinct_digits_l134_134664

theorem four_digit_numbers_distinct_digits (n : ℕ) :
  n = 154 ↔ ∃ (d : ℕ → ℕ) (thousands hundreds tens units : ℕ),
  (∀ i j, i ≠ j → d i ≠ d j) ∧  -- distinct digits
  thousands ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧  -- thousands place cannot be 0
  hundreds ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  tens ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  units ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  thousands ≠ hundreds ∧ hundreds ≠ tens ∧ tens ≠ units ∧
  abs (hundreds - units) = 8 ∧
  d 0 = thousands ∧ d 1 = hundreds ∧ d 2 = tens ∧ d 3 = units :=
sorry

end four_digit_numbers_distinct_digits_l134_134664


namespace exists_min_value_iff_l134_134372

noncomputable def f (a x : ℝ) : ℝ := log a (x ^ 2 - a * x + 1 / 2)

theorem exists_min_value_iff (a : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, (∀ z : ℝ, f a z ≥ y) ∧ f a x = y) ↔ 1 < a ∧ a < sqrt 2 :=
by
  -- Here, we assume relevant conditions and deduce the required equivalency.
  sorry

end exists_min_value_iff_l134_134372


namespace C1_cartesian_C2_cartesian_minimum_distance_l134_134395

-- Define the parametric and polar equations as given in the problem
def C1_parametric (α : ℝ) : ℝ × ℝ :=
  (sqrt 2 * sin (α + π/4), sin (2 * α) + 1)

def C2_polar (ρ θ : ℝ) : Prop :=
  ρ^2 = 4 * ρ * sin θ - 3

-- Prove the Cartesian equation of curve C1
theorem C1_cartesian :
  ∀ (x y : ℝ) (α : ℝ), (x = sqrt 2 * sin (α + π/4)) → (y = sin (2*α) + 1) → y = x^2 := by 
  sorry

-- Prove the Cartesian equation of curve C2
theorem C2_cartesian :
  ∀ (x y ρ θ : ℝ), (ρ = sqrt (x^2 + y^2)) → (y = ρ * sin θ) → (ρ^2 = 4 * ρ * sin θ - 3) → (x^2 + y^2 - 4 * y + 3 = 0) := by 
  sorry

-- Prove minimum distance between a point on C1 and a point on C2
theorem minimum_distance :
  ∀ (x₀ y₀ : ℝ), (y₀ = x₀^2) → (x₀^4 - 3*x₀^2 + 4 = (x₀^2 - 3/2)^2 + 7/4) → distance (x₀, y₀) (0, 2) = sqrt(7)/2 - 1 := by 
  sorry

end C1_cartesian_C2_cartesian_minimum_distance_l134_134395


namespace find_angle_measure_l134_134746

def complement_more_condition (x : ℝ) : Prop :=
  90 - x = (1 / 7) * x + 26

theorem find_angle_measure (x : ℝ) (h : complement_more_condition x) : x = 56 :=
sorry

end find_angle_measure_l134_134746


namespace find_value_of_x_l134_134009

-- Definitions used in conditions
variable (O B C A : Point) -- points on the circle
variable (OB OC OA : ℝ) -- Radii of the circle
variable (OB_eq_OC : OB = OC)
variable (angle_BCO : ℝ) -- Given angle ∠BCO
variable (angle_DAC : ℝ) -- Given angle ∠DAC
variable (right_angled_ACD : angle_DAC + 90 = 157) -- Triangle ACD is right-angled & 180 - 23 = 157

variable (x : ℝ) -- angle x

-- Setting specific values for given angles in degrees
noncomputable def angle_BCO_value : angle_BCO = 32
noncomputable def angle_DAC_value : angle_DAC = 67

-- What we want to prove
theorem find_value_of_x : x = 9 :=
sorry

end find_value_of_x_l134_134009


namespace angle_x_is_9_degrees_l134_134055

theorem angle_x_is_9_degrees
  (O B C A D : Type)
  (hO_center : is_center O)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (hOB_eq_OC : distance O B = distance O C)
  (hAngle_BCO : angle B C O = 32)
  (hOA_eq_OC : distance O A = distance O C)
  (hAngle_CAD : angle C A D = 67)
  : angle x = 9 :=
sorry

end angle_x_is_9_degrees_l134_134055


namespace acceleration_of_iron_ball_kinetic_energy_of_iron_ball_time_to_reach_seafloor_l134_134143

-- Definitions of conditions
def depth_of_sea : ℝ := 100
def diameter_of_sphere : ℝ := 0.20
def specific_gravity_iron : ℝ := 7.23
def specific_gravity_seawater : ℝ := 1.05
def g : ℝ := 9.81

-- Proofs of the required results
theorem acceleration_of_iron_ball 
  (depth_of_sea = 100)
  (diameter_of_sphere = 0.20)
  (specific_gravity_iron = 7.23)
  (specific_gravity_seawater = 1.05)
  (g = 9.81) : acceleration_of_ball = 8.38 := sorry

theorem kinetic_energy_of_iron_ball
  (depth_of_sea = 100)
  (diameter_of_sphere = 0.20)
  (specific_gravity_iron = 7.23)
  (specific_gravity_seawater = 1.05)
  (g = 9.81) : kinetic_energy = 2588 := sorry

theorem time_to_reach_seafloor
  (depth_of_sea = 100)
  (diameter_of_sphere = 0.20)
  (specific_gravity_iron = 7.23)
  (specific_gravity_seawater = 1.05)
  (g = 9.81) : time_to_reach_bottom = 4.89 := sorry

end acceleration_of_iron_ball_kinetic_energy_of_iron_ball_time_to_reach_seafloor_l134_134143


namespace lower_rate_of_interest_l134_134994

def principal : ℝ := 5000
def time : ℕ := 2
def higher_rate : ℝ := 18
def interest_difference : ℝ := 600

def higher_interest (P : ℝ) (R : ℝ) (T : ℕ) := P * (R / 100) * T
def lower_interest (P : ℝ) (R : ℝ) (T : ℕ) := P * (R / 100) * T

theorem lower_rate_of_interest :
  ∃ R : ℝ, lower_interest principal R time + interest_difference = higher_interest principal higher_rate time ∧ R = 12 :=
by
  sorry

end lower_rate_of_interest_l134_134994


namespace sports_channels_cost_less_l134_134783

theorem sports_channels_cost_less (basic_service_cost movie_channels_cost total_cost : ℕ) (h1 : basic_service_cost = 15) (h2 : movie_channels_cost = 12) (h3 : total_cost = 36) : 
  let sports_channels_cost := total_cost - (basic_service_cost + movie_channels_cost) in
  movie_channels_cost - sports_channels_cost = 3 :=
by
  sorry

end sports_channels_cost_less_l134_134783


namespace product_sum_divisible_by_1987_l134_134059

theorem product_sum_divisible_by_1987 :
  let A : ℕ :=
    List.prod (List.filter (λ x => x % 2 = 1) (List.range (1987 + 1)))
  let B : ℕ :=
    List.prod (List.filter (λ x => x % 2 = 0) (List.range (1987 + 1)))
  A + B ≡ 0 [MOD 1987] := by
  -- The proof goes here
  sorry

end product_sum_divisible_by_1987_l134_134059


namespace find_poly_l134_134281

open Real

noncomputable def q(x : ℝ) := x^4 - 2*x^3 - 7*x + 6
noncomputable def p(x : ℝ) := 8*x^2 - 8

theorem find_poly : 
(∀ x : ℝ, q(x) / p(x) = (x^4 - 2*x^3 - 7*x + 6) / (8*x^2 - 8)) ∧ 
p 1 = 0 ∧ 
p (-1) = 0 ∧ 
∀ x : ℝ, ¬∃ l : ℝ, (tendsto (λ x, q(x) / p(x)) at_top (nhds l) ∧ tendsto (λ x, q(x) / p(x)) at_bot (nhds l)) ∧
p 2 = 24 :=
by
  sorry

end find_poly_l134_134281


namespace acute_triangle_P_gt_Q_l134_134695

theorem acute_triangle_P_gt_Q
  (A B C : ℝ)
  (h_acute_A : 0 < A ∧ A < π/2)
  (h_acute_B : 0 < B ∧ B < π/2)
  (h_acute_C : 0 < C ∧ C < π/2)
  (h_angle_sum : A + B + C = π) :
  let P := sin A + sin B
  let Q := cos A + cos B
  in P > Q :=
by
  sorry

end acute_triangle_P_gt_Q_l134_134695


namespace fixed_point_and_shortest_chord_length_l134_134681

open Real

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

noncomputable def line_eq (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem fixed_point_and_shortest_chord_length (m : ℝ) :
  (∀ x y : ℝ, line_eq m x y → x = 3 ∧ y = 1) ∧
  (∃ m : ℝ, m = -3/4 ∧ chord_length_shortest m) := sorry

noncomputable def chord_length_shortest (m : ℝ) : Prop :=
  let d := ((2-2 -5) / sqrt (5)).abs
  ∧ (∀ x y : ℝ, circle_eq x y → dist (1, 2) (x, y) = 5)
  ∧ (sqrt (25 - d^2)) = 2 * sqrt 5 := sorry

end fixed_point_and_shortest_chord_length_l134_134681


namespace sqrt_floor_19992000_l134_134657

theorem sqrt_floor_19992000 : (Int.floor (Real.sqrt 19992000)) = 4471 := by
  sorry

end sqrt_floor_19992000_l134_134657


namespace different_result_from_division_l134_134166

theorem different_result_from_division :
  (∛64 / 4 ≠ (√16 / 2)) ∧
  (∛64 / 4 = (2 * 2⁻¹)) ∧
  (∛64 / 4 = (4^2 * 4^3 * 4⁻⁵)) ∧
  (∛64 / 4 = 3^0) :=
by
  sorry

end different_result_from_division_l134_134166


namespace angle_x_l134_134025

-- Define the conditions
variables (O A C D B : Point)
variable h : circle_center O
variable h1 : on_circumference A C D O
variable h2 : diameter AC
variable h3 : ∠ CDA = 42

-- The problem statement we want to prove
theorem angle_x (x : ℝ) : x = 58 :=
begin
  -- This is where the solution steps would go, but for now, we leave it as sorry
  sorry
end

end angle_x_l134_134025


namespace fence_perimeter_l134_134930

-- Definitions for the given conditions
def num_posts : ℕ := 36
def gap_between_posts : ℕ := 6

-- Defining the proof problem to show the perimeter equals 192 feet
theorem fence_perimeter (n : ℕ) (g : ℕ) (sq_field : ℕ → ℕ → Prop) : n = 36 ∧ g = 6 ∧ sq_field n g → 4 * ((n / 4 - 1) * g) = 192 :=
by
  intro h
  sorry

end fence_perimeter_l134_134930


namespace three_perfect_games_score_l134_134181

theorem three_perfect_games_score :
  let perfect_score := 21
  in 3 * perfect_score = 63 :=
by
  let perfect_score := 21
  show 3 * perfect_score = 63
  sorry

end three_perfect_games_score_l134_134181


namespace tank_filling_time_l134_134297

theorem tank_filling_time :
  let fill_rate_A := 1 / 20
  let fill_rate_B := 1 / 30
  let empty_rate_C := 1 / 15
  let fill_rate_D := 1 / 40
  let cycle_time := 4 * (fill_rate_A + fill_rate_B - empty_rate_C + fill_rate_D)
  let total_cycles := 1 / cycle_time
  let total_time := total_cycles * 16
  total_time = 160 := by
  let fill_rate_A := 1 / 20
  let fill_rate_B := 1 / 30
  let empty_rate_C := 1 / 15
  let fill_rate_D := 1 / 40
  let cycle_time := 4 * (fill_rate_A + fill_rate_B - empty_rate_C + fill_rate_D)
  let total_cycles := 1 / cycle_time
  let total_time := total_cycles * 16
  have fill_A := 4 * fill_rate_A
  have fill_B := 4 * fill_rate_B
  have empty_C := 4 * empty_rate_C
  have fill_D := 4 * fill_rate_D
  have net_fill := fill_A + fill_B - empty_C + fill_D
  have cycle_fill := net_fill
  have cycles := 1 / cycle_fill
  have time := cycles * 16
  show total_time = 160 from sorry

end tank_filling_time_l134_134297


namespace Alice_silver_tokens_l134_134223

namespace AliceTokens

def initial_red_tokens : Nat := 90
def initial_blue_tokens : Nat := 80

def red_from_first_booth (x : Nat) : Int := -3 * x
def blue_from_first_booth (x : Nat) : Int := 2 * x
def silver_from_first_booth (x : Nat) : Int := x

def blue_from_second_booth (y : Nat) : Int := -4 * y
def red_from_second_booth (y : Nat) : Int := 2 * y
def silver_from_second_booth (y : Nat) : Int := y

def total_red (x y : Nat) : Int := initial_red_tokens + red_from_first_booth x + red_from_second_booth y
def total_blue (x y : Nat) : Int := initial_blue_tokens + blue_from_first_booth x + blue_from_second_booth y

def total_silver (x y : Nat) : Int := silver_from_first_booth x + silver_from_second_booth y

theorem Alice_silver_tokens : total_silver 21 16 = 37 :=
by
  have initial_red : Int := initial_red_tokens
  have initial_blue : Int := initial_blue_tokens
  have x : Int := 21
  have y : Int := 16

  have red_after_first_booth : Int := initial_red - 3 * x
  have red_after_both_booths : Int := red_after_first_booth + 2 * y

  have blue_after_first_booth : Int := initial_blue + 2 * x
  have blue_after_both_booths : Int := blue_after_first_booth - 4 * y

  have total_silvers : Int := x + y
  have red_condition : red_after_both_booths < 3 := by sorry
  have blue_condition : blue_after_both_booths < 4 := by sorry

  show total_silvers = 37 from
    sorry

end AliceTokens

end Alice_silver_tokens_l134_134223


namespace length_of_FD_l134_134415

variable (A B C D E F : Type)
variables [HasAngle A B C D E F] [HasLength A B C D E F]

def isParallelogram (ABCD : Type) [HasAngle ABCD] (A B C D : ABCD) : Prop :=
  angle A B C = angle C D A

noncomputable def length_FD_approximately (A B C D E F : Type) [HasAngle A B C D E F] [HasLength A B C D E F]
  (ABCD : A B C D) (AB : ℝ) (BC : ℝ) (CE : ℝ) (angle_ABC : ℝ) (intersection_AE_BD : A E B D → Prop) : Prop :=
  let F := intersection_AE_BD(A E B D) in
  let BD := BC in
  let BE := BC + CE in
  let ratio := CE / BE in
  let FD := ratio * BD in
  FD ≈ 4.3

variables (ABCD : Type) [HasAngle ABCD] [HasLength A B C D E F]
variables (AB: ℝ) (BC: ℝ) (CE: ℝ). 
variables (angle_ABC: ℝ) (intersection_AE_BD : Type)

theorem length_of_FD (A B C D E F : Type) 
  [HasAngle A B C D E F] [HasLength A B C D E F] (ABCD : Type)
  (AB: ℝ) (BC: ℝ) (CE: ℝ) (angle_ABC: ℝ) 
  (intersection_AE_BD : A E B D → Prop) (F : intersection_AE_BD(A E B D))
  [isParallelogram ABCD A B C D] :
  length_FD_approximately A B C D E F (ABCD) (AB) (BC) (CE) (angle_ABC) (intersection_AE_BD) :=
  sorry

end length_of_FD_l134_134415


namespace tens_digit_13_power_1987_l134_134651

theorem tens_digit_13_power_1987 : (13^1987)%100 / 10 = 1 :=
by
  sorry

end tens_digit_13_power_1987_l134_134651


namespace total_wheels_in_garage_l134_134918

def bicycles: Nat := 3
def tricycles: Nat := 4
def unicycles: Nat := 7

def wheels_per_bicycle: Nat := 2
def wheels_per_tricycle: Nat := 3
def wheels_per_unicycle: Nat := 1

theorem total_wheels_in_garage (bicycles tricycles unicycles wheels_per_bicycle wheels_per_tricycle wheels_per_unicycle : Nat) :
  bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle = 25 := by
  sorry

end total_wheels_in_garage_l134_134918


namespace show_length_50_l134_134298

def Gina_sSis_three_as_often (G S : ℕ) : Prop := G = 3 * S
def sister_total_shows (G S : ℕ) : Prop := G + S = 24
def Gina_total_minutes (G : ℕ) (minutes : ℕ) : Prop := minutes = 900
def length_of_each_show (minutes shows length : ℕ) : Prop := length = minutes / shows

theorem show_length_50 (G S : ℕ) (length : ℕ) :
  Gina_sSis_three_as_often G S →
  sister_total_shows G S →
  Gina_total_minutes G 900 →
  length_of_each_show 900 G length →
  length = 50 :=
by
  intros h1 h2 h3 h4
  sorry

end show_length_50_l134_134298


namespace angle_x_l134_134027

-- Define the conditions
variables (O A C D B : Point)
variable h : circle_center O
variable h1 : on_circumference A C D O
variable h2 : diameter AC
variable h3 : ∠ CDA = 42

-- The problem statement we want to prove
theorem angle_x (x : ℝ) : x = 58 :=
begin
  -- This is where the solution steps would go, but for now, we leave it as sorry
  sorry
end

end angle_x_l134_134027


namespace polynomial_smallest_e_l134_134115

theorem polynomial_smallest_e :
  ∃ (a b c d e : ℤ), (a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ∧ a ≠ 0 ∧ e > 0 ∧ (x + 3) * (x - 6) * (x - 10) * (2 * x + 1) = 0) 
  ∧ e = 180 :=
by
  sorry

end polynomial_smallest_e_l134_134115


namespace part_one_l134_134635

def S_r (r n : ℕ) : ℕ := -- The definition of S_r should be provided here, but for statement purpose it is abstracted.

theorem part_one (r : ℕ) (hr : r > 2) : ∃ p : ℕ, Prime p ∧ ∀ n : ℕ, S_r r n ≡ n [MOD p] :=
sorry

end part_one_l134_134635


namespace increasing_function_l134_134539

def fA (x : ℝ) : ℝ := -x
def fB (x : ℝ) : ℝ := (2 / 3) ^ x
def fC (x : ℝ) : ℝ := x ^ 2
def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function (x y : ℝ) (h : x < y) : fD x < fD y := sorry

end increasing_function_l134_134539


namespace earnings_pool_cleaning_correct_l134_134556

-- Definitions of the conditions
variable (Z : ℕ) -- Number of times Zoe babysat Zachary
variable (earnings_total : ℝ := 8000) 
variable (earnings_Zachary : ℝ := 600)
variable (earnings_per_session : ℝ := earnings_Zachary / Z)
variable (sessions_Julie : ℕ := 3 * Z)
variable (sessions_Chloe : ℕ := 5 * Z)

-- Calculation of earnings from babysitting
def earnings_Julie : ℝ := sessions_Julie * earnings_per_session
def earnings_Chloe : ℝ := sessions_Chloe * earnings_per_session
def earnings_babysitting_total : ℝ := earnings_Zachary + earnings_Julie + earnings_Chloe

-- Calculation of earnings from pool cleaning
def earnings_pool_cleaning : ℝ := earnings_total - earnings_babysitting_total

-- The theorem we are interested in
theorem earnings_pool_cleaning_correct :
  earnings_pool_cleaning Z = 2600 := by
  sorry

end earnings_pool_cleaning_correct_l134_134556


namespace expected_number_of_males_in_sample_l134_134220

theorem expected_number_of_males_in_sample : 
  let total_athletes := 48 + 36
  let male_ratio := 48 / total_athletes.to_rat
  let sample_size := 21
  let expected_males := male_ratio * sample_size
  expected_males = 12 :=
by
  let total_athletes := 48 + 36
  let male_ratio := 48 / total_athletes.to_rat
  let sample_size := 21
  let expected_males := male_ratio * sample_size
  have h : total_athletes = 84 := rfl
  have r : male_ratio = 4 / 7 := by norm_num [total_athletes, male_ratio]
  have s : expected_males = (4 / 7) * 21 := by simp [male_ratio, sample_size]
  have result : expected_males = 12 := by norm_num [s]
  exact result

end expected_number_of_males_in_sample_l134_134220


namespace P_gt_Q_l134_134739

variable (x : ℝ)

def P := x^2 + 2
def Q := 2 * x

theorem P_gt_Q : P x > Q x := by
  sorry

end P_gt_Q_l134_134739


namespace carol_initial_cupcakes_l134_134295

/--
For the school bake sale, Carol made some cupcakes. She sold 9 of them and then made 28 more.
Carol had 49 cupcakes. We need to show that Carol made 30 cupcakes initially.
-/
theorem carol_initial_cupcakes (x : ℕ) 
  (h1 : x - 9 + 28 = 49) : 
  x = 30 :=
by 
  -- The proof is not required as per instruction.
  sorry

end carol_initial_cupcakes_l134_134295


namespace remainder_of_sum_of_squares_mod_l134_134958

-- Define the function to compute the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Define the specific sum for the first 15 natural numbers
def S : ℕ := sum_of_squares 15

-- State the theorem
theorem remainder_of_sum_of_squares_mod (n : ℕ) (h : n = 15) : 
  S % 13 = 5 := by
  sorry

end remainder_of_sum_of_squares_mod_l134_134958


namespace number_of_foons_correct_l134_134209

-- Define the conditions
def area : ℝ := 5  -- Area in cm^2
def thickness : ℝ := 0.5  -- Thickness in cm
def total_volume : ℝ := 50  -- Total volume in cm^3

-- Define the proof problem
theorem number_of_foons_correct :
  (total_volume / (area * thickness) = 20) :=
by
  -- The necessary computation would go here, but for now we'll use sorry to indicate the outcome
  sorry

end number_of_foons_correct_l134_134209


namespace area_of_circle_l134_134147

def circleEquation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

theorem area_of_circle :
  (∃ (center : ℝ × ℝ) (radius : ℝ), radius = 4 ∧ ∀ (x y : ℝ), circleEquation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) →
  (16 * Real.pi) = 16 * Real.pi := 
by 
  intro h
  have := h
  sorry

end area_of_circle_l134_134147


namespace angle_x_l134_134028

-- Define the conditions
variables (O A C D B : Point)
variable h : circle_center O
variable h1 : on_circumference A C D O
variable h2 : diameter AC
variable h3 : ∠ CDA = 42

-- The problem statement we want to prove
theorem angle_x (x : ℝ) : x = 58 :=
begin
  -- This is where the solution steps would go, but for now, we leave it as sorry
  sorry
end

end angle_x_l134_134028


namespace unrepresentable_integers_l134_134263

theorem unrepresentable_integers :
    {n : ℕ | ∀ a b : ℕ, a > 0 → b > 0 → n ≠ (a * (b + 1) + (a + 1) * b) / (b * (b + 1)) } =
    {1} ∪ {n | ∃ m : ℕ, n = 2^m + 2} :=
by
    sorry

end unrepresentable_integers_l134_134263


namespace rectangular_to_cylindrical_l134_134632

theorem rectangular_to_cylindrical (x y z r θ : ℝ) (h₁ : x = 3) (h₂ : y = -3 * Real.sqrt 3) (h₃ : z = 2)
  (h₄ : r = Real.sqrt (3^2 + (-3 * Real.sqrt 3)^2)) 
  (h₅ : r > 0) 
  (h₆ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₇ : θ = Float.pi * 5 / 3) : 
  (r, θ, z) = (6, 5 * Float.pi / 3, 2) :=
sorry

end rectangular_to_cylindrical_l134_134632


namespace angle_x_degrees_l134_134043

noncomputable def center_of_circle (O : Point) := true
noncomputable def radii (O B C A : Point) := (dist O B = dist O C) ∧ (dist O A = dist O C)
noncomputable def isosceles_triangle (O B C : Point) := (∠ B C O = 32) ∧ (∠ O B C = 32)
noncomputable def isosceles_base_angle (O A C : Point) := ∠ O A C = ∠ A O C = 23
noncomputable def angle_sum_property (O A C : Point) := ∠ A O C = 90

theorem angle_x_degrees
  (O B C A : Point) (h_radii : radii O B C A)
  (h_isosceles_triangle : isosceles_triangle O B C)
  (h_isosceles_base : isosceles_base_angle O A C)
  (h_sum_property : angle_sum_property O A C) :
  ∠ B C O - ∠ O A C = 9 := by sorry

end angle_x_degrees_l134_134043


namespace collinear_points_trajectory_of_point_Q_l134_134706

open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)

def on_parabola (P : Point) : Prop :=
  P.y ^ 2 = P.x

def vector_dot (A B : Point) : ℝ :=
  A.x * B.x + A.y * B.y

def collinear (A B C : Point) : Prop :=
  ∃ k : ℝ, (A.x, A.y) = (k * (B.x, B.y)) ∧ (B.x, B.y) = (k * (C.x, C.y))

theorem collinear_points (A B C : Point) (hA : on_parabola A) (hB : on_parabola B)
  (hA_non_origin : A ≠ ⟨0, 0⟩) (hB_non_origin : B ≠ ⟨0, 0⟩) (hB_distinct_A : A ≠ B)
  (h_dot_product_zero : vector_dot A B = 0) :
  collinear A C B :=
sorry

def trajectory_eq (Q : Point) : Prop :=
  (Q.x - 1 / 2) ^ 2 + Q.y ^ 2 = 1 / 4 ∧ Q.x ≠ 0

theorem trajectory_of_point_Q (A B Q : Point) (λ : ℝ) (h_A_on_parabola : on_parabola A)
  (h_B_on_parabola : on_parabola B) (hA_non_origin : A ≠ ⟨0, 0⟩)
  (hB_non_origin : B ≠ ⟨0, 0⟩) (hB_distinct_A : A ≠ B)
  (h_dot_product_zero : vector_dot A B = 0)
  (h_Q_relation : Q = ⟨λ * (B.x - A.x), λ * (B.y - A.y)⟩)
  (h_dot_oq_ab : vector_dot ⟨Q.x, Q.y⟩ ⟨A.x - B.x, A.y - B.y⟩ = 0) :
  trajectory_eq Q :=
sorry

end collinear_points_trajectory_of_point_Q_l134_134706


namespace volume_reflected_tetrahedron_l134_134313

-- Define the tetrahedron vertex types
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the conditions
noncomputable def original_tetrahedron : Point × Point × Point × Point :=
  (⟨0, 0, 0⟩, ⟨1, 0, 0⟩, ⟨1/2, √3/2, 0⟩, ⟨1/2, 1/(2√3), √(2/3)⟩)

-- Reflection operations
noncomputable def reflect (p q r : Point) : Point :=
  ⟨2 * q.x - p.x, 2 * q.y - p.y, 2 * q.z - p.z⟩

-- The reflected vertices
noncomputable def A' : Point := reflect (original_tetrahedron.1.1) (original_tetrahedron.1.2)
noncomputable def B' : Point := reflect (original_tetrahedron.1.2) (original_tetrahedron.1.3)
noncomputable def C' : Point := reflect (original_tetrahedron.1.3) (original_tetrahedron.1.4)
noncomputable def D' : Point := reflect (original_tetrahedron.1.4) (original_tetrahedron.1.1)

-- Define a function to calculate the volume of a tetrahedron given four points
noncomputable def volume_tetrahedron (A B C D : Point) : ℝ :=
  let v321 := B.x * C.y * D.z in
  let v231 := B.x * D.y * C.z in
  let v312 := C.x * B.y * D.z in
  let v132 := C.x * D.y * B.z in
  let v213 := D.x * B.y * C.z in
  let v123 := D.x * C.y * B.z in
  1/6 * abs (A.x * (v321 - v231) + A.y * (v132 - v312) + A.z * (v213 - v123))

-- Given conditions
def edge_length_1 (A B : Point) : Prop := 
  (A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2 = 1

-- Proof problem
theorem volume_reflected_tetrahedron : 
  edge_length_1 (original_tetrahedron.1.1) (original_tetrahedron.1.2) ∧
  edge_length_1 (original_tetrahedron.1.1) (original_tetrahedron.1.3) ∧
  edge_length_1 (original_tetrahedron.1.1) (original_tetrahedron.1.4) ∧
  edge_length_1 (original_tetrahedron.1.2) (original_tetrahedron.1.3) ∧
  edge_length_1 (original_tetrahedron.1.2) (original_tetrahedron.1.4) ∧
  edge_length_1 (original_tetrahedron.1.3) (original_tetrahedron.1.4) →
  volume_tetrahedron A' B' C' D' = 15 :=
by
  sorry

end volume_reflected_tetrahedron_l134_134313


namespace angle_x_is_58_l134_134016

-- Definitions and conditions based on the problem statement
variables {O A C D : Type} (x : ℝ)
variable [circular_center : O] -- O is the center of the circle
variable [radius_equal : AO = BO ∧ BO = CO] -- AO, BO, and CO are radii and hence equal
variable (angle_ACD_right : ∠ A C D = 90) -- ∠ACD is a right angle (subtended by diameter)
variable (angle_CDA : ∠ C D A = 42) -- ∠CDA is 42 degrees

-- Statement to be proven
theorem angle_x_is_58 (x : ℝ) : x = 58 := 
by sorry

end angle_x_is_58_l134_134016


namespace f_at_one_is_zero_f_is_increasing_range_of_x_l134_134309

open Function

-- Define the conditions
variable {f : ℝ → ℝ}
variable (h1 : ∀ x > 1, f x > 0)
variable (h2 : ∀ x y, f (x * y) = f x + f y)

-- Problem Statements
theorem f_at_one_is_zero : f 1 = 0 := 
sorry

theorem f_is_increasing (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (h : x₁ > x₂) : 
  f x₁ > f x₂ := 
sorry

theorem range_of_x (f3_eq_1 : f 3 = 1) (x : ℝ) (h3 : x ≥ 1 + Real.sqrt 10) : 
  f x - f (1 / (x - 2)) ≥ 2 := 
sorry

end f_at_one_is_zero_f_is_increasing_range_of_x_l134_134309


namespace local_maximum_at_x1_l134_134105

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 3)

theorem local_maximum_at_x1 : ∃ m : ℝ, (∀ x : ℝ, x ≠ 1 → ∀ ε > 0, (abs (x - 1) < ε → f x ≤ f 1)) :=
begin
  use 1,
  intros x hx ε hε h,
  sorry
end

end local_maximum_at_x1_l134_134105


namespace angle_FME_eq_angle_PCQ_l134_134425

-- Definition of the problem setup and main theorem

variables {A B C D E F G P Q M : Type} [Point A] [Point B] [Point C] [Point D] [Point E] [Point F] [Point G] [Point P] [Point Q] [Point M]
  [circumcircle : Circle (triangle A B C)]
  [circumcircle_P : onCircle P circumcircle]
  [circumcircle_Q : onCircle Q circumcircle]
  [Simson_line_P : Simson line P (triangle A B C) = DE]
  [Simson_line_Q: Simson line Q (triangle A B C) = FG ]
  [intersection_M: intersectLines DE FG M]

/-- Prove that the angle between the intersection of Simson lines DE and FG at point M is equal to the angle PCQ -/
theorem angle_FME_eq_angle_PCQ :
  angle M F E = angle P C Q :=
  sorry

end angle_FME_eq_angle_PCQ_l134_134425


namespace earnings_in_six_minutes_l134_134834

def ticket_cost : ℕ := 3
def tickets_per_minute : ℕ := 5
def minutes : ℕ := 6

theorem earnings_in_six_minutes (ticket_cost : ℕ) (tickets_per_minute : ℕ) (minutes : ℕ) :
  ticket_cost = 3 → tickets_per_minute = 5 → minutes = 6 → (tickets_per_minute * ticket_cost * minutes = 90) :=
by
  intros h_cost h_tickets h_minutes
  rw [h_cost, h_tickets, h_minutes]
  norm_num

end earnings_in_six_minutes_l134_134834


namespace leif_fruit_problem_l134_134789

theorem leif_fruit_problem :
  ∀ (apple_weight_per_apple orange_weight_per_orange : ℕ)
    (total_apple_weight : ℕ) (dozen : ℕ)
    (total_oranges_in_dozen : ℕ) (oranges_per_dozen : ℕ),
    apple_weight_per_apple = 60 →
    orange_weight_per_orange = 100 →
    total_apple_weight = 780 →
    dozen = 3 →
    oranges_per_dozen = 12 →
    total_oranges_in_dozen = dozen * oranges_per_dozen →
    (total_oranges_in_dozen * orange_weight_per_orange -
     total_apple_weight) = 2820 :=
by
  intros
  rw [← nat.mul_sub_right_distrib, mul_comm]
  sorry

end leif_fruit_problem_l134_134789


namespace aunt_ming_total_cost_l134_134674

theorem aunt_ming_total_cost : 
  let original_ticket_price := 11.25
  let senior_discount_rate := 0.2
  let child_discount_rate := 0.3
  let senior_price := original_ticket_price * (1 - senior_discount_rate)
  let child_price := original_ticket_price * (1 - child_discount_rate)
  let adult_ticket_price := original_ticket_price
  let total_cost := senior_price + 2 * adult_ticket_price + child_price
  in senior_price = 9.0 → total_cost = 39.375 :=
by 
  sorry

end aunt_ming_total_cost_l134_134674


namespace shaded_area_of_overlap_l134_134521

structure Rectangle where
  width : ℕ
  height : ℕ

structure Parallelogram where
  base : ℕ
  height : ℕ

def area_of_rectangle (r : Rectangle) : ℕ :=
  r.width * r.height

def area_of_parallelogram (p : Parallelogram) : ℕ :=
  p.base * p.height

def overlapping_area_square (side : ℕ) : ℕ :=
  side * side

theorem shaded_area_of_overlap 
  (r : Rectangle)
  (p : Parallelogram)
  (overlapping_side : ℕ)
  (h1 : r.width = 4)
  (h2 : r.height = 12)
  (h3 : p.base = 10)
  (h4 : p.height = 4)
  (h5 : overlapping_side = 4) :
  area_of_rectangle r + area_of_parallelogram p - overlapping_area_square overlapping_side = 72 :=
by
  sorry

end shaded_area_of_overlap_l134_134521


namespace quadratic_root_proof_l134_134750

noncomputable def root_condition (p q m n : ℝ) :=
  ∃ x : ℝ, x^2 + p * x + q = 0 ∧ x ≠ 0 ∧ (1/x)^2 + m * (1/x) + n = 0

theorem quadratic_root_proof (p q m n : ℝ) (h : root_condition p q m n) :
  (pn - m) * (qm - p) = (qn - 1)^2 :=
sorry

end quadratic_root_proof_l134_134750


namespace arrangement_of_bananas_l134_134645

-- Define the constants for the number of letters and repetitions in the word BANANAS.
def num_letters : ℕ := 7
def count_A : ℕ := 3
def count_N : ℕ := 2
def factorial (n : ℕ) := nat.factorial n

-- The number of ways to arrange the letters of the word BANANAS.
noncomputable def num_ways_to_arrange := 
  (factorial num_letters) / (factorial count_A * factorial count_N)

theorem arrangement_of_bananas : 
  num_ways_to_arrange = 420 :=
sorry

end arrangement_of_bananas_l134_134645


namespace find_angle_l134_134421

noncomputable def angle_between_vecs (a b c : ℝ^3) (h₁ : ‖a‖ = 1) (h₂ : ‖b‖ = 1) (h₃ : ‖c‖ = 2)
  (h₄ : a × (b × c) = (b + 2 • c) / Real.sqrt 3) (h₅ : LinearIndependent ℝ ![a,b,c]) : ℝ :=
  Real.arccos ((a ⬝ c) / (‖a‖ * ‖c‖))

theorem find_angle {a b c : ℝ^3} (h₁ : ‖a‖ = 1) (h₂ : ‖b‖ = 1) (h₃ : ‖c‖ = 2)
  (h₄ : a × (b × c) = (b + 2 • c) / Real.sqrt 3) 
  (h₅ : LinearIndependent ℝ ![a,b,c]) :
  angle_between_vecs a b c h₁ h₂ h₃ h₄ h₅ = 70.53 :=
sorry

end find_angle_l134_134421


namespace sum_of_squares_of_solution_set_l134_134478

noncomputable def sum_of_squares_of_roots (eq : ℝ → ℝ) := 
  {x : ℝ | eq x = 0}.to_finset.sum (λ x, x^2)

theorem sum_of_squares_of_solution_set :
  sum_of_squares_of_roots (λ x, x^2 + x - 1 - (x * Real.exp (x^2 - 1) + (x^2 - 1) * Real.exp x)) = 2 :=
sorry

end sum_of_squares_of_solution_set_l134_134478


namespace smallest_two_digit_multiple_of_17_l134_134959

theorem smallest_two_digit_multiple_of_17 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ 17 ∣ n ∧ ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ 17 ∣ m → n ≤ m :=
begin
  use 17,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { use 1,
    norm_num },
  intros m h1 h2 h3,
  rw ← nat.dvd_iff_mod_eq_zero at *,
  have h4 := nat.mod_eq_zero_of_dvd h3,
  cases (nat.le_of_mod_eq_zero h4),
  { linarith [nat.le_of_dvd (dec_trivial) this] },
  { exfalso,
    linarith }
end

end smallest_two_digit_multiple_of_17_l134_134959


namespace exactly_one_red_probability_l134_134198

open Finset

/-- Given a bag containing 2 red balls and 2 white balls,
    the probability of drawing exactly one red ball when 2 balls are drawn at random
    is 2/3.
-/
noncomputable def probability_exactly_one_red : ℚ :=
  let red_draws := 2
  let white_draws := 2
  let total_draws := red_draws + white_draws
  let draw_two := (total_draws.choose 2)
  let favorable_draws := (red_draws.choose 1) * (white_draws.choose 1)
  let probability := favorable_draws / draw_two
  probability

theorem exactly_one_red_probability :
  probability_exactly_one_red = 2/3 :=
by
  unfold probability_exactly_one_red
  sorry

end exactly_one_red_probability_l134_134198


namespace investment_difference_l134_134787

noncomputable def future_value_semi_annual (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + annual_rate / 2)^((years * 2))

noncomputable def future_value_monthly (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + annual_rate / 12)^((years * 12))

theorem investment_difference :
  let jose_investment := future_value_semi_annual 30000 0.03 3
  let patricia_investment := future_value_monthly 30000 0.025 3
  round (jose_investment) - round (patricia_investment) = 317 :=
by
  sorry

end investment_difference_l134_134787


namespace BANANAS_arrangement_l134_134647

open Nat

theorem BANANAS_arrangement :
  let total_letters := 7
  let freq_A := 3
  let freq_N := 2
  fact total_letters / (fact freq_A * fact freq_N) = 420 :=
by 
  sorry

end BANANAS_arrangement_l134_134647


namespace isothermal_compression_work_l134_134204

-- Definitions of initial conditions, functions and required work calculation

def R : ℝ := 0.2            -- Radius in meters
def H : ℝ := 0.8            -- Initial height in meters
def h : ℝ := 0.7            -- Piston movement in meters
def p0 : ℝ := 103.3 * 10^3  -- Initial pressure in Pascals
def S : ℝ := Math.pi * R^2  -- Area of the piston

-- Volume as a function of piston displacement x
def V (x : ℝ) : ℝ := S * (H - x)

-- Pressure as a function of x
def p (x : ℝ) : ℝ := p0 * (S * H / V(x))

-- Force as a function of x
def F (x : ℝ) : ℝ := p(x) * S

-- Integral representing the work done during the isothermal compression
def work_done : ℝ := ∫ (x : ℝ) in 0..h, F(x)

-- Final theorem statement
theorem isothermal_compression_work : work_done ≈ 21595 :=
by
  sorry

end isothermal_compression_work_l134_134204


namespace intersection_M_N_l134_134503

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (-1, 1) + x • (1, 2)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ := (1, -2) + y • (2, 3)

def set_M := {a : ℝ × ℝ | ∃ x : ℝ, a = vector_a x}
def set_N := {a : ℝ × ℝ | ∃ y : ℝ, a = vector_b y}

theorem intersection_M_N : M = {(ℝ, ℝ) | (ℝ, ℝ) == (-13, -23)} := sorry

end intersection_M_N_l134_134503


namespace largest_of_seven_consecutive_l134_134896

theorem largest_of_seven_consecutive (n : ℕ) 
  (h1: n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 3010) :
  n + 6 = 433 :=
by 
  sorry

end largest_of_seven_consecutive_l134_134896


namespace geometric_sequence_property_l134_134390

-- Assuming a geometric sequence
variables (b : ℕ → ℝ) 

-- Assuming b_7 = 1
def b7_eq_1 : Prop := b 7 = 1

-- n is a natural number and n < 13
def n_lt_13 (n : ℕ) : Prop := n < 13

-- Define the product of terms from 1 to k
noncomputable def partial_product (b : ℕ → ℝ) (k : ℕ) : ℝ := ∏ i in Finset.range k, b (i + 1)

-- The statement to prove
theorem geometric_sequence_property (n : ℕ) (h1 : n_lt_13 n) (h2 : b7_eq_1) :
  partial_product b n = partial_product b (13 - n) :=
begin
  sorry
end

end geometric_sequence_property_l134_134390


namespace four_digit_pairs_product_three_zeros_l134_134665

theorem four_digit_pairs_product_three_zeros :
  ∃ (A B C D : Nat), 
  (A ≠ 0 ∧ D ≠ 0) ∧
  (0 ≤ A ∧ A ≤ 9) ∧
  (0 ≤ B ∧ B ≤ 9) ∧
  (0 ≤ C ∧ C ≤ 9) ∧
  (0 ≤ D ∧ D ≤ 9) ∧
  ((10^3 * A + 10^2 * B + 10 * C + D) * (10^3 * D + 10^2 * C + 10 * B + A) % 1000 = 0) ∧
  (
    (A B C D = [5, 2, 1, 6]) ∨ 
    (A B C D = [5, 7, 3, 6]) ∨ 
    (A B C D = [5, 2, 6, 4]) ∨ 
    (A B C D = [5, 7, 8, 4])
  ) := by
  sorry

end four_digit_pairs_product_three_zeros_l134_134665


namespace additional_grassy_ground_l134_134596

noncomputable def pi : ℝ := Real.pi

def rope_length_original : ℝ := 12
def rope_length_increased : ℝ := 25

def area (r : ℝ) : ℝ := pi * r ^ 2

theorem additional_grassy_ground :
  let A1 := area rope_length_original
  let A2 := area rope_length_increased
  A2 - A1 = 481 * pi :=
by
  let A1 := area rope_length_original
  let A2 := area rope_length_increased
  calc
    A2 - A1 = pi * rope_length_increased ^ 2 - pi * rope_length_original ^ 2 : by sorry
    ... = 625 * pi - 144 * pi : by sorry
    ... = (625 - 144) * pi : by sorry
    ... = 481 * pi : by refl

end additional_grassy_ground_l134_134596


namespace divide_figure_into_two_congruent_parts_l134_134653

-- Definition of a figure with N cells
variables (N : ℕ) (figure : set (ℕ × ℕ))

-- Define a function to remove a cell and result in a new figure
def remove_cell (figure : set (ℕ × ℕ)) (cell : ℕ × ℕ) : set (ℕ × ℕ) :=
  figure \ {cell}

-- We need to prove that there exists a cell removal such that the remaining figure can be divided
-- into two congruent parts each containing (N-1)/2 cells
theorem divide_figure_into_two_congruent_parts (N : ℕ) (figure : set (ℕ × ℕ))
  (h : N > 1) :
  ∃ cell : ℕ × ℕ, ∃ part1 part2 : set (ℕ × ℕ),
  remove_cell figure cell = part1 ∪ part2 ∧
  part1 ≠ ∅ ∧ part2 ≠ ∅ ∧ part1 ≈ part2 ∧
  card part1 = card part2 ∧
  card part1 = (N - 1) / 2 ∧ card part2 = (N - 1) / 2 := sorry

end divide_figure_into_two_congruent_parts_l134_134653


namespace correct_statements_l134_134227

theorem correct_statements :
  (∀ θ : ℝ, θ ∈ (λ k : ℤ, 2 * k * Real.pi + Real.pi / 2) ↔ θ ∈ (λ n : ℤ, n * Real.pi + Real.pi / 2) → false) →
  (let f : ℝ → ℝ := λ x, 2 * Real.cos(x - Real.pi / 4) in
  (∃ k : ℤ, k * Real.pi + 3 * Real.pi / 4 = (3 * Real.pi / 4) ∧ f (3 * Real.pi / 4) = 0) → true) →
  (∃ x₁ x₂ : ℝ, x₁ ∈ set.Ioo 0 (Real.pi / 2) ∧ x₂ ∈ set.Ioo 0 (Real.pi / 2) ∧
  x₁ > x₂ ∧ Real.tan x₁ < Real.tan x₂) → false) →
  (∀ x : ℝ, Real.sin(2 * x - Real.pi / 3) = Real.sin(2 * (x - Real.pi / 6)) → true) →
  true :=
by sorry

end correct_statements_l134_134227


namespace QRS_percent_decrease_is_20_l134_134118

noncomputable def QRS_percent_decrease (P : ℝ) : ℝ :=
  let profit_april := P * 1.10
  let profit_may := profit_april * (1 - x / 100)
  let profit_june := profit_may * 1.50
  let profit_june_expected := P * 1.3200000000000003 in
  (profit_april - profit_may) / profit_april * 100

theorem QRS_percent_decrease_is_20 (P : ℝ) (h : P > 0) : QRS_percent_decrease P = 20 := by
  -- QRS_percent_decrease translates the natural expression in the problem.
  sorry

end QRS_percent_decrease_is_20_l134_134118


namespace find_digits_of_large_number_l134_134987

theorem find_digits_of_large_number :
  ∃ (a : Fin 12 → ℕ), (197719771977 = ∑ i, a i * (10 ^ (i : ℕ))) ∧ (∀ i, a i ≤ 9) ∧ (∑ i, a i = 72) :=
by
  sorry

end find_digits_of_large_number_l134_134987


namespace angle_x_is_58_l134_134015

-- Definitions and conditions based on the problem statement
variables {O A C D : Type} (x : ℝ)
variable [circular_center : O] -- O is the center of the circle
variable [radius_equal : AO = BO ∧ BO = CO] -- AO, BO, and CO are radii and hence equal
variable (angle_ACD_right : ∠ A C D = 90) -- ∠ACD is a right angle (subtended by diameter)
variable (angle_CDA : ∠ C D A = 42) -- ∠CDA is 42 degrees

-- Statement to be proven
theorem angle_x_is_58 (x : ℝ) : x = 58 := 
by sorry

end angle_x_is_58_l134_134015


namespace complete_square_solution_l134_134164

-- Define the initial equation 
def equation_to_solve (x : ℝ) : Prop := x^2 - 4 * x = 6

-- Define the transformed equation after completing the square
def transformed_equation (x : ℝ) : Prop := (x - 2)^2 = 10

-- Prove that solving the initial equation using completing the square results in the transformed equation
theorem complete_square_solution : 
  ∀ x : ℝ, equation_to_solve x → transformed_equation x := 
by
  -- Proof will be provided here
  sorry

end complete_square_solution_l134_134164


namespace min_distance_between_circles_on_cube_l134_134668

-- Define the length of the edge of cube
def edge_length : ℝ := 1

-- Define the radii of the two relevant spheres
def radius_sphere_intersecting_edges : ℝ := real.sqrt 2 / 2
def radius_circumsphere : ℝ := real.sqrt 3 / 2

-- Define the minimum distance between the points on these spheres
def minimum_distance : ℝ := 1 / 2 * (real.sqrt 3 - real.sqrt 2)

-- Define the theorem to prove
theorem min_distance_between_circles_on_cube :
  ∀ (ABCD A1B1C1D1 : Type) (P Q : ABCD → ℝ^3) (R S : A1B1C1D1 → ℝ^3),
    minimum_distance = 1 / 2 * (real.sqrt 3 - real.sqrt 2) :=
by sorry

end min_distance_between_circles_on_cube_l134_134668


namespace area_quadrilateral_ABEF_l134_134481

open Real

def parabola_y_squared_eq_4x (x y : ℝ) : Prop :=
  y^2 = 4 * x

def F := (1 : ℝ, 0 : ℝ)

def line_through_F_incl_60 (x y : ℝ) : Prop :=
  y = sqrt 3 * (x - 1)

def A_on_parabola (x y : ℝ) : Prop :=
  parabola_y_squared_eq_4x x y ∧ line_through_F_incl_60 x y ∧ x > 1

def B_foot_of_perpendicular (xA yA xB yB : ℝ) : Prop :=
  yB = 0 ∧ xB = xA

def E_on_x_axis (x y : ℝ) : Prop :=
  y = 0

noncomputable def area_of_ABEF (xA yA xB yB xE yE : ℝ) : ℝ :=
  abs (((xB - xE) * (yA + yE) - (xA + xE) * (yB - yE) + (xA - xB) * (yE - yA)) / 2)

theorem area_quadrilateral_ABEF :
  ∃ (xA yA xB yB xE yE : ℝ),
    A_on_parabola xA yA ∧ B_foot_of_perpendicular xA yA xB yB ∧ E_on_x_axis xE yE ∧ 
    area_of_ABEF xA yA xB yB xE yE = 6 * sqrt 3 :=
by
  sorry

end area_quadrilateral_ABEF_l134_134481


namespace determine_m_l134_134271

def f (x m : ℝ) : ℝ := x^2 - 3*x + m
def g (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem determine_m (m : ℝ) : 3 * f 5 m = 2 * g 5 m → m = 10 / 7 := 
by sorry

end determine_m_l134_134271


namespace company_fund_initial_amount_l134_134110

-- Let n be the number of employees in the company.
variable (n : ℕ)

-- Conditions from the problem.
def initial_fund := 60 * n - 10
def adjusted_fund := 50 * n + 150
def employees_count := 16

-- Given the conditions, prove that the initial fund amount was $950.
theorem company_fund_initial_amount
    (h1 : adjusted_fund n = initial_fund n)
    (h2 : n = employees_count) : 
    initial_fund n = 950 := by
  sorry

end company_fund_initial_amount_l134_134110


namespace KQ_parallel_AD_l134_134997

structure Trapezoid where
  A B C D K L M N Q : Type
  BC_parallel_AD : A → B → C → D → Prop
  CircleInscribed : A → B → C → D → K → L → M → N → Prop
  Q_intersection : A → B → C → D → K → L → M → N → Q → Prop

theorem KQ_parallel_AD (A B C D K L M N Q : Type) 
  [BC_parallel_AD : A → B → C → D → Prop] 
  [CircleInscribed : A → B → C → D → K → L → M → N → Prop]
  [Q_intersection : A → B → C → D → K → L → M → N → Q → Prop] :
  (Q_intersection A B C D K L M N Q A B C D K L M N) →
  (CircleInscribed A B C D K L M N) →
  (BC_parallel_AD A B C D) →
  Q K A D :=
sorry

end KQ_parallel_AD_l134_134997


namespace charlie_steps_in_running_session_l134_134243

variables (m_steps_3km : ℕ) (times_field : ℕ)
variables (distance_1_field : ℕ) (steps_per_km : ℕ)

-- Conditions
def charlie_steps : ℕ := 5350
def field_distance : ℕ := 3
def run_times : ℚ := 2.5

-- Statement we need to prove
theorem charlie_steps_in_running_session : 
  let distance_ran := run_times * field_distance in
  let total_steps := (charlie_steps * distance_ran) / field_distance in
  total_steps = 13375 := 
by simp [charlie_steps, field_distance, run_times]; sorry

end charlie_steps_in_running_session_l134_134243


namespace females_with_advanced_degrees_l134_134561

noncomputable def total_employees := 200
noncomputable def total_females := 120
noncomputable def total_advanced_degrees := 100
noncomputable def males_college_degree_only := 40

theorem females_with_advanced_degrees :
  (total_employees - total_females) - males_college_degree_only = 
  total_employees - total_females - males_college_degree_only ∧ 
  total_females = 120 ∧ 
  total_advanced_degrees = 100 ∧ 
  total_employees = 200 ∧ 
  males_college_degree_only = 40 ∧
  total_advanced_degrees - (total_employees - total_females - males_college_degree_only) = 60 :=
sorry

end females_with_advanced_degrees_l134_134561


namespace largest_number_of_digits_l134_134951

def is_digit (n : ℕ) : Prop := n = 1 ∨ n = 2 ∨ n = 3

def valid_number (digits : List ℕ) : Prop :=
  digits.all is_digit ∧ digits.sum = 14

noncomputable def list_to_nat (digits : List ℕ) : ℕ :=
  digits.foldl (λ n d => n * 10 + d) 0

theorem largest_number_of_digits (digits : List ℕ) (h : valid_number digits) :
  list_to_nat digits ≤ list_to_nat [3, 3, 3, 2] := sorry

end largest_number_of_digits_l134_134951


namespace find_g_3_l134_134799

def g (x : ℝ) : ℝ := 3 * x^6 - 2 * x^4 + 5 * x^2 - 7

theorem find_g_3 (h : g (-3) = 9) : g 3 = 9 :=
by
  have h_even : ∀ x, g x = g (-x),
    simp [g],
  rw [h_even 3],
  exact h

end find_g_3_l134_134799


namespace charlie_steps_in_running_session_l134_134245

variables (m_steps_3km : ℕ) (times_field : ℕ)
variables (distance_1_field : ℕ) (steps_per_km : ℕ)

-- Conditions
def charlie_steps : ℕ := 5350
def field_distance : ℕ := 3
def run_times : ℚ := 2.5

-- Statement we need to prove
theorem charlie_steps_in_running_session : 
  let distance_ran := run_times * field_distance in
  let total_steps := (charlie_steps * distance_ran) / field_distance in
  total_steps = 13375 := 
by simp [charlie_steps, field_distance, run_times]; sorry

end charlie_steps_in_running_session_l134_134245


namespace ellipse_hyperbola_equation_l134_134367

-- Definitions for the Ellipse and Hyperbola
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2) / 10 + (y^2) / m = 1
def hyperbola (x y : ℝ) (b : ℝ) : Prop := (x^2) - (y^2) / b = 1

-- Conditions
def same_foci (c1 c2 : ℝ) : Prop := c1 = c2
def intersection_at_p (x y : ℝ) : Prop := x = (Real.sqrt 10) / 3 ∧ (ellipse x y 1 ∧ hyperbola x y 8)

-- Theorem stating the mathematically equivalent proof problem
theorem ellipse_hyperbola_equation :
  ∀ (m b : ℝ) (x y : ℝ), ellipse x y m ∧ hyperbola x y b ∧ same_foci (Real.sqrt (10 - m)) (Real.sqrt (1 + b)) ∧ intersection_at_p x y
  → (m = 1) ∧ (b = 8) := 
by
  intros m b x y h
  sorry

end ellipse_hyperbola_equation_l134_134367


namespace y_coordinates_difference_l134_134398

theorem y_coordinates_difference {m n k : ℤ}
  (h1 : m = 2 * n + 5)
  (h2 : m + 4 = 2 * (n + k) + 5) :
  k = 2 :=
by
  sorry

end y_coordinates_difference_l134_134398


namespace walnut_trees_in_park_l134_134127

def num_current_walnut_trees (num_plant : ℕ) (num_total : ℕ) : ℕ :=
  num_total - num_plant

theorem walnut_trees_in_park :
  num_current_walnut_trees 6 10 = 4 :=
by
  -- By the definition of num_current_walnut_trees
  -- We have 10 (total) - 6 (to be planted) = 4 (current)
  sorry

end walnut_trees_in_park_l134_134127


namespace smallest_two_digit_multiple_of_17_l134_134961

theorem smallest_two_digit_multiple_of_17 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ n % 17 = 0 ∧ ∀ m, (10 ≤ m ∧ m < n ∧ m % 17 = 0) → false := sorry

end smallest_two_digit_multiple_of_17_l134_134961


namespace no_closed_polygonal_line_l134_134893

theorem no_closed_polygonal_line (m n : ℕ) : 
  ¬ (∃ (p : List (ℤ × ℤ)), (∀ (i j : ℕ), i < j → j < List.length p → (List.nodup p) ∧ (p.nth i ≠ p.nth j)) ∧(∀ (v : (ℤ × ℤ)), v ∈ p → (1 ≤ v.1 ∧ v.1 ≤ m) ∧ (1 ≤ v.2 ∧ v.2 ≤ n))) := 
sorry

end no_closed_polygonal_line_l134_134893


namespace altitude_equation_median_equation_l134_134347

-- Given points A, B, and C
def A := (0, 1)
def B := (-2, 0)
def C := (2, 0)

-- Problem 1: Equation of the altitude from A to AC
theorem altitude_equation :
  ∃ m b, (∀ x y, y = m * x + b ↔ y - 1 = 2 * x) :=
by 
  sorry

-- Problem 2: Equation of the median from A to BC
theorem median_equation :
  ∃ x y, x = 0 :=
by 
  sorry

end altitude_equation_median_equation_l134_134347


namespace hyperbola_condition_l134_134477

theorem hyperbola_condition (m : ℝ) : ((m - 2) * (m + 3) < 0) ↔ (-3 < m ∧ m < 0) := by
  sorry

end hyperbola_condition_l134_134477


namespace recurring_fraction_division_l134_134279

-- Define the values
def x : ℚ := 8 / 11
def y : ℚ := 20 / 11

-- The theorem statement function to prove x / y = 2 / 5
theorem recurring_fraction_division :
  (x / y = (2 : ℚ) / 5) :=
by 
  -- Skip the proof
  sorry

end recurring_fraction_division_l134_134279


namespace total_wheels_in_garage_l134_134926

theorem total_wheels_in_garage : 
  let bicycles := 3
  let tricycles := 4
  let unicycles := 7
  let wheels_per_bicycle := 2
  let wheels_per_tricycle := 3
  let wheels_per_unicycle := 1
  in (bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle) = 25 :=
by
  let bicycles := 3
  let tricycles := 4
  let unicycles := 7
  let wheels_per_bicycle := 2
  let wheels_per_tricycle := 3
  let wheels_per_unicycle := 1
  have h_bicycles := bicycles * wheels_per_bicycle
  have h_tricycles := tricycles * wheels_per_tricycle
  have h_unicycles := unicycles * wheels_per_unicycle
  show (h_bicycles + h_tricycles + h_unicycles) = 25
  sorry

end total_wheels_in_garage_l134_134926


namespace _l134_134447

-- Define a graph structure with vertices representing users and edges representing friendships
structure Graph (V : Type) :=
  (adj : V → V → Prop)
  (symm : symmetric adj)
  (no_self_loops : ∀ v, ¬ adj v v)

-- Define a connected graph with the constraints given in the problem
structure ConnectedGraph (V : Type) extends Graph V :=
  (connected : ∀ u v : V, u ≠ v → ∃ (p : List V), adj_path adj u v p)

-- Define a graph where each node has at most 10 neighbors
structure BoundedDegreeGraph (V : Type) extends ConnectedGraph V :=
  (degree_le : ∀ v, (finset.filter (λ u => adj v u) finset.univ).card ≤ 10)

-- Define the partitioning properties
structure PartitionProperties (V : Type) (G : ConnectedGraph V) :=
  (partition : V → nat)
  (partition_size_constraints : ∀ (i : nat), (let group := G.V.filter (λ v => partition v = i) in 
      if i = 0 then 1 ≤ group.card ∧ group.card ≤ 100 
      else 100 ≤ group.card ∧ group.card ≤ 900))
  (group_connected : ∀ (i : nat), is_connected_subgraph partition i G.adj)

-- Define our main theorem
noncomputable def graph_partitioning_theorem (V : Type) [h : Fintype V] (G : BoundedDegreeGraph V) :
  ∃ p : PartitionProperties V G.toConnectedGraph, true :=
by sorry

end _l134_134447


namespace arrangement_ways_l134_134598

def num_ways_arrange_boys_girls : Nat :=
  let boys := 2
  let girls := 3
  let ways_girls := Nat.factorial girls
  let ways_boys := Nat.factorial boys
  ways_girls * ways_boys

theorem arrangement_ways : num_ways_arrange_boys_girls = 12 :=
  by
    sorry

end arrangement_ways_l134_134598


namespace area_ratio_independent_l134_134790

theorem area_ratio_independent (A B C P K L : Point)
  (h₀ : right_angle_triangle A B C)
  (h₁ : on_circle P (circumcircle A B C))
  (h₂ : P ≠ A ∧ P ≠ C)
  (h₃ : perpend_to CP K)
  (h₄ : perpend_to CP L)
  (h₅ : intersect K A P)
  (h₆ : intersect L B P) :
  ratio_areas B K L A C P = (dist A B)^2 / (dist A C)^2 := 
sorry

end area_ratio_independent_l134_134790


namespace solution1_problem2_l134_134570

noncomputable def problem1 : ℝ := (25 / 9) ^ (1 / 2) + 1 + (64 / 27) ^ (1 / 3)

theorem solution1 : problem1 = 4 := 
by
  trace_state sorry

theorem problem2 (x : ℝ) (h : Real.logBase 3 (6^x - 9) = 3) : x = 2 :=
by
  trace_state sorry

end solution1_problem2_l134_134570


namespace max_ab_real_positive_l134_134677

theorem max_ab_real_positive (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 2) : 
  ab ≤ 1 :=
sorry

end max_ab_real_positive_l134_134677


namespace points_in_triangle_l134_134947

theorem points_in_triangle (n : ℕ) (pts: fin n → E) 
  (h : ∀ (i j k : fin n), i ≠ j ∧ j ≠ k ∧ k ≠ i → 
         (triangle_area (pts i) (pts j) (pts k) ≤ 1)) :
  ∃ (A B C : E), 
    (∀ (P : E), ∃ (a b c : E), (a + b + c) = P ∧ a * B + b * C + c * A = 0) ∧
    (triangle_area A B C ≤ 4) :=
sorry

end points_in_triangle_l134_134947


namespace acme_profit_l134_134602

-- Define the given problem conditions
def initial_outlay : ℝ := 12450
def cost_per_set : ℝ := 20.75
def selling_price_per_set : ℝ := 50
def num_sets : ℝ := 950

-- Define the total revenue and total manufacturing costs
def total_revenue : ℝ := num_sets * selling_price_per_set
def total_cost : ℝ := initial_outlay + (cost_per_set * num_sets)

-- State the profit calculation and the expected result
def profit : ℝ := total_revenue - total_cost

theorem acme_profit : profit = 15337.50 := by
  -- Proof goes here
  sorry

end acme_profit_l134_134602


namespace third_grade_parts_in_batch_l134_134199

-- Define conditions
variable (x y s : ℕ) (h_first_grade : 24 = 24) (h_second_grade : 36 = 36)
variable (h_sample_size : 20 = 20) (h_sample_third_grade : 10 = 10)

-- The problem: Prove the total number of third-grade parts in the batch is 60 and the number of second-grade parts sampled is 6
open Nat

theorem third_grade_parts_in_batch
  (h_total_parts : x - y = 60)
  (h_third_grade_proportion : y = (1 / 2) * x)
  (h_second_grade_proportion : s = (36 / 120) * 20) :
  y = 60 ∧ s = 6 := by
  sorry

end third_grade_parts_in_batch_l134_134199


namespace solve_for_a_and_b_l134_134308

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x

theorem solve_for_a_and_b {a b : ℝ}
  (h1 : f 1 a b = -2)
  (h2 : deriv (λ x, f x a b) 1 = 0) :
  a + 2 * b = -6 :=
by
  sorry

end solve_for_a_and_b_l134_134308


namespace coordinates_2008_l134_134597

def f(k : ℕ) : ℕ := (k - 1) / 5 - (k - 2) / 5

noncomputable def x_k : ℕ → ℕ
| 1       := 1
| (k + 1) := x_k k + 1 - 5 * f (k + 1)

noncomputable def y_k : ℕ → ℕ
| 1       := 1
| (k + 1) := y_k k + f (k + 1)

theorem coordinates_2008 : (x_k 2008, y_k 2008) = (3, 402) :=
by sorry

end coordinates_2008_l134_134597


namespace a_neg1_sufficient_not_necessary_condition_l134_134798

-- Definitions for lines l1 and l2
def line1 (a : ℝ) : ℝ → Prop := λ x y, a * x + y = 1
def line2 (a : ℝ) : ℝ → Prop := λ x y, x + a * y = 2 * a

-- The theorem we need to prove
theorem a_neg1_sufficient_not_necessary_condition (a : ℝ) :
  (∀ x y, line1 a x y ∧ line2 a x y → a = -1 ∨ a = 1) →
  (∃ x y, line1 a x y ∧ line2 a x y ∧ a ≠ -1) →
  (l1_parallel_l2 : ∀ x y, ∀ (h1 : line1 a x y) (h2 : line2 a x y), a = 1) ∧ (¬∀ x y, line1 a x y ∧ line2 a x y → a = -1) :=
sorry

end a_neg1_sufficient_not_necessary_condition_l134_134798


namespace distinct_scores_l134_134386

structure Player where
  id : ℕ

def RoundRobin (players : List Player) : Type := 
  players.all (λ p1, players.all (λ p2, p1 ≠ p2 → (p1.id, p2.id)))
  
noncomputable def points (p1 p2 : Player) : ℝ := 
  if p1.id = p2.id then 0 else if p1.id < p2.id then 1 else 0.5

def final_score (p : Player) (players : List Player) : ℝ := 
  players.foldl (λ acc p2 => acc + points p p2) 0

def property_P (m : ℕ) (players : List Player) : Prop := 
  ∀ (s : Finset Player), s.card = m → 
    (∃ w l : Player, w ∈ s ∧ l ∈ s ∧ 
                     (∀ p : Player, (p ∈ s ∧ p ≠ w → points w p = 1) ∧ 
                                    (p ∈ s ∧ p ≠ l → points l p = 0)))

theorem distinct_scores {m : ℕ} (h : m ≥ 4) (players : List Player) 
  (rr : RoundRobin players) (p_property : property_P m players)
  (n : ℕ) (hn : n = 2*m-3) : 
  ∀ (p1 p2 : Player), p1 ∈ players → p2 ∈ players → p1 ≠ p2 → 
  final_score p1 players ≠ final_score p2 players := 
by
  sorry

end distinct_scores_l134_134386


namespace squirrels_more_than_nuts_l134_134507

theorem squirrels_more_than_nuts (squirrels nuts : ℕ) (h1 : squirrels = 4) (h2 : nuts = 2) : squirrels - nuts = 2 := by
  sorry

end squirrels_more_than_nuts_l134_134507


namespace angle_x_degrees_l134_134048

noncomputable def center_of_circle (O : Point) := true
noncomputable def radii (O B C A : Point) := (dist O B = dist O C) ∧ (dist O A = dist O C)
noncomputable def isosceles_triangle (O B C : Point) := (∠ B C O = 32) ∧ (∠ O B C = 32)
noncomputable def isosceles_base_angle (O A C : Point) := ∠ O A C = ∠ A O C = 23
noncomputable def angle_sum_property (O A C : Point) := ∠ A O C = 90

theorem angle_x_degrees
  (O B C A : Point) (h_radii : radii O B C A)
  (h_isosceles_triangle : isosceles_triangle O B C)
  (h_isosceles_base : isosceles_base_angle O A C)
  (h_sum_property : angle_sum_property O A C) :
  ∠ B C O - ∠ O A C = 9 := by sorry

end angle_x_degrees_l134_134048


namespace find_value_of_x_l134_134008

-- Definitions used in conditions
variable (O B C A : Point) -- points on the circle
variable (OB OC OA : ℝ) -- Radii of the circle
variable (OB_eq_OC : OB = OC)
variable (angle_BCO : ℝ) -- Given angle ∠BCO
variable (angle_DAC : ℝ) -- Given angle ∠DAC
variable (right_angled_ACD : angle_DAC + 90 = 157) -- Triangle ACD is right-angled & 180 - 23 = 157

variable (x : ℝ) -- angle x

-- Setting specific values for given angles in degrees
noncomputable def angle_BCO_value : angle_BCO = 32
noncomputable def angle_DAC_value : angle_DAC = 67

-- What we want to prove
theorem find_value_of_x : x = 9 :=
sorry

end find_value_of_x_l134_134008


namespace median_free_throws_is_16_l134_134992

def free_throws : List ℕ := [20, 12, 17, 25, 8, 34, 15, 23, 12, 10]

def median (l : List ℕ) : ℕ :=
  let sorted := List.sort l
  let n := List.length sorted
  if n % 2 = 0 then
    (sorted.get! (n / 2 - 1) + sorted.get! (n / 2)) / 2
  else
    sorted.get! (n / 2)

theorem median_free_throws_is_16 : median free_throws = 16 := by
  sorry

end median_free_throws_is_16_l134_134992


namespace find_a6_l134_134346

theorem find_a6 (a : ℕ → ℚ) (h₁ : ∀ n, a (n + 1) = 2 * a n - 1) (h₂ : a 8 = 16) : a 6 = 19 / 4 :=
sorry

end find_a6_l134_134346


namespace place_face_value_difference_l134_134564

theorem place_face_value_difference : 
    let n := 856973
    let d := 7
    let tens_place := 10
    place_value n 7 tens_place - face_value d = 63 :=
by
    -- Definitions
    def place_value (n : ℕ) (d : ℕ) (tens_place : ℕ) : ℕ :=
        if n / tens_place % 10 = d then d * tens_place else 0
    
    def face_value (d : ℕ) : ℕ :=
        d
    
    -- The Proof needs to be filled in here
    sorry

end place_face_value_difference_l134_134564


namespace simplify_fraction_l134_134079

theorem simplify_fraction (a b c : ℕ) (h1 : a = 222) (h2 : b = 8888) (h3 : c = 44) : 
  (a : ℚ) / b * c = 111 / 101 := 
by 
  sorry

end simplify_fraction_l134_134079


namespace smallest_k_divides_polynomial_l134_134269

theorem smallest_k_divides_polynomial :
  ∃ (k : ℕ), k > 0 ∧ (∀ z : ℂ, z ≠ 0 → 
    (z ^ 11 + z ^ 9 + z ^ 7 + z ^ 6 + z ^ 5 + z ^ 2 + 1) ∣ (z ^ k - 1)) ∧ k = 11 := by
  sorry

end smallest_k_divides_polynomial_l134_134269


namespace probability_of_one_failure_l134_134513

theorem probability_of_one_failure (p1 p2 : ℝ) (h1 : p1 = 0.90) (h2 : p2 = 0.95) :
  (p1 * (1 - p2) + (1 - p1) * p2) = 0.14 :=
by
  rw [h1, h2]
  -- Additional leaning code can be inserted here to finalize the proof if this was complete
  sorry

end probability_of_one_failure_l134_134513


namespace distinct_names_impossible_l134_134777

-- Define the alphabet
inductive Letter
| a | u | o | e

-- Simplified form of words in the Mumbo-Jumbo language
def simplified_form : List Letter → List Letter
| [] => []
| (Letter.e :: xs) => simplified_form xs
| (Letter.a :: Letter.a :: Letter.a :: Letter.a :: xs) => simplified_form (Letter.a :: Letter.a :: xs)
| (Letter.o :: Letter.o :: Letter.o :: Letter.o :: xs) => simplified_form xs
| (Letter.a :: Letter.a :: Letter.a :: Letter.u :: xs) => simplified_form (Letter.u :: xs)
| (x :: xs) => x :: simplified_form xs

-- Number of possible names
def num_possible_names : ℕ := 343

-- Number of tribe members
def num_tribe_members : ℕ := 400

theorem distinct_names_impossible : num_possible_names < num_tribe_members :=
by
  -- Skipping the proof with 'sorry'
  sorry

end distinct_names_impossible_l134_134777


namespace christmas_tree_bulbs_l134_134912

def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

def number_of_bulbs_on (N : Nat) : Nat :=
  (Finset.range (N+1)).filter isPerfectSquare |>.card

def probability_bulb_on (total_bulbs bulbs_on : Nat) : Float :=
  (bulbs_on.toFloat) / (total_bulbs.toFloat)

theorem christmas_tree_bulbs :
  let N := 100
  let bulbs_on := number_of_bulbs_on N
  probability_bulb_on N bulbs_on = 0.1 :=
by
  sorry

end christmas_tree_bulbs_l134_134912


namespace find_point_P_l134_134496

theorem find_point_P :
  ∃ P : ℝ × ℝ × ℝ, 
    let (a, b, c) := P in 
    (a = 9 ∧ b = 1 ∧ c = -13) ∧
    (∀ x y z : ℝ, (12 * x + 6 * y - 18 * z = 36) → 
      ((x - 3)^2 + (y + 2)^2 + (z - 4)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2)) :=
by
  sorry

end find_point_P_l134_134496


namespace count_distinct_triangles_with_integer_area_l134_134257

def point := (ℕ × ℕ)

def valid_point (p : point) : Prop :=
  31 * p.fst + p.snd = 1001

def distinct_points (p q : point) : Prop :=
  p ≠ q

def triangle_area_integer (p q : point) : Prop :=
  ∃ (k : ℤ), k > 0 ∧ 
  |1001 * (p.fst - q.fst)| = 2 * k

theorem count_distinct_triangles_with_integer_area :
  ∃ (n : ℕ), 
    n = 256 ∧
    ∀ (p q : point),
      p ≠ q →
      valid_point p →
      valid_point q →
      ∃ (m : ℕ), distinct_points p q ∧ triangle_area_integer p q → m = n :=
sorry

end count_distinct_triangles_with_integer_area_l134_134257


namespace length_of_BC_l134_134381

-- Define the setup for the problem
variables {O A B C D : Point}
variables {circle : Circle O}
variables {chord: Chord circle A B C}

-- Define the constraints
def is_diameter (D : Point) (AD : Diameter circle A D) : Prop :=
  true

def BO_length (B O : Point) : Real :=
  7

def angle_ABO (A B O : Point) : Real :=
  45

def arc_CD (C D : Point) : Real :=
  90

-- The theorem stating the length of BC
theorem length_of_BC 
  (circle_center : Circle O)
  (diam_AD : is_diameter D)
  (chord_ABC : Chord circle A B C)
  (BO_eq_seven : BO_length B O)
  (angle_BAO_forty_five : angle_ABO A B O)
  (arc_CD_ninety : arc_CD C D) :
  length B C = 7 :=
sorry

end length_of_BC_l134_134381


namespace right_triangle_leg_ratio_l134_134385

theorem right_triangle_leg_ratio (a c : ℕ) (h₁ : a = 5) (h₂ : c = 13) : ∃ b : ℕ, b^2 + a^2 = c^2 ∧ b.gcd c = 1 ∧ (b, c) = (12, 13) :=
by {
  use 12,
  split,
  { calc 12^2 + 5^2 = 144 + 25 : by norm_num
                    ... = 169 : by norm_num
                    ... = 13^2 : by norm_num },
  split,
  { exact nat.gcd_comm 12 13 },
  { trivial },
}

end right_triangle_leg_ratio_l134_134385


namespace projections_concyclic_l134_134414

open EuclideanGeometry

variables {A B C D A' C' B' D' : Point}
variables {BD AC : Line}

-- Given conditions
variable (h1 : OnCircle A B C D)
variable (h2 : Proj A BD A')
variable (h3 : Proj C BD C')
variable (h4 : Proj B AC B')
variable (h5 : Proj D AC D')

-- Proof goal
theorem projections_concyclic :
  Concyclic A' B' C' D' := sorry

end projections_concyclic_l134_134414


namespace smallest_prime_factor_3087_l134_134158

open Nat

theorem smallest_prime_factor_3087 : (∃ p : ℕ, prime p ∧ p ∣ 3087 ∧ ∀ q : ℕ, prime q ∧ q ∣ 3087 → p ≤ q) ∧ (∃ p : ℕ, prime p ∧ p ∣ 3087 ∧ p = 3) := 
by
  sorry

end smallest_prime_factor_3087_l134_134158


namespace bananas_each_child_l134_134826

theorem bananas_each_child (x : ℕ) (B : ℕ) 
  (h1 : 660 * x = B)
  (h2 : 330 * (x + 2) = B) : 
  x = 2 := 
by 
  sorry

end bananas_each_child_l134_134826


namespace kevin_wings_record_l134_134788

-- Conditions
def alanWingsPerMinute : ℕ := 5
def additionalWingsNeeded : ℕ := 4
def kevinRecordDuration : ℕ := 8

-- Question and answer
theorem kevin_wings_record : 
  (alanWingsPerMinute + additionalWingsNeeded) * kevinRecordDuration = 72 :=
by
  sorry

end kevin_wings_record_l134_134788


namespace angle_SAB_eq_arccos_length_CQ_l134_134002

-- Definitions and necessary conditions
variables {A B C S K L M N : Type} [regular_triangle_pyramid S A B C]
variables [plane_contains_points K L M N]
variables (KL MN KN LM : ℕ) [eq KL 2] [eq MN 2] [eq KN 9] [eq LM 9]

-- Statements to be proved
theorem angle_SAB_eq_arccos : (∠ (S, A, B)) = real.arccos (2 / 3) :=
by sorry

theorem length_CQ : (length (C, Q)) = 7 / 3 :=
by sorry

end angle_SAB_eq_arccos_length_CQ_l134_134002


namespace blue_balls_needed_l134_134000

-- Conditions
variables (R Y B W : ℝ)
axiom h1 : 2 * R = 5 * B
axiom h2 : 3 * Y = 7 * B
axiom h3 : 9 * B = 6 * W

-- Proof Problem
theorem blue_balls_needed : (3 * R + 4 * Y + 3 * W) = (64 / 3) * B := by
  sorry

end blue_balls_needed_l134_134000


namespace no_C_makes_2C7_even_and_multiple_of_5_l134_134264

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

theorem no_C_makes_2C7_even_and_multiple_of_5 : ∀ C : ℕ, ¬(C < 10) ∨ ¬(is_even (2 * 100 + C * 10 + 7) ∧ is_multiple_of_5 (2 * 100 + C * 10 + 7)) :=
by
  intro C
  sorry

end no_C_makes_2C7_even_and_multiple_of_5_l134_134264


namespace exactly_three_implies_l134_134258

def statement_1 (r s : Prop) : Prop := ¬r ∧ ¬s
def statement_2 (r s : Prop) : Prop := r ∧ ¬s
def statement_3 (r s : Prop) : Prop := ¬r ∧ s
def statement_4 (r s : Prop) : Prop := r ∧ s

def implies_r_or_s (p : Prop) : Prop := p → (r ∨ s)

theorem exactly_three_implies (r s : Prop) :
  [statement_1 r s, statement_2 r s, statement_3 r s, statement_4 r s].count (implies_r_or_s r s) = 3 :=
sorry

end exactly_three_implies_l134_134258


namespace part1_condition_represents_line_part2_slope_does_not_exist_part3_x_intercept_part4_angle_condition_l134_134334

theorem part1_condition_represents_line (m : ℝ) :
  (m^2 - 2 * m - 3 ≠ 0) ∧ (2 * m^2 + m - 1 ≠ 0) ↔ m ≠ -1 :=
sorry

theorem part2_slope_does_not_exist (m : ℝ) :
  (m = 1 / 2) ↔ (m^2 - 2 * m - 3 = 0 ∧ (2 * m^2 + m - 1 = 0) ∧ ((1 * x = (4 / 3)))) :=
sorry

theorem part3_x_intercept (m : ℝ) :
  (2 * m - 6) / (m^2 - 2 * m - 3) = -3 ↔ m = -5 / 3 :=
sorry

theorem part4_angle_condition (m : ℝ) :
  -((m^2 - 2 * m - 3) / (2 * m^2 + m - 1)) = 1 ↔ m = 4 / 3 :=
sorry

end part1_condition_represents_line_part2_slope_does_not_exist_part3_x_intercept_part4_angle_condition_l134_134334


namespace num_girls_on_playground_l134_134917

-- Definitions based on conditions
def total_students : ℕ := 20
def classroom_students := total_students / 4
def playground_students := total_students - classroom_students
def boys_playground := playground_students / 3
def girls_playground := playground_students - boys_playground

-- Theorem statement
theorem num_girls_on_playground : girls_playground = 10 :=
by
  -- Begin preparing proofs
  sorry

end num_girls_on_playground_l134_134917


namespace complex_number_count_l134_134351

theorem complex_number_count (z : ℂ) (h₁ : |z| < 25) (h₂ : complex.exp z = (z - 2) / (z + 2)) :
    (∃! z : ℂ, |z| < 25 ∧ complex.exp z = (z - 2) / (z + 2)) → 17 :=
by 
  sorry

end complex_number_count_l134_134351


namespace smallest_positive_period_monotonically_increasing_interval_max_min_values_within_interval_l134_134717

noncomputable def f (x : ℝ) : ℝ :=
  sin (2 * x + π / 3) - sqrt 3 * sin (2 * x - π / 6)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
sorry

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x ∈ set.Icc (-(7 * π / 12) + k * π) (-(π / 12) + k * π), ∀ y ∈ set.Icc (-(7 * π / 12) + k * π) (-(π / 12) + k * π), x < y → f x < f y :=
sorry

theorem max_min_values_within_interval :
  ∃ (x_max x_min ∈ set.Icc (-π / 6) (π / 3)), f x_max = 2 ∧ f x_min = -sqrt 3 ∧ x_max = -π / 12 ∧ x_min = π / 3 :=
sorry

end smallest_positive_period_monotonically_increasing_interval_max_min_values_within_interval_l134_134717


namespace k_value_t_range_m_value_l134_134815

variable (a : ℝ) (k : ℝ) (m : ℝ) (t : ℝ)

-- Condition: a > 0 and a ≠ 1
variable (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1)

-- 1. Prove that if f(x) is an odd function, then k = 0
def f (x : ℝ) : ℝ := a^x + (k-1)*a^(-x) + k^2

theorem k_value (h_odd : ∀ x : ℝ, f a k x = -f a k (-x)) : k = 0 :=
sorry

-- 2. Prove that the range of t for which the inequality f(x^2+x) + f(t-2x) > 0 always holds is (1/4, +∞)
def f2 (x : ℝ) : ℝ := a^(x^2 + x) - a^(-(x^2 + x))

theorem t_range (h_gt_zero : f a 0 1 > 0) : ∀ x : ℝ, f a 0 (x^2 + x) + f a 0 (t - 2 * x) > 0 ↔ t > 1 / 4 :=
sorry

-- 3. Prove that if f(1) = 3/2, then the minimum value of the function g(x) is -1 implies m = sqrt(3)
def g (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2*m*(a^x - a^(-x))

theorem m_value (h_f_eq : f a 0 1 = 3/2) (h_min : ∀ x : ℝ, 1 ≤ x → g a 2 m x ≥ -1) : m = Real.sqrt 3 :=
sorry

end k_value_t_range_m_value_l134_134815


namespace solve_for_y_l134_134852

theorem solve_for_y (y : ℕ) (h : 5 * (2 ^ y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l134_134852


namespace part_a_part_b_l134_134806

-- Part (a)
theorem part_a (a b : ℕ) (coprime_ab : coprime a b) (c : ℕ) :
  ∃! (x y : ℕ), x ≤ b - 1 ∧ N a b x y = c :=
sorry

-- Part (b)
theorem part_b (a b : ℕ) (coprime_ab : coprime a b) :
  ∀ c : ℕ, (¬ ∃ (x y : ℕ), a * x + b * y = c) ↔ c = a * b - a - b :=
sorry

end part_a_part_b_l134_134806


namespace ellipse_equation_and_max_AB_l134_134690

-- Conditions
def ellipse (a b : ℝ) : set (ℝ × ℝ) := { p | let ⟨x, y⟩ := p in (x^2 / a^2) + (y^2 / b^2) = 1 }
def pointP : ℝ × ℝ := (-Real.sqrt 3, 1 / 2)
def focus : ℝ × ℝ := (Real.sqrt 3, 0)
def pointM : ℝ × ℝ := (0, Real.sqrt 2)

-- Derived conditions
def ellipse_condition (a b : ℝ) (E : set (ℝ × ℝ)) : Prop :=
  a > b ∧ b > 0 ∧ pointP ∈ E ∧ (focus.1 * focus.1 + focus.2 * focus.2 = a * a - b * b)

-- Proof problem statement
theorem ellipse_equation_and_max_AB :
  ∃ a b : ℝ, ∃ E : set (ℝ × ℝ), ellipse_condition a b E ∧ 
  (E = ellipse 2 1 ∧ 
   ∃ l : ℝ → ℝ, ∀ A B : ℝ × ℝ, (A ∈ E ∧ B ∈ E ∧ collinear ((0, sqrt 2) :: A :: B :: []) → dist A B ≤ 5 * sqrt 6 / 6)) :=
sorry

end ellipse_equation_and_max_AB_l134_134690


namespace juice_difference_proof_l134_134499

def barrel_initial_A := 10
def barrel_initial_B := 8
def transfer_amount := 3

def barrel_final_A := barrel_initial_A + transfer_amount
def barrel_final_B := barrel_initial_B - transfer_amount

def juice_difference := barrel_final_A - barrel_final_B

theorem juice_difference_proof : juice_difference = 8 := by
  sorry

end juice_difference_proof_l134_134499


namespace equal_real_roots_implies_m_l134_134749

theorem equal_real_roots_implies_m (m : ℝ) : (∃ (x : ℝ), x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) → m = 1/4 :=
by
  sorry

end equal_real_roots_implies_m_l134_134749


namespace length_increase_percentage_l134_134108

noncomputable def length_of_floor : ℚ := 18.9999683334125
noncomputable def breadth_of_floor := 120.332 / 18.9999683334125
noncomputable def area_of_floor := 361 / 3.00001
noncomputable def percentage_increase := ((length_of_floor - breadth_of_floor) / breadth_of_floor) * 100

theorem length_increase_percentage :
  (breadth_of_floor * length_of_floor ≈ area_of_floor) →
  percentage_increase ≈ 200 := by
  sorry

end length_increase_percentage_l134_134108


namespace rate_of_discount_l134_134991

theorem rate_of_discount (marked_price selling_price : ℝ) (h1 : marked_price = 200) (h2 : selling_price = 120) : 
  ((marked_price - selling_price) / marked_price) * 100 = 40 :=
by
  sorry

end rate_of_discount_l134_134991


namespace part1_part2_l134_134606

-- Definition for the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Points P and Q
def P : ℝ × ℝ := (Real.sqrt 2, 1)
def Q : ℝ × ℝ := (0, -Real.sqrt 2)

-- Midpoint M
def M : ℝ × ℝ := (-2 / 3, 1 / 3)

-- Definition for line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Problem 1: Prove points P and Q lie on the given ellipse
theorem part1 : ellipse (P.1) (P.2) ∧ ellipse (Q.1) (Q.2) :=
by sorry

-- Problem 2: Prove the line through points A and B has the given equation
theorem part2 : 
  ∀ (A B : ℝ × ℝ), 
  ellipse (A.1) (A.2) → ellipse (B.1) (B.2) →
  ((A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) →
  ∃ (k : ℝ), ∀ (x y : ℝ), 
  ((y - A.2) = k * (x - A.1)) →
  x - y + 1 = 0 :=
by sorry

end part1_part2_l134_134606


namespace solve_for_y_l134_134854

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l134_134854


namespace rectangular_field_length_l134_134882

theorem rectangular_field_length {w l : ℝ} (h1 : l = 2 * w) (h2 : (8 : ℝ) * 8 = 1 / 18 * (l * w)) : l = 48 :=
by sorry

end rectangular_field_length_l134_134882


namespace tangent_and_normal_lines_l134_134982

theorem tangent_and_normal_lines (x y : ℝ → ℝ) (t : ℝ) (t₀ : ℝ) 
  (h0 : t₀ = 0) 
  (h1 : ∀ t, x t = (1/2) * t^2 - (1/4) * t^4) 
  (h2 : ∀ t, y t = (1/2) * t^2 + (1/3) * t^3) :
  (∃ m : ℝ, y (x t₀) = m * (x t₀) ∧ m = 1) ∧
  (∃ n : ℝ, y (x t₀) = n * (x t₀) ∧ n = -1) :=
by 
  sorry

end tangent_and_normal_lines_l134_134982


namespace first_machine_necklaces_l134_134277

-- Define the number of necklaces made by the first machine
def n1 : ℝ 

-- Define the total number of necklaces made by both machines
def total_necklaces : ℝ := 153

-- Define the number of necklaces made by the second machine
def n2 : ℝ := 2.4 * n1

-- The proof that the first machine made 45 necklaces
theorem first_machine_necklaces : n1 + n2 = total_necklaces → n1 = 45 :=
by
  intro h
  have h1 : n1 + 2.4 * n1 = 153 := h
  sorry

end first_machine_necklaces_l134_134277


namespace smallest_positive_c_property_l134_134666

noncomputable def smallest_c : ℝ :=
  2 / 3

theorem smallest_positive_c_property (c : ℝ) :
  (c = smallest_c) → 
  ∀ (n : ℕ), n ≥ 4 →
  ∀ (A : set ℕ), A ⊆ (set.Icc 1 n) →
  (A.card > (c * n)) →
  ∃ (f : A → {1, -1}), abs (∑ a in A, (f a) * a) ≤ 1 :=
by
  sorry

end smallest_positive_c_property_l134_134666


namespace farmer_total_land_l134_134563

noncomputable def total_land_owned_by_farmer (cleared_land_with_tomato : ℝ) (cleared_percentage : ℝ) (grape_percentage : ℝ) (potato_percentage : ℝ) : ℝ :=
  let cleared_land := cleared_percentage
  let total_clearance_with_tomato := cleared_land_with_tomato
  let unused_cleared_percentage := 1 - grape_percentage - potato_percentage
  let total_cleared_land := total_clearance_with_tomato / unused_cleared_percentage
  total_cleared_land / cleared_land

theorem farmer_total_land (cleared_land_with_tomato : ℝ) (cleared_percentage : ℝ) (grape_percentage : ℝ) (potato_percentage : ℝ) :
  (cleared_land_with_tomato = 450) →
  (cleared_percentage = 0.90) →
  (grape_percentage = 0.10) →
  (potato_percentage = 0.80) →
  total_land_owned_by_farmer cleared_land_with_tomato 90 10 80 = 1666.6667 :=
by
  intro h1 h2 h3 h4
  sorry

end farmer_total_land_l134_134563


namespace differentiable_increasing_necessary_but_not_sufficient_l134_134371

variable {f : ℝ → ℝ}

theorem differentiable_increasing_necessary_but_not_sufficient (h_diff : ∀ x : ℝ, DifferentiableAt ℝ f x) :
  (∀ x : ℝ, 0 < deriv f x) → ∀ x : ℝ, MonotoneOn f (Set.univ : Set ℝ) ∧ ¬ (∀ x : ℝ, MonotoneOn f (Set.univ : Set ℝ) → ∀ x : ℝ, 0 < deriv f x) := 
sorry

end differentiable_increasing_necessary_but_not_sufficient_l134_134371


namespace smaller_pyramid_volume_l134_134213

noncomputable def original_pyramid_base_edge : ℝ := 12 * Real.sqrt 2
noncomputable def original_pyramid_slant_edge : ℝ := 15
noncomputable def height_reduction : ℝ := 5

def original_pyramid_height : ℝ :=
  Real.sqrt (original_pyramid_slant_edge ^ 2 - (original_pyramid_base_edge / 2) ^ 2)

def new_pyramid_height : ℝ :=
  original_pyramid_height - height_reduction

def new_base_edge (original_height new_height : ℝ) : ℝ :=
  12 * Real.sqrt 2 * (new_height / original_height)

noncomputable def new_base_area (base_edge : ℝ) : ℝ :=
  (base_edge / Real.sqrt 2) ^ 2 / 2

noncomputable def new_pyramid_volume (base_area new_height : ℝ) : ℝ :=
  (1 / 3) * base_area * new_height

theorem smaller_pyramid_volume :
  new_pyramid_volume 
    (new_base_area (new_base_edge original_pyramid_height new_pyramid_height)) 
    new_pyramid_height = 
  (1 / 6) * (12 * Real.sqrt 2 * (Real.sqrt 153 - 5) / Real.sqrt 153) ^ 2 * (Real.sqrt 153 - 5) :=
begin
  sorry -- Proof goes here
end

end smaller_pyramid_volume_l134_134213


namespace find_a_50_l134_134514

noncomputable def a_n : ℕ → ℕ
| 0       => a  -- dummy placeholder
| (n + 1) => a + n * d

axiom a_pos : 0 < a
axiom d_pos : 0 < d

axiom sum_cond : 
  a + (a + 3 * d) + (a + 8 * d) + (a + 15 * d) + (a + 24 * d) + (a + 35 * d) + (a + 48 * d) + (a + 63 * d) + (a + 80 * d) + (a + 99 * d) = 1000

theorem find_a_50 : a + 49 * d = 123 :=
sorry

end find_a_50_l134_134514


namespace find_x_l134_134965

theorem find_x (x y : ℤ) (h1 : y = 3) (h2 : x + 3 * y = 10) : x = 1 :=
by
  sorry

end find_x_l134_134965


namespace find_height_of_box_l134_134593

-- Definitions for the problem conditions
def numCubes : ℕ := 24
def volumeCube : ℕ := 27
def lengthBox : ℕ := 8
def widthBox : ℕ := 9
def totalVolumeBox : ℕ := numCubes * volumeCube

-- Problem statement in Lean 4
theorem find_height_of_box : totalVolumeBox = lengthBox * widthBox * 9 :=
by sorry

end find_height_of_box_l134_134593


namespace solve_for_y_l134_134858

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 :=
by sorry

end solve_for_y_l134_134858


namespace orthogonality_condition_angle_between_vectors_l134_134348

-- Condition Definitions
variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (k : ℝ)

-- Condition: norms of vectors
def norm_a : ℝ := ∥a∥
def norm_b : ℝ := ∥b∥

-- Condition: dot product condition
def dot_product_condition : ℝ := ⟪(2 • a - b), (a + 3 • b)⟫

-- Question 1: Orthogonality condition and its result
theorem orthogonality_condition : (norm_a = 1) → (norm_b = 2) → 
  (dot_product_condition = -5) → 
  (∃ k, (⟪(a - k • b), (k • a + b)⟫ = 0) ↔ (k = (3 + real.sqrt 13) / 2 ∨ k = (3 - real.sqrt 13) / 2)) :=
by sorry

-- Question 2: Find the angle between vectors
def find_angle_between_vectors : ℝ := 
  real.arccos (⟪a, (2 • a + b)⟫ / (∥a∥ * ∥2 • a + b∥))

theorem angle_between_vectors : (norm_a = 1) → (norm_b = 2) → (dot_product_condition = -5) → 
  (find_angle_between_vectors = real.pi / 6) :=
by sorry

end orthogonality_condition_angle_between_vectors_l134_134348


namespace triangle_XYZ_proof_l134_134404

-- Define the parameters for the triangle XYZ
variables (X Y Z : ℝ)
variables (angleX := 90)
variables (length_YZ : ℝ := 20)
variables (tanZ : ℝ := 3 * (XY / length_YZ))
variables (XY := X - Y)
variables (XZ := Z - X)

noncomputable def triangle_XYZ (angleX YZ tanZ XY : ℝ) : ℝ :=
  if angleX = 90 ∧ YZ = length_YZ ∧ tanZ = 3 * (XY / YZ) then
    XY
  else 0

-- Statement to be proven
theorem triangle_XYZ_proof :
  triangle_XYZ 90 20 (3 * (XY / 20)) = (40 * (Real.sqrt 2)) / 3 :=
sorry

end triangle_XYZ_proof_l134_134404


namespace quadrilateral_area_l134_134207

-- Define the coordinates for points A, B, C, D, and E
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 18)
def C : ℝ × ℝ := (10, 0)
def D : ℝ × ℝ := (0, 6)
def E : ℝ × ℝ := (5, 3)

-- Define the lines based on the given slopes and intersections
def line1 : ℝ → ℝ := λ x, -3 * x + 18 -- Line with slope -3 passing through A and B
def line2 : ℝ → ℝ := λ x, (-3 / 5) * x + 6 -- Line passing through C and D

-- Define a function to calculate the area of quadrilateral OBEC
def area_OBEC : ℝ :=
  let area_ODB := 0.5 * B.snd * D.snd in
  let area_DEB := (C.fst - E.fst) * (B.snd - D.snd) in
  let area_EBC := 0.5 * (C.fst - E.fst) * (B.snd - E.snd) in
  area_ODB + area_DEB + area_EBC

-- The statement we need to prove
theorem quadrilateral_area : area_OBEC = 211.5 :=
  by sorry

end quadrilateral_area_l134_134207


namespace find_a_b_c_l134_134877

theorem find_a_b_c (a b c : ℝ) 
  (h_min : ∀ x, -9 * x^2 + 54 * x - 45 ≥ 36) 
  (h1 : 0 = a * (1 - 1) * (1 - 5)) 
  (h2 : 0 = a * (5 - 1) * (5 - 5)) :
  a + b + c = 36 :=
sorry

end find_a_b_c_l134_134877


namespace arctan_tan_equiv_l134_134618

theorem arctan_tan_equiv (h1 : Real.tan (Real.pi / 4 + Real.pi / 12) = 1 / Real.tan (Real.pi / 4 - Real.pi / 3))
  (h2 : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3):
  Real.arctan (Real.tan (5 * Real.pi / 12) - 2 * Real.tan (Real.pi / 6)) = 5 * Real.pi / 12 := 
sorry

end arctan_tan_equiv_l134_134618


namespace exists_fixed_point_l134_134422

section FixedPoint

variable {H : Type} (f : Set H → Set H)
variable (H : Type)

-- Define the conditions
def isIncreasing (f : Set H → Set H) : Prop :=
  ∀ ⦃X Y : Set H⦄, X ⊆ Y → f X ⊆ f Y

-- The theorem we want to prove
theorem exists_fixed_point (f : Set H → Set H) (hf : isIncreasing f) :
  ∃ A : Set H, A ⊆ H ∧ f A = A :=
sorry

end FixedPoint

end exists_fixed_point_l134_134422


namespace area_of_circle_l134_134149

def circleEquation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

theorem area_of_circle :
  (∃ (center : ℝ × ℝ) (radius : ℝ), radius = 4 ∧ ∀ (x y : ℝ), circleEquation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) →
  (16 * Real.pi) = 16 * Real.pi := 
by 
  intro h
  have := h
  sorry

end area_of_circle_l134_134149


namespace determine_p_q_r_s_l134_134413

noncomputable def pqr_is_valid 
  (A B C : ℝ) (p q r s : ℕ) : Prop := 
  ∃ (p q r s : ℕ), 
    (B > π/2 ∧ A + B + C = π) ∧ 
    (cos(A)^2 + cos(B)^2 + 2 * sin(A) * sin(B) * cos(C) = 16/7) ∧ 
    (cos(B)^2 + cos(C)^2 + 2 * sin(B) * sin(C) * cos(A) = 16/9) ∧ 
    (gcd p q + gcd q s = p + q + r + s) ∧ 
    (nat.is_squarefree r ∧ r ≠ 0) ∧
    cos(C)^2 + cos(A)^2 + 2 * sin(C) * sin(A) * cos(B) = ((p : ℚ) - q * (r : ℚ)^(1/2)) / s

theorem determine_p_q_r_s (A B C : ℝ) (p q r s : ℕ) (h : pqr_is_valid A B C p q r s) : 
  ∃ (p q r s : ℕ), p + q + r + s = sorry := 
sorry

end determine_p_q_r_s_l134_134413


namespace validQuadratic_l134_134967

-- Defining the options as given equations
def OptionA : Prop := ∀ x : ℝ, x^2 + (1 / x^2) = 0
def OptionB : Prop := ∀ x : ℝ, (x - 1) * (x + 2) = 1 → x^2 + x - 3 = 0
def OptionC : Prop := ∀ a b c : ℝ, ax^2 + bx + c = 0
def OptionD : Prop := ∀ x y : ℝ, 3 * x^2 - 2 * x * y - 5 * y^2 = 0

-- The theorem stating that OptionB is the valid quadratic equation in terms of x
theorem validQuadratic (a b c : ℝ) : 
  OptionB :=
by 
  intro x hx
  calc
    (x - 1) * (x + 2) = 1 : hx
    x^2 + x - 3 = 0 : sorry

end validQuadratic_l134_134967


namespace factor_poly_l134_134101

theorem factor_poly (a b : ℤ) (h : 3*(y^2) - y - 24 = (3*y + a)*(y + b)) : a - b = 11 :=
sorry

end factor_poly_l134_134101


namespace angle_x_degrees_l134_134039

theorem angle_x_degrees 
  (O : Point)
  (A B C D : Point)
  (hC : Circle O)
  (hOB : O ∈ Segment O B)
  (hOC : O ∈ Segment O C)
  (hBCO : ∠ B C O = 32)
  (hACD : ∠ A C D = 90)
  (hCAD : ∠ C A D = 67) :
  ∠ B C A = 9 := 
  sorry

end angle_x_degrees_l134_134039


namespace probability_difference_multiple_six_l134_134940

theorem probability_difference_multiple_six :
  ∀ (S : Finset ℕ), S.card = 12 ∧ (∀ x ∈ S, 1 ≤ x ∧ x ≤ 4012) →
  ∃ (a b ∈ S), a ≠ b ∧ (a - b) % 6 = 0 := 
by
  intros S hS
  obtain ⟨hcard, hx⟩ := hS
  have pigeonhole := pigeonhole_principle_mod_6 S hcard hx
  exact pigeonhole

-- Placeholder for the proof by Pigeonhole principle
sorry

end probability_difference_multiple_six_l134_134940


namespace angle_x_is_9_degrees_l134_134051

theorem angle_x_is_9_degrees
  (O B C A D : Type)
  (hO_center : is_center O)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (hOB_eq_OC : distance O B = distance O C)
  (hAngle_BCO : angle B C O = 32)
  (hOA_eq_OC : distance O A = distance O C)
  (hAngle_CAD : angle C A D = 67)
  : angle x = 9 :=
sorry

end angle_x_is_9_degrees_l134_134051


namespace geometric_sequence_common_ratio_l134_134621

theorem geometric_sequence_common_ratio 
  (a : ℕ+ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n : ℕ+, a n > 0) 
  (h_geometric : ∀ n : ℕ+, a (n + 1) = a 1 * q ^ n)
  (h_condition : 2 * a 1 + a 2 = a 3)
  : q = 2 := 
sorry

end geometric_sequence_common_ratio_l134_134621


namespace perimeter_pentagon_is_14_l134_134156

def lengths_and_symmetry (AB BC CD DE EA : ℕ) :=
  AB = 2 ∧ BC = 2 ∧ CD = 2 ∧ DE = 4 ∧ EA = 4

theorem perimeter_pentagon_is_14 :
  lengths_and_symmetry 2 2 2 4 4 → ∑ i in [2, 2, 2, 4, 4], i = 14 :=
by
  intro h
  rw [Finset.sum_eq_add, Finset.sum_singleton, Finset.sum_eq_add, Finset.sum_singleton, Finset.sum_eq_add, Finset.sum_singleton]
  sorry

end perimeter_pentagon_is_14_l134_134156


namespace smallest_number_divisible_l134_134565

theorem smallest_number_divisible (n : ℕ) : 
  (∀ k, k ∈ [12, 16, 18, 21, 28] -> (n - 7) % k = 0) -> n = 1015 :=
by
  intros h
  have h_lcm : Nat.gcd 12 16 ≠ 0, from sorry
  have lcm_12_16 : Nat.lcm 12 16 = 48, from sorry
  have lcm_1216_18 : Nat.lcm 48 18 = 144, from sorry
  have lcm_121618_21 : Nat.lcm 144 21 = 1008, from sorry
  have lcm_12161821_28 : Nat.lcm 1008 28 = 1008, from sorry
  have lcm_all : ∀ k, k ∈ [12, 16, 18, 21, 28] -> 1008 % k = 0, from sorry
  have n_value : n = 1008 + 7, from sorry
  rw n_value,
  exact 1015

end smallest_number_divisible_l134_134565


namespace log_comparison_l134_134617

/-- Prove that log base 3 of 7 is greater than log base 7 of 27. -/
theorem log_comparison : (log 3 7) > (log 7 27) :=
by sorry

end log_comparison_l134_134617


namespace number_of_arrangements_BANANAS_l134_134641

theorem number_of_arrangements_BANANAS : 
  let total_permutations := 7!
      a_repeats := 3!
      n_repeats := 2!
  in total_permutations / (a_repeats * n_repeats) = 420 :=
by
  sorry

end number_of_arrangements_BANANAS_l134_134641


namespace xyz_value_l134_134302

-- Define the positive values x, y, z and the given conditions.
variable {x y z : ℝ}
variable (hxy : x * y = 24 * real.cbrt 3)
variable (hxz : x * z = 40 * real.cbrt 3)
variable (hyz : y * z = 15 * real.cbrt 3)

-- Proposition to prove the value of xyz.
noncomputable def prove_xyz : ℝ :=
  if x > 0 ∧ y > 0 ∧ z > 0 then 72 * real.sqrt 2 else 0

-- Theorem statement that aggregates conditions and proves the exact value.
theorem xyz_value : x > 0 → y > 0 → z > 0 → hxy ∧ hxz ∧ hyz → x * y * z = 72 * real.sqrt 2 :=
by 
  intros
  sorry

end xyz_value_l134_134302


namespace angle_x_degrees_l134_134036

theorem angle_x_degrees 
  (O : Point)
  (A B C D : Point)
  (hC : Circle O)
  (hOB : O ∈ Segment O B)
  (hOC : O ∈ Segment O C)
  (hBCO : ∠ B C O = 32)
  (hACD : ∠ A C D = 90)
  (hCAD : ∠ C A D = 67) :
  ∠ B C A = 9 := 
  sorry

end angle_x_degrees_l134_134036


namespace area_of_circle_l134_134148

def circleEquation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

theorem area_of_circle :
  (∃ (center : ℝ × ℝ) (radius : ℝ), radius = 4 ∧ ∀ (x y : ℝ), circleEquation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) →
  (16 * Real.pi) = 16 * Real.pi := 
by 
  intro h
  have := h
  sorry

end area_of_circle_l134_134148


namespace cauchy_schwarz_inequality_l134_134325

theorem cauchy_schwarz_inequality (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by
  sorry

end cauchy_schwarz_inequality_l134_134325


namespace nonagon_diagonal_relation_l134_134230

theorem nonagon_diagonal_relation 
  (a b d : ℝ)
  (regular_nonagon : ∀ (a b d : ℝ), 
    side_length a ∧ shortest_diagonal b ∧ longest_diagonal d → 
    (∃ f g h : ℝ, angle_f_g_h = 20 ∧ 
      perpendicular_point_Q_R (Q : ℝ) (R : ℝ) a b d)) :
  d = a + b := 
begin
  sorry
end

end nonagon_diagonal_relation_l134_134230


namespace simplify_f_evaluate_f_at_minus_31_over_6_pi_l134_134678

noncomputable def f (α : ℝ) : ℝ := 
  (sin (4 * Real.pi - α) * cos (Real.pi - α) * cos (3 * Real.pi / 2 + α) * cos (7 * Real.pi / 2 - α)) / 
  (cos (Real.pi + α) * sin (2 * Real.pi - α) * sin (Real.pi + α) * sin (9 * Real.pi / 2 - α))

theorem simplify_f (α : ℝ) : f α = 1 :=
  sorry

theorem evaluate_f_at_minus_31_over_6_pi : f (-31 * Real.pi / 6) = - (Real.sqrt 3) / 3 :=
  sorry

end simplify_f_evaluate_f_at_minus_31_over_6_pi_l134_134678


namespace circle_numbers_contradiction_l134_134224

theorem circle_numbers_contradiction :
  ¬ ∃ (f : Fin 25 → Fin 25), ∀ i : Fin 25, 
  let a := f i
  let b := f ((i + 1) % 25)
  (b = a + 10 ∨ b = a - 10 ∨ ∃ k : Int, b = a * k) :=
by
  sorry

end circle_numbers_contradiction_l134_134224


namespace find_increasing_function_l134_134536

-- Define each function
def fA (x : ℝ) := -x
def fB (x : ℝ) := (2 / 3) ^ x
def fC (x : ℝ) := x ^ 2
def fD (x : ℝ) := x ^ (1 / 3)

-- Define the statement that fD is the only increasing function among the options
theorem find_increasing_function (f : ℝ → ℝ) (hf : f = fD) :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fA x < fA y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fB x < fB y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fC x < fC y) :=
by {
  sorry
}

end find_increasing_function_l134_134536


namespace percentage_reduction_is_20_l134_134482

noncomputable def reduction_in_length (L W : ℝ) (x : ℝ) := 
  (L * (1 - x / 100)) * (W * 1.25) = L * W

theorem percentage_reduction_is_20 (L W : ℝ) : 
  reduction_in_length L W 20 := 
by 
  unfold reduction_in_length
  sorry

end percentage_reduction_is_20_l134_134482


namespace cos_theta_of_planes_l134_134797

-- Define the normal vectors of the planes
def normal_vector1 : ℕ × ℕ × ℕ := (3, -1, 1)
def normal_vector2 : ℕ × ℕ × ℕ := (9, -3, 7)

-- Function to compute the cosine of the angle between two vectors
def cos_angle_between_planes (nv1 nv2 : ℕ × ℕ × ℕ) : ℚ :=
  let dot_product := nv1.1 * nv2.1 + nv1.2 * nv2.2 + nv1.3 * nv2.3
  let magnitude1 := Math.sqrt (nv1.1^2 + nv1.2^2 + nv1.3^2)
  let magnitude2 := Math.sqrt (nv2.1^2 + nv2.2^2 + nv2.3^2)
  dot_product / (magnitude1 * magnitude2)

-- The theorem that captures the mathematical proof problem
theorem cos_theta_of_planes : 
  cos_angle_between_planes normal_vector1 normal_vector2 = 37 / Real.sqrt 1529 := 
sorry

end cos_theta_of_planes_l134_134797


namespace find_fixed_point_l134_134256

def g (z : ℂ) : ℂ := ((1 + complex.I) * z + (3 * complex.I - real.sqrt 2)) / 2

theorem find_fixed_point : ∃ d : ℂ, g d = d ∧ d = (-3 / 2 : ℂ) + ((3 + real.sqrt 2) / 2) * complex.I :=
by {
  use (-3 / 2 : ℂ) + ((3 + real.sqrt 2) / 2) * complex.I,
  split,
  { sorry },
  { refl }
  }

end find_fixed_point_l134_134256


namespace omega_range_l134_134718

theorem omega_range :
  ∀ {ω : ℝ}, 
    ω > 0 → 
    (∀ x ∈ set.Icc 0 (2 * real.pi / 3), 
      sin (ω * x + real.pi / 3) = 0) 
    → (ω ∈ set.Ico 4 (11 / 2)) :=
begin
  sorry
end

end omega_range_l134_134718


namespace lesser_solution_quadratic_l134_134154

theorem lesser_solution_quadratic (x : ℝ) :
  x^2 + 9 * x - 22 = 0 → x = -11 ∨ x = 2 :=
sorry

end lesser_solution_quadratic_l134_134154


namespace fence_perimeter_l134_134929

-- Definitions for the given conditions
def num_posts : ℕ := 36
def gap_between_posts : ℕ := 6

-- Defining the proof problem to show the perimeter equals 192 feet
theorem fence_perimeter (n : ℕ) (g : ℕ) (sq_field : ℕ → ℕ → Prop) : n = 36 ∧ g = 6 ∧ sq_field n g → 4 * ((n / 4 - 1) * g) = 192 :=
by
  intro h
  sorry

end fence_perimeter_l134_134929


namespace remainder_of_sum_of_squares_mod_l134_134957

-- Define the function to compute the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Define the specific sum for the first 15 natural numbers
def S : ℕ := sum_of_squares 15

-- State the theorem
theorem remainder_of_sum_of_squares_mod (n : ℕ) (h : n = 15) : 
  S % 13 = 5 := by
  sorry

end remainder_of_sum_of_squares_mod_l134_134957


namespace at_least_one_digit_even_l134_134172

-- Definitions based on conditions from the problem
variables (n m : ℕ)
variables (h_n_digits : (10^16) ≤ n ∧ n < (10^17))
variables (h_m_reverse : m = (n.to_string.reverse.to_nat))

-- Lean statement to prove the question == answer given conditions
theorem at_least_one_digit_even (n m : ℕ) (h_n_digits : (10^16) ≤ n ∧ n < (10^17))
  (h_m_reverse : m = (n.to_string.reverse.to_nat)) :
  ∃ d, d ∈ ((n + m).to_string.data) ∧ d.to_nat % 2 = 0 :=
sorry

end at_least_one_digit_even_l134_134172


namespace integer_roots_7_values_of_a_l134_134266

theorem integer_roots_7_values_of_a :
  (∃ a : ℝ, (∀ r s : ℤ, (r + s = -a ∧ (r * s = 8 * a))) ∧ (∃ n : ℕ, n = 7)) :=
sorry

end integer_roots_7_values_of_a_l134_134266


namespace find_multiple_of_smaller_integer_l134_134122

theorem find_multiple_of_smaller_integer (L S k : ℕ) 
  (h1 : S = 10) 
  (h2 : L + S = 30) 
  (h3 : 2 * L = k * S - 10) 
  : k = 5 := 
by
  sorry

end find_multiple_of_smaller_integer_l134_134122


namespace division_remainder_false_l134_134195

theorem division_remainder_false :
  ¬(1700 / 500 = 17 / 5 ∧ (1700 % 500 = 3 ∧ 17 % 5 = 2)) := by
  sorry

end division_remainder_false_l134_134195


namespace least_n_div_mod_l134_134119

theorem least_n_div_mod (n : ℕ) (h_pos : n > 1) (h_mod25 : n % 25 = 1) (h_mod7 : n % 7 = 1) : n = 176 :=
by
  sorry

end least_n_div_mod_l134_134119


namespace total_books_97_l134_134434

variable (nDarryl nLamont nLoris : ℕ)

-- Conditions
def condition1 (nLoris nLamont : ℕ) : Prop := nLoris + 3 = nLamont
def condition2 (nLamont nDarryl : ℕ) : Prop := nLamont = 2 * nDarryl
def condition3 (nDarryl : ℕ) : Prop := nDarryl = 20

-- Theorem stating the total number of books is 97
theorem total_books_97 : nLoris + nLamont + nDarryl = 97 :=
by
  have h1 : nDarryl = 20 := condition3 nDarryl
  have h2 : nLamont = 2 * nDarryl := condition2 nLamont nDarryl
  have h3 : nLoris + 3 = nLamont := condition1 nLoris nLamont
  sorry

end total_books_97_l134_134434


namespace sequence_inequality_proof_l134_134686

theorem sequence_inequality_proof (r : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ)
  (hr1 : r 1 = 2)
  (hr2 : ∀ k, 2 ≤ k → r k = (∏ i in finset.range (k - 1), r (i + 1)) + 1)
  (ha : ∑ i in finset.range n, 1 / (a (i + 1) : ℝ) < 1) :
  ∑ i in finset.range n, 1 / (a (i + 1) : ℝ) ≤ ∑ i in finset.range n, 1 / (r (i + 1) : ℝ) :=
sorry

end sequence_inequality_proof_l134_134686


namespace monotonic_intervals_extremum_values_l134_134339

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 8

theorem monotonic_intervals :
  (∀ x, x < -1 → deriv f x > 0) ∧
  (∀ x, x > 2 → deriv f x > 0) ∧
  (∀ x, -1 < x ∧ x < 2 → deriv f x < 0) := sorry

theorem extremum_values :
  ∃ a b : ℝ, (a = -12) ∧ (b = 15) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≥ b → f x = b) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≤ a → f x = a) := sorry

end monotonic_intervals_extremum_values_l134_134339


namespace meaningful_fraction_range_l134_134133

theorem meaningful_fraction_range (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) := sorry

end meaningful_fraction_range_l134_134133


namespace find_n_l134_134287

theorem find_n (n : ℤ) (h0 : 0 ≤ n) (h1 : n ≤ 180) (h2 : sin (n * real.pi / 180) = sin (845 * real.pi / 180)) :
  n = 55 :=
by
  sorry

end find_n_l134_134287


namespace probability_of_positive_test_l134_134866

theorem probability_of_positive_test :
  let P_A := 0.01
  let P_B_given_A := 0.99
  let P_B_given_not_A := 0.1
  let P_not_A := 0.99
  let P_B := P_B_given_A * P_A + P_B_given_not_A * P_not_A
  in P_B = 0.1089 :=
by
  let P_A := 0.01
  let P_B_given_A := 0.99
  let P_B_given_not_A := 0.1
  let P_not_A := 0.99
  let P_B := P_B_given_A * P_A + P_B_given_not_A * P_not_A
  have : P_B = 0.1089, from by norm_num [P_B_given_A, P_A, P_B_given_not_A, P_not_A, P_B]
  exact this

end probability_of_positive_test_l134_134866


namespace solve_for_n_l134_134850

theorem solve_for_n (n : ℤ) :
  3^(3 * n + 2) = 1 / 81 → n = -2 :=
by
  sorry

end solve_for_n_l134_134850


namespace painted_cube_l134_134581

noncomputable def cube_side_length : ℕ :=
  7

theorem painted_cube (painted_faces: ℕ) (one_side_painted_cubes: ℕ) (orig_side_length: ℕ) :
    painted_faces = 6 ∧ one_side_painted_cubes = 54 ∧ (orig_side_length + 2) ^ 2 / 6 = 9 →
    orig_side_length = cube_side_length :=
by
  sorry

end painted_cube_l134_134581


namespace find_d17_l134_134409

-- Define the positive divisors of n
def positive_divisors (n : ℕ) : list ℕ := List.filter (λ d, d ∣ n) (List.iota n)

-- Define the problem conditions
def divisors_n (n : ℕ) (d : ℕ → ℕ) := 
  ∃ (r : ℕ), 1 = d 0 ∧ d r = n ∧ ∀ i j, i < j → d i < d j ∧ d i ∣ n

def pythagorean_divisors (n : ℕ) (d : ℕ → ℕ) := 
  (d 6)^2 + (d 14)^2 = (d 15)^2

-- Define the proof statement
theorem find_d17 (n : ℕ) (d : ℕ → ℕ) : 
  (divisors_n n d) → 
  (pythagorean_divisors n d) → 
  d 16 = 28 :=
begin
  sorry
end

end find_d17_l134_134409


namespace affine_transformation_regular_pentagon_l134_134275

theorem affine_transformation_regular_pentagon 
  (P : Pentagon) 
  (h_convex : P.is_convex) 
  (h_parallel : ∀ d, d ∈ P.diagonals → ∃ s, s ∈ P.sides ∧ d ∥ s) : 
  ∃ T : AffineTransformation, T.transforms P RegularPentagon :=
by
  sorry

end affine_transformation_regular_pentagon_l134_134275


namespace arithmetic_sequence_fifth_term_l134_134102

theorem arithmetic_sequence_fifth_term (x y : ℝ) (h1 : x = 2) (h2 : y = 1) :
    let a1 := x^2 + y^2
    let a2 := x^2 - y^2
    let a3 := x^2 * y^2
    let a4 := x^2 / y^2
    let d := a2 - a1
    let a5 := a4 + d
    a5 = 2 := by
  sorry

end arithmetic_sequence_fifth_term_l134_134102


namespace race_distance_l134_134759

theorem race_distance (Va Vb Vc : ℝ) (D : ℝ) :
    (Va / Vb = 10 / 9) →
    (Va / Vc = 80 / 63) →
    (Vb / Vc = 8 / 7) →
    (D - 100) / D = 7 / 8 → 
    D = 700 :=
by
  intros h1 h2 h3 h4 
  sorry

end race_distance_l134_134759


namespace smallest_degree_of_denominator_l134_134650

noncomputable def degree_of_numerator : ℕ := 8

def has_horizontal_asymptote (deg_num : ℕ) (deg_denom : ℕ) : Prop :=
  deg_num <= deg_denom

theorem smallest_degree_of_denominator :
  ∃ deg_denom : ℕ, has_horizontal_asymptote degree_of_numerator deg_denom ∧ 
                   ∀ d : ℕ, has_horizontal_asymptote degree_of_numerator d → degree_of_numerator ≤ d :=
begin
  use 8,
  split,
  { unfold has_horizontal_asymptote,
    exact le_refl 8 },
  { intros d hd,
    exact hd }
end

end smallest_degree_of_denominator_l134_134650


namespace angle_x_is_9_degrees_l134_134057

theorem angle_x_is_9_degrees
  (O B C A D : Type)
  (hO_center : is_center O)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (hOB_eq_OC : distance O B = distance O C)
  (hAngle_BCO : angle B C O = 32)
  (hOA_eq_OC : distance O A = distance O C)
  (hAngle_CAD : angle C A D = 67)
  : angle x = 9 :=
sorry

end angle_x_is_9_degrees_l134_134057


namespace graph_passes_through_point_l134_134099

theorem graph_passes_through_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ x y : ℝ, (y = log a (x - 1) + 1) ∧ (x = 2) ∧ (y = 1) :=
by
  use 2
  use 1
  simp [log, h1, h2]
  sorry

end graph_passes_through_point_l134_134099


namespace average_of_integers_l134_134153

theorem average_of_integers (M : ℤ) :
  (∃ n : ℕ, 15 ≤ n ∧ n ≤ 23) → 
  (∃ m : ℕ, 14 < m.toInt ∧ m.toInt < 24 ∧ M = m.toInt) → 
  ∃ avg : ℤ, avg = 19 :=
by sorry

end average_of_integers_l134_134153


namespace highest_temperature_l134_134229

theorem highest_temperature
  (initial_temp : ℝ := 60)
  (final_temp : ℝ := 170)
  (heating_rate : ℝ := 5)
  (cooling_rate : ℝ := 7)
  (total_time : ℝ := 46) :
  ∃ T : ℝ, (T - initial_temp) / heating_rate + (T - final_temp) / cooling_rate = total_time ∧ T = 240 :=
by
  sorry

end highest_temperature_l134_134229


namespace prime_factors_product_l134_134506

theorem prime_factors_product {p1 p2 p3 p4 : ℕ} (h1 : p1 = 2) (h2 : p2 = 3) (h3 : p3 = 5) (h4 : p4 = 7) :
  p1 * p2 * p3 * p4 = 210 := by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end prime_factors_product_l134_134506


namespace find_n_l134_134362

theorem find_n (n : ℝ) (h : sqrt (10 + n) = 9) : n = 71 := 
by sorry

end find_n_l134_134362


namespace find_number_l134_134175

theorem find_number (x : ℤ) : (150 - x = x + 68) → x = 41 :=
by
  intro h
  sorry

end find_number_l134_134175


namespace total_cupcakes_l134_134614

noncomputable def cupcakesForBonnie : ℕ := 24
noncomputable def cupcakesPerDay : ℕ := 60
noncomputable def days : ℕ := 2

theorem total_cupcakes : (cupcakesPerDay * days + cupcakesForBonnie) = 144 := 
by
  sorry

end total_cupcakes_l134_134614


namespace monotonic_intervals_1_monotonic_intervals_2_fx_gt_gx_l134_134342

-- Definitions given
def f (x : ℝ) : ℝ := x * Real.exp x
def F (x : ℝ) (a : ℝ) : ℝ := f(x) + a * (0.5 * x^2 + x)
def g (x : ℝ) : ℝ := f(-2 - x)

-- Proof statement for monotonic intervals
theorem monotonic_intervals_1 {a : ℝ} (h : a >= 0) :
  (∀ x < -1, F' x < 0) ∧
  (∀ x > -1, F' x > 0) := by sorry

-- Proof statement for monotonic intervals with -1/e < a < 0
theorem monotonic_intervals_2 {a : ℝ} (h : -1 / Real.exp 1 < a) (h' : a < 0) :
  (∀ x < Real.log (-a), F' x > 0) ∧
  (∀ x > Real.log (-a), x < -1, F' x < 0) ∧
  (∀ x > -1, F' x > 0) := by sorry

-- Proof statement for inequality
theorem fx_gt_gx (x : ℝ) (h : x > -1) :
  f(x) > g(x) := by sorry

end monotonic_intervals_1_monotonic_intervals_2_fx_gt_gx_l134_134342


namespace program_output_l134_134532

theorem program_output (a : ℕ) (h : a = 3) : (if a < 10 then 2 * a else a * a) = 6 :=
by
  rw [h]
  norm_num

end program_output_l134_134532


namespace increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l134_134544

noncomputable def fA (x : ℝ) : ℝ := -x
noncomputable def fB (x : ℝ) : ℝ := (2/3)^x
noncomputable def fC (x : ℝ) : ℝ := x^2
noncomputable def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function_fA : ¬∀ x y : ℝ, x < y → fA x < fA y := sorry
theorem increasing_function_fB : ¬∀ x y : ℝ, x < y → fB x < fB y := sorry
theorem increasing_function_fC : ¬∀ x y : ℝ, x < y → fC x < fC y := sorry
theorem increasing_function_fD : ∀ x y : ℝ, x < y → fD x < fD y := sorry

end increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l134_134544


namespace cats_in_studio_count_l134_134233

theorem cats_in_studio_count :
  (70 + 40 + 30 + 50
  - 25 - 15 - 20 - 28
  + 5 + 10 + 12
  - 8
  + 12) = 129 :=
by sorry

end cats_in_studio_count_l134_134233


namespace combined_salary_ABC_and_E_l134_134494

def salary_D : ℕ := 7000
def avg_salary : ℕ := 9000
def num_individuals : ℕ := 5

theorem combined_salary_ABC_and_E :
  (avg_salary * num_individuals - salary_D) = 38000 :=
by
  -- proof goes here
  sorry

end combined_salary_ABC_and_E_l134_134494


namespace Lilibeth_strawberry_baskets_l134_134816

theorem Lilibeth_strawberry_baskets :
  ∀ (baskets : ℕ), (∀ (strawberries_per_basket : ℕ), strawberries_per_basket = 50 →
  (∀ (total_strawberries : ℕ), total_strawberries = 1200 →
  (Lilibeth_strawberries : ℕ), Lilibeth_strawberries = baskets * strawberries_per_basket →
  4 * Lilibeth_strawberries = total_strawberries →
  baskets = 6)) :=
  by
    intros baskets strawberries_per_basket H_str total_strawberries H_tot Lilibeth_strawberries H_Lilibeth H_eq
    sorry

end Lilibeth_strawberry_baskets_l134_134816


namespace bus_return_trip_fraction_l134_134201

theorem bus_return_trip_fraction :
  (3 / 4 * 200 + x * 200 = 310) → (x = 4 / 5) := by
  sorry

end bus_return_trip_fraction_l134_134201


namespace frog_reaches_seven_l134_134431

theorem frog_reaches_seven (p q : ℕ) (h_rel_prime : Nat.coprime p q) (h_complete_probability : p = 43 ∧ q = 64) :
  p + q = 107 :=
by
  sorry

end frog_reaches_seven_l134_134431


namespace area_triangle_ACE_l134_134384

theorem area_triangle_ACE (AB AC AD AE : ℝ) (h : (AB = 8) ∧ (AC = 12) ∧ (AD = 16) ∧ (AE = (1 / 4) * AD)) :
  let area_ABC := (1 / 2) * AB * AC in 
  let area_ratio := (AE / AC)^2 in
  (area_ratio * area_ABC = 16 / 3) :=
by
  sorry

end area_triangle_ACE_l134_134384


namespace yearly_annual_income_correct_l134_134453

theorem yearly_annual_income_correct :
  let total_amount := 2500
  let P1 := 1500
  let P2 := total_amount - P1
  let rate1 := 0.05
  let rate2 := 0.06
  let time := 1
  let I1 := P1 * rate1 * time
  let I2 := P2 * rate2 * time
  let total_income := I1 + I2
  total_income = 135 := by
{
  let total_amount := 2500
  let P1 := 1500
  let P2 := total_amount - P1
  let rate1 := 0.05
  let rate2 := 0.06
  let time := 1
  let I1 := P1 * rate1 * time
  let I2 := P2 * rate2 * time
  let total_income := I1 + I2
  have h1 : total_income = 135, by sorry
  exact h1
}

end yearly_annual_income_correct_l134_134453


namespace triangle_inequality_sum_n_l134_134218

theorem triangle_inequality_sum_n :
  let possible_n_values := {n : ℕ | 5 ≤ n ∧ n < 18}
  ∑ i in possible_n_values, i = 143 :=
by
  sorry

end triangle_inequality_sum_n_l134_134218


namespace greatest_odd_factors_lt_1000_l134_134822

theorem greatest_odd_factors_lt_1000 : ∃ n : ℕ, n < 1000 ∧ n.factors.count % 2 = 1 ∧ n = 961 := 
by {
  sorry
}

end greatest_odd_factors_lt_1000_l134_134822


namespace double_operation_result_l134_134292

noncomputable def operation (v : ℝ) : ℝ := v - v / 3

theorem double_operation_result :
  let v := 44.99999999999999 in
  operation (operation v) = 19.999999999999995 :=
by
  let v := 44.99999999999999
  let v1 := operation v
  let v2 := operation v1
  show v2 = 19.999999999999995
  sorry

end double_operation_result_l134_134292


namespace mobile_purchase_price_l134_134786

theorem mobile_purchase_price (M : ℝ) 
  (P_grinder : ℝ := 15000)
  (L_grinder : ℝ := 0.05 * P_grinder)
  (SP_grinder : ℝ := P_grinder - L_grinder)
  (SP_mobile : ℝ := 1.1 * M)
  (P_overall : ℝ := P_grinder + M)
  (SP_overall : ℝ := SP_grinder + SP_mobile)
  (profit : ℝ := 50)
  (h : SP_overall = P_overall + profit) :
  M = 8000 :=
by 
  sorry

end mobile_purchase_price_l134_134786


namespace greatest_odd_factors_lt_1000_l134_134821

theorem greatest_odd_factors_lt_1000 : ∃ n : ℕ, n < 1000 ∧ n.factors.count % 2 = 1 ∧ n = 961 := 
by {
  sorry
}

end greatest_odd_factors_lt_1000_l134_134821


namespace area_enclosed_by_region_l134_134144

theorem area_enclosed_by_region :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6*x + 8*y = -9) →
  let radius := 4 in
  let area := real.pi * radius^2 in
  area = 16 * real.pi :=
sorry

end area_enclosed_by_region_l134_134144


namespace smaller_ladder_steps_l134_134634

-- Define the number of steps climbed on the full ladder
def full_ladder_steps : ℕ := 11
def full_ladder_times : ℕ := 10
def full_ladder_total_steps := full_ladder_steps * full_ladder_times

-- Define the total steps climbed and the number of climbs on the smaller ladder
def total_steps_climbed : ℕ := 152
def small_ladder_times : ℕ := 7

-- Define the number of steps on the smaller ladder as a variable
variable (x : ℕ)

-- The proof statement
theorem smaller_ladder_steps :
  full_ladder_total_steps + small_ladder_times * x = total_steps_climbed → x = 6 :=
by
  intro h
  have h1 : 110 = full_ladder_total_steps := by norm_num
  rw [h1, add_comm] at h
  have h2 : 7 * x = 42 := by linarith
  have h3 : x = 6 := by linarith
  exact h3

end smaller_ladder_steps_l134_134634


namespace exists_rectangle_same_color_l134_134771

-- Define the necessary sets and types
def ℕ' := {n : ℕ | n > 0}

noncomputable def M : set (ℕ' × ℕ') := 
  { p | p.1 ≤ 12 ∧ p.2 ≤ 12 }

-- Define colors as an enumeration type
inductive Color
| Red
| White
| Blue

-- Assume we have a coloring function for the set M
noncomputable def coloring : (ℕ' × ℕ') → Color := sorry

-- The theorem statement
theorem exists_rectangle_same_color :
  ∃ (v₁ v₂ v₃ v₄ : ℕ' × ℕ'),
    v₁ ∈ M ∧ v₂ ∈ M ∧ v₃ ∈ M ∧ v₄ ∈ M ∧
    v₁.1 = v₂.1 ∧ v₃.1 = v₄.1 ∧
    v₁.2 = v₃.2 ∧ v₂.2 = v₄.2 ∧
    coloring v₁ = coloring v₂ ∧ coloring v₁ = coloring v₃ ∧ coloring v₁ = coloring v₄ :=
begin
  sorry
end

end exists_rectangle_same_color_l134_134771


namespace altitude_BD_median_BE_l134_134331

noncomputable theory

def point := ℝ × ℝ

-- Definition for the three points A, B, and C
def A : point := (3, -4)
def B : point := (6, 0)
def C : point := (-5, 2)

-- Predicate representing the equation of a line in the form ax + by + c = 0
def equation_of_line (a b c : ℝ) (P : point) : Prop :=
  let (x,y) := P in
  a * x + b * y + c = 0

-- The two theorem statements
theorem altitude_BD : equation_of_line 4 (-3) (-24) B := by
  sorry

theorem median_BE : equation_of_line 1 (-7) (-6) B := by
  sorry

end altitude_BD_median_BE_l134_134331


namespace remove_5_increases_probability_l134_134140

theorem remove_5_increases_probability :
  let T := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let remove_5 := T.erase 5
  let valid_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
    S.product S.filter (λ b, 10 - b ∈ S) -- pairs (a, b) with a + b = 10 and distinct

  (0 < (valid_pairs remove_5).card) ∧
  (valid_pairs remove_5).card ≤ 4 :=
begin
  let T := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ,
  let valid_pairs := λ S : Finset ℕ, S.product S.filter (λ b, 10 - b ∈ S),

  have hs: T.card = 10 := rfl,
  have h_remove_5: (T.erase 5).card = 9 := rfl,

  sorry
end

end remove_5_increases_probability_l134_134140


namespace volume_of_prism_l134_134470

variables (a b : ℝ) (α β : ℝ)
  (h1 : a > b)
  (h2 : 0 < α ∧ α < π / 2)
  (h3 : 0 < β ∧ β < π / 2)

noncomputable def volume_prism : ℝ :=
  (a^2 - b^2) * (a - b) / 8 * (Real.tan α)^2 * Real.tan β

theorem volume_of_prism (a b α β : ℝ) (h1 : a > b) (h2 : 0 < α ∧ α < π / 2) (h3 : 0 < β ∧ β < π / 2) :
  volume_prism a b α β = (a^2 - b^2) * (a - b) / 8 * (Real.tan α)^2 * Real.tan β := by
  sorry

end volume_of_prism_l134_134470


namespace simplify_fraction_l134_134848

theorem simplify_fraction (x : ℝ) (h : x ≠ 0) : (6 / (5 * x^(-4))) * ((5 * x^3) / 4) = (3 / 2) * x^7 := by
  sorry

end simplify_fraction_l134_134848


namespace question1_question2_l134_134299

-- Define the points A, B, and C
structure Point (α : Type) :=
  (x y : α)

abbreviation A : Point ℚ := { x := 1, y := 2 }
abbreviation B : Point ℚ := { x := -1, y := 4 }
abbreviation C (k : ℚ) : Point ℚ := { x := 4, y := k }

-- Define vector subtraction
def vec_sub (P Q : Point ℚ) : Point ℚ :=
  { x := P.x - Q.x, y := P.y - Q.y }

-- Define vector magnitude
def magnitude (v : Point ℚ) : ℚ :=
  Real.sqrt (v.x^2 + v.y^2)

-- Define point collinearity
def collinear (A B C : Point ℚ) : Prop :=
  (vec_sub B A).x * (vec_sub C A).y = (vec_sub B A).y * (vec_sub C A).x

-- Question 1: Prove magnitude of AC
theorem question1
  (k : ℚ)
  (h_collinear : collinear A B (C k)) :
  magnitude (vec_sub (C (-1)) A) = 3 * Real.sqrt 2 :=
sorry

-- Define Dot Product
def dot_product (v w : Point ℚ) : ℚ :=
  v.x * w.x + v.y * w.y

-- Question 2: Prove cosine of angle between AB and AC
theorem question2
  (k : ℚ)
  (h_perpendicular : dot_product (vec_sub B A) (vec_sub C k) = 0) :
  (dot_product (vec_sub B A) (vec_sub (C 1) A)) /
    (magnitude (vec_sub B A) * magnitude (vec_sub (C 1) A)) = - (2 * Real.sqrt 5) / 5 :=
sorry

end question1_question2_l134_134299


namespace earnings_in_six_minutes_l134_134833

def ticket_cost : ℕ := 3
def tickets_per_minute : ℕ := 5
def minutes : ℕ := 6

theorem earnings_in_six_minutes (ticket_cost : ℕ) (tickets_per_minute : ℕ) (minutes : ℕ) :
  ticket_cost = 3 → tickets_per_minute = 5 → minutes = 6 → (tickets_per_minute * ticket_cost * minutes = 90) :=
by
  intros h_cost h_tickets h_minutes
  rw [h_cost, h_tickets, h_minutes]
  norm_num

end earnings_in_six_minutes_l134_134833


namespace number_of_pupils_l134_134515

-- Define the number of total people
def total_people : ℕ := 803

-- Define the number of parents
def parents : ℕ := 105

-- We need to prove the number of pupils is 698
theorem number_of_pupils : (total_people - parents) = 698 := 
by
  -- Skip the proof steps
  sorry

end number_of_pupils_l134_134515


namespace min_value_quadratic_l134_134966

theorem min_value_quadratic :
  ∀ x : ℝ, (x - 7)^2 - 9 ≤ (x^2 - 14x + 40) := 
by sorry

end min_value_quadratic_l134_134966


namespace min_value_l134_134310

def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

def positive_seq (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

theorem min_value (a : ℕ → ℝ) (q : ℝ) (m n : ℕ) 
  (h_geom : geom_seq a q) 
  (h_pos : positive_seq a)
  (h_a3 : a 3 = 2 * a 1 + a 2)
  (h_sqrt : sqrt (a m * a n) = 4 * a 1) :
  (1 / m + 4 / n) = 3 / 2 := 
sorry

end min_value_l134_134310


namespace equal_numbers_l134_134517

open Nat

theorem equal_numbers (x y z : ℕ) (h1 : x ∣ gcd y z) (h2 : y ∣ gcd x z) (h3 : z ∣ gcd x y)
  (h4 : lcm x y ∣ z) (h5 : lcm y z ∣ x) (h6 : lcm z x ∣ y) : x = y ∧ y = z :=
  sorry

end equal_numbers_l134_134517


namespace charlie_steps_l134_134250

theorem charlie_steps (steps_per_run : ℕ) (runs : ℝ) (expected_steps : ℕ) :
  steps_per_run = 5350 →
  runs = 2.5 →
  expected_steps = 13375 →
  runs * steps_per_run = expected_steps :=
by intros; linarith; sorry

end charlie_steps_l134_134250


namespace positive_number_l134_134124

theorem positive_number (n : ℕ) (h : n^2 + 2 * n = 170) : n = 12 :=
sorry

end positive_number_l134_134124
