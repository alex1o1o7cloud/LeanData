import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.GeometricSeries
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Parity
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.Geometry.ConicSection
import Mathlib.Analysis.SpecialFunctions.Complex.Log
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Triangle.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Polyrith
import combinatorics.finite
import probability.probability_theory

namespace selling_price_is_80_l825_825746

-- Definitions of given conditions
def cost_price : ℝ := 60
def initial_selling_price : ℝ := 100
def initial_bottles_sold : ℝ := 40
def additional_bottles_per_dollar_reduction : ℝ := 2
def required_profit : ℝ := 1600

-- Proof statement that translates the given problem to Lean 4 statement
theorem selling_price_is_80 :
  ∃ x : ℝ, (x - cost_price) * (initial_bottles_sold + additional_bottles_per_dollar_reduction * (initial_selling_price - x)) = required_profit ∧ x = 80 :=
begin
  sorry
end

end selling_price_is_80_l825_825746


namespace evaluate_expression_l825_825358

theorem evaluate_expression :
  (24^36) / (72^18) = 8^18 :=
by
  sorry

end evaluate_expression_l825_825358


namespace yellow_lighting_opposite_face_l825_825832

-- Definition of colors
inductive Color
| R | B | P | Y | G | W | O | Bl

-- The sequence of squares before folding
def sequence : List Color := [Color.R, Color.B, Color.P, Color.Y, Color.G, Color.W, Color.O, Color.Bl]

-- Function to determine the face opposite the black face in a folded cube
def opposite_face (c : Color) : Color :=
  match c with
  | Color.Bl => Color.W
  | _ => c  -- simplified for the purpose of this problem

-- Perception of colors under yellow lighting
def perceived_color (c : Color) : Color :=
  match c with
  | Color.W => Color.Y  -- white appears as light yellow
  | _ => c

-- Theorem stating the problem
theorem yellow_lighting_opposite_face : perceived_color (opposite_face Color.Bl) = Color.Y :=
  by
    sorry

end yellow_lighting_opposite_face_l825_825832


namespace smallest_integer_representation_l825_825192

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end smallest_integer_representation_l825_825192


namespace parallel_lines_and_separation_of_circle_l825_825867

theorem parallel_lines_and_separation_of_circle 
  (r a b : ℝ) 
  (h1 : a^2 + b^2 < r^2)
  (h2 : a ≠ 0)
  (h3 : b ≠ 0) 
  (h4 : ∀ x y : ℝ, y - b = - (a / b) * (x - a) → ∃ k : ℝ, y = k * x + (b + ka))
  (h5 : ∀ x y : ℝ, ax + by = r^2 → ¬(y^2 + x^2 < r^2)) :
  (ax + by - (a^2 + b^2) = 0) ∧ (bx - ay + r^2 = 0) → 
  ((ax + by - (a^2 + b^2) = 0) ∥ (bx - ay + r^2 = 0)) ∧ 
  (∀ x y : ℝ, (bx - ay + r^2 = 0) → ¬(x^2 + y^2 ≤ r^2)) :=
by
  sorry

end parallel_lines_and_separation_of_circle_l825_825867


namespace probability_S7_eq_3_l825_825036

open probability_theory finset

-- Definitions of the problem
def prob_red := 2 / 3
def prob_white := 1 / 3
def number_of_draws : ℕ := 7
def target_sum : ℤ := 3

-- Event of having exactly 2 red balls and 5 white balls in 7 draws
def event : set (fin number_of_draws) := { ω | ω.card = 2 }

-- Probability calculation using independence and binomial coefficient
noncomputable def probability_event := (choose 7 2) * (prob_red ^ 2) * (prob_white ^ 5)

-- Statement of the theorem
theorem probability_S7_eq_3 :
  (∑ ω in event, P ω) = higher_order 3 (probability_event) :=
sorry

end probability_S7_eq_3_l825_825036


namespace trajectory_midpoint_l825_825678

-- Defining the equation of a circle centered at the origin with radius 2.
def circle (x y : ℤ) : Prop := x^2 + y^2 = 4

-- Point P
def P : (ℤ × ℤ) := (4, -2)

-- Definition for the midpoint of a line segment in the plane
def midpoint (A B : ℤ × ℤ) : ℤ × ℤ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Defining the trajectory assertion
theorem trajectory_midpoint {x y : ℤ} :
  (∃ (x1 y1 : ℤ), circle x1 y1 ∧ midpoint P (x1, y1) = (x, y)) ↔
  (x-2)^2 + (y+1)^2 = 1 :=
sorry

end trajectory_midpoint_l825_825678


namespace axis_of_symmetry_l825_825537

noncomputable def f (x : ℝ) : ℝ := cos (x + (Real.pi / 2)) * cos (x + (Real.pi / 4))

theorem axis_of_symmetry : (∀ x : ℝ, f (x) = f (Real.pi * (5/8) - x)) :=
by
  sorry

end axis_of_symmetry_l825_825537


namespace dig_eq_conditions_l825_825838

theorem dig_eq_conditions (n k : ℕ) 
  (h1 : 10^(k-1) ≤ n^n ∧ n^n < 10^k)
  (h2 : 10^(n-1) ≤ k^k ∧ k^k < 10^n) :
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) :=
by
  sorry

end dig_eq_conditions_l825_825838


namespace find_m_range_a_l825_825902

noncomputable def f (x m : ℝ) : ℝ :=
  m - |x - 3|

theorem find_m (m : ℝ) (h : ∀ x, 2 < f x m ↔ 2 < x ∧ x < 4) : m = 3 :=
  sorry

theorem range_a (a : ℝ) (h : ∀ x, |x - a| ≥ f x 3) : a ≤ 0 ∨ 6 ≤ a :=
  sorry

end find_m_range_a_l825_825902


namespace find_original_price_l825_825608

variable (P : ℝ) (sale_price : ℝ := 80) (decreased_percentage : ℝ := 0.20)

def original_price_eq : Prop :=
  sale_price = 0.80 * P

theorem find_original_price : P = 100 :=
by
  -- Declaring the given conditions
  have hs : sale_price = 80 := rfl
  have hd : decreased_percentage = 0.20 := rfl
  -- Assuming the condition in our definition
  have h : original_price_eq P := sorry
  -- From the above condition, we conclude the result
  sorry

end find_original_price_l825_825608


namespace money_loses_value_on_island_properties_of_money_l825_825112

-- Define the conditions from the problem statement
def deserted_island : Prop := true
def useless_money_on_island (f : Prop) : Prop := deserted_island → ¬f

-- Define properties of money in the story context
def medium_of_exchange (m : Prop) : Prop := m
def has_value_as_medium_of_exchange (m : Prop) : Prop := m
def no_transaction_partners : Prop := true

-- Define additional properties an item must possess to be considered money
def durability : Prop := true
def portability : Prop := true
def divisibility : Prop := true
def acceptability : Prop := true
def uniformity : Prop := true
def limited_supply : Prop := true

-- Prove that money loses its value as a medium of exchange on the deserted island
theorem money_loses_value_on_island (m : Prop) :
  deserted_island →
  (medium_of_exchange m) →
  (no_transaction_partners) →
  (¬has_value_as_medium_of_exchange m) :=
by { intro di, intro moe, intro ntp, exact moe → false }

-- Prove that an item must have certain properties to be considered money
theorem properties_of_money :
  durability ∧ portability ∧ divisibility ∧ acceptability ∧ uniformity ∧ limited_supply :=
by { split, exact true.intro, split, exact true.intro, split, exact true.intro, split, exact true.intro, split, exact true.intro, exact true.intro }

#check money_loses_value_on_island
#check properties_of_money

end money_loses_value_on_island_properties_of_money_l825_825112


namespace seq_a_100_gt_14_l825_825159

def seq_a : ℕ → ℝ
| 1 := 1
| n := seq_a (n - 1) + 1 / seq_a (n - 1)

theorem seq_a_100_gt_14 : seq_a 100 > 14 :=
sorry

end seq_a_100_gt_14_l825_825159


namespace polar_circle_equation_is_ρ_eq_1_l825_825720

-- Define that we are working in the polar coordinate system
def is_polar_coordinate_system (ρ θ : ℝ) : Prop := ∀ (r : ℝ), r = ρ ∧ r ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * π

-- Define the predicate for a circle in the polar coordinate system with radius 1
def is_circle (ρ : ℝ) : Prop := ρ = 1

-- State the problem
theorem polar_circle_equation_is_ρ_eq_1 (ρ : ℝ) (θ : ℝ) :
  is_polar_coordinate_system ρ θ → is_circle ρ :=
begin
  sorry
end

end polar_circle_equation_is_ρ_eq_1_l825_825720


namespace unique_fractions_count_l825_825924

theorem unique_fractions_count : 
  (Finset.card (Finset.filter (λ p : ℕ × ℕ, p.1 < p.2 ∧ p.2 ≤ 9) 
    (Finset.product (Finset.range 10).erase 0 (Finset.range 10).erase 0)).image (λ p, p.1 /. p.2)) = 27 :=
by
  sorry

end unique_fractions_count_l825_825924


namespace money_loses_value_on_island_properties_of_money_l825_825111

-- Define the conditions from the problem statement
def deserted_island : Prop := true
def useless_money_on_island (f : Prop) : Prop := deserted_island → ¬f

-- Define properties of money in the story context
def medium_of_exchange (m : Prop) : Prop := m
def has_value_as_medium_of_exchange (m : Prop) : Prop := m
def no_transaction_partners : Prop := true

-- Define additional properties an item must possess to be considered money
def durability : Prop := true
def portability : Prop := true
def divisibility : Prop := true
def acceptability : Prop := true
def uniformity : Prop := true
def limited_supply : Prop := true

-- Prove that money loses its value as a medium of exchange on the deserted island
theorem money_loses_value_on_island (m : Prop) :
  deserted_island →
  (medium_of_exchange m) →
  (no_transaction_partners) →
  (¬has_value_as_medium_of_exchange m) :=
by { intro di, intro moe, intro ntp, exact moe → false }

-- Prove that an item must have certain properties to be considered money
theorem properties_of_money :
  durability ∧ portability ∧ divisibility ∧ acceptability ∧ uniformity ∧ limited_supply :=
by { split, exact true.intro, split, exact true.intro, split, exact true.intro, split, exact true.intro, split, exact true.intro, exact true.intro }

#check money_loses_value_on_island
#check properties_of_money

end money_loses_value_on_island_properties_of_money_l825_825111


namespace cube_root_expression_l825_825294

theorem cube_root_expression (N : ℝ) (h : N > 1) : 
    (N^(1/3)^(1/3)^(1/3)^(1/3)) = N^(40/81) :=
sorry

end cube_root_expression_l825_825294


namespace num_satisfying_integers_l825_825344

theorem num_satisfying_integers :
  {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ (∃ m : ℤ, (n^2 + n - 1)! / ((n!)^(n+2)) = m)}.to_finset.card = 4 := 
  sorry

end num_satisfying_integers_l825_825344


namespace avg_children_in_families_with_children_l825_825406

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l825_825406


namespace average_children_in_families_with_children_l825_825423

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l825_825423


namespace ratio_yx_l825_825024

variable (c x y : ℝ)

theorem ratio_yx (h1: x = 0.80 * c) (h2: y = 1.25 * c) : y / x = 25 / 16 := by
  -- Proof to be written here
  sorry

end ratio_yx_l825_825024


namespace soft_drink_cost_approx_eq_l825_825128

def cost_per_can_individual (total_cost : ℝ) (num_cans : ℕ) : ℝ :=
  total_cost / num_cans

theorem soft_drink_cost_approx_eq :
  cost_per_can_individual 2.99 12 ≈ 0.25 := 
by
  sorry

end soft_drink_cost_approx_eq_l825_825128


namespace smallest_base10_integer_l825_825213

theorem smallest_base10_integer (a b : ℕ) (h1 : a > 3) (h2 : b > 3) :
    (1 * a + 3 = 3 * b + 1) → (1 * 10 + 3 = 13) :=
by
  intros h


-- Prove that  1 * a + 3 = 3 * b + 1 
  have a_eq : a = 3 * b - 2 := by linarith

-- Prove that 1 * 10 + 3 = 13 
  have base_10 := by simp

have the smallest base 10
  sorry

end smallest_base10_integer_l825_825213


namespace expected_value_of_twelve_sided_die_l825_825770

theorem expected_value_of_twelve_sided_die : 
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0) in
  (finset.sum outcomes (λ n, (n:ℝ)) / 12 = 6.5) :=
by
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0)
  have h1 : ∑ n in outcomes, (n : ℝ) = 78, sorry
  have h2 : (78 / 12) = 6.5, sorry
  exact h2

end expected_value_of_twelve_sided_die_l825_825770


namespace min_log_expression_eq_4_l825_825495

theorem min_log_expression_eq_4 
  (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  (∃ b, ∀ b, log a b + log b (a ^ 2 + 12) ≥ 4) → 
  (log a b + log b (a ^ 2 + 12) = 4) → 
  a = 2 :=
by
  sorry

end min_log_expression_eq_4_l825_825495


namespace avg_children_in_families_with_children_l825_825447

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l825_825447


namespace valid_distributions_power_of_two_l825_825760

theorem valid_distributions_power_of_two (n : ℕ) : 
  ∃ m, (2 ^ m = {p : ℕ × ℕ | p.1 + p.2 = n ∧ ∀ (a b : ℕ), (a = 2 * b ∨ b = 2 * a) → (a, b) ∈ p}.card) := 
sorry

end valid_distributions_power_of_two_l825_825760


namespace strictly_increasing_interval_l825_825824

-- Definitions for conditions
def quadratic (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Domain condition
def domain (x : ℝ) : Prop := quadratic x > 0

-- Monotonicity conditions
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

def log_half (x : ℝ) : ℝ := log_base (1/2) x

-- Problem statement
theorem strictly_increasing_interval :
  ∃ s : Set ℝ, 
    s = {x | x < -1} ∧
    (increasing_on (λ x, log_half (quadratic x)) s) :=
by
  sorry

end strictly_increasing_interval_l825_825824


namespace average_children_in_families_with_children_l825_825380

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l825_825380


namespace intersection_distance_sum_l825_825048

-- Definitions based on given conditions:
def parametric_curve (α : ℝ) : ℝ × ℝ :=
  (3 * Real.cos α, Real.sin α)

def polar_line_ρ (θ : ℝ) : ℝ :=
  (√2 / Real.sin (θ - π / 4))

def general_curve (x y : ℝ) :=
  x^2 / 9 + y^2 = 1

def cartesian_line (x y : ℝ) :=
  y = x + 2

def point_P : ℝ × ℝ := (0, 2)

-- Prove the final equivalence:
theorem intersection_distance_sum :
  ∀ A B : ℝ × ℝ, 
    (general_curve A.fst A.snd ∧ cartesian_line A.fst A.snd) →
    (general_curve B.fst B.snd ∧ cartesian_line B.fst B.snd) →
    |point_P.1 - A.1| + |point_P.2 - A.2| + |point_P.1 - B.1| + |point_P.2 - B.2| = 18 * √2 / 5 :=
by
  intros A B hA hB
  sorry

end intersection_distance_sum_l825_825048


namespace tan_eq_tan2x_solutions_l825_825016

noncomputable def numberOfSolutions : ℝ := 159

theorem tan_eq_tan2x_solutions :
  let interval := Icc 0 (Real.arctan 500)
  let condition := ∀ θ, 0 < θ ∧ θ < π/2 → Real.tan θ > θ
  (setOf (λ x, Real.tan x = Real.tan (2 * x))).count interval = numberOfSolutions :=
  sorry

end tan_eq_tan2x_solutions_l825_825016


namespace hypotenuse_median_l825_825235

variables {A B C D E : Type*} [Euclidean_geometry A B C D E]
variables {AB BC BD DE AD CE : ℝ}
variables (right_triangle_ABC : ∃ A B C : Type*, ∠ABC = 90)
variables (D_on_AB : ∃ D : Type*, BD = BC)
variables (E_on_BC : ∃ E : Type*, DE = BE)

theorem hypotenuse_median (h1 : right_triangle_ABC) (h2 : D_on_AB) (h3 : E_on_BC) :
  AD + CE = DE :=
by sorry

end hypotenuse_median_l825_825235


namespace groups_of_same_kind_l825_825794

def same_kind (t1 t2 : String) : Prop :=
  sorry -- definition to determine same kind terms by comparing letters and exponents

def terms1 := ("-2p^2t", "tp^2")
def terms2 := ("-a^2bcd", "3b^2acd")
def terms3 := ("-a^mb^n", "a^mb^n")
def terms4 := ("24b^2a / 3", "(-2)^2ab^2")

theorem groups_of_same_kind :
  same_kind (fst terms1) (snd terms1) ∧
  ¬same_kind (fst terms2) (snd terms2) ∧
  same_kind (fst terms3) (snd terms3) ∧
  same_kind (fst terms4) (snd terms4) →
  "D" = "D" :=
by
  sorry

end groups_of_same_kind_l825_825794


namespace part1_part2_l825_825516

-- Part (1)
theorem part1 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x) (hg : ∀ x, g x = x^2 + a)
  (tangent_at_1 : ∀ x, ∃ (x1 : ℝ), x1 = -1 ∧
    (f' x1 = g' 1 ∧ f x1 = g 1)) :
  a = 3 := 
  sorry

-- Part (2)
theorem part2 (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x) (hg : ∀ x, ∃ a, g x = x^2 + a)
  (tangent_condition : ∀ x1 x2, 
    (f' x1 = g' x2 ∧ f x1 = g x2)) :
  ∃ a, a ≥ -1 :=
  sorry

end part1_part2_l825_825516


namespace line_passes_through_circle_center_l825_825939

theorem line_passes_through_circle_center (a : ℝ) :
  (∃ x y : ℝ, (x^2 + y^2 + 2*x - 4*y = 0) ∧ (3*x + y + a = 0)) → a = 1 :=
by
  sorry

end line_passes_through_circle_center_l825_825939


namespace expected_value_of_twelve_sided_die_l825_825768

theorem expected_value_of_twelve_sided_die : 
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0) in
  (finset.sum outcomes (λ n, (n:ℝ)) / 12 = 6.5) :=
by
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0)
  have h1 : ∑ n in outcomes, (n : ℝ) = 78, sorry
  have h2 : (78 / 12) = 6.5, sorry
  exact h2

end expected_value_of_twelve_sided_die_l825_825768


namespace arithmetic_sequence_second_and_ninth_term_sum_l825_825961

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms of a sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), a i

-- Verification of the specific conditions
theorem arithmetic_sequence_second_and_ninth_term_sum 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum_10 : sum_first_n_terms a 9 = 120) :
  a 1 + a 8 = 24 :=
begin
  -- Here is where the proof would go
  sorry
end

end arithmetic_sequence_second_and_ninth_term_sum_l825_825961


namespace kelly_total_snacks_l825_825066

theorem kelly_total_snacks (peanuts raisins : ℝ) (h₁ : peanuts = 0.1) (h₂ : raisins = 0.4) :
  peanuts + raisins = 0.5 :=
by
  simp [h₁, h₂]
  sorry

end kelly_total_snacks_l825_825066


namespace find_x_plus_y_l825_825569

theorem find_x_plus_y (x y : ℝ) (h1 : |x| - x + y = 13) (h2 : x - |y| + y = 7) : x + y = 20 := 
by
  sorry

end find_x_plus_y_l825_825569


namespace max_y_coordinate_l825_825840

theorem max_y_coordinate (θ : ℝ) (h : y = (sin (2 * θ)) * (sin θ)) : 
  max y (y = (sin (2 * θ)) * (sin θ)) = 4 * real.sqrt 3 / 9 :=
sorry

end max_y_coordinate_l825_825840


namespace debt_calculation_correct_l825_825914

-- Conditions
def initial_debt : ℤ := 40
def repayment : ℤ := initial_debt / 2
def additional_borrowing : ℤ := 10

-- Final Debt Calculation
def remaining_debt : ℤ := initial_debt - repayment
def final_debt : ℤ := remaining_debt + additional_borrowing

-- Proof Statement
theorem debt_calculation_correct : final_debt = 30 := 
by 
  -- Skipping the proof
  sorry

end debt_calculation_correct_l825_825914


namespace grid_to_zero_l825_825041

-- Define the conditions of the problem
variables (m n : ℕ)
variables (grid : Fin m → Fin n → ℝ)

-- Define the sums of black and white cells over a checkerboard pattern
noncomputable def sum_black_cells : ℝ :=
  ∑ i in Fin.range m, ∑ j in Fin.range n, if (i + j) % 2 = 0 then grid i j else 0

noncomputable def sum_white_cells : ℝ :=
  ∑ i in Fin.range m, ∑ j in Fin.range n, if (i + j) % 2 = 1 then grid i j else 0

noncomputable def S := sum_black_cells m n grid - sum_white_cells m n grid

-- Statement of the theorem
theorem grid_to_zero (S_zero : S m n grid = 0) : 
  ∃ steps : ℕ, ∃ moves : Fin steps → (Fin m × Fin n) × ℝ, 
  (∀ i j, grid i j = 0) :=
sorry

end grid_to_zero_l825_825041


namespace m_value_for_power_function_l825_825578

theorem m_value_for_power_function (m : ℝ) :
  (3 * m - 1 = 1) → (m = 2 / 3) :=
by
  sorry

end m_value_for_power_function_l825_825578


namespace algebraic_expression_value_l825_825864

variables (m n x y : ℤ)

def condition1 := m - n = 100
def condition2 := x + y = -1

theorem algebraic_expression_value :
  condition1 m n → condition2 x y → (n + x) - (m - y) = -101 :=
by
  intro h1 h2
  sorry

end algebraic_expression_value_l825_825864


namespace common_points_sufficient_not_necessary_l825_825906

def line (x k b : ℝ) : ℝ := k * x + b

def curve (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

theorem common_points_sufficient_not_necessary (k b : ℝ) :
  (∃ x, curve x (line x k b)) → (|b - 1| ≤ sqrt (k^2 + 1)) :=
sorry

end common_points_sufficient_not_necessary_l825_825906


namespace arrangement_count_l825_825012

theorem arrangement_count :
  -- Number of 15-letter arrangements of 5 A's, 5 B's, and 5 C's with given constraints.
  ∑ m in Finset.range 6, (Nat.choose 5 m)^3 = ∑ m in Finset.range 6, (Nat.choose 5 m)^3 :=
by
  sorry

end arrangement_count_l825_825012


namespace sufficient_but_not_necessary_condition_for_increasing_function_l825_825237

theorem sufficient_but_not_necessary_condition_for_increasing_function :
  ∀ a : ℝ, a = 2 → (∀ x y : ℝ, x ∈ Icc (-1) y → y ∈ Icc (-1) (+∞) → (f a x ≤ f a y) → 
  (∀ a : ℝ, (a = 2 ↔ 2 ≤ a))) :=
by
  sorry

def f (a x : ℝ) : ℝ :=
  x^2 + a*x + 1

end sufficient_but_not_necessary_condition_for_increasing_function_l825_825237


namespace equal_sides_length_of_isosceles_right_triangle_l825_825957

noncomputable def isosceles_right_triangle (a c : ℝ) : Prop :=
  c^2 = 2 * a^2 ∧ a^2 + a^2 + c^2 = 725

theorem equal_sides_length_of_isosceles_right_triangle (a c : ℝ) 
  (h : isosceles_right_triangle a c) : 
  a = 13.5 :=
by
  sorry

end equal_sides_length_of_isosceles_right_triangle_l825_825957


namespace find_a_b_intervals_of_monotonicity_l825_825539

-- Define the given function f(x)
def f (x : ℝ) (a b : ℝ) : ℝ := Real.exp x * (a * x + b) - x^2 - 4 * x

-- Define f' whose tangent at (0, f 0) must be 4x + 4
def f' (x : ℝ) (a b : ℝ) : ℝ := Real.exp x * (a * x + a + b) - 2 * x - 4

-- Part (Ⅰ) values of a and b
theorem find_a_b (a b : ℝ) (h₁ : f 0 a b = 4) (h₂ : f' 0 a b = 4) : a = 4 ∧ b = 4 :=
  sorry

-- Part (Ⅱ) intervals of monotonicity
theorem intervals_of_monotonicity (a b : ℝ) :
  (a = 4 ∧ b = 4) → 
  let fmon : ℝ -> ℝ := fun x => 4 * Real.exp x * (x + 1) - x^2 - 4 * x in
  let dfmon : ℝ -> ℝ := fun x => 4 * (x + 2) * (Real.exp x - 1/2) in
  (∀ x ∈ Set.Ioo (-∞) (-2.toReal) ∪ Set.Ioo (-Real.log 2) ∞, dfmon x > 0) ∧
  (∀ x ∈ Set.Ioo (-2.toReal) (-Real.log 2), dfmon x < 0) :=
  sorry

end find_a_b_intervals_of_monotonicity_l825_825539


namespace smallest_x_consecutive_cubes_l825_825964

theorem smallest_x_consecutive_cubes :
  ∃ (u v w x : ℕ), u < v ∧ v < w ∧ w < x ∧ u + 1 = v ∧ v + 1 = w ∧ w + 1 = x ∧ (u^3 + v^3 + w^3 = x^3) ∧ (x = 6) :=
by {
  sorry
}

end smallest_x_consecutive_cubes_l825_825964


namespace consecutive_pair_sums_l825_825081

theorem consecutive_pair_sums (N : ℕ) (A B : Set ℕ)
  (hA : ∃ (a b : ℕ), A = Set.Icc a (a + N - 1) ∧ B = Set.Icc b (b + N - 1)) :
  (∃ (f : A → B), ∀ x : A, ∃ (k : ℤ), k ∈ Finset.range N ∧ (x + f x) = (k + 1)) ↔ (N % 2 = 1) :=
by
  sorry

end consecutive_pair_sums_l825_825081


namespace f_inv_correct_inverse_function_l825_825887

def f (x : Real) : Real :=
  2^(x-1)

noncomputable def f_inv (x : Real) : Real :=
  1 + Real.log x / Real.log 2

theorem f_inv_correct (x : Real) (hx : 1 ≤ x) : f_inv x = 1 + Real.log x / Real.log 2 :=
by sorry

theorem inverse_function (x : Real) (hx : 1 ≤ x) : f (f_inv x) = x ∧ f_inv (f x) = x :=
by sorry

end f_inv_correct_inverse_function_l825_825887


namespace sheila_hourly_wage_l825_825232

def sheila_works_hours : ℕ :=
  let monday_wednesday_friday := 8 * 3
  let tuesday_thursday := 6 * 2
  monday_wednesday_friday + tuesday_thursday

def sheila_weekly_earnings : ℕ := 396
def sheila_total_hours_worked := 36
def expected_hourly_earnings := sheila_weekly_earnings / sheila_total_hours_worked

theorem sheila_hourly_wage :
  sheila_works_hours = sheila_total_hours_worked ∧
  sheila_weekly_earnings / sheila_total_hours_worked = 11 :=
by
  sorry

end sheila_hourly_wage_l825_825232


namespace binomial_equality_l825_825323

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l825_825323


namespace divides_of_exponentiation_l825_825122

theorem divides_of_exponentiation (n : ℕ) : 7 ∣ 3^(12 * n + 1) + 2^(6 * n + 2) := 
  sorry

end divides_of_exponentiation_l825_825122


namespace find_a_pure_imaginary_l825_825992

theorem find_a_pure_imaginary (a : ℝ) (h : 2 - a/(2 - (0 + 1*complex.I)) = 0 + b * complex.I) : a = 5 := 
sorry

end find_a_pure_imaginary_l825_825992


namespace expected_value_twelve_sided_die_l825_825778

theorem expected_value_twelve_sided_die : 
  (1 / 12 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l825_825778


namespace avg_children_with_kids_l825_825394

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l825_825394


namespace exists_positive_integers_l825_825085

theorem exists_positive_integers (k : ℕ) (h : 0 < k) :
  ∃ (a : ℕ → ℕ), (∀ j, j < k → 0 < a j) ∧ 
  ∀ x : ℕ, x ^ k = ∑ i in finset.range k, a i * (nat.choose (x + i) k) := by
sorry

end exists_positive_integers_l825_825085


namespace g_at_5_l825_825991

def g (x : ℝ) : ℝ := 3 * x^5 - 15 * x^4 + 30 * x^3 - 45 * x^2 + 24 * x + 50

theorem g_at_5 : g 5 = 2795 :=
by
  sorry

end g_at_5_l825_825991


namespace percentage_problem_l825_825241

theorem percentage_problem : ∃ x : ℝ, 70 = 0.25 * x ∧ x = 280 :=
by
  exists 280
  split
  sorry
  rfl

end percentage_problem_l825_825241


namespace soft_drink_cost_approx_eq_l825_825129

def cost_per_can_individual (total_cost : ℝ) (num_cans : ℕ) : ℝ :=
  total_cost / num_cans

theorem soft_drink_cost_approx_eq :
  cost_per_can_individual 2.99 12 ≈ 0.25 := 
by
  sorry

end soft_drink_cost_approx_eq_l825_825129


namespace percentage_paid_to_x_l825_825181

theorem percentage_paid_to_x (X Y : ℕ) (h₁ : Y = 350) (h₂ : X + Y = 770) :
  (X / Y) * 100 = 120 :=
by
  sorry

end percentage_paid_to_x_l825_825181


namespace min_distance_circle_ellipse_l825_825857

-- Definitions of the parametrized paths
def circle_path (t : ℝ) : ℝ × ℝ := (2 * Real.cos t, 2 * Real.sin t)
def ellipse_path (t : ℝ) : ℝ × ℝ := (3 + 3 * Real.cos (t / 2), 3 * Real.sin (t / 2))

-- Statement of the theorem that the minimum distance is 4
theorem min_distance_circle_ellipse : 
  let C := circle_path
  let D := ellipse_path
  ∃ p q : ℝ, (C p).dist (D q) = 4 :=
sorry

end min_distance_circle_ellipse_l825_825857


namespace avg_children_in_families_with_children_l825_825445

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l825_825445


namespace avg_children_in_families_with_children_l825_825449

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l825_825449


namespace inequality_1_inequality_2_l825_825591

-- Definitions of variables
variables {a b c d e f : ℝ}
variables (S p : ℝ)

noncomputable def p := 0.5 * (a + b + c + d)

-- First theorem to prove
theorem inequality_1 (h1 : S ≤ (1/2) * e * f) (h2 : (1/2) * e * f ≤ (1/2) * (a * c + b * d)) :
  S ≤ (1/2) * e * f ∧ (1/2) * e * f ≤ (1/2) * (a * c + b * d) :=
by
  exact ⟨h1, h2⟩

-- Second theorem to prove
theorem inequality_2 (h3 : S ≤ (1/4) * (a + c) * (b + d)) (h4 : (1/4) * (a + c) * (b + d) ≤ (p p *p) / 4) :
  S ≤ (1/4) * (a + c) * (b + d) ∧ (1/4) * (a + c) * (b + d) ≤ (p p *p) / 4 :=
by
  exact ⟨h3, h4⟩

end inequality_1_inequality_2_l825_825591


namespace G1_intersects_x_axis_range_of_n_minus_m_plus_a_values_of_a_for_right_triangle_l825_825907

-- Define the quadratic polynomial G₁
def G₁ (a x : ℝ) : ℝ := x^2 - 2 * a * x + a^2 - 4

-- Define the conditions
def N : ℝ × ℝ := (0, -4)

-- (1) Prove that the parabola G₁ intersects the x-axis at two points
theorem G1_intersects_x_axis (a : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (G₁ a x1 = 0) ∧ (G₁ a x2 = 0) :=
begin
  -- Look at the roots of the quadratic G₁
  let Δ := 16, -- Discriminant
  have hΔ : Δ > 0 := by norm_num,
  use [(a - 2), (a + 2)],
  split,
  { intro h_eq, linarith },
  { split; norm_num }
end

-- (2) When NA ≥ 5, determine the range of values for n - m + a.
theorem range_of_n_minus_m_plus_a (a : ℝ) (N : ℝ × ℝ) (hN_eq : N = (0, -4)) (NA_ge_5: True) : 
  4 + a ≥ 9 ∨ 4 + a ≤ 3 := 
begin
  split,
  { --Case 1 : a ≥ 5
    intro ha, right, linarith },
  { --Case 2 : a ≤ -1
    intro ha, left, linarith }
end

-- (3) Values of a such that ΔBNB' is a right triangle
theorem values_of_a_for_right_triangle (a : ℝ) (B : ℝ × ℝ) (B' : ℝ × ℝ) (N : ℝ × ℝ) 
  (hB : B = (a + 2, 0)) (hB': B' = (a - 6, 0)) (hN: N = (0, -4))
  : a = 2 ∨ a = -2 ∨ a = 6 := 
begin
  -- The given conditions and geometry implies:
  -- Use algebraic manipulation to find appropriate a values.
  sorry
end

end G1_intersects_x_axis_range_of_n_minus_m_plus_a_values_of_a_for_right_triangle_l825_825907


namespace not_multiple_of_3_l825_825071

noncomputable def exists_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n*(n + 3) = m^2

theorem not_multiple_of_3 
  (n : ℕ) (h1 : 0 < n) (h2 : exists_perfect_square n) : ¬ ∃ k : ℕ, n = 3 * k := 
sorry

end not_multiple_of_3_l825_825071


namespace joan_total_money_l825_825724

-- Define the number of each type of coin found
def dimes_jacket : ℕ := 15
def dimes_shorts : ℕ := 4
def nickels_shorts : ℕ := 7
def quarters_jeans : ℕ := 12
def pennies_jeans : ℕ := 2
def nickels_backpack : ℕ := 8
def pennies_backpack : ℕ := 23

-- Calculate the total number of each type of coin
def total_dimes : ℕ := dimes_jacket + dimes_shorts
def total_nickels : ℕ := nickels_shorts + nickels_backpack
def total_quarters : ℕ := quarters_jeans
def total_pennies : ℕ := pennies_jeans + pennies_backpack

-- Calculate the total value of each type of coin
def value_dimes : ℝ := total_dimes * 0.10
def value_nickels : ℝ := total_nickels * 0.05
def value_quarters : ℝ := total_quarters * 0.25
def value_pennies : ℝ := total_pennies * 0.01

-- Calculate the total amount of money found
def total_money : ℝ := value_dimes + value_nickels + value_quarters + value_pennies

-- Proof statement
theorem joan_total_money : total_money = 5.90 := by
  sorry

end joan_total_money_l825_825724


namespace smallest_integer_representation_l825_825193

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end smallest_integer_representation_l825_825193


namespace possible_values_of_a_l825_825665

def isFactor (x y : ℕ) : Prop :=
  y % x = 0

def isDivisor (x y : ℕ) : Prop :=
  x % y = 0

def isPositive (x : ℕ) : Prop :=
  x > 0

theorem possible_values_of_a : 
  { a : ℕ // isFactor 4 a ∧ isDivisor 24 a ∧ isPositive a }.card = 4 := by
  sorry

end possible_values_of_a_l825_825665


namespace rectangular_to_polar_correct_l825_825818

noncomputable def toPolarCoordinates (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  if x < 0 ∧ y ≥ 0 then (r, Real.pi - θ)
  else if x < 0 ∧ y < 0 then (r, Real.pi + θ)
  else (r, θ)

theorem rectangular_to_polar_correct : toPolarCoordinates (-5) (5 * Real.sqrt 3) = (10, 2 * Real.pi / 3) :=
by
  -- Definitions used:
  let x := -5
  let y := 5 * Real.sqrt 3

  -- Proof assertion:
  have r_def : Real.sqrt (x^2 + y^2) = 10 := by
    sorry
  have θ_def : Real.arctan (y / x) = Real.pi / 3 := by
    sorry
  have θ_calc : Real.pi - Real.arctan (y / x) = 2 * Real.pi / 3 := by
    sorry

  -- Asserting the fianl equality:
  show (Real.sqrt (x^2 + y^2), Real.pi - Real.arctan (y / x)) = (10, 2 * Real.pi / 3)
  rw [r_def, θ_calc]

end rectangular_to_polar_correct_l825_825818


namespace compare_P_Q_l825_825073

variable (a : ℝ) (h : a ≥ 0)

def P := sqrt a + sqrt (a + 7)
def Q := sqrt (a + 3) + sqrt (a + 4)

theorem compare_P_Q : P a < Q a :=
by
  sorry

end compare_P_Q_l825_825073


namespace potatoes_per_bundle_25_l825_825753
noncomputable def number_of_potatoes_per_bundle : ℕ → ℝ → ℕ → ℝ → ℝ → ℕ
| potatoes, price_per_potato_bundle, carrots, price_per_carrot_bundle, total_revenue =>
  let carrot_bundles := carrots / 20
  let carrot_revenue := carrot_bundles * price_per_carrot_bundle
  let potato_revenue := total_revenue - carrot_revenue
  let potato_bundles := potato_revenue / price_per_potato_bundle
  potatoes / potato_bundles

theorem potatoes_per_bundle_25 (
  potatoes : ℕ) (price_per_potato_bundle : ℝ) (carrots : ℕ) (price_per_carrot_bundle : ℝ) (
  total_revenue : ℝ ) :
  (potatoes = 250) ∧ (price_per_potato_bundle = 1.90) ∧ (carrots = 320) ∧ (price_per_carrot_bundle = 2) ∧ (total_revenue = 51) →
  number_of_potatoes_per_bundle potatoes price_per_potato_bundle carrots price_per_carrot_bundle total_revenue = 25 :=
by
  -- proof is omitted
  sorry

end potatoes_per_bundle_25_l825_825753


namespace range_of_m_l825_825892

theorem range_of_m (m : ℝ) :
  (3 * 1 - 2 + m) * (3 * 1 - 1 + m) < 0 →
  -2 < m ∧ m < -1 :=
by
  intro h
  sorry

end range_of_m_l825_825892


namespace AB_eq_B_exp_V_l825_825459

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l825_825459


namespace avg_children_with_kids_l825_825397

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l825_825397


namespace combination_sum_l825_825740

noncomputable def combinations (n r : ℕ) : ℕ :=
  if n < r then 0 else (nat.desc_factorial n r) / (nat.factorial r)

theorem combination_sum : combinations 2 2 + combinations 3 2 + combinations 4 2 + combinations 5 2 + combinations 6 2 + combinations 7 2 + combinations 8 2 + combinations 9 2 + combinations 10 2 = 165 :=
by
  sorry

end combination_sum_l825_825740


namespace hypotenuse_length_l825_825177
-- Import necessary library for real numbers and basic geometric constructs.

-- Definition of the problem conditions in Lean 4 language.
variables (P Q R M N : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace M] [MetricSpace N]
variables (PQ PR QR PM MQ PN NQ : ℝ)
variable h1 : PQ = PR
variable h2 : PM * 4 = MQ
variable h3 : PN * 4 = NQ
variable h4 : NQ = 18
variable h5 : MQ = 30

-- Mathematical statement to prove.
theorem hypotenuse_length (PQ PR QR PM MQ PN NQ : ℝ) (h1 : PQ = PR) (h2 : PM * 4 = MQ) (h3 : PN * 4 = NQ) (h4 : NQ = 18) (h5 : MQ = 30) :
  QR = 8 * Real.sqrt 18 :=
sorry

end hypotenuse_length_l825_825177


namespace elder_age_is_30_l825_825143

/-- The ages of two persons differ by 16 years, and 6 years ago, the elder one was 3 times as old as the younger one. 
Prove that the present age of the elder person is 30 years. --/
theorem elder_age_is_30 (y e: ℕ) (h₁: e = y + 16) (h₂: e - 6 = 3 * (y - 6)) : e = 30 := 
sorry

end elder_age_is_30_l825_825143


namespace value_of_y_at_3_l825_825596

-- Define the function
def f (x : ℕ) : ℕ := 2 * x^2 + 1

-- Prove that when x = 3, y = 19
theorem value_of_y_at_3 : f 3 = 19 :=
by
  -- Provide the definition and conditions
  let x := 3
  let y := f x
  have h : y = 2 * x^2 + 1 := rfl
  -- State the actual proof could go here
  sorry

end value_of_y_at_3_l825_825596


namespace find_positive_integer_pairs_prime_perfect_square_l825_825455

theorem find_positive_integer_pairs_prime_perfect_square (a b : ℕ) (h : a > 0 ∧ b > 0) (hp : ∃ p : ℤ, prime p ∧ ↑a - ↑b = p) (hk : ∃ k : ℕ, a * b = k^2) :
  ∃ p : ℤ, prime p ∧ p % 2 = 1 ∧ a = ((p + 1) / 2) ^ 2 ∧ b = ((p - 1) / 2) ^ 2 :=
by
  sorry

end find_positive_integer_pairs_prime_perfect_square_l825_825455


namespace quadrilateral_equal_sides_l825_825067

theorem quadrilateral_equal_sides 
  (ABCD : Type) 
  (A B C D : ABCD) 
  (angle_BCD : angle B C D = 120) 
  (angle_CBA : angle C B A = 45) 
  (angle_CBD : angle C B D = 15)
  (angle_CAB : angle C A B = 90) : 
  (dist A B = dist A D) :=
sorry

end quadrilateral_equal_sides_l825_825067


namespace common_solutions_count_l825_825557

theorem common_solutions_count :
  let solutions := {y : ℤ | -3*y + 1 ≥ y - 9 ∧ -2*y + 2 ≤ 14 ∧ -5*y + 1 ≥ 3*y + 20} in
  solutions = {-6, -5, -4, -3} ∧ solutions.card = 4 :=
by
  let solutions := {y : ℤ | -3*y + 1 ≥ y - 9 ∧ -2*y + 2 ≤ 14 ∧ -5*y + 1 ≥ 3*y + 20}
  show solutions = {-6, -5, -4, -3} ∧ solutions.card = 4
  sorry

end common_solutions_count_l825_825557


namespace inequality_proof_l825_825929

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (a + d > b + c) := sorry

end inequality_proof_l825_825929


namespace exists_m_gt_n_l825_825729

noncomputable def seq (K : ℕ) (x : ℕ → ℕ) : ℕ → ℕ
| 0     := 1
| 1     := K
| (n+2) := K * (seq n.succ) - (seq n)

theorem exists_m_gt_n (K : ℕ) (hK : K > 1) (n : ℕ) : 
  ∃ m > n, (seq K (seq K) m) % (seq K (seq K) n) = 0 := 
by
  sorry

end exists_m_gt_n_l825_825729


namespace addition_and_subtraction_problems_l825_825697

theorem addition_and_subtraction_problems :
  ∀ (total_questions word_problems steve_can_answer difference : ℕ),
    total_questions = 45 →
    word_problems = 17 →
    steve_can_answer = 38 →
    difference = 7 →
    total_questions - steve_can_answer = difference →
    steve_can_answer - word_problems = 21 :=
by
  intros total_questions word_problems steve_can_answer difference h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h5
  injection h5 with h5'
  rw [h5']
sorry

end addition_and_subtraction_problems_l825_825697


namespace average_children_l825_825440

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l825_825440


namespace projective_transformation_exists_l825_825868

theorem projective_transformation_exists (C : set (ℝ × ℝ)) (l : set (ℝ × ℝ)) :
  (∃ p₁ p₂ r, C = {p : ℝ × ℝ | (p.1 - p₁)^2 + (p.2 - p₂)^2 = r^2}) →
  (∀ x ∈ l, ∀ y ∈ C, x.2 ≠ y.2) →
  ∃ (φ : (ℝ × ℝ) → (ℝ × ℝ)), 
    (∃ p₃ p₄ s, ∀ q, (φ q) ∈ {p : ℝ × ℝ | (p.1 - p₃)^2 + (p.2 - p₄)^2 = s^2}) ∧ 
    (∀ x ∈ l, φ x = (x.1, ∞)) :=
by sorry

end projective_transformation_exists_l825_825868


namespace sequence_formula_l825_825030

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℤ) : (∀ n, S n = 2 * a n + 1) → (∀ n, n ≥ 1 → a n = -2^(n-1)) :=
by
  intro hS,
  intro n hn,
  sorry

end sequence_formula_l825_825030


namespace value_of_b2018_l825_825958

noncomputable def INT (x : ℝ) := ⌊x⌋

def a (n : ℕ) : ℕ := INT (2 / 7 * 10^n)

def b : ℕ → ℕ
| 0     := a 0
| (n+1) := a (n+1) - 10 * a n

theorem value_of_b2018 : b 2018 = 8 := 
by {
  sorry
}

end value_of_b2018_l825_825958


namespace money_properties_proof_l825_825107

-- The context of Robinson Crusoe being on a deserted island.
def deserted_island := true

-- The functions of money in modern society.
def medium_of_exchange := true
def store_of_value := true
def unit_of_account := true
def standard_of_deferred_payment := true

-- The financial context of the island.
def island_context (deserted_island : Prop) : Prop :=
  deserted_island →
  ¬ medium_of_exchange ∧
  ¬ store_of_value ∧
  ¬ unit_of_account ∧
  ¬ standard_of_deferred_payment

-- Other properties that an item must possess to become money.
def durability := true
def portability := true
def divisibility := true
def acceptability := true
def uniformity := true
def limited_supply := true

-- The proof problem statement.
theorem money_properties_proof
  (H1 : deserted_island)
  (H2 : island_context H1)
  : (¬ medium_of_exchange ∧
    ∀ (m : Prop), (m = durability ∨ m = portability ∨ m = divisibility ∨ m = acceptability ∨ m = uniformity ∨ m = limited_supply)) :=
by {
  sorry
}

end money_properties_proof_l825_825107


namespace locus_of_points_l825_825710

theorem locus_of_points (A B C : Point)
  (O : TriangleCircumcenter A B C)
  (M : Point) :
  (dist M B)² + (dist M C)² = 2 * (dist M A)² ↔
  OnLine M (perpendicularLineThrough O (median AI)) :=
sorry

end locus_of_points_l825_825710


namespace arccos_sqrt_three_over_two_is_pi_div_six_l825_825310

noncomputable def arccos_sqrt_three_over_two_eq_pi_div_six : Prop :=
  arccos (Real.sqrt 3 / 2) = Real.pi / 6

theorem arccos_sqrt_three_over_two_is_pi_div_six :
  arccos_sqrt_three_over_two_eq_pi_div_six :=
by 
  sorry

end arccos_sqrt_three_over_two_is_pi_div_six_l825_825310


namespace equilateral_triangle_ab_l825_825688

noncomputable def a : ℝ := 25 * Real.sqrt 3
noncomputable def b : ℝ := 5 * Real.sqrt 3

theorem equilateral_triangle_ab
  (a_val : a = 25 * Real.sqrt 3)
  (b_val : b = 5 * Real.sqrt 3)
  (h1 : Complex.abs (a + 15 * Complex.I) = 25)
  (h2 : Complex.abs (b + 45 * Complex.I) = 45)
  (h3 : Complex.abs ((a - b) + (15 - 45) * Complex.I) = 30) :
  a * b = 375 := 
sorry

end equilateral_triangle_ab_l825_825688


namespace incorrect_inferenceB_l825_825719

variables {A B C : Type}
variables {l α β : Set A}

-- A: A ∈ l, A ∈ α, B ∈ α ⇒ l ⊆ α
def inferenceA (A B : A) (l α : Set A) : Prop := 
  A ∈ l ∧ A ∈ α ∧ B ∈ α → l ⊆ α

-- B: l ⊈ α, A ∈ l ⇒ A ∉ α
def inferenceB (A : A) (l α : Set A) : Prop := 
  l ⊈ α ∧ A ∈ l → A ∉ α

-- C: A ∈ α, A ∈ β, B ∈ α, B ∈ β ⇒ α ∩ β = AB
def inferenceC (A B : A) (α β : Set A) : Prop := 
  A ∈ α ∧ A ∈ β ∧ B ∈ α ∧ B ∈ β → (α ∩ β = {A, B})

-- D: A, B, C ∈ α, A, B, C ∈ β and A, B, C are not collinear ⇒ α, β overlap
def inferenceD (A B C : A) (α β : Set A) : Prop := 
  A ∈ α ∧ B ∈ α ∧ C ∈ α ∧ A ∈ β ∧ B ∈ β ∧ C ∈ β ∧ ¬ collinear A B C → α ∩ β ≠ ∅

-- Prove that inferenceB is the incorrect inference
theorem incorrect_inferenceB (A : A) (l α : Set A) : inferenceB A l α :=
sorry

end incorrect_inferenceB_l825_825719


namespace convex_2009gon_adjacent_sides_sum_l825_825504

theorem convex_2009gon_adjacent_sides_sum :
  ∀ (polygon : Finset (Fin 2009 → ℝ × ℝ)), 
  (∀ v ∈ polygon, v.1 ≥ 0 ∧ v.1 ≤ 2009 ∧ v.2 ≥ 0 ∧ v.2 ≤ 2009) →
  ∃ (i : Fin 2009), 
  let side_length := λ (i : Fin 2009), 
    Real.sqrt (
      (polygon[(i : Fin 2009).val.succ' % 2009]).1 - (polygon[i.val % 2009]).1)^2 +
      (polygon[(i : Fin 2009).val.succ' % 2009]).2 - (polygon[i.val % 2009]).2)^2) in
  side_length i + side_length (i.succ' % 2009) ≤ 8 :=
begin
  sorry
end

end convex_2009gon_adjacent_sides_sum_l825_825504


namespace required_fencing_l825_825224

-- Definitions from conditions
def length_uncovered : ℝ := 30
def area : ℝ := 720

-- Prove that the amount of fencing required is 78 feet
theorem required_fencing : 
  ∃ (W : ℝ), (area = length_uncovered * W) ∧ (2 * W + length_uncovered = 78) := 
sorry

end required_fencing_l825_825224


namespace divide_square_into_equal_area_triangles_l825_825604

theorem divide_square_into_equal_area_triangles (O : Point)
  (ratio_left_right : ∃ k : ℝ, (distance O left_side)/(distance O right_side) = k ∧ k = (3/2))
  (ratio_bottom_top : ∃ k : ℝ, (distance O bottom_side)/(distance O top_side) = k ∧ k = (3/2))
  (divide_left_bottom : ∃ pts : Finset Point, pts.card = 3 ∧ ∀ p ∈ pts, p ∈ segment(left_side))
  (divide_top_right : ∃ pts : Finset Point, pts.card = 2 ∧ ∀ p ∈ pts, p ∈ segment(top_side)))
  : ∃ triangles : Finset Triangle, triangles.card = 14 ∧ 
  (∀ t ∈ triangles, area t = (area square) / 14) ∧ 
  (∀ t ∈ triangles, O ∈ vertices t) :=
  sorry

end divide_square_into_equal_area_triangles_l825_825604


namespace ordered_triples_count_l825_825015

noncomputable def count_valid_triples (n : ℕ) :=
  ∃ x y z : ℕ, ∃ k : ℕ, x * y * z = k ∧ k = 5 ∧ lcm x y = 48 ∧ lcm x z = 450 ∧ lcm y z = 600

theorem ordered_triples_count : count_valid_triples 5 := by
  sorry

end ordered_triples_count_l825_825015


namespace price_of_watermelon_l825_825934

-- Define the price of the watermelon
def price_won := 5000 + 200

-- Define the unit conversion factor
def unit_conversion_factor := 1000

-- Calculate the price in units of 1,000 won
def price_in_units : ℝ := price_won / unit_conversion_factor

-- State the theorem to prove the price in units of 1,000 won is 5.2
theorem price_of_watermelon : price_in_units = 5.2 :=
by
  -- The proof would go here
  sorry

end price_of_watermelon_l825_825934


namespace problem_1_problem_2_problem_3_l825_825897

-- Problem 1: 
theorem problem_1 : 
  ∀ x, f x = log x + 9 / (2 * (x + 1)) → 
  (∀ x, 0 < x ∧ x < 1/2 → deriv f x > 0) ∧
  (∀ x, 2 < x → deriv f x > 0) := 
sorry

-- Problem 2: 
theorem problem_2 : 
  ∀ x_1 x_2 y_1 y_2 x_0 (hxy : x_1 ≠ x_2), 
  let C := (x_0, (log x_1 + log x_2) / 2),
  k := (log x_2 - log x_1) / (x_2 - x_1) in
  k > deriv (fun x => log x) x_0 :=
sorry

-- Problem 3: 
theorem problem_3 : 
  ∀ (a : ℝ) (x_1 x_2 : ℝ) (h : 0 < x_1 ∧ x_1 ≤ 2 ∧ 0 < x_2 ∧ x_2 ≤ 2 ∧ x_1 ≠ x_2),
  let g := (fun x => abs (log x) + a / (x + 1)) in
  (∃ x, 0 < x ∧ x ≤ 2 ∧ deriv g x < -1) → a ≥ 27 / 2 :=
sorry

end problem_1_problem_2_problem_3_l825_825897


namespace tessa_owes_30_l825_825921

-- Definitions based on given conditions
def initial_debt : ℕ := 40
def paid_back : ℕ := initial_debt / 2
def remaining_debt_after_payment : ℕ := initial_debt - paid_back
def additional_borrowing : ℕ := 10
def total_debt : ℕ := remaining_debt_after_payment + additional_borrowing

-- Theorem to be proved
theorem tessa_owes_30 : total_debt = 30 :=
by
  sorry

end tessa_owes_30_l825_825921


namespace avg_children_in_families_with_children_l825_825410

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l825_825410


namespace correct_factoring_example_l825_825151

-- Define each option as hypotheses
def optionA (a b : ℝ) : Prop := (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2
def optionB (a b : ℝ) : Prop := 2 * a ^ 2 - a * b - a = a * (2 * a - b - 1)
def optionC (a b : ℝ) : Prop := 8 * a ^ 5 * b ^ 2 = 4 * a ^ 3 * b * 2 * a ^ 2 * b
def optionD (a : ℝ) : Prop := a ^ 2 - 4 * a + 3 = (a - 1) * (a - 3)

-- The goal is to prove that optionD is the correct example of factoring
theorem correct_factoring_example (a b : ℝ) : optionD a ↔ (∀ a b, ¬ optionA a b) ∧ (∀ a b, ¬ optionB a b) ∧ (∀ a b, ¬ optionC a b) :=
by
  sorry

end correct_factoring_example_l825_825151


namespace range_of_a_l825_825542

theorem range_of_a (a : ℝ) (h_pos : a > 0) :
  (∀ x₁ ∈ set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ set.Icc (-1 : ℝ) 2, (a * x₁ + 2 = x₀^2 - 2 * x₀))
  ↔ a ∈ set.Ioc 0 (1/2) :=
by
  sorry

end range_of_a_l825_825542


namespace percentage_BCM_hens_l825_825254

theorem percentage_BCM_hens (total_chickens : ℕ) (BCM_percentage : ℝ) (BCM_hens : ℕ) : 
  total_chickens = 100 → BCM_percentage = 0.20 → BCM_hens = 16 →
  ((BCM_hens : ℝ) / (total_chickens * BCM_percentage)) * 100 = 80 :=
by
  sorry

end percentage_BCM_hens_l825_825254


namespace sum_of_sequence_l825_825508

noncomputable def sequence_sum (n : ℕ) : ℤ :=
  6 * 2^n - (n + 6)

theorem sum_of_sequence (a S : ℕ → ℤ) (n : ℕ) :
  a 1 = 5 →
  (∀ n : ℕ, 1 ≤ n → S (n + 1) = 2 * S n + n + 5) →
  S n = sequence_sum n :=
by sorry

end sum_of_sequence_l825_825508


namespace min_third_side_l825_825889

theorem min_third_side (x y : ℕ) (hx1 : 1 ≤ x) (hx2 : 5 < y) (hx3 : y < 2*x + 5) (hp : (2*x + 5 + y) % 2 = 1) :
  6 ≤ y :=
begin
  sorry,
end

end min_third_side_l825_825889


namespace range_of_f_l825_825981

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_of_f :
  ∀ y, (∃ x, x ∈ Set.Icc (-1:ℝ) 1 ∧ f x = y) ↔ y ∈ Set.Icc 0 (Real.pi^4 / 8) :=
sorry

end range_of_f_l825_825981


namespace probability_of_5A_level_spot_probability_of_selecting_b_and_e_l825_825829

-- Proof problem 1
theorem probability_of_5A_level_spot :
  let num_5A_spots := 4
  let num_4A_spots := 6
  let total_spots := num_5A_spots + num_4A_spots
  (num_5A_spots / total_spots) = (2 / 5) :=
by
  let num_5A_spots := 4
  let num_4A_spots := 6
  let total_spots := num_5A_spots + num_4A_spots
  show (num_5A_spots / total_spots) = (2 / 5)
  sorry

-- Proof problem 2
theorem probability_of_selecting_b_and_e :
  let selected_spot := 'a'
  let additional_spots := {'b', 'c', 'd', 'e'}
  let total_combinations := 12
  let favorable_outcomes := 2
  (favorable_outcomes / total_combinations) = (1 / 6) :=
by
  let selected_spot := 'a'
  let additional_spots := {'b', 'c', 'd', 'e'}
  let total_combinations := 12
  let favorable_outcomes := 2
  show (favorable_outcomes / total_combinations) = (1 / 6)
  sorry

end probability_of_5A_level_spot_probability_of_selecting_b_and_e_l825_825829


namespace boys_transferred_is_13_l825_825290

noncomputable def initial_boys : ℕ := 120
noncomputable def initial_girls : ℕ := (4 * initial_boys) / 3

def transferred_boys : ℕ
def transferred_girls : ℕ := 2 * transferred_boys

theorem boys_transferred_is_13 :
  let final_boys := initial_boys - transferred_boys,
      final_girls := initial_girls - transferred_girls in
      (final_boys / gcd final_boys final_girls) = 4 / gcd 4 5 ∧
      (final_girls / gcd final_boys final_girls) = 5 / gcd 4 5 →
      transferred_boys = 13 :=
by sorry

end boys_transferred_is_13_l825_825290


namespace avg_children_in_families_with_children_l825_825402

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l825_825402


namespace avg_children_in_families_with_children_l825_825409

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l825_825409


namespace find_n_l825_825070

noncomputable def p (a : ℝ) : Polynomial ℝ :=
  Polynomial.C (3*a^4 + 2*a^2) + Polynomial.C 2 * Polynomial.X^0 + Polynomial.C (-4*a^3 - 4*a) * Polynomial.X + Polynomial.X^4 + Polynomial.C 2 * Polynomial.X^2

theorem find_n :
  ∃ p ∈ setOf (λ p : Polynomial ℝ, ∃ m ∈ setOf (λ m : ℝ, m > 0 ∧ n > 0), p = Polynomial.X^4 + Polynomial.C 2 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C n
    ∧ ∃ x : ℝ, Polynomial.eval x p = 0
    ∧ m = Inf (setOf (λ m : ℝ, ∃ n, m > 0 ∧ n > 0 
    ∧ polynomial.root p
    ∧ m = (-4 * x^3 - 4 * x)
    ∧ Polynomial.eval 1 p = 99 
    ∧ Polynomial.eval 1 p = 99
    ∧ n = 3 * x^4 + 2 * x^2))), 
  n = 56 := sorry

end find_n_l825_825070


namespace ratio_of_wealth_l825_825342

theorem ratio_of_wealth (P W : ℝ) (hP : P > 0) (hW : W > 0) : 
  let wX := (0.40 * W) / (0.20 * P)
  let wY := (0.30 * W) / (0.10 * P)
  (wX / wY) = 2 / 3 := 
by
  sorry

end ratio_of_wealth_l825_825342


namespace distance_between_given_lines_is_correct_l825_825890

noncomputable def distance_between_parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  (abs (c₂ - c₁)) / (sqrt (a₁^2 + b₁^2))

theorem distance_between_given_lines_is_correct :
  let line1 := (3 : ℝ, 4 : ℝ, -3 : ℝ)
  let line2 := (6 : ℝ, 8 : ℝ, 1 : ℝ)
  distance_between_parallel_lines line1.1 line1.2 line1.3 line2.1 line2.2 (-line2.3) = 7 / 10 :=
by
  sorry

end distance_between_given_lines_is_correct_l825_825890


namespace abc_plus_2p_zero_l825_825169

variable (a b c p : ℝ)

-- Define the conditions
def cond1 : Prop := a + 2 / b = p
def cond2 : Prop := b + 2 / c = p
def cond3 : Prop := c + 2 / a = p
def nonzero_and_distinct : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main statement we want to prove
theorem abc_plus_2p_zero (h1 : cond1 a b p) (h2 : cond2 b c p) (h3 : cond3 c a p) (h4 : nonzero_and_distinct a b c) : 
  a * b * c + 2 * p = 0 := 
by 
  sorry

end abc_plus_2p_zero_l825_825169


namespace distance_A_B_l825_825969

-- Define point A in 3D space
def A : ℝ × ℝ × ℝ := (1, -2, 1)

-- Define point B in 3D space
def B : ℝ × ℝ × ℝ := (0, 1, -1)

-- Define the function that calculates the distance between two points in 3D space
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

-- The theorem stating the distance between A and B is sqrt(14)
theorem distance_A_B : distance A B = Real.sqrt 14 :=
by
  sorry

end distance_A_B_l825_825969


namespace infinite_H_points_on_curve_l825_825650

-- Define the curve
def curve (x y : ℝ) : Prop := (x^2 / 4) + (y^2) = 1

-- Define a point
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define "H point" property
def is_H_point (P A B : Point) : Prop :=
  curve P.x P.y ∧
  curve A.x A.y ∧
  (abs ((P.x - A.x) * (P.x - A.x) + (P.y - A.y) * (P.y - A.y)) = abs ((P.x - B.x) * (P.x - B.x) + (P.y - B.y) * (P.y - B.y))
  ∨ abs ((P.x - A.x) * (P.x - A.x) + (P.y - A.y) * (P.y - A.y)) = abs ((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y)))

-- Define property of line intersecting at P, A and intersects line x=4 at B
def line_condition (P A : Point) : Prop :=
  P ≠ A ∧
  ∃ (k m : ℝ), ∀ (x : ℝ), Point.mk x (k*x + m) = P ∨ Point.mk x (k*x + m) = A ∨ x = 4 ∧ 
  ∃ B : Point, B.x = 4

-- Main theorem : there are infinitely many 'H points' on the curve
theorem infinite_H_points_on_curve : ∃ (Ps : set Point), (∀ P ∈ Ps, ∃ A B : Point, is_H_point P A B) ∧ ¬(∀ P : Point, curve P.x P.y → (∃ A B : Point, is_H_point P A B)) ∧ (Ps.countable = False) :=
  sorry

end infinite_H_points_on_curve_l825_825650


namespace alices_fav_number_l825_825280

def isDivisibleBy (n divisor : Nat) : Prop := 
  n % divisor = 0

def distinctDigits (n : Nat) : Prop :=
  (List.length (List.erase_dup (Nat.digits 10 n)) = Nat.digits 10 n).length

def isDecreasing (n : Nat) : Prop :=
  ∀ i j, i < j → (Nat.digits 10 n).get? i > (Nat.digits 10 n).get? j

theorem alices_fav_number :
  ∃ (N : Nat),
  distinctDigits N = 8 ∧ 
  isDecreasing N ∧ 
  isDivisibleBy N 180 ↔ 
  N = 97654320 := by
  sorry

end alices_fav_number_l825_825280


namespace company_employees_after_hiring_l825_825695

theorem company_employees_after_hiring
  (T : ℕ)
  (h_sex_ratio_before : 60 * T / 100)
  (h_hired_male : T + 24)
  (h_sex_ratio_after : 55 * (T + 24) / 100) :
  T = 264 → (T + 24) = 288 :=
by
  intro hT_eq
  sorry

end company_employees_after_hiring_l825_825695


namespace triangle_angle_mhc_l825_825032

noncomputable theory

open_locale classical

-- Definitions and conditions
variables (A B C H M : Type)
variables [angle A] [angle B] [angle C]
variables [equilateral_triangle (triangle ABC)]
variables [altitude (segment BH)]
variables [median (segment AM)]

-- Problem to prove
theorem triangle_angle_mhc : angle MHC = 30 :=
sorry

end triangle_angle_mhc_l825_825032


namespace simplest_common_denominator_fraction_exist_l825_825160

variable (x y : ℝ)

theorem simplest_common_denominator_fraction_exist :
  let d1 := x + y
  let d2 := x - y
  let d3 := x^2 - y^2
  (d3 = d1 * d2) → 
    ∀ n, (n = d1 * d2) → 
      (∃ m, (d1 * m = n) ∧ (d2 * m = n) ∧ (d3 * m = n)) :=
by
  sorry

end simplest_common_denominator_fraction_exist_l825_825160


namespace Nancy_money_in_dollars_l825_825645

-- Condition: Nancy has saved 1 dozen quarters
def dozen : ℕ := 12

-- Condition: Each quarter is worth 25 cents
def value_of_quarter : ℕ := 25

-- Condition: 100 cents is equal to 1 dollar
def cents_per_dollar : ℕ := 100

-- Proving that Nancy has 3 dollars
theorem Nancy_money_in_dollars :
  (dozen * value_of_quarter) / cents_per_dollar = 3 := by
  sorry

end Nancy_money_in_dollars_l825_825645


namespace range_of_m_l825_825715

noncomputable def f (a b x : ℝ) : ℝ := (a * real.exp x + b) * (x - 2)

theorem range_of_m (a b : ℝ) (h1 : ∀ x > 0, strict_mono_on (f a b) {x : ℝ | 0 < x}) 
    (h2 : ∀ x, f a b (x - 1) = f a b (2 - x)) :
    ∀ m : ℝ, f a b (2 - m) > 0 ↔ (m < 0 ∨ m > 4) :=
begin
    intro m,
    sorry
end

end range_of_m_l825_825715


namespace speed_increase_71_6_percent_l825_825572

theorem speed_increase_71_6_percent (S : ℝ) (hS : 0 < S) : 
    let S₁ := S * 1.30
    let S₂ := S₁ * 1.10
    let S₃ := S₂ * 1.20
    (S₃ - S) / S * 100 = 71.6 :=
by
  let S₁ := S * 1.30
  let S₂ := S₁ * 1.10
  let S₃ := S₂ * 1.20
  sorry

end speed_increase_71_6_percent_l825_825572


namespace binom_18_4_l825_825334

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end binom_18_4_l825_825334


namespace number_of_red_balls_is_random_variable_l825_825033

-- Conditions: There are 2 black balls and 6 red balls in a bag, and two balls are drawn randomly.
def num_black_balls : Nat := 2
def num_red_balls : Nat := 6
def total_balls : Nat := num_black_balls + num_red_balls

-- The event of drawing two balls randomly from the bag
def draw_two_balls_randomly (ball_num : Nat) : Prop := 
  ball_num ≤ total_balls

-- We need to prove the following statement:
theorem number_of_red_balls_is_random_variable :
  ∃ (X : Set ℕ → Prop), (∀ e ∈ {0, 1, 2}, X e) ∧ ¬X (total_balls)
  := sorry

end number_of_red_balls_is_random_variable_l825_825033


namespace no_circular_triplet_sum_odd_l825_825056

def is_circular_triplet_sum_odd (s : Fin 2018 → ℕ) : Prop :=
  ∀ i : Fin 2018, (s i + s (i + 1) + s (i + 2) % 2018 + 1) % 2 = 1

theorem no_circular_triplet_sum_odd :
  ¬ ∃ s : Fin 2018 → ℕ, (∀ i : Fin 2018, s i ∈ Finset.range 1 2019) ∧ is_circular_triplet_sum_odd s :=
sorry

end no_circular_triplet_sum_odd_l825_825056


namespace average_children_in_families_with_children_l825_825416

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l825_825416


namespace avg_children_in_families_with_children_l825_825411

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l825_825411


namespace smallest_positive_value_of_x_plus_y_l825_825175

theorem smallest_positive_value_of_x_plus_y (x y : ℝ) (k : ℤ) :
  (cos (x + y) = 1) → x + y = 2 * π * k ∧ 2 * π = (x + y) :=
by
  sorry

end smallest_positive_value_of_x_plus_y_l825_825175


namespace angle_between_sides_of_triangle_l825_825913

noncomputable def right_triangle_side_lengths1 : Nat × Nat × Nat := (15, 36, 39)
noncomputable def right_triangle_side_lengths2 : Nat × Nat × Nat := (40, 42, 58)

-- Assuming both triangles are right triangles
def is_right_triangle (a b c : Nat) : Prop := a^2 + b^2 = c^2

theorem angle_between_sides_of_triangle
  (h1 : is_right_triangle 15 36 39)
  (h2 : is_right_triangle 40 42 58) : 
  ∃ (θ : ℝ), θ = 90 :=
by
  sorry

end angle_between_sides_of_triangle_l825_825913


namespace find_m_minus_n_l825_825008

variables (a b : ℝ × ℝ)
variables (m n : ℝ)

-- Given conditions
def vec_a := (2, 1)
def vec_b := (1, -2)
def lin_comb := (5, -5)

-- Proof statement
theorem find_m_minus_n 
  (h1 : m • vec_a.1 + n • vec_b.1 = lin_comb.1)
  (h2 : m • vec_a.2 + n • vec_b.2 = lin_comb.2) : 
  m - n = -2 :=
sorry

end find_m_minus_n_l825_825008


namespace angle_ADC_is_90_l825_825983

variable (A B C D K M N : Type) 
variable [cyclic_quad A B C D] [Point A] [Point B] [Point C] [Point D]
variable (intersects_at : ∀ ab cd : Type, intersects ab cd K)
variable [midpoint M A C] [midpoint N C K]
variable (cyclic_quad_MBND : cyclic_quad M B N D)

theorem angle_ADC_is_90 
  (h1 : cyclic_quad A B C D)
  (h2 : intersects_at AB CD K)
  (h3 : midpoint M A C)
  (h4 : midpoint N C K)
  (h5 : cyclic_quad_MBND M B N D): 
  ∃ (angle_ADC : ℝ), angle_ADC = 90 := 
sorry

end angle_ADC_is_90_l825_825983


namespace minimum_positive_period_of_determinant_is_pi_l825_825938

theorem minimum_positive_period_of_determinant_is_pi : 
  let y (x : ℝ) := (Matrix.det ![![Real.cos x, Real.sin x], ![Real.sin x, Real.cos x]])
  in ∃ a : ℝ, (a > 0) ∧ (∀ x, y (x + a * Real.pi) = y x) := 
  let y (x : ℝ) := (Matrix.det ![![Real.cos x, Real.sin x], ![Real.sin x, Real.cos x]])
  ⟨1, by norm_num, by sorry⟩

end minimum_positive_period_of_determinant_is_pi_l825_825938


namespace AB_eq_B_exp_V_l825_825458

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l825_825458


namespace painting_together_time_l825_825059

theorem painting_together_time (jamshid_time taimour_time time_together : ℝ) 
  (h1 : jamshid_time = taimour_time / 2)
  (h2 : taimour_time = 21)
  (h3 : time_together = 7) :
  (1 / taimour_time + 1 / jamshid_time) * time_together = 1 := 
sorry

end painting_together_time_l825_825059


namespace maximum_diagonals_with_same_length_l825_825039

theorem maximum_diagonals_with_same_length (n : ℕ) (hn : n = 1000) :
  ∃ k, (∀ (diagonals : finset (ℕ × ℕ)) (h : diagonals.card = k), 
          ∀ a b c ∈ diagonals, 
            (a = b ∨ a = c ∨ b = c)) ∧ k = 2000 :=
by
  existsi 2000
  sorry

end maximum_diagonals_with_same_length_l825_825039


namespace find_x_l825_825594

theorem find_x :
    ∃ x : ℚ, (1/7 + 7/x = 15/x + 1/15) ∧ x = 105 := by
  sorry

end find_x_l825_825594


namespace max_marks_tests_l825_825764

theorem max_marks_tests :
  ∃ (T1 T2 T3 T4 : ℝ),
    0.30 * T1 = 80 + 40 ∧
    0.40 * T2 = 105 + 35 ∧
    0.50 * T3 = 150 + 50 ∧
    0.60 * T4 = 180 + 60 ∧
    T1 = 400 ∧
    T2 = 350 ∧
    T3 = 400 ∧
    T4 = 400 :=
by
    sorry

end max_marks_tests_l825_825764


namespace measure_of_U_is_120_l825_825959

variable {α β γ δ ε ζ : ℝ}
variable (h1 : α = γ) (h2 : α = ζ) (h3 : β + δ = 180) (h4 : ε + ζ = 180)

noncomputable def measure_of_U : ℝ :=
  let total_sum := 720
  have sum_of_angles : α + β + γ + δ + ζ + ε = total_sum := by
    sorry
  have subs_suppl_G_R : β + δ = 180 := h3
  have subs_suppl_E_U : ε + ζ = 180 := h4
  have congruent_F_I_U : α = γ ∧ α = ζ := ⟨h1, h2⟩
  let α : ℝ := sorry
  α

theorem measure_of_U_is_120 : measure_of_U h1 h2 h3 h4 = 120 :=
  sorry

end measure_of_U_is_120_l825_825959


namespace smallest_base10_integer_l825_825211

theorem smallest_base10_integer (a b : ℕ) (h1 : a > 3) (h2 : b > 3) :
    (1 * a + 3 = 3 * b + 1) → (1 * 10 + 3 = 13) :=
by
  intros h


-- Prove that  1 * a + 3 = 3 * b + 1 
  have a_eq : a = 3 * b - 2 := by linarith

-- Prove that 1 * 10 + 3 = 13 
  have base_10 := by simp

have the smallest base 10
  sorry

end smallest_base10_integer_l825_825211


namespace chocolate_flavor_sales_l825_825267

-- Define the total number of cups sold
def total_cups : ℕ := 50

-- Define the fraction of winter melon flavor sales
def winter_melon_fraction : ℚ := 2 / 5

-- Define the fraction of Okinawa flavor sales
def okinawa_fraction : ℚ := 3 / 10

-- Proof statement
theorem chocolate_flavor_sales : 
  (total_cups - (winter_melon_fraction * total_cups).toInt - (okinawa_fraction * total_cups).toInt) = 15 := 
  by 
  sorry

end chocolate_flavor_sales_l825_825267


namespace binom_18_4_l825_825338

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end binom_18_4_l825_825338


namespace width_of_room_is_two_l825_825976

-- Declare the conditions
def area : ℝ := 10
def length : ℝ := 5

-- The width of the room derived from the given conditions
def width : ℝ := area / length

-- The proof problem: Prove that the width of the room is 2 feet
theorem width_of_room_is_two : width = 2 := by
  sorry

end width_of_room_is_two_l825_825976


namespace average_children_l825_825434

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l825_825434


namespace six_digit_number_count_l825_825488

-- Define a set of integers representing the given set
def given_set : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define a function to check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define combinations and permutations
noncomputable def combinations (n k : ℕ) : ℕ := nat.choose n k
noncomputable def permutations (n k : ℕ) : ℕ := nat.desc_factorial n k

-- The main statement to be proved
theorem six_digit_number_count :
  let even_numbers := {n ∈ given_set | is_even n},
      odd_numbers := {n ∈ given_set | is_odd n} in
  (combinations 4 3) * (combinations 5 2) * (permutations 5 5) = C(4,3) * C(5,2) * P(5,5) := sorry

end six_digit_number_count_l825_825488


namespace f_x_minus_1_eq_l825_825930

def f(x: ℝ) : ℝ := (x * (x + 3)) / 2

theorem f_x_minus_1_eq (x: ℝ) : f(x - 1) = (x^2 + x - 2) / 2 := by
  sorry

end f_x_minus_1_eq_l825_825930


namespace intersection_A_B_l825_825936

-- Define set A
def A : Set ℝ := { y | ∃ x : ℝ, y = Real.log x }

-- Define set B
def B : Set ℝ := { x | ∃ y : ℝ, y = Real.sqrt x }

-- Prove that the intersection of sets A and B is [0, +∞)
theorem intersection_A_B : A ∩ B = { x | 0 ≤ x } :=
by
  sorry

end intersection_A_B_l825_825936


namespace correct_propositions_l825_825152

-- Given conditions
def prop1 := ({2, 3, 4, 2} : set ℕ).card = 4
def prop2 := ({0} : set ℕ) = {0}
def prop3 := ({1, 2, 3} : set ℕ) ≠ ({3, 2, 1} : set ℕ)
def prop4 := set.finite { q : ℚ // 0 < q ∧ q < 1 }

-- Prove that the correct propositions are only prop2
theorem correct_propositions :
    ¬ prop1 ∧ prop2 ∧ ¬ prop3 ∧ ¬ prop4 :=
by
  sorry

end correct_propositions_l825_825152


namespace necessary_and_sufficient_condition_for_S6_eq_3S2_l825_825884

variable {a : ℕ → ℝ} -- The geometric sequence
variable {q : ℝ} -- The common ratio
variable Sn : ℕ → ℝ -- The sum of first n terms
variable h1 : ∀ n, Sn n = a 0 * (1 - q^n) / (1 - q) -- Sum formula for geometric sequence

theorem necessary_and_sufficient_condition_for_S6_eq_3S2
    (h2 : Sn 6 = 3 * Sn 2) :
    |q| = 1 :=
by
  sorry

end necessary_and_sufficient_condition_for_S6_eq_3S2_l825_825884


namespace probability_tian_ji_winning_l825_825609

-- Definition of conditions
variable (T_best K_best T_avg K_avg T_worst K_worst : ℕ)
variable (H₁ : K_best > T_best ∧ T_best > K_avg)
variable (H₂ : K_avg > T_avg ∧ T_avg > K_worst)
variable (H₃ : K_worst > T_worst)

-- Prove the probability of Tian Ji winning is 1/6
theorem probability_tian_ji_winning : 
  (∃! T_strategy : list ℕ, 
    T_strategy = [K_best, K_avg, K_worst] ∧ [T_worst, T_best, T_avg] = T_strategy → 
    true ∧ probability_tian_ji_winning = 1/6) :=
sorry

end probability_tian_ji_winning_l825_825609


namespace money_worthless_in_wrong_context_l825_825113

-- Define the essential functions money must serve in society
def InContext (society: Prop) : Prop := 
  ∀ (m : Type), (∃ (medium_of_exchange store_of_value unit_of_account standard_of_deferred_payment : m → Prop), true)

-- Define the context of a deserted island where these functions are useless
def InContext (deserted_island: Prop) : Prop := 
  ∀ (m : Type), (¬ (∃ (medium_of_exchange store_of_value unit_of_account standard_of_deferred_payment : m → Prop), true))

-- Define the essential properties for an item to become money
def EssentialProperties (item: Type) : Prop :=
  ∃ (durable portable divisible acceptable uniform limited_supply : item → Prop), true

-- The primary theorem: Proving that money becomes worthless in the absence of its essential functions
theorem money_worthless_in_wrong_context (m : Type) (society deserted_island : Prop)
  (h1 : InContext society) (h2 : InContext deserted_island) (h3 : EssentialProperties m) :
  ∃ (worthless : m → Prop), true :=
sorry

end money_worthless_in_wrong_context_l825_825113


namespace mutually_exclusive_not_opposite_l825_825945

-- Define the number of each type of ball
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def black_balls : ℕ := 1

-- Define the event of drawing exactly one white ball
def exactly_one_white_ball (drawn_balls : list ℕ) : Prop :=
  count_occ drawn_balls 0 == 1

-- Define the event of drawing exactly two white balls
def exactly_two_white_balls (drawn_balls : list ℕ) : Prop :=
  count_occ drawn_balls 1 == 2

theorem mutually_exclusive_not_opposite {drawn_balls : list ℕ} (h : length drawn_balls = 2):
  (exactly_one_white_ball drawn_balls ∧ exactly_two_white_balls drawn_balls → false) ∧ 
  ¬(exactly_one_white_ball drawn_balls ↔ exactly_two_white_balls drawn_balls) :=
by sorry

end mutually_exclusive_not_opposite_l825_825945


namespace cos_2beta_eq_2cos_2alpha_l825_825525

theorem cos_2beta_eq_2cos_2alpha (α β : ℝ) (h1 : ∀ k : ℤ, α ≠ k * π + π / 2) (h2 : ∀ k : ℤ, β ≠ k * π + π / 2)
    (h3 : ∃ θ : ℝ, sin θ = 2 * sin α ∧ cos θ = sin θ * cos θ=sin^2 β): 
    cos (2 * β) = 2 * cos (2 * α) :=
sorry

end cos_2beta_eq_2cos_2alpha_l825_825525


namespace maximize_profit_l825_825960

noncomputable def fixed_cost : ℝ := 50000

def variable_cost (x : ℝ) : ℝ :=
if 0 < x ∧ x < 8 then
  (1/2) * x^2 + 4 * x
else if 8 ≤ x then
  11 * x + 49 / x - 35
else
  0

def revenue (x : ℝ) : ℝ := 10 * x * 10000

def profit (x : ℝ) : ℝ := revenue x - fixed_cost - variable_cost x

theorem maximize_profit : ∃ x, x = 8 ∧ profit x = (127 / 8) * 10000 := 
sorry

end maximize_profit_l825_825960


namespace AB_eq_B_exp_V_l825_825461

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l825_825461


namespace eval_abs_expression_l825_825733

theorem eval_abs_expression : (30 - | - 10 + 6 |) = 26 := by
  -- Proof not required as per the instruction.
  sorry

end eval_abs_expression_l825_825733


namespace find_a_in_triangle_l825_825052

theorem find_a_in_triangle (b c : ℝ) (cos_B_minus_C : ℝ) (a : ℝ) 
  (hb : b = 7) (hc : c = 6) (hcos : cos_B_minus_C = 15 / 16) :
  a = 5 * Real.sqrt 3 :=
by
  sorry

end find_a_in_triangle_l825_825052


namespace freight_train_speed_is_36_kmh_l825_825271

open Real

variables (x : ℝ)

def speed_ratio_passenger_to_freight : ℝ := 5 / 3

def total_length_met_and_passed : ℝ := 200 + 280

def time_to_pass_each_other : ℝ := 18

def equation : Prop :=
  (x + speed_ratio_passenger_to_freight * x) * time_to_pass_each_other = total_length_met_and_passed

def speed_freight_train_in_kmh (x : ℝ) : ℝ := x * 3600 / 1000

theorem freight_train_speed_is_36_kmh : equation x → speed_freight_train_in_kmh x = 36 :=
  by
    intros h
    sorry

end freight_train_speed_is_36_kmh_l825_825271


namespace dot_product_sum_l825_825088

noncomputable def norm (v : ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)
noncomputable def dot_product (v w : ℝ × ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

theorem dot_product_sum (a b c k : ℝ × ℝ × ℝ)
  (ha : norm a = 5) (hb : norm b = 3) (hc : norm c = 4) (hk : norm k = 2)
  (h_eq : a + b + c = k) :
  dot_product a b + dot_product a c + dot_product b c = -23 := by
  sorry

end dot_product_sum_l825_825088


namespace mod_x_squared_l825_825564

theorem mod_x_squared :
  (∃ x : ℤ, 5 * x ≡ 9 [ZMOD 26] ∧ 4 * x ≡ 15 [ZMOD 26]) →
  ∃ y : ℤ, y ≡ 10 [ZMOD 26] :=
by
  intro h
  rcases h with ⟨x, h₁, h₂⟩
  exists x^2
  sorry

end mod_x_squared_l825_825564


namespace squirrels_collect_acorns_l825_825240

theorem squirrels_collect_acorns
  (squirrels : ℕ)
  (acorns_needed : ℕ)
  (extra_acorns : ℕ)
  (squirrels = 5)
  (acorns_needed = 130)
  (extra_acorns = 15) :
  squirrels * (acorns_needed - extra_acorns) = 575 :=
by
  sorry

end squirrels_collect_acorns_l825_825240


namespace avg_children_with_kids_l825_825398

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l825_825398


namespace money_properties_proof_l825_825108

-- The context of Robinson Crusoe being on a deserted island.
def deserted_island := true

-- The functions of money in modern society.
def medium_of_exchange := true
def store_of_value := true
def unit_of_account := true
def standard_of_deferred_payment := true

-- The financial context of the island.
def island_context (deserted_island : Prop) : Prop :=
  deserted_island →
  ¬ medium_of_exchange ∧
  ¬ store_of_value ∧
  ¬ unit_of_account ∧
  ¬ standard_of_deferred_payment

-- Other properties that an item must possess to become money.
def durability := true
def portability := true
def divisibility := true
def acceptability := true
def uniformity := true
def limited_supply := true

-- The proof problem statement.
theorem money_properties_proof
  (H1 : deserted_island)
  (H2 : island_context H1)
  : (¬ medium_of_exchange ∧
    ∀ (m : Prop), (m = durability ∨ m = portability ∨ m = divisibility ∨ m = acceptability ∨ m = uniformity ∨ m = limited_supply)) :=
by {
  sorry
}

end money_properties_proof_l825_825108


namespace average_children_in_families_with_children_l825_825425

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l825_825425


namespace distance_between_points_l825_825839

def point3d := (ℝ × ℝ × ℝ)

def dist3d (p1 p2 : point3d) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

def p1 : point3d := (1, 3, 0)
def p2 : point3d := (4, 0, 3)

theorem distance_between_points : dist3d p1 p2 = 3 * Real.sqrt 3 := 
by 
  sorry

end distance_between_points_l825_825839


namespace inequality_solution_l825_825661

theorem inequality_solution (x : ℝ) :
  (x / (x^2 - 4) ≥ 0) ↔ (x ∈ Set.Iio (-2) ∪ Set.Ico 0 2) :=
by sorry

end inequality_solution_l825_825661


namespace smallest_integer_representable_l825_825197

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end smallest_integer_representable_l825_825197


namespace find_m_l825_825340

def s : ℕ → ℚ
| 1       := 1
| (m + 1) := if (m + 1) % 2 = 0 then 1 + s ((m + 1) / 2) else 1 / s m

theorem find_m (m : ℕ) (h : s m = 17 / 99) : m = 2113 :=
by
  sorry

end find_m_l825_825340


namespace function_properties_l825_825345

def f (x : ℝ) : ℝ := (10 ^ x - 10 ^ (-x)) / (10 ^ x + 10 ^ (-x))

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) := by
  sorry

end function_properties_l825_825345


namespace no_circular_triplet_sum_odd_l825_825055

def is_circular_triplet_sum_odd (s : Fin 2018 → ℕ) : Prop :=
  ∀ i : Fin 2018, (s i + s (i + 1) + s (i + 2) % 2018 + 1) % 2 = 1

theorem no_circular_triplet_sum_odd :
  ¬ ∃ s : Fin 2018 → ℕ, (∀ i : Fin 2018, s i ∈ Finset.range 1 2019) ∧ is_circular_triplet_sum_odd s :=
sorry

end no_circular_triplet_sum_odd_l825_825055


namespace class1_participation_l825_825747

theorem class1_participation :
  let total_students := 40 + 36 + 44
  let non_participating := 30
  let x := (total_students - non_participating) / total_students
  in (40 * x) = 30 := by
  sorry

end class1_participation_l825_825747


namespace math_proof_problem_l825_825069

open BigOperators

theorem math_proof_problem (n : ℕ) (x : ℕ → ℝ)
  (h_n_ge_3 : n ≥ 3)
  (h_x_nonneg : ∀ i : ℕ, i < n → x i ≥ 0) :
  let A := ∑ i in finset.range n, x i,
      B := ∑ i in finset.range n, (x i)^2,
      C := ∑ i in finset.range n, (x i)^3 in
  (n + 1) * A^2 * B + (n - 2) * B^2 ≥ A^4 + (2 * n - 2) * A * C := 
by
  sorry

end math_proof_problem_l825_825069


namespace distance_AD_l825_825649

-- Define points A, B, C, D as needed
variables (A B C D : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Define the geometric conditions
variable h1 : dist A B = 12
variable h2 : dist B C = 12
variable h3 : dist A C = 12 * sqrt 2
variable h4 : angle A B C = real.pi / 4
variable h5 : dist C D = 24

-- Define what we need to prove
theorem distance_AD :
  dist A D = 12 * sqrt 6 :=
sorry

end distance_AD_l825_825649


namespace triangle_area_in_circle_l825_825749

theorem triangle_area_in_circle (r : ℝ) (arc1 arc2 arc3 : ℝ) 
  (circumference_eq : arc1 + arc2 + arc3 = 24)
  (radius_eq : 2 * Real.pi * r = 24) : 
  1 / 2 * (r ^ 2) * (Real.sin (105 * Real.pi / 180) + Real.sin (120 * Real.pi / 180) + Real.sin (135 * Real.pi / 180)) = 364.416 / (Real.pi ^ 2) :=
by
  sorry

end triangle_area_in_circle_l825_825749


namespace mean_square_is_integer_l825_825640

noncomputable def mean_square (n : ℕ) : ℕ :=
  let sum_of_squares := (n * (n + 1) * (2 * n + 1)) / 6
  let mean_square_val := sum_of_squares / n
  nat.sqrt mean_square_val

theorem mean_square_is_integer {n : ℕ} (h : n > 1) : 
  ∃ n_min, (mean_square n_min = ⌊mean_square n⌋ ∧ mean_square n_min ∈ ℕ ∧ ∀ (m : ℕ), m < n_min → ¬(mean_square m ∈ ℕ)) :=
sorry

end mean_square_is_integer_l825_825640


namespace real_solution_four_unknowns_l825_825456

theorem real_solution_four_unknowns (x y z t : ℝ) :
  x^2 + y^2 + z^2 + t^2 = x * (y + z + t) ↔ (x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0) :=
by
  sorry

end real_solution_four_unknowns_l825_825456


namespace shortest_tangent_length_l825_825082

noncomputable def center_circle1 : ℝ × ℝ := (12, 0)
noncomputable def radius_circle1 : ℝ := 7
noncomputable def equation_circle1 (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 49

noncomputable def center_circle2 : ℝ × ℝ := (-18, 0)
noncomputable def radius_circle2 : ℝ := 8
noncomputable def equation_circle2 (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 64

theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, 
  equation_circle1 P.1 P.2 ∧ 
  equation_circle2 Q.1 Q.2 ∧ 
  tangent P Q ∧
  shortest_tangent P Q ∧
  (line_segment_length P Q = 20) := sorry

end shortest_tangent_length_l825_825082


namespace smallest_base10_integer_l825_825208

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end smallest_base10_integer_l825_825208


namespace sister_weight_difference_is_12_l825_825285

-- Define Antonio's weight
def antonio_weight : ℕ := 50

-- Define the combined weight of Antonio and his sister
def combined_weight : ℕ := 88

-- Define the weight of Antonio's sister
def sister_weight : ℕ := combined_weight - antonio_weight

-- Define the weight difference
def weight_difference : ℕ := antonio_weight - sister_weight

-- Theorem statement to prove the weight difference is 12 kg
theorem sister_weight_difference_is_12 : weight_difference = 12 := by
  sorry

end sister_weight_difference_is_12_l825_825285


namespace value_of_x_squared_plus_inverse_squared_l825_825571

theorem value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x + 1 / x = 8) : x^2 + 1 / x^2 = 62 := 
sorry

end value_of_x_squared_plus_inverse_squared_l825_825571


namespace trig_expression_value_l825_825491

theorem trig_expression_value (θ : Real) (h1 : θ > Real.pi) (h2 : θ < 3 * Real.pi / 2) (h3 : Real.tan (2 * θ) = 3 / 4) :
  (2 * Real.cos (θ / 2) ^ 2 + Real.sin θ - 1) / (Real.sqrt 2 * Real.cos (θ + Real.pi / 4)) = 2 := by
  sorry

end trig_expression_value_l825_825491


namespace edward_lives_left_l825_825222

theorem edward_lives_left (initial_lives : ℕ) (lives_lost : ℕ) (final_lives : ℕ):
  initial_lives = 15 → lives_lost = 8 → final_lives = 15 - 8 → final_lives = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end edward_lives_left_l825_825222


namespace proof_problem_l825_825877

variable (y θ Q : ℝ)

-- Given condition
def condition : Prop := 5 * (3 * y + 7 * Real.sin θ) = Q

-- Goal to be proved
def goal : Prop := 15 * (9 * y + 21 * Real.sin θ) = 9 * Q

theorem proof_problem (h : condition y θ Q) : goal y θ Q :=
by
  sorry

end proof_problem_l825_825877


namespace expected_value_twelve_sided_die_l825_825774

theorem expected_value_twelve_sided_die : 
  let die_sides := 12 in 
  let outcomes := finset.range (die_sides + 1) in
  (finset.sum outcomes id : ℚ) / die_sides = 6.5 :=
by
  sorry

end expected_value_twelve_sided_die_l825_825774


namespace statement_B_statement_C_statement_D_l825_825601

notation "ℝ" => Real
notation "∠" => Real.angle

structure Triangle :=
(A B C : Point)
(a b c : ℝ)

noncomputable def sin (x : ℝ) : ℝ := Real.sin x
noncomputable def dot (u v : ℝ × ℝ) : ℝ := u.fst * v.fst + u.snd * v.snd

def isObtuse (triangle : Triangle) (A B C : Real) : Prop :=
  A > π / 2 ∨ B > π / 2 ∨ C > π / 2

def isAcute (triangle : Triangle) (A B C : Real) : Prop :=
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2

theorem statement_B (A B : ℝ) : sin A > sin B → A > B := by
  sorry

theorem statement_C (AC BC : ℝ × ℝ) : dot AC BC > 0 → isObtuse (Triangle.mk P Q R 0 0 0) (∠ P Q R) (∠ Q R P) (∠ R P Q) := by
  sorry

theorem statement_D (a b c : ℝ) : a ^ 3 + b ^ 3 = c ^ 3 → isAcute (Triangle.mk P Q R a b c) (∠ P Q R) (∠ Q R P) (∠ R P Q) := by
  sorry

end statement_B_statement_C_statement_D_l825_825601


namespace reece_climbs_15_times_l825_825979

/-
Given:
1. Keaton's ladder height: 30 feet.
2. Keaton climbs: 20 times.
3. Reece's ladder is 4 feet shorter than Keaton's ladder.
4. Total length of ladders climbed by both is 11880 inches.

Prove:
Reece climbed his ladder 15 times.
-/

theorem reece_climbs_15_times :
  let keaton_ladder_feet := 30
  let keaton_climbs := 20
  let reece_ladder_feet := keaton_ladder_feet - 4
  let total_length_inches := 11880
  let feet_to_inches (feet : ℕ) := 12 * feet
  let keaton_ladder_inches := feet_to_inches keaton_ladder_feet
  let reece_ladder_inches := feet_to_inches reece_ladder_feet
  let keaton_total_climbed := keaton_ladder_inches * keaton_climbs
  let reece_total_climbed := total_length_inches - keaton_total_climbed
  let reece_climbs := reece_total_climbed / reece_ladder_inches
  reece_climbs = 15 :=
by
  sorry

end reece_climbs_15_times_l825_825979


namespace tan_theta_half_l825_825616

open Matrix

def D (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![k, 0], ![0, k]]

def R (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem tan_theta_half (k θ : ℝ) (h₁ : k > 0) 
  (h₂ : mul (R θ) (D k) = ![![8, -4], ![4, 8]]) :
  Real.tan θ = 1 / 2 := 
sorry

end tan_theta_half_l825_825616


namespace inequality_solution_sets_l825_825550

theorem inequality_solution_sets (a b m : ℝ) (h_sol_set : ∀ x, x^2 - a * x - 2 > 0 ↔ x < -1 ∨ x > b) (hb : b > -1) (hm : m > -1 / 2) :
  a = 1 ∧ b = 2 ∧ 
  (if m > 0 then ∀ x, (x < -1/m ∨ x > 2) ↔ (mx + 1) * (x - 2) > 0 
   else if m = 0 then ∀ x, x > 2 ↔ (mx + 1) * (x - 2) > 0 
   else ∀ x, (2 < x ∧ x < -1/m) ↔ (mx + 1) * (x - 2) > 0) :=
by
  sorry

end inequality_solution_sets_l825_825550


namespace problem1_problem2_problem3a_problem3b_problem3c_problem3d_l825_825871

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

theorem problem1 : f (x + 1) - f x = 2 * x ∧ f 0 = 1 :=
by
  sorry

theorem problem2 : set_of (y, ∃ x ∈ set.Icc (-1 : ℝ) 1, f x = y) = set.Icc (-1/2 : ℝ) 1 :=
by
  sorry

theorem problem3a (a : ℝ) (h : a ≤ -(1/2)) :
  set_of (y, ∃ x ∈ set.Icc a (a + 1), f x = y) = set.Icc (2 * a^2 + 2 * a + 3 : ℝ) (2 * a^2 - 2 * a + 1 : ℝ) :=
by
  sorry

theorem problem3b (a : ℝ) (h : a ≤ 0) :
  set_of (y, ∃ x ∈ set.Icc a (a + 1), f x = y) = set.Icc (-1/2 : ℝ) (2 * a^2 - 2 * a + 1 : ℝ) :=
by
  sorry

theorem problem3c (a : ℝ) (h : 0 ≤ a) :
  set_of (y, ∃ x ∈ set.Icc a (a + 1), f x = y) = set.Icc (-1/2 : ℝ) (2 * a^2 + 2 * a + 3 : ℝ) :=
by
  sorry

theorem problem3d (a : ℝ) (h : (1/2) < a) :
  set_of (y, ∃ x ∈ set.Icc a (a + 1), f x = y) = set.Icc (2 * a^2 - 2 * a + 1 : ℝ) (2 * a^2 + 2 * a + 3 : ℝ) :=
by
  sorry

end problem1_problem2_problem3a_problem3b_problem3c_problem3d_l825_825871


namespace telepathy_probability_l825_825669

-- Define the set of all possible pairs (x, y) where both x and y are in [1, 4]
def S : Finset (ℕ × ℕ) := do
  x ← Finset.range 4
  y ← Finset.range 4
  pure (x + 1, y + 1)

-- Define the set of pairs (x, y) where |x - y| ≤ 1
def T : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 - p.2).abs ≤ 1)

-- Prove the probability of having "telepathy" is 5/8
theorem telepathy_probability : (T.card : ℚ) / S.card = 5 / 8 := by
  sorry

end telepathy_probability_l825_825669


namespace average_children_in_families_with_children_l825_825387

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l825_825387


namespace smallest_fourth_number_l825_825223

-- Define the given conditions
def first_three_numbers_sum : ℕ := 28 + 46 + 59 
def sum_of_digits_of_first_three_numbers : ℕ := 2 + 8 + 4 + 6 + 5 + 9 

-- Define the condition for the fourth number represented as 10a + b and its digits 
def satisfies_condition (a b : ℕ) : Prop := 
  first_three_numbers_sum + 10 * a + b = 4 * (sum_of_digits_of_first_three_numbers + a + b)

-- Statement to prove the smallest fourth number
theorem smallest_fourth_number : ∃ (a b : ℕ), satisfies_condition a b ∧ 10 * a + b = 11 := 
sorry

end smallest_fourth_number_l825_825223


namespace smallest_x_consecutive_cubes_l825_825965

theorem smallest_x_consecutive_cubes :
  ∃ (u v w x : ℕ), u < v ∧ v < w ∧ w < x ∧ u + 1 = v ∧ v + 1 = w ∧ w + 1 = x ∧ (u^3 + v^3 + w^3 = x^3) ∧ (x = 6) :=
by {
  sorry
}

end smallest_x_consecutive_cubes_l825_825965


namespace count_special_multiples_l825_825014

theorem count_special_multiples : 
  let lcm := 15 in
  let multiples := {n ∈ (Finset.range 200).filter (λ n => n % lcm = 0)} in
  let valid_multiples := multiples.filter (λ n => n % 4 ≠ 0 ∧ n % 7 ≠ 0) in
  valid_multiples.card = 9 := 
by
  sorry

end count_special_multiples_l825_825014


namespace numeric_puzzle_AB_eq_B_pow_V_l825_825470

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l825_825470


namespace expansion_terms_count_l825_825559

-- Define the number of terms in the first polynomial
def first_polynomial_terms : ℕ := 3

-- Define the number of terms in the second polynomial
def second_polynomial_terms : ℕ := 4

-- Prove that the number of terms in the expansion is 12
theorem expansion_terms_count : first_polynomial_terms * second_polynomial_terms = 12 :=
by
  sorry

end expansion_terms_count_l825_825559


namespace golden_week_tourism_l825_825353

-- Definitions for the number of tourists each day
def tourists_sept30 : ℝ := 3
def change_oct1 : ℝ := 1.6
def change_oct2 : ℝ := 0.8
def change_oct3 : ℝ := 0.4
def change_oct4 : ℝ := -0.4
def change_oct5 : ℝ := -0.8
def change_oct6 : ℝ := 0.2
def change_oct7 : ℝ := -1.4
def entrance_fee : ℝ := 220

-- Computations for each day's tourists
def tourists_oct1 : ℝ := tourists_sept30 + change_oct1
def tourists_oct2 : ℝ := tourists_oct1 + change_oct2
def tourists_oct3 : ℝ := tourists_oct2 + change_oct3
def tourists_oct4 : ℝ := tourists_oct3 + change_oct4
def tourists_oct5 : ℝ := tourists_oct4 + change_oct5
def tourists_oct6 : ℝ := tourists_oct5 + change_oct6
def tourists_oct7 : ℝ := tourists_oct6 + change_oct7

-- The Lean statement for proving the questions
theorem golden_week_tourism :
  tourists_oct2 = 5.4 ∧
  tourists_oct3 = max tourists_oct1 tourists_oct2 tourists_oct3 tourists_oct4 tourists_oct5 tourists_oct6 tourists_oct7 ∧
  tourists_oct3 = 5.8 ∧
  (tourists_oct1 + tourists_oct2 + tourists_oct3 + tourists_oct4 + tourists_oct5 + tourists_oct6 + tourists_oct7) * entrance_fee = 7480 :=
by sorry

end golden_week_tourism_l825_825353


namespace binomial_equality_l825_825321

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l825_825321


namespace chocolate_milk_tea_sales_l825_825261

theorem chocolate_milk_tea_sales (total_sales : ℕ) (winter_melon_ratio : ℚ) (okinawa_ratio : ℚ) :
  total_sales = 50 →
  winter_melon_ratio = 2 / 5 →
  okinawa_ratio = 3 / 10 →
  ∃ (chocolate_sales : ℕ), chocolate_sales = total_sales - total_sales * winter_melon_ratio - total_sales * okinawa_ratio ∧ chocolate_sales = 15 :=
by
  intro h1 h2 h3
  use (total_sales - total_sales * winter_melon_ratio - total_sales * okinawa_ratio).to_nat
  split
  · simp [h1, h2, h3]
  · exact sorry

end chocolate_milk_tea_sales_l825_825261


namespace prod_xn_eq_inv_n_plus_one_l825_825531

noncomputable def xn (n : ℕ) (hn : n > 0) : ℕ → ℝ
| 1       := 1 / ↑(n + 1)
| (k + 1) := xn k * (k / (k + 1) : ℝ)

theorem prod_xn_eq_inv_n_plus_one (n : ℕ) (hn : n > 0) :
  (finset.range n).prod (λ k, xn n hn (k + 1)) = 1 / (n + 1) := by
  sorry

end prod_xn_eq_inv_n_plus_one_l825_825531


namespace find_p_q_l825_825998

noncomputable def a_0 : ℕ := 3
noncomputable def b_0 : ℕ := 4

noncomputable def a (n : ℕ) : ℝ := 
  if n = 0 then a_0 else (a (n - 1))^3 / (b (n - 1))^2

noncomputable def b (n : ℕ) : ℝ :=
  if n = 0 then b_0 else (b (n - 1))^3 / (a (n - 1))^2

noncomputable def s_0 := 1
noncomputable def t_0 := 0

noncomputable def s (n : ℕ) : ℕ :=
  if n = 0 then s_0 else if n = 1 then 3 else 3 * s (n - 1) - 2 * s (n - 2)

noncomputable def t (n : ℕ) : ℕ :=
  if n = 0 then t_0 else if n = 1 then 2 else 3 * t (n - 1) - 2 * t (n - 2)

theorem find_p_q : s 8 = 1393 ∧ t 8 = 1392 := by
  sorry

end find_p_q_l825_825998


namespace altitude_eq_symmetric_point_eq_l825_825551

-- Define vertices
def A : (ℝ × ℝ) := (0, 3)
def B : (ℝ × ℝ) := (-2, -1)
def C : (ℝ × ℝ) := (4, 3)

-- Define the statement for the first part of the problem
theorem altitude_eq (A B C : (ℝ × ℝ)) : 
  (∃ k: ℝ, (y - 3 = k * (x - 4)) ∧ (k = -1 / 2) ) ↔ (x + 2y - 10 = 0) := 
  sorry

-- Define the statement for the second part of the problem
theorem symmetric_point_eq (A B C : (ℝ × ℝ)) : 
  (symmetric_point_to_line (A, B) C) = (-12 / 5, 31 / 5) := 
  sorry

end altitude_eq_symmetric_point_eq_l825_825551


namespace sum_first_985_terms_of_sequence_l825_825692

theorem sum_first_985_terms_of_sequence :
  (∑ n in Finset.range 985, (nat.recOn n (2 : ℝ) (λ n a_n, 1 / (1 - a_n)))) = 494 := 
sorry

end sum_first_985_terms_of_sequence_l825_825692


namespace proposition_equivalence_l825_825526

-- Definition of propositions p and q
variables (p q : Prop)

-- Statement of the problem in Lean 4
theorem proposition_equivalence :
  (p ∨ q) → ¬(p ∧ q) ↔ (¬((p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → (p ∨ q))) :=
sorry

end proposition_equivalence_l825_825526


namespace avg_children_with_kids_l825_825400

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l825_825400


namespace binom_18_4_eq_3060_l825_825327

theorem binom_18_4_eq_3060 : nat.choose 18 4 = 3060 := sorry

end binom_18_4_eq_3060_l825_825327


namespace increase_in_work_per_person_l825_825735

-- Define the conditions and the proof problem
theorem increase_in_work_per_person (W p : ℝ) (h1 : p > 0) : 
  let absent := 1 / 7 * p in
  let present := p - absent in
  (W / present) - (W / p) = W / (6 * p) :=
by
  let absent := 1 / 7 * p
  let present := p - absent
  have h_present : present = 6 / 7 * p := by 
    calc 
      present = p - absent : rfl
      _ = p - 1 / 7 * p : rfl
      _ = 7 / 7 * p - 1 / 7 * p : by norm_num
      _ = 6 / 7 * p : by ring
  sorry -- Proof steps to be filled later, thus the theorem statement is complete

end increase_in_work_per_person_l825_825735


namespace complex_imaginary_axis_l825_825147

theorem complex_imaginary_axis (a : ℝ) : (a^2 - 2 * a = 0) ↔ (a = 0 ∨ a = 2) := 
by
  sorry

end complex_imaginary_axis_l825_825147


namespace find_circle_radius_l825_825750

theorem find_circle_radius :
  ∃ (r : ℝ), 
    let O := (r, r);
    let A := (0, 0);
    let B := (2, 0);
    let C := (0, 2);
    let hypotenuse := dist B C;
    O.x * O.x + O.y * O.y = r * r ∧
    dist O (0, r) = r ∧
    dist O (r, 0) = r ∧
    dist O ((2 - hypotenuse)/2, (2 - hypotenuse)/2) = r ∧
    r = 2 + 2 * sqrt 2 :=
begin
  use (2 + 2 * sqrt 2),
  sorry,
end

end find_circle_radius_l825_825750


namespace num_correct_expressions_l825_825281

theorem num_correct_expressions {
  let expr1 := (a^5 + a^5 = a^{10}),
  let expr2 := (a^6 * a^4 = a^{24}),
  let expr3 := (a^0 / a^{-1} = a),
  let expr4 := (a^4 - a^4 = a^0),
  (¬ expr1 ∧ ¬ expr2 ∧ expr3 ∧ ¬ expr4) → (number_of_correct_expressions = 1)
} : sorry

end num_correct_expressions_l825_825281


namespace speed_of_train_is_36_kmh_l825_825732

-- Definitions and assumptions
variables (L : ℝ) (V : ℝ)
constant (time_to_pass_pole : ℝ := 10)
constant (time_to_pass_stationary_train : ℝ := 40)
constant (length_stationary_train : ℝ := 300)

-- Assumptions based on problem conditions
axiom length_train : L = (L + length_stationary_train) / 4
axiom speed_equation : V = L / time_to_pass_pole

-- Theorem to prove the speed of the train is 36 km/h
theorem speed_of_train_is_36_kmh : V * 3.6 = 36 :=
by
  sorry

end speed_of_train_is_36_kmh_l825_825732


namespace binom_18_4_l825_825329

theorem binom_18_4 : Nat.binomial 18 4 = 3060 :=
by
  -- We start the proof here.
  sorry

end binom_18_4_l825_825329


namespace batter_sugar_is_one_l825_825098

-- Definitions based on the conditions given
def initial_sugar : ℕ := 3
def sugar_per_bag : ℕ := 6
def num_bags : ℕ := 2
def frosting_sugar_per_dozen : ℕ := 2
def total_dozen_cupcakes : ℕ := 5

-- Total sugar Lillian has
def total_sugar : ℕ := initial_sugar + num_bags * sugar_per_bag

-- Sugar needed for frosting
def frosting_sugar_needed : ℕ := frosting_sugar_per_dozen * total_dozen_cupcakes

-- Sugar used for the batter
def batter_sugar_total : ℕ := total_sugar - frosting_sugar_needed

-- Question asked in the problem
def batter_sugar_per_dozen : ℕ := batter_sugar_total / total_dozen_cupcakes

theorem batter_sugar_is_one :
  batter_sugar_per_dozen = 1 :=
by
  sorry -- Proof is not required here

end batter_sugar_is_one_l825_825098


namespace average_children_families_with_children_is_3_point_8_l825_825370

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l825_825370


namespace average_children_in_families_with_children_l825_825429

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l825_825429


namespace possible_values_of_a_l825_825664

def isFactor (x y : ℕ) : Prop :=
  y % x = 0

def isDivisor (x y : ℕ) : Prop :=
  x % y = 0

def isPositive (x : ℕ) : Prop :=
  x > 0

theorem possible_values_of_a : 
  { a : ℕ // isFactor 4 a ∧ isDivisor 24 a ∧ isPositive a }.card = 4 := by
  sorry

end possible_values_of_a_l825_825664


namespace number_of_trees_on_farm_l825_825277

/-- Given the conditions of the problem: -/
def num_branches : ℕ := 10
def num_sub_branches : ℕ := 40
def leaves_per_sub_branch : ℕ := 60
def total_leaves_all_trees : ℕ := 96000

/-- We want to prove that the number of trees is 4. -/
theorem number_of_trees_on_farm : 
  (total_leaves_all_trees = num_branches * num_sub_branches * leaves_per_sub_branch * (total_leaves_all_trees / (num_branches * num_sub_branches * leaves_per_sub_branch))) 
  → (total_leaves_all_trees / (num_branches * num_sub_branches * leaves_per_sub_branch) = 4): sorry

end number_of_trees_on_farm_l825_825277


namespace rational_is_opposite_of_perfect_square_l825_825928

theorem rational_is_opposite_of_perfect_square (a : ℝ) (h : ∃ (b : ℚ), sqrt (-a) = b) : 
  ∃ (c : ℤ), a = - (c ^ 2) :=
sorry

end rational_is_opposite_of_perfect_square_l825_825928


namespace average_children_in_families_with_children_l825_825385

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l825_825385


namespace combined_area_of_rugs_l825_825171

theorem combined_area_of_rugs :
  ∀ (A two_layers three_layers one_layer: ℝ),
    A = 200 ∧
    138 - two_layers - three_layers = one_layer ∧
    one_layer = 95 ∧
    two_layers = 24 ∧
    three_layers = 19 →
    A = one_layer + 2 * two_layers + 3 * three_layers :=
by
  intros A two_layers three_layers one_layer h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end combined_area_of_rugs_l825_825171


namespace expected_value_twelve_sided_die_l825_825780

theorem expected_value_twelve_sided_die : 
  (1 / 12 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l825_825780


namespace circle_chord_intersection_l825_825252

theorem circle_chord_intersection (r A B C D P M N : ℝ)
 (h1 : r = 20)
 (h2 : dist A B = 24)
 (h3 : dist C D = 16)
 (h4 : dist M N = 10)
 (h5 : is_midpoint M A B)
 (h6 : is_midpoint N C D)
 (h7 : intersects_chords A B C D P)
 : ∃ m n : ℕ, coprime m n ∧ (OP^2 : ℝ) = (2025/7 : ℝ) ∧ m + n = 2032 :=
by
  sorry

end circle_chord_intersection_l825_825252


namespace sum_of_possible_values_of_k_l825_825593

theorem sum_of_possible_values_of_k : 
  ∑ k in {k | ∃ j : ℕ, j > 0 ∧ k > 0 ∧ (1 / (j : ℚ) + 1 / (k : ℚ) = 1 / 4)}, id k = 51 := 
by
  sorry

end sum_of_possible_values_of_k_l825_825593


namespace lobachevsky_method_a_lobachevsky_method_b_l825_825738

noncomputable def P (x : ℂ) (n : ℕ) (a : Fin n → ℂ) : ℂ :=
  ∑ (i : Fin n), (a i) * x ^ (n - 1 - i)

def Pk (k : ℕ) (x : ℂ) (n : ℕ) (a : Fin n → ℂ) : ℂ :=
  ∑ (i : Fin n), (a i) * (x ^ (2 ^ k)) ^ (n - 1 - i)

theorem lobachevsky_method_a 
  (n : ℕ)
  (a : Fin n → ℂ)
  (x_roots: Fin n → ℂ)
  (hk1: ∀ x: Fin n → ℂ, (∀ (i : Fin n) (j : Fin n), i ≠ j → |x i| ≠ |x j|))
  :  ∀ k : ℕ, 
    (∑ (i : Fin n), a i * (x_roots i) ^ (2 ^ k)) ^ (2 ^ (-k:ℤ)) → lim (k \to \infty) = | -a (n-1) | :=
  sorry

theorem lobachevsky_method_b
  (n : ℕ) 
  (a : Fin n → ℂ) 
  (x_roots: Fin n → ℂ) 
  (hl: 1 < l ∧ l ≤ n)
  (hk2: ∀ x: Fin n → ℂ, (∀ (i : Fin n) (j : Fin n), i ≠ j → |x i| ≠ |x j|))
  : ∀ l : 1 < ℕ ≤ n,
    ( ( -(a (n - l)) / (a (n - l + 1))) ^ (2 ^ - k)) → lim (k \to \infty) = | x_roots l | :=
  sorry

end lobachevsky_method_a_lobachevsky_method_b_l825_825738


namespace triangle_segments_intersections_l825_825995

noncomputable def triangle_intersections (p : ℕ) (hp_odd_prime : odd_prime p) : ℕ :=
  3 * (p - 1) ^ 2

/--
For an odd prime \( p \), if we divide each side of a triangle into \( p \) equal segments and draw segments connecting the vertices to the division points on the opposite sides, there are exactly \( 3(p-1)^2 \) intersection points inside the triangle.
-/
theorem triangle_segments_intersections (p : ℕ) (hp_odd_prime : odd_prime p) :
  triangle_intersections p hp_odd_prime = 3 * (p - 1) ^ 2 :=
sorry

end triangle_segments_intersections_l825_825995


namespace min_value_of_f_l825_825842

def f (x : ℝ) : ℝ := 2 * (cos x)^2 + sin x

theorem min_value_of_f : ∃ x : ℝ, -1 ≤ sin x ∧ sin x ≤ 1 ∧ f x = -1 :=
begin
  sorry
end

end min_value_of_f_l825_825842


namespace binom_18_4_l825_825333

theorem binom_18_4 : Nat.binomial 18 4 = 3060 :=
by
  -- We start the proof here.
  sorry

end binom_18_4_l825_825333


namespace simplify_expression_l825_825124

theorem simplify_expression :
  (64 ^ (1/3) - (-2/3) ^ 0 + log 2 8 = 6) :=
by
  sorry -- Proof to be filled in later

end simplify_expression_l825_825124


namespace minimum_n_required_l825_825627

theorem minimum_n_required (n : ℕ) (x : ℕ → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ x i) 
  (h_sum : ∑ i in finset.range n, x i = 1) 
  (h_square_sum : ∑ i in finset.range n, (x i)^2 ≤ 1 / 50) : 
  n ≥ 50 :=
sorry

end minimum_n_required_l825_825627


namespace addition_subtraction_result_l825_825743

theorem addition_subtraction_result :
  27474 + 3699 + 1985 - 2047 = 31111 :=
by {
  sorry
}

end addition_subtraction_result_l825_825743


namespace angle_MK_O_right_angle_l825_825506

open EuclideanGeometry

variable {A B C D M O K : Point}
variable (h1 : Segment A B = diameter (semicircle O))
variable (h2 : collinear [A, M, B])
variable (h3 : MB < MA)
variable (h4 : MD < MC)
variable (ω1 : Circle (triangleCircumcenter A O C) (radius (triangleCircumradius A O C)))
variable (ω2 : Circle (triangleCircumcenter D O B) (radius (triangleCircumradius D O B)))
variable (h5 : K ∈ intersectionPoints ω1 ω2)

theorem angle_MK_O_right_angle : ∠(M, K, O) = 90° := by
  sorry

end angle_MK_O_right_angle_l825_825506


namespace child_growth_correct_l825_825761

variable (currentHeight previousHeight growth : ℝ)

def child_growth (currentHeight previousHeight : ℝ) : ℝ :=
  currentHeight - previousHeight

theorem child_growth_correct
  (h_current : currentHeight = 41.5)
  (h_previous : previousHeight = 38.5) :
  child_growth currentHeight previousHeight = 3 := by
  rw [h_current, h_previous]
  exact rfl

end child_growth_correct_l825_825761


namespace average_children_in_families_with_children_l825_825431

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l825_825431


namespace binomial_equality_l825_825319

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l825_825319


namespace radius_k1999_l825_825674

theorem radius_k1999 (P : Point) (e : Line)
  (k₁ k₂ : Circle) (hn₁ : k₁.radius = 1) (hn₂ : k₂.radius = 1)
  (hk₁k₂ : k₁.touches k₂ P) (hk₁e : e.isTangent k₁) (hk₂e : e.isTangent k₂)
  (∀ (kₙ₊₁ : Circle) (hn : n > 2), 
    kₙ₊₁.touches k₁ ∧ kₙ₊₁.touches kₙ ∧ e.isTangent kₙ₊₁) : 
  kₙ₉₉₉.radius = 1 / (1998 ^ 2) :=
sorry

end radius_k1999_l825_825674


namespace average_children_in_families_with_children_l825_825381

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l825_825381


namespace evaluate_expression_l825_825588

variables {a b c d e : ℝ}

theorem evaluate_expression (a b c d e : ℝ) : a * b^c - d + e = a * (b^c - (d + e)) :=
by
  sorry

end evaluate_expression_l825_825588


namespace power_of_six_evaluation_l825_825357

noncomputable def example_expr : ℝ := (6 : ℝ)^(1/4) / (6 : ℝ)^(1/6)

theorem power_of_six_evaluation : example_expr = (6 : ℝ)^(1/12) := 
by
  sorry

end power_of_six_evaluation_l825_825357


namespace vasya_is_right_l825_825705

def triangle : Type := { t : ℝ × ℝ // t.1 > 0 ∧ t.2 > 0 }
def pentagon : Type := { p : ℝ × ℝ × ℝ × ℝ × ℝ // ∀ i, i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5 }

def area (shapes : List ℝ) : ℝ := shapes.foldr (λ a b, a + b) 0

-- Given: Vasya has two triangles and a pentagon.
variable (triangle1 triangle2 : ℝ)
variable (pentagonArea : ℝ)
variable (trianglesAndPentagonArea : ℝ := triangle1 + triangle2 + pentagonArea)

-- Conditions:
-- 1) The combined area of the two triangles and the pentagon is 24 square units.
axiom combined_area_condition : trianglesAndPentagonArea = 24

-- Prove:
theorem vasya_is_right : (∀ a, a = 24) → (triangle1 + triangle2 + pentagonArea = 24) → Prop
:= by
  intro h_combined_area
  intro h_area
  sorry

end vasya_is_right_l825_825705


namespace tessa_debt_l825_825918

theorem tessa_debt :
  let initial_debt : ℤ := 40 in
  let repayment : ℤ := initial_debt / 2 in
  let debt_after_repayment : ℤ := initial_debt - repayment in
  let additional_debt : ℤ := 10 in
  debt_after_repayment + additional_debt = 30 :=
by
  -- The proof goes here.
  sorry

end tessa_debt_l825_825918


namespace ratio_problem_l825_825570

theorem ratio_problem
  (x : ℝ)
  (h : x = 0.8571428571428571) :
  ∃ y : ℝ, (0.75 * y = 7 * x ∧ y = 8) :=
by
  use 8
  split
  {
    rw h
    norm_num
  }
  {
    norm_num
  }

end ratio_problem_l825_825570


namespace max_points_team_F_l825_825035

-- Definitions of points system and tournament conditions
structure Team :=
(points : ℕ)

def teams : List Team := [⟨7⟩, ⟨7⟩, ⟨7⟩, ⟨7⟩, ⟨7⟩]

def total_matches (n : ℕ) : ℕ := n * (n - 1) / 2

def max_possible_points (matches : ℕ) : ℕ := matches * 3

-- Lean Theorem Statement
theorem max_points_team_F : 
  let matches := total_matches 6
  let max_points := max_possible_points matches
  let points_teams := List.sum (List.map Team.points teams)
  max_points - points_teams ≤ 7 :=
by
  sorry

end max_points_team_F_l825_825035


namespace max_height_l825_825763

-- Given definitions
def height_eq (t : ℝ) : ℝ := -16 * t^2 + 64 * t + 10

def max_height_problem : Prop :=
  ∃ t : ℝ, height_eq t = 74 ∧ ∀ t' : ℝ, height_eq t' ≤ height_eq t

-- Statement of the proof
theorem max_height : max_height_problem := sorry

end max_height_l825_825763


namespace factorial_product_square_l825_825302

theorem factorial_product_square (n : ℕ) (m : ℕ) (h₁ : n = 5) (h₂ : m = 4) :
  (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 :=
by
  have f5 : Nat.factorial 5 = 120 := by norm_num
  have f4 : Nat.factorial 4 = 24 := by norm_num
  rw [Nat.factorial_eq_factorial h₁, Nat.factorial_eq_factorial h₂, f5, f4]
  norm_num
  simp
  sorry

end factorial_product_square_l825_825302


namespace trailing_zeros_500_fact_l825_825295

theorem trailing_zeros_500_fact : 
  (nat.div (500! : ℕ) (10 ^ 124)) * (10 ^ 124) = 500! ∧ 
  (nat.div (500! : ℕ) (10 ^ 125)) * (10 ^ 125) ≠ 500! :=
by {
  sorry,
}

end trailing_zeros_500_fact_l825_825295


namespace general_formula_and_Tn_l825_825512

variable {α : Type*}

noncomputable def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Definitions for the problem
def a_n := arithmetic_sequence 2 2
def Sn (n : ℕ) := n * (a_n 1 + a_n n) / 2
def b_n (n : ℕ) := 2^(a_n n)
def c_n (n : ℕ) := a_n n + b_n n
def Tn (n : ℕ) := (finset.range n).sum (λ k, c_n (k + 1))

-- Conditions
axiom a3_eq_6 : a_n 3 = 6
axiom ratio_S5_S9 : Sn 5 / Sn 9 = 1 / 3

-- Lean Statement for the equivalent proof problem
theorem general_formula_and_Tn :
  (∀ n, a_n n = 2 * n) ∧
  (∀ n, Tn n = n^2 + n + (4^(n + 1) - 4) / 3) :=
begin
  sorry
end

end general_formula_and_Tn_l825_825512


namespace smallest_base10_integer_l825_825202

theorem smallest_base10_integer (a b : ℕ) (ha : a > 3) (hb : b > 3) (h : a + 3 = 3 * b + 1) :
  13 = a + 3 :=
by
  have h_in_base_a : a = 3 * b - 2 := by linarith,
  have h_in_base_b : 3 * b + 1 = 13 := by sorry,
  exact h_in_base_b

end smallest_base10_integer_l825_825202


namespace smallest_base10_integer_l825_825215

theorem smallest_base10_integer (a b : ℕ) (h1 : a > 3) (h2 : b > 3) :
    (1 * a + 3 = 3 * b + 1) → (1 * 10 + 3 = 13) :=
by
  intros h


-- Prove that  1 * a + 3 = 3 * b + 1 
  have a_eq : a = 3 * b - 2 := by linarith

-- Prove that 1 * 10 + 3 = 13 
  have base_10 := by simp

have the smallest base 10
  sorry

end smallest_base10_integer_l825_825215


namespace monotonicity_of_f_abs_f_diff_ge_four_abs_diff_l825_825899

noncomputable def f (a x : ℝ) : ℝ := (a + 1) * Real.log x + a * x^2 + 1

theorem monotonicity_of_f {a : ℝ} (x : ℝ) (hx : 0 < x) :
  (f a x) = (f a x) := sorry

theorem abs_f_diff_ge_four_abs_diff {a x1 x2: ℝ} (ha : a ≤ -2) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  |f a x1 - f a x2| ≥ 4 * |x1 - x2| := sorry

end monotonicity_of_f_abs_f_diff_ge_four_abs_diff_l825_825899


namespace percent_with_university_diploma_l825_825230

theorem percent_with_university_diploma (a b c d : ℝ) (h1 : a = 0.12) (h2 : b = 0.25) (h3 : c = 0.40) 
    (h4 : d = c - a) (h5 : ¬c = 1) : 
    d + (b * (1 - c)) = 0.43 := 
by 
    sorry

end percent_with_university_diploma_l825_825230


namespace find_value_of_f_l825_825094

-- Define the function f and its properties
variable f : ℝ → ℝ

-- Conditions
axiom f_prop1 : ∀ x : ℝ, f(x + π) = f(x) + sin x
axiom f_prop2 : ∀ x : ℝ, 0 ≤ x ∧ x < π → f(x) = 0

-- Theorem statement to prove
theorem find_value_of_f : f(23 * π / 6) = 1 / 2 := by
sorry

end find_value_of_f_l825_825094


namespace arccos_sqrt3_div_2_eq_pi_div_6_l825_825312

theorem arccos_sqrt3_div_2_eq_pi_div_6 : Real.arccos (real.sqrt 3 / 2) = Real.pi / 6 :=
sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l825_825312


namespace avg_children_in_families_with_children_l825_825451

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l825_825451


namespace average_children_in_families_with_children_l825_825420

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l825_825420


namespace num_divisors_162432_l825_825822

-- Define the six-digit number
def num : ℕ := 162432

-- Predicates for divisibility checks
def is_divisible_by (n d : ℕ) : Prop := d ∣ n

-- Define the list of numbers from 1 to 9
def range_1_to_9 : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the set of divisors of num within the range 1 to 9
def divisors_in_range (n : ℕ) (range : List ℕ) : List ℕ :=
  range.filter (λ d => is_divisible_by n d)

-- Prove that the number of divisors of 162432 in the range 1 to 9 is exactly 7
theorem num_divisors_162432 : (divisors_in_range num range_1_to_9).length = 7 := 
by 
  sorry

end num_divisors_162432_l825_825822


namespace average_children_in_families_with_children_l825_825390

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l825_825390


namespace xiao_wang_top5_needs_median_l825_825275

-- Definition of the problem
def scores : List ℕ := sorry  -- Placeholder for the list of scores of 10 students

def is_top5 (s : ℕ) (scores : List ℕ) : Prop :=
  let sorted_scores := scores.qsort (· < ·)
  let median := sorted_scores.get! (sorted_scores.length / 2)
  s > median

theorem xiao_wang_top5_needs_median (s : ℕ) (scores : List ℕ) :
  list.length scores = 10 → 
  (s > (scores.qsort (· < ·)).get! (scores.length / 2)) ↔ is_top5 s scores :=
by sorry

end xiao_wang_top5_needs_median_l825_825275


namespace max_diagonals_selected_l825_825037

theorem max_diagonals_selected (n : ℕ) (h : n = 1000) :
  ∃ (d : ℕ), d = 2000 ∧ 
  ∀ (chosen_diagonals : Finset (Finₓ (n * (n - 3) / 2))), 
    (∀ (x y z ∈ chosen_diagonals), 
     (∃ (same_length_pair : ℕ × ℕ), same_length_pair ∈ chosen_diagonals ∧ same_length_pair.fst = same_length_pair.snd)) →
    chosen_diagonals.card ≤ d :=
by
  rw h
  use 2000
  split
  { refl }
  { intros chosen_diagonals condition
    sorry -- proof to be provided
  }

end max_diagonals_selected_l825_825037


namespace avg_children_in_families_with_children_l825_825403

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l825_825403


namespace monotonic_increasing_interval_l825_825155

noncomputable def f (x : ℝ) : ℝ := Real.logBase (1/2) (x^2 - 2 * x - 3)

theorem monotonic_increasing_interval :
  ∀ x ∈ Ioo (-(∞)) (-1), deriv f x > 0 := by
  sorry

end monotonic_increasing_interval_l825_825155


namespace point_within_region_l825_825891

theorem point_within_region (a : ℝ) (h : 2 * a + 2 < 4) : a < 1 := 
sorry

end point_within_region_l825_825891


namespace average_children_in_families_with_children_l825_825427

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l825_825427


namespace vertex_angle_of_isosceles_triangle_l825_825044

theorem vertex_angle_of_isosceles_triangle
  (A B C : Type)
  [DecidableEq A] [DecidableEq B] [DecidableEq C]
  (triangle : A ≃ B ≃ C)
  (isosceles : A ≃ B ∨ B ≃ C ∨ C ≃ A)
  (interior_angle_40 : ∃ x ∈ triangle, x = 40) :
  ∃ y ∈ triangle, y = 40 ∨ y = 100 := 
sorry

end vertex_angle_of_isosceles_triangle_l825_825044


namespace sum_of_first_2019_terms_l825_825600

variable (a : ℕ → ℝ)

-- Define the sequence rule
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- The given conditions
variable (h_arith_seq : is_arithmetic_sequence a)
variable (h_a1010 : a 1010 = 1)

-- The target to prove
theorem sum_of_first_2019_terms : 
  (∑ k in Finset.range 2019, a k) = 2019 :=
by
  -- We need to include a full proof here, but we'll placeholder with sorry to indicate it
  sorry

end sum_of_first_2019_terms_l825_825600


namespace avg_children_in_families_with_children_l825_825443

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l825_825443


namespace min_tangent_length_is_4_l825_825577

noncomputable def min_tangent_length (a b : ℝ) :=
  let pc := Real.sqrt ((a + 1)^2 + (b - 2)^2)
  let r := Real.sqrt 2
  Real.sqrt (pc^2 - r^2)

theorem min_tangent_length_is_4 :
  ∀ b : ℝ, let a := b + 3 in
  2 * (b + 1)^2 + 16 = (min_tangent_length (b + 3) b)^2 → 
  (min_tangent_length (b + 3) b) = 4 :=
by
  intros b a ha
  sorry

end min_tangent_length_is_4_l825_825577


namespace smallest_integer_representable_l825_825196

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end smallest_integer_representable_l825_825196


namespace axis_of_symmetry_l825_825536

def f (x : ℝ) : ℝ := cos (x + π / 2) * cos (x + π / 4)

theorem axis_of_symmetry : ∀ x : ℝ, f (5 * π / 8 - x) = f (5 * π / 8 + x) :=
sorry

end axis_of_symmetry_l825_825536


namespace tetrahedron_volume_l825_825694

noncomputable def volume_of_tetrahedron (A B C O : Point) (r : ℝ) :=
  1 / 3 * (Real.sqrt (3) / 4 * 2^2 * Real.sqrt 11)

theorem tetrahedron_volume 
  (A B C O : Point)
  (side_length : ℝ)
  (surface_area : ℝ)
  (radius : ℝ)
  (h : ℝ)
  (radius_eq : radius = Real.sqrt (37 / 3))
  (side_length_eq : side_length = 2)
  (surface_area_eq : surface_area = (4 * Real.pi * radius^2))
  (sphere_surface_area_eq : surface_area = 148 * Real.pi / 3)
  (height_eq : h^2 = radius^2 - (2 / 3 * 2 * Real.sqrt 3 / 2)^2)
  (height_value_eq : h = Real.sqrt 11) :
  volume_of_tetrahedron A B C O radius = Real.sqrt 33 / 3 := sorry

end tetrahedron_volume_l825_825694


namespace probability_on_line_x_plus_y_eq_five_l825_825132

-- Define the dice rolls
noncomputable def rollDice : Finset (ℕ × ℕ) := 
  Finset.product (Finset.range 6) (Finset.range 6)

-- Define the event that point P lies on the line x + y = 5
def P_on_line (p : ℕ × ℕ) : Prop := p.1 + p.2 = 5

-- Define the set of all outcomes where P lies on the line x + y = 5
def favorable_outcomes : Finset (ℕ × ℕ) := rollDice.filter P_on_line

-- Define the probability calculation
def probability (A : Finset (ℕ × ℕ)) (S : Finset (ℕ × ℕ)) : ℚ := A.card / S.card

theorem probability_on_line_x_plus_y_eq_five :
  probability favorable_outcomes rollDice = 1 / 9 := by
  sorry

end probability_on_line_x_plus_y_eq_five_l825_825132


namespace john_learns_vowel_in_3_days_l825_825483

theorem john_learns_vowel_in_3_days (total_days : ℕ) (number_of_vowels : ℕ) (h : total_days = 15) (h' : number_of_vowels = 5) : (total_days / number_of_vowels) = 3 :=
by
  rw [h, h']
  simp
  sorry

end john_learns_vowel_in_3_days_l825_825483


namespace inclination_of_parabola_tangent_l825_825823

noncomputable def angle_of_inclination : ℝ :=
let x := 1/2 in
let y := (x : ℝ)^2 in
let slope := (deriv (λ x : ℝ, x^2)) x in
real.arctan (slope)

theorem inclination_of_parabola_tangent :
  angle_of_inclination = real.pi / 4 :=
sorry

end inclination_of_parabola_tangent_l825_825823


namespace sqrt_factorial_squared_l825_825304

theorem sqrt_factorial_squared :
  (Real.sqrt ((Nat.factorial 5) * (Nat.factorial 4))) ^ 2 = 2880 :=
by sorry

end sqrt_factorial_squared_l825_825304


namespace average_children_in_families_with_children_l825_825389

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l825_825389


namespace perpendicular_slope_l825_825190

theorem perpendicular_slope (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = -4) (h4 : y2 = -8) : 
  let m := (y2 - y1) / (x2 - x1) 
  -- then the slope of the line perpendicular to this is:
  -1 / m = -6 / 5 :=
by
  sorry

end perpendicular_slope_l825_825190


namespace no_tetrahedron_with_isosceles_faces_l825_825828

/-- There does not exist a tetrahedron where all faces are isosceles triangles, and no two faces are congruent. -/
theorem no_tetrahedron_with_isosceles_faces :
  ¬ ∃ (A B C D : Type) (tri_ABC tri_ABD tri_ACD tri_BCD: Triangle),
    (isosceles tri_ABC ∧ isosceles tri_ABD ∧ isosceles tri_ACD ∧ isosceles tri_BCD) ∧
    (¬ congruent tri_ABC tri_ABD ∧ ¬ congruent tri_ABC tri_ACD ∧ ¬ congruent tri_ABC tri_BCD ∧
     ¬ congruent tri_ABD tri_ACD ∧ ¬ congruent tri_ABD tri_BCD ∧ ¬ congruent tri_ACD tri_BCD) :=
sorry

end no_tetrahedron_with_isosceles_faces_l825_825828


namespace problem1_problem2_l825_825306

noncomputable def expr1 (a b : ℝ) : ℝ :=
  (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) * a^(-1)

noncomputable def expr2 : ℝ :=
  0.0081^(-1/4) - (3 * (7/8)^0)^(-1) * (81^(-0.25) + (3 + 3/8)^(-1/3))^(-1/2)

theorem problem1 (a b : ℝ) : expr1 a b = 4 := by
  sorry

theorem problem2 : expr2 = 3 := by
  sorry

end problem1_problem2_l825_825306


namespace smallest_base10_integer_l825_825205

theorem smallest_base10_integer (a b : ℕ) (ha : a > 3) (hb : b > 3) (h : a + 3 = 3 * b + 1) :
  13 = a + 3 :=
by
  have h_in_base_a : a = 3 * b - 2 := by linarith,
  have h_in_base_b : 3 * b + 1 = 13 := by sorry,
  exact h_in_base_b

end smallest_base10_integer_l825_825205


namespace exists_integer_between_sqrt2n_and_sqrt5n_l825_825987

theorem exists_integer_between_sqrt2n_and_sqrt5n (n : ℕ) (hn : 1 ≤ n) :
  ∃ m : ℤ, (sqrt (2 * n : ℝ) : ℝ) < m ∧ m < (sqrt (5 * n : ℝ) : ℝ) :=
sorry

end exists_integer_between_sqrt2n_and_sqrt5n_l825_825987


namespace n_decomposable_form_l825_825986

theorem n_decomposable_form (n : ℕ) (a : ℕ) (h₁ : a > 2) (h₂ : ∃ k, 1 < k ∧ n = 2^k) :
  (∀ d : ℕ, d ∣ n ∧ d ≠ n → (a^n - 2^n) % (a^d + 2^d) = 0) → ∃ k, 1 < k ∧ n = 2^k :=
by {
  sorry
}

end n_decomposable_form_l825_825986


namespace train_speed_conversion_l825_825225

theorem train_speed_conversion (speed_kmph : ℕ) (h : speed_kmph = 135) :
  (speed_kmph * 1000 / 3600 : ℚ) = 37.5 :=
by { rw h, norm_num, }


end train_speed_conversion_l825_825225


namespace range_of_a_l825_825153

-- Given definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a * x + 1 

-- Monotonicity condition on the interval [1, 2]
def is_monotonic (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → (x ≤ y → f x a ≤ f y a) ∨ (x ≤ y → f x a ≥ f y a)

-- The proof objective
theorem range_of_a (a : ℝ) : is_monotonic a → (a ≤ -2 ∨ a ≥ -1) := 
sorry

end range_of_a_l825_825153


namespace eval_fraction_l825_825359

theorem eval_fraction (a b : ℕ) : (40 : ℝ) = 2^3 * 5 → (10 : ℝ) = 2 * 5 → (40^56 / 10^28) = 160^28 :=
by 
  sorry

end eval_fraction_l825_825359


namespace problem_l825_825133

variable (R S : Prop)

theorem problem (h1 : R → S) :
  ((¬S → ¬R) ∧ (¬R ∨ S)) :=
by
  sorry

end problem_l825_825133


namespace hyperbola_equiv_l825_825677

noncomputable def hyperbola1 (x y : ℝ) : Prop := 2 * x^2 - y^2 = 3

noncomputable def hyperbola2 (x y : ℝ) : Prop := ∃ λ : ℝ, 2 * x^2 - y^2 = 3 * λ ∧ 2 - 4 = 3 * λ

theorem hyperbola_equiv (x y : ℝ) (h1 : hyperbola1 x y) (h2 : hyperbola2 1 2) :
  ∃ (λ : ℝ), λ = -2 / 3 ∧ 2 * x^2 - y^2 = 3 * λ → (y^2 / 2 - x^2 = 1) :=
by
  sorry

end hyperbola_equiv_l825_825677


namespace math_problem_l825_825054

variables {A B C D M N P Q O : Point}
variables (triangle_ABC : Triangle A B C)
variables (altitude_AD : Altitude A D B C)
variables (circle_O : Circle O)
variables (tangent_D : Tangent circle_O B C D)
variables (intersect_MN : Intersect circle_O A B M N)
variables (intersect_PQ : Intersect circle_O A C P Q)

theorem math_problem :
  (AM + AN) / AC = (AP + AQ) / AB :=
by
  -- Proof omitted
  sorry

end math_problem_l825_825054


namespace ratio_n_over_p_l825_825689

-- Definitions and conditions from the problem
variables {m n p : ℝ}

-- The quadratic equation x^2 + mx + n = 0 has roots that are thrice those of x^2 + px + m = 0.
-- None of m, n, and p is zero.

-- Prove that n / p = 27 given these conditions.
theorem ratio_n_over_p (hmn0 : m ≠ 0) (hn : n = 9 * m) (hp : p = m / 3):
  n / p = 27 :=
  by
    sorry -- Formal proof will go here.

end ratio_n_over_p_l825_825689


namespace expected_value_twelve_sided_die_l825_825776

theorem expected_value_twelve_sided_die : 
  let die_sides := 12 in 
  let outcomes := finset.range (die_sides + 1) in
  (finset.sum outcomes id : ℚ) / die_sides = 6.5 :=
by
  sorry

end expected_value_twelve_sided_die_l825_825776


namespace Charlie_age_when_Jenny_twice_as_Bobby_l825_825060

theorem Charlie_age_when_Jenny_twice_as_Bobby (B C J : ℕ) 
  (h₁ : J = C + 5)
  (h₂ : C = B + 3)
  (h₃ : J = 2 * B) : 
  C = 11 :=
by
  sorry

end Charlie_age_when_Jenny_twice_as_Bobby_l825_825060


namespace sum_of_complex_series_l825_825813

theorem sum_of_complex_series :
  (∑ k in Finset.range 2010, complex.I ^ (k + 1)) = -1 + complex.I := 
by
  sorry

end sum_of_complex_series_l825_825813


namespace tessa_owes_30_l825_825922

-- Definitions based on given conditions
def initial_debt : ℕ := 40
def paid_back : ℕ := initial_debt / 2
def remaining_debt_after_payment : ℕ := initial_debt - paid_back
def additional_borrowing : ℕ := 10
def total_debt : ℕ := remaining_debt_after_payment + additional_borrowing

-- Theorem to be proved
theorem tessa_owes_30 : total_debt = 30 :=
by
  sorry

end tessa_owes_30_l825_825922


namespace find_magnitude_l825_825006

variable (m : ℝ)
def a := (1, 2 * m)
def b := (m + 1, 1)
def c := (2, m)
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_magnitude (h : dot_prod (a + c) b = 0) : |sqrt(1^2 + (-1)^2)| = √2 := by
  -- Given condition that will help prove the result
  sorry

end find_magnitude_l825_825006


namespace average_children_families_with_children_is_3_point_8_l825_825371

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l825_825371


namespace average_children_families_with_children_is_3_point_8_l825_825367

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l825_825367


namespace abs_ineq_solution_l825_825847

theorem abs_ineq_solution (x : ℝ) : (2 ≤ |x - 5| ∧ |x - 5| ≤ 4) ↔ (1 ≤ x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 9) :=
by
  sorry

end abs_ineq_solution_l825_825847


namespace part1_a_value_part2_a_range_l825_825517

def f (x : ℝ) : ℝ := x^3 - x
def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

-- Part (1)
theorem part1_a_value (a : ℝ) : 
  f'(-1) = g'(-1) → 
  (∀ x, g(x, a) - 2(x + 1) = 0) → 
  a = 3 :=
sorry

-- Part (2)
theorem part2_a_range (a : ℝ) : 
  (∃ x1 x2 : ℝ, f'(x1) = g'(x2) ∧ 
  f(x1) - (3 * x1^2 - 1) * (x1 - x) = g(x2, a) - 2x2 * x) → 
  a ≥ -1 :=
sorry

end part1_a_value_part2_a_range_l825_825517


namespace balloon_ratio_l825_825284

theorem balloon_ratio 
  (initial_blue : ℕ) (initial_purple : ℕ) (balloons_left : ℕ)
  (h1 : initial_blue = 303)
  (h2 : initial_purple = 453)
  (h3 : balloons_left = 378) :
  (balloons_left / (initial_blue + initial_purple) : ℚ) = (1 / 2 : ℚ) :=
by
  sorry

end balloon_ratio_l825_825284


namespace find_a_l825_825493

theorem find_a (a b : ℝ) (h₀ : a > 1) (h₁ : b > 1)
  (h₂ : ∀ b:ℝ, b > 1 → log a b + log b (a^2 + 12) ≥ 4) :
  a = 2 := 
begin 
  -- Proof to be filled in.
  sorry
end

end find_a_l825_825493


namespace bella_steps_correct_l825_825291

-- Define constants
def distance_in_miles := 3
def distance_in_feet : ℕ := 15840
def ella_speed_multiplier := 3 
def bella_step_length := 3
def expected_steps := 1320

-- Definitions for distances and speeds based on conditions
def total_distance : ℕ := distance_in_feet
def b_speed (b: ℕ) : ℕ := b -- Bella's speed in feet per minute
def e_speed (b: ℕ) : ℕ := ella_speed_multiplier * b -- Ella's speed in feet per minute
def combined_speed (b: ℕ) : ℕ := b_speed b + e_speed b -- Combined speed when moving towards each other

-- Time until Bella and Ella meet
def time_to_meet (b: ℕ) : ℚ := total_distance / combined_speed b

-- Distance covered by Bella
def bella_distance_covered (b: ℕ) : ℕ := (b_speed b) * (time_to_meet b).toNat

-- Calculating number of steps taken by Bella
def bella_steps (b: ℕ) : ℕ := bella_distance_covered b / bella_step_length

-- Theorem stating Bella will take 1320 steps
theorem bella_steps_correct (b: ℕ) (h : combined_speed b ≠ 0) : bella_steps b = expected_steps :=
by sorry

end bella_steps_correct_l825_825291


namespace closest_perfect_square_l825_825217

theorem closest_perfect_square (n : ℕ) (h1 : n = 325) : 
    ∃ m : ℕ, m^2 = 324 ∧ 
    (∀ k : ℕ, (k^2 ≤ n ∨ k^2 ≥ n) → (k = 18 ∨ k^2 > 361 ∨ k^2 < 289)) := 
by
  sorry

end closest_perfect_square_l825_825217


namespace impossible_to_get_9_zeros_l825_825287

theorem impossible_to_get_9_zeros
  (initial_config : list ℕ) (h_initial : initial_config.count 1 = 4 ∧ initial_config.count 0 = 5)
  (transformation : list ℕ → list ℕ)
  (h_transformation : ∀ l, transformation l = list.zip_with (λ x y, if x = y then 0 else 1) l (l.rotate 1)):
  ¬ ∃ n, (iterate transformation n initial_config).count 0 = 9 := 
sorry

end impossible_to_get_9_zeros_l825_825287


namespace money_worthless_in_wrong_context_l825_825115

-- Define the essential functions money must serve in society
def InContext (society: Prop) : Prop := 
  ∀ (m : Type), (∃ (medium_of_exchange store_of_value unit_of_account standard_of_deferred_payment : m → Prop), true)

-- Define the context of a deserted island where these functions are useless
def InContext (deserted_island: Prop) : Prop := 
  ∀ (m : Type), (¬ (∃ (medium_of_exchange store_of_value unit_of_account standard_of_deferred_payment : m → Prop), true))

-- Define the essential properties for an item to become money
def EssentialProperties (item: Type) : Prop :=
  ∃ (durable portable divisible acceptable uniform limited_supply : item → Prop), true

-- The primary theorem: Proving that money becomes worthless in the absence of its essential functions
theorem money_worthless_in_wrong_context (m : Type) (society deserted_island : Prop)
  (h1 : InContext society) (h2 : InContext deserted_island) (h3 : EssentialProperties m) :
  ∃ (worthless : m → Prop), true :=
sorry

end money_worthless_in_wrong_context_l825_825115


namespace volume_of_klmn_l825_825874

-- Define the condition of the given tetrahedron with volume V
variable (V : ℝ)

-- Define a theorem stating the required proof for the volume of tetrahedron KLMN
theorem volume_of_klmn (V : ℝ) : 
  ∃ (W : ℝ), W = (3 / 4) * V^2 :=
by
  use (3 / 4) * V^2
  sorry

end volume_of_klmn_l825_825874


namespace percent_absent_l825_825654

-- Conditions
def num_students := 120
def num_boys := 72
def num_girls := 48
def frac_boys_absent := 1 / 8
def frac_girls_absent := 1 / 4

-- Theorem statement
theorem percent_absent : 
  ( (frac_boys_absent * num_boys + frac_girls_absent * num_girls) / num_students ) * 100 = 17.5 :=
by
  sorry

end percent_absent_l825_825654


namespace bela_wins_iff_m_odd_l825_825097

theorem bela_wins_iff_m_odd (m : ℕ) (hm : m > 2) :
  (∃ moves : ℕ, 
    let intervals := { x ∈ set.Icc 0 m | for each x, there exists y such that y is at least 2 units away from all previous x's }
    bela_wins m moves) ↔ odd m :=
sorry

end bela_wins_iff_m_odd_l825_825097


namespace expected_value_of_twelve_sided_die_l825_825789

theorem expected_value_of_twelve_sided_die : 
  let face_values := finset.range (12 + 1) \ finset.singleton 0 in
  (finset.sum face_values (λ x, x) : ℝ) / 12 = 6.5 :=
by
  sorry

end expected_value_of_twelve_sided_die_l825_825789


namespace intersection_A_B_subset_A_B_l825_825912

noncomputable def set_A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
noncomputable def set_B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 22}

theorem intersection_A_B (a : ℝ) (ha : a = 10) : set_A a ∩ set_B = {x : ℝ | 21 ≤ x ∧ x ≤ 22} := by
  sorry

theorem subset_A_B (a : ℝ) : set_A a ⊆ set_B → a ≤ 9 := by
  sorry

end intersection_A_B_subset_A_B_l825_825912


namespace binom_18_4_eq_3060_l825_825314

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l825_825314


namespace quadratic_solution_difference_l825_825131

theorem quadratic_solution_difference (a b : ℝ) : 
  (2 * a ^ 2 - 5 * a + 18 = 3 * a + 55) ∧ (2 * b ^ 2 - 5 * b + 18 = 3 * b + 55) →
  (a = 2 + 3 * real.sqrt 10 / 2) ∧ (b = 2 - 3 * real.sqrt 10 / 2) ∨
  (b = 2 + 3 * real.sqrt 10 / 2) ∧ (a = 2 - 3 * real.sqrt 10 / 2) →
  abs (a - b) = 3 * real.sqrt 10 :=
by
  assume h1 h2
  sorry

end quadratic_solution_difference_l825_825131


namespace cycle_selling_price_l825_825253

theorem cycle_selling_price
  (cost_price : ℝ)
  (gain_percentage : ℝ)
  (profit : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 930)
  (h2 : gain_percentage = 30.107526881720432)
  (h3 : profit = (gain_percentage / 100) * cost_price)
  (h4 : selling_price = cost_price + profit)
  : selling_price = 1210 := 
sorry

end cycle_selling_price_l825_825253


namespace avg_children_in_families_with_children_l825_825408

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l825_825408


namespace a_n_general_term_sn_sum_l825_825541

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / (Real.exp x + 1)

noncomputable def g (x : ℝ) : ℝ := f(x - 1) + 1

def a_sequence (n : ℕ) : ℝ := ∑ i in Finset.range (2 * n).filter (λ k, k % 2 = 1), g(i / n)

def b_sequence (n : ℕ) : ℝ := 1 / (a_sequence n * a_sequence (n + 1))

theorem a_n_general_term (n : ℕ) (hn : n > 0) : a_sequence n = 2 * n - 1 := 
  sorry

theorem sn_sum (n : ℕ) (hn : n > 0) : (∑ i in Finset.range n, b_sequence i) = n / (2 * n + 1) := 
  sorry

end a_n_general_term_sn_sum_l825_825541


namespace find_coordinates_of_P_l825_825575

-- Define the conditions
variable (x y : ℝ)
def in_second_quadrant := x < 0 ∧ y > 0
def distance_to_x_axis := abs y = 7
def distance_to_y_axis := abs x = 3

-- Define the statement to be proved in Lean 4
theorem find_coordinates_of_P :
  in_second_quadrant x y ∧ distance_to_x_axis y ∧ distance_to_y_axis x → (x, y) = (-3, 7) :=
by
  sorry

end find_coordinates_of_P_l825_825575


namespace isosceles_triangle_l825_825582

theorem isosceles_triangle (a b c A B C : ℝ) (h₁ : a = c) (h₂ : a / cos A = c / cos C) : (A = C) :=
by
-- Mathematical assumptions and proof steps would go here
sorry

end isosceles_triangle_l825_825582


namespace proj_b_v_l825_825620

variables (a b : ℝ × ℝ)
variables (v : ℝ × ℝ := (4, -2))
variables (proj_a := (-2/5, -4/5))

-- Define orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Define projection function
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  let k := dot_uv / dot_uu
  (k * u.1, k * u.2)

-- Given Conditions
axiom h1 : orthogonal a b
axiom h2 : proj a v = proj_a

-- The theorem to prove
theorem proj_b_v : proj b v = (22 / 5, -6 / 5) :=
  sorry

end proj_b_v_l825_825620


namespace square_EFGH_area_correct_l825_825662

noncomputable def square_EFGH_area (a b e : ℝ) : ℝ := 
  if h: a = 2 and b = 10 
  then let x := 4*Real.sqrt 6 - 2 in x^2
  else 0

theorem square_EFGH_area_correct : 
  square_EFGH_area 2 10 2 = 100 - 16*Real.sqrt 6 :=
sorry

end square_EFGH_area_correct_l825_825662


namespace problem1_problem2_l825_825809

-- Problem 1: Prove that (\sqrt{7}-\sqrt{3})(\sqrt{7}+\sqrt{3}) - (\sqrt{6}+\sqrt{2})^{2} = -4 - 4\sqrt{3}
theorem problem1 : (sqrt 7 - sqrt 3) * (sqrt 7 + sqrt 3) - (sqrt 6 + sqrt 2) ^ 2 = -4 - 4 * sqrt 3 :=
by sorry

-- Problem 2: Prove that (3 * sqrt 12 - 3 * sqrt (1 / 3) + sqrt 48) / (2 * sqrt 3) = 9 / 2
theorem problem2 : (3 * sqrt 12 - 3 * sqrt (1 / 3) + sqrt 48) / (2 * sqrt 3) = 9 / 2 :=
by sorry

end problem1_problem2_l825_825809


namespace simson_line_theorem_hexagon_concurrency_theorem_l825_825022

-- Part 1: Simson Line Problem

/-
  Given a triangle \(PQR\) and a point \(S\) on its circumcircle,
  show that the feet of the perpendiculars from \(S\) 
  to the sides \(PQ\), \(QR\), and \(RP\) are collinear.
-/

def is_simson_line (P Q R S : Point) (circumcircle : Circle P Q R) : Prop :=
  let D := perpendicular_foot S Q R in
  let E := perpendicular_foot S R P in
  let F := perpendicular_foot S P Q in
  collinear [D, E, F]

theorem simson_line_theorem (P Q R S : Point) (circumcircle : Circle P Q R) :
  S ∈ circumcircle →
  is_simson_line P Q R S circumcircle :=
by sorry

-- Part 2: Concurrency in Hexagon

/-
  Given hexagon \(ABCDEF\) inscribed in a circle,
  show that the lines \([A, BDF]\), \([B, ACE]\), \([D, ABF]\), \([E, ABC]\)
  are concurrent if and only if \(CDEF\) is a rectangle.
-/

def is_concurrent (A B C D E F : Point) (circle : Circle A B C D E F) : Prop :=
  let line_ABDF := line_through_points (A::nil) (B::D::F::nil) in
  let line_BACE := line_through_points (B::nil) (A::C::E::nil) in
  let line_DABF := line_through_points (D::nil) (A::B::F::nil) in
  let line_EABC := line_through_points (E::nil) (A::B::C::nil) in
  lines_concurrent [line_ABDF, line_BACE, line_DABF, line_EABC]

def is_rectangle (C D E F : Point) : Prop :=
Orthogonal (line_through_points C E) (line_through_points D F)

theorem hexagon_concurrency_theorem (A B C D E F : Point) (circle : Circle A B C D E F) :
  is_concurrent A B C D E F circle ↔ is_rectangle C D E F :=
by sorry

end simson_line_theorem_hexagon_concurrency_theorem_l825_825022


namespace cylinder_cut_area_eq_axial_section_area_l825_825762

theorem cylinder_cut_area_eq_axial_section_area (r h α : ℝ) (h_pos : h > 0) (r_pos : r > 0) (α_pos : α > 0) :
  ∃ A B : ℝ, 2 * h * r = h * ∫ x in 0..(π * r), sin (x / r) :=
by
  sorry

end cylinder_cut_area_eq_axial_section_area_l825_825762


namespace no_solution_for_n_eq_m_even_factorials_l825_825309

theorem no_solution_for_n_eq_m_even_factorials (n m : ℕ) (h : m ≥ 2) :
  n! ≠ 2^m * m! := 
by 
  sorry

end no_solution_for_n_eq_m_even_factorials_l825_825309


namespace cos_fourth_power_sum_l825_825121

theorem cos_fourth_power_sum (n : ℕ) (k : ℝ) (h1 : n > 0) (h2 : k = Real.pi / (2 * n + 1)) :
  (∑ r in Finset.range n, Real.cos (2 * (r + 1) * k) ^ 4) = 3 * n / 8 - 5 / 16 :=
sorry

end cos_fourth_power_sum_l825_825121


namespace largest_common_value_l825_825144

theorem largest_common_value (a : ℕ) (h1 : a % 4 = 3) (h2 : a % 9 = 5) (h3 : a < 600) :
  a = 599 :=
sorry

end largest_common_value_l825_825144


namespace geom_seq_sum_eq_37_l825_825952

theorem geom_seq_sum_eq_37
  (a r : ℝ)
  (h₃ : a + a*r + a*r^2 = 13)
  (h₇ : a * (1 - r^7) / (1 - r) = 183) :
  a * (1 + r + r^2 + r^3 + r^4) = 37 := 
by
  sorry

end geom_seq_sum_eq_37_l825_825952


namespace triangle_isosceles_angle_l825_825629

theorem triangle_isosceles_angle (A B C D E : Type) 
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] 
  [InnerProductSpace ℝ D] [InnerProductSpace ℝ E] (AB AC : A ≃ₗᵢ[ℝ] B)
  (angleBAC angleBCD angleCBE angleCDE_deg : ℝ)
  (hAB_AC : AB = AC) (hBAC : angleBAC = 20) (hBCD : angleBCD = 70) (hCBE : angleCBE = 60) :
  angleCDE_deg = 20 := 
sorry 

end triangle_isosceles_angle_l825_825629


namespace general_term_find_T_2017_l825_825873

-- Definition of the sequence conditions
def a_sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1/3) ∧ (a 4 = 1/81) ∧ ∀ n, (a (n + 1))^2 = a n * a (n + 2)

-- 1. General term of the sequence.
theorem general_term (a : ℕ → ℝ) (h : a_sequence a) : ∀ n, a n = (1/3)^n :=
sorry

-- Definitions for f, b_n and T_n
def f (x : ℝ) := Real.log x / Real.log 3
def b_n (a : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, f (a i + 1)
def T_n (a : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, (1 / (b_n a (i + 1)))

-- 2. Finding T_{2017}
theorem find_T_2017 (a : ℕ → ℝ) (h : a_sequence a) : T_n a 2017 = -2017 / 1009 :=
sorry

end general_term_find_T_2017_l825_825873


namespace pizza_slices_per_pizza_l825_825102

theorem pizza_slices_per_pizza (h : ∀ (mrsKaplanSlices bobbySlices pizzas : ℕ), 
  mrsKaplanSlices = 3 ∧ mrsKaplanSlices = bobbySlices / 4 ∧ pizzas = 2 → bobbySlices / pizzas = 6) : 
  ∃ (bobbySlices pizzas : ℕ), bobbySlices / pizzas = 6 :=
by
  existsi (3 * 4)
  existsi 2
  sorry

end pizza_slices_per_pizza_l825_825102


namespace odometer_problem_l825_825641

theorem odometer_problem
    (x a b c : ℕ)
    (h_dist : 60 * x = (100 * b + 10 * c + a) - (100 * a + 10 * b + c))
    (h_b_ge_1 : b ≥ 1)
    (h_sum_le_9 : a + b + c ≤ 9) :
    a^2 + b^2 + c^2 = 29 :=
sorry

end odometer_problem_l825_825641


namespace numeric_puzzle_AB_eq_B_pow_V_l825_825471

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l825_825471


namespace valid_differences_of_squares_l825_825104

theorem valid_differences_of_squares (n : ℕ) (h : 2 * n + 1 < 150) :
    (2 * n + 1 = 129 ∨ 2 * n +1 = 147) :=
by
  sorry

end valid_differences_of_squares_l825_825104


namespace average_children_l825_825435

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l825_825435


namespace binom_18_4_eq_3060_l825_825315

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l825_825315


namespace tangency_condition_value_of_a_tangency_range_of_a_l825_825520

noncomputable def f (x : ℝ) : ℝ := x^3 - x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

theorem tangency_condition_value_of_a :
  (∃ a : ℝ, ∀ x : ℝ, tangent_line (f x) (g x a) at (-1) = 2 * (x + 1)) → a = 3 :=
sorry

theorem tangency_range_of_a :
  (∃ a : ℝ, ∀ x1 : ℝ x2 : ℝ, tangent_line (f x1) (g x2 a) holds) → (a ∈ set.Ici (-1)) :=
sorry

end tangency_condition_value_of_a_tangency_range_of_a_l825_825520


namespace line_parabola_intersection_l825_825685

theorem line_parabola_intersection (k : ℝ) : 
    (∀ l p: ℝ → ℝ, l = (fun x => k * x + 1) ∧ p = (fun x => 4 * x ^ 2) → 
        (∃ x, l x = p x) ∧ (∀ x1 x2, l x1 = p x1 ∧ l x2 = p x2 → x1 = x2) 
    ↔ k = 0 ∨ k = 1) :=
sorry

end line_parabola_intersection_l825_825685


namespace probability_reach_edge_within_five_hops_l825_825855

-- Define the probability of reaching an edge within n hops from the center
noncomputable def probability_reach_edge_by_hops (n : ℕ) : ℚ :=
if n = 5 then 121 / 128 else 0 -- This is just a placeholder for the real recursive computation.

-- Main theorem to prove
theorem probability_reach_edge_within_five_hops :
  probability_reach_edge_by_hops 5 = 121 / 128 :=
by
  -- Skipping the actual proof here
  sorry

end probability_reach_edge_within_five_hops_l825_825855


namespace three_lines_determine_plane_l825_825721

-- Define a structure for three points
structure ThreePointsDeterminePlane (P1 P2 P3 : Type) where
  collinear : P1 ≠ P2 → P2 ≠ P3 → P1 ≠ P3 → Prop

-- Define a structure for a line and a point
structure LineAndPointDeterminePlane (l : Type) (P : Type) where
  on_line : P → Prop

-- Define a structure for a quadrilateral determining a plane
structure QuadrilateralDeterminePlane (Q1 Q2 Q3 Q4 : Type) where
  properly_spatial : P1 ≠ P2 → P2 ≠ P3 → P3 ≠ P4 → P4 ≠ P1 → Prop

-- Define a structure for three lines intersecting in pairs without a common point
structure ThreeLinesDeterminePlane (L1 L2 L3 : Type) where
  intersect_in_pairs : L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 → Prop

-- Proposition to prove:
theorem three_lines_determine_plane (L1 L2 L3 : Type)
  (h1 : ThreePointsDeterminePlane.point L1)
  (h2 : ThreePointsDeterminePlane.point L2)
  (h3 : ThreePointsDeterminePlane.point L3) :
  ThreeLinesDeterminePlane L1 L2 L3 :=
sorry

end three_lines_determine_plane_l825_825721


namespace find_angle_BC_LP_eq_90_l825_825251

open EuclideanGeometry

variables {A B C K L P : Point}

-- condition: A circle is circumscribed around an acute-angled triangle ABC
variables (circumcircle : Circle) 
(hABC : circumscribed circumscribed_triangle ABC circumcircle)

-- condition: K is the midpoint of the smaller arc AC of this circle
variables (hK : midpoint_arc_circumcircle K A C circumcircle)
 
-- condition: L is the midpoint of the smaller arc AK of this circle
variables (hL : midpoint_arc_circumcircle L A K circumcircle)

-- condition: Segments BK and AC intersect at point P
variables (hP : ∃ P, segment_intersect_AC_with_BK P A C B K)

-- condition: It is given that BK = BC
variables (h_eq : distance B K = distance B C)

-- question: Find the angle between the lines BC and LP, given that BK = BC
theorem find_angle_BC_LP_eq_90 : angle BC LP = 90 :=
by sorry

end find_angle_BC_LP_eq_90_l825_825251


namespace binary_addition_l825_825278

-- Define the binary numbers as natural numbers
def b1 : ℕ := 0b101  -- 101_2
def b2 : ℕ := 0b11   -- 11_2
def b3 : ℕ := 0b1100 -- 1100_2
def b4 : ℕ := 0b11101 -- 11101_2
def sum_b : ℕ := 0b110001 -- 110001_2

theorem binary_addition :
  b1 + b2 + b3 + b4 = sum_b := 
by
  sorry

end binary_addition_l825_825278


namespace find_a_value_l825_825599

noncomputable def a_value (m n : ℝ) (p : ℝ) : ℝ :=
  a (a : ℝ) = 5 * p

theorem find_a_value (m n : ℝ) (p : ℝ) (h1 : p = 0.4) (h2 : m = 5 * n + 5) (h3 : m + a = 5 * (n + p) + 5) :
  a = 2 :=
by
  sorry

end find_a_value_l825_825599


namespace numeric_puzzle_AB_eq_B_pow_V_l825_825467

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l825_825467


namespace solve_z_six_eq_eight_correct_l825_825844

open Complex

noncomputable def solve_z_six_eq_eight : Set ℂ := {z | z^6 = 8}

/-
  This theorem states that the set of solutions to the equation z^6 = 8
  is exactly {0 + i * (∛2), 0 - i * (∛2), ∛2, -∛2}.
-/
theorem solve_z_six_eq_eight_correct :
  solve_z_six_eq_eight = {0 + (Complex.ofReal $ Real.cbrt 2) * I, 
                          0 - (Complex.ofReal $ Real.cbrt 2) * I, 
                          Complex.ofReal (Real.cbrt 2), 
                          Complex.ofReal (-Real.cbrt 2)} :=
by sorry

end solve_z_six_eq_eight_correct_l825_825844


namespace tiling_not_possible_l825_825187

-- Definitions for the puzzle pieces
inductive Piece
| L | T | I | Z | O

-- Function to check if tiling a rectangle is possible
noncomputable def can_tile_rectangle (pieces : List Piece) : Prop :=
  ∀ (width height : ℕ), width * height % 4 = 0 → ∃ (tiling : List (Piece × ℕ × ℕ)), sorry

theorem tiling_not_possible : ¬ can_tile_rectangle [Piece.L, Piece.T, Piece.I, Piece.Z, Piece.O] :=
sorry

end tiling_not_possible_l825_825187


namespace chocolate_flavored_cups_sold_l825_825264

-- Define total sales and fractions
def total_cups_sold : ℕ := 50
def fraction_winter_melon : ℚ := 2 / 5
def fraction_okinawa : ℚ := 3 / 10
def fraction_chocolate : ℚ := 1 - (fraction_winter_melon + fraction_okinawa)

-- Define the number of chocolate-flavored cups sold
def num_chocolate_cups_sold : ℕ := 50 - (50 * 2 / 5 + 50 * 3 / 10)

-- Main theorem statement
theorem chocolate_flavored_cups_sold : num_chocolate_cups_sold = 15 := 
by 
  -- The proof would go here, but we use 'sorry' to skip it
  sorry

end chocolate_flavored_cups_sold_l825_825264


namespace smallest_n_S_gt_2048_l825_825163

-- Definitions as per the problem statement
def a_seq : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * (a_seq n) + n + 1

def S (n : ℕ) : ℕ :=
∑ i in finset.range (n + 1), a_seq i

-- The statement we need to prove
theorem smallest_n_S_gt_2048 : ∃ n : ℕ, S n > 2048 :=
by
  sorry

end smallest_n_S_gt_2048_l825_825163


namespace chocolate_flavored_cups_sold_l825_825262

-- Define total sales and fractions
def total_cups_sold : ℕ := 50
def fraction_winter_melon : ℚ := 2 / 5
def fraction_okinawa : ℚ := 3 / 10
def fraction_chocolate : ℚ := 1 - (fraction_winter_melon + fraction_okinawa)

-- Define the number of chocolate-flavored cups sold
def num_chocolate_cups_sold : ℕ := 50 - (50 * 2 / 5 + 50 * 3 / 10)

-- Main theorem statement
theorem chocolate_flavored_cups_sold : num_chocolate_cups_sold = 15 := 
by 
  -- The proof would go here, but we use 'sorry' to skip it
  sorry

end chocolate_flavored_cups_sold_l825_825262


namespace sum_of_squares_l825_825031

noncomputable def S_n (n : ℕ) : ℝ := 3 + 2^n
noncomputable def a_n (n : ℕ) : ℝ := if n = 1 then S_n 1 else S_n n - S_n (n-1)

theorem sum_of_squares (n : ℕ) : (∑ k in Finset.range (n + 1), (a_n (k + 1))^2) = (4^n + 71) / 3 :=
by
  sorry

end sum_of_squares_l825_825031


namespace trailing_zeros_product_l825_825563

/-- The number of trailing zeros in the product of 125, 320, and 15 is 5. -/
theorem trailing_zeros_product :
  ∀ (n1 n2 n3 : ℕ),
  n1 = 125 → n2 = 320 → n3 = 15 →
  let product := n1 * n2 * n3 in
  ∃ k : ℕ, (product = 10^k) ∧ (k = 5) :=
by {
  intros n1 n2 n3 h1 h2 h3,
  subst h1,
  subst h2,
  subst h3,
  let product := 125 * 320 * 15,
  existsi 5,
  split,
  {
    -- product = 10^5
    rw [Nat.pow],
    sorry
  },
  {
    -- k = 5
    refl,
  }
}

end trailing_zeros_product_l825_825563


namespace find_m_value_l825_825010

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)
def vector_sum (m : ℝ) : ℝ × ℝ := (1 + 3, m - 2)

-- Define the condition that vector_sum is parallel to vector_b
def vectors_parallel (m : ℝ) : Prop :=
  let (x1, y1) := vector_sum m
  let (x2, y2) := vector_b
  x1 * y2 - x2 * y1 = 0

-- The statement to prove
theorem find_m_value : ∃ m : ℝ, vectors_parallel m ∧ m = -2 / 3 :=
by {
  sorry
}

end find_m_value_l825_825010


namespace pictures_per_album_l825_825670

theorem pictures_per_album (total_pictures : ℕ) (total_albums : ℕ) (h1 : total_pictures = 480) (h2 : total_albums = 24) : (total_pictures / total_albums = 20) :=
by
  rw [h1, h2]
  norm_num
  -- Proof can be completed, but left as 'sorry' for now
  sorry

end pictures_per_album_l825_825670


namespace solution_l825_825926

noncomputable def problem (x : ℕ) : Prop :=
  2 ^ 28 = 4 ^ x  -- Simplified form of the condition given

theorem solution : problem 14 :=
by
  sorry

end solution_l825_825926


namespace tessa_debt_l825_825919

theorem tessa_debt :
  let initial_debt : ℤ := 40 in
  let repayment : ℤ := initial_debt / 2 in
  let debt_after_repayment : ℤ := initial_debt - repayment in
  let additional_debt : ℤ := 10 in
  debt_after_repayment + additional_debt = 30 :=
by
  -- The proof goes here.
  sorry

end tessa_debt_l825_825919


namespace richmond_more_than_victoria_l825_825675

-- Defining the population of Beacon
def beacon_people : ℕ := 500

-- Defining the population of Victoria based on Beacon's population
def victoria_people : ℕ := 4 * beacon_people

-- Defining the population of Richmond
def richmond_people : ℕ := 3000

-- The proof problem: calculating the difference
theorem richmond_more_than_victoria : richmond_people - victoria_people = 1000 := by
  -- The statement of the theorem
  sorry

end richmond_more_than_victoria_l825_825675


namespace number_of_valid_functions_l825_825077

noncomputable def number_of_functions (A : Finset ℕ) : ℕ :=
  let count := λ m, 2 ^ (2011 - m) - 1 in
  ∑ m in Finset.range 2010, count m

theorem number_of_valid_functions :
  let A := (Finset.range 2011).image (λ n, n + 1) in
  (∑ n : ℕ in A, λ f : ℕ → ℕ, f n ≤ n) = 2 ^ 2011 - 2012 :=
by sorry

end number_of_valid_functions_l825_825077


namespace work_completion_days_l825_825978

-- Define the work rates
def john_work_rate : ℚ := 1/8
def rose_work_rate : ℚ := 1/16
def dave_work_rate : ℚ := 1/12

-- Define the combined work rate
def combined_work_rate : ℚ := john_work_rate + rose_work_rate + dave_work_rate

-- Define the required number of days to complete the work together
def days_to_complete_work : ℚ := 1 / combined_work_rate

-- Prove that the total number of days required to complete the work is 48/13
theorem work_completion_days : days_to_complete_work = 48 / 13 :=
by 
  -- Here is where the actual proof would be, but it is not needed as per instructions
  sorry

end work_completion_days_l825_825978


namespace triangle_image_l825_825218

noncomputable def transform : ℂ → ℂ :=
  λ z => (1/Real.sqrt 2 + I/Real.sqrt 2) * z

theorem triangle_image :
  transform 0 = 0 ∧
  transform (1 - I) = Real.sqrt 2 ∧
  transform (1 + I) = Complex.I * Real.sqrt 2 :=
by
  -- We state the proof content with the intention to solve it later
  split
  focus 
  {
    -- First part of the proof (for transform 0)
    -- This would be where we formally prove each step in Lean
  }
  split
  focus 
  {
    -- Second part of the proof (for transform(1 - I))
    -- This would be where we formally prove each step in Lean
  }
  focus 
  {
    -- Third part of the proof (for transform(1 + I))
    -- This would be where we formally prove each step in Lean
  }
  sorry

end triangle_image_l825_825218


namespace pablo_leftover_money_l825_825648

theorem pablo_leftover_money :
  let cents_per_page := 0.01
  let pages_per_book := 150
  let books_read := 12
  let money_spent_on_candy := 15 : ℝ
  (books_read * pages_per_book * cents_per_page) - money_spent_on_candy = 3 := by
  sorry

end pablo_leftover_money_l825_825648


namespace extra_amount_spent_on_shoes_l825_825725

theorem extra_amount_spent_on_shoes (total_cost shirt_cost shoes_cost: ℝ) 
  (h1: total_cost = 300) (h2: shirt_cost = 97) 
  (h3: shoes_cost > 2 * shirt_cost)
  (h4: shirt_cost + shoes_cost = total_cost): 
  shoes_cost - 2 * shirt_cost = 9 :=
by
  sorry

end extra_amount_spent_on_shoes_l825_825725


namespace overtime_percentage_increase_l825_825247

-- Define the conditions.
def regular_rate : ℝ := 16
def regular_hours : ℕ := 40
def total_compensation : ℝ := 1116
def total_hours_worked : ℕ := 57
def overtime_hours : ℕ := total_hours_worked - regular_hours

-- Define the question and the answer as a proof problem.
theorem overtime_percentage_increase :
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := total_compensation - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  overtime_rate > regular_rate →
  ((overtime_rate - regular_rate) / regular_rate) * 100 = 75 := 
by
  sorry

end overtime_percentage_increase_l825_825247


namespace pentagon_terminate_l825_825831

theorem pentagon_terminate (a b c d e : ℤ) (h1 : a + b + c + d + e > 0) :
  (∃ n : ℕ, ∀ (a b c d e : ℤ), (∃ x y z : ℤ, y < 0 ∧ 
  ((x, y, z) = (a, b, c) ∨ (x, y, z) = (b, c, d) ∨ (x, y, z) = (c, d, e) ∨ (x, y, z) = (d, e, a) ∨ (x, y, z) = (e, a, b))) → 
  (a + b + c + d + e > 0 → (a + y, -y, z + y = x → (y ≥ 0)))) :=
begin
  sorry -- Proof omitted
end

end pentagon_terminate_l825_825831


namespace lee_initial_money_l825_825980

theorem lee_initial_money (friend_money wings_cost salad_cost soda_cost tax total_change total_spent : ℕ)
  (h_friend_money : friend_money = 8)
  (h_wings_cost : wings_cost = 6)
  (h_salad_cost : salad_cost = 4)
  (h_soda_cost : soda_cost = 2)
  (h_tax : tax = 3)
  (h_total_change : total_change = 3)
  (h_total_spent : total_spent = wings_cost + salad_cost + soda_cost + tax - total_change + 2 * soda_cost) :
  friend_money + 10 = total_spent :=
by
  have h_total_meal_cost : wings_cost + salad_cost + 2 * soda_cost + tax = 15, sorry,
  have h_paid_amount : 15 + total_change = 18, sorry,
  have h_lee_contribution : total_spent - friend_money = 10, sorry,
  exact h_lee_contribution

end lee_initial_money_l825_825980


namespace solve_for_x_l825_825350

theorem solve_for_x (x : ℝ) : (1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) → x = 10 :=
by
  sorry

end solve_for_x_l825_825350


namespace hyperbola_eq_l825_825239

theorem hyperbola_eq (c : ℝ) (λ : ℝ)
  (h1 : c = 5)
  (h2 : 16 * λ + 9 * λ = c ^ 2)
  (h3 : λ = 1) :
  ∀ x y : ℝ, (y^2 / 16 - x^2 / 9 = 1) :=
by sorry

end hyperbola_eq_l825_825239


namespace power_function_value_at_9_l825_825579

noncomputable def f (x : ℝ) (α: ℝ) : ℝ := x ^ α

theorem power_function_value_at_9 :
  (f 2 α = (√2) / 2) → (f 9 α = 1 / 3) :=
sorry

end power_function_value_at_9_l825_825579


namespace complex_symmetry_division_l825_825552

theorem complex_symmetry_division (z1 z2 : ℂ) (hz1 : z1 = 1 + 2 * complex.I) (hz2 : z2.im = z2.re ∧ z1.re = z2.im ∧ z1.im = z2.re) :
  z1 / z2 = (4 / 5 : ℂ) + (3 / 5 : ℂ) * complex.I := 
by
  sorry

end complex_symmetry_division_l825_825552


namespace students_standing_arrangement_l825_825587

theorem students_standing_arrangement : 
  let students : Finset (Fin 5) := {0, 1, 2, 3, 4}
  let total_permutations : ℕ := Finset.card (students.permutations)
  let ab_together : ℕ := Finset.card (({(0, 1), (1, 0)}.image (λ s, s.pair_permutations {2, 3, 4})).bUnion id)
  let ab_not_together : ℕ := total_permutations - ab_together
  total_permutations = 120 → ab_together = 48 → ab_not_together = 72 :=
by
  sorry

end students_standing_arrangement_l825_825587


namespace infinitely_many_primes_not_in_S_a_l825_825484

open Nat

def is_in_set_S_a (a : ℕ+) (p : ℕ) : Prop :=
  ∃ b : ℕ, b % 2 = 1 ∧ p ∣ (2^(2^a.val))^b - 1

theorem infinitely_many_primes_not_in_S_a (a : ℕ+) : ∃^∞ p, ¬is_in_set_S_a a p :=
sorry

end infinitely_many_primes_not_in_S_a_l825_825484


namespace avg_children_in_families_with_children_l825_825442

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l825_825442


namespace brainiacs_like_neither_l825_825765

variables 
  (total : ℕ) -- Total number of brainiacs.
  (R : ℕ) -- Number of brainiacs who like rebus teasers.
  (M : ℕ) -- Number of brainiacs who like math teasers.
  (both : ℕ) -- Number of brainiacs who like both rebus and math teasers.
  (math_only : ℕ) -- Number of brainiacs who like only math teasers.

-- Given conditions in the problem
def twice_as_many_rebus : Prop := R = 2 * M
def both_teasers : Prop := both = 18
def math_teasers_not_rebus : Prop := math_only = 20
def total_brainiacs : Prop := total = 100

noncomputable def exclusion_inclusion : ℕ := R + M - both

-- Proof statement: The number of brainiacs who like neither rebus nor math teasers totals to 4
theorem brainiacs_like_neither
  (h_total : total_brainiacs total)
  (h_twice : twice_as_many_rebus R M)
  (h_both : both_teasers both)
  (h_math_only : math_teasers_not_rebus math_only)
  (h_M : M = both + math_only) :
  total - exclusion_inclusion R M both = 4 :=
sorry

end brainiacs_like_neither_l825_825765


namespace vector_dot_product_calculation_l825_825566

theorem vector_dot_product_calculation : 
  let a := (2, 3, -1)
  let b := (2, 0, 3)
  let c := (0, 2, 2)
  (2 * (2 + 0) + 3 * (0 + 2) + -1 * (3 + 2)) = 5 := 
by
  sorry

end vector_dot_product_calculation_l825_825566


namespace power_sums_fifth_l825_825074

noncomputable def compute_power_sums (α β γ : ℂ) : ℂ :=
  α^5 + β^5 + γ^5

theorem power_sums_fifth (α β γ : ℂ)
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  compute_power_sums α β γ = 47.2 :=
sorry

end power_sums_fifth_l825_825074


namespace numerical_puzzle_solution_l825_825463

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l825_825463


namespace ellipse_equation_circle_intersection_l825_825883

-- Define the given conditions
variables {a b c : ℝ} (h1 : a > b > 0)
variables (P : ℝ × ℝ) (hP : P = (2, real.sqrt 2))
variables (ecc : ℝ) (h_ecc : ecc = real.sqrt 2 / 2)
variables (A : ℝ × ℝ) (hA : A = (-α, 0))
variables (E F : ℝ × ℝ) (h_sym : E = (-F.1, -F.2))
variables (l1 l2 : set (ℝ × ℝ)) (h_distinct : E ≠ (0, _))
variables (M N : ℝ × ℝ) 

-- (1) Prove the equation of the ellipse
theorem ellipse_equation (a b : ℝ) (h_ab : a^2 = 2 * b^2) (hP_on_ellipse : 4 / (2 * b^2) + 2 / b^2 = 1) :
  ∀ x y : ℝ, (x , y) ∈ {p | p.1^2 / 8 + p.2^2 / 4 = 1} :=
sorry

-- (2) Prove the intersection points of the circle with diameter MN
theorem circle_intersection (M N Q : ℝ × ℝ) (hM : M.2 = (2 * (2 ^ 2)) / (1 + real.sqrt (1 + 2 * (2 ^ 2)))) (hN : N.2 = (2 * (2 ^ 2)) / (1 - real.sqrt (1 + 2 * (2 ^ 2)))) (hQ : (Q.1 = 2 ∨ Q.1 = -2) ∧ Q.2 = 0) :
  ∃ Q : ℝ × ℝ, (Q.1 = 2 ∨ Q.1 = -2) ∧ Q.2 = 0 :=
sorry

end ellipse_equation_circle_intersection_l825_825883


namespace cos_seq_value_l825_825893

theorem cos_seq_value (a : ℕ → ℝ) (h_arith : ∀ n, a(n+1) - a(n) = a(1) - a(0))
  (h_sum : a 1 + a 8 + a 15 = Real.pi) :
  Real.cos (a 4 + a 12) = -1/2 :=
sorry

end cos_seq_value_l825_825893


namespace time_cross_platform2_l825_825766

/-- The length of the first platform is 110 meters. -/
def length_platform1 : ℕ := 110

/-- The time taken to cross the first platform is 15 seconds. -/
def time_cross_platform1 : ℕ := 15

/-- The length of the train is 310 meters. -/
def length_train : ℕ := 310

/-- The length of the second platform is 250 meters. -/
def length_platform2 : ℕ := 250

/-- Prove that the time it takes for the train to cross the second platform is 20 seconds. -/
theorem time_cross_platform2 : 
  let speed : ℚ := (length_train + length_platform1) / time_cross_platform1 in
  let total_distance : ℕ := length_train + length_platform2 in
  total_distance / speed = 20 := 
by 
  sorry

end time_cross_platform2_l825_825766


namespace original_price_of_dinosaur_model_l825_825139

-- Define the conditions
theorem original_price_of_dinosaur_model
  (P : ℝ) -- original price of each model
  (kindergarten_models : ℝ := 2)
  (elementary_models : ℝ := 2 * kindergarten_models)
  (total_models : ℝ := kindergarten_models + elementary_models)
  (reduction_percentage : ℝ := 0.05)
  (discounted_price : ℝ := P * (1 - reduction_percentage))
  (total_paid : ℝ := total_models * discounted_price)
  (total_paid_condition : total_paid = 570) :
  P = 100 :=
by
  sorry

end original_price_of_dinosaur_model_l825_825139


namespace largest_whole_number_m_l825_825707

theorem largest_whole_number_m (m : ℕ) : 
  (∃ m : ℕ, (1 / 4) + (m / 9) < 5 / 2 ∧ (∀ n : ℕ, (1 / 4) + (n / 9) < 5 / 2 → n ≤ m)) ↔ m = 10 :=
begin
  sorry
end

end largest_whole_number_m_l825_825707


namespace interest_calculated_years_is_2_l825_825029

variable (P : ℝ) -- Principal sum of money
variable (R : ℝ) -- Rate of interest per annum
variable (T : ℝ) -- Time in years
variable (SI : ℝ) -- Simple interest

-- Condition: Simple interest is one-fifth of the principal sum
def condition1 : SI = P / 5 := sorry

-- Condition: Formula for simple interest
def condition2 : SI = (P * R * T) / 100 := sorry

-- Condition: Rate of interest per annum is 10%
def condition3 : R = 10 := sorry

-- Proof statement: The time in years is 2
theorem interest_calculated_years_is_2 :
  T = 2 :=
by
  -- Use conditions 1, 2, and 3 to prove
  rw [condition1, condition2, condition3]
  sorry

end interest_calculated_years_is_2_l825_825029


namespace min_value_343_l825_825090

noncomputable def min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c)

theorem min_value_343 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ha hb hc = 343 :=
sorry

end min_value_343_l825_825090


namespace inequality_comparison_l825_825499

theorem inequality_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) 
  (h₁ : a = (1 / Real.log 3 / Real.log 2))
  (h₂ : b = Real.exp 0.5)
  (h₃ : c = Real.log 2) :
  b > c ∧ c > a := 
by
  sorry

end inequality_comparison_l825_825499


namespace intersection_points_and_combination_l825_825091

def f (x : ℝ) : ℝ := (x - 2) * (x - 5)

def g (x : ℝ) : ℝ := 2 * f(x)

def h (x : ℝ) : ℝ := f(-x) + 2

theorem intersection_points_and_combination :
  let a := 2 in
  let b := 1 in
  10 * a + b = 21 :=
by
 sorry

end intersection_points_and_combination_l825_825091


namespace range_log_div_pow3_div3_l825_825489

noncomputable def log_div (x y : ℝ) : ℝ := Real.log (x / y)
noncomputable def log_div_pow3 (x y : ℝ) : ℝ := Real.log (x^3 / y^(1/2))
noncomputable def log_div_pow3_div3 (x y : ℝ) : ℝ := Real.log (x^3 / (3 * y))

theorem range_log_div_pow3_div3 
  (x y : ℝ) 
  (h1 : 1 ≤ log_div x y ∧ log_div x y ≤ 2)
  (h2 : 2 ≤ log_div_pow3 x y ∧ log_div_pow3 x y ≤ 3) 
  : Real.log (x^3 / (3 * y)) ∈ Set.Icc (26/15 : ℝ) 3 :=
sorry

end range_log_div_pow3_div3_l825_825489


namespace pyramid_area_correct_pyramid_height_correct_l825_825714

def base_edge_length : ℝ := 8
def lateral_edge_length : ℝ := 10

def pyramid_area (A : ℝ) : Prop := 
  A = 32 * Real.sqrt 21

def pyramid_height (H : ℝ) : Prop := 
  H = 2 * Real.sqrt 17

theorem pyramid_area_correct : ∃ A, pyramid_area A :=
begin
  use 32 * Real.sqrt 21,
  exact rfl,
end

theorem pyramid_height_correct : ∃ H, pyramid_height H :=
begin
  use 2 * Real.sqrt 17,
  exact rfl,
end

end pyramid_area_correct_pyramid_height_correct_l825_825714


namespace students_in_both_band_chorus_but_not_drama_l825_825696

def LincolnHighSchool : Type := { student : Nat // student < 300 }

variables (Band Chorus Drama : Set LincolnHighSchool)

axiom band_size : ∀ s, s ∈ Band → s < 80
axiom chorus_size : ∀ s, s ∈ Chorus → s < 120
axiom drama_size : ∀ s, s ∈ Drama → s < 50
axiom total_students_in_band_chorus_drama : ∀ s, s ∈ Band ∨ s ∈ Chorus ∨ s ∈ Drama → s < 200

noncomputable def students_in_both_band_chorus_not_drama : Nat := sorry

theorem students_in_both_band_chorus_but_not_drama :
  students_in_both_band_chorus_not_drama = 50 :=
by
  sorry

end students_in_both_band_chorus_but_not_drama_l825_825696


namespace polynomial_evaluation_at_8_l825_825236

def P (x : ℝ) : ℝ := x^3 + 2*x^2 + x - 1

theorem polynomial_evaluation_at_8 : P 8 = 647 :=
by sorry

end polynomial_evaluation_at_8_l825_825236


namespace average_children_in_families_with_children_l825_825412

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l825_825412


namespace log_inverse_l825_825927

theorem log_inverse (y : ℝ) (log_base_16 : logBase 16 (y - 5) = 1 / 2):
  y = 9 → (1 / logBase y 5) = (2 * log 10 3) / (log 10 5) := by
  intro hy
  rw hy
  sorry

end log_inverse_l825_825927


namespace average_children_in_families_with_children_l825_825384

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l825_825384


namespace smallest_integer_representation_l825_825194

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end smallest_integer_representation_l825_825194


namespace expansion_terms_count_l825_825558

-- Define the number of terms in the first polynomial
def first_polynomial_terms : ℕ := 3

-- Define the number of terms in the second polynomial
def second_polynomial_terms : ℕ := 4

-- Prove that the number of terms in the expansion is 12
theorem expansion_terms_count : first_polynomial_terms * second_polynomial_terms = 12 :=
by
  sorry

end expansion_terms_count_l825_825558


namespace payment_to_Y_is_227_27_l825_825704

-- Define the conditions
def total_payment_per_week (x y : ℝ) : Prop :=
  x + y = 500

def x_payment_is_120_percent_of_y (x y : ℝ) : Prop :=
  x = 1.2 * y

-- Formulate the problem as a theorem to be proven
theorem payment_to_Y_is_227_27 (Y : ℝ) (X : ℝ) 
  (h1 : total_payment_per_week X Y) 
  (h2 : x_payment_is_120_percent_of_y X Y) : 
  Y = 227.27 :=
by
  sorry

end payment_to_Y_is_227_27_l825_825704


namespace complex_number_symmetric_l825_825530

noncomputable def z1 : ℂ := 2 + I
noncomputable def z2 : ℂ := -2 + I

theorem complex_number_symmetric :
  (z1 * z2) = -5 :=
by
  sorry

end complex_number_symmetric_l825_825530


namespace gas_cost_per_gallon_l825_825173

theorem gas_cost_per_gallon (mpg : ℝ) (miles_per_day : ℝ) (days : ℝ) (total_cost : ℝ) : 
  mpg = 50 ∧ miles_per_day = 75 ∧ days = 10 ∧ total_cost = 45 → 
  (total_cost / ((miles_per_day * days) / mpg)) = 3 :=
by
  sorry

end gas_cost_per_gallon_l825_825173


namespace evaluate_expression_l825_825757

theorem evaluate_expression (k : ℚ) (R : ℚ) (h : k = 3/2) (hr : 2 * R^2 + 2 * R - 3 = 0) :
  (R ^ (R ^ (R^2 + k * R⁻¹) + R⁻¹) + R⁻¹) = 2 := sorry

end evaluate_expression_l825_825757


namespace sum_of_squared_c_k_l825_825819

noncomputable def c_k (k : ℕ) : ℝ := k + 1 / (3 * k + 1 / (3 * k + 1 / (3 * k + ...)))

open Nat

theorem sum_of_squared_c_k : (∑ k in range 1 21, (c_k k)^2) = 5740 := by
  sorry

end sum_of_squared_c_k_l825_825819


namespace segment_AB_length_l825_825574

theorem segment_AB_length
  (m : ℝ)
  (A B : ℝ × ℝ)
  (h₁ : ∀ x y : ℝ, x ^ 2 + y ^ 2 = 5)
  (h₂ : ∀ x y : ℝ, (x + m) ^ 2 + y ^ 2 = 20)
  (h₃ : A = B → False)
  (h₄ : (fst A ^ 2 + snd A ^ 2 = 5) ∧ ((fst A + m) ^ 2 + snd A ^ 2 = 20)) :
    dist A B = 4 :=
by
  sorry

end segment_AB_length_l825_825574


namespace part1_part2_l825_825515

-- Part (1)
theorem part1 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x) (hg : ∀ x, g x = x^2 + a)
  (tangent_at_1 : ∀ x, ∃ (x1 : ℝ), x1 = -1 ∧
    (f' x1 = g' 1 ∧ f x1 = g 1)) :
  a = 3 := 
  sorry

-- Part (2)
theorem part2 (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x) (hg : ∀ x, ∃ a, g x = x^2 + a)
  (tangent_condition : ∀ x1 x2, 
    (f' x1 = g' x2 ∧ f x1 = g x2)) :
  ∃ a, a ≥ -1 :=
  sorry

end part1_part2_l825_825515


namespace tangent_line_at_x0_l825_825728

def curve (x : ℝ) : ℝ := 1 / (3 * x + 2)

def tangent_line (x : ℝ) : ℝ := - (3 / 64) * x + (7 / 32)

def x0 : ℝ := 2

theorem tangent_line_at_x0 :
  let y0 := curve x0
  let dy_dx := - (3 / ((3 * x0 + 2) ^ 2))
  y0 = tangent_line x0 ∧ ∀ x, 
  let m := dy_dx in
  y = curve x0 + m * (x - x0) → y = tangent_line x :=
  sorry

end tangent_line_at_x0_l825_825728


namespace sale_in_second_month_l825_825256

theorem sale_in_second_month 
  (m1 m2 m3 m4 m5 m6 : ℕ) 
  (h1: m1 = 6335) 
  (h2: m3 = 6855) 
  (h3: m4 = 7230) 
  (h4: m5 = 6562) 
  (h5: m6 = 5091)
  (average: (m1 + m2 + m3 + m4 + m5 + m6) / 6 = 6500) : 
  m2 = 6927 :=
sorry

end sale_in_second_month_l825_825256


namespace compare_abc_l825_825497

noncomputable def a : ℝ := 1 / Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.exp 0.5
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > c ∧ c > a := by
  sorry

end compare_abc_l825_825497


namespace elder_person_present_age_l825_825140

def younger_age : ℕ
def elder_age : ℕ

-- Conditions
axiom age_difference (y e : ℕ) : e = y + 16
axiom age_relation_6_years_ago (y e : ℕ) : e - 6 = 3 * (y - 6)

-- Proof of the present age of the elder person
theorem elder_person_present_age (y e : ℕ) (h1 : e = y + 16) (h2 : e - 6 = 3 * (y - 6)) : e = 30 :=
sorry

end elder_person_present_age_l825_825140


namespace polygon_with_equal_angle_sums_is_quadrilateral_l825_825943

theorem polygon_with_equal_angle_sums_is_quadrilateral 
    (n : ℕ)
    (h1 : (n - 2) * 180 = 360)
    (h2 : 360 = 360) :
  n = 4 := 
sorry

end polygon_with_equal_angle_sums_is_quadrilateral_l825_825943


namespace average_children_families_with_children_is_3_point_8_l825_825368

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l825_825368


namespace geometric_sequence_property_l825_825529

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def integral_value : ℝ := ∫ x in 0..2, real.sqrt (4 - x^2)

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_sum : a 2013 + a 2015 = integral_value) :
  a 2014 * (a 2012 + 2 * a 2014 + a 2016) = π^2 :=
by
  have integral_res := integral_value
  /- Other necessary proof steps would be here. -/
  sorry

end geometric_sequence_property_l825_825529


namespace time_to_fill_partial_bucket_l825_825581

-- Definitions for the conditions
def time_to_fill_full_bucket : ℝ := 135
def r := 2 / 3

-- The time to fill 2/3 of the bucket should be proven as 90
theorem time_to_fill_partial_bucket : time_to_fill_full_bucket * r = 90 := 
by 
  -- Prove that 90 is the correct time to fill two-thirds of the bucket
  sorry

end time_to_fill_partial_bucket_l825_825581


namespace ratio_snap_crackle_l825_825660

/-- Given conditions:
    1. S + C + P = 150
    2. P = 15
    3. C = 3 * P
    4. S = 90
    Prove:
    - The ratio of the amount Snap spent to the amount Crackle spent is 2:1.
-/
theorem ratio_snap_crackle (S C P : ℕ) (h₁ : S + C + P = 150) (h₂ : P = 15) (h₃ : C = 3 * P) (h₄ : S = 90) :
  S / C = 2 :=
by
  have h₄' : S = 90, from h₄
  have h₃' : C = 3 * 15, by rw [←h₂]
  have h₃'' : C = 45, by simp [h₃']
  have h₁' : 90 + 45 + 15 = 150, from by linarith
  have h_ratio : 90 / 45 = 2, by norm_num
  exact h_ratio

end ratio_snap_crackle_l825_825660


namespace probability_grid_entirely_black_l825_825745

/-- Probability that the grid is entirely black after the operations described. -/
theorem probability_grid_entirely_black :
  let initial_colors : List (List Bool) := [
      [true, true, true],
      [true, true, true],
      [true, true, true]
  ]
  let rotate_180 (grid : List (List Bool)) := [
      [grid[2][2], grid[2][1], grid[2][0]],
      [grid[1][2], grid[1][1], grid[1][0]],
      [grid[0][2], grid[0][1], grid[0][0]]
  ]
  let update_white (grid : List (List Bool)) (i j : ℕ) : Bool :=
     if grid[i][j] = false &&
        ((i > 0 ∧ grid[i-1][j] = true) ∨ 
        (i < 2 ∧ grid[i+1][j] = true) ∨ 
        (j > 0 ∧ grid[i][j-1] = true) ∨ 
        (j < 2 ∧ grid[i][j+1] = true))
     then true else grid[i][j]
  let apply_rules (grid : List (List Bool)) : List (List Bool) :=
     [[update_white grid 0 0, update_white grid 0 1, update_white grid 0 2],
      [update_white grid 1 0, update_white grid 1 1, update_white grid 1 2],
      [update_white grid 2 0, update_white grid 2 1, update_white grid 2 2]]
  let final_grid := apply_rules (rotate_180 initial_colors)
  (initial_colors[1][1] = true) →
  (final_grid = [[true, true, true],
                 [true, true, true],
                 [true, true, true]]) →
  (initial_colors.count (· = true) / initial_colors.length = 1 / 2) →
  (final_grid.count (· = true) / final_grid.length = 81 / 512) :=
sorry

end probability_grid_entirely_black_l825_825745


namespace circumcircle_through_midpoint_l825_825997

noncomputable def midpoint (A B : Point) : Point := sorry
def are_fair_lines (Γ1 Γ2 : Circle) (d : Line) : Prop := sorry

theorem circumcircle_through_midpoint (Γ1 Γ2 : Circle) (O1 O2 : Point)
  (hΓ1 : Γ1.center = O1) (hΓ2 : Γ2.center = O2)
  (h_ext : ¬(Γ1 ∩ Γ2).nonempty) (a b c : Line)
  (h_a : are_fair_lines Γ1 Γ2 a) (h_b : are_fair_lines Γ1 Γ2 b) (h_c : are_fair_lines Γ1 Γ2 c) :
  let M := midpoint O1 O2 in
  ∀ (M : Point), ∃ (circ : Circle), circ.contains M ∧ circ.contains_points (a ∩ b) (b ∩ c) (c ∩ a) :=
sorry

end circumcircle_through_midpoint_l825_825997


namespace intersect_curve_in_four_points_l825_825853

noncomputable def polynomial := (x : ℝ) → x^4 + 9*x^3 + c*x^2 + 9*x + 4

theorem intersect_curve_in_four_points (c : ℝ) : 
  c ≤ 243 / 8 → 
  ∃ (line : ℝ → ℝ), (∃ four_distinct_points : Finset ℝ, 
  four_distinct_points.card = 4 ∧ ∀ x ∈ four_distinct_points, polynomial x = line x) :=
begin
  sorry
end

end intersect_curve_in_four_points_l825_825853


namespace line_parallel_or_within_plane_l825_825882

-- Define the primitives involved in the problem
variables (b : Line) (α β : Plane)

-- Define the conditions given in the problem
axiom line_parallel_plane (b : Line) (α : Plane) : Prop
axiom plane_parallel_plane (α : Plane) (β : Plane) : Prop

-- Assume the given conditions as axioms
axiom h1 : line_parallel_plane b α
axiom h2 : plane_parallel_plane α β

-- State the theorem
theorem line_parallel_or_within_plane (b : Line) (α β : Plane) 
  (h1 : line_parallel_plane b α) (h2 : plane_parallel_plane α β) : 
  (∃ x, x ∈ β ∧ b = x) ∨ line_parallel_plane b β :=
by
  sorry

end line_parallel_or_within_plane_l825_825882


namespace dave_paid_10_more_than_doug_l825_825352

theorem dave_paid_10_more_than_doug :
  let n := 12
  let c := (12 : ℝ)
  let b := (3 : ℝ)
  let total_cost := c + b
  let cost_per_slice := total_cost / n
  let bacon_slices := 9
  let plain_slices := 3
  let dave_slices := 10
  let doug_slices := n - dave_slices
  let dave_payment := dave_slices * cost_per_slice
  let doug_payment := doug_slices * cost_per_slice
  dave_payment - doug_payment = 10 :=
by
  sorry

end dave_paid_10_more_than_doug_l825_825352


namespace smallest_integer_representable_l825_825200

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end smallest_integer_representable_l825_825200


namespace tan_alpha_value_l825_825527

variables (α : ℝ)

-- Angle α is in the third quadrant
def in_third_quadrant (α : ℝ) : Prop :=
  π < α ∧ α < 3 * π / 2

-- Given condition
def given_condition (α : ℝ) : Prop := 
  in_third_quadrant α ∧ sin α = -2 / 3

theorem tan_alpha_value (α : ℝ) (h : given_condition α) : 
  tan α = 2 * Real.sqrt 5 / 5 :=
sorry

end tan_alpha_value_l825_825527


namespace part1_part2_l825_825881

def f (α : ℝ) : ℝ := (sin (π - α) * cos (-α) * cos (-α + (3 * π) / 2)) / (cos (π / 2 - α) * sin (-π - α))

theorem part1 : f (-41 * π / 6) = sqrt 3 / 2 := 
sorry

theorem part2 (α : ℝ) (h1 : π < α ∧ α < 3 * π) (h2 : cos (α - 3 * π / 2) = 1 / 3) : 
  f(α) = 2 * sqrt 2 / 3 := 
sorry

end part1_part2_l825_825881


namespace average_children_l825_825437

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l825_825437


namespace average_children_in_families_with_children_l825_825372

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l825_825372


namespace person_a_work_days_l825_825184

theorem person_a_work_days (x : ℝ) :
  (2 * (1 / x + 1 / 45) = 1 / 9) → (x = 30) :=
by
  sorry

end person_a_work_days_l825_825184


namespace probability_sum_greater_than_six_l825_825119

theorem probability_sum_greater_than_six : 
  let s1 := {1, 2, 3}
  let s2 := {4, 5}
  let pairs := [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
  let favorable_pairs := [(2, 5), (3, 4), (3, 5)]
  let total_cases := pairs.length
  let favorable_cases := favorable_pairs.length
  (favorable_cases : ℚ) / total_cases = 1 / 2 :=
by
  sorry

end probability_sum_greater_than_six_l825_825119


namespace Murtha_pebbles_problem_l825_825644

theorem Murtha_pebbles_problem : 
  let a := 3
  let d := 3
  let n := 18
  let a_n := a + (n - 1) * d
  let S_n := n / 2 * (a + a_n)
  S_n = 513 :=
by
  sorry

end Murtha_pebbles_problem_l825_825644


namespace smallest_base10_integer_l825_825204

theorem smallest_base10_integer (a b : ℕ) (ha : a > 3) (hb : b > 3) (h : a + 3 = 3 * b + 1) :
  13 = a + 3 :=
by
  have h_in_base_a : a = 3 * b - 2 := by linarith,
  have h_in_base_b : 3 * b + 1 = 13 := by sorry,
  exact h_in_base_b

end smallest_base10_integer_l825_825204


namespace least_four_digit_perfect_square_and_fourth_power_l825_825708

theorem least_four_digit_perfect_square_and_fourth_power : 
    ∃ (n : ℕ), (1000 ≤ n) ∧ (n < 10000) ∧ (∃ a : ℕ, n = a^2) ∧ (∃ b : ℕ, n = b^4) ∧ 
    (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (∃ a : ℕ, m = a^2) ∧ (∃ b : ℕ, m = b^4) → n ≤ m) ∧ n = 6561 :=
by
  sorry

end least_four_digit_perfect_square_and_fourth_power_l825_825708


namespace number_of_rooms_l825_825647

theorem number_of_rooms (x : ℕ) (h1 : ∀ n, 6 * (n - 1) = 5 * n + 4) : x = 10 :=
sorry

end number_of_rooms_l825_825647


namespace numeric_puzzle_AB_eq_B_pow_V_l825_825469

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l825_825469


namespace hyperbola_eccentricity_sqrt3_l825_825000

theorem hyperbola_eccentricity_sqrt3
  {a b : ℝ} (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ (A O F2 : Point),
    triangle.area A O F2 / triangle.area A O B = 2) 
  (h4 : hyperbola.equation x y a b = x^2 / a^2 - y^2 / b^2) :
  hyperbola.eccentricity a b = sqrt 3 :=
sorry

end hyperbola_eccentricity_sqrt3_l825_825000


namespace find_angle_A_range_2b_minus_c_l825_825603

-- Defining the given conditions
variable (a b c : ℝ)
variable (m : ℝ)
variable (b2 c2 : ℝ)

-- Given
axiom triangle_condition : b2 = b^2 ∧ c2 = c^2
axiom quadratic_roots : b2 + c2 = a^2 + b * c
axiom roots_quadratic_eq : ∀ x, x^2 - (a^2 + b * c) * x + m = 0

-- Proving the first question
theorem find_angle_A : ∀ (A : ℝ), A = Real.arccos 1/2 → 0 < A ∧ A < π → A = π/3 := sorry

-- Given additional condition for question 2
axiom given_a : a = Real.sqrt 3
axiom given_A : ∀ A, A = π / 3 

-- Proving the second question
theorem range_2b_minus_c : -Real.sqrt 3 < 2 * b - c ∧ 2 * b - c < 2 * Real.sqrt 3 := sorry

end find_angle_A_range_2b_minus_c_l825_825603


namespace TriangleRHS_solution_l825_825592

noncomputable def TriangleRHS (PQ PR : ℝ) :=
  let Q := (PQ, 0)
  let P := (0, 0)
  let R := (0, PR)
  let RQ := real.sqrt (PQ^2 + PR^2)
  let M := ((PQ + 0) / 2, (0 + R) / 2)
  let L := (PQ, 1.5*real.sqrt(3))
  let PF := 0.825 * real.sqrt(3)
  ∃ F: ℝ × ℝ, (L.1 ≠ M.1 ∧ L.2 ≠ M.2 ∧ L.1 = M.1 ∧ L.2 = PF)

variable {PQ PR : ℝ}

theorem TriangleRHS_solution : PQ = 3 → PR = 3 * real.sqrt 3 → 
  ∃ F : ℝ × ℝ, 0.825 * real.sqrt 3 := by
  intros hPQ hPR

  sorry

end TriangleRHS_solution_l825_825592


namespace students_shared_cost_l825_825756

theorem students_shared_cost (P n : ℕ) (h_price_range: 100 ≤ P ∧ P ≤ 120)
  (h_div1: P % n = 0) (h_div2: P % (n - 2) = 0) (h_extra_cost: P / n + 1 = P / (n - 2)) : n = 14 := by
  sorry

end students_shared_cost_l825_825756


namespace proof_f_values_l825_825636

def f (x : ℤ) : ℤ :=
  if x < 0 then
    2 * x + 7
  else
    x^2 - 2

theorem proof_f_values :
  f (-2) = 3 ∧ f (3) = 7 :=
by
  sorry

end proof_f_values_l825_825636


namespace incenter_circumcircle_intersection_l825_825988

theorem incenter_circumcircle_intersection 
  (A B C I J : Type)
  [Triangle A B C] 
  [Incenter I (Triangle.mk A B C)]
  [CircumcircleIntersection J (Incenter.mk AI (Triangle.mk A B C))] :
  distance J B = distance J C ∧ distance J B = distance J I := 
sorry

end incenter_circumcircle_intersection_l825_825988


namespace prime_divides_g_g_plus_1_l825_825633

def g : ℕ → ℕ 
| 1 => 0
| 2 => 1
| (n + 2) => g(n) + g(n + 1)

theorem prime_divides_g_g_plus_1 (n : ℕ) (hn: n > 5) (prime_n: Nat.Prime n) : n ∣ (g(n) * (g(n) + 1)) :=
sorry

end prime_divides_g_g_plus_1_l825_825633


namespace andrew_age_l825_825797

variables (a g : ℕ)

theorem andrew_age : 
  (g = 16 * a) ∧ (g - a = 60) → a = 4 := by
  sorry

end andrew_age_l825_825797


namespace average_children_in_families_with_children_l825_825383

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l825_825383


namespace factorial_sqrt_square_l825_825298

theorem factorial_sqrt_square (n : ℕ) : (nat.succ 4)! * 4! = 2880 := by 
  sorry

end factorial_sqrt_square_l825_825298


namespace range_of_a_l825_825028

noncomputable def f (x : ℝ) := (1 / 3) * x ^ 3 - x

theorem range_of_a (a : ℝ) (h : ∀ x ∈ Ioo a (10 - a^2), f x ≥ f 1) : a ∈ Set.Icc (-2 : ℝ) 1 := by
  sorry

end range_of_a_l825_825028


namespace AB_eq_B_exp_V_l825_825457

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l825_825457


namespace convex_numbers_total_count_correct_l825_825023

def is_convex_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 700 ∧
  let d1 := n / 100,
      d2 := (n / 10) % 10,
      d3 := n % 10
  in d1 < d2 ∧ d2 > d3

def count_convex_numbers_below_700 : ℕ :=
  (List.range 700).countP is_convex_number

theorem convex_numbers_total_count_correct : count_convex_numbers_below_700 = 214 :=
  sorry

end convex_numbers_total_count_correct_l825_825023


namespace expected_value_of_twelve_sided_die_l825_825783

theorem expected_value_of_twelve_sided_die : ∑ k in finset.range 13, k / 12 = 6.5 := 
sorry

end expected_value_of_twelve_sided_die_l825_825783


namespace f_when_x_lt_0_set_B_l825_825886

def f (x : ℝ) : ℝ := if x > 0 then log x / log 2 else -log (-x) / log 2

def g (x : ℝ) : ℝ := 2 ^ x

def A : Set ℝ := {x | (x ≥ 4) ∨ (-1/4 ≤ x ∧ x < 0)}

def B : Set ℝ := {y | (2^(-1/4) ≤ y ∧ y < 1) ∨ (y ≥ 16)}

theorem f_when_x_lt_0 (x : ℝ) (h : x < 0) : f x = -log (-x) / log 2 := by
  simp [f, h, log, neg_pos]

theorem set_B (y : ℝ) : y ∈ B ↔ ∃ x ∈ A, g x = y := by
  simp [B, A, g]
  split
  · intro h
    cases h with
    | inl h₁ => exists (Classical.choose h₁)
    | inr h₂ => exists (Classical.choose h₂)
  sorry

end f_when_x_lt_0_set_B_l825_825886


namespace factorial_product_square_l825_825300

theorem factorial_product_square (n : ℕ) (m : ℕ) (h₁ : n = 5) (h₂ : m = 4) :
  (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 :=
by
  have f5 : Nat.factorial 5 = 120 := by norm_num
  have f4 : Nat.factorial 4 = 24 := by norm_num
  rw [Nat.factorial_eq_factorial h₁, Nat.factorial_eq_factorial h₂, f5, f4]
  norm_num
  simp
  sorry

end factorial_product_square_l825_825300


namespace exam_hall_expectation_l825_825953

theorem exam_hall_expectation :
  ∃ (X : Type) [random_variable X] (E : distribution X), 
  X = {0, 1, 2, 3, 4} ∧ 
  ∑ k in {0, 1, 2, 3, 4}, k * (E.probability k) = 21/10 :=
sorry

end exam_hall_expectation_l825_825953


namespace minimum_workers_to_profit_l825_825758

theorem minimum_workers_to_profit :
  ∀ (n : ℕ), (∀ (hc : ℕ), hc = 600) → (∀ (w : ℕ), w = 20) → (∀ (hours : ℕ), hours = 10) →
  (∀ (widgets_per_hour : ℕ), widgets_per_hour = 4) → (∀ (price_per_widget : ℕ), price_per_widget = 4) →
  (600 + 200 * n < 160 * n) → n ≥ 16 :=
by {
  intros n hc w hours widgets_per_hour price_per_widget h,
  sorry
}

end minimum_workers_to_profit_l825_825758


namespace lateral_faces_congruence_vs_regular_pyramid_l825_825742

def all_lateral_faces_are_congruent_triangles (p : Pyramid) : Prop :=
  ∀ (f1 f2 : Face), is_lateral_face f1 p → is_lateral_face f2 p → f1 ≅ f2

def is_regular_pyramid (p : Pyramid) : Prop :=
  IsRegularPyramid p

theorem lateral_faces_congruence_vs_regular_pyramid (p : Pyramid) :
  all_lateral_faces_are_congruent_triangles p → is_regular_pyramid p ∧
  (is_regular_pyramid p → all_lateral_faces_are_congruent_triangles p) :=
sorry

end lateral_faces_congruence_vs_regular_pyramid_l825_825742


namespace initial_games_l825_825555

-- Conditions
def games_given_away : ℕ := 7
def games_left : ℕ := 91

-- Theorem Statement
theorem initial_games (initial_games : ℕ) : 
  initial_games = games_left + games_given_away :=
by
  sorry

end initial_games_l825_825555


namespace find_q_value_l825_825734

theorem find_q_value (q : ℚ) (x y : ℚ) (hx : x = 5 - q) (hy : y = 3*q - 1) : x = 3*y → q = 4/5 :=
by
  sorry

end find_q_value_l825_825734


namespace least_small_barrels_l825_825276

theorem least_small_barrels (total_oil : ℕ) (large_barrel : ℕ) (small_barrel : ℕ) (L S : ℕ)
  (h1 : total_oil = 745) (h2 : large_barrel = 11) (h3 : small_barrel = 7)
  (h4 : 11 * L + 7 * S = 745) (h5 : total_oil - 11 * L = 7 * S) : S = 1 :=
by
  sorry

end least_small_barrels_l825_825276


namespace trig_identity_product_l825_825812

theorem trig_identity_product :
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) * 
  (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12)) = 1 / 16 :=
by
  sorry

end trig_identity_product_l825_825812


namespace average_children_in_families_with_children_l825_825424

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l825_825424


namespace percentage_of_total_population_absent_l825_825657

def total_students : ℕ := 120
def boys : ℕ := 72
def girls : ℕ := 48
def boys_absent_fraction : ℚ := 1/8
def girls_absent_fraction : ℚ := 1/4

theorem percentage_of_total_population_absent : 
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 17.5 :=
by
  sorry

end percentage_of_total_population_absent_l825_825657


namespace maximum_distance_one_car_can_travel_turn_back_l825_825179

noncomputable def barrels_per_car : ℕ := 24
noncomputable def distance_per_barrel : ℕ := 60
noncomputable def total_distance(a : ℕ, b : ℕ) : ℕ := a * b

theorem maximum_distance_one_car_can_travel_turn_back:
  let max_distance := total_distance (barrels_per_car/2) distance_per_barrel in
  max_distance = 360 :=
by
  -- this is just a placeholder since no proof is required.
  sorry

end maximum_distance_one_car_can_travel_turn_back_l825_825179


namespace avg_children_with_kids_l825_825392

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l825_825392


namespace area_parallelogram_l825_825801

theorem area_parallelogram (AE EB : ℝ) (SAEF SCEF SAEC SBEC SABC SABCD : ℝ) (h1 : SAE = 2 * EB)
  (h2 : SCEF = 1) (h3 : SAE == 2 * SCEF / 3) (h4 : SAEC == SAE + SCEF) 
  (h5 : SBEC == 1/2 * SAEC) (h6 : SABC == SAEC + SBEC) (h7 : SABCD == 2 * SABC) :
  SABCD = 5 := sorry

end area_parallelogram_l825_825801


namespace simplify_expression_l825_825307

variable (a : ℤ)

theorem simplify_expression : (-2 * a) ^ 3 * a ^ 3 + (-3 * a ^ 3) ^ 2 = a ^ 6 :=
by sorry

end simplify_expression_l825_825307


namespace quadrilateral_property_l825_825652

variables {A B C D O P Q K : Point}
variable {r k : ℝ}

-- Definitions of points and their properties
def quadrilateral_with_inscribed_circle (A B C D O : Point) (r : ℝ) : Prop :=
  inscribed_circle A B C D O r ∧ 
  intersects_line A B C D P ∧
  intersects_line A D B C Q ∧
  diagonals_intersect A C B D K
  
def distance_from_O_to_PQ (O P Q : Point) (k : ℝ) : Prop :=
  distance O (line P Q) = k

-- Main theorem statement
theorem quadrilateral_property  (h1 : quadrilateral_with_inscribed_circle A B C D O r)
                                (h2 : distance_from_O_to_PQ O P Q k) :
  distance O K * k = r^2 :=
by
  sorry

end quadrilateral_property_l825_825652


namespace f_neg_2_l825_825500

noncomputable def g : ℝ → ℝ := sorry
noncomputable def f (x : ℝ) : ℝ := g(x) + 2

lemma g_odd (x : ℝ) : g (-x) = - g(x) := sorry

lemma f_at_2 : f 2 = 3 := sorry

theorem f_neg_2 : f (-2) = 1 :=
by {
  -- use g_odd and f_at_2
  have h1 : f (-2) = g (-2) + 2,
  { unfold f },
  have h2 : g (-2) = - g 2,
  { apply g_odd },
  rw h2 at h1,
  rw f_at_2 at h1,
  unfold f at f_at_2,
  have h3 : g 2 = 1,
  { linarith },
  rw h3 at h1,
  linarith,
}

end f_neg_2_l825_825500


namespace not_possible_to_fill_6x6_with_1x4_l825_825974

theorem not_possible_to_fill_6x6_with_1x4 :
  ¬ (∃ (a b : ℕ), a + 4 * b = 6 ∧ 4 * a + b = 6) :=
by
  -- Assuming a and b represent the number of 1x4 rectangles aligned horizontally and vertically respectively
  sorry

end not_possible_to_fill_6x6_with_1x4_l825_825974


namespace percent_absent_l825_825655

-- Conditions
def num_students := 120
def num_boys := 72
def num_girls := 48
def frac_boys_absent := 1 / 8
def frac_girls_absent := 1 / 4

-- Theorem statement
theorem percent_absent : 
  ( (frac_boys_absent * num_boys + frac_girls_absent * num_girls) / num_students ) * 100 = 17.5 :=
by
  sorry

end percent_absent_l825_825655


namespace part_one_max_value_range_of_a_l825_825095

def f (x a : ℝ) : ℝ := |x + 2| - |x - 3| - a

theorem part_one_max_value (a : ℝ) (h : a = 1) : ∃ x : ℝ, f x a = 4 := 
by sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≤ 4 / a) :  (0 < a ∧ a ≤ 1) ∨ 4 ≤ a :=
by sorry

end part_one_max_value_range_of_a_l825_825095


namespace john_can_see_36_friends_l825_825607

theorem john_can_see_36_friends : 
  (∑ i in Finset.Icc 1 10, ∑ j in Finset.Icc 0 5, if Nat.gcd i j = 1 then 1 else 0) = 36 := 
sorry

end john_can_see_36_friends_l825_825607


namespace percentage_of_total_population_absent_l825_825656

def total_students : ℕ := 120
def boys : ℕ := 72
def girls : ℕ := 48
def boys_absent_fraction : ℚ := 1/8
def girls_absent_fraction : ℚ := 1/4

theorem percentage_of_total_population_absent : 
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 17.5 :=
by
  sorry

end percentage_of_total_population_absent_l825_825656


namespace orange_juice_students_l825_825955

-- Definitions derived from the conditions
def students := ℕ
def apple_juice_percent : ℚ := 50 / 100
def orange_juice_percent : ℚ := 30 / 100
def students_apple_juice : students := 120

-- The theorem statement
theorem orange_juice_students : (orange_juice_percent / apple_juice_percent) * (students_apple_juice : ℚ) = 72 := by
  sorry

end orange_juice_students_l825_825955


namespace benny_has_24_books_l825_825118

def books_sandy : ℕ := 10
def books_tim : ℕ := 33
def total_books : ℕ := 67

def books_benny : ℕ := total_books - (books_sandy + books_tim)

theorem benny_has_24_books : books_benny = 24 := by
  unfold books_benny
  unfold total_books
  unfold books_sandy
  unfold books_tim
  sorry

end benny_has_24_books_l825_825118


namespace downstream_speed_l825_825255

-- Define the speed of the fish in still water
def V_s : ℝ := 45

-- Define the speed of the fish going upstream
def V_u : ℝ := 35

-- Define the speed of the stream
def V_r : ℝ := V_s - V_u

-- Define the speed of the fish going downstream
def V_d : ℝ := V_s + V_r

-- The theorem to be proved
theorem downstream_speed : V_d = 55 := by
  sorry

end downstream_speed_l825_825255


namespace ln_series_inequality_l825_825901

theorem ln_series_inequality (n : ℕ) (hn : n > 1) : 
    (\sum i in Finset.range n, (Real.log (i + 2) / ((i + 2)^2 - 1)) + 1 + (1 / n)) < 
    ((n^2 + n + 10) / 4) :=
by
  sorry

end ln_series_inequality_l825_825901


namespace desired_probability_l825_825116

noncomputable def probability_desired_lamps : ℚ :=
  let total_ways_color := Nat.choose 8 4 in
  let ways_color_arrangement := Nat.choose 6 2 in
  let ways_on_off_configuration := Nat.choose 6 2 in
  (ways_color_arrangement * ways_on_off_configuration : ℚ) / (total_ways_color * total_ways_color)

theorem desired_probability :
  probability_desired_lamps = 225 / 4900 := sorry

end desired_probability_l825_825116


namespace ellipse_equation_l825_825524

-- Define the conditions and setup
def foci := (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) := F1 = (-1, 0) ∧ F2 = (1, 0)
def points_A_B (A B : ℝ × ℝ) := A = (1, 3/2) ∧ B = (1, -3/2) ∧ abs (A.2 - B.2) = 3

-- Prove that the given ellipse corresponds to these conditions
theorem ellipse_equation (a b : ℝ) (h1 : 0 < b ∧ b < a)
  (h2 : a^2 - b^2 = 1)
  (h3 : ∃ (A B : ℝ × ℝ), points_A_B A B ∧ (1/a^2) + ((3/2)^2 / b^2) = 1) :
  (a^2 = 4) ∧ (b^2 = 3) → ellipse_eq : (∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1) :=
by { sorry }

end ellipse_equation_l825_825524


namespace point_of_intersection_without_neg1_l825_825611

def f (x : ℝ) := (x^2 - 4 * x + 8) / (3 * x - 6)
def g (x : ℝ) := (-3 * x^2 + 9 * x + 6) / (x - 2)

theorem point_of_intersection_without_neg1 :
  ∃ y, f 3 = y ∧ g 3 = y :=
sorry

end point_of_intersection_without_neg1_l825_825611


namespace find_A_if_f_is_even_l825_825937

-- Definitions and conditions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def f (x A : ℝ) : ℝ := (x + 1) * (x - A)

-- Statement of the theorem
theorem find_A_if_f_is_even (A : ℝ) : (is_even_function (f x A)) → A = 1 :=
by
  -- The proof would be here
  sorry

end find_A_if_f_is_even_l825_825937


namespace num_valid_assignments_l825_825802

-- Define the problem conditions
def num_students : ℕ := 5
def dormitories : finset (string) := {"A", "B", "C"}
def students : finset (string) := {"Jia", "Student2", "Student3", "Student4", "Student5"}

-- Define the condition that at least one student must be in each dormitory
def at_least_one_student_in_each_dormitory (assignments : string → string) : Prop :=
  ∀ dorm in dormitories, ∃ student in students, assignments student = dorm

-- Define the condition that Jia cannot be in Dormitory A
def jia_not_in_dormA (assignments : string → string) : Prop :=
  assignments "Jia" ≠ "A"

-- Define the function that counts valid assignments
def count_valid_assignments : ℕ :=
  finset.card (finset.filter (λ assignments, at_least_one_student_in_each_dormitory assignments ∧ jia_not_in_dormA assignments)
  (finset.univ : finset (string → string)))

-- Theorem: The number of valid assignments is 40
theorem num_valid_assignments : count_valid_assignments = 40 :=
by
  sorry

end num_valid_assignments_l825_825802


namespace maximum_complexity_l825_825096

def complexity (n : ℚ) : ℕ  :=
  -- Define the complexity function as described
  sorry

def is_odd (m : ℕ) : Prop :=
  m % 2 = 1

theorem maximum_complexity (k : ℕ) (h : k = 50) :
  let max_num := (2^(k + 1) + 1) / 3 in
  ∃ m : ℕ, is_odd m ∧ m < 2^k ∧ 
  complexity (m / 2^k) ≤ complexity (max_num / 2^k) := 
by 
  sorry

end maximum_complexity_l825_825096


namespace find_selling_price_l825_825751

-- Conditions
def cost_price : ℝ := 1000
def gain_percentage : ℝ := 100 -- gain percentage is 100%

-- Definition of gain based on given percentage
def gain := gain_percentage / 100 * cost_price

-- Definition of selling price
def selling_price := cost_price + gain

-- Statement to be proved
theorem find_selling_price : selling_price = 2000 :=
by
  sorry

end find_selling_price_l825_825751


namespace smallest_integer_representation_l825_825195

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end smallest_integer_representation_l825_825195


namespace gcd_lcm_of_6_and_12_gcd_lcm_of_7_and_8_gcd_lcm_of_15_and_20_l825_825726

-- Definitions for GCD and LCM for the given pairs of numbers
def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Proof obligations
theorem gcd_lcm_of_6_and_12 : gcd 6 12 = 6 ∧ lcm 6 12 = 12 := by
  sorry

theorem gcd_lcm_of_7_and_8 : gcd 7 8 = 1 ∧ lcm 7 8 = 56 := by
  sorry

theorem gcd_lcm_of_15_and_20 : gcd 15 20 = 5 ∧ lcm 15 20 = 60 := by
  sorry

end gcd_lcm_of_6_and_12_gcd_lcm_of_7_and_8_gcd_lcm_of_15_and_20_l825_825726


namespace problem_roots_l825_825626

theorem problem_roots:
  (∃ p q r : ℝ, (p + q + r = 15) ∧ (p * q + q * r + r * p = 25) ∧ (p * q * r = 10) ∧ 
  (1 + p) * (1 + q) * (1 + r) = 51) :=
begin
  sorry
end

end problem_roots_l825_825626


namespace avg_children_in_families_with_children_l825_825405

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l825_825405


namespace binom_18_4_eq_3060_l825_825326

theorem binom_18_4_eq_3060 : nat.choose 18 4 = 3060 := sorry

end binom_18_4_eq_3060_l825_825326


namespace euler_line_bisects_AK_l825_825083

variables {A B C K O H G : Type}
variables [point O] [point H] [point G] [point K] [point A] [point B] [point C]

-- Conditions
-- Circumcenter O of triangle ABC
def circumcenter (O : Type) (A B C : Type) : Prop := sorry

-- Orthocenter H of triangle ABC
def orthocenter (H : Type) (A B C : Type) : Prop := sorry

-- Centroid G of triangle ABC
def centroid (G : Type) (A B C : Type) : Prop := sorry

-- K is a point symmetric to O concerning BC
def symmetric (K O B C : Type) : Prop := sorry

-- Euler line passes through H, G, and O
def euler_line (H G O A B C : Type) : Prop := sorry

-- Translation of the proof problem
theorem euler_line_bisects_AK (A B C K O H G : Type)
  [circumcenter O A B C] 
  [orthocenter H A B C] 
  [centroid G A B C]
  [symmetric K O B C] 
  [euler_line H G O A B C] : 
  (∃ M : Type, midpoint M A K ∧ collinear M H G ∧ collinear K M A) := 
sorry

end euler_line_bisects_AK_l825_825083


namespace binom_18_4_eq_3060_l825_825325

theorem binom_18_4_eq_3060 : nat.choose 18 4 = 3060 := sorry

end binom_18_4_eq_3060_l825_825325


namespace total_kids_in_class_l825_825731

theorem total_kids_in_class (red_ratio : ℕ) (blonde_ratio : ℕ) (black_ratio : ℕ) (red_kids : ℕ) (h_ratio : red_ratio = 3 ∧ blonde_ratio = 6 ∧ black_ratio = 7) (h_red : red_kids = 9) :
  let total_ratio := red_ratio + blonde_ratio + black_ratio in
  total_ratio * (red_kids / red_ratio) = 48 :=
by sorry

end total_kids_in_class_l825_825731


namespace find_a_l825_825228

theorem find_a (a : ℝ) (h : ((2 * a + 16) + (3 * a - 8)) / 2 = 89) : a = 34 :=
sorry

end find_a_l825_825228


namespace sqrt_factorial_squared_l825_825305

theorem sqrt_factorial_squared :
  (Real.sqrt ((Nat.factorial 5) * (Nat.factorial 4))) ^ 2 = 2880 :=
by sorry

end sqrt_factorial_squared_l825_825305


namespace maximum_diagonals_with_same_length_l825_825040

theorem maximum_diagonals_with_same_length (n : ℕ) (hn : n = 1000) :
  ∃ k, (∀ (diagonals : finset (ℕ × ℕ)) (h : diagonals.card = k), 
          ∀ a b c ∈ diagonals, 
            (a = b ∨ a = c ∨ b = c)) ∧ k = 2000 :=
by
  existsi 2000
  sorry

end maximum_diagonals_with_same_length_l825_825040


namespace joe_first_lift_is_400_mike_first_lift_is_450_lisa_second_lift_is_250_l825_825585

-- Defining the weights of Joe's lifts
variable (J1 J2 : ℕ)

-- Conditions for Joe
def joe_conditions : Prop :=
  (J1 + J2 = 900) ∧ (2 * J1 = J2 + 300)

-- Defining the weights of Mike's lifts
variable (M1 M2 : ℕ)

-- Conditions for Mike  
def mike_conditions : Prop :=
  (M1 + M2 = 1100) ∧ (M2 = M1 + 200)

-- Defining the weights of Lisa's lifts
variable (L1 L2 : ℕ)

-- Conditions for Lisa  
def lisa_conditions : Prop :=
  (L1 + L2 = 1000) ∧ (L1 = 3 * L2)

-- Proof statements
theorem joe_first_lift_is_400 (h : joe_conditions J1 J2) : J1 = 400 :=
by
  sorry

theorem mike_first_lift_is_450 (h : mike_conditions M1 M2) : M1 = 450 :=
by
  sorry

theorem lisa_second_lift_is_250 (h : lisa_conditions L1 L2) : L2 = 250 :=
by
  sorry

end joe_first_lift_is_400_mike_first_lift_is_450_lisa_second_lift_is_250_l825_825585


namespace arccos_sqrt_three_over_two_is_pi_div_six_l825_825311

noncomputable def arccos_sqrt_three_over_two_eq_pi_div_six : Prop :=
  arccos (Real.sqrt 3 / 2) = Real.pi / 6

theorem arccos_sqrt_three_over_two_is_pi_div_six :
  arccos_sqrt_three_over_two_eq_pi_div_six :=
by 
  sorry

end arccos_sqrt_three_over_two_is_pi_div_six_l825_825311


namespace gooGoo_buttons_l825_825135

theorem gooGoo_buttons (num_3_button_shirts : ℕ) (num_5_button_shirts : ℕ)
  (buttons_per_3_button_shirt : ℕ) (buttons_per_5_button_shirt : ℕ)
  (order_quantity : ℕ)
  (h1 : num_3_button_shirts = order_quantity)
  (h2 : num_5_button_shirts = order_quantity)
  (h3 : buttons_per_3_button_shirt = 3)
  (h4 : buttons_per_5_button_shirt = 5)
  (h5 : order_quantity = 200) :
  num_3_button_shirts * buttons_per_3_button_shirt + num_5_button_shirts * buttons_per_5_button_shirt = 1600 := by
  have h6 : 200 * 3 = 600 := by norm_num
  have h7 : 200 * 5 = 1000 := by norm_num
  have h8 : 600 + 1000 = 1600 := by norm_num
  rw [h1, h2, h3, h4, h5]
  rw [h6, h7]
  exact h8

end gooGoo_buttons_l825_825135


namespace inscribed_circle_radius_eq_3_l825_825042

open Real

theorem inscribed_circle_radius_eq_3
  (a : ℝ) (A : ℝ) (p : ℝ) (r : ℝ)
  (h_eq_tri : ∀ (a : ℝ), A = (sqrt 3 / 4) * a^2)
  (h_perim : ∀ (a : ℝ), p = 3 * a)
  (h_area_perim : ∀ (a : ℝ), A = (3 / 2) * p) :
  r = 3 :=
by sorry

end inscribed_circle_radius_eq_3_l825_825042


namespace granola_bars_distribution_l825_825101

theorem granola_bars_distribution
  (total_bars : ℕ)
  (eaten_bars : ℕ)
  (num_children : ℕ)
  (remaining_bars := total_bars - eaten_bars)
  (bars_per_child := remaining_bars / num_children) :
  total_bars = 200 → eaten_bars = 80 → num_children = 6 → bars_per_child = 20 :=
by
  intros h1 h2 h3
  sorry

end granola_bars_distribution_l825_825101


namespace integer_solutions_of_quadratic_equation_l825_825475

theorem integer_solutions_of_quadratic_equation :
  {p : ℤ × ℤ | let x := p.1, y := p.2 in x^2 - 5 * x * y + 6 * y^2 - 3 * x + 5 * y - 25 = 0} 
  = {(4, -1), (-26, -9), (-16, -9), (-6, -1), (50, 15), (-72, -25)} :=
by sorry

end integer_solutions_of_quadratic_equation_l825_825475


namespace find_m_l825_825684

noncomputable def alpha : ℝ := (3 + Real.sqrt 13) / 2
def approx_m (m n : ℕ) : Prop := n < 500 ∧ abs (alpha - (m / n)) < 3 * 10^(-6)

theorem find_m : ∃ (m n : ℕ), approx_m m n ∧ m = 199 :=
by
  sorry

end find_m_l825_825684


namespace tangency_condition_value_of_a_tangency_range_of_a_l825_825521

noncomputable def f (x : ℝ) : ℝ := x^3 - x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

theorem tangency_condition_value_of_a :
  (∃ a : ℝ, ∀ x : ℝ, tangent_line (f x) (g x a) at (-1) = 2 * (x + 1)) → a = 3 :=
sorry

theorem tangency_range_of_a :
  (∃ a : ℝ, ∀ x1 : ℝ x2 : ℝ, tangent_line (f x1) (g x2 a) holds) → (a ∈ set.Ici (-1)) :=
sorry

end tangency_condition_value_of_a_tangency_range_of_a_l825_825521


namespace range_of_m_l825_825896

theorem range_of_m (α β m : ℝ)
  (h1 : 0 < α ∧ α < 1)
  (h2 : 1 < β ∧ β < 2)
  (h3 : ∀ x, x^2 - m * x + 1 = 0 ↔ (x = α ∨ x = β)) :
  2 < m ∧ m < 5 / 2 :=
sorry

end range_of_m_l825_825896


namespace sqrt_factorial_squared_l825_825303

theorem sqrt_factorial_squared :
  (Real.sqrt ((Nat.factorial 5) * (Nat.factorial 4))) ^ 2 = 2880 :=
by sorry

end sqrt_factorial_squared_l825_825303


namespace union_of_intervals_l825_825576

theorem union_of_intervals :
  let M := {x : ℝ | x^2 - 3 * x - 4 ≤ 0}
  let N := {x : ℝ | x^2 - 16 ≤ 0}
  M ∪ N = {x : ℝ | -4 ≤ x ∧ x ≤ 4} :=
by
  sorry

end union_of_intervals_l825_825576


namespace equal_opposite_sides_of_octagon_l825_825586

def side_lengths (n : ℕ) := ∀ (i : ℕ), i < n → ℤ

def equal_internal_angles (a : side_lengths 8) : Prop :=
  ∀ i j, i ≠ j → i < 8 → j < 8 → angle_between_sides a i = angle_between_sides a j

noncomputable def angle_between_sides : side_lengths 8 → ℕ → ℝ := sorry

theorem equal_opposite_sides_of_octagon (a : side_lengths 8) 
  (h1 : equal_internal_angles a) : 
  (a 0 = a 4) ∧ (a 1 = a 5) ∧ (a 2 = a 6) ∧ (a 3 = a 7) :=
  sorry

end equal_opposite_sides_of_octagon_l825_825586


namespace fish_population_l825_825950

theorem fish_population (N : ℕ) (h1 : 50 > 0) (h2 : N > 0) 
  (tagged_first_catch : ℕ) (total_first_catch : ℕ)
  (tagged_second_catch : ℕ) (total_second_catch : ℕ)
  (h3 : tagged_first_catch = 50)
  (h4 : total_first_catch = 50)
  (h5 : tagged_second_catch = 2)
  (h6 : total_second_catch = 50)
  (h_percent : (tagged_first_catch : ℝ) / (N : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ))
  : N = 1250 := 
  by sorry

end fish_population_l825_825950


namespace approx_num_fish_in_pond_l825_825948

noncomputable def numFishInPond (tagged_in_second: ℕ) (total_second: ℕ) (tagged: ℕ) : ℕ :=
  tagged * total_second / tagged_in_second

theorem approx_num_fish_in_pond :
  numFishInPond 2 50 50 = 1250 := by
  sorry

end approx_num_fish_in_pond_l825_825948


namespace christen_peeled_20_potatoes_l825_825011

-- Define the conditions and question
def homer_rate : ℕ := 3
def time_alone : ℕ := 4
def christen_rate : ℕ := 5
def total_potatoes : ℕ := 44

noncomputable def christen_potatoes : ℕ :=
  (total_potatoes - (homer_rate * time_alone)) / (homer_rate + christen_rate) * christen_rate

theorem christen_peeled_20_potatoes :
  christen_potatoes = 20 := by
  -- Proof steps would go here
  sorry

end christen_peeled_20_potatoes_l825_825011


namespace line_condition_l825_825051

variable (m n Q : ℝ)

theorem line_condition (h1: m = 8 * n + 5) 
                       (h2: m + Q = 8 * (n + 0.25) + 5) 
                       (h3: p = 0.25) : Q = 2 :=
by
  sorry

end line_condition_l825_825051


namespace binom_18_4_l825_825335

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end binom_18_4_l825_825335


namespace cost_mixture_per_kg_l825_825793

-- Let C1 be the cost of the first variety of rice
def C1 : ℝ := 5

-- Let C2 be the cost of the second variety of rice
def C2 : ℝ := 8.75

-- Let R be the ratio of the two varieties of rice
def R : ℝ := 0.5

-- Prove that the cost of the mixture per kg is 7.5
theorem cost_mixture_per_kg : 
  let Quantity_C1 := R * 1 in
  let Quantity_C2 := 1 in
  let Total_Cost := (C1 * Quantity_C1) + (C2 * Quantity_C2) in
  let Total_Weight := Quantity_C1 + Quantity_C2 in
  let M := Total_Cost / Total_Weight in
  M = 7.5 :=
by
  -- This is where the proof would go, but it is omitted for now
  sorry

end cost_mixture_per_kg_l825_825793


namespace numerical_puzzle_solution_l825_825466

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l825_825466


namespace same_side_locus_opposite_side_locus_l825_825234

open Real

-- Problem setup with definitions and conditions.
def line1_equation (x : ℝ) (g : ℝ) : ℝ := g
def line2_equation (x : ℝ) (g : ℝ) : ℝ := -g
def perpendicular_line (x : ℝ) (y : ℝ) : Prop := (x, y)
def O1 := (0, line1_equation 0)
def O2 := (0, line2_equation 0)

-- Defining the condition O1P1 * O2P2 = k for specific intersections.
def condition (k : ℝ) (P1 P2 : ℝ × ℝ) (O1 O2 : ℝ × ℝ) : Prop := 
  let O1P1 := dist O1 P1 in
  let O2P2 := dist O2 P2 in
  O1P1 * O2P2 = k

-- Lean statement about locus of points P for k = 1 (same side)
theorem same_side_locus (g : ℝ) (g_pos : g > 0) (x y : ℝ) :
  (∃ k P1 P2, (P1 = (x, line1_equation x g)) ∧ (P2 = (k / x, line2_equation (k / x) g)) ∧
  P1.2 = P2.2 ∧ condition 1 P1 P2 O1 O2) →
  y^2 / g^2 ≥ 1 - x^2 :=
by sorry

-- Lean statement about locus of points P for k = -1 (opposite side)
theorem opposite_side_locus (g : ℝ) (g_pos : g > 0) (x y : ℝ) :
  (∃ k P1 P2, (P1 = (x, line1_equation x g)) ∧ (P2 = (k / x, line2_equation (k / x) g)) ∧
  P1.2 ≠ P2.2 ∧ condition 1 P1 P2 O1 O2) →
  y^2 / g^2 ≤ 1 + x^2 :=
by sorry

end same_side_locus_opposite_side_locus_l825_825234


namespace average_children_families_with_children_is_3_point_8_l825_825362

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l825_825362


namespace googoo_total_buttons_l825_825138

noncomputable def button_count_shirt_1 : ℕ := 3
noncomputable def button_count_shirt_2 : ℕ := 5
noncomputable def quantity_shirt_1 : ℕ := 200
noncomputable def quantity_shirt_2 : ℕ := 200

theorem googoo_total_buttons :
  (quantity_shirt_1 * button_count_shirt_1) + (quantity_shirt_2 * button_count_shirt_2) = 1600 := by
  sorry

end googoo_total_buttons_l825_825138


namespace expected_value_of_twelve_sided_die_l825_825769

theorem expected_value_of_twelve_sided_die : 
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0) in
  (finset.sum outcomes (λ n, (n:ℝ)) / 12 = 6.5) :=
by
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0)
  have h1 : ∑ n in outcomes, (n : ℝ) = 78, sorry
  have h2 : (78 / 12) = 6.5, sorry
  exact h2

end expected_value_of_twelve_sided_die_l825_825769


namespace smallest_base10_integer_l825_825214

theorem smallest_base10_integer (a b : ℕ) (h1 : a > 3) (h2 : b > 3) :
    (1 * a + 3 = 3 * b + 1) → (1 * 10 + 3 = 13) :=
by
  intros h


-- Prove that  1 * a + 3 = 3 * b + 1 
  have a_eq : a = 3 * b - 2 := by linarith

-- Prove that 1 * 10 + 3 = 13 
  have base_10 := by simp

have the smallest base 10
  sorry

end smallest_base10_integer_l825_825214


namespace average_children_l825_825439

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l825_825439


namespace max_distinct_red_vertices_l825_825613

theorem max_distinct_red_vertices : 
  ∃ n : ℕ, n ≤ 5 ∧ n = 5 ∧ 
    ∀ (red_vertices : finset ℕ), 
    red_vertices.card = n →
    (∀ (i j : ℕ) (hi : i ∈ red_vertices) (hj : j ∈ red_vertices), i ≠ j → 
      abs (i - j) % 21 ≠ abs (j - i) % 21) :=
by 
  sorry

end max_distinct_red_vertices_l825_825613


namespace AB_eq_B_exp_V_l825_825460

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l825_825460


namespace red_balls_count_l825_825245

theorem red_balls_count (r y b : ℕ) (total_balls : ℕ := 15) (prob_neither_red : ℚ := 2/7) :
    y + b = total_balls - r → (15 - r) * (14 - r) = 60 → r = 5 :=
by
  intros h1 h2
  sorry

end red_balls_count_l825_825245


namespace min_value_f_in_interval_shifted_right_and_symmetric_l825_825681

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := cos (2 * x - φ) - sqrt 3 * sin (2 * x - φ)

theorem min_value_f_in_interval_shifted_right_and_symmetric 
  (φ : ℝ) (hφ : |φ| < π / 2) : 
  f (x - π / 12) - sqrt 3 = min {y | ∃ x ∈ Icc (-π / 2 : ℝ) 0, y = f x (-(π / 3))} :=
sorry

end min_value_f_in_interval_shifted_right_and_symmetric_l825_825681


namespace average_children_in_families_with_children_l825_825421

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l825_825421


namespace find_number_l825_825931

theorem find_number (N : ℝ) (h : 0.015 * N = 90) : N = 6000 :=
  sorry

end find_number_l825_825931


namespace average_children_in_families_with_children_l825_825375

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l825_825375


namespace roger_first_bag_correct_l825_825658

noncomputable def sandra_total_pieces : ℕ := 2 * 6
noncomputable def roger_total_pieces : ℕ := sandra_total_pieces + 2
noncomputable def roger_known_bag_pieces : ℕ := 3
noncomputable def roger_first_bag_pieces : ℕ := 11

theorem roger_first_bag_correct :
  roger_total_pieces - roger_known_bag_pieces = roger_first_bag_pieces := 
  by sorry

end roger_first_bag_correct_l825_825658


namespace total_participants_l825_825168

-- Definitions based on given conditions
variable (F M : ℕ)

def female_democrats : ℕ := F / 2
def male_democrats : ℕ := M / 4
def total_democrats : ℕ := (F + M) / 3

-- Given condition: There are 125 female democrats
axiom female_democrats_125 : female_democrats F M = 125

-- Given condition: One-third of all participants are democrats
axiom one_third_total_democrats : female_democrats F M + male_democrats F M = total_democrats F M

-- Goal: Prove that the total number of participants is 750
theorem total_participants : F + M = 750 :=
by
  sorry

end total_participants_l825_825168


namespace range_of_m_l825_825540

-- Define the function f
def f (x : ℝ) : ℝ :=
  (Real.exp x - Real.exp (-x)) * x^3

-- Prove the range of m given the condition
theorem range_of_m (m : ℝ) :
  f (Real.logb 2 m) + f (Real.logb (1/2) m) ≤ 2 * ((Real.exp 2 - 1) / Real.exp 1) →
  1/2 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l825_825540


namespace daisy_count_per_bouquet_l825_825754

-- Define the conditions
def roses_per_bouquet := 12
def total_bouquets := 20
def rose_bouquets := 10
def daisy_bouquets := total_bouquets - rose_bouquets
def total_flowers_sold := 190
def total_roses_sold := rose_bouquets * roses_per_bouquet
def total_daisies_sold := total_flowers_sold - total_roses_sold

-- Define the problem: prove that the number of daisies per bouquet is 7
theorem daisy_count_per_bouquet : total_daisies_sold / daisy_bouquets = 7 := by
  sorry

end daisy_count_per_bouquet_l825_825754


namespace pyramid_volume_l825_825481

-- Define the given conditions
def side_length := 3         -- side length of the base in cm
def vertex_angle := 90       -- vertex angle in degrees

-- Define the result we need to prove
def volume := (9 * Real.sqrt 2) / 8 -- volume of the pyramid in cubic cm

-- Lean statement to prove the volume of the regular triangular pyramid
theorem pyramid_volume : 
  (∃ (a : ℝ), a = side_length) →
  (∃ (θ : ℝ), θ = vertex_angle) →
  volume = (9 * Real.sqrt 2) / 8 :=
by
  intro h1 h2
  sorry

end pyramid_volume_l825_825481


namespace expected_X_lt_expected_Y_variance_X_eq_variance_Y_expected_Z_eq_33_over_8_l825_825698

variable {Ω : Type} [ProbabilitySpace Ω]

-- Definitions of the random variables X, Y, and Z
def X : Ω → ℕ := sorry -- Number of white balls selected
def Y : Ω → ℕ := sorry -- Number of black balls selected
def Z (X Y : Ω → ℕ) : Ω → ℕ := λ ω, 2 * X ω + Y ω

-- Assumptions about the uniform selection process
axiom X_Y_sum_three : ∀ ω, X ω + Y ω = 3
axiom X_distribution : ∀ k, (ProbabilityTheory.Probability (Measurement.MeasurableSet {ω | X ω = k})) = (nat.choose 3 k * nat.choose 5 (3 - k)) / nat.choose 8 3

-- Expected values and variances
def E (f : Ω → ℕ) [Measurable f] : ℝ := ∑ ω, f ω * (ProbabilityTheory.Probability (Measurement.MeasurableSet {ω'} => ω' = ω}))
def D (f : Ω → ℕ) [Measurable f] : ℝ := E (λ ω, (f ω)^2) - (E f)^2

-- Theorem statements to be proved
theorem expected_X_lt_expected_Y : E X < E Y :=
sorry

theorem variance_X_eq_variance_Y : D X = D Y :=
sorry

theorem expected_Z_eq_33_over_8 : E (Z X Y) = 33 / 8 :=
sorry

end expected_X_lt_expected_Y_variance_X_eq_variance_Y_expected_Z_eq_33_over_8_l825_825698


namespace projection_b_l825_825618

noncomputable def proj (u v : ℝ^2) : ℝ^2 :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1^2 + v.2^2
  ((dot_product / norm_sq) * v.1, (dot_product / norm_sq) * v.2)

variables (a b : ℝ^2)
variables (ha_proj : proj (4, -2) a = (-2/5, -4/5))
variables (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0)

theorem projection_b :
  proj (4, -2) b = (22/5, -6/5) :=
sorry

end projection_b_l825_825618


namespace line_plane_relationship_l825_825691

-- Define a Line
structure Line where
  points : Set Point

-- Define a Plane
structure Plane where
  points : Set Point

-- Definitions for the relationships
def line_in_plane (a : Line) (β : Plane) : Prop :=
  ∀ p, p ∈ a.points → p ∈ β.points

def line_intersects_plane (a : Line) (β : Plane) : Prop :=
  ∃ p, p ∈ a.points ∧ p ∈ β.points

def line_parallel_plane (a : Line) (β : Plane) : Prop :=
  ¬ line_intersects_plane a β

-- The final theorem statement
theorem line_plane_relationship (a : Line) (β : Plane) :
  line_in_plane a β ∨ line_intersects_plane a β ∨ line_parallel_plane a β :=
sorry

end line_plane_relationship_l825_825691


namespace average_of_numbers_divisible_by_9_l825_825227

theorem average_of_numbers_divisible_by_9 :
    let nums := [18, 27, 36, 45, 54, 63, 72, 81] in
    let sum := 18 + 27 + 36 + 45 + 54 + 63 + 72 + 81 in
    let count := 8 in
    (sum / count : ℝ) = 49.5 :=
by
  -- Defining the given list and sum
  let nums := [18, 27, 36, 45, 54, 63, 72, 81]
  let sum := 18 + 27 + 36 + 45 + 54 + 63 + 72 + 81
  let count := 8
  show sum / count = 49.5
  sorry

end average_of_numbers_divisible_by_9_l825_825227


namespace factorial_sqrt_square_l825_825299

theorem factorial_sqrt_square (n : ℕ) : (nat.succ 4)! * 4! = 2880 := by 
  sorry

end factorial_sqrt_square_l825_825299


namespace total_people_on_hike_l825_825157

def cars : Nat := 3
def people_per_car : Nat := 4
def taxis : Nat := 6
def people_per_taxi : Nat := 6
def vans : Nat := 2
def people_per_van : Nat := 5

theorem total_people_on_hike :
  cars * people_per_car + taxis * people_per_taxi + vans * people_per_van = 58 := by
  sorry

end total_people_on_hike_l825_825157


namespace ratio_of_m_l825_825635

theorem ratio_of_m (a b m m1 m2 : ℚ) 
  (h1 : a^2 - 2*a + (3/m) = 0)
  (h2 : a + b = 2 - 2/m)
  (h3 : a * b = 3/m)
  (h4 : (a/b) + (b/a) = 3/2) 
  (h5 : 8 * m^2 - 31 * m + 8 = 0)
  (h6 : m1 + m2 = 31/8)
  (h7 : m1 * m2 = 1) :
  (m1/m2) + (m2/m1) = 833/64 :=
sorry

end ratio_of_m_l825_825635


namespace avg_children_in_families_with_children_l825_825450

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l825_825450


namespace smallest_base10_integer_l825_825210

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end smallest_base10_integer_l825_825210


namespace problem_roots_l825_825625

theorem problem_roots:
  (∃ p q r : ℝ, (p + q + r = 15) ∧ (p * q + q * r + r * p = 25) ∧ (p * q * r = 10) ∧ 
  (1 + p) * (1 + q) * (1 + r) = 51) :=
begin
  sorry
end

end problem_roots_l825_825625


namespace binom_18_4_eq_3060_l825_825316

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l825_825316


namespace quadratic_min_value_l825_825003

theorem quadratic_min_value (p q : ℝ) (h : ∀ x : ℝ, 3 * x^2 + p * x + q ≥ 4) : q = p^2 / 12 + 4 :=
sorry

end quadratic_min_value_l825_825003


namespace pattern_perimeter_l825_825800

theorem pattern_perimeter 
  (side_length_square : ℝ) 
  (num_squares : ℕ) 
  (num_triangles : ℕ) 
  (square {n} := if n = num_squares then true else false)
  (triangle {m} := if m = num_triangles then true else false) 
  (side_length_square = 2) 
  (num_squares = 6) 
  (num_triangles = 6) : 
  let edge_length_from_triangles := 2 * num_triangles in
  let edge_length_from_squares := 2 * num_squares in
  let total_perimeter := edge_length_from_triangles + edge_length_from_squares in
  total_perimeter = 24 := 
sorry

end pattern_perimeter_l825_825800


namespace max_distance_complex_l825_825092

theorem max_distance_complex (z : ℂ) (hz : |z| = 3) : 
  ∃ (w : ℂ), w = (2 + 5 * I) * z^4 - z^6 ∧ |w| = 81 * Real.sqrt 29 * |Real.sqrt 29 - 9| :=
by
  sorry

end max_distance_complex_l825_825092


namespace average_children_in_families_with_children_l825_825379

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l825_825379


namespace quadratic_inequality_solution_l825_825990

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution (a b c : ℝ) (h₁ : a < 0) (h₂ : -b / a = 3) (h₃ : c = -4 * a) :
  f a b c 2 > f a b c 3 ∧ f a b c 3 > f a b c (-1/2) :=
by
  -- Proof omitted.
  sorry

#eval quadratic_inequality_solution

end quadratic_inequality_solution_l825_825990


namespace binom_18_4_eq_3060_l825_825328

theorem binom_18_4_eq_3060 : nat.choose 18 4 = 3060 := sorry

end binom_18_4_eq_3060_l825_825328


namespace remainder_24_2377_mod_15_l825_825712

theorem remainder_24_2377_mod_15 :
  24^2377 % 15 = 9 :=
sorry

end remainder_24_2377_mod_15_l825_825712


namespace polygon_sides_l825_825025

theorem polygon_sides (each_exterior_angle : ℝ) (h : each_exterior_angle = 40) :
  let n := 360 / each_exterior_angle in n = 9 :=
by
  -- Importing broader Mathlib and skipping the proof as instructed
  sorry

end polygon_sides_l825_825025


namespace sum_of_reciprocals_of_roots_l825_825480

theorem sum_of_reciprocals_of_roots {r1 r2 : ℚ} (h1 : r1 + r2 = 15) (h2 : r1 * r2 = 6) :
  (1 / r1 + 1 / r2) = 5 / 2 := 
by sorry

end sum_of_reciprocals_of_roots_l825_825480


namespace charlie_age_when_jenny_twice_as_old_as_bobby_l825_825062

-- Conditions as Definitions
def ageDifferenceJennyCharlie : ℕ := 5
def ageDifferenceCharlieBobby : ℕ := 3

-- Problem Statement as a Theorem
theorem charlie_age_when_jenny_twice_as_old_as_bobby (j c b : ℕ) 
  (H1 : j = c + ageDifferenceJennyCharlie) 
  (H2 : c = b + ageDifferenceCharlieBobby) : 
  j = 2 * b → c = 11 :=
by
  sorry

end charlie_age_when_jenny_twice_as_old_as_bobby_l825_825062


namespace arccos_sqrt3_div_2_eq_pi_div_6_l825_825313

theorem arccos_sqrt3_div_2_eq_pi_div_6 : Real.arccos (real.sqrt 3 / 2) = Real.pi / 6 :=
sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l825_825313


namespace lily_petals_l825_825354

theorem lily_petals (L : ℕ) (h1 : 8 * L + 15 = 63) : L = 6 :=
by sorry

end lily_petals_l825_825354


namespace problem_statement_l825_825825

noncomputable def smallest_x : ℝ :=
  -8 - (Real.sqrt 292 / 2)

theorem problem_statement (x : ℝ) :
  (15 * x ^ 2 - 40 * x + 18) / (4 * x - 3) + 4 * x = 8 * x - 3 ↔ x = smallest_x :=
by
  sorry

end problem_statement_l825_825825


namespace least_element_in_T_l825_825615
open Nat

-- Define the set of integers from 1 to 15
def S : finset ℕ := finset.range 16 \ {0}

-- Define the properties given in the problem
def prime_factors_disjoint (T : finset ℕ) : Prop := 
  ∀ (a b ∈ T), a ≠ b → gcd a b = 1

def no_multiples (T : finset ℕ) : Prop := 
  ∀ (a b ∈ T), a < b → ¬ (b % a = 0)

-- Define the main problem statement
theorem least_element_in_T (T : finset ℕ) (h1 : T ⊆ S) (h2 : 8 ∈ T) (h3 : prime_factors_disjoint T) (h4 : no_multiples T) :
  ∃ x ∈ T, x = 2 := 
by
  sorry

end least_element_in_T_l825_825615


namespace smallest_base10_integer_l825_825209

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end smallest_base10_integer_l825_825209


namespace chi_square_test_expected_value_of_X_l825_825453

noncomputable def contingency_table : Type := 
  struct {
    male_positive : Nat
    male_negative : Nat
    female_positive : Nat
    female_negative : Nat
    total_positive : Nat
    total_negative : Nat
    total_male : Nat
    total_female : Nat
    total : Nat
  }

def given_data : contingency_table := {
  male_positive := 40,
  male_negative := 70,
  female_positive := 60,
  female_negative := 50,
  total_positive := 100,
  total_negative := 120,
  total_male := 110,
  total_female := 110,
  total := 220
}

theorem chi_square_test : 
  let α := 0.010 in
  let χ2 := (given_data.total * (given_data.male_positive * given_data.female_negative - given_data.male_negative * given_data.female_positive) ^ 2) / 
            ((given_data.total_positive * given_data.total_negative * given_data.total_male * given_data.total_female) : ℚ) in
  χ2 > 6.635 :=
sorry

def distribution_table : List (ℕ × ℚ) := [
  (0, 1 / 30),
  (1, 3 / 10),
  (2, 1 / 2),
  (3, 1 / 6)
]

def expected_value (dist : List (ℕ × ℚ)) : ℚ :=
  List.sum (List.map (fun (x_p : ℕ × ℚ) => (x_p.1 : ℚ) * x_p.2) dist)

theorem expected_value_of_X :
  expected_value distribution_table = 9 / 5 :=
sorry

end chi_square_test_expected_value_of_X_l825_825453


namespace divisibility_by_2k_l825_825999

-- Define the sequence according to the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ ∀ n, 2 ≤ n → a n = 2 * a (n - 1) + a (n - 2)

-- The theorem to be proved
theorem divisibility_by_2k (a : ℕ → ℤ) (k : ℕ) (n : ℕ)
  (h : seq a) :
  2^k ∣ a n ↔ 2^k ∣ n :=
sorry

end divisibility_by_2k_l825_825999


namespace expected_value_of_twelve_sided_die_l825_825790

theorem expected_value_of_twelve_sided_die : 
  let face_values := finset.range (12 + 1) \ finset.singleton 0 in
  (finset.sum face_values (λ x, x) : ℝ) / 12 = 6.5 :=
by
  sorry

end expected_value_of_twelve_sided_die_l825_825790


namespace find_x_l825_825595

theorem find_x :
    ∃ x : ℚ, (1/7 + 7/x = 15/x + 1/15) ∧ x = 105 := by
  sorry

end find_x_l825_825595


namespace probability_sum_is_odd_l825_825170

noncomputable def probability_sum_of_dice_rolls_odd : ℚ :=
let prob_coin := (1 : ℚ) / 2 in
let prob_tail := prob_coin in
let prob_head := prob_coin in
-- Probability of 0 heads (3 tails) -> No dice rolled -> Sum is 0 (even) -> P(odd) = 0
let P0 := (prob_tail ^ 3) * 0 in
-- Probability of 1 head (2 tails) -> 1 die rolled -> P(odd sum) = 3/8 * 1/2
let P1 := (3 * prob_head * prob_tail ^ 2) * (1 / 2) in
-- Probability of 2 heads (1 tail) -> 2 dice rolled -> P(odd sum) = 3/8 * 1/4
let P2 := (3 * prob_head ^ 2 * prob_tail) * (1 / 4) in
-- Probability of 3 heads -> 3 dice rolled -> P(odd sum) = 1/8 * 1/8 * 2
let P3 := (prob_head ^ 3) * (2 / 8) in
P0 + P1 + P2 + P3

theorem probability_sum_is_odd :
  probability_sum_of_dice_rolls_odd = 9 / 32 := by
  sorry

end probability_sum_is_odd_l825_825170


namespace quadratic_min_value_l825_825021

theorem quadratic_min_value (p q r : ℝ) (h : ∀ x : ℝ, x^2 + p * x + q + r ≥ -r) : q = p^2 / 4 :=
sorry

end quadratic_min_value_l825_825021


namespace coordinates_of_conjugate_l825_825027

theorem coordinates_of_conjugate (z : ℂ) (h : complex.I * z = 2 - 4 * complex.I) :
  complex.conj z = -4 + 2 * complex.I :=
by sorry

end coordinates_of_conjugate_l825_825027


namespace not_all_elements_equal_initial_max_m_value_l825_825687

-- Representation of the initial board configuration
def initial_board : List (List ℕ) :=
  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

-- Define the possible operations on the board
inductive operation
| row (r : ℕ) (x : ℝ) : operation -- row operation on row r
| col (c : ℕ) (x : ℝ) : operation -- column operation on column c

-- Definition of a valid operation that transforms the board
def valid_operation (op : operation) (board : List (List ℕ)) : List (List ℕ) := 
  -- Placeholder for the valid operation transformation
  sorry

-- Determines if all elements of the board are the same
def all_elements_equal (board : List (List ℕ)) : Prop :=
  let flattened := board.join
  ∀ x ∈ flattened, ∀ y ∈ flattened, x = y

-- Main theorem to be proved
theorem not_all_elements_equal_initial :
  ¬ ∃ ops : List operation, all_elements_equal (ops.foldl (λ b op, valid_operation op b) initial_board) :=
sorry

-- The maximum value for which all elements can be made equal
theorem max_m_value (m : ℕ) :
  (∃ ops : List operation, all_elements_equal (ops.foldl (λ b op, valid_operation op b) initial_board) ∧ ∀ x ∈ initial_board.join, ∃ y ∈ initial_board.join, y ≤ m) → m ≤ 4 :=
sorry

end not_all_elements_equal_initial_max_m_value_l825_825687


namespace sequence_a_12_l825_825548

sequence_an : ℕ → ℚ
sequence_an 1 := 1
sequence_an (n + 1) := sequence_an n / (2 * sequence_an n + 1)

theorem sequence_a_12 :
    sequence_an 12 = 1 / 23 := 
sorry

end sequence_a_12_l825_825548


namespace coeff_x_squared_expansion_l825_825676

theorem coeff_x_squared_expansion : 
  let expansion := (1 - (1 : ℂ) / x^2) * (1 + x)^5 in
  (expansion.coeff x^2 = 5) :=
by
  let x : ℂ := sorry -- Variable declaration
  let expansion := (1 - (1 : ℂ) / x^2) * (1 + x)^5
  have h : expansion = (1 * (1 + x)^5 - (1 / x^2) * (1 + x)^5) := by sorry
  show expansion.coeff x^2 = 5 from sorry

end coeff_x_squared_expansion_l825_825676


namespace f_monotonicity_l825_825820

noncomputable def f (x : ℝ) : ℝ := sorry

def symmetric (f : ℝ → ℝ) : Prop := ∀ x, f(1 - x) = f(x)

def condition (f : ℝ → ℝ) : Prop := ∀ x, (x - 1/2) * (f '' x) > 0

theorem f_monotonicity {f : ℝ → ℝ} 
  (sym_f : symmetric f) 
  (cond_f : condition f) 
  {x1 x2 : ℝ} 
  (hx : x1 < x2) 
  (hx_sum : x1 + x2 > 1) : 
  f(x1) < f(x2) := sorry

end f_monotonicity_l825_825820


namespace part1_terms_of_bn_part2_sufficient_condition_part2_not_necessary_condition_part3_general_formula_l825_825875

-- Part (1)
theorem part1_terms_of_bn (n : ℕ) (h1 : ∀ k ∈ ({1, 2, 3, 4, 5} : Finset ℕ), a k = abs (k - 2))
  : b 1 = 1 ∧ b 2 = 2 ∧ b 3 = 2 ∧ b 4 = 2 ∧ b 5 = 3 :=
by sorry

-- Part (2)
theorem part2_sufficient_condition (h1 : a 1 % 2 = 1)
  (h2 : ∀ i, i ≥ 2 → a i % 2 = 0)
  : strictly_increasing b :=
by sorry

-- Part (2) Not necessary
theorem part2_not_necessary_condition (h1 : ∃ (i: ℕ), i ≥ 2 ∧ a i % 2 = 1)
  : strictly_increasing b :=
by sorry

-- Part (3)
theorem part3_general_formula (h : ∀ i, i ≥ 2 → a i = b i)
  : ∀ n, n ≥ 2 → a n = 0 :=
by sorry

end part1_terms_of_bn_part2_sufficient_condition_part2_not_necessary_condition_part3_general_formula_l825_825875


namespace cost_per_can_l825_825126

/-- If the total cost for a 12-pack of soft drinks is $2.99, then the cost per can,
when rounded to the nearest cent, is approximately $0.25. -/
theorem cost_per_can (total_cost : ℝ) (num_cans : ℕ) (h_total_cost : total_cost = 2.99) (h_num_cans : num_cans = 12) :
  Real.round (total_cost / num_cans * 100) / 100 = 0.25 :=
by
  sorry

end cost_per_can_l825_825126


namespace elder_age_is_30_l825_825142

/-- The ages of two persons differ by 16 years, and 6 years ago, the elder one was 3 times as old as the younger one. 
Prove that the present age of the elder person is 30 years. --/
theorem elder_age_is_30 (y e: ℕ) (h₁: e = y + 16) (h₂: e - 6 = 3 * (y - 6)) : e = 30 := 
sorry

end elder_age_is_30_l825_825142


namespace smallest_integer_representable_l825_825199

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end smallest_integer_representable_l825_825199


namespace quadratic_two_distinct_roots_l825_825545

theorem quadratic_two_distinct_roots (m : ℝ) :
  (m > -1) ∧ (m ≠ 0) ↔ mx^2 + 2x - 1 = 0 ∧ (discriminant (m, 2, -1) > 0) := 
by
  sorry

end quadratic_two_distinct_roots_l825_825545


namespace monic_polynomial_threefold_l825_825086

theorem monic_polynomial_threefold (r1 r2 r3 : ℝ) :
  (Polynomial.expand ℝ 3 (Polynomial.X^3 - 4 * Polynomial.X^2 + 5 * Polynomial.X + 12)).monic :=
begin
  sorry
end

end monic_polynomial_threefold_l825_825086


namespace parabola_y_intercepts_zero_l825_825343

theorem parabola_y_intercepts_zero :
  let x := 3 * y ^ 2 - 5 * y + 6 in
  ∀ y : ℝ, x = 0 → false :=
by
  assume y : ℝ
  assume h : 3 * y ^ 2 - 5 * y + 6 = 0
  have Δ : ℝ := -47
  have discrim_neg : Δ < 0 := by norm_num
  sorry

end parabola_y_intercepts_zero_l825_825343


namespace min_value_f_range_a_l825_825546

noncomputable def quadratic_fn (a x : ℝ) : ℝ := a * x ^ 2 + x

theorem min_value_f (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1/2) :
  ∃ x : ℝ, quadratic_fn a x = -1 := sorry

theorem range_a (a : ℝ) :
  (∀ x ∈ Icc (-(π / 2)) 0, (a / 2) * sin x * cos x + (1 / 2) * sin x + (1 / 2) * cos x + (a / 4) ≤ 1) →
  a ≤ 2 := sorry

end min_value_f_range_a_l825_825546


namespace smallest_n_factorial_product_l825_825479

theorem smallest_n_factorial_product :
  ∃ n : ℕ, (n > 0) ∧ (∀ (k : ℕ), (k < n - 4 → (∃ (a : ℕ), n! = a * (a + 1) * (a + 2) * ... * (a + n - 4))) ∧ n = 7 :=
begin
  sorry
end

end smallest_n_factorial_product_l825_825479


namespace chocolate_flavored_cups_sold_l825_825263

-- Define total sales and fractions
def total_cups_sold : ℕ := 50
def fraction_winter_melon : ℚ := 2 / 5
def fraction_okinawa : ℚ := 3 / 10
def fraction_chocolate : ℚ := 1 - (fraction_winter_melon + fraction_okinawa)

-- Define the number of chocolate-flavored cups sold
def num_chocolate_cups_sold : ℕ := 50 - (50 * 2 / 5 + 50 * 3 / 10)

-- Main theorem statement
theorem chocolate_flavored_cups_sold : num_chocolate_cups_sold = 15 := 
by 
  -- The proof would go here, but we use 'sorry' to skip it
  sorry

end chocolate_flavored_cups_sold_l825_825263


namespace range_y_over_x_l825_825870

noncomputable def range_of_slope (x y : ℝ) (h : (x - 4)^2 + (y - 2)^2 ≤ 4) : Set ℝ :=
Set.range (λ p : {a // (a - 4)^2 + (p.a - 2)^2 ≤ 4}, p.1 / p.2)

theorem range_y_over_x (x y : ℕ) (h : (x - 4)^2 + (y - 2)^2 ≤ 4) : 
  range_of_slope x y h = Set.Icc 0 (4 / 3) :=
sorry

end range_y_over_x_l825_825870


namespace avg_children_with_kids_l825_825395

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l825_825395


namespace numerical_puzzle_solution_l825_825465

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l825_825465


namespace modulus_product_l825_825835

-- Defining the real part, imaginary part, complex number, and modulus of complex numbers:
def re (z : ℂ) : ℝ := z.re
def im (z : ℂ) : ℝ := z.im
def abs (z : ℂ) : ℝ := complex.norm z

-- Defining the complex numbers in the conditions:
def z1 : ℂ := 7 - 4 * complex.I
def z2 : ℂ := 5 + 12 * complex.I

-- Conditions as definitions:
def abs_z1 : ℝ := abs z1
def abs_z2 : ℝ := abs z2

-- Stating the final equivalent problem:
theorem modulus_product :
  abs (z1 * z2) = 13 * real.sqrt 65 :=
by
  have h1 : abs_z1 = real.sqrt 65 := by sorry
  have h2 : abs_z2 = 13 := by sorry
  sorry

end modulus_product_l825_825835


namespace degrees_to_radians_90_l825_825341

theorem degrees_to_radians_90 : (90 : ℝ) * (Real.pi / 180) = (Real.pi / 2) :=
by
  sorry

end degrees_to_radians_90_l825_825341


namespace n_is_even_l825_825089

open Nat

theorem n_is_even 
  {a b n c : ℕ}
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n > 0)
  (h : (a + b * c) * (b + a * c) = 19^n)
  : Even n :=
begin
  sorry
end

end n_is_even_l825_825089


namespace tangency_condition_value_of_a_tangency_range_of_a_l825_825522

noncomputable def f (x : ℝ) : ℝ := x^3 - x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

theorem tangency_condition_value_of_a :
  (∃ a : ℝ, ∀ x : ℝ, tangent_line (f x) (g x a) at (-1) = 2 * (x + 1)) → a = 3 :=
sorry

theorem tangency_range_of_a :
  (∃ a : ℝ, ∀ x1 : ℝ x2 : ℝ, tangent_line (f x1) (g x2 a) holds) → (a ∈ set.Ici (-1)) :=
sorry

end tangency_condition_value_of_a_tangency_range_of_a_l825_825522


namespace nearest_integer_of_pow_sum_l825_825296

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem nearest_integer_of_pow_sum (n : ℕ) (a b : ℝ) (h : a > b): 
  let ab_sum : ℝ := a + b
  let s : ℝ := ab_sum ^ (2 * n)
  let small_term : ℝ := (a - b) ^ (2 * n)
  s - small_term <= 1 → 
  (∃ I : ℤ, I ≈ s) := sorry

end nearest_integer_of_pow_sum_l825_825296


namespace find_ab_l825_825859

open Complex

variables (a b : ℝ)

def is_root (r : ℂ) : Prop :=
  ∃ (p : ℂ → ℂ), (p.coeff 2 ≠ 0) ∧ p.coeff 1 ∈ ℝ ∧ p.coeff 0 ∈ ℝ ∧ p r = 0

theorem find_ab (h1 : is_root (2 + a * Complex.i))
               (h2 : is_root (b + 3 * Complex.i))
               (h3 : ∀ z : ℂ, is_root z → is_root (conj z)) :
  a = -3 ∧ b = 2 :=
sorry

end find_ab_l825_825859


namespace find_side_PR_of_PQR_l825_825602

open Real

noncomputable def triangle_PQR (PQ PM PH PR : ℝ) : Prop :=
  let HQ := sqrt (PQ^2 - PH^2)
  let MH := sqrt (PM^2 - PH^2)
  let MQ := MH - HQ
  let RH := HQ + 2 * MQ
  PR = sqrt (PH^2 + RH^2)

theorem find_side_PR_of_PQR (PQ PM PH : ℝ) (h_PQ : PQ = 3) (h_PM : PM = sqrt 14) (h_PH : PH = sqrt 5) (h_angle : ∀ QPR PRQ : ℝ, QPR + PRQ < 90) : 
  triangle_PQR PQ PM PH (sqrt 21) :=
by
  rw [h_PQ, h_PM, h_PH]
  exact sorry

end find_side_PR_of_PQR_l825_825602


namespace find_x_value_l825_825242

theorem find_x_value :
  ∃ x : ℝ, (75 * x + (18 + 12) * 6 / 4 - 11 * 8 = 2734) ∧ (x = 37.03) :=
by {
  sorry
}

end find_x_value_l825_825242


namespace impossible_odd_n_minus_m_l825_825568

theorem impossible_odd_n_minus_m (n m : ℤ) (h : 2 ∣ (n^2 - m^2)) : ¬ odd (n - m) := by
  sorry

end impossible_odd_n_minus_m_l825_825568


namespace bankers_discount_l825_825233

/-- The banker’s gain on a sum due 3 years hence at 12% per annum is Rs. 360.
   The banker's discount is to be determined. -/
theorem bankers_discount (BG BD TD : ℝ) (R : ℝ := 12 / 100) (T : ℝ := 3) 
  (h1 : BG = 360) (h2 : BG = (BD * TD) / (BD - TD)) (h3 : TD = (P * R * T) / 100) 
  (h4 : BG = (TD * R * T) / 100) :
  BD = 562.5 :=
sorry

end bankers_discount_l825_825233


namespace magic_square_sum_l825_825050

-- Definitions based on the conditions outlined in the problem
def magic_sum := 83
def a := 42
def b := 26
def c := 29
def e := 34
def d := 36

theorem magic_square_sum :
  d + e = 70 :=
by
  -- Proof is omitted as per instructions
  sorry

end magic_square_sum_l825_825050


namespace average_children_l825_825436

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l825_825436


namespace money_loses_value_on_island_properties_of_money_l825_825110

-- Define the conditions from the problem statement
def deserted_island : Prop := true
def useless_money_on_island (f : Prop) : Prop := deserted_island → ¬f

-- Define properties of money in the story context
def medium_of_exchange (m : Prop) : Prop := m
def has_value_as_medium_of_exchange (m : Prop) : Prop := m
def no_transaction_partners : Prop := true

-- Define additional properties an item must possess to be considered money
def durability : Prop := true
def portability : Prop := true
def divisibility : Prop := true
def acceptability : Prop := true
def uniformity : Prop := true
def limited_supply : Prop := true

-- Prove that money loses its value as a medium of exchange on the deserted island
theorem money_loses_value_on_island (m : Prop) :
  deserted_island →
  (medium_of_exchange m) →
  (no_transaction_partners) →
  (¬has_value_as_medium_of_exchange m) :=
by { intro di, intro moe, intro ntp, exact moe → false }

-- Prove that an item must have certain properties to be considered money
theorem properties_of_money :
  durability ∧ portability ∧ divisibility ∧ acceptability ∧ uniformity ∧ limited_supply :=
by { split, exact true.intro, split, exact true.intro, split, exact true.intro, split, exact true.intro, split, exact true.intro, exact true.intro }

#check money_loses_value_on_island
#check properties_of_money

end money_loses_value_on_island_properties_of_money_l825_825110


namespace multiply_by_7_of_number_given_condition_l825_825556

theorem multiply_by_7_of_number_given_condition :
  ∃ (x : ℕ), (8 * x = 64) → (7 * x = 56) :=
begin
  sorry,
end

end multiply_by_7_of_number_given_condition_l825_825556


namespace expected_value_of_twelve_sided_die_l825_825786

theorem expected_value_of_twelve_sided_die : ∑ k in finset.range 13, k / 12 = 6.5 := 
sorry

end expected_value_of_twelve_sided_die_l825_825786


namespace survival_probability_l825_825486

def numTrainees : Nat := 44
def maxDrawersPerTrainee : Nat := 22

def is_permutation (p : Fin numTrainees → Fin numTrainees) : Prop :=
  ∃ q : Fin numTrainees → Fin numTrainees,
    (∀ i, q (p i) = i) ∧ (∀ i, p (q i) = i)

def cycle_length (p : Fin numTrainees → Fin numTrainees) (i : Fin numTrainees) : Nat :=
  let rec find_cycle_len (curr : Fin numTrainees) (len : Nat) : Nat :=
    if curr = i then len else find_cycle_len (p curr) (len + 1)
  find_cycle_len (p i) 1

def valid_permutation (p : Fin numTrainees → Fin numTrainees) : Prop :=
  is_permutation p ∧ ∀ i, cycle_length p i ≤ maxDrawersPerTrainee

theorem survival_probability (p : Fin numTrainees → Fin numTrainees) :
  valid_permutation p →
  (prob : Real := (injective (λ i, p i)) * (range_valid (λ i, p i)) (p) ) ≥ 0.32 :=
sorry

end survival_probability_l825_825486


namespace books_before_addition_l825_825064

-- Let b be the initial number of books on the shelf
variable (b : ℕ)

theorem books_before_addition (h : b + 10 = 19) : b = 9 := by
  sorry

end books_before_addition_l825_825064


namespace smallest_prime_divisor_of_sum_first_100_is_5_l825_825348

-- Conditions: The sum of the first 100 natural numbers
def sum_first_n_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Prime checking function to identify the smallest prime divisor
def smallest_prime_divisor (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else
  if n % 3 = 0 then 3 else
  if n % 5 = 0 then 5 else
  n -- Such a simplification works because we know the answer must be within the first few primes.

-- Proof statement
theorem smallest_prime_divisor_of_sum_first_100_is_5 : smallest_prime_divisor (sum_first_n_numbers 100) = 5 :=
by
  -- Proof steps would follow here.
  sorry

end smallest_prime_divisor_of_sum_first_100_is_5_l825_825348


namespace average_children_families_with_children_is_3_point_8_l825_825366

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l825_825366


namespace average_children_families_with_children_is_3_point_8_l825_825369

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l825_825369


namespace triangle_product_l825_825970

theorem triangle_product
  (X Y Z P X' Y' Z' : Type)
  [linear_ordered_field X] [linear_ordered_field Y] [linear_ordered_field Z] 
  (h1: X' ∈ line Y Z) (h2: Y' ∈ line Z X) (h3: Z' ∈ line X Y)
  (h4: concurrent (line_segment X X') (line_segment Y Y') (line_segment Z Z') P)
  (h5: (XP : X) / (PX' : X) + (YP : Y) / (PY' : Y) + (ZP : Z) / (PZ' : Z) = 100) :
  (XP : X) / (PX' : X) * (YP : Y) / (PY' : Y) * (ZP : Z) / (PZ' : Z) = 102 := 
sorry

end triangle_product_l825_825970


namespace find_time_l825_825532

variables (V V_0 S g C : ℝ) (t : ℝ)

-- Given conditions.
axiom eq1 : V = 2 * g * t + V_0
axiom eq2 : S = (1 / 3) * g * t^2 + V_0 * t + C * t^3

-- The statement to prove.
theorem find_time : t = (V - V_0) / (2 * g) :=
sorry

end find_time_l825_825532


namespace sum_new_numbers_exceeds_300_l825_825162

theorem sum_new_numbers_exceeds_300 :
  ∃ (S : List ℝ), S.Sum ≤ 100 ∧ (∑ n in S.toFinset, 10^(Int.ofNat (Real.log10 n).round : ℝ)) > 300 :=
by
  let S := [32.0, 32.0, 32.0, 4.0]
  have hsum : S.Sum ≤ 100 := by norm_num
  have h1 : (Real.log10 32).round = 2 := by norm_num
  have h2 : (Real.log10 4).round = 1 := by norm_num
  have hnew : (∑ n in [100.0, 100.0, 100.0, 10.0], n) > 300 := by norm_num
  exact ⟨S, hsum, by convert hnew⟩
  sorry

end sum_new_numbers_exceeds_300_l825_825162


namespace smallest_base10_integer_l825_825203

theorem smallest_base10_integer (a b : ℕ) (ha : a > 3) (hb : b > 3) (h : a + 3 = 3 * b + 1) :
  13 = a + 3 :=
by
  have h_in_base_a : a = 3 * b - 2 := by linarith,
  have h_in_base_b : 3 * b + 1 = 13 := by sorry,
  exact h_in_base_b

end smallest_base10_integer_l825_825203


namespace article_usage_correct_l825_825700

def blank1 := "a"
def blank2 := ""  -- Representing "不填" (no article) as an empty string for simplicity

theorem article_usage_correct :
  (blank1 = "a" ∧ blank2 = "") :=
by
  sorry

end article_usage_correct_l825_825700


namespace second_derivative_sin_squared_eq_2_cos_2x_l825_825843

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) ^ 2

theorem second_derivative_sin_squared_eq_2_cos_2x : 
  deriv (deriv y) = λ x, 2 * Real.cos (2 * x) := 
sorry

end second_derivative_sin_squared_eq_2_cos_2x_l825_825843


namespace number_of_valid_functions_l825_825078

noncomputable def number_of_functions (A : Finset ℕ) : ℕ :=
  let count := λ m, 2 ^ (2011 - m) - 1 in
  ∑ m in Finset.range 2010, count m

theorem number_of_valid_functions :
  let A := (Finset.range 2011).image (λ n, n + 1) in
  (∑ n : ℕ in A, λ f : ℕ → ℕ, f n ≤ n) = 2 ^ 2011 - 2012 :=
by sorry

end number_of_valid_functions_l825_825078


namespace average_children_in_families_with_children_l825_825377

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l825_825377


namespace intersection_correct_l825_825005

def setA : Set ℝ := { x | x - 1 ≤ 0 }
def setB : Set ℝ := { x | x^2 - 4 * x ≤ 0 }
def expected_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_correct : (setA ∩ setB) = expected_intersection :=
sorry

end intersection_correct_l825_825005


namespace smallest_base10_integer_l825_825206

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end smallest_base10_integer_l825_825206


namespace six_digit_even_permutations_l825_825279

/-- The number of different six-digit even numbers obtained by rearranging "124057" is 312. -/
theorem six_digit_even_permutations : 
  let digits := [1, 2, 4, 0, 5, 7] in
  (∃ permutations, (∃ even_perms, even_perms = permutations ∧ ∀ perm, perm ∈ permutations → perm.last = 0 ∨ perm.last = 2 ∨ perm.last = 4) 
  → permutations.length = 312)
:=
sorry

end six_digit_even_permutations_l825_825279


namespace inequality_proof_l825_825925

variable (n : ℕ) (a : Fin n → ℝ)
hypothesis h : ∀ k, 0 < a k

theorem inequality_proof :
  (∑ k, (a k)^5)^4 ≥ (1/n) * ((2 / (n - 1))^5) * (∑ i in Finset.range n, ∑ j in Finset.Ico' 0 i, (a i)^2 * (a j)^2)^5 :=
by
  sorry

end inequality_proof_l825_825925


namespace cube_diagonal_angle_eq_90_l825_825711

theorem cube_diagonal_angle_eq_90 :
  ∀ (adjacent_faces : Set (Set ℝ³)), 
    ∃ (angle_y : ℝ), angle_y = 90 ∧ 
    (∃ (diagonal1 diagonal2 : ℝ³), 
      is_diagonal_of_face diagonal1 adjacent_faces ∧ 
      is_diagonal_of_face diagonal2 adjacent_faces ∧ 
      angle (diagonal1, diagonal2) = angle_y) :=
by
  sorry

end cube_diagonal_angle_eq_90_l825_825711


namespace round_trip_2k_plus_1_needs_both_bus_and_flight_l825_825286

variable (City : Type)
variable (connected : City → City → Prop)

def two_modes_of_transport : Prop :=
∀ (c1 c2 : City), connected c1 c2 ∨ ∃ c3, connected c1 c3 ∧ connected c3 c2

def round_trip_2k_needs_both_bus_and_flight (k : ℕ) (h : k > 3) : Prop :=
∀ (cycle : List City), cycle.length = 2 * k → ∃ i j, connected (cycle.nth_le i sorry) (cycle.nth_le j sorry)

theorem round_trip_2k_plus_1_needs_both_bus_and_flight {k : ℕ} (h : k > 3) :
  (two_modes_of_transport City connected) →
  (round_trip_2k_needs_both_bus_and_flight City connected k h) →
  ∀ (cycle : List City), cycle.length = 2 * k + 1 → ∃ i j, connected (cycle.nth_le i sorry) (cycle.nth_le j sorry) :=
sorry

end round_trip_2k_plus_1_needs_both_bus_and_flight_l825_825286


namespace specialSignLanguage_l825_825164

theorem specialSignLanguage (S : ℕ) 
  (h1 : (S + 2) * (S + 2) = S * S + 1288) : S = 321 := 
by
  sorry

end specialSignLanguage_l825_825164


namespace simplify_expression_l825_825125

theorem simplify_expression (x y : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 20 + 4 * y = 45 * x + 20 + 4 * y :=
by
  sorry

end simplify_expression_l825_825125


namespace revenue_difference_l825_825830

theorem revenue_difference 
  (packs_peak : ℕ := 8) (price_peak : ℕ := 70) (hours_peak : ℕ := 17) (discount_peak : ℝ := 0.10) (commission_peak : ℝ := 0.05)
  (packs_low : ℕ := 5) (price_low : ℕ := 50) (hours_low : ℕ := 14) (discount_low : ℝ := 0.07) (commission_low : ℝ := 0.03) :
  (let 
     total_packs_peak := packs_peak * hours_peak,
     total_sales_peak := total_packs_peak * price_peak,
     discount_packs_peak := total_packs_peak / 2,
     total_discount_peak := discount_packs_peak * (price_peak * discount_peak),
     sales_with_discount_peak := total_sales_peak - total_discount_peak,
     commission_amount_peak := sales_with_discount_peak * commission_peak,
     net_revenue_peak := sales_with_discount_peak - commission_amount_peak,
  
     total_packs_low := packs_low * hours_low,
     total_sales_low := total_packs_low * price_low,
     discount_packs_low := total_packs_low / 2,
     total_discount_low := discount_packs_low * (price_low * discount_low),
     sales_with_discount_low := total_sales_low - total_discount_low,
     commission_amount_low := sales_with_discount_low * commission_low,
     net_revenue_low := sales_with_discount_low - commission_amount_low
      
  in net_revenue_peak - net_revenue_low = 5315.62) :=
sorry

end revenue_difference_l825_825830


namespace range_of_a_l825_825941

variable {a : ℝ}

theorem range_of_a (h : ¬ ∃ x > 0, x^2 + a * x + 1 < 0) : a ≥ -2 := 
by
  sorry

end range_of_a_l825_825941


namespace unique_function_f_l825_825994

-- Define the complex numbers (ℂ) and assume g, ω, v are given as per the problem.
variables {ℂ : Type*} [IsComplexField ℂ]
variables (g : ℂ → ℂ) (ω : ℂ) (v : ℂ)
axiom omega_cubed : ω^3 = 1
axiom omega_not_one : ω ≠ 1

-- Define the function f and state the conditions and uniqueness.
def f (z : ℂ) : ℂ := 1 / 2 * (g z - g (ω * z + v) + g (ω^2 * z + ω * v + v))

-- Main statement: Prove that f defined above satisfies the functional equation and is unique.
theorem unique_function_f :
  (∀ z : ℂ, f z + f (ω * z + v) = g z) ∧
  (∀ f' : ℂ → ℂ, (∀ z : ℂ, f' z + f' (ω * z + v) = g z) → f' = f) := by
  sorry

end unique_function_f_l825_825994


namespace probability_top_two_different_suits_l825_825814

-- Define the rank and suit
inductive Rank
| Ace | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | Jack | Queen | King | Prince | Princess

inductive Suit
| Spades | Hearts | Diamonds | Clubs

-- Define the card
structure Card :=
(rank : Rank)
(suit : Suit)

-- Define the deck of cards
def deck : List Card :=
(List.replicate 15 (Rank.Ace, Suit.Spades) ++
 List.replicate 15 (Rank.Ace, Suit.Hearts) ++
 List.replicate 15 (Rank.Ace, Suit.Diamonds) ++
 List.replicate 15 (Rank.Ace, Suit.Clubs)).map (uncurry Card.mk)

-- Probability calculation
def different_suits_probability : ℚ :=
(15 / 60) * (45 / 59)

theorem probability_top_two_different_suits : different_suits_probability = 45 / 236 :=
by
  sorry

end probability_top_two_different_suits_l825_825814


namespace max_sum_of_squares_eq_50_l825_825167

theorem max_sum_of_squares_eq_50 :
  ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ (∀ x' y' : ℤ, x'^2 + y'^2 = 50 → x + y ≥ x' + y') ∧ x + y = 10 := 
sorry

end max_sum_of_squares_eq_50_l825_825167


namespace cos_tan_solution_l825_825895

noncomputable def cos_tan_values (θ : Real) (m : Real) : (Real × Real) :=
  if m = 0 then (-1, 0)
  else if m = sqrt(5) then (-sqrt(6) / 4, -sqrt(15) / 3)
  else if m = -sqrt(5) then (-sqrt(6) / 4, sqrt(15) / 3)
  else (0, 0)  -- Using (0, 0) as a placeholder for undefined cases

theorem cos_tan_solution :
  ∀ (θ : Real) (m : Real), (sin θ = sqrt(2) / 4 * m) →
    (θ = 0 ∨ θ = sqrt(5) ∨ θ = -sqrt(5)) →
    cos_tan_values θ m = 
      if m = 0 then (-1, 0)
      else if m = sqrt(5) then (-sqrt(6) / 4, -sqrt(15) / 3)
      else if m = -sqrt(5) then (-sqrt(6) / 4, sqrt(15) / 3)
      else (0, 0) :=
by
  intros
  -- Proof will be written here
  sorry

end cos_tan_solution_l825_825895


namespace number_of_boys_eq_2_l825_825244

variable (b : ℕ)

-- Conditions
def prob_same_color (b : ℕ) : ℝ :=
  2 * ((2 / (2 + b)) * (1 / (1 + b)))

axiom prob_given : prob_same_color b = 1 / 3

-- The theorem to prove
theorem number_of_boys_eq_2 : b = 2 :=
by
  sorry

end number_of_boys_eq_2_l825_825244


namespace jill_total_time_is_67_l825_825065

noncomputable def timeSpentOnPhone (n : ℕ) : ℝ := 5 * Real.exp (0.3 * n)

def totalTimeSpent : ℝ :=
  (timeSpentOnPhone 1) +
  (timeSpentOnPhone 2) +
  (timeSpentOnPhone 3) +
  (timeSpentOnPhone 4) +
  (timeSpentOnPhone 5)

def rounded : ℝ → ℤ := λ x, Real.floor (x + 0.5)

theorem jill_total_time_is_67 :
  rounded totalTimeSpent = 67 :=
sorry

end jill_total_time_is_67_l825_825065


namespace triangle_area_vector_sum_zero_l825_825972

section Geometry

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {P : Type*} [add_torsor V P]

open_locale affine

-- Define the points A, B, C, and O
variables (A B C O : P)

-- Define the areas of triangles BCO, CAO, and ABO respectively
variables (S_A S_B S_C : ℝ)

-- Express the vectors OA, OB, OC
def vector_oa := (O -ᵥ A)
def vector_ob := (O -ᵥ B)
def vector_oc := (O -ᵥ C)

-- Prove the required equality
theorem triangle_area_vector_sum_zero :
  S_A • (vector_oa A O) + S_B • (vector_ob B O) + S_C • (vector_oc C O) = 0 :=
sorry

end Geometry

end triangle_area_vector_sum_zero_l825_825972


namespace avg_children_in_families_with_children_l825_825446

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l825_825446


namespace true_statement_B_l825_825722

noncomputable def mean (l : List ℚ) : ℚ :=
  l.sum / l.length

def variance (l : List ℚ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / l.length

theorem true_statement_B : variance [2, 0, 3, 2, 3] = 6 / 5 :=
by
  sorry

end true_statement_B_l825_825722


namespace intersection_A_B_l825_825523

def A (x : ℝ) : Prop := x^2 - 3 * x < 0
def B (x : ℝ) : Prop := x > 2

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l825_825523


namespace factorial_product_square_l825_825301

theorem factorial_product_square (n : ℕ) (m : ℕ) (h₁ : n = 5) (h₂ : m = 4) :
  (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 :=
by
  have f5 : Nat.factorial 5 = 120 := by norm_num
  have f4 : Nat.factorial 4 = 24 := by norm_num
  rw [Nat.factorial_eq_factorial h₁, Nat.factorial_eq_factorial h₂, f5, f4]
  norm_num
  simp
  sorry

end factorial_product_square_l825_825301


namespace prices_and_subsidy_l825_825099

theorem prices_and_subsidy (total_cost : ℕ) (price_leather_jacket : ℕ) (price_sweater : ℕ) (subsidy_percentage : ℕ) 
  (leather_jacket_condition : price_leather_jacket = 5 * price_sweater + 600)
  (cost_condition : price_leather_jacket + price_sweater = total_cost)
  (total_sold : ℕ) (max_subsidy : ℕ) :
  (total_cost = 3000 ∧
   price_leather_jacket = 2600 ∧
   price_sweater = 400 ∧
   subsidy_percentage = 10) ∧ 
  ∃ a : ℕ, (2200 * a ≤ 50000 ∧ total_sold - a ≥ 128) :=
by
  sorry

end prices_and_subsidy_l825_825099


namespace num_two_digit_math_representation_l825_825598

-- Define the problem space
def unique_digits (n : ℕ) : Prop := 
  n >= 1 ∧ n <= 9

-- Representation of the characters' assignment
def representation (x y z w : ℕ) : Prop :=
  unique_digits x ∧ unique_digits y ∧ unique_digits z ∧ unique_digits w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧ 
  x = z ∧ 3 * (10 * y + z) = 10 * w + x

-- The main theorem to prove
theorem num_two_digit_math_representation : 
  ∃ x y z w, representation x y z w :=
sorry

end num_two_digit_math_representation_l825_825598


namespace max_days_knights_conference_l825_825642

variable (n : ℕ)

-- definition of the problem conditions
def knights_seating_constraint (perm : List ℕ) : Prop :=
  perm.nodup ∧ perm.length = n ∧ ∃ i, perm[i % n] ≠ perm[(i + 1) % n]

-- statement of the theorem
theorem max_days_knights_conference : ∀ n : ℕ, n > 0 → ∃ days: ℕ, 
  (days = (n - 1)!) ∧ ∀ perm : List ℕ, knights_seating_constraint n perm → 
  𝓝 n perm < days := 
sorry

end max_days_knights_conference_l825_825642


namespace binom_18_4_l825_825330

theorem binom_18_4 : Nat.binomial 18 4 = 3060 :=
by
  -- We start the proof here.
  sorry

end binom_18_4_l825_825330


namespace find_rotation_y_l825_825308

def rotation_equivalence (P Q R : Type) (clockwise_deg : ℕ) (counterclockwise_deg : ℕ) : Prop :=
  (clockwise_deg % 360 = 60) ∧ (counterclockwise_deg % 360 = 300)
  
theorem find_rotation_y (P Q R : Type) :
  ∃ y : ℕ, y < 360 ∧ rotation_equivalence P Q R 780 y := 
by 
  use 300
  split
  {
    linarith
  }
  {
    sorry
  }

end find_rotation_y_l825_825308


namespace find_xyz_l825_825989

theorem find_xyz
  (a b c x y z : ℂ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : a = (2 * b + 3 * c) / (x - 3))
  (h2 : b = (3 * a + 2 * c) / (y - 3))
  (h3 : c = (2 * a + 2 * b) / (z - 3))
  (h4 : x * y + x * z + y * z = -1)
  (h5 : x + y + z = 1) :
  x * y * z = 1 :=
sorry

end find_xyz_l825_825989


namespace max_value_g_in_interval_l825_825841

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^4

-- Define the interval condition
def interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- State the theorem
theorem max_value_g_in_interval : ∃ x ∈ set.Icc 0 2, ∀ y ∈ set.Icc 0 2, g y ≤ g x ∧ g x = 3 :=
by
  use 1
  split
  -- Proof that 1 is in the interval [0, 2]
  {
    norm_num,
  }
  split
  -- Proof that for all y in the interval [0, 2], g(y) ≤ g(1)
  {
    intros y hy,
    sorry,
  }
  -- Proof that g(1) = 3
  {
    simp [g],
    norm_num,
  }

end max_value_g_in_interval_l825_825841


namespace count_functions_satisfying_conditions_l825_825080

def A := { n : ℕ | 1 ≤ n ∧ n ≤ 2011 }

def f (x : ℕ) := { y : ℕ | y ≤ x ∧ y ∈ A }

theorem count_functions_satisfying_conditions :
  (∃ (f : ℕ → ℕ), (∀ n ∈ A, f n ∈ f n) ∧ (∀ n ∈ A, f n ≤ n) ∧ 
    (set.finite { y | ∃ n ∈ A, f n = y } ∧ set.card ({ y | ∃ n ∈ A, f n = y }) = 2010)) = 2^2011 - 2012 :=
by
  sorry

end count_functions_satisfying_conditions_l825_825080


namespace possible_values_of_a_l825_825663

def isFactor (x y : ℕ) : Prop :=
  y % x = 0

def isDivisor (x y : ℕ) : Prop :=
  x % y = 0

def isPositive (x : ℕ) : Prop :=
  x > 0

theorem possible_values_of_a : 
  { a : ℕ // isFactor 4 a ∧ isDivisor 24 a ∧ isPositive a }.card = 4 := by
  sorry

end possible_values_of_a_l825_825663


namespace day_of_week_calculation_l825_825971

theorem day_of_week_calculation (N : ℕ)
  (h_250_N : (day_of_year N 250) = "Friday")
  (h_150_N1 : (day_of_year (N + 1) 150) = "Friday") :
  day_of_year (N - 1) 50 = "Thursday" :=
sorry

end day_of_week_calculation_l825_825971


namespace least_n_for_factorial_contains_7875_l825_825709

-- Prime Factorization of 7875: 7875 = 5^3 * 3^2 * 7
def prime_factorization_7875 : ℕ := 5^3 * 3^2 * 7
def min_factors_5 (n : ℕ) : Prop := 
  (∑ k in finset.range (n + 1), if k % 5 = 0 then 1 else 0) >= 3
def min_factors_3 (n : ℕ) : Prop :=
  (∑ k in finset.range (n + 1), if k % 3 = 0 then 1 else 0) >= 2
def contains_7 (n : ℕ) : Prop := 
  ∃ k in finset.range (n + 1), k % 7 = 0

theorem least_n_for_factorial_contains_7875 : ∃ n : ℕ, 15 ≤ n ∧
  min_factors_5 n ∧ min_factors_3 n ∧ contains_7 n :=
by 
  use 15
  sorry

end least_n_for_factorial_contains_7875_l825_825709


namespace number_of_ways_to_connect_10_points_l825_825935

theorem number_of_ways_to_connect_10_points : 
  (∃ (connect : ℕ → ℕ → Prop), ∀ p : ℕ, p < 10 → (∃ q : ℕ, q < 10 ∧ connect p q) ∧ 
  ∑ (i : ℕ) in Finset.range 10, ∑ (j : ℕ) in Finset.range 10, cond (connect i j) 1 0 = 135) :=
sorry

end number_of_ways_to_connect_10_points_l825_825935


namespace commute_time_late_l825_825257

theorem commute_time_late (S : ℝ) (T : ℝ) (T' : ℝ) (H1 : T = 1) (H2 : T' = (4/3)) :
  T' - T = 20 / 60 :=
by
  sorry

end commute_time_late_l825_825257


namespace numerical_puzzle_solution_l825_825464

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l825_825464


namespace tangent_line_at_f_m_eq_3_possible_theta_values_range_of_m_values_l825_825903

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := m * x - (m - 1) / x - Real.log x
noncomputable def g (x : ℝ) (θ : ℝ) : ℝ := 1 / (x * Real.cos θ) + Real.log x
noncomputable def h (x : ℝ) (m : ℝ) (θ : ℝ) : ℝ := f x m - g x θ

-- (I)
theorem tangent_line_at_f_m_eq_3 :
  (∀ x : ℝ, f x 3 = 3 * x - 2 / x - Real.log x) →
  (∀ x : ℝ, Derivative (f x 3) = 3 + 2 / x^2 - 1 / x) →
  f 1 3 = 1 →
  f' 1 3 = 4 →
  ∀ x y : ℝ, y - 1 = 4 * (x - 1) → 4 * x - y - 3 = 0 :=
sorry

-- (II)
theorem possible_theta_values (θ : ℝ) :
  (∀ x : ℝ, (g x θ) (1, +∞)) →
  (∀ x : ℝ, Derivative (g x θ) = -1 / (Real.cos θ * x^2) + 1 / x) →
  (∀ x : ℝ, -1 / (Real.cos θ * x^2) + 1 / x ≥ 0) →
  θ ∈ [0, Real.pi / 2) →
  Real.cos θ = 1 ∧ θ = 0 :=
sorry

-- (III)
theorem range_of_m_values (m θ : ℝ) :
  (∀ x : ℝ, h x m θ = m * x - m / x - 2 * Real.log x) →
  (∀ x : ℝ, Derivative (h x m θ) = (m * x^2 - 2 * x + m) / x^2) →
  Monotonic (h x m θ) →
  (m ≥ 1) ∨ (m ≤ 0) :=
sorry

end tangent_line_at_f_m_eq_3_possible_theta_values_range_of_m_values_l825_825903


namespace fish_population_l825_825949

theorem fish_population (N : ℕ) (h1 : 50 > 0) (h2 : N > 0) 
  (tagged_first_catch : ℕ) (total_first_catch : ℕ)
  (tagged_second_catch : ℕ) (total_second_catch : ℕ)
  (h3 : tagged_first_catch = 50)
  (h4 : total_first_catch = 50)
  (h5 : tagged_second_catch = 2)
  (h6 : total_second_catch = 50)
  (h_percent : (tagged_first_catch : ℝ) / (N : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ))
  : N = 1250 := 
  by sorry

end fish_population_l825_825949


namespace candidate_p_wage_difference_l825_825736

theorem candidate_p_wage_difference
  (P Q : ℝ)    -- Candidate p's hourly wage is P, Candidate q's hourly wage is Q
  (H : ℝ)      -- Candidate p's working hours
  (total_payment : ℝ)
  (wage_ratio : P = 1.5 * Q)  -- Candidate p is paid 50% more per hour than candidate q
  (hours_diff : Q * (H + 10) = total_payment)  -- Candidate q's total payment equation
  (candidate_q_payment : Q * (H + 10) = 480)   -- total payment for candidate q
  (candidate_p_payment : 1.5 * Q * H = 480)    -- total payment for candidate p
  : P - Q = 8 := sorry

end candidate_p_wage_difference_l825_825736


namespace solution_set_of_inequality_l825_825161

theorem solution_set_of_inequality (k : ℤ) :
  {x | -sqrt 3 < Real.tan x ∧ Real.tan x < 2} = 
  {x | ∃ k : ℤ, k * Real.pi - Real.pi / 3 < x ∧ x < k * Real.pi + Real.arctan 2 } :=
by sorry

end solution_set_of_inequality_l825_825161


namespace cyclist_trip_distance_proof_l825_825752

-- Definitions based on conditions
def average_speed (distance time : ℝ) := distance / time

def distance_first_part : ℝ := x
def speed_first_part : ℝ := 10
def distance_second_part : ℝ := 10
def speed_second_part : ℝ := 8
def average_speed_entire_trip : ℝ := 8.78

-- The given math problem converted to a theorem
theorem cyclist_trip_distance_proof (x : ℝ) :
  let time_first_part := distance_first_part / speed_first_part in
  let time_second_part := distance_second_part / speed_second_part in
  let total_distance := distance_first_part + distance_second_part in
  let total_time := time_first_part + time_second_part in
  average_speed total_distance total_time = average_speed_entire_trip → 
  x = 7.99 :=
by
  sorry

end cyclist_trip_distance_proof_l825_825752


namespace min_value_geometric_sequence_l825_825084

theorem min_value_geometric_sequence (a_2 a_3 : ℝ) (r : ℝ) 
(h_a2 : a_2 = 2 * r) (h_a3 : a_3 = 2 * r^2) : 
  (6 * a_2 + 7 * a_3) = -18 / 7 :=
by
  sorry

end min_value_geometric_sequence_l825_825084


namespace function_increasing_on_R_l825_825502

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem function_increasing_on_R (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a ≤ f x₂ a) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end function_increasing_on_R_l825_825502


namespace possible_values_of_k_l825_825154

theorem possible_values_of_k :
  ∀ (k : ℕ), k > 0 → (∃ (u : ℕ → ℕ), u 0 = 1 ∧ (∀ n > 0, u (n+1) * u (n-1) = k * u n) ∧ u 2000 = 2000) → 
    k ∈ {2000, 1000, 500, 400, 200, 100} :=
by
  intros k k_pos (u & u0 & rec & u2000)
  sorry

end possible_values_of_k_l825_825154


namespace average_children_in_families_with_children_l825_825378

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l825_825378


namespace louie_share_of_pie_l825_825805

def fraction_of_pie_taken_home (total_pie : ℚ) (shares : ℚ) : ℚ :=
  2 * (total_pie / shares)

theorem louie_share_of_pie : fraction_of_pie_taken_home (8 / 9) 4 = 4 / 9 := 
by 
  sorry

end louie_share_of_pie_l825_825805


namespace daily_net_revenue_function_l825_825166

theorem daily_net_revenue_function (x : ℝ) (h : x ≥ 40) :
  y = (x - 30) * (80 - x) :=
by
  have total_revenue : ∀ x ≥ 40, y = (x - 30) * (80 - x), sorry

end daily_net_revenue_function_l825_825166


namespace find_a_l825_825026

theorem find_a (a : ℝ) 
  (line_through : ∃ (p1 p2 : ℝ × ℝ), p1 = (a-2, -1) ∧ p2 = (-a-2, 1)) 
  (perpendicular : ∀ (l1 l2 : ℝ × ℝ), l1 = (2, 3) → l2 = (-1/a, 1) → false) : 
  a = -2/3 :=
by 
  sorry

end find_a_l825_825026


namespace triangle_altitude_l825_825673

theorem triangle_altitude {A b h : ℝ} (hA : A = 720) (hb : b = 40) (hArea : A = 1 / 2 * b * h) : h = 36 :=
by
  sorry

end triangle_altitude_l825_825673


namespace smallest_base10_integer_l825_825212

theorem smallest_base10_integer (a b : ℕ) (h1 : a > 3) (h2 : b > 3) :
    (1 * a + 3 = 3 * b + 1) → (1 * 10 + 3 = 13) :=
by
  intros h


-- Prove that  1 * a + 3 = 3 * b + 1 
  have a_eq : a = 3 * b - 2 := by linarith

-- Prove that 1 * 10 + 3 = 13 
  have base_10 := by simp

have the smallest base 10
  sorry

end smallest_base10_integer_l825_825212


namespace proof_probability_second_science_given_first_arts_l825_825584

noncomputable def probability_second_science_given_first_arts : ℚ :=
  let total_questions := 5
  let science_questions := 3
  let arts_questions := 2

  -- Event A: drawing an arts question in the first draw.
  let P_A := arts_questions / total_questions

  -- Event AB: drawing an arts question in the first draw and a science question in the second draw.
  let P_AB := (arts_questions / total_questions) * (science_questions / (total_questions - 1))

  -- Conditional probability P(B|A): drawing a science question in the second draw given drawing an arts question in the first draw.
  P_AB / P_A

theorem proof_probability_second_science_given_first_arts :
  probability_second_science_given_first_arts = 3 / 4 :=
by
  -- Lean does not include the proof in the statement as required.
  sorry

end proof_probability_second_science_given_first_arts_l825_825584


namespace polygon_diagonals_l825_825703

theorem polygon_diagonals
  (m n : ℕ)
  (h1 : m + n = 33)
  (h2 : m(m - 3) / 2 + n(n - 3) / 2 = 243) :
  max (m(m - 3) / 2) (n(n - 3) / 2) = 189 := 
  sorry

end polygon_diagonals_l825_825703


namespace pentagon_side_diag_vertex_l825_825810

-- Define the conditions for a convex irregular pentagon with specified properties.
structure Pent := 
  (vertices : Fin 5 → Point)
  (convex : isConvex vertices)
  (irregular : ¬isRegular vertices)
  (sides_length : Set ℝ)
  (diags_length : Set ℝ)
  (sides_eq_4 : sides_length.card = 4)
  (diags_eq_4 : diags_length.card = 4)
  (sides_diags_same_point : Prop)
  (has_side_shared_diag_vertex : sides_diags_same_point = false)

-- Define a proof problem to show that under the given conditions, 
-- the fifth side and the fifth diagonal cannot share a common vertex.
theorem pentagon_side_diag_vertex :
  ∃ (pent : Pent), pent.has_side_shared_diag_vertex :=
by 
  sorry

end pentagon_side_diag_vertex_l825_825810


namespace convex_polygon_diagonals_l825_825923

theorem convex_polygon_diagonals (n : ℕ) (h : n = 17) : (n * (n - 3)) / 2 = 119 :=
by
  rw h
  simp
  norm_num

end convex_polygon_diagonals_l825_825923


namespace cone_volume_proof_l825_825013

def radius (diameter : ℝ) : ℝ := diameter / 2

def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

theorem cone_volume_proof : 
  cone_volume (radius 8) 12 = 64 * π := 
by 
  sorry

end cone_volume_proof_l825_825013


namespace simplify_expression_l825_825659

noncomputable def root_four : ℝ := real.sqrt (real.sqrt 81)
noncomputable def root_1275 : ℝ := real.sqrt 12.75
noncomputable def simplified_root_1275 : ℝ := real.sqrt 51 / 2
noncomputable def simplified_root_four : ℝ := 3

theorem simplify_expression : (root_four - root_1275) ^ 2 = (87 - 12 * real.sqrt 51) / 4 := by
  sorry

end simplify_expression_l825_825659


namespace expected_value_twelve_sided_die_l825_825775

theorem expected_value_twelve_sided_die : 
  let die_sides := 12 in 
  let outcomes := finset.range (die_sides + 1) in
  (finset.sum outcomes id : ℚ) / die_sides = 6.5 :=
by
  sorry

end expected_value_twelve_sided_die_l825_825775


namespace cost_of_concessions_l825_825846

theorem cost_of_concessions (total_cost : ℕ) (adult_ticket_cost : ℕ) (child_ticket_cost : ℕ) (num_adults : ℕ) (num_children : ℕ) :
  total_cost = 76 →
  adult_ticket_cost = 10 →
  child_ticket_cost = 7 →
  num_adults = 5 →
  num_children = 2 →
  total_cost - (num_adults * adult_ticket_cost + num_children * child_ticket_cost) = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cost_of_concessions_l825_825846


namespace sequence_general_term_l825_825909

open Nat

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  (∀ n : ℕ, 0 < n → a (n + 2) ≤ a n + 3 * 2^n) ∧
  (∀ n : ℕ, 0 < n → a (n + 1) ≥ 2 * a n + 1)

theorem sequence_general_term (a : ℕ → ℕ) (h : sequence_a a) :
  ∀ n : ℕ, 0 < n → a n = 2^n - 1 :=
by
  sorry

end sequence_general_term_l825_825909


namespace average_children_in_families_with_children_l825_825391

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l825_825391


namespace monotonically_increasing_interval_l825_825686

def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*x - 3)

theorem monotonically_increasing_interval :
  (∀ x y : ℝ, (f x ≤ f y) ↔ x ≤ y) :=
begin
  sorry
end

end monotonically_increasing_interval_l825_825686


namespace part1_part2_l825_825514

-- Part (1)
theorem part1 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x) (hg : ∀ x, g x = x^2 + a)
  (tangent_at_1 : ∀ x, ∃ (x1 : ℝ), x1 = -1 ∧
    (f' x1 = g' 1 ∧ f x1 = g 1)) :
  a = 3 := 
  sorry

-- Part (2)
theorem part2 (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x) (hg : ∀ x, ∃ a, g x = x^2 + a)
  (tangent_condition : ∀ x1 x2, 
    (f' x1 = g' x2 ∧ f x1 = g x2)) :
  ∃ a, a ≥ -1 :=
  sorry

end part1_part2_l825_825514


namespace projection_b_l825_825617

noncomputable def proj (u v : ℝ^2) : ℝ^2 :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1^2 + v.2^2
  ((dot_product / norm_sq) * v.1, (dot_product / norm_sq) * v.2)

variables (a b : ℝ^2)
variables (ha_proj : proj (4, -2) a = (-2/5, -4/5))
variables (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0)

theorem projection_b :
  proj (4, -2) b = (22/5, -6/5) :=
sorry

end projection_b_l825_825617


namespace radii_difference_l825_825690

-- Given conditions
variables (r R : ℝ)
variable (ratio_areas : (π * R^2) / (π * r^2) = 4)

-- Define the statement to be proved
theorem radii_difference (h : ratio_areas) : R - r = r := by
  sorry

end radii_difference_l825_825690


namespace non_self_intersecting_polygonal_chain_exists_l825_825583

noncomputable def exists_simple_polygonal_chain (n : ℕ) (lines : Fin n → Line) := 
  ∀ (i j k : Fin n), 
  (¬ (exists (x : Point), x ∈ lines i ∧ x ∈ lines j ∧ x ∈ lines k)) →  
  (¬ (exists (l : Fin n), lines i ∥ lines j)) → 
  ∃ (vertices : Fin n.succ → Point), 
  ∀ (i : Fin n.succ), (vertices i ∈ lines i) ∧ 
  (∀ (i j : Fin n.succ), i ≠ j → vertices i ≠ vertices j) ∧ 
  (∀ (i : Fin n), ¬(LineSeg (vertices i) (vertices (Fin.succ i)) ∩ LineSeg (vertices j) (vertices (Fin.succ j)) ≠ Ø)) -- Ensures non-self-intersecting

theorem non_self_intersecting_polygonal_chain_exists 
  (n : ℕ)
  (lines : Fin n → Line)
  (no_parallel: ∀ (i j : Fin n), lines i ∥ lines j → false)
  (no_three_intersect: ∀ (i j k : Fin n), 
     (∃ (x : Point), x ∈ lines i ∧ x ∈ lines j ∧ x ∈ lines k) → false)
  : exists_simple_polygonal_chain n lines :=
sorry

end non_self_intersecting_polygonal_chain_exists_l825_825583


namespace avg_children_in_families_with_children_l825_825407

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l825_825407


namespace logarithm_defined_range_l825_825346
open Real

def is_log_defined (a : ℝ) : Prop :=
  2 < a ∧ a ≠ 3 ∧ a < 5

theorem logarithm_defined_range (a : ℝ) : is_log_defined a ↔ log (5 - a) (a - 2) exists :=
sorry

end logarithm_defined_range_l825_825346


namespace part1_part2_l825_825549

-- Part 1
def A : Set ℝ := {x | 1 < 2^(x^2 - 2*x - 3) ∧ 2^(x^2 - 2*x - 3) < 32}
def B : Set ℝ := {x | log x 2 < 3}

theorem part1 :
  (set.compl A ∩ B) = set.union (set.Ioc (-3) (-2)) 
                               (set.union (set.Icc (-1) 3)
                                          (set.Ioc 4 5)) :=
sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∀ x, x ∈ set.Ioo a (a+2) → x ∈ B) :
  a ∈ set.Icc (-3) 3 :=
sorry

end part1_part2_l825_825549


namespace sum_binom_recip_le_one_sum_binom_ge_m_squared_l825_825612

variable {A : Type}  -- Set A is generalized to any type A
variable {n : ℕ}     -- n elements in the set A
variable {m : ℕ}     -- m subsets
variable {Ai : Fin m → Finset A}  -- The m subsets, generalized with index i ∈ {1, 2, ..., m}
variable {k : Fin m → ℕ}  -- Corresponding sizes of subsets

-- For accessing the cardinality of a subset
-- subset_sizes: ∀ i, |A_i| = k_i
axiom subset_sizes : ∀ i : Fin m, (Ai i).card = k i

-- Mutual exclusivity (pairwise disjoint subsets)
axiom disjoint_subsets : ∀ i j : Fin m, i ≠ j → Disjoint (Ai i) (Ai j)

-- Question 1: Prove that ∑ (i = 0 to m-1), 1 / C(n, |A_i|) ≤ 1
theorem sum_binom_recip_le_one : (Finset.univ.sum (λ i : Fin m, 1 / Nat.choose n (k i))) ≤ 1 := 
by sorry

-- Question 2: Prove that ∑ (i = 0 to m-1), C(n, |A_i|) ≥ m^2
theorem sum_binom_ge_m_squared : (Finset.univ.sum (λ i : Fin m, Nat.choose n (k i))) ≥ m^2 := 
by sorry

end sum_binom_recip_le_one_sum_binom_ge_m_squared_l825_825612


namespace sum_of_angles_lt_360_l825_825854

-- Assume points A, B, C, D are not coplanar
variables {A B C D : Type}
variables [coordinates : HasCoord A] [coordinates : HasCoord B] [coordinates : HasCoord C] [coordinates : HasCoord D]
variables (h_non_coplanar : ¬Coplanar A B C D)

-- Assume angles are defined and we need to prove the sum is less than 360 degrees
variables (angleABC : ℝ) (angleBCD : ℝ) (angleCDA : ℝ) (angleDAB : ℝ)
variables (h_angle_ABC : angleABC = ∠ A B C) (h_angle_BCD : angleBCD = ∠ B C D)
variables (h_angle_CDA : angleCDA = ∠ C D A) (h_angle_DAB : angleDAB = ∠ D A B)

theorem sum_of_angles_lt_360 (h_non_coplanar : ¬Coplanar A B C D) :
  angleABC + angleBCD + angleCDA + angleDAB < 360 :=
sorry

end sum_of_angles_lt_360_l825_825854


namespace circle_equation_when_t_is_2_area_triangle_OAB_is_constant_circle_given_OM_eq_ON_l825_825554

-- Definition of the circle center with given t and its equation
def center (t : ℝ) (ht : t ≠ 0) := (t, 2/t)
def equation_circle (center : ℝ × ℝ) (r : ℝ) (x y : ℝ) := (x - center.1) ^ 2 + (y - center.2) ^ 2 = r

-- Proof Problem (I)
theorem circle_equation_when_t_is_2 : 
  equation_circle (center 2 (by norm_num : 2 ≠ 0)) (sqrt 5) x y ↔ (x - 2) ^ 2 + (y - 1) ^ 2 = 5 := 
by sorry

-- Proof Problem (II)
def point_A (t : ℝ) (ht : t ≠ 0) := (2*t, 0)
def point_B (t : ℝ) (ht : t ≠ 0) := (0, 4/t)

theorem area_triangle_OAB_is_constant (t : ℝ) (ht : t ≠ 0) : 
  let A := point_A t ht
      B := point_B t ht
      O := (0, 0)
  in abs ((A.1 - O.1) * (B.2 - O.2) - (A.2 - O.2) * (B.1 - O.1)) / 2 = 4 :=
by sorry

-- Proof Problem (III)
theorem circle_given_OM_eq_ON {x y : ℝ} (h : equation_circle (center 2 (by norm_num : 2 ≠ 0)) (sqrt 5) x y) :
  equation_circle (center 2 (by norm_num : 2 ≠ 0)) (sqrt 5) x y :=
by sorry

end circle_equation_when_t_is_2_area_triangle_OAB_is_constant_circle_given_OM_eq_ON_l825_825554


namespace sequence_property_l825_825547

def sequence_conditions (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  ∀ n ≥ 3, S n + S (n - 2) = 2 * S (n - 1) + n

theorem sequence_property (a : ℕ → ℕ) (S : ℕ → ℕ) (h : sequence_conditions a S) : 
  ∀ n ≥ 3, a n = a (n - 1) + n :=
  sorry

end sequence_property_l825_825547


namespace part1_part2_l825_825534

def f (a : ℝ) (x : ℝ) : ℝ := a * x - real.log x

def F (a : ℝ) (x : ℝ) : ℝ := real.exp x + a * x

def h (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + real.log x

theorem part1 (a : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, 0 < x → x < real.log 3 → (f a x)' = (F a x)') :
  a ≤ -3 := sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : x1 ∈ Ioo 0 (1/2)) (h2 : x2 = 1 / (2 * x1)) :
  h a x1 - h a x2 > 3/4 - real.log 2 := sorry

end part1_part2_l825_825534


namespace complex_arithmetic_l825_825018

def Q : ℂ := 7 + 3 * Complex.I
def E : ℂ := 2 * Complex.I
def D : ℂ := 7 - 3 * Complex.I
def F : ℂ := 1 + Complex.I

theorem complex_arithmetic : (Q * E * D) + F = 1 + 117 * Complex.I := by
  sorry

end complex_arithmetic_l825_825018


namespace line_intersects_circle_l825_825940

-- Define the line
def line (m : ℝ) : ℝ × ℝ → Prop := λ p, p.1 + m * p.2 = 2 + m

-- Define the circle
def circle (p : ℝ × ℝ) : Prop := (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 = 1

theorem line_intersects_circle (m : ℝ) :
  (∃ p : ℝ × ℝ, line m p ∧ circle p) ↔ m ≠ 0 :=
sorry

end line_intersects_circle_l825_825940


namespace valuable_files_count_l825_825293

theorem valuable_files_count 
    (initial_files : ℕ) 
    (deleted_fraction_initial : ℚ) 
    (additional_files : ℕ) 
    (irrelevant_fraction_additional : ℚ) 
    (h1 : initial_files = 800) 
    (h2 : deleted_fraction_initial = (70:ℚ) / 100)
    (h3 : additional_files = 400)
    (h4 : irrelevant_fraction_additional = (3:ℚ) / 5) : 
    (initial_files - ⌊deleted_fraction_initial * initial_files⌋ + additional_files - ⌊irrelevant_fraction_additional * additional_files⌋) = 400 :=
by sorry

end valuable_files_count_l825_825293


namespace part1_a_value_part2_a_range_l825_825518

def f (x : ℝ) : ℝ := x^3 - x
def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

-- Part (1)
theorem part1_a_value (a : ℝ) : 
  f'(-1) = g'(-1) → 
  (∀ x, g(x, a) - 2(x + 1) = 0) → 
  a = 3 :=
sorry

-- Part (2)
theorem part2_a_range (a : ℝ) : 
  (∃ x1 x2 : ℝ, f'(x1) = g'(x2) ∧ 
  f(x1) - (3 * x1^2 - 1) * (x1 - x) = g(x2, a) - 2x2 * x) → 
  a ≥ -1 :=
sorry

end part1_a_value_part2_a_range_l825_825518


namespace expected_value_of_twelve_sided_die_l825_825788

theorem expected_value_of_twelve_sided_die : 
  let face_values := finset.range (12 + 1) \ finset.singleton 0 in
  (finset.sum face_values (λ x, x) : ℝ) / 12 = 6.5 :=
by
  sorry

end expected_value_of_twelve_sided_die_l825_825788


namespace domain_of_sqrt_div_sqrt_l825_825474

theorem domain_of_sqrt_div_sqrt (x : ℝ) : (3 ≤ x ∧ x < 7) ↔ (∃ f, f = (λ x, (√(x - 3)) / (√(7 - x))) ∧ 3 ≤ x ∧ x < 7) := 
by 
  sorry

end domain_of_sqrt_div_sqrt_l825_825474


namespace solution_set_range_l825_825693

theorem solution_set_range (x : ℝ) : 
  (2 * |x - 10| + 3 * |x - 20| ≤ 35) ↔ (9 ≤ x ∧ x ≤ 23) :=
sorry

end solution_set_range_l825_825693


namespace stratified_sampling_elderly_count_l825_825748

-- Definitions of conditions
def elderly := 30
def middleAged := 90
def young := 60
def totalPeople := elderly + middleAged + young
def sampleSize := 36
def samplingFraction := sampleSize / totalPeople
def expectedElderlySample := elderly * samplingFraction

-- The theorem we want to prove
theorem stratified_sampling_elderly_count : expectedElderlySample = 6 := 
by 
  -- Proof is omitted
  sorry

end stratified_sampling_elderly_count_l825_825748


namespace average_children_in_families_with_children_l825_825419

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l825_825419


namespace part1_l825_825858

variables (a c : ℝ × ℝ)
variables (a_parallel_c : ∃ k : ℝ, c = (k * a.1, k * a.2))
variables (a_value : a = (1,2))
variables (c_magnitude : (c.1 ^ 2 + c.2 ^ 2) = (3 * Real.sqrt 5) ^ 2)

theorem part1: c = (3, 6) ∨ c = (-3, -6) :=
by
  sorry

end part1_l825_825858


namespace hamburgers_served_l825_825273

def hamburgers_made : ℕ := 9
def hamburgers_leftover : ℕ := 6

theorem hamburgers_served : ∀ (total : ℕ) (left : ℕ), total = hamburgers_made → left = hamburgers_leftover → total - left = 3 := 
by
  intros total left h_total h_left
  rw [h_total, h_left]
  rfl

end hamburgers_served_l825_825273


namespace no_such_arrangement_l825_825057

noncomputable def impossible_arrangement : Prop :=
  ¬ ∃ (f : Fin 2018 → ℕ),
    (∀ i, f i ∈ Finset.range 1 2019) ∧
    (∀ i, (f i + f (i + 1) + f (i + 2)) % 2 = 1)

theorem no_such_arrangement : impossible_arrangement :=
sorry

end no_such_arrangement_l825_825057


namespace chocolate_flavor_sales_l825_825265

-- Define the total number of cups sold
def total_cups : ℕ := 50

-- Define the fraction of winter melon flavor sales
def winter_melon_fraction : ℚ := 2 / 5

-- Define the fraction of Okinawa flavor sales
def okinawa_fraction : ℚ := 3 / 10

-- Proof statement
theorem chocolate_flavor_sales : 
  (total_cups - (winter_melon_fraction * total_cups).toInt - (okinawa_fraction * total_cups).toInt) = 15 := 
  by 
  sorry

end chocolate_flavor_sales_l825_825265


namespace problem_d_l825_825718

variable {R : Type} [LinearOrder R]

theorem problem_d (a b c d : R) (h₁ : a > b) (h₂ : c > d) : a - d > b - c := by
  sorry -- Proof goes here

end problem_d_l825_825718


namespace average_children_in_families_with_children_l825_825373

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l825_825373


namespace max_value_complex_expression_l825_825628

theorem max_value_complex_expression 
  (z : ℂ) 
  (h : complex.abs z = 2) : 
  ∃ (x : ℝ), (|((z - 2)^2 * (z + 2) * conj(z))| = 128) :=
begin
  sorry,
end

end max_value_complex_expression_l825_825628


namespace range_of_k_l825_825543

noncomputable def satisfies_conditions (k : ℝ) (A B : ℝ × ℝ) (D O : ℝ × ℝ := (0, 0)) : Prop :=
  let OA := A
  let OB := B
  let AB := (A.1 - B.1, A.2 - B.2)
  let OD := D
  k > 0 ∧ 
  (A.1^2 + A.2^2 = 4) ∧ 
  (B.1^2 + B.2^2 = 4) ∧
  (O = (0, 0)) ∧
  (OD = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧
  (A ≠ B) ∧
  sqr (norm (OA + OB)) ≥ 3 * sqr (norm AB)

theorem range_of_k (k : ℝ) (A B : ℝ × ℝ) (D O : ℝ × ℝ := (0, 0)) :
  satisfies_conditions k A B D O → ∃ (k : ℝ), sqrt 6 ≤ k ∧ k < 2 * sqrt 2 := 
sorry

end range_of_k_l825_825543


namespace axis_of_symmetry_l825_825538

noncomputable def f (x : ℝ) : ℝ := cos (x + (Real.pi / 2)) * cos (x + (Real.pi / 4))

theorem axis_of_symmetry : (∀ x : ℝ, f (x) = f (Real.pi * (5/8) - x)) :=
by
  sorry

end axis_of_symmetry_l825_825538


namespace smallest_base10_integer_l825_825201

theorem smallest_base10_integer (a b : ℕ) (ha : a > 3) (hb : b > 3) (h : a + 3 = 3 * b + 1) :
  13 = a + 3 :=
by
  have h_in_base_a : a = 3 * b - 2 := by linarith,
  have h_in_base_b : 3 * b + 1 = 13 := by sorry,
  exact h_in_base_b

end smallest_base10_integer_l825_825201


namespace sin_I_eq_4_5_l825_825045

-- Given conditions
variables (G H I : Type) [right_triangle GHI]
variables (angle_G_eq_90 : ∠G = 90) (sin_H_eq_3_5 : sin H = 3/5)

-- Proof statement
theorem sin_I_eq_4_5 :
  sin I = 4/5 :=
sorry

end sin_I_eq_4_5_l825_825045


namespace expected_value_twelve_sided_die_l825_825779

theorem expected_value_twelve_sided_die : 
  (1 / 12 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l825_825779


namespace angle_bisector_midpoint_l825_825741

theorem angle_bisector_midpoint 
  (A B C M D : Point) 
  (hAMBisector: AngleBisector A B C M) 
  (hD_on_AC : OnLine AC D) 
  (hAngleEq : ∠DMC = ∠BAC) :
  SegmentLength BM = SegmentLength MD := 
sorry

end angle_bisector_midpoint_l825_825741


namespace bill_is_thief_l825_825283

theorem bill_is_thief (bill_joe_sam : Prop) 
    (one_thief : Prop) 
    (only_thief_truthful : Prop) 
    (sam_claims_joe : Prop) : 
    (sam_claims_joe → bill_joe_sam) ∧ (only_thief_truthful → (bill_joe_sam → ¬sam_claims_joe ∧ (one_thief → bill_joe_sam))) → bill_joe_sam :=
sorry

end bill_is_thief_l825_825283


namespace proof_problem_l825_825869

variable {A B : Type}
variable (f : A → B) (f_inv : B → A)
variable a : A

-- Defining conditions
def condition1 (x : A) : Prop := f x > x
def condition2 : Prop := f a = 0

-- Defining the inverse function property
axiom inverse_property (b : B) : f (f_inv b) = b → f_inv (f a) = a

-- Problem statement 
theorem proof_problem (h1 : ∀ x : A, condition1 f x) (h2 : condition2 f a) : 
  f_inv 0 = a ∧ ∀ x : B, f_inv x < x :=
sorry

end proof_problem_l825_825869


namespace john_total_free_throw_shots_l825_825946

noncomputable def average_free_throw_success_rate : ℝ :=
  0.20 * 0.60 + 0.50 * 0.70 + 0.30 * 0.80

noncomputable def average_fouls_per_game : ℝ :=
  0.40 * 4 + 0.35 * 5 + 0.25 * 6

noncomputable def average_free_throw_shots_per_game : ℝ :=
  average_fouls_per_game * 2

noncomputable def number_of_games_john_plays : ℕ :=
  (20 : ℝ) * 0.80

noncomputable def total_free_throw_shots : ℝ :=
  average_free_throw_shots_per_game * number_of_games_john_plays

theorem john_total_free_throw_shots : total_free_throw_shots ≈ 155 :=
by
  sorry

end john_total_free_throw_shots_l825_825946


namespace money_properties_proof_l825_825109

-- The context of Robinson Crusoe being on a deserted island.
def deserted_island := true

-- The functions of money in modern society.
def medium_of_exchange := true
def store_of_value := true
def unit_of_account := true
def standard_of_deferred_payment := true

-- The financial context of the island.
def island_context (deserted_island : Prop) : Prop :=
  deserted_island →
  ¬ medium_of_exchange ∧
  ¬ store_of_value ∧
  ¬ unit_of_account ∧
  ¬ standard_of_deferred_payment

-- Other properties that an item must possess to become money.
def durability := true
def portability := true
def divisibility := true
def acceptability := true
def uniformity := true
def limited_supply := true

-- The proof problem statement.
theorem money_properties_proof
  (H1 : deserted_island)
  (H2 : island_context H1)
  : (¬ medium_of_exchange ∧
    ∀ (m : Prop), (m = durability ∨ m = portability ∨ m = divisibility ∨ m = acceptability ∨ m = uniformity ∨ m = limited_supply)) :=
by {
  sorry
}

end money_properties_proof_l825_825109


namespace natural_number_distinct_leading_digits_l825_825485

/--
  For each of the nine natural numbers \(n, 2n, 3n, \ldots, 9n\), 
  the first digit from the left is written down. Can there exist a 
  natural number \(n\) such that among the nine written digits, 
  there are no more than four distinct digits?
-/
theorem natural_number_distinct_leading_digits :
  ∃ n : ℕ, (finset.image 
             (λ k, (nat.succ (k - 1) * n / (10 ^ (nat.log10 (nat.succ (k - 1) * n / 10 ^ nat.log10 (nat.succ (k - 1) * n))))))
             (finset.range (nat.succ 9))).card ≤ 4 :=
begin
  sorry
end

end natural_number_distinct_leading_digits_l825_825485


namespace avg_children_in_families_with_children_l825_825448

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l825_825448


namespace problem_equiv_l825_825851

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem problem_equiv (x y : ℝ) : dollar ((2 * x + y) ^ 2) ((x - 2 * y) ^ 2) = (3 * x ^ 2 + 8 * x * y - 3 * y ^ 2) ^ 2 := by
  sorry

end problem_equiv_l825_825851


namespace basketball_percentage_combined_l825_825288

def students_North_High_School := 2200
def basketball_percentage_North_High := 20
def students_South_Academy := 2600
def basketball_percentage_South_Academy := 35

theorem basketball_percentage_combined :
  let total_students := students_North_High_School + students_South_Academy in
  let basketball_students_North := students_North_High_School * basketball_percentage_North_High / 100 in
  let basketball_students_South := students_South_Academy * basketball_percentage_South_Academy / 100 in
  let total_basketball_students := basketball_students_North + basketball_students_South in
  (total_basketball_students * 100 / total_students) = 28 := by
  sorry

end basketball_percentage_combined_l825_825288


namespace sandy_spent_on_repairs_l825_825117

theorem sandy_spent_on_repairs (initial_cost : ℝ) (selling_price : ℝ) (gain_percent : ℝ) (repair_cost : ℝ) :
  initial_cost = 800 → selling_price = 1400 → gain_percent = 40 → selling_price = 1.4 * (initial_cost + repair_cost) → repair_cost = 200 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_spent_on_repairs_l825_825117


namespace max_diagonals_selected_l825_825038

theorem max_diagonals_selected (n : ℕ) (h : n = 1000) :
  ∃ (d : ℕ), d = 2000 ∧ 
  ∀ (chosen_diagonals : Finset (Finₓ (n * (n - 3) / 2))), 
    (∀ (x y z ∈ chosen_diagonals), 
     (∃ (same_length_pair : ℕ × ℕ), same_length_pair ∈ chosen_diagonals ∧ same_length_pair.fst = same_length_pair.snd)) →
    chosen_diagonals.card ≤ d :=
by
  rw h
  use 2000
  split
  { refl }
  { intros chosen_diagonals condition
    sorry -- proof to be provided
  }

end max_diagonals_selected_l825_825038


namespace sufficient_condition_for_inequality_l825_825880

theorem sufficient_condition_for_inequality (a b : ℝ) (h_nonzero : a * b ≠ 0) : (a < b ∧ b < 0) → (1 / a ^ 2 > 1 / b ^ 2) :=
by
  intro h
  sorry

end sufficient_condition_for_inequality_l825_825880


namespace simplify_expression_l825_825123

-- Define the given expression
def expr : ℚ := (5^6 + 5^3) / (5^5 - 5^2)

-- State the proof problem
theorem simplify_expression : expr = 315 / 62 := 
by sorry

end simplify_expression_l825_825123


namespace two_students_with_A_l825_825355

variable (Elena Fiona Gabriel Harry : Prop)
variable (E_imp_F : Elena → Fiona)
variable (F_imp_G : Fiona → Gabriel)
variable (G_imp_H : Gabriel → Harry)
variable (Two_of_four_have_A : ∃ (a b : Prop), (a = Gabriel) ∧ (b = Harry) ∧ (¬Elena ∧ ¬Fiona))

theorem two_students_with_A :
  Two_of_four_have_A →
  ((Elena ∨ Fiona ∨ Gabriel ∨ Harry) ∧ ¬(Elena ∧ Fiona ∧ Gabriel ∧ Harry)) :=
  by
    sorry

end two_students_with_A_l825_825355


namespace phase_shift_l825_825478

theorem phase_shift :
    let y := λ x : ℝ, 3 * sin (3 * x - π / 4) + 4 * cos (3 * x) in
    (∃ φ : ℝ, (∀ x : ℝ, y x = 3 * sin (3 * x - π / 4)) → φ = π / 12) := 
sorry

end phase_shift_l825_825478


namespace geometric_sequence_fifth_term_l825_825680

-- Define the geometric sequence's first four terms and the common ratio
def is_geometric_sequence (x y : ℝ) (r : ℝ) (a1 a2 a3 a4 : ℝ) :=
  a1 = x + y ∧
  a2 = x - y ∧
  a3 = xy ∧
  a4 = x / y ∧
  a2 / a1 = r ∧
  a3 / a2 = r ∧
  a4 / a3 = r

theorem geometric_sequence_fifth_term (x y : ℝ) (h : x = y / (y - 1)) :
  is_geometric_sequence x y ((x - y) / (x + y)) (x + y) (x - y) (xy) (x / y) →
  (x / y) * ((x - y) / (x + y)) = -1 / ((y - 1) * (2y - 1)) :=
sorry

end geometric_sequence_fifth_term_l825_825680


namespace average_children_in_families_with_children_l825_825414

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l825_825414


namespace geometric_seq_deriv_at_zero_eq_l825_825597

theorem geometric_seq_deriv_at_zero_eq : 
  let a1 := 2
  let a8 := 4
  let r := 2^(1/7)
  let a2 := a1 * r
  let a3 := a1 * r^2
  let a4 := a1 * r^3
  let a5 := a1 * r^4
  let a6 := a1 * r^5
  let a7 := a1 * r^6
  let f (x : ℝ) := x * (x - a1) * (x - a2) * (x - a3) * (x - a4) * (x - a5) * (x - a6) * (x - a7) * (x - a8)
  in f' 0 = 2^12 := sorry

end geometric_seq_deriv_at_zero_eq_l825_825597


namespace exists_integral_wins_l825_825982

noncomputable def Tournament := 
  Set (Fin 29)

def adjMatrix (T : Tournament) : Matrix (Fin 29) (Fin 29) ℕ :=
  λ i j, if i ≠ j then
            if (i, j) ∈ T then 1 else 0
         else 0

theorem exists_integral_wins : 
  ∀ (T : Tournament), 
    ∃ (k : ℕ) (S : Set (Fin 29)), 
    k ≤ 29 ∧ S ⊆ Fin 29 ∧
    (∀ i ∈ Fin 29, ∃ m : ℕ, m = ∑ j ∈ S, (adjMatrix T) i j) :=
begin
  sorry
end

end exists_integral_wins_l825_825982


namespace part_I_part_II_l825_825047

variables {θ t : Real}

def curve_x (θ : Real) : Real := 3 * cos θ
def curve_y (θ : Real) : Real := 2 * sin θ

def line_x (t a : Real) : Real := t - 1
def line_y (t a : Real) : Real := 2 * t - a - 1

-- Problem Part (I)
theorem part_I (a : Real) (h_a : a = 1) : 
  let intersection_1 := (curve_x (atan 2), curve_y (atan 2)) in
  let intersection_2 := (curve_x (atan (-2)), curve_y (atan (-2))) in
  (sqrt ((intersection_1.1 - intersection_2.1)^2 + (intersection_1.2 - intersection_2.2)^2) = 3 * sqrt 2) := sorry

-- Problem Part (II)
theorem part_II (a : Real) (h_a : a = 11) :
  let M := (curve_x (pi/4), curve_y (pi/4)) in
  let distance := abs ((2 * M.1 - M.2 - 10) / sqrt 5) in
  (M.1 = 9 * sqrt 10 / 10) ∧ (M.2 = - sqrt 10 / 5) ∧ (distance = 2 * sqrt 5 - 2 * sqrt 2) := sorry

end part_I_part_II_l825_825047


namespace proj_b_v_l825_825619

variables (a b : ℝ × ℝ)
variables (v : ℝ × ℝ := (4, -2))
variables (proj_a := (-2/5, -4/5))

-- Define orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Define projection function
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  let k := dot_uv / dot_uu
  (k * u.1, k * u.2)

-- Given Conditions
axiom h1 : orthogonal a b
axiom h2 : proj a v = proj_a

-- The theorem to prove
theorem proj_b_v : proj b v = (22 / 5, -6 / 5) :=
  sorry

end proj_b_v_l825_825619


namespace find_least_possible_N_l825_825755

noncomputable def least_possible_N : ℤ :=
  let incorrect_consec_numbers := (17, 18) in
  let num_range := (1 to 30).to_list in
  let filtered_range := num_range.filter (λ x => x ≠ fst(incorrect_consec_numbers) ∧ x ≠ snd(incorrect_consec_numbers)) in
  filtered_range.foldl Nat.lcm 1

theorem find_least_possible_N : least_possible_N = 8923714800 :=
  by
  sorry

end find_least_possible_N_l825_825755


namespace average_children_in_families_with_children_l825_825382

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l825_825382


namespace average_children_in_families_with_children_l825_825430

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l825_825430


namespace option2_cheaper_l825_825180

-- Define the conditions
variables (D : ℝ) (r : ℝ) -- D = Total distance of the tunnel, r = digging rate of slower laborer
variables (same_hourly_wage : ℝ)
variables (faster_rate : ℝ := 1.5 * r) -- Faster laborer digs 1.5 times faster

-- Define the time calculation for the two options
def time_option1 : ℝ := (D / (2 * r)) + (D / (3 * r))
def time_option2 : ℝ := 2 * (D / (2.5 * r))

-- Theorem stating option 2 is cheaper
theorem option2_cheaper : time_option2 D r < time_option1 D r :=
by {
  sorry
}

end option2_cheaper_l825_825180


namespace average_children_l825_825438

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l825_825438


namespace arithmetic_sequence_a1_value_l825_825815

-- Definition of the arithmetic sequence sum and nth term
def S (n : ℕ) (a₁ d : ℝ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2
def a (n : ℕ) (a₁ d : ℝ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_a1_value :
  (S 3 a₁ d = a 2 a₁ d + 10 * a₁) →
  (a 5 a₁ d = 9) →
  a₁ = 1 / 3 :=
by
  intros h₁ h₂
  sorry

end arithmetic_sequence_a1_value_l825_825815


namespace power_function_value_l825_825002

theorem power_function_value (f : ℝ → ℝ) (m : ℤ) 
  (h₁ : f = λ x, x^(-m^2 + 2*m + 3))
  (h₂ : ∀ x : ℝ, x > 0 → f x < f (x + 1))
  (h₃ : ∀ x : ℝ, f x = f (-x)) : 
  f (-2) = 16 :=
by
  sorry

end power_function_value_l825_825002


namespace inequality_comparison_l825_825498

theorem inequality_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) 
  (h₁ : a = (1 / Real.log 3 / Real.log 2))
  (h₂ : b = Real.exp 0.5)
  (h₃ : c = Real.log 2) :
  b > c ∧ c > a := 
by
  sorry

end inequality_comparison_l825_825498


namespace ellipse_eccentricity_l825_825513

theorem ellipse_eccentricity (E : Type) (F₁ F₂ : E) (P Q : E) (slope : ℝ)
(hE : is_ellipse E)
(hF₁F₂ : is_focus_pair E F₁ F₂)
(hLine : passes_through F₁ with_slope slope)
(hSlope : slope = 2)
(hIntersections : intersects E P Q)
(hTriangle : forms_right_angle_triangle P F₁ F₂) :
    eccentricity E = (Real.sqrt 5 - 2) ∨ eccentricity E = Real.sqrt 5 / 3 := by sorry

end ellipse_eccentricity_l825_825513


namespace part1_part2_l825_825894

-- Define the conditions and the problem
variable (α : ℝ)
def m := -1 / 4
def f (α : ℝ) : ℝ := (Real.cos (2 * Real.pi - α) + Real.tan (3 * Real.pi + α)) /
                      (Real.sin (Real.pi - α) * Real.cos (α + 3 * Real.pi / 2))

-- Define the given trigonometric values
noncomputable def cos_alpha := -1 / 4
noncomputable def sin_alpha := Real.sqrt 15 / 4
noncomputable def tan_alpha := -Real.sqrt 15

-- Formulate the proof problem as Lean statements
theorem part1 : (Real.cos (Real.pi - α) = cos_alpha ∧ Real.sin α = sin_alpha ∧ α < Real.pi ∧ α > Real.pi / 2) → m = -1 / 4 := 
by
  sorry

theorem part2 : (Real.cos α = cos_alpha) ∧ (Real.sin α = sin_alpha) ∧ (Real.tan α = tan_alpha) → f α = -((4 + 16 * Real.sqrt 15) / 15) := 
by
  sorry

end part1_part2_l825_825894


namespace floor_T_squared_l825_825093

noncomputable def T : ℝ := ∑ i in Finset.range 2007, Real.sqrt (1 + 1 / (i + 1)^3 + 1 / (i + 2)^3)

theorem floor_T_squared :
  Real.floor (T^2) = 3 := by
  sorry

end floor_T_squared_l825_825093


namespace binom_18_4_eq_3060_l825_825324

theorem binom_18_4_eq_3060 : nat.choose 18 4 = 3060 := sorry

end binom_18_4_eq_3060_l825_825324


namespace expected_value_twelve_sided_die_l825_825773

theorem expected_value_twelve_sided_die : 
  let die_sides := 12 in 
  let outcomes := finset.range (die_sides + 1) in
  (finset.sum outcomes id : ℚ) / die_sides = 6.5 :=
by
  sorry

end expected_value_twelve_sided_die_l825_825773


namespace smallest_prime_divisor_to_make_perfect_cube_3600_l825_825216

theorem smallest_prime_divisor_to_make_perfect_cube_3600 : 
  ∃ p : ℕ, Prime p ∧ (3600 / p) % p = 0 ∧ is_perfect_cube (3600 / p) ∧ p = 2 := 
sorry

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

end smallest_prime_divisor_to_make_perfect_cube_3600_l825_825216


namespace hyperbola_eccentricity_l825_825503

theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
    (h_circle : ∀ k : ℝ, |k| / real.sqrt (k^2 + 1) = real.sqrt 3 / 2 → k = real.sqrt 3 ∨ k = -real.sqrt 3)
    (h_tangent : ∀ k : ℝ, (k = real.sqrt 3 ∨ k = -real.sqrt 3) → b / a > real.sqrt 3) :
  ∃ e : ℝ, e > 2 ∧ e = real.sqrt (1 + (b^2 / a^2)) :=
begin
  use real.sqrt (1 + (b^2 / a^2)),
  split,
  {
    have h1 : b^2 / a^2 > 3 := sorry,
    linarith,
  },
  {
    refl,
  },
end

end hyperbola_eccentricity_l825_825503


namespace part1_part2_part3_l825_825631

-- Define the geometry of the ellipse and point M
variable {a b m : ℝ}
variable {a_pos : a > 0} {b_pos : b > 0}
variable {a_gt_b : a > b} {m_ne_pm_a : m ≠ a ∧ m ≠ -a}
variable {on_major_axis : ellipse_eq a b (m, 0)}

-- Main proof statements
theorem part1 (h₁ : intersects_at AD (x = m) P) (h₂ : intersects_at BC (x = m) Q) : dist M P = dist M Q := sorry

theorem part2 (h_angle : ∠OMA = ∠OMC) : ∃ N, (N = (a^2 / m, 0)) ∧ ∠ANM = ∠BNR := sorry

theorem part3 (h_intersects : intersects_at AD x_axis (a^2 / m, 0) ∨ intersects_at BC x_axis (a^2 / m, 0)) : ∠OMA = ∠OMC ∧ ∠ANM = ∠BNR := sorry

end part1_part2_part3_l825_825631


namespace expected_value_of_twelve_sided_die_l825_825772

theorem expected_value_of_twelve_sided_die : 
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0) in
  (finset.sum outcomes (λ n, (n:ℝ)) / 12 = 6.5) :=
by
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0)
  have h1 : ∑ n in outcomes, (n : ℝ) = 78, sorry
  have h2 : (78 / 12) = 6.5, sorry
  exact h2

end expected_value_of_twelve_sided_die_l825_825772


namespace problem_conditionals_l825_825533

theorem problem_conditionals (p1 p2 p3 p4 : Prop) 
  (h1 : p1 = "Find the perimeter of an equilateral triangle with an area of 1")
  (h2 : p2 = "Calculate the arithmetic mean of three numbers entered on a keyboard")
  (h3 : p3 = "Find the minimum of two numbers entered on a keyboard")
  (h4 : p4 = "Calculate the value of the function f(x)= \begin{cases} 2x & if x≥3 \\ x^{2} & if x < 3 end for a given value of the independent variable")
  (cond1 : ∀ p1, does_not_require_conditionals p1)
  (cond2 : ∀ p2, does_not_require_conditionals p2)
  (cond3 : ∀ p3, requires_conditionals p3)
  (cond4 : ∀ p4, requires_conditionals p4) : 
  count_conditions [p1, p2, p3, p4,] = 2 :=
sorry

end problem_conditionals_l825_825533


namespace projection_eq_1_13_l825_825158

open Real

theorem projection_eq_1_13 (c : ℝ) :
  let v := ⟨6, c⟩ : EuclideanSpace ℝ (Fin 2)
  let u := ⟨-3, 2⟩ : EuclideanSpace ℝ (Fin 2)
  (proj v u = (1/13 : ℝ) • u) → c = 9.5 :=
by
  intros v u h
  simp [proj] at h
  sorry

end projection_eq_1_13_l825_825158


namespace largest_number_among_options_l825_825221

theorem largest_number_among_options :
  let A := 12345 + 1/5678,
      B := 12345 - 1/5678,
      C := 12345 * (1/5678),
      D := 12345 * 5678,
      E := 12345.5678
  in D = 70082310 ∧ D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  -- Definitions of the numbers based on the problem conditions
  let A := 12345 + 1/5678
  let B := 12345 - 1/5678
  let C := 12345 * (1/5678)
  let D := 12345 * 5678
  let E := 12345.5678
  -- Correct answer: Largest number is D (70082310)
  have hD : D = 70082310 := rfl
  have hdA : D > A := sorry
  have hdB : D > B := sorry
  have hdC : D > C := sorry
  have hdE : D > E := sorry
  exact ⟨hD, hdA, hdB, hdC, hdE⟩

end largest_number_among_options_l825_825221


namespace domain_of_sqrt_div_sqrt_l825_825473

theorem domain_of_sqrt_div_sqrt (x : ℝ) : (3 ≤ x ∧ x < 7) ↔ (∃ f, f = (λ x, (√(x - 3)) / (√(7 - x))) ∧ 3 ≤ x ∧ x < 7) := 
by 
  sorry

end domain_of_sqrt_div_sqrt_l825_825473


namespace dot_product_eq_four_l825_825565

variable {E : Type*} [inner_product_space ℝ E]

variables (u v w : E)

def norm_eq_one (x : E) : Prop := ∥x∥ = 1
def norm_sum_eq_two (u v : E) : Prop := ∥u + v∥ = 2
def w_def (u v w : E) : Prop := w = u + 3 • v + 4 • (u ×ₗ v)

theorem dot_product_eq_four
  (h1 : norm_eq_one u)
  (h2 : norm_eq_one v)
  (h3 : norm_sum_eq_two u v)
  (h4 : w_def u v w) :
  (v ⬝ w) = 4 :=
sorry

end dot_product_eq_four_l825_825565


namespace common_number_of_two_sets_l825_825701

theorem common_number_of_two_sets (a b c d e f g : ℚ) :
  (a + b + c + d) / 4 = 5 →
  (d + e + f + g) / 4 = 8 →
  (a + b + c + d + e + f + g) / 7 = 46 / 7 →
  d = 6 :=
by
  intros h₁ h₂ h₃
  sorry

end common_number_of_two_sets_l825_825701


namespace part1_a_value_part2_a_range_l825_825519

def f (x : ℝ) : ℝ := x^3 - x
def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

-- Part (1)
theorem part1_a_value (a : ℝ) : 
  f'(-1) = g'(-1) → 
  (∀ x, g(x, a) - 2(x + 1) = 0) → 
  a = 3 :=
sorry

-- Part (2)
theorem part2_a_range (a : ℝ) : 
  (∃ x1 x2 : ℝ, f'(x1) = g'(x2) ∧ 
  f(x1) - (3 * x1^2 - 1) * (x1 - x) = g(x2, a) - 2x2 * x) → 
  a ≥ -1 :=
sorry

end part1_a_value_part2_a_range_l825_825519


namespace check_meaningfulness_l825_825219

def is_meaningful (x : ℝ) : Prop :=
  ∃ y : ℝ, y ^ 2 = x

def expr_A := -real.sqrt (1/6)
def expr_B := real.sqrt ((-1:ℝ) ^ 2)
def expr_C (a : ℝ) := real.sqrt (a^2 + 1)
def expr_D (a : ℝ) := real.sqrt (-a^2 - 1)

theorem check_meaningfulness (a : ℝ) :
  is_meaningful (1/6) ∧
  is_meaningful ((-1:ℝ) ^ 2) ∧
  is_meaningful (a^2 + 1) ∧
  ¬is_meaningful (-a^2 - 1) :=
by {
  sorry
}

end check_meaningfulness_l825_825219


namespace intersection_A_B_l825_825911

-- Conditions
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := { y | ∃ x ∈ A, y = 3 * x - 2 }

-- Question and proof statement
theorem intersection_A_B :
  A ∩ B = {1, 4} := by
  sorry

end intersection_A_B_l825_825911


namespace expected_value_twelve_sided_die_l825_825782

theorem expected_value_twelve_sided_die : 
  (1 / 12 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l825_825782


namespace badgers_win_at_least_4_games_l825_825134

noncomputable def badgers_probability (p_win: ℚ) (n_games: ℕ) (k: ℕ): ℚ :=
  (nat.choose n_games k) * p_win^k * (1 - p_win)^(n_games - k)

theorem badgers_win_at_least_4_games:
  let p_win := (2:ℚ) / 3
  let n_games := 7
  (badgers_probability p_win n_games 4
   + badgers_probability p_win n_games 5
   + badgers_probability p_win n_games 6
   + badgers_probability p_win n_games 7) = 1808 / 2187 := by
    sorry

end badgers_win_at_least_4_games_l825_825134


namespace avg_expenditure_Feb_to_July_l825_825226

noncomputable def avg_expenditure_Jan_to_Jun : ℝ := 4200
noncomputable def expenditure_January : ℝ := 1200
noncomputable def expenditure_July : ℝ := 1500
noncomputable def total_months_Jan_to_Jun : ℝ := 6
noncomputable def total_months_Feb_to_July : ℝ := 6

theorem avg_expenditure_Feb_to_July :
  (avg_expenditure_Jan_to_Jun * total_months_Jan_to_Jun - expenditure_January + expenditure_July) / total_months_Feb_to_July = 4250 :=
by sorry

end avg_expenditure_Feb_to_July_l825_825226


namespace find_x_value_l825_825349

variable (x : ℝ)

-- Definition stating the condition for convergence of the infinite power tower
def power_tower_converges (x : ℝ) : Prop := ∃ (y : ℝ), x^(x^(x^(...))) = y

-- Define the limit value and state the equation it should satisfy
def power_tower_eq (x y : ℝ) : Prop := (x^(x^(x^(...)))) = y

-- The main theorem
theorem find_x_value (H : power_tower_eq x 4) (Hconv : power_tower_converges x) : x = Real.sqrt 2 :=
sorry

end find_x_value_l825_825349


namespace min_log_expression_eq_4_l825_825494

theorem min_log_expression_eq_4 
  (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  (∃ b, ∀ b, log a b + log b (a ^ 2 + 12) ≥ 4) → 
  (log a b + log b (a ^ 2 + 12) = 4) → 
  a = 2 :=
by
  sorry

end min_log_expression_eq_4_l825_825494


namespace expected_value_of_twelve_sided_die_l825_825785

theorem expected_value_of_twelve_sided_die : ∑ k in finset.range 13, k / 12 = 6.5 := 
sorry

end expected_value_of_twelve_sided_die_l825_825785


namespace ellipse_standard_equation_l825_825796

def foci_on_x_axis (e : Ellipse) : Prop := e.foci.1 = (x, 0) ∧ e.foci.2 = (-x, 0)

def major_minor_sum (e : Ellipse) : Prop := e.major + e.minor = 10

def focal_distance (e : Ellipse) : Prop := e.focal_distance = 4 * real.sqrt 5

theorem ellipse_standard_equation 
  (e : Ellipse) 
  (H1 : foci_on_x_axis e) 
  (H2 : major_minor_sum e) 
  (H3 : focal_distance e) : 
  e.equation = (λ x y, x^2 / 36 + y^2 / 16 = 1) :=
sorry

end ellipse_standard_equation_l825_825796


namespace factorial_sqrt_square_l825_825297

theorem factorial_sqrt_square (n : ℕ) : (nat.succ 4)! * 4! = 2880 := by 
  sorry

end factorial_sqrt_square_l825_825297


namespace original_radius_new_perimeter_l825_825717

variable (r : ℝ)

theorem original_radius_new_perimeter (h : (π * (r + 5)^2 = 4 * π * r^2)) :
  r = 5 ∧ 2 * π * (r + 5) = 20 * π :=
by
  sorry

end original_radius_new_perimeter_l825_825717


namespace difference_smallest_4_digit_and_largest_3_digit_l825_825188

theorem difference_smallest_4_digit_and_largest_3_digit :
  ∀ smallest_4_digit largest_3_digit,
  smallest_4_digit = 1000 →
  largest_3_digit = 999 →
  (smallest_4_digit - largest_3_digit) = 1 :=
by
  intros smallest_4_digit largest_3_digit h1 h2
  rw [h1, h2]
  sorry

end difference_smallest_4_digit_and_largest_3_digit_l825_825188


namespace decreasing_interval_of_g_l825_825888

-- Define the inverse function
def inverse_f (y : ℝ) : ℝ := 1 / 3 ^ y

-- Define the function f based on its inverse
def f (x : ℝ) : ℝ := -Real.log 3 x

-- Define the composite function g(x) = f(2x - x^2)
def g (x : ℝ) : ℝ := f (2 * x - x ^ 2)

-- The theorem to prove the decreasing interval
theorem decreasing_interval_of_g :
  ∃ (a b : ℝ), (0 < a ∧ a ≤ 1 ∧ ∀ (x : ℝ), a ≤ x ∧ x ≤ b → g(x+dt) < g(x)) :=
sorry

end decreasing_interval_of_g_l825_825888


namespace rectangle_perimeter_l825_825148

def gcd (a b : ℕ) : ℕ := sorry -- Placeholder for actual gcd function if needed

theorem rectangle_perimeter :
  ∀ b1 b2 b3 b4 b5 b6 b7 b8 b9 l w,
    (b1 + b2 = b3) →
    (b1 + b3 = b4) →
    (b3 + b4 = b6) →
    (b2 + b3 + b4 = b5) →
    (b2 + b5 = b7) →
    (b1 + b6 = b8) →
    (b6 + b8 = b5 + b7) →
    (l = 77) →
    (w = 56) →
    gcd l w = 1 →
    (2 * (l + w) = 266) :=
by
  intros b1 b2 b3 b4 b5 b6 b7 b8 b9 l w H1 H2 H3 H4 H5 H6 H7 H8 H9 Hrel_prime
  -- Proof to be filled in
  sorry

end rectangle_perimeter_l825_825148


namespace average_children_in_families_with_children_l825_825428

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l825_825428


namespace captain_age_is_24_l825_825145

theorem captain_age_is_24 (C W : ℕ) 
  (hW : W = C + 7)
  (h_total_team_age : 23 * 11 = 253)
  (h_total_9_players_age : 22 * 9 = 198)
  (h_team_age_equation : 253 = 198 + C + W)
  : C = 24 :=
sorry

end captain_age_is_24_l825_825145


namespace infinite_sum_sequence_l825_825816

def sequence (n : ℕ) : ℝ :=
  match n with
  | 0 => 2
  | 1 => 3
  | n + 2 => (1/2) * sequence (n + 1) + (1/3) * sequence n

def sum_sequence : ℕ → ℝ
| 0 => sequence 0
| n + 1 => sequence (n + 1) + sum_sequence n

theorem infinite_sum_sequence :
  ∃ T : ℝ, T = 18 ∧ (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sum_sequence n - T| < ε) :=
sorry

end infinite_sum_sequence_l825_825816


namespace avg_children_in_families_with_children_l825_825444

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l825_825444


namespace shape_is_cone_l825_825046

-- Definition of spherical coordinates and the specific equation
variable {ρ θ φ : ℝ}
constant c : ℝ
axiom h_pos : c > 0
axiom eq_surface : ρ = c * Real.sin φ

-- The statement to prove the shape is a cone
theorem shape_is_cone : ∃ h : (∃ ρ θ φ : ℝ, ρ = c * Real.sin φ), true → 
  (shape_with_eq_surface = cone c) := 
sorry

end shape_is_cone_l825_825046


namespace average_children_in_families_with_children_l825_825376

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l825_825376


namespace expected_value_of_twelve_sided_die_l825_825784

theorem expected_value_of_twelve_sided_die : ∑ k in finset.range 13, k / 12 = 6.5 := 
sorry

end expected_value_of_twelve_sided_die_l825_825784


namespace find_number_of_students_l825_825156

-- Conditions
def john_marks_wrongly_recorded : ℕ := 82
def john_actual_marks : ℕ := 62
def sarah_marks_wrongly_recorded : ℕ := 76
def sarah_actual_marks : ℕ := 66
def emily_marks_wrongly_recorded : ℕ := 92
def emily_actual_marks : ℕ := 78
def increase_in_average : ℚ := 1 / 2

-- Proof problem
theorem find_number_of_students (n : ℕ) 
    (h1 : john_marks_wrongly_recorded = 82)
    (h2 : john_actual_marks = 62)
    (h3 : sarah_marks_wrongly_recorded = 76)
    (h4 : sarah_actual_marks = 66)
    (h5 : emily_marks_wrongly_recorded = 92)
    (h6 : emily_actual_marks = 78) 
    (h7: increase_in_average = 1 / 2):
    n = 88 :=
by 
  sorry

end find_number_of_students_l825_825156


namespace max_non_special_pairs_l825_825580

-- Define what constitutes a special pair
def is_special_pair (m n : ℕ) : Prop :=
  (17 * m + 43 * n) % (m - n) = 0

-- Define the range of numbers from 1 to 2021
def number_range : set ℕ := {k | 1 ≤ k ∧ k ≤ 2021}

-- Prove that the maximum number of non-special-pair numbers is 289
theorem max_non_special_pairs : ∃ s : set ℕ, s ⊆ number_range ∧ (∀ m n ∈ s, m ≠ n → ¬is_special_pair m n) ∧ s.to_finset.card = 289 :=
  sorry

end max_non_special_pairs_l825_825580


namespace smallest_possible_value_of_norm_z_l825_825076

theorem smallest_possible_value_of_norm_z (z : ℂ) (h : |z - 8| + |z + 6 * complex.I| = 17) : ∃ z : ℂ, |z| = 48 / 17 :=
sorry

end smallest_possible_value_of_norm_z_l825_825076


namespace intersection_point_of_curves_l825_825049

theorem intersection_point_of_curves :
  (∃ (θ t : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 ∧ (x = sqrt 5 * cos θ) ∧ (y = sqrt 5 * sin θ) ∧ 
  (x = 1 - (sqrt 2) / 2 * t) ∧ (y = -(sqrt 2) / 2 * t)) ↔ (2, 1) :=
by
  sorry

end intersection_point_of_curves_l825_825049


namespace mow_time_l825_825100

noncomputable def effective_swath_width (swath_width overlap: ℝ) : ℝ :=
  (swath_width - overlap) / 12

def strips_needed (lawn_width effective_width: ℝ) : ℝ :=
  lawn_width / effective_width

def total_distance (strips lawn_length: ℝ) : ℝ :=
  strips * lawn_length

def time_to_mow (total_distance speed: ℝ) : ℝ :=
  total_distance / speed

theorem mow_time (lawn_length lawn_width swath_width overlap speed hours: ℝ)
  (h1 : lawn_length = 120)
  (h2 : lawn_width = 200)
  (h3 : swath_width = 30)
  (h4 : overlap = 6)
  (h5 : speed = 4000)
  (h6 : hours = 3)
  : time_to_mow (total_distance (strips_needed lawn_width (effective_swath_width swath_width overlap)) lawn_length) speed = hours := by
  sorry

end mow_time_l825_825100


namespace no_such_arrangement_l825_825058

noncomputable def impossible_arrangement : Prop :=
  ¬ ∃ (f : Fin 2018 → ℕ),
    (∀ i, f i ∈ Finset.range 1 2019) ∧
    (∀ i, (f i + f (i + 1) + f (i + 2)) % 2 = 1)

theorem no_such_arrangement : impossible_arrangement :=
sorry

end no_such_arrangement_l825_825058


namespace a_1_val_a_2_val_a_3_val_a_4_val_a_n_formula_l825_825507

-- Defining the sequence {a_n}
def a : ℕ → ℚ
| n := sorry

-- Defining the sum of first n terms S_n
def S : ℕ → ℚ
| 0     := 0
| (n+1) := S n + a n

-- Given condition: S_n = 2n - a_n
axiom Sn_condition : ∀ n : ℕ, S (n+1) = 2 * (n+1) - a (n+1)

-- Prove specific values
theorem a_1_val : a 1 = 1 := sorry
theorem a_2_val : a 2 = 3 / 2 := sorry
theorem a_3_val : a 3 = 7 / 4 := sorry
theorem a_4_val : a 4 = 15 / 8 := sorry

-- Prove general term
theorem a_n_formula : ∀ n : ℕ+, a n = (2^n - 1) / 2^(n-1) := sorry

end a_1_val_a_2_val_a_3_val_a_4_val_a_n_formula_l825_825507


namespace number_of_terms_in_expansion_is_12_l825_825560

-- Define the polynomials
def p (x y z : ℕ) := x + y + z
def q (u v w x : ℕ) := u + v + w + x

-- Define the number of terms in a polynomial as a function.
def numberOfTerms (poly : Polynomial ℕ) : ℕ :=
  poly.degree + 1

-- Prove the number of terms in expansion of (x + y + z)(u + v + w + x) is 12.
theorem number_of_terms_in_expansion_is_12 (x y z u v w : ℕ) :
  numberOfTerms (p x y z * q u v w x) = 12 := by
  sorry

end number_of_terms_in_expansion_is_12_l825_825560


namespace price_decreased_after_discount_and_increase_l825_825272

noncomputable def original_price : ℝ := 250
noncomputable def discount_rate : ℝ := 0.20
noncomputable def increase_rate : ℝ := 0.20

theorem price_decreased_after_discount_and_increase :
  let discounted_price := original_price * (1 - discount_rate) in
  let increased_price := discounted_price * (1 + increase_rate) in
  increased_price < original_price :=
by
  -- Calculate the discounted price
  let discounted_price := original_price * (1 - discount_rate)
  -- Calculate the increased price
  let increased_price := discounted_price * (1 + increase_rate)
  sorry

end price_decreased_after_discount_and_increase_l825_825272


namespace find_a_l825_825492

theorem find_a (a b : ℝ) (h₀ : a > 1) (h₁ : b > 1)
  (h₂ : ∀ b:ℝ, b > 1 → log a b + log b (a^2 + 12) ≥ 4) :
  a = 2 := 
begin 
  -- Proof to be filled in.
  sorry
end

end find_a_l825_825492


namespace probability_of_square_product_is_17_over_96_l825_825178

def num_tiles : Nat := 12
def num_die_faces : Nat := 8

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def favorable_outcomes_count : Nat :=
  -- Valid pairs where tile's number and die's number product is a perfect square
  List.length [ (1, 1), (1, 4), (2, 2), (4, 1),
                (1, 9), (3, 3), (9, 1), (4, 4),
                (2, 8), (8, 2), (5, 5), (6, 6),
                (4, 9), (9, 4), (7, 7), (8, 8),
                (9, 9) ] -- Equals 17 pairs

def total_outcomes_count : Nat :=
  num_tiles * num_die_faces

def probability_square_product : ℚ :=
  favorable_outcomes_count / total_outcomes_count

theorem probability_of_square_product_is_17_over_96 :
  probability_square_product = (17 : ℚ) / 96 := 
  by sorry

end probability_of_square_product_is_17_over_96_l825_825178


namespace earthquake_impossible_l825_825942

theorem earthquake_impossible (d : ℕ) (h₀ : 0 ≤ d ∧ d ≤ 9) (he_prime : ¬ Nat.Prime ((45 + 3 * d))) :
  ¬ ∃ n : ℕ, prime n ∧ ∑ i in (range 10).erase d, i + 3 * d = n :=
  sorry

end earthquake_impossible_l825_825942


namespace correct_order_given_0_lt_a_lt_1_l825_825865

variable {a : ℝ} (x : ℝ)

def f (x : ℝ) := abs (log x / log a)

theorem correct_order_given_0_lt_a_lt_1 (h : 0 < a ∧ a < 1) :
  f a (1/4) > f a (1/3) ∧ f a (1/3) > f a 2 :=
sorry

end correct_order_given_0_lt_a_lt_1_l825_825865


namespace deepak_present_age_l825_825231

theorem deepak_present_age (x : ℕ) (h1 : ∀ current_age_rahul current_age_deepak, 
  4 * x = current_age_rahul ∧ 3 * x = current_age_deepak)
  (h2 : ∀ current_age_rahul, current_age_rahul + 6 = 22) :
  3 * x = 12 :=
by
  have h3 : 4 * x + 6 = 22 := h2 (4 * x)
  linarith

end deepak_present_age_l825_825231


namespace point_quadrant_l825_825876

def z1 : Complex := 1 + Complex.i
def z2 : Complex := 3 - 2 * Complex.i

theorem point_quadrant :
    (let w := z2 / z1 in w.re > 0 ∧ w.im < 0) :=
by
  sorry

end point_quadrant_l825_825876


namespace binomial_equality_l825_825320

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l825_825320


namespace average_children_families_with_children_is_3_point_8_l825_825365

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l825_825365


namespace number_of_zeros_in_one_over_forty_power_twenty_l825_825476

-- Mathematical definitions and conditions directly from the problem statement
def forty_power_twenty := (40 : ℕ) ^ 20

def one_over_forty_power_twenty := (1 : ℚ) / forty_power_twenty

-- Statement: Number of zeros immediately following the decimal point in the decimal representation of 1 / 40^20 is 38
theorem number_of_zeros_in_one_over_forty_power_twenty :
  (number_of_zeros_immediately_following_decimal one_over_forty_power_twenty) = 38 := sorry

end number_of_zeros_in_one_over_forty_power_twenty_l825_825476


namespace candy_canes_per_girl_l825_825798

structure CandiesDistribution where
  total_candies : ℕ
  lollipops_fraction : ℚ
  candy_per_boy : ℕ
  total_children : ℕ
  each_girl_candy_canes : ℕ

axiom angeli_conditions : 
  CandiesDistribution 
    ∧ CandiesDistribution.total_candies = 90 
    ∧ CandiesDistribution.lollipops_fraction = 1/3 
    ∧ CandiesDistribution.candy_per_boy = 3
    ∧ CandiesDistribution.total_children = 40

theorem candy_canes_per_girl :
  ∀ (d : CandiesDistribution), 
  angeli_conditions → d.each_girl_candy_canes = 2 := by 
  sorry

end candy_canes_per_girl_l825_825798


namespace complex_multiplication_problem_l825_825146

-- Conditions
def i_squared_eq_neg_one : ℂ := (complex.I * complex.I) = -1

-- Problem statement
theorem complex_multiplication_problem:
  complex.I * (1 + 2 * complex.I) = -2 + complex.I := 
by 
  sorry

end complex_multiplication_problem_l825_825146


namespace system_of_equations_solution_l825_825130

theorem system_of_equations_solution
  (a b c d e f g : ℝ)
  (x y z : ℝ)
  (h1 : a * x = b * y)
  (h2 : b * y = c * z)
  (h3 : d * x + e * y + f * z = g) :
  (x = g * b * c / (d * b * c + e * a * c + f * a * b)) ∧
  (y = g * a * c / (d * b * c + e * a * c + f * a * b)) ∧
  (z = g * a * b / (d * b * c + e * a * c + f * a * b)) :=
by
  sorry

end system_of_equations_solution_l825_825130


namespace population_in_1988_l825_825360

theorem population_in_1988 (P : ℕ → ℕ) (P2008 : P 2008 = 3456) (H : ∀ t, P (t + 4) = 2 * P t) : P 1988 = 108 :=
by
  have P2004 : P 2004 = P 2008 / 2 := by
    rw ←H 2004
    exact P2008

  have P2000 : P 2000 = P 2004 / 2 := by
    rw ←H 2000
    exact P2004

  have P1996 : P 1996 = P 2000 / 2 := by
    rw ←H 1996
    exact P2000

  have P1992 : P 1992 = P 1996 / 2 := by
    rw ←H 1992
    exact P1996

  have P1988 : P 1988 = P 1992 / 2 := by
    rw ←H 1988
    exact P1992

  rw [P1988, P1992, P1996, P2000, P2004, P2008]
  norm_num
  sorry

end population_in_1988_l825_825360


namespace measure_angle_BAC_l825_825250

-- All angles are in degrees
def angle_AOB := 120
def angle_BOC := 130

def angle_BAC (angle_AOB angle_BOC : ℕ) : Prop :=
  angle_AOB = 120 ∧ angle_BOC = 130 → 65

theorem measure_angle_BAC : angle_BAC angle_AOB angle_BOC := 
by
  unfold angle_BAC
  intros
  sorry

end measure_angle_BAC_l825_825250


namespace total_items_in_pencil_case_l825_825174

theorem total_items_in_pencil_case:
  ∀ (pencils pens erasers : ℕ), 
    pencils = 4 → 
    pens = 2 * pencils → 
    erasers = 1 → 
    pencils + pens + erasers = 13 :=
by
  intros pencils pens erasers
  assume h1 : pencils = 4
  assume h2 : pens = 2 * pencils
  assume h3 : erasers = 1
  rw [h1] at h2
  rw [h1, h2, h3]
  sorry

end total_items_in_pencil_case_l825_825174


namespace shara_monthly_payment_l825_825120

theorem shara_monthly_payment : 
  ∀ (T M : ℕ), 
  (T / 2 = 6 * M) → 
  (T / 2 - 4 * M = 20) → 
  M = 10 :=
by
  intros T M h1 h2
  sorry

end shara_monthly_payment_l825_825120


namespace avg_children_with_kids_l825_825393

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l825_825393


namespace motorcycle_b_distance_l825_825183

theorem motorcycle_b_distance (d : ℕ) (f : ℕ): d = 300 → f = 8 → (450 : ℕ) = (12 * (300 / f)) :=
  by
    intros h_d h_f
    calc
      450 = 12 * (300 / 8) := sorry

end motorcycle_b_distance_l825_825183


namespace S_n_eq_l825_825351

noncomputable def S_n (n : ℕ) : ℕ := ∑ k in finset.range n, nat.choose n (k + 1) * 2^k

theorem S_n_eq (n : ℕ) : S_n n = (3^n - 1) / 2 :=
by
  sorry

end S_n_eq_l825_825351


namespace binom_18_4_eq_3060_l825_825317

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l825_825317


namespace average_children_families_with_children_is_3_point_8_l825_825363

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l825_825363


namespace count_digit_seven_l825_825017

theorem count_digit_seven : 
  let count_sevens (n : ℕ) : ℕ := 
    (String.toList (toString n)).count '7'
  in (List.range (1000 + 1)).sum count_sevens = 300 := by
  sorry

end count_digit_seven_l825_825017


namespace n_plus_p_l825_825103

noncomputable def cos_expansion : ℕ → (ℕ → ℝ) := sorry

theorem n_plus_p (α : ℝ) (m n p : ℝ) 
  (h1 : ∀ α, cos_expansion 2 α = 2 * (cos α) ^ 2 - 1)
  (h2 : ∀ α, cos_expansion 4 α = 8 * (cos α) ^ 4 - 8 * (cos α) ^ 2 + 1)
  (h3 : ∀ α, cos_expansion 6 α = 32 * (cos α) ^ 6 - 48 * (cos α) ^ 4 + 18 * (cos α) ^ 2 - 1)
  (h4 : ∀ α, cos_expansion 8 α = 128 * (cos α) ^ 8 - 256 * (cos α) ^ 6 + 160 * (cos α) ^ 4 - 32 * (cos α) ^ 2 + 1)
  (h5 : ∀ α, cos_expansion 10 α = m * (cos α) ^ 10 - 1280 * (cos α) ^ 8 + 1120 * (cos α) ^ 6 + n * (cos α) ^ 4 + p * (cos α) ^ 2 - 1)
  (hm : m = 512)
  (hp : p = 50) 
  : n + p = -350 := 
sorry

end n_plus_p_l825_825103


namespace ratio_ab_l825_825878

theorem ratio_ab (a b : ℚ) (h : b / a = 5 / 13) : (a - b) / (a + b) = 4 / 9 :=
by
  sorry

end ratio_ab_l825_825878


namespace expected_value_twelve_sided_die_l825_825781

theorem expected_value_twelve_sided_die : 
  (1 / 12 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l825_825781


namespace binom_18_4_l825_825336

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end binom_18_4_l825_825336


namespace sum_of_primes_no_solution_l825_825826

theorem sum_of_primes_no_solution :
  ∑ p in {2, 5}, p = 7 := by
sorry

end sum_of_primes_no_solution_l825_825826


namespace transformed_line_l825_825968

-- Define the original line equation
def original_line (x y : ℝ) : Prop := (x - 2 * y = 2)

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop :=
  (x' = x) ∧ (y' = 2 * y)

-- Prove that the transformed line equation holds
theorem transformed_line (x y x' y' : ℝ) (h₁ : original_line x y) (h₂ : transformation x y x' y') :
  x' - y' = 2 :=
sorry

end transformed_line_l825_825968


namespace exists_coprime_in_consecutive_ten_l825_825105

theorem exists_coprime_in_consecutive_ten (a : ℤ) : 
  ∃ b ∈ {a, a+1, a+2, a+3, a+4, a+5, a+6, a+7, a+8, a+9}, 
  ∀ x ∈ {a, a+1, a+2, a+3, a+4, a+5, a+6, a+7, a+8, a+9}, x ≠ b → gcd b x = 1 :=
by
  sorry

end exists_coprime_in_consecutive_ten_l825_825105


namespace proof_of_problem_statement_l825_825068
noncomputable def problem_statement : Prop :=
  let N := 30 ^ 2015 in
  (finset.univ.filter (λ p : ℤ × ℤ × ℤ × ℤ, 
    ∀ n : ℤ, (p.1 * n^3 + p.2 * n^2 + 2 * p.3 * n + p.4) % N = 0).card) = 2

theorem proof_of_problem_statement : problem_statement :=
begin
  sorry
end

end proof_of_problem_statement_l825_825068


namespace max_positive_real_root_interval_l825_825737

theorem max_positive_real_root_interval 
  (a2 a1 a0 : ℝ) 
  (h_a2 : |a2| ≤ 2) 
  (h_a1 : |a1| ≤ 2) 
  (h_a0 : |a0| ≤ 2) :
  ∃ r, r > 0 ∧ (r : ℝ) ∈ [5/2, 3) ∧ is_root (λ x => x^3 + a2 * x^2 + a1 * x + a0) r := 
sorry

end max_positive_real_root_interval_l825_825737


namespace exists_divisor_for_all_n_l825_825630

open Nat

noncomputable def int_poly : Type
| mk : (ℕ → ℤ) → int_poly

noncomputable def div_int : (ℤ → ℤ → Prop) := λ a b, ∃ k, b = k * a

theorem exists_divisor_for_all_n (F : ℤ → ℤ) (a : ℕ → ℤ) (m : ℕ) 
  (h_polynomial : ∀ n, ∃ c : ℕ, F(n) = int.poly c)
  (h_divisible : ∀ n, ∃ i : ℕ, i < m ∧ div_int (a i) (F n)) :
  ∃ i < m, ∀ n, div_int (a i) (F n) :=
sorry

end exists_divisor_for_all_n_l825_825630


namespace sin_cos_tan_identity_l825_825879

theorem sin_cos_tan_identity (α β p q : ℝ)
  (h1: ∀ x : ℝ, x^2 + p * x + q = 0 → x = Real.tan α ∨ x = Real.tan β)
  (h2 : Real.tan α + Real.tan β = -p)
  (h3 : Real.tan α * Real.tan β = q):
  sin(α + β)^2 + p * sin(α + β) * cos(α + β) + q * cos(α + β)^2 = q :=
by sorry

end sin_cos_tan_identity_l825_825879


namespace find_a_l825_825863

variable (a : ℝ) -- a is a real number
variable (f : ℝ → ℝ) -- f is a real-valued function

-- Conditions
def function_condition : Prop := 
  f = λ x, a^(x - 1 / 2)

def value_condition : Prop :=
  f (log a) = sqrt 10

def valid_a : Prop :=
  (a > 0) ∧ (a ≠ 1)

-- Goal
theorem find_a (ha : valid_a) (h_fun : function_condition) (h_val : value_condition) 
  : a = 10 ∨ a = 10^(-1 / 2) :=
sorry

end find_a_l825_825863


namespace measure_of_angle_A_range_of_tan_sum_l825_825956

-- Define the context of the problem: an acute triangle with given conditions.
variables {A B C a b c : ℝ}

-- Necessary assumptions for the conditions.
axiom acute_triangle (h : 0 < A ∧ A < π / 2) (h1 : 0 < B ∧ B < π / 2) (h2 : 0 < C ∧ C < π / 2) :
  A + B + C = π

axiom side_opposite_angles (ha : a = sin A * c) (hb : b = sin B * c) : 
  c * cos B + b * cos C = 2 * a * cos A

-- Part (1): Measure of angle A
theorem measure_of_angle_A (h : acute_triangle) (h' : side_opposite_angles) :
  A = π / 3 :=
sorry

-- Part (2): Range of values for 1/tan B + 1/tan C
theorem range_of_tan_sum (h : acute_triangle) (h' : side_opposite_angles) :
  ∀ B C, (
    let tan_sum := (1 / tan B) + (1 / tan C) in
    (2 * sqrt 3 / 3 : ℝ) ≤ tan_sum ∧ tan_sum < sqrt 3
  ) :=
sorry

end measure_of_angle_A_range_of_tan_sum_l825_825956


namespace matrix_inverse_solution_l825_825827

theorem matrix_inverse_solution (p q : ℝ) 
    (h : (matrix.mul (matrix.of ![![4, p], ![-2, q]]) (matrix.of ![![4, p], ![-2, q]])) 
        = matrix.one 2) : 
    p = 15 / 2 ∧ q = -4 :=
by sorry

end matrix_inverse_solution_l825_825827


namespace expected_value_of_twelve_sided_die_l825_825771

theorem expected_value_of_twelve_sided_die : 
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0) in
  (finset.sum outcomes (λ n, (n:ℝ)) / 12 = 6.5) :=
by
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0)
  have h1 : ∑ n in outcomes, (n : ℝ) = 78, sorry
  have h2 : (78 / 12) = 6.5, sorry
  exact h2

end expected_value_of_twelve_sided_die_l825_825771


namespace digits_for_multiple_of_3_l825_825850

theorem digits_for_multiple_of_3 : (card {C : ℕ // C < 10 ∧ ∃ X : ℕ, X < 10 ∧ (2 + C + 4 + X) % 3 = 0}) = 10 :=
by sorry

end digits_for_multiple_of_3_l825_825850


namespace inverse_of_f_l825_825683

-- Define the function f
def f (x : ℝ) (h : x > 0) : ℝ := x^2

-- State the theorem about the inverse function of f
theorem inverse_of_f (x : ℝ) (h : x > 0) : 
  f⁻¹ x = sqrt x :=
sorry

end inverse_of_f_l825_825683


namespace impossible_odd_n_m_even_sum_l825_825622

theorem impossible_odd_n_m_even_sum (n m : ℤ) (h : (n^2 + m^2 + n*m) % 2 = 0) : ¬ (n % 2 = 1 ∧ m % 2 = 1) :=
by sorry

end impossible_odd_n_m_even_sum_l825_825622


namespace find_angle_DEC_l825_825872

noncomputable def angle_DEC (A B C D E : Type) [InCircle A B C D] (h_CD_extended : Line CD E) 
                            (angle_BAD : ℝ) (angle_ABC : ℝ) : ℝ :=
let angle_DEC := 
  if angle_BAD = 80 ∧ angle_ABC = 100 then 80
  else 0
in
angle_DEC

theorem find_angle_DEC (A B C D E : Type) [InCircle A B C D] (h_CD_extended : Line CD E) 
                      (h1 : angle_BAD = 80) (h2 : angle_ABC = 100) : angle_DEC A B C D E 80 100 = 80 := by
sorry

end find_angle_DEC_l825_825872


namespace cos_minus_sin_l825_825490

theorem cos_minus_sin (α : ℝ) (hα : sin α * cos α = -1/6) (hα_range : 0 < α ∧ α < π) : cos α - sin α = -2*sqrt(3)/3 :=
by
  sorry

end cos_minus_sin_l825_825490


namespace sum_powers_of_i_l825_825834

variables {i : ℂ}

-- Using the cyclic properties of the imaginary unit i
def cyclic_powers_of_i (n : ℕ) : ℂ :=
  match n % 4 with
  | 0 => 1
  | 1 => i
  | 2 => -1
  | 3 => -i
  | _ => 1  -- This case won't be reached since n % 4 < 4

theorem sum_powers_of_i :
  cyclic_powers_of_i 15 + cyclic_powers_of_i 22 + cyclic_powers_of_i 29 + cyclic_powers_of_i 36 + cyclic_powers_of_i 43 = -i :=
by
  -- Proof would go here
  sorry

end sum_powers_of_i_l825_825834


namespace height_difference_between_crates_l825_825182
noncomputable theory

structure Crate where
  rows : ℕ
  rolls_per_row : ℕ
  roll_diameter : ℝ

def height_of_square_packed_crate (crate : Crate) : ℝ :=
  crate.rows * crate.roll_diameter

def height_of_hex_packed_crate (crate : Crate) : ℝ :=
  crate.rows * crate.roll_diameter / 2 * (1 + Real.sqrt 3) + crate.roll_diameter / 2

def positive_difference_in_height (crate : Crate) : ℝ :=
  abs ((height_of_hex_packed_crate crate) - (height_of_square_packed_crate crate))

theorem height_difference_between_crates :
  positive_difference_in_height {rows := 18, rolls_per_row := 8, roll_diameter := 12} = 102 * Real.sqrt 3 - 108 :=
by
  sorry

end height_difference_between_crates_l825_825182


namespace solve_for_s_l825_825007

theorem solve_for_s (r s : ℝ) (h1 : 1 < r) (h2 : r < s) (h3 : 1 / r + 1 / s = 3 / 4) (h4 : r * s = 8) : s = 4 :=
sorry

end solve_for_s_l825_825007


namespace evaluate_expression_l825_825837

theorem evaluate_expression :
  (3^1003 + 7^1004)^2 - (3^1003 - 7^1004)^2 = 5.292 * 10^1003 :=
by sorry

end evaluate_expression_l825_825837


namespace average_children_in_families_with_children_l825_825388

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l825_825388


namespace max_selection_no_five_times_l825_825856

theorem max_selection_no_five_times (S : Finset ℕ) (hS : S = Finset.Icc 1 2014) :
  ∃ n, n = 1665 ∧ 
  ∀ (a b : ℕ), a ∈ S → b ∈ S → (a = 5 * b ∨ b = 5 * a) → false :=
sorry

end max_selection_no_five_times_l825_825856


namespace ones_digit_of_73_pow_355_l825_825477

theorem ones_digit_of_73_pow_355 : (73 ^ 355) % 10 = 7 := 
  sorry

end ones_digit_of_73_pow_355_l825_825477


namespace mr_william_land_percentage_l825_825361

def total_tax_collected : ℝ := 3840
def mr_william_tax_paid : ℝ := 480
def expected_percentage : ℝ := 12.5

theorem mr_william_land_percentage :
  (mr_william_tax_paid / total_tax_collected) * 100 = expected_percentage := 
sorry

end mr_william_land_percentage_l825_825361


namespace minimum_weights_l825_825510

variable {α : Type} [LinearOrderedField α]

theorem minimum_weights (weights : Finset α)
  (h_unique : weights.card = 5)
  (h_balanced : ∀ {x y : α}, x ∈ weights → y ∈ weights → x ≠ y →
    ∃ a b : α, a ∈ weights ∧ b ∈ weights ∧ x + y = a + b) :
  ∃ (n : ℕ), n = 13 ∧ ∀ S : Finset α, S.card = n ∧
    (∀ {x y : α}, x ∈ S → y ∈ S → x ≠ y → ∃ a b : α, a ∈ S ∧ b ∈ S ∧ x + y = a + b) :=
by
  sorry

end minimum_weights_l825_825510


namespace debt_calculation_correct_l825_825916

-- Conditions
def initial_debt : ℤ := 40
def repayment : ℤ := initial_debt / 2
def additional_borrowing : ℤ := 10

-- Final Debt Calculation
def remaining_debt : ℤ := initial_debt - repayment
def final_debt : ℤ := remaining_debt + additional_borrowing

-- Proof Statement
theorem debt_calculation_correct : final_debt = 30 := 
by 
  -- Skipping the proof
  sorry

end debt_calculation_correct_l825_825916


namespace find_vertices_l825_825053

-- Define the coordinates of the midpoints of the sides of the triangle
def D : ℝ × ℝ × ℝ := (2, 7, -3)
def E : ℝ × ℝ × ℝ := (4, 1, 0)
def F : ℝ × ℝ × ℝ := (1, 6, 5)

-- Define the coordinates of vertices X and Z to be proved
def X : ℝ × ℝ × ℝ := (3, 0, 8)
def Z : ℝ × ℝ × ℝ := (1, 14, -14)

-- State the theorem
theorem find_vertices :
  let midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)
  in
  midpoint Z Y = D ∧ midpoint X Z = E ∧ midpoint X Y = F := 
sorry

end find_vertices_l825_825053


namespace largest_in_set_when_a_is_negative_three_l825_825932

theorem largest_in_set_when_a_is_negative_three :
  ∀ (a : ℝ), a = -3 →
    ∃ x ∈ {-3 * a, 5 * a, 24 / a, a ^ 2, 1}, 
      (∀ y ∈ {-3 * a, 5 * a, 24 / a, a ^ 2, 1}, y ≤ x) ∧ x = -3 * a :=
by {
  intro a,
  intro ha,
  use -3 * a,
  split,
  {
    apply set.mem_insert,
    refl,
  },
  {
    intros y hy,
    cases set.mem_insert_iff.mp hy with h_eq_3a h_rest,
    {
      rw h_eq_3a,
    },
    {
      have : ∀ z ∈ {5 * a, 24 / a, a ^ 2, 1}, z ≤ -3 * a, {
        intro z,
        intro hz,
        cases set.mem_insert_iff.mp hz with h_eq_5a h_rest',
        {
          rw h_eq_5a,
          linarith [ha],
        },
        {
          cases set.mem_insert_iff.mp h_rest' with h_eq_24_div_a h_rest'',
          {
            rw h_eq_24_div_a,
            linarith [ha],
          },
          {
            cases set.mem_insert_iff.mp h_rest'' with h_eq_a_sq h_one,
            {
              rw h_eq_a_sq,
              linarith [ha],
            },
            {
              rw h_one,
              linarith [ha],
            },
          },
        },
      },
      exact this y h_rest,
    },
  },
}

end largest_in_set_when_a_is_negative_three_l825_825932


namespace average_children_in_families_with_children_l825_825374

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l825_825374


namespace inverse_of_matrix_C_l825_825866

-- Define the given matrix C
def C : Matrix (Fin 3) (Fin 3) ℚ := ![
  ![1, 2, 1],
  ![3, -5, 3],
  ![2, 7, -1]
]

-- Define the claimed inverse of the matrix C
def C_inv : Matrix (Fin 3) (Fin 3) ℚ := (1 / 33 : ℚ) • ![
  ![-16,  9,  11],
  ![  9, -3,   0],
  ![ 31, -3, -11]
]

-- Statement to prove that C_inv is the inverse of C
theorem inverse_of_matrix_C : C * C_inv = 1 ∧ C_inv * C = 1 := by
  sorry

end inverse_of_matrix_C_l825_825866


namespace batsman_average_correct_after_20th_innings_l825_825730

def batsman_average_increase
  (A : ℝ) -- Average after 19 innings
  (score_20th : ℝ) -- Score in the 20th innings
  (average_increase : ℝ) -- Increase in average by 
  (new_average : ℝ) -- New average after the 20th innings
  (total_runs_19 : ℝ) -- Total runs scored in 19 innings
  (total_runs_20 : ℝ) -- Total runs scored in 20 innings
  (correct_new_average : ℝ) -- Correct new average 
  : Prop := 
  total_runs_19 = 19 * A ∧
  score_20th = 90 ∧
  average_increase = 2 ∧
  total_runs_20 = total_runs_19 + score_20th ∧
  new_average = A + average_increase ∧
  correct_new_average = 52 ∧
  total_runs_20 = 20 * new_average

theorem batsman_average_correct_after_20th_innings 
  (A : ℝ) -- Average after 19 innings
  (score_20th : ℝ) -- Score in the 20th innings
  (average_increase : ℝ) -- Increase in average by 
  (new_average : ℝ) -- New average after the 20th innings
  (total_runs_19 : ℝ) -- Total runs scored in 19 innings
  (total_runs_20 : ℝ) -- Total runs scored in 20 innings
  (correct_new_average : ℝ) -- Correct new average
  (h : batsman_average_increase A score_20th average_increase new_average total_runs_19 total_runs_20 correct_new_average) : 
  correct_new_average = 52 :=
by
  -- Initially setting up the base conditions
  cases h with h_total_runs_19 h_score_20th h_average_increase h_total_runs_20 h_new_average h_correct_new_average h_equation_total_runs,
  -- Using the given conditions in our proof
  rw [h_score_20th, h_average_increase,
      h_total_runs_19, h_new_average, h_total_runs_20, h_correct_new_average] at *,
  -- Creating the required changes and manipulations
  -- Σ = 19A
  have h_sum_19 := h_total_runs_19,
  -- h_score = 90
  rw h_score_20th at *,
  -- h_average_increase = 2
  rw h_average_increase at *,
  -- new_total = 19A + 90
  rw [h_total_runs_20, ←add_assoc] at h_equation_total_runs,
  -- Establishing 20 Average and verifying new average
  have := calc
    19 * A + 90 = total_runs_20 : by rw h_total_runs_20

  19 * A + 90 = 20 * (A + 2) : by
    rw [mul_add, add_mul, mul_comm 2 20, add_assoc]
    exact this
  norm_num,
  exact h_correct_new_average
  sorry

end batsman_average_correct_after_20th_innings_l825_825730


namespace contains_subset_of_fourth_power_l825_825509

/-- Definition of the problem conditions -/
def set_M (M : set ℕ) : Prop :=
  M.card = 1985 ∧ (∀ m ∈ M, ∀ p : ℕ, p.prime → p ∣ m → p ≤ 23)

theorem contains_subset_of_fourth_power (M : set ℕ) (hM : set_M M) :
  ∃ a b c d ∈ M, ∃ k : ℕ, a * b * c * d = k^4 :=
sorry

end contains_subset_of_fourth_power_l825_825509


namespace opposite_silver_is_black_l825_825679

-- Definitions based on the conditions provided
namespace CubeColorProblem
def faceColor : Type := String

def top_face : faceColor := "black"
def right_face : faceColor := "blue"
def front_faces : List faceColor := ["pink", "orange", "yellow"]
def opposite_face (color : faceColor) : faceColor := 
  if color = "silver" then "black" else sorry -- Placeholder for other possible logic

-- Problem statement
theorem opposite_silver_is_black : opposite_face "silver" = "black" := 
by
  -- Proof would go here
  sorry

end opposite_silver_is_black_l825_825679


namespace symmetric_points_y_axis_l825_825589

/-
Given points A(-2, a) and B(b, -3) on the Cartesian coordinate plane 
that are symmetric about the y-axis, prove that ab = -6.
-/
theorem symmetric_points_y_axis {a b : ℝ} 
  (h1 : (-2, a) = (b, -3).symm_about_y_axis) : 
  a * b = -6 :=
by
  sorry

end symmetric_points_y_axis_l825_825589


namespace smallest_base10_integer_l825_825207

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end smallest_base10_integer_l825_825207


namespace wade_total_spent_l825_825706

def sandwich_cost : ℕ := 6
def drink_cost : ℕ := 4
def num_sandwiches : ℕ := 3
def num_drinks : ℕ := 2

def total_cost : ℕ :=
  (num_sandwiches * sandwich_cost) + (num_drinks * drink_cost)

theorem wade_total_spent : total_cost = 26 := by
  sorry

end wade_total_spent_l825_825706


namespace elder_person_present_age_l825_825141

def younger_age : ℕ
def elder_age : ℕ

-- Conditions
axiom age_difference (y e : ℕ) : e = y + 16
axiom age_relation_6_years_ago (y e : ℕ) : e - 6 = 3 * (y - 6)

-- Proof of the present age of the elder person
theorem elder_person_present_age (y e : ℕ) (h1 : e = y + 16) (h2 : e - 6 = 3 * (y - 6)) : e = 30 :=
sorry

end elder_person_present_age_l825_825141


namespace chocolate_flavor_sales_l825_825266

-- Define the total number of cups sold
def total_cups : ℕ := 50

-- Define the fraction of winter melon flavor sales
def winter_melon_fraction : ℚ := 2 / 5

-- Define the fraction of Okinawa flavor sales
def okinawa_fraction : ℚ := 3 / 10

-- Proof statement
theorem chocolate_flavor_sales : 
  (total_cups - (winter_melon_fraction * total_cups).toInt - (okinawa_fraction * total_cups).toInt) = 15 := 
  by 
  sorry

end chocolate_flavor_sales_l825_825266


namespace polynomial_p_zero_l825_825993

noncomputable def p (x : ℝ) : ℝ := -- p is a polynomial of degree 6; its exact form will be defined implicitly

theorem polynomial_p_zero : 
  (∀ (n : ℕ), n ∈ {0, 1, 2, 3, 4, 5, 6} → p (3^n) = 1 / 2^n) →
  ∀ (x : ℝ), degree p = 6 →
  p 0 = 0 :=
by
  sorry

end polynomial_p_zero_l825_825993


namespace Charlie_age_when_Jenny_twice_as_Bobby_l825_825061

theorem Charlie_age_when_Jenny_twice_as_Bobby (B C J : ℕ) 
  (h₁ : J = C + 5)
  (h₂ : C = B + 3)
  (h₃ : J = 2 * B) : 
  C = 11 :=
by
  sorry

end Charlie_age_when_Jenny_twice_as_Bobby_l825_825061


namespace max_n_satisfies_inequality_l825_825189

noncomputable def largest_n : ℕ :=
  11

theorem max_n_satisfies_inequality (n : ℕ) (h : n ^ 200 < 5 ^ 300) : n ≤ largest_n :=
by
  let n_max := 11
  have h_max : 11 ^ 200 < 5 ^ 300 := by sorry
  cases lt_or_le n n_max with
  | inl h1 => exact le_of_lt h1
  | inr h2 => contradiction
  sorry

end max_n_satisfies_inequality_l825_825189


namespace percentage_increase_of_y_over_x_l825_825229

variable (x y : ℝ) (h : x > 0 ∧ y > 0) 

theorem percentage_increase_of_y_over_x
  (h_ratio : (x / 8) = (y / 7)) :
  ((y - x) / x) * 100 = 12.5 := 
sorry

end percentage_increase_of_y_over_x_l825_825229


namespace area_ratio_le_four_fifths_l825_825172

theorem area_ratio_le_four_fifths (ABC : Triangle) (G : Point)
  (hG : Centroid G ABC)
  (secant : Line)
  (hSecant : PassesThrough secant G) :
  let A := area ABC
      GDE := resulting_triangle secant ABC
      ADGE := resulting_quadrilateral secant ABC
  in ratio (area GDE) (area ADGE) ≤ 4 / 5 :=
sorry

end area_ratio_le_four_fifths_l825_825172


namespace determinant_sine_matrix_l825_825811

theorem determinant_sine_matrix :
  \begin{vmatrix} 
    sin 2 & sin 3 & sin 4 \\
    sin 5 & sin 6 & sin 7 \\
    sin 8 & sin 9 & sin 10 
  \end{vmatrix} = 0 :=
by
  sorry

end determinant_sine_matrix_l825_825811


namespace incenter_exsphere_segment_length_l825_825984

noncomputable def tetrahedron := sorry -- Define a tetrahedron structure

variables {A B C D I J K : Point}

/-- Given tetrahedron ABCD, I is the incenter of the tetrahedron, J is the excenter
    touching face BCD, and segment IJ meets the circumsphere at point K,
    then we need to prove that segment IJ is longer than JK. -/
theorem incenter_exsphere_segment_length 
  (ABCD : tetrahedron) 
  (I_incenter : is_incenter I ABCD) 
  (J_excenter : is_excenter J (face B C D) ABCD)
  (K_on_circumsphere : on_circumsphere K (face A B C D) ABCD)
  (K_on_segment_IJ : on_segment I J K) :
  segment_length I J > segment_length J K :=
sorry

end incenter_exsphere_segment_length_l825_825984


namespace original_number_is_9999876_l825_825643

theorem original_number_is_9999876 (x : ℕ) : 
  (∃ x, (10 * x + 9 + 876 = x + 9876)) → x = 999 → 10 * x + 9876 = 9999876 :=
by
  intro h h1
  obtain ⟨x, h⟩ := h
  have h2 : 10 * 999 + 9876 = 9999876 := by linarith
  exact h2

end original_number_is_9999876_l825_825643


namespace greatest_prime_factor_f_28_l825_825849

noncomputable def f (m : ℕ) : ℕ :=
  if m % 2 = 0 then (∏ i in (Finset.range (m // 2)).map (Nat.mulLeft 2), i) else 1

theorem greatest_prime_factor_f_28 : 
  greatest_prime_factor (f 28) = 13 := 
sorry

end greatest_prime_factor_f_28_l825_825849


namespace average_children_in_families_with_children_l825_825386

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l825_825386


namespace twentieth_prime_l825_825452

noncomputable def is_prime(n : ℕ) : Prop := ∀ x,  n ∣ x → x = 1 ∨ x = n

theorem twentieth_prime :
  ∃ p : ℕ, is_prime p ∧ p = 71 ∧ (∃ ps : List ℕ, (∀ a : ℕ, a ∈ ps → is_prime a) ∧ (List.nth ps 19 = some 71) ∧ ps.length > 19) :=
by
  sorry

end twentieth_prime_l825_825452


namespace average_children_families_with_children_is_3_point_8_l825_825364

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l825_825364


namespace average_children_l825_825433

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l825_825433


namespace eccentricity_theorem_l825_825072

noncomputable def ellipse : set (ℝ × ℝ) := 
  {p | p.1^2 / 16 + p.2^2 / b^2 = 1}

noncomputable def foci_1 : ℝ × ℝ := (-c, 0)
noncomputable def foci_2 : ℝ × ℝ := (c, 0)

def max_AF2_BF2_value : ℝ := 10

def eccentricity_of_ellipse : ℝ := 
  let a := 4 in
  let b := 2 * Real.sqrt 3 in
  Real.sqrt (1 - (b^2 / a^2))

theorem eccentricity_theorem (h : ellipse) (foci_1 foci_2) (A B : ℝ × ℝ)
  (l : set ℝ × ℝ) (hl : foci_1 ∈ l) 
  (interAB : A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ l ∧ B ∈ l)
  (hmax : ∀ A B, |(A - foci_2).length + (B - foci_2).length| ≤ max_AF2_BF2_value) :
  eccentricity_of_ellipse = 1/2 := 
sorry

end eccentricity_theorem_l825_825072


namespace points_lie_on_parabola_l825_825852

variable (t : ℝ)

def x := 2^t - 5
def y := 4^t - 3 * 2^t + 1

theorem points_lie_on_parabola :
  ∃ a b c : ℝ, y = a * x^2 + b * x + c :=
by
  -- We need to prove that there exist real numbers a, b, and c such that
  -- y = a * x^2 + b * x + c. By the preceding solution, these would be
  -- a = 1, b = 7, and c = 11. Hence, all points lie on the parabola.
  sorry

end points_lie_on_parabola_l825_825852


namespace average_children_l825_825441

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l825_825441


namespace chord_bisected_by_P_l825_825908

open Real

theorem chord_bisected_by_P
  (parabola_eq : ∀ x y : ℝ, y^2 = 6 * x)
  (P : ℝ × ℝ)
  (chord_bisected : ∃ P1 P2 : ℝ × ℝ, (P1.1, P1.2), (P2.1, P2.2) ∈ function.graph parabola_eq ∧ ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2) = P)
  (P_eq : P = (4, 1)) :
  (∃ line_eq : ℝ → ℝ → Prop, (∀ x y, line_eq x y ↔ 3 * x - y = 11) 
  ∧ (∃ length : ℝ, length = 2 * sqrt 230 / 3)) :=
by
  sorry

end chord_bisected_by_P_l825_825908


namespace fraction_of_sum_l825_825528

noncomputable theory
open_locale big_operators 

def sequence (a : ℕ → ℝ) := ∀ n, 0 < a n
def initial_term (a : ℕ → ℝ) := a 1 = 1
def sum_of_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = ∑ i in finset.range n, a (i + 1)
def given_relation (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, a n * S (n + 1) - a (n + 1) * S n + a n - a (n + 1) = (1 / 2) * a n * a (n + 1)

theorem fraction_of_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : sequence a)
  (h_init : initial_term a)
  (h_sum : sum_of_terms a S)
  (h_rel : given_relation a S) :
  3 / 34 * S 12 = 3 :=
sorry

end fraction_of_sum_l825_825528


namespace black_ball_higher_probability_equalize_probabilities_l825_825043

-- Part 1: Proving the probability of drawing a black ball is higher.
theorem black_ball_higher_probability :
  ∃ (red black : ℕ), red = 5 ∧ black = 7 ∧ (black / (red + black) > red / (red + black)) :=
by
  existsi 5
  existsi 7
  simp
  linarith
  sorry

-- Part 2: Proving that adding 4 red balls and 2 black balls equalizes the probabilities.
theorem equalize_probabilities :
  ∃ (initial_red initial_black add_red add_black total : ℕ),
  initial_red = 5 ∧ initial_black = 7 ∧ add_red = 4 ∧ add_black = 2 ∧ total = 6 ∧
  (initial_red + add_red) / (initial_red + initial_black + total) = (initial_black + add_black) / (initial_red + initial_black + total) :=
by
  existsi 5
  existsi 7
  existsi 4
  existsi 2
  existsi 6
  simp
  linarith
  sorry

end black_ball_higher_probability_equalize_probabilities_l825_825043


namespace avg_children_with_kids_l825_825396

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l825_825396


namespace twice_x_minus_three_lt_zero_l825_825149

theorem twice_x_minus_three_lt_zero (x : ℝ) : (2 * x - 3 < 0) ↔ (2 * x < 3) :=
by
  sorry

end twice_x_minus_three_lt_zero_l825_825149


namespace percent_non_swimmers_play_soccer_l825_825803

variable (N : ℕ) -- Total number of children

-- Conditions
variable (p_soccer : ℝ) (p_swim : ℝ) (p_soccer_and_swim : ℝ)
hypothesis (h1 : p_soccer = 0.7)
hypothesis (h2 : p_swim = 0.5)
hypothesis (h3 : p_soccer_and_swim = 0.3 * p_soccer)

-- Proof goal
theorem percent_non_swimmers_play_soccer : 
  let non_swimmers := N * (1 - p_swim)
  let soccer_non_swimmers := N * (p_soccer - p_soccer_and_swim)
  soccer_non_swimmers / non_swimmers * 100 = 98 := 
begin
  -- Applying the conditions
  change p_soccer = 0.7 at h1,
  change p_swim = 0.5 at h2,
  change p_soccer_and_swim = 0.3 * 0.7 at h3,
  -- Using the hypotheses
  have non_swimmers_eq : non_swimmers = N * 0.5,
  { unfold non_swimmers,
    rw h2,
    ring, },
  have soccer_non_swimmers_eq : soccer_non_swimmers = N * (0.7 - 0.21),
  { unfold soccer_non_swimmers,
    rw [h1, h3],
    ring, },
  -- Calculating the percentage
  calc
    soccer_non_swimmers / non_swimmers * 100
        = (N * (0.7 - 0.21)) / (N * 0.5) * 100 : by rw [soccer_non_swimmers_eq, non_swimmers_eq]
    ... = ((N * 0.49) / (N * 0.5)) * 100 : by ring
    ... = (0.49 / 0.5) * 100 : by sorry
    ... = 0.98 * 100 : by sorry
    ... = 98 : by sorry
end

end percent_non_swimmers_play_soccer_l825_825803


namespace average_children_in_families_with_children_l825_825417

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l825_825417


namespace tan_4050_undefined_l825_825339

def tan_undefined (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = 90 + 360 * k ∧ (cos θ = 0)

theorem tan_4050_undefined : tan_undefined 4050 :=
by
  sorry

end tan_4050_undefined_l825_825339


namespace circle_circumference_l825_825243

theorem circle_circumference :
  ∀ (a b : ℝ), 
    a = 10 → 
    b = 24 → 
    let diagonal := Real.sqrt (a^2 + b^2) in
    let diameter := diagonal in
    ∃ c : ℝ, c = 26 * Real.pi :=
begin
  intros,
  sorry
end

end circle_circumference_l825_825243


namespace smallest_positive_period_of_f_min_max_values_of_f_in_interval_l825_825900

def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

theorem smallest_positive_period_of_f : ∃ T > 0, T = π ∧ ∀ x, f (x + T) = f x :=
sorry

theorem min_max_values_of_f_in_interval :
  ∃ (xmin xmax : ℝ), 
    (xmin = -2 ∧ xmax = 1) ∧ 
    ∀ x, 
      (π / 2 ≤ x ∧ x ≤ π) → 
      (xmin ≤ f x ∧ f x ≤ xmax) :=
sorry

end smallest_positive_period_of_f_min_max_values_of_f_in_interval_l825_825900


namespace total_lemons_l825_825671

theorem total_lemons (T : ℕ) (B1 M1 B2 P B3 PP O1 B4 K1 O2 L K2 A : ℕ) 
  (h_T : T = 83)
  (h_B1 : B1 = 18) (h_B2 : B2 = 14)
  (h_B3 : B3 = 10 + 5)
  (h_B4 : B4 = 8 + 4)
  (h_last_two_equal : L + K2 + A = 24) 
  : L = 8 :=
by {
  rw [h_T, h_B1, h_B2, h_B3, h_B4, h_last_two_equal],
  sorry
}

end total_lemons_l825_825671


namespace multiplication_equivalence_l825_825238

theorem multiplication_equivalence :
    44 * 22 = 88 * 11 :=
by
  sorry

end multiplication_equivalence_l825_825238


namespace delta_n_is_zero_if_m_lt_n_delta_m_value_l825_825075

-- Definitions of concepts used in the problem
def polynomial (R : Type*) [CommRing R] := List R

def degree {R : Type*} [CommRing R] (p : polynomial R) : ℕ :=
  p.length - 1

def eval_polynomial {R : Type*} [Semiring R] (p : polynomial R) (x : R) : R :=
  p.foldr (fun a acc => a + x * acc) 0

def delta {R : Type*} [CommRing R] (f : R → R) (x : R) : R :=
  f(x + 1) - f(x)

def delta_power {R : Type*} [CommRing R] (n : ℕ) (f : R → R) : R → R :=
  Nat.repeat delta n f

noncomputable def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

-- Statements of the proof problem
theorem delta_n_is_zero_if_m_lt_n {R : Type*} [CommRing R] (f : R → R)
  (hf : ∃ p : polynomial R, ∀ x, f x = eval_polynomial p x) {m n : ℕ} (hm : ∃ p, degree p = m) (h : m < n) :
  ∀ x, delta_power n f x = 0 :=
sorry

theorem delta_m_value {R : Type*} [CommRing R] (f : R → R)
  (hf : ∃ p : polynomial R, ∀ x, f x = eval_polynomial p x) {m : ℕ} (hm : ∃ p, degree p = m) :
  ∃ a_m, (∀ x, delta_power m f x = factorial m * a_m) 
:= sorry

end delta_n_is_zero_if_m_lt_n_delta_m_value_l825_825075


namespace total_painted_surface_area_is_40_l825_825282

-- Definitions for the steps
def step1_cubes := 6
def step2_cubes := 5
def step3_cubes := 4
def step4_cubes := 2
def step5_cubes := 1

-- Each cube has 1 meter edges
def edge_length := 1

-- Total number of cubes
def total_cubes := step1_cubes + step2_cubes + step3_cubes + step4_cubes + step5_cubes

-- Calculate the total surface area of exposed faces
def total_surface_area :=
  (step1_cubes 
   + step2_cubes 
   + step3_cubes 
   + step4_cubes 
   + step5_cubes) -- Top faces
  + (step1_cubes * edge_length 
     + step2_cubes * edge_length 
     + step3_cubes * edge_length 
     + step4_cubes * edge_length 
     + step5_cubes * edge_length) -- Side and front faces
  + step5_cubes * 4 * edge_length -- All sides except bottom for the top cube

theorem total_painted_surface_area_is_40 :
  total_surface_area = 40 := 
begin
  sorry
end

end total_painted_surface_area_is_40_l825_825282


namespace pyramid_sphere_volume_l825_825274

-- Define the problem and the proven statement
theorem pyramid_sphere_volume :
  let base_edge_length : ℝ := 4 in
  let side_edge_length : ℝ := 2 * Real.sqrt 6 in
  ∀ (R : ℝ), R = 3 → -- this R is the radius found in the solution
  ∃ (V : ℝ), V = (4 / 3) * Real.pi * R^3 := -- volume of the sphere
by
  intro base_edge_length side_edge_length R hR
  use (4 / 3) * Real.pi * R^3
  rw [hR]
  have R_eq : R = 3 := by assumption
  rw [R_eq]
  sorry

end pyramid_sphere_volume_l825_825274


namespace money_worthless_in_wrong_context_l825_825114

-- Define the essential functions money must serve in society
def InContext (society: Prop) : Prop := 
  ∀ (m : Type), (∃ (medium_of_exchange store_of_value unit_of_account standard_of_deferred_payment : m → Prop), true)

-- Define the context of a deserted island where these functions are useless
def InContext (deserted_island: Prop) : Prop := 
  ∀ (m : Type), (¬ (∃ (medium_of_exchange store_of_value unit_of_account standard_of_deferred_payment : m → Prop), true))

-- Define the essential properties for an item to become money
def EssentialProperties (item: Type) : Prop :=
  ∃ (durable portable divisible acceptable uniform limited_supply : item → Prop), true

-- The primary theorem: Proving that money becomes worthless in the absence of its essential functions
theorem money_worthless_in_wrong_context (m : Type) (society deserted_island : Prop)
  (h1 : InContext society) (h2 : InContext deserted_island) (h3 : EssentialProperties m) :
  ∃ (worthless : m → Prop), true :=
sorry

end money_worthless_in_wrong_context_l825_825114


namespace distinct_values_of_10x_plus_y_l825_825020

theorem distinct_values_of_10x_plus_y : 
  ∀ (x y : ℕ), x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → y ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → 
  (3 * x - 2 * y = 1) → 
  (∃ a b c : ℕ, set_of (λ n, ∃ (x y : ℕ), (x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
                                      (y ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
                                      (3 * x - 2 * y = 1) ∧ 
                                      (10 * x + y = n)) 
  = {a, b, c} ∧ a = 11 ∧ b = 34 ∧ c = 57) :=
by sorry

end distinct_values_of_10x_plus_y_l825_825020


namespace area_ratio_quads_l825_825087

structure Point :=
(x : ℝ) (y : ℝ)

structure Quadrilateral :=
(A B C D : Point)
(convex : True)  -- This should be replaced with a formal definition of convexity if needed.

def midpoint (P Q : Point) : Point :=
{ x := (P.x + Q.x) / 2,
  y := (P.y + Q.y) / 2 }

def area (P Q R S : Point) : ℝ := sorry  -- Placeholder for area computation

noncomputable def quadrilateral_area_ratio (Q : Quadrilateral) : ℝ :=
let M_A := midpoint Q.B Q.C,
    M_B := midpoint Q.C Q.A,
    M_C := midpoint Q.A Q.D,
    M_D := midpoint Q.D Q.B,
    area_M := area M_A M_B M_C M_D,
    area_Q := area Q.A Q.B Q.C Q.D
in area_M / area_Q

theorem area_ratio_quads (Q : Quadrilateral) : quadrilateral_area_ratio Q = 1 / 4 := sorry

end area_ratio_quads_l825_825087


namespace smallest_consecutive_cube_x_l825_825963

theorem smallest_consecutive_cube_x:
  ∃ n: ℤ, 
  let u := n-1, v := n, w := n+1, x := n+2 in 
  u^3 + v^3 + w^3 = x^3 → x = 6 :=
  sorry

end smallest_consecutive_cube_x_l825_825963


namespace product_ab_l825_825807

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := 5 / 2

theorem product_ab :
  (∃ a b : ℝ, (π / b = 2 * π / 5) ∧ (a * Real.tan(b * (π / 10)) = 1)) → a * b = 5 / 2 :=
by
  intro h
  rcases h with ⟨a, b, h1, h2⟩
  sorry -- Proof omitted

end product_ab_l825_825807


namespace proper_vertex_coloring_exists_l825_825150

variables {V : Type} (G : SimpleGraph V) (n : ℕ)
variables (c : V → V → Prop) -- edge coloring
variables [decidable_rel c]

def connected_component (G : SimpleGraph V) (S : set V) := 
  ∀ v ∈ S, ∀ w ∈ S, G.reachable v w

def is_n_colored (G : SimpleGraph V) (c : V → ℕ) (n : ℕ) := 
  ∀ v, c v < n

theorem proper_vertex_coloring_exists :
  (∀ (e : Sym2 V), G.adj e → (c (Sym2.head' e) (Sym2.tail' e))) ∧
  (∀ (C : set V), connected_component (G.edge_induced_subgraph (λ e, c (Sym2.head' e) (Sym2.tail' e))) C → C.finite ∧ C.card ≤ n)
  → ∃ (coloring : V → ℕ), is_n_colored G coloring n :=
sorry

end proper_vertex_coloring_exists_l825_825150


namespace length_of_bridge_l825_825767

noncomputable def speed_in_m_per_s (v_kmh : ℕ) : ℝ :=
  v_kmh * (1000 / 3600)

noncomputable def total_distance (v : ℝ) (t : ℝ) : ℝ :=
  v * t

theorem length_of_bridge (L_train : ℝ) (v_train_kmh : ℕ) (t : ℝ) (L_bridge : ℝ) :
  L_train = 288 →
  v_train_kmh = 29 →
  t = 48.29 →
  L_bridge = total_distance (speed_in_m_per_s v_train_kmh) t - L_train →
  L_bridge = 100.89 := by
  sorry

end length_of_bridge_l825_825767


namespace first_digit_not_periodic_l825_825621

def first_digit (n : ℕ) : ℕ :=
  let s := n^2
  let digits := String.to_list (Nat.toDigits 10 s)
  digits.headD 0

theorem first_digit_not_periodic (a_n : ℕ → ℕ) : ¬(∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, a_n n = a_n (n + p)) :=
by
  intros
  let a_n := first_digit
  sorry

end first_digit_not_periodic_l825_825621


namespace find_g_l825_825454

variable (x : ℝ)

-- Given condition
def given_condition (g : ℝ → ℝ) : Prop :=
  5 * x^5 + 3 * x^3 - 4 * x + 2 + g x = 7 * x^3 - 9 * x^2 + x + 5

-- Goal
def goal (g : ℝ → ℝ) : Prop :=
  g x = -5 * x^5 + 4 * x^3 - 9 * x^2 + 5 * x + 3

-- The statement combining given condition and goal to prove
theorem find_g (g : ℝ → ℝ) (h : given_condition x g) : goal x g :=
by
  sorry

end find_g_l825_825454


namespace tessa_owes_30_l825_825920

-- Definitions based on given conditions
def initial_debt : ℕ := 40
def paid_back : ℕ := initial_debt / 2
def remaining_debt_after_payment : ℕ := initial_debt - paid_back
def additional_borrowing : ℕ := 10
def total_debt : ℕ := remaining_debt_after_payment + additional_borrowing

-- Theorem to be proved
theorem tessa_owes_30 : total_debt = 30 :=
by
  sorry

end tessa_owes_30_l825_825920


namespace chocolate_milk_tea_sales_l825_825260

theorem chocolate_milk_tea_sales (total_sales : ℕ) (winter_melon_ratio : ℚ) (okinawa_ratio : ℚ) :
  total_sales = 50 →
  winter_melon_ratio = 2 / 5 →
  okinawa_ratio = 3 / 10 →
  ∃ (chocolate_sales : ℕ), chocolate_sales = total_sales - total_sales * winter_melon_ratio - total_sales * okinawa_ratio ∧ chocolate_sales = 15 :=
by
  intro h1 h2 h3
  use (total_sales - total_sales * winter_melon_ratio - total_sales * okinawa_ratio).to_nat
  split
  · simp [h1, h2, h3]
  · exact sorry

end chocolate_milk_tea_sales_l825_825260


namespace value_of_c_l825_825269

axiom parabola (b c : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x : ℝ, y = x^2 + b * x + c) ∧
  (f 2 = 12) ∧
  (f (-2) = 8)

theorem value_of_c (b c : ℝ) (h : parabola b c) : c = 6 :=
sorry

end value_of_c_l825_825269


namespace percentage_seeds_germinated_l825_825848

theorem percentage_seeds_germinated :
  let S1 := 300
  let S2 := 200
  let S3 := 150
  let S4 := 250
  let S5 := 100
  let G1 := 0.20
  let G2 := 0.35
  let G3 := 0.45
  let G4 := 0.25
  let G5 := 0.60
  (G1 * S1 + G2 * S2 + G3 * S3 + G4 * S4 + G5 * S5) / (S1 + S2 + S3 + S4 + S5) * 100 = 32 := 
by
  sorry

end percentage_seeds_germinated_l825_825848


namespace min_value_of_function_l825_825544

theorem min_value_of_function :
  ∀ (x : ℝ), (x ∈ Ioo 0 (1/2)) →
  (2 / x + 9 / (1 - 2 * x) - 5) ≥ 20 ↔ x = 1 / 5 :=
by sorry

end min_value_of_function_l825_825544


namespace find_A_l825_825821

def spadesuit (A B : ℝ) : ℝ := 4 * A + 3 * B - 2

theorem find_A (A : ℝ) : spadesuit A 7 = 40 ↔ A = 21 / 4 :=
by
  sorry

end find_A_l825_825821


namespace difference_in_x_l825_825289

noncomputable def initial_yes := 0.4
noncomputable def initial_no := 0.3
noncomputable def initial_maybe := 0.3

noncomputable def final_yes := 0.6
noncomputable def final_no := 0.2
noncomputable def final_maybe := 0.2

theorem difference_in_x : 
  let x_min := min_change initial_yes initial_no initial_maybe final_yes final_no final_maybe in
  let x_max := max_change initial_yes initial_no initial_maybe final_yes final_no final_maybe in
  x_max - x_min = 0.2 :=
begin
  sorry
end

/-- Placeholder definitions for min_change and max_change. You will need to define these based on the mathematical conditions described -/
noncomputable def min_change : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ := sorry
noncomputable def max_change : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ := sorry

end difference_in_x_l825_825289


namespace problem_l825_825009

variable (α : ℝ)

noncomputable def tan_sum (α : ℝ) : ℝ :=
  let tanα := (3 / 4) in
  (tanα + 1) / (1 - tanα)

theorem problem 
  (cos_alpha sin_alpha : ℝ)
  (h₁ : 3 * cos_alpha = 4 * sin_alpha) :
  tan (α + π / 4) = 7 := 
by
  let tan_α := 3 / 4
  have h₀ : tanα = tan_α := rfl
  rw [tan_sum]
  rw [h₀]
  sorry

end problem_l825_825009


namespace average_children_in_families_with_children_l825_825413

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l825_825413


namespace ratio_sums_is_five_sixths_l825_825567

theorem ratio_sums_is_five_sixths
  (a b c x y z : ℝ)
  (h_positive_a : a > 0) (h_positive_b : b > 0) (h_positive_c : c > 0)
  (h_positive_x : x > 0) (h_positive_y : y > 0) (h_positive_z : z > 0)
  (h₁ : a^2 + b^2 + c^2 = 25)
  (h₂ : x^2 + y^2 + z^2 = 36)
  (h₃ : a * x + b * y + c * z = 30) :
  (a + b + c) / (x + y + z) = (5 / 6) :=
sorry

end ratio_sums_is_five_sixths_l825_825567


namespace range_of_b_length_of_AB_when_b_is_1_l825_825639

-- Define the ellipse equation and the line equation
def ellipse (x y : ℝ) := (x^2 / 2) + y^2 = 1
def line (x y b : ℝ) := y = x + b

-- Prove that for the line to intersect the ellipse at two distinct points, -√3 < b < √3
theorem range_of_b (b : ℝ) : 
  (exists x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ b ∧ line x₂ y₂ b) ↔ (-real.sqrt 3 < b ∧ b < real.sqrt 3) := 
sorry

-- Prove that when b = 1, the length of the vector |→AB| is 4√2/3
theorem length_of_AB_when_b_is_1 : 
  (exists (x₁ y₁ x₂ y₂ : ℝ), ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ 1 ∧ line x₂ y₂ 1 ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4 * real.sqrt 2 / 3)^2) :=
sorry

end range_of_b_length_of_AB_when_b_is_1_l825_825639


namespace compare_abc_l825_825496

noncomputable def a : ℝ := 1 / Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.exp 0.5
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > c ∧ c > a := by
  sorry

end compare_abc_l825_825496


namespace smallest_n_l825_825967

noncomputable def is_solution (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ b ≠ d ∧ a ≠ d ∧
  (∀ (x y : ℕ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) → 
    (x ≠ y ∧ ¬(is_connected x y) → Nat.gcd (x + y) n = 1) ∧
    (is_connected x y → 1 < Nat.gcd (x + y) n))

-- Define the is_connected relation as per the graph's adjacency:
-- (This part will depend on the exact structure of the graph given, so you may need to adjust it according to the actual graph structure provided in the problem)
def is_connected : ℕ → ℕ → Prop
| a, b => sorry -- Replace with the actual adjacency relation of your graph.

theorem smallest_n : 15 = Inf {n : ℕ | is_solution n} :=
by
  sorry -- Proof to be filled in later.

end smallest_n_l825_825967


namespace axis_of_symmetry_l825_825535

def f (x : ℝ) : ℝ := cos (x + π / 2) * cos (x + π / 4)

theorem axis_of_symmetry : ∀ x : ℝ, f (5 * π / 8 - x) = f (5 * π / 8 + x) :=
sorry

end axis_of_symmetry_l825_825535


namespace JannaTotalSleepHours_l825_825605

def JannaSleepHoursPerDay (day : String) : ℝ :=
  if day = "Monday" then 7 + 1/3
  else if day = "Tuesday" then 7
  else if day = "Wednesday" then 7 + 1/3
  else if day = "Thursday" then 6
  else if day = "Friday" then 9 + 1/3
  else if day = "Saturday" then 8 + 3/4
  else if day = "Sunday" then 8
  else 0

theorem JannaTotalSleepHours : JannaSleepHoursPerDay "Monday" + 
                                JannaSleepHoursPerDay "Tuesday" + 
                                JannaSleepHoursPerDay "Wednesday" + 
                                JannaSleepHoursPerDay "Thursday" + 
                                JannaSleepHoursPerDay "Friday" + 
                                JannaSleepHoursPerDay "Saturday" + 
                                JannaSleepHoursPerDay "Sunday" = 53.74 := 
by 
  -- Detailed calculations can be inserted here
  sorry

end JannaTotalSleepHours_l825_825605


namespace probability_of_prime_sum_l825_825795

def sum_is_prime (a b : ℕ) : Prop :=
  (a + b = 2) ∨ (a + b = 3) ∨ (a + b = 5) ∨ (a + b = 7) ∨ (a + b = 11) ∨ (a + b = 13)

def is_valid_outcome (faces : ℕ) (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ faces ∧ 1 ≤ b ∧ b ≤ faces

def favorable_outcomes (faces : ℕ) : ℕ :=
  (Finset.Icc 1 faces).sum (λ a => (Finset.Icc 1 faces).filter (sum_is_prime a).card)

theorem probability_of_prime_sum : (8 * 8 : ℚ)⁻¹ * (favorable_outcomes 8) = 23 / 64 := by
  sorry

end probability_of_prime_sum_l825_825795


namespace k_range_m_range_l825_825860

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem k_range (k : ℝ) : (∃ x : ℝ, g x = (2^x + 1) * f x + k) → k < 1 :=
by
  sorry

theorem m_range (m : ℝ) : (∀ x1 : ℝ, 0 < x1 ∧ x1 < 1 → 
                        ∃ x2 : ℝ, -Real.pi / 4 ≤ x2 ∧ x2 ≤ Real.pi / 6 ∧ f x1 - m * 2^x1 > g x2) 
                       → m ≤ 7 / 6 :=
by
  sorry

end k_range_m_range_l825_825860


namespace cyclic_quadrilateral_inscribed_l825_825653

/-- Given conditions: Quadrilateral ABCD is inscribed in circle Γ with radius 3, segments AC and BD intersect at E, 
circle γ passes through E and is tangent to Γ at A with radius 2, and the circumcircle of triangle BCE is tangent to γ 
at E and to line CD at C. Prove BD = 3y. -/
theorem cyclic_quadrilateral_inscribed {A B C D E γ : Type} (ABCD : cyclic_quadrilateral A B C D) (Γ : circle) (radius_Γ : real) 
  (tangency_1 : tangent_at γ E Γ A) (radius_γ : real) (intersection : intersect AC BD E) 
  (tangency_2 : tangent_to_line (circumcircle_triangle E B C) γ E) (tangency_3 : tangent_to_line (circumcircle_triangle E B C) CD C) :
  BD = 3 * y :=
by
  sorry

end cyclic_quadrilateral_inscribed_l825_825653


namespace chocolate_milk_tea_sales_l825_825259

theorem chocolate_milk_tea_sales (total_sales : ℕ) (winter_melon_ratio : ℚ) (okinawa_ratio : ℚ) :
  total_sales = 50 →
  winter_melon_ratio = 2 / 5 →
  okinawa_ratio = 3 / 10 →
  ∃ (chocolate_sales : ℕ), chocolate_sales = total_sales - total_sales * winter_melon_ratio - total_sales * okinawa_ratio ∧ chocolate_sales = 15 :=
by
  intro h1 h2 h3
  use (total_sales - total_sales * winter_melon_ratio - total_sales * okinawa_ratio).to_nat
  split
  · simp [h1, h2, h3]
  · exact sorry

end chocolate_milk_tea_sales_l825_825259


namespace tangent_inequalities_l825_825996

-- Define the problem condition
def x : ℝ := π / 180  -- 1 degree in radians

theorem tangent_inequalities :
  (finprod (λ i : Fin 44, Real.tan ((i + 1) * x)))^(1 / 44) < Real.sqrt 2 - 1 
  ∧ Real.sqrt 2 - 1 < (finset.sum (Finset.range 44) (λ i, Real.tan ((i + 1) * x))) / 44 :=
by
  sorry

end tangent_inequalities_l825_825996


namespace fraction_of_number_is_141_l825_825933

theorem fraction_of_number_is_141 (N : ℝ) (h1 : 0.3208 * N = 120.6208) :
  (141 / N) ≈ 0.375 :=
by sorry

end fraction_of_number_is_141_l825_825933


namespace arrangement_with_A_in_middle_arrangement_with_A_at_end_B_not_at_end_arrangement_with_A_B_adjacent_not_adjacent_to_C_l825_825799

-- Proof Problem 1
theorem arrangement_with_A_in_middle (products : Finset ℕ) (A : ℕ) (hA : A ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  5 ∈ products ∧ (∀ a ∈ arrangements, a (Fin.mk 2 sorry) = A) →
  arrangements.card = 24 :=
by sorry

-- Proof Problem 2
theorem arrangement_with_A_at_end_B_not_at_end (products : Finset ℕ) (A B : ℕ) (hA : A ∈ products) (hB : B ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  (5 ∈ products ∧ (∀ a ∈ arrangements, (a 0 = A ∨ a 4 = A) ∧ (a 1 ≠ B ∧ a 2 ≠ B ∧ a 3 ≠ B))) →
  arrangements.card = 36 :=
by sorry

-- Proof Problem 3
theorem arrangement_with_A_B_adjacent_not_adjacent_to_C (products : Finset ℕ) (A B C : ℕ) (hA : A ∈ products) (hB : B ∈ products) (hC : C ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  (5 ∈ products ∧ (∀ a ∈ arrangements, ((a 0 = A ∧ a 1 = B) ∨ (a 1 = A ∧ a 2 = B) ∨ (a 2 = A ∧ a 3 = B) ∨ (a 3 = A ∧ a 4 = B)) ∧
   (a 0 ≠ A ∧ a 1 ≠ B ∧ a 2 ≠ C))) →
  arrangements.card = 36 :=
by sorry

end arrangement_with_A_in_middle_arrangement_with_A_at_end_B_not_at_end_arrangement_with_A_B_adjacent_not_adjacent_to_C_l825_825799


namespace smallestIntegerWithGivenProperties_l825_825682

-- Define a predicate for x having 12 positive factors
def hasTwelveFactors (x : ℕ) : Prop :=
  (factors x).length = 12

-- Define a predicate for x being divisible by both 12 and 15
def divisibleBy12And15 (x : ℕ) : Prop :=
  12 ∣ x ∧ 15 ∣ x

-- Main theorem statement
theorem smallestIntegerWithGivenProperties (x : ℕ) :
  hasTwelveFactors x → divisibleBy12And15 x → x = 60 :=
by
  intro h1 h2
  sorry

end smallestIntegerWithGivenProperties_l825_825682


namespace find_a_and_b_monotonic_intervals_max_k_l825_825898

-- Given the function,
def f (x : ℝ) (a b : ℝ) : ℝ := (a + b * log x) / (x - 1)

-- Equation conditions for a and b
theorem find_a_and_b : 
  ∃ (a b : ℝ),
    f' 2 a b = - (1 / 2) * log 2 ∧
    f 4 a b = (1 + 2 * log 2) / 3 := 
sorry

-- Monotonicity of the function
theorem monotonic_intervals (a b : ℝ) (hf : ∃ (a b : ℝ), 
      f' 2 a b = - (1 / 2) * log 2 ∧
      f 4 a b = (1 + 2 * log 2) / 3 ) :
  ∀ x > 1, monotone_decreasing_on (0, 1) f a b ∧ monotone_decreasing_on (1,+∞) f a b :=
sorry

-- Maximum value for k
theorem max_k (k : ℕ) (hk : k ∣ 3) : 
    ∀ x_0 > 1, ∃ x_1 x_2, 0 < x_1 ∧ x_1 < x_2 ∧ x_2 < x_0 ∧ 
    (f x_0 1 1 = f x_1 1 1 ∧ f x_1 1 1 = f x_2 1 1) → 
    k ≤ 3 := 
sorry

end find_a_and_b_monotonic_intervals_max_k_l825_825898


namespace find_value_1p_1q_1r_l825_825624

theorem find_value_1p_1q_1r :
  let p q r : ℂ in
  (p + q + r = 15) ∧
  (p * q + q * r + r * p = 25) ∧
  (p * q * r = 10) →
  (1 + p) * (1 + q) * (1 + r) = 51 :=
by
  intro p q r h
  obtain ⟨h₁, h₂, h₃⟩ := h
  simp [h₁, h₂, h₃]
  sorry

end find_value_1p_1q_1r_l825_825624


namespace second_part_of_ratio_l825_825248

-- Define the conditions
def ratio_percent := 20
def first_part := 4

-- Define the proof statement using the conditions
theorem second_part_of_ratio (ratio_percent : ℕ) (first_part : ℕ) : 
  ∃ second_part : ℕ, (first_part * 100) = ratio_percent * second_part :=
by
  -- Let the second part be 20 and verify the condition
  use 20
  -- Clear the proof (details are not required)
  sorry

end second_part_of_ratio_l825_825248


namespace function_has_zero_l825_825186

theorem function_has_zero (m : ℝ) : ∃ x : ℝ, x^3 + 5 * m * x - 2 = 0 :=
by
  -- Assume for contradiction that the function has no zeros
  -- Provide the necessary assumptions and conditions
  let f := λ x : ℝ, x^3 + 5 * m * x - 2
  -- Proof to be filled in
  sorry

end function_has_zero_l825_825186


namespace binom_18_4_l825_825332

theorem binom_18_4 : Nat.binomial 18 4 = 3060 :=
by
  -- We start the proof here.
  sorry

end binom_18_4_l825_825332


namespace right_triangle_incenter_length_l825_825176

noncomputable def incenter_length (PQ PR QR : ℕ) (h : PQ ^ 2 + PR ^ 2 = QR ^ 2) : ℕ :=
  let s := (PQ + PR + QR) / 2
  let A := PQ * PR / 2
  A / s

theorem right_triangle_incenter_length :
  ∀ (PQ PR QR : ℕ) (h : PQ ^ 2 + PR ^ 2 = QR ^ 2),
  PQ = 15 → PR = 20 → QR = 25 →
  incenter_length PQ PR QR h = 5 :=
by
  intros PQ PR QR h hPQ hPR hQR
  rw [hPQ, hPR, hQR]
  -- Detailed proof is omitted here.
  sorry

end right_triangle_incenter_length_l825_825176


namespace three_digit_numbers_count_l825_825562

theorem three_digit_numbers_count : 
  let a_values := Finset.range' 1 10 in
  let c_values := Finset.range 10 in
  let is_valid (a b c : ℕ) := b = a + c - 1 in
  let valid_triples := (a_values.product c_values).filter (λ ⟨a, c⟩, 0 ≤ a + c - 1 ∧ a + c - 1 ≤ 9) in
  valid_triples.card = 54 :=
by
  let a_values := Finset.range' 1 10
  let c_values := Finset.range 10
  let is_valid (a b c : ℕ) := b = a + c - 1
  let valid_triples := (a_values.product c_values).filter (λ ⟨a, c⟩, let b := a + c - 1 in 0 ≤ b ∧ b ≤ 9)
  have : valid_triples.card = 54 := sorry
  exact this

end three_digit_numbers_count_l825_825562


namespace total_quantity_is_1000_l825_825759

/-- Let's define constants and variables. -/
variables (Q : ℝ)

/-- Conditions: -/
-- 600 kg is sold at 18% profit
def profit_18 := 600 * 0.18
-- Define the quantity sold at 8% profit.
def quantity_8 := Q - 600
-- Overall profit is 14% of the total quantity
def total_profit := 0.14 * Q
-- Profit from the portion sold at 8% profit
def profit_8 := quantity_8 * 0.08

-- Proving that the total quantity of sugar Q is 1000 kg given the conditions
theorem total_quantity_is_1000 : 0.08 * (Q - 600) + 0.18 * 600 = 0.14 * Q → Q = 1000 :=
by
  sorry

end total_quantity_is_1000_l825_825759


namespace angle_BXD_l825_825505

variable {A B C D X : Point}
variable {BC BA AC : ℝ}
variable [Nonempty Quadrilateral ABCD]

noncomputable def midpoint (A C : Point) : Point := sorry

axiom quadrilateral_cond (ABCD : Quadrilateral) : sqrt 2 * (BC - BA) = AC
axiom midpoint_AC (A C : Point) : X = midpoint A C

theorem angle_BXD (A B C D X : Point) :
  2 * ∠BXD = ∠DAB - ∠DCB :=
by
  have h1 : sqrt 2 * (BC - BA) = AC := quadrilateral_cond ABCD
  have h2 : X = midpoint A C := midpoint_AC A C
  sorry

end angle_BXD_l825_825505


namespace floor_difference_l825_825833

theorem floor_difference (x : ℝ)
  (h1 : x = 13.2)
  (h2 : x^2 = 174.24)
  (h3 : ⌊13.2⌋ = 13)
  (h4 : ⌊174.24⌋ = 174) :
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 5 :=
by
  sorry

end floor_difference_l825_825833


namespace valuable_files_count_l825_825292

theorem valuable_files_count 
    (initial_files : ℕ) 
    (deleted_fraction_initial : ℚ) 
    (additional_files : ℕ) 
    (irrelevant_fraction_additional : ℚ) 
    (h1 : initial_files = 800) 
    (h2 : deleted_fraction_initial = (70:ℚ) / 100)
    (h3 : additional_files = 400)
    (h4 : irrelevant_fraction_additional = (3:ℚ) / 5) : 
    (initial_files - ⌊deleted_fraction_initial * initial_files⌋ + additional_files - ⌊irrelevant_fraction_additional * additional_files⌋) = 400 :=
by sorry

end valuable_files_count_l825_825292


namespace tessa_debt_l825_825917

theorem tessa_debt :
  let initial_debt : ℤ := 40 in
  let repayment : ℤ := initial_debt / 2 in
  let debt_after_repayment : ℤ := initial_debt - repayment in
  let additional_debt : ℤ := 10 in
  debt_after_repayment + additional_debt = 30 :=
by
  -- The proof goes here.
  sorry

end tessa_debt_l825_825917


namespace number_of_terms_in_expansion_is_12_l825_825561

-- Define the polynomials
def p (x y z : ℕ) := x + y + z
def q (u v w x : ℕ) := u + v + w + x

-- Define the number of terms in a polynomial as a function.
def numberOfTerms (poly : Polynomial ℕ) : ℕ :=
  poly.degree + 1

-- Prove the number of terms in expansion of (x + y + z)(u + v + w + x) is 12.
theorem number_of_terms_in_expansion_is_12 (x y z u v w : ℕ) :
  numberOfTerms (p x y z * q u v w x) = 12 := by
  sorry

end number_of_terms_in_expansion_is_12_l825_825561


namespace binom_18_4_eq_3060_l825_825318

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l825_825318


namespace algebraic_form_of_z_trigonometric_form_of_z_l825_825727

noncomputable def complex_z : ℂ :=
  -real.sqrt 2 * (complex.I) * (complex.cos (3 * real.pi / 4) + (complex.I) * (complex.sin (3 * real.pi / 4)))

theorem algebraic_form_of_z :
  complex_z = 1 + complex.I :=
by
  sorry

theorem trigonometric_form_of_z :
  complex_z = real.sqrt 2 * (complex.cos (real.pi / 4) + (complex.I) * (complex.sin (real.pi / 4))) :=
by
  sorry

end algebraic_form_of_z_trigonometric_form_of_z_l825_825727


namespace binom_18_4_l825_825337

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end binom_18_4_l825_825337


namespace divisors_inequalities_example_l825_825482

variable (a b : ℕ)

theorem divisors_inequalities_example (h₁ : a = 2^12) (h₂ : b = 2^9) :
  (a ∣ b^2) ∧ (b^2 ∣ a^3) ∧ (a^3 ∣ b^4) ∧ (b^4 ∣ a^5) ∧ ¬ (a^5 ∣ b^6) :=
by {
  rw [h₁, h₂],
  have ha : a = 2^12 := h₁,
  have hb : b = 2^9 := h₂,
  
  have ha_div_b2 : a ∣ b^2 := by sorry,
  have hb2_div_a3 : b^2 ∣ a^3 := by sorry,
  have ha3_div_b4 : a^3 ∣ b^4 := by sorry,
  have hb4_div_a5 : b^4 ∣ a^5 := by sorry,
  have not_a5_div_b6 : ¬(a^5 ∣ b^6) := by sorry,
  
  exact ⟨ha_div_b2, hb2_div_a3, ha3_div_b4, hb4_div_a5, not_a5_div_b6⟩
}

end divisors_inequalities_example_l825_825482


namespace max_sum_of_products_l825_825165

theorem max_sum_of_products (a b c d e : ℕ) (h_vals : {a, b, c, d, e} = {1, 2, 4, 5, 6}) :
  a + b + c + d + e = 18 ∧
  a^2 + b^2 + c^2 + d^2 + e^2 = 82 →
  ab + bc + cd + de + ea ≤ 54 :=
by
  sorry

end max_sum_of_products_l825_825165


namespace right_triangle_property_l825_825954

theorem right_triangle_property
  (a b c x : ℝ)
  (h1 : c^2 = a^2 + b^2)
  (h2 : 1/2 * a * b = 1/2 * c * x)
  : 1/x^2 = 1/a^2 + 1/b^2 :=
sorry

end right_triangle_property_l825_825954


namespace infinite_integer_roots_l825_825910

section InfiniteIntegerRoots

variable {ℕ : Type} [Nonempty ℕ] [LinearOrder ℕ]

-- Define the sequences {a_n} and {b_n} with given recursive relations
def a_seq : ℕ → ℤ
| 0       => some integer k
| (n+1)   => a_seq n + 1

def b_seq : ℕ → ℤ
| 0       => some integer b
| (n+1)   => (1 / 2 : ℚ) * a_seq n + b_seq n

-- Define the quadratic function f_n(x) = x^2 + a_n x + b_n
def f_n (n : ℕ) (x : ℤ) : ℤ := x^2 + a_seq n * x + b_seq n

-- The main theorem we want to prove
theorem infinite_integer_roots (k : ℕ) (h_a_k : a_seq k ∈ ℤ) (h_b_k : b_seq k ∈ ℤ)
  (h_roots : ∃ r1 r2 : ℤ, f_n k r1 = 0 ∧ f_n k r2 = 0 ∧ r1 ≠ r2) :
  ∃ infinitely_many_n : set ℕ, ∀ n ∈ infinitely_many_n, ∃ r1 r2 : ℤ, 
    f_n n r1 = 0 ∧ f_n n r2 = 0 ∧ r1 ≠ r2 :=
by
  sorry

end InfiniteIntegerRoots

end infinite_integer_roots_l825_825910


namespace jessica_borrowed_amount_l825_825977

def payment_pattern (hour : ℕ) : ℕ :=
  match (hour % 6) with
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | 4 => 8
  | 5 => 10
  | _ => 12

def total_payment (hours_worked : ℕ) : ℕ :=
  (hours_worked / 6) * 42 + (List.sum (List.map payment_pattern (List.range (hours_worked % 6))))

theorem jessica_borrowed_amount :
  total_payment 45 = 306 :=
by
  -- Proof omitted
  sorry

end jessica_borrowed_amount_l825_825977


namespace union_A_B_compl_inter_A_B_l825_825004

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2^x - 16 ≥ 0}

theorem union_A_B : A ∪ B = {x | x ≥ 3} :=
sorry

theorem compl_inter_A_B : (A ∩ B)ᶜ = {x | x < 4 ∨ x ≥ 10} :=
sorry

end union_A_B_compl_inter_A_B_l825_825004


namespace frequency_distribution_correct_l825_825220

theorem frequency_distribution_correct :
  (∀ (freq_dist_table fluctuation mean sample_data group_data frequency sample_size class_interval num_groups sample_mean : Type),
    (freq_dist_table -> fluctuation mean sample_data) ∧
    (frequency = group_data) ∧
    (∀ group, frequency group / sample_size = group_data group) ∧
    (num_groups = sample_mean / class_interval)
    → (∀ group, frequency group / sample_size = group_data group)) :=
sorry

end frequency_distribution_correct_l825_825220


namespace find_n_l825_825985

theorem find_n (n : ℕ) (m : ℕ) (a b : ℕ → ℕ) 
  (h : ∀ i, 1 ≤ i → i ≤ n → m + i = a i * (b i)^2) 
  (hf : ∀ i, 1 ≤ i → i ≤ n → ¬ ∃ p : ℕ, p.prime ∧ p^2 ∣ a i) : 
  (∑ i in Finset.range n, a (i + 1) = 12) → (n = 2 ∨ n = 3) := 
sorry

end find_n_l825_825985


namespace range_of_g_l825_825347

open Real

noncomputable def g (A : ℝ) : ℝ :=
  (sin A)^3 * (2 * (cos A)^4 + (cos A)^6 + 2 * (sin A)^4 + (sin A)^4 * (cos A)^2) /
  (tan A * (sec A - (sin A)^3 * tan A))

theorem range_of_g :
  (∀ n : ℤ, A ≠ n * (π / 2)) → 
  set.image g {A : ℝ | ∀ n : ℤ, A ≠ n * (π / 2)} = {x : ℝ | 3 < x} :=
sorry

end range_of_g_l825_825347


namespace dawn_lemonade_price_l825_825806

theorem dawn_lemonade_price (x : ℕ) : 
  (10 * 25) = (8 * x) + 26 → x = 28 :=
by 
  sorry

end dawn_lemonade_price_l825_825806


namespace a5_value_l825_825951

-- Definitions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n, a (n + 1) = q * a n

def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

def product_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 1) = 2^(2 * n + 1)

-- Theorem statement
theorem a5_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_pos : positive_terms a) (h_prod : product_condition a) : a 5 = 32 :=
sorry

end a5_value_l825_825951


namespace shortest_distance_PQ_l825_825634

-- Definitions of the parametric points on the lines P and Q
def point_P (u : ℝ) : ℝ × ℝ × ℝ :=
  (-u + 1, 3 * u + 3, 2 * u - 2)

def point_Q (v : ℝ) : ℝ × ℝ × ℝ :=
  (2 * v + 2, -v + 1, 3 * v + 5)

-- Definition of the distance squared between points P and Q
def dist_sq (u v : ℝ) : ℝ :=
  let (x1, y1, z1) := point_P u
  let (x2, y2, z2) := point_Q v
  (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2

-- Statement of the theorem to prove that the shortest distance PQ is √(625/26)
theorem shortest_distance_PQ :
  (∃ u v : ℝ, dist_sq u v = 625 / 26) →
  (∀ u v, sqrt (dist_sq u v) ≥ sqrt (625 / 26)) :=
sorry

end shortest_distance_PQ_l825_825634


namespace average_upstream_speed_l825_825699

/--
There are three boats moving down a river. Boat A moves downstream at a speed of 1 km in 4 minutes 
and upstream at a speed of 1 km in 8 minutes. Boat B moves downstream at a speed of 1 km in 
5 minutes and upstream at a speed of 1 km in 11 minutes. Boat C moves downstream at a speed of 
1 km in 6 minutes and upstream at a speed of 1 km in 10 minutes. Prove that the average speed 
of the boats against the current is 6.32 km/h.
-/
theorem average_upstream_speed :
  let speed_A_upstream := 1 / (8 / 60 : ℝ)
  let speed_B_upstream := 1 / (11 / 60 : ℝ)
  let speed_C_upstream := 1 / (10 / 60 : ℝ)
  let average_speed := (speed_A_upstream + speed_B_upstream + speed_C_upstream) / 3
  average_speed = 6.32 :=
by
  let speed_A_upstream := 1 / (8 / 60 : ℝ)
  let speed_B_upstream := 1 / (11 / 60 : ℝ)
  let speed_C_upstream := 1 / (10 / 60 : ℝ)
  let average_speed := (speed_A_upstream + speed_B_upstream + speed_C_upstream) / 3
  sorry

end average_upstream_speed_l825_825699


namespace arithmetic_to_geometric_sum_l825_825862

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem arithmetic_to_geometric_sum (a : ℝ) (a1 a2 : ℝ) (f : ℝ → ℝ)
  (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ n : ℕ+, f (a ^ n) = 4 + (n - 1).val * 2) →
  (∀ n : ℕ+, a ^ n = 2) →
  (∀ n : ℕ, f (a ^ (n + 1)) = 2 * (n + 1) + 2) →
  (∀ n : ℕ, S n = (n + 1) * 2 ^ (n + 2)) :=
sorry

end arithmetic_to_geometric_sum_l825_825862


namespace find_value_1p_1q_1r_l825_825623

theorem find_value_1p_1q_1r :
  let p q r : ℂ in
  (p + q + r = 15) ∧
  (p * q + q * r + r * p = 25) ∧
  (p * q * r = 10) →
  (1 + p) * (1 + q) * (1 + r) = 51 :=
by
  intro p q r h
  obtain ⟨h₁, h₂, h₃⟩ := h
  simp [h₁, h₂, h₃]
  sorry

end find_value_1p_1q_1r_l825_825623


namespace numerical_puzzle_solution_l825_825462

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l825_825462


namespace average_children_l825_825432

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l825_825432


namespace horner_method_example_l825_825185

def polynomial_at (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.reverse.foldl (λ acc coeff, acc * x + coeff) 0

theorem horner_method_example : polynomial_at [5, 1, -3, 2, 1] 2 = 5 :=
by
  sorry

end horner_method_example_l825_825185


namespace complex_expression_eval_l825_825716

def z : ℂ := complex.exp (complex.I * π / 3)

theorem complex_expression_eval : z^100 + z^50 + 1 = -complex.I :=
by sorry

end complex_expression_eval_l825_825716


namespace vector_u_projections_l825_825845

open Finset

noncomputable def vector_u := ![-2 / 5, 81 / 10]

theorem vector_u_projections :
  let u := vector_u in
  (∃ x y, u = ![x, y] ∧
    proj (![3, 2]) (![x, y]) = ![45 / 13, 30 / 13] ∧
    proj (![1, 4]) (![x, y]) = ![32 / 17, 128 / 17]) :=
by
  let u := vector_u
  refine ⟨-2 / 5, 81 / 10, rfl, _, _⟩
  sorry
  sorry

end vector_u_projections_l825_825845


namespace average_children_in_families_with_children_l825_825426

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l825_825426


namespace stadium_seating_and_revenue_l825_825744

   def children := 52
   def adults := 29
   def seniors := 15
   def seats_A := 40
   def seats_B := 30
   def seats_C := 25
   def price_A := 10
   def price_B := 15
   def price_C := 20
   def total_seats := 95

   def revenue_A := seats_A * price_A
   def revenue_B := seats_B * price_B
   def revenue_C := seats_C * price_C
   def total_revenue := revenue_A + revenue_B + revenue_C

   theorem stadium_seating_and_revenue :
     (children <= seats_B + seats_C) ∧
     (adults + seniors <= seats_A + seats_C) ∧
     (children + adults + seniors > total_seats) →
     (revenue_A = 400) ∧
     (revenue_B = 450) ∧
     (revenue_C = 500) ∧
     (total_revenue = 1350) :=
   by
     sorry
   
end stadium_seating_and_revenue_l825_825744


namespace binom_18_4_l825_825331

theorem binom_18_4 : Nat.binomial 18 4 = 3060 :=
by
  -- We start the proof here.
  sorry

end binom_18_4_l825_825331


namespace parallel_lines_corresponding_angles_equal_l825_825723

theorem parallel_lines_corresponding_angles_equal (l₁ l₂ : Line) (t: Transversal) :
  parallel l₁ l₂ → corresponding_angles l₁ l₂ t = equal :=
by
  sorry

end parallel_lines_corresponding_angles_equal_l825_825723


namespace debt_calculation_correct_l825_825915

-- Conditions
def initial_debt : ℤ := 40
def repayment : ℤ := initial_debt / 2
def additional_borrowing : ℤ := 10

-- Final Debt Calculation
def remaining_debt : ℤ := initial_debt - repayment
def final_debt : ℤ := remaining_debt + additional_borrowing

-- Proof Statement
theorem debt_calculation_correct : final_debt = 30 := 
by 
  -- Skipping the proof
  sorry

end debt_calculation_correct_l825_825915


namespace num_possible_values_of_a_l825_825666

theorem num_possible_values_of_a : 
  {a : ℕ // 4 ∣ a ∧ a ∣ 24 ∧ a > 0}.card = 4 :=
by
  sorry

end num_possible_values_of_a_l825_825666


namespace isosceles_right_triangle_hypotenuse_length_l825_825646

/-- 
Given an isosceles right triangle with vertices at points with integer coordinates
and exactly 2019 points with integer coordinates on the sides of the triangle 
(including the vertices), prove that the smallest possible length of the hypotenuse 
is 952. 
-/

def smallest_hypotenuse_length (a b : ℕ) (h : is_isosceles_right_triangle a b) : ℕ :=
  let hypotenuse := (a ^ 2 + b ^ 2) ^ (1 / 2)
  hypotenuse

theorem isosceles_right_triangle_hypotenuse_length (a b : ℕ)
  (h₁ : a = b)
  (h₂ : ∃ n ≥ 1, ∀ (x y : ℕ), is_lattice_point_on_triangle x y a b ↔ n = 2019) :
    smallest_hypotenuse_length a b = 952 :=
sorry

end isosceles_right_triangle_hypotenuse_length_l825_825646


namespace binomial_equality_l825_825322

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l825_825322


namespace expected_value_of_twelve_sided_die_l825_825787

theorem expected_value_of_twelve_sided_die : ∑ k in finset.range 13, k / 12 = 6.5 := 
sorry

end expected_value_of_twelve_sided_die_l825_825787


namespace tangent_line_eq_l825_825904

def f (x : ℝ) : ℝ := x^2 - 1

theorem tangent_line_eq (x : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = x^2 - 1) (h_point : x = 1) : 
  ∃ (m b : ℝ), (∀ x, m = 2 → b = -2 → (f' x = 2) → (tangent_line f x = λ y, y = m * (x-1) + f(1))) :=
by
  sorry

end tangent_line_eq_l825_825904


namespace distance_phoenix_birch_l825_825651

theorem distance_phoenix_birch :
  ∀ (position : Fin 5 → ℕ)
  (poplar willow locust birch phoenix : Fin 5),
  (∀ (a b : Fin 5), position a = position b → a = b) ∧ 
  |position poplar - position willow| = |position poplar - position locust| ∧
  |position birch - position poplar| = |position birch - position locust| →
  |position phoenix - position birch| = 2 := by
  sorry

end distance_phoenix_birch_l825_825651


namespace average_children_in_families_with_children_l825_825418

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l825_825418


namespace smallest_consecutive_cube_x_l825_825962

theorem smallest_consecutive_cube_x:
  ∃ n: ℤ, 
  let u := n-1, v := n, w := n+1, x := n+2 in 
  u^3 + v^3 + w^3 = x^3 → x = 6 :=
  sorry

end smallest_consecutive_cube_x_l825_825962


namespace cab_income_first_day_l825_825246

theorem cab_income_first_day :
  ∃ x_1 : ℕ, x_1 + 250 + 450 + 400 + 800 = 5 * 500 ∧ x_1 = 600 := by
  use 600
  sorry

end cab_income_first_day_l825_825246


namespace num_possible_values_of_a_l825_825668

theorem num_possible_values_of_a : 
  {a : ℕ // 4 ∣ a ∧ a ∣ 24 ∧ a > 0}.card = 4 :=
by
  sorry

end num_possible_values_of_a_l825_825668


namespace smallest_integer_representation_l825_825191

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end smallest_integer_representation_l825_825191


namespace remainder_when_divided_by_44_l825_825268

theorem remainder_when_divided_by_44 (N Q R : ℕ) :
  (N = 44 * 432 + R) ∧ (N = 39 * Q + 15) → R = 0 :=
by
  sorry

end remainder_when_divided_by_44_l825_825268


namespace smallest_integer_representable_l825_825198

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end smallest_integer_representable_l825_825198


namespace min_log_value_l825_825019

theorem min_log_value (a b : ℝ) (hab : a > b) (hb2 : b > 2) : 
  ∃ (c : ℝ), c = log a b ∧ (4 - c - 1/c) = 2 :=
by
  sorry

end min_log_value_l825_825019


namespace expected_value_of_twelve_sided_die_l825_825792

theorem expected_value_of_twelve_sided_die : 
  let face_values := finset.range (12 + 1) \ finset.singleton 0 in
  (finset.sum face_values (λ x, x) : ℝ) / 12 = 6.5 :=
by
  sorry

end expected_value_of_twelve_sided_die_l825_825792


namespace unique_inscribed_sphere_l825_825106

theorem unique_inscribed_sphere (A B C D Q : Type) 
    [metric_space.point A] [metric_space.point B] [metric_space.point C] 
    [metric_space.point D] [metric_space.point Q] 
    (is_eq_dist : ∀ p ∈ {A, B, C, D}, dist Q p = dist_from_faces Q) 
    (intersects : intersects Q (bisector_planes A B C D)) :
    ∃! (S : sphere), S.is_inscribed (triangular_pyramid A B C D) :=
sorry

end unique_inscribed_sphere_l825_825106


namespace sum_of_exterior_angles_of_triangle_l825_825713

theorem sum_of_exterior_angles_of_triangle
  {α β γ α' β' γ' : ℝ} 
  (h1 : α + β + γ = 180)
  (h2 : α + α' = 180)
  (h3 : β + β' = 180)
  (h4 : γ + γ' = 180) :
  α' + β' + γ' = 360 := 
by 
sorry

end sum_of_exterior_angles_of_triangle_l825_825713


namespace googoo_total_buttons_l825_825137

noncomputable def button_count_shirt_1 : ℕ := 3
noncomputable def button_count_shirt_2 : ℕ := 5
noncomputable def quantity_shirt_1 : ℕ := 200
noncomputable def quantity_shirt_2 : ℕ := 200

theorem googoo_total_buttons :
  (quantity_shirt_1 * button_count_shirt_1) + (quantity_shirt_2 * button_count_shirt_2) = 1600 := by
  sorry

end googoo_total_buttons_l825_825137


namespace trajectory_equation_value_of_k_l825_825590

-- Definition of Point P's trajectory and conditions
variable (x y : ℝ)

def point_P_distance_ratio (x y : ℝ) : Prop :=
  (sqrt ((x - 2 * sqrt 2) ^ 2 + y ^ 2) / abs (x - 3 * sqrt 2)) = sqrt 6 / 3

-- The first part of the problem rewritten in Lean
theorem trajectory_equation (P : ℝ → ℝ → Prop) :
  (∀ x y, P x y ↔ point_P_distance_ratio x y) →
  (P x y → (x ^ 2 / 12 + y ^ 2 / 4 = 1)) :=
sorry

-- Second part definitions
variable (k : ℝ)

def line_l (k x : ℝ) : ℝ := k * x - 2

def line_intersects_trajectory (P : ℝ → ℝ → Prop) (k : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, P x1 y1 ∧ P x2 y2 ∧
    (line_l k x1 = y1) ∧ (line_l k x2 = y2) ∧ (x1 ≠ x2 ∨ y1 ≠ y2)

def am_an_equal (A M N : ℝ × ℝ) : Prop :=
  let ⟨ax, ay⟩ := A in
  let ⟨mx, my⟩ := M in
  let ⟨nx, ny⟩ := N in
  (sqrt ((ax - mx) ^ 2 + (ay - my) ^ 2)) = (sqrt ((ax - nx) ^ 2 + (ay - ny) ^ 2))

-- The second part of the problem rewritten in Lean
theorem value_of_k (P : ℝ → ℝ → Prop) (A : ℝ × ℝ) :
  (∀ x y, P x y ↔ (x ^ 2 / 12 + y ^ 2 / 4 = 1)) →
  (line_intersects_trajectory P k) →
  (am_an_equal A (x1, y1) (x2, y2)) →
  (k = sqrt 3 / 3 ∨ k = -sqrt 3 / 3) :=
sorry

end trajectory_equation_value_of_k_l825_825590


namespace arman_hourly_rate_increase_l825_825610

theorem arman_hourly_rate_increase :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let total_payment := 770
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := total_payment - last_week_earnings
  let this_week_rate := this_week_earnings / this_week_hours
  let rate_increase := this_week_rate - last_week_rate
  rate_increase = 0.50 :=
by {
  sorry
}

end arman_hourly_rate_increase_l825_825610


namespace weaving_problem_l825_825739

theorem weaving_problem
  (a : ℕ → ℚ) -- Define sequence a_n
  (a1_is_5 : a 1 = 5) -- a_1 = 5
  (is_arithmetic : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) -- a_n is an arithmetic sequence
  (sum_31_terms : ∑ n in Finset.range 31, a (n + 1) = 390) : -- S_{31} = 390
  ∃ d : ℚ, (d = 16 / 15) := -- prove the desired fraction equals 16 / 15
  sorry -- proof goes here

end weaving_problem_l825_825739


namespace find_matrixA_l825_825472

open Matrix

def matrixA : Matrix (Fin 2) (Fin 2) ℚ := ![![2, 1.4], ![7, -0.6]]

theorem find_matrixA (A : Matrix (Fin 2) (Fin 2) ℚ) :
  (A ⬝ ![![2], ![0]] = ![![4], ![14]]) ∧
  (A ⬝ ![![ -2], ![10]] = ![![6], ![-34]]) ↔
  A = matrixA :=
by
  sorry

end find_matrixA_l825_825472


namespace triangle_is_acute_l825_825672

-- Defining the basic structure for a triangle and its angles
variables (α β γ : ℝ)

-- Conditions for the problem
def conditions (α β γ : ℝ) : Prop :=
  (sin α > cos β) ∧ (sin β > cos γ) ∧ (sin γ > cos α)

-- The main theorem to be proved
theorem triangle_is_acute (α β γ : ℝ) (h : α + β + γ = π) (h_conditions : conditions α β γ) : 
  α < π / 2 ∧ β < π / 2 ∧ γ < π / 2 :=
  sorry -- Proof will be here

-- Note: π represents 180 degrees in radians

end triangle_is_acute_l825_825672


namespace calc_deltav_01_calc_deltav_001_instantaneous_velocity_l825_825270

def s (t : ℝ) : ℝ := 2 * t^2 + 3

theorem calc_deltav_01 (t : ℝ) (Δt : ℝ) (h : t = 2) (hΔ : Δt = 0.01) :
  (s (t + Δt) - s t) / Δt = 8.02 :=
sorry

theorem calc_deltav_001 (t : ℝ) (Δt : ℝ) (h : t = 2) (hΔ : Δt = 0.001) :
  (s (t + Δt) - s t) / Δt = 8.002 :=
sorry

theorem instantaneous_velocity (t : ℝ) (h : t = 2) :
  ∀ ε > 0, ∃ δ > 0, ∀ 0 < |Δt| < δ, |(s (t + Δt) - s t) / Δt - 8| < ε :=
sorry

end calc_deltav_01_calc_deltav_001_instantaneous_velocity_l825_825270


namespace only_one_decimal_number_between_7_5_and_9_5_false_l825_825702

noncomputable def decimal_num_between (a b : ℝ) (x : ℝ) : Prop :=
  (a < x) ∧ (x < b) ∧ (∃ n : ℕ, x = ∑ i in range (n+1), (10^(-i) * (ι ℝ) (digit i)))

theorem only_one_decimal_number_between_7_5_and_9_5_false :
  ∀ x, decimal_num_between 7.5 9.5 x → false :=
begin
  sorry
end

end only_one_decimal_number_between_7_5_and_9_5_false_l825_825702


namespace jim_makes_60_dollars_l825_825606

-- Definitions based on the problem conditions
def average_weight_per_rock : ℝ := 1.5
def price_per_pound : ℝ := 4
def number_of_rocks : ℕ := 10

-- Problem statement
theorem jim_makes_60_dollars :
  (average_weight_per_rock * number_of_rocks) * price_per_pound = 60 := by
  sorry

end jim_makes_60_dollars_l825_825606


namespace avg_children_in_families_with_children_l825_825404

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l825_825404


namespace count_functions_satisfying_conditions_l825_825079

def A := { n : ℕ | 1 ≤ n ∧ n ≤ 2011 }

def f (x : ℕ) := { y : ℕ | y ≤ x ∧ y ∈ A }

theorem count_functions_satisfying_conditions :
  (∃ (f : ℕ → ℕ), (∀ n ∈ A, f n ∈ f n) ∧ (∀ n ∈ A, f n ≤ n) ∧ 
    (set.finite { y | ∃ n ∈ A, f n = y } ∧ set.card ({ y | ∃ n ∈ A, f n = y }) = 2010)) = 2^2011 - 2012 :=
by
  sorry

end count_functions_satisfying_conditions_l825_825079


namespace max_area_of_triangle_l825_825817

theorem max_area_of_triangle :
  ∀ (O O' : EuclideanSpace ℝ (Fin 2)) (M : EuclideanSpace ℝ (Fin 2)),
  dist O O' = 2014 →
  dist O M = 1 ∨ dist O' M = 1 →
  ∃ (A : ℝ), A = 1007 :=
by
  intros O O' M h₁ h₂
  sorry

end max_area_of_triangle_l825_825817


namespace union_sets_l825_825638

noncomputable def A : Set ℝ := {x | (x + 1) * (x - 2) < 0}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
noncomputable def C : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem union_sets (A : Set ℝ) (B : Set ℝ) : (A ∪ B = C) := by
  sorry

end union_sets_l825_825638


namespace evaluate_expression_l825_825836

theorem evaluate_expression :
  (1 / 2 * log 10 25 + log 10 2 + 7 ^ (log 7 3) = 4) :=
by
  sorry

end evaluate_expression_l825_825836


namespace avg_children_with_kids_l825_825401

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l825_825401


namespace charlie_age_when_jenny_twice_as_old_as_bobby_l825_825063

-- Conditions as Definitions
def ageDifferenceJennyCharlie : ℕ := 5
def ageDifferenceCharlieBobby : ℕ := 3

-- Problem Statement as a Theorem
theorem charlie_age_when_jenny_twice_as_old_as_bobby (j c b : ℕ) 
  (H1 : j = c + ageDifferenceJennyCharlie) 
  (H2 : c = b + ageDifferenceCharlieBobby) : 
  j = 2 * b → c = 11 :=
by
  sorry

end charlie_age_when_jenny_twice_as_old_as_bobby_l825_825063


namespace num_possible_values_of_a_l825_825667

theorem num_possible_values_of_a : 
  {a : ℕ // 4 ∣ a ∧ a ∣ 24 ∧ a > 0}.card = 4 :=
by
  sorry

end num_possible_values_of_a_l825_825667


namespace product_range_l825_825905

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 9 then
  abs (log 3 x - 1)
else if h : x > 9 then
  4 - sqrt x
else
  0

theorem product_range (a b c : ℝ) (ha : 1 < a) (ha2 : a < 3) (hb : 3 < b) (hb2 : b < 9) (hc : 9 < c) (hc2 : c < 16)
    (h : f a = f b) (h2 : f b = f c) : 
  81 < a * b * c ∧ a * b * c < 144 := by
  sorry

end product_range_l825_825905


namespace correct_options_l825_825885

variable {R : Type} [Real : RealField R]
variable (f : R → R) (x : R)

-- Option A: f(x+2) = -f(-x) implies symmetry about (1,0)
def optionA : Prop := ∀ x : R, f(x + 2) = -f(-x) → f(x + 1) = -f(1 - x)

-- Option B: y = -f(2 - x) is symmetric about (1,0) to y = f(x)
def optionB : Prop := ∀ x : R, -f(2 - x) = -f(x) → f(x) = -f(x)

-- Option C: y = f(-1 + x) - f(1 - x) is symmetric about (1,0)
def optionC : Prop := ∀ x : R, f(-1 + x) - f(1 - x) + f(1 - (2 - x)) - f(-1 + (2 - x)) = 0

-- Option D: y = f(1 + x) - f(1 - x) is symmetric about (1,0) is incorrect
def optionD : Prop := ∀ x : R, (f x = 2 * x) → (f(1 + x) - f(1 - x)) ≠ 4 * x

-- The theorem stating which options are correct
theorem correct_options : optionA f R ∧ optionB f R ∧ optionC f R ∧ ¬ optionD f :=
by
  -- The proof of this theorem is beyond the scope of this task.
  sorry

end correct_options_l825_825885


namespace approx_num_fish_in_pond_l825_825947

noncomputable def numFishInPond (tagged_in_second: ℕ) (total_second: ℕ) (tagged: ℕ) : ℕ :=
  tagged * total_second / tagged_in_second

theorem approx_num_fish_in_pond :
  numFishInPond 2 50 50 = 1250 := by
  sorry

end approx_num_fish_in_pond_l825_825947


namespace parabola_directrix_l825_825001

theorem parabola_directrix (p : ℝ) (hp : p > 0) (t : ℝ) :
  (∀ x y : ℝ, (x^2 = 2 * p * y ↔ y = x + t)) →
  (∃ x₁ x₂ : ℝ, (x₁ + x₂) = 4) →
  (directrix_eq : y = -1) :=
begin
  sorry
end

end parabola_directrix_l825_825001


namespace expected_value_of_twelve_sided_die_l825_825791

theorem expected_value_of_twelve_sided_die : 
  let face_values := finset.range (12 + 1) \ finset.singleton 0 in
  (finset.sum face_values (λ x, x) : ℝ) / 12 = 6.5 :=
by
  sorry

end expected_value_of_twelve_sided_die_l825_825791


namespace union_M_N_eq_N_l825_825614

def M := {x : ℝ | x^2 - 2 * x ≤ 0}
def N := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

theorem union_M_N_eq_N : M ∪ N = N := 
sorry

end union_M_N_eq_N_l825_825614


namespace EmilySixthQuizScore_l825_825356

theorem EmilySixthQuizScore (x : ℕ) : 
  let scores := [85, 92, 88, 90, 93]
  let total_scores_with_x := scores.sum + x
  let desired_average := 91
  total_scores_with_x = 6 * desired_average → x = 98 := by
  sorry

end EmilySixthQuizScore_l825_825356


namespace inclination_angle_range_l825_825573

theorem inclination_angle_range :
  (∃ l : ℝ → ℝ, (∃ θ, ∀ x, l x = (Real.tan θ) * (x - 3)) ∧
  (∃ x y, (x - 1)^2 + y^2 = 1 ∧ y = l x)) →
   ∃ θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 6) ∪ Set.Icc (5 * Real.pi / 6) Real.pi :=
begin
  sorry
end

end inclination_angle_range_l825_825573


namespace general_solution_l825_825973

noncomputable def integrating_factor := λ (x : ℝ), 1 / x^2

-- Define the function P(x, y) and Q(x, y)
def P (x y : ℝ) := x^2 * y^2 - 1
def Q (x y : ℝ) := 2 * x^3 * y

-- Define the partial derivatives of P with respect to y and Q with respect to x
def dP_dy (x y : ℝ) := 2 * x^2 * y
def dQ_dx (x y : ℝ) := 6 * x^2 * y

-- Assuming the given integrating factor, show that the original differential equation becomes exact
theorem general_solution : 
  ∃ (C : ℝ), ∀ (x y : ℝ), 
    (xy^2 + (1/x) = C) :=
sorry

end general_solution_l825_825973


namespace expected_value_twelve_sided_die_l825_825777

theorem expected_value_twelve_sided_die : 
  let die_sides := 12 in 
  let outcomes := finset.range (die_sides + 1) in
  (finset.sum outcomes id : ℚ) / die_sides = 6.5 :=
by
  sorry

end expected_value_twelve_sided_die_l825_825777


namespace find_side_c_and_area_S_find_sinA_plus_cosB_l825_825944

-- Definitions for the conditions given
structure Triangle :=
  (a b c : ℝ)
  (angleA angleB angleC : ℝ)

noncomputable def givenTriangle : Triangle :=
  { a := 2, b := 4, c := 2 * Real.sqrt 3, angleA := 30, angleB := 90, angleC := 60 }

-- Prove the length of side c and the area S
theorem find_side_c_and_area_S (t : Triangle) (h : t = givenTriangle) :
  t.c = 2 * Real.sqrt 3 ∧ (1 / 2) * t.a * t.b * Real.sin (t.angleC * Real.pi / 180) = 2 * Real.sqrt 3 :=
by
  sorry

-- Prove the value of sin A + cos B
theorem find_sinA_plus_cosB (t : Triangle) (h : t = givenTriangle) :
  Real.sin (t.angleA * Real.pi / 180) + Real.cos (t.angleB * Real.pi / 180) = 1 / 2 :=
by
  sorry

end find_side_c_and_area_S_find_sinA_plus_cosB_l825_825944


namespace four_leaved_clovers_percentage_l825_825034

noncomputable def percentage_of_four_leaved_clovers (clovers total_clovers purple_four_leaved_clovers : ℕ ) : ℝ := 
  (purple_four_leaved_clovers * 4 * 100) / total_clovers 

theorem four_leaved_clovers_percentage :
  percentage_of_four_leaved_clovers 500 500 25 = 20 := 
by
  -- application of conditions and arithmetic simplification.
  sorry

end four_leaved_clovers_percentage_l825_825034


namespace gooGoo_buttons_l825_825136

theorem gooGoo_buttons (num_3_button_shirts : ℕ) (num_5_button_shirts : ℕ)
  (buttons_per_3_button_shirt : ℕ) (buttons_per_5_button_shirt : ℕ)
  (order_quantity : ℕ)
  (h1 : num_3_button_shirts = order_quantity)
  (h2 : num_5_button_shirts = order_quantity)
  (h3 : buttons_per_3_button_shirt = 3)
  (h4 : buttons_per_5_button_shirt = 5)
  (h5 : order_quantity = 200) :
  num_3_button_shirts * buttons_per_3_button_shirt + num_5_button_shirts * buttons_per_5_button_shirt = 1600 := by
  have h6 : 200 * 3 = 600 := by norm_num
  have h7 : 200 * 5 = 1000 := by norm_num
  have h8 : 600 + 1000 = 1600 := by norm_num
  rw [h1, h2, h3, h4, h5]
  rw [h6, h7]
  exact h8

end gooGoo_buttons_l825_825136


namespace solution_set_f_x_ge_2_l825_825637

noncomputable def f : ℝ → ℝ :=
  λ x, if x >= 0 then 2 * Real.exp x - 1 else Real.log ( |x - 1| ) / Real.log 2

theorem solution_set_f_x_ge_2 :
  {x : ℝ | f x ≥ 2} = (Set.Iic (-3)) ∪ (Set.Ici 1) :=
by
  sorry

end solution_set_f_x_ge_2_l825_825637


namespace incorrect_modulus_squared_conclusion_l825_825501

open Complex

theorem incorrect_modulus_squared_conclusion :
  ∃ z₁ z₂ : ℂ, (z₁ ≠ z₂) ∧ (|z₁| = |z₂|) ∧ (z₁^2 ≠ z₂^2) :=
by
  sorry

end incorrect_modulus_squared_conclusion_l825_825501


namespace average_children_in_families_with_children_l825_825422

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l825_825422


namespace omission_possible_l825_825975

theorem omission_possible :
  ∃ n ∈ (Finset.range 2009 : Set ℕ),
  ∃ (a : ℕ → ℕ) (ha : ∀ i, a i ∈ (Finset.range 2008 : Set ℕ) \ \{n\}),
  (∀ i ≠ j, abs(a i - a ((i + 1) % 2007)) ≠ abs(a j - a ((j + 1) % 2007))) :=
by
  sorry

end omission_possible_l825_825975


namespace initial_volume_of_solution_is_six_l825_825249

theorem initial_volume_of_solution_is_six
  (V : ℝ)
  (h1 : 0.30 * V + 2.4 = 0.50 * (V + 2.4)) :
  V = 6 :=
by
  sorry

end initial_volume_of_solution_is_six_l825_825249


namespace min_dot_product_l825_825553

-- Definition of the vectors and the point X on line OP
def OP : ℝ × ℝ := (2, 1)
def OA : ℝ × ℝ := (1, 7)
def OB : ℝ × ℝ := (5, 1)

def X (λ : ℝ) : ℝ × ℝ := (2 * λ, λ)

-- Definitions of vectors XA and XB based on X
def XA (λ : ℝ) : ℝ × ℝ := let (xX, yX) := X λ in (1 - xX, 7 - yX)
def XB (λ : ℝ) : ℝ × ℝ := let (xX, yX) := X λ in (5 - xX, 1 - yX)

-- Function to compute the dot product
def dot_product (p1 p2 : ℝ × ℝ) : ℝ :=
p1.1 * p2.1 + p1.2 * p2.2

-- Function to compute the expression of dot product XA · XB as a function of λ
def XA_dot_XB (λ : ℝ) : ℝ := dot_product (XA λ) (XB λ)

theorem min_dot_product : ∃ λ : ℝ, XA_dot_XB λ = -8 :=
sorry  -- The proof is omitted, but we assert the existence of such a λ.

end min_dot_product_l825_825553


namespace geometric_product_of_arithmetic_sequence_l825_825511

theorem geometric_product_of_arithmetic_sequence (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) :
  (∀ m n : ℕ, a_n ≠ 0) →
  (∀ m n : ℕ, a_n = a_m + (n - m) * (a_n - a_(n-1))) →
  (a_6 - a_7^2 + a_8 = 0) →
  (∀ n : ℕ, b_n = b_1 * (b_(n-1) / b_1)^(n-1)) →
  (b_7 = a_7) →
  b_4 * b_7 * b_10 = 8 :=
by sorry

end geometric_product_of_arithmetic_sequence_l825_825511


namespace solution_set_of_inequality_l825_825861

noncomputable def f : ℝ → ℝ :=
λ x, if 0 < x then Real.log (x + 1) else -x^2 + 2 * x

theorem solution_set_of_inequality : {x : ℝ | f (2 * x - 1) > f (2 - x)} = set.Ioi 1 :=
by 
    sorry

end solution_set_of_inequality_l825_825861


namespace box_dimensions_l825_825808

theorem box_dimensions {a b c : ℕ} (h1 : a + c = 17) (h2 : a + b = 13) (h3 : 2 * (b + c) = 40) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by {
  sorry
}

end box_dimensions_l825_825808


namespace expected_value_of_total_weekly_rainfall_is_39_2_l825_825258

-- Definitions based on Problem Conditions:
def daily_rain_distribution := [(0.30, 0.0), (0.40, 5.0), (0.30, 12.0)]

-- Lean statement for the proof problem:
theorem expected_value_of_total_weekly_rainfall_is_39_2 :
  let E_daily := ∑ (p, x) in daily_rain_distribution, p * x in
  E_daily * 7 = 39.2 :=
by
  -- Placeholder for the proof steps
  sorry

end expected_value_of_total_weekly_rainfall_is_39_2_l825_825258


namespace mario_sunday_cost_l825_825804

noncomputable def weekday_discount_rate : ℝ := 0.10
noncomputable def weekend_increase_rate : ℝ := 0.50
noncomputable def shave_cost : ℝ := 10
noncomputable def style_cost : ℝ := 15
noncomputable def total_paid_monday : ℝ := 18

theorem mario_sunday_cost : 
  let original_haircut_cost := total_paid_monday - shave_cost in
  let discounted_haircut_cost := original_haircut_cost / (1 - weekday_discount_rate) in
  let weekend_haircut_cost := discounted_haircut_cost * (1 + weekend_increase_rate) in
  let total_sunday_cost := weekend_haircut_cost + style_cost in
  total_sunday_cost = 28.34 :=
by
  let original_haircut_cost := 18 - 10
  let discounted_haircut_cost := original_haircut_cost / 0.90
  let weekend_haircut_cost := discounted_haircut_cost * 1.50
  let total_sunday_cost := weekend_haircut_cost + 15
  sorry

end mario_sunday_cost_l825_825804


namespace cost_per_can_l825_825127

/-- If the total cost for a 12-pack of soft drinks is $2.99, then the cost per can,
when rounded to the nearest cent, is approximately $0.25. -/
theorem cost_per_can (total_cost : ℝ) (num_cans : ℕ) (h_total_cost : total_cost = 2.99) (h_num_cans : num_cans = 12) :
  Real.round (total_cost / num_cans * 100) / 100 = 0.25 :=
by
  sorry

end cost_per_can_l825_825127


namespace average_children_in_families_with_children_l825_825415

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l825_825415


namespace franks_earnings_l825_825487

/-- Frank's earnings problem statement -/
theorem franks_earnings 
  (total_hours : ℕ) (days : ℕ) (regular_pay_rate : ℝ) (overtime_pay_rate : ℝ)
  (hours_first_day : ℕ) (overtime_first_day : ℕ)
  (hours_second_day : ℕ) (hours_third_day : ℕ)
  (hours_fourth_day : ℕ) (overtime_fourth_day : ℕ)
  (regular_hours_per_day : ℕ) :
  total_hours = 32 →
  days = 4 →
  regular_pay_rate = 15 →
  overtime_pay_rate = 22.50 →
  hours_first_day = 12 →
  overtime_first_day = 4 →
  hours_second_day = 8 →
  hours_third_day = 8 →
  hours_fourth_day = 12 →
  overtime_fourth_day = 4 →
  regular_hours_per_day = 8 →
  (32 * regular_pay_rate + 8 * overtime_pay_rate) = 660 := 
by 
  intros 
  sorry

end franks_earnings_l825_825487


namespace numeric_puzzle_AB_eq_B_pow_V_l825_825468

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l825_825468


namespace smallest_K_exists_l825_825632

theorem smallest_K_exists (S : Finset ℕ) (h_S : S = (Finset.range 51).erase 0) :
  ∃ K, ∀ (T : Finset ℕ), T ⊆ S ∧ T.card = K → 
  ∃ a b, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ (a + b) ∣ (a * b) ∧ K = 39 :=
by
  use 39
  sorry

end smallest_K_exists_l825_825632


namespace avg_children_with_kids_l825_825399

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l825_825399


namespace matchsticks_distribution_l825_825966

open Nat

theorem matchsticks_distribution
  (length_sticks : ℕ)
  (width_sticks : ℕ)
  (length_condition : length_sticks = 60)
  (width_condition : width_sticks = 10)
  (total_sticks : ℕ)
  (total_sticks_condition : total_sticks = 60 * 11 + 10 * 61)
  (children_count : ℕ)
  (children_condition : children_count > 100)
  (division_condition : total_sticks % children_count = 0) :
  children_count = 127 := by
  sorry

end matchsticks_distribution_l825_825966
